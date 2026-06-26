import { Log } from "../util/log"
import path from "path"
import fs from "fs/promises"
import { Global } from "../global"
import { Filesystem } from "../util/filesystem"
import { lazy } from "../util/lazy"
import { Lock } from "../util/lock"
import { $ } from "bun"
import { NamedError } from "@thesolaceproject/emberharmony-util/error"
import z from "zod"

export namespace Storage {
  const log = Log.create({ service: "storage" })

  type Migration = (dir: string) => Promise<void>

  export const NotFoundError = NamedError.create(
    "NotFoundError",
    z.object({
      message: z.string(),
    }),
  )

  const MIGRATIONS: Migration[] = [
    async (dir) => {
      const project = path.resolve(dir, "../project")
      if (!(await Filesystem.isDir(project))) return
      for await (const projectDir of new Bun.Glob("*").scan({
        cwd: project,
        onlyFiles: false,
      })) {
        log.info(`migrating project ${projectDir}`)
        let projectID = projectDir
        const fullProjectDir = path.join(project, projectDir)
        let worktree = "/"

        if (projectID !== "global") {
          for await (const msgFile of new Bun.Glob("storage/session/message/*/*.json").scan({
            cwd: path.join(project, projectDir),
            absolute: true,
          })) {
            const json = await Bun.file(msgFile).json()
            worktree = json.path?.root
            if (worktree) break
          }
          if (!worktree) continue
          if (!(await Filesystem.isDir(worktree))) continue
          const [id] = await $`git rev-list --max-parents=0 --all`
            .quiet()
            .nothrow()
            .cwd(worktree)
            .text()
            .then((x) =>
              x
                .split("\n")
                .filter(Boolean)
                .map((x) => x.trim())
                .toSorted(),
            )
          if (!id) continue
          projectID = id

          await Bun.write(
            path.join(dir, "project", projectID + ".json"),
            JSON.stringify({
              id,
              vcs: "git",
              worktree,
              time: {
                created: Date.now(),
                initialized: Date.now(),
              },
            }),
          )

          log.info(`migrating sessions for project ${projectID}`)
          for await (const sessionFile of new Bun.Glob("storage/session/info/*.json").scan({
            cwd: fullProjectDir,
            absolute: true,
          })) {
            const dest = path.join(dir, "session", projectID, path.basename(sessionFile))
            log.info("copying", {
              sessionFile,
              dest,
            })
            const session = await Bun.file(sessionFile).json()
            await Bun.write(dest, JSON.stringify(session))
            log.info(`migrating messages for session ${session.id}`)
            for await (const msgFile of new Bun.Glob(`storage/session/message/${session.id}/*.json`).scan({
              cwd: fullProjectDir,
              absolute: true,
            })) {
              const dest = path.join(dir, "message", session.id, path.basename(msgFile))
              log.info("copying", {
                msgFile,
                dest,
              })
              const message = await Bun.file(msgFile).json()
              await Bun.write(dest, JSON.stringify(message))

              log.info(`migrating parts for message ${message.id}`)
              for await (const partFile of new Bun.Glob(`storage/session/part/${session.id}/${message.id}/*.json`).scan(
                {
                  cwd: fullProjectDir,
                  absolute: true,
                },
              )) {
                const dest = path.join(dir, "part", message.id, path.basename(partFile))
                const part = await Bun.file(partFile).json()
                log.info("copying", {
                  partFile,
                  dest,
                })
                await Bun.write(dest, JSON.stringify(part))
              }
            }
          }
        }
      }
    },
    async (dir) => {
      for await (const item of new Bun.Glob("session/*/*.json").scan({
        cwd: dir,
        absolute: true,
      })) {
        const session = await Bun.file(item).json()
        if (!session.projectID) continue
        if (!session.summary?.diffs) continue
        const { diffs } = session.summary
        await Bun.file(path.join(dir, "session_diff", session.id + ".json")).write(JSON.stringify(diffs))
        await Bun.file(path.join(dir, "session", session.projectID, session.id + ".json")).write(
          JSON.stringify({
            ...session,
            summary: {
              additions: diffs.reduce((sum: any, x: any) => sum + x.additions, 0),
              deletions: diffs.reduce((sum: any, x: any) => sum + x.deletions, 0),
            },
          }),
        )
      }
    },
  ]

  const state = lazy(async () => {
    const dir = path.join(Global.Path.data, "storage")
    const migration = await Bun.file(path.join(dir, "migration"))
      .json()
      .then((x) => parseInt(x))
      .catch(() => 0)
    for (let index = migration; index < MIGRATIONS.length; index++) {
      log.info("running migration", { index })
      const migration = MIGRATIONS[index]
      await migration(dir).catch(() => log.error("failed to run migration", { index }))
      await Bun.write(path.join(dir, "migration"), (index + 1).toString())
    }
    return {
      dir,
    }
  })

  // Write-behind overlay. `pending[target]` holds the latest content intended
  // for a key that may not yet be durably on disk; it is cleared once THAT exact
  // content has been flushed. read()/update() consult it, so a hot-path caller
  // (Session.create) can return before its disk write completes — which matters
  // when the underlying filesystem stalls (a host virtual-disk I/O write barrier
  // froze POST /session for ~100s in production). The dir is derivable
  // synchronously (state() only gates first-run migrations), so pending is set
  // synchronously for immediate read-your-writes consistency.
  const SLOW_FLUSH_MS = 5_000
  const pending = new Map<string, unknown>()
  let flushSeq = 0

  function targetPath(key: string[]): string {
    return path.join(Global.Path.data, "storage", ...key) + ".json"
  }

  async function atomicWrite(target: string, content: unknown) {
    // temp + rename so a reader never observes a half-written file, and a stalled
    // write can't corrupt the existing file
    const tmp = `${target}.tmp-${process.pid}-${flushSeq++}`
    await Bun.write(tmp, JSON.stringify(content, null, 2))
    await fs.rename(tmp, target)
  }

  async function flush(target: string, content: unknown, attempt = 0) {
    if (pending.get(target) !== content) return // superseded before we even started
    const started = Date.now()
    try {
      using _ = await Lock.write(target)
      if (pending.get(target) !== content) return // a durable write/newer writeBehind won
      await atomicWrite(target, content)
    } catch (e) {
      // transient disk faults (EIO/ENOSPC/barrier) — retry a few times with
      // backoff before giving up, so a brief fault doesn't silently lose the
      // write. Bail if a newer write superseded us meanwhile.
      if (attempt < 3 && pending.get(target) === content) {
        log.warn("write-behind flush failed; retrying", {
          target,
          attempt,
          error: e instanceof Error ? e.message : String(e),
        })
        await new Promise((r) => setTimeout(r, 200 * (attempt + 1)))
        return flush(target, content, attempt + 1)
      }
      // give up: keep the entry in `pending` so reads stay consistent in-process,
      // but surface the durability loss loudly
      log.error("write-behind flush failed permanently; content kept in memory only", {
        target,
        error: e instanceof Error ? e.message : String(e),
      })
      return
    } finally {
      const took = Date.now() - started
      if (took > SLOW_FLUSH_MS) log.warn("write-behind flush slow (host I/O stall?)", { target, ms: took })
    }
    if (pending.get(target) === content) pending.delete(target)
  }

  /**
   * Persist without blocking the caller on disk. Returns synchronously after
   * recording the content in the overlay; the durable write happens in the
   * background. Use for hot-path writes (session create) where the response must
   * not hang on a slow filesystem. Trade-off: a crash inside the flush window
   * loses this write. Read-your-writes is preserved via read()/update().
   */
  export function writeBehind<T>(key: string[], content: T): void {
    const target = targetPath(key)
    pending.set(target, content)
    void state().then(() => flush(target, content))
  }

  export async function remove(key: string[]) {
    const dir = await state().then((x) => x.dir)
    const target = path.join(dir, ...key) + ".json"
    return withErrorHandling(async () => {
      // serialize with flush()/write()/update() so an in-flight write-behind
      // flush can't rename its temp file back into place AFTER we unlink
      // (resurrecting a deleted record)
      using _ = await Lock.write(target)
      pending.delete(target)
      await fs.unlink(target).catch(() => {})
    })
  }

  export async function read<T>(key: string[]) {
    const dir = await state().then((x) => x.dir)
    const target = path.join(dir, ...key) + ".json"
    if (pending.has(target)) return structuredClone(pending.get(target)) as T // read-your-writes
    return withErrorHandling(async () => {
      using _ = await Lock.read(target)
      const result = await Bun.file(target).json()
      return result as T
    })
  }

  export async function update<T>(key: string[], fn: (draft: T) => void) {
    const dir = await state().then((x) => x.dir)
    const target = path.join(dir, ...key) + ".json"
    return withErrorHandling(async () => {
      using _ = await Lock.write(target)
      const content = (pending.has(target) ? structuredClone(pending.get(target)) : await Bun.file(target).json()) as T
      fn(content)
      pending.set(target, content) // keep reads consistent across the durable write
      await atomicWrite(target, content)
      if (pending.get(target) === content) pending.delete(target)
      return content
    })
  }

  export async function write<T>(key: string[], content: T) {
    const dir = await state().then((x) => x.dir)
    const target = path.join(dir, ...key) + ".json"
    return withErrorHandling(async () => {
      using _ = await Lock.write(target)
      pending.set(target, content) // supersede any in-flight writeBehind + keep reads consistent
      await atomicWrite(target, content)
      if (pending.get(target) === content) pending.delete(target)
    })
  }

  async function withErrorHandling<T>(body: () => Promise<T>) {
    return body().catch((e) => {
      if (!(e instanceof Error)) throw e
      const errnoException = e as NodeJS.ErrnoException
      if (errnoException.code === "ENOENT") {
        throw new NotFoundError({ message: `Resource not found: ${errnoException.path}` })
      }
      throw e
    })
  }

  const glob = new Bun.Glob("**/*")
  export async function list(prefix: string[]) {
    const dir = await state().then((x) => x.dir)
    const result: string[][] = []
    const seen = new Set<string>()
    const add = (key: string[]) => {
      const id = key.join(" ")
      if (seen.has(id)) return
      seen.add(id)
      result.push(key)
    }
    try {
      const onDisk = await Array.fromAsync(
        glob.scan({
          cwd: path.join(dir, ...prefix),
          onlyFiles: true,
        }),
      )
      for (const x of onDisk) {
        // ignore atomic-write temp files (target.json.tmp-<pid>-<seq>) and any
        // other non-.json stragglers so they never surface as bogus keys
        if (!x.endsWith(".json")) continue
        add([...prefix, ...x.slice(0, -5).split(path.sep)])
      }
    } catch {
      // directory may not exist yet — pending-only entries can still match below
    }
    // include write-behind entries under this prefix that haven't flushed to disk
    // yet, so list-based reads (Session.list/children) are read-after-create
    // consistent with read()/update()
    const prefixDir = path.join(dir, ...prefix) + path.sep
    for (const target of pending.keys()) {
      if (!target.endsWith(".json") || !target.startsWith(prefixDir)) continue
      add(target.slice(dir.length + 1, -5).split(path.sep))
    }
    result.sort()
    return result
  }
}
