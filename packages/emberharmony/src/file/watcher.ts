import { BusEvent } from "@/bus/bus-event"
import { Bus } from "@/bus"
import z from "zod"
import { Instance } from "../project/instance"
import { Log } from "../util/log"
import { FileIgnore } from "./ignore"
import { Config } from "../config/config"
import path from "path"
// @ts-ignore
import { createWrapper } from "@parcel/watcher/wrapper"
import { lazy } from "@/util/lazy"
import type ParcelWatcher from "@parcel/watcher"
import { $ } from "bun"
import { Flag } from "@/flag/flag"
import { readdir } from "fs/promises"

declare const EMBERHARMONY_LIBC: string | undefined

export namespace FileWatcher {
  const log = Log.create({ service: "file.watcher" })

  export const Event = {
    Updated: BusEvent.define(
      "file.watcher.updated",
      z.object({
        file: z.string(),
        event: z.union([z.literal("add"), z.literal("change"), z.literal("unlink")]),
      }),
    ),
  }

  const watcher = lazy((): typeof import("@parcel/watcher") | undefined => {
    try {
      const binding = require(
        `@parcel/watcher-${process.platform}-${process.arch}${process.platform === "linux" ? `-${EMBERHARMONY_LIBC || "glibc"}` : ""}`,
      )
      return createWrapper(binding) as typeof import("@parcel/watcher")
    } catch (error) {
      log.error("failed to load watcher binding", { error })
      return
    }
  })

  interface WatcherState {
    subs: ParcelWatcher.AsyncSubscription[]
    disposed: boolean
  }

  // The file watcher is a CONVENIENCE: it surfaces edits made OUTSIDE the app so
  // things like the VCS branch view refresh. In-app edits already publish
  // Event.Updated directly (tool/edit, tool/write, …), so the OS watch is never a
  // dependency. It must therefore never block boot or hold a request, and never
  // pretend to control the filesystem: on a slow/remote/unsupported mount
  // (Parallels share, NFS/NAS, sshfs) the OS subscribe can take many seconds or
  // hang forever. So `state()` returns SYNCHRONOUSLY and all setup — including the
  // subscribe — runs detached in the background; the subscription is fire-and-
  // forget (never awaited, no timeout). If it resolves we attach; if it hangs we
  // simply run without external file events; if it rejects we log and move on.
  const state = Instance.state<WatcherState>(
    () => {
      const handle: WatcherState = { subs: [], disposed: false }
      if (Instance.project.vcs !== "git") return handle
      void setupWatches(handle)
      return handle
    },
    async (handle) => {
      handle.disposed = true
      await Promise.all(handle.subs.splice(0).map((sub) => sub?.unsubscribe().catch(() => {})))
    },
  )

  function errMessage(err: unknown): string {
    return err instanceof Error ? err.message : String(err)
  }

  async function setupWatches(handle: WatcherState): Promise<void> {
    try {
      log.info("init")
      const backend = (() => {
        if (process.platform === "win32") return "windows" as const
        if (process.platform === "darwin") return "fs-events" as const
        if (process.platform === "linux") return "inotify" as const
        return undefined
      })()
      if (!backend) {
        log.warn("watcher backend not supported; continuing without file watching", { platform: process.platform })
        return
      }
      const w = watcher()
      if (!w) return
      const cfg = await Config.get().catch(() => undefined)
      const cfgIgnores = cfg?.watcher?.ignore ?? []

      const onEvents: ParcelWatcher.SubscribeCallback = (err, evts) => {
        if (err) return
        for (const evt of evts) {
          if (evt.type === "create") Bus.publish(Event.Updated, { file: evt.path, event: "add" })
          if (evt.type === "update") Bus.publish(Event.Updated, { file: evt.path, event: "change" })
          if (evt.type === "delete") Bus.publish(Event.Updated, { file: evt.path, event: "unlink" })
        }
      }

      // fire-and-forget: NEVER await the subscribe. Attach if/when it resolves.
      const attach = (dir: string, ignore: string[]) => {
        if (handle.disposed) return
        w.subscribe(dir, onEvents, { ignore, backend })
          .then((sub) => {
            if (handle.disposed) {
              void sub.unsubscribe().catch(() => {})
              return
            }
            handle.subs.push(sub)
            log.info("watching", { dir })
          })
          .catch((err) => log.warn("file watch unavailable; continuing without it", { dir, error: errMessage(err) }))
      }

      if (Flag.EMBERHARMONY_EXPERIMENTAL_FILEWATCHER) {
        attach(Instance.directory, [...FileIgnore.PATTERNS, ...cfgIgnores])
      }

      const vcsDir = await $`git rev-parse --git-dir`
        .quiet()
        .nothrow()
        .cwd(Instance.worktree)
        .text()
        .then((x) => path.resolve(Instance.worktree, x.trim()))
        .catch(() => undefined)
      if (vcsDir && !cfgIgnores.includes(".git") && !cfgIgnores.includes(vcsDir)) {
        const gitDirContents = await readdir(vcsDir).catch(() => [])
        attach(
          vcsDir,
          gitDirContents.filter((entry) => entry !== "HEAD"),
        )
      }
    } catch (err) {
      log.warn("file watcher setup failed; continuing without it", { error: errMessage(err) })
    }
  }

  export function init() {
    if (Flag.EMBERHARMONY_EXPERIMENTAL_DISABLE_FILEWATCHER) {
      return
    }
    state()
  }
}
