import { existsSync, unlinkSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { createConnection } from "node:net"
import os from "node:os"
import type { Config } from "../config/config"
import { Global } from "../global"
import { Log } from "../util/log"
import { Voice } from "./token"

const GRACEFUL_TIMEOUT_MS = 10_000

/**
 * Manages the voice agent worker as a child process of `emberharmony serve`.
 *
 * Lifecycle uses IPC (Unix socket) for shutdown: stop() writes "shutdown" to
 * the socket, and the agent process calls AgentServer.drain() + close() to
 * gracefully stop all forked job processes. SIGTERM is a fallback only.
 *
 * Lifecycle uses graceful shutdown: stop() sends SIGTERM so the LiveKit
 * agents framework can drain active jobs and close its WebSocket, then waits
 * up to GRACEFUL_TIMEOUT_MS before escalating to SIGKILL on the entire
 * process group.
 *
 * Two launch modes:
 *  - **Bundled runtime** (packaged desktop app): the LiveKit agents framework
 *    forks node_modules scripts and dynamically imports the agent file, so it
 *    cannot run inside the compiled single-file CLI. The desktop app ships a
 *    self-contained runtime (bun + agent.js + node_modules + models) and points
 *    the sidecar at it via EMBERHARMONY_VOICE_RUNTIME_DIR; we spawn that.
 *  - **Source** (dev / `bun run`): spawn `./agent.ts` with the current Bun.
 */
export namespace VoiceWorker {
  const log = Log.create({ service: "voice.worker" })

  let proc: ReturnType<typeof Bun.spawn> | undefined
  let ipcSocketPath: string | undefined
  let lastServerUrl: string | undefined

  // Serialize every lifecycle op (start/stop/restart) onto a single chain.
  // Without this, start() is not atomic: the `running()` guard runs, then two
  // awaits (stop + Voice.settings) yield before `proc` is assigned, so two
  // concurrent starts — e.g. serve boot racing a config-PATCH restart — both
  // pass the guard and both Bun.spawn, leaking an untracked agent.js worker
  // (and its forked LiveKit job processes). Running ops exclusively makes the
  // `running()` idempotency check meaningful and guarantees one worker.
  let lifecycle: Promise<unknown> = Promise.resolve()
  function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const next = lifecycle.then(fn, fn)
    // keep the queue moving even if this op throws; callers still see the error
    lifecycle = next.then(
      () => {},
      () => {},
    )
    return next
  }

  interface Launch {
    mode: "bundled" | "source"
    cmd: string[]
    cwd?: string
    env: Record<string, string>
  }

  function resolveLaunch(): Launch | undefined {
    const runtimeDir = process.env["EMBERHARMONY_VOICE_RUNTIME_DIR"]
    if (runtimeDir) {
      const bunBin = path.join(runtimeDir, process.platform === "win32" ? "bun.exe" : "bun")
      const agentJs = path.join(runtimeDir, "agent.js")
      if (existsSync(bunBin) && existsSync(agentJs)) {
        return {
          mode: "bundled",
          cmd: [bunBin, agentJs, "start"],
          cwd: runtimeDir,
          env: {
            // hf_utils caches under os.homedir()/.cache/huggingface/hub and
            // ignores HF_HOME; os.homedir() honors $HOME, so HOME=models/ is what
            // makes the worker resolve the turn-detector ONNX bundled at build.
            HOME: path.join(runtimeDir, "models"),
            HF_HOME: path.join(runtimeDir, "models"),
            XDG_CACHE_HOME: path.join(runtimeDir, "models"),
          },
        }
      }
      log.warn("EMBERHARMONY_VOICE_RUNTIME_DIR set but bundle incomplete", { runtimeDir })
    }
    const agentPath = fileURLToPath(new URL("./agent.ts", import.meta.url))
    if (existsSync(agentPath)) {
      return { mode: "source", cmd: [process.execPath, "run", agentPath, "start"], env: {} }
    }
    return undefined
  }

  function allocateSocket(): string {
    const id = `emberharmony-voice-${process.pid}-${Date.now()}`
    // node:net IPC on Windows requires a named-pipe path (\\.\pipe\...), not a
    // filesystem path; agent.ts passes this straight to ipc.listen(), so a
    // plain temp path would make the worker fail to register on Windows.
    if (process.platform === "win32") return `\\\\.\\pipe\\${id}`
    return path.join(os.tmpdir(), `${id}.sock`)
  }

  async function sendShutdown(): Promise<boolean> {
    if (!ipcSocketPath) return false
    const socketPath = ipcSocketPath
    // The agent's IPC server may not be listening yet on a fast stop-after-start
    // (the socket file appears only once agent.js runs ipc.listen). Retry briefly,
    // and log the real connection error so a genuinely-unreachable socket is
    // diagnosable instead of silently degrading to a SIGTERM fallback every time.
    // Keep the window short: this runs inside the lifecycle mutex, so a long hold
    // blocks any queued stop/restart.
    const deadline = Date.now() + 1200
    let lastErr: string | undefined
    while (Date.now() < deadline) {
      const ok = await new Promise<boolean>((resolve) => {
        const conn = createConnection(socketPath, () => {
          conn.write("shutdown")
          conn.end()
          resolve(true)
        })
        conn.on("error", (err) => {
          lastErr = err instanceof Error ? err.message : String(err)
          resolve(false)
        })
        setTimeout(() => {
          conn.destroy()
          resolve(false)
        }, 500)
      })
      if (ok) return true
      if (proc?.exitCode != null) return false // worker already exited; nothing to reach
      await new Promise((r) => setTimeout(r, 100))
    }
    log.warn("voice IPC shutdown could not reach worker socket", { socket: socketPath, error: lastErr })
    return false
  }

  // --- orphan tracking -----------------------------------------------------
  // The worker is spawned `detached`, so an ungraceful death of `serve` (SIGKILL,
  // crash, VM reset) leaves the worker (and its forked LiveKit job processes)
  // running and reparented to init. Each serve records its worker in a pidfile
  // KEYED BY THE OWNING SERVE PID (so concurrent serves never clobber or reap
  // each other's records), and a worker counts as a reapable orphan ONLY when
  // its owning serve is gone. Verifying the worker's argv still looks like ours
  // additionally guards against pid reuse.

  interface PidRecord {
    pid?: number
    socket?: string
    ppid?: number // the serve process that owns this worker
  }

  function ownPidfile(): string {
    return path.join(Global.Path.state, `voice-worker.${process.pid}.json`)
  }

  function writePidfile(pid: number, socket: string): void {
    try {
      mkdirSync(Global.Path.state, { recursive: true })
      writeFileSync(ownPidfile(), JSON.stringify({ pid, socket, ppid: process.pid } satisfies PidRecord))
    } catch {}
  }

  function clearPidfile(): void {
    try {
      unlinkSync(ownPidfile())
    } catch {}
  }

  function unlinkSocketFile(socket?: string): void {
    if (!socket || process.platform === "win32") return
    try {
      unlinkSync(socket)
    } catch {}
  }

  function isAlive(pid: number): boolean {
    try {
      process.kill(pid, 0)
      return true
    } catch {
      return false
    }
  }

  function looksLikeVoiceWorker(pid: number): boolean {
    try {
      if (process.platform === "linux") {
        const cmd = readFileSync(`/proc/${pid}/cmdline`, "utf8")
        return cmd.includes("agent.js") || cmd.includes("agent.ts")
      }
      const out = Bun.spawnSync(["ps", "-p", String(pid), "-o", "command="]).stdout.toString()
      return out.includes("agent.js") || out.includes("agent.ts")
    } catch {
      return false
    }
  }

  function reapStaleWorkers(): void {
    let files: string[]
    try {
      files = readdirSync(Global.Path.state).filter((f) => /^voice-worker\.\d+\.json$/.test(f))
    } catch {
      return
    }
    for (const file of files) {
      const full = path.join(Global.Path.state, file)
      let record: PidRecord
      try {
        record = JSON.parse(readFileSync(full, "utf8"))
      } catch {
        try {
          unlinkSync(full)
        } catch {}
        continue
      }
      const owner = record.ppid
      if (owner === process.pid) continue // our own current record — doStop manages it
      // a worker is only an orphan once its owning serve is gone; a live owner
      // means another serve is using it — never reap a live peer's worker
      if (typeof owner === "number" && isAlive(owner)) continue
      const pid = record.pid
      if (typeof owner === "number" && typeof pid === "number" && pid !== process.pid && isAlive(pid)) {
        if (looksLikeVoiceWorker(pid)) {
          log.warn("reaping orphaned voice worker from a dead serve", { pid, owner })
          try {
            process.kill(-pid, "SIGKILL")
          } catch {
            try {
              process.kill(pid, "SIGKILL")
            } catch {}
          }
        } else {
          log.info("recorded voice-worker pid alive but unverified; not killing", { pid })
        }
      }
      unlinkSocketFile(record.socket)
      try {
        unlinkSync(full)
      } catch {}
    }
  }

  export function start(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    return runExclusive(() => doStart(serverUrl, override))
  }

  async function doStart(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    if (running()) {
      log.info("voice agent worker already running; skipping duplicate start")
      return true
    }
    await doStop()
    // we hold no tracked worker — reap any worker whose owning serve has died
    // (a previous crashed serve of ours, or another instance's orphan)
    reapStaleWorkers()
    lastServerUrl = serverUrl
    const settings = await Voice.settings(override)
    if (!settings.available) {
      log.info("voice not configured; agent worker not started")
      return false
    }
    const launch = resolveLaunch()
    if (!launch) {
      log.warn("voice agent runtime not available in this build; start the worker manually with `bun run voice-agent`")
      return false
    }
    ipcSocketPath = allocateSocket()
    proc = Bun.spawn({
      cmd: launch.cmd,
      cwd: launch.cwd,
      // Run the worker as its own process-group leader. The LiveKit framework
      // forks job processes as children of the worker; a group-leader parent
      // lets the force-kill paths below (and killSync) signal the whole tree
      // via process.kill(-pid), instead of orphaning those job processes.
      detached: true,
      env: {
        ...process.env,
        ...launch.env,
        EMBERHARMONY_LIVEKIT_URL: settings.url!,
        EMBERHARMONY_LIVEKIT_API_KEY: settings.apiKey!,
        EMBERHARMONY_LIVEKIT_API_SECRET: settings.apiSecret!,
        LIVEKIT_URL: settings.url!,
        LIVEKIT_API_KEY: settings.apiKey!,
        LIVEKIT_API_SECRET: settings.apiSecret!,
        EMBERHARMONY_VOICE_STT_MODEL: settings.stt,
        EMBERHARMONY_VOICE_TTS_MODEL: settings.tts,
        EMBERHARMONY_VOICE_INTENT_MODEL: settings.intent,
        EMBERHARMONY_VOICE_SERVER_URL: serverUrl,
        EMBERHARMONY_VOICE_WORKER_PORT: "0",
        EMBERHARMONY_VOICE_IPC_SOCKET: ipcSocketPath,
      },
      stdout: "inherit",
      stderr: "inherit",
      onExit: (_proc, exitCode) => {
        if (proc && exitCode !== null && exitCode !== 0) {
          log.error("voice agent worker exited", { exitCode })
        }
      },
    })
    writePidfile(proc.pid, ipcSocketPath)
    log.info("voice agent worker started", {
      pid: proc.pid,
      mode: launch.mode,
      stt: settings.stt,
      tts: settings.tts,
      ipc: ipcSocketPath,
    })
    return true
  }

  export function running(): boolean {
    return proc !== undefined && proc.exitCode === null
  }

  /**
   * Synchronous best-effort kill for the process `exit` handler, where async
   * teardown (IPC drain) can't run. detached spawn makes the worker its own
   * group leader, so the negative-pid signal tears down the worker and its
   * forked job processes together rather than orphaning them.
   */
  export function killSync(): void {
    const p = proc
    if (!p || p.exitCode !== null) return
    proc = undefined
    const socket = ipcSocketPath
    ipcSocketPath = undefined
    try {
      process.kill(-p.pid, "SIGKILL")
    } catch {
      try {
        p.kill("SIGKILL")
      } catch {}
    }
    unlinkSocketFile(socket)
    clearPidfile()
  }

  export function stop(): Promise<void> {
    return runExclusive(() => doStop())
  }

  async function doStop(): Promise<void> {
    const p = proc
    if (!p) return
    proc = undefined
    const socket = ipcSocketPath // keep ipcSocketPath set so sendShutdown can reach it
    if (p.exitCode !== null) {
      ipcSocketPath = undefined
      unlinkSocketFile(socket)
      clearPidfile()
      return
    }

    const ppid = p.pid
    try {
      log.info("gracefully stopping voice agent worker via IPC", { pid: ppid })

      const sent = await sendShutdown()
      if (sent) {
        const exited = new Promise<boolean>((resolve) => {
          const timer = setTimeout(() => resolve(false), GRACEFUL_TIMEOUT_MS)
          p.exited
            .then(() => {
              clearTimeout(timer)
              resolve(true)
            })
            .catch(() => {
              clearTimeout(timer)
              resolve(true)
            })
        })
        const graceful = await exited
        if (graceful) {
          log.info("voice agent worker stopped gracefully via IPC", { pid: ppid })
          return
        }
      }

      log.warn("IPC shutdown failed or timed out; falling back to SIGTERM", { pid: ppid })
      p.kill("SIGTERM")

      const exited = new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => resolve(false), GRACEFUL_TIMEOUT_MS)
        p.exited
          .then(() => {
            clearTimeout(timer)
            resolve(true)
          })
          .catch(() => {
            clearTimeout(timer)
            resolve(true)
          })
      })

      const graceful = await exited
      if (graceful) {
        log.info("voice agent worker stopped via SIGTERM", { pid: ppid })
        return
      }

      log.warn("voice agent worker did not exit; force killing", { pid: ppid })
      if (process.platform === "win32") {
        Bun.spawn(["taskkill", "/pid", String(ppid), "/f", "/t"], { stdout: "ignore", stderr: "ignore" })
      } else {
        try {
          process.kill(-ppid, "SIGKILL")
        } catch {
          p.kill("SIGKILL")
        }
      }
    } finally {
      // tracked worker is gone (or being force-killed): drop the socket + pidfile
      // so a later boot doesn't try to reap a pid we already handled
      ipcSocketPath = undefined
      unlinkSocketFile(socket)
      clearPidfile()
    }
  }

  export function restart(override?: Config.Voice): Promise<boolean> {
    return runExclusive(async () => {
      if (!lastServerUrl) return false
      await doStop()
      return doStart(lastServerUrl, override)
    })
  }
}
