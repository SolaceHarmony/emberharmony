import { existsSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { createConnection } from "node:net"
import os from "node:os"
import type { Config } from "../config/config"
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
    const tmpDir = os.tmpdir()
    return path.join(tmpDir, `emberharmony-voice-${Date.now()}.sock`)
  }

  async function sendShutdown(): Promise<boolean> {
    if (!ipcSocketPath) return false
    const socketPath = ipcSocketPath
    return new Promise((resolve) => {
      const conn = createConnection(socketPath, () => {
        conn.write("shutdown")
        conn.end()
        resolve(true)
      })
      conn.on("error", () => resolve(false))
      setTimeout(() => {
        conn.destroy()
        resolve(false)
      }, 2000)
    })
  }

  export async function start(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    if (running()) {
      log.info("voice agent worker already running; skipping duplicate start")
      return true
    }
    await stop()
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

  export async function stop(): Promise<void> {
    const p = proc
    if (!p) return
    proc = undefined
    if (p.exitCode !== null) return

    const ppid = p.pid
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
  }

  export async function restart(override?: Config.Voice): Promise<boolean> {
    if (!lastServerUrl) return false
    await stop()
    return start(lastServerUrl, override)
  }
}
