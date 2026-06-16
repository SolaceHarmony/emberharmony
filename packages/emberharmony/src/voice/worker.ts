import { existsSync, writeFileSync, unlinkSync, readFileSync, mkdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import type { Config } from "../config/config"
import { Log } from "../util/log"
import { Voice } from "./token"

const PID_DIR = path.join(
  process.env["XDG_RUNTIME_DIR"] ?? path.join(process.env["HOME"] ?? "/tmp", ".local", "share", "emberharmony"),
  "voice",
)
const PID_FILE = path.join(PID_DIR, "worker.pid")
const LOCK_PORT = 47819

/**
 * Manages the voice agent worker as a child process of `emberharmony serve`.
 * The worker gets the resolved voice settings (config + credential store)
 * injected as environment variables at spawn, so the UI is the single
 * configuration path.
 *
 * Two launch modes:
 *  - **Bundled runtime** (packaged desktop app): the LiveKit agents framework
 *    forks node_modules scripts and dynamically imports the agent file, so it
 *    cannot run inside the compiled single-file CLI. The desktop app ships a
 *    self-contained runtime (bun + agent.js + node_modules + models) and points
 *    the sidecar at it via EMBERHARMONY_VOICE_RUNTIME_DIR; we spawn that.
 *  - **Source** (dev / `bun run`): spawn `./agent.ts` with the current Bun.
 *
 * Process management:
 *  - A PID file at $XDG_RUNTIME_DIR/emberharmony/voice/worker.pid tracks the
 *    running worker, enabling stale-process detection across restarts.
 *  - On start, any stale worker (same PID file, dead process) is killed.
 *  - On stop, SIGTERM is sent first, then SIGKILL after 3s if the process
 *    hasn't exited.
 */
export namespace VoiceWorker {
  const log = Log.create({ service: "voice.worker" })

  let proc: ReturnType<typeof Bun.spawn> | undefined
  let lastServerUrl: string | undefined
  let lastSettings: Voice.Settings | undefined

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

  function readPidFile(): number | undefined {
    try {
      const content = readFileSync(PID_FILE, "utf-8").trim()
      const pid = Number(content)
      return Number.isFinite(pid) ? pid : undefined
    } catch {
      return undefined
    }
  }

  function writePidFile(pid: number) {
    mkdirSync(PID_DIR, { recursive: true })
    writeFileSync(PID_FILE, String(pid), "utf-8")
  }

  function removePidFile() {
    try {
      unlinkSync(PID_FILE)
    } catch {}
  }

  function isProcessAlive(pid: number): boolean {
    try {
      process.kill(pid, 0)
      return true
    } catch {
      return false
    }
  }

  async function killProcess(pid: number): Promise<boolean> {
    if (!isProcessAlive(pid)) return true
    try {
      process.kill(pid, "SIGTERM")
    } catch {
      return false
    }
    // Poll every 100ms for up to 3 seconds
    for (let i = 0; i < 30; i++) {
      await Bun.sleep(100)
      if (!isProcessAlive(pid)) return true
    }
    // Force kill
    try {
      process.kill(pid, "SIGKILL")
    } catch {}
    return !isProcessAlive(pid)
  }

  async function killStaleWorker() {
    const stalePid = readPidFile()
    if (stalePid === undefined) return
    if (isProcessAlive(stalePid)) {
      log.info("killing stale voice worker", { pid: stalePid })
      await killProcess(stalePid)
    }
    removePidFile()
  }

  export async function start(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    stop()
    await killStaleWorker()
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
        EMBERHARMONY_VOICE_WORKER_PORT: String(LOCK_PORT),
      },
      stdout: "inherit",
      stderr: "inherit",
      onExit: (_proc, exitCode) => {
        if (proc && exitCode !== null && exitCode !== 0) {
          log.error("voice agent worker exited", { exitCode })
        }
        removePidFile()
      },
    })
    writePidFile(proc.pid)
    log.info("voice agent worker started", { pid: proc.pid, mode: launch.mode, stt: settings.stt, tts: settings.tts })
    lastSettings = settings
    return true
  }

  export function running(): boolean {
    return proc !== undefined && proc.exitCode === null
  }

  export async function stop() {
    if (!proc) return
    const p = proc
    proc = undefined
    lastSettings = undefined
    try {
      process.kill(p.pid, "SIGTERM")
    } catch {}
    // Wait up to 3 seconds for graceful exit
    const exited = await new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), 3000)
      p.exited
        .then(() => {
          clearTimeout(timeout)
          resolve(true)
        })
        .catch(() => {
          clearTimeout(timeout)
          resolve(false)
        })
    })
    if (!exited) {
      p.kill()
    }
    removePidFile()
  }

  export async function restart(override?: Config.Voice): Promise<boolean> {
    if (!lastServerUrl) return false
    const next = await Voice.settings(override)
    if (lastSettings && settingsEqual(lastSettings, next) && running()) {
      log.info("voice settings unchanged; skipping worker restart")
      return running()
    }
    stop()
    return start(lastServerUrl, override)
  }

  function settingsEqual(a: Voice.Settings, b: Voice.Settings): boolean {
    return (
      a.url === b.url &&
      a.apiKey === b.apiKey &&
      a.apiSecret === b.apiSecret &&
      a.stt === b.stt &&
      a.tts === b.tts &&
      a.intent === b.intent &&
      a.disabled === b.disabled
    )
  }
}
