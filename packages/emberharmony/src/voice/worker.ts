import { existsSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import type { Config } from "../config/config"
import { Log } from "../util/log"
import { Voice } from "./token"

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

  /**
   * Resolve how to launch the worker. Prefer the bundled runtime the desktop
   * app ships (EMBERHARMONY_VOICE_RUNTIME_DIR); otherwise fall back to running
   * the TypeScript source with the current Bun (dev). Returns undefined when
   * neither is available (e.g. compiled CLI with no bundled runtime).
   */
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
          // Point the HF/agents model caches at the bundled models so the
          // worker loads VAD + turn-detector offline.
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

  export async function start(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    stop()
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
        // LiveKit Inference (STT/TTS) reads the standard env var names
        LIVEKIT_URL: settings.url!,
        LIVEKIT_API_KEY: settings.apiKey!,
        LIVEKIT_API_SECRET: settings.apiSecret!,
        EMBERHARMONY_VOICE_STT_MODEL: settings.stt,
        EMBERHARMONY_VOICE_TTS_MODEL: settings.tts,
        EMBERHARMONY_VOICE_INTENT_MODEL: settings.intent,
        EMBERHARMONY_VOICE_SERVER_URL: serverUrl,
        EMBERHARMONY_VOICE_WORKER_PORT: "0",
      },
      stdout: "inherit",
      stderr: "inherit",
      onExit: (_proc, exitCode) => {
        if (proc && exitCode !== null && exitCode !== 0) {
          log.error("voice agent worker exited", { exitCode })
        }
      },
    })
    log.info("voice agent worker started", { pid: proc.pid, mode: launch.mode, stt: settings.stt, tts: settings.tts })
    lastSettings = settings
    return true
  }

  export function running(): boolean {
    return proc !== undefined && proc.exitCode === null
  }

  export function stop() {
    if (!proc) return
    const p = proc
    proc = undefined
    lastSettings = undefined
    p.kill()
  }

  /**
   * Respawn with freshly resolved settings (after a config change). Also
   * handles the boot-unconfigured case: serve always records its URL via
   * start(), so configuring voice later starts the worker without a restart
   * of serve. No-op when not running under serve at all. Pass the just-merged
   * voice config — instance caches dispose asynchronously after a config
   * write, so resolving through Config.get() here would race a stale cache.
   */
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
