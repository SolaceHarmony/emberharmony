import { existsSync } from "node:fs"
import { fileURLToPath } from "node:url"
import type { Config } from "../config/config"
import { Log } from "../util/log"
import { Voice } from "./token"

/**
 * Manages the voice agent worker as a child process of `emberharmony serve`.
 * The worker gets the resolved voice settings (config + credential store)
 * injected as environment variables at spawn, so the UI is the single
 * configuration path. Requires running from source — the compiled CLI cannot
 * spawn the agent yet (see LIVEKIT_JOURNAL.md); use `bun run voice-agent`
 * manually in that case.
 */
export namespace VoiceWorker {
  const log = Log.create({ service: "voice.worker" })

  let proc: ReturnType<typeof Bun.spawn> | undefined
  let lastServerUrl: string | undefined

  export async function start(serverUrl: string, override?: Config.Voice): Promise<boolean> {
    lastServerUrl = serverUrl
    const settings = await Voice.settings(override)
    if (!settings.available) {
      log.info("voice not configured; agent worker not started")
      return false
    }
    const agentPath = fileURLToPath(new URL("./agent.ts", import.meta.url))
    if (!existsSync(agentPath)) {
      log.warn("voice agent source not available in this build; start the worker manually with `bun run voice-agent`")
      return false
    }
    proc = Bun.spawn({
      cmd: [process.execPath, "run", agentPath, "start"],
      env: {
        ...process.env,
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
    log.info("voice agent worker started", { pid: proc.pid, stt: settings.stt, tts: settings.tts })
    return true
  }

  export function running(): boolean {
    return proc !== undefined && proc.exitCode === null
  }

  export function stop() {
    if (!proc) return
    const p = proc
    proc = undefined
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
    stop()
    return start(lastServerUrl, override)
  }
}
