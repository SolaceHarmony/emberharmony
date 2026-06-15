import { fileURLToPath } from "node:url"
import { type JobContext, type JobProcess, ServerOptions, cli, defineAgent, inference, voice } from "@livekit/agents"
import * as livekit from "@livekit/agents-plugin-livekit"
import * as silero from "@livekit/agents-plugin-silero"
import { llm } from "@livekit/agents"
import { Flag } from "../flag/flag"
import { SessionLLM } from "./bridge"
import { VoiceRegistry } from "./registry"
import { VOICE_AGENT_NAME } from "./constants"
import { VoiceWorkflow, VOICE_SYSTEM_PROMPT } from "./workflow"
import { BRAIN_SYSTEM_PROMPT, VOICE_PROJECT_DIR } from "./brain"

// Model strings accept an optional ":language" (STT) or ":voice" (TTS) suffix.
// The serve command injects these from the resolved voice config when it
// spawns the worker; standalone workers fall back to the registry defaults.
const STT_MODEL = process.env["EMBERHARMONY_VOICE_STT_MODEL"] ?? VoiceRegistry.DEFAULT_STT
const TTS_MODEL = process.env["EMBERHARMONY_VOICE_TTS_MODEL"] ?? VoiceRegistry.DEFAULT_TTS
// Small fast gateway model that routes plan/build per spoken turn
const INTENT_MODEL = process.env["EMBERHARMONY_VOICE_INTENT_MODEL"] ?? VoiceRegistry.DEFAULT_INTENT

class EmberHarmonyAgent extends voice.Agent {
  #workflow: VoiceWorkflow

  constructor(workflow: VoiceWorkflow) {
    super({
      // The session bridge holds the real context server-side; these
      // instructions only exist because voice.Agent requires them.
      instructions: "You are EmberHarmony, a voice interface to a coding assistant session.",
    })
    this.#workflow = workflow
  }

  override async onUserTurnCompleted(_chatCtx: llm.ChatContext, newMessage: llm.ChatMessage): Promise<void> {
    await this.#workflow.route(newMessage.textContent ?? "")
  }
}

/**
 * Discover the brain session at startup. Calls GET /voice/brain on the
 * EmberHarmony server to find or create the permanent voice brain session.
 * Returns { sessionID, directory, system } for use by SessionLLM.
 */
async function discoverBrainSession(serverUrl: string): Promise<{
  sessionID: string
  directory: string
  system: string
}> {
  const url = new URL("/voice/brain", serverUrl)
  const headers: Record<string, string> = {
    "x-emberharmony-directory": encodeURIComponent(VOICE_PROJECT_DIR),
  }
  const username = Flag.EMBERHARMONY_SERVER_USERNAME
  const password = Flag.EMBERHARMONY_SERVER_PASSWORD
  if (password) {
    const user = username ?? "emberharmony"
    headers["authorization"] = `Basic ${Buffer.from(`${user}:${password}`).toString("base64")}`
  }

  const response = await fetch(url, { headers })
  if (!response.ok) {
    throw new Error(`failed to discover brain session: ${response.status} ${await response.text().catch(() => "")}`)
  }
  const body = (await response.json()) as { sessionID: string; directory: string; system: string }
  return body
}

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load()
  },
  entry: async (ctx: JobContext) => {
    const metadata = (() => {
      try {
        return JSON.parse(ctx.job.metadata || "{}")
      } catch {
        return {}
      }
    })()
    const { sessionID, directory, serverUrl, model } = metadata
    if (!sessionID || !directory || !serverUrl) {
      throw new Error(
        `voice agent dispatched without session metadata (got: ${ctx.job.metadata || "<empty>"}) — ` +
          "rooms must be created through EmberHarmony's POST /voice/token endpoint",
      )
    }

    // Discover the brain session — a permanent EmberHarmony session in
    // ~/.local/share/emberharmony/voice/ that holds the voice brain's
    // context, history, and tools. The bridge targets this session instead
    // of the user's project session. The user's session is only accessed
    // via server-side tools (submit_prompt, attach_session, etc.).
    const brain = await discoverBrainSession(process.env["EMBERHARMONY_VOICE_SERVER_URL"] ?? serverUrl)

    const vad = ctx.proc.userData.vad as silero.VAD
    const workflow = new VoiceWorkflow(inference.LLM.fromModelString(INTENT_MODEL))
    const session = new voice.AgentSession({
      stt: inference.STT.fromModelString(STT_MODEL),
      llm: new SessionLLM({
        // Target the brain session, not the user's project session.
        // The brain session lives in the voice project directory with its
        // own context, history, and system prompt.
        serverUrl: process.env["EMBERHARMONY_VOICE_SERVER_URL"] ?? serverUrl,
        directory: brain.directory,
        sessionID: brain.sessionID,
        username: Flag.EMBERHARMONY_SERVER_USERNAME,
        password: Flag.EMBERHARMONY_SERVER_PASSWORD,
        fallbackModel: model,
        agent: () => workflow.agent(),
        system: brain.system,
      }),
      tts: inference.TTS.fromModelString(TTS_MODEL),
      vad,
      turnDetection: new livekit.turnDetector.MultilingualModel(),
    })

    await session.start({ agent: new EmberHarmonyAgent(workflow), room: ctx.room })
    await ctx.connect()
    session.say(
      "Hey, I'm listening. We're in plan mode — tell me what you'd like to work on, and say the word when you want me to build.",
    )
  },
})

if (import.meta.main) {
  const url = Flag.EMBERHARMONY_LIVEKIT_URL
  const apiKey = Flag.EMBERHARMONY_LIVEKIT_API_KEY
  const apiSecret = Flag.EMBERHARMONY_LIVEKIT_API_SECRET
  const connects = process.argv.some((arg) => arg === "dev" || arg === "start" || arg === "connect")
  if (connects && (!url || !apiKey || !apiSecret)) {
    console.error(
      "Voice agent requires EMBERHARMONY_LIVEKIT_URL, EMBERHARMONY_LIVEKIT_API_KEY, and EMBERHARMONY_LIVEKIT_API_SECRET (or their LIVEKIT_* equivalents) to be set.",
    )
    process.exit(1)
  }
  cli.runApp(
    new ServerOptions({
      agent: fileURLToPath(import.meta.url),
      agentName: VOICE_AGENT_NAME,
      wsURL: url,
      apiKey,
      apiSecret,
      // health-check server port; 0 = ephemeral (used for serve-managed
      // workers so they never collide with a manually started one)
      port: Number(process.env["EMBERHARMONY_VOICE_WORKER_PORT"] ?? 8081),
    }),
  )
}
