import { fileURLToPath } from "node:url"
import { type JobContext, type JobProcess, ServerOptions, cli, defineAgent, inference, voice } from "@livekit/agents"
import * as livekit from "@livekit/agents-plugin-livekit"
import * as silero from "@livekit/agents-plugin-silero"
import { llm } from "@livekit/agents"
import { Flag } from "../flag/flag"
import { SessionLLM } from "./bridge"
import { VoiceRegistry } from "./registry"
import { VOICE_AGENT_NAME } from "./constants"
import { VoiceWorkflow, type Stage } from "./workflow"
import { BRAIN_SYSTEM_PROMPT, VOICE_PROJECT_DIR } from "./brain"

// Model strings accept an optional ":language" (STT) or ":voice" (TTS) suffix.
// The serve command injects these from the resolved voice config when it
// spawns the worker; standalone workers fall back to the registry defaults.
const STT_MODEL = process.env["EMBERHARMONY_VOICE_STT_MODEL"] ?? VoiceRegistry.DEFAULT_STT
const TTS_MODEL = process.env["EMBERHARMONY_VOICE_TTS_MODEL"] ?? VoiceRegistry.DEFAULT_TTS
// Small fast gateway model that routes plan/build per spoken turn
const INTENT_MODEL = process.env["EMBERHARMONY_VOICE_INTENT_MODEL"] ?? VoiceRegistry.DEFAULT_INTENT

/** Participant attribute keys published by the voice agent */
const ATTR_ATTACHED_SESSION = "emberharmony.attached_session"

class EmberHarmonyAgent extends voice.Agent {
  #workflow: VoiceWorkflow
  #publishState: () => Promise<void>

  constructor(workflow: VoiceWorkflow, publishState: () => Promise<void>) {
    super({
      // The session bridge holds the real context server-side; these
      // instructions only exist because voice.Agent requires them.
      instructions: "You are EmberHarmony, a voice interface to a coding assistant session.",
    })
    this.#workflow = workflow
    this.#publishState = publishState
  }

  override async onUserTurnCompleted(_chatCtx: llm.ChatContext, newMessage: llm.ChatMessage): Promise<void> {
    await this.#workflow.route(newMessage.textContent ?? "")
    await this.#publishState()
  }
}

/**
 * Publish the attached-session pointer as a participant attribute.
 *
 * Plan/build is a coding-session concept — only meaningful when attached to a
 * project session, and gated in submit_prompt — NOT a conversation "mode." So we
 * deliberately do NOT publish voice_mode / voice_stage here; surfacing them made
 * the agent claim "we're in plan mode" with nothing attached. They return in a
 * later phase, scoped to the attached/Operator state once the worker tracks it.
 * setAttributes is a full replacement, so we merge with existing attrs.
 */
async function publishStage(room: JobContext["room"], attachedSessionID?: string): Promise<void> {
  const lp = room.localParticipant
  if (!lp || !room.isConnected) return
  const existing = lp.attributes ?? {}
  await lp.setAttributes({
    ...existing,
    [ATTR_ATTACHED_SESSION]: attachedSessionID ?? "",
  })
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
    const { projectID, directory, serverUrl, model, brainModel, structured } = metadata
    if (!projectID || !directory || !serverUrl) {
      throw new Error(
        `voice agent dispatched without project metadata (got: ${ctx.job.metadata || "<empty>"}) — ` +
          "rooms must be created through EmberHarmony's POST /voice/token endpoint",
      )
    }

    // Discover the brain session — a permanent EmberHarmony session in
    // ~/.local/share/emberharmony/voice/ that holds the voice brain's
    // context, history, and tools. The bridge targets this session instead
    // of the user's project session. The user's session is only accessed
    // via server-side tools (submit_prompt, attach_session, etc.).
    const brain = await discoverBrainSession(process.env["EMBERHARMONY_VOICE_SERVER_URL"] ?? serverUrl)

    // Parse the configured brain model (e.g. "anthropic/claude-sonnet-4-20250514")
    // into { providerID, modelID } for the session bridge fallback
    const brainFallback = brainModel
      ? { providerID: brainModel.split("/")[0], modelID: brainModel.split("/").slice(1).join("/") }
      : model

    const vad = ctx.proc.userData.vad as silero.VAD
    const workflow = new VoiceWorkflow({
      intent: inference.LLM.fromModelString(INTENT_MODEL),
      structured: structured ?? false,
    })
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
        fallbackModel: brainFallback,
        agent: () => workflow.agent(),
        system: brain.system,
        // The brain session should NOT be aborted on every voice interruption.
        // The brain controls aborts deliberately — it only aborts the attached
        // project session when it decides the session should stop, not as a
        // side effect of the user interrupting narration.
        abortOnInterrupt: false,
      }),
      tts: inference.TTS.fromModelString(TTS_MODEL),
      vad,
      turnDetection: new livekit.turnDetector.MultilingualModel(),
      // Increase the minimum endpointing delay to reduce premature turn
      // completions. The default 500ms is too aggressive for natural
      // speech pauses — 800ms gives the user more time to continue
      // speaking without the agent cutting in.
      turnHandling: {
        endpointing: { minDelay: 800 },
      },
    })

    await session.start({
      agent: new EmberHarmonyAgent(workflow, async () => {
        await publishStage(ctx.room)
      }),
      room: ctx.room,
    })
    await ctx.connect()

    // Publish initial state. Plan/build is not surfaced as a conversation mode.
    await publishStage(ctx.room)

    // Casual, non-plan/build opening — nothing is attached, so there is no
    // "mode," just conversation. (A later phase makes this orient-first: survey
    // recent voice sessions + compaction summaries, then greet informed by what
    // was last discussed.)
    session.say("Hey — good to hear you. What's on your mind?")
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
      port: Number(process.env["EMBERHARMONY_VOICE_WORKER_PORT"] ?? 47819),
    }),
  )
}
