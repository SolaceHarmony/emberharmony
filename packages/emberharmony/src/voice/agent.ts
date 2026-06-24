import { fileURLToPath } from "node:url"
import {
  type JobContext,
  type JobProcess,
  ServerOptions,
  AgentServer,
  defineAgent,
  inference,
  voice,
  initializeLogger,
} from "@livekit/agents"
import * as livekit from "@livekit/agents-plugin-livekit"
import * as silero from "@livekit/agents-plugin-silero"
import { llm } from "@livekit/agents"
import { Flag } from "../flag/flag"
import { SessionLLM } from "./bridge"
import { VoiceRegistry } from "./registry"
import { VOICE_AGENT_NAME } from "./constants"
import { VoiceWorkflow, VOICE_SYSTEM_PROMPT } from "./workflow"
import { createServer as createIpcServer, type Server as IpcServer } from "node:net"

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

    const vad = ctx.proc.userData.vad as silero.VAD
    const workflow = new VoiceWorkflow(inference.LLM.fromModelString(INTENT_MODEL))
    const session = new voice.AgentSession({
      stt: inference.STT.fromModelString(STT_MODEL),
      llm: new SessionLLM({
        serverUrl: process.env["EMBERHARMONY_VOICE_SERVER_URL"] ?? serverUrl,
        directory,
        sessionID,
        username: Flag.EMBERHARMONY_SERVER_USERNAME,
        password: Flag.EMBERHARMONY_SERVER_PASSWORD,
        fallbackModel: model,
        agent: () => workflow.agent(),
        system: VOICE_SYSTEM_PROMPT,
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
  if (!url || !apiKey || !apiSecret) {
    console.error(
      "Voice agent requires EMBERHARMONY_LIVEKIT_URL, EMBERHARMONY_LIVEKIT_API_KEY, and EMBERHARMONY_LIVEKIT_API_SECRET (or their LIVEKIT_* equivalents) to be set.",
    )
    process.exit(1)
  }

  initializeLogger({ pretty: false, level: process.env["EMBERHARMONY_VOICE_LOG_LEVEL"] ?? "info" })

  const server = new AgentServer(
    new ServerOptions({
      agent: fileURLToPath(import.meta.url),
      agentName: VOICE_AGENT_NAME,
      wsURL: url,
      apiKey,
      apiSecret,
      port: Number(process.env["EMBERHARMONY_VOICE_WORKER_PORT"] ?? 8081),
    }),
  )

  let ipc: IpcServer | undefined

  const shutdown = async () => {
    if (ipc) ipc.close()
    await server.drain(5000).catch(() => {})
    await server.close().catch(() => {})
    process.exit(0)
  }

  const socketPath = process.env["EMBERHARMONY_VOICE_IPC_SOCKET"]
  if (socketPath) {
    ipc = createIpcServer((conn) => {
      conn.on("data", (data) => {
        if (data.toString().trim() === "shutdown") void shutdown()
      })
      conn.on("error", () => {})
    })
    ipc.listen(socketPath)
  }

  process.on("SIGTERM", () => void shutdown())
  process.on("SIGINT", () => void shutdown())

  void server.run().catch((err) => {
    console.error("voice agent worker failed:", err)
    void shutdown()
  })
}
