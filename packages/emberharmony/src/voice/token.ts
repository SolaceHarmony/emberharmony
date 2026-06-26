import { AccessToken, AgentDispatchClient, RoomServiceClient } from "livekit-server-sdk"
import { RoomConfiguration, RoomAgentDispatch } from "@livekit/protocol"
import z from "zod"
import { NamedError } from "@thesolaceproject/emberharmony-util/error"
import { Auth } from "../auth"
import { Config } from "../config/config"
import { Flag } from "../flag/flag"
import { Log } from "../util/log"
import { VoiceRegistry } from "./registry"
import { VOICE_AGENT_NAME } from "./constants"

export namespace Voice {
  const log = Log.create({ service: "voice.dispatch" })

  export const AGENT_NAME = VOICE_AGENT_NAME

  /** Auth store entry holding the LiveKit API key (key) and secret (secret) */
  export const AUTH_PROVIDER_ID = "livekit"

  export const NotConfiguredError = NamedError.create("VoiceNotConfiguredError", z.object({}))

  export interface Settings {
    disabled: boolean
    url?: string
    apiKey?: string
    apiSecret?: string
    stt: string
    tts: string
    intent: string
    available: boolean
  }

  /**
   * Effective voice settings: config + credential store first, environment
   * variables (EMBERHARMONY_LIVEKIT_* / LIVEKIT_* / EMBERHARMONY_VOICE_*) as
   * fallback for CI and the standalone worker. Pass `override` to resolve
   * against a just-written voice config instead of the instance cache.
   */
  export async function settings(override?: Config.Voice): Promise<Settings> {
    const voice: Config.Voice = override ?? (await Config.get()).voice ?? {}
    const auth = await Auth.get(AUTH_PROVIDER_ID)
    const credentials = auth?.type === "api" ? auth : undefined
    const url = voice.livekit?.url ?? Flag.EMBERHARMONY_LIVEKIT_URL
    const apiKey = credentials?.key ?? Flag.EMBERHARMONY_LIVEKIT_API_KEY
    const apiSecret = credentials?.secret ?? Flag.EMBERHARMONY_LIVEKIT_API_SECRET
    const disabled = voice.disabled ?? Flag.EMBERHARMONY_VOICE_DISABLE
    return {
      disabled,
      url,
      apiKey,
      apiSecret,
      stt: voice.stt ?? process.env["EMBERHARMONY_VOICE_STT_MODEL"] ?? VoiceRegistry.DEFAULT_STT,
      tts: voice.tts ?? process.env["EMBERHARMONY_VOICE_TTS_MODEL"] ?? VoiceRegistry.DEFAULT_TTS,
      intent: voice.intent ?? process.env["EMBERHARMONY_VOICE_INTENT_MODEL"] ?? VoiceRegistry.DEFAULT_INTENT,
      available: Boolean(!disabled && url && apiKey && apiSecret),
    }
  }

  export async function token(opts: {
    roomName: string
    identity: string
    name?: string
    agentName?: string
    agentMetadata?: string
  }) {
    const resolved = await settings()
    if (!resolved.available) throw new NotConfiguredError({})
    const token = new AccessToken(resolved.apiKey!, resolved.apiSecret!, {
      identity: opts.identity,
      name: opts.name,
      ttl: "15m",
    })
    token.addGrant({
      room: opts.roomName,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    })
    if (opts.agentName) {
      token.roomConfig = new RoomConfiguration({
        agents: [new RoomAgentDispatch({ agentName: opts.agentName, metadata: opts.agentMetadata })],
      })
    }
    return { token: await token.toJwt(), url: resolved.url! }
  }

  /**
   * Token roomConfig dispatch only fires when LiveKit creates the room. A
   * reconnect shortly after a disconnect can join a still-lingering room and
   * end up with no agent. If the room already exists without an agent
   * participant, dispatch one explicitly.
   *
   * Concurrent calls for the same room are deduplicated via `inflight` so
   * only one dispatch is created per room at a time.
   */
  const inflight = new Map<string, Promise<void>>()

  export async function ensureAgentDispatched(opts: { roomName: string; agentName: string; metadata: string }) {
    const existing = inflight.get(opts.roomName)
    if (existing) {
      log.info("dispatch dedup: joining inflight request", { room: opts.roomName })
      return existing
    }
    const promise = doEnsureAgentDispatched(opts).finally(() => inflight.delete(opts.roomName))
    inflight.set(opts.roomName, promise)
    return promise
  }

  async function doEnsureAgentDispatched(opts: { roomName: string; agentName: string; metadata: string }) {
    const resolved = await settings()
    if (!resolved.available) return
    const url = resolved.url!.replace(/^ws/, "http")
    const rooms = new RoomServiceClient(url, resolved.apiKey!, resolved.apiSecret!)
    const existing = await rooms.listRooms([opts.roomName]).catch(() => [])
    if (existing.length === 0) {
      // fresh room — token roomConfig dispatches the agent on creation
      log.info("dispatch skip: fresh room, roomConfig will dispatch", { room: opts.roomName })
      return
    }
    const participants = await rooms.listParticipants(opts.roomName).catch(() => [])
    if (participants.some((p) => p.identity.startsWith("agent"))) {
      log.info("dispatch skip: agent already a participant", { room: opts.roomName })
      return
    }
    const dispatch = new AgentDispatchClient(url, resolved.apiKey!, resolved.apiSecret!)
    // An agent dispatched via the token's roomConfig (on a slightly earlier
    // connect) may have been requested but not yet joined as a participant —
    // listParticipants can't see it, so without this check a reconnect during
    // that window creates a second dispatch and the worker forks a duplicate
    // agent job process for the same room. Skip if this agent is already
    // dispatched (pending or active).
    const dispatched = await dispatch.listDispatch(opts.roomName).catch(() => [])
    if (dispatched.some((d) => d.agentName === opts.agentName)) {
      log.info("dispatch skip: agent already dispatched (pending/active)", { room: opts.roomName })
      return
    }
    log.info("dispatch create: no agent present, dispatching", { room: opts.roomName, agent: opts.agentName })
    await dispatch.createDispatch(opts.roomName, opts.agentName, { metadata: opts.metadata })
  }
}
