import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { createEffect, createResource, createSignal, on, onCleanup, type ParentProps } from "solid-js"
import { useParams } from "@solidjs/router"
import { ConnectionState, Room, RoomEvent } from "livekit-client"
import {
  RoomContext,
  RoomAudioRenderer,
  useConnectionState,
  useLocalParticipant,
  useTranscriptions,
  useVoiceAssistant,
} from "@thesolaceproject/livekit-components-solid"
import { useSDK } from "./sdk"

export type VoiceState = "disconnected" | "connecting" | "connected" | "error"
export type MicState = "muted" | "unmuted" | "unavailable"
export type { AgentState } from "@thesolaceproject/livekit-components-solid"

// Voice intent that survives provider remounts (project switches unmount the
// whole directory-scoped provider tree) within a single app run. Deliberately
// not persisted: the microphone must never auto-enable on a fresh launch.
// Cleared only by an explicit user disconnect.
let followProjects = false
let followModel: { providerID: string; modelID: string } | undefined

const { use: useVoice, provider: VoiceValueProvider } = createSimpleContext({
  name: "Voice",
  init: (props: { room: Room }) => {
    const room = props.room
    const sdk = useSDK()
    const params = useParams()
    const [error, setError] = createSignal<string | undefined>(undefined)
    const [connecting, setConnecting] = createSignal(false)

    // the provider outlives session navigation; a connected room bridges into
    // the session it was started for, so leaving that session must hang up —
    // and when voice was active, follow the user into the new session's room
    // (each session has its own room and agent, keyed by session id)
    let followGeneration = 0
    createEffect(
      on(
        () => params.id,
        (id, prev) => {
          if (prev === undefined) {
            // first session opened after a (re)mount — resume if voice was
            // active before a project switch
            if (followProjects && id && state() === "disconnected" && !connecting()) {
              connect(id, followModel).catch(() => {})
            }
            return
          }
          const wasActive = state() === "connected" || connecting()
          const generation = ++followGeneration
          void room.disconnect().then(() => {
            if (!wasActive || !id) return
            if (generation !== followGeneration) return
            connect(id, followModel).catch(() => {})
          })
        },
        { defer: true },
      ),
    )

    const connectionState = useConnectionState(room)
    const local = useLocalParticipant(room)
    const assistant = useVoiceAssistant(room)
    const transcriptions = useTranscriptions({ room })

    // recover if WebKit blocks playback after connect (e.g. output device change)
    room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
      if (room.canPlaybackAudio) return
      const unlock = new AudioContext()
      unlock
        .resume()
        .then(() => room.startAudio())
        .then(() => unlock.close())
        .catch((err) => setError(err instanceof Error ? err.message : String(err)))
    })

    const [status, { refetch: refetchStatus }] = createResource(
      () => sdk.client,
      (client) =>
        client.voice
          .status()
          .then((x) => x.data)
          .catch(() => undefined),
    )
    const available = () => status()?.available === true

    // voice can be configured in settings while a session is open; poll until
    // available so the mic button appears without a reload
    const statusPoll = setInterval(() => {
      if (!available()) refetchStatus()
    }, 30_000)
    onCleanup(() => clearInterval(statusPoll))

    const state = (): VoiceState => {
      if (error()) return "error"
      if (connecting()) return "connecting"
      switch (connectionState()) {
        case ConnectionState.Connected:
        case ConnectionState.Reconnecting:
        case ConnectionState.SignalReconnecting:
          return "connected"
        case ConnectionState.Connecting:
          return "connecting"
        default:
          return "disconnected"
      }
    }

    const micState = (): MicState => {
      if (state() !== "connected") return "unavailable"
      return local.isMicrophoneEnabled() ? "unmuted" : "muted"
    }

    async function connect(sessionID: string, model?: { providerID: string; modelID: string }) {
      if (state() === "connecting" || state() === "connected") return
      setError(undefined)
      setConnecting(true)
      followProjects = true
      if (model) followModel = model
      try {
        const grant = await sdk.client.voice.token({ sessionID, model }).then((x) => x.data)
        if (!grant) throw new Error("voice token request failed")
        // WKWebView keeps media silently "playing" until the page's audio
        // session activates; resuming an AudioContext inside the connect
        // gesture activates it (observed: audio elements were healthy but
        // inaudible until an AudioContext was created)
        const unlock = new AudioContext()
        try {
          await unlock.resume().catch(() => {})
          await room.connect(grant.url, grant.token)
          await room.startAudio()
        } finally {
          await unlock.close().catch(() => {})
        }
        await room.localParticipant.setMicrophoneEnabled(true)
      } catch (err) {
        // unwind a partially joined room so a retry starts from clean state
        await room.disconnect().catch(() => {})
        setError(err instanceof Error ? err.message : String(err))
        throw err
      } finally {
        setConnecting(false)
      }
    }

    async function disconnect() {
      followProjects = false
      setError(undefined)
      await room.disconnect()
    }

    async function toggleMute() {
      if (state() !== "connected") return
      await room.localParticipant.setMicrophoneEnabled(!local.isMicrophoneEnabled())
    }

    onCleanup(() => {
      room.disconnect()
    })

    // resume voice after a project switch remounted the provider
    if (followProjects && params.id) {
      void connect(params.id, followModel).catch(() => {})
    }

    return {
      room,
      state,
      micState,
      error,
      available,
      connect,
      disconnect,
      toggleMute,
      agentState: assistant.state,
      agentAudioTrack: assistant.audioTrack,
      transcriptions,
    }
  },
})

export { useVoice }

export function VoiceProvider(props: ParentProps) {
  const room = new Room({ adaptiveStream: true, dynacast: true })
  return (
    <RoomContext room={room}>
      <VoiceValueProvider room={room}>
        <RoomAudioRenderer />
        {props.children}
      </VoiceValueProvider>
    </RoomContext>
  )
}
