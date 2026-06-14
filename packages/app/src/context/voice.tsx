import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { createEffect, createResource, createSignal, on, onCleanup, type ParentProps, Show } from "solid-js"
import { useParams } from "@solidjs/router"
import { ConnectionState, Room, RoomEvent } from "livekit-client"
import {
  RoomContext,
  RoomAudioRenderer,
  useConnectionState,
  useLocalParticipant,
  useTranscriptions,
  useVoiceAssistant,
  type AgentState,
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

// The room is created lazily on first connect() and reused across
// disconnect/reconnect cycles. It is never destroyed during the app's
// lifetime — only room.disconnect() is called, which releases WebRTC
// resources while keeping the Room object viable for reconnect.
// This avoids the ~50MB+ per-cycle WebRTC leak that pushed WKWebView
// past its ~1.5GB OOM ceiling on repeated session switches.
let roomInstance: Room | undefined

function getOrCreateRoom(): Room {
  if (!roomInstance) {
    roomInstance = new Room({ adaptiveStream: true, dynacast: true })
  }
  return roomInstance
}

const { use: useVoice, provider: VoiceValueProvider } = createSimpleContext({
  name: "Voice",
  init: () => {
    const sdk = useSDK()
    const params = useParams()
    const [error, setError] = createSignal<string | undefined>(undefined)
    const [connecting, setConnecting] = createSignal(false)
    const [room, setRoom] = createSignal<Room | undefined>(roomInstance)

    // Reactive state bridged from the room island. Until a room connects,
    // these are their default values. The island writes to these signals
    // inside RoomContext where the livekit-solid hooks can run.
    const [connectionState, setConnectionState] = createSignal<ConnectionState>(ConnectionState.Disconnected)
    const [micEnabled, setMicEnabled] = createSignal(false)
    const [agentState, setAgentState] = createSignal<AgentState>("disconnected")
    // Track and transcription types come from livekit-components-core which
    // isn't a direct app dependency. These are bridged from the room island
    // where the livekit hooks run — the types match what useVoiceAssistant and
    // useTranscriptions return.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [agentAudioTrack, setAgentAudioTrack] = createSignal<any>(undefined)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [voiceTranscriptions, setVoiceTranscriptions] = createSignal<any[]>([])

    // the provider outlives session navigation; a connected room bridges into
    // the session it was started for, so leaving that session must hang up —
    // and when voice was active, follow the user into the new session's room
    let followGeneration = 0
    createEffect(
      on(
        () => params.id,
        (id, prev) => {
          const r = room()
          if (!r) return
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
          void r.disconnect().then(() => {
            if (!wasActive || !id) return
            if (generation !== followGeneration) return
            connect(id, followModel).catch(() => {})
          })
        },
        { defer: true },
      ),
    )

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
      return micEnabled() ? "unmuted" : "muted"
    }

    async function connect(sessionID: string, model?: { providerID: string; modelID: string }) {
      if (state() === "connecting" || state() === "connected") return
      setError(undefined)
      setConnecting(true)
      followProjects = true
      if (model) followModel = model
      try {
        const r = getOrCreateRoom()
        setRoom(r)
        const grant = await sdk.client.voice.token({ sessionID, model }).then((x) => x.data)
        if (!grant) throw new Error("voice token request failed")
        // WKWebView keeps media silently "playing" until the page's audio
        // session activates; resuming an AudioContext inside the connect
        // gesture activates it (observed: audio elements were healthy but
        // inaudible until an AudioContext was created)
        const unlock = new AudioContext()
        try {
          await unlock.resume().catch(() => {})
          await r.connect(grant.url, grant.token)
          await r.startAudio()
        } finally {
          await unlock.close().catch(() => {})
        }
        await r.localParticipant.setMicrophoneEnabled(true)
      } catch (err) {
        // unwind a partially joined room so a retry starts from clean state
        const r = room()
        if (r) await r.disconnect().catch(() => {})
        setError(err instanceof Error ? err.message : String(err))
        throw err
      } finally {
        setConnecting(false)
      }
    }

    async function disconnect() {
      followProjects = false
      setError(undefined)
      const r = room()
      if (r) await r.disconnect()
    }

    async function toggleMute() {
      if (state() !== "connected") return
      const r = room()
      if (r) await r.localParticipant.setMicrophoneEnabled(!r.localParticipant.isMicrophoneEnabled)
    }

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
      agentState,
      agentAudioTrack,
      transcriptions: voiceTranscriptions,
      // bridge setters for the room island
      setConnectionState,
      setMicEnabled,
      setAgentState,
      setAgentAudioTrack,
      setVoiceTranscriptions,
    }
  },
})

export { useVoice }

/**
 * Mounts LiveKit hooks against a real Room. Only rendered when a Room exists,
 * so hooks always receive a valid Room instance. Bridges reactive state back
 * to the outer VoiceValueProvider so consumers that never enter RoomContext
 * can still read connection state, mic state, etc.
 */
function VoiceRoomIsland(props: { room: Room }) {
  const voice = useVoice()
  const room = props.room
  const connectionState = useConnectionState(room)
  const local = useLocalParticipant(room)
  const assistant = useVoiceAssistant(room)
  const voiceTranscriptions = useTranscriptions({ room })

  // Bridge reactive state from the island to the outer context
  createEffect(() => voice.setConnectionState(connectionState()))
  createEffect(() => voice.setMicEnabled(local.isMicrophoneEnabled()))
  createEffect(() => voice.setAgentState(assistant.state()))
  createEffect(() => voice.setAgentAudioTrack(assistant.audioTrack()))
  createEffect(() => voice.setVoiceTranscriptions(voiceTranscriptions()))

  // recover if WebKit blocks playback after connect (e.g. output device change)
  const onAudioPlaybackStatusChanged = () => {
    if (room.canPlaybackAudio) return
    const unlock = new AudioContext()
    unlock
      .resume()
      .then(() => room.startAudio())
      .catch((err) => {
        const message = err instanceof Error ? err.message : String(err)
        console.error("[voice] startAudio failed after AudioContext unlock:", message)
      })
      // always release the AudioContext — leaking one per playback-status
      // flip accumulates WebAudio buffers and pushes the WKWebView content
      // process toward its ~1.5GB OOM ceiling
      .finally(() => void unlock.close().catch(() => {}))
  }
  room.on(RoomEvent.AudioPlaybackStatusChanged, onAudioPlaybackStatusChanged)
  onCleanup(() => room.off(RoomEvent.AudioPlaybackStatusChanged, onAudioPlaybackStatusChanged))

  return (
    <>
      <RoomAudioRenderer />
    </>
  )
}

export function VoiceProvider(props: ParentProps) {
  return (
    <VoiceValueProvider>
      {props.children}
      <RoomIsland />
    </VoiceValueProvider>
  )
}

/**
 * Conditionally mounts RoomContext + LiveKit hooks only when a Room exists.
 * This is a sibling of VoiceProvider's children, not a gate — children always
 * render regardless of whether voice is connected. The island only mounts the
 * heavy WebRTC/audio resources when the user actually presses the mic button.
 */
function RoomIsland() {
  const voice = useVoice()
  return (
    <Show when={voice.room()}>
      {(room) => (
        <RoomContext room={room()}>
          <VoiceRoomIsland room={room()} />
        </RoomContext>
      )}
    </Show>
  )
}
