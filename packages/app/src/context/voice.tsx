import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { createResource, createSignal, onCleanup, type ParentProps } from "solid-js"
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

const { use: useVoice, provider: VoiceValueProvider } = createSimpleContext({
  name: "Voice",
  init: (props: { room: Room }) => {
    const room = props.room
    const sdk = useSDK()
    const [error, setError] = createSignal<string | undefined>(undefined)
    const [connecting, setConnecting] = createSignal(false)

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

    const [status] = createResource(
      () => sdk.client,
      (client) =>
        client.voice
          .status()
          .then((x) => x.data)
          .catch(() => undefined),
    )
    const available = () => status()?.available === true

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
      try {
        const grant = await sdk.client.voice.token({ sessionID, model }).then((x) => x.data)
        if (!grant) throw new Error("voice token request failed")
        // WKWebView keeps media silently "playing" until the page's audio
        // session activates; resuming an AudioContext inside the connect
        // gesture activates it (observed: audio elements were healthy but
        // inaudible until an AudioContext was created)
        const unlock = new AudioContext()
        await unlock.resume().catch(() => {})
        await room.connect(grant.url, grant.token)
        await room.startAudio()
        await unlock.close().catch(() => {})
        await room.localParticipant.setMicrophoneEnabled(true)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        throw err
      } finally {
        setConnecting(false)
      }
    }

    async function disconnect() {
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
