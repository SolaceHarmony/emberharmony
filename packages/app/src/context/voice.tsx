import { createSimpleContext } from "@thesolaceproject/emberharmony-ui/context"
import { createEffect, createResource, createSignal, on, onCleanup, type Accessor, type ParentProps } from "solid-js"
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
import {
  beginVoiceTypedInput,
  getVoiceSettingsState,
  getVoiceStatus,
  interruptVoice,
  isDesktop,
  setVoiceMicEnabled,
  startVoice,
  stopVoice,
  VOICE_SETTINGS_CHANGED,
  type NativeVoiceEvent,
  type NativeVoiceState,
  type VoiceStartContext,
  type VoiceProvider,
} from "@/lib/voice-settings"
import type { VoiceNativeStatus } from "@/lib/voice-state"
import { useSDK } from "./sdk"

export type VoiceState = "disconnected" | "connecting" | "connected" | "error"
export type MicState = "muted" | "unmuted" | "unavailable"
export type { AgentState } from "@thesolaceproject/livekit-components-solid"

type VoiceTrack = ReturnType<typeof useVoiceAssistant>["audioTrack"] extends Accessor<infer Track> ? Track : never
type VoiceTranscription = ReturnType<ReturnType<typeof useTranscriptions>>[number]
type StartContext = Omit<VoiceStartContext, "sessionID" | "directory">

type VoiceValue = {
  state: () => VoiceState
  micState: () => MicState
  enabled: () => boolean
  provider: () => VoiceProvider
  error: Accessor<string | undefined>
  available: () => boolean
  connect: (sessionID: string, ctx?: StartContext) => Promise<void>
  disconnect: () => Promise<void>
  interrupt: () => Promise<void>
  beginTypedInput: () => Promise<void>
  setMicEnabled: (enabled: boolean) => Promise<void>
  toggleMute: () => Promise<void>
  agentState: () => AgentState
  agentAudioTrack: () => VoiceTrack | undefined
  agentLevel: () => number | undefined
  turnActive: () => boolean
  transcriptions: () => VoiceTranscription[]
}

// Voice intent that survives provider remounts within one desktop app run.
// It is intentionally not persisted; the microphone must not auto-enable after launch.
let followProjects = false
let followContext: StartContext | undefined

async function loadNativeStatus(): Promise<VoiceNativeStatus | undefined> {
  const plan = await getVoiceStatus().catch(() => undefined)
  if (!plan) return undefined
  const state = await getVoiceSettingsState().catch(() => undefined)
  if (!state) return undefined
  return { plan, settings: state.settings, stored: state.stored }
}

function nativeAgentState(state: NativeVoiceState): AgentState {
  if (state === "loading") return "initializing"
  return state
}

function nativeTranscript(line: { agent: boolean; text: string } | undefined): VoiceTranscription[] {
  if (!line) return []
  return [
    {
      text: line.text,
      participantInfo: { identity: line.agent ? "agent-native" : "user-native" },
    },
  ] as VoiceTranscription[]
}

function createDesktopVoice(): VoiceValue {
  const sdk = useSDK()
  const params = useParams()
  const [error, setError] = createSignal<string | undefined>(undefined)
  const [connecting, setConnecting] = createSignal(false)
  const [state, setState] = createSignal<VoiceState>("disconnected")
  const [agent, setAgent] = createSignal<AgentState>("disconnected")
  const [line, setLine] = createSignal<{ agent: boolean; text: string } | undefined>(undefined)
  const [level, setLevel] = createSignal(0)
  const [native, { refetch: refetchNative, mutate: setNative }] = createResource(loadNativeStatus)

  const provider = () => native()?.plan.provider ?? "off"
  const enabled = () => native()?.plan.enabled ?? false
  const running = () => native()?.plan.running === true

  createEffect(
    on(
      () => native()?.plan,
      (plan) => {
        if (!plan) return
        if (plan.running) return
        if (state() !== "connected" && state() !== "connecting" && agent() !== "thinking" && agent() !== "speaking") {
          return
        }
        clear()
      },
    ),
  )

  function clear() {
    setConnecting(false)
    setState("disconnected")
    setAgent("disconnected")
    setLine(undefined)
    setLevel(0)
  }

  async function refresh() {
    const next = await loadNativeStatus()
    if (next) setNative(next)
    return next
  }

  function handle(event: NativeVoiceEvent) {
    switch (event.type) {
      case "state":
        setAgent(nativeAgentState(event.state))
        if (event.state !== "speaking") setLevel(0)
        setState(
          event.state === "loading"
            ? "connecting"
            : event.state === "idle" && !running()
              ? "disconnected"
              : "connected",
        )
        setError(undefined)
        break
      case "transcript":
        setLine({ agent: event.role === "assistant", text: event.text })
        break
      case "level":
        setLevel(Math.min(1, Math.max(0, event.rms * 24)))
        break
      case "ended":
        clear()
        refresh().catch(() => {})
        if (event.reason) setError(event.reason)
        break
      case "error":
        setError(event.message)
        setState("error")
        setAgent("failed")
        setLevel(0)
        refresh().catch(() => {})
        break
    }
  }

  createEffect(
    on(
      () => params.id,
      (id, prev) => {
        if (prev === undefined) {
          if (followProjects && id && state() === "disconnected" && !connecting()) {
            connect(id, followContext).catch(() => {})
          }
          return
        }
        const wasActive = state() === "connected" || connecting()
        stopVoice()
          .catch(() => {})
          .then(() => {
            clear()
            if (!wasActive || !id) return
            connect(id, followContext).catch(() => {})
          })
      },
    ),
  )

  const poll = setInterval(() => {
    refetchNative()
  }, 30_000)
  onCleanup(() => clearInterval(poll))

  const refreshSettings = () => {
    refresh().catch(() => {})
  }
  window.addEventListener(VOICE_SETTINGS_CHANGED, refreshSettings)
  onCleanup(() => window.removeEventListener(VOICE_SETTINGS_CHANGED, refreshSettings))

  async function connect(sessionID: string, ctx?: StartContext) {
    if (state() === "connecting" || state() === "connected") return
    const current = await refresh()
    const plan = current?.plan
    followProjects = true
    followContext = ctx
    setError(undefined)
    setConnecting(true)
    setState("connecting")
    setAgent("connecting")
    setLine(undefined)
    setLevel(0)
    try {
      const result = await startVoice(
        {
          sessionID,
          directory: sdk.directory,
          ...ctx,
        },
        handle,
      )
      if (plan && result.provider !== plan.provider) throw new Error("Voice provider changed while starting.")
      await refresh().catch(() => undefined)
    } catch (err) {
      followProjects = false
      followContext = undefined
      await stopVoice().catch(() => {})
      clear()
      setError(err instanceof Error ? err.message : String(err))
      throw err
    } finally {
      setConnecting(false)
    }
  }

  async function disconnect() {
    followProjects = false
    followContext = undefined
    setError(undefined)
    await stopVoice()
    await refresh().catch(() => undefined)
    setConnecting(false)
    clear()
  }

  async function interrupt() {
    if (state() !== "connected" && state() !== "connecting") return
    await interruptVoice()
  }

  async function setMicEnabled(enabled: boolean) {
    if (state() !== "connected") return
    await setVoiceMicEnabled(enabled)
    await refresh().catch(() => undefined)
  }

  async function beginTypedInput() {
    if (state() !== "connected") return
    await beginVoiceTypedInput()
    await refresh().catch(() => undefined)
  }

  const micState = (): MicState => {
    if (state() !== "connected") return "unavailable"
    return native()?.plan.micEnabled ? "unmuted" : "muted"
  }

  async function toggleMute() {
    await setMicEnabled(micState() !== "unmuted")
  }

  onCleanup(() => {
    stopVoice().catch(() => {})
  })

  return {
    state: () => (error() ? "error" : connecting() ? "connecting" : state()),
    micState,
    enabled,
    provider,
    error,
    available: () => state() === "connected" || state() === "connecting" || native()?.plan.ready === true,
    connect,
    disconnect,
    interrupt,
    beginTypedInput,
    setMicEnabled,
    toggleMute,
    agentState: () => agent(),
    agentAudioTrack: () => undefined,
    agentLevel: () => (running() ? level() : undefined),
    turnActive: () => agent() === "thinking" || agent() === "speaking",
    transcriptions: () => nativeTranscript(line()),
  }
}

function createWebVoice(room: Room): VoiceValue {
  const sdk = useSDK()
  const params = useParams()
  const [error, setError] = createSignal<string | undefined>(undefined)
  const [connecting, setConnecting] = createSignal(false)
  const connectionState = useConnectionState(room)
  const local = useLocalParticipant(room)
  const assistant = useVoiceAssistant(room)
  const transcriptions = useTranscriptions({ room })
  const [status, { refetch: refetchStatus }] = createResource(
    () => sdk.client,
    (client) =>
      client.voice
        .status()
        .then((x) => x.data)
        .catch(() => undefined),
  )

  const poll = setInterval(() => {
    if (!status()?.available) refetchStatus()
  }, 30_000)
  onCleanup(() => clearInterval(poll))

  room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
    if (room.canPlaybackAudio) return
    const unlock = new AudioContext()
    unlock
      .resume()
      .then(() => room.startAudio())
      .then(() => unlock.close())
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
  })

  createEffect(
    on(
      () => params.id,
      (id, prev) => {
        if (prev === undefined) {
          if (followProjects && id && state() === "disconnected" && !connecting()) {
            connect(id, followContext).catch(() => {})
          }
          return
        }
        const wasActive = state() === "connected" || connecting()
        room
          .disconnect()
          .then(() => {
            if (!wasActive || !id) return
            connect(id, followContext).catch(() => {})
          })
          .catch(() => {})
      },
    ),
  )

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

  async function connect(sessionID: string, ctx?: StartContext) {
    if (state() === "connecting" || state() === "connected") return
    followProjects = true
    followContext = ctx
    setError(undefined)
    setConnecting(true)
    try {
      const grant = await sdk.client.voice.token({ sessionID, model: ctx?.model }).then((x) => x.data)
      if (!grant) throw new Error("voice token request failed")
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
      followProjects = false
      followContext = undefined
      await room.disconnect().catch(() => {})
      setError(err instanceof Error ? err.message : String(err))
      throw err
    } finally {
      setConnecting(false)
    }
  }

  async function disconnect() {
    followProjects = false
    followContext = undefined
    setError(undefined)
    await room.localParticipant.setMicrophoneEnabled(false).catch(() => {})
    await room.disconnect()
  }

  async function interrupt() {
    if (state() !== "connected" && state() !== "connecting") return
    followProjects = false
    followContext = undefined
    await room.localParticipant.setMicrophoneEnabled(false).catch(() => {})
    await room.disconnect()
    setError(undefined)
    setConnecting(false)
  }

  async function setMicEnabled(enabled: boolean) {
    if (state() !== "connected") return
    await room.localParticipant.setMicrophoneEnabled(enabled)
  }

  async function beginTypedInput() {
    await setMicEnabled(false)
  }

  async function toggleMute() {
    if (state() !== "connected") return
    await room.localParticipant.setMicrophoneEnabled(!local.isMicrophoneEnabled())
  }

  onCleanup(() => {
    room.disconnect()
  })

  return {
    state,
    micState: () => {
      if (state() !== "connected") return "unavailable"
      return local.isMicrophoneEnabled() ? "unmuted" : "muted"
    },
    enabled: () => status()?.available === true,
    provider: () => "livekit",
    error,
    available: () => state() === "connected" || state() === "connecting" || status()?.available === true,
    connect,
    disconnect,
    interrupt,
    beginTypedInput,
    setMicEnabled,
    toggleMute,
    agentState: () => assistant.state(),
    agentAudioTrack: () => assistant.audioTrack(),
    agentLevel: () => undefined,
    turnActive: () => {
      const current = assistant.state()
      return current === "thinking" || current === "speaking"
    },
    transcriptions,
  }
}

const { use: useVoice, provider: VoiceValueProvider } = createSimpleContext({
  name: "Voice",
  init: (props: { room?: Room }) => {
    if (isDesktop()) return createDesktopVoice()
    if (!props.room) throw new Error("LiveKit room is required outside the desktop shell.")
    return createWebVoice(props.room)
  },
})

export { useVoice }

export function VoiceProvider(props: ParentProps) {
  if (isDesktop()) {
    return <VoiceValueProvider>{props.children}</VoiceValueProvider>
  }

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
