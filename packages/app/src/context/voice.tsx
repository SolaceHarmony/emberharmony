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
  type AgentState,
} from "@thesolaceproject/livekit-components-solid"
import {
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
  type VoicePlan,
  type VoiceProvider,
  type VoiceSettingsChangedEvent,
} from "@/lib/voice-settings"
import {
  shouldStopRuntimeForProviderChange,
  voiceEnabled,
  voiceProvider,
  type VoiceNativeStatus,
} from "@/lib/voice-state"
import { useSDK } from "./sdk"

export type VoiceState = "disconnected" | "connecting" | "connected" | "error"
export type MicState = "muted" | "unmuted" | "unavailable"
export type { AgentState } from "@thesolaceproject/livekit-components-solid"

// Voice intent that survives provider remounts (project switches unmount the
// whole directory-scoped provider tree) within a single app run. Deliberately
// not persisted: the microphone must never auto-enable on a fresh launch.
// Cleared only by an explicit user disconnect.
let followProjects = false
let followContext: Omit<VoiceStartContext, "sessionID" | "directory"> | undefined

async function loadNativeStatus(desktop: boolean): Promise<VoiceNativeStatus | undefined> {
  if (!desktop) return undefined
  const plan = await getVoiceStatus().catch(() => undefined)
  if (!plan) return undefined
  const state = await getVoiceSettingsState().catch(() => undefined)
  if (!state) return undefined
  return { plan, settings: state.settings, stored: state.stored }
}

const { use: useVoice, provider: VoiceValueProvider } = createSimpleContext({
  name: "Voice",
  init: (props: { room: Room }) => {
    const room = props.room
    const sdk = useSDK()
    const params = useParams()
    const desktop = isDesktop()
    const [error, setError] = createSignal<string | undefined>(undefined)
    const [connecting, setConnecting] = createSignal(false)
    const [nativeState, setNativeState] = createSignal<VoiceState>("disconnected")
    const [nativeAgent, setNativeAgent] = createSignal<AgentState>("disconnected")
    const [nativeMic, setNativeMic] = createSignal(true)
    const [nativeLine, setNativeLine] = createSignal<{ agent: boolean; text: string } | undefined>(undefined)
    const [nativeLevel, setNativeLevel] = createSignal(0)
    const [nativeRun, setNativeRun] = createSignal(false)

    // the provider outlives session navigation; a connected room bridges into
    // the session it was started for, so leaving that session must hang up —
    // and when voice was active, follow the user into the new session's room
    // (each session has its own room and agent, keyed by session id)
    let followGeneration = 0
    createEffect(
      on(
        () => params.id,
        (id, prev) => {
          if (desktop && nativeRun()) {
            if (prev === undefined) return
            const wasActive = state() === "connected" || connecting()
            const generation = ++followGeneration
            void stopVoice()
              .catch(() => {})
              .then(() => {
                setNativeState("disconnected")
                setNativeAgent("disconnected")
                setNativeMic(true)
                setNativeLine(undefined)
                setNativeLevel(0)
                setNativeRun(false)
                if (!wasActive || !id) return
                if (generation !== followGeneration) return
                connect(id, followContext).catch(() => {})
              })
            return
          }
          if (prev === undefined) {
            // first session opened after a (re)mount — resume if voice was
            // active before a project switch
            if (followProjects && id && state() === "disconnected" && !connecting()) {
              connect(id, followContext).catch(() => {})
            }
            return
          }
          const wasActive = state() === "connected" || connecting()
          const generation = ++followGeneration
          void room
            .disconnect()
            .then(() => (desktop ? stopVoice().catch(() => {}) : undefined))
            .then(() => {
              markNativeRuntime(false, false)
              if (!wasActive || !id) return
              if (generation !== followGeneration) return
              connect(id, followContext).catch(() => {})
            })
        },
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

    const [status, { refetch: refetchStatus, mutate: setStatus }] = createResource(
      () => sdk.client,
      (client) =>
        client.voice
          .status()
          .then((x) => x.data)
          .catch(() => undefined),
    )
    const [native, { refetch: refetchNative, mutate: setNative }] = createResource(
      () => desktop,
      (enabled) => loadNativeStatus(enabled),
    )
    const provider = () => voiceProvider(desktop, native(), status())
    const enabled = () => voiceEnabled(desktop, native(), status())
    const nativeActive = () => desktop && (nativeRun() || native()?.plan.runningProvider === "lfm2")

    function runtimeProvider() {
      return native()?.plan.runningProvider ?? provider()
    }

    function markNativeRuntime(running: boolean, mic = running, active = running ? runtimeProvider() : undefined) {
      const current = native()
      if (!current) return
      setNative({ ...current, plan: { ...current.plan, running, runningProvider: active, micEnabled: mic } })
    }

    function clearNativeRuntime() {
      setNativeState("disconnected")
      setNativeAgent("disconnected")
      setNativeMic(true)
      setNativeLine(undefined)
      setNativeLevel(0)
      markNativeRuntime(false, false)
      setNativeRun(false)
    }

    async function stopNativeRuntime(provider?: VoiceProvider) {
      followProjects = false
      followContext = undefined
      if (provider === "livekit") {
        await room.localParticipant.setMicrophoneEnabled(false).catch(() => {})
        await room.disconnect().catch(() => {})
      }
      await stopVoice().catch(() => {})
      setConnecting(false)
      clearNativeRuntime()
    }

    const syncRoomDisconnect = () => {
      if (!desktop) return
      if (native()?.plan.runningProvider !== "livekit") return
      stopVoice().catch(() => {})
      clearNativeRuntime()
      setConnecting(false)
    }
    room.on(RoomEvent.Disconnected, syncRoomDisconnect)
    onCleanup(() => room.off(RoomEvent.Disconnected, syncRoomDisconnect))

    createEffect(() => {
      const current = native()
      if (!desktop || !current) return
      const running = current.plan.runningProvider === "lfm2"
      if (!running) {
        if (!nativeRun()) return
        clearNativeRuntime()
        return
      }
      setNativeRun(true)
      setNativeMic(current.plan.micEnabled)
      if (nativeState() === "disconnected") setNativeState("connected")
      if (nativeAgent() === "disconnected") setNativeAgent("listening")
    })

    async function refreshNative() {
      if (!desktop) return undefined
      const next = await loadNativeStatus(true)
      if (next) setNative(next)
      return next
    }

    function nativeAgentState(state: NativeVoiceState): AgentState {
      if (state === "loading") return "initializing"
      return state
    }

    function handleNative(event: NativeVoiceEvent) {
      switch (event.type) {
        case "state":
          setNativeAgent(nativeAgentState(event.state))
          if (event.state !== "speaking") setNativeLevel(0)
          setNativeState(
            event.state === "loading" ? "connecting" : event.state === "idle" ? "disconnected" : "connected",
          )
          setError(undefined)
          break
        case "transcript":
          setNativeLine({ agent: event.role === "assistant", text: event.text })
          break
        case "level":
          setNativeLevel(Math.min(1, Math.max(0, event.rms * 24)))
          break
        case "ended":
          setNativeState("disconnected")
          setNativeAgent("disconnected")
          setNativeLine(undefined)
          setNativeLevel(0)
          setNativeMic(true)
          markNativeRuntime(false, false)
          setNativeRun(false)
          if (event.reason) setError(event.reason)
          break
        case "error":
          setError(event.message)
          setNativeState("error")
          setNativeAgent("failed")
          setNativeLevel(0)
          markNativeRuntime(false, false)
          setNativeRun(false)
          break
      }
    }

    const state = (): VoiceState => {
      if (error()) return "error"
      if (connecting()) return "connecting"
      if (nativeActive()) return nativeState()
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
      if (nativeActive()) {
        if (state() !== "connected") return "unavailable"
        return nativeMic() ? "unmuted" : "muted"
      }
      if (state() !== "connected") return "unavailable"
      return local.isMicrophoneEnabled() ? "unmuted" : "muted"
    }

    const available = () => {
      if (state() === "connected" || state() === "connecting") return true
      if (desktop && provider() === "lfm2") return native()?.plan.ready === true
      if (desktop && provider() !== "livekit") return false
      return status()?.available === true
    }

    // Voice can be configured in settings while a session is open; poll both the
    // native provider switch and LiveKit availability so the mic button tracks it.
    const statusPoll = setInterval(() => {
      if (desktop) refetchNative()
      if (provider() === "livekit" && !available()) refetchStatus()
    }, 30_000)
    onCleanup(() => clearInterval(statusPoll))
    if (desktop) {
      const refresh = (event: Event) => {
        const current = native()?.plan.runningProvider
        const settings = (event as VoiceSettingsChangedEvent).detail
        refreshNative()
          .then((next) => {
            if (!shouldStopRuntimeForProviderChange(current, settings)) return
            if (!next) return
            stopNativeRuntime(current).catch(() => {})
          })
          .catch(() => {
            if (!shouldStopRuntimeForProviderChange(current, settings)) return
            stopNativeRuntime(current).catch(() => {})
          })
        refetchStatus()
      }
      window.addEventListener(VOICE_SETTINGS_CHANGED, refresh)
      onCleanup(() => window.removeEventListener(VOICE_SETTINGS_CHANGED, refresh))
    }

    async function connect(sessionID: string, ctx?: Omit<VoiceStartContext, "sessionID" | "directory">) {
      if (state() === "connecting" || state() === "connected") return
      setError(undefined)
      setConnecting(true)
      try {
        const current = desktop ? await refreshNative() : undefined
        const server = await (async () => {
          if (desktop && current?.stored !== false) return status()
          const next = await sdk.client.voice
            .status()
            .then((x) => x.data)
            .catch(() => undefined)
          setStatus(next)
          return next
        })()
        const active = voiceProvider(desktop, current, server)
        if (desktop && active === "lfm2") {
          followProjects = true
          if (ctx) followContext = ctx
          const delegateTarget = current?.settings.lfm2.delegate.enabled
            ? current.settings.lfm2.delegate.target
            : undefined
          setNativeRun(true)
          markNativeRuntime(true, true)
          setNativeState("connecting")
          setNativeAgent("connecting")
          setNativeMic(true)
          setNativeLine(undefined)
          setNativeLevel(0)
          const result = await startVoice(
            {
              sessionID,
              directory: sdk.directory,
              delegateTarget,
              ...ctx,
            },
            handleNative,
          )
          if (result.provider !== "lfm2") throw new Error("Voice provider changed while starting.")
          return
        }
        if (desktop && active !== "livekit") {
          throw new Error(current?.plan.detail || "Voice is not set to LiveKit.")
        }
        followProjects = true
        if (ctx) followContext = ctx
        const grant = await (async () => {
          if (!desktop) return sdk.client.voice.token({ sessionID, model: ctx?.model }).then((x) => x.data)
          const result = await startVoice(
            {
              sessionID,
              directory: sdk.directory,
              ...ctx,
            },
            handleNative,
          )
          markNativeRuntime(true, true)
          if (result.provider !== "livekit") throw new Error("Voice provider changed while starting.")
          return result.grant
        })()
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
        followProjects = false
        followContext = undefined
        clearNativeRuntime()
        if (desktop) await stopVoice().catch(() => {})
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
      if (nativeActive()) {
        await stopVoice()
        clearNativeRuntime()
        return
      }
      await room.localParticipant.setMicrophoneEnabled(false).catch(() => {})
      await room.disconnect()
      if (desktop) {
        await stopVoice()
        markNativeRuntime(false, false)
      }
    }

    async function interrupt() {
      if (nativeActive()) {
        await interruptVoice()
        return
      }
      if (state() !== "connected" && state() !== "connecting") return
      followProjects = false
      followContext = undefined
      await room.localParticipant.setMicrophoneEnabled(false).catch(() => {})
      await room.disconnect()
      setError(undefined)
      setConnecting(false)
      if (desktop) await stopVoice().catch(() => {})
      clearNativeRuntime()
    }

    async function setMicEnabled(enabled: boolean) {
      if (nativeActive()) {
        await setVoiceMicEnabled(enabled)
        setNativeMic(enabled)
        markNativeRuntime(true, enabled)
        return
      }
      if (state() !== "connected") return
      await room.localParticipant.setMicrophoneEnabled(enabled)
      if (desktop) {
        await setVoiceMicEnabled(enabled).catch(() => {})
        markNativeRuntime(true, enabled)
      }
    }

    async function toggleMute() {
      if (nativeActive()) {
        const enabled = micState() !== "unmuted"
        await setVoiceMicEnabled(enabled)
        setNativeMic(enabled)
        markNativeRuntime(true, enabled)
        return
      }
      if (state() !== "connected") return
      const enabled = !local.isMicrophoneEnabled()
      await room.localParticipant.setMicrophoneEnabled(enabled)
      if (desktop) {
        await setVoiceMicEnabled(enabled).catch(() => {})
        markNativeRuntime(true, enabled)
      }
    }

    onCleanup(() => {
      if (desktop) stopVoice().catch(() => {})
      room.disconnect()
    })

    return {
      room,
      state,
      micState,
      enabled,
      provider,
      error,
      available,
      connect,
      disconnect,
      interrupt,
      setMicEnabled,
      toggleMute,
      agentState: () => (nativeActive() ? nativeAgent() : assistant.state()),
      agentAudioTrack: () => (nativeActive() ? undefined : assistant.audioTrack()),
      agentLevel: () => (nativeActive() ? nativeLevel() : undefined),
      turnActive: () => {
        const current = nativeActive() ? nativeAgent() : assistant.state()
        return current === "thinking" || current === "speaking"
      },
      transcriptions: () => {
        if (!nativeActive()) return transcriptions()
        const line = nativeLine()
        if (!line) return [] as ReturnType<typeof transcriptions>
        return [
          {
            text: line.text,
            participantInfo: { identity: line.agent ? "agent-native" : "user-native" },
          },
        ] as ReturnType<typeof transcriptions>
      },
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
