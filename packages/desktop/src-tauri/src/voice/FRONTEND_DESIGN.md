# Voice Frontend Design

> **Current production document.** This records the shipped frontend seam. The
> normative native ticket observer and truthful capture/compute/playback meter
> design is
> [`specs/11-kcoro-native-migration/12-ticketed-orchestration-and-observability.md`](../../../../../specs/11-kcoro-native-migration/12-ticketed-orchestration-and-observability.md).
> It supplies the detailed observer contract; no legacy copy is retained.

This document tracks the SolidJS side of the native desktop voice kernel described in
`desktop-stack.md`. The key rule is simple: SolidJS is intent and rendering only. The Tauri
process owns provider readiness, microphone truth, LiveKit rooms, LFM2 model lifecycle, stop
semantics, and audio buffering.

## Provider Ownership

Desktop voice has two native providers behind the same Tauri command surface:

- `lfm2`: local LFM2-Audio, running inside the desktop process.
- `livekit`: native LiveKit/WebRTC room plus an in-process LFM2 agent, also inside the desktop
  process.

The browser LiveKit `Room` path remains only for non-desktop web builds. In the desktop shell,
`context/voice.tsx` must return the Tauri voice client before constructing `new Room`, rendering
`RoomAudioRenderer`, or calling `sdk.client.voice.status()` / `sdk.client.voice.token()`.

## Tauri Command Surface

The desktop command surface is intentionally small and provider-agnostic:

```text
voice_status(app, runtime)
  -> VoicePlan { provider, enabled, surface, running, runningProvider, micEnabled, ready, detail }

voice_start(app, runtime, server, ctx, channel)
  -> VoiceStartResult::{ lfm2 | livekit }

voice_stop(runtime)
  -> stop and join the active native VoiceSession

voice_interrupt(runtime)
  -> interrupt the current reply without disconnecting the session

voice_set_mic_enabled(runtime, enabled)
  -> pause/resume native microphone capture

voice_begin_typed_input(runtime)
  -> atomically pause native microphone capture and interrupt the active reply

voice_settings_get / voice_settings_state / voice_settings_set
  -> read/write provider settings through Tauri; runtime reconciliation happens in Rust

voice_livekit_credentials_set / voice_livekit_credentials_status
  -> store/read LiveKit API credentials in the OS keychain
```

`SessionCtx` carries session identity and ordinary prompt context:

```text
SessionCtx { sessionID, directory, model?, agent?, variant?, promptMode? }
```

It does not carry `delegateTarget`. Delegation target is native settings, read by Rust from
`settings.lfm2.delegate`, and the desktop kernel defaults delegated execution to the safe planning
agent unless and until a native classifier exists.

## VoiceEvent Contract

Both providers emit the same event stream over a Tauri `Channel<VoiceEvent>`:

```text
State { loading | idle | listening | thinking | speaking }
Transcript { role, text }
Level { rms }
Ended { reason? }
Error { message }
```

For continuous native sessions, PCM stays in the native capture/playback docks; Rust owns only the
platform callback endpoints. The webview gets scalar `Level` updates for the meter, transcripts for
display, and state transitions for controls. PCM never crosses webview IPC.

## SolidJS Voice Context

The desktop context state is derived from Tauri settings, `voice_status`, and streamed
`VoiceEvent`s:

```text
provider       off | lfm2 | livekit
enabled        whether the native provider is turned on
available      native readiness from voice_status().ready
state          disconnected | connecting | connected | error
micState       unavailable | muted | unmuted
agentState     disconnected | initializing | listening | thinking | speaking | failed
agentLevel     scalar RMS for NativeVoiceMeter
transcriptions cumulative native transcript rows
```

The actions map directly to Tauri commands:

```text
connect(sessionID, ctx)   -> voice_start(ctx, channel)
disconnect()              -> voice_stop()
interrupt()               -> voice_interrupt()
beginTypedInput()         -> voice_begin_typed_input()
setMicEnabled(enabled)    -> voice_set_mic_enabled(enabled)
toggleMute()              -> setMicEnabled(!current)
```

Provider switching, model directory changes, device changes, LiveKit URL changes, and keychain
credential writes are reconciled in Rust. Sesame speech policy is native and has no frontend
threshold control. The frontend dispatches the
settings event so UI state refetches, but it must not decide which native session to stop.

## Prompt Input Integration

The voice button is an affordance for the selected native provider. It remains visible and
pressed when voice is enabled even if the provider is not ready yet; clicking it attempts
`voice_start`, and the native readiness detail explains why startup failed.

The stop button has two native meanings:

- while voice is speaking/thinking: call `voice_interrupt()` and keep the session alive;
- when the voice toggle is turned off: call `voice_stop()` and leave voice mode.

Typed input is a kernel event, not a UI-only mute. When the user begins typing or prompt execution
is active, `prompt-input.tsx` calls `voice_begin_typed_input()` so Rust pauses capture and
interrupts the active response before the typed prompt proceeds. When the prompt is clean and the
app is idle again, the UI may call `voice_set_mic_enabled(true)` to resume capture.

The visualizer branches by source:

- web fallback: `BarVisualizer` samples a browser LiveKit `MediaStreamTrack`;
- desktop native: `NativeVoiceMeter` renders `VoiceEvent::Level` because no audio track enters the
  webview.

## Regression Guards

The current tests intentionally guard these seams:

- desktop `voice.tsx` does not construct or connect a browser LiveKit room;
- desktop start context does not contain `delegateTarget`;
- desktop availability comes from Tauri `voice_status`, not server LiveKit status;
- stop, interrupt, mic, typed input, settings, and provider invalidation go through the bounded
  Tauri runtime queue;
- LFM2 and LiveKit both hang off `VoiceRuntime` as native `VoiceSession` implementations;
- prompt input pauses native voice through `voice_begin_typed_input` before typed work proceeds.
