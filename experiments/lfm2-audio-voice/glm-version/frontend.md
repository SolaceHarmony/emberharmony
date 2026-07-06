# Frontend voice integration (SolidJS + Tauri)

**Source:** `packages/app/src/context/voice.tsx`, `packages/app/src/lib/voice-settings.ts`,
`packages/app/src/lib/voice-state.ts`, `packages/app/src/lib/voice-state.test.ts`,
`packages/app/src/components/settings-voice.tsx`, `packages/app/src/components/prompt-input.tsx`
· **On the LFM2-Audio inference path:** yes (the UI surface)

> This documents the SolidJS frontend that consumes the Tauri voice commands
> and renders the voice UI — the mic button, bar visualizer, transcript strip,
> and settings panel. The key change: `voice.tsx` was rewritten from a LiveKit
> `Room` wrapper into a provider-branched, event-driven context.

## `voice.tsx` — the voice context (rewritten)

### What it was
A LiveKit `Room` wrapper: created `new Room(...)`, wrapped everything in
`RoomContext`/`RoomAudioRenderer`, used `useVoiceAssistant`/`useTranscriptions`
from `@thesolaceproject/livekit-components-solid`, and called
`sdk.client.voice.token()` → `room.connect()`. Pure LiveKit, no provider
branching.

### What it is now
A provider-branched, event-driven context that supports both `lfm2` (native
Tauri) and `livekit` (the existing LiveKit path):

**`lfm2` path:**
1. `connect()` calls `startVoice(ctx, handleNative)` — a Tauri command that
   returns `VoiceStartResult::Lfm2`.
2. `handleNative` is a callback that receives `NativeVoiceEvent`s from the
   Tauri `Channel`:
   - `State` → `setNativeAgent`/`setNativeState` (drives the mic button +
     visualizer)
   - `Transcript` → `setNativeLine` (drives the transcript strip)
   - `Level` → `setNativeLevel` (drives the `BarVisualizer` — replaces the
     LiveKit `MediaStreamTrack` that doesn't exist in the native path)
   - `Ended` → `clearNativeRuntime()` (reset all signals)
   - `Error` → `setError` + `clearNativeRuntime()`
3. `disconnect()` calls `stopVoice()` (Tauri `voice_stop`) +
   `clearNativeRuntime()`.
4. `setMicEnabled()` calls `setVoiceMicEnabled()` (Tauri
   `voice_set_mic_enabled`) + `setNativeMic`.
5. `interrupt()` calls `stopVoice()` + `clearNativeRuntime()`.

**`livekit` path:**
1. `connect()` calls `startVoice(ctx, handleNative)` — returns
   `VoiceStartResult::Livekit { grant }`.
2. Uses `grant.token`/`grant.url` to `room.connect()` (the existing LiveKit
   path).
3. `RoomContext`/`RoomAudioRenderer`/`useVoiceAssistant`/`useTranscriptions`
   drive the same signals (`nativeAgent`, `nativeState`, `nativeLevel`,
   `nativeLine`).
4. `disconnect()` calls `room.disconnect()` + `stopVoice()` +
   `markNativeRuntime(false, false)`.

**The `Room` is still created unconditionally** — needed for the `livekit`
path. For `lfm2` it's unused (no LiveKit connection). This means the LiveKit
client library is loaded even when the user only uses the local model. A
future cleanup could lazy-load the LiveKit client only when the provider is
`livekit`.

**Session navigation:** the `createEffect(on(params.id))` that follows the
user across sessions now also calls `stopVoice()` on the native path before
reconnecting (matching the LiveKit `room.disconnect().then(reconnect)`
pattern).

**Settings change listener:** `window.addEventListener(VOICE_SETTINGS_CHANGED,
refresh)` — on settings change, calls `refreshNative()` + checks
`shouldStopRuntimeForProviderChange()`. If the provider changed while the
runtime is running, calls `stopNativeRuntime()` (stops the old provider's
session).

## `voice-settings.ts` — Tauri command wrappers

Typed wrappers over the Tauri commands:
- `getVoiceSettings()` → `invoke("voice_settings_get")`
- `setVoiceSettings(settings)` → `invoke("voice_settings_set", { settings })`
  + dispatches `VOICE_SETTINGS_CHANGED` event with the new settings
- `getVoiceStatus()` → `invoke("voice_status")` — returns `VoicePlan` with
  `running`/`runningProvider`/`micEnabled` runtime fields
- `startVoice(ctx, onEvent)` → creates a `Channel`, calls
  `invoke("voice_start", { ctx, channel })`, returns `VoiceStartResult`
- `stopVoice()` → `invoke("voice_stop")`
- `setVoiceMicEnabled(on)` → `invoke("voice_set_mic_enabled", { on })`

All guarded by `tauriInvoke()` — in the web build (no Tauri runtime), these
no-op to defaults.

## `voice-state.ts` — pure decision functions (NEW, extracted, tested)

The provider/enabled/button/mic decision logic was extracted from `voice.tsx`
into pure functions so they're testable without rendering components:

| Function | Purpose |
|---|---|
| `voiceProvider(desktop, native, server)` | Which provider is active: desktop → Tauri `plan.provider`; web → `"livekit"`; migration fallback for unset native settings |
| `voiceEnabled(desktop, native, server)` | Whether the mic button should be visible |
| `voiceButtonOn(state, enabled)` | Whether the mic button is on (connected OR enabled) |
| `voiceMicTarget(state, dirty, busy)` | Whether the mic should be on: connected && !typing && !working → `true`; else `undefined` (no change) |
| `shouldStopRuntimeForProviderChange(running, next)` | Whether a provider change requires stopping the running runtime |

6 unit tests in `voice-state.test.ts` cover all decision paths.

## `settings-voice.tsx` — the settings panel

- **Provider selector** — Off / Local (LFM2-Audio) / LiveKit.
- **LFM2 settings:**
  - "Hugging Face model" field — the repo id (default
    `LiquidAI/LFM2.5-Audio-1.5B`), used for HF cache lookup + first-run
    download.
  - "Local snapshot directory" field — optional path to a local model
    directory (overrides the HF download).
  - Compute device (CPU / Metal), VAD threshold, max tokens, seed, delegate
    config.
- **LiveKit settings:** URL, STT/TTS model strings, intent model.
- The P2a "Off not persisted" bug (the migration fallback treating `"off"` as
  unset) is a known issue — the `provider()` function in `settings-voice.tsx`
  skips `"off"` and falls through to `"livekit"` for existing LiveKit users.

## `prompt-input.tsx` — the mic button + typing interaction

- **Mic button visibility:** `voiceButtonOn(voice.state(), voice.enabled())` —
  shows the mic when connected OR when voice is enabled.
- **Mic-pause-on-typing:** `voiceMicTarget(voice.state(), prompt.dirty(),
  working())` — when connected and the user is typing (prompt is dirty) or a
  generation is running (working), the mic is paused via
  `voice.setMicEnabled(false)`. Resumed when idle.
- **BarVisualizer:** `state={voice.agentState()}` + `track={voice.agentAudioTrack()}`
  — for the native path, `agentAudioTrack()` returns `undefined` (no
  LiveKit track); the visualizer should read `voice.level()` instead. This is
  a known gap — the `Level` event is emitted by the Rust side but the
  `BarVisualizer` hasn't been switched to consume it yet.
- **Stop button:** `abort()` calls `sdk.client.session.abort()` — does NOT
  call `voice_stop`. The voice session keeps running after the stop button.
  This is a known gap — the stop button should also call `voice.interrupt()`
  or `voice_stop` to stop TTS + generation.

## Known gaps

| Gap | What | Fix |
|---|---|---|
| `BarVisualizer` reads `agentAudioTrack()` | No LiveKit track in the native path | Switch to `voice.level()` from `VoiceEvent::Level` |
| Stop button doesn't stop voice | `abort()` only calls `sdk.client.session.abort()` | Also call `voice.interrupt()` or `voice_stop` |
| `Room` created unconditionally | LiveKit client loaded even for `lfm2`-only users | Lazy-load the LiveKit client when provider is `livekit` |
| "Off" not persisted (P2a) | `provider()` skips `"off"` in the migration fallback | Respect any explicit stored value; migrate only when truly unset |
| `voice_status` reports `ready: true` for LFM2 even when model not downloaded | The actual load/download happens in `voice_start` | Consider a background model-load check or a "downloading" state |

## Cross-references

- [`tauri-voice.md`](tauri-voice.md) — the Tauri commands these wrappers call.
- [`AS_BUILT_claude_changes.md`](AS_BUILT_claude_changes.md) §7 — the as-built
  record of the frontend rewrite.
- `packages/desktop/src-tauri/src/voice/FRONTEND_DESIGN.md` — the design doc
  (turn mode + live mode, one event-driven core).