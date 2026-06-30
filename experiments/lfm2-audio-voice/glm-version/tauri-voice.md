# Tauri voice service (`voice/control.rs` + `voice/runtime.rs` + `settings.rs`)

**Source:** `packages/desktop/src-tauri/src/voice/control.rs`, `runtime.rs`, `session.rs`, `mod.rs`
· **Settings:** `packages/desktop/src-tauri/src/settings.rs`
· **On the LFM2-Audio inference path:** yes (the command surface)

> This documents the Tauri-side voice service — the in-process command layer
> that connects the `liquid-audio` crate's `RealtimePipeline` to the SolidJS
> frontend via Tauri Channels. No HTTP for the LFM2 path; the only HTTP is
> LiveKit token minting (the credentials live on the server side).

## Role

The Tauri voice service is the **in-process integration layer** between the
`liquid-audio` crate (the model + `RealtimePipeline` + cpal I/O) and the
SolidJS frontend (the mic button, visualizer, transcript). It exposes Tauri
commands that the webview invokes, streams `VoiceEvent`s over a
`tauri::ipc::Channel`, and manages the session lifecycle (start/stop/mic) via
`tauri::State<VoiceRuntime>`.

## Architecture

```
Webview (SolidJS)                 Tauri process (tokio)              Worker thread
                                  VoiceRuntime (State)              "lfm2-inference"

invoke("voice_start", {ctx,ch})   → plan() readiness check
                                  → VoiceRuntime::start_lfm2()
                                    → build_engine() (loads model)
                                    → VoiceRuntime (liquid-audio) spawns:
                                      → RealtimePipeline::spawn(Lfm2VoiceEngine)  ──→  owns model+proc
                                      → cpal mic thread (VAD → submit Utterance)  ──→  recv Utterance
                                      → cpal output (ring buffer playback)            → respond (gen+decode)
                                      → crossbeam → Tauri Channel bridge task         → emit VoiceEvent
                                    → return VoiceStartResult::Lfm2
                                                                                  → TurnComplete/Interrupted

invoke("voice_stop")              → VoiceRuntime::stop()
                                    → pipeline.interrupt() (AtomicBool)           → cancel breaks gen loop
                                    → stop cpal streams                           → worker joins
                                    → drop session

invoke("voice_set_mic_enabled")   → session.set_mic_enabled(bool)
                                    → cpal mic pause/resume (AtomicBool)

invoke("voice_status")            → plan() + runtime.active_provider()
                                    → VoicePlan { provider, running, micEnabled, ... }
```

## Commands

### `voice_start(app, server, ctx, channel) -> Result<VoiceStartResult, String>`

Returns a discriminated union:
- `VoiceStartResult::Lfm2` — the native LFM2 pipeline is running; events stream
  over the `channel`.
- `VoiceStartResult::Livekit { grant: LiveKitGrant }` — the LiveKit token was
  minted; the webview uses `grant.token`/`grant.url` to `room.connect()`.

For `lfm2`: calls `VoiceRuntime::start_lfm2()` which builds `Lfm2VoiceEngine`
(loads model + processor via `from_pretrained`), spawns the
`RealtimePipeline` worker thread, starts cpal capture (VAD), and bridges
crossbeam `VoiceEvent`s → the Tauri `Channel`.

For `livekit`: calls `VoiceRuntime::start_livekit()` (registers the session),
then mints a LiveKit token via HTTP to the local sidecar
(`livekit_grant()` at `control.rs:289`). If the token mint fails, the runtime
is stopped and the error returned.

### `voice_stop() -> Result<(), String>`

Calls `VoiceRuntime::stop()` — interrupts the pipeline (sets the `AtomicBool`
cancel flag), stops cpal streams, drops the session. The worker thread's
generate loop checks `cancel` at the top of every decode step and breaks; the
crossbeam channel closes; the bridge task ends. Non-blocking from the
command's perspective.

### `voice_status(app, runtime) -> Result<VoicePlan, String>`

Returns `VoicePlan` with both settings-derived fields (`provider`, `enabled`,
`surface`, `ready`, `detail`) and runtime-derived fields (`running`,
`running_provider`, `mic_enabled`) read from `VoiceRuntime`.

### `voice_set_mic_enabled(on: bool) -> Result<(), String>`

Pauses/resumes the cpal mic capture via `AtomicBool`. Does not stop the
session — the model stays loaded, the pipeline stays alive. Used for
mic-pause-on-typing and the mic toggle button.

## `VoiceEvent` contract

Streamed over `tauri::ipc::Channel<VoiceEvent>` (ordered, high-throughput):

| Event | Fields | When |
|---|---|---|
| `State` | `state: Idle\|Listening\|Thinking\|Speaking` | session state changed |
| `Transcript` | `role: User\|Assistant`, `text: String` | reply text (cumulative) |
| `Level` | `rms: f32` | audio amplitude for the bar visualizer |
| `AudioClip` | `wav: Vec<u8>`, `ms: u32` | decoded reply clip (turn mode → `<audio>` player) |
| `Ended` | `reason: Option<String>` | session ended |
| `Error` | `message: String` | error occurred |

The engine-side `VoiceEvent` (`realtime.rs`: `Text`/`Audio`/`TurnComplete`/
`Interrupted`/`Error`) is mapped to the Tauri-side `VoiceEvent` by the
bridge task.

## `VoiceRuntime` (desktop `runtime.rs`)

Manages the session lifecycle via `tauri::State`:

- `VoiceSession` enum — `Lfm2(Lfm2Session)` / `Livekit(LiveKitSession)`.
  Dispatches `is_finished`/`provider`/`session_id`/`interrupt`/
  `set_mic_enabled`/`mic_enabled`/`stop` per variant.
- `ThreadManager` — `Vec<JoinHandle>` with `reap()` (joins finished threads),
  `wait()` (joins all before starting a new session), `Drop` (joins all on
  shutdown). No detached threads.
- `Lfm2Session` — owns the `Lfm2Runtime` (from `voice_runtime.rs` in the
  liquid-audio crate), the bridge cancel `AtomicBool`, and the done flag.
- `LiveKitSession` — lightweight state tracker (ctx + mic `AtomicBool`). The
  actual LiveKit room lives in the webview; `stop()` is a no-op.

## `settings.rs`

- `VoiceProvider` — `Off`/`Lfm2`/`Livekit`.
- `Lfm2Settings` — `model_dir` (optional local snapshot), `model` (HF repo id,
  default `"LiquidAI/LFM2.5-Audio-1.5B"`), `device` (`Cpu`/`Metal`),
  `vad_threshold`, `max_tokens`, `seed`, `delegate`.
- `lfm2_model_ref()` — resolves: explicit `model` → `model_dir` (if
  `config.json` exists) → `DEFAULT_LFM2_MODEL`. Default is a downloadable HF
  repo id.
- `VoicePlan` readiness for `Lfm2` always returns `ready: true` (the model
  can download from HF on first start).

## `session.rs`

The HTTP session bridge — used **only** for the LiveKit provider's delegate
path. An SSE reducer that drives `POST /session/:id/prompt_async` +
`GET /event` for the LiveKit provider when delegation is configured. The LFM2
path is fully in-process and does not use this module.

## Known issues

- **`livekit_grant` uses HTTP** to the local sidecar for token minting. The
  LFM2 path is fully in-process; the LiveKit path still goes through HTTP
  because the LiveKit credentials live on the server side.
- **`voice_status` reports `ready: true` for LFM2 even when the model isn't
  downloaded.** The actual load/download happens in `voice_start` →
  `build_engine` → `from_pretrained`.
- **`voice.tsx` still creates a `Room` unconditionally** — the LiveKit client
  library is loaded even when the user only uses the local model.

## Cross-references

- [`AS_BUILT_claude_changes.md`](AS_BUILT_claude_changes.md) §7 — the full
  as-built record of this integration.
- [`voice_runtime.md`](voice_runtime.md) — the `voice_runtime.rs` module
  (cpal VAD + playback) inside the `liquid-audio` crate.
- [`frontend.md`](frontend.md) — the SolidJS frontend that consumes these
  commands.
- `packages/desktop/src-tauri/src/voice/FRONTEND_DESIGN.md` — the design doc.