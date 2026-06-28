# Voice frontend design — turn mode + live mode (one event-driven core)

## Phasing (the directive)

- **Phase 1 — NOW: absolute parity with Liquid AI's WebGPU demo.** Turn-based, multi-modal
  (ASR / TTS / Interleaved), clip-based. Prove we can do exactly what Liquid AI did, with our
  native Rust `liquid-audio` engine instead of transformers.js/ONNX. This is the whole
  near-term scope; everything below under "turn mode" is Phase 1.
- **Phase 2 — LATER: natural, full-duplex conversation.** The end state is *not* turn-based —
  speech should be continuous and interruptible. The right model for that is **Moshi (the LM),
  not just Mimi (the codec)** — Moshi is architecturally full-duplex (it processes input and
  output streams every frame), so it beats LFM2 + a hand-rolled VAD loop (`duplex_chat.rs`,
  which is a stopgap, not the destination). The `moshi` crate already gives us Mimi; Phase 2
  brings in its LM. "Live mode" below is the Phase-2 shape.

The point of designing both now is that the **event-driven core is identical** — Phase 2 is
*additive* (a new trigger + a different model behind the same `VoiceEvent` session), not a
rewrite. Build Phase 1 to parity; do not paint Phase 2 into a corner.

---

Design for the native (Tauri) voice frontend after moving off LiveKit. It supports **both**
reference UX models on one core, phased as above:

- **Turn mode** — Liquid AI's WebGPU demo (`spaces/LiquidAI/LFM2.5-Audio-1.5B-transformers-js`,
  `main.js`): record/upload a clip *or* type → **Send** → streamed text + an audio reply.
  Explicit modes **ASR / TTS / Interleaved**. One turn at a time (`isGenerating` guard).
  Audio reply is decoded after the turn and shown as an inline `<audio>` player.
- **Live mode** — `liquid_audio/demo/chat.py` (`ReplyOnPause`, `can_interrupt=False`) and our
  `liquid-audio` `RealtimePipeline` / `duplex_chat.rs`: continuous mic, VAD-detected
  utterances, streaming playback, barge-in. Hands-free.

## The unifying primitive

Both are the **same turn**: `(mode, audio?, text?) → stream(text tokens, audio frames) → done`.
The only differences:

| | trigger | audio out | concurrency |
|---|---|---|---|
| **turn** | user presses Send | clip → webview `<audio>` player | one turn, then idle |
| **live** | VAD onset/pause (in Rust) | streamed → cpal speaker (in Rust) | continuous; barge-in |

So the frontend models one **event-driven session** that emits `VoiceEvent`s; `voiceMode ∈
{off, turn, live}` selects trigger + playback. The Rust `RealtimePipeline.submit(Utterance)` →
event-drain is exactly the shared turn; `duplex_chat.rs`'s continuous-VAD wrapper is *only* the
live trigger, not a different engine.

## Tauri command surface

`voice_settings_get` / `voice_settings_set` / `voice_status` already exist; `voice_status`
must report **runtime** readiness (model/device for `lfm2`; sidecar reachability for
`livekit`), not just "provider says ready" (GPT/GLM).

```
voice_status() -> VoicePlan { provider, ready, detail }          # branch by provider

# turn mode — one turn, streams VoiceEvents, ends with State::Idle
voice_generate_turn(req: TurnRequest, channel: Channel<VoiceEvent>) -> Result<()>
  TurnRequest { mode: asr|tts|interleaved, audio?: { pcm: bytes, rate }, text?: string, ctx: SessionCtx }

# live mode — continuous session, streams VoiceEvents until stopped
voice_start_live(ctx: SessionCtx, channel: Channel<VoiceEvent>) -> Result<()>
voice_stop_live() -> Result<()>

# shared turn control (both modes)
voice_abort_turn() -> Result<()>          # cancel the in-flight generation (the
                                          # generate_interleaved_cancellable AtomicBool)
voice_set_mic_enabled(on: bool) -> Result<()>   # live only: pause/resume STT capture
```

`SessionCtx { sessionID, directory, model, agent, delegateTarget?, promptMode: plan|build }`
— GPT's point that voice must carry session context, not just a channel. `voice_start`'s
current `(app, channel)` stub becomes these two typed entrypoints; `voice_stop`'s no-op
becomes `voice_stop_live` + `voice_abort_turn`.

## `VoiceEvent` contract (revised)

Current (`control.rs:94-103`): `State{Idle|Listening|Thinking|Speaking}`, `Transcript{role,text}`,
`Ended{reason}`, `Error{message}`. Revisions to cover both modes + the gap reading the UI
surfaced:

```
enum VoiceEvent {
  State { state }                         # idle|listening|thinking|speaking
  Transcript { role, text }               # text is CUMULATIVE (demo streams cumulative)
  Level { rms: f32 }                      # NEW — amplitude for the visualizer (see below)
  AudioClip { wav: bytes, ms: u32 }       # NEW — turn mode: the decoded reply as a player clip
  Ended { reason: Option<String> }
  Error { message }
}
```

**Why `Level` is required, not optional:** the `BarVisualizer` reads
`track={voice.agentAudioTrack()}` (`prompt-input.tsx:2105`) — a LiveKit `MediaStreamTrack` it
samples to draw bars. In the native path there is **no track in the webview** (audio is cpal
in Rust, live mode; or a decoded blob, turn mode). GLM's "components already exist, just
populate the signals" is wrong here: `agentAudioTrack` has no native source. The Rust loop
emits `Level{rms}` from the PCM it is playing; the visualizer draws from that scalar.

**Audio out, per mode:**
- **live**: cpal plays in Rust; webview gets only `Level` (+ `Transcript`/`State`). No PCM
  crosses the IPC boundary — the whole point of in-process audio.
- **turn**: the reply is decoded to a clip and sent as `AudioClip{wav}` at `TurnComplete`,
  rendered as an `<audio>` player in the assistant message (exactly the demo, `main.js:464-485`).

## SolidJS voice context (`context/voice.tsx`, rewritten)

`voice.tsx` is **gutted** from a LiveKit `Room` wrapper (it is one end-to-end: `Room` ctor,
`RoomContext`/`RoomAudioRenderer`, `connectionState`, `useVoiceAssistant`, `useTranscriptions`,
`token()`→`room.connect()`) into an event-driven, provider/mode-branched context:

```ts
// store
provider   // off | lfm2 | livekit          (from voice_settings_get)
ready      // bool                          (from voice_status)
mode       // off | turn | live             (interaction model)
state      // idle | listening | thinking | speaking | error   (VoiceEvent::State)
mic        // muted | unmuted | unavailable (voice_set_mic_enabled, live only)
transcript // string                        (VoiceEvent::Transcript, cumulative)
level      // number                        (VoiceEvent::Level — drives BarVisualizer)
error      // string | undefined

// actions
submitTurn({ audio?, text?, mode })  // turn: voice_generate_turn + listen on channel
startLive() / stopLive()             // live: voice_start_live / voice_stop_live
abortTurn()                          // voice_abort_turn (stop button + typed barge-in)
setMicEnabled(on)                    // live: voice_set_mic_enabled
```

`lfm2` → these Tauri commands; `livekit` → the existing `Room` path kept behind the same
surface (legacy, until removed). `available()` = `voice_status().ready`, branched by provider
— not `sdk.client.voice.status()` for everything. The `BarVisualizer` call site changes from
`track={agentAudioTrack()}` to `level={voice.level()}`; `transcript`/`state`/mic-button
consumers keep their shape but read the new signals.

## Prompt-input integration

**Turn mode (default — models the demo onto the existing prompt):**
- Audio is a **turn-input attachment**, like an image: a record/upload control adds an audio
  clip to the prompt. Mode is inferred from the turn (audio+interleaved = conversation;
  text-only+tts = speech; audio+asr = transcribe) or set by a small selector.
- `handleSubmit` (`prompt-input.tsx:1170`): when the turn has audio (or voice mode is on),
  route through `submitTurn(...)`; stream `Transcript` into the assistant message; render
  `AudioClip` as an inline `<audio>` player.
- **Stop**: `abort()` (`:942-957`) additionally calls `voice.abortTurn()` — cancels the
  in-flight generation. Returns to `idle`.
- **Typing**: no conflict — audio is an attachment composed into one turn; text and audio are
  the *same* turn, never racing. (This is why the demo model dissolves the live-mode typing
  problem.)

**Live mode (hands-free toggle):**
- The mic button enters live mode: `startLive()`. `BarVisualizer` shows `level`; the transcript
  strip shows `Transcript`; audio plays via cpal in Rust. Barge-in is handled in Rust
  (`RealtimePipeline.interrupt()` on VAD onset).
- **Stop button** = `abortTurn()` (stop *this* turn, stay live → `listening`). The mic/live
  toggle off = `stopLive()` (leave live mode).
- **Typing while live**: a typed submit is a barge-in — `abortTurn()` first, then submit the
  text; and while the input is focused-and-non-empty, `setMicEnabled(false)` so typing doesn't
  trip a voice turn. Resume on idle. (GPT's rule, scoped to live mode where it actually applies.)

## Reconciliation — GLM & GPT 5.5

| Their proposal | Verdict | Note |
|---|---|---|
| GPT: `voice_start` carries session context | **keep** | `SessionCtx` on both `voice_generate_turn` + `voice_start_live`. |
| GPT/GLM: rewrite `voice.tsx` to Tauri events | **keep** | It's a full gut of a `Room` wrapper, not a patch. |
| GPT: `voice_status` = runtime readiness | **keep** | branch by provider; `livekit` checks sidecar. |
| GPT: add `voice_abort_turn` / `voice_stop({turn})` | **keep** | one `voice_abort_turn`; the cancel hook already exists. |
| GPT: `voice_set_mic_enabled` / pause listening | **keep, live-only** | N/A in turn mode (no continuous mic). |
| GPT/GLM: stream state/transcript/partial/error/ended | **keep + extend** | add **`Level`** (visualizer) and **`AudioClip`** (turn-mode player). |
| GPT: stop = barge-in (abort + TTS stop → listening) | **revise per mode** | turn → idle; live → listening. |
| GPT: typing = barge-in + pause STT | **keep, live-only** | turn mode has no race to resolve. |
| GLM: "components exist, just populate signals" | **revise** | `agentAudioTrack` has **no** native source → `Level` is mandatory. |
| Both: only designed the **live** model | **the gap this fills** | their design = "live mode"; the demo = "turn mode"; both share the event-driven session + `voice_abort_turn`. |

## Build order
1. Extend `VoiceEvent` (`Level`, `AudioClip`) + add the commands (`voice_generate_turn`,
   `voice_start_live`/`stop_live`, `voice_abort_turn`, `voice_set_mic_enabled`).
2. Wire `voice_generate_turn[lfm2]` → `RealtimePipeline.submit` (turn mode is the smaller,
   testable first slice; live mode adds the VAD trigger on top).
3. Rewrite `voice.tsx` as the event-driven context; switch `BarVisualizer` to `level`.
4. Prompt-input: audio attachment + mode + audio-player rendering (turn); live toggle + barge-in.
5. Stop button + typed-submit → `abortTurn()` (both), `setMicEnabled` (live).

## RESOLVED decisions (Phase-1 Tauri wiring) — settle these; stop re-litigating

Grounded in the `liquid-audio` examples (`duplex_chat.rs` = the reference consumer + VAD loop)
and `realtime.rs` (`RealtimePipeline` owns the engine on a worker thread; `Channel::send` is
`Send+Sync`).

1. **Model lifecycle → eager-async on `provider == lfm2`, NOT app-start, NOT first-press.**
   A 1.5B model is ~3 GB RAM + ~21 s CPU (faster on Metal). Don't pay it at startup (most users
   won't use local voice) and don't pay it as a 21 s mic-button hang. Load on a background
   thread the moment the active provider becomes `lfm2` (at boot if settings already say so, or
   on the settings switch). `voice_status` gains `loading | ready` so the UI shows a spinner;
   the mic enables on `ready`. "Lazy w.r.t. intent, eager w.r.t. first use."

2. **Pipeline storage → ONE persistent `RealtimePipeline` in `State<Mutex<Option<…>>>`; the
   warm model lives inside it; never reload per session.** `model`+`proc` are owned by
   `Lfm2VoiceEngine`, owned by the pipeline's worker thread — so the pipeline-in-State *is* the
   warm-model store (you can't also keep an un-`Clone` copy in State). Created once when the load
   finishes; reused for every turn (it already persists across turns — `worker_persists_across_turns`
   — and now accumulates `conv` across turns); dropped only on provider-change / app-exit.

3. **crossbeam → Tauri `Channel` bridge → a plain `std::thread`, `for ev in rx.iter() {
   channel.send(map(ev)) }`. No tokio, no `try_recv` timer.** `Channel::send` is `Send+Sync`
   (tauri `OnMessageFn = Box<dyn Fn(..)+Send+Sync>`), callable from any OS thread, so no runtime
   is needed; blocking `rx.iter()` is lowest-latency (a timer poll adds latency + busy-loops).
   This IS the `duplex_chat` consumer thread with `ring.extend(pcm)` swapped for `channel.send`.
   ONE bridge per session, draining `pipe.events().clone()`, ending on the turn/session terminal
   event — there is only ever one active session (the demo's `isGenerating` guard), so no
   competing consumers steal events from the shared pipeline.

4. **Event mapping — the naïve `realtime::VoiceEvent → control::VoiceEvent` table has 4 gaps:**
   - `Text(frag)`: realtime emits per-token FRAGMENTS; `control::Transcript` is CUMULATIVE.
     The bridge ACCUMULATES a per-turn `String` and emits `Transcript{Assistant, cumulative}`.
   - `Audio(pcm)`: must be PLAYED, not just measured. **Live** → push to a cpal output ring
     (Rust-side) + emit `Level{rms}` (no PCM crosses IPC). **Turn** → ACCUMULATE pcm; at
     `TurnComplete` encode → `AudioClip{wav}` for the webview `<audio>` (no cpal in turn mode).
   - `TurnComplete`: turn → emit `AudioClip{wav}` THEN `State{Idle}`; live → `State{Listening}`.
   - `Interrupted`: turn → `State{Idle}`; live → `State{Listening}`.
   - SYNTHESIZE the states realtime doesn't emit: `State{Thinking}` on submit (before first
     token), `State{Speaking}` on the first `Audio`. (Listening→Thinking→Speaking, like the demo.)

5. **Phase 1 = `voice_generate_turn` (one-shot) BUILT ON the persistent pipeline — not a
   separate path, and NOT continuous live yet.** Using the pipeline keeps the model warm (#2),
   gives abort/barge-in for free, and makes Phase 2 *additive* (live just adds the cpal VAD
   trigger + streaming playback on the SAME engine). **Turn mode needs NO cpal:** audio-in is a
   webview-recorded PCM clip (`TurnRequest{audio:{pcm,rate}}`), audio-out is the accumulated
   `AudioClip{wav}` the webview plays — exactly the Liquid demo. Implement `voice_generate_turn`
   + `voice_abort_turn`(→`pipe.interrupt()`) now; `voice_start_live`/`voice_stop_live` (cpal +
   VAD) are Phase 2 stubs.
