# `voice_runtime.rs` — cpal VAD + playback runtime (liquid-audio crate)

**Source:** `packages/desktop/src-tauri/crates/liquid-audio/src/voice_runtime.rs` (598 lines)
· **On the LFM2-Audio inference path:** yes (the in-process voice loop)

> This documents the `VoiceRuntime` in the `liquid-audio` crate — the cpal
> mic capture (energy VAD) + speaker playback runtime that wraps
> `RealtimePipeline` into a managed service. This is the layer that makes the
> voice loop fully in-process (no HTTP, no subprocess, no LiveKit).

## Role

`VoiceRuntime` is the **real-time I/O layer** between the cpal audio hardware
and the `RealtimePipeline`'s inference worker. It owns:
- The cpal input stream (mic capture with energy VAD)
- The cpal output stream (ring-buffer playback)
- The `RealtimePipeline` (the inference worker thread)
- The `mic_enabled` `AtomicBool` (pause/resume without killing the stream)

It sits inside the `liquid-audio` crate (not the desktop crate) because it's
reusable — any Rust binary that wants the full voice loop (the `duplex_chat`
example, a future standalone server) can use it. The desktop crate's
`voice/runtime.rs` wraps it in a `VoiceSession` with Tauri-specific lifecycle
management.

## How it works

### `RuntimeConfig`
- `vad_threshold: f32` (default 0.012) — RMS threshold for speech onset.
- `silence_ms: u64` (default 800) — pause duration to end an utterance.
- `min_utterance_s: f32` (default 0.3) — minimum utterance duration (filters
  clicks/noise).
- `can_interrupt: bool` (default false) — whether the user can barge in while
  the assistant is speaking. `false` matches `chat.py`'s
  `ReplyOnPause(can_interrupt=False)` — mic input is dropped while the
  assistant speaks (prevents the speaker output from re-triggering VAD).

### The VAD loop (`vad_loop`)

1. **Mic capture** — cpal input callback pushes mono f32 samples into a
   `Mutex<Vec<f32>>` buffer. The callback does downmix-to-mono inline (no
   separate `downmix` function — each frame is averaged across channels in the
   callback itself).

2. **Energy VAD** — the loop reads the buffer in chunks, computes RMS, and
   detects speech onset (RMS > `vad_threshold`) and end-of-utterance (silence
   for `silence_ms`).

3. **`can_interrupt` gate** — when the assistant is speaking and
   `can_interrupt` is false, mic input is cleared and skipped (no utterance
   submitted). This prevents acoustic echo from triggering false turns.

4. **Utterance submission** — on end-of-utterance (silence detected after
   speech), the captured samples are submitted to the pipeline:
   `pipeline.submit(Utterance { samples, rate })`.

5. **Event drain** — the loop drains `VoiceEvent`s from the pipeline's
   crossbeam receiver:
   - `Text(t)` → forwarded to the UI callback
   - `Audio(pcm)` → pushed to the cpal output ring buffer; the `assistant`
     `AtomicBool` is set (so the VAD gate knows the assistant is speaking)
   - `TurnComplete`/`Interrupted` → `assistant` cleared, loop returns to
     listening
   - `Error(e)` → forwarded to the UI callback, `assistant` cleared

### cpal output (ring buffer playback)

- A `Arc<Mutex<VecDeque<f32>>>` ring buffer feeds the cpal output callback.
- **Prebuffer** — playback starts only after `rate/5` samples accumulate
  (avoids underrun stutter on the first chunk).
- **Idle reset** — after `rate/2` empty frames, the output stream resets to
  idle (stops pulling from the ring, outputs silence). This prevents a
  continuous low-level hum when no audio is playing.

### `StreamingPcmResampler` (in `realtime.rs`)

The model's Mimi detokenizer outputs at 24 kHz; the cpal output device is
usually 48 kHz on macOS. The `StreamingPcmResampler` (in `realtime.rs`, used
by the `Lfm2VoiceEngine`) maintains `prev: Option<f32>` across decoded chunks
for the integer-upsample case (24k→48k), doing linear interpolation between
the last sample of the previous chunk and the first of the next. This
eliminates the audible discontinuity that the old per-chunk `resample_slice`
caused.

For non-integer ratios, it falls back to `resample_slice` (no cross-chunk
continuity — a known limitation, but the common macOS case is integer 2×).

### Mic pause/resume

`mic_enabled` `AtomicBool` — when false, the VAD loop skips all mic input
(clears the buffer, doesn't compute RMS, doesn't submit utterances). The cpal
stream stays alive (not killed), so resume is instant. This is used for:
- The mic toggle button (`voice_set_mic_enabled`)
- Mic-pause-on-typing (the frontend pauses the mic while the user is typing)

## Wiring

**Upstream:** cpal mic → VAD loop → `RealtimePipeline::submit(Utterance)`.

**Downstream:** `RealtimePipeline::events()` → VAD loop drains → cpal output
ring + UI callback (`RuntimeEvent` stream).

**Owned by:** the desktop crate's `Lfm2Session` (`voice/runtime.rs`), which
wraps it in a `VoiceSession` with Tauri lifecycle management. Also used
directly by the `duplex_chat` example.

## Cross-references

- [`tauri-voice.md`](tauri-voice.md) — how the desktop crate wraps this.
- [`AS_BUILT_claude_changes.md`](AS_BUILT_claude_changes.md) §6–7 — the
  as-built record.
- `packages/desktop/src-tauri/crates/liquid-audio/src/realtime.rs` — the
  `RealtimePipeline` + `Lfm2VoiceEngine` that this runtime drives.
- `packages/desktop/src-tauri/crates/liquid-audio/examples/duplex_chat.rs` —
  the full-duplex live demo that uses this runtime directly.