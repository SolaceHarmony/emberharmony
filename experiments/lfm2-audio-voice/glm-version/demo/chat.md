# demo_chat (Rust port)
**Source:** `liquid-audio-rs/examples/mic_chat.rs` · **Python:** `upstream-liquid-audio/…/demo/chat.py` (not in the vendored tree; see `ARCH/demo/chat.md`) · **On the LFM2-Audio inference path:** no

> Companion to [`ARCH/demo/chat.md`](../../ARCH/demo/chat.md). The original
> documents the Python `chat.py` Gradio/fastrtc demo; this documents the Rust
> `mic_chat.rs` headless equivalent.

## Role
`mic_chat.rs` is the realtime speech-to-speech demo harness in the Rust port.
It is the *orchestration shell* that wraps the model: it owns turn-taking (mic
energy VAD), assembles a `ChatState` (the prefill bundle), drives
`LFM2AudioModel::generate_interleaved` as a **synchronous callback stream**, and
routes each yielded token to either text-display or streaming Mimi audio decode.
None of the neural math lives here — this file is glue. It exists so a human can
talk to the 1.5B model in real time without batching the whole reply into one
WAV.

## How it works (Rust)
The control flow here (`mic_chat.rs`) is a single thread + a callback closure (no
producer/consumer queue split, unlike the Python `Thread` + `queue.Queue`) — **half-duplex**,
like the Python `ReplyOnPause`/`can_interrupt=False` demo.

> **There is now also a worker-thread path.** `src/realtime.rs` (`RealtimePipeline`) +
> `examples/duplex_chat.rs` *do* have the producer/consumer split — a persistent inference
> worker thread owns the model and talks to the consumer over `crossbeam-channel`, with live
> capture (full-duplex) and explicit `AtomicBool` barge-in. That is the faithful analog of
> `chat.py`'s `chat_producer` `Thread` + `queue.Queue`, extended toward `moshi/server.py`.
> The full Python↔Rust threading dissection (torch intra/inter-op, the GIL, both demos) is in
> **[`../threading.md`](../threading.md)**.

**Turn-taking (hand-rolled energy VAD).** `record_utterance` (`mic_chat.rs:68`)
reads from cpal, computes 200 ms RMS windows vs `LFM_VAD_THRESHOLD` (default
0.012), starts on the first window above threshold, stops after 800 ms of
silence or a 30 s cap. The Python demo uses `fastrtc.ReplyOnPause` (an external
VAD); the Rust port has no fastrtc, so it hand-rolls. Both are
`can_interrupt=False`-equivalent (a turn completes before the next is accepted,
no barge-in).

**ChatState assembly.** On the first turn it injects a system turn
(`"<|im_start|>system\n"` + `"Respond with interleaved text and audio."` +
`"<|im_end|>\n"`), then opens a user turn. The mic audio (cpal int16) is
normalized to f32 by `/32768.0` and handed to `chat.add_audio(wave,
sampling_rate)`. Inside `ChatState::add_audio` (`processor.rs:247`): resample to
16 kHz (`crate::resample`), run the mel front-end, append a run of
`LFMModality::AudioIn` flags of length `mel2emb_len(T')`. Then `end_turn()` +
`new_turn("assistant")` primes the model to continue as the assistant.

**Generation (single-thread callback).** `generate_interleaved(&chat, &params,
|tok| …)` (`mic_chat.rs:245`) drives the model. Each `GenToken` is dispatched
in the callback:
- `GenToken::Text(u32)` → decode incrementally with the tokenizer, strip a
  trailing `<|text_end|>`, print.
- `GenToken::Audio(Vec<u32>)` → if the frame contains `2048` (EOAudio),
  early-return (skip decode, flip scheduler back to text). Otherwise
  `mimi.decode_step(codes (1,8,1))` (`mic_chat.rs:263`) → a 1920-sample
  waveform chunk, pushed to the cpal output ring buffer.

**Playback.** cpal output ring buffer + `resample_slice(24k → device_rate)`
(`mic_chat.rs:206`, `:265`). The Rust resamples Mimi's 24 kHz to the device
rate; the Gradio demo fixes both at 24 kHz.

**History writeback.** After generation, `chat.append(text, audio_out,
modality_flag)` folds the assistant's own emitted tokens back into `ChatState`
so the *next* turn's prefill sees them — this is what makes it a conversation.
Then `end_turn()` + `new_turn("user")` re-prime.

**`generate_interleaved` (the actual model loop).** Not in this file but the
engine it drives — see [`glm-version/model/lfm2_audio.md`](../model/lfm2_audio.md).

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| mic capture (cpal int16) | cpal input stream | f32 `(1, N)` (`/32768.0`) |
| `add_audio` resample → mel | f32 `(1, N)` @ rate | mel `(128, T')` F32, stored in `audio_in` |
| modality flags (audio-in) | mel width `T'` | `mel2emb_len(T')` AUDIO_IN I64 |
| text tokens | `&str` | I64 `(1, ·)` ids |
| `generate_interleaved` prefill emb | `ChatState` | `in_emb (1, L, 2048)` model dtype |
| yielded `GenToken::Text` | hidden `(2048,)` → logits `(65536,)` | `u32` |
| yielded `GenToken::Audio` | hidden `(2048,)` → depthformer | `Vec<u32>` (8 codes, 0..2048; 2048=EOAudio) |
| `mimi.decode_step` | codes `(1, 8, 1)` u32 | waveform `(1920,)` F32 @ 24 kHz |
| cpal playback | `(1920,)` F32 @ 24 kHz | resampled to device rate, int16 |

## Wiring (Rust)
**Upstream:** cpal mic stream → `record_utterance` energy VAD. `processor.rs` —
`ChatState::new(proc)`: the `proc` provides the tokenizer + mel front-end + Mimi
handle. See [`glm-version/processor.md`](../processor.md).

**Downstream:** `model/lfm2_audio.rs` — `generate_interleaved(&chat, &params,
on_token)` consumes the `ChatState` fields; yields `GenToken::Text(u32)` and
`GenToken::Audio(Vec<u32>)`. See
[`glm-version/model/lfm2_audio.md`](../model/lfm2_audio.md). `processor.rs` —
`chat.append(...)` writes the assistant's emitted tokens back into the same
`ChatState`. `audio_out.rs::MimiDetokenizer` — `decode_step(codes)` →
`(1920,)` F32 @ 24 kHz. See [`glm-version/moshi/README.md`](../moshi/README.md).

## Python ↔ Rust — where the port differs

| Python (`chat.py`) | Rust (`mic_chat.rs`) | Difference | Why |
|---|---|---|---|
| `fastrtc.ReplyOnPause` (external VAD) | `record_utterance` energy VAD (`:68`) | **deliberate: hand-rolled VAD** | no fastrtc in Rust; hand-rolled RMS-window VAD. Both `can_interrupt=False`-equivalent. |
| `Thread` + `queue.Queue` producer/consumer | **two paths:** `mic_chat.rs` = single thread + callback closure (`:245`); `realtime.rs`/`duplex_chat.rs` = worker thread + `crossbeam-channel` | **`mic_chat`: sync callback; `realtime`: faithful producer/consumer** | `mic_chat` decodes in the callback (no queue). `realtime.rs` is the faithful `Thread`+`Queue` analog — worker owns the model, channels carry `VoiceEvent`s, **+ explicit barge-in**. See [`../threading.md`](../threading.md). |
| `mimi.streaming(1)` + `mimi.decode(t[None,:,None])` | `mimi.reset_stream()` + `mimi.decode_step(codes (1,8,1))` (`:239`, `:263`) | **deliberate: moshi-crate reuse** | §2.3. Streaming codec state across frames. |
| `wav / 32_768` f32 norm | cpal int16→f32 `/32768.0` (`:59`, `:97`) | identical | same int16→f32 convention. |
| Gradio WebRTC sink, `yield (24000, int16)` | cpal output ring buffer + `resample_slice(24k→out_rate)` (`:206`, `:265`) | **deliberate: cpal** | Rust resamples Mimi's 24 kHz to the device rate; Gradio fixes both at 24 kHz. |
| `device="cuda"` (hard-coded) | `select_device()` CPU/F32 or Metal/bf16 (`:35`) | **deliberate: device-agnostic** | §2.1. Python demo is CUDA-pinned; Rust runs CPU. |
| `temp=1.0, topk=4` (greedy text) | `GenParams{audio_temperature:Some(1.0), audio_top_k:Some(4), text_*:None}` (`:210`) | identical | same README interleaved defaults; `0→None` greedy mapping. |
| `EOAudio (t==2048)` skip decode | `frame.contains(&2048)` early-return (`:258`) | identical | same terminator semantics. |

`demo/` is explicitly out of the parity surface (PYTHON_VS_RUST §4);
`mic_chat.rs` is a faithful headless re-expression, not a numerically-graded port.

## Precision / gotchas (Rust-specific)
- **EOAudio = 2048.** A frame containing 2048 is the audio-stream terminator:
  it is *not* decoded to waveform (`mimi.decode` would reject codes ≥ 2048),
  it flips the scheduler back to text, and the callback early-returns past it.
- **Special text tokens.** Token 7 = `<|im_end|>` ends the assistant turn;
  token 130 = `<|text_end|>` signals the text segment is done and forces the
  next interleave block to audio. The display strips a trailing
  `<|text_end|>`.
- **Frame-size arithmetic.** 1920 samples/frame = 24000 Hz ÷ 12.5 Hz Mimi frame
  rate; an 8-code frame is exactly one Mimi step. The `GenToken` enum dispatch
  (`Text`/`Audio`) is load-bearing — it is the only thing distinguishing
  text/codes on the wire.
- **bf16 storage of a precision-sensitive front-end.** The mel is computed in
  f32 (with f64 window/filterbank/twiddles) and stored f32 in `audio_in` (the
  Rust port keeps f32 on CPU; the Python casts to bf16 at the storage
  boundary). See [`glm-version/model/conformer/processor.md`](../model/conformer/processor.md).
- **CPU-friendly.** The Rust demo runs on CPU/f32 (no candle CPU bf16 matmul),
  where the Python demo hard-codes `device="cuda"` and won't boot CPU-only.
- **History mutation is the conversation.** Forgetting `chat.append` would
  make every turn a cold single-shot; the append + re-prime (`new_turn("user")`)
  is what threads context across turns.
- **Energy VAD thresholds.** `LFM_VAD_THRESHOLD` (0.012), 200 ms windows, 800 ms
  silence stop, 30 s cap. These are tuned for a quiet room; a noisy environment
  may need a higher threshold.

## Cross-references
- [`ARCH/demo/chat.md`](../../ARCH/demo/chat.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.3 (moshi-crate
  reuse), §4 (demo out of parity surface).
- `liquid-audio-rs/PORT_STATUS.md` — the IO model (sync streaming → sync callback).