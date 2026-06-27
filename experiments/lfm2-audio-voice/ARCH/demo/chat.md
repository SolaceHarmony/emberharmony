# demo_chat
**Code:** `DM01` · **Source:** `demo/chat.py` · **Rust:** `examples/mic_chat.rs` · **On the LFM2-Audio inference path:** no

## Role
The realtime speech-to-speech demo harness. It is the *orchestration shell* that wraps the model: it owns turn-taking (mic VAD), assembles a `ChatState` (the prefill bundle), drives `LFM2AudioModel.generate_interleaved` as a synchronous streaming generator, and routes each yielded token to either text-display or streaming Mimi audio decode. None of the neural math lives here — this file is glue. It exists so a human can talk to the 1.5B model in real time without batching the whole reply into one WAV.

## How it works
The control flow is a producer/consumer split across two threads, fed by an external VAD that decides *when* a turn ends.

**Turn-taking (external VAD).** `fastrtc.ReplyOnPause(chat_response, input_sample_rate=24_000, output_sample_rate=24_000, can_interrupt=False)` (`chat.py:122-128`) wraps the response callback. `ReplyOnPause` runs its own silence/pause detector over the WebRTC mic stream and only invokes `chat_response` once the user pauses — so the demo never implements VAD itself, and `can_interrupt=False` means a turn must fully complete before the next is accepted (no barge-in). The Rust port (`mic_chat.rs:68-138`) has no fastrtc, so it hand-rolls an **energy VAD**: 200 ms RMS windows vs `LFM_VAD_THRESHOLD` (default 0.012), start on first window above threshold, stop after 800 ms of silence or a 30 s cap.

**ChatState assembly (the prefill bundle).** `chat_response` (`chat.py:40`) mutates a persistent `gr.State(ChatState(proc))`. On the first turn it injects a system turn `"<|im_start|>system\n"` + `"Respond with interleaved text and audio."` + `"<|im_end|>\n"` (`chat.py:51-54`), then opens a user turn. The mic audio (a `(rate, np.ndarray)` int16-range pair) is normalized to f32 by `wav / 32_768` and handed to `chat.add_audio` (`chat.py:59`). Inside `ChatState.add_audio` (`processor.py:226`): resample to 16 kHz (`torchaudio.functional.resample`), run the mel front-end `self.proc.audio(wave, length)` → `(128, T')` log-mel, cast to `self.dtype` (bf16), and append a run of `LFMModality.AUDIO_IN` flags of length `mel2emb_len(T')` (the conformer's 8× subsample shrink — the modality stream tracks *embedding* slots, not mel frames). Then `end_turn()` appends `"<|im_end|>\n"` and `new_turn("assistant")` appends `"<|im_start|>assistant\n"`, leaving the sequence primed for the model to continue *as the assistant*. `ChatState` is a `Mapping` (`processor.py:184`) exposing exactly `["text","audio_in","audio_in_lens","audio_out","modality_flag"]`, so `**chat` splats straight into `generate_interleaved`'s keyword args.

**Producer thread.** `chat_producer` (`chat.py:14`) runs under `torch.no_grad()` and `mimi.streaming(1)` (batch-1 streaming codec context) and iterates `lfm2_audio.generate_interleaved(**chat, max_new_tokens=1024, audio_temperature=temp, audio_top_k=topk)`. Each yielded tensor `t` is pushed to a `queue.Queue`. Critically, **the producer also does the audio decode inline**: if `t.numel() > 1` (an 8-code audio frame) and it is *not* the EOAudio terminator (`(t == 2048).any()` → `continue`, skip decode), it calls `mimi.decode(t[None, :, None])[0]` to get a 1920-sample waveform chunk and pushes *that* too (`chat.py:30-35`). `mimi.decode` here is the streaming Mimi codec; under `mimi.streaming(1)` it maintains the SEANet/transformer decoder state across calls so consecutive frames concatenate seamlessly. The 1920 = one 12.5 Hz Mimi frame at 24 kHz (24000/12.5).

**Consumer / generator.** `chat_response` drains the queue (`chat.py:72-89`) and dispatches by `numel()`:
- `numel()==1` → a text token: decode incrementally with `proc.text.decode`, strip a trailing `<|text_end|>`, and `yield AdditionalOutputs(cur_string)` to update the Gradio textbox.
- `numel()==8` → an audio code frame: buffer it into `out_audio` for later history append (not played here — the producer already enqueued its waveform).
- `numel()==1920` → a decoded waveform chunk: scale f32→int16 (`* 32_767`, cast `int16`) and `yield (24_000, np_chunk)` to the WebRTC sink for immediate playback.
- `None` sentinel → end of turn.

**generate_interleaved (the actual model loop, `lfm2_audio.py:234`).** Not in this file but the engine this file drives. `_prefill` (`lfm2_audio.py:307`) scatters three embedding sources into one `(1,L,2048)` tensor by modality mask: text via `embed_tokens`, audio-in via conformer→`audio_adapter`, audio-out via summed `audio_embedding` over `audio_out + codebook_offsets`. Then a per-step loop with a hybrid-conv KV cache: at each step the LFM backbone produces a hidden, and the **modality scheduler** alternates `interleaved_n_text` text tokens then `interleaved_n_audio` audio frames (`lfm2_audio.py:256-305`). Text branch: `logits = linear(hidden[0,-1], embed_tokens.weight)` (tied head, 65536 vocab), sample, break on token 7 (`<|im_end|>`), flip to audio when the text budget runs out or token 130 (`<|text_end|>`) appears; next input embedding is `embed_tokens(token)`. Audio branch: `_sample_audio_frame(hidden[0,-1])` runs the depthformer to emit an 8-vector of codes (0..2047, or 2048=EOAudio); a leading 2048 forces the whole frame to 2048 and flips back to text; next input embedding is `audio_embedding(frame + codebook_offsets).sum(0)`. Sampling knobs: greedy text by default, sampled audio at `temp=1.0, top_k=4` (README interleaved defaults; the demo passes `temp`/`topk` with 0→`None`=greedy mapping at `chat.py:41-44`).

**History writeback.** After the queue drains, `chat.append(text=stack(out_text), audio_out=stack(out_audio), modality_flag=tensor(out_modality))` (`chat.py:91-95`) folds the assistant's own emitted tokens back into `ChatState` so the *next* turn's prefill sees them — this is what makes it a conversation, not isolated single-shots. Then `end_turn()` + `new_turn("user")` re-prime for the next utterance.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| mic capture (`audio`) | `(rate:int, np.ndarray int16-range f64)` | — |
| f32 normalize (`wav/32768`) | int16-range | `(1,N)` f32 |
| `add_audio` resample → mel | `(1,N)` f32 @ rate | mel `(128,T')` computed f32/f64, **stored bf16** in `audio_in` |
| modality flags (audio-in) | mel width `T'` | `mel2emb_len(T')` AUDIO_IN ints (int64 enum) |
| text tokens | str | `(1,·)` int64 ids |
| `generate_interleaved` prefill emb | `text/audio_in/audio_out/modality_flag` | `in_emb (1,L,2048)` model dtype (bf16) |
| yielded text token | hidden `(2048,)` → logits `(65536,)` | `(1,)` int64 |
| yielded audio frame | hidden `(2048,)` → depthformer | `(8,)` int (codes 0..2048; 2048=EOAudio) |
| `mimi.decode(t[None,:,None])` | codes `(1,8,1)` int (u32 in Rust) | waveform `(1920,)` f32 @ 24 kHz |
| playback yield | `(1920,)` f32 | `(24000, (1920,) int16)` |

Promotions: mel front-end upcasts to f32/f64 then rounds once to bf16 for storage; backbone/norm/softmax run in model dtype with f32 upcast internally (see neighbors); token ids stay int64; audio codes are small ints (u32 on the Rust side).

## Wiring
**Upstream**
- `fastrtc.ReplyOnPause` mic stream → `(rate, np.ndarray)` int16-range — external VAD gating, not a port neighbor.
- [core_processor](../processor.md) — `ChatState(proc)`: the `proc` provides the tokenizer + mel front-end + Mimi handle; `add_audio` calls `proc.audio` (mel `(128,T')` bf16) and `add_text` calls `proc.text.encode` (int64 ids). `**chat` splats the 5-key bundle into the model.

**Downstream**
- [model_lfm2_audio](../model/lfm2_audio.md) — `generate_interleaved(**chat,…)` consumes `{text int64 (1,L), audio_in bf16 (128,ΣT'), audio_in_lens int64, audio_out int (8,·), modality_flag int64 (1,L)}`; yields text `(1,)` int64 and audio frames `(8,)` int.
- [core_processor](../processor.md) — `chat.append(...)` writes the assistant's emitted `text (1,·) int64` / `audio_out (8,·) int` / `modality_flag` back into the same `ChatState`.
- [moshi_compression](../moshi/models/compression.md) — `mimi.decode(codes (1,8,1) int)` under `mimi.streaming(1)` → `(1920,)` f32 @ 24 kHz waveform chunk (Rust: `mimi.decode_step`).

## Python ↔ Rust
| Python (`chat.py`) | Rust (`mic_chat.rs`) | Note |
|---|---|---|
| `ReplyOnPause` (fastrtc external VAD) | `record_utterance` energy VAD (`mic_chat.rs:68`) | **Divergence:** no fastrtc; hand-rolled RMS-window VAD. Both `can_interrupt=False`-equivalent (turn completes before next). |
| `Thread` + `queue.Queue` producer/consumer | single thread + callback closure (`generate_interleaved(&chat, &params, |tok| …)`, `mic_chat.rs:245`) | Sync streaming generator → sync callback stream (PORT_STATUS §IO model). No queue: decode happens in the callback. |
| `mimi.streaming(1)` + `mimi.decode(t[None,:,None])` | `mimi.reset_stream()` + `mimi.decode_step(codes (1,8,1))` (`mic_chat.rs:239,263`) | moshi-crate reuse (PYTHON_VS_RUST §2.3). Streaming codec state across frames. |
| `wav / 32_768` f32 norm; cpal int16→f32 `/32768.0` | `mic_chat.rs:59,97` | Same int16→f32 convention; mic wav = f32. |
| Gradio WebRTC sink, `yield (24000, int16)` | cpal output ring buffer + `resample_slice(24k→out_rate)` (`mic_chat.rs:206,265`) | Rust resamples Mimi's 24 kHz to the device rate; Gradio fixes both at 24 kHz. |
| `device="cuda"` (model.py:18, chat.py:94) | `select_device()` CPU/f32 or Metal/bf16 (`mic_chat.rs:35`) | **Divergence:** device-agnostic (PYTHON_VS_RUST §2.1). Python demo is CUDA-pinned; Rust runs CPU. |
| `temp=1.0, topk=4` (greedy text) | `GenParams{audio_temperature:Some(1.0), audio_top_k:Some(4), text_*:None}` (`mic_chat.rs:210`) | Same README interleaved defaults; `0→None` greedy mapping. |
| `EOAudio (t==2048)` skip decode | `frame.contains(&2048)` early-return (`mic_chat.rs:258`) | Same terminator semantics. |

`demo/` is explicitly out of the parity surface (PYTHON_VS_RUST §4); `mic_chat.rs` is a faithful headless re-expression, not a numerically-graded port.

## Precision / gotchas
- **EOAudio = 2048.** A frame containing 2048 is the audio-stream terminator: it is *not* decoded to waveform (`mimi.decode` would reject codes ≥ 2048 — valid codes are 0..2047), it flips the scheduler back to text, and `chat_producer` `continue`s past it (`chat.py:31-32`). The model's `_sample_audio_frame` only sets 2048 in position 0 then broadcasts it across all 8 codebooks (`lfm2_audio.py:300-301`).
- **Special text tokens.** Token 7 = `<|im_end|>` ends the assistant turn (loop `break`); token 130 = `<|text_end|>` signals the text segment is done and forces the next interleave block to audio (`lfm2_audio.py:276,281`). The display strips a trailing `<|text_end|>` (`chat.py:80`).
- **Frame-size arithmetic.** 1920 samples/frame = 24000 Hz ÷ 12.5 Hz Mimi frame rate; an 8-code frame is exactly one Mimi step. The consumer's `numel()` dispatch (1 / 8 / 1920) is load-bearing — it is the only thing distinguishing text/codes/waveform on the wire, so any shape drift silently mis-routes.
- **bf16 storage of a precision-sensitive front-end.** The mel is computed in f32/f64 (the NeMo preprocessor "is not robust to low precision") but stored bf16 in `audio_in`; the one rounding to bf16 happens at the storage boundary (`processor.py:238`), matching the Rust "extended precision until the very end" rule (PYTHON_VS_RUST §1.4).
- **CUDA-pinned demo.** `chat.py:94` hard-codes `device="cuda"` for the writeback modality tensor and `model.py` warms up on CUDA; as shipped the Python demo will not boot CPU-only. The Rust `mic_chat.rs` is the portable equivalent (CPU/f32 or Metal/bf16).
- **History mutation is the conversation.** Forgetting `chat.append` would make every turn a cold single-shot; the append + re-prime (`new_turn("user")`) at the end of `chat_response` is what threads context across turns.
