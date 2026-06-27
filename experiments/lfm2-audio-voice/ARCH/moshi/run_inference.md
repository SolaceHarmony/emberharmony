# moshi_run_inference
**Code:** `TR04` · **Source:** `moshi/run_inference.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
The vendored Kyutai **offline** streaming-inference CLI for the **Moshi / Hibiki / STT** family — a *different* model from LFM2-Audio. It reads one audio file, chops it into fixed Mimi frames, feeds the codes through `LMGen.step` at a constant 12.5 Hz cadence, and emits per-item `(text_tokens, audio_tokens)`, optionally vocoding the audio codes back to a wav. It is **off the LFM2-Audio path**: LFM2-Audio uses its own backbone + depthformer (`model_lfm2_audio`) driven by a synchronous streaming generator and `mimi.streaming(1)` decode in `demo/chat.py`, never `LMGen`, never `lm_gen.step`. It lives here only because the whole `moshi/` subtree was vendored wholesale, and it is the conceptual reference for the fixed-frame, EOS-on-frame offline loop.

## How it works
There is **no neural code in this file** — every tensor op is delegated to `MimiModel` (`moshi_compression`) and `LMGen` (`moshi_lm`). The mechanism is a fixed-cadence frame loop plus a per-batch-item turn-end state machine. State lives in the `InferenceState` dataclass; the loop is `InferenceState.run`.

**Construction (`run_inference.py:66-95`).** Stores the Mimi codec, a `sentencepiece` tokenizer, and builds `self.lm_gen = LMGen(lm, cfg_coef, condition_tensors=…, **kwargs)`. Two load-bearing scalars and one mode switch:
- `self.frame_size = int(mimi.sample_rate / mimi.frame_rate)` (`:87`). With `SAMPLE_RATE=24000`, `FRAME_RATE=12.5` → exactly **1920 samples** = one Mimi frame = 80 ms. This is the quantum everything is chunked on.
- `condition_tensors = get_condition_tensors(...)` (`:82`, `:34-57`): only `hibiki` builds real conditions (`text={"description":"very_good"}`, plus a `"very_bad"` negative branch appended when `cfg_coef != 1.0` for classifier-free guidance); any other model with a conditioner `raise`s. Moshi/STT pass through with `{}`.
- Both stateful modules go into permanent streaming mode: `mimi.streaming_forever(batch_size)` + `lm_gen.streaming_forever(batch_size)` (`:89-90`). This installs the persistent KV-cache / conv-ring-buffer streaming state on the codec transformers and the LM, so each `.encode`/`.step`/`.decode` is incremental, not a full re-encode.

**STT-only input padding (`run_inference.py:121-127`).** For `model_type == "stt"` it pads the raw PCM with `audio_silence_prefix_seconds` of left silence and `(audio_delay_seconds + 1.0)` of right silence (both × 24000 samples, `mode="constant"`). This bakes the STT alignment delay into the input stream. Moshi/Hibiki skip this.

**Frame deque (`run_inference.py:128-135`).** `in_pcms.split(frame_size, dim=2)` then keeps **only fully-sized frames** (`chunk.shape[-1] == frame_size`) — any trailing partial frame is dropped (no zero-pad of the last frame). The kept frames become a `collections.deque`, popped left-to-right.

**The main loop (`run_inference.py:138-202`), `while not all(eos_reached)`.** Per iteration:
1. **Source a frame of codes.**
   - If the deque is non-empty: `chunk = chunks.popleft()`, `codes = mimi.encode(chunk)` → int codes `(B, n_q, T)` (`:140-141`).
   - Else, end-of-file behavior is model-specific (`:142-163`):
     - **hibiki**: the *first* post-EOF frame feeds an explicit end-of-stream marker — a code tensor filled with `mimi.cardinality` (= **2048**) on *all* codebooks, shape `(B, num_codebooks, 1)`, `dtype=long` (`:144-154`); subsequent post-EOF frames encode `frame_size` of silence (`:155-160`). This lets the model keep generating its translation tail after the input ends, until it emits text-EOS.
     - **other models (moshi/stt)**: `break` immediately at EOF (`:161-163`).
2. **First-frame priming (`run_inference.py:164-170`).** On the very first frame it calls `lm_gen.step(codes)` an *extra* time and discards the result; if `max(delays) > 0` that priming step must return `None` (asserted). Rationale (comment `:165-166`): without it the first real slice of codes would be overwritten by `LMGen`'s initial-token bootstrap, so the model never "sees" frame 0.
3. **The LM step (`run_inference.py:171-174`).** `tokens = lm_gen.step(codes)`. `LMGen` internally applies the per-codebook **acoustic-delay** pattern (`_delay_sequence` / `_undelay_sequence`, `lm.py:344-369`), runs the Moshi backbone + depformer, and **returns `None` until the delay warmup is satisfied** (`continue` on `None`). Once warm it returns `(B, dep_q+1, 1)` — index 0 is the **text** (inner-monologue) stream, indices `1..dep_q+1` are the `dep_q` audio codebooks. `assert tokens.shape[1] == dep_q + 1` (`:174`).
4. **Decode + per-item turn-end (`run_inference.py:175-201`), when `dep_q > 0`.**
   - `out_pcm = mimi.decode(tokens[:, 1:]).cpu()` — drop the text row, vocode the audio codes → `(B,1,frame_size)` f32 @ 24 kHz.
   - Per batch item `b`: if already `eos_reached[b]`, skip. Else if the text token equals `text_tokenizer.eos_id()` → mark `eos_reached[b]=True` (but warn "EOS sampled too early" if `need_eos_input` is still set, i.e. the model emitted EOS before the input file even ended — `:182-187`). Append the text token and the decoded pcm to the per-item accumulators.
   - For `b == 0` only, live-print: skip ids `0` (pad) and `3` (epad/special), else `id_to_piece` → replace SentencePiece `▁` with space → `printer.print_token` (`:191-195`).
   - **Text-only models (`dep_q == 0`, `run_inference.py:196-201`)**: no Mimi decode; just print `tokens[0,0]` with the same `0`/`3` skip.

**Sampling.** This file does not sample — `use_sampling`, `temp` (audio temperature), `temp_text` (text temperature) are owned by `LMGen` and only *logged* here (`:113-115`). The actual top-k/top-p multinomial lives in `moshi_util_sampling`.

**Output assembly (`run_inference.py:208-217`).** Per item, `torch.cat(one_texts, dim=0)` (text ids along time) and `torch.cat(one_pcms, dim=1)` (waveform along time) → `list[(text_tokens, audio_tokens)]`. `main()` then writes each item's pcm to a wav via `sphn.write_wav` at `mimi.sample_rate` (`:304-315`).

**`main()` plumbing (`run_inference.py:220-315`).** argparse, `seed_all(4242)` (`:23-31`: sets torch/cuda/python/numpy seeds, `cudnn.deterministic=False`), `CheckpointInfo.from_hf_repo` (→ `moshi_loaders`), `get_mimi`/`get_text_tokenizer`/`get_moshi`, `--device` defaults **cuda**, `--half` toggles bf16→fp16 (`dtype` default `torch.bfloat16`, `:245-252`), `--cfg-coef` CFG coefficient, `--batch-size` default 8 (forced to 1 when `dep_q == 0`). Input read with `sphn.read(infile, sample_rate=mimi.sample_rate)` → f32, then `in_pcms[None, 0:1].expand(batch_size, -1, -1)` broadcasts the **same** mono clip across the batch. Whole program runs under `with torch.no_grad()` (`:318-320`).

## Dtypes & shapes
| Stage | In | Out |
|---|---|---|
| `sphn.read(infile)` | wav file | f32 PCM `(C, N)` @ 24 kHz |
| batch expand (`:290`) | f32 `(1,1,N)` | f32 `(B,1,N)` (same clip broadcast) |
| frame slice (deque) | f32 `(B,1,N)` | f32 frames `(B,1,1920)` each (partial dropped) |
| `mimi.encode` (`:141`) | f32 `(B,1,1920)` | int codes `(B, n_q, T)` (u32 in Rust); EOF marker = `(B,n_q,1)` filled `2048`, int64 |
| `lm_gen.step` (`:171`) | int codes `(B, n_q, 1)` | `None` during delay warmup, else int `(B, dep_q+1, 1)` |
| text row | `tokens[:,0]` int64 | SentencePiece piece (skip ids 0, 3; stop on `eos_id()`) |
| `mimi.decode(tokens[:,1:])` (`:176`) | int codes `(B, dep_q, 1)` | f32 waveform `(B,1,1920)` @ 24 kHz |
| `run` return | — | `list[(text_tokens int64 (Ttok,), audio_tokens f32 (1, Nsamp))]` |

Notes: no f32-upcast norm/softmax/mel happens *in this file* (all inside `moshi_compression`/`moshi_lm`); EOF marker code value `2048` = `mimi.cardinality` = the per-codebook EOAudio/end-of-stream sentinel on this codec.

## Wiring
**Upstream (feeds this):**
- [moshi_loaders](models/loaders.md) — `CheckpointInfo.from_hf_repo` → Mimi codec, Moshi LM, `sentencepiece` tokenizer, `lm_gen_config`.
- [MimiModel](models/compression.md) — `mimi.encode` produces the int code frames `(B, n_q, 1920→T)` that drive each step; `mimi.decode` vocodes the audio codes back to f32 @ 24 kHz. Edge in: f32 `(B,1,1920)`; edge out (decode): f32 `(B,1,1920)`.
- [moshi_lm / LMGen](models/lm.md) — `LMGen.step` consumes int codes `(B, n_q, 1)` and returns the `(B, dep_q+1, 1)` text+audio token frame. The delay/undelay + depformer math lives there.

**Downstream (consumes this output):**
- Terminal printer ([moshi_client_utils](client_utils.md)) — `Printer`/`RawPrinter.print_token` renders the text stream live (decoded pieces, skipping ids 0/3).
- `sphn.write_wav` (external) — the accumulated per-item f32 audio `(1, Nsamp)` @ 24 kHz is written to disk in `main()`.
- No LFM2-Audio component consumes this — it is a leaf CLI, not part of the LFM2 graph ([model_lfm2_audio](../model/lfm2_audio.md), [demo_chat](../demo/chat.md) are the on-path analogues).

## Python ↔ Rust
**Not ported.** `Rust: -`. The `liquid-audio-rs` `core` scope excludes the vendored `moshi/` subtree by design (PYTHON_VS_RUST.md §4: "vendored `liquid_audio/moshi/**` is reused as the `moshi` crate … not re-ported"). The Rust port reuses Kyutai's **`moshi` crate** for the Mimi codec only (PYTHON_VS_RUST.md §2.3) and never reconstructs `LMGen`, this CLI, or the Moshi 7B LM — LFM2-Audio's Rust path is `model_lfm2_audio` + its own depthformer + `demo/chat.py`-shaped synchronous streaming, not this fixed-cadence `lm_gen.step` loop. The closest Rust analogue in spirit (offline frame loop driving the model) is `examples/generate.rs`, which drives the **LFM2** model, not Moshi.

## Precision / gotchas
- **Off-path.** Nothing in the LFM2-Audio inference graph calls this; do not treat `dep_q+1` text+audio interleaving or `LMGen` delays as the LFM2 contract. LFM2-Audio's depthformer emits an `(8,)` audio frame with `2048 = EOAudio` per codebook and a separate text head — a different head/cadence than Moshi's `dep_q+1` row.
- **Cardinality sentinel `2048`.** `mimi.cardinality` is reused both as the per-codebook EOAudio/end-of-stream marker (hibiki EOF, `:148-154`) and as the count of valid code values — code ids run `0..2048` with `2048` reserved. Same numeric sentinel as LFM2-Audio's EOAudio, different model.
- **Partial-frame truncation (`:131-134`).** The trailing sub-1920-sample remainder of the input is silently dropped, not zero-padded — output is quantized to whole 80 ms frames.
- **Same clip across the batch (`:290`).** `expand(batch_size,…)` broadcasts one mono channel; the batch is *not* independent clips — it is the same audio replicated, so per-item `eos_reached` divergence is purely from stochastic sampling, not different inputs.
- **First-frame double-step (`:164-170`).** The discarded priming `lm_gen.step` is required; skipping it makes the model miss frame 0. With non-zero delays the primer must return `None` (asserted).
- **"EOS sampled too early" (`:182-186`).** A warning, not a stop: if text-EOS arrives while `need_eos_input` is still true (input not yet exhausted), the item is *not* marked done — it logs and keeps going, because EOS before EOF is considered impossible/anomalous for hibiki.
- **CUDA-coupled.** `--device` defaults `cuda`; `seed_all` touches `torch.cuda` guarded by `is_available()`. No device-agnostic path here (contrast PYTHON_VS_RUST.md §2.1 for the LFM2 Rust port's CPU-first design).
