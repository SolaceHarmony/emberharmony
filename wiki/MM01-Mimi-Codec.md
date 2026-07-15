<!-- topic: Mimi Codec — Models -->
# MM01 · MimiModel codec
**Code:** `MM01` · **Source:** `moshi/models/compression.py` · **Rust:** `moshi crate mimi::Mimi` · **On the LFM2-Audio inference path:** yes

## Role
`MimiModel` is Kyutai's neural audio codec: a learned, streaming, residual-VQ transform between a 24 kHz waveform and a stack of discrete integer codes (the LFM2-Audio "audio frame"). It is the *top-level orchestrator* in this file — it owns the SEANet encoder/decoder, the optional enc/dec transformers, the framerate-resampling bridge (25 Hz ↔ 12.5 Hz), and the split-RVQ quantizer, wiring them into `encode` / `decode` / `forward` plus a streaming state. In LFM2-Audio it is a *peripheral the processor owns*, not the language model: it (1) encodes reference speech into the 8-codebook `audio_out` target codes for training, and (2) decodes generated 8-code frames back to waveform in the demo (and as the fallback vocoder when no LFM2 ISTFT detokenizer ships). It is never on the audio-*input* path — mic audio uses the conformer mel front-end instead.

## How it works
`MimiModel` is a `StreamingModule[_MimiState]` (`compression.py:105`, base API `CompressionModel` `:40-94`). The constructor (`:128-217`) stores `encoder`, `decoder`, optional `encoder_transformer` / `decoder_transformer`, the `quantizer`, and three rates: `frame_rate` (12.5 Hz, the codec output rate), `encoder_frame_rate` (`SAMPLE_RATE/encoder.hop_length` = 24000/960 = 25 Hz, `loaders.py:318`), and `sample_rate` (24000). `frame_size = int(sample_rate / frame_rate) = 1920` (`:244-246`) — one code column = 1920 waveform samples.

**Build (loaders).** `get_mimi` (`loaders.py:296-333`) assembles it from module dicts: SEANet `dim=512, n_filters=64, ratios=[8,6,5,4]` ⇒ `hop_length=960` ⇒ 25 Hz encoder rate; `_quantizer_kwargs` `dimension=256, n_q=32, bins=2048, input/output_dimension=512`; `_transformer_kwargs` `d_model=512`. `resample_method="conv"` (`loaders.py:320`) and `set_num_codebooks(num_codebooks)` (`:332`) → 8 active codebooks (1 semantic `rvq_first` + 7 acoustic `rvq_rest`).

**Framerate resampling — the 25↔12.5 bridge.** Because `encoder_frame_rate (25) != frame_rate (12.5)`, the constructor builds learnt `ConvDownsample1d`/`ConvTrUpsample1d` with `downsample_stride = 25/12.5 = 2` (`:189-217`). `interpolate` is forbidden with a causal model (`:190-192`), so this checkpoint uses `resample_method="conv"` (`learnt=True`). `_to_framerate` (`:267-278`) and `_to_encoder_framerate` (`:280-291`) dispatch on `resample_method`: for `"conv"` they call `self.downsample` / `self.upsample`; the `interpolate` branch (`nn.functional.interpolate(..., mode="linear")`) is dead for this config. The upsample is built `channel_wise=upsample_channel_wise_bug` (default True) — a deliberately preserved bug-compatible channel-wise transposed conv (`resample.py:68-109`).

**`encode` (`:376-388`).** `_encode_to_unquantized_latent` (`:338-374`) → `quantizer.encode`.
- One-shot (no streaming state, `:354-359`): `x = pad_for_conv1d(x, frame_size, frame_size)` to force an exact multiple of 1920 (the convs no longer accept partial inputs), then `emb = self.encoder(x)` (SEANet ⇒ `(B,512,T/960)` @ 25 Hz).
- Streaming (`:360-366`): rejects any `x` whose length is not a positive multiple of 1920 ("you are responsible for buffering"); runs `state.graphed_encoder(x).clone()`.
- Then optional `encoder_transformer` (`:367-372`), then `_to_framerate` → downsample to 12.5 Hz `(B,512,T/1920)`.
- `quantizer.encode(emb)` projects 512→256 and runs split-RVQ → **int codes `(B, K=8, T/1920)`**, values in `[0, 2047]` (`compression.py:387`, `vq.py:269-280`).

**`decode` (`:406-429`) — the inference-path direction.** `decode_latent(codes)` = `quantizer.decode(codes)` (`:431-433`, `vq.py:141-150`): index-selects codebook vectors, sums the residual stack, projects 256→512 → latent `(B,512,T)` @ 12.5 Hz. Then `_to_encoder_framerate` → upsample to 25 Hz; optional `decoder_transformer`; then `self.decoder(emb)` (or `state.graphed_decoder(emb).clone()` streaming) → SEANet decoder ×960 upsample → waveform `(B,1,T')` @ 24 kHz. Note `decode` does **not** trim trailing conv padding (it returns the raw decoder output); only `forward` trims `out[..., :length]`.

**Split-RVQ (the quantizer this orchestrates).** `SplitResidualVectorQuantizer` (`vq.py:170`) = `rvq_first` (`n_q_semantic=1`, `force_projection=True`) ⊕ `rvq_rest` (`n_q=7`). `encode` concatenates `rvq_first.encode(x)` with `rvq_rest.encode(x)` along the codebook axis (`vq.py:269-280`); each inner `ResidualVectorQuantizer.encode` (`vq.py:126-139`) projects via a `Conv1d` `input_proj` (512→256) and runs the residual cdist-argmin loop over codebooks (in `core_vq.py`). `decode` (`vq.py:141-150`) dequantizes + sums + `output_proj` (256→512). Cardinality = `bins` = 2048; `total_codebooks=32`, `num_codebooks=8` active.

**`forward` (`:293-336`)** is the full reconstruction loop used in training/analysis (not the split encode/decode entry points): encoder → enc-transformer → `_to_framerate` → asserts `|emb.shape[-1] − frame_rate·length/sample_rate| < 1` (`:315-320`) → `quantizer(emb, frame_rate)` (returns a `QuantizedResult`) → `_to_encoder_framerate` → dec-transformer → decoder → trim to input `length`. Also handles per-level quantizer freezing (`:298-309`).

**Streaming state.** `_init_streaming_state` (`:219-230`) wraps `encoder`, `decoder`, and each transformer in `CUDAGraphed` with **`disable = device.type != 'cuda'`** (`:221`) — CUDA-graph capture engages only on GPU; on CPU/Metal everything runs eager. `_MimiState` (`:97-102`) holds those four graphed callables. Entering `mimi.streaming(1)` persists conv/transformer state across `decode` calls so per-frame chunks stitch gaplessly; the demo decodes exactly one frame at a time inside the context (`demo/chat.py:21,34`).

## Dtypes & shapes
| Path | Input dtype+shape | Output dtype+shape | Internal notes |
|---|---|---|---|
| `encode` (audio-in for training targets) | f32 waveform `(B,1,L)` @ 24 kHz (resampled to 24k in mapper, `mapper.py:226-229`) | int codes `(B,8,T=L/1920)`, values `[0,2047]` | SEANet/transformer in module dtype (Python cuda/bf16; Rust CPU f32 / Metal bf16); `pad_for_conv1d` to ×1920; latent `(B,512,·)`→proj `(B,256,·)` for VQ |
| `decode` (inference audio-out) | int codes `(B,8,T)` (Rust casts to **u32** for RVQ `index_select`, `audio_out.rs:89,114`) | f32 waveform `(B,1,T·1920)` @ 24 kHz | `decode_latent` 256→512; upsample 12.5→25 Hz; SEANet decoder ×960 |
| `decode_step` (streaming, per frame) | int codes `(1,8,1)` (→u32) | `Option<f32 (1,1,~1920)>` (None during codec warmup) | persists conv/transformer state across calls |
| `forward` | f32 `(B,1,L)` | `QuantizedResult` with `.x` = f32 `(B,1,L)` reconstruction | upsample/quantize/downsample round trip; trims to `L` |

Weights = bf16 on disk. Codes are integers (the model's audio vocabulary; `2048`=EOAudio sentinel is a *model-side* token, never produced by Mimi). Resampling/conv arithmetic runs in the module compute dtype.

## Wiring
**Upstream (encode side, training):** [data_mapper](DA02-Chat-Mapper) `_encode_audio_out` feeds f32 waveform `(B,1,L)` @ 24 kHz (after `torchaudio.functional.resample` to Mimi's sample_rate) into `processor.mimi.encode`, getting `audio_out` target codes `(B,8,T)`. The encoder consumes [moshi_seanet](MO01-SEANet) `SEANetEncoder` (waveform→`(B,512,T/960)`), [moshi_transformer](MO03-Codec-Transformer) `ProjectedTransformer` (the enc/dec transformers), [moshi_resample](MO04-Framerate-Resample) `ConvDownsample1d`/`ConvTrUpsample1d` (25↔12.5 Hz), and [moshi_vq](QZ01-Split-RVQ) `SplitResidualVectorQuantizer` (latent↔codes).

**Upstream (decode side, inference):** [model_lfm2_audio](MD01-LFM2AudioModel) emits the audio frame `(8,)` int per step; [core_processor](CO01-Processor-ChatState) assembles frames into codes `(1,8,T)` and dispatches to this codec when no LFM2 ISTFT detokenizer is available (`processor.py` `decode`/demo path).

**Downstream:** the reconstructed f32 waveform `(1,1,1920)` @ 24 kHz per frame flows to:
- [demo_chat](DM01-Realtime-Chat) — `mimi.decode(t[None,:,None])` inside `mimi.streaming(1)`, played as a 1920-sample chunk (`chat.py:34,85`).
- [core_processor](CO01-Processor-ChatState) — codec is the fallback vocoder under `processor.decode` when `audio_detokenizer/` weights are absent.
- On the encode edge, the int codes `(B,8,T)` flow back into [data_mapper](DA02-Chat-Mapper) as the `audio_out` training targets consumed by [model_lfm2_audio](MD01-LFM2AudioModel)'s depthformer audio head loss.

## Python ↔ Rust
| Concern | Python (`compression.py`) | Rust (`moshi` crate `mimi::Mimi`, wrapped by `audio_out.rs`) |
|---|---|---|
| Codec model | `MimiModel` (`:105`) | `moshi::mimi::Mimi`, behind `MimiDetokenizer { inner: RefCell<Mimi> }` (`audio_out.rs:77-79`) |
| `encode` | `:376-388` | `MimiDetokenizer::encode` → `reset_state()` + `Mimi::encode` (`audio_out.rs:98-102`) |
| `decode` (one-shot) | `:406-429` | `MimiDetokenizer::decode` → `to_dtype(U32)` + `reset_state()` + `Mimi::decode` (`audio_out.rs:88-93`) |
| `decode` (streaming) | `mimi.streaming(1)` + per-frame `decode` | `decode_step` → `Mimi::decode_step(StreamTensor, StreamMask::empty)` (`audio_out.rs:113-118`) |
| reset stream | `streaming(...)` ctx / `reset_streaming` | `reset_stream` → `Mimi::reset_state` (`audio_out.rs:105-107`) |
| codebooks | `set_num_codebooks(8)` (`loaders.py:332`) | `Some(codebooks)` to `moshi::mimi::load` |
| quantizer keys | `quantizer.rvq_first.*` / `rvq_rest.*` | matched natively by `moshi::mimi` |

**Deliberate divergences** (PYTHON_VS_RUST.md §2.1, §2.2, §2.3, §2.10): (1) **Codec reuse** — Rust does not re-port this file; it reuses Kyutai's published `moshi` crate, chosen specifically because that crate matches this checkpoint's `rvq_first`/`rvq_rest` weight naming (candle-transformers' Mimi does not). The Python `MimiModel`/`CompressionModel`/`WrapperCompressionModel` orchestration is therefore *not* mapped symbol-for-symbol — the trait `AudioDetokenizer` (`audio_out.rs:25-62`) is the design seam instead (required `decode`; `encode` defaults to an error since only the codec is an encoder, faithful to Python where only `MimiModel.encode` exists; `decode_step` defaults to one-shot). (2) **Device-agnostic** — Python defaults `device="cuda"`; Rust takes `device: &Device` (Cpu/F32 default, Metal/bf16 opt-in). (3) **CUDA graphs** — the `CUDAGraphed` wrapping in `_init_streaming_state` is GPU-only and absent in Rust (candle eager); numerically irrelevant, latency-only. (4) **SDPA, not flash** — the enc/dec transformers' `F.scaled_dot_product_attention` maps to eager matmul+mask+softmax in moshi.

## Precision / gotchas
- **`disable = device.type != 'cuda'`** (`:221`) is the single line that makes CUDA-graph capture engage only on GPU; the same checkpoint runs eager on CPU/Metal with identical math (graphs are a capture/replay latency optimization, not a numerics change).
- **`encode` length contract:** one-shot pads with `pad_for_conv1d` (`:358`); streaming **raises** on any non-multiple-of-1920 length (`:361-365`). Buffering to 1920-sample boundaries is the caller's responsibility (the 1920 = `frame_size`).
- **`decode` does not trim**, only `forward` does `out[..., :length]` (`:332`). Decoded waveform carries the codec's trailing conv padding unless the caller slices it.
- **`channel_wise` upsample bug-compat** (`upsample_channel_wise_bug=True`, `:141,216`) is intentionally preserved to reproduce the original checkpoint's behavior — not a bug to fix.
- **Codes vs EOAudio.** Mimi emits/consumes integer codes in `[0, 2047]` (cardinality 2048). The value `2048` = EOAudio is a *model-side* sentinel from LFM2-Audio's depthformer head; the processor rejects codes `>= 2048` or `< 0` before decode (`processor.py:174`). Mimi itself never produces 2048; an out-of-range code reaching the RVQ `index_select` is "a dramatic CUDA crash" per `vq.py:144`.
- **RVQ index dtype:** Rust casts codes to **u32** before `Mimi::decode`/`decode_step` because the residual-VQ codebook lookup is an `index_select` (`audio_out.rs:89,114`); Python keeps them as torch int.
- **Cross-library f32 floor:** the Rust path runs F32 on CPU (candle has no CPU bf16 matmul) / bf16 on Metal; Mimi decode is validated as a smoke test (waveform `[1,1,30720]`, peak 0.7395 — PYTHON_VS_RUST.md §1.2), not byte-exact, since the moshi-crate reuse + candle gemm/FFT ordering sit at the ~1e-6 cross-framework floor.
