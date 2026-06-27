# moshi_resample
**Code:** `MO04` · **Source:** `moshi/modules/resample.py` · **Rust:** `moshi crate` · **On the LFM2-Audio inference path:** yes

## Role
`ConvDownsample1d` / `ConvTrUpsample1d` are the **frame-rate bridge** inside the Mimi codec: a pair of learnt strided (transposed) convolutions that convert the SEANet encoder's latent stream between the encoder frame rate (25 Hz) and the codec's overall quantizer frame rate (12.5 Hz). The downsample sits *after* the encoder transformer and *before* the split-RVQ (so the quantizer codes the stream at 12.5 Hz); the upsample sits *after* RVQ decode and *before* the decoder transformer (back up to 25 Hz). They exist because Mimi deliberately codes at half the SEANet rate (one 8-code column per 1920 audio samples) while the SEANet front-end runs at 960-sample hops — the stride-2 conv is the learnable rate change between those two grids.

## How it works
**Instantiation (the only config that matters for LFM2-Audio).** Mimi is built with `resample_method="conv"` (`loaders.py:320`), `encoder_frame_rate = 24000 / hop_length = 24000/960 = 25 Hz`, `frame_rate = 12.5 Hz`, `causal=True`, `upsample_channel_wise_bug=True` (the `MimiModel` default, `compression.py:141`). `MimiModel.__init__` (`compression.py:189-217`) computes `downsample_stride = encoder_frame_rate / frame_rate = 2`, asserts it is an integer and that `encoder_frame_rate > frame_rate` ("Cannot upsample with conv"), sets `learnt = (resample_method == "conv") = True`, and builds:
- `self.downsample = ConvDownsample1d(stride=2, dimension=512, learnt=True, causal=True)` — `channel_wise` defaults `False`.
- `self.upsample   = ConvTrUpsample1d(stride=2, dimension=512, learnt=True, causal=True, channel_wise=True)`.

So **the non-learnt branches in `resample.py` are dead code for this checkpoint** — they never run. The only live config is `learnt=True`.

**ConvDownsample1d (`resample.py:14-65`).** Wraps a single `StreamingConv1d(in=512, out=512, kernel=2*stride=4, stride=2, causal=True, groups=1, bias=False, pad_mode="replicate")` (`resample.py:43-52`). Because `learnt=True` and `channel_wise=False`, `groups=1` — it is a **full (dense) 512→512 strided conv**, not depthwise. `forward` (`resample.py:58-65`): for the learnt path it does **not** rearrange — it feeds `(B, 512, T@25Hz)` straight through the conv, producing `(B, 512, T/2 @12.5Hz)`. (The `rearrange "b c t -> (b c) () t"` and its inverse only execute in the non-learnt branch, where the conv is a fixed 1-channel box filter applied per-channel.) Kernel = `2*stride` so each output frame integrates a 4-wide window with 50% overlap; `causal=True` + `pad_mode="replicate"` means the streaming conv left-pads `padding_total = kernel - stride = 2` with edge replication (no peeking at the future). The non-learnt branch (not used) would freeze the weight to `1/(2*stride) = 0.25` — a plain averaging downsample (`resample.py:53-56`).

**ConvTrUpsample1d (`resample.py:68-119`).** Wraps `StreamingConvTranspose1d(in=512, out=512, kernel=4, stride=2, causal=True, groups=dimension=512, bias=False)` (`resample.py:95-103`). With `learnt=True, channel_wise=True`, `groups=512` ⇒ **depthwise (channel-wise) transposed conv** — each of the 512 channels is upsampled independently by its own length-4 kernel. `forward` (`resample.py:109-119`): learnt path again **skips rearrange and skips normalization** — it is just `y = self.convtr(x)`, mapping `(B,512,T/2@12.5Hz) → (B,512,T@25Hz)`. The `x_for_normalization = ones_like; normalization = convtr(ones); y = y / normalization` block (`resample.py:114-117`) is the overlap-add edge-correction for the **non-learnt** averaging upsampler and **does not run** here. The "bug" in `upsample_channel_wise_bug` is the historical name for forcing `groups=dimension` (depthwise) on the upsampler — kept `True` for weight-compat with the trained checkpoint, hence "bug-compat".

**Where they fire on the signal path.** Encode: `encoder → encoder_transformer → _to_framerate → downsample → quantizer.encode` (`compression.py:314, 373`; `_to_framerate` at `compression.py:267-278` dispatches to `self.downsample` since `resample_method != "interpolate"`). Decode: `quantizer.decode → _to_encoder_framerate → upsample → decoder_transformer → decoder` (`compression.py:324, 417`; `_to_encoder_framerate` at `compression.py:280-291`). Both `_to_*framerate` helpers short-circuit (return `x` unchanged) iff `encoder_frame_rate == frame_rate`; for Mimi they always run.

**Streaming state.** As a `StreamingConv1d`/`StreamingConvTranspose1d`, each holds a `state_prev_xs` ring of the last `kernel-stride` samples so that per-frame `step()` calls produce the same result as the one-shot `forward()` (gapless real-time decode). `reset_state` clears it at a turn boundary. This is what lets the demo decode exactly one 8-code frame at a time inside `mimi.streaming(1)`.

## Dtypes & shapes
| Tensor | dtype | shape |
|---|---|---|
| **Downsample in** (post encoder-transformer latent) | model dtype (Python cuda/bf16; Rust CPU f32 / Metal bf16) | `(B, 512, T@25Hz)` |
| **Downsample out** (→ RVQ) | model dtype | `(B, 512, T/2 @12.5Hz)` |
| downsample weight | bf16 on disk / f32 CPU / bf16 Metal | `(512, 512, 4)` (dense, groups=1) |
| **Upsample in** (post RVQ-decode latent) | model dtype | `(B, 512, T'@12.5Hz)` |
| **Upsample out** (→ decoder transformer) | model dtype | `(B, 512, 2·T' @25Hz)` |
| upsample weight | bf16 / f32 / bf16 | `(512, 1, 4)` (depthwise, groups=512) |

No internal dtype promotion — these are plain conv/transposed-conv in the codec's compute dtype. No f32 norm upcast (there is no normalization layer here), no softmax, no f64. `bias=False` throughout. The latent values are continuous (not codes); the int `u32` codes only exist *between* downsample-out→RVQ and RVQ→upsample-in, i.e. these convs never touch integer code tensors.

## Wiring
**Upstream (encode side):** `[moshi_transformer](transformer.md)` (the Mimi `encoder_transformer`, `ProjectedTransformer`) emits the 25 Hz latent `(B,512,T@25Hz)` model-dtype → **ConvDownsample1d**. Ultimately fed by `[moshi_seanet](seanet.md)` `SEANetEncoder` (waveform→`(B,512,T@25Hz)`). Orchestrated by `[moshi_compression](../models/compression.md)` `MimiModel.encode/_to_framerate`.

**Downstream (encode side):** ConvDownsample1d out `(B,512,T/2@12.5Hz)` model-dtype → `[moshi_vq](../quantization/vq.md)` `SplitResidualVectorQuantizer.encode` (which input-projects 512→256, runs RVQ, emits int codes).

**Upstream (decode side):** `[moshi_vq](../quantization/vq.md)` `.decode` emits latent `(B,512,T'@12.5Hz)` model-dtype → **ConvTrUpsample1d**.

**Downstream (decode side):** ConvTrUpsample1d out `(B,512,2·T'@25Hz)` model-dtype → `[moshi_transformer](transformer.md)` (Mimi `decoder_transformer`) → `[moshi_seanet](seanet.md)` `SEANetDecoder` → waveform `(B,1,T)`@24kHz. Top-level consumer is `[moshi_compression](../models/compression.md)` `MimiModel.decode/decode_step`, which is what `[core_processor](../../processor.md)` and `[demo_chat](../../../demo/chat.md)` call to turn generated 8-code frames into 24 kHz audio.

## Python ↔ Rust
The whole module is **reused, not re-ported**: `liquid-audio-rs` consumes Kyutai's published `moshi` crate (`moshi-0.6.4`) rather than re-implementing the vendored Python codec (PYTHON_VS_RUST.md §2.3 "Mimi codec → `moshi::mimi`", §4 "out of scope / reused").

| Python (`resample.py`) | Rust (`moshi-0.6.4/src/conv.rs`) |
|---|---|
| `ConvDownsample1d` | `conv::ConvDownsample1d` (`conv.rs:505`) |
| `ConvTrUpsample1d` | `conv::ConvTrUpsample1d` (`conv.rs:558`) |
| `StreamingConv1d` (inner) | `conv::StreamableConv1d` (`conv.rs:227`) |
| `StreamingConvTranspose1d` (inner) | `conv::StreamableConvTranspose1d` (`conv.rs:374`) |
| `forward` | `impl Module::forward` + streaming `step` |
| construction in `compression.py:202/211` | `mimi.rs:143-156` (`downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize`) |

**Deliberate divergences:**
1. **Learnt-only.** The Rust `ConvDownsample1d::new` / `ConvTrUpsample1d::new` hard-`bail!("only learnt=true is supported")` (`conv.rs:517-519, 570-572`). Faithful for this checkpoint (LFM2-Audio's Mimi is always `resample_method="conv"` ⇒ `learnt=True`); the non-learnt averaging branches and their normalization division simply do not exist in Rust. The dead Python normalization (`y/normalization`, `resample.py:114-117`) therefore has no Rust counterpart by design.
2. **Channel-wise depthwise transposed conv → materialized block-diagonal dense conv.** Python `groups=dimension=512` depthwise transposed conv. The Rust `NormConvTranspose1d` (`conv.rs:144-150`) detects `groups == out_c && in_c == out_c` and rewrites the `(512,1,4)` depthwise weight into a `(512,512,4)` block-diagonal weight via an identity-matrix multiply, then runs `conv_transpose1d` with `groups=1`. Numerically identical to torch's grouped transposed conv; this is a candle-op workaround for grouped `conv_transpose1d` quirks (and a noted Metal stability fix, `conv.rs:176-181`), per PYTHON_VS_RUST.md §2.2 "custom CUDA kernels → portable candle ops".
3. **conv_layout.** The moshi crate carries the latent channels-last `(B,T,C)` internally (the `xs.dims3() = (_b,_t,_c)` in `StreamableConv1d::forward`, `conv.rs:287`) and transposes around the raw conv; the Python keeps `(B,C,T)`. Pure layout bookkeeping inside the streaming conv — the resample wrapper's externally-visible contract is unchanged.
4. **Device/dtype.** Python pins cuda/bf16; Rust is device-agnostic (CPU→f32 since candle has no CPU bf16 matmul, Metal→bf16), PYTHON_VS_RUST.md §2.1. Numerically irrelevant for these convs beyond the f32-vs-bf16 floor.

## Precision / gotchas
- **`upsample_channel_wise_bug=True` is load-bearing, not optional.** It is the historical name for "make the upsampler depthwise (`groups=dimension`)". The trained weights were fit with it `True`, so the `(512,1,4)` depthwise shape is mandatory for `load_state_dict(strict=True)` to succeed. Flipping it would change the weight shape and break loading — "bug-compat" means reproduce it exactly.
- **The learnt path bypasses normalization.** Easy to mis-read `resample.py`: the `y / normalization` overlap-add correction and the `rearrange` reshapes are **inside `if not self.learnt`** and never execute for LFM2-Audio. Do not port them as if they were on-path.
- **Causal + replicate padding.** Downsample uses `pad_mode="replicate"` and `causal=True`; left-pad = `kernel - stride = 2` replicated edge samples. There is no center/reflect padding here. The transposed upsample uses no input padding (its causal trimming is handled by the streaming conv-transpose), and `bias=False` everywhere — no bias to fold or zero-init.
- **No f64 / no softmax / no RMSNorm here.** Unlike the precision-sensitive mel front-end (f64) or the attention/norm layers (f32 upcast), this module is just two convs in the codec compute dtype; the only numerical floor is the cross-library gemm/conv rounding (~f32 epsilon), and the depthwise→dense rewrite is exact. The codes that flow immediately downstream are int (`u32` in Rust, `[0,2047]`), but the resample convs themselves operate only on the continuous latent, never on the integer codes or the `2048=EOAudio` sentinel.
- **One frame = stride·1920/2 = 1920 samples.** The stride-2 at 25↔12.5 Hz is exactly what makes one quantizer column correspond to `frame_size = 24000/12.5 = 1920` waveform samples — the unit the streaming decoder emits per step.
