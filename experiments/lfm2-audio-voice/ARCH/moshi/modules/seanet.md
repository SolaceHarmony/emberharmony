# moshi_seanet
**Code:** `MO01` · **Source:** `moshi/modules/seanet.py` · **Rust:** `moshi crate seanet` · **On the LFM2-Audio inference path:** yes

## Role
`SEANetEncoder` / `SEANetDecoder` are the convolutional waveform⇄latent ends of the Mimi codec. The encoder strides raw 24 kHz audio down to a 25 Hz, 512-dim continuous latent (the thing the RVQ then quantizes into 8-codebook codes); the decoder is its near-mirror, mapping a 25 Hz latent back to 24 kHz samples. `SEANetResnetBlock` is the dilated-residual unit both share. They are pure causal conv stacks — no attention, no quantization — sandwiched between Mimi's encoder/decoder transformers and the conv-framerate resamplers (`moshi_compression`). On the LFM2-Audio path only the **decoder** runs at inference (codes → waveform via `mimi.decode`); the encoder runs at training-data prep (`mimi.encode` to mint `audio_out` targets).

## How it works

**Channel-doubling downsample ladder (encoder, `seanet.py:169-236`).** With `_seanet_kwargs` (dim=512, n_filters=64, ratios `[8,6,5,4]`), the encoder is `n_blocks = len(ratios)+2 = 6` blocks. Note `self.ratios = list(reversed(ratios))` (`:154`) — the encoder downsamples in **reverse** of the listed (decoder-order) ratios, so it actually strides `4 → 5 → 6 → 8`. Structure:
1. init `StreamingConv1d(channels=1 → 64, k=7)` (`:170-178`).
2. for each ratio: `n_residual_layers=1` `SEANetResnetBlock`s at the current width, then `ELU`, then a strided `StreamingConv1d(mult·64 → mult·128, k=2·ratio, stride=ratio)` (`:204-216`). `mult` doubles each block: channels go `64→128→256→512→1024`.
3. final `ELU` + `StreamingConv1d(1024 → dimension=512, k=last_kernel_size=3)` (`:221-234`).

The total stride is `hop_length = ∏ratios = 8·6·5·4 = 960` (`:157`), i.e. 24000/960 = **25 Hz** at the encoder output. (Mimi's extra `ConvDownsample1d` after the encoder transformer takes 25 → 12.5 Hz; that is *not* in SEANet.)

**Resnet block (`seanet.py:38-93`).** `hidden = dim // compress` (compress=2). With `kernel_sizes=[residual_kernel_size, 1]=[3,1]` and `dilations=[2**j, 1]`, the branch is `ELU → Conv1d(dim→hidden, k=3, dilation=2^j) → ELU → Conv1d(hidden→dim, k=1)` (`:59-75`). `true_skip=True` ⇒ `shortcut = nn.Identity()` (`:78`); forward is `shortcut(x) + block(x)` with a shape assert (`:91-92`). Because `n_residual_layers=1`, only `j=0` ⇒ dilation 1 in this checkpoint, so the dilation ladder is degenerate here (it would grow `dilation_base**j` for deeper stacks).

**Activation = ELU, not GELU/SiLU.** `act = getattr(nn, "ELU")` with `alpha=1.0` (`:56, :167`): `ELU(x) = x if x>0 else (e^x − 1)`. This is the only nonlinearity in SEANet; pre-activation ordering (`ELU` *before* each conv).

**Convolutions are causal, weight-norm folded.** `norm="none"` (`loaders.py:53`): training used `weight_norm` but the export folds it into a plain `nn.Conv1d`, so inference loads ordinary conv weights — `apply_parametrization_norm` is a no-op for `"none"` (`conv.py:42-45`). All convs are `causal=True` with `pad_mode="constant"` (`loaders.py:41,54`). The causal pad amount is `_padding_total = _effective_kernel_size − stride` where `_effective_kernel_size = (k−1)·dilation + 1` (`conv.py:223-231`); for a causal conv this is applied entirely on the **left** (zeros, since pad_mode constant), so output length stays causal and, for the strided downsamplers, `T_out = T_in / stride` (the forward asserts `T % stride == 0`, `conv.py:248`). `disable_norm_outer_blocks=0`, so no block-specific norm disabling matters here.

**Decoder is the inverse ladder (`seanet.py:315-388`).** `self.ratios = ratios` (NOT reversed — `:300`), `mult = 2**len(ratios) = 16`. init `StreamingConv1d(dim=512 → 16·64=1024, k=7)`; then per ratio: `ELU → StreamingConvTranspose1d(mult·64 → mult·64//2, k=2·ratio, stride=ratio)` (the upsample), then the residual block(s) at the halved width; `mult //= 2`. Final `ELU → StreamingConv1d(64 → channels=1, k=last_kernel_size=3)` (`:371-382`), optional `final_activation` (None here, so no tanh). The transpose convs invert the 960× downsample back to 24 kHz; `trim_right_ratio=1.0` trims all causal overhang on the right.

**Streaming state (the real-time decode contract).** Both are `StreamingContainer`s; the per-conv `_StreamingConv1dState` keeps a `previous` ring buffer of `kernel − stride` samples (`conv.py:240-243`) so successive `decode_step` calls concatenate prior context (`x = cat([state.previous, x], -1)`, `conv.py:261`) — this is what makes gapless per-frame decode work. The decoder transpose conv keeps a `partial` overlap-add buffer (`_StreamingConvTr1dState`). One Mimi frame in = 1920 samples out per decode step.

## Dtypes & shapes

| Stage | Input | Output |
|---|---|---|
| `SEANetEncoder.forward` (training prep) | waveform `(B,1,T)` f32 @ 24 kHz | latent `(B,512,T/960)` model dtype @ 25 Hz |
| `SEANetResnetBlock` | `(B,C,t)` | `(B,C,t)` (identity skip + branch, same dtype) |
| `SEANetDecoder.forward` (inference) | latent `(B,512,t)` model dtype @ 25 Hz | waveform `(B,1,t·960)` **f32** @ 24 kHz |
| `decode_step` (streaming, 1 frame) | latent `(1,512,1)` | waveform `(1,1,1920)` f32 |

Compute dtype is the codec module dtype: **Python** bf16/f32 on CUDA; **Rust** f32 on CPU (no CPU bf16 matmul), bf16 on Metal. Weights are bf16 on disk, folded weight-norm. No int / quantization happens inside SEANet — the integer RVQ codes live one stage over in `moshi_vq`/`moshi_core_vq`. No f32/f64 upcasts inside SEANet (no norm, no softmax); the precision-sensitive f64 work is the mel front-end, not here.

## Wiring

**Encoder (training-prep path):**
- Upstream: raw waveform `(B,1,T)` f32 @ 24 kHz from `mimi.encode`, fed by [moshi_compression](../models/compression.md) (`MimiModel.encode`, which `pad_for_conv1d`s to a multiple of frame_size first).
- Downstream: latent `(B,512,T/960)` model dtype @ 25 Hz → Mimi's encoder transformer → `ConvDownsample1d` ([moshi_resample](resample.md), 25→12.5 Hz) → [moshi_vq](../quantization/vq.md) `SplitResidualVectorQuantizer` (512→256 proj, cdist-argmin RVQ) → codes `(B,8,·)`.

**Decoder (inference path):**
- Upstream: latent `(B,512,t)` model dtype @ 25 Hz from `quantizer.decode` → `ConvTrUpsample1d` ([moshi_resample](resample.md), 12.5→25 Hz) → decoder transformer, all orchestrated by [moshi_compression](../models/compression.md) (`MimiModel.decode` / `_decode_frame`).
- Downstream: waveform `(B,1,t·960)` f32 @ 24 kHz → back to [moshi_compression](../models/compression.md) which returns it to [core_processor](../../processor.md) `decode()` and ultimately the demo audio sink ([demo_chat](../../../demo/chat.md), which plays each `(1,1,1920)` chunk inside `mimi.streaming(1)`).

The 8-code frames that feed the decode side originate at [model_lfm2_audio](../../model/lfm2_audio.md)'s depthformer audio head (`frame (8,)` int, codes 0..2047, 2048=EOAudio) — but those go through the quantizer/resampler first, not directly into SEANet.

## Python ↔ Rust

| Python (`seanet.py`) | Rust (`moshi-0.6.4 src/seanet.rs`) |
|---|---|
| `SEANetResnetBlock` | `SeaNetResnetBlock` (block `Vec<StreamableConv1d>`, optional `shortcut`, `StreamingBinOp::Add` skip) |
| `SEANetEncoder` | `SeaNetEncoder` (`init_conv1d`, `layers: Vec<EncoderLayer{residuals,downsample}>`, `final_conv1d`) |
| `SEANetDecoder` | `SeaNetDecoder` (`init_conv1d`, `layers: Vec<DecoderLayer{upsample,residuals}>`, `final_conv1d`, `final_activation`) |
| `nn.Sequential(model)` flat list | reorganized into explicit `EncoderLayer`/`DecoderLayer` structs (same op order, named members) |
| `getattr(nn, "ELU")(alpha=1.0)` | `candle_nn::Activation::Elu(1.0)` config field |
| `StreamingConv1d`/`StreamingConvTranspose1d` | `StreamableConv1d`/`StreamableConvTranspose1d` ([moshi_conv](conv.md)) |
| `self.ratios = reversed(ratios)` (encoder) | `cfg.ratios.iter().rev()` in `SeaNetEncoder::new` (config keeps decoder-order ratios) |
| forward `self.model(x)` | `impl Module::forward` + `impl StreamingModule::step` (explicit streaming half) |

**Config provenance.** Rust `mimi.rs::Config::v0_1` hardcodes the same hyperparameters as Python's `_seanet_kwargs` (dimension 512, n_filters 64, ratios `[8,6,5,4]`, kernel 7, residual_kernel 3, last_kernel 3, n_residual_layers 1, compress 2, dilation_base 2, causal true, pad_mode `Constant`, true_skip true, disable_norm_outer_blocks 0, final_activation None). One Rust-only field, `lstm: 0` — the moshi crate's SEANet supports an optional LSTM bottleneck and **bails if `lstm > 0`**; LFM2's Mimi sets it 0, so the path is identical to Python (which has no LSTM at all).

**Deliberate divergences** (PYTHON_VS_RUST.md): §2.3 upstream reuse — the whole vendored `liquid_audio/moshi/**` is *not* re-ported; Rust reuses Kyutai's published `moshi` crate (0.6.4, pins candle ^0.9.1 = repo's 0.9.2), chosen because this checkpoint uses `quantizer.rvq_first`/`rvq_rest` weight naming that candle-transformers' Mimi can't load (§2.3, ARCH_1 §6-7). §2.1 device-agnostic: Python is CUDA-coupled (defaults `device="cuda"`, codec wrapped in `CUDAGraphed`), Rust is `Device`-parametric with no graph layer (eager candle conv ops; CUDA graphs are latency-only, numerically irrelevant). §2.2: the codec uses no bespoke CUDA kernels even in Python — stock torch conv + SDPA — so there is nothing kernel-shaped to reimplement in SEANet itself.

## Precision / gotchas

- **Reversed encoder ratios.** `SEANetEncoder` reverses `ratios` internally (`:154`); the decoder does not (`:300`). Config carries *decoder-order* `[8,6,5,4]` for both. Mis-reading this flips the stride schedule — the Rust honors it with `.iter().rev()` only in the encoder.
- **`true_skip=True` is load-bearing for weight keys.** Identity shortcut means no `shortcut.*` conv weights exist in the residual blocks; the Rust sets `shortcut = None`. A wrong `true_skip` would expect/skip a weight tensor and break `load_state_dict(strict=True)`.
- **`hop_length=960`, not 1920.** SEANet alone is 24000/960 = 25 Hz. The codec's 1920-sample / 12.5 Hz frame is SEANet's 960 **times** the `ConvDownsample1d` stride-2 ([moshi_resample](resample.md)). Don't attribute the 1920 to SEANet.
- **No norm inside SEANet** (`norm="none"`, weight-norm folded at export) — so there is no RMSNorm-vs-LayerNorm bf16 ordering subtlety here (that gotcha lives in the backbone/depthformer). The only numeric floor is conv/matmul reduction order (candle gemm vs torch BLAS, ~1e-6 / f32 round-off; PYTHON_VS_RUST §1.4) — SEANet has no transcendental-heavy op (only ELU's `exp` on negatives), so it sits well within the cross-library floor. Mimi decode is smoke-verified end-to-end (waveform `[1,1,30720]`, peak 0.7395; PYTHON_VS_RUST §1.2).
- **Causal left-pad only.** `pad_mode="constant"` + causal ⇒ zeros on the left, never reflect — a `reflect` default (torch's general conv default) would be wrong here and corrupt the streaming `previous`-buffer contract.
- **EOAudio (2048) never reaches SEANet.** The 2048 sentinel is a *code-space* special token handled upstream in the LM/quantizer; SEANet only ever sees the dequantized continuous latent, so codes 0..2047 are already gone by the time the decoder runs.
