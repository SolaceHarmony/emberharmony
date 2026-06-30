# conformer_modules (Rust port)
**Source:** `liquid-audio/src/model/conformer/modules.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/modules.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/conformer/modules.md`](../../../wiki/model/conformer/modules.md).

## Role
The per-layer building blocks of the FastConformer audio-in encoder in the
Rust port: `ConformerLayer` (one Macaron-style Conformer block), its two
sub-modules `ConformerFeedForward` and `ConformerConvolution`, and the
streaming-capable `CausalConv1D`. `ConformerEncoder` stacks `n_layers=17` of
these `ConformerLayer`s over the subsampled, rel-pos-encoded mel features
(`d_model=512`) to produce the contextualized acoustic encoding that the
`audio_adapter` MLP lifts to the 2048-dim backbone embedding space. This is
the body of the speech-understanding (audio-in) front-end; it does not touch
audio generation.

## How it works (Rust)

**`ConformerLayer` macaron structure** (`modules.rs:220`). One block is
`FF/2 → MHSA → Conv → FF/2 → out-norm`, all pre-LayerNorm with residual adds,
with the two feed-forwards scaled by `const FC: f64 = 0.5` (`:295`, the
half-step "Macaron" weighting). `forward_cache` (`:286`) is the full path:
1. `h = feed_forward1.forward(norm_feed_forward1.forward(residual))`; `residual
   = residual + h * FC` (`:297-298`).
2. `h = norm_self_att.forward(residual)`; `self_attn.forward_cache(h, h, h,
   att_mask, pos_emb, cache_last_channel)`; `residual = residual + h`
   (`:300-307`).
3. `h = conv.forward_cache(norm_conv.forward(residual), pad_mask,
   cache_last_time)`; `residual = residual + h` (`:309-310`).
4. `h = feed_forward2.forward(norm_feed_forward2.forward(residual))`;
   `residual = residual + h * FC` (`:312-313`).
5. `norm_out.forward(residual)` (`:315`).

`forward` (`:260`) delegates to `forward_cache(…, None, None)` (offline path).
There are **5 `LayerNorm`s per block** (`norm_feed_forward1`, `norm_self_att`,
`norm_conv`, `norm_feed_forward2`, `norm_out`), all via
`crate::model::norm::layer_norm` (the differentiable `ops::layer_norm_slow`
wrapper, eps=1e-5) — standard LayerNorm (mean + variance + affine), **not**
RMSNorm (RMSNorm appears in the LFM2 backbone/depthformer, not in the NeMo
conformer).

**Self-attention selection** (`modules.rs:215`). `enum SelfAttention { RelPos(
RelPositionMultiHeadAttention), Abs(MultiHeadAttention) }` — the Rust analog
of Python's `self_attention_model` string. For this checkpoint `rel_pos`
(`:243`), so `self_attn` is a `RelPositionMultiHeadAttention`. It receives the
rel-pos table `pos_emb` produced by the encoder's `RelPositionalEncoding`.
`n_heads=8`, `d_model=512` ⇒ `d_k=64`, scale `1/sqrt(64)`. With
`att_context_size=[-1,-1]` (unlimited) and offline batch-1, the encoder's
`_create_masks` returns `att_mask=None`/`pad_mask=None` — full bidirectional
attention. The `abs_pos` branch (`:244`) is constructible but unused;
`rel_pos_local_attn` (Longformer) is not ported.

**`ConformerConvolution`** (`modules.rs:47`). The gated convolution path:
- `x.transpose(1,2)` → `(B, d_model, T)` (`:81`).
- `pointwise_conv1: Conv1d(d_model → 2*d_model, k=1)` (`:62`).
- **GLU** over channel dim 1: `a = x.narrow(1, 0, c/2)`, `b = x.narrow(1, c/2,
  c/2)`, `x = a * sigmoid(b)` (`:84-87`) — **hand-rolled** since candle has no
  `F.glu`. Back to `d_model` channels.
- Optional pad-mask zeroing (`:89-93`; no-op offline).
- `depthwise_conv: CausalConv1D(d_model → d_model, k=conv_kernel_size=9,
  groups=d_model)` — **depthwise** (one filter per channel). For this checkpoint
  `conv_kernel_size=9`; `CausalPadding::Symmetric((k-1)/2 = 4)` (`:66`) ⇒
  **symmetric** padding `[4,4]`, so offline it is mathematically a plain
  "same"-padded depthwise conv. The `CausalConv1D` machinery is only
  asymmetric/causal when a streaming cache is threaded.
- `batch_norm: BatchNorm(d_model, eps=1e-5)` — `forward_t(x, false)` (`:96`)
  uses frozen running mean/var (inference mode).
- `silu` (`:97`) — swish, **not** GLU here; the GLU was the pointwise-1 gate,
  SiLU is the post-BN nonlinearity.
- `pointwise_conv2: Conv1d(d_model → d_model, k=1)` (`:68`).
- `x.transpose(1,2)` back to `(B, T, d_model)` (`:99`).

**`ConformerFeedForward`** (`modules.rs:19`). Plain 2-layer MLP
`Linear(d_model→d_ff) → SiLU → Linear(d_ff→d_model)` (`:32-36`). `d_ff =
d_model * 4 = 2048`. Activation is `silu` (`:34`) — **not** GLU/SwiGLU; the
conformer FF is a simple SiLU-MLP.

**`CausalConv1D`** (`modules.rs:130`). Wraps a `padding=0` `Conv1d` plus
manual `pad_with_zeros` (`:170`), with a `CausalPadding` enum (`:115`)
mirroring Python's `None`/`int`/`[l,r]` cases:
- `Causal` ⇒ `left = k-1`, `right = stride-1`.
- `Symmetric(p)` ⇒ `left == right == p` (the offline conformer's case).
- `Asymmetric(l, r)` ⇒ requires `l + r == k-1`, stride 1.

`update_cache` (`:170`): offline (`cache=None`) pads left+right; streaming
(`cache=Some`) pads right only, prepends the cache via `cat`, rolls the window
back to the cache length (dropping `cache_drop_size` trailing steps, `:179`).
`forward` (`:190`) = `update_cache` then the `padding=0` conv.

**Streaming caches in `ConformerLayer`** (`modules.rs:286`). When
`cache_last_channel`/`cache_last_time` are `Some`, `forward_cache` returns
`(x, next_channel, next_time)`; the attention/conv return their next caches.
Offline LFM2-Audio passes both `None`, so the next caches are `None` — the
entire streaming path is cold.

## Dtypes & shapes (Rust)
All compute is in **model dtype** (Rust CPU f32, Metal bf16). `LayerNorm`/
`BatchNorm` normalize statistics in f32 internally then cast back.
`d_model=512`, `d_ff=2048`, `n_heads=8`, `d_k=64`, `conv_kernel_size=9`.

| Tensor | dtype | shape |
|---|---|---|
| `ConformerLayer` input `x` | model | `(B, T', 512)` (T' = subsampled frames) |
| `pos_emb` (rel-pos table) | model | `(1, 2T'−1, 512)` |
| `att_mask` / `pad_mask` (offline) | — | `None` (unlimited ctx, batch-1) |
| `ConformerLayer` output `x` | model | `(B, T', 512)` |
| FF internal hidden | model | `(B, T', 2048)` |
| Conv after pointwise1 | model | `(B, 1024, T')` |
| Conv after GLU | model | `(B, 512, T')` |
| Conv depthwise out (symmetric pad) | model | `(B, 512, T')` |
| `cache_last_time` (streaming only) | model | `(B, 512, T_cache)` |
| `cache_last_channel` (streaming only) | model | `(B, T_cache, 512)` |

`LayerNorm`/`BatchNorm` promote to f32 for the mean/variance reduction and
`rsqrt`, then cast to model dtype. No int/u32 codes here (this is the audio-in
understanding path, upstream of any RVQ/codes).

## Wiring (Rust)
**Upstream** — fed by:
- `model/conformer/subsampling.rs`: the `dw_striding` `ConvSubsampling` 8×
  downsamples the `(B,128,T_mel)` mel into `(B, T', 512)`. See
  [`glm-version/model/conformer/subsampling.md`](subsampling.md).
- `model/conformer/mha.rs`'s `RelPositionalEncoding`, which (in
  `model/conformer/encoder.rs`) adds positional info to the subsampled features
  and emits the `pos_emb` table `(1, 2T'−1, 512)` consumed by every layer's
  `RelPositionMultiHeadAttention`. See
  [`glm-version/model/conformer/mha.md`](mha.md).
- `model/conformer/encoder.rs`: owns the `Vec<ConformerLayer>` of 17 layers,
  threads `att_mask`/`pad_mask`/`pos_emb`, and (offline) calls each layer with
  `None` caches. See [`glm-version/model/conformer/encoder.md`](encoder.md).

**Downstream** — consumes this output:
- `model/conformer/encoder.rs`: collects the stacked-layer output, transposes
  to `(B, 512, T')`, and (no `out_proj`, `feat_out=-1`) returns `(B, T', 512)`.
- `model/mlp.rs`: the `audio_adapter` MLP `Linear(512→2048) → GELU(erf) →
  Linear(2048→2048)` lifts the `_feat_out=512` encoding to backbone width
  `(ΣT', 2048)`. See [`glm-version/model/mlp.md`](model/mlp.md).
- `model/lfm2_audio.rs`: the adapted `audio_in_emb` is modality-scattered into
  the prefill sequence and fed to the LFM2 backbone. See
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).

## Python ↔ Rust — where the port differs

| Python (`modules.py`) | Rust (`modules.rs`) | Difference | Why |
|---|---|---|---|
| `ConformerLayer.__init__/forward` | `ConformerLayer::new` / `forward` + `forward_cache` (`:232-316`) | **split offline/forward_cache** | Rust splits `forward` (passes `None` caches) from `forward_cache` (threads `cache_last_channel`/`cache_last_time`), matching the Python tuple-return contract. `fc_factor` is `const FC: f64 = 0.5` (`:295`). |
| `self_attn` dispatch by string | `enum SelfAttention { RelPos, Abs }` (`:215`) | **deliberate: string → enum** | Rust's enum dispatch is the idiomatic analog of Python's `if self_attention_model == "rel_pos"`. Only `rel_pos` is wired for this checkpoint; `abs_pos` constructible but unused; `rel_pos_local_attn`/Longformer not ported (off-path). |
| `F.glu(x, dim=1)` | hand-rolled `a * sigmoid(b)` via `narrow` on dim 1 (`:84-87`) | **deliberate: hand-rolled GLU** | candle has no `F.glu`. The `narrow`-then-multiply reproduces `a * sigmoid(b)` exactly. |
| `nn.SiLU()` | `candle_nn::ops::silu` (`:34`, `:97`) | identical | — |
| `nn.LayerNorm(d_model)` | `crate::model::norm::layer_norm(d_model, 1e-5, …)` | **deliberate: differentiable wrapper** | `candle_nn::LayerNorm`'s fused `forward` severs autograd (the conformer is trained); the port's `norm.rs` wrapper calls `ops::layer_norm_slow`. Same forward output. See [`glm-version/model/mlp.md`](model/mlp.md). |
| `nn.BatchNorm1d(d_model)` | `candle_nn::BatchNorm` with `forward_t(x, false)` (`:96`) | **deliberate: inference mode** | `forward_t(x, false)` uses frozen running mean/var (inference), not batch statistics. A batch-1 BN in training mode would normalize each frame to ~0 and corrupt the encoding. |
| `nn.Conv1d` (pointwise, k=1) | `candle_nn::conv1d` with `Conv1dConfig { padding: 0, stride: 1, groups: 1 }` (`:58`) | identical | — |
| `CausalConv1D` (`nn.Conv1d` subclass, padding=0 + manual `F.pad`) | `CausalConv1D` struct wrapping a `padding=0` `Conv1d` + `pad_with_zeros` (`:130-199`) + `CausalPadding` enum | **deliberate: subclass → struct + enum** | Rust has no `nn.Conv1d` subclass; the struct holds the inner conv + padding config. `CausalPadding` enum mirrors Python's `None`/`int`/`[l,r]`. `update_cache`/`forward`/`set_cache_drop_size` are 1:1. |
| `reset_parameters_ff`/`reset_parameters_conv` (training-time init) | no-op stubs (`:41`, `:110`) | **deliberate: no-op** | the port loads pretrained weights via `VarBuilder`; init is dead at inference. Kept for the 170/170 symbol inventory. |
| `scaled_dot_product_attention` (CUDA-gated, inside rel-pos MHA) | hand-rolled eager SDPA in `mha.rs` | **deliberate: kernel-free** | §2.2. See [`glm-version/model/conformer/mha.md`](mha.md). |
| cache-aware streaming/ONNX-export apparatus | cold off-path (`forward_cache` threads `None` caches) | **deliberate: off-path** | §2.5. The offline conformer passes both `None` caches. |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. f32 on CPU; bf16 on Metal. |

**Parity:** conformer layer 0 = 1.056e-6, final = 8.25e-7 vs Python golden
tensors (PARITY.md).

## Precision / gotchas (Rust-specific)
- **`LayerNorm` not RMSNorm here.** This NeMo conformer uses standard
  `LayerNorm` (5 per block, via `crate::model::norm::layer_norm` — the
  differentiable `ops::layer_norm_slow` wrapper) and `BatchNorm1d` in the conv.
  The f32-multiply-order RMSNorm subtlety (§2.4) applies to the LFM2
  backbone/depthformer, **not** to these modules. Do not "fix" the conv
  BatchNorm into a LayerNorm.
- **`LayerNorm` is the differentiable wrapper, not `candle_nn::LayerNorm`.**
  `candle_nn::LayerNorm`'s fused `forward` severs autograd (the conformer is
  trained); the port's `norm.rs` wrapper calls `ops::layer_norm_slow`. Same
  forward output; only the backward path differs.
- **Symmetric vs causal conv.** The depthwise conv is named `CausalConv1D` but
  is configured `CausalPadding::Symmetric((k-1)/2 = 4)` (`:66`) for the offline
  non-causal encoder. It is *not* causal in LFM2-Audio's forward; the
  causal/asymmetric and `cache_drop_size` logic only activates when a streaming
  `cache_last_time` is threaded, which the offline path never does.
- **Two distinct gates.** `ConformerConvolution` has a GLU (pointwise-1 →
  `a*sigmoid(b)`, hand-rolled at `:84-87`) *and* a SiLU (post-BatchNorm, `:97`).
  The FF modules use SiLU only (no GLU/SwiGLU). Easy to conflate; they are
  different ops at different positions.
- **`fc_factor = 0.5` on both FFs** is load-bearing (Macaron half-steps); the
  `const FC: f64 = 0.5` (`:295`) applies it. Dropping it doubles the FF
  contribution.
- **BatchNorm running stats.** Inference must use frozen running mean/var
  (`forward_t(x, false)`, `:96`), not batch statistics — a batch-1 BN in
  training mode would normalize each frame to ~0 and corrupt the encoding.
- **GLU is hand-rolled.** `a = x.narrow(1, 0, c/2)`, `b = x.narrow(1, c/2, c/2)`,
  `x = a * sigmoid(b)` (`:84-87`). candle has no `F.glu`; the `narrow`-then-
  multiply is the faithful equivalent.
- **`CausalPadding` enum.** `Causal`/`Symmetric(p)`/`Asymmetric(l,r)` (`:115`)
  — the Rust analog of Python's `padding=None`/`int`/`[l,r]`. The conformer uses
  `Symmetric(4)`. Striding is rejected for non-symmetric padding (`:151-153`).
- **Offline masking contract.** With batch-1 unlimited-context, masks are
  `None` and attention is fully bidirectional. A padded multi-clip batch would
  require the full `_create_masks` port (§5.1) — a documented gap, not a bug.
- **Cross-library f32 floor** (~1e-6): candle gemm reduction order and libm
  transcendentals differ last-bit from torch; the conformer agrees to that
  floor (8.25e-7 final), the irreducible limit of any cross-framework port.

## Cross-references
- [`wiki/model/conformer/modules.md`](../../../wiki/model/conformer/modules.md)
  — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.2 (kernel-free SDPA), §2.5 (off-path
  streaming stubs).
- `liquid-audio/parity/PARITY.md` — conformer layer 0 1.056e-6, final
  8.25e-7.