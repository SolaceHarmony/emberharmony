# model_transformer (Rust port)
**Source:** `liquid-audio/src/model/transformer.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/transformer.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/transformer.md`](../../../wiki/model/transformer.md).
> The original documents the Python `transformer.py` depthformer stack; this
> documents the Rust port in `liquid-audio/src/model/transformer.rs` and
> where it deliberately diverges from the source.

## Role
Identical purpose: the **depthformer** — the small autoregressive transformer
that turns one backbone hidden vector into an 8-codebook audio frame, one
codebook at a time. Defines the reusable sequence-model primitives (`RmsNorm`,
`Glu`, `Mha` with `BoundedAttention`, `StandardBlock`, `SharedEmbedding`,
`RawLmBackbone`) and the `SequenceModel` trait. LFM2-Audio instantiates one
`RawLmBackbone` here: a **6-layer × 1024-dim** depthformer driven inside
`_sample_audio_frame`. It exists because the main LFM2 backbone only emits a
per-step hidden + a text logit; the RVQ audio codes need their own depth-wise
causal model sampled coarse-to-fine within a single time step.

## How it works (Rust)

**Stack shape.** `RawLmBackbone::new(layers, embedding, dim)` (`transformer.rs:594`)
holds a `Vec<StandardBlock>` with `embedding: Option<SharedEmbedding> = None`
(the depthformer path uses `has_embedding=False`; the eight per-codebook
`SharedEmbedding`s live in `lfm2_audio.rs` as `depth_embeddings`). Each block's
`Mha` is built with `HeadStyle::Gqa`, `gqa_dim=8`, `num_heads=32`, `head_dim =
1024/32 = 32`, `qk_layernorm=true`, `norm_eps=1e-5`, `theta=1e6`,
`max_seq_len=128_000`. Query heads = 32, KV heads = 8 → GQA repeat factor 4.

**`StandardBlock`** (`transformer.rs:492`) is pre-LN with two residuals:
```rust
let h = (self.operator.forward(&self.operator_norm.forward(x)?, cache)? + x)?;
let h_glu = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
h + h_glu
```
Both norms are `RmsNorm`; `feed_forward` is the `Glu`. The streaming variant
`forward_cached` (`transformer.rs:527`) threads a per-layer `LayerCache` through
the operator and returns the updated cache.

**`RmsNorm`** (`transformer.rs:109`). The exact op order in `forward`
(`transformer.rs:135`) is the load-bearing detail:
```rust
let in_dtype = x.dtype();
let x = x.to_dtype(DType::F32)?;            // upcast to f32 first
let normed = self.norm(&x)?;                // x * recip(sqrt(mean(x²)+eps))
let w = self.weight.to_dtype(DType::F32)?;   // weight also f32
normed.broadcast_mul(&w)?.to_dtype(in_dtype) // multiply in f32, THEN cast back
```
This is normalize-in-f32 → weight-multiply-in-f32 → cast, *not* the
cast-then-multiply order `candle_nn::RmsNorm`/moshi use. `norm` (`:129`) is
un-folded from `forward` for 1:1 parity with the Python helper, and uses
`recip(sqrt(z))` because candle has no fused `rsqrt` (so ~1 ULP from torch's
fused `rsqrt` — the cross-library floor).

**`Glu` / SwiGLU** (`transformer.rs:152`). The SwiGLU path (`:185`):
```rust
let a = candle_nn::ops::silu(&self.w1.forward(x)?)?;
let b = self.w3.as_ref().unwrap().forward(x)?;
self.w2.forward(&(a * b)?)
```
The non-SwiGLU branch uses `gelu_erf` (`:190`). The `ff_dim` sizing (`:168-173`)
mirrors Python: `4*dim` → `2/3·ff` → `ffn_dim_multiplier·ff` →
`multiple_of * ff.div_ceil(multiple_of)`. For `dim=1024, multiple_of=256`:
`4096 → 2730 → 2816`. `div_ceil` is the Rust idiom for Python's `ceil(...)`.

**`BoundedAttention`** (`transformer.rs:278`). `forward` (`:318`):
1. Reshape q/k/v to `[b, t, heads, head_dim]` (`:331-333`).
2. Optional qk-RMSNorm per-head over `head_dim=32` (`:335-338`).
3. Transpose to `[b, heads, t, head_dim]`, **upcast to f32**, apply interleaved
   RoPE via `rope_i_slow`, cast back to input dtype, transpose back
   (`:344-350`).
4. Update the `LayerKvCache` (in-place when `Some`) (`:352-355`).
5. Transpose to `[b, heads, t, head_dim]`, `repeat_kv` for GQA (`:361-363`).
6. **Hand-rolled SDPA**: upcast q/k/v to f32, `matmul(q, kᵀ) * scale`,
   add `causal_mask` if `q_len != 1`, `softmax`, `matmul(attn, v)`, cast back
   (`:368-378`). `scale = 1/sqrt(head_dim)`.

**Causal masking is shape-dependent** (`causal_mask`, `:262`): the mask is
`causal_lower_right` — query `i` attends keys `≤ i + (kv_len - q_len)`. The
depthformer decodes with `q_len == 1` so `attn_mask=None` would be the Python
path; the Rust code instead *skips* the mask add when `q_len == 1` (`:373`),
which is mathematically equivalent (no future keys to mask when the cache is
causal and `q_len == 1`). For `q_len > 1` the additive `causal_mask` is applied.

**RoPE is interleaved (GPT-J style), not half-split.** `precompute_freqs_cis`
(`:210`) builds `inv_freq = 1/theta^(2i/dim)` for `i in 0..half`, then
`freqs = outer(t, inv_freq)`, returned as `(cos, sin)` f32 of shape
`[end, head_dim/2]`. `apply_rotary_emb` (`:228`) calls `rope_i_slow` on q and k.
There is no complex dtype in candle — the Python `view_as_complex * freqs_cis`
becomes the real-valued interleaved rotation. **`rope_i_slow` is deliberately
not the fused `rope_i`**: the depthformer runs inside the trainable
`logits`/`forward` graph and the fused op severs autograd (the
`rotary_is_differentiable` test at `:725` pins this).

**`SharedEmbedding`** (`:547`). `Embedding` + pre-logits `RmsNorm` +
untied `Linear` `to_logits`. `get_logits(e) = to_logits(embedding_norm(e))`
(`:574`). The norm is applied before the logit projection. The depthformer
itself uses `has_embedding=None`; the eight per-codebook `SharedEmbedding`s
provide both input embedding and output `get_logits`. `tie_embedding` is a
train-time flag; the checkpoint always ships a concrete `to_logits.weight`, so
the Rust loads it from disk rather than assuming the tie.

**`Mha`** (`:386`). `forward` (`:428`) does the fused `qkv_proj` (no bias) of
`total_width = dim + 2*head_dim*gqa_dim` for GQA (`:412-416`), narrows into
`xq/xk/xv` (`:436-438`), cache-aware freqs slice (`:441-443`), and delegates to
`BoundedAttention`. `forward_cached` (`:466`) builds a `LayerKvCache` from the
incoming `LayerCache`, runs `forward`, and returns `(ys, kv_cache.to_cache())`.

**`LayerKvCache`** (`:55`). A thin **adapter** over the vendored
`ConcatKvCache` (from candle-nn 0.10.2, in `src/candle_ext/kv_cache.rs`).
`update(k, v)` delegates to `ConcatKvCache::append` (cat on dim=1, the time axis
of `[b, t, heads, head_dim]`). `from_cache` seeds from an `Option<(Tensor,
Tensor)>`. `to_cache` extracts the current `(k, v)` as a `LayerCache`.

**`RawLmBackbone`** (`:581`). `forward` (`:598`) iterates layers, threading an
`Option<&mut [LayerKvCache]>`. `forward_cached` (`:635`) threads an
`Option<Vec<LayerCache>>`, seeding `None`-filled when absent. Both return the
hidden state; `forward_cached` also returns the updated per-layer cache vector.

**The `SequenceModel` trait** (`:673`) is the Rust analog of Python's ABC:
`forward` / `forward_cached` / `dim` / `dim_out`. `RawLmBackbone` impls it
(`:699`). The Python `__init__` no-op maps to a `fn new() where Self: Sized`
default (kept object-safe).

**The actual decode loop** lives in `lfm2_audio.rs::_sample_audio_frame`, not
here, but it is the only caller of `RawLmBackbone::forward_cached`. 8 sequential
single-token steps per audio frame, the KV cache spanning the 8 codebook
positions, conditioned on the one backbone hidden via additive `depth_linear`.
EOAudio is code index **2048** (vocab `2048+1=2049`).

## Dtypes & shapes (Rust)

| Stage | Input | Output |
|---|---|---|
| `depth_linear` (in lfm2_audio) | backbone hidden `(2048,)` | `(1024·8,)` → reshape `[8,1024]` |
| `RawLmBackbone::forward_cached` (one step) | `[1,1,1024]` | `[1,1,1024]` |
| `RmsNorm` internal | x model dtype | upcast **f32** for norm+weight-mul, cast back |
| `apply_rotary_emb` internal | q/k model dtype | upcast **f32**, `rope_i_slow`, cast back |
| qkv_proj (GQA) | `[1,1,1024]` | `[1,1,1536]` → narrow q`[…,1024]` k`[…,256]` v`[…,256]` |
| Hand-rolled SDPA | q/k/v | scores in **f32** (explicit upcast), out cast back to model dtype |
| KV cache tensors | k,v `[1,t,8,32]` | grows to `[1,t+1,8,32]` via `ConcatKvCache::append` |
| `depth_embeddings[i].get_logits` | `[1024]` | logits `(2049,)` |
| sampled token | logits `(2049,)` | i64 scalar (code 0..2048; 2048=EOAudio) |
| frame (8 codebooks) | — | `(8,)` int codes |

`cos`/`sin` tables are f32 `[max_seq_len, head_dim/2=16]` (candle has no
complex64; the Python `freqs_cis` is complex64).

## Wiring (Rust)
**Upstream.**
- The backbone hidden driving every audio frame comes from
  `model/lfm2_hf.rs` (the HF `Lfm2Model` backbone) via the top model:
  `(1,L,2048)` hidden → `depth_linear` → `[8,1024]` per-codebook seeds.
  See [`glm-version/model/lfm2_backbone.md`](lfm2_backbone.md).
- The per-codebook input/output embeddings are the eight `SharedEmbedding`s
  (defined here) owned by `lfm2_audio.rs` as `depth_embeddings`; each carries
  the previously sampled code's embedding `(1024,)` back into the loop.

**Downstream.**
- The sampled 8-code audio frame is consumed by
  [`glm-version/model/lfm2_audio.md`](lfm2_audio.md) `_sample_audio_frame` /
  `generate_*`, which re-embeds it through `audio_embedding` for the next
  backbone step and emits the frame.
- The accumulated audio codes `(8,)` int per frame flow to the codec for
  waveform synthesis: either [`glm-version/detokenizer.md`](detokenizer.md)
  (LFM2 ISTFT vocoder) or `MimiDetokenizer` (`audio_out.rs`), both via
  `processor.rs::decode`. Codes 0..2047 (2048=EOAudio terminates the turn, not
  decoded) → f32 waveform @ 24 kHz.

## Python ↔ Rust — where the port differs

| Python (`transformer.py`) | Rust (`transformer.rs`) | Difference | Why |
|---|---|---|---|
| `RMSNorm(nn.Module)` | `struct RmsNorm` with hand-composed `forward` | **deliberate** | `candle_nn::RmsNorm` casts back to input dtype *before* the weight multiply; liquid_audio multiplies in f32 then casts. At bf16 these differ. The Rust port composes the norm from candle tensor ops in liquid_audio's order (PYTHON_VS_RUST.md §2.4). |
| `nn.Linear` | `candle_nn::linear_no_bias` | identical (no bias in qkv/out) | — |
| `GLU` with `nn.GELU()` / `silu` | `Glu` with `gelu_erf()` / `candle_nn::ops::silu` | identical | — |
| `BoundedAttention` with `F.scaled_dot_product_attention(is_causal=…, enable_gqa=True)` | hand-rolled: `matmul + additive_mask + softmax + matmul + repeat_kv` | **deliberate** | eager **sdpa/no-flash** math, the path the f32 goldens were dumped from (§2.2). Also: candle's SDPA doesn't expose `enable_gqa`, so GQA is done by `repeat_kv` (head-repeat) before the matmul. |
| `MHA` / `LayerKVCache` | `Mha` / `LayerKvCache` (adapter over `ConcatKvCache`) | **deliberate reuse** | `ConcatKvCache` (vendored from candle-nn 0.10.2) is a structural 1:1 of the Python `torch.cat([k_cache, k], dim=1)`. The adapter only re-exposes the Python method names + the `(k, v)`-tuple constructor. (§2.3) |
| `StandardBlock`, `SharedEmbedding`, `RawLMBackbone` | `StandardBlock`, `SharedEmbedding`, `RawLmBackbone` | structural 1:1 | — |
| `apply_rotary_emb` via `view_as_complex` (complex64) | `rope_i_slow(q, cos, sin)` (real interleaved) | **deliberate: real + slow** | candle has no complex dtype → the complex rotation is the real `rope_i` interleaved-pair rotation. **`rope_i_slow` not fused `rope_i`** because the depthformer runs inside the trainable graph and the fused op severs autograd (test `rotary_is_differentiable` at `:725`). |
| `precompute_freqs_cis` → complex64 `polar(1, outer(t, inv_freq))` | `precompute_freqs_cis` → `(cos, sin)` f32 | **deliberate** | same math, real form for `rope_i`. |
| `wrap_activation_checkpoint` (torch checkpoint) | omitted | **deliberate omission** | no autograd on the inference path; an identity wrapper only obscured the runtime surface. |
| `zip(self.layers, cache, strict=True)` (Python raises on length mismatch) | explicit `if cs.len() != self.layers.len() { return Err(...) }` | **deliberate hardening** | Rust `zip` silently stops at the shorter side, which would run a *prefix* of the backbone — a silent correctness bug. The Rust port errors loudly instead. Test `forward_rejects_cache_length_mismatch` at `:744`. |
| `CacheType = torch.Tensor \| None \| Sequence[CacheType]` (recursive union) | `LayerCache = Option<(Tensor, Tensor)>` + `Vec<LayerCache>` at the backbone level | **deliberate** | the two concrete shapes the union takes in this module, without an open-ended enum. Documented at `:36-44`. |
| `SequenceModel(nn.Module, ABC)` | `trait SequenceModel` | **ABC → trait** | the standard Rust substitution; concrete types own construction. |
| `forward_cached(x, cache) -> (out, cache)` returning a fresh cache | same signature, returns `(Tensor, Vec<LayerCache>)` | identical | the functional cache contract; `forward(x, Some(&mut cache))` is the in-place cached path (PYTHON_VS_RUST.md §2.1 — Rust models the in-place Python `forward` with `Option<&mut [LayerKvCache]>`). |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. `(Cpu, F32)` for parity; Metal/bf16 opt-in. |

**Parity:** backbone hidden 6.558e-6, text logits 5.505e-6, **depthformer audio
frame token-EXACT** `[213,836,182,416,782,1796,202,578]` (PARITY.md).

## Precision / gotchas (Rust-specific)
- **`RmsNorm` is the subtle one — and it's not `candle_nn::RmsNorm`.** The port
  composes its own `RmsNorm` from candle tensor ops to preserve the
  f32-multiply-then-cast order. A naive `candle_nn::RmsNorm` wrap would cast
  first and multiply in bf16, diverging at bf16. The `norm` helper uses
  `recip(sqrt(z))` because candle has no fused `rsqrt` — ~1 ULP from torch's
  fused `rsqrt` (the floor, §1.4). f32 parity is unaffected.
- **`rope_i_slow` vs `rope_i`.** The fused `rope_i` (the `apply_op3_no_bwd`
  kernel) severs autograd; `rope_i_slow` is the basic-op differentiable path.
  The `rotary_is_differentiable` test (`:725`) pins this — it fails if anyone
  swaps back to the fused op. Same forward values; only the backward path
  differs.
- **`softmax` vs `softmax_last_dim`.** The attention uses `candle_nn::ops::softmax`
  (basic ops, differentiable), *not* `softmax_last_dim` (the fused
  `apply_op1_no_bwd` kernel that severs autograd). Same forward values. This
  matters because the depthformer runs in the trainable `logits`/`forward`
  graph; the comment at `:21-23` records the choice.
- **`BoundedAttention` upcasts q/k/v to f32 explicitly** (`:368-370`) before the
  matmuls, mirroring torch SDPA's f32 accumulation. Output is cast back to
  `in_dtype` (`:378`). At bf16 this matches the f32-accumulation semantics of
  `F.scaled_dot_product_attention`.
- **RoPE upcast contract.** `apply_rotary_emb`'s callers pass already-f32 q/k
  (`:345-346` upcast before the call, `:349-350` cast back after). This mirrors
  the Python `xq.float() ... type_as(xq)`. Getting the upcast boundary wrong
  would feed bf16 into `rope_i_slow` and silently degrade parity.
- **`causal_mask` skip when `q_len == 1`.** The depthformer always decodes with
  `q_len == 1`, so the mask is skipped (`:373`). This is mathematically
  equivalent to applying a no-op mask (no future keys when the cache is causal
  and `q_len == 1`). For `q_len > 1` the additive `causal_mask` is applied.
- **`tie_embedding` is a train-time flag only.** The checkpoint always ships a
  concrete `to_logits.weight`; the Rust loads it from disk rather than assuming
  the tie — faithful for both tied (depthformer) and untied (`audio_embedding`)
  cases. Comment at `:540-546`.
- **`div_ceil` for the `ff_dim` round-up.** `multiple_of * ff.div_ceil(multiple_of)`
  is the Rust idiom for Python's `multiple_of * ceil(ff / multiple_of)`.
  `usize::div_ceil` is stable since Rust 1.73.
- **`ConcatKvCache` is vendored, not re-implemented.** The cat-on-dim-1 cache
  logic is from candle-nn 0.10.2 (`src/candle_ext/kv_cache.rs`), kept on the
  0.9.2 pin. The `LayerKvCache` adapter only re-exposes the Python method names.
  Re-implementing the cat would be a needless fork.
- **`forward_rejects_cache_length_mismatch` test.** The Rust port hardens
  against a length-mismatched cache slice — Python's `zip(strict=True)` raises,
  but Rust's `zip` silently stops at the shorter side. The explicit length
  check (`:606`) errors loudly instead of running a prefix of the backbone.
- **EOAudio = code 2048; audio vocab is `2048+1=2049`.** A `2048` sampled in
  codebook 0 forces the entire 8-code frame to 2048 and ends the audio turn; it
  is never sent to the codec for decoding. (Enforced in `lfm2_audio.rs`, not
  here, but the depthformer's `SharedEmbedding` vocab is `2049` to admit it.)

## Cross-references
- [`wiki/model/transformer.md`](../../../wiki/model/transformer.md) — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.2 (CUDA kernels
  → portable candle ops), §2.3 (`ConcatKvCache` reuse), §2.4 (RMSNorm bf16
  order), §2.5 (off-path stubs).
- `liquid-audio/parity/PARITY.md` — depthformer audio frame token-EXACT.
- `liquid-audio/src/candle_ext/kv_cache.rs` — the vendored `ConcatKvCache`.
