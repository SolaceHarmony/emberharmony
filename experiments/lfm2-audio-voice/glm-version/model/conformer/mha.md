# conformer_mha (Rust port)
**Source:** `liquid-audio-rs/src/model/conformer/mha.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/mha.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`ARCH/model/conformer/mha.md`](../../ARCH/model/conformer/mha.md).

## Role
NeMo FastConformer self-attention primitives in the Rust port:
`PositionalEncoding` + `RelPositionalEncoding` (Transformer-XL relative
positional-encoding table), and `MultiHeadAttention` +
`RelPositionMultiHeadAttention` (relative-position MHA). One
`RelPositionalEncoding` lives at the encoder root and produces a single
`pos_emb` table threaded into every one of the N=17 `ConformerLayer`s; each
layer owns one `RelPositionMultiHeadAttention` as its `self_attn`. This is the
only place global token-to-token mixing happens in the conformer audio-in
front-end.

## How it works (Rust)

### `PositionalEncoding` (`mha.rs:57`)
The base sinusoidal positional encoding. Fields: `d_model`, `max_len`,
`xscale: Option<f64>`. `create_pe` (`:71`) builds the interleaved table:
`pe[:,0::2]=sin(pos·div)`, `pe[:,1::2]=cos(pos·div)` via
`Tensor::stack(&[sin, cos], 2)?.reshape((pos_length, d_model))` (`:84`) —
**interleaved, not half-split**. `div = exp(arange(0,d_model,2) ·
-(ln(INF_VAL)/d_model))` built in **f32** (`:75-77`), cast to the target dtype
at the end (`:85`). `INF_VAL = 10000.0` (`:33`) — overloaded as both the
sinusoid period and the masking `-INF` fill. `extend_pe` (`:89`) builds absolute
positions `arange(0, length)`. `forward` (`:97`) **adds** the encoding to `x`
(after optional `xscale`) — the base class adds, the rel-pos subclass does not.

### `RelPositionalEncoding` (`mha.rs:113`)
Composes `base: PositionalEncoding` (`:114`) — Rust has no inheritance, so the
subclass holds its base and calls its methods (the module header `:1-9`
records this). `extend_pe` (`:124`) builds relative positions
`arange(length-1, -length, -1)` → `(2L-1,)` (`:126`), then the inherited
`create_pe` builds the interleaved table. `forward` (`:134`) does **not** add
to `x` — returns `(x * xscale, pos_emb)`. The table is sized exactly to the
`2·input_len-1` window, sidestepping Python's `center_pos` slicing.

### `MultiHeadAttention` (`mha.rs:154`)
The base MHA. Config for LFM2-Audio's FastConformer: `n_feat = d_model = 512`,
`n_head = 8`, so `d_k = d_v = 64` and `s_d_k = sqrt(64) = 8` (`:180`). Four
`Linear(512, 512)` projections `linear_q/k/v/out` (`:181-184`). This is plain
MHA (not GQA): kv-heads == q-heads == 8, no RoPE, no qk-norm. `forward_qkv`
(`:190`) projects and reshapes to `(b, h, t, d_k)`. `forward_attention` (`:203`)
runs `masked_softmax` then `attn @ v` → transpose+reshape → `linear_out`.

### `masked_softmax` (`mha.rs:39`)
NeMo's `scores.masked_fill(mask, -INF_VAL) → softmax(-1) →
.masked_fill(mask, 0.0)`. Uses `where_cond` to **SET** (not add) `-INF_VAL` /
`0.0`, bit-identical to torch `masked_fill` rather than an additive
approximation. The post-softmax zeroing (`:46`) handles fully-masked rows.
`None` ⇒ plain `softmax`. Unit test `masked_softmax_matches_python` (`:350`)
pins the four cases.

### `RelPositionMultiHeadAttention` (`mha.rs:259`)
Composes `base: MultiHeadAttention` plus `linear_pos: Linear` (bias-free,
`:272`) and `pos_bias_u`/`pos_bias_v` `(h, d_k) = (8, 64)` (`:273-274`).
`forward` (`:292`) delegates to `forward_cache(…, None)` (offline path).
`forward_cache` (`:306`):
1. `update_cache` (offline `cache=None` ⇒ no-op) (`:316`).
2. `forward_qkv` → q/k/v `(b, h, t, d_k)`; q transposed to `(b, t, h, d_k)` for
   the per-head bias broadcast (`:317-318`).
3. Positional stream: `p = linear_pos(pos_emb).reshape((n_batch_pos, (), h,
   d_k)).transpose(1, 2)` → `(1, h, 2L-1, d_k)` (`:321-326`).
4. Two biased queries (Transformer-XL §3.3): `q_with_bias_u = (q +
   pos_bias_u).transpose(1,2)`, `q_with_bias_v = (q + pos_bias_v).transpose(1,2)`
   (`:328-331`).
5. **matrix_bd** (content↔position): `q_with_bias_v @ pᵀ` → `(b, h, t, 2L-1)`
   (`:333`), then `rel_shift` (`:284`) realigns: pad one column, reshape to
   `(b, h, pos_len+1, qlen)`, drop the first row, reshape back to
   `(b, h, qlen, pos_len)`.
6. **matrix_ac** (content↔content): `q_with_bias_u @ kᵀ` → `(b, h, t, t2)`
   (`:336`); `matrix_bd` sliced to `t2` (`:338`).
7. **scores** `= (matrix_ac + matrix_bd) / s_d_k` — the `1/sqrt(d_k)=1/8` scale
   is applied once to the **sum** (`:339`).
8. `forward_attention` (`:340`): `masked_softmax` → `attn @ v` →
   `linear_out`.

**Mask on the offline inference path is effectively None.** The FastConformer
offline config uses full context (`att_context_size = [-1,-1]`), so
`_create_masks` returns `(None, None)`; `masked_softmax(scores, None)` is plain
`softmax`.

## Dtypes & shapes (Rust)
| Stage | dtype | shape |
|---|---|---|
| `x` into `RelPositionalEncoding::forward` | model dtype (f32 CPU / bf16 Metal) | `(B, T', 512)` |
| `pos_emb` out of pos-enc | model dtype (`div` computed in **f32** then cast) | `(1, 2T'-1, 512)` |
| q/k/v after `forward_qkv` | model dtype | `(B, 8, T', 64)` |
| `p` (projected pos) | model dtype | `(1, 8, 2T'-1, 64)` |
| `pos_bias_u/v` | model dtype (loaded from `VarBuilder`) | `(8, 64)` |
| `matrix_ac`, `matrix_bd` (post `rel_shift`+slice) | model dtype | `(B, 8, T', T')` |
| `scores` | model dtype; `softmax` internally upcasts to f32 (`ops::softmax`) | `(B, 8, T', T')` |
| attention output | model dtype | `(B, T', 512)` |

## Wiring (Rust)
**Upstream**
- `model/conformer/subsampling.rs` → produces the subsampled `(B, T', 512)`
  embeddings that `RelPositionalEncoding::forward` `xscale`s and that
  `forward_qkv` projects. See
  [`glm-version/model/conformer/subsampling.md`](subsampling.md).
- `model/conformer/encoder.rs` → constructs the single
  `RelPositionalEncoding`, owns the shared `pos_bias_u/v`, computes the
  (offline ⇒ effectively None) `att_mask`, and threads `pos_emb`
  `(1, 2T'-1, 512)` into every layer. See
  [`glm-version/model/conformer/encoder.md`](encoder.md).
- `model/conformer/modules.rs` → each `ConformerLayer` calls
  `self_attn.forward(query=x, key=x, value=x, mask, pos_emb)` on its pre-LN'd
  `(B, T', 512)` input. See
  [`glm-version/model/conformer/modules.md`](modules.md).

**Downstream**
- `model/conformer/modules.rs` ← the `(B, T', 512)` attention output flows
  back into the `ConformerLayer` (residual add → conv module → second half-step
  FF).
- `model/conformer/encoder.rs` ← after all 17 layers, the encoded
  `(B, T', 512)` becomes the conformer encoder output, which the audio adapter
  MLP lifts to 2048 for the backbone.

## Python ↔ Rust — where the port differs

| Python (`mha.py`) | Rust (`mha.rs`) | Difference | Why |
|---|---|---|---|
| `PositionalEncoding` ← `RelPositionalEncoding` (inheritance) | `PositionalEncoding` + `RelPositionalEncoding { base: PositionalEncoding }` (composition) | **deliberate: inheritance → composition** | Rust has no inheritance; the subclass holds its base and calls its methods (`create_pe`), exactly where Python calls `super()`. Both structs live in this module so the subclass reads the base's private fields. Module header `:1-9`. |
| `MultiHeadAttention` ← `RelPositionMultiHeadAttention` (inheritance) | `MultiHeadAttention` + `RelPositionMultiHeadAttention { base: MultiHeadAttention, … }` (composition) | **deliberate: inheritance → composition** | same pattern. `forward_cache` calls `self.base.update_cache` / `self.base.forward_qkv` / `self.base.forward_attention`. |
| `register_buffer("pe", …, persistent=False)` (mutable buffer) | `create_pe`/`extend_pe` return a fresh table per call (`:54-56`) | **deliberate: no mutable buffer** | Python extends/re-slices a non-persistent buffer; Rust returns a freshly sized table, sidestepping the `center_pos` slicing because the table is built exactly to the `2·input_len-1` window. |
| `use_pytorch_sdpa=False` (eager) and `use_pytorch_sdpa=True` (SDPA) — two branches | **one Rust path** (the eager `softmax((matrix_ac+matrix_bd)/sqrt(d_k))·v`) | **deliberate: both branches collapse to one** | the SDPA branch is algebraically identical (it pre-scales `matrix_bd` and lets SDPA fold in `q·kᵀ/sqrt(d_k)`). Verified by `rel_pos_attention_sdpa_parity` against a Python `use_pytorch_sdpa=True` golden (1.07e-6; Python's own True-vs-False diff is 1.5e-7). Module header `:14-25`. |
| `F.scaled_dot_product_attention` (fused) | hand-rolled matmul + `masked_softmax` + matmul | **deliberate: avoid fused `ops::sdpa`** | candle's `ops::sdpa` is `apply_op*_no_bwd` and severs autograd — the conformer attention runs in the trainable `logits` graph (audio-in encode). §2.2. |
| `softmax` (torch) | `candle_nn::ops::softmax` (differentiable basic ops) | **deliberate: not `softmax_last_dim`** | the fused `softmax_last_dim` severs autograd. Same forward values. `:28-31`. |
| `scores.masked_fill(mask, -INF_VAL)` (additive-ish) | `where_cond` to SET `-INF_VAL` / `0.0` (`:39-50`) | **deliberate: SET not add** | bit-identical to torch `masked_fill` rather than an additive approximation. |
| `avoid_float16_autocast_context()` (context manager) | not called on the offline path | **deliberate: no-op** | candle has no autocast; the conformer attention upcasts to f32 explicitly inside the score path. See [`glm-version/model/conformer/utils.md`](utils.md). |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. f32 on CPU; bf16 on Metal. |

## Precision / gotchas (Rust-specific)
- **Both Python branches collapse to one Rust path.** The Rust keeps only the
  eager `softmax((matrix_ac+matrix_bd)/sqrt(d_k))·v` and proves it matches a
  Python `use_pytorch_sdpa=True` golden (1.07e-6). So one Rust impl faithfully
  covers both legs. Don't add an SDPA branch — it would diverge from the
  verified path.
- **Fused `ops::sdpa` deliberately avoided.** It is `apply_op*_no_bwd` and
  severs autograd — the conformer attention runs in the trainable `logits`
  graph. Likewise `ops::softmax` (differentiable) over `softmax_last_dim`
  (fused, no-bwd). Same forward values, gradient-safe. `:28-31`.
- **`masked_softmax` uses `where_cond` (SET), not additive.** Bit-identical to
  torch `masked_fill`, not an additive `-INF` approximation. The post-softmax
  zero (`:46`) handles fully-masked rows — don't drop it. Harmless on the
  offline LFM2-Audio path (mask ≈ None) but load-bearing for streaming/padded
  batches.
- **`INF_VAL = 10000.0` is overloaded.** It is both the sinusoid period in
  `div_term` and the `-INF_VAL` fill used for masked scores (`:33`). Do not
  "fix" one without the other.
- **The `1/sqrt(d_k)` scale is applied once, to the sum.** It is NOT applied
  separately to `matrix_ac` and `matrix_bd`; the eager path divides
  `(matrix_ac + matrix_bd)` by `s_d_k=8` (`:339`). The SDPA path (not used here)
  instead pre-scales `matrix_bd` and lets SDPA scale `q·kᵀ` — same result,
  different bookkeeping.
- **`rel_shift` then slice ordering matters.** `matrix_bd` is
  `(b, h, t, 2L-1)`; you must `rel_shift` first (`:334`), then truncate to `t2`
  (`:338`). Truncating before the shift would mis-align the relative offsets.
  The Rust `rel_shift` (`:284`) pads one column, reshapes to
  `(b, h, pos_len+1, qlen)`, drops the first row, reshapes back.
- **No mutable pos-emb buffer.** Python registers/extends a non-persistent
  `pe` buffer; Rust returns a fresh table per call, sized exactly to the
  `2·input_len-1` window. This sidesteps Python's `center_pos` slicing and the
  off-by-one in `center_pos = pe.size(1)//2 + 1`.
- **Cross-library f32 floor on CPU.** Rust CPU has no bf16 matmul, so on CPU
  the whole attention runs f32 (weights upcast bf16→f32); Metal stays bf16.
  Numerically the f32 CPU path is the more precise one; expect small
  bf16-vs-f32 deltas vs the Python golden, not a bug.
- **No causal masking here.** The conformer is a bidirectional (full-context,
  `[-1,-1]`) encoder; causality lives only downstream in the
  backbone/depthformer, not in this attention.

## Cross-references
- [`ARCH/model/conformer/mha.md`](../../ARCH/model/conformer/mha.md) — Python
  original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.2 (kernel-free SDPA), §2.5 (off-path
  streaming stubs).
- `liquid-audio-rs/parity/PARITY.md` — conformer rel-pos embedding 9.537e-7,
  conformer layer 0 1.056e-6, conformer final 1.592e-6/8.25e-7.