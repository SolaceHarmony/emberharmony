<!-- topic: Conformer Encoder -->
# CF02 · RelPosition MultiHeadAttention
**Code:** `CF02` · **Source:** `model/conformer/mha.py` · **Rust:** `model/conformer/mha.rs` · **On the LFM2-Audio inference path:** yes

## Role
NeMo FastConformer self-attention primitives: a Transformer-XL **relative** positional-encoding table (`RelPositionalEncoding`) and the matching **relative-position multi-head attention** (`RelPositionMultiHeadAttention`). One `RelPositionalEncoding` lives at the encoder root and produces a single `pos_emb` table that is threaded into every one of the N=17 `ConformerLayer`s; each layer owns one `RelPositionMultiHeadAttention` as its `self_attn`. This is the only place global token-to-token mixing happens in the conformer audio-in front-end, and it is the layer that turns the 512-dim subsampled mel embeddings into context-aware encoder states.

## How it works

### RelPositionalEncoding (centered 2L-1 table)
This is the absolute-sinusoid `PositionalEncoding` (`mha.py:45`) subclassed (`mha.py:108`) to hold **relative** offsets instead of absolute indices.

- **Table construction** (`create_pe`, `mha.py:67`): for positions vector of length `P`, build `pe[P, d_model]` with `pe[:, 0::2]=sin(pos·div)`, `pe[:, 1::2]=cos(pos·div)` interleaved (NOT half-split), where `div = exp(arange(0, d_model, 2) · -(ln(10000)/d_model))` (`mha.py:70-75`). Note the base constant is `INF_VAL=10000.0` reused as the sinusoid `theta` (`mha.py:43`). `div_term` is forced to **f32** regardless of model dtype (`mha.py:71`), then `pe` is cast to the model dtype at the end (`mha.py:76`).
- **Relative positions** (`extend_pe`, `mha.py:119`): for input length `L`, allocate `needed_size = 2L-1` and fill positions `arange(L-1, -L, -1)` — i.e. from `+(L-1)` down through `0` to `-(L-1)`. Positive = left context, negative = right context. The buffer `pe` is non-persistent (`register_buffer(..., persistent=False)`, `mha.py:80`) and only re-extended when too small.
- **forward** (`mha.py:129`): unlike the absolute base, it does **NOT** add the encoding to `x`; it optionally `xscale`s `x` (here `xscale=sqrt(d_model)`, set by the encoder), then slices the centered window `pe[:, start_pos:end_pos]` where `center_pos = pe.size(1)//2 + 1`, `start = center - input_len`, `end = center + input_len - 1`, returning `(x, pos_emb)` with `pos_emb` shape `(1, 2L-1, d_model)`. `dropout`/`dropout_emb` are eval-time identities.

### RelPositionMultiHeadAttention (the Transformer-XL ac/bd decomposition)
Subclasses the standard `MultiHeadAttention` (`mha.py:155`). Config for LFM2-Audio's FastConformer: `n_feat = d_model = 512`, `n_head = 8`, so `d_k = d_v = 64` and `s_d_k = sqrt(64) = 8` (`mha.py:191-194`). Four `nn.Linear(512,512)` projections `linear_q/k/v/out` (`mha.py:196-199`) plus a bias-free `linear_pos(512,512)` for the positional stream (`mha.py:348`), plus two learnable per-head bias vectors `pos_bias_u`, `pos_bias_v` of shape `(h, d_k) = (8, 64)` (`mha.py:352-353`, passed in from the encoder so they are **shared across all 17 layers**). This is plain MHA (not GQA): kv-heads == q-heads == 8, no RoPE, no qk-norm.

forward (`mha.py:375`), manual eager path (`use_pytorch_sdpa=False` on this path):
1. `update_cache` (`mha.py:307`) — offline `cache=None`, a no-op returning k/v/q unchanged.
2. `forward_qkv` (`mha.py:204`) — project and reshape to `(b, h, t, d_k)`; then `q` is transposed to `(b, t, h, d_k)` (`mha.py:397`) so the per-head bias can be broadcast-added.
3. Positional stream: `p = linear_pos(pos_emb).view(1, -1, h, d_k).transpose(1,2)` → `(1, h, 2L-1, d_k)` (`mha.py:401-402`).
4. Two biased queries (Transformer-XL §3.3): `q_with_bias_u = (q + pos_bias_u).transpose(1,2)` and `q_with_bias_v = (q + pos_bias_v).transpose(1,2)`, both `(b, h, t, d_k)` (`mha.py:405-407`).
5. **matrix_bd** (content↔position term): `matrix_bd = q_with_bias_v @ pᵀ` → `(b, h, t, 2L-1)` (`mha.py:416`), then `rel_shift` (`mha.py:362`) realigns the `2L-1` relative axis to the `t2` key axis via the pad-reshape-slice trick: left-pad one column, view `(b,h,pos_len+1,qlen)`, drop the first row, view back to `(b,h,qlen,pos_len)`.
6. **matrix_ac** (content↔content term): `matrix_ac = q_with_bias_u @ kᵀ` → `(b, h, t, t2)` (`mha.py:449`). `matrix_bd` is then sliced to `matrix_ac.size(-1)` to discard the extra relative columns (`mha.py:450`).
7. **scores** `= (matrix_ac + matrix_bd) / s_d_k` — the `1/sqrt(d_k)=1/8` scale is applied once to the **sum** (`mha.py:451`).
8. `forward_attention` (`mha.py:227`): masked softmax over the last dim. NeMo's masking is `scores.masked_fill(mask, -INF_VAL)` → `softmax(dim=-1)` → `.masked_fill(mask, 0.0)` (the post-softmax zeroing handles fully-masked rows; `mha.py:240-241`). Then `attn @ v` → `(b,h,t,d_k)`, transpose+reshape to `(b, t, d_model)`, and `linear_out` (`mha.py:246-249`).

**Mask on the offline inference path is effectively None.** The FastConformer offline config uses full context (`att_context_size = [-1,-1]`), so `_create_masks` (`encoder.py:737`) skips both the `triu`/`tril` band clamps (guarded by `>= 0`) — `att_mask` stays all-True and `~att_mask` becomes all-False, and with single-utterance no-padding `pad_mask` adds nothing. The id map records this as `_create_masks returns (None, None)`, so the eager branch runs the plain-softmax leg.

**SDPA branch (alt, training/cuda):** `use_pytorch_sdpa=True` (`mha.py:419`) pre-scales `matrix_bd` by `1/sqrt(d_k)`, bakes the mask into it as additive `-INF`, and passes it as `attn_mask` to `F.scaled_dot_product_attention(q_with_bias_u, k, v, ...)`, letting SDPA fold in `q·kᵀ/sqrt(d_k)`. This is algebraically identical to the eager `softmax((matrix_ac+matrix_bd)/sqrt(d_k))·v`; an explicit all-masked-row zeroing (`mha.py:439-443`) matches the eager post-softmax `masked_fill(mask,0)`.

## Dtypes & shapes
| Stage | dtype | shape |
|---|---|---|
| `x` into `RelPositionalEncoding.forward` (subsampled mel emb) | model dtype (bf16 cuda / f32 Rust-CPU / bf16 Metal) | `(B, T', 512)` |
| `pos_emb` out of pos-enc | model dtype (`div_term` computed in **f32** then cast) | `(1, 2T'-1, 512)` |
| q/k/v after `forward_qkv` | model dtype | `(B, 8, T', 64)` |
| `p` (projected pos) | model dtype | `(1, 8, 2T'-1, 64)` |
| `pos_bias_u/v` | f32 params (cast to model dtype in add) | `(8, 64)` |
| `matrix_ac`, `matrix_bd` (post rel_shift+slice) | model dtype | `(B, 8, T', T')` |
| `scores` | model dtype; **softmax internally upcasts to f32** | `(B, 8, T', T')` |
| attention output of `RelPositionMultiHeadAttention.forward` | model dtype | `(B, T', 512)` |

Internal promotions: sinusoid `div_term` is f32-built then cast (`mha.py:71,76`); torch `softmax` accumulates in f32 internally even on a bf16 input. No int/u32 here (codes are far downstream). No f64.

## Wiring
**Upstream**
- [conformer_subsampling](CF05-Subsampling) → produces the subsampled `(B, T', 512)` model-dtype embeddings that `RelPositionalEncoding.forward` `xscale`s and that `forward_qkv` projects.
- [conformer_encoder](CF01-Conformer-Encoder) → constructs the single `RelPositionalEncoding`, owns the shared `pos_bias_u/v`, computes the (offline ⇒ effectively None) `att_mask`, and threads `pos_emb` `(1, 2T'-1, 512)` model-dtype into every layer.
- [conformer_modules](CF03-Conformer-Layer) → each `ConformerLayer` calls `self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)` on its pre-LN'd `(B, T', 512)` input.

**Downstream**
- [conformer_modules](CF03-Conformer-Layer) ← the `(B, T', 512)` model-dtype attention output flows back into the ConformerLayer (residual add → conv module → second half-step FF).
- [conformer_encoder](CF01-Conformer-Encoder) ← after all 17 layers, the encoded `(B, T', 512)` becomes the conformer encoder output, which the audio adapter MLP lifts to 2048 for the backbone.

## Python ↔ Rust
Symbol map (py → rust, all in `mha.rs`):
- `PositionalEncoding` → `PositionalEncoding` (struct); `create_pe`/`extend_pe`/`forward` → same names.
- `RelPositionalEncoding` → `RelPositionalEncoding { base: PositionalEncoding }` (composition, since Rust has no inheritance; `mha.rs:113`). `extend_pe` rebuilds positions `arange(L-1,-L,-1)` → `(2L-1,)` (`mha.rs:124`); `forward` does not add to `x` (`mha.rs:134`).
- `MultiHeadAttention` → `MultiHeadAttention` (struct, `mha.rs:154`); `forward_qkv`/`forward_attention`/`update_cache` → same names; `masked_softmax` is a free fn (`mha.rs:39`).
- `RelPositionMultiHeadAttention` → `RelPositionMultiHeadAttention { base: MultiHeadAttention, linear_pos, pos_bias_u, pos_bias_v }` (`mha.rs:259`); `rel_shift`/`forward` → same; streaming `forward_cache` mirrors `update_cache`.

Deliberate divergences (PORT_STATUS.md:17-18,59; module note `mha.rs:14-25`):
- **Both Python branches collapse to one Rust path.** The Rust keeps only the eager `softmax((matrix_ac+matrix_bd)/sqrt(d_k))·v` and proves it matches a Python `use_pytorch_sdpa=True` golden (test `rel_pos_attention_sdpa_parity`, diff 1.07e-6; Python's own True-vs-False diff is 1.5e-7). So one Rust impl faithfully covers the eager AND SDPA legs.
- **Fused SDPA deliberately avoided.** candle ships `ops::sdpa`, but it is an `apply_op*_no_bwd` op that severs autograd and would break the trainable audio-in encode graph; likewise `softmax` (differentiable) is used over `softmax_last_dim` (fused, no-bwd) (`mha.rs:28-31`). Same forward values, gradient-safe.
- **No mutable pos-emb buffer.** Python registers/extends a non-persistent `pe` buffer; Rust just returns a freshly sized table per call (`mha.rs:54-56`), sidestepping the `center_pos` slicing because the table is built exactly to the `2·input_len-1` window.
- **Masking via `where_cond` (SET), not additive.** `masked_softmax` uses `where_cond` to set `-INF_VAL`/`0.0`, bit-identical to torch `masked_fill` rather than an additive approximation (`mha.rs:35-50`).
- Device-agnostic candle `Tensor` ops replace device-specific CUDA kernels; `avoid_float16_autocast_context` (`mha.py:270`) is a no-op in both (a candle analog).

## Precision / gotchas
- **Cross-library f32 floor on CPU.** Rust CPU has no bf16 matmul, so on CPU the whole attention runs f32 (weights upcast bf16→f32); Metal stays bf16, Python cuda stays bf16. Numerically the f32 CPU path is the more precise one; expect small bf16-vs-f32 deltas vs the Python golden, not a bug.
- **theta == 10000 is the same constant as the masking `-INF`.** `INF_VAL=10000.0` (`mha.py:43`) is overloaded: it is both the sinusoid period in `div_term` and the `-INF_VAL` fill used for masked scores. Do not "fix" one without the other.
- **The `1/sqrt(d_k)` scale is applied once, to the sum.** It is NOT applied separately to `matrix_ac` and `matrix_bd`; the eager path divides `(matrix_ac + matrix_bd)` by `s_d_k=8` (`mha.py:451`). The SDPA path instead pre-scales `matrix_bd` and lets SDPA scale `q·kᵀ` — same result, different bookkeeping.
- **`rel_shift` then slice ordering matters.** `matrix_bd` is `(b,h,t,2L-1)`; you must `rel_shift` first, then truncate to `matrix_ac.size(-1)` (`mha.py:450`). Truncating before the shift would mis-align the relative offsets. Off-by-one lives in `center_pos = pe.size(1)//2 + 1` (`mha.py:146`) on the Python slicing side; the Rust avoids it by sizing the table exactly.
- **Masked softmax does the post-softmax zero too.** Don't drop the second `masked_fill(mask, 0.0)` after softmax (`mha.py:241`) — it zeros fully-masked rows that would otherwise be uniform. Harmless on the offline LFM2-Audio path (mask ≈ None) but load-bearing for streaming/padded batches.
- **No causal masking here.** The conformer is a bidirectional (full-context, `[-1,-1]`) encoder; causality lives only downstream in the backbone/depthformer, not in this attention.
