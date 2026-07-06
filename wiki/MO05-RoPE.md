<!-- topic: Mimi Codec — Modules -->
# MO05 · RotaryEmbedding
**Code:** `MO05` · **Source:** `moshi/modules/rope.py` · **Rust:** `moshi crate rope` · **On the LFM2-Audio inference path:** yes

## Role
`RotaryEmbedding` / `apply_rope` is the rotary positional-embedding (RoPE, Su et al. 2022) primitive used by the **Mimi codec's** encoder and decoder `StreamingTransformer`s (`positional_embedding: "rope"`, `loaders.py:76,98`). On the LFM2-Audio path it is the *only* positional signal injected into the codec's self-attention: it rotates query/key head vectors by an angle proportional to absolute position so that the `q·k` dot product encodes *relative* offset. It is a pure function of `(q, k, offset)` with no learned parameters — `max_period` (10000) is the single fixed hyperparameter. (Note: the depthformer/LFM2-backbone RoPE is a *separate* implementation — see "Python ↔ Rust".)

## How it works
`apply_rope(q, k, offset, max_period=10000, time_before_heads=False)` (`rope.py:12`) is the whole component; `RotaryEmbedding.forward` (`rope.py:82`) just forwards into it with `self.max_period`. It is wrapped in `@torch_compile_lazy` (`rope.py:11`) so under CUDA it fuses; off-CUDA it is plain eager.

Step by step, for q/k of shape `[B, H, T, D]` (the codec calls it with `time_before_heads=False`, `transformer.py:548`):

1. **Frequency bank** (`rope.py:37-38`). `ds = arange(D//2)`; `freqs = exp(ds · (-log(max_period)·2/D))`. This is the standard inverse-frequency geometric series `freqs[j] = max_period^(-2j/D)` for `j ∈ [0, D/2)`, computed in **f32**. With `D=64` (codec head_dim, `d_model 512 / num_heads 8`) and `max_period 10000` this gives 32 frequencies from 1.0 down to ~10000^(-62/64).
2. **Per-position phase** (`rope.py:39-43`). `ts = offset.float().view(-1,1) + arange(T)` — absolute time indices, *per-batch-element* because `offset` is a length-`B` long tensor (the streaming KV-cache write position, `transformer.py:528`). For `time_before_heads=False` it is reshaped to `[B, 1, T, 1]` so it broadcasts over heads. f32 throughout.
3. **Interleaved pair view** (`rope.py:45-47`). q and k are reshaped `[..., D] -> [..., D/2, 2]`. This is the **interleaved / GPT-J** convention: adjacent channels `(2j, 2j+1)` form one complex pair, NOT the half-split `(j, j+D/2)` NeoX convention.
4. **Upcast + rotate** (`rope.py:50-62`). Real/imag parts `qr=q[...,0].float()`, `qi=q[...,1].float()` (and same for k) are extracted **in f32** regardless of incoming dtype. The rotation matrix is `rotr = cos(freqs·ts)`, `roti = sin(freqs·ts)` (broadcast `[B,1,T,1]·[D/2] -> [B,1,T,D/2]`), then the complex multiply:
   ```
   qor = qr*rotr - qi*roti ; qoi = qr*roti + qi*rotr
   ```
   i.e. `(qr + i·qi)·e^{i·freqs·ts}`. Identical formula for k.
5. **Cast back + re-interleave** (`rope.py:64-68`). `stack([qor, qoi], dim=-1)` re-forms the `[..., D/2, 2]` pairs, each part cast back to the *original* q dtype (bf16/f32), then `view(..., D)` restores `[B, H, T, D]`. Returns rotated `(qo, ko)`; v is untouched.

In the codec attention (`transformer.py:547-548`) rope is applied **after** the fused qkv projection and head-split (`rearrange "b t (p h d) -> p b h t d"`) and **before** the KV cache is completed (`_complete_kv`, `transformer.py:550`) and before SDPA. Because `offset` advances by `T` per streaming step (`transformer.py:569-573`), cached keys keep the phase they were rotated with at write time and new queries get the current phase — relative-position is preserved across the streaming boundary. No `cos`/`sin` table is cached across calls here; it is recomputed each forward from `offset+arange(T)` (cheap: `D/2=32` freqs).

## Dtypes & shapes
| Tensor | Dtype | Shape | Notes |
|---|---|---|---|
| `q`, `k` (in) | model dtype (bf16 cuda/Metal, f32 CPU) | `[B, H, T, D]` = `[B, 8, T, 64]` (codec) | v not passed |
| `offset` (in) | int64 (long) | `[B]` | streaming KV write pos; `.float()` internally |
| `freqs` | f32 | `[D/2]` = `[32]` | f32 always (`rope.py:37`) |
| `ts` | f32 | `[B,1,T,1]` | `offset + arange(T)` |
| `rotr`,`roti` | f32 | `[B,1,T,D/2]` | cos/sin of `freqs·ts` |
| internal `qr,qi,kr,ki` | **f32 upcast** | `[B,H,T,D/2]` | forced `.float()` (`rope.py:50-54`) |
| `qo`, `ko` (out) | model dtype (cast back) | `[B, H, T, D]` | re-interleaved, v unchanged |

Key promotion: rotation math is **always f32** (freq bank, phase, complex multiply), result **cast back to q's dtype** at the final `stack` (`rope.py:65-66`). No int/u32 here; codes/ids never reach this op (it sits inside the codec attention, post-projection).

## Wiring
**Upstream (feeds q/k into rope):** the Mimi codec's encoder/decoder `StreamingTransformer` self-attention — `q,k,v` come from the fused in-projection inside [moshi_transformer](MO03-Codec-Transformer) (`StreamingMultiheadAttention`, q/k of shape `[B,8,T,64]`, model dtype). `offset` (int64 `[B]`) comes from the streaming `_MHAState`. Those transformers are embedded in [moshi_compression](MM01-Mimi-Codec) (MimiModel enc/dec), which itself runs inside the SEANet pipeline of [moshi_seanet](MO01-SEANet).

**Downstream (consumes rotated q/k):** the rotated `(qo, ko)` flow straight back into `StreamingMultiheadAttention` of [moshi_transformer](MO03-Codec-Transformer) — KV-cache completion then `F.scaled_dot_product_attention`. The codec result ultimately surfaces to [moshi_compression](MM01-Mimi-Codec) `encode`/`decode`, i.e. the RVQ tokens consumed by [moshi_vq](QZ01-Split-RVQ) and the waveform produced for [core_processor](CO01-Processor-ChatState). RoPE has no direct tensor consumer outside the attention that calls it.

## Python ↔ Rust
The Rust side has **two** RoPE implementations, and `moshi/modules/rope.py` maps to neither of the in-tree ones — it maps to the upstream **`moshi` crate**:

- **`RotaryEmbedding` / `apply_rope` (codec) → `moshi` crate's Mimi rope.** liquid-audio-rs does NOT re-port the codec; it depends on Kyutai's own `moshi = "0.6"` crate and loads Mimi via `use moshi::mimi` (`loader.rs:23`, `Cargo.toml:51`), wrapped by [audio_out.rs](../../audio_out.md). The codec's interleaved RoPE (theta 10000, head_dim 64) therefore lives inside that crate's mimi transformer, not in any local `.rs`. This is the deliberate **moshi-crate-reuse** divergence: candle-transformers ships the same Mimi algorithm but with HF-format RVQ weight-key names (`…semantic_residual_vector_quantizer.*`); the checkpoint uses `quantizer.rvq_first`/`rvq_rest`, which `moshi::mimi` reads natively (`Cargo.toml:38-51`). Vendoring would be a fork of an identical algorithm — explicitly rejected.
- **In-tree `precompute_freqs_cis` + `apply_rotary_emb` (`transformer.rs:210,228`)** are the *depthformer / `RawLMBackbone`* RoPE, NOT this file's. They are the **same interleaved (GPT-J) convention** as `apply_rope` — candle has no complex dtype, so the complex multiply is the real-valued `candle_nn::rotary_emb::rope_i_slow` fed a `(cos, sin)` table built locally (no candle table-builder exists; `transformer.rs:205-209`). Deliberate choice: **`rope_i_slow`, not the fused `rope_i`** — the fused op uses `apply_op3_no_bwd` and severs autograd; the depthformer is trainable, so the differentiable slow path is required (`transformer.rs:17,725-740`).
- **`lfm2_hf.rs:218 rope()`** is the LFM2 *backbone* RoPE — a **different convention** (`rope_slow`, half-split NeoX) with **theta 1,000,000** (`lfm2_hf.rs:74`), matching HF `Lfm2Model`. Unrelated to this file beyond sharing the RoPE concept.

Off-CUDA, the `@torch_compile_lazy` fusion (`rope.py:11`) is a no-op; candle has no torch.compile, so the moshi-crate rope just runs eager candle ops — a benign **eager-vs-compiled** divergence.

## Precision / gotchas
- **f32 rotation floor.** Frequencies, phase, and the complex multiply are computed in f32 even when q/k are bf16 (`rope.py:37,50-54`); only the final write is cast back. This matches the cross-library policy where the precision-sensitive trig stays f32 — do not "optimize" it to bf16, it changes phase accuracy at large `offset`.
- **Interleaved ≠ half-split.** This file is GPT-J interleaved (`view(D/2,2)`, `rope.py:46`). The LFM2 backbone RoPE is NeoX half-split. Mixing the two conventions, or feeding this op's q/k to a half-split applier, silently corrupts positions — they are not interchangeable despite both being "RoPE".
- **Per-batch `offset` is a tensor, not a scalar.** `offset.float().view(-1,1)` (`rope.py:39`) means each batch element can sit at a different streaming position (used by `exec_mask`/ragged streaming). A scalar-offset port would be wrong for batched streaming.
- **`max_period` divisor edge case.** The codec rope here uses `exp(ds·(-log P·2/D))` (`rope.py:38`), which is well-defined for all `D≥2`. Note the *sin/cos absolute* positional helper in the same module file (`create_sin_embedding`, used for the `"sin"`/`"sin_rope"` strategies) divides by `(half_dim-1)` and would div-by-zero at `half_dim==1`; that path is not this component but shares the `max_period` config.
- **Theta mismatch across the three RoPEs.** Codec = 10000 (this file), depthformer = 10000, LFM2 backbone = 1,000,000. Using the wrong theta is a position-encoding bug, not a crash; the Rust `precompute_freqs_cis` keeps theta as an explicit arg precisely to avoid hard-coding it.
- **v is never rotated** — only q and k. A port that rotates v breaks attention.
