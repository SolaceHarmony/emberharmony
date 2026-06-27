# moshi_util_utils
**Code:** `MU05` · **Source:** `moshi/utils/utils.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
A single vendored Moshi-LM training utility: a `@torch_compile_lazy`-decorated `cross_entropy` that computes **per-codebook** cross-entropy between a multi-stream LM's logits `[B,K,T,card]` and integer target codes `[B,K,T]`, masking out invalid timesteps. It exists because PyTorch's built-in `F.cross_entropy` is "super slow with large cardinality" (the comment at `utils.py:46-47`) for the multi-codebook audio LM, so Moshi reimplements it as a chunked manual log-partition. **It is not part of LFM2.5-Audio** — nothing in `liquid_audio` (and, as verified, nothing else in the vendored `moshi/` tree) imports it. LFM2-Audio's own training loss uses stock `nn.functional.cross_entropy(reduction="none")` at `model/lfm2_audio.py:460,462` instead. Despite the id-map gloss ("misc helpers, defaultdict, etc."), this file contains exactly one symbol: `cross_entropy`.

## How it works
The forward is a hand-rolled negative-log-likelihood reimplemented to dodge two PyTorch pain points (slow large-cardinality CE; f32-upcast OOMs), with optional logit soft-clipping (`utils.py:7-52`).

1. **Shape contract + flatten** (`:25-30`). Asserts `logits.shape[:-1] == targets.shape == mask.shape`. Records `output_shape = targets.shape` (`[B,K,T]`), then flattens: `logits → (-1, card)`, `targets → (-1,)`, `mask → (-1,)`. K (codebooks) and T (time) are collapsed into one row axis `N = B·K·T`; the per-codebook structure is preserved only by re-`view`ing at the end, so this is genuinely a per-codebook loss (no reduction across K).

2. **Safe targets** (`:32-36`). `safe_targets = where(mask, targets, 0)` — masked-out positions get a dummy class index 0 so the later `gather` is in-bounds; their contribution is zeroed afterward. `mask` is a boolean tensor; the fill value is `zeros(1, dtype=targets.dtype)` (an int code).

3. **Chunked manual cross-entropy in `dtype` (default f32)** (`:38-49`). To bound peak memory, both flattened `logits` and `safe_targets` are split into **4** chunks along the row axis (`torch.chunk(..., 4)`), and the f32 upcast + the rest happen per chunk:
   - `logits_chunk = logits_chunk.to(dtype)` — upcast bf16→f32 only on the live chunk (`:41`).
   - **Optional soft-clip** (`:42-43`): `logits = soft_clip · tanh(logits / soft_clip)` — a smooth bound to `±soft_clip` (recommended 30.0) for numerical stability; identity-like near 0, saturating at the extremes. Skipped when `logits_soft_clip is None`.
   - **Log-partition** (`:44`): `log_partition = logsumexp(logits_chunk, dim=-1, keepdim=True)` — the stable `log Σ exp` normalizer, i.e. `logZ`.
   - **Per-row NLL** (`:48`): `ce_chunk = log_partition − logits.gather(-1, target[...,None])`. This is exactly `−log softmax(logits)[target] = logZ − logit_target`, the cross-entropy of one row, but computed without materializing the full softmax. `gather` pulls the target class's logit; `[..., None]` adds the gather axis.

4. **Reassemble + re-mask + reshape** (`:49-52`). Concatenate the 4 chunks back to `(N,1)`, squeeze the gather axis (`ce[...,0]` → `(N,)`), then `where(mask, ce, 0)` to zero invalid timesteps (the safe-target dummies), and finally `view(output_shape)` back to `[B,K,T]`. Result dtype = `dtype` (f32).

Numerically `logsumexp − gather` is the standard stable CE identity; the only non-obvious choices are the **4-way chunking** (an OOM guard around the f32 upcast, not a math change) and the **soft-clip pre-nonlinearity** (a regularizer/stabilizer, off by default). The `@torch_compile_lazy` decorator (`:6`, from `moshi/utils/compile.py`) JIT-fuses the whole thing under `torch.compile` lazily on first call; it degrades to plain eager when compilation is disabled (off-CUDA / `no_compile`).

## Dtypes & shapes
| Tensor | dtype | shape |
|---|---|---|
| `logits` (in) | model dtype, typically bf16 (Moshi LM) | `[B, K, T, card]` |
| `targets` (in) | int64 codes | `[B, K, T]` |
| `mask` (in) | bool | `[B, K, T]` |
| `dtype` (param) | `torch.float32` default | — |
| `logits_soft_clip` (param) | f32 scalar or `None` | — |
| `safe_targets` (internal) | int64 (= `targets.dtype`) | `(N,)`, `N=B·K·T` |
| `logits_chunk` after `.to(dtype)` (internal) | **f32** (upcast) | `(N/4, card)` |
| `log_partition` (internal) | f32 | `(N/4, 1)` |
| `ce` (out) | f32 (= `dtype`) | `[B, K, T]` |

Internal promotions: bf16 logits are **upcast to f32 per chunk** before `logsumexp`/`gather` (the standard CE-in-f32 rule, here applied chunk-wise as an OOM guard). Targets stay int64 throughout. The mask is boolean and never participates in arithmetic except via `where`.

## Wiring
**Upstream (in Moshi, NOT in LFM2-Audio):** the Moshi multi-stream LM's text+audio logit heads produce `[B,K,T,card]` and the training pipeline supplies integer code targets + a validity mask. The natural caller would be Moshi's training/loss code paired with [moshi_lm](../models/lm.md) (the 7B multi-stream LM) and its sampler/loss utilities; `compile.py`'s `torch_compile_lazy` ([moshi_util_compile](compile.md)) is its only intra-file dependency. **No call site exists in this checkout** — grep confirms zero importers across `moshi/` and `liquid_audio/`.

**Downstream:** none on the LFM2-Audio inference path. The function returns a `[B,K,T]` f32 loss tensor that a (here-absent) Moshi trainer would reduce/backward. For the LFM2-Audio analog of "per-element CE," see [model_lfm2_audio](../../model/lfm2_audio.md), whose `forward` calls stock `F.cross_entropy(reduction="none")` (`lfm2_audio.py:460,462`) — a *different* function, ported to Rust as `candle_ext::loss::cross_entropy_none`, not as a port of this file.

## Python ↔ Rust
**No Rust counterpart.** This `cross_entropy` is not ported because it has no call site in the inference port (the `core` parity scope explicitly excludes vendored `moshi/**`; see `PYTHON_VS_RUST.md` §4 "Out of scope / reused, not ported", and §2.3 which reuses the Mimi codec from the `moshi` crate rather than re-porting `moshi/` Python).

The superficially similar Rust symbol — `candle_ext::loss::cross_entropy_none` (`src/candle_ext/loss.rs:20`) — is the port of the **stock** `nn.functional.cross_entropy(reduction="none")` used by LFM2-Audio's own loss (`lfm2_audio.py:460,462` → `lfm2_audio.rs:581-582`), NOT of this Moshi helper. It differs in mechanism: it calls candle's `log_softmax` after a single f32 upcast over the whole tensor (no 4-way chunking, no soft-clip, no per-codebook `[B,K,T]` layout — it operates on flat `(N,C)`/`(N,)`). See `PYTHON_VS_RUST.md` §2.3 ("`cross_entropy(reduction="none")` → `candle_ext::loss::cross_entropy_none`"). The `@torch_compile_lazy` decorator maps to candle's no-op compile story (`PYTHON_VS_RUST.md` §2.2: CUDA-gated `torch.compile` → portable eager candle ops; `compile.md`).

## Precision / gotchas
- **f32 CE floor.** The upcast-to-f32-before-`logsumexp` is the same numerical rule the rest of the port honors for softmax/norm (`PYTHON_VS_RUST.md` §1.4); any port would inherit the cross-library `exp`/`logsumexp` last-bit floor (~1e-6) — but there is no port here, so this is informational.
- **Chunking is memory, not math.** The 4-way `torch.chunk` is purely an OOM guard around the f32 upcast (comment `:38`); it does not change results vs. an un-chunked computation (concat restores exact order). Do not mistake it for a streaming/blocked-softmax approximation — each chunk's `logsumexp` is over the full `card` axis, exact.
- **Masked positions read class 0.** `safe_targets` substitutes index 0 at masked rows so `gather` stays in-bounds; correctness depends on the **final** `where(mask, ce, 0)` zeroing them — if a caller dropped that second mask, masked rows would leak the CE of class 0. Both masks are load-bearing.
- **Soft-clip is opt-in.** `logits_soft_clip=None` by default; passing 30.0 bounds logits via `tanh` before `logsumexp` (changes the loss, intended as a stabilizer for the multi-codebook head). It is applied to logits *before* the partition, so it shifts both `logZ` and the gathered logit consistently.
- **Not LFM2-Audio's loss.** The single biggest gotcha: this is the **Moshi 7B** LM's loss, reference-only. LFM2.5-Audio's per-codebook `audio_loss_weights` weighting and `ignore_index=-100` masking live in `model/lfm2_audio.py:455-470`, using a different (stock-CE) code path. Citing this file as "the LFM2-Audio training loss" would be wrong.
