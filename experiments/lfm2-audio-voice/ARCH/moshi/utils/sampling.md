# moshi_util_sampling
**Code:** `MU01` · **Source:** `moshi/utils/sampling.py` · **Rust:** `candle LogitsProcessor (analog)` · **On the LFM2-Audio inference path:** no

## Role
A 128-line standalone next-token sampler for the **Moshi 7B LM** (`moshi/models/lm.py`), not for LFM2-Audio. It provides `sample_token(logits, use_sampling, temp, top_k, top_p)` — greedy argmax, or temperature-softmax with classic (sort/cumsum) top-p or top-k — plus a sync-free `multinomial` specialization. It lives in the vendored Kyutai tree and is reused only by the Moshi LM and Moshi TTS; **LFM2-Audio uses its own inline samplers** (`_sample_text_token` / `_sample_audio_frame` in `model/lfm2_audio.py`) with a different top-k convention, so this file never executes on the LFM2-Audio mic→wav path.

## How it works
Four functions, all operating on the last (candidate/`Card`) dimension. No grad, no state, pure tensor ops.

**`multinomial(input, num_samples, replacement=False)` (`sampling.py:15`)** — a `torch.multinomial` wrapper that flattens to `(-1, Card)` (`:31`) and, for the hot path `replacement=False, num_samples==1`, *avoids* `torch.multinomial`'s CUDA synchronization point with the Gumbel-max / exponential trick (`:43-46`): draw `q ~ Exponential(1)` (`empty_like(input_).exponential_(1)`), form `input_/q`, and take `argmax(dim=-1)`. This is the inverse-CDF identity `argmax_i(p_i / e_i)` with `e_i ~ Exp(1)` ≡ a categorical draw from `p` — same distribution as `multinomial`, but a single elementwise+argmax with no device sync. Otherwise it calls `torch.multinomial` directly (`:37`). Output is reshaped back to `input.shape[:-1] + (num_samples,)` (`:47`). Note: `input` here is a **probability** vector (or any non-negative weights), not logits.

**`sample_top_k(probs, k)` (`sampling.py:51`)** — `k = min(k, Card)` (`:60`), then `torch.topk(probs, k, dim=-1)` returns the `k` largest probs and their indices (`:61`); `multinomial(probs, 1)` samples a *position* within those `k` (`:62`); `indices.gather(-1, next_token)` maps the position back to the vocab id (`:63`). This is the **fixed-cardinality** top-k: exactly `k` survivors, ties at the boundary broken arbitrarily by `topk`. (Contrast with LFM2-Audio's threshold top-k below.)

**`sample_top_p(probs, p)` (`sampling.py:67`)** — nucleus sampling. Sort probs descending (`torch.sort`, `:76`), cumulative sum `probs_sum` (`:77`), build `mask = probs_sum - probs_sort > p` (`:78`) — i.e. keep a token iff the cumulative mass *strictly before* it is ≤ `p` (so the first token that crosses `p` is still kept). Zero out the masked tail by multiplying with `(~mask).float()` (`:79`), renormalize in place `probs_sort.div_(probs_sort.sum(-1, keepdim))` (`:80`), `multinomial(...,1)` over the truncated-renormalized distribution (`:81`), and `gather` the sorted index back to vocab space (`:82`).

**`sample_token(logits, use_sampling=False, temp=1.0, top_k=0, top_p=0.0)` (`sampling.py:86`)** — the dispatcher. If `use_sampling and temp > 0`: `probs = softmax(logits / temp, dim=-1)` (`:96`), then top-p if `top_p>0` (`:98`), else top-k if `top_k>0` (`:100`), else plain `multinomial` (`:102`). Else (greedy / `temp<=0`, the zero-division guard noted at `:94`): `torch.argmax(logits, dim=-1, keepdim=True)` (`:104`). Asserts the trailing sample dim is 1 and squeezes it (`:105-106`), returning shape `[*]`. **Precedence: top-p wins over top-k** when both are set — they are mutually exclusive here, unlike samplers that compose them.

**No softmax/argmax precision tricks beyond torch defaults.** `softmax` runs in the logits' dtype; the Moshi LM call sites upcast first — `sample_token(text_logits.float(), ...)` (`lm.py:730`, `:827`) — so the softmax effectively runs in **f32**. Temperature scaling is a plain `logits / temp` (no log-space). No streaming state; each call is independent.

**Why it is off the LFM2-Audio path.** `sample_token` is imported only at `moshi/models/lm.py:25` and called for the Moshi LM's text stream (`lm.py:730`) and its depformer codebooks (`lm.py:827`). LFM2-Audio's `_sample_text_token` (`lfm2_audio.py:486-497`) and `_sample_audio_frame` (`lfm2_audio.py:519-529`) reimplement sampling inline with a **threshold** top-k: `min_score = topk(logits, k).values[-1]; logits[logits < min_score] = -inf; multinomial(softmax(logits),1)` — which *keeps ties* at the k-th value (variable survivor count), the opposite of this file's fixed-`k` `gather`. They also have **no top-p** path. So the two samplers are not interchangeable.

## Dtypes & shapes
| Function | Input(s) | Output |
|---|---|---|
| `multinomial` | `input` probs `(…, Card)` float (Moshi: f32 after upcast) | `(…, num_samples)` int64 |
| `sample_top_k` | `probs (…, Card)` f32, `k:int` | `(…, 1)` int64 |
| `sample_top_p` | `probs (…, Card)` f32, `p:float` | `(…, 1)` int64 |
| `sample_token` | `logits (…, Card)` f32 (callers pass `.float()`); for Moshi text `Card=text_card`, depformer `Card=audio_card` | `(…,)` int64 (squeezed) |

Internal promotions: `(~mask).float()` upcasts the boolean mask to f32 for the multiply (`:79`); `exponential_` draws in `input`'s dtype. The Moshi LM forces **f32** softmax via `.float()` at the call site (`lm.py:730/827`). No f64 anywhere. Token ids returned are int64 (torch `argmax`/`multinomial` default LongTensor).

## Wiring
**Upstream (feeds this):**
- [moshi_lm](../models/lm.md) — Moshi LM text logits `f32 (B,1,1,text_card)` (`lm.py:730`, upcast via `.float()`) and per-codebook depformer logits `f32 (B,1,1,audio_card)` (`lm.py:827`). These are the *only* on-tree producers.
- [moshi_tts](../models/tts.md) — drives the Moshi LM's `LMGen`, so it reaches this sampler transitively (off-path).

**Downstream (consumes this output):**
- [moshi_lm](../models/lm.md) — the sampled token id `int64 (B,)` is appended to the Moshi LM's text/audio streams and re-embedded for the next `lm_gen.step` (`lm.py` `LMGen`). This is the sole consumer.

**Not wired to:** [model_lfm2_audio](../../model/lfm2_audio.md) (uses its own threshold-top-k samplers), [core_processor](../../processor.md), or any LFM2-Audio decode component. There is no edge from this file into the LFM2-Audio tensor path.

## Python ↔ Rust
There is **no direct Rust port of `sampling.py`** — it belongs to the Moshi-LM reference subsystem, which is reused (not re-ported) from Kyutai's `moshi` crate. The *analogous* sampler that LFM2-Audio's Rust port actually runs is the `Sampler` struct in `model/lfm2_audio.rs:174-207`, built on `candle_transformers::generation::{LogitsProcessor, Sampling}`.

| Python (`sampling.py`) | Rust analog | Notes |
|---|---|---|
| `sample_token(...)` greedy branch (`:104`) | `Sampling::ArgMax` → `LogitsProcessor::sample_argmax` (`lfm2_audio.rs:189-190`) | `argmax(-1)`; deterministic ⇒ parity preserved (depthformer token-exact). |
| `sample_token(...)` stochastic (`:96-102`) | `Sampling::All { temperature }` → `LogitsProcessor::sample` (`lfm2_audio.rs:193`) | temperature softmax + multinomial. |
| `sample_top_k` fixed-`k` gather (`:51-64`) | `torch_topk_mask` via `LogitsProcessor::sample_f` (`lfm2_audio.rs:202,215-228`) | Rust mirrors **LFM2-Audio's threshold top-k** (`p < min_score → 0`, ties kept), *not* this file's fixed-`k`. candle's built-in `Sampling::TopK` keeps exactly `k` and was deliberately bypassed. |
| `sample_top_p` (`:67-83`) | (none) | LFM2-Audio has no top-p; not implemented in the port. |
| `multinomial` Gumbel/exponential no-sync trick (`:43-46`) | candle `WeightedIndex` (rand_pcg) inside `LogitsProcessor` | Different RNG stream; not byte-reproducible by design. |

**Deliberate divergences** (per `PYTHON_VS_RUST.md`): §2.3 "Sampling → `candle_transformers::generation::LogitsProcessor` (the sampler moshi itself uses), wrapped by a `Sampler` that injects Torch's threshold-style top-k via the `sample_f` hook; greedy = `ArgMax`"; §2.8 stochastic RNG is `rand_pcg`, not torch's `multinomial` generator — "the token set and proportional distribution match, but not byte-reproducible." Greedy is fully deterministic and identical.

## Precision / gotchas
- **Wrong model.** Do not treat this as the LFM2-Audio sampler — it is the **Moshi LM** sampler. The two differ in top-k semantics (fixed-`k` gather here vs threshold-`< min_score` in LFM2-Audio) and in top-p support (present here, absent in LFM2-Audio).
- **top-p vs top-k mutual exclusion.** `sample_token` checks `top_p>0` *before* `top_k>0` (`:97-100`); you cannot stack them, and top-p silently wins.
- **`temp<=0` guard.** `temp` at or below 0 (or `use_sampling=False`) routes to greedy argmax to dodge the `logits/temp` zero-division (`:94-95`) — so "temperature 0" means deterministic, never a degenerate softmax.
- **`multinomial` expects probabilities, not logits.** `sample_top_k/top_p` pass already-softmaxed `probs`; the no-sync trick `input_/q` assumes non-negative weights. Feeding raw logits would be a silent correctness bug.
- **top-p keep-rule off-by-one.** The mask `probs_sum - probs_sort > p` (`:78`) keeps the *first* token that crosses `p` (compares cumulative-before-this-token), so the nucleus always contains ≥1 token even when `top1 > p`.
- **RNG non-reproducibility.** Stochastic draws are not byte-reproducible across the torch reference and the candle analog (different generators); only greedy/argmax matches bit-for-bit. EOAudio (code 2048) and EOS handling are decided by the *callers* (`generate_interleaved` / `LMGen`), not by this sampler — it only returns the argmax/multinomial index.
