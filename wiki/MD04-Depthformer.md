<!-- topic: Model -->
# MD04 · RawLMBackbone depthformer
**Code:** `MD04` · **Source:** `model/transformer.py` · **Rust:** `model/transformer.rs` · **On the LFM2-Audio inference path:** yes

## Role
This file is the **depthformer** — the small autoregressive transformer that turns one backbone hidden vector into an 8-codebook audio frame, one codebook at a time. It defines the reusable sequence-model primitives (`RMSNorm`, `GLU`/SwiGLU, `MHA` with `BoundedAttention`, `StandardBlock`, `SharedEmbedding`) and assembles them as `RawLMBackbone`. LFM2-Audio instantiates exactly one `RawLMBackbone` here: a **6-layer × 1024-dim** depthformer (`lfm2_audio.py:115-121`) driven inside `_sample_audio_frame` (`lfm2_audio.py:501-534`). It exists because the main LFM2 backbone (the external HF `Lfm2Model`) only emits a per-step hidden state and a *text* logit; the residual-vector-quantized audio codes need their own depth-wise causal model to be sampled coarse-to-fine within a single time step.

## How it works

**Stack shape.** `RawLMBackbone(layers, has_embedding=False)` (`transformer.py:510`, built at `lfm2_audio.py:121`) is a `ModuleList` of `StandardBlock`s with **no** embedding/logit head of its own (the audio token embedding + logit projection live in the separate `depth_embeddings` list and `depth_linear`, see Wiring). Each block wraps an `MHA` operator constructed as `MHA(depthformer_dim=1024)`, taking the class defaults: `num_heads=32`, `head_style="gqa"`, `gqa_dim=8` (KV heads), `qk_layernorm=True`, `norm_eps=1e-5`, `theta=1e6`, `max_seq_len=128_000`. So `head_dim = 1024/32 = 32`; query heads = 32, KV heads = 8 → GQA repeat factor 4.

**StandardBlock** (`transformer.py:378-390`) is pre-LN with two residuals:
```
h   = x + operator(operator_norm(x))      # attention sub-block
out = h + feed_forward(ffn_norm(h))       # SwiGLU sub-block
```
Both norms are `RMSNorm`; `feed_forward` is the `GLU`.

**RMSNorm** (`transformer.py:65-82`). `_norm(x) = x * rsqrt(mean(x², -1, keepdim) + eps)`, `eps=1e-5` (or `1e-6` for the base default; depthformer passes `1e-5`). The exact op order in `forward` (`:77-78`) is the load-bearing detail: **upcast to f32 first** (`x.float()`), compute the normalize, then **multiply the weight while still in f32**, *then* cast back to input dtype: `(_norm(x.float()) * self.weight).type_as(x)`. This is normalize-in-f32 → weight-multiply-in-f32 → cast, **not** the cast-then-multiply order that `candle_nn::RmsNorm`/moshi use.

**GLU / SwiGLU** (`transformer.py:84-134`). `use_swiglu=True` path: `w2(silu(w1(x)) * w3(x))` (`:132`). SiLU = `x·sigmoid(x)`. The hidden width sizing (`:101-108`) is: start `ff_dim = 4*dim` if unset, then `ff_dim = int(2*ff_dim/3)`, apply `ffn_dim_multiplier` (1.0), then round up to a multiple of `multiple_of=256`: `ff_dim = 256 * ceil(ff_dim/256)`. For `dim=1024` → `4*1024=4096` → `2/3·4096=2730` → round up to `2816`. The non-swiglu branch uses `gelu` (erf form in Rust via `gelu_erf`).

**MHA / BoundedAttention** (`transformer.py:140-341`). `MHA.forward_cached` (`:306`) does a single fused `qkv_proj` (no bias) of width `total_width = dim + 2·head_dim·gqa_dim` for GQA (`:262-264` → `1024 + 2·32·8 = 1536`), then splits into `xq (1024), xk (256), xv (256)` (`:321-325`). It slices the RoPE table cache-aware: `freqs_cis[cache_size : cache_size+seq_len]` (`:329-335`). `BoundedAttention.forward` (`:171-226`) reshapes to `[b,t,heads,head_dim]` (q:32 heads, k/v:8 heads, `:189-192`), applies **qk-RMSNorm** per head (`q_layernorm`/`k_layernorm` over `head_dim=32`, `:194-196`), applies interleaved RoPE (`:198-199`), updates the KV cache (`:201-202`), transposes to `[b,heads,t,head_dim]`, and calls PyTorch SDPA with `enable_gqa=True`.

**Causal masking is shape-dependent** (`:212-221`): if `q_len == kv_len` (prefill) it uses `is_causal=True`; if `q_len == 1` (the normal decode step, which is how the depthformer is driven) `attn_mask=None` (a single query attends to all cached keys); otherwise it builds a `causal_lower_right(q_len, kv_len)` mask (lower-right-aligned causal — query `i` attends keys `≤ i + (kv_len-q_len)`). SDPA applies the `1/sqrt(head_dim)` scale internally.

**RoPE is interleaved (GPT-J style), not half-split.** `precompute_freqs_cis` (`:450-470`) builds `inv_freq = 1/theta^(arange(0,dim,2)/dim)`, `theta=1e6`, then `freqs = outer(t, inv_freq)` and `polar(1, freqs)` → complex64 table of shape `[end, head_dim/2]`. `apply_rotary_emb` (`:393-422`) **upcasts q/k to f32** (`xq.float()`), reshapes the last dim as adjacent `(D, 2)` pairs via `view_as_complex`, multiplies by the complex `freqs_cis`, `view_as_real`+flatten, then `type_as` back to the input dtype. Pairing **adjacent** elements (`(D two)`, `two=2`) is the interleaved convention.

**SharedEmbedding** (`transformer.py:473-507`) is an `Embedding` + a pre-logits `RMSNorm` (`embedding_norm`) + an untied-or-tied `to_logits` linear. `get_logits(e) = to_logits(embedding_norm(e))` (`:506-507`) — note the **norm is applied before the logit projection**, not after the embedding lookup. The depthformer does not use this class internally (`has_embedding=False`); the eight per-codebook `depth_embeddings` (each a `SharedEmbedding(dim=1024, vocab=2049)`) provide both the input token embedding and the output `get_logits`.

**The actual decode loop** lives in `_sample_audio_frame` (`lfm2_audio.py:501-534`), not in this file, but it is the only caller of `RawLMBackbone.forward_cached`. Given one backbone hidden `embedding` (lfm hidden_size = 2048), it does `depth_linear` (2048 → `1024·8`) and reshapes to `[C=8, D=1024]` (`:509`). Then it loops the 8 codebooks coarse→fine:
1. `cur = depthformer_in[i] + depthformer_token` (the previously sampled codebook's embedding, zero for i=0) (`:515`).
2. `out, cache = depthformer.forward_cached(cur[None,None,:], cache)` — a `q_len==1` step that grows the per-layer KV cache (`:516`).
3. `logits = depth_embeddings[i].get_logits(out.squeeze())` over vocab 2049 (`:517`).
4. Sample: greedy `argmax` (`:520`) or `temperature` + threshold `top_k` + `softmax` + `multinomial` (`:523-529`).
5. `depthformer_token = depth_embeddings[i](next_token)` feeds step i+1 (`:532`).

So the depthformer runs **8 sequential single-token steps per audio frame**, the KV cache spanning the 8 codebook positions, all conditioned on the one backbone hidden via the additive `depth_linear` projection. EOAudio is code index **2048** (vocab `2048+1`); a `2048` in codebook 0 forces the whole frame to 2048 (`lfm2_audio.py:226-227,300-301`).

**KV cache** (`LayerKVCache`, `transformer.py:38-62`) concatenates `k`/`v` along `dim=1` (time) pre-transpose, i.e. shape `[b, t, heads, head_dim]`. `RawLMBackbone.forward_cached` (`:554-566`) threads a per-layer list `[None]*n_layers` on the first step and returns the updated list, iterating with `zip(..., strict=True)`.

## Dtypes & shapes

| Stage | Input | Output |
|---|---|---|
| `depth_linear` (in lfm2_audio, feeds this) | backbone hidden `(2048,)` model dtype (bf16/f32) | `(1024·8,)` → reshape `[8,1024]` |
| `RawLMBackbone.forward_cached` (one step) | `[1,1,1024]` model dtype | `[1,1,1024]` model dtype |
| `RMSNorm` internal | x model dtype | upcast **f32** for `_norm`+weight-mul, cast back to model dtype |
| `apply_rotary_emb` internal | q/k model dtype | upcast **f32** for the rotation, `type_as` back |
| qkv_proj (GQA) | `[1,1,1024]` | `[1,1,1536]` → split q`[…,1024]` k`[…,256]` v`[…,256]` |
| SDPA scores/softmax | q/k/v | scores accumulated in **f32** (SDPA), out cast back to model dtype |
| KV cache tensors | k,v `[1,t,8,32]` model dtype | grows to `[1,t+1,8,32]` |
| `depth_embeddings[i].get_logits` | `[1024]` model dtype | logits `(2049,)` |
| sampled token | logits `(2049,)` | int64 scalar (code 0..2048; 2048=EOAudio) |
| frame (8 codebooks) | — | `(8,)` int codes |

`freqs_cis` table is complex64 (Python) / `(cos,sin)` f32 (Rust) of `[max_seq_len, head_dim/2=16]`. Token ids into `depth_embeddings` are int64; embedding weights are model dtype (bf16 on disk / f32 on Rust CPU / bf16 on Metal).

## Wiring

**Upstream.**
- The backbone hidden that drives every audio frame comes from [model_lfm2_backbone](MD02-LFM2-Backbone) via the top model: `(1,L,2048)` model dtype hidden → `depth_linear` → `[8,1024]` per-codebook seeds. (Top model: [model_lfm2_audio](MD01-LFM2AudioModel).)
- The per-codebook input/output embeddings are the eight `SharedEmbedding`s (defined in this file) owned by `model_lfm2_audio` as `depth_embeddings`; each carries the previously sampled code's embedding `(1024,)` back into the loop.

**Downstream.**
- The sampled 8-code audio frame is consumed by [model_lfm2_audio](MD01-LFM2AudioModel) `_sample_audio_frame`/`generate_*`, which re-embeds it through `audio_embedding` for the next backbone step and emits the frame.
- The accumulated audio codes `(8,)` int per frame flow to the codec for waveform synthesis: either [core_detokenizer](CO02-Detokenizer) (LFM2 ISTFT vocoder) or [moshi_compression](MM01-Mimi-Codec) (`MimiModel.decode`), both via [core_processor](CO01-Processor-ChatState) `decode()`. Edge: `int`/`u32` codes 0..2047 (2048=EOAudio terminates the turn, not decoded) → `f32` waveform @ 24 kHz.

## Python ↔ Rust

| Python (`transformer.py`) | Rust (`transformer.rs`) | Note |
|---|---|---|
| `RMSNorm` | `RmsNorm` | f32 normalize + **weight-multiply-in-f32** then cast (§2.4); not a `candle_nn::RmsNorm` wrap |
| `GLU` | `Glu` | identical `ff_dim` sizing; `silu`/`gelu_erf` via `candle_nn` |
| `BoundedAttention` | `BoundedAttention` | hand-rolled SDPA: matmul + additive mask + `softmax` + matmul + `repeat_kv` (§2.2) |
| `MHA` / `LayerKVCache` | `Mha` / `LayerKvCache` (over vendored `ConcatKvCache`) | cat on dim=1 (§2.3) |
| `StandardBlock`, `SharedEmbedding`, `RawLMBackbone` | `StandardBlock`, `SharedEmbedding`, `RawLmBackbone` | structural 1:1 |
| `apply_rotary_emb` (`view_as_complex`) | `rope_i_slow(q,cos,sin)` | candle has no complex dtype → interleaved real rotary; **`rope_i_slow` not fused `rope_i`** because the depthformer runs inside the trainable graph and the fused op severs autograd (test `rotary_is_differentiable`) |
| `precompute_freqs_cis` → complex64 | returns `(cos,sin)` f32 | same `polar(1,outer(t,inv_freq))` math |
| `scaled_dot_product_attention(enable_gqa)` | manual matmul SDPA + `repeat_kv` | eager **sdpa/no-flash** math, the path the f32 goldens were dumped from (§2.2) |
| `wrap_activation_checkpoint` | identity passthrough | train-only, no autograd on inference path (§2.5) |

**Deliberate divergences** (all from PYTHON_VS_RUST.md): device-agnostic (no hard-coded `.cuda()`), CUDA kernels → portable candle ops (§2.2), KV-cache reuse via `ConcatKvCache` (§2.3), the RMSNorm bf16 weight-multiply order (§2.4). Rust also hardens `RawLMBackbone.forward` to **error on cache-length mismatch** rather than silently running a prefix (Python's `zip(strict=True)` would raise; candle's `zip` would not). Parity: backbone hidden 6.558e-6, text logits 5.505e-6, **depthformer audio frame token-EXACT** `[213,836,182,416,782,1796,202,578]`.

## Precision / gotchas

- **RMSNorm order is the subtle one.** `(_norm(x.float()) * weight).type_as(x)` multiplies the weight **in f32** before casting; a naive candle/moshi RMSNorm casts to input dtype first and multiplies in bf16. At bf16 these differ; the Rust port deliberately reproduces the f32-multiply order (§2.4). f32 parity is unaffected.
- **RoPE is interleaved** (adjacent-pair `view_as_complex`), **theta = 1e6** (not the 10000 base default of `precompute_freqs_cis`). Using a half-split rotary or the wrong theta would silently break parity.
- **qk-RMSNorm is per-head over head_dim=32** (`norm_eps=1e-5`), applied before RoPE — easy to drop and still "look" correct.
- **The cross-library f32 floor (~1e-6)** is irreducible: candle gemm reduction order, `exp`/`cos`/`sin`/`rsqrt` last-bit differences. candle has no fused `rsqrt`, so RMSNorm uses `recip(sqrt(z))` — ~1 ULP from torch's fused `rsqrt` (§1.4). Despite this floor, the depthformer output is **token-exact** because argmax over logits is robust to sub-1e-6 perturbations.
- **EOAudio = code 2048**; audio vocab is `2048+1=2049`. A `2048` sampled in codebook 0 forces the entire 8-code frame to 2048 and ends the audio turn (`lfm2_audio.py:226-227,300-301`); it is never sent to the codec for decoding.
- **Mask shape branching:** the depthformer always decodes with `q_len==1` so it hits the `attn_mask=None` branch (`transformer.py:215-216`) — the off-by-one-prone `causal_lower_right` path (`q_len>1`, `q_len<kv_len`) is only exercised by mismatched-length prefill, which this caller does not use.
- **`tie_embedding` is a train-time flag only:** the checkpoint always ships a concrete `to_logits.weight` (equal to the embedding matrix when tied, e.g. depthformer `tie=True`; distinct when untied, e.g. `audio_embedding` with `tie_audio_embeddings=False`). The Rust loads `to_logits.weight` from disk rather than assuming the tie — faithful for both.
