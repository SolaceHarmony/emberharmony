# moshi_lm
**Code:** `MM03` · **Source:** `moshi/models/lm.py` · **Rust:** `moshi crate lm (NOT used by LFM2-Audio)` · **On the LFM2-Audio inference path:** no

## Role
`LMModel` is Kyutai's **Moshi 7B multi-stream language model**: a single Transformer that jointly models one text stream and `n_q` (8) audio-codebook streams of Mimi codes, with a small **depformer** (depth transformer) that predicts the audio codebooks of a frame autoregressively given the backbone hidden state. `LMGen` is its streaming inference driver (delay-pattern cache + CUDA-graphed step). This is the *Moshi* architecture, a **different model** from LFM2.5-Audio — LFM2-Audio brings its own HF `Lfm2Model` backbone (`model_lfm2_backbone`) plus its own `RawLMBackbone` depthformer (`model_transformer`) and never instantiates `LMModel`. It is on-path only for the vendored `moshi/` server/TTS/run_inference scripts, which LFM2-Audio's runtime does not use. Reference only.

## How it works

### Construction (`lm.py:76`–235)
Two stacked transformers sharing most kwargs:
- **Main backbone** `self.transformer = StreamingTransformer(d_model=dim, num_heads, dim_feedforward=hidden_scale*dim, …)` (`lm.py:145`) — the wide temporal model over the muxed text+audio embedding stream. See `moshi_transformer` for its internals (RoPE, `F.scaled_dot_product_attention`, LayerScale 0.01, context window, KV cache).
- **Depformer** `self.depformer = StreamingTransformer(d_model=depformer_dim, …)` (`lm.py:198`) — a narrow transformer that runs **inside a single time step** over the `dep_q` codebooks (its sequence axis is *codebook index*, not time). `self.depformer.set_streaming_detached(True)` (`lm.py:212`) decouples its streaming cursor from the main step loop so it can be re-run per frame.

Embeddings (all `ScaledEmbedding` from `moshi_lm_utils`, scaled by `1/sqrt(dim)`, `zero_idx=-1` to make the `zero_token_id` row a hard zero, `norm_emb` optional RMS/LayerNorm on the looked-up vector):
- `self.emb[k]` for `k in 0..n_q`, each `card+1` rows (`lm.py:134`, the audio codebooks; `+1` = the `initial_token_id == card` SOS row).
- `self.text_emb`, `text_card+1` rows (`lm.py:138`).
- `self.depformer_emb[k]` for `k in 0..dep_q-1` and `self.depformer_text_emb` (`lm.py:188`,191) — the *teacher-forcing inputs to the depformer* (low-rank optional via `depformer_low_rank_embeddings`).

Heads:
- `self.text_linear: Linear(dim, text_card_out, bias=bias_proj)` (`lm.py:140`) — the text logits head off the main backbone.
- `self.depformer_in[i]: Linear(dim, depformer_dim, bias=False)` (`lm.py:178`/183) — projects the backbone hidden into depformer latent space; one-per-codebook iff `depformer_multi_linear`, else a single shared linear.
- `self.linears[k]: Linear(depformer_dim, card, bias=bias_proj)` for `k in 0..dep_q` (`lm.py:224`) — per-codebook audio logits off the depformer output.
- `self.extra_heads` (`lm.py:218`) — optional auxiliary `Linear(dim, extra_heads_dim)` softmaxed (e.g. VAD-style side outputs), `lm.py:797`.

Normalization op order is delegated entirely to `StreamingTransformer`/`create_norm_fn(norm, dim)` (`out_norm`, `lm.py:158`); `norm` defaults `"layer_norm"`. The depformer/backbone RMSNorm-vs-LayerNorm choice, eps, and the f32-multiply ordering are *not* implemented here — they live in `moshi_transformer`. (Contrast: LFM2-Audio's own depthformer in `model_transformer` does `(_norm(x.float())*w).type_as(x)`.)

### Delay pattern (the multi-stream trick)
`delays: list[int]` has length `num_codebooks = n_q+1` (text + audio). Audio codebooks are emitted with staggered delays (e.g. `[0,0,1,1,…]`, see `moshi_loaders`) so the backbone, at each step, conditions on already-committed codes of earlier codebooks. `_delay_sequence`/`_undelay_sequence` (from `moshi_lm_utils`) shear the `[B,K,T]` code grid along time per codebook and back. `_get_initial_token()` (`lm.py:300`) builds the `[B,K,1]` SOS column: text row = `text_card` (`text_initial_token_id`), every audio row = `card` (`initial_token_id`).

### Training forward (`forward`, `lm.py:316`)
1. `B,K,T = codes.shape`, assert `K == num_codebooks`.
2. Delay the codes and prepend the initial column: `delayed = cat([initial, _delay_sequence(delays, codes, initial)], dim=2)` (`lm.py:343`–346). The last delayed step is dropped (never an input).
3. `forward_text(delayed[:,:,:-1])` (`lm.py:373`): for each audio codebook sum `emb[k](seq[:,k+1])`, add `text_emb(seq[:,0])`, optionally add `sum_condition`, run the main transformer, apply `out_norm`, then `text_linear` → `text_logits [B,1,S,text_card]`. Returns `(transformer_out, text_logits)`.
4. `forward_depformer_training(delayed[:,:,1:], transformer_out)` (`lm.py:404`): for each of `dep_q` codebooks form `depformer_in[idx](transformer_out) + token_emb`, where `token_emb` is `depformer_text_emb(seq0)` for cb 0 else `depformer_emb[cb-1]`. Stack to `[B,T,Ka,D]`, **reshape to `[B*T, Ka, D]`** (each frame becomes an independent depformer sequence of length `Ka`), run `self.depformer`, then per-cb `linears[cb]` → `[B,Ka,T,card]`.
5. `_undelay_sequence` re-aligns logits to original code timeline, filling invalid positions with `NaN`; masks `&= codes != zero_token_id` (`lm.py:365`–370). Returns `LMOutput(logits, mask, text_logits, text_mask)`.

### Streaming inference (`LMGen._step`, `lm.py:662`)
This is the real per-frame engine (`@torch.no_grad`):
- State `cache: [B, num_codebooks, max_delay+2]` ring buffer initialized to `ungenerated_token_id (-2)`; `offsets` per-batch cursor; `condition_sum/cross` precomputed (`lm.py:599`–660). The whole step is wrapped in `lm_model.streaming(batch_size)` (KV cache live).
- Write the user-provided input codebooks into the ring at delayed write positions via `scatter_with_mask_` (skips masked/non-executing lanes, `lm.py:31`,689).
- Gather the current input column `[B,K,1]`; where `offsets <= delay` substitute the `initial` SOS token (`lm.py:692`–696). Optional **CFG**: duplicate the batch, build a text/condition-masked null branch, and after the backbone combine `logits_null + (logits - logits_null)*cfg_coef` (`lm.py:708`–726).
- `state.graphed_main(input_, cond_sum, cond_cross)` = CUDA-graphed `forward_text` → `transformer_out, text_logits`. **Sample text** with `sample_token(text_logits.float(), use_sampling, temp_text, top_k_text)` (`moshi_util_sampling`; note `.float()` upcast before sampling, `lm.py:730`).
- `depformer_step(text_token, transformer_out)` (`lm.py:803`): enter `depformer.streaming(B_cfg)`, then loop `cb_index in 0..dep_q`: feed the *previous* token (text token for cb 0, else previous audio code) through `forward_depformer(cb, [prev], transformer_out)` (`lm.py:444`) → `linears[cb]` logits → `sample_token(…temp, top_k)` → `next_token`; chain `prev = next` so each codebook is conditioned on the freshly sampled earlier one. Returns `[B, dep_q]` audio frame.
- Commit text + audio tokens back into the ring at `offsets % CT`, advance `offsets`. Until `offset_cpu > max_delay` returns `None` (warmup); thereafter gathers the de-delayed output frame `[B, num_codebooks, 1]` (`lm.py:771`–777), masking still-ungenerated lanes.

### Attention / RoPE / activations
All in `moshi_transformer` (`StreamingTransformer`): RoPE `max_period=10000`, `F.scaled_dot_product_attention` (eager/SDPA, CUDA-gated `torch.compile`), gated FFN (`moshi_gating`, silu/gelu). `LMModel` itself contains no attention/RoPE math — it is the muxing + depformer-orchestration + sampling layer.

### Sampling
`sample_token` (`moshi_util_sampling`): if `use_sampling` and `temp>0`, temperature scale → top-k (250 audio / 25 text) → top-p → `torch.multinomial`; else argmax. Text and audio use independent `(temp, top_k)`.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `forward` (train) `codes` | int64 `[B, n_q+1, T]` | `LMOutput.logits` (model dtype, NaN-filled) `[B, dep_q, T, card]` + `text_logits` `[B,1,T,text_card]` + bool masks |
| `text_emb`/`emb[k]` lookup | int64 ids (with `-1` zero rows) | model dtype (bf16/f32) `[B, S, dim]` |
| main `transformer` | model dtype `[B, S, dim]` | model dtype `[B, S, dim]` |
| `text_linear` | model dtype `[B,S,dim]` | model dtype `[B,S,text_card]` → sampled in **f32** (`.float()` `lm.py:731`) |
| `depformer_in[i]` + token emb | model dtype `[B*T, Ka, dim]→[…,depformer_dim]` | model dtype `[B*T, Ka, depformer_dim]` |
| `linears[k]` | model dtype `[·, depformer_dim]` | model dtype `[·, card]` → sampled in **f32** |
| `LMGen.step` `input_tokens` | int64 `[B, K_in, 1]` | int64 frame `[B, n_q+1, 1]` (codes 0..card, plus SOS/pad specials) or `None` during warmup |

Internal promotions: weights bf16 on CUDA (Python default), softmax/`sample_token` upcast to **f32** explicitly; token ids/cache/offsets are **int64**; condition tensors kept **f32** until cast to model dtype at fuse (`lm.py:618`). Special ids: `zero_token_id=-1` (no-op/skip), `ungenerated_token_id=-2` (cache sentinel), `initial_token_id=card`, `text_initial_token_id=text_card`, `existing_text_padding_id=3`.

## Wiring
**Upstream (feeds this — off LFM2-Audio path):**
- Mimi codes int `[B, n_q, T]` from the codec ← [MimiModel (compression.py)](compression.md) (encode side, training/`run_inference`).
- Conditioning tensors f32 ← `ConditionProvider`/`ConditionFuser` (`moshi_cond_base`, summed/cross-attn), gated by `fuser`.
- Sub-components it imports: `ScaledEmbedding` + `_delay_sequence`/`_undelay_sequence`/`_init_layer` ← [lm_utils.py](lm_utils.md); `sample_token` ← `moshi/utils/sampling.py`; `StreamingTransformer`/`create_norm_fn` ← `moshi/modules/transformer.py`; `CUDAGraphed` ← `moshi/utils/compile.py`.

**Downstream (consumes this output — off LFM2-Audio path):**
- `LMGen.step` audio frame int `[B,8,1]` → [MimiModel (compression.py)](compression.md)`.decode` → waveform f32 @24kHz, inside `moshi/server.py` / `moshi/run_inference.py`.
- `LMOutput` (train) → the Moshi trainer's cross-entropy (not LFM2-Audio's `core_trainer`).
- `TTSModel` ([tts.py](tts.md)) wraps `LMModel`+`LMGen` for script-driven TTS (off-path).

LFM2-Audio's actual audio LM head is **not** this file — it is [lfm2_audio.py](../../model/lfm2_audio.md)'s depthformer over [lfm2_backbone.md](../../model/lfm2_backbone.md); this `LMModel` has no consumer on the LFM2-Audio tensor path.

## Python ↔ Rust
The Rust counterpart is the published **`moshi-0.6.4` crate** `src/lm.rs` (pulled into `liquid-audio-rs` only for Mimi; its `LmModel` is **never constructed** by the LFM2-Audio port — confirmed: `liquid-audio-rs` uses `moshi::mimi` only, per ARCHAEOLOGY §Q1). Symbol map:

| Python (`lm.py`) | Rust (`moshi-0.6.4/src/lm.rs`) | Notes |
|---|---|---|
| `LMModel.__init__` / config kwargs | `LmModel::new` / `new_` (`lm.rs:708`,716) + `Config`/`DepFormerConfig` (`lm.rs:37`,24) | config as typed struct vs kwargs |
| `forward_text` | `LmModel::forward`/`forward_ca` (`lm.rs:672`,914) | streaming-only in Rust (no training `forward`) |
| `forward_depformer` / `depformer_step` | `DepFormer::sample` / `LmModel::depformer_sample` (`lm.rs:536`,969); `DepFormerSlice` (`lm.rs:475`) | per-slice linear_in + emb |
| CFG path (`cfg_coef`) | `DepFormer::sample_cfg` / `depformer_sample_cfg` (`lm.rs:583`,986) | same `null + (l-null)*coef` |
| `ScaledEmbedding` low-rank | `LowRankEmbeddings` (`lm.rs:437`) | |
| `LMGen` ring/delay state machine | `lm_generate.rs` / `lm_generate_multistream.rs` + `LmModel::reset_state` (`lm.rs:798`) | the `_LMGenState` cache lives in the generate module |
| `sample_token` | `candle_transformers` `LogitsProcessor` / crate sampler | |
| `StreamingTransformer` | `moshi::transformer` | RoPE/SDPA there |

**Deliberate divergences** (general port choices from PYTHON_VS_RUST.md, applicable because Rust reuses this crate rather than re-implementing): device-agnostic (`device: &Device` everywhere vs Python `device="cuda"` default), **eager/SDPA attention vs flash-attn** (`§2.2`), **no CUDA-graph layer** (`CUDAGraphed` → plain candle ops, numerically irrelevant), **upstream crate reuse** instead of re-port (`§2.3` — Mimi via `moshi::mimi`; this `lm.rs` ships in the same crate but is dead code for LFM2-Audio), and **stochastic-sampling RNG** differs (`§2.8`, greedy is identical). None of these alter the (off-path) numerics.

## Precision / gotchas
- **Off-path by design.** This is the Moshi LM, not LFM2-Audio's LM. Don't conflate its `delays = [0,0,1,1,…]` acoustic-delay pattern (`moshi_loaders` `_lm_kwargs`) with LFM2-Audio's interleave cadence (6 text / 12 audio frames, its own depthformer) — ARCH_1 §2 flags exactly this conflation.
- **Special-token sentinels.** `zero_token_id=-1` and `ungenerated_token_id=-2` are *negative* and must never reach an embedding row directly; `ScaledEmbedding(zero_idx=-1)` zeroes the `-1` row, and `_step` substitutes `initial` for warmup/ungenerated positions before lookup (`lm.py:696`). `scatter_with_mask_` (`lm.py:31`) is the off-by-one-safe write that preserves un-executed lanes (`exec_mask`).
- **f32 sampling floor.** Logits are explicitly `.float()` before `sample_token` (`lm.py:731`,831) — sampling is done in f32 even when the backbone runs bf16, so the categorical draw is precision-stable.
- **EOAudio note is LFM2's, not Moshi's.** The `2048 = EOAudio` / `card+1` audio-padding semantics in the global dtype facts are the LFM2-Audio depthformer's; here the depformer **cannot emit the audio pad token** (Rust comment `lm.rs:521`–524 caps audio logits at `audio_vocab_size-1`), a different special-token regime.
- **RMSNorm order lives elsewhere.** The bf16 `(_norm(x.float())*w).type_as(x)` vs cast-then-multiply distinction (PYTHON_VS_RUST §2.4) applies to `moshi_transformer`/`model_transformer`, not to any op defined in this file — `LMModel` does no norm math itself.
- **Warmup returns `None`.** `LMGen.step` yields `None` for the first `max_delay` frames; callers (`moshi_server`/`moshi_run_inference`) must tolerate it before audio appears.
