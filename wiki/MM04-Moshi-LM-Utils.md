<!-- topic: Moshi LM (off-path) -->
# MM04 · Moshi LM utils
**Code:** `MM04` · **Source:** `moshi/models/lm_utils.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
`lm_utils.py` is a grab-bag of helpers for the **Moshi 7B multi-stream LM** (`moshi/models/lm.py`) and the Moshi `TTSModel` (`moshi/models/tts.py`): a custom `ScaledEmbedding` table (zero-token masking, optional post-embedding LayerNorm, optional low-rank factorization, optional cartesian "second-stream demux"), the acoustic-codebook **delay / undelay** sequence shifters, and truncated-normal weight-init helpers. It is **vendored Kyutai code** that LFM2.5-Audio carries because it imports the whole `moshi` package for the Mimi codec — but LFM2-Audio's own backbone, depthformer, and embedding tables (`model/lfm2_audio.py`, `model/transformer.py`) **never import or call anything in this file**. It exists only to make the Moshi LM (`moshi_lm`, reference-only) constructible.

## How it works

**`ScaledEmbedding(nn.Embedding)` forward** (`lm_utils.py:102-124`) — a vocab lookup with three orthogonal twists, gated by constructor flags:

- *zero-idx masking* (always active). `zero_idx` must be negative and is asserted `== -1` in the demux branch (`lm_utils.py:88,96`). Forward records `is_zero = input == zero_idx`, then `input.clamp(min=0)` so the lookup never indexes negative (`:103-105`). After the lookup, `torch.where(is_zero[...,None], zero, y)` forces those positions to an **exact 0 vector** (`:116,121`). `zero` is a 1-element tensor in the input's dtype/device (`:104`) — i.e. the "no token here" sentinel that distinguishes a true codebook value 0 from an empty slot.
- *post-embedding norm* (`norm=True`). Builds `create_norm_fn("layer_norm", embedding_dim)` (`:87`) = `nn.LayerNorm(dim, eps=1e-5)` (transformer.py:116-117), applied to the looked-up rows **before** the zero-masking `where` (`:119-121`). Standard LayerNorm: mean/var over the last dim, `(x-μ)/sqrt(σ²+1e-5)·γ+β`.
- *low-rank factorization* (`low_rank=int`). The underlying `nn.Embedding` is created with width `low_rank` instead of `embedding_dim` (`:84`), and a bias-free `nn.Linear(low_rank, embedding_dim)` projects up **after** masking (`:92-93, 122-123`). Cuts parameter count for very large vocabs (used by the Moshi depformer's `depformer_low_rank_embeddings`, lm.py:185).
- *second-stream demux* (`demux_second_stream=True`, `:106-116`). The input id encodes **two** tokens as `tok2 * num_embeddings + tok1`. `left = input % num_embeddings`, `right = input // num_embeddings - 1` (the `-1` makes `right == -1` the zero/empty value, `:107-109`). Both halves go through the **same** embedding table, then through **separate** bias-free linears `out1`/`out2`; the result is `out1(left) + where(right<0, 0, out2(right))` (`:115`). Asserts mutually exclusive with `norm` (`:98`). This is Moshi's trick for embedding a "delayed text + audio" cartesian-product stream with one shared table.

**Delay / undelay** — the Moshi acoustic-codebook time-shift used to stagger codebooks so a causal LM can predict semantic-before-acoustic within a frame:

- `_delay_sequence(delays, tensor, padding)` (`:9-20`): input `(B,K,T)`; for each codebook `k`, `tensor[:,k].roll(delay, dims=1)` shifts right in time, and the first `delay` positions are overwritten with `padding[:,k]` (`:16-18`). `assert len(delays)==K`. Stacks back to `(B,K,T)`.
- `_undelay_sequence(delays, tensor, fill_value=NaN)` (`:23-38`): the inverse — `roll(-delay)` plus a validity **mask** `(B,K,T) bool` set 0 on the last `delay` positions (now invalid), and those positions filled with `fill_value` (`:33-37`). Short-circuits to `(tensor, all-ones mask)` when every delay is 0 (`:29-30`). Returns `(undelayed, mask)`. Moshi's `LMGen` calls these to align logits/codes per stream (lm.py:344,365-369). **NB:** these delays (`[0,0,1,1,...]`) are the Moshi 7B pattern from `loaders.py:110`, called out in `ARCH_1_MIMI_CODEC.md:84-85` as belonging to `get_moshi_lm`, **not** LFM2-Audio (whose 8 audio codebooks are emitted in lockstep by the depthformer with no inter-codebook delay).

**Init helpers** (`:41-63`): `_get_init_fn(input_dim)` returns a truncated-normal initializer with `std = 1/sqrt(input_dim)`, truncation `[-3σ, 3σ]` (`:43,48`). To dodge the fact that `trunc_normal_` is unimplemented for half/bf16 on CPU, it upcasts to f32, samples, and copies back (`:45-50`). `_init_layer(m, zero_bias_init=True)` (`:54-63`) applies it: `nn.Linear` → init weight by `in_features`, zero the bias; `nn.Embedding` → init weight by `embedding_dim`. Pure training-time weight initialization — irrelevant at inference (weights are loaded from checkpoint).

There is **no forward pass, no attention, no RoPE, no convolution, no sampling** in this file — it is embedding/init plumbing for a model LFM2-Audio does not run.

## Dtypes & shapes

| Symbol | Input(s) | Output(s) | Internal notes |
|---|---|---|---|
| `ScaledEmbedding.forward` (plain) | `input` int64 `(...,)` | model-dtype `(...,D)` (bf16/f32) | `is_zero` mask; clamp≥0; `where` forces 0-rows |
| `ScaledEmbedding.forward` (+norm) | int64 `(...,)` | model-dtype `(...,D)` | LayerNorm over D in **f32 internally** (`nn.LayerNorm` upcasts mean/var), eps 1e-5, cast back |
| `ScaledEmbedding.forward` (+low_rank=r) | int64 `(...,)` | model-dtype `(...,D)` | table width r; up-proj Linear `(r→D)` |
| `ScaledEmbedding.forward` (+demux) | int64 `(...,)`, id = `tok2·card+tok1` | model-dtype `(...,D)` | `left=id%N`, `right=id//N-1`; two Linears `(low_rank or D)→D` |
| `_delay_sequence` | `tensor (B,K,T)` (int codes or model-dtype emb), `padding (B,K,…)` | same dtype `(B,K,T)` | per-k `roll(+delay)`, prefix overwrite |
| `_undelay_sequence` | `tensor (B,K,T,*)` | `(undelayed (B,K,T,*), mask (B,K,T) bool)` | per-k `roll(-delay)`, suffix `fill_value` (NaN default) |
| `_get_init_fn` / `_init_layer` | `nn.Module` (train init) | in-place weight write | f32 upcast for trunc-normal on CPU half/bf16 |

`D` = Moshi LM `dim` (e.g. 4096 for the 7B), not LFM2's 2048. The zero-row sentinel and the demux `right=-1` empty value are exact integer/where ops (no float reduction).

## Wiring

This component is **not on the LFM2-Audio tensor path**; its only consumers are other Moshi-reference components.

- **Upstream (constructs/feeds it):**
  - [moshi_lm](MM03-Moshi-LM) builds `ScaledEmbedding` tables for `self.emb` (per audio codebook), `self.text_emb`, and the depformer `depformer_emb`/`depformer_text_emb` via an `EmbeddingFactory = partial(ScaledEmbedding, …)` (lm.py:127-138,185-196), feeding int64 token ids `(B,K,T)`. It calls `_delay_sequence`/`_undelay_sequence` on int codes `(B,K,T)` and on text logits in `LMGen` (lm.py:344,365-369), and `_init_layer` over every embedding/linear at construction (lm.py:500-513).
  - [moshi_tts](MM05-Moshi-TTS) imports `ScaledEmbedding` only to **zero specific codebook embedding rows** (`tts_model.lm.emb[q].weight.data[:] = 0`, tts.py:427-429).
  - `create_norm_fn` (the LayerNorm factory) comes from [moshi_transformer](MO03-Codec-Transformer).
- **Downstream (consumes its output):** within [moshi_lm](MM03-Moshi-LM) the `(B,K,T,D)` embeddings flow into the Moshi `StreamingTransformer` backbone/depformer; the delay-aligned logits/codes flow to the Moshi sampler ([moshi_util_sampling](MU01-Sampling)). **None of this reaches** [model_lfm2_audio](MD01-LFM2AudioModel) or [core_processor](CO01-Processor-ChatState) — LFM2-Audio uses its own `SharedEmbedding`/`FusedEmbedding` (model/transformer.py, detokenizer.py) instead.

## Python ↔ Rust

**No Rust counterpart, by design.** Per `PORT_STATUS.md:68` and `PYTHON_VS_RUST.md §4`, the entire vendored `liquid_audio/moshi/**` is **reused as the Kyutai `moshi` crate** (its own faithful Rust port), not re-ported into `liquid-audio-rs/src/`; `compare_symbols.py --scope core` excludes it (170/170 covers only the LFM2-specific surface). So `ScaledEmbedding`, `_delay_sequence`, `_undelay_sequence`, `_init_layer`, `_get_init_fn` have **no `src/` symbol** — their behavior lives inside the `moshi` crate's `lm.rs`/`lm_utils.rs`, pulled in only for the Mimi codec.

Deliberate divergence (`PYTHON_VS_RUST.md §1.3, §2.3`): the LFM2-Audio Rust port reimplements its **own** embedding tables (`candle_nn::Embedding`, `SharedEmbedding`) rather than routing through Moshi's `ScaledEmbedding`, because LFM2-Audio's backbone is HF `Lfm2Model` + a custom depthformer, not the Moshi 7B LM. The Moshi delay/undelay machinery has no LFM2 call site because LFM2's 8 audio codebooks carry **no inter-codebook delay** (the depthformer predicts a full frame in one autoregressive sweep) — confirmed in `ARCH_1_MIMI_CODEC.md:84-85`.

## Precision / gotchas

- **`zero_idx` must be `-1`.** Asserted at construction (`:88` requires negative; `:96` requires exactly `-1` for demux) so `right = id//N - 1` correctly maps the empty second-stream value to `-1` before clamping. Getting this wrong silently shifts every second-stream token by one row.
- **Exact-zero rows, not learned padding.** The zero-token path emits a literal zero vector via `where` (`:116,121`), placed **after** any LayerNorm/low-rank, so a true codebook index `0` (a real token) is not confused with an empty slot. This is an integer/select op — no float drift.
- **LayerNorm here, not RMSNorm.** `ScaledEmbedding`'s optional norm is `nn.LayerNorm(eps=1e-5)` (`:87` → transformer.py:117), distinct from the `RMSNorm`/`rms_norm_f32` (eps 1e-8) used inside the Moshi transformer blocks. Don't conflate the embedding-norm with the block-norm.
- **trunc-normal CPU half/bf16 upcast** (`:45-50`) — training-only; harmless at inference but a real correctness guard (PyTorch's `trunc_normal_` errors on CPU bf16/f16, so init upcasts to f32 and copies back).
- **`_undelay_sequence` fills with NaN by default** (`:24,35`); downstream Moshi code relies on the returned bool **mask**, not on the NaN values, to discard invalid (delay-truncated) positions. Reading the filled tensor without honoring the mask propagates NaNs.
- **Off-path caveat.** Any "bug" or behavioral nuance here affects only the Moshi 7B reference LM (`moshi_lm`), never LFM2.5-Audio inference — do not treat divergences in this file as LFM2-Audio defects.
