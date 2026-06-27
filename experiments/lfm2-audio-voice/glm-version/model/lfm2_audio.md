# model_lfm2_audio (Rust port)
**Source:** `liquid-audio-rs/src/model/lfm2_audio.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/lfm2_audio.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`ARCH/model/lfm2_audio.md`](../../ARCH/model/lfm2_audio.md). The
> original documents the Python `LFM2AudioModel`; this documents the Rust port
> and where it deliberately diverges.

## Role
`LFM2AudioModel` (the `struct` in `lfm2_audio.rs:236`) is the top-level
orchestrator of LFM2.5-Audio in the Rust port: it owns every sub-network (the
`lfm2_hf::Model` hybrid backbone, the FastConformer encoder + adapter MLP, the
audio-token `SharedEmbedding`, and a depthformer `RawLmBackbone`) and wires them
into one autoregressive loop. It assembles a single mixed-modality embedding
sequence (text tokens, encoded audio-in, embedded audio-out codes) in
`prefill_inputs`, runs it through the backbone, and then alternately emits text
tokens (via the tied LM head) and 8-codebook audio frames (via the depthformer
head). It is the only module that knows the turn structure
(`generate_interleaved` / `generate_sequential`) and the special-token control
flow (`<|audio_start|>`=128, `<|text_end|>`=130, `<|im_end|>`=7,
EOAudio=2048). Training loss (`logits` / `forward`) also lives here but is off
the inference path.

## How it works (Rust)

### Construction (`LFM2AudioModel::new`, `lfm2_audio.rs:263`)
- `lfm: Lfm2Model::new(&lfm_cfg, vb.pp("lfm"))` — the Rust `lfm2_hf::Model`
  backbone (`hidden_size`=2048; see
  [`glm-version/model/lfm2_backbone.md`](lfm2_backbone.md)). Holds `embed_tokens`
  (the tied text embedding/LM-head weight, vocab 65536) and the per-layer
  short-conv/GQA stack.
- `conformer: ConformerEncoder::new(enc_cfg, vb.pp("conformer"))` — the
  FastConformer encoder.
- `feat_out` selection (`:276`): `if enc_cfg.feat_out > 0 && enc_cfg.feat_out !=
  enc_cfg.d_model { enc_cfg.feat_out } else { enc_cfg.d_model }` — the
  Python `_feat_out` resolution.
- `audio_adapter: MLP::new(feat_out, hidden, &[hidden], true, true, 0.0,
  vb.pp("audio_adapter"))` — the `model_mlp` GELU-erf adapter.
- `audio_embedding: SharedEmbedding::new(hidden, AUDIO_VOCAB_SIZE * codebooks,
  1e-5, vb.pp("audio_embedding"))` — one flat embedding table holding **all 8
  codebooks concatenated**; `AUDIO_VOCAB_SIZE = 2048 + 1` (the +1 is EOAudio).
- `codebook_offsets: Vec<i64>` (`:298`): `(0..codebooks).map(|i| i *
  AUDIO_VOCAB_SIZE as i64)` — `[0, 2049, 4098, …]`, used to map per-codebook code
  `c∈[0,2049)` into the flat table row `c + offset[k]`. A `Vec<i64>` rather than
  a registered buffer (Python's `register_buffer`); the offsets are a constant
  held on the model.
- `audio_loss_weights: Tensor` (`:310-326`): the `log`/`linear` per-codebook loss
  schedule, built from `LossConf` at construction. Construction-only; never
  affects any generation path. `LossConf::default()` mirrors the Python defaults
  (`codebook_weight="linear"`, `semantic_codebook_factor=1.0`,
  `text_loss_multiplier=1.0`, `audio_loss_multiplier=1.0`).
- Depthformer (`:281-289`): a `Vec<StandardBlock>` of `Mha::new(depth_cfg.dim,
  32, HeadStyle::Gqa, true, 1e-5, 8, 128_000, 1_000_000.0, ...)` blocks;
  `RawLmBackbone::new(layers, None, depth_cfg.dim)` (no embedding —
  `has_embedding=False`).
- `depth_linear: Linear::new(hidden, depth_cfg.dim * codebooks,
  vb.pp("depth_linear"))` — projects one backbone hidden into 8 per-codebook
  depthformer inputs.
- `depth_embeddings: Vec<SharedEmbedding>` (`:293-296`): one
  `SharedEmbedding::new(depth_cfg.dim, AUDIO_VOCAB_SIZE, 1e-5, ...)` per
  codebook, each with its own `embedding_norm` (RMSNorm) + `to_logits`.

### Prefill / modality scatter (`prefill_inputs`, `lfm2_audio.rs:641`)
This is the heart of the multimodal input assembly. Inputs: `text (B,L_t)`
int, `audio_in (128, ΣT_mel)` mel, `audio_in_lens (n_clips,)`,
`audio_out (≥C, L_ao)` int, `modality_flag (B,L)` with values from `LFMModality`.
Steps:
1. **Modality read** (`:658-660`): the full `(B, L)` flag is read as i64 and
   flattened — Python uses 2-D boolean masks over the whole batch; for
   inference (B=1) this is identical to row 0.
2. **Text** (`:663`): `text_emb = self.lfm.embed(text)?.i(0)?` → `(n_text, D)`.
3. **Audio-in** (`:672-692`): split the concatenated mel along time by
   `audio_in_lens` (read as i64 — see §Gotchas for the dtype bug that was
   fixed here), cast each segment to the model dtype **before** the conformer
   (matching Python's `mel.to(text_emb.dtype)`), run `conformer.forward(&seg)`
   → `(1, d, T')`, transpose to `(T', d)`, run `audio_adapter.forward(&enc)` →
   `(T', hidden)`, concatenate all clips' rows. Empty-lens case → `None`.
4. **Audio-out** (`:695-706`): `codes = audio_out.narrow(0, 0, codebooks)?`,
   `offset_codes = codes.broadcast_add(&offs)?` (the per-codebook offset),
   `emb = self.audio_embedding.embed(&offset_codes)?` → `(codebooks, m, D)`,
   `emb.sum(0)?` → `(m, D)`. The sum-over-codebooks is the audio-frame
   embedding. Empty case → `None`.
5. **Scatter** (`:708-757`): concatenate `text_emb`, `audio_in_emb`, `audio_out_emb`
   into `combined (n_total, D)`, then build a per-position `index: Vec<u32>` by
   walking the modality flag — `TEXT` → `text_base + ct`, `AUDIO_IN` → `ai_base +
   cai`, `AUDIO_OUT` → `ao_base + cao`. An **unknown modality flag errors**
   (`:741`) instead of silently bucketing as AudioOut (the Python asserts the
   flag is one of the 3). A count-mismatch errors (`:747`) mirroring the Python
   asserts. Finally `in_emb = combined.index_select(&index, 0)?.reshape((B, L,
   D))`.

This is a **deliberate substitution** for Python's boolean-mask scatter
(`in_emb[modality==TEXT] = text_emb`): Rust uses `index_select` with a
per-position index. Same result; the `index_select` form avoids the
boolean-assignment idiom candle doesn't have (PYTHON_VS_RUST.md §1.3).

### Generation — interleaved (`generate_interleaved` / `generate_from_embeds`,
`lfm2_audio.rs:853` / `:862`)
Decode is a **synchronous callback stream** `FnMut(GenToken)` — faithful to the
Python `@torch.no_grad()` generator (sync streaming; async lives only at the
transport, per the design). After `prefill_inputs`, it loops up to
`max_new_tokens`, maintaining `current: LFMModality`, `modality_left: i64` (a
countdown), `text_done: bool`, and an `LfmCache`. Each step:
- `lfm.forward_embeds(&in_emb, index_pos, &mut cache, None)` →
  `h (1, seq, D)`; take the **last** position `h.i((0, seq_len - 1))?` (`:877`).
- **TEXT mode** (`:880`): `text_logits(&h_last)?` (f32-upcast matmul against
  `lfm.embed_weight()`), `sample_text_token`, break on `<|im_end|>`(7),
  set `text_done` on `<|text_end|>`(130); when `modality_left <= 0` or
  `text_done`, flip to `AudioOut` with `modality_left = interleaved_n_audio`.
  Next `in_emb = self.lfm.embed(&tok)?.reshape((1, 1, hidden))`.
- **AUDIO_OUT mode** (`:897`): `sample_audio_frame(&h_last, &mut audio_sampler)?`
  → 8 codes; if `modality_left <= 0 && !text_done`, flip back to TEXT
  (`modality_left = interleaved_n_text`); if `frame[0] == 2048` (EOAudio) force
  the whole frame to 2048 and flip to TEXT; next
  `in_emb = self.audio_frame_embed(&frame)?` — exactly the prefill audio-frame
  embedding, fed back autoregressively.

`generate_sequential` (`:803`) is the same but emits **all** text first, switches
to `AudioOut` only on `<|audio_start|>`(128), and has no interleave countdown —
the ASR/TTS path.

### Audio-frame decode (`sample_audio_frame`, `lfm2_audio.rs:769`)
This is the **depthformer inner loop** — a tiny autoregressive transformer over
the 8 codebooks for one acoustic frame:
1. `emb2d = embedding.flatten_all()?.unsqueeze(0)?` (`:773`) — **candle's
   `Linear` needs a 2-D input**; Python's `nn.Linear` accepts a 1-D vector
   directly. This reshape was a caught bug (the latent 1-D `Linear` in the
   depthformer sampler, PYTHON_VS_RUST.md).
2. `din = depth_linear.forward(&emb2d)?.reshape((codebooks,
   depthformer_dim))?` (`:774`) → `(8, D)`, one input vector per codebook.
3. `df_token = zeros(D)`; `caches: Vec<LayerKvCache>` one per depthformer layer
   (`:776`); loop `i in 0..codebooks`:
   - `cur = (din.i(i)? + &df_token)?.reshape((1, 1, D))?` (`:779`).
   - `dout = depthformer.forward(&cur, Some(caches.as_mut_slice()))?` (`:780`)
     — a `q_len==1` step that grows the per-layer KV cache; the 8 codebooks are
     the "sequence".
   - `logits = depth_embeddings[i].get_logits(&dout.reshape((1, D))?)?.i(0)?`
     (`:782`) — per-codebook RMSNorm → tied/own `to_logits`, `(2049,)`.
   - `token = sampler.sample(&logits)?` (`:783`).
   - `df_token = depth_embeddings[i].embed(&Tensor::from_vec(vec![token],
     (1,), ...))?.reshape((D,))?` (`:786`) — feed it back for the next codebook.
4. Return `out: Vec<u32>` (`:788`) — `(8,)`.

So codebook `i` is conditioned on the backbone hidden + the embeddings of codes
`0..i-1` — residual-vector-quantizer-style coarse-to-fine prediction.

### Sampling (`Sampler`, `lfm2_audio.rs:174`)
Built on `candle_transformers::generation::LogitsProcessor` (the same sampler
`moshi` uses for depformer decoding), rather than a private softmax+multinomial.
Faithful to `_sample_text_token` and the per-codebook step of
`_sample_audio_frame`:
- `greedy = temperature is None || temperature <= 0 || top_k == 1` (`:183-188`).
  Greedy ⇒ `Sampling::ArgMax` — `LogitsProcessor::sample_argmax` is
  `logits.argmax(-1)`, byte-identical to the previous greedy path, so
  generation parity (incl. the token-exact depthformer) is preserved.
- Stochastic ⇒ `Sampling::All { temperature }` (temperature softmax +
  multinomial), with **Torch's threshold top-k injected through
  `LogitsProcessor::sample_f`** (`:202`): candle's built-in `Sampling::TopK`
  keeps exactly `k` tokens, whereas Torch keeps every token `≥` the k-th
  largest (ties included). `torch_topk_mask` (`:215`) zeroes every probability
  below the k-th largest; softmax is monotonic so the logit threshold and the
  probability threshold select the same tokens. The kept probabilities need no
  renormalization — `WeightedIndex` samples proportionally.
- `seed: u64` is explicit (Python relies on the global `torch` generator; Rust
  makes it reproducible). Tests: `greedy_when_no_temperature`,
  `greedy_when_temp_nonpositive_or_topk_one`, `topk_restricts_support`,
  `seed_is_reproducible`, `sampling_can_pick_nonargmax` (`:928-977`).

### Training (`logits` / `forward`, `lfm2_audio.rs:433` / `:575` — off inference path)
Teacher-forced: `prefill_inputs`, run the backbone once (`backbone_forward_embeds`,
`use_cache=False` analog), shift `out_emb[:, :-1]` (via `narrow(1, 0, ll-1)` +
reshape, `:443`), select supervised text/audio positions via a 2-D walk over the
full `(B, L)` modality/supervision masks (`:460-476`). Text logits via the tied
head (f32-upcast matmul). Audio: `depth_linear` → `(n, C, D)`, add the rolled
teacher-code embeddings (`roll(+1)` along C so codebook `i` sees code `i-1`;
the last codebook's contribution is zeroed before the roll → codebook 0 sees
zero, `:514-518`), run the depthformer **in parallel** over all C as one
causally-masked sequence, **chunked** if `n ≥ 2^14` (`:526-545` — Python's
`torch.chunk(num_chunks)`), per-codebook logits, `cross_entropy_none`
(`candle_ext::loss`, §2.6), per-codebook `audio_loss_weights` weighting.

`forward` (`:575`) computes the cross-entropy losses + the weighted total loss,
self-contained on the model (loss weights/multipliers are stored fields).
`LFM2AudioModelOutput` carries `loss`, `audio_loss`, `text_loss`, and the three
token counts (`audio_out_tokens`, `text_tokens`, `audio_in_tokens` — the last
via `mel2emb_len(batch.audio_in_lens).sum()`, `:618`).

## Dtypes & shapes (Rust)
| Stage | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `prefill_inputs` text branch | `text` int (U32 from ChatState / I64 from dataloader) `(B,L_t)` | text_emb model-dtype `(n_text, 2048)` |
| `prefill_inputs` audio-in | mel model-dtype `(128, ΣT_mel)` (cast before conformer) | audio_in_emb model-dtype `(ΣT', 2048)`, T'=⌈T_mel/8⌉ |
| `prefill_inputs` audio-out | codes int `(C, L_ao)` | audio_out_emb model-dtype `(L_ao, 2048)` (sum over 8 codebooks) |
| `prefill_inputs` assembled | `modality_flag` I64 `(B,L)` | `in_emb` model-dtype `(B, L, 2048)` (via `index_select`) |
| backbone step (`lfm.forward_embeds`) | `in_emb` model-dtype `(1, seq, 2048)` | hidden model-dtype `(1, seq, 2048)`; last → `(2048,)` |
| text head (tied, f32-upcast) | hidden `(2048,)` | text_logits **f32** `(65536,)` → sampled `u32` |
| `depth_linear` | hidden `(2048,)` (reshaped to `(1, 2048)` for candle `Linear`) | `(C·depthformer_dim,)` → `(8, depthformer_dim)` |
| depthformer step | `(1, 1, depthformer_dim)` + `Vec<LayerKvCache>` | `(1, 1, depthformer_dim)`; per-cb logits `(2049,)` |
| `sample_audio_frame` | hidden `(2048,)` | audio frame `Vec<u32>` (8 codes, 0..2048; 2048=EOAudio) |
| audio-frame feedback | frame `&[u32]` (8) | `in_emb` model-dtype `(1, 1, 2048)` |

Internal promotions: `RmsNorm` (backbone, depthformer, `embedding_norm`)
normalizes in **f32** then casts back; `text_logits` upcasts to f32 for the
matmul; `softmax` in attention/sampling in f32; RoPE built in f32 (`cos`/`sin`
f32 tables). Backbone/depthformer weights bf16 on disk; Rust CPU compute f32,
Metal bf16. Token ids: ChatState feeds `U32`, the dataloader feeds `I64`;
`prefill_inputs` casts to `I64` for the modality read and the audio-in lens.

## Wiring (Rust)
**Upstream (feeds `prefill_inputs`):**
- `processor.rs` — builds the `ChatState` (text U32 `(1, L_t)`, mel
  model-dtype `(128, ΣT_mel)`, `audio_in_lens`, `audio_out` codes,
  `modality_flag` I64 `(1, L)`) consumed by `prefill`. See
  [`glm-version/processor.md`](processor.md).
- `model/conformer/encoder.rs` — `self.conformer`; mel model-dtype
  `(B, 128, T)` → `(B, 512, T')`. See
  [`glm-version/model/conformer/encoder.md`](model/conformer/encoder.md).
- `model/mlp.rs` — `self.audio_adapter`; conformer `(ΣT', 512)` →
  `(ΣT', 2048)`. See [`glm-version/model/mlp.md`](model/mlp.md).
- `model/lfm2_hf.rs` — `self.lfm`; `in_emb (1, seq, 2048)` → hidden
  `(1, seq, 2048)` + `embed_weight()` for the tied head. See
  [`glm-version/model/lfm2_backbone.md`](model/lfm2_backbone.md).
- `model/transformer.rs` — `RawLmBackbone`/`StandardBlock`/`Mha`/
  `SharedEmbedding`; the depthformer + audio embedding tables. See
  [`glm-version/model/transformer.md`](model/transformer.md).
- `utils.rs` — `LFMModality` enum + `mel2emb_len` for length math. See
  [`glm-version/utils.md`](utils.md).

**Downstream (consumes this output):**
- `processor.rs` — `generate_interleaved`/`generate_sequential` yield
  `GenToken::Text(u32)` (detokenized to string) and `GenToken::Audio(Vec<u32>)`
  per Mimi-frame, which the processor routes to `decode()`.
- `detokenizer.rs` (LFM2 ISTFT vocoder) / `audio_out.rs::MimiDetokenizer` — the
  emitted audio frames `Vec<u32>` (8 codes) are fed (via the processor's
  dispatch through `Box<dyn AudioDetokenizer>`) to the LFM2 ISTFT detokenizer or
  Mimi `decode` → f32 waveform @ 24 kHz. See
  [`glm-version/detokenizer.md`](detokenizer.md).

## Python ↔ Rust — where the port differs

| Python symbol | Rust symbol | Difference | Why |
|---|---|---|---|
| `LFM2AudioModel.__init__` | `LFM2AudioModel::new` | **`LossConf` bundles training-only fields** | Rust has no keyword defaults; the loss config fields are grouped into one struct to keep `new`'s signature clean. `LossConf::default()` mirrors the Python defaults. |
| `from_pretrained(dir, *, device, dtype)` (defaults `cuda`/`bf16`) | `from_pretrained(dir, dtype, device)` → `crate::loader::from_pretrained` | **device/dtype-agnostic** | §2.1. No `.cuda()`; `(Cpu, F32)` for parity, Metal/bf16 opt-in. Python returns just the model; Rust returns `(model, processor)` (loaded alongside). |
| `register_buffer("codebook_offsets", arange(codebooks)*2049)` | `codebook_offsets: Vec<i64>` field | **buffer → plain Vec** | the offsets are a constant; a registered buffer would be a needless tensor. Held on the model and cloned into a `(codebooks, 1)` tensor when needed (`:701`, `:794`). |
| `register_buffer("audio_loss_weights", …)` | `audio_loss_weights: Tensor` field | identical (a real tensor) | built from `LossConf` at construction; consumed only by `forward`. |
| `_prefill` boolean-mask scatter (`in_emb[modality==TEXT] = text_emb`) | `prefill_inputs` `index_select` with a per-position `index: Vec<u32>` | **deliberate** | candle has no boolean-assignment idiom; `index_select` is the faithful equivalent. §1.3. Unknown modality flag **errors** instead of silently bucketing as AudioOut (the Python asserts the flag is one of the 3). |
| `generate_interleaved`/`generate_sequential` (Python generator, `yield`) | same names, **sync callback** `FnMut(GenToken)` | **deliberate: generator → callback** | sync streaming faithful to the Python generator (async only at the transport, per the design). `GenToken::Text(u32)` / `GenToken::Audio(Vec<u32>)` replace the yielded ints/frames. |
| `_sample_text_token` / per-codebook sampler | `Sampler` over `candle_transformers::LogitsProcessor` | **deliberate reuse** | the same sampler `moshi` uses for depformer decoding. Greedy=`Sampling::ArgMax` (byte-identical); Torch **threshold** top-k injected via `sample_f` hook (candle's `TopK` keeps exactly k; Torch keeps ties) — §2.3/§2.8. |
| `_sample_audio_frame` (1-D `nn.Linear`) | `sample_audio_frame` (reshapes to 2-D for candle `Linear`) | **deliberate: 1-D → 2-D** | candle's `Linear` needs a 2-D input; Python's `nn.Linear` accepts the 1-D vector directly. This reshape was a caught bug (the latent 1-D `Linear` in the depthformer sampler, §1.4). |
| `lfm.embed_tokens` tied head via `F.linear(h, embed_tokens.weight)` | `text_logits`: `lfm.embed_weight().to_dtype(F32)?.matmul(&h.to_dtype(F32)?)` | **f32-upcast** | the text head runs in f32 regardless of model dtype, matching torch's f32-accumulation. |
| `self.lfm` (HF `Lfm2Model`) | `crate::model::lfm2_hf::Model` | **external → in-tree port** | the Rust `lfm2_hf.rs` is the readable spec; flash/sdpa CUDA kernels → eager matmul+causal-mask+softmax (§2.2). |
| `Lfm2HybridConvCache` | `lfm2_hf::Cache` (`LfmCache`) | **deliberate** | short-conv `conv_L_cache` + GQA KV cache; candle `Conv1d`/gather (§2.2). |
| `logits`/`forward` (training) | `logits`/`forward` | present for inventory | `cross_entropy(reduction="none")` → `candle_ext::loss::cross_entropy_none` (§2.6). |
| `torch.chunk(num_chunks)` along dim 0 | `n.div_ceil(num_chunks)`-sized pieces in a `while` loop | **deliberate** | candle has no `chunk`; the loop reproduces `torch.chunk`'s ceil-sized pieces (the last may be smaller). For parity `n < 16384` ⇒ a single call identical to the unsplit forward. |
| `rearrange("(C D) -> C D")` (einops) | `reshape((codebooks, depthformer_dim))` | identical | candle's `reshape` is the direct equivalent. |
| `roll(1)` along C (zero last, then shift) | `cat([&dtok.narrow(1, 0, c-1)?, &zero_last], 1)?` then `cat([&last, &rest], 1)?` | **manual** | candle has no `roll`; the two `cat`s reproduce zero-last-then-shift. |
| `torch.multinomial` RNG (global generator) | `LogitsProcessor::from_sampling(seed, …)` (seeded `StdRng`) | **deliberate: explicit seed** | Python relies on the global `torch` generator; Rust makes it explicit and reproducible (`GenParams::seed`). Stochastic sampling is not byte-reproducible cross-framework (§2.8) but the seed makes it reproducible *within* the Rust port. |
| `@torch.no_grad()` | (no equivalent needed) | — | candle's inference path has no autograd by default; the `no_grad` context manager is implicit. |

**Parity:** backbone hidden 6.558e-6, text logits 5.505e-6, **depthformer audio
frame token-EXACT** `[213,836,182,416,782,1796,202,578]` (PARITY.md), prefill
modality-scatter 1.118e-6.

## Precision / gotchas (Rust-specific)
- **`text_logits` f32-upcast.** `lfm.embed_weight().to_dtype(DType::F32)?` and
  `h_last.to_dtype(DType::F32)?` before the matmul (`:761-763`). The text head
  runs in f32 regardless of model dtype, matching torch's f32-accumulation. A
  bf16-matmul-then-upcast would diverge at the ~1e-3 level, not the floor.
- **`audio_in_lens` dtype read.** `audio_in_lens.to_dtype(DType::I64)?.
  to_vec1::<i64>()?` (`:672`) — the comment at `:651-655` documents a fixed bug:
  reading an I64 tensor as `u32` would silently return an empty `lens`
  (`unwrap_or_default`) and drop all audio-in. The cast-to-I64-then-read is
  load-bearing; the dataloader feeds I64, ChatState feeds U32, and the cast
  handles both. The old `unwrap_or_default()` swallowed malformed-lens errors
  into an empty `lens`, which would scatter zero audio-in and then trip the
  count check with a confusing message instead of the real cause.
- **EOAudio handling.** Code value `2048` (= `AUDIO_VOCAB_SIZE - 1`) is EOAudio.
  Generation checks **only codebook 0** (`frame[0] == 2048`); on hit it forces
  the *entire* 8-code frame to 2048 (`for c in frame.iter_mut() { *c = 2048; }`,
  `:837-839`/`:904-906`) and returns to TEXT. The flat `audio_embedding` table
  is sized `2049 * codebooks` precisely to host this per-codebook EOAudio row.
- **`AUDIO_VOCAB_SIZE` is `2048 + 1`.** The `+1` is EOAudio; easy to drop and
  silently truncate the vocab.
- **`Sampler::new` greedy detection.** `greedy = temperature is None ||
  temperature <= 0 || top_k == 1` (`:183-188`). The `top_k == 1` case is
  load-bearing: a stochastic sampler with `top_k == 1` is *not* stochastic, it's
  argmax. The test `greedy_when_temp_nonpositive_or_topk_one` pins this.
- **`torch_topk_mask` operates on probabilities, not logits.** The
  `sample_f` hook receives the post-softmax probability vector, not the raw
  logits. The mask zeroes every probability below the k-th largest. Softmax is
  monotonic, so the logit threshold and the probability threshold select the
  same tokens; ties at the boundary are kept (matching Torch's
  `logits[logits < min_score] = -inf`). The kept probabilities need no
  renormalization — `WeightedIndex` samples proportionally. The comment at
  `:210-214` records this.
- **`candle` `Linear` needs 2-D input.** `sample_audio_frame` reshapes the
  1-D `embedding (D,)` to `(1, D)` before `depth_linear.forward` (`:773`).
  Python's `nn.Linear` accepts the 1-D vector directly. This was a caught bug.
- **`index_select` scatter, not boolean assignment.** `prefill_inputs` builds
  a per-position `index: Vec<u32>` and uses `combined.index_select(&index, 0)?`
  (`:754`). An unknown modality flag errors (`:741`); a count-mismatch errors
  (`:747`). The Python asserts; the Rust port errors loudly.
- **`chunk` via `while` loop.** The depthformer training split (`:526-545`)
  reproduces `torch.chunk(num_chunks)` with `n.div_ceil(num_chunks)`-sized pieces
  in a `while` loop, concatenating back along dim 0. For parity `n < 16384` ⇒
  a single call. The `should_split` gate (`n >= 16384`, `2**14`) matches
  Python's threshold.
- **`roll(+1)` along C is manual.** Zero the last codebook's contribution
  (`cat([&dtok.narrow(1, 0, c-1)?, &zero_last], 1)?`, `:516`), then shift
  (`cat([&last, &rest], 1)?`, `:517`). Two `cat`s reproduce Python's
  `roll(1)`. Easy to get backwards — the test suite + parity harness guard it.
- **`GenParams::seed` is explicit.** Python relies on the global `torch`
  generator; Rust requires a seed (default `42`). Stochastic sampling is not
  byte-reproducible vs torch (different RNG stream), but is reproducible
  *within* the Rust port — `seed_is_reproducible` test pins this.
- **`cross_entropy_none` is vendored.** `crate::candle_ext::loss::
  cross_entropy_none` is the `reduction="none"` path candle's mean-only
  `cross_entropy` lacks (§2.6). Reused rather than re-derived.
- **Tied vs untied heads.** The text head reuses `lfm.embed_weight()` directly
  via `matmul` (no separate projection); the depthformer per-codebook heads use
  `SharedEmbedding::get_logits` = `to_logits(embedding_norm(x))`, tied to the
  codebook's embedding when `depthformer_tie`. Don't confuse the two: text
  logits skip an `embedding_norm`, audio logits don't.
- **Special tokens are literal ints.** `7`=`<|im_end|>`,
  `128`=`<|audio_start|>`, `130`=`<|text_end|>` are hard-coded (not config) and
  drive the modality state machine.
- **Audio-frame embedding is a sum, not a concat.** Both prefill and feedback
  embed the 8 codes through one shared `2049 * codebooks`-row table (offset per
  codebook) and **sum** them into a single 2048-vector — losing per-codebook
  identity by design; the depthformer recovers ordering at decode via its
  causal C-sequence.
- **`mel2emb_len = -(l // -8)`** is ceil-division (`⌈l/8⌉`), the 8× conformer
  subsample; smallest valid mel length is 9. Used in the prefill count check
  and `forward`'s `audio_in_tokens` (`:618`). An off-by-one here desyncs the
  modality scatter.
- **Cross-library f32 floor.** Backbone hidden parity 6.56e-6, text logits
  5.51e-6, depthformer audio frame **token-exact** (no float reduction in
  argmax/gather) — the ~1e-6 residual is gemm-order/transcendental/FFT
  rounding, irreducible without re-implementing candle's kernels (§1.4).
- **Stochastic RNG diverges.** Greedy decoding is deterministic and matches
  Python token-for-token; `temperature > 0` uses `LogitsProcessor`'s RNG
  (rand_pcg, seeded) — a different stream from `torch.multinomial` — never
  byte-reproducible cross-framework, but the token set + threshold-top-k
  distribution match (§2.8).

## Cross-references
- [`ARCH/model/lfm2_audio.md`](../../ARCH/model/lfm2_audio.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §1.3 (`index_select` scatter), §2.1
  (device-agnostic), §2.2 (CUDA kernels → portable candle ops), §2.3
  (`LogitsProcessor` reuse + threshold top-k), §2.6 (`cross_entropy_none`),
  §2.8 (stochastic RNG).
- `liquid-audio-rs/parity/PARITY.md` — prefill modality-scatter 1.1e-6,
  depthformer audio frame token-EXACT.
- `liquid-audio-rs/src/loader.rs` — `from_pretrained` / `from_pretrained_hub`.