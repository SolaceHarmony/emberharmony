<!-- topic: Model -->
# MD01 · LFM2AudioModel (prefill + generate)
**Code:** `MD01` · **Source:** `model/lfm2_audio.py` · **Rust:** `model/lfm2_audio.rs / LFM2AudioModel` · **On the LFM2-Audio inference path:** yes

## Role
`LFM2AudioModel` is the top-level orchestrator of LFM2.5-Audio: it owns every sub-network (the HF `Lfm2Model` hybrid backbone, the FastConformer audio encoder + adapter MLP, the audio-token `SharedEmbedding`, and a small depthformer) and wires them into one autoregressive loop. It assembles a single mixed-modality embedding sequence (text tokens, encoded audio-in, embedded audio-out codes) in `_prefill`, runs it through the LFM2 backbone, and then alternately emits text tokens (via the tied LM head) and 8-codebook audio frames (via the depthformer head). It is the only module that knows the turn structure (`generate_interleaved` / `generate_sequential`) and the special-token control flow (`<|audio_start|>`=128, `<|text_end|>`=130, `<|im_end|>`=7, EOAudio=2048). Training loss (`forward`/`logits`) also lives here but is off the inference path.

## How it works

### Construction (`__init__`, py:73-133)
- `self.lfm = Lfm2Model(conf.lfm)` — the HF transformers hybrid backbone (`hidden_size`=2048, the `model_lfm2_backbone` component). Holds `embed_tokens` (the tied text embedding/LM-head weight, vocab 65536) and the per-layer short-conv/GQA stack.
- `self.conformer` = FastConformer encoder (`conformer_encoder`); `self.audio_adapter = MLP(conformer._feat_out=512 → 2048, hidden=[2048])` (`model_mlp`, GELU-erf).
- `self.audio_embedding = SharedEmbedding(dim=2048, vocab_size=audio_vocab_size·codebooks = 2049·8 = 16392, norm_eps=1e-5, tie=conf.tie_audio_embeddings)` — one flat embedding table holding **all 8 codebooks concatenated**; `audio_vocab_size = 2048+1` (the +1 is EOAudio).
- `register_buffer("codebook_offsets", arange(codebooks)*2049)` — `[0, 2049, 4098, …]`, used to map per-codebook code `c∈[0,2049)` into the flat table row `c + offset[k]` (py:102).
- `audio_loss_weights` buffer (training only): `log` schedule = `exp(linspace(1,0,C)·log(semantic_codebook_factor))`, else `ones` with `w[0]*=factor` (py:104-113).
- Depthformer: `RawLMBackbone([StandardBlock(MHA(dim))·layers], has_embedding=False)` with `out_init_scale = 1/sqrt(2·layers)` (py:115-121). `MHA` here is the `model_transformer` GQA attention (32 heads, gqa_dim=8 kv-heads, qk-RMSNorm, interleaved RoPE θ=1e6, max_seq_len=128000; head_dim=dim/32). `self.depth_linear = Linear(2048 → depthformer_dim·codebooks)` projects one backbone hidden into 8 per-codebook depthformer inputs (py:123). `self.depth_embeddings = ModuleList([SharedEmbedding(depthformer_dim, 2049, tie=depthformer_tie)]·8)` — one head per codebook, each with its own `embedding_norm` (RMSNorm) + `to_logits` (py:124-133).

### Prefill / modality scatter (`_prefill`, py:307-372)
This is the heart of the multimodal input assembly. Inputs: `text (1,L_t) int64`, `audio_in (128, ΣT_mel) bf16` (mel features, 128 bins, all clips concatenated along time), `audio_in_lens (n_clips,)`, `audio_out (≥C, L_ao) int`, `modality_flag (1,L)` with values from `LFMModality` (TEXT=1, AUDIO_IN=2, AUDIO_OUT=3). Steps:
1. **Text:** `text_emb = lfm.embed_tokens(text[0])` → `(L_t, 2048)` (py:334).
2. **Audio-in:** split the concatenated mel along time by `audio_in_lens`, `pad_sequence` to a batch, run `self.conformer(mel.to(text_emb.dtype), lens)` → `(B, 512, T')` plus per-clip `audio_in_len` (8× subsampled, `mel2emb_len = ceil(l/8) = -(l//-8)`). A length mask un-pads and concatenates valid frames → `(ΣT', 512)`, then `audio_adapter` → `audio_in_emb (ΣT', 2048)` (py:339-353). The mel is cast to the model dtype **before** the conformer (the conformer runs in model dtype).
3. **Audio-out:** `offset_audio_tokens = audio_out[:C] + codebook_offsets[:,None]` maps each codebook's codes into the flat table, then `audio_embedding(offset).sum(0)` sums the 8 per-codebook embeddings into one `(L_ao, 2048)` vector per frame (py:358-359). This sum-over-codebooks is the audio-frame embedding.
4. **Scatter:** allocate `in_emb (B,L,2048)` and boolean-scatter each part into its modality positions: `in_emb[modality==TEXT]=text_emb`, `[==AUDIO_IN]=audio_in_emb`, `[==AUDIO_OUT]=audio_out_emb` (py:366-370). Asserts enforce that the flag counts exactly match the part lengths.

### Generation — interleaved (`generate_interleaved`, py:233-305)
Decode is a `@torch.no_grad()` Python generator (sync stream). After `_prefill`, it loops up to `max_new_tokens`, maintaining `current_modality`, `modality_left` (a countdown), `text_done`, and an `Lfm2HybridConvCache`. Each step:
- `lfm(inputs_embeds=in_emb, past_key_values=cache, use_cache=True)` → `last_hidden_state (1,seq,2048)`; take the **last** position `output_embeddings[0,-1]` (py:264-269).
- **TEXT mode:** `text_logits = F.linear(h, lfm.embed_tokens.weight)` — the **tied** LM head, `(65536,)`; sample; break on `<|im_end|>`(7); set `text_done` on `<|text_end|>`(130); when `modality_left` hits 0 or `text_done`, flip to AUDIO_OUT with `modality_left=interleaved_n_audio`. Next `in_emb = lfm.embed_tokens(next_token)` (py:272-287).
- **AUDIO_OUT mode:** `next_token = _sample_audio_frame(h)` → 8 codes; if `modality_left==0 and not text_done`, flip back to TEXT (`interleaved_n_text`); if `code[0]==2048` (EOAudio) force the whole frame to 2048 and flip to TEXT; next `in_emb = audio_embedding(frame + codebook_offsets).sum(0)[None,None,:]` — exactly the prefill audio-frame embedding, fed back autoregressively (py:289-305).

`generate_sequential` (py:171-231) is the same but emits **all** text first, switches to AUDIO_OUT only on `<|audio_start|>`(128), and has no interleave countdown — the ASR/TTS path.

### Audio-frame decode (`_sample_audio_frame`, py:501-534)
This is the **depthformer inner loop** — a tiny autoregressive transformer over the 8 codebooks for one acoustic frame:
1. `depthformer_in = rearrange(depth_linear(embedding), "(C D) -> C D")` → `(8, depthformer_dim)`, one input vector per codebook (py:509).
2. `depthformer_token = zeros(D)`; loop `i in 0..C`: `cur = depthformer_in[i] + depthformer_token`; run **one** depthformer step `depthformer.forward_cached(cur[None,None,:], cache)` with a growing KV cache (the C codebooks are the "sequence"); `logits = depth_embeddings[i].get_logits(out)` (per-codebook RMSNorm → tied/own `to_logits`, `(2049,)`); sample `next_token`; **feed it back** as `depthformer_token = depth_embeddings[i](next_token)` for the next codebook (py:514-532). So codebook `i` is conditioned on the backbone hidden + the embeddings of codes `0..i-1` — residual-vector-quantizer-style coarse-to-fine prediction.
3. Return `cat(out_tokens)` → `(8,)`.

### Sampling (`_sample_text_token` py:483-499, audio per-codebook py:519-529)
`greedy = temperature is None or temperature<=0 or top_k==1`. Greedy ⇒ `logits.argmax`. Else `logits /= temperature`; **threshold** top-k: `min_score = topk(logits,k).values[-1]; logits[logits<min_score] = -inf` (ties at the boundary are **kept** — not exactly-k); `probs = softmax(logits)`; `torch.multinomial(probs,1)`.

### Training (`logits`/`forward`, py:374-481 — off inference path)
Teacher-forced: prefill, run the backbone once (`use_cache=False`), shift `out_emb[:, :-1]`, select supervised text/audio positions. Text logits via the tied head. Audio: `depth_linear` → `(n,C,D)`, add the rolled teacher-code embeddings (`roll(1)` along C so codebook `i` sees code `i-1`; the last codebook's contribution is zeroed before the roll → codebook 0 sees zero), run the depthformer **in parallel** over all C as one causally-masked sequence (chunked if `n≥2^14`), per-codebook logits, cross-entropy with `audio_loss_weights`.

## Dtypes & shapes
| Stage | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `_prefill` text branch | `text` int64 `(1,L_t)` | text_emb model-dtype (bf16/f32) `(L_t,2048)` |
| `_prefill` audio-in | mel bf16 `(128,ΣT_mel)` → cast model-dtype | audio_in_emb model-dtype `(ΣT',2048)`, T'=⌈T_mel/8⌉ |
| `_prefill` audio-out | codes int `(C,L_ao)` | audio_out_emb model-dtype `(L_ao,2048)` (sum over 8 codebooks) |
| `_prefill` assembled | masks `(1,L)` | `in_emb` model-dtype `(1,L,2048)` |
| backbone step (`self.lfm`) | `in_emb` model-dtype `(1,seq,2048)` | hidden model-dtype `(1,seq,2048)`; last → `(2048,)` |
| text head (tied) | hidden `(2048,)` | text_logits `(65536,)` (f32-upcast matmul in Rust) → sampled int64 `(1,)` |
| `depth_linear` | hidden `(2048,)` | `(C·depthformer_dim,)` → `(8,depthformer_dim)` |
| depthformer step | `(1,1,depthformer_dim)` + KV cache | `(1,1,depthformer_dim)`; per-cb logits `(2049,)` |
| `_sample_audio_frame` | hidden `(2048,)` | audio frame int `(8,)`, codes 0..2048 (2048=EOAudio) |
| audio-frame feedback | frame `(8,)` int | `in_emb` model-dtype `(1,1,2048)` |

Internal promotions: RMSNorm (backbone, depthformer, embedding_norm) normalizes in **f32** then casts back; softmax in attention/sampling in f32; RoPE built in f32 (`view_as_complex`). Backbone/depthformer weights bf16 on disk; Rust CPU compute f32, Metal bf16. Token ids int64 (Python) / u32 (Rust ChatState); mel f32/f64 front-end → stored bf16 in ChatState.

## Wiring
**Upstream (feeds `_prefill`):**
- [core_processor](CO01-Processor-ChatState) — builds the `ChatState` (text int64 `(1,L_t)`, mel bf16 `(128,ΣT_mel)`, audio_in_lens, audio_out codes, modality_flag) consumed by `_prefill`.
- [conformer_encoder](CF01-Conformer-Encoder) — `self.conformer`; mel model-dtype `(B,128,T)` → `(B,512,T')`.
- [model_mlp](MD03-Audio-Adapter-MLP) — `self.audio_adapter`; conformer `(ΣT',512)` → `(ΣT',2048)`.
- [model_lfm2_backbone](MD01-LFM2AudioModel) — `self.lfm`; `in_emb (1,seq,2048)` → hidden `(1,seq,2048)` + `embed_tokens` weight for the tied head.
- [model_transformer](MD04-Depthformer) — `RawLMBackbone`/`StandardBlock`/`MHA`/`SharedEmbedding`; the depthformer + audio embedding tables.
- [core_utils](CO03-Utils) — `LFMModality` enum + `mel2emb_len` for length math.

**Downstream (consumes this output):**
- [core_processor](CO01-Processor-ChatState) — `generate_interleaved`/`generate_sequential` yield: text ids int64 `(1,)` (detokenized to string) and audio frames int `(8,)` per Mimi-frame, which the processor routes to `decode()`.
- [core_detokenizer](CO02-Detokenizer) / [moshi_compression](MM01-Mimi-Codec) — the emitted audio frames `(8,)` u32 codes are fed (via the processor's dispatch) to the LFM2 ISTFT detokenizer or Mimi `decode` → f32 waveform @ 24kHz.

## Python ↔ Rust
| Python symbol | Rust symbol | Note |
|---|---|---|
| `LFM2AudioModel.__init__` | `LFM2AudioModel::new` | same assembly; `LossConf` bundles training-only fields |
| `from_pretrained` | `from_pretrained` → `crate::loader::from_pretrained` | Python defaults `device="cuda",bf16`; Rust device/dtype-agnostic (CPU→f32, Metal→bf16) — §2.1 |
| `_prefill` | `prefill_inputs` | Python boolean-mask scatter → Rust `index_select` with a per-position index (deliberate, §1.3); unknown flag errors instead of silently bucketing |
| `generate_interleaved`/`generate_sequential` (generator) | same names, **sync callback** `FnMut(GenToken)` | sync streaming faithful to the Python generator (async only at transport) |
| `_sample_text_token` / per-codebook sampler | `Sampler` over `candle_transformers::LogitsProcessor` | greedy=`ArgMax` (byte-identical); Torch **threshold** top-k injected via `sample_f` hook (candle's `TopK` keeps exactly k; Torch keeps ties) — §2.3/§2.8 |
| `_sample_audio_frame` | `sample_audio_frame` | `LayerKvCache` per depthformer layer; Python 1-D `nn.Linear` → Rust needs 2-D input (caught bug) |
| `lfm.embed_tokens` tied head | `lfm.embed_weight()` matmul, **f32-upcast** | `text_logits` in f32 regardless of model dtype |
| `self.lfm` (HF `Lfm2Model`) | `crate::model::lfm2_hf::Model` | external transformers → the Rust `lfm2_hf.rs` is the readable spec; flash/sdpa CUDA kernels → eager matmul+causal-mask+softmax (§2.2) |
| `Lfm2HybridConvCache` | `lfm2_hf::Cache` (`LfmCache`) | short-conv `conv_L_cache` + GQA KV cache; candle `Conv1d`/gather (§2.2) |
| `logits`/`forward` (training) | `logits`/`forward` | present for inventory; `cross_entropy(reduction="none")` → `candle_ext::loss::cross_entropy_none` (§2.6) |

## Precision / gotchas
- **RMSNorm bf16 order.** Every RMSNorm in this graph (backbone, depthformer block norms, qk-RMSNorm, the `embedding_norm` inside each `SharedEmbedding.get_logits`) does `(_norm(x.float()) * weight).type_as(x)` (transformer.py:77-78) — normalize in f32, **multiply weight in f32, then cast**. A naive `candle_nn::RmsNorm` casts back before the weight multiply; at bf16 these differ, so the Rust composes the op in liquid_audio's order (§2.4). eps=1e-5 for these heads (1e-6 default elsewhere).
- **EOAudio handling.** Code value `2048` (= `audio_vocab_size-1`) is EOAudio. Generation checks **only codebook 0** (`next_token[0]==2048`); on hit it forces the *entire* 8-code frame to 2048 (`next_token[:]=2048`) and returns to TEXT (py:226-228, 300-302). The flat `audio_embedding` table is sized `2049·8` precisely to host this per-codebook EOAudio row.
- **Tied vs untied heads.** The text head reuses `lfm.embed_tokens.weight` directly via `F.linear` (no separate projection); the depthformer per-codebook heads use `SharedEmbedding.get_logits` = `to_logits(embedding_norm(x))`, tied to the codebook's embedding when `depthformer_tie`. Don't confuse the two: text logits skip an `embedding_norm`, audio logits don't.
- **Special tokens are literal ints.** 7=`<|im_end|>`, 128=`<|audio_start|>`, 130=`<|text_end|>` are hard-coded (not config) and drive the modality state machine.
- **Audio-frame embedding is a sum, not a concat.** Both prefill and feedback embed the 8 codes through one shared 16392-row table (offset per codebook) and **sum** them into a single 2048-vector — losing per-codebook identity by design; the depthformer recovers ordering at decode via its causal C-sequence.
- **`mel2emb_len = -(l // -8)`** is ceil-division (`⌈l/8⌉`), the 8× conformer subsample; smallest valid mel length is 9. Used in the prefill asserts and the audio-in length mask; an off-by-one here desyncs the modality scatter.
- **Cross-library f32 floor.** Backbone hidden parity 6.56e-6, text logits 5.51e-6, depthformer audio frame **token-exact** (no float reduction in argmax/gather) — the ~1e-6 residual is gemm-order/transcendental/FFT rounding, irreducible without re-implementing candle's kernels (§1.4).
- **Stochastic RNG diverges.** Greedy decoding is deterministic and matches Python token-for-token; temperature>0 uses candle's `LogitsProcessor` RNG (different stream from torch.multinomial) — never byte-reproducible cross-framework, but the token set + threshold-top-k distribution match (§2.8).
