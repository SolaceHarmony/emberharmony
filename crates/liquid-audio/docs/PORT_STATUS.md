# liquid_audio → Rust port status

> **Archived oracle history.** This file records the former Rust/Candle
> transliteration and is not the production ownership ledger. Current LFM2.5
> production is fully native; it loads main plus `audio_detokenizer/` into one
> sealed image and routes output through `REQ_AUDIO_DETOKENIZE`. Rust loads no
> weights and owns no model math. Native Mimi is retained only for future Moshi
> and is not an LFM2.5 fallback. See `RUST_DELETION_PLAN.md`,
> `DECODE_ENGINE.md`, and `AUDIO_DETOKENIZER_PORT.md`.

Faithful on the executable inference/training surface, without compatibility
methods that only returned their inputs or did nothing. Same IO model (the model is a
**synchronous streaming generator** — async only at the websocket transport, per
upstream). Python-only framework hooks are documented omissions rather than callable
Rust no-ops. All modules compile without warnings.
**Faithful = numerically verified** against Python — see PARITY.md.

## 1:1 inventory (38/38 in-scope classes mapped)

Every runtime class and operation in the in-scope Python modules (`utils`, `detokenizer`,
`processor`, `model/mlp`, `model/transformer`, `model/lfm2_audio`, and all of
`model/conformer/`) has a Rust counterpart. Python **inheritance** is modelled by
**composition** (the subclass holds its base and calls its methods, where Python
uses `super()`); Python **ABCs** become Rust **traits**:

- `PositionalEncoding` ← `RelPositionalEncoding`, `MultiHeadAttention` ←
  `RelPositionMultiHeadAttention` — base structs composed by the rel-pos subclasses (`mha.rs`).
- `SequenceModel` (ABC) → trait, impl'd by `RawLmBackbone` (`transformer.rs`).
- `AudioPreprocessor` (ABC) → trait; `AudioToMelSpectrogramPreprocessor` wraps
  `FilterbankFeatures` (`src/processor.rs`).
- `CausalConv1D` (`modules.rs`), `MaskedConvSequential` (`subsampling.rs`),
  `CacheAwareStreamingConfig` + stochastic-depth/autocast fns (`conformer/utils.rs`),
  `LFM2_HFConfig`/`LFM2AudioConfig`/`LFM2AudioModelOutput` (`lfm2_audio.rs`), and
  the full `ConformerEncoder` streaming/export method set — all ported.

The un-fold **changed structure only**: the full parity suite re-ran byte-identical
afterward (conformer 8.25e-7, backbone 6.6e-6, depthformer token-exact, prefill 1.1e-6).

### Python-only framework hooks

Members with no candle behavior are omitted instead of exposed as callable no-ops:

- torch activation-checkpoint wrappers and `nn.Module` mode/device bookkeeping;
- NeMo pickle persistence and the preprocessor's ONNX dummy-input hook;
- training-time weight re-initializers (weights are loaded from safetensors).

The concrete Conformer streaming/export helpers remain because they construct real
masks, cache tensors, and outputs; they are not inventory placeholders.

The IO model (sync streaming generator) is the only deliberate semantic mapping:
Python's `forward_cached(x, cache) -> (out, cache)` is Rust's in-place
`forward(x, Some(&mut cache))` (documented on the `SequenceModel` trait).

| Python module | LOC | Rust module | Status |
|---|---:|---|---|
| `utils.py` | 54 | `src/utils.rs` | ✅ done (incl. `get_model_dir` snapshot-download via `hf-hub`) |
| `model/mlp.py` | 40 | `src/model/mlp.rs` | ✅ done |
| `model/transformer.py` | 578 | `src/model/transformer.rs` | ✅ done — depthformer + shared embeddings backbone (RMSNorm, SwiGLU, GQA attn + qk-RMSNorm + interleaved RoPE, MHA, StandardBlock, SharedEmbedding, RawLmBackbone, KV cache) |
| (HF `Lfm2Model`) | ~660 | `src/model/lfm2_hf.rs` | ✅ done — **main** backbone (hybrid short-conv + GQA attn), adapted from candle `lfm2.rs`; all-position hidden + custom-mask forward |
| `model/conformer/utils.py` | 112 | `src/model/conformer/utils.rs` | ✅ done — `CacheAwareStreamingConfig` + `compute_stochastic_depth_drop_probs`; `avoid_float16_autocast_context` is a `// PORT:` no-op (no candle autocast) |
| `model/conformer/mha.py` | 457 | `src/model/conformer/mha.rs` | ✅ done — RelPositionalEncoding + RelPositionMultiHeadAttention |
| `model/conformer/modules.py` | 471 | `src/model/conformer/modules.rs` | ✅ done — ConformerLayer / Conv / FeedForward |
| `model/conformer/subsampling.py` | 605 | `src/model/conformer/subsampling.rs` | ✅ done — dw_striding ConvSubsampling |
| `model/conformer/encoder.py` | 1163 | `src/model/conformer/encoder.rs` | ✅ done — offline ConformerEncoder forward |
| `model/conformer/processor.py` | 556 | `src/processor.rs` | ✅ done — mel featurizer (rustfft STFT, computed hann + slaney mel) |
| `detokenizer.py` | 136 | `src/detokenizer.rs` | ✅ done — FusedEmbedding + Vocos ISTFT (rustfft) + lfm2_hf backbone |
| `processor.py` | 269 | `src/processor.rs` | ✅ done — tokenizer (`tokenizers`) + mel + `ChatState` + detok decode |
| `model/lfm2_audio.py` | 534 | `src/model/lfm2_audio.rs` | ✅ done — LFM2AudioModel + prefill + `generate_interleaved` (callback stream) + depthformer |
| (from_pretrained) | — | `src/loader.rs` | ✅ done — config.json → configs → safetensors VarBuilder → model + processor |
| `moshi/*` | 8715 | — | ♻ reuse the `moshi` crate (Kyutai's own Rust port — identical upstream) |

## Remaining refinements (documented, non-structural)
- **Sampling**: ✅ done — greedy + temperature/top-k (multinomial via seeded
  `StdRng`) for text and audio, faithful to `_sample_text_token` /
  `_sample_audio_frame`; `GenParams` threaded through `generate_interleaved` and
  the now-ported `generate_sequential`. Unit-tested (greedy/top-k/determinism).
- **dtype**: ✅ resolved — `from_pretrained(dir, device)` derives persistent
  weight dtype from the safetensors tensor headers. BF16 checkpoint weights stay
  BF16 on CPU and Metal; the caller cannot request an F32 model copy. F32 remains
  only where the implementation explicitly needs local accumulation or audio/math
  buffers.
- **hf-hub auto-download**: ✅ done — `get_model_dir(repo_or_path, revision)`
  snapshot-downloads a HF repo id via the `hf-hub` crate (sync, the
  `snapshot_download` analog) and returns the snapshot dir; local paths pass
  through (revision-with-path is an error, as in Python). Gated behind the
  `download` feature (on by default). `from_pretrained_hub(repo_id, ...)` is the
  faithful repo-id entry point.
- **Audio-out**: ✅ unified behind an `AudioDetokenizer` trait we own
  (`src/runtime/audio_out.rs`) — the processor dispatches `decode` through
  `Box<dyn AudioDetokenizer>` and never touches a concrete codec. Two backends:
  - **LFM2 detokenizer** (LFM2.5 models): ported in-tree (`detokenizer.rs`), pure
    candle; weights in `LiquidAI/LFM2.5-Audio-1.5B` under `audio_detokenizer/`.
  - **Mimi codec** (v1 models): the **`moshi` crate** (Kyutai's own Mimi, the Rust
    port of the vendored `liquid_audio/moshi`; pins candle ^0.9.1 = our 0.9.2),
    wrapped as `MimiDetokenizer`. It loads the moshi-format checkpoint
    (`encoder.model.N.conv.conv…`, split `rvq_first`/`rvq_rest`) that ships in the
    repo. (candle-transformers' `mimi`, 0.9 *and* 0.10, uses the Encodec-style
    `encoder.layers.N`/weight-norm layout and can NOT load this checkpoint.)

  The historical Rust loader picked the LFM2 detokenizer when
  `audio_detokenizer/` was present and otherwise selected Mimi. That fallback
  no longer exists in production: the native LFM2.5 loader requires the
  released detokenizer and fails if it is absent.
  Fully vendoring the Mimi codec in-tree (mirroring `liquid_audio/moshi`, ~3.8k
  candle LOC) would drop the external crate — a documented option, not yet done.
- **Parity**: ✅ verified against the real upstream + actual weights
  (LFM2-Audio-1.5B, f32, CPU) across the full pipeline — understanding,
  generation heads, and the prefill assembly:
  - mel featurizer **1.1e-5**
  - FastConformer encoder **8.3e-7** (every stage ≤ 1.6e-6)
  - lfm backbone **6.6e-6**
  - text head (`text_logits`) **5.5e-6**
  - depthformer audio frame **token-exact**
  - prefill modality-scatter **1.1e-6**

  Bugs caught + fixed via the harness: STFT frame off-by-one
  (`torch.stft(center=True)` ⇒ `1 + L/hop`); the `lfm.*` weight-key layout (bare
  `Lfm2Model`, `embedding_norm`); the conformer length convention (full mel width,
  not `mel_len`); and a latent 1-D `Linear` in the depthformer sampler (candle
  needs a 2-D input). The only remaining untested piece is the **audio-out
  detokenizer** (needs the `audio_detokenizer/` weights — absent from the 1.5B
  repo, which ships the v1 Mimi path; deferred with the v1 Mimi decode). The
  generate loop is a deterministic state machine over these verified components.
  Workflow in PARITY.md.

## IO model (faithful to Python)
- Model / `generate_interleaved`: synchronous streaming → Rust synchronous callback stream (no async).
- Demo thread+queue → `std::thread` + channel; `moshi` websocket server/client → async (tokio) only if the transport is ported.
