# liquid_audio → Rust port status

Faithful, module-for-module on candle. Goal: equivalent structure, function lists,
and API calls; same IO model (the model is a **synchronous streaming generator** —
async only at the websocket transport, per upstream). All modules compile.
**Faithful = numerically verified** against Python — see PARITY.md (harness built;
run it with the model present to close the parity column).

| Python module | LOC | Rust module | Status |
|---|---:|---|---|
| `utils.py` | 54 | `src/utils.rs` | ✅ done (incl. `get_model_dir` snapshot-download via `hf-hub`) |
| `model/mlp.py` | 40 | `src/model/mlp.rs` | ✅ done |
| `model/transformer.py` | 578 | `src/model/transformer.rs` | ✅ done — depthformer + shared embeddings backbone (RMSNorm, SwiGLU, GQA attn + qk-RMSNorm + interleaved RoPE, MHA, StandardBlock, SharedEmbedding, RawLmBackbone, KV cache) |
| (HF `Lfm2Model`) | ~660 | `src/model/lfm2_hf.rs` | ✅ done — **main** backbone (hybrid short-conv + GQA attn), adapted from candle `lfm2.rs`; all-position hidden + custom-mask forward |
| `model/conformer/utils.py` | 112 | — | ⏭ skip (autocast/streaming/stochastic-depth — off inference path) |
| `model/conformer/mha.py` | 457 | `src/model/conformer/mha.rs` | ✅ done — RelPositionalEncoding + RelPositionMultiHeadAttention |
| `model/conformer/modules.py` | 471 | `src/model/conformer/modules.rs` | ✅ done — ConformerLayer / Conv / FeedForward |
| `model/conformer/subsampling.py` | 605 | `src/model/conformer/subsampling.rs` | ✅ done — dw_striding ConvSubsampling |
| `model/conformer/encoder.py` | 1163 | `src/model/conformer/encoder.rs` | ✅ done — offline ConformerEncoder forward |
| `model/conformer/processor.py` | 556 | `src/model/conformer/processor.rs` | ✅ done — mel featurizer (rustfft STFT, computed hann + slaney mel) |
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
- **dtype**: ✅ resolved — `from_pretrained(dir, dtype, device)` mirrors the
  Python `dtype=` kwarg. The on-disk checkpoint is bf16, so `DType::F32` loads
  the *exact* bf16-rounded weights and upcasts (bf16→f32 is lossless): the weight
  values already match the deployed model, and compute is simply more precise.
  The parity reference is dumped at `torch.float32`, so there is no dtype gap
  against it. True in-memory bf16 is accepted for CUDA/Metal but rejected on CPU
  (candle has no CPU bf16 matmul kernel) with a clear error.
- **hf-hub auto-download**: ✅ done — `get_model_dir(repo_or_path, revision)`
  snapshot-downloads a HF repo id via the `hf-hub` crate (sync, the
  `snapshot_download` analog) and returns the snapshot dir; local paths pass
  through (revision-with-path is an error, as in Python). Gated behind the
  `download` feature (on by default). `from_pretrained_hub(repo_id, ...)` is the
  faithful repo-id entry point.
- **Mimi audio-out (v1)**: the LFM2.5 detokenizer path is ported; the v1 `processor.mimi` (moshi-crate) decode path is deferred.
- **Parity**: harness built (PARITY.md); run against the model to verify the numbers.

## IO model (faithful to Python)
- Model / `generate_interleaved`: synchronous streaming → Rust synchronous callback stream (no async).
- Demo thread+queue → `std::thread` + channel; `moshi` websocket server/client → async (tokio) only if the transport is ported.
