# liquid_audio → Rust port status

Faithful, module-for-module. Goal: equivalent structure, function lists, and API
calls; same IO model (sync streaming generator for the model — async only at the
websocket transport, per upstream). Parity = numerically verified against the
Python with shared weights + fixed inputs.

| Python module | LOC | Rust module | Status | Notes |
|---|---:|---|---|---|
| `utils.py` | 54 | `src/utils.rs` | **partial** | `LFMModality`, `mel2emb_len`, `emb2mel_len`, `module_exists` ported; `get_model_dir` lands with `processor.rs` (its consumer, needs hf-hub) |
| `model/mlp.py` | 40 | `src/model/mlp.rs` | **done** | candle `Sequential`; GELU = erf (matches `nn.GELU()`); weight indices mirror `nn.Sequential` |
| `processor.py` | 269 | `src/processor.rs` | **done (compiles; parity pending)** | tokenizer (`tokenizers`), mel preprocessor, Mimi decode (via `moshi` crate), `ChatState` |
| `model/transformer.py` | 578 | `src/model/transformer.rs` | **done (compiles; parity pending)** | LFM2 backbone (own impl, not HF Lfm2): RMSNorm, SwiGLU GLU, BoundedAttention (GQA + qk-RMSNorm + interleaved RoPE via `rope_i`), MHA, StandardBlock, SharedEmbedding (tied), RawLmBackbone, LayerKvCache. Training-only bits (init scales, activation checkpoint, `forward_cached` split) omitted |
| `detokenizer.py` | 136 | `src/detokenizer.rs` | **done (compiles; parity pending)** | FusedEmbedding + Vocos-style ISTFT (needs inverse FFT via `rustfft` + overlap-add `fold`) + Lfm2Model backbone |
| `processor.py` | 269 | `src/processor.rs` | **done (compiles; parity pending)** | tokenizer (`tokenizers`), mel preprocessor, Mimi decode (via `moshi` crate), `ChatState` |
| `model/lfm2_audio.rs` | 534 | `src/model/lfm2_audio.rs` | **done (compiles; parity pending)** | `LFM2AudioModel` + `generate_interleaved` (sync streaming iterator) |
| (HF transformers `Lfm2Model`) | ~660 | `src/model/lfm2_hf.rs` | **done (compiles; parity pending)** | main LFM2 backbone (hybrid short-conv + GQA attn), adapted from candle main lfm2.rs to candle 0.9; returns all-position hidden state + custom-mask forward |
| `model/conformer/utils.py` | 112 | — | **skip** | autocast/streaming/stochastic-depth helpers — not on inference path |
| `model/conformer/mha.py` | 457 | `src/model/conformer/mha.rs` | **done (compiles; parity pending)** | RelPositionalEncoding + RelPositionMultiHeadAttention (manual branch, no cache/streaming/sdpa) |
| `model/conformer/modules.py` | 471 | `src/model/conformer/modules.rs` | **done (compiles; parity pending)** | ConformerLayer / ConformerConvolution / FeedForward / CausalConv1D |
| `model/conformer/subsampling.rs` | 605 | `src/model/conformer/subsampling.rs` | **done (compiles; parity pending)** | ConvSubsampling forward (skip conv chunking) |
| `model/conformer/encoder.rs` | 1163 | `src/model/conformer/encoder.rs` | **done (compiles; parity pending)** | ConformerEncoder offline forward (skip streaming/export/stochastic) |
| `model/conformer/processor.rs` | 556 | `src/model/conformer/processor.rs` | **done (compiles; parity pending)** | AudioToMelSpectrogramPreprocessor / FilterbankFeatures (STFT via `rustfft`) |
| `moshi/*` | 8715 | — | **reuse** | the `moshi` crate (Kyutai's own Rust port) — identical upstream to the vendored copy |

## IO model (faithful to Python)
- Model / `generate_interleaved`: **synchronous generator** → Rust **sync streaming `Iterator`** (no async).
- Demo: background thread + queue → Rust `std::thread` + channel.
- `moshi/server.py`, `moshi/client.py`: asyncio + aiohttp websockets → Rust **async (tokio)** *only if* we port the transport.

## Verification
Per-module numerical parity harness (planned): dump Python reference tensors for
fixed inputs + shared safetensors weights, load the same in Rust, assert match
within tolerance before moving up the dependency chain.
