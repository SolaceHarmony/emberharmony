# liquid_audio → Rust port status

Faithful, module-for-module. Goal: equivalent structure, function lists, and API
calls; same IO model (sync streaming generator for the model — async only at the
websocket transport, per upstream). Parity = numerically verified against the
Python with shared weights + fixed inputs.

| Python module | LOC | Rust module | Status | Notes |
|---|---:|---|---|---|
| `utils.py` | 54 | `src/utils.rs` | **partial** | `LFMModality`, `mel2emb_len`, `emb2mel_len`, `module_exists` ported; `get_model_dir` lands with `processor.rs` (its consumer, needs hf-hub) |
| `model/mlp.py` | 40 | `src/model/mlp.rs` | **done** | candle `Sequential`; GELU = erf (matches `nn.GELU()`); weight indices mirror `nn.Sequential` |
| `processor.py` | 269 | `src/processor.rs` | pending | tokenizer (`tokenizers`), mel preprocessor, Mimi decode (via `moshi` crate), `ChatState` |
| `model/transformer.py` | 578 | `src/model/transformer.rs` | **done (compiles; parity pending)** | LFM2 backbone (own impl, not HF Lfm2): RMSNorm, SwiGLU GLU, BoundedAttention (GQA + qk-RMSNorm + interleaved RoPE via `rope_i`), MHA, StandardBlock, SharedEmbedding (tied), RawLmBackbone, LayerKvCache. Training-only bits (init scales, activation checkpoint, `forward_cached` split) omitted |
| `detokenizer.py` | 136 | `src/detokenizer.rs` | pending (after backbone) | FusedEmbedding + Vocos-style ISTFT (needs inverse FFT via `rustfft` + overlap-add `fold`) + Lfm2Model backbone |
| `processor.py` | 269 | `src/processor.rs` | pending | tokenizer (`tokenizers`), mel preprocessor, Mimi decode (via `moshi` crate), `ChatState` |
| `model/lfm2_audio.py` | 534 | `src/model/lfm2_audio.rs` | pending | `LFM2AudioModel` + `generate_interleaved` (sync streaming iterator) |
| `model/conformer/*` | ~3360 | `src/model/conformer/*` | **decision: A (candle line-for-line port)** | FastConformer encoder; no Rust equiv. Big separable batch — candidate for parallel drafting |
| `moshi/*` | 8715 | — | **reuse** | the `moshi` crate (Kyutai's own Rust port) — identical upstream to the vendored copy |

## IO model (faithful to Python)
- Model / `generate_interleaved`: **synchronous generator** → Rust **sync streaming `Iterator`** (no async).
- Demo: background thread + queue → Rust `std::thread` + channel.
- `moshi/server.py`, `moshi/client.py`: asyncio + aiohttp websockets → Rust **async (tokio)** *only if* we port the transport.

## Verification
Per-module numerical parity harness (planned): dump Python reference tensors for
fixed inputs + shared safetensors weights, load the same in Rust, assert match
within tolerance before moving up the dependency chain.
