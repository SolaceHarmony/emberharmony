# Moshi per-file status (Rust port)

The vendored Python `liquid_audio/moshi/**` is **reused via the `moshi` crate**,
not re-ported in-tree. Each Python file's Rust status is a single line below.
The Rust port's only in-tree moshi code is `audio_out.rs::MimiDetokenizer`
(a thin adapter over `moshi::mimi::Mimi`).

## On-path (codec — decode at inference, encode at training-prep)

| Python file | Rust status | Notes |
|---|---|---|
| `models/compression.py` (`MimiModel`) | **reused** via `moshi::mimi::Mimi` | wrapped by `audio_out.rs::MimiDetokenizer`; `encode`/`decode`/`decode_step`/`reset_stream` exposed through the `AudioDetokenizer` trait. |
| `models/loaders.py` (`get_mimi`) | **reused** via `moshi::mimi::load` | `Some(codebooks)` arg selects 8 active codebooks. Off-path `CheckpointInfo`/`get_moshi_lm` not ported. |
| `modules/seanet.py` (`SEANetEncoder/Decoder`) | **inside the `moshi` crate** | not in-tree; Kyutai's own Rust port. |
| `modules/resample.py` (`ConvDownsample1d`/`ConvTrUpsample1d`) | **inside the `moshi` crate** | the 25↔12.5 Hz bridge. |
| `modules/transformer.py` (`ProjectedTransformer`) | **inside the `moshi` crate** | enc/dec transformers at 25 Hz. |
| `modules/conv.py` | **inside the `moshi` crate** | causal conv helpers. |
| `modules/streaming.py` | **inside the `moshi` crate** | `StreamingModule`/`StreamingContainer` state machine. |
| `modules/rope.py` | **inside the `moshi` crate** | RoPE for the enc/dec transformers. |
| `modules/gating.py` | **inside the `moshi` crate** | gating primitives. |
| `modules/lora.py` | **inside the `moshi` crate** | LoRA (off-path for LFM2-Audio). |
| `quantization/vq.py` (`SplitResidualVectorQuantizer`) | **inside the `moshi` crate** | `rvq_first`/`rvq_rest`; weight keys match this checkpoint. |
| `quantization/core_vq.py` (`EuclideanCodebook`) | **inside the `moshi` crate** | `cdist`+`argmin` nearest-centroid. |
| `quantization/base.py` | **inside the `moshi` crate** | quantizer base. |

## Off-path (not ported to Rust — a different model / reference only)

| Python file | Rust status | Notes |
|---|---|---|
| `models/lm.py` (`LMModel`/`LMGen`) | **not ported** | Moshi 7B multi-stream LM; LFM2-Audio uses its own backbone + depthformer. |
| `models/lm_utils.py` | **not ported** | Moshi LM helpers. |
| `models/tts.py` | **not ported** | Moshi TTS. |
| `server.py` | **not ported** | asyncio/ws transport for Moshi 7B; no async runtime in the Rust port. |
| `client.py` / `client_gradio.py` / `client_utils.py` | **not ported** | Moshi client tooling. |
| `run_inference.py` / `run_tts.py` | **not ported** | Moshi demo scripts. |
| `conditioners/base.py` / `tensors.py` / `text.md` | **not ported** | Moshi-LM conditioning. |
| `utils/autocast.py` / `compile.py` / `quantize.py` / `sampling.py` / `utils.py` | **not ported** | Moshi training/inference utils. |

## The Rust seam

```
                  AudioDetokenizer trait (audio_out.rs)
                           ▲
            ┌──────────────┴──────────────┐
   MimiDetokenizer            LFM2AudioDetokenizer
   (wraps moshi::mimi::Mimi)  (in-tree, detokenizer.rs)
            ▲
            │ processor.rs::decode dispatches
            │ audio_out.or(mimi)
```

The processor (`processor.rs:138`) dispatches `decode` through
`Box<dyn AudioDetokenizer>` and never touches a concrete codec type. The loader
picks the LFM2 detokenizer if `audio_detokenizer/` is present, else Mimi.

## See also
- [`glm-version/moshi/README.md`](README.md) — the overview.
- [`ARCH/moshi/README.md`](../../ARCH/moshi/README.md) — the vendored Python
  overview (with the component wiring diagram + per-file specs).
- `liquid-audio-rs/src/audio_out.rs` — the `AudioDetokenizer` trait +
  `MimiDetokenizer` adapter.