# demo_model (Rust port — not ported)
**Source:** no Rust counterpart · **Python:** `upstream-liquid-audio/…/demo/model.py` (not in the vendored tree; see `ARCH/demo/model.md`) · **On the LFM2-Audio inference path:** no

> Companion to [`ARCH/demo/model.md`](../../ARCH/demo/model.md). The original
> documents the Python `model.py` singleton-loader/warmup module; the Rust port
> does **not** port it — the Rust tree achieves the same singleton-construction
> device-agnostically in `loader.rs` / `mic_chat.rs`.

## Role
`demo/model.py` is the Python demo's **singleton-loader / warmup module**: 26
lines whose entire job is to construct, at import time, the three shared
objects the Gradio chat needs — `proc`, `lfm2_audio`, `mimi` — and run a
5-iteration CUDA warmup of the Mimi streaming decoder. It exists so the
expensive `from_pretrained` loads and the codec's lazy CUDA-graph / kernel JIT
happen **once, eagerly, before the first user turn**. It carries no model math
of its own and is **not ported to Rust**.

## How the Rust port achieves the same thing
The Rust tree builds its singletons device-agnostically in `loader.rs` /
`mic_chat.rs` — no warmup loop, no CUDA-graph capture, no `torch.compile`.

| Python (`demo/model.py`) | Rust equivalent |
|---|---|
| `LFM2AudioProcessor.from_pretrained(HF_DIR)` | `loader.rs::from_pretrained(dir, dtype, device)` → processor (`processor.rs`) |
| `LFM2AudioModel.from_pretrained(HF_DIR)` | `loader.rs::from_pretrained` → `LFM2AudioModel` (`model/lfm2_audio.rs`); `from_pretrained_hub` for repo-id |
| `proc.mimi` alias | `loader.rs::load_mimi` → `moshi::mimi::load(...)` wrapped as `MimiDetokenizer` (`audio_out.rs`) |
| `mimi.streaming(1)` + 5× `mimi.decode` CUDA warmup | **no warmup loop** — candle has no CUDA-graph capture / `torch.compile`, so the JIT-warm rationale evaporates; cpal callbacks call `decode_step` directly (`mic_chat.rs`) |
| `device="cuda"`, `dtype=bf16` hard-coded | nothing hardcodes a device; loaders take `device: &Device` + `dtype: DType`, default `(Cpu, F32)`, Metal opt-in via `LFM_DEVICE=metal` → bf16 (§2.1) |
| attn = `flash_attention_2`/`sdpa` chosen at load | always eager `matmul + additive causal mask + softmax` (the sdpa/no-flash math; §2.2) |

## Why no warmup in Rust
candle has no CUDA-graph capture, no `torch.compile`, no cuDNN algorithm
selection — the first `mimi.decode_step` call runs the same eager kernels as
every subsequent call. There is no lazy init to pay down. The Rust
`mic_chat.rs` calls `decode_step` directly from the cpal callback; the first
frame has the same latency as the rest (modulo CPU caches warming up, which is
not something the program can or should pre-empt).

## The single deliberate divergence relevant here
**Device-coupling.** As written, `demo/model.py` requires CUDA (the warmup's
`device="cuda"` and the detok's `.cuda()` in `processor.py:151`); the Rust port
boots the full 1.5B end-to-end on `Device::Cpu` with no warmup stage. This is
§2.1 (device-agnostic) applied to the demo glue.

## Precision / gotchas (from the Python original, for reference)
- **CUDA-only by construction.** `torch.randint(2048, (1,8,1), device="cuda")`
  hard-pins CUDA; the module raises on a CPU/Metal-only host. Not a Rust
  concern — the Rust port doesn't have this module.
- **`randint(2048, …)` excludes EOAudio.** Codes are sampled in `[0, 2048)`, so
  the warmup never feeds the `2048` EOAudio sentinel into `mimi.decode` —
  matching the real loop, which skips any frame containing `2048`.
- **`.eval()` is load-bearing, not cosmetic.** It freezes the conformer conv
  module's BatchNorm running stats and disables dropout. The Rust port
  implicitly always runs in eval mode (candle places dtype/device at load;
  inference is always eval — no `train()`/`eval()` toggle).
- **No dtype gotchas in this file** — it introduces no casts. The bf16-weight /
  f32-CPU-floor / f64-mel / RMSNorm-bf16-order subtleties all live in the loaded
  components (see [`glm-version/model/lfm2_backbone.md`](../model/lfm2_backbone.md),
  [`glm-version/model/conformer/processor.md`](../model/conformer/processor.md),
  [`glm-version/model/transformer.md`](../model/transformer.md)).

## Cross-references
- [`ARCH/demo/model.md`](../../ARCH/demo/model.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.2 (kernel-free
  SDPA), §4 (demo out of parity surface — `liquid_audio/demo/**` is not ported).
- `liquid-audio-rs/src/loader.rs` — `from_pretrained` / `from_pretrained_hub`
  (the shared model+processor loader).
- `liquid-audio-rs/examples/mic_chat.rs` — the Rust demo that builds its own
  singletons. See [`glm-version/demo/chat.md`](chat.md).