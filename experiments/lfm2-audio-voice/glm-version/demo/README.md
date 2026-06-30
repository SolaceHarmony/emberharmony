# `demo/` — Rust-side architecture docs for the demo harness

Rust-first companions for the `liquid-audio/examples/` demo. The demo is
**out of the parity surface** (PYTHON_VS_RUST.md §4) — it is a faithful
headless re-expression, not a numerically-graded port.

| File | Rust source | Python source | Role |
|---|---|---|---|
| [`chat.md`](chat.md) | `examples/mic_chat.rs` | `demo/chat.py` | realtime speech-to-speech demo (cpal mic VAD → `ChatState` → `generate_interleaved` → `mimi.decode_step` → cpal playback) |
| [`model.md`](model.md) | **not ported** | `demo/model.py` | singleton-loader / CUDA warmup module — the Rust tree builds its singletons in `loader.rs`/`mic_chat.rs` instead |

The key Rust-specific difference: the Python demo hard-codes `device="cuda"`
and won't boot CPU-only; the Rust `mic_chat.rs` is device-agnostic (CPU/f32 or
Metal/bf16) and has no warmup loop (candle has no CUDA-graph capture /
`torch.compile` to pre-pay). See [`../README.md`](../README.md) for the
recurring divergences. See `wiki/demo/` for the Python-first companions.