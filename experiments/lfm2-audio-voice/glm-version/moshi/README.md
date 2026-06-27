# Kyutai Moshi stack — Rust port (reused via the `moshi` crate)

> Companion to [`ARCH/moshi/README.md`](../../ARCH/moshi/README.md). The
> original documents the **vendored Python** `liquid_audio/moshi/**`; this
> documents the **Rust port's** relationship to that code.

## Summary
The Rust port **does not re-port** the vendored `liquid_audio/moshi/**`. It
**reuses Kyutai's published `moshi` crate** (`moshi::mimi::Mimi`), chosen
specifically because that crate matches the LFM2-Audio checkpoint's
`rvq_first`/`rvq_rest` weight naming — candle-transformers' Mimi (0.9 and 0.10)
uses the Encodec-style `encoder.layers.N`/weight-norm layout and **cannot load
this checkpoint** (PYTHON_VS_RUST.md §2.3).

The `moshi` crate pins `candle ^0.9.1` (= our 0.9.2). It is consumed in-tree via
`liquid-audio-rs/src/audio_out.rs::MimiDetokenizer`, a thin adapter wrapping
`RefCell<Mimi>` behind the `AudioDetokenizer` trait. The processor
(`processor.rs`) dispatches `decode` through `Box<dyn AudioDetokenizer>` and
never touches a concrete codec type.

## On-path vs off-path (same as the Python)
For LFM2-Audio only **one thing in the moshi tree is on the inference path: the
Mimi neural audio codec** — the decode direction (codes → waveform). The
encode direction (waveform → codes) runs only at training-data prep via
`data/mapper.rs`. The rest of the moshi subtree — the Moshi 7B multi-stream LM,
the asyncio WebSocket transport, the conditioners, and the clients — is **a
different model / off-path reference only**, and is **not ported** to Rust.

## What the Rust port actually has
- `liquid-audio-rs/src/audio_out.rs` — the `AudioDetokenizer` trait + the
  `MimiDetokenizer` adapter over `moshi::mimi::Mimi`. This is the **only**
  in-tree Rust code for the moshi codec. It exposes `decode`, `decode_step`
  (streaming), `encode` (for training-data prep), and `reset_stream`.
- `liquid-audio-rs/Cargo.toml` — the `moshi` crate dependency.

## What the Rust port does NOT have
- No in-tree port of `compression.py`, `vq.py`, `core_vq.py`, `seanet.py`,
  `resample.py`, `transformer.py`, `loaders.py`. These live inside the `moshi`
  crate (Kyutai's own Rust port of the same Python). The Rust port trusts that
  crate's fidelity to the upstream — it is Kyutai's *own* port, not a
  third-party reimplementation.
- No port of `lm.py` (Moshi 7B `LMModel`/`LMGen`), `server.py` (asyncio/ws
  transport), the conditioners, the TTS, the clients, `run_inference`,
  `run_tts`. These are off-path for LFM2-Audio and deliberately not ported.

## Per-file status
Because the moshi tree is reused rather than re-ported, the Rust-focused docs
for each Python file are a single-line status. See
[`glm-version/moshi/STATUS.md`](STATUS.md) for the table.

## Deliberate divergences (PYTHON_VS_RUST.md §2.3)
- **Codec reuse, not re-port.** The Python `MimiModel`/`CompressionModel`/
  `WrapperCompressionModel` orchestration is *not* mapped symbol-for-symbol —
  the `AudioDetokenizer` trait (`audio_out.rs`) is the design seam instead.
- **Device-agnostic.** Python defaults `device="cuda"`; Rust takes
  `device: &Device` (Cpu/F32 default, Metal/bf16 opt-in).
- **No CUDA graphs.** The `CUDAGraphed` wrapping in
  `_init_streaming_state` is GPU-only and absent in Rust (candle eager);
  numerically irrelevant, latency-only.
- **SDPA, not flash.** The enc/dec transformers' `F.scaled_dot_product_attention`
  maps to eager matmul+mask+softmax in the `moshi` crate.
- **Codes cast to U32.** Rust casts codes to `u32` before `Mimi::decode`/
  `decode_step` because the residual-VQ codebook lookup is an `index_select`
  (`audio_out.rs:89,114`); Python keeps them as torch int.

## Precision / gotchas (Rust-specific)
- **`moshi::mimi` weight-key compatibility is the reason for the reuse.** The
  `moshi` crate's `quantizer.rvq_first.*`/`rvq_rest.*` names match this
  checkpoint; candle-transformers' Mimi does not. Don't swap to
  candle-transformers' Mimi without verifying the weight keys.
- **Mimi decode is smoke-validated, not byte-exact.** `mimi_decode_smoke`
  (waveform `[1,1,30720]`, peak 0.7395) — the moshi-crate reuse + candle
  gemm/FFT ordering sit at the ~1e-6 cross-framework floor. Not a faithfulness
  defect.
- **EOAudio = 2048 is a model-side sentinel.** Mimi itself only ever
  emits/consumes codes `0..2047`. The processor rejects codes `>= 2048` before
  decode (`processor.rs:147-151`).
- **`frame_size = 1920`** (one code column = 1920 samples = 80 ms at 24 kHz).
  Streaming `decode_step` rejects non-multiple-of-1920 lengths.

## Cross-references
- [`ARCH/moshi/README.md`](../../ARCH/moshi/README.md) — the vendored Python
  overview (component wiring diagram + per-file specs).
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.3 (codec reuse), §2.10 (the reverted
  f64 detour — does not apply to Mimi).
- `liquid-audio-rs/src/audio_out.rs` — the `AudioDetokenizer` trait +
  `MimiDetokenizer` adapter.