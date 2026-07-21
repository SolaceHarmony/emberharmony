# liquid-audio

Native LFM2.5-Audio engine used by the desktop voice stack. Native
C++/kcoro/Flashkern and architecture kernels own the model image, inference,
turn policy, CoreAudio callbacks, and PCM docks. Rust exposes opaque lifecycle,
control, and bounded observation only. Candle lives in the separate offline
`liquid-audio-oracle` crate and is not linked into this production crate.

- Detailed architecture docs: `docs/`.
- Native C/C++ sources: `native/`.
- Public Rust API: `src/lib.rs`.
- Tauri integration: `packages/desktop/src-tauri` depends on this crate by path.

Run from the repo root:

```sh
cargo test -p liquid-audio --lib -- --nocapture
cargo test -p liquid-audio --tests -- --test-threads=1
```

The complete native two-agent, memory-only speech gate is explicit because it
opens the real checkpoint:

```sh
LFM_MODEL_DIR=/absolute/LFM2.5-Audio-1.5B \
  cargo test -p liquid-audio --test native_speech_to_speech \
  -- --ignored --nocapture --test-threads=1
```

Set `LFM_SPEECH_GATE_AUDIBLE=1` to prebuffer the first deterministic exchange
and play it through the default CoreAudio speaker. Use
`LFM_SPEECH_GATE_AUDIBLE=stream` to exercise live generation cadence and report
hardware-buffer underruns. Both monitors are native, bounded, callback driven,
and read the same in-memory PCM blocks without creating an audio file.

On Apple Silicon, cross-build the Darwin x86_64 suites and let macOS execute
the test binaries through Rosetta directly:

```sh
cargo test -p kcoro-sys -p liquid-audio \
  --target x86_64-apple-darwin -- --test-threads=1
```

Feature-gated SIMD tests skip when Rosetta does not expose the required ISA;
x86 compilation, linking, scalar assembly ABI, scheduler, and dispatch checks
still run. Actual AVX2/AVX-512 instruction correctness requires a native x86
runner that advertises those features.
