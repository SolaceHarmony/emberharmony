# liquid-audio

Native LFM2.5-Audio engine and transitional Rust host rim used by the desktop
voice stack. C++ and architecture kernels own the inference substrate; the
remaining Candle compatibility paths are migration work, not the target design.

- Detailed architecture docs: `docs/`.
- Native C/C++ sources: `native/`.
- Public Rust API: `src/lib.rs`.
- Tauri integration: `packages/desktop/src-tauri` depends on this crate by path.

Run from the repo root:

```sh
cargo test -p liquid-audio --lib -- --nocapture
cargo build -p liquid-audio --all-targets
```

On Apple Silicon, the local-only Rosetta lane cross-builds Darwin x86_64 and
runs the resulting tests explicitly through Rosetta:

```sh
./crates/liquid-audio/scripts/test-rosetta.sh
```

The script reports whether Rosetta exposes AVX2. When it does not, feature-gated
SIMD tests skip; x86 compilation, linking, ABI, scheduler, and dispatch checks
still run. Actual AVX2/AVX-512 instruction correctness requires an x86 runner
that advertises those features.
