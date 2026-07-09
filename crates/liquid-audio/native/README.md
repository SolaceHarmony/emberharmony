# liquid-audio Native Sources

- `include/` — shared native ABI headers.
- `src/engine/` — resident native stage-machine implementation.
- `kernels/aarch64/` — NEON/AArch64 kernels.
- `kernels/x86_64/` — AVX/x86-64 kernels.
- `reference/` — reference or fallback kernels.

Cargo builds these sources through `../build.rs`; symbol names are kept stable for
the Rust FFI layer in `src/compute/flashkern`.
