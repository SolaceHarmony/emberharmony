# Native Runtime Map

The reusable Rust, C++, C, and assembly code lives at the repo root under `crates/`.
The TypeScript/Bun app remains under `packages/`, and Tauri keeps its conventional
Rust entrypoint at `packages/desktop/src-tauri`.

## Crates

- `crates/liquid-audio` — LFM2.5-Audio model engine, runtime pipeline, and CPU native kernels.
- `crates/candle-flashfftconv` — Candle CPU/Metal FlashFFTConv operators used by the voice path.
- `crates/kcoro-sys` — build-only sys crate for the vendored kcoro C/assembly coroutine runtime.

## Boundaries

- Bun/TypeScript calls Tauri commands; it does not import or build native crates directly.
- Tauri depends on `liquid-audio` by path and owns desktop command/session orchestration.
- `liquid-audio` owns model/runtime APIs and private C ABI declarations.
- `kcoro-sys` owns compiling kcoro; it does not expose a safe Rust runtime API.

## Native Source Layout

`crates/liquid-audio/native/` uses C++/assembly-friendly directories:

- `include/` — native ABI headers when a C/C++ interface needs to be shared.
- `src/engine/` — resident native stage-machine code.
- `kernels/aarch64/` — NEON/AArch64 kernels.
- `kernels/x86_64/` — AVX/x86-64 kernels.
- `reference/` — portable or historical reference kernels.

Detailed design notes remain adjacent to the owning crate under `crates/*/docs/`.
