# Native Runtime Map

The reusable Rust, C++, C, and assembly code lives at the repo root under `crates/`.
The TypeScript/Bun app remains under `packages/`, and Tauri keeps its conventional
Rust entrypoint at `packages/desktop/src-tauri`.

## Crates

- `crates/liquid-audio` - current hybrid LFM2.5-Audio crate; target is a thin
  Rust handle wrapper over the complete C++/SIMD/assembly local voice runtime.
- `crates/candle-flashfftconv` - migration-only Candle operators, deleted when
  the native production kernel gates pass.
- `crates/kcoro-sys` - build-only sys crate for the vendored kcoro coordination,
  ticket, fixed-executor, and host-port runtime; no safe Rust scheduler API.

## Boundaries

- Bun/TypeScript calls Tauri commands; it does not import or build native crates directly.
- Tauri depends on `liquid-audio` by path and owns persisted settings, command
  registration, opaque handle lifetime, and bounded event projection.
- `liquid-audio` owns the private C ABI declarations; its C++ runtime owns model
  loading, session state machines, pass/ticket orchestration, recurrence, and
  numerical dispatch.
- `kcoro-sys` owns compiling kcoro; it does not expose a safe Rust runtime API.

## Native Source Layout

`crates/liquid-audio/native/` uses C++/assembly-friendly directories:

- `include/` — native ABI headers when a C/C++ interface needs to be shared.
- `src/engine/` — resident native stage-machine code.
- `kernels/aarch64/` — NEON/AArch64 kernels.
- `kernels/x86_64/` — AVX/x86-64 kernels.
- `reference/` - test-only scalar oracles excluded from the production link map;
  no historical production implementation is retained.

Detailed design notes remain adjacent to the owning crate under `crates/*/docs/`.

## Integration Guides

- [`KCORO_ARENA_INTEGRATION.md`](KCORO_ARENA_INTEGRATION.md) - current and target
  architecture, memory ownership, fixed Flashkern lanes, zero-spin waits,
  tickets/callbacks, host adapters, Tauri observation, durable workflows,
  shutdown ordering, and verification gates.
- [`10-stateful-multi-agent-runtime.md`](../../specs/10-stateful-multi-agent-runtime.md) -
  one shared model with many conversation images, fast switching, perspective
  forks, macro batching, delta hibernation, and WAL-backed orchestration.

## Performance Evidence

- [`G0_FENCE_SPIN_321538F1.md`](baselines/G0_FENCE_SPIN_321538F1.md) - frozen
  spin-era latency, wake, allocation, and raw percentile baseline.
- [`G3_SHARED_DOORBELLS_D2C43ABD.md`](baselines/G3_SHARED_DOORBELLS_D2C43ABD.md) -
  committed shared-doorbell contract, exact wake accounting, zero-spin idle
  behavior, and the G0 percentile comparison.
