# Native Runtime Map

The reusable Rust, C++, C, and assembly code lives at the repo root under `crates/`.
The TypeScript/Bun app remains under `packages/`, and Tauri keeps its conventional
Rust entrypoint at `packages/desktop/src-tauri`.

## Crates

- `crates/liquid-audio` - current hybrid LFM2.5-Audio crate; target is a Rust
  PCM/control host over non-numerical C++ control and assembly-only inference.
- `crates/candle-flashfftconv` - migration-only Candle operators, deleted when
  the native production kernel gates pass.
- `crates/kcoro` - dependency-free safe Rust coordination kernel for PCM/control
  docking and future Tauri-side asynchronous work. It does not broker Flashkern
  model passes.
- `crates/kcoro-sys` - build-only sys crate for the vendored C conformance
  oracle and expected-value host waits. Its C ticket scheduler is tested but is
  not on Flashkern's production pass path.

## Boundaries

- Bun/TypeScript calls Tauri commands; it does not import or build native crates directly.
- Tauri depends on `liquid-audio` by path and owns persisted settings, command
  registration, opaque handle lifetime, and bounded event projection.
- `liquid-audio` currently owns the hybrid Rust/Candle migration rim and private
  C ABI. C++ owns the resident checkpoint image, native SQ/CQ, descriptors, and
  fixed lane control; paired `.S` files increasingly own the numerical bodies.
- `crates/kcoro` owns Rust PCM/control scopes and tickets. Model recurrence and
  model pass tickets stay native.
- `kcoro-sys` compiles the C oracle and platform wait adapter. It is not the
  product policy scheduler.

## Native Source Layout

`crates/liquid-audio/native/` uses C++/assembly-friendly directories:

- `include/` — native ABI headers when a C/C++ interface needs to be shared.
- `src/engine/` — resident non-numerical stage, queue, and barrier control.
- `kernels/aarch64/` — AArch64/NEON assembly math.
- `kernels/x86_64/` — x86-64 assembly math.
- `reference/` - test-only scalar oracles excluded from the production link map;
  no historical production implementation is retained.

Detailed design notes remain adjacent to the owning crate under `crates/*/docs/`.

## Integration Guides

- [`KCORO_ARENA_INTEGRATION.md`](KCORO_ARENA_INTEGRATION.md) - current and target
  architecture, the source-exact mounted pass sequence, memory ownership, fixed
  Flashkern lanes, wait exceptions, tickets/callbacks, host adapters, Tauri
  observation, durable workflows, shutdown ordering, and verification gates.
- [`10-stateful-multi-agent-runtime.md`](../../specs/10-stateful-multi-agent-runtime.md) -
  one shared model with many conversation images, fast switching, perspective
  forks, macro batching, delta hibernation, and WAL-backed orchestration.

## Performance Evidence

- [`G0_FENCE_SPIN_321538F1.md`](baselines/G0_FENCE_SPIN_321538F1.md) - frozen
  spin-era latency, wake, allocation, and raw percentile baseline.
- [`G3_SHARED_DOORBELLS_D2C43ABD.md`](baselines/G3_SHARED_DOORBELLS_D2C43ABD.md) -
  committed shared-doorbell contract, exact wake accounting, zero-spin idle
  behavior, and the G0 percentile comparison.
