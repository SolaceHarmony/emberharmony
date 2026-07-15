# Native Runtime Map

The reusable Rust, C++, C, and assembly code lives at the repo root under `crates/`.
The TypeScript/Bun app remains under `packages/`, and Tauri keeps its conventional
Rust entrypoint at `packages/desktop/src-tauri`.

## Crates

- `crates/liquid-audio` - current hybrid LFM2.5-Audio crate; target is a thin
  Rust handle wrapper over the complete C++/SIMD/assembly local voice runtime.
- `crates/candle-flashfftconv` - migration-only Candle operators, deleted when
  the native production kernel gates pass.
- `crates/kcoro` - dependency-free safe Rust coordination kernel. Its first
  production mount owns the sole Flashkern SQ broker future and exact CQ wake
  edge; scopes, child tickets, recurrence, and service policy remain open.
- `crates/kcoro-sys` - build-only sys crate for the vendored C conformance
  oracle and expected-value host waits. Its C ticket scheduler is tested but is
  not on Flashkern's production pass path.

## Boundaries

- Bun/TypeScript calls Tauri commands; it does not import or build native crates directly.
- Tauri depends on `liquid-audio` by path and owns persisted settings, command
  registration, opaque handle lifetime, and bounded event projection.
- `liquid-audio` currently owns the hybrid Rust/Candle model rim, the safe Rust
  coordinator endpoint, and the private C ABI. C++ owns the resident checkpoint
  image, native SQ/CQ and descriptors, fixed numerical lanes, and SIMD/assembly
  dispatch.
- `crates/kcoro` owns product coordination. At the current mount it routes one
  synchronous borrowed-pointer pass; at cutover it owns scopes, tickets, and
  recurrence without owning model math.
- `kcoro-sys` compiles the C oracle and platform wait adapter. It is not the
  product policy scheduler.

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
