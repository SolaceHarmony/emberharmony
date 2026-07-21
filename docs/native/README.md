# Native Runtime Map

The production voice path is split between a thin Rust host and a native
LFM2-Audio runtime. Rust owns platform callbacks, application control, and UI
projection. Native C++/kcoro/Flashkern owns checkpoint loading, model state,
turn policy, scheduling, PCM storage, and inference. The production
`liquid-audio` dependency graph is Candle-free.

## Crates

- `crates/liquid-audio` — production LFM2-Audio host API plus the native
  C++/assembly runtime. Its Rust code sees opaque lifecycle handles, typed
  control/events, and borrowed PCM spans; it does not load weights or execute
  model math.
- `crates/kcoro-sys` — production Rust bindings and build integration for the
  vendored `kcoro_arena` runtime. The vendored C runtime owns retained
  services, realtime notifier edges, fixed teams, fixed scopes, doorbells, and
  correlated monotonic deadlines.
- `crates/liquid-audio-oracle` — workspace-only Candle reference, training, and
  fixture-capture code. It is not a production fallback.
- `crates/candle-flashfftconv` — workspace-only experimental Candle/Metal
  kernels. The desktop does not depend on it.

Rust docking uses the production wrappers in `kcoro-sys`; model-pass scheduling
remains native.

## Ownership Boundaries

- Bun/TypeScript calls Tauri commands and receives bounded observations. It
  does not load a model or carry numerical payloads.
- Tauri owns settings, platform stream lifetime, opaque native handles, and UI
  event projection.
- A hardware capture callback writes directly into a preallocated native PCM
  lease and publishes a typed chunk record. A playback callback resolves and
  drains a native playback lease. PCM is not transferred through stdout,
  files, framework tensors, or a Rust-owned model buffer.
- The native session owns the exact Sesame microphone detector and sample-clock
  turn policy, route/recurrence state, native capture/playback storage, and
  reliable ticket-correlated delivery.
- `safetensors.cpp` loads the immutable resident checkpoint image. Bound model
  values are pointer/shape/stride views into that image; dtype, alignment, or
  layout copies are forbidden compatibility materialization.
- Flashkern currently has one fixed `kc_team` per engine. A two-`BlockDomain`
  grid remains proposed work; documentation must not describe it as landed.

## Native Source Layout

`crates/liquid-audio/native/` uses C++/assembly-friendly directories:

- `include/` — versioned product ABI plus explicitly private diagnostic and
  test interfaces.
- `src/io/` — native safetensors image loading and metadata validation.
- `src/runtime/` — native sessions, PCM docks, Sesame policy, and bridge
  records.
- `src/engine/` — Flashkern route/pass state and fixed-team dispatch.
- `src/model/`, `src/frontend/`, and `src/mimi/` — immutable plans, byte views,
  activation arenas, and native model programs.
- `kernels/aarch64/` and `kernels/x86_64/` — paired architecture math leaves.
- `reference/` and `native/tests/fixtures/` — test-only evidence excluded from
  the product link graph.

## Current Supervision Truth

Each Flashkern team generation is guarded by a correlated hard deadline. The
team records generation-stamped member entry and return state; completion and
expiry race through one terminal decision. A hard expiry captures a reserved
fatal capsule and aborts rather than recycling possibly live state.

Two release gates remain open:

- the fatal capsule is not yet exported to a durable, observable crash sink;
- the one-second hard budget is provisional rather than calibrated per
  stage/shape from the required target benchmark.

The native Sesame implementation contains independent microphone and playback
state, but the production session currently feeds only microphone evidence.
Playback-aware Sesame/echo classification remains open; Rust RMS is telemetry,
not turn detection.

## Integration Guides

- [`KCORO_ARENA_INTEGRATION.md`](KCORO_ARENA_INTEGRATION.md) — live ownership,
  callback, ticket, PCM, deadline, and teardown contracts.
- [`10-stateful-multi-agent-runtime.md`](../../specs/10-stateful-multi-agent-runtime.md) —
  future multi-conversation scheduling and persistence design.

## Performance Evidence

- [`G0_FENCE_SPIN_321538F1.md`](baselines/G0_FENCE_SPIN_321538F1.md) — historical
  spin-era baseline.
- [`G3_SHARED_DOORBELLS_D2C43ABD.md`](baselines/G3_SHARED_DOORBELLS_D2C43ABD.md) —
  shared-doorbell wake accounting and idle behavior.
