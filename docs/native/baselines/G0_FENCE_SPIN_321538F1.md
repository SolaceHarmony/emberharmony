# G0 FENCE_SPIN Baseline

Status: reproduced from committed source on 2026-07-13.

## Identity

- EmberHarmony revision: `321538f1`
- Machine: Apple M2 Max, 12 logical CPUs, 32 GiB RAM
- OS: macOS 26.6 (`25G5028f`)
- Rust: `rustc 1.96.0 (ac68faa20)`, LLVM 22.1.2
- Architecture: `aarch64-apple-darwin`
- Shape: BF16 fused MLP, `H=1024`, `I=4096`, 8 lanes

The baseline was built and run in a detached worktree at the named revision.
The source tree under review was not used to compile the old engine.

## Command

```sh
cargo test -p liquid-audio native_engine_mlp_bit_parity -- --nocapture
cargo test --release -p liquid-audio native_engine_mlp_bit_parity -- --nocapture
```

`native_engine_mlp_bit_parity` checks bit parity first, then times 50 warmed
`lfm_engine_mlp` calls and prints their arithmetic mean. The test retains no
individual pass samples, performs no explicit warmup phase, and therefore does
not provide p50, p95, or p99.

## Reproduced Latency

Debug test-profile native-engine means, in milliseconds per pass:

```text
0.373  0.368  0.358  0.333  0.343
```

- minimum: `0.333 ms`
- median: `0.358 ms`
- mean: `0.355 ms`
- maximum: `0.373 ms`

Release test-profile native-engine means:

```text
0.470  0.495  0.376  0.474  0.409
```

- minimum: `0.376 ms`
- median: `0.470 ms`
- mean: `0.445 ms`
- maximum: `0.495 ms`

The previously circulated `0.280 ms/pass` value was not reproduced by this
procedure. Release mode was also noisier and slower during this capture. These
numbers are a source-pinned regression reference, not a statistically isolated
microbenchmark. G3 must retain raw per-pass samples, include warmup, and report
p50/p95/p99 rather than comparing a new p99 to this old 50-pass mean.

## Raw-Sample Supplement

The percentile harness committed later at
`3625df4e5616c1af6af853115c7badceaa338e9e` was applied as a test-only change to
the detached `321538f1` worktree. Production sources remained at G0. The harness
warms both paths for 20 passes, alternates measurement order, retains 1,000
individual pass durations per path, and uses nearest-rank percentiles.

Five G0 runs reported native `p50 / p95 / p99` milliseconds:

```text
0.316 / 0.481 / 0.523
0.335 / 0.612 / 0.745
0.338 / 0.567 / 0.732
0.330 / 0.576 / 0.733
0.326 / 0.630 / 0.727
```

The median across those five run-level percentiles is:

```text
p50 0.330 ms   p95 0.576 ms   p99 0.732 ms
```

The raw harness initially reused the main checkout's `CARGO_TARGET_DIR` to save
compile time. That correctly built and measured G0, but it then left G0 native
archives under paths Cargo reused for the current worktree. The next current
link first lacked `lfm_engine_snapshot`, then lacked the new kcoro ticket/wait
symbols. `cargo clean -p liquid-audio` plus `cargo clean -p kcoro-sys` repaired
the generated state. Different revisions must use different target directories;
sharing one target across baseline worktrees is now a forbidden procedure.

## Wake Topology

At `321538f1`, one MLP pass executes four `run_stage` entry fences plus the
program-final fence:

```text
Rust rim
  -> unpark coordinator                         1 doorbell
  -> coordinator bumps lane generation
     -> unpark lanes 1..7                       7 doorbells
  -> SUMSQ entry fence
  -> NORM entry fence
  -> GATEUP entry fence
  -> DOWN entry fence
  -> final completion fence                     5 fences total
  -> signal blocking rim                        1 condvar signal
```

Each non-last fence arrival spins for up to `FENCE_SPIN = 8192` relax cycles.
If the generation still has not advanced, that lane sets its bit in
`park_mask`, rechecks the generation, and parks. The last arriver exchanges the
mask and unparks only declared waiters. With 8 lanes, the upper bound is 35
fence unparks for this MLP shape; the actual count depends on which lanes cross
each fence during the spin window. Including dispatch, the pass can issue up to
43 kcoro unparks plus the final rim condvar signal.

Relevant committed source:

- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:587`:
  `lane_fence`
- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:605`:
  8,192-iteration spin tier
- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:611`:
  park declaration and recheck
- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:643`:
  four-stage MLP program
- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:988`:
  final pass fence
- `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:1015`:
  coordinator dispatch and completion wake

## Allocation And Copy Contract

- Engine creation allocates the engine, dispatcher, and eight 512 KiB virtual
  coroutine stacks. Untouched stack pages are not resident.
- The first pass at a larger shape may grow persistent scratch vectors outside
  execution. Repeated passes at the resident shape perform no heap allocation.
- Pass fields are written into one engine-owned slot. Inputs, weights, scratch,
  and output remain pointer-referenced; the pass does not copy tensor payloads.
- The four-byte `memcpy` in RMS normalization is a float-to-bits operation, not
  descriptor or tensor staging.
- No allocation or registry lookup occurs inside `lane_fence`; its slow path is
  a stackful kcoro park/unpark.

## Comparison Rule

The ticketed zero-spin executor may replace this baseline only when it preserves
bit parity and allocation freedom, proves exact wake accounting, and publishes a
repeatable raw-sample benchmark. Compare the five-run median of p50/p95/p99 to
the raw supplement above. Keep the original 50-pass mean only for continuity
with earlier reports; never relabel it as p99.
