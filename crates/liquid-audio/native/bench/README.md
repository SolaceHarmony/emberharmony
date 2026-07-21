# native/bench

Standalone microbenchmarks and experiments for the numerical kernels. These are
**not** built by `../../build.rs` and are not part of the product; they exist to
settle bare-metal design questions with measurement instead of argument. Each is
a single self-contained source with its build line in the header comment.

## `amx_vs_neon.c` — register-resident NEON vs Accelerate/AMX

Answers: for a flashkern linear, where does hand-written register-resident NEON
win and where should the work be handed to Accelerate/AMX?

```
clang -O3 -ffp-contract=off amx_vs_neon.c -framework Accelerate -o amx_vs_neon
./amx_vs_neon
```

### Measured on Apple M2 Max (128 KB L1d, 16 MB L2/cluster, 128-byte lines)

**GEMV, M=1 (the decode regime), `y = W·x`:**

| kernel | 16 MB W (fits L2) | 64 MB W (DRAM-bound) |
|---|---:|---:|
| scalar | 5.0 GB/s | ~4.7 |
| NEON ×1 (serial FMA) | 14.9 | ~14 |
| **NEON ×4 (independent accumulators)** | **51.2** | **~50** |
| NEON ×4 + fused BF16 RNE | 51.2 | ~50 |
| Accelerate/AMX + BF16 epilogue | 112.8 | ~48–57 |

**GEMM (the prefill regime), `Y = X·W`, N=K=2048, Accelerate/AMX.** Run
`VECLIB_MAXIMUM_THREADS=1` for the fair single-lane figure — Accelerate spawns
its own threads, which both inflates the number and violates "kcoro owns every
computation thread":

| M | single-lane (`VECLIB_MAXIMUM_THREADS=1`) | multi-thread (Accelerate default) |
|---:|---:|---:|
| 1 | 43 | 48 |
| 16 | 227 | 240 |
| 64 | **631** | 1159 |
| 256 | **990** | 1360 |

### What it shows

1. **The seam is L2-residency, not batch alone.** When weights fit L2, AMX is
   ~2.2× the NEON path (118 vs 54 GB/s single-lane). When weights exceed L2 both
   converge on the ~52 GB/s single-core DRAM ceiling — at K=8192 it is a dead tie
   (52.2 vs 52.3). So at DRAM-bound decode the throughput is a **wash**; NEON's
   case is **ownership + a fused epilogue (no round-trip)**, not raw speed. A real
   multi-hundred-MB decode is far past L2, so **decode belongs to register-resident
   NEON**; prefill (M large, compute-bound, ~15× the GEMV ceiling *single-lane*)
   **belongs to AMX**. Note the multi-thread column: ~1.8× of Accelerate's M=64
   lead is hidden threads — reclaimable only if kcoro itself dispatches AMX across
   lanes, not by calling cblas.
2. **Independent accumulators are the ILP win** — ×4 is 3.4× over ×1 (51 vs 15
   GB/s). The renamer already handles WAW/WAR; breaking the RAW FMA chain is the
   lever.
3. **The fused BF16 epilogue is free** (51.2 = 51.2 GB/s, one write); the AMX path
   must write a plane and read it back to round. Eliminate the plane, not the
   precision.
4. **Reordering the reduction perturbs the result** — ×4 vs pinned-sequential
   diverged 7.8e-7 at K=8192 (0 at shorter K). BF16 absorbed it here, but that is
   magnitude-luck, not a guarantee: keep the reduction order pinned where the
   model's numerics are validated.

### Initial design consequence (qualified by the direct-BF16 probe below)

Register-resident fused NEON for the DRAM-bound decode path (MLP down-proj,
ShortConv carry, Depthformer norm+RoPE, decode-attention tiles); hand the batched
prefill GEMM to Accelerate/AMX. AMX stays a documented-interface call, never a
hand-emitted raw-instruction path. The second probe below tests that last claim
directly and finds a useful raw-AMX mechanism, but not yet an admissible product
replacement under both the layout and numerical invariants.

## `amx_bf16_hybrid.cpp` — checkpoint-native BF16 hybrid ground truth

This second probe removes the assumptions in `amx_vs_neon.c` that do not hold in
production. It calls the real checkpoint-layout leaf with the real contract:

```
A: bf16 [M,K]
W: immutable bf16 [N,K]
C: f32 [M,N] = A * W^T
```

Both source pointers are deliberately offset by two bytes. There is no F32
weight mirror, `KxN` view, packed panel, or timed allocation. The experiment now
separates backend choice from numerical schedule:

- `NEON FMA faithful`: the production leaf, with two widened-F32 vector
  dependency chains per row and its pinned exact-8 reduction.
- `NEON FMA fast`: the same ordinary FMA instruction with four register-resident
  vector accumulators per row through `M=4` and two above it.
- `NEON BFDOT RO/FTZ`: direct raw-layout hardware BF16 dot. This is tuned and
  spill-free, but M2 BFDOT uses pair-dot round-to-odd/flush-to-zero semantics;
  it is a distinct numerical contract, not merely another reduction tree.
- `AMX fast32`: 32 independent F32 partials through M2 `VECFP`, including
  `AMX_SET/CLR`, a complete split plane, and a distinct noinline NEON consumer.
- `AMX exact8`: four eight-lane AMX steps reproduce the production leaf's two
  F32x4 accumulator order exactly.
- `tile-local`: STZI then NEON-reduce every output before returning to AMX.
- `NEON reduce32 hot`: a diagnostic measured only after explicitly rebuilding
  a fast32 plane. It is outside the randomized contestant order.

The split-plane paths measure the proposed AMX-to-scratch-to-assembly memory
boundary, but not a kcoro wake.

Build and run:

```
clang++ -O3 -std=c++23 -ffp-contract=off -mcpu=apple-m2 \
  amx_bf16_hybrid.cpp ../kernels/aarch64/flashkern_neon.cpp \
  -I../include -framework Accelerate -o /tmp/amx_bf16_hybrid
/tmp/amx_bf16_hybrid
/tmp/amx_bf16_hybrid --raw   # include all 17 randomized samples per case
```

`-mcpu=apple-m2` is the Apple-Clang target-tuning flag; its effective feature
list explicitly contains `+neon`, `+bf16`, `+i8mm`, `+dotprod`, and the M2/A15
scheduling model. NEON/Advanced SIMD is mandatory in AArch64, so there is no
separate `-mneon` switch. The shipping-parity build was also tested with the
repository's exact kernel flags:

```
clang++ -O3 -std=c++23 -ffp-contract=off -march=armv8.3-a+bf16+i8mm \
  amx_bf16_hybrid.cpp ../kernels/aarch64/flashkern_neon.cpp \
  -I../include -framework Accelerate -o /tmp/amx_bf16_hybrid_shipping
```

Apple Clang reports `+neon` for that target as well. Both configurations use
explicit NEON intrinsics in the product leaf and consumer; `-O3` already enables
the loop and SLP vectorizers.

### Measured ground truth on this M2 Max

Ranges below are medians from three fresh processes on 2026-07-18 using
`-mcpu=apple-m2`; every median contains 17 randomized candidate orders. Every
numerical path uses one execution thread. This is a leaf comparison, not the
eight-lane engine latency. Three more fresh processes using the repository's
shipping flags produced the same ranking.

| production shape | FMA faithful | FMA fast | BFDOT RO/FTZ | AMX fast32 | AMX exact8 | reduce32 hot |
|---|---:|---:|---:|---:|---:|---:|
| `M1 N2048 K2048` | 0.641–0.688 ms | 0.307–0.323 ms | 0.209–0.224 ms | **0.183–0.192 ms** | 0.667–0.744 ms | 0.003 ms |
| `M4 N8192 K2048` | 2.969–3.035 ms | 2.876–2.973 ms | 2.728–2.765 ms | **1.504–1.529 ms** | 5.542–5.609 ms | 0.052–0.068 ms |
| `M4 N2048 K8192` | 2.989–3.122 ms | 2.942–3.091 ms | 2.799–2.922 ms | **1.482–1.524 ms** | 5.462–6.122 ms | 0.014–0.017 ms |
| `M7 N2048 K512` | 0.362–0.376 ms | 0.299–0.309 ms | 0.299–0.307 ms | **0.131–0.152 ms** | 0.347–0.374 ms | 0.021–0.023 ms |

This closes the missing-cell objection. The production decode leaf is
dependency-limited: ordinary four-chain FMA is about 2.1x faster, and BFDOT
closes most of the former AMX gap. AMX fast32 is only about 10–20% faster than
BFDOT at `M=1`. That correction does not erase the multirow result: at `M=4`
and `M=7`, the spill-free BFDOT path is only modestly faster than ordinary NEON,
while AMX fast32 remains about 1.8–2.3x faster.

The split boundary is not the slowdown. In fact, phase separation is faster
than alternating AMX and NEON at every output: the fast32 tile-local medians
were about 0.29, 3.0, 1.8, and 0.8 ms respectively. Keeping the coprocessor
phase contiguous matters more than the scratch-plane traffic at these sizes.

The fast results are **not numerically substitutable yet**. The original
cancellation vector proves that changing accumulator partitions can survive an
isolated BF16 rounding:

```
FMA faithful = 0          -> bf16 0x0000
FMA fast = 0.000122070    -> bf16 0x3900
BFDOT/AMX = 0.000183105   -> bf16 0x3940
AMX exact8 = 0            -> bf16 0x0000
```

More importantly, BFDOT is not the NEON equivalent of AMX fast32. An adjacent
pair adversary produced:

```
faithful/FMA-fast/AMX/exact8 = 0.0000610351562 -> bf16 0x3880
BFDOT                         = 0.00390625      -> bf16 0x3b80
```

That is BFDOT's pair-dot round-to-odd behavior made visible after cancellation;
M2 also flushes subnormals on this instruction. `exact8` remains bit-identical
to the product leaf over every measured output and both relevant exactness
checks, but removes the AMX speedup. The printed `bf16-diff` rounds the isolated
dot only; actual product admission still requires the real fused epilogue and
full hidden/logit/PCM parity.

### The MATFP layout result

M2 MATFP is a 32x32 outer product. At reduction step `k`, it needs 32 output
weights adjacent in memory. In checkpoint `W[N,K]`, those words are K-strided.
`LDX/LDY` only load contiguous 64-byte spans, indexed MATFP only gathers lanes
already resident in an AMX register, and NEON cannot transfer registers directly
to AMX. A C++ "view" can change indexes but cannot change physical adjacency.

Consequently, a full-strength MATFP linear requires a transposed or packed
weight panel. Apple's AMX BLIS kernel has exactly that split: a separate pack
kernel and a hot loop consuming `[k][32]` panels. That is incompatible with the
current `compatibility_copied_bytes == 0` invariant. The hybrid output seam is
real; it does not erase the input-layout constraint.

### What this does and does not establish

- It establishes that direct BF16 AMX, unaligned checkpoint views, SET/CLR, STZI
  output, and a later NEON continuation all work.
- It establishes that Apple Clang emits real `BFDOT` instructions and keeps the
  tuned FMA/BFDOT accumulator sets in registers without vector spills.
- It establishes that extra ordinary-FMA chains explain much of the old decode
  comparison, but not the AMX advantage at the tested multirow shapes.
- It establishes that phase separation can beat per-tile AMX/NEON alternation.
- It establishes that the scratch read is small here relative to the math.
- It does **not** measure kcoro notify/park/resume, the fixed eight-lane product
  path, cold-model streaming, energy, or cross-conversation AMX contention.
- The split prototype writes 32 F32 partials per output (1–4 MiB for the large
  cases), not only the final scalar. An in-AMX exact reduction or a fused
  consumer is still required before this can satisfy the final-write-only goal.

## `monarch_fused_coop.c` — ticketed two-phase FFT without operation waiters

This AArch64 BF16 microbenchmark directly links the same vendored `kc_runtime`,
`kc_service`, and `kc_team` implementations used by the production flashkern
engine. It creates a benchmark-local runtime, retained service, team, context,
and storage; it is not dispatched through the engine request bridge or pass
slots, and it does not replace the production FFT.

For row-major input `x[n][l]`, the probe evaluates a columns-first two-factor
forward transform: an N-point DFT down each column, the
`W_(N*L)^(l*k2)` twiddle, then an L-point DFT across each intermediate row.
Its output is the DFT of `x.flatten()` in natural order `K = k1*N + k2`.

The earlier analysis incorrectly called the recovered MLX rows-first formula
an impostor. It is instead the DFT of the factor-ordered input
`x.T.flatten()`, in natural order `K = p*L + q`. The executable now checks both
formulas against their proper O(points²) direct DFT oracles at up to 4096
points, including the unequal `16x32` shape. It does not call or validate the
MLX kernel itself.

Each FFT is one real `KC_TICKET_KIND_PASS` identity whose retained record holds
phase A or B. The retained service dispatches phase A as one team generation
and returns dormant. The final member return copies the exact ticket and phase
to the completion record, release-publishes its generation, and uses a retained
service notifier to make the orchestration record runnable. The service
acquire-consumes that publication: A advances the same ticket to B and
dispatches a second team generation; B settles the ticket and schedules the
next benchmark FFT. The ticket sequence advances once per complete FFT while
the team generation advances twice.

There is no `kc_team_wait`, `kc_collective`, member rendezvous, per-stage
ticket, host-mediated phase transition, polling, sleep, timer, condition
variable, or waiter thread. The main thread enters only terminal teardown
joins; the team join can return only after the retained service publishes the
stop edge, and teardown never advances a numerical phase.

The two phases explicitly materialize the BF16 complex intermediate in
ordinary static storage. The boundary publishes those writes; it is not a
physical transpose. The probe has no cache or traffic counters, so it does not
establish L1/L2 residency or the absence of DRAM traffic. BFDOT is Arm's BF16
dot-product operation with FP32 accumulation, not an 8x8 tensor-matrix
operation.

Build from the repository root:

```sh
clang -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror \
  -ffp-contract=off -march=armv8.6-a+bf16 \
  crates/liquid-audio/native/bench/monarch_fused_coop.c \
  crates/kcoro-sys/vendor/kcoro_arena/core/src/kc_runtime.c \
  crates/kcoro-sys/vendor/kcoro_arena/core/src/kc_service.c \
  crates/kcoro-sys/vendor/kcoro_arena/core/src/kcoro_stackless.c \
  crates/kcoro-sys/vendor/kcoro_arena/core/src/kc_team.c \
  crates/kcoro-sys/vendor/kcoro_arena/core/src/kc_doorbell.c \
  crates/kcoro-sys/vendor/kcoro_arena/port/posix.c \
  -Icrates/kcoro-sys/vendor/kcoro_arena/include \
  -Icrates/kcoro-sys/vendor/kcoro_arena/port \
  -pthread -lm -o /tmp/monarch_fused_coop
/tmp/monarch_fused_coop
```

Add `-D_GNU_SOURCE` on Linux. Add `-DMFC_N=16 -DMFC_L=32` or
`-DMFC_N=128 -DMFC_L=128` before the source argument for the other measured
shapes.

### M2 Max measurements, 2026-07-19

| factors | points | columns / rows double residual | BF16 normalized max | 1 lane | 2 lanes | 4 lanes | 8 lanes | 8-lane speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `16x32` | 512 | `7.01e-12 / 4.28e-12` | `1.54e-02` | 0.013 ms | 0.017 ms | 0.025 ms | 0.054 ms | 0.24x |
| `32x32` | 1024 | `3.26e-11 / 1.78e-11` | `2.65e-02` | 0.018 ms | 0.020 ms | 0.026 ms | 0.054 ms | 0.34x |
| `128x128` | 16384 | skipped | `4.85e-02` | 0.558 ms | 0.297 ms | 0.180 ms | 0.137 ms | 4.07x |

Every measured lane count is bit-identical to the direct single-lane leaf.
Timings are the best of six 400-FFT batches after 50 warmup FFTs and remain
scheduler-sensitive. They include both team generations and both retained
service completion edges. Wall clock records telemetry only; elapsed time never
makes the continuation runnable. The small shapes remain orchestration-bound,
while the 16384-point shape reaches 4.07x at eight lanes.

For each lane count the executable completes one correctness FFT, 50 warmups,
and 2400 measured FFTs. It gates all of the following:

- direct columns-first and rows-first convention residuals when points are at
  most 4096, plus the factorized double reference and BF16 error limit at every
  size;
- bit-identical partitioned output for 1, 2, 4, and 8 members;
- exactly one ticket-sequence increment and exactly two team generations per
  completed FFT, ruling out a ticket per phase;
- exact dispatched/completed team generations, final-member completion,
  terminal team join, and an exact phase-B publication for the final ticket;
- one handled retained-service callback for the kickoff and every team
  completion publication, followed by natural service retirement; and
- non-finite timing, ticket, dispatch, service, and teardown failures.

The strict build and all three factor sweeps pass normally. The default
`32x32` sweep also passes under ASan/UBSan and TSan. The probe establishes
forward factorization, exact ticketed phase handoff, and fixed-member partition
equivalence only: there is no inverse, frequency-domain multiply, real packing,
overlap/save, end-to-end convolution, or product-buffer integration.

## `register_cache_chain.cpp` + `.S` — register FIFO ground truth

Tests the question the simpler pointwise cache probe could not answer: does a
real chained computation benefit when live intermediates stay in NEON registers,
and does a deliberately optimized multi-tile FIFO retain enough instruction-
level parallelism to beat phase-separated planes?

The representative chain is RMS normalization → multiplicative gate → causal
ShortConv-3 → four-row projection. It compares:

- one-tile fused assembly, with intermediates kept in registers;
- the same arithmetic forced through 192 bytes of stack scratch;
- rotating external scratch from 4 KiB through 64 MiB;
- a naive per-tile fused batch;
- phase-separated full intermediate planes; and
- a four-tile register FIFO using all 32 NEON registers.

Build and run with the shipping ISA contract:

```
clang++ -O3 -std=c++23 -Wall -Wextra -Wpedantic -Werror \
  -ffp-contract=off -march=armv8.3-a+bf16+i8mm \
  register_cache_chain.cpp register_cache_chain_aarch64.S \
  -o /tmp/register_cache_chain
/tmp/register_cache_chain
```

The four-tile FIFO keeps four activation tiles, four independent reductions,
four projection accumulators, weights, shifts, and convolution state in
`v0-v31`. Its public wrapper saves/restores `d8-d15` once per entire batch (128
bytes total), not once per tile. Disassembly inspection confirms no call, no
per-tile stack access, no intermediate store, and only terminal stores in the
hot loop.

### Measured ground truth on this M2 Max

- Immediate register fusion is roughly 9 ns/tile. Forced stack or arena
  materialization is roughly 11–12 ns/tile: materialization adds about 16–30%
  in this small chain.
- Naive one-tile fusion is roughly 8.0–8.4 ns/tile across the delayed-reuse
  sweep. The optimized four-tile FIFO is roughly 6.4–6.7 ns/tile, a repeatable
  **20–24%** improvement.
- Full planes can tie or narrowly win at the very smallest footprints. From
  roughly 120 KiB of live plane storage onward, the register FIFO wins in the
  measured sweep; at 64–128 MiB it is about 20% faster than naive fusion and
  19–24% faster than planes.
- All variants produce bit-identical terminal spans across 257 randomized
  immediate cases and 26 delayed footprints. Guard regions remain intact.

The lesson is not merely “fuse.” Naive fusion can lose the instruction-level
parallelism that phase separation exposes. Rotate several independent tiles
through register roles so the out-of-order core can overlap dependency chains,
then write only terminal values.

Stack and arena storage are ordinary cache-backed virtual memory. The printed
live bytes, address span, and reuse distance do not identify L1, L2, or DRAM;
that requires hardware counters unavailable to this probe.

## `down_projection_chain.cpp` + `.S` — production-chain proof

Applies the register/cache idea to the current `ST_DOWN` numerical contract,
not a synthetic affine loop:

```
checkpoint BF16 GEMV (pinned exact-two-accumulator order)
  → integer BF16 RNE
  → residual BF16 add
  → integer BF16 RNE
  → terminal BF16
```

The comparison includes the production three-leaf chain with stack scratch, the
same leaves with a preallocated arena, a one-row final-store-only fused leaf,
and a four-row register FIFO. Views are deliberately unaligned. No path widens,
packs, transposes, or copies the immutable BF16 weight image.

```
clang++ -O3 -std=c++23 -Wall -Wextra -Wpedantic -Werror \
  -ffp-contract=off -march=armv8.3-a+bf16+i8mm \
  down_projection_chain.cpp down_projection_chain_aarch64.S \
  ../kernels/aarch64/flashkern_neon.cpp -I../include \
  -o /tmp/down_projection_chain
/tmp/down_projection_chain
```

### Measured ground truth on this M2 Max

| shape | three leaves | fused one row | fused four-row FIFO | activation reads |
|---|---:|---:|---:|---:|
| `N=256 K=8192` | 0.28–0.32 ms | 0.28–0.31 ms | **0.086–0.104 ms** | 4 MiB → 1 MiB |
| `N=2048 K=8192` | 2.28–2.58 ms | 2.22–2.51 ms | **0.75–0.87 ms** | 32 MiB → 8 MiB |

Deleting the small intermediate plane by itself buys only about 3–5% here. The
roughly **3×** result comes from processing four output rows together so one
activation load feeds four immutable weight rows, while independent exact-order
accumulators expose ILP. Weight bytes are unchanged; activation reads fall 4×;
only terminal BF16 is written. Every terminal word is bit-exact against the
production chain, including non-multiple tails, and all canaries pass.

That is the production rule this probe supports: keep each tile's live chain in
registers, reuse resident input values across several independent rows, and
materialize only at a true fan-out, cross-lane publication, or terminal output.
