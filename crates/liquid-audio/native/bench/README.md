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

## `pipeline_probe.c` — the scheduling spine (fixed team + kc_port doorbell)

Moves the question up a level: not "which leaf is fastest" but "does a
flashkern-style fixed lane team, parked on the real in-repo `kc_port` doorbell,
scale those leaves across the M2's cores with low, zero-spin overhead?" It links
`kc_port` (`port/posix.c`) directly — not `kc_team`, which is mid-migration — so
the dispatch/park/wake path is honest. Work is a decode projection fanned out by
an atomic tile-claim (flashkern's shared-counter pattern); leaves are BFDOT.
Workers set `QOS_CLASS_USER_INTERACTIVE` to bias onto P-cores.

### Measured on this M2 Max

**Decode layer, N=2048 K=2048, 8 MiB BF16 weights, 64-row tiles:**

| workers | ms/gen | GB/s | speedup |
|---:|---:|---:|---:|
| 1 | 0.206 | 40.7 | 1.00× |
| 2 | 0.115 | 72.7 | 1.78× |
| 4 | 0.071 | 118.7 | 2.91× |
| 8 | 0.058 | 145.1 | 3.56× |

**Orchestration overhead (tiny 1-tile work isolates the doorbell round-trip):**

| workers | µs/gen |
|---:|---:|
| 1 | 2.59 |
| 4 | 11.51 |
| 8 | 23.62 |

### What it establishes

1. **The doorbell spine adds no tax to the leaf.** One worker hits 40.7 GB/s —
   the standalone BFDOT number. `kc_port` park/wake is genuinely zero-spin and
   free at the compute level.
2. **Scaling is weight-bandwidth-capped:** 8 workers give **3.56×, not 8×**, and
   4→8 buys only +0.65×. The decode projection is bandwidth-bound, so neither
   more NEON cores nor AMX escapes the ceiling (both read the same weights from
   the same memory). The only lever past it is **weight reuse — batch at the
   barrier** (read W once, serve M tokens), which raises arithmetic intensity.
3. **The per-generation wake is ~3 µs/worker (~24 µs for 8).** Against a
   bandwidth-bound decode layer (~34 µs of compute at 8 workers) that is ~40%
   overhead — so **per-layer full-team dispatch is wasteful for small work.**

### Design consequence

The fastest path is not "max workers per layer." It is: a **resident team with
coarse generations** (dispatch a whole decode *step*, not one layer, so the ~24 µs
wake amortizes — this is why the fence belongs at pass boundaries, not per op);
**batching for reuse, not cores for parallelism** (the 3.56× ceiling says added
cores are nearly spent by 4–8); and for a single bandwidth-bound layer, ~4 workers
is the efficiency knee (2.91× at 16% overhead vs 3.56× at 40%).

It does **not** yet measure the batched (weight-reuse) path, `kc_team`'s
completion-callback edge, cross-conversation contention, or a heterogeneous
AMX+NEON schedule — those are the next probes.
