# The AMX / NEON seam and register residency — design synthesis

A single reasoning thread for *where dense BF16 linear work runs* in the native
LFM2 engine, and at what numerical cost. It ties together the two probes in this
directory (`amx_vs_neon.c`, `amx_bf16_hybrid.cpp`) with the register-residency
invariant and the kcoro scheduling substrate. Status is stated honestly at the
end: some cells are measured, some are not.

---

## 1. The invariant we optimize under

> **Register-resident inside one uninterrupted numerical leaf; publish only at a
> true dependency boundary — and stay numerically faithful.**

Concretely:

- Weights are the immutable BF16 checkpoint, laid out `[N, K]` (rows), read in
  place. **No packed panel, widened plane, transpose pool, or aligned copy** —
  `compatibility_copied_bytes == 0` is a hard invariant.
- A leaf keeps its tile in registers across one `.S` body and writes only the
  finished output plus the carry the next call needs.
- Lifetime decides where state lives:

  | Lifetime | Correct home |
  |---|---|
  | within one uninterrupted tile | architectural registers |
  | ABI preservation + a few leaf-locals | small fixed stack frame |
  | across a lane fence | preallocated ticket / lane arena |
  | across route suspension / another token | conversation state |
  | shared across conversations | immutable model image |

- **Fusion eliminates the intermediate *plane*, never the intermediate
  *precision*.** A BF16 checkpoint can stop being a memory write; it cannot stop
  being a BF16 rounding event. Reductions keep their pinned order where the
  model's numerics were validated.

Everything below is the search for the fastest dense-projection kernel that does
not break this invariant.

---

## 2. What is off the table, and why

- **MATFP (the 32×32 AMX outer product) is excluded.** At reduction step `k` it
  needs 32 output weights *physically adjacent* in memory. In `W[N,K]` those
  words are `K`-strided. `LDX/LDY` load only contiguous 64-byte spans; AMX has no
  gather and no NEON↔AMX register move; a C++ "view" can change indices but not
  physical adjacency. A full-strength MATFP linear therefore requires a packed
  panel — forbidden.
- **Weight prepacking is forbidden** for the same reason (it is an owned copy).

So the admissible kernels are:

- **NEON widen** — widen each BF16 to F32 (a shift), F32 FMA, pinned order.
  *Faithful.*
- **NEON BFDOT** — `vbfdotq_f32`, native BF16 dot along contiguous `K`, no widen.
  *Fast; different reduction order.*
- **AMX VECFP** — pointwise BF16 FMA on the raw `[N,K]` layout, unaligned loads,
  `SET/CLR` bracketed, results stored through scratch. Two reduction schedules:
  `fast32` (32 independent partials, *different* order) and `exact8` (reproduces
  the NEON accumulator order *exactly*).

---

## 3. The evidence — a 2×2 matrix of {faithful, fast} × {NEON, AMX}

All numbers are single-thread, single-kernel, on the M2 Max, GB/s counting the
`2·N·K` BF16 checkpoint bytes. Decode is the clean, apples-to-apples cell:

**backbone decode, M=1, N=2048, K=2048**

| kernel | GB/s | faithful? |
|---|---:|---|
| NEON widen, production leaf (2 accumulators) | 11.4 | ✓ bit-exact |
| NEON widen, 4 accumulators | **27.0** | ✓ bit-exact |
| NEON **BFDOT** (fast) | **40.3** | ✗ order differs |
| AMX VECFP **fast32** | 41.5 | ✗ order differs |
| AMX VECFP **exact8** | 11.3 | ✓ bit-exact |

The matrix:

|          | NEON | AMX |
|----------|------|-----|
| **faithful** | widen (baseline) | `exact8` — wash at M=1/M=7, **1.85× slower at M=4** |
| **fast**     | `BFDOT` ≈ 40 GB/s | `fast32` ≈ 41 GB/s |

Three things fall out of it:

1. **Among faithful kernels, NEON wins or ties.** AMX `exact8` is bit-identical
   (including the cancellation gate) but never faster — a wash at M=1/M=7 and
   ~1.85× *slower* at M=4. Faithful AMX buys nothing.
2. **The fast paths tie, and the fast one is NEON's to keep.** BFDOT (40.3) ≈
   fast32 (41.5), but BFDOT stays in the NEON register file — no coprocessor, no
   `SET/CLR`, no scratch plane. For decode, AMX earns nothing on raw speed.
3. **The baseline was under-tuned.** The production leaf ran 2 accumulator chains
   at M=1; going to 4 recovers **2.4×** (11.4 → 27.0) while staying bit-exact. A
   real slice of the apparent "AMX win" was simply an under-parallelized NEON leaf.

**Honest gaps in the evidence.** The M=4 (prefill) cell is *not yet fairly
measured* — the BFDOT probe re-reads `W[n]` per row and is memory-bound for M>1,
unlike the production `gemm_nt_impl`, which streams `W[n]` once across all rows.
A W-reuse-tiled BFDOT GEMM is required before claiming the prefill cell.

---

## 4. The real decision is a numerics contract, not a speed number

Both fast paths (`fast32`, BFDOT) win *by changing the reduction order*. On
random trained-like inputs they publish identical BF16 to the faithful path, but
an explicit cancellation vector proves the divergence survives BF16:

```
NEON widen = 0            -> bf16 0x0000
AMX fast32 = 0.000183     -> bf16 0x3940
AMX exact8 = 0            -> bf16 0x0000
```

So "adopt the fast path" is a **model-value decision** — does the trained model
tolerate the different reduction? — not a microbenchmark tolerance call. The rule
that follows:

- Dense projections whose reduction the model tolerates → fast path (BFDOT first;
  it needs no coprocessor).
- Cancellation-sensitive reductions and anything numerics-validated at a specific
  order → faithful path.
- Every fast kernel ships behind a per-op parity gate plus the cancellation probe.

---

## 5. The argument the microbenchmarks cannot see: heterogeneous concurrency

A single kernel on an idle core cannot measure the one real reason to keep AMX in
the toolkit: **AMX is a distinct execution unit.** Its value is not beating NEON
at matmul (BFDOT ties it); it is running a matmul *concurrently* with NEON doing
the work AMX cannot — RoPE, norm/rsqrt, softmax, gate, ShortConv, gather. Under
multi-lane load, all-NEON makes the matmul and the transcendental work contend
for the same FMA pipelines; moving the matmul to AMX frees those pipelines.

This is bounded by its own scarcity: **AMX is shared per cluster, not per core.**
The M2 Max is 8 P-cores in 2 clusters, so roughly **2 AMX blocks vs 8 NEON
pipelines** (block count to be confirmed by measurement). So AMX adds ~2
heterogeneous execution lanes on top of the 8 NEON — not contention-free, and
mix-dependent:

- **Matmul-dominated:** 8 NEON pipelines parallelize ~4× better than 2 AMX
  blocks. NEON wins.
- **Mixed chain (real decode):** 2 AMX blocks carrying matmuls *while* 8 NEON
  pipelines carry the rest beats 8 NEON pipelines doing both serially. AMX wins —
  by relieving the NEON bottleneck, not by raw speed.

This is untested and is the decisive open experiment (§8).

---

## 6. How the seam is scheduled — the kcoro substrate

The hybrid is a route suspension, resolved by two substrate primitives:

- **`kc_service`** — a retained stackless continuation on the runtime that
  *creates no thread*. Notifications are edge-coalesced while the callback drains
  its predicate; the runtime closes notify-before-park and notify-during-callback
  races. This is "kcoro owns every computation thread," "callbacks wake work," and
  "no spin / no lost wake" as one reusable object — the home for an asynchronous
  AMX service.
- **`kc_team_dispatch_notify`** — the fixed lane team gains a completion-callback
  *edge*: when the last member retires a generation the callback fires once and a
  resumed continuation may immediately dispatch the next generation. That is the
  barrier-completion hook the batched path needs.

The handoff itself:

```
NEON .S leaf  → publish AMX request {views, output lease, shape, ticket}
              → route parks; Flashkern lane released
AMX service   → SET → BF16 VECFP into F32 → STZ into ticket scratch → CLR → complete(ticket)
next .S leaf  → load scratch tile → exact BF16 round / activation / residual → final destination
```

Two properties are load-bearing:

- **Suspend the *route*, not the lane.** A resumed route may land on a different
  lane, so architectural registers cannot be the continuation state — the ticket
  scratch is.
- **The `STZ → scratch → LDR` crossing is irreducible.** AMX and NEON are
  distinct register files with no direct handoff; it is a compute-unit boundary,
  not a convenience copy. It is measured cheap (~4–50 µs across shapes).

Two modes: an **inline AMX leaf** (small tile, same lane, no suspension) versus an
**asynchronous AMX service** (coarse work that overlaps another route; the route
parks and releases its lane).

---

## 7. How AMX earns its keep under kcoro ownership: batch at the barrier

A single-token decode GEMV is bandwidth-bound and cannot use AMX's throughput.
The fix is to let kcoro own the parallelism instead of Accelerate: at a shared-
weight projection, the fence **gathers the ready lanes' activation vectors into
one `[M_lanes, K]` matrix, runs one batched matmul, and scatters the results
back.** Weight reuse across the batch amortizes the `W` read, moving the op off
the bandwidth ceiling into compute-bound territory — where the coprocessor pays.

The parallelism then comes from the *batch size*, not from Accelerate spawning
hidden threads (which both inflates numbers and violates thread ownership;
~1.8× of Accelerate's measured M=64 lead was hidden threads). Because the batch
is the set of concurrent conversations at the same layer, **decode throughput
scales with concurrency** — the multi-conversation runtime is what makes the
coprocessor efficient.

---

## 8. The resulting architecture, and what still has to be measured

**The split.** Fast dense BF16 projections (backbone up/down, decode, Conformer/
adapter, Mimi matrices) run on the fast path — BFDOT first, AMX only where §5 or a
batched barrier justifies the coprocessor. NEON register-resident leaves own the
transcendental and irregular work (RoPE, norm/rsqrt, softmax, gate, ShortConv,
gather) and every cancellation-sensitive reduction. Cluster consecutive same-unit
ops to minimize AMX↔NEON crossings; publish compact partials at the fence.

**Open experiments, in priority order:**

1. **W-reuse-tiled BFDOT GEMM at M=4**, through the same parity + cancellation +
   timing gates, to fill the prefill cell honestly (§3 gap).
2. **The contention harness (§5):** the realistic per-lane mix run across K
   concurrent lanes, all-NEON vs (AMX-matmul + NEON-rest), reporting aggregate
   throughput *and* tail latency. This is the first measurement of the *system*
   rather than a leaf; it also confirms the real per-cluster AMX block count.
3. **Model-value parity** for the chosen fast reduction contract — does the
   trained LFM2 tolerate the fast32/BFDOT order across the full generation?

**Status.** The decode cell is measured and clean; the faithful/fast axis is
measured; the prefill cell, the contention question, and the model-value decision
are open. Nothing here is a gated production claim — these are ground-truth probes
that decide whether the coprocessor earns its complexity before any of it lands in
the engine.
