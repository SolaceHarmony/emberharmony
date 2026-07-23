# Assembly Kernels, Accelerate Dispatch, and Callback Continuations

Status: normative design. The fixed Flashkern lane mount, typed
resample/frontend/whole-Conformer request, capacity-2 SQ/CQ, per-ticket scratch
slots, and exact-CQ continuation primitive are implemented. There is no
operation waiter: a numerical member returns, the full-team return publishes
the ticket's completion edge, and the retained continuation either submits the
next phase or releases the exact packed `{generation,state}` slot lease.
Adversarial tests cover peer admission, stale-owner ABA, stop, and capacity
accounting.
The V2.0 substrate repair is also implemented in the working tree: hot
doorbell/cache words and internal SQ/CQ storage cells have 128-byte Apple base alignment and stride
through kcoro, the engine, bridge, session, and model gate; request, layer, and
modality selectors are closed; invalid worker/logical-lane geometry rejects; and
four physical workers preserve the eight-way logical fold with parity. The
idle-dormancy gate remains green.
The bounded production audio route now uses that continuation across
`TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE`: playback is reserved before
admission, token context commits on its CQ, and Mimi writes direct at equal rate
or feeds a prepared native resampler that writes the reserved device-rate PCM
span. The exact slot is released before reliable publication.
The retained route service is the sole SQ producer. Each submission names its
fixed pass slot and exact ticket generation directly; the bridge has no generic
descriptor pool, borrowed mode, producer mutex, or retrying admission loop.
Peer admission is represented by bounded route and pass-slot claims rather than
hidden behind a synchronous fast path.
The coordinator is a retained service. Its durable `SessionAction` and
`ResultRecord` records survive numerical completion and outward backpressure;
no host stack or sleeping thread represents that suspended action. Pooled
routes/results and the broker are implemented. Two `BLOCK4` domains,
reverse-order per-block CQs, and concurrent numerical passes remain open in
designs 14 and 16. The full Moshi migration is a later tranche.

Baseline: EmberHarmony `321538f11749`.

## Goal

Fix the compute substrate for every native stage in documents 03 through 07:

- **No tensor library in the production CPU path.** No Candle, Eigen, MLX-on-CPU,
  generic tensor-object or expression-template framework, or SLEEF/SVML enters
  Flashkern. The word "tensor" is not a production type: values are immutable
  byte-image views or mutable activation spans with plan-validated extents and
  strides. A later MLX C++/Metal device engine is an independent tranche, never
  a fallback inside Flashkern.
- **All math is assembly.** AArch64 NEON and x86_64 AVX2/AVX-512 numerical
  kernels are hand-written `.S`. Production Rust and C++ contain no floating
  arithmetic, SIMD intrinsics, scalar libm loop, tensor object, or compiler-owned
  numerical body. Scalar oracles exist only in a separately linked native test
  target.
- **Accelerate is an Apple-native stage backend, not a tensor framework.** At
  model open, actual model shapes are benchmarked against house kernels from a
  versioned tuning profile. A stage binds `cblas_sgemm` only when parity passes
  and the recorded target-machine result wins. The ABI is pointers, dimensions,
  and strides against our planes. Do not claim a private Apple execution unit;
  Accelerate's implementation is opaque.
- **Computational progress is edge driven.** Tickets, stages, results, playback
  capacity, capture data, and control changes never own waiters. Their producers
  publish an edge that makes a retained continuation runnable. Correlated
  monotonic one-shots are separate supervision/policy sources: expiry may
  publish a terminal quorum fault or satisfy one half of a Sesame pause gate,
  but it cannot fabricate numerical completion. Only a resident kernel worker
  whose complete ready predicate is empty may become dormant on one indefinite
  expected-value doorbell. That worker is shared execution capacity, not the
  suspended operation. There is no bounded spin, timed polling, per-stage park,
  or terminal-result wait tier.
- **The numerical call graph is native pass descriptor -> C++ fixed executor ->
  assembly table.** Rust converts settings and owns control/observation only;
  native code owns PCM and platform I/O scopes.
  C++ owns model loading, pointer binding, stage planning, state ownership, and
  dispatch, but performs no arithmetic. Sampling, state transforms, and every
  kernel leaf execute in assembly. Transitional Rust and C++ numerical bodies
  are deleted per documents 02 and 07, not optimized.

## Current Ownership Debt

The new boundary replaces concrete production work rather than adding another
native helper below it:

Two substrate debts in the original audit are now closed. Kcoro worker progress
uses one runtime-owned expected-value doorbell: a worker observes its generation,
rechecks the protected queues and retained-service predicates, then becomes
dormant on that exact value only when the entire ready predicate remains empty.
The session's `lifecycle_cv` is a terminal administrative join latch only: it
cannot make a route, PCM record, or numerical pass runnable and has no deadline.
A retained
`kc_service_notifier` publishes a callback edge without allocation, a mutex, a
deadline, or invoking the callback; creation rejects the pthread fallback, and
its lifetime lease prevents service destruction until the producer is quiescent.
Flashkern commit `d2c43abd` introduced the private address-dormancy substrate;
`FENCE_SPIN`, operation-level park/unpark, and numerical fence waiters are now
absent. The working-tree V2.0
repair replaces its insufficient 64-byte isolation with 128-byte Apple base
alignment **and stride** and extends that rule to the hot kcoro ring, engine,
bridge, session/model-gate, and SQ/CQ storage cells. ABI-v1 command/completion
values retain their 64-byte caller alignment. The four-versus-eight lane parity
and current-selector rejection gates pass. The bounded three-node production
audio route now advances token through Depthformer and Mimi with a total outcome
map and a pre-reserved playback span. The pooled asynchronous executor, two
independent `BLOCK4` domains, and an architectural idle-dormancy backend are not
implemented by this repair.

The Rust resampler, frontend, Conformer, backbone, sampler, recurrence, and
native-pass trampoline cited by the original audit have been deleted. LFM2
production inference is native. Remaining ownership debt is within the native
implementation, not permission to restore a Rust oracle or transport rim:

| Remaining debt | Evidence | Required owner |
|---|---|---|
| optional Moshi-style frame shell still carries owned Rust PCM vectors | `crates/liquid-audio/src/runtime/realtime.rs` | later independent native Moshi session; no LFM2 fallback |
| historical C++ intrinsic kernels and thread-local work vectors | `native/kernels/{aarch64,x86_64}/flashkern_{neon,x86}.cpp` | paired `.S` leaves plus plan-owned fixed activation arenas |
| create-time scalar Conformer table construction and residual C++ movement | `native/src/model/lfm_conformer.cpp` | formula-derived immutable tables are allowed; all value-producing pass math and avoidable movement move to `.S`/Accelerate stages |
| compatibility synchronous pass/result surfaces | `native/src/engine/flashkern_engine.cpp` and the product ABI audit | delete after their production callers become retained ticket continuations; never preserve as a legacy path |

## The Library Law

| Category | Production voice path | Native test target | Baseline fixture capture |
|---|---|---|---|
| Candle, Moshi-Candle, MLX-CPU, Eigen, CPU tensor frameworks | Banned from Flashkern and final CPU inference | Not linked or called | May run only from a pinned Git worktree; deleted code is never copied forward |
| MLX C++/Metal device backend | Required peer backend; never linked into Flashkern | Allowed in its own device tests | Temporary Candle Metal may supply migration fixtures |
| Accelerate BLAS (`cblas_sgemm` family) on Apple | Allowed only as a profile-selected opaque machine-code/AMX backend; C++ dispatches but performs no math | Allowed | Not applicable |
| Accelerate vDSP/vForce/BNNS | Not used initially; vDSP FFT requires document 05's separate parity gate | Allowed | Not applicable |
| External BLAS on x86_64 (MKL, OpenBLAS) | Not used; house kernels own x86_64 | Allowed as a benchmark oracle only | Not applicable |
| architecture `.S` | Production numerical substrate | Allowed | Not applicable |
| `<arm_neon.h>`, `<immintrin.h>`, C++ inline asm | Migration debt; absent at cutover | Allowed in a separately linked oracle/benchmark target | Not applicable |
| scalar C++ and scalar libm transcendentals | Migration debt; absent at cutover | Allowed as oracle | Not applicable |

Transcendentals used by the model — `exp` (softmax), `tanh`, exact-`erf` GELU,
`log` (mel guard) — must become house vector kernels with stored fixtures. Their
polynomial/range-reduction choices are recorded per kernel and gated against
committed fixtures plus the test-only scalar C++ oracle. Baseline fixtures may
have been generated by the pinned Rust/Candle commit, but native tests never
call that code. The current scalar calls are an explicit migration debt, not an
as-built claim of compliance; the completed production path may not retain them.

## The Byte-Movement Law

"Absolute zero-copy" means exactly two legal payload movements:

1. **HAL ingress**: the hardware callback's one bounded copy from the ephemeral
   device buffer into the owned capture ring (document 04). Named, counted,
   singular.
2. **Kernel destination writes**: a kernel consuming read-only sources and
   writing its declared destination plane. Embedding-row gathers into the
   prefill plane (document 07) are destination writes, not copies.

Everything else moves as `(region_id, generation, offset, length)`. Raw
`memcpy`/`memmove` on payload planes is banned outside the two named sites;
ingress and destination writes are wrapped in named helpers so a symbol/audit
pass can prove the rule mechanically.

## Kernel Families and the Accelerate Split

One decision rule, applied once per stage when the plan is built at model open
(document 06: "selected once in the plan"), never per pass:

| Stage shape | Bound by | Apple dispatch | x86_64 dispatch |
|---|---|---|---|
| Decode-path GEMV (backbone token pass, heads, depthformer blocks) | Memory bandwidth | House NEON BF16 kernels; the large-matrix adapter is not selected for this shape without contrary measurement | House AVX2/AVX-512 kernels |
| Large GEMM (multi-token prefill chunks, Conformer projections and attention scores, DFT-basis and mel-filterbank matmuls) | Compute | House BF16/F32 kernels or Accelerate `cblas_sgemm`, selected from a versioned measured profile | House AVX2/AVX-512 GEMM tiles |
| Elementwise, residual, activation, GLU | Bandwidth | House NEON | House AVX |
| Reductions (LayerNorm, softmax, mel statistics) | Bandwidth/latency | House NEON (`FMLA`, `ADDV`, `FRECPE`/`FRSQRTE` + Newton steps) | House AVX (masked tails) |
| Depthwise/short convolution | Mixed | House NEON | House AVX |
| Resampler polyphase taps | Compute (small) | House NEON `FMLA` | House FMA3 |
| Format conversion, downmix (capture callback) | Bandwidth | `LD2`/`LD4` de-interleave + convert | `PMOVZX`/`CVTDQ2PS` family |
| ChaCha20 PRNG block | Register/latency | Hand-written AArch64 scalar-register block; `SecRandomCopyBytes` only seeds/reseeds outside passes | Four-row SSE2 block; platform CSPRNG only seeds/reseeds outside passes |

The DFT-basis mel stage is a GEMM in disguise — `(2·bins × window) ×
(window × frames)` — so it is eligible for the same plan-time Apple comparison
as prefill while retaining document 05's exact-DFT numerics. Eligibility does
not predetermine the winner.

BF16 handling is exact everywhere: a bf16 value is the top 16 bits of its f32,
so expansion is a zero-extend and 16-bit left shift — `SHLL`/`ZIP` on NEON,
`VPMOVZXWD` + `VPSLLD` on x86. Where hardware dot products exist they are used
(`BFDOT`/`BFMMLA` on ARMv8.6, `VDPBF16PS` under AVX512-BF16), with the exact
in-register expand-to-f32 path as the fallback and oracle. Accumulation is F32,
matching the reference ladder in documents 05 and 06. A BF16 checkpoint weight
is never widened into an F32 plane for Accelerate. Accelerate may be selected
only when it can consume the resident dtype/layout directly or when the source
is already a legitimate F32 view. Otherwise the plan binds the house BF16 leaf;
layout, alignment, transpose, or dtype staging is forbidden weight
materialization rather than a destination write.

Accelerate integration constraints:

- Calls happen inside a declared serial or tiled stage of a pass, from lane
  context the plan assigns. They must not oversubscribe the lane team; whether
  a stage issues one large call or per-lane tiled calls is fixed by
  measurement at plan-build time, and the document 03 idle/wake gates apply
  unchanged during Accelerate-dispatched stages.
- No Accelerate call appears in a hardware audio callback or holds a fence.
- Accumulation order inside Accelerate may differ from house kernels; the per-stage
  recorded tolerance policy (documents 05/06) governs, and end-to-end
  token/wav-hash gates remain the backstop. A stage may not relax its own
  tolerance because Accelerate disagreed with the oracle.

## aarch64 Primitive Inventory

Product target: Apple M2-class aarch64. The audited M2 Max reports
`hw.optional.arm.FEAT_BF16=1`, `FEAT_I8MM=1`, and a 128-byte cache line, but the
binary still treats these as runtime capabilities. A baseline NEON object and
separate BF16/I8MM objects are compiled independently; `sysctlbyname` feeds the
capability bits in document 01 before C++ binds the table. The current global
`-march=armv8.3-a+bf16+i8mm` at `crates/liquid-audio/build.rs:53-56` is removed so
an unavailable feature cannot leak into baseline code before the check.

Math:

| Primitive | Use |
|---|---|
| `FMLA`/`FMLS` (vector FMA) | All f32 accumulation ladders |
| `BFDOT`, `BFMMLA` | bf16 dot/2×2 matmul-accumulate microkernels for GEMV tiles |
| `BFCVT`/`BFCVTN`, shift-expand | Exact bf16↔f32 |
| `SMMLA`/`UMMLA`/`USMMLA` (I8MM) | Reserved for future quantized paths |
| `TBL`/`TBX`, `ZIP1/2`, `UZP1/2`, `TRN1/2`, `EXT` | In-register shuffles and transposes — "a transpose is an indexing decision" (document 06) made literal |
| `LD2`/`LD4`, `ST2`/`ST4` | Channel de-interleave in capture conversion; codebook-major layouts |
| `FRECPE`/`FRSQRTE` + Newton steps | Fast reciprocal/rsqrt in normalization when the parity policy admits them |
| `ADDV`, `CNT` | Horizontal reductions; active-lane mask population |

Memory:

| Primitive | Use |
|---|---|
| `PRFM PLDL1KEEP`/`PLDL1STRM` | Weight streaming reads (reads = floor); STRM for one-shot planes |
| `STNP`/`LDNP` (non-temporal hint) | Candidates for write-once playback blocks and large scratch spills. Cache effects are implementation-dependent; kept vs. non-temporal is chosen by measurement per plane and recorded in the plan. |
| `DC ZVA` (block size from `DCZID_EL0`) | Bulk zeroing: silence fills, scratch init, ring block reset |
| 128-byte Apple alignment rule | Every hot atomic has 128-byte base alignment and 128-byte array/member stride on both Apple slices, including x86_64 under Rosetta. Non-Apple targets bind a compile-time platform value and verify it at startup; a runtime value cannot define C++ object layout. |

Concurrency (the "multitasking" set):

| Primitive | Use |
|---|---|
| LSE `LDADD` | Optional relaxed tile-claim specialization after capability selection; stronger acquire/release forms are not required for a counter whose stage board is already published |
| LSE `SWP`, `CAS` | Slot ownership using the narrowest atomic state actually required. `CASP` is reserved for a future proven 128-bit identity transition, not required by the current 64-bit packed lease. |
| `LDAR`/`STLR` | Ring cursor publish/observe (release/acquire), matching document 04's ordering rules |
| release/acquire shared doorbells and logical generations | Publish a ready edge after durable ticket state is visible; the retained continuation consumes the edge and advances the route |
| private idle-dormancy backend | OS expected-value address dormancy for an otherwise idle resident worker; optional guarded `LDXR`/`WFE` experiment on AArch64. Neither backend is exposed to an operation or admits a user-space polling loop. |

## x86_64 Primitive Inventory

Baseline AVX2+FMA3; AVX-512 (F/BW/VL, VNNI, BF16) and WAITPKG detected via
`CPUID` at runtime — capabilities compiled in, dispatch selected once.

| Primitive | Use |
|---|---|
| `VFMADD231PS` family | f32 accumulation |
| `VDPBF16PS`, `VCVTNE2PS2BF16` (AVX512-BF16) | bf16 dot fast path where present |
| `VPMOVZXWD` + `VPSLLD` | Exact bf16 expand fallback |
| AVX-512 mask registers | Ragged tails without scalar epilogue loops — frame counts and bin counts rarely divide the vector width |
| `VPERMPS`, `VPERM2F128`, `VGATHERDPS` | Layout shuffles; gathers where profitable |
| `VPDPBUSD` (VNNI) | Reserved for future quantized paths |
| `MOVNTPS`/`MOVNTDQ` + `SFENCE`, `PREFETCHNTA` | Non-temporal stores/loads, same plane policy as `STNP` |
| `LOCK XADD`, `CMPXCHG16B` | Tile claims; 16-byte descriptor publish |
| `RDTSCP` | Stage timing for the performance gates |

## The Callback and Dormancy Boundary

Document 03 defines one computational progress path:

```text
producer publishes durable state -> ready edge -> retained continuation runs
numerical stage returns as a full team -> exact ticket CQ -> next route label
```

No ticket, stage, route, result, audio block, or capacity condition owns a
waiter. A suspended operation is only durable state: ticket identity, route
label, phase cursor, epoch, borrowed-span leases, scratch-bank lease, and output
reservation. The producing edge makes that record runnable. Callback code may
drain ready work or publish another edge; it may not sleep, poll, allocate, or
invoke a foreign runtime.

The fixed team executes exactly one non-suspending numerical stage per
generation. Every member returns from the stage. The team completion callback is
the quorum edge; it advances the ticket cursor and either dispatches the next
stage or publishes the terminal CQ record. Depthformer, Conformer, prefill,
sampling, GEMM, DD FFT, and Mimi use this same callback-phased rule. There is no
per-stage ticket, `lane_fence`, numerical collective waiter, or host-mediated
transpose boundary.

One private exception exists below the operation model: a resident runtime or
team worker with no runnable continuation may become dormant on one shared
expected-value doorbell. The operation is not attached to that worker; any
worker may consume the next ready record. The doorbell surface is deliberately
deadline-free:

```c
uint32_t kc_doorbell_observe(const kc_doorbell *doorbell);
int kc_doorbell_park(kc_doorbell *doorbell, uint32_t expected); /* runtime idle only */
void kc_doorbell_ring(kc_doorbell *doorbell);
```

The low-level `kc_port_wait_u32` name remains an implementation detail of that
single idle-dormancy adapter and administrative tests; it is not callable by a
computational operation. The adapter is prepared once at runtime construction,
uses aligned raw 32-bit storage owned by the private board, rechecks the entire
ready predicate before becoming dormant, tolerates spurious host returns, and
has no deadline. Shutdown publishes a stop edge, joins the resident workers,
releases the adapter exactly once, and only then frees the board. There is no
condition-variable fallback on a realtime callback path and no spin fallback.

`kc_deadline_source` is a different primitive. It is created and sealed during
runtime readiness, retains fixed child identity until cancellation/expiry is
acknowledged, and publishes a small correlated record from an OS monotonic
one-shot. Its handler never dereferences numerical, route, conversation, or
scratch storage. Healthy numerical completion retires the matching arm without
publishing an event; a winning expiry resumes only the owning supervisor.

All atomic access to the doorbell uses one internal `kc_atomic_*` helper family
with explicit memory order from C and C++. Do not cast between `_Atomic
uint32_t` and `std::atomic<uint32_t>`; their cross-language layout is not the
contract. The build selects one board owner and one helper implementation. It
does not mix C11 atomic objects, compiler builtins, and `atomic_ref` accesses on
the same word.

For design 16's block mode, fixed members run one non-suspending stage and all
return. The final return may run the declared bounded mixer exactly once before
publishing the callback edge. No lane may suspend, retire, or switch programs
mid-stage. Assembly owns complete tiles; retained continuations sequence the
stages. Dynamic audio fragment assembly leaves the route frame dormant; its
retained fragment record captures quorum, and the final fragment makes the
exact frame runnable before numerical admission.

The committed evidence is
[`G0_FENCE_SPIN_321538F1.md`](../../docs/native/baselines/G0_FENCE_SPIN_321538F1.md)
and
[`G3_SHARED_DOORBELLS_D2C43ABD.md`](../../docs/native/baselines/G3_SHARED_DOORBELLS_D2C43ABD.md).
Across five 1,000-pass runs on the audited M2 Max, G3 versus G0 changed median
run-level p50 from `0.330` to `0.439 ms`, p95 from `0.576` to `0.524 ms`, and p99
from `0.732` to `0.574 ms`. The median remains an optimization target while the
tail materially improves. The next work is barrier-economy/fused-plan work plus
full token/frame percentiles, not a hidden spin tier.

Capture/playback space/data publication uses the same edge discipline. Hardware
callbacks only publish bounded records and ring; they never become dormant.

## Dispatch and Capability Detection

- One kernel table per architecture, chosen once at model open from runtime
  detection (`sysctlbyname` hw.optional / `CPUID`), recorded in the plan, and
  reported through `lfm_runtime_get_capabilities` (document 01). No per-pass
  branching on ISA, no environment variables.
- Scalar oracle kernels build only with native tests and are absent from the
  production archive and runtime dispatch table.
- Capability detection has two layers: architectural presence and a guarded
  startup probe. A feature is bound only if both pass. The release build never
  executes a feature instruction merely to discover that the host lacks it.

## Implementation Map

1. `native/tests/oracles/` scalar C++ oracles for every kernel family, with
   stored fixtures (inputs, outputs, tolerances) per document 05/06 policy.
2. `native/kernels/aarch64/`: bf16 GEMV tiles (`BFDOT`/`BFMMLA`), f32 GEMM
   tiles, elementwise/reduction/transcendental families, conversion kernels.
   The existing bf16 NEON decode kernels are the seed; they are inventoried
   and re-homed here, not rewritten.
3. `native/kernels/x86_64/`: AVX2 baseline of the same families; AVX-512
   variants behind runtime dispatch.
4. **As built:** `flashkern_prng.S` on both architectures expands one
   snapshot-stable ChaCha20 stream block with no allocation or syscall. The
   Apple entropy thunk calls `SecRandomCopyBytes` only at conversation creation
   or explicit reseed. `run_sampler` is absorbed into token and Depthformer
   passes and adds no per-draw or per-codebook scheduling.
5. Kcoro doorbell substrate: one observe/recheck/dormancy wrapper used only by
   otherwise-idle resident workers. Keep the OS address-dormancy adapter as the
   baseline; admit an architectural event backend only through capability probe
   plus the design-16 measurement gate. Operations publish ready records and
   never call this wrapper. Add C/C++
   address-identity and memory-order litmus tests proving one selected atomic
   helper owns every board access.
6. Accelerate stage adapter: `cblas_sgemm` calls as declared stages with
   plan-recorded shapes and tiling, Apple-only, behind the same pass contract.
   A versioned tuning profile records the measured house/Accelerate winner; an
   unknown machine defaults to a startup benchmark before readiness, never a
   per-pass race.
7. Link/symbol audit in CI: Apple production binary links Accelerate and
   nothing else numerical; x86_64 links no BLAS; no `cblas_` symbol outside
   the Accelerate adapter; no libm vector calls in kernel objects; `memcpy`
   audit passes on kernel directories.
8. Bandwidth and dispatch microbenchmarks recorded per machine: GEMV kernels
   against measured STREAM bandwidth; callback-dispatch p50/p99, host wake
   counts, ready-record depth, and idle CPU.

## Acceptance Gates

- Every kernel family passes its stored-fixture parity gate against the scalar
  oracle on both architectures before entering any pass program.
- PRNG known-answer tests pin exact block bytes and post-draw state on both
  architectures; snapshot replay is exact and the warmed expansion path has no
  entropy call, allocation, payload copy, or per-draw scheduler edge.
- Decode GEMV sustains its recorded fraction of measured memory bandwidth on
  the target machine; regressions against the recorded baseline fail the gate.
- Accelerate-dispatched stages meet the same per-stage tolerance fixtures as
  house kernels; no stage-local tolerance is widened to admit Accelerate.
- Callback-dispatch latency meets the recorded p50/p99/max envelope; numerical
  stage transitions cause no host wake, while an edge into a completely idle
  runtime causes at most one host wake; idle CPU is indistinguishable from a
  dormant process.
- Disassembly/source audit finds no compiled spin loop, `PAUSE`, `YIELD`,
  operation wait, numerical fence, monitor budget, or timed polling. If an
  event-register idle backend is selected, the audit proves one private
  arm-and-dormancy loop around the empty-ready predicate and an OS fallback.
- C/C++ and TSan tests prove the doorbell adapter observes the same aligned raw word
  used by the selected atomic helper; no atomic-object reinterpret cast or mixed
  helper access exists.
- The link/symbol audit passes: no tensor-framework, BLAS (outside Accelerate
  on Apple), SLEEF/SVML, or vector-libm symbol in the production native
  library.
- A call-stack/symbol test proves the local numerical path begins at descriptor
  dispatch in the fixed C++ executor, enters an architecture `.S` symbol before
  touching payload values, and contains no Rust frame, C++ numerical body, or
  payload-bearing Rust FFI symbol.
- The release link map contains no scalar oracle objects. Unsupported ISA
  selection fails before model readiness rather than falling back.
- bf16 expand path is bit-exact against the shift-expand definition on both
  architectures; hardware bf16 dot paths meet their recorded tolerance.
- Non-temporal store selections are justified by a recorded measurement per
  plane; no plane uses NT stores without one.
- Release kernels perform no `std::vector` growth, panel packing, weight
  transpose, or dynamic allocation. Measurement may select a different direct
  byte-view kernel, but never authorizes immutable weight repacking at model
  open.

## Non-Goals

- No SVE/SVE2 code paths (no target hardware in the product fleet).
- No quantized inference paths yet; I8MM/VNNI are inventory, not commitments.
- No FFT replacement of the DFT basis in this document — that remains
  document 05's separately gated decision, Accelerate vDSP or otherwise.
- No claim about undocumented Apple execution units. A future documented SME or
  other ISA backend is a new capability-tested kernel table.
- No cross-compilation guarantees beyond macOS/aarch64 (product) and
  Linux/x86_64 (CI/reference).
