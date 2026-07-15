# Assembly Kernels, Accelerate Dispatch, and Wait Primitives

Status: normative design. The zero-spin wait-word substrate is implemented at
upstream `bd530f4c9196` and the fixed Flashkern lane mount at `d2c43abd`; the
remaining native math migration is not.

Baseline: EmberHarmony `321538f11749`.

## Goal

Fix the compute substrate for every native stage in documents 03 through 07:

- **No tensor library in the production CPU path.** No Candle, Eigen, MLX-on-CPU,
  generic tensor-object or expression-template framework, or SLEEF/SVML enters
  Flashkern. A CPU "tensor" is a pointer, a shape fact recorded in a plan, and a
  kernel. Apple GPU matrix coprocessing is mandatory and belongs to a separate
  MLX C++/Metal device engine, never inside Flashkern.
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
- **Waiting is zero-spin.** A fence or doorbell reads its generation once,
  registers and rechecks, then blocks through the host wait-word adapter. There
  is no bounded spin or monitor-wait budget before parking.
- **The numerical call graph is native pass descriptor -> C++ fixed executor ->
  assembly table.** Rust converts settings and owns PCM/control I/O scopes only.
  C++ owns model loading, pointer binding, stage planning, state ownership, and
  dispatch, but performs no arithmetic. Sampling, state transforms, and every
  kernel leaf execute in assembly. Transitional Rust and C++ numerical bodies
  are deleted per documents 02 and 07, not optimized.

## Current Ownership Debt

The new boundary replaces concrete production work rather than adding another
native helper below it:

Two substrate debts in the original audit are now closed. Vendor commit
`8d510f83` pins arena `bd530f4c9196`, which separates signal-one `work_cv` from
lifecycle notification at
`crates/kcoro-sys/vendor/kcoro_arena/core/src/kc_runtime.c:225-324`.
Flashkern commit `d2c43abd` owns cache-line-isolated shared dispatch and fence
words and blocks through prepared `kc_port_wait_u32` handles at
`native/src/engine/flashkern_engine.cpp:634-662` and `1041-1052`;
`FENCE_SPIN`, `kcoro_park`, and `kcoro_unpark` are absent.

| Current work in Rust | Evidence | Required native owner |
|---|---|---|
| resampling and audio accumulation | `crates/liquid-audio/src/processor.rs:1089-1163` | native frontend plan and SIMD resampler |
| DFT, mel filtering, log, and normalization | `crates/liquid-audio/src/processor.rs:254-472` | native mel stages and reduction kernels |
| Conformer and adapter tensor graph | `crates/liquid-audio/src/model/conformer/encoder.rs:185-317` and `crates/liquid-audio/src/model/lfm2_audio.rs:403-419` | C++ pass plan over assembly-table entries |
| sampling and token recurrence | Native collective: `native/src/engine/flashkern_engine.cpp:806-970`; transitional owner/recurrence: `crates/liquid-audio/src/model/lfm2_audio.rs:199-281` and `1630-1743` | sampler leaves are assembly-mounted; move opaque RNG/state append and next-pass selection into native conversation control; Rust receives no per-pass result IDs |
| Moshi frame arithmetic/state | `crates/liquid-audio/src/runtime/realtime.rs:1850-2065` | native Moshi pass program |
| native pass entered through Rust capture/trampoline | `crates/liquid-audio/src/compute/flashkern/native_engine.rs:544-593` and `native/src/engine/flashkern_engine.cpp:1283-1291` | model-bound C++ plan with no Rust callback |
| aarch64 feature flags applied to the whole kernel translation unit | `crates/liquid-audio/build.rs:45-58` | baseline and BF16/I8MM objects compiled separately; C++ binds one table after capability checks |
| hot-call panel storage and packing | `crates/liquid-audio/native/kernels/aarch64/flashkern_neon.cpp:74-162` and `native/kernels/x86_64/flashkern_x86.cpp:74-154` | prepack immutable weights at model open and reserve mutable scratch in the plan; no `std::vector`, resize, assign, or payload repack in a pass |
| C++ intrinsic/libm activation and softmax loops | `flashkern_neon.cpp` and `flashkern_x86.cpp` | paired fixed-shape `.S` transcendental kernels with test-only scalar oracle; delete the C++ bodies |
| sampler and PRNG | Native ChaCha20 block kernels are implemented in `native/kernels/{aarch64,x86_64}/flashkern_prng.S`; `run_sampler` is mounted inside token and Depthformer passes, with `REQ_SAMPLE` as a standalone fallback/conformance leaf. | move the one shared stream image from the Rust generation rim into each native conversation; never issue a pass ticket per random draw or codebook |
| CPU streaming short-conv | `REQ_DEPTHWISE_STREAM`, `lfm_depthwise_stream_bf16`, and `flashkern_conv.h` borrow split state/input/weight planes and write output/state directly | keep this CPU-only; replace the sibling Candle Metal route with MLX C++/Metal rather than adding Metal dispatch to Flashkern |

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
expand-to-f32 path as the fallback and the oracle. Accumulation is F32,
matching the reference ladder in documents 05 and 06. Accelerate GEMM operates
on f32 planes (expanded once into a declared scratch plane when a bf16 source
feeds a GEMM stage — a destination write under the byte-movement law, taken
only where the plan proves the stage is compute-bound enough to pay for it).

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
| `STNP`/`LDNP` (non-temporal hint) | Write-once playback blocks and large scratch spills, so PCM/scratch traffic does not evict streamed weights. Kept vs. non-temporal is chosen by measurement per plane and recorded in the plan. |
| `DC ZVA` (block size from `DCZID_EL0`) | Bulk zeroing: silence fills, scratch init, ring block reset |
| 128-byte alignment rule | Contended atomics (fence generation, ring cursors, tile counter) are `alignas(128)` — M-series lines are 128 B; on x86 this also defeats adjacent-line prefetch false sharing |

Concurrency (the "multitasking" set):

| Primitive | Use |
|---|---|
| LSE `LDADD` | The tile-claim counter — one instruction per claim |
| LSE `SWP`, `CAS`, `CASP` | Slot ownership; `CASP` publishes a 16-byte `{generation, offset}` descriptor atomically without a seqlock |
| `LDAR`/`STLR` | Ring cursor publish/observe (release/acquire), matching document 04's ordering rules |
| release/acquire shared doorbells and logical generations | Publish command/stage identity and fan out the exact declared waiter set through one blocking host wait-word edge |
| host wait-word adapter | Immediate blocking for unchanged generation; no `WFE`, `YIELD`, or user-space polling loop is part of the kernel contract |

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

## The Zero-Spin Wait Boundary

Document 03 defines two distinct blocking paths:

```text
coordination work   one ready permit -> signal one kcoro worker
fixed compute       generation unchanged -> register/recheck -> block wait word
```

Neither canonical path has a spin tier. `PAUSE`, `YIELD`, repeated loads,
WFE/UMWAIT time budgets, and timed polling are absent from the generated native
wait code. The former `REQ_CALL` Depthformer exception is gone:
`run_depth_frame` uses `lane_fence` for every cross-lane dependency, and the Rust
`SpinBarrier`/lane callback were deleted together. The former GEMM and DD FFT
grids are typed native passes. DD FFT bit reversal and butterfly stages use the
same zero-spin generation fence; no generic callback or third wait primitive remains.

The host adapter supplies operations equivalent to:

```c
typedef struct kc_port_wait_word kc_port_wait_word;

int kc_port_wait_u32_prepare(uint32_t *address, kc_port_wait_word **out);
int kc_port_wait_u32(kc_port_wait_word *word, uint32_t expected,
                     uint64_t deadline_ns);
void kc_port_wake_u32_one(kc_port_wait_word *word);
void kc_port_wake_u32_all(kc_port_wait_word *word);
void kc_port_wait_u32_release(kc_port_wait_word *word);
```

The address points to aligned raw 32-bit storage owned by the private board. One
internal lock-free `kc_atomic_*` helper family performs every load, decrement,
and increment with explicit memory order from both C and C++. The adapter uses
the address only for expected-value blocking/ordinary-wake bookkeeping and never
publishes a sequence during an ordinary wake. Exact-once handle release may
publish one terminal increment to drain an already-entered waiter. Do not cast
between `_Atomic uint32_t` and
`std::atomic<uint32_t>`; their cross-language layout is not the contract.
The build selects one board owner and one helper implementation. It does not mix
C11 atomic objects, compiler builtins, and `atomic_ref` accesses on the same
word.

A prepared handle backed by a direct futex, supported platform wait, or
condition-variable fallback may implement the contract. A C++ adapter may use
`std::atomic_ref<uint32_t>::wait/notify` over the aligned raw word only when the
selected library implementation is audited to block immediately without a
pre-block spin tier; it may not reinterpret the address as a distinct
`std::atomic<uint32_t>` object. The adapter must close the register/recheck race
and tolerate spurious returns. If the platform cannot supply a conforming
blocking wait, the fixed-executor capability is unavailable; inference does not
silently fall back to spinning.

Prepare the shared dispatch and fence words during executor creation so backend
selection and fallback allocation happen once. Hot waits and wakes use each
handle directly and never search an address registry. Stage/idle waits use an
infinite deadline; coordination timers own deadlines. Shutdown advances and
wakes the dispatch word, joins workers, releases both handles exactly once, and
only then frees the board.

Every fixed lane waits on the shared dispatch word between passes. At a stage
transition, non-last lanes declare bits in the logical park mask, recheck stage
generation, and wait on the shared fence word. The last lane exchanges the mask;
when nonempty it advances that word and performs one address wake-all. Peers that
observed the new generation before blocking clear their own declarations. The
logical generation carries identity; the shared word only delivers the edge.

The committed evidence is
[`G0_FENCE_SPIN_321538F1.md`](../../docs/native/baselines/G0_FENCE_SPIN_321538F1.md)
and
[`G3_SHARED_DOORBELLS_D2C43ABD.md`](../../docs/native/baselines/G3_SHARED_DOORBELLS_D2C43ABD.md).
Across five 1,000-pass runs on the audited M2 Max, G3 versus G0 changed median
run-level p50 from `0.330` to `0.439 ms`, p95 from `0.576` to `0.524 ms`, and p99
from `0.732` to `0.574 ms`. The median remains an optimization target while the
tail materially improves. The next work is barrier-economy/fused-plan work plus
full token/frame percentiles, not a hidden spin tier.

Capture/playback space/data doorbells use the same zero-spin primitive outside
hardware callbacks. Hardware callbacks never wait.

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
5. `native/src/runtime/wait.{h,cpp}`: one register/recheck/block wrapper used by
   shared fixed-executor generations and audio doorbells; host wait-word adapter
   below it, with no spin or monitor-wait inlines. Add C/C++ address-identity and
   memory-order litmus tests proving one selected atomic helper owns every board
   access.
6. Accelerate stage adapter: `cblas_sgemm` calls as declared stages with
   plan-recorded shapes and tiling, Apple-only, behind the same pass contract.
   A versioned tuning profile records the measured house/Accelerate winner; an
   unknown machine defaults to a startup benchmark before readiness, never a
   per-pass race.
7. Link/symbol audit in CI: Apple production binary links Accelerate and
   nothing else numerical; x86_64 links no BLAS; no `cblas_` symbol outside
   the Accelerate adapter; no libm vector calls in kernel objects; `memcpy`
   audit passes on kernel directories.
8. Bandwidth and fence microbenchmarks recorded per machine: GEMV kernels
   against measured STREAM bandwidth; blocking fence p50/p99, syscall/wake
   counts, logical park-mask population, and idle CPU.

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
- Fence latency: blocking generation wake meets the recorded p50/p99/max
  envelope; each nonempty logical park mask causes at most one host wake and no
  coordination wake; idle CPU is indistinguishable from a blocked process.
- Disassembly/source audit finds no compiled spin loop, `PAUSE`, `YIELD`,
  WFE/UMWAIT budget, or timed polling in a fence, command wait, or doorbell.
- C/C++ and TSan tests prove the wait adapter observes the same aligned raw word
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
  transpose, or dynamic allocation during a pass. Immutable repacking, when a
  measured kernel requires it, occurs once at model open and is retained.

## Non-Goals

- No SVE/SVE2 code paths (no target hardware in the product fleet).
- No quantized inference paths yet; I8MM/VNNI are inventory, not commitments.
- No FFT replacement of the DFT basis in this document — that remains
  document 05's separately gated decision, Accelerate vDSP or otherwise.
- No claim about undocumented Apple execution units. A future documented SME or
  other ISA backend is a new capability-tested kernel table.
- No cross-compilation guarantees beyond macOS/aarch64 (product) and
  Linux/x86_64 (CI/reference).
