# 15 — The Weight Engine: Byte-Exact Load and High-Speed Streaming

Status: **design, under review — not authoritative.** The byte-exact model is
endorsed; the corrections to make before this is authoritative are (a) a stronger
memory-ownership model and (b) sharper wake correctness — both added below (§0.1,
§3c). Kept as-is per review: no Holo numerical re-encoding, direct parallel
`pread` into final spans, profile-before-rewrite discipline, no runtime Candle
fallback.

Substrate under P1 (residency) and the decode bandwidth ceiling. Investigates the
provenance of `native/src/io/safetensors.cpp` and plans a high-speed path for
getting weights off disk and streaming them through compute in the correct
order — **byte-exact, no numerical transform.**

## 0.1 Memory ownership — three tiers, plus separately-accounted derived storage

Not one blob. Four distinct lifetimes, each owned and accounted separately:

1. **Immutable model image** — the byte-exact checkpoint bytes, one aligned
   allocation, exposed as unaligned pointer *views*. Never mutated, never copied
   after load. Shared read-only by every plan, every conversation.
2. **Derived storage — separately accounted.** Anything computed *from* the image
   that is not the checkpoint bytes: prefolded tables (BatchNorm denom, rope
   cos/sin), Apple f32 GEMM staging, any re-laid buffer. This is real resident
   memory and must be reported on its own line — `directly_bound_bytes` (tier 1)
   vs a distinct derived/compat counter — never folded into "the model is 2.9 GB."
   `compatibility_copied_bytes` must read 0 in production; derived storage is a
   separate, legitimate, bounded number.
3. **Immutable shared plans** — the bound weight descriptors + shape/stride facts
   for a stage (GEMV geometry, layer table). Built once at model open from tier 1,
   shared by all conversations, never mutated in a pass.
4. **Per-conversation persistent state** — KV planes, short-conv carry, sampler
   CSPRNG, codec state, cursor/epoch. One set per live conversation, mutated
   across that conversation's passes only.
5. **Per-ticket transient scratch** — activation planes for one pass, drawn from a
   pre-reserved arena, valid only until that pass's completion. Never persists.

Tiers 1 and 3 are shared and immutable; tier 4 is per-conversation; tier 5 is
per-ticket. A pass reads tiers 1/3, mutates tier 4, and borrows tier 5. Keeping
these lifetimes distinct is what makes "one image, many conversations" sound and
what keeps the accounting honest.

## 0. The weight model — confirmed, and already realized at load

The proposal — *address weights by shape/stride from one byte-exact blob, never
convert them* — is already the loader's design, not a future state:

- The whole checkpoint is read into one 64-byte-aligned allocation
  (`AlignedBytes`, `posix_memalign`), each shard base 64-aligned, tensors packed
  gapless (`safetensors.cpp`).
- `fill_view` hands out a pure pointer plus metadata:
  `view.data = storage.data() + tensor.offset`, with `shape`, `elements`,
  `bytes`, `rank`, `dtype`, `shard`. **No copy. No conversion. bf16 stays bf16.**
- The safetensors on-disk payload *is* the raw little-endian tensor bytes,
  contiguous from the data section. So a byte-exact load is literally "seek to
  `offset`, read `byte_count`, done"; a view is "base + offset, interpret at
  `dtype` width." Row-major, so per-dim strides are derived from the element
  width — there is no stride table to store.

The only place bytes are *converted* today is `CandleBridge`
(`Tensor::from_raw_buffer(...).to_dtype()`) — the ~2.94 GB copy P1 deletes — and,
at compute time, the Apple AMX path widening bf16→f32 into staging per GEMM. Load
itself is already conversion-free.

## 1. Provenance of `safetensors.cpp`

The header states the lineage:

> *"The span planning and whole-file residency discipline comes from the
> safetensors path in ember-ml. This version deliberately stops before UKM's
> numerical ingress: model payloads remain byte-exact checkpoint storage and
> kernels receive immutable pointers into one process-long aligned image."*

So liquid-audio took ember-ml's **span-planning + whole-file residency** and
deliberately dropped ember-ml's **numerical ingress** — the HoloEngine /
singularity / mailbox machinery that re-encodes weights into a rotor format. That
drop was correct and stays correct (see §4).

ember-ml's loader itself (for reference; `/Volumes/stuff/ukm/ember-ml`): pure
`read`/`pread` (no `mmap`), a `ParallelFileReader` that slices a span into ~8 MiB
chunks across a 4-thread pool with a small in-flight ring, a reusable-buffer
chunk streamer, and a global in-flight-bytes throttle (~512 MiB). That parallel
positioned-read is the transferable load technique (§3). Note the ember-ml tree
is mid-refactor: several referenced loader functions (`SafetensorsResidentBlock`,
`build_safetensors_span_plan`) are declared but not defined — so we borrow the
*technique*, there is no drop-in code.

## 2. The honest finding on the "singularity kernel"

The singularity kernel / `holo_a64.S` / `event_horizon.cpp` were pointed at as
inspiration for "a wickedly high-speed assembly engine to read weights in the
correct order." Having read them: **it is not a bandwidth weight-streaming
engine.** It is a single-threaded opcode interpreter that drains an SPSC ring and
runs small algebra ops (Möbius/Cayley rotor composition) on a 6-slot NEON
register bank. Concretely, across the whole ember-ml tree:

- **No `PRFM` (prefetch). No `LDNP`/`STNP` (non-temporal). No unrolling. No
  software pipelining.** Grep-verified: zero occurrences.
- The only byte-mover is a single `ldr q`/`str q` (16 B/iter) or a paired
  `ldp q,q`/`stp q,q` (32 B/iter) post-increment loop.
- Its "register-resident rotor" pins *algebra state* in `v0..v11`, amortizing
  **opcode-dispatch** overhead — not weight bandwidth. It is not a GEMV/GEMM
  streaming trick. The one `OP_GEMV_BF16_DD` opcode that existed just `bl`s a C
  helper that is **not defined anywhere in the tree**.

So there is no fast weight-read hot loop to copy. The high-speed read kernel is
ours to write, and the techniques it needs are exactly the ones ember-ml lacks.

What the singularity design *does* offer is **coordination discipline**, and a
couple of ideas worth weighing (§4).

## 3. The plan — two halves

### 3a. Fast LOAD: parallel positioned read into the resident image

The current loader reads each shard with a serial `read_file` into
`storage.data() + source.offset`. For a ~3 GB checkpoint that is one thread
against the SSD. The win is byte-exact and self-contained:

- Slice each shard's byte range into fixed chunks (start ~8 MiB) and issue
  **`pread` across a small fixed thread pool** (size = a few IO threads, not the
  P-core compute lanes — IO is not the lane team's job), each `pread` landing
  **directly into the final aligned image slice** (no staging buffer, no per-chunk
  allocation — the destination already exists).
- Bound outstanding IO with a small in-flight window and a global in-flight-bytes
  cap (ember-ml's `LoadThrottle` idea) so a huge model can't spike resident RAM
  during load.
- Alignment is a non-issue for correctness: `pread` writes exact bytes at exact
  offsets; the 64-byte base alignment of the image is preserved because chunks
  are placed by absolute offset.
- This stays entirely on the **byte-exact** side of ember-ml's pipe — cut before
  `tensor_ingress`; there is no descriptor/mailbox/worker needed for a synchronous
  positioned read into an existing buffer.

*Gate:* load wall-clock drops toward SSD-parallel-bandwidth-bound (seconds, not
tens of seconds); resident bytes and every tensor view are byte-identical to the
serial loader (a checksum over the image proves it). Independent of P1–P4 — can
land first.

### 3b. High-speed STREAM: measure the 66→250 GB/s gap, then close it

Our decode is M=1 GEMV — every token streams the whole model, so tok/s ≈
model_bytes / achieved_bandwidth. The engine realizes ~66 of a ~250 GB/s
practical bound. The GEMV leaf is *already* contiguous-streaming, prefetched
(256 B ahead), 2-row-ILP, and banded across lanes on the NK path — so the gap is
**not** a naive hot loop. Before rewriting anything, decompose where the
bandwidth goes; the likely levers, in order of expected payoff:

1. **Non-temporal streaming loads (`LDNP`) for the weight stream.** Decode weights
   are *use-once per token* — they pollute cache they'll never reuse. Our leaf
   uses `__builtin_prefetch(..., locality 0)` but still issues normal `LD1`.
   Switching the weight-side loads to non-temporal (hand-asm; intrinsics can't
   express `LDNP`) may be the single biggest untried win: it stops the weight
   stream from evicting the hot activation/accumulator working set. **Measure
   first** — NT loads help only when cache pollution is the actual stall.
2. **Deeper unroll + tighter schedule via hand-asm.** The intrinsics form is
   2-row ILP; a hand-written leaf can run 4×+ load streams, schedule PRFM
   distance explicitly, and keep more accumulators live to hide FMA latency.
   Only worth it if (1) shows the loop is compute/issue-bound, not DRAM-bound.
3. **Cut the Apple AMX per-call widening traffic.** For M>1 (prefill, suffix
   chunks) every GEMM widens bf16→f32 into `gemm_amx_*` staging per call — extra
   write+read traffic Accelerate then re-reads. Options: keep it (staged tiles are
   cache-friendly) vs. a one-time f32 shadow of the hot matrices (doubles their
   footprint — rejected, see doc 14 Trade-off 5). Measure whether the widen is on
   the critical path at all before touching it.
4. **Prefetch-distance / stream-count tuning** on the existing leaf — cheap,
   measurable, no rewrite.

The discipline: **profile the decode bandwidth ledger per stage, attribute the
missing ~180 GB/s, then apply the specific lever the profile names.** No blind
asm rewrite of a leaf that already declares itself bandwidth-bound. Widen the
accumulator (double-double) where dynamic range needs it, never the storage —
weights stay bf16 (this is the one genuinely useful HOLOFLOAT principle:
`GEMV_BF16_DD` reads bf16 directly, only the accumulator widens).

### 3c. Coordination: wake correctness first, then `wfe`/`sev`

**Wake correctness, stated precisely (the load-bearing invariant):** zero-spin
does **not** mean "avoid sleeping" — it means **park on a shared predicate that
the waker advances, so no wake is ever lost and no waiter ever polls.** The
predicate is the shared word (the pass generation / expected value); the futex or
`wfe` is only the *edge* that makes a parked waiter re-check it. A correct wait is:
read the shared predicate; if already satisfied, proceed without sleeping; else
park keyed to the value observed, so a waker's release-increment either is seen
before parking or wakes the park. Any design that sleeps for a fixed time, or
wakes and re-polls a value it didn't park on, is wrong regardless of how little it
spins. Our `kc_port_wait_u32(word, expected, deadline)` is exactly this shape and
is the contract every wait in the engine must keep.

Given that invariant, `wfe`/`sev` is an *implementation* of the wake edge, not a
replacement for the shared predicate — the predicate (pass generation) stays
either way.

Our lane dispatch and fence use `kc_port_wait_u32` (expected-value word over
`os_sync_wait_on_address`/futex). ember-ml's resident worker instead parks on the
ARM64 **`wfe`/`sev`** event register: `sev` sets a sticky, edge-latched per-core
event bit, so `wfe` cannot miss a wake that races the pre-park check — the
lost-wakeup problem solved in two instructions with **no shared doorbell word and
no compare-value**. `sevl` primes the register so the worker inspects the queue
once before ever parking.

Where it could apply: the **lane `dispatch_word`** specifically — a small fixed
resident team is exactly the case `sev` fits (it broadcasts to all PEs, which is
wasteful for many waiters but free for ~8 pinned lanes). Caveats that keep this a
*targeted* swap, not a wholesale one:

- `sev` is a broadcast, not addressed to one waiter — fine for the lane team,
  wrong for per-conversation completion routing.
- ember-ml's model has **no completion queue** — completion is "ring drained to
  empty," in-order. Ours has a real CQ with per-pass results and epoch routing;
  that is strictly more capable and must not regress.
- The producer side in ember-ml busy-spins (`sched_yield`) on `flush`; ours
  blocks properly. Keep ours.

So: consider `wfe`/`sev` for the lane wake edge (a micro-optimization of an
already-zero-spin path), keep the expected-value CQ everywhere completion must be
addressed. Low priority; correctness-neutral; do it only if lane wake latency
ever shows up in a profile.

## 4. The ember-ml borrow ledger (explicit)

**Take (all byte-exact / format-agnostic):**

| Idea | Where it lands here |
|---|---|
| Parallel positioned `pread` into the final buffer, bounded in-flight | §3a — the load win |
| In-flight-bytes throttle during load | §3a |
| Host-as-bus / persistent resident kernel owns compute | already the flashkern lane team |
| Register-resident hot tile, explicit memory-crossing only | already the leaves; sharpen with hand-asm §3b |
| Stream → consume → drop for load (pointer walk, bounded scratch) | §3a + the resident-image binding (P1) |
| Widen the accumulator, not the storage (dd over bf16) | §3b, item under the list |
| `wfe`/`sev` sticky-event doorbell | §3c — lane wake only, optional |

**Reject (numerical transform / weaker than what we have):**

| Idea | Why rejected |
|---|---|
| HOLOFLOAT rotor re-encoding of weights | Not byte-exact; redefines the arithmetic. Our law: bf16 bytes untouched. |
| Möbius group-law "matmul" | Computes a different function than the bf16 dot product. |
| `from_holo`/voxel array sink | Non-materializable — throws on raw pointer access. Antithetical to weight views. |
| Drain-to-empty completion (no CQ) | Our SQ/**CQ** with epoch routing is strictly more capable. |
| Producer busy-spin on flush | Our two-sided expected-value doorbell already blocks properly. |
| 6-slot bank / NEON-only / single-worker / no-fallback absolutism | Instance-specific limits; the pattern generalizes, this instance doesn't. |

## 5. How it fits the phases

- **§3a (parallel load)** is independent — it can land any time, gated by a
  byte-identity checksum. Good early, low-risk, visible win ("seconds to load").
- **§3b (stream bandwidth)** is a *measurement-gated* track that runs alongside
  P1–P3; the resident-image binding (P1) is its precondition (one image, no Candle
  copy thrashing the cache). No rewrite lands without a profile attributing the
  gap.
- **§3c (`wfe`/`sev`)** is an optional micro-opt on the lane wake edge; not on any
  critical path; defer until a profile asks for it.

## 6. The one-line synthesis

The load is already conversion-free and can be made *fast* by borrowing ember-ml's
parallel `pread`; the high-speed *read* is ours to write because ember-ml has no
such kernel — and our GEMV is already near the memory wall, so the remaining win
is measured bandwidth work (non-temporal streaming, hand-asm scheduling), not a
naive→optimized rewrite. The singularity machinery contributes coordination
discipline (`wfe`/`sev`, host-as-bus, stream→consume→drop) and a cautionary line:
its speed thesis is inseparable from a numerical re-encoding we must never adopt.
