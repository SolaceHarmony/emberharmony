# 15 — The Weight Engine: Byte-Exact Load and High-Speed Streaming

Status: **authoritative for the LFM2 native loader and weight-consumption
contract.** The working tree implements the byte-exact, one-image loader and
direct BF16 consumers described here. Load-throughput tuning remains
measurement-driven; it may not weaken the ownership or no-materialization
rules.

**Terminology:** "tensor" below is shorthand for a non-owning typed view
(pointer, byte length, dtype, shape, and derived strides). No production tensor
container owns or materializes model bytes.

Substrate under P1 (residency) and the decode bandwidth ceiling. Investigates the
provenance of `native/src/io/safetensors.cpp` and plans a high-speed path for
getting weights off disk and streaming them through compute in the correct
order — **byte-exact, no numerical transform.**

## 0.1 Memory ownership — five tiers

Five distinct lifetimes are owned and accounted separately:

1. **Immutable model image** — the byte-exact checkpoint bytes, one aligned
   allocation, exposed as unaligned pointer *views*. Never mutated, never copied
   after load. Shared read-only by every plan, every conversation.
2. **Derived storage — separately accounted.** Only formula-changing immutable
   values computed from the image are admitted: rope/window/FFT tables, BatchNorm
   denominators, Mimi effective-codebook folds, and mathematically required
   weight-normalization folds. Apple BF16-to-F32 staging, transposes, repacks,
   alignment copies, and any re-laid weight buffer are forbidden compatibility
   materialization, not derived storage. `compatibility_copied_bytes` must read
   0 in production.
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

- The whole checkpoint is read into one page-aligned virtual-memory allocation
  (`mmap` / `VirtualAlloc`), each shard base 64-aligned, tensors packed gapless
  (`safetensors.cpp`). Publication changes the complete allocation to read-only.
- `fill_view` hands out a pure pointer plus metadata:
  `view.data = storage.data() + tensor.offset`, with `shape`, `elements`,
  `bytes`, `rank`, `dtype`, `shard`. **No copy. No conversion. bf16 stays bf16.**
- The safetensors on-disk payload *is* the raw little-endian tensor bytes,
  contiguous from the data section. So a byte-exact load is literally "seek to
  `offset`, read `byte_count`, done"; a view is "base + offset, interpret at
  `dtype` width." Row-major, so per-dim strides are derived from the element
  width — there is no stride table to store.

Production performs no whole-weight conversion. Architecture leaves load
possibly unaligned little-endian BF16 words, shift them into the high half of an
f32 register value, and accumulate in registers. The deleted compatibility path
used `CandleBridge` and Apple F32 RHS staging; neither is an admissible fallback.

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
chunks across a 4-thread pool. Its reusable-buffer streamer and byte throttle
belong to a staged-copy design and are deliberately not borrowed here; only the
parallel positioned-read technique transfers (§3). Note the ember-ml tree
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

The loader now opens and fingerprints every shard before allocating the image,
then slices each shard into 8 MiB tasks consumed by at most four transient I/O
workers. POSIX uses retrying `pread`; Windows uses positioned overlapped
`ReadFile`. Every task lands **directly into its disjoint final image slice**.
There is no staging buffer, per-chunk payload allocation, or application copy.

- Worker count is the complete concurrency bound. A byte throttle would not
  reduce resident RAM because tasks borrow the already allocated destination;
  it would only serialize I/O.
- All workers are joined before an error can unwind the image. The loader then
  re-stats the same open handles and reports the lowest source/offset failure
  deterministically.
- Only inter-source and trailing alignment padding is zeroed. Every source byte,
  including its safetensors header, remains byte-exact.
- `LfmWeightLoadStatsV1` publishes complete source bytes (excluding padding),
  aligned resident bytes, and the actual task and worker counts for model-level
  memory/load accounting; it is a transitional native C surface, not a Rust
  tensor API.
- Alignment is a non-issue for correctness: `pread` writes exact bytes at exact
  offsets; the 64-byte base alignment of the image is preserved because chunks
  are placed by absolute offset.
- This stays entirely on the **byte-exact** side of ember-ml's pipe — cut before
  `tensor_ingress`; there is no descriptor/mailbox/worker needed for a synchronous
  positioned read into an existing buffer.

*Gate:* `parallel_read_is_byte_exact_across_chunks_and_zeroes_only_padding`
crosses both the 8 and 16 MiB task boundaries, compares complete source slices
and tensor payloads byte-for-byte, and verifies deterministic padding. Real
checkpoint cold/warm wall-clock and throughput remain the hardware gate.

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
3. **Direct small-M reuse is landed.** Prefill/suffix chunks of up to four rows
   use checkpoint-layout BF16 kernels that load each weight vector once for the
   active row group. Apple and x86/Rosetta create no `gemm_amx_*` widening plane,
   one-time f32 shadow, pack, or transpose. Further tuning is a measured kernel
   scheduling question, not permission to restore weight staging.
4. **Prefetch-distance / stream-count tuning** on the existing leaf — cheap,
   measurable, no rewrite.

The discipline: **profile the decode bandwidth ledger per stage, attribute the
missing ~180 GB/s, then apply the specific lever the profile names.** No blind
asm rewrite of a leaf that already declares itself bandwidth-bound. Widen the
accumulator (double-double) where dynamic range needs it, never the storage —
weights stay bf16 (this is the one genuinely useful HOLOFLOAT principle:
`GEMV_BF16_DD` reads bf16 directly, only the accumulator widens).

### 3c. Coordination: causal edges first, optional hardware dormancy second

**The load-bearing invariant:** an operation never owns a waiter. Its ticket,
route label, phase, leases, and scratch identity are durable records. A producer
publishes the changed predicate and makes the retained continuation runnable;
the exact CQ supplies addressed completion and epoch correlation. No timer,
sleep, bounded spin, or repeatedly yielding coroutine can cause progress.

Only resident execution capacity whose complete ready predicate is empty may
become dormant. The worker observes the shared doorbell generation, rechecks
every ready predicate, and uses the private deadline-free address-dormancy
adapter only if nothing is runnable. A publication that races this sequence
either changes the observed generation or wakes the dormant worker. This
adapter is not attached to the ticket and is never called at a numerical stage
boundary.

`wfe`/`sev` is therefore, at most, an implementation experiment for the fixed
team's **idle dispatch doorbell**. It cannot replace the shared predicate or the
CQ. It is admitted only if a guarded startup probe and profile show lower
latency/power than `os_sync_wait_on_address`/futex without polling or lost edges.
Rosetta and unsupported hosts keep the OS backend. Broadcast `sev` is unsuitable
for per-conversation completion routing; ticket publication remains addressed.
ember-ml's producer-side `sched_yield` loop and drain-to-empty completion model
remain rejected.

## 4. The ember-ml borrow ledger (explicit)

**Take (all byte-exact / format-agnostic):**

| Idea | Where it lands here |
|---|---|
| Parallel positioned `pread` into the final buffer, bounded by worker count | §3a — the load win |
| Host-as-bus / persistent resident kernel owns compute | already the flashkern lane team |
| Register-resident hot tile, explicit memory-crossing only | already the leaves; sharpen with hand-asm §3b |
| Stream → consume → drop for load (pointer walk, bounded scratch) | §3a + the resident-image binding (P1) |
| Widen the accumulator, not the storage (dd over bf16) | §3b, item under the list |
| `wfe`/`sev` event backend | §3c — idle team dormancy only, optional and measured |

**Reject (numerical transform / weaker than what we have):**

| Idea | Why rejected |
|---|---|
| HOLOFLOAT rotor re-encoding of weights | Not byte-exact; redefines the arithmetic. Our law: bf16 bytes untouched. |
| Möbius group-law "matmul" | Computes a different function than the bf16 dot product. |
| `from_holo`/voxel array sink | Non-materializable — throws on raw pointer access. Antithetical to weight views. |
| Drain-to-empty completion (no CQ) | Our SQ/**CQ** with epoch routing is strictly more capable. |
| Producer busy-spin on flush | Producers publish retained ready records and return; only idle kernel capacity may become dormant. |
| In-flight-bytes load throttle | Direct reads borrow disjoint spans of the already-final image; worker count is the complete bound, so a byte throttle only serializes I/O. |
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
