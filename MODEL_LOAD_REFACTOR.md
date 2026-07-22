# MODEL_LOAD_REFACTOR

Spec and running journal for moving model weight storage to a wired, cross-process
shared segment, and for the larger boundary decision it belongs to. Treat the top
half as the contract; the journal at the bottom records what actually happened,
by whom, with results. Entries are append-only.

**Status:** design accepted, amended per Sol's review (E2) · all spikes landed (E3, E5, E6) — every design assumption measured, none refuted · implementation awaiting go
**Scope owner:** Sydney · **Engineering:** Vera (+ delegated agents, logged per entry)
**Started:** 2026-07-21

---

## 1. The decision this work lives under

The model runtime — loading, inference, capture, VAD, encode, decode, playback —
lives **entirely in C++/assembly**. This was a hard choice made after the mixed
Rust/FFI arrangement kept violating the design principles (Rust-side heap copies,
cooperative-kernel confusion, ownership split across the FFI line).

Rules that follow from it:

1. **Rust is not allowed to load anything into the Rust heap.** No weight bytes,
   no activation state, no PCM ownership. Rust holds handles and speaks protocol.
2. **Part 1 (this document): weight storage.** Native-only work. We do not touch
   the Rust runtime path at all.
3. **Part 2 (recorded here, specced separately): the native runtime runs
   standalone** — microphone, VAD/Sesame, encode, decode, speaker — like any
   model server. Finished chunks stream out in order. The desktop app receives
   them through `kcoro-rs`, which speaks kcoro-native so coroutine token passing
   is fluid across the boundary. Transport is IPC (mechanism TBD: UDS message
   channel vs shared-memory ring; the semantic contract — ordered chunks,
   backpressure by coroutine parking — is fixed regardless).
4. **Licensing boundary (direction, not commitment):** the native core stays
   BSD-licensed. Because the boundary is IPC, a GPL'd consumer or kernel-side
   component (eBPF experiments, etc.) can sit on the other side without license
   contamination in either direction.
5. **Don't get distracted by Rust.** The Rust cleanup is a second pass after the
   native piece runs standalone. The only Rust we touch in Part 1 is mechanical
   test-expectation maintenance, because today's native test harnesses happen to
   be Rust test binaries (`native_safetensors.rs` etc.). No new Rust
   functionality. A native standalone gate binary gets added so Part 2 can drop
   the Rust harness dependency entirely.

---

## 2. Part 1 requirements (weight storage)

R1. **Zero byte transformation.** Not one byte of checkpoint data is converted,
    swapped, or re-laid-out between disk and the resident image. (Doctrine;
    already true today — verified, see §3.)
R2. **Wired.** Weights are hot, active memory: never evicted, never faulted
    mid-decode. On every supported OS, user space only gets virtual memory; the
    lever the OS provides is wiring (`mlock` / `VirtualLock`). Wiring failure is
    a **loud hard error** carrying the exact remediation (ulimit/sysctl text) —
    never a silent unwired fallback.
R3. **Shared across processes.** Multiple processes attach to the same physical
    pages. One machine-wide copy, ever. This is also what makes multiple model
    instances ~free: marginal cost of another process = page tables + its own
    derived tables/plans.
R4. **GPU-readable by pointer.** The segment must satisfy Metal's
    `newBufferWithBytesNoCopy` contract (page-aligned base, page-multiple
    length, mmap-class memory) so a struct/pointer handoff gives the GPU the
    weights with zero copy. Proven by spike before implementation (journal E2).
R5. **Tracked.** A process-global registry keyed on checkpoint identity ensures
    a second open in the same process attaches to the same image object. Across
    processes, the segment name itself is the registry.
R6. **Per-OS page granularity compensated once, in the layout.** One alignment
    constant (64 KiB) covers 4 K/16 K Linux, 16 K Apple Silicon, and Windows'
    64 KiB section granularity. No per-OS layout code.

---

## 3. Ground truth (audited 2026-07-21, adversarially verified)

Eight-agent audit over the loader, view binding, call sites, and lifecycle.
Every claim below was re-derived by an independent verifier; corrections were
phrasing-level only. File references are the anchor points for the change.

**Loader** (`crates/liquid-audio/native/src/io/safetensors.cpp`):
- The image is one anonymous private allocation: `mmap(PROT_READ|PROT_WRITE,
  MAP_PRIVATE|MAP_ANONYMOUS)` in `AlignedBytes` (line ~133; Windows twin
  `VirtualAlloc` ~129). Page-rounded; logical size is the 64-byte-aligned
  (`kWeightAlign=64`, `checked_align`) concatenation of all source files.
- Writes into the image, exhaustively: (a) verbatim `pread` of whole source
  files (8 MiB tasks, ≤4 workers, `read_sources` ~583/643); (b) `memset(0)` of
  the inter-source alignment gaps and page-rounding tail (~1087/1092). Nothing
  else. R1 holds today.
- Sealed once with `mprotect(PROT_READ)` (~1128); torn down by `munmap` via
  `lfm_weights_close` → `delete` (~1362).
- Component offsets live per-`Source` `{offset, component}` and per-tensor
  (`TensorMeta.offset` absolute into the image); per-component name maps on the
  image.
- **No cache, registry, or dedup of any kind.** Two opens of the same paths
  build two full images. A test locks the current behavior in:
  `concurrent_opens_publish_independent_complete_images` asserts 8 concurrent
  opens produce 8 independent images. Changing that is deliberate, not drift.

**View binding** (verified across model/conformer/detokenizer/engine):
- After plan bind, **every consumer holds absolute per-tensor pointers**
  captured once (`fill_view`: `view.data = storage.data() + tensor.offset`,
  ~1138). No kernel or assembly leaf takes an image base. BF16 leaves tolerate
  byte-unaligned starts; F32 views need 4-byte alignment, which in-file
  safetensors offsets under any page-aligned mapping preserve.
- The single-VA-range assumption exists only inside the loader itself, in two
  C-ABI accessors product code never calls (`lfm_weights_data`,
  `lfm_weights_resident_bytes`), and in tests/bench walkers. The wired segment
  keeps single-range layout anyway, so even those survive with re-specced
  expectations.

**Call sites:**
- Stack: `lfm_weights_open*` ← `lfm_model_open` ← `lfm_runtime_model_open`
  (one model slot per runtime) ← `NativeVoiceModel::open_with_config` (fresh
  runtime per call) ← `resident_lfm2` (desktop).
- The only dedup anywhere: the desktop-local static `LFM2_RESIDENT`
  (`packages/desktop/src-tauri/src/voice/runtime.rs:219`) — path-keyed,
  single-flight, never evicts. The `liquid-audio` crate itself has none.
- Multi-process weight use does not exist in the product today (sidecar is the
  TS agent server; single-instance plugin blocks a second app). Cheapest real
  double-load: the running app + any `LFM_MODEL_DIR`-gated test/bench process.
- Quirk for the record: within one bundle open, main==detokenizer path equality
  would read the same file into the image twice.

**Identity & accounting:**
- The loader already `fstat`s every shard and holds
  `FileState{size, dev, ino, mtime, ctime}` (re-verified after reads). **No
  content hashing exists anywhere.** Registry identity comes free from these
  tuples.
- `lfm_model_memory` / `lfm_weights_load_stats` report `resident_image_bytes`,
  `payload_read_calls/bytes`, `load_ns/workers/tasks`; tests assert exact
  equations (`payload_read_calls == load_tasks + 2`,
  `payload_read_bytes == source_bytes + config + index`);
  `bench_native_load` computes GiB/s from `source_bytes/time`. All of these
  change meaning under attach-vs-build and are re-specced in §6, not loosened.
- Lock hazard to respect: `runtime->children_mutex` is held across the entire
  multi-GB load (open at `voice_session.cpp:5392`, close at 5426). Registry
  locking must never call back up into runtime/model APIs.
- Parked observations (not this refactor): `ModelOwner::drop` has a second
  leak path (join refusal skips destroy); the children_mutex-across-load is a
  UI-latency landmine on its own.

**Current sizes** (LFM2.5 main + detokenizer):
main file 2,940,723,992 B · detokenizer file 314,137,512 B · today's image
3,254,861,568 B logical / 3,254,861,824 B page-rounded. Segment overhead added
by this design (64 KiB header + 64 KiB source alignment) is ~100–200 KB —
computed by the same `checked_align` concatenation at build time, never by
hand.

---

## 4. Design: the wired weight segment

### 4.1 Object and naming

One named POSIX shared-memory segment per checkpoint set
(Windows: named `CreateFileMapping` section; same layout).

- Name: `/lfm-<hash>` where `<hash>` is a short (≤24 hex chars — macOS caps shm
  names at 31 bytes, spike E2 verifies) digest of: layout version ‖ ordered
  per-source `FileState{dev, ino, size, mtime, ctime}` tuples. No file content
  is read for identity; the tuples already exist at open (§3). A checkpoint
  edit changes mtime → different name → different segment; stale segments are
  garbage, removed by the evict command.
- The segment **object persists after the last process exits** — attach after an
  app restart is microseconds (measured: 8 µs, E3) and never re-reads the
  checkpoint files. But **wiring does not persist**: `mlock` is a property of a
  mapping, and the last exit takes the last mapping with it. Unwired shm pages
  stay allocated (they are anonymous — they can never be dropped to disk) but
  become eligible for compression/swap under memory pressure. So host-less
  restart-warm is fast and *usually* resident (E3 measured 12.46 GB/s post-exit
  reads on a quiet machine), while the absolute "never evictable, ever"
  guarantee comes only from the model host (§4.6) holding a locked mapping for
  the machine's lifetime. Correction credited to Sol's review (E2).
  RAM is reclaimed only by explicit evict (`lfm_weights_evict` / CLI verb) or
  reboot.

### 4.2 Layout

```
offset 0        header block (64 KiB reserved; one page used)
offset 64Ki     source 0 bytes, verbatim            (main checkpoint)
align 64Ki      source 1 bytes, verbatim            (audio detokenizer)
...
align 64Ki      end (segment size page/granule-rounded)
```

- Inter-source alignment: `kWeightAlign` goes 64 B → **65,536 B** (R6). Same
  running `checked_align` concatenation as today, one constant changed, plus
  the leading header block.
- Alignment gaps and tail are zeroed by the builder, exactly like today's two
  `memset`s — still the only non-file bytes in the segment (R1).
- Header contents: magic, layout version, identity hash (file identity: the
  fstat-tuple digest that also names the segment), **content digest** (payload +
  layout, computed by the builder *while streaming* — the bytes pass through its
  hands anyway, so content identity costs no extra I/O), total size, source
  count and table `{offset, bytes, component, label hash}`, generation counter,
  build state (`BUILDING → READY | POISONED` — C11 atomics, cross-process on
  cache-coherent hardware), owner identity (builder pid + process start time,
  so liveness checks survive pid reuse), and build stats (ns, workers, tasks)
  so attachers can report honest provenance.
- File identity vs content identity are distinct on purpose: the fstat tuples
  are a *provisional lookup key* (cheap, no reads); `READY` binds the segment to
  a *verified content digest*. An attacher trusts bytes only after magic,
  version, size-vs-fstat cross-check, full header bounds validation, owner uid
  check (`fstat().st_uid == geteuid()`), and digest presence all pass. A
  same-name object that fails any of these is stale-or-hostile: hard error,
  never mapped into a model.

### 4.3 Build / attach protocol

- **Election:** `shm_open(name, O_CREAT|O_EXCL)`. Winner is the builder; EEXIST
  means attach.
- **Builder:** `ftruncate` to the exact final size immediately — macOS shm is
  ftruncate-once, and E3 proved it returns EINVAL even at the *same* size, so
  there is no idempotent re-truncate; the size is set exactly once by the
  creator. Then `mmap(PROT_READ|PROT_WRITE, MAP_SHARED)`, zero the gaps, stream
  file bytes with the **existing 4-worker positioned-read loop unchanged**
  (destination base changes, nothing else), hashing the stream into the content
  digest as it goes, validate shards exactly as today, write header, `READY`
  release-store on the generation, then drop its RW mapping and remap
  `PROT_READ`. Sealing semantics: attachers can only ever map read-only; the
  builder's RW window is bounded and private to the build.
- **Attacher:** `shm_open(name, O_RDONLY)` → `mmap(PROT_READ, MAP_SHARED)` →
  run the full §4.2 validation ladder. If `BUILDING`: the attacher **suspends
  its continuation**; builder completion publishes a correlated readiness edge
  that resumes it — no polling loop, no thread parked beside a flag (the same
  contract as issue #148's Condvar replacement; once the model host exists the
  edge is simply an IPC completion). Crash takeover must distinguish an
  *actively building owner* from an *abandoned generation* before unlinking
  anything: owner liveness = pid + start-time match (a pid alone is reusable),
  and the generation stamp must match the one observed at attach. Only a
  proven-abandoned `BUILDING` generation may be transitioned to `POISONED` and
  unlinked; then re-elect. A half-built corpse can never be attached: `READY`
  is written after full validation, digest mismatch fails the attach loudly,
  and `POISONED` is terminal for its generation. No fallback
  rebuild-in-private; hard error paths stay hard.
- **Wire:** every process (builder and attachers) `mlock`s its mapping (R2).
  The kernel refcounts page wiring; the pages stay wired while any living
  process holds them locked. `mlock` failure text names the exact limit that
  refused (`RLIMIT_MEMLOCK` / `vm.user_wire_limit` / `vm.global_user_wire_limit`
  — spike E2 measures which bind on macOS 27) and the fix. No unwired
  operation.
- **Detach:** `lfm_weights_close` becomes detach: `munlock`+`munmap`+refcount
  down in the in-process registry. Never unlinks.
- **Evict:** explicit `lfm_weights_evict(identity)` → `shm_unlink`. Mapped
  processes keep their pages (POSIX semantics); new opens rebuild.

### 4.4 SegmentLease and the in-process registry (R5)

The unit of ownership is the **SegmentLease**: identity + generation + mapping
+ wire lifetime. Everything that needs the bytes alive holds a lease; the
mapping and its `mlock` live exactly as long as the lease count is nonzero.

```
Native model host (§4.6)
  └── SegmentLease: identity + generation + mapping + wire lifetime
       ├── LfmWeightImage
       │    └── validated immutable WeightViews
       └── retained Metal buffer (releases its lease from the no-copy
           deallocator — verified E6: Metal fires it only after every
           command buffer using the buffer has completed (+5 µs measured),
           but on an ARBITRARY thread, so the release hook must be
           thread-safe and assume nothing about its calling context)
```

Everything downstream receives byte views, offsets, shapes, and strides.
Nothing downstream receives a loader, a tensor object, a conversion buffer, or
Rust-owned model memory.

The process-global registry lives in the weights layer (the audit confirmed
this is the only correct granularity: `LfmModel` can't be shared across
runtimes because plans bind per-runtime engine ids; the **image** can):
identity hash → refcounted lease over one `LfmWeightImage`. Open twice → same
object, lease++. Own mutex, leaf-level: never calls up into runtime/model code
(respects the `children_mutex` lock order, §3). Eviction may `shm_unlink` the
name, but can never reclaim live mappings — POSIX guarantees mapped pages
survive the unlink, and the lease guarantees we never munmap under a live
consumer, GPU included. The desktop `LFM2_RESIDENT` cache remains as
object-level dedup above; it just stops being the only thing standing between
us and a duplicate 3 GiB.

### 4.5 GPU handoff (R4)

Segment base is page-aligned and granule-multiple by construction — exactly the
`newBufferWithBytesNoCopy` contract. One `MTLBuffer` wraps the whole segment;
every tensor binds as buffer + offset, the same offsets the CPU views carry.
Nothing is built against Metal in Part 1; the spikes prove the design is
GPU-safe for when we want it. **Proven on the M2 Max (E3):** a no-copy buffer
over an shm mapping is genuinely zero-copy (`buffer.contents == mmap base`),
and the GPU checksum matches the CPU over both the RW mapping and — the
attacher case — a `PROT_READ`-only mapping. **Proven (E6):**
`device.maxBufferLength` = 18.72 GiB on the M2 Max, ~6× headroom over the full
3.03 GiB image as one lazy no-copy buffer; the deallocator fires only after
command-buffer completion (lease release from it is safe, but it arrives on an
arbitrary thread); and `shm_unlink` during in-flight GPU work is harmless —
name dies instantly, mapping and checksums untouched, which is exactly the
evict semantics §4.4 promises. (For the record: Apple GPUs translate addresses — unified memory
maps into the GPU's address space — but the driver wires what the GPU touches;
the property that matters, pointer handoff with no copy and no fault, is the
one proven.)

### 4.6 The native model host (keeper)

The refinement that makes the guarantees whole: not merely a shared allocation,
but a **persistent native model service**. A long-lived native process holds a
locked mapping of the segment for the machine's lifetime — the only thing that
makes R2 true *between* client processes, because wiring is per-mapping (§4.1)
and someone must keep a mapping alive. The host owns:

- durable wiring (the keeper lease that outlives every client),
- build election and the build itself (with a host present, O_EXCL election
  degenerates to "ask the host"),
- readiness edges over IPC (an attach request parks the client's continuation;
  the host's completion message is the correlated edge — kcoro-native
  semantics across the process boundary),
- lease bookkeeping, including GPU buffer retention.

The desktop app becomes a client of this service. This is the bridge into
Part 2 (§8): the host *is* the standalone native runtime's first
responsibility, grown from the gate binary (§7 step 5). Host-less operation
(library mode) remains fully supported for tests and single-process use —
same segment, same protocol, per-process wiring only, and honest about what
that mode does not guarantee.

### 4.7 What deliberately does NOT change

- View math: `image_base + offset` single-range addressing, `fill_view`, all
  absolute per-tensor pointers, every kernel and assembly leaf. Untouched.
- The positioned-read worker loader (it just writes into the segment).
- Shard parsing/validation, the safetensors index cross-check.
- `materialized_weight_bytes == 0` doctrine and its acceptance gate.

---

## 5. C ABI evolution

| Symbol | Today | After |
|---|---|---|
| `lfm_weights_open*` | always builds a private image | attach-or-build on the named segment; identical signature |
| `lfm_weights_close` | munmap + delete | detach (refcount, munlock/munmap); segment persists |
| `lfm_weights_evict` | — | new: unlink segment by identity/path |
| `lfm_weights_load_stats` | source/resident/tasks/workers | five-way taxonomy (§6.3): `source_bytes` / `segment_constructed_bytes` / `attached_shared_bytes` / `wired_bytes` / process-attributed RSS, plus `attached`, `build_ns` vs `attach_ns`, generation + content digest |
| `lfm_weights_data` / `_resident_bytes` | single-span accessors (tests only) | keep, spanning the whole segment (single-range layout preserved) |

Open options gain: wire policy (`require_wired` is the default and only mode —
the field exists so a test harness can *measure* the unwired case explicitly,
never so production can fall back), build worker count (existing).

---

## 6. Deliberate breakage — re-specced, not loosened

Every consumer of load accounting gets explicit build-path vs attach-path
expectations:

1. `concurrent_opens_publish_independent_complete_images` → becomes the
   single-flight test: N concurrent opens = 1 build + N−1 attaches, all views
   byte-identical, publication-safe (READY ordering observed).
2. Exact equations split: build path keeps
   `payload_read_calls == load_tasks + 2` etc.; attach path asserts
   `payload_read_calls == small-const` (header/index verification reads only)
   and `payload_read_bytes == 0` for tensor payload.
3. Accounting becomes a five-way taxonomy, not one blurry number:
   `source_bytes` (on disk), `segment_constructed_bytes` (built by this
   process), `attached_shared_bytes` (mapped, built elsewhere), `wired_bytes`
   (locked by this process), and process-attributed RSS (reported, never
   summed across processes). **Shared pages must never be reported as a fresh
   3 GiB allocation by every model instance.** `lfm_model_memory` consumers
   (`validate_voice_model`, speech-gate before/after equality, Rust `memory()`
   surface) updated to the taxonomy.
4. `bench_native_load` labels build vs attach runs. Build GiB/s remains
   meaningful only with the existing `uncached`/`F_NOCACHE` option; attach runs
   report attach latency instead. Without this the bench silently inverts into
   a page-cache benchmark. (House rule: attack your own benchmark setup first.)
5. Fork-safety/`MAP_PRIVATE` semantics tests re-pointed at `MAP_SHARED`
   read-only expectations.

---

## 7. Build order and acceptance gates

0. **Spikes**: core results landed (E3) — shm-backed `newBufferWithBytesNoCopy`
   incl. the read-only attacher mapping with matching GPU checksums;
   `mlock(3.2 GiB)` in 0.176 s under default limits; ftruncate-once (EINVAL
   even at the same size); 31-char name cap; survives-builder-exit attach at
   8 µs. Follow-ups in flight (E4): `maxBufferLength` vs the full segment,
   deallocator/retirement ordering, unlink during in-flight GPU work, and the
   wire-drop-after-last-exit staging that demonstrates the keeper requirement.
   All landed (E5, E6): every design assumption measured, none refuted. The
   spike gate is cleared.
1. **Segment allocator**: `AlignedBytes` → `WeightSegment` (named, MAP_SHARED,
   header, election, wire). Same `load()` flow around it.
2. **Registry + refcounted attach** in the weights layer.
3. **Stats/ABI split** (build vs attach, wired_bytes) + evict entry point.
4. **Test re-spec** per §6.
5. **Standalone gate binary** (native C++ CLI): attach-or-build, run the
   wav-hash parity oracle, print stats. First brick of the Part 2 standalone
   runtime; removes the Rust-harness dependency for native verification.
6. **Model host** (§4.6): the gate binary grows a daemon mode — keeper lease,
   build service, readiness edges over IPC. Desktop cutover to client mode is
   Part 2 work, but the host lands here so G4b can gate.

Gates (all must hold before this document's status flips to "landed"):

- G1. Fixed-seed wav-hash byte-identity: same seed, same machine, before vs
  after the refactor. Full-length comparison — length equality asserted first,
  never truncated.
- G2. `materialized_weight_bytes == 0` unchanged.
- G3. 8-way concurrent open: exactly 1 build, 7 attaches, byte-identical views.
- G4a. Host-less restart-warm: builder process exits; fresh process attaches
  and passes G1 without touching the checkpoint files.
- G4b. Keeper residency: with the host holding its lease, host wired-page
  count stays ≈ segment size across arbitrary client attach/exit cycles. The
  host-less converse — wiring drops after the last locker exits even though
  the segment survives — is demonstrated, not papered over.
- G5. Wire verification: host wired-page delta ≈ segment size after first
  attach; `mlock` failure path produces the prescribed error text (tested via
  an artificially lowered limit, through the harness seam — not an env var).
- G6. Two-process concurrency: builder + attacher live simultaneously, both
  decode, one physical copy (wired delta counted once).
- G7. Hostile/stale same-name object (wrong magic, wrong size, wrong uid,
  truncated header, `BUILDING` with a dead owner) is rejected on every path
  with the prescribed error; the `POISONED` transition is observed exactly
  once per abandoned generation.

---

## 8. Part 2 horizon (recorded, specced separately)

Standalone native runtime: capture → Sesame/VAD → turn commit → encode →
decode → detokenize → playback, all in one native process, no Rust in the loop.
The model host (§4.6) is this process's first-born responsibility — the
architectural refinement out of Sol's review is that the endpoint is not a
shared allocation but a **persistent native model service** whose immutable
image is attached by CPU and GPU views alike, with the desktop as one client
among possibly many. Outputs are ordered finished chunks over IPC; `kcoro-rs` receives them speaking
kcoro-native, so token passing suspends/resumes across the boundary the same
way it does inside the C++ runtime (channel ops in coroutine context; parking,
not spinning). The desktop app becomes a pure consumer. Rust cleanup — removing
the in-process FFI ownership chain (`ModelOwner`/`ConversationOwner` down
through `lfm_runtime_*`) — happens only after the standalone gate binary passes
G1–G6 and speaks the stream.

---

## 9. Journal

Append-only. Newest last. Author + date on every entry; delegated agent work is
logged by the delegator with the agent's role named.

### E0 — 2026-07-21 — Vera — Decision + audit
- Sydney set the boundary: model runtime entirely C++/assembly; Rust barred
  from loading into Rust heap; native piece must run standalone before any Rust
  work; weight storage is Part 1.
- Ran an 8-agent audit (4 readers × 4 adversarial verifiers) over loader, view
  binding, call sites, lifecycle. All key claims verified; §3 is the distilled
  result. Full structured findings preserved in the session workflow record.
- Design accepted by Sydney: wired named shared segment (§4), after explicit
  rejection of two alternatives: file-backed mmap (fails R2's determinism
  bar — clean-page eviction — and Sydney's requirement of hot active memory)
  and Mach memory-entry passing (needs bespoke IPC; wrong tool for an
  untransformed read-only artifact).

### E1 — 2026-07-21 — Vera — Spikes launched
- Spike 1 (delegated, hardware agent): Metal `newBufferWithBytesNoCopy` over an
  shm-backed mapping on the M2 Max, including the read-only attacher mapping;
  GPU checksum vs CPU; ftruncate-once; 31-byte name cap; O_EXCL election.
- Spike 2 (delegated, hardware agent): full-size (3,254,861,824 B) segment
  fill + `mlock` under default macOS 27 limits with wired-page accounting;
  builder-exit → fresh-process attach with pattern verification and post-exit
  read bandwidth; attacher-side mlock; unlink discipline.
- Results: pending — recorded when the agents report. [They landed as E3;
  E2 became Sol's review, which arrived first.]

### E2 — 2026-07-21 — Vera — Sol's review; contracts adopted
Sol reviewed from outside the seam (his lane stays downstream of
`LfmWeightImage`: conversation/activation arenas, coroutine restoration, PCM
docks, E2E gates; this loader/residency seam is exclusively this workstream).
Adopted in full:
- **The crucial correction:** shm objects survive process exit; **wiring does
  not** — `mlock` dies with the last mapping. "Wired across restarts"
  therefore requires the persistent model host, now §4.6 and build step 6
  with its own gate (G4b). §4.1's restart-warm claim was demoted from
  guarantee to happy-path.
- File identity ≠ content identity: fstat tuples are the provisional key;
  `READY` binds a builder-computed content digest (§4.2).
- `BUILDING → READY | POISONED` generations; readiness is a correlated
  continuation edge, never a polled flag (§4.3). Same contract as open issue
  #148 ("voice: replace resident model-load Condvar with a correlated
  readiness continuation") — the segment's readiness publication resolves its
  substance.
- Crash takeover distinguishes an active builder (pid + start time,
  generation stamp) from an abandoned generation before unlinking (§4.3).
- Metal buffers retain the SegmentLease until command-buffer retirement; the
  no-copy deallocator is the release hook; evict unlinks names, never
  reclaims live mappings (§4.4/§4.5).
- Security ladder for stale/hostile same-name objects (§4.2, G7); five-way
  accounting taxonomy (§6.3) — shared pages never reported as a fresh 3 GiB
  per instance.
Lane cross-references: open issues #168 ("replace per-conversation vector
forest with one sealed arena and compact RoPE state") and #169 ("owner-scoped
high-water accounting, shrink activation/session arenas") stay with Sol —
they subsume the Conformer/KV/RoPE fix thread investigated earlier today
(ownership move proven safe: the engine bridge runs one pass program
end-to-end, so a single model-owned Conformer workspace cannot be raced).
That investigation hands off to Sol's lane rather than resuming here.

### E3 — 2026-07-21 — Vera — Core spike results (both landed, delegated hardware agents)
Spike 1 — Metal over shm (M2 Max, macOS 27, 16 KiB pages), 7/7 probes PASS:
- `newBufferWithBytesNoCopy` over an shm RW mapping: non-nil,
  `buffer.contents == mmap base` (genuinely zero-copy); GPU checksum == CPU
  checksum (137,422,176,256, independently verified against the closed form).
- **Read-only attacher mapping accepted by Metal**; GPU read correct. Two
  mappings of one fd get distinct VAs; each wraps into its own MTLBuffer
  aliasing the same physical pages.
- ftruncate-once: **EINVAL even at the identical size** — no idempotent init;
  size is set exactly once by the creator.
- 40-char name → ENAMETOOLONG (31-char cap confirmed); second O_CREAT|O_EXCL
  → EEXIST (election is reliable). Segment verified gone after unlink.
Spike 2 — full-size wiring + cross-process, all green under default limits:
- This machine: RLIMIT_MEMLOCK infinite by default; `vm.user_wire_limit` =
  `vm.global_user_wire_limit` = 25.28 GiB; `hw.memsize` = 32 GiB; page 16 KiB.
- Build: fill 10.78 GB/s; `mlock(3,254,861,824)` **succeeded in 0.176 s**,
  host wired delta +3,077.5 MiB (~99% of segment; remainder is machine noise).
- Builder exited; fresh process attached in **8.0 µs**; 1000/1000 fixed-seed
  content probes pass; post-exit sequential read **12.46 GB/s** — pages stayed
  resident on a quiet machine (happy path; per E2, not a guarantee).
- `mlock` on the read-only attacher mapping succeeded (0.117 s). Clean unlink,
  nothing left behind.
Folded into the contract: exact-size-once creation, ≤31-char names, the
RO-mapping GPU path, and the real wire-limit numbers for G5's error text.

### E4 — 2026-07-21 — Vera — Follow-up spikes launched (pending)
Extensions sent to both hardware agents after Sol's review:
- Spike 1: `device.maxBufferLength` and a full-size (3,254,861,824 B) no-copy
  buffer over a lazy segment; deallocator-vs-command-buffer-retirement
  ordering (validates the lease-release hook); `shm_unlink` during in-flight
  GPU work.
- Spike 2: wire_count staging table — baseline → build+mlock → keeper alive →
  client attach/exit → **keeper killed** (the direct demonstration of E2's
  correction: segment survives, wiring drops) → attacher mlock → exit →
  unlink.
Results append here when the agents report. Implementation stays gated on E4
(§7 step 0).

### E5 — 2026-07-21 — Vera — Wire-drop staging landed (spike 2 extension)
The direct demonstration of E2's correction, measured (segment 3104 MiB;
wired deltas vs baseline, noise budget ~100 MiB, worst deviation +73 MiB):

| Stage | Wired delta | Meaning |
|---|---:|---|
| keeper mlocked (zero-fill, no data fill) | +3109 MiB | wiring never-touched pages works; 0.240 s incl. fault-in |
| keeper alive | +3177 MiB | steady |
| no-mlock client attached, read, exited | +3171 MiB | clients don't disturb the keeper's wiring |
| **keeper killed, segment alive** | **+30 MiB** | **wiring died with the keeper; segment still openable** |
| RO attacher mlocked | +3099 MiB | any process can re-wire a read-only mapping (0.099 s) |
| attacher exited | −70 MiB | dropped again |
| after shm_unlink | −69 MiB | unlink itself moves nothing (POSIX confirmed) |

Conclusions bound into the contract: (a) the keeper is necessary and
sufficient for durable wiring — G4b is testable exactly as written; (b) a
keeper can wire the segment *before* the fill (zero-fill wiring works), so the
host may pre-wire at creation; (c) re-wiring after a keeper crash is a 0.1 s
recovery, worth encoding in the host's restart path.

### E6 — 2026-07-21 — Vera — Metal follow-ups landed (spike 1 extension); spike gate cleared
All three probes PASS on the M2 Max; GPU work verified in-flight by LCG spin
calibration (predicted vs observed runtimes within 3%), checksums verified
against bit-exact CPU replicas:
- **maxBufferLength = 20,100,448,256 B (18.72 GiB).** The full 3,254,861,824 B
  image (exactly 198,661 × 16 KiB pages) wraps as ONE lazy `PROT_READ` no-copy
  buffer with ~6× headroom. Creation does not fault pages in.
- **Deallocator ordering: safe.** With a command buffer in flight at release
  time, the deallocator fired **+5 µs after completion**, never before; with a
  completed command buffer it fires synchronously on the releasing thread.
  Design consequence (bound into §4.4): lease release from the deallocator is
  correct w.r.t. GPU lifetime, but arrives on an **arbitrary thread** — the
  hook must be thread-safe and context-free.
- **`shm_unlink` during 243 ms of in-flight GPU work: harmless.** Name gone
  instantly (ENOENT on re-open), command buffer completed with no error,
  GPU + CPU checksums match, CPU re-reads of the unlinked mapping fine. POSIX
  orphan semantics hold for the GPU path — §4.4's evict contract is real.
Every design assumption in §4 has now been measured on the target machine and
none was refuted. Implementation is unblocked pending review (§7 step 0
cleared; G-gates unchanged).
