# 14 — Whole-Chain Coroutine-Driven, Zero-Copy, Zero-Wait Inference

Status: **authoritative target, with the LFM2 ownership cutover implemented.**
The working tree now has the immutable combined main+codec image, exact typed
views, native frontend/Conformer/backbone/Depthformer/Mimi, native tokenizer and
recurrence, per-conversation state/rollover, fixed PCM leases, reliable ticketed
events, interruption epochs, and expected-value parking. Resample, frontend, and
whole-Conformer/adapter orchestration now enter Flashkern as one typed,
model-correlated fixed-team SQ/CQ pass over borrowed pointer spans and sealed
conversation workspaces. Its native coordinator caller still waits synchronously;
this pass-boundary cut is not completion-continuation recurrence. The engine now
has a capacity-2 SQ/CQ, two per-ticket request/scratch slots, and an exact-CQ
callback substrate that retains a packed generation/state lease for its exact
slot and can atomically resubmit that slot without Rust progress. Deterministic
tests cover callback-slot theft, stale-owner ABA, stop, and live-slot accounting.
The V2.0 safety subset is also landed: 128-byte Apple isolation covers hot
kcoro/engine/bridge/session/model-gate words and internal SQ/CQ storage cells
without changing ABI-v1 value alignment; request, layer,
and modality dispatch is closed; invalid worker/logical-lane geometry rejects;
and four physical workers reproduce the eight-way logical fold. The bounded
production audio branch now reserves playback before admission and retains one
exact slot across `TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE`; Mimi writes direct
at equal rate or feeds the route's native device-rate resampler, and the
coordinator publishes only after terminal slot release.
The bounded route holds exclusive SQ-producer authority and borrows three
pre-created immutable descriptors, so the CQ callback performs no mutex-taking
descriptor work or generic submission. That exclusivity is temporary migration
scaffolding, not a claim that the peer slot can run unrelated work yet.
The coordinator still makes one outward expected-value wait. Its fixed
conversation-owned result and stack callback state are a bounded bridge, not the
future pooled asynchronous route instance. Two `BLOCK4` domains, the ready
broker, reverse-order per-block CQs, event-register waits, pooled tickets, and
concurrent numerical passes remain open. The physical mic/speaker adapter also
still bridges the native dock into the legacy Rust `VoiceEvent` surface. The full Moshi port and physical
kcoro device dock are subsequent tranches; neither permits a Candle fallback.

This is the convergence target for specs 02, 03, 07, and 10 —
the picture they are each a slice of. It describes the end state where the entire
inference chain, microphone PCM to speaker PCM, runs as native passes on the
fixed Flashkern lane team, clocked by completion doorbells rather than a Rust
loop, over one zero-copy weight pool.

**Terminology is normative:** the production object is a **buffer view**:
pointer/base offset + byte count + dtype + dimensions + derived byte strides
over an existing buffer. Older text that says "tensor view" refers only to this
same record; the tensor term is deprecated because it too easily implies an
owner. There is no owning tensor object, framework allocation, or materialized
payload. Production data moves as retained buffer spans.

The load-bearing observation: **the substrate for this already exists.** Flashkern
is already a GPU-threadgroup engine — a fixed P-core lane team, atomic tile
claims, one dispatcher, one final-return quorum callback, and no spin tier.
The SQ/CQ bridge correlates exact tickets directly with fixed pass slots; it has
no generic descriptor registry or operation waiter. The safetensors loader
already demonstrates the required one-ingress-write discipline: a byte-exact
resident image with immutable views. Mimi's private folded arena is not the
model-image precedent and must not be generalized into a second weight pool.
At the design baseline, what was missing was not primitives but three ownership
cuts (retained here as historical rationale):

1. Rust drove recurrence by **blocking** on a **single-slot** pass.
2. The numerical pipeline was **Candle above the assembly leaves** (prefill, modality
   scatter, the token/frame loop, KV ownership).
3. The model was **resident twice** — a byte-exact native image *and* a ~2.94 GB
   Candle copy that the backbone/depthformer passes actually ran off of.

This document plans the collapse of all three.

---

## 0. The north star (job 1)

**Divorce Rust from inference entirely — no math, no memory allocation, no
threading — and move all of it into Flashkern, where it belongs.** That is job 1;
everything else is downstream of it.

After the migration, Rust does exactly two things with audio, and nothing else on
the inference path:

- **grab PCM from the microphone**, and
- **grab PCM from the model.**

There is no `.wav` generation anywhere, ever, as a principle — PCM is a live
stream, not a rendered file. (A `.wav` render is the tell of a TTS mindset; this
is an interleaved real-time model, not a text-to-speech renderer.)

The Rust audio dock uses the narrow `kcoro-sys` retained-service surface and
native PCM reservations, so **Rust std channels are absent** from that layer.
No `mpsc`, `crossbeam`, polling loop, or operation waiter is permitted. A device
callback publishes a bounded edge; that edge makes the fixed-owner continuation
runnable. Only a resident kernel worker with no runnable ticket may become
dormant.

---

## 1. Requirements

### 1.1 Functional

- Every stage of the chain runs as a lane-uniform native pass on the resident
  lane team: resample, mel, Conformer, adapter, prefill + modality scatter,
  backbone prefill, backbone decode, text sampling, Depthformer, Mimi decode.
- **Recurrence is native.** The token/frame loop is an eagerly submitted native
  route program: an exact pass completion advances its fixed per-ticket instance
  through a compact immutable forwarding table and enqueues the already-resolved
  next pass directly, without a host round-trip.
  This is the "device recurrence" row of the Flashkern GPU-equivalence table,
  realized.
- **Rust's only production roles** are the two docks and the observer:
  - fill a non-cloneable native capture reservation from the device callback,
  - drain a ticket/epoch-tagged native playback reservation at the final device
    callback,
  - submit control tickets (start turn, interrupt, configure),
  - observe transcript on the reliable event channel and telemetry on a lossy
    side channel.
  Rust never blocks on a numerical completion and never owns model state.
- Multiple conversations share one model image and one lane team; completions
  route to the correct session by conversation id and epoch.

### 1.2 Non-functional

- **Zero-copy after ingress.** Weights are bound from the single byte-exact
  resident image in checkpoint-native `(N,K)` bf16; tensor starts may be
  unaligned and kernels must accommodate that. The Candle duplicate is deleted.
  A weight is only `image base + byte offset + dtype + dimensions + byte
  strides`; no plan owns a numerical weight array. Fused leaf intermediates stay
  in registers, with the stack reserved for small bounded lane-local state. An
  activation is written to preallocated scratch only when it must survive a
  barrier, fan-out, recurrence step, continuation, or register-sized tile.
  Scratch spans alias after their last consumer. No stage materializes a
  `Tensor` or grows a numerical `std::vector` on the progress path.
- **No operation waiters.** No polling, bounded spin, or host thread represents
  suspended work. Producers publish durable records and make retained
  continuations runnable. Only otherwise-idle resident runtime/team workers may
  become dormant on one indefinite expected-value doorbell. The idle team stays
  under the existing `< 0.1%` CPU gate (`engine_idle_zero_spin`).
- **Real-time.** The pipeline overlaps stages, so wall-clock is the critical path,
  not the sum of stages. Per-frame Mimi decode (~14 ms) must keep pace with
  playback; backpressure leaves the route record dormant and releases the
  engine slot until a speaker-capacity edge re-admits it.
- **Faithful numerics.** bf16 bit-matched to the captured fixtures across every
  ported stage; `-ffp-contract=off`; Accelerate/AMX permitted for matmul-shaped
  stages on Apple. Seeded CSPRNG native; fixed-seed byte-identity per turn.

### 1.3 Constraints

- **No tensors in the production data plane** — buffers with pointers. Candle
  survives only as an *offline* capture/parity oracle, never wired into the
  shipped path.
- **No math, memory allocation, or threading in Rust for inference.** All three
  belong to Flashkern (see §0). Rust owns only the two PCM docks and control.
- **No `.wav` generation, ever.** PCM is a live stream out of the model; there is
  no file-render step on the audio path.
- **Rust channels are dumped** at the audio dock in favour of bounded records
  and retained kcoro services resumed by callback/capacity/control edges.
- **No fallback chains.** A native gate failure is a terminal completion with a
  cause, not a silent drop to Candle. (This is a real sequencing constraint —
  see Trade-off 4.)
- Rust host; C++ owns plans, sessions, and recurrence; assembly owns all math.
- Target hardware: M2 Max — 8 performance-core lanes (E-cores excluded by
  policy), 400 GB/s, bandwidth-bound at decode.

---

## 2. High-level design

The whole chain is **one native session state machine eagerly executing a compact,
model-owned forwarding table on the Flashkern lane domains.** Each accepted
command owns a fixed per-ticket route instance. Exact completions advance that
instance; they do not invoke an allocated list of callbacks. The table is a
closed direct-threaded continuation map, not a DAG VM, graph compiler, lazy
evaluator, or public model bytecode.

### 2.1 Three planes (do not merge them)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PLANE 1 — native model SQ/CQ (compute)                              │
  │   fixed lane team · exact ticket/slot · final-return callbacks       │
  │   THE progress path. exact-once completions. no host on it.          │
  ├─────────────────────────────────────────────────────────────────────┤
  │ PLANE 2 — PCM / control dock (I/O)                                   │
  │   mic lease in · speaker lease out · control tickets                 │
  │   Rust lives here on retained native reservations — NOT std channels.│
  │   ticket/epoch retained to device callback. zero-copy. callback edge.│
  ├─────────────────────────────────────────────────────────────────────┤
  │ PLANE 3 — reliable events + lossy observer (two sub-planes)         │
  │   text/transcript events: RELIABLE, ticketed, exactly-once.          │
  │   telemetry + waveform-viz ONLY: lossy, coalescible, sampled.        │
  │   neither drives numerical progress; the reliable half must not drop.│
  └─────────────────────────────────────────────────────────────────────┘
```

Text and transcript are **not** telemetry: a dropped token is a corrupted
conversation, so text/transcript events ride a reliable ticketed channel
(exactly-once delivery, like the completion path). Only sampled telemetry and
waveform visualization may be lossy/coalescible.

### 2.2 The shift, in one picture

**Design baseline — Rust drove, blocking, one slot:**

```
Rust generate_with_cache loop  (holds a thread the entire turn):
  loop over tokens:
    pass_lock.lock()
    submit_pass(TOKEN_PASS) ─▶ [lane team] ─▶ CQ ─▶ unblock   ← thread blocked
    submit_pass(DEPTH_FRAME)─▶ [lane team] ─▶ CQ ─▶ unblock
    mimi decode_step ───────▶ [lane team] ─▶ CQ ─▶ unblock
    (Rust owns: cursor, KV cache, sampling loop, Candle prefill + scatter)
  SQ capacity = 1 · one pass in flight · no overlap · no native recurrence
```

**Target — eager native route execution, exact-CQ-driven:**

```
Rust: submit TURN ticket  (borrowed mic PCM lease) ──┐
                                                      ▼
                          ┌──────────────────────────────────────────┐
                          │        NATIVE SESSION STATE MACHINE        │
                          │  cursor · KV/conv planes · CSPRNG · epoch  │
                          └──────────────────────────────────────────┘
   TURN accepted ─▶ RouteInstance(label=AUDIO_ENCODE)
                              │
                              ├─▶ PASS(audio encode + prefill) ─▶ CQ ─┐
                              │                                      │ exact-CQ:
                              ├─▶ PASS(token + sample) ─▶ CQ ─────────┤ advance pc;
                              │         │                            │ submit one
                              │         ├─ text ─▶ fixed result cell ├ ready pass
                              │         │                            │ or release
                              │         └─ audio                     │ exact slot
                              │              │ retained playback     │
                              └◀─ recur ◀─ CQ(Mimi) ◀─ CQ(Depth) ◀───┘

   session coordinator: result cell ─▶ reliable text / PCM publication
                        external pressure ─▶ park instance, then re-admit
Rust only: fills the mic lease, drains the speaker lease, reads the transcript.
SQ capacity ≥ 2 · no synchronous numerical wait · no bridge-thread I/O wait.
```

### 2.3 Components

| Component | What it is | Exists? |
|---|---|---|
| **Weight image** | One allocation containing byte-exact main+codec source files; tensor views are `base + offset` and may be unaligned. | **Landed; page-table read-only after validation.** |
| **Scratch arenas** | Per-plan/per-conversation storage sized before readiness; zero steady-state growth. | **Partially landed.** Mimi and Conformer use fixed arenas; engine/backbone and conversation activation regions still need consolidation from separately owned vectors into one liveness map. |
| **Session state machine** | One per conversation. Owns cursor, KV/conv planes, sampler CSPRNG, codec state, epoch, and recurrence. | **Landed natively; Rust no longer drives model progress.** |
| **Pass program set** | Native resample, mel, Conformer, prefill, token, Depthformer, and Mimi stages. | **Typed boundary landed for LFM2.** Resample/frontend/whole-Conformer/adapter is one model-correlated ticket over borrowed spans and conversation-owned workspace. Conformer GEMMs and every other numerical substage advance through full-team return callbacks; no nested submission or numerical stage wait remains. |
| **Immutable forwarding tables** | Model-owned, eagerly executed, total outcome maps over coarse precompiled programs; no callback objects, DAG VM, lazy evaluation, or per-turn construction. | **Bounded production route landed.** The audio branch uses a closed three-node/four-outcome `TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE` table with bounds-checked terminal edges. Full-turn token-class and publication routing remain open. |
| **Per-ticket route instances** | Fixed records carrying separate flow, ticket, route-label, media-position, model-position, retained leases, and terminal state. | **Landed for the native LFM2 routes.** Fixed pooled records retain every cross-callback phase and lease; no stack callback or sleeping thread is continuation state. |
| **SQ/CQ (capacity ≥ 2) + exact-CQ callback substrate** | A completion can retain and tail-resubmit its exact slot without a synchronous coordinator wait. | **Production use landed.** Token commits on its CQ, Depthformer advances to Mimi when the reserved playback span is live, and the slot releases before reliable PCM publication. The fair broker re-admits runnable nodes; outward result/playback backpressure is retained state resumed by an explicit host-capacity edge. |
| **Docks** | Generation-checked mic/speaker PCM leases and bounded control/events. | **Native dock landed.** Physical Rust device adapter remains a later tranche. |
| **Host collapse** | Rust submits tickets, services PCM, and observes events; it owns no model state. | **Landed in the desktop production path; oracle rims are non-release.** |

### 2.4 Tickets identify work; labels select the next program

Do not overload protocol, model, media, and sequence identity. The concrete
representation may pack fields, but it preserves these separate facts:

```c++
struct FlowKey       { SessionId session; ConversationId conversation; uint64_t epoch; };
struct TicketId      { uint64_t runtime_epoch; uint64_t sequence; uint32_t generation; TicketKind kind; };
struct RouteLabel    { PlanId plan; uint32_t plan_generation; NodeId node; };
struct EndpointRoute { EndpointId source; EndpointId completion; };
struct LeaseRef      { SlotId slot; uint32_t generation; };
struct MediaSpan     { uint64_t stream_epoch; uint64_t chunk_sequence; uint64_t first_sample; uint32_t frames; };
struct ModelPosition { uint64_t absolute_position; };

struct BufferView {
    LeaseRef lease;
    uint64_t offset;
    uint64_t byte_length;
    Format format;
    Extents extents;
    ByteStrides strides;
};
```

These are private ownership facts, not a new public ABI; Rust continues to hold
opaque runtime/session handles, dock leases, control tickets, and events only.

`FlowKey` selects mutable conversation state. `TicketId` identifies one accepted
action and its exact terminal acknowledgement. `RouteLabel` selects one trusted
model program. `EndpointRoute` selects closed source/completion mailboxes, never
a callback address. `MediaSpan` orders audio fragments; `ModelPosition` orders
model state. A lane id, plan id, epoch, token id, and chunk sequence are never
substituted for one another. A `BufferView` is a validated non-owning view over a
retained byte lease; it is the only production meaning of “tensor.”

Model-open selects a compiled model template and publishes one compact immutable
forwarding table:

```c++
struct RouteEntry {
    ProgramOpcode opcode;             // coarse precompiled pass
    TeamPolicy team;                  // BLOCK4 or GANG8
    ScratchClass scratch;
    AccessSet access;
    CommitPolicy commit;
    RouteEdge outcomes[OUTCOME_COUNT];
};

struct RoutePlan {
    PlanId id;
    uint32_t generation;
    std::span<const RouteEntry> entries;
    NodeId entry;
    uint32_t max_live_leases;
    uint32_t max_scratch_class;
};
```

Every outcome row is total: each value maps to a validated next label, terminal,
or fault. Invalid runtime labels, opcodes, modalities, model tokens, and outcomes
fault before indexing. Model tokens first pass through a model-owned,
vocabulary-validated `TokenClass` table; checkpoint data never supplies opcodes
or function pointers. Construction resolves view dtype/shape/stride, kernel
specialization, aliases, scratch high-water, access conflicts, and route targets.
Execution performs no name lookup, dtype branch, allocation, callback traversal,
or general DAG scheduling.

**Evaluation is eager.** Once a command retains its declared input, output,
scratch, and result leases, submit immediately dispatches its entry program.
Programs are coarse fused passes such as audio encode, token, Depthformer, and
Mimi—not deferred scalar expressions. A coroutine may suspend only on an exact
CQ, unavailable predeclared capacity, an epoch/control edge, or a scheduling
quantum. There is no lazy array, `eval`, graph recording, JIT, on-demand output
allocation, or first-use construction on the inference path.

Each accepted turn uses a record from a fixed session-owned pool:

```c++
struct RouteInstance {
    FlowKey flow;
    TicketId ticket;
    RouteLabel label;
    EndpointRoute endpoints;
    MediaSpan media;
    ModelPosition position;
    ConversationLease conversation;
    ScratchLease scratch;
    CaptureLease input;
    PlaybackLease output;
    CompletionCellLease result;
    uint32_t quantum_left;
    AtomicTerminal terminal;
};

struct CompletionAck {
    FlowKey flow;
    TicketId ticket;
    RouteLabel label;
    ExecutionStatus execution;
    CommitStatus state_commit;
    PublicationStatus publication;
    Cause cause;
    TerminalStatus terminal;
};
```

Every payload view is reachable through one retained lease. The instance is
bound to one conversation scratch generation and borrows an exact engine slot
only while a program is queued or running. It outlives individual passes. At
quantum exhaustion it releases the slot; re-admission restores the exact flow,
ticket, label, position, and leases without rebuilding or replaying work.

### 2.5 The non-blocking completion boundary

In V1, `bridge_main` is the sole native SQ and CQ consumer. In design 16's
two-block executor, each block must first receive its own SPSC CQ and exact
doorbell; the ready broker is the sole consumer of both. A shared multi-producer
CQ is not an allowed shortcut. In either geometry, completion handling has a
deliberately tiny contract:

1. validate the exact `TicketId`, pass-slot generation, route-instance
   generation, `FlowKey`, and `RouteLabel` generation;
2. record the program outcome and apply its predeclared commit policy;
3. map that outcome through the total forwarding row;
4. tail-submit at most one already-ready numerical program using the retained exact
   slot, **or** release that slot and publish the instance to a pre-reserved
   session completion/ready cell;
5. ring one expected-value doorbell.

The compatibility engine currently overloads submission `conversation_id` and
`epoch` with a backbone or Depthformer plan id. The route cut separates those
identities: the immutable entry binds the model/depth plan; the SQ/CQ record
carries the real flow and ticket; media and model positions remain distinct. A
stale plan generation or mismatched ticket is a terminal fault. A correctly
identified completion whose publication epoch has become stale still settles
and applies its declared state-commit policy, but it cannot publish or take a
successor edge. Without this split, cancellation and shared-model routing are
not enforceable.

It must never call a synchronous engine API, wait, allocate, tokenize or format
text, reserve a PCM slot, publish into a potentially full reliable event ring,
invoke Rust/Tauri, or run a user callback. In particular, calling the current
`lfm_conversation_next_native` or `lfm_conversation_decode_native` from this
callback would deadlock: both submit and synchronously wait on the same bridge
thread that is executing the callback.

The session coordinator owns all publication work that can encounter
backpressure. It drains fixed result cells, attempts reliable
text/terminal or PCM publication, services control/interrupts, and re-admits
route instances whose external resource became available. A full destination
parks that instance in a fixed set; it does not make the coordinator wait inside
the publication call while unrelated ready work exists. The coordinator never
waits synchronously for a numerical CQ. Only when no command, result,
resource-ready edge, or route instance is runnable does it park on the composite
shared expected-value predicate. The bridge remains free to consume other SQ/CQ
cells throughout.

**Playback is admission, not a completion-side allocation.** Before a Mimi program
may be submitted, its route instance must already retain a playback reservation
large enough for the complete fixed-capacity PCM result. The Mimi pass writes
directly into that reservation. If the interleave schedule says an audio branch
is next, the session reserves the output before admitting the branch; no space
means the route instance parks without holding an engine slot. If Depthformer
produces EOAudio, the unused reservation is released exactly once. A stale epoch
also releases rather than publishes it.

Reliable text has the same pressure law even though its payload is small: a
fixed result cell is reserved before the producing program is admitted. If the
reliable channel cannot accept that cell, the route instance parks at the
publication boundary and releases its compute slot. Telemetry may be dropped; text,
terminal records, and PCM ownership facts may not.

---

## 3. Deep dive

### 3.1 The zero-copy weight pool

At the design baseline the backbone and Depthformer ran off `PtrLen` views into
Candle-owned tensors while the native image sat beside them. The production path
now binds every LFM2 and Mimi weight directly from the one image; `PtrLen` and
Candle ownership survive only inside the offline oracle feature.

Target: make the resident image the sole weight owner for every plan.

- **One byte-exact image.** Complete source bytes land once in the final
  allocation. Alignment padding exists only between sources; tensors remain at
  their safetensors offsets. Alignment is never repaired by copying a weight.
- **Binding.** Plans carry compact byte-addressed `{base/offset, bytes, dtype,
  shape, layout=NK}` descriptors, not Candle tensors or unaligned C++ typed
  pointers. `lfm_model.cpp` already performs most of the name/shape binding.
- **Consumption.** Architecture leaves load BF16 words from `(N,K)` views and
  unlift them in registers. No `.t().contiguous()`, packed RHS, F32 shadow, or
  per-call whole-weight widening is admitted on Apple or non-Apple paths.
- **Derived storage.** Only formula-changing immutable values such as rope
  tables, window/FFT tables, BN denominators, or required weight-normalization
  folds may persist. Their bytes are reported separately from the model image.
- **Deletion.** `candle_builder` / `CandleBridge` and the ~2.94 GB copy go away;
  the loader stops copying; the working set halves — which matters at decode,
  where M=1 GEMV streams the whole model per token and cache thrash is the enemy.

### 3.2 Scratch arena discipline

One arena per pass-program, sized at ctx/plan build to a high-water bound,
bump-allocated, **zero allocation in steady state**, abort on overflow. Mimi's
fixed arena and the Conformer workspace follow this ownership form. The engine
and conversation still describe several readiness-sized numerical regions with
independent `std::vector` owners; that is transitional even when those vectors
never grow during a pass. They must become offset views over one liveness-planned
arena. Two extensions:

- Keep the Conformer's conversation-owned byte arena sealed after readiness;
  mount its offset views directly on the audio route rather than creating a
  second engine-owned payload or rematerializing named stage planes.
- Add frontend (resample, mel) and prefill scratch to the same discipline.

Activations never become `Tensor`s. The mel plane, Conformer rows, adapter
output, hidden state, logits, depth codes, and Mimi PCM all live in engine
scratch or caller-owned buffers and pass between stages by pointer. The three
transport round-trips that exist today — mel→`u16` blob→`Tensor`, adapter
out→`Tensor`, Mimi codes→`Tensor`→`Vec` — are deleted; the session holds the
pointers across the pass boundary instead.

This is a **register-first materialization law**, not permission to allocate a
plane for every named expression in the reference graph. Within a fused leaf,
load checkpoint bytes through their view, unlift/accumulate/normalize in
registers, and write only the value that has a real downstream lifetime. Large
cross-leaf values use one readiness-sized arena and an immutable liveness map;
they do not use per-object vector ownership. Metadata arrays may describe views
and lifetimes, but never become numerical payload owners. “Final” is scoped to
one uninterrupted lifetime: a value may be written before a barrier, fan-out,
recurrence update, coroutine suspension, or device publication because registers
cannot carry it across that boundary. A function boundary by itself is not such
a lifetime boundary and never justifies materialization.

### 3.3 The native recurrence loop (the heart)

The Rust `generate_with_cache` state machine has been replaced by a native one.
As built, the C++ session coordinator performs the following steps after parking
for each exact completion through synchronous compatibility calls. The target
does not transplant that loop into a bridge-thread callback. It encodes the
ordered math as immutable forwarding rows and lets the ticket's `RouteInstance`
advance between programs:

1. reads the sampled token (sampling already native, folded into the pass),
2. checks stop / EOS,
3. advances the token cursor and the KV/short-conv cursor,
4. submits the next pass — decode `t+1`, or the Depthformer frame, or the Mimi
   frame — per the interleave schedule.

```
          ┌──────── token-pass CQ ──────────────┐
          ▼                                      │
   RouteInstance(label=TOKEN_DONE) ── EOS? ─▶ terminal-result cell
          │  no
          ├─▶ ready(DEPTH_FRAME) ─▶ CQ ─▶ pc=DEPTH_DONE
          │                                │
          │                                ├─▶ ready(MIMI_FRAME,
          │                                │          retained playback lease)
          │                                │              │
          │                                │              ▼ CQ
          │                                │       completed-PCM result cell
          └─▶ ready(TOKEN_PASS t+1) ◀──────┘

session coordinator: drain fixed result cell → reliable publish / re-admit
```

The exact-CQ callback may take a ready edge only when every input, output,
scratch, and result-cell lease for the destination node is already retained. If
an edge requires external capacity, it releases the compute slot, places the
instance in the session's fixed parked set, and rings the session doorbell. The
session coordinator later re-admits the same instance; it does not rebuild the
route or replay a completed program.

### 3.4 Cancellation and resource retirement

- **Epoch first.** Interrupt / barge-in advances the publication epoch before
  waking either coordinator. Every route instance and every SQ record carries
  its captured epoch. An already accepted assembly pass reaches its boundary.
  Its state-authoritative commit is not retroactively cancelled, but an old-epoch
  result cannot publish text or PCM and cannot enqueue another numerical node.
- **Commit is a program property.** Each forwarding entry declares whether successful
  conversational state commits before publication, commits only with reliable
  publication, or is speculative and rolls back. An emission delivered before
  interruption remains history: its pending token/code tuple is committed
  without another sample so KV/ShortConv agrees with the reliable transcript and
  audio stream. No generic cancellation path guesses this policy.
- **One terminal winner.** Completion, stale epoch, interrupt, fault, and stop
  race through one generation-checked terminal claim on the route instance.
  Losers do not publish or release resources a second time.
- **Retire after the last possible reader.** A capture lease releases only after
  its final frontend/prefill node completes. An unpublished playback reservation
  releases on EOAudio, stale epoch, cancellation, or fault. A result-cell lease
  releases after the coordinator consumes it. Conversation and scratch leases
  release only after the final accepted CQ is consumed and no parked/publication
  record can refer to them. Slot generation is recycled last.
- **Stop closes admission, then drains.** Stop prevents new route and pass
  admission, advances the epoch, and wakes every parked predicate. Accepted
  nodes settle; callbacks return their instances to the retirement queue instead
  of taking recurrence edges. Join waits for zero live route instances, engine
  slots, descriptors, result cells, capture leases, and playback leases before
  model/session destruction.

No cancellation or stop path waits on the bridge thread, frees a pointer still
reachable from an SQ/CQ record, or preempts an assembly operation.

### 3.5 Prefill + modality scatter as a native pass

The Candle scatter/cache owner is gone from production. C++ admits the whole
turn without mutating the window, then feeds text embedding rows or
Conformer-adapted BF16 buffer views into
`REQ_PREFILL` in causal groups of at most four. The kernel reads checkpoint BF16
directly, reuses each loaded weight vector across the group, and commits native
KV/ShortConv state in order. Longer inputs are chunked without exposing an
embedding plane to Rust.

The data and typed-executor boundary is now closed: one model-correlated audio
encode request carries resample, frontend, and whole-Conformer/adapter over
borrowed pointer spans and sealed conversation workspace. The remaining boundary
is recurrence ownership: the native coordinator still submits and waits for this
request synchronously instead of advancing it through a route instance.

The route ABI reserves an internal `SequenceMixerDesc`, not a public tensor or
operator language. LFM2 binds its existing ShortConv (`K=3`, causal halo two)
and attention programs; the Conformer convolution is a separate `K=9` operator
with halo eight. A future `MonarchLongConv` descriptor is admitted only with a
Hyena-family model whose checkpoint and oracle define that math. It is not a
drop-in LFM2 attention replacement, and speculative LFM2 context extension is a
separate research track.

### 3.6 The docks (I/O plane) — kcoro-rs, no std channels

The dock is where Rust std channels are dumped. Every hand-off below is a
kcoro-rs `ring` (bounded SPSC, `SendFuture`/`RecvFuture` that park on wake) or a
kcoro-rs `promise` (exact-once completion), so the audio path suspends and
resumes on the same expected-value substrate the lanes use — never on `mpsc`,
`crossbeam`, or a polling loop. kcoro-rs owns policy and lifecycle only; it never
touches PCM/weights/math and never runs on a compute lane or an audio callback
(its own contract).

- **Mic in — native chunked capture, NOT turn-batched.** The device callback
  writes PCM into a bounded ring; small fixed **chunks** are leased
  (`kc_descriptor` BORROWED) to the native session *as they arrive*, and the
  native session runs VAD, resample, and mel on the streaming chunks and detects
  turn boundaries itself. Rust does **not** accumulate a whole utterance and hand
  it over at turn-close — that turn-batching is the defect this replaces; Rust
  only moves chunks and holds the lease until the native side signals consumed.
  The callback stays a thin writer and does not run kcoro-rs.
- **Speaker out — native chunked playback.** Each Mimi frame pass publishes its
  PCM chunk into a descriptor-leased ring as it is produced; the Rust output dock
  drains chunks via a kcoro-rs `recv` future → `StreamingPcmResampler` (host
  rate-match, a permanent Rust surface) → device. Playback is continuous per
  chunk, not a whole-reply buffer. No `Tensor`, no `.wav`, no std channel crosses
  this seam.

The ~57 std-channel sites in the current voice runtime are retired here; the
`ThreadManager`/`done_rx` polling that motivated the coroutine work in the first
place is replaced by ring/promise waits.

### 3.7 Host collapse

`NativeEngine.pass_lock` now protects only the blocking compatibility rim; the
private exact-CQ proof can use the capacity-2 per-ticket slots directly. Once the
session drives recurrence through immutable route instances, the lock and every
blocking `submit_pass` rim are deleted. Rust's engine
surface collapses to: create session, submit TURN ticket with a mic lease,
receive PCM leases, submit control tickets. That is spec 10's end state.

---

## 4. Scale & reliability

### 4.1 Scheduling and fairness

One weight image serves many conversations. V1 still runs **one numerical pass
at a time**; SQ depth buys a queued successor and removes host round trips, not
parallel execution. Design 16 then extracts two independent logical four-lane
blocks that can gang into the existing eight-lane team. A block is a software
execution domain, not a promise of macOS cluster placement, private L2, or AMX
ownership.

Each block owns its stage board, scratch mount, active
command, SQ, and SPSC CQ. A ganged pass reserves both blocks and uses a dedicated
eight-lane board. Every fixed member runs the same
`{ticket,program,stage,generation}` and returns after that non-suspending stage.
The final member return publishes the completion callback; no operation member
parks at an internal boundary. Initially, simultaneous four-lane programs
must belong to different conversations. One state-mutating numerical program per
conversation is an invariant, not an inferred aliasing optimization.

The working-tree P0 already proves that the current single board can run with
four physical workers while preserving eight fixed logical partitions and their
fold order. Block extraction must preserve and rerun that parity gate; it does
not need to rediscover the logical-width rule.

The as-built exact-CQ substrate is FIFO over two descriptor-addressed pass slots.
A private test continuation tail-submits its same slot, which lets an already
queued peer run first. That is necessary, but it is not a broker and does not
prove fairness for more than two live route instances.

The route executor adds a fixed-capacity model/runtime ready broker:

- ready instances are grouped by bounded service class (`REALTIME`,
  `INTERACTIVE`, `BACKGROUND`) and ordered FIFO within a conversation quantum;
- a callback may retain its exact pass slot only for a bounded number of
  immediately ready programs;
- at quantum exhaustion, external backpressure, cancellation, or a program that
  lacks a retained resource, it releases the pass slot and returns the route
  instance to the broker's durable dormant set;
- age promotion prevents a continuously recurring conversation from starving a
  third conversation when engine capacity is two;
- an interrupt/control edge may outrank future numerical admission, but it does
  not preempt a running assembly node;
- dispatch always revalidates instance generation, conversation id, epoch, and
  scratch generation before mounting views on the lane board.

Fairness can act only at fused-program boundaries. The longest admitted program,
not the nominal quantum, is therefore the preemption and third-conversation wait
floor; acceptance reports that measured duration.

No fairness mechanism polls. Enqueue and quantum-return publish one retained
service edge. Only the resident runtime worker may become dormant when the
entire ready predicate is empty.

### 4.2 Throughput and failure

- **Bandwidth is the decode ceiling.** M2 Max is 400 GB/s; the engine currently
  realizes ~66 of a ~250 GB/s practical bound. Decode is M=1 GEMV — every token
  streams the whole model, so
  `tokens/s ≈ achieved_bandwidth / model_bytes_per_token`. The pool does not
  change that arithmetic, but deleting the duplicate keeps the working set from
  thrashing and keeps weights bf16 (half the bytes) in `(N,K)` (no repack). The
  win from zero-copy is measured in GB/s of avoided **activation** traffic.
- **Time is observation only.** Tickets and doorbells have no deadline-driven
  transition or timeout cause. Latency timestamps and device-liveness watchdogs
  may report a separate telemetry/control fault, but elapsed time never admits,
  completes, publishes, or advances numerical work.
- **Failure is terminal, not degraded.** No fallback. A stale command rejected
  before admission or an unmet gate is a terminal completion with a cause. An
  already accepted stale-epoch pass settles under its commit policy but cannot
  publish or recur. Neither case silently drops to Candle.

### 4.3 Deterministic route-executor acceptance tests

The exact-slot theft, stale-owner ABA, active-pass stop/drain, live-slot
accounting, total three-node/four-outcome map, routed-versus-split
hidden/ShortConv/code/PRNG parity, token-commit-before-Depth-failure, and an
injected third-node Mimi failure through the real SQ/lane/CQ path validate the
landed bounded route. The asynchronous route cutover is not complete
until deterministic tests also prove:

1. **Plan validation:** bad route labels, non-total outcome rows, invalid runtime
   token classes/modalities, missing kernel specializations, wrong view
   dtype/shape/stride, illegal aliases, and scratch or lease high-water overflow
   fail before publication. The immutable table has a stable digest.
2. **Ordered route math:** token → Depthformer → Mimi runs from one route
   instance with full hidden/logit/code/PCM parity, while every program receives
   the exact prebound view and conversation scratch generation.
3. **More-than-capacity fairness:** run at least three multi-program route
   instances through a capacity-2 engine with a quantum of one. A captured dispatch trace
   proves bounded round-robin/age promotion; no instance retains a pass slot
   across its quantum and all three make progress.
4. **Peer arrival during recurrence:** a peer submitted after program 1 but before
   program 2 is dispatched before the recurring instance can exceed its quantum.
   Repeating this for more than two route edges produces the same trace.
5. **Backpressure split:** fill the reliable result ring and the playback ring,
   separately. The affected instance parks with zero engine slots retained; the
   sole bridge thread continues settling an unrelated ticket. Draining the exact
   ring wakes and resumes only the parked instance—no retry loop.
6. **Playback admission:** Mimi is never dispatched without a retained,
   capacity-checked playback reservation. EOAudio, stale epoch, fault, and stop
   each release an unused reservation exactly once; successful equal-rate Mimi
   writes directly into that reservation, while differing-rate output is written
   there by the same retained route's prepared native resampler.
7. **Interrupt at every boundary:** advance epoch before submit, while queued,
   while running, after CQ before resubmit, while parked for reliable output,
   and while parked for playback. Accepted math may settle and a
   state-authoritative program may commit, but no stale route
   publishes or takes another numerical edge.
8. **Stop and retirement:** stop from every state produces one terminal winner
   and joins with zero route instances, pass slots, descriptors, result cells,
   capture leases, playback leases, and wait registrations. Owner-drop ordering
   runs under ASan/UBSan/TSan.
9. **No bridge blocking:** instrument the exact-CQ callback and fail if it calls
   a synchronous engine/model API, waits, allocates, formats text, reserves a
   dock slot, or invokes a host callback. A stalled session coordinator cannot
   prevent the bridge from consuming another accepted SQ/CQ pair.
10. **Steady state:** one million routed-program completions allocate nothing
    after readiness, use no timed waits/sleeps/spin, leave every lease counter at
    zero, and preserve the existing idle CPU gate on aarch64 and x86_64/Rosetta.
11. **Block completion:** two different conversations complete in reverse order
    through separate block CQs; a ganged program excludes both blocks, and no
    same-conversation state mutation overlaps.
12. **Collective participation:** every fixed member arrives exactly once at
    every declared stage; an early return faults the test, and each last-arrival
    mixer executes exactly once.

Load tests continue to report p50/p95/p99/max latency. Two conversations sharing
one image remain the numerical-isolation gate; the fairness gate uses at least
three instances so it cannot accidentally pass because the SQ and slot capacity
are both two.

---

## 5. Trade-offs (explicit)

1. **Resident image vs the Candle duplicate.** Binding the resident image
   directly (no pool, no repack — spec 02) halves RAM, but `candle_builder` cannot
   die until *every* consumer is native, and Candle owns prefill today. The
   Depthformer share has already dropped; the remaining backbone/embedding copy
   drops atomically when production adopts the completed native model. Adding a
   second native model beside the Rust model is forbidden because it would create
   a third main-checkpoint image. Candle remains an offline parity oracle only.
2. **Native recurrence vs the Rust loop.** The whole point: removes the blocked
   host thread and enables overlap. Cost: the hardest code in the project — a
   native state machine replacing a readable Rust loop, harder to debug.
   Mitigation: the per-phase-stop and soak gates; fixture-first parity per pass.
3. **SQ capacity ≥ 2 vs 1.** On V1 this enables queueing and host-dispatch
   overlap, not parallel numerical execution. V2 may run two block programs only
   after each block owns a board, scratch mount, and CQ. Start at capacity two;
   a ganged eight-lane program consumes both block permits.
4. **No-fallback law vs incremental migration.** The law forbids a silent Candle
   `.or_else`, yet the migration needs Candle alive until the native path is
   complete. Resolution: Candle is a *build-time / offline* oracle, never wired as
   a runtime fallback. The runtime gate is "native or terminal error," consistent
   with the Mimi-required rule.
5. **Apple direct BF16 kernels vs Accelerate staging.** Resolved in favor of the
   image contract: the M≤4 path reads checkpoint-layout BF16 directly and reuses
   each loaded weight vector across the active rows. It creates no widened RHS,
   packing buffer, or layout copy. Activation scratch may change precision when
   the numerical contract requires it, but resident weights are unlifted only in
   registers.
6. **Prefill native vs leaving it Candle.** Prefill is per-turn, but it is the
   ownership gate for deleting the remaining compatibility image. Develop it
   offline against Candle fixtures; do not ship a hybrid native/Candle fallback.
7. **Moshi.** Moshi stays a **supported model** — it is not dropped. It is
   partially on Flashkern already and gets ported the rest of the way, but as its
   own later phase (P5), because it is a second whole model and would otherwise
   stall the LFM2 hot-loop work. Decision: flip the shipped default to LFM2 only
   in the atomic native cutover. Moshi remains buildable and exercised offline
   until its native port lands.
8. **Logical blocks vs hardware clusters.** `2×4` mirrors the measured M2 Max
   topology, but macOS supplies no hard cluster pinning and Accelerate exposes no
   AMX reservation. Correctness depends only on software-owned boards and fences;
   locality and matrix overlap are measured benefits, never invariants.
9. **Tile-stationary reuse vs layer pinning.** L2 is hardware-managed cache, not
   addressable threadgroup memory. An LFM2 FFN alone is about 96 MiB at
   `hidden=2048`, `ffn=8192`, BF16, so a complete layer cannot fit in 16 MiB.
   A bounded pass instead reuses one weight tile/stripe over an opportunistic
   snapshot of ready conversation rows and never delays an interactive row to
   form a batch.

---

## 6. Build order

### 6.0 What already exists — and why it reorders the plan

This section records the pre-cutover sequencing decision. The ownership work it
describes is now landed for LFM2:

- **A native LFM2 model exists** — `native/src/model/lfm_model.cpp` binds
  the whole backbone by name off the resident image (every layer's norms, FFN,
  short-conv, attention + qk-norms), plus embeddings, head, and Depthformer, all
  zero-copy. The product surface is now opaque runtime/model/conversation/session
  lifecycle plus PCM/control/event docks; numerical direct calls are oracle-only.
- **No weight pool needs to be built.** Spec 02 is explicit: kernels bind the
  resident image *unaligned* and must not repack. The resident image is the pool.
  The earlier "build a re-aligned pool" framing was wrong; drop it.
- **Production voice now uses this path exclusively.** It constructs the native
  runtime/model/conversation/session and fails hard for unsupported engines or
  devices. Frontend, Conformer, Mimi, modality assembly, tokenizer, sampling,
  recurrence, and context rollover are native-owned.

That sequencing constraint has now been discharged: native prefill and atomic
desktop adoption landed together, so `compatibility_copied_bytes == 0` is the
production gate rather than an aspirational counter. The resident image remains
the only weight pool.

### 6.1 Historical phase record and live follow-ons

P0–P3 below are retained as migration rationale, not current-state claims. Their
numerical ownership outcomes are landed; their asynchronous scheduling outcome
is not. The live LFM2 work is now:

1. **V2.0 safety subset — landed.** Keep the green 128-byte isolation,
   closed request/layer/modality selectors, invalid-lane rejection,
   four-worker/eight-logical-lane parity, and zero-spin gates;
2. **V2.1 bounded audio route — landed; asynchronous executor open.** Keep the
   total `TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE` production route,
   reserve-before-admit playback contract, direct PCM write, and terminal
   cleanup gates. Replace its one outward expected-value wait and fixed
   conversation-result/stack-callback bridge with model-owned token classes,
   pooled route/result/playback instances, and the fair broker;
3. extract two logical four-lane block domains plus a gang lease, but run only
   one block/ganged program until each block owns an independent SPSC CQ and the
   four-lane logical-fold parity gate passes;
4. add fixed-membership block-mode kcoro continuations and evaluate hardware
   event waits behind the OS-wait correctness fallback;
5. enable two-block execution first for different conversations, then add
   opportunistic tile-stationary multi-conversation decode without admission
   delay;
6. replace the transitional Rust PCM/`VoiceEvent` adapter with physical kcoro
   audio-device docks and move oracle/training sources into
   `liquid-audio-oracle`.

Each phase ends at a gate and deletes the Rust/Candle owner it replaces.

- **P0 — done.** Mel, resample, Conformer + adapter native behind rims.
- **P1 — Adopt the native model where it is already complete; first copy drop.**
  Unwire Moshi from the default (below). Rebind the **fully-native-consumed**
  weights — the Depthformer, whose only consumer is the native depth-frame pass —
  from the resident image instead of `PtrLen`-into-Candle-storage, and stop
  copying them. Route the text / audio-out discrete-token path through
  `lfm_conversation_*`. *Gate:* Depthformer `compatibility_copied_bytes`
  contribution → 0; parity holds; native discrete-token recurrence drives a turn.

  **Depthformer cut — LANDED.** `build_depth_decode_resident`
  (`model/lfm2_audio.rs`) binds the depth plan straight from the resident image by
  name, with rope from the native `lfm_rope_table_f32` kernel — the same one
  `lfm_model.cpp` uses. It is now the production depth path; the Candle depth
  modules (`depthformer` / `depth_linear` / `depth_embeddings`) are built only on
  the non-resident training path (now `Option`, guarded in the training `forward`).
  Verified: `depth_resident_binder_matches_candle_binder` proves byte-identical
  greedy tokens vs the Candle-bound plan; the production load's Candle-copy ledger
  fell **231 → 151 tensors, 2.711 → 2.475 GB** (~236 MB / 80 depth tensors no
  longer duplicated). The remaining ~2.475 GB is the backbone + embeddings, whose
  copy is coupled to native prefill (P2/P3) — Candle owns prefill until then.
- **P2 — Native audio-in prefill + modality scatter (close gap a).** Extend the
  conversation ABI with a continuous-embedding prefill input so the already-native
  mel → Conformer → adapter rows scatter into the backbone natively, by modality
  flag, with no Candle. This is the unlock: once prefill is native, nothing on the
  input side needs the Candle model. *Gate:* audio-in prefill parity vs the Candle
  reference; no `Tensor` at the mel/adapter seam.

  **This is C++-owned, not a Rust rim.** The native prefill lives in
  `lfm_conversation_prefill` (`lfm_model.cpp`): C++ owns the prefill recurrence,
  and Rust only hands the Conformer output over as a *view* and submits a ticket.
  `native_engine.rs` stays a transitional **parity rim** — inference is never
  wired through it. (Guardrail: growing the Rust rim to drive prefill would keep
  Rust as the inference driver, which the whole migration exists to end.)

  **Native audio-in prefill — LANDED (capability; adoption pending).**
  - `lfm_engine_token_pass` gained an `embed_kind == 2` "provided embedding" path
    (`flashkern_engine.cpp`): a bf16 `[H]` hidden view fed verbatim into the pass
    scratch, skipping the table lookup — the point-and-stride way to feed a
    Conformer row (no weight copy). ABI carries a trailing `provided_embed`
    pointer, `nullptr` on every discrete-token caller.
  - `lfm_conversation_prefill_audio` (`lfm_model.cpp`, exposed as
    `NativeConversation::prefill_audio` in `handles.rs`) prefills `[n, hidden]`
    Conformer rows (a borrowed view) into KV, one provided-embedding pass per row —
    same sequential-per-position shape the discrete `lfm_conversation_prefill`
    already uses, so the "sequential vs parallel" worry was moot for the first cut.
  - **Verified:** `native_audio_prefill_matches_discrete_for_the_same_embedding`
    proves `embed_kind == 2` fed a token's own `embed_tokens` row yields the
    identical greedy next-token as the discrete `embed_kind == 0` path — i.e. the
    provided-embedding path produces byte-identical backbone state. 167 lib + 7
    native_safetensors green; decode unaffected.

  Still **C++-owned via the native `LfmModel` conversation** (per the steer:
  `native_engine.rs` stays a parity rim). What remains to drop the backbone copy:
  production voice must *adopt* this native conversation (route the Conformer
  output view + the interleave schedule through it, retire the Candle
  `forward_embeds`). A native *parallel* multi-token prefill pass is the perf
  follow-up for long context.
- **P3 — Adopt the native model for the whole turn; delete the Candle path (close
  gap b).** Move the interleaved generate schedule into the native session so
  `generate_with_cache` / `generate_interleaved` and the Candle
  `LFM2AudioModel` / `Lfm2Model` construction and `candle_builder` all delete
  together; the SQ gains depth (capacity 2), and an immutable eager route table
  chains decode → depth → Mimi through exact-CQ advancement; `pass_lock` and the
  blocking rims go. *Gate:*
  `compatibility_copied_bytes == 0` for LFM2; RAM halves; 1M-pass soak;
  at least three route instances fair through the capacity-2 executor;
  zero-alloc-after-ready; idle `< 0.1%`; no session coordinator blocks on a
  numerical CQ. (This is the old P1+P2+P4 collapsed, because the native vehicle
  already exists.)
- **P4 — Physical zero-copy docks (open).** Native mic/speaker descriptor leases,
  expected-value rings, and the direct Mimi-to-playback-reservation seam are
  landed. Replace the remaining Rust device/`VoiceEvent` copies and std-channel
  adapter with physical kcoro device callbacks. *Gate:* no owned PCM copy or std
  channel on the audio path (static audit).
- **P5 — Finish the Moshi port to Flashkern.** Moshi is a supported model; carry
  its partially-native pipeline the rest of the way onto the lane team and the
  resident image, then delete Candle from the shipped pipeline entirely. Until then
  Moshi is unwired from the default but remains buildable / exercised offline.

**Moshi default-switch outcome:** the shipped default is native LFM2. Moshi is
offline/oracle-only until its native port replaces the Candle implementation;
production never falls back between them.

**Revisit as it grows:** SQ capacity (2 → N as multi-conversation load rises);
arena high-water sizing under long contexts; an E-core `BACKGROUND` lane for
speculative decode / telemetry (currently P-core only); direct-BF16 kernel tile
geometry and prefetch distance when profiling identifies the bottleneck.

---

## 7. Assumptions

- The checkpoint stays bf16 `(N,K)`; no retrain, no requant.
- One model image serves all conversations; there is no per-conversation weight
  specialization.
- Apple and non-Apple production weight paths consume checkpoint BF16 directly;
  no backend may require a complete RHS conversion or repack.
- Real-time targets follow the Sesame latency bands already encoded in the voice
  runtime.
- Candle can be reduced to an offline oracle — nothing in the shipped product
  requires a Candle tensor at runtime once P4/P5 land.
