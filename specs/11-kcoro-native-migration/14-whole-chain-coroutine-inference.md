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
That substrate is **not yet the production graph executor**: the only chained
proof is a private test request, and production recurrence still enters blocking
compatibility calls. Two follow-ons remain explicit: the session coordinator
must adopt the immutable graph/per-ticket execution boundary specified below,
and the physical mic/speaker adapter still bridges the native dock into the
legacy Rust `VoiceEvent` surface. The full Moshi port and physical
kcoro device dock are subsequent tranches; neither permits a Candle fallback.

This is the convergence target for specs 02, 03, 07, and 10 —
the picture they are each a slice of. It describes the end state where the entire
inference chain, microphone PCM to speaker PCM, runs as native passes on the
fixed Flashkern lane team, clocked by completion doorbells rather than a Rust
loop, over one zero-copy weight pool.

**Terminology is normative:** where this document says "tensor view," it means
only pointer + byte count + dtype + shape/derived-stride metadata over an existing
buffer. It never means an owning tensor object, framework allocation, or a
materialized payload. Production data moves as retained buffer spans.

The load-bearing observation: **the substrate for this already exists.** Flashkern
is already a GPU-threadgroup engine — a fixed P-core lane team, generation-fence
barriers, atomic tile-claim, one dispatcher, expected-value doorbells, no spin
tier. The SQ/CQ bridge with descriptor leases exists. The safetensors loader
already demonstrates the required one-ingress-write discipline: a byte-exact
resident image with immutable views. Mimi's private folded arena is not the
model-image precedent and must not be generalized into a second weight pool.
At the design baseline, what was missing was not primitives but three ownership
cuts (retained here as historical rationale):

1. Rust drove recurrence by **blocking** on a **single-slot** pass.
2. The graph was **Candle above the assembly leaves** (prefill, modality
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

And the Rust audio dock uses **kcoro-rs** (`crates/kcoro`) — the same
non-blocking, park-on-wake mechanism as the native layers — so that **Rust std
channels are dumped entirely** at that layer. No `mpsc`, no `crossbeam`, no
polling loop. The dock's rings and promises wake on the same expected-value
substrate the lanes use. (Today the voice runtime holds ~57 std-channel sites;
those are the debt this retires.)

---

## 1. Requirements

### 1.1 Functional

- Every stage of the chain runs as a lane-uniform native pass on the resident
  lane team: resample, mel, Conformer, adapter, prefill + modality scatter,
  backbone prefill, backbone decode, text sampling, Depthformer, Mimi decode.
- **Recurrence is native.** The token/frame loop is an eagerly submitted native
  graph: an exact pass completion advances its fixed per-ticket instance and
  enqueues an already-resolved next pass directly, without a host round-trip.
  This is the "device recurrence" row of the Flashkern GPU-equivalence table,
  realized.
- **Rust's only production roles** are the two docks and the observer:
  - dock microphone PCM in as a borrowed descriptor lease,
  - drain speaker PCM out of a borrowed descriptor lease,
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
  Activations live in engine-owned scratch planes and descriptor-leased rings and
  are passed between passes by pointer. No stage materializes a `Tensor`.
- **Zero-wait.** No polling, no bounded spin, no host thread blocked on the
  progress path. Every wait is an expected-value doorbell. The idle lane team
  stays under the existing `< 0.1%` CPU gate (`engine_idle_zero_spin`).
- **Real-time.** The pipeline overlaps stages, so wall-clock is the critical path,
  not the sum of stages. Per-frame Mimi decode (~14 ms) must keep pace with
  playback; backpressure is a doorbell park on the speaker dock, never a sleep.
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
- **Rust channels are dumped** at the audio dock in favour of kcoro-rs rings /
  promises, which park on the native wait-word substrate.
- **No fallback chains.** A native gate failure is a terminal completion with a
  cause, not a silent drop to Candle. (This is a real sequencing constraint —
  see Trade-off 4.)
- Rust host; C++ owns plans, sessions, and recurrence; assembly owns all math.
- Target hardware: M2 Max — 8 performance-core lanes (E-cores excluded by
  policy), 400 GB/s, bandwidth-bound at decode.

---

## 2. High-level design

The whole chain is **one native session state machine eagerly executing an
immutable, model-owned pass graph on the existing lane team.** Each accepted
command owns a fixed per-ticket execution instance. Exact completions advance
that instance; they do not invoke an allocated list of callbacks. This is a
precompiled native dataflow graph clocked by doorbells, not a lazy evaluator.

### 2.1 Three planes (do not merge them)

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ PLANE 1 — native model SQ/CQ (compute)                              │
  │   fixed lane team · pass descriptors · generation fences · doorbells │
  │   THE progress path. exact-once completions. no host on it.          │
  ├─────────────────────────────────────────────────────────────────────┤
  │ PLANE 2 — PCM / control dock (I/O)                                   │
  │   mic lease in · speaker lease out · control tickets                 │
  │   Rust lives here, on kcoro-rs rings/promises — NOT std channels.    │
  │   borrowed descriptor regions. zero-copy. park-on-wake.              │
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

**Target — eager native graph execution, exact-CQ-driven:**

```
Rust: submit TURN ticket  (borrowed mic PCM lease) ──┐
                                                      ▼
                          ┌──────────────────────────────────────────┐
                          │        NATIVE SESSION STATE MACHINE        │
                          │  cursor · KV/conv planes · CSPRNG · epoch  │
                          └──────────────────────────────────────────┘
   TURN accepted ─▶ GraphInstance(graph=TURN, pc=AUDIO_ENCODE)
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
| **Scratch arenas** | Per-plan/per-conversation storage sized before readiness; zero steady-state growth. | **Landed for the complete LFM2 chain.** |
| **Session state machine** | One per conversation. Owns cursor, KV/conv planes, sampler CSPRNG, codec state, epoch, and recurrence. | **Landed natively; Rust no longer drives model progress.** |
| **Pass program set** | Native resample, mel, Conformer, prefill, token, Depthformer, and Mimi stages. | **Typed boundary landed for LFM2.** Resample/frontend/whole-Conformer/adapter is one model-correlated SQ/CQ request over borrowed spans and conversation-owned workspace; its Conformer GEMMs are in-ticket team substages rather than recursive submissions. The native coordinator still submits and waits for that request synchronously, so graph-executor integration remains open. |
| **Immutable execution graphs** | Model-owned, eagerly executed, prevalidated opcode/DAG descriptions over typed views; no callback objects, lazy evaluation, or per-turn graph construction. | **Not landed.** Production still encodes recurrence in `voice_session.cpp` / `lfm_model.cpp` call flow. |
| **Per-ticket graph instances** | Fixed records carrying graph id, program counter/dependencies, epoch, conversation scratch lease, retained I/O leases, and terminal state. | **Not landed.** Current `PassSlot` owns one request and scratch bank, not a complete turn execution. |
| **SQ/CQ (capacity ≥ 2) + exact-CQ callback substrate** | A completion can retain and tail-resubmit its exact slot without a synchronous coordinator wait. | **Engine substrate landed.** Capacity 2, exact ticket routing, generation-checked exact-slot resubmission, and per-ticket scratch are live and adversarially tested. The private PRNG chain proves the mechanism; session recurrence still submits through the synchronous compatibility rim. |
| **Docks** | Generation-checked mic/speaker PCM leases and bounded control/events. | **Native dock landed.** Physical Rust device adapter remains a later tranche. |
| **Host collapse** | Rust submits tickets, services PCM, and observes events; it owns no model state. | **Landed in the desktop production path; oracle rims are non-release.** |

### 2.4 Ticket means a graph execution instance

A production inference ticket does not name one C++ function call. It names one
execution of a model-owned, immutable graph over prebound views. Graphs are built
and validated when `LfmModel` becomes ready, then published read-only with the
model plan:

```c++
struct GraphNode {
    GraphOpcode opcode;          // preselected kernel/pass family
    uint32_t dependency_mask;    // bounded graph, no pointer chasing
    uint32_t input_bindings;     // indices into validated view/lease tables
    uint32_t output_bindings;
    uint32_t scratch_class;
    CommitPolicy commit;
    GraphEdge success;
    GraphEdge alternate;         // bounded result branch: EOS/modality/etc.
};

struct GraphPlan {
    GraphId id;
    std::span<const GraphNode> nodes;
    uint32_t entry;
    uint32_t max_live_leases;
    uint32_t max_scratch_class;
};
```

The concrete representation may pack these fields, but the ownership law does
not change. Plan construction resolves tensor dtype/shape/stride, kernel
specialization, input/output alias legality, scratch high-water bounds, and all
edge targets. Execution performs no tensor-name lookup, dtype branch, graph
allocation, function-object allocation, or callback-list traversal. The graph
is not a public JIT format and cannot contain arbitrary function pointers. A
declared recurrence edge may return to an earlier node; all other dependency
edges are acyclic and are validated as such.

**Evaluation is eager.** Model-open resolves every opcode and edge into a
private, direct-threaded/prelinked continuation table. An edge is a validated
index into that closed table, not a late name lookup or heap callback. Once a
command has retained its declared inputs, outputs, scratch, and result cells,
submit dispatches the entry pass immediately. Compute opcodes are coarse fused
passes such as audio-encode, token, Depthformer, and Mimi—not deferred scalar or
tensor expressions. Coroutine tokens may suspend only on a real dependency:
an exact CQ, an unavailable predeclared I/O/result lease, an epoch/control edge,
or a conversation scheduling quantum. There is no deferred array, `eval`,
partial evaluation, dynamic graph recording, on-demand output allocation, or
first-use plan construction anywhere on the inference path.

Each accepted turn uses a record from a fixed session-owned pool:

```c++
struct GraphInstance {
    TicketId ticket;
    GraphId graph;
    uint32_t pc;
    uint32_t ready_mask;
    uint32_t complete_mask;
    uint64_t epoch;
    ConversationLease conversation;
    ScratchLease scratch;
    CaptureLease input;
    PlaybackLease output;
    CompletionCellLease result;
    uint32_t quantum_left;
    AtomicTerminal terminal;
};
```

These are ownership fields, not necessarily the public layout. Every pointer
read by a node is reachable through one retained lease in the instance. The
instance is bound to exactly one conversation scratch generation and can borrow
one exact engine `PassSlot` only while a node is queued or running. The graph
instance outlives individual engine passes; the pass slot does not have to.

This distinction is essential for capacity two: retaining an engine slot across
an unbounded turn would leave only one slot for every other conversation. A
bounded node quantum returns the instance to the native ready queue and releases
the exact slot. Re-admission later restores the same `pc`, dependency facts,
epoch, scratch generation, and leases without reconstructing the graph.

### 2.5 The non-blocking completion boundary

`bridge_main` is the sole native SQ consumer **and** CQ consumer. Therefore its
exact-CQ callback has a deliberately tiny contract:

1. validate the exact ticket, pass-slot generation, graph instance generation,
   conversation id, and epoch;
2. record the node terminal fact and apply its predeclared commit policy;
3. advance `pc` / dependency bits;
4. tail-submit at most one already-ready numerical node using the retained exact
   slot, **or** release that slot and publish the instance to a pre-reserved
   session completion/ready cell;
5. ring one expected-value doorbell.

It must never call a synchronous engine API, wait, allocate, tokenize or format
text, reserve a PCM slot, publish into a potentially full reliable event ring,
invoke Rust/Tauri, or run a user callback. In particular, calling the current
`lfm_conversation_next_native` or `lfm_conversation_decode_native` from this
callback would deadlock: both submit and synchronously wait on the same bridge
thread that is executing the callback.

The session coordinator owns all publication work that can encounter
backpressure. It drains fixed graph-result cells, attempts reliable
text/terminal or PCM publication, services control/interrupts, and re-admits
graph instances whose external resource became available. A full destination
parks that instance in a fixed set; it does not make the coordinator wait inside
the publication call while unrelated ready work exists. The coordinator never
waits synchronously for a numerical CQ. Only when no command, result,
resource-ready edge, or graph instance is runnable does it park on the composite
shared expected-value predicate. The bridge remains free to consume other SQ/CQ
cells throughout.

**Playback is admission, not a completion-side allocation.** Before a Mimi node
may be submitted, its graph instance must already retain a playback reservation
large enough for the complete fixed-capacity PCM result. The Mimi pass writes
directly into that reservation. If the interleave schedule says an audio branch
is next, the session reserves the output before admitting the branch; no space
means the graph instance parks without holding an engine slot. If Depthformer
produces EOAudio, the unused reservation is released exactly once. A stale epoch
also releases rather than publishes it.

Reliable text has the same pressure law even though its payload is small: a
fixed graph-result cell is reserved before the producing node is admitted. If
the reliable channel cannot accept that cell, the graph parks at the publication
boundary and releases its compute slot. Telemetry may be dropped; text,
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
bump-allocated, **zero allocation in steady state**, abort on overflow. This is
already true for the engine ctx scratch, `DepthPlan`, `BackbonePlan`, the Mimi
256 MiB arena, and the Conformer workspace. Two extensions:

- Fold the Conformer's per-call `create/destroy` workspace into the resident
  engine scratch so even audio-in prefill is allocation-free.
- Add frontend (resample, mel) and prefill scratch to the same discipline.

Activations never become `Tensor`s. The mel plane, Conformer rows, adapter
output, hidden state, logits, depth codes, and Mimi PCM all live in engine
scratch or caller-owned buffers and pass between stages by pointer. The three
transport round-trips that exist today — mel→`u16` blob→`Tensor`, adapter
out→`Tensor`, Mimi codes→`Tensor`→`Vec` — are deleted; the session holds the
pointers across the pass boundary instead.

### 3.3 The native recurrence loop (the heart)

The Rust `generate_with_cache` state machine has been replaced by a native one.
As built, the C++ session coordinator performs the following steps after parking
for each exact completion through synchronous compatibility calls. The target
does not transplant that loop into a bridge-thread callback. It encodes the
ordered math as an immutable graph and lets the ticket's `GraphInstance` advance
between nodes:

1. reads the sampled token (sampling already native, folded into the pass),
2. checks stop / EOS,
3. advances the token cursor and the KV/short-conv cursor,
4. submits the next pass — decode `t+1`, or the Depthformer frame, or the Mimi
   frame — per the interleave schedule.

```
          ┌──────── token-pass CQ ──────────────┐
          ▼                                      │
   GraphInstance(pc=TOKEN_DONE) ── EOS? ─▶ terminal-result cell
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
graph or replay a completed node.

### 3.4 Cancellation and resource retirement

- **Epoch first.** Interrupt / barge-in advances the publication epoch before
  waking either coordinator. Every graph instance and every SQ record carries
  its captured epoch. An already accepted assembly pass reaches its boundary,
  but an old-epoch result cannot publish text or PCM and cannot enqueue another
  numerical node.
- **Commit is a node property.** Each graph node declares whether successful
  conversational state commits before publication, commits only with reliable
  publication, or is speculative and rolls back. An emission delivered before
  interruption remains history: its pending token/code tuple is committed
  without another sample so KV/ShortConv agrees with the reliable transcript and
  audio stream. No generic cancellation path guesses this policy.
- **One terminal winner.** Completion, stale epoch, interrupt, fault, and stop
  race through one generation-checked terminal claim on the graph instance.
  Losers do not publish or release resources a second time.
- **Retire after the last possible reader.** A capture lease releases only after
  its final frontend/prefill node completes. An unpublished playback reservation
  releases on EOAudio, stale epoch, cancellation, or fault. A result-cell lease
  releases after the coordinator consumes it. Conversation and scratch leases
  release only after the final accepted CQ is consumed and no parked/publication
  record can refer to them. Slot generation is recycled last.
- **Stop closes admission, then drains.** Stop prevents new graph and pass
  admission, advances the epoch, and wakes every parked predicate. Accepted
  nodes settle; callbacks return their instances to the retirement queue instead
  of taking recurrence edges. Join waits for zero live graph instances, engine
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
request synchronously instead of advancing it through a graph instance.

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
session drives recurrence through immutable graph instances, the lock and every
blocking `submit_pass` rim are deleted. Rust's engine
surface collapses to: create session, submit TURN ticket with a mic lease,
receive PCM leases, submit control tickets. That is spec 10's end state.

---

## 4. Scale & reliability

### 4.1 Scheduling and fairness

One weight image and one lane team serve many conversations. The lane team still
runs **one numerical pass at a time** (it is a threadgroup); SQ depth buys a
queued successor and removes host round trips, not concurrent mutation of the
stage board.

The as-built exact-CQ substrate is FIFO over two descriptor-addressed pass slots.
A private test continuation tail-submits its same slot, which lets an already
queued peer run first. That is necessary, but it is not a broker and does not
prove fairness for more than two live graph instances.

The graph executor adds a fixed-capacity model/runtime ready broker:

- ready instances are grouped by bounded service class (`DEADLINE`,
  `INTERACTIVE`, `BACKGROUND`) and ordered FIFO within a conversation quantum;
- a callback may retain its exact pass slot only for a bounded number of
  immediately ready nodes;
- at quantum exhaustion, external backpressure, cancellation, or a node that
  lacks a retained resource, it releases the pass slot and returns the graph
  instance to the broker/parked set;
- age promotion prevents a continuously recurring conversation from starving a
  third conversation when engine capacity is two;
- an interrupt/control edge may outrank future numerical admission, but it does
  not preempt a running assembly node;
- dispatch always revalidates instance generation, conversation id, epoch, and
  scratch generation before mounting views on the lane board.

No fairness mechanism polls. Enqueue and quantum-return ring the broker's shared
expected-value doorbell.

### 4.2 Throughput and failure

- **Bandwidth is the decode ceiling.** M2 Max is 400 GB/s; the engine currently
  realizes ~66 of a ~250 GB/s practical bound. Decode is M=1 GEMV — every token
  streams the whole model, so tok/s ≈ model_bytes / bandwidth. The pool does not
  change that arithmetic, but deleting the duplicate keeps the working set from
  thrashing and keeps weights bf16 (half the bytes) in `(N,K)` (no repack). The
  win from zero-copy is measured in GB/s of avoided **activation** traffic.
- **Deadlines.** Doorbell waits take an absolute `deadline_ns`. A missed real-time
  deadline is a soft ticket cause — observable, not a crash.
- **Failure is terminal, not degraded.** No fallback. A rejected pass (stale
  epoch, unmet gate) is a terminal completion with a cause; the session decides
  (abort the turn), it never silently drops to Candle.

### 4.3 Deterministic graph-executor acceptance tests

The current exact-slot theft, stale-owner ABA, active-pass stop/drain, and live
slot accounting tests validate only the landed substrate. The graph cutover is
not complete until deterministic tests also prove:

1. **Plan validation:** bad node indices, undeclared cycles, missing kernel
   specializations, wrong view dtype/shape/stride, illegal aliases, and scratch
   or lease high-water overflow fail before model publication. The published
   plan is immutable and has a stable digest.
2. **Ordered graph math:** token → Depthformer → Mimi runs from one graph
   instance with full hidden/logit/code/PCM parity, while every node receives the
   exact prebound view and conversation scratch generation.
3. **More-than-capacity fairness:** at least three multi-node graph instances run
   through a capacity-2 engine with a quantum of one. A captured dispatch trace
   proves bounded round-robin/age promotion; no instance retains a pass slot
   across its quantum and all three make progress.
4. **Peer arrival during recurrence:** a peer submitted after node 1 but before
   node 2 is dispatched before the recurring instance can exceed its quantum.
   Repeating this for more than two graph edges produces the same trace.
5. **Backpressure split:** fill the reliable result ring and the playback ring,
   separately. The affected instance parks with zero engine slots retained; the
   sole bridge thread continues settling an unrelated ticket. Draining the exact
   ring wakes and resumes only the parked instance—no retry loop.
6. **Playback admission:** Mimi is never dispatched without a retained,
   capacity-checked playback reservation. EOAudio, stale epoch, fault, and stop
   each release an unused reservation exactly once; successful Mimi writes
   directly into and publishes that same reservation.
7. **Interrupt at every boundary:** advance epoch before submit, while queued,
   while running, after CQ before resubmit, while parked for reliable output,
   and while parked for playback. Accepted math may settle, but no stale graph
   publishes or takes another numerical edge.
8. **Stop and retirement:** stop from every state produces one terminal winner
   and joins with zero graph instances, pass slots, descriptors, result cells,
   capture leases, playback leases, and wait registrations. Owner-drop ordering
   runs under ASan/UBSan/TSan.
9. **No bridge blocking:** instrument the exact-CQ callback and fail if it calls
   a synchronous engine/model API, waits, allocates, formats text, reserves a
   dock slot, or invokes a host callback. A stalled session coordinator cannot
   prevent the bridge from consuming another accepted SQ/CQ pair.
10. **Steady state:** one million graph-node completions allocate nothing after
    readiness, use no timed waits/sleeps/spin, leave every lease counter at zero,
    and preserve the existing idle CPU gate on aarch64 and x86_64/Rosetta.

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
3. **SQ capacity ≥ 2 vs 1.** Enables recurrence-driven overlap, but multiple
   in-flight passes over one scratch arena require per-slot arenas. Start at
   capacity 2 (double-buffer), one arena per in-flight slot.
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

1. build and validate the immutable, direct-threaded LFM2 graph set at
   model-open, plus the fixed graph-instance/result-cell pools and fair ready
   broker;
2. route the landed coarse audio-encode, prefill, token, Depthformer, and Mimi
   passes through those eager graph instances, with pre-retained playback/text
   capacity and exact-CQ advancement, then delete synchronous coordinator waits;
3. replace the transitional Rust PCM/`VoiceEvent` adapter with physical kcoro
   audio-device docks;
4. physically move oracle/training sources into `liquid-audio-oracle`.

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
  together; the SQ gains depth (capacity 2) and an immutable eager graph chains
  decode → depth → Mimi through exact-CQ advancement; `pass_lock` and the
  blocking rims go. *Gate:*
  `compatibility_copied_bytes == 0` for LFM2; RAM halves; 1M-pass soak;
  at least three graph instances fair through the capacity-2 executor;
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
  resident image, then delete Candle from the shipped graph entirely. Until then
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
