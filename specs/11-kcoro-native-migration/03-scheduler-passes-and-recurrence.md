# Scheduler, Passes, Tickets, And Callback-Driven Recurrence

Status: normative design with the fixed-lane substrate at `d2c43abd`, Rust
coordination foundation at `3a5b1431`, native SQ/CQ leaf at `2a2adcea`, and
production bridge mount at `95069bd5`. Retained descriptor pooling is committed
at `fa35a624`, and first Rust broker/CQ ownership is mounted at `4f06a3d5`.
Service scheduling, scope wakes, owned pass slots, and product recurrence remain
incomplete.

Baselines: EmberHarmony `321538f11749`; `kcoro_arena` `447d04f0246b`.

Upstream contracts:

- `/Volumes/stuff/Projects/kotlinmania/kcoro_arena/docs/GPU_KERNEL_CONTRACT.md`
- `/Volumes/stuff/Projects/kotlinmania/kcoro_arena/docs/TICKETS_AND_CALLBACKS.md`

## Goal

Use two deliberately different resident executors:

- Rust kcoro futures for coarse orchestration, tickets, timers, audio policy,
  callbacks, cancellation, and workflows;
- a persistent fixed Flashkern worker team for model stages, shared scratch,
  tile fan-out, SIMD, and assembly.

Move token/frame recurrence into the resident Rust coordinator while every
numerical operation stays native. Rust publishes compact commands and consumes
compact completions through SQ/CQ rings. Tauri rings coarse policy tokens and
receives bounded metadata; it is never in a token, frame, stage, or ticket
completion loop.

The CPU runtime can do something a static GPU command list cannot: after one
full pass it can inspect live context, modality, workflow, deadline, and
interrupt state, then immediately recur, switch conversations, fork a branch,
or stop.

## Correction To The Earlier Lane-Frame Design

The earlier version of this document proposed N movable stackless lane
continuations and a `LaneFrame` that flattened the current six-deep C++ call
tower. That rewrite is rejected.

The previous engine had N logical lanes on N dispatcher threads. During
an intra-pass barrier, a lane cannot run unrelated model work because the active
pass cannot advance until all required lanes reach the stage boundary. Moving
those lanes onto the general kcoro ready queue would add continuation state,
worker migration, queue arbitration, and wake routing without exposing useful
parallelism.

The fixed team therefore keeps ordinary nested C++ call stacks. Rust kcoro
schedules the coarse pass ticket and receives the completion doorbell.
Flashkern schedules the numerical stage and tiles. No `LaneFrame` rewrite is
required.

This also follows from the serialization boundary. Conversation images are legal
only when no pass is active; the capture gate in document 11 requires quiescence,
and spec 10 excludes coroutine frames, barriers, thread IDs, and allocator state
at `specs/10-stateful-multi-agent-runtime.md:602-610`. There is therefore no
conversation-specific numerical call stack to serialize at a valid checkpoint.
Stackless explicit state belongs to the coordinator, workflows, timers, and
conversation image. The persistent compute service keeps ordinary stacks that
are empty of pass work whenever capture is admitted.

## Current Scheduler Map

| Current symbol | Evidence | Design action |
|---|---|---|
| `Pass` | `crates/liquid-audio/native/src/engine/flashkern_engine.cpp:82` stores borrowed pointers and shared scratch. | Current single-pass slot is pointer-stable. Promote it to an owned generation-protected slot pool before the Rust broker admits multiple producers or the compatibility caller returns asynchronously. |
| `Stage` | `flashkern_engine.cpp:119` uses one atomic tile claim counter. | Preserve as the micro-scheduler; add an active-lane mask only for plans that intentionally use a subset. |
| `Fence` | `flashkern_engine.cpp:134` stores arrival, logical generation, and park mask. | Implemented at `d2c43abd`: last arriver publishes generation and fans declared waiters through one shared expected-value word without spin. |
| Request kinds | `flashkern_engine.cpp:140-150` includes MLP, layer, token, and transitional callback requests. | Replace `REQ_CALL` with typed Depthformer/fan-out passes; keep pass-granularity ticket IDs. |
| Engine ownership | At `4f06a3d5`, `flashkern_engine.cpp:321-399` stores fixed pthreads, one mechanical SQ dispatcher, shared doorbells, one native bridge, request slots, plans, and scratch. `coordinator.rs:343-431` owns the Rust SQ broker and CQ ingress. Neither path stores a C arena runtime or ticket. | Keep this two-machine boundary. Move ticket identity/recurrence policy into Rust without routing stages or tiles through it. |
| `lane_fence` | `flashkern_engine.cpp:634-662`. | Implemented at `d2c43abd`: immediate shared expected-value block preserves generation/last-arriver correctness. |
| transitional `REQ_CALL` stage wait | `crates/liquid-audio/src/compute/flashkern/decode.rs:24-67`, `1001-1019` uses a local Rust `SpinBarrier` for `DepthDecode`. | This is the remaining active-spin exception. Port the program to a typed C++ pass and delete the barrier with `REQ_CALL`; add no new user. |
| `run_stage` | `flashkern_engine.cpp:670-684`. | Keep atomic disjoint tile claiming and one serial transition. |
| Nested lane program | `flashkern_engine.cpp:1009-1037`. | Keep ordinary C++ calls; port the remaining Rust callback bodies without flattening this tower. |
| Engine construction | `flashkern_engine.cpp:1201-1247` creates the native bridge, mechanical dispatcher, fixed pthreads, and two prepared lane wait words; `native_engine.rs:259-305` adds and registers the Rust coordinator. | Add readiness/affinity policy and million-pass soak. |
| External handback | At `4f06a3d5`, `submit_pass` at `flashkern_engine.cpp:1142-1190` invokes the registered Rust callback. `coordinator.rs:304-431` admits a fixed slot, submits through the sole broker, blocks on CQ ingress, and resolves the exact caller. | CQ ownership is complete. Replace borrowed `Pass` storage with owned pass slots, then make the public rim asynchronous and delete `REQ_CALL`. |

The coordination API is vendored under
`crates/kcoro-sys/vendor/kcoro_arena/include/`. Work and lifecycle conditions are
separate at `core/src/kc_runtime.c:225-324`; work and ticket arrivals signal one
worker. The ticket slab and intrusive completion queue live in
`core/src/kc_ticket.c:20-457`. The POSIX expected-value wait adapter is at
`port/posix.c:156-305`.

## Two Scheduling Levels

| Level | Owner | Unit of work | Worker identity |
|---|---|---|---|
| Macro | Rust kcoro coordinator | session command, actor step, timer, action ticket, full pass ticket, callback, context switch, snapshot capture request | movable Rust future on dedicated workers |
| Micro | Flashkern fixed executor | pass stage, GEMV rows, convolution channels, attention heads, FFT/mel bins, adapter rows | stable logical lane on stable OS worker |

A kcoro channel/ticket per tile, tensor operation, layer, or SIMD block is
forbidden. All fixed lanes observe one read-only pass descriptor, claim disjoint
tiles from one board, and write declared shared destinations.

## Target Object And Lifetime Graph

```mermaid
flowchart TB
    Runtime["LfmRuntime"]
    Coord["Rust kcoro coordinator"]
    Exec["FlashkernExecutor fixed lanes"]
    Model["immutable ModelPlan"]
    Session["LfmSession"]
    ConvA["Conversation A state"]
    ConvB["Conversation B state"]
    Parent["turn/frame action ticket"]
    Child["single full-pass ticket"]
    Slot["preallocated PassDescriptor slot"]

    Runtime --> Coord
    Runtime --> Exec
    Runtime --> Model
    Session --> Coord
    Session --> Exec
    Session --> ConvA
    Session --> ConvB
    Parent --> Child
    Child --> Slot
    Slot --> Model
    Slot --> ConvA
```

`LfmRuntime` owns the Rust coordination and native fixed executors. The model plan and
weight image are immutable. Each conversation owns mutable context. A pass
ticket retains its pass slot and context through completion-target consumption.
Session stop joins active tickets and both executors before releasing any owner.

## Target Coordination Runtime

Each product `LfmRuntime` owns a fixed-capacity `kcoro::Executor`; it does not
mount the C actor scheduler as its policy runtime. The C substrate remains the
conformance oracle and supplies native expected-value wait helpers.

The C oracle repair ledger is:

1. **Done (`bcdc03d1a073`):** split `work_cv` from `lifecycle_cv` in `kc_runtime`.
2. **Done (`bcdc03d1a073`):** signal one work waiter for each newly runnable continuation or
   terminal ticket; lifecycle transitions notify only lifecycle predicates.
3. **Done (`bcdc03d1a073`):** add a preallocated ticket slab, exact intrusive completion queue,
   arena-worker callbacks, cancel/deadline/stop disposition, and snapshots.
4. **Done (`bcdc03d1a073`):** add raw-word atomics and zero-spin expected-value waits with precise
   teardown in the host adapter.
5. **Not mounted in product:** unregistered `KORO_WAIT_UNTIL` at
   `include/kcoro_stackless.h:125-133`; every park owns a retained operation,
   ticket, timer, channel wait, or explicit doorbell subscription.
6. **Not mounted in product:** `actor_step` at
   `core/src/kc_actor.c:33-55` can otherwise monopolize a worker while a mailbox
   is continuously replenished.
7. **Open in the oracle:** make configured capabilities truthful; `core/src/kc_admin.c:11-18` currently reports
   optional durable/transport services unconditionally.

Rust commit `3a5b1431` implements fixed capacity, nonzero explicit worker count,
bounded draining, generation-protected task reuse, preallocated task wakers,
exact-once promises, inherited scope words, and edge-woken SPSC rings. Commits
`2a2adcea` and `95069bd5` mirror the protocol in C and mount a native-owned ring
leaf in Flashkern. `fa35a624` adds retained descriptors, and `4f06a3d5` mounts
the sole Rust SQ broker plus dedicated CQ ingress. Open work is scope-control
doorbell subscription, child-ticket recurrence, service-class fairness,
multi-board admission, and platform QoS binding.

Coordination-worker count and fixed kernel-lane count are separate persisted
runtime settings. A zero worker count is rejected; it never means one worker or
CPU autodetection.

## Target Fixed Flashkern Executor

```c++
struct FlashkernExecutor {
    std::vector<NativeThread> workers;
    std::vector<LaneState> lanes;
    StageBoard board;
    SubmissionQueue submissions;
    CompletionQueue completions;
    const ModelKernelTable *kernels;
    void *scratch;
    uint32_t lane_count;
};

struct LaneState {
    FlashkernExecutor *executor;
    uint32_t lane;
    uint32_t wake_sequence;
    uint32_t reserved;
    uint64_t logical_generation;
};
```

`LaneState` is not a coroutine stack replacement. It stores only stable lane
identity and observed doorbell generations. The nested kernel program remains
ordinary C++ and assembly.

Executor creation:

1. Allocate boards, pass slots, ticket capacity, completion capacity, lane
   state, and maximum scratch before readiness.
2. Start exactly `lane_count` persistent workers.
3. Each worker publishes its lane identity and blocks on the command generation.
4. The coordinator waits for all lane-ready edges once during startup.
5. Runtime readiness fails if any worker, required ISA, plan, wait-word adapter,
   or scratch reservation is unavailable.

Thread affinity and performance-core policy are host-adapter settings. Stable
logical lane identity is mandatory even when the OS declines a requested CPU
binding.

## Target Pass Descriptor And Ticket

```c++
enum class PassKind : uint32_t {
    Mel,
    Conformer,
    Adapter,
    BackbonePrefill,
    BackboneToken,
    Depthformer,
    MimiEncode,
    MimiDecode,
    MoshiFrame
};

struct PassDescriptor {
    uint32_t size;
    PassKind kind;
    uint64_t pass_id;
    uint64_t conversation_id;
    uint64_t epoch;
    const ModelPlan *model;
    ConversationState *state;
    const void *input;
    void *output;
    void *scratch;
    uint32_t input_count;
    uint32_t flags;
    uint32_t slot_generation;
};
```

The target slot is preallocated and remains alive through completion
consumption. A Rust child ticket owns its opaque native descriptor lease. The
submission queue copies only the fixed control cell containing that descriptor
ID; it never copies the descriptor or payload. Weights, activations, KV, PCM,
and scratch never enter either queue. The mounted path already has an eight-slot,
generation-protected descriptor pool whose queue lease survives CQ consumption.
Its descriptor payload is currently only `Engine*` plus the request kind; the
actual numerical pointers remain in one borrowed engine request slot. It is not
yet the owned `PassDescriptor` slot or region-lease graph described here.

The upstream baseline `koro_send_begin_ex` always copies through
`kc_descriptor_create_copy` at `kcoro_arena/core/src/kcoro_stackless.c:94-107`.
The product executor must use the new descriptor-transfer/ticket submission
surface, never this copy-mode helper.

### Target SQ/CQ Boundary

The executor boundary is an explicit submission-queue/completion-queue pair,
matching a command processor rather than a general actor channel:

```text
many session/workflow actors
    -> bounded Rust broker admission by service class
    -> policy selects one retained ticket
    -> bounded SPSC SQ: broker -> fixed executor
    -> one full native pass
    -> bounded SPSC CQ: final lane -> Rust completion continuation
    -> ticket terminal publication and exact Rust coordinator wake
```

The admission queue is not the SQ. It holds schedulable tickets while one board
is busy. ABI v1 may use a one-slot, generation-protected SQ because a board has
one dispatched pass. The CQ has completion capacity reserved before SQ
publication. SQ release-publication orders the pass descriptor before the first
lane wake; CQ release-publication orders state/output writes before coordination
reads them. The SQ copies one fixed control cell containing a
generation-protected native descriptor ID; the descriptor and payload remain in
place. The CQ copies terminal facts and at most eight token/codebook IDs.
Producer/consumer sequences are cache-line separated, and neither edge calls
copy-mode `KORO_SEND`.

SPSC describes logical endpoint ownership. The last-arriving lane may differ
between passes, but only one pass can complete at a time, the dispatch permit is
not restored until coordination consumes the prior CQ entry, and the next SQ
publication transfers completion-producer ownership through acquire/release
edges. If a later design overlaps passes, it must use independent SQ/CQ pairs or
a separately proven multiproducer completion structure.

The CQ rings the Rust continuation that makes progress. A Tauri or visualizer
callback is a separate, sampled observer after arbitration and can never be the
callback that makes computation progress.

Ticket terminal status distinguishes:

```text
execution:    not_dispatched | completed | failed
state:        none | committed | rolled_back | poisoned
publication:  none | committed | stale
cause:        success | rejected | canceled | timed_out | stale_epoch | stop | fault
```

An interrupt during a continuous-state pass may yield
`completed + state=committed + publication=stale`: generated thought remains in
model context while old-epoch text/audio and recurrence are suppressed. A
speculative candidate may instead yield
`completed + state=rolled_back + publication=stale`. Neither is an imaginary
half-kernel cancellation.

A fatal fault is different. Each pass plan declares discard, boundary-mark
restore, or poison. In-place state that cannot be restored is marked poisoned;
the coordinator cannot recur or snapshot it and may only destroy it or restore
a previously durable image.

## Target Stage Board

Each immutable plan stage declares:

```c++
struct StagePlan {
    StageKind kind;
    uint64_t active_lane_mask;
    uint32_t tile_count;
    uint32_t claim_grain;
    KernelEntry kernel;
    SerialEntry serial;
    uint32_t next_stage;
};
```

The containing immutable pass plan also carries a measured warm p99 dispatch
budget and a configured admission ceiling. These are scheduler facts, not
kernel branches.

Each plan binds an `extern "C"`, non-throwing fused tile thunk. C++ catches an
exception in that thunk and returns a typed fault; assembly leaf functions see
only raw pointers/shapes/strides and never call kcoro or publish board state.
Public C ABI records contain no `_Atomic` or `std::atomic` layout. The private
board stores cache-line-aligned integer words and uses one lock-free internal
`kc_atomic_*` helper family from both C and C++; it never casts a C11 atomic to a
C++ atomic or exposes the words through the product ABI. Runtime creation fails
the fixed-executor capability if required 32-bit atomic operations are not
lock-free. Assembly never touches board atomics.
One build chooses one board owner/helper implementation. It never mixes C11
atomic objects, compiler builtins, and C++ `atomic_ref` operations on the same
word.

On a typed fault, the first lane claims the board fault record and that lane
stops taking new tiles. Peer lanes do not poll the fault inside their tile loops;
they finish the ordinary stage claim loop, and every active lane still decrements
the stage countdown once. The last lane publishes no next stage, completes the
ticket as failed, and applies the plan's rollback/poison rule. No lane returns
early and strands a barrier. A hardware memory fault or illegal instruction
remains a process fault rather than a fictional recoverable ticket result.

The mutable board stores logical pass/stage generation, active mask, remaining
lane count, tile claim counter, pass pointer, fault word, one cache-line-isolated
shared dispatch word, one shared fence word, and a logical fence park mask. ABI
v1 admits 1 through 64 lanes through one `uint64_t` plan mask; the current mounted
engine admits 1 through 32 because its committed park mask is `uint32_t`.

Every lane:

1. reads the immutable stage plan;
2. claims tile ranges with one atomic fetch-add;
3. invokes the prebound kernel on disjoint destinations;
4. acquire-release decrements the active-lane countdown;
5. either runs the short serial transition as last lane or declares its bit,
   rechecks logical generation, and blocks on the shared fence word.

### Barrier economy and stage fusion

Immediate blocking is not permission to park between tiny operators. The model
plan emits a barrier only for a true cross-lane dependency, active-mask change,
scratch ownership transfer, or one serial transition. Consecutive lane-local
operations execute as one fused stage program.

Bias, residual, activation, elementwise epilogues, local reductions, and format
conversion are fused into their producer stage when numerical parity permits.
One barrier per tensor expression, helper function, or assembly call is
forbidden. Each immutable plan records declared stage/barrier count and active
mask so a model change cannot silently multiply host waits.

## Native Zero-Spin Barrier

`FENCE_SPIN = 8192` is removed from the native Flashkern fence. There is no
bounded spin, `YIELD`, `PAUSE`, WFE budget, UMWAIT budget, timed poll, or
repeated atomic-load loop in the C++ dispatch/fence waits. The transitional
Rust `REQ_CALL` exception is recorded in the current scheduler map and does not
weaken the target: it is deleted, not generalized.

```mermaid
flowchart TD
    Done["lane finished declared tiles"] --> Last{"release-decrement makes zero?"}
    Last -->|yes| Serial["run one serial transition"]
    Serial --> Publish["publish next stage and logical generation"]
    Publish --> Mask["exchange logical park mask"]
    Mask --> Wake["if nonempty: increment shared fence word and wake-all"]
    Last -->|no| Expected["read shared sequence, declare lane bit, recheck generation"]
    Expected --> Block["wait on shared word with expected value"]
    Block --> Recheck{"logical generation changed?"}
    Recheck -->|yes| Continue["acquire board and continue"]
    Recheck -->|spurious| Block
    Wake --> Continue
```

The wait-word adapter prepares two opaque handles per executor, one for dispatch
and one for stage fences, and provides expected-value/recheck semantics through
a direct futex, supported platform wait, or condition-variable fallback. Hot
waits and wakes perform no process-global address lookup.
A C++ adapter may use `std::atomic_ref<uint32_t>::wait/notify` over the aligned
raw word only when the selected library implementation is audited to block
immediately without a pre-block spin tier; it cannot reinterpret the word as a
separate `std::atomic<uint32_t>` object. An unsupported adapter disables the
fixed-executor capability; it does not fall back to spinning.

The last arriver exchanges the logical park mask. If it is nonempty, it advances
the shared fence word and issues one host wake-all for threads blocked on that
address. A lane that observes the new logical generation before blocking clears
its declaration and continues; expected-value semantics close the race. The mask
is exact logical waiter accounting, while the host wake is one fan-out operation,
not one syscall per lane. The kcoro coordination domain remains separate and uses
signal-one work permits.

After a non-last arrival decrements the countdown, it reads the shared fence
word, publishes its park bit, and rechecks logical generation. If the last lane
already advanced generation, it clears the bit and continues. If the word
advances during wait entry, the adapter's changed-before-wait check returns
immediately; waiting without the logical recheck would be a lost-wake bug.

### Memory-ordering contract

1. The broker writes the complete pass slot/first stage, release-publishes pass
   generation, advances the shared dispatch word, and wake-alls the fixed team;
   lanes acquire generation before reading any pass field.
2. Every lane finishes declared writes and acquire-release decrements the stage
   countdown; the last lane therefore acquires all prior lane publications
   before the serial transition.
3. The last lane release-publishes serial output and the next logical generation,
   exchanges the park mask, and advances the shared fence word once when that
   mask is nonempty; awakened lanes acquire generation before claiming work.
4. Final output writes and full-pass epoch disposition happen before the native
   executor release-publishes the CQ cell; the Rust ingress thread acquires that
   cell before resolving the mounted result slot or, after recurrence lands, the
   target terminal promise.

The host wait primitive may return spuriously but cannot weaken these edges.
SIMD/assembly kernels inherit synchronized pointer views and add no hidden
publication protocol.

## Target Full Pass

A full pass starts when the coordinator publishes one SQ cell naming a retained
pass descriptor and ends when all stages have reached a valid model-state
boundary and the reserved CQ completion cell has been published.

No host callback, Tauri event, disk write, CRC, cancellation poll, descriptor
allocation, or payload copy occurs during that interval.

The last pass lane performs bounded completion publication:

1. release-publish final destination writes and pass status;
2. compare the pass epoch with the native bridge's control/output epochs and
   apply the pass plan's committed, rolled-back, stale, or fault disposition;
3. fill the pre-reserved 128-byte CQ cell with the ticket ID, four terminal
   facts, status, and compact results;
4. release-publish that cell and ring one coordinator doorbell;
5. block on the next executor command.

It never invokes arbitrary callbacks. The Rust ingress thread drains the cell
and resolves the ticket promise outside native executor locks.

## Target Doorbells And Interrupts

An interrupt is an epoch doorbell:

1. A Rust scope transition or native VAD reflex atomically advances the relevant
   control/output epoch and rings one prepared doorbell.
2. It wakes the session coordinator or native reflex handler exactly once.
3. A queued old-epoch pass is canceled before dispatch.
4. Lanes finish one already-dispatched full pass.
5. At the full-pass boundary, native code compares the descriptor epoch with
   the current bridge/output epoch.
6. If stale, it discards unpublished output, applies the declared rollback or
   continuous-state policy, flushes old-epoch playback, and publishes the CQ
   cell as completed with committed or rolled-back state and stale publication.
7. The parent Rust action receives one terminal promise resolution and does not
   create another old-epoch child ticket.

Stop follows the same full-pass rule and has priority over queued prepare/start
work. There is no stop load inside `run_tile`, GEMV, attention, FFT, mel, layer,
or barrier code.

Explicit cancellation and hard publication deadlines also act only at a pass
boundary after dispatch. A queue-only deadline may reject before dispatch but
becomes a lateness metric once a pass starts; a soft deadline is always a metric.
A hard deadline or cancellation that arrives during execution allows numerical
completion, then suppresses publication/recurrence with committed or rolled-back
state as declared by the plan. Fault wins if no valid boundary exists. Otherwise
the unclaimed boundary precedence is runtime stop, stale epoch, explicit cancel,
then hard timeout. None of these are polled by a lane.

## Target Callback-Driven Recurrence

The current path returns logits to Rust so `Sampler` and
`generate_with_cache` choose the next token at
`crates/liquid-audio/src/model/lfm2_audio.rs:1308-1500` and `1630-1733`.
The target native pass owns sampling and state append. Its completion cell
carries only the selected token/codebook IDs and terminal facts. The Rust
coordinator owns the policy decision that follows:

```mermaid
stateDiagram-v2
    [*] --> Ready
    Ready --> Dispatch: create child pass ticket
    Dispatch --> Complete: fixed lanes finish, sample, and append
    Complete --> Stop: stop epoch observed
    Complete --> Stale: interrupt epoch observed
    Complete --> Commit: current epoch and compact result ready
    Commit --> Depth: audio result needs typed depth/codec pass
    Commit --> Text: text result continues token policy
    Depth --> Dispatch: depth/codebook pass
    Text --> Ready: append and recur
    Depth --> Decode: frame complete
    Decode --> Ready: publish playback and recur
    Commit --> Switch: another context has priority
    Switch --> Ready: replace context pointer
    Stale --> Ready: policy permits future input
    Stop --> [*]
```

The serial sampler runs only as a declared native serial stage. It writes the
selected token into native conversation state before CQ publication. Rust never
receives logits, probabilities, RNG state, or tensor views. The resumed Rust
continuation reads compact result IDs and submits the next descriptor. UI text
notification is a separate side effect and never gates the next pass.

## Target Conversation Switching And Branches

At any completed child ticket, the coordinator may replace the pass context
pointer while retaining the same immutable model plan and weights. It may:

- continue the active user conversation;
- run an affective, technical, or dissenting advisor branch;
- switch to another live user conversation;
- service a deadline-sensitive Moshi frame;
- capture a quiescent context generation for the snapshot writer.

Copy-on-write context pages make branch memory cheap, not compute free. Policy
uses deadlines and bounded branch token budgets. A cognitive parent ticket joins
required and optional branch tickets and produces a semantic capsule, never a
raw KV merge.

### One broker, many hot contexts

One Flashkern stage board executes one full pass at a time. A Rust
`KernelBroker` future is its sole command producer. Session, frame, and cognitive
futures submit retained child tickets through bounded service-class admission;
the broker publishes one fixed command cell naming a native pass slot, and the
fixed team returns one fixed completion cell through the CQ.

Admission uses preallocated ticket slots, not `KORO_SEND` or a copied payload
descriptor. One ticket can be admitted to only one broker; bounded queue
rejection leaves caller ownership and every native lease unchanged.

The broker schedules deadline-sensitive audio first within its service class,
then interactive work, then bounded advisor/maintenance work. Runtime settings
define maximum consecutive passes and time quantum per conversation. Recurrence
keeps the current context only if its quantum remains and no more urgent ticket
is ready; waiting tickets receive age promotion. Stop and old-epoch cancellation
are applied before the next dispatch.

The quantum is checked only at full-pass boundaries. The broker compares each
plan's measured pass budget with deadline slack before dispatch; it rejects or
defers work that cannot meet policy rather than polling inside a kernel. Long
prefill is split only at state-valid suffix-cache block boundaries, with one
single-shot child ticket per block. It is never split at arbitrary operators or
SIMD loops. The longest admitted pass is therefore the ordinary stop/context
switch latency floor and must stay inside the release budget.

This is simultaneous conversation residency and interleaved execution, not a
claim that one shared scratch board runs two passes concurrently. Multiple
passes may execute at once only when the runtime creates independent executors
with separate workers, boards, scratch, and broker bindings.

## `REQ_CALL` Disposition

`REQ_CALL` at `flashkern_engine.cpp:1283-1291` lets Rust callbacks execute on the
fixed lane team. Current users enter through `NativeEngine::run_lanes`/`grid`
starting at `crates/liquid-audio/src/compute/flashkern/native_engine.rs:544`.
`DepthDecode::frame` currently constructs a local `SpinBarrier` at
`decode.rs:1001-1019`; therefore the mounted engine is zero-spin while idle and
at native C++ fences, but not yet at every internal stage of this Rust callback
program. That exception is one reason `REQ_CALL` cannot survive cutover.

Migration rules:

1. Treat every current production call site as explicit migration debt; add no
   new caller.
2. A callback is non-suspending, non-reentrant, and cannot unwind
   across C.
3. It runs to completion on its lane's ordinary OS stack and may not call kcoro,
   Tauri, storage, or model recurrence.
4. Port every production callback into a typed native pass.
5. Delete `REQ_CALL`, its Rust trampoline, and production callback request kind
   when the last call site is gone. Do not preserve a legacy mode.
6. At `d2c43abd`, fixed lanes, expected-value waits, and deletion of the
   stackful dispatcher, saved stacks, and context-switch assembly are complete.
   They do not require flattening or preserving `REQ_CALL`.

## Source Changes

1. **Done (`8d510f83`):** vendor arena `bd530f4c9196` explicit runtime, ticket,
   and wait-word contracts into `crates/kcoro-sys`; delete the old runtime tree.
2. **Done oracle (`bcdc03d1a073`):** split work/lifecycle waits and add the C
   ticket slab/completion queue and exact callback fixtures. The product mount
   no longer links that ticket/runtime path; retained-descriptor channel
   transfer and actor fairness remain oracle cleanup, not product dependencies.
3. **Done (`3a5b1431`):** add `crates/kcoro` with the bounded
   Rust executor, exact promises, scope words, protocol records, and SPSC edge
   semantics.
4. **Done first mount (`2a2adcea`, `95069bd5`, `fa35a624`, `4f06a3d5`):** add
   and mount the private native-owned SQ/CQ leaf with prepared doorbells, CQ
   reservation, stop races, exact final-lane publication, retained descriptors,
   one Rust broker, and dedicated CQ-to-slot/promise routing. Add service queues,
   Rust-owned child recurrence, and scope-control doorbells next.
5. **Open:** port all production `REQ_CALL` users into typed native passes, keep
   sampling native as specified in document 07, and delete the Rust lane
   trampoline.
6. **Partly done (`d2c43abd`, `95069bd5`, `4f06a3d5`):**
   `flashkern_engine.cpp:321-399` owns fixed workers, one pointer-stable request
   slot, stage board, two shared zero-spin lane words, and the native SQ/CQ leaf.
   The Rust broker now owns endpoints; preserve this executor and do not
   flatten `lane_program` into stackless PCs.
7. **Done (`d2c43abd`):** the fence at `flashkern_engine.cpp:634-662` uses the
   shared raw-word atomic helper, logical park mask, and immediate expected-value
   block. Address identity is covered by upstream and Cargo wait-word tests.
8. **Done (`d2c43abd` ancestry):** remove the stackful dispatcher, lane-stack
   allocation, old vendor tree, and context-switch assembly; retain only OS
   worker stacks.
9. **Partly done (`fa35a624`, `4f06a3d5`):** each blocking compatibility
   submission carries independent ticket and descriptor generations, owns a
   native CQ reservation before dispatch, and resolves one preallocated Rust
   result slot. Rust-owned parent/child ticket policy, scope identity, and
   recurrence remain open.
10. **Open:** add post-transition ticket projections and generation-checked periodic board
   sampling per document 12; no UI callback enters this executor, and an
   inconsistent board read is skipped rather than retried.
11. **Open:** delete the blocking Rust request surface and every `REQ_CALL`
    artifact after independent fixtures pass; keep no product or source fallback.

## Acceptance Gates

- Every fixed worker publishes one stable lane identity and immediately blocks
  before runtime readiness.
- One million idle/start/pass/idle cycles produce one terminal child ticket and
  one parent wake each.
- Delay every lane before park-mask publication, after declaration, around
  expected-value wait entry, and around the last arrival; no shared sequence or
  logical generation is missed or consumed twice.
- Compiled wait paths contain no spin loop, `PAUSE`, `YIELD`, WFE/UMWAIT budget,
  or timed polling. Idle and barrier wait CPU are zero within measurement noise.
- C/C++ layout, address-identity, memory-order litmus, and TSan tests prove every
  board word uses one selected atomic helper and the wait adapter blocks on that
  exact storage without reinterpret-casting an atomic object.
- One Rust coordination enqueue wakes one worker and never polls one
  continuation concurrently. The C oracle's `finish_cont`/`suspend_cont` and
  lifecycle tests remain conformance fixtures. One nonempty fence park mask
  causes one shared host wake; logical waiter count equals the mask population
  and no coordination worker is touched by that fan-out.
- SQ/CQ full, wrap, stale-generation, stop, and completion races never overwrite
  an entry; each accepted dispatch already owns one CQ reservation.
- Declared barriers correspond one-for-one with audited cross-lane dependencies;
  fused stage, wait-registration, host-block, and wake counts remain within the
  recorded model-plan budget.
- No allocation or payload copy occurs from pass publication through terminal
  callback after warmup.
- Pointer instrumentation proves the pass slot, model region, context pages, and
  output reservations retain the same addresses across submission and lane use.
- Stop before dispatch yields no kernel entry. Stop/interrupt during a pass
  permits one full completion and no old-epoch recurrence.
- Queued prepare/start is skipped when stop has already advanced the epoch.
- Current attention, convolution, MLP, full token, Depthformer, fan-out, and
  temporary `REQ_CALL` parity tests pass during migration.
- The final production symbol/call-graph audit contains no `REQ_CALL`, Rust lane
  trampoline, stackful kcoro context switch, or saved lane-stack allocation.
- A recurrent 1,000-token loop performs one exact CQ-to-Rust continuation edge
  per declared pass and zero Tauri, webview IPC, polling, or observer edges; it
  responds at the next full-pass boundary.
- Two hot conversations alternate at every pass without changing a weight
  address or corrupting state.
- A Rust continuation cannot run on a compute/audio/store thread, overlap
  itself, or fire after joined destruction.
- Actor flood tests cannot starve completion publication, timers, or stop.
- ASan, UBSan, and TSan report no concurrent lane identity, retained-ticket reuse,
  use-after-free, missed wake, or live object after teardown.

## Non-Goals

- No stackless `LaneFrame` rewrite of the nested numerical program.
- No actor or movable continuation masquerading as a hardware lane.
- No channel/ticket per numerical tile or tensor operation.
- No per-operation cancellation polling inside a pass.
- No spin tier, even when a measured barrier is usually short.
- No persistence, WAL, CRC, compression, Tauri callback, or disk write inside a
  model pass.
- No callback from a fixed compute lane into Rust or TypeScript.
