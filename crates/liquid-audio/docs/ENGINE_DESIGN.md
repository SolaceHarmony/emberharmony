# Flashkern engine design

Status: current implementation plus explicitly marked next steps.

## Boundary

Flashkern is the CPU inference device:

- native C++ owns the engine object, plans, request/pass slots, descriptors,
  queues, direct byte views, scratch, and numerical lifecycle;
- kcoro owns every resident control worker and fixed-team member, their idle
  dormancy, stop, and join;
- architecture assembly is the primary numerical implementation, with Apple
  Accelerate/AMX admitted only behind an explicit large-matrix ABI;
- Rust docks control/observation through opaque native handles only; native
  platform callbacks own PCM;
- Metal is not part of Flashkern.

Production enters through `lfm_runtime_create`. That implementation creates the
private engine with `lfm_engine_new_status` so deadline-backend failure can be
reported before work is admitted. The deterministic manual-deadline
constructors are private test interfaces only.

An operation never owns a sleeping thread. Its suspension is a stackless frame:
saved program counter, fixed locals, exact ticket, and retained references to
its `PassSlot`, route, conversation, or session records. A correlated callback
makes that exact frame runnable on any free eligible worker. Only a resident
kcoro worker whose complete ready predicate is empty may enter expected-value
dormancy.

## Current Command Flow

```mermaid
sequenceDiagram
    participant S as Native session/caller
    participant R as Route/bridge frame
    participant SQ as Ticketed SQ
    participant T as kcoro fixed team
    participant CQ as Native CQ

    S->>R: retain input leases + create route ticket
    R->>SQ: publish validated pass descriptor
    SQ-->>T: dispatch generation
    T->>T: claim assembly tiles; every member returns once
    T-->>R: final return resumes exact bridge ticket
    alt another route label
        R->>SQ: publish next generation on the same ticket
    else terminal outcome
        R->>CQ: publish exact ticket completion
        CQ-->>S: make retained session delivery runnable
        S->>S: validate ticket/epoch + release leases
    end
```

`bridge_continuation_step` validates submissions and drains completions from a
fixed saved frame. `bridge_team_complete` only publishes the completed
generation and resumes that exact continuation; it never shepherds the next
stage inline. `PassSlot::ProgramCursor` and request-specific records own every
numerical value that survives a return. The Rust submitter callback and Rust
numerical coordinator are gone.

An accepted submission carries only `{pass_slot, ticket_generation}`. The
engine-owned slot retains its typed byte views, program cursor, continuation,
and input/output/conversation leases until the exact terminal callback. There
is no descriptor registry or hot-path lookup lock. Slot generation advances
only after the callback releases or resubmits it; no callback context points
into a caller's stack.

## Fixed Lanes

`lfm_engine_new_status`, reached through `lfm_runtime_create`, creates one
bounded kcoro runtime, a saved bridge continuation, route and team-supervisor
services, and one stable logical `kc_team` on that same pool. Flashkern and
`kc_team` create no lane pthreads. The runtime owns one infrastructure
doorbell; there is no operation-owned idle registration or per-pass fence word.

Every member executes the same published stage program. Members fetch-add
disjoint tiles and return after their complete assembly leaf. The final return
publishes one edge to the bridge frame. The resumed bridge transition either
publishes the next stage generation or the terminal CQ record. Members do not
block one another at a barrier; after returning they are simply available for
the next generation and workers become dormant only if the entire pool has no
work.

This is one team. The current engine may describe logical blocks for accounting,
but it does not own two independent four-lane teams and cannot run two numerical
programs concurrently. Independent `BlockDomain`s remain an unimplemented V2
step.

## Math ABI

C++ routes pointers, dimensions, strides, and stage identity. That is its target
steady-state role; the remaining value-producing C++ called out below is an
open transliteration gap, not an alternate numerical tier.

Current hand-written assembly files include:

- `native/kernels/aarch64/flashkern_math.S`
- `native/kernels/x86_64/flashkern_math.S`
- `native/kernels/{aarch64,x86_64}/flashkern_prng.S`
- `native/kernels/{aarch64,x86_64}/flashkern_rope.S`
- `native/kernels/{aarch64,x86_64}/flashkern_sampler.S`
- `native/kernels/{aarch64,x86_64}/flashkern_frontend.S`
- `native/kernels/{aarch64,x86_64}/flashkern_conformer.S`
- `native/kernels/{aarch64,x86_64}/flashkern_sesame.S`
- `native/kernels/{aarch64,x86_64}/flashkern_capture_format.S`

`flashkern_math.S` currently owns reciprocal RMS scaling, fixed-order f32
reduction, strided BF16 sum-of-squares, BF16 bias addition, and exact BF16 NeoX
rotary. Existing value-producing C++ code elsewhere in the engine and
architecture `.cpp` files is migration debt and must move to assembly; it is not
a sanctioned fallback tier.

Production assembly leaves:

- never allocate, throw, call Rust, publish a ticket, inspect stop state, or
  perform I/O;
- receive counted raw planes whose lifetime is retained by the pass slot;
- write only declared disjoint destinations or a fence-owned serial result;
- preserve the documented rounding and reduction order;
- expose the same C ABI on AArch64 and x86_64.

AMX/Accelerate remains the Apple matrix coprocessor. Its invocation must sit
behind the architecture math ABI; C++ may select the leaf but may not prepare or
evaluate model values in the pass scheduler.

The private loader type named `LfmTensorView` is metadata over checkpoint bytes,
not an owning tensor. Production never constructs a framework tensor or loads a
checkpoint in Rust.

## State And Memory

Setup-time C++ containers build immutable plans, formula-derived tables, and
workspace high-water marks. Production activation workspaces are reserved and
sealed before session readiness. All model-sized allocation belongs before
readiness:

- immutable model and Depthformer plans;
- per-lane panels and temporary accumulators;
- QKV, attention, FFN, logits, sampler, FFT, and codec scratch;
- generation-protected pass slots and descriptor table;
- conversation-owned KV, convolution carry, sampler and codec state.

No pass may resize a vector, allocate a stack-dependent variable-length buffer,
or throw across `extern "C"`. Plan construction tracks maxima across every layer,
not only the final layer geometry.

Weights remain views into the resident model image; an individual checkpoint
view may be unaligned and must be loaded safely by its architecture leaf.
Activations and state mutate in declared native buffers. SQ/CQ records contain
only fixed control facts and IDs.

## Recurrence

Recurrence belongs to a native route/session continuation:

1. acquire a pass slot;
2. retain model/conversation/input/output leases;
3. publish native SQ;
4. return to the runtime with durable route state;
5. receive the final-team callback and publish the exact CQ;
6. arbitrate terminal facts and commit/rollback state in the retained
   continuation;
7. release the slot or submit the next labeled native action.

Missing PCM, playback capacity, or reliable-output capacity leaves the exact
route/session frame dormant and releases the compute slot. Its retained records
remain backing data; the corresponding producer callback makes the saved frame
runnable. No host loop, timeout, or thread represents those resources. Rust
handles only platform-audio and control edges.

## Native Audio Policy

The session consumes typed capture-chunk records over its preallocated native
PCM arena. Every 20 ms of incoming samples, it runs the paired architecture
Sesame leaf over the exact 256-sample Blackman-windowed 600–2400 Hz view. The
detector, adaptive microphone state, sample-count thresholds, and pause
generation belong to the native session. Rust does not run an RMS VAD.

The 200 ms prepare gate currently records retained policy readiness only.
Candidate-owned activation scratch and speculative numerical execution remain
open work.

The detector also implements separate playback adaptive state, but the product
session currently calls only the microphone stream. Feeding exact playback
evidence through Sesame remains an open integration gate; Rust output RMS is
telemetry only.

## Correlated Deadlines

Each numerical team generation is hard-supervised by a readiness-time
`kc_deadline_source`. On macOS it uses a monotonic GCD one-shot; non-Apple
production runtime construction returns `LFM_STATUS_UNSUPPORTED` before
admission, while private tests use a deterministic manual backend.

Before dispatch, the engine copies pointer-free ticket, pass, stage, shape, and
generation identity into retained supervision state and arms a one-second hard
deadline. Each team member release-publishes its own generation-stamped entry
and return. Normal final return retires the exact deadline. Completion and
expiry race through one terminal CAS, so neither path can publish twice.

If expiry wins, the supervisor captures the expected, entered, returned,
never-entered, and entered-not-returned masks in a reserved fatal capsule,
suppresses CQ/recurrence/scratch retirement, and aborts. There is no numerical
retry or potentially-live scratch reuse.

The mechanism is landed, but two release requirements remain open: the capsule
is not yet exported to a durable observable crash sink, and the one-second
budget is a provisional floor rather than a calibrated per-stage/shape value.
Soft nudge/rebroadcast behavior is not part of production.

## Teardown

Stop closes admission and resumes the bridge frame. Accepted work settles into
terminal CQ records, stale epochs lose publication authority, and all retained
leases release exactly once. The canceled-and-empty SQ is the terminal edge
that retires the logical team; the runtime then owns only terminal physical
worker teardown. The bridge frame is destroyed only when submissions,
completions, routes, pass slots, and retained I/O leases are all settled.

There is no production synchronous compatibility wrapper. Concurrent callers
either acquire a generation-protected slot or receive a bounded admission
result before mutating payload state.

## Verification

Current tests include:

- `kcoro-sys/tests/fixed_team.rs` for fixed membership, callback completion,
  quorum snapshots, and stop/join ownership;
- `engine_hard_supervision.rs` for deadline retirement, terminal arbitration,
  and fatal-capsule content;
- `sesame_detector.rs` for exact browser evidence, circular windows, separate
  stream state, and malformed input;
- `native_voice_session.rs` for PCM leases, exact sample-clock policy, pause
  deadline races, callback failure, stop, and no-operation-wait source gates;
- `native_product_abi.rs` for the opaque production export allowlist;
- an explicitly ignored real-checkpoint truth gate that drives typed input and
  two audio turns on one retained conversation through native audio-token
  generation and Mimi playback. Direct model-to-model conversation remains a
  future native audio-token/code dock, never an acoustic VAD loopback.

Required cutover gates:

1. one million passes with exact descriptor/slot settlement;
2. stop during every submit/dispatch/final-return/CQ phase;
3. zero allocation after readiness;
4. no C++ production numerical expressions;
5. no Rust model/numerical symbol in the release graph;
6. two or more conversations scheduled fairly over one model image;
7. p50/p95/p99/max callback and pass latency against frozen baselines;
8. ASan, UBSan, Linux TSan, AArch64, x86_64, and Rosetta gates;
9. observable platform fatal diagnostics and benchmark-calibrated hard budgets;
10. playback-fed Sesame/echo evidence instead of host RMS policy.

Source-shape gates additionally reject production `wait_submitted_slot`, raw
lane pthread creation, operation-scoped address parking, timer-driven progress,
caller-stack continuation state, and completion channels. Fixed heap-backed
stackless frames are required rather than forbidden.
