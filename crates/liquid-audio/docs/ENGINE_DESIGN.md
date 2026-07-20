# Flashkern engine design

Status: current implementation plus explicitly marked next steps.

## Boundary

Flashkern is the CPU inference device:

- native C++ owns the engine object, plans, request/pass slots, descriptors,
  queues, direct byte views, scratch, and numerical lifecycle;
- kcoro owns every resident control worker and fixed-team member, their idle
  dormancy, stop, and join;
- architecture assembly owns numerical values;
- Rust docks platform PCM and control through opaque native handles only;
- Metal is not part of Flashkern.

An operation never owns a sleeping thread. Its suspension is a durable
`PassSlot`, route, conversation, or session record. A callback edge makes that
record runnable. Only a resident kcoro worker whose complete ready predicate is
empty may enter expected-value dormancy.

## Current Command Flow

```mermaid
sequenceDiagram
    participant S as Native session/caller
    participant R as Route/service
    participant SQ as Ticketed SQ
    participant T as kcoro fixed team
    participant CQ as Native CQ

    S->>R: retain input leases + create route ticket
    R->>SQ: publish validated pass descriptor
    SQ-->>T: dispatch generation
    T->>T: claim assembly tiles; every member returns once
    T-->>R: final return invokes pass continuation
    alt another route label
        R->>SQ: publish next generation on the same ticket
    else terminal outcome
        R->>CQ: publish exact ticket completion
        CQ-->>S: make retained session delivery runnable
        S->>S: validate ticket/epoch + release leases
    end
```

`bridge_service_main` validates submissions and drains completions;
`bridge_team_complete` is kcoro's one final-return callback for a dispatched
generation. `PassSlot::ProgramCursor` and request-specific records own every
value that survives a return. The Rust submitter callback and Rust numerical
coordinator are gone.

An accepted submission carries only `{pass_slot, ticket_generation}`. The
engine-owned slot retains its typed byte views, program cursor, continuation,
and input/output/conversation leases until the exact terminal callback. There
is no descriptor registry or hot-path lookup lock. Slot generation advances
only after the callback releases or resubmits it; no callback context points
into a caller's stack.

## Fixed Lanes

`lfm_engine_new` creates a retained kcoro bridge service, a retained route
service, and one stable `kc_team`. Flashkern does not create lane pthreads.
The team owns one cache-isolated idle event and one generation-return counter;
there is no per-pass or per-stage fence word.

Every member executes the same published stage program. Members fetch-add
disjoint tiles and return after their complete assembly leaf. The final return
runs the bounded transition exactly once. It either publishes the next stage
generation or the terminal CQ record. Members do not block one another at a
barrier; after returning they are simply available for the next generation and
become dormant only if the entire team has no work.

## Math ABI

C++ routes pointers, dimensions, strides, and stage identity. It does not own
the numerical ladder.

Current hand-written assembly files include:

- `native/kernels/aarch64/flashkern_math.S`
- `native/kernels/x86_64/flashkern_math.S`
- `native/kernels/{aarch64,x86_64}/flashkern_prng.S`
- `native/kernels/{aarch64,x86_64}/flashkern_rope.S`

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

## State And Memory

The engine currently owns grow-only vectors for model plans and scratch. Final
design requires all model-sized allocation before readiness:

- immutable model and Depthformer plans;
- per-lane panels and temporary accumulators;
- QKV, attention, FFN, logits, sampler, FFT, and codec scratch;
- generation-protected pass slots and descriptor table;
- conversation-owned KV, convolution carry, sampler and codec state.

No pass may resize a vector, allocate a stack-dependent variable-length buffer,
or throw across `extern "C"`. Plan construction tracks maxima across every layer,
not only the final layer geometry.

Weights remain views into the resident aligned model image. Activations and
state mutate in declared native buffers. SQ/CQ records contain only fixed control
facts and IDs.

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

Missing PCM, playback capacity, or reliable-output capacity leaves a durable
route/session record dormant and releases the compute slot. The corresponding
producer callback makes it runnable. No host loop, timeout, or thread waits for
those resources. Rust handles only platform-audio and control edges.

## Teardown

Stop closes admission and marks retained services retiring. Accepted work
settles into terminal CQ records, stale epochs lose publication authority, and
all retained leases release exactly once. Services join before the fixed team;
the team joins before its idle registration is released; the bridge destroys
only when submissions, completions, routes, pass slots, and retained I/O leases
are all settled.

There is no production synchronous compatibility wrapper. Concurrent callers
either acquire a generation-protected slot or receive a bounded admission
result before mutating payload state.

## Verification

Current tests include:

- `raw_engine_owns_its_sq_cq_without_rust_progress`;
- `native_engine_bridge_and_fence_soak` (10,000 passes);
- exact MLP, conv, attention, PRNG, sampler, FFT, GEMM, and plan-lifetime tests;
- `scalar_assembly_math_abi_is_bit_exact_without_simd_feature_gates`;
- native AArch64 and local Rosetta x86_64 execution.

Required cutover gates:

1. one million passes with exact descriptor/slot settlement;
2. stop during every submit/dispatch/final-return/CQ phase;
3. zero allocation after readiness;
4. no C++ production numerical expressions;
5. no Rust model/numerical symbol in the release graph;
6. two or more conversations scheduled fairly over one model image;
7. p50/p95/p99/max callback and pass latency against frozen baselines;
8. ASan, UBSan, Linux TSan, AArch64, x86_64, and Rosetta gates.

Source-shape gates additionally reject production `wait_submitted_slot`, raw
lane pthread creation, operation-scoped address parking, timer-driven progress,
caller-stack continuation state, and completion channels.
