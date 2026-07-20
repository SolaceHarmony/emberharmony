# 16 — Flashkern V2: The Eager Coroutine Grid

Status: **V2.0–V2.1 and kcoro ownership of the one fixed numerical team are
landed. V2.2 block extraction and every form of independent block execution
remain unimplemented.** The working engine has `block_count == 1`, one stage
board, one scratch mount, and one active team generation. The proposed V2
extracts two independent logical four-lane blocks that can gang back into an
eight-lane team, then drives them with design 14's compact forwarding table. It
is an eager message-routed compute fabric, not a lazy tensor system, DAG VM, or
promise of macOS hardware placement.

**Landed through V2.1 at `1f6d1c5d4339`.** Hot kcoro, engine, bridge, session,
and model-gate words plus internal SQ/CQ storage cells now have 128-byte Apple
base alignment and stride while ABI-v1 values retain 64-byte caller alignment;
current request, layer, and modality selectors are closed; invalid
worker/logical-lane geometry rejects; four physical workers reproduce the
eight-way logical fold; and the zero-spin gate is green. A bounded production
`TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE` route now advances through exact CQs
with a total three-node/four-outcome table, reserve-before-admit playback, and a
direct Mimi write into the retained PCM span. That cited revision still exposed
a synchronous terminal collection seam and stack-scoped callback state. The
working-tree route/session conversion replaces both. They are historical
implementation debt, not a permitted fallback design.

**Broker and session follow-on landed in the working tree.** Routes now come
from a fixed 64-instance pool, matching the maximum session count. A native
callback-driven broker creates one ordinary
descriptor per coarse program, applies FIFO sequence order with bounded age
promotion, and reacquires capacity only when the node is runnable. The exact-CQ
callback commits declared state, releases the pass slot, and marks the next node
ready; it never allocates or takes a submission/descriptor mutex. Text uses the
same pool as a terminal single-node sampled-token route. Terminal notification
makes the retained session delivery continuation runnable; its
coordinator-owned `SessionAction` performs exact-generation collection without
installing an operation waiter for numerical or playback capacity.

**No `BlockDomain` exists in the working tree.** The current `gang_lease` is an
exclusive lease over the one fixed team, not proof of two completion domains.
There are no private per-block stage boards, scratch mounts, CQs, return
counters, ready rings, or simultaneous numerical programs. Earlier synthetic
two-block completion accounting did not establish the ownership boundary and
must not be cited as V2.2 implementation evidence.

**Kcoro ownership follow-on implemented in the working tree.** The stable
numerical members are created, generation-dispatched, stopped, and joined by
`kc_team`; Flashkern no longer owns lane pthread lifecycle. The current
completion seam counts fixed-team returns, and the final return invokes the
bounded continuation exactly once. The continuation advances durable pass
state and dispatches the next route label or publishes the terminal ticket.
No peer is suspended on behalf of that pass. Resident team members become
dormant only when the engine as a whole has no runnable generation. The team is
still one ganged execution domain; this is thread ownership, not yet two
independent block executors.

## 0. Ground truth and its limits

The audited M2 Max reports:

```text
hw.perflevel0.physicalcpu  : 8
hw.perflevel0.cpusperl2    : 4
hw.perflevel0.l2cachesize  : 16777216
hw.perflevel0.l1dcachesize : 131072
hw.cachelinesize           : 128
hw.memsize                 : 34359738368
FEAT_BF16=1  FEAT_I8MM=1  FEAT_LSE=1  FEAT_LSE2=1
FEAT_SME=0   FEAT_SME2=0
```

These values justify testing a `2×4` software geometry and require 128-byte
cache-line isolation on every Apple slice. They do **not** prove that macOS will
place four workers on one physical cluster, that an allocation remains in one
L2, or that one Apple matrix unit is reservable per block. Apple AMX is
undocumented and Accelerate is an opaque backend. The topology earns an
experiment, not a placement promise. Correctness cannot depend on any of those
hypotheses.

## 1. What V1 actually leaves open

### 1.1 Cache-line bug — repaired

V1's 64-byte `WaitWord` let adjacent `dispatch_word` and `fence_word` share a
128-byte Apple line despite the isolation assertion. V2.0 now gives every hot
kcoro ring, engine, bridge, session/model-gate, and SQ/CQ storage cell 128-byte
base alignment **and member/array stride** on arm64 and x86_64 under Rosetta.
The ABI-v1 command/completion value types remain 64-byte aligned. The supported
Apple slices use a conservative compile-time 128-byte storage law; a runtime
observation may be reported separately but cannot define C++ object alignment.

### 1.2 One stage board means one numerical pass

V1 has one active request, stage board, fence generation, mixer, and fixed lane
team. Its capacity-two SQ provides queueing, exact-slot recurrence, and dispatch
overlap; it does not execute two numerical passes concurrently. Running commit
and Mimi “on two slots” therefore serializes today. Two queue slots are not two
execution domains.

An eight-lane fence is also wider than the measured four-cores-per-L2 topology
suggests may be ideal, but there is no proof that the eight workers currently
occupy two particular clusters. Splitting the logical domain can reduce shared
fence traffic and expose independent work; locality remains a measured outcome.

## 2. The gangable `2×4` grid

The GPU analogy is a software ownership model, not a hardware equivalence claim:

| Concept | V1 | V2 contract |
|---|---|---|
| grid | absent | broker plus two logical blocks |
| block/threadgroup | one eight-lane domain | two independent four-lane domains |
| gang | implicit whole team | explicit lease over both blocks for one eight-lane program |
| shared memory | engine-owned scratch | block-owned scratch mount; cache residency unpromised |
| matrix backend | serial opaque call | initially one global permit; profile before widening |
| completion | one CQ | one SPSC CQ per block, drained by one broker |
| recurrence | exact completion callback | total route outcome selects the next coarse program |

Each `BlockDomain` owns its active command, stage board, fence and dispatch
words, scratch mount, SQ, SPSC CQ, and exact doorbell. Before two-block execution
is enabled, both private CQs and exact ticket lookup must exist. A shared
multi-producer CQ would change the ownership contract, not simplify it.

The broker uses this deterministic policy:

- `GANG8` reserves both blocks and mounts a dedicated eight-lane board.
- `BLOCK4` may occupy either free block.
- With one latency-critical runnable instance, dispatch immediately; never
  delay it to manufacture parallelism or a batch.
- With two independent runnable conversations, one `BLOCK4` program may run on
  each block.
- Initially, two state-mutating programs from the same conversation never
  overlap. Later relaxation requires an explicit disjoint `AccessSet` and a
  dedicated test. “Probably disjoint” is not an access contract.
- No numerical program performs a cross-block barrier. A real join settles
  through exact CQs and the route instance outside both block domains.

The landed V2.0 gate already proves four physical workers can process eight
fixed logical partitions with the same fold order on the current single board.
Block extraction preserves and reruns that proof. Output parity, not worker
count, decides whether a program may use `BLOCK4`.

## 3. Fixed-member returns and lane coroutines

A mounted numerical stage is one fixed-team generation identified by
`{ticket, route_label, stage, generation, team}`:

1. every team member observes the same stage descriptor;
2. lanes claim disjoint tiles only after generation publication;
3. every member returns exactly once after completing its assigned tile work;
4. the final return runs the declared bounded mixer/transition exactly once;
5. that callback advances durable pass state and either dispatches the next
   generation or publishes the one terminal ticket.

Once mounted, each member must return exactly once before it may accept another
generation or retire. Coroutines sequence only completed generations; assembly
owns a complete tile and never yields inside a kernel. Large parallel transforms
remain their own cooperative stages—the final-return callback does not serialize
their math.

Dynamic audio-fragment quorum is a different primitive. Missing media leaves a
durable route record dormant before numerical admission; it never mounts half a
team or assigns a thread to an absent fragment. The final fragment makes the
route runnable. Media ordering, model position, route identity, and lane
identity remain separate as specified in design 14.

Block-mode kcoro therefore uses bounded per-domain ready rings plus lane-affine
mailboxes, not the general runtime's mutex-protected global queue. There is no
work stealing inside a mounted generation; atomic tile claim remains the
in-program load balancer.

## 4. Compact routing, not a graph machine

Model-open selects a trusted compiled template and publishes an immutable table
of coarse programs. A route label indexes that closed table; every outcome maps
to `NEXT`, `TERMINAL`, or `FAULT`. Runtime token values first pass through a
vocabulary-validated token-class map and never become opcodes or function
pointers. Tokens remain data; arriving at runtime does not confer executable
authority. Invalid labels, modalities, tokens, and outcomes fault before
indexing.

The table validates direct byte views, aliases, scratch bounds, access sets, and
all route targets at construction. It does not compile an arbitrary DAG,
interpret checkpoint-supplied bytecode, allocate callbacks, or record lazy
expressions. The hot path is bounds check → coarse program → exact CQ → total
outcome lookup.

The completion record acknowledges one exact ticket and reports execution,
state commit, publication eligibility, cause, and terminal status separately.
It is not TCP retransmission: stateful numerical work is never blindly replayed
after dispatch. A state-authoritative accepted pass may commit after its
publication epoch becomes stale, but stale work cannot publish or take another
route edge. That accepted pass may finish its authoritative commit; it has lost
the microphone.

## 5. Idle-capacity dormancy: correctness first

An expected-value address park is permitted only inside a resident worker's
idle loop after the runtime has proved that its complete ready predicate is
empty. It is not an operation, route, ticket, fence, playback, or capacity
primitive. Operation suspension is a durable record with no attached thread.
An AArch64 `LDXR`/`WFE` or x86 `UMONITOR`/`UMWAIT` idle backend is optional and
ships only if all of the following hold:

- architectural detection and a guarded startup probe both succeed;
- the loop arms the monitor, becomes dormant while the word equals the captured
  value, tolerates spurious events, and rechecks before return;
- monitor retirement, stop, wrap, and lost-wake tests pass;
- idle-resume p50/p95/p99, idle CPU, and power beat or match the OS path;
- the OS backend remains available. Rosetta always takes the fallback.

This is an alternative idle backend, never a bounded pre-dormancy spin tier. An
idle word's block ownership prevents accidental cross-domain use, but allocation
from a block arena does not guarantee physical L2 placement. A wake is not
progress; an explicit callback edge changed a ready predicate, and the runtime
rechecks that predicate before dispatch.

Candidate instructions remain capability- and measurement-selected:

| Need | Candidate | Contract |
|---|---|---|
| tile claim | relaxed `LDADD` / `LOCK XADD` | counter only; stage publication supplies ordering |
| slot lease | narrow `CAS` / `CMPXCHG` | use 128-bit CAS only if a future state truly requires it |
| BF16 math | `BFDOT`/`BFMMLA`, `VDPBF16PS` | exact direct-view fallback remains available |
| prefetch/cache hint | `PRFM`, `LDNP`, `PREFETCHNTA` | hint only; bind from a recorded profile |
| scratch zero | `DC ZVA`, vector stores | query block size; do not assume 128 bytes per instruction |
| ordering | the weakest proven acquire/release/barrier | `DMB ISH` is not defined as “one L2 cluster” |

Opaque Accelerate calls begin behind one global permit. Two simultaneous calls
are enabled only if profiling shows a makespan win without harming the latency
path. There is no “AMX lease per block” claim until a documented, directly
schedulable backend exists.

## 6. Tile-stationary conversation reuse

For bandwidth-bound decode the governing approximation is:

```text
tokens/s ≈ achieved_bandwidth / model_bytes_streamed_per_token
```

L2 is hardware-managed cache, not addressable threadgroup memory, and weights
cannot be pinned there. Whole-layer stationarity is also geometrically false for
LFM2: at `hidden=2048`, `ffn=8192`, the three BF16 FFN matrices alone occupy
`3 × 2048 × 8192 × 2 = 96 MiB`, before mixer and projection weights.

V2 therefore uses **tile/stripe-stationary opportunistic batching**:

- snapshot up to the precomputed scratch capacity of already-ready conversation
  rows; never delay the first interactive row to fill a batch;
- stream one direct-view weight tile and apply it across those rows before moving
  to the next tile;
- keep per-conversation activations and mutable state disjoint;
- use ganged eight-lane batch-one execution for the latency path and split
  four-lane work only when measurement supports it;
- report L2 misses, DRAM bytes/token, achieved bandwidth, batch size, p50/p95,
  and power. Do not report “resident” or “pinned” bytes as an allocation fact.

With one conversation this may provide no reuse benefit. With several, the win
is fewer weight bytes fetched for the group, not a promise of `C×` throughput;
both blocks still share the memory system.

## 7. Sequence mixers and the recovered Monarch work

V2 defines an internal `SequenceMixerDesc` seam with three distinct state laws:

- **LFM2 ShortConv:** chronological causal FIR, `K=3`, halo/carry two.
- **Attention:** absolute-position KV state; prefill positions may be partitioned
  where the declared causal dependencies permit it, while single-flow decode
  remains serial from token `j` to token `j+1`.
- **Future Monarch/Hyena long convolution:** explicit factor geometry,
  padding/truncation, filter/twiddle views, gating, overlap state, and scratch
  class. It is supported only by a model trained with that operator.

The Conformer depthwise convolution is not LFM2 ShortConv: its `K=9` kernel has
halo eight. Tests and descriptors name the operator so those halos cannot be
silently exchanged.

The recovered `flash-fft-conv-mlx` material establishes useful implementation
shape, but not the claims previously attached to it. Recovered code gets credit
for what it proves, no more:

- “banded” tests mean rectangular Monarch factors (`L != N`), not independent
  frequency-band partitioning;
- the recovered Metal kernel fuses the FFT stages **or** the IFFT stages, not
  FFT → filter multiply → IFFT; the multiply remains a separate materialized
  operation in that port;
- its fusion predicate `2 × N × L × sizeof(T) <= scratch_capacity` applies to an
  explicitly allocated scratch plane. A 16 MiB cache is not 16 MiB of allocatable
  threadgroup storage and cannot be substituted into that inequality;
- the recovered path was correctness-oriented and does not establish BF16
  parity, end-to-end fusion, or a production speedup.

The transferable pieces are the exact Cooley–Tukey/Monarch row-transform →
twiddle/transpose → column-transform decomposition, storage-narrow/F32-accumulate
policy, barrier placement, and offline reference fixtures. LFM2 keeps its trained
ShortConv and attention math. A future Hyena-family tranche may implement the
complete projected/gated long-convolution chain as one Flashkern program and
must compare full outputs and state against a real checkpoint oracle.

In-context learning, compressed memory, retrieval, or learned adapters that use
long convolution to stretch LFM2 context are separate research programs. They do
not enter this faithful-inference cutover or masquerade as a kernel optimization.

## 8. Build order

Each step is independently gated and leaves a correct fallback geometry:

1. **V2.0 — safety subset landed.** 128-byte Apple hot-word/SQ-CQ isolation,
   closed current request/layer/modality selectors, invalid-lane rejection,
   four-worker/eight-logical-lane parity, and zero-spin are green.
2. **V2.1 — bounded routed V1 audio, landed.** A total three-node/four-outcome
   table retains one exact slot across `TOKEN_PASS -> DEPTH_FRAME -> MIMI_DECODE`,
   commits token context, writes equal-rate Mimi PCM directly into pre-admitted
   playback or native-resamples codec scratch into a device-rate reservation,
   and releases compute before publication. Each ticket carries an immutable
   fixed-slot locator; the deleted generic descriptor registry and borrowed
   submission mode cannot reintroduce a callback mutex. The fixed route pool
   and callback-driven fair broker replace the historical exclusive-producer
   and synchronous-collection seam; each node releases its compute slot before
   it re-enters the ready set. Session-facing asynchronous terminal
   collection and total model-owned token classification are now mounted; block
   concurrency is the next scheduler boundary.
3. **V2.2 — extract block completion state, open.** Create two real
   `BlockDomain`s with private stage boards, scratch mounts, SPSC CQs, return
   accounting, and exact doorbells. Until all of those owners exist, the engine
   remains one team and one program regardless of any logical lane fold.
4. **V2.3 — block-mode kcoro, open after thread ownership.** Fixed-team thread
   lifecycle and final-return generation completion already belong to kcoro,
   and the bridge/route loops are retained kcoro services. V2.3 must extract two
   domain-local teams and ready rings, then prove exact-return and idle-event
   behavior independently for each domain.
5. **V2.4 — two independent programs.** Admit two `BLOCK4` programs only for
   different conversations. Profile actual overlap and shared-bandwidth effects;
   retain gang mode when it wins latency or parity.
6. **V2.5 — cooperative math and tile reuse.** Make Mimi and remaining lane-zero
   programs fixed-team generations, then add no-delay tile-stationary
   conversation snapshots.
7. **V2.6 — mixer seam.** Bind LFM2 ShortConv and attention descriptors and add
   explicit unsupported `MonarchLongConv` validation. A full Hyena port is a
   later model tranche, not part of V2 enablement.

## 9. Acceptance and invariants

- Two conversations may complete in reverse order through separate CQs without
  ticket confusion, lost ACKs, or double retirement.
- A ganged program excludes both blocks; same-conversation mutating programs do
  not overlap in the initial release.
- Every member returns exactly once for every dispatched generation; the final
  return callback runs once; a duplicate or missing return is a test failure.
- Invalid runtime selectors terminal-fault before table indexing and release all
  retained resources exactly once.
- Greedy-token equality is evidence, not acceptance. Full hidden states,
  logits, sampled codes, KV/ShortConv state, and PCM match the V1/oracle path.
- No allocation occurs after readiness, no weight is materialized or repacked,
  and blocked instances retain no compute slot.
- Interrupt at every program boundary preserves the declared state commit while
  preventing stale publication and recurrence.
- Fairness is measured at fused-program boundaries. The longest admitted
  program, not the nominal quantum, bounds when a third conversation can next be
  admitted.
- Idle CPU stays below the existing zero-spin gate on aarch64 and
  x86_64/Rosetta.

If block placement, cache reuse, idle-event backends, or opaque matrix overlap
provide no benefit, V2 remains correct as one ganged block with the OS idle
backend. These
are optimizations behind measurements, never correctness assumptions.
