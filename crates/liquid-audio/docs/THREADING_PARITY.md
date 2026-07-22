# Threading and execution ownership

Status: current production architecture. The torch/Candle comparison at the end
is explicitly historical.

## Production rule

Production no longer tries to reproduce Candle's Rust/Rayon execution model.
`liquid-audio` has no Candle dependency and does not own a Rust model, tensor,
sampler, KV cache, Conformer, Depthformer, Mimi codec, or inference loop. The
former workspace-only Rust/Candle model and training implementation is retained
only in Git history, not as a callable fallback.

The live ownership split is:

| Domain | Owner | Execution contract |
| --- | --- | --- |
| capture/playback device callbacks | native CoreAudio AUHAL units | direct native lease write or drain, publish edge, return |
| host voice state/UI delivery | one Rust `kcoro_sys::Service` | opaque handles, settings/control, bounded outward events only |
| native session coordination/delivery | native `kc_service` continuations | callback-driven, exact tickets, migratable on the bounded pool |
| native route/bridge/supervision | saved bridge frame plus retained route/supervisor services | same bounded pool as the numerical lanes |
| numerical work | one engine `kc_team` | fixed logical members on kcoro workers; one active generation |
| model math | architecture leaves and explicit Accelerate/AMX seams | pointer/stride views; no Rust numerical body |

## Rust host

`VoiceRuntime` mounts `SessionTask` as a retained kcoro service with one worker.
The service owns outward event delivery and opaque native handles; it does not
own a platform callback or numerical state. Its controls are realtime notifier
edges, not timeout loops or Crossbeam channels. Native device faults arrive as
bounded session events.

The AUHAL input callback reserves one generation-checked native span and asks
CoreAudio to render the complete device block directly into it before publishing
one chunk record. The AUHAL output callback claims, resolves, drains, and releases
exact native playback leases. The Rust host never sees PCM and no layer writes
it to a file or event channel to move it between stages.

The callback cadence reported by CoreAudio is device geometry, not a scheduler
timer. An input callback advances the sample clock even for acoustic silence.
Device failure is an explicit native fault edge. Wall-clock observations are
response telemetry/watchdogs only and do not decide speech boundaries.

## kcoro runtime workers

`kc_runtime` owns a fixed worker set and a bounded atomic ready board. A
stackless continuation retains its program counter, locals frame, and exact
ticket independently of physical workers. Producer notification is an atomic
publication edge; any free eligible worker may claim the frame. There is no
work-stealing deque, generic future executor, or operation-owned thread. An
explicit eligibility mask exists only for a genuinely thread-affine host
resource.

A service callback drains durable state and returns one of three outcomes:

- dormant because its complete predicate is empty;
- locally runnable again because bounded work remains;
- terminal after stop/fault and admitted edges have settled.

Only an otherwise-idle resident runtime worker may enter expected-value
dormancy. A model operation creates no waiter and preserves no blocked native
stack; its fixed stackless frame is the literal continuation.

## Flashkern fixed team

The engine creates one stable `kc_team` with `kernel_lanes` logical members
(eight by default) on the same kcoro pool as its bridge and control services.
`kcoro_arena` owns resumption, stop, and generation entry/return stamps;
`kc_team` creates no pthread or private idle doorbell. Flashkern owns the
lane-uniform numerical program and ticketed pass state.

One generation is active at a time. All members observe the same generation,
claim disjoint tiles, run a complete assembly leaf without yielding, record
return, and become available. The final return invokes a minimal callback
exactly once; that callback publishes the completed generation and resumes the
exact suspended bridge ticket. A free kcoro worker continues at its saved
program counter and may publish the next eager stage without a host round trip.

The current implementation is one team, not two concurrently executing
four-lane `BlockDomain`s. Logical block counters do not change that fact. A V2
grid must not be counted as landed until independent teams, scratch domains,
completion paths, and cross-domain ownership tests exist.

## Numerical parallelism

- BF16 checkpoint bytes stay in the immutable native image and are unlifted in
  registers by the selected architecture leaf.
- NEON/x86 SIMD assembly is the primary implementation for elementwise,
  reduction, convolution, sampler, frontend, Sesame, and model-stage work.
- Large Apple matrix operations may cross the documented Accelerate/AMX seam.
  This is a numerical backend call, not a second scheduler or an alignment/dtype
  staging license.
- A register-resident chain may keep intermediate values inside one assembly
  leaf. A team-generation return is the materialization boundary: only the
  compact state needed by the next retained stage is written to its sealed
  native arena.
- Remaining value-producing C++ loops are migration debt and cannot serve as a
  production fallback once their assembly replacement lands.

## Speech timing and deadlines

The native Sesame microphone policy consumes one evidence update per 20 ms of
incoming samples. Minimum utterance, prepare, endpoint, and forced endpoint are
sample-count state. The current prepare edge records policy readiness; it does
not yet run candidate-owned speculative model scratch. Pause gates own one-shot
deadline children; an expiry publishes a typed edge for the exact pause
generation and never runs inference inline.

Flashkern team generations are guarded by a hard correlated monotonic deadline.
Member entry/return stamps identify a missing lane without adding an arrival
spin. Normal completion retires the deadline; expiry and completion race through
one terminal decision.

Current limitations:

- the native detector's playback state exists, but production feeds only the
  microphone stream;
- the Flashkern one-second hard deadline is provisional and not yet calibrated
  by stage/shape;
- the fatal capsule is captured before abort but is not yet routed to a durable,
  observable platform crash sink.

## Teardown ownership

Normal stop first closes native progress admission. Platform streams are then
disconnected so no callback can publish through a retired notifier. Retained
services settle terminal records, the fixed team stops after its admitted
generation, deadline handlers acknowledge cancellation, and administrative
join proves ownership has drained before storage is freed.

Join is lifecycle observation, never the means by which model progress occurs.

## Historical torch/Candle reference

During the deleted Candle migration, the project compared torch's P-core
intra-op sizing with Candle/Rayon and explored an Accelerate feature plus a BF16
CPU GEMM shim. Those measurements explained early performance differences, but
they are not the production threading architecture. Git history preserves that
deleted implementation and its former intra-op thread policy.

Similarly, the former `RealtimePipeline` worker, `crossbeam-channel` event path,
Rust RMS VAD, and Rust model-owned cancellation loop are historical. Current
source of truth is:

- `src/runtime/voice_runtime.rs` for the platform callback/service rim;
- `src/native_voice.rs` for opaque runtime/session and PCM endpoint bindings;
- `native/src/runtime/voice_session.cpp` for sessions, PCM docks, Sesame policy,
  and delivery;
- `native/src/engine/flashkern_engine.cpp` for route/pass continuations and team
  supervision;
- `crates/kcoro-sys/vendor/kcoro_arena` for runtime, service, scope, deadline,
  team, and doorbell ownership.

## Gates

```bash
cargo test -p kcoro-sys
cargo test -p liquid-audio --lib
cargo test -p liquid-audio --tests
git diff --check
```

The ignored real-checkpoint truth gate must be executed explicitly with the
model image. It runs two native LFM2 conversations entirely in memory: typed
input produces audio-token/Mimi playback for Agent A, and hardware-sized native
PCM reservations drive Agent B through Sesame and the full speech path. The
fixed-seed exchange is repeated and must match. Release additionally requires
the product linkage audit, zero post-readiness allocation/model reads,
calibrated supervision, observable fatal evidence, real-device rate/geometry
coverage, and AArch64 plus x86_64/Rosetta coverage.
