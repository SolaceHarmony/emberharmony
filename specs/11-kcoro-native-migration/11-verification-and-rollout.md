# Verification and Rollout

Status: normative gate plan.

Baselines: EmberHarmony `321538f11749`; `kcoro_arena` `447d04f0246b`.

## Goal

Prove the migration one ownership boundary at a time. Correctness means more
than a close final waveform: it includes exact shapes, state progression,
conversation semantics, full-pass interruption, bounded memory, precise wakes,
no post-load disk traffic, no hidden fallback, and the actual Tauri microphone
and speaker path. It also means the local numerical stack has the fixed shape
Rust control -> C ABI -> C++ coordinator -> architecture kernel table, with no
Rust arithmetic or payload-bearing math call.

No phase becomes the production default because it compiles or wins a microbench.
It advances only when its numerical, lifecycle, memory, latency, dependency, and
product gates all pass.

## Existing Evidence to Preserve

The migration starts with useful tests, not a blank slate:

| Existing evidence | Location | Required disposition |
|---|---|---|
| native safetensors image tests | `crates/liquid-audio/tests/native_safetensors.rs` | Convert to direct native-loader and malformed-input tests. |
| native attention/conv/MLP parity | `src/compute/flashkern/native_engine.rs:592-1050` | Preserve as boundary fixtures independent of Candle calls. |
| lane/fanout numerical tests | `src/compute/flashkern/fanout.rs:1251-1888` and architecture kernel modules | Move stable fixture vectors into native tests. |
| engine idle/zero-spin test | `tests/engine_idle_zero_spin.rs` | Extend to coordination signal-one and fixed-executor blocking wait/syscall assertions. |
| speculative prefill and cache tests | `tests/speculative_prefill.rs`, `tests/cache_equivalence.rs` | Re-express against native conversation marks and suffix state. |
| end-to-end generation | `tests/e2e_generate.rs` | Pin full token/text/audio-code traces and state hashes. |
| native Mimi parity | `tests/mimi_native_parity.rs` | Retain full-length comparison, KV wrap, and direct-output variant. |
| end-to-end voice runtime | `tests/e2e_voice_runtime.rs` | Replace env selection and Rust runtime assumptions with explicit ABI config. |
| frame interrupt/no-reset tests | `src/runtime/realtime.rs:2852-3019` | Port behavior to native frame epochs before deleting Rust worker. |
| shutdown/queued-prepare tests | `runtime/realtime.rs:3377-3599` | Port to native terminal arbitration and stop priority. |
| event backpressure/interrupt tests | `runtime/realtime.rs:3711-3799` | Preserve bounded failure and prompt full-pass interruption. |

The current workflow at `.github/workflows/rust-voice.yml:62-94` builds all
targets, executes the library and hermetic integration suites on Linux and
macOS, and repeats both suites with Metal on macOS. Checkpoint-dependent and
audible-output e2e tests are explicit `--ignored` host gates; they do not count
as executed merely because CI compiled their binaries. Every required gate must
retain an explicit execution step as the native implementation replaces it.

## Evidence Manifest

Every fixture and benchmark result carries:

```text
format_version
ember_commit
baseline_commit
model_fingerprint
engine_kind
backend
architecture and ISA
dtype and accumulation mode
input shape and logical lengths
sampling settings and seed
expected output shape
expected bytes/hash or tolerance policy
```

Do not commit undocumented raw dumps. The fixture generator records the exact
baseline function and line-range. Full output is compared; tests do not truncate
native/fixture arrays to a common length. Deleted baseline code is run only from
a pinned worktree during fixture capture, never retained as a test crate.

## Parity Ladder

Numerical cutover proceeds from smallest stable boundary to product output:

1. Loader byte spans, dtype, shape, and tensor names.
2. Scalar/vector primitives: BF16 conversion, GEMV/GEMM, norm, activation,
   convolution, softmax, resampling, DFT, filterbank.
3. One stage: mel, subsampling, attention, Conformer convolution, backbone
   attention/short-conv, depth block, Mimi units.
4. One layer with complete state and residual ladder.
5. One segment/frame/token pass.
6. Full/suffix prefill and cache state.
7. Deterministic generation trace: text IDs, audio IDs, modality schedule, RNG.
8. Codec PCM and playback publication.
9. Multi-turn/multi-frame conversation continuation.
10. Actual app microphone-to-speaker behavior.

At each boundary assert shape, dtype, logical length, finite/nonfinite policy,
state cursor, and full values before advancing. An end-to-end match cannot excuse
a wrong intermediate state that happens not to affect one fixture.

## Numerical Policies

Classify each output before implementation:

| Class | Policy |
|---|---|
| IDs, counts, offsets, modalities, epochs, cursor state | exact |
| resident weight bytes and integer embeddings | exact |
| BF16 stored activation where native/reference ladders are identical | bit exact |
| F32 reductions/normalization/softmax | documented absolute/relative/ULP tolerance plus nonfinite equality |
| stochastic sample | exact selected IDs and exact post-draw PRNG state |
| PCM | full-length comparison, max/RMS error, continuity, and stable waveform hash when bit exact |

No tolerance is chosen after seeing a failure. The design owner records it from
the current cross-library floor and model sensitivity. A wider tolerance needs
an explicit review and fixture-version bump.

## Model Loader Gates

- Open every supported single- and multi-shard layout.
- Reject overlap, overflow, truncation, malformed JSON, duplicate names,
  incompatible dtype, wrong shape, and missing components.
- Prove all returned views lie within one retained aligned image.
- Record resident image bytes, backend-resident bytes, and compatibility-copy
  bytes separately; production requires compatibility copies equal zero.
- Hash source files before and after load and hash resident spans.
- Trace file I/O after model readiness; inference performs zero open/read/mmap
  operations and causes no avoidable major page faults after warmup.
- Load two conversations from one model and prove every weight address is shared.
- Destroy model only after all retained sessions/conversations are released.

Current copy debt is measured at
`crates/liquid-audio/native/src/io/README.md:48-66`; the gate must make that
number zero by removing the compatibility bridge, not by hiding its counter.

## Native Kernel Boundary Gates

Before any native stage replaces its current owner:

- the release call graph enters through an approved `lfm_*` control operation,
  remains in the C++ coordinator, and dispatches through the model's immutable
  architecture kernel table;
- `lfm_voice.h` exposes no weight, activation, PCM, mel, KV, logits, codebook,
  sampler-state, or generic tensor operation to Rust;
- production Rust contains no local DSP, tensor transform, model arithmetic,
  token sampler, codec arithmetic, kernel callback, or per-token/per-frame FFI
  loop;
- scalar C++ oracles compile into `lfm_voice_oracles` only and are absent from
  the production archive, release link map, and runtime dispatch table;
- required ISA absence returns `LFM_UNSUPPORTED_BACKEND` before model readiness;
  it never enters a scalar C++ or Rust fallback;
- every intrinsic kernel has stored disassembly and benchmark evidence for its
  target compiler flags; hand-scheduled `.S` kernels have ABI/unwind tests;
- Apple native-library stages are named adapters with pointer/shape contracts,
  not tensor objects, and pass the same fixture and allocation/copy gates.

The test harness may invoke individual kernel symbols. Rust production code may
not: internal kernel symbols are hidden and only the C++ plan owns their table.

## Allocation and Copy Gates

Install test-only allocation and payload-copy instrumentation in native code.
Tag every allowed startup allocation and every permitted copy site.

The product budget is:

- model/session/conversation creation: bounded and reported;
- hardware callback to capture ring: one required PCM copy;
- host-bound text/error metadata: one bounded copy;
- backend upload at model open: allowed, persistent, and reported;
- capture-to-playback numerical path after warmup: zero process allocation and
  zero handoff payload copy.

Tests cover ring wrap, page-pool growth, maximum utterance/context, full playback,
speculative rollback, multi-conversation switch, and codec drain. A copy counter
alone is not enough: poison source buffers after handoff and verify consumers
read the retained destination/pointer contract rather than stale staging memory.

## Scheduler and Wake Gates

Before mounting the product lane team:

- split worker and lifecycle condition variables as required by document 03;
  replace the broadcasts in `queue_locked`, `finish_cont`, and `suspend_cont`,
  not only the enqueue site;
- run the `kcoro_arena` 100,000-iteration terminal race gate;
- prove one continuation never executes concurrently on two workers;
- prove wake-before-suspend, wake-during-step, and wake-after-suspend each resume
  exactly once;
- prove finish/suspend transitions wake no coordination work waiter while
  run-until-idle, join, and stop waiters still observe every predicate change;
- prove request-stop cancels every legal parked continuation through a retained
  operation/doorbell; forced continuation destruction is not accepted as the
  wake source for a raw predicate wait;
- prove each fixed compute lane retains stable logical worker identity;
- prove each barrier generation runs one serial transition;
- prove the shared dispatch/fence expected-value waits close early-wake,
  declaration, wait-entry, and last-arriver races across changing generations;
- inject a typed fault at every lane/stage/tile boundary; all active lanes drain
  the collective once, no later stage runs, and rollback/poison publishes
  exactly once without deadlock;
- prove idle and barrier waits block immediately with no spin tier or periodic
  wake;
- prove every accepted full pass owns one child ticket and one descriptor lease,
  delivers one terminal event to its configured target, and causes one bounded
  orchestration handler invocation;
- prove every SQ publication already owns CQ capacity; full, wrap,
  stale-generation, cancel, and stop races overwrite neither command nor
  completion pointer;
- prove actor mailbox floods cannot starve ticket completion, timers, or stop.

Measure per pass:

```text
continuation enqueues
requested worker wakes
actual condition-variable wakes
workers finding no runnable continuation
fixed-lane wait registrations and host wake calls
logical fence park-mask population
declared stages, true barriers, and waits entering the host primitive
coordination continuation migrations
ticket accepts, dispatches, terminals, callbacks, and losing causes
syscalls
```

One coordination enqueue should not wake every sleeping worker. One nonempty
fence park mask should cause one shared address wake for the fixed team, and the
logical waiter count must equal the mask population. Any coordination wake herd
or per-peer fence syscall fan-out is a release blocker even when average kernel
latency is unchanged. Source and disassembly audits must find no `PAUSE`,
`YIELD`, repeated generation-load loop, WFE/UMWAIT budget, or timed poll in wait
paths.

Run ASan+UBSan and TSan in separate builds. TSan on macOS is useful and remains
part of local/CI coverage; Linux TSan provides a second scheduler/libc surface.
Both 100,000-iteration operation and ticket terminal-race gates run inside each
sanitizer job. A lower default iteration count is not evidence for this gate; if
a runner cannot complete it, move the job to a supported runner rather than
self-skipping green.

## Interrupt and Lifecycle Matrix

Inject stop/interrupt at every declared boundary:

| State | Required outcome |
|---|---|
| model load | cancel at supported load boundary or complete load then stop; never publish partial model |
| idle capture | exact wake and one terminal event |
| active audio callback | callback returns; coordinator stops outside callback |
| VAD candidate | rollback or commit once according to winning epoch |
| mel/Conformer/prefill pass | finish full pass, then discard or commit by mark |
| generation token pass | finish/sample/append atomic token, then stop recurrence |
| Moshi frame pass | advance continuous model state; discard stale output epoch |
| playback full | wake on flush/space/stop; no timed drain loop |
| queued prepare plus stop | stop wins; prepare never executes |
| reliable callback/event failure | native terminal host-sink failure, join, no callback after destroy |
| telemetry callback failure | observer drops/detaches only; native ticket and session continue |

Run every pairwise race among match/commit, resume, interrupt, stop, callback
failure, playback flush, and destroy. Terminal state, wake count, page leases,
and live-object counters must all equal their expected final values.

Ticket races assert execution, state disposition, publication, and terminal
cause separately. Interrupt during an active continuous-state pass may produce
`completed + committed state + stale publication`; speculative work may produce
`completed + rolled-back state + stale publication`. Neither may report a
partial kernel cancellation or publish another old-epoch child ticket. An
in-place fatal fault that cannot restore its boundary mark must poison the
context and prove that recurrence and snapshot capture reject it.

## Audio Gates

Use the real ring and adapter implementation, not a duplicate queue model:

- mono/stereo/device-format conversion and one callback copy;
- wrapped two-range capture span;
- capture overrun and playback underrun;
- output block reservation, publish, consume, flush, and generation reuse;
- reference-audio active/tail behavior;
- VAD endpoint, false pause, barge-in, self-echo, and overlapping speech traces;
- LFM2 committed-turn and speculative-pause paths;
- Moshi continuous frame clock, pressure drop, mic pause, and no-reset interrupt;
- stateful input/output resampler continuity.

The hardware callback test fails if it observes allocation, mutex blocking,
kcoro entry, file I/O, host callback, or model computation.

## Conversation Gates

- Append text, input audio, generated text, and audio codes without changing
  prefix bytes or addresses.
- Full prefill and suffix prefill produce identical state from equivalent marks.
- Invalid marks fail before mutation.
- Candidate rollback restores every logical length, cache tail, page lease, and
  sampler state.
- Partial assistant output remains in model context when playback is flushed.
- Two hot conversations alternate every token/pass for a long run and remain
  isolated while sharing all model weight addresses.
- Broker tests mix deadline, interactive, and background tickets; enforce
  consecutive-pass/time quanta, age promotion, one active command slot, and no
  starvation or command overwrite.
- Long-prefill tests stop between state-valid child blocks, never inside an
  operator; measured longest-pass time bounds ordinary interrupt/switch latency
  and deadline admission records every defer/miss.
- Context overflow follows explicit policy and never reallocates inside a pass.
- Quiesce reports no active pass and emits a relocation-clean region inventory.
- Restore from an in-memory image continues with the exact next deterministic
  token. Disk/delta/WAL tests belong to spec 10 after this memory gate passes.

The handoff to spec 10 must carry these non-negotiable durable gates:

- checkpoint `accepted` cannot publish while a child pass ticket is dispatched;
- checkpoint `durable` is a separate child ticket after immutable object sync, inactive
  A/B manifest sync/publication, and required WAL association sync;
- pending periodic deltas are cumulative or merged and staging is bounded to two
  configured slots;
- chain length, delta bytes, retained bases, and total disk bytes remain bounded
  through compaction;
- the previous valid manifest survives every injected crash boundary;
- a real host adapter is tested on disk, including `F_FULLFSYNC` or proven
  equivalent for a macOS power-loss durability claim;
- the current append-only `kc_wal_snapshot_write` store is not used for
  long-running conversation images;
- forced writes, compaction, and sync during active speech cause no pass p99 or
  playback-underrun regression outside the approved budget.

## Performance Method

Record cold and warm behavior separately. Cold includes file read, validation,
backend residency, page faults, scheduler bootstrap, codec initialization, and
warmup. Warm excludes all of those and starts only after a declared ready edge.

Required warm distributions include p50, p95, p99, and maximum for:

- capture callback and callback-to-coordinator wake;
- endpoint-to-mel, mel, Conformer, adapter, and suffix prefill;
- each backbone token and Depthformer frame;
- Moshi encode, LM, Depformer, decode, and total frame pass;
- codec-to-playback publish and playback callback;
- interrupt-to-last-old-epoch-audio;
- stop-to-joined;
- hot conversation switch.

Also record lane utilization, CPU residency, context switches, scheduler wakes,
cache misses where tooling permits, allocations, bytes copied, queue/ring depth,
underruns, page faults, and disk bytes. Correlate all measurements with
runtime/action-ticket/pass-ticket/conversation/epoch IDs.

Do not compare a debug/sanitized native build to an optimized baseline. Pin
power mode, hardware, sample, model, lane count, compiler flags, warmup, and
background load. Keep raw run artifacts with the summary.

### Stutter budget

Before implementation, measure the current app and record a release budget for:

- missed Moshi frame deadlines;
- token-pass outliers;
- playback underruns;
- wake-to-run outliers;
- user-perceived endpoint-to-first-audio latency.

An average improvement does not pass if p99/max or underruns regress beyond the
approved budget. The scheduler wake herd is evaluated primarily as tail latency.

## Product App Gate

Each user-visible phase ends in the real Tauri application, not an example:

1. persisted settings select exact engine/backend/device;
2. model readiness and errors reach the settings UI;
3. microphone permission and selected device work;
4. LFM2 hears, endpoints, speculatively prepares, speaks, and handles barge-in;
5. Moshi runs continuous full duplex and soft interrupt without reset;
6. playback/AEC/reference behavior does not self-interrupt;
7. typed input atomically pauses mic and interrupts;
8. device change, app sleep/wake, session restart, and app quit join cleanly;
9. a long soak shows bounded RAM, queues, descriptors, pages, and worker count;
10. logs/status accurately name the native owner and backend;
11. the optional kernel observer may close or overflow without stopping voice;
12. the five-bar native visualizer uses real capture activity while listening,
    fixed-lane activity while thinking, playback activity while speaking, and
    no synthetic work while idle.

Capture one test trace with source timestamps from hardware callback through
every native pass and playback consumption. A UI “connected” state alone is not
evidence that audio flowed.

## Static Release Gates

Run from a clean release configuration:

```bash
cargo tree --manifest-path packages/desktop/src-tauri/Cargo.toml --target all
rg -n "candle|moshi|candle-flashfftconv|rayon|cpal" \
  crates/liquid-audio/Cargo.toml packages/desktop/src-tauri/Cargo.toml
rg -n "std::env::(var|var_os)|env::(var|var_os)" \
  crates/liquid-audio packages/desktop/src-tauri/src
rg -n "std::thread::spawn|ThreadManager::.*spawn" \
  packages/desktop/src-tauri/src/voice/native crates/liquid-audio/src
rg -n "Tensor|Vec<f32>|&\[f32\]|&mut \[f32\]|sample|logits|codebook" \
  crates/liquid-audio/src packages/desktop/src-tauri/src/voice/native
```

The searches are allowlist-based CI checks, not blind zero-match commands: path
derivation may use host environment APIs outside native voice configuration,
and UI metadata may contain words such as `sample_rate`. Each allowlisted match
must name the control-only reason it remains.

Add symbol audits for:

- approved `lfm_*` C exports only;
- no unresolved OS calls in the portable kcoro core archive;
- no Candle/Rust Moshi symbols in the product binary;
- every `lfm_voice.h` declaration linked exactly once;
- architecture kernel symbols present on their target but hidden from the
  public ABI;
- no scalar-oracle object or Rust numerical symbol in the release link map;
- no production C ABI operation accepting a numerical payload span.

Replace `crates/liquid-audio/scripts/gate.sh:47-50` environment-based device
selection with explicit test config files or CLI arguments parsed by the test
harness and passed through the same ABI config structs as Tauri.

## CI Matrix

Minimum mandatory jobs:

| Job | Platform | Coverage |
|---|---|---|
| native-debug | Linux x86_64 | C++ unit, test-oracle/SIMD parity, ABI, loader malformed cases |
| native-release | Linux x86_64 | AVX runtime path and performance smoke |
| native-arm | macOS aarch64 | NEON/BF16/Accelerate path, audio adapter compile/integration |
| native-x86-mac | macOS x86_64 or Rosetta smoke plus native x86 CI | Apple x86 linkage and runtime dispatch |
| ASan+UBSan | Linux and macOS where supported | memory/undefined behavior |
| TSan | Linux and macOS | scheduler, rings, context races |
| terminal-race | Linux release | 100,000 arbitration iterations |
| integration | Linux/macOS | explicit execution of every native integration binary |
| Tauri | macOS aarch64 plus Linux build | settings/ABI/command/event lifecycle |
| ticket-observer | macOS aarch64 plus Linux | exact callback, observer isolation, coalescing, TypeScript lossless IDs, visualizer signal truth |
| dependency/symbol | release artifact | no fallback framework, ABI surface, licenses |
| soak | scheduled hardware job | long full-duplex, context switch, bounded memory/wakes |

MLX/Metal adds its own macOS job only when that backend is implemented. A
compile-only Metal feature is not execution evidence.

## Migration Gates

| Gate | Scope | Exit evidence |
|---|---|---|
| G0 | baseline freeze | fixture manifest, current app latency/wake/allocation/copy report |
| G1 | ABI and settings | layout/link tests, explicit config mapping, capability errors |
| G2 | loader and model binding | one image, zero inference I/O, zero compatibility copies for bound components |
| G3 | coordination/ticket/fixed executor | terminal races, exact callbacks, signal-one coordination, one wake per nonempty fence mask, zero-spin barriers, pointer submission, parity and tail-latency report |
| G4 | native kernel/wait substrate | kernel parity, no-Rust-math ABI/link/call-stack audit, measured bandwidth and wait latency |
| G5 | native PCM/VAD | sole callback copy, endpoint traces, interrupt/playback lifecycle |
| G6 | mel/Conformer/adapter | every boundary parity, zero hot allocation/copy |
| G7 | conversation/prefill/generation | full/suffix state parity, native recurrence, deterministic sampling |
| G8 | LFM2 codec/product | direct playback, multiturn app gate, partial-thought semantics |
| G9 | native Moshi | full frame parity, continuous app gate, no-reset pressure/interrupt |
| G10 | seam inversion | control-only Rust host, clean production dependency/symbol/thread audits |
| G11 | snapshot readiness | quiesce, region inventory, two-context switch; handoff to spec 10 |

A gate may first run in the native test harness before becoming the sole product
owner. Do not add an in-process old/new implementation selector. The gate
captures independent fixtures first, then deletes the replaced owner; later
stages may not depend on calling removed Rust/Candle code.

## Rollout and Rollback

Use git history and separate worktrees, not a retained fallback:

1. Before cutover, a clean worktree pinned to the baseline commit may generate
   or verify the fixture manifest.
2. Normal CI builds only the current native tree against committed fixtures; it
   does not build a legacy artifact or backup crate.
3. A native-only canary build ships to the first release cohort with expanded
   telemetry and bounded trace capture.
4. Promotion follows measured crash, underrun, stutter, load, and teardown data.
5. Rollback ships the previously known-good application build. A failed native
   model does not instantiate Candle inside the same process.
6. Replaced Rust/Candle sources are deleted in the gate that supersedes them.

No rollout selector is an environment variable. User engine/backend choices
remain persisted settings; implementation A/B is development/release policy,
not an end-user hidden fallback.

## Definition of Complete

The migration is complete only when:

- G0 through G11 pass on required architectures;
- LFM2 and the product-selected Moshi path run entirely behind the native ABI;
- runtime weights, activations, state, PCM, and recurrence never enter Rust;
- Rust invokes control/config/lifecycle methods only; model loading and every
  numerical operation are owned by C++ and the selected native kernel table;
- the release link map contains no scalar test oracle or numerical Rust symbol;
- post-load inference performs no disk access;
- after the one hardware callback copy, payload handoffs are pointer/offset
  descriptors and kernels write final destinations;
- stop/interrupt decisions occur at full pass boundaries;
- every accepted action/pass is represented by a single-shot ticket and its
  orchestration callback remains wholly native;
- one model serves multiple isolated hot conversations;
- Rust/TypeScript public control behavior remains compatible;
- the production dependency and symbol graph contains no Candle inference path,
  and the replaced Rust model/runtime sources are absent;
- current architecture documentation describes the shipped owner truth;
- optional Tauri ticket observation and the visualizer cannot backpressure,
  cancel, or participate in native recurrence.

## Non-Goals

- Do not use one end-to-end hash as the only correctness proof.
- Do not accept mean latency as a substitute for tail/stutter measurement.
- Do not call a build of integration tests an executed test suite.
- Do not keep an unmeasured fallback in the release binary for comfort.
- Do not begin durable snapshot/WAL rollout before in-memory quiesce and exact
  continuation are proven.
