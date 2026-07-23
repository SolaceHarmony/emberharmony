# Rust inference deletion plan

Status: **LFM2 production numerical ownership cutover complete; fixed-team and
repository-structure follow-on ledger**, audited against the working tree on
2026-07-19.

## Ruling

Rust owns settings/control mapping, opaque native handles, and host projection.
Native code owns platform microphone/speaker callbacks, endpoint lifetimes, the
exact Sesame detector, and sample-clock endpointing policy. Rust owns no production
model math, DSP, tensor, weight, token, sampling, KV/codec state, model-pass
scheduling, recurrence, speech evidence, or turn boundary.

C++ owns native plans, immutable views, conversations, sessions, queues, leases,
stages, and recurrence. Production pass arithmetic belongs to typed
AArch64/x86_64 leaves; approved Apple matrix machinery is reached behind that
leaf boundary. Formula-derived immutable tables may be constructed at readiness
and are reported separately. Layout, alignment, dtype, transpose, framework
ownership, or convenience copies of weights are forbidden.

The default crate and desktop production graph are native-only. The former
Rust/Candle LFM2 model, training surface, fixture-capture code, and Moshi
dependencies were removed from the workspace after native ownership landed.
There is no oracle feature, callable numerical ABI, or Rust model fallback.

## As-built / open-gaps ledger

| Area | Working-tree state | Remaining work |
|---|---|---|
| Main + LFM2.5 detokenizer weights | **Landed.** One byte-exact allocation, direct parallel positioned reads, component-scoped catalog, source handles closed, image page-table read-only after validation. | Keep real-checkpoint digest/load benchmarks as release gates. Mimi is a distinct future-Moshi component and is not loaded here. |
| Typed binding | **Landed.** Exact BF16/F32 dtype, rank, shape, layer, codebook, and vocabulary checks; possibly unaligned tensors remain byte views. | None for LFM2. |
| Weight consumption | **Landed.** Frontend, Conformer, backbone, Depthformer, and the released audio detokenizer bind the same image; BF16 unlift occurs in registers and detokenizer F32 views remain direct. | `compatibility_copied_bytes == 0` remains an acceptance assertion. Its counters were **stubs returning a literal 0** (review 2026-07-16) — the gate could not fail; now wired to real per-plan tallies. See "Accounting is a tally, not a constant" below. |
| Native model chain | **Landed for numerical ownership.** One typed, model-correlated `REQ_AUDIO_ENCODE` pass owns resample → valid-only BF16 frontend → whole Conformer/adapter over borrowed spans and pre-reserved conversation buffers. Modality assembly, M≤4 checkpoint-BF16 prefill, backbone, sampling, Depthformer, released audio detokenizer, and tokenizer are also native-owned. | No remaining numerical-stage ownership gap for LFM2.5. Mimi remains isolated for future Moshi and has no LFM2.5 route. |
| Conversation/session | **Landed.** Native KV/ShortConv/detokenizer state, PRNG, cursor, recurrence, text/PCM tickets, reliable events, epochs, interrupt, stop, and join. Rust does not drive progress. `kc::PermitBroker` owns the fixed route pool, admission, fairness, generation leases, and one saved continuation frame per multi-hop route. Each pass completion resumes that exact route frame; the `kc::TeamExecutor` frame advances only team generations. Both use the same bounded pool as the supervisor and logical team, not private pthreads or a one-worker side runtime. Text and audio routes notify a coordinator-owned `SessionAction`, which collects the exact handle without a numerical wait. | Real V2.2 `BlockDomain` extraction is separate scheduler work. |
| Context rollover | **Landed.** Fixed capacity+runway BF16 state, monotonic cursor, absolute RoPE range generation, nonmutating whole-action admission, causal row-by-row eviction, and in-place compaction. | None for the activation-state sliding-window contract. |
| Shared model | **Landed.** Per-conversation state/scratch and a fair model-owned expected-value pass gate; engine `-EBUSY` does not leak as scheduling policy. | Capacity-2 continuations may improve overlap; fairness is already correct. |
| Production graph | **Landed.** Desktop creates `NativeVoiceModel` and opaque native conversations/sessions only; default dependencies do not enable Candle or Moshi. | Native Metal/MLX remains a separate future backend and must fail explicitly until mounted. |
| Physical audio dock | **Landed for LFM2 on macOS.** Native AUHAL callbacks own the device units and the sole capture/playback endpoints. CoreAudio renders capture directly into a page-mirrored, generation-checked circular-arena reservation; the callback publishes typed chunk/XRUN records consumed by native Sesame and turn policy. Native playback resolves a retained lease only in the device callback and preserves ticket/epoch/generation through release. Rust retains only an opaque platform-audio handle; there are no progress heartbeats, Crossbeam audio edges, Rust PCM endpoints/rings, utterance vectors, or Rust VAD buffers. | Keep real-device rate/geometry, fault, stale-epoch, full-ring/XRUN, and teardown races in the release gate. Other platforms fail explicitly until their native adapters land. |
| Moshi | **Not ported.** It is offline/oracle-only and is not the shipped default. | A full native Moshi port is a subsequent tranche; this LFM2 ledger does not claim it. |

## Completed LFM2 cutover

### One immutable model image

- `native/src/io/safetensors.cpp` opens and fingerprints all selected shards,
  computes checked 64-byte source bases, and allocates exactly one combined
  main+detokenizer image.
- Up to four workers perform retrying 8 MiB positioned reads directly into
  disjoint final spans. There are no chunk allocations, payload staging buffers,
  payload zero-fill, or application payload `memcpy` calls. Only inter-source
  alignment padding is zeroed.
- Every worker joins before failure unwinds the image. The loader deterministically
  selects failures, verifies the same open handles, closes them before publication,
  validates metadata/spans, and seals the allocation read-only.
- `LfmModelMemoryV2` reports source bytes, shared segment build/attach/wire
  ownership, tensor-payload reads, directly bound bytes,
  formula-derived immutable bytes, compatibility-copied bytes, load time, worker
  count, and task count. Directly bound bytes come from successful exact binders
  (deduplicated by resident span), not from summing every checkpoint entry; unused
  entries therefore remain source/resident bytes without masquerading as consumers.

### Direct native consumers

- `LfmModel` is the sole image owner. Exact byte-addressed views bind embeddings,
  every backbone layer, Conformer, Depthformer, and the released audio
  detokenizer. No public production ABI
  exposes names, shapes, weight pointers, mel rows, hidden rows, logits, KV, or
  codec codes.
- BF16 checkpoint storage is not widened, aligned, transposed, packed, or copied.
  Architecture kernels load unaligned little-endian words and unlift them in
  registers; scalar tails use safe byte loads.
- Formula-changing tables—RoPE, frontend/window/FFT, BatchNorm denominators, and
  detokenizer inverse-DFT/RoPE tables—are the only admitted derived storage and
  are accounted separately.
- Frontend power aliases dead STFT real storage, valid mel writes the BF16
  Conformer destination, and Conformer writes the native prefill plane. Every
  eligible M≤4 backbone prefill projection now keeps its f32 accumulators inside
  the architecture leaf and publishes one exact-RNE BF16 result directly into
  the strided consumer plane. The former `bcxf`, `qkvf`, and `projf` planes are
  deleted, saving `16 * (4h + qkv_max)` bytes per prefill workspace. The
  LFM2.5 detokenizer writes directly into a 24 kHz playback reservation when
  rates match; for the 48 kHz desktop dock it writes conversation-owned
  detokenizer scratch and the same
  retained route stream-rate-converts directly into the device-rate reservation.
  Neither path performs a transport copy or returns numerical work to Rust.

### Accounting is a tally, not a constant

Review finding (2026-07-16), fixed: `compatibility_copied_bytes` — the gate this
ledger and spec 15 both cite as the proof that no weight is materialized — was a
**compile-time constant 0**. Both contributors ignored their argument and
returned a literal:

- `lfm_conformer_materialized_weight_bytes(const LfmConformer *c) { (void)c; return 0; }`
- `mimi_decode_plan_compatibility_copied_bytes(const MimiDecodePlan *) { return 0; }`

The pre-detokenizer `lfm_model.cpp` summed exactly those two, so
`voice_session.cpp`'s
`if (memory.compatibility_copied_bytes != 0) reject` was dead code and
`native_safetensors.rs`'s `assert_eq!(…, 0)` asserted a literal. A staging
buffer, transpose, repack, or alignment copy could have been reintroduced — the
exact thing the doctrine forbids — and every gate would have stayed green.

The conformer and retained Mimi implementation now have real per-object tallies.
The current LFM2.5 model accounting sums the conformer and
`LfmAudioDetokenizerPlan::compatibility_copied_bytes`; all remain zero because
nothing materializes a weight today—but now that is a measurement rather than
a literal.
No deliberately copying compatibility builder is retained merely to make the
counter positive; doing so would preserve the forbidden implementation beside
its replacement. The gate is therefore two-sided structurally: real-checkpoint
accounting must report zero, while dependency, symbol, and source audits reject
every weight-staging/repack owner from the production graph.

**Invariant:** any code that materializes a weight MUST add its bytes to the
owning tally, exactly as binding adds to `bound_weight_bytes`. A weight-norm fold
or formula table is DERIVED, not materialized, and belongs in `derived_bytes`.

**Honest limit:** a runtime counter is only as strong as that discipline — it
counts what someone increments, no more. The genuinely structural gate for "no
Candle duplicate" is the dependency tree: the default build links **zero**
candle (`cargo tree -p liquid-audio -e normal` → 0 `candle-core`), which is a
fact no author can forget to update.

### Native conversation and recurrence

- `LfmConversation` owns fixed BF16 KV and ShortConv state, frontend/resampler/
  Conformer/audio-detokenizer workspaces, bounded tokenizer storage, sampler
  PRNG, generation
  cadence, context cursor, and epoch-sensitive state.
- Text, PCM, and mixed text+PCM actions validate their complete row requirement
  without mutating the window before the first backbone pass. Eviction then occurs
  causally per row/chunk, so future input rows cannot evict context needed by the
  first row. No caller supplies hidden geometry.
- `LfmSession` owns bounded commands, ticket-correlated reliable text/terminal
  events, capture/playback leases, interruption epochs, stop, join, and the native
  token → sample → Depthformer → audio-detokenizer recurrence loop. A stale
  pass may finish but
  cannot publish.
- Operations own no waiters. Producers publish exact edges; a suspended
  orchestration is a fixed stackless frame holding its program counter, locals,
  exact ticket, and retained record references. Any free eligible kcoro worker
  may resume it. Only an otherwise-idle runtime worker may enter indefinite
  expected-value dormancy. The model gate, engine SQ/CQ, team generations,
  event/command capacity, and PCM leases do not poll or spin.

### Exact context contract

- The live window keeps `position`, physical `start`, absolute `rope_base`, and a
  monotonic public `cursor`. A fixed runway of
  `min(configured_capacity, 256)` avoids copying on each eviction.
- When the runway fills, retained K/V rows compact in place; ShortConv carry is
  preserved. Retained keys are never re-rotated, and new absolute RoPE rows come
  from `lfm_rope_range_f32` into preallocated scratch.
- This is exact latest-window activation-state continuation. It is not presented
  as raw-tail replay equivalence because retained K/V already encode evicted
  history.

### Atomic product and dependency cutover

- `packages/desktop/src-tauri/src/voice/runtime.rs` caches one
  `NativeVoiceModel`; it never constructs the old Rust `LFM2AudioModel`, processor,
  Candle device, Rust safetensors builder, or a simultaneous compatibility image.
- `liquid-audio` defaults to the opaque native lifecycle. Rust model/tensor/
  generation exports and dependencies are absent. No Cargo feature restores the
  deleted Rust/Candle implementation or a private numerical conformance ABI.
- Unsupported native Metal and Moshi selections fail explicitly. There is no
  native/Candle, CPU/Metal, or model-version fallback chain.
- Native Mimi source remains build-checked for the future Moshi tranche and can
  bind only the distinct `LFM_WEIGHT_COMPONENT_MIMI`. The LFM2.5 loader never
  populates that component, constructs a Mimi plan/state, or exposes a Mimi
  request. No fallback crosses this model boundary.

## Remaining LFM2 follow-ons

### F0 — Typed audio-input stage pass — landed

`REQ_AUDIO_ENCODE` retains one model-correlated PCM span through prepared
resample, valid-only BF16 frontend, and whole-Conformer workspaces. Its direct
BF16 linears execute as fixed-team substages without nested SQ submission. The
parity gate proves exact stage output and unchanged prepared-storage capacity on
a second pass. No additional per-stage ticket split is required.

### F1 — Pooled completion continuations — landed

The engine substrate is landed: two native request/scratch slots, a capacity-2
SQ/CQ, exact-ticket completion routing, callback-driven follow-on admission, and
full-pass serialization. Slot generation and state form one atomic lease; exact
CQ retains that lease across the callback. The bridge itself is again a literal
stackless continuation: the final lane publishes one generation and resumes its
exact ticket, and a free kcoro worker continues from the saved program counter.
The bounded route still uses a fixed 64-instance pool and fair broker. An
exact-CQ callback commits the declared state, releases its pass slot, marks only
the next coarse node ready, and rings the broker; it does not wait, allocate, or
take a submission/descriptor mutex. FIFO sequence order plus bounded age
promotion chooses ready work, so a route does not retain either capacity-2
compute slot across a node boundary.

The C++ session coordinator owns one pooled `SessionAction`. Text uses a
terminal single-node token route; audio uses token → Depthformer → released
audio detokenizer. Both
return after admission, ring the existing expected-value doorbell at terminal
completion, and are collected by exact generation. No engine slot remains
mounted while a route waits for playback or reliable-output capacity. One
mutating route per conversation and one scratch slot per admitted program remain
hard invariants.

Correct full/suffix/audio prefill is native and production-owned. Its M≤4
checkpoint-layout BF16 specialization reuses each loaded weight vector across
the row group without widening or packing, preserves causal KV/ShortConv commit
order, and chunks longer prompts (including 4+3 tails) under one conversation
execution claim.

### F2 — Physical kcoro audio-device adapter — landed

Platform callbacks remain in Rust but own only opaque non-cloneable endpoints.
The capture callback passes its ephemeral interleaved span to native format
conversion, which writes one sealed circular-arena reservation and publishes a
typed chunk or XRUN edge. The playback callback resolves a retained native span,
renders directly into the device buffer, and releases its exact lease. Native
Sesame policy commits turns; bounded transcripts/control remain reliable while
waveform telemetry may be lossy.

### Subsequent — Native Moshi

Moshi remains a supported future model, not part of this completed LFM2 tranche.
Its full model and codec recurrence must move onto the same image/session/leaf
discipline before it can return to the production graph. Until then it remains
offline/oracle-only and cannot serve as fallback.

## Gates and current evidence

- On 2026-07-20, the complete default aarch64 suites passed: **86 kcoro tests**
  plus **146 liquid-audio tests**, with only the one-million calibration and two
  real-checkpoint gates ignored by default. The same default suites passed as
  x86_64 binaries under Rosetta. Both ignored real-checkpoint gates were then
  invoked explicitly on a complete LFM2.5-Audio main-plus-detokenizer image:
  model
  accounting passed, and the deterministic two-native-agent in-memory speech
  exchange passed twice inside one invocation.
- On 2026-07-16, the focused default-graph aarch64 run passed **32 tests** with
  two explicit opt-in tests ignored: native safetensors/schema 17/18,
  session/lease 8/9, rollover 3/3, mixed-turn admission 2/2, and tokenizer 2/2.
- The allocation-free lease gate completed 100,000 cycles in 0.030 s (about
  3.38 million cycles/s). The separate one-million-cycle soak remains an explicit
  opt-in gate.
- `engine_idle_zero_spin` measured 0.003% cold-idle and 0.004% post-pass process
  CPU with eight parked lanes.
- Rollover and schema fixtures pass on both aarch64 and x86_64/Rosetta. They cover
  absolute RoPE, latest-window retention, whole-action admission, shared-model
  fairness, dtype/shape swaps with equal byte counts, missing middle layers, and
  vocabulary/codebook mismatch.
- `cargo check -p liquid-audio --no-default-features` passes. The default feature
  declaration does not enable Candle or Moshi.
- The real-checkpoint `LFM_MODEL_DIR` gate is intentionally explicit. It checks
  one complete main+detokenizer lifecycle image and
  `compatibility_copied_bytes == 0`; reviews must not report it as run when the
  checkpoint is unavailable.
- Stop, interruption, reliable-event saturation, capture/playback backpressure,
  stale generations, callback failure, and exact join/release behavior have
  implementation-backed tests. No ignored test is silently counted as green.

## Current default Rust surface

```text
src/
  lib.rs                    native-only exports
  ffi.rs                    private opaque native declarations
  native_voice.rs           RAII lifecycle + opaque callback endpoints/events
  voice_api.rs              product VoiceEngine/VoiceEvent boundary
  runtime/voice_runtime.rs  opaque native service/platform handles, control/telemetry
  utils.rs                  model location/download helpers
```

`src/model/**`, processor/training code, direct numerical Rust rims, Candle, and
Moshi are absent from the workspace model path. Git history is the reference
for deleted Rust ownership, not an alternate inference or training runtime.
