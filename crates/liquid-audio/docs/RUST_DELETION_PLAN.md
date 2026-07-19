# Rust inference deletion plan

Status: **LFM2 production numerical ownership cutover complete; fixed-team and
repository-structure follow-on ledger**, audited against the working tree on
2026-07-16.

## Ruling

Rust owns platform microphone/speaker callbacks, VAD/endpointing, opaque
lifetimes, settings/control mapping, and host projection. It owns no production
model math, DSP, tensor, weight, token, sampling, KV/codec state, model-pass
scheduling, or recurrence.

C++ owns native plans, immutable views, conversations, sessions, queues, leases,
stages, and recurrence. Production pass arithmetic belongs to typed
AArch64/x86_64 leaves; approved Apple matrix machinery is reached behind that
leaf boundary. Formula-derived immutable tables may be constructed at readiness
and are reported separately. Layout, alignment, dtype, transpose, framework
ownership, or convenience copies of weights are forbidden.

The default crate and desktop production graph are native-only. Candle, the old
Rust inference implementation, training, fixture capture, and Moshi are gated by
the opt-in `oracle` feature; the workspace-only `liquid-audio-oracle` crate
currently re-exports that feature rather than physically owning those sources.
That source move remains repository-structure work. An oracle is never a
production fallback.

## As-built / open-gaps ledger

| Area | Working-tree state | Remaining work |
|---|---|---|
| Main + codec weights | **Landed.** One byte-exact allocation, direct parallel positioned reads, component-scoped catalog, source handles closed, image page-table read-only after validation. | Keep real-checkpoint digest/load benchmarks as release gates. |
| Typed binding | **Landed.** Exact BF16/F32 dtype, rank, shape, layer, codebook, and vocabulary checks; possibly unaligned tensors remain byte views. | None for LFM2. |
| Weight consumption | **Landed.** Frontend, Conformer, backbone, Depthformer, and Mimi bind the same image; BF16 unlift occurs in registers. | `compatibility_copied_bytes == 0` remains an acceptance assertion. Its counters were **stubs returning a literal 0** (review 2026-07-16) — the gate could not fail; now wired to real per-plan tallies. See "Accounting is a tally, not a constant" below. |
| Native model chain | **Landed for numerical ownership.** One typed, model-correlated `REQ_AUDIO_ENCODE` pass owns resample → valid-only BF16 frontend → whole Conformer/adapter over borrowed spans and pre-reserved conversation buffers. Modality assembly, M≤4 checkpoint-BF16 prefill, backbone, sampling, Depthformer, Mimi, and tokenizer are also native-owned. | No remaining numerical-stage ownership gap for LFM2. The coordinator-to-continuation scheduling cut is tracked in the conversation/session row. |
| Conversation/session | **Landed.** Native KV/ShortConv/codec state, PRNG, cursor, recurrence, text/PCM tickets, reliable events, epochs, interrupt, stop, and join. Rust does not drive progress. A fixed route pool and native expected-value broker release capacity between coarse nodes; exact-CQ callbacks only commit, retire the slot, and publish readiness. Text and audio routes notify a coordinator-owned `SessionAction`, which collects the exact handle without a numerical wait. | V2 BlockDomains remain separate scheduler work. |
| Context rollover | **Landed.** Fixed capacity+runway BF16 state, monotonic cursor, absolute RoPE range generation, nonmutating whole-action admission, causal row-by-row eviction, and in-place compaction. | None for the activation-state sliding-window contract. |
| Shared model | **Landed.** Per-conversation state/scratch and a fair model-owned expected-value pass gate; engine `-EBUSY` does not leak as scheduling policy. | Capacity-2 continuations may improve overlap; fairness is already correct. |
| Production graph | **Landed.** Desktop creates `NativeVoiceModel` and opaque native conversations/sessions only; default dependencies do not enable Candle or Moshi. | Native Metal/MLX remains a separate future backend and must fail explicitly until mounted. |
| Physical audio dock | **Partial.** Native generation-checked capture/playback leases and zero-spin doorbells are live. Playback is consumed synchronously from the borrowed native span by an installed Rust `PcmSink`; the lease is then released and only `Playback { ticket }` crosses the control channel. | Capture still copies `Utterance.samples` into a lease. Direct callback-filled chunks need a distinct VAD commit command so publishing a chunk does not incorrectly begin a turn. |
| Moshi | **Not ported.** It is offline/oracle-only and is not the shipped default. | A full native Moshi port is a subsequent tranche; this LFM2 ledger does not claim it. |

## Completed LFM2 cutover

### One immutable model image

- `native/src/io/safetensors.cpp` opens and fingerprints all selected shards,
  computes checked 64-byte source bases, and allocates exactly one combined
  main+codec image.
- Up to four workers perform retrying 8 MiB positioned reads directly into
  disjoint final spans. There are no chunk allocations, payload staging buffers,
  payload zero-fill, or application payload `memcpy` calls. Only inter-source
  alignment padding is zeroed.
- Every worker joins before failure unwinds the image. The loader deterministically
  selects failures, verifies the same open handles, closes them before publication,
  validates metadata/spans, and seals the allocation read-only.
- `LfmModelMemoryV1` reports source bytes, resident bytes, directly bound bytes,
  formula-derived immutable bytes, compatibility-copied bytes, load time, worker
  count, and task count. Directly bound bytes come from successful exact binders
  (deduplicated by resident span), not from summing every checkpoint entry; unused
  entries therefore remain source/resident bytes without masquerading as consumers.

### Direct native consumers

- `LfmModel` is the sole image owner. Exact byte-addressed views bind embeddings,
  every backbone layer, Conformer, Depthformer, and Mimi. No public production ABI
  exposes names, shapes, weight pointers, mel rows, hidden rows, logits, KV, or
  codec codes.
- BF16 checkpoint storage is not widened, aligned, transposed, packed, or copied.
  Architecture kernels load unaligned little-endian words and unlift them in
  registers; scalar tails use safe byte loads.
- Formula-changing tables—RoPE, frontend/window/FFT, BatchNorm denominators, and
  Mimi folds—are the only admitted derived storage and are accounted separately.
- Frontend power aliases dead STFT real storage, valid mel writes the BF16
  Conformer destination, Conformer writes the native prefill plane, and Mimi
  writes PCM directly into a playback reservation.

### Accounting is a tally, not a constant

Review finding (2026-07-16), fixed: `compatibility_copied_bytes` — the gate this
ledger and spec 15 both cite as the proof that no weight is materialized — was a
**compile-time constant 0**. Both contributors ignored their argument and
returned a literal:

- `lfm_conformer_materialized_weight_bytes(const LfmConformer *c) { (void)c; return 0; }`
- `mimi_decode_plan_compatibility_copied_bytes(const MimiDecodePlan *) { return 0; }`

`lfm_model.cpp` sums exactly those two, so `voice_session.cpp`'s
`if (memory.compatibility_copied_bytes != 0) reject` was dead code and
`native_safetensors.rs`'s `assert_eq!(…, 0)` asserted a literal. A staging
buffer, transpose, repack, or alignment copy could have been reintroduced — the
exact thing the doctrine forbids — and every gate would have stayed green.

Both are now real per-object tallies (`LfmConformer::materialized_weight_bytes`,
`MimiDecodePlan::compatibility_copied_bytes`), sitting beside the tallies that
were already real (`bound_weight_bytes`, `derived_bytes`). They still read 0,
because nothing materializes a weight today — but now that is a *measurement*.
The positive oracle witness is
`rust_owner_drives_the_compatibility_builder_without_reopening_the_file`: it
starts at zero, drives the deliberately copying Candle bridge, and requires the
counter to report exactly one tensor and its payload bytes. Production plan
counters remain zero because there is intentionally no legal production
copying path to exercise.

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
  Conformer/Mimi workspaces, bounded tokenizer storage, sampler PRNG, generation
  cadence, context cursor, and epoch-sensitive state.
- Text, PCM, and mixed text+PCM actions validate their complete row requirement
  without mutating the window before the first backbone pass. Eviction then occurs
  causally per row/chunk, so future input rows cannot evict context needed by the
  first row. No caller supplies hidden geometry.
- `LfmSession` owns bounded commands, ticket-correlated reliable text/terminal
  events, capture/playback leases, interruption epochs, stop, join, and the native
  token → sample → Depthformer → Mimi recurrence loop. A stale pass may finish but
  cannot publish.
- All waits use shared expected-value predicates. The model pass gate, engine
  SQ/CQ, lane fences, event capacity, command capacity, and PCM lease capacity do
  not poll or spin.

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
  generation exports and dependencies are `#[cfg(feature = "oracle")]` only.
  `liquid-audio-oracle` is `publish = false` and opts into that feature for
  training and fixture work. It is currently a thin re-export; physically move
  the oracle/training sources there before calling the repository split complete.
- Unsupported native Metal and Moshi selections fail explicitly. There is no
  native/Candle, CPU/Metal, or model-version fallback chain.
- The standalone `mimi_decoder_new_from_file` parity wrapper is compiled only
  with `oracle`; it is absent from the shared header and production native
  archive. Shipped Mimi can bind only the codec component owned by `LfmModel`.

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
CQ retains that lease across the callback, and deterministic tests cover peer
producer handoff, stale-owner ABA, stop during execution, and capacity
accounting. The bounded route now uses a fixed 64-instance pool (the maximum
runtime session count) and a
native expected-value broker. An exact-CQ callback commits the declared state,
releases its pass slot, marks only the next coarse node ready, and rings the
broker; it does not submit, wait, allocate, or take either submission/descriptor
mutex. Ordinary bridge descriptors are created by the broker and live for one
program. FIFO sequence order plus bounded age promotion chooses ready work, so a
route does not retain either capacity-2 compute slot across a node boundary.

The C++ session coordinator owns one pooled `SessionAction`. Text uses a
terminal single-node token route; audio uses token → Depthformer → Mimi. Both
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

### F2 — Physical kcoro audio-device adapter

Keep platform device callbacks in Rust, but have them reserve/fill capture leases
and drain playback leases directly. Playback now drains its borrowed lease
through `PcmSink` without `Vec<f32>`, `Reply::Audio`, or native
`VoiceEvent::Audio` projection. Delete the remaining capture copy by separating
callback-filled capture publication from the Rust-VAD turn-commit command.
Preserve bounded reliable transcript/control delivery;
only waveform/telemetry observation may be lossy.

### Subsequent — Native Moshi

Moshi remains a supported future model, not part of this completed LFM2 tranche.
Its full model and codec recurrence must move onto the same image/session/leaf
discipline before it can return to the production graph. Until then it remains
offline/oracle-only and cannot serve as fallback.

## Gates and current evidence

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
  one complete main+Mimi lifecycle image and
  `compatibility_copied_bytes == 0`; reviews must not report it as run when the
  checkpoint is unavailable.
- Stop, interruption, reliable-event saturation, capture/playback backpressure,
  stale generations, callback failure, and exact join/release behavior have
  implementation-backed tests. No ignored test is silently counted as green.

## Current default Rust surface

```text
src/
  lib.rs                    native-only exports; oracle modules feature-gated
  ffi.rs                    private opaque native declarations
  native_voice.rs           RAII lifecycle + current PCM/event adapter
  voice_api.rs              product VoiceEngine/VoiceEvent boundary
  runtime/realtime.rs       generic host worker; Candle engine is oracle-only
  runtime/voice_runtime.rs  platform audio, VAD, endpointing, control
  runtime/resample.rs       host/device compatibility; oracle math gated
  utils.rs                  model location/download helpers
```

`src/model/**`, processor/training code, direct numerical Rust rims, Candle, and
Moshi remain reachable only from the non-release oracle graph. Git history is the
reference for deleted production ownership; the oracle is not an alternate
inference runtime.
