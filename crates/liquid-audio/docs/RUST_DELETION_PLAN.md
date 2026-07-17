# Rust inference deletion plan

Status: **LFM2 production ownership cutover complete; follow-on ledger**, audited
against the working tree on 2026-07-16.

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
Rust inference implementation, training, fixture capture, and Moshi are isolated
behind the opt-in `oracle` feature and workspace-only `liquid-audio-oracle` crate.
An oracle is never a production fallback.

## As-built / open-gaps ledger

| Area | Working-tree state | Remaining work |
|---|---|---|
| Main + codec weights | **Landed.** One byte-exact allocation, direct parallel positioned reads, component-scoped catalog, source handles closed, image page-table read-only after validation. | Keep real-checkpoint digest/load benchmarks as release gates. |
| Typed binding | **Landed.** Exact BF16/F32 dtype, rank, shape, layer, codebook, and vocabulary checks; possibly unaligned tensors remain byte views. | None for LFM2. |
| Weight consumption | **Landed.** Frontend, Conformer, backbone, Depthformer, and Mimi bind the same image; BF16 unlift occurs in registers. | `compatibility_copied_bytes == 0` remains an acceptance assertion. |
| Native model chain | **Landed.** Resample, mel, Conformer/adapter, modality assembly, backbone, sampling, Depthformer, Mimi, and tokenizer are native-owned. | Multi-row prefill specialization is still open; correct prefill currently advances admitted rows through the native token pass. |
| Conversation/session | **Landed.** Native KV/ShortConv/codec state, PRNG, cursor, recurrence, text/PCM tickets, reliable events, epochs, interrupt, stop, and join. Rust does not drive progress. | The native coordinator still synchronously parks on the engine's capacity-1 completion before calling the next pass. Capacity-2 completion continuations are open. |
| Context rollover | **Landed.** Fixed capacity+runway BF16 state, monotonic cursor, absolute RoPE range generation, whole-action reservation, and in-place compaction. | None for the activation-state sliding-window contract. |
| Shared model | **Landed.** Per-conversation state/scratch and a fair model-owned expected-value pass gate; engine `-EBUSY` does not leak as scheduling policy. | Capacity-2 continuations may improve overlap; fairness is already correct. |
| Production graph | **Landed.** Desktop creates `NativeVoiceModel` and opaque native conversations/sessions only; default dependencies do not enable Candle or Moshi. | Native Metal/MLX remains a separate future backend and must fail explicitly until mounted. |
| Physical audio dock | **Partial.** Native generation-checked capture/playback leases and zero-spin doorbells are live. | The Rust adapter still copies `Utterance.samples` into a capture lease and copies playback with `to_vec()` into crossbeam `Reply::Audio`/`VoiceEvent::Audio`. Replace it with direct kcoro device callbacks. |
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
  count, and task count.

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

### Native conversation and recurrence

- `LfmConversation` owns fixed BF16 KV and ShortConv state, frontend/resampler/
  Conformer/Mimi workspaces, bounded tokenizer storage, sampler PRNG, generation
  cadence, context cursor, and epoch-sensitive state.
- Text, PCM, and mixed text+PCM actions validate and reserve their complete row
  requirement before the first backbone mutation. No caller supplies hidden
  geometry.
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
  training and fixture work.
- Unsupported native Metal and Moshi selections fail explicitly. There is no
  native/Candle, CPU/Metal, or model-version fallback chain.

## Remaining LFM2 follow-ons

### F1 — Capacity-2 completion continuations and multi-row prefill

The current C++ coordinator correctly owns recurrence and parks without spin, but
it waits synchronously for each pass through the engine's single mutable request
slot. Move to two native request/scratch slots so a completion can enqueue its
follow-on directly. Preserve full-pass fairness and one scratch slot per in-flight
ticket.

Correct full/suffix/audio prefill is already native and production-owned. Add a
checkpoint-layout BF16 multi-row specialization for long prompts without widening,
packing, or changing row-commit order. This is a performance follow-on, not a
reason to restore Rust recurrence or Candle ownership.

### F2 — Physical kcoro audio-device adapter

Keep platform device callbacks in Rust, but have them reserve/fill capture leases
and drain playback leases directly. Delete the current `Vec<f32>` capture/playback
copies, playback thread projection, crossbeam `Reply::Audio`, and legacy
`VoiceEvent::Audio` bridge. Preserve bounded reliable transcript/control delivery;
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
  absolute RoPE, latest-window retention, whole-action reservation, shared-model
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
