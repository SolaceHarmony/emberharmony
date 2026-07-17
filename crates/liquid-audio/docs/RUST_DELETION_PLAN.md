# Rust inference deletion plan

Status: active execution ledger, audited against the working tree on 2026-07-15.

## Ruling

Rust owns audio streams in and out, opaque lifetimes, settings/control mapping,
and host projection. It owns no model math, DSP, tensors, tokens, sampling,
model-pass scheduling, or recurrence.

C++ owns native plans, state, queues, leases, stages, and recurrence. Every
value-producing production operation belongs to an AArch64/x86_64 assembly
leaf. Apple AMX machine code may be reached only through that architecture
assembly ABI; a C++ numerical call site is not an exception.

## Completed In This Tranche

- **Mimi PCM `Tensor` round-trip deleted (P3 seam 1 of 3).** The streaming
  per-frame decode on the production `respond` path now runs host codes
  `&[u32]` → `MimiStreaming::decode_codes` → `AudioDetokenizer::decode_step_codes`
  (native Mimi kernel) → `Vec<f32>` → resampler, with **no `Tensor` in either
  direction**. `decode_step_codes` is now an `AudioDetokenizer` trait method
  (default wraps the `Tensor` `decode_step` for non-native backends; the native
  Mimi backend overrides it with the direct `&[u32]` → `Vec<f32>` kernel path).
  The former `decode_audio_frame` `Tensor` adapter (`moshi/demo/chat.rs`) is
  deleted (dead). Transport-only — PCM values unchanged (native Mimi still
  produces the same `Vec<f32>`); 167 lib tests green. Remaining P3 seams:
  mel→`Tensor` and adapter→`Tensor`, both prefill-coupled.
- **Native audio-in prefill capability (P2/P4 enabler).**
  `lfm_engine_token_pass` `embed_kind == 2` provided-embedding path +
  `lfm_conversation_prefill_audio` (`NativeConversation::prefill_audio`) prefill
  Conformer rows into KV as a borrowed view; parity-proven against the discrete
  embed path (`native_audio_prefill_matches_discrete_for_the_same_embedding`).
  Not yet adopted by production voice (the Candle backbone copy drop needs the
  full `self.lfm` elimination — see doc 14 P2/P3).
- **Depthformer Candle copy dropped on the resident path.**
  `build_depth_decode_resident` (`model/lfm2_audio.rs`) binds the depth plan
  directly from the resident checkpoint image by name (zero-copy), with rope from
  the native `lfm_rope_table_f32` kernel — the same kernel `lfm_model.cpp` feeds
  its native plan. It is now the production depth path; the Candle depth modules
  (`depthformer` / `depth_linear` / `depth_embeddings`) became `Option`, built
  only on the non-resident training path (guarded in the training `forward`).
  Verified byte-identical to the Candle-bound plan by
  `depth_resident_binder_matches_candle_binder` (greedy tokens, so identical
  logits ⇒ identical argmax). Production Candle-copy ledger fell **231 → 151
  tensors, 2.711 → 2.475 GB** (~236 MB of depth weights no longer duplicated). The
  remaining ~2.475 GB is the backbone + audio embedding, whose copy is coupled to
  native prefill (Candle still owns prefill/`forward_embeds`).
- Deleted `src/compute/flashkern/coordinator.rs` and the registered Rust
  submitter callback ABI. `submit_pass` now uses the native SQ/CQ directly.
- Added `raw_engine_owns_its_sq_cq_without_rust_progress`, proving a complete
  pass needs no Rust callback.
- Deleted the 1,693-line Rust `fanout.rs` implementation.
- Reduced `dd.rs` from a Rust arithmetic implementation to a test-only ABI
  record.
- Removed the Candle fallback from audio-frame sampling.
- Added AArch64/x86_64 assembly leaves for reciprocal RMS scaling, fixed-order
  reduction, strided BF16 sum-of-squares, BF16 bias addition, and BF16 NeoX
  rotary.
- Added a no-feature-gate assembly ABI fixture that executes natively and under
  Rosetta.
- Deleted `src/compute/bf16_gemm.rs` and its Candle `CustomOp2` owner. The
  temporary Candle rim now borrows storage and submits one typed `REQ_GEMM`;
  capability truth comes from the native Flashkern ABI.
- **Conformer encoder and audio adapter are native.** Deleted
  `src/model/conformer/*` (encoder, mha, modules, subsampling, utils) and the
  now-orphaned `src/model/mlp.rs` + `src/model/norm.rs`. `lfm_conformer.{h,cpp}`
  binds all encoder+adapter weight views from the resident image and runs one
  segment as a sequence of stages: subsampling (conv2d im2col + f32 GEMM +
  depthwise/pointwise), rel-pos table, 17 Conformer layers (macaron FFN, rel-pos
  attention, conv module with BatchNorm-eval + GLU + depthwise-k9), adapter
  (LN + gelu-erf). Every value comes from `flashkern_conformer.S` (both arches:
  LayerNorm, BatchNorm, SiLU, GLU, gelu-erf, softmax, residuals, dw-conv,
  sgemm/sgemm-nt, pe-table, bias) or the engine bf16 GEMM pass, with the f32
  matmul stages on Accelerate (Apple) / house sgemm leaf. Production BF16
  ladder mirrored exactly (fixtures arbitrate). Baselines
  `native/tests/fixtures/conformer/` (real LFM2.5-Audio checkpoint, per-stage,
  captured from the deleted Rust); gate `tests/native_conformer_parity.rs`
  (shape-first, adapter within BF16-ladder tolerance — measured worst relative
  divergence 5.1e-3 across 1/4/7-row segments). The mel-plane transport tensor
  at the prefill seam still exists; it dies at the doc 07 conversation cutover.

- **Mel frontend and resampler are native.** Deleted the in-crate featurizer
  (hann/slaney/DFT tables, the candle STFT/normalization in `processor.rs`)
  and the pure-Rust windowed-sinc resampler body. `lfm_frontend.{h,cpp}` +
  `flashkern_frontend.S` (both arches) own the math: preemphasis, |X|^2,
  log-guard, ddof-1 row statistics, normalization, and the f64 resample conv
  are assembly leaves; the two matmul-shaped stages ride Accelerate on Apple
  (mimi pattern). Table construction is init-time f64 C++ (the Mimi
  weight-fold class) — its N2 extraction is a named follow-up, not a silent
  exemption. Baselines: `native/tests/fixtures/{mel,resample}/` captured from
  the deleted Rust; gates: `tests/native_frontend_parity.rs` (resampler
  bitwise; mel padded/valid shape-first + tolerance policy). Realtime borrows
  retained 16 kHz PCM directly; the frontend reuses a rim-owned high-water
  workspace, aliases dead signal/DFT/power planes, and writes the valid mel
  destination without a padded crop-copy. The Rust `FilterbankFeatures` /
  `resample_slice` names survive as opaque-handle/transport rims only — the
  remaining Metal mel upload dies with the Conformer cutover (doc 06).

## Current Production Violations

| Seam | Why it is still live | Replacement required before deletion |
|---|---|---|
| `src/model/**` | Rust/Candle still owns backbone/depthformer construction, prefill/generation, sampling, and tensors. The Conformer encoder + adapter are now native (deleted). | Native model/session owns tokenizer, prefill, recurrence, sampling, and state. |
| `src/compute/weights.rs` | Rust can still reconstruct Candle tensors from views into the native resident image. | Native plans bind immutable views directly; delete the Candle builder and tensor-copy adapter. |
| `src/model/linear.rs` | A temporary Candle ownership rim still exposes tensor storage to `REQ_GEMM`; it performs no math. | Production callers use `NativeModel`/`NativeConversation`, then this rim is deleted. |
| `src/compute/flashkern/candle_ops.rs` | ShortConv compatibility path still converts Candle storage. | Native conversation owns convolution carry and typed stage. |
| `native_engine.rs` pass methods | Temporary tests and compatibility callers still submit numerical buffers from Rust. | Rust exposes only PCM/control dock and opaque handles. Numerical methods become native tests or are deleted. |
| `processor.rs` mel rim, `runtime/resample.rs` rim | Frontend math is native. Realtime 16 kHz PCM is pointer-through and valid mel is direct, but the shared compatibility workspace can grow inside a pass and serialize engines; every pass allocates its Candle destination; Metal then uploads it. Non-16 kHz resampling still uses a temporary Rust output and per-call native work. Beyond transport, mel OWNERSHIP also remains Rust: the plan/workspace handles, `ChatState.audio_in` storage and `Tensor::cat` growth, the Candle Conformer feed, and cross-turn persistence in `ConversationState`. | Give each native session a pre-reserved workspace and mel segment, let the resampler write its first plane, and have Conformer consume that segment directly; then delete both rims. Storage/persistence residue (`ChatState`/`ConversationState` mel tensors) clears at the doc 07 conversation cutover; handle ownership transfers when a native session object exists. |
| Rust Mimi/Moshi owners | Continuous model and codec state remain in Rust/Candle. | Native Moshi session and codec recurrence. |
| Candle/moshi dependencies | Required by the remaining seams above. | Remove after every production owner is native and fixture gates pass. |

## Execution Order

### R0 - Native ownership and pass recurrence

1. Finish binding every model component from the resident safetensors image.
2. Replace borrowed engine request storage with native pass slots and leases.
3. Move tokenizer, prompt assembly, prefill, sampling, token/frame recurrence,
   conversation marks, and context switching into the native session.
4. Prove 1,000 native passes recur with no Rust callback or token crossing.

### R1 - Assembly extraction

1. Move remaining engine/model floating-point expressions into typed assembly
   leaves.
2. Move hot architecture `.cpp` numerical bodies to `.S`; retain C++ only for
   capability selection and invocation.
3. Put AMX/Accelerate behind a narrow leaf so the scheduler never performs
   numerical preparation during a pass.
4. Reject scalar production fallbacks; keep scalar oracles test-only.

### R2 - Frontend and codec

1. Native VAD, resample, FFT/mel, normalization.
2. Native Conformer and adapter.
3. Native Depthformer and Mimi/Moshi encode/decode state.
4. End-to-end PCM ingress -> native model -> PCM egress fixtures.

### R3 - Rust audio dock

1. Keep platform mic/speaker stream callbacks in Rust.
2. Add preallocated PCM lease pools and callback-driven kcoro rings.
3. Add one compact control ring and one lossy observer projection.
4. Prove independent capture/playback/control tasks park without blocking each
   other and root cancellation settles every child exactly once.

### R4 - Delete owners immediately after each gate

1. Delete the remaining Candle linear ownership rim after native model callers land.
2. Delete `candle_ops.rs` and Rust convolution state.
3. Delete Rust model/frontend/codec modules as their native owners land.
4. Delete numerical `native_engine.rs` methods and Rust arch wrappers.
5. Remove Candle, candle-nn, candle-transformers, and moshi inference deps.
6. Remove public token/model/generation exports. Git history is the reference.

## Gates

- `cargo test -p liquid-audio --lib` and integration tests remain green during
  each cut.
- Rosetta executes x86 assembly fixtures even when AVX2 tests skip.
- No Rust production FFI takes numerical/model payloads.
- No C++ scheduler/model/session source performs production model arithmetic.
- No allocation or scratch growth occurs after native readiness.
- No model payload copy occurs at Rust/native boundaries.
- Stop, interrupt, timeout, close, and completion settle tickets/leases once.
- Full workspace and Tauri builds pass after Candle removal.

## Final Rust Surface

```text
src/
  lib.rs       opaque handles + audio dock exports
  ffi.rs       lifecycle/settings/control/PCM declarations only
  handles.rs   RAII and status mapping
  audio/
    input.rs
    output.rs
    dock.rs
  tauri.rs     small event/control projection
```

No legacy feature, backup crate, or alternate inference runtime remains.
