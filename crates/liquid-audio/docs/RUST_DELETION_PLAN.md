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

## Current Production Violations

| Seam | Why it is still live | Replacement required before deletion |
|---|---|---|
| `src/model/**` | Rust/Candle still owns portions of model construction, tensors, and calls. | Native model/session owns tokenizer, frontend, all plans, recurrence, and state. |
| `src/compute/weights.rs` | Rust can still reconstruct Candle tensors from views into the native resident image. | Native plans bind immutable views directly; delete the Candle builder and tensor-copy adapter. |
| `src/model/linear.rs` | A temporary Candle ownership rim still exposes tensor storage to `REQ_GEMM`; it performs no math. | Production callers use `NativeModel`/`NativeConversation`, then this rim is deleted. |
| `src/compute/flashkern/candle_ops.rs` | ShortConv compatibility path still converts Candle storage. | Native conversation owns convolution carry and typed stage. |
| `native_engine.rs` pass methods | Temporary tests and compatibility callers still submit numerical buffers from Rust. | Rust exposes only PCM/control dock and opaque handles. Numerical methods become native tests or are deleted. |
| `processor/mel.rs`, `runtime/resample.rs` | Rust still computes frontend values. | Native frontend plan and assembly kernels. |
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
