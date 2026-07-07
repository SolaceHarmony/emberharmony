# The NEON "zoo" — aarch64 SIMD mirrors of the Metal JIT kernels

`csrc/neon_zoo.cpp` is a library of hand-written aarch64 NEON procedures that mirror the GPU
idioms of the crate's JIT-embedded Metal kernels (the MLX-style MSL strings compiled at
runtime in `experiments/lfm2-audio-voice/candle-flashfftconv/`). Each Metal construct is
mapped to its closest — deliberately obscure — NEON opcode, "as tight as a GPU." The Rust
bridge is `src/neon_zoo.rs`; the build is wired in `build.rs`.

The headline win is the **tightened bf16 GEMM**, which replaces the reference kernel
(`bf16_gemm.c`) on the live CPU bf16-matmul path (`Bf16Gemm::cpu_fwd`, reached from
`model/linear.rs` `bf16_matmul` / `linear_logits`). The rest ship as a tested, adoptable
library.

## Metal idiom → NEON opcode map

| Metal / GPU construct | NEON procedure(s) | Opcode | Feature |
|---|---|---|---|
| `simdgroup_multiply_accumulate` (`simdgroup_float8x8`, fp32 accum), `fused_monarch.rs` | `lfm_bf16_gemm_f32_v2` | **BFMMLA** `vbfmmlaq_f32` | FEAT_BF16 |
| skinny GEMV / dot form (decode step) | `lfm_bf16_gemv_f32` | **BFDOT** `vbfdotq_f32` | FEAT_BF16 |
| int tensor-core MAC (dtype generalization) | `lfm_s8_gemm_s32` | **SMMLA** `vmmlaq_s32` | FEAT_I8MM |
| threadgroup reduce (would-be `simd_sum`) | `lfm_reduce_sum_f32` / `_max_f32` | **ADDV/FADDP** `vaddvq_f32` | baseline |
| `simd_shuffle` / gather | `lfm_permute_u8` | **TBL/TBX** `vqtbl1q_u8` | baseline |
| bf16-store / f32-accum conv1d, `conv1d.rs` | `lfm_depthwise_causal_conv1d_bf16` | FMA + **BFCVT** `vcvth_bf16_f32` | FEAT_BF16 |
| radix-2 complex butterfly, `FFTConv.metal` | `lfm_fft_radix2_f32` | **FCMLA** `vcmla_f32`+`_rot90` | FEAT_FCMA |
| double-double `two_prod`/`two_sum`, `double_double.metal` | `lfm_dd_sum_f32` / `lfm_dd_dot_f32` | FMA error-free transforms | baseline |
| GPU `rcp` / `rsqrt` fast-math | `lfm_recip_f32` / `lfm_rsqrt_f32` | **FRECPE/FRSQRTE** + Newton | baseline |
| `threadgroup` shared memory + staging | thread-local packed panels + **PRFM** (`__builtin_prefetch`) | — | — |
| `threadgroup_barrier` + `dispatch_thread_groups` | rayon row-block tiling (Rust side, reuses `threads.rs`) | — | — |

## The tightened GEMM

`C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16`, f32 accumulate (torch's CPU bf16 numerics). The
8×8 output tile is a 4×4 grid of BFMMLA 2×2 sub-tiles → **16 independent `float32x4_t`
accumulators**, mirroring `simdgroup_float8x8`. Per 4-deep K-block: 8 bf16 loads (4 A
row-pairs + 4 B col-pairs) feed 16 `vbfmmlaq_f32` with independent accumulator chains — the
instruction-level parallelism the 2×2 reference kernel lacks. A/B are packed **once** into
thread-local scratch (reused across same-shape calls, no per-call `calloc`), zero-padded to
8×8×(K→4) so full tiles are always in-bounds; the ragged edge is masked on store. `M==1`
(the autoregressive decode step) takes the BFDOT GEMV instead of wasting half of BFMMLA on a
padded row. Rust parallelizes over M-row blocks with rayon (`bf16_gemm_into`), one block per
task == one Metal threadgroup per `(batch,head)`.

Numerics: products stay exact (bf16×bf16 in f32); only the summation order differs from the
reference, verified inside `rel < 1e-2` up to K=512.

## Build & feature gating

`build.rs` compiles `neon_zoo.cpp` only on aarch64 (`cfg(has_neon_zoo)`). Feature-specific
opcodes are confined to functions carrying a per-compiler target attribute:

* **clang** (macOS, the shipped build) exposes ACLE intrinsics only when the base `-march`
  enables the feature, so clang gets `-march=armv8.3-a+bf16+i8mm` and the in-file
  `ZOO_TGT_*` macros are empty.
* **gcc** (Linux) always declares the intrinsics and honours per-function
  `target("arch=…")`, so it keeps a low base march (`armv8.2-a`) and each opcode stays
  isolated to its function — nothing leaks into an ungated function.

Every call is runtime-gated on `neon_zoo::NeonFeatures` (macOS `sysctl hw.optional.arm.FEAT_*`
+ Linux `getauxval` HWCAP/HWCAP2 — the latter also fixes the old bf16 probe's Linux `false`).
A binary stays portable: a feature-specific proc is never called on a core that lacks it.

## Verification

* **Rust** (`src/neon_zoo.rs`, CI `cargo test --lib` on macos-arm64): `gemm_v2_matches_f32_bf16_ref`
  (ragged + large-K), `rsqrt_matches_scalar`, `dd_sum_beats_naive`. Each self-skips when its
  feature is absent.
* **Standalone qemu harness** (development): cross-compile with `aarch64-linux-gnu-g++
  -march=armv8.2-a` and run under `qemu-aarch64 -cpu max` — 18 checks across all six groups
  vs scalar / f64 references (GEMM, GEMV, SMMLA, reductions, TBL, conv1d, FFT forward +
  round-trip, double-double vs f64, fast-math).
