# flashkern — CPU replicas of the Metal JIT kernels (aarch64 NEON + x86-64 AVX)

`crates/candle-flashfftconv/` embeds the migration-era Metal kernels as MSL
source. Flashkern is the CPU kernel library: native SIMD lives in
`native/kernels/{aarch64,x86_64}/`, while `fanout.rs`, `dd.rs`, and the Rust
reference programs preserve parity fixtures during the native migration.
`build.rs` wires the native objects into `liquid-audio`.

**Backend boundary:** Flashkern is CPU-only. Metal source is a numerical and
dispatch-shape oracle for CPU translation; no Metal API, Metal command queue, MLX
object, or Metal dependency belongs inside Flashkern. Matrix acceleration is not
optional: aarch64 uses BFMMLA/BFDOT/NEON and x86_64 uses AVX2/AVX-512-BF16. The
required Apple GPU peer is a separate future MLX C++/Metal engine selected above
Flashkern. Until it lands, Candle Metal remains a temporary sibling backend.

The current CPU decode path has two coarse execution forms:

* **Native backbone token pass.** `Lfm2Model::native_token_pass` enters
  `REQ_TOKEN_PASS` once for embed, every ShortConv/attention/MLP block, final norm,
  and optional logits. The fixed C++ lane team executes the whole program using
  the NEON/AVX procedures below. Its request crosses the mounted safe Rust kcoro
  broker and native SQ/CQ; it does not run the old Rust fused-MLP dispatch. The
  `fused_mlp_reference` function in `decode.rs` is test-only.
* **Native Depthformer frame pass.** `DepthDecode::frame` enters
  `REQ_DEPTH_FRAME` once. `run_depth_frame` owns projection, every codebook and
  transformer layer, tiny KV planes, logits, sampling, and sampled-embedding
  recurrence on the fixed C++ lane team. Weight payloads and the backbone hidden
  row are borrowed by pointer through the exact completion; mutable scratch is
  reserved at plan build. Stage boundaries use the same zero-spin native
  generation fence as the backbone. Rust performs no frame arithmetic.
* **Native streaming short-conv pass.** CPU bf16 prefill/continuation enters
  `REQ_DEPTHWISE_STREAM` once. Lanes own complete `(batch, channel)` rows and call
  `lfm_depthwise_stream_bf16`; the prior `K-1` state, incoming samples, weights,
  output, and next-state are separate borrowed planes. No `[cache | x]` tensor is
  constructed. The program-final generation fence produces one ticket completion.
* **Typed matrix and DD FFT passes.** `REQ_GEMM` owns KN GEMM, checkpoint-native
  NK GEMM, and the M=1 GEMV specialization. `REQ_FFT_CONV_DD` and `REQ_IRFFT_DD`
  own the double-double fan-out paths and f64-split twiddles. FFT convolution uses
  one shared reusable work plane: every fixed lane grid-strides bit reversal and
  butterflies, then crosses the generation fence at each radix-2 stage. The whole
  signal grid remains one ticket. The deleted generic callback cannot put a Rust
  frame on a compute lane.

Individual Candle custom-op and pure-Rust fanout paths still exist for unfinished
prefill/Metal/reference surfaces and strict-gate fallbacks. They are not the target
CPU executor. `set_depth_flash_enabled(false)` remains a parity seam that selects
the Candle depthformer reference.

## Coverage — the full Metal JIT kernel inventory

| Metal JIT kernel (candle-flashfftconv) | CPU port | Where |
|---|---|---|
| `fft_radix2` (FFTConv.metal) | `fft_lane` / `fused_fft`; flat FCMLA kernel `lfm_fft_radix2_f32` | fanout.rs; both `.cpp` |
| `fft_conv` — the FlashFFTConv product kernel | `fused_fft_conv` | fanout.rs |
| `rfft_kernel` / `irfft_kernel` | `rfft` / `irfft` | fanout.rs |
| `fft_conv_dd` (FFTConvDd.metal) | `fused_fft_conv_dd` | fanout.rs + dd.rs |
| `irfft_dd` (IrfftDd.metal) | `irfft_dd` | fanout.rs + dd.rs |
| `double_double.metal` toolkit | `src/compute/flashkern/dd.rs` (formulation-faithful); SIMD `lfm_dd_sum/dot_f32` | dd.rs; both `.cpp` |
| `complex_mul` (ComplexMul.metal) | `lfm_complex_mul_f32` — fixed order, **no FMA** | both `.cpp` |
| `complex_mul_dd` (dd_complex_mul.rs) | `dd::cdd_mul` (correctly-rounded, tested) | dd.rs |
| `depthwise3` / `depthwise3_causal` (Depthwise3.metal) | `lfm_depthwise3(_causal)_f32` — fixed order, **no FMA** | both `.cpp` |
| `depthwise_causal_conv1d_bf16` (conv1d.rs) | `lfm_depthwise_causal_conv1d_bf16` | both `.cpp` |
| streaming `depthwise_conv1d_stream` contract (conv1d.rs) | `REQ_DEPTHWISE_STREAM` + `lfm_depthwise_stream_bf16`; split state/input pointers, one fixed-team ticket | engine + both `.cpp` |
| `causal_conv1d_update_fused_{float,bfloat}` (conv1d_update.rs) | `lfm_conv1d_update_{f32,bf16}` — FMA (trained regime) | both `.cpp` |
| `monarch_fused_fwd_f32` (fused_monarch.rs) | `fused_monarch_fwd` | fanout.rs |
| `monarch_fused_conv_f32` / `_padded_f32` (gates + u·D skip) | `fused_monarch_conv` / `_padded` | fanout.rs |
| `butterfly_row_dft` / `_twiddle` / `_col_dft` (butterfly.rs) | the same three stages inside the monarch ports | fanout.rs |
| `butterfly_row_idft_real_f32` | `row_idft_real` | fanout.rs |
| `sg_probe_f32` | n/a — GPU simdgroup capability probe | — |

**Native decode-engine additions** (not candle-flashfftconv Metal ports — new CPU kernels for
the fused decode path, so they sit outside the inventory above):

* **`lfm_bf16_gemm_nt_f32`** — the native-layout `[N,K]` decode GEMM/GEMV (`Bf16GemmNt`):
  `A[M,K]·W[N,K]ᵀ`, each output dotting a CONTIGUOUS weight row, rayon over N. No transpose
  copy. Both `.cpp`. See *The tightened GEMM* below.
* **Group H — decode stage kernels** (both `.cpp`): the per-stage device functions of
  `DepthDecode` — `lfm_bf16_sumsq_f32`, `lfm_bf16_rmsnorm`, `lfm_bf16_add`, `lfm_swiglu_bf16`,
  `lfm_softmax_scaled_f32`, `lfm_attn_qk_f32`, `lfm_attn_av_f32`, `lfm_rope_i_f32`,
  `lfm_bf16_to_f32`, `lfm_f32_to_bf16`. See *Group H* below.

**FP-contraction policy.** Both `.cpp` TUs compile with `-ffp-contract=off`: kernels that
promise the Metal source's fixed evaluation order (`complex_mul`, `depthwise3`, the FFT
butterflies) must not have `a·b − c·d` silently fused at `-O3`. Fusion is always *explicit* —
intrinsics (`vfmaq`/`_mm256_fmadd`/BFMMLA) or `fmaf` — and reserved for kernels whose trained
regime is contracted (`conv1d_update`, GEMM) or that need exact FMA residuals (double-double).

## Metal idiom → NEON opcode map

| Metal / GPU construct | NEON procedure(s) | Opcode | Feature |
|---|---|---|---|
| `simdgroup_multiply_accumulate` (`simdgroup_float8x8`, fp32 accum), `fused_monarch.rs` | `lfm_bf16_gemm_f32_v2` | **BFMMLA** `vbfmmlaq_f32` | FEAT_BF16 |
| skinny GEMV / dot form (decode step) | `lfm_bf16_gemv_f32` | **BFDOT** `vbfdotq_f32` | FEAT_BF16 |
| int tensor-core MAC (dtype generalization) | `lfm_s8_gemm_s32` | **SMMLA** `vmmlaq_s32` | FEAT_I8MM |
| threadgroup reduce (would-be `simd_sum`) | `lfm_reduce_sum_f32` / `_max_f32` | **ADDV/FADDP** `vaddvq_f32` | baseline |
| `simd_shuffle` / gather | `lfm_permute_u8` | **TBL/TBX** `vqtbl1q_u8` | baseline |
| bf16-store / f32-accum conv1d, `conv1d.rs` | `lfm_depthwise_causal_conv1d_bf16` | FMA + **BFCVT** `vcvth_bf16_f32` | FEAT_BF16 |
| streaming depthwise rows | `lfm_depthwise_stream_bf16` | NEON FMA+BFCVT / AVX2 FMA | FEAT_BF16 / AVX2+FMA |
| radix-2 complex butterfly, `FFTConv.metal` | `lfm_fft_radix2_f32` | **FCMLA** `vcmla_f32`+`_rot90` | FEAT_FCMA |
| double-double `two_prod`/`two_sum`, `double_double.metal` | `lfm_dd_sum_f32` / `lfm_dd_dot_f32` | FMA error-free transforms | baseline |
| GPU `rcp` / `rsqrt` fast-math | `lfm_recip_f32` / `lfm_rsqrt_f32` | **FRECPE/FRSQRTE** + Newton | baseline |
| `threadgroup` shared memory + staging | thread-local packed panels + **PRFM** (`__builtin_prefetch`) | — | — |
| `threadgroup_barrier` + `dispatch_thread_groups` | fixed native generation fence in production; rayon/scoped threads in reference ports | — | — |

## The tightened GEMM

`C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16`, f32 accumulate (torch's CPU bf16 numerics). The
8×8 output tile is a 4×4 grid of BFMMLA 2×2 sub-tiles → **16 independent `float32x4_t`
accumulators**, mirroring `simdgroup_float8x8`. Per 4-deep K-block: 8 bf16 loads (4 A
row-pairs + 4 B col-pairs) feed 16 `vbfmmlaq_f32` with independent accumulator chains — the
instruction-level parallelism a single 2×2 accumulator lacks. A/B are packed **once** into
thread-local scratch (reused across same-shape calls, no per-call `calloc`), zero-padded to
8×8×(K→4) so full tiles are always in-bounds; the ragged edge is masked on store. Rust
parallelizes over M-row blocks with rayon (`bf16_gemm_into`), one block per task == one Metal
threadgroup per `(batch,head)`.

**The decode step (`M==1`) never transposes the weight.** Two hard-won rules, both learned by
profiling a 0.13 tok/s CPU decode down to its 97%-of-time hot spot (weight copies, not math):

* `bf16_gemm_nt_into` / `lfm_bf16_gemm_nt_f32` consume the weight in its checkpoint-native
  row-major `[N,K]` layout — each output dots a CONTIGUOUS weight row, rayon fans the N
  outputs across cores. The model's `matmul_flat`/`linear_logits` route `rows ≤ 4` here with
  the weight AS STORED: no `.t()`, no `.contiguous()`, no copy of any kind.
* The transposed-layout GEMV (`lfm_bf16_gemv_f32`, kept for `[K,N]` callers) is the
  row-streaming axpy form: B read once, contiguously — never a per-call staging transpose.

Numerics: products stay exact (bf16×bf16 in f32); only the summation order differs from the
reference, verified inside `rel < 1e-2` up to K=512.

## Group H — the decode stage kernels

Group H is the per-stage device-function library the typed C++ Depthformer
program in `native/src/engine/flashkern_engine.cpp` executes between native
zero-spin generation fences. Each kernel consumes/produces bf16 bit planes or f32 scratch
**exactly at the torch rounding points**, so the flash frame is value-equivalent to the candle
op chain at bf16 resolution; the fixed C++ lane team slices rows and fences between stages:

| kernel | stage | ladder |
|---|---|---|
| `lfm_bf16_sumsq_f32` | RMSNorm reduction | `Σ f32(x)²` (f32 accumulate, `vfmaq`) |
| `lfm_bf16_rmsnorm` | RMSNorm apply | `out = rb(f32(x)·inv_rms·f32(w))` — f32 throughout, ONE round |
| `lfm_bf16_add` | residual add | `out = rb(f32(a) + f32(b))` |
| `lfm_swiglu_bf16` | SwiGLU | `out = rb(rb(silu(rb(g)))·rb(u))` — the op chain's rounds |
| `lfm_softmax_scaled_f32` | attention softmax | scaled, max-subtracted, f32 |
| `lfm_attn_qk_f32` / `lfm_attn_av_f32` | attention scores / weighted-sum | per-head dots over the resident planes |
| `lfm_rope_i_f32` | interleaved RoPE | in-place rotate at `pos` against the shared cos/sin table |
| `lfm_bf16_to_f32` / `lfm_f32_to_bf16` | dtype crossings | exact upcast / RNE store (bf16 `type_as`) |

`run_depth_frame` folds `inv_rms` and per-frame reductions in fixed lane order,
then calls these vectorized bodies. Both architecture TUs carry the full set under
identical `extern "C"` names, so the typed frame program is arch-agnostic exactly
as the GEMM is.

## Build & feature gating

`build.rs` compiles `flashkern_neon.cpp` on aarch64 and `flashkern_x86.cpp` on x86-64.
Unsupported architectures fail the build. Feature-specific
opcodes are confined to functions carrying a per-compiler target attribute:

* **clang** (macOS, the shipped build) exposes ACLE intrinsics only when the base `-march`
  enables the feature, so clang gets `-march=armv8.3-a+bf16+i8mm` and the in-file
  `FK_TGT_*` macros are empty.
* **gcc** (Linux) always declares the intrinsics and honours per-function
  `target("arch=…")`, so it keeps a low base march (`armv8.2-a`) and each opcode stays
  isolated to its function — nothing leaks into an ungated function.

## No fallbacks

flashkern is an architecture-kernel library, not a scalar portability layer. Procedures
compile under plain `#[cfg(target_arch)]` and call the matching NEON or x86 kernel; there
is deliberately **no scalar implementation masquerading as flashkern**. Runtime feature
checks prevent feature-specific procedures from executing on unsupported cores.
Feature-gated procedures
(FFT→FCMA, `s8_gemm`→I8MM, `conv1d`→BF16) document their precondition; verify
`flashkern::neon::NeonFeatures` (macOS `sysctl hw.optional.arm.FEAT_*` + Linux `getauxval` HWCAP/HWCAP2
— the latter also fixes the old bf16 probe's Linux `false`) before calling.
The x86 streaming pass returns `ENOTSUP` before submission when AVX2/FMA is not
advertised. Rosetta therefore skips that opcode leaf; it does not run a scalar
substitute and does not weaken the x86 production contract.

## Verification

* **Rust** (`src/compute/flashkern/neon.rs`): the `#[cfg(test)] mod tests` is itself aarch64+flashkern-gated, so it
  runs on the macOS arm64 CI leg (`rust-voice.yml`) where the hardware executes the kernels — on
  x86 CI the crate still builds, there is just nothing here to run. Feature-specific tests
  (`gemm_v2`, `conv1d`, `s8_gemm`, `fft`) skip when the runner's CPU lacks the extension (e.g. an
  M1 runner has FCMA but not BF16/I8MM); the baseline ones (`rsqrt`, `dd_sum`) always run.
* **Standalone qemu harness** (development): cross-compile with `aarch64-linux-gnu-g++
  -march=armv8.2-a` and run under `qemu-aarch64 -cpu max` — 18 checks across all six groups
  vs scalar / f64 references (GEMM, GEMV, SMMLA, reductions, TBL, conv1d, FFT forward +
  round-trip, double-double vs f64, fast-math).
* **Typed streaming parity:**
  `typed_depthwise_stream_matches_split_buffer_oracle_and_uses_one_ticket` pins
  fresh/resumed state, `T < K-1`, `K=1`, output bits, next-state bits, and exactly
  one SQ/CQ completion per pass. `typed_stream_matches_metal_reference_contract_across_chunks`
  compares chunked CPU output/state against the temporary reference backend.
* **Typed launch accounting:**
  `typed_gemm_layouts_and_gemv_use_one_ticket_each` and
  `typed_fft_grids_use_one_ticket_each` prove one submission/completion per
  matrix or FFT grid and zero live descriptors afterward.
  `typed_fft_is_bit_exact_across_physical_lane_counts` proves the cooperative
  one-worker and four-worker transforms are bit-identical and records every
  threadgroup-equivalent stage fence. DD FFT/IRFFT retain their f64-oracle gates.

## x86-64 sibling (`native/kernels/x86_64/flashkern_x86.cpp` + `src/compute/flashkern/x86.rs`)

The Intel/AMD sibling exposes the **same `extern "C"` kernels** (identical symbol names), so the
live `bf16_matmul` path and the Rust wrappers are arch-agnostic — `build.rs` compiles exactly one
of the two per target, and `Bf16Gemm::cpu_fwd` dispatches to `flashkern::neon` on aarch64 or `flashkern::x86`
on x86-64. The idiom → opcode map crosses over directly:

| NEON | x86-64 |
|---|---|
| BFMMLA / BFDOT | **VDPBF16PS** (`_mm512_dpbf16_ps`, AVX-512-BF16), else AVX2 upconvert+FMA |
| BFCVT (bf16 store) | **VCVTNEPS2BF16** |
| TBL/TBX | **PSHUFB** (`_mm256_shuffle_epi8`) |
| ADDV/FADDP | AVX horizontal reduce |
| FRECPE / FRSQRTE + Newton | **RCPPS / RSQRTPS** (`_mm256_rcp_ps` / `_mm256_rsqrt_ps`) + Newton |
| SMMLA | **VPMADDWD** (`_mm512_madd_epi16`, AVX-512-BW) |
| double-double two_prod/two_sum | FMA error-free transforms (`_mm256_fmadd_ps` / `fmsub`) |

**Fan-out.** The GPU threadgroup-grid dispatch maps to the CPU's cores: `flashkern::x86::bf16_gemm_into`
fans the GEMM out over M-row blocks with rayon (reusing `threads.rs`'s physical-core pool), one
task per block running the SIMD micro-kernel — exactly as the NEON side does. The bf16 GEMM also
dispatches internally on CPUID: VDPBF16PS when AVX-512-BF16 is present, else the AVX2 baseline
(present on essentially all x86-64), so the same call works across the Intel/AMD fleet.

Runtime-gated on `flashkern::x86::X86Features` (`is_x86_feature_detected!`); baseline is AVX2 + FMA,
with `s8_gemm` requiring AVX-512F/BW. **Verified natively** (this is x86, no emulation needed):
`cargo test --lib` on an AVX-512-BF16 host — the live `bf16_gemm_matches_f32_reference` plus the
`flashkern::x86` suite (fan-out GEMM, GEMV, SMMLA, PSHUFB, FFT round-trip + non-pow2 rejection,
double-double vs f64, fast-math, size-guard `should_panic`) all pass.
