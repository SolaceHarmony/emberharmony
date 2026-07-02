//! `candle-flashfftconv` — FlashFFTConv kernels for candle (CPU + Metal).
//!
//! A reusable crate of the conv operators that ML stacks normally gate behind
//! custom CUDA kernels (`causal_conv1d`, FlashFFTConv), reimplemented as candle
//! [`CustomOp`](candle_core::CustomOp3)s so they run on **CPU** (a faithful
//! reference) **and Metal** (a real fused kernel) from one call site — no CUDA,
//! no torch.
//!
//! The Metal shaders are CUDA→Metal translations. The translation follows the
//! `mx.fast.metal_kernel` ports in the owner's `mlxports` work
//! (`csm-mlx/csm_mlx/monarch_metal/conv1d_forward.py`, itself a faithful port of
//! `flashfftconv/conv1d/conv1d_bhl.cu`): the CUDA `blockIdx/threadIdx` mapping
//! becomes `thread_position_in_grid`, and the K=3 fast path is preserved. Only the
//! host side differs — MLX's `fast.metal_kernel` auto-generates the signature,
//! whereas candle binds buffers explicitly, so the shader carries `[[buffer(i)]]`
//! attributes and we dispatch with the candle-metal-kernels idiom.
//!
//! ## Operators
//! - [`depthwise_conv1d`] — depthwise (grouped, one filter per channel) causal
//!   conv1d. This is the LFM2 short-conv (`conv_L_cache`) and the FlashFFTConv
//!   short-filter path. **Done + verified** (metal == cpu, 5.96e-8).
//! - [`depthwise_conv1d_stream`] — the same conv with a cache of the prior `K-1`
//!   inputs (a *valid* conv over the cache-prepended stream): one op for prefill
//!   (no cache) and single-step decode (cache), the form LFM2's short-conv decode
//!   needs. **Done + verified**: chunked streaming incl. `T=1` == full sequence (0.0).
//! - [`butterfly_fft_forward`] / [`butterfly_fft_inverse`] — the Monarch butterfly
//!   FFT and its inverse (row-DFT → twiddle → col-DFT, and the mirror), with
//!   [`fft_matrix`]/[`ifft_matrix`]/[`twiddle_factors_fft`]/[`twiddle_factors_ifft`].
//!   The transform is the `M = N·L`-point DFT for a **column-major** input layout
//!   (`tensor[n,l]` ↔ time index `l·N + n`). **Done + verified.**
//! - [`monarch_conv`] — the FlashFFTConv long convolution `IFFT(FFT(u) ⊙ k_f)`
//!   (via [`complex_mul`]). Arbitrary length. **Done + verified**: `monarch_conv ==
//!   direct circular convolution` (9.69e-8), and metal == cpu (2.24e-8) at every stage.
//! - [`fused_fft_conv`] — the **single-dispatch** radix-2 FlashFFTConv path: one Metal
//!   threadgroup per `(batch, channel)` does `rfft → ⊙ k_f → irfft → +u·D` with the
//!   radix-2 FFT in threadgroup memory (no global round-trips). For `fft_size` a
//!   power of two `≤ 1024` (linear conv via `2·seqlen` zero-pad). **Done +
//!   verified**: `== direct linear convolution` (1.19e-7); metal == cpu (2.98e-8).
//!
//! ## Fused tensor-core path (`simdgroup_matrix`)
//!
//! The Monarch transform expressed on Apple's matrix units (the CUDA `wmma` analog):
//! every sub-DFT is an 8×8 `simdgroup_float8x8` GEMM with **fp32 accumulate**, the whole
//! pipeline fused into **one** dispatch with the `[N,L]` intermediate resident in
//! `threadgroup` memory (one threadgroup per `(b,h)`, tiled — full GPU occupancy).
//! - [`butterfly_fft_forward_fused`] — fused forward butterfly (row-DFT→twiddle→col-DFT
//!   in one kernel). **Done + verified**: metal == cpu == un-fused (3.81e-6).
//! - [`monarch_conv_fused`] — the **full** fused conv `IFFT(FFT(u) ⊙ k_f)` in one kernel
//!   (forward + in-kernel `×k_f` + inverse, ping-ponged so col-DFTs never alias their
//!   input). Drop-in for [`monarch_conv`]; **any `N,L`** via edge-tile zero-fill (matrices
//!   padded to mult-of-8 at pack time, ragged boundary handled in-kernel — no caller
//!   padding). **Done + verified**: metal == `monarch_conv` (1.5e-8) incl. non-mult-of-8
//!   dims; == circular convolution (9.7e-8). fp32 (bf16/fp16/gated/padded are follow-ons).
//! - [`warmup`] — pre-compile the fused kernels at init so the realtime path never eats a
//!   first-frame compile.
//!
//! Compiled pipelines are cached **process-wide** (global, thread-safe), so each kernel
//! compiles once and is shared across threads — the compiled kernel vs. per-dispatch
//! instances. See `ARCHITECTURE.md` for the full kernel design and verification story.
//!
//! ## Two precision regimes — faithful vs precise
//!
//! The FlashFFTConv CUDA kernels run in **bf16/f16** (`butterfly_cuda_bf16.cu`:
//! `__nv_bfloat16` DFT matrices, twiddles, and per-butterfly stores via
//! `__float22bfloat162_rn`; only the inner `wmma` matmul accumulates in `float`).
//! That coarse rounding is part of the trained model — the weights were fit *around*
//! it — so there are two correct ports, not one:
//!
//! - **Faithful (bug-for-bug):** [`monarch_conv_bf16`] reproduces that exact dtype
//!   chain (bf16 coeffs + activations + per-stage stores, f32 accumulate) so the
//!   result matches what the network saw in training. candle's `BF16` dtype is
//!   `half::bf16` RNE, identical to CUDA `_rn`, so this needs **no new shaders** —
//!   it runs on CPU and Metal from one path.
//! - **Precise:** [`fused_fft_conv_dd`] / [`complex_mul_dd`] carry the transform in
//!   **double-double** (~f64), below every existing implementation including the
//!   original CUDA. Use when you want the true convolution, not the trained-around one.
//!
//! Measured on the same 256-point circular convolution vs an f64 ground truth:
//! **bf16-faithful 2.69e-1, clean f32 [`monarch_conv`] 7.63e-6, double-double
//! ≤ 1.18e-7** — the bf16 regime is ~35000× coarser than f32, which is the whole
//! point: that is the gap the training compensates for.

mod butterfly;
mod conv1d;
mod dd_complex_mul;
mod dw3;
mod fused_fft_conv;
mod fused_fft_conv_dd;
mod fused_monarch;
mod irfft;
#[cfg(feature = "metal")]
mod metal_util;

pub use butterfly::{
    butterfly_fft_forward, butterfly_fft_inverse, complex_mul, fft_matrix, ifft_matrix,
    monarch_conv, monarch_conv_bf16, twiddle_factors_fft, twiddle_factors_ifft,
};
pub use conv1d::{depthwise_conv1d, depthwise_conv1d_stream, DepthwiseCausalConv1d};
pub use dd_complex_mul::complex_mul_dd;
pub use dw3::depthwise3_causal;
pub use fused_fft_conv::{fused_fft_conv, FusedFftConv};
pub use fused_fft_conv_dd::{fused_fft_conv_dd, FusedFftConvDd};
pub use fused_monarch::{butterfly_fft_forward_fused, monarch_conv_fused, warmup};
pub use irfft::{irfft, irfft_dd, FftNorm};
