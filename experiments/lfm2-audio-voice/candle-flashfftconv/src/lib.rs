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
//! - [`butterfly_fft_forward`] / [`butterfly_fft_inverse`] — the Monarch butterfly
//!   FFT and its inverse (row-DFT → twiddle → col-DFT, and the mirror), with
//!   [`fft_matrix`]/[`ifft_matrix`]/[`twiddle_factors_fft`]/[`twiddle_factors_ifft`].
//!   The transform is the `M = N·L`-point DFT for a **column-major** input layout
//!   (`tensor[n,l]` ↔ time index `l·N + n`). **Done + verified.**
//! - [`monarch_conv`] — the FlashFFTConv long convolution `IFFT(FFT(u) ⊙ k_f)`
//!   (via [`complex_mul`]). Arbitrary length. **Done + verified**: `monarch_conv ==
//!   direct circular convolution` (9.69e-8), and metal == cpu (2.24e-8) at every stage.
//! - [`fused_fft_conv`] — the **single-dispatch** FlashFFTConv path: one Metal
//!   threadgroup per `(batch, channel)` does `rfft → ⊙ k_f → irfft → +u·D` with the
//!   radix-2 FFT in threadgroup memory (no global round-trips). For `fft_size` a
//!   power of two `≤ 1024` (linear conv via `2·seqlen` zero-pad). **Done +
//!   verified**: `== direct linear convolution` (1.19e-7); metal == cpu (2.98e-8).
//! - bf16 variants (the LFM2 Metal runtime dtype) and wiring into liquid-audio-rs
//!   are the remaining steps; the f32 kernels are the verified reference.

mod butterfly;
mod conv1d;
mod dd_complex_mul;
mod dw3;
mod fused_fft_conv;
mod fused_fft_conv_dd;
#[cfg(feature = "metal")]
mod metal_util;

pub use butterfly::{
    butterfly_fft_forward, butterfly_fft_inverse, complex_mul, fft_matrix, ifft_matrix, monarch_conv,
    twiddle_factors_fft, twiddle_factors_ifft,
};
pub use conv1d::{depthwise_conv1d, DepthwiseCausalConv1d};
pub use dd_complex_mul::complex_mul_dd;
pub use dw3::depthwise3_causal;
pub use fused_fft_conv::{fused_fft_conv, FusedFftConv};
pub use fused_fft_conv_dd::{fused_fft_conv_dd, FusedFftConvDd};
