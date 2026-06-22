//! Monarch butterfly FFT — **forward** (real → complex), the FFT half of
//! FlashFFTConv's long convolution.
//!
//! An `M = N·L`-point DFT is factored (4-step / Bailey) into three phases over a
//! `[B, H, N, L]` reshape, matching the CUDA `complex_matmul_r2c → twiddle →
//! complex_matmul` and the MLX port in `csm-mlx/.../monarch_metal/butterfly_forward.py`:
//!
//! 1. **row DFT** — length-`L` DFT along the last axis: `Y[..,n,l'] = Σ_k d_f_L[l',k]·x[..,n,k]` (real→complex)
//! 2. **twiddle** — element-wise complex multiply by `twiddles[n,l]`
//! 3. **col DFT** — length-`N` DFT along the `N` axis: `Z[..,n',l] = Σ_k d_f_N[n',k]·Y[..,k,l]` (complex→complex)
//!
//! Complex tensors are f32 with a trailing size-2 axis (`[…, 2]` = real, imag),
//! matching the MLX/CUDA layout. The DFT matrices and twiddles are precomputed
//! inputs (see [`fft_matrix`] / [`twiddle_factors_fft`]) so the convention lives
//! with the caller, exactly as in the reference.
//!
//! Each phase is a candle [`CustomOp2`] carrying a CPU reference and a Metal kernel,
//! so [`butterfly_fft_forward`] runs the exact reference on CPU and the fused
//! shaders on Metal from one call.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

#[cfg(feature = "metal")]
use crate::metal_util;

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("butterfly fft expects contiguous f32 inputs"),
    }
}

// ---------------------------------------------------------------------------
// Phase 1: row DFT (real input → complex output), length L along the last axis.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
const SRC_ROW_DFT: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void butterfly_row_dft_f32(
    device const float* x     [[buffer(0)]],   // [B,H,N,L] real
    device const float* d_f_L [[buffer(1)]],   // [L,L,2] complex
    device float*       out   [[buffer(2)]],   // [B,H,N,L,2] complex
    constant uint& B [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& N [[buffer(5)]], constant uint& L [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * H * N * L;
    if (gid >= total) { return; }
    uint l_out = gid % L;
    uint tmp = gid / L; uint n = tmp % N; uint tmp2 = tmp / N; uint h = tmp2 % H; uint b = tmp2 / H;
    float sr = 0.0f, si = 0.0f;
    uint x_base = ((b * H + h) * N + n) * L;
    for (uint k = 0; k < L; k++) {
        uint df = (l_out * L + k) * 2;
        float xv = x[x_base + k];
        sr = fma(d_f_L[df],     xv, sr);
        si = fma(d_f_L[df + 1], xv, si);
    }
    out[gid * 2] = sr; out[gid * 2 + 1] = si;
}
"#;

struct RowDft;

impl CustomOp2 for RowDft {
    fn name(&self) -> &'static str {
        "butterfly_row_dft"
    }
    fn cpu_fwd(&self, xs: &CpuStorage, xl: &Layout, ds: &CpuStorage, dl: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l) = xl.shape().dims4()?;
        let x = contig_f32(xs, xl)?;
        let df = contig_f32(ds, dl)?; // [L,L,2]
        let mut out = vec![0f32; b * h * n * l * 2];
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    let x_base = ((bi * h + hi) * n + ni) * l;
                    for lo in 0..l {
                        let (mut sr, mut si) = (0f32, 0f32);
                        for k in 0..l {
                            let dr = df[(lo * l + k) * 2];
                            let di = df[(lo * l + k) * 2 + 1];
                            let xv = x[x_base + k];
                            sr += dr * xv;
                            si += di * xv;
                        }
                        let o = (x_base + lo) * 2;
                        out[o] = sr;
                        out[o + 1] = si;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l, 2))))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        ds: &candle_core::MetalStorage,
        dl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (b, h, n, l) = xl.shape().dims4()?;
        metal_util::dispatch_dft(xs, xl, ds, dl, SRC_ROW_DFT, "butterfly_row_dft_f32", b, h, n, l, true)
    }
}

// ---------------------------------------------------------------------------
// Phase 2: twiddle (element-wise complex multiply by twiddles[n,l]).
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
const SRC_TWIDDLE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void butterfly_twiddle_f32(
    device const float* x   [[buffer(0)]],   // [B,H,N,L,2] complex
    device const float* tw  [[buffer(1)]],   // [N,L,2] complex
    device float*       out [[buffer(2)]],   // [B,H,N,L,2] complex
    constant uint& B [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& N [[buffer(5)]], constant uint& L [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * H * N * L;
    if (gid >= total) { return; }
    uint l = gid % L; uint tmp = gid / L; uint n = tmp % N;
    float xr = x[gid * 2], xi = x[gid * 2 + 1];
    uint ti = (n * L + l) * 2;
    float tr = tw[ti], tii = tw[ti + 1];
    out[gid * 2]     = xr * tr - xi * tii;
    out[gid * 2 + 1] = xr * tii + xi * tr;
}
"#;

struct Twiddle;

impl CustomOp2 for Twiddle {
    fn name(&self) -> &'static str {
        "butterfly_twiddle"
    }
    fn cpu_fwd(&self, xs: &CpuStorage, xl: &Layout, ts: &CpuStorage, tl: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        let x = contig_f32(xs, xl)?;
        let tw = contig_f32(ts, tl)?; // [N,L,2]
        let mut out = vec![0f32; b * h * n * l * 2];
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    for li in 0..l {
                        let gid = ((bi * h + hi) * n + ni) * l + li;
                        let xr = x[gid * 2];
                        let xi = x[gid * 2 + 1];
                        let tr = tw[(ni * l + li) * 2];
                        let tii = tw[(ni * l + li) * 2 + 1];
                        out[gid * 2] = xr * tr - xi * tii;
                        out[gid * 2 + 1] = xr * tii + xi * tr;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l, 2))))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        ts: &candle_core::MetalStorage,
        tl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        metal_util::dispatch_dft(xs, xl, ts, tl, SRC_TWIDDLE, "butterfly_twiddle_f32", b, h, n, l, true)
    }
}

// ---------------------------------------------------------------------------
// Phase 3: col DFT (complex → complex), length N along the N axis.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
const SRC_COL_DFT: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void butterfly_col_dft_f32(
    device const float* x     [[buffer(0)]],   // [B,H,N,L,2] complex
    device const float* d_f_N [[buffer(1)]],   // [N,N,2] complex
    device float*       out   [[buffer(2)]],   // [B,H,N,L,2] complex
    constant uint& B [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& N [[buffer(5)]], constant uint& L [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * H * N * L;
    if (gid >= total) { return; }
    uint l = gid % L; uint tmp = gid / L; uint n_out = tmp % N; uint tmp2 = tmp / N; uint h = tmp2 % H; uint b = tmp2 / H;
    float sr = 0.0f, si = 0.0f;
    for (uint k = 0; k < N; k++) {
        uint df = (n_out * N + k) * 2;
        float dr = d_f_N[df], di = d_f_N[df + 1];
        uint xi = (((b * H + h) * N + k) * L + l) * 2;
        float xr = x[xi], xii = x[xi + 1];
        sr += dr * xr - di * xii;
        si += dr * xii + di * xr;
    }
    out[gid * 2] = sr; out[gid * 2 + 1] = si;
}
"#;

struct ColDft;

impl CustomOp2 for ColDft {
    fn name(&self) -> &'static str {
        "butterfly_col_dft"
    }
    fn cpu_fwd(&self, xs: &CpuStorage, xl: &Layout, ds: &CpuStorage, dl: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        let x = contig_f32(xs, xl)?;
        let df = contig_f32(ds, dl)?; // [N,N,2]
        let mut out = vec![0f32; b * h * n * l * 2];
        for bi in 0..b {
            for hi in 0..h {
                for no in 0..n {
                    for li in 0..l {
                        let (mut sr, mut si) = (0f32, 0f32);
                        for k in 0..n {
                            let dr = df[(no * n + k) * 2];
                            let di = df[(no * n + k) * 2 + 1];
                            let xi = (((bi * h + hi) * n + k) * l + li) * 2;
                            let xr = x[xi];
                            let xii = x[xi + 1];
                            sr += dr * xr - di * xii;
                            si += dr * xii + di * xr;
                        }
                        let o = (((bi * h + hi) * n + no) * l + li) * 2;
                        out[o] = sr;
                        out[o + 1] = si;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l, 2))))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        ds: &candle_core::MetalStorage,
        dl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        metal_util::dispatch_dft(xs, xl, ds, dl, SRC_COL_DFT, "butterfly_col_dft_f32", b, h, n, l, true)
    }
}

/// Forward Monarch butterfly FFT of `x` `[B, H, N, L]` (real) → `[B, H, N, L, 2]`
/// (complex). `d_f_n` `[N,N,2]`, `d_f_l` `[L,L,2]` are DFT matrices and `twiddles`
/// `[N,L,2]` the twiddle factors (see [`fft_matrix`] / [`twiddle_factors_fft`]).
pub fn butterfly_fft_forward(x: &Tensor, d_f_n: &Tensor, d_f_l: &Tensor, twiddles: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let d_f_l = d_f_l.contiguous()?;
    let d_f_n = d_f_n.contiguous()?;
    let twiddles = twiddles.contiguous()?;
    let y = x.apply_op2(&d_f_l, RowDft)?; // [B,H,N,L,2]
    let y = y.apply_op2(&twiddles, Twiddle)?;
    y.apply_op2(&d_f_n, ColDft)
}

/// DFT matrix `[n, n, 2]`: `d[a,b] = exp(-2πi·a·b/n)` (real, imag).
pub fn fft_matrix(n: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * n * 2];
    for a in 0..n {
        for b in 0..n {
            let ang = -2.0 * std::f64::consts::PI * (a as f64) * (b as f64) / (n as f64);
            v[(a * n + b) * 2] = ang.cos() as f32;
            v[(a * n + b) * 2 + 1] = ang.sin() as f32;
        }
    }
    Tensor::from_vec(v, (n, n, 2), device)
}

/// Forward twiddle factors `[n, m, 2]`: `tw[a,b] = exp(-2πi·a·b/(n·m))`.
pub fn twiddle_factors_fft(n: usize, m: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mn = (n * m) as f64;
    let mut v = vec![0f32; n * m * 2];
    for a in 0..n {
        for b in 0..m {
            let ang = -2.0 * std::f64::consts::PI * (a as f64) * (b as f64) / mn;
            v[(a * m + b) * 2] = ang.cos() as f32;
            v[(a * m + b) * 2 + 1] = ang.sin() as f32;
        }
    }
    Tensor::from_vec(v, (n, m, 2), device)
}

// ---------------------------------------------------------------------------
// Inverse phase: row IDFT (complex → real), length L, scaled by 1/(N·L).
// (Col-IDFT reuses `ColDft` with an IDFT matrix; conj-twiddle reuses `Twiddle`
// with ifft twiddles — only the complex→real row phase is new.)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
const SRC_ROW_IDFT_REAL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void butterfly_row_idft_real_f32(
    device const float* x      [[buffer(0)]],   // [B,H,N,L,2] complex
    device const float* id_f_L [[buffer(1)]],   // [L,L,2] complex (IDFT)
    device float*       out    [[buffer(2)]],   // [B,H,N,L] real
    constant uint& B [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& N [[buffer(5)]], constant uint& L [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = B * H * N * L;
    if (gid >= total) { return; }
    uint l_out = gid % L;
    uint tmp = gid / L; uint n = tmp % N; uint tmp2 = tmp / N; uint h = tmp2 % H; uint b = tmp2 / H;
    uint x_base = ((b * H + h) * N + n) * L;
    float sr = 0.0f;
    for (uint k = 0; k < L; k++) {
        uint df = (l_out * L + k) * 2;
        float dr = id_f_L[df], di = id_f_L[df + 1];
        uint xi = (x_base + k) * 2;
        sr += dr * x[xi] - di * x[xi + 1];   // real part of the complex matmul
    }
    out[gid] = sr / (float)(N * L);
}
"#;

struct RowIDftReal;

impl CustomOp2 for RowIDftReal {
    fn name(&self) -> &'static str {
        "butterfly_row_idft_real"
    }
    fn cpu_fwd(&self, xs: &CpuStorage, xl: &Layout, ds: &CpuStorage, dl: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        let x = contig_f32(xs, xl)?;
        let df = contig_f32(ds, dl)?; // [L,L,2]
        let scale = 1.0f32 / (n * l) as f32;
        let mut out = vec![0f32; b * h * n * l];
        for bi in 0..b {
            for hi in 0..h {
                for ni in 0..n {
                    let x_base = ((bi * h + hi) * n + ni) * l;
                    for lo in 0..l {
                        let mut sr = 0f32;
                        for k in 0..l {
                            let dr = df[(lo * l + k) * 2];
                            let di = df[(lo * l + k) * 2 + 1];
                            let xi = (x_base + k) * 2;
                            sr += dr * x[xi] - di * x[xi + 1];
                        }
                        out[x_base + lo] = sr * scale;
                    }
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l))))
    }
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        ds: &candle_core::MetalStorage,
        dl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let (b, h, n, l, _two) = xl.shape().dims5()?;
        metal_util::dispatch_dft(xs, xl, ds, dl, SRC_ROW_IDFT_REAL, "butterfly_row_idft_real_f32", b, h, n, l, false)
    }
}

/// IDFT matrix `[n, n, 2]`: `id[a,b] = exp(+2πi·a·b/n)` (conjugate of [`fft_matrix`]).
pub fn ifft_matrix(n: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut v = vec![0f32; n * n * 2];
    for a in 0..n {
        for b in 0..n {
            let ang = 2.0 * std::f64::consts::PI * (a as f64) * (b as f64) / (n as f64);
            v[(a * n + b) * 2] = ang.cos() as f32;
            v[(a * n + b) * 2 + 1] = ang.sin() as f32;
        }
    }
    Tensor::from_vec(v, (n, n, 2), device)
}

/// Inverse twiddle factors `[n, m, 2]`: `tw[a,b] = exp(+2πi·a·b/(n·m))` (conj of
/// [`twiddle_factors_fft`]).
pub fn twiddle_factors_ifft(n: usize, m: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mn = (n * m) as f64;
    let mut v = vec![0f32; n * m * 2];
    for a in 0..n {
        for b in 0..m {
            let ang = 2.0 * std::f64::consts::PI * (a as f64) * (b as f64) / mn;
            v[(a * m + b) * 2] = ang.cos() as f32;
            v[(a * m + b) * 2 + 1] = ang.sin() as f32;
        }
    }
    Tensor::from_vec(v, (n, m, 2), device)
}

/// Element-wise complex multiply of `a` and `b` (trailing `[…, 2]` axis),
/// broadcasting `b` over `a` (so a `[H,N,L,2]` filter applies to a `[B,H,N,L,2]`
/// signal). Plain candle ops — runs on CPU and Metal.
pub fn complex_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let d = a.rank() - 1;
    let (ar, ai) = (a.narrow(d, 0, 1)?, a.narrow(d, 1, 1)?);
    let (br, bi) = (b.narrow(d, 0, 1)?, b.narrow(d, 1, 1)?);
    let real = (ar.broadcast_mul(&br)? - ai.broadcast_mul(&bi)?)?;
    let imag = (ar.broadcast_mul(&bi)? + ai.broadcast_mul(&br)?)?;
    Tensor::cat(&[&real, &imag], d)
}

/// Inverse Monarch butterfly FFT of `z` `[B,H,N,L,2]` (complex) → `[B,H,N,L]`
/// (real). Mirror of [`butterfly_fft_forward`]: col-IDFT → conj-twiddle →
/// row-IDFT (real, `1/(N·L)`-scaled). `id_f_n`/`id_f_l` from [`ifft_matrix`],
/// `ifft_twiddles` from [`twiddle_factors_ifft`].
pub fn butterfly_fft_inverse(z: &Tensor, id_f_n: &Tensor, id_f_l: &Tensor, ifft_twiddles: &Tensor) -> Result<Tensor> {
    let z = z.contiguous()?;
    let id_f_n = id_f_n.contiguous()?;
    let id_f_l = id_f_l.contiguous()?;
    let ifft_twiddles = ifft_twiddles.contiguous()?;
    let y = z.apply_op2(&id_f_n, ColDft)?; // col IDFT (matmul along N with the IDFT matrix)
    let y = y.apply_op2(&ifft_twiddles, Twiddle)?; // conjugate twiddle
    y.apply_op2(&id_f_l, RowIDftReal) // row IDFT, real part, 1/(N·L)
}

/// FlashFFTConv long convolution via the Monarch FFT: `out = IFFT(FFT(u) ⊙ k_f)`.
///
/// `u` `[B,H,N,L]` real; `k_f` `[…,N,L,2]` is the filter's **forward FFT** (in the
/// same Monarch order, e.g. `butterfly_fft_forward(kernel, …)`), broadcast over the
/// batch. By the convolution theorem this equals the length-`N·L` circular
/// convolution of `u` and the kernel (the Monarch output ordering cancels between
/// the forward and inverse transforms).
#[allow(clippy::too_many_arguments)]
pub fn monarch_conv(
    u: &Tensor,
    k_f: &Tensor,
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
    id_f_n: &Tensor,
    id_f_l: &Tensor,
    ifft_twiddles: &Tensor,
) -> Result<Tensor> {
    let uf = butterfly_fft_forward(u, d_f_n, d_f_l, twiddles)?;
    let prod = complex_mul(&uf, k_f)?;
    butterfly_fft_inverse(&prod, id_f_n, id_f_l, ifft_twiddles)
}

/// Round `t` to bfloat16 and back to f32 — the exact rounding the FlashFFTConv bf16
/// CUDA kernel applies whenever it stores an intermediate
/// (`__float22bfloat162_rn`). candle's `BF16` dtype is `half::bf16`
/// (round-to-nearest-even), so this round-trip matches CUDA's `_rn` bit-for-bit and
/// runs natively on both CPU and Metal.
fn bf16_round(t: &Tensor) -> Result<Tensor> {
    t.to_dtype(candle_core::DType::BF16)?.to_dtype(candle_core::DType::F32)
}

/// **Faithful** FlashFFTConv long convolution — the same [`monarch_conv`] math run
/// in the exact dtype regime of the bf16 CUDA kernels
/// (`csrc/flashfftconv/butterfly/butterfly_cuda_bf16.cu`), i.e. the numerics the
/// model was actually trained around.
///
/// Every value those kernels keep in `__nv_bfloat16` is rounded to bf16 here:
/// - the DFT matrices `d_f_*` and the twiddle factors (loaded as `__nv_bfloat16`),
/// - the input `u` and the filter spectrum `k_f` (bf16 activations),
/// - each butterfly's stored output (`__float22bfloat162_rn` after the wmma matmul
///   + twiddle multiply).
///
/// Inside a butterfly the DFT matmul and the twiddle multiply accumulate in **f32**
/// (the `wmma::fragment<accumulator, …, float>` is float, and the twiddle uses the
/// f32 accumulator), and the row-DFT result is held in f32 *through* the twiddle
/// before the single bf16 store — so the rounding count matches CUDA exactly (one
/// bf16 store per butterfly pass, not one per phase). The result tracks the bf16
/// regime (~1e-2 relative), **not** f32 or double-double — which is the point: the
/// trained weights expect this rounding, so this is the bug-for-bug reference to
/// compare against [`crate::fused_fft_conv_dd`] (double-double, ~f64) and the clean
/// f32 [`monarch_conv`].
#[allow(clippy::too_many_arguments)]
pub fn monarch_conv_bf16(
    u: &Tensor,
    k_f: &Tensor,
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
    id_f_n: &Tensor,
    id_f_l: &Tensor,
    ifft_twiddles: &Tensor,
) -> Result<Tensor> {
    // bf16 storage of every coefficient + activation (CUDA holds these as bf16).
    let u = bf16_round(&u.contiguous()?)?;
    let k_f = bf16_round(&k_f.contiguous()?)?;
    let d_f_l = bf16_round(&d_f_l.contiguous()?)?;
    let d_f_n = bf16_round(&d_f_n.contiguous()?)?;
    let twiddles = bf16_round(&twiddles.contiguous()?)?;
    let id_f_n = bf16_round(&id_f_n.contiguous()?)?;
    let id_f_l = bf16_round(&id_f_l.contiguous()?)?;
    let ifft_twiddles = bf16_round(&ifft_twiddles.contiguous()?)?;

    // Forward butterfly 1: row DFT (f32 accumulate), held in f32 through the twiddle,
    // then a single bf16 store. Butterfly 2: col DFT, bf16 store.
    let y = u.apply_op2(&d_f_l, RowDft)?;
    let y = y.apply_op2(&twiddles, Twiddle)?;
    let y = bf16_round(&y)?;
    let y = y.apply_op2(&d_f_n, ColDft)?;
    let y = bf16_round(&y)?;
    // Frequency-domain multiply by the filter spectrum, stored bf16.
    let prod = bf16_round(&complex_mul(&y, &k_f)?)?;
    // Inverse butterfly 1: col IDFT (f32 accumulate) through the conj-twiddle, bf16
    // store. Inverse butterfly 2: row IDFT (real, 1/(N·L)), final bf16 store.
    let y = prod.apply_op2(&id_f_n, ColDft)?;
    let y = y.apply_op2(&ifft_twiddles, Twiddle)?;
    let y = bf16_round(&y)?;
    let y = y.apply_op2(&id_f_l, RowIDftReal)?;
    bf16_round(&y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Naive length-L DFT along the last axis (real input), for the absolute check.
    fn naive_row_dft(x: &[f32], b: usize, h: usize, n: usize, l: usize) -> Vec<f32> {
        let mut out = vec![0f32; b * h * n * l * 2];
        for idx in 0..(b * h * n) {
            let base = idx * l;
            for lo in 0..l {
                let (mut sr, mut si) = (0f64, 0f64);
                for k in 0..l {
                    let ang = -2.0 * std::f64::consts::PI * (lo as f64) * (k as f64) / (l as f64);
                    sr += ang.cos() * x[base + k] as f64;
                    si += ang.sin() * x[base + k] as f64;
                }
                out[(base + lo) * 2] = sr as f32;
                out[(base + lo) * 2 + 1] = si as f32;
            }
        }
        out
    }

    #[test]
    fn row_dft_matches_naive() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (1usize, 2, 3, 8);
        let x: Vec<f32> = (0..b * h * n * l).map(|i| (i as f32 * 0.13).sin()).collect();
        let xt = Tensor::from_vec(x.clone(), (b, h, n, l), &dev).unwrap();
        let d_f_l = fft_matrix(l, &dev).unwrap();
        let got: Vec<f32> = xt.apply_op2(&d_f_l, RowDft).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let exp = naive_row_dft(&x, b, h, n, l);
        let maxd = got.iter().zip(exp.iter()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-4, "row DFT vs naive max diff {maxd}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn forward_metal_matches_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, h, n, l) = (2usize, 3, 16, 16);
        let x: Vec<f32> = (0..b * h * n * l).map(|i| ((i * 11 % 17) as f32 * 0.05) - 0.4).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let xt = Tensor::from_vec(x.clone(), (b, h, n, l), dev).unwrap();
            let d_f_l = fft_matrix(l, dev).unwrap();
            let d_f_n = fft_matrix(n, dev).unwrap();
            let tw = twiddle_factors_fft(n, l, dev).unwrap();
            butterfly_fft_forward(&xt, &d_f_n, &d_f_l, &tw)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        assert_eq!(cpu.len(), met.len());
        let maxd = cpu.iter().zip(met.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-4, "forward FFT metal vs cpu max diff {maxd}");
        eprintln!("butterfly fft forward: metal == cpu, max diff {maxd:.2e}");
    }

    // direct length-M circular convolution: y[n] = Σ_j u[j]·k[(n−j) mod M].
    fn circular_conv(u: &[f32], k: &[f32]) -> Vec<f32> {
        let m = u.len();
        let mut y = vec![0f32; m];
        for n in 0..m {
            let mut acc = 0f64;
            for j in 0..m {
                acc += u[j] as f64 * k[(n + m - j) % m] as f64;
            }
            y[n] = acc as f32;
        }
        y
    }

    #[test]
    fn inverse_undoes_forward() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (1usize, 1, 4, 4);
        let x: Vec<f32> = (0..b * h * n * l).map(|i| (i as f32 * 0.3).cos()).collect();
        let xt = Tensor::from_vec(x.clone(), (b, h, n, l), &dev).unwrap();
        let z = butterfly_fft_forward(
            &xt,
            &fft_matrix(n, &dev).unwrap(),
            &fft_matrix(l, &dev).unwrap(),
            &twiddle_factors_fft(n, l, &dev).unwrap(),
        )
        .unwrap();
        let xr = butterfly_fft_inverse(
            &z,
            &ifft_matrix(n, &dev).unwrap(),
            &ifft_matrix(l, &dev).unwrap(),
            &twiddle_factors_ifft(n, l, &dev).unwrap(),
        )
        .unwrap();
        let got: Vec<f32> = xr.flatten_all().unwrap().to_vec1().unwrap();
        let maxd = got.iter().zip(x.iter()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-4, "ifft(fft(x)) != x, max diff {maxd}");
        eprintln!("ifft(fft(x)) == x, max diff {maxd:.2e}");
    }

    #[test]
    fn monarch_conv_matches_circular() {
        let dev = Device::Cpu;
        let (n, l) = (4usize, 4);
        let m = n * l;
        // Time-domain signals (length M).
        let u_time: Vec<f32> = (0..m).map(|i| (i as f32 * 0.21).sin()).collect();
        let k_time: Vec<f32> = (0..m).map(|i| (i as f32 * 0.11 + 1.0).cos() * 0.5).collect();
        // The Monarch transform reads input column-major: tensor[ni,li] (row-major
        // ni*L+li) holds time index li*N + ni.
        let lay = |t: &[f32]| -> Vec<f32> {
            let mut v = vec![0f32; m];
            for ni in 0..n {
                for li in 0..l {
                    v[ni * l + li] = t[li * n + ni];
                }
            }
            v
        };
        let ut = Tensor::from_vec(lay(&u_time), (1, 1, n, l), &dev).unwrap();
        let kt = Tensor::from_vec(lay(&k_time), (1, 1, n, l), &dev).unwrap();
        let (dfn, dfl, tw) = (fft_matrix(n, &dev).unwrap(), fft_matrix(l, &dev).unwrap(), twiddle_factors_fft(n, l, &dev).unwrap());
        let (idfn, idfl, itw) = (ifft_matrix(n, &dev).unwrap(), ifft_matrix(l, &dev).unwrap(), twiddle_factors_ifft(n, l, &dev).unwrap());
        let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();
        let y = monarch_conv(&ut, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap();
        let out_flat: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // Read back the same column-major convention: out[ni,li] = time index li*N+ni.
        let mut y_time = vec![0f32; m];
        for ni in 0..n {
            for li in 0..l {
                y_time[li * n + ni] = out_flat[ni * l + li];
            }
        }
        let exp = circular_conv(&u_time, &k_time);
        let maxd = y_time.iter().zip(exp.iter()).fold(0f32, |mm, (a, e)| mm.max((a - e).abs()));
        assert!(maxd < 1e-3, "monarch conv != circular conv, max diff {maxd}");
        eprintln!("monarch_conv == circular conv (col-major time order), max diff {maxd:.2e}");
    }

    // The two-version measurement: the SAME 256-point circular convolution computed
    // in the faithful bf16 CUDA regime vs clean f32, each scored against the f64
    // ground truth (`circular_conv`). Shows how far the trained-around bf16 numerics
    // sit from the true convolution — the gap the double-double path closes.
    #[test]
    fn regimes_bf16_vs_f32_vs_f64() {
        let dev = Device::Cpu;
        let (n, l) = (16usize, 16);
        let m = n * l; // 256-point circular conv
        let u_time: Vec<f32> = (0..m).map(|i| (i as f32 * 0.21).sin() * 2.0).collect();
        let k_time: Vec<f32> = (0..m).map(|i| (i as f32 * 0.037 + 0.5).cos()).collect();
        // Monarch reads input column-major: tensor[ni,li] holds time index li*N+ni.
        let lay = |t: &[f32]| -> Vec<f32> {
            let mut v = vec![0f32; m];
            for ni in 0..n {
                for li in 0..l {
                    v[ni * l + li] = t[li * n + ni];
                }
            }
            v
        };
        let ut = Tensor::from_vec(lay(&u_time), (1, 1, n, l), &dev).unwrap();
        let kt = Tensor::from_vec(lay(&k_time), (1, 1, n, l), &dev).unwrap();
        let (dfn, dfl, tw) = (fft_matrix(n, &dev).unwrap(), fft_matrix(l, &dev).unwrap(), twiddle_factors_fft(n, l, &dev).unwrap());
        let (idfn, idfl, itw) = (ifft_matrix(n, &dev).unwrap(), ifft_matrix(l, &dev).unwrap(), twiddle_factors_ifft(n, l, &dev).unwrap());
        let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();

        let read = |y: &Tensor| -> Vec<f32> {
            let f: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
            let mut o = vec![0f32; m];
            for ni in 0..n {
                for li in 0..l {
                    o[li * n + ni] = f[ni * l + li];
                }
            }
            o
        };
        let y_f32 = read(&monarch_conv(&ut, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap());
        let y_bf16 = read(&monarch_conv_bf16(&ut, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap());

        let exp = circular_conv(&u_time, &k_time); // f64 ground truth
        let err = |y: &[f32]| y.iter().zip(exp.iter()).fold(0f32, |mx, (a, e)| mx.max((a - e).abs()));
        let (e_f32, e_bf16) = (err(&y_f32), err(&y_bf16));
        eprintln!(
            "circular conv (M={m}) vs f64 truth:  f32 {e_f32:.3e}   bf16-faithful {e_bf16:.3e}   (bf16 is {:.0}x the f32 error)",
            e_bf16 / e_f32.max(f32::MIN_POSITIVE)
        );
        // The faithful bf16 regime is far coarser than f32 against the true conv —
        // that coarseness is exactly what the trained weights were fit to.
        assert!(e_f32 < e_bf16, "clean f32 ({e_f32:e}) should be closer to truth than bf16 ({e_bf16:e})");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn monarch_conv_metal_matches_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, h, n, l) = (2usize, 2, 16, 16);
        let u: Vec<f32> = (0..b * h * n * l).map(|i| ((i * 13 % 19) as f32 * 0.05) - 0.4).collect();
        let kf: Vec<f32> = (0..b * h * n * l * 2).map(|i| ((i * 7 % 11) as f32 * 0.03) - 0.15).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let ut = Tensor::from_vec(u.clone(), (b, h, n, l), dev).unwrap();
            let k_f = Tensor::from_vec(kf.clone(), (b, h, n, l, 2), dev).unwrap();
            let (dfn, dfl, tw) = (fft_matrix(n, dev).unwrap(), fft_matrix(l, dev).unwrap(), twiddle_factors_fft(n, l, dev).unwrap());
            let (idfn, idfl, itw) = (ifft_matrix(n, dev).unwrap(), ifft_matrix(l, dev).unwrap(), twiddle_factors_ifft(n, l, dev).unwrap());
            monarch_conv(&ut, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        let maxd = cpu.iter().zip(met.iter()).fold(0f32, |mm, (a, b)| mm.max((a - b).abs()));
        assert!(maxd < 1e-4, "monarch conv metal vs cpu max diff {maxd}");
        eprintln!("monarch_conv: metal == cpu, max diff {maxd:.2e}");
    }
}
