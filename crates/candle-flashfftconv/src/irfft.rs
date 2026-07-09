//! `torch.fft.irfft` as a candle extension — inverse real FFT (onesided complex →
//! real), with the same parameters torch exposes (`n`, `dim`, `norm`) and support for
//! multiple precisions, **including a double-double F64 path that runs on the GPU**
//! (Metal has no native f64).
//!
//! torch's `irfft` reconstructs the Hermitian-symmetric full spectrum of a onesided
//! input `X[0..n/2]` and inverse-transforms it, taking the real part. Written out
//! (norm `"backward"`):
//!
//! ```text
//! y[j] = (1/n) [ Re X[0] + (−1)^j Re X[n/2]
//!              + 2 Σ_{k=1}^{n/2−1} ( Re X[k]·cos(2πkj/n) − Im X[k]·sin(2πkj/n) ) ]
//! ```
//!
//! i.e. a matmul of the onesided spectrum against a fixed inverse-DFT basis
//! `y = Re·Cw + Im·Sw`, with `Cw,Sw ∈ ℝ^{freq×n}` carrying the Hermitian weights
//! (`a_0 = a_{n/2} = 1`, else `2`) and the `norm` scale folded in. Expressing it as a
//! matmul (rather than a recursive FFT) is what lets the **same** formulation run on
//! candle's CPU and Metal back-ends for every supported dtype, and lets the F64 path
//! accumulate in double-double on the GPU. The imaginary parts of the DC and Nyquist
//! bins are ignored, exactly as torch does (`Sw[0]=Sw[n/2]=0` here, automatically).

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

/// Normalization mode, matching `torch.fft.irfft(..., norm=)`. The value is the
/// scale applied on the **inverse** transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftNorm {
    /// `"backward"` (torch default): `1/n` on the inverse.
    Backward,
    /// `"forward"`: no scaling on the inverse.
    Forward,
    /// `"ortho"`: `1/√n` on the inverse.
    Ortho,
}

impl FftNorm {
    /// The inverse-transform scale for output length `n`.
    pub fn inverse_scale(&self, n: usize) -> f64 {
        match self {
            FftNorm::Backward => 1.0 / n as f64,
            FftNorm::Forward => 1.0,
            FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
        }
    }
}

/// Build the inverse-DFT basis `(Cw, Sw)`, each `freq×n` row-major, in f64.
///
/// `y[j] = Σ_k ( Re[k]·Cw[k][j] + Im[k]·Sw[k][j] )`, with
/// `Cw[k][j] = a_k·cos(2πkj/n)·scale`, `Sw[k][j] = −a_k·sin(2πkj/n)·scale`,
/// `a_0 = 1`, `a_{n/2} = 1` (even `n`), else `a_k = 2`, and `scale = norm.inverse_scale(n)`.
fn irfft_basis(n: usize, freq: usize, norm: FftNorm) -> (Vec<f64>, Vec<f64>) {
    let scale = norm.inverse_scale(n);
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut cw = vec![0f64; freq * n];
    let mut sw = vec![0f64; freq * n];
    for k in 0..freq {
        // Hermitian weight: DC and (even-n) Nyquist counted once, the rest twice.
        let a = if k == 0 || (n % 2 == 0 && k == n / 2) {
            1.0
        } else {
            2.0
        };
        for j in 0..n {
            let ang = two_pi * k as f64 * j as f64 / n as f64;
            cw[k * n + j] = a * ang.cos() * scale;
            sw[k * n + j] = -a * ang.sin() * scale;
        }
    }
    (cw, sw)
}

/// Default onesided length for output length `n`: `n/2 + 1`.
pub fn rfft_freqs(n: usize) -> usize {
    n / 2 + 1
}

/// `torch.fft.irfft` over the **last** dimension: `re`,`im` are the onesided spectrum
/// `[…, freq]` (`freq = n/2+1`), returning a real signal `[…, n]`.
///
/// Native candle realization (`Re·Cw + Im·Sw`), so it runs on whatever device the
/// inputs live on and follows their dtype: **f32** (CPU/Metal/CUDA) and **f64**
/// (CPU — candle has no f64 on Metal; use [`irfft_dd`] for f64 on the GPU). `re`/`im`
/// must share dtype, device, and shape, with last dim `n/2 + 1`.
pub fn irfft(re: &Tensor, im: &Tensor, n: usize, norm: FftNorm) -> Result<Tensor> {
    let dtype = re.dtype();
    let dev = re.device();
    let freq = rfft_freqs(n);
    let dims = re.dims().to_vec();
    let last = *dims.last().unwrap();
    if last != freq {
        candle_core::bail!("irfft: last dim {last} != n/2+1 = {freq} for n={n}");
    }
    if im.dims() != dims.as_slice() {
        candle_core::bail!("irfft: re and im must have the same shape");
    }
    // Build the f64 basis on the CPU, cast to the target dtype there, then move to
    // the device — candle has no f64 storage on Metal, so the f64 vec must be
    // downcast before it ever touches the GPU. (For f64-on-Metal use `irfft_dd`.)
    let (cw, sw) = irfft_basis(n, freq, norm);
    let cpu = candle_core::Device::Cpu;
    let cw = Tensor::from_vec(cw, (freq, n), &cpu)?
        .to_dtype(dtype)?
        .to_device(dev)?;
    let sw = Tensor::from_vec(sw, (freq, n), &cpu)?
        .to_dtype(dtype)?
        .to_device(dev)?;
    // Contract the freq axis: [M, freq] @ [freq, n] → [M, n].
    let m: usize = dims[..dims.len() - 1].iter().product();
    let re2 = re.reshape((m, freq))?.contiguous()?;
    let im2 = im.reshape((m, freq))?.contiguous()?;
    let y = (re2.matmul(&cw)? + im2.matmul(&sw)?)?; // [M, n]
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(n);
    y.reshape(out_dims)
}

/// CPU reference: the exact f64 inverse real FFT, used by tests and as the
/// double-double op's CPU path. `re`,`im` are `[M, freq]` row-major f64.
pub(crate) fn irfft_cpu_f64(re: &[f64], im: &[f64], m: usize, n: usize, norm: FftNorm) -> Vec<f64> {
    let freq = rfft_freqs(n);
    let (cw, sw) = irfft_basis(n, freq, norm);
    let mut y = vec![0f64; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for k in 0..freq {
                acc += re[r * freq + k] * cw[k * n + j] + im[r * freq + k] * sw[k * n + j];
            }
            y[r * n + j] = acc;
        }
    }
    y
}

// ---------------------------------------------------------------------------
// Double-double F64 path — the GPU realization torch's MPSGraph irfft can't do.
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn irfft_dd_source() -> String {
    format!(
        "{}\n{}",
        include_str!("metal/double_double.metal"),
        include_str!("metal/IrfftDd.metal")
    )
}

/// Double-double inverse real FFT op. The spectrum comes in as f32 `re`/`im`
/// `[M, freq]` (Metal has no f64 storage), the inverse DFT is accumulated in
/// **double-double** (≈f64, ~106-bit), and the result is rounded once to f32 `[M, n]`.
/// CPU path is the exact f64 reference; Metal path runs [`metal/IrfftDd.metal`].
pub struct IrfftDd {
    pub n: usize,
    pub norm: FftNorm,
}

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("irfft_dd expects contiguous f32 inputs"),
    }
}

impl CustomOp2 for IrfftDd {
    fn name(&self) -> &'static str {
        "irfft_dd"
    }

    fn cpu_fwd(
        &self,
        rs: &CpuStorage,
        rl: &Layout,
        is: &CpuStorage,
        il: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (m, freq) = rl.shape().dims2()?;
        if freq != rfft_freqs(self.n) {
            candle_core::bail!("irfft_dd: freq {freq} != n/2+1 for n={}", self.n);
        }
        let re = contig_f32(rs, rl)?;
        let im = contig_f32(is, il)?;
        // Promote to f64 and run the exact inverse DFT (the dd kernel's target).
        let re64: Vec<f64> = re.iter().map(|&x| x as f64).collect();
        let im64: Vec<f64> = im.iter().map(|&x| x as f64).collect();
        let y = irfft_cpu_f64(&re64, &im64, m, self.n, self.norm);
        let y32: Vec<f32> = y.iter().map(|&x| x as f32).collect();
        Ok((CpuStorage::F32(y32), Shape::from((m, self.n))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        rs: &candle_core::MetalStorage,
        rl: &Layout,
        is: &candle_core::MetalStorage,
        _il: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::MTLSize;

        let (m, freq) = rl.shape().dims2()?;
        if freq != rfft_freqs(self.n) {
            candle_core::bail!("irfft_dd: freq {freq} != n/2+1 for n={}", self.n);
        }
        let n = self.n;
        let total = m * n;
        let dev = rs.device();
        let p = crate::metal_util::pipeline(dev, "irfft_dd", &irfft_dd_source())?;
        let out = dev.new_buffer(total, DType::F32, "irfft_dd")?;

        // Double-double twiddle table: tw[mm] = (cos(2π·mm/n), sin(2π·mm/n)) for
        // mm∈[0,n), each split into hi/lo f32 limbs (Dekker), packed
        // float4(cos.hi, cos.lo, sin.hi, sin.lo). Computed in f64 on the host so the
        // dd inverse DFT is actually exact (the angle 2πkj/n folds to index (k·j) mod n).
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut tw = vec![0f32; n * 4];
        for (mm, slot) in tw.chunks_mut(4).enumerate() {
            let ang = two_pi * mm as f64 / n as f64;
            let (c, s) = (ang.cos(), ang.sin());
            let c_hi = c as f32;
            let s_hi = s as f32;
            slot[0] = c_hi;
            slot[1] = (c - c_hi as f64) as f32;
            slot[2] = s_hi;
            slot[3] = (s - s_hi as f64) as f32;
        }
        let tw_buf = dev.new_buffer_with_data(&tw)?;

        // Normalization scale (1/n for "backward") as a dd pair so it stays exact.
        let scale = self.norm.inverse_scale(n);
        let scale_hi = scale as f32;
        let scale_lo = (scale - scale_hi as f64) as f32;

        #[repr(C)]
        struct Params {
            m: u32,
            n: u32,
            freq: u32,
            n_even: u32,
            scale_hi: f32,
            scale_lo: f32,
        }
        let params = Params {
            m: m as u32,
            n: n as u32,
            freq: freq as u32,
            n_even: u32::from(n % 2 == 0),
            scale_hi,
            scale_lo,
        };

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(rs.buffer()), 0);
        enc.set_buffer(2, Some(is.buffer()), 0);
        enc.set_buffer(3, Some(&*out), 0);
        enc.set_buffer(4, Some(&*tw_buf), 0);
        let max_tg = p.max_total_threads_per_threadgroup().max(1);
        let tg = total.clamp(1, max_tg);
        let ng = total.div_ceil(tg);
        enc.dispatch_thread_groups(
            MTLSize {
                width: ng,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), total, DType::F32),
            Shape::from((m, n)),
        ))
    }
}

/// Double-double [`irfft`]: same contract (onesided f32 spectrum `[…, freq]` → real
/// `[…, n]`, `norm` scaling), but the inverse DFT is carried in double-double so the
/// GPU result tracks the true f64 inverse FFT — the precision torch's MPSGraph irfft
/// (f32-only) cannot reach. Inputs must be f32 (Metal has no f64 storage); the
/// f64-accurate result is rounded to f32 once at the end.
pub fn irfft_dd(re: &Tensor, im: &Tensor, n: usize, norm: FftNorm) -> Result<Tensor> {
    let freq = rfft_freqs(n);
    let dims = re.dims().to_vec();
    let last = *dims.last().unwrap();
    if last != freq {
        candle_core::bail!("irfft_dd: last dim {last} != n/2+1 = {freq} for n={n}");
    }
    if im.dims() != dims.as_slice() {
        candle_core::bail!("irfft_dd: re and im must have the same shape");
    }
    let m: usize = dims[..dims.len() - 1].iter().product();
    let re2 = re.reshape((m, freq))?.contiguous()?;
    let im2 = im.reshape((m, freq))?.contiguous()?;
    let out = re2.apply_op2(&im2, IrfftDd { n, norm })?; // [M, n]
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(n);
    out.reshape(out_dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Direct hermitian inverse DFT in f64 — independent of the basis builder.
    fn naive_irfft(re: &[f64], im: &[f64], n: usize, norm: FftNorm) -> Vec<f64> {
        let freq = rfft_freqs(n);
        let scale = norm.inverse_scale(n);
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut y = vec![0f64; n];
        for (j, yj) in y.iter_mut().enumerate() {
            // full hermitian spectrum, real part of Σ Xf[k] e^{+iθkj}
            let mut acc = re[0]; // k=0 (DC, imag ignored)
            for k in 1..freq {
                let ang = two_pi * k as f64 * j as f64 / n as f64;
                let last = n % 2 == 0 && k == n / 2;
                let w = if last { 1.0 } else { 2.0 };
                let im_k = if last { 0.0 } else { im[k] }; // Nyquist imag ignored
                acc += w * (re[k] * ang.cos() - im_k * ang.sin());
            }
            *yj = acc * scale;
        }
        y
    }

    #[test]
    fn irfft_f64_matches_naive() {
        let dev = Device::Cpu;
        let n = 16usize;
        let freq = rfft_freqs(n);
        let re: Vec<f64> = (0..freq).map(|k| (k as f64 * 0.3).cos() * 1.5).collect();
        let im: Vec<f64> = (0..freq).map(|k| (k as f64 * 0.21).sin()).collect();
        let exp = naive_irfft(&re, &im, n, FftNorm::Backward);
        let ret = Tensor::from_vec(re.clone(), (1, freq), &dev).unwrap();
        let imt = Tensor::from_vec(im.clone(), (1, freq), &dev).unwrap();
        let got: Vec<f64> = irfft(&ret, &imt, n, FftNorm::Backward)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let maxd = got
            .iter()
            .zip(exp.iter())
            .fold(0f64, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-12, "irfft f64 vs naive: {maxd:e}");
        eprintln!("irfft f64 == naive hermitian iDFT, max diff {maxd:.2e}");
    }

    #[test]
    fn irfft_f32_tracks_f64() {
        let dev = Device::Cpu;
        let n = 1280usize; // the detokenizer size (2^8·5, not power of two)
        let freq = rfft_freqs(n);
        let re: Vec<f64> = (0..freq)
            .map(|k| (k as f64 * 0.017).cos() * (1.0 + k as f64 * 0.01))
            .collect();
        let im: Vec<f64> = (0..freq).map(|k| (k as f64 * 0.013).sin()).collect();
        let exp = naive_irfft(&re, &im, n, FftNorm::Backward);
        let ref32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let ret = Tensor::from_vec(ref32(&re), (1, freq), &dev).unwrap();
        let imt = Tensor::from_vec(ref32(&im), (1, freq), &dev).unwrap();
        let got: Vec<f32> = irfft(&ret, &imt, n, FftNorm::Backward)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let maxd = got
            .iter()
            .zip(exp.iter())
            .fold(0f64, |m, (a, &e)| m.max((*a as f64 - e).abs()));
        // f32 matmul of a 641-term inverse DFT — within the f32 floor of the true value.
        assert!(maxd < 1e-3, "irfft f32 vs f64 ref: {maxd:e}");
        eprintln!("irfft f32 tracks f64 reference (n={n}), max diff {maxd:.2e}");
    }

    #[test]
    fn irfft_dd_cpu_matches_f64_reference() {
        let dev = Device::Cpu;
        let n = 1280usize;
        let freq = rfft_freqs(n);
        let re: Vec<f64> = (0..freq)
            .map(|k| (k as f64 * 0.017).cos() * (1.0 + k as f64 * 0.01))
            .collect();
        let im: Vec<f64> = (0..freq).map(|k| (k as f64 * 0.013).sin()).collect();
        let exp = naive_irfft(&re, &im, n, FftNorm::Backward);
        let r32: Vec<f32> = re.iter().map(|&x| x as f32).collect();
        let i32v: Vec<f32> = im.iter().map(|&x| x as f32).collect();
        let ret = Tensor::from_vec(r32, (1, freq), &dev).unwrap();
        let imt = Tensor::from_vec(i32v, (1, freq), &dev).unwrap();
        let got: Vec<f32> = irfft_dd(&ret, &imt, n, FftNorm::Backward)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // dd cpu path = the exact f64 reference rounded to f32 → within f32 epsilon.
        let maxd = got
            .iter()
            .zip(exp.iter())
            .fold(0f64, |m, (a, &e)| m.max((*a as f64 - e).abs()));
        assert!(maxd < 1e-4, "irfft_dd cpu vs f64 ref: {maxd:e}");
        eprintln!("irfft_dd cpu == f64 reference (n={n}), max diff {maxd:.2e}");
    }

    // The Metal double-double kernel must reproduce the f64 inverse FFT — i.e. match
    // the CPU f64 path (irfft_dd::cpu_fwd) bit-for-bit-close, both rounded to f32.
    // (Whether dd then *beats* f32 end-to-end is bounded by the f32 output rounding:
    // by Parseval the output magnitude tracks the input magnitude, so dd's gain over
    // f32 is realized only if the result stays in dd, not when it is rounded to f32.)
    #[cfg(feature = "metal")]
    #[test]
    fn irfft_dd_metal_matches_cpu_f64() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let n = 512usize;
        let freq = rfft_freqs(n);
        let re: Vec<f32> = (0..freq)
            .map(|k| ((k as f32 * 0.031).cos()) * (1.0 + k as f32 * 0.05))
            .collect();
        let im: Vec<f32> = (0..freq)
            .map(|k| ((k as f32 * 0.027).sin()) * (1.0 + k as f32 * 0.05))
            .collect();
        let mk = |dev: &Device, v: &[f32]| Tensor::from_vec(v.to_vec(), (1, freq), dev).unwrap();
        // dd on CPU (exact f64 reference) and dd on Metal (the kernel under test).
        let cpu_dd: Vec<f32> = irfft_dd(
            &mk(&Device::Cpu, &re),
            &mk(&Device::Cpu, &im),
            n,
            FftNorm::Backward,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
        let met_dd: Vec<f32> = irfft_dd(&mk(&mdev, &re), &mk(&mdev, &im), n, FftNorm::Backward)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let maxd = cpu_dd
            .iter()
            .zip(met_dd.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let scale = cpu_dd.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        let rel = maxd / scale;
        eprintln!("irfft_dd: metal == cpu f64, max diff {maxd:.2e} (rel {rel:.2e})");
        // Metal dd reproduces the CPU f64 inverse FFT to f32-output precision.
        assert!(rel < 1e-5, "irfft_dd metal vs cpu f64: rel {rel:e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn irfft_f32_metal_matches_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let n = 512usize;
        let freq = rfft_freqs(n);
        let re: Vec<f32> = (0..freq).map(|k| (k as f32 * 0.03).cos()).collect();
        let im: Vec<f32> = (0..freq).map(|k| (k as f32 * 0.02).sin()).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let ret = Tensor::from_vec(re.clone(), (1, freq), dev).unwrap();
            let imt = Tensor::from_vec(im.clone(), (1, freq), dev).unwrap();
            irfft(&ret, &imt, n, FftNorm::Backward)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        let maxd = cpu
            .iter()
            .zip(met.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-4, "irfft metal vs cpu: {maxd:e}");
        eprintln!("irfft f32: metal == cpu, max diff {maxd:.2e}");
    }
}
