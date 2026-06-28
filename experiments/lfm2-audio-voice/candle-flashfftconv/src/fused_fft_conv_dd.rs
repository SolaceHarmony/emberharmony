//! Double-double fused FFT convolution — the below-float32 path.
//!
//! Same interface and result as [`crate::fused_fft_conv`], but the Metal kernel
//! ([`metal/FFTConvDd.metal`]) carries the radix-2 FFT, the frequency-domain
//! multiply, and the IFFT in **double-double** using your `double_double.metal`
//! primitives — so the GPU result tracks the true (f64) convolution to ~f64
//! accuracy instead of the f32 floor. Threadgroup memory is `complex_dd` (4 f32 =
//! 16 bytes/element), so `fft_size ≤ 1024` still fits.

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
#[cfg(feature = "metal")]
use candle_core::DType;

#[cfg(feature = "metal")]
fn dd_source() -> String {
    format!(
        "{}\n{}",
        include_str!("metal/double_double.metal"),
        include_str!("metal/FFTConvDd.metal")
    )
}

/// Fused FFT convolution op, double-double precision.
pub struct FusedFftConvDd {
    pub fft_size: usize,
}

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("fused_fft_conv_dd expects contiguous f32 inputs"),
    }
}

impl CustomOp3 for FusedFftConvDd {
    fn name(&self) -> &'static str {
        "fused_fft_conv_dd"
    }

    /// CPU reference: the high-accuracy (f64) convolution — the target the
    /// double-double GPU kernel reproduces. Identical math to
    /// [`crate::FusedFftConv`]'s cpu path.
    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ks: &CpuStorage,
        kl: &Layout,
        ds: &CpuStorage,
        dl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, c, seqlen) = ul.shape().dims3()?;
        let fft_size = self.fft_size;
        let half = fft_size / 2 + 1;
        let u = contig_f32(us, ul)?;
        let kf = contig_f32(ks, kl)?;
        let dd = contig_f32(ds, dl)?;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut y = vec![0f32; b * c * seqlen];
        for bi in 0..b {
            for ci in 0..c {
                let u_base = (bi * c + ci) * seqlen;
                let mut spec_re = vec![0f64; fft_size];
                let mut spec_im = vec![0f64; fft_size];
                for k in 0..fft_size {
                    let (mut sr, mut si) = (0f64, 0f64);
                    for t in 0..seqlen {
                        let ang = -two_pi * (k as f64) * (t as f64) / fft_size as f64;
                        let uv = u[u_base + t] as f64;
                        sr += uv * ang.cos();
                        si += uv * ang.sin();
                    }
                    let (kr, ki) = if k < half {
                        (kf[(ci * half + k) * 2] as f64, kf[(ci * half + k) * 2 + 1] as f64)
                    } else {
                        let m = fft_size - k;
                        (kf[(ci * half + m) * 2] as f64, -(kf[(ci * half + m) * 2 + 1] as f64))
                    };
                    spec_re[k] = sr * kr - si * ki;
                    spec_im[k] = sr * ki + si * kr;
                }
                let scale = 1.0 / fft_size as f64;
                for t in 0..seqlen {
                    let mut acc = 0f64;
                    for k in 0..fft_size {
                        let ang = two_pi * (k as f64) * (t as f64) / fft_size as f64;
                        acc += spec_re[k] * ang.cos() - spec_im[k] * ang.sin();
                    }
                    y[u_base + t] = (acc * scale) as f32 + u[u_base + t] * dd[ci];
                }
            }
        }
        Ok((CpuStorage::F32(y), Shape::from((b, c, seqlen))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ks: &candle_core::MetalStorage,
        _kl: &Layout,
        ds: &candle_core::MetalStorage,
        _dl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::MTLSize;

        let (b, c, seqlen) = ul.shape().dims3()?;
        let fft_size = self.fft_size;
        if !fft_size.is_power_of_two() {
            candle_core::bail!("fused_fft_conv_dd: fft_size must be a power of two (got {fft_size})");
        }
        if fft_size > 1024 {
            candle_core::bail!("fused_fft_conv_dd: fft_size {fft_size} exceeds 1024 (one thread/element)");
        }
        let dev = us.device();
        let p = crate::metal_util::pipeline(dev, "fft_conv_dd", &dd_source())?;
        let out = dev.new_buffer(b * c * seqlen, DType::F32, "fused_fft_conv_dd")?;

        // Double-double twiddle table, computed in f64 on the host and split into
        // hi/lo f32 limbs (Dekker), packed float4(re.hi, re.lo, im.hi, im.lo) for
        // j in [0, fft_size/2). This replaces the GPU f32 cos/sin so the dd
        // butterflies are actually exact (the "DD Taylor series" TODO, done host-side).
        let half_tw = fft_size / 2;
        let mut tw = vec![0f32; half_tw * 4];
        for j in 0..half_tw {
            let ang = -2.0 * std::f64::consts::PI * j as f64 / fft_size as f64;
            let (re, im) = (ang.cos(), ang.sin());
            let re_hi = re as f32;
            let im_hi = im as f32;
            tw[j * 4] = re_hi;
            tw[j * 4 + 1] = (re - re_hi as f64) as f32;
            tw[j * 4 + 2] = im_hi;
            tw[j * 4 + 3] = (im - im_hi as f64) as f32;
        }
        let tw_buf = dev.new_buffer_with_data(&tw)?;

        #[repr(C)]
        struct FFTConvParams {
            batch: u32,
            channels: u32,
            seqlen: u32,
            fft_size: u32,
        }
        let params = FFTConvParams {
            batch: b as u32,
            channels: c as u32,
            seqlen: seqlen as u32,
            fft_size: fft_size as u32,
        };

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), 0);
        enc.set_buffer(2, Some(ks.buffer()), 0);
        enc.set_buffer(3, Some(ds.buffer()), 0);
        enc.set_buffer(4, Some(&*out), 0);
        enc.set_buffer(5, Some(&*tw_buf), 0);
        // complex_dd = 4 f32 = 16 bytes per FFT element.
        enc.set_threadgroup_memory_length(0, fft_size * 16);
        enc.dispatch_thread_groups(
            MTLSize { width: b, height: c, depth: 1 },
            MTLSize { width: fft_size, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), b * c * seqlen, DType::F32), Shape::from((b, c, seqlen))))
    }
}

/// Double-double [`crate::fused_fft_conv`]: `y = irfft(rfft(u) ⊙ k_f) + u·D`, with
/// the FFT/multiply/IFFT carried in double-double on Metal (below the f32 floor).
pub fn fused_fft_conv_dd(u: &Tensor, k_f: &Tensor, d: &Tensor, fft_size: usize) -> Result<Tensor> {
    let u = u.contiguous()?;
    let k_f = k_f.contiguous()?;
    let d = d.contiguous()?;
    u.apply_op3(&k_f, &d, FusedFftConvDd { fft_size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn rfft_half(k: &[f32], fft_size: usize) -> Vec<f32> {
        let half = fft_size / 2 + 1;
        let mut out = vec![0f32; half * 2];
        let two_pi = 2.0 * std::f64::consts::PI;
        for kk in 0..half {
            let (mut re, mut im) = (0f64, 0f64);
            for (t, &kv) in k.iter().enumerate() {
                let ang = -two_pi * kk as f64 * t as f64 / fft_size as f64;
                re += kv as f64 * ang.cos();
                im += kv as f64 * ang.sin();
            }
            out[kk * 2] = re as f32;
            out[kk * 2 + 1] = im as f32;
        }
        out
    }

    #[cfg(feature = "metal")]
    #[test]
    fn dd_metal_beats_f32_metal_vs_f64() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (seqlen, fft_size) = (64usize, 128);
        // ill-conditioned filter: large dynamic range so f32 accumulation drifts.
        let u: Vec<f32> = (0..seqlen).map(|i| (i as f32 * 0.37).sin() * 8.0).collect();
        let k: Vec<f32> = (0..seqlen).map(|i| (i as f32 * 0.11).cos() * (1.0 + i as f32 * 0.3)).collect();
        let kf = rfft_half(&k, fft_size);
        let cpu = Device::Cpu;

        let ut = |dev: &Device| Tensor::from_vec(u.clone(), (1, 1, seqlen), dev).unwrap();
        let kt = |dev: &Device| Tensor::from_vec(kf.clone(), (1, fft_size / 2 + 1, 2), dev).unwrap();
        let dt = |dev: &Device| Tensor::from_vec(vec![0.0f32], (1,), dev).unwrap();

        // f64 reference = the dd op's cpu path.
        let f64ref: Vec<f32> = fused_fft_conv_dd(&ut(&cpu), &kt(&cpu), &dt(&cpu), fft_size)
            .unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // f32 Metal kernel.
        let f32m: Vec<f32> = crate::fused_fft_conv(&ut(&mdev), &kt(&mdev), &dt(&mdev), fft_size)
            .unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // double-double Metal kernel (your primitives).
        let ddm: Vec<f32> = fused_fft_conv_dd(&ut(&mdev), &kt(&mdev), &dt(&mdev), fft_size)
            .unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let err = |v: &[f32]| v.iter().zip(f64ref.iter()).fold(0f32, |m, (a, r)| m.max((a - r).abs()));
        let (e_dd, e_f32) = (err(&ddm), err(&f32m));
        eprintln!("fused conv vs f64: double-double {e_dd:.3e}  vs  f32 {e_f32:.3e}");
        // The dd FFT/multiply/IFFT (with the host f64 twiddle table) is strictly
        // closer to the f64 reference than the f32 kernel.
        assert!(e_dd < e_f32, "double-double ({e_dd:e}) should beat f32 ({e_f32:e})");
    }
}
