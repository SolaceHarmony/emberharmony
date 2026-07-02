//! Fused single-pass FFT convolution (FlashFFTConv "all-in-one" path).
//!
//! One Metal dispatch does the **entire** pipeline per `(batch, channel)` —
//! `rfft(u) → ⊙ k_f → irfft → + u·D` — with the radix-2 FFT living in
//! `threadgroup` memory, no global round-trips. Ported from the experimental
//! `m2-bert-mlx/experimental/metal_bitexact/FFTConv.metal`.
//!
//! This is the fast path for sequences that fit a threadgroup: `fft_size` must be
//! a **power of two** and `≤ 1024` (one thread per FFT element), and is normally
//! `2·seqlen` so the convolution is **linear** (zero-padded), not circular. `k_f`
//! is the precomputed half-spectrum `rfft` of the (zero-padded) filter; `D` is the
//! per-channel skip term `y += u·D`. For longer / non-pow2 sequences use the
//! Monarch path ([`crate::monarch_conv`]).
//!
//! Fix vs the source kernel: the `(b,c)` indices come from
//! `threadgroup_position_in_grid` (the dispatch is `(batch, channels)` threadgroups
//! of `fft_size` threads), not `thread_position_in_grid`.

#[cfg(feature = "metal")]
use candle_core::DType;
use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

#[cfg(feature = "metal")]
/// The kernel is your vendored `metal/FFTConv.metal` (verbatim + documented
/// candle mods); we only add the host-side binding below.
const SRC: &str = include_str!("metal/FFTConv.metal");

/// Fused FFT convolution op. `fft_size` is the (power-of-two) transform length.
pub struct FusedFftConv {
    pub fft_size: usize,
}

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("fused_fft_conv expects contiguous f32 inputs"),
    }
}

impl CustomOp3 for FusedFftConv {
    fn name(&self) -> &'static str {
        "fused_fft_conv"
    }

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
        let kf = contig_f32(ks, kl)?; // [C, half, 2]
        let dd = contig_f32(ds, dl)?; // [C]
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut y = vec![0f32; b * c * seqlen];
        for bi in 0..b {
            for ci in 0..c {
                let u_base = (bi * c + ci) * seqlen;
                // zero-padded real input → full spectrum via naive DFT.
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
                    // multiply by k_f (Hermitian: mirror the upper half).
                    let (kr, ki) = if k < half {
                        (
                            kf[(ci * half + k) * 2] as f64,
                            kf[(ci * half + k) * 2 + 1] as f64,
                        )
                    } else {
                        let m = fft_size - k;
                        (
                            kf[(ci * half + m) * 2] as f64,
                            -(kf[(ci * half + m) * 2 + 1] as f64),
                        )
                    };
                    spec_re[k] = sr * kr - si * ki;
                    spec_im[k] = sr * ki + si * kr;
                }
                // inverse DFT → real, truncate to seqlen, add skip u·D.
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
            candle_core::bail!("fused_fft_conv: fft_size must be a power of two (got {fft_size})");
        }
        if fft_size > 1024 {
            candle_core::bail!("fused_fft_conv: fft_size {fft_size} exceeds 1024 (one thread/element); use monarch_conv");
        }
        let dev = us.device();
        let p = crate::metal_util::pipeline(dev, "fft_conv", SRC)?;
        let out = dev.new_buffer(b * c * seqlen, DType::F32, "fused_fft_conv")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        // Your kernel reads `constant FFTConvParams& [[buffer(0)]]`, then u/k_f/D/y
        // at buffers 1-4. Match that interface exactly.
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
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), 0);
        enc.set_buffer(2, Some(ks.buffer()), 0);
        enc.set_buffer(3, Some(ds.buffer()), 0);
        enc.set_buffer(4, Some(&*out), 0);
        // threadgroup `Complex shared[fft_size]` = fft_size * 8 bytes.
        enc.set_threadgroup_memory_length(0, fft_size * 8);
        // One threadgroup per (batch, channel); fft_size threads each.
        enc.dispatch_thread_groups(
            MTLSize {
                width: b,
                height: c,
                depth: 1,
            },
            MTLSize {
                width: fft_size,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), b * c * seqlen, DType::F32),
            Shape::from((b, c, seqlen)),
        ))
    }
}

/// Fused FFT convolution `y = irfft(rfft(u) ⊙ k_f) + u·D`, one Metal dispatch per
/// `(batch, channel)`. `u` `[B,C,seqlen]` real; `k_f` `[C, fft_size/2+1, 2]` is the
/// filter half-spectrum; `d` `[C]` the skip term. `fft_size` must be a power of two
/// `≤ 1024` (use `2·seqlen` for a linear convolution).
pub fn fused_fft_conv(u: &Tensor, k_f: &Tensor, d: &Tensor, fft_size: usize) -> Result<Tensor> {
    let u = u.contiguous()?;
    let k_f = k_f.contiguous()?;
    let d = d.contiguous()?;
    u.apply_op3(&k_f, &d, FusedFftConv { fft_size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Naive half-spectrum rfft of a length-`seqlen` real signal zero-padded to fft_size.
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

    // direct linear convolution truncated to seqlen, plus the u·D skip.
    fn direct_linear_conv(u: &[f32], k: &[f32], seqlen: usize, d: f32) -> Vec<f32> {
        let mut y = vec![0f32; seqlen];
        for t in 0..seqlen {
            let mut acc = 0f64;
            for j in 0..=t {
                acc += u[j] as f64 * k[t - j] as f64;
            }
            y[t] = acc as f32 + u[t] * d;
        }
        y
    }

    #[test]
    fn cpu_matches_direct_linear_conv() {
        let dev = Device::Cpu;
        let (seqlen, fft_size) = (16usize, 32);
        let u: Vec<f32> = (0..seqlen).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..seqlen)
            .map(|i| (i as f32 * 0.17 + 0.5).cos() * 0.4)
            .collect();
        let dval = 0.25f32;
        let kf = rfft_half(&k, fft_size);
        let ut = Tensor::from_vec(u.clone(), (1, 1, seqlen), &dev).unwrap();
        let kt = Tensor::from_vec(kf, (1, fft_size / 2 + 1, 2), &dev).unwrap();
        let dt = Tensor::from_vec(vec![dval], (1,), &dev).unwrap();
        let y: Vec<f32> = fused_fft_conv(&ut, &kt, &dt, fft_size)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let exp = direct_linear_conv(&u, &k, seqlen, dval);
        let maxd = y
            .iter()
            .zip(exp.iter())
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(
            maxd < 1e-3,
            "fused conv != direct linear conv, max diff {maxd}"
        );
        eprintln!("fused_fft_conv == direct linear conv, max diff {maxd:.2e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, c, seqlen, fft_size) = (2usize, 3, 64, 128);
        let half = fft_size / 2 + 1;
        let u: Vec<f32> = (0..b * c * seqlen)
            .map(|i| ((i * 13 % 23) as f32 * 0.04) - 0.4)
            .collect();
        let kf: Vec<f32> = (0..c * half * 2)
            .map(|i| ((i * 5 % 9) as f32 * 0.05) - 0.2)
            .collect();
        let dd: Vec<f32> = (0..c).map(|i| 0.1 * i as f32).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let ut = Tensor::from_vec(u.clone(), (b, c, seqlen), dev).unwrap();
            let kt = Tensor::from_vec(kf.clone(), (c, half, 2), dev).unwrap();
            let dt = Tensor::from_vec(dd.clone(), (c,), dev).unwrap();
            fused_fft_conv(&ut, &kt, &dt, fft_size)
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
        assert!(maxd < 1e-3, "fused conv metal vs cpu max diff {maxd}");
        eprintln!("fused_fft_conv: metal == cpu, max diff {maxd:.2e}");
    }
}
