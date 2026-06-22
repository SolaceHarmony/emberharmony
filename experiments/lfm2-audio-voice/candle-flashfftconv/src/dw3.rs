//! LFM2 short-conv via your deterministic 3-tap depthwise kernel.
//!
//! The Metal kernel is your vendored `metal/Depthwise3.metal` (`depthwise3_causal`,
//! the LFM2-causal variant added alongside your verbatim `depthwise3`): a depthwise
//! K=3 conv with a **fixed multiply-add order** `(x0·w0 + x1·w1) + x2·w2`. For LFM2
//! the window looks backward (`out[t] = x[t-2]·w0 + x[t-1]·w1 + x[t]·w2`), i.e. the
//! causal short-conv (`conv_L_cache=3`, no bias). Equivalent to
//! [`crate::depthwise_conv1d`] with `padding=2` narrowed to `L`, but driven by your
//! deterministic kernel.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

#[cfg(feature = "metal")]
const SRC: &str = include_str!("metal/Depthwise3.metal");

struct Depthwise3Causal;

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("depthwise3 expects contiguous f32 inputs"),
    }
}

impl CustomOp2 for Depthwise3Causal {
    fn name(&self) -> &'static str {
        "depthwise3_causal"
    }

    fn cpu_fwd(&self, xs: &CpuStorage, xl: &Layout, ks: &CpuStorage, kl: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, c, l) = xl.shape().dims3()?;
        let (ck, three) = kl.shape().dims2()?;
        if ck != c || three != 3 {
            candle_core::bail!("depthwise3: weight must be [C,3], got [{ck},{three}] for C={c}");
        }
        let x = contig_f32(xs, xl)?;
        let k = contig_f32(ks, kl)?;
        let mut y = vec![0f32; b * c * l];
        for bi in 0..b {
            for ci in 0..c {
                let base = (bi * c + ci) * l;
                let (w0, w1, w2) = (k[ci * 3], k[ci * 3 + 1], k[ci * 3 + 2]);
                for t in 0..l {
                    let x0 = if t >= 2 { x[base + t - 2] } else { 0.0 };
                    let x1 = if t >= 1 { x[base + t - 1] } else { 0.0 };
                    let x2 = x[base + t];
                    // same fixed order as the kernel: (x0*w0 + x1*w1) + x2*w2
                    let acc = (x0 * w0) + (x1 * w1);
                    y[base + t] = acc + (x2 * w2);
                }
            }
        }
        Ok((CpuStorage::F32(y), Shape::from((b, c, l))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        xs: &candle_core::MetalStorage,
        xl: &Layout,
        ks: &candle_core::MetalStorage,
        _kl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::MTLSize;

        let (b, c, l) = xl.shape().dims3()?;
        let total = b * c * l;
        let dev = xs.device();
        let p = crate::metal_util::pipeline(dev, "depthwise3_causal", SRC)?;
        let out = dev.new_buffer(total, DType::F32, "depthwise3_causal")?;

        #[repr(C)]
        struct Params {
            batch: u32,
            channels: u32,
            length: u32,
        }
        let params = Params { batch: b as u32, channels: c as u32, length: l as u32 };

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(xs.buffer()), 0);
        enc.set_buffer(2, Some(ks.buffer()), 0);
        enc.set_buffer(3, Some(&*out), 0);
        let max_tg = p.max_total_threads_per_threadgroup().max(1);
        let tg = total.clamp(1, max_tg);
        let ng = total.div_ceil(tg);
        enc.dispatch_thread_groups(
            MTLSize { width: ng, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), total, DType::F32), Shape::from((b, c, l))))
    }
}

/// LFM2 causal short-conv via your `Depthwise3` kernel: `x` `[B,C,L]`, depthwise
/// weight `k` `[C,3]` → `[B,C,L]`, `out[t] = x[t-2]·w0 + x[t-1]·w1 + x[t]·w2`.
pub fn depthwise3_causal(x: &Tensor, k: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let k = k.contiguous()?;
    x.apply_op2(&k, Depthwise3Causal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn naive_causal(x: &[f32], k: &[f32], b: usize, c: usize, l: usize) -> Vec<f32> {
        let mut y = vec![0f32; b * c * l];
        for bi in 0..b {
            for ci in 0..c {
                let base = (bi * c + ci) * l;
                for t in 0..l {
                    let mut acc = 0f64;
                    for j in 0..3usize {
                        // out[t] = Σ_j x[t-2+j]·w[j]
                        let idx = t as i64 - 2 + j as i64;
                        if idx >= 0 {
                            acc += x[base + idx as usize] as f64 * k[ci * 3 + j] as f64;
                        }
                    }
                    y[base + t] = acc as f32;
                }
            }
        }
        y
    }

    #[test]
    fn cpu_matches_naive_causal() {
        let dev = Device::Cpu;
        let (b, c, l) = (2usize, 3, 11);
        let x: Vec<f32> = (0..b * c * l).map(|i| (i as f32 * 0.2).sin()).collect();
        let k: Vec<f32> = (0..c * 3).map(|i| (i as f32 * 0.1 + 0.3).cos() * 0.5).collect();
        let xt = Tensor::from_vec(x.clone(), (b, c, l), &dev).unwrap();
        let kt = Tensor::from_vec(k.clone(), (c, 3), &dev).unwrap();
        let y: Vec<f32> = depthwise3_causal(&xt, &kt).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let exp = naive_causal(&x, &k, b, c, l);
        let maxd = y.iter().zip(exp.iter()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-6, "depthwise3 causal vs naive: {maxd}");
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
        let (b, c, l) = (2usize, 4, 37);
        let x: Vec<f32> = (0..b * c * l).map(|i| ((i * 7 % 13) as f32 * 0.1) - 0.6).collect();
        let k: Vec<f32> = (0..c * 3).map(|i| ((i * 5 % 7) as f32 * 0.05)).collect();
        let run = |dev: &Device| -> Vec<f32> {
            let xt = Tensor::from_vec(x.clone(), (b, c, l), dev).unwrap();
            let kt = Tensor::from_vec(k.clone(), (c, 3), dev).unwrap();
            depthwise3_causal(&xt, &kt).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        let maxd = cpu.iter().zip(met.iter()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "depthwise3 metal vs cpu: {maxd}");
        eprintln!("depthwise3_causal: metal == cpu, max diff {maxd:.2e}");
    }
}
