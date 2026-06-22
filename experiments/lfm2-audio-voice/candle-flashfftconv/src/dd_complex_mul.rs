//! Extended-precision (double-double) complex multiply on Metal.
//!
//! Uses the vendored `metal/double_double.metal` toolkit — specifically `cdd_mul`,
//! which the source calls "the CRITICAL operation for FFT frequency-domain
//! multiply." A naïve f32 complex multiply `(ar·br − ai·bi, ar·bi + ai·br)` rounds
//! three times; lifting each input to a `complex_dd` (two f32 limbs), multiplying in
//! double-double (exact `two_prod` + compensated adds), and rounding **once** at the
//! end yields the correctly-rounded result — i.e. ~f64 accuracy in f32 storage, which
//! is the path *below* the float32 floor on the GPU (Metal has no f64).
//!
//! The Metal kernel is small; all the arithmetic is your `double_double.metal`.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// The kernel that calls `cdd_mul`; compiled with your `double_double.metal`
/// prepended (it provides `complex_dd`, `cdd_mul`, `cdd_to_float2`, and pulls in
/// `<metal_stdlib>`).
#[cfg(feature = "metal")]
const KERNEL: &str = r#"
kernel void complex_mul_dd(
    constant uint& n          [[buffer(0)]],
    const device float2* a    [[buffer(1)]],   // [N] complex
    const device float2* b    [[buffer(2)]],   // [N] complex
    device float2* out        [[buffer(3)]],   // [N] complex
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    out[gid] = cdd_to_float2(cdd_mul(complex_dd(a[gid]), complex_dd(b[gid])));
}
"#;

#[cfg(feature = "metal")]
fn dd_source() -> String {
    format!("{}\n{}", include_str!("metal/double_double.metal"), KERNEL)
}

struct ComplexMulDd;

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("complex_mul_dd expects contiguous f32 inputs"),
    }
}

impl CustomOp2 for ComplexMulDd {
    fn name(&self) -> &'static str {
        "complex_mul_dd"
    }

    /// CPU reference: the correctly-rounded complex product — compute in f64, round
    /// once to f32. This is exactly what the double-double Metal kernel targets.
    fn cpu_fwd(&self, as_: &CpuStorage, al: &Layout, bs: &CpuStorage, bl: &Layout) -> Result<(CpuStorage, Shape)> {
        let dims = al.shape().dims().to_vec();
        let n: usize = dims.iter().product::<usize>() / 2; // last axis is the size-2 complex
        let a = contig_f32(as_, al)?;
        let b = contig_f32(bs, bl)?;
        let mut out = vec![0f32; n * 2];
        for i in 0..n {
            let (ar, ai) = (a[i * 2] as f64, a[i * 2 + 1] as f64);
            let (br, bi) = (b[i * 2] as f64, b[i * 2 + 1] as f64);
            out[i * 2] = (ar * br - ai * bi) as f32;
            out[i * 2 + 1] = (ar * bi + ai * br) as f32;
        }
        Ok((CpuStorage::F32(out), Shape::from(dims)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        as_: &candle_core::MetalStorage,
        al: &Layout,
        bs: &candle_core::MetalStorage,
        _bl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::MetalStorage;
        use objc2_metal::MTLSize;

        let dims = al.shape().dims().to_vec();
        let n: usize = dims.iter().product::<usize>() / 2;
        let dev = as_.device();
        // pipeline() caches by fn name; the source (dd toolkit + kernel) compiles once.
        let p = crate::metal_util::pipeline(dev, "complex_mul_dd", &dd_source())?;
        let out = dev.new_buffer(n * 2, DType::F32, "complex_mul_dd")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &(n as u32));
        enc.set_buffer(1, Some(as_.buffer()), 0);
        enc.set_buffer(2, Some(bs.buffer()), 0);
        enc.set_buffer(3, Some(&*out), 0);
        let max_tg = p.max_total_threads_per_threadgroup().max(1);
        let tg = n.clamp(1, max_tg);
        let ng = n.div_ceil(tg);
        enc.dispatch_thread_groups(
            MTLSize { width: ng, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        Ok((MetalStorage::new(out, dev.clone(), n * 2, DType::F32), Shape::from(dims)))
    }
}

/// Correctly-rounded (double-double) element-wise complex multiply of `a` and `b`
/// (trailing `[…, 2]` axis). On Metal this runs your `cdd_mul`; on CPU it is the
/// f64-rounded reference. More accurate than a naïve f32 complex multiply.
pub fn complex_mul_dd(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    a.apply_op2(&b, ComplexMulDd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn naive_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = a.len() / 2;
        let mut out = vec![0f32; n * 2];
        for i in 0..n {
            let (ar, ai, br, bi) = (a[i * 2], a[i * 2 + 1], b[i * 2], b[i * 2 + 1]);
            out[i * 2] = ar * br - ai * bi; // all-f32: three roundings
            out[i * 2 + 1] = ar * bi + ai * br;
        }
        out
    }
    fn ref_f64(a: &[f32], b: &[f32]) -> Vec<f64> {
        let n = a.len() / 2;
        let mut out = vec![0f64; n * 2];
        for i in 0..n {
            let (ar, ai) = (a[i * 2] as f64, a[i * 2 + 1] as f64);
            let (br, bi) = (b[i * 2] as f64, b[i * 2 + 1] as f64);
            out[i * 2] = ar * br - ai * bi;
            out[i * 2 + 1] = ar * bi + ai * br;
        }
        out
    }

    // Inputs chosen so ar·br ≈ ai·bi (catastrophic cancellation) — where naïve f32 hurts.
    fn data(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0f32; n * 2];
        let mut b = vec![0f32; n * 2];
        for i in 0..n {
            let t = i as f32;
            a[i * 2] = 1.0 + t * 1e-3;
            a[i * 2 + 1] = 1.0 + t * 1.0001e-3;
            b[i * 2] = 1.0 + t * 1.0002e-3;
            b[i * 2 + 1] = 1.0 + t * 0.9999e-3;
        }
        (a, b)
    }

    #[test]
    fn dd_beats_naive_f32() {
        // cpu_fwd is the f64-rounded result; show it is closer to true f64 than naïve f32.
        let dev = Device::Cpu;
        let (a, b) = data(64);
        let at = Tensor::from_vec(a.clone(), (64, 2), &dev).unwrap();
        let bt = Tensor::from_vec(b.clone(), (64, 2), &dev).unwrap();
        let dd: Vec<f32> = complex_mul_dd(&at, &bt).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let f32n = naive_f32(&a, &b);
        let refd = ref_f64(&a, &b);
        let err = |v: &[f32]| v.iter().zip(refd.iter()).fold(0f64, |m, (x, r)| m.max((*x as f64 - r).abs()));
        let (e_dd, e_f32) = (err(&dd), err(&f32n));
        assert!(e_dd <= e_f32, "dd ({e_dd:.2e}) should not be worse than naive f32 ({e_f32:.2e})");
        eprintln!("complex_mul: dd/correctly-rounded err {e_dd:.2e} ≤ naive-f32 err {e_f32:.2e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_dd_matches_correctly_rounded_cpu() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (a, b) = data(256);
        let run = |dev: &Device| -> Vec<f32> {
            let at = Tensor::from_vec(a.clone(), (256, 2), dev).unwrap();
            let bt = Tensor::from_vec(b.clone(), (256, 2), dev).unwrap();
            complex_mul_dd(&at, &bt).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let cpu = run(&Device::Cpu); // f64-rounded reference
        let met = run(&mdev); // your cdd_mul on the GPU
        let maxd = cpu.iter().zip(met.iter()).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        // double-double should reproduce the correctly-rounded result to f32 ulp.
        assert!(maxd < 1e-6, "metal cdd_mul vs correctly-rounded cpu: {maxd:e}");
        eprintln!("metal cdd_mul == correctly-rounded cpu, max diff {maxd:.2e}");
    }
}
