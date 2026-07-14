use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv2d, Linear, Module};

use crate::bf16_gemm::{bf16_matmul, bf16_matmul_accel, bf16_matmul_nt};

/// Decode-side row-count bound for the no-transpose matmul: at small M the weight-transpose
/// copy (`w.t().contiguous()`) dominates the actual math — profiled at ~97% of CPU decode
/// time — so `[rows ≤ 4]` (decode steps and suffix chunks) dots the weight in its native
/// `[N,K]` layout instead. Prefill-scale M keeps the BFMMLA GEMM, where one transpose
/// amortizes over the M rows.
const NT_MAX_ROWS: usize = 4;

#[derive(Clone, Debug)]
pub struct Bf16Linear {
    inner: Linear,
}

impl Bf16Linear {
    pub fn new(inner: Linear) -> Self {
        Self { inner }
    }
}

impl Module for Bf16Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        linear_forward(&self.inner, x)
    }
}

pub fn linear_forward(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    if !needs_bf16_cpu_path(linear, x) {
        return linear.forward(x);
    }

    let output = linear.weight().dim(0)?;
    let y = match x.dims() {
        [k] => matmul_flat(linear, &x.reshape((1, *k))?)?.squeeze(0)?,
        dims if dims.len() >= 2 => {
            let k = *dims.last().unwrap();
            let rows = dims[..dims.len() - 1].iter().product::<usize>();
            let flat = x.contiguous()?.reshape((rows, k))?;
            let mut shape = dims[..dims.len() - 1].to_vec();
            shape.push(output);
            matmul_flat(linear, &flat)?.reshape(shape)?
        }
        _ => linear.forward(x)?,
    };

    cast_like_input(add_bias(linear, &y)?, x.dtype())
}

pub fn linear_logits(weight: &Tensor, x: &Tensor) -> Result<Tensor> {
    let y = if weight.device().is_cpu() && x.device().is_cpu() && weight.dtype() == DType::BF16 {
        // Same small-M dispatch as matmul_flat: at decode the logits head is M==1 against
        // the biggest weight in the model — the transpose copy hurts the most here.
        let rows = if x.rank() == 2 { x.dim(0)? } else { usize::MAX };
        let nt = if rows <= NT_MAX_ROWS {
            bf16_matmul_nt(x, weight)?
        } else {
            None
        };
        if let Some(y) = nt {
            y
        } else {
            let Some(y) = bf16_matmul(x, &weight.t()?)? else {
                candle_core::bail!(
                    "CPU bf16 linear requested but the NEON BFMMLA kernel is unavailable"
                );
            };
            y
        }
    } else {
        x.matmul(&weight.t()?)?
    };
    y.to_dtype(DType::F32)
}

pub fn conv1d_forward(conv: &Conv1d, x: &Tensor) -> Result<Tensor> {
    if !needs_bf16_cpu_conv(conv.weight(), x) {
        return conv.forward(x);
    }
    let cfg = conv.config();
    let y = x.to_dtype(DType::F32)?.conv1d_with_algo(
        &conv.weight().to_dtype(DType::F32)?,
        cfg.padding,
        cfg.stride,
        cfg.dilation,
        cfg.groups,
        cfg.cudnn_fwd_algo,
    )?;
    let y = match conv.bias() {
        Some(bias) => {
            let b = bias.dims1()?;
            y.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, b, 1))?)?
        }
        None => y,
    };
    y.to_dtype(DType::BF16)
}

pub fn conv2d_forward(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    if !needs_bf16_cpu_conv(conv.weight(), x) {
        return conv.forward(x);
    }
    let cfg = conv.config();
    let y = x.to_dtype(DType::F32)?.conv2d_with_algo(
        &conv.weight().to_dtype(DType::F32)?,
        cfg.padding,
        cfg.stride,
        cfg.dilation,
        cfg.groups,
        cfg.cudnn_fwd_algo,
    )?;
    let y = match conv.bias() {
        Some(bias) => {
            let b = bias.dims1()?;
            y.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, b, 1, 1))?)?
        }
        None => y,
    };
    y.to_dtype(DType::BF16)
}

fn needs_bf16_cpu_path(linear: &Linear, x: &Tensor) -> bool {
    x.device().is_cpu()
        && linear.weight().device().is_cpu()
        && x.dtype() == DType::BF16
        && linear.weight().dtype() == DType::BF16
}

fn needs_bf16_cpu_conv(weight: &Tensor, x: &Tensor) -> bool {
    x.device().is_cpu()
        && weight.device().is_cpu()
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
}

fn matmul_flat(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    if x.dim(0)? <= NT_MAX_ROWS {
        // `None` means the runtime CPU feature gate rejected the native-layout kernel.
        if let Some(y) = bf16_matmul_nt(x, linear.weight())? {
            return Ok(y);
        }
    }
    // Prefill-scale M: the Accelerate/AMX backend (ENGINE_DESIGN.md §E4 — measured
    // 19–28× the BFMMLA chain, native [N,K] weight, no transpose copy). `None` off
    // macOS → the BFMMLA path below, same numerics tier as always.
    if let Some(y) = bf16_matmul_accel(x, linear.weight())? {
        return Ok(y);
    }
    let Some(y) = bf16_matmul(x, &linear.weight().t()?)? else {
        candle_core::bail!("CPU bf16 linear requested but the NEON BFMMLA kernel is unavailable");
    };
    Ok(y)
}

fn add_bias(linear: &Linear, y: &Tensor) -> Result<Tensor> {
    match linear.bias() {
        Some(bias) => y.broadcast_add(&bias.to_dtype(y.dtype())?),
        None => Ok(y.clone()),
    }
}

fn cast_like_input(y: Tensor, dtype: DType) -> Result<Tensor> {
    if dtype == DType::BF16 {
        y.to_dtype(DType::BF16)
    } else {
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    //! Pipeline parity for the bf16 GEMM kernel — the same methodology as
    //! `tests/short_conv_parity.rs`: synthetic candle tensors (no model weights) pushed through
    //! the **real** consumer ([`linear_forward`], which routes bf16 CPU linears through the
    //! NEON/x86 kernel via [`bf16_matmul`]), compared against an f32 reference that reproduces the
    //! kernel's numerics (bf16-rounded inputs, f32 accumulate, bf16-rounded output). Exercises the
    //! kernel where it's actually used: a single linear, a chained 2-layer stack, a gated MLP
    //! block, and the M==1 decode GEMV — at model-scale contraction depth (K≈2048).
    use super::*;
    use candle_core::{Device, Tensor};

    fn dev() -> Device {
        Device::Cpu
    }

    // A candle Linear with deterministic pseudo-random bf16 weight [out,in] (+ optional bias).
    fn mk_linear(inp: usize, out: usize, seed: u64, bias: bool) -> Linear {
        let w: Vec<f32> = (0..out * inp)
            .map(|i| {
                (((i as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 500.0)
                    - 1.0
            })
            .collect();
        let w = Tensor::from_vec(w, (out, inp), &dev())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = bias.then(|| {
            let bv: Vec<f32> = (0..out)
                .map(|i| {
                    (((i as u64).wrapping_mul(40503).wrapping_add(seed) % 1000) as f32 / 1000.0)
                        - 0.5
                })
                .collect();
            Tensor::from_vec(bv, (out,), &dev())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        });
        Linear::new(w, b)
    }

    // bf16 input [rows, in].
    fn mk_input(rows: usize, inp: usize, seed: u64) -> Tensor {
        let x: Vec<f32> = (0..rows * inp)
            .map(|i| {
                (((i as u64).wrapping_mul(2246822519).wrapping_add(seed) % 1000) as f32 / 500.0)
                    - 1.0
            })
            .collect();
        Tensor::from_vec(x, (rows, inp), &dev())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    // f32 reference reproducing `linear_forward`'s bf16 numerics: (bf16 x · bf16 w) in f32,
    // + f32 bias, rounded back to bf16 — the exact-product / f32-accumulate the kernel targets.
    fn ref_linear(lin: &Linear, x: &Tensor) -> Tensor {
        let xf = x.to_dtype(DType::F32).unwrap();
        let wf = lin.weight().to_dtype(DType::F32).unwrap();
        let mut y = xf.matmul(&wf.t().unwrap()).unwrap();
        if let Some(b) = lin.bias() {
            y = y.broadcast_add(&b.to_dtype(DType::F32).unwrap()).unwrap();
        }
        y.to_dtype(DType::BF16).unwrap()
    }

    fn max_rel(a: &Tensor, b: &Tensor) -> f32 {
        let av: Vec<f32> = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let bv: Vec<f32> = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(av.len(), bv.len());
        let md = av
            .iter()
            .zip(&bv)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let sc = bv.iter().fold(1e-6f32, |m, &x| m.max(x.abs()));
        md / sc
    }

    #[test]
    fn accel_prefill_matches_bfmmla_at_f32_tier() {
        // The E4 backend contract: Accelerate sgemm (AMX order) vs the BFMMLA chain over
        // the SAME bf16 inputs — identical products, different accumulation order, so the
        // bound is the f32 tier (measured ≈1e-5 at prefill shapes). Direct fn A/B — the
        // reference path needs no runtime flag.
        if !(crate::bf16_gemm::bf16_gemm_accel_available()
            && crate::bf16_gemm::bf16_gemm_available())
        {
            eprintln!("accel or bfmmla backend unavailable — skipping");
            return;
        }
        use crate::bf16_gemm::{bf16_matmul, bf16_matmul_accel};
        for &(m, k, n) in &[(64usize, 512usize, 384usize), (350, 2048, 512)] {
            let lin = mk_linear(k, n, 11, false);
            let x = mk_input(m, k, 17);
            let a = bf16_matmul_accel(&x, lin.weight())
                .unwrap()
                .expect("accel available");
            let r = bf16_matmul(&x, &lin.weight().t().unwrap())
                .unwrap()
                .expect("bfmmla available");
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let rv: Vec<f32> = r.flatten_all().unwrap().to_vec1().unwrap();
            let (mut md, mut sc) = (0f32, 1e-6f32);
            for (x, y) in av.iter().zip(&rv) {
                md = md.max((x - y).abs());
                sc = sc.max(y.abs());
            }
            assert!(
                md / sc < 1e-4,
                "m={m} k={k} n={n}: accel vs bfmmla rel {}",
                md / sc
            );
        }
    }

    fn skip() -> bool {
        if !crate::bf16_gemm::bf16_gemm_available() {
            eprintln!("bf16 GEMM kernel unavailable on this target — skipping pipeline parity");
            return true;
        }
        false
    }

    #[test]
    fn single_linear_parity() {
        if skip() {
            return;
        }
        // (rows, in, out): prefill batch×seq at model-scale K.
        for &(rows, inp, out, bias) in &[(16usize, 2048usize, 512usize, true), (7, 320, 129, false)]
        {
            let lin = mk_linear(inp, out, 11, bias);
            let x = mk_input(rows, inp, 23);
            let got = linear_forward(&lin, &x).unwrap();
            let rel = max_rel(&got, &ref_linear(&lin, &x));
            assert!(
                rel < 1e-2,
                "single_linear rows={rows} in={inp} out={out} rel={rel}"
            );
        }
    }

    #[test]
    fn two_layer_stack_parity() {
        if skip() {
            return;
        }
        // Composed matmuls: the kernel is used twice, errors propagate through the stack.
        let (rows, inp, hid, out) = (12usize, 1024usize, 2048usize, 256usize);
        let l1 = mk_linear(inp, hid, 1, true);
        let l2 = mk_linear(hid, out, 2, true);
        let x = mk_input(rows, inp, 3);
        let got = linear_forward(&l2, &linear_forward(&l1, &x).unwrap()).unwrap();
        let want = ref_linear(&l2, &ref_linear(&l1, &x));
        let rel = max_rel(&got, &want);
        assert!(rel < 2e-2, "two_layer rel={rel}");
    }

    #[test]
    fn mlp_block_parity() {
        if skip() {
            return;
        }
        // A realistic gated-MLP-ish block: linear → SiLU → linear, the nonlinearity applied
        // identically (bf16) in both the kernel path and the reference.
        let (rows, inp, hid, out) = (8usize, 2048usize, 2048usize, 2048usize);
        let l1 = mk_linear(inp, hid, 5, false);
        let l2 = mk_linear(hid, out, 6, false);
        let x = mk_input(rows, inp, 7);
        let got = linear_forward(&l2, &linear_forward(&l1, &x).unwrap().silu().unwrap()).unwrap();
        let want = ref_linear(&l2, &ref_linear(&l1, &x).silu().unwrap());
        let rel = max_rel(&got, &want);
        assert!(rel < 3e-2, "mlp_block rel={rel}");
    }

    #[test]
    fn decode_gemv_parity() {
        if skip() {
            return;
        }
        // M==1 single-token decode — the hot path (Bf16GemmNt native-layout dot; widening FMA).
        let (inp, out) = (2048usize, 2048usize);
        let lin = mk_linear(inp, out, 9, true);
        let x = mk_input(1, inp, 13);
        let got = linear_forward(&lin, &x).unwrap();
        let rel = max_rel(&got, &ref_linear(&lin, &x));
        assert!(rel < 1e-2, "decode_gemv rel={rel}");
    }
}
