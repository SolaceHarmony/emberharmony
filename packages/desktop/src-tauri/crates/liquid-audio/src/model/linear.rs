use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv2d, Linear, Module};

use crate::bf16_gemm::bf16_matmul;

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
        let Some(y) = bf16_matmul(x, &weight.t()?)? else {
            candle_core::bail!(
                "CPU bf16 linear requested but the NEON BFMMLA kernel is unavailable"
            );
        };
        y
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
