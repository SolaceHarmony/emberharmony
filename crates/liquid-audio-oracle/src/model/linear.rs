//! Differentiable Candle linear helpers for the offline training oracle.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

pub fn linear_forward(linear: &Linear, input: &Tensor) -> Result<Tensor> {
    linear.forward(input)
}

pub fn linear_logits(weight: &Tensor, input: &Tensor) -> Result<Tensor> {
    input.matmul(&weight.t()?)?.to_dtype(DType::F32)
}
