//! Differentiable norm layers.
//!
//! candle_nn's `LayerNorm`/`RmsNorm`/`BatchNorm` `forward` takes a fused
//! `apply_op*_no_bwd` fast path on contiguous inputs that **severs autograd** — fine
//! for inference, but it gives ZERO gradient during training (verified: backbone /
//! conformer attention + norm params got no grad). candle does expose the basic-op,
//! differentiable equivalents (`ops::layer_norm_slow` / `ops::rms_norm_slow`) — these
//! thin layers wrap them (SAME forward), so the trainable graph keeps its gradients.
//! Reused rather than re-derived; only the *layer* wrapper is local (candle ships the
//! fused layer but not a differentiable one).

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// LayerNorm (weight + bias) over the last dim, via `ops::layer_norm_slow`.
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?, bias: vb.get(dim, "bias")?, eps: eps as f32 })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm_slow(x, &self.weight, &self.bias, self.eps)
    }
}

/// `layer_norm(dim, eps, vb)` — mirrors `candle_nn::layer_norm`'s constructor but
/// builds the differentiable wrapper.
pub fn layer_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    LayerNorm::new(dim, eps, vb)
}
