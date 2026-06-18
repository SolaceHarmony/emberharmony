//! Port of `liquid_audio/model/conformer/modules.py` (NeMo conformer blocks).
//!
//! Inference path: `ConformerFeedForward`, `ConformerConvolution`,
//! `ConformerLayer`. `CausalConv1D` collapses to a symmetric "same"-padded
//! depthwise conv (`_left_padding == _right_padding == (k-1)/2`, the conformer's
//! config) — the cache/streaming branch is not ported. Dropout is identity at
//! inference; `norm_type='batch_norm'` (the default) → `BatchNorm1d`.

use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv1d, layer_norm, linear, ops::sigmoid, ops::silu, BatchNorm, Conv1d, Conv1dConfig, LayerNorm,
    Linear, Module, ModuleT, VarBuilder,
};

use super::mha::RelPositionMultiHeadAttention;

/// `ConformerFeedForward`: Linear → SiLU → Linear.
pub struct ConformerFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl ConformerFeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(d_model, d_ff, vb.pp("linear1"))?,
            linear2: linear(d_ff, d_model, vb.pp("linear2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = silu(&x)?;
        self.linear2.forward(&x)
    }
}

/// `ConformerConvolution`: pointwise→GLU→(pad mask)→depthwise→BatchNorm→SiLU→pointwise.
pub struct ConformerConvolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: BatchNorm,
    pointwise_conv2: Conv1d,
}

impl ConformerConvolution {
    pub fn new(d_model: usize, kernel_size: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        assert!((kernel_size - 1) % 2 == 0);
        let context = (kernel_size - 1) / 2;
        let pw = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() };
        let dw = Conv1dConfig { padding: context, stride: 1, dilation: 1, groups: d_model, ..Default::default() };
        // candle's conv1d always loads a bias; the conformer uses use_bias=true.
        debug_assert!(use_bias);
        Ok(Self {
            pointwise_conv1: conv1d(d_model, d_model * 2, 1, pw, vb.pp("pointwise_conv1"))?,
            depthwise_conv: conv1d(d_model, d_model, kernel_size, dw, vb.pp("depthwise_conv"))?,
            batch_norm: batch_norm(d_model, 1e-5, vb.pp("batch_norm"))?,
            pointwise_conv2: conv1d(d_model, d_model, 1, pw, vb.pp("pointwise_conv2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // (B, d_model, T)
        let x = self.pointwise_conv1.forward(&x)?; // (B, 2*d_model, T)
        // GLU over channel dim 1: a * sigmoid(b)
        let c = x.dim(1)?;
        let a = x.narrow(1, 0, c / 2)?;
        let b = x.narrow(1, c / 2, c / 2)?;
        let mut x = (a * sigmoid(&b)?)?;

        if let Some(pm) = pad_mask {
            // pad_mask (B, T) true where padded → zero those time steps
            let keep = (1.0 - pm.unsqueeze(1)?.to_dtype(x.dtype())?)?; // (B,1,T)
            x = x.broadcast_mul(&keep)?;
        }

        let x = self.depthwise_conv.forward(&x)?;
        let x = self.batch_norm.forward_t(&x, false)?;
        let x = silu(&x)?;
        let x = self.pointwise_conv2.forward(&x)?;
        x.transpose(1, 2)?.contiguous()
    }
}

/// `ConformerLayer`: FF/2 → self-attn (rel-pos) → conv → FF/2, pre-LayerNorm with
/// residuals (`fc_factor = 0.5` on the feed-forwards), then output LayerNorm.
pub struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: ConformerFeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPositionMultiHeadAttention,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn new(d_model: usize, d_ff: usize, n_heads: usize, conv_kernel_size: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_feed_forward1: layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?,
            feed_forward1: ConformerFeedForward::new(d_model, d_ff, vb.pp("feed_forward1"))?,
            norm_self_att: layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?,
            self_attn: RelPositionMultiHeadAttention::new(n_heads, d_model, use_bias, vb.pp("self_attn"))?,
            norm_conv: layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?,
            conv: ConformerConvolution::new(d_model, conv_kernel_size, use_bias, vb.pp("conv"))?,
            norm_feed_forward2: layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?,
            feed_forward2: ConformerFeedForward::new(d_model, d_ff, vb.pp("feed_forward2"))?,
            norm_out: layer_norm(d_model, 1e-5, vb.pp("norm_out"))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        att_mask: Option<&Tensor>,
        pos_emb: &Tensor,
        pad_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        const FC: f64 = 0.5;
        let residual = x.clone();
        let h = self.feed_forward1.forward(&self.norm_feed_forward1.forward(&residual)?)?;
        let residual = (residual + (h * FC)?)?;

        let h = self.norm_self_att.forward(&residual)?;
        let h = self.self_attn.forward(&h, &h, &h, att_mask, pos_emb)?;
        let residual = (residual + h)?;

        let h = self.conv.forward(&self.norm_conv.forward(&residual)?, pad_mask)?;
        let residual = (residual + h)?;

        let h = self.feed_forward2.forward(&self.norm_feed_forward2.forward(&residual)?)?;
        let residual = (residual + (h * FC)?)?;

        self.norm_out.forward(&residual)
    }
}
