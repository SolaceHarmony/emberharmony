//! Port of `liquid_audio/model/conformer/modules.py` (NeMo conformer blocks).
//!
//! Inference path: `ConformerFeedForward`, `ConformerConvolution`,
//! `ConformerLayer` (the offline conformer uses the symmetric "same"-padded
//! depthwise conv inside `ConformerConvolution`). `CausalConv1D` (causal /
//! asymmetric padding + streaming cache) is cold on the offline path but ported
//! 1:1. Dropout is identity at inference; `norm_type='batch_norm'` → `BatchNorm1d`.

use candle_core::{Result, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, linear, ops::sigmoid, ops::silu, BatchNorm, Conv1d, Conv1dConfig,
    Linear, Module, ModuleT, VarBuilder,
};

use super::mha::{MultiHeadAttention, RelPositionMultiHeadAttention};
use crate::model::norm::{layer_norm, LayerNorm};

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

    /// PORT: `reset_parameters_ff` — Xavier/uniform weight re-initialization at
    /// construction (training). The port loads pretrained weights via VarBuilder,
    /// so there is nothing to re-initialize; no-op, preserved for 1:1 inventory.
    pub fn reset_parameters_ff(&self) {}
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
        assert!((kernel_size - 1).is_multiple_of(2));
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

    /// PORT: `reset_parameters_conv` — conv weight re-initialization at
    /// construction (training). The port loads pretrained weights, so this is a
    /// no-op, preserved for 1:1 inventory.
    pub fn reset_parameters_conv(&self) {}
}

/// Padding modes for [`CausalConv1D`], mirroring the Python `padding` arg:
/// `None` (causal), an `int` (symmetric), or a `[left, right]` pair (asymmetric).
pub enum CausalPadding {
    /// `padding=None`: causal — `left = k-1`, `right = stride-1`.
    Causal,
    /// `padding=int`: symmetric `left == right == p`.
    Symmetric(usize),
    /// `padding=[l, r]` with `l + r == k-1` (stride 1 only): asymmetric.
    Asymmetric(usize, usize),
}

/// `CausalConv1D` (`nn.Conv1d` subclass) — causal/asymmetric padding so each step
/// sees a controlled number of right/left neighbours, with a streaming cache.
///
/// PORT: cold on the offline conformer forward (which uses the symmetric depthwise
/// conv in `ConformerConvolution`); ported 1:1 for inventory. Padding is applied
/// manually (the inner `Conv1d` is `padding=0`), faithful to NeMo's `F.pad` + conv.
pub struct CausalConv1D {
    conv: Conv1d,
    left_padding: usize,
    right_padding: usize,
    /// trailing steps dropped from the streaming cache (`0` ⇒ keep all).
    cache_drop_size: usize,
}

impl CausalConv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: CausalPadding,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (left_padding, right_padding) = match padding {
            CausalPadding::Causal => (kernel_size - 1, stride.saturating_sub(1)),
            CausalPadding::Symmetric(p) => {
                if stride != 1 && p != kernel_size - 1 {
                    return Err(candle_core::Error::Msg("No striding allowed for non-symmetric convolutions!".into()));
                }
                (p, p)
            }
            CausalPadding::Asymmetric(l, r) => {
                if l + r != kernel_size - 1 {
                    return Err(candle_core::Error::Msg(format!("Invalid padding param: [{l}, {r}]!")));
                }
                (l, r)
            }
        };
        let cfg = Conv1dConfig { padding: 0, stride, dilation: 1, groups, ..Default::default() };
        Ok(Self { conv: conv1d(in_channels, out_channels, kernel_size, cfg, vb)?, left_padding, right_padding, cache_drop_size: 0 })
    }

    /// `update_cache(x, cache)` → `(padded_x, next_cache)`. Offline (`cache=None`):
    /// pad left+right. Streaming: pad right only, prepend cache, roll the window
    /// back to the cache length (dropping `cache_drop_size` trailing steps).
    pub fn update_cache(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        match cache {
            None => Ok((x.pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?, None)),
            Some(c) => {
                let new_x = x.pad_with_zeros(D::Minus1, 0, self.right_padding)?;
                let new_x = Tensor::cat(&[c, &new_x], D::Minus1)?;
                let total = new_x.dim(D::Minus1)?;
                let kept = if self.cache_drop_size > 0 { total - self.cache_drop_size } else { total };
                let next = new_x.narrow(D::Minus1, 0, kept)?;
                let clen = c.dim(D::Minus1)?;
                let nlen = next.dim(D::Minus1)?;
                let start = nlen.saturating_sub(clen);
                Ok((new_x, Some(next.narrow(D::Minus1, start, nlen - start)?)))
            }
        }
    }

    /// `forward(x, cache)` = `update_cache` then the (padding=0) conv.
    pub fn forward(&self, x: &Tensor, cache: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let (x, next_cache) = self.update_cache(x, cache)?;
        Ok((self.conv.forward(&x)?, next_cache))
    }
}

/// `ConformerLayer`: FF/2 → self-attn (rel-pos) → conv → FF/2, pre-LayerNorm with
/// residuals (`fc_factor = 0.5` on the feed-forwards), then output LayerNorm.
/// `self_attention_model` selects the layer's attention, faithful to NeMo's
/// ConformerLayer (which holds one of these as `self.self_attn`):
/// * `rel_pos` — Transformer-XL relative-position attention (the LFM2.5-Audio model's
///   config; takes the rel `pos_emb`).
/// * `abs_pos` — standard scaled-dot-product attention; the encoder instead adds an
///   absolute `PositionalEncoding` to the input and the layer passes NO `pos_emb`.
///
/// The `abs_pos` variant needs an abs_pos checkpoint (base MHA — no `linear_pos` /
/// `pos_bias_*`), which the LFM2.5-Audio model is not, so it is never constructed here;
/// the attention itself is verified by `rel_pos_attention_sdpa_parity`'s sibling
/// `abs_attention_parity`. (The encoder-side absolute pos-enc swap is documented in
/// `ConformerEncoder` — only `rel_pos` is wired in this inference port.)
enum SelfAttention {
    RelPos(RelPositionMultiHeadAttention),
    Abs(MultiHeadAttention),
}

pub struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: ConformerFeedForward,
    norm_self_att: LayerNorm,
    self_attn: SelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        use_bias: bool,
        self_attention_model: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = match self_attention_model {
            "rel_pos" => SelfAttention::RelPos(RelPositionMultiHeadAttention::new(n_heads, d_model, use_bias, vb.pp("self_attn"))?),
            "abs_pos" => SelfAttention::Abs(MultiHeadAttention::new(n_heads, d_model, use_bias, vb.pp("self_attn"))?),
            other => candle_core::bail!("ConformerLayer: unsupported self_attention_model '{other}' (rel_pos | abs_pos)"),
        };
        Ok(Self {
            norm_feed_forward1: layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?,
            feed_forward1: ConformerFeedForward::new(d_model, d_ff, vb.pp("feed_forward1"))?,
            norm_self_att: layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?,
            self_attn,
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
        // rel_pos passes the relative `pos_emb`; abs_pos uses standard attention with no
        // pos_emb (the encoder would have added an absolute PositionalEncoding upstream).
        let h = match &self.self_attn {
            SelfAttention::RelPos(a) => a.forward(&h, &h, &h, att_mask, pos_emb)?,
            SelfAttention::Abs(a) => a.forward(&h, &h, &h, att_mask)?,
        };
        let residual = (residual + h)?;

        let h = self.conv.forward(&self.norm_conv.forward(&residual)?, pad_mask)?;
        let residual = (residual + h)?;

        let h = self.feed_forward2.forward(&self.norm_feed_forward2.forward(&residual)?)?;
        let residual = (residual + (h * FC)?)?;

        self.norm_out.forward(&residual)
    }
}
