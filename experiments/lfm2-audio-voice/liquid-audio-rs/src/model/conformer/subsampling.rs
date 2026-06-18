//! Port of `liquid_audio/model/conformer/subsampling.py` (NeMo ConvSubsampling).
//!
//! Inference path for the **`dw_striding`** scheme (what FastConformer uses):
//! a Conv2d stem + depthwise/pointwise Conv2d stack with ReLU, then a Linear that
//! flattens channel×freq → `feat_out`. The conv-chunking optimizations
//! (`conv_split_by_batch/channel`) and the length-masking (`MaskedConvSequential`)
//! are not ported — for a single offline clip the mask is all-ones (identity).
//! Other subsampling schemes (vggnet/striding/*_conv1d) are not ported.

use candle_core::{Result, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};

/// Faithful to `calc_length`: output length after `repeat_num` strided convs.
pub fn calc_length(length: usize, all_paddings: i64, kernel_size: i64, stride: i64, ceil_mode: bool, repeat_num: usize) -> usize {
    let add_pad = (all_paddings - kernel_size) as f64;
    let mut l = length as f64;
    for _ in 0..repeat_num {
        l = (l + add_pad) / stride as f64 + 1.0;
        l = if ceil_mode { l.ceil() } else { l.floor() };
    }
    l as usize
}

enum Op {
    Conv(Conv2d),
    Relu,
}

pub struct ConvSubsampling {
    layers: Vec<Op>,
    out: Linear,
}

impl ConvSubsampling {
    /// `dw_striding` builder. `feat_in` = mel bins, `feat_out` = encoder d_model.
    pub fn new(subsampling_factor: usize, feat_in: usize, feat_out: usize, conv_channels: usize, vb: VarBuilder) -> Result<Self> {
        let sampling_num = (subsampling_factor as f64).log2() as usize;
        let stride = 2usize;
        let k = 3usize;
        let pad = (k - 1) / 2; // symmetric, non-causal
        let dw_cfg = |groups: usize| Conv2dConfig { padding: pad, stride, dilation: 1, groups, ..Default::default() };
        let pw_cfg = Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() };

        let conv = vb.pp("conv");
        let mut layers = Vec::new();
        let mut idx = 0usize;

        // Layer 1: Conv2d(1 -> conv_channels), ReLU
        layers.push(Op::Conv(conv2d(1, conv_channels, k, dw_cfg(1), conv.pp(idx.to_string()))?));
        idx += 1;
        layers.push(Op::Relu);
        idx += 1;

        for _ in 0..(sampling_num - 1) {
            // depthwise
            layers.push(Op::Conv(conv2d(conv_channels, conv_channels, k, dw_cfg(conv_channels), conv.pp(idx.to_string()))?));
            idx += 1;
            // pointwise (1x1)
            layers.push(Op::Conv(conv2d(conv_channels, conv_channels, 1, pw_cfg, conv.pp(idx.to_string()))?));
            idx += 1;
            layers.push(Op::Relu);
            idx += 1;
        }

        let all_paddings = (2 * pad) as i64;
        let out_freq = calc_length(feat_in, all_paddings, k as i64, stride as i64, false, sampling_num);
        let out = linear(conv_channels * out_freq, feat_out, vb.pp("out"))?;

        Ok(Self { layers, out })
    }

    /// `(B, T, feat_in)` → `(B, T', feat_out)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.forward_conv(x)?;
        // (B, C, T', F') → (B, T', C*F')
        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, c * f))?;
        self.out.forward(&x)
    }

    /// Debug: conv stack output `(B, C, T', F')` before the flatten + `out` linear.
    #[doc(hidden)]
    pub fn forward_conv(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.unsqueeze(1)?; // (B, 1, T, F)
        for op in &self.layers {
            x = match op {
                Op::Conv(c) => c.forward(&x)?,
                Op::Relu => x.relu()?,
            };
        }
        Ok(x)
    }
}
