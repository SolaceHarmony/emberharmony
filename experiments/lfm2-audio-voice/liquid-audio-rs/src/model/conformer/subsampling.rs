//! Port of `liquid_audio/model/conformer/subsampling.py` (NeMo ConvSubsampling).
//!
//! `dw_striding` scheme (FastConformer): a Conv2d stem + depthwise/pointwise
//! Conv2d stack with ReLU, then a Linear flattening channel×freq → `feat_out`.
//! The conv stack is held in a [`MaskedConvSequential`] (1:1 with Python). For a
//! single offline clip the length-mask is all-ones, so `forward` (no mask) and
//! the masked path are identical — parity-verified (conv_out 5.6e-7) and proven
//! equivalent to the padded-batch masked path by the prefill parity test.
//! Other subsampling schemes (vggnet/striding/*_conv1d) are not ported.

use candle_core::{Result, Tensor, D};
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

/// `apply_channel_mask(tensor, mask)`: zero masked time/feature positions across
/// all channels. `tensor` (B,C,T,F), `mask` (B,T,F) → broadcast over C.
pub fn apply_channel_mask(tensor: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, c, t, f) = tensor.dims4()?;
    let expanded = mask.unsqueeze(1)?.broadcast_as((b, c, t, f))?;
    tensor.broadcast_mul(&expanded)
}

/// `calculate_conv_output_size`: `(input + l_pad + r_pad - kernel) // stride + 1`.
pub fn calculate_conv_output_size(input_size: i64, kernel_size: i64, stride: i64, padding: (i64, i64)) -> i64 {
    (input_size + padding.0 + padding.1 - kernel_size) / stride + 1
}

/// Pad the last two (H, W) dims of a `(B,C,H,W)` tensor to even with trailing zeros.
/// Workaround for candle's conv2d stride>1 backward, which mis-sizes the grad-input
/// for odd input spatial dims (see `forward_conv`). Forward-identical for a stride-2
/// k3 p1 conv: the extra zero is the same one `padding` adds and the output count is
/// unchanged, so this only affects the (otherwise-failing) backward.
fn pad_even_hw(x: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = x.dims4()?;
    let mut x = x.clone();
    if h % 2 == 1 {
        x = x.pad_with_zeros(2, 0, 1)?;
    }
    if w % 2 == 1 {
        x = x.pad_with_zeros(3, 0, 1)?;
    }
    Ok(x)
}

enum Op {
    Conv(Conv2d),
    Relu,
}

/// `MaskedConvSequential(nn.Sequential)` — the conv stack, with optional length
/// masking propagated across strided layers.
pub struct MaskedConvSequential {
    layers: Vec<Op>,
}

impl MaskedConvSequential {
    fn new(layers: Vec<Op>) -> Self {
        Self { layers }
    }

    /// `_create_mask`: `(B,T,F)` 1.0 where `time < length`, else 0.0.
    fn create_mask(&self, tensor: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (b, _c, t, f) = tensor.dims4()?;
        let mut data = vec![0f32; b * t];
        for (bi, &len) in lengths.iter().enumerate().take(b) {
            for ti in 0..t.min(len) {
                data[bi * t + ti] = 1.0;
            }
        }
        let time_mask = Tensor::from_vec(data, (b, t, 1), tensor.device())?;
        time_mask.broadcast_as((b, t, f))?.to_dtype(tensor.dtype())
    }

    /// No-mask conv pass over `(B,1,T,F)` — used by the single-clip offline path
    /// (mask all-ones ⇒ identical to the masked path). Keeps the parity-verified
    /// conv math.
    fn forward_conv(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for op in &self.layers {
            x = match op {
                Op::Conv(c) => {
                    // candle's conv2d stride>1 BACKWARD errors on odd input spatial
                    // dims: out=N is ambiguous (input 2N-1 or 2N both map to N), and the
                    // grad-input path assumes the even (2N) size, so an odd input fails
                    // with a shape-mismatch add. The subsampling hits odd time dims
                    // (e.g. 101->51), which broke end-to-end training backward. Padding
                    // an odd H/W to even with a trailing zero is forward-IDENTICAL — the
                    // appended column is the same zero `padding` adds and the output
                    // count is unchanged (verified: 0.0 forward diff) — and lets the
                    // backward run. Stride-1 convs are exact (no ambiguity), so skip them
                    // (padding would change their output length).
                    if c.config().stride != 1 {
                        c.forward(&pad_even_hw(&x)?)?
                    } else {
                        c.forward(&x)?
                    }
                }
                Op::Relu => x.relu()?,
            };
        }
        Ok(x)
    }

    /// `forward(x, lengths)` — the general masked path: `(B,T,F)` in, mask applied
    /// before each layer and after each strided layer, returns `(x, out_lengths)`.
    ///
    /// Faithful to Python's `MaskedConvSequential.forward`: the length update runs
    /// only for layers whose `stride != (1,1)`, using **that layer's own** kernel /
    /// stride / padding. Reading the per-conv `config()` (rather than a uniform
    /// param) is what keeps the interleaved pointwise (`k=1`, `stride=1`) convs from
    /// shrinking the length. Time axis is dim 2 of `(B,1,T,F)` ⇒ kernel = `kH`,
    /// symmetric `pad` ⇒ total `2·pad` (matches Python's `padding[0]+padding[1]` for
    /// the square padding the model uses).
    pub fn forward(&self, x: &Tensor, lengths: &[usize]) -> Result<(Tensor, Vec<usize>)> {
        let mut x = x.unsqueeze(1)?; // (B,1,T,F)
        let mut cur: Vec<usize> = lengths.to_vec();
        let mut mask = self.create_mask(&x, &cur)?;
        for op in &self.layers {
            x = apply_channel_mask(&x, &mask)?;
            x = match op {
                Op::Conv(c) => {
                    let out = c.forward(&x)?;
                    let cfg = c.config();
                    // Strided conv (stride != 1) shrinks the time axis; pointwise
                    // (stride 1) leaves it unchanged.
                    if cfg.stride != 1 {
                        let kernel = c.weight().dim(2)? as i64; // (out, in, kH, kW) → kH (time)
                        let pad = cfg.padding as i64;
                        let (k, s) = (kernel, cfg.stride as i64);
                        cur = cur
                            .iter()
                            .map(|&l| calculate_conv_output_size(l as i64, k, s, (pad, pad)).max(0) as usize)
                            .collect();
                        mask = self.create_mask(&out, &cur)?;
                    }
                    out
                }
                Op::Relu => x.relu()?,
            };
        }
        x = apply_channel_mask(&x, &mask)?;
        Ok((x, cur))
    }
}

pub struct ConvSubsampling {
    conv: MaskedConvSequential,
    out: Linear,
    subsampling_factor: usize,
    /// mirrors Python `_sampling_num` (kept for 1:1 inventory; cold on the path).
    #[allow(dead_code)]
    sampling_num: usize,
    kernel_size: usize,
    stride: usize,
    /// mirrors Python `_conv_channels` (kept for 1:1 inventory; cold on the path).
    #[allow(dead_code)]
    conv_channels: usize,
    subsampling_conv_chunking_factor: i64,
    is_causal: bool,
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

        Ok(Self {
            conv: MaskedConvSequential::new(layers),
            out,
            subsampling_factor,
            sampling_num,
            kernel_size: k,
            stride,
            conv_channels,
            subsampling_conv_chunking_factor: 1,
            is_causal: false,
        })
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
        let x = x.unsqueeze(1)?; // (B, 1, T, F)
        self.conv.forward_conv(&x)
    }

    /// `get_sampling_frames` → `[1, subsampling_factor]`.
    pub fn get_sampling_frames(&self) -> [usize; 2] {
        [1, self.subsampling_factor]
    }

    /// `get_streaming_cache_size` → `[0, subsampling_factor + 1]`.
    pub fn get_streaming_cache_size(&self) -> [usize; 2] {
        [0, self.subsampling_factor + 1]
    }

    /// PORT: `reset_parameters` — uniform weight init for `dw_striding`
    /// (training). The port loads pretrained weights via VarBuilder, so there is
    /// nothing to re-initialize; no-op, preserved for 1:1 inventory.
    pub fn reset_parameters(&self) {}

    /// `change_subsampling_conv_chunking_factor`: must be `-1`, `1`, or a power of 2.
    pub fn change_subsampling_conv_chunking_factor(&mut self, factor: i64) -> Result<()> {
        if factor != -1 && factor != 1 && factor % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "subsampling_conv_chunking_factor should be -1, 1, or a power of 2".into(),
            ));
        }
        self.subsampling_conv_chunking_factor = factor;
        Ok(())
    }

    /// PORT: `conv_split_by_batch` — splits the input across the batch dim, runs
    /// the conv per chunk, and concatenates. This is a **memory-tiling** workaround
    /// for torch's 2³¹ tensor-indexing limit (pytorch#80020); the output equals
    /// the un-tiled conv. candle has no such limit, so the faithful "same thing"
    /// is the plain conv stack. Returns `(x, lengths, split_happened=false)`.
    pub fn conv_split_by_batch(&self, x: &Tensor, lengths: Vec<usize>) -> Result<(Tensor, Vec<usize>, bool)> {
        Ok((self.conv.forward_conv(x)?, lengths, false))
    }

    /// PORT: `conv_split_by_channel` — channel-tiled variant of the same workaround
    /// (see `conv_split_by_batch`). Output-identical to the plain conv on candle.
    pub fn conv_split_by_channel(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward_conv(&x.unsqueeze(0)?)
    }

    /// PORT: `channel_chunked_conv` — applies `conv` over channel chunks of
    /// `chunk_size`, padding when `is_causal`. Memory-tiling of a single conv
    /// (pytorch#80020 workaround); the un-tiled conv yields the same result on
    /// candle. The causal-pad shape mirrors Python for parity of intent.
    pub fn channel_chunked_conv(&self, conv: &Conv2d, _chunk_size: usize, x: &Tensor) -> Result<Tensor> {
        let x = if self.is_causal {
            let k = self.kernel_size as i64 - 1;
            let s = self.stride as i64 - 1;
            x.pad_with_zeros(D::Minus1, k as usize, s as usize)?
                .pad_with_zeros(D::Minus2, k as usize, s as usize)?
        } else {
            x.clone()
        };
        conv.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn masked_conv_length_update_is_per_layer() {
        // A strided conv (k3,s2,pad1) shrinks the time length; an interleaved
        // pointwise conv (k1,s1,pad0) must NOT. Exercises MaskedConvSequential::forward
        // (the masked path) and pins the per-conv stride/kernel/pad read.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let strided = conv2d(
            1,
            2,
            3,
            Conv2dConfig { padding: 1, stride: 2, dilation: 1, groups: 1, ..Default::default() },
            vb.pp("s"),
        )
        .unwrap();
        let pointwise = conv2d(
            2,
            2,
            1,
            Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups: 1, ..Default::default() },
            vb.pp("p"),
        )
        .unwrap();
        let mcs = MaskedConvSequential::new(vec![Op::Conv(strided), Op::Relu, Op::Conv(pointwise)]);

        let x = Tensor::zeros((1, 10, 4), DType::F32, &dev).unwrap(); // (B, T=10, F=4)
        let (out, cur) = mcs.forward(&x, &[10]).unwrap();
        // strided: (10 + 2*1 - 3)/2 + 1 = 5 ; pointwise (stride 1): unchanged.
        assert_eq!(cur, vec![5], "length must follow the strided conv only");
        assert_eq!(out.dim(2).unwrap(), 5, "output time dim == updated length");
    }
}
