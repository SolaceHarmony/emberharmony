//! Loss extensions that backfill candle's mean-only reductions.
//!
//! candle ships [`candle_nn::loss::cross_entropy`] / [`candle_nn::loss::nll`], both
//! of which **mean-reduce** over the batch. The LFM2 audio loss needs the per-token
//! vector (`reduction="none"`) so it can apply per-codebook weights before its own
//! normalization (Python `LFM2AudioModel.forward`). This is the missing reduction,
//! written in candle's `loss.rs` style and built from candle's public ops.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops::log_softmax;

/// `torch.nn.functional.cross_entropy(inp, target, reduction="none")` — the
/// per-row negative log-likelihood `-log softmax(inp)[target]`.
///
/// `inp` is `(N, C)` unnormalized logits, `target` is `(N,)` class indices.
/// Returns `(N,)`; an empty input returns `(0,)`. Computed in f32 (the reduction
/// is precision-sensitive and the Python preprocessor likewise upcasts), matching
/// the numerics the parity suite verified. The `-100`/`ignore_index` case never
/// arises here: callers restrict logits/labels to supervised positions upstream.
pub fn cross_entropy_none(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let n = inp.dim(0)?;
    if n == 0 {
        return Tensor::zeros((0,), DType::F32, inp.device());
    }
    let logp = log_softmax(&inp.to_dtype(DType::F32)?, D::Minus1)?; // (N, C)
    let target = target.to_dtype(DType::U32)?;
    // gather each row's true-class log-prob, negate → per-row NLL.
    let picked = logp.gather(&target.unsqueeze(1)?, D::Minus1)?.squeeze(1)?; // (N,)
    picked.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn matches_manual_nll() {
        // 2 rows, 3 classes; labels [0, 2].
        let logits = Tensor::from_vec(vec![2.0f32, 1.0, 0.1, 0.0, 0.0, 3.0], (2, 3), &Device::Cpu).unwrap();
        let labels = Tensor::from_vec(vec![0u32, 2u32], (2,), &Device::Cpu).unwrap();
        let loss: Vec<f32> = cross_entropy_none(&logits, &labels).unwrap().to_vec1().unwrap();
        let nll = |v: &[f32], k: usize| {
            let m = v.iter().cloned().fold(f32::MIN, f32::max);
            let denom: f32 = v.iter().map(|x| (x - m).exp()).sum();
            -((v[k] - m).exp() / denom).ln()
        };
        assert!((loss[0] - nll(&[2.0, 1.0, 0.1], 0)).abs() < 1e-5);
        assert!((loss[1] - nll(&[0.0, 0.0, 3.0], 2)).abs() < 1e-5);
    }

    #[test]
    fn empty_is_empty() {
        let logits = Tensor::zeros((0, 5), DType::F32, &Device::Cpu).unwrap();
        let labels = Tensor::zeros((0,), DType::U32, &Device::Cpu).unwrap();
        assert_eq!(cross_entropy_none(&logits, &labels).unwrap().dim(0).unwrap(), 0);
    }
}
