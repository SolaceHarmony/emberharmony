//! Port of `liquid_audio/model/conformer/mha.py` (NeMo multi-head attention).
//!
//! Faithful class layout — Python uses inheritance:
//!   `PositionalEncoding`  ← `RelPositionalEncoding`
//!   `MultiHeadAttention`  ← `RelPositionMultiHeadAttention`
//! Rust has no inheritance, so each subclass **composes** its base (`base: …`)
//! and calls the base's methods (`forward_qkv` / `forward_attention` /
//! `create_pe`), exactly where Python calls `super()`. Both struct pairs live in
//! this module, so the subclass reads the base's private fields directly.
//!
//! Inference path = the rel-pos subclasses via the manual (non-`use_pytorch_sdpa`)
//! branch, no cache/streaming. `dropout` / `dropout_emb` are eval-time identities
//! and are omitted.
//!
//! The `use_pytorch_sdpa=True` branch (both base and rel-pos) is algebraically the
//! SAME computation as the manual path, not a separate algorithm: it pre-scales
//! `matrix_bd` by `1/√d`, bakes the mask in as an additive `-INF`, and lets
//! `scaled_dot_product_attention` fold in `q·kᵀ/√d`, reconstructing the manual
//! `softmax((matrix_ac+matrix_bd)/√d)·v`; its explicit all-masked-row zeroing matches
//! `forward_attention`'s post-softmax `masked_fill(mask, 0)`. So the manual path here
//! faithfully implements BOTH Python branches — verified by `rel_pos_attention_sdpa_-
//! parity` against a Python `use_pytorch_sdpa=True` golden (1.07e-6; the Python
//! True-vs-False diff is itself 1.5e-7). candle DOES ship a fused `ops::sdpa`, but it
//! is `apply_op*_no_bwd` (severs autograd, cf. d2f4a80) and would break the trainable
//! audio-in encode graph, so it is deliberately avoided.

use candle_core::{DType, Device, Result, Tensor, D};
// `ops::softmax` (differentiable basic ops), NOT `softmax_last_dim` (fused
// `apply_op1_no_bwd`, severs autograd): the conformer attention runs in the trainable
// `logits` graph (audio-in encode). Same forward values.
use candle_nn::{linear, linear_no_bias, ops::softmax, Linear, Module, VarBuilder};

const INF_VAL: f64 = 10000.0;

/// NeMo masked softmax: `scores.masked_fill(mask, -INF_VAL) → softmax(-1) →
/// .masked_fill(mask, 0.0)`. `mask` is `(b, t1, t2)` nonzero at masked positions
/// (broadcast over heads); `None` ⇒ plain softmax. Uses `where_cond` to SET, so it
/// is bit-identical to torch `masked_fill` (not an additive approximation).
fn masked_softmax(scores: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    match mask {
        Some(m) => {
            let cond = m.unsqueeze(1)?.broadcast_as(scores.dims())?.to_dtype(DType::U8)?.contiguous()?;
            let neg = Tensor::full(-INF_VAL as f32, scores.dims(), scores.device())?.to_dtype(scores.dtype())?;
            let masked = cond.where_cond(&neg, scores)?; // masked_fill(mask, -INF_VAL)
            let attn = softmax(&masked, D::Minus1)?;
            cond.where_cond(&attn.zeros_like()?, &attn) // .masked_fill(mask, 0.0)
        }
        None => softmax(scores, D::Minus1),
    }
}

/// `PositionalEncoding` (base) — fixed sinusoidal positional encoding.
///
/// Python registers the table as a non-persistent buffer via `create_pe` /
/// `extend_pe`; here those return the table (no mutable buffer state), which the
/// callers size to the input length. `dropout`/`dropout_emb` are eval identities.
pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_len: usize,
    pub xscale: Option<f64>,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, xscale: Option<f64>) -> Self {
        Self { d_model, max_len, xscale }
    }

    /// `create_pe(positions, dtype)` → `(1, pos_length, d_model)`:
    /// `pe[:,0::2]=sin(positions*div)`, `pe[:,1::2]=cos(positions*div)` with
    /// `div = exp(arange(0,d_model,2) * -(ln(INF_VAL)/d_model))`.
    pub fn create_pe(&self, positions: &Tensor, dtype: DType) -> Result<Tensor> {
        let device = positions.device();
        let pos_length = positions.elem_count();
        let half = self.d_model / 2;
        let div: Vec<f32> = (0..half)
            .map(|i| (-(INF_VAL.ln() / self.d_model as f64) * (2 * i) as f64).exp() as f32)
            .collect();
        let div = Tensor::from_vec(div, (1, half), device)?;
        let positions = positions.reshape((pos_length, 1))?.to_dtype(DType::F32)?;
        let angles = positions.broadcast_mul(&div)?; // (pos_length, half)
        let sin = angles.sin()?;
        let cos = angles.cos()?;
        // interleave: pe[:,0::2]=sin, pe[:,1::2]=cos
        let pe = Tensor::stack(&[sin, cos], 2)?.reshape((pos_length, self.d_model))?;
        pe.unsqueeze(0)?.to_dtype(dtype)
    }

    /// `extend_pe(length, ...)` for the base = absolute positions `arange(0,length)`.
    pub fn extend_pe(&self, length: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let positions: Vec<f32> = (0..length).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (length,), device)?;
        self.create_pe(&positions, dtype)
    }

    /// `forward(x, cache_len=0)` for the base = ADD absolute pos-emb to `x`.
    /// Returns `(x + pos_emb, pos_emb)`.
    pub fn forward(&self, x: &Tensor, cache_len: usize) -> Result<(Tensor, Tensor)> {
        let input_len = x.dim(1)? + cache_len;
        let x = match self.xscale {
            Some(s) => (x * s)?,
            None => x.clone(),
        };
        let pe = self.extend_pe(input_len, x.device(), x.dtype())?;
        let pos_emb = pe.narrow(1, 0, input_len)?;
        let x = x.broadcast_add(&pos_emb)?;
        Ok((x, pos_emb))
    }
}

/// `RelPositionalEncoding(PositionalEncoding)` — Transformer-XL relative
/// positions. Overrides `extend_pe` (positions `arange(L-1, -L, -1)` → `2L-1`)
/// and `forward` (does **not** add to `x`); reuses the base `create_pe`.
pub struct RelPositionalEncoding {
    base: PositionalEncoding,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, xscale: Option<f64>) -> Self {
        Self { base: PositionalEncoding::new(d_model, 5000, xscale) }
    }

    /// override `extend_pe`: `positions = arange(length-1, -length, -1)` →
    /// `(2L-1,)`, then the inherited `create_pe` builds the interleaved table.
    fn extend_pe(&self, length: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let pos_len = 2 * length - 1;
        let positions: Vec<f32> = (0..pos_len).map(|i| (length as i64 - 1 - i as i64) as f32).collect();
        let positions = Tensor::from_vec(positions, (pos_len,), device)?;
        self.base.create_pe(&positions, dtype)
    }

    /// override `forward`: rel-pos does NOT add to `x`. With the table sized to
    /// the input length, Python's slice `pe[:, start:end]` is the whole table.
    /// Returns `(x * xscale, pos_emb)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let length = x.dim(1)?;
        let pe = self.extend_pe(length, x.device(), x.dtype())?;
        let x = match self.base.xscale {
            Some(s) => (x * s)?,
            None => x.clone(),
        };
        Ok((x, pe))
    }
}

/// `MultiHeadAttention` (base) — q/k/v/out projections + scaled-dot-product
/// attention (the manual, non-`use_pytorch_sdpa` path).
pub struct MultiHeadAttention {
    h: usize,
    d_k: usize,
    s_d_k: f64,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    /// `cache_drop_size` for streaming `update_cache`; 0 on the offline path.
    cache_drop_size: usize,
}

impl MultiHeadAttention {
    pub fn new(n_head: usize, n_feat: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        assert!(n_feat.is_multiple_of(n_head));
        let d_k = n_feat / n_head;
        let mk = |name: &str, vb: VarBuilder| -> Result<Linear> {
            if use_bias {
                linear(n_feat, n_feat, vb.pp(name))
            } else {
                linear_no_bias(n_feat, n_feat, vb.pp(name))
            }
        };
        Ok(Self {
            h: n_head,
            d_k,
            s_d_k: (d_k as f64).sqrt(),
            linear_q: mk("linear_q", vb.clone())?,
            linear_k: mk("linear_k", vb.clone())?,
            linear_v: mk("linear_v", vb.clone())?,
            linear_out: mk("linear_out", vb.clone())?,
            cache_drop_size: 0,
        })
    }

    /// `forward_qkv` → q,k,v each `(b, h, t, d_k)`.
    pub fn forward_qkv(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (nb, t1, _) = query.dims3()?;
        let t2 = key.dim(1)?;
        let q = self.linear_q.forward(query)?.reshape((nb, t1, self.h, self.d_k))?.transpose(1, 2)?;
        let k = self.linear_k.forward(key)?.reshape((nb, t2, self.h, self.d_k))?.transpose(1, 2)?;
        let v = self.linear_v.forward(value)?.reshape((nb, t2, self.h, self.d_k))?.transpose(1, 2)?;
        Ok((q.contiguous()?, k.contiguous()?, v.contiguous()?))
    }

    /// `forward_attention`. `mask` (if given) is `(b, t1, t2)`, nonzero at masked
    /// positions; faithful to NeMo's `scores.masked_fill(mask, -INF) → softmax →
    /// .masked_fill(mask, 0)`. Uses `where_cond` to SET (not add) `-INF_VAL`, so it is
    /// bit-identical to `masked_fill` rather than a near-equivalent additive approx.
    pub fn forward_attention(&self, value: &Tensor, scores: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (nb, _h, time, _t2) = scores.dims4()?;
        let attn = masked_softmax(scores, mask)?;
        let x = attn.matmul(value)?; // (b,h,t1,d_k)
        let x = x.transpose(1, 2)?.reshape((nb, time, self.h * self.d_k))?;
        self.linear_out.forward(&x)
    }

    /// Base `forward` (standard scaled-dot-product). The encoder uses the rel-pos
    /// subclass; this is the faithful base path. `use_pytorch_sdpa=True` reduces to the
    /// same `softmax(q·kᵀ/√d + mask)·v` (see the module note) — both Python branches map
    /// here; candle's fused `ops::sdpa` is no_bwd and avoided.
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (q, k, v) = self.forward_qkv(query, key, value)?;
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / self.s_d_k)?;
        self.forward_attention(&v, &scores, mask)
    }

    /// `update_cache`: streaming KV concat. `cache=None` (offline) ⇒ no-op.
    /// Faithful to Python: `key = value = cat([cache, key]); cache = cat([cache[q_keep:], query[:q_keep]])`.
    pub fn update_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        query: &Tensor,
        cache: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        match cache {
            None => Ok((key.clone(), value.clone(), query.clone(), None)),
            Some(c) => {
                let kv = Tensor::cat(&[c, key], 1)?;
                let q_keep = query.dim(1)?.saturating_sub(self.cache_drop_size);
                let c_len = c.dim(1)?;
                let new_cache = Tensor::cat(&[&c.narrow(1, q_keep, c_len - q_keep)?, &query.narrow(1, 0, q_keep)?], 1)?;
                Ok((kv.clone(), kv, query.clone(), Some(new_cache)))
            }
        }
    }
}

/// `RelPositionMultiHeadAttention(MultiHeadAttention)` — adds the relative
/// positional projection + Transformer-XL `(u,v)` biases and overrides `forward`;
/// reuses the base `forward_qkv` / `forward_attention`.
pub struct RelPositionMultiHeadAttention {
    base: MultiHeadAttention,
    linear_pos: Linear,
    pos_bias_u: Tensor, // (h, d_k)
    pos_bias_v: Tensor, // (h, d_k)
}

impl RelPositionMultiHeadAttention {
    pub fn new(n_head: usize, n_feat: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        let base = MultiHeadAttention::new(n_head, n_feat, use_bias, vb.clone())?;
        let d_k = n_feat / n_head;
        Ok(Self {
            base,
            linear_pos: linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((n_head, d_k), "pos_bias_u")?,
            pos_bias_v: vb.get((n_head, d_k), "pos_bias_v")?,
        })
    }

    /// `rel_shift`: (b,h,qlen,pos_len) → shifted (b,h,qlen,pos_len).
    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, qlen, pos_len) = x.dims4()?;
        let x = x.pad_with_zeros(D::Minus1, 1, 0)?; // (b,h,qlen,pos_len+1)
        let x = x.reshape((b, h, pos_len + 1, qlen))?;
        let x = x.narrow(2, 1, pos_len)?.contiguous()?; // drop first row → (b,h,pos_len,qlen)
        x.reshape((b, h, qlen, pos_len))
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let (h, d_k, s_d_k) = (self.base.h, self.base.d_k, self.base.s_d_k);
        let (q, k, v) = self.base.forward_qkv(query, key, value)?; // (b,h,t,d_k)
        let q = q.transpose(1, 2)?; // (b,t,h,d_k)

        let n_batch_pos = pos_emb.dim(0)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, (), h, d_k))?
            .transpose(1, 2)?
            .contiguous()?; // (1,h,pos_len,d_k)

        let bias_u = self.pos_bias_u.reshape((1, 1, h, d_k))?;
        let bias_v = self.pos_bias_v.reshape((1, 1, h, d_k))?;
        let q_with_bias_u = q.broadcast_add(&bias_u)?.transpose(1, 2)?.contiguous()?; // (b,h,t,d_k)
        let q_with_bias_v = q.broadcast_add(&bias_v)?.transpose(1, 2)?.contiguous()?;

        let matrix_bd = q_with_bias_v.matmul(&p.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // (b,h,t,pos_len)
        let matrix_bd = self.rel_shift(&matrix_bd)?;

        let matrix_ac = q_with_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // (b,h,t,t2)
        let t2 = matrix_ac.dim(D::Minus1)?;
        let matrix_bd = matrix_bd.narrow(D::Minus1, 0, t2)?;
        let scores = ((matrix_ac + matrix_bd)? / s_d_k)?;
        self.base.forward_attention(&v, &scores, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn masked_softmax_matches_python() {
        // vs torch: scores.masked_fill(mask,-INF).softmax(-1).masked_fill(mask,0).
        let dev = Device::Cpu;
        let scores = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5], (1, 1, 2, 3), &dev).unwrap();
        let mask = Tensor::from_vec(vec![0u8, 1, 0, 1, 1, 0], (1, 2, 3), &dev).unwrap();
        let got = masked_softmax(&scores, Some(&mask)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = [0.119203f32, 0.0, 0.880797, 0.0, 0.0, 1.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "masked softmax vs Python: got {got:?} want {want:?}");
        }
        // None ⇒ plain softmax (row sums to 1, no zeros).
        let plain = masked_softmax(&scores, None).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((plain[0] + plain[1] + plain[2] - 1.0).abs() < 1e-6);
    }
}
