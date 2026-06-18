//! Port of `liquid_audio/model/conformer/mha.py` (NeMo multi-head attention).
//!
//! Inference path: `RelPositionalEncoding` + `RelPositionMultiHeadAttention` via
//! the manual (non-`use_pytorch_sdpa`) branch, no cache/streaming. The base
//! `MultiHeadAttention`'s `forward_qkv` / `forward_attention` / linears are folded
//! into the rel-pos struct (the only subclass the encoder uses).

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, ops::softmax_last_dim, Linear, Module, VarBuilder};

const INF_VAL: f64 = 10000.0;

/// `RelPositionalEncoding` (Transformer-XL relative positions). Computes the
/// `(1, 2L-1, d_model)` positional table for the input length on the fly.
pub struct RelPositionalEncoding {
    d_model: usize,
    xscale: Option<f64>,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, xscale: Option<f64>) -> Self {
        Self { d_model, xscale }
    }

    /// `create_pe` for `positions = arange(L-1, -L, -1)` → `(1, 2L-1, d_model)`.
    fn create_pe(&self, length: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let pos_len = 2 * length - 1;
        let positions: Vec<f32> = (0..pos_len).map(|i| (length as i64 - 1 - i as i64) as f32).collect();
        let positions = Tensor::from_vec(positions, (pos_len, 1), device)?;
        let half = self.d_model / 2;
        let div: Vec<f32> = (0..half)
            .map(|i| (-(INF_VAL.ln() / self.d_model as f64) * (2 * i) as f64).exp() as f32)
            .collect();
        let div = Tensor::from_vec(div, (1, half), device)?;
        let angles = positions.broadcast_mul(&div)?; // (pos_len, half)
        let sin = angles.sin()?;
        let cos = angles.cos()?;
        // interleave: pe[:,0::2]=sin, pe[:,1::2]=cos
        let pe = Tensor::stack(&[sin, cos], 2)?.reshape((pos_len, self.d_model))?;
        pe.unsqueeze(0)?.to_dtype(dtype)
    }

    /// Returns `(x_scaled, pos_emb)`. With the table sized to the input length,
    /// the Python slice `pe[:, start:end]` is the whole table.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let length = x.dim(1)?;
        let pe = self.create_pe(length, x.device(), x.dtype())?;
        let x = match self.xscale {
            Some(s) => (x * s)?,
            None => x.clone(),
        };
        Ok((x, pe))
    }
}

/// `RelPositionMultiHeadAttention` (+ the base `MultiHeadAttention` pieces it uses).
pub struct RelPositionMultiHeadAttention {
    h: usize,
    d_k: usize,
    s_d_k: f64,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor, // (h, d_k)
    pos_bias_v: Tensor, // (h, d_k)
}

impl RelPositionMultiHeadAttention {
    pub fn new(n_head: usize, n_feat: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        assert!(n_feat % n_head == 0);
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
            linear_pos: linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((n_head, d_k), "pos_bias_u")?,
            pos_bias_v: vb.get((n_head, d_k), "pos_bias_v")?,
        })
    }

    /// `forward_qkv` → q,k,v each `(b, h, t, d_k)`.
    fn forward_qkv(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (nb, t1, _) = q.dims3()?;
        let t2 = k.dim(1)?;
        let q = self.linear_q.forward(q)?.reshape((nb, t1, self.h, self.d_k))?.transpose(1, 2)?;
        let k = self.linear_k.forward(k)?.reshape((nb, t2, self.h, self.d_k))?.transpose(1, 2)?;
        let v = self.linear_v.forward(v)?.reshape((nb, t2, self.h, self.d_k))?.transpose(1, 2)?;
        Ok((q.contiguous()?, k.contiguous()?, v.contiguous()?))
    }

    /// `forward_attention`. `mask` (if given) is `(b, t1, t2)` with 1.0 at masked
    /// positions; faithful to NeMo's `masked_fill(-INF)` → softmax → `masked_fill(0)`.
    fn forward_attention(&self, value: &Tensor, scores: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (nb, _h, time, _t2) = scores.dims4()?;
        let attn = match mask {
            Some(m) => {
                let m = m.unsqueeze(1)?.to_dtype(scores.dtype())?; // (b,1,t1,t2)
                let neg = (&m * (-INF_VAL))?;
                let scores = scores.broadcast_add(&neg)?;
                let attn = softmax_last_dim(&scores)?;
                let keep = (1.0 - &m)?;
                attn.broadcast_mul(&keep)?
            }
            None => softmax_last_dim(scores)?,
        };
        let x = attn.matmul(value)?; // (b,h,t1,d_k)
        let x = x.transpose(1, 2)?.reshape((nb, time, self.h * self.d_k))?;
        self.linear_out.forward(&x)
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
        let (q, k, v) = self.forward_qkv(query, key, value)?; // (b,h,t,d_k)
        let q = q.transpose(1, 2)?; // (b,t,h,d_k)

        let n_batch_pos = pos_emb.dim(0)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, (), self.h, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?; // (1,h,pos_len,d_k)

        let bias_u = self.pos_bias_u.reshape((1, 1, self.h, self.d_k))?;
        let bias_v = self.pos_bias_v.reshape((1, 1, self.h, self.d_k))?;
        let q_with_bias_u = q.broadcast_add(&bias_u)?.transpose(1, 2)?.contiguous()?; // (b,h,t,d_k)
        let q_with_bias_v = q.broadcast_add(&bias_v)?.transpose(1, 2)?.contiguous()?;

        let matrix_bd = q_with_bias_v.matmul(&p.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // (b,h,t,pos_len)
        let matrix_bd = self.rel_shift(&matrix_bd)?;

        let matrix_ac = q_with_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // (b,h,t,t2)
        let t2 = matrix_ac.dim(D::Minus1)?;
        let matrix_bd = matrix_bd.narrow(D::Minus1, 0, t2)?;
        let scores = ((matrix_ac + matrix_bd)? / self.s_d_k)?;
        self.forward_attention(&v, &scores, mask)
    }
}
