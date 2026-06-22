//! Port of `liquid_audio/model/transformer.py` — the LFM2 attention backbone.
//!
//! Faithful to the Python classes: `RMSNorm`, `GLU` (SwiGLU), `BoundedAttention`,
//! `MHA`, `StandardBlock`, `SharedEmbedding`, `RawLMBackbone`, plus the rotary
//! helpers `precompute_freqs_cis` / `apply_rotary_emb`.
//!
//! Notes on the candle mapping:
//! - No complex dtype in candle: `precompute_freqs_cis` returns `(cos, sin)` and
//!   rotary is applied with `candle_nn::rotary_emb::rope_i` (interleaved / GPT-J
//!   pairing), which matches the Python `view_as_complex` on adjacent pairs.
//! - `scaled_dot_product_attention(is_causal=...)` is implemented by hand
//!   (matmul + additive causal mask + softmax + matmul) with GQA head repeat.
//! - Parity (esp. attention + rotary) is to be verified against the Python with
//!   shared weights before this is trusted numerically. See PORT_STATUS.md.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, ops::softmax_last_dim, rotary_emb::rope_i, Embedding, Linear, Module, VarBuilder};

/// `head_style` for attention. Mirrors the `Literal["mha","gqa","mqa"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadStyle {
    Mha,
    Gqa,
    Mqa,
}

/// Per-layer KV cache. Mirrors `LayerKVCache`: stores key/value pre-transpose
/// (shape `[b, t, heads, head_dim]`) and concatenates new steps along dim 1.
#[derive(Default)]
pub struct LayerKvCache {
    key_cache: Option<Tensor>,
    value_cache: Option<Tensor>,
}

impl LayerKvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.key_cache {
            None => k.clone(),
            Some(prev) => Tensor::cat(&[prev, k], 1)?,
        };
        let v = match &self.value_cache {
            None => v.clone(),
            Some(prev) => Tensor::cat(&[prev, v], 1)?,
        };
        self.key_cache = Some(k.clone());
        self.value_cache = Some(v.clone());
        Ok((k, v))
    }

    pub fn get_cache_size(&self) -> usize {
        match &self.key_cache {
            None => 0,
            Some(k) => k.dim(1).unwrap_or(0),
        }
    }
}

/// `RMSNorm`. Faithful: normalize in f32 (`x * rsqrt(mean(x^2)+eps)`), multiply by
/// the weight, then cast back to the input dtype.
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(mean_sq + self.eps)?.sqrt()?)?;
        let w = self.weight.to_dtype(DType::F32)?;
        normed.broadcast_mul(&w)?.to_dtype(in_dtype)
    }
}

/// `GLU` (SwiGLU feed-forward). The `ff_dim` sizing mirrors the Python so the
/// linear shapes match the checkpoint.
pub struct Glu {
    w1: Linear,
    w2: Linear,
    w3: Option<Linear>,
    use_swiglu: bool,
}

impl Glu {
    pub fn new(
        dim: usize,
        ff_dim: Option<usize>,
        use_swiglu: bool,
        multiple_of: usize,
        ffn_dim_multiplier: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut ff = ff_dim.unwrap_or(4 * dim);
        if use_swiglu {
            ff = 2 * ff / 3;
            ff = (ffn_dim_multiplier * ff as f64) as usize;
            ff = multiple_of * ff.div_ceil(multiple_of);
        }
        let w1 = linear_no_bias(dim, ff, vb.pp("w1"))?;
        let w3 = if use_swiglu {
            Some(linear_no_bias(dim, ff, vb.pp("w3"))?)
        } else {
            None
        };
        let w2 = linear_no_bias(ff, dim, vb.pp("w2"))?;
        Ok(Self { w1, w2, w3, use_swiglu })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.use_swiglu {
            let a = candle_nn::ops::silu(&self.w1.forward(x)?)?;
            let b = self.w3.as_ref().unwrap().forward(x)?;
            self.w2.forward(&(a * b)?)
        } else {
            self.w2.forward(&self.w1.forward(x)?.gelu_erf()?)
        }
    }
}

/// Precompute rotary `(cos, sin)` of shape `[end, dim/2]`. Faithful to
/// `precompute_freqs_cis` (`polar(1, outer(t, inv_freq))`), but returned as the
/// real `cos`/`sin` candle's `rope_i` consumes.
pub fn precompute_freqs_cis(dim: usize, end: usize, theta: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32).collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let t: Vec<f32> = (0..end).map(|i| i as f32).collect();
    let t = Tensor::from_vec(t, (end, 1), device)?;
    let freqs = t.broadcast_mul(&inv_freq)?; // [end, half]
    Ok((freqs.cos()?, freqs.sin()?))
}

/// `apply_rotary_emb(xq, xk, freqs_cis)` — interleaved (GPT-J) rotary applied to
/// query and key together (Python takes/returns both).
///
/// PORT: candle has no complex dtype, so Python's `view_as_complex(...) *
/// freqs_cis → view_as_real` rotation is the real-valued `rope_i` (the exact
/// interleaved-pair rotation), with the complex `freqs_cis` table carried as
/// `(cos, sin)` from `precompute_freqs_cis`. Faithful to the upcast-rotate
/// contract: callers pass f32 q/k and cast the result back (`type_as`).
pub fn apply_rotary_emb(xq: &Tensor, xk: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    Ok((rope_i(xq, cos, sin)?, rope_i(xk, cos, sin)?))
}

/// `reshape_for_broadcast(freqs_cis, x)` — reshape the `(seq, dim/2)` freq table
/// to broadcast against `x` of rank `ndim`: size kept on dims `1` and `ndim-1`,
/// `1` elsewhere. In the candle path `rope_i` performs this broadcast internally
/// over `[b, heads, t, head_dim]`; provided for 1:1 parity with the Python helper.
pub fn reshape_for_broadcast(freqs_cis: &Tensor, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let ndim = dims.len();
    let shape: Vec<usize> = (0..ndim).map(|i| if i == 1 || i == ndim - 1 { dims[i] } else { 1 }).collect();
    freqs_cis.reshape(shape)
}

/// PORT: `wrap_activation_checkpoint` — training-only gradient (activation)
/// checkpointing (`torch.utils.checkpoint`). There is no autograd/backward pass
/// on the candle inference path, so there is nothing to checkpoint; this is an
/// identity wrapper, preserved for 1:1 inventory.
pub fn wrap_activation_checkpoint<T>(module: T) -> T {
    module
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, h, n_rep, t, d))?.reshape((b, h * n_rep, t, d))
}

/// Additive causal mask of shape `[q_len, kv_len]`. For `q_len == kv_len` this is
/// the standard lower-triangular mask; for `q_len < kv_len` (decode with cache)
/// it is `causal_lower_right`: query `i` attends to keys `<= i + (kv_len-q_len)`.
fn causal_mask(q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
    let offset = kv_len - q_len;
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            if j > i + offset {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (q_len, kv_len), device)
}

/// `BoundedAttention`. Takes already-projected q/k/v (flat last dim), reshapes to
/// heads, applies optional qk RMSNorm + rotary, updates the cache, runs masked
/// SDPA with GQA repeat, and returns the flattened output.
pub struct BoundedAttention {
    num_heads: usize,
    head_dim: usize,
    head_style: HeadStyle,
    gqa_dim: usize,
    q_layernorm: Option<RmsNorm>,
    k_layernorm: Option<RmsNorm>,
}

impl BoundedAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_style: HeadStyle,
        gqa_dim: usize,
        qk_layernorm: bool,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let (q_layernorm, k_layernorm) = if qk_layernorm {
            (
                Some(RmsNorm::new(head_dim, norm_eps, vb.pp("q_layernorm"))?),
                Some(RmsNorm::new(head_dim, norm_eps, vb.pp("k_layernorm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self { num_heads, head_dim, head_style, gqa_dim, q_layernorm, k_layernorm })
    }

    fn kv_heads(&self) -> usize {
        match self.head_style {
            HeadStyle::Mha => self.num_heads,
            HeadStyle::Mqa => 1,
            HeadStyle::Gqa => self.gqa_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut LayerKvCache>,
    ) -> Result<Tensor> {
        let (bsz, seqlen, _) = q.dims3()?;
        let kvh = self.kv_heads();

        // [b, t, heads, head_dim]
        let mut q = q.reshape((bsz, seqlen, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((bsz, seqlen, kvh, self.head_dim))?;
        let v = v.reshape((bsz, seqlen, kvh, self.head_dim))?;

        if let (Some(qn), Some(kn)) = (&self.q_layernorm, &self.k_layernorm) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }

        // rotary on [b, heads, t, head_dim]; rope_i wants cos/sin [t, head_dim/2].
        // Python `apply_rotary_emb` upcasts q/k to fp32, rotates, then casts back
        // (`type_as`). Do the same so the native bf16/f16 path matches torch and
        // `rope_i` sees fp32 operands consistent with the fp32 cos/sin tables.
        let in_dtype = q.dtype();
        let q_t = q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let k_t = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let (q_t, k_t) = apply_rotary_emb(&q_t, &k_t, cos, sin)?;
        // back to [b, t, heads, head_dim] in the original dtype for cache concat
        let q = q_t.transpose(1, 2)?.contiguous()?.to_dtype(in_dtype)?;
        let k = k_t.transpose(1, 2)?.contiguous()?.to_dtype(in_dtype)?;

        let (k, v) = match cache {
            Some(c) => c.update(&k, &v)?,
            None => (k, v),
        };

        let q_len = q.dim(1)?;
        let kv_len = k.dim(1)?;

        // [b, heads, t, head_dim]
        let query = q.transpose(1, 2)?.contiguous()?;
        let key = repeat_kv(&k.transpose(1, 2)?.contiguous()?, self.num_heads / kvh)?;
        let value = repeat_kv(&v.transpose(1, 2)?.contiguous()?, self.num_heads / kvh)?;

        // `scaled_dot_product_attention` accumulates scores, softmax, and the
        // value-weighting in fp32 regardless of input dtype; mirror that (upcast
        // q/k/v, do the math in fp32, cast the result back) so bf16/f16 matches.
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn = (query.matmul(&key.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        if q_len != 1 {
            let mask = causal_mask(q_len, kv_len, q.device())?.to_dtype(attn.dtype())?;
            attn = attn.broadcast_add(&mask)?;
        }
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&value)?.to_dtype(in_dtype)?; // [b, heads, t, head_dim], back to input dtype

        out.transpose(1, 2)?.reshape((bsz, seqlen, self.num_heads * self.head_dim))
    }
}

/// `MHA`. qkv projection, head split, rotary frequency slice (cache-aware),
/// `BoundedAttention`, output projection. Holds the precomputed rotary tables.
pub struct Mha {
    dim: usize,
    head_dim: usize,
    head_style: HeadStyle,
    gqa_dim: usize,
    qkv_proj: Linear,
    out_proj: Linear,
    attention: BoundedAttention,
    cos: Tensor,
    sin: Tensor,
}

impl Mha {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_style: HeadStyle,
        qk_layernorm: bool,
        norm_eps: f64,
        gqa_dim: usize,
        max_seq_len: usize,
        theta: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let total_width = match head_style {
            HeadStyle::Mha => 3 * dim,
            HeadStyle::Mqa => dim + 2 * head_dim,
            HeadStyle::Gqa => dim + 2 * head_dim * gqa_dim,
        };
        let qkv_proj = linear_no_bias(dim, total_width, vb.pp("qkv_proj"))?;
        let out_proj = linear_no_bias(dim, dim, vb.pp("out_proj"))?;
        let attention = BoundedAttention::new(dim, num_heads, head_style, gqa_dim, qk_layernorm, norm_eps, vb.pp("bounded_attention"))?;
        let (cos, sin) = precompute_freqs_cis(head_dim, max_seq_len, theta, vb.device())?;
        Ok(Self { dim, head_dim, head_style, gqa_dim, qkv_proj, out_proj, attention, cos, sin })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn forward(&self, x: &Tensor, cache: Option<&mut LayerKvCache>) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let x = self.qkv_proj.forward(x)?;
        let (q_w, kv_w) = match self.head_style {
            HeadStyle::Mha => (self.dim, self.dim),
            HeadStyle::Mqa => (self.dim, self.head_dim),
            HeadStyle::Gqa => (self.dim, self.head_dim * self.gqa_dim),
        };
        let xq = x.narrow(D::Minus1, 0, q_w)?;
        let xk = x.narrow(D::Minus1, q_w, kv_w)?;
        let xv = x.narrow(D::Minus1, q_w + kv_w, kv_w)?;

        // freqs slice: from cache size (decode) else from 0.
        let cache_size = cache.as_ref().map(|c| c.get_cache_size()).unwrap_or(0);
        let cos = self.cos.narrow(0, cache_size, seq_len)?;
        let sin = self.sin.narrow(0, cache_size, seq_len)?;

        let ys = self.attention.forward(&xq, &xk, &xv, &cos, &sin, cache)?;
        self.out_proj.forward(&ys)
    }
}

/// `StandardBlock`: operator(norm(x)) + x, then GLU(norm(h)) + h.
pub struct StandardBlock {
    operator: Mha,
    feed_forward: Glu,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl StandardBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operator: Mha,
        ff_dim: Option<usize>,
        use_swiglu: bool,
        multiple_of: usize,
        ffn_dim_multiplier: f64,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = operator.dim();
        let feed_forward = Glu::new(dim, ff_dim, use_swiglu, multiple_of, ffn_dim_multiplier, vb.pp("feed_forward"))?;
        let operator_norm = RmsNorm::new(dim, norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = RmsNorm::new(dim, norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self { operator, feed_forward, operator_norm, ffn_norm })
    }

    pub fn forward(&self, x: &Tensor, cache: Option<&mut LayerKvCache>) -> Result<Tensor> {
        let h = (self.operator.forward(&self.operator_norm.forward(x)?, cache)? + x)?;
        let h_glu = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        h + h_glu
    }
}

/// `SharedEmbedding`: input embedding + pre-logits RMSNorm + output projection.
///
/// Python `tie_embedding` is a *train-time* parameter-sharing flag; the saved
/// checkpoint **always** ships a separate `to_logits.weight` (equal to
/// `embedding.weight` when tied, e.g. the depthformer with `depthformer.tie=True`;
/// distinct when untied, e.g. `audio_embedding` with `tie_audio_embeddings=False`).
/// So we load `to_logits.weight` from the checkpoint rather than assuming the tie
/// and reusing the embedding matrix — faithful for both cases.
pub struct SharedEmbedding {
    embedding: Embedding,
    embedding_norm: RmsNorm,
    to_logits: Linear,
}

impl SharedEmbedding {
    pub fn new(dim: usize, vocab_size: usize, norm_eps: f64, vb: VarBuilder) -> Result<Self> {
        let emb_w = vb.pp("embedding").get((vocab_size, dim), "weight")?;
        let embedding = Embedding::new(emb_w, dim);
        let embedding_norm = RmsNorm::new(dim, norm_eps, vb.pp("embedding_norm"))?;
        let to_logits_w = vb.pp("to_logits").get((vocab_size, dim), "weight")?;
        let to_logits = Linear::new(to_logits_w, None);
        Ok(Self { embedding, embedding_norm, to_logits })
    }

    pub fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embedding.forward(tokens)
    }

    pub fn get_logits(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.to_logits.forward(&self.embedding_norm.forward(embeddings)?)
    }
}

/// `RawLMBackbone`: a stack of `StandardBlock`s over continuous embeddings, with
/// an optional `SharedEmbedding` for token in / logits out.
pub struct RawLmBackbone {
    pub layers: Vec<StandardBlock>,
    pub embedding: Option<SharedEmbedding>,
    pub dim: usize,
}

impl RawLmBackbone {
    pub fn forward(&self, x: &Tensor, caches: Option<&mut [LayerKvCache]>) -> Result<Tensor> {
        let mut x = x.clone();
        match caches {
            Some(cs) => {
                for (layer, c) in self.layers.iter().zip(cs.iter_mut()) {
                    x = layer.forward(&x, Some(c))?;
                }
            }
            None => {
                for layer in self.layers.iter() {
                    x = layer.forward(&x, None)?;
                }
            }
        }
        Ok(x)
    }
}

/// `SequenceModel` (Python `class SequenceModel(nn.Module, ABC)`) — the
/// sequence-model contract: `[N,T,dim] → [N,T',dim_out]`, with `forward` /
/// `forward_cached`.
///
/// PORT: Python's `forward_cached(x, cache) -> (out, cache)` returns a fresh
/// cache; Rust mutates the cache in place via `Option<&mut [LayerKvCache]>`, so
/// `forward(x, Some(cache))` *is* `forward_cached` — the two abstract methods
/// collapse to one signature covering both the cached and uncached paths.
pub trait SequenceModel {
    fn dim(&self) -> usize;
    fn dim_out(&self) -> usize;
    fn forward(&self, x: &Tensor, cache: Option<&mut [LayerKvCache]>) -> Result<Tensor>;
}

impl SequenceModel for RawLmBackbone {
    fn dim(&self) -> usize {
        self.dim
    }
    fn dim_out(&self) -> usize {
        self.dim // the raw backbone returns hidden states of `dim`
    }
    fn forward(&self, x: &Tensor, cache: Option<&mut [LayerKvCache]>) -> Result<Tensor> {
        // Delegate to the inherent method (inherent resolution wins → no recursion).
        RawLmBackbone::forward(self, x, cache)
    }
}
