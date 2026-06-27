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
// `rope_i_slow` (the basic-op rotary), NOT `rope_i` (the fused `apply_op3_no_bwd`
// kernel): the depthformer runs inside the trainable `logits`/`forward` graph, and
// the fused op SEVERS autograd (verified — no gradient reaches q/k). The slow path is
// the SAME interleaved rotation, just differentiable.
// `ops::softmax` (basic ops, differentiable), NOT `softmax_last_dim` (the fused
// `apply_op1_no_bwd` kernel that severs autograd) — this attention runs in the
// trainable `logits`/`forward` graph. Same forward values.
use candle_nn::{linear_no_bias, ops::softmax, rotary_emb::rope_i_slow, Embedding, Linear, Module, VarBuilder};

use crate::candle_ext::kv_cache::ConcatKvCache;

/// `head_style` for attention. Mirrors the `Literal["mha","gqa","mqa"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadStyle {
    Mha,
    Gqa,
    Mqa,
}

/// `CacheType` (py 13: `type CacheType = torch.Tensor | None | Sequence["CacheType"]`).
///
/// PORT: the Python alias is a recursive union — at the leaf a `forward_cached`
/// layer cache is a `(key, value)` tensor tuple or `None`; at the backbone level
/// it is a `Sequence` of those. We model the *leaf* (per-layer) cache as
/// [`LayerCache`] (an `Option<(Tensor, Tensor)>`) and the backbone-level sequence
/// as `Vec<LayerCache>` (see [`RawLmBackbone::forward_cached`]). This captures the
/// two concrete shapes the union takes in this module without an open-ended enum.
pub type LayerCache = Option<(Tensor, Tensor)>;

/// Per-layer KV cache. Port of `LayerKVCache`: stores key/value pre-transpose
/// (shape `[b, t, heads, head_dim]`) and concatenates new steps along dim 1.
///
/// This is a thin **adapter** over candle's cat-based [`ConcatKvCache`]
/// (vendored from candle-nn 0.10.2), which is itself a structural 1:1 of the
/// Python class (`torch.cat([key_cache, k], dim=1)` ⇒ `append` on `dim=1`). The
/// adapter only re-exposes the Python method names (`update`, `get_cache_size`)
/// and the `(key, value)`-tuple constructor the depthformer threads through
/// `forward_cached`; the cat itself is candle's, not re-implemented here.
pub struct LayerKvCache {
    inner: ConcatKvCache,
}

impl Default for LayerKvCache {
    fn default() -> Self {
        // dim=1: the Python `update` concatenates along the time axis of
        // `[b, t, heads, head_dim]` (`torch.cat(..., dim=1)`).
        Self { inner: ConcatKvCache::new(1) }
    }
}

impl LayerKvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// `LayerKVCache.__init__(cache)` (py 43-46): build from an optional
    /// `(key, value)` tensor tuple. Faithful to the Python `assert cache is None
    /// or len(cache) == 2` — a `Some` carries exactly two tensors by construction.
    pub fn from_cache(cache: LayerCache) -> Self {
        let mut inner = ConcatKvCache::new(1);
        if let Some((k, v)) = cache {
            // Seed the cache with the supplied (key, value) pair; `append` from
            // empty is exactly "set" (no prior to concatenate with).
            let _ = inner.append(&k, &v);
        }
        Self { inner }
    }

    /// `LayerKVCache.update(k, v)` (py 48-56): concatenate the new step and return
    /// the full `(key, value)`. Delegates to `ConcatKvCache::append` (cat on dim 1).
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.inner.append(k, v)
    }

    /// `LayerKVCache.get_cache_size()` (py 58-62): the cached time length, or 0.
    pub fn get_cache_size(&self) -> usize {
        self.inner.current_seq_len()
    }

    /// Extract the current `(key, value)` pair as a [`LayerCache`]. Mirrors the
    /// Python `forward_cached` returning `new_cache` (the `(k, v)` tuple produced
    /// by `LayerKVCache.update`). Returns `None` when the cache is still empty.
    pub fn to_cache(&self) -> LayerCache {
        match (self.inner.k(), self.inner.v()) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
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

    /// `RMSNorm._norm` (py 71-72): `x * rsqrt(mean(x^2, -1, keepdim) + eps)`.
    ///
    /// UN-FOLDED from `forward` for 1:1 parity with the Python helper. Operates on
    /// whatever dtype it is handed (the Python `_norm` is dtype-agnostic; `forward`
    /// upcasts to f32 first). Matches torch's `x * rsqrt(z)` *structure* —
    /// `recip(sqrt(z))` then multiply, rather than divide-by-sqrt. candle has no
    /// *fused* rsqrt, so the reciprocal-sqrt is two ops and still differs from torch's
    /// fused `rsqrt` by ~1 ULP: that is the cross-library floor (see PYTHON_VS_RUST.md
    /// §1.4), not a faithfulness defect.
    pub fn norm(&self, x: &Tensor) -> Result<Tensor> {
        let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (mean_sq + self.eps)?.sqrt()?.recip()?; // 1/sqrt(z) ≈ rsqrt(z)
        x.broadcast_mul(&rsqrt)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let normed = self.norm(&x)?;
        let w = self.weight.to_dtype(DType::F32)?;
        normed.broadcast_mul(&w)?.to_dtype(in_dtype)
    }

    /// `RMSNorm.forward_cached` (py 80-81): `return self(x, cache), None`. RMSNorm
    /// holds no KV state, so it threads through a `None` cache unchanged.
    pub fn forward_cached(&self, x: &Tensor) -> Result<(Tensor, LayerCache)> {
        Ok((self.forward(x)?, None))
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

    /// `GLU.forward_cached` (py 136-137): `return self(x, cache), None`. The GLU
    /// feed-forward is stateless, so it returns a `None` cache.
    pub fn forward_cached(&self, x: &Tensor) -> Result<(Tensor, LayerCache)> {
        Ok((self.forward(x)?, None))
    }
}

/// Precompute rotary `(cos, sin)` of shape `[end, dim/2]`. Faithful to
/// `precompute_freqs_cis` (`polar(1, outer(t, inv_freq))`), but returned as the
/// real `cos`/`sin` candle's `rope_i` consumes.
///
/// Reuse checked: candle-nn's `rotary_emb` exposes the rope *application*
/// (`rope`/`rope_slow`/`rope_i`/`rope_i_slow` — all reused) but no cos/sin *table*
/// builder; the table is model-specific (theta, interleaving) and moshi's
/// `RotaryEmbedding` bakes its own convention, so this small builder stays local
/// (backbone parity 6.3e-6 confirms the table matches HF Lfm2).
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
    Ok((rope_i_slow(xq, cos, sin)?, rope_i_slow(xk, cos, sin)?))
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
        let attn = softmax(&attn, D::Minus1)?;
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

    /// `MHA._validate_cache` (py 295-301): TypeGuard that the cache is a 2-tuple of
    /// tensors.
    ///
    /// PORT: in the Rust type system a [`LayerCache`] of the `Some` variant is, by
    /// construction, exactly a `(Tensor, Tensor)` pair — the Python runtime checks
    /// (`isinstance tuple`, `len == 2`, both entries `torch.Tensor`) are enforced
    /// statically. This returns the boolean the Python `TypeGuard` returns: `true`
    /// for `Some(_)`, `false` for `None` (so `assert self._validate_cache(cache)`
    /// inside the `cache is not None` branch maps to `Some` ⇒ `true`).
    pub fn validate_cache(&self, cache: &LayerCache) -> bool {
        cache.is_some()
    }

    /// `MHA.forward_cached` (py 306-341): build a [`LayerKvCache`] from the incoming
    /// cache (validating it when present), run the qkv projection / head split /
    /// cache-aware rotary slice / `BoundedAttention` / output projection, and return
    /// `(ys, new_cache)` where `new_cache` is the updated `(k, v)` tuple.
    pub fn forward_cached(&self, x: &Tensor, cache: LayerCache) -> Result<(Tensor, LayerCache)> {
        // py 307-311: `if cache is not None: assert self._validate_cache(cache);
        //             kv_cache = LayerKVCache(cache) else: kv_cache = None`.
        //
        // PORT: in the Python, `kv_cache=None` still flows `(k, v)` *back out* as
        // `new_cache` — `BoundedAttention` returns the freshly-projected `(k, v)`
        // even when it skips `update`. To surface that same tuple (and to keep the
        // streaming chain alive, since `RawLMBackbone.forward_cached` seeds the
        // first step with `[None] * n_layers`), we always build a `LayerKvCache`.
        // For an empty (`None`-seeded) cache, `update` *initializes* to `(k, v)`,
        // which is byte-identical to the no-`update` return — so this is faithful to
        // both Python branches, differing only in that we never lose the new tuple.
        if cache.is_some() {
            debug_assert!(self.validate_cache(&cache));
        }
        let mut kv_cache = LayerKvCache::from_cache(cache);

        // `Mha::forward` carries the whole cache-aware path (qkv split, freqs slice
        // from `get_cache_size`, `BoundedAttention` with in-place `update`). Run it
        // with the constructed cache, then surface the updated `(k, v)` tuple.
        let ys = self.forward(x, Some(&mut kv_cache))?;
        Ok((ys, kv_cache.to_cache()))
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

    /// `StandardBlock.forward_cached` (py 385-390): thread the cache through
    /// `operator.forward_cached(operator_norm(x))` (which returns the updated cache),
    /// add the residual, then the stateless `feed_forward(ffn_norm(h))` + residual.
    /// Returns `(out, new_cache)`.
    pub fn forward_cached(&self, x: &Tensor, cache: LayerCache) -> Result<(Tensor, LayerCache)> {
        // py 386: `h, new_cache = self.operator.forward_cached(self.operator_norm(x), cache)`
        let (h, new_cache) = self.operator.forward_cached(&self.operator_norm.forward(x)?, cache)?;
        // py 387: `h += x`
        let h = (h + x)?;
        // py 388: `h_glu = self.feed_forward.forward(self.ffn_norm(h))`
        let h_glu = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        // py 389-390: `out = h + h_glu; return out, new_cache`
        Ok(((h + h_glu)?, new_cache))
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

    /// `SharedEmbedding.forward` (py 500-501): `return self.embed(tokens)` — the
    /// plain embedding lookup. UN-FOLDED to delegate to [`SharedEmbedding::embed`]
    /// exactly as the Python `forward` delegates to `embed`.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed(tokens)
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
    /// `RawLMBackbone.__init__(layers, vocab_size=65536, norm_eps=1e-5,
    /// embed_init_scale=1.0, *, has_embedding=True, tie_embedding=True)` (py 517).
    /// Python derives `self.dim = layers[0].dim` (asserted equal to the last
    /// layer's `dim_out`); the shared embedding is built by the caller via
    /// `VarBuilder` (weights are loaded, not initialized) and passed in — `None`
    /// is `has_embedding=False`. `dim` is the backbone width.
    pub fn new(layers: Vec<StandardBlock>, embedding: Option<SharedEmbedding>, dim: usize) -> Self {
        Self { layers, embedding, dim }
    }

    pub fn forward(&self, x: &Tensor, caches: Option<&mut [LayerKvCache]>) -> Result<Tensor> {
        let mut x = x.clone();
        match caches {
            Some(cs) => {
                // Python iterates `zip(self.layers, cache, strict=True)` (py 549/562):
                // a length mismatch raises. `zip` here silently stops at the shorter
                // side, so a short cache slice would run only a PREFIX of the backbone.
                // Require exactly one cache entry per layer instead.
                if cs.len() != self.layers.len() {
                    return Err(candle_core::Error::Msg(format!(
                        "RawLmBackbone::forward: cache has {} entries, expected one per layer ({})",
                        cs.len(),
                        self.layers.len()
                    )));
                }
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

    /// `RawLMBackbone.forward_cached` (py 554-566): run each layer's
    /// `forward_cached`, threading a per-layer cache and collecting the updated
    /// caches into a `Vec`. Returns `(hidden, Vec<LayerCache>)`.
    ///
    /// PORT: the Python `CacheType` here is the `Sequence[CacheType]` arm — a list
    /// with one `(k, v)` (or `None`) entry per layer. When `cache is None` it seeds
    /// `[None] * len(self.layers)` (py 559); we mirror that by consuming an
    /// `Option<Vec<LayerCache>>` and defaulting to a `None`-filled vector. The
    /// `assert len(cache) == len(self.layers)` (py 557) maps to the length check.
    pub fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<Vec<LayerCache>>,
    ) -> Result<(Tensor, Vec<LayerCache>)> {
        // py 555-559: validate length when present, else seed `[None] * n_layers`.
        let cache = match cache {
            Some(c) => {
                assert!(c.len() == self.layers.len(), "expected one cache entry per layer");
                c
            }
            None => (0..self.layers.len()).map(|_| None).collect(),
        };

        // py 561-564: `for layer, layer_cache in zip(...): x, new = layer.forward_cached(x, layer_cache); cache_out.append(new)`
        let mut x = x.clone();
        let mut cache_out: Vec<LayerCache> = Vec::with_capacity(self.layers.len());
        for (layer, layer_cache) in self.layers.iter().zip(cache.into_iter()) {
            let (next, new_cache) = layer.forward_cached(&x, layer_cache)?;
            x = next;
            cache_out.push(new_cache);
        }

        // py 566: `return x, cache_out`
        Ok((x, cache_out))
    }
}

/// `SequenceModel` (Python `class SequenceModel(nn.Module, ABC)`) — the
/// sequence-model contract: `[N,T,dim] → [N,T',dim_out]`, with `forward` /
/// `forward_cached`.
///
/// PORT: Python's offline `forward(x, cache) -> out` mutates an in-place
/// `LayerKVCache`; Rust models that with `Option<&mut [LayerKvCache]>`, so the
/// offline `forward(x, Some(cache))` is the in-place cached path. The streaming
/// `forward_cached(x, cache) -> (out, cache)` (py 34-35) instead *returns* a fresh
/// per-layer cache vector — a faithful port of the functional cache contract that
/// `RawLMBackbone.forward_cached` (py 554) builds on.
pub trait SequenceModel {
    /// `SequenceModel.__init__(*args, **kwargs)` (py 28-29) — the abstract base's
    /// constructor is just `super().__init__()` (nn.Module bookkeeping), which has
    /// no candle referent in an inference port. Faithfully a no-op; concrete models
    /// construct via their own `new`. (`where Self: Sized` keeps the trait
    /// object-safe.)
    fn new()
    where
        Self: Sized,
    {
    }

    fn dim(&self) -> usize;
    fn dim_out(&self) -> usize;
    fn forward(&self, x: &Tensor, cache: Option<&mut [LayerKvCache]>) -> Result<Tensor>;

    /// `SequenceModel.forward_cached` (py 34-35): abstract streaming step returning
    /// `(out, new_cache)`. The cache is the `Sequence[CacheType]` arm of the union —
    /// one `(k, v)` (or `None`) per layer — modeled as `Vec<LayerCache>`.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<Vec<LayerCache>>,
    ) -> Result<(Tensor, Vec<LayerCache>)>;
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
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<Vec<LayerCache>>,
    ) -> Result<(Tensor, Vec<LayerCache>)> {
        // Delegate to the inherent method (inherent resolution wins → no recursion).
        RawLmBackbone::forward_cached(self, x, cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_is_differentiable() {
        // apply_rotary_emb runs inside the trainable depthformer graph, so it must NOT
        // sever autograd. The fused `rope_i` does (apply_op3_no_bwd); `rope_i_slow`
        // does not. This fails if anyone swaps it back to the fused op.
        use candle_core::Var;
        let dev = Device::Cpu;
        let (b, h, t, d) = (1usize, 1, 4, 8);
        let cos = Tensor::ones((t, d / 2), DType::F32, &dev).unwrap();
        let sin = Tensor::zeros((t, d / 2), DType::F32, &dev).unwrap();
        let q = Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, h, t, d), &dev).unwrap()).unwrap();
        let k = Var::from_tensor(&Tensor::randn(0f32, 1f32, (b, h, t, d), &dev).unwrap()).unwrap();
        let (qr, kr) = apply_rotary_emb(q.as_tensor(), k.as_tensor(), &cos, &sin).unwrap();
        let loss = (qr.sqr().unwrap().sum_all().unwrap() + kr.sqr().unwrap().sum_all().unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        assert!(grads.get(&q).is_some(), "rotary severed the gradient to q (fused rope_i?)");
        assert!(grads.get(&k).is_some(), "rotary severed the gradient to k");
    }

    #[test]
    fn forward_rejects_cache_length_mismatch() {
        // A RawLmBackbone with 0 layers exercises the cache-length guard without any
        // weights: a cache slice whose len != layer count must error (Python's
        // zip(strict=True)), not silently run a prefix of the backbone.
        let bb = RawLmBackbone::new(vec![], None, 4);
        let x = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();

        // Matching lengths (0 caches, 0 layers) → Ok (no layers run).
        let mut empty: Vec<LayerKvCache> = vec![];
        assert!(bb.forward(&x, Some(empty.as_mut_slice())).is_ok());

        // Mismatch (1 cache, 0 layers) → Err, not a silent prefix run.
        let mut one = vec![LayerKvCache::new()];
        assert!(
            bb.forward(&x, Some(one.as_mut_slice())).is_err(),
            "cache length mismatch must error, not skip layers"
        );
    }
}
