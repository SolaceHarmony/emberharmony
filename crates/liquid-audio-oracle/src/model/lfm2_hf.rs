//! HF `Lfm2Model` — the main LFM2 backbone (hybrid short-conv + GQA attention).
//!
//! Adapted from candle-transformers `models/lfm2.rs` (a faithful port of HF
//! transformers' `modeling_lfm2.py`) onto plain `candle_nn` (candle 0.9.x has
//! `mimi`/`quantized_lfm2` but not full-precision `lfm2`). Differences from the
//! candle reference, needed by liquid_audio:
//! - returns the **all-position** `last_hidden_state` (post embedding-norm), with
//!   no `lm_head` — both consumers (`LFM2AudioModel` and the detokenizer) do their
//!   own projection.
//! - `forward_embeds` accepts `inputs_embeds` and an optional **custom additive
//!   attention mask** (the detokenizer's sliding window); the main path passes
//!   `None` and gets a causal mask.
//! - tracing spans are dropped, and so are the reference's **custom CUDA kernels**.
//!   HF selects `flash_attention_2`/`sdpa` for attention (lfm2_audio.py L162) and
//!   binds the short-conv (`conv_L_cache`) to the `causal_conv1d` kernel when it is
//!   importable. Here attention is the eager matmul+softmax math (the kernel-free
//!   `sdpa`/no-flash path, *not* flash-attn's reordered online-softmax) and the
//!   short-conv is a plain candle `Conv1d` (prefill) / gather-mul-sum (single step).
//!   No custom kernels — which is precisely what lets this backbone run byte-exact
//!   on `Device::Cpu` (LFM2's "no GPU needed" design point, which the CUDA-gated
//!   reference stack cannot deliver as shipped). Verified: backbone parity 6.558e-6.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{
    embedding, linear_no_bias, Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder,
};

// The exact two `crate::utils::*` helpers candle-transformers' `models/lfm2.rs` imports,
// vendored onto the 0.9.2 pin (see `candle_ext`). Using the reference helpers — not a
// hand-rolled `causal_mask`/`repeat_kv` — keeps this a faithful port.
use crate::candle_ext::transformers_utils::{build_causal_mask, repeat_kv};

// The differentiable RMSNorm (basic ops), NOT candle_nn::RmsNorm — whose `forward`
// calls the fused `ops::rms_norm` (`apply_op2_no_bwd`) on contiguous inputs and so
// SEVERS autograd. The backbone is trained, so every norm must keep the gradient.
// Same `x * rsqrt(mean(x^2)+eps) * weight` forward.
use crate::model::transformer::RmsNorm;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "d_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default = "d_theta")]
    pub rope_theta: f32,
    #[serde(default = "d_maxpos")]
    pub max_position_embeddings: usize,
    #[serde(default = "d_lcache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    #[serde(default = "d_ffn_mult")]
    pub block_ffn_dim_multiplier: f32,
    #[serde(default = "d_mult_of")]
    pub block_multiple_of: usize,
    /// `block_ff_dim` — the pre-SwiGLU FFN size (0 ⇒ fall back to `4*hidden`,
    /// matching Python's `ff_dim is None` branch).
    #[serde(default)]
    pub block_ff_dim: usize,
    /// `block_auto_adjust_ff_dim` — apply the SwiGLU `2/3` reduction + multiple-of
    /// rounding (LFM2 default). When false, `block_ff_dim` is used as-is.
    #[serde(default = "d_true")]
    pub block_auto_adjust_ff_dim: bool,
}

fn d_kv_heads() -> usize {
    8
}
fn d_eps() -> f64 {
    1e-5
}
fn d_theta() -> f32 {
    1_000_000.0
}
fn d_maxpos() -> usize {
    128_000
}
fn d_lcache() -> usize {
    3
}
fn d_ffn_mult() -> f32 {
    1.0
}
fn d_mult_of() -> usize {
    256
}
fn d_true() -> bool {
    true
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// LFM2 SwiGLU FFN size, faithful to the Python Lfm2 MLP / GLU:
    /// start from `block_ff_dim` (or `4*hidden` if unset); with
    /// `block_auto_adjust_ff_dim`, reduce by `2/3`, scale by the ffn multiplier,
    /// then round **up** to `block_multiple_of`. The previous `hidden*4` form only
    /// coincided when `block_ff_dim == 6*hidden` (the main backbone); it underrounds
    /// the audio detokenizer (`block_ff_dim=3328` ⇒ 2304, not 2048).
    pub fn intermediate_size(&self) -> usize {
        let ff = if self.block_ff_dim > 0 {
            self.block_ff_dim
        } else {
            4 * self.hidden_size
        };
        if self.block_auto_adjust_ff_dim {
            let reduced = (2 * ff) / 3; // int(2*ff/3)
            let scaled = (self.block_ffn_dim_multiplier * reduced as f32) as usize;
            scaled.div_ceil(self.block_multiple_of) * self.block_multiple_of
        } else {
            ff
        }
    }
}

/// KV + conv-state cache plus the rotary tables.
///
/// Mirrors candle-transformers' `lfm2.rs` `Cache`, including the `masks` memo: each
/// causal mask is built once per `(seq_len, kv_len)` shape and reused across every
/// attention layer and decode step, instead of being reconstructed on every call.
/// Initial per-layer KV-cache capacity along the sequence axis. candle's [`KvCache`]
/// preallocates this many slots and grows by the same amount when exceeded (a `Tensor::cat`,
/// amortized negligible), so a turn shorter than this never reallocates. Kept modest so a
/// single short turn doesn't preallocate gigabytes; long contexts grow into it.
const KV_CACHE_INITIAL_CAP: usize = 512;

pub struct Cache {
    pub use_kv_cache: bool,
    /// Per-layer preallocated KV cache (candle's `KvCache`): `append` slice-sets the new K/V
    /// into a fixed buffer (O(new)) instead of `Tensor::cat`-ing the whole cache every step
    /// (O(seqlen) realloc+copy). Same accumulated values, far less memory traffic; also the
    /// unit that inter-turn persistence will carry across turns. Seq axis is dim 2 of
    /// `(b, n_kv, seq, head_dim)`.
    kvs: Vec<KvCache>,
    conv_states: Vec<Option<Tensor>>,
    // Memoized boolean causal masks, keyed by (seq_len, kv_len) — see `mask`.
    masks: HashMap<(usize, usize), Tensor>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        cfg: &Lfm2Config,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let idx_theta = t.matmul(&inv_freq)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            use_kv_cache,
            kvs: (0..cfg.num_hidden_layers)
                .map(|_| KvCache::new(2, KV_CACHE_INITIAL_CAP))
                .collect(),
            conv_states: vec![None; cfg.num_hidden_layers],
            masks: HashMap::new(),
            cos,
            sin,
            device: device.clone(),
        })
    }

    /// Memoized boolean causal mask `(seq_len, kv_len)`, faithful to `lfm2.rs`'s
    /// `Cache::mask`: build once per shape via the vendored [`build_causal_mask`], reuse
    /// for every attention layer and every decode step. The mask only depends on the
    /// `(seq_len, index_pos)` geometry, so caching by `(seq_len, kv_len)` is exact.
    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear(&mut self) {
        self.kvs.iter_mut().for_each(|c| c.reset());
        self.conv_states.iter_mut().for_each(|v| *v = None);
        // `masks` are shape-keyed and turn-independent, so they survive `clear` (a fresh
        // turn reuses the same geometry) — matching the reference, which never drops them.
    }
}

/// `masked_fill` (candle-transformers `lfm2.rs`): keep `on_false` where `mask` is zero,
/// substitute the scalar `on_true` where `mask` is nonzero — the boolean-mask way to set
/// future positions to `-inf` before the softmax (equivalent to an additive `-inf` mask).
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(mask.shape())?;
    mask.where_cond(&on_true, on_false)
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Lfm2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size();
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("w1"))?,
            up_proj: linear_no_bias(h, i, vb.pp("w3"))?,
            down_proj: linear_no_bias(i, h, vb.pp("w2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Lfm2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        Ok(Self {
            q_proj: linear_no_bias(h, nh * hd, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(h, nkv * hd, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(h, nkv * hd, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(nh * hd, h, vb.pp("out_proj"))?,
            q_norm: RmsNorm::new(hd, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: RmsNorm::new(hd, cfg.norm_eps, vb.pp("k_layernorm"))?,
            n_head: nh,
            n_kv: nkv,
            head_dim: hd,
        })
    }

    fn rope(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        // `rope_slow` (differentiable), NOT `rope` (fused `apply_op3_no_bwd`): the
        // backbone is trained, and the fused op severs the gradient to q/k. Same NeoX
        // (half-split) rotation, same forward values.
        candle_nn::rotary_emb::rope_slow(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        add_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq_len, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq_len, self.n_kv, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = self.rope(&q, index_pos, cache)?;
        let k = self.rope(&k, index_pos, cache)?;

        let (k, v) = if cache.use_kv_cache {
            // Preallocated candle `KvCache`: `append` slice-sets the new K/V into the fixed
            // buffer (O(seq_len), not an O(kv_len) `cat` + `clone` every step) and returns the
            // full accumulated (k, v) — identical values to the old `Tensor::cat` form. The
            // src must be contiguous for `slice_set`; the returned views feed `repeat_kv` /
            // the f32 upcast, which produce contiguous outputs for the score matmul. With a
            // fresh cache per generation this also subsumes the old `index_pos > 0` guard
            // (an empty cache appends as the prefill, a populated one appends as a decode step).
            cache.kvs[block_idx].append(&k.contiguous()?, &v.contiguous()?)?
        } else {
            (k, v)
        };

        let k = repeat_kv(k, self.n_head / self.n_kv)?;
        let v = repeat_kv(v, self.n_head / self.n_kv)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match add_mask {
            // Detokenizer's sliding window: an *additive* f32 mask supplied by the caller
            // (our documented deviation — the reference has no custom-mask path).
            Some(m) => att.broadcast_add(&m.to_dtype(DType::F32)?)?,
            None if seq_len == 1 => att,
            // Causal: the reference's memoized boolean mask + masked_fill(-inf).
            None => {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            }
        };
        // `softmax` (differentiable), NOT `softmax_last_dim` (fused no_bwd): backbone
        // attention is trained. Same forward values.
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(x.dtype())?;
        let y = y
            .transpose(1, 2)?
            .reshape((b, seq_len, self.n_head * self.head_dim))?;
        self.o_proj.forward(&y)
    }
}

struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor, // (hidden, 1, l_cache)
    l_cache: usize,
    hidden_size: usize,
}

impl ShortConv {
    fn new(cfg: &Lfm2Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            in_proj: linear_no_bias(h, 3 * h, vb.pp("in_proj"))?,
            out_proj: linear_no_bias(h, h, vb.pp("out_proj"))?,
            conv_weight: vb.get((h, 1, cfg.conv_l_cache), "conv.weight")?,
            l_cache: cfg.conv_l_cache,
            hidden_size: h,
        })
    }

    fn forward(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let bgate = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;
        let bx = (bgate * &x_proj)?.contiguous()?;

        let conv_out = if seq_len == 1 {
            let conv_weight = self.conv_weight.squeeze(1)?;
            let mut state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => {
                    Tensor::zeros((b, self.hidden_size, self.l_cache), bx.dtype(), bx.device())?
                }
            };
            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }
            if cache.use_kv_cache {
                cache.conv_states[block_idx] = Some(state.clone());
            }
            (state * conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                self.conv_weight.clone(),
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: self.hidden_size,
                    ..Default::default()
                },
            );
            let out = conv.forward(&bx)?.narrow(2, 0, seq_len)?;
            if cache.use_kv_cache && self.l_cache > 0 {
                let start = seq_len.saturating_sub(self.l_cache);
                let clen = seq_len - start;
                let mut src = bx.narrow(2, start, clen)?;
                if clen < self.l_cache {
                    let zeros = Tensor::zeros(
                        (b, self.hidden_size, self.l_cache - clen),
                        src.dtype(),
                        src.device(),
                    )?;
                    src = Tensor::cat(&[zeros, src], 2)?;
                }
                cache.conv_states[block_idx] = Some(src);
            }
            out
        };

        let conv_out = (c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

enum LayerKind {
    Attention(Box<Attention>),
    ShortConv(ShortConv),
}

struct DecoderLayer {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
}

impl DecoderLayer {
    fn new(cfg: &Lfm2Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let operator_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let mlp = Mlp::new(cfg, vb.pp("feed_forward"))?;
        let kind = match cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention)
        {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"))?),
        };
        Ok(Self {
            operator_norm,
            ffn_norm,
            mlp,
            kind,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        add_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.operator_norm.forward(x)?;
        let h = match &self.kind {
            LayerKind::Attention(a) => a.forward(&h, index_pos, block_idx, cache, add_mask)?,
            LayerKind::ShortConv(c) => c.forward(&h, block_idx, cache)?,
        };
        let x = (h + residual)?;
        let residual = &x;
        let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        h + residual
    }
}

/// HF `Lfm2Model` — returns the all-position last hidden state (post embedding-norm).
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
}

impl Model {
    pub fn new(cfg: &Lfm2Config, vb: VarBuilder) -> Result<Self> {
        // `lfm` is a bare HF `Lfm2Model` (not `Lfm2ForCausalLM`), so weights sit
        // directly under the given prefix — no `.model.` wrapper. Final norm is
        // `embedding_norm` (verified against LFM2-Audio-1.5B's safetensors keys:
        // lfm.embed_tokens / lfm.layers.N / lfm.embedding_norm).
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, vb_l.pp(i.to_string()))?);
        }
        let embedding_norm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("embedding_norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
        })
    }

    /// Embedding lookup (the model's input token embedding).
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// The tied embedding weight (for an external lm_head).
    pub fn embed_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    /// Run the backbone over `inputs_embeds`, returning the all-position hidden
    /// state. `add_mask` overrides the default causal mask (e.g. sliding window).
    pub fn forward_embeds(
        &self,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        add_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = embeds.clone();
        for (block_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, index_pos, block_idx, cache, add_mask)?;
        }
        self.embedding_norm.forward(&hidden)
    }

    /// Convenience: token ids → all-position hidden state (causal).
    pub fn forward_ids(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_embeds(&embeds, index_pos, cache, None)
    }

    /// Last-position hidden state convenience (e.g. for a separate lm_head).
    pub fn last_hidden(&self, hidden: &Tensor) -> Result<Tensor> {
        let seq_len = hidden.dim(1)?;
        hidden.i((.., seq_len - 1, ..))?.contiguous()
    }
}
