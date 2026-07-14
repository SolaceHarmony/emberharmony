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
//!   short-conv routes through the FlashFFTConv `depthwise_conv1d_stream` CustomOp (CPU
//!   reference + one Metal kernel) for BOTH prefill and single-step decode — one causal
//!   depthwise path, the prior K-1 inputs streamed via the conv cache. Identical math to
//!   candle's `Conv1d` (matched bit-exact — see `tests/short_conv_parity.rs`). The op
//!   carries a CPU reference, so the backbone still runs on
//!   `Device::Cpu` (LFM2's "no GPU needed" design point, which the CUDA-gated reference
//!   stack cannot deliver as shipped). Verified: backbone parity 6.558e-6.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

// The exact two `crate::utils::*` helpers candle-transformers' `models/lfm2.rs` imports,
// vendored onto the 0.9.2 pin (see `candle_ext`). Using the reference helpers — not a
// hand-rolled `causal_mask`/`repeat_kv` — keeps this a faithful port.
use crate::candle_ext::transformers_utils::{build_causal_mask, repeat_kv};
use crate::model::linear::linear_forward;

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
pub struct Cache {
    pub use_kv_cache: bool,
    /// Decode-step ShortConv path selector: `true` (the production default set by
    /// [`Cache::new`]) runs the fused causal-conv1d update kernel; `false` runs the
    /// composed candle ops — the reference semantics the fused kernel must match.
    /// A per-cache field so tests A/B the two paths on the same weights by
    /// constructing a cache per path; never an ambient/global toggle.
    pub fused_conv_decode: bool,
    /// Decode-step GQA path selector: `true` (production default) computes attention with
    /// q regrouped `[b, n_kv, group, hd]` against the SHARED kv heads — no `repeat_kv`
    /// materialization (two full-cache copies per step). Same dot products; the GEMM
    /// reduction order differs, so outputs sit at the f32-ulp floor rather than byte-parity
    /// with the expanded form (rel < 1e-5, pinned by `grouped_gqa_matches_expanded_at_f32_ulp`).
    /// Ulps CAN flip a near-tied greedy argmax and WILL diverge sampled streams — measured:
    /// a 96-token greedy+seeded run picks different (equally sensible) slogans. `false` pins
    /// the reference expanded form: byte-identical output (verified by wav hash).
    pub grouped_gqa_decode: bool,
    /// Per-layer RESIDENT KV storage (runtime-architecture audit, item 1): preallocated
    /// planes in the incoming projection dtype, written in place and read as zero-copy
    /// narrows. On the live CPU bf16 decode path these are bf16 planes. This deliberately
    /// replaces the reference `Tensor::cat` append — which recopied the whole accumulated
    /// cache per layer per token (plus a full-cache f32 re-upcast) and made decode degrade
    /// with context. History note: an earlier `candle_nn::KvCache` swap was reverted as a
    /// parity deviation; this one is held to a stricter standard than that attempt — the
    /// narrow views feed attention byte-identical rows: with `grouped_gqa_decode=false`
    /// a greedy+seeded generate produces a BIT-IDENTICAL wav before/after this change, so
    /// the storage swap itself is exact; only the storage shape deviates from the
    /// reference, by design and on the record.
    kvs: Vec<Option<KvSlot>>,
    conv_states: Vec<Option<Tensor>>,
    // Memoized boolean causal masks, keyed by (seq_len, kv_len) — see `mask`.
    masks: HashMap<(usize, usize), Tensor>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    /// Resident per-layer state table for the native token pass — capacity allocated
    /// once, entries REWRITTEN each token (fresh pointer captures are the correctness
    /// mechanism: rollback clones conv states, candle-path steps and KV growth move
    /// storages; capturing per token self-heals against all of them). What this kills
    /// is the per-token `Vec` allocation, not the captures.
    pub(crate) native_states: crate::flashkern::native_engine::StateTable,
}

/// A layer's resident KV storage: preallocated `[B, n_kv, cap, head_dim]` bf16 planes (the
/// checkpoint/cache dtype — half the bytes and read bandwidth of f32 for identical values,
/// since bf16→f32 widening is exact) with a
/// length cursor. Appends write in place (`slice_set`); reads are zero-copy narrows past the
/// cursor; rollback is a cursor move (rows beyond it are stale and never read). Capacity
/// doubles on demand — one narrow-copy, amortized O(1) per stream.
#[derive(Clone)]
struct KvSlot {
    k: Tensor,
    v: Tensor,
    len: usize,
}

/// Rollback point for a [`Cache`] — see [`Cache::snapshot`] / [`Cache::rollback`].
/// Used by the engine's speculative prefill: prefill the next utterance during
/// the VAD pause window, roll back if the user resumes speaking.
pub struct CacheSnapshot {
    kv_lens: Vec<Option<usize>>,
    conv_states: Vec<Option<Tensor>>,
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
            fused_conv_decode: true,
            grouped_gqa_decode: true,
            kvs: vec![None; cfg.num_hidden_layers],
            conv_states: vec![None; cfg.num_hidden_layers],
            masks: HashMap::new(),
            cos,
            sin,
            device: device.clone(),
            native_states: Default::default(),
        })
    }

    /// Capture a rollback point for speculative prefill: per-layer KV sequence
    /// lengths plus independent copies of the carried conv states (tiny
    /// `[B,D,K-1]` tensors).
    /// The KV tensors themselves are not copied — [`Self::rollback`] restores by
    /// narrowing back to the recorded lengths.
    pub fn snapshot(&self) -> Result<CacheSnapshot> {
        let kv_lens = self
            .kvs
            .iter()
            .map(|kv| kv.as_ref().map(|sl| sl.len))
            .collect();
        let conv_states = self
            .conv_states
            .iter()
            .map(|state| state.as_ref().map(Tensor::copy).transpose())
            .collect::<Result<Vec<_>>>()?;
        Ok(CacheSnapshot {
            kv_lens,
            conv_states,
        })
    }

    /// Undo everything appended since `snap` was taken: truncate each layer's KV
    /// back to the recorded length and restore the conv states. Only valid for a
    /// snapshot taken from THIS cache with no rollback-crossing mutations other
    /// than appends (the speculative-prefill contract).
    pub fn rollback(&mut self, snap: &CacheSnapshot) -> Result<()> {
        if snap.kv_lens.len() != self.kvs.len() || snap.conv_states.len() != self.conv_states.len()
        {
            candle_core::bail!(
                "cache rollback: snapshot layer count {}/{} does not match cache {}/{}",
                snap.kv_lens.len(),
                snap.conv_states.len(),
                self.kvs.len(),
                self.conv_states.len()
            );
        }
        for (kv, len) in self.kvs.iter_mut().zip(&snap.kv_lens) {
            match len {
                None => *kv = None,
                Some(n) => match kv.as_mut() {
                    Some(sl) => {
                        if sl.len < *n {
                            candle_core::bail!(
                                "cache rollback: KV shrank below snapshot ({} < {n})",
                                sl.len
                            );
                        }
                        // O(1): rows past the cursor are stale storage, never read.
                        sl.len = *n;
                    }
                    None => {
                        candle_core::bail!("cache rollback: KV lost since snapshot (had len {n})")
                    }
                },
            }
        }
        self.conv_states = snap
            .conv_states
            .iter()
            .map(|state| state.as_ref().map(Tensor::copy).transpose())
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    /// Append this step's K/V rows IN PLACE into the layer's resident slot and return
    /// zero-copy views of the live rows — no `Tensor::cat`, no dtype conversion, no
    /// clone-back. The planes hold bf16 (checkpoint/cache dtype); the rows land verbatim.
    /// `index_pos == 0` begins a new stream (cursor reset; the buffer is reused when the
    /// geometry matches). `kf`/`vf` are the step's `[B, n_kv, s, hd]` bf16 rows;
    /// `index_pos` must equal the slot cursor (the streaming contract).
    /// Pre-grow (or create) a layer's KV slot so rows `0..need` fit — the native
    /// attention path captures raw plane pointers and must never trigger growth while
    /// they are live. Same alloc/grow/copy logic as [`Self::append_kv`], no row writes.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_kv_capacity(
        &mut self,
        block_idx: usize,
        b: usize,
        n_kv: usize,
        hd: usize,
        need: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        let slot = &mut self.kvs[block_idx];
        match slot.as_mut() {
            None => {
                let cap = need.next_power_of_two().max(256);
                let k = Tensor::zeros((b, n_kv, cap, hd), dtype, device)?;
                let v = Tensor::zeros((b, n_kv, cap, hd), dtype, device)?;
                *slot = Some(KvSlot { k, v, len: 0 });
            }
            Some(sl) => {
                let cap = sl.k.dim(2)?;
                if need > cap {
                    let ncap = need.next_power_of_two().max(cap * 2);
                    let k = Tensor::zeros((b, n_kv, ncap, hd), sl.k.dtype(), sl.k.device())?;
                    let v = Tensor::zeros((b, n_kv, ncap, hd), sl.v.dtype(), sl.v.device())?;
                    if sl.len > 0 {
                        k.slice_set(&sl.k.narrow(2, 0, sl.len)?.contiguous()?, 2, 0)?;
                        v.slice_set(&sl.v.narrow(2, 0, sl.len)?.contiguous()?, 2, 0)?;
                    }
                    sl.k = k;
                    sl.v = v;
                }
            }
        }
        Ok(())
    }

    /// Advance a layer's KV cursor by one row — the native attention path wrote the
    /// step's rows directly into the planes; the ledger stays here.
    pub fn advance_kv_cursor(&mut self, block_idx: usize) {
        if let Some(sl) = self.kvs[block_idx].as_mut() {
            sl.len += 1;
        }
    }

    fn append_kv(
        &mut self,
        block_idx: usize,
        kf: &Tensor,
        vf: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b, n_kv, s_new, hd) = kf.dims4()?;
        let slot = &mut self.kvs[block_idx];
        if index_pos == 0 {
            if let Some(sl) = slot.as_mut() {
                let (sb, sn, _, sh) = sl.k.dims4()?;
                if sb == b && sn == n_kv && sh == hd {
                    sl.len = 0; // new stream, same geometry: reuse the planes
                } else {
                    *slot = None;
                }
            }
        }
        match (slot.as_ref(), index_pos) {
            (Some(sl), p) if sl.len != p => candle_core::bail!(
                "kv append: cursor {} != index_pos {p} (layer {block_idx})",
                sl.len
            ),
            (None, p) if p != 0 => {
                candle_core::bail!("kv append: empty slot at index_pos {p} (layer {block_idx})")
            }
            _ => {}
        }
        let need = index_pos + s_new;
        match slot.as_mut() {
            None => {
                let cap = need.next_power_of_two().max(256);
                let k = Tensor::zeros((b, n_kv, cap, hd), kf.dtype(), kf.device())?;
                let v = Tensor::zeros((b, n_kv, cap, hd), vf.dtype(), vf.device())?;
                *slot = Some(KvSlot { k, v, len: 0 });
            }
            Some(sl) => {
                let cap = sl.k.dim(2)?;
                if need > cap {
                    let ncap = need.next_power_of_two().max(cap * 2);
                    let k = Tensor::zeros((b, n_kv, ncap, hd), kf.dtype(), kf.device())?;
                    let v = Tensor::zeros((b, n_kv, ncap, hd), vf.dtype(), vf.device())?;
                    if sl.len > 0 {
                        // The narrow is strided across kv heads whenever len < cap (suffix
                        // appends can grow mid-plane); slice_set needs a contiguous source.
                        // One copy per growth, amortized O(1) per stream.
                        k.slice_set(&sl.k.narrow(2, 0, sl.len)?.contiguous()?, 2, 0)?;
                        v.slice_set(&sl.v.narrow(2, 0, sl.len)?.contiguous()?, 2, 0)?;
                    }
                    sl.k = k;
                    sl.v = v;
                }
            }
        }
        let sl = slot.as_mut().expect("slot allocated above");
        sl.k.slice_set(&kf.contiguous()?, 2, sl.len)?;
        sl.v.slice_set(&vf.contiguous()?, 2, sl.len)?;
        sl.len += s_new;
        Ok((sl.k.narrow(2, 0, sl.len)?, sl.v.narrow(2, 0, sl.len)?))
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
        self.kvs.iter_mut().for_each(|v| *v = None);
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
        let gate = candle_nn::ops::silu(&linear_forward(&self.gate_proj, x)?)?;
        let up = linear_forward(&self.up_proj, x)?;
        linear_forward(&self.down_proj, &(gate * up)?)
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
        let q = linear_forward(&self.q_proj, x)?
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = linear_forward(&self.k_proj, x)?
            .reshape((b, seq_len, self.n_kv, self.head_dim))?
            .transpose(1, 2)?;
        let v = linear_forward(&self.v_proj, x)?
            .reshape((b, seq_len, self.n_kv, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = self.rope(&q, index_pos, cache)?;
        let k = self.rope(&k, index_pos, cache)?;

        // Resident KV (audit item 1): the step's rows are upcast ONCE (bf16→f32 is exact)
        // and written IN PLACE into the layer slot; attention reads a zero-copy narrow of
        // exactly the rows `Tensor::cat` used to rebuild — identical values, no per-token
        // O(ctx) copy, no full-cache re-upcast, no clone-back.
        // Resident KV stays in CHECKPOINT dtype (bf16 — torch's cache dtype): the step's
        // rb'd rows are written in place verbatim; no dtype conversion exists on the
        // append at all. f32 KV doubled every cache byte and every attention-read byte
        // for identical values (bf16→f32 is exact) — reverted on review.
        let (k, v) = if cache.use_kv_cache {
            cache.append_kv(block_idx, &k, &v, index_pos)?
        } else {
            (k, v)
        };

        let group = self.n_head / self.n_kv;
        let scale = (self.head_dim as f64).sqrt();
        // Decode shape: at seq_len==1 (no mask) the flag-on path runs flashkern attention
        // straight over the resident bf16 planes — per-head dots with in-register widening,
        // no repeat_kv materialization, no cache upcast, no candle op. Flag off pins the
        // reference expanded chain (upcast + repeat_kv + candle matmul) for parity runs.
        let flashkern_path = cache.grouped_gqa_decode
            && seq_len == 1
            && add_mask.is_none()
            && b == 1
            && x.device().is_cpu()
            && x.dtype() == DType::BF16
            && crate::bf16_gemm::bf16_gemm_nt_available();
        let y = if flashkern_path {
            self.attn_decode_flash(&q, &k, &v, x.dtype())?
        } else {
            let q = q.to_dtype(DType::F32)?;
            let k = repeat_kv(k.to_dtype(DType::F32)?.contiguous()?, group)?;
            let v = repeat_kv(v.to_dtype(DType::F32)?.contiguous()?, group)?;
            let att = (q.matmul(&k.t()?)? / scale)?;
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
            att.matmul(&v.contiguous()?)?.to_dtype(x.dtype())?
        };
        let y = y
            .transpose(1, 2)?
            .reshape((b, seq_len, self.n_head * self.head_dim))?;
        linear_forward(&self.o_proj, &y)
    }

    /// The flag-on decode attention: flashkern per-head dots over the resident bf16 KV
    /// planes (`flashkern::decode::attn_decode_bf16`) — q's post-rope bf16 bits in, one
    /// bf16 round at the per-head store out. Zero copies of K/V at any width.
    fn attn_decode_flash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out_dtype: DType,
    ) -> Result<Tensor> {
        use candle_core::Storage;
        fn bits_view<'a>(guard: &'a std::sync::RwLockReadGuard<'_, Storage>) -> Result<&'a [u16]> {
            match &**guard {
                Storage::Cpu(candle_core::CpuStorage::BF16(vv)) => {
                    // SAFETY: half::bf16 is repr(transparent) over u16.
                    Ok(unsafe { std::slice::from_raw_parts(vv.as_ptr() as *const u16, vv.len()) })
                }
                _ => candle_core::bail!("attn_decode_flash: expected CPU bf16 storage"),
            }
        }
        let q = q.contiguous()?;
        let (qs, ql) = q.storage_and_layout();
        let qb =
            &bits_view(&qs)?[ql.start_offset()..ql.start_offset() + self.n_head * self.head_dim];
        let (ks, kl) = k.storage_and_layout();
        let (vs, vl) = v.storage_and_layout();
        let len = kl.dims()[2];
        let head_stride = kl.stride()[1]; // cap·hd — the slot plane's head pitch
        debug_assert_eq!(head_stride, vl.stride()[1]);
        let kb = bits_view(&ks)?[kl.start_offset()..].as_ptr();
        let vb = bits_view(&vs)?[vl.start_offset()..].as_ptr();
        let mut out = vec![0u16; self.n_head * self.head_dim];
        // SAFETY: plane geometry from the narrow views' own layouts; storage guards live
        // until after the call; q/out sized by the asserts inside.
        unsafe {
            crate::flashkern::decode::attn_decode_bf16(
                qb,
                kb,
                vb,
                head_stride,
                len,
                self.n_head,
                self.n_kv,
                self.head_dim,
                &mut out,
            );
        }
        drop((qs, ks, vs));
        let out: Vec<half::bf16> = out.iter().map(|&x| half::bf16::from_bits(x)).collect();
        Tensor::from_vec(out, (1, self.n_head, 1, self.head_dim), q.device())?.to_dtype(out_dtype)
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
        let (_b, seq_len, _h) = x.dims3()?;
        let bcx = linear_forward(&self.in_proj, x)?.transpose(1, 2)?;

        // Fused decode fast path (candle-flashfftconv `conv1d_update`): when a carried
        // conv state exists (mid-stream continuation — decode steps and short suffix
        // chunks) do `B⊙x → K-tap causal conv → C⊙` in ONE dispatch with a register
        // window — no `[state|x]` concat staging, no gate intermediates. ~3.3× the
        // composed path at LFM2 shape (D=2048, K=3, bf16 Metal); rounding count
        // matches the CUDA-trained regime (Bx and conv-out round through bf16).
        // Sequence START (conv_states None) stays on the composed path below, whose
        // zero-pad prefill is the reference semantics.
        if cache.fused_conv_decode && self.l_cache > 0 && seq_len <= 4 && cache.use_kv_cache {
            if let Some(prev) = cache.conv_states[block_idx].clone() {
                // CPU device with the flashkern SIMD kernel built + supported → the
                // liquid-audio CustomOp (channel-vectorized NEON/AVX decode step); otherwise
                // the candle-flashfftconv op (JIT Metal kernel on Metal, scalar CPU
                // reference where flashkern isn't available). Same shapes, same
                // trained-regime rounding points — gated on availability, never degrading
                // silently (the flashkern op errors rather than falling back).
                let use_flashkern = bcx.device().is_cpu()
                    && crate::flashkern::candle_ops::conv1d_update_available();
                // One line per DEVICE (not per process): a live run PROVES this
                // path executed and on which silicon. Per-device matters — the
                // app can switch compute device between sessions, and a
                // process-wide `Once` would keep showing the first session's
                // device forever (exactly the misdirection that hid a CPU run).
                static FUSED_ANNOUNCED: std::sync::Mutex<Option<candle_core::DeviceLocation>> =
                    std::sync::Mutex::new(None);
                let loc = bcx.device().location();
                if let Ok(mut last) = FUSED_ANNOUNCED.lock() {
                    if *last != Some(loc) {
                        *last = Some(loc);
                        let kernel = if use_flashkern {
                            "flashkern conv1d_update"
                        } else {
                            "candle-flashfftconv causal_conv1d_update"
                        };
                        eprintln!("[voice] fused conv decode kernel active ({kernel}, {loc:?})");
                    }
                }
                let w = self.conv_weight.squeeze(1)?; // (H, K)
                let bcx = bcx.contiguous()?;
                let (y, new_state) = if use_flashkern {
                    crate::flashkern::candle_ops::causal_conv1d_update_fused(&bcx, &prev, &w)?
                } else {
                    candle_flashfftconv::causal_conv1d_update_fused(&bcx, &prev, &w)?
                };
                cache.conv_states[block_idx] = Some(new_state);
                let y = y.transpose(1, 2)?.contiguous()?;
                return linear_forward(&self.out_proj, &y);
            }
        }

        let bgate = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;
        let bx = (bgate * &x_proj)?.contiguous()?;

        // One causal depthwise short-conv path for prefill, decode, AND multi-token
        // continuation, through the FlashFFTConv `depthwise_conv1d_stream` kernel in the
        // model's NATIVE dtype — bf16 on Metal runs f32-accumulate / bf16-store (the
        // deployed, trained-around regime), no upcast. The carried state is keyed on
        // presence, not seq_len: at sequence start `conv_states` is `None` (fresh or
        // cleared cache) so prefill zero-pads exactly as the reference; when a suffix
        // chunk continues an existing stream (persistent cross-turn cache), the carried
        // K-1 inputs make chunked forward numerically equal to one full-sequence forward
        // — causal conv has no other cross-boundary dependence.
        let w = self.conv_weight.squeeze(1)?; // (H, K)
        let prev = if self.l_cache > 0 {
            cache.conv_states[block_idx].clone()
        } else {
            None
        };
        let (conv_out, new_cache) =
            candle_flashfftconv::depthwise_conv1d_stream(&bx, &w, prev.as_ref())?;
        if cache.use_kv_cache && self.l_cache > 0 {
            cache.conv_states[block_idx] = Some(new_cache);
        }

        let conv_out = (c * &conv_out)?.transpose(1, 2)?.contiguous()?;
        linear_forward(&self.out_proj, &conv_out)
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
        native_ctx: Option<&crate::flashkern::native_engine::BackboneCtxGuard>,
    ) -> Result<Tensor> {
        let residual = x;
        // Fused ShortConv residual block (flashkern): norm → in_proj → conv update →
        // out_proj → residual as ONE dispatch — no candle op, no transposes. Same gates as
        // the op-chain fast path (carried state exists, decode shape); sequence START keeps
        // the composed path (its zero-pad prefill is the reference semantics).
        if let LayerKind::ShortConv(c) = &self.kind {
            if cache.fused_conv_decode
                && x.device().is_cpu()
                && x.dtype() == DType::BF16
                && x.dims3().map(|(b, s, _)| b * s == 1).unwrap_or(false)
                && crate::bf16_gemm::bf16_gemm_nt_available()
                && cache.conv_states[block_idx].is_some()
            {
                // The whole layer in one native doorbell when the resident table is
                // live (bit-identical to the composed fused blocks by parity test).
                if let Some(y) = self.native_conv_layer(x, c, block_idx, cache, native_ctx)? {
                    return Ok(y);
                }
                if let Some(y) = self.fused_shortconv_decode(x, c, block_idx, cache)? {
                    let x = y;
                    if x.device().is_cpu()
                        && x.dtype() == DType::BF16
                        && crate::flashkern::decode::fused_mlp_available()
                        && x.dims3().map(|(b, s, _)| b * s == 1).unwrap_or(false)
                    {
                        if let Some(y) = self.fused_mlp_decode(&x)? {
                            return Ok(y);
                        }
                    }
                    let residual = &x;
                    let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
                    return h + residual;
                }
            }
        }
        // Whole attention layer in one native doorbell (grouped/ulp tier; the
        // reference chain and prefill keep the candle path below).
        if let LayerKind::Attention(a) = &self.kind {
            if add_mask.is_none() {
                if let Some(y) =
                    self.native_attn_layer(x, a, index_pos, block_idx, cache, native_ctx)?
                {
                    return Ok(y);
                }
            }
        }
        let h = self.operator_norm.forward(x)?;
        let h = match &self.kind {
            LayerKind::Attention(a) => a.forward(&h, index_pos, block_idx, cache, add_mask)?,
            LayerKind::ShortConv(c) => c.forward(&h, block_idx, cache)?,
        };
        let x = (h + residual)?;
        // Fused FFN residual block on the flashkern GPU-dispatch model: ONE threadgroup
        // dispatch (lanes + spin barriers + shared scratch) replaces the eight-op candle
        // chain (norm, 3 linears + casts, silu, gating mul, residual add) — CPU decode only;
        // Metal and prefill keep the op chain. Same bf16 rounding points, so numerics stay
        // in the trained regime; only the dispatch shape changes.
        if x.device().is_cpu()
            && x.dtype() == DType::BF16
            && crate::flashkern::decode::fused_mlp_available()
            && x.dims3().map(|(b, s, _)| b * s == 1).unwrap_or(false)
        {
            if let Some(y) = self.fused_mlp_decode(&x)? {
                return Ok(y);
            }
        }
        let residual = &x;
        let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        h + residual
    }

    /// The CPU-decode fused ShortConv block: zero-copy weight views into
    /// [`flashkern::decode::fused_shortconv_decode`] — norm, in_proj, the fused conv
    /// update (same kernel as the op path), out_proj, residual, in one dispatch. Returns
    /// `Ok(None)` if any operand isn't a contiguous CPU bf16 tensor.
    fn fused_shortconv_decode(
        &self,
        x: &Tensor,
        conv: &ShortConv,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Option<Tensor>> {
        use candle_core::Storage;
        fn bits<'a>(
            guard: &'a std::sync::RwLockReadGuard<'_, Storage>,
            layout: &candle_core::Layout,
        ) -> Option<&'a [u16]> {
            match &**guard {
                Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                    let (s, e) = layout.contiguous_offsets()?;
                    let sl = &v[s..e];
                    // SAFETY: half::bf16 is repr(transparent) over u16.
                    Some(unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) })
                }
                _ => None,
            }
        }
        let k = conv.l_cache;
        let hdim = x.dim(2)?;
        let x = x.contiguous()?;
        let w2d = conv.conv_weight.squeeze(1)?.contiguous()?;
        let state = cache.conv_states[block_idx].clone().expect("gated on Some");
        let state = state.contiguous()?;
        let (xs, xl) = x.storage_and_layout();
        let (ns, nl) = self.operator_norm.weight().storage_and_layout();
        let (is_, il) = conv.in_proj.weight().storage_and_layout();
        let (cs, cl) = w2d.storage_and_layout();
        let (os, ol) = conv.out_proj.weight().storage_and_layout();
        let (ss, sl) = state.storage_and_layout();
        let (Some(xb), Some(nb), Some(iw), Some(cw), Some(ow), Some(sb)) = (
            bits(&xs, xl),
            bits(&ns, nl),
            bits(&is_, il),
            bits(&cs, cl),
            bits(&os, ol),
            bits(&ss, sl),
        ) else {
            return Ok(None);
        };
        let weights = crate::flashkern::decode::FusedShortConvWeights {
            norm_w: nb,
            in_w: iw,
            conv_w: cw,
            out_w: ow,
            eps: self.operator_norm.eps() as f32,
            k,
        };
        let mut out = vec![0u16; hdim];
        let mut state_out = vec![0u16; hdim * (k - 1)];
        crate::flashkern::decode::fused_shortconv_decode(
            xb,
            &weights,
            sb,
            &mut state_out,
            &mut out,
            rayon::current_num_threads().max(1),
        );
        drop((xs, ns, is_, cs, os, ss));
        let state_out: Vec<half::bf16> = state_out
            .iter()
            .map(|&b| half::bf16::from_bits(b))
            .collect();
        cache.conv_states[block_idx] =
            Some(Tensor::from_vec(state_out, (1, hdim, k - 1), x.device())?);
        let out: Vec<half::bf16> = out.iter().map(|&b| half::bf16::from_bits(b)).collect();
        Ok(Some(Tensor::from_vec(out, (1, 1, hdim), x.device())?))
    }

    /// The CPU-decode fused FFN block: zero-copy bf16 views of the checkpoint weights
    /// (`storage_and_layout`) into [`flashkern::decode::fused_mlp_decode`]. Returns
    /// `Ok(None)` if any operand isn't a contiguous CPU bf16 tensor (caller keeps the op
    /// chain) — an availability gate, not a silent numeric fallback.
    fn fused_mlp_decode(&self, x: &Tensor) -> Result<Option<Tensor>> {
        use candle_core::Storage;

        fn bits<'a>(
            guard: &'a std::sync::RwLockReadGuard<'_, Storage>,
            layout: &candle_core::Layout,
        ) -> Option<&'a [u16]> {
            match &**guard {
                Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                    let (s, e) = layout.contiguous_offsets()?;
                    let sl = &v[s..e];
                    // SAFETY: half::bf16 is repr(transparent) over u16.
                    Some(unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u16, sl.len()) })
                }
                _ => None,
            }
        }

        let x = x.contiguous()?;
        let hdim = x.dim(2)?;
        let (xs, xl) = x.storage_and_layout();
        let (ns, nl) = self.ffn_norm.weight().storage_and_layout();
        let (g1, l1) = self.mlp.gate_proj.weight().storage_and_layout();
        let (g3, l3) = self.mlp.up_proj.weight().storage_and_layout();
        let (g2, l2) = self.mlp.down_proj.weight().storage_and_layout();
        let (Some(xb), Some(nb), Some(w1), Some(w3), Some(w2)) = (
            bits(&xs, xl),
            bits(&ns, nl),
            bits(&g1, l1),
            bits(&g3, l3),
            bits(&g2, l2),
        ) else {
            return Ok(None);
        };
        let weights = crate::flashkern::decode::FusedMlpWeights {
            norm_w: nb,
            w1,
            w3,
            w2,
            eps: self.ffn_norm.eps() as f32,
        };
        let mut out = vec![0u16; hdim];
        // The resident native stage machine when it is up — bit-identical to the
        // threadgroup port by parity test, so the fallback changes scheduling only,
        // never numerics.
        {
            // One lanes value for the attempt AND its parity fallback (the contract is
            // "bit-identical at the same lanes") — sized from OUR team when it exists.
            let engine = crate::flashkern::native_engine::process_engine();
            let lanes = engine.lanes_total().max(1);
            let ran_native = engine.fused_mlp(xb, &weights, &mut out, lanes);
            if !ran_native {
                crate::flashkern::decode::fused_mlp_decode(xb, &weights, &mut out, lanes);
            }
        }
        let out: Vec<half::bf16> = out.iter().map(|&b| half::bf16::from_bits(b)).collect();
        Ok(Some(Tensor::from_vec(out, (1, 1, hdim), x.device())?))
    }
}

impl DecoderLayer {
    /// Capture this layer's weights for the resident native layer table. Conv layers
    /// only; attention layers get a placeholder until rung 2. Derived tensors (the
    /// squeezed-contiguous conv weight) are pushed into `held`, whose owner must keep
    /// them alive for the table's lifetime.
    fn native_conv_desc(
        &self,
        held: &mut Vec<Tensor>,
    ) -> Option<crate::flashkern::native_engine::LayerDesc> {
        use crate::flashkern::decode::PtrLen;
        use crate::flashkern::native_engine::LayerDesc;
        let conv = match &self.kind {
            LayerKind::ShortConv(c) => c,
            LayerKind::Attention(a) => {
                // Attention capture (rung 2). Any failure degrades THIS slot to
                // unserved (q_w null) — conv layers still run natively.
                let attn = (|| -> Option<LayerDesc> {
                    Some(LayerDesc {
                        kind: 1,
                        k: 0,
                        op_eps: self.operator_norm.eps() as f32,
                        ffn_eps: self.ffn_norm.eps() as f32,
                        op_norm_w: PtrLen::bf16(self.operator_norm.weight())?.addr() as *const u16,
                        ffn_norm_w: PtrLen::bf16(self.ffn_norm.weight())?.addr() as *const u16,
                        in_w: std::ptr::null(),
                        conv_w: std::ptr::null(),
                        out_w: std::ptr::null(),
                        w1: PtrLen::bf16(self.mlp.gate_proj.weight())?.addr() as *const u16,
                        w3: PtrLen::bf16(self.mlp.up_proj.weight())?.addr() as *const u16,
                        w2: PtrLen::bf16(self.mlp.down_proj.weight())?.addr() as *const u16,
                        n_head: a.n_head as u32,
                        n_kv: a.n_kv as u32,
                        hd: a.head_dim as u32,
                        qk_eps: a.q_norm.eps() as f32,
                        q_w: PtrLen::bf16(a.q_proj.weight())?.addr() as *const u16,
                        k_w: PtrLen::bf16(a.k_proj.weight())?.addr() as *const u16,
                        v_w: PtrLen::bf16(a.v_proj.weight())?.addr() as *const u16,
                        o_w: PtrLen::bf16(a.o_proj.weight())?.addr() as *const u16,
                        qn_w: PtrLen::bf16(a.q_norm.weight())?.addr() as *const u16,
                        kn_w: PtrLen::bf16(a.k_norm.weight())?.addr() as *const u16,
                    })
                })();
                return Some(attn.unwrap_or_else(LayerDesc::attn_placeholder));
            }
        };
        let w2d = conv.conv_weight.squeeze(1).ok()?.contiguous().ok()?;
        let desc = LayerDesc {
            kind: 0,
            k: conv.l_cache as u32,
            op_eps: self.operator_norm.eps() as f32,
            ffn_eps: self.ffn_norm.eps() as f32,
            op_norm_w: PtrLen::bf16(self.operator_norm.weight())?.addr() as *const u16,
            ffn_norm_w: PtrLen::bf16(self.ffn_norm.weight())?.addr() as *const u16,
            in_w: PtrLen::bf16(conv.in_proj.weight())?.addr() as *const u16,
            conv_w: PtrLen::bf16(&w2d)?.addr() as *const u16,
            out_w: PtrLen::bf16(conv.out_proj.weight())?.addr() as *const u16,
            w1: PtrLen::bf16(self.mlp.gate_proj.weight())?.addr() as *const u16,
            w3: PtrLen::bf16(self.mlp.up_proj.weight())?.addr() as *const u16,
            w2: PtrLen::bf16(self.mlp.down_proj.weight())?.addr() as *const u16,
            ..LayerDesc::attn_placeholder()
        };
        held.push(w2d);
        Some(desc)
    }

    /// The whole attention layer (attention + MLP) in ONE native doorbell. The
    /// engine replicates the GROUPED decode path (ulp tier), so this only runs when
    /// `grouped_gqa_decode` is on — the reference chain keeps the candle path.
    /// `Ok(None)` = unserved (caller keeps the existing mixed path, bit-identical).
    fn native_attn_layer(
        &self,
        x: &Tensor,
        attn: &Attention,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
        native_ctx: Option<&crate::flashkern::native_engine::BackboneCtxGuard>,
    ) -> Result<Option<Tensor>> {
        if !(cache.grouped_gqa_decode
            && cache.use_kv_cache
            && x.device().is_cpu()
            && x.dtype() == DType::BF16
            && x.dims3().map(|(b, s, _)| b * s == 1).unwrap_or(false)
            && crate::bf16_gemm::bf16_gemm_nt_available())
        {
            return Ok(None);
        }
        let Some(ctx) = native_ctx else {
            return Ok(None);
        };
        let hdim = x.dim(2)?;
        let (n_kv, hd) = (attn.n_kv, attn.head_dim);
        // Grow BEFORE capturing plane pointers — growth reallocates the planes.
        cache.ensure_kv_capacity(block_idx, 1, n_kv, hd, index_pos + 1, x.dtype(), x.device())?;
        let Some(sl) = cache.kvs[block_idx].as_ref() else {
            return Ok(None);
        };
        if sl.len != index_pos {
            return Ok(None); // cursor out of step — let the candle path sort it out
        }
        let cap = sl.k.dim(2)?;
        let head_stride = cap * hd;

        let x = x.contiguous()?;
        let (xs, xl) = x.storage_and_layout();
        let (ks, kl) = sl.k.storage_and_layout();
        let (vs, vl) = sl.v.storage_and_layout();
        let (cs, cl) = cache.cos.storage_and_layout();
        let (ss, sl2) = cache.sin.storage_and_layout();
        let bits = |g: &std::sync::RwLockReadGuard<'_, candle_core::Storage>,
                    l: &candle_core::Layout|
         -> Option<(*const u16, usize)> {
            match &**g {
                candle_core::Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                    let (a, b) = l.contiguous_offsets()?;
                    Some((v[a..b].as_ptr() as *const u16, b - a))
                }
                _ => None,
            }
        };
        let (Some((xp, xl)), Some((kp, kl)), Some((vp, vl)), Some((cp, cl)), Some((sp, sl))) = (
            bits(&xs, xl),
            bits(&ks, kl),
            bits(&vs, vl),
            bits(&cs, cl),
            bits(&ss, sl2),
        ) else {
            return Ok(None);
        };
        drop((xs, ks, vs, cs, ss));
        if xl != hdim {
            return Ok(None);
        }
        // SAFETY (transitional Candle rim): `&mut Cache` gives this call exclusive
        // ownership of the K/V tensors, and the tensors keep their allocations alive
        // after the read guards used for pointer capture are dropped. C++ validates
        // every supplied extent and the owning context id before mutating a row.
        let (kp, vp) = (kp as *mut u16, vp as *mut u16);
        let xb = unsafe { std::slice::from_raw_parts(xp, hdim) };
        let mut out = vec![0u16; hdim];
        let lanes = ctx.lanes_total().max(1);
        let ok = unsafe {
            ctx.attn_layer(
                block_idx,
                xb,
                kp,
                kl,
                vp,
                vl,
                head_stride,
                index_pos,
                cp,
                sp,
                cl.min(sl),
                &mut out,
                lanes,
            )
        };
        if !ok {
            return Ok(None);
        }
        cache.advance_kv_cursor(block_idx);
        let out: Vec<half::bf16> = out.iter().map(|&b| half::bf16::from_bits(b)).collect();
        Ok(Some(Tensor::from_vec(out, (1, 1, hdim), x.device())?))
    }

    /// The whole conv layer (shortconv + MLP) in ONE native doorbell — bit-identical
    /// to the composed fused blocks. `Ok(None)` when the engine/ctx isn't live or an
    /// operand isn't a contiguous CPU bf16 tensor (caller keeps the per-block path).
    fn native_conv_layer(
        &self,
        x: &Tensor,
        conv: &ShortConv,
        block_idx: usize,
        cache: &mut Cache,
        native_ctx: Option<&crate::flashkern::native_engine::BackboneCtxGuard>,
    ) -> Result<Option<Tensor>> {
        let Some(ctx) = native_ctx else {
            return Ok(None);
        };
        let k = conv.l_cache;
        let hdim = x.dim(2)?;
        let x = x.contiguous()?;
        let state = cache.conv_states[block_idx].clone().expect("gated on Some");
        let state = state.contiguous()?;
        let (xs, xl) = x.storage_and_layout();
        let (ss, sl) = state.storage_and_layout();
        let bits = |g: &std::sync::RwLockReadGuard<'_, candle_core::Storage>,
                    l: &candle_core::Layout|
         -> Option<*const u16> {
            match &**g {
                candle_core::Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                    let (a, b) = l.contiguous_offsets()?;
                    Some(v[a..b].as_ptr() as *const u16)
                }
                _ => None,
            }
        };
        let (Some(xp), Some(sp)) = (bits(&xs, xl), bits(&ss, sl)) else {
            return Ok(None);
        };
        // SAFETY: xp/sp point into guarded storages held live across the blocking call.
        let (xb, sb) = unsafe {
            (
                std::slice::from_raw_parts(xp, hdim),
                std::slice::from_raw_parts(sp, hdim * (k - 1)),
            )
        };
        let mut out = vec![0u16; hdim];
        let mut state_out = vec![0u16; hdim * (k - 1)];
        let lanes = ctx.lanes_total().max(1);
        if !ctx.conv_layer(block_idx, xb, sb, &mut state_out, &mut out, lanes) {
            return Ok(None);
        }
        drop((xs, ss));
        let state_out: Vec<half::bf16> = state_out
            .iter()
            .map(|&b| half::bf16::from_bits(b))
            .collect();
        cache.conv_states[block_idx] =
            Some(Tensor::from_vec(state_out, (1, hdim, k - 1), x.device())?);
        let out: Vec<half::bf16> = out.iter().map(|&b| half::bf16::from_bits(b)).collect();
        Ok(Some(Tensor::from_vec(out, (1, 1, hdim), x.device())?))
    }
}

/// HF `Lfm2Model` — returns the all-position last hidden state (post embedding-norm).
pub struct Model {
    /// Resident native-engine layer table guard. MUST be declared before the weight
    /// fields: Rust drops fields in declaration order, and the guard clears the C-side
    /// pointer table before the weights it points into are freed.
    native_ctx: Option<crate::flashkern::native_engine::BackboneCtxGuard>,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
}

impl Model {
    /// Build + install the resident native-engine layer table (rung 1: conv layers;
    /// attention slots are placeholders). No-op when any capture fails or the engine
    /// is unavailable — the per-block paths remain, bit-identical.
    pub fn install_native_ctx(&mut self, max_ctx: usize) {
        if self.native_ctx.is_some() {
            return;
        }
        let mut held = Vec::new();
        let mut descs = Vec::with_capacity(self.layers.len());
        let mut h = 0usize;
        let mut ffn = 0usize;
        for layer in &self.layers {
            let Some(d) = layer.native_conv_desc(&mut held) else {
                return; // a conv layer failed capture — table would be partial
            };
            if d.kind == 0 {
                use crate::flashkern::decode::PtrLen;
                if h == 0 {
                    h = PtrLen::bf16(layer.operator_norm.weight())
                        .map(|p| p.size())
                        .unwrap_or(0);
                    let w1_len = PtrLen::bf16(layer.mlp.gate_proj.weight())
                        .map(|p| p.size())
                        .unwrap_or(0);
                    if h > 0 {
                        ffn = w1_len / h;
                    }
                }
            }
            descs.push(d);
        }
        if h == 0 || ffn == 0 {
            return;
        }
        self.native_ctx =
            crate::flashkern::native_engine::install_backbone_ctx(&descs, h, ffn, max_ctx, held);
    }

    /// Final pre-logits norm (head-table capture for the native token pass).
    pub(crate) fn embedding_norm(&self) -> &RmsNorm {
        &self.embedding_norm
    }

    pub(crate) fn native_ctx(&self) -> Option<&crate::flashkern::native_engine::BackboneCtxGuard> {
        self.native_ctx.as_ref()
    }

    /// ONE token through the native engine: embed → every layer → final norm →
    /// (optionally) logits into the caller's buffers. Returns `Ok(false)` when any
    /// gate fails — the caller takes the candle path, bit-identical. On success every
    /// attention cursor and nothing else has advanced; the caller still owns
    /// `index_pos`. Steady state this allocates NOTHING: the state table is
    /// cache-resident (entries rewritten — fresh pointer captures each token are the
    /// correctness mechanism against rollback clones, candle-path interleaves, and
    /// KV growth) and the outputs land in caller-owned storage.
    pub(crate) fn native_token_pass(
        &self,
        cache: &mut Cache,
        index_pos: usize,
        ids: &[u32],
        embed_kind: u32,
        out_hidden: &mut [u16],
        out_logits: Option<&mut [f32]>,
    ) -> Result<bool> {
        use crate::flashkern::decode::PtrLen;
        use crate::flashkern::native_engine::LayerState;
        if !(cache.grouped_gqa_decode
            && cache.use_kv_cache
            && cache.fused_conv_decode
            && crate::bf16_gemm::bf16_gemm_nt_available())
        {
            return Ok(false);
        }
        let Some(ctx) = self.native_ctx.as_ref() else {
            return Ok(false);
        };
        let hdim = self.embed_tokens.embeddings().dim(1)?;
        assert_eq!(out_hidden.len(), hdim, "native_token_pass: out_hidden != H");
        // Per-layer state: ensure attention capacity FIRST (growth reallocates), then
        // capture fresh pointers into the resident table. Any miss → unserved.
        cache.native_states.0.clear();
        cache.native_states.0.reserve(self.layers.len());
        for (l, layer) in self.layers.iter().enumerate() {
            match &layer.kind {
                LayerKind::Attention(a) => {
                    cache.ensure_kv_capacity(
                        l,
                        1,
                        a.n_kv,
                        a.head_dim,
                        index_pos + 1,
                        DType::BF16,
                        &candle_core::Device::Cpu,
                    )?;
                    let Some(sl) = cache.kvs[l].as_ref() else {
                        return Ok(false);
                    };
                    if sl.len != index_pos {
                        return Ok(false);
                    }
                    let (Some(kp), Some(vp)) = (PtrLen::bf16(&sl.k), PtrLen::bf16(&sl.v)) else {
                        return Ok(false);
                    };
                    let cap = sl.k.dim(2)?;
                    cache.native_states.0.push(LayerState {
                        k_plane: kp.addr() as *mut u16,
                        v_plane: vp.addr() as *mut u16,
                        head_stride: cap * a.head_dim,
                        k_len: kp.size(),
                        v_len: vp.size(),
                        conv_state: std::ptr::null_mut(),
                        conv_len: 0,
                    });
                }
                LayerKind::ShortConv(_) => {
                    let Some(st) = cache.conv_states[l].as_ref() else {
                        return Ok(false);
                    };
                    let Some(sp) = PtrLen::bf16(st) else {
                        return Ok(false);
                    };
                    let mut ls = LayerState::none();
                    // SAFETY (in-place advance): the engine shifts the carried window
                    // through this pointer — the same in-place storage mutation
                    // candle's slice_set performs; decode is sequential and this
                    // thread blocks for the pass.
                    ls.conv_state = sp.addr() as *mut u16;
                    ls.conv_len = sp.size();
                    cache.native_states.0.push(ls);
                }
            }
        }
        let (Some(cosp), Some(sinp)) = (PtrLen::bf16(&cache.cos), PtrLen::bf16(&cache.sin)) else {
            return Ok(false);
        };
        // Tile counts come from OUR team's width, not a foreign pool's. Both resolve
        // to the P-core count today, so the pinned banding is unchanged.
        let lanes = ctx.lanes_total().max(1);
        // SAFETY: all raw pointers were captured from live cache/model tensors after
        // capacity growth; `&mut Cache` excludes concurrent state access. C++ checks
        // the context id and every extent before dispatch.
        let ok = unsafe {
            ctx.token_pass(
                ids,
                embed_kind,
                &cache.native_states.0,
                index_pos,
                cosp.addr() as *const u16,
                sinp.addr() as *const u16,
                cosp.size().min(sinp.size()),
                out_hidden,
                out_logits,
                lanes,
            )
        };
        if !ok {
            return Ok(false);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            if matches!(layer.kind, LayerKind::Attention(_)) {
                cache.advance_kv_cursor(l);
            }
        }
        Ok(true)
    }

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
            native_ctx: None,
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
            hidden = layer.forward(
                &hidden,
                index_pos,
                block_idx,
                cache,
                add_mask,
                self.native_ctx.as_ref(),
            )?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_snapshot_owns_independent_conv_state() {
        use crate::flashkern::decode::PtrLen;
        use half::bf16;

        let dev = Device::Cpu;
        let cfg = Lfm2Config {
            vocab_size: 16,
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 8,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: vec![LayerType::Conv],
            block_ffn_dim_multiplier: 1.0,
            block_multiple_of: 1,
            block_ff_dim: 4,
            block_auto_adjust_ff_dim: false,
        };
        let mut cache = Cache::new(true, DType::BF16, &cfg, &dev).unwrap();
        cache.conv_states[0] = Some(
            Tensor::from_vec(
                vec![
                    bf16::from_f32(1.0),
                    bf16::from_f32(2.0),
                    bf16::from_f32(3.0),
                    bf16::from_f32(4.0),
                ],
                (1, 2, 2),
                &dev,
            )
            .unwrap(),
        );

        let live = PtrLen::bf16(cache.conv_states[0].as_ref().unwrap())
            .unwrap()
            .addr();
        let snap = cache.snapshot().unwrap();
        let saved = PtrLen::bf16(snap.conv_states[0].as_ref().unwrap())
            .unwrap()
            .addr();
        assert_ne!(live, saved, "snapshot must not alias in-place native state");

        cache.rollback(&snap).unwrap();
        let restored = PtrLen::bf16(cache.conv_states[0].as_ref().unwrap())
            .unwrap()
            .addr();
        assert_ne!(
            saved, restored,
            "rollback must preserve a reusable snapshot"
        );
        assert_eq!(
            cache.conv_states[0]
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap(),
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ]
        );
    }

    #[test]
    fn grouped_gqa_matches_expanded_at_f32_ulp() {
        // The grouped_gqa_decode path must compute the SAME per-head attention as the
        // reference expanded (repeat_kv) form — identical dot products, different GEMM
        // tiling — so the divergence budget is f32 summation-order ulps, not structure.
        // This test pins that bound in-tree; byte-parity runs pin grouped_gqa_decode=false.
        let dev = Device::Cpu;
        let (b, nh, nkv, hd, len) = (1usize, 32usize, 8usize, 64usize, 333usize);
        let group = nh / nkv;
        let rnd = |n: usize, s: u64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(s);
                    ((x >> 33) as f32 / 2f32.powi(30)) - 1.0
                })
                .collect()
        };
        let q = Tensor::from_vec(rnd(b * nh * hd, 1), (b, nh, 1, hd), &dev).unwrap();
        let k = Tensor::from_vec(rnd(b * nkv * len * hd, 2), (b, nkv, len, hd), &dev).unwrap();
        let v = Tensor::from_vec(rnd(b * nkv * len * hd, 3), (b, nkv, len, hd), &dev).unwrap();
        let scale = (hd as f64).sqrt();
        // reference: expanded heads via repeat_kv (the s>1 / parity-pinned form)
        let ke = repeat_kv(k.clone(), group).unwrap();
        let ve = repeat_kv(v.clone(), group).unwrap();
        let att = (q.matmul(&ke.t().unwrap()).unwrap() / scale).unwrap();
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1).unwrap();
        let ye = att.matmul(&ve.contiguous().unwrap()).unwrap();
        // decode path: grouped view, no materialization
        let qg = q.reshape((b, nkv, group, hd)).unwrap();
        let att = (qg.matmul(&k.t().unwrap()).unwrap() / scale).unwrap();
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1).unwrap();
        let yg = att.matmul(&v).unwrap().reshape((b, nh, 1, hd)).unwrap();
        let a: Vec<f32> = ye.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = yg.flatten_all().unwrap().to_vec1().unwrap();
        let (mut md, mut sc) = (0f32, 1e-6f32);
        for (x, y) in a.iter().zip(&g) {
            md = md.max((x - y).abs());
            sc = sc.max(x.abs());
        }
        assert!(md / sc < 1e-5, "grouped vs expanded rel {}", md / sc);
    }

    #[test]
    fn append_kv_growth_preserves_independent_v_dtype() {
        let dev = Device::Cpu;
        let cfg = Lfm2Config {
            vocab_size: 16,
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 8,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: vec![LayerType::FullAttention],
            block_ffn_dim_multiplier: 1.0,
            block_multiple_of: 1,
            block_ff_dim: 4,
            block_auto_adjust_ff_dim: false,
        };
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev).unwrap();

        let k0 = Tensor::zeros((1, 1, 256, 2), DType::F32, &dev).unwrap();
        let v0 = Tensor::zeros((1, 1, 256, 2), DType::BF16, &dev).unwrap();
        cache.append_kv(0, &k0, &v0, 0).unwrap();

        let k1 = Tensor::zeros((1, 1, 1, 2), DType::F32, &dev).unwrap();
        let v1 = Tensor::zeros((1, 1, 1, 2), DType::BF16, &dev).unwrap();
        let (k, v) = cache.append_kv(0, &k1, &v1, 256).unwrap();

        assert_eq!(k.dtype(), DType::F32);
        assert_eq!(v.dtype(), DType::BF16);
        assert_eq!(cache.kvs[0].as_ref().unwrap().v.dtype(), DType::BF16);
    }
}
