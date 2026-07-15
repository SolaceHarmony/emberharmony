//! Port of `liquid_audio/model/conformer/encoder.py` (NeMo ConformerEncoder).
//!
//! Inference path: `dw_striding` ConvSubsampling → RelPositionalEncoding →
//! N × ConformerLayer → optional out projection. For a single offline clip with
//! unlimited attention context (`att_context_size = [-1,-1]`) and no padding, the
//! attention/pad masks are identity, so they are passed as `None`. Streaming,
//! cache, stochastic depth, reduction, and export paths are not ported.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::mha::RelPositionalEncoding;
use super::modules::ConformerLayer;
use super::subsampling::ConvSubsampling;
use super::utils::{CacheAwareStreamingConfig, IntOrPair};
use crate::model::linear::linear_forward;

/// Python `pos_emb_max_len` default (encoder.py `__init__`); also the
/// `RelPositionalEncoding` table max length.
const POS_EMB_MAX_LEN: usize = 5000;

/// Subset of `ConformerEncoderConfig` needed for the offline forward path.
#[derive(Debug, Clone)]
pub struct ConformerEncoderConfig {
    pub feat_in: usize,
    pub feat_out: usize, // -1 in Python → 0 here means "= d_model"
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling_factor: usize,
    pub subsampling_conv_channels: usize, // 0 → = d_model
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub xscaling: bool,
    /// `rel_pos` (the model's config) or `abs_pos` — selects the per-layer attention.
    pub self_attention_model: String,
}

/// Python union for `conv_context_size`: the string `"causal"` or a list of two
/// integers `[left, right]` with `left + right + 1 == conv_kernel_size`. Used by
/// [`ConformerEncoder::calc_context_sizes`] (encoder.py L805-851).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvContextSize {
    /// `"causal"` → resolves to `[conv_kernel_size - 1, 0]`.
    Causal,
    /// `[left, right]`.
    Size(i64, i64),
}

pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    pos_enc: RelPositionalEncoding,
    layers: Vec<ConformerLayer>,
    out_proj: Option<Linear>,
    // ---- Config / streaming state (mirrors the Python `__init__` attributes;
    // cold on the offline forward but maintained 1:1 for the streaming/export
    // methods). `att_context_style`/`self_attention_model` are the offline
    // defaults ("regular"/"rel_pos") — the only path this encoder supports. ----
    feat_in: usize,
    d_model: usize,
    n_layers: usize,
    subsampling_factor: usize,
    att_context_style: String,
    self_attention_model: String,
    /// `att_context_size_all` — every configured look-ahead (offline: `[[-1,-1]]`).
    att_context_size_all: Vec<Vec<i64>>,
    /// `att_context_size` — the current `[left, right]` (mutable via streaming setters).
    att_context_size: Vec<i64>,
    /// `att_context_probs` — sampling distribution over `att_context_size_all`.
    att_context_probs: Vec<f64>,
    /// `conv_context_size` — resolved `[left, right]` depthwise-conv context.
    conv_context_size: (i64, i64),
    pos_emb_max_len: usize,
    /// `max_audio_length` — grows the (on-the-fly) positional table; see `set_max_audio_length`.
    max_audio_length: usize,
    /// `use_pad_mask` — pad-masking toggle (offline single clip ⇒ effectively off).
    use_pad_mask: bool,
    /// `export_cache_support` — whether the export path exposes streaming caches.
    export_cache_support: bool,
    /// `streaming_cfg` — the cache-aware streaming parameters (`setup_streaming_params`).
    streaming_cfg: CacheAwareStreamingConfig,
}

impl ConformerEncoder {
    pub fn new(cfg: &ConformerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d_ff = cfg.d_model * cfg.ff_expansion_factor;
        let conv_channels = if cfg.subsampling_conv_channels == 0 {
            cfg.d_model
        } else {
            cfg.subsampling_conv_channels
        };

        let pre_encode = ConvSubsampling::new(
            cfg.subsampling_factor,
            cfg.feat_in,
            cfg.d_model,
            conv_channels,
            vb.pp("pre_encode"),
        )?;

        let xscale = if cfg.xscaling {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };
        let pos_enc = RelPositionalEncoding::new(cfg.d_model, xscale);

        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ConformerLayer::new(
                cfg.d_model,
                d_ff,
                cfg.n_heads,
                cfg.conv_kernel_size,
                true,
                &cfg.self_attention_model,
                layers_vb.pp(i.to_string()),
            )?);
        }

        let out_proj = if cfg.feat_out > 0 && cfg.feat_out != cfg.d_model {
            Some(linear(cfg.d_model, cfg.feat_out, vb.pp("out_proj"))?)
        } else {
            None
        };

        // Resolve the att/conv context sizes from the offline defaults (Python
        // `_calc_context_sizes` with `att_context_size=None` ⇒ `[[-1,-1]]`,
        // `att_context_style="regular"`, `conv_context_size=None`). This encoder
        // only supports the `rel_pos` / unlimited-context offline path.
        let (att_context_size_all, att_context_size, att_context_probs, conv_ctx) =
            Self::calc_context_sizes(
                None,
                None,
                None,
                "regular",
                None,
                cfg.conv_kernel_size as i64,
            )?;
        let conv_context_size = match conv_ctx {
            ConvContextSize::Size(l, r) => (l, r),
            ConvContextSize::Causal => (cfg.conv_kernel_size as i64 - 1, 0),
        };

        // Python `__init__`: `set_max_audio_length(pos_emb_max_len)` then
        // `setup_streaming_params()`, then `export_cache_support = False`.
        let streaming_cfg = Self::compute_streaming_cfg(
            &att_context_size,
            "regular",
            cfg.n_layers,
            conv_context_size,
            cfg.subsampling_factor,
            IntOrPair::Pair(1, cfg.subsampling_factor as i64), // pre_encode.get_sampling_frames()
            IntOrPair::Pair(0, cfg.subsampling_factor as i64 + 1), // get_streaming_cache_size()
            None,
            None,
            None,
            10_000,
        );

        Ok(Self {
            pre_encode,
            pos_enc,
            layers,
            out_proj,
            feat_in: cfg.feat_in,
            d_model: cfg.d_model,
            n_layers: cfg.n_layers,
            subsampling_factor: cfg.subsampling_factor,
            att_context_style: "regular".to_string(),
            self_attention_model: cfg.self_attention_model.clone(),
            att_context_size_all,
            att_context_size,
            att_context_probs,
            conv_context_size,
            pos_emb_max_len: POS_EMB_MAX_LEN,
            max_audio_length: POS_EMB_MAX_LEN,
            use_pad_mask: true,
            export_cache_support: false,
            streaming_cfg,
        })
    }

    /// `audio_signal` is `(B, feat_in, T)` (mel features). Returns `(B, d_out, T')`.
    ///
    /// **Contract: one unpadded clip (effectively `B==1`, all `T` frames valid).**
    /// The padded-batch machinery — `MaskedConvSequential`, per-step length
    /// tracking, and `_create_masks` (`att_mask`/`pad_mask`) — is intentionally
    /// not ported; masks are `None`. Callers with multiple segments must encode
    /// each individually (as `_prefill` does), which is numerically equivalent to
    /// the Python padded-batch+length-mask path (verified in `prefill_parity`,
    /// 2 segments, 1.1e-6) precisely because that masking only exists to neutralize
    /// padding. Do NOT feed a zero-padded batch here.
    pub fn forward(&self, audio_signal: &Tensor) -> Result<Tensor> {
        let x = audio_signal.transpose(1, 2)?.contiguous()?; // (B, T, feat_in)
        let x = self.pre_encode.forward(&x)?; // (B, T', d_model)
        let (mut x, pos_emb) = self.pos_enc.forward(&x)?;
        for layer in &self.layers {
            x = layer.forward(&x, None, &pos_emb, None)?;
        }
        if let Some(p) = &self.out_proj {
            x = linear_forward(p, &x)?;
        }
        x.transpose(1, 2)?.contiguous() // (B, d_out, T')
    }

    /// PORT: `forward_internal` cache-aware STREAMING path (encoder.py L537-702).
    ///
    /// `audio_signal`: `(B, feat_in, T)` mel features for this chunk. The caches are the
    /// per-layer stacks from [`Self::get_initial_cache_state`] (or a previous step):
    /// `cache_last_channel` `(n_layers, B, cache_len, d_model)` (attention KV),
    /// `cache_last_time` `(n_layers, B, d_model, T_cache)` (depthwise-conv state),
    /// `cache_last_channel_len` `(B,)` (valid cached frames). Returns
    /// `(encoded (B, d_out, T'), length (B,), next_channel, next_time, next_channel_len)`.
    ///
    /// `setup_streaming_params` must have run (sets `streaming_cfg` + the per-layer
    /// `cache_drop_size`). The att context is `self.att_context_size`.
    pub fn forward_streaming(
        &self,
        audio_signal: &Tensor,
        length: Option<&Tensor>,
        cache_last_channel: &Tensor,
        cache_last_time: &Tensor,
        cache_last_channel_len: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let device = audio_signal.device();
        let b = audio_signal.dim(0)?;

        let in_len: Vec<i64> = match length {
            Some(l) => l.to_dtype(DType::I64)?.to_vec1::<i64>()?,
            None => vec![audio_signal.dim(2)? as i64; b],
        };

        // pre_encode: (B, feat_in, T) -> (B, T', d_model); track length via calc_length.
        let x = audio_signal.transpose(1, 2)?.contiguous()?; // (B, T, feat_in)
        let mut x = self.pre_encode.forward(&x)?; // (B, T', d_model)
        let mut length: Vec<i64> = self.pre_encode.out_lengths(&in_len);

        // drop_extra_pre_encoded: in the streaming entry `cache_last_channel is not None`
        // is ALWAYS true, so Python drops the leading `drop` pre-encoded frames every
        // chunk (the first chunk too — those frames are streaming warm-up).
        let drop = self.streaming_cfg.drop_extra_pre_encoded.max(0) as usize;
        let cllen: Vec<i64> = cache_last_channel_len
            .to_dtype(DType::I64)?
            .to_vec1::<i64>()?;
        if drop > 0 {
            let t = x.dim(1)?;
            x = x.narrow(1, drop, t.saturating_sub(drop))?;
            length = length.iter().map(|&l| (l - drop as i64).max(0)).collect();
        }

        let max_audio_pre = x.dim(1)?; // before adding cache_len (Python's cache_keep base)
        let cache_len = self.streaming_cfg.last_channel_cache_size.max(0) as usize;
        // cache_keep_size = max_audio_length - cache_drop_size — Python does NOT clamp,
        // so this is signed (negative when the lookahead exceeds the chunk).
        let cache_keep: i64 = max_audio_pre as i64 - self.streaming_cfg.cache_drop_size;
        let max_audio_length = max_audio_pre + cache_len;
        let padding_length: Vec<i64> = length.iter().map(|&l| l + cache_len as i64).collect();
        let offset: Vec<i64> = cllen.iter().map(|&l| -l + cache_len as i64).collect();

        // pos-enc shifted by the cache length.
        let (x_scaled, pos_emb) = self.pos_enc.forward_cache(&x, cache_len)?;
        let mut x = x_scaled;

        // masks over cache+current, then drop the cache region from the QUERY axis.
        let ctx: [i64; 2] = [self.att_context_size[0], self.att_context_size[1]];
        let pad_t = Tensor::from_vec(padding_length, (b,), device)?;
        let off_t = Tensor::from_vec(offset, (b,), device)?;
        let (pad_mask, att_mask) =
            self.create_masks(ctx, &pad_t, max_audio_length, Some(&off_t), device)?;
        let t_cur = max_audio_length - cache_len;
        let pad_mask = pad_mask.narrow(1, cache_len, t_cur)?;
        let att_mask = match att_mask {
            Some(am) => Some(am.narrow(1, cache_len, t_cur)?),
            None => None,
        };

        // layers, threading the per-layer KV + conv caches.
        let mut next_channel = Vec::with_capacity(self.layers.len());
        let mut next_time = Vec::with_capacity(self.layers.len());
        for (lth, layer) in self.layers.iter().enumerate() {
            let ch = cache_last_channel.get(lth)?; // (B, cache_len, d_model)
            let ti = cache_last_time.get(lth)?; // (B, d_model, T_cache)
            let (xo, nc, nt) = layer.forward_cache(
                &x,
                att_mask.as_ref(),
                &pos_emb,
                Some(&pad_mask),
                Some(&ch),
                Some(&ti),
            )?;
            x = xo;
            next_channel.push(nc.expect("streaming layer returns a channel cache"));
            next_time.push(nt.expect("streaming layer returns a time cache"));
        }

        if let Some(p) = &self.out_proj {
            x = linear_forward(p, &x)?;
        }
        let encoded = x.transpose(1, 2)?.contiguous()?; // (B, d_out, T')

        let next_channel = Tensor::stack(&next_channel, 0)?;
        let next_time = Tensor::stack(&next_time, 0)?;
        // clamp(cache_last_channel_len + cache_keep_size, max=cache_len) — no min clamp,
        // so it can be negative (matches Python's `torch.clamp(..., max=cache_len)`).
        let next_len: Vec<i64> = cllen
            .iter()
            .map(|&l| (l + cache_keep).min(cache_len as i64))
            .collect();
        Ok((
            encoded,
            Tensor::from_vec(length, (b,), device)?,
            next_channel,
            next_time,
            Tensor::from_vec(next_len, (b,), device)?,
        ))
    }

    /// Debug: conv-stack output `(B, C, T', F')` before subsampling's flatten+linear.
    #[doc(hidden)]
    pub fn subsampling_conv_out(&self, audio_signal: &Tensor) -> Result<Tensor> {
        let x = audio_signal.transpose(1, 2)?.contiguous()?;
        self.pre_encode.forward_conv(&x)
    }

    /// Debug variant returning stage intermediates for parity localization:
    /// (post-subsampling, pos-encoded x, rel pos-emb, after-layer-0, final).
    #[doc(hidden)]
    pub fn forward_stages(
        &self,
        audio_signal: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x = audio_signal.transpose(1, 2)?.contiguous()?;
        let sub = self.pre_encode.forward(&x)?;
        let (mut x, pos_emb) = self.pos_enc.forward(&sub)?;
        let posx = x.clone();
        let mut layer0 = None;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, None, &pos_emb, None)?;
            if i == 0 {
                layer0 = Some(x.clone());
            }
        }
        if let Some(p) = &self.out_proj {
            x = linear_forward(p, &x)?;
        }
        let final_out = x.transpose(1, 2)?.contiguous()?;
        Ok((sub, posx, pos_emb, layer0.unwrap(), final_out))
    }

    // ---- Off the offline forward path; ported 1:1 for inventory (see mod.rs). ----

    /// `forward_internal` — the core encode. The offline form is [`Self::forward`]; the
    /// cache-aware streaming form (`forward_internal` with caches) is
    /// [`Self::forward_streaming`].
    pub fn forward_internal(&self, audio_signal: &Tensor) -> Result<Tensor> {
        self.forward(audio_signal)
    }

    /// PORT: `forward_for_export` (encoder.py L435-464). Python transposes any supplied
    /// caches `(0,1)` (external `(B, n_layers, …)` ⇄ internal `(n_layers, B, …)`), runs
    /// `forward_internal`, then `streaming_post_process(keep_all_outputs=False)`.
    ///
    /// `caches=None` ⇒ the offline export (`(encoded, length, None)`). `caches=Some`
    /// (external layout) ⇒ the streaming export: transpose in, [`Self::forward_streaming`],
    /// post-process, transpose the next caches back out
    /// (`(encoded, length, Some((next_channel, next_time, next_len)))`).
    pub fn forward_for_export(
        &self,
        audio_signal: &Tensor,
        length: Option<&Tensor>,
        caches: Option<(&Tensor, &Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor, Tensor)>)> {
        let Some((cch, ctime, clen)) = caches else {
            let encoded = self.forward_internal(audio_signal)?; // (B, d_out, T')
            let (b, _d, t) = encoded.dims3()?;
            let len = match length {
                Some(l) => l.clone(),
                None => Tensor::from_vec(vec![t as i64; b], (b,), encoded.device())?,
            };
            let (encoded, len, _) = self.streaming_post_process(encoded, len, None, false)?;
            return Ok((encoded, len, None));
        };
        // external (B, n_layers, …) → internal (n_layers, B, …).
        let cch_i = cch.transpose(0, 1)?.contiguous()?;
        let ctime_i = ctime.transpose(0, 1)?.contiguous()?;
        let (encoded, len, next_ch, next_time, next_len) =
            self.forward_streaming(audio_signal, length, &cch_i, &ctime_i, clen)?;
        let (encoded, len, next_ch) =
            self.streaming_post_process(encoded, len, Some(next_ch), false)?;
        let next_ch = next_ch.expect("streaming post-process keeps the channel cache");
        // internal → external.
        Ok((
            encoded,
            len,
            Some((
                next_ch.transpose(0, 1)?.contiguous()?,
                next_time.transpose(0, 1)?.contiguous()?,
                next_len,
            )),
        ))
    }

    /// PORT: `_create_masks` (encoder.py L737-791) → `(pad_mask, att_mask)`, both `bool`
    /// (`u8`) with `1 = IGNORE` — the convention `forward_attention` (mask → `-INF`) and
    /// the conv `pad_mask` consume. `att_context_size = [left, right]`. On the offline
    /// single clip (`[-1,-1]`, full `padding_length`, no `offset`) both come back
    /// all-zero, i.e. the `None` the offline `forward` passes; the limited/chunked-
    /// context logic is the streaming path. Delegates to [`Self::build_masks`] with the
    /// encoder's `self_attention_model` / `att_context_style`.
    pub fn create_masks(
        &self,
        att_context_size: [i64; 2],
        padding_length: &Tensor,
        max_audio_length: usize,
        offset: Option<&Tensor>,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        Self::build_masks(
            &self.self_attention_model,
            &self.att_context_style,
            att_context_size,
            padding_length,
            max_audio_length,
            offset,
            device,
        )
    }

    /// The `_create_masks` body, factored out of `self` so the mask logic is testable
    /// with arbitrary `self_attention_model` / `att_context_style` (the model is fixed
    /// `rel_pos`/`regular`). Boolean grids are tiny and built once, so explicit loops —
    /// not strided tensor ops — keep the `triu`/`tril`/chunk math 1:1 with torch.
    #[allow(clippy::too_many_arguments)]
    pub fn build_masks(
        self_attention_model: &str,
        att_context_style: &str,
        att_context_size: [i64; 2],
        padding_length: &Tensor,
        max_audio_length: usize,
        offset: Option<&Tensor>,
        device: &Device,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let m = max_audio_length;
        let (left, right) = (att_context_size[0], att_context_size[1]);

        // att_mask grid (m,m), 1 = VISIBLE so far. `None` for rel_pos_local_attn.
        let att_visible: Option<Vec<u8>> = if self_attention_model != "rel_pos_local_attn" {
            let mut a = vec![1u8; m * m];
            for r in 0..m {
                for c in 0..m {
                    let d = c as i64 - r as i64; // col - row
                    let mut vis = true;
                    if att_context_style == "regular" {
                        // triu(diagonal=-left): keep d >= -left; tril(diagonal=right): keep d <= right.
                        if left >= 0 && d < -left {
                            vis = false;
                        }
                        if right >= 0 && d > right {
                            vis = false;
                        }
                    } else if att_context_style == "chunked_limited" {
                        if right == -1 {
                            if left >= 0 && d < -left {
                                vis = false;
                            }
                        } else {
                            let chunk_size = right + 1;
                            let left_chunks_num = if left >= 0 { left / chunk_size } else { 10000 };
                            // diff_chunks = chunk_idx[row] - chunk_idx[col]; keep 0 <= diff <= left_chunks_num.
                            let dc = (r as i64) / chunk_size - (c as i64) / chunk_size;
                            if !(dc >= 0 && dc <= left_chunks_num) {
                                vis = false;
                            }
                        }
                    }
                    if !vis {
                        a[r * m + c] = 0;
                    }
                }
            }
            Some(a)
        } else {
            None
        };

        // pad_valid (b,m), 1 = VALID: arange(m) < padding_length [and >= offset].
        let plen = padding_length.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let b = plen.len();
        let off = match offset {
            Some(o) => Some(o.to_dtype(DType::I64)?.to_vec1::<i64>()?),
            None => None,
        };
        let mut pad_valid = vec![0u8; b * m];
        for (bi, &len) in plen.iter().enumerate() {
            for ti in 0..m {
                let mut v = (ti as i64) < len;
                if let Some(o) = &off {
                    v = v && (ti as i64) >= o[bi];
                }
                pad_valid[bi * m + ti] = v as u8;
            }
        }

        // att_mask = ~(pad_for_att & att_visible), 1 = IGNORE. pad_for_att[b,r,c] =
        // pad_valid[b,r] & pad_valid[b,c] (pad_mask AND its transpose).
        let att_mask = match &att_visible {
            Some(av) => {
                let mut ig = vec![0u8; b * m * m];
                for bi in 0..b {
                    for r in 0..m {
                        let pr = pad_valid[bi * m + r];
                        for c in 0..m {
                            let combined = pr & pad_valid[bi * m + c] & av[r * m + c];
                            ig[bi * m * m + r * m + c] = 1 - combined;
                        }
                    }
                }
                Some(Tensor::from_vec(ig, (b, m, m), device)?)
            }
            None => None,
        };

        // pad_mask = ~pad_valid, 1 = IGNORE.
        let pad_ignore: Vec<u8> = pad_valid.iter().map(|&v| 1 - v).collect();
        let pad_mask = Tensor::from_vec(pad_ignore, (b, m), device)?;
        Ok((pad_mask, att_mask))
    }

    /// PORT: `update_max_seq_length` (encoder.py L704-722). Grow `max_audio_length`
    /// to `seq_length` when it exceeds the current max. The Python distributed
    /// `all_reduce(MAX)` across ranks is a single-process no-op here.
    pub fn update_max_seq_length(&mut self, seq_length: usize, _device: &Device) {
        if seq_length > self.max_audio_length {
            self.set_max_audio_length(seq_length);
        }
    }

    /// PORT: `set_max_audio_length` (encoder.py L724-735). Sets the max input length.
    /// Python pre-extends `pos_enc.extend_pe(max)`; the Rust `RelPositionalEncoding`
    /// recomputes its table sized to the input on each `forward`, so there is no
    /// fixed buffer to grow — recording `max_audio_length` is the faithful update.
    pub fn set_max_audio_length(&mut self, max_audio_length: usize) {
        self.max_audio_length = max_audio_length;
    }

    /// PORT: `enable_pad_mask` (encoder.py L793-803). Toggle pad masking and return
    /// the previous state. Offline uses no pad mask (single unpadded clip); the
    /// flag is honoured for the (cold) padded/streaming path.
    pub fn enable_pad_mask(&mut self, on: bool) -> bool {
        let prev = self.use_pad_mask;
        self.use_pad_mask = on;
        prev
    }

    /// PORT: `_calc_context_sizes` (encoder.py L805-851).
    ///
    /// Computes the cache-aware streaming context sizes from the encoder config.
    /// Faithful 1:1 of the Python staticmethod-style helper (it does not touch
    /// `self` on its computation path apart from the error-message interpolation
    /// of `self.conv_context_size`, which is only reachable on an error branch).
    /// Returns the 4-tuple `(att_context_size_all, att_context_size, att_context_probs,
    /// conv_context_size)` where `att_context_size` is `att_context_size_all[0]`.
    ///
    /// Inputs mirror the Python union types:
    /// * `att_context_size` — `None` (empty), a flat `[l, r]`, or a list of `[l, r]`.
    ///   Python accepts both `list[int]` and `list[list[int]]`. We split these into
    ///   two args: `att_context_size_flat: Option<Vec<i64>>` for the bare `list[int]`
    ///   case (wrapped to `[[...]]`, matching the `isinstance(..., int)` branch) and
    ///   `att_context_size: Option<Vec<Vec<i64>>>` for the list-of-lists case.
    /// * `conv_context_size` — `None`, `"causal"`, or a `[l, r]` list (see
    ///   [`ConvContextSize`]).
    ///
    /// Off the offline parity path (offline uses unlimited rel-pos `[-1,-1]`); ported
    /// fully for inventory completeness.
    #[allow(clippy::type_complexity)]
    pub fn calc_context_sizes(
        att_context_size_flat: Option<Vec<i64>>,
        att_context_size: Option<Vec<Vec<i64>>>,
        att_context_probs: Option<Vec<f64>>,
        att_context_style: &str,
        conv_context_size: Option<ConvContextSize>,
        conv_kernel_size: i64,
    ) -> Result<(Vec<Vec<i64>>, Vec<i64>, Vec<f64>, ConvContextSize)> {
        // convert att_context_size to a standard list of lists
        //
        // Python: `if att_context_size:` is truthy for a non-empty list. The bare
        // `list[int]` form (`att_context_size_flat`) is wrapped into `[[...]]`; the
        // list-of-lists form is used as-is. An absent / empty value falls through to
        // the `[[-1, -1]]` default.
        let att_context_size_all: Vec<Vec<i64>> = {
            let provided: Option<Vec<Vec<i64>>> = match (att_context_size_flat, att_context_size) {
                // `isinstance(att_context_size_all[0], int)` branch: wrap the flat list.
                (Some(flat), _) if !flat.is_empty() => Some(vec![flat]),
                (_, Some(nested)) if !nested.is_empty() => Some(nested),
                _ => None,
            };
            match provided {
                Some(all) => {
                    for (i, att_cs) in all.iter().enumerate() {
                        if att_context_style == "chunked_limited" {
                            if att_cs[0] > 0 && att_cs[0] % (att_cs[1] + 1) > 0 {
                                return Err(candle_core::Error::Msg(format!(
                                    "att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!"
                                )));
                            }
                            if att_cs[1] < 0 && all.len() <= 1 {
                                return Err(candle_core::Error::Msg(format!(
                                    "Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                                )));
                            }
                        }
                    }
                    all
                }
                None => vec![vec![-1, -1]],
            }
        };

        let att_context_probs: Vec<f64> = match att_context_probs {
            Some(probs) if !probs.is_empty() => {
                if probs.len() != att_context_size_all.len() {
                    return Err(candle_core::Error::Msg(
                        "The size of the att_context_probs should be the same as att_context_size."
                            .to_string(),
                    ));
                }
                // Python compares `sum(att_context_probs) != 1` exactly (no tolerance).
                if probs.iter().sum::<f64>() != 1.0 {
                    return Err(candle_core::Error::Msg(
                        "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                            .to_string(),
                    ));
                }
                probs
            }
            _ => {
                let n = att_context_size_all.len();
                vec![1.0 / n as f64; n]
            }
        };

        let conv_context_size: ConvContextSize = match conv_context_size {
            Some(ConvContextSize::Causal) => ConvContextSize::Size(conv_kernel_size - 1, 0),
            Some(ConvContextSize::Size(l, r)) => {
                if l + r + 1 != conv_kernel_size {
                    // Python interpolates `self.conv_context_size` here (the as-yet-unset
                    // attribute); we surface the offending list instead, which is the
                    // intended diagnostic.
                    return Err(candle_core::Error::Msg(format!(
                        "Invalid conv_context_size: [{l}, {r}]!"
                    )));
                }
                ConvContextSize::Size(l, r)
            }
            None => {
                let half = (conv_kernel_size - 1) / 2;
                ConvContextSize::Size(half, half)
            }
        };

        let att_context_size = att_context_size_all[0].clone();
        Ok((
            att_context_size_all,
            att_context_size,
            att_context_probs,
            conv_context_size,
        ))
    }

    /// PORT: `set_default_att_context_size` (encoder.py L853-868). Set the current
    /// look-ahead and re-derive the streaming params. Warns (Python `logging.warning`)
    /// if it is not one of the configured `att_context_size_all`.
    pub fn set_default_att_context_size(&mut self, att_context_size: Vec<i64>) {
        if !self.att_context_size_all.contains(&att_context_size) {
            eprintln!(
                "att_context_size={att_context_size:?} is not among the supported look-aheads: {:?}",
                self.att_context_size_all
            );
        }
        self.att_context_size = att_context_size;
        self.setup_streaming_params();
    }

    /// PORT: `change_attention_model` (encoder.py L1017-1144). Switch the attention
    /// model / look-ahead at RUNTIME. This inference port wires only `rel_pos → rel_pos`
    /// here; the Python `abs_pos` / `rel_pos_local_attn` branches rebuild the positional
    /// encoder and per-layer attention from new weights (`load_state_dict`), which has
    /// no candle analog at runtime. (Note: an `abs_pos` encoder CAN be built at
    /// construction — `ConformerLayer` selects `RelPos`/`Abs` from
    /// `cfg.self_attention_model`, base-MHA path verified by `abs_attention_parity` —
    /// but the encoder still holds a fixed `RelPositionalEncoding`, so the abs absolute
    /// pos-enc swap remains config-gated and a live switch is not supported.) So
    /// `rel_pos → rel_pos` is a faithful reconfiguration (the `RelPositionalEncoding`
    /// is already the right type and `set_max_audio_length` resets the table); other
    /// targets error rather than silently no-op.
    pub fn change_attention_model(
        &mut self,
        self_attention_model: Option<&str>,
        att_context_size: Option<Vec<i64>>,
    ) -> Result<()> {
        let att_context_size = att_context_size
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| self.att_context_size.clone());
        let sam = self_attention_model
            .unwrap_or(&self.self_attention_model)
            .to_string();
        if sam == "rel_pos_local_attn" && att_context_size.iter().copied().max().unwrap_or(-1) <= 0
        {
            return Err(candle_core::Error::Msg(
                "When using local attention, context size must be set > 0".into(),
            ));
        }
        if sam != "rel_pos" {
            return Err(candle_core::Error::Msg(format!(
                "change_attention_model: '{sam}' is not wired in this port (only 'rel_pos' is supported); \
                 abs_pos/rel_pos_local_attn require rebuilding the pos-enc + attention from new weights"
            )));
        }
        self.self_attention_model = sam;
        self.att_context_size = att_context_size;
        self.set_max_audio_length(self.pos_emb_max_len);
        Ok(())
    }

    /// PORT: `setup_streaming_params` (encoder.py L870-975). Re-derive the cache-aware
    /// [`CacheAwareStreamingConfig`] from the current `att_context_size`. Uses the
    /// default args (no `chunk_size`/`shift_size`/`left_chunks` overrides), as the
    /// Python `__init__` call does.
    pub fn setup_streaming_params(&mut self) {
        self.streaming_cfg = Self::compute_streaming_cfg(
            &self.att_context_size,
            &self.att_context_style,
            self.n_layers,
            self.conv_context_size,
            self.subsampling_factor,
            IntOrPair::Pair(1, self.subsampling_factor as i64),
            IntOrPair::Pair(0, self.subsampling_factor as i64 + 1),
            None,
            None,
            None,
            10_000,
        );
        // Python L968-974: propagate cache_drop_size onto each layer's MHA + conv.
        let cds = self.streaming_cfg.cache_drop_size.max(0) as usize;
        for layer in &mut self.layers {
            layer.set_cache_drop_size(cds);
        }
    }

    /// Set the inference attention context `[left, right]` and refresh the streaming
    /// config (so a caller can stream with a bounded left cache). `right < 0` keeps the
    /// unlimited-right (offline) behaviour.
    pub fn set_streaming_att_context(&mut self, att_context_size: [i64; 2]) {
        self.att_context_size = att_context_size.to_vec();
        self.setup_streaming_params();
    }

    /// The body of [`Self::setup_streaming_params`] (encoder.py L891-966), pulled out
    /// so `new` can seed `streaming_cfg` before `self` exists. `sampling_frames` /
    /// `pre_encode_cache` are `pre_encode.get_sampling_frames()` /
    /// `get_streaming_cache_size()` (the `ConvSubsampling` returns the `[..]` pairs).
    /// The Python per-layer `cache_drop_size` propagation onto MHA/CausalConv1D
    /// (L968-974) drives only the streaming KV/conv caches, which are off this
    /// inference port's path; the config itself is computed 1:1.
    #[allow(clippy::too_many_arguments)]
    fn compute_streaming_cfg(
        att_context_size: &[i64],
        att_context_style: &str,
        n_layers: usize,
        conv_context_size: (i64, i64),
        subsampling_factor: usize,
        sampling_frames: IntOrPair,
        pre_encode_cache: IntOrPair,
        chunk_size: Option<i64>,
        shift_size: Option<i64>,
        left_chunks: Option<i64>,
        max_context: i64,
    ) -> CacheAwareStreamingConfig {
        let mut cfg = CacheAwareStreamingConfig::default();
        let sf = subsampling_factor as i64;

        // lookahead_steps + cache_drop_size (L897-910).
        let lookahead_steps: Option<i64> = if let Some(cs) = chunk_size {
            cfg.cache_drop_size = cs - shift_size.unwrap_or(0);
            Some(cs - 1)
        } else if att_context_style == "chunked_limited" {
            cfg.cache_drop_size = 0;
            Some(att_context_size[1])
        } else if att_context_style == "regular" {
            let la = att_context_size[1] * n_layers as i64 + conv_context_size.1 * n_layers as i64;
            cfg.cache_drop_size = la;
            Some(la)
        } else {
            cfg.cache_drop_size = 0;
            None
        };

        // last_channel_cache_size (L912-923).
        cfg.last_channel_cache_size = if chunk_size.is_none() {
            if att_context_size[0] >= 0 {
                att_context_size[0]
            } else {
                max_context
            }
        } else if let Some(lc) = left_chunks {
            lc * chunk_size.unwrap()
        } else if att_context_size[0] >= 0 {
            att_context_size[0]
        } else {
            max_context
        };

        // chunk_size / shift_size / valid_out_len (L925-951). `la` is set for the
        // regular/chunked styles (the only ones reached here); guard the unknown style.
        let la = lookahead_steps.unwrap_or(0);
        match sampling_frames {
            IntOrPair::Pair(s0, s1) => {
                cfg.chunk_size = IntOrPair::Pair(s0 + sf * la, s1 + sf * la);
                let shift0 = s0 + s1 * (la - cfg.cache_drop_size);
                let shift1 = s1 + s1 * (la - cfg.cache_drop_size);
                cfg.shift_size = IntOrPair::Pair(shift0, shift1);
                cfg.valid_out_len = (shift1 - s1) / sf + 1;
            }
            IntOrPair::Int(s) => {
                cfg.chunk_size = IntOrPair::Int(s * (1 + la));
                let shift = s * (1 + la - cfg.cache_drop_size);
                cfg.shift_size = IntOrPair::Int(shift);
                cfg.valid_out_len = shift / sf;
            }
        }

        // pre_encode_cache_size / drop_extra_pre_encoded (L953-966).
        cfg.pre_encode_cache_size = pre_encode_cache.clone();
        cfg.drop_extra_pre_encoded = match pre_encode_cache {
            IntOrPair::Pair(_, p1) => {
                if p1 >= 1 {
                    1 + (p1 - 1) / sf
                } else {
                    0
                }
            }
            IntOrPair::Int(p) => p / sf,
        };
        cfg
    }

    /// PORT: `streaming_post_process` (encoder.py L466-489). Trim the streaming
    /// output / `last_channel` cache to the valid window. The 2-element offline form
    /// (no `cache_last_channel_next`) is returned unchanged (Python `if len(rets)==2`).
    pub fn streaming_post_process(
        &self,
        encoded: Tensor,
        encoded_len: Tensor,
        cache_last_channel_next: Option<Tensor>,
        keep_all_outputs: bool,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        // 2-element (no-cache / offline) rets: returned unchanged.
        let Some(cache) = cache_last_channel_next else {
            return Ok((encoded, encoded_len, None));
        };

        // last_channel cache: keep the last `last_channel_cache_size` steps (dim 2 of
        // (layers, B, T_cache, D)).
        let cache_out = if self.streaming_cfg.last_channel_cache_size > 0 {
            let n = self.streaming_cfg.last_channel_cache_size as usize;
            let t = cache.dim(2)?;
            if t > n {
                cache.narrow(2, t - n, n)?
            } else {
                cache
            }
        } else {
            cache
        };

        // encoded: cap time to valid_out_len (dim 2 of (B, D, T)); clamp the length.
        let (encoded, encoded_len) = if self.streaming_cfg.valid_out_len > 0
            && (!keep_all_outputs || self.att_context_style == "regular")
        {
            let v = self.streaming_cfg.valid_out_len as usize;
            let t = encoded.dim(2)?;
            let enc = if t > v {
                encoded.narrow(2, 0, v)?
            } else {
                encoded
            };
            let len = encoded_len.clamp(0i64, self.streaming_cfg.valid_out_len)?;
            (enc, len)
        } else {
            (encoded, encoded_len)
        };
        Ok((encoded, encoded_len, Some(cache_out)))
    }

    /// PORT: `get_initial_cache_state` (encoder.py L977-1015). Allocate the
    /// `last_channel` / `last_time` caches + zeroed lengths. `max_dim == 0` (the
    /// real init) → zeros; `max_dim > 0` (export tracing) → random fills. The Python
    /// per-batch random-length tail-zeroing (`max_dim > 0`) is a tracing nicety and
    /// the lengths stay zero here.
    pub fn get_initial_cache_state(
        &self,
        batch_size: usize,
        dtype: DType,
        device: &Device,
        max_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let last_time_cache_size = self.conv_context_size.0.max(0) as usize;
        let lc = self.streaming_cfg.last_channel_cache_size.max(0) as usize;
        let shape_ch = (self.layers.len(), batch_size, lc, self.d_model);
        let shape_t = (
            self.layers.len(),
            batch_size,
            self.d_model,
            last_time_cache_size,
        );
        let (cache_last_channel, cache_last_time) = if max_dim > 0 {
            (
                Tensor::randn(0f32, 1f32, shape_ch, device)?.to_dtype(dtype)?,
                Tensor::randn(0f32, 1f32, shape_t, device)?.to_dtype(dtype)?,
            )
        } else {
            (
                Tensor::zeros(shape_ch, dtype, device)?,
                Tensor::zeros(shape_t, dtype, device)?,
            )
        };
        let cache_last_channel_len = Tensor::zeros((batch_size,), DType::I64, device)?;
        Ok((cache_last_channel, cache_last_time, cache_last_channel_len))
    }

    /// PORT: `input_example` (encoder.py L176-216). Build dummy tracing inputs: a
    /// random `(max_batch, feat_in, T)` signal + lengths, plus the initial caches
    /// when `export_cache_support`. Returns the tuple as a `Vec<Tensor>` (Python
    /// returns a 2- or 5-element tuple). Lengths use a deterministic `max_dim` stand-in
    /// for the Python `randint` (a tracing dummy).
    pub fn input_example(
        &self,
        max_batch: usize,
        max_dim: usize,
        device: &Device,
    ) -> Result<Vec<Tensor>> {
        if self.export_cache_support {
            let window_size = (self.streaming_cfg.chunk_size.second()
                + self.streaming_cfg.pre_encode_cache_size.second())
            .max(1) as usize;
            let input = Tensor::randn(0f32, 1f32, (max_batch, self.feat_in, window_size), device)?;
            let length =
                Tensor::from_vec(vec![window_size as i64; max_batch], (max_batch,), device)?;
            let (cc, ct, cl) =
                self.get_initial_cache_state(max_batch, DType::F32, device, max_dim)?;
            Ok(vec![
                input,
                length,
                cc.transpose(0, 1)?,
                ct.transpose(0, 1)?,
                cl,
            ])
        } else {
            let input = Tensor::randn(0f32, 1f32, (max_batch, self.feat_in, max_dim), device)?;
            let mut lens = vec![(max_dim / 2) as i64; max_batch];
            if let Some(first) = lens.first_mut() {
                *first = max_dim as i64; // Python pins lengths[0]=max_dim in the cache branch
            }
            let length = Tensor::from_vec(lens, (max_batch,), device)?;
            Ok(vec![input, length])
        }
    }

    /// PORT: `disabled_deployment_input_names` (encoder.py L218-223). The cache
    /// inputs are disabled unless `export_cache_support`.
    pub fn disabled_deployment_input_names(&self) -> Vec<&'static str> {
        if !self.export_cache_support {
            vec![
                "cache_last_channel",
                "cache_last_time",
                "cache_last_channel_len",
            ]
        } else {
            vec![]
        }
    }

    /// PORT: `disabled_deployment_output_names` (encoder.py L225-230). The `*_next`
    /// cache outputs are disabled unless `export_cache_support`.
    pub fn disabled_deployment_output_names(&self) -> Vec<&'static str> {
        if !self.export_cache_support {
            vec![
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ]
        } else {
            vec![]
        }
    }

    /// Set `export_cache_support` (Python attribute toggled before export tracing).
    pub fn set_export_cache_support(&mut self, on: bool) {
        self.export_cache_support = on;
    }

    /// `change_subsampling_conv_chunking_factor` — forwards to the pre-encode
    /// subsampling (a memory-tiling control; see `ConvSubsampling`).
    pub fn change_subsampling_conv_chunking_factor(&mut self, factor: i64) -> Result<()> {
        self.pre_encode
            .change_subsampling_conv_chunking_factor(factor)
    }

    // ---- Read accessors for the Python instance attributes (state queryable as in
    // Python, e.g. `self.streaming_cfg`, `self.att_context_probs`). ----

    /// `att_context_size_all` — every configured look-ahead.
    pub fn att_context_size_all(&self) -> &[Vec<i64>] {
        &self.att_context_size_all
    }
    /// `att_context_size` — the current `[left, right]` look-ahead.
    pub fn att_context_size(&self) -> &[i64] {
        &self.att_context_size
    }
    /// `att_context_probs` — sampling distribution over `att_context_size_all`
    /// (the training-time random att-context choice in `forward_internal`).
    pub fn att_context_probs(&self) -> &[f64] {
        &self.att_context_probs
    }
    /// `conv_context_size` — resolved `[left, right]` depthwise-conv context.
    pub fn conv_context_size(&self) -> (i64, i64) {
        self.conv_context_size
    }
    /// `streaming_cfg` — the current cache-aware streaming parameters.
    pub fn streaming_cfg(&self) -> &CacheAwareStreamingConfig {
        &self.streaming_cfg
    }
    /// `max_audio_length` — current positional-table max length.
    pub fn max_audio_length(&self) -> usize {
        self.max_audio_length
    }
    /// `use_pad_mask` — current pad-mask toggle.
    pub fn use_pad_mask(&self) -> bool {
        self.use_pad_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_cfg_regular_offline() {
        // Faithful re-derivation of `setup_streaming_params` for the offline default
        // (`att_context_size=[-1,-1]`, "regular"). Hand-computed from encoder.py:
        //   n_layers=4, conv_kernel=9 ⇒ conv_ctx=(4,4), subsampling_factor=8,
        //   sampling_frames=[1,8], pre_encode_cache=[0,9].
        //   lookahead = att[1]*L + conv_ctx[1]*L = -1*4 + 4*4 = 12 = cache_drop_size
        //   last_channel_cache_size = max_context (att[0]=-1<0) = 10000
        //   chunk = [1+8*12, 8+8*12] = [97,104]; shift = [1,8] (la-drop=0)
        //   valid_out_len = (8-8)/8 + 1 = 1; drop_extra = 1 + (9-1)/8 = 2
        let cfg = ConformerEncoder::compute_streaming_cfg(
            &[-1, -1],
            "regular",
            4,
            (4, 4),
            8,
            IntOrPair::Pair(1, 8),
            IntOrPair::Pair(0, 9),
            None,
            None,
            None,
            10_000,
        );
        assert_eq!(cfg.cache_drop_size, 12);
        assert_eq!(cfg.last_channel_cache_size, 10_000);
        assert_eq!(cfg.chunk_size, IntOrPair::Pair(97, 104));
        assert_eq!(cfg.shift_size, IntOrPair::Pair(1, 8));
        assert_eq!(cfg.valid_out_len, 1);
        assert_eq!(cfg.pre_encode_cache_size, IntOrPair::Pair(0, 9));
        assert_eq!(cfg.drop_extra_pre_encoded, 2);
    }

    #[test]
    fn streaming_cfg_chunked_with_overrides() {
        // chunk_size/shift_size override branch (encoder.py L897-901): chunk=8, shift=4
        // ⇒ cache_drop=4, lookahead=7; scalar sampling_frames=2.
        let cfg = ConformerEncoder::compute_streaming_cfg(
            &[16, 7],
            "chunked_limited",
            4,
            (4, 4),
            8,
            IntOrPair::Int(2),
            IntOrPair::Int(5),
            Some(8), // chunk_size
            Some(4), // shift_size
            Some(3), // left_chunks
            10_000,
        );
        assert_eq!(cfg.cache_drop_size, 8 - 4); // chunk - shift
                                                // last_channel_cache_size = left_chunks * chunk_size = 3*8 = 24
        assert_eq!(cfg.last_channel_cache_size, 24);
        // scalar: chunk = s*(1+la) = 2*(1+7) = 16; shift = 2*(1+7-4) = 8
        assert_eq!(cfg.chunk_size, IntOrPair::Int(16));
        assert_eq!(cfg.shift_size, IntOrPair::Int(8));
        assert_eq!(cfg.valid_out_len, 8 / 8); // shift // sf = 1
        assert_eq!(cfg.drop_extra_pre_encoded, 5 / 8); // p // sf = 0
    }
}
