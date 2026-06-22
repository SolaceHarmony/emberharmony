//! Port of `liquid_audio/model/conformer/encoder.py` (NeMo ConformerEncoder).
//!
//! Inference path: `dw_striding` ConvSubsampling → RelPositionalEncoding →
//! N × ConformerLayer → optional out projection. For a single offline clip with
//! unlimited attention context (`att_context_size = [-1,-1]`) and no padding, the
//! attention/pad masks are identity, so they are passed as `None`. Streaming,
//! cache, stochastic depth, reduction, and export paths are not ported.

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::modules::ConformerLayer;
use super::mha::RelPositionalEncoding;
use super::subsampling::ConvSubsampling;

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
}

pub struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    pos_enc: RelPositionalEncoding,
    layers: Vec<ConformerLayer>,
    out_proj: Option<Linear>,
}

impl ConformerEncoder {
    pub fn new(cfg: &ConformerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d_ff = cfg.d_model * cfg.ff_expansion_factor;
        let conv_channels = if cfg.subsampling_conv_channels == 0 { cfg.d_model } else { cfg.subsampling_conv_channels };

        let pre_encode = ConvSubsampling::new(cfg.subsampling_factor, cfg.feat_in, cfg.d_model, conv_channels, vb.pp("pre_encode"))?;

        let xscale = if cfg.xscaling { Some((cfg.d_model as f64).sqrt()) } else { None };
        let pos_enc = RelPositionalEncoding::new(cfg.d_model, xscale);

        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ConformerLayer::new(cfg.d_model, d_ff, cfg.n_heads, cfg.conv_kernel_size, true, layers_vb.pp(i.to_string()))?);
        }

        let out_proj = if cfg.feat_out > 0 && cfg.feat_out != cfg.d_model {
            Some(linear(cfg.d_model, cfg.feat_out, vb.pp("out_proj"))?)
        } else {
            None
        };

        Ok(Self { pre_encode, pos_enc, layers, out_proj })
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
            x = p.forward(&x)?;
        }
        x.transpose(1, 2)?.contiguous() // (B, d_out, T')
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
    pub fn forward_stages(&self, audio_signal: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
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
            x = p.forward(&x)?;
        }
        let final_out = x.transpose(1, 2)?.contiguous()?;
        Ok((sub, posx, pos_emb, layer0.unwrap(), final_out))
    }

    // ---- Off the offline forward path; ported 1:1 for inventory (see mod.rs). ----

    /// `forward_internal` — the core encode. Our [`Self::forward`] *is*
    /// `forward_internal` (the public `forward` in Python just length-handles then
    /// calls it); kept as an alias for the 1:1 mapping.
    pub fn forward_internal(&self, audio_signal: &Tensor) -> Result<Tensor> {
        self.forward(audio_signal)
    }

    /// PORT: `forward_for_export` — ONNX/TorchScript export wrapper around the
    /// forward. No export path here; faithfully delegates to `forward`.
    pub fn forward_for_export(&self, audio_signal: &Tensor) -> Result<Tensor> {
        self.forward(audio_signal)
    }

    /// `_create_masks(att_context_size, padding_length, max_audio_length, ...)`
    /// → `(att_mask, pad_mask)`. On the offline single-clip path (unlimited
    /// context `[-1,-1]`, no padding) both masks are identity, hence `None`
    /// (matching what `forward` passes). The padded-batch case is handled by
    /// per-segment encode (see the `forward` contract / `prefill_parity`).
    pub fn create_masks(&self) -> (Option<Tensor>, Option<Tensor>) {
        (None, None)
    }

    /// PORT: `update_max_seq_length` / `set_max_audio_length` — grow the cached
    /// positional-encoding table to a max length. The port computes the rel-pos
    /// table on the fly sized to the input (`RelPositionalEncoding::forward`), so
    /// there is no fixed buffer to extend; no-op, preserved for 1:1 inventory.
    pub fn update_max_seq_length(&self, _seq_length: usize, _device: &candle_core::Device) {}

    /// See [`Self::update_max_seq_length`].
    pub fn set_max_audio_length(&self, _max_audio_length: usize) {}

    /// PORT: `enable_pad_mask` — toggle pad masking. The offline path uses no pad
    /// mask (single unpadded clip); returns the previous state (always `false`).
    pub fn enable_pad_mask(&self, _on: bool) -> bool {
        false
    }

    /// PORT: `_calc_context_sizes` / `set_default_att_context_size` /
    /// `change_attention_model` — limited-context & att-model switching for
    /// streaming. Offline uses unlimited rel-pos attention (`[-1,-1]`); no-op
    /// stubs, preserved for 1:1 inventory.
    pub fn set_default_att_context_size(&self, _att_context_size: (i64, i64)) {}

    /// See [`Self::set_default_att_context_size`].
    pub fn change_attention_model(&self, _self_attention_model: &str) {}

    /// PORT: `setup_streaming_params` / `get_initial_cache_state` /
    /// `streaming_post_process` — cache-aware streaming setup & cache tensors.
    /// Not on the offline path; no-op stubs, preserved for 1:1 inventory.
    pub fn setup_streaming_params(&self) {}

    /// See [`Self::setup_streaming_params`]. Returns no cache (offline).
    pub fn get_initial_cache_state(&self) -> Option<Tensor> {
        None
    }

    /// See [`Self::setup_streaming_params`].
    pub fn streaming_post_process(&self, rets: Tensor) -> Tensor {
        rets
    }

    /// PORT: `input_example` — ONNX-export dummy input (random tensor for tracing).
    /// `disabled_deployment_{input,output}_names` — export hooks. No export path
    /// here; preserved for 1:1 inventory.
    pub fn input_example(&self, _max_batch: usize, _max_dim: usize) {}

    /// See [`Self::input_example`].
    pub fn disabled_deployment_input_names(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// See [`Self::input_example`].
    pub fn disabled_deployment_output_names(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// `change_subsampling_conv_chunking_factor` — forwards to the pre-encode
    /// subsampling (a memory-tiling control; see `ConvSubsampling`).
    pub fn change_subsampling_conv_chunking_factor(&mut self, factor: i64) -> Result<()> {
        self.pre_encode.change_subsampling_conv_chunking_factor(factor)
    }
}
