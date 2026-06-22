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
                        "The size of the att_context_probs should be the same as att_context_size.".to_string(),
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
        Ok((att_context_size_all, att_context_size, att_context_probs, conv_context_size))
    }

    /// PORT: `set_default_att_context_size` / `change_attention_model` —
    /// limited-context & att-model switching for streaming. Offline uses unlimited
    /// rel-pos attention (`[-1,-1]`); no-op stubs, preserved for 1:1 inventory.
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
