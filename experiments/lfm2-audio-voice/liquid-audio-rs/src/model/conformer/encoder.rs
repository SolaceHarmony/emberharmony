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
}
