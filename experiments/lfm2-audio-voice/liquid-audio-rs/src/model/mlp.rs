//! Port of `liquid_audio/model/mlp.py`.

use candle_core::{Result, Tensor};
use crate::model::norm::layer_norm;
use candle_nn::{linear, linear_no_bias, seq, Activation, Module, Sequential, VarBuilder};

/// Faithful port of `MLP(nn.Module)`.
///
/// Builds the same `nn.Sequential`:
///   `[LayerNorm?] , (Linear, [GELU, Dropout?])* , Linear`
/// over `channels = [in_channels, *hidden_dim, out_channels]`.
///
/// Weight paths mirror the Python `nn.Sequential` child indices ("model.{i}"),
/// including the no-weight GELU/Dropout slots, so a trained checkpoint loads
/// 1:1. Dropout is identity at inference, so we only advance the index for its
/// slot (when `dropout > 0`) rather than instantiating it.
///
/// Note (fidelity): PyTorch `nn.GELU()` defaults to the exact erf form. candle's
/// `Activation::Gelu` maps to `gelu_erf` (exact), which is what we want — not the
/// tanh approximation (`NewGelu`/`GeluPytorchTanh`).
pub struct MLP {
    model: Sequential,
}

impl MLP {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_dim: &[usize],
        bias: bool,
        use_layer_norm: bool,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut channels = Vec::with_capacity(hidden_dim.len() + 2);
        channels.push(in_channels);
        channels.extend_from_slice(hidden_dim);
        channels.push(out_channels);

        let mut model = seq();
        let mut idx = 0usize; // mirrors nn.Sequential child index for weight names

        if use_layer_norm {
            model = model.add(layer_norm(channels[0], 1e-5, vb.pp(format!("model.{idx}")))?);
            idx += 1;
        }

        for i in 0..(channels.len() - 1) {
            let lin = if bias {
                linear(channels[i], channels[i + 1], vb.pp(format!("model.{idx}")))?
            } else {
                linear_no_bias(channels[i], channels[i + 1], vb.pp(format!("model.{idx}")))?
            };
            model = model.add(lin);
            idx += 1;

            if i != channels.len() - 2 {
                model = model.add(Activation::Gelu);
                idx += 1;
                if dropout > 0.0 {
                    // Dropout is identity at inference; only reserve its index slot.
                    idx += 1;
                }
            }
        }

        Ok(Self { model })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.model.forward(x)
    }
}
