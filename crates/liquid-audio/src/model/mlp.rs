//! Port of `liquid_audio/model/mlp.py`.

use crate::model::norm::layer_norm;
use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, Activation, Module, VarBuilder};

use crate::model::linear::Bf16Linear;

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
///
/// Layers are held as `Vec<Box<dyn Module + Send + Sync>>` rather than `candle_nn::Sequential`
/// (whose `Vec<Box<dyn Module>>` is **not** `Send`) so the model — and the processor that
/// owns it — can move onto a dedicated inference worker thread (the realtime full-duplex
/// pipeline). `Linear`, our `LayerNorm`, and `Activation` are all `Send`. Forward semantics
/// are identical: a left fold applying each layer in order.
pub struct MLP {
    model: Vec<Box<dyn Module + Send + Sync>>,
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

        let mut model: Vec<Box<dyn Module + Send + Sync>> = Vec::new();
        let mut idx = 0usize; // mirrors nn.Sequential child index for weight names

        if use_layer_norm {
            model.push(Box::new(layer_norm(
                channels[0],
                1e-5,
                vb.pp(format!("model.{idx}")),
            )?));
            idx += 1;
        }

        for i in 0..(channels.len() - 1) {
            let lin = if bias {
                linear(channels[i], channels[i + 1], vb.pp(format!("model.{idx}")))?
            } else {
                linear_no_bias(channels[i], channels[i + 1], vb.pp(format!("model.{idx}")))?
            };
            model.push(Box::new(Bf16Linear::new(lin)));
            idx += 1;

            if i != channels.len() - 2 {
                model.push(Box::new(Activation::Gelu));
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
        // Left fold, same as `nn.Sequential.forward`. The first `forward` reads `x` by
        // reference; each step rebinds `h` to the next op's output (candle tensors are
        // Arc-backed handles — rebinding is a refcount bump, not a data copy).
        let mut h = self.model[0].forward(x)?;
        for layer in &self.model[1..] {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn build(in_c: usize, out_c: usize, hidden: &[usize], bias: bool, ln: bool) -> MLP {
        // VarMap-backed builder inits the (missing) weights, so we can exercise the wiring
        // without a checkpoint. Names still follow "model.{idx}", proving the index walk.
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        MLP::new(in_c, out_c, hidden, bias, ln, 0.0, vb).unwrap()
    }

    #[test]
    fn forward_maps_in_channels_to_out_channels() {
        let dev = Device::Cpu;
        // bias on/off and layer-norm on/off — all four wirings must preserve shape.
        for &(bias, ln) in &[(true, false), (false, false), (true, true), (false, true)] {
            let mlp = build(8, 3, &[16, 5], bias, ln);
            let x = Tensor::randn(0f32, 1f32, (2, 8), &dev).unwrap();
            let y = mlp.forward(&x).unwrap();
            assert_eq!(y.dims(), &[2, 3], "bias={bias} ln={ln}");
            let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                v.iter().all(|f| f.is_finite()),
                "non-finite output (bias={bias} ln={ln})"
            );
        }
    }

    #[test]
    fn single_linear_no_hidden() {
        // channels = [in, out] → exactly one Linear, no GELU. Verifies the loop's
        // "GELU only between layers" edge (no trailing activation).
        let mlp = build(4, 2, &[], true, false);
        assert_eq!(mlp.model.len(), 1, "one Linear, no activation");
        let x = Tensor::zeros((3, 4), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(mlp.forward(&x).unwrap().dims(), &[3, 2]);
    }

    #[test]
    fn mlp_is_send() {
        // The whole point of the Vec<Box<dyn Module + Send + Sync>> rewrite: MLP must be Send so
        // LFM2AudioModel is Send and can be owned by the inference worker thread.
        fn is_send<T: Send>() {}
        is_send::<MLP>();
    }
}
