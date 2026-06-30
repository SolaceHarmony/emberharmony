//! Port of `liquid_audio/model/conformer/utils.py` (NeMo cast/streaming utils).
//!
//! Off the offline-inference forward path — autocast guarding, cache-aware
//! streaming config, and stochastic-depth are training/streaming features — but
//! ported 1:1 for inventory completeness.

use candle_core::DType;

/// A NeMo streaming size that is either a single value or a `[first_step, others]`
/// pair. Python stores `chunk_size` / `shift_size` / `pre_encode_cache_size` as a
/// bare `int` OR a 2-element list depending on whether the pre-encoder reports a
/// list of `get_sampling_frames()` — the [`crate::model::conformer::subsampling::ConvSubsampling`]
/// does (`[1, subsampling_factor]`), so for this model they are always pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntOrPair {
    Int(i64),
    Pair(i64, i64),
}

impl Default for IntOrPair {
    fn default() -> Self {
        IntOrPair::Int(0)
    }
}

impl IntOrPair {
    /// The `[1]` (others) element a Python `cfg.x[1]` access reads; for a scalar it
    /// is the value itself (Python would index an int — but the list branch is the
    /// one this model takes).
    pub fn second(&self) -> i64 {
        match self {
            IntOrPair::Int(v) => *v,
            IntOrPair::Pair(_, b) => *b,
        }
    }
}

/// `CacheAwareStreamingConfig` (dataclass) — cache-aware streaming parameters.
/// Field docs mirror the Python comments.
#[derive(Debug, Clone, Default)]
pub struct CacheAwareStreamingConfig {
    /// size of each chunk at each step (scalar or `[first, others]`)
    pub chunk_size: IntOrPair,
    /// size of the shift in each step (scalar or `[first, others]`)
    pub shift_size: IntOrPair,
    /// number of steps to drop from the cache
    pub cache_drop_size: i64,
    /// size of the needed cache for last-channel layers
    pub last_channel_cache_size: i64,
    /// number of final-output steps that are valid (== offline mode)
    pub valid_out_len: i64,
    /// cache size for the pre-encoding part (avoids caching inside pre-encode)
    pub pre_encode_cache_size: IntOrPair,
    /// steps dropped after the pre-encoding layer
    pub drop_extra_pre_encoded: i64,
    /// number of last-channel layers (e.g. MHA) needing caching
    pub last_channel_num: i64,
    /// number of last-time layers (e.g. convolutions) needing caching
    pub last_time_num: i64,
}

/// PORT: `avoid_float16_autocast_context` (utils.py L25-40).
///
/// NeMo's "if the active autocast dtype is f16, compute in bf16 (if supported)
/// else f32; otherwise leave it" guard. Python returns a *context manager* keyed
/// off the global autocast state; candle has no implicit autocast, so the faithful
/// port is the same **decision as a pure function** over the would-be autocast
/// dtype, returning the dtype the enclosed block should run in (`None` =
/// `nullcontext()`, no override).
///
/// The `torch.jit.is_scripting()/is_tracing()` branch (which forces f32) has no
/// candle analog and is taken as false. On the offline path the conformer attention
/// already upcasts to f32 explicitly (see `mha.rs`), so this is the realized logic.
pub fn avoid_float16_autocast_context(
    autocast_dtype: Option<DType>,
    bf16_supported: bool,
) -> Option<DType> {
    match autocast_dtype {
        // f16 autocast active → avoid it: bf16 if the device supports it, else f32.
        Some(DType::F16) => Some(if bf16_supported {
            DType::BF16
        } else {
            DType::F32
        }),
        // not in f16 autocast → `nullcontext()`: no dtype override.
        _ => None,
    }
}

/// `compute_stochastic_depth_drop_probs` — per-layer drop probabilities for
/// stochastic-depth regularization (training). Faithful port: layer 0 never
/// drops, `start_layer` ≥ 1, linear ramp or uniform.
pub fn compute_stochastic_depth_drop_probs(
    num_layers: usize,
    stochastic_depth_drop_prob: f64,
    stochastic_depth_mode: &str,
    stochastic_depth_start_layer: usize,
) -> Vec<f64> {
    assert!(
        (0.0..1.0).contains(&stochastic_depth_drop_prob),
        "stochastic_depth_drop_prob has to be in [0, 1)."
    );
    assert!(
        (1..=num_layers).contains(&stochastic_depth_start_layer),
        "stochastic_depth_start_layer has to be in [1, num layers]."
    );

    // Layers before `start_layer` are never dropped.
    let mut layer_drop_probs = vec![0.0_f64; stochastic_depth_start_layer];

    // Layers from `start_layer` on may be dropped.
    let big_l = num_layers as i64 - stochastic_depth_start_layer as i64;
    if big_l > 0 {
        let big_l = big_l as usize;
        match stochastic_depth_mode {
            "linear" => {
                // start at 1/L * drop_prob, end at the desired drop probability.
                layer_drop_probs.extend((1..=big_l).map(|l| l as f64 / big_l as f64 * stochastic_depth_drop_prob));
            }
            "uniform" => layer_drop_probs.extend(std::iter::repeat_n(stochastic_depth_drop_prob, big_l)),
            other => panic!(
                "stochastic_depth_mode has to be one of [\"linear\", \"uniform\"]. Current value: {other}"
            ),
        }
    }
    layer_drop_probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avoid_float16_decision() {
        // f16 autocast active → bf16 when supported, else f32.
        assert_eq!(
            avoid_float16_autocast_context(Some(DType::F16), true),
            Some(DType::BF16)
        );
        assert_eq!(
            avoid_float16_autocast_context(Some(DType::F16), false),
            Some(DType::F32)
        );
        // not in f16 autocast → nullcontext (no override).
        assert_eq!(
            avoid_float16_autocast_context(Some(DType::BF16), true),
            None
        );
        assert_eq!(avoid_float16_autocast_context(Some(DType::F32), true), None);
        assert_eq!(avoid_float16_autocast_context(None, true), None);
    }

    #[test]
    fn int_or_pair_second() {
        assert_eq!(IntOrPair::Int(7).second(), 7);
        assert_eq!(IntOrPair::Pair(3, 9).second(), 9);
    }
}
