//! Port of `liquid_audio/model/conformer/utils.py` (NeMo cast/streaming utils).
//!
//! Off the offline-inference forward path — autocast guarding, cache-aware
//! streaming config, and stochastic-depth are training/streaming features — but
//! ported 1:1 for inventory completeness.

/// `CacheAwareStreamingConfig` (dataclass) — cache-aware streaming parameters.
/// Field docs mirror the Python comments.
#[derive(Debug, Clone, Default)]
pub struct CacheAwareStreamingConfig {
    /// size of each chunk at each step
    pub chunk_size: i64,
    /// size of the shift in each step
    pub shift_size: i64,
    /// number of steps to drop from the cache
    pub cache_drop_size: i64,
    /// size of the needed cache for last-channel layers
    pub last_channel_cache_size: i64,
    /// number of final-output steps that are valid (== offline mode)
    pub valid_out_len: i64,
    /// cache size for the pre-encoding part (avoids caching inside pre-encode)
    pub pre_encode_cache_size: i64,
    /// steps dropped after the pre-encoding layer
    pub drop_extra_pre_encoded: i64,
    /// number of last-channel layers (e.g. MHA) needing caching
    pub last_channel_num: i64,
    /// number of last-time layers (e.g. convolutions) needing caching
    pub last_time_num: i64,
}

/// PORT: `avoid_float16_autocast_context` — torch AMP autocast dtype guard
/// (fp16 → bf16/fp32). candle has no autocast; the compute dtype is explicit
/// (`from_pretrained(dtype)`), so there is no fp16 autocast context to avoid.
/// No-op, preserved for 1:1 inventory.
pub fn avoid_float16_autocast_context() {}

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
