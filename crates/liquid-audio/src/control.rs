/// Metadata displayed by the desktop for a host-owned voice session.
///
/// The native host publishes these counters through its bounded control
/// mailbox. This Rust record owns no PCM, model state, worker, or callback.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize,
    serde::Deserialize,
)]
pub struct AudioStatsSnapshot {
    pub decoded_samples: u64,
    pub queued_samples: u64,
    pub dropped_samples: u64,
    pub played_samples: u64,
    pub underrun_frames: u64,
    pub turn_count: u64,
    pub last_turn_latency_ms: u64,
    pub mean_turn_latency_ms: u64,
}
