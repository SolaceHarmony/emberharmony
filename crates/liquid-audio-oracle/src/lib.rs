//! Workspace-only reference implementation for native LFM2 development.
//!
//! Candle, Moshi, training, and fixture-capture code live here rather than in
//! the production `liquid-audio` crate. Dependency flow is one-way: this crate
//! consumes the opaque native runtime and its hidden conformance ABI.

pub mod audio_out;
pub mod candle_ext;
pub mod chat_template;
pub mod data;
pub mod detokenizer;
mod ffi;
#[path = "compute/flashkern/mod.rs"]
pub mod flashkern;
pub mod handles;
pub mod loader;
pub mod mimi_native;
pub mod model;
pub mod moshi;
pub mod processor;
pub mod realtime;
pub mod resample;
#[path = "compute/threads.rs"]
pub mod threads;
pub mod trainer;
pub mod utils;
#[path = "compute/weights.rs"]
pub mod weights;

pub use audio_out::{AudioDetokenizer, MimiDetokenizer};
pub use detokenizer::LFM2AudioDetokenizer;
pub use handles::{
    ConversationConfig as NativeConversationConfig, EmbeddingKind, ModelInfo as NativeModelInfo,
    NativeConversation, NativeError, NativeModel, TokenResult as NativeTokenResult,
};
pub use liquid_audio::{
    CaptureDock, CaptureTicket, FrameConfig, FrameSubmitError, PcmSink, RealtimeFramePipeline,
    RealtimeFramePipelineHandle, RealtimePipeline, RealtimePipelineHandle, Utterance, VoiceEngine,
    VoiceEvent,
};
pub use loader::{from_pretrained, from_pretrained_hub};
pub use model::lfm2_audio::{GenParams, GenToken, LFM2AudioModel, PrefillCursor};
pub use processor::{ChatState, LFM2AudioProcessor, SpecialTokenIds};
pub use realtime::{
    ConversationVault, Lfm2VoiceEngine, MoshiVoiceEngine, RealtimeMoshi, RealtimeMoshiEvent,
    RealtimeMoshiFiles, RealtimeMoshiParams, REALTIME_MOSHI_WARMUP_FRAMES,
};
pub use threads::{configure_intraop_threads, intraop_default_num_threads};
pub use trainer::{Trainer, TrainerConfig};
pub use utils::{get_model_dir, LFMModality};

#[macro_export]
macro_rules! vtrace {
    ($($arg:tt)*) => {{
        if $crate::voice_trace_enabled() {
            eprintln!("[voice +{:.3}s] {}", $crate::voice_trace_elapsed(), format_args!($($arg)*));
        }
    }};
}

#[doc(hidden)]
pub fn voice_trace_enabled() -> bool {
    std::env::var_os("EMBER_VOICE_TRACE").is_some()
}

#[doc(hidden)]
pub fn voice_trace_elapsed() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}
