//! Native LFM2-Audio host surface.
//!
//! The default dependency graph is the opaque native runtime and contains no
//! Candle or Moshi numerical implementation. The deleted Rust implementation,
//! training tools, and fixture-capture references remain available only through
//! the workspace-only `liquid-audio-oracle` package / `oracle` feature.

#[cfg(feature = "oracle")]
#[path = "runtime/audio_out.rs"]
pub mod audio_out; // AudioDetokenizer trait + backends (LFM2 detok / Mimi)
#[cfg(feature = "oracle")]
pub mod candle_ext; // vendored candle 0.10 backports + extensions (kept on the 0.9.2 pin)
#[cfg(feature = "oracle")]
pub mod chat_template; // load-time verification vs the snapshot chat_template.jinja
#[cfg(feature = "oracle")]
pub mod data; // data/ (data-pipeline value types)
#[cfg(feature = "oracle")]
pub mod detokenizer; // detokenizer.py
mod ffi;
#[cfg(feature = "oracle")]
#[path = "compute/flashkern/mod.rs"]
pub mod flashkern; // temporary Rust ABI rims for the native Flashkern engine
#[cfg(feature = "oracle")]
pub mod handles;
#[cfg(feature = "oracle")]
pub mod loader; // config.json + safetensors → model + processor
#[cfg(feature = "oracle")]
pub mod mimi_native; // native C++/NEON/AMX Mimi decode kernel rim (native/src/mimi)
#[cfg(feature = "oracle")]
pub mod model;
#[cfg(feature = "oracle")]
pub mod moshi; // Liquid-Audio-facing facade over Kyutai's Rust moshi crate
pub mod native_voice; // opaque native LFM2 lifecycle + PCM dock host seam
#[cfg(feature = "oracle")]
pub mod processor; // processor.py
#[path = "runtime/realtime.rs"]
pub mod realtime; // multi-threaded worker pipeline (chat.py producer/consumer threading)
#[path = "runtime/resample.rs"]
pub mod resample; // torchaudio.functional.resample (windowed-sinc) port
#[cfg(feature = "oracle")]
#[path = "compute/threads.rs"]
pub mod threads; // intra-op thread-pool parity with torch (at::intraop_default_num_threads)
#[cfg(feature = "oracle")]
pub mod trainer; // trainer.py
pub mod utils;
mod voice_api;
#[path = "runtime/voice_runtime.rs"]
pub mod voice_runtime;
#[cfg(feature = "oracle")]
#[path = "compute/weights.rs"]
pub mod weights; // native resident checkpoint image + temporary Candle compatibility boundary // in-process thread-managed voice service (external I/O, VAD, realtime)

#[cfg(feature = "oracle")]
pub use audio_out::{AudioDetokenizer, MimiDetokenizer};
#[cfg(feature = "oracle")]
pub use detokenizer::LFM2AudioDetokenizer;
#[cfg(feature = "oracle")]
pub use handles::{
    ConversationConfig as NativeConversationConfig, EmbeddingKind, ModelInfo as NativeModelInfo,
    NativeConversation, NativeError, NativeModel, TokenResult as NativeTokenResult,
};
#[cfg(feature = "oracle")]
pub use loader::{from_pretrained, from_pretrained_hub};
#[cfg(feature = "oracle")]
pub use model::lfm2_audio::{GenParams, GenToken, LFM2AudioModel, PrefillCursor};
pub use native_voice::{
    NativeConversationVault, NativeLfm2VoiceEngine, NativeVoiceModel, NativeVoiceModelMemory,
    NativeVoiceRuntimeConfig, NativeVoiceSampling,
};
#[cfg(feature = "oracle")]
pub use processor::{ChatState, LFM2AudioProcessor, SpecialTokenIds};
pub use realtime::{
    FrameSubmitError, RealtimeFramePipeline, RealtimeFramePipelineHandle, RealtimePipeline,
    RealtimePipelineHandle,
};
#[cfg(feature = "oracle")]
pub use realtime::{ConversationVault, Lfm2VoiceEngine, MoshiVoiceEngine};
#[cfg(feature = "oracle")]
pub use threads::{configure_intraop_threads, intraop_default_num_threads};
#[cfg(feature = "oracle")]
pub use trainer::{Trainer, TrainerConfig};
pub use utils::{get_model_dir, LFMModality};
pub use voice_api::{FrameConfig, Utterance, VoiceEngine, VoiceEvent};
#[cfg(feature = "download")]
pub use utils::{snapshot_download_to, snapshot_download_with, DownloadProgress};
pub use voice_runtime::{
    AudioStatsSnapshot, ExternalAudioInput, ExternalAudioInputWriter, ExternalAudioOutput,
    RuntimeConfig, RuntimeEvent, SessionState, VoiceRuntime,
};
// pub use model::lfm2_audio::LFM2AudioModel;
// pub use processor::{ChatState, LFM2AudioProcessor};
