//! Native LFM2-Audio host surface.
//!
//! Model computation and checkpoint interpretation live behind the opaque
//! C++/Flashkern runtime. Framework-backed reference code is physically owned
//! by the workspace-only `liquid-audio-oracle` crate.

mod ffi;
pub mod native_voice;
#[path = "runtime/realtime.rs"]
pub mod realtime;
#[path = "runtime/resample.rs"]
pub mod resample;
pub mod utils;
mod voice_api;
#[path = "runtime/voice_runtime.rs"]
pub mod voice_runtime;

pub use native_voice::{
    NativeConversationVault, NativeLfm2VoiceEngine, NativeVoiceModel, NativeVoiceModelMemory,
    NativeVoiceRuntimeConfig, NativeVoiceSampling,
};
pub use realtime::{
    FrameSubmitError, RealtimeFramePipeline, RealtimeFramePipelineHandle, RealtimePipeline,
    RealtimePipelineHandle,
};
#[cfg(feature = "download")]
pub use utils::{snapshot_download_to, snapshot_download_with, DownloadProgress};
pub use voice_api::{
    CaptureDock, CaptureTicket, FrameConfig, PcmSink, Utterance, VoiceEngine, VoiceEvent,
};
pub use voice_runtime::{
    AudioStatsSnapshot, ExternalAudioInput, ExternalAudioInputWriter, ExternalAudioOutput,
    RuntimeConfig, RuntimeEvent, SessionState, VoiceRuntime,
};
