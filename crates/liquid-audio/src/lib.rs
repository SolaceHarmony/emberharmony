//! Native LFM2-Audio host surface.
//!
//! Model computation and checkpoint interpretation live behind the opaque
//! C++/Flashkern runtime. Framework-backed reference code is physically owned
//! by the workspace-only `liquid-audio-oracle` crate.

mod ffi;
pub mod native_voice;
pub mod utils;
mod voice_api;
#[path = "runtime/voice_runtime.rs"]
pub mod voice_runtime;

pub use native_voice::{
    NativeConversationVault, NativeLfm2VoiceEngine, NativeVoiceModel, NativeVoiceModelMemory,
    NativeVoiceRuntimeConfig, NativeVoiceSampling,
};
#[cfg(feature = "download")]
pub use utils::{snapshot_download_to, snapshot_download_with, DownloadProgress};
pub use voice_api::{
    CaptureSink, CaptureWrite, EngineProgress, PlaybackSource, PlaybackWrite, VoiceEngine,
    VoiceEvent,
};
pub use voice_runtime::{
    AudioStatsSnapshot, RuntimeConfig, RuntimeEvent, SessionState, VoiceRuntime,
};
