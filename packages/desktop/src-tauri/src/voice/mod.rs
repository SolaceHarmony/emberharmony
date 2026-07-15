//! Native (desktop-only) voice pipeline.
//!
//! Replaces voice-side worker processes with services owned by the Tauri desktop
//! process:
//!
//! ```text
//! Solid UI -> Tauri command -> VoiceRuntime kernel
//!                         -> LFM2: native mic -> Rust VAD -> RealtimePipeline worker
//!                         -> LiveKit: Rust Room + PlatformAudio + native WebRTC media
//!                         -> Tauri Channel state/events
//! ```
//!
//! `control` is the settings-driven command seam. `runtime` is the managed
//! service layer: provider sessions, audio callbacks, VAD, playback,
//! interruption, model inference, LiveKit media, and cleanup all live in Rust.
//! `session` is the reducer/runner for delegated turns into the EmberHarmony
//! session backend, not the desktop media owner.
pub mod control;
pub mod livekit;
pub mod model;
pub mod runtime;
pub mod session;
mod threads;
