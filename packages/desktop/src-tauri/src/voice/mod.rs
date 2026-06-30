//! Native (desktop-only) voice pipeline.
//!
//! Replaces voice-side worker processes with services owned by the Tauri desktop
//! process:
//!
//! ```text
//! Solid UI -> Tauri command -> VoiceRuntime
//!                         -> cpal mic -> Rust VAD -> RealtimePipeline worker
//!                         -> cpal playback + Tauri Channel state/events
//! ```
//!
//! `control` is the settings-driven command seam. `runtime` is the managed
//! service layer: CPAL streams, VAD, playback, interruption, model inference, and
//! cleanup all live in Rust threads. `session` remains the reducer for the
//! sidecar-backed LiveKit bridge while that provider still exists.
pub mod control;
pub mod runtime;
pub mod session;
