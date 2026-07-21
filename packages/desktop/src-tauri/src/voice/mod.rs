//! Native (desktop-only) voice pipeline.
//!
//! Replaces voice-side worker processes with services owned by the Tauri desktop
//! process:
//!
//! ```text
//! Solid UI -> Tauri command -> VoiceRuntime kernel
//!                         -> LFM2: Rust device callbacks -> native PCM docks
//!                                  -> native Sesame -> kcoro/Flashkern inference
//!                         -> LiveKit: Rust Room + PlatformAudio + native WebRTC media
//!                         -> Tauri Channel state/events
//! ```
//!
//! `control` is the settings-driven command seam. `runtime` is the managed
//! service layer: provider sessions, platform callbacks, playback ownership,
//! interruption, LiveKit media, and cleanup live in Rust. Native LFM2 owns
//! Sesame turn policy, model inference, and coroutine progress behind opaque
//! capture/playback endpoints.
//! `session` is the reducer/runner for delegated turns into the EmberHarmony
//! session backend, not the desktop media owner.
pub mod control;
pub mod livekit;
pub mod model;
pub mod runtime;
pub mod session;
mod threads;
