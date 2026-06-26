//! Native (desktop-only) voice pipeline.
//!
//! Replaces the LiveKit + `@livekit/agents` Node worker with a pure Rust pipeline
//! running on real tokio tasks inside the Tauri process:
//!
//! ```text
//! cpal mic -> Silero VAD (ort) -> [turn detector] -> STT
//!          -> session sidecar (HTTP /prompt_async + /event SSE)   <- this module
//!          -> TTS -> cpal playback        (barge-in -> stop TTS + POST /abort)
//! ```
//!
//! Phase 0 (`session`) is the bridge to the EmberHarmony session sidecar — a
//! direct port of `packages/emberharmony/src/voice/bridge.ts`, whose 16-test
//! harness is the behavioural spec. `control` is the settings-driven seam that
//! selects between the two providers (the local LFM2-Audio loop and this session
//! bridge). The audio/STT/TTS loop lands on top.
pub mod control;
pub mod session;
