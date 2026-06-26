//! Voice control — the seam between the settings store and the two provider
//! pipelines: the local LFM2-Audio loop (`lfm2`) and the LiveKit session bridge
//! (`livekit`, see [`super::session`]).
//!
//! This phase wires **settings → provider readiness**, exposed to the webview as
//! the `voice_status` command. The streaming start/stop loop plugs in here next:
//! for `lfm2`, cpal capture + the candle model from `experiments/lfm2-audio-voice`;
//! for `livekit`, the SSE reducer in [`super::session`]. Both will surface
//! transcript/state to the webview over a `tauri::ipc::Channel<VoiceEvent>` —
//! ordered, high-throughput streaming rather than events.

use crate::settings::{self, VoiceProvider, VoiceSettings};
use serde::{Deserialize, Serialize};
use tauri::AppHandle;

/// Whether the active provider is ready to start a voice session, and what to do
/// about it if not. Drives the readiness hint in the voice settings panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoicePlan {
    /// The active provider.
    pub provider: VoiceProvider,
    /// Ready to start.
    pub ready: bool,
    /// Human-readable detail — what to configure if not ready.
    pub detail: String,
}

/// Pure decision: given the settings, which provider runs and is it ready.
pub fn plan(settings: &VoiceSettings) -> VoicePlan {
    match settings.provider {
        VoiceProvider::Off => VoicePlan {
            provider: VoiceProvider::Off,
            ready: false,
            detail: "Voice is off.".into(),
        },
        VoiceProvider::Lfm2 => {
            let has_model = settings
                .lfm2
                .model_dir
                .as_deref()
                .is_some_and(|d| !d.trim().is_empty());
            VoicePlan {
                provider: VoiceProvider::Lfm2,
                ready: has_model,
                detail: if has_model {
                    "Local LFM2-Audio model ready.".into()
                } else {
                    "Set the model directory to enable the local model.".into()
                },
            }
        }
        VoiceProvider::Livekit => VoicePlan {
            provider: VoiceProvider::Livekit,
            // LiveKit readiness (URL + credentials) is owned by the sidecar / the
            // LiveKit panel, not this store — the session bridge picks it up at
            // dispatch — so the native side reports it as configured there.
            ready: true,
            detail: "LiveKit is configured in the connection panel.".into(),
        },
    }
}

/// Report whether the configured voice provider is ready to start.
#[tauri::command]
pub fn voice_status(app: AppHandle) -> Result<VoicePlan, String> {
    Ok(plan(&settings::load(&app)))
}

// ---- streaming contract: the run loops emit these to the webview over a
// `tauri::ipc::Channel` (ordered, high-throughput — see the calling-rust docs) ----

/// Voice session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceState {
    Idle,
    Listening,
    Thinking,
    Speaking,
}

/// Who is speaking in a transcript chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// An event streamed to the webview during a voice session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum VoiceEvent {
    /// session state changed
    State { state: VoiceState },
    /// a chunk of recognized speech (user) or spoken reply (assistant)
    Transcript { role: Role, text: String },
    /// the session ended (cleanly, or with a reason)
    Ended { reason: Option<String> },
    /// an error occurred
    Error { message: String },
}

/// Start a voice session for the configured provider, streaming [`VoiceEvent`]s
/// over `channel`. Errors (with the readiness detail) if the provider isn't ready.
///
/// The provider loop bodies are the next phase: for `lfm2`, cpal capture + the
/// candle model from `experiments/lfm2-audio-voice`; for `livekit`, an SSE runner
/// driving [`super::session`]'s reducer. Both stream `Transcript`/`State` here.
#[tauri::command]
pub async fn voice_start(app: AppHandle, channel: tauri::ipc::Channel<VoiceEvent>) -> Result<(), String> {
    let p = plan(&settings::load(&app));
    if !p.ready {
        return Err(p.detail);
    }
    let _ = channel.send(VoiceEvent::State { state: VoiceState::Idle });
    match p.provider {
        VoiceProvider::Lfm2 => Err("the local LFM2-Audio loop is not wired up yet".into()),
        VoiceProvider::Livekit => Err("the LiveKit session-bridge loop is not wired up yet".into()),
        VoiceProvider::Off => Err("Voice is off.".into()),
    }
}

/// Stop the active voice session.
#[tauri::command]
pub async fn voice_stop(_app: AppHandle) -> Result<(), String> {
    // next phase: signal the active provider task to drain + stop.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Lfm2Settings;

    fn settings(provider: VoiceProvider, model_dir: Option<&str>) -> VoiceSettings {
        VoiceSettings {
            provider,
            lfm2: Lfm2Settings {
                model_dir: model_dir.map(String::from),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn off_is_not_ready() {
        assert!(!plan(&settings(VoiceProvider::Off, None)).ready);
    }

    #[test]
    fn lfm2_needs_a_model_dir() {
        assert!(!plan(&settings(VoiceProvider::Lfm2, None)).ready);
        assert!(!plan(&settings(VoiceProvider::Lfm2, Some("   "))).ready);
        assert!(plan(&settings(VoiceProvider::Lfm2, Some("/models/lfm2"))).ready);
    }

    #[test]
    fn livekit_defers_to_the_sidecar() {
        assert!(plan(&settings(VoiceProvider::Livekit, None)).ready);
    }

    #[test]
    fn voice_event_tagged_serialization() {
        let t = serde_json::to_value(VoiceEvent::Transcript {
            role: Role::User,
            text: "hi".into(),
        })
        .unwrap();
        assert_eq!(t["type"], "transcript");
        assert_eq!(t["role"], "user");
        assert_eq!(t["text"], "hi");
        let s = serde_json::to_value(VoiceEvent::State {
            state: VoiceState::Speaking,
        })
        .unwrap();
        assert_eq!(s["type"], "state");
        assert_eq!(s["state"], "speaking");
    }
}
