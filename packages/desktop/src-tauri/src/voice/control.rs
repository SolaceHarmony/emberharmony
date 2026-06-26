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
}
