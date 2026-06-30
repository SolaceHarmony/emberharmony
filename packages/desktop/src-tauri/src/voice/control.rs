//! Voice control — the seam between the settings store and the two provider
//! pipelines: the local LFM2-Audio loop (`lfm2`) and the LiveKit session bridge
//! (`livekit`, see [`super::session`]).
//!
//! This layer wires **settings → provider readiness → runtime control**. For
//! `lfm2`, `voice_start` enters the in-process Rust service; CPAL capture,
//! playback, VAD, interruption, and the model worker are owned by Tauri. LiveKit
//! remains the legacy bridge path while that provider still exists.

use crate::settings::{self, VoiceProvider, VoiceSettings};
use crate::{ServerReadyData, ServerState};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tauri::{AppHandle, State};

use super::runtime::VoiceRuntime;
use super::session::{SessionBridgeConfig, SessionBridgeModel, VOICE_SYSTEM_PROMPT, url_encode};

/// Model selected by the session when the voice runtime was started.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModel {
    #[serde(rename = "providerID")]
    pub provider_id: String,
    #[serde(rename = "modelID")]
    pub model_id: String,
}

/// Session context bound to a native voice run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCtx {
    #[serde(rename = "sessionID")]
    pub session_id: String,
    pub directory: String,
    pub agent: Option<String>,
    pub model: Option<SessionModel>,
    pub variant: Option<String>,
    #[serde(rename = "delegateTarget")]
    pub delegate_target: Option<String>,
    #[serde(rename = "promptMode")]
    pub prompt_mode: Option<String>,
}

/// LiveKit room grant returned by the local server dispatch endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveKitGrant {
    pub token: String,
    pub url: String,
    #[serde(rename = "roomName")]
    pub room_name: String,
}

/// Result of starting the configured provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum VoiceStartResult {
    Lfm2,
    Livekit { grant: LiveKitGrant },
}

/// Whether the active provider is ready to start a voice session, and what to do
/// about it if not. Drives the readiness hint in the voice settings panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoicePlan {
    /// The active provider.
    pub provider: VoiceProvider,
    /// Whether voice mode is enabled at all.
    pub enabled: bool,
    /// Which runtime surface owns the selected provider.
    pub surface: VoiceSurface,
    /// Whether the native runtime has an active service thread.
    pub running: bool,
    /// Provider currently owned by the runtime, if any.
    #[serde(rename = "runningProvider")]
    pub running_provider: Option<VoiceProvider>,
    /// Whether the native runtime currently accepts microphone input.
    #[serde(rename = "micEnabled")]
    pub mic_enabled: bool,
    /// Ready to start.
    pub ready: bool,
    /// Human-readable detail — what to configure if not ready.
    pub detail: String,
}

/// Runtime surface for the active provider. LFM2 is owned by the desktop Rust
/// service; LiveKit is still the webview media room while that provider exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceSurface {
    Off,
    Native,
    Livekit,
}

/// Pure decision: given the settings, which provider runs and is it ready.
pub fn plan(settings: &VoiceSettings) -> VoicePlan {
    match settings.provider {
        VoiceProvider::Off => VoicePlan {
            provider: VoiceProvider::Off,
            enabled: false,
            surface: VoiceSurface::Off,
            running: false,
            running_provider: None,
            mic_enabled: false,
            ready: false,
            detail: "Voice is off.".into(),
        },
        VoiceProvider::Lfm2 => {
            let configured_dir = settings::lfm2_model_dir(&settings.lfm2);
            let valid_dir = configured_dir
                .as_ref()
                .is_some_and(|d| d.join("config.json").is_file());
            let model = settings::lfm2_model_ref(&settings.lfm2);
            let remote = model == settings::DEFAULT_LFM2_MODEL;
            VoicePlan {
                provider: VoiceProvider::Lfm2,
                enabled: true,
                surface: VoiceSurface::Native,
                running: false,
                running_provider: None,
                mic_enabled: false,
                ready: true,
                detail: if valid_dir {
                    "Local LFM2-Audio model ready.".into()
                } else if remote {
                    "LFM2-Audio will use the Hugging Face cache and download on first start if needed."
                        .into()
                } else {
                    format!("LFM2-Audio model `{model}` will download on first start if needed.")
                },
            }
        }
        VoiceProvider::Livekit => VoicePlan {
            provider: VoiceProvider::Livekit,
            enabled: true,
            surface: VoiceSurface::Livekit,
            running: false,
            running_provider: None,
            mic_enabled: false,
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
pub fn voice_status(app: AppHandle, runtime: State<'_, VoiceRuntime>) -> Result<VoicePlan, String> {
    let mut p = plan(&settings::load(&app));
    let active = runtime.active_provider()?;
    p.running = active.is_some();
    p.running_provider = active;
    p.mic_enabled = p.running && runtime.mic_enabled()?;
    Ok(p)
}

// ---- streaming contract: the run loops emit these to the webview over a
// `tauri::ipc::Channel` (ordered, high-throughput — see the calling-rust docs) ----

/// Voice session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceState {
    Loading,
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
///
/// Covers both the Phase-1 turn flow (the Liquid AI demo: `Transcript` streams the reply text,
/// `AudioClip` delivers the decoded reply for an `<audio>` player) and the Phase-2 live flow
/// (`Level` drives the visualizer since native audio never enters the webview as a track —
/// see `FRONTEND_DESIGN.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum VoiceEvent {
    /// session state changed
    State { state: VoiceState },
    /// reply text so far (cumulative, matching the demo's streamed text)
    Transcript { role: Role, text: String },
    /// audio amplitude (RMS) for the bar visualizer — the native path has no
    /// `MediaStreamTrack` in the webview, so the loop emits this instead.
    Level { rms: f32 },
    /// the decoded audio reply as a WAV clip (turn mode → inline `<audio>` player).
    AudioClip { wav: Vec<u8>, ms: u32 },
    /// the session ended (cleanly, or with a reason)
    Ended { reason: Option<String> },
    /// an error occurred
    Error { message: String },
}

/// The Liquid AI demo's three modes — same model, different system prompt + generate path.
/// (`audio-model.js`: `Perform ASR.` / `Perform TTS. Use the UK female voice.` /
/// `Respond with interleaved text and audio.`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnMode {
    /// audio in → text out (transcription). `generate_sequential`, text only.
    Asr,
    /// text in → audio out (speech). `generate_sequential`, text then audio.
    Tts,
    /// audio (± text) in → interleaved text + audio out. `generate_interleaved`.
    Interleaved,
}

impl TurnMode {
    /// The demo's per-mode system prompt (verbatim from `audio-model.js`).
    pub fn system_prompt(self) -> &'static str {
        match self {
            TurnMode::Asr => "Perform ASR.",
            TurnMode::Tts => "Perform TTS. Use the UK female voice.",
            TurnMode::Interleaved => "Respond with interleaved text and audio.",
        }
    }

    /// The demo's per-mode token budget (`DEFAULT_MAX_TOKENS_*`).
    pub fn max_new_tokens(self) -> usize {
        match self {
            TurnMode::Asr => 100,
            TurnMode::Tts | TurnMode::Interleaved => 1024,
        }
    }
}

/// Start a voice session for the configured provider.
///
/// For `lfm2`, this starts the thread-managed desktop service: CPAL mic,
/// speaker playback, VAD, interruption, and the realtime model pipeline all run
/// inside the Tauri process. The command returns after the service thread is
/// spawned; events stream over `channel`.
#[tauri::command]
pub async fn voice_start(
    app: AppHandle,
    runtime: State<'_, VoiceRuntime>,
    server: State<'_, ServerState>,
    ctx: SessionCtx,
    channel: tauri::ipc::Channel<VoiceEvent>,
) -> Result<VoiceStartResult, String> {
    let settings = settings::load(&app);
    let p = plan(&settings);
    if !p.ready {
        return Err(p.detail);
    }
    match p.provider {
        VoiceProvider::Lfm2 => {
            let ready = server
                .status
                .clone()
                .await
                .map_err(|_| "Failed to get server status".to_string())??;
            let bridge = session_bridge_config(&ctx, ready);
            runtime.start_lfm2(ctx, settings, channel, bridge)?;
            Ok(VoiceStartResult::Lfm2)
        }
        VoiceProvider::Livekit => {
            let ready = server
                .status
                .clone()
                .await
                .map_err(|_| "Failed to get server status".to_string())??;
            runtime.start_livekit(ctx.clone())?;
            let grant = match livekit_grant(&ctx, ready).await {
                Ok(grant) => grant,
                Err(error) => {
                    let _ = runtime.stop();
                    return Err(error);
                }
            };
            if !runtime.is_running_session(&ctx.session_id)? {
                return Err("Voice start was cancelled.".into());
            }
            Ok(VoiceStartResult::Livekit { grant })
        }
        VoiceProvider::Off => Err("Voice is off.".into()),
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LiveKitTokenRequest {
    #[serde(rename = "sessionID")]
    session_id: String,
    model: Option<SessionModel>,
}

async fn livekit_grant(ctx: &SessionCtx, server: ServerReadyData) -> Result<LiveKitGrant, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("voice token client: {e}"))?;
    let body = LiveKitTokenRequest {
        session_id: ctx.session_id.clone(),
        model: ctx.model.clone(),
    };
    let req = client
        .post(format!("{}/voice/token", server.url.trim_end_matches('/')))
        .header("content-type", "application/json")
        .header("x-emberharmony-directory", url_encode(&ctx.directory))
        .json(&body);
    let req = match server.password.as_deref() {
        Some(password) => req.basic_auth("emberharmony", Some(password)),
        None => req,
    };
    let response = req
        .send()
        .await
        .map_err(|e| format!("voice token request failed: {e}"))?;
    if response.status().is_success() {
        return response
            .json::<LiveKitGrant>()
            .await
            .map_err(|e| format!("voice token response failed: {e}"));
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(format!("voice token request failed: {status} {body}"))
}

fn session_bridge_config(ctx: &SessionCtx, server: ServerReadyData) -> Option<SessionBridgeConfig> {
    let target = ctx.delegate_target.as_deref()?.trim();
    if target.is_empty() {
        return None;
    }
    Some(SessionBridgeConfig {
        server_url: server.url,
        directory: ctx.directory.clone(),
        session_id: ctx.session_id.clone(),
        username: server.password.as_ref().map(|_| "emberharmony".to_string()),
        password: server.password,
        agent: ctx.prompt_mode.clone().or_else(|| ctx.agent.clone()),
        model: parse_model_ref(target).or_else(|| {
            ctx.model.as_ref().map(|model| SessionBridgeModel {
                provider_id: model.provider_id.clone(),
                model_id: model.model_id.clone(),
            })
        }),
        variant: ctx.variant.clone(),
        system: Some(VOICE_SYSTEM_PROMPT.to_string()),
    })
}

fn parse_model_ref(value: &str) -> Option<SessionBridgeModel> {
    let (provider, model) = value.split_once('/')?;
    let provider = provider.trim();
    let model = model.trim();
    if provider.is_empty() || model.is_empty() {
        return None;
    }
    Some(SessionBridgeModel {
        provider_id: provider.to_string(),
        model_id: model.to_string(),
    })
}

/// Stop the active voice session.
#[tauri::command]
pub async fn voice_stop(runtime: State<'_, VoiceRuntime>) -> Result<(), String> {
    runtime.stop()
}

/// Interrupt the current native reply without disconnecting the session.
#[tauri::command]
pub async fn voice_interrupt(runtime: State<'_, VoiceRuntime>) -> Result<(), String> {
    runtime.interrupt()
}

/// Pause/resume native microphone capture without tearing down the session.
#[tauri::command]
pub async fn voice_set_mic_enabled(
    runtime: State<'_, VoiceRuntime>,
    enabled: bool,
) -> Result<(), String> {
    runtime.set_mic_enabled(enabled)
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
    fn lfm2_uses_downloadable_default_without_a_model_dir() {
        assert!(plan(&settings(VoiceProvider::Lfm2, None)).ready);
        assert!(plan(&settings(VoiceProvider::Lfm2, Some("   "))).ready);
    }

    #[test]
    fn lfm2_accepts_a_model_dir_with_config() {
        let dir =
            std::env::temp_dir().join(format!("emberharmony-lfm2-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        let path = dir.to_string_lossy().into_owned();
        assert!(plan(&settings(VoiceProvider::Lfm2, Some(&path))).ready);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn livekit_defers_to_the_sidecar() {
        assert!(plan(&settings(VoiceProvider::Livekit, None)).ready);
    }

    #[test]
    fn plan_reports_enabled_surface_and_runtime_defaults() {
        let p = plan(&settings(VoiceProvider::Livekit, None));
        assert!(p.enabled);
        assert_eq!(p.surface, VoiceSurface::Livekit);
        assert!(!p.running);
        assert_eq!(p.running_provider, None);
        assert!(!p.mic_enabled);
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

    #[test]
    fn voice_start_result_serializes_provider_tag() {
        let v = serde_json::to_value(VoiceStartResult::Livekit {
            grant: LiveKitGrant {
                token: "t".into(),
                url: "wss://voice.example".into(),
                room_name: "emberharmony_ses_123".into(),
            },
        })
        .unwrap();
        assert_eq!(v["provider"], "livekit");
        assert_eq!(v["grant"]["roomName"], "emberharmony_ses_123");
    }

    #[test]
    fn parses_delegate_model_ref() {
        assert_eq!(
            parse_model_ref("glm/z1"),
            Some(SessionBridgeModel {
                provider_id: "glm".into(),
                model_id: "z1".into()
            })
        );
        assert_eq!(parse_model_ref("glm"), None);
        assert_eq!(parse_model_ref("/z1"), None);
        assert_eq!(parse_model_ref("glm/"), None);
    }

    #[test]
    fn session_ctx_accepts_frontend_names() {
        let ctx: SessionCtx = serde_json::from_value(serde_json::json!({
            "sessionID": "ses_123",
            "directory": "/tmp/project",
            "agent": "plan",
            "model": {
                "providerID": "openai",
                "modelID": "gpt-5"
            },
            "variant": "xhigh",
            "delegateTarget": "glm/z1",
            "promptMode": "plan"
        }))
        .unwrap();
        assert_eq!(ctx.session_id, "ses_123");
        assert_eq!(ctx.directory, "/tmp/project");
        assert_eq!(ctx.agent.as_deref(), Some("plan"));
        assert_eq!(ctx.model.as_ref().unwrap().provider_id, "openai");
        assert_eq!(ctx.model.as_ref().unwrap().model_id, "gpt-5");
        assert_eq!(ctx.variant.as_deref(), Some("xhigh"));
        assert_eq!(ctx.delegate_target.as_deref(), Some("glm/z1"));
        assert_eq!(ctx.prompt_mode.as_deref(), Some("plan"));
    }
}
