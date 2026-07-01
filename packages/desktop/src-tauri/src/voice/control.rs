//! Voice control — the seam between the settings store and the two native
//! provider pipelines: the local LFM2-Audio loop (`lfm2`) and the Rust LiveKit
//! session (`livekit`).
//!
//! This layer wires **settings → provider readiness → runtime control**. For
//! `lfm2`, `voice_start` enters the in-process Rust service; CPAL capture,
//! playback, VAD, interruption, and the model worker are owned by Tauri. LiveKit
//! follows the same Tauri-owned runtime path: the webview asks for voice, Rust
//! owns the room, microphone, stop, interrupt, and level events.

use crate::settings::{self, VoiceProvider, VoiceSettings};
use crate::{ServerReadyData, ServerState};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

use super::livekit;
use super::runtime::VoiceRuntime;
use super::session::{SessionBridgeConfig, SessionBridgeModel, VOICE_SYSTEM_PROMPT};

const LIVEKIT_READY_DETAIL: &str =
    "LiveKit URL, credentials, and local LFM2-Audio model ready for the native agent.";

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
    #[serde(rename = "promptMode")]
    pub prompt_mode: Option<String>,
}

/// LiveKit room grant minted by the native desktop provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveKitGrant {
    pub token: String,
    #[serde(rename = "agentToken")]
    pub agent_token: String,
    pub url: String,
    #[serde(rename = "roomName")]
    pub room_name: String,
    #[serde(rename = "userIdentity")]
    pub user_identity: String,
    #[serde(rename = "agentIdentity")]
    pub agent_identity: String,
}

/// Result of starting the configured provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum VoiceStartResult {
    Lfm2,
    Livekit,
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

/// Runtime surface for the active provider. Desktop providers are owned by the
/// Tauri voice kernel; `Livekit` remains only for non-desktop/web legacy status.
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
            // Fail-hard, decoupled: the RUN path loads only a local snapshot dir. A repo id is
            // a download *source*, not a ready model — typing one never silently downloads at
            // start. Ready iff a local model dir with `config.json` exists.
            let active = settings::lfm2_active_model_dir(&settings.lfm2).is_some();
            VoicePlan {
                provider: VoiceProvider::Lfm2,
                enabled: true,
                surface: VoiceSurface::Native,
                running: false,
                running_provider: None,
                mic_enabled: false,
                ready: active,
                detail: if active {
                    "Local LFM2-Audio model ready.".into()
                } else {
                    "No local model. Download a model or choose a model directory below.".into()
                },
            }
        }
        VoiceProvider::Livekit => VoicePlan {
            provider: VoiceProvider::Livekit,
            enabled: true,
            surface: VoiceSurface::Native,
            running: false,
            running_provider: None,
            mic_enabled: false,
            ready: settings
                .livekit
                .url
                .as_deref()
                .is_some_and(|url| !url.trim().is_empty())
                && settings::lfm2_active_model_dir(&settings.lfm2).is_some(),
            detail: "LiveKit needs a URL, keychain-stored API credentials, and a local LFM2-Audio model.".into(),
        },
    }
}

/// Report whether the configured voice provider is ready to start.
#[tauri::command]
pub async fn voice_status(
    app: AppHandle,
    runtime: State<'_, VoiceRuntime>,
) -> Result<VoicePlan, String> {
    let settings = settings::load(&app);
    let mut p = plan(&settings);
    if settings.provider == VoiceProvider::Livekit {
        p.ready = livekit::configured(&settings)?
            && settings::lfm2_active_model_dir(&settings.lfm2).is_some();
        p.detail = if p.ready {
            LIVEKIT_READY_DETAIL.into()
        } else if settings::lfm2_active_model_dir(&settings.lfm2).is_none() {
            "Choose or download a local LFM2-Audio model for the native LiveKit agent.".into()
        } else {
            "Enter your LiveKit URL, API key, and API secret in voice settings.".into()
        };
    }
    let active = runtime.snapshot().await?;
    p.running = active.running;
    p.running_provider = active.running_provider;
    p.mic_enabled = active.mic_enabled;
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
    let mut p = plan(&settings);
    if settings.provider == VoiceProvider::Livekit {
        p.ready = livekit::configured(&settings)?
            && settings::lfm2_active_model_dir(&settings.lfm2).is_some();
        p.detail = if p.ready {
            LIVEKIT_READY_DETAIL.into()
        } else if settings::lfm2_active_model_dir(&settings.lfm2).is_none() {
            "Choose or download a local LFM2-Audio model for the native LiveKit agent.".into()
        } else {
            "Enter your LiveKit URL, API key, and API secret in voice settings.".into()
        };
    }
    if !p.ready {
        return Err(p.detail);
    }
    match p.provider {
        VoiceProvider::Lfm2 => {
            let bridge = lfm2_bridge_config(&settings, &ctx, &server).await?;
            runtime.start_lfm2(ctx, settings, channel, bridge).await?;
            Ok(VoiceStartResult::Lfm2)
        }
        VoiceProvider::Livekit => {
            let bridge = lfm2_bridge_config(&settings, &ctx, &server).await?;
            let grant = livekit::grant(&settings, &ctx).await?;
            runtime
                .start_livekit(ctx.clone(), settings, grant, channel, bridge)
                .await?;
            if !runtime.is_running_session(&ctx.session_id).await? {
                return Err("Voice start was cancelled.".into());
            }
            Ok(VoiceStartResult::Livekit)
        }
        VoiceProvider::Off => Err("Voice is off.".into()),
    }
}

async fn lfm2_bridge_config(
    settings: &VoiceSettings,
    ctx: &SessionCtx,
    server: &ServerState,
) -> Result<Option<SessionBridgeConfig>, String> {
    if !settings.lfm2.delegate.enabled {
        return Ok(None);
    }
    let Some(target) = settings
        .lfm2
        .delegate
        .target
        .as_deref()
        .map(str::trim)
        .filter(|target| !target.is_empty())
    else {
        return Ok(None);
    };
    let ready = server
        .status
        .clone()
        .await
        .map_err(|_| "Failed to get server status".to_string())??;
    Ok(Some(session_bridge_config(ctx, ready, target)))
}

fn session_bridge_config(
    ctx: &SessionCtx,
    server: ServerReadyData,
    target: &str,
) -> SessionBridgeConfig {
    SessionBridgeConfig {
        server_url: server.url,
        directory: ctx.directory.clone(),
        session_id: ctx.session_id.clone(),
        username: server.password.as_ref().map(|_| "emberharmony".to_string()),
        password: server.password,
        // The old LiveKit agent used a server-side VoiceWorkflow to decide
        // plan/build per spoken turn. Until that classifier is native, the
        // desktop kernel must not trust the webview's selected agent/promptMode
        // to grant execution.
        agent: Some("plan".to_string()),
        model: parse_model_ref(target),
        variant: ctx.variant.clone(),
        system: Some(VOICE_SYSTEM_PROMPT.to_string()),
    }
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
    runtime.stop().await
}

/// Interrupt the current native reply without disconnecting the session.
#[tauri::command]
pub async fn voice_interrupt(runtime: State<'_, VoiceRuntime>) -> Result<(), String> {
    runtime.interrupt().await
}

/// Pause/resume native microphone capture without tearing down the session.
#[tauri::command]
pub async fn voice_set_mic_enabled(
    runtime: State<'_, VoiceRuntime>,
    enabled: bool,
) -> Result<(), String> {
    runtime.set_mic_enabled(enabled).await
}

/// Atomically pause native microphone capture and interrupt the active voice turn
/// before a typed prompt proceeds.
#[tauri::command]
pub async fn voice_begin_typed_input(runtime: State<'_, VoiceRuntime>) -> Result<(), String> {
    runtime.begin_typed_input().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{Lfm2Settings, LiveKitSettings};

    fn settings(provider: VoiceProvider, model_dir: Option<&str>) -> VoiceSettings {
        VoiceSettings {
            provider,
            lfm2: Lfm2Settings {
                model_dir: model_dir.map(String::from),
                ..Default::default()
            },
            livekit: LiveKitSettings {
                url: if provider == VoiceProvider::Livekit {
                    Some("wss://livekit.invalid".into())
                } else {
                    None
                },
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
    fn lfm2_not_ready_without_local_model() {
        // A repo id alone is a download source, not a ready model — no silent download.
        assert!(!plan(&settings(VoiceProvider::Lfm2, None)).ready);
        assert!(!plan(&settings(VoiceProvider::Lfm2, Some("   "))).ready);
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
    fn livekit_plan_requires_a_url_before_keychain_readiness() {
        let missing = VoiceSettings {
            provider: VoiceProvider::Livekit,
            livekit: LiveKitSettings::default(),
            ..Default::default()
        };
        assert!(!plan(&missing).ready);
        assert!(!plan(&settings(VoiceProvider::Livekit, None)).ready);
    }

    #[test]
    fn plan_reports_enabled_surface_and_runtime_defaults() {
        let p = plan(&settings(VoiceProvider::Livekit, None));
        assert!(p.enabled);
        assert_eq!(p.surface, VoiceSurface::Native);
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
        let v = serde_json::to_value(VoiceStartResult::Livekit).unwrap();
        assert_eq!(v["provider"], "livekit");
        assert!(v.get("grant").is_none());
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
            "promptMode": "plan"
        }))
        .unwrap();
        assert_eq!(ctx.session_id, "ses_123");
        assert_eq!(ctx.directory, "/tmp/project");
        assert_eq!(ctx.agent.as_deref(), Some("plan"));
        assert_eq!(ctx.model.as_ref().unwrap().provider_id, "openai");
        assert_eq!(ctx.model.as_ref().unwrap().model_id, "gpt-5");
        assert_eq!(ctx.variant.as_deref(), Some("xhigh"));
        assert_eq!(ctx.prompt_mode.as_deref(), Some("plan"));
    }

    #[test]
    fn session_bridge_defaults_voice_delegation_to_plan() {
        let ctx: SessionCtx = serde_json::from_value(serde_json::json!({
            "sessionID": "ses_123",
            "directory": "/tmp/project",
            "agent": "build",
            "model": {
                "providerID": "openai",
                "modelID": "gpt-5"
            },
            "promptMode": "build"
        }))
        .unwrap();
        let cfg = session_bridge_config(
            &ctx,
            ServerReadyData {
                url: "http://127.0.0.1:4096".into(),
                password: None,
            },
            "glm/z1",
        );
        assert_eq!(cfg.agent.as_deref(), Some("plan"));
    }

    #[test]
    fn session_bridge_does_not_trust_webview_model_override() {
        let ctx: SessionCtx = serde_json::from_value(serde_json::json!({
            "sessionID": "ses_123",
            "directory": "/tmp/project",
            "model": {
                "providerID": "openai",
                "modelID": "gpt-5"
            }
        }))
        .unwrap();
        let cfg = session_bridge_config(
            &ctx,
            ServerReadyData {
                url: "http://127.0.0.1:4096".into(),
                password: None,
            },
            "not-a-model-ref",
        );
        assert_eq!(cfg.model, None);
    }
}
