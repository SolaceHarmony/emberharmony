//! Voice settings — the single source of truth for the native Tauri voice layer.
//!
//! Persisted in the Tauri store (`emberharmony.settings.dat`, the same store the
//! rest of the desktop uses) under the `voice` key, so it is read **natively** by
//! the Rust voice loop (`crate::voice`) via [`load`] and edited by the webview
//! through the [`voice_settings_get`] / [`voice_settings_set`] commands — a
//! proper Tauri command binding, no sidecar round-trip and no `LFM_*` env vars.
//!
//! Secrets do NOT live here. The Hugging Face token is kept in the OS keychain
//! (see `voice::model`), and the desktop LiveKit API key/secret are kept there
//! too (see `voice::livekit`). This file holds only non-secret config.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, State};
use tauri_plugin_store::StoreExt;

use crate::voice::runtime::VoiceRuntime;

/// Same store file the rest of the desktop settings use (see `lib.rs`).
const SETTINGS_STORE: &str = "emberharmony.settings.dat";
const VOICE_KEY: &str = "voice";
pub const DEFAULT_LFM2_MODEL: &str = "LiquidAI/LFM2.5-Audio-1.5B";

fn default_last_provider() -> Option<VoiceProvider> {
    Some(VoiceProvider::Lfm2)
}

/// Which voice provider is active — two providers behind one surface.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceProvider {
    /// Voice disabled.
    #[default]
    Off,
    /// Local LFM2.5-Audio model, fully native in this process.
    Lfm2,
    /// LiveKit room + the EmberHarmony session as the brain.
    Livekit,
}

/// LiveKit provider config — non-secret only; the API key/secret live in the
/// OS keychain via `voice::livekit`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LiveKitSettings {
    /// e.g. `wss://<project>.livekit.cloud`
    pub url: Option<String>,
    /// model strings, e.g. `deepgram/nova-3:multi`
    pub stt: Option<String>,
    pub tts: Option<String>,
    /// small model that routes plan/build, e.g. `openai/gpt-5.4-nano`
    pub intent: Option<String>,
}

/// Compute device for the local LFM2 model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lfm2Device {
    #[default]
    Cpu,
    Metal,
}

/// Where a `DELEGATE: <task>` line from the local model routes the hard work
/// (e.g. a GLM model id, or the EmberHarmony session).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DelegateSettings {
    pub enabled: bool,
    pub target: Option<String>,
}

/// Local LFM2.5-Audio provider config — replaces the old `LFM_*` env vars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Lfm2Settings {
    /// Optional local snapshot directory containing `config.json`, weights, and tokenizer files.
    /// Leave unset to resolve `model` through Hugging Face's cache/download flow.
    pub model_dir: Option<String>,
    pub device: Lfm2Device,
    /// Energy-VAD threshold (mic_chat default 0.012).
    pub vad_threshold: f32,
    /// Max tokens per turn (mic_chat default 512).
    pub max_tokens: u32,
    /// Hugging Face model id used by the cache/download resolver.
    pub model: Option<String>,
    /// Optional fixed seed for reproducible generation.
    pub seed: Option<u64>,
    /// Git revision (branch/tag/commit) of the HF repo to DOWNLOAD. Download-source
    /// only; ignored once `model_dir` points at a local snapshot. `None` = default branch.
    pub revision: Option<String>,
    pub delegate: DelegateSettings,
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()))
        .map(PathBuf::from)
}

pub fn expand_user_path(value: &str) -> PathBuf {
    let value = value.trim();
    if value == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from(value));
    }
    if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("~\\"))
    {
        return home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| PathBuf::from(value));
    }
    PathBuf::from(value)
}

pub fn lfm2_model_dir(settings: &Lfm2Settings) -> Option<PathBuf> {
    settings
        .model_dir
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(expand_user_path)
}

/// The active LOCAL model directory iff it exists and contains `config.json`. No repo-id
/// fallback, no default — this is the fail-hard run-path resolver: with no local model the
/// caller reports not-ready instead of silently downloading. `model`/`revision` are the
/// download *source*; this is what the runtime actually loads.
pub fn lfm2_active_model_dir(settings: &Lfm2Settings) -> Option<PathBuf> {
    lfm2_model_dir(settings).filter(|dir| dir.join("config.json").is_file())
}

impl Default for Lfm2Settings {
    fn default() -> Self {
        Self {
            model_dir: None,
            device: Lfm2Device::default(),
            vad_threshold: 0.012,
            max_tokens: 512,
            model: Some(DEFAULT_LFM2_MODEL.to_string()),
            seed: None,
            revision: None,
            delegate: DelegateSettings::default(),
        }
    }
}

/// The whole voice settings object. `#[serde(default)]` makes deserialization
/// tolerant of missing fields, so adding settings later never breaks an older
/// persisted store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VoiceSettings {
    pub provider: VoiceProvider,
    /// Last non-off provider selected in the UI. This keeps "voice off" from
    /// erasing whether the user meant to re-enable local LFM2 or LiveKit later.
    #[serde(default = "default_last_provider")]
    pub last_provider: Option<VoiceProvider>,
    pub livekit: LiveKitSettings,
    pub lfm2: Lfm2Settings,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        Self {
            provider: VoiceProvider::Off,
            last_provider: Some(VoiceProvider::Lfm2),
            livekit: LiveKitSettings::default(),
            lfm2: Lfm2Settings::default(),
        }
    }
}

/// Settings plus whether the `voice` key was actually present in the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSettingsState {
    pub settings: VoiceSettings,
    pub stored: bool,
}

/// Lenient in-process read for the native voice loop: returns defaults on any
/// error (missing store / unset / parse failure) so the loop never fails to
/// start over config.
pub fn load(app: &AppHandle) -> VoiceSettings {
    app.store(SETTINGS_STORE)
        .ok()
        .and_then(|store| store.get(VOICE_KEY))
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

/// Read the persisted voice settings (defaults if unset). Unlike [`load`], this
/// surfaces a parse error to the UI rather than silently defaulting.
#[tauri::command]
pub fn voice_settings_get(app: AppHandle) -> Result<VoiceSettings, String> {
    let store = app
        .store(SETTINGS_STORE)
        .map_err(|e| format!("Failed to open settings store: {}", e))?;
    match store.get(VOICE_KEY) {
        Some(value) => serde_json::from_value(value)
            .map_err(|e| format!("Failed to parse voice settings: {}", e)),
        None => Ok(VoiceSettings::default()),
    }
}

/// Read persisted voice settings and report whether they were explicitly stored.
#[tauri::command]
pub fn voice_settings_state(app: AppHandle) -> Result<VoiceSettingsState, String> {
    let store = app
        .store(SETTINGS_STORE)
        .map_err(|e| format!("Failed to open settings store: {}", e))?;
    match store.get(VOICE_KEY) {
        Some(value) => Ok(VoiceSettingsState {
            settings: serde_json::from_value(value)
                .map_err(|e| format!("Failed to parse voice settings: {}", e))?,
            stored: true,
        }),
        None => Ok(VoiceSettingsState {
            settings: VoiceSettings::default(),
            stored: false,
        }),
    }
}

/// Persist the whole voice settings object.
#[tauri::command]
pub async fn voice_settings_set(
    app: AppHandle,
    runtime: State<'_, VoiceRuntime>,
    settings: VoiceSettings,
) -> Result<(), String> {
    let value = serde_json::to_value(&settings)
        .map_err(|e| format!("Failed to serialize voice settings: {}", e))?;
    runtime.apply_settings(settings.clone()).await?;
    {
        let store = app
            .store(SETTINGS_STORE)
            .map_err(|e| format!("Failed to open settings store: {}", e))?;
        store.set(VOICE_KEY, value);
        store
            .save()
            .map_err(|e| format!("Failed to save settings: {}", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_camel_case() {
        let v = VoiceSettings::default();
        let json = serde_json::to_value(&v).unwrap();
        assert_eq!(json["provider"], "off");
        assert_eq!(json["lastProvider"], "lfm2");
        assert_eq!(json["lfm2"]["maxTokens"], 512);
        assert_eq!(json["lfm2"]["device"], "cpu");
        assert!(json["lfm2"]["vadThreshold"].is_number()); // exact-float compare is brittle
        let back: VoiceSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back.provider, VoiceProvider::Off);
        assert_eq!(back.last_provider, Some(VoiceProvider::Lfm2));
        assert_eq!(back.lfm2.vad_threshold, 0.012); // f32 round-trips losslessly
    }

    #[test]
    fn partial_json_fills_defaults() {
        // older/partial store missing most fields must not fail
        let json = serde_json::json!({ "provider": "lfm2", "lfm2": { "device": "metal" } });
        let v: VoiceSettings = serde_json::from_value(json).unwrap();
        assert_eq!(v.provider, VoiceProvider::Lfm2);
        assert_eq!(v.last_provider, Some(VoiceProvider::Lfm2));
        assert_eq!(v.lfm2.device, Lfm2Device::Metal);
        assert_eq!(v.lfm2.vad_threshold, 0.012); // filled from Default
        assert_eq!(v.lfm2.max_tokens, 512);
    }

    #[test]
    fn expands_home_relative_model_dirs() {
        let path = expand_user_path("~/models/lfm2-audio");
        assert!(path.is_absolute() || std::env::var_os("HOME").is_none());
        assert!(
            path.ends_with("models/lfm2-audio"),
            "got {}",
            path.display()
        );
    }

    #[test]
    fn default_lfm2_model_is_a_repo_id_not_a_directory() {
        let s = Lfm2Settings::default();
        assert_eq!(s.model.as_deref(), Some(DEFAULT_LFM2_MODEL));
        assert_eq!(s.model_dir, None);
    }

    #[test]
    fn provider_and_device_serialize_lowercase() {
        assert_eq!(
            serde_json::to_value(VoiceProvider::Livekit).unwrap(),
            "livekit"
        );
        assert_eq!(serde_json::to_value(Lfm2Device::Metal).unwrap(), "metal");
    }

    #[test]
    fn revision_defaults_to_none_and_round_trips() {
        assert_eq!(Lfm2Settings::default().revision, None);
        let s = Lfm2Settings {
            revision: Some("refs/pr/3".into()),
            ..Default::default()
        };
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["revision"], "refs/pr/3");
        let back: Lfm2Settings = serde_json::from_value(json).unwrap();
        assert_eq!(back.revision.as_deref(), Some("refs/pr/3"));
    }

    #[test]
    fn active_model_dir_requires_config_json() {
        // No dir → None.
        let mut s = Lfm2Settings {
            model_dir: None,
            ..Default::default()
        };
        assert!(lfm2_active_model_dir(&s).is_none());
        // Dir without config.json → None (never silently active).
        let dir =
            std::env::temp_dir().join(format!("emberharmony-active-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        s.model_dir = Some(dir.to_string_lossy().into_owned());
        assert!(lfm2_active_model_dir(&s).is_none());
        // Dir with config.json → Some.
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert_eq!(lfm2_active_model_dir(&s), Some(dir.clone()));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
