//! Voice settings — the single source of truth for the native Tauri voice layer.
//!
//! Persisted in the Tauri store (`emberharmony.settings.dat`, the same store the
//! rest of the desktop uses) under the `voice` key, so it is read **natively** by
//! the Rust voice loop (`crate::voice`) via [`load`] and edited by the webview
//! through the [`voice_settings_get`] / [`voice_settings_set`] commands — a
//! proper Tauri command binding, no sidecar round-trip and no `LFM_*` env vars.
//!
//! Secrets (LiveKit API key/secret, provider keys) do NOT live here — they stay
//! in the secure credentials store and are referenced by id.

use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

/// Same store file the rest of the desktop settings use (see `lib.rs`).
const SETTINGS_STORE: &str = "emberharmony.settings.dat";
const VOICE_KEY: &str = "voice";

/// Which voice provider is active — two providers behind one surface.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceProvider {
    /// Voice disabled.
    #[default]
    Off,
    /// Local LFM2.5-Audio model, fully native in this process.
    Lfm2,
    /// LiveKit room + the EmberHarmony session as the brain (via the sidecar).
    Livekit,
}

/// LiveKit provider config — non-secret only; the API key/secret live in the
/// credentials store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DelegateSettings {
    pub enabled: bool,
    pub target: Option<String>,
}

/// Local LFM2.5-Audio provider config — replaces the old `LFM_*` env vars.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Lfm2Settings {
    /// Path to the model directory (GGUF + tokenizer).
    pub model_dir: Option<String>,
    pub device: Lfm2Device,
    /// Energy-VAD threshold (mic_chat default 0.012).
    pub vad_threshold: f32,
    /// Max tokens per turn (mic_chat default 512).
    pub max_tokens: u32,
    /// Optional model-file override.
    pub model: Option<String>,
    /// Optional fixed seed for reproducible generation.
    pub seed: Option<u64>,
    pub delegate: DelegateSettings,
}

/// Default local-model directory: `<config>/emberharmony/models`, the folder the
/// Node global bootstrap (`global/index.ts`, via xdg-basedir) creates next to the
/// config dir's `node_modules`. Must resolve to the *same* path on every OS as
/// xdg-basedir does, so the config base is `XDG_CONFIG_HOME` else `<home>/.config`,
/// where `<home>` matches Node's `os.homedir()`: `HOME` on POSIX, `USERPROFILE` on
/// Windows. Returned absolute so the native loop uses it directly; `None` only if
/// none of those env vars are set.
fn default_model_dir() -> Option<String> {
    let home_config = || {
        std::env::var_os("HOME")
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()))
            .map(|h| std::path::PathBuf::from(h).join(".config"))
    };
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(home_config)?;
    Some(
        base.join("emberharmony")
            .join("models")
            .to_string_lossy()
            .into_owned(),
    )
}

impl Default for Lfm2Settings {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            device: Lfm2Device::default(),
            vad_threshold: 0.012,
            max_tokens: 512,
            model: None,
            seed: None,
            delegate: DelegateSettings::default(),
        }
    }
}

/// The whole voice settings object. `#[serde(default)]` makes deserialization
/// tolerant of missing fields, so adding settings later never breaks an older
/// persisted store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VoiceSettings {
    pub provider: VoiceProvider,
    pub livekit: LiveKitSettings,
    pub lfm2: Lfm2Settings,
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
        Some(value) => {
            serde_json::from_value(value).map_err(|e| format!("Failed to parse voice settings: {}", e))
        }
        None => Ok(VoiceSettings::default()),
    }
}

/// Persist the whole voice settings object.
#[tauri::command]
pub fn voice_settings_set(app: AppHandle, settings: VoiceSettings) -> Result<(), String> {
    let store = app
        .store(SETTINGS_STORE)
        .map_err(|e| format!("Failed to open settings store: {}", e))?;
    let value = serde_json::to_value(&settings)
        .map_err(|e| format!("Failed to serialize voice settings: {}", e))?;
    store.set(VOICE_KEY, value);
    store
        .save()
        .map_err(|e| format!("Failed to save settings: {}", e))?;
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
        assert_eq!(json["lfm2"]["maxTokens"], 512);
        assert_eq!(json["lfm2"]["device"], "cpu");
        assert!(json["lfm2"]["vadThreshold"].is_number()); // exact-float compare is brittle
        let back: VoiceSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back.provider, VoiceProvider::Off);
        assert_eq!(back.lfm2.vad_threshold, 0.012); // f32 round-trips losslessly
    }

    #[test]
    fn partial_json_fills_defaults() {
        // older/partial store missing most fields must not fail
        let json = serde_json::json!({ "provider": "lfm2", "lfm2": { "device": "metal" } });
        let v: VoiceSettings = serde_json::from_value(json).unwrap();
        assert_eq!(v.provider, VoiceProvider::Lfm2);
        assert_eq!(v.lfm2.device, Lfm2Device::Metal);
        assert_eq!(v.lfm2.vad_threshold, 0.012); // filled from Default
        assert_eq!(v.lfm2.max_tokens, 512);
    }

    #[test]
    fn default_model_dir_is_under_emberharmony_models() {
        // Every branch (XDG_CONFIG_HOME / HOME / USERPROFILE) ends the same way; the
        // path is absolute and points at the config-dir models folder. `Path::ends_with`
        // is component-wise, so this holds under Windows backslash separators too.
        let dir = Lfm2Settings::default()
            .model_dir
            .expect("HOME/USERPROFILE/XDG set in test env");
        let p = std::path::Path::new(&dir);
        assert!(p.ends_with("emberharmony/models"), "got {dir}");
        assert!(p.is_absolute(), "got {dir}");
    }

    #[test]
    fn provider_and_device_serialize_lowercase() {
        assert_eq!(serde_json::to_value(VoiceProvider::Livekit).unwrap(), "livekit");
        assert_eq!(serde_json::to_value(Lfm2Device::Metal).unwrap(), "metal");
    }
}
