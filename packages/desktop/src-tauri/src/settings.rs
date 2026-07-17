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
pub const DEFAULT_MOSHI_MODEL: &str = "kyutai/moshiko-candle-bf16";

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

/// Compute device for the local voice model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lfm2Device {
    Cpu,
    Metal,
}

impl Default for Lfm2Device {
    fn default() -> Self {
        // CPU on every platform, macOS included. Measured A/B on the two-turn speaker
        // e2e (same clip, same stack, 2026-07-09): CPU 24k underrun samples / 1508 ms
        // mean pause→first-audio vs Metal 167k underruns / 2043 ms — GPU dispatch
        // jitter lands directly in the audio at batch-of-one real-time decode. Metal
        // remains a user-selectable choice for prefill-heavy/offline use.
        Self::Cpu
    }
}

/// Native local engine loop. LFM2-Audio is turn/interleaved; Moshi is frame-realtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LocalVoiceEngine {
    Lfm2Interleaved,
    MoshiRealtime,
}

impl Default for LocalVoiceEngine {
    fn default() -> Self {
        Self::Lfm2Interleaved
    }
}

/// Where a `DELEGATE: <task>` line from the local model routes the hard work
/// (e.g. a GLM model id, or the EmberHarmony session).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DelegateSettings {
    pub enabled: bool,
    pub target: Option<String>,
}

/// One turn mode's decoding regime. The vendor demo (`audio-model.js`) runs a
/// DIFFERENT regime per mode — ASR greedy/100, TTS text 0.7 + audio 0.8/top-64,
/// interleaved text 1.0 + audio 1.0/top-4 — so each mode gets its own group.
/// Convention throughout: `0` = off (temperature 0 = greedy, top-k 0 = no cutoff).
///
/// Deliberately NOT `Deserialize`/`Default`: three modes have three different
/// defaults, so a partial stored group must fill from ITS OWN mode's default.
/// [`Lfm2Settings`] performs that merge while also migrating the legacy top-level
/// `maxTokens` field into the interleaved group.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Lfm2ModeSampling {
    /// Text sampling temperature. `0` = greedy decoding — a repetition machine
    /// at 1.2B in open conversation, but exactly right for ASR.
    pub text_temperature: f64,
    /// Text top-k cutoff. `0` = no cutoff (full multinomial over the vocabulary).
    pub text_top_k: u32,
    /// Audio sampling temperature. `0` = greedy — degenerate for the
    /// Depthformer (unintelligible speech); the model is trained for sampled audio.
    pub audio_temperature: f64,
    /// Audio top-k cutoff. `0` = no cutoff.
    pub audio_top_k: u32,
    /// Max tokens per turn. Interleaved steps: every audio frame costs one.
    pub max_tokens: u32,
}

impl Lfm2ModeSampling {
    /// ASR (`audio-model.js`): greedy text, no audio out, 100-token budget.
    fn asr_default() -> Self {
        Self {
            text_temperature: 0.0,
            text_top_k: 0,
            audio_temperature: 0.0,
            audio_top_k: 0,
            max_tokens: 100,
        }
    }

    /// TTS (`audio-model.js`): text 0.7, audio 0.8/top-64, 1024-token budget.
    fn tts_default() -> Self {
        Self {
            text_temperature: 0.7,
            text_top_k: 0,
            audio_temperature: 0.8,
            audio_top_k: 64,
            max_tokens: 1024,
        }
    }

    /// Interleaved conversation — the live path. Text/audio sampling is the
    /// demo regime (text 1.0 full multinomial, audio 1.0/top-4); the budget is
    /// OUR raised 8192 ceiling (demo ships 1024 ≈ 1 min; the model's 32,768
    /// context is the real limit and hitting the cap is LOUD, never silent).
    fn interleaved_default() -> Self {
        Self {
            text_temperature: 1.0,
            text_top_k: 0,
            audio_temperature: 1.0,
            audio_top_k: 4,
            max_tokens: 8192,
        }
    }
}

/// A stored group as written (every field optional). Merging onto the mode's
/// own default is what keeps a partial `tts` group a TTS regime.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Lfm2ModeSamplingPatch {
    text_temperature: Option<f64>,
    text_top_k: Option<u32>,
    audio_temperature: Option<f64>,
    audio_top_k: Option<u32>,
    max_tokens: Option<u32>,
}

impl Lfm2ModeSampling {
    fn merged(base: Self, patch: Lfm2ModeSamplingPatch) -> Self {
        Self {
            text_temperature: patch.text_temperature.unwrap_or(base.text_temperature),
            text_top_k: patch.text_top_k.unwrap_or(base.text_top_k),
            audio_temperature: patch.audio_temperature.unwrap_or(base.audio_temperature),
            audio_top_k: patch.audio_top_k.unwrap_or(base.audio_top_k),
            max_tokens: patch.max_tokens.unwrap_or(base.max_tokens),
        }
    }
}

fn de_asr<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Lfm2ModeSampling, D::Error> {
    Ok(Lfm2ModeSampling::merged(
        Lfm2ModeSampling::asr_default(),
        Lfm2ModeSamplingPatch::deserialize(d)?,
    ))
}

fn de_tts<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Lfm2ModeSampling, D::Error> {
    Ok(Lfm2ModeSampling::merged(
        Lfm2ModeSampling::tts_default(),
        Lfm2ModeSamplingPatch::deserialize(d)?,
    ))
}

fn de_interleaved<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Lfm2ModeSampling, D::Error> {
    Ok(Lfm2ModeSampling::merged(
        Lfm2ModeSampling::interleaved_default(),
        Lfm2ModeSamplingPatch::deserialize(d)?,
    ))
}

/// Local voice provider config — replaces the old `LFM_*` env vars.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Lfm2Settings {
    pub engine: LocalVoiceEngine,
    /// Optional local snapshot directory containing `config.json`, weights, and tokenizer files.
    /// Leave unset to resolve `model` through Hugging Face's cache/download flow.
    pub model_dir: Option<String>,
    /// Optional Moshi realtime snapshot directory. Kept separate from LFM2 so the
    /// selected engine controls the run path instead of guessing from one folder.
    pub moshi_model_dir: Option<String>,
    pub device: Lfm2Device,
    /// Energy-VAD threshold (mic_chat default 0.012).
    pub vad_threshold: f32,
    /// Timestamped native voice call-graph diagnostics. Persisted and explicit;
    /// never inferred from the desktop process environment.
    pub trace: bool,
    /// Per-mode decoding regimes (ASR / TTS / interleaved conversation).
    /// Each field fills absent AND partial stored groups from its own mode's
    /// default — see the `Lfm2ModeSampling` doc for why there is no shared one.
    pub asr: Lfm2ModeSampling,
    pub tts: Lfm2ModeSampling,
    pub interleaved: Lfm2ModeSampling,
    /// Hugging Face model id used by the cache/download resolver.
    pub model: Option<String>,
    /// Optional fixed seed for reproducible generation.
    pub seed: Option<u64>,
    /// Git revision (branch/tag/commit) of the HF repo to DOWNLOAD. Download-source
    /// only; ignored once `model_dir` points at a local snapshot. `None` = default branch.
    pub revision: Option<String>,
    /// Hugging Face repo/revision used to download the Moshi realtime snapshot.
    pub moshi_model: Option<String>,
    pub moshi_revision: Option<String>,
    pub delegate: DelegateSettings,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Lfm2SettingsSerde {
    engine: LocalVoiceEngine,
    model_dir: Option<String>,
    moshi_model_dir: Option<String>,
    device: Lfm2Device,
    vad_threshold: f32,
    trace: bool,
    #[serde(deserialize_with = "de_asr", default = "Lfm2ModeSampling::asr_default")]
    asr: Lfm2ModeSampling,
    #[serde(deserialize_with = "de_tts", default = "Lfm2ModeSampling::tts_default")]
    tts: Lfm2ModeSampling,
    #[serde(
        deserialize_with = "de_interleaved",
        default = "Lfm2ModeSampling::interleaved_default"
    )]
    interleaved: Lfm2ModeSampling,
    model: Option<String>,
    seed: Option<u64>,
    revision: Option<String>,
    moshi_model: Option<String>,
    moshi_revision: Option<String>,
    delegate: DelegateSettings,
}

impl Default for Lfm2SettingsSerde {
    fn default() -> Self {
        let value = Lfm2Settings::default();
        Self {
            engine: value.engine,
            model_dir: value.model_dir,
            moshi_model_dir: value.moshi_model_dir,
            device: value.device,
            vad_threshold: value.vad_threshold,
            trace: value.trace,
            asr: value.asr,
            tts: value.tts,
            interleaved: value.interleaved,
            model: value.model,
            seed: value.seed,
            revision: value.revision,
            moshi_model: value.moshi_model,
            moshi_revision: value.moshi_revision,
            delegate: value.delegate,
        }
    }
}

impl<'de> Deserialize<'de> for Lfm2Settings {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let mut value = serde_json::Value::deserialize(deserializer)?;
        if let Some(object) = value.as_object_mut() {
            if !object.contains_key("interleaved") {
                if let Some(max_tokens) = object.remove("maxTokens") {
                    object.insert(
                        "interleaved".into(),
                        serde_json::json!({ "maxTokens": max_tokens }),
                    );
                }
            } else {
                object.remove("maxTokens");
            }
        }
        let value: Lfm2SettingsSerde = serde_json::from_value(value).map_err(D::Error::custom)?;
        Ok(Self {
            engine: value.engine,
            model_dir: value.model_dir,
            moshi_model_dir: value.moshi_model_dir,
            device: value.device,
            vad_threshold: value.vad_threshold,
            trace: value.trace,
            asr: value.asr,
            tts: value.tts,
            interleaved: value.interleaved,
            model: value.model,
            seed: value.seed,
            revision: value.revision,
            moshi_model: value.moshi_model,
            moshi_revision: value.moshi_revision,
            delegate: value.delegate,
        })
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()))
        .map(PathBuf::from)
}

fn hf_snapshot_dir(dir: PathBuf) -> PathBuf {
    let snapshots = dir.join("snapshots");
    if !snapshots.is_dir() {
        return dir;
    }
    let refs = dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&refs) {
        let candidate = snapshots.join(rev.trim());
        if candidate.is_dir() {
            return candidate;
        }
    }
    let Ok(mut entries) = std::fs::read_dir(&snapshots).map(|entries| {
        entries
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|t| t.is_dir()))
            .map(|entry| entry.path())
            .collect::<Vec<_>>()
    }) else {
        return dir;
    };
    entries.sort();
    if entries.len() == 1 {
        return entries.remove(0);
    }
    dir
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
        .map(hf_snapshot_dir)
}

pub fn moshi_model_dir(settings: &Lfm2Settings) -> Option<PathBuf> {
    settings
        .moshi_model_dir
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(expand_user_path)
        .map(hf_snapshot_dir)
}

/// The active LFM2-Audio directory iff it contains a local model snapshot. No repo-id
/// fallback, no default — this is the fail-hard run-path resolver. `model`/`revision`
/// are the download *source*; this is what the runtime actually loads.
pub fn lfm2_active_model_dir(settings: &Lfm2Settings) -> Option<PathBuf> {
    lfm2_model_dir(settings).filter(|dir| dir.join("config.json").is_file())
}

fn decode_voice_settings(value: serde_json::Value) -> Result<VoiceSettings, serde_json::Error> {
    serde_json::from_value(value)
}

impl Default for Lfm2Settings {
    fn default() -> Self {
        Self {
            engine: LocalVoiceEngine::default(),
            model_dir: None,
            moshi_model_dir: None,
            device: Lfm2Device::default(),
            vad_threshold: 0.012,
            trace: false,
            asr: Lfm2ModeSampling::asr_default(),
            tts: Lfm2ModeSampling::tts_default(),
            interleaved: Lfm2ModeSampling::interleaved_default(),
            model: Some(DEFAULT_LFM2_MODEL.to_string()),
            seed: None,
            revision: None,
            moshi_model: Some(DEFAULT_MOSHI_MODEL.to_string()),
            moshi_revision: None,
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
        .and_then(|value| decode_voice_settings(value).ok())
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
        Some(value) => decode_voice_settings(value)
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
            settings: decode_voice_settings(value)
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
        assert_eq!(json["lfm2"]["engine"], "lfm2Interleaved");
        assert_eq!(json["lfm2"]["trace"], false);
        // Per-mode regimes, camelCase groups. Interleaved = the live path.
        assert_eq!(json["lfm2"]["interleaved"]["maxTokens"], 8192);
        assert_eq!(json["lfm2"]["interleaved"]["textTemperature"], 1.0);
        assert_eq!(json["lfm2"]["interleaved"]["textTopK"], 0);
        assert_eq!(json["lfm2"]["interleaved"]["audioTemperature"], 1.0);
        assert_eq!(json["lfm2"]["interleaved"]["audioTopK"], 4);
        // ASR is greedy/short; TTS is the demo's 0.7 / 0.8+top-64 regime.
        assert_eq!(json["lfm2"]["asr"]["textTemperature"], 0.0);
        assert_eq!(json["lfm2"]["asr"]["maxTokens"], 100);
        assert_eq!(json["lfm2"]["tts"]["textTemperature"], 0.7);
        assert_eq!(json["lfm2"]["tts"]["audioTemperature"], 0.8);
        assert_eq!(json["lfm2"]["tts"]["audioTopK"], 64);
        assert_eq!(json["lfm2"]["tts"]["maxTokens"], 1024);
        // CPU on every platform since the measured flip (engine work, 2026-07-09):
        // the lane-team engine leads Metal on both latency and underruns.
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
        let v = decode_voice_settings(json).unwrap();
        assert_eq!(v.provider, VoiceProvider::Lfm2);
        assert_eq!(v.last_provider, Some(VoiceProvider::Lfm2));
        assert_eq!(v.lfm2.device, Lfm2Device::Metal);
        assert_eq!(v.lfm2.vad_threshold, 0.012); // filled from Default
        assert!(!v.lfm2.trace); // diagnostics stay off unless the stored settings enable them
        // Absent mode groups take the correct PER-MODE defaults (older
        // stores predate the groups entirely).
        assert_eq!(v.lfm2.interleaved.max_tokens, 8192);
        assert_eq!(v.lfm2.interleaved.text_temperature, 1.0);
        assert_eq!(v.lfm2.interleaved.audio_top_k, 4);
        assert_eq!(v.lfm2.asr.text_temperature, 0.0);
        assert_eq!(v.lfm2.asr.max_tokens, 100);
        assert_eq!(v.lfm2.tts.audio_top_k, 64);
        assert_eq!(v.lfm2.engine, LocalVoiceEngine::Lfm2Interleaved);
        assert_eq!(v.lfm2.moshi_model.as_deref(), Some(DEFAULT_MOSHI_MODEL));
    }

    #[test]
    fn legacy_max_tokens_migrates_to_interleaved_budget() {
        let json = serde_json::json!({
            "lfm2": {
                "maxTokens": 1536
            }
        });
        let value = decode_voice_settings(json).unwrap();
        assert_eq!(value.lfm2.interleaved.max_tokens, 1536);
        assert_eq!(value.lfm2.interleaved.audio_top_k, 4);

        let stored = serde_json::to_value(value).unwrap();
        assert_eq!(stored["lfm2"]["interleaved"]["maxTokens"], 1536);
        assert!(stored["lfm2"].get("maxTokens").is_none());
    }

    #[test]
    fn interleaved_group_wins_over_legacy_max_tokens() {
        let json = serde_json::json!({
            "lfm2": {
                "maxTokens": 1536,
                "interleaved": { "maxTokens": 4096 }
            }
        });
        let value = decode_voice_settings(json).unwrap();
        assert_eq!(value.lfm2.interleaved.max_tokens, 4096);
    }

    #[test]
    fn partial_mode_group_fills_from_its_own_modes_default() {
        // Sparse stored groups (hand edits today; EVERY store the day a field
        // is added to Lfm2ModeSampling): explicit keys win, and the rest fill
        // from that mode's OWN default — a sparse tts group must stay a TTS
        // regime, never pick up interleaved values.
        let json = serde_json::json!({
            "lfm2": {
                "interleaved": { "textTopK": 64, "textTemperature": 0.7 },
                "tts": { "maxTokens": 512 },
                "asr": { "maxTokens": 50 },
            }
        });
        let v = decode_voice_settings(json).unwrap();
        assert_eq!(v.lfm2.interleaved.text_top_k, 64);
        assert_eq!(v.lfm2.interleaved.text_temperature, 0.7);
        assert_eq!(v.lfm2.interleaved.max_tokens, 8192);
        assert_eq!(v.lfm2.interleaved.audio_top_k, 4);
        // The wrong-fill case: tts keeps its 0.7/0.8/top-64 shape.
        assert_eq!(v.lfm2.tts.max_tokens, 512);
        assert_eq!(v.lfm2.tts.text_temperature, 0.7);
        assert_eq!(v.lfm2.tts.audio_temperature, 0.8);
        assert_eq!(v.lfm2.tts.audio_top_k, 64);
        // ASR stays greedy — a sampled-text fill would be a hallucinating
        // transcriber.
        assert_eq!(v.lfm2.asr.max_tokens, 50);
        assert_eq!(v.lfm2.asr.text_temperature, 0.0);
        assert_eq!(v.lfm2.asr.text_top_k, 0);
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
    fn model_dirs_accept_huggingface_cache_repo_roots() {
        let root =
            std::env::temp_dir().join(format!("emberharmony-hf-root-{}", std::process::id()));
        let snap = root.join("snapshots").join("abc123");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs").join("main"), "abc123\n").unwrap();
        std::fs::write(snap.join("config.json"), "{}").unwrap();

        let s = Lfm2Settings {
            model_dir: Some(root.to_string_lossy().into_owned()),
            moshi_model_dir: Some(root.to_string_lossy().into_owned()),
            ..Default::default()
        };
        assert_eq!(lfm2_model_dir(&s), Some(snap.clone()));
        assert_eq!(lfm2_active_model_dir(&s), Some(snap.clone()));
        assert_eq!(moshi_model_dir(&s), Some(snap));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_lfm2_model_is_a_repo_id_not_a_directory() {
        let s = Lfm2Settings::default();
        assert_eq!(s.model.as_deref(), Some(DEFAULT_LFM2_MODEL));
        assert_eq!(s.model_dir, None);
        assert_eq!(s.moshi_model.as_deref(), Some(DEFAULT_MOSHI_MODEL));
        assert_eq!(s.moshi_model_dir, None);
        assert_eq!(s.engine, LocalVoiceEngine::Lfm2Interleaved);
    }

    #[test]
    fn stored_voice_settings_without_engine_use_native_lfm2_default() {
        let json = serde_json::json!({
            "provider": "lfm2",
            "lastProvider": "lfm2",
            "lfm2": {
                "model": DEFAULT_LFM2_MODEL,
                "modelDir": "/tmp/old-lfm2"
            }
        });
        let v = decode_voice_settings(json).unwrap();
        assert_eq!(v.lfm2.engine, LocalVoiceEngine::Lfm2Interleaved);
    }

    #[test]
    fn provider_and_device_serialize_lowercase() {
        assert_eq!(
            serde_json::to_value(VoiceProvider::Livekit).unwrap(),
            "livekit"
        );
        assert_eq!(serde_json::to_value(Lfm2Device::Metal).unwrap(), "metal");
        assert_eq!(
            serde_json::to_value(LocalVoiceEngine::MoshiRealtime).unwrap(),
            "moshiRealtime"
        );
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
    fn lfm2_active_model_dir_requires_config_json() {
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
        // LFM2-style dir with config.json → Some.
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert_eq!(lfm2_active_model_dir(&s), Some(dir.clone()));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn lfm2_active_model_dir_rejects_legacy_moshi_snapshot_files() {
        let dir =
            std::env::temp_dir().join(format!("emberharmony-moshi-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), "").unwrap();
        std::fs::write(dir.join("tokenizer-e351c8d8-checkpoint125.safetensors"), "").unwrap();
        std::fs::write(dir.join("tokenizer_spm_32k_3.model"), "").unwrap();
        let s = Lfm2Settings {
            model_dir: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        };
        assert_eq!(lfm2_active_model_dir(&s), None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn moshi_model_dir_is_separate_from_lfm2_model_dir() {
        let dir = std::env::temp_dir().join(format!(
            "emberharmony-moshi-dir-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = Lfm2Settings {
            moshi_model_dir: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        };
        assert_eq!(moshi_model_dir(&s), Some(dir.clone()));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
