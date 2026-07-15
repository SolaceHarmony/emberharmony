//! Native LiveKit configuration and token minting for the desktop provider.
//!
//! The desktop runtime owns the LiveKit room. This module keeps the server URL in
//! the Tauri voice settings, keeps API credentials in the OS keychain, and mints
//! room tokens in-process so `voice_start` does not call the local server's
//! `/voice/token` endpoint.

use std::time::Duration;

use livekit_api::access_token::{AccessToken, VideoGrants};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::settings::{VoiceProvider, VoiceSettings};

use super::control::{LiveKitGrant, SessionCtx};
use super::runtime::VoiceRuntime;

const KEYRING_SERVICE: &str = "ai.ofharmony.emberharmony.voice";
const KEYRING_API_KEY_USER: &str = "livekit-api-key";
const KEYRING_API_SECRET_USER: &str = "livekit-api-secret";
const LIVEKIT_TOKEN_TTL: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveKitCredentialsStatus {
    pub stored: bool,
}

struct LiveKitSecret {
    url: String,
    api_key: String,
    api_secret: String,
}

fn entry(user: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYRING_SERVICE, user).map_err(|e| format!("keychain: {e}"))
}

fn secret_entry() -> Result<(keyring::Entry, keyring::Entry), String> {
    Ok((
        entry(KEYRING_API_KEY_USER)?,
        entry(KEYRING_API_SECRET_USER)?,
    ))
}

fn read_password(entry: &keyring::Entry) -> Result<Option<String>, String> {
    match entry.get_password() {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        Ok(_) | Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(format!("keychain: {error}")),
    }
}

fn delete_password(entry: &keyring::Entry) -> Result<(), String> {
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(format!("keychain: {error}")),
    }
}

fn credentials() -> Result<Option<(String, String)>, String> {
    let (key, secret) = secret_entry()?;
    let key = read_password(&key)?;
    let secret = read_password(&secret)?;
    Ok(match (key, secret) {
        (Some(key), Some(secret)) => Some((key, secret)),
        _ => None,
    })
}

fn config(settings: &VoiceSettings) -> Result<Option<LiveKitSecret>, String> {
    let url = settings
        .livekit
        .url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty());
    let Some(url) = url else {
        return Ok(None);
    };
    let Some((api_key, api_secret)) = credentials()? else {
        return Ok(None);
    };
    Ok(Some(LiveKitSecret {
        url: url.to_string(),
        api_key,
        api_secret,
    }))
}

pub fn configured(settings: &VoiceSettings) -> Result<bool, String> {
    Ok(config(settings)?.is_some())
}

pub(crate) async fn grant(
    settings: &VoiceSettings,
    ctx: &SessionCtx,
) -> Result<LiveKitGrant, String> {
    let config = config(settings)?.ok_or_else(|| {
        "LiveKit URL and credentials must be stored in the desktop voice settings.".to_string()
    })?;
    let room_name = format!("emberharmony_{}", ctx.session_id);
    let user_identity = format!("user_{}", ctx.session_id);
    let agent_identity = format!("agent_{}", ctx.session_id);
    let token = livekit_token(&config, &room_name, &user_identity, "EmberHarmony Desktop")?;
    let agent_token = livekit_token(
        &config,
        &room_name,
        &agent_identity,
        "EmberHarmony Native Voice Agent",
    )?;
    Ok(LiveKitGrant {
        token,
        agent_token,
        url: config.url,
        room_name,
        user_identity,
        agent_identity,
    })
}

fn livekit_token(
    config: &LiveKitSecret,
    room_name: &str,
    identity: &str,
    name: &str,
) -> Result<String, String> {
    AccessToken::with_api_key(&config.api_key, &config.api_secret)
        .with_ttl(LIVEKIT_TOKEN_TTL)
        .with_identity(identity)
        .with_name(name)
        .with_grants(VideoGrants {
            room: room_name.to_string(),
            room_join: true,
            can_publish: true,
            can_subscribe: true,
            can_publish_data: true,
            ..Default::default()
        })
        .to_jwt()
        .map_err(|e| format!("LiveKit token failed: {e}"))
}

#[tauri::command]
pub async fn voice_livekit_credentials_set(
    runtime: State<'_, VoiceRuntime>,
    api_key: String,
    api_secret: String,
) -> Result<(), String> {
    let key = api_key.trim();
    let secret = api_secret.trim();
    if key.is_empty() && secret.is_empty() {
        let (key_entry, secret_entry) = secret_entry()?;
        delete_password(&key_entry)?;
        delete_password(&secret_entry)?;
        return runtime.invalidate_provider(VoiceProvider::Livekit).await;
    }
    if key.is_empty() || secret.is_empty() {
        return Err("Enter both the LiveKit API key and API secret.".into());
    }
    let (key_entry, secret_entry) = secret_entry()?;
    key_entry
        .set_password(key)
        .map_err(|e| format!("keychain: {e}"))?;
    secret_entry
        .set_password(secret)
        .map_err(|e| format!("keychain: {e}"))?;
    runtime.invalidate_provider(VoiceProvider::Livekit).await
}

#[tauri::command]
pub async fn voice_livekit_credentials_status() -> Result<LiveKitCredentialsStatus, String> {
    Ok(LiveKitCredentialsStatus {
        stored: credentials()?.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{LiveKitSettings, VoiceProvider};

    #[test]
    fn missing_url_is_not_configured_without_reading_credentials() {
        let settings = VoiceSettings {
            provider: VoiceProvider::Livekit,
            livekit: LiveKitSettings::default(),
            ..Default::default()
        };
        assert!(!configured(&settings).unwrap());
    }
}
