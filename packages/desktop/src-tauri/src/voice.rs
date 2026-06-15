//! Voice module — native LiveKit audio transport.
//!
//! Uses the LiveKit Rust SDK to connect to voice rooms, capture microphone
//! audio, and play back the agent's speech. This replaces the JS worker's
//! WebRTC with cross-platform Rust audio.
//!
//! ## Architecture
//!
//! The voice module runs inside the Tauri app process. It:
//! - Connects to a LiveKit room using the token from `POST /voice/token`
//! - Captures audio from the local microphone
//! - Publishes the local audio track to the room
//! - Subscribes to the agent's audio track and plays it back
//! - Exposes state (connected, speaking, agent stage) via Tauri commands
//!
//! ## Usage
//!
//! Enable the `voice` feature flag to include the LiveKit dependencies.
//! The JS frontend calls Tauri commands to start/stop voice sessions.
//!
//! ## Next steps
//!
//! - Wire `Room::connect` from the `livekit` crate
//! - Add audio capture via `LocalAudioTrack::create`
//! - Handle `RoomEvent::TrackSubscribed` for agent audio playback
//! - Watch `ParticipantAttributesChanged` for workflow stage updates
//! - Bridge state changes to the JS frontend via Tauri events

use serde::Serialize;
use tauri::AppHandle;

/// Voice session state, exposed to the frontend via Tauri commands.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceState {
    /// Whether connected to a LiveKit room
    pub connected: bool,
    /// Current room name
    pub room: Option<String>,
    /// The agent's workflow stage (from participant attributes)
    pub agent_stage: Option<String>,
    /// Whether the agent is in plan or build mode
    pub agent_mode: Option<String>,
    /// Whether the local mic is muted
    pub mic_muted: bool,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            connected: false,
            room: None,
            agent_stage: None,
            agent_mode: None,
            mic_muted: false,
        }
    }
}

/// Connect to a LiveKit room and start the voice session.
///
/// Stub — the actual LiveKit connection will be implemented when the
/// `voice` feature is enabled and the `livekit` crate is wired in.
#[tauri::command]
pub async fn voice_connect(
    _app: AppHandle,
    url: String,
    _token: String,
) -> Result<VoiceState, String> {
    // TODO: Implement using the livekit crate:
    //
    //   let (room, mut events) = Room::connect(&url, &token)
    //       .await
    //       .map_err(|e| e.to_string())?;
    //
    //   // Publish local audio track (microphone)
    //   let local_audio = LocalAudioTrack::create(AudioSource::default());
    //   room.publish_track(local_audio).await.map_err(|e| e.to_string())?;
    //
    //   // Subscribe to agent's audio track
    //   // Spawn task to handle room events and update state
    //
    // The room events stream gives us:
    // - TrackSubscribed: agent started speaking
    // - TrackUnsubscribed: agent stopped speaking
    // - ParticipantAttributesChanged: workflow stage/mode updates

    println!("voice_connect (stub): url={url}");

    Ok(VoiceState {
        connected: true,
        room: Some("voice_room".into()),
        agent_stage: Some("gathering".into()),
        agent_mode: Some("plan".into()),
        mic_muted: false,
    })
}

/// Disconnect from the current voice room.
#[tauri::command]
pub async fn voice_disconnect(_app: AppHandle) -> Result<VoiceState, String> {
    // TODO: Disconnect from the LiveKit room
    println!("voice_disconnect (stub)");

    Ok(VoiceState::default())
}

/// Toggle the local microphone mute state.
#[tauri::command]
pub async fn voice_toggle_mute(_app: AppHandle) -> Result<bool, String> {
    // TODO: Toggle mute on the published local audio track
    println!("voice_toggle_mute (stub)");

    Ok(false)
}

/// Get the current voice session state.
#[tauri::command]
pub async fn voice_state(_app: AppHandle) -> Result<VoiceState, String> {
    // TODO: Return actual state from the active voice session
    Ok(VoiceState::default())
}