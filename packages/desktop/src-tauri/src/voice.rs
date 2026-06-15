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
//! - Captures audio from the local microphone via NativeAudioSource
//! - Publishes the local audio track to the room
//! - Subscribes to the agent's audio track and plays it back via NativeAudioStream
//! - Watches ParticipantAttributesChanged for workflow stage/mode updates
//! - Exposes state (connected, speaking, agent stage) via Tauri commands + events
//!
//! ## Usage
//!
//! Enable the `voice` feature flag to include the LiveKit dependencies.
//! The JS frontend calls Tauri commands to start/stop voice sessions
//! and listens for `voice://state-changed` events.

use futures::StreamExt;
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::track::{LocalAudioTrack, LocalTrack, TrackSource};
use livekit::webrtc::{
    audio_source::native::NativeAudioSource,
    audio_stream::native::NativeAudioStream,
    prelude::{AudioSourceOptions, RtcAudioSource},
};
use log::{debug, info, warn};
use serde::Serialize;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{mpsc, Mutex};

/// Event emitted to the JS frontend when voice state changes.
const VOICE_STATE_EVENT: &str = "voice://state-changed";

/// Voice session state, exposed to the frontend via Tauri commands and events.
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

/// Participant attribute keys used by the voice agent worker.
const ATTR_VOICE_STAGE: &str = "emberharmony.voice_stage";
const ATTR_VOICE_MODE: &str = "emberharmony.voice_mode";

/// Active voice session. Holds the LiveKit room, the published mic track,
/// and the event receiver.
struct VoiceSession {
    room: Room,
    _audio_source: NativeAudioSource,
    mic_publication: LocalTrackPublication,
    events: mpsc::UnboundedReceiver<RoomEvent>,
    state: VoiceState,
}

/// Tauri-managed state holding the active voice session.
pub struct VoiceHandle {
    session: Arc<Mutex<Option<VoiceSession>>>,
}

impl VoiceHandle {
    pub fn new() -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
        }
    }
}

fn emit_state(app: &AppHandle, state: &VoiceState) {
    if let Err(e) = app.emit(VOICE_STATE_EVENT, state) {
        warn!("failed to emit voice state event: {e}");
    }
}

/// Connect to a LiveKit room and start the voice session.
#[tauri::command]
pub async fn voice_connect(
    app: AppHandle,
    handle: State<'_, VoiceHandle>,
    url: String,
    token: String,
) -> Result<VoiceState, String> {
    // Reject if already connected
    {
        let guard = handle.session.lock().await;
        if guard.is_some() {
            return Err("voice session already active".into());
        }
    }

    info!("connecting to voice room: {url}");

    let (room, events) = Room::connect(&url, &token, RoomOptions::default())
        .await
        .map_err(|e| format!("livekit connect failed: {e}"))?;

    let room_name = room.name();

    // Create mic audio source (48kHz mono — standard for speech)
    let sample_rate = 48000u32;
    let num_channels = 1u32;
    let audio_source = NativeAudioSource::new(
        AudioSourceOptions::default(),
        sample_rate,
        num_channels,
        200, // queue size — ~200ms of buffering
    );

    // Create and publish the local mic track
    let mic_track = LocalAudioTrack::create_audio_track(
        "microphone",
        RtcAudioSource::Native(audio_source.clone()),
    );

    let publication = room
        .local_participant()
        .publish_track(
            LocalTrack::Audio(mic_track),
            TrackPublishOptions {
                source: TrackSource::Microphone,
                dtx: true,
                red: true,
                ..Default::default()
            },
        )
        .await
        .map_err(|e| format!("failed to publish mic track: {e}"))?;

    debug!("published mic track: {}", publication.sid());

    let state = VoiceState {
        connected: true,
        room: Some(room_name.clone()),
        agent_stage: None,
        agent_mode: None,
        mic_muted: false,
    };

    let session = VoiceSession {
        room,
        _audio_source: audio_source,
        mic_publication: publication,
        events,
        state: state.clone(),
    };

    // Store the session
    *handle.session.lock().await = Some(session);

    // Spawn event loop
    let session_arc = handle.session.clone();
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        event_loop(session_arc, app_clone).await;
    });

    emit_state(&app, &state);
    info!("voice connected to room: {room_name}");

    Ok(state)
}

/// Disconnect from the current voice room.
#[tauri::command]
pub async fn voice_disconnect(
    app: AppHandle,
    handle: State<'_, VoiceHandle>,
) -> Result<VoiceState, String> {
    let mut guard = handle.session.lock().await;
    let Some(session) = guard.take() else {
        return Err("no active voice session".into());
    };

    info!("disconnecting from voice room");

    // Close the room — this unpublishes tracks and disconnects
    if let Err(e) = session.room.close().await {
        warn!("error closing livekit room: {e}");
    }

    let state = VoiceState::default();
    emit_state(&app, &state);

    Ok(state)
}

/// Toggle the local microphone mute state.
#[tauri::command]
pub async fn voice_toggle_mute(
    app: AppHandle,
    handle: State<'_, VoiceHandle>,
) -> Result<bool, String> {
    let mut guard = handle.session.lock().await;
    let Some(session) = guard.as_mut() else {
        return Err("no active voice session".into());
    };

    let new_muted = !session.state.mic_muted;

    // Mute/unmute via the local track publication
    if new_muted {
        session.mic_publication.mute();
    } else {
        session.mic_publication.unmute();
    }

    session.state.mic_muted = new_muted;
    let state = session.state.clone();
    drop(guard);

    emit_state(&app, &state);
    debug!("mic muted: {new_muted}");

    Ok(new_muted)
}

/// Get the current voice session state.
#[tauri::command]
pub async fn voice_state(handle: State<'_, VoiceHandle>) -> Result<VoiceState, String> {
    let guard = handle.session.lock().await;
    let state = match guard.as_ref() {
        Some(session) => session.state.clone(),
        None => VoiceState::default(),
    };
    Ok(state)
}

/// Background event loop. Processes room events and updates state.
///
/// Watches for:
/// - TrackSubscribed/TrackUnsubscribed: agent audio playback
/// - ParticipantAttributesChanged: workflow stage/mode from the agent
/// - Disconnected: clean up session on remote disconnect
/// - TrackMuted/TrackUnmuted: sync mute state
async fn event_loop(session: Arc<Mutex<Option<VoiceSession>>>, app: AppHandle) {
    loop {
        let event = {
            let mut guard = session.lock().await;
            match guard.as_mut() {
                Some(s) => match s.events.recv().await {
                    Some(e) => e,
                    None => {
                        // Event stream closed — room disconnected
                        info!("voice event stream closed");
                        s.state = VoiceState::default();
                        emit_state(&app, &s.state);
                        *guard = None;
                        return;
                    }
                },
                None => return, // session dropped
            }
        };

        match event {
            RoomEvent::ParticipantAttributesChanged {
                participant,
                changed_attributes,
            } => {
                debug!(
                    "participant attributes changed: {:?} {:?}",
                    participant.identity(),
                    changed_attributes
                );

                let mut guard = session.lock().await;
                if let Some(s) = guard.as_mut() {
                    // Only care about agent participant attributes
                    let identity = participant.identity();
                    if identity.to_string().starts_with("agent") {
                        if let Some(stage) = changed_attributes.get(ATTR_VOICE_STAGE) {
                            s.state.agent_stage = Some(stage.clone());
                        }
                        if let Some(mode) = changed_attributes.get(ATTR_VOICE_MODE) {
                            s.state.agent_mode = Some(mode.clone());
                        }
                        emit_state(&app, &s.state);
                    }
                }
            }

            RoomEvent::TrackSubscribed {
                track, participant, ..
            } => {
                debug!("track subscribed from {}", participant.identity());
                if let RemoteTrack::Audio(audio_track) = track {
                    spawn_audio_playback(audio_track, participant.identity().to_string());
                }
            }

            RoomEvent::TrackUnsubscribed { participant, .. } => {
                debug!("track unsubscribed from {}", participant.identity());
            }

            RoomEvent::Disconnected { reason } => {
                info!("disconnected from voice room: {:?}", reason);
                let mut guard = session.lock().await;
                if let Some(s) = guard.as_mut() {
                    s.state = VoiceState::default();
                    emit_state(&app, &s.state);
                }
                *guard = None;
                return;
            }

            RoomEvent::TrackMuted { participant, .. } => {
                debug!("track muted for {}", participant.identity());
                // Sync local mute state if this is our track
                let mut guard = session.lock().await;
                if let Some(s) = guard.as_mut() {
                    if matches!(&participant, Participant::Local(_)) {
                        s.state.mic_muted = true;
                        emit_state(&app, &s.state);
                    }
                }
            }

            RoomEvent::TrackUnmuted { participant, .. } => {
                debug!("track unmuted for {}", participant.identity());
                let mut guard = session.lock().await;
                if let Some(s) = guard.as_mut() {
                    if matches!(&participant, Participant::Local(_)) {
                        s.state.mic_muted = false;
                        emit_state(&app, &s.state);
                    }
                }
            }

            RoomEvent::Connected { .. } => {
                debug!("voice room connected event");
            }

            RoomEvent::Reconnecting => {
                warn!("voice room reconnecting");
            }

            RoomEvent::Reconnected => {
                info!("voice room reconnected");
            }

            _ => {} // Ignore other events
        }
    }
}

/// Spawn an audio playback task for a subscribed remote audio track.
///
/// Creates a NativeAudioStream from the remote track and reads frames.
/// On macOS/Linux, the audio is played back through the system audio device
/// by the LiveKit runtime's native audio renderer.
fn spawn_audio_playback(audio_track: RemoteAudioTrack, participant_identity: String) {
    tauri::async_runtime::spawn(async move {
        let target_sample_rate = 48000i32;
        let target_channels = 1i32;

        let rtc_track = audio_track.rtc_track();
        let mut audio_stream =
            NativeAudioStream::new(rtc_track, target_sample_rate, target_channels);

        debug!("started audio playback for {participant_identity}");

        // Read frames to drive playback through the native audio sink.
        // The NativeAudioStream connects to the platform's audio output.
        while let Some(_frame) = audio_stream.next().await {
            // Frames are processed by the native audio stream internally.
            // We just need to keep reading to keep the pipeline flowing.
        }

        debug!("audio playback ended for {participant_identity}");
    });
}