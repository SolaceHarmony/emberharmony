//! Voice module — native LiveKit room transport.
//!
//! Uses the LiveKit Rust SDK to connect to voice rooms and manage audio
//! tracks. This is the native (non-webview) path; it will replace the JS
//! worker's WebRTC for cross-platform audio.
//!
//! ## Architecture
//!
//! The voice module runs inside the Tauri app process. It:
//! - Connects to a LiveKit room using the token from `POST /voice/token`
//! - Creates a local audio track + source and publishes it to the room
//! - Subscribes to the agent's audio track
//! - Watches ParticipantAttributesChanged for workflow stage/mode updates
//! - Exposes state (connected, speaking, agent stage) via Tauri commands + events
//!
//! ## NOT YET WIRED: OS audio device I/O
//!
//! The room/track plumbing is in place, but the bridge to the operating
//! system's microphone and speakers is NOT implemented yet:
//! - The published mic track's `NativeAudioSource` is never fed real mic
//!   frames, so it currently carries silence.
//! - Subscribed agent audio is read from `NativeAudioStream` but is NOT routed
//!   to an output device, so nothing is played back.
//!
//! Wiring this needs an OS audio layer (e.g. `cpal`) running on a dedicated
//! thread (cpal streams are `!Send`), bridged to these LiveKit frames. Until
//! then, this transport connects but is SILENT in both directions.
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

    // Create mic audio source (48kHz mono — standard for speech).
    // NOTE: this source is NOT yet fed real microphone frames (no cpal capture),
    // so the published track currently carries silence. See module docs.
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
        state: state.clone(),
    };

    // Store the session
    *handle.session.lock().await = Some(session);

    // Spawn the event loop. The event receiver is owned by the loop (not stored
    // in the session) so it can await the next event WITHOUT holding the session
    // lock — otherwise voice_toggle_mute / voice_state / voice_disconnect would
    // block until an unrelated room event happened to arrive.
    let session_arc = handle.session.clone();
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        event_loop(session_arc, app_clone, events).await;
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
async fn event_loop(
    session: Arc<Mutex<Option<VoiceSession>>>,
    app: AppHandle,
    mut events: mpsc::UnboundedReceiver<RoomEvent>,
) {
    loop {
        // Await the next event WITHOUT holding the session lock, so Tauri
        // commands (toggle_mute / state / disconnect) can touch the session
        // concurrently instead of blocking until an event arrives.
        let Some(event) = events.recv().await else {
            // Event stream closed — room disconnected.
            info!("voice event stream closed");
            let mut guard = session.lock().await;
            if let Some(s) = guard.as_mut() {
                s.state = VoiceState::default();
                emit_state(&app, &s.state);
            }
            *guard = None;
            return;
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

/// Drain a subscribed remote audio track's decoded frames.
///
/// NOTE: `NativeAudioStream` is a *read* stream — it yields decoded PCM frames
/// but does NOT play them to any speaker, and there is no "native audio
/// renderer" that does so automatically. Routing these frames to an OS output
/// device (e.g. via `cpal`) is NOT implemented yet, so the agent's voice is
/// currently inaudible. We drain the stream to avoid the decoder backing up
/// until the output-device bridge lands.
fn spawn_audio_playback(audio_track: RemoteAudioTrack, participant_identity: String) {
    tauri::async_runtime::spawn(async move {
        let target_sample_rate = 48000i32;
        let target_channels = 1i32;

        let rtc_track = audio_track.rtc_track();
        let mut audio_stream =
            NativeAudioStream::new(rtc_track, target_sample_rate, target_channels);

        debug!("draining agent audio for {participant_identity} (playback not yet wired)");

        // TODO(voice): route these frames to an OS output device (cpal).
        while let Some(_frame) = audio_stream.next().await {}

        debug!("agent audio stream ended for {participant_identity}");
    });
}