//! Voice module — native full-duplex audio transport with echo cancellation.
//!
//! Connects to a LiveKit room, captures the local microphone via cpal, publishes
//! it as a LocalAudioTrack, subscribes to the agent's remote audio track, and
//! plays it back through the OS speakers. An AEC3 pipeline cancels the speaker
//! echo from the mic path so the agent doesn't hear itself.
//!
//! ## Architecture
//!
//! ```text
//! mic (cpal) → resample → AEC3 → NativeAudioSource → LiveKit publish
//!                                                  ↕ (room events)
//! LiveKit subscribe → NativeAudioStream → ring buffer → cpal (speakers)
//!                                                 ↓
//!                                            AEC3 render reference
//! ```
//!
//! cpal streams are !Send, so they live on dedicated std::threads that
//! communicate with the async LiveKit code via channels.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use livekit::options::TrackPublishOptions;
use livekit::prelude::*;
use livekit::track::{LocalAudioTrack, LocalTrack, TrackSource};
use livekit::webrtc::audio_frame::AudioFrame;
use livekit::webrtc::{
    audio_source::native::NativeAudioSource,
    audio_stream::native::NativeAudioStream,
    prelude::{AudioSourceOptions, RtcAudioSource},
};
use log::{debug, error, info, warn};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{mpsc, Mutex as AsyncMutex};

const VOICE_STATE_EVENT: &str = "voice://state-changed";
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u32 = 1;
const FRAME_SAMPLES: usize = (SAMPLE_RATE as usize) / 100;
const SCALE: f32 = 32768.0;
const RENDER_CAP: usize = SAMPLE_RATE as usize / 10;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceState {
    pub connected: bool,
    pub room: Option<String>,
    pub agent_stage: Option<String>,
    pub agent_mode: Option<String>,
    pub mic_muted: bool,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self { connected: false, room: None, agent_stage: None, agent_mode: None, mic_muted: false }
    }
}

const ATTR_VOICE_STAGE: &str = "emberharmony.voice_stage";
const ATTR_VOICE_MODE: &str = "emberharmony.voice_mode";

/// LiveKit room and event receiver. Audio streams live on their own threads
/// and are dropped via the shutdown channel.
struct VoiceSession {
    room: Room,
    mic_publication: LocalTrackPublication,
    events: mpsc::UnboundedReceiver<RoomEvent>,
    playback: Arc<Mutex<VecDeque<i16>>>,
    state: VoiceState,
    /// Sending on this shuts down the cpal audio threads (input + output).
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    /// Playback shutdown — drops the output stream.
    playback_shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
}

pub struct VoiceHandle {
    session: Arc<AsyncMutex<Option<VoiceSession>>>,
}

impl VoiceHandle {
    pub fn new() -> Self {
        Self { session: Arc::new(AsyncMutex::new(None)) }
    }
}

fn emit_state(app: &AppHandle, state: &VoiceState) {
    if let Err(e) = app.emit(VOICE_STATE_EVENT, state) {
        warn!("failed to emit voice state event: {e}");
    }
}

#[tauri::command]
pub async fn voice_connect(
    app: AppHandle,
    handle: State<'_, VoiceHandle>,
    url: String,
    token: String,
) -> Result<VoiceState, String> {
    if handle.session.lock().await.is_some() {
        return Err("voice session already active".into());
    }

    info!("connecting to voice room: {url}");

    let (room, events) = Room::connect(&url, &token, RoomOptions::default())
        .await
        .map_err(|e| format!("livekit connect failed: {e}"))?;

    let room_name = room.name();
    info!("connected to room: {room_name}");

    let source = NativeAudioSource::new(
        AudioSourceOptions { echo_cancellation: false, noise_suppression: false, auto_gain_control: false },
        SAMPLE_RATE,
        CHANNELS,
        1000,
    );

    let mic = LocalAudioTrack::create_audio_track("microphone", RtcAudioSource::Native(source.clone()));
    let mic_publication = room
        .local_participant()
        .publish_track(
            LocalTrack::Audio(mic),
            TrackPublishOptions { source: TrackSource::Microphone, dtx: true, red: true, ..Default::default() },
        )
        .await
        .map_err(|e| format!("failed to publish mic track: {e}"))?;

    debug!("published mic track: {}", mic_publication.sid());

    let playback: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));
    let render: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));

    // Shutdown channel to stop the cpal threads when disconnecting
    let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();

    start_capture(source, render.clone(), shutdown_rx);

    let (playback_shutdown_tx, playback_shutdown_rx) = std::sync::mpsc::channel::<()>();
    start_playback(playback.clone(), render, playback_shutdown_rx);

    let state = VoiceState {
        connected: true,
        room: Some(room_name.clone()),
        agent_stage: None,
        agent_mode: None,
        mic_muted: false,
    };

    let session = VoiceSession {
        room,
        mic_publication,
        events,
        playback,
        state: state.clone(),
        shutdown_tx: Some(shutdown_tx),
        playback_shutdown_tx: Some(playback_shutdown_tx),
    };

    *handle.session.lock().await = Some(session);

    let session_arc = handle.session.clone();
    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        event_loop(session_arc, app_clone).await;
    });

    emit_state(&app, &state);
    info!("voice connected to room: {room_name}");
    Ok(state)
}

#[tauri::command]
pub async fn voice_disconnect(
    app: AppHandle,
    handle: State<'_, VoiceHandle>,
) -> Result<VoiceState, String> {
    let mut guard = handle.session.lock().await;
    let Some(mut session) = guard.take() else {
        return Err("no active voice session".into());
    };

    info!("disconnecting from voice room");

    // Signal the cpal threads to stop, then close the room
    drop(session.shutdown_tx.take());
    drop(session.playback_shutdown_tx.take());
    if let Err(e) = session.room.close().await {
        warn!("error closing livekit room: {e}");
    }
    drop(session);

    let state = VoiceState::default();
    emit_state(&app, &state);
    Ok(state)
}

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

#[tauri::command]
pub async fn voice_state(handle: State<'_, VoiceHandle>) -> Result<VoiceState, String> {
    let guard = handle.session.lock().await;
    Ok(guard.as_ref().map(|s| s.state.clone()).unwrap_or_default())
}

async fn event_loop(session: Arc<AsyncMutex<Option<VoiceSession>>>, app: AppHandle) {
    loop {
        let event = {
            let mut guard = session.lock().await;
            match guard.as_mut() {
                Some(s) => match s.events.recv().await {
                    Some(e) => e,
                    None => {
                        info!("voice event stream closed");
                        s.state = VoiceState::default();
                        emit_state(&app, &s.state);
                        *guard = None;
                        return;
                    }
                },
                None => return,
            }
        };

        match event {
            RoomEvent::ParticipantAttributesChanged { participant, changed_attributes } => {
                debug!("participant attributes changed: {:?} {:?}", participant.identity(), changed_attributes);
                let mut guard = session.lock().await;
                if let Some(s) = guard.as_mut() {
                    if participant.identity().to_string().starts_with("agent") {
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

            RoomEvent::TrackSubscribed { track, participant, .. } => {
                debug!("track subscribed from {}", participant.identity());
                if let RemoteTrack::Audio(audio_track) = track {
                    let guard = session.lock().await;
                    let pb = guard.as_ref().map(|s| s.playback.clone()).unwrap_or_default();
                    drop(guard);
                    spawn_audio_playback(audio_track, participant.identity().to_string(), pb);
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

            RoomEvent::Connected { .. } => { debug!("voice room connected event"); }
            RoomEvent::Reconnecting => { warn!("voice room reconnecting"); }
            RoomEvent::Reconnected => { info!("voice room reconnected"); }
            _ => {}
        }
    }
}

fn spawn_audio_playback(audio_track: RemoteAudioTrack, participant_identity: String, playback: Arc<Mutex<VecDeque<i16>>>) {
    tauri::async_runtime::spawn(async move {
        let mut audio_stream = NativeAudioStream::new(audio_track.rtc_track(), SAMPLE_RATE as i32, CHANNELS as i32);
        debug!("started audio playback for {participant_identity}");
        while let Some(frame) = audio_stream.next().await {
            let mut buf = playback.lock().unwrap();
            buf.extend(frame.data.iter().copied());
            let max = SAMPLE_RATE as usize * 2;
            while buf.len() > max { buf.pop_front(); }
        }
        debug!("audio playback ended for {participant_identity}");
    });
}

fn start_capture(
    source: NativeAudioSource,
    render: Arc<Mutex<VecDeque<i16>>>,
    shutdown_rx: std::sync::mpsc::Receiver<()>,
) {
    let (raw_tx, raw_rx) = std::sync::mpsc::channel::<(Vec<i16>, u32)>();
    let (clean_tx, mut clean_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<i16>>();

    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => { error!("no microphone device found"); return; }
        };
        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(e) => { error!("input config error: {e}"); return; }
        };
        let dev_rate = config.sample_rate().0;
        let dev_ch = config.channels() as usize;
        info!("mic: {} ch @ {} Hz", dev_ch, dev_rate);

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    let mut mono = Vec::with_capacity(data.len() / dev_ch);
                    for ch in data.chunks(dev_ch) {
                        let avg = ch.iter().copied().sum::<f32>() / dev_ch as f32;
                        mono.push((avg.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
                    }
                    let _ = raw_tx.send((mono, dev_rate));
                },
                |e| error!("input stream error: {e}"),
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let mut mono = Vec::with_capacity(data.len() / dev_ch);
                    for ch in data.chunks(dev_ch) {
                        let sum: i32 = ch.iter().map(|&s| s as i32).sum();
                        mono.push((sum / dev_ch as i32) as i16);
                    }
                    let _ = raw_tx.send((mono, dev_rate));
                },
                |e| error!("input stream error: {e}"),
                None,
            ),
            other => { error!("unsupported input sample format: {other:?}"); return; }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(e) => { error!("failed to build input stream: {e}"); return; }
        };

        if let Err(e) = stream.play() {
            error!("failed to start input stream: {e}");
            return;
        }

        // Block until shutdown signal — keeping the stream alive
        let _ = shutdown_rx.recv();
        drop(stream);
    });

    // AEC thread: resample → echo-cancel → chunk to 10ms frames
    std::thread::spawn(move || {
        let format = AudioFormat::ten_ms(SAMPLE_RATE, CHANNELS as u16);
        let mut pipeline = match linear::builder(format, format).initial_delay_ms(116).build() {
            Ok(p) => Some(p),
            Err(e) => {
                error!("AEC init failed ({e}); mic passes through WITHOUT echo cancellation");
                None
            }
        };
        let mut acc: Vec<i16> = Vec::new();
        let mut render_f = vec![0.0f32; FRAME_SAMPLES];
        let mut cap_f = vec![0.0f32; FRAME_SAMPLES];
        let mut out_f = vec![0.0f32; FRAME_SAMPLES];

        while let Ok((mono, rate)) = raw_rx.recv() {
            acc.extend(resample(&mono, rate, SAMPLE_RATE));
            while acc.len() >= FRAME_SAMPLES {
                let chunk: Vec<i16> = acc.drain(..FRAME_SAMPLES).collect();
                let cleaned: Vec<i16> = if let Some(pl) = pipeline.as_mut() {
                    {
                        let mut r = render.lock().unwrap();
                        for slot in render_f.iter_mut() {
                            *slot = r.pop_front().unwrap_or(0) as f32 / SCALE;
                        }
                    }
                    for (dst, &s) in cap_f.iter_mut().zip(chunk.iter()) {
                        *dst = s as f32 / SCALE;
                    }
                    let _ = pl.handle_render_frame(&render_f);
                    match pl.process_capture_frame(&cap_f, &mut out_f) {
                        Ok(_) => out_f.iter().map(|&s| (s * SCALE).clamp(i16::MIN as f32, i16::MAX as f32) as i16).collect(),
                        Err(e) => { error!("AEC capture error: {e}"); chunk }
                    }
                } else {
                    chunk
                };
                let _ = clean_tx.send(cleaned);
            }
        }
    });

    // Tokio task: push cleaned frames into the LiveKit source
    tauri::async_runtime::spawn(async move {
        while let Some(chunk) = clean_rx.recv().await {
            let frame = AudioFrame {
                data: Cow::Owned(chunk),
                num_channels: CHANNELS,
                sample_rate: SAMPLE_RATE,
                samples_per_channel: FRAME_SAMPLES as u32,
            };
            if let Err(e) = source.capture_frame(&frame).await {
                error!("capture_frame error: {e}");
            }
        }
    });
}

fn start_playback(
    playback: Arc<Mutex<VecDeque<i16>>>,
    render: Arc<Mutex<VecDeque<i16>>>,
    shutdown_rx: std::sync::mpsc::Receiver<()>,
) {
    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_output_device() {
            Some(d) => d,
            None => { error!("no speaker device found"); return; }
        };
        let config = match device.default_output_config() {
            Ok(c) => c,
            Err(e) => { error!("output config error: {e}"); return; }
        };
        let dev_ch = config.channels() as usize;
        info!("speaker: {} ch @ {} Hz", dev_ch, config.sample_rate().0);

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_output_stream(
                &config.into(),
                move |out: &mut [f32], _| {
                    let mut b = playback.lock().unwrap();
                    let mut r = render.lock().unwrap();
                    for frame in out.chunks_mut(dev_ch) {
                        let s = b.pop_front().unwrap_or(0);
                        r.push_back(s);
                        let f = s as f32 / i16::MAX as f32;
                        for o in frame.iter_mut() { *o = f; }
                    }
                    while r.len() > RENDER_CAP { r.pop_front(); }
                },
                |e| error!("output stream error: {e}"),
                None,
            ),
            cpal::SampleFormat::I16 => device.build_output_stream(
                &config.into(),
                move |out: &mut [i16], _| {
                    let mut b = playback.lock().unwrap();
                    let mut r = render.lock().unwrap();
                    for frame in out.chunks_mut(dev_ch) {
                        let s = b.pop_front().unwrap_or(0);
                        r.push_back(s);
                        for o in frame.iter_mut() { *o = s; }
                    }
                    while r.len() > RENDER_CAP { r.pop_front(); }
                },
                |e| error!("output stream error: {e}"),
                None,
            ),
            other => { error!("unsupported output sample format: {other:?}"); return; }
        };

        let stream = match stream {
            Ok(s) => s,
            Err(e) => { error!("failed to build output stream: {e}"); return; }
        };

        if let Err(e) = stream.play() {
            error!("failed to start output stream: {e}");
            return;
        }

        // Block until shutdown signal — keeping the stream alive
        let _ = shutdown_rx.recv();
        drop(stream);
    });
}

fn resample(input: &[i16], from: u32, to: u32) -> Vec<i16> {
    if from == to || input.is_empty() { return input.to_vec(); }
    let ratio = to as f64 / from as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f64;
        let a = input[idx.min(last)] as f64;
        let b = input[(idx + 1).min(last)] as f64;
        out.push((a + (b - a) * frac).round() as i16);
    }
    out
}