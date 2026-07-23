//! Tauri-owned voice runtime.
//!
//! The desktop owns UI commands and events only. The standalone C++23 host owns
//! platform audio, kcoro, Flashkern, model state, and inference. Until the host
//! control mailbox exposes session operations, LFM2 startup fails explicitly;
//! there is no in-process Rust or FFI inference path.

use std::{
    borrow::Cow,
    sync::{Arc, Mutex},
    time::Duration,
};

use liquid_audio::AudioStatsSnapshot;
use livekit::{
    AudioProcessingOptions, PlatformAudio,
    rtc_engine::lk_runtime::LkRuntime,
    track::LocalAudioTrack,
    webrtc::{
        audio_frame::AudioFrame,
        audio_source::{AudioSourceOptions, RtcAudioSource, native::NativeAudioSource},
        ice_candidate::IceCandidate,
        media_stream_track::MediaStreamTrack,
        peer_connection::{AnswerOptions, OfferOptions, PeerConnection, PeerConnectionState},
        peer_connection_factory::{RtcConfiguration, native::PeerConnectionFactoryExt},
    },
};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot, watch};

use crate::settings::{
    Lfm2Settings, VoiceProvider, VoiceSettings,
};

use super::control::{LiveKitGrant, SessionCtx, VoiceEvent};
use super::session::SessionBridgeConfig;
use super::threads::ThreadManager;

const VOICE_COMMAND_CAP: usize = 16;
const LIVEKIT_AGENT_AUDIO_RATE: u32 = 48_000;
const LIVEKIT_AGENT_AUDIO_CHANNELS: u32 = 1;
const LIVEKIT_AGENT_AUDIO_QUEUE_MS: u32 = 250;
const LOCAL_WEBRTC_OUTPUT_READY_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP: usize = 16;
const LOCAL_WEBRTC_OUTPUT_STREAM_ID: &str = "local-lfm2-output";
/// One active native voice service for the desktop app.
pub struct VoiceRuntime {
    commands: mpsc::Sender<RuntimeCommand>,
    state: watch::Receiver<RuntimeSnapshot>,
}

impl VoiceRuntime {
    pub fn new() -> Self {
        let (commands, rx) = mpsc::channel(VOICE_COMMAND_CAP);
        let (state_tx, state) = watch::channel(RuntimeSnapshot::default());
        tauri::async_runtime::spawn(kernel_loop(rx, state_tx));
        Self { commands, state }
    }

    pub async fn start_lfm2(
        &self,
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
    ) -> Result<(), String> {
        self.request(|reply| RuntimeCommand::StartLfm2 {
            ctx,
            settings,
            channel,
            bridge,
            reply,
        })
        .await
    }

    pub async fn start_livekit(
        &self,
        _ctx: SessionCtx,
        _settings: VoiceSettings,
        _grant: LiveKitGrant,
        _channel: tauri::ipc::Channel<VoiceEvent>,
        _bridge: Option<SessionBridgeConfig>,
    ) -> Result<(), String> {
        Err(
            "LiveKit voice inference was removed; LFM2 native voice is the only shipped path"
                .into(),
        )
    }

    pub async fn stop(&self) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::Stop { reply })
            .await
    }

    pub async fn interrupt(&self) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::Interrupt { reply })
            .await
    }

    pub async fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::SetMicEnabled { enabled, reply })
            .await
    }

    pub async fn begin_typed_input(&self) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::BeginTypedInput { reply })
            .await
    }

    pub async fn apply_settings(&self, settings: VoiceSettings) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::ApplySettings { settings, reply })
            .await
    }

    pub async fn invalidate_provider(&self, provider: VoiceProvider) -> Result<(), String> {
        self.request_critical(|reply| RuntimeCommand::InvalidateProvider { provider, reply })
            .await
    }

    pub async fn snapshot(&self) -> Result<RuntimeSnapshot, String> {
        self.request(|reply| RuntimeCommand::Snapshot { reply })
            .await
    }

    pub fn cached_snapshot(&self) -> RuntimeSnapshot {
        self.state.borrow().clone()
    }

    pub async fn is_running_session(&self, session_id: &str) -> Result<bool, String> {
        let session_id = session_id.to_string();
        self.request(|reply| RuntimeCommand::IsRunningSession { session_id, reply })
            .await
    }

    async fn request<T>(
        &self,
        cmd: impl FnOnce(oneshot::Sender<Result<T, String>>) -> RuntimeCommand,
    ) -> Result<T, String>
    where
        T: Send + 'static,
    {
        let (reply, rx) = oneshot::channel();
        self.commands.try_send(cmd(reply)).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => "voice kernel command queue is full".to_string(),
            mpsc::error::TrySendError::Closed(_) => "voice kernel stopped".to_string(),
        })?;
        rx.await
            .map_err(|_| "voice kernel dropped command reply".to_string())?
    }

    async fn request_critical<T>(
        &self,
        cmd: impl FnOnce(oneshot::Sender<Result<T, String>>) -> RuntimeCommand,
    ) -> Result<T, String>
    where
        T: Send + 'static,
    {
        let (reply, rx) = oneshot::channel();
        self.commands
            .send(cmd(reply))
            .await
            .map_err(|_| "voice kernel stopped".to_string())?;
        rx.await
            .map_err(|_| "voice kernel dropped command reply".to_string())?
    }
}

impl Default for VoiceRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeSnapshot {
    pub running: bool,
    pub running_provider: Option<VoiceProvider>,
    pub mic_enabled: bool,
    pub session_id: Option<String>,
    pub audio_stats: Option<AudioStatsSnapshot>,
}

enum RuntimeCommand {
    StartLfm2 {
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Stop {
        reply: oneshot::Sender<Result<(), String>>,
    },
    Interrupt {
        reply: oneshot::Sender<Result<(), String>>,
    },
    SetMicEnabled {
        enabled: bool,
        reply: oneshot::Sender<Result<(), String>>,
    },
    BeginTypedInput {
        reply: oneshot::Sender<Result<(), String>>,
    },
    ApplySettings {
        settings: VoiceSettings,
        reply: oneshot::Sender<Result<(), String>>,
    },
    InvalidateProvider {
        provider: VoiceProvider,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Snapshot {
        reply: oneshot::Sender<Result<RuntimeSnapshot, String>>,
    },
    IsRunningSession {
        session_id: String,
        reply: oneshot::Sender<Result<bool, String>>,
    },
}

async fn kernel_loop(
    mut rx: mpsc::Receiver<RuntimeCommand>,
    state: watch::Sender<RuntimeSnapshot>,
) {
    let threads = ThreadManager::default();
    let mut session: Option<VoiceSession> = None;
    publish_snapshot(&state, &session);

    while let Some(cmd) = rx.recv().await {
        reap_finished(&mut session, &threads);
        match cmd {
            RuntimeCommand::StartLfm2 {
                ctx,
                settings,
                channel,
                bridge,
                reply,
            } => {
                let result = start_lfm2_session(
                    &mut session,
                    &threads,
                    ctx,
                    settings,
                    channel,
                    bridge,
                );
                let _ = reply.send(result);
            }
            RuntimeCommand::Stop { reply } => {
                let result = stop_session(&mut session, &threads);
                let _ = reply.send(result);
            }
            RuntimeCommand::Interrupt { reply } => {
                let result = match session.as_ref() {
                    Some(session) => session.interrupt(),
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::SetMicEnabled { enabled, reply } => {
                let result = match session.as_ref() {
                    Some(session) => session.set_mic_enabled(enabled),
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::BeginTypedInput { reply } => {
                let result = match session.as_ref() {
                    Some(session) => match session.set_mic_enabled(false) {
                        Ok(()) => session.interrupt(),
                        Err(error) => Err(error),
                    },
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::ApplySettings { settings, reply } => {
                let result = apply_settings_to_session(&mut session, &threads, settings);
                let _ = reply.send(result);
            }
            RuntimeCommand::InvalidateProvider { provider, reply } => {
                let result = if session
                    .as_ref()
                    .is_some_and(|session| session.provider() == provider)
                {
                    stop_session(&mut session, &threads)
                } else {
                    Ok(())
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::Snapshot { reply } => {
                reap_finished(&mut session, &threads);
                let snap = runtime_snapshot(&session);
                let _ = reply.send(Ok(snap));
            }
            RuntimeCommand::IsRunningSession { session_id, reply } => {
                let running = session.as_ref().is_some_and(|session| {
                    !session.is_finished() && session.session_id() == session_id
                });
                let _ = reply.send(Ok(running));
            }
        }
        publish_snapshot(&state, &session);
    }

    let _ = stop_session(&mut session, &threads);
    let _ = threads.wait();
}

fn start_lfm2_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
    ctx: SessionCtx,
    settings: VoiceSettings,
    channel: tauri::ipc::Channel<VoiceEvent>,
    bridge: Option<SessionBridgeConfig>,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session.is_some() {
        return Err("Voice is already running.".into());
    }
    threads.wait()?;
    *session = Some(VoiceSession(Lfm2Session::spawn(
        threads, ctx, settings, channel, bridge,
    )?));
    Ok(())
}

fn apply_settings_to_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
    settings: VoiceSettings,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session
        .as_ref()
        .is_some_and(|session| !session.matches_settings(&settings))
    {
        return stop_session(session, threads);
    }
    Ok(())
}

fn stop_session(session: &mut Option<VoiceSession>, threads: &ThreadManager) -> Result<(), String> {
    if let Some(session) = session.take() {
        session.stop(threads)?;
    }
    Ok(())
}

fn reap_finished(session: &mut Option<VoiceSession>, threads: &ThreadManager) {
    let _ = threads.reap();
    cleanup_finished(session);
}

fn cleanup_finished(session: &mut Option<VoiceSession>) {
    if session.as_ref().is_some_and(VoiceSession::is_finished) {
        let _ = session.take();
    }
}

fn publish_snapshot(state: &watch::Sender<RuntimeSnapshot>, session: &Option<VoiceSession>) {
    let _ = state.send(runtime_snapshot(session));
}

fn runtime_snapshot(session: &Option<VoiceSession>) -> RuntimeSnapshot {
    let Some(session) = session.as_ref().filter(|session| !session.is_finished()) else {
        return RuntimeSnapshot::default();
    };
    RuntimeSnapshot {
        running: true,
        running_provider: Some(session.provider()),
        mic_enabled: session.mic_enabled(),
        session_id: Some(session.session_id().to_string()),
        audio_stats: session.audio_stats(),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SessionSettingsKey {
    provider: VoiceProvider,
    lfm2: Lfm2Settings,
}

impl SessionSettingsKey {
    fn lfm2(settings: &VoiceSettings) -> Self {
        Self {
            provider: VoiceProvider::Lfm2,
            lfm2: settings.lfm2.clone(),
        }
    }

    fn matches(&self, settings: &VoiceSettings) -> bool {
        if settings.provider != self.provider {
            return false;
        }
        if self.provider == VoiceProvider::Lfm2 {
            return self == &Self::lfm2(settings);
        }
        false
    }
}

struct VoiceSession(Lfm2Session);

impl VoiceSession {
    fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    fn provider(&self) -> VoiceProvider {
        VoiceProvider::Lfm2
    }

    fn session_id(&self) -> &str {
        &self.0.ctx.session_id
    }

    fn interrupt(&self) -> Result<(), String> {
        self.0.interrupt()
    }

    fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        self.0.set_mic_enabled(enabled)
    }

    fn mic_enabled(&self) -> bool {
        self.0.mic_enabled()
    }

    fn audio_stats(&self) -> Option<AudioStatsSnapshot> {
        self.0.audio_stats()
    }

    fn matches_settings(&self, settings: &VoiceSettings) -> bool {
        self.0.settings.matches(settings)
    }

    fn stop(self, threads: &ThreadManager) -> Result<(), String> {
        stop_lfm2(threads, self.0)
    }
}

fn stop_lfm2(threads: &ThreadManager, session: Lfm2Session) -> Result<(), String> {
    let stopped = session.stop();
    let joined = threads.wait();
    match (stopped, joined) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(stop), Ok(())) => Err(stop),
        (Ok(()), Err(join)) => Err(join),
        (Err(stop), Err(join)) => Err(format!("{stop}; {join}")),
    }
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn livekit_audio_frame_count(samples: usize, rate: u32, channels: u32) -> usize {
    if samples == 0 || rate == 0 || channels == 0 {
        return 0;
    }
    let samples_per_channel = (rate / 100).max(1);
    let samples_per_frame = (samples_per_channel * channels) as usize;
    samples.div_ceil(samples_per_frame)
}

fn livekit_audio_frames(pcm: &[f32], rate: u32, channels: u32) -> Vec<AudioFrame<'static>> {
    if pcm.is_empty() || rate == 0 || channels == 0 {
        return Vec::new();
    }
    let samples_per_channel = (rate / 100).max(1);
    let samples_per_frame = (samples_per_channel * channels) as usize;
    pcm.chunks(samples_per_frame)
        .map(|chunk| {
            let mut data = vec![0; samples_per_frame];
            for (dst, src) in data.iter_mut().zip(chunk) {
                *dst = f32_to_i16(*src);
            }
            AudioFrame {
                data: Cow::Owned(data),
                sample_rate: rate,
                num_channels: channels,
                samples_per_channel,
            }
        })
        .collect()
}

struct Lfm2Session {
    ctx: SessionCtx,
    settings: SessionSettingsKey,
}

impl Lfm2Session {
    fn spawn(
        _threads: &ThreadManager,
        _ctx: SessionCtx,
        _settings: VoiceSettings,
        _channel: tauri::ipc::Channel<VoiceEvent>,
        _bridge: Option<SessionBridgeConfig>,
    ) -> Result<Self, String> {
        Err(
            "The C++23 voice host session mailbox is not mounted. In-process Rust/FFI inference has been removed."
                .into(),
        )
    }

    fn is_finished(&self) -> bool {
        true
    }

    fn interrupt(&self) -> Result<(), String> {
        Err("no C++23 voice host session is active".into())
    }

    fn set_mic_enabled(&self, _enabled: bool) -> Result<(), String> {
        Err("no C++23 voice host session is active".into())
    }

    fn mic_enabled(&self) -> bool {
        false
    }

    fn audio_stats(&self) -> Option<AudioStatsSnapshot> {
        None
    }

    fn stop(self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceAudioProbeReport {
    pub sample_rate: u32,
    pub samples_written: usize,
    pub webrtc_frames: usize,
    pub duration_ms: u64,
    pub playout_devices: usize,
    pub recording_devices: usize,
    pub playout_device: Option<String>,
    pub adm_playout_enabled: bool,
    pub playout_initialized: bool,
}

pub async fn play_local_webrtc_probe(
    duration: Duration,
    frequency_hz: f32,
    amplitude: f32,
) -> Result<VoiceAudioProbeReport, String> {
    let state = Arc::new(local_webrtc_output_loopback().await?);
    let rate = LIVEKIT_AGENT_AUDIO_RATE;
    let frames = ((duration.as_secs_f32() * rate as f32).ceil() as usize).max(1);
    let freq = if frequency_hz.is_finite() && frequency_hz > 0.0 {
        frequency_hz
    } else {
        440.0
    };
    let amp = if amplitude.is_finite() {
        amplitude.clamp(0.0, 0.5)
    } else {
        0.12
    };
    let fade = (rate as usize / 100).max(1);
    let mut samples = Vec::with_capacity(frames);
    for i in 0..frames {
        let edge = i.min(frames.saturating_sub(1).saturating_sub(i));
        let env = (edge as f32 / fade as f32).min(1.0);
        let phase = std::f32::consts::TAU * freq * i as f32 / rate as f32;
        samples.push(phase.sin() * amp * env);
    }
    let write = state.clone();
    tauri::async_runtime::spawn_blocking(move || write.write(&samples, rate))
        .await
        .map_err(|e| format!("audio probe task failed: {e}"))??;
    tokio::time::sleep(duration + Duration::from_millis(300)).await;
    state.clear();
    Ok(state.probe_report(rate, frames, duration))
}

struct LocalWebRtcOutputState {
    audio: PlatformAudio,
    source: NativeAudioSource,
    _track: LocalAudioTrack,
    sender: PeerConnection,
    receiver: PeerConnection,
    _remote_track: livekit::webrtc::audio_track::RtcAudioTrack,
}

impl LocalWebRtcOutputState {
    fn probe_report(
        &self,
        sample_rate: u32,
        samples_written: usize,
        duration: Duration,
    ) -> VoiceAudioProbeReport {
        let runtime = LkRuntime::instance();
        let pcf = runtime.pc_factory();
        let playout_device = self
            .audio
            .playout_devices()
            .next()
            .map(|device| device.name)
            .filter(|name| !name.is_empty());
        VoiceAudioProbeReport {
            sample_rate,
            samples_written,
            webrtc_frames: livekit_audio_frame_count(
                samples_written,
                sample_rate,
                LIVEKIT_AGENT_AUDIO_CHANNELS,
            ),
            duration_ms: duration.as_millis() as u64,
            playout_devices: self.audio.playout_devices().count(),
            recording_devices: self.audio.recording_devices().count(),
            playout_device,
            adm_playout_enabled: pcf.adm_playout_enabled(),
            playout_initialized: pcf.playout_is_initialized(),
        }
    }

    fn write(&self, pcm: &[f32], rate: u32) -> Result<(), String> {
        if rate != LIVEKIT_AGENT_AUDIO_RATE {
            return Err(format!(
                "local WebRTC output expected {} Hz, got {rate} Hz",
                LIVEKIT_AGENT_AUDIO_RATE
            ));
        }
        for frame in livekit_audio_frames(pcm, rate, LIVEKIT_AGENT_AUDIO_CHANNELS) {
            tauri::async_runtime::block_on(self.source.capture_frame(&frame))
                .map_err(|e| format!("local WebRTC speaker capture failed: {e}"))?;
        }
        Ok(())
    }

    fn clear(&self) {
        self.source.clear_buffer();
    }
}

impl Drop for LocalWebRtcOutputState {
    fn drop(&mut self) {
        self.source.clear_buffer();
        stop_local_webrtc_playout();
        self.sender.close();
        self.receiver.close();
    }
}

fn configure_platform_audio(audio: &PlatformAudio) -> Result<(), String> {
    audio
        .configure_audio_processing(AudioProcessingOptions {
            echo_cancellation: true,
            noise_suppression: true,
            auto_gain_control: true,
            prefer_hardware_processing: false,
        })
        .map_err(|error| error.to_string())
}

async fn local_webrtc_output_loopback() -> Result<LocalWebRtcOutputState, String> {
    let audio = PlatformAudio::new().map_err(|e| format!("WebRTC speaker device failed: {e}"))?;
    configure_platform_audio(&audio)
        .map_err(|e| format!("WebRTC speaker audio processing failed: {e}"))?;
    let source = NativeAudioSource::new(
        AudioSourceOptions::default(),
        LIVEKIT_AGENT_AUDIO_RATE,
        LIVEKIT_AGENT_AUDIO_CHANNELS,
        LIVEKIT_AGENT_AUDIO_QUEUE_MS,
    );
    let track = LocalAudioTrack::create_audio_track(
        "local-assistant",
        RtcAudioSource::Native(source.clone()),
    );
    let runtime = LkRuntime::instance();
    let factory = runtime.pc_factory();
    let sender = factory
        .create_peer_connection(RtcConfiguration::default())
        .map_err(|e| format!("local WebRTC output sender failed: {e}"))?;
    let receiver = factory
        .create_peer_connection(RtcConfiguration::default())
        .map_err(|e| format!("local WebRTC output receiver failed: {e}"))?;
    let (sender_ice_tx, sender_ice_rx) =
        mpsc::channel::<IceCandidate>(LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP);
    let (receiver_ice_tx, receiver_ice_rx) =
        mpsc::channel::<IceCandidate>(LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP);
    let (sender_state_tx, sender_state_rx) = watch::channel(sender.connection_state());
    let (receiver_state_tx, receiver_state_rx) = watch::channel(receiver.connection_state());
    let (remote_tx, remote_rx) = oneshot::channel::<livekit::webrtc::audio_track::RtcAudioTrack>();
    let remote_tx = Arc::new(Mutex::new(Some(remote_tx)));

    sender.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = sender_ice_tx.try_send(candidate);
    })));
    receiver.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = receiver_ice_tx.try_send(candidate);
    })));
    sender.on_connection_state_change(Some(Box::new(move |state| {
        sender_state_tx.send_replace(state);
    })));
    receiver.on_connection_state_change(Some(Box::new(move |state| {
        receiver_state_tx.send_replace(state);
    })));
    receiver.on_track(Some(Box::new(move |event| {
        if let MediaStreamTrack::Audio(track) = event.track {
            if let Ok(mut tx) = remote_tx.lock()
                && let Some(tx) = tx.take()
            {
                let _ = tx.send(track);
            }
        }
    })));

    sender
        .add_track(
            MediaStreamTrack::Audio(track.rtc_track()),
            &[LOCAL_WEBRTC_OUTPUT_STREAM_ID],
        )
        .map_err(|e| format!("local WebRTC output add_track failed: {e}"))?;
    let offer = sender
        .create_offer(OfferOptions::default())
        .await
        .map_err(|e| format!("local WebRTC output offer failed: {e}"))?;
    sender
        .set_local_description(offer.clone())
        .await
        .map_err(|e| format!("local WebRTC output set local offer failed: {e}"))?;
    receiver
        .set_remote_description(offer)
        .await
        .map_err(|e| format!("local WebRTC output set remote offer failed: {e}"))?;
    let answer = receiver
        .create_answer(AnswerOptions::default())
        .await
        .map_err(|e| format!("local WebRTC output answer failed: {e}"))?;
    receiver
        .set_local_description(answer.clone())
        .await
        .map_err(|e| format!("local WebRTC output set local answer failed: {e}"))?;
    sender
        .set_remote_description(answer)
        .await
        .map_err(|e| format!("local WebRTC output set remote answer failed: {e}"))?;
    exchange_loopback_ice(
        &sender,
        &receiver,
        sender_ice_rx,
        receiver_ice_rx,
        sender_state_rx,
        receiver_state_rx,
    )
    .await?;
    let remote_track = tokio::time::timeout(
        Duration::from_millis(LOCAL_WEBRTC_OUTPUT_READY_TIMEOUT_MS),
        remote_rx,
    )
    .await
    .map_err(|_| "local WebRTC speaker track did not become ready".to_string())?
    .map_err(|_| "local WebRTC speaker track setup channel closed".to_string())?;
    start_local_webrtc_playout()?;

    Ok(LocalWebRtcOutputState {
        audio,
        source,
        _track: track,
        sender,
        receiver,
        _remote_track: remote_track,
    })
}

/// Holders of the shared factory-level ADM playout. The Settings speaker probe
/// creates a short-lived `LocalWebRtcOutputState` WHILE a live voice session owns
/// one — an unconditional stop on the probe's drop would silence the live session.
/// Playout stops only when the last holder releases it.
static LOCAL_PLAYOUT_HOLDERS: Mutex<usize> = Mutex::new(0);

fn start_local_webrtc_playout() -> Result<(), String> {
    let mut holders = LOCAL_PLAYOUT_HOLDERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if *holders == 0 {
        let runtime = LkRuntime::instance();
        let factory = runtime.pc_factory();
        factory.set_adm_playout_enabled(true);
        if !factory.playout_is_initialized() && !factory.init_playout() {
            return Err("local WebRTC speaker playout init failed".into());
        }
        if !factory.start_playout() {
            return Err("local WebRTC speaker playout start failed".into());
        }
    }
    *holders += 1;
    Ok(())
}

fn stop_local_webrtc_playout() {
    let mut holders = LOCAL_PLAYOUT_HOLDERS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *holders = holders.saturating_sub(1);
    if *holders == 0 {
        let runtime = LkRuntime::instance();
        let _ = runtime.pc_factory().stop_playout();
    }
}

async fn exchange_loopback_ice(
    sender: &PeerConnection,
    receiver: &PeerConnection,
    mut sender_ice: mpsc::Receiver<IceCandidate>,
    mut receiver_ice: mpsc::Receiver<IceCandidate>,
    mut sender_state: watch::Receiver<PeerConnectionState>,
    mut receiver_state: watch::Receiver<PeerConnectionState>,
) -> Result<(), String> {
    let deadline = tokio::time::sleep(Duration::from_millis(LOCAL_WEBRTC_OUTPUT_ICE_TIMEOUT_MS));
    tokio::pin!(deadline);
    let mut sender_open = true;
    let mut receiver_open = true;
    loop {
        if loopback_connected(&sender_state, &receiver_state) {
            return Ok(());
        }
        if loopback_failed(&sender_state, &receiver_state) {
            return Err("local WebRTC audio loopback connection failed".into());
        }
        tokio::select! {
            _ = &mut deadline => {
                return Err("local WebRTC audio loopback did not connect before timeout".into())
            },
            candidate = sender_ice.recv(), if sender_open => match candidate {
                Some(candidate) => receiver
                    .add_ice_candidate(candidate)
                    .await
                    .map_err(|e| format!("local WebRTC output receiver ICE failed: {e}"))?,
                None => sender_open = false,
            },
            candidate = receiver_ice.recv(), if receiver_open => match candidate {
                Some(candidate) => sender
                    .add_ice_candidate(candidate)
                    .await
                    .map_err(|e| format!("local WebRTC output sender ICE failed: {e}"))?,
                None => receiver_open = false,
            },
            changed = sender_state.changed() => {
                changed.map_err(|_| "local WebRTC output sender state closed".to_string())?;
            },
            changed = receiver_state.changed() => {
                changed.map_err(|_| "local WebRTC output receiver state closed".to_string())?;
            },
        }
    }
}

fn loopback_connected(
    sender: &watch::Receiver<PeerConnectionState>,
    receiver: &watch::Receiver<PeerConnectionState>,
) -> bool {
    matches!(*sender.borrow(), PeerConnectionState::Connected)
        && matches!(*receiver.borrow(), PeerConnectionState::Connected)
}

fn loopback_failed(
    sender: &watch::Receiver<PeerConnectionState>,
    receiver: &watch::Receiver<PeerConnectionState>,
) -> bool {
    matches!(
        *sender.borrow(),
        PeerConnectionState::Failed | PeerConnectionState::Closed
    ) || matches!(
        *receiver.borrow(),
        PeerConnectionState::Failed | PeerConnectionState::Closed
    )
}
