//! Tauri-owned voice runtime.
//!
//! The audio service itself lives in `liquid_audio::VoiceRuntime`: CPAL mic
//! capture, speaker playback, VAD, barge-in, and the realtime inference worker
//! all run in-process on Rust threads. This module is the desktop kernel wrapper:
//! it loads Tauri settings, builds the LFM2 engine, owns the single active
//! session, and maps runtime events onto the webview `Channel`.

use std::{
    borrow::Cow,
    fs,
    future::Future,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use candle_core::Device;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use futures::StreamExt;
use liquid_audio::{
    GenParams, Lfm2VoiceEngine, RealtimePipeline, RealtimePipelineHandle, RuntimeConfig,
    RuntimeEvent, SessionState, Utterance, VoiceEvent as RealtimeEvent,
    VoiceRuntime as Lfm2Runtime, from_pretrained,
};
use livekit::{
    AudioProcessingOptions, ConnectionState, DataPacket, PlatformAudio, Room, RoomEvent,
    RoomOptions,
    options::TrackPublishOptions,
    track::{LocalAudioTrack, LocalTrack, RemoteAudioTrack, RemoteTrack, TrackSource},
    webrtc::audio_stream::native::NativeAudioStream,
    webrtc::{
        audio_frame::AudioFrame,
        audio_source::{AudioSourceOptions, RtcAudioSource, native::NativeAudioSource},
    },
};
use tokio::sync::{mpsc, oneshot, watch};

use crate::settings::{
    self, Lfm2Device, Lfm2Settings, LiveKitSettings, VoiceProvider, VoiceSettings,
};

use super::control::{LiveKitGrant, Role, SessionCtx, VoiceEvent, VoiceState};
use super::session::{SessionBridgeConfig, SessionBridgeEvent, run_turn};
use super::threads::ThreadManager;

type UiChannel = Arc<UiEvents>;
type AsyncTask = tauri::async_runtime::JoinHandle<()>;
const VOICE_COMMAND_CAP: usize = 16;
const LIVEKIT_COMMAND_CAP: usize = 16;
const UI_EVENT_CAP: usize = 256;
const LIVEKIT_AUDIO_LEVEL_RATE: i32 = 48_000;
const LIVEKIT_AUDIO_LEVEL_CHANNELS: i32 = 1;
const LIVEKIT_REMOTE_SPEAKING_RMS: f32 = 0.006;
const LIVEKIT_REMOTE_SILENCE_MS: u64 = 350;
const LIVEKIT_AGENT_AUDIO_TIMEOUT_MS: u64 = 20_000;
const LIVEKIT_DONE_POLL_MS: u64 = 50;
const LIVEKIT_CONTROL_TOPIC: &str = "emberharmony.voice.control";
const LIVEKIT_CONTROL_INTERRUPT: &[u8] = br#"{"type":"interrupt"}"#;
const LIVEKIT_AGENT_AUDIO_RATE: u32 = 48_000;
const LIVEKIT_AGENT_AUDIO_CHANNELS: u32 = 1;
const LIVEKIT_AGENT_AUDIO_RATE_I32: i32 = 48_000;
const LIVEKIT_AGENT_AUDIO_CHANNELS_I32: i32 = 1;
const LIVEKIT_AGENT_AUDIO_QUEUE_MS: u32 = 250;
const LIVEKIT_AGENT_SILENCE_MS: u64 = 800;
const LIVEKIT_AGENT_MAX_UTTERANCE_SECONDS: usize = 30;
const LIVEKIT_AGENT_ECHO_GATE_MS: u64 = 700;
const LIVEKIT_AGENT_ECHO_MULTIPLIER: f32 = 2.5;
const LFM2_CONVERSE_SYSTEM_PROMPT: &str = concat!(
    "Respond with interleaved text and audio. You are a warm, brief voice assistant. ",
    "Chat naturally and answer simple questions yourself in one or two short spoken sentences. ",
    "But when the user asks for real engineering, coding, research, or file/system work, ",
    "do NOT attempt it yourself. Instead, briefly say you'll get your engineer on it, ",
    "and on the TEXT channel output exactly one line of the form: ",
    "DELEGATE: <a clear, self-contained description of the task>. ",
    "Only emit DELEGATE for genuine work, never for small talk."
);

/// One active native voice service for the desktop app.
pub struct VoiceRuntime {
    commands: mpsc::Sender<RuntimeCommand>,
    state: watch::Receiver<RuntimeSnapshot>,
}

impl VoiceRuntime {
    pub fn new() -> Self {
        let (commands, rx) = mpsc::channel(VOICE_COMMAND_CAP);
        let (state_tx, state) = watch::channel(RuntimeSnapshot::default());
        tauri::async_runtime::spawn(kernel_loop(rx, state_tx, commands.clone()));
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
        ctx: SessionCtx,
        settings: VoiceSettings,
        grant: LiveKitGrant,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
    ) -> Result<(), String> {
        self.request(|reply| RuntimeCommand::StartLivekit {
            ctx,
            settings,
            grant,
            channel,
            bridge,
            reply,
        })
        .await
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
}

struct UiEvents {
    tx: mpsc::Sender<VoiceEvent>,
    task: AsyncTask,
}

impl UiEvents {
    fn new(channel: tauri::ipc::Channel<VoiceEvent>) -> UiChannel {
        let (tx, mut rx) = mpsc::channel(UI_EVENT_CAP);
        let task = tauri::async_runtime::spawn(async move {
            while let Some(event) = rx.recv().await {
                if channel.send(event).is_err() {
                    break;
                }
            }
        });
        Arc::new(Self { tx, task })
    }

    fn send(&self, event: VoiceEvent) -> bool {
        self.tx.try_send(event).is_ok()
    }
}

impl Drop for UiEvents {
    fn drop(&mut self) {
        self.task.abort();
    }
}

enum RuntimeCommand {
    StartLfm2 {
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    StartLivekit {
        ctx: SessionCtx,
        settings: VoiceSettings,
        grant: LiveKitGrant,
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
    Reap,
}

async fn kernel_loop(
    mut rx: mpsc::Receiver<RuntimeCommand>,
    state: watch::Sender<RuntimeSnapshot>,
    wake: mpsc::Sender<RuntimeCommand>,
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
                    wake.clone(),
                );
                let _ = reply.send(result);
            }
            RuntimeCommand::StartLivekit {
                ctx,
                settings,
                grant,
                channel,
                bridge,
                reply,
            } => {
                let result = start_livekit_session(
                    &mut session,
                    &threads,
                    ctx,
                    settings,
                    grant,
                    channel,
                    bridge,
                    wake.clone(),
                );
                let _ = reply.send(result);
            }
            RuntimeCommand::Stop { reply } => {
                let result = stop_session(&mut session, &threads).await;
                let _ = reply.send(result);
            }
            RuntimeCommand::Interrupt { reply } => {
                let result = match session.as_ref() {
                    Some(session) => session.interrupt().await,
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::SetMicEnabled { enabled, reply } => {
                let result = match session.as_ref() {
                    Some(session) => session.set_mic_enabled(enabled).await,
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::BeginTypedInput { reply } => {
                let result = match session.as_ref() {
                    Some(session) => match session.set_mic_enabled(false).await {
                        Ok(()) => session.interrupt().await,
                        Err(error) => Err(error),
                    },
                    None => Ok(()),
                };
                let _ = reply.send(result);
            }
            RuntimeCommand::ApplySettings { settings, reply } => {
                let result = apply_settings_to_session(&mut session, &threads, settings).await;
                let _ = reply.send(result);
            }
            RuntimeCommand::InvalidateProvider { provider, reply } => {
                let result = if session
                    .as_ref()
                    .is_some_and(|session| session.provider() == provider)
                {
                    stop_session(&mut session, &threads).await
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
            RuntimeCommand::Reap => {
                reap_finished(&mut session, &threads);
            }
        }
        publish_snapshot(&state, &session);
    }

    let _ = stop_session(&mut session, &threads).await;
    let _ = threads.wait();
}

fn start_lfm2_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
    ctx: SessionCtx,
    settings: VoiceSettings,
    channel: tauri::ipc::Channel<VoiceEvent>,
    bridge: Option<SessionBridgeConfig>,
    wake: mpsc::Sender<RuntimeCommand>,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session.is_some() {
        return Err("Voice is already running.".into());
    }
    threads.wait()?;
    *session = Some(VoiceSession::Lfm2(Lfm2Session::spawn(
        threads, ctx, settings, channel, bridge, wake,
    )?));
    Ok(())
}

fn start_livekit_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
    ctx: SessionCtx,
    settings: VoiceSettings,
    grant: LiveKitGrant,
    channel: tauri::ipc::Channel<VoiceEvent>,
    bridge: Option<SessionBridgeConfig>,
    wake: mpsc::Sender<RuntimeCommand>,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session.is_some() {
        return Err("Voice is already running.".into());
    }
    threads.wait()?;
    *session = Some(VoiceSession::Livekit(LiveKitSession::spawn(
        threads, ctx, settings, grant, channel, bridge, wake,
    )?));
    Ok(())
}

async fn apply_settings_to_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
    settings: VoiceSettings,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session
        .as_ref()
        .is_some_and(|session| !session.matches_settings(&settings))
    {
        return stop_session(session, threads).await;
    }
    Ok(())
}

async fn stop_session(
    session: &mut Option<VoiceSession>,
    threads: &ThreadManager,
) -> Result<(), String> {
    if let Some(session) = session.take() {
        session.stop(threads).await?;
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

fn wake_kernel(commands: &mpsc::Sender<RuntimeCommand>) {
    let _ = commands.try_send(RuntimeCommand::Reap);
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
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SessionSettingsKey {
    provider: VoiceProvider,
    lfm2: Lfm2Settings,
    livekit: Option<LiveKitSettings>,
}

impl SessionSettingsKey {
    fn lfm2(settings: &VoiceSettings) -> Self {
        Self {
            provider: VoiceProvider::Lfm2,
            lfm2: settings.lfm2.clone(),
            livekit: None,
        }
    }

    fn livekit(settings: &VoiceSettings) -> Self {
        Self {
            provider: VoiceProvider::Livekit,
            lfm2: settings.lfm2.clone(),
            livekit: Some(settings.livekit.clone()),
        }
    }

    fn matches(&self, settings: &VoiceSettings) -> bool {
        if settings.provider != self.provider {
            return false;
        }
        if self.provider == VoiceProvider::Lfm2 {
            return self == &Self::lfm2(settings);
        }
        if self.provider == VoiceProvider::Livekit {
            return self == &Self::livekit(settings);
        }
        false
    }
}

enum VoiceSession {
    Lfm2(Lfm2Session),
    Livekit(LiveKitSession),
}

impl VoiceSession {
    fn is_finished(&self) -> bool {
        match self {
            VoiceSession::Lfm2(session) => session.is_finished(),
            VoiceSession::Livekit(session) => session.is_finished(),
        }
    }

    fn provider(&self) -> VoiceProvider {
        match self {
            VoiceSession::Lfm2(_) => VoiceProvider::Lfm2,
            VoiceSession::Livekit(_) => VoiceProvider::Livekit,
        }
    }

    fn session_id(&self) -> &str {
        match self {
            VoiceSession::Lfm2(session) => &session.ctx.session_id,
            VoiceSession::Livekit(session) => &session.ctx.session_id,
        }
    }

    async fn interrupt(&self) -> Result<(), String> {
        match self {
            VoiceSession::Lfm2(session) => session.interrupt(),
            VoiceSession::Livekit(session) => session.interrupt().await,
        }
    }

    async fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        match self {
            VoiceSession::Lfm2(session) => session.set_mic_enabled(enabled),
            VoiceSession::Livekit(session) => session.set_mic_enabled(enabled).await,
        }
    }

    fn mic_enabled(&self) -> bool {
        match self {
            VoiceSession::Lfm2(session) => session.mic_enabled(),
            VoiceSession::Livekit(session) => session.mic_enabled(),
        }
    }

    fn matches_settings(&self, settings: &VoiceSettings) -> bool {
        match self {
            VoiceSession::Lfm2(session) => session.settings.matches(settings),
            VoiceSession::Livekit(session) => session.settings.matches(settings),
        }
    }

    async fn stop(self, threads: &ThreadManager) -> Result<(), String> {
        match self {
            VoiceSession::Lfm2(session) => stop_lfm2(threads, session).await,
            VoiceSession::Livekit(session) => stop_livekit(threads, session).await,
        }
    }
}

async fn stop_lfm2(threads: &ThreadManager, session: Lfm2Session) -> Result<(), String> {
    let (tx, rx) = oneshot::channel();
    threads.spawn("voice-lfm2-stop", move || {
        session.stop();
        let _ = tx.send(());
    })?;
    rx.await
        .map_err(|_| "LFM2 stop task dropped before joining".to_string())?;
    threads.wait()
}

async fn stop_livekit(threads: &ThreadManager, session: LiveKitSession) -> Result<(), String> {
    let (tx, rx) = oneshot::channel();
    threads.spawn("voice-livekit-stop", move || {
        let result = session.stop();
        let _ = tx.send(result);
    })?;
    rx.await
        .map_err(|_| "LiveKit stop task dropped before signalling".to_string())??;
    threads.wait()
}

struct LiveKitSession {
    ctx: SessionCtx,
    settings: SessionSettingsKey,
    commands: mpsc::Sender<LiveKitCommand>,
    done: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    bridge_cancel: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
}

enum LiveKitCommand {
    Interrupt,
    SetMicEnabled(bool),
    Stop,
}

impl LiveKitSession {
    fn spawn(
        threads: &ThreadManager,
        ctx: SessionCtx,
        settings: VoiceSettings,
        grant: LiveKitGrant,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
        wake: mpsc::Sender<RuntimeCommand>,
    ) -> Result<Self, String> {
        let key = SessionSettingsKey::livekit(&settings);
        let channel = UiEvents::new(channel);
        let (commands, rx) = mpsc::channel(LIVEKIT_COMMAND_CAP);
        let done = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let bridge_cancel = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let thread_done = done.clone();
        let thread_finished = finished.clone();
        let thread_bridge_cancel = bridge_cancel.clone();
        let thread_mic = mic_enabled.clone();
        let thread_manager = threads.clone();
        threads.spawn("voice-livekit-session", move || {
            tauri::async_runtime::block_on(livekit_session_loop(
                thread_manager,
                grant,
                settings,
                channel,
                bridge,
                rx,
                thread_done,
                thread_bridge_cancel,
                thread_mic,
                wake.clone(),
            ));
            thread_finished.store(true, Ordering::SeqCst);
            wake_kernel(&wake);
        })?;
        Ok(Self {
            ctx,
            settings: key,
            commands,
            done,
            finished,
            bridge_cancel,
            mic_enabled,
        })
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    async fn interrupt(&self) -> Result<(), String> {
        self.bridge_cancel.store(true, Ordering::SeqCst);
        self.send(LiveKitCommand::Interrupt).await
    }

    async fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        self.send(LiveKitCommand::SetMicEnabled(enabled)).await?;
        self.mic_enabled.store(enabled, Ordering::SeqCst);
        Ok(())
    }

    fn mic_enabled(&self) -> bool {
        self.mic_enabled.load(Ordering::SeqCst)
    }

    fn stop(self) -> Result<(), String> {
        if !self.done.swap(true, Ordering::SeqCst) {
            self.bridge_cancel.store(true, Ordering::SeqCst);
            let _ = self.commands.try_send(LiveKitCommand::Stop);
        }
        Ok(())
    }

    async fn send(&self, command: LiveKitCommand) -> Result<(), String> {
        if self.done.load(Ordering::SeqCst) {
            return Ok(());
        }
        match self.commands.send(command).await {
            Ok(()) => Ok(()),
            Err(_) => {
                self.done.store(true, Ordering::SeqCst);
                Err("LiveKit command queue closed".into())
            }
        }
    }
}

impl Drop for LiveKitSession {
    fn drop(&mut self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        self.bridge_cancel.store(true, Ordering::SeqCst);
        let _ = self.commands.try_send(LiveKitCommand::Stop);
    }
}

async fn livekit_session_loop(
    threads: ThreadManager,
    grant: LiveKitGrant,
    settings: VoiceSettings,
    channel: UiChannel,
    bridge: Option<SessionBridgeConfig>,
    mut rx: mpsc::Receiver<LiveKitCommand>,
    done: Arc<AtomicBool>,
    bridge_cancel: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    wake: mpsc::Sender<RuntimeCommand>,
) {
    if !send_livekit(
        &channel,
        &done,
        VoiceEvent::State {
            state: VoiceState::Loading,
        },
    ) {
        return;
    }
    let vad_threshold = settings.lfm2.vad_threshold;
    let engine = match build_engine(settings, LIVEKIT_AGENT_AUDIO_RATE) {
        Ok(engine) => engine,
        Err(error) => {
            livekit_fail(
                &channel,
                &done,
                format!("Native LiveKit agent model load failed: {error}"),
            );
            return;
        }
    };
    let agent_pipe = match RealtimePipeline::spawn(engine) {
        Ok(pipe) => pipe,
        Err(error) => {
            livekit_fail(
                &channel,
                &done,
                format!("Native LiveKit agent pipeline failed to start: {error}"),
            );
            return;
        }
    };
    let Some(agent_handle) = agent_pipe.handle() else {
        livekit_fail(
            &channel,
            &done,
            "Native LiveKit agent pipeline failed to expose its control handle.".into(),
        );
        return;
    };

    let options = livekit_room_options();
    let agent_options = livekit_room_options();

    let (room, mut events) =
        match livekit_until_done(&done, Room::connect(&grant.url, &grant.token, options)).await {
            Some(Ok(room)) => room,
            Some(Err(error)) => {
                livekit_fail(&channel, &done, format!("LiveKit connect failed: {error}"));
                return;
            }
            None => return,
        };
    let audio = match PlatformAudio::new() {
        Ok(audio) => audio,
        Err(error) => {
            let _ = room.close().await;
            livekit_fail(
                &channel,
                &done,
                format!("LiveKit audio device failed: {error}"),
            );
            return;
        }
    };
    if let Err(error) = configure_livekit_audio(&audio) {
        let _ = room.close().await;
        livekit_fail(
            &channel,
            &done,
            format!("LiveKit audio processing failed: {error}"),
        );
        return;
    }
    let track = LocalAudioTrack::create_audio_track("microphone", audio.rtc_source());
    let user = room.local_participant();
    match livekit_until_done(
        &done,
        user.publish_track(
            LocalTrack::Audio(track.clone()),
            TrackPublishOptions {
                source: TrackSource::Microphone,
                ..Default::default()
            },
        ),
    )
    .await
    {
        Some(Ok(_)) => {}
        Some(Err(error)) => {
            let _ = room.close().await;
            livekit_fail(
                &channel,
                &done,
                format!("LiveKit microphone publish failed: {error}"),
            );
            return;
        }
        None => {
            let _ = room.close().await;
            return;
        }
    }
    let (agent_room, mut agent_events) = match livekit_until_done(
        &done,
        Room::connect(&grant.url, &grant.agent_token, agent_options),
    )
    .await
    {
        Some(Ok(room)) => room,
        Some(Err(error)) => {
            let _ = room.close().await;
            livekit_fail(
                &channel,
                &done,
                format!("Native LiveKit agent connect failed: {error}"),
            );
            return;
        }
        None => {
            let _ = room.close().await;
            return;
        }
    };
    let agent_source = NativeAudioSource::new(
        AudioSourceOptions::default(),
        LIVEKIT_AGENT_AUDIO_RATE,
        LIVEKIT_AGENT_AUDIO_CHANNELS,
        LIVEKIT_AGENT_AUDIO_QUEUE_MS,
    );
    let playback = LiveKitPlaybackReference::new();
    let agent_track = LocalAudioTrack::create_audio_track(
        "assistant",
        RtcAudioSource::Native(agent_source.clone()),
    );
    let agent = agent_room.local_participant();
    match livekit_until_done(
        &done,
        agent.publish_track(
            LocalTrack::Audio(agent_track),
            TrackPublishOptions {
                source: TrackSource::Microphone,
                ..Default::default()
            },
        ),
    )
    .await
    {
        Some(Ok(_)) => {}
        Some(Err(error)) => {
            let _ = agent_room.close().await;
            let _ = room.close().await;
            livekit_fail(
                &channel,
                &done,
                format!("Native LiveKit agent audio publish failed: {error}"),
            );
            return;
        }
        None => {
            let _ = agent_room.close().await;
            let _ = room.close().await;
            return;
        }
    }
    let agent_events_spawned = spawn_native_livekit_agent_events(
        &threads,
        agent_pipe.events().clone(),
        agent_source.clone(),
        channel.clone(),
        done.clone(),
        bridge,
        bridge_cancel.clone(),
        mic_enabled.clone(),
        playback.clone(),
        wake.clone(),
    )
    .map_err(|error| {
        livekit_fail(&channel, &done, error);
    })
    .ok();
    if agent_events_spawned.is_none() {
        let _ = agent_room.close().await;
        let _ = room.close().await;
        return;
    };
    if !mic_enabled.load(Ordering::SeqCst)
        && !livekit_set_mic(&audio, &track, false, &channel, &done)
    {
        let _ = agent_room.close().await;
        let _ = room.close().await;
        return;
    }
    if !reset_livekit_audio_state_or_done(&channel, &done, mic_enabled.load(Ordering::SeqCst)) {
        let _ = room.close().await;
        return;
    }

    let mut ended = false;
    let mut audio_tasks = Vec::<LiveKitMediaTask>::new();
    let mut agent_audio_tasks = Vec::<LiveKitMediaTask>::new();
    let agent_audio_timeout =
        tokio::time::sleep(Duration::from_millis(LIVEKIT_AGENT_AUDIO_TIMEOUT_MS));
    tokio::pin!(agent_audio_timeout);
    let mut done_poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    done_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut remote_audio_seen = false;
    loop {
        tokio::select! {
            _ = done_poll.tick(), if done.load(Ordering::SeqCst) => {
                break;
            }
            _ = &mut agent_audio_timeout, if !remote_audio_seen => {
                livekit_fail(
                    &channel,
                    &done,
                    format!(
                        "Native LiveKit agent audio track did not subscribe within {}s.",
                        LIVEKIT_AGENT_AUDIO_TIMEOUT_MS / 1000,
                    ),
                );
                ended = true;
                break;
            }
            command = rx.recv() => {
                let Some(command) = command else {
                    break;
                };
                if done.load(Ordering::SeqCst) && !matches!(command, LiveKitCommand::Stop) {
                    break;
                }
                match command {
                    LiveKitCommand::Interrupt => {
                        bridge_cancel.store(true, Ordering::SeqCst);
                        agent_handle.interrupt();
                        agent_source.clear_buffer();
                        playback.clear();
                        send_livekit_interrupt(&room).await;
                        if !reset_livekit_audio_state_or_done(
                            &channel,
                            &done,
                            mic_enabled.load(Ordering::SeqCst),
                        )
                        {
                            break;
                        }
                    }
                    LiveKitCommand::SetMicEnabled(enabled) => {
                        mic_enabled.store(enabled, Ordering::SeqCst);
                        if !livekit_set_mic(&audio, &track, enabled, &channel, &done) {
                            break;
                        }
                    }
                    LiveKitCommand::Stop => {
                        break;
                    }
                }
            }
            event = events.recv() => {
                let Some(event) = event else {
                    break;
                };
                if livekit_agent_audio_subscribed(&event, &grant.agent_identity) {
                    remote_audio_seen = true;
                }
                if let Some(reason) = handle_livekit_event(
                    &threads,
                    &channel,
                    event,
                    &mut audio_tasks,
                    &done,
                    &mic_enabled,
                    &grant.agent_identity,
                ) {
                    if let Some(reason) = reason {
                        let _ = send_livekit(
                            &channel,
                            &done,
                            VoiceEvent::Ended {
                                reason: Some(reason),
                            },
                        );
                        ended = true;
                    }
                    break;
                }
            }
            event = agent_events.recv() => {
                let Some(event) = event else {
                    break;
                };
                handle_native_livekit_agent_event(
                    &threads,
                    event,
                    &grant.user_identity,
                    &mut agent_audio_tasks,
                    agent_handle.clone(),
                    channel.clone(),
                    done.clone(),
                    mic_enabled.clone(),
                    playback.clone(),
                    vad_threshold,
                );
            }
        }
    }

    done.store(true, Ordering::SeqCst);
    cancel_livekit_media_tasks(&mut agent_audio_tasks);
    cancel_livekit_media_tasks(&mut audio_tasks);
    let _ = threads.wait();
    drop(agent_pipe);
    let _ = agent_room.close().await;
    let _ = room.close().await;
    if !ended {
        let _ = send_livekit(&channel, &done, VoiceEvent::Ended { reason: None });
    }
}

fn livekit_room_options() -> RoomOptions {
    let mut options = RoomOptions::default();
    options.auto_subscribe = true;
    options.dynacast = true;
    options
}

fn spawn_native_livekit_agent_events(
    threads: &ThreadManager,
    events: Receiver<RealtimeEvent>,
    source: NativeAudioSource,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    bridge: Option<SessionBridgeConfig>,
    bridge_cancel: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    wake: mpsc::Sender<RuntimeCommand>,
) -> Result<(), String> {
    let bridge_threads = threads.clone();
    threads.spawn("voice-livekit-agent-events", move || {
        let mut bridge_state =
            BridgeState::new(bridge_threads, channel.clone(), bridge, bridge_cancel, wake);
        let mut transcript = String::new();
        while !done.load(Ordering::SeqCst) {
            match events.recv_timeout(Duration::from_millis(50)) {
                Ok(RealtimeEvent::Text(text)) => {
                    bridge_state.handle_realtime_text(&text);
                    transcript.push_str(&text);
                    if !send(
                        &channel,
                        VoiceEvent::Transcript {
                            role: Role::Assistant,
                            text: transcript.clone(),
                        },
                    ) {
                        done.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                Ok(RealtimeEvent::Audio(pcm)) => {
                    bridge_state.handle_realtime_audio();
                    playback.observe(&pcm, LIVEKIT_AGENT_AUDIO_RATE);
                    if !send(
                        &channel,
                        VoiceEvent::State {
                            state: VoiceState::Speaking,
                        },
                    ) {
                        done.store(true, Ordering::SeqCst);
                        break;
                    }
                    let rms = rms_f32(&pcm);
                    for frame in livekit_audio_frames(
                        &pcm,
                        LIVEKIT_AGENT_AUDIO_RATE,
                        LIVEKIT_AGENT_AUDIO_CHANNELS,
                    ) {
                        if done.load(Ordering::SeqCst) {
                            break;
                        }
                        if let Err(error) =
                            tauri::async_runtime::block_on(source.capture_frame(&frame))
                        {
                            done.store(true, Ordering::SeqCst);
                            let _ = send(
                                &channel,
                                VoiceEvent::Error {
                                    message: format!(
                                        "Native LiveKit agent audio publish failed: {error}"
                                    ),
                                },
                            );
                            return;
                        }
                    }
                    if !send(&channel, VoiceEvent::Level { rms }) {
                        done.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                Ok(RealtimeEvent::TurnComplete) => {
                    bridge_state.handle_realtime_turn_complete();
                    transcript.clear();
                    if !reset_livekit_audio_state_or_done(
                        &channel,
                        &done,
                        mic_enabled.load(Ordering::SeqCst),
                    ) {
                        break;
                    }
                }
                Ok(RealtimeEvent::Interrupted) => {
                    bridge_state.handle_realtime_interrupted();
                    transcript.clear();
                    source.clear_buffer();
                    playback.clear();
                    if !reset_livekit_audio_state_or_done(
                        &channel,
                        &done,
                        mic_enabled.load(Ordering::SeqCst),
                    ) {
                        break;
                    }
                }
                Ok(RealtimeEvent::Error(message)) => {
                    bridge_state.handle_realtime_error();
                    if !send_livekit(&channel, &done, VoiceEvent::Error { message }) {
                        break;
                    }
                    if !reset_livekit_audio_state_or_done(
                        &channel,
                        &done,
                        mic_enabled.load(Ordering::SeqCst),
                    ) {
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    })
}

struct LiveKitMediaTask {
    cancel: Arc<AtomicBool>,
}

impl LiveKitMediaTask {
    fn cancel(self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

fn cancel_livekit_media_tasks(tasks: &mut Vec<LiveKitMediaTask>) {
    for task in tasks.drain(..) {
        task.cancel();
    }
}

fn livekit_media_cancelled(done: &Arc<AtomicBool>, cancel: &Arc<AtomicBool>) -> bool {
    done.load(Ordering::SeqCst) || cancel.load(Ordering::SeqCst)
}

struct LiveKitPlaybackReference {
    until_ms: AtomicU64,
    rms_bits: AtomicU32,
}

impl LiveKitPlaybackReference {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            until_ms: AtomicU64::new(0),
            rms_bits: AtomicU32::new(0.0f32.to_bits()),
        })
    }

    fn observe(&self, pcm: &[f32], rate: u32) {
        let audio_ms = if rate == 0 {
            0
        } else {
            (pcm.len() as u64).saturating_mul(1000) / rate as u64
        };
        let until = livekit_now_ms()
            .saturating_add(audio_ms)
            .saturating_add(LIVEKIT_AGENT_ECHO_GATE_MS);
        self.rms_bits
            .store(rms_f32(pcm).to_bits(), Ordering::SeqCst);
        self.until_ms.fetch_max(until, Ordering::SeqCst);
    }

    fn clear(&self) {
        self.until_ms.store(0, Ordering::SeqCst);
        self.rms_bits.store(0.0f32.to_bits(), Ordering::SeqCst);
    }

    fn active(&self) -> bool {
        self.until_ms.load(Ordering::SeqCst) > livekit_now_ms()
    }

    fn reference_vad_threshold(&self, base: f32) -> f32 {
        if !self.active() {
            return base;
        }
        let rms = f32::from_bits(self.rms_bits.load(Ordering::SeqCst));
        base.max(rms * LIVEKIT_AGENT_ECHO_MULTIPLIER)
    }
}

fn livekit_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn handle_native_livekit_agent_event(
    threads: &ThreadManager,
    event: RoomEvent,
    user_identity: &str,
    audio_tasks: &mut Vec<LiveKitMediaTask>,
    handle: RealtimePipelineHandle,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
) {
    match event {
        RoomEvent::TrackSubscribed {
            track: RemoteTrack::Audio(track),
            participant,
            ..
        } => {
            if participant.identity().0 != user_identity {
                return;
            }
            match spawn_native_livekit_agent_mic(
                threads,
                handle,
                track,
                channel.clone(),
                done.clone(),
                mic_enabled,
                playback,
                vad_threshold,
            ) {
                Ok(task) => audio_tasks.push(task),
                Err(error) => livekit_fail(&channel, &done, error),
            }
        }
        RoomEvent::TrackUnsubscribed {
            track: RemoteTrack::Audio(_),
            participant,
            ..
        } => {
            if participant.identity().0 != user_identity {
                return;
            }
            cancel_livekit_media_tasks(audio_tasks);
            let _ = threads.reap();
        }
        RoomEvent::ConnectionStateChanged(ConnectionState::Disconnected)
        | RoomEvent::Disconnected { .. } => {
            done.store(true, Ordering::SeqCst);
        }
        _ => {}
    }
}

fn spawn_native_livekit_agent_mic(
    threads: &ThreadManager,
    handle: RealtimePipelineHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
) -> Result<LiveKitMediaTask, String> {
    let cancel = Arc::new(AtomicBool::new(false));
    let thread_cancel = cancel.clone();
    threads.spawn("voice-livekit-agent-mic", move || {
        tauri::async_runtime::block_on(native_livekit_agent_mic_loop(
            handle,
            track,
            channel,
            done,
            mic_enabled,
            playback,
            vad_threshold,
            thread_cancel,
        ));
    })?;
    Ok(LiveKitMediaTask { cancel })
}

async fn native_livekit_agent_mic_loop(
    handle: RealtimePipelineHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
    cancel: Arc<AtomicBool>,
) {
    let mut stream = NativeAudioStream::new(
        track.rtc_track(),
        LIVEKIT_AGENT_AUDIO_RATE_I32,
        LIVEKIT_AGENT_AUDIO_CHANNELS_I32,
    );
    let silence = Duration::from_millis(LIVEKIT_AGENT_SILENCE_MS);
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut speaking = false;
    let mut last_voice = Instant::now();
    let mut samples = Vec::<f32>::new();
    while !livekit_media_cancelled(&done, &cancel) {
        let frame = tokio::select! {
            frame = stream.next() => frame,
            _ = poll.tick() => {
                if livekit_media_cancelled(&done, &cancel) {
                    break;
                }
                continue;
            }
        };
        let Some(frame) = frame else {
            break;
        };
        if !mic_enabled.load(Ordering::SeqCst) {
            speaking = false;
            samples.clear();
            continue;
        }
        let rate = frame.sample_rate;
        if rate == 0 {
            let _ = send_livekit(
                &channel,
                &done,
                VoiceEvent::Error {
                    message: "native LiveKit microphone frame sample rate is zero".into(),
                },
            );
            break;
        }
        let pcm = i16_to_f32(frame.data.as_ref());
        let rms = rms_f32(&pcm);
        let threshold = playback.reference_vad_threshold(vad_threshold);
        if rms > threshold {
            if !speaking {
                speaking = true;
                samples.clear();
            }
            last_voice = Instant::now();
        } else if playback.active() {
            speaking = false;
            samples.clear();
            continue;
        }
        if !speaking {
            continue;
        }
        samples.extend_from_slice(&pcm);
        let too_long = samples.len() >= rate as usize * LIVEKIT_AGENT_MAX_UTTERANCE_SECONDS;
        let silent = last_voice.elapsed() >= silence;
        if !silent && !too_long {
            continue;
        }
        let utt = Utterance {
            samples: std::mem::take(&mut samples),
            rate,
        };
        speaking = false;
        handle.interrupt();
        if !handle.submit(utt) {
            let _ = send_livekit(
                &channel,
                &done,
                VoiceEvent::Error {
                    message: "native LiveKit agent utterance queue is full".into(),
                },
            );
            done.store(true, Ordering::SeqCst);
            break;
        }
        if !send_livekit(
            &channel,
            &done,
            VoiceEvent::State {
                state: VoiceState::Thinking,
            },
        ) {
            break;
        }
    }
}

async fn send_livekit_interrupt(room: &Room) {
    let packet = DataPacket {
        payload: LIVEKIT_CONTROL_INTERRUPT.to_vec(),
        topic: Some(LIVEKIT_CONTROL_TOPIC.to_string()),
        reliable: true,
        ..Default::default()
    };
    let _ = room.local_participant().publish_data(packet).await;
}

fn livekit_set_mic(
    audio: &PlatformAudio,
    track: &LocalAudioTrack,
    enabled: bool,
    channel: &UiChannel,
    done: &Arc<AtomicBool>,
) -> bool {
    if enabled {
        track.enable();
        track.unmute();
        let _ = audio.start_recording();
        return send_livekit(
            channel,
            done,
            VoiceEvent::State {
                state: VoiceState::Listening,
            },
        );
    }
    track.mute();
    track.disable();
    let _ = audio.stop_recording();
    if !send_livekit(
        channel,
        done,
        VoiceEvent::State {
            state: VoiceState::Idle,
        },
    ) {
        return false;
    }
    send_livekit(channel, done, VoiceEvent::Level { rms: 0.0 })
}

fn configure_livekit_audio(audio: &PlatformAudio) -> Result<(), String> {
    audio
        .configure_audio_processing(AudioProcessingOptions {
            echo_cancellation: true,
            noise_suppression: true,
            auto_gain_control: true,
            prefer_hardware_processing: false,
        })
        .map_err(|e| e.to_string())
}

fn handle_livekit_event(
    threads: &ThreadManager,
    channel: &UiChannel,
    event: RoomEvent,
    audio_tasks: &mut Vec<LiveKitMediaTask>,
    done: &Arc<AtomicBool>,
    mic_enabled: &Arc<AtomicBool>,
    agent_identity: &str,
) -> Option<Option<String>> {
    match event {
        RoomEvent::ConnectionStateChanged(ConnectionState::Connected) => {
            if !reset_livekit_audio_state_or_done(channel, done, mic_enabled.load(Ordering::SeqCst))
            {
                return Some(None);
            }
            None
        }
        RoomEvent::ConnectionStateChanged(ConnectionState::Reconnecting) => {
            if !send_livekit(
                channel,
                done,
                VoiceEvent::State {
                    state: VoiceState::Loading,
                },
            ) {
                return Some(None);
            }
            None
        }
        RoomEvent::ConnectionStateChanged(ConnectionState::Disconnected) => Some(None),
        RoomEvent::Disconnected { reason } => Some(Some(format!("{reason:?}"))),
        RoomEvent::TrackSubscribed {
            track, participant, ..
        } => {
            if participant.identity().0 != agent_identity {
                return None;
            }
            if let RemoteTrack::Audio(track) = track {
                match spawn_livekit_audio_monitor(
                    threads,
                    channel.clone(),
                    track,
                    done.clone(),
                    mic_enabled.clone(),
                ) {
                    Ok(task) => audio_tasks.push(task),
                    Err(error) => {
                        livekit_fail(channel, done, error);
                        return Some(None);
                    }
                }
            }
            None
        }
        RoomEvent::TrackUnsubscribed {
            track, participant, ..
        } => {
            if participant.identity().0 != agent_identity {
                return None;
            }
            if matches!(track, RemoteTrack::Audio(_)) {
                if !clear_livekit_audio(
                    threads,
                    channel,
                    done,
                    audio_tasks,
                    mic_enabled.load(Ordering::SeqCst),
                ) {
                    return Some(None);
                }
            }
            None
        }
        _ => None,
    }
}

fn livekit_agent_audio_subscribed(event: &RoomEvent, agent_identity: &str) -> bool {
    match event {
        RoomEvent::TrackSubscribed {
            track: RemoteTrack::Audio(_),
            participant,
            ..
        } => participant.identity().0 == agent_identity,
        _ => false,
    }
}

async fn livekit_until_done<T, E>(
    done: &Arc<AtomicBool>,
    future: impl Future<Output = Result<T, E>>,
) -> Option<Result<T, E>> {
    tokio::pin!(future);
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            result = &mut future => return Some(result),
            _ = poll.tick() => {
                if done.load(Ordering::SeqCst) {
                    return None;
                }
            }
        }
    }
}

fn clear_livekit_audio(
    threads: &ThreadManager,
    channel: &UiChannel,
    done: &Arc<AtomicBool>,
    audio_tasks: &mut Vec<LiveKitMediaTask>,
    mic_enabled: bool,
) -> bool {
    cancel_livekit_media_tasks(audio_tasks);
    let _ = threads.reap();
    reset_livekit_audio_state_or_done(channel, done, mic_enabled)
}

fn reset_livekit_audio_state(channel: &UiChannel, mic_enabled: bool) -> bool {
    if !send(
        channel,
        VoiceEvent::State {
            state: if mic_enabled {
                VoiceState::Listening
            } else {
                VoiceState::Idle
            },
        },
    ) {
        return false;
    }
    send(channel, VoiceEvent::Level { rms: 0.0 })
}

fn reset_livekit_audio_state_or_done(
    channel: &UiChannel,
    done: &Arc<AtomicBool>,
    mic_enabled: bool,
) -> bool {
    if reset_livekit_audio_state(channel, mic_enabled) {
        return true;
    }
    done.store(true, Ordering::SeqCst);
    false
}

fn spawn_livekit_audio_monitor(
    threads: &ThreadManager,
    channel: UiChannel,
    track: RemoteAudioTrack,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
) -> Result<LiveKitMediaTask, String> {
    let cancel = Arc::new(AtomicBool::new(false));
    let thread_cancel = cancel.clone();
    threads.spawn("voice-livekit-audio-monitor", move || {
        tauri::async_runtime::block_on(livekit_audio_monitor_loop(
            channel,
            track,
            done,
            mic_enabled,
            thread_cancel,
        ));
    })?;
    Ok(LiveKitMediaTask { cancel })
}

async fn livekit_audio_monitor_loop(
    channel: UiChannel,
    track: RemoteAudioTrack,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
) {
    let mut stream = NativeAudioStream::new(
        track.rtc_track(),
        LIVEKIT_AUDIO_LEVEL_RATE,
        LIVEKIT_AUDIO_LEVEL_CHANNELS,
    );
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut speaking = false;
    let mut last_voice = Instant::now();
    let silence = Duration::from_millis(LIVEKIT_REMOTE_SILENCE_MS);
    while !livekit_media_cancelled(&done, &cancel) {
        let frame = tokio::select! {
            frame = stream.next() => frame,
            _ = poll.tick() => {
                if livekit_media_cancelled(&done, &cancel) {
                    break;
                }
                continue;
            }
        };
        let Some(frame) = frame else {
            break;
        };
        let rms = rms_i16(frame.data.as_ref());
        if rms > LIVEKIT_REMOTE_SPEAKING_RMS {
            last_voice = Instant::now();
            if !speaking {
                speaking = true;
                if !send(
                    &channel,
                    VoiceEvent::State {
                        state: VoiceState::Speaking,
                    },
                ) {
                    done.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
        if !send(&channel, VoiceEvent::Level { rms }) {
            done.store(true, Ordering::SeqCst);
            break;
        }
        if speaking && last_voice.elapsed() >= silence {
            speaking = false;
            if !reset_livekit_audio_state_or_done(
                &channel,
                &done,
                mic_enabled.load(Ordering::SeqCst),
            ) {
                break;
            }
        }
    }
    let _ = reset_livekit_audio_state_or_done(&channel, &done, mic_enabled.load(Ordering::SeqCst));
}

fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples
        .iter()
        .map(|sample| {
            let v = *sample as f32 / i16::MAX as f32;
            v * v
        })
        .sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| *sample as f32 / i16::MAX as f32)
        .collect()
}

fn rms_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples
        .iter()
        .map(|sample| {
            let v = *sample;
            v * v
        })
        .sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
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

fn send_livekit(channel: &UiChannel, done: &Arc<AtomicBool>, event: VoiceEvent) -> bool {
    if send(channel, event) {
        return true;
    }
    done.store(true, Ordering::SeqCst);
    false
}

fn livekit_fail(channel: &UiChannel, done: &Arc<AtomicBool>, message: String) {
    done.store(true, Ordering::SeqCst);
    let _ = send(
        channel,
        VoiceEvent::Error {
            message: message.clone(),
        },
    );
    let _ = send(
        channel,
        VoiceEvent::Ended {
            reason: Some(message),
        },
    );
}

struct Lfm2Session {
    ctx: SessionCtx,
    settings: SessionSettingsKey,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    bridge_cancel: Arc<AtomicBool>,
    live: Option<Lfm2Runtime>,
}

impl Lfm2Session {
    fn spawn(
        threads: &ThreadManager,
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
        wake: mpsc::Sender<RuntimeCommand>,
    ) -> Result<Self, String> {
        let key = SessionSettingsKey::lfm2(&settings);
        let done = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let bridge_cancel = Arc::new(AtomicBool::new(false));
        let channel = UiEvents::new(channel);
        let sink_done = done.clone();
        let sink_channel = channel.clone();
        let sink_wake = wake.clone();
        let bridge_threads = threads.clone();
        let bridge_wake = sink_wake.clone();
        let mut bridge_state = BridgeState::new(
            bridge_threads,
            sink_channel.clone(),
            bridge,
            bridge_cancel.clone(),
            bridge_wake,
        );
        let cfg = RuntimeConfig {
            vad_threshold: settings.lfm2.vad_threshold,
            can_interrupt: true,
            ..RuntimeConfig::default()
        };
        let (live, main) = Lfm2Runtime::prepare(
            cfg,
            move |out_rate| build_engine(settings, out_rate),
            move |event| {
                if matches!(event, RuntimeEvent::Ended(_)) {
                    sink_done.store(true, Ordering::SeqCst);
                    wake_kernel(&sink_wake);
                }
                let sent = bridge_state.handle(event);
                if !sent {
                    sink_done.store(true, Ordering::SeqCst);
                    wake_kernel(&sink_wake);
                }
                sent
            },
        );
        let thread_finished = finished.clone();
        threads.spawn("voice-session", move || {
            main();
            thread_finished.store(true, Ordering::SeqCst);
            wake_kernel(&wake);
        })?;
        Ok(Self {
            ctx,
            settings: key,
            channel,
            done,
            finished,
            bridge_cancel,
            live: Some(live),
        })
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
            || self.live.as_ref().is_some_and(Lfm2Runtime::is_finished)
    }

    fn interrupt(&self) -> Result<(), String> {
        self.bridge_cancel.store(true, Ordering::SeqCst);
        if let Some(live) = self.live.as_ref() {
            live.interrupt();
        }
        self.emit_ready()
    }

    fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        if let Some(live) = self.live.as_ref() {
            live.set_mic_enabled(enabled);
        }
        self.emit_ready()
    }

    fn emit_ready(&self) -> Result<(), String> {
        let state_sent = send(
            &self.channel,
            VoiceEvent::State {
                state: if self.mic_enabled() {
                    VoiceState::Listening
                } else {
                    VoiceState::Idle
                },
            },
        );
        let level_sent = send(&self.channel, VoiceEvent::Level { rms: 0.0 });
        if !state_sent || !level_sent {
            self.done.store(true, Ordering::SeqCst);
            return Err("voice event channel closed".into());
        }
        Ok(())
    }

    fn mic_enabled(&self) -> bool {
        self.live.as_ref().is_some_and(Lfm2Runtime::mic_enabled)
    }

    fn stop(mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.bridge_cancel.store(true, Ordering::SeqCst);
        if let Some(live) = self.live.take() {
            live.stop();
        }
    }
}

impl Drop for Lfm2Session {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.bridge_cancel.store(true, Ordering::SeqCst);
        if let Some(live) = self.live.take() {
            live.stop();
        }
    }
}

struct BridgeState {
    threads: ThreadManager,
    channel: UiChannel,
    bridge: Option<SessionBridgeConfig>,
    cancel: Arc<AtomicBool>,
    wake: mpsc::Sender<RuntimeCommand>,
    task: Option<DelegateTask>,
    transcript: String,
    speaking: bool,
}

struct DelegateTask {
    done: Arc<AtomicBool>,
}

struct DelegateDone {
    done: Arc<AtomicBool>,
    wake: mpsc::Sender<RuntimeCommand>,
}

impl Drop for DelegateDone {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        wake_kernel(&self.wake);
    }
}

impl BridgeState {
    fn new(
        threads: ThreadManager,
        channel: UiChannel,
        bridge: Option<SessionBridgeConfig>,
        cancel: Arc<AtomicBool>,
        wake: mpsc::Sender<RuntimeCommand>,
    ) -> Self {
        Self {
            threads,
            channel,
            bridge,
            cancel,
            wake,
            task: None,
            transcript: String::new(),
            speaking: false,
        }
    }

    fn handle(&mut self, event: RuntimeEvent) -> bool {
        match &event {
            RuntimeEvent::Transcript(text) => {
                self.transcript = text.clone();
            }
            RuntimeEvent::State(SessionState::Speaking) => {
                self.speaking = true;
            }
            RuntimeEvent::State(SessionState::Listening | SessionState::Idle) => {
                if self.speaking {
                    self.finish_turn();
                }
            }
            RuntimeEvent::Ended(_) | RuntimeEvent::Error(_) => {
                self.cancel.store(true, Ordering::SeqCst);
                self.clear_turn();
            }
            RuntimeEvent::Audio { .. } | RuntimeEvent::Level(_) | RuntimeEvent::State(_) => {}
        }
        send_runtime(&self.channel, event)
    }

    fn handle_realtime_text(&mut self, text: &str) {
        self.transcript.push_str(text);
        self.speaking = true;
    }

    fn handle_realtime_audio(&mut self) {
        self.speaking = true;
    }

    fn handle_realtime_turn_complete(&mut self) {
        self.finish_turn();
    }

    fn handle_realtime_interrupted(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.clear_turn();
    }

    fn handle_realtime_error(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.clear_turn();
    }

    fn finish_turn(&mut self) {
        if !self.speaking && self.transcript.trim().is_empty() {
            self.transcript.clear();
            return;
        }
        self.speaking = false;
        self.maybe_delegate();
    }

    fn clear_turn(&mut self) {
        self.speaking = false;
        self.transcript.clear();
    }

    fn maybe_delegate(&mut self) {
        self.reap_delegate();
        let Some(task) = delegate_task(&self.transcript) else {
            self.transcript.clear();
            return;
        };
        self.transcript.clear();
        let Some(cfg) = self.bridge.clone() else {
            return;
        };
        if self.task.is_some() {
            return;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let cancel = self.cancel.clone();
        let channel = self.channel.clone();
        let done = Arc::new(AtomicBool::new(false));
        let thread_done = done.clone();
        let thread_wake = self.wake.clone();
        let spawn = self.threads.spawn("voice-delegate-turn", move || {
            tauri::async_runtime::block_on(async move {
                let _done = DelegateDone {
                    done: thread_done,
                    wake: thread_wake,
                };
                if !send_or_cancel(
                    &channel,
                    &cancel,
                    VoiceEvent::State {
                        state: VoiceState::Thinking,
                    },
                ) {
                    return;
                }
                let delta_channel = channel.clone();
                let delta_cancel = cancel.clone();
                let mut reply = String::new();
                let result = run_turn(cfg, task, cancel.clone(), move |event| match event {
                    SessionBridgeEvent::Delta { text, .. } => {
                        reply.push_str(&text);
                        send_or_cancel(
                            &delta_channel,
                            &delta_cancel,
                            VoiceEvent::Transcript {
                                role: Role::Assistant,
                                text: reply.clone(),
                            },
                        )
                    }
                    SessionBridgeEvent::Done => true,
                })
                .await;
                if let Err(error) = result {
                    let _ = send_or_cancel(&channel, &cancel, VoiceEvent::Error { message: error });
                }
                if !cancel.load(Ordering::SeqCst) {
                    let _ = send_or_cancel(
                        &channel,
                        &cancel,
                        VoiceEvent::State {
                            state: VoiceState::Listening,
                        },
                    );
                }
            });
        });
        match spawn {
            Ok(()) => {
                self.task = Some(DelegateTask { done });
            }
            Err(message) => {
                done.store(true, Ordering::SeqCst);
                let _ = send_or_cancel(&self.channel, &self.cancel, VoiceEvent::Error { message });
            }
        }
    }

    fn reap_delegate(&mut self) {
        if self
            .task
            .as_ref()
            .is_some_and(|task| task.done.load(Ordering::SeqCst))
        {
            let _ = self.task.take();
            let _ = self.threads.reap();
        }
    }
}

impl Drop for BridgeState {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

fn delegate_task(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let line = line.trim_start();
        let prefix = line.get(.."DELEGATE:".len())?;
        if !prefix.eq_ignore_ascii_case("DELEGATE:") {
            return None;
        }
        let task = line.get("DELEGATE:".len()..)?.trim();
        (!task.is_empty()).then(|| task.to_string())
    })
}

fn send(channel: &UiChannel, event: VoiceEvent) -> bool {
    channel.send(event)
}

fn send_or_cancel(channel: &UiChannel, cancel: &Arc<AtomicBool>, event: VoiceEvent) -> bool {
    if send(channel, event) {
        return true;
    }
    cancel.store(true, Ordering::SeqCst);
    false
}

fn send_runtime(channel: &UiChannel, event: RuntimeEvent) -> bool {
    match event {
        RuntimeEvent::State(state) => send(
            channel,
            VoiceEvent::State {
                state: match state {
                    SessionState::Loading => VoiceState::Loading,
                    SessionState::Listening => VoiceState::Listening,
                    SessionState::Thinking => VoiceState::Thinking,
                    SessionState::Speaking => VoiceState::Speaking,
                    SessionState::Idle => VoiceState::Idle,
                },
            },
        ),
        RuntimeEvent::Transcript(text) => send(
            channel,
            VoiceEvent::Transcript {
                role: Role::Assistant,
                text,
            },
        ),
        RuntimeEvent::Level(rms) => send(channel, VoiceEvent::Level { rms }),
        RuntimeEvent::Audio { .. } => true,
        RuntimeEvent::Ended(reason) => send(channel, VoiceEvent::Ended { reason }),
        RuntimeEvent::Error(message) => send(channel, VoiceEvent::Error { message }),
    }
}

fn build_engine(settings: VoiceSettings, out_rate: u32) -> Result<Lfm2VoiceEngine, String> {
    // Fail-hard, no network at start: load ONLY a local snapshot dir. The repo id/revision are
    // the download source (Settings → Download), not a run-time fetch — never auto-download here.
    let dir = settings::lfm2_active_model_dir(&settings.lfm2).ok_or_else(|| {
        "No local LFM2-Audio model — download a model or select a model directory in Settings."
            .to_string()
    })?;
    let device = select_device(&settings.lfm2.device)?;
    let codebooks = codebooks(&dir)?;
    let (model, proc) =
        from_pretrained(&dir, &device).map_err(|e| format!("failed to load LFM2-Audio: {e}"))?;
    let params = GenParams {
        max_new_tokens: settings.lfm2.max_tokens as usize,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed: settings.lfm2.seed.unwrap_or(0),
    };
    let engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
    if settings.lfm2.delegate.enabled
        && settings
            .lfm2
            .delegate
            .target
            .as_deref()
            .is_some_and(|target| !target.trim().is_empty())
    {
        return Ok(engine.with_system_prompt(LFM2_CONVERSE_SYSTEM_PROMPT));
    }
    Ok(engine)
}

fn select_device(device: &Lfm2Device) -> Result<Device, String> {
    match device {
        Lfm2Device::Cpu => {
            if liquid_audio::bf16_gemm::bf16_gemm_available() {
                Ok(Device::Cpu)
            } else {
                Err(
                    "CPU LFM2 voice requires the NEON BF16 matmul kernel; choose Metal on this Mac."
                        .into(),
                )
            }
        }
        Lfm2Device::Metal => {
            #[cfg(target_os = "macos")]
            {
                Device::new_metal(0).map_err(|e| format!("failed to open Metal device: {e}"))
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err("Metal voice inference is only available on macOS.".into())
            }
        }
    }
}

fn codebooks(dir: &Path) -> Result<usize, String> {
    let config = fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("failed to read model config: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&config).map_err(|e| format!("failed to parse model config: {e}"))?;
    json["codebooks"]
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| "model config is missing `codebooks`".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn thread_manager_reaps_finished_threads_and_keeps_live_threads() {
        let threads = ThreadManager::default();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        threads
            .spawn("voice-test-finished", move || {
                done_tx.send(()).unwrap();
            })
            .unwrap();
        threads
            .spawn("voice-test-live", move || {
                release_rx.recv().unwrap();
            })
            .unwrap();

        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        threads.reap().unwrap();
        assert_eq!(threads.tracked_len(), 1);

        release_tx.send(()).unwrap();
        threads.wait().unwrap();
        assert_eq!(threads.tracked_len(), 0);
    }

    #[test]
    fn thread_manager_wait_does_not_join_the_calling_thread() {
        let threads = ThreadManager::default();
        let nested = threads.clone();
        let barrier = Arc::new(Barrier::new(2));
        let thread_barrier = barrier.clone();
        let (done_tx, done_rx) = std::sync::mpsc::channel();

        threads
            .spawn("voice-test-self-wait", move || {
                thread_barrier.wait();
                nested.wait().unwrap();
                done_tx.send(()).unwrap();
            })
            .unwrap();

        barrier.wait();
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        threads.wait().unwrap();
        assert_eq!(threads.tracked_len(), 0);
    }

    #[test]
    fn livekit_stop_does_not_block_when_provider_queue_is_full() {
        let (commands, _rx) = mpsc::channel(LIVEKIT_COMMAND_CAP);
        for _ in 0..LIVEKIT_COMMAND_CAP {
            assert!(commands.try_send(LiveKitCommand::Interrupt).is_ok());
        }
        let settings = VoiceSettings {
            provider: VoiceProvider::Livekit,
            ..VoiceSettings::default()
        };
        let done = Arc::new(AtomicBool::new(false));
        let bridge_cancel = Arc::new(AtomicBool::new(false));
        let session = LiveKitSession {
            ctx: SessionCtx {
                session_id: "test-session".into(),
                directory: "/tmp".into(),
                agent: None,
                model: None,
                variant: None,
                prompt_mode: None,
            },
            settings: SessionSettingsKey::livekit(&settings),
            commands,
            done: done.clone(),
            finished: Arc::new(AtomicBool::new(false)),
            bridge_cancel: bridge_cancel.clone(),
            mic_enabled: Arc::new(AtomicBool::new(true)),
        };

        let start = Instant::now();
        session.stop().unwrap();

        assert!(start.elapsed() < Duration::from_millis(100));
        assert!(done.load(Ordering::SeqCst));
        assert!(bridge_cancel.load(Ordering::SeqCst));
    }
}
