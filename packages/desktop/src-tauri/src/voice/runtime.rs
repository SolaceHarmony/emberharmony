//! Tauri-owned voice runtime.
//!
//! The audio service itself lives in `liquid_audio::VoiceRuntime`: the Tauri layer
//! feeds local microphone frames from WebRTC/PlatformAudio and routes local speaker
//! PCM through WebRTC/PlatformAudio, while VAD, barge-in, and realtime inference run on Rust
//! threads. This module is the desktop kernel wrapper: it loads Tauri settings,
//! builds the LFM2 engine, owns the single active session, and maps runtime events
//! onto the webview `Channel`.

use std::{
    borrow::Cow,
    collections::HashMap,
    fs,
    future::Future,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use candle_core::Device;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use futures::StreamExt;
use liquid_audio::moshi::models::{realtime_moshi_files, safetensors_floating_dtype};
use liquid_audio::{
    AudioStatsSnapshot, ConversationVault, ExternalAudioInput, ExternalAudioInputWriter,
    ExternalAudioOutput, FrameSubmitError, GenParams, LFM2AudioModel, LFM2AudioProcessor,
    Lfm2VoiceEngine, MoshiVoiceEngine, RealtimeFramePipeline, RealtimeFramePipelineHandle,
    RealtimePipeline, RealtimePipelineHandle, RuntimeConfig, RuntimeEvent, SessionState, Utterance,
    VoiceEngine, VoiceEvent as RealtimeEvent, VoiceRuntime as Lfm2Runtime, from_pretrained,
};
use livekit::{
    AudioProcessingOptions, ConnectionState, DataPacket, PlatformAudio, Room, RoomEvent,
    RoomOptions,
    options::TrackPublishOptions,
    rtc_engine::lk_runtime::LkRuntime,
    track::{LocalAudioTrack, LocalTrack, RemoteAudioTrack, RemoteTrack, TrackSource},
    webrtc::audio_stream::native::{NativeAudioStream, NativeAudioStreamOptions},
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
    self, Lfm2Device, Lfm2Settings, LiveKitSettings, LocalVoiceEngine, VoiceProvider, VoiceSettings,
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
const LOCAL_WEBRTC_MIC_QUEUE_FRAMES: usize = 4;
const LOCAL_WEBRTC_MIC_READY_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_READY_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP: usize = 16;
const LOCAL_WEBRTC_INPUT_STREAM_ID: &str = "local-lfm2-input";
const LOCAL_WEBRTC_OUTPUT_STREAM_ID: &str = "local-lfm2-output";
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

/// Resident LFM2 weights (spec 09): loading 1.5B params takes seconds, so the loaded
/// model lives for the app lifetime and every voice session borrows it via `Arc` —
/// building an engine must never mean loading the model again. Keyed by (model dir,
/// device); changing either in Settings reloads once. The ~3GB residency is the price
/// of instant session starts.
struct ResidentLfm2 {
    dir: PathBuf,
    device_setting: Lfm2Device,
    /// The ONE candle Device for the app's lifetime. candle Metal state is
    /// per-`MetalDevice` INSTANCE (command queue, built-in kernel cache, buffer
    /// pool) — minting a fresh `Device::new_metal(0)` per session start left the
    /// resident weights on the first instance while sessions ran on new ones:
    /// candle recompiled its kernels every session and two live queues could
    /// interleave. The compiled-kernel context is resident state, same as the
    /// weights.
    device: Device,
    model: Arc<LFM2AudioModel>,
    proc: Arc<LFM2AudioProcessor>,
}

static LFM2_RESIDENT: Mutex<Option<ResidentLfm2>> = Mutex::new(None);

/// One conversation vault per chat session: the model's conversation must survive
/// UI-driven voice session rebuilds (route changes, settings writes) — the upstream
/// `chat.py` invariant, where `ChatState` outlives everything but an explicit reset.
static CONVERSATION_VAULTS: Mutex<Option<HashMap<String, ConversationVault>>> = Mutex::new(None);

fn conversation_vault(session_id: &str) -> ConversationVault {
    let mut slot = CONVERSATION_VAULTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    slot.get_or_insert_with(HashMap::new)
        .entry(session_id.to_string())
        .or_default()
        .clone()
}

fn resident_lfm2(
    dir: &Path,
    device_setting: &Lfm2Device,
) -> Result<(Arc<LFM2AudioModel>, Arc<LFM2AudioProcessor>, Device), String> {
    // Double-checked resident init (#148). Hold the lock ONLY to check for a hit
    // and clone handles out — never across `from_pretrained`, a multi-second
    // Metal load. Holding it there stalled anything else touching the slot and
    // pinned a tokio worker for the whole load.
    {
        let slot = LFM2_RESIDENT
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(resident) = slot.as_ref() {
            if resident.dir == dir && &resident.device_setting == device_setting {
                // Per-session ground truth in the log: which silicon this session's
                // model actually lives on (the settings store can say one thing and
                // a stale resident another — this line is the arbiter).
                eprintln!(
                    "[voice] LFM2 session: setting {:?} -> resident model reused on {:?}",
                    device_setting,
                    resident.device.location()
                );
                return Ok((
                    resident.model.clone(),
                    resident.proc.clone(),
                    resident.device.clone(),
                ));
            }
            eprintln!(
                "[voice] LFM2 session: setting changed ({:?} -> {:?}); reloading model",
                resident.device_setting, device_setting
            );
        }
    } // lock released before the load

    // First load (or dir/device change), OUTSIDE the lock: create the device HERE
    // so it lives and dies with the resident weights — one queue, one kernel
    // cache, one pool.
    let device = select_device(device_setting)?;
    eprintln!(
        "[voice] LFM2 session: setting {:?} -> loading model onto {:?}",
        device_setting,
        device.location()
    );
    let (model, proc) =
        from_pretrained(dir, &device).map_err(|e| format!("failed to load LFM2-Audio: {e}"))?;
    let (model, proc) = (Arc::new(model), Arc::new(proc));

    // Re-lock to publish. Double-check: a concurrent start may have loaded the
    // SAME dir+device while we were loading — if so, adopt theirs and drop ours,
    // so the app keeps exactly ONE resident device instance (the whole point of
    // the resident cache). Serialized starts never hit this; the check makes the
    // rare race correct instead of leaking a second device.
    let mut slot = LFM2_RESIDENT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(resident) = slot.as_ref() {
        if resident.dir == dir && &resident.device_setting == device_setting {
            eprintln!(
                "[voice] LFM2 session: concurrent load raced; adopting the resident model on {:?}",
                resident.device.location()
            );
            return Ok((
                resident.model.clone(),
                resident.proc.clone(),
                resident.device.clone(),
            ));
        }
    }
    *slot = Some(ResidentLfm2 {
        dir: dir.to_path_buf(),
        device_setting: device_setting.clone(),
        device: device.clone(),
        model: model.clone(),
        proc: proc.clone(),
    });
    Ok((model, proc, device))
}

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
    pub audio_stats: Option<AudioStatsSnapshot>,
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
                )
                .await;
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

async fn start_lfm2_session(
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
    *session = Some(VoiceSession::Lfm2(
        Lfm2Session::spawn(threads, ctx, settings, channel, bridge, wake).await?,
    ));
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
        audio_stats: session.audio_stats(),
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

    fn audio_stats(&self) -> Option<AudioStatsSnapshot> {
        match self {
            VoiceSession::Lfm2(session) => session.audio_stats(),
            VoiceSession::Livekit(_) => None,
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
    let interrupts_playback = can_interrupt_playback(&settings);
    let engine = match build_engine(settings, LIVEKIT_AGENT_AUDIO_RATE, None) {
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
    let agent_pipe = match NativeAgentPipe::spawn(engine) {
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
        agent_pipe.events(),
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
                    interrupts_playback,
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

enum NativeAgentPipe {
    Turn(RealtimePipeline),
    Frame(RealtimeFramePipeline),
}

impl NativeAgentPipe {
    fn spawn(engine: Box<dyn VoiceEngine>) -> Result<Self, String> {
        if engine.frame_config().is_some() {
            return RealtimeFramePipeline::spawn(engine).map(Self::Frame);
        }
        RealtimePipeline::spawn(engine).map(Self::Turn)
    }

    fn events(&self) -> Receiver<RealtimeEvent> {
        match self {
            Self::Turn(pipe) => pipe.events().clone(),
            Self::Frame(pipe) => pipe.events().clone(),
        }
    }

    fn handle(&self) -> Option<NativeAgentHandle> {
        match self {
            Self::Turn(pipe) => pipe.handle().map(NativeAgentHandle::Turn),
            Self::Frame(pipe) => pipe.handle().map(NativeAgentHandle::Frame),
        }
    }
}

#[derive(Clone)]
enum NativeAgentHandle {
    Turn(RealtimePipelineHandle),
    Frame(RealtimeFramePipelineHandle),
}

impl NativeAgentHandle {
    fn interrupt(&self) {
        match self {
            Self::Turn(handle) => handle.interrupt(),
            Self::Frame(handle) => handle.interrupt(),
        }
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
    handle: NativeAgentHandle,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
    can_interrupt_playback: bool,
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
                can_interrupt_playback,
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
    handle: NativeAgentHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
    can_interrupt_playback: bool,
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
            can_interrupt_playback,
            thread_cancel,
        ));
    })?;
    Ok(LiveKitMediaTask { cancel })
}

async fn native_livekit_agent_mic_loop(
    handle: NativeAgentHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
    can_interrupt_playback: bool,
    cancel: Arc<AtomicBool>,
) {
    match handle {
        NativeAgentHandle::Turn(handle) => {
            native_livekit_agent_turn_mic_loop(
                handle,
                track,
                channel,
                done,
                mic_enabled,
                playback,
                vad_threshold,
                can_interrupt_playback,
                cancel,
            )
            .await;
        }
        NativeAgentHandle::Frame(handle) => {
            native_livekit_agent_frame_mic_loop(
                handle,
                track,
                channel,
                done,
                mic_enabled,
                playback,
                vad_threshold,
                can_interrupt_playback,
                cancel,
            )
            .await;
        }
    }
}

async fn native_livekit_agent_turn_mic_loop(
    handle: RealtimePipelineHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback: Arc<LiveKitPlaybackReference>,
    vad_threshold: f32,
    can_interrupt_playback: bool,
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
        if playback.active() && !can_interrupt_playback {
            speaking = false;
            samples.clear();
            continue;
        }
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
            handle.interrupt();
            samples.clear();
            if !send_livekit(
                &channel,
                &done,
                VoiceEvent::State {
                    state: VoiceState::Listening,
                },
            ) {
                break;
            }
            continue;
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

async fn native_livekit_agent_frame_mic_loop(
    handle: RealtimeFramePipelineHandle,
    track: RemoteAudioTrack,
    channel: UiChannel,
    done: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    _playback: Arc<LiveKitPlaybackReference>,
    _vad_threshold: f32,
    _can_interrupt_playback: bool,
    cancel: Arc<AtomicBool>,
) {
    let mut stream = NativeAudioStream::new(
        track.rtc_track(),
        LIVEKIT_AGENT_AUDIO_RATE_I32,
        LIVEKIT_AGENT_AUDIO_CHANNELS_I32,
    );
    let frame = handle.config();
    let mut resampler = LiveKitFrameResampler::new(LIVEKIT_AGENT_AUDIO_RATE, frame.sample_rate);
    let mut model = Vec::with_capacity(frame.frame_size * 2);
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let interval = Duration::from_secs_f64(frame.frame_size as f64 / frame.sample_rate as f64);
    let mut silence = tokio::time::interval(interval);
    silence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    silence.tick().await;
    let mut next_silence = Instant::now() + interval;
    let mut backpressure_reported = false;
    while !livekit_media_cancelled(&done, &cancel) {
        let (source, audio) = tokio::select! {
            frame = stream.next() => (0u8, frame),
            _ = poll.tick() => {
                if livekit_media_cancelled(&done, &cancel) {
                    break;
                }
                (1u8, None)
            }
            _ = silence.tick() => {
                if livekit_media_cancelled(&done, &cancel) {
                    break;
                }
                (2u8, None)
            }
        };
        if source == 1 {
            continue;
        }
        match source {
            2 => {
                if !mic_enabled.load(Ordering::SeqCst) {
                    continue;
                }
                let now = Instant::now();
                if now < next_silence {
                    continue;
                }
                pad_next_livekit_model_frame(&mut model, frame.frame_size);
                next_silence = now + interval;
            }
            _ => {
                let Some(audio) = audio else {
                    break;
                };
                if !mic_enabled.load(Ordering::SeqCst) {
                    // Mic paused — stop feeding frames but keep the loop alive.
                    // Don't clear the model buffer; when mic resumes, we pick up
                    // the live stream. Moshi handles silence frames natively.
                    continue;
                }
                if audio.sample_rate == 0 {
                    let _ = send_livekit(
                        &channel,
                        &done,
                        VoiceEvent::Error {
                            message: "native LiveKit microphone frame sample rate is zero".into(),
                        },
                    );
                    break;
                }
                let pcm = i16_to_f32(audio.data.as_ref());
                // Full-duplex Moshi semantics: always feed real mic audio to the model.
                // No VAD gating, no mic zeroing, no barge-in resets. The multistream LM
                // handles turn-taking natively. Echo/AEC is handled by WebRTC's
                // PlatformAudio (AEC/NS/AGC), not by zeroing model input.
                // Explicit Stop/Interrupt are session-level controls only.
                let rate = audio.sample_rate as u32;
                if resampler.from() != rate {
                    resampler = LiveKitFrameResampler::new(rate, frame.sample_rate);
                    model.clear();
                }
                model.extend(resampler.process(&pcm));
            }
        }
        while model.len() >= frame.frame_size {
            let next = model[..frame.frame_size].to_vec();
            model.drain(..frame.frame_size);
            match handle.try_submit_frame(next) {
                Ok(()) => {
                    backpressure_reported = false;
                    next_silence = Instant::now() + interval;
                    continue;
                }
                Err(FrameSubmitError::Full) => {
                    // Queue pressure is timing pressure, not a semantic
                    // interruption. Keep the Moshi stream state intact and
                    // drop buffered capture until inference catches up.
                    model.clear();
                    if !backpressure_reported {
                        backpressure_reported = true;
                        if !send_livekit(
                            &channel,
                            &done,
                            VoiceEvent::State {
                                state: VoiceState::Listening,
                            },
                        ) {
                            break;
                        }
                    }
                }
                Err(FrameSubmitError::Disconnected | FrameSubmitError::WrongSize) => break,
            }
            break;
        }
        if model.len() > frame.frame_size * 4 {
            model.clear();
        }
    }
}

fn pad_next_livekit_model_frame(model: &mut Vec<f32>, frame_size: usize) {
    if frame_size == 0 {
        return;
    }
    let partial = model.len() % frame_size;
    let needed = if partial == 0 {
        frame_size
    } else {
        frame_size - partial
    };
    model.resize(model.len() + needed, 0.0);
}

struct LiveKitFrameResampler {
    from: u32,
    to: u32,
    carry: Option<f32>,
}

impl LiveKitFrameResampler {
    fn new(from: u32, to: u32) -> Self {
        Self {
            from,
            to,
            carry: None,
        }
    }

    fn from(&self) -> u32 {
        self.from
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() || self.from == 0 || self.to == 0 {
            return Vec::new();
        }
        if self.from == self.to {
            return input.to_vec();
        }
        if self.from == self.to.saturating_mul(2) {
            return self.downsample_by_two(input);
        }
        self.carry = input.last().copied();
        liquid_audio::resample::resample_slice(input, self.from, self.to)
    }

    fn downsample_by_two(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len() / 2 + 1);
        let mut start = 0usize;
        if let Some(prev) = self.carry.take() {
            if let Some(&sample) = input.first() {
                out.push((prev + sample) * 0.5);
                start = 1;
            }
        }
        let pairs = &input[start..];
        for pair in pairs.chunks_exact(2) {
            out.push((pair[0] + pair[1]) * 0.5);
        }
        if pairs.len() % 2 == 1 {
            self.carry = pairs.last().copied();
        }
        out
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
    async fn spawn(
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
        let cfg = local_runtime_config(&settings);
        let vault = conversation_vault(&ctx.session_id);
        let input = start_local_webrtc_input(threads, done.clone()).await?;
        let output = match start_local_webrtc_output().await {
            Ok(output) => output,
            Err(error) => {
                done.store(true, Ordering::SeqCst);
                return Err(error);
            }
        };
        let (live, main) = Lfm2Runtime::prepare_with_io(
            cfg,
            Some(input),
            Some(output),
            move |out_rate| build_engine(settings, out_rate, Some(vault)),
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
        let thread_wake = wake.clone();
        match threads.spawn("voice-session", move || {
            main();
            thread_finished.store(true, Ordering::SeqCst);
            wake_kernel(&thread_wake);
        }) {
            Ok(()) => {}
            Err(error) => {
                done.store(true, Ordering::SeqCst);
                live.stop();
                return Err(error);
            }
        }
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

    fn audio_stats(&self) -> Option<AudioStatsSnapshot> {
        self.live.as_ref().map(Lfm2Runtime::audio_stats)
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

fn local_runtime_config(settings: &VoiceSettings) -> RuntimeConfig {
    RuntimeConfig {
        vad_threshold: settings.lfm2.vad_threshold,
        can_interrupt: can_interrupt_playback(settings),
        ..RuntimeConfig::default()
    }
}

fn can_interrupt_playback(settings: &VoiceSettings) -> bool {
    settings.lfm2.engine != LocalVoiceEngine::MoshiRealtime
}

fn build_engine(
    settings: VoiceSettings,
    out_rate: u32,
    vault: Option<ConversationVault>,
) -> Result<Box<dyn VoiceEngine>, String> {
    // Fail-hard, no network at start: load ONLY a local snapshot dir. The repo id/revision are
    // the download source (Settings → Download), not a run-time fetch — never auto-download here.
    if settings.lfm2.engine == LocalVoiceEngine::MoshiRealtime {
        let device = select_device(&settings.lfm2.device)?;
        let dir = settings::moshi_model_dir(&settings.lfm2).ok_or_else(|| {
            "No local Moshi realtime model — download Moshiko or select a Moshi snapshot directory in Settings."
                .to_string()
        })?;
        let files = realtime_moshi_files(&dir)
            .map_err(|e| format!("failed to inspect Moshi snapshot: {e}"))?
            .ok_or_else(|| {
                "Selected Moshi directory is not a realtime Moshi snapshot.".to_string()
            })?;
        let dtype = safetensors_floating_dtype(&files.moshi_weights)
            .map_err(|e| format!("failed to read Moshi checkpoint dtype: {e}"))?;
        let engine = MoshiVoiceEngine::from_files(
            &files,
            dtype,
            &device,
            files
                .params
                .with_seed(settings.lfm2.seed.unwrap_or(files.params.seed)),
            out_rate,
        )
        .map_err(|e| format!("failed to load Moshi realtime model: {e}"))?;
        return Ok(Box::new(engine));
    }
    let dir = settings::lfm2_active_model_dir(&settings.lfm2).ok_or_else(|| {
        "No local LFM2-Audio model — download a model or select a model directory in Settings."
            .to_string()
    })?;
    if realtime_moshi_files(&dir)
        .map_err(|e| format!("failed to inspect local voice model: {e}"))?
        .is_some()
    {
        return Err("Selected local engine is LFM2-Audio, but the directory contains a Moshi realtime snapshot. Switch Local engine to Moshi realtime or choose an LFM2-Audio snapshot.".into());
    }
    let codebooks = codebooks(&dir)?;
    // The Device rides the resident cache: ONE candle MetalDevice instance (one
    // command queue, one built-in kernel cache, one buffer pool) for the app's
    // lifetime — sessions borrow it, never mint their own.
    let (model, proc, device) = resident_lfm2(&dir, &settings.lfm2.device)?;
    let params = GenParams {
        max_new_tokens: settings.lfm2.max_tokens as usize,
        // Sampled text, NOT greedy — vendor-exact: LiquidAI's transformers-js
        // conversational demo runs textTemperature = 1.0 (full multinomial).
        // Greedy text at 1.2B is a repetition machine: the model re-emits its
        // favorite phrasings every turn regardless of context.
        text_temperature: Some(1.0),
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed: settings.lfm2.seed.unwrap_or(0),
    };
    let mut engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
    if let Some(vault) = vault {
        engine = engine.with_conversation_vault(vault);
    }
    if settings.lfm2.delegate.enabled
        && settings
            .lfm2
            .delegate
            .target
            .as_deref()
            .is_some_and(|target| !target.trim().is_empty())
    {
        return Ok(Box::new(
            engine.with_system_prompt(LFM2_CONVERSE_SYSTEM_PROMPT),
        ));
    }
    Ok(Box::new(engine))
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
    let output = external_output_from_state(state.clone())?;
    let rate = output.rate();
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
    let write = output.clone();
    tauri::async_runtime::spawn_blocking(move || write.write_mono_f32(&samples))
        .await
        .map_err(|e| format!("audio probe task failed: {e}"))??;
    tokio::time::sleep(duration + Duration::from_millis(300)).await;
    output.clear();
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

async fn start_local_webrtc_output() -> Result<ExternalAudioOutput, String> {
    let state = Arc::new(local_webrtc_output_loopback().await?);
    external_output_from_state(state)
}

fn external_output_from_state(
    state: Arc<LocalWebRtcOutputState>,
) -> Result<ExternalAudioOutput, String> {
    let write_state = state.clone();
    let clear_state = state.clone();
    ExternalAudioOutput::new(
        LIVEKIT_AGENT_AUDIO_RATE,
        move |pcm, rate| write_state.write(pcm, rate),
        move || clear_state.clear(),
    )
}

async fn local_webrtc_output_loopback() -> Result<LocalWebRtcOutputState, String> {
    let audio = PlatformAudio::new().map_err(|e| format!("WebRTC speaker device failed: {e}"))?;
    configure_livekit_audio(&audio)
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
    let (remote_tx, remote_rx) = oneshot::channel::<livekit::webrtc::audio_track::RtcAudioTrack>();
    let remote_tx = Arc::new(Mutex::new(Some(remote_tx)));

    sender.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = sender_ice_tx.try_send(candidate);
    })));
    receiver.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = receiver_ice_tx.try_send(candidate);
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
    exchange_loopback_ice(&sender, &receiver, sender_ice_rx, receiver_ice_rx).await?;
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
) -> Result<(), String> {
    let deadline = tokio::time::sleep(Duration::from_millis(LOCAL_WEBRTC_OUTPUT_ICE_TIMEOUT_MS));
    tokio::pin!(deadline);
    let mut poll = tokio::time::interval(Duration::from_millis(20));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut sender_open = true;
    let mut receiver_open = true;
    loop {
        if local_webrtc_output_connected(sender, receiver) {
            return Ok(());
        }
        tokio::select! {
            _ = &mut deadline => {
                return Err("local WebRTC audio loopback did not connect before timeout".into())
            },
            _ = poll.tick() => {}
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
        }
    }
}

fn local_webrtc_output_connected(sender: &PeerConnection, receiver: &PeerConnection) -> bool {
    matches!(sender.connection_state(), PeerConnectionState::Connected)
        && matches!(receiver.connection_state(), PeerConnectionState::Connected)
}

struct LocalWebRtcInputState {
    _audio: PlatformAudio,
    _track: LocalAudioTrack,
    sender: PeerConnection,
    receiver: PeerConnection,
    remote_track: livekit::webrtc::audio_track::RtcAudioTrack,
}

impl Drop for LocalWebRtcInputState {
    fn drop(&mut self) {
        // Paired with the explicit start_recording in local_webrtc_mic_loop.
        let _ = self._audio.stop_recording();
        self.remote_track.set_enabled(false);
        self.sender.close();
        self.receiver.close();
    }
}

async fn local_webrtc_input_loopback(
    audio: PlatformAudio,
) -> Result<LocalWebRtcInputState, String> {
    let track = LocalAudioTrack::create_audio_track("local-microphone", audio.rtc_source());
    track.enable();
    track.unmute();
    let runtime = LkRuntime::instance();
    let factory = runtime.pc_factory();
    let sender = factory
        .create_peer_connection(RtcConfiguration::default())
        .map_err(|e| format!("local WebRTC input sender failed: {e}"))?;
    let receiver = factory
        .create_peer_connection(RtcConfiguration::default())
        .map_err(|e| format!("local WebRTC input receiver failed: {e}"))?;
    let (sender_ice_tx, sender_ice_rx) =
        mpsc::channel::<IceCandidate>(LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP);
    let (receiver_ice_tx, receiver_ice_rx) =
        mpsc::channel::<IceCandidate>(LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP);
    let (remote_tx, remote_rx) = oneshot::channel::<livekit::webrtc::audio_track::RtcAudioTrack>();
    let remote_tx = Arc::new(Mutex::new(Some(remote_tx)));

    sender.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = sender_ice_tx.try_send(candidate);
    })));
    receiver.on_ice_candidate(Some(Box::new(move |candidate| {
        let _ = receiver_ice_tx.try_send(candidate);
    })));
    receiver.on_track(Some(Box::new(move |event| {
        if let MediaStreamTrack::Audio(track) = event.track {
            track.set_enabled(false);
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
            &[LOCAL_WEBRTC_INPUT_STREAM_ID],
        )
        .map_err(|e| format!("local WebRTC input add_track failed: {e}"))?;
    let offer = sender
        .create_offer(OfferOptions::default())
        .await
        .map_err(|e| format!("local WebRTC input offer failed: {e}"))?;
    sender
        .set_local_description(offer.clone())
        .await
        .map_err(|e| format!("local WebRTC input set local offer failed: {e}"))?;
    receiver
        .set_remote_description(offer)
        .await
        .map_err(|e| format!("local WebRTC input set remote offer failed: {e}"))?;
    let answer = receiver
        .create_answer(AnswerOptions::default())
        .await
        .map_err(|e| format!("local WebRTC input answer failed: {e}"))?;
    receiver
        .set_local_description(answer.clone())
        .await
        .map_err(|e| format!("local WebRTC input set local answer failed: {e}"))?;
    sender
        .set_remote_description(answer)
        .await
        .map_err(|e| format!("local WebRTC input set remote answer failed: {e}"))?;
    exchange_loopback_ice(&sender, &receiver, sender_ice_rx, receiver_ice_rx).await?;
    let remote_track = tokio::time::timeout(
        Duration::from_millis(LOCAL_WEBRTC_MIC_READY_TIMEOUT_MS),
        remote_rx,
    )
    .await
    .map_err(|_| "local WebRTC microphone track did not become ready".to_string())?
    .map_err(|_| "local WebRTC microphone track setup channel closed".to_string())?;
    remote_track.set_enabled(false);

    Ok(LocalWebRtcInputState {
        _audio: audio,
        _track: track,
        sender,
        receiver,
        remote_track,
    })
}

async fn start_local_webrtc_input(
    threads: &ThreadManager,
    done: Arc<AtomicBool>,
) -> Result<ExternalAudioInput, String> {
    let (input, writer) = ExternalAudioInput::new(LIVEKIT_AGENT_AUDIO_RATE)?;
    let (ready_tx, ready_rx) = oneshot::channel();
    let thread_done = done.clone();
    threads.spawn("voice-local-webrtc-mic", move || {
        if let Err(error) =
            tauri::async_runtime::block_on(local_webrtc_mic_loop(writer, thread_done, ready_tx))
        {
            eprintln!("[voice] local WebRTC microphone stopped: {error}");
        }
    })?;
    match tokio::time::timeout(
        Duration::from_millis(LOCAL_WEBRTC_MIC_READY_TIMEOUT_MS),
        ready_rx,
    )
    .await
    {
        Ok(Ok(Ok(()))) => Ok(input),
        Ok(Ok(Err(error))) => Err(error),
        Ok(Err(_)) => Err("local WebRTC microphone setup channel closed".into()),
        Err(_) => {
            done.store(true, Ordering::SeqCst);
            Err("local WebRTC microphone did not become ready before timeout".into())
        }
    }
}

async fn local_webrtc_mic_loop(
    writer: ExternalAudioInputWriter,
    done: Arc<AtomicBool>,
    ready: oneshot::Sender<Result<(), String>>,
) -> Result<(), String> {
    let mut ready = Some(ready);
    let audio = match PlatformAudio::new() {
        Ok(audio) => audio,
        Err(error) => {
            let message = format!("WebRTC audio device failed: {error}");
            send_local_webrtc_ready(&mut ready, Err(message.clone()));
            return Err(message);
        }
    };
    if let Err(error) = configure_livekit_audio(&audio) {
        let message = format!("WebRTC audio processing failed: {error}");
        send_local_webrtc_ready(&mut ready, Err(message.clone()));
        return Err(message);
    }
    let input = match local_webrtc_input_loopback(audio).await {
        Ok(input) => input,
        Err(error) => {
            send_local_webrtc_ready(&mut ready, Err(error.clone()));
            return Err(error);
        }
    };
    // Explicitly start ADM recording, like the native LiveKit mic path does
    // (livekit_set_mic). Without it, capture only runs incidentally via the track
    // and an OS-level denial would surface as silent no-frames instead of an
    // error — and the APM capture chain (AEC/NS/AGC) may not fully engage.
    // ORDER MATTERS: start_recording is a resume-style call — init_recording only
    // succeeds once the device-source track exists (calling it pre-track failed
    // with "init_recording failed"), so it runs AFTER the loopback is up.
    if let Err(error) = input._audio.start_recording() {
        let message = format!("WebRTC microphone start failed: {error}");
        send_local_webrtc_ready(&mut ready, Err(message.clone()));
        return Err(message);
    }
    let mut stream = NativeAudioStream::with_options(
        input.remote_track.clone(),
        LIVEKIT_AGENT_AUDIO_RATE_I32,
        LIVEKIT_AGENT_AUDIO_CHANNELS_I32,
        NativeAudioStreamOptions {
            queue_size_frames: Some(LOCAL_WEBRTC_MIC_QUEUE_FRAMES),
        },
    );
    let first = tokio::time::timeout(
        Duration::from_millis(LOCAL_WEBRTC_MIC_READY_TIMEOUT_MS),
        next_local_webrtc_mic_frame(&mut stream, &done),
    )
    .await
    .map_err(|_| "local WebRTC microphone produced no frames before timeout".to_string())?;
    match first {
        Some(frame) => {
            push_local_webrtc_mic_frame(&writer, &frame);
            send_local_webrtc_ready(&mut ready, Ok(()));
        }
        None => {
            let message = "local WebRTC microphone stream ended before first frame".to_string();
            send_local_webrtc_ready(&mut ready, Err(message.clone()));
            return Err(message);
        }
    }
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    while !done.load(Ordering::SeqCst) {
        let frame = tokio::select! {
            frame = stream.next() => frame,
            _ = poll.tick() => {
                if done.load(Ordering::SeqCst) {
                    break;
                }
                continue;
            }
        };
        let Some(frame) = frame else {
            break;
        };
        push_local_webrtc_mic_frame(&writer, &frame);
    }
    writer.clear();
    Ok(())
}

fn send_local_webrtc_ready(
    ready: &mut Option<oneshot::Sender<Result<(), String>>>,
    result: Result<(), String>,
) {
    if let Some(ready) = ready.take() {
        let _ = ready.send(result);
    }
}

async fn next_local_webrtc_mic_frame(
    stream: &mut NativeAudioStream,
    done: &Arc<AtomicBool>,
) -> Option<AudioFrame<'static>> {
    let mut poll = tokio::time::interval(Duration::from_millis(LIVEKIT_DONE_POLL_MS));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let frame = tokio::select! {
            frame = stream.next() => frame,
            _ = poll.tick() => {
                if done.load(Ordering::SeqCst) {
                    return None;
                }
                continue;
            }
        };
        let Some(frame) = frame else {
            return None;
        };
        if frame.sample_rate == 0 {
            continue;
        }
        if audio_frame_to_mono_f32(&frame).is_empty() {
            continue;
        }
        return Some(frame);
    }
}

fn push_local_webrtc_mic_frame(writer: &ExternalAudioInputWriter, frame: &AudioFrame<'_>) -> bool {
    if frame.sample_rate == 0 {
        return false;
    }
    let pcm = audio_frame_to_mono_f32(frame);
    if pcm.is_empty() {
        return false;
    }
    let _ = writer.push_mono_f32(&pcm);
    true
}

fn audio_frame_to_mono_f32(frame: &AudioFrame<'_>) -> Vec<f32> {
    let channels = (frame.num_channels as usize).max(1);
    if channels == 1 {
        return i16_to_f32(frame.data.as_ref());
    }
    frame
        .data
        .as_ref()
        .chunks(channels)
        .map(|chunk| {
            let sum = chunk
                .iter()
                .map(|sample| *sample as f32 / i16::MAX as f32)
                .sum::<f32>();
            sum / chunk.len().max(1) as f32
        })
        .collect()
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
    fn livekit_silence_padding_tops_off_one_model_frame() {
        let mut partial = vec![1.0, 2.0, 3.0];
        pad_next_livekit_model_frame(&mut partial, 5);
        assert_eq!(partial, vec![1.0, 2.0, 3.0, 0.0, 0.0]);

        let mut empty = Vec::new();
        pad_next_livekit_model_frame(&mut empty, 4);
        assert_eq!(empty, vec![0.0, 0.0, 0.0, 0.0]);

        let mut aligned = vec![1.0, 2.0, 3.0, 4.0];
        pad_next_livekit_model_frame(&mut aligned, 4);
        assert_eq!(aligned, vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
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
