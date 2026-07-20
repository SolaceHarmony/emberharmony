//! Tauri-owned voice runtime.
//!
//! The audio service itself lives in `liquid_audio::VoiceRuntime`: the Tauri layer
//! owns the platform microphone and speaker callbacks directly, while VAD,
//! barge-in, and realtime inference run on Rust threads. This module is the desktop
//! kernel wrapper: it loads Tauri settings,
//! builds the LFM2 engine, owns the single active session, and maps runtime events
//! onto the webview `Channel`.

use std::{
    borrow::Cow,
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use liquid_audio::{
    AudioStatsSnapshot, NativeConversationVault, NativeLfm2VoiceEngine, NativeVoiceModel,
    NativeVoiceRuntimeConfig, NativeVoiceSampling, RuntimeConfig, RuntimeEvent, SessionState,
    VoiceEngine, VoiceRuntime as Lfm2Runtime,
};
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
    self, Lfm2Device, Lfm2Settings, LocalVoiceEngine, VoiceProvider, VoiceSettings,
};

use super::control::{LiveKitGrant, Role, SessionCtx, VoiceEvent, VoiceState};
use super::session::{
    CancelScope, CancelSignal, SessionBridgeConfig, SessionBridgeEvent, run_turn,
};
use super::threads::ThreadManager;

type UiChannel = Arc<UiEvents>;
type AsyncTask = tauri::async_runtime::JoinHandle<()>;
const VOICE_COMMAND_CAP: usize = 16;
const UI_EVENT_CAP: usize = 256;
const LIVEKIT_AGENT_AUDIO_RATE: u32 = 48_000;
const LIVEKIT_AGENT_AUDIO_CHANNELS: u32 = 1;
const LIVEKIT_AGENT_AUDIO_QUEUE_MS: u32 = 250;
const LOCAL_WEBRTC_OUTPUT_READY_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_TIMEOUT_MS: u64 = 5_000;
const LOCAL_WEBRTC_OUTPUT_ICE_QUEUE_CAP: usize = 16;
const LOCAL_WEBRTC_OUTPUT_STREAM_ID: &str = "local-lfm2-output";
/// Identity of the one native resident image. The runtime configuration is part
/// of the key because it controls native worker topology and dock capacities.
#[derive(Clone, PartialEq)]
struct ResidentLfm2Key {
    dir: PathBuf,
    device_setting: Lfm2Device,
    runtime: NativeVoiceRuntimeConfig,
}

struct ResidentLoad<K, V> {
    key: K,
    result: Mutex<Option<Result<V, String>>>,
    ready: Condvar,
}

impl<K, V: Clone> ResidentLoad<K, V> {
    fn wait(&self) -> Result<V, String> {
        let mut result = self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while result.is_none() {
            result = self
                .ready
                .wait(result)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        result
            .as_ref()
            .expect("resident load completed without a result")
            .clone()
    }

    fn complete(&self, result: Result<V, String>) {
        *self
            .result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result);
        self.ready.notify_all();
    }
}

struct ResidentCacheState<K, V> {
    resident: Option<(K, V)>,
    load: Option<Arc<ResidentLoad<K, V>>>,
}

struct ResidentCache<K, V> {
    state: Mutex<ResidentCacheState<K, V>>,
}

enum ResidentClaim<K, V> {
    Resident(V),
    Mismatch,
    Wait {
        load: Arc<ResidentLoad<K, V>>,
        same_key: bool,
    },
    Load(Arc<ResidentLoad<K, V>>),
}

impl<K: Clone + PartialEq, V: Clone> ResidentCache<K, V> {
    const fn new() -> Self {
        Self {
            state: Mutex::new(ResidentCacheState {
                resident: None,
                load: None,
            }),
        }
    }

    fn get_or_try_init(
        &self,
        key: K,
        init: impl FnOnce() -> Result<V, String>,
    ) -> Result<V, String> {
        let mut init = Some(init);
        loop {
            let claim = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some((resident_key, resident)) = state.resident.as_ref() {
                    if resident_key == &key {
                        ResidentClaim::Resident(resident.clone())
                    } else {
                        /* Replacing a live cache entry would open the new
                         * multi-gigabyte image before outstanding model/session
                         * clones can release the old one. Refuse the reload so
                         * the desktop's one-image invariant is physical, not
                         * merely one pointer in this cache. */
                        ResidentClaim::Mismatch
                    }
                } else if let Some(load) = state.load.as_ref() {
                    ResidentClaim::Wait {
                        same_key: load.key == key,
                        load: load.clone(),
                    }
                } else {
                    let load = Arc::new(ResidentLoad {
                        key: key.clone(),
                        result: Mutex::new(None),
                        ready: Condvar::new(),
                    });
                    state.load = Some(load.clone());
                    ResidentClaim::Load(load)
                }
            };

            match claim {
                ResidentClaim::Resident(resident) => return Ok(resident),
                ResidentClaim::Mismatch => {
                    return Err(
                        "a different native LFM2 model is already resident; restart the desktop before changing the model path or runtime topology"
                            .into(),
                    );
                }
                ResidentClaim::Wait { load, same_key } => {
                    let result = load.wait();
                    if same_key {
                        return result;
                    }
                }
                ResidentClaim::Load(load) => {
                    let init = init
                        .take()
                        .expect("resident initializer consumed before claiming the load");
                    let result = init();
                    {
                        let mut state = self
                            .state
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if let Ok(resident) = result.as_ref() {
                            state.resident = Some((key.clone(), resident.clone()));
                        }
                        if state
                            .load
                            .as_ref()
                            .is_some_and(|active| Arc::ptr_eq(active, &load))
                        {
                            state.load = None;
                        }
                    }
                    load.complete(result.clone());
                    return result;
                }
            }
        }
    }
}

/// The one native resident image. Callers park behind the active loader, so the
/// desktop can never transiently construct duplicate multi-gigabyte images.
static LFM2_RESIDENT: ResidentCache<ResidentLfm2Key, NativeVoiceModel> = ResidentCache::new();

/// One conversation vault per chat session: the model's conversation must survive
/// UI-driven voice session rebuilds (route changes, settings writes) — the upstream
/// `chat.py` invariant, where `ChatState` outlives everything but an explicit reset.
static CONVERSATION_VAULTS: Mutex<Option<HashMap<String, NativeConversationVault>>> =
    Mutex::new(None);

fn conversation_vault(session_id: &str) -> NativeConversationVault {
    let mut slot = CONVERSATION_VAULTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    slot.get_or_insert_with(HashMap::new)
        .entry(session_id.to_string())
        .or_default()
        .clone()
}

fn resident_lfm2(dir: &Path, device_setting: &Lfm2Device) -> Result<NativeVoiceModel, String> {
    if *device_setting != Lfm2Device::Cpu {
        return Err(
            "Native LFM2 Metal/MLX is not available yet; select CPU. The desktop will not fall back to Candle Metal."
                .into(),
        );
    }
    let runtime = NativeVoiceRuntimeConfig::default();
    let key = ResidentLfm2Key {
        dir: dir.to_path_buf(),
        device_setting: device_setting.clone(),
        runtime,
    };
    LFM2_RESIDENT.get_or_try_init(key, || {
        // The loader writes each shard directly into the final immutable native
        // image. There is no Rust safetensors builder or Candle compatibility copy.
        eprintln!("[voice] LFM2 session: loading native resident image");
        NativeVoiceModel::open_with_config(dir, runtime)
            .map_err(|error| format!("failed to load native LFM2-Audio: {error}"))
    })
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
            RuntimeCommand::Reap => {
                reap_finished(&mut session, &threads);
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
    wake: mpsc::Sender<RuntimeCommand>,
) -> Result<(), String> {
    reap_finished(session, threads);
    if session.is_some() {
        return Err("Voice is already running.".into());
    }
    threads.wait()?;
    *session = Some(VoiceSession(Lfm2Session::spawn(
        threads, ctx, settings, channel, bridge, wake,
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
    session.stop();
    threads.wait()
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
    channel: UiChannel,
    done: Arc<AtomicBool>,
    bridge_cancel: Arc<CancelSignal>,
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
        let bridge_cancel = Arc::new(CancelSignal::default());
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
        let live = Lfm2Runtime::prepare(
            cfg,
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
        )?;
        Ok(Self {
            ctx,
            settings: key,
            channel,
            done,
            bridge_cancel,
            live: Some(live),
        })
    }

    fn is_finished(&self) -> bool {
        self.live.as_ref().is_none_or(Lfm2Runtime::is_finished)
    }

    fn interrupt(&self) -> Result<(), String> {
        self.bridge_cancel.cancel();
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
        self.bridge_cancel.cancel();
        if let Some(live) = self.live.take() {
            live.stop();
        }
    }
}

impl Drop for Lfm2Session {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.bridge_cancel.cancel();
        if let Some(live) = self.live.take() {
            live.stop();
        }
    }
}

struct BridgeState {
    threads: ThreadManager,
    channel: UiChannel,
    bridge: Option<SessionBridgeConfig>,
    cancel: Arc<CancelSignal>,
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
        cancel: Arc<CancelSignal>,
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
                self.cancel.cancel();
                self.clear_turn();
            }
            RuntimeEvent::Level(_) | RuntimeEvent::State(_) => {}
        }
        send_runtime(&self.channel, event)
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
        let signal = self.cancel.clone();
        let cancel = signal.scope();
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
                if !send_scoped(
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
                        send_scoped(
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
                    let _ = send_scoped(&channel, &cancel, VoiceEvent::Error { message: error });
                }
                if !cancel.is_cancelled() {
                    let _ = send_scoped(
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
                let _ = send_or_cancel(&self.channel, &signal, VoiceEvent::Error { message });
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
        self.cancel.cancel();
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

fn send_or_cancel(channel: &UiChannel, cancel: &CancelSignal, event: VoiceEvent) -> bool {
    if send(channel, event) {
        return true;
    }
    cancel.cancel();
    false
}

fn send_scoped(channel: &UiChannel, cancel: &CancelScope, event: VoiceEvent) -> bool {
    if cancel.is_cancelled() {
        return false;
    }
    if send(channel, event) {
        return true;
    }
    cancel.cancel();
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
        RuntimeEvent::Ended(reason) => send(channel, VoiceEvent::Ended { reason }),
        RuntimeEvent::Error(message) => send(channel, VoiceEvent::Error { message }),
    }
}

/// Map persisted sampling policy into the versioned native conversation/session
/// configuration. No sampler or token loop runs in Rust.
fn native_sampling(mode: &settings::Lfm2ModeSampling, seed: Option<u64>) -> NativeVoiceSampling {
    NativeVoiceSampling {
        // The UI enforces >= 1; a hand-edited store must not mute the voice
        // (0 would end every turn instantly with only an EXHAUSTED warning).
        max_new_tokens: mode.max_tokens.max(1),
        text_temperature: (mode.text_temperature > 0.0).then_some(mode.text_temperature),
        text_top_k: (mode.text_top_k > 0).then_some(mode.text_top_k),
        audio_temperature: (mode.audio_temperature > 0.0).then_some(mode.audio_temperature),
        audio_top_k: (mode.audio_top_k > 0).then_some(mode.audio_top_k),
        seed,
    }
}

fn local_runtime_config(settings: &VoiceSettings) -> RuntimeConfig {
    RuntimeConfig {
        vad_threshold: settings.lfm2.vad_threshold,
        can_interrupt: can_interrupt_playback(settings),
        trace: settings.lfm2.trace,
        ..RuntimeConfig::default()
    }
}

fn can_interrupt_playback(settings: &VoiceSettings) -> bool {
    settings.lfm2.engine != LocalVoiceEngine::MoshiRealtime
}

fn build_engine(
    settings: VoiceSettings,
    out_rate: u32,
    vault: Option<NativeConversationVault>,
) -> Result<Box<dyn VoiceEngine>, String> {
    // Fail-hard, no network at start: load ONLY a local snapshot dir. The repo id/revision are
    // the download source (Settings → Download), not a run-time fetch — never auto-download here.
    if settings.lfm2.engine == LocalVoiceEngine::MoshiRealtime {
        return Err(
            "Moshi realtime inference is offline-oracle only in this release; select native LFM2 interleaved. No Candle fallback is linked."
                .into(),
        );
    }
    let dir = settings::lfm2_active_model_dir(&settings.lfm2).ok_or_else(|| {
        "No local LFM2-Audio model — download a model or select a model directory in Settings."
            .to_string()
    })?;
    if settings.lfm2.delegate.enabled
        && settings
            .lfm2
            .delegate
            .target
            .as_deref()
            .is_some_and(|target| !target.trim().is_empty())
    {
        return Err(
            "Delegation requires a configurable native system-prompt command, which is not in the current session ABI. Native LFM2 will not instantiate the Candle engine as a fallback."
                .into(),
        );
    }
    let model = resident_lfm2(&dir, &settings.lfm2.device)?;
    let sampling = native_sampling(&settings.lfm2.interleaved, settings.lfm2.seed);
    let engine: NativeLfm2VoiceEngine = model.engine(sampling, vault, out_rate)?;
    Ok(Box::new(engine))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    #[test]
    fn lfm2_desktop_factory_and_cache_are_native_only() {
        const FACTORY: fn(&Path, &Lfm2Device) -> Result<NativeVoiceModel, String> = resident_lfm2;
        let _ = FACTORY;
        let error = resident_lfm2(Path::new("unused-for-device-rejection"), &Lfm2Device::Metal)
            .err()
            .expect("Metal must fail instead of constructing a compatibility model");
        assert!(error.contains("will not fall back to Candle Metal"));
    }

    #[test]
    fn resident_cache_single_flights_concurrent_same_key_loads() {
        const CALLERS: usize = 16;
        let cache = Arc::new(ResidentCache::<u8, usize>::new());
        let start = Arc::new(Barrier::new(CALLERS + 1));
        let release = Arc::new(Barrier::new(2));
        let calls = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let threads = (0..CALLERS)
            .map(|_| {
                let cache = cache.clone();
                let start = start.clone();
                let release = release.clone();
                let calls = calls.clone();
                let entered_tx = entered_tx.clone();
                std::thread::spawn(move || {
                    start.wait();
                    cache
                        .get_or_try_init(7, || {
                            calls.fetch_add(1, AtomicOrdering::SeqCst);
                            entered_tx.send(()).expect("announce active loader");
                            release.wait();
                            Ok(41)
                        })
                        .expect("single-flight load")
                })
            })
            .collect::<Vec<_>>();

        start.wait();
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("one caller must claim the load");
        release.wait();

        for thread in threads {
            assert_eq!(thread.join().expect("resident cache caller"), 41);
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn resident_cache_rejects_a_second_key_without_opening_it() {
        let cache = ResidentCache::<u8, usize>::new();
        assert_eq!(cache.get_or_try_init(1, || Ok(41)).unwrap(), 41);
        let opened = AtomicUsize::new(0);
        let error = cache
            .get_or_try_init(2, || {
                opened.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(42)
            })
            .unwrap_err();
        assert_eq!(opened.load(AtomicOrdering::SeqCst), 0);
        assert!(error.contains("already resident"));
        assert_eq!(cache.get_or_try_init(1, || Ok(99)).unwrap(), 41);
    }

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
        // The done_tx.send above races with actual OS-thread termination:
        // is_finished() can still be false right after the recv. Poll reap
        // until the finished thread is gone (bounded, so a real regression
        // still fails fast).
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            threads.reap().unwrap();
            if threads.tracked_len() == 1 || std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
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
    fn local_runtime_config_comes_from_persisted_settings() {
        let mut settings = VoiceSettings::default();
        settings.lfm2.vad_threshold = 0.027;
        settings.lfm2.trace = true;

        let cfg = local_runtime_config(&settings);
        assert_eq!(cfg.vad_threshold, 0.027);
        assert!(cfg.trace);
        assert_eq!(cfg.can_interrupt, can_interrupt_playback(&settings));
    }
}
