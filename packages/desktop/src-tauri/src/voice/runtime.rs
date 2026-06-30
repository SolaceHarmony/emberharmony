//! Tauri-owned voice runtime.
//!
//! The audio service itself lives in `liquid_audio::VoiceRuntime`: CPAL mic
//! capture, speaker playback, VAD, barge-in, and the realtime inference worker
//! all run in-process on Rust threads. This module is the desktop kernel wrapper:
//! it loads Tauri settings, builds the LFM2 engine, owns the single active
//! session, and maps runtime events onto the webview `Channel`.

use std::{
    fs,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use candle_core::{DType, Device};
use liquid_audio::{
    GenParams, Lfm2VoiceEngine, RuntimeConfig, RuntimeEvent, SessionState,
    VoiceRuntime as Lfm2Runtime, from_pretrained,
};

use crate::settings::{self, Lfm2Device, VoiceProvider, VoiceSettings};

use super::control::{Role, SessionCtx, VoiceEvent, VoiceState};
use super::session::{SessionBridgeConfig, SessionBridgeEvent, run_turn};

type UiChannel = Arc<Mutex<tauri::ipc::Channel<VoiceEvent>>>;

/// One active native voice service for the desktop app.
#[derive(Default)]
pub struct VoiceRuntime {
    session: Mutex<Option<VoiceSession>>,
    threads: ThreadManager,
}

impl VoiceRuntime {
    pub fn start_lfm2(
        &self,
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
    ) -> Result<(), String> {
        let mut guard = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?;
        self.threads.wait()?;
        if guard.as_ref().is_some_and(VoiceSession::is_finished) {
            if let Some(session) = guard.take() {
                session.stop();
            }
        }
        if guard.is_some() {
            return Err("Voice is already running.".into());
        }
        *guard = Some(VoiceSession::Lfm2(Lfm2Session::spawn(
            ctx, settings, channel, bridge,
        )));
        Ok(())
    }

    pub fn start_livekit(&self, ctx: SessionCtx) -> Result<(), String> {
        let mut guard = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?;
        self.threads.wait()?;
        if guard.as_ref().is_some_and(VoiceSession::is_finished) {
            if let Some(session) = guard.take() {
                session.stop();
            }
        }
        if guard.is_some() {
            return Err("Voice is already running.".into());
        }
        *guard = Some(VoiceSession::Livekit(LiveKitSession::new(ctx)));
        Ok(())
    }

    pub fn stop(&self) -> Result<(), String> {
        let mut guard = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?;
        if let Some(session) = guard.take() {
            self.threads
                .spawn("voice-session-stop", move || session.stop())?;
        }
        Ok(())
    }

    pub fn interrupt(&self) -> Result<(), String> {
        if let Some(session) = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
        {
            session.interrupt();
        }
        Ok(())
    }

    pub fn set_mic_enabled(&self, enabled: bool) -> Result<(), String> {
        if let Some(session) = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
        {
            session.set_mic_enabled(enabled);
        }
        Ok(())
    }

    pub fn is_running(&self) -> Result<bool, String> {
        self.threads.reap()?;
        Ok(self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
            .is_some_and(|session| !session.is_finished()))
    }

    pub fn mic_enabled(&self) -> Result<bool, String> {
        Ok(self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
            .is_some_and(VoiceSession::mic_enabled))
    }

    pub fn active_provider(&self) -> Result<Option<VoiceProvider>, String> {
        self.threads.reap()?;
        Ok(self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
            .filter(|session| !session.is_finished())
            .map(VoiceSession::provider))
    }

    pub fn is_running_session(&self, session_id: &str) -> Result<bool, String> {
        Ok(self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .as_ref()
            .is_some_and(|session| !session.is_finished() && session.session_id() == session_id))
    }
}

#[derive(Default)]
struct ThreadManager {
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl ThreadManager {
    fn spawn(&self, name: &'static str, run: impl FnOnce() + Send + 'static) -> Result<(), String> {
        self.reap()?;
        let handle = thread::Builder::new()
            .name(name.into())
            .spawn(run)
            .map_err(|e| format!("failed to spawn {name}: {e}"))?;
        self.handles
            .lock()
            .map_err(|_| "voice thread manager lock poisoned".to_string())?
            .push(handle);
        Ok(())
    }

    fn reap(&self) -> Result<(), String> {
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| "voice thread manager lock poisoned".to_string())?;
        let mut live = Vec::new();
        for handle in handles.drain(..) {
            if handle.is_finished() {
                let _ = handle.join();
                continue;
            }
            live.push(handle);
        }
        *handles = live;
        Ok(())
    }

    fn wait(&self) -> Result<(), String> {
        let handles = {
            let mut handles = self
                .handles
                .lock()
                .map_err(|_| "voice thread manager lock poisoned".to_string())?;
            handles.drain(..).collect::<Vec<_>>()
        };
        for handle in handles {
            let _ = handle.join();
        }
        Ok(())
    }
}

impl Drop for ThreadManager {
    fn drop(&mut self) {
        let Ok(handles) = self.handles.get_mut() else {
            return;
        };
        for handle in handles.drain(..) {
            let _ = handle.join();
        }
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
            VoiceSession::Livekit(_) => false,
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

    fn interrupt(&self) {
        match self {
            VoiceSession::Lfm2(session) => session.interrupt(),
            VoiceSession::Livekit(_) => {}
        }
    }

    fn set_mic_enabled(&self, enabled: bool) {
        match self {
            VoiceSession::Lfm2(session) => session.set_mic_enabled(enabled),
            VoiceSession::Livekit(session) => session.set_mic_enabled(enabled),
        }
    }

    fn mic_enabled(&self) -> bool {
        match self {
            VoiceSession::Lfm2(session) => session.mic_enabled(),
            VoiceSession::Livekit(session) => session.mic_enabled(),
        }
    }

    fn stop(self) {
        match self {
            VoiceSession::Lfm2(session) => session.stop(),
            VoiceSession::Livekit(session) => session.stop(),
        }
    }
}

struct LiveKitSession {
    ctx: SessionCtx,
    mic_enabled: AtomicBool,
}

impl LiveKitSession {
    fn new(ctx: SessionCtx) -> Self {
        Self {
            ctx,
            mic_enabled: AtomicBool::new(true),
        }
    }

    fn set_mic_enabled(&self, enabled: bool) {
        self.mic_enabled.store(enabled, Ordering::SeqCst);
    }

    fn mic_enabled(&self) -> bool {
        self.mic_enabled.load(Ordering::SeqCst)
    }

    fn stop(self) {}
}

struct Lfm2Session {
    ctx: SessionCtx,
    done: Arc<AtomicBool>,
    bridge_cancel: Arc<AtomicBool>,
    live: Option<Lfm2Runtime>,
}

impl Lfm2Session {
    fn spawn(
        ctx: SessionCtx,
        settings: VoiceSettings,
        channel: tauri::ipc::Channel<VoiceEvent>,
        bridge: Option<SessionBridgeConfig>,
    ) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let bridge_cancel = Arc::new(AtomicBool::new(false));
        let channel = Arc::new(Mutex::new(channel));
        let sink_done = done.clone();
        let sink_channel = channel.clone();
        let mut bridge_state =
            BridgeState::new(sink_channel.clone(), bridge, bridge_cancel.clone());
        let cfg = RuntimeConfig {
            vad_threshold: settings.lfm2.vad_threshold,
            ..RuntimeConfig::default()
        };
        let live = Lfm2Runtime::start(
            cfg,
            move |out_rate| build_engine(settings, out_rate),
            move |event| {
                if matches!(event, RuntimeEvent::Ended(_)) {
                    sink_done.store(true, Ordering::SeqCst);
                }
                let sent = bridge_state.handle(event);
                if !sent {
                    sink_done.store(true, Ordering::SeqCst);
                }
                sent
            },
        );
        Self {
            ctx,
            done,
            bridge_cancel,
            live: Some(live),
        }
    }

    fn is_finished(&self) -> bool {
        self.done.load(Ordering::SeqCst) || self.live.as_ref().is_some_and(Lfm2Runtime::is_finished)
    }

    fn interrupt(&self) {
        self.bridge_cancel.store(true, Ordering::SeqCst);
        if let Some(live) = self.live.as_ref() {
            live.interrupt();
        }
    }

    fn set_mic_enabled(&self, enabled: bool) {
        if let Some(live) = self.live.as_ref() {
            live.set_mic_enabled(enabled);
        }
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
    channel: UiChannel,
    bridge: Option<SessionBridgeConfig>,
    cancel: Arc<AtomicBool>,
    active: Arc<AtomicBool>,
    transcript: String,
    speaking: bool,
}

impl BridgeState {
    fn new(
        channel: UiChannel,
        bridge: Option<SessionBridgeConfig>,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            channel,
            bridge,
            cancel,
            active: Arc::new(AtomicBool::new(false)),
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
                    self.speaking = false;
                    self.maybe_delegate();
                }
            }
            RuntimeEvent::Ended(_) | RuntimeEvent::Error(_) => {
                self.cancel.store(true, Ordering::SeqCst);
            }
            RuntimeEvent::Level(_) | RuntimeEvent::State(_) => {}
        }
        send_runtime(&self.channel, event)
    }

    fn maybe_delegate(&mut self) {
        let Some(task) = delegate_task(&self.transcript) else {
            self.transcript.clear();
            return;
        };
        self.transcript.clear();
        let Some(cfg) = self.bridge.clone() else {
            return;
        };
        if self.active.swap(true, Ordering::SeqCst) {
            return;
        }
        self.cancel.store(false, Ordering::SeqCst);
        let cancel = self.cancel.clone();
        let active = self.active.clone();
        let channel = self.channel.clone();
        tauri::async_runtime::spawn(async move {
            let _ = send(
                &channel,
                VoiceEvent::State {
                    state: VoiceState::Thinking,
                },
            );
            let delta_channel = channel.clone();
            let mut reply = String::new();
            let result = run_turn(cfg, task, cancel.clone(), move |event| match event {
                SessionBridgeEvent::Delta { text, .. } => {
                    reply.push_str(&text);
                    send(
                        &delta_channel,
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
                let _ = send(&channel, VoiceEvent::Error { message: error });
            }
            let _ = send(
                &channel,
                VoiceEvent::State {
                    state: VoiceState::Listening,
                },
            );
            active.store(false, Ordering::SeqCst);
        });
    }
}

fn delegate_task(text: &str) -> Option<String> {
    text.lines().rev().find_map(|line| {
        let marker = line.find("DELEGATE:")?;
        let task = line[marker + "DELEGATE:".len()..].trim();
        (!task.is_empty()).then(|| task.to_string())
    })
}

fn send(channel: &UiChannel, event: VoiceEvent) -> bool {
    channel
        .lock()
        .map(|channel| channel.send(event).is_ok())
        .unwrap_or(false)
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

fn build_engine(settings: VoiceSettings, out_rate: u32) -> Result<Lfm2VoiceEngine, String> {
    // Fail-hard, no network at start: load ONLY a local snapshot dir. The repo id/revision are
    // the download source (Settings → Download), not a run-time fetch — never auto-download here.
    let dir = settings::lfm2_active_model_dir(&settings.lfm2).ok_or_else(|| {
        "No local LFM2-Audio model — download a model or select a model directory in Settings."
            .to_string()
    })?;
    let (device, dtype) = select_device(&settings.lfm2.device)?;
    let codebooks = codebooks(&dir)?;
    let (model, proc) = from_pretrained(&dir, dtype, &device)
        .map_err(|e| format!("failed to load LFM2-Audio: {e}"))?;
    let params = GenParams {
        max_new_tokens: settings.lfm2.max_tokens as usize,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed: settings.lfm2.seed.unwrap_or(0),
    };
    Ok(Lfm2VoiceEngine::new(
        model, proc, params, codebooks, device, out_rate,
    ))
}

fn select_device(device: &Lfm2Device) -> Result<(Device, DType), String> {
    match device {
        Lfm2Device::Cpu => Ok((Device::Cpu, DType::F32)),
        Lfm2Device::Metal => {
            #[cfg(target_os = "macos")]
            {
                Device::new_metal(0)
                    .map(|device| (device, DType::BF16))
                    .map_err(|e| format!("failed to open Metal device: {e}"))
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
