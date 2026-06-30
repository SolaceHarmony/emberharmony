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
    thread,
};

use candle_core::{DType, Device};
use liquid_audio::{
    GenParams, Lfm2VoiceEngine, RuntimeConfig, RuntimeEvent, SessionState,
    VoiceRuntime as Lfm2Runtime, from_pretrained, get_model_dir,
};

use crate::settings::{Lfm2Device, VoiceSettings};

use super::control::{Role, SessionCtx, VoiceEvent, VoiceState};
use super::session::{SessionBridgeConfig, SessionBridgeEvent, run_turn};

type UiChannel = Arc<Mutex<tauri::ipc::Channel<VoiceEvent>>>;

/// One active native voice service for the desktop app.
#[derive(Default)]
pub struct VoiceRuntime {
    session: Mutex<Option<VoiceSession>>,
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
        if guard.as_ref().is_some_and(VoiceSession::is_finished) {
            if let Some(session) = guard.take() {
                session.stop();
            }
        }
        if guard.is_some() {
            return Err("Voice is already running.".into());
        }
        *guard = Some(VoiceSession::spawn(ctx, settings, channel, bridge));
        Ok(())
    }

    pub fn stop(&self) -> Result<(), String> {
        let session = self
            .session
            .lock()
            .map_err(|_| "voice runtime lock poisoned".to_string())?
            .take();
        if let Some(session) = session {
            thread::Builder::new()
                .name("voice-lfm2-stop".into())
                .spawn(move || session.stop())
                .map_err(|e| format!("failed to spawn voice cleanup thread: {e}"))?;
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
}

struct VoiceSession {
    _ctx: SessionCtx,
    done: Arc<AtomicBool>,
    bridge_cancel: Arc<AtomicBool>,
    live: Option<Lfm2Runtime>,
}

impl VoiceSession {
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
            _ctx: ctx,
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

    fn stop(mut self) {
        self.done.store(true, Ordering::SeqCst);
        self.bridge_cancel.store(true, Ordering::SeqCst);
        if let Some(live) = self.live.take() {
            live.stop();
        }
    }
}

impl Drop for VoiceSession {
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
    let model_ref = model_ref(&settings)?;
    let (device, dtype) = select_device(&settings.lfm2.device)?;
    let dir = get_model_dir(&model_ref, None)
        .map_err(|e| format!("failed to resolve model `{model_ref}`: {e}"))?;
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

fn model_ref(settings: &VoiceSettings) -> Result<String, String> {
    if let Some(model) = settings
        .lfm2
        .model
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        return Ok(model.trim().to_string());
    }
    settings
        .lfm2
        .model_dir
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "Set the model directory to enable the local model.".to_string())
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
