//! Callback-driven host audio service mounted on one retained kcoro continuation.
//!
//! Platform callbacks publish bounded native PCM/control edges and return. The
//! retained [`SessionTask`] owns only device handles, outward UI delivery, and
//! engine-event draining; native kcoro owns detector, turn, and model state. It
//! never polls a clock or blocks an operating-system thread for model progress.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "audio-io")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use kcoro_sys::{
    RealtimeNotifier, Runtime as CoroutineRuntime, RuntimeConfig as CoroutineConfig,
    Service as CoroutineService, ServiceOutcome, SharedRealtimeNotifier,
};
use serde::{Deserialize, Serialize};

use crate::voice_api::{CaptureSink, PlaybackSource};
use crate::{EngineProgress, VoiceEngine, VoiceEvent};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Stats = Arc<AudioStats>;
#[cfg(feature = "audio-io")]
type HostStream = cpal::Stream;

#[cfg(feature = "audio-io")]
struct DeviceStreams {
    _input: HostStream,
    _output: HostStream,
}

#[cfg(not(feature = "audio-io"))]
struct DeviceStreams;

/// Trace the voice call graph when the host enables `RuntimeConfig::trace`.
/// Configuration is explicit so an inherited process environment cannot alter
/// production timing or logging behavior.
static VOICE_TRACE_ENABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn voice_trace_enabled() -> bool {
    VOICE_TRACE_ENABLED.load(Ordering::Relaxed)
}

pub(crate) fn voice_trace_elapsed() -> f64 {
    static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// `vtrace!("vad: speech-start (streak {n})")` — one stderr line when tracing is on.
#[macro_export]
macro_rules! vtrace {
    ($($arg:tt)*) => {
        if $crate::voice_runtime::voice_trace_enabled() {
            eprintln!(
                "[voice-trace +{:8.3}s] {}",
                $crate::voice_runtime::voice_trace_elapsed(),
                format!($($arg)*)
            );
        }
    };
}

const DEVICE_FAULT_INPUT: u32 = 1;
const DEVICE_FAULT_OUTPUT: u32 = 1 << 1;
#[cfg(feature = "audio-io")]
/// Requested hardware callback cadence. This is setup geometry, not a progress
/// timer. CPAL's cross-platform `Fixed` value is only the requested cadence;
/// native capture admission is sealed independently at the product maximum.
const CAPTURE_CALLBACK_REQUEST_MS: u32 = 20;
#[cfg(feature = "audio-io")]
const CAPTURE_CALLBACK_MAX_MS: u32 = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CaptureContract {
    rate: u32,
    requested_frames: u32,
    max_callback_frames: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioStatsSnapshot {
    pub decoded_samples: u64,
    pub queued_samples: u64,
    pub dropped_samples: u64,
    pub played_samples: u64,
    pub underrun_frames: u64,
    /// Reserved for a future native correlated onset event; currently zero.
    pub turn_count: u64,
    /// Reserved for a future native correlated onset event; currently zero.
    pub last_turn_latency_ms: u64,
    /// Reserved for a future native correlated onset event; currently zero.
    pub mean_turn_latency_ms: u64,
}

#[derive(Default)]
struct AudioStats {
    decoded_samples: std::sync::atomic::AtomicU64,
    queued_samples: std::sync::atomic::AtomicU64,
    dropped_samples: std::sync::atomic::AtomicU64,
    played_samples: std::sync::atomic::AtomicU64,
    underrun_frames: std::sync::atomic::AtomicU64,
}

/// One terminal ownership transfer from the fixed coroutine owner back to the
/// lifecycle caller. The producer writes once before service completion; the
/// caller reads only after `service.join()` has established settlement.
struct EngineHandoff {
    engine: UnsafeCell<Option<Box<dyn VoiceEngine>>>,
    ready: AtomicBool,
}

unsafe impl Send for EngineHandoff {}
unsafe impl Sync for EngineHandoff {}

impl EngineHandoff {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            engine: UnsafeCell::new(None),
            ready: AtomicBool::new(false),
        })
    }

    fn publish(&self, engine: Box<dyn VoiceEngine>) {
        debug_assert!(!self.ready.load(Ordering::Acquire));
        unsafe { *self.engine.get() = Some(engine) };
        self.ready.store(true, Ordering::Release);
    }

    fn take(&self) -> Option<Box<dyn VoiceEngine>> {
        if !self.ready.swap(false, Ordering::AcqRel) {
            return None;
        }
        unsafe { (*self.engine.get()).take() }
    }
}

impl AudioStats {
    fn snapshot(&self) -> AudioStatsSnapshot {
        AudioStatsSnapshot {
            decoded_samples: self.decoded_samples.load(Ordering::Relaxed),
            queued_samples: self.queued_samples.load(Ordering::Relaxed),
            dropped_samples: self.dropped_samples.load(Ordering::Relaxed),
            played_samples: self.played_samples.load(Ordering::Relaxed),
            underrun_frames: self.underrun_frames.load(Ordering::Relaxed),
            turn_count: 0,
            last_turn_latency_ms: 0,
            mean_turn_latency_ms: 0,
        }
    }
}

/// Session state the UI reflects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Loading,
    Listening,
    Thinking,
    Speaking,
    Idle,
}

/// High-level event emitted by the voice service.
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    State(SessionState),
    Transcript(String),
    Level(f32),
    Ended(Option<String>),
    Error(String),
}

/// Host-only voice service settings. Turn detection and speech policy are
/// sealed in the native session configuration, never duplicated in Rust.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub trace: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { trace: false }
    }
}

/// Handle to a running voice service.
pub struct VoiceRuntime {
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    runtime: Arc<CoroutineRuntime>,
    service: Option<CoroutineService>,
    control: SharedRealtimeNotifier,
    audio: Stats,
    done: Arc<AtomicBool>,
    engine: Arc<EngineHandoff>,
    closed: AtomicBool,
}

impl VoiceRuntime {
    /// Mount and start the retained native coroutine service using platform
    /// audio devices. The engine factory receives capture rate, playback rate,
    /// and the sealed maximum capture callback size; CPAL's smaller requested
    /// cadence remains a platform-owner detail.
    pub fn prepare(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32, u32, u32) -> Result<Box<dyn VoiceEngine>, String>
            + Send
            + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<VoiceRuntime, String> {
        VOICE_TRACE_ENABLED.store(cfg.trace, Ordering::Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let playback_flush = Arc::new(AtomicBool::new(false));
        let device_fault = Arc::new(AtomicU32::new(0));
        let audio = Arc::new(AudioStats::default());
        let done = Arc::new(AtomicBool::new(false));
        let input = default_input_contract().map_err(|error| format!("audio input: {error}"))?;
        let out_rate = default_output_rate().map_err(|error| format!("audio output: {error}"))?;
        let mut sink: EventSink = Box::new(sink);
        if !emit_or_stop(&mut sink, &stop, RuntimeEvent::State(SessionState::Loading)) {
            return Err("voice event sink rejected the loading state".into());
        }
        let engine = build_engine(input.rate, out_rate, input.max_callback_frames)
            .map_err(|error| format!("model load: {error}"))?;
        let handoff = EngineHandoff::new();
        let mut resources = InitResources::new(engine, handoff.clone());
        let capture = resources
            .engine_mut()?
            .take_capture_sink()?
            .ok_or("callback-driven native engine did not expose a capture sink")?;
        resources.capture = Some(capture);
        let capture_rate = resources.capture_rate()?;
        if capture_rate != input.rate {
            return Err(format!(
                "native PCM rate does not match the input device: received {} Hz, expected {in_rate} Hz",
                capture_rate,
                in_rate = input.rate,
            ));
        }
        let mut init = SessionInit {
            resources,
            sink,
            stop: stop.clone(),
            interrupt: interrupt.clone(),
            mic_enabled: mic_enabled.clone(),
            playback_flush: playback_flush.clone(),
            device_fault: device_fault.clone(),
            control: None,
            audio: audio.clone(),
            in_rate: input.rate,
            in_request_frames: input.requested_frames,
            in_max_callback_frames: input.max_callback_frames,
            out_rate,
            done: done.clone(),
            events: None,
        };
        let runtime = Arc::new(
            CoroutineRuntime::with_config(CoroutineConfig {
                workers: 1,
                ..CoroutineConfig::default()
            })
            .map_err(|status| format!("create voice kcoro runtime failed: {status}"))?,
        );
        let (service, control) = runtime
            .owner_state_service_factory(|setup| {
                let events = setup.realtime_notifier()?;
                let playback_events = setup.realtime_notifier()?;
                let control = setup.shared_realtime_notifier();
                let source = init
                    .resources
                    .engine_mut()
                    .map_err(|_| -1)?
                    .take_playback_source(playback_events)
                    .map_err(|_| -1)?
                    .ok_or(-1)?;
                init.resources.source = Some(source);
                init.control = Some(control.clone());
                init.events = Some(events);
                let task = SessionTask::Init(Some(init));
                Ok((
                    move || {
                        let mut owner = OwnerSession::new(task);
                        move || owner.advance()
                    },
                    control,
                ))
            })
            .map_err(|status| format!("mount voice service failed: {status}"))?;
        runtime
            .start()
            .map_err(|status| format!("start voice kcoro runtime failed: {status}"))?;
        service
            .start()
            .map_err(|status| format!("start voice service failed: {status}"))?;
        control
            .notify()
            .map_err(|status| format!("start voice state machine failed: {status}"))?;
        Ok(VoiceRuntime {
            stop: stop.clone(),
            interrupt: interrupt.clone(),
            mic_enabled: mic_enabled.clone(),
            runtime,
            service: Some(service),
            control,
            audio: audio.clone(),
            done: done.clone(),
            engine: handoff,
            closed: AtomicBool::new(false),
        })
    }

    /// Start the retained voice service and return immediately.
    pub fn start(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32, u32, u32) -> Result<Box<dyn VoiceEngine>, String>
            + Send
            + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<Self, String> {
        Self::prepare(cfg, build_engine, sink)
    }

    /// Abort the current reply and flush queued playback.
    pub fn interrupt(&self) -> Result<(), String> {
        if self.is_finished() {
            return Ok(());
        }
        self.interrupt.store(true, Ordering::SeqCst);
        if let Err(code) = self.control.notify() {
            self.interrupt.store(false, Ordering::SeqCst);
            if self.is_finished() {
                return Ok(());
            }
            return Err(format!(
                "native interrupt edge was rejected with status {code}"
            ));
        }
        Ok(())
    }

    /// Pause/resume mic capture without ending the session.
    pub fn set_mic_enabled(&self, on: bool) -> Result<(), String> {
        if self.is_finished() {
            return Ok(());
        }
        self.mic_enabled.store(on, Ordering::SeqCst);
        if let Err(code) = self.control.notify() {
            if self.is_finished() {
                return Ok(());
            }
            return Err(format!(
                "native mic-control edge was rejected with status {code}"
            ));
        }
        Ok(())
    }

    /// Whether mic capture is currently allowed.
    pub fn mic_enabled(&self) -> bool {
        self.mic_enabled.load(Ordering::SeqCst)
    }

    pub fn audio_stats(&self) -> AudioStatsSnapshot {
        self.audio.snapshot()
    }

    /// Whether the retained service has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.done.load(Ordering::SeqCst)
            || self
                .service
                .as_ref()
                .is_some_and(CoroutineService::callback_panicked)
    }

    /// Signal stop and administratively join the retained service.
    pub fn stop(mut self) -> Result<(), String> {
        self.shutdown()
    }

    fn shutdown(&mut self) -> Result<(), String> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let mut errors = Vec::new();
        self.stop.store(true, Ordering::SeqCst);
        let already_done = self.done.load(Ordering::Acquire);
        let notified = already_done || self.control.notify().is_ok();
        if let Some(service) = self.service.take() {
            /* Normal shutdown is an owned state transition, not an
             * out-of-band service kill. The control edge makes LiveTask ask
             * native to stop; its final correlated event returns Complete.
             * OwnerSession then drops both CPAL streams before kcoro closes
             * notifier admission. Only an already-closed service needs the
             * administrative stop fallback. */
            if !notified {
                errors.push("voice stop edge was rejected before terminal settlement".into());
                service.stop();
            }
            if let Err(code) = service.join() {
                errors.push(format!("join voice continuation failed with status {code}"));
            }
            if service.callback_panicked() {
                errors.push("voice continuation panicked before terminal settlement".into());
            }
            if let Some(code) = service.reschedule_error() {
                errors.push(format!(
                    "voice continuation reschedule failed with status {code}"
                ));
            }
            drop(service);
        }
        if let Some(mut engine) = self.engine.take() {
            if let Err(error) = engine.stop_session() {
                errors.push(error);
            }
        }
        self.runtime.stop();
        if let Err(code) = self.runtime.join() {
            errors.push(format!(
                "join voice kcoro runtime failed with status {code}"
            ));
        }
        self.done.store(true, Ordering::SeqCst);
        if errors.is_empty() {
            return Ok(());
        }
        Err(errors.join("; "))
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("[flashkern] voice runtime teardown failed: {error}");
        }
    }
}

fn emit(sink: &mut EventSink, event: RuntimeEvent) -> bool {
    sink(event)
}

fn emit_or_stop(sink: &mut EventSink, stop: &Arc<AtomicBool>, event: RuntimeEvent) -> bool {
    if emit(sink, event) {
        return true;
    }
    stop.store(true, Ordering::SeqCst);
    false
}

type EventSink = Box<dyn FnMut(RuntimeEvent) -> bool + Send + 'static>;

struct SessionInit {
    resources: InitResources,
    sink: EventSink,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    device_fault: Arc<AtomicU32>,
    control: Option<SharedRealtimeNotifier>,
    audio: Stats,
    in_rate: u32,
    in_request_frames: u32,
    in_max_callback_frames: u32,
    out_rate: u32,
    done: Arc<AtomicBool>,
    events: Option<RealtimeNotifier>,
}

struct InitResources {
    capture: Option<Box<dyn CaptureSink>>,
    source: Option<Box<dyn PlaybackSource>>,
    engine: Option<Box<dyn VoiceEngine>>,
    handoff: Arc<EngineHandoff>,
    stopping: bool,
}

impl InitResources {
    fn new(engine: Box<dyn VoiceEngine>, handoff: Arc<EngineHandoff>) -> Self {
        Self {
            capture: None,
            source: None,
            engine: Some(engine),
            handoff,
            stopping: false,
        }
    }

    fn engine_mut(&mut self) -> Result<&mut Box<dyn VoiceEngine>, String> {
        self.engine
            .as_mut()
            .ok_or_else(|| "voice engine was already consumed".into())
    }

    fn capture_rate(&self) -> Result<u32, String> {
        self.capture
            .as_ref()
            .map(|capture| capture.rate())
            .ok_or_else(|| "platform capture endpoint was already consumed".into())
    }

    fn retire_engine(&mut self) {
        self.request_stop();
        if let Some(engine) = self.engine.take() {
            self.handoff.publish(engine);
        }
    }

    fn request_stop(&mut self) {
        if self.stopping {
            return;
        }
        self.stopping = true;
        if let Some(engine) = self.engine.as_mut() {
            engine.request_stop();
        }
    }
}

impl Drop for InitResources {
    fn drop(&mut self) {
        /* Close native admission before hardware endpoint destruction. A
         * capture producer disappearing while the session is still live is a
         * real device-loss edge; administrative shutdown must not forge one. */
        self.request_stop();
        drop(self.capture.take());
        drop(self.source.take());
        self.retire_engine();
    }
}

enum SessionTask {
    Init(Option<SessionInit>),
    Live(LiveTask),
    Done,
}

/// One permanently-owned coroutine state machine. Platform stream handles are
/// constructed by kcoro's owner initializer and live beside the continuation,
/// never in TLS or on a blocked task stack. Retirement first stops hardware
/// callbacks, then settles the ticketed session state.
struct OwnerSession {
    task: SessionTask,
    streams: Option<DeviceStreams>,
    setup_error: Option<String>,
    retired: bool,
}

impl OwnerSession {
    fn new(mut task: SessionTask) -> Self {
        let setup = match &mut task {
            SessionTask::Init(Some(init)) if !init.stop.load(Ordering::Acquire) => {
                start_devices(init).map(Some)
            }
            SessionTask::Init(_) | SessionTask::Live(_) | SessionTask::Done => Ok(None),
        };
        match setup {
            Ok(streams) => Self {
                task,
                streams,
                setup_error: None,
                retired: false,
            },
            Err(error) => Self {
                task,
                streams: None,
                setup_error: Some(error),
                retired: false,
            },
        }
    }

    fn advance(&mut self) -> ServiceOutcome {
        if let Some(error) = self.setup_error.take() {
            self.task.report_error(error);
            self.retire();
            return ServiceOutcome::Complete;
        }
        self.quiesce();
        let outcome = self.task.advance();
        self.quiesce();
        if outcome == ServiceOutcome::Complete {
            self.retire();
        }
        outcome
    }

    fn quiesce(&mut self) {
        if !self.task.stop_requested() {
            return;
        }
        /* Stop closes native admission synchronously; endpoint destruction can
         * then retire callback ownership without being misclassified as an
         * unexpected device loss. No join occurs until both streams are gone,
         * so this ordering creates no callback/session cycle. */
        self.task.begin_stop();
        drop(self.streams.take());
    }

    fn retire(&mut self) {
        if self.retired {
            return;
        }
        self.task.begin_stop();
        drop(self.streams.take());
        self.task.finish();
        self.retired = true;
    }
}

impl Drop for OwnerSession {
    fn drop(&mut self) {
        self.retire();
    }
}

impl SessionTask {
    fn stop_requested(&self) -> bool {
        match self {
            Self::Init(Some(init)) => init.stop.load(Ordering::Acquire),
            Self::Live(task) => task.stop.load(Ordering::Acquire),
            Self::Init(None) | Self::Done => false,
        }
    }

    fn begin_stop(&mut self) {
        match self {
            Self::Init(Some(init)) => init.resources.request_stop(),
            Self::Live(task) => task.begin_stop(),
            Self::Init(None) | Self::Done => {}
        }
    }

    fn advance(&mut self) -> ServiceOutcome {
        let state = std::mem::replace(self, Self::Done);
        match state {
            Self::Init(Some(init)) if init.stop.load(Ordering::Acquire) => {
                *self = Self::Init(Some(init));
                ServiceOutcome::Complete
            }
            Self::Init(Some(init)) => match init.mount() {
                Ok(next) => {
                    *self = next;
                    ServiceOutcome::Continue
                }
                Err((mut init, message)) => {
                    let _ = emit_or_stop(
                        &mut init.sink,
                        &init.stop,
                        RuntimeEvent::Error(message.clone()),
                    );
                    let _ = emit_or_stop(
                        &mut init.sink,
                        &init.stop,
                        RuntimeEvent::Ended(Some(message)),
                    );
                    init.done.store(true, Ordering::Release);
                    *self = Self::Init(Some(init));
                    ServiceOutcome::Complete
                }
            },
            Self::Init(None) | Self::Done => ServiceOutcome::Complete,
            Self::Live(mut task) => {
                let outcome = task.step();
                *self = Self::Live(task);
                outcome
            }
        }
    }

    fn report_error(&mut self, message: String) {
        let Self::Init(Some(init)) = self else {
            return;
        };
        let _ = emit_or_stop(
            &mut init.sink,
            &init.stop,
            RuntimeEvent::Error(message.clone()),
        );
        let _ = emit_or_stop(
            &mut init.sink,
            &init.stop,
            RuntimeEvent::Ended(Some(message)),
        );
        init.done.store(true, Ordering::Release);
    }

    fn finish(&mut self) {
        match std::mem::replace(self, Self::Done) {
            Self::Init(Some(mut init)) => {
                init.done.store(true, Ordering::Release);
                init.resources.retire_engine();
            }
            Self::Live(mut task) => {
                task.done.store(true, Ordering::Release);
                task.finish();
            }
            Self::Init(None) | Self::Done => {}
        }
    }
}

impl SessionInit {
    fn mount(mut self) -> Result<SessionTask, (Self, String)> {
        let Some(mut engine) = self.resources.engine.take() else {
            return Err((self, "voice engine was already consumed".into()));
        };
        let Some(events) = self.events.take() else {
            self.resources.engine = Some(engine);
            return Err((self, "voice event notifier was already consumed".into()));
        };
        if let Err(error) = engine.mount_events(events) {
            self.resources.engine = Some(engine);
            return Err((self, format!("mount native event edge: {error}")));
        }
        if !emit_or_stop(
            &mut self.sink,
            &self.stop,
            RuntimeEvent::State(SessionState::Listening),
        ) {
            self.resources.engine = Some(engine);
            return Err((self, "voice event sink rejected listening state".into()));
        }
        Ok(SessionTask::Live(LiveTask {
            sink: self.sink,
            stop: self.stop,
            interrupt: self.interrupt,
            mic_enabled: self.mic_enabled,
            playback_flush: self.playback_flush,
            device_fault: self.device_fault,
            done: self.done,
            handoff: self.resources.handoff.clone(),
            stopping: false,
            driver: Some(TurnDriver {
                engine,
                events: EventState::default(),
            }),
        }))
    }
}

#[derive(Default)]
struct EventState {
    transcript: String,
    speaking: bool,
    failed: bool,
}

impl EventState {
    fn emit(
        &mut self,
        event: VoiceEvent,
        sink: &mut EventSink,
        stop: &Arc<AtomicBool>,
        mic_enabled: &AtomicBool,
    ) {
        if self.failed {
            return;
        }
        let ok = match event {
            VoiceEvent::TurnStarted => {
                self.transcript.clear();
                self.speaking = false;
                emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Thinking))
            }
            VoiceEvent::Text(text) => {
                self.transcript.push_str(&text);
                let state = self.speaking
                    || emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Speaking));
                self.speaking = state;
                state
                    && emit_or_stop(
                        sink,
                        stop,
                        RuntimeEvent::Transcript(self.transcript.clone()),
                    )
            }
            VoiceEvent::TurnComplete | VoiceEvent::Interrupted => {
                self.transcript.clear();
                self.speaking = false;
                emit_ready(sink, stop, mic_enabled)
            }
            VoiceEvent::Error(error) => {
                self.transcript.clear();
                self.speaking = false;
                emit_or_stop(sink, stop, RuntimeEvent::Error(error))
                    && emit_ready(sink, stop, mic_enabled)
            }
        };
        self.failed = !ok;
    }
}

struct TurnDriver {
    engine: Box<dyn VoiceEngine>,
    events: EventState,
}

impl TurnDriver {
    fn interrupt(&mut self) -> Result<u64, String> {
        self.engine.interrupt_stream()
    }

    fn request_stop(&mut self) {
        self.engine.request_stop();
    }

    fn advance_events(
        &mut self,
        stop: &Arc<AtomicBool>,
        sink: &mut EventSink,
        mic_enabled: &AtomicBool,
    ) -> Result<EngineProgress, String> {
        self.engine.advance_events(stop, &mut |event| {
            self.events.emit(event, sink, stop, mic_enabled);
        })
    }
}

struct LiveTask {
    sink: EventSink,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    device_fault: Arc<AtomicU32>,
    done: Arc<AtomicBool>,
    handoff: Arc<EngineHandoff>,
    stopping: bool,
    driver: Option<TurnDriver>,
}

impl LiveTask {
    fn begin_stop(&mut self) {
        if self.stopping {
            return;
        }
        self.stopping = true;
        if let Some(driver) = self.driver.as_mut() {
            driver.request_stop();
        }
    }

    fn step(&mut self) -> ServiceOutcome {
        if self.driver.is_none() {
            return ServiceOutcome::Complete;
        }
        let faults = self.device_fault.swap(0, Ordering::AcqRel);
        if faults != 0 {
            let device = match faults {
                DEVICE_FAULT_INPUT => "input",
                DEVICE_FAULT_OUTPUT => "output",
                _ => "input and output",
            };
            let _ = emit_or_stop(
                &mut self.sink,
                &self.stop,
                RuntimeEvent::Error(format!("platform audio {device} device stopped")),
            );
            self.stop.store(true, Ordering::Release);
        }
        if !self.stopping && self.interrupt.swap(false, Ordering::SeqCst) {
            let interrupt = self
                .driver
                .as_mut()
                .expect("driver checked above")
                .interrupt();
            if let Err(error) = interrupt {
                let _ = emit_or_stop(
                    &mut self.sink,
                    &self.stop,
                    RuntimeEvent::Error(format!("native interrupt failed: {error}")),
                );
                self.stop.store(true, Ordering::Release);
                return ServiceOutcome::Dormant;
            }
            self.playback_flush.store(true, Ordering::SeqCst);
            if !emit_ready(&mut self.sink, &self.stop, &self.mic_enabled) {
                return ServiceOutcome::Dormant;
            }
        }
        match self
            .driver
            .as_mut()
            .expect("driver checked above")
            .advance_events(&self.stop, &mut self.sink, &self.mic_enabled)
        {
            Ok(EngineProgress::Continue | EngineProgress::Complete) => ServiceOutcome::Continue,
            Ok(EngineProgress::Dormant) => ServiceOutcome::Dormant,
            Ok(EngineProgress::Stopped) => ServiceOutcome::Complete,
            Err(error) => {
                let _ = emit_or_stop(
                    &mut self.sink,
                    &self.stop,
                    RuntimeEvent::Error(format!("native event drain failed: {error}")),
                );
                self.stop.store(true, Ordering::Release);
                ServiceOutcome::Dormant
            }
        }
    }

    fn finish(&mut self) {
        // OwnerSession has already destroyed both hardware streams, so no
        // callback can still hold a ticketed capture/playback endpoint while
        // this state crosses the terminal handoff.
        if let Some(mut driver) = self.driver.take() {
            driver.request_stop();
            self.handoff.publish(driver.engine);
        }
        let _ = emit_or_stop(
            &mut self.sink,
            &self.stop,
            RuntimeEvent::State(SessionState::Idle),
        );
        let _ = emit_or_stop(&mut self.sink, &self.stop, RuntimeEvent::Ended(None));
        self.done.store(true, Ordering::Release);
    }
}

impl Drop for LiveTask {
    fn drop(&mut self) {
        if !self.done.load(Ordering::Acquire) {
            self.finish();
        }
    }
}

fn ready_state(mic_enabled: &AtomicBool) -> SessionState {
    if mic_enabled.load(Ordering::SeqCst) {
        return SessionState::Listening;
    }
    SessionState::Idle
}

fn emit_ready(sink: &mut EventSink, stop: &Arc<AtomicBool>, mic_enabled: &AtomicBool) -> bool {
    emit_or_stop(sink, stop, RuntimeEvent::State(ready_state(mic_enabled)))
        && emit_or_stop(sink, stop, RuntimeEvent::Level(0.0))
}

#[cfg(feature = "audio-io")]
fn start_devices(init: &mut SessionInit) -> Result<DeviceStreams, String> {
    let capture = init
        .resources
        .capture
        .take()
        .ok_or("platform capture endpoint was already consumed")?;
    let source = init
        .resources
        .source
        .take()
        .ok_or("platform playback endpoint was already consumed")?;
    let control = init
        .control
        .as_ref()
        .ok_or("voice control edge was not installed before device start")?
        .clone();
    if source.rate() != init.out_rate {
        return Err(format!(
            "native PCM rate does not match the output device: received {} Hz, expected {} Hz",
            source.rate(),
            init.out_rate
        ));
    }
    let output = start_output(
        source,
        init.audio.clone(),
        init.playback_flush.clone(),
        init.device_fault.clone(),
        control.clone(),
    )
    .map_err(|error| format!("audio output: {error}"))?;
    let input = start_input(
        capture,
        init.in_rate,
        init.in_request_frames,
        init.in_max_callback_frames,
        init.mic_enabled.clone(),
        init.audio.clone(),
        init.device_fault.clone(),
        control,
    )
    .map_err(|error| format!("audio input: {error}"))?;
    Ok(DeviceStreams {
        _input: input,
        _output: output,
    })
}

#[cfg(not(feature = "audio-io"))]
fn start_devices(_init: &mut SessionInit) -> Result<DeviceStreams, String> {
    // No callback can own these endpoints in a build without platform audio;
    // retire them deterministically before reporting the configuration error.
    drop(_init.resources.capture.take());
    drop(_init.resources.source.take());
    let _ = (
        _init.in_rate,
        _init.in_request_frames,
        _init.in_max_callback_frames,
        _init.out_rate,
        _init.audio.snapshot(),
        &_init.device_fault,
        &_init.control,
    );
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(feature = "audio-io")]
fn default_input_contract() -> Res<CaptureContract> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 {
        return Err("audio input sample rate is zero".into());
    }
    capture_contract(rate, supported.buffer_size()).map_err(Into::into)
}

#[cfg(not(feature = "audio-io"))]
fn default_input_contract() -> Res<CaptureContract> {
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(feature = "audio-io")]
fn default_output_rate() -> Res<u32> {
    let host = cpal::default_host();
    let dev = host
        .default_output_device()
        .ok_or("no default output device")?;
    let rate = dev.default_output_config()?.sample_rate().0;
    if rate == 0 {
        return Err("audio output sample rate is zero".into());
    }
    Ok(rate)
}

#[cfg(not(feature = "audio-io"))]
fn default_output_rate() -> Res<u32> {
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(feature = "audio-io")]
#[inline]
fn gate_capture_callback(
    failed: &AtomicBool,
    frames: usize,
    max_frames: usize,
    terminal: impl FnOnce(),
) -> bool {
    if failed.load(Ordering::Acquire) {
        return true;
    }
    if frames <= max_frames {
        return false;
    }
    if !failed.swap(true, Ordering::AcqRel) {
        terminal();
    }
    true
}

#[cfg(feature = "audio-io")]
fn start_input(
    sink: Box<dyn CaptureSink>,
    expected_rate: u32,
    expected_request_frames: u32,
    expected_max_callback_frames: u32,
    enabled: Arc<AtomicBool>,
    audio: Stats,
    fault: Arc<AtomicU32>,
    control: SharedRealtimeNotifier,
) -> Res<HostStream> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 || rate != expected_rate {
        return Err("audio input sample rate changed during setup".into());
    }
    let contract = capture_contract(rate, supported.buffer_size())?;
    if contract.requested_frames != expected_request_frames
        || contract.max_callback_frames != expected_max_callback_frames
    {
        return Err(format!(
            "audio input callback contract changed during setup: negotiated request/max {expected_request_frames}/{expected_max_callback_frames}, device now selects {}/{}",
            contract.requested_frames, contract.max_callback_frames,
        )
        .into());
    }
    let sealed = sink.max_callback_frames();
    if sealed != expected_max_callback_frames {
        return Err(format!(
            "native capture admission is sealed at {sealed} frames, expected {expected_max_callback_frames}"
        )
        .into());
    }
    let channels = supported.channels() as usize;
    if channels == 0 {
        return Err("audio input device exposes zero channels".into());
    }
    let fmt = supported.sample_format();
    let mut cfg: cpal::StreamConfig = supported.into();
    cfg.buffer_size = cpal::BufferSize::Fixed(expected_request_frames);
    macro_rules! stream {
        ($t:ty, $write:ident) => {{
            let mut sink = sink;
            let enabled = enabled.clone();
            let audio = audio.clone();
            let callback_fault = fault.clone();
            let error_fault = fault.clone();
            let callback_control = control.clone();
            let error_control = control.clone();
            let failed = Arc::new(AtomicBool::new(false));
            let callback_failed = failed.clone();
            dev.build_input_stream(
                &cfg,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    let frames = data.len().div_ceil(channels);
                    if gate_capture_callback(&callback_failed, frames, sealed as usize, || {
                        /* Preserve one-callback/one-record semantics. The
                         * native endpoint rejects this whole callback and
                         * publishes its explicit GAP/XRUN descriptor; the
                         * correlated device-fault edge then makes the owner
                         * retire the stream instead of repeatedly dropping. */
                        let write = sink.$write(data, channels);
                        audio
                            .dropped_samples
                            .fetch_add(write.dropped_frames as u64, Ordering::Relaxed);
                        callback_fault.fetch_or(DEVICE_FAULT_INPUT, Ordering::Release);
                        let _ = callback_control.notify();
                    }) {
                        return;
                    }
                    if !enabled.load(Ordering::Acquire) {
                        let _ = sink.mute(frames, channels);
                        return;
                    }
                    let write = sink.$write(data, channels);
                    if write.dropped_frames != 0 {
                        audio
                            .dropped_samples
                            .fetch_add(write.dropped_frames as u64, Ordering::Relaxed);
                    }
                },
                move |_| {
                    if failed.swap(true, Ordering::AcqRel) {
                        return;
                    }
                    error_fault.fetch_or(DEVICE_FAULT_INPUT, Ordering::Release);
                    let _ = error_control.notify();
                },
                None,
            )
        }};
    }

    let stream = match fmt {
        cpal::SampleFormat::F32 => stream!(f32, write_f32),
        cpal::SampleFormat::I16 => stream!(i16, write_i16),
        cpal::SampleFormat::U16 => stream!(u16, write_u16),
        other => return Err(format!("unsupported input sample format {other:?}").into()),
    }?;
    stream.play()?;
    Ok(stream)
}

#[cfg(feature = "audio-io")]
fn capture_contract(
    rate: u32,
    supported: &cpal::SupportedBufferSize,
) -> Result<CaptureContract, String> {
    if rate == 0 {
        return Err("native capture callback contract is empty".into());
    }
    let cpal::SupportedBufferSize::Range { min, max } = supported else {
        return Err("input device does not expose a sealable callback-buffer range".into());
    };
    if min > max || *max == 0 {
        return Err("input device exposed an invalid callback-buffer range".into());
    }
    /* Some CPAL backends use zero to mean that the backend imposes no lower
     * bound. One frame is the smallest usable fixed request; zero itself is
     * never submitted to the device. */
    let min = (*min).max(1);
    let requested = ((rate as u64 * CAPTURE_CALLBACK_REQUEST_MS as u64) + 999) / 1_000;
    let requested = u32::try_from(requested)
        .map_err(|_| "audio input callback request overflow".to_string())?;
    let admitted = ((rate as u64 * CAPTURE_CALLBACK_MAX_MS as u64) + 999) / 1_000;
    let admitted = u32::try_from(admitted)
        .map_err(|_| "audio input callback admission overflow".to_string())?;
    if min > admitted {
        return Err(format!(
            "input device requires at least {min} callback frames but native admission is sealed at {admitted}"
        ));
    }
    Ok(CaptureContract {
        rate,
        requested_frames: requested.max(min).min(*max),
        max_callback_frames: admitted,
    })
}

#[cfg(feature = "audio-io")]
fn start_output(
    source: Box<dyn PlaybackSource>,
    audio: Stats,
    flush: Arc<AtomicBool>,
    fault: Arc<AtomicU32>,
    control: SharedRealtimeNotifier,
) -> Res<HostStream> {
    let host = cpal::default_host();
    let dev = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = dev.default_output_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 {
        return Err("audio output sample rate is zero".into());
    }
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    if source.rate() != rate {
        return Err("audio output sample rate changed during setup".into());
    }
    macro_rules! stream {
        ($t:ty, $write:ident) => {{
            let mut source = source;
            let flush = flush.clone();
            let audio = audio.clone();
            let fault = fault.clone();
            let control = control.clone();
            dev.build_output_stream(
                &cfg,
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let reset = flush.swap(false, Ordering::AcqRel);
                    let write = source.$write(data, channels, reset);
                    if write.claimed_samples != 0 {
                        audio
                            .decoded_samples
                            .fetch_add(write.claimed_samples as u64, Ordering::Relaxed);
                        audio
                            .queued_samples
                            .fetch_add(write.claimed_samples as u64, Ordering::Relaxed);
                    }
                    if write.dropped_samples != 0 {
                        audio
                            .dropped_samples
                            .fetch_add(write.dropped_samples as u64, Ordering::Relaxed);
                    }
                    if write.played_frames != 0 {
                        audio
                            .played_samples
                            .fetch_add(write.played_frames as u64, Ordering::Relaxed);
                    }
                    if write.underrun_frames != 0 {
                        audio
                            .underrun_frames
                            .fetch_add(write.underrun_frames as u64, Ordering::Relaxed);
                    }
                },
                move |_| {
                    fault.fetch_or(DEVICE_FAULT_OUTPUT, Ordering::Release);
                    let _ = control.notify();
                },
                None,
            )
        }};
    }

    let stream = match fmt {
        cpal::SampleFormat::F32 => stream!(f32, write_f32),
        cpal::SampleFormat::I16 => stream!(i16, write_i16),
        cpal::SampleFormat::U16 => stream!(u16, write_u16),
        other => return Err(format!("unsupported output sample format {other:?}").into()),
    }?;
    stream.play()?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn audio_stats_do_not_fabricate_turn_latency() {
        let stats = AudioStats::default();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.turn_count, 0);
        assert_eq!(snapshot.last_turn_latency_ms, 0);
        assert_eq!(snapshot.mean_turn_latency_ms, 0);
    }

    #[test]
    fn terminal_turn_state_follows_mic_enabled() {
        let mic = AtomicBool::new(true);
        assert_eq!(ready_state(&mic), SessionState::Listening);

        mic.store(false, Ordering::SeqCst);
        assert_eq!(ready_state(&mic), SessionState::Idle);
    }

    #[test]
    fn rejected_sink_preserves_native_stopped_progress() {
        struct Stopped;

        impl VoiceEngine for Stopped {
            fn take_capture_sink(&mut self) -> Result<Option<Box<dyn CaptureSink>>, String> {
                Ok(None)
            }

            fn take_playback_source(
                &mut self,
                _: RealtimeNotifier,
            ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
                Ok(None)
            }

            fn mount_events(&mut self, _: RealtimeNotifier) -> Result<(), String> {
                Ok(())
            }

            fn advance_events(
                &mut self,
                _: &AtomicBool,
                emit: &mut dyn FnMut(VoiceEvent),
            ) -> Result<EngineProgress, String> {
                emit(VoiceEvent::TurnStarted);
                Ok(EngineProgress::Stopped)
            }

            fn interrupt_stream(&mut self) -> Result<u64, String> {
                Ok(1)
            }

            fn request_stop(&mut self) {}

            fn stop_session(&mut self) -> Result<(), String> {
                Ok(())
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        let mic = AtomicBool::new(true);
        let mut sink: EventSink = Box::new(|_| false);
        let mut driver = TurnDriver {
            engine: Box::new(Stopped),
            events: EventState::default(),
        };
        assert_eq!(
            driver.advance_events(&stop, &mut sink, &mic).unwrap(),
            EngineProgress::Stopped
        );
        assert!(driver.events.failed);
        assert!(stop.load(Ordering::Acquire));
    }

    #[cfg(feature = "audio-io")]
    #[test]
    fn capture_request_is_distinct_from_native_admission_bound() {
        let range = cpal::SupportedBufferSize::Range {
            min: 128,
            max: 2_048,
        };
        assert_eq!(
            capture_contract(48_000, &range).unwrap(),
            CaptureContract {
                rate: 48_000,
                requested_frames: 960,
                max_callback_frames: 1_920,
            }
        );
        assert_eq!(
            capture_contract(44_100, &range).unwrap(),
            CaptureContract {
                rate: 44_100,
                requested_frames: 882,
                max_callback_frames: 1_764,
            }
        );
        assert_eq!(
            capture_contract(
                48_000,
                &cpal::SupportedBufferSize::Range {
                    min: 0,
                    max: u32::MAX,
                },
            )
            .unwrap(),
            CaptureContract {
                rate: 48_000,
                requested_frames: 960,
                max_callback_frames: 1_920,
            }
        );

        let oversized = cpal::SupportedBufferSize::Range {
            min: 2_048,
            max: 4_096,
        };
        assert!(capture_contract(48_000, &oversized).is_err());
        assert!(capture_contract(48_000, &cpal::SupportedBufferSize::Unknown).is_err());
    }

    #[cfg(feature = "audio-io")]
    #[test]
    fn oversized_callback_publishes_one_terminal_action_then_stays_retired() {
        let failed = AtomicBool::new(false);
        let calls = std::sync::atomic::AtomicUsize::new(0);
        assert!(!gate_capture_callback(&failed, 960, 1_920, || {
            calls.fetch_add(1, Ordering::Relaxed);
        }));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(gate_capture_callback(&failed, 1_921, 1_920, || {
            calls.fetch_add(1, Ordering::Relaxed);
        }));
        assert!(gate_capture_callback(&failed, 1_921, 1_920, || {
            calls.fetch_add(1, Ordering::Relaxed);
        }));
        assert!(gate_capture_callback(&failed, 960, 1_920, || {
            calls.fetch_add(1, Ordering::Relaxed);
        }));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn abandoned_initializer_retires_endpoints_before_engine_handoff() {
        struct Capture(Arc<Mutex<Vec<&'static str>>>);
        struct Playback(Arc<Mutex<Vec<&'static str>>>);
        struct Engine(Arc<Mutex<Vec<&'static str>>>);

        impl Drop for Capture {
            fn drop(&mut self) {
                self.0.lock().expect("lifecycle log").push("capture-drop");
            }
        }

        impl Drop for Playback {
            fn drop(&mut self) {
                self.0.lock().expect("lifecycle log").push("playback-drop");
            }
        }

        impl Drop for Engine {
            fn drop(&mut self) {
                self.0.lock().expect("lifecycle log").push("engine-drop");
            }
        }

        impl CaptureSink for Capture {
            fn rate(&self) -> u32 {
                48_000
            }

            fn max_callback_frames(&self) -> u32 {
                1_920
            }

            fn write_f32(&mut self, _: &[f32], _: usize) -> crate::voice_api::CaptureWrite {
                crate::voice_api::CaptureWrite::default()
            }

            fn write_i16(&mut self, _: &[i16], _: usize) -> crate::voice_api::CaptureWrite {
                crate::voice_api::CaptureWrite::default()
            }

            fn write_u16(&mut self, _: &[u16], _: usize) -> crate::voice_api::CaptureWrite {
                crate::voice_api::CaptureWrite::default()
            }

            fn mute(&mut self, _: usize, _: usize) -> crate::voice_api::CaptureMute {
                crate::voice_api::CaptureMute::default()
            }
        }

        impl PlaybackSource for Playback {
            fn rate(&self) -> u32 {
                48_000
            }

            fn write_f32(
                &mut self,
                _: &mut [f32],
                _: usize,
                _: bool,
            ) -> crate::voice_api::PlaybackWrite {
                crate::voice_api::PlaybackWrite::default()
            }

            fn write_i16(
                &mut self,
                _: &mut [i16],
                _: usize,
                _: bool,
            ) -> crate::voice_api::PlaybackWrite {
                crate::voice_api::PlaybackWrite::default()
            }

            fn write_u16(
                &mut self,
                _: &mut [u16],
                _: usize,
                _: bool,
            ) -> crate::voice_api::PlaybackWrite {
                crate::voice_api::PlaybackWrite::default()
            }
        }

        impl VoiceEngine for Engine {
            fn take_capture_sink(&mut self) -> Result<Option<Box<dyn CaptureSink>>, String> {
                Ok(None)
            }

            fn take_playback_source(
                &mut self,
                _: RealtimeNotifier,
            ) -> Result<Option<Box<dyn PlaybackSource>>, String> {
                Ok(None)
            }

            fn mount_events(&mut self, _: RealtimeNotifier) -> Result<(), String> {
                Ok(())
            }

            fn advance_events(
                &mut self,
                _: &AtomicBool,
                _: &mut dyn FnMut(VoiceEvent),
            ) -> Result<EngineProgress, String> {
                Ok(EngineProgress::Dormant)
            }

            fn interrupt_stream(&mut self) -> Result<u64, String> {
                Ok(1)
            }

            fn request_stop(&mut self) {
                self.0.lock().expect("lifecycle log").push("request-stop");
            }

            fn stop_session(&mut self) -> Result<(), String> {
                self.0.lock().expect("lifecycle log").push("stop-session");
                Ok(())
            }
        }

        let log = Arc::new(Mutex::new(Vec::new()));
        let handoff = EngineHandoff::new();
        let mut resources = InitResources::new(Box::new(Engine(log.clone())), handoff.clone());
        resources.capture = Some(Box::new(Capture(log.clone())));
        resources.source = Some(Box::new(Playback(log.clone())));
        let runtime = Arc::new(CoroutineRuntime::new().expect("lifecycle runtime"));
        let service = runtime.service(|| {}).expect("lifecycle service");
        let init = SessionInit {
            resources,
            sink: Box::new(|_| false),
            stop: Arc::new(AtomicBool::new(false)),
            interrupt: Arc::new(AtomicBool::new(false)),
            mic_enabled: Arc::new(AtomicBool::new(true)),
            playback_flush: Arc::new(AtomicBool::new(false)),
            device_fault: Arc::new(AtomicU32::new(0)),
            control: Some(service.shared_realtime_notifier()),
            audio: Arc::new(AudioStats::default()),
            in_rate: 48_000,
            in_request_frames: 960,
            in_max_callback_frames: 1_920,
            out_rate: 48_000,
            done: Arc::new(AtomicBool::new(false)),
            events: Some(service.realtime_notifier().expect("event edge")),
        };
        let failed = match init.mount() {
            Err((init, error)) => {
                assert!(error.contains("rejected listening state"));
                init
            }
            Ok(_) => panic!("rejecting sink unexpectedly admitted the session"),
        };
        drop(failed);
        assert_eq!(
            log.lock().expect("lifecycle log").as_slice(),
            ["request-stop", "capture-drop", "playback-drop"]
        );

        let mut engine = handoff
            .take()
            .expect("engine crosses administrative handoff");
        engine.stop_session().expect("administrative settlement");
        drop(engine);
        assert_eq!(
            log.lock().expect("lifecycle log").as_slice(),
            [
                "request-stop",
                "capture-drop",
                "playback-drop",
                "stop-session",
                "engine-drop",
            ]
        );
    }

    #[test]
    fn interrupt_is_a_fallible_realtime_control_edge() {
        let runtime = Arc::new(CoroutineRuntime::new().unwrap());
        let service = runtime.service(|| {}).unwrap();
        let control = service.shared_realtime_notifier();
        runtime.start().unwrap();
        service.start().unwrap();
        let live = VoiceRuntime {
            stop: Arc::new(AtomicBool::new(false)),
            interrupt: Arc::new(AtomicBool::new(false)),
            mic_enabled: Arc::new(AtomicBool::new(true)),
            runtime,
            service: Some(service),
            control,
            audio: Arc::new(AudioStats::default()),
            done: Arc::new(AtomicBool::new(false)),
            engine: EngineHandoff::new(),
            closed: AtomicBool::new(false),
        };

        assert!(!live.interrupt.load(Ordering::SeqCst));

        live.interrupt().unwrap();

        assert!(live.interrupt.load(Ordering::SeqCst));

        // This unit fixture uses a generic dormant service rather than the
        // production OwnerSession state machine, so retire it explicitly.
        live.service.as_ref().unwrap().stop();
        live.service.as_ref().unwrap().join().unwrap();
        live.closed.store(true, Ordering::Release);
    }

    #[test]
    fn ready_transition_clears_output_level() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let mut sink: EventSink = Box::new(move |event| {
            captured.lock().expect("events lock").push(event);
            true
        });
        let stop = Arc::new(AtomicBool::new(false));
        let mic = AtomicBool::new(false);

        assert!(emit_ready(&mut sink, &stop, &mic));
        let events = events.lock().expect("events lock");
        assert!(matches!(
            events.as_slice(),
            [
                RuntimeEvent::State(SessionState::Idle),
                RuntimeEvent::Level(level)
            ] if *level == 0.0
        ));
    }
}
