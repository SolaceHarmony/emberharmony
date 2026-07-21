//! Callback-driven host audio service mounted on one retained kcoro continuation.
//!
//! Platform callbacks publish bounded native PCM/control edges and return. The
//! retained [`SessionTask`] owns only device handles, outward UI delivery, and
//! engine-event draining; native kcoro owns detector, turn, and model state. It
//! never polls a clock or blocks an operating-system thread for model progress.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use kcoro_sys::{
    RealtimeNotifier, Runtime as CoroutineRuntime, RuntimeConfig as CoroutineConfig,
    Service as CoroutineService, ServiceOutcome, SharedRealtimeNotifier,
};
use serde::{Deserialize, Serialize};

use crate::{
    default_platform_audio_config, EngineProgress, PlatformAudioSnapshot, VoiceEngine, VoiceEvent,
};

type Stats = Arc<AudioStats>;

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

    fn publish(&self, snapshot: PlatformAudioSnapshot) {
        self.decoded_samples
            .store(snapshot.claimed_playback_frames, Ordering::Relaxed);
        self.queued_samples
            .store(snapshot.claimed_playback_frames, Ordering::Relaxed);
        self.dropped_samples.store(
            snapshot
                .dropped_capture_frames
                .saturating_add(snapshot.dropped_playback_frames),
            Ordering::Relaxed,
        );
        self.played_samples
            .store(snapshot.played_frames, Ordering::Relaxed);
        self.underrun_frames
            .store(snapshot.silent_playback_frames, Ordering::Relaxed);
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
    /// and the sealed native callback size. Rust never owns a PCM endpoint.
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
        let audio = Arc::new(AudioStats::default());
        let done = Arc::new(AtomicBool::new(false));
        let device = default_platform_audio_config()
            .map_err(|error| format!("native platform audio: {error}"))?;
        let mut sink: EventSink = Box::new(sink);
        if !emit_or_stop(&mut sink, &stop, RuntimeEvent::State(SessionState::Loading)) {
            return Err("voice event sink rejected the loading state".into());
        }
        let mut engine = build_engine(
            device.capture_rate,
            device.playback_rate,
            device.capture_frames,
        )
        .map_err(|error| format!("model load: {error}"))?;
        engine
            .mount_platform_audio(device)
            .map_err(|error| format!("mount native platform audio: {error}"))?;
        let handoff = EngineHandoff::new();
        let resources = InitResources::new(engine, handoff.clone());
        let mut init = SessionInit {
            resources,
            sink,
            stop: stop.clone(),
            interrupt: interrupt.clone(),
            mic_enabled: mic_enabled.clone(),
            control: None,
            audio: audio.clone(),
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
                let control = setup.shared_realtime_notifier();
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
             * Native CoreAudio retirement closes callback admission before
             * kcoro closes notifier admission. Only an already-closed service
             * needs the administrative stop fallback. */
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
    control: Option<SharedRealtimeNotifier>,
    audio: Stats,
    done: Arc<AtomicBool>,
    events: Option<RealtimeNotifier>,
}

struct InitResources {
    engine: Option<Box<dyn VoiceEngine>>,
    handoff: Arc<EngineHandoff>,
    stopping: bool,
}

impl InitResources {
    fn new(engine: Box<dyn VoiceEngine>, handoff: Arc<EngineHandoff>) -> Self {
        Self {
            engine: Some(engine),
            handoff,
            stopping: false,
        }
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
        self.request_stop();
        self.retire_engine();
    }
}

enum SessionTask {
    Init(Option<SessionInit>),
    Live(LiveTask),
    Done,
}

/// One retained coroutine state machine for outward UI delivery and controls.
/// CoreAudio and every PCM lease live in the native session.
struct OwnerSession {
    task: SessionTask,
    retired: bool,
}

impl OwnerSession {
    fn new(task: SessionTask) -> Self {
        Self {
            task,
            retired: false,
        }
    }

    fn advance(&mut self) -> ServiceOutcome {
        if self.task.stop_requested() {
            self.task.begin_stop();
        }
        let outcome = self.task.advance();
        if self.task.stop_requested() {
            self.task.begin_stop();
            /* A sink/control failure may create the stop edge inside this
             * invocation. Dormancy would then consume that edge without a
             * successor. Keep this retained continuation runnable once so it
             * observes the native stop publication it just initiated. */
            if outcome == ServiceOutcome::Dormant {
                return ServiceOutcome::Continue;
            }
        }
        if outcome == ServiceOutcome::Complete {
            self.retire();
        }
        outcome
    }

    fn retire(&mut self) {
        if self.retired {
            return;
        }
        self.task.begin_stop();
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
        if let Err(error) = engine.start_platform_audio() {
            engine.request_stop();
            self.resources.engine = Some(engine);
            return Err((self, format!("start native platform audio: {error}")));
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
            capture_enabled: true,
            audio: self.audio,
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

    fn set_capture_enabled(&mut self, enabled: bool) -> Result<(), String> {
        self.engine.set_capture_enabled(enabled)
    }

    fn audio_snapshot(&self) -> Result<PlatformAudioSnapshot, String> {
        self.engine.platform_audio_snapshot()
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
    capture_enabled: bool,
    audio: Stats,
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
        let enabled = self.mic_enabled.load(Ordering::Acquire);
        if enabled != self.capture_enabled {
            let result = self
                .driver
                .as_mut()
                .expect("driver checked above")
                .set_capture_enabled(enabled);
            if let Err(error) = result {
                let _ = emit_or_stop(
                    &mut self.sink,
                    &self.stop,
                    RuntimeEvent::Error(format!("native capture control failed: {error}")),
                );
                self.stop.store(true, Ordering::Release);
                return ServiceOutcome::Dormant;
            }
            self.capture_enabled = enabled;
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
            if !emit_ready(&mut self.sink, &self.stop, &self.mic_enabled) {
                return ServiceOutcome::Dormant;
            }
        }
        if let Ok(snapshot) = self
            .driver
            .as_ref()
            .expect("driver checked above")
            .audio_snapshot()
        {
            self.audio.publish(snapshot);
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
        // The native engine retires CoreAudio before joining its session.
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
