//! Callback-driven host audio service mounted on one retained kcoro continuation.
//!
//! Platform callbacks publish bounded PCM/control edges and return. The retained
//! [`SessionTask`] owns the durable VAD and orchestration state and advances only
//! when a producer makes one of its predicates ready. It never polls a clock or
//! blocks an operating-system thread on behalf of model progress.

use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "audio-io")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use kcoro_sys::{
    RealtimeNotifier, Runtime as CoroutineRuntime, RuntimeConfig as CoroutineConfig,
    Service as CoroutineService, ServiceOutcome, SharedRealtimeNotifier,
};
use serde::{Deserialize, Serialize};

use crate::voice_api::{CaptureReservation, PlaybackSource};
use crate::{EngineProgress, VoiceEngine, VoiceEvent};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Capture = Arc<CaptureIngress>;
type Playback = Arc<PlaybackReference>;
type Stats = Arc<AudioStats>;
#[cfg(feature = "audio-io")]
type HostStream = cpal::Stream;
#[cfg(not(feature = "audio-io"))]
struct HostStream;

#[cfg(feature = "audio-io")]
struct DeviceStreams {
    _input: HostStream,
    _output: HostStream,
}

#[cfg(not(feature = "audio-io"))]
struct DeviceStreams;

const CAPTURE_BLOCKS: usize = 2;
const MAX_UTTERANCE_SECONDS: usize = 30;
const PLAYBACK_VAD_MULTIPLIER: f32 = 3.0;
const PLAYBACK_ECHO_MULTIPLIER: f32 = 2.5;
/// Barge-in sustain gate (spec 09, W5): while the assistant is audible, this many
/// CONSECUTIVE voiced VAD windows (200ms each at the in_rate/5 window size) are
/// required before an interrupt fires. One loud window is how echo blips and coughs
/// stop the assistant; 400ms of sustained speech is how a human interjects. When the
/// assistant is quiet the first voiced window still engages immediately.
const BARGE_IN_SUSTAIN_WINDOWS: usize = 2;
/// Echo-tail speech policy. This duration is converted once to capture frames;
/// microphone callbacks, including callbacks containing acoustic silence, advance
/// the cursor. A stopped device never masquerades as elapsed silence.
const PLAYBACK_ECHO_TAIL_MS: u64 = 700;

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

struct AudioInputWriter {
    capture: Capture,
    notify: RealtimeNotifier,
    // One endpoint owns the SPSC producer cursor. Cell is Send but not Sync,
    // preventing shared-reference publication from manufacturing another
    // concurrent producer over the native reservation.
    producer: std::marker::PhantomData<Cell<()>>,
}

impl AudioInputWriter {
    fn publish(&mut self, dropped: usize, frames: usize) -> usize {
        // The edge represents a callback/sample-clock transition, not only a
        // successful write. A seal racing an over-capacity callback may first
        // observe WRITING and yield; that callback restores OPEN even when it
        // drops the whole block, so it must resume the continuation to close
        // the turn. Coalescing happens inside the retained notifier.
        if frames != 0 {
            self.notify.notify().expect("capture edge notify failed");
        }
        dropped
    }

    /// Downmix one interleaved signed-16 callback block directly into the
    /// capture ring. Admission is block-atomic: the return value is either zero
    /// or the complete frame count, so a realtime callback never publishes a
    /// prefix/suffix splice and never allocates an intermediate PCM vector.
    fn push_interleaved_i16(&mut self, samples: &[i16], channels: usize) -> usize {
        if samples.is_empty() {
            return 0;
        }
        if channels == 0 {
            return samples.len();
        }
        if samples.len() % channels != 0 {
            return samples.len().div_ceil(channels);
        }
        let frames = samples.len() / channels;
        let dropped = self.capture.write_interleaved_i16(samples, channels);
        self.publish(dropped, frames)
    }

    fn push_interleaved_f32(&mut self, samples: &[f32], channels: usize) -> usize {
        // A live device always reports at least one channel; zero is a corrupt
        // config, not a case to silently report as zero frames.
        let frames = samples.len() / channels;
        let dropped = self.capture.write_interleaved_f32(samples, channels);
        self.publish(dropped, frames)
    }

    fn push_interleaved_u16(&mut self, samples: &[u16], channels: usize) -> usize {
        let frames = samples.len() / channels;
        let dropped = self.capture.write_interleaved_u16(samples, channels);
        self.publish(dropped, frames)
    }
}

/// Bounded callback-to-continuation dock. The callback owns only the block
/// selected by `active`; the retained VAD continuation seals that block before
/// changing the index. The sample clock advances for every valid hardware
/// callback, including a block dropped under backpressure, so acoustic policy
/// is driven by device samples rather than a timer.
struct CaptureIngress {
    blocks: [Arc<CaptureReservation>; CAPTURE_BLOCKS],
    active: AtomicUsize,
    clock: Arc<AtomicUsize>,
    stats: Stats,
}

impl CaptureIngress {
    fn new(
        blocks: [Arc<CaptureReservation>; CAPTURE_BLOCKS],
        clock: Arc<AtomicUsize>,
        stats: Stats,
    ) -> Capture {
        Arc::new(Self {
            blocks,
            active: AtomicUsize::new(0),
            clock,
            stats,
        })
    }

    fn write_interleaved_f32(&self, samples: &[f32], channels: usize) -> usize {
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        self.clock.fetch_add(frames, Ordering::Release);
        let Some(block) = self.blocks.get(self.active.load(Ordering::Acquire)) else {
            self.record_drop(frames);
            return frames;
        };
        let dropped = block.write_interleaved_f32(samples, channels);
        self.record_drop(dropped);
        dropped
    }

    fn write_interleaved_i16(&self, samples: &[i16], channels: usize) -> usize {
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        self.clock.fetch_add(frames, Ordering::Release);
        let Some(block) = self.blocks.get(self.active.load(Ordering::Acquire)) else {
            self.record_drop(frames);
            return frames;
        };
        let dropped = block.write_interleaved_i16(samples, channels);
        self.record_drop(dropped);
        dropped
    }

    fn write_interleaved_u16(&self, samples: &[u16], channels: usize) -> usize {
        if channels == 0 || samples.len() % channels != 0 {
            return samples.len().div_ceil(channels.max(1));
        }
        let frames = samples.len() / channels;
        self.clock.fetch_add(frames, Ordering::Release);
        let Some(block) = self.blocks.get(self.active.load(Ordering::Acquire)) else {
            self.record_drop(frames);
            return frames;
        };
        let dropped = block.write_interleaved_u16(samples, channels);
        self.record_drop(dropped);
        dropped
    }

    fn record_drop(&self, dropped: usize) {
        if dropped != 0 {
            self.stats
                .dropped_samples
                .fetch_add(dropped as u64, Ordering::Relaxed);
        }
    }

    fn deactivate(&self) {
        self.active.store(CAPTURE_BLOCKS, Ordering::Release);
    }

    fn activate(&self, index: usize) {
        debug_assert!(index < CAPTURE_BLOCKS);
        self.active.store(index, Ordering::Release);
    }

    fn next_open(&self, exclude: usize) -> Option<usize> {
        (0..CAPTURE_BLOCKS).find(|index| *index != exclude && self.blocks[*index].is_open())
    }

    fn rearm(&self) -> Result<(), String> {
        for block in &self.blocks {
            let _ = block.try_rearm()?;
        }
        Ok(())
    }
}

struct PlaybackReference {
    active: AtomicBool,
    rms_bits: AtomicU32,
    capture_clock: Arc<AtomicUsize>,
    tail_frames: usize,
    /// Capture cursor at the first playback-idle edge. `usize::MAX` means no tail.
    idle_at_frame: AtomicUsize,
}

impl PlaybackReference {
    fn new(capture_clock: Arc<AtomicUsize>, capture_rate: u32) -> Playback {
        Arc::new(Self {
            active: AtomicBool::new(false),
            rms_bits: AtomicU32::new(0.0f32.to_bits()),
            capture_clock,
            tail_frames: ((capture_rate as u64 * PLAYBACK_ECHO_TAIL_MS) / 1_000) as usize,
            idle_at_frame: AtomicUsize::new(usize::MAX),
        })
    }

    fn set_playing(&self, rms: f32) {
        self.rms_bits
            .store(rms.max(0.0).to_bits(), Ordering::Release);
        self.active.store(true, Ordering::Release);
        self.idle_at_frame.store(usize::MAX, Ordering::Release);
    }

    fn set_idle(&self) {
        // Only the first idle after playing starts the tail cursor; repeated
        // set_idle calls (TurnComplete then Error, etc.) must not extend it.
        if self.active.swap(false, Ordering::AcqRel) {
            self.idle_at_frame.store(
                self.capture_clock.load(Ordering::Acquire),
                Ordering::Release,
            );
        }
        self.rms_bits.store(0.0f32.to_bits(), Ordering::Release);
    }

    fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// True while playing OR within the post-playback echo tail — the speaker
    /// drain plus room decay window in which mic energy is presumed to be our own
    /// voice coming back. The turn is not over until the room is quiet.
    fn active_or_tail(&self) -> bool {
        if self.active() {
            return true;
        }
        let idle_at = self.idle_at_frame.load(Ordering::Acquire);
        idle_at != usize::MAX
            && self
                .capture_clock
                .load(Ordering::Acquire)
                .wrapping_sub(idle_at)
                < self.tail_frames
    }

    fn rms(&self) -> f32 {
        f32::from_bits(self.rms_bits.load(Ordering::Acquire))
    }
}

/// Sentinel for "no voiced input observed yet" in [`TurnLatency`].
const TURN_LATENCY_NO_VOICE: u64 = u64::MAX;
/// Minimum output-chunk RMS that counts as the assistant audibly speaking.
/// Filters silence frames so the measurement matches voice onset, not stream onset.
const TURN_LATENCY_AGENT_RMS: f32 = 0.01;

/// Turn-responsiveness telemetry (spec 09, W1): the gap between the user's
/// last voiced mic input and the first audible assistant PCM per turn — the
/// same measurement the Sesame demo client grades on 300/500/1000/3000ms bands.
struct TurnLatency {
    origin: Instant,
    /// Milliseconds since `origin` of the most recent voiced mic window.
    last_voice_ms: std::sync::atomic::AtomicU64,
    /// Armed while a reply is pending; the first audible chunk measures and disarms.
    awaiting: AtomicBool,
    first_word_logged: AtomicBool,
}

impl TurnLatency {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            origin: Instant::now(),
            last_voice_ms: std::sync::atomic::AtomicU64::new(TURN_LATENCY_NO_VOICE),
            awaiting: AtomicBool::new(false),
            first_word_logged: AtomicBool::new(false),
        })
    }

    fn now_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    fn mark_voice(&self) {
        self.last_voice_ms.store(self.now_ms(), Ordering::Release);
    }

    fn arm(&self) {
        if self.last_voice_ms.load(Ordering::Acquire) != TURN_LATENCY_NO_VOICE {
            self.awaiting.store(true, Ordering::Release);
        }
    }

    fn disarm(&self) {
        self.awaiting.store(false, Ordering::Release);
    }

    /// If armed, measure last-voice → now and disarm. Returns the gap in ms.
    fn try_measure(&self) -> Option<u64> {
        if !self.awaiting.swap(false, Ordering::AcqRel) {
            return None;
        }
        let last = self.last_voice_ms.load(Ordering::Acquire);
        if last == TURN_LATENCY_NO_VOICE {
            return None;
        }
        Some(self.now_ms().saturating_sub(last))
    }

    /// Session-start → first audible assistant word, logged once.
    fn first_word(&self) -> Option<u64> {
        if self.first_word_logged.swap(true, Ordering::AcqRel) {
            return None;
        }
        Some(self.now_ms())
    }
}

/// Sesame's agent-response-latency rating bands (recovered demo client,
/// `getAgentResponseLatencyRating`): <300ms = 5 … ≥3000ms = 1.
#[cfg(test)]
fn turn_latency_rating(ms: u64) -> u8 {
    match ms {
        0..=299 => 5,
        300..=499 => 4,
        500..=999 => 3,
        1000..=2999 => 2,
        _ => 1,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioStatsSnapshot {
    pub decoded_samples: u64,
    pub queued_samples: u64,
    pub dropped_samples: u64,
    pub played_samples: u64,
    pub underrun_frames: u64,
    /// Completed pause→first-audio measurements this session (spec 09, W1).
    pub turn_count: u64,
    /// Most recent pause→first-audio gap in milliseconds.
    pub last_turn_latency_ms: u64,
    /// Mean pause→first-audio gap in milliseconds across the session.
    pub mean_turn_latency_ms: u64,
}

#[derive(Default)]
struct AudioStats {
    decoded_samples: std::sync::atomic::AtomicU64,
    queued_samples: std::sync::atomic::AtomicU64,
    dropped_samples: std::sync::atomic::AtomicU64,
    played_samples: std::sync::atomic::AtomicU64,
    underrun_frames: std::sync::atomic::AtomicU64,
    turn_count: std::sync::atomic::AtomicU64,
    last_turn_latency_ms: std::sync::atomic::AtomicU64,
    total_turn_latency_ms: std::sync::atomic::AtomicU64,
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
    fn record_turn_latency(&self, ms: u64) {
        self.turn_count.fetch_add(1, Ordering::Relaxed);
        self.last_turn_latency_ms.store(ms, Ordering::Relaxed);
        self.total_turn_latency_ms.fetch_add(ms, Ordering::Relaxed);
    }

    fn snapshot(&self) -> AudioStatsSnapshot {
        let turn_count = self.turn_count.load(Ordering::Relaxed);
        let total_turn_latency_ms = self.total_turn_latency_ms.load(Ordering::Relaxed);
        AudioStatsSnapshot {
            decoded_samples: self.decoded_samples.load(Ordering::Relaxed),
            queued_samples: self.queued_samples.load(Ordering::Relaxed),
            dropped_samples: self.dropped_samples.load(Ordering::Relaxed),
            played_samples: self.played_samples.load(Ordering::Relaxed),
            underrun_frames: self.underrun_frames.load(Ordering::Relaxed),
            turn_count,
            last_turn_latency_ms: self.last_turn_latency_ms.load(Ordering::Relaxed),
            mean_turn_latency_ms: total_turn_latency_ms
                .checked_div(turn_count)
                .unwrap_or_default(),
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

/// VAD and capture-loop knobs.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub vad_threshold: f32,
    pub silence_ms: u64,
    pub min_utterance_s: f32,
    pub can_interrupt: bool,
    pub trace: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.012,
            // End-of-turn silence (spec 09, W3): 800ms burned most of the response
            // budget before any model work. Sesame's demo grades <300ms pause→first-
            // word as excellent. 350ms proved too eager in live testing — it committed
            // echo blips as turns and split mid-sentence pauses — so 500ms until the
            // AEC verification (W6) lands.
            silence_ms: 500,
            min_utterance_s: 0.3,
            can_interrupt: false,
            trace: false,
        }
    }
}

/// Handle to a running voice service.
pub struct VoiceRuntime {
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
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
    /// audio devices.
    pub fn prepare(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<VoiceRuntime, String> {
        VOICE_TRACE_ENABLED.store(cfg.trace, Ordering::Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let playback_flush = Arc::new(AtomicBool::new(false));
        let audio = Arc::new(AudioStats::default());
        let done = Arc::new(AtomicBool::new(false));
        let in_rate = default_input_rate().map_err(|error| format!("audio input: {error}"))?;
        let out_rate = default_output_rate().map_err(|error| format!("audio output: {error}"))?;
        let mut sink: EventSink = Box::new(sink);
        if !emit_or_stop(&mut sink, &stop, RuntimeEvent::State(SessionState::Loading)) {
            return Err("voice event sink rejected the loading state".into());
        }
        let clock = Arc::new(AtomicUsize::new(0));
        let mut engine = build_engine(out_rate).map_err(|error| format!("model load: {error}"))?;
        let mut dock = engine
            .take_capture_dock()?
            .ok_or("callback-driven native engine did not expose capture leases")?;
        let frames = (in_rate as usize * MAX_UTTERANCE_SECONDS).max(1);
        let first = dock
            .reserve(frames, in_rate)?
            .ok_or("native capture dock has no first setup reservation")?;
        let second = dock
            .reserve(frames, in_rate)?
            .ok_or("native capture dock has no second setup reservation")?;
        let capture = CaptureIngress::new([first, second], clock, audio.clone());
        let playback = PlaybackReference::new(capture.clock.clone(), in_rate);
        let latency = TurnLatency::new();
        let handoff = EngineHandoff::new();
        let runtime = Arc::new(
            CoroutineRuntime::with_config(CoroutineConfig {
                workers: 1,
                ..CoroutineConfig::default()
            })
            .map_err(|status| format!("create voice kcoro runtime failed: {status}"))?,
        );
        let (service, control) = runtime
            .owner_state_service_factory(|setup| {
                let capture_edge = setup.realtime_notifier()?;
                let events = setup.realtime_notifier()?;
                let playback_events = setup.realtime_notifier()?;
                let control = setup.shared_realtime_notifier();
                let source = engine
                    .take_playback_source(playback_events)
                    .map_err(|_| -1)?
                    .ok_or(-1)?;
                let task = SessionTask::Init(Some(SessionInit {
                    cfg,
                    engine: Some(engine),
                    sink,
                    stop: stop.clone(),
                    interrupt: interrupt.clone(),
                    mic_enabled: mic_enabled.clone(),
                    playback_flush: playback_flush.clone(),
                    audio: audio.clone(),
                    capture: Some(capture.clone()),
                    writer: Some(AudioInputWriter {
                        capture: capture.clone(),
                        notify: capture_edge,
                        producer: std::marker::PhantomData,
                    }),
                    source: Some(source),
                    in_rate,
                    out_rate,
                    playback: playback.clone(),
                    latency: latency.clone(),
                    done: done.clone(),
                    handoff: handoff.clone(),
                    events: Some(events),
                }));
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
            playback_flush: playback_flush.clone(),
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
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<Self, String> {
        Self::prepare(cfg, build_engine, sink)
    }

    /// Abort the current reply and flush queued playback.
    pub fn interrupt(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
        self.playback_flush.store(true, Ordering::SeqCst);
        // The flags are inert until this edge lands: a dropped notify is a
        // barge-in that silently never happens.
        self.control.notify().expect("interrupt notify failed");
    }

    /// Pause/resume mic capture without ending the session.
    pub fn set_mic_enabled(&self, on: bool) {
        self.mic_enabled.store(on, Ordering::SeqCst);
        self.control.notify().expect("mic-enable notify failed");
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
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        self.stop.store(true, Ordering::SeqCst);
        let _ = self.control.notify();
        if let Some(service) = self.service.take() {
            service.stop();
            let _ = service.join();
            drop(service);
        }
        if let Some(mut engine) = self.engine.take() {
            let _ = engine.stop_session();
        }
        self.runtime.stop();
        let _ = self.runtime.join();
        self.done.store(true, Ordering::SeqCst);
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        self.shutdown();
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
    cfg: RuntimeConfig,
    engine: Option<Box<dyn VoiceEngine>>,
    sink: EventSink,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    audio: Stats,
    capture: Option<Capture>,
    writer: Option<AudioInputWriter>,
    source: Option<Box<dyn PlaybackSource>>,
    in_rate: u32,
    out_rate: u32,
    playback: Playback,
    latency: Arc<TurnLatency>,
    done: Arc<AtomicBool>,
    handoff: Arc<EngineHandoff>,
    events: Option<RealtimeNotifier>,
}

impl SessionInit {
    fn retire_engine(&mut self) {
        if let Some(engine) = self.engine.take() {
            self.handoff.publish(engine);
        }
    }
}

enum SessionTask {
    Init(Option<SessionInit>),
    Turn(VadTask),
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
            SessionTask::Init(_) | SessionTask::Turn(_) | SessionTask::Done => Ok(None),
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
        let outcome = self.task.advance();
        if outcome == ServiceOutcome::Complete {
            self.retire();
        }
        outcome
    }

    fn retire(&mut self) {
        if self.retired {
            return;
        }
        // Dropping CPAL streams closes the producer callbacks. Only then may
        // capture/playback leases and the native engine cross the terminal
        // handoff, preserving ticket and epoch ownership through the device.
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
            Self::Turn(mut task) => {
                let outcome = task.step();
                *self = Self::Turn(task);
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
                init.retire_engine();
            }
            Self::Turn(mut task) => {
                task.done.store(true, Ordering::Release);
                task.finish();
            }
            Self::Init(None) | Self::Done => {}
        }
    }
}

impl SessionInit {
    fn mount(mut self) -> Result<SessionTask, (Self, String)> {
        let Some(mut engine) = self.engine.take() else {
            return Err((self, "voice engine was already consumed".into()));
        };
        let Some(capture) = self.capture.take() else {
            return Err((
                self,
                "native capture reservations were already consumed".into(),
            ));
        };
        let assistant = Arc::new(AtomicBool::new(false));

        let Some(events) = self.events.take() else {
            self.engine = Some(engine);
            return Err((self, "voice event notifier was already consumed".into()));
        };
        let native = match engine.mount_events(events) {
            Ok(native) => native,
            Err(error) => {
                self.engine = Some(engine);
                return Err((self, format!("mount native event edge: {error}")));
            }
        };
        if !native {
            self.engine = Some(engine);
            return Err((
                self,
                "voice engine does not implement the native callback continuation contract".into(),
            ));
        }
        if !emit_or_stop(
            &mut self.sink,
            &self.stop,
            RuntimeEvent::State(SessionState::Listening),
        ) {
            return Err((self, "voice event sink rejected listening state".into()));
        }
        Ok(SessionTask::Turn(VadTask::new(
            self,
            capture,
            TurnDriver {
                engine,
                events: EventState::default(),
            },
            assistant,
        )))
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
        assistant: &AtomicBool,
        mic_enabled: &AtomicBool,
        latency: &TurnLatency,
        playback: &PlaybackReference,
    ) {
        if self.failed {
            return;
        }
        let ok = match event {
            VoiceEvent::Text(text) => {
                assistant.store(true, Ordering::Release);
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
                assistant.store(false, Ordering::Release);
                latency.disarm();
                playback.set_idle();
                self.transcript.clear();
                self.speaking = false;
                emit_ready(sink, stop, mic_enabled)
            }
            VoiceEvent::Error(error) => {
                assistant.store(false, Ordering::Release);
                latency.disarm();
                playback.set_idle();
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
    fn interrupt(&mut self) {
        let _ = self.engine.interrupt_stream();
    }

    fn begin(&mut self, ticket: crate::CaptureTicket) -> Result<bool, String> {
        self.engine.begin_capture(ticket)
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_events(
        &mut self,
        stop: &Arc<AtomicBool>,
        sink: &mut EventSink,
        assistant: &AtomicBool,
        mic_enabled: &AtomicBool,
        latency: &TurnLatency,
        playback: &PlaybackReference,
    ) -> Result<EngineProgress, String> {
        let progress = self.engine.advance_events(stop, &mut |event| {
            self.events
                .emit(event, sink, stop, assistant, mic_enabled, latency, playback);
        })?;
        if self.events.failed {
            return Ok(EngineProgress::Complete);
        }
        Ok(progress)
    }
}

struct VadTask {
    cfg: RuntimeConfig,
    sink: EventSink,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    capture: Option<Capture>,
    block: Option<usize>,
    in_rate: u32,
    done: Arc<AtomicBool>,
    handoff: Arc<EngineHandoff>,
    playback: Playback,
    driver: Option<TurnDriver>,
    pending: Option<crate::CaptureTicket>,
    assistant: Arc<AtomicBool>,
    latency: Arc<TurnLatency>,
    window: usize,
    max_local: usize,
    silence_frames: usize,
    speaking: bool,
    start: usize,
    read: usize,
    voice_end: usize,
    voiced_streak: usize,
    streak_start: usize,
}

impl VadTask {
    fn new(
        init: SessionInit,
        capture: Capture,
        driver: TurnDriver,
        assistant: Arc<AtomicBool>,
    ) -> Self {
        let window = (init.in_rate as usize / 5).max(1);
        let max_local = (init.in_rate as usize * MAX_UTTERANCE_SECONDS).max(window);
        let silence_frames = ((init.in_rate as u64 * init.cfg.silence_ms) / 1_000) as usize;
        Self {
            cfg: init.cfg,
            sink: init.sink,
            stop: init.stop,
            interrupt: init.interrupt,
            mic_enabled: init.mic_enabled,
            playback_flush: init.playback_flush,
            capture: Some(capture),
            block: Some(0),
            in_rate: init.in_rate,
            done: init.done,
            handoff: init.handoff,
            playback: init.playback,
            driver: Some(driver),
            pending: None,
            assistant,
            latency: init.latency,
            window,
            max_local,
            silence_frames,
            speaking: false,
            start: 0,
            read: 0,
            voice_end: 0,
            voiced_streak: 0,
            streak_start: 0,
        }
    }

    fn outcome(&self) -> ServiceOutcome {
        if self.stop.load(Ordering::Acquire) {
            return ServiceOutcome::Complete;
        }
        if self.interrupt.load(Ordering::Acquire) {
            return ServiceOutcome::Continue;
        }
        let Some(capture) = self.capture.as_ref() else {
            return ServiceOutcome::Complete;
        };
        if self
            .block
            .and_then(|index| capture.blocks.get(index))
            .is_some_and(|block| self.read + self.window <= block.len())
        {
            return ServiceOutcome::Continue;
        }
        ServiceOutcome::Dormant
    }

    fn reset_capture_state(&mut self) {
        self.read = 0;
        self.start = 0;
        self.voice_end = 0;
        self.speaking = false;
        self.voiced_streak = 0;
        self.streak_start = 0;
    }

    fn prepare_capture(&mut self) -> Result<(), String> {
        let Some(capture) = self.capture.as_ref() else {
            return Err("native capture ingress was retired".into());
        };
        capture.rearm()?;
        if self.block.is_some() {
            return Ok(());
        }
        let Some(index) = (0..CAPTURE_BLOCKS).find(|index| capture.blocks[*index].is_open()) else {
            capture.deactivate();
            return Ok(());
        };
        capture.activate(index);
        self.block = Some(index);
        self.reset_capture_state();
        Ok(())
    }

    /// Retire the current capture contents without changing native storage.
    /// `false` means a hardware callback owns WRITING; that callback publishes
    /// its tail and rings the service edge, so this continuation simply yields.
    fn discard_capture(&mut self) -> bool {
        let Some(capture) = self.capture.as_ref() else {
            return true;
        };
        let Some(index) = self.block else {
            return true;
        };
        let block = &capture.blocks[index];
        if !block.try_seal() {
            return false;
        }
        capture.deactivate();
        block.discard_sealed();
        capture.activate(index);
        self.reset_capture_state();
        true
    }

    fn publish_capture(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<Option<crate::CaptureTicket>, String> {
        let capture = self
            .capture
            .as_ref()
            .ok_or("native capture ingress was retired")?;
        let index = self.block.ok_or("native capture block is not armed")?;
        let block = Arc::clone(&capture.blocks[index]);
        if !block.try_seal() {
            return Ok(None);
        }

        // Switch producer ownership before publishing the sealed view. A
        // callback racing this transition either finishes the old block before
        // sealing or observes the next block; it can never write a published
        // generation.
        let next = capture.next_open(index);
        capture.deactivate();
        if let Some(next) = next {
            capture.activate(next);
        }
        let ticket = block.publish(start, end)?;
        self.block = next;
        self.reset_capture_state();
        Ok(ticket)
    }

    fn step(&mut self) -> ServiceOutcome {
        if self.stop.load(Ordering::Acquire) {
            return ServiceOutcome::Complete;
        }
        if self.driver.is_none() {
            return ServiceOutcome::Complete;
        }
        let playback = self.playback.clone();
        if let Err(error) = self.prepare_capture() {
            let _ = emit_or_stop(
                &mut self.sink,
                &self.stop,
                RuntimeEvent::Error(format!("native capture rearm failed: {error}")),
            );
            return ServiceOutcome::Complete;
        }
        if let Some(ticket) = self.pending {
            match self
                .driver
                .as_mut()
                .expect("driver checked above")
                .begin(ticket)
            {
                Ok(true) => {
                    self.pending = None;
                    self.latency.arm();
                }
                Ok(false) => {}
                Err(error) => {
                    let _ = emit_or_stop(
                        &mut self.sink,
                        &self.stop,
                        RuntimeEvent::Error(format!(
                            "native pending capture admission failed: {error}"
                        )),
                    );
                    return ServiceOutcome::Complete;
                }
            }
        }
        let progress = self
            .driver
            .as_mut()
            .expect("driver checked above")
            .advance_events(
                &self.stop,
                &mut self.sink,
                &self.assistant,
                &self.mic_enabled,
                &self.latency,
                &playback,
            );
        match progress {
            Ok(EngineProgress::Continue) => return ServiceOutcome::Continue,
            Ok(EngineProgress::Dormant | EngineProgress::Complete) => {}
            Err(error) => {
                let _ = emit_or_stop(
                    &mut self.sink,
                    &self.stop,
                    RuntimeEvent::Error(format!("native event drain failed: {error}")),
                );
                return ServiceOutcome::Complete;
            }
        }
        if let Some(ticket) = self.pending {
            match self
                .driver
                .as_mut()
                .expect("driver checked above")
                .begin(ticket)
            {
                Ok(true) => {
                    self.pending = None;
                    self.latency.arm();
                }
                Ok(false) => return ServiceOutcome::Dormant,
                Err(error) => {
                    let _ = emit_or_stop(
                        &mut self.sink,
                        &self.stop,
                        RuntimeEvent::Error(format!(
                            "native pending capture admission failed: {error}"
                        )),
                    );
                    return ServiceOutcome::Complete;
                }
            }
        }

        if self.interrupt.swap(false, Ordering::SeqCst) {
            self.driver
                .as_mut()
                .expect("driver checked above")
                .interrupt();
            self.playback_flush.store(true, Ordering::SeqCst);
            self.assistant.store(false, Ordering::SeqCst);
            self.speaking = false;
            self.voiced_streak = 0;
            if !emit_ready(&mut self.sink, &self.stop, &self.mic_enabled) {
                return ServiceOutcome::Complete;
            }
        }

        if !self.mic_enabled.load(Ordering::SeqCst) {
            if !self.discard_capture() {
                return ServiceOutcome::Dormant;
            }
            return self.outcome();
        }

        if reference_audio_active(&self.assistant, &playback) && !self.cfg.can_interrupt {
            if !self.discard_capture() {
                return ServiceOutcome::Dormant;
            }
            return self.outcome();
        }

        let Some(capture) = self.capture.as_ref() else {
            return ServiceOutcome::Complete;
        };
        let Some(index) = self.block else {
            return self.outcome();
        };
        let block = Arc::clone(&capture.blocks[index]);
        let n = block.len();
        let mut windows = 0usize;
        while self.read + self.window <= n && windows < 8 {
            let threshold =
                reference_vad_threshold(self.cfg.vad_threshold, &self.assistant, &playback);
            let voiced = block
                .with_span(self.read, self.read + self.window, rms)
                .is_some_and(|level| level > threshold);
            if voiced {
                if !self.speaking {
                    if self.voiced_streak == 0 {
                        self.streak_start = self.read;
                    }
                    self.voiced_streak += 1;
                    let reference = reference_audio_active(&self.assistant, &playback);
                    let needed = if reference {
                        BARGE_IN_SUSTAIN_WINDOWS
                    } else {
                        1
                    };
                    if self.voiced_streak >= needed {
                        if reference {
                            self.driver
                                .as_mut()
                                .expect("driver checked above")
                                .interrupt();
                            self.playback_flush.store(true, Ordering::SeqCst);
                        }
                        self.speaking = true;
                        self.start = self.streak_start;
                        self.voiced_streak = 0;
                        if !emit_or_stop(
                            &mut self.sink,
                            &self.stop,
                            RuntimeEvent::State(SessionState::Listening),
                        ) {
                            return ServiceOutcome::Complete;
                        }
                    }
                }
                self.latency.mark_voice();
                self.voice_end = self.read + self.window;
            } else if !self.speaking {
                self.voiced_streak = 0;
            }
            self.read += self.window;
            windows += 1;
        }
        if self.read + self.window <= n {
            return ServiceOutcome::Continue;
        }

        let silent_frames = n.saturating_sub(self.voice_end);
        let forced_end = self.speaking && n >= self.max_local;
        if self.speaking && (silent_frames >= self.silence_frames || forced_end) {
            let end = (self.voice_end + self.window).clamp(self.start, self.read.min(n));
            let dur_s = (end - self.start) as f32 / self.in_rate as f32;
            let submission = if dur_s >= self.cfg.min_utterance_s {
                if !emit_or_stop(
                    &mut self.sink,
                    &self.stop,
                    RuntimeEvent::State(SessionState::Thinking),
                ) {
                    return ServiceOutcome::Complete;
                }
                match self.publish_capture(self.start, end) {
                    Ok(Some(ticket)) => Some((
                        ticket,
                        self.driver
                            .as_mut()
                            .expect("driver checked above")
                            .begin(ticket),
                    )),
                    Ok(None) => return ServiceOutcome::Dormant,
                    Err(error) => Some((crate::CaptureTicket::default(), Err(error))),
                }
            } else {
                if !self.discard_capture() {
                    return ServiceOutcome::Dormant;
                }
                None
            };
            if let Some((ticket, submission)) = submission {
                match submission {
                    Ok(true) => self.latency.arm(),
                    Ok(false) => self.pending = Some(ticket),
                    Err(error) => {
                        let _ = emit_or_stop(
                            &mut self.sink,
                            &self.stop,
                            RuntimeEvent::Error(format!(
                                "native capture admission failed: {error}"
                            )),
                        );
                        return ServiceOutcome::Complete;
                    }
                }
            }
        } else if !self.speaking && n > self.in_rate as usize * 5 && !self.discard_capture() {
            return ServiceOutcome::Dormant;
        }
        self.outcome()
    }

    fn finish(&mut self) {
        // OwnerSession has already destroyed both hardware streams, so no
        // callback can still hold a ticketed capture/playback endpoint while
        // this state crosses the terminal handoff.
        drop(self.capture.take());
        if let Some(driver) = self.driver.take() {
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

impl Drop for VadTask {
    fn drop(&mut self) {
        if !self.done.load(Ordering::Acquire) {
            self.finish();
        }
    }
}

fn reference_audio_active(assistant: &AtomicBool, playback: &PlaybackReference) -> bool {
    assistant.load(Ordering::SeqCst) || playback.active_or_tail()
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

fn reference_vad_threshold(base: f32, assistant: &AtomicBool, playback: &PlaybackReference) -> f32 {
    if !reference_audio_active(assistant, playback) {
        return base;
    }
    (base * PLAYBACK_VAD_MULTIPLIER).max(playback.rms() * PLAYBACK_ECHO_MULTIPLIER)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt()
}

#[cfg(feature = "audio-io")]
fn start_devices(init: &mut SessionInit) -> Result<DeviceStreams, String> {
    let writer = init
        .writer
        .take()
        .ok_or("platform capture endpoint was already consumed")?;
    let source = init
        .source
        .take()
        .ok_or("platform playback endpoint was already consumed")?;
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
        init.playback.clone(),
        init.latency.clone(),
    )
    .map_err(|error| format!("audio output: {error}"))?;
    let input =
        start_input(writer, init.in_rate).map_err(|error| format!("audio input: {error}"))?;
    Ok(DeviceStreams {
        _input: input,
        _output: output,
    })
}

#[cfg(not(feature = "audio-io"))]
fn start_devices(_init: &mut SessionInit) -> Result<DeviceStreams, String> {
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(feature = "audio-io")]
fn default_input_rate() -> Res<u32> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 {
        return Err("audio input sample rate is zero".into());
    }
    Ok(rate)
}

#[cfg(not(feature = "audio-io"))]
fn default_input_rate() -> Res<u32> {
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
fn start_input(mut writer: AudioInputWriter, expected_rate: u32) -> Res<HostStream> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 || rate != expected_rate {
        return Err("audio input sample rate changed during setup".into());
    }
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let err = |e| eprintln!("[voice] input stream error: {e}");

    macro_rules! stream {
        ($t:ty, $push:expr) => {{
            dev.build_input_stream(
                &cfg,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    $push(&mut writer, data, channels);
                },
                err,
                None,
            )
        }};
    }

    let stream = match fmt {
        cpal::SampleFormat::F32 => stream!(f32, AudioInputWriter::push_interleaved_f32),
        cpal::SampleFormat::I16 => stream!(i16, AudioInputWriter::push_interleaved_i16),
        cpal::SampleFormat::U16 => stream!(u16, AudioInputWriter::push_interleaved_u16),
        other => return Err(format!("unsupported input sample format {other:?}").into()),
    }?;
    stream.play()?;
    Ok(stream)
}

#[cfg(not(feature = "audio-io"))]
fn start_input(_writer: AudioInputWriter, _expected_rate: u32) -> Res<HostStream> {
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(feature = "audio-io")]
fn start_output(
    source: Box<dyn PlaybackSource>,
    audio: Stats,
    flush: Arc<AtomicBool>,
    playback: Playback,
    latency: Arc<TurnLatency>,
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
    let err = |e| eprintln!("[voice] output stream error: {e}");

    macro_rules! stream {
        ($t:ty, $write:ident) => {{
            let mut source = source;
            let flush = flush.clone();
            let playback = playback.clone();
            let audio = audio.clone();
            let latency = latency.clone();
            dev.build_output_stream(
                &cfg,
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let reset = flush.swap(false, Ordering::AcqRel);
                    let write = source.$write(data, channels, reset);
                    if reset {
                        playback.set_idle();
                    }
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
                        playback.set_playing(write.rms);
                        if write.rms > TURN_LATENCY_AGENT_RMS {
                            if let Some(ms) = latency.try_measure() {
                                audio.record_turn_latency(ms);
                            }
                            let _ = latency.first_word();
                        }
                    } else if !write.active {
                        playback.set_idle();
                    }
                    if write.underrun_frames != 0 {
                        audio
                            .underrun_frames
                            .fetch_add(write.underrun_frames as u64, Ordering::Relaxed);
                    }
                },
                err,
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

#[cfg(not(feature = "audio-io"))]
fn start_output(
    _source: Box<dyn PlaybackSource>,
    _audio: Stats,
    _flush: Arc<AtomicBool>,
    _playback: Playback,
    _latency: Arc<TurnLatency>,
) -> Res<HostStream> {
    Err("liquid-audio was built without platform audio support".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn playback_reference_extends_echo_gate_after_generation_finishes() {
        let assistant = AtomicBool::new(false);
        let capture = Arc::new(AtomicUsize::new(0));
        let playback = PlaybackReference::new(capture.clone(), 1_000);

        // Never played: no tail, not reference audio.
        assert!(!reference_audio_active(&assistant, &playback));
        assert_eq!(reference_vad_threshold(0.012, &assistant, &playback), 0.012);

        playback.set_playing(0.08);
        assert!(reference_audio_active(&assistant, &playback));
        assert!(reference_vad_threshold(0.012, &assistant, &playback) >= 0.1);

        // Idle starts the ECHO TAIL: the speaker is still draining and the room
        // still ringing, so the reference gate must stay up (the bug this guards:
        // the model's own trailing audio re-entering as a fresh "user" utterance).
        playback.set_idle();
        assert!(!playback.active(), "raw active flag drops on idle");
        assert!(
            reference_audio_active(&assistant, &playback),
            "echo tail keeps reference audio active right after idle"
        );

        // Acoustic-silence callbacks advance the sample cursor. Wall-clock time
        // alone is never allowed to advance the speech state machine.
        let tail = playback.tail_frames;
        capture.fetch_add(tail, Ordering::Release);
        assert!(!reference_audio_active(&assistant, &playback));

        // Playing again clears the tail bookkeeping.
        playback.set_playing(0.05);
        assert!(reference_audio_active(&assistant, &playback));
    }

    #[test]
    fn playback_reference_requires_barge_in_above_echo_floor() {
        let assistant = AtomicBool::new(false);
        let playback = PlaybackReference::new(Arc::new(AtomicUsize::new(0)), 1_000);
        playback.set_playing(0.08);

        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback),
            0.08 * PLAYBACK_ECHO_MULTIPLIER
        );
    }

    #[test]
    fn assistant_generation_is_reference_audio_even_before_playback_starts() {
        let assistant = AtomicBool::new(true);
        let playback = PlaybackReference::new(Arc::new(AtomicUsize::new(0)), 1_000);

        assert!(reference_audio_active(&assistant, &playback));
        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback),
            0.012 * PLAYBACK_VAD_MULTIPLIER
        );
    }

    #[test]
    fn turn_latency_measures_only_when_armed_after_voice() {
        let latency = TurnLatency::new();

        // Not armed, no voice: nothing to measure.
        assert_eq!(latency.try_measure(), None);

        // Arming without any voiced input is a no-op.
        latency.arm();
        assert_eq!(latency.try_measure(), None);

        latency.mark_voice();
        latency.arm();
        let measured = latency.try_measure().expect("armed after voice measures");
        assert!(measured < 1000, "same-test measurement should be near-zero");

        // Measurement disarms: a second audible chunk doesn't re-measure.
        assert_eq!(latency.try_measure(), None);
    }

    #[test]
    fn turn_latency_disarm_cancels_pending_measurement() {
        let latency = TurnLatency::new();
        latency.mark_voice();
        latency.arm();
        latency.disarm();
        assert_eq!(latency.try_measure(), None);
    }

    #[test]
    fn turn_latency_first_word_logs_once() {
        let latency = TurnLatency::new();
        assert!(latency.first_word().is_some());
        assert_eq!(latency.first_word(), None);
    }

    #[test]
    fn turn_latency_rating_matches_sesame_bands() {
        assert_eq!(turn_latency_rating(0), 5);
        assert_eq!(turn_latency_rating(299), 5);
        assert_eq!(turn_latency_rating(300), 4);
        assert_eq!(turn_latency_rating(499), 4);
        assert_eq!(turn_latency_rating(500), 3);
        assert_eq!(turn_latency_rating(999), 3);
        assert_eq!(turn_latency_rating(1000), 2);
        assert_eq!(turn_latency_rating(2999), 2);
        assert_eq!(turn_latency_rating(3000), 1);
    }

    #[test]
    fn audio_stats_snapshot_reports_mean_turn_latency() {
        let stats = AudioStats::default();
        assert_eq!(stats.snapshot().mean_turn_latency_ms, 0);

        stats.record_turn_latency(400);
        stats.record_turn_latency(800);
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.turn_count, 2);
        assert_eq!(snapshot.last_turn_latency_ms, 800);
        assert_eq!(snapshot.mean_turn_latency_ms, 600);
    }

    #[test]
    fn terminal_turn_state_follows_mic_enabled() {
        let mic = AtomicBool::new(true);
        assert_eq!(ready_state(&mic), SessionState::Listening);

        mic.store(false, Ordering::SeqCst);
        assert_eq!(ready_state(&mic), SessionState::Idle);
    }

    #[test]
    fn interrupt_flushes_queued_playback_from_runtime_handle() {
        let runtime = Arc::new(CoroutineRuntime::new().unwrap());
        let service = runtime.service(|| {}).unwrap();
        let control = service.shared_realtime_notifier();
        runtime.start().unwrap();
        service.start().unwrap();
        let live = VoiceRuntime {
            stop: Arc::new(AtomicBool::new(false)),
            interrupt: Arc::new(AtomicBool::new(false)),
            mic_enabled: Arc::new(AtomicBool::new(true)),
            playback_flush: Arc::new(AtomicBool::new(false)),
            runtime,
            service: Some(service),
            control,
            audio: Arc::new(AudioStats::default()),
            done: Arc::new(AtomicBool::new(false)),
            engine: EngineHandoff::new(),
            closed: AtomicBool::new(false),
        };

        assert!(!live.interrupt.load(Ordering::SeqCst));
        assert!(!live.playback_flush.load(Ordering::SeqCst));

        live.interrupt();

        assert!(live.interrupt.load(Ordering::SeqCst));
        assert!(live.playback_flush.load(Ordering::SeqCst));
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
