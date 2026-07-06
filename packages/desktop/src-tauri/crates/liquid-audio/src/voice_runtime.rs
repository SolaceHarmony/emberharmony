//! In-process, thread-managed voice service.
//!
//! This promotes the full-duplex example loop into a reusable runtime: microphone
//! input and speaker output can be fed by the Tauri WebRTC device path, while the
//! standalone examples still use the CPAL fallback. Energy VAD, realtime model
//! inference, playback buffering, barge-in, and mic gating all run on Rust threads
//! in this process. The Tauri layer owns one of these handles and maps
//! [`RuntimeEvent`]s onto its IPC channel.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(feature = "audio-io")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};
use serde::{Deserialize, Serialize};

use crate::{
    FrameSubmitError, RealtimeFramePipeline, RealtimePipeline, Utterance, VoiceEngine, VoiceEvent,
};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Ring = Arc<PcmRing>;
type Mic = Arc<PcmRing>;
type Playback = Arc<PlaybackReference>;
type Stats = Arc<AudioStats>;
pub type RuntimeMain = Box<dyn FnOnce() + Send + 'static>;
#[cfg(feature = "audio-io")]
type HostStream = cpal::Stream;
#[cfg(not(feature = "audio-io"))]
struct HostStream;

const MIC_RING_SECONDS: usize = 6;
const SPEAKER_RING_SECONDS: usize = 4;
const MAX_UTTERANCE_SECONDS: usize = 30;
#[cfg(feature = "audio-io")]
const SPEAKER_PREBUFFER_MS: usize = 30;
const PLAYBACK_VAD_MULTIPLIER: f32 = 3.0;
const PLAYBACK_ECHO_MULTIPLIER: f32 = 2.5;
/// Barge-in sustain gate (spec 09, W5): while the assistant is audible, this many
/// CONSECUTIVE voiced VAD windows (200ms each at the in_rate/5 window size) are
/// required before an interrupt fires. One loud window is how echo blips and coughs
/// stop the assistant; 400ms of sustained speech is how a human interjects. When the
/// assistant is quiet the first voiced window still engages immediately.
const BARGE_IN_SUSTAIN_WINDOWS: usize = 2;
/// Echo-tail hangover: `playback.set_idle()` fires on TurnComplete, but the speaker
/// is still draining and the room still ringing for a beat after — in that window the
/// model's own voice was walking back in as a fresh "user" utterance (observed as
/// 17–125ms pause→first-audio measurements and self-repeating turns). The turn is not
/// over until the room is quiet: reference-audio gating extends this long past idle.
/// Matches the LiveKit agent path's LIVEKIT_AGENT_ECHO_GATE_MS.
const PLAYBACK_ECHO_TAIL_MS: u64 = 700;

/// `LIQUID_VOICE_TRACE=1` — trace the voice call graph (VAD decisions, submits,
/// pipeline turns, engine context/cursor movements, playback transitions) to stderr
/// with timestamps relative to session start. Zero cost when off.
pub(crate) fn voice_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LIQUID_VOICE_TRACE").is_ok_and(|v| !v.is_empty() && v != "0")
    })
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

/// Externally-fed microphone input for the desktop runtime.
///
/// Tauri uses this to feed WebRTC/PlatformAudio microphone frames into the same native
/// speech pipeline without making `liquid-audio` depend on LiveKit/libwebrtc.
pub struct ExternalAudioInput {
    mic: Mic,
    rate: u32,
}

#[derive(Clone)]
pub struct ExternalAudioOutput {
    rate: u32,
    write: Arc<dyn Fn(&[f32], u32) -> Result<(), String> + Send + Sync>,
    clear: Arc<dyn Fn() + Send + Sync>,
}

#[derive(Clone)]
pub struct ExternalAudioInputWriter {
    mic: Mic,
}

impl ExternalAudioInput {
    pub fn new(rate: u32) -> Result<(Self, ExternalAudioInputWriter), String> {
        if rate == 0 {
            return Err("external audio input sample rate is zero".into());
        }
        let mic = PcmRing::new(rate as usize * MIC_RING_SECONDS);
        Ok((
            Self {
                mic: mic.clone(),
                rate,
            },
            ExternalAudioInputWriter { mic },
        ))
    }
}

impl ExternalAudioInputWriter {
    /// Push mono f32 PCM into the capture ring. Returns the number of dropped samples.
    pub fn push_mono_f32(&self, samples: &[f32]) -> usize {
        self.mic.push_slice(samples)
    }

    pub fn clear(&self) {
        self.mic.clear();
        self.mic.notify();
    }
}

impl ExternalAudioOutput {
    pub fn new(
        rate: u32,
        write: impl Fn(&[f32], u32) -> Result<(), String> + Send + Sync + 'static,
        clear: impl Fn() + Send + Sync + 'static,
    ) -> Result<Self, String> {
        if rate == 0 {
            return Err("external audio output sample rate is zero".into());
        }
        Ok(Self {
            rate,
            write: Arc::new(write),
            clear: Arc::new(clear),
        })
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn write_mono_f32(&self, samples: &[f32]) -> Result<(), String> {
        (self.write)(samples, self.rate)
    }

    pub fn clear(&self) {
        (self.clear)();
    }
}

struct PcmRing {
    buf: Box<[UnsafeCell<f32>]>,
    cap: usize,
    read: AtomicUsize,
    write: AtomicUsize,
    wake_tx: Sender<()>,
    wake_rx: Receiver<()>,
}

// The runtime uses each ring as single-producer/single-consumer: mic input -> VAD,
// model event consumer -> speaker output. `clear` stays on the consumer side: VAD clears
// the mic ring, and the output callback consumes playback flush requests.
unsafe impl Send for PcmRing {}
unsafe impl Sync for PcmRing {}

impl PcmRing {
    fn new(cap: usize) -> Ring {
        let cap = cap.max(1);
        let buf = (0..cap)
            .map(|_| UnsafeCell::new(0.0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let (wake_tx, wake_rx) = bounded(1);
        Arc::new(Self {
            buf,
            cap,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
            wake_tx,
            wake_rx,
        })
    }

    fn len(&self) -> usize {
        let read = self.read.load(Ordering::Acquire);
        let write = self.write.load(Ordering::Acquire);
        write.saturating_sub(read).min(self.cap)
    }

    fn push(&self, sample: f32) -> bool {
        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.saturating_sub(read) >= self.cap {
            return false;
        }
        unsafe {
            *self.buf[write % self.cap].get() = sample;
        }
        self.write.store(write.wrapping_add(1), Ordering::Release);
        true
    }

    fn push_slice(&self, samples: &[f32]) -> usize {
        let mut dropped = 0usize;
        for &sample in samples {
            if !self.push(sample) {
                dropped += 1;
            }
        }
        if !samples.is_empty() {
            self.notify();
        }
        dropped
    }

    fn notify(&self) {
        let _ = self.wake_tx.try_send(());
    }

    fn wait_for_input(&self, timeout: Duration) -> bool {
        if self.len() > 0 {
            return true;
        }
        match self.wake_rx.recv_timeout(timeout) {
            Ok(()) => true,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => false,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => self.len() > 0,
        }
    }

    fn pop(&self) -> Option<f32> {
        let read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        if read == write {
            return None;
        }
        let sample = unsafe { *self.buf[read % self.cap].get() };
        self.read.store(read.wrapping_add(1), Ordering::Release);
        Some(sample)
    }

    fn drain_into(&self, out: &mut Vec<f32>, limit: usize) {
        while out.len() < limit {
            let Some(sample) = self.pop() else {
                break;
            };
            out.push(sample);
        }
    }

    /// Drain all available samples into a new Vec. Non-blocking; returns
    /// empty if the ring has no data. Used by the output thread to pull
    /// PCM chunks without blocking on the ring.
    fn drain_all(&self) -> Vec<f32> {
        let len = self.len();
        if len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(len);
        while let Some(sample) = self.pop() {
            out.push(sample);
        }
        out
    }

    fn clear(&self) {
        let write = self.write.load(Ordering::Acquire);
        self.read.store(write, Ordering::Release);
    }
}

struct PlaybackReference {
    active: AtomicBool,
    rms_bits: AtomicU32,
    /// Milliseconds (since `origin`) of the most recent `set_idle` — the start of
    /// the echo tail. `u64::MAX` = never played, no tail.
    idle_at_ms: std::sync::atomic::AtomicU64,
    origin: Instant,
}

impl PlaybackReference {
    fn new() -> Playback {
        Arc::new(Self {
            active: AtomicBool::new(false),
            rms_bits: AtomicU32::new(0.0f32.to_bits()),
            idle_at_ms: std::sync::atomic::AtomicU64::new(u64::MAX),
            origin: Instant::now(),
        })
    }

    fn now_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    fn set_playing(&self, rms: f32) {
        self.rms_bits
            .store(rms.max(0.0).to_bits(), Ordering::Release);
        self.active.store(true, Ordering::Release);
        self.idle_at_ms.store(u64::MAX, Ordering::Release);
    }

    fn set_idle(&self) {
        // Only the first idle after playing starts the tail clock; repeated
        // set_idle calls (TurnComplete then Error, etc.) must not extend it.
        if self.active.swap(false, Ordering::AcqRel) {
            self.idle_at_ms.store(self.now_ms(), Ordering::Release);
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
        let idle_at = self.idle_at_ms.load(Ordering::Acquire);
        idle_at != u64::MAX && self.now_ms().saturating_sub(idle_at) < PLAYBACK_ECHO_TAIL_MS
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
    /// Decoded assistant PCM kept inside the native desktop kernel. UI code should consume
    /// `Level`; native transports such as LiveKit can consume this directly.
    Audio {
        pcm: Vec<f32>,
        rate: u32,
    },
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
        }
    }
}

/// Handle to a running voice service.
pub struct VoiceRuntime {
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    control: Option<Mic>,
    output: Option<ExternalAudioOutput>,
    audio: Stats,
    done: Arc<AtomicBool>,
    session: Option<JoinHandle<()>>,
}

impl VoiceRuntime {
    /// Build the runtime handle plus the service-thread body without spawning it. The Tauri
    /// desktop kernel uses this to make its ThreadManager own the top-level LFM2 service thread;
    /// [`Self::start`] remains a fallible standalone convenience path for examples.
    pub fn prepare(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> (VoiceRuntime, RuntimeMain) {
        Self::prepare_with_input(cfg, None, build_engine, sink)
    }

    pub fn prepare_with_input(
        cfg: RuntimeConfig,
        input: Option<ExternalAudioInput>,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> (VoiceRuntime, RuntimeMain) {
        Self::prepare_with_io(cfg, input, None, build_engine, sink)
    }

    pub fn prepare_with_io(
        cfg: RuntimeConfig,
        input: Option<ExternalAudioInput>,
        output: Option<ExternalAudioOutput>,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> (VoiceRuntime, RuntimeMain) {
        let stop = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let playback_flush = Arc::new(AtomicBool::new(false));
        let audio = Arc::new(AudioStats::default());
        let done = Arc::new(AtomicBool::new(false));
        let control = input.as_ref().map(|input| input.mic.clone());
        let live_output = output.clone();
        let live = VoiceRuntime {
            stop: stop.clone(),
            interrupt: interrupt.clone(),
            mic_enabled: mic_enabled.clone(),
            playback_flush: playback_flush.clone(),
            control,
            output: live_output,
            audio: audio.clone(),
            done: done.clone(),
            session: None,
        };
        let main: RuntimeMain = Box::new(move || {
            session_loop(
                cfg,
                build_engine,
                sink,
                stop,
                interrupt,
                mic_enabled,
                playback_flush,
                audio,
                input,
                output,
            );
            done.store(true, Ordering::SeqCst);
        });
        (live, main)
    }

    /// Spawn the session thread and return immediately.
    pub fn start(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<Self, String> {
        Self::start_with_input(cfg, None, build_engine, sink)
    }

    pub fn start_with_input(
        cfg: RuntimeConfig,
        input: Option<ExternalAudioInput>,
        build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<Self, String> {
        let (mut live, main) = Self::prepare_with_io(cfg, input, None, build_engine, sink);
        let session = std::thread::Builder::new()
            .name("voice-session".into())
            .spawn(main)
            .map_err(|e| format!("spawn voice-session thread failed: {e}"))?;
        live.session = Some(session);
        Ok(live)
    }

    /// Abort the current reply and flush queued playback.
    pub fn interrupt(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
        self.playback_flush.store(true, Ordering::SeqCst);
        if let Some(output) = self.output.as_ref() {
            output.clear();
        }
        self.wake_control();
    }

    /// Pause/resume mic capture without ending the session.
    pub fn set_mic_enabled(&self, on: bool) {
        self.mic_enabled.store(on, Ordering::SeqCst);
        self.wake_control();
    }

    /// Whether mic capture is currently allowed.
    pub fn mic_enabled(&self) -> bool {
        self.mic_enabled.load(Ordering::SeqCst)
    }

    pub fn audio_stats(&self) -> AudioStatsSnapshot {
        self.audio.snapshot()
    }

    /// Whether the session thread has exited.
    pub fn is_finished(&self) -> bool {
        self.done.load(Ordering::SeqCst)
            || self.session.as_ref().is_some_and(JoinHandle::is_finished)
    }

    /// Signal stop and join the session thread.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(output) = self.output.as_ref() {
            output.clear();
        }
        self.wake_control();
        if let Some(session) = self.session.take() {
            let _ = session.join();
            self.done.store(true, Ordering::SeqCst);
        }
    }

    fn wake_control(&self) {
        if let Some(control) = self.control.as_ref() {
            control.notify();
        }
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(output) = self.output.as_ref() {
            output.clear();
        }
        self.wake_control();
        if let Some(session) = self.session.take() {
            let _ = session.join();
            self.done.store(true, Ordering::SeqCst);
        }
    }
}

fn emit<S: FnMut(RuntimeEvent) -> bool>(sink: &Mutex<S>, event: RuntimeEvent) -> bool {
    sink.lock().map(|mut sink| (*sink)(event)).unwrap_or(false)
}

fn emit_or_stop<S: FnMut(RuntimeEvent) -> bool>(
    sink: &Mutex<S>,
    stop: &Arc<AtomicBool>,
    event: RuntimeEvent,
) -> bool {
    if emit(sink, event) {
        return true;
    }
    stop.store(true, Ordering::SeqCst);
    false
}

fn session_loop<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    cfg: RuntimeConfig,
    build_engine: impl FnOnce(u32) -> Result<Box<dyn VoiceEngine>, String>,
    sink: S,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    audio: Stats,
    input: Option<ExternalAudioInput>,
    output: Option<ExternalAudioOutput>,
) {
    let sink = Arc::new(Mutex::new(sink));

    if !emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Loading)) {
        return;
    }
    struct OutputGuard {
        _stream: Option<HostStream>,
    }
    let (out_guard, ring, playback, out_rate, external_output) = match output {
        Some(output) => (
            OutputGuard { _stream: None },
            PcmRing::new(output.rate() as usize * SPEAKER_RING_SECONDS),
            PlaybackReference::new(),
            output.rate(),
            Some(output),
        ),
        None => match start_output(audio.clone(), playback_flush.clone()) {
            Ok((stream, ring, playback, rate)) => (
                OutputGuard {
                    _stream: Some(stream),
                },
                ring,
                playback,
                rate,
                None,
            ),
            Err(error) => {
                if !emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Error(format!("audio output: {error}")),
                ) {
                    return;
                }
                emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Ended(Some("audio output unavailable".into())),
                );
                return;
            }
        },
    };
    let engine = match build_engine(out_rate) {
        Ok(engine) => engine,
        Err(error) => {
            if !emit_or_stop(
                &sink,
                &stop,
                RuntimeEvent::Error(format!("model load: {error}")),
            ) {
                return;
            }
            emit_or_stop(
                &sink,
                &stop,
                RuntimeEvent::Ended(Some("model load failed".into())),
            );
            return;
        }
    };
    if stop.load(Ordering::SeqCst) {
        emit_or_stop(&sink, &stop, RuntimeEvent::Ended(None));
        return;
    }
    struct InputGuard {
        _stream: Option<HostStream>,
    }
    let (input_guard, mic, in_rate) = match input {
        Some(input) => (InputGuard { _stream: None }, input.mic, input.rate),
        None => match start_input() {
            Ok((stream, mic, rate)) => (
                InputGuard {
                    _stream: Some(stream),
                },
                mic,
                rate,
            ),
            Err(error) => {
                if !emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Error(format!("audio input: {error}")),
                ) {
                    return;
                }
                emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Ended(Some("microphone unavailable".into())),
                );
                return;
            }
        },
    };

    enum Pipeline {
        Turn(RealtimePipeline),
        Frame(RealtimeFramePipeline),
    }
    let pipeline = if engine.frame_config().is_some() {
        match RealtimeFramePipeline::spawn(engine) {
            Ok(pipe) => Pipeline::Frame(pipe),
            Err(error) => {
                if !emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Error(format!("realtime frame pipeline: {error}")),
                ) {
                    return;
                }
                emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Ended(Some("realtime frame pipeline unavailable".into())),
                );
                return;
            }
        }
    } else {
        match RealtimePipeline::spawn(engine) {
            Ok(pipe) => Pipeline::Turn(pipe),
            Err(error) => {
                if !emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Error(format!("realtime pipeline: {error}")),
                ) {
                    return;
                }
                emit_or_stop(
                    &sink,
                    &stop,
                    RuntimeEvent::Ended(Some("realtime pipeline unavailable".into())),
                );
                return;
            }
        }
    };
    let assistant = Arc::new(AtomicBool::new(false));
    let latency = TurnLatency::new();
    let events = match &pipeline {
        Pipeline::Turn(pipe) => pipe.events().clone(),
        Pipeline::Frame(pipe) => pipe.events().clone(),
    };
    let consumer = match spawn_consumer(
        events,
        ring.clone(),
        out_rate,
        assistant.clone(),
        mic_enabled.clone(),
        audio.clone(),
        sink.clone(),
        stop.clone(),
        playback_flush.clone(),
        playback.clone(),
        latency.clone(),
        external_output,
    ) {
        Ok(consumer) => consumer,
        Err(error) => {
            if !emit_or_stop(
                &sink,
                &stop,
                RuntimeEvent::Error(format!("voice consumer: {error}")),
            ) {
                return;
            }
            emit_or_stop(
                &sink,
                &stop,
                RuntimeEvent::Ended(Some("voice consumer unavailable".into())),
            );
            return;
        }
    };

    if !emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Listening)) {
        drop(input_guard);
        drop(out_guard);
        drop(pipeline);
        let _ = consumer.join();
        return;
    }
    match &pipeline {
        Pipeline::Turn(pipe) => vad_loop(
            pipe,
            &mic,
            &ring,
            &playback_flush,
            &playback,
            &assistant,
            &sink,
            &stop,
            &interrupt,
            &mic_enabled,
            &latency,
            cfg,
            in_rate,
        ),
        Pipeline::Frame(pipe) => frame_loop(
            pipe,
            &mic,
            &ring,
            &playback_flush,
            &playback,
            &assistant,
            &sink,
            &stop,
            &interrupt,
            &mic_enabled,
            &latency,
            cfg,
            in_rate,
        ),
    }

    drop(input_guard);
    drop(out_guard);
    drop(pipeline);
    let _ = consumer.join();
    if emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Idle)) {
        emit_or_stop(&sink, &stop, RuntimeEvent::Ended(None));
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_consumer<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    events: crossbeam_channel::Receiver<VoiceEvent>,
    ring: Ring,
    out_rate: u32,
    assistant: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    audio: Stats,
    sink: Arc<Mutex<S>>,
    stop: Arc<AtomicBool>,
    playback_flush: Arc<AtomicBool>,
    playback: Playback,
    latency: Arc<TurnLatency>,
    output: Option<ExternalAudioOutput>,
) -> Result<JoinHandle<()>, String> {
    // Spawn a dedicated output thread when using ExternalAudioOutput (WebRTC path).
    // The consumer pushes PCM into a bounded ring (non-blocking); the output thread
    // drains the ring and calls the blocking write_mono_f32 (which does
    // block_on(capture_frame)). This decouples model generation from output latency:
    // the consumer never blocks on capture_frame, so the inference worker's event
    // channel never backs up, and the model keeps generating at full speed.
    let output_ring = output.as_ref().map(|_| {
        PcmRing::new(out_rate as usize * SPEAKER_RING_SECONDS)
    });
    let output_flush = Arc::new(AtomicBool::new(false));
    let output_stop = stop.clone();

    if let (Some(output), Some(ring)) = (&output, &output_ring) {
        let ring = ring.clone();
        let output = output.clone();
        let flush = output_flush.clone();
        let stop = output_stop.clone();
        std::thread::Builder::new()
            .name("voice-output".into())
            .spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    if flush.swap(false, Ordering::SeqCst) {
                        output.clear();
                        ring.clear();
                    }
                    // Drain a chunk from the ring (non-blocking pop)
                    let chunk = ring.drain_all();
                    if chunk.is_empty() {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    }
                    if let Err(e) = output.write_mono_f32(&chunk) {
                        eprintln!("[voice-output] speaker write failed: {e}");
                        break;
                    }
                }
                output.clear();
            })
            .map_err(|e| format!("spawn voice-output thread failed: {e}"))?;
    }

    std::thread::Builder::new()
        .name("voice-consumer".into())
        .spawn(move || {
            let mut transcript = String::new();
            let mut speaking = false;
            for event in events.iter() {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                let is_turn_complete = matches!(&event, VoiceEvent::TurnComplete);
                match event {
                    VoiceEvent::Text(text) => {
                        assistant.store(true, Ordering::SeqCst);
                        transcript.push_str(&text);
                        if !speaking {
                            speaking = true;
                            if !emit_or_stop(
                                &sink,
                                &stop,
                                RuntimeEvent::State(SessionState::Speaking),
                            ) {
                                break;
                            }
                        }
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::Transcript(transcript.clone()))
                        {
                            break;
                        }
                    }
                    VoiceEvent::Audio(pcm) => {
                        assistant.store(true, Ordering::SeqCst);
                        if !speaking {
                            speaking = true;
                            if !emit_or_stop(
                                &sink,
                                &stop,
                                RuntimeEvent::State(SessionState::Speaking),
                            ) {
                                break;
                            }
                        }
                        let level = rms(&pcm);
                        if !speaking {
                            vtrace!(
                                "consumer: first-audio of reply ({} samples, rms {level:.3})",
                                pcm.len()
                            );
                        }
                        if level > TURN_LATENCY_AGENT_RMS {
                            if let Some(ms) = latency.try_measure() {
                                audio.record_turn_latency(ms);
                                eprintln!(
                                    "[voice-latency] pause→first-audio {ms} ms (rating {}/5)",
                                    turn_latency_rating(ms)
                                );
                            }
                            if let Some(ms) = latency.first_word() {
                                eprintln!(
                                    "[voice-latency] first word {ms} ms after session start"
                                );
                            }
                        }
                        let external_output = output.is_some();
                        // ONE output path per I/O mode, no fallback chain:
                        // external output ⇒ the non-blocking ring drained by the
                        // dedicated voice-output thread (output_ring is ALWAYS
                        // constructed alongside an external output — the legacy
                        // direct write_mono_f32 branch was unreachable and is gone);
                        // no external output ⇒ the internal ring the CPAL callback
                        // consumes (standalone examples).
                        let dropped = if let Some(output_ring) = output_ring.as_ref() {
                            if playback_flush.swap(false, Ordering::SeqCst) {
                                output_flush.store(true, Ordering::SeqCst);
                            }
                            let d = output_ring.push_slice(&pcm);
                            if d == 0 {
                                playback.set_playing(level);
                            }
                            d
                        } else {
                            ring.push_slice(&pcm)
                        };
                        let queued = pcm.len().saturating_sub(dropped);
                        audio
                            .decoded_samples
                            .fetch_add(pcm.len() as u64, Ordering::Relaxed);
                        audio
                            .queued_samples
                            .fetch_add(queued as u64, Ordering::Relaxed);
                        audio
                            .dropped_samples
                            .fetch_add(dropped as u64, Ordering::Relaxed);
                        if external_output && queued > 0 {
                            audio
                                .played_samples
                                .fetch_add(queued as u64, Ordering::Relaxed);
                        }
                        if !emit_or_stop(
                            &sink,
                            &stop,
                            RuntimeEvent::Audio {
                                pcm,
                                rate: out_rate,
                            },
                        ) {
                            break;
                        }
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::Level(level)) {
                            break;
                        }
                    }
                    VoiceEvent::TurnComplete | VoiceEvent::Interrupted => {
                        vtrace!(
                            "consumer: {} -> playback idle (echo tail {}ms)",
                            if is_turn_complete {
                                "turn-complete"
                            } else {
                                "interrupted"
                            },
                            PLAYBACK_ECHO_TAIL_MS
                        );
                        assistant.store(false, Ordering::SeqCst);
                        latency.disarm();
                        if output_ring.is_some() {
                            playback.set_idle();
                        } else if output.is_some() {
                            playback.set_idle();
                        }
                        transcript.clear();
                        speaking = false;
                        if !emit_ready(&sink, &stop, &mic_enabled) {
                            break;
                        }
                    }
                    VoiceEvent::Error(error) => {
                        assistant.store(false, Ordering::SeqCst);
                        latency.disarm();
                        if output_ring.is_some() {
                            playback.set_idle();
                        } else if output.is_some() {
                            playback.set_idle();
                        }
                        transcript.clear();
                        speaking = false;
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::Error(error)) {
                            break;
                        }
                        if !emit_ready(&sink, &stop, &mic_enabled) {
                            break;
                        }
                    }
                }
            }
        })
        .map_err(|e| format!("spawn voice-consumer thread failed: {e}"))
}

#[allow(clippy::too_many_arguments)]
fn vad_loop<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    pipe: &RealtimePipeline,
    mic: &Mic,
    speaker: &Ring,
    playback_flush: &Arc<AtomicBool>,
    playback: &Playback,
    assistant: &Arc<AtomicBool>,
    sink: &Arc<Mutex<S>>,
    stop: &Arc<AtomicBool>,
    interrupt: &Arc<AtomicBool>,
    mic_enabled: &Arc<AtomicBool>,
    latency: &Arc<TurnLatency>,
    cfg: RuntimeConfig,
    in_rate: u32,
) {
    let window = (in_rate as usize / 5).max(1);
    let max_local = (in_rate as usize * MAX_UTTERANCE_SECONDS).max(window);
    let silence_stop = Duration::from_millis(cfg.silence_ms);
    let mut mic_buf = Vec::with_capacity(window * 2);
    let mut speaking = false;
    let mut start = 0usize;
    let mut read = 0usize;
    let mut last_voice = Instant::now();
    // Consecutive voiced windows observed while not yet `speaking`, and where that
    // run began — the barge-in sustain gate (spec 09, W5).
    let mut voiced_streak = 0usize;
    let mut streak_start = 0usize;
    // Trace-only: report gate transitions once, not every loop iteration.
    let mut traced_gate = false;

    while !stop.load(Ordering::SeqCst) {
        let _ = mic.wait_for_input(Duration::from_millis(40));

        if interrupt.swap(false, Ordering::SeqCst) {
            vtrace!("vad: session interrupt -> pipe.interrupt + playback flush");
            pipe.interrupt();
            playback_flush.store(true, Ordering::SeqCst);
            assistant.store(false, Ordering::SeqCst);
            speaking = false;
            voiced_streak = 0;
            if !emit_ready(sink, stop, mic_enabled) {
                return;
            }
        }

        if !mic_enabled.load(Ordering::SeqCst) {
            if !traced_gate {
                traced_gate = true;
                vtrace!("vad: mic disabled -> clearing capture");
            }
            mic.clear();
            mic_buf.clear();
            read = 0;
            speaking = false;
            voiced_streak = 0;
            continue;
        }

        if reference_audio_active(assistant, playback, speaker) && !cfg.can_interrupt {
            if !traced_gate {
                traced_gate = true;
                vtrace!("vad: reference audio active (no-interrupt mode) -> mic gated");
            }
            mic.clear();
            mic_buf.clear();
            read = 0;
            speaking = false;
            voiced_streak = 0;
            continue;
        }
        traced_gate = false;

        mic.drain_into(&mut mic_buf, max_local);
        let n = mic_buf.len();
        while read + window <= n {
            let threshold =
                reference_vad_threshold(cfg.vad_threshold, assistant, playback, speaker);
            if rms(&mic_buf[read..read + window]) > threshold {
                if !speaking {
                    if voiced_streak == 0 {
                        streak_start = read;
                    }
                    voiced_streak += 1;
                    // While the assistant is audible, one voiced window is as likely
                    // echo as speech — demand sustained voice before barging in.
                    // When it's quiet, engage on the first voiced window as before.
                    let reference = reference_audio_active(assistant, playback, speaker);
                    let needed = if reference { BARGE_IN_SUSTAIN_WINDOWS } else { 1 };
                    if voiced_streak >= needed {
                        vtrace!(
                            "vad: speech-start (streak {voiced_streak}, threshold {threshold:.4}, \
                             reference-active {reference}{})",
                            if reference { " -> BARGE-IN interrupt" } else { "" }
                        );
                        if reference {
                            pipe.interrupt();
                            playback_flush.store(true, Ordering::SeqCst);
                        }
                        speaking = true;
                        start = streak_start;
                        voiced_streak = 0;
                        if !emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Listening))
                        {
                            return;
                        }
                    }
                }
                last_voice = Instant::now();
                latency.mark_voice();
            } else if !speaking {
                voiced_streak = 0;
            }
            read += window;
        }

        let forced_end = speaking && n >= max_local;
        if speaking && (last_voice.elapsed() >= silence_stop || forced_end) {
            let end = read.min(n);
            let samples = mic_buf[start..end].to_vec();
            mic_buf.clear();
            read = 0;
            speaking = false;

            let dur_s = samples.len() as f32 / in_rate as f32;
            if dur_s >= cfg.min_utterance_s {
                if !emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Thinking)) {
                    return;
                }
                if pipe.submit(Utterance {
                    samples,
                    rate: in_rate,
                }) {
                    vtrace!("vad: utterance-end {dur_s:.2}s -> submitted");
                    latency.arm();
                } else {
                    // A full utterance queue silently eating the user's speech is a
                    // real user-facing event — always say so, not just under trace.
                    eprintln!(
                        "[voice] utterance ({dur_s:.2}s) DROPPED — pipeline busy \
                         (queue full); interrupting current reply"
                    );
                    pipe.interrupt();
                    playback_flush.store(true, Ordering::SeqCst);
                    assistant.store(false, Ordering::SeqCst);
                    if !emit_ready(sink, stop, mic_enabled) {
                        return;
                    }
                    continue;
                }
            } else {
                vtrace!("vad: utterance-end {dur_s:.2}s -> discarded (under min_utterance_s)");
            }
        } else if !speaking && n > in_rate as usize * 5 {
            mic_buf.clear();
            read = 0;
            voiced_streak = 0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn frame_loop<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    pipe: &RealtimeFramePipeline,
    mic: &Mic,
    _speaker: &Ring,
    playback_flush: &Arc<AtomicBool>,
    _playback: &Playback,
    assistant: &Arc<AtomicBool>,
    sink: &Arc<Mutex<S>>,
    stop: &Arc<AtomicBool>,
    interrupt: &Arc<AtomicBool>,
    mic_enabled: &Arc<AtomicBool>,
    latency: &Arc<TurnLatency>,
    cfg: RuntimeConfig,
    in_rate: u32,
) {
    let frame = pipe.config();
    let mut input = Vec::with_capacity((in_rate as usize / 20).max(1));
    let mut model = Vec::with_capacity(frame.frame_size * 2);
    let mut resampler = InputFrameResampler::new(in_rate, frame.sample_rate);
    let mut backpressure_reported = false;
    let interval = Duration::from_secs_f64(frame.frame_size as f64 / frame.sample_rate as f64);
    let mut next_silence = Instant::now() + interval;

    // Full-duplex frame loop (Moshi semantics): the model receives user audio
    // continuously, regardless of whether the assistant is speaking. No VAD
    // gating, no mic zeroing, no barge-in resets. The multistream LM handles
    // turn-taking natively — it processes user + assistant channels every frame.
    // Echo/AEC belongs below the model (hardware/WebRTC), not as model-input
    // zeroing. Explicit Stop/Interrupt are session-level controls only.
    while !stop.load(Ordering::SeqCst) {
        let has_input = mic.wait_for_input(Duration::from_millis(10));

        // Session-level interrupt (Stop button / typed input) — cuts queued
        // output/playback, but does NOT zero mic input or reset Mimi/LM state.
        if interrupt.swap(false, Ordering::SeqCst) {
            pipe.interrupt();
            playback_flush.store(true, Ordering::SeqCst);
            assistant.store(false, Ordering::SeqCst);
            backpressure_reported = false;
            if !emit_ready(sink, stop, mic_enabled) {
                return;
            }
        }

        // Mic pause (typed input / mic toggle) — stop capturing, but don't
        // feed silence to the model. When paused, the model simply doesn't
        // receive new frames; when resumed, it picks up the live stream.
        if !mic_enabled.load(Ordering::SeqCst) {
            mic.clear();
            input.clear();
            continue;
        }

        if !has_input && mic.len() == 0 {
            // No new audio from the native callback. Feed at most one silence
            // frame per model frame interval so the stream clock can advance
            // without flooding the bounded inference queue.
            let now = Instant::now();
            if now < next_silence {
                continue;
            }
            pad_next_model_frame(&mut model, frame.frame_size);
            next_silence = now + interval;
        } else {
            let before = input.len();
            mic.drain_into(&mut input, (in_rate as usize / 4).max(1));
            if input.len() == before && mic.len() == 0 {
                continue;
            } else {
                let chunk = input.split_off(before);
                // Telemetry only (spec 09, W1): note voiced input and arm the
                // pause→first-audio measurement when the assistant is quiet.
                // This never gates what reaches the model.
                if rms(&chunk) > cfg.vad_threshold {
                    latency.mark_voice();
                    if !assistant.load(Ordering::SeqCst) {
                        latency.arm();
                    }
                }
                // Always feed real mic audio — no zeroing, no VAD gating.
                model.extend(resampler.process(&chunk));
            }
        }

        while model.len() >= frame.frame_size {
            let next = model[..frame.frame_size].to_vec();
            model.drain(..frame.frame_size);
            match pipe.try_submit_frame(next) {
                Ok(()) => {
                    backpressure_reported = false;
                    next_silence = Instant::now() + interval;
                    continue;
                }
                Err(FrameSubmitError::Full) => {
                    // Stay realtime and nonblocking: drop buffered capture
                    // until the model catches up. Queue pressure is not a
                    // semantic interruption and must not reset Mimi/LM state.
                    model.clear();
                    if !backpressure_reported {
                        backpressure_reported = true;
                        if !emit_ready(sink, stop, mic_enabled) {
                            return;
                        }
                    }
                }
                Err(FrameSubmitError::Disconnected | FrameSubmitError::WrongSize) => return,
            }
            break;
        }

        if input.len() > in_rate as usize {
            input.clear();
        }
    }
}

fn pad_next_model_frame(model: &mut Vec<f32>, frame_size: usize) {
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

struct InputFrameResampler {
    from: u32,
    to: u32,
    carry: Option<f32>,
}

impl InputFrameResampler {
    fn new(from: u32, to: u32) -> Self {
        Self {
            from,
            to,
            carry: None,
        }
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
        crate::resample::resample_slice(input, self.from, self.to)
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

fn reference_audio_active(
    assistant: &AtomicBool,
    playback: &PlaybackReference,
    speaker: &PcmRing,
) -> bool {
    assistant.load(Ordering::SeqCst) || playback.active_or_tail() || speaker.len() > 0
}

fn ready_state(mic_enabled: &AtomicBool) -> SessionState {
    if mic_enabled.load(Ordering::SeqCst) {
        return SessionState::Listening;
    }
    SessionState::Idle
}

fn emit_ready<S: FnMut(RuntimeEvent) -> bool>(
    sink: &Mutex<S>,
    stop: &Arc<AtomicBool>,
    mic_enabled: &AtomicBool,
) -> bool {
    emit_or_stop(sink, stop, RuntimeEvent::State(ready_state(mic_enabled)))
        && emit_or_stop(sink, stop, RuntimeEvent::Level(0.0))
}

fn reference_vad_threshold(
    base: f32,
    assistant: &AtomicBool,
    playback: &PlaybackReference,
    speaker: &PcmRing,
) -> f32 {
    if !reference_audio_active(assistant, playback, speaker) {
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
fn start_input() -> Res<(HostStream, Mic, u32)> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    if rate == 0 {
        return Err("audio input sample rate is zero".into());
    }
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let mic = PcmRing::new(rate as usize * MIC_RING_SECONDS);
    let err = |e| eprintln!("[voice] input stream error: {e}");

    macro_rules! stream {
        ($t:ty, $conv:expr) => {{
            let mic = mic.clone();
            dev.build_input_stream(
                &cfg,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    if channels <= 1 {
                        for &sample in data {
                            mic.push($conv(sample));
                        }
                        mic.notify();
                        return;
                    }
                    for frame in data.chunks(channels) {
                        let sum = frame.iter().map(|&s| $conv(s)).sum::<f32>();
                        mic.push(sum / frame.len() as f32);
                    }
                    mic.notify();
                },
                err,
                None,
            )
        }};
    }

    let stream = match fmt {
        cpal::SampleFormat::F32 => stream!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => stream!(i16, |s: i16| s as f32 / 32768.0),
        cpal::SampleFormat::U16 => stream!(u16, |s: u16| (s as f32 - 32768.0) / 32768.0),
        other => return Err(format!("unsupported input sample format {other:?}").into()),
    }?;
    stream.play()?;
    Ok((stream, mic, rate))
}

#[cfg(not(feature = "audio-io"))]
fn start_input() -> Res<(HostStream, Mic, u32)> {
    Err("liquid-audio was built without the `audio-io` fallback; external WebRTC audio input is required".into())
}

#[cfg(feature = "audio-io")]
fn start_output(audio: Stats, flush: Arc<AtomicBool>) -> Res<(HostStream, Ring, Playback, u32)> {
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
    let ring = PcmRing::new(rate as usize * SPEAKER_RING_SECONDS);
    let playback = PlaybackReference::new();
    let prebuffer = (rate as usize * SPEAKER_PREBUFFER_MS / 1000).max(1);
    let idle_reset = (rate as usize / 2).max(1);
    let err = |e| eprintln!("[voice] output stream error: {e}");

    macro_rules! stream {
        ($t:ty, $conv:expr) => {{
            let ring = ring.clone();
            let flush = flush.clone();
            let playback = playback.clone();
            let audio = audio.clone();
            let mut started = false;
            let mut empty_frames = 0usize;
            let mut starve_ticks = 0usize;
            dev.build_output_stream(
                &cfg,
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let silence: $t = $conv(0.0);
                    if flush.swap(false, Ordering::SeqCst) {
                        ring.clear();
                        playback.set_idle();
                        started = false;
                        empty_frames = 0;
                        starve_ticks = 0;
                    }
                    if !started {
                        let queued = ring.len();
                        if queued >= prebuffer {
                            started = true;
                        } else if queued > 0 {
                            // A reply SHORTER than the prebuffer never crosses the
                            // threshold on its own and would sit queued forever.
                            // After a few callbacks with data waiting and nothing
                            // arriving, start anyway.
                            starve_ticks += 1;
                            if starve_ticks >= 3 {
                                started = true;
                            }
                        } else {
                            starve_ticks = 0;
                        }
                        if !started {
                            playback.set_idle();
                            for out in data.iter_mut() {
                                *out = silence;
                            }
                            return;
                        }
                        empty_frames = 0;
                        starve_ticks = 0;
                    }
                    let mut played = false;
                    let mut played_frames = 0usize;
                    let mut underrun_frames = 0usize;
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for frame in data.chunks_mut(channels) {
                        let Some(next) = ring.pop() else {
                            underrun_frames += 1;
                            for out in frame.iter_mut() {
                                *out = silence;
                            }
                            continue;
                        };
                        played = true;
                        played_frames += 1;
                        sum += next * next;
                        count += 1;
                        let sample: $t = $conv(next);
                        for out in frame.iter_mut() {
                            *out = sample;
                        }
                    }
                    if played {
                        audio
                            .played_samples
                            .fetch_add(played_frames as u64, Ordering::Relaxed);
                        playback.set_playing((sum / count.max(1) as f32).sqrt());
                        empty_frames = 0;
                    } else if started {
                        empty_frames += data.len() / channels;
                        if empty_frames >= idle_reset {
                            playback.set_idle();
                            started = false;
                            empty_frames = 0;
                        }
                    }
                    if underrun_frames > 0 {
                        audio
                            .underrun_frames
                            .fetch_add(underrun_frames as u64, Ordering::Relaxed);
                    }
                },
                err,
                None,
            )
        }};
    }

    let stream = match fmt {
        cpal::SampleFormat::F32 => stream!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => stream!(i16, |s: f32| (s.clamp(-1.0, 1.0) * 32767.0) as i16),
        cpal::SampleFormat::U16 => stream!(u16, |s: f32| {
            ((s.clamp(-1.0, 1.0) * 32767.0) as i32 + 32768) as u16
        }),
        other => return Err(format!("unsupported output sample format {other:?}").into()),
    }?;
    stream.play()?;
    Ok((stream, ring, playback, rate))
}

#[cfg(not(feature = "audio-io"))]
fn start_output(_audio: Stats, _flush: Arc<AtomicBool>) -> Res<(HostStream, Ring, Playback, u32)> {
    Err("liquid-audio was built without the `audio-io` fallback; external WebRTC audio output is required".into())
}

/// Play a short tone through the same native output path used by the realtime voice session.
///
/// This is intentionally production code, not a test helper: the desktop settings UI uses it
/// to isolate "CoreAudio/output path is silent" from "the model did not emit PCM".
pub fn play_output_probe(
    duration: Duration,
    frequency_hz: f32,
    amplitude: f32,
) -> Result<(), String> {
    let audio = Arc::new(AudioStats::default());
    let flush = Arc::new(AtomicBool::new(false));
    let (_stream, ring, _playback, rate) =
        start_output(audio, flush).map_err(|e| format!("audio output: {e}"))?;
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
    let dropped = ring.push_slice(&samples);
    if dropped > 0 {
        return Err(format!("audio output ring overflowed by {dropped} samples"));
    }
    std::thread::sleep(duration + Duration::from_millis(250));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_ring_preserves_fifo_order() {
        let ring = PcmRing::new(3);
        assert!(ring.push(1.0));
        assert!(ring.push(2.0));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop(), Some(1.0));
        assert!(ring.push(3.0));
        assert_eq!(ring.pop(), Some(2.0));
        assert_eq!(ring.pop(), Some(3.0));
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn pcm_ring_is_bounded_and_drops_new_samples_when_full() {
        let ring = PcmRing::new(2);
        assert!(ring.push(1.0));
        assert!(ring.push(2.0));
        assert!(!ring.push(3.0));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop(), Some(1.0));
        assert_eq!(ring.pop(), Some(2.0));
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn pcm_ring_clear_drops_buffered_samples() {
        let ring = PcmRing::new(4);
        assert_eq!(ring.push_slice(&[1.0, 2.0, 3.0]), 0);
        ring.clear();
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.pop(), None);
        assert!(ring.push(4.0));
        assert_eq!(ring.pop(), Some(4.0));
    }

    #[test]
    fn playback_reference_extends_echo_gate_after_generation_finishes() {
        let assistant = AtomicBool::new(false);
        let playback = PlaybackReference::new();
        let speaker = PcmRing::new(4);

        // Never played: no tail, not reference audio.
        assert!(!reference_audio_active(&assistant, &playback, &speaker));
        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback, &speaker),
            0.012
        );

        playback.set_playing(0.08);
        assert!(reference_audio_active(&assistant, &playback, &speaker));
        assert!(reference_vad_threshold(0.012, &assistant, &playback, &speaker) >= 0.1);

        // Idle starts the ECHO TAIL: the speaker is still draining and the room
        // still ringing, so the reference gate must stay up (the bug this guards:
        // the model's own trailing audio re-entering as a fresh "user" utterance).
        playback.set_idle();
        assert!(!playback.active(), "raw active flag drops on idle");
        assert!(
            reference_audio_active(&assistant, &playback, &speaker),
            "echo tail keeps reference audio active right after idle"
        );

        // After the tail window passes, the room is presumed quiet.
        std::thread::sleep(Duration::from_millis(PLAYBACK_ECHO_TAIL_MS + 60));
        assert!(!reference_audio_active(&assistant, &playback, &speaker));

        // Playing again clears the tail bookkeeping.
        playback.set_playing(0.05);
        assert!(reference_audio_active(&assistant, &playback, &speaker));
    }

    #[test]
    fn playback_reference_requires_barge_in_above_echo_floor() {
        let assistant = AtomicBool::new(false);
        let playback = PlaybackReference::new();
        let speaker = PcmRing::new(4);
        playback.set_playing(0.08);

        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback, &speaker),
            0.08 * PLAYBACK_ECHO_MULTIPLIER
        );
    }

    #[test]
    fn assistant_generation_is_reference_audio_even_before_playback_starts() {
        let assistant = AtomicBool::new(true);
        let playback = PlaybackReference::new();
        let speaker = PcmRing::new(4);

        assert!(reference_audio_active(&assistant, &playback, &speaker));
        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback, &speaker),
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
    fn silence_padding_tops_off_one_model_frame() {
        let mut partial = vec![1.0, 2.0, 3.0];
        pad_next_model_frame(&mut partial, 5);
        assert_eq!(partial, vec![1.0, 2.0, 3.0, 0.0, 0.0]);

        let mut empty = Vec::new();
        pad_next_model_frame(&mut empty, 4);
        assert_eq!(empty, vec![0.0, 0.0, 0.0, 0.0]);

        let mut aligned = vec![1.0, 2.0, 3.0, 4.0];
        pad_next_model_frame(&mut aligned, 4);
        assert_eq!(aligned, vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn interrupt_flushes_queued_playback_from_runtime_handle() {
        let (live, _main) =
            VoiceRuntime::prepare(RuntimeConfig::default(), |_| unreachable!(), |_| true);

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
        let sink = Mutex::new(move |event| {
            captured.lock().expect("events lock").push(event);
            true
        });
        let stop = Arc::new(AtomicBool::new(false));
        let mic = AtomicBool::new(false);

        assert!(emit_ready(&sink, &stop, &mic));
        let events = events.lock().expect("events lock");
        assert!(matches!(
            events.as_slice(),
            [
                RuntimeEvent::State(SessionState::Idle),
                RuntimeEvent::Level(level)
            ] if *level == 0.0
        ));
    }

    #[test]
    fn consumer_preserves_decoded_pcm_for_native_transports() {
        let (tx, rx) = crossbeam_channel::bounded(4);
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let sink = Arc::new(Mutex::new(move |event| {
            captured.lock().expect("events lock").push(event);
            true
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let mic = Arc::new(AtomicBool::new(true));
        let assistant = Arc::new(AtomicBool::new(false));
        let ring = PcmRing::new(16);
        let audio = Arc::new(AudioStats::default());

        let playback = PlaybackReference::new();
        let consumer = spawn_consumer(
            rx,
            ring,
            48_000,
            assistant,
            mic,
            audio.clone(),
            sink,
            stop,
            Arc::new(AtomicBool::new(false)),
            playback,
            TurnLatency::new(),
            None,
        )
        .expect("spawn consumer");
        tx.send(VoiceEvent::Audio(vec![0.25, -0.25])).unwrap();
        drop(tx);
        consumer.join().unwrap();

        let events = events.lock().expect("events lock");
        assert!(events.iter().any(|event| matches!(
            event,
            RuntimeEvent::Audio { pcm, rate } if pcm == &vec![0.25, -0.25] && *rate == 48_000
        )));
        assert!(events
            .iter()
            .any(|event| matches!(event, RuntimeEvent::Level(level) if *level > 0.0)));
        assert_eq!(
            audio.snapshot(),
            AudioStatsSnapshot {
                decoded_samples: 2,
                queued_samples: 2,
                dropped_samples: 0,
                played_samples: 0,
                underrun_frames: 0,
                turn_count: 0,
                last_turn_latency_ms: 0,
                mean_turn_latency_ms: 0,
            }
        );
    }

    #[test]
    fn consumer_counts_external_output_as_played_after_native_write() {
        let (tx, rx) = crossbeam_channel::bounded(4);
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let sink = Arc::new(Mutex::new(move |event| {
            captured.lock().expect("events lock").push(event);
            true
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let mic = Arc::new(AtomicBool::new(true));
        let assistant = Arc::new(AtomicBool::new(false));
        let ring = PcmRing::new(16);
        let audio = Arc::new(AudioStats::default());
        let written = Arc::new(AtomicUsize::new(0));
        let counted = written.clone();
        let output = ExternalAudioOutput::new(
            48_000,
            move |pcm, rate| {
                assert_eq!(rate, 48_000);
                counted.fetch_add(pcm.len(), Ordering::SeqCst);
                Ok(())
            },
            || {},
        )
        .expect("external output");
        let playback = PlaybackReference::new();

        let consumer = spawn_consumer(
            rx,
            ring,
            48_000,
            assistant,
            mic,
            audio.clone(),
            sink,
            stop,
            Arc::new(AtomicBool::new(false)),
            playback,
            TurnLatency::new(),
            Some(output),
        )
        .expect("spawn consumer");
        tx.send(VoiceEvent::Audio(vec![0.25, -0.25])).unwrap();
        drop(tx);
        consumer.join().unwrap();

        // The consumer pushes PCM to a bounded ring; a dedicated output thread
        // drains the ring and calls write_mono_f32. Wait for it to finish.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while written.load(Ordering::SeqCst) < 2 {
            if std::time::Instant::now() >= deadline {
                panic!("output thread did not drain the ring within 2s");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert_eq!(written.load(Ordering::SeqCst), 2);
        assert_eq!(
            audio.snapshot(),
            AudioStatsSnapshot {
                decoded_samples: 2,
                queued_samples: 2,
                dropped_samples: 0,
                played_samples: 2,
                underrun_frames: 0,
                turn_count: 0,
                last_turn_latency_ms: 0,
                mean_turn_latency_ms: 0,
            }
        );
    }

    #[test]
    fn queued_speaker_audio_is_reference_audio_before_output_callback_runs() {
        let assistant = AtomicBool::new(false);
        let playback = PlaybackReference::new();
        let speaker = PcmRing::new(4);

        assert!(!reference_audio_active(&assistant, &playback, &speaker));
        assert_eq!(speaker.push_slice(&[0.1, -0.1]), 0);
        assert!(reference_audio_active(&assistant, &playback, &speaker));
        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback, &speaker),
            0.012 * PLAYBACK_VAD_MULTIPLIER
        );

        speaker.clear();
        assert!(!reference_audio_active(&assistant, &playback, &speaker));
    }
}
