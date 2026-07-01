//! In-process, thread-managed voice service.
//!
//! This promotes the full-duplex example loop into a reusable runtime: CPAL mic
//! capture, energy VAD, realtime model inference, CPAL playback, barge-in, and
//! mic gating all run on Rust threads in this process. The Tauri layer owns one
//! of these handles and maps [`RuntimeEvent`]s onto its IPC channel.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{Lfm2VoiceEngine, RealtimePipeline, Utterance, VoiceEvent};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Ring = Arc<PcmRing>;
type Mic = Arc<PcmRing>;
type Playback = Arc<PlaybackReference>;
pub type RuntimeMain = Box<dyn FnOnce() + Send + 'static>;

const MIC_RING_SECONDS: usize = 6;
const SPEAKER_RING_SECONDS: usize = 4;
const MAX_UTTERANCE_SECONDS: usize = 30;
const PLAYBACK_VAD_MULTIPLIER: f32 = 3.0;
const PLAYBACK_ECHO_MULTIPLIER: f32 = 2.5;

struct PcmRing {
    buf: Box<[UnsafeCell<f32>]>,
    cap: usize,
    read: AtomicUsize,
    write: AtomicUsize,
}

// The runtime uses each ring as single-producer/single-consumer: CPAL input -> VAD,
// model event consumer -> CPAL output. `clear` stays on the consumer side: VAD clears
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
        Arc::new(Self {
            buf,
            cap,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
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
        dropped
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

    fn clear(&self) {
        let write = self.write.load(Ordering::Acquire);
        self.read.store(write, Ordering::Release);
    }
}

struct PlaybackReference {
    active: AtomicBool,
    rms_bits: AtomicU32,
}

impl PlaybackReference {
    fn new() -> Playback {
        Arc::new(Self {
            active: AtomicBool::new(false),
            rms_bits: AtomicU32::new(0.0f32.to_bits()),
        })
    }

    fn set_playing(&self, rms: f32) {
        self.rms_bits
            .store(rms.max(0.0).to_bits(), Ordering::Release);
        self.active.store(true, Ordering::Release);
    }

    fn set_idle(&self) {
        self.active.store(false, Ordering::Release);
        self.rms_bits.store(0.0f32.to_bits(), Ordering::Release);
    }

    fn active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    fn rms(&self) -> f32 {
        f32::from_bits(self.rms_bits.load(Ordering::Acquire))
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
            silence_ms: 800,
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
    done: Arc<AtomicBool>,
    session: Option<JoinHandle<()>>,
}

impl VoiceRuntime {
    /// Build the runtime handle plus the service-thread body without spawning it. The Tauri
    /// desktop kernel uses this to make its ThreadManager own the top-level LFM2 service thread;
    /// [`Self::start`] remains a fallible standalone convenience path for examples.
    pub fn prepare(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Lfm2VoiceEngine, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> (VoiceRuntime, RuntimeMain) {
        let stop = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let done = Arc::new(AtomicBool::new(false));
        let live = VoiceRuntime {
            stop: stop.clone(),
            interrupt: interrupt.clone(),
            mic_enabled: mic_enabled.clone(),
            done: done.clone(),
            session: None,
        };
        let main: RuntimeMain = Box::new(move || {
            session_loop(cfg, build_engine, sink, stop, interrupt, mic_enabled);
            done.store(true, Ordering::SeqCst);
        });
        (live, main)
    }

    /// Spawn the session thread and return immediately.
    pub fn start(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Lfm2VoiceEngine, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> Result<Self, String> {
        let (mut live, main) = Self::prepare(cfg, build_engine, sink);
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
    }

    /// Pause/resume mic capture without ending the session.
    pub fn set_mic_enabled(&self, on: bool) {
        self.mic_enabled.store(on, Ordering::SeqCst);
    }

    /// Whether mic capture is currently allowed.
    pub fn mic_enabled(&self) -> bool {
        self.mic_enabled.load(Ordering::SeqCst)
    }

    /// Whether the session thread has exited.
    pub fn is_finished(&self) -> bool {
        self.done.load(Ordering::SeqCst)
            || self.session.as_ref().is_some_and(JoinHandle::is_finished)
    }

    /// Signal stop and join the session thread.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(session) = self.session.take() {
            let _ = session.join();
            self.done.store(true, Ordering::SeqCst);
        }
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
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
    build_engine: impl FnOnce(u32) -> Result<Lfm2VoiceEngine, String>,
    sink: S,
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
) {
    let sink = Arc::new(Mutex::new(sink));

    if !emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Loading)) {
        return;
    }
    let (out_stream, ring, playback_flush, playback, out_rate) = match start_output() {
        Ok(output) => output,
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
    let (in_stream, mic, in_rate) = match start_input() {
        Ok(input) => input,
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
    };

    let pipe = match RealtimePipeline::spawn(engine) {
        Ok(pipe) => pipe,
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
    };
    let assistant = Arc::new(AtomicBool::new(false));
    let consumer = match spawn_consumer(
        pipe.events().clone(),
        ring.clone(),
        out_rate,
        assistant.clone(),
        mic_enabled.clone(),
        sink.clone(),
        stop.clone(),
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
        drop(in_stream);
        drop(out_stream);
        drop(pipe);
        let _ = consumer.join();
        return;
    }
    vad_loop(
        &pipe,
        &mic,
        &ring,
        &playback_flush,
        &playback,
        &assistant,
        &sink,
        &stop,
        &interrupt,
        &mic_enabled,
        cfg,
        in_rate,
    );

    drop(in_stream);
    drop(out_stream);
    drop(pipe);
    let _ = consumer.join();
    if emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Idle)) {
        emit_or_stop(&sink, &stop, RuntimeEvent::Ended(None));
    }
}

fn spawn_consumer<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    events: crossbeam_channel::Receiver<VoiceEvent>,
    ring: Ring,
    out_rate: u32,
    assistant: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    sink: Arc<Mutex<S>>,
    stop: Arc<AtomicBool>,
) -> Result<JoinHandle<()>, String> {
    std::thread::Builder::new()
        .name("voice-consumer".into())
        .spawn(move || {
            let mut transcript = String::new();
            let mut speaking = false;
            for event in events.iter() {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
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
                        ring.push_slice(&pcm);
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
                        assistant.store(false, Ordering::SeqCst);
                        transcript.clear();
                        speaking = false;
                        if !emit_ready(&sink, &stop, &mic_enabled) {
                            break;
                        }
                    }
                    VoiceEvent::Error(error) => {
                        assistant.store(false, Ordering::SeqCst);
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

    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(40));

        if interrupt.swap(false, Ordering::SeqCst) {
            pipe.interrupt();
            playback_flush.store(true, Ordering::SeqCst);
            assistant.store(false, Ordering::SeqCst);
            speaking = false;
            if !emit_ready(sink, stop, mic_enabled) {
                return;
            }
        }

        if !mic_enabled.load(Ordering::SeqCst) {
            mic.clear();
            mic_buf.clear();
            read = 0;
            speaking = false;
            continue;
        }

        if reference_audio_active(assistant, playback, speaker) && !cfg.can_interrupt {
            mic.clear();
            mic_buf.clear();
            read = 0;
            speaking = false;
            continue;
        }

        mic.drain_into(&mut mic_buf, max_local);
        let n = mic_buf.len();
        while read + window <= n {
            let threshold =
                reference_vad_threshold(cfg.vad_threshold, assistant, playback, speaker);
            if rms(&mic_buf[read..read + window]) > threshold {
                if !speaking {
                    if reference_audio_active(assistant, playback, speaker) {
                        pipe.interrupt();
                        playback_flush.store(true, Ordering::SeqCst);
                    }
                    speaking = true;
                    start = read;
                    if !emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Listening)) {
                        return;
                    }
                }
                last_voice = Instant::now();
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

            if samples.len() as f32 / in_rate as f32 >= cfg.min_utterance_s {
                if !emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Thinking)) {
                    return;
                }
                if !pipe.submit(Utterance {
                    samples,
                    rate: in_rate,
                }) {
                    emit_or_stop(
                        sink,
                        stop,
                        RuntimeEvent::Error("voice inference worker busy or stopped".into()),
                    );
                    return;
                }
            }
        } else if !speaking && n > in_rate as usize * 5 {
            mic_buf.clear();
            read = 0;
        }
    }
}

fn reference_audio_active(
    assistant: &AtomicBool,
    playback: &PlaybackReference,
    speaker: &PcmRing,
) -> bool {
    assistant.load(Ordering::SeqCst) || playback.active() || speaker.len() > 0
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

fn start_input() -> Res<(cpal::Stream, Mic, u32)> {
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
                        return;
                    }
                    for frame in data.chunks(channels) {
                        let sum = frame.iter().map(|&s| $conv(s)).sum::<f32>();
                        mic.push(sum / frame.len() as f32);
                    }
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

fn start_output() -> Res<(cpal::Stream, Ring, Arc<AtomicBool>, Playback, u32)> {
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
    let flush = Arc::new(AtomicBool::new(false));
    let playback = PlaybackReference::new();
    let prebuffer = (rate as usize / 5).max(1);
    let idle_reset = (rate as usize / 2).max(1);
    let err = |e| eprintln!("[voice] output stream error: {e}");

    macro_rules! stream {
        ($t:ty, $conv:expr) => {{
            let ring = ring.clone();
            let flush = flush.clone();
            let playback = playback.clone();
            let mut started = false;
            let mut empty_frames = 0usize;
            dev.build_output_stream(
                &cfg,
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let silence: $t = $conv(0.0);
                    if flush.swap(false, Ordering::SeqCst) {
                        ring.clear();
                        playback.set_idle();
                        started = false;
                        empty_frames = 0;
                    }
                    if !started {
                        if ring.len() < prebuffer {
                            playback.set_idle();
                            for out in data.iter_mut() {
                                *out = silence;
                            }
                            return;
                        }
                        started = true;
                        empty_frames = 0;
                    }
                    let mut played = false;
                    let mut sum = 0.0f32;
                    let mut count = 0usize;
                    for frame in data.chunks_mut(channels) {
                        let Some(next) = ring.pop() else {
                            for out in frame.iter_mut() {
                                *out = silence;
                            }
                            continue;
                        };
                        played = true;
                        sum += next * next;
                        count += 1;
                        let sample: $t = $conv(next);
                        for out in frame.iter_mut() {
                            *out = sample;
                        }
                    }
                    if played {
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
    Ok((stream, ring, flush, playback, rate))
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

        assert!(!reference_audio_active(&assistant, &playback, &speaker));
        assert_eq!(
            reference_vad_threshold(0.012, &assistant, &playback, &speaker),
            0.012
        );

        playback.set_playing(0.08);
        assert!(reference_audio_active(&assistant, &playback, &speaker));
        assert!(reference_vad_threshold(0.012, &assistant, &playback, &speaker) >= 0.1);

        playback.set_idle();
        assert!(!reference_audio_active(&assistant, &playback, &speaker));
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
    fn terminal_turn_state_follows_mic_enabled() {
        let mic = AtomicBool::new(true);
        assert_eq!(ready_state(&mic), SessionState::Listening);

        mic.store(false, Ordering::SeqCst);
        assert_eq!(ready_state(&mic), SessionState::Idle);
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

        let consumer =
            spawn_consumer(rx, ring, 48_000, assistant, mic, sink, stop).expect("spawn consumer");
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
