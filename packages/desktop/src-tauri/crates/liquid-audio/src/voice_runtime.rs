//! In-process, thread-managed voice service.
//!
//! This promotes the full-duplex example loop into a reusable runtime: CPAL mic
//! capture, energy VAD, realtime model inference, CPAL playback, barge-in, and
//! mic gating all run on Rust threads in this process. The Tauri layer owns one
//! of these handles and maps [`RuntimeEvent`]s onto its IPC channel.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::{Lfm2VoiceEngine, RealtimePipeline, Utterance, VoiceEvent};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
type Ring = Arc<Mutex<VecDeque<f32>>>;
type Mic = Arc<Mutex<Vec<f32>>>;

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
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.012,
            silence_ms: 800,
            min_utterance_s: 0.3,
        }
    }
}

/// Handle to a running voice service.
pub struct VoiceRuntime {
    stop: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    mic_enabled: Arc<AtomicBool>,
    session: Option<JoinHandle<()>>,
}

impl VoiceRuntime {
    /// Spawn the session thread and return immediately.
    pub fn start(
        cfg: RuntimeConfig,
        build_engine: impl FnOnce(u32) -> Result<Lfm2VoiceEngine, String> + Send + 'static,
        sink: impl FnMut(RuntimeEvent) -> bool + Send + 'static,
    ) -> VoiceRuntime {
        let stop = Arc::new(AtomicBool::new(false));
        let interrupt = Arc::new(AtomicBool::new(false));
        let mic_enabled = Arc::new(AtomicBool::new(true));
        let session = std::thread::Builder::new()
            .name("voice-session".into())
            .spawn({
                let stop = stop.clone();
                let interrupt = interrupt.clone();
                let mic_enabled = mic_enabled.clone();
                move || session_loop(cfg, build_engine, sink, stop, interrupt, mic_enabled)
            })
            .expect("spawn voice-session thread");
        VoiceRuntime {
            stop,
            interrupt,
            mic_enabled,
            session: Some(session),
        }
    }

    /// Abort the current reply and flush queued playback.
    pub fn interrupt(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
    }

    /// Pause/resume mic capture without ending the session.
    pub fn set_mic_enabled(&self, on: bool) {
        self.mic_enabled.store(on, Ordering::SeqCst);
    }

    /// Whether the session thread has exited.
    pub fn is_finished(&self) -> bool {
        self.session.as_ref().is_some_and(JoinHandle::is_finished)
    }

    /// Signal stop and join the session thread.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(session) = self.session.take() {
            let _ = session.join();
        }
    }
}

impl Drop for VoiceRuntime {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(session) = self.session.take() {
            let _ = session.join();
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
    let (out_stream, ring, out_rate) = match start_output() {
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

    let pipe = RealtimePipeline::spawn(engine);
    let assistant = Arc::new(AtomicBool::new(false));
    let consumer = spawn_consumer(
        pipe.events().clone(),
        ring.clone(),
        assistant.clone(),
        sink.clone(),
        stop.clone(),
    );

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
    assistant: Arc<AtomicBool>,
    sink: Arc<Mutex<S>>,
    stop: Arc<AtomicBool>,
) -> JoinHandle<()> {
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
                        if let Ok(mut ring) = ring.lock() {
                            ring.extend(pcm);
                        }
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::Level(level)) {
                            break;
                        }
                    }
                    VoiceEvent::TurnComplete | VoiceEvent::Interrupted => {
                        assistant.store(false, Ordering::SeqCst);
                        transcript.clear();
                        speaking = false;
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Listening))
                        {
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
                        if !emit_or_stop(&sink, &stop, RuntimeEvent::State(SessionState::Listening))
                        {
                            break;
                        }
                    }
                }
            }
        })
        .expect("spawn voice-consumer thread")
}

#[allow(clippy::too_many_arguments)]
fn vad_loop<S: FnMut(RuntimeEvent) -> bool + Send + 'static>(
    pipe: &RealtimePipeline,
    mic: &Mic,
    ring: &Ring,
    assistant: &Arc<AtomicBool>,
    sink: &Arc<Mutex<S>>,
    stop: &Arc<AtomicBool>,
    interrupt: &Arc<AtomicBool>,
    mic_enabled: &Arc<AtomicBool>,
    cfg: RuntimeConfig,
    in_rate: u32,
) {
    let window = (in_rate as usize / 5).max(1);
    let silence_stop = Duration::from_millis(cfg.silence_ms);
    let mut speaking = false;
    let mut start = 0usize;
    let mut read = 0usize;
    let mut last_voice = Instant::now();

    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(40));

        if interrupt.swap(false, Ordering::SeqCst) {
            pipe.interrupt();
            if let Ok(mut ring) = ring.lock() {
                ring.clear();
            }
            assistant.store(false, Ordering::SeqCst);
            speaking = false;
            if !emit_or_stop(sink, stop, RuntimeEvent::State(SessionState::Listening)) {
                return;
            }
        }

        if !mic_enabled.load(Ordering::SeqCst) {
            if let Ok(mut mic) = mic.lock() {
                mic.clear();
            }
            read = 0;
            speaking = false;
            continue;
        }

        let mut mic_buf = match mic.lock() {
            Ok(mic_buf) => mic_buf,
            Err(_) => {
                emit_or_stop(
                    sink,
                    stop,
                    RuntimeEvent::Error("voice mic lock poisoned".into()),
                );
                return;
            }
        };
        let n = mic_buf.len();
        while read + window <= n {
            let threshold = if assistant.load(Ordering::SeqCst) {
                cfg.vad_threshold * 3.0
            } else {
                cfg.vad_threshold
            };
            if rms(&mic_buf[read..read + window]) > threshold {
                if !speaking {
                    if assistant.load(Ordering::SeqCst) {
                        pipe.interrupt();
                        if let Ok(mut ring) = ring.lock() {
                            ring.clear();
                        }
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

        if speaking && last_voice.elapsed() >= silence_stop {
            let end = read.min(n);
            let samples = mic_buf[start..end].to_vec();
            mic_buf.clear();
            read = 0;
            speaking = false;
            drop(mic_buf);

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
                        RuntimeEvent::Error("voice inference worker stopped".into()),
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

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
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
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let mic: Mic = Arc::new(Mutex::new(Vec::new()));
    let err = |e| eprintln!("[voice] input stream error: {e}");

    macro_rules! stream {
        ($t:ty, $conv:expr) => {{
            let mic = mic.clone();
            dev.build_input_stream(
                &cfg,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> = data.iter().map(|&s| $conv(s)).collect();
                    if let Ok(mut mic) = mic.try_lock() {
                        mic.extend_from_slice(&downmix(&samples, channels));
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

fn start_output() -> Res<(cpal::Stream, Ring, u32)> {
    let host = cpal::default_host();
    let dev = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = dev.default_output_config()?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let ring: Ring = Arc::new(Mutex::new(VecDeque::new()));
    let prebuffer = (rate as usize / 5).max(1);
    let err = |e| eprintln!("[voice] output stream error: {e}");

    macro_rules! stream {
        ($t:ty, $conv:expr) => {{
            let ring = ring.clone();
            let mut started = false;
            dev.build_output_stream(
                &cfg,
                move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let silence: $t = $conv(0.0);
                    let Ok(mut ring) = ring.try_lock() else {
                        for out in data.iter_mut() {
                            *out = silence;
                        }
                        return;
                    };
                    if !started {
                        if ring.len() < prebuffer {
                            for out in data.iter_mut() {
                                *out = silence;
                            }
                            return;
                        }
                        started = true;
                    }
                    for frame in data.chunks_mut(channels) {
                        let Some(next) = ring.pop_front() else {
                            started = false;
                            for out in frame.iter_mut() {
                                *out = silence;
                            }
                            continue;
                        };
                        let sample: $t = $conv(next);
                        for out in frame.iter_mut() {
                            *out = sample;
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
    Ok((stream, ring, rate))
}
