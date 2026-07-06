//! Full-duplex live speech-to-speech on the [`RealtimePipeline`] worker thread.
//!
//! The difference from `mic_chat` (half-duplex: the mic is dropped while the model
//! generates, `can_interrupt=False`) is that here the **mic stays live the whole time**
//! and the model runs on a dedicated inference worker thread — the structure of
//! `moshi/server.py`'s recv/inference/send split:
//!
//! - a **continuous** input stream appends mono mic audio to a shared buffer (never torn
//!   down between turns);
//! - the **main thread** runs energy VAD over that buffer to find utterance boundaries and,
//!   crucially, to detect the user speaking again *while the assistant is talking* — which
//!   triggers **barge-in** (`pipe.interrupt()` + flush the speaker) and starts a new turn;
//! - the [`RealtimePipeline`] **worker thread** owns the model + detokenizer and turns each
//!   submitted utterance into a stream of [`VoiceEvent`]s;
//! - a **consumer thread** drains those events: text to stdout, PCM to the speaker ring.
//!
//! Caveat: there is no acoustic echo cancellation here, so the assistant's own voice from
//! the speakers can re-trigger the mic VAD — use headphones, or raise `LFM_VAD_THRESHOLD`.
//!
//! Run (Apple GPU, real-time):
//!   LFM_MODEL_DIR=../model LFM_DEVICE=metal \
//!     cargo run --release --features metal --example duplex_chat
//!
//! Knobs: LFM_VAD_THRESHOLD (default 0.012), LFM_MAX_TOKENS (default 512), LFM_SEED.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::Device;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use liquid_audio::{
    from_pretrained, GenParams, Lfm2VoiceEngine, RealtimePipeline, Utterance, VoiceEvent,
};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn select_device() -> Res<Device> {
    match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("LFM_DEVICE=metal needs a build with `--features metal`".into())
            }
        }
        Some("cpu") | None => {
            if liquid_audio::bf16_gemm::bf16_gemm_available() {
                Ok(Device::Cpu)
            } else {
                Err("CPU BF16 needs the NEON BFMMLA kernel; use Metal on this Mac".into())
            }
        }
        Some(other) => Err(format!("unknown LFM_DEVICE={other}; use cpu or metal").into()),
    }
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|c| c.iter().sum::<f32>() / channels as f32)
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Continuous input: an always-on stream appending mono f32 mic audio to a shared buffer.
fn start_input() -> Res<(cpal::Stream, Arc<Mutex<Vec<f32>>>, u32)> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or("no default input device")?;
    let supported = dev.default_input_config()?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let err_fn = |e| eprintln!("[mic] input stream error: {e}");

    macro_rules! input_stream {
        ($t:ty, $conv:expr) => {{
            let buf = buf.clone();
            dev.build_input_stream(
                &cfg,
                move |d: &[$t], _: &cpal::InputCallbackInfo| {
                    let f: Vec<f32> = d.iter().map(|&s| $conv(s)).collect();
                    buf.lock()
                        .unwrap()
                        .extend_from_slice(&downmix(&f, channels));
                },
                err_fn,
                None,
            )?
        }};
    }
    let stream = match fmt {
        cpal::SampleFormat::F32 => input_stream!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => input_stream!(i16, |s: i16| s as f32 / 32768.0),
        cpal::SampleFormat::U16 => input_stream!(u16, |s: u16| (s as f32 - 32768.0) / 32768.0),
        other => return Err(format!("unsupported input sample format {other:?}").into()),
    };
    stream.play()?;
    Ok((stream, buf, rate))
}

/// Persistent speaker: an output stream draining a shared ring (mono f32, fanned to every
/// channel). Returns the ring so the consumer can push PCM and flush on barge-in.
fn start_output() -> Res<(cpal::Stream, Arc<Mutex<VecDeque<f32>>>, u32)> {
    let host = cpal::default_host();
    let dev = host
        .default_output_device()
        .ok_or("no default output device")?;
    let supported = dev.default_output_config()?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let cfg: cpal::StreamConfig = supported.into();
    let ring: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let err_fn = |e| eprintln!("[spk] output stream error: {e}");

    macro_rules! output_stream {
        ($t:ty, $conv:expr) => {{
            let ring = ring.clone();
            dev.build_output_stream(
                &cfg,
                move |d: &mut [$t], _: &cpal::OutputCallbackInfo| {
                    let mut q = ring.lock().unwrap();
                    for frame in d.chunks_mut(channels) {
                        let v: $t = $conv(q.pop_front().unwrap_or(0.0));
                        for x in frame.iter_mut() {
                            *x = v;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }};
    }
    let stream = match fmt {
        cpal::SampleFormat::F32 => output_stream!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => {
            output_stream!(i16, |s: f32| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        }
        cpal::SampleFormat::U16 => output_stream!(u16, |s: f32| ((s.clamp(-1.0, 1.0) * 32767.0)
            as i32
            + 32768) as u16),
        other => return Err(format!("unsupported output sample format {other:?}").into()),
    };
    stream.play()?;
    Ok((stream, ring, rate))
}

fn main() -> Res<()> {
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    let max_new_tokens: usize = std::env::var("LFM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let seed: u64 = std::env::var("LFM_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device = select_device()?;

    eprintln!("[load] resolving model `{model_ref}`…");
    let dir = liquid_audio::get_model_dir(&model_ref, None)?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"]
        .as_u64()
        .ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!(
        "[load] LFM2.5-Audio from {} (safetensor dtype, {device:?})…",
        dir.display()
    );
    let t0 = Instant::now();
    let (model, proc) = from_pretrained(&dir, &device)?;
    eprintln!("[load] done in {:.1}s.", t0.elapsed().as_secs_f32());

    let (_out_stream, ring, out_rate) = start_output()?;
    let (_in_stream, mic, in_rate) = start_input()?;
    eprintln!("[io] mic @ {in_rate} Hz (live), speaker @ {out_rate} Hz");

    // The inference worker thread owns the model + processor; it resamples PCM to the
    // speaker rate so the consumer can push chunks straight into the ring.
    let params = GenParams {
        max_new_tokens,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed,
    };
    let engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
    let pipe = RealtimePipeline::spawn(engine)?;

    // `assistant_active` is true between the first reply event and the turn terminal — the
    // capture loop reads it to decide whether a fresh voice onset is a barge-in.
    let assistant_active = Arc::new(AtomicBool::new(false));

    // Consumer thread: drain reply events → speaker ring + stdout.
    let consumer = {
        let events = pipe.events().clone();
        let ring = ring.clone();
        let active = assistant_active.clone();
        std::thread::spawn(move || {
            for ev in events.iter() {
                match ev {
                    VoiceEvent::Text(t) => {
                        active.store(true, Ordering::SeqCst);
                        print!("{t}");
                        std::io::stdout().flush().ok();
                    }
                    VoiceEvent::Audio(pcm) => {
                        active.store(true, Ordering::SeqCst);
                        ring.lock().unwrap().extend(pcm);
                    }
                    VoiceEvent::TurnComplete | VoiceEvent::Interrupted => {
                        active.store(false, Ordering::SeqCst);
                        println!();
                    }
                    VoiceEvent::Error(e) => {
                        active.store(false, Ordering::SeqCst);
                        eprintln!("\n[engine] error: {e}");
                    }
                }
            }
        })
    };

    let thr: f32 = std::env::var("LFM_VAD_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.012);
    let window = (in_rate as usize / 5).max(1); // 200 ms VAD window
    let silence_stop = Duration::from_millis(800);

    println!("\nLFM2.5-Audio FULL-DUPLEX speech-to-speech. Speak any time (barge-in enabled); Ctrl-C to quit.\n");

    // VAD state machine over the live mic buffer. `utt_start` is the sample index (within
    // the current buffer) where the in-progress utterance began.
    let mut speaking = false;
    let mut utt_start = 0usize;
    let mut read_pos = 0usize;
    let mut last_voice = Instant::now();

    loop {
        std::thread::sleep(Duration::from_millis(40));
        let mut mic_buf = mic.lock().unwrap();
        let n = mic_buf.len();
        while read_pos + window <= n {
            let voiced = rms(&mic_buf[read_pos..read_pos + window]) > thr;
            if voiced {
                if !speaking {
                    // Voice onset. If the assistant is mid-reply, this is barge-in.
                    if assistant_active.load(Ordering::SeqCst) {
                        pipe.interrupt();
                        ring.lock().unwrap().clear(); // stop playback immediately
                        eprintln!("\n[barge-in] interrupting reply");
                    }
                    speaking = true;
                    utt_start = read_pos;
                }
                last_voice = Instant::now();
            }
            read_pos += window;
        }

        if speaking && last_voice.elapsed() >= silence_stop {
            // Utterance ended — slice it out, submit to the worker, and reset the buffer.
            let samples: Vec<f32> = mic_buf[utt_start..read_pos.min(n)].to_vec();
            mic_buf.clear();
            read_pos = 0;
            speaking = false;
            drop(mic_buf);

            let secs = samples.len() as f32 / in_rate as f32;
            if secs < 0.3 {
                continue; // too short — ignore
            }
            eprintln!("[turn] captured {secs:.2}s; generating…");
            print!("assistant: ");
            std::io::stdout().flush().ok();
            if !pipe.submit(Utterance {
                samples,
                rate: in_rate,
            }) {
                break; // worker gone
            }
        } else if !speaking && n > in_rate as usize * 5 {
            // Idle: trim the buffer so continuous capture doesn't grow without bound.
            mic_buf.clear();
            read_pos = 0;
        }
    }

    drop(pipe); // close channels + join the worker
    let _ = consumer.join();
    let _ = (_in_stream, _out_stream);
    Ok(())
}
