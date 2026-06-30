//! Live microphone speech-to-speech — the real end-to-end test: you, a mic, and
//! a speaker. Headless port of `liquid_audio/demo/chat.py`'s real-time loop
//! (no Gradio/WebRTC).
//!
//! Each turn: capture a spoken utterance from the default mic (energy VAD),
//! prefill it, then `generate_interleaved` while STREAMING the reply — text to
//! stdout token-by-token, and each generated 8-code audio frame decoded the
//! instant it is produced via moshi's streaming Mimi `decode_step`, resampled to
//! the speaker rate, and pushed straight to the output device. Chunked and
//! real-time — never batched into one WAV.
//!
//! Run on the Apple GPU (real-time):
//!   LFM_MODEL_DIR=../model LFM_DEVICE=metal \
//!     cargo run --release --features metal --example mic_chat
//! Run on CPU (works, but well below real-time for a 1.5B model):
//!   LFM_MODEL_DIR=../model cargo run --release --example mic_chat
//!
//! Knobs: LFM_VAD_THRESHOLD (default 0.012), LFM_MAX_TOKENS (default 512),
//! LFM_SEED. No torch; Mimi audio-out is the moshi crate.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{DType, Device};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use liquid_audio::{
    from_pretrained, GenParams, Lfm2VoiceEngine, Utterance, VoiceEngine, VoiceEvent,
};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn select_device() -> Res<(Device, DType)> {
    match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                Ok((Device::new_metal(0)?, DType::BF16))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("LFM_DEVICE=metal needs a build with `--features metal`".into())
            }
        }
        _ => Ok((Device::Cpu, DType::F32)),
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

/// Capture one spoken utterance with energy VAD: append all mic audio to a shared
/// buffer; in the main thread, start once a window crosses the threshold and stop
/// after ~0.8 s of silence (or a 30 s cap). Returns mono f32 + the input rate.
fn record_utterance() -> Res<(Vec<f32>, u32)> {
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
                    let mono = downmix(&f, channels);
                    buf.lock().unwrap().extend_from_slice(&mono);
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

    let thr: f32 = std::env::var("LFM_VAD_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.012);
    let window = (rate as usize / 5).max(1); // 200 ms
    let max_samples = rate as usize * 30; // 30 s cap
    let silence_stop = Duration::from_millis(800);

    eprintln!("[mic] listening… speak now (a short pause ends your turn)");
    let mut started = false;
    let mut last_voice = Instant::now();
    let mut read_pos = 0usize;
    loop {
        std::thread::sleep(Duration::from_millis(60));
        let len = {
            let b = buf.lock().unwrap();
            let n = b.len();
            // scan new audio in windows for voice activity
            while read_pos + window <= n {
                let r = rms(&b[read_pos..read_pos + window]);
                if r > thr {
                    started = true;
                    last_voice = Instant::now();
                }
                read_pos += window;
            }
            n
        };
        if started && last_voice.elapsed() >= silence_stop {
            break;
        }
        if len >= max_samples {
            break;
        }
    }
    drop(stream);
    let out = buf.lock().unwrap().clone();
    Ok((out, rate))
}

/// Persistent speaker: an output stream draining a shared ring buffer (mono f32,
/// fanned to every channel). The generate loop pushes decoded chunks into it.
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
                        let s = q.pop_front().unwrap_or(0.0);
                        let v: $t = $conv(s);
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
    // Model source: a HF repo id is snapshot-downloaded to the HF cache (nothing
    // in the source tree); a local path passes through. Override via LFM_MODEL.
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
    let (device, dtype) = select_device()?;

    eprintln!("[load] resolving model `{model_ref}` (repo id → HF cache download, or local path)…");
    let dir = liquid_audio::get_model_dir(&model_ref, None)?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"]
        .as_u64()
        .ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!(
        "[load] LFM2.5-Audio from {} ({dtype:?}, {device:?})…",
        dir.display()
    );
    let t0 = Instant::now();
    let (model, proc) = from_pretrained(&dir, dtype, &device)?;
    eprintln!("[load] done in {:.1}s.", t0.elapsed().as_secs_f32());

    // Held for the program's lifetime so the output stream keeps draining the ring.
    let (_out_stream, ring, out_rate) = start_output()?;
    eprintln!("[spk] output @ {out_rate} Hz");

    // README interleaved-chat params: greedy text, sampled audio (temp 1.0, top-k 4).
    let params = GenParams {
        max_new_tokens,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed,
    };

    // Drive the real `Lfm2VoiceEngine` (the SAME engine the Tauri voice path uses). It owns the
    // model + processor and PERSISTS the conversation across turns: each `respond` seeds from the
    // accumulated `conv` — the prior user audio-in AND the assistant's prior discrete audio-out —
    // generates, decodes audio to PCM at the speaker rate, then appends the new turn back. So this
    // is a real multi-turn conversation WITH MEMORY, not a fresh `ChatState` per turn. Turn-based
    // (we never listen while it speaks), so there is no barge-in and open speakers are fine.
    let mut engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
    let cancel = AtomicBool::new(false);

    println!("\nLFM2.5-Audio live speech-to-speech (with conversation memory). Speak when prompted; Ctrl-C to quit.\n");
    loop {
        let (utt, in_rate) = record_utterance()?;
        let secs = utt.len() as f32 / in_rate as f32;
        if secs < 0.3 {
            eprintln!("[turn] {secs:.2}s — too short, listening again.");
            continue;
        }
        eprintln!("[turn] captured {secs:.2}s @ {in_rate} Hz; generating…");

        print!("assistant: ");
        std::io::stdout().flush().ok();
        cancel.store(false, Ordering::SeqCst);
        let tg = Instant::now();
        let utterance = Utterance {
            samples: utt,
            rate: in_rate,
        };
        // The engine streams the reply: text fragments to stdout, decoded PCM (already at the
        // speaker rate) into the playback ring. On clean completion it appends this turn to `conv`.
        let res = engine.respond(&utterance, &cancel, &mut |ev| match ev {
            VoiceEvent::Text(t) => {
                print!("{t}");
                std::io::stdout().flush().ok();
            }
            VoiceEvent::Audio(pcm) => {
                ring.lock().unwrap().extend(pcm);
            }
            // `respond`'s callback only emits Text/Audio; these terminal variants come from the
            // `RealtimePipeline` wrapper, not here — handled for match exhaustiveness.
            VoiceEvent::TurnComplete | VoiceEvent::Interrupted | VoiceEvent::Error(_) => {}
        });
        println!(); // terminate the streamed "assistant: …" line
        if let Err(e) = res {
            return Err(e.into());
        }
        eprintln!("[turn] done in {:.1}s", tg.elapsed().as_secs_f32());

        // let the speaker finish draining the reply
        loop {
            if ring.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    // The loop only exits by `?` (error) or Ctrl-C; `_out_stream` lives until then.
}
