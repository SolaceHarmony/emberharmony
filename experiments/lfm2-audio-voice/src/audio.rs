//! Native mic capture + speaker playback via cpal — the same I/O approach as the
//! rest of the native voice code, minus LiveKit. Captures an utterance to a 16 kHz
//! mono WAV (what llama-liquid-audio-cli wants) and plays WAVs it produces.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const ASR_RATE: u32 = 16_000; // LFM2.5-Audio / FastConformer encoder input rate

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Linear resample of mono f32. Fine for speech; matches the native client's approach.
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let last = input.len() - 1;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = src - idx as f64;
        let a = input[idx.min(last)] as f64;
        let b = input[(idx + 1).min(last)] as f64;
        out.push((a + (b - a) * frac) as f32);
    }
    out
}

/// Record from the default mic, starting at first speech and ending after a pause.
/// Writes 16 kHz mono PCM to `out_wav`. Returns false if nothing was heard.
pub fn record_utterance(out_wav: &Path) -> Result<bool> {
    let silence = Duration::from_secs_f32(env_f32("LFM_SILENCE_SEC", 1.0));
    let max = Duration::from_secs_f32(env_f32("LFM_MAX_SEC", 20.0));
    let threshold = env_f32("LFM_RMS_THRESHOLD", 0.012);

    let host = cpal::default_host();
    let device = host.default_input_device().context("no input (microphone) device")?;
    let supported = device.default_input_config().context("no default input config")?;
    let dev_rate = supported.sample_rate().0;
    let dev_ch = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.clone().into();

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let started = Arc::new(AtomicBool::new(false));
    let last_voice = Arc::new(Mutex::new(Instant::now()));

    let push = {
        let buf = buf.clone();
        let started = started.clone();
        let last_voice = last_voice.clone();
        move |mono: Vec<f32>| {
            let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len().max(1) as f32).sqrt();
            if rms >= threshold {
                started.store(true, Ordering::Relaxed);
                *last_voice.lock().unwrap() = Instant::now();
            }
            if started.load(Ordering::Relaxed) {
                buf.lock().unwrap().extend(mono);
            }
        }
    };

    let err_fn = |e| eprintln!("[lfm-voice] input stream error: {e}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                let mono: Vec<f32> = data.chunks(dev_ch).map(|c| c.iter().copied().sum::<f32>() / dev_ch as f32).collect();
                push(mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &_| {
                let mono: Vec<f32> = data
                    .chunks(dev_ch)
                    .map(|c| c.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / dev_ch as f32)
                    .collect();
                push(mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _: &_| {
                // cpal U16 is unsigned with silence at 32768; center then normalize.
                let mono: Vec<f32> = data
                    .chunks(dev_ch)
                    .map(|c| c.iter().map(|&s| (s as f32 - 32768.0) / 32768.0).sum::<f32>() / dev_ch as f32)
                    .collect();
                push(mono);
            },
            err_fn,
            None,
        )?,
        other => return Err(anyhow!("unsupported input sample format: {other:?}")),
    };

    print!("  …listening (speak)…");
    use std::io::Write;
    std::io::stdout().flush().ok();
    stream.play()?;

    let begin = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(50));
        if begin.elapsed() >= max {
            break;
        }
        if started.load(Ordering::Relaxed) && last_voice.lock().unwrap().elapsed() >= silence {
            break;
        }
    }
    drop(stream);
    println!(" done.");

    if !started.load(Ordering::Relaxed) {
        return Ok(false);
    }
    let mono = buf.lock().unwrap().clone();
    let mono16 = resample(&mono, dev_rate, ASR_RATE);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: ASR_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(out_wav, spec).context("create wav")?;
    for s in mono16 {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    Ok(true)
}

/// Play a WAV (any rate/channels) on the default output device.
pub fn play_wav(path: &Path) -> Result<()> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("open wav {}", path.display()))?;
    let spec = reader.spec();
    let wav_ch = spec.channels as usize;
    let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / scale).collect(),
    };
    // downmix to mono
    let mono: Vec<f32> = interleaved.chunks(wav_ch).map(|c| c.iter().copied().sum::<f32>() / wav_ch as f32).collect();

    let host = cpal::default_host();
    let device = host.default_output_device().context("no output (speaker) device")?;
    let supported = device.default_output_config().context("no default output config")?;
    let out_rate = supported.sample_rate().0;
    let out_ch = supported.channels() as usize;
    let config: cpal::StreamConfig = supported.clone().into();

    let samples = resample(&mono, spec.sample_rate, out_rate);
    let queue: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(samples.into_iter().collect()));
    let done = Arc::new(AtomicBool::new(false));

    let q = queue.clone();
    let d = done.clone();
    let err_fn = |e| eprintln!("[lfm-voice] output stream error: {e}");
    let fill = move |out: &mut [f32]| {
        let mut q = q.lock().unwrap();
        for frame in out.chunks_mut(out_ch) {
            let s = q.pop_front().unwrap_or(0.0);
            for o in frame.iter_mut() {
                *o = s;
            }
        }
        if q.is_empty() {
            d.store(true, Ordering::Relaxed);
        }
    };

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config,
            move |out: &mut [f32], _: &_| fill(out),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let q = queue.clone();
            let d = done.clone();
            device.build_output_stream(
                &config,
                move |out: &mut [i16], _: &_| {
                    let mut q = q.lock().unwrap();
                    for frame in out.chunks_mut(out_ch) {
                        let s = q.pop_front().unwrap_or(0.0);
                        for o in frame.iter_mut() {
                            *o = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        }
                    }
                    if q.is_empty() {
                        d.store(true, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let q = queue.clone();
            let d = done.clone();
            device.build_output_stream(
                &config,
                move |out: &mut [u16], _: &_| {
                    let mut q = q.lock().unwrap();
                    for frame in out.chunks_mut(out_ch) {
                        let s = q.pop_front().unwrap_or(0.0);
                        // f32 [-1,1] → U16 centered at 32768 (silence == 0.0 → 32768).
                        let v = ((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i32 + 32768).clamp(0, u16::MAX as i32) as u16;
                        for o in frame.iter_mut() {
                            *o = v;
                        }
                    }
                    if q.is_empty() {
                        d.store(true, Ordering::Relaxed);
                    }
                },
                err_fn,
                None,
            )?
        }
        other => return Err(anyhow!("unsupported output sample format: {other:?}")),
    };

    stream.play()?;
    // wait until drained, plus a short tail so the last buffer flushes
    while !done.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(150));
    drop(stream);
    Ok(())
}
