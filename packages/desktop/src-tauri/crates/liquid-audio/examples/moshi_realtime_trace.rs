//! Dump a deterministic trace from the native Rust realtime Moshi loop.
//!
//! Pair with `parity/dump_moshi_realtime.py` and compare the JSON outputs. This
//! harness intentionally exercises the same hot loop the desktop runtime uses:
//! exact PCM frame -> Mimi encode_step -> multistream LM step -> Mimi decode_step.
//!
//! Usage:
//!   MOSHI_GREEDY=1 MOSHI_TRACE_FRAMES=16 MOSHI_WARMUP_FRAMES=4 MOSHI_SEED=42424242 \
//!     cargo run --release --example moshi_realtime_trace -- \
//!       /path/to/moshiko-candle-bf16 /path/to/input-24khz.wav /tmp/rust-moshi.json

use std::{io::Read, path::Path};

use candle_core::Device;
use liquid_audio::moshi::models::{
    load_realtime_moshi_with_warmup, realtime_moshi_files, safetensors_floating_dtype,
    RealtimeMoshiEvent, REALTIME_MOSHI_WARMUP_FRAMES,
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
                Err("LFM_DEVICE=metal needs `--features metal`".into())
            }
        }
        Some("cpu") | None => Ok(Device::Cpu),
        Some(other) => Err(format!("unknown LFM_DEVICE={other}; use cpu or metal").into()),
    }
}

fn read_wav_mono_f32(path: &Path) -> Res<(Vec<f32>, u32)> {
    let b = std::fs::read(path)?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()).into());
    }
    let mut pos = 12usize;
    let (mut rate, mut channels, mut bits) = (0u32, 1u16, 16u16);
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let sz = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        let end = (body + sz).min(b.len());
        match id {
            b"fmt " if body + 16 <= b.len() => {
                channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
                rate = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
                bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
            }
            b"data" => data = Some(&b[body..end]),
            _ => {}
        }
        pos = end + (sz & 1);
    }
    if bits != 16 {
        return Err(format!("only PCM16 WAV is supported, got {bits}-bit").into());
    }
    let data = data.ok_or("no data chunk")?;
    let ch = channels.max(1) as usize;
    let total = data.len() / 2;
    let mut mono = Vec::with_capacity(total / ch);
    let mut i = 0usize;
    while i + ch <= total {
        let mut acc = 0f32;
        for c in 0..ch {
            acc += i16::from_le_bytes([data[(i + c) * 2], data[(i + c) * 2 + 1]]) as f32 / 32768.0;
        }
        mono.push(acc / ch as f32);
        i += ch;
    }
    Ok((mono, rate))
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|v| v * v).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn file_fingerprint(path: &Path) -> Res<serde_json::Value> {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 1024 * 1024];
    let mut hash = OFFSET;
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        for b in &buf[..n] {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    Ok(serde_json::json!({
        "path": path.display().to_string(),
        "bytes": bytes,
        "fnv1a64": format!("{hash:016x}"),
    }))
}

fn main() -> Res<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            "usage: moshi_realtime_trace <moshi_model_dir> <input-24khz.wav> <out.json>".into(),
        );
    }
    let model = Path::new(&args[0]);
    let wav = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let max_frames = std::env::var("MOSHI_TRACE_FRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let greedy = std::env::var("MOSHI_GREEDY")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let warmup_frames = std::env::var("MOSHI_WARMUP_FRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(REALTIME_MOSHI_WARMUP_FRAMES);
    let seed = std::env::var("MOSHI_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());

    let files = realtime_moshi_files(model)?.ok_or("selected directory is not a Moshi snapshot")?;
    let mut params = files.params.with_seed(seed.unwrap_or(files.params.seed));
    if greedy {
        params.use_sampling = false;
    }
    let dtype = safetensors_floating_dtype(&files.moshi_weights)?;
    let dtype_name = format!("{dtype:?}").to_lowercase();
    let device = select_device()?;
    let mut realtime = load_realtime_moshi_with_warmup(
        files
            .moshi_weights
            .to_str()
            .ok_or("Moshi checkpoint path is not UTF-8")?,
        files
            .mimi_weights
            .to_str()
            .ok_or("Mimi checkpoint path is not UTF-8")?,
        dtype,
        &device,
        params,
        warmup_frames,
    )?;

    let (samples, rate) = read_wav_mono_f32(wav)?;
    if rate != realtime.sample_rate() {
        return Err(format!(
            "input WAV must already be {} Hz for strict parity, got {rate}",
            realtime.sample_rate()
        )
        .into());
    }

    let mut text = Vec::<u32>::new();
    let mut input_audio_tokens = Vec::<Vec<u32>>::new();
    let mut audio_tokens = Vec::<Vec<u32>>::new();
    let mut audio = Vec::<serde_json::Value>::new();
    let mut frames = 0usize;
    for chunk in samples
        .chunks(realtime.frame_size())
        .filter(|chunk| chunk.len() == realtime.frame_size())
    {
        if frames >= max_frames {
            break;
        }
        frames += 1;
        for event in realtime.step_pcm_frame(chunk)? {
            match event {
                RealtimeMoshiEvent::InputAudioTokenFrame(codes) => input_audio_tokens.push(codes),
                RealtimeMoshiEvent::TextToken(token) => text.push(token),
                RealtimeMoshiEvent::AudioTokenFrame(codes) => audio_tokens.push(codes),
                RealtimeMoshiEvent::Audio { pcm, rate } => audio.push(serde_json::json!({
                    "samples": pcm.len(),
                    "rate": rate,
                    "rms": rms(&pcm),
                    "first": pcm.first().copied().unwrap_or(0.0),
                })),
            }
        }
    }

    let trace = serde_json::json!({
        "source": "rust",
        "mode": "step",
        "model_dir": model.display().to_string(),
        "model_type": files.model_type,
        "checkpoint": {
            "moshi": file_fingerprint(&files.moshi_weights)?,
            "mimi": file_fingerprint(&files.mimi_weights)?,
            "tokenizer": file_fingerprint(&files.tokenizer)?,
        },
        "input": wav.display().to_string(),
        "greedy": greedy,
        "seed": params.seed,
        "dtype": dtype_name,
        "cfg_coef": 1.0,
        "generation": {
            "use_sampling": params.use_sampling,
            "temp": params.audio_temperature,
            "temp_text": params.text_temperature,
            "top_k": params.audio_top_k,
            "top_k_text": params.text_top_k,
            "cfg_coef": 1.0,
        },
        "sample_rate": realtime.sample_rate(),
        "frame_size": realtime.frame_size(),
        "warmup_frames": warmup_frames,
        "input_frames": frames,
        "input_audio_tokens": input_audio_tokens,
        "text_tokens": text,
        "audio_tokens": audio_tokens,
        "audio_chunks": audio,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&trace)?)?;
    Ok(())
}
