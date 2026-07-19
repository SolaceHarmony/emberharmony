//! Real LFM2-Audio load/inference timing harness.
//!
//! This is intentionally an example binary, not a unit test: it loads the real
//! checkpoint and runs the real interleaved voice path. It prints one JSON object
//! with phase timings and RSS deltas so slow model startup can be measured without
//! guessing from the desktop spinner.
//!
//! Run:
//!   LFM_DEVICE=metal LFM_MAX_TOKENS=64 cargo run --release --features metal --example bench_load_infer
//!   LFM_DEVICE=cpu   LFM_MAX_TOKENS=64 cargo run --release --example bench_load_infer
//!
//! Environment:
//!   LFM_MODEL or LFM_MODEL_DIR   HF repo id or local snapshot path.
//!   LFM_BENCH_AUDIO             Optional PCM16 WAV path.
//!   LFM_BENCH_DECODE_AUDIO=0    Skip batch reply detokenization.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use candle_core::Device;
use liquid_audio_oracle::moshi::demo::chat::decode_audio_reply;
use liquid_audio_oracle::moshi::models::MimiModel;
use liquid_audio_oracle::{from_pretrained, ChatState, GenParams, GenToken};
use serde_json::{json, Value};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

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
            if liquid_audio_oracle::flashkern::native_engine::bf16_gemm_available() {
                Ok(Device::Cpu)
            } else {
                Err("CPU BF16 needs the NEON BFMMLA kernel; use Metal on this Mac".into())
            }
        }
        Some(other) => Err(format!("unknown LFM_DEVICE={other}; use cpu or metal").into()),
    }
}

fn rss_kib() -> Option<u64> {
    let pid = std::process::id().to_string();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn measure<T, F>(metrics: &mut Vec<Value>, phase: &str, detail: Value, f: F) -> Res<T>
where
    F: FnOnce() -> Res<T>,
{
    let rss_before = rss_kib();
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed();
    let rss_after = rss_kib();
    let delta = match (rss_before, rss_after) {
        (Some(a), Some(b)) => Some(b as i64 - a as i64),
        _ => None,
    };
    eprintln!(
        "[bench] {phase}: {:.1} ms, rss {:?} -> {:?} KiB, delta {:?} KiB",
        millis(elapsed),
        rss_before,
        rss_after,
        delta
    );
    let error = out.as_ref().err().map(|e| e.to_string());
    metrics.push(json!({
        "phase": phase,
        "elapsed_ms": millis(elapsed),
        "rss_kib_before": rss_before,
        "rss_kib_after": rss_after,
        "rss_kib_delta": delta,
        "error": error,
        "detail": detail,
    }));
    out
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
        return Err(format!("only PCM16 WAV supported, got {bits}-bit").into());
    }
    let data = data.ok_or("no data chunk")?;
    let ch = channels.max(1) as usize;
    let total = data.len() / 2;
    let mut mono = Vec::with_capacity(total / ch);
    let mut i = 0;
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

fn safetensor_stats(dir: &Path) -> Res<Value> {
    let mut count = 0usize;
    let mut bytes = 0u64;
    for root in [dir.to_path_buf(), dir.join("audio_detokenizer")] {
        if !root.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&root)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
                continue;
            }
            count += 1;
            bytes += std::fs::metadata(&path)?.len();
        }
    }
    Ok(json!({ "safetensors": count, "safetensor_bytes": bytes }))
}

fn main() -> Res<()> {
    let mut metrics = Vec::new();
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    let audio_path = std::env::var("LFM_BENCH_AUDIO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/question.wav"))
        });
    let max_new_tokens = env_usize("LFM_MAX_TOKENS", 64);
    let seed = env_u64("LFM_SEED", 0);
    let decode_audio = std::env::var("LFM_BENCH_DECODE_AUDIO").ok().as_deref() != Some("0");

    let device = measure(&mut metrics, "select_device", json!({}), || select_device())?;
    let dir = measure(
        &mut metrics,
        "resolve_model_dir",
        json!({ "model_ref": model_ref }),
        || Ok(liquid_audio_oracle::get_model_dir(&model_ref, None)?),
    )?;
    let (codebooks, stats) = measure(
        &mut metrics,
        "read_config_and_stat_checkpoint",
        json!({ "model_dir": dir.display().to_string() }),
        || {
            let cfg: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
            let codebooks = cfg["codebooks"]
                .as_u64()
                .ok_or("config.json: missing `codebooks`")? as usize;
            Ok((codebooks, safetensor_stats(&dir)?))
        },
    )?;

    let (model, proc) = measure(
        &mut metrics,
        "load_model_processor",
        json!({ "model_dir": dir.display().to_string(), "checkpoint": stats }),
        || Ok(from_pretrained(&dir, &device)?),
    )?;

    let mimi = MimiModel::new(
        proc.mimi()
            .ok_or("Mimi codec not loaded; needed for audio reply decode")?,
    );

    let (samples, rate) = measure(
        &mut metrics,
        "read_input_wav",
        json!({ "audio_path": audio_path.display().to_string() }),
        || read_wav_mono_f32(&audio_path),
    )?;
    let sample_count = samples.len();

    let chat = measure(
        &mut metrics,
        "build_audio_chat_state",
        json!({ "samples": sample_count, "sample_rate": rate, "codebooks": codebooks }),
        || {
            let mut chat = ChatState::new(&proc, codebooks)?;
            chat.new_turn("system")?;
            chat.add_text("Respond with interleaved text and audio.")?;
            chat.end_turn()?;
            chat.new_turn("user")?;
            chat.add_audio_slice(&samples, rate)?;
            chat.end_turn()?;
            chat.new_turn("assistant")?;
            Ok(chat)
        },
    )?;

    let in_emb = measure(&mut metrics, "prefill_audio_prompt", json!({}), || {
        Ok(model.prefill_chat(&chat)?)
    })?;

    let params = GenParams {
        max_new_tokens,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed,
    };
    let mut text_ids = Vec::new();
    let mut audio_frames = Vec::new();
    let mut first_token_ms: Option<f64> = None;
    let gen_start = Instant::now();
    measure(
        &mut metrics,
        "decode_interleaved",
        json!({ "max_new_tokens": max_new_tokens, "seed": seed }),
        || {
            model.generate_from_embeds(in_emb, &params, |tok| {
                if first_token_ms.is_none() {
                    first_token_ms = Some(millis(gen_start.elapsed()));
                }
                match tok {
                    GenToken::Text(id) => text_ids.push(id),
                    GenToken::Audio(frame) => audio_frames.push(frame),
                }
            })?;
            Ok(())
        },
    )?;
    let decode_elapsed = gen_start.elapsed();
    let total_tokens = text_ids.len() + audio_frames.len();
    let text = measure(
        &mut metrics,
        "decode_text_tokens",
        json!({ "text_tokens": text_ids.len() }),
        || Ok(proc.text().decode(&text_ids, true)?),
    )?;

    let audio_seconds = if decode_audio && !audio_frames.is_empty() {
        measure(
            &mut metrics,
            "decode_audio_reply_batch",
            json!({ "audio_frames": audio_frames.len() }),
            || {
                let Some(wav) = decode_audio_reply(&mimi, &audio_frames, codebooks, &device)?
                else {
                    return Ok(None);
                };
                let len = wav.dim(wav.rank() - 1)?;
                Ok(Some(len as f64 / mimi.sample_rate() as f64))
            },
        )?
    } else {
        None
    };

    let summary = json!({
        "model_ref": model_ref,
        "model_dir": dir.display().to_string(),
        "device": format!("{device:?}"),
        "bf16_gemm_available": liquid_audio_oracle::flashkern::native_engine::bf16_gemm_available(),
        "audio_path": audio_path.display().to_string(),
        "input_seconds": sample_count as f64 / rate as f64,
        "max_new_tokens": max_new_tokens,
        "text_tokens": text_ids.len(),
        "audio_frames": audio_frames.len(),
        "total_tokens": total_tokens,
        "first_token_ms_after_prefill": first_token_ms,
        "decode_ms_after_prefill": millis(decode_elapsed),
        "tokens_per_second_after_prefill": total_tokens as f64 / decode_elapsed.as_secs_f64().max(1e-9),
        "reply_audio_seconds": audio_seconds,
        "reply_text": text,
        "metrics": metrics,
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
