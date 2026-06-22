//! Headless end-to-end example — the real test that the assembled port works.
//!
//! Faithful headless port of `liquid_audio/demo/chat.py` (the `chat_response` /
//! `chat_producer` generate loop) minus the Gradio/WebRTC UI: load the real
//! LFM2.5-Audio model + processor, feed a spoken question, greedily generate an
//! interleaved text+audio reply, print the text, and write the reply audio to a
//! WAV. Mimi audio-out is the `moshi` crate (already wired behind the processor's
//! `decode`), not reimplemented here.
//!
//! Run (CPU, f32 — the on-disk bf16 weights upcast losslessly):
//!   LFM_MODEL_DIR=../model \
//!   cargo run --release --example generate -- ../upstream-liquid-audio/assets/question.wav
//!
//! Determinism: greedy (no temperature/top-k), seed fixed — same input → same output.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use liquid_audio::{from_pretrained, ChatState, GenParams, GenToken};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Minimal PCM16 WAV reader (mono-downmixed f32 in [-1, 1]); returns (samples, sample_rate).
/// soundfile/symphonia would handle every container, but the assets are plain PCM16 WAV.
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
        pos = end + (sz & 1); // chunks are word-aligned
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

/// Minimal PCM16 mono WAV writer.
fn write_wav_mono_f32(path: &Path, samples: &[f32], rate: u32) -> Res<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn main() -> Res<()> {
    let model_dir = std::env::var("LFM_MODEL_DIR").unwrap_or_else(|_| "../model".into());
    let audio_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../upstream-liquid-audio/assets/question.wav".into());
    let max_new_tokens: usize = std::env::var("LFM_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(96);
    let device = Device::Cpu;

    // `codebooks` is a config field (Python LFM2AudioConfig); ChatState needs it.
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(Path::new(&model_dir).join("config.json"))?)?;
    let codebooks = cfg["codebooks"].as_u64().ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!("[load] model + processor from {model_dir} (f32, CPU)…");
    let (model, proc) = from_pretrained(Path::new(&model_dir), DType::F32, &device)?;
    eprintln!("[load] done.");

    let (samples, rate) = read_wav_mono_f32(Path::new(&audio_path))?;
    eprintln!("[input] {} samples @ {rate} Hz ({:.2}s) from {audio_path}", samples.len(), samples.len() as f32 / rate as f32);
    let n = samples.len();
    let wave = Tensor::from_vec(samples, (1, n), &device)?;

    // chat: system instruction + user audio turn + open assistant turn (chat.py).
    let mut chat = ChatState::new(&proc, codebooks)?;
    chat.new_turn("system")?;
    chat.add_text("Respond with interleaved text and audio.")?;
    chat.end_turn()?;
    chat.new_turn("user")?;
    chat.add_audio(&wave, rate)?;
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    let params = GenParams {
        max_new_tokens,
        text_temperature: None, // greedy → deterministic
        text_top_k: None,
        audio_temperature: None,
        audio_top_k: None,
        seed: 0,
    };

    eprintln!("[gen] generate_interleaved (greedy, max {max_new_tokens} tokens)…");
    let mut text_ids: Vec<u32> = Vec::new();
    let mut audio_frames: Vec<Vec<u32>> = Vec::new();
    model.generate_interleaved(&chat, &params, |tok| match tok {
        GenToken::Text(id) => text_ids.push(id),
        GenToken::Audio(frame) => audio_frames.push(frame),
    })?;

    // --- text reply ---
    let text = proc.text().decode(&text_ids, true)?;
    println!("\n=== TEXT REPLY ({} tokens) ===\n{text}\n", text_ids.len());

    // --- audio reply: drop EOAudio (2048) terminator frames, stack (1, C, frames), Mimi-decode ---
    let frames: Vec<&Vec<u32>> = audio_frames.iter().filter(|f| !f.contains(&2048)).collect();
    eprintln!("[audio] {} frames generated, {} after stripping EOAudio", audio_frames.len(), frames.len());
    if frames.is_empty() {
        println!("=== AUDIO REPLY ===\n(no audio frames generated)");
        return Ok(());
    }
    let nf = frames.len();
    let mut flat = Vec::with_capacity(codebooks * nf);
    for c in 0..codebooks {
        for f in &frames {
            flat.push(f[c]);
        }
    }
    let codes = Tensor::from_vec(flat, (1, codebooks, nf), &device)?;
    let wav = proc.decode(&codes)?; // moshi Mimi codec (proven by mimi_decode_smoke)
    let wav_v = wav.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let out_rate = proc.mimi_sample_rate().unwrap_or(24_000);
    write_wav_mono_f32(Path::new("out.wav"), &wav_v, out_rate)?;
    println!(
        "=== AUDIO REPLY ===\n{} frames → {} samples @ {out_rate} Hz ({:.2}s) → out.wav",
        nf,
        wav_v.len(),
        wav_v.len() as f32 / out_rate as f32
    );
    Ok(())
}
