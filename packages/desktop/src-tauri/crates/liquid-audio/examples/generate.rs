//! Headless end-to-end example — the real test that the assembled port works.
//!
//! Faithful headless port of `liquid_audio/demo/chat.py` (the `chat_response` /
//! `chat_producer` generate loop) minus the Gradio/WebRTC UI: load the real
//! LFM2.5-Audio model + processor, feed a spoken question, greedily generate an
//! interleaved text+audio reply, print the text, and write the reply audio to a
//! WAV. Mimi audio-out goes through the same `proc.mimi` interface as the Python
//! demo, not through the generic processor detokenizer dispatch.
//!
//! Run (CPU BF16 via the in-tree NEON kernel):
//!   LFM_MODEL_DIR=../model cargo run --release --example generate -- path/to/audio.wav
//! Use `LFM_DEVICE=metal` for Apple GPU BF16.
//! With no path argument it defaults to the upstream reference clip
//! (`assets/question.wav`, the upstream reference clip, vendored in-crate), resolved
//! from the manifest dir so it works regardless of the working directory.
//!
//! Determinism: greedy (no temperature/top-k), seed fixed — same input → same output.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use liquid_audio::moshi::demo::chat::decode_audio_reply;
use liquid_audio::moshi::models::MimiModel;
use liquid_audio::{from_pretrained, ChatState, GenParams, GenToken};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// `LFM_DEVICE=metal` → Apple GPU BF16. Default/`cpu` → CPU BF16 through the
/// in-tree NEON kernel.
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
    // Model source: a HF repo id is snapshot-downloaded to the HF cache (nothing
    // lives in the source tree); a local path passes through. Override with
    // LFM_MODEL (or LFM_MODEL_DIR) = repo id or path.
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    let audio_path = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| {
        // The upstream reference clip, vendored in-crate. Resolved from
        // CARGO_MANIFEST_DIR (compile-time, CWD-independent).
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/question.wav").into()
    });
    let max_new_tokens: usize = std::env::var("LFM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    let device = select_device()?;

    eprintln!("[load] resolving model `{model_ref}` (repo id → HF cache download, or local path)…");
    let dir = liquid_audio::get_model_dir(&model_ref, None)?;

    // `codebooks` is a config field (Python LFM2AudioConfig); ChatState needs it.
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"]
        .as_u64()
        .ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!(
        "[load] model + processor from {} (safetensor dtype, {device:?})…",
        dir.display()
    );
    let t0 = std::time::Instant::now();
    #[allow(unused_mut)]
    let (mut model, proc) = from_pretrained(&dir, &device)?;
    // `--reference`: the byte-parity reference chain (DECODE_ENGINE.md §5) — every
    // ulp-tier decode deviation pinned off so the run reproduces the recorded
    // wav-hash baseline bit-for-bit.
    if std::env::args().any(|a| a == "--reference") {
        model.set_reference_numerics(true);
        eprintln!("[mode] reference numerics: grouped-GQA off, depth-flash off");
    }
    let mimi = MimiModel::new(
        proc.mimi()
            .ok_or("Mimi codec not loaded — required by liquid_audio/demo/chat.py")?,
    );
    eprintln!("[load] done in {:.1}s.", t0.elapsed().as_secs_f32());

    let (samples, rate) = read_wav_mono_f32(Path::new(&audio_path))?;
    eprintln!(
        "[input] {} samples @ {rate} Hz ({:.2}s) from {audio_path}",
        samples.len(),
        samples.len() as f32 / rate as f32
    );
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

    // The HF example's exact params: text greedy (no text temperature passed),
    // audio sampled at temperature 1.0 / top-k 4. Greedy audio is degenerate — the model
    // is trained for sampled audio — so this matters for the audio reply being intelligible.
    let params = GenParams {
        max_new_tokens,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed: 0,
    };

    eprintln!(
        "[gen] generate_interleaved (text greedy, audio temp=1.0 top-k=4, max {max_new_tokens})…"
    );
    let mut text_ids: Vec<u32> = Vec::new();
    let mut audio_frames: Vec<Vec<u32>> = Vec::new();
    let tg = std::time::Instant::now();
    model.generate_interleaved(&chat, &params, |tok| match tok {
        GenToken::Text(id) => text_ids.push(id),
        GenToken::Audio(frame) => audio_frames.push(frame),
    })?;
    let n_tok = text_ids.len() + audio_frames.len();
    let secs = tg.elapsed().as_secs_f32();
    eprintln!(
        "[gen] {n_tok} tokens in {secs:.1}s = {:.1} tok/s",
        n_tok as f32 / secs
    );

    // --- text reply ---
    let text = proc.text().decode(&text_ids, true)?;
    println!("\n=== TEXT REPLY ({} tokens) ===\n{text}\n", text_ids.len());

    // --- audio reply: drop EOAudio (2048) terminator frame, stack (1, C, frames), Mimi-decode ---
    eprintln!(
        "[audio] {} frames generated, {} after stripping EOAudio",
        audio_frames.len(),
        audio_frames.len().saturating_sub(1)
    );
    let Some(wav) = decode_audio_reply(&mimi, &audio_frames, codebooks, &device)? else {
        println!("=== AUDIO REPLY ===\n(no audio frames generated)");
        return Ok(());
    };
    let wav_v = wav.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let out_rate = mimi.sample_rate();
    let nf = audio_frames.len() - 1;
    write_wav_mono_f32(Path::new("out.wav"), &wav_v, out_rate)?;
    println!(
        "=== AUDIO REPLY ===\n{} frames → {} samples @ {out_rate} Hz ({:.2}s) → out.wav",
        nf,
        wav_v.len(),
        wav_v.len() as f32 / out_rate as f32
    );
    Ok(())
}
