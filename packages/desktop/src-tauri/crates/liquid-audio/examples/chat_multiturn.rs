//! Canonical **multi-turn** proof — the first run that exercises the discrete `audio_out` →
//! context path end-to-end (`ChatState::append` + the prefill `audio_out` scatter).
//!
//! 1:1 port of the README's two-turn getting-started example
//! (`upstream-liquid-audio/README.md`), the real use of the model:
//!
//!   Turn 1: spoken question (`assets/question.wav`, CONTINUOUS audio-in via the Conformer) →
//!           `generate_interleaved` → collect text + DISCRETE audio_out frames + their
//!           interleaved modality flags → Mimi-decode `answer1.wav` (drop the last EOAudio
//!           frame) → `chat.append(text, audio_out, modality_flag)` (FULL audio_out, incl.
//!           EOAudio) → `end_turn`.
//!   Turn 2: a TEXT follow-up ("…chairs…") → `generate_interleaved`. Turn 2's prefill scatters
//!           the appended discrete `audio_out` back in as context, so the reply is CONDITIONED
//!           on turn 1's spoken answer (canonical text: "Comfortable Chairs, Crafted with
//!           Care…") → `answer2.wav`.
//!
//! Single-turn (`examples/generate`) never appends generated audio, so it never exercises this
//! path; this is the example that proves multi-turn conversation works.
//!
//! Run (Apple GPU bf16 — the deployed numerics; needs `--features metal`):
//!   LFM_DEVICE=metal cargo run --release --features metal --example chat_multiturn
//! CPU f32 (the parity reference; slower) is the default with no `LFM_DEVICE`.
//!
//! Writes `answer1.wav` + `answer2.wav` to the working directory.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use liquid_audio::{from_pretrained, ChatState, GenParams, GenToken, LFMModality};

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// `LFM_DEVICE=metal` → Apple GPU at bf16 (the deployed dtype; needs `--features metal`).
/// Otherwise CPU at f32 (candle has no CPU bf16 matmul; f32 loads the bf16 weights losslessly).
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

/// Minimal PCM16 WAV reader (mono-downmixed f32 in [-1, 1]); returns (samples, sample_rate).
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

/// Stack a slice of audio frames into `(1, codebooks, k)` column-major (each frame is a
/// column) for `processor.decode` — the Rust form of `torch.stack(frames, 1).unsqueeze(0)`.
fn frames_to_codes(frames: &[Vec<u32>], codebooks: usize, device: &Device) -> Res<Tensor> {
    let k = frames.len();
    let mut flat = Vec::with_capacity(codebooks * k);
    for c in 0..codebooks {
        for f in frames {
            flat.push(f[c]);
        }
    }
    Ok(Tensor::from_vec(flat, (1, codebooks, k), device)?)
}

fn main() -> Res<()> {
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    let audio_path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../../experiments/lfm2-audio-voice/upstream-liquid-audio/assets/question.wav"
        )
        .into()
    });
    let max_new_tokens: usize =
        std::env::var("LFM_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let (device, dtype) = select_device()?;

    eprintln!("[load] resolving model `{model_ref}`…");
    let dir = liquid_audio::get_model_dir(&model_ref, None)?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"].as_u64().ok_or("config.json: missing `codebooks`")? as usize;

    eprintln!("[load] model + processor from {} ({dtype:?}, {device:?})…", dir.display());
    let t0 = std::time::Instant::now();
    let (model, proc) = from_pretrained(&dir, dtype, &device)?;
    eprintln!("[load] done in {:.1}s.", t0.elapsed().as_secs_f32());

    // README params: text greedy, audio sampled (temp 1.0 / top-k 4). Greedy audio is degenerate.
    let params = GenParams {
        max_new_tokens,
        text_temperature: None,
        text_top_k: None,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed: 0,
    };

    // ONE persistent chat across both turns (exactly the README): the system turn is added once.
    let mut chat = ChatState::new(&proc, codebooks)?;
    chat.new_turn("system")?;
    chat.add_text("Respond with interleaved text and audio.")?;
    chat.end_turn()?;

    // ---------- TURN 1: spoken question → interleaved reply ----------
    let (samples, rate) = read_wav_mono_f32(Path::new(&audio_path))?;
    eprintln!(
        "[turn 1] {} samples @ {rate} Hz ({:.2}s) from {audio_path}",
        samples.len(),
        samples.len() as f32 / rate as f32
    );
    let wave = Tensor::from_vec(samples.clone(), (1, samples.len()), &device)?;
    chat.new_turn("user")?;
    chat.add_audio(&wave, rate)?; // CONTINUOUS audio-in
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    // Collect the SEPARATED streams (text_ids, audio_frames) + the INTERLEAVED modality order.
    let mut text_ids: Vec<u32> = Vec::new();
    let mut audio_frames: Vec<Vec<u32>> = Vec::new();
    let mut modality_out: Vec<i64> = Vec::new();
    let tg = std::time::Instant::now();
    model.generate_interleaved(&chat, &params, |tok| match tok {
        GenToken::Text(id) => {
            text_ids.push(id);
            modality_out.push(LFMModality::Text as i64);
        }
        GenToken::Audio(frame) => {
            audio_frames.push(frame);
            modality_out.push(LFMModality::AudioOut as i64);
        }
    })?;
    let secs = tg.elapsed().as_secs_f32();
    let n_tok = text_ids.len() + audio_frames.len();
    eprintln!("[turn 1] {n_tok} tokens in {secs:.1}s = {:.1} tok/s", n_tok as f32 / secs.max(1e-6));

    let text1 = proc.text().decode(&text_ids, true)?;
    println!("\n=== TURN 1 TEXT ({} tokens) ===\n{text1}\n", text_ids.len());

    // Mimi-decode, dropping the LAST (EOAudio) frame — README `torch.stack(audio_out[:-1], 1)`.
    if audio_frames.len() > 1 {
        let keep = &audio_frames[..audio_frames.len() - 1];
        let codes = frames_to_codes(keep, codebooks, &device)?;
        let wav = proc.decode(&codes)?;
        let wav_v = wav.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let out_rate = proc.mimi_sample_rate().unwrap_or(24_000);
        write_wav_mono_f32(Path::new("answer1.wav"), &wav_v, out_rate)?;
        eprintln!("[turn 1] {} frames → answer1.wav ({:.2}s)", keep.len(), wav_v.len() as f32 / out_rate as f32);
    } else {
        eprintln!("[turn 1] no audio frames generated");
    }

    // Append the generated tokens to history — the discrete audio_out → context path.
    // text (1, n_text), audio_out (codebooks, n_audio) FULL incl. EOAudio, modality (1, n_text+n_audio).
    let n_text = text_ids.len();
    let text_t = Tensor::from_vec(text_ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), (1, n_text), &device)?;
    let n_audio = audio_frames.len();
    let mut aflat = Vec::with_capacity(codebooks * n_audio);
    for c in 0..codebooks {
        for f in &audio_frames {
            aflat.push(f[c] as i64);
        }
    }
    let audio_t = Tensor::from_vec(aflat, (codebooks, n_audio), &device)?;
    let mod_t = Tensor::from_vec(modality_out.clone(), (1, modality_out.len()), &device)?;
    chat.append(&text_t, &audio_t, &mod_t)?;
    chat.end_turn()?;

    // ---------- TURN 2: text follow-up, conditioned on turn 1's spoken audio ----------
    chat.new_turn("user")?;
    chat.add_text("My business specialized in chairs, can you give me something related to that?")?;
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    let mut text_ids2: Vec<u32> = Vec::new();
    let mut audio_frames2: Vec<Vec<u32>> = Vec::new();
    let tg2 = std::time::Instant::now();
    model.generate_interleaved(&chat, &params, |tok| match tok {
        GenToken::Text(id) => text_ids2.push(id),
        GenToken::Audio(frame) => audio_frames2.push(frame),
    })?;
    let secs2 = tg2.elapsed().as_secs_f32();
    let n_tok2 = text_ids2.len() + audio_frames2.len();
    eprintln!("[turn 2] {n_tok2} tokens in {secs2:.1}s = {:.1} tok/s", n_tok2 as f32 / secs2.max(1e-6));

    let text2 = proc.text().decode(&text_ids2, true)?;
    println!("\n=== TURN 2 TEXT ({} tokens) ===\n{text2}\n", text_ids2.len());
    println!("(expect a CHAIRS slogan — proves turn 2 was conditioned on turn 1's appended audio)\n");

    if audio_frames2.len() > 1 {
        let keep = &audio_frames2[..audio_frames2.len() - 1];
        let codes = frames_to_codes(keep, codebooks, &device)?;
        let wav = proc.decode(&codes)?;
        let wav_v = wav.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let out_rate = proc.mimi_sample_rate().unwrap_or(24_000);
        write_wav_mono_f32(Path::new("answer2.wav"), &wav_v, out_rate)?;
        eprintln!("[turn 2] {} frames → answer2.wav ({:.2}s)", keep.len(), wav_v.len() as f32 / out_rate as f32);
    } else {
        eprintln!("[turn 2] no audio frames generated");
    }

    Ok(())
}
