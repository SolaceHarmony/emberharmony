//! The model talks to ITSELF — two `Lfm2VoiceEngine` instances on the shared
//! resident model, one playing the assistant and one role-playing the human.
//! Each engine hears ONLY the other's generated audio (`Utterance` PCM at the
//! speaker rate — the same path a real mic turn takes), keeps its own
//! cross-turn conversation memory, and the whole exchange plays live on the
//! speaker while it runs.
//!
//! Besides the fun, it's a data harness: every turn logs wall time,
//! respond→first-text and respond→first-audio latency (the pure model-side
//! slice of the pause→first-audio number the runtime reports), tokens/sec, and
//! audio seconds; per-turn WAVs, a stitched conversation WAV, and a
//! transcript.json land in `self_chat_out/`.
//!
//! The conversation is HARD-CAPPED: `LFM_SELF_TURNS` (default 6, clamped to
//! 1..=1000) total replies, so it can never run away.
//!
//! THE 32K SOAK rides this harness (her design: two instances, the output of
//! one is the input of the other, audio only — a conversation, not a
//! rehearsal): set `LFM_SELF_TARGET_CTX` (e.g. 30000) and a high turn cap,
//! and the run stops as soon as EITHER side's conversation context reaches
//! the target — proving the climb toward the model's 32,768 ceiling on the
//! real two-party path. Per-turn budget via `LFM_SELF_MAX_NEW` (default 512
//! both sides — interleaved steps, every audio frame costs one, so 160 was an
//! 8.5 s speech ceiling; the truncation warning is loud if it binds). Text
//! sampling via `LFM_SELF_TEXT_TEMP` / `LFM_SELF_TEXT_TOPK` (0 = greedy / no
//! cutoff; default is the production regime).
//!
//!   LFM_DEVICE=cpu LFM_MODEL_DIR=/path/to/model LFM_SELF_TURNS=400 \
//!     LFM_SELF_TARGET_CTX=30000 \
//!     cargo run --release --features accelerate --example self_chat
//!
//! Run (Apple GPU bf16 — audible):
//!   LFM_DEVICE=metal LFM_MODEL_DIR=/path/to/model \
//!     cargo run --release --features metal --example self_chat

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::Device;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use liquid_audio_oracle::{
    from_pretrained, GenParams, Lfm2VoiceEngine, Utterance, VoiceEngine, VoiceEvent,
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
            if liquid_audio_oracle::flashkern::native_engine::bf16_gemm_available() {
                Ok(Device::Cpu)
            } else {
                Err("CPU BF16 needs the NEON BFMMLA kernel; use Metal on this Mac".into())
            }
        }
        Some(other) => Err(format!("unknown LFM_DEVICE={other}; use cpu or metal").into()),
    }
}

/// Minimal PCM16 WAV reader (mono-downmixed f32 in [-1, 1]); returns (samples, rate).
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
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Persistent speaker: an output stream draining a shared ring (mono f32 fanned to
/// every channel) — same as mic_chat.
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

struct TurnRecord {
    turn: usize,
    role: &'static str,
    transcript: String,
    wall_s: f32,
    first_text_ms: Option<u64>,
    first_audio_ms: Option<u64>,
    audio_s: f32,
}

fn main() -> Res<()> {
    let model_ref = std::env::var("LFM_MODEL")
        .or_else(|_| std::env::var("LFM_MODEL_DIR"))
        .unwrap_or_else(|_| "LiquidAI/LFM2.5-Audio-1.5B".into());
    // The hard cap Sydney asked for: bounded ALWAYS (default stays 6); the
    // 32k soak raises it explicitly — the ceiling exists so no invocation can
    // run away.
    let max_turns: usize = std::env::var("LFM_SELF_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6)
        .clamp(1, 1000);
    // 32k soak: stop as soon as EITHER side's context reaches this.
    let target_ctx: u64 = std::env::var("LFM_SELF_TARGET_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let max_new: usize = std::env::var("LFM_SELF_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let device = select_device()?;

    eprintln!("[load] resolving model `{model_ref}`…");
    let dir = liquid_audio_oracle::get_model_dir(&model_ref, None)?;
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let codebooks = cfg["codebooks"]
        .as_u64()
        .ok_or("config.json: missing `codebooks`")? as usize;

    let t0 = Instant::now();
    let (model, proc) = from_pretrained(&dir, &device)?;
    let (model, proc) = (Arc::new(model), Arc::new(proc));
    eprintln!(
        "[load] done in {:.1}s ({device:?}).",
        t0.elapsed().as_secs_f32()
    );

    let (_out_stream, ring, out_rate) = start_output()?;
    eprintln!("[spk] output @ {out_rate} Hz");

    // Production decoding regime (the app's settings defaults): sampled text at
    // 1.0 (vendor demo), sampled audio 1.0/top-k 4. Different seeds per engine so
    // the two voices don't mirror each other. Text sampling overridable for A/B
    // runs: LFM_SELF_TEXT_TEMP (0 = greedy) and LFM_SELF_TEXT_TOPK (0 = no cutoff).
    let text_temp: f64 = std::env::var("LFM_SELF_TEXT_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let text_top_k: usize = std::env::var("LFM_SELF_TEXT_TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let gen = move |max_new_tokens: usize, seed| GenParams {
        max_new_tokens: if max_new > 0 { max_new } else { max_new_tokens },
        text_temperature: (text_temp > 0.0).then_some(text_temp),
        text_top_k: (text_top_k > 0).then_some(text_top_k),
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        seed,
    };

    // Engine A: the assistant, verbatim production prompt.
    let mut assistant = Lfm2VoiceEngine::new(
        model.clone(),
        proc.clone(),
        gen(512, 42),
        codebooks,
        device.clone(),
        out_rate,
    );
    // Engine B: the "human" — same model role-playing the user side of the call.
    let ctx_a = assistant.context_positions();
    let mut human = Lfm2VoiceEngine::new(model, proc, gen(512, 1337), codebooks, device, out_rate)
        .with_system_prompt(
            "You are a curious person having a casual spoken conversation. React briefly \
             to what you just heard and ask one natural follow-up question. Respond with \
             interleaved text and audio.",
        );
    let ctx_b = human.context_positions();

    let out_dir = Path::new("self_chat_out");
    std::fs::create_dir_all(out_dir)?;

    // Seed utterance: the reference spoken question, resampled by the engine's
    // own input path exactly like a live mic turn.
    let seed_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav");
    let (seed, seed_rate) = read_wav_mono_f32(&seed_path)?;
    eprintln!(
        "[seed] {:.2}s of speech from {} → assistant\n",
        seed.len() as f32 / seed_rate as f32,
        seed_path.display()
    );

    let cancel = AtomicBool::new(false);
    let mut incoming = Utterance {
        samples: seed,
        rate: seed_rate,
    };
    let mut records: Vec<TurnRecord> = Vec::new();
    let mut conversation: Vec<f32> = Vec::new();
    let gap = vec![0f32; (out_rate as usize) * 3 / 10]; // 300 ms between turns

    for turn in 1..=max_turns {
        // Assistant and "human" alternate; the assistant answers the seed first.
        let (engine, role): (&mut Lfm2VoiceEngine, &'static str) = if turn % 2 == 1 {
            (&mut assistant, "assistant")
        } else {
            (&mut human, "human")
        };

        print!("[turn {turn:02}] {role}: ");
        use std::io::Write as _;
        std::io::stdout().flush().ok();

        let mut transcript = String::new();
        let mut pcm: Vec<f32> = Vec::new();
        let (mut first_text_ms, mut first_audio_ms) = (None, None);
        let tg = Instant::now();
        let completed = engine.respond(&incoming, &cancel, &mut |ev| match ev {
            VoiceEvent::Text(t) => {
                first_text_ms.get_or_insert_with(|| tg.elapsed().as_millis() as u64);
                print!("{t}");
                std::io::stdout().flush().ok();
                transcript.push_str(&t);
            }
            VoiceEvent::Audio { pcm: chunk, .. } => {
                first_audio_ms.get_or_insert_with(|| tg.elapsed().as_millis() as u64);
                ring.lock().unwrap().extend(chunk.iter().copied());
                pcm.extend(chunk);
            }
            VoiceEvent::TurnComplete | VoiceEvent::Interrupted | VoiceEvent::Error(_) => {}
        })?;
        println!();
        let wall_s = tg.elapsed().as_secs_f32();
        let audio_s = pcm.len() as f32 / out_rate as f32;
        eprintln!(
            "[turn {turn:02}] {role}: {wall_s:.1}s wall | first text {} ms | first audio {} ms | {audio_s:.1}s audio | completed={completed}",
            first_text_ms.map_or("-".into(), |m| m.to_string()),
            first_audio_ms.map_or("-".into(), |m| m.to_string()),
        );

        write_wav_mono_f32(
            &out_dir.join(format!("turn_{turn:02}_{role}.wav")),
            &pcm,
            out_rate,
        )?;
        conversation.extend_from_slice(&pcm);
        conversation.extend_from_slice(&gap);
        records.push(TurnRecord {
            turn,
            role,
            transcript: transcript.clone(),
            wall_s,
            first_text_ms,
            first_audio_ms,
            audio_s,
        });

        if pcm.is_empty() {
            eprintln!("[turn {turn:02}] {role} produced no audio — ending the conversation here.");
            break;
        }

        // Let the speaker finish this turn before the next one starts thinking.
        loop {
            if ring.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // Context climb (the 32k soak's whole point): report both sides,
        // stop the moment either reaches the target.
        let (pa, pb) = (
            ctx_a.load(std::sync::atomic::Ordering::Relaxed),
            ctx_b.load(std::sync::atomic::Ordering::Relaxed),
        );
        eprintln!("[turn {turn:02}] ctx assistant {pa} | human {pb} / 32768");
        if target_ctx > 0 && (pa >= target_ctx || pb >= target_ctx) {
            eprintln!(
                "[soak] CTX TARGET REACHED: assistant {pa}, human {pb} \
                 (target {target_ctx}) after {turn} turns"
            );
            break;
        }

        // The other side hears exactly what was played.
        incoming = Utterance {
            samples: pcm,
            rate: out_rate,
        };
    }

    write_wav_mono_f32(&out_dir.join("conversation.wav"), &conversation, out_rate)?;
    let json = serde_json::to_string_pretty(
        &records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "turn": r.turn,
                    "role": r.role,
                    "transcript": r.transcript,
                    "wall_s": r.wall_s,
                    "first_text_ms": r.first_text_ms,
                    "first_audio_ms": r.first_audio_ms,
                    "audio_s": r.audio_s,
                })
            })
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(out_dir.join("transcript.json"), json)?;

    eprintln!("\n=== self-chat summary ({} turns) ===", records.len());
    for r in &records {
        eprintln!(
            "  {:02} {:9} wall {:5.1}s  first-audio {:>5} ms  audio {:4.1}s",
            r.turn,
            r.role,
            r.wall_s,
            r.first_audio_ms.map_or("-".into(), |m| m.to_string()),
            r.audio_s,
        );
    }
    eprintln!("[data] self_chat_out/: per-turn WAVs, conversation.wav, transcript.json");
    Ok(())
}
