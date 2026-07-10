//! The 32k-context soak: the model in conversation with ITSELF, audibly.
//!
//! The two-turn e2e is scripted — a canned WAV question, twice, tiny context.
//! This test removes the script: after one seeded question, every subsequent
//! "utterance" is the model's OWN spoken reply fed back through the mock mic.
//! No text tokens steer it; the model hears speech and answers with speech,
//! turn after turn, while the conversation context climbs toward the model's
//! 32,768-token ceiling. The pass criterion is reaching CTX_TARGET positions
//! with zero runtime errors, non-silent audio every turn, and the drain
//! guarantee holding (every queued sample played before each turn ends).
//!
//! This is a SOAK (tens of minutes on CPU), not a gate rung — run explicitly:
//!   LFM_MODEL_DIR=/path/to/model LFM_DEVICE=cpu \
//!     cargo test --release --features accelerate,audio-io \
//!     --test e2e_self_talk_32k -- --nocapture --ignored
#![cfg(feature = "audio-io")]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use candle_core::Device;
use liquid_audio::{
    ExternalAudioInput, GenParams, Lfm2VoiceEngine, RuntimeConfig, RuntimeEvent, SessionState,
    VoiceRuntime,
};

const CTX_TARGET: u64 = 30_000; // proves the climb; ceiling is 32,768
const MAX_TURNS: usize = 64;
const FEED_RATE: u32 = 24_000; // Mimi's output rate — replies loop back natively

/// Minimal PCM16 WAV reader (mono-downmixed f32) — same as the two-turn e2e.
fn read_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let b = std::fs::read(path).expect("read wav");
    assert!(b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE");
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
    assert_eq!(bits, 16, "only PCM16 WAV supported");
    let data = data.expect("no data chunk");
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
    (mono, rate)
}

/// Event mirror that KEEPS the reply PCM (unlike the two-turn e2e) — the whole
/// point is feeding it back.
#[derive(Debug, Clone)]
enum Ev {
    State(SessionState),
    Transcript(String),
    Audio { pcm: Vec<f32>, rate: u32 },
    Ended(Option<String>),
    Error(String),
}

fn forward_events(tx: Sender<Ev>) -> impl FnMut(RuntimeEvent) -> bool + Send + 'static {
    move |event| {
        let ev = match event {
            RuntimeEvent::State(s) => Some(Ev::State(s)),
            RuntimeEvent::Transcript(t) => Some(Ev::Transcript(t)),
            RuntimeEvent::Audio { pcm, rate } => Some(Ev::Audio { pcm, rate }),
            RuntimeEvent::Level(_) => None,
            RuntimeEvent::Ended(reason) => Some(Ev::Ended(reason)),
            RuntimeEvent::Error(e) => Some(Ev::Error(e)),
        };
        if let Some(ev) = ev {
            return tx.send(ev).is_ok();
        }
        true
    }
}

fn wait_for_state(
    rx: &Receiver<Ev>,
    want: SessionState,
    deadline: Duration,
    turn: usize,
    mut observe: impl FnMut(&Ev),
) {
    let t0 = Instant::now();
    loop {
        let remaining = deadline
            .checked_sub(t0.elapsed())
            .unwrap_or_else(|| panic!("turn {turn}: timed out waiting for {want:?}"));
        let ev = match rx.recv_timeout(remaining) {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => {
                panic!("turn {turn}: timed out waiting for {want:?}")
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("turn {turn}: runtime ended while waiting for {want:?}")
            }
        };
        match &ev {
            Ev::Error(e) => panic!("turn {turn}: runtime error: {e}"),
            Ev::Ended(reason) => panic!("turn {turn}: session ended early ({reason:?})"),
            _ => {}
        }
        observe(&ev);
        if matches!(&ev, Ev::State(s) if *s == want) {
            return;
        }
    }
}

#[test]
#[ignore = "32k-context soak (tens of minutes, audible) — run explicitly; see module docs"]
fn e2e_self_talk_reaches_32k_context() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let dir = std::path::PathBuf::from(dir);

    // Seed question, resampled to the feed rate (the crate's own windowed-sinc).
    let wav_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav");
    let (seed_raw, seed_rate) = read_wav_mono_f32(&wav_path);
    let seed = liquid_audio::resample::resample_slice(&seed_raw, seed_rate, FEED_RATE);

    // The one mock: a 24 kHz mic playing a queue at wall-clock cadence.
    let (input, writer) = ExternalAudioInput::new(FEED_RATE).expect("external input");
    let feed: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let feeder_stop = Arc::new(AtomicBool::new(false));
    let feeder = {
        let feed = feed.clone();
        let stop = feeder_stop.clone();
        let chunk = (FEED_RATE as usize / 50).max(1); // 20 ms
        std::thread::spawn(move || {
            let period = Duration::from_millis(20);
            let mut next = Instant::now() + period;
            let mut buf = Vec::with_capacity(chunk);
            while !stop.load(Ordering::SeqCst) {
                buf.clear();
                {
                    let mut q = feed.lock().unwrap();
                    for _ in 0..chunk {
                        buf.push(q.pop_front().unwrap_or(0.0));
                    }
                }
                writer.push_mono_f32(&buf);
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                }
                next += period;
            }
        })
    };

    let (tx, rx) = channel();
    let ctx_probe: Arc<OnceLock<Arc<AtomicU64>>> = Arc::new(OnceLock::new());
    let ctx_slot = ctx_probe.clone();
    let runtime = VoiceRuntime::start_with_input(
        RuntimeConfig::default(),
        Some(input),
        move |out_rate| {
            let device = match std::env::var("LFM_DEVICE").ok().as_deref() {
                Some("metal") => Device::new_metal(0).map_err(|e| format!("metal: {e}"))?,
                _ => Device::Cpu,
            };
            let (model, proc) = liquid_audio::from_pretrained(&dir, &device)
                .map_err(|e| format!("load model: {e}"))?;
            let cfg: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(dir.join("config.json"))
                    .map_err(|e| format!("config.json: {e}"))?,
            )
            .map_err(|e| format!("config.json: {e}"))?;
            let codebooks = cfg["codebooks"]
                .as_u64()
                .ok_or("config.json: missing `codebooks`")? as usize;
            let params = GenParams {
                max_new_tokens: 8192,
                audio_temperature: Some(1.0),
                audio_top_k: Some(4),
                ..GenParams::default()
            };
            let engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
            let _ = ctx_slot.set(engine.context_positions());
            Ok(Box::new(engine) as Box<dyn liquid_audio::VoiceEngine>)
        },
        forward_events(tx),
    )
    .expect("start voice runtime");

    wait_for_state(&rx, SessionState::Listening, Duration::from_secs(180), 0, |_| {});
    let ctx = ctx_probe.get().expect("engine built").clone();
    println!("[soak] session up; seeding one question, then the model talks to itself");

    let soak_t0 = Instant::now();
    let mut next_utterance: Vec<f32> = seed;
    let mut reached = false;
    let mut last_transcript = String::new();
    let mut repeats = 0usize;
    for turn in 1..=MAX_TURNS {
        let utt_secs = next_utterance.len() as f32 / FEED_RATE as f32;
        feed.lock().unwrap().extend(next_utterance.iter().copied());
        println!("[soak] turn {turn}: feeding {utt_secs:.1}s of speech");

        let capture = Duration::from_secs_f32(utt_secs) + Duration::from_secs(180);
        wait_for_state(&rx, SessionState::Thinking, capture, turn, |_| {});
        wait_for_state(&rx, SessionState::Speaking, Duration::from_secs(600), turn, |_| {});

        // Collect the whole reply until the runtime reopens the mic (which, per
        // the drain guarantee, means every queued sample was played).
        let mut reply: Vec<f32> = Vec::new();
        let mut reply_rate: u32 = FEED_RATE;
        let mut transcript = String::new();
        wait_for_state(
            &rx,
            SessionState::Listening,
            Duration::from_secs(900),
            turn,
            |ev| match ev {
                Ev::Audio { pcm, rate } => {
                    reply_rate = *rate;
                    reply.extend_from_slice(pcm);
                }
                Ev::Transcript(t) => transcript = t.clone(),
                _ => {}
            },
        );
        // THE bug this harness caught in production shape: the reply arrives at
        // the OUTPUT DEVICE rate (the event carries it) — feeding it back at an
        // assumed 24 kHz played it half-speed and the model heard rumble
        // ("I'm sorry, I didn't catch your words", forever). Resample by the
        // carried rate before it becomes the next utterance.
        if reply_rate != FEED_RATE {
            reply = liquid_audio::resample::resample_slice(&reply, reply_rate, FEED_RATE);
        }
        let rms = if reply.is_empty() {
            0.0
        } else {
            (reply.iter().map(|x| x * x).sum::<f32>() / reply.len() as f32).sqrt()
        };
        assert!(
            !reply.is_empty() && rms > 1e-4,
            "turn {turn}: silent/empty reply (rms {rms:.2e}) — the conversation died"
        );
        // Degeneracy tripwire: a self-conversation stuck on one sentence
        // (e.g. the "didn't catch your words" apology loop) fails FAST with a
        // diagnosis instead of burning an hour reaching MAX_TURNS.
        if transcript == last_transcript {
            repeats += 1;
            assert!(
                repeats < 3,
                "turn {turn}: transcript identical 3 turns running — the \
                 conversation degenerated (audio unintelligible to the model, \
                 or template/turn-structure bug): {transcript:?}"
            );
        } else {
            repeats = 0;
            last_transcript = transcript.clone();
        }
        let pos = ctx.load(Ordering::Relaxed);
        println!(
            "[soak] turn {turn}: reply {:.1}s (rms {rms:.3}), ctx {pos}/32768, {:.1} min elapsed — {:?}",
            reply.len() as f32 / FEED_RATE as f32,
            soak_t0.elapsed().as_secs_f32() / 60.0,
            transcript.chars().take(80).collect::<String>()
        );
        if pos >= CTX_TARGET {
            println!("[soak] CTX TARGET REACHED: {pos} positions after {turn} turns");
            reached = true;
            break;
        }
        next_utterance = reply;
    }
    assert!(
        reached,
        "context did not reach {CTX_TARGET} within {MAX_TURNS} turns — \
         either replies are too short or the conversation cache is not growing"
    );

    feeder_stop.store(true, Ordering::SeqCst);
    let _ = feeder.join();
    drop(runtime);
}
