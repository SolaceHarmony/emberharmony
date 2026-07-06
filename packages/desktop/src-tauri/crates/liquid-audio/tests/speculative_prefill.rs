//! Speculative prefill (prepare-during-pause) correctness against the real model.
//!
//! The contract: `prepare` is a pure ACCELERATOR. Whatever sequence of
//! prepare/discard calls precedes a `respond`, the reply must be what a plain
//! `respond` would have produced. Four equivalences, all through the public
//! engine API (the one mock is nothing at all — real model, real weights):
//!
//! 1. prepare(u) → respond(u)  ==  respond(u)                    (consume path)
//! 2. prepare(u′) → respond(u) ==  respond(u)                    (stale rollback)
//! 3. prepare(u) → discard → respond(u) == respond(u)            (explicit rollback)
//! 4. respond(u₁) → prepare(u₂) → respond(u₂) == two plain turns (cross-turn cache)
//!
//! Exactness target: the FIRST text run of each reply (greedy, conditioned only
//! on the prefill under test) — the same standard as cache_equivalence.rs.
//!
//! Run: LFM_DEVICE=metal LFM_MODEL_DIR=/path/to/model \
//!      cargo test --release --features metal --test speculative_prefill -- --nocapture

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use liquid_audio::{GenParams, Lfm2VoiceEngine, Utterance, VoiceEngine, VoiceEvent};

/// Minimal PCM16 WAV reader (mono-downmixed f32) — same as examples/generate.rs.
fn read_wav_mono_f32(path: &Path) -> (Vec<f32>, u32) {
    let b = std::fs::read(path).expect("read wav");
    assert!(
        b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE",
        "not a RIFF/WAVE file"
    );
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

/// One reply's observable stream: the leading greedy text run (exactness target)
/// and counts for sanity.
struct Reply {
    first_text_run: String,
    n_text_events: usize,
    n_audio_events: usize,
    wall_s: f32,
}

fn respond(engine: &mut Lfm2VoiceEngine, utt: &Utterance) -> Reply {
    let mut first_text_run = String::new();
    let (mut n_text, mut n_audio) = (0usize, 0usize);
    let t0 = Instant::now();
    let completed = engine
        .respond(utt, &AtomicBool::new(false), &mut |ev| match ev {
            VoiceEvent::Text(t) => {
                n_text += 1;
                if n_audio == 0 {
                    first_text_run.push_str(&t);
                }
            }
            VoiceEvent::Audio(_) => n_audio += 1,
            _ => {}
        })
        .expect("respond");
    assert!(completed, "reply did not run to completion");
    Reply {
        first_text_run,
        n_text_events: n_text,
        n_audio_events: n_audio,
        wall_s: t0.elapsed().as_secs_f32(),
    }
}

#[test]
fn prepare_is_a_pure_accelerator() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let dir = Path::new(&dir);
    let device = match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => Device::new_metal(0).expect("metal device"),
        _ => Device::Cpu,
    };
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .unwrap();
    let codebooks = cfg["codebooks"].as_u64().expect("config.json: codebooks") as usize;
    let (model, proc) = liquid_audio::from_pretrained(dir, &device).expect("load model");
    let (model, proc) = (Arc::new(model), Arc::new(proc));

    // Greedy text (the exactness target), sampled audio with a fixed seed.
    let params = GenParams {
        max_new_tokens: 48,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        ..GenParams::default()
    };
    let engine =
        || Lfm2VoiceEngine::new(model.clone(), proc.clone(), params.clone(), codebooks, device.clone(), 24_000);

    let wav_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav");
    let (samples, rate) = read_wav_mono_f32(&wav_path);
    let utt1 = Utterance {
        samples: samples.clone(),
        rate,
    };
    // A distinct second utterance: the first 2.5s of the same clip.
    let utt2 = Utterance {
        samples: samples[..(rate as usize * 5 / 2).min(samples.len())].to_vec(),
        rate,
    };

    // Reference: plain respond, fresh engine.
    let mut e_ref = engine();
    let r_ref = respond(&mut e_ref, &utt1);
    assert!(!r_ref.first_text_run.trim().is_empty(), "no leading text run");
    assert!(r_ref.n_audio_events > 0, "no audio generated");
    println!(
        "reference: {:.1}s wall, text {:?}",
        r_ref.wall_s, r_ref.first_text_run
    );

    // 1. Consume path: prepare(u) → respond(u).
    let mut e1 = engine();
    let tp = Instant::now();
    e1.prepare_turn(&utt1).expect("prepare");
    let prep_s = tp.elapsed().as_secs_f32();
    let r1 = respond(&mut e1, &utt1);
    println!(
        "prepared: prepare {prep_s:.2}s + respond {:.1}s wall (vs {:.1}s plain), text {:?}",
        r1.wall_s, r_ref.wall_s, r1.first_text_run
    );
    assert_eq!(
        r1.first_text_run, r_ref.first_text_run,
        "consume path diverged from plain respond"
    );

    // 2. Stale rollback: prepare a DIFFERENT utterance, respond with u.
    let mut e2 = engine();
    e2.prepare_turn(&utt2).expect("prepare other");
    let r2 = respond(&mut e2, &utt1);
    assert_eq!(
        r2.first_text_run, r_ref.first_text_run,
        "stale-prepare rollback diverged from plain respond"
    );

    // 3. Explicit rollback: prepare(u) → discard → respond(u).
    let mut e3 = engine();
    e3.prepare_turn(&utt1).expect("prepare");
    e3.discard_prepared_turn();
    let r3 = respond(&mut e3, &utt1);
    assert_eq!(
        r3.first_text_run, r_ref.first_text_run,
        "explicit rollback diverged from plain respond"
    );

    // 4. Cross-turn: the prepared second turn must match two plain turns —
    // the speculative path must not corrupt the persistent cross-turn cache.
    let mut e_plain = engine();
    let _ = respond(&mut e_plain, &utt1);
    let p2_ref = respond(&mut e_plain, &utt2);
    assert!(
        !p2_ref.first_text_run.trim().is_empty(),
        "no leading text run on turn 2"
    );

    let mut e4 = engine();
    let _ = respond(&mut e4, &utt1);
    e4.prepare_turn(&utt2).expect("prepare turn 2");
    let p2 = respond(&mut e4, &utt2);
    println!(
        "turn-2: plain {:.1}s vs prepared {:.1}s wall, text {:?}",
        p2_ref.wall_s, p2.wall_s, p2.first_text_run
    );
    assert_eq!(
        p2.first_text_run, p2_ref.first_text_run,
        "prepared turn 2 diverged from plain turn 2 (cross-turn cache corrupted)"
    );
    assert!(
        p2.n_text_events > 0 && p2.n_audio_events > 0,
        "prepared turn 2 produced an empty reply"
    );

    // 5. Rollback then a DIFFERENT next turn: prepare(u1) on top of turn 1,
    // discard, respond(u2) — must equal the plain two-turn run.
    let mut e5 = engine();
    let _ = respond(&mut e5, &utt1);
    e5.prepare_turn(&utt1).expect("prepare");
    e5.discard_prepared_turn();
    let p5 = respond(&mut e5, &utt2);
    assert_eq!(
        p5.first_text_run, p2_ref.first_text_run,
        "post-rollback turn 2 diverged (rollback did not restore the cache)"
    );
}
