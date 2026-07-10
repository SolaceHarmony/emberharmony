//! The REAL end-to-end: a mocked microphone and everything else live. The test
//! streams `assets/question.wav` into the full `VoiceRuntime` service exactly as
//! a mic driver would — 20 ms mono chunks at wall-clock cadence, silence between
//! turns — and lets the production stack do the rest: the VAD/turn loop, the
//! `RealtimePipeline` worker threads, the `Lfm2VoiceEngine` with cross-turn
//! conversation memory, streaming Mimi decode, and the DEFAULT OUTPUT DEVICE.
//! You hear both replies on the speaker; `played_samples` counts what the CPAL
//! output callback actually delivered to the hardware.
//!
//! The one mock is the mic (`ExternalAudioInput` — the same seam the Tauri app
//! feeds WebRTC frames through). Output is deliberately NOT mocked: the point is
//! that audio leaves the box.
//!
//! Two full spoken turns assert: Listening→Thinking→Speaking→Listening per turn,
//! a non-empty transcript per turn, non-silent decoded audio, real samples
//! played on the device, and both pause→first-audio latency measurements.
//!
//! Run (audible — needs a speaker and the local model; `audio-io` enables the
//! built-in CPAL output device, same opt-in the standalone examples use):
//!   LFM_DEVICE=metal LFM_MODEL_DIR=/path/to/model \
//!     cargo test --release --features metal,audio-io --test e2e_voice_runtime -- --nocapture
#![cfg(feature = "audio-io")]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::Device;
use liquid_audio::{
    ExternalAudioInput, GenParams, Lfm2VoiceEngine, RuntimeConfig, RuntimeEvent, SessionState,
    VoiceRuntime,
};

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

/// Compact mirror of the runtime events (Audio PCM reduced to len + rms so the
/// channel doesn't buffer megabytes of samples).
#[derive(Debug, Clone)]
enum Ev {
    State(SessionState),
    Transcript(String),
    Audio {
        #[allow(dead_code)] // read via Debug in the failure-log dumps
        samples: usize,
        rms: f32,
    },
    Ended(Option<String>),
    Error(String),
}

fn forward_events(tx: Sender<Ev>) -> impl FnMut(RuntimeEvent) -> bool + Send + 'static {
    move |event| {
        let ev = match event {
            RuntimeEvent::State(s) => Some(Ev::State(s)),
            RuntimeEvent::Transcript(t) => Some(Ev::Transcript(t)),
            RuntimeEvent::Audio { pcm, .. } => {
                let rms = if pcm.is_empty() {
                    0.0
                } else {
                    (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt()
                };
                Some(Ev::Audio {
                    samples: pcm.len(),
                    rms,
                })
            }
            RuntimeEvent::Level(_) => None,
            RuntimeEvent::Ended(reason) => Some(Ev::Ended(reason)),
            RuntimeEvent::Error(e) => Some(Ev::Error(e)),
        };
        if let Some(ev) = ev {
            // Receiver gone = test is done tearing down; tell the runtime to stop.
            return tx.send(ev).is_ok();
        }
        true
    }
}

/// Drain events until `want` arrives (panicking on Error/Ended), with a deadline.
/// Every event passes through `observe` so callers can accumulate transcripts etc.
fn wait_for_state(
    rx: &Receiver<Ev>,
    want: SessionState,
    deadline: Duration,
    log: &mut Vec<String>,
    mut observe: impl FnMut(&Ev),
) {
    let t0 = Instant::now();
    loop {
        let remaining = deadline
            .checked_sub(t0.elapsed())
            .unwrap_or_else(|| panic!("timed out waiting for {want:?}; events so far: {log:#?}"));
        let ev = match rx.recv_timeout(remaining) {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => {
                panic!("timed out waiting for {want:?}; events so far: {log:#?}")
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("runtime ended while waiting for {want:?}; events so far: {log:#?}")
            }
        };
        log.push(format!("+{:6.2}s {ev:?}", t0.elapsed().as_secs_f32()));
        match &ev {
            Ev::Error(e) => panic!("runtime error: {e}; events so far: {log:#?}"),
            Ev::Ended(reason) => {
                panic!("session ended early ({reason:?}); events so far: {log:#?}")
            }
            _ => {}
        }
        observe(&ev);
        if matches!(&ev, Ev::State(s) if *s == want) {
            return;
        }
    }
}

#[test]
fn e2e_voice_runtime_speaks_two_turns_through_real_speaker() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let dir = std::path::PathBuf::from(dir);

    let wav_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav");
    let (utterance, mic_rate) = read_wav_mono_f32(&wav_path);
    assert!(
        utterance.len() as f32 / mic_rate as f32 > 1.0,
        "question.wav should be > 1s of speech"
    );

    // ---- The one mock: a mic that plays a queue of samples in real time. ----
    // 20ms mono chunks at wall-clock cadence, zeros when the queue is empty —
    // exactly what a hardware capture callback delivers between utterances.
    let (input, writer) = ExternalAudioInput::new(mic_rate).expect("external input");
    let feed: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let feeder_stop = Arc::new(AtomicBool::new(false));
    let feeder = {
        let feed = feed.clone();
        let stop = feeder_stop.clone();
        let chunk = (mic_rate as usize / 50).max(1); // 20 ms
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

    // ---- Everything else is the production stack, speaker included. ----
    let (tx, rx) = channel();
    // Speculative-prefill counters, smuggled out of the engine the session
    // thread builds: the live proof that prepare-during-pause actually CONSUMES
    // (equivalence alone is satisfied by an accelerator that never fires).
    let spec: Arc<
        std::sync::OnceLock<(
            Arc<std::sync::atomic::AtomicU64>,
            Arc<std::sync::atomic::AtomicU64>,
        )>,
    > = Arc::new(std::sync::OnceLock::new());
    let spec_slot = spec.clone();
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
                // Production parity (desktop TurnMode budget): the model speaks
                // until <|im_end|>, never a mid-sentence cap — the old 128 cut
                // turn 2 off at a comma and shipped as a green test. 8192 lets
                // long self-talk genuinely stress the 32,768-token context.
                max_new_tokens: 8192,
                audio_temperature: Some(1.0),
                audio_top_k: Some(4),
                ..GenParams::default()
            };
            let engine = Lfm2VoiceEngine::new(model, proc, params, codebooks, device, out_rate);
            let _ = spec_slot.set(engine.speculative_counters());
            Ok(Box::new(engine) as Box<dyn liquid_audio::VoiceEngine>)
        },
        forward_events(tx),
    )
    .expect("start voice runtime");

    let mut log = Vec::new();
    // Model load happens inside the session thread → generous first deadline.
    wait_for_state(
        &rx,
        SessionState::Listening,
        Duration::from_secs(180),
        &mut log,
        |_| {},
    );
    println!("[e2e] session up and listening");

    let mut transcripts = [String::new(), String::new()];
    let mut audio_rms_max = [0f32; 2];
    for (turn, transcript) in transcripts.iter_mut().enumerate() {
        // "Speak" the question into the mock mic; the feeder's trailing zeros are
        // the end-of-turn silence the VAD commits on.
        feed.lock().unwrap().extend(utterance.iter().copied());
        println!(
            "[e2e] turn {}: speaking {:.2}s into the mock mic",
            turn + 1,
            utterance.len() as f32 / mic_rate as f32
        );

        let utt_secs = utterance.len() as f32 / mic_rate as f32;
        let capture = Duration::from_secs_f32(utt_secs) + Duration::from_secs(30);
        wait_for_state(&rx, SessionState::Thinking, capture, &mut log, |_| {});
        wait_for_state(
            &rx,
            SessionState::Speaking,
            Duration::from_secs(60),
            &mut log,
            |_| {},
        );
        // Reply streams until the runtime reopens the mic (post playback + echo tail).
        let rms_slot = &mut audio_rms_max[turn];
        wait_for_state(
            &rx,
            SessionState::Listening,
            Duration::from_secs(120),
            &mut log,
            |ev| match ev {
                Ev::Transcript(t) => *transcript = t.clone(),
                Ev::Audio { rms, .. } => *rms_slot = rms_slot.max(*rms),
                _ => {}
            },
        );
        println!("[e2e] turn {} transcript: {transcript:?}", turn + 1);
    }

    // Let the speaker DRAIN before teardown: the last reply's tail is still in the
    // output queue, and stopping now truncates audible playback (and undercounts
    // played_samples — earlier runs showed played < queued by ~28k samples). Poll
    // until the output callback has delivered everything that was queued, with a
    // bounded deadline so a stalled device fails loudly instead of hanging.
    let drain_deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let s = runtime.audio_stats();
        if s.played_samples >= s.queued_samples {
            break;
        }
        assert!(
            Instant::now() < drain_deadline,
            "speaker did not drain: played {} of {} queued",
            s.played_samples,
            s.queued_samples
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    let stats = runtime.audio_stats();
    feeder_stop.store(true, Ordering::SeqCst);
    runtime.stop();
    let _ = feeder.join();

    // The drain gate, kept as a hard assertion: every decoded-and-queued sample must
    // actually reach the hardware — a truncated reply is a product bug, not a flake.
    assert_eq!(
        stats.played_samples, stats.queued_samples,
        "output truncated at teardown"
    );

    println!(
        "[e2e] stats: decoded {} queued {} played {} dropped {} underruns {} | \
         turns {} last-latency {}ms mean-latency {}ms",
        stats.decoded_samples,
        stats.queued_samples,
        stats.played_samples,
        stats.dropped_samples,
        stats.underrun_frames,
        stats.turn_count,
        stats.last_turn_latency_ms,
        stats.mean_turn_latency_ms,
    );

    for (turn, transcript) in transcripts.iter().enumerate() {
        assert!(
            !transcript.trim().is_empty(),
            "turn {}: empty transcript; events: {log:#?}",
            turn + 1
        );
        assert!(
            audio_rms_max[turn] > 1e-4,
            "turn {}: decoded reply audio is silent (max chunk rms {}); events: {log:#?}",
            turn + 1,
            audio_rms_max[turn]
        );
    }
    assert!(stats.decoded_samples > 0, "no audio decoded");
    // The heart of the test: the REAL output device consumed reply samples —
    // audio physically left the box.
    assert!(
        stats.played_samples > 0,
        "no samples reached the speaker; events: {log:#?}"
    );
    assert_eq!(
        stats.turn_count, 2,
        "expected two pause→first-audio turn measurements; events: {log:#?}"
    );

    // The speculative prefill must have actually FIRED AND CONSUMED through the
    // live VAD path (pause-onset prepare → byte-identical committed trim →
    // fingerprint match). Reply equivalence cannot prove this — a prepare that
    // never matches degrades into silent waste while every other assertion
    // above still passes. The mock mic's pauses are clean, so both turns
    // should consume; require at least one and report both counts.
    let (consumed, discarded) = spec
        .get()
        .map(|(c, d)| {
            (
                c.load(std::sync::atomic::Ordering::Relaxed),
                d.load(std::sync::atomic::Ordering::Relaxed),
            )
        })
        .expect("engine was never built");
    println!("[e2e] speculative prefill: consumed {consumed}, discarded {discarded}");
    assert!(
        consumed >= 1,
        "no speculative prefill was consumed through the live VAD path \
         (prepare/commit trim identity broken?); consumed {consumed}, discarded {discarded}"
    );
}
