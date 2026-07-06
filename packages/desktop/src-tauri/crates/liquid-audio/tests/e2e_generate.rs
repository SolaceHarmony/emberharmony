//! End-to-end: real model + real SPEECH (`assets/question.wav`, the upstream
//! reference clip) through the full changed stack — the wav-driven successor to
//! the retired parity examples, kept as a standing regression gate for any
//! change to the model/kernel stack.
//!
//! One two-turn spoken conversation covers:
//! - mel → Conformer → backbone prefill of real speech (not synthetic sines),
//! - interleaved generation (text + Depthformer audio frames),
//! - the persistent cross-turn cache (turn 2 runs the SUFFIX path),
//! - the fused ShortConv decode kernel (T=1 steps carrying conv state),
//! - Mimi streaming-decode of the reply to PCM, asserted non-silent.
//!
//! Then the whole FOUR-turn conversation is re-run with
//! `Cache::fused_conv_decode = false` (composed candle ops — injected through
//! the cache the test constructs, no ambient state). Turn 1's FULL stream
//! (greedy text + seeded sampled audio) must be IDENTICAL — the fused kernel
//! is built to match the composed path's bf16 rounding bit-for-bit at decode
//! shapes, so with identical logits the seeded sampler picks identical tokens;
//! a hard assertion, not a printout. Every later turn's first greedy text run
//! must then match too. Four turns, not two: compounding cursor drift lives at
//! turn 3+, and each turn asserts the suffix path (not a silent full
//! re-prefill) actually served it.
//!
//! Run: LFM_DEVICE=metal LFM_MODEL_DIR=/path/to/model \
//!      cargo test --release --features metal --test e2e_generate -- --nocapture

use std::path::Path;
use std::sync::atomic::AtomicBool;

use candle_core::{DType, Device, Tensor};
use liquid_audio::model::lfm2_hf::Cache as LfmCache;
use liquid_audio::moshi::demo::chat::decode_audio_reply;
use liquid_audio::moshi::models::MimiModel;
use liquid_audio::{
    ChatState, GenParams, GenToken, LFM2AudioModel, LFMModality, PrefillCursor,
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

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

#[derive(Clone, PartialEq, Debug)]
struct TurnOut {
    text_ids: Vec<u32>,
    audio_frames: Vec<Vec<u32>>,
    modality: Vec<i64>,
}

impl TurnOut {
    /// Greedy text tokens BEFORE the first audio frame — conditioned only on the
    /// prefill, so it is the exactness target for kernel-path A/B (everything
    /// after conditions on sampled audio; see cache_equivalence.rs assertion 3).
    fn first_text_run(&self) -> Vec<u32> {
        let n = self
            .modality
            .iter()
            .take_while(|m| **m == LFMModality::Text as i64)
            .count();
        self.text_ids[..n].to_vec()
    }
}

/// One spoken user turn + generated assistant turn through the persistent-cache
/// path, with engine-style bookkeeping (append + cursor advance).
#[allow(clippy::too_many_arguments)]
fn run_turn(
    model: &LFM2AudioModel,
    chat: &mut ChatState,
    cache: &mut Option<LfmCache>,
    cursor: &mut PrefillCursor,
    wave: &Tensor,
    rate: u32,
    params: &GenParams,
    codebooks: usize,
    device: &Device,
    fused_conv: bool,
    turn_idx: usize,
) -> TurnOut {
    chat.new_turn("user").unwrap();
    chat.add_audio(wave, rate).unwrap();
    chat.end_turn().unwrap();
    chat.new_turn("assistant").unwrap();

    let n_ctx = chat.modality_flag.dim(1).unwrap();
    let in_emb = model.prefill_suffix(chat, cursor).expect("suffix prefill");
    // The persistent cache must actually be taking the SUFFIX path on every
    // turn after the first — a silent fall-back to full re-prefill would keep
    // all outputs correct while quietly converting the accelerator into
    // per-turn latency that grows with conversation length. Gating on the turn
    // INDEX (not on `cursor.positions > 0`) is the point: the desync fallback
    // resets the cursor to zero, which is exactly the failure this must catch,
    // not a condition that lets it skip the check.
    let n_suffix = in_emb.dim(1).unwrap();
    if turn_idx > 0 {
        assert!(
            cursor.positions > 0,
            "turn {}: persistent cache was dropped (cursor reset to zero) — \
             a silent full re-prefill served this turn",
            turn_idx + 1
        );
        assert_eq!(
            n_suffix,
            n_ctx - cursor.positions,
            "suffix prefill length must be exactly the unseen context tail"
        );
    }
    if cache.is_none() {
        let mut fresh = model.make_cache(in_emb.dtype(), device).unwrap();
        fresh.fused_conv_decode = fused_conv;
        *cache = Some(fresh);
    }
    let mut index_pos = cursor.positions;
    let t0 = std::time::Instant::now();
    let mut out = TurnOut {
        text_ids: Vec::new(),
        audio_frames: Vec::new(),
        modality: Vec::new(),
    };
    model
        .generate_with_cache(
            cache.as_mut().unwrap(),
            &mut index_pos,
            in_emb,
            params,
            &AtomicBool::new(false),
            |tok| match tok {
                GenToken::Text(id) => {
                    out.text_ids.push(id);
                    out.modality.push(LFMModality::Text as i64);
                }
                GenToken::Audio(frame) => {
                    out.audio_frames.push(frame);
                    out.modality.push(LFMModality::AudioOut as i64);
                }
            },
        )
        .expect("generate");
    let n_tok = out.modality.len();
    eprintln!(
        "[turn] {n_tok} tokens in {:.1}s = {:.1} tok/s",
        t0.elapsed().as_secs_f32(),
        n_tok as f32 / t0.elapsed().as_secs_f32().max(1e-6)
    );

    // Append the generated turn exactly as the engine does.
    let n_text = out.text_ids.len();
    let text_t = Tensor::from_vec(
        out.text_ids.iter().map(|&i| i as i64).collect::<Vec<_>>(),
        (1, n_text),
        device,
    )
    .unwrap();
    let n_audio = out.audio_frames.len();
    let mut aflat = Vec::with_capacity(codebooks * n_audio);
    for c in 0..codebooks {
        for f in &out.audio_frames {
            aflat.push(f[c] as i64);
        }
    }
    let audio_t = if n_audio == 0 {
        Tensor::zeros((codebooks, 1), DType::I64, device)
            .unwrap()
            .narrow(1, 0, 0)
            .unwrap()
    } else {
        Tensor::from_vec(aflat, (codebooks, n_audio), device).unwrap()
    };
    let mod_t = Tensor::from_vec(out.modality.clone(), (1, out.modality.len()), device).unwrap();
    chat.append(&text_t, &audio_t, &mod_t).unwrap();
    chat.end_turn().unwrap();

    // Cursor advance: `index_pos` positions were forwarded, in context order, so
    // the per-modality forwarded totals are the flag counts over that prefix
    // (cumulative across turns — equivalent to cache_equivalence's incremental
    // accounting, proven there against a full re-prefill).
    let forwarded = index_pos - n_ctx;
    assert!(forwarded <= out.modality.len(), "cache advanced past emitted tokens");
    let flags: Vec<i64> = chat.modality_flag.flatten_all().unwrap().to_vec1().unwrap();
    cursor.positions = index_pos;
    cursor.text = flags
        .iter()
        .take(index_pos)
        .filter(|m| **m == LFMModality::Text as i64)
        .count();
    cursor.audio_out = flags
        .iter()
        .take(index_pos)
        .filter(|m| **m == LFMModality::AudioOut as i64)
        .count();
    cursor.audio_segments = chat.audio_in_lens.dim(0).unwrap();

    out
}

/// A full FOUR-turn spoken conversation (turn-3+ is where compounding cursor
/// drift would live; two-turn tests are structurally blind to it); returns all
/// turns, the last turn's reply text, and its Mimi-decoded PCM (+ sample rate).
fn run_conversation(
    device: &Device,
    dir: &Path,
    fused_conv: bool,
) -> (Vec<TurnOut>, String, Vec<f32>, u32) {
    let (model, proc) = liquid_audio::from_pretrained(dir, device).expect("load model");
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .unwrap();
    let codebooks = cfg["codebooks"].as_u64().expect("config.json: codebooks") as usize;
    let mimi = MimiModel::new(proc.mimi().expect("Mimi codec required"));

    // Real speech in.
    let wav_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/question.wav");
    let (samples, rate) = read_wav_mono_f32(&wav_path);
    assert!(
        samples.len() as f32 / rate as f32 > 1.0,
        "question.wav should be > 1s of speech"
    );
    let wave = Tensor::from_vec(samples.clone(), (1, samples.len()), device).unwrap();

    // README regime: greedy text (the A/B exactness target), sampled audio
    // (greedy audio is degenerate for the Depthformer), fixed seed.
    let params = GenParams {
        max_new_tokens: 64,
        audio_temperature: Some(1.0),
        audio_top_k: Some(4),
        ..GenParams::default()
    };

    let mut chat = ChatState::new(&proc, codebooks).expect("chat");
    chat.new_turn("system").unwrap();
    chat.add_text("Respond with interleaved text and audio.").unwrap();
    chat.end_turn().unwrap();

    let mut cache: Option<LfmCache> = None;
    let mut cursor = PrefillCursor::default();
    let turns: Vec<TurnOut> = (0..4)
        .map(|turn_idx| {
            run_turn(
                &model, &mut chat, &mut cache, &mut cursor, &wave, rate, &params, codebooks,
                device, fused_conv, turn_idx,
            )
        })
        .collect();

    let last = turns.last().expect("four turns");
    let text_last = proc.text().decode(&last.text_ids, true).unwrap_or_default();
    let pcm = decode_audio_reply(&mimi, &last.audio_frames, codebooks, device)
        .expect("mimi decode")
        .map(|t| {
            t.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .unwrap_or_default();

    (turns, text_last, pcm, mimi.sample_rate())
}

#[test]
fn e2e_four_turns_real_speech_and_fused_conv_ab() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let dir = Path::new(&dir);
    let device = match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => Device::new_metal(0).expect("metal device"),
        _ => Device::Cpu,
    };

    // ---- Phase A: fused ShortConv decode path (the production default). ----
    let (a, text_a, pcm_a, mimi_rate) = run_conversation(&device, dir, true);

    for (i, t) in a.iter().enumerate() {
        let name = format!("turn{}", i + 1);
        assert!(!t.text_ids.is_empty(), "{name}: no text generated");
        assert!(!t.audio_frames.is_empty(), "{name}: no audio generated");
        assert!(
            !t.first_text_run().is_empty(),
            "{name}: reply does not open with a text run (interleave broken)"
        );
    }
    println!("turn-4 reply text (fused): {text_a:?}");

    // The spoken reply must decode to real, non-silent audio of plausible length.
    assert!(!pcm_a.is_empty(), "turn-4 reply produced no PCM");
    let dur = pcm_a.len() as f32 / mimi_rate as f32;
    let level = rms(&pcm_a);
    println!("turn-4 reply audio: {dur:.2}s @ {mimi_rate} Hz, rms {level:.4}");
    assert!(dur > 0.2, "turn-4 reply audio implausibly short: {dur:.2}s");
    assert!(level > 1e-4, "turn-4 reply audio decodes to silence (rms {level})");

    // ---- Phase B: composed candle ops (fused kernel off via the cache seam). ----
    let (b, text_b, _pcm_b, _) = run_conversation(&device, dir, false);
    println!("turn-4 reply text (composed): {text_b:?}");

    // Turn 1's first text run conditions only on the deterministic prefill:
    // fused and composed paths must agree EXACTLY.
    assert_eq!(
        a[0].first_text_run(),
        b[0].first_text_run(),
        "turn-1 first text run diverged between fused and composed conv paths"
    );

    // The fused kernel rounds through bf16 to match the composed path
    // bit-for-bit at the decode shapes, and audio sampling is seeded — so with
    // identical logits the FULL turn-1 stream (text + sampled audio) must be
    // identical. This is a hard assertion, deliberately: the earlier version
    // downgraded it to a println with a "near-tie sampling" escape hatch, which
    // let any fused-kernel corruption past the first text run ride through the
    // suite unasserted. If this ever fails, the burden of proof is on showing
    // the divergence really was a near-tie logit — not on the test to assume it.
    assert_eq!(
        a[0], b[0],
        "turn-1 FULL stream (text + sampled audio) diverged between fused and \
         composed conv paths — kernel numerics drifted"
    );
    // With turn 1 identical, later turns share identical context: compare the
    // greedy text runs on every remaining turn.
    for i in 1..a.len() {
        assert_eq!(
            a[i].first_text_run(),
            b[i].first_text_run(),
            "turn-{} first text run diverged between fused and composed conv paths",
            i + 1
        );
    }
}
