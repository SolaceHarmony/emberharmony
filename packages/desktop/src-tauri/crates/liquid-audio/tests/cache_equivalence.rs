//! Cache-equivalence invariant (spec 09, W2a): the persistent cross-turn KV cache
//! must be a pure accelerator. Kept after the Python-parity suite was retired —
//! this guards OUR contract (suffix prefill + carried cache == full re-prefill),
//! not the port's faithfulness, which was proven and archived in the
//! candle-audio-rs repository together with the golden-file parity tests.
//!
//! Run: LFM_MODEL_DIR=/path/to/model cargo test --release --test cache_equivalence -- --nocapture
//! (LFM_DEVICE=metal for the deployed bf16 Metal numerics.)

use std::path::Path;

use candle_core::{Device, Tensor};

fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
    let a = a
        .flatten_all()
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap();
    let b = b
        .flatten_all()
        .unwrap()
        .to_dtype(candle_core::DType::F32)
        .unwrap();
    let diff = (&a - &b)
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let scale = b
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .max(1e-6);
    diff / scale
}

/// Emitted-token collector for the suffix-cache equivalence test: text ids as
/// `(id, [])`, audio frames as `(-1, frame)`, with the interleaved modality order.
fn collect_tokens<'a>(
    toks: &'a mut Vec<(i64, Vec<u32>)>,
    mods: &'a mut Vec<i64>,
) -> impl FnMut(liquid_audio::GenToken) + 'a {
    move |tok| match tok {
        liquid_audio::GenToken::Text(id) => {
            toks.push((id as i64, Vec::new()));
            mods.push(liquid_audio::LFMModality::Text as i64);
        }
        liquid_audio::GenToken::Audio(frame) => {
            toks.push((-1, frame));
            mods.push(liquid_audio::LFMModality::AudioOut as i64);
        }
    }
}

/// Spec 09, W2a: the persistent cross-turn cache must be a pure accelerator. Three
/// assertions, strongest to weakest against bf16 noise:
///
/// 1. **Pure forward**: the same context embeddings forwarded whole vs split at the
///    turn boundary (chunk 1 through a cache, then chunk 2 continuing it) must match
///    within bf16 tolerance — the numerical contract of conv-state + KV continuation.
/// 2. **Suffix construction**: `prefill_suffix` embeds must equal the tail of the
///    full-prefill embeds (same conformer runs, same lookups).
/// 3. **Generation**: the FIRST greedy text run must be IDENTICAL between the
///    suffix-cache and full-re-prefill turns — it conditions solely on the context
///    prefill under test. Everything after it conditions on previously generated
///    AUDIO frames, and greedy audio is degenerate for this model (near-tie logits),
///    so chunk-shape-dependent bf16 GEMM rounding legitimately flips audio argmax
///    ties and shifts all downstream tokens (observed on both CPU and Metal).
///    Production audio is sampled (temp 1.0 / top-k 4), never greedy.
///
/// Run: LFM_MODEL_DIR=/path/to/model cargo test --release --test parity suffix_cache -- --nocapture
#[test]
fn suffix_cache_matches_full_prefill() -> anyhow::Result<()> {
    use liquid_audio::{ChatState, GenParams, LFMModality, PrefillCursor};
    use std::sync::atomic::AtomicBool;

    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    // Default CPU (parity-suite convention); LFM_DEVICE=metal runs the deployed
    // bf16 Metal numerics and is ~20× faster for the generation phase.
    let device = match std::env::var("LFM_DEVICE").ok().as_deref() {
        Some("metal") => Device::new_metal(0).map_err(|e| anyhow::anyhow!("metal: {e}"))?,
        _ => Device::Cpu,
    };
    let (model, proc) = liquid_audio::from_pretrained(Path::new(&dir), &device)?;
    let codebooks = 8usize;

    // Deterministic generation: greedy text AND greedy audio on both paths.
    let params = GenParams {
        max_new_tokens: 32,
        ..GenParams::default()
    };

    // Synthetic spoken turns: half a second of a sine tone (the model's reply content
    // is irrelevant — only path-equality matters, and greedy makes it reproducible).
    let sine = |hz: f32| -> Vec<f32> {
        (0..8000)
            .map(|i| (std::f32::consts::TAU * hz * i as f32 / 16_000.0).sin() * 0.3)
            .collect()
    };
    let wave1 = Tensor::from_vec(sine(440.0), (1, 8000), &device)?;
    let wave2 = Tensor::from_vec(sine(660.0), (1, 8000), &device)?;

    // ---- Turn 1 (shared prefix): build context, generate with a fresh cache. ----
    let mut chat = ChatState::new(&proc, codebooks)?;
    chat.new_turn("system")?;
    chat.add_text("Respond with interleaved text and audio.")?;
    chat.end_turn()?;
    chat.new_turn("user")?;
    chat.add_audio(&wave1, 16_000)?;
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    let n_ctx1 = chat.modality_flag.dim(1)?;
    let in_emb = model.prefill_suffix(&chat, &PrefillCursor::default())?;
    let mut cache = model.make_cache(in_emb.dtype(), &device)?;
    let mut index_pos = 0usize;
    let (mut toks1, mut mods1) = (Vec::new(), Vec::new());
    model.generate_with_cache(
        &mut cache,
        &mut index_pos,
        in_emb,
        &params,
        &AtomicBool::new(false),
        collect_tokens(&mut toks1, &mut mods1),
    )?;
    assert!(!toks1.is_empty(), "turn 1 generated nothing");

    // Append the generated turn exactly as the engine does, then advance the cursor
    // with the engine's accounting: everything forwarded = context + all emitted
    // tokens except (possibly) the last.
    let text_ids: Vec<i64> = toks1.iter().filter(|(t, _)| *t >= 0).map(|(t, _)| *t).collect();
    let frames: Vec<&Vec<u32>> = toks1.iter().filter(|(t, _)| *t < 0).map(|(_, f)| f).collect();
    let text_t = Tensor::from_vec(text_ids.clone(), (1, text_ids.len()), &device)?;
    let mut flat = Vec::with_capacity(codebooks * frames.len());
    for c in 0..codebooks {
        for f in &frames {
            flat.push(f[c] as i64);
        }
    }
    let audio_t = if frames.is_empty() {
        Tensor::zeros((codebooks, 1), candle_core::DType::I64, &device)?.narrow(1, 0, 0)?
    } else {
        Tensor::from_vec(flat, (codebooks, frames.len()), &device)?
    };
    let mod_t = Tensor::from_vec(mods1.clone(), (1, mods1.len()), &device)?;
    chat.append(&text_t, &audio_t, &mod_t)?;
    chat.end_turn()?;

    let forwarded = index_pos - n_ctx1;
    assert!(forwarded <= mods1.len(), "cache advanced past emitted tokens");
    // Engine accounting: cursor = per-modality totals at generation start + the
    // forwarded prefix of the emitted stream. `end_turn` added "<|im_end|>\n" AFTER
    // generation (never forwarded): pre-gen text total = text now − generated − trailing.
    let trailing_end_turn = chat.modality_flag.dim(1)? - n_ctx1 - mods1.len();
    let mut cursor = PrefillCursor {
        positions: index_pos,
        text: chat.text.dim(1)? - text_ids.len() - trailing_end_turn,
        audio_segments: 1,
        audio_out: 0,
    };
    for m in mods1.iter().take(forwarded) {
        if *m == LFMModality::Text as i64 {
            cursor.text += 1;
        } else {
            cursor.audio_out += 1;
        }
    }

    // ---- Turn 2 context: another spoken user turn. ----
    chat.new_turn("user")?;
    chat.add_audio(&wave2, 16_000)?;
    chat.end_turn()?;
    chat.new_turn("assistant")?;

    // The turn grammar must be IN the context the model attends over: role fences
    // as real text tokens, audio as flagged runs between them. This is the live
    // answer to "does the model know where it ends and the user begins".
    let transcript = chat.transcript()?;
    println!("--- context transcript (turn 2 start) ---\n{transcript}\n---");
    for fence in [
        "<|startoftext|>",
        "<|im_start|>system\n",
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
        "<|im_end|>",
        "⟨audio-in ×",
        "⟨audio-out ×",
    ] {
        assert!(
            transcript.contains(fence),
            "context transcript missing {fence:?}"
        );
    }

    // ---- Assertion 2: suffix construction equals the tail of the full prefill. ----
    let suffix = model.prefill_suffix(&chat, &cursor)?;
    let full_embeds = model.prefill_suffix(&chat, &PrefillCursor::default())?;
    let n_full = full_embeds.dim(1)?;
    let n_suffix = suffix.dim(1)?;
    let tail = full_embeds.narrow(1, n_full - n_suffix, n_suffix)?;
    let e_construct = rel_err(&suffix, &tail);
    println!("suffix-embed construction rel-err: {e_construct:.3e}");
    assert!(
        e_construct < 1e-4,
        "prefill_suffix embeds diverge from full-prefill tail: {e_construct}"
    );

    // ---- Assertion 1: pure chunked forward == full forward (bf16 tolerance). ----
    // Split the SAME embeddings at the suffix boundary; forward whole vs two chunks.
    {
        let mut cache_whole = model.make_cache(full_embeds.dtype(), &device)?;
        let h_whole = model.forward_embeds_debug(&full_embeds, 0, &mut cache_whole)?;
        let h_whole_tail = h_whole.narrow(1, n_full - n_suffix, n_suffix)?;

        let head = full_embeds.narrow(1, 0, n_full - n_suffix)?;
        let mut cache_split = model.make_cache(full_embeds.dtype(), &device)?;
        let _ = model.forward_embeds_debug(&head, 0, &mut cache_split)?;
        let h_split_tail =
            model.forward_embeds_debug(&suffix, n_full - n_suffix, &mut cache_split)?;

        let e_forward = rel_err(&h_split_tail, &h_whole_tail);
        println!("chunked-vs-full forward rel-err: {e_forward:.3e}");
        // Tolerance calibration: identical math, different GEMM shapes. Measured noise
        // is 1.1e-2 on CPU bf16 and 2.1e-2 on Metal bf16 (shape-dependent tiling and
        // accumulation order over 16 layers). A real continuation bug — zeroed conv
        // state, misaligned positions — measures O(0.5–2.0). 5e-2 sits far above the
        // noise floor and far below any bug; the text-token equality below is the
        // exactness guard.
        assert!(
            e_forward < 5e-2,
            "chunked continuation forward diverges from full forward: {e_forward}"
        );
    }

    // ---- Assertion 3: generation — suffix-cache turn vs full-re-prefill turn. ----
    let (mut toks_b, mut mods_b) = (Vec::new(), Vec::new());
    let mut pos_b = cursor.positions;
    model.generate_with_cache(
        &mut cache,
        &mut pos_b,
        suffix,
        &params,
        &AtomicBool::new(false),
        collect_tokens(&mut toks_b, &mut mods_b),
    )?;

    // Path A: reference — full re-prefill of the identical context, fresh cache.
    let (mut toks_a, mut mods_a) = (Vec::new(), Vec::new());
    let mut cache_a = model.make_cache(full_embeds.dtype(), &device)?;
    let mut pos_a = 0usize;
    model.generate_with_cache(
        &mut cache_a,
        &mut pos_a,
        full_embeds,
        &params,
        &AtomicBool::new(false),
        collect_tokens(&mut toks_a, &mut mods_a),
    )?;

    println!(
        "turn2 tokens: full-prefill {} vs suffix-cache {}",
        toks_a.len(),
        toks_b.len()
    );
    assert!(!toks_a.is_empty(), "reference path generated nothing");
    // Only the leading text run is a valid exactness target — see doc comment #3.
    fn first_text_run(toks: &[(i64, Vec<u32>)]) -> Vec<i64> {
        toks.iter()
            .take_while(|(t, _)| *t >= 0)
            .map(|(t, _)| *t)
            .collect()
    }
    let run_a = first_text_run(&toks_a);
    let run_b = first_text_run(&toks_b);
    println!("first text run: full-prefill {run_a:?} vs suffix-cache {run_b:?}");
    assert!(!run_a.is_empty(), "reference path produced no leading text run");
    assert_eq!(run_a, run_b, "first text run diverged");
    Ok(())
}
