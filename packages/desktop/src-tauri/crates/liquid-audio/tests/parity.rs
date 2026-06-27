//! Numerical parity vs the Python `liquid_audio` front-end.
//!
//! Two tiers:
//! - `mel_parity` needs ONLY a tiny config.json (no weights): it dumps the real
//!   NeMo mel featurizer and compares. Run with:
//!     python parity/dump_mel_reference.py
//!     cargo test --test parity mel_parity -- --ignored --nocapture
//! - `front_end_parity` additionally needs the model weights + conformer dump:
//!     python parity/dump_reference.py /path/to/model parity/golden
//!     LFM_MODEL_DIR=/path/to/model cargo test --test parity -- --ignored --nocapture

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};

use liquid_audio::model::conformer::processor::FilterbankFeatures;
use liquid_audio::processor::PreprocessorConfig;

fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let diff = (&a - &b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    let scale = b.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6);
    diff / scale
}

#[test]
#[ignore = "needs parity/golden/mel_refs.safetensors (run dump_mel_reference.py)"]
fn mel_parity() -> anyhow::Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let refs_path = manifest.join("parity/golden/mel_refs.safetensors");
    let cfg_path = manifest.join("parity/cfg/config.json");
    let device = Device::Cpu;

    let refs = candle_core::safetensors::load(&refs_path, &device)?;
    let wav = refs.get("wav").expect("wav in refs").clone();
    let mel_ref = refs.get("mel").expect("mel in refs").clone();

    // Build the featurizer from the same config block the Python side used.
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let prep: PreprocessorConfig = serde_json::from_value(config["preprocessor"].clone())?;
    let fb = FilterbankFeatures::new(prep.mel_config(), &device)?;

    let mel = fb.forward(&wav)?;
    println!("mel rust shape {:?}  ref shape {:?}", mel.dims(), mel_ref.dims());
    assert_eq!(mel.dims(), mel_ref.dims(), "mel shape mismatch");

    let err = rel_err(&mel, &mel_ref);
    println!("mel rel-err: {err:.3e}");
    assert!(err < 5e-3, "mel parity failed: {err}");
    Ok(())
}

/// Detector for the off-path `exact_pad=True` STFT branch (`center=False` + an
/// explicit `(n_fft - hop)//2` signal pad before preemph, then a timemask). The
/// LFM2.5-Audio config uses `center=True`, so this is exercised only by forcing
/// `exact_pad` on. Golden: `dump_mel_reference.py` → `mel_refs_exactpad`.
#[test]
#[ignore = "needs parity/golden/mel_refs_exactpad.safetensors (run dump_mel_reference.py)"]
fn mel_exact_pad_parity() -> anyhow::Result<()> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let refs_path = manifest.join("parity/golden/mel_refs_exactpad.safetensors");
    let cfg_path = manifest.join("parity/cfg/config.json");
    let device = Device::Cpu;

    let refs = candle_core::safetensors::load(&refs_path, &device)?;
    let wav = refs.get("wav").expect("wav in refs").clone();
    let mel_ref = refs.get("mel").expect("mel in refs").clone();

    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let prep: PreprocessorConfig = serde_json::from_value(config["preprocessor"].clone())?;
    let mut mc = prep.mel_config();
    mc.exact_pad = true; // off-path branch; the config omits it (defaults False)
    let fb = FilterbankFeatures::new(mc, &device)?;

    let mel = fb.forward(&wav)?;
    println!("exact_pad mel rust {:?}  ref {:?}", mel.dims(), mel_ref.dims());
    assert_eq!(mel.dims(), mel_ref.dims(), "exact_pad mel shape mismatch (center=False framing)");

    let err = rel_err(&mel, &mel_ref);
    println!("exact_pad mel rel-err: {err:.3e}");
    assert!(err < 5e-3, "exact_pad mel parity failed: {err}");
    Ok(())
}

/// Detector for the off-path `use_pytorch_sdpa=True` rel-pos attention branch.
///
/// That branch pre-scales `matrix_bd` by `1/√d`, bakes the mask in as an additive
/// `-INF`, and lets `scaled_dot_product_attention` fold in `q_u·kᵀ/√d` — algebraically
/// the SAME `softmax((matrix_ac+matrix_bd)/√d)·v` the manual path computes (its
/// all-masked-row zeroing matches `forward_attention`'s post-softmax `masked_fill`).
/// candle's fused `ops::sdpa` is no_bwd (would sever training grads, cf. d2f4a80), so
/// the manual differentiable path is the faithful translation of BOTH branches. The
/// golden dumps Python's `use_pytorch_sdpa=True` output on shared weights (the Python
/// True-vs-False diff is 1.5e-7); this asserts the Rust manual port reproduces it.
/// Golden: `parity/dump_mha_sdpa.py`.
#[test]
#[ignore = "needs parity/golden/mha_sdpa_refs.safetensors (run dump_mha_sdpa.py)"]
fn rel_pos_attention_sdpa_parity() -> anyhow::Result<()> {
    use liquid_audio::model::conformer::mha::RelPositionMultiHeadAttention;
    use std::collections::HashMap;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/mha_sdpa_refs.safetensors"), &device)?;
    let q = g.get("q").expect("q").clone();
    let pos_emb = g.get("pos_emb").expect("pos_emb").clone();
    let out_sdpa = g.get("out_sdpa").expect("out_sdpa").clone();
    let out_manual = g.get("out_manual").expect("out_manual").clone();

    // weights → VarBuilder (strip the "w." dump prefix → linear_q/k/v/out, linear_pos…).
    let mut ws = HashMap::new();
    for (k, v) in g.iter() {
        if let Some(name) = k.strip_prefix("w.") {
            ws.insert(name.to_string(), v.clone());
        }
    }
    let vb = candle_nn::VarBuilder::from_tensors(ws, DType::F32, &device);
    let att = RelPositionMultiHeadAttention::new(8, 512, true, vb)?;

    let out = att.forward(&q, &q, &q, None, &pos_emb)?;
    let e_sdpa = rel_err(&out, &out_sdpa);
    let e_manual = rel_err(&out, &out_manual);
    println!("rust rel-pos attn vs Python: use_pytorch_sdpa {e_sdpa:.3e}  manual {e_manual:.3e}");
    assert!(e_sdpa < 5e-3, "Rust manual path must reproduce Python use_pytorch_sdpa=True: {e_sdpa}");
    assert!(e_manual < 5e-3, "Rust vs Python manual: {e_manual}");
    Ok(())
}

/// END-TO-END greedy generation parity vs Python `generate_interleaved`. This is the
/// only test that exercises the full autoregressive loop — multi-step KV cache (moshi
/// `LfmCache` vs HF `Lfm2HybridConvCache`), text sampling, the depthformer per audio
/// frame, and the interleaved text/audio modality switching — against Python. Greedy ⇒
/// every generated token id must match EXACTLY. Golden: `parity/dump_generate.py`.
#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/{prefill,generate}_refs.safetensors (run dump_generate.py)"]
fn generate_interleaved_parity() -> anyhow::Result<()> {
    use liquid_audio::model::lfm2_audio::{GenParams, GenToken};
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let r = candle_core::safetensors::load(manifest.join("parity/golden/prefill_refs.safetensors"), &device)?;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/generate_refs.safetensors"), &device)?;
    let (model, _proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;

    let in_emb = model.prefill_inputs(
        &r.get("text").unwrap().to_dtype(DType::I64)?,
        &r.get("audio_in").unwrap().to_dtype(DType::F32)?,
        &r.get("audio_in_lens").unwrap().to_dtype(DType::I64)?,
        &r.get("audio_out").unwrap().to_dtype(DType::I64)?,
        &r.get("modality_flag").unwrap().to_dtype(DType::I64)?,
    )?;

    let want_mod = g.get("seq_mod").unwrap().to_dtype(DType::I64)?.to_vec1::<i64>()?;
    let params = GenParams {
        max_new_tokens: want_mod.len(),
        text_temperature: None, text_top_k: None, audio_temperature: None, audio_top_k: None, seed: 0,
    };

    let mut seq_mod: Vec<i64> = Vec::new();
    let mut text_vals: Vec<i64> = Vec::new();
    let mut audio_vals: Vec<Vec<i64>> = Vec::new();
    model.generate_from_embeds(in_emb, &params, |tok| match tok {
        GenToken::Text(id) => {
            seq_mod.push(0);
            text_vals.push(id as i64);
        }
        GenToken::Audio(frame) => {
            seq_mod.push(1);
            audio_vals.push(frame.iter().map(|&c| c as i64).collect());
        }
    })?;

    // exact token-by-token match vs Python greedy.
    assert_eq!(seq_mod, want_mod, "generation modality sequence diverged from Python");
    assert_eq!(text_vals, g.get("text_vals").unwrap().to_dtype(DType::I64)?.to_vec1::<i64>()?, "text token ids diverged from Python");
    let want_audio = g.get("audio_vals").unwrap().to_dtype(DType::I64)?;
    let got_audio = Tensor::from_iter(audio_vals.iter().flatten().copied(), &device)?.reshape(want_audio.dims())?;
    assert_eq!(got_audio.to_vec2::<i64>()?, want_audio.to_vec2::<i64>()?, "audio frame codes diverged from Python");
    println!("generate parity: {} tokens ({} text, {} audio) — all ids match Python exactly",
        seq_mod.len(), text_vals.len(), audio_vals.len());
    Ok(())
}

/// Cache-aware streaming conformer forward (`forward_streaming` = `forward_internal`
/// with caches), verified DIRECTLY against the upstream Python streaming on the same
/// weights + config: output AND all three next caches. Golden:
/// `parity/dump_conformer_streaming.py` (att context `[29,-1]`, initial zero caches).
/// This exercises the whole streaming path — pre-encode + drop_extra_pre_encoded,
/// pos-enc cache_len, `_create_masks` with offset, per-layer KV-cache threading, the
/// depthwise-conv cache, and the next-cache production (`cache_keep`/`cache_drop`).
#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/conformer_streaming_refs.safetensors (run dump_conformer_streaming.py)"]
fn conformer_streaming_parity() -> anyhow::Result<()> {
    use liquid_audio::model::conformer::encoder::{ConformerEncoder, ConformerEncoderConfig};
    use std::collections::HashMap;
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;

    // Conformer weights from the snapshot shards (strip the `conformer.` prefix).
    let mut ws: HashMap<String, Tensor> = HashMap::new();
    for entry in std::fs::read_dir(&dir)? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        let Ok(sd) = candle_core::safetensors::load(&p, &device) else { continue };
        for (k, v) in sd {
            if let Some(rest) = k.strip_prefix("conformer.") {
                ws.insert(rest.to_string(), v.to_dtype(DType::F32)?);
            }
        }
    }
    let cfg = ConformerEncoderConfig {
        feat_in: 128, feat_out: 0, n_layers: 17, d_model: 512, subsampling_factor: 8,
        subsampling_conv_channels: 256, ff_expansion_factor: 4, n_heads: 8, conv_kernel_size: 9,
        xscaling: false, self_attention_model: "rel_pos".to_string(),
    };
    let vb = candle_nn::VarBuilder::from_tensors(ws, DType::F32, &device);
    let mut enc = ConformerEncoder::new(&cfg, vb)?;

    let refs = candle_core::safetensors::load(manifest.join("parity/golden/refs.safetensors"), &device)?;
    let mel = refs.get("mel").expect("mel").to_dtype(DType::F32)?; // (1, 128, 101)
    let g = candle_core::safetensors::load(manifest.join("parity/golden/conformer_streaming_refs.safetensors"), &device)?;

    // Same config as the Python golden: bounded left context [29, -1], initial zero caches.
    enc.set_streaming_att_context([29, -1]);
    let (cch, ctime, clen) = enc.get_initial_cache_state(1, DType::F32, &device, 0)?;
    let (out, out_len, next_ch, next_time, next_len) = enc.forward_streaming(&mel, None, &cch, &ctime, &clen)?;

    let want_out = g.get("out").expect("out");
    assert_eq!(out.dims(), want_out.dims(), "streaming output shape vs Python");
    let e = rel_err(&out, want_out);
    println!("streaming output vs Python: {e:.3e} (shape {:?})", out.dims());
    assert!(e < 5e-3, "streaming conformer output vs Python: {e}");

    // next caches: shape + values (the channel cache holds the cached frames; the conv
    // cache is empty for this config — cache_drop_size 51 > chunk).
    let want_ch = g.get("next_channel").expect("next_channel");
    assert_eq!(next_ch.dims(), want_ch.dims(), "next_channel shape vs Python");
    let e_ch = rel_err(&next_ch, want_ch);
    println!("next_channel vs Python: {e_ch:.3e}");
    assert!(e_ch < 5e-3, "next_channel vs Python: {e_ch}");
    assert_eq!(next_time.dims(), g.get("next_time").expect("next_time").dims(), "next_time shape vs Python");

    // out_len / next_len exact (integers; next_len is negative for this config).
    assert_eq!(out_len.to_vec1::<i64>()?, g.get("out_len").unwrap().to_dtype(DType::I64)?.to_vec1::<i64>()?, "out_len vs Python");
    assert_eq!(next_len.to_vec1::<i64>()?, g.get("next_len").unwrap().to_dtype(DType::I64)?.to_vec1::<i64>()?, "next_len vs Python");
    println!("out_len {:?} next_len {:?} — match Python", out_len.to_vec1::<i64>()?, next_len.to_vec1::<i64>()?);
    Ok(())
}

/// Detector for the off-path ConvSubsampling schemes (vggnet / striding / the conv1d
/// pair). The model is `dw_striding`; these are alternative architectures verified
/// against torch on shared weights. The golden applies the REAL conv layers directly
/// (`parity/dump_subsampling_schemes.py`) because the upstream masking wrapper is
/// conv2d-only and breaks on MaxPool2d / Conv1d — the masking it skips is a no-op at
/// full length, so the conv/pool/linear ops are the faithful reference.
#[test]
#[ignore = "needs parity/golden/subsampling_schemes_refs.safetensors (run dump_subsampling_schemes.py)"]
fn subsampling_schemes_parity() -> anyhow::Result<()> {
    use liquid_audio::model::conformer::subsampling::ConvSubsampling;
    use std::collections::HashMap;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/subsampling_schemes_refs.safetensors"), &device)?;
    let x = g.get("x").expect("x").clone(); // (B, T, feat_in)

    // factor=8, feat_in=128, feat_out=512, conv_channels=256 (matches the dump).
    // (golden_key, scheme, is_causal)
    let cases: [(&str, &str, bool); 6] = [
        ("vggnet", "vggnet", false),
        ("striding", "striding", false),
        ("striding_conv1d", "striding_conv1d", false),
        ("striding_conv1d_causal", "striding_conv1d", true),
        ("dw_striding_conv1d", "dw_striding_conv1d", false),
        ("dw_striding", "dw_striding", false),
    ];
    for (key, name, is_causal) in cases {
        let prefix = format!("{key}.w.");
        let mut ws = HashMap::new();
        for (k, v) in g.iter() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                ws.insert(rest.to_string(), v.clone());
            }
        }
        let vb = candle_nn::VarBuilder::from_tensors(ws, DType::F32, &device);
        let sub = ConvSubsampling::new_scheme(name, 8, 128, 512, 256, is_causal, vb)?;
        let out = sub.forward(&x)?;
        let want = g.get(&format!("{key}.out")).expect("out");
        assert_eq!(out.dims(), want.dims(), "{key} out shape vs Python");
        let e = rel_err(&out, want);
        println!("{key:24} rust {:?}  rel-err {e:.3e}", out.dims());
        assert!(e < 5e-3, "{key} subsampling-scheme parity vs torch: {e}");
    }
    Ok(())
}

/// Detector for `_create_masks` (the limited/chunked-context attention masks). The
/// offline path uses unlimited context (all-zero masks ⇒ `None`), so the `triu`/`tril`
/// band + `chunked_limited` logic is otherwise unexercised. Golden runs the REAL
/// upstream method (extracted via `ast`) for several configs — `parity/dump_create_masks.py`.
#[test]
#[ignore = "needs parity/golden/create_masks_refs.safetensors (run dump_create_masks.py)"]
fn create_masks_parity() -> anyhow::Result<()> {
    use liquid_audio::model::conformer::encoder::ConformerEncoder;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/create_masks_refs.safetensors"), &device)?;
    let plen = g.get("padding_length").expect("padding_length").clone();
    let off = g.get("offset").expect("offset").clone();
    let m = 8usize;

    // Mirrors dump_create_masks.py exactly (name, self_attention_model, style, [left,right], use_offset).
    let cases: [(&str, &str, &str, [i64; 2], bool); 6] = [
        ("regular_unlimited", "rel_pos", "regular", [-1, -1], false),
        ("regular_band11", "rel_pos", "regular", [1, 1], false),
        ("regular_left2", "rel_pos", "regular", [2, -1], false),
        ("chunked_c4", "rel_pos", "chunked_limited", [4, 3], false),
        ("chunked_rightunlim", "rel_pos", "chunked_limited", [2, -1], false),
        ("regular_band11_offset", "rel_pos", "regular", [1, 1], true),
    ];
    let diff = |a: &Tensor, b: &Tensor| -> anyhow::Result<i64> {
        Ok((a.to_dtype(DType::I64)? - b.to_dtype(DType::I64)?)?.abs()?.sum_all()?.to_scalar::<i64>()?)
    };
    for (name, sam, style, acs, use_off) in cases {
        let offset = if use_off { Some(&off) } else { None };
        let (pad_mask, att_mask) = ConformerEncoder::build_masks(sam, style, acs, &plen, m, offset, &device)?;
        let want_pm = g.get(&format!("{name}.pad_mask")).expect("pad_mask");
        assert_eq!(diff(&pad_mask, want_pm)?, 0, "{name} pad_mask mismatch vs Python");
        let want_am = g.get(&format!("{name}.att_mask")).expect("att_mask");
        let am = att_mask.expect("att_mask present");
        assert_eq!(am.dims(), want_am.dims(), "{name} att_mask shape");
        assert_eq!(diff(&am, want_am)?, 0, "{name} att_mask mismatch vs Python");
        println!("{name}: pad+att masks bit-exact vs upstream _create_masks");
    }
    Ok(())
}

/// Detector for the base (abs_pos) `MultiHeadAttention.forward`. The encoder uses the
/// rel-pos subclass, so this standard scaled-dot-product path is otherwise unexercised;
/// it is the attention the `abs_pos` `ConformerLayer` variant dispatches to. Golden:
/// `parity/dump_mha_sdpa.py` (`mha_abs_refs`).
#[test]
#[ignore = "needs parity/golden/mha_abs_refs.safetensors (run dump_mha_sdpa.py)"]
fn abs_attention_parity() -> anyhow::Result<()> {
    use liquid_audio::model::conformer::mha::MultiHeadAttention;
    use std::collections::HashMap;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/mha_abs_refs.safetensors"), &device)?;
    let q = g.get("q").expect("q").clone();
    let out_ref = g.get("out").expect("out").clone();

    let mut ws = HashMap::new();
    for (k, v) in g.iter() {
        if let Some(name) = k.strip_prefix("w.") {
            ws.insert(name.to_string(), v.clone());
        }
    }
    let vb = candle_nn::VarBuilder::from_tensors(ws, DType::F32, &device);
    let att = MultiHeadAttention::new(8, 512, true, vb)?;

    let out = att.forward(&q, &q, &q, None)?;
    let e = rel_err(&out, &out_ref);
    println!("rust abs (base) MHA vs Python: {e:.3e}");
    assert!(e < 5e-3, "Rust base MHA vs Python: {e}");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/conformer_stages.safetensors"]
fn conformer_stages_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let stages = candle_core::safetensors::load(manifest.join("parity/golden/conformer_stages.safetensors"), &device)?;
    let refs = candle_core::safetensors::load(manifest.join("parity/golden/refs.safetensors"), &device)?;
    let mel = refs.get("mel").expect("mel").clone();

    let (model, _) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let conv_out = model.conformer_sub_conv(&mel)?;
    let (sub, posx, posemb, layer0, final_out) = model.conformer_stages(&mel)?;

    // Stage-by-stage localization: subsampling conv stack → subsampling out →
    // pos-encoded x → rel pos-emb → after layer 0 → final.
    for (name, got) in [("conv_out", &conv_out), ("sub", &sub), ("posx", &posx), ("posemb", &posemb), ("layer0", &layer0), ("final", &final_out)] {
        let want = stages.get(name).unwrap_or_else(|| panic!("{name} missing in ref"));
        let e = rel_err(got, want);
        println!("{name:8} rust {:?}  ref {:?}  rel-err {e:.3e}", got.dims(), want.dims());
        assert!(e < 5e-3, "{name} parity failed: {e}");
    }
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR (loads the Mimi weights shipped in the repo)"]
fn mimi_decode_smoke() -> anyhow::Result<()> {
    // Pure-candle audio-out through the Mimi codec (moshi crate), decoding
    // 8-codebook tokens to 24 kHz. No torch. Resolved via `proc.mimi()` — the Mimi
    // codec specifically — so this exercises Mimi on BOTH v1 and full LFM2.5
    // snapshots (where the decode backend is the LFM2 detokenizer, not Mimi).
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let (_model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let mimi = proc.mimi().expect("Mimi codec (tokenizer-…checkpoint125.safetensors)");

    // 16 frames of valid Mimi indices (codebook size 2048), shape (1, 8, T).
    let (k, t) = (8usize, 16usize);
    let codes: Vec<u32> = (0..k * t).map(|i| (i * 37 % 2048) as u32).collect();
    let codes = Tensor::from_vec(codes, (1, k, t), &device)?;

    let wav = mimi.decode(&codes)?;
    let flat = wav.flatten_all()?.to_dtype(DType::F32)?;
    let n = flat.dims1()?;
    let max = flat.abs()?.max(0)?.to_scalar::<f32>()?;
    println!("mimi decode: codes {:?} -> waveform {:?}  ({} samples, max|amp| {:.4})", codes.dims(), wav.dims(), n, max);
    // Mimi: 12.5 Hz frame rate, 24 kHz → 1920 samples/frame.
    assert_eq!(n, t * 1920, "unexpected sample count");
    assert!(max.is_finite() && max > 0.0, "waveform is empty/NaN");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR (loads the Mimi weights shipped in the repo)"]
fn mimi_streaming_decode_smoke() -> anyhow::Result<()> {
    // The real-time path: moshi's streaming `decode_step` decodes one generated
    // frame at a time (keeping codec state), instead of a one-shot batch decode —
    // exactly what the Python demo does inside `mimi.streaming(1)`. No torch.
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let (_model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let mimi = proc.mimi().expect("Mimi codec (tokenizer-…checkpoint125.safetensors)");

    let (k, t) = (8usize, 16usize);
    let codes: Vec<u32> = (0..k * t).map(|i| (i * 37 % 2048) as u32).collect();
    let codes = Tensor::from_vec(codes, (1, k, t), &device)?;

    // Stream: reset at the turn boundary, then feed each (1, 8, 1) frame.
    mimi.reset_stream();
    let mut chunks: Vec<Tensor> = Vec::new();
    let mut warmup_none = 0usize;
    for ti in 0..t {
        let frame = codes.narrow(2, ti, 1)?; // (1, codebooks, 1)
        match mimi.decode_step(&frame)? {
            Some(chunk) => chunks.push(chunk.flatten_all()?.to_dtype(DType::F32)?),
            None => warmup_none += 1,
        }
    }
    assert!(!chunks.is_empty(), "streaming produced no audio chunks");
    let stream = Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 0)?;
    let n_stream = stream.dims1()?;
    let max = stream.abs()?.max(0)?.to_scalar::<f32>()?;
    println!(
        "mimi STREAMING decode: {t} frames -> {n_stream} samples ({warmup_none} warmup-None, {} emitting frames), max|amp| {:.4}",
        t - warmup_none,
        max
    );
    assert!(max.is_finite() && max > 0.0, "streaming waveform empty/NaN");
    // Each emitting frame yields 1920 samples (12.5 Hz @ 24 kHz).
    assert_eq!(n_stream, (t - warmup_none) * 1920, "unexpected streaming sample count");
    Ok(())
}

/// The data mapper's `_encode_audio_out` path: `processor.mimi.encode(wav)`.
///
/// On a full LFM2.5 snapshot the DECODE backend is the LFM2 detokenizer, so this
/// fails unless the Mimi codec is loaded SEPARATELY (the bug: a single shared
/// backend made `mimi()` return the decode-only detokenizer → `mimi_encode` errors,
/// breaking dataset preprocessing). Runs on CPU, where moshi's `CodebookEncode`
/// CustomOp is supported.
#[test]
#[ignore = "needs LFM_MODEL_DIR (loads the Mimi weights shipped in the repo)"]
fn mimi_encode_smoke() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let (_model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let sr = proc.mimi_sample_rate().expect("Mimi sample rate (codec must be loaded)");

    // 0.5 s sine at the codec's input rate, shape (B=1, C=1, L) — what
    // `_encode_audio_out` feeds after `wav.unsqueeze(0)`.
    let n = (sr / 2) as usize;
    let wav: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let wav = Tensor::from_vec(wav, (1, 1, n), &device)?;

    let codes = proc.mimi_encode(&wav)?; // (1, codebooks_all, T)
    println!("mimi encode: {sr} Hz wav (1,1,{n}) -> codes {:?}", codes.dims());
    assert_eq!(codes.dim(0)?, 1, "batch dim");
    assert!(codes.dim(1)? >= 8, "expected >=8 codebooks, got {}", codes.dim(1)?);
    assert!(codes.dim(2)? > 0, "no frames encoded");
    // Codes must be valid codebook indices (< 2048) so `_encode_audio_out` can keep
    // the first 8 rows and append the EOAudio sentinel.
    let max = codes.flatten_all()?.to_dtype(DType::U32)?.max(0)?.to_scalar::<u32>()?;
    assert!(max < 2048, "code {max} out of Mimi codebook range");
    Ok(())
}

/// `from_pretrained_trainable` must cast the STORED checkpoint dtype to the
/// requested dtype before `Var::set`.
///
/// The snapshot is bf16; CPU training needs F32 (candle has no CPU bf16 matmul).
/// `Var::set` is a same-dtype storage copy, so without the cast a bf16 tensor into
/// an F32 Var errors on load. This asserts every trainable Var is F32 after load.
#[test]
#[ignore = "needs LFM_MODEL_DIR (allocates the full param set as F32 — ~6 GB)"]
fn trainable_load_upcasts_to_f32() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let tl = liquid_audio::loader::from_pretrained_trainable(Path::new(&dir), DType::F32, &device)?;
    let vars = tl.varmap.all_vars();
    assert!(!vars.is_empty(), "no trainable vars loaded");
    let non_f32 = vars.iter().filter(|v| v.dtype() != DType::F32).count();
    println!("trainable load: {} vars, {} not F32", vars.len(), non_f32);
    assert_eq!(non_f32, 0, "{non_f32} vars were not upcast to F32 (Var::set skipped the dtype cast)");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/prefill_refs.safetensors"]
fn prefill_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let r = candle_core::safetensors::load(manifest.join("parity/golden/prefill_refs.safetensors"), &device)?;

    let (model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    // Build a ChatState from the Python-dumped raw fields (identical inputs, no
    // template re-tokenization). Fields are u32 in the port; the dump is i64.
    let mut chat = liquid_audio::ChatState::new(&proc, 8)?;
    chat.text = r.get("text").unwrap().to_dtype(DType::U32)?;
    chat.audio_in = r.get("audio_in").unwrap().to_dtype(DType::F32)?;
    chat.audio_in_lens = r.get("audio_in_lens").unwrap().to_dtype(DType::U32)?;
    chat.audio_out = r.get("audio_out").unwrap().to_dtype(DType::U32)?;
    chat.modality_flag = r.get("modality_flag").unwrap().to_dtype(DType::U32)?;

    // The reference has TWO audio-in segments of different lengths, so the Python
    // side pads them into a batch + length-masks; the Rust side encodes each
    // segment individually. Matching here proves per-segment encode ≡ padded-batch
    // (the conformer masking exists precisely to make them equal).
    let in_emb = model.prefill_chat(&chat)?;
    let want = r.get("in_emb").expect("in_emb");
    let e = rel_err(&in_emb, want);
    println!("prefill rel-err: {e:.3e}  rust {:?}  ref {:?}", in_emb.dims(), want.dims());
    assert_eq!(in_emb.dims(), want.dims(), "prefill shape mismatch");
    assert!(e < 2e-2, "prefill parity failed: {e}");
    Ok(())
}

/// The ConformerEncoder streaming/export inventory methods on the REAL encoder.
///
/// Off the offline forward path, but ported 1:1 (not stubbed) — this exercises the
/// tensor-producing methods on the loaded encoder: cache allocation shapes, the
/// streaming output trim, the export dummy inputs, and the deployment name lists.
#[test]
#[ignore = "needs LFM_MODEL_DIR"]
fn conformer_streaming_inventory() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let (model, _proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let enc = model.conformer();

    // get_initial_cache_state(batch=2, max_dim=0) → zeros of the documented shapes.
    let (cc, ct, cl) = enc.get_initial_cache_state(2, DType::F32, &device, 0)?;
    println!("cache shapes: last_channel {:?} last_time {:?} len {:?}", cc.dims(), ct.dims(), cl.dims());
    assert_eq!(cc.dim(1)?, 2, "last_channel batch");
    assert_eq!(ct.dim(1)?, 2, "last_time batch");
    assert_eq!(cl.dims(), [2], "cache_last_channel_len shape");
    // last_channel_cache_size = max_context (att_context [-1,-1]) = 10000.
    assert_eq!(cc.dim(2)?, enc.streaming_cfg().last_channel_cache_size as usize);

    // streaming_post_process: 5-element (Some cache) trims encoded to valid_out_len.
    let valid = enc.streaming_cfg().valid_out_len as usize;
    // d (channel dim) is arbitrary — streaming_post_process trims the time dim only.
    let (b, d, t) = (1usize, 8usize, valid + 5);
    let encoded = Tensor::zeros((b, d, t), DType::F32, &device)?;
    let elen = Tensor::from_vec(vec![t as i64], (b,), &device)?;
    let (enc_out, elen_out, cache_out) = enc.streaming_post_process(encoded, elen, Some(cc.clone()), false)?;
    assert_eq!(enc_out.dim(2)?, valid, "encoded trimmed to valid_out_len");
    assert_eq!(elen_out.to_vec1::<i64>()?, vec![valid as i64], "length clamped");
    assert!(cache_out.is_some());
    // 2-element (None cache) form is returned unchanged.
    let passthrough = Tensor::zeros((b, d, t), DType::F32, &device)?;
    let plen = Tensor::from_vec(vec![t as i64], (b,), &device)?;
    let (pt, _pl, pc) = enc.streaming_post_process(passthrough, plen, None, false)?;
    assert_eq!(pt.dim(2)?, t, "no-cache form unchanged");
    assert!(pc.is_none());

    // input_example (export_cache_support=false default) → (signal, length).
    let ex = enc.input_example(1, 256, &device)?;
    assert_eq!(ex.len(), 2, "non-cache export = (signal, length)");
    assert_eq!(ex[0].dim(2)?, 256, "signal time dim");

    // deployment name lists (export_cache_support=false).
    assert_eq!(enc.disabled_deployment_input_names().len(), 3);
    assert_eq!(enc.disabled_deployment_output_names().len(), 3);
    Ok(())
}

/// Training gradients must reach the attention + norm params.
///
/// candle's fused `softmax_last_dim` / `rope`(`_i`) / `RmsNorm`+`LayerNorm`::forward
/// are `apply_op*_no_bwd` — they SEVER autograd. Using them in the trainable graph
/// silently gave the q/k projections and every norm weight ZERO gradient (a full
/// forward+backward landed at 152/912 vars with a gradient). After switching to the
/// differentiable variants (`softmax`, `rope_slow`, `rope_i_slow`, `layer_norm_slow`,
/// the differentiable `RmsNorm`), a backward must reach those params.
///
/// Validated per subsystem (conformer encode, backbone) AND end-to-end via the real
/// training path (`forward(batch) -> loss.backward()`). The full path additionally
/// covers the conformer subsampling stem: candle's conv2d stride-2 BACKWARD errors on
/// odd input spatial dims (out=N is ambiguous, the grad-input assumes the even size),
/// and the subsampling hits odd time dims (101->51), so the full backward used to fail
/// with `[..,32]` vs `[..,31]`. `subsampling::pad_even_hw` (forward-identical even-pad
/// before strided convs) fixes it; this test is the regression guard — it both runs the
/// full backward and asserts the subsampling conv weights receive a gradient.
#[test]
#[ignore = "needs LFM_MODEL_DIR (full-model load + backward, ~6 GB F32)"]
fn training_gradients_reach_attention_and_norms() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let tl = liquid_audio::loader::from_pretrained_trainable(Path::new(&dir), DType::F32, &device)?;
    let data = tl.varmap.data().lock().unwrap();

    // helper: backward a scalar, return the params (by prefix) that got NO gradient.
    let severed = |loss: &Tensor, prefix: &str| -> anyhow::Result<(usize, usize, Vec<String>)> {
        let grads = loss.backward()?;
        let (mut have, mut total, mut miss) = (0usize, 0usize, Vec::new());
        for (name, var) in data.iter() {
            if !name.starts_with(prefix) {
                continue;
            }
            total += 1;
            // running_mean/var are non-trainable buffers; embed_tokens is bypassed by
            // forward_embeds (pre-embedded input) — neither gets a gradient here.
            if name.contains("running_") || name == "lfm.embed_tokens.weight" {
                continue;
            }
            if grads.get(var).is_some() {
                have += 1;
            } else {
                miss.push(name.clone());
            }
        }
        Ok((have, total, miss))
    };

    // conformer: encode a mel segment → every conformer param must get a gradient
    // (LayerNorm/attention were severed before the fix).
    let mel = Tensor::randn(0f32, 1f32, (1, 128, 120), &device)?;
    let (h, t, mut miss) = severed(&tl.model.conformer_encode(&mel)?.sum_all()?, "conformer.")?;
    println!("conformer grad coverage: {h}/{t} (excl. running buffers)");
    miss.sort();
    assert!(miss.is_empty(), "conformer params with NO gradient (autograd severed): {:?}", &miss[..miss.len().min(8)]);

    // backbone: every lfm param must get a gradient (RmsNorm/rope/softmax were severed).
    let embeds = Tensor::randn(0f32, 1f32, (1, 24, 2048), &device)?;
    let (h2, t2, mut miss2) = severed(&tl.model.backbone_forward_embeds(&embeds)?.sum_all()?, "lfm.")?;
    println!("backbone grad coverage: {h2}/{t2}");
    miss2.sort();
    assert!(miss2.is_empty(), "backbone params with NO gradient: {:?}", &miss2[..miss2.len().min(8)]);

    // Full training path: forward(batch) -> loss.backward(). This exercises the
    // conformer subsampling stem (per-segment conv2d), the scatter into the backbone,
    // and the audio/text heads together. Before pad_even_hw, the strided-conv backward
    // errored on odd time dims (101->51); this section is the regression guard.
    use liquid_audio::model::lfm2_audio::LFM2AudioModelInput;
    let r = candle_core::safetensors::load(manifest.join("parity/golden/prefill_refs.safetensors"), &device)?;
    let text = r.get("text").unwrap().to_dtype(DType::I64)?;
    let audio_in = r.get("audio_in").unwrap().to_dtype(DType::F32)?;
    let audio_in_lens = r.get("audio_in_lens").unwrap().to_dtype(DType::I64)?;
    let audio_out = r.get("audio_out").unwrap().to_dtype(DType::I64)?;
    let modality = r.get("modality_flag").unwrap().to_dtype(DType::I64)?;
    let sup = Tensor::ones((1, modality.dim(1)?), DType::U8, &device)?;
    let batch = LFM2AudioModelInput {
        text,
        audio_in,
        audio_in_lens,
        audio_out,
        modality_flag: modality,
        supervision_mask: sup,
    };
    let out = tl.model.forward(&batch)?;
    // backward must NOT error (the conv2d odd-input shape bug) and must reach the
    // subsampling convs — the params the bug previously starved.
    let grads = out.loss.backward()?;
    let (mut sub_have, mut sub_total) = (0usize, 0usize);
    for (name, var) in data.iter() {
        if name.starts_with("conformer.pre_encode.conv.") && name.ends_with("weight") {
            sub_total += 1;
            if grads.get(var).is_some() {
                sub_have += 1;
            }
        }
    }
    println!("full forward+backward OK; subsampling conv grad coverage: {sub_have}/{sub_total}");
    assert!(sub_total > 0, "expected to find subsampling conv weights by name");
    assert_eq!(sub_have, sub_total, "subsampling conv weights with NO gradient after full backward");
    Ok(())
}

/// Batched (B=2) training path vs a PYTHON golden — `logits` + `forward`.
///
/// This is the real detector for the B>1 row-0 bug (unlike the self-consistency test
/// below, which never asks Python anything). The old code read only batch row 0, so
/// its `text_logits` were `(n_text, V)` — HALF of Python's `(2·n_text, V)`; that is a
/// hard shape failure here. Golden: `parity/dump_batched_logits.py` (duplicate of the
/// prefill_refs sample, collated, all-ones supervision).
#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/{prefill,batched_logits}_refs.safetensors"]
fn batched_logits_python_parity() -> anyhow::Result<()> {
    use liquid_audio::model::lfm2_audio::LFM2AudioModelInput;
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let r = candle_core::safetensors::load(manifest.join("parity/golden/prefill_refs.safetensors"), &device)?;
    let g = candle_core::safetensors::load(manifest.join("parity/golden/batched_logits_refs.safetensors"), &device)?;
    let (model, _proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;

    let text = r.get("text").unwrap().to_dtype(DType::I64)?;
    let audio_in = r.get("audio_in").unwrap().to_dtype(DType::F32)?;
    let audio_in_lens = r.get("audio_in_lens").unwrap().to_dtype(DType::I64)?;
    let audio_out = r.get("audio_out").unwrap().to_dtype(DType::I64)?;
    let modality = r.get("modality_flag").unwrap().to_dtype(DType::I64)?;
    let l = modality.dim(1)?;
    let sup = Tensor::ones((1, l), DType::U8, &device)?;
    // B=2 duplicate, lfm2_collator dims (text/audio cat dim=1; modality/sup cat dim=0).
    let b2 = LFM2AudioModelInput {
        text: Tensor::cat(&[&text, &text], 1)?,
        audio_in: Tensor::cat(&[&audio_in, &audio_in], 1)?,
        audio_in_lens: Tensor::cat(&[&audio_in_lens, &audio_in_lens], 0)?,
        audio_out: Tensor::cat(&[&audio_out, &audio_out], 1)?,
        modality_flag: Tensor::cat(&[&modality, &modality], 0)?,
        supervision_mask: Tensor::cat(&[&sup, &sup], 0)?,
    };

    let (tl, _al, tt, _at) = model.logits(&b2)?;
    let want_tl = g.get("text_logits").expect("text_logits");
    // SHAPE first: the row-0 bug yields half the rows → (28,V) vs Python (56,V).
    assert_eq!(tl.dims(), want_tl.dims(), "text_logits shape (row-0 bug → half the supervised positions)");
    let e_tl = rel_err(&tl, want_tl);
    // text labels must match Python exactly (integer).
    let got_tt = tt.to_dtype(DType::I64)?.to_vec1::<i64>()?;
    let want_tt = g.get("text_labels").unwrap().to_dtype(DType::I64)?.to_vec1::<i64>()?;
    assert_eq!(got_tt, want_tt, "batched text labels vs Python");

    // forward loss vs Python.
    let out = model.forward(&b2)?;
    let e_loss = rel_err(&out.loss.reshape((1,))?, g.get("loss").unwrap());
    println!("batched vs Python: text_logits {:?} rel-err {e_tl:.3e}; loss rel-err {e_loss:.3e}", tl.dims());
    assert!(e_tl < 2e-2, "batched text_logits vs Python: {e_tl}");
    assert!(e_loss < 2e-2, "batched loss vs Python: {e_loss}");
    Ok(())
}

/// Batched (B>1) self-consistency for `prefill_inputs` / `logits`.
///
/// A B=2 batch of two IDENTICAL samples must yield prefill/logits equal to the
/// B=1 result duplicated. This verifies the batched scatter/index paths WITHOUT a
/// Python reference: any model staleness cancels because both sides share weights.
/// Batch shapes mirror `lfm2_collator` (text/audio_in/audio_out cat dim=1;
/// modality/supervision cat dim=0).
#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/prefill_refs.safetensors"]
fn batched_prefill_logits_self_consistency() -> anyhow::Result<()> {
    use liquid_audio::model::lfm2_audio::LFM2AudioModelInput;
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let r = candle_core::safetensors::load(manifest.join("parity/golden/prefill_refs.safetensors"), &device)?;
    let (model, _proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;

    let text = r.get("text").unwrap().to_dtype(DType::I64)?; // (1, n)
    let audio_in = r.get("audio_in").unwrap().to_dtype(DType::F32)?; // (128, f)
    let audio_in_lens = r.get("audio_in_lens").unwrap().to_dtype(DType::I64)?; // (s,)
    let audio_out = r.get("audio_out").unwrap().to_dtype(DType::I64)?; // (C, a)
    let modality = r.get("modality_flag").unwrap().to_dtype(DType::I64)?; // (1, L)
    let l = modality.dim(1)?;
    let sup = Tensor::ones((1, l), DType::U8, &device)?; // supervise all positions

    let b1 = LFM2AudioModelInput {
        text: text.clone(),
        audio_in: audio_in.clone(),
        audio_in_lens: audio_in_lens.clone(),
        audio_out: audio_out.clone(),
        modality_flag: modality.clone(),
        supervision_mask: sup.clone(),
    };
    let b2 = LFM2AudioModelInput {
        text: Tensor::cat(&[&text, &text], 1)?,
        audio_in: Tensor::cat(&[&audio_in, &audio_in], 1)?,
        audio_in_lens: Tensor::cat(&[&audio_in_lens, &audio_in_lens], 0)?,
        audio_out: Tensor::cat(&[&audio_out, &audio_out], 1)?,
        modality_flag: Tensor::cat(&[&modality, &modality], 0)?,
        supervision_mask: Tensor::cat(&[&sup, &sup], 0)?,
    };

    // prefill: (2,L,D); both rows equal the (1,L,D) result.
    let p1 = model.prefill_inputs(&b1.text, &b1.audio_in, &b1.audio_in_lens, &b1.audio_out, &b1.modality_flag)?;
    let p2 = model.prefill_inputs(&b2.text, &b2.audio_in, &b2.audio_in_lens, &b2.audio_out, &b2.modality_flag)?;
    assert_eq!(p2.dims(), [2, l, p1.dim(2)?], "batched prefill shape");
    let (e0, e1) = (rel_err(&p2.i(0)?, &p1.i(0)?), rel_err(&p2.i(1)?, &p1.i(0)?));
    println!("batched prefill rel-err row0 {e0:.3e} row1 {e1:.3e}  dims {:?}", p2.dims());
    assert!(e0 < 1e-5 && e1 < 1e-5, "batched prefill not row-consistent: {e0} {e1}");

    // logits: B=2 == B=1 duplicated along dim 0.
    let (tl1, al1, ttok1, atok1) = model.logits(&b1)?;
    let (tl2, al2, ttok2, atok2) = model.logits(&b2)?;
    let dup = |t: &Tensor| -> anyhow::Result<Tensor> { Ok(Tensor::cat(&[t, t], 0)?) };
    println!(
        "batched logits dims  text {:?}->{:?}  audio {:?}->{:?}",
        tl1.dims(),
        tl2.dims(),
        al1.dims(),
        al2.dims()
    );
    // B=2 row count must be exactly 2× B=1, and the values must match the duplication.
    // (rel_err reduces, so it is undefined on a 0-row tensor — guard it.)
    let check = |name: &str, x1: &Tensor, x2: &Tensor| -> anyhow::Result<()> {
        assert_eq!(x2.dim(0)?, 2 * x1.dim(0)?, "{name}: B=2 rows != 2× B=1");
        if x1.dim(0)? > 0 {
            let e = rel_err(x2, &Tensor::cat(&[x1, x1], 0)?);
            println!("  {name} rel-err {e:.3e}");
            assert!(e < 1e-5, "{name} mismatch: {e}");
        }
        Ok(())
    };
    check("text_logits", &tl1, &tl2)?;
    check("audio_logits", &al1, &al2)?;
    assert_eq!(ttok2.to_vec1::<i64>()?, dup(&ttok1)?.to_vec1::<i64>()?, "batched text labels");
    assert_eq!(atok2.to_vec1::<u32>()?, dup(&atok1)?.to_vec1::<u32>()?, "batched audio labels");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/depthformer_refs.safetensors"]
fn depthformer_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let refs = candle_core::safetensors::load(manifest.join("parity/golden/depthformer_refs.safetensors"), &device)?;
    let embedding = refs.get("embedding").expect("embedding").clone();
    let want: Vec<u32> = refs.get("tokens").expect("tokens").to_dtype(DType::U32)?.to_vec1::<u32>()?;

    let (model, _) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let got = model.audio_frame_greedy(&embedding)?;
    println!("depthformer rust {got:?}  ref {want:?}");
    assert_eq!(got, want, "depthformer greedy tokens differ");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/backbone_refs.safetensors"]
fn backbone_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let device = Device::Cpu;
    let refs = candle_core::safetensors::load(manifest.join("parity/golden/backbone_refs.safetensors"), &device)?;
    let embeds = refs.get("embeds").expect("embeds").clone();

    let (model, _) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;
    let got = model.backbone_forward_embeds(&embeds)?;
    let want = refs.get("backbone").expect("backbone");
    let e = rel_err(&got, want);
    println!("backbone rel-err: {e:.3e}  shape {:?}", got.dims());
    assert!(e < 2e-2, "backbone parity failed: {e}");

    // text head: tied-embedding logits for the last position
    let l = got.dim(1)?;
    let h_last = got.i((0, l - 1))?.contiguous()?;
    let logits = model.text_logits_of(&h_last)?;
    let lt = rel_err(&logits, refs.get("text_logits").expect("text_logits"));
    println!("text_logits rel-err: {lt:.3e}  shape {:?}", logits.dims());
    assert!(lt < 2e-2, "text_logits parity failed: {lt}");
    Ok(())
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/golden/refs.safetensors"]
fn front_end_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let refs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity/golden/refs.safetensors");
    let device = Device::Cpu;

    let refs = candle_core::safetensors::load(&refs_path, &device)?;
    let wav = refs.get("wav").expect("wav in refs").clone();

    // f32 to match the reference dump (dump_reference.py uses dtype=torch.float32).
    let (model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;

    // mel featurizer
    let mel = proc.audio.forward(&wav)?;
    let mel_err = rel_err(&mel, refs.get("mel").expect("mel"));
    println!("mel rel-err: {mel_err:.2e}  shape {:?}", mel.dims());
    assert!(mel_err < 5e-3, "mel parity failed: {mel_err}");

    // conformer encoder
    let enc = model.conformer_encode(&mel)?;
    let enc_err = rel_err(&enc, refs.get("conformer").expect("conformer"));
    println!("conformer rel-err: {enc_err:.2e}  shape {:?}", enc.dims());
    assert!(enc_err < 2e-2, "conformer parity failed: {enc_err}");

    Ok(())
}
