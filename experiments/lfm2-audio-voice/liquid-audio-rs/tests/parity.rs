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
    // Pure-candle audio-out: the Kyutai Mimi codec from candle-transformers
    // decodes 8-codebook tokens to a 24 kHz waveform. No torch, no moshi crate.
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let device = Device::Cpu;
    let (_model, proc) = liquid_audio::from_pretrained(Path::new(&dir), DType::F32, &device)?;

    // 16 frames of valid Mimi indices (codebook size 2048), shape (1, 8, T).
    let (k, t) = (8usize, 16usize);
    let codes: Vec<u32> = (0..k * t).map(|i| (i * 37 % 2048) as u32).collect();
    let codes = Tensor::from_vec(codes, (1, k, t), &device)?;

    let wav = proc.mimi_decode(&codes)?;
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

    let in_emb = model.prefill_chat(&chat)?;
    let want = r.get("in_emb").expect("in_emb");
    let e = rel_err(&in_emb, want);
    println!("prefill rel-err: {e:.3e}  rust {:?}  ref {:?}", in_emb.dims(), want.dims());
    assert_eq!(in_emb.dims(), want.dims(), "prefill shape mismatch");
    assert!(e < 2e-2, "prefill parity failed: {e}");
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
