//! Numerical parity vs the Python `liquid_audio` front-end.
//!
//! Ignored by default (needs the model + reference dump). Run with:
//!   python parity/dump_reference.py /path/to/model parity/refs
//!   LFM_MODEL_DIR=/path/to/model cargo test --test parity -- --ignored --nocapture

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

fn rel_err(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap();
    let diff = (&a - &b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
    let scale = b.abs().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6);
    diff / scale
}

#[test]
#[ignore = "needs LFM_MODEL_DIR + parity/refs/refs.safetensors"]
fn front_end_parity() -> anyhow::Result<()> {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let refs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity/refs/refs.safetensors");
    let device = Device::Cpu;

    let refs = candle_core::safetensors::load(&refs_path, &device)?;
    let wav = refs.get("wav").expect("wav in refs").clone();

    let (model, proc) = liquid_audio::from_pretrained(Path::new(&dir), &device)?;

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
