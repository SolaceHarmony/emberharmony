//! Framework-backed safetensors and native-prefill oracles.
//!
//! These tests deliberately live outside the production crate: Candle is used
//! only to measure the preserved reference implementation, never as a runtime
//! model path or a numerical payload boundary.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use liquid_audio_oracle::weights::{ResidentWeights, WeightDType};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn workspace_model_dir() -> PathBuf {
    PathBuf::from("../../experiments/lfm2-audio-voice/model")
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "emberharmony-oracle-safetensors-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap_or_default();
    }
}

struct Entry<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [u64],
    data: &'a [u8],
}

fn write_file(path: &Path, entries: &[Entry<'_>]) {
    let mut root = serde_json::Map::new();
    let mut offset = 0usize;
    for entry in entries {
        let end = offset + entry.data.len();
        root.insert(
            entry.name.into(),
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let data = entries
        .iter()
        .flat_map(|entry| entry.data.iter().copied())
        .collect::<Vec<_>>();
    let mut header = serde_json::to_vec(&serde_json::Value::Object(root)).unwrap();
    header.resize((header.len() + 7) & !7, b' ');
    let mut file = Vec::with_capacity(8 + header.len() + data.len());
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

#[test]
#[ignore = "requires LFM_MODEL_DIR and the real LFM2.5-Audio checkpoint"]
fn native_audio_prefill_matches_discrete_for_the_same_embedding() {
    // The native audio-in prefill (`embed_kind == 2`, a provided embedding VIEW)
    // must produce the same backbone state as the discrete text path when fed
    // the exact same resident embedding row.
    let dir = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .expect("LFM_MODEL_DIR must name the real LFM2.5-Audio checkpoint"),
    );
    let model = liquid_audio_oracle::NativeModel::open(&dir).expect("native model");
    let hidden = model.info().expect("model info").hidden as usize;

    let resident = ResidentWeights::open(&dir.join("model.safetensors")).expect("resident image");
    let embed = resident
        .image()
        .find("lfm.embed_tokens.weight")
        .expect("embed_tokens");
    let bytes = embed.data();
    let token = 100usize;
    let row = bytes[token * hidden * 2..(token + 1) * hidden * 2]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    assert_eq!(row.len(), hidden);
    let probe = 5u32;
    let config = || liquid_audio_oracle::NativeConversationConfig {
        seed: Some(7),
        temperature: None,
        top_k: None,
    };

    let mut discrete = model.conversation(config()).expect("discrete conversation");
    discrete
        .step(&[token as u32], liquid_audio_oracle::EmbeddingKind::Text)
        .expect("discrete prefill");
    let discrete_token = discrete
        .step(&[probe], liquid_audio_oracle::EmbeddingKind::Text)
        .expect("discrete probe")
        .sampled_token;

    let mut audio = model.conversation(config()).expect("audio conversation");
    assert_eq!(audio.prefill_audio(&row).expect("audio prefill"), 1);
    let audio_token = audio
        .step(&[probe], liquid_audio_oracle::EmbeddingKind::Text)
        .expect("audio probe")
        .sampled_token;
    assert_eq!(audio_token, discrete_token);
}

#[test]
fn rust_oracle_records_a_real_compatibility_copy() {
    let temp = Temp::new();
    let path = temp.0.join("weights.safetensors");
    let values = [1.25f32, -3.5f32];
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        &[Entry {
            name: "model.weight",
            dtype: "F32",
            shape: &[2],
            data: &data,
        }],
    );

    let resident = ResidentWeights::open(&path).unwrap();
    assert_eq!(resident.dtype(), candle_core::DType::F32);
    assert_eq!(resident.image().len(), 1);
    let view = resident.image().find("model.weight").unwrap();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(
        view.data_ptr() as usize,
        resident.image().base() as usize + view.offset() as usize
    );

    let witness = resident.clone();
    let builder = resident.candle_builder(&candle_core::Device::Cpu);
    drop(resident);
    let tensor = builder.get((2,), "model.weight").unwrap();
    assert_eq!(tensor.to_vec1::<f32>().unwrap(), values);
    assert_eq!(
        witness.compatibility_copies(),
        liquid_audio_oracle::weights::CompatibilityCopies {
            tensors: 1,
            bytes: data.len() as u64,
        }
    );
}

#[test]
#[ignore = "needs the repository LFM2.5-Audio fixture and about 3 GB of free memory"]
fn real_model_checkpoint_loads_without_candle_materialization() {
    let dir = workspace_model_dir();
    assert!(dir.join("model.safetensors").is_file());
    let bytes = std::fs::metadata(dir.join("model.safetensors"))
        .unwrap()
        .len();
    let resident = ResidentWeights::open(&dir).unwrap();
    assert_eq!(resident.dtype(), candle_core::DType::BF16);
    let image = resident.image();
    let count = image.len();
    assert!(count > 100, "real model exposed only {count} entries");
    assert_eq!(image.resident_bytes(), (bytes + 63) & !63);

    let base = image.base() as usize;
    let end = base + image.resident_bytes() as usize;
    let mut bf16 = 0usize;
    for index in 0..count {
        let view = image.at(index).unwrap();
        assert!(!view.name().unwrap().is_empty());
        assert_eq!(view.data_ptr() as usize, base + view.offset() as usize);
        assert!(view.data_ptr() as usize >= base);
        assert!(view.data_ptr() as usize + view.bytes() as usize <= end);
        bf16 += usize::from(view.dtype().unwrap() == WeightDType::BF16);
    }
    assert!(bf16 > 100, "real model exposed only {bf16} BF16 entries");
}

#[test]
#[ignore = "needs the repository LFM2.5-Audio fixture and about 8 GB of free memory"]
fn real_oracle_loader_reports_every_compatibility_copy() {
    let dir = workspace_model_dir();
    assert!(dir.join("model.safetensors").is_file());
    let (model, _processor) =
        liquid_audio_oracle::from_pretrained(&dir, &candle_core::Device::Cpu).unwrap();
    let image = model
        .resident_weights()
        .expect("oracle model lost its native checkpoint owner");
    assert!(image.len() > 100);
    assert!(image.resident_bytes() > 2_000_000_000);

    let copies = model.compatibility_copies();
    assert!(copies.tensors > 500);
    assert!(copies.bytes > 2_000_000_000);
}
