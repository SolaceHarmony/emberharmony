//! Port of `liquid_audio/data/preprocess.py` — `preprocess_dataset`.
//!
//! Maps each chat (`list[ChatMessage]`) through an [`LFM2AudioChatMapper`], applies
//! the `max_context_length` filter, and writes the kept samples to disk as a real
//! HuggingFace `datasets.Dataset` (`save_to_disk`): an Arrow IPC shard plus the
//! `dataset_info.json` / `state.json` sidecars. The Arrow write/read lives in
//! [`crate::data::arrow_io`] (pure Rust `arrow` crates — no torch, no pyarrow), and
//! [`crate::data::dataloader::LFM2DataLoader::load_from_disk`] reads it back.
//!
//! | Python | Rust |
//! |---|---|
//! | `data: Iterable[list[ChatMessage]]` | `data: impl IntoIterator<Item = Vec<ChatMessage>>` |
//! | `mapper: LFM2AudioChatMapper` | `mapper: &impl ChatMapper` (the real mapper satisfies it) |
//! | `mapper(messages) -> LFM2AudioTrainingSample` | `mapper.map(&messages)` |
//! | `max_context_length: int = -1` | `max_context_length: i64` |
//! | `sample.modality_flag.shape[-1]` | `sample.modality_flag.dim(D::Minus1)?` |
//! | `print("WARNING: skipping sample …")` | identical `eprintln!` |
//! | `Dataset.from_generator(gen, features).save_to_disk(out_dir)` | [`crate::data::arrow_io::save_to_disk`] |
//! | `out_dir.mkdir(parents=True, exist_ok=False)` | [`create_output_dir`] |

use std::path::Path;

use candle_core::{Result, D};

use crate::data::types::{ChatMessage, LFM2AudioTrainingSample};
use crate::data::LFM2AudioChatMapper;

/// The injected mapper. Python takes a concrete `LFM2AudioChatMapper` and only ever
/// calls `mapper(messages)`; this trait captures exactly that `messages -> sample`
/// contract so the preprocessor depends on the behaviour, not the concrete type.
///
/// `map` mirrors `LFM2AudioChatMapper.__call__(messages) -> LFM2AudioTrainingSample`.
pub trait ChatMapper {
    /// `mapper(messages)` — map one chat into a packed training sample.
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample>;
}

/// The real mapper from `data/mapper.py` is the canonical implementor.
impl ChatMapper for LFM2AudioChatMapper<'_> {
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample> {
        self.call(messages)
    }
}

/// Blanket impl so a plain closure is also a mapper (the Python `mapper` is just a
/// callable).
impl<F> ChatMapper for F
where
    F: Fn(&[ChatMessage]) -> Result<LFM2AudioTrainingSample>,
{
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample> {
        self(messages)
    }
}

/// `out_dir.mkdir(parents=True, exist_ok=False)` — create the output directory and
/// all parents, but fail if it already exists (faithful to `exist_ok=False`).
pub fn create_output_dir(out_dir: &Path) -> Result<()> {
    if out_dir.exists() {
        return Err(candle_core::Error::Msg(format!(
            "output path already exists: {} (exist_ok=False)",
            out_dir.display()
        )));
    }
    std::fs::create_dir_all(out_dir)
        .map_err(|e| candle_core::Error::Msg(format!("mkdir {}: {e}", out_dir.display())))
}

/// Faithful port of `preprocess_dataset`.
///
/// ```python
/// def preprocess_dataset(data, output_path, mapper, max_context_length=-1) -> None:
///     out_dir = Path(output_path); out_dir.mkdir(parents=True, exist_ok=False)
///     features = Features({...})
///     def generator():
///         for i, messages in enumerate(data):
///             sample = mapper(messages)
///             if 0 <= max_context_length < int(sample.modality_flag.shape[-1]):
///                 print(f"WARNING: skipping sample {i} ..."); continue
///             yield {field: sample.field.tolist() for field in ...}
///     datasets.Dataset.from_generator(generator, features=features).save_to_disk(out_dir)
/// ```
///
/// Returns the number of rows written. Python returns `None`, but the kept-row
/// count is the natural Rust signal (and lets callers assert the skip behaviour).
pub fn preprocess_dataset(
    data: impl IntoIterator<Item = Vec<ChatMessage>>,
    output_path: impl AsRef<Path>,
    mapper: &impl ChatMapper,
    max_context_length: i64,
) -> Result<usize> {
    let out_dir = output_path.as_ref();
    create_output_dir(out_dir)?;

    // The `generator()` body: map → context-length skip. `arrow`'s `RecordBatch`
    // builders accumulate columns, so (like the current safetensors path, and like
    // a single `from_generator` batch) the kept samples are held before the flush.
    let mut kept: Vec<LFM2AudioTrainingSample> = Vec::new();
    for (i, messages) in data.into_iter().enumerate() {
        let sample = mapper.map(&messages)?;
        let sample_len = sample.modality_flag.dim(D::Minus1)? as i64; // int(...shape[-1])
                                                                      // `if 0 <= max_context_length < sample_len` — the half-open range
                                                                      // `[0, sample_len)` contains `max_context_length` iff both bounds hold.
        if (0..sample_len).contains(&max_context_length) {
            eprintln!("WARNING: skipping sample {i} with {sample_len} tokens (max_context_length={max_context_length})");
            continue;
        }
        kept.push(sample);
    }

    // `Dataset.from_generator(...).save_to_disk(out_dir)`.
    crate::data::arrow_io::save_to_disk(out_dir, &kept)?;
    Ok(kept.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataloader::LFM2DataLoader;
    use crate::data::types::{ChatContentSegment, Role, TextSegment};
    use candle_core::{DType, Device, Tensor};
    use std::path::PathBuf;

    /// Build a minimal sample whose `modality_flag` has `n` columns (the only field
    /// the skip logic inspects). The other fields are shape-faithful so the
    /// dataloader can pad/collate them after the Arrow round-trip.
    fn sample_with_len(n: usize, dev: &Device) -> Result<LFM2AudioTrainingSample> {
        Ok(LFM2AudioTrainingSample {
            text: Tensor::zeros((1, n), DType::I64, dev)?,
            audio_in: Tensor::zeros((128, 0), DType::F32, dev)?,
            audio_in_lens: Tensor::zeros((0,), DType::I64, dev)?,
            audio_out: Tensor::zeros((8, 0), DType::I64, dev)?,
            modality_flag: Tensor::ones((1, n), DType::I64, dev)?, // LFMModality::TEXT
            supervision_mask: Tensor::zeros((1, n), DType::U8, dev)?,
        })
    }

    fn one_msg(n: usize) -> Vec<ChatMessage> {
        let seg: ChatContentSegment = TextSegment::new("x".repeat(n)).into();
        vec![ChatMessage::new(Role::User, vec![seg])]
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("lfm2_preprocess_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn writes_real_hf_arrow_layout() {
        let out = tmp_dir("layout");
        let mapper = |m: &[ChatMessage]| sample_with_len(m.len().max(1), &Device::Cpu);
        preprocess_dataset(vec![one_msg(1), one_msg(1)], &out, &mapper, -1).unwrap();
        // save_to_disk writes the Arrow shard + the two HF json sidecars.
        assert!(out.join("data-00000-of-00001.arrow").is_file());
        assert!(out.join("dataset_info.json").is_file());
        assert!(out.join("state.json").is_file());
        let info: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("dataset_info.json")).unwrap()).unwrap();
        assert_eq!(info["features"]["text"]["_type"], "Sequence");
        assert_eq!(
            info["features"]["audio_in"]["feature"]["feature"]["dtype"],
            "float32"
        );
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn skips_samples_over_max_context_length_and_reloads_via_dataloader() {
        let dev = Device::Cpu;
        let out = tmp_dir("skip");
        let mapper = |m: &[ChatMessage]| {
            let n: usize = m
                .iter()
                .flat_map(|msg| msg.content())
                .map(|s| match s {
                    ChatContentSegment::Text(t) => t.text().len(),
                    _ => 0,
                })
                .sum();
            sample_with_len(n, &Device::Cpu)
        };
        // lengths 2, 5, 3 ; max_context_length=4 ⇒ the length-5 sample is skipped.
        let written =
            preprocess_dataset(vec![one_msg(2), one_msg(5), one_msg(3)], &out, &mapper, 4).unwrap();
        assert_eq!(written, 2, "the length-5 sample should be filtered out");

        // The on-disk Arrow dataset round-trips through the crate's dataloader.
        let loader = LFM2DataLoader::load_from_disk(&out, 4096, dev).unwrap();
        assert_eq!(loader.len(), 2);
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn negative_max_context_length_keeps_everything() {
        let out = tmp_dir("keepall");
        let mapper = |m: &[ChatMessage]| sample_with_len(m.len().max(1), &Device::Cpu);
        // max_context_length=-1 ⇒ `0 <= -1` is false ⇒ nothing is skipped.
        let written =
            preprocess_dataset(vec![one_msg(2), one_msg(9999)], &out, &mapper, -1).unwrap();
        assert_eq!(written, 2);
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn mkdir_rejects_existing_dir() {
        let out = tmp_dir("exists");
        std::fs::create_dir_all(&out).unwrap();
        let mapper = |m: &[ChatMessage]| sample_with_len(m.len(), &Device::Cpu);
        let r = preprocess_dataset(Vec::<Vec<ChatMessage>>::new(), &out, &mapper, -1);
        assert!(r.is_err(), "exist_ok=False ⇒ pre-existing dir must error");
        std::fs::remove_dir_all(&out).ok();
    }
}
