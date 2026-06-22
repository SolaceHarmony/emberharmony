//! Port of `liquid_audio/data/preprocess.py` — `preprocess_dataset`.
//!
//! The Python function maps each chat (a `list[ChatMessage]`) through an
//! [`LFM2AudioChatMapper`], applies a `max_context_length` filter, and writes the
//! resulting per-sample tensors to disk as a HuggingFace `datasets.Dataset`
//! (`Features` schema + `save_to_disk`). There is no `datasets`/Arrow analog in
//! the candle stack, so the faithful Rust equivalent keeps every observable
//! behaviour — the mapper call, the identical context-length skip + WARNING, and
//! a lossless on-disk dataset — while swapping Arrow for the crate's existing
//! safetensors persistence (the same `r{idx}.{field}` layout
//! [`crate::data::dataloader::LFM2DataLoader::load_from_disk`] reads back, so a
//! preprocessed dataset feeds straight into the dataloader).
//!
//! Faithful mapping of the Python pieces:
//!
//! | Python | Rust |
//! |---|---|
//! | `data: Iterable[list[ChatMessage]]` | `data: impl IntoIterator<Item = Vec<ChatMessage>>` |
//! | `mapper: LFM2AudioChatMapper` | `mapper: &impl ChatMapper` (the [`LFM2AudioChatMapper`] satisfies it; the trait mirrors that Python passes any callable mapper) |
//! | `mapper(messages) -> LFM2AudioTrainingSample` | `mapper.map(&messages) -> `[`LFM2AudioTrainingSample`] |
//! | `max_context_length: int = -1` | `max_context_length: i64` (pass `-1` for the default) |
//! | `sample.modality_flag.shape[-1]` | `sample.modality_flag.dim(D::Minus1)?` |
//! | `datasets.Features({...})` | [`DatasetFeatures`] (serialized to `dataset_info.json`) |
//! | `print("WARNING: skipping sample ...")` | identical `eprintln!` |
//! | `Dataset.from_generator(gen).save_to_disk(out_dir)` | one `data.safetensors` shard of `r{idx}.{field}` rows + `dataset_info.json` |
//! | `out_dir.mkdir(parents=True, exist_ok=False)` | [`create_output_dir`] (`create_dir_all`, error if it already exists) |
//!
//! Pure candle, no torch, no Arrow.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Result, Tensor, D};

use crate::data::types::{ChatMessage, LFM2AudioTrainingSample};
use crate::data::LFM2AudioChatMapper;

/// The injected mapper. Python takes a concrete `LFM2AudioChatMapper`
/// (`data/mapper.py`) and only ever calls `mapper(messages)`; this trait captures
/// exactly that `messages -> sample` contract so the preprocessor depends on the
/// behaviour, not the concrete type (faithful to the loose Python coupling and
/// trivially testable with a closure).
///
/// `map` mirrors `LFM2AudioChatMapper.__call__(messages) -> LFM2AudioTrainingSample`.
pub trait ChatMapper {
    /// `mapper(messages)` — map one chat into a packed training sample.
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample>;
}

/// The real mapper from `data/mapper.py` is the canonical implementor — its
/// `call` is the Python `__call__`.
impl ChatMapper for LFM2AudioChatMapper<'_> {
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample> {
        self.call(messages)
    }
}

/// Blanket impl so a plain closure `Fn(&[ChatMessage]) -> Result<LFM2AudioTrainingSample>`
/// is also a mapper (the Python `mapper` is just a callable).
impl<F> ChatMapper for F
where
    F: Fn(&[ChatMessage]) -> Result<LFM2AudioTrainingSample>,
{
    fn map(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample> {
        self(messages)
    }
}

/// One field's leaf dtype in the dataset schema. Mirrors the `Value(...)` dtypes
/// of the Python `Features` (`int64` / `float32` / `bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureDtype {
    Int64,
    Float32,
    Bool,
}

/// One field of the dataset schema: a leaf dtype nested under `nesting` levels of
/// `Sequence(...)`. Faithful to the `Features` entries in `preprocess.py`, e.g.
/// `Sequence(Sequence(Value("int64")))` is `{ nesting: 2, dtype: Int64 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FeatureSpec {
    /// Number of `Sequence(...)` wrappers around the leaf value.
    pub nesting: u8,
    /// The leaf `Value(...)` dtype.
    pub dtype: FeatureDtype,
}

/// The dataset schema — the Rust analog of the Python
/// `datasets.Features({...})`. Serialized verbatim to `dataset_info.json` so the
/// on-disk layout is self-describing (mirrors what `save_to_disk` records).
///
/// The field order matches the Python `Features` dict (the column write order).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetFeatures {
    /// `"text": Sequence(Sequence(Value("int64")))`.
    pub text: FeatureSpec,
    /// `"audio_in": Sequence(Sequence(Value("float32")))`.
    pub audio_in: FeatureSpec,
    /// `"audio_in_lens": Sequence(Value("int64"))`.
    pub audio_in_lens: FeatureSpec,
    /// `"audio_out": Sequence(Sequence(Value("int64")))`.
    pub audio_out: FeatureSpec,
    /// `"modality_flag": Sequence(Sequence(Value("int64")))`.
    pub modality_flag: FeatureSpec,
    /// `"supervision_mask": Sequence(Sequence(Value("bool")))`.
    pub supervision_mask: FeatureSpec,
}

impl DatasetFeatures {
    /// The exact schema the Python `preprocess_dataset` declares:
    /// ```python
    /// Features({
    ///     "text": Sequence(Sequence(Value("int64"))),
    ///     "audio_in": Sequence(Sequence(Value("float32"))),
    ///     "audio_in_lens": Sequence(Value("int64")),
    ///     "audio_out": Sequence(Sequence(Value("int64"))),
    ///     "modality_flag": Sequence(Sequence(Value("int64"))),
    ///     "supervision_mask": Sequence(Sequence(Value("bool"))),
    /// })
    /// ```
    pub fn lfm2_audio() -> Self {
        let seq2 = |dtype| FeatureSpec { nesting: 2, dtype };
        let seq1 = |dtype| FeatureSpec { nesting: 1, dtype };
        Self {
            text: seq2(FeatureDtype::Int64),
            audio_in: seq2(FeatureDtype::Float32),
            audio_in_lens: seq1(FeatureDtype::Int64),
            audio_out: seq2(FeatureDtype::Int64),
            modality_flag: seq2(FeatureDtype::Int64),
            supervision_mask: seq2(FeatureDtype::Bool),
        }
    }
}

/// The `dataset_info.json` sidecar — `save_to_disk` writes a descriptor next to
/// the data; this is the faithful, minimal analog (the schema + the number of
/// kept rows). The single shard's row keys are `r{idx}.{field}`, the same layout
/// [`crate::data::dataloader::LFM2DataLoader::load_from_disk`] reads.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetInfo {
    /// The `Features` schema (see [`DatasetFeatures`]).
    pub features: DatasetFeatures,
    /// Number of samples actually written (after the `max_context_length` skip).
    pub num_rows: usize,
}

/// The data shard the rows are written into (read back by the dataloader, which
/// scans every `*.safetensors` under the dataset dir).
const DATA_SHARD: &str = "data.safetensors";

/// `out_dir.mkdir(parents=True, exist_ok=False)` — create the output directory
/// and all parents, but fail if it already exists (faithful to `exist_ok=False`,
/// which raises `FileExistsError`).
pub fn create_output_dir(out_dir: &Path) -> Result<()> {
    if out_dir.exists() {
        return Err(candle_core::Error::Msg(format!(
            "output path already exists: {} (exist_ok=False)",
            out_dir.display()
        )));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| candle_core::Error::Msg(format!("mkdir {}: {e}", out_dir.display())))
}

/// Faithful port of `preprocess_dataset`.
///
/// ```python
/// def preprocess_dataset(data, output_path, mapper, max_context_length=-1) -> None:
///     out_dir = Path(output_path)
///     out_dir.mkdir(parents=True, exist_ok=False)
///     features = Features({...})
///     def generator():
///         for i, messages in enumerate(data):
///             sample = mapper(messages)
///             sample_len = int(sample.modality_flag.shape[-1])
///             if 0 <= max_context_length < sample_len:
///                 print(f"WARNING: skipping sample {i} ...")
///                 continue
///             yield {field: sample.field.tolist() for field in ...}
///     preprocessed = datasets.Dataset.from_generator(generator, features=features)
///     preprocessed.save_to_disk(out_dir)
/// ```
///
/// `data` is consumed lazily (the Python generator never materializes the whole
/// dataset); each kept sample's six tensors are accumulated under `r{idx}.{field}`
/// keys and flushed to a single safetensors shard at the end — the lossless analog
/// of yielding `.tolist()` dicts into Arrow and `save_to_disk`. Returns the number
/// of rows written (Python returns `None`, but the kept-row count is the natural
/// Rust signal and lets callers assert the skip behaviour).
pub fn preprocess_dataset(
    data: impl IntoIterator<Item = Vec<ChatMessage>>,
    output_path: impl AsRef<Path>,
    mapper: &impl ChatMapper,
    max_context_length: i64,
) -> Result<usize> {
    let out_dir = output_path.as_ref();
    create_output_dir(out_dir)?;

    let features = DatasetFeatures::lfm2_audio();

    // The `generator()` body, run to disk (the Rust analog of
    // `from_generator(...).save_to_disk(...)`): map → context-length filter →
    // pack each kept row under `r{kept_idx}.{field}`.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut kept = 0usize;
    for (i, messages) in data.into_iter().enumerate() {
        let sample = mapper.map(&messages)?;
        // `int(sample.modality_flag.shape[-1])`.
        let sample_len = sample.modality_flag.dim(D::Minus1)? as i64;
        // `if 0 <= max_context_length < sample_len: print(...); continue` — the
        // half-open range `[0, sample_len)` contains `max_context_length` exactly
        // when both `0 <= max_context_length` and `max_context_length < sample_len`.
        if (0..sample_len).contains(&max_context_length) {
            eprintln!(
                "WARNING: skipping sample {i} with {sample_len} tokens (max_context_length={max_context_length})"
            );
            continue;
        }
        // `yield {...}` → stage the row's six tensors. Keys are `r{idx}.{field}`,
        // indexed by post-skip position so the dataloader sees a dense `0..num_rows`
        // (mirrors the ordered `datasets.Dataset` it reads).
        tensors.insert(format!("r{kept}.text"), sample.text);
        tensors.insert(format!("r{kept}.audio_in"), sample.audio_in);
        tensors.insert(format!("r{kept}.audio_in_lens"), sample.audio_in_lens);
        tensors.insert(format!("r{kept}.audio_out"), sample.audio_out);
        tensors.insert(format!("r{kept}.modality_flag"), sample.modality_flag);
        tensors.insert(format!("r{kept}.supervision_mask"), sample.supervision_mask);
        kept += 1;
    }

    // `save_to_disk`: flush the row store + write the schema sidecar.
    candle_core::safetensors::save(&tensors, out_dir.join(DATA_SHARD))?;
    let info = DatasetInfo { features, num_rows: kept };
    let info_path = out_dir.join("dataset_info.json");
    let json = serde_json::to_string_pretty(&info)
        .map_err(|e| candle_core::Error::Msg(format!("serialize dataset_info: {e}")))?;
    std::fs::write(&info_path, json)
        .map_err(|e| candle_core::Error::Msg(format!("write {}: {e}", info_path.display())))?;

    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataloader::LFM2DataLoader;
    use crate::data::types::{ChatContentSegment, Role, TextSegment};
    use candle_core::{DType, Device};
    use std::path::PathBuf;

    /// Build a minimal sample whose `modality_flag` has `n` columns (the only
    /// field the skip logic inspects). The other fields are shape-faithful so the
    /// dataloader can pad/collate them.
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
    fn features_schema_matches_python() {
        let f = DatasetFeatures::lfm2_audio();
        assert_eq!(f.text, FeatureSpec { nesting: 2, dtype: FeatureDtype::Int64 });
        assert_eq!(f.audio_in, FeatureSpec { nesting: 2, dtype: FeatureDtype::Float32 });
        assert_eq!(f.audio_in_lens, FeatureSpec { nesting: 1, dtype: FeatureDtype::Int64 });
        assert_eq!(f.audio_out, FeatureSpec { nesting: 2, dtype: FeatureDtype::Int64 });
        assert_eq!(f.modality_flag, FeatureSpec { nesting: 2, dtype: FeatureDtype::Int64 });
        assert_eq!(f.supervision_mask, FeatureSpec { nesting: 2, dtype: FeatureDtype::Bool });
    }

    #[test]
    fn skips_samples_over_max_context_length_and_reloads_via_dataloader() {
        let dev = Device::Cpu;
        let out = tmp_dir("skip");
        // Mapper turns each chat into a sample whose length is its content's text len.
        let mapper = |m: &[ChatMessage]| {
            let n: usize = m.iter().flat_map(|msg| msg.content()).map(|s| match s {
                ChatContentSegment::Text(t) => t.text().len(),
                _ => 0,
            }).sum();
            sample_with_len(n, &Device::Cpu)
        };
        // lengths 2, 5, 3 ; max_context_length=4 ⇒ the length-5 sample is skipped.
        let data = vec![one_msg(2), one_msg(5), one_msg(3)];
        let written = preprocess_dataset(data, &out, &mapper, 4).unwrap();
        assert_eq!(written, 2, "the length-5 sample should be filtered out");

        // The on-disk format round-trips through the crate's own dataloader.
        let loader = LFM2DataLoader::load_from_disk(&out, 4096, dev).unwrap();
        assert_eq!(loader.len(), 2);
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn negative_max_context_length_keeps_everything() {
        let out = tmp_dir("keepall");
        let mapper = |m: &[ChatMessage]| sample_with_len(m.len().max(1), &Device::Cpu);
        let data = vec![one_msg(2), one_msg(9999)];
        // max_context_length=-1 ⇒ `0 <= -1` is false ⇒ nothing is skipped.
        let written = preprocess_dataset(data, &out, &mapper, -1).unwrap();
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
