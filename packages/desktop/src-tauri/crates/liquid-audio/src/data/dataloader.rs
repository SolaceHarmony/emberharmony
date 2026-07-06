//! Port of `liquid_audio/data/dataloader.py` — `LFM2DataLoader` (the map-style
//! dataset) + `lfm2_collator` (batch collation → `LFM2AudioModelInput`).
//!
//! Python relies on `torch.utils.data.Dataset` + HuggingFace `datasets`
//! (`load_from_disk`, Arrow-backed). candle has no DataLoader/Arrow analog, so the
//! faithful real equivalents are:
//!
//! * `LFM2DataLoader` ⇒ a map-style dataset (`len` = `__len__`, `get` =
//!   `__getitem__`) that owns its rows and applies the **same** right-padding to
//!   `context_length` that the Python `__getitem__` does (`F.pad` →
//!   `pad_with_zeros` for zero-fill, concat-with-constant for the `TEXT`/`False`
//!   pad values). The on-disk source (`load_from_disk`) is the crate's safetensors
//!   persistence convention rather than Arrow — [`RawRow`] is the per-row record an
//!   Arrow/safetensors reader would yield, and [`LFM2DataLoader::new`] takes those
//!   rows directly (the loader does the padding, exactly as Python does).
//! * `lfm2_collator` ⇒ a free function over `&[LFM2AudioRow]` that concatenates the
//!   per-field tensors along the same dims as the torch `torch.cat` calls and
//!   returns [`LFM2AudioModelInput`].
//!
//! Both [`LFM2AudioRow`] and [`LFM2AudioModelInput`] are the canonical types from
//! [`crate::data::types`] (Python `data/types.py`) — reused, not redefined.
//!
//! Pure candle, no torch. `Result` is `candle_core::Result`, matching the crate.

use candle_core::{DType, Device, Result, Tensor};

use crate::data::types::{LFM2AudioModelInput, LFM2AudioRow};
use crate::utils::LFMModality;

/// One un-padded record as an on-disk reader (`load_from_disk` / a safetensors
/// row store) would yield it, before [`LFM2DataLoader`] applies its padding.
///
/// Field names match the Python dataset columns read in `__getitem__`
/// (`row["text"]`, `row["audio_in"]`, …). This is the input to
/// [`LFM2DataLoader::new`]; the loader performs the `torch.as_tensor` dtype casts
/// and the `context_length` padding, faithfully to the Python `__getitem__`.
#[derive(Debug, Clone)]
pub struct RawRow {
    /// `row["text"]` — `(1, n)` token ids.
    pub text: Tensor,
    /// `row["audio_in"]` — `(nfilt, total_frames)` mel features.
    pub audio_in: Tensor,
    /// `row["audio_in_lens"]` — `(k,)` per-segment frame counts.
    pub audio_in_lens: Tensor,
    /// `row["audio_out"]` — `(codebooks, m)` audio-out codes.
    pub audio_out: Tensor,
    /// `row["modality_flag"]` — `(1, n)` modality flags.
    pub modality_flag: Tensor,
    /// `row["supervision_mask"]` — `(1, n)` supervision mask (0/1).
    pub supervision_mask: Tensor,
}

/// `LFM2DataLoader(TorchDataset[LFM2AudioRow])` — a map-style dataset that loads
/// pre-packed rows and pads each to `context_length` on access.
///
/// Python wraps `datasets.load_from_disk(dataset_path)` (Arrow-backed); the
/// faithful real equivalent here owns the decoded rows in memory (loaded from the
/// crate's safetensors persistence). `__len__` / `__getitem__` become
/// [`len`](Self::len) / [`get`](Self::get).
pub struct LFM2DataLoader {
    /// `self.dataset_path` — `Path(dataset_path)`. Kept for the 1:1 inventory /
    /// diagnostics.
    pub dataset_path: std::path::PathBuf,
    /// `self.context_length` — the fixed padded length every row is padded to.
    pub context_length: usize,
    /// `self.dataset` — the in-memory row store (`load_from_disk` result).
    rows: Vec<RawRow>,
    /// Device the padded tensors are built on.
    device: Device,
}

impl LFM2DataLoader {
    /// `__init__(self, dataset_path, context_length=4096)`. Python's default
    /// `context_length` is 4096.
    ///
    /// `rows` are the decoded records (the `load_from_disk` result). `device` is
    /// where the padded row tensors are built (torch leaves these on CPU here;
    /// pass [`Device::Cpu`] to match).
    pub fn new(
        dataset_path: impl Into<std::path::PathBuf>,
        context_length: usize,
        rows: Vec<RawRow>,
        device: Device,
    ) -> Self {
        Self {
            dataset_path: dataset_path.into(),
            context_length,
            rows,
            device,
        }
    }

    /// `self.dataset = load_from_disk(self.dataset_path)` — read the HuggingFace
    /// `datasets` directory (`state.json` + `dataset_info.json` + the Arrow IPC
    /// shard(s)) into the in-memory row store, via [`crate::data::arrow_io`] (pure
    /// Rust `arrow`). Rows come back in dataset order, faithful to the ordered
    /// `datasets.Dataset` indexing `__getitem__(idx)` relies on. The default
    /// `context_length` matching Python's is [`LFM2DataLoader::DEFAULT_CONTEXT_LENGTH`].
    pub fn load_from_disk(
        dataset_path: impl Into<std::path::PathBuf>,
        context_length: usize,
        device: Device,
    ) -> Result<Self> {
        let dataset_path = dataset_path.into();
        let rows = crate::data::arrow_io::load_from_disk(&dataset_path, &device)?;
        Ok(Self::new(dataset_path, context_length, rows, device))
    }

    /// Python's `context_length: int = 4096` default.
    pub const DEFAULT_CONTEXT_LENGTH: usize = 4096;

    /// `__len__(self)` → `len(self.dataset)`.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when the dataset has no rows (paired with [`len`](Self::len) for the
    /// `len`-without-`is_empty` lint).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// `__getitem__(self, idx)` → a padded [`LFM2AudioRow`].
    ///
    /// Faithful to the Python body: cast each column (`torch.as_tensor(..., dtype)`
    /// → candle `to_dtype`), compute `pad_len = context_length - modality.shape[1]`,
    /// raise on a negative pad (sample longer than `context_length`), then
    /// right-pad `text` (zero), `modality` (with `LFMModality::TEXT`) and
    /// `supervision` (with `False`). `audio_in` / `audio_in_lens` / `audio_out` are
    /// returned unpadded — collate concatenates them.
    pub fn get(&self, idx: usize) -> Result<LFM2AudioRow> {
        let row = self.rows.get(idx).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "index {idx} out of range (len {})",
                self.rows.len()
            ))
        })?;

        // torch.as_tensor casts: long → I64 (torch.long is int64 — keep it; candle's
        // index_select/embedding accept I64, and the model casts to U32 only at the
        // ops that need it, exactly as torch does; narrowing to U32 here would be a
        // lossy int64→u32→i64 round-trip with no upside). float32 → F32. bool → U8
        // (candle has no bool dtype; U8 is the 0/1 carrier — the one forced deviation).
        let text = row.text.to_dtype(DType::I64)?;
        let audio_in = row.audio_in.to_dtype(DType::F32)?;
        let audio_in_lens = row.audio_in_lens.to_dtype(DType::I64)?;
        let audio_out = row.audio_out.to_dtype(DType::I64)?;
        let modality = row.modality_flag.to_dtype(DType::I64)?;
        let supervision = row.supervision_mask.to_dtype(DType::U8)?;

        // pad_len = self.context_length - int(modality.shape[1])
        let cur_len = modality.dim(1)?;
        if cur_len > self.context_length {
            // ValueError(f"sample at index {idx} has {…} tokens, which is longer
            //            than context_length={…}")
            return Err(candle_core::Error::Msg(format!(
                "sample at index {idx} has {cur_len} tokens, which is longer than context_length={}",
                self.context_length
            )));
        }
        let pad_len = self.context_length - cur_len;

        // text = F.pad(text, (0, pad_len))  — right-pad the last dim with 0.
        let text = if pad_len > 0 {
            text.pad_with_zeros(1, 0, pad_len)?
        } else {
            text
        };

        // modality = F.pad(modality, (0, pad_len), value=int(LFMModality.TEXT))
        let modality = pad_right_with(&modality, pad_len, LFMModality::Text as i64, &self.device)?;

        // supervision = F.pad(supervision, (0, pad_len), value=False)  — False == 0,
        // so a zero-pad is faithful; pad_with_zeros keeps the U8 dtype.
        let supervision = if pad_len > 0 {
            supervision.pad_with_zeros(1, 0, pad_len)?
        } else {
            supervision
        };

        // return LFM2AudioRow(text=…, audio_in=…, …)
        Ok(LFM2AudioRow {
            text,
            audio_in,
            audio_in_lens,
            audio_out,
            modality_flag: modality,
            supervision_mask: supervision,
        })
    }

    /// Iterate the padded rows (the `for row in loader` / DataLoader iteration the
    /// collator consumes). Errors from [`get`](Self::get) are surfaced per-item.
    pub fn iter(&self) -> impl Iterator<Item = Result<LFM2AudioRow>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }
}

/// `F.pad(x, (0, pad_len), value=v)` for a `(rows, n)` integer tensor — candle's
/// `pad_with_zeros` only zero-fills, so a non-zero pad value is built explicitly
/// and concatenated on the right of the last dim. The pad is cast to `x`'s dtype,
/// so this works for any integer `x` (I64 here).
fn pad_right_with(x: &Tensor, pad_len: usize, value: i64, device: &Device) -> Result<Tensor> {
    if pad_len == 0 {
        return Ok(x.clone());
    }
    if value == 0 {
        return x.pad_with_zeros(1, 0, pad_len);
    }
    let rows = x.dim(0)?;
    let pad = Tensor::from_vec(vec![value; rows * pad_len], (rows, pad_len), device)?
        .to_dtype(x.dtype())?;
    Tensor::cat(&[x, &pad], 1)
}

/// `lfm2_collator(batch: list[LFM2AudioRow]) -> LFM2AudioModelInput` —
/// concatenate a list of padded [`LFM2AudioRow`]s into one batched
/// [`LFM2AudioModelInput`].
///
/// Faithful to the Python `torch.cat` dims:
/// * `audio_in`        — `dim=1` (frames axis; rows are `(nfilt, frames)`)
/// * `audio_in_lens`   — `dim=0` (1-D concat)
/// * `text`            — `dim=1` (rows are `(1, context_length)`)
/// * `audio_out`       — `dim=1` (rows are `(codebooks, m)`)
/// * `modality_flag`   — `dim=0`
/// * `supervision_mask`— `dim=0`
///
/// (`modality_flag` / `supervision_mask` cat on `dim=0`, stacking each row's
/// single `(1, context_length)` row into `(batch, context_length)`, exactly as
/// torch does.)
pub fn lfm2_collator(batch: &[LFM2AudioRow]) -> Result<LFM2AudioModelInput> {
    if batch.is_empty() {
        return Err(candle_core::Error::Msg("lfm2_collator: empty batch".into()));
    }
    let cat = |tensors: Vec<&Tensor>, dim: usize| Tensor::cat(&tensors, dim);

    let audio_in = cat(batch.iter().map(|r| &r.audio_in).collect(), 1)?;
    let audio_in_lens = cat(batch.iter().map(|r| &r.audio_in_lens).collect(), 0)?;
    let text = cat(batch.iter().map(|r| &r.text).collect(), 1)?;
    let audio_out = cat(batch.iter().map(|r| &r.audio_out).collect(), 1)?;
    let modality_flag = cat(batch.iter().map(|r| &r.modality_flag).collect(), 0)?;
    let supervision_mask = cat(batch.iter().map(|r| &r.supervision_mask).collect(), 0)?;

    Ok(LFM2AudioModelInput {
        text,
        audio_in,
        audio_in_lens,
        audio_out,
        modality_flag,
        supervision_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    fn raw_row(n: usize, nfilt: usize, frames: usize, codebooks: usize, m: usize) -> RawRow {
        let dev = Device::Cpu;
        RawRow {
            text: Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), (1, n), &dev).unwrap(),
            audio_in: Tensor::zeros((nfilt, frames), DType::F32, &dev).unwrap(),
            audio_in_lens: Tensor::from_vec(vec![frames as u32], (1,), &dev).unwrap(),
            audio_out: Tensor::zeros((codebooks, m), DType::U32, &dev).unwrap(),
            // first token text, rest audio-out, to exercise the non-zero pad value.
            modality_flag: Tensor::from_vec(
                {
                    let mut v = vec![LFMModality::Text as u32];
                    v.extend(std::iter::repeat(LFMModality::AudioOut as u32).take(n - 1));
                    v
                },
                (1, n),
                &dev,
            )
            .unwrap(),
            supervision_mask: Tensor::from_vec(vec![1u8; n], (1, n), &dev).unwrap(),
        }
    }

    #[test]
    fn getitem_pads_to_context_length() {
        let ctx = 16;
        let loader = LFM2DataLoader::new("mem", ctx, vec![raw_row(5, 4, 7, 8, 3)], Device::Cpu);
        assert_eq!(loader.len(), 1);
        let row = loader.get(0).unwrap();
        // text / modality / supervision padded to context_length.
        assert_eq!(row.text.dims(), &[1, ctx]);
        assert_eq!(row.modality_flag.dims(), &[1, ctx]);
        assert_eq!(row.supervision_mask.dims(), &[1, ctx]);
        // audio_in / audio_out unpadded.
        assert_eq!(row.audio_in.dims(), &[4, 7]);
        assert_eq!(row.audio_out.dims(), &[8, 3]);

        // dtype is I64 (torch.long), not narrowed to U32.
        assert_eq!(row.text.dtype(), DType::I64);
        assert_eq!(row.modality_flag.dtype(), DType::I64);
        assert_eq!(row.audio_out.dtype(), DType::I64);
        // modality pad value is TEXT; supervision pad value is 0 (False).
        let modality: Vec<i64> = row.modality_flag.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(modality[5], LFMModality::Text as i64);
        assert_eq!(modality[ctx - 1], LFMModality::Text as i64);
        let sup: Vec<u8> = row.supervision_mask.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(sup[5], 0);
        assert_eq!(sup[4], 1);
        // text pad value is 0.
        let text: Vec<i64> = row.text.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(text[ctx - 1], 0);
    }

    #[test]
    fn getitem_rejects_overlong_sample() {
        let loader = LFM2DataLoader::new("mem", 4, vec![raw_row(5, 4, 7, 8, 3)], Device::Cpu);
        assert!(loader.get(0).is_err());
    }

    #[test]
    fn collator_concatenates_along_python_dims() {
        let ctx = 16;
        let rows = vec![raw_row(5, 4, 7, 8, 3), raw_row(6, 4, 9, 8, 2)];
        let loader = LFM2DataLoader::new("mem", ctx, rows, Device::Cpu);
        let batch: Vec<LFM2AudioRow> = loader.iter().collect::<Result<_>>().unwrap();
        let input = lfm2_collator(&batch).unwrap();

        // text: cat dim=1 → (1, 2*ctx); modality/supervision: cat dim=0 → (2, ctx).
        assert_eq!(input.text.dims(), &[1, 2 * ctx]);
        assert_eq!(input.modality_flag.dims(), &[2, ctx]);
        assert_eq!(input.supervision_mask.dims(), &[2, ctx]);
        // audio_in: cat dim=1 → (4, 7+9); audio_out: cat dim=1 → (8, 3+2).
        assert_eq!(input.audio_in.dims(), &[4, 16]);
        assert_eq!(input.audio_out.dims(), &[8, 5]);
        // audio_in_lens: cat dim=0 → (2,).
        assert_eq!(input.audio_in_lens.dims(), &[2]);
    }
}
