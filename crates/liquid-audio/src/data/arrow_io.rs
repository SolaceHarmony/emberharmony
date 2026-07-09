//! HuggingFace `datasets.Dataset.save_to_disk` / `load_from_disk` — the real
//! Arrow format, via the pure-Rust `arrow` crates (no torch, no pyarrow/C deps).
//!
//! `liquid_audio/data/preprocess.py` builds a `datasets.Dataset` with this schema
//! and calls `save_to_disk`, which writes a directory:
//!   - `data-00000-of-00001.arrow` — Arrow **IPC stream** of one `RecordBatch`
//!     whose columns are the six dataset fields,
//!   - `dataset_info.json` — the `Features` schema (+ empty descriptor fields),
//!   - `state.json` — `{_data_files, _fingerprint, _format_*, _split, …}`.
//! `load_from_disk` reads `state.json` for the shard list, then the IPC stream.
//!
//! Schema (`Features`):
//! ```text
//! text             Sequence(Sequence(int64))   -> List<List<Int64>>
//! audio_in         Sequence(Sequence(float32)) -> List<List<Float32>>
//! audio_in_lens    Sequence(int64)             -> List<Int64>
//! audio_out        Sequence(Sequence(int64))   -> List<List<Int64>>
//! modality_flag    Sequence(Sequence(int64))   -> List<List<Int64>>
//! supervision_mask Sequence(Sequence(bool))    -> List<List<Boolean>>
//! ```
//! One dataset row == one preprocessed sample; each `Sequence(Sequence(...))`
//! column holds that sample's 2-D tensor as an outer list of per-row inner lists.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float32Builder, Int64Builder, ListBuilder};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Int64Array, ListArray, RecordBatch,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{Field, Schema};
use candle_core::{DType, Device, Result, Tensor};

use crate::data::dataloader::RawRow;
use crate::data::types::LFM2AudioTrainingSample;

const DATA_FILE: &str = "data-00000-of-00001.arrow";

fn err(m: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(m.to_string())
}

// ---- tensor <-> nested-vec helpers (CPU; the dataloader builds on CPU) --------

fn rows_i64(t: &Tensor) -> Result<Vec<Vec<i64>>> {
    t.to_dtype(DType::I64)?.to_vec2::<i64>()
}
fn rows_f32(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    t.to_dtype(DType::F32)?.to_vec2::<f32>()
}
fn rows_bool(t: &Tensor) -> Result<Vec<Vec<bool>>> {
    Ok(t.to_dtype(DType::U8)?
        .to_vec2::<u8>()?
        .into_iter()
        .map(|r| r.into_iter().map(|x| x != 0).collect())
        .collect())
}
fn flat_i64(t: &Tensor) -> Result<Vec<i64>> {
    t.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()
}

fn tensor2_i64(rows: &[Vec<i64>], dev: &Device) -> Result<Tensor> {
    let (r, c) = (rows.len(), rows.first().map_or(0, |x| x.len()));
    Tensor::from_vec(
        rows.iter().flatten().copied().collect::<Vec<_>>(),
        (r, c),
        dev,
    )
}
fn tensor2_f32(rows: &[Vec<f32>], dev: &Device) -> Result<Tensor> {
    let (r, c) = (rows.len(), rows.first().map_or(0, |x| x.len()));
    Tensor::from_vec(
        rows.iter().flatten().copied().collect::<Vec<_>>(),
        (r, c),
        dev,
    )
}
fn tensor2_u8(rows: &[Vec<bool>], dev: &Device) -> Result<Tensor> {
    let (r, c) = (rows.len(), rows.first().map_or(0, |x| x.len()));
    Tensor::from_vec(
        rows.iter().flatten().map(|&b| b as u8).collect::<Vec<_>>(),
        (r, c),
        dev,
    )
}

// ---- write -------------------------------------------------------------------

/// Append one sample's 2-D `int64` tensor as a `List<List<Int64>>` element.
fn push_ll_i64(b: &mut ListBuilder<ListBuilder<Int64Builder>>, t: &Tensor) -> Result<()> {
    for row in rows_i64(t)? {
        b.values().values().append_slice(&row);
        b.values().append(true);
    }
    b.append(true);
    Ok(())
}
fn push_ll_f32(b: &mut ListBuilder<ListBuilder<Float32Builder>>, t: &Tensor) -> Result<()> {
    for row in rows_f32(t)? {
        b.values().values().append_slice(&row);
        b.values().append(true);
    }
    b.append(true);
    Ok(())
}
fn push_ll_bool(b: &mut ListBuilder<ListBuilder<BooleanBuilder>>, t: &Tensor) -> Result<()> {
    for row in rows_bool(t)? {
        for x in row {
            b.values().values().append_value(x);
        }
        b.values().append(true);
    }
    b.append(true);
    Ok(())
}

/// `Dataset.from_generator(...).save_to_disk(out_dir)` — write the kept samples as
/// the HF Arrow dataset (one IPC-stream shard + the two JSON sidecars).
pub fn save_to_disk(out_dir: &Path, samples: &[LFM2AudioTrainingSample]) -> Result<()> {
    let (mut text, mut audio_in, mut audio_out, mut modality) = (
        ListBuilder::new(ListBuilder::new(Int64Builder::new())),
        ListBuilder::new(ListBuilder::new(Float32Builder::new())),
        ListBuilder::new(ListBuilder::new(Int64Builder::new())),
        ListBuilder::new(ListBuilder::new(Int64Builder::new())),
    );
    let mut supervision = ListBuilder::new(ListBuilder::new(BooleanBuilder::new()));
    let mut lens = ListBuilder::new(Int64Builder::new());

    for s in samples {
        push_ll_i64(&mut text, &s.text)?;
        push_ll_f32(&mut audio_in, &s.audio_in)?;
        push_ll_i64(&mut audio_out, &s.audio_out)?;
        push_ll_i64(&mut modality, &s.modality_flag)?;
        push_ll_bool(&mut supervision, &s.supervision_mask)?;
        lens.values().append_slice(&flat_i64(&s.audio_in_lens)?);
        lens.append(true);
    }

    // Field order matches the Python `Features` dict (the column order).
    let cols: Vec<(&str, ArrayRef)> = vec![
        ("text", Arc::new(text.finish())),
        ("audio_in", Arc::new(audio_in.finish())),
        ("audio_in_lens", Arc::new(lens.finish())),
        ("audio_out", Arc::new(audio_out.finish())),
        ("modality_flag", Arc::new(modality.finish())),
        ("supervision_mask", Arc::new(supervision.finish())),
    ];
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();

    // Embed the HF `Features`/info in the schema metadata (key "huggingface"), the
    // same way pyarrow does, so the shard is self-describing for `datasets`.
    let mut meta = HashMap::new();
    meta.insert(
        "huggingface".to_string(),
        format!("{{\"info\": {{\"features\": {}}}}}", features_json()),
    );
    let schema = Arc::new(Schema::new_with_metadata(fields, meta));
    let batch = RecordBatch::try_new(schema.clone(), cols.into_iter().map(|(_, a)| a).collect())
        .map_err(err)?;

    std::fs::create_dir_all(out_dir).map_err(err)?;
    let file = std::fs::File::create(out_dir.join(DATA_FILE)).map_err(err)?;
    let mut w = StreamWriter::try_new(file, schema.as_ref()).map_err(err)?;
    w.write(&batch).map_err(err)?;
    w.finish().map_err(err)?;

    write_sidecars(out_dir)?;
    Ok(())
}

/// `dataset_info.json` (the `Features` schema + empty descriptor fields) and
/// `state.json` (the shard list + format state) that `save_to_disk` writes.
fn write_sidecars(out_dir: &Path) -> Result<()> {
    let info = serde_json::json!({
        "builder_name": null, "citation": "", "config_name": null,
        "dataset_size": null, "description": "", "download_checksums": null,
        "download_size": null, "features": serde_json::from_str::<serde_json::Value>(&features_json()).map_err(err)?,
        "homepage": "", "license": "", "post_processed": null, "post_processing_size": null,
        "size_in_bytes": null, "splits": null, "supervised_keys": null, "version": null,
    });
    let state = serde_json::json!({
        "_data_files": [{"filename": DATA_FILE}],
        "_fingerprint": "0000000000000000",
        "_format_columns": null, "_format_kwargs": {}, "_format_type": null,
        "_indexes": {}, "_output_all_columns": false, "_split": null,
    });
    std::fs::write(
        out_dir.join("dataset_info.json"),
        serde_json::to_vec_pretty(&info).map_err(err)?,
    )
    .map_err(err)?;
    std::fs::write(
        out_dir.join("state.json"),
        serde_json::to_vec_pretty(&state).map_err(err)?,
    )
    .map_err(err)?;
    Ok(())
}

/// The HF `Features` JSON: `Sequence(Sequence(Value(...)))` nests as
/// `{"feature": {"feature": {"dtype": …, "_type": "Value"}, "_type": "Sequence"}, "_type": "Sequence"}`.
fn features_json() -> String {
    let val = |dt: &str| format!("{{\"dtype\": \"{dt}\", \"_type\": \"Value\"}}");
    let seq = |inner: String| format!("{{\"feature\": {inner}, \"_type\": \"Sequence\"}}");
    let seq2 = |dt: &str| seq(seq(val(dt)));
    format!(
        "{{\"text\": {}, \"audio_in\": {}, \"audio_in_lens\": {}, \"audio_out\": {}, \"modality_flag\": {}, \"supervision_mask\": {}}}",
        seq2("int64"),
        seq2("float32"),
        seq(val("int64")),
        seq2("int64"),
        seq2("int64"),
        seq2("bool"),
    )
}

// ---- read --------------------------------------------------------------------

/// `load_from_disk(dir)` — read the dataset rows back into [`RawRow`]s on `device`.
/// Reads `state.json` for the shard list (falling back to scanning `*.arrow`), then
/// each Arrow IPC stream; every batch row reconstructs the six column tensors.
pub fn load_from_disk(dir: &Path, device: &Device) -> Result<Vec<RawRow>> {
    let shards = data_files(dir)?;
    let mut rows = Vec::new();
    for shard in shards {
        let file = std::fs::File::open(&shard)
            .map_err(|e| err(format!("open {}: {e}", shard.display())))?;
        let reader = StreamReader::try_new(file, None).map_err(err)?;
        for batch in reader {
            let batch = batch.map_err(err)?;
            read_batch(&batch, device, &mut rows)?;
        }
    }
    Ok(rows)
}

/// Resolve the `.arrow` shard paths: prefer `state.json`'s `_data_files`, else scan.
fn data_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    if let Ok(bytes) = std::fs::read(dir.join("state.json")) {
        if let Ok(state) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(files) = state.get("_data_files").and_then(|v| v.as_array()) {
                let paths: Vec<_> = files
                    .iter()
                    .filter_map(|f| f.get("filename").and_then(|n| n.as_str()))
                    .map(|n| dir.join(n))
                    .collect();
                if !paths.is_empty() {
                    return Ok(paths);
                }
            }
        }
    }
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .map_err(err)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("arrow"))
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(err(format!(
            "load_from_disk: no .arrow shards under {}",
            dir.display()
        )));
    }
    Ok(out)
}

fn col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    b.column_by_name(name)
        .ok_or_else(|| err(format!("load_from_disk: missing column `{name}`")))
}
fn as_list(a: &ArrayRef) -> Result<&ListArray> {
    a.as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| err("expected a List column"))
}

/// One outer-list element (one sample) of a `List<List<T>>` column → `Vec<Vec>`.
fn ll_i64(outer: &ListArray, i: usize) -> Result<Vec<Vec<i64>>> {
    let inner = outer.value(i);
    let inner = as_list(&inner)?;
    (0..inner.len())
        .map(|j| {
            let v = inner.value(j);
            let v = v
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| err("expected Int64"))?;
            Ok(v.values().to_vec())
        })
        .collect()
}
fn ll_f32(outer: &ListArray, i: usize) -> Result<Vec<Vec<f32>>> {
    let inner = outer.value(i);
    let inner = as_list(&inner)?;
    (0..inner.len())
        .map(|j| {
            let v = inner.value(j);
            let v = v
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| err("expected Float32"))?;
            Ok(v.values().to_vec())
        })
        .collect()
}
fn ll_bool(outer: &ListArray, i: usize) -> Result<Vec<Vec<bool>>> {
    let inner = outer.value(i);
    let inner = as_list(&inner)?;
    (0..inner.len())
        .map(|j| {
            let v = inner.value(j);
            let v = v
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| err("expected Boolean"))?;
            Ok((0..v.len()).map(|k| v.value(k)).collect())
        })
        .collect()
}

fn read_batch(batch: &RecordBatch, dev: &Device, out: &mut Vec<RawRow>) -> Result<()> {
    let text = as_list(col(batch, "text")?)?;
    let audio_in = as_list(col(batch, "audio_in")?)?;
    let lens = as_list(col(batch, "audio_in_lens")?)?;
    let audio_out = as_list(col(batch, "audio_out")?)?;
    let modality = as_list(col(batch, "modality_flag")?)?;
    let supervision = as_list(col(batch, "supervision_mask")?)?;

    for i in 0..batch.num_rows() {
        // audio_in_lens is a single-nested List<Int64>.
        let lens_i = lens.value(i);
        let lens_v = lens_i
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| err("expected Int64 lens"))?;
        out.push(RawRow {
            text: tensor2_i64(&ll_i64(text, i)?, dev)?,
            audio_in: tensor2_f32(&ll_f32(audio_in, i)?, dev)?,
            audio_in_lens: Tensor::from_vec(lens_v.values().to_vec(), (lens_v.len(),), dev)?,
            audio_out: tensor2_i64(&ll_i64(audio_out, i)?, dev)?,
            modality_flag: tensor2_i64(&ll_i64(modality, i)?, dev)?,
            supervision_mask: tensor2_u8(&ll_bool(supervision, i)?, dev)?,
        });
    }
    Ok(())
}
