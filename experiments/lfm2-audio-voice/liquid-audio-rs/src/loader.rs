//! Model loading — `config.json` → configs → safetensors VarBuilder → model.
//!
//! Mirrors `LFM2AudioModel.from_pretrained` / `LFM2AudioProcessor.from_pretrained`:
//! parse the config, construct the typed configs, memory-map the safetensors, and
//! build the model + processor. `dtype` mirrors the Python keyword arg
//! (`dtype: torch.dtype = torch.bfloat16`): pass `DType::BF16` on CUDA/Metal to
//! match the deployed model, or `DType::F32` for the parity harness (which dumps
//! the Python reference at `torch.float32`) and for CPU.
//!
//! Note: the on-disk checkpoint is stored bf16, so `DType::F32` still loads the
//! *faithful* (bf16-rounded) weight values and upcasts them — on CPU this is the
//! correct path, because candle's CPU backend has no bf16 matmul kernel. Request-
//! ing `DType::BF16` on a CPU device is therefore rejected up front (see guard)
//! rather than failing later with a cryptic "unsupported dtype BF16 for op
//! matmul". Expects a local model directory (download the HF repo first; hf-hub
//! auto-download is a follow-up).

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use moshi::mimi;
use serde_json::Value;

use crate::detokenizer::LFM2AudioDetokenizer;
use crate::model::conformer::encoder::ConformerEncoderConfig;
use crate::model::conformer::processor::FilterbankFeatures;
use crate::model::lfm2_audio::{DepthformerConfig, LFM2AudioModel};
use crate::model::lfm2_hf::Lfm2Config;
use crate::processor::{LFM2AudioProcessor, PreprocessorConfig};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(err)? {
        let p = entry.map_err(err)?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            out.push(p);
        }
    }
    if out.is_empty() {
        return Err(err(format!("no .safetensors in {}", dir.display())));
    }
    out.sort();
    Ok(out)
}

fn parse_encoder(e: &Value) -> ConformerEncoderConfig {
    let u = |k: &str| e[k].as_u64().unwrap_or(0) as usize;
    let feat_out = e["feat_out"].as_i64().unwrap_or(-1);
    let conv_ch = e["subsampling_conv_channels"].as_i64().unwrap_or(-1);
    ConformerEncoderConfig {
        feat_in: u("feat_in"),
        feat_out: if feat_out > 0 { feat_out as usize } else { 0 },
        n_layers: u("n_layers"),
        d_model: u("d_model"),
        subsampling_factor: u("subsampling_factor"),
        subsampling_conv_channels: if conv_ch > 0 { conv_ch as usize } else { 0 },
        ff_expansion_factor: u("ff_expansion_factor"),
        n_heads: u("n_heads"),
        conv_kernel_size: u("conv_kernel_size"),
        xscaling: e["xscaling"].as_bool().unwrap_or(true),
    }
}

/// Resolve `repo_id` (or a local path) via [`get_model_dir`] — snapshot-
/// downloading from the Hub if needed — then load at `dtype`. This is the
/// faithful analog of the Python `LFM2AudioModel.from_pretrained(repo_id, ...)`
/// entry point (which calls `get_model_dir` internally).
pub fn from_pretrained_hub(
    repo_id: &str,
    revision: Option<&str>,
    dtype: DType,
    device: &Device,
) -> Result<(LFM2AudioModel, LFM2AudioProcessor)> {
    let dir = crate::utils::get_model_dir(repo_id, revision).map_err(err)?;
    from_pretrained(&dir, dtype, device)
}

/// Load the main model + processor from a local model directory, at `dtype`
/// (mirrors the Python `dtype=` keyword; `DType::BF16` matches the deployed
/// model, `DType::F32` matches the parity reference).
pub fn from_pretrained(dir: &Path, dtype: DType, device: &Device) -> Result<(LFM2AudioModel, LFM2AudioProcessor)> {
    if dtype == DType::BF16 && device.is_cpu() {
        return Err(err(
            "bf16 on CPU is unsupported (candle has no CPU bf16 matmul); use DType::F32 \
             — it still loads the bf16-stored weights and upcasts them faithfully",
        ));
    }
    let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?).map_err(err)?;

    let lfm_cfg: Lfm2Config = serde_json::from_value(config["lfm"].clone()).map_err(err)?;
    let enc_cfg = parse_encoder(&config["encoder"]);
    let depth_cfg = DepthformerConfig {
        layers: config["depthformer"]["layers"].as_u64().unwrap_or(0) as usize,
        dim: config["depthformer"]["dim"].as_u64().unwrap_or(0) as usize,
        tie: config["depthformer"]["tie"].as_bool().unwrap_or(true),
    };
    let codebooks = config["codebooks"].as_u64().unwrap_or(8) as usize;
    let n_text = config["interleaved_n_text"].as_u64().unwrap_or(1) as usize;
    let n_audio = config["interleaved_n_audio"].as_u64().unwrap_or(1) as usize;

    let safes = safetensors_in(dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safes, dtype, device)? };
    let model = LFM2AudioModel::new(lfm_cfg, &enc_cfg, &depth_cfg, codebooks, n_text, n_audio, vb)?;

    let prep: PreprocessorConfig = serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    let detok = load_detokenizer(dir, dtype, device).ok();
    let mimi = load_mimi(dir, codebooks, device)?;
    let proc = LFM2AudioProcessor::new(tokenizer, audio, detok, mimi, device.clone());

    Ok((model, proc))
}

/// Load the Kyutai Mimi codec (v1 `processor.mimi` audio-out) from
/// `<dir>/tokenizer-e351c8d8-checkpoint125.safetensors` if present. Reused from
/// the `moshi` crate (Kyutai's own — the Rust port of the vendored
/// `liquid_audio/moshi`, so it loads the moshi-format checkpoint). Returns `None`
/// if the file is absent; propagates a real load error (no silent fallback) if
/// the file exists but can't be loaded.
fn load_mimi(dir: &Path, codebooks: usize, device: &Device) -> Result<Option<mimi::Mimi>> {
    let path = dir.join("tokenizer-e351c8d8-checkpoint125.safetensors");
    if !path.exists() {
        return Ok(None);
    }
    let p = path.to_str().ok_or_else(|| err("non-utf8 mimi weights path"))?;
    Ok(Some(mimi::load(p, Some(codebooks), device)?))
}

/// Load the LFM2.5 audio detokenizer from `<dir>/audio_detokenizer/` if present.
fn load_detokenizer(dir: &Path, dtype: DType, device: &Device) -> Result<LFM2AudioDetokenizer> {
    let detok_dir = dir.join("audio_detokenizer");
    let mut cfg: Value = serde_json::from_str(&fs::read_to_string(detok_dir.join("config.json")).map_err(err)?).map_err(err)?;
    // llama.cpp → transformers compat: "sliding_attention" → "full_attention"
    if let Some(arr) = cfg["layer_types"].as_array_mut() {
        for v in arr.iter_mut() {
            if v.as_str() == Some("sliding_attention") {
                *v = Value::String("full_attention".into());
            }
        }
    }
    let sliding_window = cfg["sliding_window"].as_u64().unwrap_or(30) as usize;
    let lfm_cfg: Lfm2Config = serde_json::from_value(cfg).map_err(err)?;
    let safes = safetensors_in(&detok_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safes, dtype, device)? };
    LFM2AudioDetokenizer::new(lfm_cfg, sliding_window, vb)
}
