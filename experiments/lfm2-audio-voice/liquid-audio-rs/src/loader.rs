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
use candle_nn::{VarBuilder, VarMap};
use moshi::mimi;
use serde_json::Value;

use crate::audio_out::{AudioDetokenizer, MimiDetokenizer};

use crate::detokenizer::LFM2AudioDetokenizer;
use crate::model::conformer::encoder::ConformerEncoderConfig;
use crate::model::conformer::processor::FilterbankFeatures;
use crate::model::lfm2_audio::{DepthformerConfig, LFM2AudioModel, LossConf};
use crate::model::lfm2_hf::Lfm2Config;
use crate::processor::{LFM2AudioProcessor, PreprocessorConfig};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// Required uint config field — hard error if missing/invalid (no silent default).
/// Mirrors Python's dataclass `TypeError` on a missing required kwarg.
fn req_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| err(format!("config: missing/invalid required field `{key}`")))
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

fn parse_encoder(e: &Value) -> Result<ConformerEncoderConfig> {
    // Structural fields are required (a wrong default = a silently-broken model).
    // `feat_out`/`subsampling_conv_channels` keep the upstream `-1 → use d_model`
    // sentinel (a genuine NeMo default, not a silent fallback); `xscaling` keeps
    // the upstream default of True.
    let feat_out = e["feat_out"].as_i64().unwrap_or(-1);
    let conv_ch = e["subsampling_conv_channels"].as_i64().unwrap_or(-1);
    Ok(ConformerEncoderConfig {
        feat_in: req_usize(e, "feat_in")?,
        feat_out: if feat_out > 0 { feat_out as usize } else { 0 },
        n_layers: req_usize(e, "n_layers")?,
        d_model: req_usize(e, "d_model")?,
        subsampling_factor: req_usize(e, "subsampling_factor")?,
        subsampling_conv_channels: if conv_ch > 0 { conv_ch as usize } else { 0 },
        ff_expansion_factor: req_usize(e, "ff_expansion_factor")?,
        n_heads: req_usize(e, "n_heads")?,
        conv_kernel_size: req_usize(e, "conv_kernel_size")?,
        xscaling: e["xscaling"].as_bool().unwrap_or(true),
        self_attention_model: e["self_attention_model"].as_str().unwrap_or("rel_pos").to_string(),
    })
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
    // Size the global rayon pool (candle's matmul/conv use it) like torch's intra-op
    // default — Apple-Silicon performance cores, not all logical cores. Must run before
    // the first tensor op; idempotent. (No-op for thread parallelism on the Metal path,
    // but the f64 mel front-end and other CPU ops still benefit.)
    crate::threads::configure_intraop_threads();
    if dtype == DType::BF16 && device.is_cpu() {
        // candle supports bf16 broadly (dtype, conversions, every elementwise op, AND
        // Metal matmul). The single gap is candle 0.9.2's CPU *gemm* matmul allowlist
        // (`cpu_backend/mod.rs`: `DType::F16 | F32 | F64`; the Accelerate path is F32/F64
        // only) — bf16 falls through to `UnsupportedDTypeForOp`. It's a candle allowlist
        // choice, not a `gemm` limit (`gemm-f16` handles the `half` types), so true bf16
        // CPU matmul is a `candle_ext` backport away if ever needed. For now f32 on CPU is
        // the right call: the bf16→f32 weight upcast is lossless and the f32 parity
        // goldens were dumped at f32, so this is the faithful CPU/test path.
        return Err(err(
            "bf16 matmul is not in candle 0.9.2's CPU gemm allowlist (F16/F32/F64); use \
             DType::F32 on CPU — it loads the bf16 weights and upcasts them losslessly. \
             (bf16 runs natively on Metal.)",
        ));
    }
    let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?).map_err(err)?;

    let lfm_cfg: Lfm2Config = serde_json::from_value(config["lfm"].clone()).map_err(err)?;
    let enc_cfg = parse_encoder(&config["encoder"])?;
    let depth = &config["depthformer"];
    let depth_cfg = DepthformerConfig {
        layers: req_usize(depth, "layers")?,
        dim: req_usize(depth, "dim")?,
        tie: depth
            .get("tie")
            .and_then(Value::as_bool)
            .ok_or_else(|| err("config: missing/invalid required field `depthformer.tie`"))?,
    };
    let codebooks = req_usize(&config, "codebooks")?;
    let n_text = req_usize(&config, "interleaved_n_text")?;
    let n_audio = req_usize(&config, "interleaved_n_audio")?;

    // Loss-weight hyperparameters (Python `LFM2AudioConfig`) feeding the
    // `audio_loss_weights` buffer + loss multipliers built in `LFM2AudioModel::new`.
    let loss_conf = LossConf {
        codebook_weight: config["codebook_weight"].as_str().unwrap_or("linear").to_string(),
        semantic_codebook_factor: config["semantic_codebook_factor"].as_f64().unwrap_or(1.0),
        text_loss_multiplier: config["text_loss_multiplier"].as_f64().unwrap_or(1.0),
        audio_loss_multiplier: config["audio_loss_multiplier"].as_f64().unwrap_or(1.0),
    };

    let safes = safetensors_in(dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safes, dtype, device)? };
    let model = LFM2AudioModel::new(lfm_cfg, &enc_cfg, &depth_cfg, codebooks, n_text, n_audio, &loss_conf, vb)?;

    let prep: PreprocessorConfig = serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    // Two INDEPENDENT audio backends, mirroring the Python processor's separate
    // `_mimi` / `_audio_detokenizer` fields:
    //  - `mimi`: the Mimi codec (`tokenizer-…checkpoint125.safetensors`), loaded
    //    whenever present. The data mapper's `_encode_audio_out` calls
    //    `processor.mimi.encode` even on full LFM2.5 snapshots, so the encoder must
    //    survive regardless of which backend decodes.
    //  - `audio_out`: the in-tree LFM2 detokenizer (`audio_detokenizer/`, LFM2.5
    //    only) used by `decode`. `None` for v1, where `decode` falls back to `mimi`.
    // No silent fallback for full snapshots: a present `audio_detokenizer/`
    // propagates any load error rather than quietly dropping to Mimi.
    let mimi: Option<Box<dyn AudioDetokenizer>> =
        load_mimi(dir, codebooks, device)?.map(|m| Box::new(MimiDetokenizer::new(m)) as Box<dyn AudioDetokenizer>);
    let audio_out: Option<Box<dyn AudioDetokenizer>> = if dir.join("audio_detokenizer").is_dir() {
        Some(Box::new(load_detokenizer(dir, dtype, device)?))
    } else {
        None
    };
    let proc = LFM2AudioProcessor::new(tokenizer, audio, audio_out, mimi, device.clone());

    Ok((model, proc))
}

/// Loss-weight hyperparameters read from `config.json`, needed by the training
/// `forward` (Python `LFM2AudioModel.forward` reads them off `self.conf` and the
/// `audio_loss_weights` buffer built in `__init__`). Mirrors the relevant fields
/// of `LFM2AudioConfig` so the [`crate::trainer`] can compute the weighted loss
/// without re-parsing the model. Not `Clone`/`Debug`: it owns a live `VarMap`
/// (the trainable parameter set) and the model holding those `Var`s.
pub struct TrainableLoad {
    pub model: LFM2AudioModel,
    /// The `VarMap` backing every trainable parameter (candle's analog of
    /// `model.parameters()` — the optimizer steps these `Var`s in place).
    pub varmap: VarMap,
    pub processor: LFM2AudioProcessor,
    /// `codebooks`, `codebook_weight`, `semantic_codebook_factor`,
    /// `text_loss_multiplier`, `audio_loss_multiplier` — the loss config.
    pub codebooks: usize,
    pub codebook_weight: String,
    pub semantic_codebook_factor: f64,
    pub text_loss_multiplier: f64,
    pub audio_loss_multiplier: f64,
}

/// Trainable analog of [`from_pretrained`]: build the model from a `VarMap`-backed
/// `VarBuilder` so every weight is a trainable [`candle_core::Var`] (the candle
/// equivalent of `nn.Module.parameters()` participating in autograd), then load
/// the checkpoint values into those `Var`s. Mirrors the Python `Trainer.__init__`
/// step `LFM2AudioModel.from_pretrained(model_id, dtype=torch.bfloat16)` followed
/// by `accelerator.prepare(model, ...)` — except the params are real `Var`s the
/// candle optimizer can update, not frozen mmaped tensors.
///
/// `dtype` mirrors the Python `torch.bfloat16`; pass `DType::F32` on CPU (candle
/// has no CPU bf16 matmul) — the bf16-stored weights upcast faithfully.
pub fn from_pretrained_trainable(dir: &Path, dtype: DType, device: &Device) -> Result<TrainableLoad> {
    if dtype == DType::BF16 && device.is_cpu() {
        return Err(err(
            "bf16 on CPU is unsupported (candle has no CPU bf16 matmul); use DType::F32",
        ));
    }
    let config: Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?).map_err(err)?;

    let lfm_cfg: Lfm2Config = serde_json::from_value(config["lfm"].clone()).map_err(err)?;
    let enc_cfg = parse_encoder(&config["encoder"])?;
    let depth = &config["depthformer"];
    let depth_cfg = DepthformerConfig {
        layers: req_usize(depth, "layers")?,
        dim: req_usize(depth, "dim")?,
        tie: depth
            .get("tie")
            .and_then(Value::as_bool)
            .ok_or_else(|| err("config: missing/invalid required field `depthformer.tie`"))?,
    };
    let codebooks = req_usize(&config, "codebooks")?;
    let n_text = req_usize(&config, "interleaved_n_text")?;
    let n_audio = req_usize(&config, "interleaved_n_audio")?;

    // Loss-weight hyperparameters (Python `LFM2AudioConfig`) — parsed up front so
    // they feed both `LFM2AudioModel::new` (the `audio_loss_weights` buffer + loss
    // multipliers) and the returned `TrainableLoad` below.
    let codebook_weight = config["codebook_weight"].as_str().unwrap_or("linear").to_string();
    let semantic_codebook_factor = config["semantic_codebook_factor"].as_f64().unwrap_or(1.0);
    let text_loss_multiplier = config["text_loss_multiplier"].as_f64().unwrap_or(1.0);
    let audio_loss_multiplier = config["audio_loss_multiplier"].as_f64().unwrap_or(1.0);
    let loss_conf = LossConf {
        codebook_weight: codebook_weight.clone(),
        semantic_codebook_factor,
        text_loss_multiplier,
        audio_loss_multiplier,
    };

    // Build over a fresh VarMap so `LFM2AudioModel::new` allocates trainable Vars,
    // then load the checkpoint into them (faithful: the architecture defines the
    // param set, the safetensors provide the pretrained init).
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = LFM2AudioModel::new(lfm_cfg, &enc_cfg, &depth_cfg, codebooks, n_text, n_audio, &loss_conf, vb)?;
    // Load the checkpoint into the freshly-allocated Vars. `VarMap::load` is *not*
    // usable here: it opens a single file and demands every Var be present in it,
    // so it breaks on a sharded checkpoint *and* on the extra non-model safetensors
    // in the dir (the Mimi tokenizer `tokenizer-…checkpoint125.safetensors`). Mirror
    // `VarBuilder::from_mmaped_safetensors` instead — one lazy index over every
    // shard, pulling each param by name. Strict: a param missing from *every* shard
    // is a hard error (never a silent zero-init); extra tensors in the dir that no
    // Var names are simply never requested.
    let shards = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&safetensors_in(dir)?)? };
    {
        let mut ws = varmap.data().lock().unwrap();
        for (name, var) in ws.iter_mut() {
            let tensor = shards
                .load(name, var.device())
                .map_err(|e| err(format!("checkpoint: param `{name}` not found in any shard: {e}")))?;
            // Cast the STORED checkpoint dtype to the Var's dtype before `set`. Unlike
            // `VarBuilder::get` (which casts on read), `Var::set` is a same-dtype
            // storage copy and errors on a dtype mismatch. A bf16 checkpoint loaded
            // into F32 Vars — the required path for CPU training, where candle has no
            // bf16 matmul — would otherwise fail. Mirrors the Python upcast folded
            // into `from_pretrained(dtype=…)`. No-op when stored dtype == `dtype`.
            let tensor = tensor.to_dtype(var.dtype())?;
            var.set(&tensor)?;
        }
    }

    let prep: PreprocessorConfig = serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    // Mimi codec + LFM2 detokenizer loaded independently (see `from_pretrained`):
    // training preprocessing also encodes audio-out via `processor.mimi.encode`.
    let mimi: Option<Box<dyn AudioDetokenizer>> =
        load_mimi(dir, codebooks, device)?.map(|m| Box::new(MimiDetokenizer::new(m)) as Box<dyn AudioDetokenizer>);
    let audio_out: Option<Box<dyn AudioDetokenizer>> = if dir.join("audio_detokenizer").is_dir() {
        Some(Box::new(load_detokenizer(dir, dtype, device)?))
    } else {
        None
    };
    let processor = LFM2AudioProcessor::new(tokenizer, audio, audio_out, mimi, device.clone());

    Ok(TrainableLoad {
        model,
        varmap,
        processor,
        codebooks,
        codebook_weight,
        semantic_codebook_factor,
        text_loss_multiplier,
        audio_loss_multiplier,
    })
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
