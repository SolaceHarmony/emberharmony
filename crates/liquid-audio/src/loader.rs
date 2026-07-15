//! Model loading — `config.json` + one native resident weight image → model.
//!
//! Mirrors `LFM2AudioModel.from_pretrained` / `LFM2AudioProcessor.from_pretrained`:
//! parse the config, construct the typed configs, load safetensors once through
//! the native C++ image, and build the model + processor. Remaining Candle model
//! components use the explicit compatibility bridge in `compute/weights.rs`;
//! production loading does not reparse or remap the checkpoint through Candle.
//! Persistent model weights keep the floating dtype stored in the checkpoint.
//! This module does not inspect process environment for model or device choices;
//! its host resolves persisted application settings and passes both explicitly.
//! Expects a local model directory (download the HF repo first; hf-hub
//! auto-download is a follow-up).

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use serde_json::Value;

use crate::audio_out::{AudioDetokenizer, MimiDetokenizer};

use crate::detokenizer::LFM2AudioDetokenizer;
use crate::model::conformer::encoder::ConformerEncoderConfig;
use crate::model::lfm2_audio::{DepthformerConfig, LFM2AudioModel, LossConf};
use crate::model::lfm2_hf::Lfm2Config;
use crate::moshi::models::get_mimi;
use crate::processor::FilterbankFeatures;
use crate::processor::{LFM2AudioProcessor, PreprocessorConfig};
use crate::weights::ResidentWeights;

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

fn model_safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let safes: Vec<PathBuf> = safetensors_in(dir)?
        .into_iter()
        .filter(|p| {
            !p.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("tokenizer-"))
        })
        .collect();
    if safes.is_empty() {
        return Err(err(format!("no model .safetensors in {}", dir.display())));
    }
    Ok(safes)
}

fn is_floating_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::BF16
            | DType::F16
            | DType::F32
            | DType::F64
            | DType::F8E4M3
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0
    )
}

fn safetensors_floating_dtype(safes: &[PathBuf]) -> Result<DType> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(safes)? };
    let mut found: Option<(DType, String)> = None;
    for (name, view) in tensors.tensors() {
        let dtype: DType = view.dtype().try_into()?;
        if !is_floating_dtype(dtype) {
            continue;
        }
        match &found {
            Some((prev, first)) if *prev != dtype => {
                return Err(err(format!(
                    "mixed floating safetensor dtypes: `{first}` is {prev:?}, `{name}` is {dtype:?}"
                )));
            }
            None => found = Some((dtype, name)),
            _ => {}
        }
    }
    found
        .map(|(dtype, _)| dtype)
        .ok_or_else(|| err("checkpoint has no floating safetensor tensors"))
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
        self_attention_model: e["self_attention_model"]
            .as_str()
            .unwrap_or("rel_pos")
            .to_string(),
    })
}

/// Resolve `repo_id` (or a local path) via [`get_model_dir`] — snapshot-
/// downloading from the Hub if needed — then load using the floating dtype stored
/// in the model safetensors.
pub fn from_pretrained_hub(
    repo_id: &str,
    revision: Option<&str>,
    device: &Device,
) -> Result<(LFM2AudioModel, LFM2AudioProcessor)> {
    let dir = crate::utils::get_model_dir(repo_id, revision).map_err(err)?;
    from_pretrained(&dir, device)
}

/// Load the main model + processor from a local model directory. Persistent
/// floating weights keep the dtype stored in the model safetensors.
pub fn from_pretrained(
    dir: &Path,
    device: &Device,
) -> Result<(LFM2AudioModel, LFM2AudioProcessor)> {
    // Size the global rayon pool (candle's matmul/conv use it) like torch's intra-op
    // default — Apple-Silicon performance cores, not all logical cores. Must run before
    // the first tensor op; idempotent. (No-op for thread parallelism on the Metal path,
    // but the f64 mel front-end and other CPU ops still benefit.)
    crate::threads::configure_intraop_threads();
    let config: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?)
            .map_err(err)?;
    let resident = ResidentWeights::open(dir).map_err(err)?;
    let dtype = resident.dtype();
    if dtype == DType::BF16 && device.is_cpu() && !crate::bf16_gemm::bf16_gemm_available() {
        return Err(err(
            "CPU bf16 inference requires the in-tree NEON BFMMLA matmul kernel, but it is \
             unavailable on this machine. Use Metal or a CPU with FEAT_BF16.",
        ));
    }

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
        codebook_weight: config["codebook_weight"]
            .as_str()
            .unwrap_or("linear")
            .to_string(),
        semantic_codebook_factor: config["semantic_codebook_factor"].as_f64().unwrap_or(1.0),
        text_loss_multiplier: config["text_loss_multiplier"].as_f64().unwrap_or(1.0),
        audio_loss_multiplier: config["audio_loss_multiplier"].as_f64().unwrap_or(1.0),
    };

    // The tokenizer loads BEFORE the model: the generation-control token ids
    // (<|im_end|>, <|text_end|>, <|audio_start|>) are defined by the snapshot and
    // pass through to the generation loops — never hardcoded. The config's own
    // `lfm.eos_token_id` cross-checks the tokenizer's <|im_end|> so a mismatched
    // snapshot fails at load, not as silent garbage generation.
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    let special = LFM2AudioProcessor::special_token_ids(&tokenizer)?;
    if let Some(eos) = config["lfm"]["eos_token_id"].as_u64() {
        if eos as u32 != special.im_end {
            return Err(err(&format!(
                "config lfm.eos_token_id ({eos}) disagrees with tokenizer <|im_end|> ({})",
                special.im_end
            )));
        }
    }
    // ChatState's turn format must tokenize identically to what the snapshot's own
    // chat_template.jinja renders — template drift fails the load, never prompts
    // the model off-distribution.
    crate::chat_template::verify_snapshot(dir, &tokenizer)?;

    let model = LFM2AudioModel::new_resident(
        lfm_cfg, &enc_cfg, &depth_cfg, codebooks, n_text, n_audio, special, &loss_conf, resident,
        device,
    )?;
    let copies = model.compatibility_copies();
    eprintln!(
        "[voice] native checkpoint resident: {} tensors / {} bytes; Candle compatibility: {} tensors / {} bytes copied",
        model.resident_weights().map_or(0, |weights| weights.len()),
        model
            .resident_weights()
            .map_or(0, |weights| weights.resident_bytes()),
        copies.tensors,
        copies.bytes
    );

    let prep: PreprocessorConfig =
        serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
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
        load_mimi(dir, codebooks, device)?.map(|m| Box::new(m) as Box<dyn AudioDetokenizer>);
    let audio_out: Option<Box<dyn AudioDetokenizer>> = if dir.join("audio_detokenizer").is_dir() {
        Some(Box::new(load_detokenizer(dir, device)?))
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
/// the checkpoint values into those `Var`s. The params are real `Var`s the candle
/// optimizer can update, not frozen mmaped tensors.
///
/// CPU BF16 training is rejected because the NEON inference matmul bridge is
/// no-bwd; callers should not upcast persistent BF16 weights to F32 just to make
/// CPU training run.
pub fn from_pretrained_trainable(dir: &Path, device: &Device) -> Result<TrainableLoad> {
    let config: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?)
            .map_err(err)?;
    let safes = model_safetensors_in(dir)?;
    let dtype = safetensors_floating_dtype(&safes)?;
    if dtype == DType::BF16 && device.is_cpu() {
        return Err(err(
            "trainable CPU bf16 is unsupported because the NEON bf16 matmul bridge is no-bwd",
        ));
    }

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
    let codebook_weight = config["codebook_weight"]
        .as_str()
        .unwrap_or("linear")
        .to_string();
    let semantic_codebook_factor = config["semantic_codebook_factor"].as_f64().unwrap_or(1.0);
    let text_loss_multiplier = config["text_loss_multiplier"].as_f64().unwrap_or(1.0);
    let audio_loss_multiplier = config["audio_loss_multiplier"].as_f64().unwrap_or(1.0);
    let loss_conf = LossConf {
        codebook_weight: codebook_weight.clone(),
        semantic_codebook_factor,
        text_loss_multiplier,
        audio_loss_multiplier,
    };

    // Same tokenizer-defined token-id pass-through as the inference loader.
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    let special = LFM2AudioProcessor::special_token_ids(&tokenizer)?;

    // Build over a fresh VarMap so `LFM2AudioModel::new` allocates trainable Vars,
    // then load the checkpoint into them (faithful: the architecture defines the
    // param set, the safetensors provide the pretrained init).
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = LFM2AudioModel::new(
        lfm_cfg, &enc_cfg, &depth_cfg, codebooks, n_text, n_audio, special, &loss_conf, vb,
    )?;
    // Load the checkpoint into the freshly-allocated Vars. `VarMap::load` is *not*
    // usable here: it opens a single file and demands every Var be present in it,
    // so it breaks on a sharded checkpoint *and* on the extra non-model safetensors
    // in the dir (the Mimi tokenizer `tokenizer-…checkpoint125.safetensors`). Mirror
    // `VarBuilder::from_mmaped_safetensors` instead — one lazy index over every
    // shard, pulling each param by name. Strict: a param missing from *every* shard
    // is a hard error (never a silent zero-init); extra tensors in the dir that no
    // Var names are simply never requested.
    let shards = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&safes)? };
    {
        let mut ws = varmap.data().lock().unwrap();
        for (name, var) in ws.iter_mut() {
            let tensor = shards.load(name, var.device()).map_err(|e| {
                err(format!(
                    "checkpoint: param `{name}` not found in any shard: {e}"
                ))
            })?;
            // Cast the STORED checkpoint dtype to the Var's dtype before `set`. Unlike
            // `VarBuilder::get` (which casts on read), `Var::set` is a same-dtype
            // storage copy and errors on a dtype mismatch. This is a no-op for
            // model-card-faithful loads; the cast remains because `Var::set`
            // requires exact dtype equality.
            let tensor = tensor.to_dtype(var.dtype())?;
            var.set(&tensor)?;
        }
    }

    let prep: PreprocessorConfig =
        serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    // Mimi codec + LFM2 detokenizer loaded independently (see `from_pretrained`):
    // training preprocessing also encodes audio-out via `processor.mimi.encode`.
    let mimi: Option<Box<dyn AudioDetokenizer>> =
        load_mimi(dir, codebooks, device)?.map(|m| Box::new(m) as Box<dyn AudioDetokenizer>);
    let audio_out: Option<Box<dyn AudioDetokenizer>> = if dir.join("audio_detokenizer").is_dir() {
        Some(Box::new(load_detokenizer(dir, device)?))
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
fn load_mimi(dir: &Path, codebooks: usize, device: &Device) -> Result<Option<MimiDetokenizer>> {
    let path = dir.join("tokenizer-e351c8d8-checkpoint125.safetensors");
    if !path.exists() {
        return Ok(None);
    }
    let p = path
        .to_str()
        .ok_or_else(|| err("non-utf8 mimi weights path"))?;
    // Both halves from the same checkpoint: the moshi codec (turn-level tooling:
    // encode + the one-shot decode the byte oracles pin) and the NATIVE streaming
    // decoder (the per-frame hot path). Native init failure is a hard load error —
    // the kernel is the streaming substrate, not an optional acceleration.
    let mimi = get_mimi(p, codebooks, device)?;
    let native = crate::mimi_native::NativeMimi::new(&path, codebooks).map_err(err)?;
    Ok(Some(MimiDetokenizer::new(mimi, native)))
}

/// Load the LFM2.5 audio detokenizer from `<dir>/audio_detokenizer/` if present.
fn load_detokenizer(dir: &Path, device: &Device) -> Result<LFM2AudioDetokenizer> {
    let detok_dir = dir.join("audio_detokenizer");
    let mut cfg: Value =
        serde_json::from_str(&fs::read_to_string(detok_dir.join("config.json")).map_err(err)?)
            .map_err(err)?;
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
    let resident = ResidentWeights::open(&detok_dir).map_err(err)?;
    LFM2AudioDetokenizer::new_resident(lfm_cfg, sliding_window, resident, device)
}
