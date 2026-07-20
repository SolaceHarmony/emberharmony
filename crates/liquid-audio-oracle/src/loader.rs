//! Offline training-oracle loading.
//!
//! `safetensors.cpp` is the only checkpoint loader and parser. Rust receives an
//! opaque resident image and component-qualified validated views. Mutable
//! Candle variables are initialized from those views because autograd requires
//! owned writable storage; that compatibility copy is explicit and counted.

use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use serde_json::Value;

use crate::audio_out::{AudioEncoder, MimiEncoder};
use crate::model::lfm2_audio::{DepthformerConfig, LFM2AudioModel, LossConf};
use crate::model::lfm2_hf::Lfm2Config;
use crate::processor::{FilterbankFeatures, LFM2AudioProcessor, PreprocessorConfig};
use crate::weights::{ResidentWeights, WeightComponent};

fn err(error: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

fn req_usize(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| err(format!("config: missing/invalid required field `{key}`")))
}

pub struct TrainableLoad {
    pub model: LFM2AudioModel,
    pub varmap: VarMap,
    pub processor: LFM2AudioProcessor,
    resident: ResidentWeights,
    pub codebooks: usize,
    pub codebook_weight: String,
    pub semantic_codebook_factor: f64,
    pub text_loss_multiplier: f64,
    pub audio_loss_multiplier: f64,
}

impl TrainableLoad {
    pub fn compatibility_copies(&self) -> (usize, u64) {
        let copies = self.resident.compatibility_copies();
        (copies.tensors, copies.bytes)
    }
}

/// Build the mutable Candle training oracle from one native Main+Codec image.
/// There is no inference loader or Rust checkpoint fallback in this crate.
pub fn from_pretrained_trainable(dir: &Path, device: &Device) -> Result<TrainableLoad> {
    crate::threads::configure_intraop_threads();
    let config: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("config.json")).map_err(err)?)
            .map_err(err)?;
    let codec = dir.join("tokenizer-e351c8d8-checkpoint125.safetensors");
    if !codec.is_file() {
        return Err(err(format!(
            "training snapshot is missing Mimi codec {}",
            codec.display()
        )));
    }
    let resident = ResidentWeights::open_bundle(dir, &codec).map_err(err)?;
    let dtype = resident.dtype(WeightComponent::Main).map_err(err)?;
    if dtype == DType::BF16 && device.is_cpu() {
        return Err(err(
            "trainable CPU bf16 is unsupported; select a differentiable device backend",
        ));
    }

    let lfm: Lfm2Config = serde_json::from_value(config["lfm"].clone()).map_err(err)?;
    let depth = &config["depthformer"];
    let depth = DepthformerConfig {
        layers: req_usize(depth, "layers")?,
        dim: req_usize(depth, "dim")?,
        tie: depth
            .get("tie")
            .and_then(Value::as_bool)
            .ok_or_else(|| err("config: missing/invalid required field `depthformer.tie`"))?,
    };
    let codebooks = req_usize(&config, "codebooks")?;
    let codebook_weight = config["codebook_weight"]
        .as_str()
        .unwrap_or("linear")
        .to_owned();
    let semantic_codebook_factor = config["semantic_codebook_factor"].as_f64().unwrap_or(1.0);
    let text_loss_multiplier = config["text_loss_multiplier"].as_f64().unwrap_or(1.0);
    let audio_loss_multiplier = config["audio_loss_multiplier"].as_f64().unwrap_or(1.0);
    let loss = LossConf {
        codebook_weight: codebook_weight.clone(),
        semantic_codebook_factor,
        text_loss_multiplier,
        audio_loss_multiplier,
    };

    let tokenizer = LFM2AudioProcessor::load_tokenizer(dir)?;
    let special = LFM2AudioProcessor::special_token_ids(&tokenizer)?;
    if let Some(eos) = config["lfm"]["eos_token_id"].as_u64() {
        if eos as u32 != special.im_end {
            return Err(err(format!(
                "config lfm.eos_token_id ({eos}) disagrees with tokenizer <|im_end|> ({})",
                special.im_end
            )));
        }
    }
    crate::chat_template::verify_snapshot(dir, &tokenizer)?;

    let varmap = VarMap::new();
    let builder = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = LFM2AudioModel::new(lfm, &depth, codebooks, &loss, builder)?;
    resident.copy_into_varmap(WeightComponent::Main, &varmap)?;

    let prep: PreprocessorConfig =
        serde_json::from_value(config["preprocessor"].clone()).map_err(err)?;
    let audio = FilterbankFeatures::new(prep.mel_config(), device)?;
    let builder = resident
        .candle_builder(WeightComponent::Codec, device)
        .map_err(err)?;
    let mimi = ::moshi::mimi::Mimi::new(::moshi::mimi::Config::v0_1(Some(codebooks)), builder)?;
    let mimi: Option<Box<dyn AudioEncoder>> = Some(Box::new(MimiEncoder::new(mimi)));
    let processor = LFM2AudioProcessor::new(tokenizer, audio, mimi, device.clone());

    Ok(TrainableLoad {
        model,
        varmap,
        processor,
        resident,
        codebooks,
        codebook_weight,
        semantic_codebook_factor,
        text_loss_multiplier,
        audio_loss_multiplier,
    })
}
