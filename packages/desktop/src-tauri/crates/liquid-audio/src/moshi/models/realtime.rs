//! In-process realtime Moshi frame loop.
//!
//! This is the Rust form of the core loop in upstream `liquid_audio/moshi/server.py`:
//! fixed-size 24 kHz PCM frame -> Mimi streaming encode -> LMGen-style step ->
//! Mimi streaming decode. No websocket, no Python process, no HTTP boundary.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Error, Result, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

const DEFAULT_MOSHI_NAME: &str = "model.safetensors";
const DEFAULT_MIMI_NAME: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const DEFAULT_TEXT_TOKENIZER_NAME: &str = "tokenizer_spm_32k_3.model";

#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeMoshiFiles {
    pub moshi_weights: PathBuf,
    pub mimi_weights: PathBuf,
    pub tokenizer: PathBuf,
    pub model_type: String,
    pub params: RealtimeMoshiParams,
}

/// Sampling defaults from Python `moshi.models.lm.LMGen`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealtimeMoshiParams {
    pub max_steps: usize,
    pub seed: u64,
    pub use_sampling: bool,
    pub audio_temperature: f64,
    pub audio_top_k: usize,
    pub text_temperature: f64,
    pub text_top_k: usize,
}

impl Default for RealtimeMoshiParams {
    fn default() -> Self {
        Self {
            max_steps: 4096,
            seed: 42424242,
            use_sampling: true,
            audio_temperature: 0.8,
            audio_top_k: 250,
            text_temperature: 0.7,
            text_top_k: 25,
        }
    }
}

impl RealtimeMoshiParams {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn from_lm_gen_config(config: Option<&serde_json::Value>) -> Result<Self> {
        let mut params = Self::default();
        let Some(config) = config.and_then(serde_json::Value::as_object) else {
            return Ok(params);
        };
        if let Some(value) = config.get("max_steps").and_then(serde_json::Value::as_u64) {
            params.max_steps = usize::try_from(value)
                .map_err(|_| Error::Msg("lm_gen_config.max_steps does not fit usize".into()))?;
        }
        if let Some(value) = config
            .get("use_sampling")
            .and_then(serde_json::Value::as_bool)
        {
            params.use_sampling = value;
        }
        if let Some(value) = config.get("temp").and_then(serde_json::Value::as_f64) {
            params.audio_temperature = value;
        }
        if let Some(value) = config.get("top_k").and_then(serde_json::Value::as_u64) {
            params.audio_top_k = usize::try_from(value)
                .map_err(|_| Error::Msg("lm_gen_config.top_k does not fit usize".into()))?;
        }
        if let Some(value) = config.get("temp_text").and_then(serde_json::Value::as_f64) {
            params.text_temperature = value;
        }
        if let Some(value) = config.get("top_k_text").and_then(serde_json::Value::as_u64) {
            params.text_top_k = usize::try_from(value)
                .map_err(|_| Error::Msg("lm_gen_config.top_k_text does not fit usize".into()))?;
        }
        Ok(params)
    }

    fn sampling(&self, temperature: f64, top_k: usize) -> Sampling {
        if !self.use_sampling || temperature <= 1e-7 {
            return Sampling::ArgMax;
        }
        if top_k == 0 {
            return Sampling::All { temperature };
        }
        Sampling::TopK {
            k: top_k,
            temperature,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeMoshiEvent {
    InputAudioTokenFrame(Vec<u32>),
    TextToken(u32),
    AudioTokenFrame(Vec<u32>),
    Audio { pcm: Vec<f32>, rate: u32 },
}

pub struct RealtimeMoshi {
    device: Device,
    mimi: ::moshi::mimi::Mimi,
    lm: ::moshi::lm::LmModel,
    state: ::moshi::lm_generate_multistream::State,
    config: ::moshi::lm_generate_multistream::Config,
    params: RealtimeMoshiParams,
    text_token: u32,
    text_pad_token: u32,
    text_eop_token: u32,
    generated_codebooks: usize,
    sample_rate: u32,
    frame_size: usize,
    skip_frames: usize,
}

impl RealtimeMoshi {
    pub fn new(
        mut mimi: ::moshi::mimi::Mimi,
        lm: ::moshi::lm::LmModel,
        device: Device,
        params: RealtimeMoshiParams,
    ) -> Self {
        let cfg = ::moshi::lm_generate_multistream::Config::v0_1();
        let text_token = cfg.text_start_token;
        let text_pad_token = cfg.text_pad_token;
        let text_eop_token = cfg.text_eop_token;
        let generated_codebooks = cfg.generated_audio_codebooks;
        let sample_rate = mimi.config().sample_rate as u32;
        let frame_size = (mimi.config().sample_rate / mimi.config().frame_rate) as usize;
        let state = Self::new_state(lm.clone(), params, cfg.clone());
        mimi.reset_state();
        Self {
            device,
            mimi,
            lm,
            state,
            config: cfg,
            params,
            text_token,
            text_pad_token,
            text_eop_token,
            generated_codebooks,
            sample_rate,
            frame_size,
            skip_frames: 1,
        }
    }

    fn new_state(
        lm: ::moshi::lm::LmModel,
        params: RealtimeMoshiParams,
        config: ::moshi::lm_generate_multistream::Config,
    ) -> ::moshi::lm_generate_multistream::State {
        let audio_lp = LogitsProcessor::from_sampling(
            params.seed,
            params.sampling(params.audio_temperature, params.audio_top_k),
        );
        let text_lp = LogitsProcessor::from_sampling(
            params.seed,
            params.sampling(params.text_temperature, params.text_top_k),
        );
        ::moshi::lm_generate_multistream::State::new(
            lm,
            params.max_steps,
            audio_lp,
            text_lp,
            None,
            None,
            None,
            config,
        )
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn reset_stream(&mut self) {
        self.mimi.reset_state();
        self.state = Self::new_state(self.lm.clone(), self.params, self.config.clone());
        self.text_token = self.config.text_start_token;
        self.skip_frames = 1;
    }

    pub fn warmup(&mut self) -> Result<()> {
        for _ in 0..4 {
            let wav = Tensor::zeros((1, 1, self.frame_size), DType::F32, &self.device)?;
            let codes = self.mimi.encode_step(
                &::moshi::StreamTensor::from_tensor(wav),
                &::moshi::StreamMask::empty(),
            )?;
            if let Some(codes) = codes.as_option() {
                let _ = self.step_codes(codes, false)?;
            }
        }
        self.reset_stream();
        Ok(())
    }

    pub fn step_pcm_frame(&mut self, pcm: &[f32]) -> Result<Vec<RealtimeMoshiEvent>> {
        if pcm.len() != self.frame_size {
            return Err(Error::Msg(format!(
                "Moshi realtime frame must be exactly {} samples, got {}",
                self.frame_size,
                pcm.len()
            )));
        }
        let wav = Tensor::from_vec(pcm.to_vec(), (1, 1, pcm.len()), &self.device)?;
        let codes = self.mimi.encode_step(
            &::moshi::StreamTensor::from_tensor(wav),
            &::moshi::StreamMask::empty(),
        )?;
        let Some(codes) = codes.as_option() else {
            return Ok(Vec::new());
        };
        let reset_mimi_after_encode = self.skip_frames > 0;
        if reset_mimi_after_encode {
            // Python server.py encodes the first PCM frame, resets Mimi's streaming state,
            // then still feeds those codes into LMGen.step. The reset reapplies Mimi's
            // left-padding structure on the next encoder call without shifting LMGen.
            self.mimi.reset_state();
            self.skip_frames -= 1;
        }
        self.step_codes(codes, true)
    }

    fn step_codes(&mut self, codes: &Tensor, emit_events: bool) -> Result<Vec<RealtimeMoshiEvent>> {
        let codes = codes.to_dtype(DType::U32)?.to_vec3::<u32>()?;
        let mut events = Vec::new();
        for frame in 0..codes[0][0].len() {
            let input = codes[0]
                .iter()
                .map(|codebook| codebook[frame])
                .collect::<Vec<_>>();
            if emit_events {
                events.push(RealtimeMoshiEvent::InputAudioTokenFrame(input.clone()));
            }
            let text = self
                .state
                .step_without_ca_src(self.text_token, &input, None)?;
            self.text_token = text;
            if emit_events && text != self.text_pad_token && text != self.text_eop_token {
                events.push(RealtimeMoshiEvent::TextToken(text));
            }
            let Some(audio) = self.state.last_audio_tokens() else {
                continue;
            };
            let generated = audio[..self.generated_codebooks.min(audio.len())].to_vec();
            if emit_events {
                events.push(RealtimeMoshiEvent::AudioTokenFrame(generated.clone()));
            }
            let frame = Tensor::from_vec(generated.clone(), (1, generated.len(), 1), &self.device)?;
            let out = self.mimi.decode_step(
                &::moshi::StreamTensor::from_tensor(frame),
                &::moshi::StreamMask::empty(),
            )?;
            if let Some(out) = out.as_option() {
                if emit_events {
                    let pcm = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    events.push(RealtimeMoshiEvent::Audio {
                        pcm,
                        rate: self.sample_rate,
                    });
                }
            }
        }
        Ok(events)
    }
}

pub fn load_realtime_moshi(
    moshi_weights: &str,
    mimi_weights: &str,
    dtype: DType,
    device: &Device,
    params: RealtimeMoshiParams,
) -> Result<RealtimeMoshi> {
    let cfg = ::moshi::lm_generate_multistream::Config::v0_1();
    let mimi = ::moshi::mimi::load_b(
        None,
        mimi_weights,
        Some(cfg.generated_audio_codebooks),
        device,
    )?;
    let lm = ::moshi::lm::load_streaming_both_ways(moshi_weights, dtype, device)?;
    let mut realtime = RealtimeMoshi::new(mimi, lm, device.clone(), params);
    realtime.warmup()?;
    Ok(realtime)
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

pub fn safetensors_floating_dtype(path: &Path) -> Result<DType> {
    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
    let mut found: Option<(DType, String)> = None;
    for (name, view) in tensors.tensors() {
        let dtype: DType = view.dtype().try_into()?;
        if !is_floating_dtype(dtype) {
            continue;
        }
        match &found {
            Some((prev, first)) if *prev != dtype => {
                return Err(Error::Msg(format!(
                    "mixed floating safetensor dtypes: `{first}` is {prev:?}, `{name}` is {dtype:?}",
                )));
            }
            None => found = Some((dtype, name)),
            _ => {}
        }
    }
    found
        .map(|(dtype, _)| dtype)
        .ok_or_else(|| Error::Msg("checkpoint has no floating safetensor tensors".into()))
}

fn validate_candle_moshi_checkpoint(path: &Path) -> Result<()> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Ok(());
    }

    let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&[path])? };
    let mut has_candle_layout = false;
    let mut has_python_layout = false;
    for (name, _) in tensors.tensors() {
        if name.starts_with("depformer.") && name.ends_with(".linear_in.weight") {
            has_candle_layout = true;
        }
        if name.starts_with("depformer_in.")
            || name.starts_with("linears.")
            || name.starts_with("depformer_emb.")
        {
            has_python_layout = true;
        }
    }
    if has_candle_layout {
        return Ok(());
    }
    if has_python_layout {
        return Err(Error::Msg(
            "Moshi checkpoint uses the PyTorch weight layout; the native desktop runtime uses the Candle Moshi layout. Download `kyutai/moshiko-candle-bf16` or choose a Candle Moshi snapshot."
                .into(),
        ));
    }
    Err(Error::Msg(
        "Moshi checkpoint does not look like a Candle Moshi snapshot: missing `depformer.0.linear_in.weight`. Download `kyutai/moshiko-candle-bf16` or choose a Candle Moshi snapshot."
            .into(),
    ))
}

fn has_unimplemented_conditioning(config: &serde_json::Value) -> bool {
    const KEYS: &[&str] = &[
        "conditioners",
        "condition_provider",
        "condition_tensors",
        "fuser",
    ];
    KEYS.iter()
        .any(|key| config.get(key).is_some_and(|value| !value.is_null()))
}

fn has_unimplemented_cfg(config: Option<&serde_json::Value>) -> bool {
    let Some(config) = config else {
        return false;
    };
    if config
        .get("cfg_is_masked_until")
        .is_some_and(|value| !value.is_null())
        || config
            .get("cfg_is_no_text")
            .is_some_and(|value| value.as_bool().unwrap_or(true))
    {
        return true;
    }
    ["cfg_coef", "cfg_alpha"].iter().any(|key| {
        config.get(key).is_some_and(|value| {
            value
                .as_f64()
                .map(|n| (n - 1.0).abs() > f64::EPSILON)
                .unwrap_or(!value.is_null())
        })
    })
}

pub fn realtime_moshi_files(dir: &Path) -> Result<Option<RealtimeMoshiFiles>> {
    let config = dir.join("config.json");
    let (moshi_name, mimi_name, tokenizer_name, model_type, params) = if config.is_file() {
        let value: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config).map_err(|e| Error::Msg(e.to_string()))?,
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
        let lm = value
            .get("lm_config")
            .and_then(serde_json::Value::as_object);
        let name = |key: &str, default: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    lm.and_then(|lm| lm.get(key))
                        .and_then(serde_json::Value::as_str)
                })
                .unwrap_or(default)
                .to_string()
        };
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                lm.and_then(|lm| lm.get("model_type"))
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("moshi")
            .to_string();
        if model_type != "moshi" {
            return Err(Error::Msg(format!(
                "native realtime Moshi only supports plain `moshi` checkpoints; `{model_type}` needs the upstream Python conditioning/streaming path"
            )));
        }
        let lm_value = value.get("lm_config");
        let conditioned = has_unimplemented_conditioning(&value)
            || lm_value.is_some_and(has_unimplemented_conditioning);
        let cfg = has_unimplemented_cfg(Some(&value))
            || has_unimplemented_cfg(value.get("lm_gen_config"))
            || has_unimplemented_cfg(lm_value.and_then(|lm| lm.get("lm_gen_config")));
        if conditioned || cfg {
            return Err(Error::Msg(
                "native realtime Moshi does not yet implement Liquid's condition_tensors/CFG fuser path; use an unconditioned Moshiko Candle snapshot"
                    .into(),
            ));
        }
        (
            name("moshi_name", DEFAULT_MOSHI_NAME),
            name("mimi_name", DEFAULT_MIMI_NAME),
            name("tokenizer_name", DEFAULT_TEXT_TOKENIZER_NAME),
            model_type,
            RealtimeMoshiParams::from_lm_gen_config(value.get("lm_gen_config"))?,
        )
    } else {
        (
            DEFAULT_MOSHI_NAME.to_string(),
            DEFAULT_MIMI_NAME.to_string(),
            DEFAULT_TEXT_TOKENIZER_NAME.to_string(),
            "moshi".to_string(),
            RealtimeMoshiParams::default(),
        )
    };
    let files = RealtimeMoshiFiles {
        moshi_weights: dir.join(moshi_name),
        mimi_weights: dir.join(mimi_name),
        tokenizer: dir.join(tokenizer_name),
        model_type,
        params,
    };
    if files.moshi_weights.is_file() && files.mimi_weights.is_file() && files.tokenizer.is_file() {
        validate_candle_moshi_checkpoint(&files.moshi_weights)?;
        return Ok(Some(files));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: &Path) {
        std::fs::write(path, "").unwrap();
    }

    fn write_candle_moshi(path: &Path) {
        let mut tensors = std::collections::HashMap::new();
        let value = Tensor::zeros((1, 1), DType::BF16, &Device::Cpu).unwrap();
        tensors.insert("depformer.0.linear_in.weight", value);
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    fn write_python_moshi(path: &Path) {
        let mut tensors = std::collections::HashMap::new();
        let value = Tensor::zeros((1, 1), DType::BF16, &Device::Cpu).unwrap();
        tensors.insert("depformer_in.0.weight", value);
        candle_core::safetensors::save(&tensors, path).unwrap();
    }

    #[test]
    fn realtime_moshi_files_accepts_legacy_default_names() {
        let dir = temp_dir("emberharmony-moshi-default");
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let files = realtime_moshi_files(&dir).unwrap().unwrap();
        assert_eq!(files.moshi_weights, dir.join(DEFAULT_MOSHI_NAME));
        assert_eq!(files.mimi_weights, dir.join(DEFAULT_MIMI_NAME));
        assert_eq!(files.tokenizer, dir.join(DEFAULT_TEXT_TOKENIZER_NAME));
        assert_eq!(files.model_type, "moshi");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_uses_python_root_config_overrides() {
        let dir = temp_dir("emberharmony-moshi-config");
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "moshi_name": "moshi-custom.safetensors",
                "mimi_name": "mimi-custom.safetensors",
                "tokenizer_name": "custom.model",
                "model_type": "moshi",
                "lm_gen_config": {
                    "temp": 0.6,
                    "top_k": 40,
                    "temp_text": 0.5,
                    "top_k_text": 7,
                    "use_sampling": false
                }
            }"#,
        )
        .unwrap();
        write_candle_moshi(&dir.join("moshi-custom.safetensors"));
        touch(&dir.join("mimi-custom.safetensors"));
        touch(&dir.join("custom.model"));

        let files = realtime_moshi_files(&dir).unwrap().unwrap();
        assert_eq!(files.moshi_weights, dir.join("moshi-custom.safetensors"));
        assert_eq!(files.mimi_weights, dir.join("mimi-custom.safetensors"));
        assert_eq!(files.tokenizer, dir.join("custom.model"));
        assert_eq!(files.model_type, "moshi");
        assert_eq!(files.params.audio_temperature, 0.6);
        assert_eq!(files.params.audio_top_k, 40);
        assert_eq!(files.params.text_temperature, 0.5);
        assert_eq!(files.params.text_top_k, 7);
        assert!(!files.params.use_sampling);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_accepts_legacy_lm_config_overrides() {
        let dir = temp_dir("emberharmony-moshi-legacy-config");
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "lm_config": {
                    "moshi_name": "moshi-custom.safetensors",
                    "mimi_name": "mimi-custom.safetensors",
                    "tokenizer_name": "custom.model"
                }
            }"#,
        )
        .unwrap();
        write_candle_moshi(&dir.join("moshi-custom.safetensors"));
        touch(&dir.join("mimi-custom.safetensors"));
        touch(&dir.join("custom.model"));

        let files = realtime_moshi_files(&dir).unwrap().unwrap();
        assert_eq!(files.moshi_weights, dir.join("moshi-custom.safetensors"));
        assert_eq!(files.mimi_weights, dir.join("mimi-custom.safetensors"));
        assert_eq!(files.tokenizer, dir.join("custom.model"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_incomplete_snapshot() {
        let dir = temp_dir("emberharmony-moshi-incomplete");
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));

        assert!(realtime_moshi_files(&dir).unwrap().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_pytorch_weight_layout() {
        let dir = temp_dir("emberharmony-moshi-pytorch");
        write_python_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let err = realtime_moshi_files(&dir).unwrap_err().to_string();
        assert!(err.contains("PyTorch weight layout"), "{err}");
        assert!(err.contains("kyutai/moshiko-candle-bf16"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_conditioned_model_types() {
        let dir = temp_dir("emberharmony-moshi-conditioned");
        std::fs::write(dir.join("config.json"), r#"{ "model_type": "hibiki" }"#).unwrap();
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let err = realtime_moshi_files(&dir).unwrap_err().to_string();
        assert!(err.contains("plain `moshi`"), "{err}");
        assert!(err.contains("upstream Python conditioning"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_conditioner_fuser_config() {
        let dir = temp_dir("emberharmony-moshi-fuser");
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "moshi",
                "lm_config": {
                    "conditioners": { "description": {} },
                    "fuser": { "sum": ["description"] }
                }
            }"#,
        )
        .unwrap();
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let err = realtime_moshi_files(&dir).unwrap_err().to_string();
        assert!(err.contains("condition_tensors/CFG"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_condition_provider_config() {
        let dir = temp_dir("emberharmony-moshi-condition-provider");
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "moshi",
                "lm_config": {
                    "condition_provider": { "description": {} }
                }
            }"#,
        )
        .unwrap();
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let err = realtime_moshi_files(&dir).unwrap_err().to_string();
        assert!(err.contains("condition_tensors/CFG"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn realtime_moshi_files_rejects_cfg_generation_config() {
        let dir = temp_dir("emberharmony-moshi-cfg");
        std::fs::write(
            dir.join("config.json"),
            r#"{
                "model_type": "moshi",
                "lm_gen_config": {
                    "cfg_coef": 2.0,
                    "cfg_is_no_text": true
                }
            }"#,
        )
        .unwrap();
        write_candle_moshi(&dir.join(DEFAULT_MOSHI_NAME));
        touch(&dir.join(DEFAULT_MIMI_NAME));
        touch(&dir.join(DEFAULT_TEXT_TOKENIZER_NAME));

        let err = realtime_moshi_files(&dir).unwrap_err().to_string();
        assert!(err.contains("condition_tensors/CFG"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
