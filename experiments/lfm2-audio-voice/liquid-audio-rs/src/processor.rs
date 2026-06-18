//! Port of `liquid_audio/processor.py` — `LFM2AudioProcessor` + `ChatState`.
//!
//! `LFM2AudioProcessor` bundles the text tokenizer (HF AutoTokenizer →
//! `tokenizers` crate), the mel audio preprocessor (`conformer::processor`), the
//! LFM2.5 audio detokenizer (`decode`), and the Kyutai Mimi codec (`mimi_decode`,
//! the v1 `processor.mimi` audio-out path). `ChatState` builds the model inputs
//! (text tokens, audio-in mel, lengths, audio-out codes, modality flags) the way
//! the Python usage example does (`new_turn`/`add_text`/`add_audio`/`end_turn`/
//! `append`).
//!
//! Mimi is reused from the `moshi` crate — Kyutai's own Mimi, the Rust port of
//! the exact code the Python lib vendors under `liquid_audio/moshi`, so it loads
//! the moshi-format checkpoint natively. Pure candle (moshi pins candle ^0.9.1 =
//! our 0.9.2), no torch.

use std::cell::RefCell;
use std::path::Path;

use candle_core::{Device, Result, Tensor};
use moshi::mimi;
use tokenizers::Tokenizer;

use crate::detokenizer::LFM2AudioDetokenizer;
use crate::model::conformer::processor::{FilterbankFeatures, MelConfig};
use crate::utils::{mel2emb_len, LFMModality};

/// Matches the `preprocessor` block of the model's config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub normalize: String,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub features: usize,
    pub n_fft: usize,
    pub log: bool,
    pub frame_splicing: usize,
    pub dither: f64,
    pub pad_to: usize,
    pub pad_value: f64,
}

impl PreprocessorConfig {
    /// Used by the model builder (lfm2_audio) to construct the featurizer.
    pub fn mel_config(&self) -> MelConfig {
        MelConfig {
            sample_rate: self.sample_rate,
            n_window_size: (self.window_size * self.sample_rate as f64).round() as usize,
            n_window_stride: (self.window_stride * self.sample_rate as f64).round() as usize,
            n_fft: self.n_fft,
            nfilt: self.features,
            preemph: 0.97,            // FilterbankFeatures default
            log_zero_guard_value: 2f64.powi(-24),
            mag_power: 2.0,
            pad_to: self.pad_to,
        }
    }
}

pub struct LFM2AudioProcessor {
    pub tokenizer: Tokenizer,
    pub audio: FilterbankFeatures,
    pub detokenizer: Option<LFM2AudioDetokenizer>,
    /// Kyutai Mimi codec (v1 `processor.mimi`). `RefCell` because Mimi decode is a
    /// streaming model with internal conv/transformer state (mirrors the Python
    /// `mimi.streaming(1)` mutation), kept behind `&self` for ergonomics.
    pub mimi: Option<RefCell<mimi::Mimi>>,
    pub device: Device,
}

impl LFM2AudioProcessor {
    /// Build from a local model directory: `tokenizer.json` + the mel buffers
    /// (`window`/`fb`) under a VarBuilder rooted at the audio preprocessor.
    pub fn new(
        tokenizer: Tokenizer,
        audio: FilterbankFeatures,
        detokenizer: Option<LFM2AudioDetokenizer>,
        mimi: Option<mimi::Mimi>,
        device: Device,
    ) -> Self {
        Self { tokenizer, audio, detokenizer, mimi: mimi.map(RefCell::new), device }
    }

    pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))
    }

    /// Encode text without auto special tokens → token id row `(1, n)`.
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let n = ids.len();
        Tensor::from_vec(ids, (1, n), &self.device)
    }

    /// Detokenize audio codes `(1, 8, T)` → 24 kHz waveform `(1, T')` via the
    /// LFM2-based detokenizer (LFM2.5 models, `processor.decode`).
    pub fn decode(&self, audio_codes: &Tensor) -> Result<Tensor> {
        let detok = self
            .detokenizer
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("no audio detokenizer loaded".into()))?;
        // detokenizer expects (B, L, codebooks)
        let codes = audio_codes.transpose(1, 2)?.contiguous()?;
        detok.forward(&codes)
    }

    /// v1 audio-out (`processor.mimi`): decode Mimi codes `(1, 8, T)` → 24 kHz
    /// waveform via the Kyutai Mimi codec (`moshi` crate). Codes are u32 indices.
    /// `reset_state` first so repeated calls are independent.
    pub fn mimi_decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mimi = self.mimi.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("model does not provide Mimi weights (processor.mimi)".into())
        })?;
        let codes = codes.to_dtype(candle_core::DType::U32)?;
        let mut m = mimi.borrow_mut();
        m.reset_state();
        m.decode(&codes)
    }
}

/// `ChatState` — accumulates model inputs across turns. Mirrors the Python
/// fields; `**chat` unpacking becomes direct field access in `generate_*`.
pub struct ChatState<'a> {
    proc: &'a LFM2AudioProcessor,
    codebooks: usize,
    pub text: Tensor,          // (1, n) u32 token ids
    pub audio_in: Tensor,      // (nfilt, total_frames) f32 mel
    pub audio_in_lens: Tensor, // (k,) u32
    pub audio_out: Tensor,     // (codebooks, m) u32
    pub modality_flag: Tensor, // (1, n) u32 (LFMModality)
}

impl<'a> ChatState<'a> {
    pub fn new(proc: &'a LFM2AudioProcessor, codebooks: usize) -> Result<Self> {
        let dev = &proc.device;
        let text = proc.encode("<|startoftext|>")?;
        let n = text.dim(1)?;
        let nfilt = proc.audio.nfilt();
        let modality_flag = Tensor::from_vec(vec![LFMModality::Text as u32; n], (1, n), dev)?;
        Ok(Self {
            proc,
            codebooks,
            text,
            audio_in: Tensor::zeros((nfilt, 0), candle_core::DType::F32, dev)?,
            audio_in_lens: Tensor::from_vec(Vec::<u32>::new(), (0,), dev)?,
            audio_out: Tensor::from_vec(Vec::<u32>::new(), (codebooks, 0), dev)?,
            modality_flag,
        })
    }

    pub fn add_text(&mut self, text: &str) -> Result<()> {
        let new_text = self.proc.encode(text)?;
        let n = new_text.dim(1)?;
        let new_mod = Tensor::from_vec(vec![LFMModality::Text as u32; n], (1, n), &self.proc.device)?;
        self.text = Tensor::cat(&[&self.text, &new_text], 1)?;
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        Ok(())
    }

    /// `wave`: (1, L) at `sampling_rate`. Resampling to 16 kHz is the caller's
    /// responsibility (kept out of the core port); pass 16 kHz mono.
    pub fn add_audio_16k(&mut self, wave: &Tensor) -> Result<()> {
        let (mel, _) = (self.proc.audio.forward(wave)?, ());
        let new_audio_in = mel.i(0)?; // (nfilt, frames)
        let frames = new_audio_in.dim(1)?;
        let emb_len = mel2emb_len(frames as i64) as usize;
        let new_mod = Tensor::from_vec(vec![LFMModality::AudioIn as u32; emb_len], (1, emb_len), &self.proc.device)?;
        let new_len = Tensor::from_vec(vec![frames as u32], (1,), &self.proc.device)?;
        self.audio_in = Tensor::cat(&[&self.audio_in, &new_audio_in], 1)?;
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        self.audio_in_lens = Tensor::cat(&[&self.audio_in_lens, &new_len], 0)?;
        Ok(())
    }

    pub fn new_turn(&mut self, role: &str) -> Result<()> {
        self.add_text(&format!("<|im_start|>{role}\n"))
    }

    pub fn end_turn(&mut self) -> Result<()> {
        self.add_text("<|im_end|>\n")
    }

    /// Append generated text + audio-out tokens with their modality flags.
    pub fn append(&mut self, text: &Tensor, audio_out: &Tensor, modality_flag: &Tensor) -> Result<()> {
        self.text = Tensor::cat(&[&self.text, text], 1)?;
        self.audio_out = Tensor::cat(&[&self.audio_out, audio_out], 1)?;
        let mf = if modality_flag.rank() == 1 { modality_flag.unsqueeze(0)? } else { modality_flag.clone() };
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &mf], 1)?;
        let _ = self.codebooks;
        Ok(())
    }
}

use candle_core::IndexOp;
