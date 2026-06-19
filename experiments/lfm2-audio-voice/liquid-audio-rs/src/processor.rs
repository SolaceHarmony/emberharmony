//! Port of `liquid_audio/processor.py` — `LFM2AudioProcessor` + `ChatState`.
//!
//! `LFM2AudioProcessor` bundles the text tokenizer (HF AutoTokenizer →
//! `tokenizers` crate), the mel audio preprocessor (`conformer::processor`), and
//! the audio-out backend behind the [`AudioDetokenizer`](crate::audio_out)
//! trait (`decode`). `ChatState` builds the model inputs (text tokens, audio-in
//! mel, lengths, audio-out codes, modality flags) the way the Python usage
//! example does (`new_turn`/`add_text`/`add_audio`/`end_turn`/`append`).
//!
//! The audio-out backend is selected at load time (LFM2 detokenizer for LFM2.5
//! models, Mimi codec for v1) but the processor only knows the trait — pure
//! candle, no torch.

use std::path::Path;

use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::audio_out::AudioDetokenizer;
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
    /// Audio-out backend (LFM2 detokenizer or Mimi), behind the trait we own so
    /// the processor never touches a concrete codec type.
    pub audio_out: Option<Box<dyn AudioDetokenizer>>,
    pub device: Device,
}

impl LFM2AudioProcessor {
    /// Build from a local model directory: `tokenizer.json` + the mel buffers
    /// (`window`/`fb`) under a VarBuilder rooted at the audio preprocessor.
    pub fn new(
        tokenizer: Tokenizer,
        audio: FilterbankFeatures,
        audio_out: Option<Box<dyn AudioDetokenizer>>,
        device: Device,
    ) -> Self {
        Self { tokenizer, audio, audio_out, device }
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

    /// Detokenize audio codes `(1, codebooks, T)` → 24 kHz waveform via whichever
    /// audio-out backend was selected at load (LFM2 detokenizer or Mimi). The
    /// processor dispatches through the [`AudioDetokenizer`](crate::audio_out)
    /// trait — it doesn't know which concrete backend it holds.
    pub fn decode(&self, audio_codes: &Tensor) -> Result<Tensor> {
        // Python guard: reject codes outside [0, 2047] before detokenizing (the
        // EOAudio sentinel 2048 must be stripped by the caller). u32 ⇒ ≥0 already;
        // check the upper bound rather than index OOB in the codebook embedding.
        let max_code = audio_codes
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .max(0)?
            .to_scalar::<u32>()?;
        if max_code > 2047 {
            return Err(candle_core::Error::Msg(format!(
                "audio code {max_code} out of range [0, 2047] (strip the EOAudio frame before decode)"
            )));
        }
        self.audio_out
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("no audio-out backend loaded".into()))?
            .decode(audio_codes)
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
    ///
    /// Mirrors the Python `ChatState.append` invariants: `text` is one row,
    /// `audio_out` has `codebooks` rows, `modality_flag` is one row, and the flag
    /// count equals `text_len + audio_out_len` (the scatter depends on it).
    pub fn append(&mut self, text: &Tensor, audio_out: &Tensor, modality_flag: &Tensor) -> Result<()> {
        let mf = if modality_flag.rank() == 1 { modality_flag.unsqueeze(0)? } else { modality_flag.clone() };
        if text.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(format!("append: text must be 1 row, got {}", text.dim(0)?)));
        }
        if audio_out.dim(0)? != self.codebooks {
            return Err(candle_core::Error::Msg(format!(
                "append: audio_out must have {} codebook rows, got {}",
                self.codebooks,
                audio_out.dim(0)?
            )));
        }
        if mf.dim(0)? != 1 {
            return Err(candle_core::Error::Msg("append: modality_flag must be 1 row".into()));
        }
        let (n_text, n_audio, n_flag) = (text.dim(1)?, audio_out.dim(1)?, mf.dim(1)?);
        if n_flag != n_text + n_audio {
            return Err(candle_core::Error::Msg(format!(
                "append: modality_flag len {n_flag} != text {n_text} + audio_out {n_audio}"
            )));
        }
        self.text = Tensor::cat(&[&self.text, text], 1)?;
        self.audio_out = Tensor::cat(&[&self.audio_out, audio_out], 1)?;
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &mf], 1)?;
        Ok(())
    }
}

use candle_core::IndexOp;
