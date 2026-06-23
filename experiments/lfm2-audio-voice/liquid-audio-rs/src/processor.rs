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

    /// PORT: `LFM2AudioProcessor.from_pretrained(repo_id, *, device)` (py 56).
    ///
    /// The Python classmethod resolves the model dir (`get_model_dir`), reads
    /// `config.json`, and constructs the processor (tokenizer + mel featurizer +
    /// audio-out backend) on `device`. The crate's loader already performs that
    /// exact construction inside [`crate::loader::from_pretrained`] (it builds both
    /// the model and the processor in one pass over the checkpoint). To avoid
    /// duplicating the loader logic this delegates to it and returns just the
    /// processor — the model is dropped here (the Python classmethod likewise only
    /// returns the processor). `dtype` mirrors the Python `to(device, dtype)` move
    /// folded into load.
    pub fn from_pretrained(dir: &Path, dtype: candle_core::DType, device: &Device) -> Result<Self> {
        let (_model, processor) = crate::loader::from_pretrained(dir, dtype, device)?;
        Ok(processor)
    }

    pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))
    }

    /// Encode text without auto special tokens → token id row `(1, n)`.
    ///
    /// I64 (torch.long): Python `text.encode(..., return_tensors="pt")` yields a long
    /// tensor, and every downstream id field (`audio_out` = `text.new_empty`,
    /// `modality_flag` = `full_like(text)`) inherits it. candle's index_select/embedding
    /// accept I64, so there is no reason to narrow to U32.
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
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
    pub text: Tensor,          // (1, n) i64 token ids (torch.long)
    pub audio_in: Tensor,      // (nfilt, total_frames) f32 mel
    pub audio_in_lens: Tensor, // (k,) i64 (torch.long)
    pub audio_out: Tensor,     // (codebooks, m) i64 (torch.long)
    pub modality_flag: Tensor, // (1, n) i64 (LFMModality; torch.long)
}

impl<'a> ChatState<'a> {
    pub fn new(proc: &'a LFM2AudioProcessor, codebooks: usize) -> Result<Self> {
        let dev = &proc.device;
        let text = proc.encode("<|startoftext|>")?;
        let n = text.dim(1)?;
        let nfilt = proc.audio.nfilt();
        let modality_flag = Tensor::from_vec(vec![LFMModality::Text as i64; n], (1, n), dev)?;
        Ok(Self {
            proc,
            codebooks,
            text,
            // Empty placeholders as zero-length VIEWS of a 1-element buffer. candle
            // can't allocate a zero-size buffer on Metal, so a bare `zeros((nfilt,0))`
            // fails on GPU; a valid 1-col buffer narrowed to length 0 reports 0
            // elements, is read only via `dim()` while empty, and is replaced (not
            // cat'd) on the first add — so no zero-size buffer is ever created.
            audio_in: Tensor::zeros((nfilt, 1), candle_core::DType::F32, dev)?.narrow(1, 0, 0)?,
            audio_in_lens: Tensor::zeros((1,), candle_core::DType::I64, dev)?.narrow(0, 0, 0)?,
            audio_out: Tensor::zeros((codebooks, 1), candle_core::DType::I64, dev)?.narrow(1, 0, 0)?,
            modality_flag,
        })
    }

    pub fn add_text(&mut self, text: &str) -> Result<()> {
        let new_text = self.proc.encode(text)?;
        let n = new_text.dim(1)?;
        let new_mod = Tensor::from_vec(vec![LFMModality::Text as i64; n], (1, n), &self.proc.device)?;
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
        let new_mod = Tensor::from_vec(vec![LFMModality::AudioIn as i64; emb_len], (1, emb_len), &self.proc.device)?;
        let new_len = Tensor::from_vec(vec![frames as i64], (1,), &self.proc.device)?;
        // Replace the empty placeholder on the first add (avoids cat-ing a
        // zero-length Metal view); otherwise append.
        self.audio_in = if self.audio_in.dim(1)? == 0 {
            new_audio_in
        } else {
            Tensor::cat(&[&self.audio_in, &new_audio_in], 1)?
        };
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        self.audio_in_lens = if self.audio_in_lens.dim(0)? == 0 {
            new_len
        } else {
            Tensor::cat(&[&self.audio_in_lens, &new_len], 0)?
        };
        Ok(())
    }

    /// PORT: `ChatState.add_audio(wave, sampling_rate)` (py 226).
    ///
    /// Faithful port of the full Python method: assert `wave` is `(1, L)`,
    /// resample from `sampling_rate` to 16 kHz (Python:
    /// `torchaudio.functional.resample(wave, sampling_rate, 16_000)`), run the mel
    /// front-end, then append the new audio-in mel, its `AUDIO_IN` modality flags
    /// (one per `mel2emb_len(frames)`), and the frame length — exactly the same
    /// three `torch.cat`s as Python (py 248-250).
    ///
    /// The resample is the faithful windowed-sinc [`crate::resample`] (a 1:1 port
    /// of `torchaudio.functional.resample`, shared with `data::mapper`),
    /// `L' = ceil(L * 16000 / sampling_rate)`. The post-resample mel/append path
    /// delegates to [`Self::add_audio_16k`] so the parity computation is shared
    /// and unchanged.
    pub fn add_audio(&mut self, wave: &Tensor, sampling_rate: u32) -> Result<()> {
        // Python: `assert len(wave.shape) == 2` and `assert wave.shape[0] == 1`.
        if wave.rank() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "add_audio: wave must be 2-D (1, L), got rank {}",
                wave.rank()
            )));
        }
        if wave.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(format!(
                "add_audio: wave must have 1 channel, got {}",
                wave.dim(0)?
            )));
        }

        // Python: `wave = torchaudio.functional.resample(wave, sampling_rate, 16_000)`.
        let wave16 = Self::resample_16k(wave, sampling_rate)?;
        self.add_audio_16k(&wave16)
    }

    /// `torchaudio.functional.resample(wave, orig, 16_000)` — the faithful
    /// windowed-sinc resampler (default `sinc_interp_hann`, width 6, rolloff 0.99),
    /// shared with `data::mapper`. `wave` is `(1, L)` → `(1, L')` f32 with
    /// `L' = ceil(L * 16000 / orig)`. See [`crate::resample`] (1:1 torchaudio port).
    fn resample_16k(wave: &Tensor, orig: u32) -> Result<Tensor> {
        crate::resample::resample(wave, orig, 16_000)
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
        // The state carries I64 (torch.long); cast the incoming ids to match (the
        // generation loop hands back U32 sampled tokens). Faithful — torch keeps long.
        let i64t = candle_core::DType::I64;
        let (text, audio_out, mf) = (text.to_dtype(i64t)?, audio_out.to_dtype(i64t)?, mf.to_dtype(i64t)?);
        self.text = Tensor::cat(&[&self.text, &text], 1)?;
        // Replace the empty placeholder on the first append (Metal: no zero-len cat).
        self.audio_out = if self.audio_out.dim(1)? == 0 {
            audio_out
        } else {
            Tensor::cat(&[&self.audio_out, &audio_out], 1)?
        };
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &mf], 1)?;
        Ok(())
    }
}

use candle_core::IndexOp;

impl LFM2AudioProcessor {
    /// `text` → the text tokenizer.
    pub fn text(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// `audio` → the mel audio preprocessor (Python `AudioToMelSpectrogramPreprocessor`;
    /// the port's featurizer).
    pub fn audio(&self) -> &FilterbankFeatures {
        &self.audio
    }

    /// `device` → the device tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `audio_detokenizer` / `mimi` → the audio-out backend. Python exposes the
    /// concrete `LFM2AudioDetokenizer` / `MimiModel`; the port dispatches `decode`
    /// through `Box<dyn AudioDetokenizer>`, so both accessors return that backend.
    pub fn audio_detokenizer(&self) -> Option<&dyn AudioDetokenizer> {
        self.audio_out.as_deref()
    }

    /// See [`Self::audio_detokenizer`].
    pub fn mimi(&self) -> Option<&dyn AudioDetokenizer> {
        self.audio_out.as_deref()
    }

    /// `mimi.sample_rate` — the audio-out codec's expected input sample rate.
    /// Used by the data mapper (`_encode_audio_out`) to decide whether to
    /// resample before encoding.
    pub fn mimi_sample_rate(&self) -> Option<u32> {
        self.audio_out.as_deref().map(|d| d.sample_rate())
    }

    /// `mimi.encode(wav)` — encode a `(B, 1, L)` waveform to codes via the
    /// audio-out backend (errors if the backend is decode-only).
    pub fn mimi_encode(&self, wav: &Tensor) -> Result<Tensor> {
        self.audio_out
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("no audio-out backend loaded".into()))?
            .encode(wav)
    }

    /// PORT: `to(device, dtype)` — torch in-place device/dtype move. candle places
    /// tensors at load (`from_pretrained(device, dtype)`); there is no in-place
    /// move. No-op, preserved for 1:1 inventory.
    pub fn to(&self) {}

    /// PORT: `eval` / `train` — torch training-mode toggle. Inference is always
    /// eval (dropout/BatchNorm are eval here); no-op, preserved for 1:1 inventory.
    pub fn eval(&self) {}

    /// See [`Self::eval`].
    pub fn train(&self) {}
}

impl ChatState<'_> {
    /// `model_inputs` — the model-input field names (Python `model_inputs`).
    pub fn model_inputs(&self) -> [&'static str; 5] {
        ["text", "audio_in", "audio_in_lens", "audio_out", "modality_flag"]
    }

    /// `__len__` → number of model-input fields.
    pub fn len(&self) -> usize {
        self.model_inputs().len()
    }

    /// The model-input field set is fixed and non-empty (kept for the
    /// `len`-without-`is_empty` lint).
    pub fn is_empty(&self) -> bool {
        self.model_inputs().is_empty()
    }

    /// `__iter__` → iterate the model-input field names.
    pub fn iter(&self) -> impl Iterator<Item = &'static str> {
        self.model_inputs().into_iter()
    }

    /// `__getitem__(name)` → the tensor field by model-input name.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        match name {
            "text" => Ok(&self.text),
            "audio_in" => Ok(&self.audio_in),
            "audio_in_lens" => Ok(&self.audio_in_lens),
            "audio_out" => Ok(&self.audio_out),
            "modality_flag" => Ok(&self.modality_flag),
            other => Err(candle_core::Error::Msg(format!(
                "expected one of {:?}, got {other}.",
                ["text", "audio_in", "audio_in_lens", "audio_out", "modality_flag"]
            ))),
        }
    }

    /// `device` → the processor's device.
    pub fn device(&self) -> &Device {
        &self.proc.device
    }
}

impl std::fmt::Display for ChatState<'_> {
    /// `__repr__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChatState(text_tok: {}, audio_in: {}, audio_out: {})",
            self.text.dim(1).unwrap_or(0),
            self.audio_in.dim(1).unwrap_or(0),
            self.audio_out.dim(1).unwrap_or(0),
        )
    }
}
