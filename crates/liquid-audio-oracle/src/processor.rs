//! Port of `liquid_audio/processor.py` — `LFM2AudioProcessor` + `ChatState`.
//!
//! `LFM2AudioProcessor` bundles the text tokenizer (HF AutoTokenizer →
//! `tokenizers` crate), the mel audio frontend, and
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

use crate::audio_out::AudioEncoder;
use crate::utils::{mel2emb_len, LFMModality};

mod mel {
    //! Rust rim over the native mel frontend (native/src/frontend/lfm_frontend.cpp
    //! + flashkern_frontend.S). The former in-crate featurizer — hann/slaney/DFT
    //! table construction, the candle STFT, normalization — is DELETED; its
    //! numerics live natively and are gated by the committed fixtures under
    //! native/tests/fixtures/mel/ (captured from the deleted implementation).
    //!
    //! This oracle-only rim owns the opaque native plan/workspace lifetimes and
    //! transports the result into a Candle tensor solely for offline parity and
    //! training consumers. The production path already binds the native
    //! frontend directly to the native Conformer destination; this module is
    //! unreachable without the opt-in `oracle` feature. It is reference
    //! transport, not a shipped inference seam.

    use candle_core::{CpuStorage, DType, Device, Result, Storage, Tensor};

    /// Subset of NeMo's preprocessor config needed offline (unchanged shape —
    /// consumers build it from [`super::PreprocessorConfig::mel_config`]).
    #[derive(Debug, Clone)]
    pub struct MelConfig {
        pub sample_rate: usize,        // 16000
        pub n_window_size: usize,      // win_length (e.g. 400)
        pub n_window_stride: usize,    // hop_length (e.g. 160)
        pub n_fft: usize,              // e.g. 512
        pub nfilt: usize,              // mel bins (feat_in of the encoder)
        pub preemph: f64,              // 0.97
        pub log_zero_guard_value: f64, // 2^-24
        pub mag_power: f64,            // 2.0 (the only regime the native rim admits)
        pub pad_to: usize,             // 16
        /// NeMo `exact_pad` — switches the STFT to `center=False` with an
        /// explicit `(n_fft - hop)//2` signal pad.
        pub exact_pad: bool,
    }

    #[repr(C)]
    struct NativeFrontend {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct NativeFrontendWorkspace {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct NativeFrontendConfig {
        size: u32,
        abi_version: u32,
        sample_rate: u32,
        n_window_size: u32,
        n_window_stride: u32,
        n_fft: u32,
        nfilt: u32,
        exact_pad: u32,
        pad_to: u32,
        reserved0: u32,
        preemph: f64,
        log_zero_guard_value: f64,
        mag_power: f64,
        reserved: [u64; 4],
    }

    const FRONTEND_ABI: u32 = 1;
    const FORWARD_VALID_ONLY: u32 = 1;

    unsafe extern "C" {
        fn lfm_frontend_create(
            config: *const NativeFrontendConfig,
            out: *mut *mut NativeFrontend,
        ) -> i32;
        fn lfm_frontend_destroy(frontend: *mut NativeFrontend) -> i32;
        fn lfm_frontend_workspace_create(out: *mut *mut NativeFrontendWorkspace) -> i32;
        fn lfm_frontend_workspace_destroy(workspace: *mut NativeFrontendWorkspace) -> i32;
        fn lfm_frontend_workspace_reserve(
            frontend: *const NativeFrontend,
            workspace: *mut NativeFrontendWorkspace,
            max_sample_count: u64,
            flags: u32,
        ) -> i32;
        fn lfm_frontend_seq_len(frontend: *const NativeFrontend, sample_count: u64) -> u64;
        fn lfm_frontend_out_frames(
            frontend: *const NativeFrontend,
            sample_count: u64,
            out_frames: *mut u64,
        ) -> i32;
        fn lfm_frontend_forward_workspace(
            frontend: *const NativeFrontend,
            workspace: *mut NativeFrontendWorkspace,
            pcm: *const f32,
            sample_count: u64,
            out_mel: *mut f32,
            out_capacity_values: u64,
            flags: u32,
        ) -> i32;
    }

    /// The mel featurizer, native-backed. Public shape preserved for the
    /// remaining consumers (loader, realtime): `new`, `forward`, `get_seq_len`,
    /// `nfilt`, `mel_config`.
    pub struct FilterbankFeatures {
        cfg: MelConfig,
        handle: *mut NativeFrontend,
        workspace: *mut NativeFrontendWorkspace,
        device: Device,
    }

    // The native plan is immutable and the session workspace serializes its
    // own reuse (per lfm_frontend.h); raw pointers prevent auto-derivation.
    unsafe impl Send for FilterbankFeatures {}
    unsafe impl Sync for FilterbankFeatures {}

    impl Drop for FilterbankFeatures {
        fn drop(&mut self) {
            unsafe {
                let _ = lfm_frontend_workspace_destroy(self.workspace);
                let _ = lfm_frontend_destroy(self.handle);
            }
        }
    }

    impl FilterbankFeatures {
        pub fn new(cfg: MelConfig, device: &Device) -> Result<Self> {
            let narrow = |name: &str, value: usize| {
                u32::try_from(value).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "native mel frontend: {name}={value} exceeds the u32 ABI"
                    ))
                })
            };
            let native = NativeFrontendConfig {
                size: std::mem::size_of::<NativeFrontendConfig>() as u32,
                abi_version: FRONTEND_ABI,
                sample_rate: narrow("sample_rate", cfg.sample_rate)?,
                n_window_size: narrow("n_window_size", cfg.n_window_size)?,
                n_window_stride: narrow("n_window_stride", cfg.n_window_stride)?,
                n_fft: narrow("n_fft", cfg.n_fft)?,
                nfilt: narrow("nfilt", cfg.nfilt)?,
                exact_pad: cfg.exact_pad as u32,
                pad_to: narrow("pad_to", cfg.pad_to)?,
                reserved0: 0,
                preemph: cfg.preemph,
                log_zero_guard_value: cfg.log_zero_guard_value,
                mag_power: cfg.mag_power,
                reserved: [0; 4],
            };
            let mut handle: *mut NativeFrontend = std::ptr::null_mut();
            let rc = unsafe { lfm_frontend_create(&native, &mut handle) };
            if rc != 0 || handle.is_null() {
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend rejected the config (status {rc}): {cfg:?}"
                )));
            }
            let mut workspace: *mut NativeFrontendWorkspace = std::ptr::null_mut();
            let rc = unsafe { lfm_frontend_workspace_create(&mut workspace) };
            if rc != 0 || workspace.is_null() {
                unsafe {
                    let _ = lfm_frontend_destroy(handle);
                }
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend workspace creation failed (status {rc})"
                )));
            }
            Ok(Self {
                cfg,
                handle,
                workspace,
                device: device.clone(),
            })
        }

        /// Number of mel bins (encoder `feat_in`).
        pub fn nfilt(&self) -> usize {
            self.cfg.nfilt
        }

        /// The mel config (hop/window/fft sizes) backing this featurizer.
        pub fn mel_config(&self) -> MelConfig {
            self.cfg.clone()
        }

        /// Valid mel frames for `seq_len` input samples — the native
        /// floor-divide contract (single source of truth).
        pub fn get_seq_len(&self, seq_len: usize) -> usize {
            unsafe { lfm_frontend_seq_len(self.handle, seq_len as u64) as usize }
        }

        fn run(&self, pcm: &[f32], frames: usize, flags: u32) -> Result<Vec<f32>> {
            if pcm.is_empty() {
                return Err(candle_core::Error::Msg(
                    "native mel frontend: empty input clip".into(),
                ));
            }
            if frames == 0 {
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend: clip of {} samples has no valid frames",
                    pcm.len()
                )));
            }
            let values = self.cfg.nfilt.checked_mul(frames).ok_or_else(|| {
                candle_core::Error::Msg("native mel frontend output size overflow".into())
            })?;
            let mut out = Vec::<std::mem::MaybeUninit<f32>>::new();
            out.try_reserve_exact(values).map_err(|err| {
                candle_core::Error::Msg(format!(
                    "native mel frontend could not reserve {values} output values: {err}"
                ))
            })?;
            // This compatibility rim admits arbitrary offline clip lengths, so
            // it explicitly raises the high-water mark before the strict hot
            // call. Native sessions reserve their fixed PCM lease capacity once
            // at readiness and never execute this branch per command.
            let rc = unsafe {
                lfm_frontend_workspace_reserve(self.handle, self.workspace, pcm.len() as u64, flags)
            };
            if rc != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend workspace reserve failed (status {rc}, {} samples)",
                    pcm.len()
                )));
            }
            let rc = unsafe {
                lfm_frontend_forward_workspace(
                    self.handle,
                    self.workspace,
                    pcm.as_ptr(),
                    pcm.len() as u64,
                    out.as_mut_ptr().cast::<f32>(),
                    values as u64,
                    flags,
                )
            };
            if rc != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend forward failed (status {rc}, {} samples)",
                    pcm.len()
                )));
            }
            // SAFETY: a successful native call writes every one of the `values`
            // destination cells. The MaybeUninit vector has length zero on all
            // error paths, so partially written output is never observed.
            unsafe {
                out.set_len(values);
                let ptr = out.as_mut_ptr().cast::<f32>();
                let len = out.len();
                let capacity = out.capacity();
                std::mem::forget(out);
                Ok(Vec::from_raw_parts(ptr, len, capacity))
            }
        }

        fn forward_padded_slice(&self, pcm: &[f32]) -> Result<Tensor> {
            let mut frames = 0u64;
            let rc = unsafe { lfm_frontend_out_frames(self.handle, pcm.len() as u64, &mut frames) };
            let frames = usize::try_from(frames).map_err(|_| {
                candle_core::Error::Msg("native mel frontend frame count exceeds usize".into())
            })?;
            if rc != 0 || frames == 0 {
                return Err(candle_core::Error::Msg(format!(
                    "native mel frontend: clip of {} samples has no output frames (status {rc})",
                    pcm.len()
                )));
            }
            let out = self.run(pcm, frames, 0)?;
            Tensor::from_vec(out, (1, self.cfg.nfilt, frames), &self.device)
        }

        /// Compatibility contract for fixture and Python-shape consumers.
        /// Returns `(1, nfilt, T_padded)` while borrowing contiguous CPU PCM
        /// storage through the synchronous native call instead of `to_vec1`.
        pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
            let pcm = samples
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .contiguous()?;
            let (storage, layout) = pcm.storage_and_layout();
            let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::Msg("native mel frontend requires contiguous PCM".into())
            })?;
            let Storage::Cpu(CpuStorage::F32(values)) = &*storage else {
                return Err(candle_core::Error::Msg(
                    "native mel frontend requires CPU f32 PCM".into(),
                ));
            };
            self.forward_padded_slice(&values[start..end])
        }

        /// Production tensor seam. Returns the tightly packed
        /// `(nfilt, valid_frames)` plane; no centered/pad_to tail is uploaded
        /// and no caller-side crop is required.
        pub fn forward_valid(&self, samples: &Tensor) -> Result<Tensor> {
            let pcm = samples
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .contiguous()?;
            let (storage, layout) = pcm.storage_and_layout();
            let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::Msg("native mel frontend requires contiguous PCM".into())
            })?;
            let Storage::Cpu(CpuStorage::F32(values)) = &*storage else {
                return Err(candle_core::Error::Msg(
                    "native mel frontend requires CPU f32 PCM".into(),
                ));
            };
            self.forward_slice(&values[start..end])
        }

        /// Production PCM-span seam. The input is borrowed read-only for the
        /// synchronous native call and the returned plane is exactly
        /// `(nfilt, valid_frames)`.
        pub fn forward_slice(&self, pcm: &[f32]) -> Result<Tensor> {
            if pcm.is_empty() {
                return Err(candle_core::Error::Msg(
                    "native mel frontend: empty input clip".into(),
                ));
            }
            let frames = self.get_seq_len(pcm.len());
            if frames == 0 {
                return Tensor::zeros((self.cfg.nfilt, 1), DType::F32, &self.device)?
                    .narrow(1, 0, 0);
            }
            let out = self.run(pcm, frames, FORWARD_VALID_ONLY)?;
            Tensor::from_vec(out, (self.cfg.nfilt, frames), &self.device)
        }
    }
}

pub use mel::{FilterbankFeatures, MelConfig};

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
    /// NeMo `exact_pad` (constructor arg, not always in the checkpoint JSON; the
    /// LFM2.5-Audio config omits it ⇒ False). True switches the STFT to `center=False`
    /// with an explicit `(n_fft - hop)//2` signal pad.
    #[serde(default)]
    pub exact_pad: bool,
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
            preemph: 0.97, // FilterbankFeatures default
            log_zero_guard_value: 2f64.powi(-24),
            mag_power: 2.0,
            pad_to: self.pad_to,
            exact_pad: self.exact_pad,
        }
    }
}

/// The turn-format strings `ChatState` writes — single source of truth, shared with
/// [`crate::chat_template`]'s load-time verification against the snapshot's own
/// `chat_template.jinja`. Faithful to the Python `ChatState` (`new_turn`/`end_turn`).
pub const SEQUENCE_START: &str = "<|startoftext|>";
pub const TURN_FOOTER: &str = "<|im_end|>\n";
pub fn turn_header(role: &str) -> String {
    format!("<|im_start|>{role}\n")
}

/// Generation-control token ids resolved BY NAME from the model's own tokenizer at
/// load time — the model defines them, so they pass through instead of living as
/// literals in the generation loops (`<|im_end|>` also cross-checks the config's
/// `lfm.eos_token_id`). Resolution is hard-error: a snapshot whose tokenizer lacks
/// these names is not an LFM2-Audio model, and guessing ids would generate garbage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    /// `<|im_end|>` — closes a turn; text sampling stops here.
    pub im_end: u32,
    /// `<|text_end|>` — the interleaved text channel is finished for this turn.
    pub text_end: u32,
    /// `<|audio_start|>` — sequential (TTS) generation flips to audio frames here.
    pub audio_start: u32,
}

impl SpecialTokenIds {
    pub fn resolve(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |name: &str| -> Result<u32> {
            let id = tokenizer.token_to_id(name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "tokenizer does not define {name} — not an LFM2-Audio tokenizer"
                ))
            })?;
            // The grammar only works if the fence string ENCODES back to this single
            // id — i.e. the tokenizer matches it as an added special token rather
            // than char-splitting it into ordinary text. Everything downstream
            // (turn boundaries, end-of-turn detection, the chat template) rides on
            // this round-trip, so a tokenizer that fails it must fail the load.
            let enc = tokenizer
                .encode(name, false)
                .map_err(|e| candle_core::Error::Msg(format!("tokenizer encode {name}: {e}")))?;
            if enc.get_ids() != [id] {
                return Err(candle_core::Error::Msg(format!(
                    "tokenizer does not round-trip {name}: encodes to {:?}, expected [{id}] \
                     — grammar fences would enter context as plain text",
                    enc.get_ids()
                )));
            }
            Ok(id)
        };
        // <|im_start|> is not generation-control (never sampled against) but it IS
        // the other half of every fence — same round-trip requirement.
        let _ = id("<|im_start|>")?;
        Ok(Self {
            im_end: id("<|im_end|>")?,
            text_end: id("<|text_end|>")?,
            audio_start: id("<|audio_start|>")?,
        })
    }
}

pub struct LFM2AudioProcessor {
    pub tokenizer: Tokenizer,
    pub audio: FilterbankFeatures,
    /// Training-data codec. Production codec work is native and is deliberately
    /// absent from this crate's public surface.
    pub mimi: Option<Box<dyn AudioEncoder>>,
    pub device: Device,
}

impl LFM2AudioProcessor {
    /// Build from a local model directory: `tokenizer.json` + the mel buffers
    /// (`window`/`fb`) under a VarBuilder rooted at the audio preprocessor.
    ///
    pub fn new(
        tokenizer: Tokenizer,
        audio: FilterbankFeatures,
        mimi: Option<Box<dyn AudioEncoder>>,
        device: Device,
    ) -> Self {
        Self {
            tokenizer,
            audio,
            mimi,
            device,
        }
    }

    pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
        Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))
    }

    /// Resolve the generation-control token ids from THIS model's tokenizer.
    /// See [`SpecialTokenIds`].
    pub fn special_token_ids(tokenizer: &Tokenizer) -> Result<SpecialTokenIds> {
        SpecialTokenIds::resolve(tokenizer)
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
        let text = proc.encode(SEQUENCE_START)?;
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
            audio_out: Tensor::zeros((codebooks, 1), candle_core::DType::I64, dev)?
                .narrow(1, 0, 0)?,
            modality_flag,
        })
    }

    /// Seed a `ChatState` from a previously persisted conversation (the five model-input
    /// fields) instead of `new`'s fresh `<|startoftext|>` start.
    ///
    /// `ChatState<'a>` borrows the processor, so it cannot itself be held across turns by an
    /// owner that also owns the processor (self-referential). The realtime engine instead
    /// holds the accumulated *tensors* (`Lfm2VoiceEngine::conv`) and rebuilds a transient
    /// `ChatState` each turn via this constructor — the Rust analog of Python keeping ONE
    /// persistent `ChatState` object across `append`/`new_turn` calls (README getting-started,
    /// the two-turn example). No `<|startoftext|>` is prepended: the persisted `text` already
    /// begins with it from the first turn. Fields map 1:1 to `new`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        proc: &'a LFM2AudioProcessor,
        codebooks: usize,
        text: Tensor,
        audio_in: Tensor,
        audio_in_lens: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
    ) -> Result<Self> {
        // The persisted audio_out is restored from a prior turn's `append`; guard the codebook
        // row count before it reaches the prefill `audio_out` scatter (which asserts on it).
        if audio_out.dim(0)? != codebooks {
            return Err(candle_core::Error::Msg(format!(
                "from_parts: audio_out must have {codebooks} codebook rows, got {}",
                audio_out.dim(0)?
            )));
        }
        Ok(Self {
            proc,
            codebooks,
            text,
            audio_in,
            audio_in_lens,
            audio_out,
            modality_flag,
        })
    }

    pub fn add_text(&mut self, text: &str) -> Result<()> {
        let new_text = self.proc.encode(text)?;
        let n = new_text.dim(1)?;
        let new_mod =
            Tensor::from_vec(vec![LFMModality::Text as i64; n], (1, n), &self.proc.device)?;
        self.text = Tensor::cat(&[&self.text, &new_text], 1)?;
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        Ok(())
    }

    /// `wave`: (1, L) at `sampling_rate`. Resampling to 16 kHz is the caller's
    /// responsibility (kept out of the core port); pass 16 kHz mono.
    pub fn add_audio_16k(&mut self, wave: &Tensor) -> Result<()> {
        let mel = self.proc.audio.forward_valid(wave)?;
        self.append_audio_in(mel)
    }

    /// Pointer-through 16 kHz seam used by realtime: the retained PCM slice is
    /// borrowed synchronously and native writes the valid mel plane directly.
    pub fn add_audio_16k_slice(&mut self, pcm: &[f32]) -> Result<()> {
        let mel = self.proc.audio.forward_slice(pcm)?;
        self.append_audio_in(mel)
    }

    fn append_audio_in(&mut self, new_audio_in: Tensor) -> Result<()> {
        let frames = new_audio_in.dim(1)?;
        let emb_len = mel2emb_len(frames as i64) as usize;
        let new_len = Tensor::from_vec(vec![frames as i64], (1,), &self.proc.device)?;
        // A sub-hop clip has zero valid frames. Preserve its length record but
        // do not ask Metal to allocate/cat a zero-byte modality or mel tensor.
        if frames > 0 {
            // Replace the empty placeholder on the first add; otherwise append.
            self.audio_in = if self.audio_in.dim(1)? == 0 {
                new_audio_in
            } else {
                Tensor::cat(&[&self.audio_in, &new_audio_in], 1)?
            };
            let new_mod = Tensor::from_vec(
                vec![LFMModality::AudioIn as i64; emb_len],
                (1, emb_len),
                &self.proc.device,
            )?;
            self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        }
        self.audio_in_lens = if self.audio_in_lens.dim(0)? == 0 {
            new_len
        } else {
            Tensor::cat(&[&self.audio_in_lens, &new_len], 0)?
        };
        Ok(())
    }

    /// Plain-slice counterpart to [`Self::add_audio`]. At 16 kHz this passes
    /// the caller's retained buffer straight to native; other rates create
    /// only the resampler's required output buffer before doing the same.
    pub fn add_audio_slice(&mut self, pcm: &[f32], sampling_rate: u32) -> Result<()> {
        if sampling_rate == 0 {
            return Err(candle_core::Error::Msg(
                "add_audio_slice: sampling_rate must be non-zero".into(),
            ));
        }
        if sampling_rate == 16_000 {
            return self.add_audio_16k_slice(pcm);
        }
        let pcm = crate::resample::resample_slice(pcm, sampling_rate, 16_000);
        self.add_audio_16k_slice(&pcm)
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
        if sampling_rate == 0 {
            return Err(candle_core::Error::Msg(
                "add_audio: sampling_rate must be non-zero".into(),
            ));
        }
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
        self.add_text(&turn_header(role))
    }

    pub fn end_turn(&mut self) -> Result<()> {
        self.add_text(TURN_FOOTER)
    }

    /// Render the context as a human-readable transcript: the text channel decoded
    /// with special tokens VISIBLE (the turn grammar is the point), and contiguous
    /// audio runs shown as `⟨audio-in ×N⟩` / `⟨audio-out ×N⟩` placeholders at their
    /// sequence positions. This is exactly the sequence the model attends over —
    /// the debug answer to "are the role fences actually in context?".
    pub fn transcript(&self) -> Result<String> {
        let flags: Vec<i64> = self
            .modality_flag
            .to_dtype(candle_core::DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let ids: Vec<i64> = self.text.flatten_all()?.to_vec1::<i64>()?;
        let mut out = String::new();
        let mut text_run: Vec<u32> = Vec::new();
        let mut audio_run: Option<(i64, usize)> = None; // (modality, len)
        let mut ti = 0usize;

        let flush_text = |out: &mut String, run: &mut Vec<u32>| -> Result<()> {
            if run.is_empty() {
                return Ok(());
            }
            let s = self
                .proc
                .text()
                .decode(run, false)
                .map_err(|e| candle_core::Error::Msg(format!("transcript decode: {e}")))?;
            out.push_str(&s);
            run.clear();
            Ok(())
        };
        let flush_audio = |out: &mut String, run: &mut Option<(i64, usize)>| {
            if let Some((m, n)) = run.take() {
                let name = if m == LFMModality::AudioIn as i64 {
                    "audio-in"
                } else {
                    "audio-out"
                };
                out.push_str(&format!("⟨{name} ×{n}⟩"));
            }
        };

        for &flag in &flags {
            if flag == LFMModality::Text as i64 {
                flush_audio(&mut out, &mut audio_run);
                text_run.push(ids[ti] as u32);
                ti += 1;
            } else {
                flush_text(&mut out, &mut text_run)?;
                audio_run = match audio_run {
                    Some((m, n)) if m == flag => Some((m, n + 1)),
                    Some(_) => {
                        flush_audio(&mut out, &mut audio_run);
                        Some((flag, 1))
                    }
                    None => Some((flag, 1)),
                };
            }
        }
        flush_text(&mut out, &mut text_run)?;
        flush_audio(&mut out, &mut audio_run);
        Ok(out)
    }

    /// Append generated text + audio-out tokens with their modality flags.
    ///
    /// Mirrors the Python `ChatState.append` invariants: `text` is one row,
    /// `audio_out` has `codebooks` rows, `modality_flag` is one row, and the flag
    /// count equals `text_len + audio_out_len` (the scatter depends on it).
    pub fn append(
        &mut self,
        text: &Tensor,
        audio_out: &Tensor,
        modality_flag: &Tensor,
    ) -> Result<()> {
        let mf = if modality_flag.rank() == 1 {
            modality_flag.unsqueeze(0)?
        } else {
            modality_flag.clone()
        };
        if text.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(format!(
                "append: text must be 1 row, got {}",
                text.dim(0)?
            )));
        }
        if audio_out.dim(0)? != self.codebooks {
            return Err(candle_core::Error::Msg(format!(
                "append: audio_out must have {} codebook rows, got {}",
                self.codebooks,
                audio_out.dim(0)?
            )));
        }
        if mf.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(
                "append: modality_flag must be 1 row".into(),
            ));
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
        let (text, audio_out, mf) = (
            text.to_dtype(i64t)?,
            audio_out.to_dtype(i64t)?,
            mf.to_dtype(i64t)?,
        );
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

    pub fn mimi(&self) -> Option<&dyn AudioEncoder> {
        self.mimi.as_deref()
    }

    /// `mimi.sample_rate` — the Mimi codec's expected input sample rate. Used by the
    /// data mapper (`_encode_audio_out`) to resample before encoding. Reads the Mimi
    /// codec (not the decode backend): on a full snapshot the LFM2 detokenizer's rate
    /// is irrelevant to the encode path.
    pub fn mimi_sample_rate(&self) -> Option<u32> {
        self.mimi.as_deref().map(|d| d.sample_rate())
    }

    /// `mimi.encode(wav)` — encode a `(B, 1, L)` waveform to codes via the Mimi codec
    /// (errors if no Mimi checkpoint was loaded). Always the Mimi codec, never the
    /// LFM2 detokenizer (which is decode-only).
    pub fn mimi_encode(&self, wav: &Tensor) -> Result<Tensor> {
        self.mimi
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "no Mimi codec loaded (encode needs the Mimi tokenizer checkpoint `tokenizer-…checkpoint125.safetensors`)".into(),
                )
            })?
            .encode(wav)
    }
}

impl ChatState<'_> {
    /// `model_inputs` — the model-input field names (Python `model_inputs`).
    pub fn model_inputs(&self) -> [&'static str; 5] {
        [
            "text",
            "audio_in",
            "audio_in_lens",
            "audio_out",
            "modality_flag",
        ]
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
                [
                    "text",
                    "audio_in",
                    "audio_in_lens",
                    "audio_out",
                    "modality_flag"
                ]
            ))),
        }
    }

    /// `device` → the processor's device.
    pub fn device(&self) -> &Device {
        &self.proc.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;

    fn test_processor() -> LFM2AudioProcessor {
        let dev = Device::Cpu;
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<|startoftext|>".to_string(), 1);
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );
        let audio = FilterbankFeatures::new(
            MelConfig {
                sample_rate: 16_000,
                n_window_size: 400,
                n_window_stride: 160,
                n_fft: 512,
                nfilt: 8,
                preemph: 0.97,
                log_zero_guard_value: 2f64.powi(-24),
                mag_power: 2.0,
                pad_to: 16,
                exact_pad: false,
            },
            &dev,
        )
        .unwrap();
        LFM2AudioProcessor::new(tokenizer, audio, None, dev)
    }

    #[test]
    fn add_audio_16k_uses_valid_mel_length_not_center_padding() {
        let proc = test_processor();
        let wave = Tensor::zeros((1, 1280), DType::F32, proc.device()).unwrap();
        let raw = proc.audio().forward(&wave).unwrap();
        let valid = proc.audio().get_seq_len(1280);

        // STFT with center=True: center_pad = n_fft/2 = 256, padded_len = 1280+512=1792,
        // T = (1792-512)/160+1 = 9. Then pad_to=16 pads time to the next multiple → 16.
        assert_eq!(raw.dim(2).unwrap(), 16); // 9 STFT frames, padded to 16 by pad_to
        assert_eq!(valid, 8);
        assert_eq!(mel2emb_len(raw.dim(2).unwrap() as i64), 2);
        assert_eq!(mel2emb_len(valid as i64), 1);

        // add_audio_16k asks native for the valid-only destination, so audio_in
        // never carries or crop-copies the padded count.
        let mut chat = ChatState::new(&proc, 8).unwrap();
        chat.add_audio_16k(&wave).unwrap();

        assert_eq!(chat.audio_in.dim(1).unwrap(), valid);
        assert_eq!(chat.audio_in_lens.to_vec1::<i64>().unwrap(), vec![8]);
        let flags = chat.modality_flag.to_vec2::<i64>().unwrap();
        let audio_flags = flags[0]
            .iter()
            .filter(|&&flag| flag == LFMModality::AudioIn as i64)
            .count();
        assert_eq!(audio_flags, mel2emb_len(valid as i64) as usize);
    }

    #[test]
    fn add_audio_slice_matches_tensor_compatibility_seam() {
        let proc = test_processor();
        let pcm: Vec<f32> = (0..1280)
            .map(|i| ((i as f32 * 0.03125).sin() * 0.5).clamp(-1.0, 1.0))
            .collect();
        let wave = Tensor::from_vec(pcm.clone(), (1, pcm.len()), proc.device()).unwrap();
        let mut tensor = ChatState::new(&proc, 8).unwrap();
        tensor.add_audio_16k(&wave).unwrap();
        let mut slice = ChatState::new(&proc, 8).unwrap();
        slice.add_audio_slice(&pcm, 16_000).unwrap();

        assert_eq!(tensor.audio_in.dims(), slice.audio_in.dims());
        assert_eq!(
            tensor
                .audio_in
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            slice
                .audio_in
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tensor.audio_in_lens.to_vec1::<i64>().unwrap(),
            slice.audio_in_lens.to_vec1::<i64>().unwrap()
        );
        assert_eq!(
            tensor.modality_flag.to_vec2::<i64>().unwrap(),
            slice.modality_flag.to_vec2::<i64>().unwrap()
        );
    }

    #[test]
    fn sub_hop_audio_records_zero_length_without_zero_tensor_construction() {
        let proc = test_processor();
        let mut chat = ChatState::new(&proc, 8).unwrap();
        chat.add_audio_slice(&[0.0; 80], 16_000).unwrap();
        assert_eq!(chat.audio_in.dim(1).unwrap(), 0);
        assert_eq!(chat.audio_in_lens.to_vec1::<i64>().unwrap(), vec![0]);
        assert_eq!(
            chat.modality_flag
                .to_vec2::<i64>()
                .unwrap()
                .iter()
                .flatten()
                .filter(|&&flag| flag == LFMModality::AudioIn as i64)
                .count(),
            0
        );
    }

    #[test]
    fn add_audio_rejects_zero_sampling_rate() {
        let proc = test_processor();
        let wave = Tensor::zeros((1, 320), DType::F32, proc.device()).unwrap();
        let mut chat = ChatState::new(&proc, 8).unwrap();
        let err = chat.add_audio(&wave, 0).unwrap_err().to_string();
        assert!(err.contains("sampling_rate must be non-zero"), "{err}");
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
