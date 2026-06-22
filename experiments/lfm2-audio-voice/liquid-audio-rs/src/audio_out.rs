//! Audio-out abstraction — a trait we own over the interchangeable detokenizer
//! backends, so the rest of the crate depends on *our* interface rather than any
//! foreign concrete type:
//!
//! - [`LFM2AudioDetokenizer`] — the LFM2-based detokenizer (LFM2.5 models),
//!   ported in-tree (`detokenizer.rs`), pure candle.
//! - [`MimiDetokenizer`] — the Kyutai Mimi codec (v1 models), reused from the
//!   `moshi` crate (the Rust port of the vendored `liquid_audio/moshi`).
//!
//! Rust has no class inheritance, so "override the backend" is expressed the
//! idiomatic way: a trait with a required method (each backend is *forced* to
//! implement `decode`), foreign types wrapped behind it (newtype + composition),
//! and runtime selection via `Box<dyn AudioDetokenizer>`.

use std::cell::RefCell;

use candle_core::{DType, Result, Tensor};

use crate::detokenizer::LFM2AudioDetokenizer;

/// Decode audio codes `(1, codebooks, T)` → 24 kHz mono waveform.
///
/// Required method (`decode`) ⇒ every backend must provide it (the compile-time
/// "force the override"); `sample_rate` is a default a backend may override.
pub trait AudioDetokenizer {
    fn decode(&self, codes: &Tensor) -> Result<Tensor>;
    fn sample_rate(&self) -> u32 {
        24_000
    }

    /// Encode a `(B, 1, L)` waveform at [`sample_rate`](Self::sample_rate) →
    /// integer codes `(B, codebooks, T)`. Mirrors `processor.mimi.encode` used by
    /// the data mapper (`LFM2AudioChatMapper._encode_audio_out`).
    ///
    /// Only the codec backend (Mimi) is an *encoder*; the LFM2 detokenizer is a
    /// vocoder (decode-only), so the default rejects the call rather than
    /// pretending — faithful to the Python where only `MimiModel` exposes
    /// `encode`.
    fn encode(&self, _wav: &Tensor) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "this audio-out backend is decode-only (no encoder); audio-out codes require the Mimi codec".into(),
        ))
    }
}

/// The in-tree LFM2-based detokenizer behind the shared trait. Its `forward`
/// takes `(B, T, codebooks)`, so we transpose from the canonical
/// `(1, codebooks, T)` code layout.
impl AudioDetokenizer for LFM2AudioDetokenizer {
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = codes.transpose(1, 2)?.contiguous()?;
        self.forward(&codes)
    }
}

/// Wraps the `moshi` crate's Mimi codec so it satisfies our trait, with interior
/// mutability for Mimi's streaming conv/transformer state (mirrors the Python
/// `mimi.streaming(1)` mutation). `reset_state` first ⇒ independent decodes.
pub struct MimiDetokenizer {
    inner: RefCell<moshi::mimi::Mimi>,
}

impl MimiDetokenizer {
    pub fn new(mimi: moshi::mimi::Mimi) -> Self {
        Self { inner: RefCell::new(mimi) }
    }
}

impl AudioDetokenizer for MimiDetokenizer {
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = codes.to_dtype(DType::U32)?; // RVQ index_select wants u32
        let mut m = self.inner.borrow_mut();
        m.reset_state();
        m.decode(&codes)
    }

    /// `mimi.encode(wav)` — `(B, 1, L)` 24 kHz waveform → codes `(B, codebooks, T)`.
    /// `reset_state` first ⇒ independent (non-streaming) encode, matching the
    /// Python `mimi.encode` call on a fresh clip in the data mapper.
    fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let mut m = self.inner.borrow_mut();
        m.reset_state();
        m.encode(wav)
    }
}
