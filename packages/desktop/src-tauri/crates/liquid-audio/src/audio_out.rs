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
// `Send` so the processor (and thus the model bundle) can move to a dedicated inference
// worker thread — the realtime full-duplex pipeline owns it there rather than sharing by
// `&` (the Mimi backend holds a `!Sync` `RefCell`). Both backends are `Send`.
pub trait AudioDetokenizer: Send {
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

    /// Reset streaming-decode state. Call once at the start of each generation
    /// turn, before the first [`decode_step`](Self::decode_step) — the analog of
    /// entering the Python `mimi.streaming(1)` context.
    fn reset_stream(&self) {}

    /// Streaming decode of a single audio frame `(1, codebooks, 1)` → an optional
    /// audio chunk. A streaming codec buffers a few frames before emitting, so the
    /// first call(s) may return `None`. Unlike [`decode`](Self::decode), this keeps
    /// codec state **across** calls (no per-call reset), so chunks stitch
    /// gaplessly — the real-time path the Python demo uses inside
    /// `mimi.streaming(1)` (`mimi.decode(frame)` per generated frame).
    ///
    /// Default: backends without a true streaming path fall back to a one-shot
    /// decode of the single frame.
    fn decode_step(&self, frame: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(self.decode(frame)?))
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
    inner: RefCell<::moshi::mimi::Mimi>,
}

impl MimiDetokenizer {
    pub fn new(mimi: ::moshi::mimi::Mimi) -> Self {
        Self {
            inner: RefCell::new(mimi),
        }
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

    /// Reset the moshi Mimi streaming conv/transformer state (turn boundary).
    fn reset_stream(&self) {
        self.inner.borrow_mut().reset_state();
    }

    /// Real streaming decode via moshi's `Mimi::decode_step` — keeps codec state
    /// across calls (no `reset_state` here; [`reset_stream`](Self::reset_stream)
    /// does that at the turn boundary). `frame` is `(1, codebooks, 1)`; the codec's
    /// warmup latency means the first call(s) yield `None`.
    fn decode_step(&self, frame: &Tensor) -> Result<Option<Tensor>> {
        let codes = frame.to_dtype(DType::U32)?; // RVQ index_select wants u32
        let mut m = self.inner.borrow_mut();
        let out = m.decode_step(
            &::moshi::StreamTensor::from_tensor(codes),
            &::moshi::StreamMask::empty(),
        )?;
        Ok(out.as_option().cloned())
    }
}
