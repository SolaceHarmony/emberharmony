//! Faithful Rust-facing wrapper for Python `moshi.models.compression.MimiModel`.
//!
//! This is intentionally an interface adapter. The implementation is Kyutai's Rust
//! `moshi::mimi::Mimi`, already wrapped behind [`AudioDetokenizer`].

use candle_core::{DType, Result, Tensor};

use crate::audio_out::AudioDetokenizer;

/// The Python demo's `mimi` object: encode/decode plus a streaming context.
pub struct MimiModel<'a> {
    inner: &'a dyn AudioDetokenizer,
}

impl<'a> MimiModel<'a> {
    pub fn new(inner: &'a dyn AudioDetokenizer) -> Self {
        Self { inner }
    }

    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Python `mimi.encode(wav)`.
    pub fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        self.inner.encode(wav)
    }

    /// Python `mimi.decode(codes)` outside a streaming context.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        self.inner.decode(&codes.to_dtype(DType::U32)?)
    }

    /// Python `with mimi.streaming(batch_size): ...`.
    ///
    /// The LFM2 demo only uses batch size 1. Keeping this explicit catches accidental
    /// broadening before the rest of the pipeline is made batch-aware.
    pub fn streaming(&self, batch_size: usize) -> Result<MimiStreaming<'a>> {
        if batch_size != 1 {
            return Err(candle_core::Error::Msg(format!(
                "MimiModel::streaming only supports batch_size=1, got {batch_size}"
            )));
        }
        self.inner.reset_stream();
        Ok(MimiStreaming { inner: self.inner })
    }
}

/// Active `mimi.streaming(1)` context.
pub struct MimiStreaming<'a> {
    inner: &'a dyn AudioDetokenizer,
}

impl MimiStreaming<'_> {
    /// Python `mimi.decode(frame)` inside `with mimi.streaming(1)`.
    pub fn decode(&mut self, frame: &Tensor) -> Result<Option<Tensor>> {
        self.inner.decode_step(&frame.to_dtype(DType::U32)?)
    }

    /// Host-codes streaming decode: `frame` is the raw `[codebooks]` code ids,
    /// PCM comes back as `Vec<f32>` — no `Tensor` round-trip in either
    /// direction. The end-of-audio sentinel (`2048`) yields `None`, matching the
    /// former `decode_audio_frame`.
    pub fn decode_codes(&mut self, frame: &[u32]) -> Result<Option<Vec<f32>>> {
        if frame.contains(&2048) {
            return Ok(None);
        }
        self.inner.decode_step_codes(frame)
    }
}

impl Drop for MimiStreaming<'_> {
    fn drop(&mut self) {
        self.inner.reset_stream();
    }
}
