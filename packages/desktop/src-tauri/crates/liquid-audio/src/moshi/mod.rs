//! Rust facade for the upstream `liquid_audio/moshi` interface used by the demo.
//!
//! The Python package imports Kyutai's Moshi stack and the LFM2 demo reaches it as:
//!
//! ```python
//! mimi = proc.mimi
//! with mimi.streaming(1):
//!     wav_chunk = mimi.decode(frame[None, :, None])
//! ```
//!
//! The actual codec/model kernels come from Kyutai's published Rust `moshi` crate.
//! This module owns the Liquid-Audio-facing interface so callers do not accidentally
//! route generated audio through the generic `processor.decode` detokenizer path.

pub mod demo;
pub mod models;

pub use ::moshi::{StreamMask, StreamTensor, StreamingModule};
