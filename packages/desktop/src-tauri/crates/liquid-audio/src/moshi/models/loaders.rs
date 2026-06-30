//! Rust facade for `liquid_audio/moshi/models/loaders.py` on the LFM2-Audio path.

use candle_core::{Device, Result};

/// Python `moshi.models.loaders.get_mimi(...)`.
///
/// The published Rust `moshi` crate owns the codec implementation; this function
/// preserves the Liquid-Audio-facing loader shape and the active-codebook selection.
pub fn get_mimi(path: &str, codebooks: usize, device: &Device) -> Result<::moshi::mimi::Mimi> {
    ::moshi::mimi::load(path, Some(codebooks), device)
}
