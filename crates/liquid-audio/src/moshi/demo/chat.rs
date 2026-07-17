//! Core of `liquid_audio/demo/chat.py`: generated Mimi frames are decoded through
//! `proc.mimi`, not through `processor.decode`.

use candle_core::{Device, Result, Tensor};

use crate::moshi::models::MimiModel;

/// Rust form of `torch.stack(audio_out[:-1], 1).unsqueeze(0)`.
pub fn frames_to_codes(frames: &[Vec<u32>], codebooks: usize, device: &Device) -> Result<Tensor> {
    let n = frames.len();
    let mut flat = Vec::with_capacity(codebooks * n);
    for c in 0..codebooks {
        for frame in frames {
            flat.push(frame[c]);
        }
    }
    Tensor::from_vec(flat, (1, codebooks, n), device)
}

/// Offline demo decode for a completed assistant reply.
///
/// Mirrors:
///
/// ```python
/// audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
/// waveform = processor.mimi.decode(audio_codes)
/// ```
pub fn decode_audio_reply(
    mimi: &MimiModel<'_>,
    audio_out: &[Vec<u32>],
    codebooks: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    if audio_out.len() <= 1 {
        return Ok(None);
    }
    let codes = frames_to_codes(&audio_out[..audio_out.len() - 1], codebooks, device)?;
    Ok(Some(mimi.decode(&codes)?))
}

// The streaming per-frame decode is now `MimiStreaming::decode_codes` (host
// codes → host PCM, no `Tensor` round-trip). The former `decode_audio_frame`
// Tensor adapter is deleted.
