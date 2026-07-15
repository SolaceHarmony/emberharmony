//! Core of `liquid_audio/demo/chat.py`: generated Mimi frames are decoded through
//! `proc.mimi`, not through `processor.decode`.

use candle_core::{Device, Result, Tensor};

use crate::moshi::models::{MimiModel, MimiStreaming};

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

/// Streaming demo decode for one generated frame.
///
/// Mirrors `chat.py`: collect the frame as model output, skip EOAudio for playback,
/// then call `mimi.decode(t[None, :, None])` inside `mimi.streaming(1)`.
pub fn decode_audio_frame(
    stream: &mut MimiStreaming<'_>,
    frame: &[u32],
    codebooks: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    if frame.contains(&2048) {
        return Ok(None);
    }
    let codes = Tensor::from_vec(frame.to_vec(), (1, codebooks, 1), device)?;
    stream.decode(&codes)
}
