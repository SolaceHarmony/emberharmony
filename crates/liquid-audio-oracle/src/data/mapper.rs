//! Port of `liquid_audio/data/mapper.py` — `LFM2AudioChatMapper`.
//!
//! Maps a list of [`ChatMessage`]s into a packed [`LFM2AudioTrainingSample`] (the
//! per-sample tensor bundle: interleaved text / audio-in mel / audio-out codes,
//! the per-position modality flags, and the supervision mask). This is the
//! single-sample front of the data pipeline — the structural twin of
//! [`crate::processor::ChatState`] (which builds *inference* inputs turn by turn);
//! the mapper builds the *supervised training* sample from a whole conversation
//! at once.
//!
//! Faithful to the Python class:
//! - `__call__(messages)` — walk the chat, emitting the `<|startoftext|>`,
//!   `<|im_start|>{role}\n`, segment, and `<|im_end|>\n` pieces, then concat.
//! - `_append_text` / `_append_interleaved_out` / `_append_audio_in` /
//!   `_append_audio_out` — push tokens/mel/codes and extend the modality +
//!   supervision sequences.
//! - `_encode_audio_out` — resample to the codec rate, encode to codebook tokens,
//!   append the EOAudio (`2048`) frame.
//! - `_load_audio_bytes` — decode encoded-audio bytes to a mono f32 waveform.
//!
//! No torch: tensor ops go through candle, the mel front-end through
//! [`crate::processor::LFM2AudioProcessor::audio`], and the audio-out encode
//! through the [`crate::audio_out::AudioDetokenizer::encode`] backend
//! (`processor.mimi`). The two Python deps that have no candle referent —
//! `soundfile` (WAV decode) and `torchaudio.functional.resample` — are
//! implemented in-tree below as real, dependency-free routines rather than
//! stubbed out.

use candle_core::{DType, IndexOp, Result, Tensor};

use crate::data::types::{
    ChatContentSegment, ChatMessage, InterleavedSegment, LFM2AudioTrainingSample, Role,
};
use crate::processor::LFM2AudioProcessor;
use crate::utils::{mel2emb_len, LFMModality};

/// The codec rate the LFM2 codebooks are computed at. The `processor.mimi`
/// backend reports its own [`sample_rate`](crate::audio_out::AudioDetokenizer::sample_rate);
/// this is the fallback used when no backend is loaded (matches the LFM2 codec).
const DEFAULT_MIMI_SAMPLE_RATE: u32 = 24_000;

/// The mel front-end's input rate. The Python `_append_audio_in` resamples to
/// 16 kHz before the featurizer.
const AUDIO_IN_SAMPLE_RATE: u32 = 16_000;

/// End-of-audio sentinel appended after every encoded audio-out clip
/// (`torch.full((codebooks, 1), 2048)`).
const END_OF_AUDIO: u32 = 2048;

/// `LFM2AudioChatMapper` — "Map a chat into an LFM2 training sample."
///
/// Holds a borrow of the [`LFM2AudioProcessor`] (text tokenizer + mel front-end +
/// audio-out codec) plus the interleave granularity knobs. Faithful to:
/// ```python
/// class LFM2AudioChatMapper:
///     def __init__(self, processor, *, codebooks=8,
///                  interleaved_text_tokens=6, interleaved_audio_tokens=12): ...
/// ```
pub struct LFM2AudioChatMapper<'a> {
    /// `self.processor`.
    processor: &'a LFM2AudioProcessor,
    /// `self.codebooks` — number of audio codebook rows kept from the codec.
    codebooks: usize,
    /// `self.interleaved_text_tokens` — text tokens per interleave chunk.
    interleaved_text_tokens: usize,
    /// `self.interleaved_audio_tokens` — audio frames per interleave chunk.
    interleaved_audio_tokens: usize,
}

/// Running accumulators shared by the `_append_*` helpers. In Python these are
/// six `list[...]` locals passed by reference through every helper; bundling them
/// keeps the helper signatures faithful while satisfying the borrow checker.
struct Acc {
    /// `text_parts` — token-id rows, each `(n,)` u32.
    text_parts: Vec<Tensor>,
    /// `mel_parts` — mel feature blocks, each `(nfilt, frames)` f32.
    mel_parts: Vec<Tensor>,
    /// `audio_out_parts` — codebook-token blocks, each `(codebooks, m)` u32.
    audio_out_parts: Vec<Tensor>,
    /// `audio_in_lens` — per-segment valid mel-frame counts (torch.long → I64).
    audio_in_lens: Vec<i64>,
    /// `modality_seq` — per-position [`LFMModality`] flags (torch.long → I64).
    modality_seq: Vec<i64>,
    /// `supervision_seq` — per-position loss mask.
    supervision_seq: Vec<bool>,
}

impl Acc {
    fn new() -> Self {
        Self {
            text_parts: Vec::new(),
            mel_parts: Vec::new(),
            audio_out_parts: Vec::new(),
            audio_in_lens: Vec::new(),
            modality_seq: Vec::new(),
            supervision_seq: Vec::new(),
        }
    }
}

impl<'a> LFM2AudioChatMapper<'a> {
    /// `__init__(processor, *, codebooks=8, interleaved_text_tokens=6,
    /// interleaved_audio_tokens=12)`.
    pub fn new(
        processor: &'a LFM2AudioProcessor,
        codebooks: usize,
        interleaved_text_tokens: usize,
        interleaved_audio_tokens: usize,
    ) -> Self {
        Self {
            processor,
            codebooks,
            interleaved_text_tokens,
            interleaved_audio_tokens,
        }
    }

    /// `__init__` with the Python keyword defaults (`codebooks=8`,
    /// `interleaved_text_tokens=6`, `interleaved_audio_tokens=12`).
    pub fn with_defaults(processor: &'a LFM2AudioProcessor) -> Self {
        Self::new(processor, 8, 6, 12)
    }

    /// `__call__(self, messages) -> LFM2AudioTrainingSample`.
    ///
    /// Faithful to the Python control flow: emit `<|startoftext|>` (unsupervised),
    /// then for each message emit `<|im_start|>{role}\n`, walk the content
    /// segments (interleaved / text / audio, with the assistant-vs-user audio
    /// split), and close with `<|im_end|>\n`. Finally concat the parts into the
    /// six tensors of the sample.
    pub fn call(&self, messages: &[ChatMessage]) -> Result<LFM2AudioTrainingSample> {
        let mut acc = Acc::new();

        self.append_text("<|startoftext|>", false, &mut acc)?;

        for msg in messages {
            self.append_text(
                &format!("<|im_start|>{}\n", msg.role().as_str()),
                false,
                &mut acc,
            )?;

            for segment in msg.content() {
                match segment {
                    ChatContentSegment::Interleaved(seg) => {
                        if msg.role() != Role::Assistant {
                            // raise ValueError(...)
                            return Err(candle_core::Error::Msg(
                                "InterleavedSegment is only supported for assistant messages"
                                    .into(),
                            ));
                        }
                        self.append_interleaved_out(seg, &mut acc)?;
                    }
                    ChatContentSegment::Text(seg) => {
                        self.append_text(seg.text(), msg.role() == Role::Assistant, &mut acc)?;
                    }
                    ChatContentSegment::Audio(seg) => {
                        let (wav, sampling_rate) = Self::load_audio_bytes(seg.audio())?;
                        if msg.role() == Role::Assistant {
                            self.append_text("<|audio_start|>", true, &mut acc)?;
                            self.append_audio_out(&wav, sampling_rate, &mut acc)?;
                        } else {
                            self.append_audio_in(&wav, sampling_rate, &mut acc)?;
                        }
                    }
                }
            }

            self.append_text("<|im_end|>\n", msg.role() == Role::Assistant, &mut acc)?;
        }

        self.finish(acc)
    }

    /// Faithful to the tail of `__call__` that concatenates the parts:
    /// ```python
    /// text = torch.cat(text_parts, 0).unsqueeze(0).to(torch.long)
    /// audio_in = torch.cat(mel_parts, 1) if mel_parts else torch.empty((128, 0))
    /// audio_in_lens_t = torch.tensor(audio_in_lens, torch.long)
    /// audio_out = torch.cat(audio_out_parts, 1).to(torch.long) if ... else torch.empty((C, 0), torch.long)
    /// modality_flag = torch.tensor(modality_seq).unsqueeze(0)
    /// supervision_mask = torch.tensor(supervision_seq, torch.bool).unsqueeze(0)
    /// ```
    fn finish(&self, acc: Acc) -> Result<LFM2AudioTrainingSample> {
        let dev = self.processor.device();
        let nfilt = self.processor.audio().nfilt();

        // text = cat(text_parts, 0).unsqueeze(0)  → (1, n) I64 (torch.long; the
        // crate's token-id dtype, matching the dataloader/ChatState).
        let text = if acc.text_parts.is_empty() {
            Tensor::from_vec(Vec::<i64>::new(), (1, 0), dev)?
        } else {
            let refs: Vec<&Tensor> = acc.text_parts.iter().collect();
            Tensor::cat(&refs, 0)?.unsqueeze(0)?.to_dtype(DType::I64)?
        };

        // audio_in = cat(mel_parts, 1) else empty((nfilt, 0)) — Python hardcodes
        // 128; the crate parameterizes it as the featurizer's nfilt.
        let audio_in = if acc.mel_parts.is_empty() {
            Tensor::zeros((nfilt, 0), DType::F32, dev)?
        } else {
            let refs: Vec<&Tensor> = acc.mel_parts.iter().collect();
            Tensor::cat(&refs, 1)?
        };

        let audio_in_lens = {
            let n = acc.audio_in_lens.len();
            Tensor::from_vec(acc.audio_in_lens, (n,), dev)?
        };

        // audio_out = cat(audio_out_parts, 1) else empty((codebooks, 0)). I64
        // (torch.long); mimi.encode hands back u32 codes, cast to match.
        let audio_out = if acc.audio_out_parts.is_empty() {
            Tensor::from_vec(Vec::<i64>::new(), (self.codebooks, 0), dev)?
        } else {
            let refs: Vec<&Tensor> = acc.audio_out_parts.iter().collect();
            Tensor::cat(&refs, 1)?.to_dtype(DType::I64)?
        };

        let n_mod = acc.modality_seq.len();
        let modality_flag = Tensor::from_vec(acc.modality_seq, (1, n_mod), dev)?;

        // supervision_mask: torch.bool → candle u8 (candle has no Bool dtype; the
        // model reads it back via `to_dtype(U8)`, so u8 is the faithful storage).
        let n_sup = acc.supervision_seq.len();
        let sup_u8: Vec<u8> = acc.supervision_seq.iter().map(|&b| b as u8).collect();
        let supervision_mask = Tensor::from_vec(sup_u8, (1, n_sup), dev)?;

        Ok(LFM2AudioTrainingSample {
            text,
            audio_in,
            audio_in_lens,
            audio_out,
            modality_flag,
            supervision_mask,
        })
    }

    /// `_append_interleaved_out(text, audio, ...)`.
    ///
    /// Tokenize `"{text}<|text_end|>"`, encode the audio to codes, then walk the
    /// two streams in `interleaved_text_tokens` / `interleaved_audio_tokens`
    /// chunks, extending the modality + supervision sequences (both supervised).
    fn append_interleaved_out(&self, seg: &InterleavedSegment, acc: &mut Acc) -> Result<()> {
        let text_tokens = self.encode_ids(&format!("{}<|text_end|>", seg.text()))?;
        let text_left_total = text_tokens.dim(0)?;
        acc.text_parts.push(text_tokens);

        let (wav, sampling_rate) = Self::load_audio_bytes(seg.audio())?;
        let audio_out = self.encode_audio_out(&wav, sampling_rate)?;
        let audio_left_total = audio_out.dim(1)?;
        acc.audio_out_parts.push(audio_out);

        let n_text = self.interleaved_text_tokens;
        let n_audio = self.interleaved_audio_tokens;
        let mut text_left = text_left_total;
        let mut audio_left = audio_left_total;
        while text_left > 0 || audio_left > 0 {
            let take_text = n_text.min(text_left);
            if take_text > 0 {
                acc.modality_seq
                    .extend(std::iter::repeat_n(LFMModality::Text as i64, take_text));
                acc.supervision_seq
                    .extend(std::iter::repeat_n(true, take_text));
                text_left -= take_text;
            }

            let take_audio = n_audio.min(audio_left);
            if take_audio > 0 {
                acc.modality_seq.extend(std::iter::repeat_n(
                    LFMModality::AudioOut as i64,
                    take_audio,
                ));
                acc.supervision_seq
                    .extend(std::iter::repeat_n(true, take_audio));
                audio_left -= take_audio;
            }
        }
        Ok(())
    }

    /// `_append_text(text, *, supervised, ...)` — tokenize, push the row, and
    /// extend the modality (TEXT) + supervision sequences by the token count.
    fn append_text(&self, text: &str, supervised: bool, acc: &mut Acc) -> Result<()> {
        let text_tokens = self.encode_ids(text)?;
        let n = text_tokens.dim(0)?;
        acc.text_parts.push(text_tokens);
        acc.modality_seq
            .extend(std::iter::repeat_n(LFMModality::Text as i64, n));
        acc.supervision_seq
            .extend(std::iter::repeat_n(supervised, n));
        Ok(())
    }

    /// `_append_audio_in(wav, sampling_rate, ...)`.
    ///
    /// Resample to 16 kHz, run the mel front-end, keep the valid mel frames, push
    /// the block + its length, and extend the modality (AUDIO_IN) + supervision
    /// (`False`) sequences by `mel2emb_len(valid)`.
    fn append_audio_in(&self, wav: &Tensor, sampling_rate: u32, acc: &mut Acc) -> Result<()> {
        let wav = wav.to_dtype(DType::F32)?;
        let wav = if sampling_rate != AUDIO_IN_SAMPLE_RATE {
            resample(&wav, sampling_rate, AUDIO_IN_SAMPLE_RATE)?
        } else {
            wav
        };

        // Native writes only the valid `(nfilt, mel_len)` destination. Do not
        // upload centered/pad_to tail columns merely to crop-copy them here.
        let cur_mel = self.processor.audio().forward_valid(&wav)?;
        let cur_len = cur_mel.dim(1)?;

        acc.mel_parts.push(cur_mel);
        acc.audio_in_lens.push(cur_len as i64);

        let n_emb = mel2emb_len(cur_len as i64) as usize;
        acc.modality_seq
            .extend(std::iter::repeat_n(LFMModality::AudioIn as i64, n_emb));
        acc.supervision_seq
            .extend(std::iter::repeat_n(false, n_emb));
        Ok(())
    }

    /// `_append_audio_out(wav, sampling_rate, ...)` — encode to codes, push the
    /// block, and extend the modality (AUDIO_OUT) + supervision (`True`)
    /// sequences by the code-frame count.
    fn append_audio_out(&self, wav: &Tensor, sampling_rate: u32, acc: &mut Acc) -> Result<()> {
        let codes = self.encode_audio_out(wav, sampling_rate)?;
        let n = codes.dim(1)?;
        acc.audio_out_parts.push(codes);
        acc.modality_seq
            .extend(std::iter::repeat_n(LFMModality::AudioOut as i64, n));
        acc.supervision_seq.extend(std::iter::repeat_n(true, n));
        Ok(())
    }

    /// `_encode_audio_out(wav, sampling_rate) -> (codebooks, m+1)`.
    ///
    /// Resample to the codec rate, `mimi.encode(wav.unsqueeze(0))[0]`, keep the
    /// first `codebooks` rows, and append the EOAudio (`2048`) column. Faithful to:
    /// ```python
    /// codes = self.processor.mimi.encode(wav.unsqueeze(0))[0].cpu()
    /// codes = codes[:self.codebooks].to(torch.long)
    /// end_of_audio = torch.full((self.codebooks, 1), 2048, torch.long)
    /// return torch.cat([codes, end_of_audio], dim=1)
    /// ```
    fn encode_audio_out(&self, wav: &Tensor, sampling_rate: u32) -> Result<Tensor> {
        let wav = wav.to_dtype(DType::F32)?;
        let mimi_sample_rate = self
            .processor
            .mimi_sample_rate()
            .unwrap_or(DEFAULT_MIMI_SAMPLE_RATE);
        let wav = if sampling_rate != mimi_sample_rate {
            resample(&wav, sampling_rate, mimi_sample_rate)?
        } else {
            wav
        };

        // wav.unsqueeze(0): the codec wants (B, 1, L); `wav` is (1, L) → (1, 1, L).
        let wav3 = wav.unsqueeze(0)?;
        let codes = self.processor.mimi_encode(&wav3)?; // (1, codebooks_all, T)
        let codes = codes.i(0)?; // [0] → (codebooks_all, T)
        let kept = codes.dim(0)?.min(self.codebooks);
        let codes = codes.narrow(0, 0, kept)?.to_dtype(DType::U32)?; // (codebooks, T)

        // end_of_audio = full((codebooks, 1), 2048)
        let end_of_audio = Tensor::full(END_OF_AUDIO, (self.codebooks, 1), wav.device())?;
        // The codec may return fewer rows than `codebooks`; the EOAudio frame is
        // always `codebooks` rows, so widen the kept codes to match before cat.
        let codes = if kept == self.codebooks {
            codes
        } else {
            let pad = Tensor::zeros(
                (self.codebooks - kept, codes.dim(1)?),
                DType::U32,
                wav.device(),
            )?;
            Tensor::cat(&[&codes, &pad], 0)?
        };
        Tensor::cat(&[&codes, &end_of_audio], 1)
    }

    /// `self.processor.text.encode(text, add_special_tokens=False).squeeze(0)` →
    /// a `(n,)` u32 token-id row. The crate's [`LFM2AudioProcessor::encode`]
    /// returns `(1, n)` (already without special tokens), so we drop the batch
    /// dim to match the Python `.squeeze(0)`.
    fn encode_ids(&self, text: &str) -> Result<Tensor> {
        self.processor.encode(text)?.squeeze(0)
    }

    /// `_load_audio_bytes(audio) -> (wav (1, L) f32, sampling_rate)`.
    ///
    /// Python decodes the encoded-audio `bytes` with `soundfile.read(..., dtype=
    /// "float32", always_2d=True)`, transposes to channel-major, and mono-downmixes
    /// (`mean over channels`) if multichannel. We decode with `symphonia` (pure
    /// Rust), which reads the same containers libsndfile/`soundfile` does — WAV,
    /// FLAC, OGG/Vorbis, AIFF, … and more — to interleaved f32, then keep the
    /// leading channel dim and mono-downmix, matching the Python exactly.
    pub fn load_audio_bytes(audio: &[u8]) -> Result<(Tensor, u32)> {
        let DecodedAudio {
            samples,
            channels,
            sample_rate,
        } = decode_audio(audio)?;
        let n_frames = if channels == 0 {
            0
        } else {
            samples.len() / channels as usize
        };

        // data.T then mean over channels if > 1 → a single (L,) mono row.
        let mut mono = vec![0f32; n_frames];
        if channels <= 1 {
            mono.copy_from_slice(&samples[..n_frames]);
        } else {
            let c = channels as usize;
            for (frame, slot) in mono.iter_mut().enumerate() {
                let mut s = 0f32;
                for ch in 0..c {
                    s += samples[frame * c + ch];
                }
                *slot = s / c as f32;
            }
        }

        // wav: (1, L) — Python keeps the leading channel dim (keepdim mean / .T).
        let wav = Tensor::from_vec(mono, (1, n_frames), &candle_core::Device::Cpu)?;
        Ok((wav, sample_rate))
    }
}

/// Decoded audio in interleaved channel order (frame-major), as f32 in [-1, 1].
struct DecodedAudio {
    samples: Vec<f32>,
    channels: u16,
    sample_rate: u32,
}

/// `soundfile.read(stream, dtype="float32", always_2d=True)` equivalent — decode
/// an encoded-audio byte stream to interleaved f32 using `symphonia` (pure Rust).
/// Probes the container (WAV, FLAC, OGG/Vorbis, AIFF, MP3, … — a superset of the
/// formats libsndfile reads), decodes every packet, and concatenates the
/// interleaved samples. No torch, no C deps.
fn decode_audio(bytes: &[u8]) -> Result<DecodedAudio> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let err = |m: String| candle_core::Error::Msg(format!("load_audio_bytes: {m}"));

    let mss = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        Default::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| err(format!("unsupported/undecodable audio container: {e}")))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| err("no decodable audio track".into()))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| err(format!("no decoder for codec: {e}")))?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels: u16 = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream (symphonia signals EOF as an UnexpectedEof IoError).
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(err(format!("read error: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                if sample_rate == 0 {
                    sample_rate = spec.rate;
                }
                if channels == 0 {
                    channels = spec.channels.count() as u16;
                }
                let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sb.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sb.samples());
            }
            // A single corrupt packet is skippable (libsndfile is similarly lenient).
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(err(format!("decode error: {e}"))),
        }
    }

    if sample_rate == 0 {
        return Err(err("decoded audio sample rate is zero".into()));
    }
    if channels == 0 || samples.is_empty() {
        return Err(err("decoded no audio samples".into()));
    }
    Ok(DecodedAudio {
        samples,
        channels,
        sample_rate,
    })
}

/// `torchaudio.functional.resample(wav, orig, new)` — the faithful windowed-sinc
/// resampler (default `sinc_interp_hann`, width 6, rolloff 0.99), shared with the
/// processor. `wav` is `(1, L)` f32 → `(1, L')` f32, `L' = ceil(L * new / orig)`.
/// See [`crate::resample`] for the kernel construction (a 1:1 port of torchaudio).
fn resample(wav: &Tensor, orig: u32, new: u32) -> Result<Tensor> {
    crate::resample::resample(wav, orig, new)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a little-endian 16-bit PCM mono WAV for decoder tests.
    fn make_wav_mono_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let mut v = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&1u16.to_le_bytes()); // channels
        v.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        v.extend_from_slice(&byte_rate.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes()); // block align
        v.extend_from_slice(&16u16.to_le_bytes()); // bits
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// Two-channel interleaved 16-bit PCM WAV (for the mono-downmix path).
    fn make_wav_stereo_i16(frames: &[(i16, i16)], sample_rate: u32) -> Vec<u8> {
        let mut v = Vec::new();
        let data_len = (frames.len() * 4) as u32;
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVE");
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes()); // 2 channels
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&(sample_rate * 4).to_le_bytes());
        v.extend_from_slice(&4u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for (l, r) in frames {
            v.extend_from_slice(&l.to_le_bytes());
            v.extend_from_slice(&r.to_le_bytes());
        }
        v
    }

    #[test]
    fn decodes_mono_pcm16() {
        let wav_bytes = make_wav_mono_i16(&[0, 16384, -16384, 32767], 16_000);
        let (wav, sr) = LFM2AudioChatMapper::load_audio_bytes(&wav_bytes).unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(wav.dims(), &[1, 4]);
        let v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-3);
        assert!((v[2] + 0.5).abs() < 1e-3);
    }

    #[test]
    fn downmixes_stereo_to_mono() {
        // (L, R) frames; mono = mean → (L+R)/2.
        let wav_bytes = make_wav_stereo_i16(&[(16384, -16384), (32767, 32767)], 24_000);
        let (wav, sr) = LFM2AudioChatMapper::load_audio_bytes(&wav_bytes).unwrap();
        assert_eq!(sr, 24_000);
        assert_eq!(wav.dims(), &[1, 2]);
        let v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v[0].abs() < 1e-3, "0.5 and -0.5 average to ~0");
        assert!((v[1] - 0.99997).abs() < 1e-3);
    }

    #[test]
    fn rejects_undecodable_bytes() {
        // Not WAV-specific any more: symphonia accepts every libsndfile-ish
        // container, but truly undecodable bytes still error (no silent silence).
        let err = LFM2AudioChatMapper::load_audio_bytes(b"this is not audio at all").unwrap_err();
        let _ = format!("{err}"); // any error is fine; must not panic / succeed
    }

    /// Multi-format decode (the soundfile-equivalent [P1] fix): decode real AIFF +
    /// ALAC/m4a files — formats the old WAV-only decoder rejected. Skips if the
    /// repository fixtures aren't present (generate beside `question.wav` with
    /// `afconvert -f AIFF question.wav question.aiff` and
    /// `afconvert -f m4af -d alac question.wav question.m4a`).
    #[test]
    fn decodes_non_wav_containers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        for name in ["question.aiff", "question.m4a"] {
            let path = root.join(name);
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{name} {path:?}: {e}"));
            let (wav, sr) = LFM2AudioChatMapper::load_audio_bytes(&bytes)
                .unwrap_or_else(|e| panic!("decode {name} failed: {e}"));
            assert_eq!(wav.dims().len(), 2, "{name}: expected (1, L)");
            assert_eq!(wav.dim(0).unwrap(), 1, "{name}: expected mono row");
            assert!(wav.dim(1).unwrap() > 1000, "{name}: too few samples");
            assert!(sr >= 8000, "{name}: implausible sample rate {sr}");
            let v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let peak = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
            assert!(peak.is_finite() && peak > 0.0, "{name}: silent/NaN decode");
            eprintln!(
                "{name}: decoded {} samples @ {sr} Hz, peak {peak:.3}",
                v.len()
            );
        }
    }

    #[test]
    fn resample_changes_length_by_ratio() {
        let x = Tensor::from_vec(
            (0..100).map(|i| i as f32).collect::<Vec<_>>(),
            (1, 100),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let down = resample(&x, 24_000, 16_000).unwrap();
        assert_eq!(down.dims(), &[1, 67]); // round(100 * 16000/24000) = 67
        let up = resample(&x, 16_000, 24_000).unwrap();
        assert_eq!(up.dims(), &[1, 150]); // round(100 * 24000/16000) = 150
    }

    #[test]
    fn resample_noop_when_rates_equal() {
        let x = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3), &candle_core::Device::Cpu).unwrap();
        let y = resample(&x, 16_000, 16_000).unwrap();
        assert_eq!(y.dims(), &[1, 3]);
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1f32, 2., 3.]
        );
    }
}
