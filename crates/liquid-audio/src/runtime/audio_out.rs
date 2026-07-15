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

use std::sync::Mutex;

use candle_core::{DType, Result, Tensor};

use crate::detokenizer::LFM2AudioDetokenizer;

/// Decode audio codes `(1, codebooks, T)` → 24 kHz mono waveform.
///
/// Required method (`decode`) ⇒ every backend must provide it (the compile-time
/// "force the override"); `sample_rate` is a default a backend may override.
// `Send` so the processor (and thus the model bundle) can move to a dedicated inference
// worker thread — the realtime full-duplex pipeline owns it there rather than sharing by
// `&` (the Mimi backend holds its streaming state behind a `Mutex`, so both backends are `Send + Sync` — required for the app-resident `Arc`-shared model, spec 09).
pub trait AudioDetokenizer: Send + Sync {
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

/// The Mimi codec behind the shared trait. The STREAMING decode path — the
/// per-frame hot path the realtime pipeline runs — is the native C++/NEON/AMX
/// kernel ([`crate::mimi_native::NativeMimi`], ~14 ms/frame, parity-gated
/// against moshi at ≤ 4.2e-6 worst PCM error). The moshi-Rust codec remains
/// ONLY for turn-level tooling: `encode` (the trainer's data mapper) and the
/// one-shot whole-clip `decode` (the byte-oracle example pins its bytes).
/// There is no cross-fallback in either direction (no-fallbacks doctrine).
pub struct MimiDetokenizer {
    inner: Mutex<::moshi::mimi::Mimi>,
    native: crate::mimi_native::NativeMimi,
}

impl MimiDetokenizer {
    pub fn new(mimi: ::moshi::mimi::Mimi, native: crate::mimi_native::NativeMimi) -> Self {
        Self {
            inner: Mutex::new(mimi),
            native,
        }
    }
}

impl MimiDetokenizer {
    /// Slice-native streaming decode — no tensor plumbing in either direction
    /// (review P1: the codes already exist as host integers in the generation
    /// loop, and every consumer wants host PCM; the Tensor round-trip added
    /// two device syncs per frame on Metal). The trait's `decode_step` is the
    /// Tensor adapter over this.
    pub fn decode_step_codes(
        &self,
        codes: &[u32],
    ) -> std::result::Result<Option<Vec<f32>>, String> {
        let pcm = self.native.decode_step(codes)?;
        Ok(if pcm.is_empty() { None } else { Some(pcm) })
    }
}

impl AudioDetokenizer for MimiDetokenizer {
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = codes.to_dtype(DType::U32)?; // RVQ index_select wants u32
        let mut m = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        m.reset_state();
        m.decode(&codes)
    }

    /// `mimi.encode(wav)` — `(B, 1, L)` 24 kHz waveform → codes `(B, codebooks, T)`.
    /// `reset_state` first ⇒ independent (non-streaming) encode, matching the
    /// Python `mimi.encode` call on a fresh clip in the data mapper.
    fn encode(&self, wav: &Tensor) -> Result<Tensor> {
        let mut m = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        m.reset_state();
        m.encode(wav)
    }

    /// Turn boundary: re-arm the NATIVE streaming state (conv carries, KV ring)
    /// — the hot path — and the moshi state alongside it so the tooling-tier
    /// codec stays coherent for any interleaved `encode` use.
    fn reset_stream(&self) {
        self.native.reset();
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .reset_state();
    }

    /// Streaming decode on the NATIVE kernel — the per-frame production path
    /// (the deprecated moshi `decode_step` call is gone from the pipeline).
    /// `frame` is `(1, codebooks, 1)`; state is kept across calls, reset at
    /// the turn boundary by [`reset_stream`](Self::reset_stream).
    fn decode_step(&self, frame: &Tensor) -> Result<Option<Tensor>> {
        let codes: Vec<u32> = frame.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?;
        match self
            .decode_step_codes(&codes)
            .map_err(candle_core::Error::Msg)?
        {
            Some(pcm) => {
                let n = pcm.len();
                // ALWAYS a CPU tensor (review P1): the consumer downloads to
                // host samples immediately — materializing PCM on frame.device()
                // (Metal) added an upload+download sync pair per frame.
                Ok(Some(Tensor::from_vec(
                    pcm,
                    (1, 1, n),
                    &candle_core::Device::Cpu,
                )?))
            }
            None => Ok(None),
        }
    }
}
