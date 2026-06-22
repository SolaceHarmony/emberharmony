//! Port of `liquid_audio/detokenizer.py` — the LFM2.5 custom audio detokenizer.
//!
//! `FusedEmbedding` (codes → embeddings) → ×6 nearest upsample → `Lfm2Model`
//! backbone under a sliding-window causal mask → Linear(512→1282) → split into
//! log-magnitude + angle → polar → Vocos-style `ISTFT` → 24 kHz waveform.
//! The ISTFT uses an inverse FFT (`rustfft`) + overlap-add with window-envelope
//! normalization ("same" padding), faithful to the Python.

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{linear, Embedding, Linear, Module, VarBuilder};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::model::lfm2_hf::{Cache, Lfm2Config, Model as Lfm2Model};

const AUDIO_VOCAB: usize = 2048;
const CODEBOOKS: usize = 8;
const EMB_DIM: usize = 512;

/// `FusedEmbedding`: per-codebook offset embedding, averaged over codebooks.
struct FusedEmbedding {
    emb: Embedding,
    offsets: Tensor, // (codebooks,)
}

impl FusedEmbedding {
    fn new(vb: VarBuilder) -> Result<Self> {
        let emb = candle_nn::embedding(CODEBOOKS * AUDIO_VOCAB, EMB_DIM, vb.pp("emb"))?;
        let offs: Vec<i64> = (0..CODEBOOKS as i64).map(|i| i * AUDIO_VOCAB as i64).collect();
        let offsets = Tensor::from_vec(offs, (CODEBOOKS,), vb.device())?;
        Ok(Self { emb, offsets })
    }

    /// `x`: (B, L, codebooks) u32 codes → (B, L, dim).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // offset_x = offsets[None,None,:] + x   (codebooks last)
        let offsets = self.offsets.reshape((1, 1, CODEBOOKS))?.to_dtype(x.dtype())?;
        let offset_x = x.broadcast_add(&offsets)?; // (B, L, codebooks)
        let (b, l, _) = offset_x.dims3()?;
        let flat = offset_x.reshape((b * l * CODEBOOKS,))?;
        let emb = self.emb.forward(&flat)?.reshape((b, l, CODEBOOKS, EMB_DIM))?;
        emb.mean(2) // average over codebooks → (B, L, dim)
    }
}

/// Vocos-style inverse STFT, "same" padding. n_fft=1280, hop=320, win=1280.
struct Istft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
}

impl Istft {
    fn new(n_fft: usize, hop: usize, win_length: usize, vb: VarBuilder) -> Result<Self> {
        let window = vb.get(win_length, "window")?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        Ok(Self { n_fft, hop, window })
    }

    /// `re`/`im`: (1, n_fft/2+1, T) → waveform (1, L).
    fn forward(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        let (_b, n, t) = re.dims3()?;
        let device = re.device().clone();
        let re = re.i(0)?.to_vec2::<f32>()?; // [n][t]
        let im = im.i(0)?.to_vec2::<f32>()?;
        let n_fft = self.n_fft;
        let hop = self.hop;
        let pad = (self.window.len() - hop) / 2;
        let out_size = (t - 1) * hop + self.window.len();
        let win_sq: Vec<f32> = self.window.iter().map(|w| w * w).collect();

        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(n_fft);

        let mut y = vec![0f32; out_size];
        let mut env = vec![0f32; out_size];
        let mut buf = vec![Complex { re: 0f32, im: 0f32 }; n_fft];
        for ti in 0..t {
            for c in buf.iter_mut() {
                *c = Complex { re: 0.0, im: 0.0 };
            }
            // one-sided → full hermitian spectrum; DC & Nyquist imag ignored (irfft)
            buf[0] = Complex { re: re[0][ti], im: 0.0 };
            for k in 1..n {
                buf[k] = Complex { re: re[k][ti], im: im[k][ti] };
            }
            if n_fft.is_multiple_of(2) {
                buf[n_fft / 2] = Complex { re: re[n - 1][ti], im: 0.0 };
            }
            for k in 1..(n_fft / 2) {
                buf[n_fft - k] = buf[k].conj();
            }
            ifft.process(&mut buf);
            let scale = 1.0 / n_fft as f32;
            for j in 0..n_fft {
                let v = buf[j].re * scale * self.window[j];
                y[ti * hop + j] += v;
                env[ti * hop + j] += win_sq[j];
            }
        }

        let lo = pad;
        let hi = out_size - pad;
        let mut out = Vec::with_capacity(hi - lo);
        for i in lo..hi {
            out.push(y[i] / env[i]);
        }
        let len = out.len();
        Tensor::from_vec(out, (1, len), &device)
    }
}

/// `LFM2AudioDetokenizer`: codes → 24 kHz waveform.
pub struct LFM2AudioDetokenizer {
    emb: FusedEmbedding,
    lfm: Lfm2Model,
    lfm_cfg: Lfm2Config,
    lin: Linear,
    istft: Istft,
    sliding_window: usize,
}

impl LFM2AudioDetokenizer {
    pub fn new(backbone_cfg: Lfm2Config, sliding_window: usize, vb: VarBuilder) -> Result<Self> {
        let emb = FusedEmbedding::new(vb.pp("emb"))?;
        let lfm = Lfm2Model::new(&backbone_cfg, vb.pp("lfm"))?;
        let lin = linear(EMB_DIM, 1282, vb.pp("lin"))?;
        let istft = Istft::new(1280, 320, 1280, vb.pp("istft"))?;
        Ok(Self { emb, lfm, lfm_cfg: backbone_cfg, lin, istft, sliding_window })
    }

    /// Additive sliding-window causal mask `(1,1,n,n)`: attend where `i-w < j <= i`.
    fn sliding_mask(&self, n: usize, device: &candle_core::Device) -> Result<Tensor> {
        let w = self.sliding_window as i64;
        let mut data = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = j as i64 - i as i64;
                if !(d <= 0 && d > -w) {
                    data[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, n, n), device)
    }

    /// `codes`: (B, L, codebooks) → waveform (1, samples).
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let x = self.emb.forward(codes)?; // (B, L, dim)
        let l = x.dim(1)?;
        // ×6 nearest-exact upsample over time = repeat-interleave by 6
        let (b, _, d) = x.dims3()?;
        let x = x.unsqueeze(2)?.broadcast_as((b, l, 6, d))?.reshape((b, l * 6, d))?.contiguous()?;
        let n = l * 6;

        let mask = self.sliding_mask(n, x.device())?;
        let mut cache = Cache::new(false, x.dtype(), &self.lfm_cfg, x.device())?;
        let h = self.lfm.forward_embeds(&x, 0, &mut cache, Some(&mask))?; // (B, n, dim)
        let x = self.lin.forward(&h)?; // (B, n, 1282)

        // split (over the 1282 feature dim, after transpose to (B,1282,n)) into
        // log-magnitude and angle, each (B, 641, n)
        let xt = x.transpose(1, 2)?.contiguous()?; // (B,1282,n)
        let half = 1282 / 2;
        let log_abs = xt.narrow(1, 0, half)?;
        let angle = xt.narrow(1, half, half)?;
        let abs = log_abs.exp()?;
        let re = (&abs * angle.cos()?)?;
        let im = (&abs * angle.sin()?)?;
        self.istft.forward(&re, &im)
    }
}
