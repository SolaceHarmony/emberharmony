//! Port of `liquid_audio/model/conformer/processor.py` — NeMo mel featurizer
//! (`AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures`), inference path.
//!
//! The `window` (Hann) and mel filterbank `fb` are persistent buffers in the
//! checkpoint, so they are loaded verbatim (no librosa/hann reimplementation).
//! Pipeline: preemphasis → centered STFT (`rustfft`) → power spectrum → mel →
//! log → per-feature normalization → pad to a multiple of `pad_to`.
//! Training-only bits (dither, nb-augmentation, frame splicing) are skipped.

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use rustfft::{num_complex::Complex, FftPlanner};

/// Subset of NeMo's preprocessor config needed offline.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub n_window_size: usize,   // win_length (e.g. 400)
    pub n_window_stride: usize, // hop_length (e.g. 160)
    pub n_fft: usize,           // e.g. 512
    pub nfilt: usize,           // mel bins (feat_in of the encoder)
    pub preemph: f64,           // 0.97
    pub log_zero_guard_value: f64, // 2^-24
    pub mag_power: f64,         // 2.0
    pub pad_to: usize,          // 16
}

const CONSTANT: f64 = 1e-5;

pub struct FilterbankFeatures {
    cfg: MelConfig,
    window: Vec<f32>, // loaded (win_length,), padded to n_fft at use
    fb: Tensor,       // (nfilt, n_fft/2+1)
    device: Device,
}

impl FilterbankFeatures {
    pub fn new(cfg: MelConfig, vb: VarBuilder) -> Result<Self> {
        let window = vb.get(cfg.n_window_size, "window")?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        let freq = cfg.n_fft / 2 + 1;
        // fb is registered as (1, nfilt, freq); accept either shape.
        let fb = vb.get((1, cfg.nfilt, freq), "fb")?.reshape((cfg.nfilt, freq))?.to_dtype(candle_core::DType::F32)?;
        let device = vb.device().clone();
        Ok(Self { cfg, window, fb, device })
    }

    /// Window padded (centered) to n_fft, as torch.stft does for win_length < n_fft.
    fn padded_window(&self) -> Vec<f32> {
        let n = self.cfg.n_fft;
        let w = &self.window;
        if w.len() == n {
            return w.clone();
        }
        let left = (n - w.len()) / 2;
        let mut out = vec![0f32; n];
        out[left..left + w.len()].copy_from_slice(w);
        out
    }

    /// `samples` is mono PCM in [-1,1] as `(L,)` or `(1, L)`. Returns `(1, nfilt, T)`.
    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        let x = samples.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        let l = x.len();
        let hop = self.cfg.n_window_stride;
        let n_fft = self.cfg.n_fft;
        let freq = n_fft / 2 + 1;
        // get_seq_len (center): floor(L / hop)
        let t = l / hop;

        // preemphasis: y[0]=x[0]; y[i]=x[i]-preemph*x[i-1]
        let pre = self.cfg.preemph as f32;
        let mut y = vec![0f32; l];
        if l > 0 {
            y[0] = x[0];
            for i in 1..l {
                y[i] = x[i] - pre * x[i - 1];
            }
        }

        // center pad with n_fft/2 zeros each side (pad_mode="constant")
        let pad = n_fft / 2;
        let mut padded = vec![0f32; l + 2 * pad];
        padded[pad..pad + l].copy_from_slice(&y);

        let window = self.padded_window();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // power spectrum, freq-major: power[f*t_total + ti]
        let mut power = vec![0f32; freq * t];
        let mut buf = vec![Complex { re: 0f32, im: 0f32 }; n_fft];
        for ti in 0..t {
            let start = ti * hop;
            for j in 0..n_fft {
                buf[j].re = padded[start + j] * window[j];
                buf[j].im = 0.0;
            }
            fft.process(&mut buf);
            for f in 0..freq {
                // mag_power=2 → power = re^2 + im^2
                power[f * t + ti] = buf[f].re * buf[f].re + buf[f].im * buf[f].im;
            }
        }

        let spec = Tensor::from_vec(power, (freq, t), &self.device)?; // (freq, T)
        // mel: (nfilt, freq) @ (freq, T) → (nfilt, T)
        let mut mel = self.fb.matmul(&spec)?;
        // log(x + guard)
        mel = (mel + self.cfg.log_zero_guard_value)?.log()?;
        // per-feature normalization over time (ddof=1)
        mel = normalize_per_feature(&mel, t)?;
        // pad time to a multiple of pad_to with zeros
        if self.cfg.pad_to > 0 {
            let rem = t % self.cfg.pad_to;
            if rem != 0 {
                let padding = Tensor::zeros((self.cfg.nfilt, self.cfg.pad_to - rem), mel.dtype(), &self.device)?;
                mel = Tensor::cat(&[&mel, &padding], 1)?;
            }
        }
        mel.unsqueeze(0) // (1, nfilt, T_padded)
    }
}

/// `normalize_batch(..., "per_feature")` for a single full clip: mean/std over
/// time per mel bin, std with ddof=1, `+ CONSTANT`.
fn normalize_per_feature(x: &Tensor, t: usize) -> Result<Tensor> {
    let mean = x.mean_keepdim(1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(1)? / (t as f64 - 1.0))?;
    let std = (var.sqrt()? + CONSTANT)?;
    centered.broadcast_div(&std)
}
