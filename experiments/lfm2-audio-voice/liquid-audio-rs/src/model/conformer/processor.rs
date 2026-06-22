//! Port of `liquid_audio/model/conformer/processor.py` — NeMo mel featurizer
//! (`AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures`), inference path.
//!
//! The `window` (Hann) and mel filterbank `fb` are **computed at construction**
//! (`torch.hann_window(periodic=False)` + `librosa.filters.mel(norm="slaney")`),
//! exactly as the Python preprocessor does in its `__init__` — they are NOT
//! checkpoint tensors. (Parity-verified to 1e-5 against the upstream featurizer.)
//! Pipeline: preemphasis → centered STFT (`rustfft`) → magnitude^`mag_power` →
//! mel → log → per-feature normalization → pad to a multiple of `pad_to`.
//! Training-only bits (dither, nb-augmentation, frame splicing) are skipped.

use candle_core::{Device, Result, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

/// Subset of NeMo's preprocessor config needed offline.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,     // 16000
    pub n_window_size: usize,   // win_length (e.g. 400)
    pub n_window_stride: usize, // hop_length (e.g. 160)
    pub n_fft: usize,           // e.g. 512
    pub nfilt: usize,           // mel bins (feat_in of the encoder)
    pub preemph: f64,           // 0.97
    pub log_zero_guard_value: f64, // 2^-24
    pub mag_power: f64,         // 2.0
    pub pad_to: usize,          // 16
}

/// Symmetric Hann window (`periodic=False`), faithful to `torch.hann_window(N, periodic=False)`.
fn hann(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos())
        .map(|w| w as f32)
        .collect()
}

// librosa slaney mel scale.
fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}
fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")`,
/// returned flattened `(nfilt * freq)` row-major. `freq = n_fft/2+1`.
fn mel_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let freq = n_fft / 2 + 1;
    let fmin = 0.0;
    let fmax = sr as f64 / 2.0;
    let fft_freqs: Vec<f64> = (0..freq).map(|k| k as f64 * sr as f64 / n_fft as f64).collect();
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut fb = vec![0f32; n_mels * freq];
    for m in 0..n_mels {
        let lower = mel_pts[m];
        let center = mel_pts[m + 1];
        let upper = mel_pts[m + 2];
        let enorm = 2.0 / (upper - lower); // slaney normalization
        for (k, &f) in fft_freqs.iter().enumerate() {
            let down = (f - lower) / (center - lower);
            let up = (upper - f) / (upper - center);
            let w = down.min(up).max(0.0);
            fb[m * freq + k] = (w * enorm) as f32;
        }
    }
    fb
}

const CONSTANT: f64 = 1e-5;

pub struct FilterbankFeatures {
    cfg: MelConfig,
    window: Vec<f32>, // loaded (win_length,), padded to n_fft at use
    fb: Tensor,       // (nfilt, n_fft/2+1)
    device: Device,
}

impl FilterbankFeatures {
    /// Computes the Hann window and slaney mel filterbank (as the Python
    /// preprocessor does at init — they are not checkpoint tensors).
    pub fn new(cfg: MelConfig, device: &Device) -> Result<Self> {
        let window = hann(cfg.n_window_size);
        let freq = cfg.n_fft / 2 + 1;
        let fb_data = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.nfilt);
        let fb = Tensor::from_vec(fb_data, (cfg.nfilt, freq), device)?;
        Ok(Self { cfg, window, fb, device: device.clone() })
    }

    /// Number of mel bins (encoder `feat_in`).
    pub fn nfilt(&self) -> usize {
        self.cfg.nfilt
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
        // torch.stft(center=True) emits `1 + L/hop` frames; NeMo's get_seq_len is
        // floor(L/hop), so the trailing frame is a pad column (masked below).
        let seq_len = l / hop;
        let t = 1 + l / hop;

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
            let mag_power = self.cfg.mag_power as f32;
            for f in 0..freq {
                // Python: x = sqrt(re^2+im^2); if mag_power != 1: x = x.pow(mag_power).
                // Fast path for the common mag_power == 2 (= re^2 + im^2); honor
                // any other configured power faithfully.
                let p = buf[f].re * buf[f].re + buf[f].im * buf[f].im;
                power[f * t + ti] = if mag_power == 2.0 { p } else { p.sqrt().powf(mag_power) };
            }
        }

        let spec = Tensor::from_vec(power, (freq, t), &self.device)?; // (freq, T)
        // mel: (nfilt, freq) @ (freq, T) → (nfilt, T)
        let mut mel = self.fb.matmul(&spec)?;
        // log(x + guard)
        mel = (mel + self.cfg.log_zero_guard_value)?.log()?;
        // per-feature normalization (ddof=1) over the valid frames only, applied
        // to all frames — faithful to normalize_batch's valid_mask.
        mel = normalize_per_feature(&mel, seq_len)?;
        // mask the trailing pad frame(s) [seq_len, t) to pad_value (0).
        if seq_len < t {
            let valid = mel.narrow(1, 0, seq_len)?;
            let pad = Tensor::zeros((self.cfg.nfilt, t - seq_len), mel.dtype(), &self.device)?;
            mel = Tensor::cat(&[&valid, &pad], 1)?;
        }
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

/// `normalize_batch(..., "per_feature")` for a single clip: mean/std per mel bin
/// over the first `valid` frames (std with ddof=1, `+ CONSTANT`), applied to ALL
/// frames of `x` (the trailing pad frames are masked to 0 by the caller).
fn normalize_per_feature(x: &Tensor, valid: usize) -> Result<Tensor> {
    let xv = x.narrow(1, 0, valid)?;
    let mean = xv.mean_keepdim(1)?;
    let var = (xv.broadcast_sub(&mean)?.sqr()?.sum_keepdim(1)? / (valid as f64 - 1.0))?;
    let std = (var.sqrt()? + CONSTANT)?;
    x.broadcast_sub(&mean)?.broadcast_div(&std)
}

impl FilterbankFeatures {
    /// `filter_banks` — the `(nfilt, n_fft/2+1)` mel filterbank (Python
    /// `featurizer.filter_banks`).
    pub fn filter_banks(&self) -> &Tensor {
        &self.fb
    }
}

/// `AudioPreprocessor` (Python `class AudioPreprocessor(nn.Module, ABC)`): the
/// base preprocessor contract — `forward(input, length)` delegates to the
/// abstract `get_features`.
pub trait AudioPreprocessor {
    /// abstract `get_features(input_signal, length)` — subclasses implement.
    fn get_features(&self, input_signal: &Tensor, length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)>;

    /// `forward(input_signal, length)` = `get_features` (Python wraps it in
    /// `torch.no_grad()`; inference here is already grad-free).
    fn forward(&self, input_signal: &Tensor, length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        self.get_features(input_signal, length)
    }
}

/// `AudioToMelSpectrogramPreprocessor(AudioPreprocessor)` — wraps the mel
/// `FilterbankFeatures` (Python `self.featurizer`); `get_features` / `filter_banks`
/// delegate to it.
pub struct AudioToMelSpectrogramPreprocessor {
    featurizer: FilterbankFeatures,
}

impl AudioToMelSpectrogramPreprocessor {
    pub fn new(featurizer: FilterbankFeatures) -> Self {
        Self { featurizer }
    }

    /// `filter_banks` → the featurizer's mel filterbank.
    pub fn filter_banks(&self) -> &Tensor {
        self.featurizer.filter_banks()
    }

    /// PORT: `save_to` — NeMo `.nemo` archive (tar + yaml config + pickled
    /// weights). No candle/Rust analog; persistence is via safetensors +
    /// `from_pretrained`. No-op, preserved for 1:1 inventory.
    pub fn save_to(&self, _save_path: &str) {}

    /// PORT: `restore_from` — load from a NeMo `.nemo` archive (classmethod).
    /// No candle analog (see `save_to`); use `from_pretrained`. Preserved for 1:1.
    pub fn restore_from(_restore_path: &str) {}

    /// PORT: `input_example` — ONNX-export dummy input (random tensor for tracing).
    /// No export path here; preserved for 1:1 inventory.
    pub fn input_example(&self, _max_batch: usize, _max_dim: usize, _min_length: usize) {}
}

impl AudioPreprocessor for AudioToMelSpectrogramPreprocessor {
    /// `get_features` → `self.featurizer(input_signal, length)`. The Rust mel
    /// featurizer returns the features; per-clip valid length is tracked by the
    /// caller (`ChatState`), so the length slot is `None` here.
    fn get_features(&self, input_signal: &Tensor, _length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        Ok((self.featurizer.forward(input_signal)?, None))
    }
}
