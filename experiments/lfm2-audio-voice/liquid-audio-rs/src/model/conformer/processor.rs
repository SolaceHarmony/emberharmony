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

    /// The mel config (hop/window/fft sizes) backing this featurizer. Lets
    /// callers (e.g. the data mapper) recover the hop length to compute the valid
    /// frame count `floor(L/hop)` — the Python `mel_len`.
    pub fn mel_config(&self) -> MelConfig {
        self.cfg.clone()
    }

    /// PORT: `FilterbankFeatures.get_seq_len` (py L412-416).
    ///
    /// Computes the number of valid mel frames produced for an input of
    /// `seq_len` samples. Python:
    /// ```python
    /// pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
    /// seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
    /// ```
    /// This port only supports the centered (`exact_pad=False`) path, so
    /// `stft_pad_amount is None` and `pad_amount = (n_fft // 2) * 2`. With an
    /// even `n_fft`, `pad_amount == n_fft`, so the formula collapses to
    /// `floor_divide(seq_len, hop_length)` — i.e. `seq_len / hop` in integer
    /// arithmetic. Exposed publicly so callers (the data mapper) can use the
    /// featurizer-computed length instead of recomputing `L / hop` by hand.
    pub fn get_seq_len(&self, seq_len: usize) -> usize {
        // pad_amount for the centered path: (n_fft // 2) * 2.
        let pad_amount = (self.cfg.n_fft / 2) * 2;
        // torch.floor_divide on non-negative ints == integer division. Guard the
        // (seq_len + pad_amount - n_fft) subtraction against underflow; for the
        // even-n_fft case pad_amount == n_fft so this is exactly seq_len.
        let numer = seq_len + pad_amount;
        let numer = numer.saturating_sub(self.cfg.n_fft);
        if self.cfg.n_window_stride > 0 {
            numer / self.cfg.n_window_stride
        } else {
            0
        }
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

    /// PORT: `FilterbankFeatures.stft` (py L385-395).
    ///
    /// Centered short-time Fourier transform. Python calls `torch.stft(x, n_fft,
    /// hop_length, win_length, center=True, window=..., return_complex=True,
    /// pad_mode="constant")` and returns the complex spectrogram of shape
    /// `[freq, T]` (per clip). This port replays the same operation with
    /// `rustfft`: the (already preemphasised) signal `y` is centre-padded with
    /// `n_fft/2` zeros each side (`pad_mode="constant"`, matching `center=True`),
    /// each `n_fft`-sample frame at stride `hop_length` is windowed and FFT'd, and
    /// only the first `freq = n_fft/2 + 1` (non-redundant) bins are kept. Returns
    /// the complex bins freq-major: `out[f * t + ti]`, with `t = 1 + L/hop` frames
    /// (the `center=True` frame count). Byte-identical to the previously inlined
    /// FFT loop — only the FFT planning + per-frame transform moved here.
    fn stft(&self, y: &[f32]) -> Vec<Complex<f32>> {
        let l = y.len();
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.n_window_stride;
        let freq = n_fft / 2 + 1;
        // `t = 1 + L/hop` is torch.stft(center=True)'s emitted frame count.
        let t = 1 + l / hop;

        // center pad with n_fft/2 zeros each side (pad_mode="constant")
        let pad = n_fft / 2;
        let mut padded = vec![0f32; l + 2 * pad];
        padded[pad..pad + l].copy_from_slice(y);

        let window = self.padded_window();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let mut out = vec![Complex { re: 0f32, im: 0f32 }; freq * t];
        let mut buf = vec![Complex { re: 0f32, im: 0f32 }; n_fft];
        for ti in 0..t {
            let start = ti * hop;
            for j in 0..n_fft {
                buf[j].re = padded[start + j] * window[j];
                buf[j].im = 0.0;
            }
            fft.process(&mut buf);
            for f in 0..freq {
                out[f * t + ti] = buf[f];
            }
        }
        out
    }

    /// PORT: `FilterbankFeatures.log_zero_guard_value_fn` (py L397-410).
    ///
    /// Returns the additive/clamp epsilon used before `log()`. Python supports a
    /// plain number or the strings `"tiny"`/`"eps"` (resolved via `torch.finfo`).
    /// This port carries `log_zero_guard_value` as a concrete `f64` in `MelConfig`
    /// (the checkpoint configs use the numeric default `2**-24`), so the string
    /// branches are pre-resolved at config load; the value is returned as-is,
    /// matching the numeric branch (`return self.log_zero_guard_value`). `_x` is
    /// accepted to mirror the Python signature (`log_zero_guard_value_fn(self, x)`),
    /// where it only matters for the dtype-dependent `"tiny"`/`"eps"` cases.
    fn log_zero_guard_value_fn(&self, _x: &Tensor) -> f64 {
        self.cfg.log_zero_guard_value
    }

    /// `samples` is mono PCM in [-1,1] as `(L,)` or `(1, L)`. Returns `(1, nfilt, T)`.
    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        let x = samples.flatten_all()?.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        let l = x.len();
        let n_fft = self.cfg.n_fft;
        let freq = n_fft / 2 + 1;
        // NeMo `get_seq_len`: floor(L/hop) valid frames. torch.stft(center=True)
        // emits `1 + L/hop` frames, so the trailing frame is a pad column (masked
        // below). seq_len is now produced by the `get_seq_len` method.
        let seq_len = self.get_seq_len(l);
        let t = 1 + l / self.cfg.n_window_stride;

        // preemphasis: y[0]=x[0]; y[i]=x[i]-preemph*x[i-1]
        let pre = self.cfg.preemph as f32;
        let mut y = vec![0f32; l];
        if l > 0 {
            y[0] = x[0];
            for i in 1..l {
                y[i] = x[i] - pre * x[i - 1];
            }
        }

        // disable autocast to get full range of stft values: x = self.stft(x).
        // Returns the complex spectrogram freq-major (`spec[f * t + ti]`).
        let spec = self.stft(&y);

        // torch stft returns a complex tensor; convert to magnitude then power.
        // Python: x = sqrt(re^2 + im^2 + guard); if mag_power != 1: x = x.pow(mag_power).
        // (guard == 0 for the inference path, use_grads=False.)
        let mag_power = self.cfg.mag_power as f32;
        let mut power = vec![0f32; freq * t];
        for (i, c) in spec.iter().enumerate() {
            // Fast path for the common mag_power == 2 (= re^2 + im^2); honor any
            // other configured power faithfully.
            let p = c.re * c.re + c.im * c.im;
            power[i] = if mag_power == 2.0 { p } else { p.sqrt().powf(mag_power) };
        }

        let spec = Tensor::from_vec(power, (freq, t), &self.device)?; // (freq, T)
        // mel: (nfilt, freq) @ (freq, T) → (nfilt, T)
        let mut mel = self.fb.matmul(&spec)?;
        // log(x + guard) — guard from log_zero_guard_value_fn (log_zero_guard_type="add").
        // Bind the guard first: `mel + …` moves `mel`, so the `&mel` borrow must resolve before.
        let guard = self.log_zero_guard_value_fn(&mel);
        mel = (mel + guard)?.log()?;
        // per-feature normalization (ddof=1) over the valid frames only, applied
        // to all frames — faithful to normalize_batch's valid_mask.
        mel = normalize_batch(&mel, seq_len)?;
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

/// PORT: module-level `normalize_batch(x, seq_len, normalize_type)` (py L503-556),
/// `"per_feature"` branch, for a single clip.
///
/// Per-mel-bin (per-feature) mean/std normalization across time. Python masks the
/// time axis with `valid_mask = time_steps < seq_len` and computes, per feature:
/// `x_mean = sum(x where valid) / count(valid)`, `x_std = sqrt(sum((x - x_mean)^2
/// where valid) / (count - 1))` (ddof=1 bias correction), then `x_std += CONSTANT`
/// (1e-5) to avoid divide-by-zero, returning `(x - x_mean) / x_std` broadcast over
/// ALL time steps. Here the `valid` frames are the leading `[0, valid)` columns
/// (the trailing centred-STFT pad frame is excluded from the statistics and is
/// masked to 0 by the caller afterwards). Byte-identical to the previously inlined
/// `normalize_per_feature` — only the function name now matches Python. Other
/// `normalize_type` branches (`"all_features"`, `"fixed_mean"/"fixed_std"`, none)
/// are unreachable on the checkpoint config (`normalize="per_feature"`) and so are
/// not exercised on the inference path.
fn normalize_batch(x: &Tensor, valid: usize) -> Result<Tensor> {
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
///
/// `win_length` / `hop_length` are the base-class `AudioPreprocessor.__init__`
/// state (py L34-58): Python's `AudioToMelSpectrogramPreprocessor.__init__`
/// computes `n_window_size = int(window_size * sample_rate)` /
/// `n_window_stride = int(window_stride * sample_rate)` and forwards them to
/// `super().__init__(n_window_size, n_window_stride)`. The `torch_windows` dict and
/// `dtype_sentinel_tensor` buffer from the base `__init__` have no candle analog
/// (the window function is resolved in `FilterbankFeatures::new`; the output is
/// always f32 here, so the dtype sentinel is a no-op) and are intentionally not
/// carried.
pub struct AudioToMelSpectrogramPreprocessor {
    featurizer: FilterbankFeatures,
    /// base `AudioPreprocessor.win_length` (= `n_window_size`).
    win_length: usize,
    /// base `AudioPreprocessor.hop_length` (= `n_window_stride`).
    hop_length: usize,
}

impl AudioToMelSpectrogramPreprocessor {
    /// PORT: `AudioToMelSpectrogramPreprocessor.__init__` (py L152-227) +
    /// `AudioPreprocessor.__init__` (py L34-58). The full Python ctor wires a long
    /// config into a `FilterbankFeatures`; here the `featurizer` is built
    /// separately (`FilterbankFeatures::new`) and injected, and the base-class
    /// `win_length`/`hop_length` are recovered from its `MelConfig`
    /// (`n_window_size`/`n_window_stride`) — matching Python's
    /// `super().__init__(n_window_size, n_window_stride)`.
    pub fn new(featurizer: FilterbankFeatures) -> Self {
        let cfg = featurizer.mel_config();
        let win_length = cfg.n_window_size;
        let hop_length = cfg.n_window_stride;
        Self { featurizer, win_length, hop_length }
    }

    /// base `AudioPreprocessor.win_length` (py L37).
    pub fn win_length(&self) -> usize {
        self.win_length
    }

    /// base `AudioPreprocessor.hop_length` (py L38).
    pub fn hop_length(&self) -> usize {
        self.hop_length
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
