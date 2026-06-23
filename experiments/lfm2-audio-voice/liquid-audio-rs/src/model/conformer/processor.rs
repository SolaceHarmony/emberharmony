//! Port of `liquid_audio/model/conformer/processor.py` — NeMo mel featurizer
//! (`AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures`), inference path.
//!
//! The `window` (Hann), mel filterbank `fb`, and the STFT DFT-basis kernel are
//! **computed at construction** (`torch.hann_window(periodic=False)` +
//! `librosa.filters.mel(norm="slaney")`), exactly as the Python preprocessor does in
//! its `__init__` — they are NOT checkpoint tensors. (Parity-verified against the
//! upstream featurizer.)
//!
//! Pipeline: preemphasis → centered STFT → magnitude^`mag_power` → mel → log →
//! per-feature normalization → pad to a multiple of `pad_to`. **The whole chain runs
//! in candle tensor ops on the model device** (CPU or Metal) in f32 — a direct port
//! of `torch.stft` (`aten/.../SpectralOps.cpp`: center-pad → frame → ×window →
//! `_fft_r2c` → transpose) rather than an external FFT library, so it matches torch's
//! single-precision reference *and* runs on the GPU. The windowed real→complex DFT is
//! realized as a `Conv1d` against a precomputed DFT basis (stride = hop), the same
//! formulation torchaudio uses for its GPU spectrogram.
//! Training-only bits (dither, nb-augmentation, frame splicing) are skipped.

use candle_core::{Device, DType, Result, Tensor};

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

/// Window centered/padded to `n_fft`, as `torch.stft` does for `win_length < n_fft`.
fn pad_window_to(window: &[f32], n_fft: usize) -> Vec<f32> {
    if window.len() == n_fft {
        return window.to_vec();
    }
    let left = (n_fft - window.len()) / 2;
    let mut out = vec![0f32; n_fft];
    out[left..left + window.len()].copy_from_slice(window);
    out
}

/// DFT-basis `Conv1d` kernel `(2·freq, 1, n_fft)` realizing torch's onesided
/// `_fft_r2c` as a strided cross-correlation. Channel `k < freq` carries the real
/// filter `window[n]·cos(2πkn/N)`; channel `freq+k` the imag filter
/// `−window[n]·sin(2πkn/N)`, so convolving the windowed frame gives `Re`/`Im` of bin
/// `k`. Twiddles are computed in f64 and stored f32 (accurate basis, single-precision
/// storage — matching torch's f32 FFT), with the window folded in.
fn dft_conv_kernel(n_fft: usize, padded_window: &[f32], device: &Device) -> Result<Tensor> {
    let freq = n_fft / 2 + 1;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut k = vec![0f32; 2 * freq * n_fft];
    for kk in 0..freq {
        for nn in 0..n_fft {
            let ang = two_pi * kk as f64 * nn as f64 / n_fft as f64;
            let w = padded_window[nn] as f64;
            k[kk * n_fft + nn] = (w * ang.cos()) as f32;
            k[(freq + kk) * n_fft + nn] = (-(w * ang.sin())) as f32;
        }
    }
    Tensor::from_vec(k, (2 * freq, 1, n_fft), device)
}

pub struct FilterbankFeatures {
    cfg: MelConfig,
    window: Vec<f32>, // loaded (win_length,), padded to n_fft at use
    fb: Tensor,       // (nfilt, n_fft/2+1)
    /// DFT-basis Conv1d kernel `(2·freq, 1, n_fft)` realizing torch's `_fft_r2c`:
    /// channels `[0, freq)` are the real filters `window[n]·cos(2πkn/N)`, channels
    /// `[freq, 2·freq)` the imag filters `−window[n]·sin(2πkn/N)`.
    stft_kernel: Tensor,
    device: Device,
}

impl FilterbankFeatures {
    /// Computes the Hann window, slaney mel filterbank, and the STFT DFT-basis kernel
    /// (as the Python preprocessor does at init — they are not checkpoint tensors).
    pub fn new(cfg: MelConfig, device: &Device) -> Result<Self> {
        let window = hann(cfg.n_window_size);
        let freq = cfg.n_fft / 2 + 1;
        let fb_data = mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.nfilt);
        let fb = Tensor::from_vec(fb_data, (cfg.nfilt, freq), device)?;
        let padded_win = pad_window_to(&window, cfg.n_fft);
        let stft_kernel = dft_conv_kernel(cfg.n_fft, &padded_win, device)?;
        Ok(Self { cfg, window, fb, stft_kernel, device: device.clone() })
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

    /// PORT: `FilterbankFeatures.stft` (py L385-395) → `torch.stft`
    /// (`aten/src/ATen/native/SpectralOps.cpp::stft`), computed **natively in candle**
    /// so it runs on the model device (CPU or Metal) — no external FFT library.
    ///
    /// torch.stft is: center-pad (`pad_mode="constant"`) → frame (`as_strided`, stride
    /// `hop`) → `×window` → `_fft_r2c` (onesided) → transpose. The windowed
    /// real→complex DFT is realized here as a `Conv1d` against the precomputed
    /// [`Self::stft_kernel`] DFT basis at stride `hop` — cross-correlation, no kernel
    /// flip, so `out[k][t] = Σ_n sig[t·hop+n]·window[n]·cos/sin(2πkn/N)` is exactly
    /// `Re`/`Im` of the bin. `_fft_r2c` keeps the input precision (f32 → complex64 via
    /// `DFTI_SINGLE`/cuFFT-R2C), so this single-precision path matches torch's
    /// reference and, unlike rustfft, runs on the GPU.
    ///
    /// `y`: `(1, L)` real signal on the device → `(re, im)` each `(1, freq, T)`,
    /// `T = 1 + L/hop` (the `center=True` frame count).
    fn stft(&self, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.n_window_stride;
        let freq = n_fft / 2 + 1;
        let l = y.dim(1)?;
        // center=True, pad_mode="constant": n_fft/2 zeros each side of the signal.
        let xin = y.reshape((1, 1, l))?.pad_with_zeros(2, n_fft / 2, n_fft / 2)?;
        // _fft_r2c as a strided DFT-basis convolution → (1, 2·freq, T).
        let out = xin.conv1d(&self.stft_kernel, 0, hop, 1, 1)?;
        let re = out.narrow(1, 0, freq)?; // (1, freq, T)
        let im = out.narrow(1, freq, freq)?; // (1, freq, T)
        Ok((re, im))
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
    ///
    /// The whole featurizer — preemphasis, the `torch.stft` port, magnitude, mel,
    /// log, and per-feature normalization — runs in candle tensor ops **on the model
    /// device** (CPU or Metal) in f32, matching torch's reference (the Python wraps
    /// the STFT in `autocast(enabled=False)`, i.e. deliberate f32). There is no
    /// external FFT library and no host round-trip: the STFT is a `Conv1d` against the
    /// DFT basis (see [`Self::stft`]), so on Metal the front-end is GPU-resident like
    /// the rest of the model.
    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        let dev = &self.device;
        let hop = self.cfg.n_window_stride;
        // signal → (L,) f32 on the model device.
        let x = samples.flatten_all()?.to_dtype(DType::F32)?.to_device(dev)?;
        let l = x.dim(0)?;
        // NeMo `get_seq_len`: floor(L/hop) valid frames. torch.stft(center=True)
        // emits `1 + L/hop` frames, so the trailing frame is a pad column (masked below).
        let seq_len = self.get_seq_len(l);
        let t = 1 + l / hop;

        // preemphasis: y[0]=x[0]; y[i]=x[i]-preemph·x[i-1], in candle on the device.
        let x2 = x.reshape((1, l))?;
        let y = if self.cfg.preemph != 0.0 && l > 1 {
            let pre = self.cfg.preemph;
            let head = x2.narrow(1, 0, 1)?; // x[0]
            // x[1:] - preemph·x[:-1]  (scalar via affine; candle has f64·Tensor, not Tensor·f64)
            let tail = (x2.narrow(1, 1, l - 1)? - x2.narrow(1, 0, l - 1)?.affine(pre, 0.0)?)?;
            Tensor::cat(&[&head, &tail], 1)? // (1, L)
        } else {
            x2
        };

        // torch.stft (candle-native, on device) → (re, im) each (1, freq, T).
        let (re, im) = self.stft(&y)?;
        // magnitude^mag_power: |X|^p. mag_power==2 → re²+im² (guard==0 on the
        // inference path, use_grads=False); else sqrt(re²+im²)^mag_power.
        let p2 = (re.sqr()? + im.sqr()?)?; // (1, freq, T)
        let power = if self.cfg.mag_power == 2.0 {
            p2
        } else {
            p2.sqrt()?.powf(self.cfg.mag_power)?
        };
        let power = power.squeeze(0)?.contiguous()?; // (freq, T)

        // mel: (nfilt, freq) @ (freq, T) → (nfilt, T), f32 on device (autocast off).
        let mut mel = self.fb.matmul(&power)?;
        // log(x + guard) — guard from log_zero_guard_value_fn (log_zero_guard_type="add").
        // Bind the guard first: `mel + …` moves `mel`, so the `&mel` borrow must resolve before.
        let guard = self.log_zero_guard_value_fn(&mel);
        mel = (mel + guard)?.log()?;
        // per-feature normalization (ddof=1) over the valid frames only, applied
        // to all frames — faithful to normalize_batch's valid_mask.
        let mut mel = normalize_batch(&mel, seq_len, &NormalizeType::PerFeature)?;
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
/// where valid) / (count - 1))` (ddof=1 bias correction), then **`x_std =
/// x_std.masked_fill(x_std.isnan(), 0.0)`** and `x_std += CONSTANT` (1e-5),
/// returning `(x - x_mean) / x_std` broadcast over ALL time steps. Here the `valid`
/// frames are the leading `[0, valid)` columns (the trailing centred-STFT pad frame
/// is excluded from the statistics and is masked to 0 by the caller afterwards).
///
/// The NaN guard matters for a clip with a single valid frame: the ddof=1 variance
/// is then `0/0 → NaN`, which Python masks to 0 so `x_std == CONSTANT` and the
/// features stay finite. That NaN only ever arises at `valid <= 1` (for `valid > 1`
/// the variance is finite), so we avoid the `0/0` outright — `valid <= 1` ⇒ the
/// masked std is 0 — rather than relying on NaN propagation through `sqrt`.
///
/// All `normalize_type` branches are translated (single-clip form): `per_feature`
/// (the checkpoint config), `all_features`, `fixed_mean`/`fixed_std`, and the
/// pass-through `none`.
#[derive(Debug, Clone)]
pub enum NormalizeType {
    /// `"per_feature"` — per-mel-bin mean/std over valid time.
    PerFeature,
    /// `"all_features"` — a single mean/std over the whole valid clip (all bins×time).
    AllFeatures,
    /// `{"fixed_mean": …, "fixed_std": …}` — per-feature fixed stats (len == nfilt).
    Fixed { mean: Vec<f32>, std: Vec<f32> },
    /// any other `normalize_type` — identity (Python's `else: return x`).
    None,
}

fn normalize_batch(x: &Tensor, valid: usize, kind: &NormalizeType) -> Result<Tensor> {
    match kind {
        NormalizeType::PerFeature => {
            let xv = x.narrow(1, 0, valid)?;
            let mean = xv.mean_keepdim(1)?;
            // ddof=1 std over the valid frames. valid<=1 ⇒ the Python NaN-masked std
            // is 0, so std collapses to CONSTANT (no 0/0 NaN reaches the divide).
            let std_pre = if valid > 1 {
                (xv.broadcast_sub(&mean)?.sqr()?.sum_keepdim(1)? / (valid as f64 - 1.0))?.sqrt()?
            } else {
                mean.zeros_like()? // (nfilt, 1) — the NaN-masked-to-0 std
            };
            let std = (std_pre + CONSTANT)?;
            x.broadcast_sub(&mean)?.broadcast_div(&std)
        }
        NormalizeType::AllFeatures => {
            // Python: x_mean[i] = x[i,:,:len].mean(); x_std[i] = x[i,:,:len].std()
            // (one scalar per clip over ALL bins × valid time); x_std += CONSTANT.
            // ddof=1; Python does NOT NaN-mask this branch, so we mirror that.
            let xv = x.narrow(1, 0, valid)?; // (nfilt, valid)
            let n = (xv.dim(0)? * valid) as f64;
            let mean = xv.mean_all()?; // scalar
            let var = (xv.broadcast_sub(&mean)?.sqr()?.sum_all()? / (n - 1.0))?;
            let std = (var.sqrt()? + CONSTANT)?;
            x.broadcast_sub(&mean)?.broadcast_div(&std)
        }
        NormalizeType::Fixed { mean, std } => {
            // Python: (x - fixed_mean[:,None]) / fixed_std[:,None], per feature.
            let nfilt = x.dim(0)?;
            let mean = Tensor::from_vec(mean.clone(), (nfilt, 1), x.device())?.to_dtype(x.dtype())?;
            let std = Tensor::from_vec(std.clone(), (nfilt, 1), x.device())?.to_dtype(x.dtype())?;
            x.broadcast_sub(&mean)?.broadcast_div(&std)
        }
        NormalizeType::None => Ok(x.clone()),
    }
}

impl FilterbankFeatures {
    /// `filter_banks` — the `(nfilt, n_fft/2+1)` mel filterbank (Python
    /// `featurizer.filter_banks`).
    pub fn filter_banks(&self) -> &Tensor {
        &self.fb
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
/// `torch_windows` keys (py L40-47): the analysis windows the base
/// `AudioPreprocessor` knows about — `None` maps to `ones`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    Hann,
    Hamming,
    Blackman,
    Bartlett,
    Ones,
}

/// `torch.{hann,hamming,blackman,bartlett}_window(n)` / `torch.ones(n)` with the
/// torch STFT default `periodic=True` (denominator `n`, not `n-1`).
fn analysis_window(kind: WindowKind, n: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let nn = n.max(1) as f64;
    (0..n)
        .map(|i| {
            let x = i as f64;
            let w = match kind {
                WindowKind::Ones => 1.0,
                WindowKind::Hann => 0.5 - 0.5 * (2.0 * PI * x / nn).cos(),
                WindowKind::Hamming => 0.54 - 0.46 * (2.0 * PI * x / nn).cos(),
                WindowKind::Blackman => 0.42 - 0.5 * (2.0 * PI * x / nn).cos() + 0.08 * (4.0 * PI * x / nn).cos(),
                WindowKind::Bartlett => 1.0 - (2.0 * x / nn - 1.0).abs(),
            };
            w as f32
        })
        .collect()
}

/// `AudioPreprocessor(nn.Module, ABC)` (py L28) — the abstract base of the audio
/// front-end: it holds the STFT `win_length`/`hop_length` and the window-function
/// table. [`AudioToMelSpectrogramPreprocessor`] composes it (Rust composition for
/// the Python `super().__init__(...)` inheritance). The Python non-persistent
/// `dtype_sentinel_tensor` buffer has no field — candle's compute dtype is explicit.
pub struct AudioPreprocessor {
    /// `self.win_length` (py L37).
    pub win_length: usize,
    /// `self.hop_length` (py L38).
    pub hop_length: usize,
}

impl AudioPreprocessor {
    /// `AudioPreprocessor.__init__(win_length, hop_length)` (py L34-58).
    pub fn new(win_length: usize, hop_length: usize) -> Self {
        Self { win_length, hop_length }
    }

    /// `torch_windows[kind](win_length)` — the length-`win_length` analysis window.
    pub fn window(&self, kind: WindowKind) -> Vec<f32> {
        analysis_window(kind, self.win_length)
    }

    /// `AudioPreprocessor.forward` input guard (py L60-66): the base, under
    /// `@torch.no_grad()`, warns if the signal is not f32 and casts it to f32
    /// before delegating to the abstract `get_features`. This is the base half of
    /// the template; the concrete [`AudioToMelSpectrogramPreprocessor::forward`]
    /// runs `get_features` and casts the output back to the sentinel (f32) dtype.
    /// Inference here is already grad-free, so there is no `no_grad` to mirror.
    pub fn forward(&self, input_signal: &Tensor) -> Result<Tensor> {
        if input_signal.dtype() != DType::F32 {
            eprintln!(
                "AudioPreprocessor received an input signal of dtype {:?}, rather than f32; \
                 it runs in float32 and the input will be cast (mantissa loss is not recoverable).",
                input_signal.dtype()
            );
        }
        input_signal.to_dtype(DType::F32)
    }

    /// `AudioPreprocessor.get_features` (py L71) — the `@abstractmethod` feature
    /// extractor. The base has no featurizer; concrete preprocessors
    /// ([`AudioToMelSpectrogramPreprocessor::get_features`]) implement it. Calling
    /// it on the base bails, mirroring Python's `NotImplementedError` contract.
    pub fn get_features(&self, _input_signal: &Tensor, _length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        candle_core::bail!("AudioPreprocessor::get_features is abstract; use a concrete preprocessor")
    }
}

pub struct AudioToMelSpectrogramPreprocessor {
    featurizer: FilterbankFeatures,
    /// `super().__init__(n_window_size, n_window_stride)` — the base preprocessor.
    base: AudioPreprocessor,
}

impl AudioToMelSpectrogramPreprocessor {
    /// PORT: `AudioToMelSpectrogramPreprocessor.__init__` (py L152-227). The full
    /// Python ctor wires a long config into a `FilterbankFeatures`; here the
    /// `featurizer` is built separately (`FilterbankFeatures::new`) and injected,
    /// and the base [`AudioPreprocessor`] is `super().__init__(n_window_size,
    /// n_window_stride)` from its `MelConfig`.
    pub fn new(featurizer: FilterbankFeatures) -> Self {
        let cfg = featurizer.mel_config();
        let base = AudioPreprocessor::new(cfg.n_window_size, cfg.n_window_stride);
        Self { featurizer, base }
    }

    /// base `AudioPreprocessor.win_length` (py L37).
    pub fn win_length(&self) -> usize {
        self.base.win_length
    }

    /// base `AudioPreprocessor.hop_length` (py L38).
    pub fn hop_length(&self) -> usize {
        self.base.hop_length
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

impl AudioToMelSpectrogramPreprocessor {
    /// `get_features` → `self.featurizer(input_signal, length)`. The Rust mel
    /// featurizer returns the features; per-clip valid length is tracked by the
    /// caller (`ChatState`), so the length slot is `None` here.
    pub fn get_features(&self, input_signal: &Tensor, _length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        Ok((self.featurizer.forward(input_signal)?, None))
    }

    /// `AudioPreprocessor.forward(input_signal, length)` (py L60-68): the base
    /// applies its f32 input guard ([`AudioPreprocessor::forward`]), delegates to
    /// the abstract `get_features` (here the mel featurizer), then casts the
    /// features back to the `dtype_sentinel_tensor` dtype (f32).
    pub fn forward(&self, input_signal: &Tensor, length: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let guarded = self.base.forward(input_signal)?;
        let (signal, len) = self.get_features(&guarded, length)?;
        Ok((signal.to_dtype(DType::F32)?, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_one_frame_matches_python() {
        // REAL Python comparison, not a "finite" lock: the upstream
        // `normalize_batch(x, seq_len=1, "per_feature")` on CPU masks the 0/0 ddof=1
        // std to 0 (=> std == CONSTANT 1e-5). The buggy divide-by-(valid-1) yields NaN
        // here, so this DETECTS that regression. Golden = the actual Python output.
        let dev = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1.0f32, 5.0, -2.0, 0.5, 9.0, 0.0, -3.0, 2.0, 4.0, 1.0, 7.0, -1.0],
            (4, 3),
            &dev,
        )
        .unwrap();
        let got = normalize_batch(&x, 1, &NormalizeType::PerFeature).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // From: python -c "normalize_batch(x[None], tensor([1]), 'per_feature')"
        let want = [
            0.0f32, 400000.0, -300000.0, 0.0, 850000.0, -50000.0, 0.0, 500000.0, 700000.0, 0.0, 600000.0, -200000.0,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(g.is_finite(), "one-frame normalize produced non-finite: {got:?}");
            let rel = (g - w).abs() / w.abs().max(1.0);
            assert!(rel < 1e-4, "normalize one-frame diverges from Python: got {got:?} want {want:?}");
        }
    }

    #[test]
    fn normalize_two_frames_unchanged() {
        // valid>=2 keeps the ddof=1 path (regression guard for the one-frame branch).
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 4.0], (2, 2), &dev).unwrap();
        let out = normalize_batch(&x, 2, &NormalizeType::PerFeature).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // The newly-translated normalize branches, each vs actual Python output.
    fn x43() -> Tensor {
        Tensor::from_vec(
            vec![1.0f32, 5.0, -2.0, 0.5, 9.0, 0.0, -3.0, 2.0, 4.0, 1.0, 7.0, -1.0],
            (4, 3),
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn normalize_all_features_matches_python() {
        let got = normalize_batch(&x43(), 3, &NormalizeType::AllFeatures).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // normalize_batch(x[None], tensor([3]), "all_features")
        let want = [
            -0.26375f32, 0.8371, -1.08938, -0.40135, 1.93795, -0.53896, -1.3646, 0.01147, 0.56189, -0.26375, 1.38753,
            -0.81417,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "all_features vs Python: got {got:?} want {want:?}");
        }
    }

    #[test]
    fn normalize_fixed_matches_python() {
        let kind = NormalizeType::Fixed { mean: vec![0.0, 1.0, 2.0, 3.0], std: vec![1.0, 2.0, 4.0, 0.5] };
        let got = normalize_batch(&x43(), 3, &kind).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // normalize_batch(x[None], tensor([3]), {"fixed_mean":[0,1,2,3],"fixed_std":[1,2,4,.5]})
        let want = [1.0f32, 5.0, -2.0, -0.25, 4.0, -0.5, -1.25, 0.0, 0.5, -4.0, 8.0, -8.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "fixed vs Python: got {got:?} want {want:?}");
        }
    }

    #[test]
    fn normalize_none_is_identity() {
        let got = normalize_batch(&x43(), 3, &NormalizeType::None).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, x43().flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
}
