//! Port of `liquid_audio/processor.py` — `LFM2AudioProcessor` + `ChatState`.
//!
//! `LFM2AudioProcessor` bundles the text tokenizer (HF AutoTokenizer →
//! `tokenizers` crate), the mel audio frontend, and
//! the audio-out backend behind the [`AudioDetokenizer`](crate::audio_out)
//! trait (`decode`). `ChatState` builds the model inputs (text tokens, audio-in
//! mel, lengths, audio-out codes, modality flags) the way the Python usage
//! example does (`new_turn`/`add_text`/`add_audio`/`end_turn`/`append`).
//!
//! The audio-out backend is selected at load time (LFM2 detokenizer for LFM2.5
//! models, Mimi codec for v1) but the processor only knows the trait — pure
//! candle, no torch.

use std::path::Path;

use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use crate::audio_out::AudioDetokenizer;
use crate::utils::{mel2emb_len, LFMModality};

mod mel {
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
    //!
    //! Reuse checked (no candle primitive fits): candle-transformers' Whisper/Voxtral mel
    //! is a different convention — power spectrum, *precomputed* filters passed in, log10 +
    //! Whisper normalization, no preemphasis — and its `fft`/`dft`/`log_mel_spectrogram_w`
    //! helpers are module-private (only `pcm_to_mel` is `pub`). candle has no Hann window,
    //! slaney mel-filterbank, or STFT builtin. So this NeMo chain (slaney-computed filters,
    //! preemphasis, `mag_power`, `log(x+guard)`, per-feature norm) is necessarily local.
    //! Training-only bits (dither, nb-augmentation, frame splicing) are skipped.

    use candle_core::{DType, Device, Result, Tensor};

    /// Subset of NeMo's preprocessor config needed offline.
    #[derive(Debug, Clone)]
    pub struct MelConfig {
        pub sample_rate: usize,        // 16000
        pub n_window_size: usize,      // win_length (e.g. 400)
        pub n_window_stride: usize,    // hop_length (e.g. 160)
        pub n_fft: usize,              // e.g. 512
        pub nfilt: usize,              // mel bins (feat_in of the encoder)
        pub preemph: f64,              // 0.97
        pub log_zero_guard_value: f64, // 2^-24
        pub mag_power: f64,            // 2.0
        pub pad_to: usize,             // 16
        /// NeMo `exact_pad`. False (the checkpoint default) ⇒ `torch.stft(center=True)`
        /// (symmetric `n_fft//2` internal pad). True ⇒ `center=False` with the signal
        /// pre-padded by `(n_fft - hop)//2` each side in `forward`, so that the frame
        /// count equals `audio_length // hop`.
        pub exact_pad: bool,
    }

    impl MelConfig {
        /// NeMo `self.stft_pad_amount = (n_fft - hop_length) // 2 if exact_pad else None`.
        pub fn stft_pad_amount(&self) -> Option<usize> {
            if self.exact_pad {
                Some((self.n_fft - self.n_window_stride) / 2)
            } else {
                None
            }
        }
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
        let fft_freqs: Vec<f64> = (0..freq)
            .map(|k| k as f64 * sr as f64 / n_fft as f64)
            .collect();
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
        fb: Tensor, // (nfilt, n_fft/2+1)
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
            // The Hann window is folded into the DFT-basis kernel here (it multiplies each basis
            // function), so the STFT applies it via `stft_kernel`; the raw window vector is not
            // stored — it has no use after the kernel is built.
            let padded_win = pad_window_to(&window, cfg.n_fft);
            let stft_kernel = dft_conv_kernel(cfg.n_fft, &padded_win, device)?;
            Ok(Self {
                cfg,
                fb,
                stft_kernel,
                device: device.clone(),
            })
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
        /// `pad_amount = stft_pad_amount*2` when `exact_pad` (`stft_pad_amount` set),
        /// else `(n_fft // 2) * 2` — the centered path. For even `n_fft` the centered
        /// case collapses to `floor_divide(seq_len, hop)`; the exact_pad case gives
        /// `floor_divide(seq_len + 2·stft_pad_amount - n_fft, hop)`. Exposed publicly so
        /// callers (the data mapper) can use the featurizer-computed length.
        pub fn get_seq_len(&self, seq_len: usize) -> usize {
            // pad_amount: stft_pad_amount*2 (exact_pad) else (n_fft // 2) * 2 (centered).
            let pad_amount = match self.cfg.stft_pad_amount() {
                Some(p) => p * 2,
                None => (self.cfg.n_fft / 2) * 2,
            };
            // torch.floor_divide on non-negative ints == integer division. Guard the
            // (seq_len + pad_amount - n_fft) subtraction against underflow.
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
        /// `y`: `(1, L)` real signal on the device → `(re, im)` each `(1, freq, T)`.
        /// `center_pad` is the symmetric zero pad torch applies for `center=True`
        /// (`n_fft/2`); for `center=False` (the exact_pad path, where `forward` has
        /// already padded the signal) it is `0`. `T = 1 + (L + 2·center_pad - n_fft)/hop`.
        fn stft(&self, y: &Tensor, center_pad: usize) -> Result<(Tensor, Tensor)> {
            let n_fft = self.cfg.n_fft;
            let hop = self.cfg.n_window_stride;
            let freq = n_fft / 2 + 1;
            let l = y.dim(1)?;
            // pad_mode="constant": center_pad zeros each side (n_fft/2 for center=True, 0
            // for center=False).
            let xin = y
                .reshape((1, 1, l))?
                .pad_with_zeros(2, center_pad, center_pad)?;
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
            let n_fft = self.cfg.n_fft;
            // signal → (L,) f32 on the model device.
            let x = samples
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_device(dev)?;
            // `seq_len_time` in Python: the valid sample count = L for a single clip; it is
            // the preemphasis timemask boundary (NOT shifted by the exact_pad padding).
            let l = x.dim(0)?;
            let seq_len = self.get_seq_len(l); // valid frame count

            // exact_pad: F.pad(x, (stft_pad_amount, stft_pad_amount)) on the RAW signal
            // BEFORE preemph, then stft(center=False) (center_pad=0). Centered: no signal
            // pad — the n_fft/2 pad happens inside stft (center=True).
            let (center_pad, x_in) = match self.cfg.stft_pad_amount() {
                Some(p) => (0usize, x.reshape((1, l))?.pad_with_zeros(1, p, p)?), // (1, L+2p)
                None => (n_fft / 2, x.reshape((1, l))?),                          // (1, L)
            };
            let li = x_in.dim(1)?;

            // preemphasis: y[0]=x_in[0]; y[i]=x_in[i]-preemph·x_in[i-1] (over the padded
            // signal in the exact_pad case, matching Python).
            let y = if self.cfg.preemph != 0.0 && li > 1 {
                let pre = self.cfg.preemph;
                let head = x_in.narrow(1, 0, 1)?; // x_in[0]
                                                  // x_in[1:] - preemph·x_in[:-1] (scalar via affine; candle has f64·Tensor).
                let tail =
                    (x_in.narrow(1, 1, li - 1)? - x_in.narrow(1, 0, li - 1)?.affine(pre, 0.0)?)?;
                Tensor::cat(&[&head, &tail], 1)? // (1, li)
            } else {
                x_in
            };
            // masked_fill(~(arange(li) < seq_len_time), 0): zero positions >= L. Centered
            // path li == L ⇒ no-op; exact_pad zeros the [L, li) tail (left pad stays 0).
            let y = if li > l {
                let kept = y.narrow(1, 0, l)?;
                let zeros = Tensor::zeros((1, li - l), y.dtype(), dev)?;
                Tensor::cat(&[&kept, &zeros], 1)?
            } else {
                y
            };

            // torch.stft (candle-native, on device) → (re, im) each (1, freq, T).
            let (re, im) = self.stft(&y, center_pad)?;
            // frame count T (center=True ⇒ 1+L/hop; exact_pad ⇒ L/hop) — read from the
            // actual framing so both paths agree with their seq_len.
            let t = re.dim(2)?;
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
                    let padding = Tensor::zeros(
                        (self.cfg.nfilt, self.cfg.pad_to - rem),
                        mel.dtype(),
                        &self.device,
                    )?;
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
                    (xv.broadcast_sub(&mean)?.sqr()?.sum_keepdim(1)? / (valid as f64 - 1.0))?
                        .sqrt()?
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
                let mean =
                    Tensor::from_vec(mean.clone(), (nfilt, 1), x.device())?.to_dtype(x.dtype())?;
                let std =
                    Tensor::from_vec(std.clone(), (nfilt, 1), x.device())?.to_dtype(x.dtype())?;
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
                    WindowKind::Blackman => {
                        0.42 - 0.5 * (2.0 * PI * x / nn).cos() + 0.08 * (4.0 * PI * x / nn).cos()
                    }
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
            Self {
                win_length,
                hop_length,
            }
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
        pub fn get_features(
            &self,
            _input_signal: &Tensor,
            _length: Option<&Tensor>,
        ) -> Result<(Tensor, Option<Tensor>)> {
            candle_core::bail!(
                "AudioPreprocessor::get_features is abstract; use a concrete preprocessor"
            )
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
    }

    impl AudioToMelSpectrogramPreprocessor {
        /// `get_features` → `self.featurizer(input_signal, length)`. The Rust mel
        /// featurizer returns the features; per-clip valid length is tracked by the
        /// caller (`ChatState`), so the length slot is `None` here.
        pub fn get_features(
            &self,
            input_signal: &Tensor,
            _length: Option<&Tensor>,
        ) -> Result<(Tensor, Option<Tensor>)> {
            Ok((self.featurizer.forward(input_signal)?, None))
        }

        /// `AudioPreprocessor.forward(input_signal, length)` (py L60-68): the base
        /// applies its f32 input guard ([`AudioPreprocessor::forward`]), delegates to
        /// the abstract `get_features` (here the mel featurizer), then casts the
        /// features back to the `dtype_sentinel_tensor` dtype (f32).
        pub fn forward(
            &self,
            input_signal: &Tensor,
            length: Option<&Tensor>,
        ) -> Result<(Tensor, Option<Tensor>)> {
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
                vec![
                    1.0f32, 5.0, -2.0, 0.5, 9.0, 0.0, -3.0, 2.0, 4.0, 1.0, 7.0, -1.0,
                ],
                (4, 3),
                &dev,
            )
            .unwrap();
            let got = normalize_batch(&x, 1, &NormalizeType::PerFeature)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // From: python -c "normalize_batch(x[None], tensor([1]), 'per_feature')"
            let want = [
                0.0f32, 400000.0, -300000.0, 0.0, 850000.0, -50000.0, 0.0, 500000.0, 700000.0, 0.0,
                600000.0, -200000.0,
            ];
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    g.is_finite(),
                    "one-frame normalize produced non-finite: {got:?}"
                );
                let rel = (g - w).abs() / w.abs().max(1.0);
                assert!(
                    rel < 1e-4,
                    "normalize one-frame diverges from Python: got {got:?} want {want:?}"
                );
            }
        }

        #[test]
        fn normalize_two_frames_unchanged() {
            // valid>=2 keeps the ddof=1 path (regression guard for the one-frame branch).
            let dev = Device::Cpu;
            let x = Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 4.0], (2, 2), &dev).unwrap();
            let out = normalize_batch(&x, 2, &NormalizeType::PerFeature)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(out.iter().all(|v| v.is_finite()));
        }

        // The newly-translated normalize branches, each vs actual Python output.
        fn x43() -> Tensor {
            Tensor::from_vec(
                vec![
                    1.0f32, 5.0, -2.0, 0.5, 9.0, 0.0, -3.0, 2.0, 4.0, 1.0, 7.0, -1.0,
                ],
                (4, 3),
                &Device::Cpu,
            )
            .unwrap()
        }

        #[test]
        fn normalize_all_features_matches_python() {
            let got = normalize_batch(&x43(), 3, &NormalizeType::AllFeatures)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // normalize_batch(x[None], tensor([3]), "all_features")
            let want = [
                -0.26375f32,
                0.8371,
                -1.08938,
                -0.40135,
                1.93795,
                -0.53896,
                -1.3646,
                0.01147,
                0.56189,
                -0.26375,
                1.38753,
                -0.81417,
            ];
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < 1e-4,
                    "all_features vs Python: got {got:?} want {want:?}"
                );
            }
        }

        #[test]
        fn normalize_fixed_matches_python() {
            let kind = NormalizeType::Fixed {
                mean: vec![0.0, 1.0, 2.0, 3.0],
                std: vec![1.0, 2.0, 4.0, 0.5],
            };
            let got = normalize_batch(&x43(), 3, &kind)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // normalize_batch(x[None], tensor([3]), {"fixed_mean":[0,1,2,3],"fixed_std":[1,2,4,.5]})
            let want = [
                1.0f32, 5.0, -2.0, -0.25, 4.0, -0.5, -1.25, 0.0, 0.5, -4.0, 8.0, -8.0,
            ];
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < 1e-4,
                    "fixed vs Python: got {got:?} want {want:?}"
                );
            }
        }

        #[test]
        fn normalize_none_is_identity() {
            let got = normalize_batch(&x43(), 3, &NormalizeType::None)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got, x43().flatten_all().unwrap().to_vec1::<f32>().unwrap());
        }
    }
}

pub use mel::{
    AudioPreprocessor, AudioToMelSpectrogramPreprocessor, FilterbankFeatures, MelConfig,
    NormalizeType, WindowKind,
};

/// Matches the `preprocessor` block of the model's config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PreprocessorConfig {
    pub sample_rate: usize,
    pub normalize: String,
    pub window_size: f64,
    pub window_stride: f64,
    pub window: String,
    pub features: usize,
    pub n_fft: usize,
    pub log: bool,
    pub frame_splicing: usize,
    pub dither: f64,
    pub pad_to: usize,
    pub pad_value: f64,
    /// NeMo `exact_pad` (constructor arg, not always in the checkpoint JSON; the
    /// LFM2.5-Audio config omits it ⇒ False). True switches the STFT to `center=False`
    /// with an explicit `(n_fft - hop)//2` signal pad.
    #[serde(default)]
    pub exact_pad: bool,
}

impl PreprocessorConfig {
    /// Used by the model builder (lfm2_audio) to construct the featurizer.
    pub fn mel_config(&self) -> MelConfig {
        MelConfig {
            sample_rate: self.sample_rate,
            n_window_size: (self.window_size * self.sample_rate as f64).round() as usize,
            n_window_stride: (self.window_stride * self.sample_rate as f64).round() as usize,
            n_fft: self.n_fft,
            nfilt: self.features,
            preemph: 0.97, // FilterbankFeatures default
            log_zero_guard_value: 2f64.powi(-24),
            mag_power: 2.0,
            pad_to: self.pad_to,
            exact_pad: self.exact_pad,
        }
    }
}

/// The turn-format strings `ChatState` writes — single source of truth, shared with
/// [`crate::chat_template`]'s load-time verification against the snapshot's own
/// `chat_template.jinja`. Faithful to the Python `ChatState` (`new_turn`/`end_turn`).
pub const SEQUENCE_START: &str = "<|startoftext|>";
pub const TURN_FOOTER: &str = "<|im_end|>\n";
pub fn turn_header(role: &str) -> String {
    format!("<|im_start|>{role}\n")
}

/// Generation-control token ids resolved BY NAME from the model's own tokenizer at
/// load time — the model defines them, so they pass through instead of living as
/// literals in the generation loops (`<|im_end|>` also cross-checks the config's
/// `lfm.eos_token_id`). Resolution is hard-error: a snapshot whose tokenizer lacks
/// these names is not an LFM2-Audio model, and guessing ids would generate garbage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    /// `<|im_end|>` — closes a turn; text sampling stops here.
    pub im_end: u32,
    /// `<|text_end|>` — the interleaved text channel is finished for this turn.
    pub text_end: u32,
    /// `<|audio_start|>` — sequential (TTS) generation flips to audio frames here.
    pub audio_start: u32,
}

impl SpecialTokenIds {
    pub fn resolve(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |name: &str| -> Result<u32> {
            let id = tokenizer.token_to_id(name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "tokenizer does not define {name} — not an LFM2-Audio tokenizer"
                ))
            })?;
            // The grammar only works if the fence string ENCODES back to this single
            // id — i.e. the tokenizer matches it as an added special token rather
            // than char-splitting it into ordinary text. Everything downstream
            // (turn boundaries, end-of-turn detection, the chat template) rides on
            // this round-trip, so a tokenizer that fails it must fail the load.
            let enc = tokenizer
                .encode(name, false)
                .map_err(|e| candle_core::Error::Msg(format!("tokenizer encode {name}: {e}")))?;
            if enc.get_ids() != [id] {
                return Err(candle_core::Error::Msg(format!(
                    "tokenizer does not round-trip {name}: encodes to {:?}, expected [{id}] \
                     — grammar fences would enter context as plain text",
                    enc.get_ids()
                )));
            }
            Ok(id)
        };
        // <|im_start|> is not generation-control (never sampled against) but it IS
        // the other half of every fence — same round-trip requirement.
        let _ = id("<|im_start|>")?;
        Ok(Self {
            im_end: id("<|im_end|>")?,
            text_end: id("<|text_end|>")?,
            audio_start: id("<|audio_start|>")?,
        })
    }
}

pub struct LFM2AudioProcessor {
    pub tokenizer: Tokenizer,
    pub audio: FilterbankFeatures,
    /// Audio-out DECODE backend: the LFM2 detokenizer (LFM2.5 snapshots ship
    /// `audio_detokenizer/`). `None` for v1 models, where `decode` falls back to
    /// [`Self::mimi`]. Behind the trait so the processor never touches a concrete
    /// codec type. Mirrors the Python processor's `_audio_detokenizer` field.
    pub audio_out: Option<Box<dyn AudioDetokenizer>>,
    /// The Mimi codec (`tokenizer-…checkpoint125.safetensors`), loaded INDEPENDENTLY
    /// of `audio_out`. Mirrors the Python processor's separate `_mimi` field: the
    /// data mapper's `_encode_audio_out` calls `processor.mimi.encode` even on full
    /// LFM2.5 snapshots — where the decode backend is the LFM2 detokenizer, not Mimi.
    /// Conflating the two (one shared field) loses the encoder on full snapshots.
    pub mimi: Option<Box<dyn AudioDetokenizer>>,
    pub device: Device,
}

impl LFM2AudioProcessor {
    /// Build from a local model directory: `tokenizer.json` + the mel buffers
    /// (`window`/`fb`) under a VarBuilder rooted at the audio preprocessor.
    ///
    /// `audio_out` is the LFM2 detokenizer (decode; `None` for v1); `mimi` is the
    /// Mimi codec (encode + v1 decode fallback). They are separate so a full
    /// snapshot keeps both — Python loads `_audio_detokenizer` and `_mimi`
    /// independently.
    pub fn new(
        tokenizer: Tokenizer,
        audio: FilterbankFeatures,
        audio_out: Option<Box<dyn AudioDetokenizer>>,
        mimi: Option<Box<dyn AudioDetokenizer>>,
        device: Device,
    ) -> Self {
        Self {
            tokenizer,
            audio,
            audio_out,
            mimi,
            device,
        }
    }

    /// PORT: `LFM2AudioProcessor.from_pretrained(repo_id, *, device)` (py 56).
    ///
    /// The Python classmethod resolves the model dir (`get_model_dir`), reads
    /// `config.json`, and constructs the processor (tokenizer + mel featurizer +
    /// audio-out backend) on `device`. The crate's loader already performs that
    /// exact construction inside [`crate::loader::from_pretrained`] (it builds both
    /// the model and the processor in one pass over the checkpoint). To avoid
    /// duplicating the loader logic this delegates to it and returns just the
    /// processor — the model is dropped here (the Python classmethod likewise only
    /// returns the processor).
    pub fn from_pretrained(dir: &Path, device: &Device) -> Result<Self> {
        let (_model, processor) = crate::loader::from_pretrained(dir, device)?;
        Ok(processor)
    }

    pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
        Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer: {e}")))
    }

    /// Resolve the generation-control token ids from THIS model's tokenizer.
    /// See [`SpecialTokenIds`].
    pub fn special_token_ids(tokenizer: &Tokenizer) -> Result<SpecialTokenIds> {
        SpecialTokenIds::resolve(tokenizer)
    }

    /// Encode text without auto special tokens → token id row `(1, n)`.
    ///
    /// I64 (torch.long): Python `text.encode(..., return_tensors="pt")` yields a long
    /// tensor, and every downstream id field (`audio_out` = `text.new_empty`,
    /// `modality_flag` = `full_like(text)`) inherits it. candle's index_select/embedding
    /// accept I64, so there is no reason to narrow to U32.
    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        let n = ids.len();
        Tensor::from_vec(ids, (1, n), &self.device)
    }

    /// Detokenize audio codes `(1, codebooks, T)` → 24 kHz waveform via whichever
    /// audio-out backend was selected at load (LFM2 detokenizer or Mimi). The
    /// processor dispatches through the [`AudioDetokenizer`](crate::audio_out)
    /// trait — it doesn't know which concrete backend it holds.
    pub fn decode(&self, audio_codes: &Tensor) -> Result<Tensor> {
        // Python guard: reject codes outside [0, 2047] before detokenizing (the
        // EOAudio sentinel 2048 must be stripped by the caller). u32 ⇒ ≥0 already;
        // check the upper bound rather than index OOB in the codebook embedding.
        let max_code = audio_codes
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .max(0)?
            .to_scalar::<u32>()?;
        if max_code > 2047 {
            return Err(candle_core::Error::Msg(format!(
                "audio code {max_code} out of range [0, 2047] (strip the EOAudio frame before decode)"
            )));
        }
        // Decode through the LFM2 detokenizer when present (LFM2.5), else the Mimi
        // codec (v1). Python's `decode` uses `audio_detokenizer`; v1 models ship no
        // detokenizer and detokenize via `mimi` directly — the port unifies both
        // behind `decode` via this `audio_out → mimi` fallback.
        self.audio_out
            .as_ref()
            .or(self.mimi.as_ref())
            .ok_or_else(|| candle_core::Error::Msg("no audio-out backend loaded".into()))?
            .decode(audio_codes)
    }
}

/// `ChatState` — accumulates model inputs across turns. Mirrors the Python
/// fields; `**chat` unpacking becomes direct field access in `generate_*`.
pub struct ChatState<'a> {
    proc: &'a LFM2AudioProcessor,
    codebooks: usize,
    pub text: Tensor,          // (1, n) i64 token ids (torch.long)
    pub audio_in: Tensor,      // (nfilt, total_frames) f32 mel
    pub audio_in_lens: Tensor, // (k,) i64 (torch.long)
    pub audio_out: Tensor,     // (codebooks, m) i64 (torch.long)
    pub modality_flag: Tensor, // (1, n) i64 (LFMModality; torch.long)
}

impl<'a> ChatState<'a> {
    pub fn new(proc: &'a LFM2AudioProcessor, codebooks: usize) -> Result<Self> {
        let dev = &proc.device;
        let text = proc.encode(SEQUENCE_START)?;
        let n = text.dim(1)?;
        let nfilt = proc.audio.nfilt();
        let modality_flag = Tensor::from_vec(vec![LFMModality::Text as i64; n], (1, n), dev)?;
        Ok(Self {
            proc,
            codebooks,
            text,
            // Empty placeholders as zero-length VIEWS of a 1-element buffer. candle
            // can't allocate a zero-size buffer on Metal, so a bare `zeros((nfilt,0))`
            // fails on GPU; a valid 1-col buffer narrowed to length 0 reports 0
            // elements, is read only via `dim()` while empty, and is replaced (not
            // cat'd) on the first add — so no zero-size buffer is ever created.
            audio_in: Tensor::zeros((nfilt, 1), candle_core::DType::F32, dev)?.narrow(1, 0, 0)?,
            audio_in_lens: Tensor::zeros((1,), candle_core::DType::I64, dev)?.narrow(0, 0, 0)?,
            audio_out: Tensor::zeros((codebooks, 1), candle_core::DType::I64, dev)?
                .narrow(1, 0, 0)?,
            modality_flag,
        })
    }

    /// Seed a `ChatState` from a previously persisted conversation (the five model-input
    /// fields) instead of `new`'s fresh `<|startoftext|>` start.
    ///
    /// `ChatState<'a>` borrows the processor, so it cannot itself be held across turns by an
    /// owner that also owns the processor (self-referential). The realtime engine instead
    /// holds the accumulated *tensors* (`Lfm2VoiceEngine::conv`) and rebuilds a transient
    /// `ChatState` each turn via this constructor — the Rust analog of Python keeping ONE
    /// persistent `ChatState` object across `append`/`new_turn` calls (README getting-started,
    /// the two-turn example). No `<|startoftext|>` is prepended: the persisted `text` already
    /// begins with it from the first turn. Fields map 1:1 to `new`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        proc: &'a LFM2AudioProcessor,
        codebooks: usize,
        text: Tensor,
        audio_in: Tensor,
        audio_in_lens: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
    ) -> Result<Self> {
        // The persisted audio_out is restored from a prior turn's `append`; guard the codebook
        // row count before it reaches the prefill `audio_out` scatter (which asserts on it).
        if audio_out.dim(0)? != codebooks {
            return Err(candle_core::Error::Msg(format!(
                "from_parts: audio_out must have {codebooks} codebook rows, got {}",
                audio_out.dim(0)?
            )));
        }
        Ok(Self {
            proc,
            codebooks,
            text,
            audio_in,
            audio_in_lens,
            audio_out,
            modality_flag,
        })
    }

    pub fn add_text(&mut self, text: &str) -> Result<()> {
        let new_text = self.proc.encode(text)?;
        let n = new_text.dim(1)?;
        let new_mod =
            Tensor::from_vec(vec![LFMModality::Text as i64; n], (1, n), &self.proc.device)?;
        self.text = Tensor::cat(&[&self.text, &new_text], 1)?;
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        Ok(())
    }

    /// `wave`: (1, L) at `sampling_rate`. Resampling to 16 kHz is the caller's
    /// responsibility (kept out of the core port); pass 16 kHz mono.
    pub fn add_audio_16k(&mut self, wave: &Tensor) -> Result<()> {
        let mel = self.proc.audio.forward(wave)?;
        let samples = wave.flatten_all()?.dim(0)?;
        let padded = mel.dim(2)?;
        let frames = self.proc.audio.get_seq_len(samples).min(padded);
        let new_audio_in = mel.i(0)?.narrow(1, 0, frames)?.contiguous()?; // (nfilt, valid_frames)
        let emb_len = mel2emb_len(frames as i64) as usize;
        let new_mod = Tensor::from_vec(
            vec![LFMModality::AudioIn as i64; emb_len],
            (1, emb_len),
            &self.proc.device,
        )?;
        let new_len = Tensor::from_vec(vec![frames as i64], (1,), &self.proc.device)?;
        // Replace the empty placeholder on the first add (avoids cat-ing a
        // zero-length Metal view); otherwise append.
        self.audio_in = if self.audio_in.dim(1)? == 0 {
            new_audio_in
        } else {
            Tensor::cat(&[&self.audio_in, &new_audio_in], 1)?
        };
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &new_mod], 1)?;
        self.audio_in_lens = if self.audio_in_lens.dim(0)? == 0 {
            new_len
        } else {
            Tensor::cat(&[&self.audio_in_lens, &new_len], 0)?
        };
        Ok(())
    }

    /// PORT: `ChatState.add_audio(wave, sampling_rate)` (py 226).
    ///
    /// Faithful port of the full Python method: assert `wave` is `(1, L)`,
    /// resample from `sampling_rate` to 16 kHz (Python:
    /// `torchaudio.functional.resample(wave, sampling_rate, 16_000)`), run the mel
    /// front-end, then append the new audio-in mel, its `AUDIO_IN` modality flags
    /// (one per `mel2emb_len(frames)`), and the frame length — exactly the same
    /// three `torch.cat`s as Python (py 248-250).
    ///
    /// The resample is the faithful windowed-sinc [`crate::resample`] (a 1:1 port
    /// of `torchaudio.functional.resample`, shared with `data::mapper`),
    /// `L' = ceil(L * 16000 / sampling_rate)`. The post-resample mel/append path
    /// delegates to [`Self::add_audio_16k`] so the parity computation is shared
    /// and unchanged.
    pub fn add_audio(&mut self, wave: &Tensor, sampling_rate: u32) -> Result<()> {
        // Python: `assert len(wave.shape) == 2` and `assert wave.shape[0] == 1`.
        if sampling_rate == 0 {
            return Err(candle_core::Error::Msg(
                "add_audio: sampling_rate must be non-zero".into(),
            ));
        }
        if wave.rank() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "add_audio: wave must be 2-D (1, L), got rank {}",
                wave.rank()
            )));
        }
        if wave.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(format!(
                "add_audio: wave must have 1 channel, got {}",
                wave.dim(0)?
            )));
        }

        // Python: `wave = torchaudio.functional.resample(wave, sampling_rate, 16_000)`.
        let wave16 = Self::resample_16k(wave, sampling_rate)?;
        self.add_audio_16k(&wave16)
    }

    /// `torchaudio.functional.resample(wave, orig, 16_000)` — the faithful
    /// windowed-sinc resampler (default `sinc_interp_hann`, width 6, rolloff 0.99),
    /// shared with `data::mapper`. `wave` is `(1, L)` → `(1, L')` f32 with
    /// `L' = ceil(L * 16000 / orig)`. See [`crate::resample`] (1:1 torchaudio port).
    fn resample_16k(wave: &Tensor, orig: u32) -> Result<Tensor> {
        crate::resample::resample(wave, orig, 16_000)
    }

    pub fn new_turn(&mut self, role: &str) -> Result<()> {
        self.add_text(&turn_header(role))
    }

    pub fn end_turn(&mut self) -> Result<()> {
        self.add_text(TURN_FOOTER)
    }

    /// Render the context as a human-readable transcript: the text channel decoded
    /// with special tokens VISIBLE (the turn grammar is the point), and contiguous
    /// audio runs shown as `⟨audio-in ×N⟩` / `⟨audio-out ×N⟩` placeholders at their
    /// sequence positions. This is exactly the sequence the model attends over —
    /// the debug answer to "are the role fences actually in context?".
    pub fn transcript(&self) -> Result<String> {
        let flags: Vec<i64> = self
            .modality_flag
            .to_dtype(candle_core::DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?;
        let ids: Vec<i64> = self.text.flatten_all()?.to_vec1::<i64>()?;
        let mut out = String::new();
        let mut text_run: Vec<u32> = Vec::new();
        let mut audio_run: Option<(i64, usize)> = None; // (modality, len)
        let mut ti = 0usize;

        let flush_text = |out: &mut String, run: &mut Vec<u32>| -> Result<()> {
            if run.is_empty() {
                return Ok(());
            }
            let s = self
                .proc
                .text()
                .decode(run, false)
                .map_err(|e| candle_core::Error::Msg(format!("transcript decode: {e}")))?;
            out.push_str(&s);
            run.clear();
            Ok(())
        };
        let flush_audio = |out: &mut String, run: &mut Option<(i64, usize)>| {
            if let Some((m, n)) = run.take() {
                let name = if m == LFMModality::AudioIn as i64 {
                    "audio-in"
                } else {
                    "audio-out"
                };
                out.push_str(&format!("⟨{name} ×{n}⟩"));
            }
        };

        for &flag in &flags {
            if flag == LFMModality::Text as i64 {
                flush_audio(&mut out, &mut audio_run);
                text_run.push(ids[ti] as u32);
                ti += 1;
            } else {
                flush_text(&mut out, &mut text_run)?;
                audio_run = match audio_run {
                    Some((m, n)) if m == flag => Some((m, n + 1)),
                    Some(_) => {
                        flush_audio(&mut out, &mut audio_run);
                        Some((flag, 1))
                    }
                    None => Some((flag, 1)),
                };
            }
        }
        flush_text(&mut out, &mut text_run)?;
        flush_audio(&mut out, &mut audio_run);
        Ok(out)
    }

    /// Append generated text + audio-out tokens with their modality flags.
    ///
    /// Mirrors the Python `ChatState.append` invariants: `text` is one row,
    /// `audio_out` has `codebooks` rows, `modality_flag` is one row, and the flag
    /// count equals `text_len + audio_out_len` (the scatter depends on it).
    pub fn append(
        &mut self,
        text: &Tensor,
        audio_out: &Tensor,
        modality_flag: &Tensor,
    ) -> Result<()> {
        let mf = if modality_flag.rank() == 1 {
            modality_flag.unsqueeze(0)?
        } else {
            modality_flag.clone()
        };
        if text.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(format!(
                "append: text must be 1 row, got {}",
                text.dim(0)?
            )));
        }
        if audio_out.dim(0)? != self.codebooks {
            return Err(candle_core::Error::Msg(format!(
                "append: audio_out must have {} codebook rows, got {}",
                self.codebooks,
                audio_out.dim(0)?
            )));
        }
        if mf.dim(0)? != 1 {
            return Err(candle_core::Error::Msg(
                "append: modality_flag must be 1 row".into(),
            ));
        }
        let (n_text, n_audio, n_flag) = (text.dim(1)?, audio_out.dim(1)?, mf.dim(1)?);
        if n_flag != n_text + n_audio {
            return Err(candle_core::Error::Msg(format!(
                "append: modality_flag len {n_flag} != text {n_text} + audio_out {n_audio}"
            )));
        }
        // The state carries I64 (torch.long); cast the incoming ids to match (the
        // generation loop hands back U32 sampled tokens). Faithful — torch keeps long.
        let i64t = candle_core::DType::I64;
        let (text, audio_out, mf) = (
            text.to_dtype(i64t)?,
            audio_out.to_dtype(i64t)?,
            mf.to_dtype(i64t)?,
        );
        self.text = Tensor::cat(&[&self.text, &text], 1)?;
        // Replace the empty placeholder on the first append (Metal: no zero-len cat).
        self.audio_out = if self.audio_out.dim(1)? == 0 {
            audio_out
        } else {
            Tensor::cat(&[&self.audio_out, &audio_out], 1)?
        };
        self.modality_flag = Tensor::cat(&[&self.modality_flag, &mf], 1)?;
        Ok(())
    }
}

use candle_core::IndexOp;

impl LFM2AudioProcessor {
    /// `text` → the text tokenizer.
    pub fn text(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// `audio` → the mel audio preprocessor (Python `AudioToMelSpectrogramPreprocessor`;
    /// the port's featurizer).
    pub fn audio(&self) -> &FilterbankFeatures {
        &self.audio
    }

    /// `device` → the device tensors live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `audio_detokenizer` → the audio-out DECODE backend: the LFM2 detokenizer
    /// (LFM2.5) when present, else the Mimi codec (v1). Python's `audio_detokenizer`
    /// property is strictly the LFM2 detokenizer; the port folds the v1 Mimi-decode
    /// path in here so callers (`decode`, streaming `decode_step` in `mic_chat`) get
    /// a working backend on both model families.
    pub fn audio_detokenizer(&self) -> Option<&dyn AudioDetokenizer> {
        self.audio_out.as_deref().or(self.mimi.as_deref())
    }

    /// `mimi` → the Mimi CODEC, independent of the decode backend (Python `_mimi`).
    /// This is what the data mapper's `_encode_audio_out` uses (`processor.mimi.encode`),
    /// so on a full LFM2.5 snapshot it must return Mimi — NOT the LFM2 detokenizer
    /// that `audio_detokenizer`/`decode` use.
    pub fn mimi(&self) -> Option<&dyn AudioDetokenizer> {
        self.mimi.as_deref()
    }

    /// `mimi.sample_rate` — the Mimi codec's expected input sample rate. Used by the
    /// data mapper (`_encode_audio_out`) to resample before encoding. Reads the Mimi
    /// codec (not the decode backend): on a full snapshot the LFM2 detokenizer's rate
    /// is irrelevant to the encode path.
    pub fn mimi_sample_rate(&self) -> Option<u32> {
        self.mimi.as_deref().map(|d| d.sample_rate())
    }

    /// `mimi.encode(wav)` — encode a `(B, 1, L)` waveform to codes via the Mimi codec
    /// (errors if no Mimi checkpoint was loaded). Always the Mimi codec, never the
    /// LFM2 detokenizer (which is decode-only).
    pub fn mimi_encode(&self, wav: &Tensor) -> Result<Tensor> {
        self.mimi
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "no Mimi codec loaded (encode needs the Mimi tokenizer checkpoint `tokenizer-…checkpoint125.safetensors`)".into(),
                )
            })?
            .encode(wav)
    }
}

impl ChatState<'_> {
    /// `model_inputs` — the model-input field names (Python `model_inputs`).
    pub fn model_inputs(&self) -> [&'static str; 5] {
        [
            "text",
            "audio_in",
            "audio_in_lens",
            "audio_out",
            "modality_flag",
        ]
    }

    /// `__len__` → number of model-input fields.
    pub fn len(&self) -> usize {
        self.model_inputs().len()
    }

    /// The model-input field set is fixed and non-empty (kept for the
    /// `len`-without-`is_empty` lint).
    pub fn is_empty(&self) -> bool {
        self.model_inputs().is_empty()
    }

    /// `__iter__` → iterate the model-input field names.
    pub fn iter(&self) -> impl Iterator<Item = &'static str> {
        self.model_inputs().into_iter()
    }

    /// `__getitem__(name)` → the tensor field by model-input name.
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        match name {
            "text" => Ok(&self.text),
            "audio_in" => Ok(&self.audio_in),
            "audio_in_lens" => Ok(&self.audio_in_lens),
            "audio_out" => Ok(&self.audio_out),
            "modality_flag" => Ok(&self.modality_flag),
            other => Err(candle_core::Error::Msg(format!(
                "expected one of {:?}, got {other}.",
                [
                    "text",
                    "audio_in",
                    "audio_in_lens",
                    "audio_out",
                    "modality_flag"
                ]
            ))),
        }
    }

    /// `device` → the processor's device.
    pub fn device(&self) -> &Device {
        &self.proc.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;

    fn test_processor() -> LFM2AudioProcessor {
        let dev = Device::Cpu;
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<|startoftext|>".to_string(), 1);
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );
        let audio = FilterbankFeatures::new(
            MelConfig {
                sample_rate: 16_000,
                n_window_size: 400,
                n_window_stride: 160,
                n_fft: 512,
                nfilt: 8,
                preemph: 0.97,
                log_zero_guard_value: 2f64.powi(-24),
                mag_power: 2.0,
                pad_to: 16,
                exact_pad: false,
            },
            &dev,
        )
        .unwrap();
        LFM2AudioProcessor::new(tokenizer, audio, None, None, dev)
    }

    #[test]
    fn add_audio_16k_uses_valid_mel_length_not_center_padding() {
        let proc = test_processor();
        let wave = Tensor::zeros((1, 1280), DType::F32, proc.device()).unwrap();
        let raw = proc.audio().forward(&wave).unwrap();
        let valid = proc.audio().get_seq_len(1280);

        // STFT with center=True: center_pad = n_fft/2 = 256, padded_len = 1280+512=1792,
        // T = (1792-512)/160+1 = 9. Then pad_to=16 pads time to the next multiple → 16.
        assert_eq!(raw.dim(2).unwrap(), 16); // 9 STFT frames, padded to 16 by pad_to
        assert_eq!(valid, 8);
        assert_eq!(mel2emb_len(raw.dim(2).unwrap() as i64), 2);
        assert_eq!(mel2emb_len(valid as i64), 1);

        // add_audio_16k narrows to the valid frame count (not the padded count),
        // so audio_in carries only the real frames.
        let mut chat = ChatState::new(&proc, 8).unwrap();
        chat.add_audio_16k(&wave).unwrap();

        assert_eq!(chat.audio_in.dim(1).unwrap(), valid);
        assert_eq!(chat.audio_in_lens.to_vec1::<i64>().unwrap(), vec![8]);
        let flags = chat.modality_flag.to_vec2::<i64>().unwrap();
        let audio_flags = flags[0]
            .iter()
            .filter(|&&flag| flag == LFMModality::AudioIn as i64)
            .count();
        assert_eq!(audio_flags, mel2emb_len(valid as i64) as usize);
    }

    #[test]
    fn add_audio_rejects_zero_sampling_rate() {
        let proc = test_processor();
        let wave = Tensor::zeros((1, 320), DType::F32, proc.device()).unwrap();
        let mut chat = ChatState::new(&proc, 8).unwrap();
        let err = chat.add_audio(&wave, 0).unwrap_err().to_string();
        assert!(err.contains("sampling_rate must be non-zero"), "{err}");
    }
}

impl std::fmt::Display for ChatState<'_> {
    /// `__repr__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ChatState(text_tok: {}, audio_in: {}, audio_out: {})",
            self.text.dim(1).unwrap_or(0),
            self.audio_in.dim(1).unwrap_or(0),
            self.audio_out.dim(1).unwrap_or(0),
        )
    }
}
