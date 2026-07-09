//! Port of `liquid_audio/detokenizer.py` — the LFM2.5 custom audio detokenizer.
//!
//! `FusedEmbedding` (codes → embeddings) → ×6 nearest upsample → `Lfm2Model`
//! backbone under a sliding-window causal mask → Linear(512→1282) → split into
//! log-magnitude + angle → polar → Vocos-style `ISTFT` → 24 kHz waveform.
//!
//! The ISTFT runs **natively in candle on the model device** (CPU or Metal), in f32 —
//! a port of `torch.fft.irfft` (MPS = `MPSGraph HermiteanToRealFFTWithTensor`,
//! f32-only) followed by the Vocos/torchaudio overlap-add. The inverse real FFT is the
//! inverse-DFT basis matmul (`y = Re·Cw + Im·Sw`, run through candle's tiled `matmul`),
//! and the windowed overlap-add + window-envelope normalization are `conv_transpose1d`
//! with an identity kernel at stride `hop`. f32 throughout, matching torch's reference
//! (the detokenizer backbone was trained against torch's f32 irfft, so f32 — not f64 —
//! is the faithful match).

use candle_core::{Result, Tensor};
use candle_nn::{linear, Embedding, Linear, Module, VarBuilder};

use crate::model::lfm2_hf::{Cache, Lfm2Config, Model as Lfm2Model};
use crate::model::linear::linear_forward;

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
        let offs: Vec<i64> = (0..CODEBOOKS as i64)
            .map(|i| i * AUDIO_VOCAB as i64)
            .collect();
        let offsets = Tensor::from_vec(offs, (CODEBOOKS,), vb.device())?;
        Ok(Self { emb, offsets })
    }

    /// `x`: (B, L, codebooks) u32 codes → (B, L, dim).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // offset_x = offsets[None,None,:] + x   (codebooks last)
        let offsets = self
            .offsets
            .reshape((1, 1, CODEBOOKS))?
            .to_dtype(x.dtype())?;
        let offset_x = x.broadcast_add(&offsets)?; // (B, L, codebooks)
        let (b, l, _) = offset_x.dims3()?;
        let flat = offset_x.reshape((b * l * CODEBOOKS,))?;
        let emb = self
            .emb
            .forward(&flat)?
            .reshape((b, l, CODEBOOKS, EMB_DIM))?;
        emb.mean(2) // average over codebooks → (B, L, dim)
    }
}

/// Vocos-style inverse STFT, "same" padding. n_fft=1280, hop=320, win=1280.
///
/// Everything is precomputed on the device so each call is matmul + conv_transpose1d.
struct Istft {
    hop: usize,
    /// `(win_length - hop)/2` — the "same"-padding trim on each side.
    pad: usize,
    /// Inverse real-DFT basis `(freq, n_fft)` with the Hermitian weights and the
    /// `norm="backward"` `1/n` scale folded in: `y = Re·cw + Im·sw`. cos/sin computed
    /// in f64, stored f32 (accurate basis, f32 storage — matching torch's f32 irfft).
    cw: Tensor,
    sw: Tensor,
    /// Analysis window `(1, n_fft, 1)` for the broadcast multiply, and its square for
    /// the overlap envelope.
    window: Tensor,
    win_sq: Tensor,
    /// Identity overlap-add kernel `(n_fft, 1, n_fft)` for `conv_transpose1d`.
    ola: Tensor,
}

impl Istft {
    fn new(n_fft: usize, hop: usize, win_length: usize, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let window_vec = vb
            .get(win_length, "window")?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        // Center the analysis window in an n_fft frame (torch pads when win < n_fft).
        let win: Vec<f32> = if window_vec.len() == n_fft {
            window_vec
        } else {
            let left = (n_fft - window_vec.len()) / 2;
            let mut w = vec![0f32; n_fft];
            w[left..left + window_vec.len()].copy_from_slice(&window_vec);
            w
        };

        // Inverse real-DFT basis, norm="backward" (1/n on the inverse), Hermitian
        // weights a_k (DC and even-n Nyquist ×1, the rest ×2). torch.fft.irfft ignores
        // the imag of DC/Nyquist, which falls out here since sw[0]=sw[n/2]=0.
        let freq = n_fft / 2 + 1;
        let two_pi = 2.0 * std::f64::consts::PI;
        let scale = 1.0 / n_fft as f64;
        let mut cw = vec![0f32; freq * n_fft];
        let mut sw = vec![0f32; freq * n_fft];
        for k in 0..freq {
            let a = if k == 0 || (n_fft % 2 == 0 && k == n_fft / 2) {
                1.0
            } else {
                2.0
            };
            for j in 0..n_fft {
                let ang = two_pi * k as f64 * j as f64 / n_fft as f64;
                cw[k * n_fft + j] = (a * ang.cos() * scale) as f32;
                sw[k * n_fft + j] = (-(a * ang.sin() * scale)) as f32;
            }
        }
        let cw = Tensor::from_vec(cw, (freq, n_fft), &dev)?;
        let sw = Tensor::from_vec(sw, (freq, n_fft), &dev)?;

        let win_sq_vec: Vec<f32> = win.iter().map(|w| w * w).collect();
        let window = Tensor::from_vec(win.clone(), (1, n_fft, 1), &dev)?;
        let win_sq = Tensor::from_vec(win_sq_vec, (1, n_fft, 1), &dev)?;

        // Identity kernel: conv_transpose1d places frame sample j at output t·hop + j,
        // i.e. overlap-add. ola[ci][0][j] = (ci==j).
        let mut eye = vec![0f32; n_fft * n_fft];
        for i in 0..n_fft {
            eye[i * n_fft + i] = 1.0;
        }
        let ola = Tensor::from_vec(eye, (n_fft, 1, n_fft), &dev)?;

        let pad = (win_length - hop) / 2;
        Ok(Self {
            hop,
            pad,
            cw,
            sw,
            window,
            win_sq,
            ola,
        })
    }

    /// `re`/`im`: `(B, n_fft/2+1, T)` complex spectrogram → waveform `(B, L)`.
    fn forward(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        let (b, freq, t) = re.dims3()?;
        let n = self.cw.dim(1)?; // n_fft

        // irfft along the freq axis as the inverse-DFT basis matmul:
        // frames[b,t,:] = Re[b,:,t]·cw + Im[b,:,t]·sw. Contract freq via candle matmul.
        // Cast the spectrum to f32: the backbone runs at the model dtype (bf16 on
        // Metal), but torch runs this irfft in f32 (`torch.polar` upcasts the bf16
        // output) and the STFT is precision-sensitive — so f32 here is the faithful
        // match and also the dtype of the basis `cw`/`sw`.
        let f32 = candle_core::DType::F32;
        let re_t = re
            .to_dtype(f32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, freq))?; // (B·T, freq)
        let im_t = im
            .to_dtype(f32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, freq))?;
        let frames = (re_t.matmul(&self.cw)? + im_t.matmul(&self.sw)?)?; // (B·T, n_fft)
        let frames = frames.reshape((b, t, n))?.transpose(1, 2)?.contiguous()?; // (B, n_fft, T)

        // Window, then overlap-add (conv_transpose1d, stride=hop) → (B, out_size).
        let frames = frames.broadcast_mul(&self.window)?;
        let y = frames
            .conv_transpose1d(&self.ola, 0, 0, self.hop, 1, 1)?
            .squeeze(1)?;
        // Window-overlap envelope: same overlap-add applied to win² over every frame.
        let env = self
            .win_sq
            .broadcast_as((b, n, t))?
            .contiguous()?
            .conv_transpose1d(&self.ola, 0, 0, self.hop, 1, 1)?
            .squeeze(1)?;

        // Trim the "same" padding on both sides and normalize by the envelope.
        let out_size = (t - 1) * self.hop + n;
        let valid = out_size - 2 * self.pad;
        let y = y.narrow(1, self.pad, valid)?;
        let env = env.narrow(1, self.pad, valid)?;
        y.broadcast_div(&env)
    }
}

/// Additive sliding-window causal mask `(1,1,n,n)`: attend where `i-w < j <= i`.
///
/// Built **on-device** with vectorized candle ops — a faithful port of Python's
/// `detokenizer.py:126-128` (`idx - idx[:,None]`, then `(d<=0) & (d>-w)`), *not* a
/// host-side scalar double-loop + `Tensor::from_vec` host→device copy. Since `n = 6·L`
/// grows with the reply and this runs on every forward pass, the GPU build avoids an
/// O(n²) CPU loop and a per-frame host→device transfer. Stateless, exactly like Python.
fn build_sliding_mask(
    n: usize,
    sliding_window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    use candle_core::DType;
    let w = sliding_window as i64;
    let idx = Tensor::arange(0i64, n as i64, device)?; // (n,)
                                                       // d[i][j] = idx[j] - idx[i] = j - i  (broadcast (1,n) - (n,1))
    let d = idx.reshape((1, n))?.broadcast_sub(&idx.reshape((n, 1))?)?;
    // attend = (d <= 0) & (d > -w), as u8 (1 = attend); `&` via where_cond to avoid u8 mul
    let attend = d
        .le(0i64)?
        .where_cond(&d.gt(-w)?, &Tensor::zeros((n, n), DType::U8, device)?)?;
    // additive mask: 0 where attend, -inf elsewhere (matches the eager-SDPA convention)
    let neg = Tensor::full(f32::NEG_INFINITY, (n, n), device)?;
    let zeros = Tensor::zeros((n, n), DType::F32, device)?;
    attend.where_cond(&zeros, &neg)?.reshape((1, 1, n, n))
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
        Ok(Self {
            emb,
            lfm,
            lfm_cfg: backbone_cfg,
            lin,
            istft,
            sliding_window,
        })
    }

    /// Additive sliding-window causal mask `(1,1,n,n)` — see [`build_sliding_mask`].
    fn sliding_mask(&self, n: usize, device: &candle_core::Device) -> Result<Tensor> {
        build_sliding_mask(n, self.sliding_window, device)
    }

    /// `codes`: (B, L, codebooks) → waveform (1, samples).
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let x = self.emb.forward(codes)?; // (B, L, dim)
        let l = x.dim(1)?;
        // ×6 nearest-exact upsample over time = repeat-interleave by 6
        let (b, _, d) = x.dims3()?;
        let x = x
            .unsqueeze(2)?
            .broadcast_as((b, l, 6, d))?
            .reshape((b, l * 6, d))?
            .contiguous()?;
        let n = l * 6;

        let mask = self.sliding_mask(n, x.device())?;
        let mut cache = Cache::new(false, x.dtype(), &self.lfm_cfg, x.device())?;
        let h = self.lfm.forward_embeds(&x, 0, &mut cache, Some(&mask))?; // (B, n, dim)
        let x = linear_forward(&self.lin, &h)?; // (B, n, 1282)

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    // Independent f64 reference: per-frame hermitian inverse real DFT (norm="backward")
    // + windowed overlap-add + window-envelope normalization ("same" trim) — the exact
    // algorithm the candle Istft realizes, in f64.
    fn ref_istft(
        re: &[Vec<f32>],
        im: &[Vec<f32>],
        window: &[f32],
        n_fft: usize,
        hop: usize,
    ) -> Vec<f32> {
        let t = re[0].len();
        let freq = n_fft / 2 + 1;
        let win: Vec<f64> = window.iter().map(|&w| w as f64).collect();
        let win_sq: Vec<f64> = win.iter().map(|w| w * w).collect();
        let pad = (window.len() - hop) / 2;
        let out_size = (t - 1) * hop + n_fft;
        let two_pi = 2.0 * std::f64::consts::PI;
        let (mut y, mut env) = (vec![0f64; out_size], vec![0f64; out_size]);
        for ti in 0..t {
            for j in 0..n_fft {
                let mut acc = re[0][ti] as f64; // k=0 (DC, imag ignored)
                for k in 1..(n_fft / 2) {
                    let ang = two_pi * k as f64 * j as f64 / n_fft as f64;
                    acc += 2.0 * (re[k][ti] as f64 * ang.cos() - im[k][ti] as f64 * ang.sin());
                }
                let ang = two_pi * (n_fft / 2) as f64 * j as f64 / n_fft as f64;
                acc += re[freq - 1][ti] as f64 * ang.cos(); // Nyquist, weight 1, imag ignored
                let v = acc / n_fft as f64;
                y[ti * hop + j] += v * win[j];
                env[ti * hop + j] += win_sq[j];
            }
        }
        (pad..out_size - pad)
            .map(|i| (y[i] / env[i]) as f32)
            .collect()
    }

    #[test]
    fn candle_istft_matches_f64_reference() {
        let dev = Device::Cpu;
        let (n_fft, hop, t) = (16usize, 4usize, 5usize);
        let freq = n_fft / 2 + 1;
        // symmetric Hann window (periodic=False), strictly positive envelope under OLA.
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                (std::f64::consts::PI * i as f64 / (n_fft - 1) as f64)
                    .sin()
                    .powi(2) as f32
            })
            .collect();
        let re: Vec<Vec<f32>> = (0..freq)
            .map(|k| {
                (0..t)
                    .map(|ti| ((k * 7 + ti * 3) as f32 * 0.1).cos())
                    .collect()
            })
            .collect();
        let im: Vec<Vec<f32>> = (0..freq)
            .map(|k| {
                (0..t)
                    .map(|ti| ((k * 5 + ti * 2) as f32 * 0.13).sin())
                    .collect()
            })
            .collect();
        let exp = ref_istft(&re, &im, &window, n_fft, hop);

        // Build the candle Istft via a VarBuilder carrying the window.
        let win_t = Tensor::from_vec(window.clone(), (n_fft,), &dev).unwrap();
        let vb = VarBuilder::from_tensors(
            HashMap::from([("window".to_string(), win_t)]),
            DType::F32,
            &dev,
        );
        let istft = Istft::new(n_fft, hop, n_fft, vb).unwrap();
        // (1, freq, T) spectra.
        let re_flat: Vec<f32> = (0..freq).flat_map(|k| re[k].clone()).collect();
        let im_flat: Vec<f32> = (0..freq).flat_map(|k| im[k].clone()).collect();
        let re_t = Tensor::from_vec(re_flat, (1, freq, t), &dev).unwrap();
        let im_t = Tensor::from_vec(im_flat, (1, freq, t), &dev).unwrap();
        let got: Vec<f32> = istft
            .forward(&re_t, &im_t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(got.len(), exp.len(), "ISTFT length mismatch");
        let maxd = got
            .iter()
            .zip(exp.iter())
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        let scale = exp.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        eprintln!(
            "candle ISTFT vs f64 ref: max diff {maxd:.2e} (rel {:.2e})",
            maxd / scale
        );
        assert!(
            maxd / scale < 1e-4,
            "candle ISTFT vs f64 ref rel {}",
            maxd / scale
        );
    }

    #[test]
    fn sliding_mask_matches_reference_band() {
        let dev = Device::Cpu;
        for (n, w) in [(7usize, 3usize), (13, 5), (20, 30)] {
            // Reference = the former host-loop definition: additive 0 where
            // (j<=i && j>i-w), -inf elsewhere.
            let mut exp = vec![0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let d = j as i64 - i as i64;
                    if !(d <= 0 && d > -(w as i64)) {
                        exp[i * n + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let m = build_sliding_mask(n, w, &dev).unwrap();
            assert_eq!(m.dims(), &[1, 1, n, n]);
            let got: Vec<f32> = m
                .reshape((n, n))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (k, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
                let same = (g == e)
                    || (g.is_infinite()
                        && e.is_infinite()
                        && g.is_sign_negative() == e.is_sign_negative());
                assert!(
                    same,
                    "mask mismatch at {k} (n={n}, w={w}): got {g}, exp {e}"
                );
            }
        }
    }
}
