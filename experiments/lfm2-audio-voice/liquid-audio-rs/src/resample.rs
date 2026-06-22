//! Faithful pure-Rust port of `torchaudio.functional.resample` (windowed-sinc).
//!
//! `liquid_audio` resamples audio via `torchaudio.functional.resample` in two
//! places: `ChatState.add_audio` / the processor (input waveform → 16 kHz) and
//! `data/mapper` (`_load_audio_bytes` resample-to-16k and `_encode_audio_out`
//! resample-to-Mimi-rate). torch is not a dependency in this port, so rather than
//! approximate with linear interpolation this reproduces torchaudio's algorithm
//! exactly: one windowed-sinc kernel per output phase (the library default
//! `resampling_method="sinc_interp_hann"`, `lowpass_filter_width=6`,
//! `rolloff=0.99`), applied as a strided conv1d over the gcd-reduced rates, then
//! truncated to `ceil(new_freq * length / orig_freq)` samples.
//!
//! Ref: `torchaudio/functional/functional.py` — `_get_sinc_resample_kernel`
//! (kernel construction) and `_apply_sinc_resample_kernel` (pad + conv1d).

use candle_core::{DType, Result, Tensor};

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// `torchaudio.functional.resample(wave, orig_freq, new_freq)` with the library
/// defaults. `wave` is `(1, L)` → `(1, L')` f32 with `L' = ceil(L * new/orig)`.
pub fn resample(wave: &Tensor, orig_freq: u32, new_freq: u32) -> Result<Tensor> {
    if orig_freq == new_freq {
        return wave.contiguous();
    }
    let dev = wave.device().clone();
    let x = wave.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let y = resample_slice(&x, orig_freq, new_freq);
    let n = y.len();
    Tensor::from_vec(y, (1, n), &dev)
}

/// The torchaudio kernel construction + strided conv on a plain f32 slice.
pub fn resample_slice(x: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    if orig_freq == new_freq || x.is_empty() {
        return x.to_vec();
    }
    use std::f64::consts::PI;
    const LOWPASS_WIDTH: i64 = 6; // torchaudio default lowpass_filter_width
    const ROLLOFF: f64 = 0.99; // torchaudio default rolloff

    let g = gcd(orig_freq, new_freq);
    let orig = (orig_freq / g) as i64;
    let new = (new_freq / g) as i64;
    let base_freq = (orig.min(new) as f64) * ROLLOFF;
    let width = ((LOWPASS_WIDTH as f64) * (orig as f64) / base_freq).ceil() as i64;
    let kernel_len = (2 * width + orig) as usize;
    let scale = base_freq / (orig as f64);

    // One kernel per output phase i in 0..new; index j runs over -width..(width+orig).
    // t = (-i/new + idx/orig) * base_freq, clamped to ±lowpass_width; Hann window
    // cos(t·π / lpw / 2)²; sinc = sin(πt)/(πt) (1 at t==0); kernel = sinc·window·scale.
    let mut kernels = vec![vec![0f64; kernel_len]; new as usize];
    for (i, k) in kernels.iter_mut().enumerate() {
        for (j, idx) in (-width..(width + orig)).enumerate() {
            let mut t = (-(i as f64) / (new as f64) + (idx as f64) / (orig as f64)) * base_freq;
            t = t.clamp(-(LOWPASS_WIDTH as f64), LOWPASS_WIDTH as f64);
            let window = (t * PI / (LOWPASS_WIDTH as f64) / 2.0).cos().powi(2);
            let tp = t * PI;
            let sinc = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            k[j] = sinc * window * scale;
        }
    }

    // torchaudio pads (width, width + orig) then conv1d(stride=orig); the
    // (new, 1, kernel_len) kernel + transpose/reshape interleave the phases so the
    // output order is block0[phase0..new], block1[phase0..new], …
    let length = x.len();
    let pad_left = width as usize;
    let pad_right = (width + orig) as usize;
    let padded_len = pad_left + length + pad_right;
    let mut padded = vec![0f64; padded_len];
    for (i, &s) in x.iter().enumerate() {
        padded[pad_left + i] = s as f64;
    }

    let stride = orig as usize;
    let blocks = if padded_len >= kernel_len { (padded_len - kernel_len) / stride + 1 } else { 0 };
    let mut out: Vec<f32> = Vec::with_capacity(blocks * new as usize);
    let mut start = 0usize;
    while start + kernel_len <= padded_len {
        for k in &kernels {
            let mut acc = 0f64;
            for (j, &kj) in k.iter().enumerate() {
                acc += padded[start + j] * kj;
            }
            out.push(acc as f32);
        }
        start += stride;
    }
    let target = (((new as f64) * (length as f64)) / (orig as f64)).ceil() as usize;
    out.truncate(target);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_when_equal() {
        let x = vec![1.0f32, 2.0, 3.0];
        assert_eq!(resample_slice(&x, 16_000, 16_000), x);
    }

    #[test]
    fn target_length_is_ceil() {
        // torchaudio target length = ceil(new * len / orig).
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(resample_slice(&x, 24_000, 16_000).len(), 67); // ceil(16000*100/24000)
        assert_eq!(resample_slice(&x, 16_000, 24_000).len(), 150); // ceil(24000*100/16000)
    }

    #[test]
    fn preserves_a_pure_tone_amplitude() {
        // A 440 Hz sine at 16 kHz, resampled to 24 kHz, should stay a ~unit sine
        // (windowed-sinc passes in-band content; amplitude within a few %).
        let n = 16_000usize;
        let f = 440.0f64;
        let x: Vec<f32> = (0..n).map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / 16_000.0).sin() as f32).collect();
        let y = resample_slice(&x, 16_000, 24_000);
        // ignore edges (kernel transient); check the interior peak ≈ 1.0
        let mid = &y[2000..y.len() - 2000];
        let peak = mid.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!((0.9..=1.05).contains(&peak), "peak {peak} out of expected band");
    }
}
