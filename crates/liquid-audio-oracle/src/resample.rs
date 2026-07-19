//! Candle adapter around the production native resampler.

use candle_core::{DType, Result, Tensor};

pub use liquid_audio::resample::resample_slice;

/// `torchaudio.functional.resample(wave, orig_freq, new_freq)` with the library
/// defaults. `wave` is `(1, L)` -> `(1, L')` f32 with
/// `L' = ceil(L * new/orig)`.
pub fn resample(wave: &Tensor, orig_freq: u32, new_freq: u32) -> Result<Tensor> {
    if orig_freq == 0 || new_freq == 0 {
        return Err(candle_core::Error::Msg(format!(
            "resample: sample rates must be non-zero, got orig={orig_freq}, new={new_freq}"
        )));
    }
    if orig_freq == new_freq {
        return wave.contiguous();
    }
    let dev = wave.device().clone();
    let x = wave.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let y = resample_slice(&x, orig_freq, new_freq);
    let n = y.len();
    Tensor::from_vec(y, (1, n), &dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_sample_rate() {
        let x = Tensor::from_vec(vec![0.0f32, 1.0], (1, 2), &candle_core::Device::Cpu).unwrap();
        let err = resample(&x, 0, 16_000).unwrap_err().to_string();
        assert!(err.contains("sample rates must be non-zero"), "{err}");
        let err = resample(&x, 16_000, 0).unwrap_err().to_string();
        assert!(err.contains("sample rates must be non-zero"), "{err}");
    }
}
