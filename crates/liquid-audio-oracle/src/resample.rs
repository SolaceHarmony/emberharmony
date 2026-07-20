//! Offline Candle adapter around the native torchaudio-exact resampler.

use candle_core::{DType, Result, Tensor};

unsafe extern "C" {
    fn lfm_resample_f32(
        input: *const f32,
        length: u64,
        original_rate: u32,
        new_rate: u32,
        output: *mut f32,
        output_capacity: u64,
        output_length: *mut u64,
    ) -> i32;
}

/// Oracle-only owned-vector adapter. Production passes borrowed PCM views
/// directly through the native frontend/session and exposes no Rust numerical
/// resampling surface.
pub fn resample_slice(input: &[f32], original_rate: u32, new_rate: u32) -> Vec<f32> {
    if original_rate == new_rate || input.is_empty() {
        return input.to_vec();
    }
    let mut left = original_rate;
    let mut right = new_rate;
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    let original = (original_rate / left) as f64;
    let new = (new_rate / left) as f64;
    let target = ((new * input.len() as f64) / original).ceil() as usize;
    let mut output = vec![0.0f32; target];
    let mut written = 0u64;
    let status = unsafe {
        lfm_resample_f32(
            input.as_ptr(),
            input.len() as u64,
            original_rate,
            new_rate,
            output.as_mut_ptr(),
            target as u64,
            &mut written,
        )
    };
    assert!(
        status == 0 && written as usize == target,
        "native oracle resample failed (status {status}, {original_rate} -> {new_rate} Hz, {} samples)",
        input.len()
    );
    output
}

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
