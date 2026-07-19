//! Rust rim over the native torchaudio-exact resampler
//! (native/src/frontend/lfm_frontend.cpp + flashkern_frontend.S).
//!
//! The former pure-Rust windowed-sinc implementation is DELETED; the native
//! port reproduces `torchaudio.functional.resample` (sinc_interp_hann,
//! `lowpass_filter_width=6`, `rolloff=0.99`, f64 kernels/accumulation,
//! truncate to `ceil(new * len / orig)`) and is gated by the committed
//! fixtures under native/tests/fixtures/resample/ (captured from the deleted
//! implementation).

unsafe extern "C" {
    fn lfm_resample_f32(
        x: *const f32,
        length: u64,
        orig_freq: u32,
        new_freq: u32,
        out: *mut f32,
        out_capacity: u64,
        out_length: *mut u64,
    ) -> i32;
}

/// The native resample on a plain f32 slice. Rates must be non-zero (the
/// tensor path validates; direct callers pass device/model rates that are
/// non-zero by construction) — a native rejection is a programmer error and
/// panics rather than degrading.
pub fn resample_slice(x: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    if orig_freq == new_freq || x.is_empty() {
        return x.to_vec();
    }
    // ceil(len * new/orig) over gcd-reduced rates, exactly the target the
    // native side truncates to — sizes the output buffer without a probe call.
    let mut a = orig_freq;
    let mut b = new_freq;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    let (orig, new) = ((orig_freq / a) as f64, (new_freq / a) as f64);
    let target = ((new * x.len() as f64) / orig).ceil() as usize;
    let mut out = vec![0f32; target];
    let mut out_len: u64 = 0;
    let rc = unsafe {
        lfm_resample_f32(
            x.as_ptr(),
            x.len() as u64,
            orig_freq,
            new_freq,
            out.as_mut_ptr(),
            target as u64,
            &mut out_len,
        )
    };
    assert!(
        rc == 0 && out_len as usize == target,
        "native resample failed (status {rc}, {} -> {} Hz, {} samples)",
        orig_freq,
        new_freq,
        x.len()
    );
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
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / 16_000.0).sin() as f32)
            .collect();
        let y = resample_slice(&x, 16_000, 24_000);
        // ignore edges (kernel transient); check the interior peak ≈ 1.0
        let mid = &y[2000..y.len() - 2000];
        let peak = mid.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(
            (0.9..=1.05).contains(&peak),
            "peak {peak} out of expected band"
        );
    }
}
