//! Native frontend parity gate: the native mel featurizer and resampler versus
//! the fixtures captured from the deleted Rust implementation
//! (native/tests/fixtures/{mel,resample}/, working tree e018540c).
//!
//! Policy per stage class (spec 11 doc 11):
//! - lengths and shapes: exact, asserted FIRST, full-length comparison;
//! - resampler: bitwise (same f64 kernel construction and accumulation order);
//! - mel end-to-end: tolerance — the two matmul-shaped stages ride Accelerate
//!   on Apple, whose accumulation order differs from the reference candle
//!   matmul; everything after amplifies that reordering noise through log and
//!   a divide by std >= 1e-5. Bound: per-element |diff| <= 2e-3 OR rel 1e-3,
//!   and RMS(diff) <= 1e-4. Recorded maxima print on every run.

use liquid_audio::processor::{FilterbankFeatures, MelConfig};

unsafe extern "C" {
    fn lfm_preemph_f32(x: *const f32, y: *mut f32, n: u64, coef: f32) -> i32;
    fn lfm_power_spec_f32(re: *const f32, im: *const f32, out: *mut f32, n: u64);
}

fn fixture_dir(sub: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("native/tests/fixtures")
        .join(sub)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("fixture {} unreadable: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}: not f32le", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn manifest(sub: &str) -> serde_json::Value {
    let p = fixture_dir(sub).join("manifest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn cfg_a() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        n_window_size: 400,
        n_window_stride: 160,
        n_fft: 512,
        nfilt: 128,
        preemph: 0.97,
        log_zero_guard_value: 2f64.powi(-24),
        mag_power: 2.0,
        pad_to: 0,
        exact_pad: false,
    }
}

fn cfg_b() -> MelConfig {
    MelConfig {
        pad_to: 16,
        exact_pad: true,
        ..cfg_a()
    }
}

/// Full-length compare with the mel tolerance policy; returns (max_abs, rms).
fn compare_mel(name: &str, got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut max_abs = 0f64;
    let mut sumsq = 0f64;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}[{i}]: non-finite (got {g}, want {w})"
        );
        let d = (*g as f64 - *w as f64).abs();
        let rel = d / (w.abs() as f64).max(1e-12);
        assert!(
            d <= 2e-3 || rel <= 1e-3,
            "{name}[{i}]: got {g}, want {w} (abs {d:.3e}, rel {rel:.3e})"
        );
        max_abs = max_abs.max(d);
        sumsq += d * d;
    }
    let rms = (sumsq / want.len().max(1) as f64).sqrt();
    assert!(rms <= 1e-4, "{name}: RMS diff {rms:.3e} exceeds 1e-4");
    (max_abs, rms)
}

#[test]
fn resampler_matches_deleted_rust() {
    // aarch64: bitwise — the fixtures were captured on this architecture, so
    // the same libm feeds both sides and the f64 accumulation order is
    // identical. Other arches: the f64 sin/cos/pow kernel tables differ from
    // the capture host's libm by ulps, so the gate is a tight tolerance
    // (observed cross-arch diffs are ~1e-19 absolute on ~1e-16 values).
    let dir = fixture_dir("resample");
    for (name, orig, new) in [
        ("ramp24_to16", 24_000u32, 16_000u32),
        ("tone48_to16", 48_000, 16_000),
        ("tone16_to24", 16_000, 24_000),
    ] {
        let input = read_f32(&dir.join(format!("input_{name}.bin")));
        let want = read_f32(&dir.join(format!("output_{name}.bin")));
        let got = liquid_audio::resample::resample_slice(&input, orig, new);
        assert_eq!(got.len(), want.len(), "{name}: output length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            #[cfg(target_arch = "aarch64")]
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{name}[{i}]: native {g} != reference {w} (bitwise gate)"
            );
            #[cfg(not(target_arch = "aarch64"))]
            {
                let d = (*g as f64 - *w as f64).abs();
                let rel = d / (w.abs() as f64).max(1e-12);
                assert!(
                    d <= 1e-9 || rel <= 1e-5,
                    "{name}[{i}]: native {g} != reference {w} (abs {d:.3e}, rel {rel:.3e})"
                );
            }
        }
    }
}

#[test]
fn seq_len_matches_reference_table() {
    let m = manifest("mel");
    let lens: Vec<u64> = serde_json::from_value(m["get_seq_len"]["lens"].clone()).unwrap();
    let want_a: Vec<u64> = serde_json::from_value(m["get_seq_len"]["config_a"].clone()).unwrap();
    let want_b: Vec<u64> = serde_json::from_value(m["get_seq_len"]["config_b"].clone()).unwrap();
    let dev = candle_core::Device::Cpu;
    let fa = FilterbankFeatures::new(cfg_a(), &dev).unwrap();
    let fb = FilterbankFeatures::new(cfg_b(), &dev).unwrap();
    for (i, &l) in lens.iter().enumerate() {
        assert_eq!(
            fa.get_seq_len(l as usize) as u64,
            want_a[i],
            "cfg_a len {l}"
        );
        assert_eq!(
            fb.get_seq_len(l as usize) as u64,
            want_b[i],
            "cfg_b len {l}"
        );
    }
}

#[test]
fn mel_forward_matches_reference_fixtures() {
    let dir = fixture_dir("mel");
    let dev = candle_core::Device::Cpu;
    let fa = FilterbankFeatures::new(cfg_a(), &dev).unwrap();
    // Every config-A input has an end-to-end fixture.
    for name in [
        "tone440_8000",
        "ramp_mix_4000",
        "impulse_800",
        "silence_1600",
        "single_200",
        "sub_300",
    ] {
        let input = read_f32(&dir.join(format!("input_{name}.bin")));
        let want = read_f32(&dir.join(format!("a_{name}_forward.bin")));
        let x = candle_core::Tensor::from_vec(input.clone(), input.len(), &dev).unwrap();
        let out = fa.forward(&x).unwrap();
        assert_eq!(out.dims()[1], 128, "a_{name}: nfilt");
        assert_eq!(
            out.dims()[1] * out.dims()[2],
            want.len(),
            "a_{name}: frame count (shape-first gate)"
        );
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (max_abs, rms) = compare_mel(&format!("a_{name}"), &got, &want);
        eprintln!("[parity] a_{name}: max_abs {max_abs:.3e} rms {rms:.3e}");
    }
    // Config B (exact_pad + pad_to) against the staged finals.
    let fb = FilterbankFeatures::new(cfg_b(), &dev).unwrap();
    for name in ["tone440_8000", "single_200"] {
        let input = read_f32(&dir.join(format!("input_{name}.bin")));
        let want = read_f32(&dir.join(format!("b_{name}_mel_final.bin")));
        let x = candle_core::Tensor::from_vec(input.clone(), input.len(), &dev).unwrap();
        let out = fb.forward(&x).unwrap();
        assert_eq!(
            out.dims()[1] * out.dims()[2],
            want.len(),
            "b_{name}: frame count (shape-first gate)"
        );
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (max_abs, rms) = compare_mel(&format!("b_{name}"), &got, &want);
        eprintln!("[parity] b_{name}: max_abs {max_abs:.3e} rms {rms:.3e}");
    }
}

#[test]
fn mel_valid_span_matches_cropped_reference_and_preserves_pcm() {
    let dir = fixture_dir("mel");
    let dev = candle_core::Device::Cpu;
    for (cfg, prefix, names) in [
        (
            cfg_a(),
            "a",
            &[
                "tone440_8000",
                "ramp_mix_4000",
                "impulse_800",
                "silence_1600",
                "single_200",
                "sub_300",
            ][..],
        ),
        (cfg_b(), "b", &["tone440_8000", "single_200"][..]),
    ] {
        let features = FilterbankFeatures::new(cfg, &dev).unwrap();
        for &name in names {
            let input = read_f32(&dir.join(format!("input_{name}.bin")));
            let before: Vec<u32> = input.iter().map(|v| v.to_bits()).collect();
            let path = if prefix == "a" {
                dir.join(format!("a_{name}_forward.bin"))
            } else {
                dir.join(format!("b_{name}_mel_final.bin"))
            };
            let padded = read_f32(&path);
            let valid = features.get_seq_len(input.len());
            let stride = padded.len() / 128;
            let mut want = Vec::with_capacity(128 * valid);
            for row in padded.chunks_exact(stride) {
                want.extend_from_slice(&row[..valid]);
            }

            let out = features.forward_slice(&input).unwrap();
            assert_eq!(out.dims(), &[128, valid], "{prefix}_{name}: shape");
            let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            compare_mel(&format!("{prefix}_{name}_valid"), &got, &want);
            assert_eq!(
                input.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                before,
                "{prefix}_{name}: native mutated retained PCM"
            );

            // A contiguous CPU view with a nonzero storage offset must borrow
            // exactly its layout span, not the storage's sentinel cells.
            let mut stored = Vec::with_capacity(input.len() + 2);
            stored.push(1234.0);
            stored.extend_from_slice(&input);
            stored.push(-1234.0);
            let tensor = candle_core::Tensor::from_vec(stored, input.len() + 2, &dev)
                .unwrap()
                .narrow(0, 1, input.len())
                .unwrap();
            let view = features.forward_valid(&tensor).unwrap();
            let view = view.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            compare_mel(&format!("{prefix}_{name}_offset"), &view, &want);
        }
    }
}

#[test]
fn aliased_frontend_leaves_match_disjoint_outputs() {
    let input = [0.25f32, -0.5, 0.75, -1.0, 0.125, 0.375, -0.625];
    let mut separate = [0.0f32; 7];
    let mut inplace = input;
    unsafe {
        assert_eq!(
            lfm_preemph_f32(
                input.as_ptr(),
                separate.as_mut_ptr(),
                input.len() as u64,
                0.97
            ),
            0
        );
        assert_eq!(
            lfm_preemph_f32(
                inplace.as_ptr(),
                inplace.as_mut_ptr(),
                inplace.len() as u64,
                0.97
            ),
            0
        );
    }
    assert_eq!(
        inplace.map(f32::to_bits),
        separate.map(f32::to_bits),
        "preemphasis must support exact x == y aliasing"
    );

    let real = [0.5f32, -1.5, 2.0, -0.25];
    let imag = [1.0f32, 0.25, -0.5, 3.0];
    let mut expected = [0.0f32; 4];
    let mut aliased = real;
    unsafe {
        lfm_power_spec_f32(real.as_ptr(), imag.as_ptr(), expected.as_mut_ptr(), 4);
        lfm_power_spec_f32(aliased.as_ptr(), imag.as_ptr(), aliased.as_mut_ptr(), 4);
    }
    assert_eq!(
        aliased.map(f32::to_bits),
        expected.map(f32::to_bits),
        "power must support exact out == re aliasing"
    );
}
