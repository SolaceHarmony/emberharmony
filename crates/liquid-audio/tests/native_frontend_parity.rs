#![cfg(feature = "oracle")]

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

#[repr(C)]
struct NativeFrontend {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeFrontendWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeResampler {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeResamplerWorkspace {
    _private: [u8; 0],
}

#[repr(C)]
struct NativeFrontendConfig {
    size: u32,
    abi_version: u32,
    sample_rate: u32,
    n_window_size: u32,
    n_window_stride: u32,
    n_fft: u32,
    nfilt: u32,
    exact_pad: u32,
    pad_to: u32,
    reserved0: u32,
    preemph: f64,
    log_zero_guard_value: f64,
    mag_power: f64,
    reserved: [u64; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeF32Span {
    data: *const f32,
    length: u64,
}

const FRONTEND_ABI: u32 = 1;
const VALID_ONLY: u32 = 1;
const BF16_OUTPUT: u32 = 2;

unsafe extern "C" {
    fn lfm_preemph_f32(x: *const f32, y: *mut f32, n: u64, coef: f32) -> i32;
    fn lfm_power_spec_f32(re: *const f32, im: *const f32, out: *mut f32, n: u64);
    fn lfm_frontend_create(
        config: *const NativeFrontendConfig,
        out: *mut *mut NativeFrontend,
    ) -> i32;
    fn lfm_frontend_destroy(frontend: *mut NativeFrontend) -> i32;
    fn lfm_frontend_workspace_create(out: *mut *mut NativeFrontendWorkspace) -> i32;
    fn lfm_frontend_workspace_destroy(workspace: *mut NativeFrontendWorkspace) -> i32;
    fn lfm_frontend_workspace_reserve(
        frontend: *const NativeFrontend,
        workspace: *mut NativeFrontendWorkspace,
        max_sample_count: u64,
        flags: u32,
    ) -> i32;
    fn lfm_frontend_seq_len(frontend: *const NativeFrontend, sample_count: u64) -> u64;
    fn lfm_frontend_forward_workspace(
        frontend: *const NativeFrontend,
        workspace: *mut NativeFrontendWorkspace,
        pcm: *const f32,
        sample_count: u64,
        out_mel: *mut f32,
        out_capacity_values: u64,
        flags: u32,
    ) -> i32;
    fn lfm_frontend_forward_bf16_workspace(
        frontend: *const NativeFrontend,
        workspace: *mut NativeFrontendWorkspace,
        pcm: *const f32,
        sample_count: u64,
        out_mel: *mut u16,
        out_capacity_values: u64,
    ) -> i32;
    fn lfm_resampler_create(orig_freq: u32, new_freq: u32, out: *mut *mut NativeResampler) -> i32;
    fn lfm_resampler_destroy(resampler: *mut NativeResampler) -> i32;
    fn lfm_resampler_workspace_create(out: *mut *mut NativeResamplerWorkspace) -> i32;
    fn lfm_resampler_workspace_destroy(workspace: *mut NativeResamplerWorkspace) -> i32;
    fn lfm_resampler_workspace_reserve(
        resampler: *const NativeResampler,
        workspace: *mut NativeResamplerWorkspace,
        max_sample_count: u64,
    ) -> i32;
    fn lfm_resampler_process(
        resampler: *const NativeResampler,
        workspace: *mut NativeResamplerWorkspace,
        input: *const f32,
        sample_count: u64,
        destination: *mut f32,
        destination_capacity: u64,
        result: *mut NativeF32Span,
    ) -> i32;
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

fn native_cfg(cfg: &MelConfig) -> NativeFrontendConfig {
    NativeFrontendConfig {
        size: std::mem::size_of::<NativeFrontendConfig>() as u32,
        abi_version: FRONTEND_ABI,
        sample_rate: cfg.sample_rate as u32,
        n_window_size: cfg.n_window_size as u32,
        n_window_stride: cfg.n_window_stride as u32,
        n_fft: cfg.n_fft as u32,
        nfilt: cfg.nfilt as u32,
        exact_pad: cfg.exact_pad as u32,
        pad_to: cfg.pad_to as u32,
        reserved0: 0,
        preemph: cfg.preemph,
        log_zero_guard_value: cfg.log_zero_guard_value,
        mag_power: cfg.mag_power,
        reserved: [0; 4],
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

#[test]
fn prepared_resampler_borrows_equal_rate_and_writes_different_rate_directly() {
    let input = read_f32(&fixture_dir("resample").join("input_ramp24_to16.bin"));

    unsafe {
        let mut same = std::ptr::null_mut();
        assert_eq!(lfm_resampler_create(16_000, 16_000, &mut same), 0);
        let mut same_ws = std::ptr::null_mut();
        assert_eq!(lfm_resampler_workspace_create(&mut same_ws), 0);
        assert_eq!(
            lfm_resampler_workspace_reserve(same, same_ws, input.len() as u64),
            0
        );
        let mut borrowed = NativeF32Span {
            data: std::ptr::null(),
            length: 0,
        };
        assert_eq!(
            lfm_resampler_process(
                same,
                same_ws,
                input.as_ptr(),
                input.len() as u64,
                std::ptr::null_mut(),
                0,
                &mut borrowed,
            ),
            0
        );
        assert_eq!(
            borrowed.data,
            input.as_ptr(),
            "equal-rate PCM must alias input"
        );
        assert_eq!(borrowed.length as usize, input.len());
        assert_eq!(lfm_resampler_workspace_destroy(same_ws), 0);
        assert_eq!(lfm_resampler_destroy(same), 0);

        let want = read_f32(&fixture_dir("resample").join("output_ramp24_to16.bin"));
        let mut down = std::ptr::null_mut();
        assert_eq!(lfm_resampler_create(24_000, 16_000, &mut down), 0);
        let mut down_ws = std::ptr::null_mut();
        assert_eq!(lfm_resampler_workspace_create(&mut down_ws), 0);
        assert_eq!(
            lfm_resampler_workspace_reserve(down, down_ws, input.len() as u64),
            0
        );
        let mut output = vec![f32::NAN; want.len()];
        let mut written = NativeF32Span {
            data: std::ptr::null(),
            length: 0,
        };
        assert_eq!(
            lfm_resampler_process(
                down,
                down_ws,
                input.as_ptr(),
                input.len() as u64,
                output.as_mut_ptr(),
                output.len() as u64,
                &mut written,
            ),
            0
        );
        assert_eq!(
            written.data,
            output.as_ptr(),
            "resample must publish destination"
        );
        assert_eq!(written.length as usize, want.len());
        for (got, expected) in output.iter().zip(&want) {
            #[cfg(target_arch = "aarch64")]
            assert_eq!(got.to_bits(), expected.to_bits());
            #[cfg(not(target_arch = "aarch64"))]
            assert!((*got - *expected).abs() <= 1e-9);
        }

        // 100 * 2/3 -> 67 exercises a partial final phase block. The assembly
        // leaf must stop at the exact destination value count and preserve the
        // adjacent canary (the removed staging plane used to hide this edge).
        let mut partial = vec![0.0f32; 68];
        partial[67] = f32::from_bits(0x7f12_3456);
        assert_eq!(
            lfm_resampler_process(
                down,
                down_ws,
                input.as_ptr(),
                100,
                partial.as_mut_ptr(),
                67,
                &mut written,
            ),
            0
        );
        assert_eq!(written.length, 67);
        assert_eq!(partial[67].to_bits(), 0x7f12_3456);

        // Execution is strict: a larger unreserved command is rejected rather
        // than growing the workspace in the hot call.
        let larger = vec![0.0f32; input.len() * 2];
        let mut larger_out = vec![0.0f32; want.len() * 2];
        assert_ne!(
            lfm_resampler_process(
                down,
                down_ws,
                larger.as_ptr(),
                larger.len() as u64,
                larger_out.as_mut_ptr(),
                larger_out.len() as u64,
                &mut written,
            ),
            0,
            "unprepared resample must not allocate"
        );
        assert_eq!(lfm_resampler_workspace_destroy(down_ws), 0);
        assert_eq!(lfm_resampler_destroy(down), 0);
    }
}

#[test]
fn prepared_frontend_rounds_directly_into_bf16_destination() {
    let cfg = cfg_a();
    let native = native_cfg(&cfg);
    let input = read_f32(&fixture_dir("mel").join("input_tone440_8000.bin"));
    unsafe {
        let mut frontend = std::ptr::null_mut();
        assert_eq!(lfm_frontend_create(&native, &mut frontend), 0);
        let mut workspace = std::ptr::null_mut();
        assert_eq!(lfm_frontend_workspace_create(&mut workspace), 0);
        assert_eq!(
            lfm_frontend_workspace_reserve(
                frontend,
                workspace,
                input.len() as u64,
                VALID_ONLY | BF16_OUTPUT,
            ),
            0
        );
        let frames = lfm_frontend_seq_len(frontend, input.len() as u64) as usize;
        let values = cfg.nfilt * frames;
        let mut direct = vec![0u16; values];
        assert_eq!(
            lfm_frontend_forward_bf16_workspace(
                frontend,
                workspace,
                input.as_ptr(),
                input.len() as u64,
                direct.as_mut_ptr(),
                direct.len() as u64,
            ),
            0
        );

        // The BF16 run's larger alias layout also admits the f32 parity run.
        // Both execute without changing workspace capacity.
        let mut f32s = vec![0.0f32; values];
        assert_eq!(
            lfm_frontend_forward_workspace(
                frontend,
                workspace,
                input.as_ptr(),
                input.len() as u64,
                f32s.as_mut_ptr(),
                f32s.len() as u64,
                VALID_ONLY,
            ),
            0
        );
        let expected: Vec<u16> = f32s
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_bits())
            .collect();
        assert_eq!(
            direct, expected,
            "direct BF16 seam must round final mel exactly"
        );

        let larger = vec![0.0f32; input.len() * 2];
        let larger_frames = lfm_frontend_seq_len(frontend, larger.len() as u64) as usize;
        let mut larger_out = vec![0u16; cfg.nfilt * larger_frames];
        assert_ne!(
            lfm_frontend_forward_bf16_workspace(
                frontend,
                workspace,
                larger.as_ptr(),
                larger.len() as u64,
                larger_out.as_mut_ptr(),
                larger_out.len() as u64,
            ),
            0,
            "unprepared frontend command must not allocate"
        );
        assert_eq!(lfm_frontend_workspace_destroy(workspace), 0);
        assert_eq!(lfm_frontend_destroy(frontend), 0);
    }
}
