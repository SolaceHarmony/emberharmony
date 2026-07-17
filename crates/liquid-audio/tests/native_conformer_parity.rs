#![cfg(feature = "oracle")]

//! Native Conformer parity gate: the native encoder+adapter vs the fixtures
//! captured from the deleted Rust (native/tests/fixtures/conformer/, real
//! checkpoint, BF16 production ladder).
//!
//! Requires `LFM_MODEL_DIR` to name the local LFM2.5-Audio snapshot used by the
//! fixture manifest. The gate is explicitly ignored in checkpoint-free CI;
//! invoking it without the checkpoint is an error, never a silent pass.
//!
//! Policy: out_rows exact (shape-first, asserted before values); adapter
//! values within a BF16-ladder tolerance across 17 layers. The comparison is
//! over BF16 bit patterns widened to f32. Recorded max/RMS print every run.

use liquid_audio::model::native_conformer::{ConformerGeometry, NativeConformer};
use liquid_audio::weights::ResidentWeights;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/tests/fixtures/conformer")
}
fn mel_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/tests/fixtures/mel")
}

fn read_bf16(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}
fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn geometry() -> ConformerGeometry {
    ConformerGeometry {
        feat_in: 128,
        d_model: 512,
        n_layers: 17,
        n_heads: 8,
        d_ff: 2048,
        conv_kernel: 9,
        subsampling: 8,
        conv_channels: 256,
        adapter_hidden: 2048,
        adapter_out: 2048,
    }
}

#[test]
#[ignore = "requires LFM_MODEL_DIR and the real LFM2.5-Audio checkpoint"]
fn native_conformer_matches_reference_fixtures() {
    let snapshot = std::path::PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .expect("LFM_MODEL_DIR must name the real LFM2.5-Audio checkpoint"),
    );
    let resident =
        ResidentWeights::open(&snapshot.join("model.safetensors")).expect("open resident image");
    let dev = candle_core::Device::Cpu;
    let conf = NativeConformer::new(resident, geometry(), &dev).expect("native conformer");
    let initial = conf.memory();
    assert!(
        initial.bound_weight_bytes > 0,
        "checkpoint views must be bound"
    );
    assert_eq!(initial.derived_bytes, 17 * 512 * 2 + 256 * 4);
    assert_eq!(initial.materialized_weight_bytes, 0);
    assert_eq!(initial.direct_gemm_calls, 0);

    let cases = ["tone440", "ramp_mix", "single"];
    // The mel fixture planes are f32 (feat_in x T); the native path casts BF16
    // internally, exactly as the Rust capture harness did before the encoder.
    let plane_name = |c: &str| match c {
        "tone440" => "a_tone440_8000_forward.bin",
        "ramp_mix" => "a_ramp_mix_4000_forward.bin",
        "single" => "a_single_200_forward.bin",
        _ => unreachable!(),
    };

    // The `single` segment yields one encoder row → an M=1 backbone GEMM, which
    // needs the native bf16 GEMV kernel. Under Rosetta (no AVX512-BF16) that
    // kernel is unavailable, so M=1 legitimately returns -ENOTSUP (no fallback
    // by design). Skip `single` there; tone440/ramp_mix (M≥4) use the AMX/cblas
    // path and still prove the x86 leaves. On aarch64 the kernel is present and
    // `single` runs fully.
    let m1_bf16 = liquid_audio::flashkern::native_engine::bf16_gemm_available();

    let mut worst = 0f64;
    let mut prior = initial;
    const DIRECT_GEMMS_PER_FORWARD: u64 = 3 + 17 * 11 + 2;
    for case in cases {
        let mel = read_f32(&mel_dir().join(plane_name(case)));
        let frames = mel.len() / 128;
        let mel_t = candle_core::Tensor::from_vec(mel, (1, 128, frames), &dev).unwrap();
        let got = match conf.forward_segment(&mel_t) {
            Ok(t) => t,
            Err(e) if case == "single" && !m1_bf16 => {
                eprintln!("[parity] single skipped — M=1 bf16 GEMV unavailable: {e}");
                continue;
            }
            Err(e) => panic!("{case}: {e}"),
        };
        let memory = conf.memory();
        assert_eq!(memory.bound_weight_bytes, initial.bound_weight_bytes);
        assert_eq!(memory.derived_bytes, initial.derived_bytes);
        assert_eq!(memory.materialized_weight_bytes, 0);
        assert_eq!(
            memory.direct_gemm_calls - prior.direct_gemm_calls,
            DIRECT_GEMMS_PER_FORWARD,
            "{case}: every linear and pointwise pass must use the direct checkpoint-layout ticket"
        );
        prior = memory;

        let want = read_bf16(&fixture_dir().join(format!("{case}_adapter.bf16.bin")));
        let (rows, cols) = got.dims2().unwrap();
        assert_eq!(cols, 2048, "{case}: adapter_out");
        assert_eq!(
            rows * cols,
            want.len(),
            "{case}: out_rows mismatch (shape-first gate): {rows}x{cols} vs {}",
            want.len()
        );
        let g: Vec<f32> = got
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut max_abs = 0f64;
        let mut sumsq = 0f64;
        let mut scale = 1e-6f64;
        for (a, b) in g.iter().zip(want.iter()) {
            let d = (*a as f64 - *b as f64).abs();
            max_abs = max_abs.max(d);
            sumsq += d * d;
            scale = scale.max(b.abs() as f64);
        }
        let rms = (sumsq / want.len() as f64).sqrt();
        eprintln!(
            "[parity] {case}: rows={rows} max_abs={max_abs:.4e} rms={rms:.4e} scale={scale:.3}"
        );
        // 17 layers of BF16 arithmetic with GEMM-order reassociation in the
        // matmul-shaped stages. Relative bound against the row scale.
        let rel = max_abs / scale;
        assert!(
            rel < 6e-2,
            "{case}: adapter diverges (max_abs {max_abs:.4e}, scale {scale:.3}, rel {rel:.3})"
        );
        worst = worst.max(rel);
    }
    eprintln!("[parity] native conformer worst relative divergence: {worst:.4e}");
}
