//! Parity oracle: the native C++/NEON/AMX Mimi decoder vs moshi-Rust, frame by
//! frame on the REAL checkpoint, far enough to cross the transformer's
//! 250-slot rotating-KV wrap (130 frames × 2 positions/frame = 260 positions).
//!
//! The acceptance band comes from the independent shadow-review validation:
//! worst absolute PCM error 4.11e-6, min correlation 0.999999999989. The
//! assert here is 5e-5 — an order of magnitude of headroom over the measured
//! band, tight enough that any structural regression (wrong ring slot, mask
//! drift, conv carry desync, reduction-order change) fails immediately rather
//! than "sounding a bit off".
//!
//! Run (needs the local model):
//!   LFM_MODEL_DIR=/path/to/model cargo test --release --test mimi_native_parity -- --ignored
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use candle_core::{DType, Device, Tensor};
use liquid_audio::mimi_native::NativeMimi;

const FRAMES: usize = 130;
const CODEBOOKS: usize = 8;
const MAX_ABS: f32 = 5e-5;

#[test]
#[ignore = "requires a Mimi checkpoint selected by LFM_MODEL_DIR"]
fn native_mimi_matches_moshi_across_kv_wrap() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR to the local model dir");
    let ckpt = std::path::Path::new(&dir).join("tokenizer-e351c8d8-checkpoint125.safetensors");
    assert!(ckpt.exists(), "mimi checkpoint missing at {ckpt:?}");

    let device = Device::Cpu;
    let mut moshi_mimi = liquid_audio::moshi::models::get_mimi(
        ckpt.to_str().expect("utf8 path"),
        CODEBOOKS,
        &device,
    )
    .expect("load moshi mimi");
    let native = NativeMimi::new(&ckpt, CODEBOOKS).expect("init native mimi");

    let mask = ::moshi::StreamMask::empty();
    let mut worst = 0f32;
    let mut worst_frame = 0usize;
    for frame in 0..FRAMES {
        let codes: Vec<u32> = (0..CODEBOOKS as u32)
            .map(|j| (frame as u32 * 173 + j * 257 + frame as u32 * j * 3) % 2048)
            .collect();

        let t = Tensor::from_vec(codes.clone(), (1, CODEBOOKS, 1), &device)
            .and_then(|t| t.to_dtype(DType::U32))
            .expect("codes tensor");
        let reference: Vec<f32> = moshi_mimi
            .decode_step(&::moshi::StreamTensor::from_tensor(t), &mask)
            .expect("moshi decode_step")
            .as_option()
            .expect("moshi emitted no frame")
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("moshi pcm");

        let got = native.decode_step(&codes).expect("native decode_step");

        assert_eq!(
            got.len(),
            reference.len(),
            "frame {frame}: native {} samples vs moshi {}",
            got.len(),
            reference.len()
        );
        assert_eq!(got.len(), 1920, "frame {frame}: expected 1920 samples");
        for (i, (&a, &b)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                a.is_finite(),
                "frame {frame} sample {i}: native produced non-finite {a}"
            );
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                worst_frame = frame;
            }
        }
    }
    eprintln!(
        "[mimi-parity] {FRAMES} frames (KV wrap crossed at ~125): worst |Δ| = {worst:.3e} \
         at frame {worst_frame} (band 4.11e-6 measured, {MAX_ABS:.0e} asserted)"
    );
    assert!(
        worst <= MAX_ABS,
        "native/moshi divergence {worst:.3e} exceeds {MAX_ABS:.0e}"
    );

    // Turn boundary: both sides reset, first post-reset frame must agree too.
    native.reset();
    moshi_mimi.reset_state();
    let codes: Vec<u32> = (0..CODEBOOKS as u32).map(|j| (j * 331) % 2048).collect();
    let t = Tensor::from_vec(codes.clone(), (1, CODEBOOKS, 1), &device)
        .and_then(|t| t.to_dtype(DType::U32))
        .expect("codes tensor");
    let reference: Vec<f32> = moshi_mimi
        .decode_step(&::moshi::StreamTensor::from_tensor(t), &mask)
        .expect("moshi decode_step")
        .as_option()
        .expect("moshi emitted no frame")
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("moshi pcm");
    let got = native.decode_step(&codes).expect("native decode_step");
    let worst_reset = got
        .iter()
        .zip(reference.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[mimi-parity] post-reset frame: worst |Δ| = {worst_reset:.3e}");
    assert!(
        worst_reset <= MAX_ABS,
        "post-reset divergence {worst_reset:.3e}"
    );
}
