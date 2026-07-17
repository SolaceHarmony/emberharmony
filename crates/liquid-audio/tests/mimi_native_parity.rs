#![cfg(feature = "oracle")]

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
use std::ffi::{c_char, c_void, CStr, CString};

const FRAMES: usize = 130;
const CODEBOOKS: usize = 8;
const MAX_ABS: f32 = 5e-5;

extern "C" {
    fn lfm_weights_open_bundle(
        main: *const c_char,
        codec: *const c_char,
        image: *mut *mut c_void,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn lfm_weights_close(image: *mut c_void);
    fn mimi_decode_plan_new_from_image(
        plan: *mut *mut c_void,
        image: *const c_void,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn mimi_decode_plan_free(plan: *mut c_void);
    fn mimi_decode_plan_derived_bytes(plan: *const c_void) -> u64;
    fn mimi_decode_plan_bound_weight_bytes(plan: *const c_void) -> u64;
    fn mimi_decode_plan_compatibility_copied_bytes(plan: *const c_void) -> u64;
    fn mimi_decode_state_new(
        state: *mut *mut c_void,
        plan: *const c_void,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn mimi_decode_state_step(state: *mut c_void, codes: *const u32, pcm: *mut f32) -> i32;
    fn mimi_decode_state_free(state: *mut c_void);
    fn mimi_decode_state_bytes(state: *const c_void) -> u64;
}

struct Image(*mut c_void);

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { lfm_weights_close(self.0) };
    }
}

struct Plan(*mut c_void);

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe { mimi_decode_plan_free(self.0) };
    }
}

struct State(*mut c_void);

impl Drop for State {
    fn drop(&mut self) {
        unsafe { mimi_decode_state_free(self.0) };
    }
}

#[test]
fn null_mimi_state_is_an_error_not_a_priming_frame() {
    let codes = [0u32; CODEBOOKS];
    let mut pcm = [0.0f32; 1];
    assert!(
        unsafe { mimi_decode_state_step(std::ptr::null_mut(), codes.as_ptr(), pcm.as_mut_ptr()) }
            < 0
    );
}

#[test]
#[ignore = "requires the complete LFM2 model and Mimi checkpoint selected by LFM_MODEL_DIR"]
fn mimi_binds_codec_component_without_reopening_or_copying_weights() {
    let dir = std::env::var("LFM_MODEL_DIR").expect("set LFM_MODEL_DIR");
    let codec = std::path::Path::new(&dir).join("tokenizer-e351c8d8-checkpoint125.safetensors");
    let main = CString::new(dir.as_bytes()).unwrap();
    let codec = CString::new(codec.as_os_str().as_encoded_bytes()).unwrap();
    let mut error = [0i8; 512];
    let mut image = std::ptr::null_mut();
    let status = unsafe {
        lfm_weights_open_bundle(
            main.as_ptr(),
            codec.as_ptr(),
            &mut image,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    assert_eq!(
        status,
        0,
        "bundle load: {}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    let image = Image(image);

    let mut plan = std::ptr::null_mut();
    let status = unsafe {
        mimi_decode_plan_new_from_image(&mut plan, image.0, error.as_mut_ptr(), error.len())
    };
    assert_eq!(
        status,
        0,
        "Mimi bind: {}",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    );
    let plan = Plan(plan);
    let derived = unsafe { mimi_decode_plan_derived_bytes(plan.0) };
    let bound = unsafe { mimi_decode_plan_bound_weight_bytes(plan.0) };
    assert!(
        derived > 0,
        "formula-derived codebooks/RoPE must be accounted"
    );
    assert!(bound > 0, "required codec views must be accounted exactly");
    assert_eq!(
        unsafe { mimi_decode_plan_compatibility_copied_bytes(plan.0) },
        0,
        "layout/dtype/alignment copies are forbidden"
    );

    let create = |error: &mut [i8; 512]| {
        let mut state = std::ptr::null_mut();
        let status =
            unsafe { mimi_decode_state_new(&mut state, plan.0, error.as_mut_ptr(), error.len()) };
        assert_eq!(
            status,
            0,
            "Mimi state: {}",
            unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
        );
        State(state)
    };
    let first = create(&mut error);
    let second = create(&mut error);
    assert!(unsafe { mimi_decode_state_bytes(first.0) } > 0);
    assert_eq!(unsafe { mimi_decode_state_bytes(first.0) }, unsafe {
        mimi_decode_state_bytes(second.0)
    });
    eprintln!(
        "[mimi-image] bound={bound} bytes, derived={} bytes, state={} bytes/conversation, compatibility=0",
        derived,
        unsafe { mimi_decode_state_bytes(first.0) }
    );

    let mut pcm = vec![0.0f32; 3840];
    let first_codes = (0..CODEBOOKS as u32)
        .map(|codebook| (codebook * 257) % 2048)
        .collect::<Vec<_>>();
    let first_samples =
        unsafe { mimi_decode_state_step(first.0, first_codes.as_ptr(), pcm.as_mut_ptr()) };
    let mut peer = vec![0.0f32; 3840];
    let peer_samples =
        unsafe { mimi_decode_state_step(second.0, first_codes.as_ptr(), peer.as_mut_ptr()) };
    assert_eq!((first_samples, peer_samples), (1920, 1920));
    assert_eq!(&pcm[..1920], &peer[..1920]);

    for frame in 0..2u32 {
        let codes = (0..CODEBOOKS as u32)
            .map(|codebook| (frame * 173 + codebook * 257) % 2048)
            .collect::<Vec<_>>();
        let samples = unsafe { mimi_decode_state_step(first.0, codes.as_ptr(), pcm.as_mut_ptr()) };
        assert_eq!(samples, 1920);
        assert!(pcm[..samples as usize]
            .iter()
            .all(|sample| sample.is_finite()));
        assert_eq!(
            unsafe { mimi_decode_plan_bound_weight_bytes(plan.0) },
            bound
        );
        assert_eq!(unsafe { mimi_decode_plan_derived_bytes(plan.0) }, derived);
        assert_eq!(
            unsafe { mimi_decode_plan_compatibility_copied_bytes(plan.0) },
            0
        );
    }
}

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
