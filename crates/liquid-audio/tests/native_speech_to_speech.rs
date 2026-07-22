#![cfg(target_os = "macos")]

use std::ffi::{c_char, CString};
use std::path::PathBuf;

use liquid_audio as _;

unsafe extern "C" {
    fn lfm_native_speech_to_speech_gate(
        model_path: *const c_char,
        audible: u32,
        kernel_lanes: u32,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

#[test]
#[ignore = "requires LFM_MODEL_DIR and runs two complete native speech conversations"]
fn two_native_lfm2_agents_speak_through_memory_only() {
    let path = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .expect("LFM_MODEL_DIR must name the complete native LFM2 checkpoint"),
    );
    let path = CString::new(path.as_os_str().as_encoded_bytes())
        .expect("native model path contains a NUL byte");
    let audible = match std::env::var_os("LFM_SPEECH_GATE_AUDIBLE") {
        None => 0,
        Some(value) if value == "stream" => 2,
        Some(_) => 1,
    };
    let lanes = std::env::var("LFM_SPEECH_GATE_LANES")
        .map(|value| {
            value
                .parse::<u32>()
                .expect("LFM_SPEECH_GATE_LANES must be a positive integer")
        })
        .unwrap_or(8);
    let mut error = [0i8; 512];
    let status = unsafe {
        lfm_native_speech_to_speech_gate(
            path.as_ptr(),
            audible,
            lanes,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    let end = error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(error.len());
    let message = String::from_utf8_lossy(
        &error[..end]
            .iter()
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned();
    assert_eq!(status, 0, "{message}");
}
