#![cfg(target_os = "macos")]

use std::ffi::{c_char, CString};
use std::path::PathBuf;

use liquid_audio as _;

unsafe extern "C" {
    fn lfm_native_inner_voice_probe_gate(
        model_path: *const c_char,
        probe_dir: *const c_char,
        item_filter: *const c_char,
        kernel_lanes: u32,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

#[test]
#[ignore = "requires LFM_MODEL_DIR, LFM_PROBE_DIR, and runs twelve native listening probes"]
fn inner_voice_listening_probe_reads_text_head_per_audio_row() {
    let model = PathBuf::from(
        std::env::var_os("LFM_MODEL_DIR")
            .expect("LFM_MODEL_DIR must name the complete native LFM2 checkpoint"),
    );
    let model = CString::new(model.as_os_str().as_encoded_bytes())
        .expect("native model path contains a NUL byte");
    let probe_dir = PathBuf::from(
        std::env::var_os("LFM_PROBE_DIR")
            .expect("LFM_PROBE_DIR must name the synthesized probe dataset directory"),
    );
    let probe_dir = CString::new(probe_dir.as_os_str().as_encoded_bytes())
        .expect("probe dataset path contains a NUL byte");
    let item_filter = std::env::var_os("LFM_PROBE_ITEM").map(|item| {
        CString::new(item.as_encoded_bytes()).expect("probe item filter contains a NUL byte")
    });
    let lanes = std::env::var("LFM_SPEECH_GATE_LANES")
        .map(|value| {
            value
                .parse::<u32>()
                .expect("LFM_SPEECH_GATE_LANES must be a positive integer")
        })
        .unwrap_or(8);
    let mut error = [0i8; 512];
    let status = unsafe {
        lfm_native_inner_voice_probe_gate(
            model.as_ptr(),
            probe_dir.as_ptr(),
            item_filter
                .as_ref()
                .map_or(std::ptr::null(), |item| item.as_ptr()),
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
