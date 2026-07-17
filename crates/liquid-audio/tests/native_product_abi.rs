//! Product-header boundary gate.
//!
//! The exported lifecycle/session surface must stay opaque even while the
//! offline oracle continues to link the transitional numerical symbols.

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
use std::collections::BTreeSet;
#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
use std::path::Path;
#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
use std::process::Command;

const PRODUCT: [(&str, &str); 4] = [
    ("lfm_types.h", include_str!("../native/include/lfm_types.h")),
    (
        "lfm_runtime.h",
        include_str!("../native/include/lfm_runtime.h"),
    ),
    (
        "lfm_session.h",
        include_str!("../native/include/lfm_session.h"),
    ),
    ("lfm_model.h", include_str!("../native/include/lfm_model.h")),
];

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
const PRODUCT_SYMBOLS: [&str; 29] = [
    "lfm_runtime_create",
    "lfm_runtime_start",
    "lfm_runtime_request_stop",
    "lfm_runtime_join",
    "lfm_runtime_snapshot",
    "lfm_runtime_destroy",
    "lfm_runtime_model_open",
    "lfm_runtime_model_memory",
    "lfm_runtime_model_close",
    "lfm_runtime_conversation_create",
    "lfm_runtime_conversation_close",
    "lfm_session_create",
    "lfm_session_start",
    "lfm_session_submit_text",
    "lfm_session_wait_submit_text",
    "lfm_session_interrupt",
    "lfm_session_request_stop",
    "lfm_session_join",
    "lfm_session_snapshot",
    "lfm_session_destroy",
    "lfm_audio_dock_reserve",
    "lfm_audio_dock_wait_reserve",
    "lfm_session_submit_mixed",
    "lfm_session_wait_submit_mixed",
    "lfm_audio_dock_resolve_mut",
    "lfm_audio_dock_resolve",
    "lfm_audio_dock_publish",
    "lfm_audio_dock_wait_playback",
    "lfm_audio_dock_release",
];

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
unsafe extern "C" {
    fn lfm_runtime_create();
    fn lfm_runtime_start();
    fn lfm_runtime_request_stop();
    fn lfm_runtime_join();
    fn lfm_runtime_snapshot();
    fn lfm_runtime_destroy();
    fn lfm_runtime_model_open();
    fn lfm_runtime_model_memory();
    fn lfm_runtime_model_close();
    fn lfm_runtime_conversation_create();
    fn lfm_runtime_conversation_close();
    fn lfm_session_create();
    fn lfm_session_start();
    fn lfm_session_submit_text();
    fn lfm_session_wait_submit_text();
    fn lfm_session_interrupt();
    fn lfm_session_request_stop();
    fn lfm_session_join();
    fn lfm_session_snapshot();
    fn lfm_session_destroy();
    fn lfm_audio_dock_reserve();
    fn lfm_audio_dock_wait_reserve();
    fn lfm_session_submit_mixed();
    fn lfm_session_wait_submit_mixed();
    fn lfm_audio_dock_resolve_mut();
    fn lfm_audio_dock_resolve();
    fn lfm_audio_dock_publish();
    fn lfm_audio_dock_wait_playback();
    fn lfm_audio_dock_release();
}

#[test]
fn product_headers_expose_only_opaque_model_lifecycle() {
    let forbidden = [
        "LfmConversationConfigV1",
        "LfmInputV1",
        "LfmTokenResultV1",
        "LfmAudioResultV1",
        "LfmModelInfoV1",
        "lfm_model_open(",
        "lfm_model_close(",
        "lfm_model_info(",
        "lfm_model_memory(",
        "lfm_conversation_create(",
        "lfm_conversation_step(",
        "lfm_conversation_prefill",
        "lfm_conversation_audio_frame(",
        "lfm_conversation_reset(",
        "lfm_conversation_close(",
        "sampled_token",
        "tensor_name",
        "mel_rows",
        "logits",
        "codec_codes",
        "kv_state",
        "uint16_t *rows",
        "float *pcm",
    ];

    for (name, header) in PRODUCT {
        for term in forbidden {
            assert!(
                !header.contains(term),
                "product header {name} exposes forbidden numerical seam `{term}`"
            );
        }
    }
}

#[test]
fn product_headers_include_runtime_scoped_conversation_lifecycle() {
    let runtime = include_str!("../native/include/lfm_runtime.h");
    for term in [
        "LfmSamplingPolicyV1",
        "LfmConversationOptionsV1",
        "lfm_runtime_conversation_create(",
        "lfm_runtime_conversation_close(",
    ] {
        assert!(runtime.contains(term), "product ABI lost `{term}`");
    }
}

#[test]
fn transitional_numerical_abi_is_source_private() {
    let legacy = include_str!("../native/src/model/lfm_model_legacy.h");
    for symbol in [
        "LfmModelInfoV1",
        "lfm_model_info(",
        "lfm_conversation_step(",
        "lfm_conversation_prefill_audio(",
        "lfm_conversation_audio_frame(",
    ] {
        assert!(legacy.contains(symbol), "legacy ABI lost `{symbol}`");
    }
}

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
fn archive_symbols(name: &str) -> String {
    let path = Path::new(env!("LFM_NATIVE_ARCHIVE_DIR")).join(name);
    let output = Command::new("nm")
        .arg("-m")
        .arg(&path)
        .output()
        .unwrap_or_else(|error| panic!("could not inspect {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "nm failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap_or_else(|error| {
        panic!(
            "nm emitted non-UTF-8 output for {}: {error}",
            path.display()
        )
    })
}

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
fn symbol_line<'a>(symbols: &'a str, symbol: &str) -> &'a str {
    let suffix = format!(" _{symbol}");
    symbols
        .lines()
        .find(|line| line.ends_with(&suffix) && !line.contains("(undefined)"))
        .unwrap_or_else(|| panic!("archive did not define `{symbol}`"))
}

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
fn default_native_definitions(symbols: &str) -> BTreeSet<String> {
    symbols
        .lines()
        .filter(|line| !line.contains("(undefined)"))
        .filter(|line| line.contains(") external _"))
        .filter(|line| !line.contains("private external"))
        .filter_map(|line| line.rsplit_once(" _").map(|(_, symbol)| symbol))
        .filter(|symbol| symbol.starts_with("lfm_") || symbol.starts_with("mimi_"))
        .map(str::to_owned)
        .collect()
}

#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
#[allow(function_casts_as_integer)]
fn retain_product_surface() {
    // Retain the Rust owner rim so Cargo propagates this crate's native archive
    // link set into the integration-test artifact before the C roots below are
    // forced live.
    std::hint::black_box(liquid_audio::NativeVoiceModel::open);
    std::hint::black_box([
        lfm_runtime_create as usize,
        lfm_runtime_start as usize,
        lfm_runtime_request_stop as usize,
        lfm_runtime_join as usize,
        lfm_runtime_snapshot as usize,
        lfm_runtime_destroy as usize,
        lfm_runtime_model_open as usize,
        lfm_runtime_model_memory as usize,
        lfm_runtime_model_close as usize,
        lfm_runtime_conversation_create as usize,
        lfm_runtime_conversation_close as usize,
        lfm_session_create as usize,
        lfm_session_start as usize,
        lfm_session_submit_text as usize,
        lfm_session_wait_submit_text as usize,
        lfm_session_interrupt as usize,
        lfm_session_request_stop as usize,
        lfm_session_join as usize,
        lfm_session_snapshot as usize,
        lfm_session_destroy as usize,
        lfm_audio_dock_reserve as usize,
        lfm_audio_dock_wait_reserve as usize,
        lfm_session_submit_mixed as usize,
        lfm_session_wait_submit_mixed as usize,
        lfm_audio_dock_resolve_mut as usize,
        lfm_audio_dock_resolve as usize,
        lfm_audio_dock_publish as usize,
        lfm_audio_dock_wait_playback as usize,
        lfm_audio_dock_release as usize,
    ]);
}

#[test]
#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
fn production_archives_keep_numerical_seams_private_external() {
    let weights = archive_symbols("liblfm_safetensors.a");
    for symbol in [
        "lfm_weights_open",
        "lfm_weights_open_files",
        "lfm_weights_open_bundle",
        "lfm_weights_data",
        "lfm_weights_count",
        "lfm_weights_at",
        "lfm_weights_find",
        "lfm_weights_at_component",
        "lfm_weights_find_component",
    ] {
        assert!(
            symbol_line(&weights, symbol).contains("private external"),
            "raw checkpoint seam `{symbol}` is a default-visible export"
        );
    }

    let conformer = archive_symbols("liblfm_conformer.a");
    for symbol in [
        "lfm_conformer_create",
        "lfm_conformer_out_rows",
        "lfm_conformer_forward",
        "lfm_conformer_forward_engine_team",
    ] {
        assert!(
            symbol_line(&conformer, symbol).contains("private external"),
            "direct Conformer seam `{symbol}` is a default-visible export"
        );
    }

    let frontend = archive_symbols("liblfm_frontend.a");
    for symbol in [
        "lfm_frontend_create",
        "lfm_frontend_forward",
        "lfm_frontend_forward_valid",
        "lfm_frontend_forward_bf16_workspace",
        "lfm_resampler_process",
        "lfm_resample_f32",
    ] {
        assert!(
            symbol_line(&frontend, symbol).contains("private external"),
            "frontend numerical seam `{symbol}` is a default-visible export"
        );
    }

    let mimi = archive_symbols("liblfm_mimi.a");
    for symbol in [
        "mimi_decode_plan_new_from_image",
        "mimi_decode_state_new",
        "mimi_decode_state_step",
        "mimi_decode_state_reset",
    ] {
        assert!(
            symbol_line(&mimi, symbol).contains("private external"),
            "Mimi numerical seam `{symbol}` is a default-visible export"
        );
    }

    let model = archive_symbols("liblfm_flashkern_engine.a");
    for symbol in [
        "lfm_model_open",
        "lfm_model_info",
        "lfm_model_memory",
        "lfm_conversation_create",
        "lfm_conversation_step",
        "lfm_conversation_prefill",
        "lfm_conversation_prefill_audio",
        "lfm_conversation_prefill_pcm_f32",
        "lfm_conversation_audio_frame",
        "lfm_conversation_reset",
        "lfm_conversation_close",
        "lfm_engine_audio_encode",
    ] {
        assert!(
            symbol_line(&model, symbol).contains("private external"),
            "transitional numerical seam `{symbol}` is a default-visible export"
        );
    }

    let product = archive_symbols("liblfm_voice_session.a");
    for symbol in PRODUCT_SYMBOLS {
        let line = symbol_line(&product, symbol);
        assert!(
            line.contains(") external") && !line.contains("private external"),
            "product lifecycle symbol `{symbol}` is not default-visible: {line}"
        );
    }

    let internal = [
        "liblfm_voice_protocol_c.a",
        "liblfm_flashkern_engine.a",
        "liblfm_kernel_bridge.a",
        "liblfm_kernel_protocol_c.a",
        "liblfm_frontend.a",
        "liblfm_conformer.a",
        "liblfm_flashkern_prng.a",
        if cfg!(target_arch = "aarch64") {
            "liblfm_flashkern_neon.a"
        } else {
            "liblfm_flashkern_x86.a"
        },
        "liblfm_mimi.a",
        "liblfm_safetensors.a",
    ];
    for archive in internal {
        let exports = default_native_definitions(&archive_symbols(archive));
        assert!(
            exports.is_empty(),
            "private production archive {archive} exports numerical symbols: {exports:?}"
        );
    }
}

#[test]
#[cfg(all(not(feature = "oracle"), target_os = "macos"))]
fn linked_product_exports_exact_lifecycle_allowlist() {
    retain_product_surface();
    let executable = std::env::current_exe().expect("test executable path");
    let output = Command::new("nm")
        .arg("-m")
        .arg(&executable)
        .output()
        .unwrap_or_else(|error| panic!("could not inspect {}: {error}", executable.display()));
    assert!(
        output.status.success(),
        "nm failed for {}: {}",
        executable.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let symbols = String::from_utf8(output.stdout).expect("nm emitted UTF-8");
    let actual = default_native_definitions(&symbols);
    let expected = PRODUCT_SYMBOLS
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "linked product exposes a numerical native ABI or lost lifecycle ABI"
    );
}
