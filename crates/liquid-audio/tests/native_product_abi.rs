//! Product-header boundary gate.
//!
//! The exported lifecycle/session surface stays opaque, and deleted direct
//! numerical entry points must not survive as hidden compatibility symbols.

#[cfg(target_os = "macos")]
use std::collections::BTreeSet;
#[cfg(target_os = "macos")]
use std::path::Path;
#[cfg(target_os = "macos")]
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

const DETOKENIZER_CPP: &str = include_str!("../native/src/detokenizer/lfm_detokenizer.cpp");
const DETOKENIZER_AARCH64: &str = include_str!("../native/kernels/aarch64/flashkern_detokenizer.S");
const DETOKENIZER_X86_64: &str = include_str!("../native/kernels/x86_64/flashkern_detokenizer.S");

#[cfg(target_os = "macos")]
const PRODUCT_SYMBOLS: [&str; 40] = [
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
    "lfm_session_interrupt",
    "lfm_session_host_capacity",
    "lfm_session_request_stop",
    "lfm_session_join",
    "lfm_session_snapshot",
    "lfm_session_destroy",
    "lfm_capture_chunk_producer_create",
    "lfm_capture_producer_claim_chunk",
    "lfm_capture_producer_resolve_chunk",
    "lfm_capture_producer_commit_chunk",
    "lfm_capture_producer_write_interleaved",
    "lfm_capture_producer_abort_chunk",
    "lfm_capture_producer_publish_gap",
    "lfm_capture_producer_destroy",
    "lfm_playback_consumer_create",
    "lfm_playback_consumer_claim",
    "lfm_playback_consumer_render_f32",
    "lfm_playback_consumer_render_i16",
    "lfm_playback_consumer_render_u16",
    "lfm_playback_consumer_observe",
    "lfm_playback_consumer_release",
    "lfm_playback_consumer_destroy",
    "lfm_session_playback_policy_snapshot",
    "lfm_session_control_create",
    "lfm_session_control_interrupt",
    "lfm_session_control_destroy",
];

#[cfg(target_os = "macos")]
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
    fn lfm_session_interrupt();
    fn lfm_session_host_capacity();
    fn lfm_session_request_stop();
    fn lfm_session_join();
    fn lfm_session_snapshot();
    fn lfm_session_destroy();
    fn lfm_capture_chunk_producer_create();
    fn lfm_capture_producer_claim_chunk();
    fn lfm_capture_producer_resolve_chunk();
    fn lfm_capture_producer_commit_chunk();
    fn lfm_capture_producer_write_interleaved();
    fn lfm_capture_producer_abort_chunk();
    fn lfm_capture_producer_publish_gap();
    fn lfm_capture_producer_destroy();
    fn lfm_playback_consumer_create();
    fn lfm_playback_consumer_claim();
    fn lfm_playback_consumer_render_f32();
    fn lfm_playback_consumer_render_i16();
    fn lfm_playback_consumer_render_u16();
    fn lfm_playback_consumer_observe();
    fn lfm_playback_consumer_release();
    fn lfm_playback_consumer_destroy();
    fn lfm_session_playback_policy_snapshot();
    fn lfm_session_control_create();
    fn lfm_session_control_interrupt();
    fn lfm_session_control_destroy();
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
fn payload_reader_is_a_private_native_construction_seam() {
    let private = include_str!("../native/src/model/lfm_payload_reader.h");
    for term in [
        "LfmPayloadReadOwner",
        "LfmPayloadReadScope",
        "lfm_weights_open_owned",
        "lfm_weights_open_bundle_owned",
        "lfm_tokenizer_open_owned",
    ] {
        assert!(
            private.contains(term),
            "private source ledger lost `{term}`"
        );
        for (name, header) in PRODUCT {
            assert!(
                !header.contains(term),
                "product header {name} exposed private source ledger `{term}`"
            );
        }
    }
}

#[test]
fn native_owner_header_contains_lifecycle_but_no_direct_numerical_abi() {
    let internal = include_str!("../native/src/model/lfm_model_internal.h");
    for symbol in [
        "LfmModelInfoV1",
        "lfm_model_info(",
        "lfm_conversation_create(",
        "lfm_conversation_close(",
    ] {
        assert!(
            internal.contains(symbol),
            "native owner ABI lost `{symbol}`"
        );
    }
    for symbol in [
        "lfm_conversation_step(",
        "lfm_conversation_prefill(",
        "lfm_conversation_prefill_audio(",
        "lfm_conversation_prefill_pcm_f32(",
        "lfm_conversation_audio_frame(",
        "lfm_engine_audio_encode(",
    ] {
        assert!(
            !internal.contains(symbol),
            "deleted direct numerical ABI returned as `{symbol}`"
        );
    }
}

#[test]
fn detokenizer_payload_math_is_owned_by_paired_assembly() {
    for forbidden in [
        "<arm_neon.h>",
        "<immintrin.h>",
        "std::pow(",
        "std::acos(",
        "std::cos(",
        "std::sin(",
        "1.0f / sum",
        "static_cast<float>(position)",
        "lfm_detok_dot32_f32",
        "lfm_detok_scale_f32",
    ] {
        assert!(
            !DETOKENIZER_CPP.contains(forbidden),
            "detokenizer C++ regained numerical implementation `{forbidden}`"
        );
    }

    for seam in ["cblas_sgemm(", "vvexpf(", "vvsincosf("] {
        assert!(
            DETOKENIZER_CPP.contains(seam),
            "declared Accelerate/vForce seam `{seam}` disappeared"
        );
    }

    for symbol in [
        "lfm_detok_copy_f32",
        "lfm_detok_add_f32",
        "lfm_detok_embed_f32",
        "lfm_detok_rms_f32",
        "lfm_detok_swiglu_f32",
        "lfm_detok_dot32_scaled_f32",
        "lfm_detok_max_f32",
        "lfm_detok_subtract_f32",
        "lfm_detok_sum_f32",
        "lfm_detok_normalize_f32",
        "lfm_detok_rope_angles_f32",
        "lfm_detok_rope_f32",
        "lfm_detok_rope_inverse_f32",
        "lfm_detok_ifft_basis_f32",
        "lfm_detok_conv_f32",
        "lfm_detok_weighted_f32",
        "lfm_detok_polar_f32",
        "lfm_detok_overlap_f32",
        "lfm_detok_emit_f32",
    ] {
        assert!(
            DETOKENIZER_AARCH64.contains(symbol),
            "AArch64 detokenizer assembly lost `{symbol}`"
        );
        assert!(
            DETOKENIZER_X86_64.contains(symbol),
            "x86_64 detokenizer assembly lost `{symbol}`"
        );
    }
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
fn symbol_line<'a>(symbols: &'a str, symbol: &str) -> &'a str {
    let suffix = format!(" _{symbol}");
    symbols
        .lines()
        .find(|line| line.ends_with(&suffix) && !line.contains("(undefined)"))
        .unwrap_or_else(|| panic!("archive did not define `{symbol}`"))
}

#[cfg(target_os = "macos")]
fn defines_symbol(symbols: &str, symbol: &str) -> bool {
    let suffix = format!(" _{symbol}");
    symbols
        .lines()
        .any(|line| line.ends_with(&suffix) && !line.contains("(undefined)"))
}

#[cfg(target_os = "macos")]
fn references_symbol(symbols: &str, symbol: &str) -> bool {
    let suffix = format!(" _{symbol}");
    symbols
        .lines()
        .any(|line| line.ends_with(&suffix) && line.contains("(undefined)"))
}

#[cfg(target_os = "macos")]
fn default_native_definitions(symbols: &str) -> BTreeSet<String> {
    symbols
        .lines()
        .filter(|line| !line.contains("(undefined)"))
        .filter(|line| line.contains(") external _"))
        .filter(|line| !line.contains("private external"))
        .filter_map(|line| line.rsplit_once(" _").map(|(_, symbol)| symbol))
        .filter(|symbol| symbol.starts_with("lfm_"))
        .map(str::to_owned)
        .collect()
}

#[cfg(target_os = "macos")]
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
        lfm_session_interrupt as usize,
        lfm_session_host_capacity as usize,
        lfm_session_request_stop as usize,
        lfm_session_join as usize,
        lfm_session_snapshot as usize,
        lfm_session_destroy as usize,
        lfm_capture_chunk_producer_create as usize,
        lfm_capture_producer_claim_chunk as usize,
        lfm_capture_producer_resolve_chunk as usize,
        lfm_capture_producer_commit_chunk as usize,
        lfm_capture_producer_write_interleaved as usize,
        lfm_capture_producer_abort_chunk as usize,
        lfm_capture_producer_publish_gap as usize,
        lfm_capture_producer_destroy as usize,
        lfm_playback_consumer_create as usize,
        lfm_playback_consumer_claim as usize,
        lfm_playback_consumer_render_f32 as usize,
        lfm_playback_consumer_render_i16 as usize,
        lfm_playback_consumer_render_u16 as usize,
        lfm_playback_consumer_observe as usize,
        lfm_playback_consumer_release as usize,
        lfm_playback_consumer_destroy as usize,
        lfm_session_playback_policy_snapshot as usize,
        lfm_session_control_create as usize,
        lfm_session_control_interrupt as usize,
        lfm_session_control_destroy as usize,
    ]);
}

#[test]
#[cfg(target_os = "macos")]
fn production_archives_keep_only_native_owner_lifecycle_private_external() {
    let build = include_str!("../build.rs");
    for deleted in [
        "lfm_kernel_bridge",
        "lfm_kernel_protocol_c",
    ] {
        assert!(
            !build.contains(deleted),
            "deleted in-process bridge target `{deleted}` remains in the build"
        );
    }

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
    for symbol in ["lfm_conformer_create", "lfm_conformer_out_rows"] {
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

    let detokenizer = archive_symbols("liblfm_detokenizer.a");
    let detokenizer_kernels = archive_symbols(if cfg!(target_arch = "aarch64") {
        "liblfm_flashkern_neon.a"
    } else {
        "liblfm_flashkern_x86.a"
    });
    for symbol in [
        "lfm_detokenizer_plan_new_from_image",
        "lfm_detokenizer_state_new",
        "lfm_detokenizer_state_reset",
        "lfm_detokenizer_program_begin",
        "lfm_detokenizer_program_run",
        "lfm_detokenizer_program_advance",
        "lfm_detokenizer_program_cancel",
    ] {
        assert!(
            symbol_line(&detokenizer, symbol).contains("private external"),
            "detokenizer numerical seam `{symbol}` is a default-visible export"
        );
    }
    for symbol in [
        "lfm_detok_copy_f32",
        "lfm_detok_add_f32",
        "lfm_detok_embed_f32",
        "lfm_detok_rms_f32",
        "lfm_detok_swiglu_f32",
        "lfm_detok_dot32_scaled_f32",
        "lfm_detok_max_f32",
        "lfm_detok_subtract_f32",
        "lfm_detok_sum_f32",
        "lfm_detok_normalize_f32",
        "lfm_detok_rope_angles_f32",
        "lfm_detok_rope_f32",
        "lfm_detok_rope_inverse_f32",
        "lfm_detok_ifft_basis_f32",
        "lfm_detok_conv_f32",
        "lfm_detok_weighted_f32",
        "lfm_detok_polar_f32",
        "lfm_detok_overlap_f32",
        "lfm_detok_emit_f32",
    ] {
        assert!(
            symbol_line(&detokenizer_kernels, symbol).contains("private external"),
            "detokenizer assembly leaf `{symbol}` is not private external"
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
            "future Moshi Mimi seam `{symbol}` is a default-visible export"
        );
    }

    let model = archive_symbols("liblfm_flashkern_engine.a");
    for symbol in [
        "lfm_detokenizer_program_begin",
        "lfm_detokenizer_program_run",
        "lfm_detokenizer_program_advance",
        "lfm_detokenizer_program_cancel",
    ] {
        assert!(
            references_symbol(&model, symbol),
            "released LFM2.5 archive does not mount detokenizer stage `{symbol}`"
        );
    }
    for symbol in ["lfm_detokenizer_state_step", "lfm_detokenizer_state_flush"] {
        assert!(
            !detokenizer.lines().any(|line| line.contains(symbol)),
            "retired whole-graph detokenizer seam `{symbol}` remains linked"
        );
    }
    for symbol in [
        "mimi_decode_plan_new_from_image",
        "mimi_decode_state_new",
        "mimi_decode_state_step",
        "mimi_decode_state_reset",
    ] {
        assert!(
            !references_symbol(&model, symbol),
            "released LFM2.5 archive still references future-Moshi seam `{symbol}`"
        );
    }
    for symbol in [
        "lfm_model_open",
        "lfm_model_info",
        "lfm_model_memory",
        "lfm_conversation_create",
        "lfm_conversation_reset",
        "lfm_conversation_close",
    ] {
        assert!(
            symbol_line(&model, symbol).contains("private external"),
            "native owner lifecycle `{symbol}` is a default-visible export"
        );
    }
    for symbol in [
        "lfm_conversation_step",
        "lfm_conversation_prefill",
        "lfm_conversation_prefill_audio",
        "lfm_conversation_prefill_pcm_f32",
        "lfm_conversation_audio_frame",
        "lfm_engine_audio_encode",
    ] {
        assert!(
            !defines_symbol(&model, symbol),
            "deleted direct numerical seam `{symbol}` remains in the archive"
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
        "liblfm_frontend.a",
        "liblfm_conformer.a",
        "liblfm_flashkern_prng.a",
        if cfg!(target_arch = "aarch64") {
            "liblfm_flashkern_neon.a"
        } else {
            "liblfm_flashkern_x86.a"
        },
        "liblfm_detokenizer.a",
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
#[cfg(target_os = "macos")]
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
