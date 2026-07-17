//! Product-header boundary gate.
//!
//! The exported lifecycle/session surface must stay opaque even while the
//! offline oracle continues to link the transitional numerical symbols.

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
