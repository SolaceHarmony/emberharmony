use std::ffi::{c_char, c_void};

pub(crate) const ABI: u32 = 3;
pub(crate) const SAMPLE_ABI: u32 = 1;
pub(crate) const SAMPLE_GREEDY: u32 = 1;
pub(crate) const SEED_SYSTEM: u32 = 1;
pub(crate) const MODEL_CAP_DEPTHFORMER: u32 = 1;
pub(crate) const MODEL_CAP_FRONTEND: u32 = 2;
pub(crate) const MODEL_CAP_CONFORMER: u32 = 4;
pub(crate) const MODEL_CAP_MIMI: u32 = 8;

#[repr(C)]
pub(crate) struct Model {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Conversation {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SamplerConfig {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) flags: u32,
    pub(crate) top_k: u32,
    pub(crate) temperature: f64,
    pub(crate) reserved: u64,
}

#[repr(C)]
pub(crate) struct ConversationConfig {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) flags: u32,
    pub(crate) reserved0: u32,
    pub(crate) seed: u64,
    pub(crate) text_sampler: SamplerConfig,
    pub(crate) audio_sampler: SamplerConfig,
    pub(crate) reserved: [u64; 4],
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct ModelInfo {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) resident_bytes: u64,
    pub(crate) plan_id: u64,
    pub(crate) depth_plan_id: u64,
    pub(crate) hidden: u32,
    pub(crate) ffn: u32,
    pub(crate) layers: u32,
    pub(crate) vocab: u32,
    pub(crate) max_context: u32,
    pub(crate) codebooks: u32,
    pub(crate) capabilities: u32,
    pub(crate) reserved: [u32; 5],
}

#[repr(C)]
#[derive(Default)]
pub(crate) struct ModelMemory {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) source_bytes: u64,
    pub(crate) resident_image_bytes: u64,
    pub(crate) directly_bound_bytes: u64,
    pub(crate) derived_immutable_bytes: u64,
    pub(crate) compatibility_copied_bytes: u64,
    pub(crate) load_ns: u64,
    pub(crate) load_workers: u32,
    pub(crate) load_tasks: u32,
    pub(crate) reserved: [u64; 4],
}

#[repr(C)]
pub(crate) struct TokenResult {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) position: u64,
    pub(crate) sampled_token: u32,
    pub(crate) input_count: u32,
    pub(crate) embedding_kind: u32,
    pub(crate) flags: u32,
    pub(crate) reserved: [u64; 4],
}

unsafe extern "C" {
        pub(crate) fn lfm_model_open(
        engine: *mut c_void,
        path: *const c_char,
        out: *mut *mut Model,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
        pub(crate) fn lfm_model_close(model: *mut Model) -> i32;
        pub(crate) fn lfm_model_info(model: *const Model, out: *mut ModelInfo) -> i32;
        pub(crate) fn lfm_model_memory(model: *const Model, out: *mut ModelMemory) -> i32;
        pub(crate) fn lfm_conversation_create(
        model: *mut Model,
        config: *const ConversationConfig,
        out: *mut *mut Conversation,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
        pub(crate) fn lfm_conversation_step(
        conversation: *mut Conversation,
        ids: *const u32,
        id_count: usize,
        embedding_kind: u32,
        out: *mut TokenResult,
    ) -> i32;
        pub(crate) fn lfm_conversation_prefill_audio(
        conversation: *mut Conversation,
        rows: *const u16,
        element_count: usize,
        out_position: *mut u64,
    ) -> i32;
        pub(crate) fn lfm_conversation_reset(conversation: *mut Conversation) -> i32;
        pub(crate) fn lfm_conversation_close(conversation: *mut Conversation) -> i32;
}
