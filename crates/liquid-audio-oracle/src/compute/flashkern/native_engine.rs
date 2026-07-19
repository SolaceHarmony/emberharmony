//! The Rust rim of the resident native decode engine (native/src/engine/flashkern_engine.cpp).
//!
//! Everything below the ABI line is native: C++ owns plans, lifetimes, and pass
//! scheduling, while architecture-specific assembly owns model numerics. Rust is not
//! an inference scheduler; this temporary rim disappears when callers dock PCM leases
//! through the native audio-session ABI.

use std::ffi::c_void;
use std::sync::Mutex;

/// Mirror of the C `LfmLayerDesc` (flashkern_engine.cpp) — one per backbone block,
/// indexed by block_idx. Field order/types must match the C struct exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerDesc {
    pub kind: u32, // 0 = shortconv+mlp, 1 = attention
    pub k: u32,
    pub op_eps: f32,
    pub ffn_eps: f32,
    pub op_norm_w: *const u16,
    pub ffn_norm_w: *const u16,
    pub in_w: *const u16,
    pub conv_w: *const u16,
    pub out_w: *const u16,
    pub w1: *const u16,
    pub w3: *const u16,
    pub w2: *const u16,
    // Attention fields (kind 1); q_w null ⇒ slot unserved (conv layers still run).
    pub n_head: u32,
    pub n_kv: u32,
    pub hd: u32,
    pub qk_eps: f32,
    pub q_w: *const u16,
    pub k_w: *const u16,
    pub v_w: *const u16,
    pub o_w: *const u16,
    pub qn_w: *const u16,
    pub kn_w: *const u16,
}

impl LayerDesc {
    /// An attention-slot placeholder (kind 1, everything null) — keeps the table
    /// indexed by block_idx.
    pub fn attn_placeholder() -> Self {
        Self {
            kind: 1,
            k: 0,
            op_eps: 0.0,
            ffn_eps: 0.0,
            op_norm_w: std::ptr::null(),
            ffn_norm_w: std::ptr::null(),
            in_w: std::ptr::null(),
            conv_w: std::ptr::null(),
            out_w: std::ptr::null(),
            w1: std::ptr::null(),
            w3: std::ptr::null(),
            w2: std::ptr::null(),
            n_head: 0,
            n_kv: 0,
            hd: 0,
            qk_eps: 0.0,
            q_w: std::ptr::null(),
            k_w: std::ptr::null(),
            v_w: std::ptr::null(),
            o_w: std::ptr::null(),
            qn_w: std::ptr::null(),
            kn_w: std::ptr::null(),
        }
    }
}

/// Mirror of the C `LfmLayerState` — per-layer per-generation state for the token
/// pass. Pointers are captured fresh each token AFTER capacity is ensured.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerState {
    pub k_plane: *mut u16,
    pub v_plane: *mut u16,
    pub head_stride: usize,
    pub k_len: usize,
    pub v_len: usize,
    pub conv_state: *mut u16,
    pub conv_len: usize,
}

/// Snapshot-stable native ChaCha20 stream state. This belongs to one conversation,
/// never to the process engine; native passes borrow it mutably for their duration.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct PrngState {
    size: u32,
    abi_version: u32,
    cursor: u32,
    flags: u32,
    core: [u32; 16],
    block: [u32; 16],
    reserved: [u8; 48],
}

const _: [(); 192] = [(); std::mem::size_of::<PrngState>()];
const _: [(); 64] = [(); std::mem::align_of::<PrngState>()];

impl PrngState {
    pub(crate) fn from_seed(seed: u64) -> Result<Self, i32> {
        let mut state = std::mem::MaybeUninit::<Self>::zeroed();
        // SAFETY: native code initializes the complete aligned ABI object.
        let rc = unsafe { lfm_prng_seed_u64(state.as_mut_ptr(), seed) };
        if rc != 0 {
            return Err(rc);
        }
        // SAFETY: success initialized every byte.
        Ok(unsafe { state.assume_init() })
    }

    #[cfg(test)]
    pub(crate) fn from_system() -> Result<Self, i32> {
        let mut state = std::mem::MaybeUninit::<Self>::zeroed();
        // SAFETY: `state` is aligned storage for the complete C ABI object.
        let rc = unsafe { lfm_prng_seed_system(state.as_mut_ptr()) };
        if rc != 0 {
            return Err(rc);
        }
        // SAFETY: success initializes every byte of the state.
        Ok(unsafe { state.assume_init() })
    }

    #[cfg(test)]
    fn from_material(key: &[u8; 32], nonce: &[u8; 8]) -> Result<Self, i32> {
        let mut state = std::mem::MaybeUninit::<Self>::zeroed();
        // SAFETY: all pointers name their fixed ABI extents for the call.
        let rc =
            unsafe { lfm_prng_seed_material(state.as_mut_ptr(), key.as_ptr(), nonce.as_ptr()) };
        if rc != 0 {
            return Err(rc);
        }
        // SAFETY: success initializes every byte of the state.
        Ok(unsafe { state.assume_init() })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SampleConfig {
    size: u32,
    abi_version: u32,
    flags: u32,
    top_k: u32,
    temperature: f64,
    reserved: u64,
}

const _: [(); 32] = [(); std::mem::size_of::<SampleConfig>()];

impl SampleConfig {
    pub(crate) fn new(temperature: Option<f64>, top_k: Option<usize>) -> Self {
        const GREEDY: u32 = 1;
        let greedy = temperature.is_none_or(|value| value <= 0.0) || top_k == Some(1);
        Self {
            size: std::mem::size_of::<Self>() as u32,
            abi_version: 1,
            flags: if greedy { GREEDY } else { 0 },
            top_k: top_k.unwrap_or(0).min(u32::MAX as usize) as u32,
            temperature: temperature.unwrap_or(1.0),
            reserved: 0,
        }
    }
}

impl LayerState {
    pub fn none() -> Self {
        Self {
            k_plane: std::ptr::null_mut(),
            v_plane: std::ptr::null_mut(),
            head_stride: 0,
            k_len: 0,
            v_len: 0,
            conv_state: std::ptr::null_mut(),
            conv_len: 0,
        }
    }
}

#[cfg(test)]
#[repr(C)]
#[derive(Default)]
struct EngineSnapshot {
    size: u32,
    abi_version: u32,
    pass_submissions: u64,
    pass_completions: u64,
    bridge_dispatches: u64,
    dispatch_wakes: u64,
    fence_wake_calls: u64,
    fence_wakes: u64,
    fence_generations: u64,
    descriptor_acquires: u64,
    descriptor_retains: u64,
    descriptor_releases: u64,
    descriptor_callbacks: u64,
    descriptor_capacity: u32,
    descriptors_live: u32,
    max_descriptor_generation: u32,
    attention_qkv_capacity: u32,
    attention_y_capacity: u32,
    attention_score_capacity: u32,
    pass_claimed: u32,
    bridge_capacity: u32,
    pass_slot_capacity: u32,
    pass_slots_live: u32,
    max_pass_slots_live: u32,
    continuation_submissions: u64,
    route_capacity: u32,
    routes_live: u32,
    routes_ready: u32,
    reserved0: u32,
    route_dispatches: u64,
    route_parks: u64,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
struct ContextWindow {
    capacity: u64,
    runway: u64,
    position: u64,
    start: u64,
    cursor: u64,
    rope_base: u64,
}

#[cfg(test)]
#[repr(C)]
struct TokenCommitRecord {
    window: *mut ContextWindow,
    expected_position: u64,
    expected_start: u64,
    expected_cursor: u64,
    expected_rope_base: u64,
    token_committed: *mut u32,
}

#[cfg(test)]
#[repr(C)]
#[derive(Default)]
struct AudioRouteResult {
    status: i32,
    token_completed: u32,
    token_committed: u32,
    depth_completed: u32,
    mimi_completed: u32,
    eoaudio: u32,
    reserved: u32,
    pcm_samples: usize,
    codes: [u32; 8],
}

#[cfg(test)]
#[repr(C)]
struct AudioRouteTarget {
    epoch: *const c_void,
    expected_epoch: u64,
    pcm: *mut f32,
    pcm_capacity: usize,
    codec_pcm: *mut f32,
    codec_pcm_capacity: usize,
    resampler_stream: *mut c_void,
}

#[cfg(test)]
#[repr(C)]
#[derive(Default)]
struct AudioRouteHandle {
    record: *mut c_void,
    generation: u64,
}

extern "C" {
    fn lfm_bf16_gemm_available() -> i32;
    #[cfg(test)]
    fn lfm_bf16_gemm_nt_f32(
        a: *const u16,
        weights: *const c_void,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    );
    #[cfg(test)]
    fn lfm_bf16_gemm_nt_f32_scalar(
        a: *const u16,
        weights: *const c_void,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    );
    #[cfg(test)]
    fn lfm_bf16_gemm_nt_bias_bf16(
        activation: *const u16,
        weights: *const c_void,
        bias: *const c_void,
        out: *mut u16,
        rows: i32,
        columns: i32,
        inner: i32,
        stride: i32,
    );
    #[cfg(test)]
    fn lfm_bf16_gemv_rne_bf16(
        input: *const c_void,
        weights: *const c_void,
        out: *mut u16,
        rows: usize,
        depth: usize,
    );
    #[cfg(test)]
    fn lfm_bf16_gemv_rne_add_bf16(
        input: *const c_void,
        weights: *const c_void,
        residual: *const c_void,
        out: *mut u16,
        rows: usize,
        depth: usize,
    );
    #[cfg(test)]
    fn lfm_bf16_gemv_pair_swiglu_bf16(
        input: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        out: *mut u16,
        rows: usize,
        depth: usize,
    );
    #[cfg(test)]
    fn lfm_swiglu_bf16(gate: *const f32, up: *const f32, out: *mut u16, count: i32);
    #[cfg(test)]
    fn lfm_rsqrt_size(value: usize) -> f32;
    #[cfg(test)]
    fn lfm_inv_rms_f32(sum: f32, count: usize, epsilon: f32) -> f32;
    #[cfg(test)]
    fn lfm_sum_f32(values: *const f32, count: usize) -> f32;
    #[cfg(test)]
    fn lfm_bf16_sumsq_stride_f32(
        values: *const u16,
        count: usize,
        start: usize,
        stride: usize,
    ) -> f32;
    #[cfg(test)]
    fn lfm_bf16_bias_add_f32(values: *mut f32, bias: *const c_void, count: usize);
    #[cfg(test)]
    fn lfm_bf16_copy_bytes(source: *const c_void, destination: *mut u16, count: usize);
    #[cfg(test)]
    fn lfm_bf16_rope_neox(values: *mut u16, cosine: *const u16, sine: *const u16, head_dim: usize);
    #[cfg(test)]
    fn lfm_sampler_exp_sum_f32(
        values: *const f32,
        weights: *mut f32,
        count: usize,
        scale: f32,
        maximum: f32,
        threshold: f32,
    ) -> f32;
    #[cfg(test)]
    fn lfm_sampler_exp_sum_bf16(
        values: *const u16,
        weights: *mut f32,
        count: usize,
        scale: u16,
        maximum: f32,
        threshold: f32,
    ) -> f32;
    fn lfm_depthwise_stream_bf16_available() -> i32;
    #[cfg(test)]
    fn lfm_prng_seed_system(state: *mut PrngState) -> i32;
    #[cfg(test)]
    fn lfm_prng_seed_material(state: *mut PrngState, key: *const u8, nonce: *const u8) -> i32;
    fn lfm_prng_seed_u64(state: *mut PrngState, seed: u64) -> i32;
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_free(e: *mut c_void);
    fn lfm_engine_request_stop(e: *mut c_void);
    fn lfm_ctx_build(
        e: *mut c_void,
        descs: *const LayerDesc,
        n_layers: usize,
        h: usize,
        ffn: usize,
        max_ctx: usize,
        out_id: *mut u64,
    ) -> i32;
    fn lfm_engine_attn_layer(
        e: *mut c_void,
        id: u64,
        layer: usize,
        x: *const u16,
        x_len: usize,
        k_plane: *mut u16,
        k_len: usize,
        v_plane: *mut u16,
        v_len: usize,
        head_stride: usize,
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out: *mut u16,
        out_len: usize,
        lanes: usize,
    ) -> i32;
    fn lfm_ctx_clear(e: *mut c_void, id: u64) -> i32;
    fn lfm_engine_lanes(e: *mut c_void) -> u32;
    #[cfg(test)]
    fn lfm_engine_snapshot(e: *mut c_void, out: *mut EngineSnapshot) -> i32;
    fn lfm_ctx_set_heads(
        e: *mut c_void,
        id: u64,
        embed_w: *const u16,
        embed_len: usize,
        vocab: usize,
        audio_embed_w: *const u16,
        audio_embed_len: usize,
        audio_rows: usize,
        emb_norm_w: *const u16,
        emb_norm_len: usize,
        emb_norm_eps: f32,
    ) -> i32;
    fn lfm_engine_token_pass(
        e: *mut c_void,
        id: u64,
        ids: *const u32,
        n_ids: usize,
        embed_kind: u32,
        states: *const LayerState,
        n_states: usize,
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out_hidden: *mut u16,
        out_hidden_len: usize,
        out_logits: *mut f32,
        out_logits_len: usize,
        sampler: *const SampleConfig,
        sample_state: *mut PrngState,
        out_token: *mut u32,
        lanes: usize,
        provided_embed: *const u16,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_audio_recurrence(
        e: *mut c_void,
        model_id: u64,
        depth_id: u64,
        ids: *const u32,
        id_count: usize,
        embedding_kind: u32,
        states: *const LayerState,
        state_count: usize,
        position: usize,
        rope_cos: *const u16,
        rope_sin: *const u16,
        rope_elements: usize,
        out_hidden: *mut u16,
        hidden_elements: usize,
        sampler: *const SampleConfig,
        prng: *mut PrngState,
        out_codes: *mut u32,
        code_count: usize,
        lanes: usize,
        commit: *const TokenCommitRecord,
        out_token_completed: *mut u32,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_audio_route_submit(
        e: *mut c_void,
        model_id: u64,
        depth_id: u64,
        ids: *const u32,
        id_count: usize,
        embedding_kind: u32,
        states: *const LayerState,
        state_count: usize,
        position: usize,
        rope_cos: *const u16,
        rope_sin: *const u16,
        rope_elements: usize,
        out_hidden: *mut u16,
        hidden_elements: usize,
        sampler: *const SampleConfig,
        prng: *mut PrngState,
        mimi: *mut c_void,
        target: *const AudioRouteTarget,
        result: *mut AudioRouteResult,
        lanes: usize,
        commit: *const TokenCommitRecord,
        notify: extern "C" fn(*mut c_void),
        notify_context: *mut c_void,
        out_handle: *mut AudioRouteHandle,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_audio_route_collect(e: *mut c_void, handle: *mut AudioRouteHandle) -> i32;
    #[cfg(test)]
    fn lfm_engine_token_route_submit(
        e: *mut c_void,
        model_id: u64,
        ids: *const u32,
        id_count: usize,
        embedding_kind: u32,
        states: *const LayerState,
        state_count: usize,
        position: usize,
        rope_cos: *const u16,
        rope_sin: *const u16,
        rope_elements: usize,
        out_hidden: *mut u16,
        hidden_elements: usize,
        sampler: *const SampleConfig,
        prng: *mut PrngState,
        out_token: *mut u32,
        lanes: usize,
        commit: *const TokenCommitRecord,
        out_token_completed: *mut u32,
        notify: extern "C" fn(*mut c_void),
        notify_context: *mut c_void,
        out_handle: *mut AudioRouteHandle,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_prefill_workspace_create(
        e: *mut c_void,
        id: u64,
        out_workspace: *mut *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_prefill_workspace_destroy(workspace: *mut c_void);
    #[cfg(test)]
    fn lfm_engine_prefill(
        e: *mut c_void,
        id: u64,
        workspace: *mut c_void,
        ids: *const u32,
        provided_rows: *const u16,
        row_count: usize,
        embed_kind: u32,
        states: *const LayerState,
        state_count: usize,
        position: usize,
        rope_cos: *const u16,
        rope_sin: *const u16,
        rope_elements: usize,
        out_hidden: *mut u16,
        hidden_elements: usize,
        sampler: *const SampleConfig,
        prng: *mut PrngState,
        out_token: *mut u32,
        lanes: usize,
    ) -> i32;
    fn lfm_engine_sample(
        e: *mut c_void,
        logits: *const c_void,
        count: usize,
        dtype: u32,
        config: *const SampleConfig,
        state: *mut PrngState,
        out_token: *mut u32,
    ) -> i32;
    fn lfm_engine_depth_build(
        e: *mut c_void,
        plan: *const super::decode::DepthPlan,
        out_id: *mut u64,
    ) -> i32;
    fn lfm_engine_depth_frame(
        e: *mut c_void,
        id: u64,
        hidden: *const u16,
        hidden_count: usize,
        sampler: *const SampleConfig,
        sample_state: *mut PrngState,
        out_tokens: *mut u32,
        out_token_count: usize,
    ) -> i32;
    fn lfm_engine_depth_clear(e: *mut c_void, id: u64) -> i32;
    fn lfm_engine_depthwise_stream_bf16(
        e: *mut c_void,
        x: *const u16,
        x_count: usize,
        cache: *const u16,
        cache_count: usize,
        weights: *const u16,
        weight_count: usize,
        out: *mut u16,
        out_count: usize,
        next: *mut u16,
        next_count: usize,
        batch: usize,
        channels: usize,
        steps: usize,
        kernel: usize,
    ) -> i32;
    fn lfm_engine_bf16_gemm_f32(
        e: *mut c_void,
        a: *const u16,
        a_count: usize,
        rhs: *const u16,
        rhs_count: usize,
        out: *mut f32,
        out_count: usize,
        m: usize,
        n: usize,
        k: usize,
        rhs_layout: u32,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_bf16_gemm_nt_direct_f32(
        e: *mut c_void,
        a: *const u16,
        a_count: usize,
        weights: *const c_void,
        weight_count: usize,
        out: *mut f32,
        out_count: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_fft_conv_dd(
        e: *mut c_void,
        input: *const f32,
        input_count: usize,
        kernel: *const f32,
        kernel_count: usize,
        skip: *const f32,
        skip_count: usize,
        out: *mut f32,
        out_count: usize,
        batch: usize,
        channels: usize,
        steps: usize,
        fft_size: usize,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_irfft_dd(
        e: *mut c_void,
        real: *const f32,
        real_count: usize,
        imag: *const f32,
        imag_count: usize,
        out: *mut f32,
        out_count: usize,
        rows: usize,
        fft_size: usize,
        scale_hi: f32,
        scale_lo: f32,
    ) -> i32;
    fn lfm_engine_conv_layer(
        e: *mut c_void,
        id: u64,
        layer: usize,
        x: *const u16,
        x_len: usize,
        state_in: *const u16,
        state_in_len: usize,
        state_out: *mut u16,
        state_out_len: usize,
        out: *mut u16,
        out_len: usize,
        lanes: usize,
    ) -> i32;
    fn lfm_engine_mlp(
        e: *mut c_void,
        x: *const u16,
        norm_w: *const u16,
        w1: *const u16,
        w3: *const u16,
        w2: *const u16,
        out: *mut u16,
        h: usize,
        i: usize,
        eps: f32,
        lanes: usize,
    ) -> i32;
    #[cfg(test)]
    fn lfm_engine_prng_fill(
        e: *mut c_void,
        state: *mut PrngState,
        out: *mut u64,
        count: usize,
    ) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_prng_continuation_for_test(
        e: *mut c_void,
        state: *mut PrngState,
        out: *mut u64,
        pass_count: usize,
    ) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_pause_boundary_for_test(e: *mut c_void, kind: u32, action: u32) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_wait_pass_slots_for_test(e: *mut c_void, live: u32) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_request_kind_valid_for_test(kind: u32) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_wait_word_layout_for_test(e: *mut c_void) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_grid_snapshot_for_test(
        e: *mut c_void,
        blocks: *mut u32,
        completions: *mut u64,
        generations: *mut u64,
        lease: *mut u64,
    ) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_audio_route_edge_for_test(
        node: u32,
        outcome: u32,
        target: *mut u32,
    ) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_audio_token_class_for_test(token: u32) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_audio_route_service_for_test(
        snapshot: u64,
        enqueued: u64,
        service: u32,
    ) -> u32;
    #[cfg(test)]
    fn lfm_internal_engine_fail_audio_route_depth_for_test(e: *mut c_void, status: i32) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_fail_audio_route_mimi_for_test(e: *mut c_void, status: i32) -> i32;
    #[cfg(test)]
    fn lfm_internal_audio_route_epoch_new_for_test(value: u64) -> *mut c_void;
    #[cfg(test)]
    fn lfm_internal_audio_route_epoch_free_for_test(epoch: *mut c_void);
    #[cfg(test)]
    fn lfm_internal_engine_arm_lane_pause_for_test(e: *mut c_void) -> i32;
    #[cfg(test)]
    fn lfm_internal_engine_wait_lane_pause_for_test(e: *mut c_void) -> i32;
}

pub(crate) fn depthwise_stream_available() -> bool {
    // SAFETY: capability query accepts no pointers and mutates no state.
    unsafe { lfm_depthwise_stream_bf16_available() != 0 }
}

/// Native assembly GEMM capability. This is a property of the mounted
/// Flashkern backend; Rust neither selects an implementation nor performs math.
pub fn bf16_gemm_available() -> bool {
    // SAFETY: capability query accepts no pointers and mutates no state.
    unsafe { lfm_bf16_gemm_available() != 0 }
}

/// Handle to the persistent native engine. One per process is the intended shape
/// (the lane team executes one full pass at a time). The C side owns two queued
/// request/scratch slots and mounts exactly one onto the executor board. This
/// compatibility wrapper serializes its blocking calls under `pass_lock`; native
/// completion continuations do not require Rust progress. The raw C ABI independently
/// claims a slot before touching request state.
pub struct NativeEngine {
    ptr: *mut c_void,
    pass_lock: Mutex<()>,
}

// SAFETY: Send — the handle is an opaque pointer to a C-heap object with no thread
// affinity. Sync — provided by `pass_lock` above serializing every compatibility
// call. The C side's atomic claim rejects unsafe concurrent compatibility callers
// before request setup, but it does not make two safe Rust borrows of the same output
// buffer legal or define queue ordering.
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    pub(crate) fn raw_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn new(workers: usize) -> Option<Self> {
        if !(1..=16).contains(&workers) {
            return None;
        }
        let _ = kcoro_sys::link_anchor as fn();
        // SAFETY: plain constructor call; null = failure.
        let p = unsafe { lfm_engine_new(workers as i32) };
        if p.is_null() {
            return None;
        }
        Some(Self {
            ptr: p,
            pass_lock: Mutex::new(()),
        })
    }

    #[cfg(test)]
    fn snapshot(&self) -> EngineSnapshot {
        let mut out = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        // SAFETY: `out` is the ABI-sized destination and the engine outlives the call.
        assert_eq!(unsafe { lfm_engine_snapshot(self.ptr, &mut out) }, 0);
        out
    }

    /// One fused-MLP decode block, entirely native and parity-pinned against the
    /// test-only reference implementation at the same `lanes`.
    #[must_use = "false = native pass did not run; caller must surface the rejection"]
    pub fn fused_mlp(
        &self,
        x: &[u16],
        w: &super::decode::FusedMlpWeights,
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
        if !bf16_gemm_available() {
            return false;
        }
        let h = x.len();
        let i = w.w1.len() / h;
        assert!(h > 0 && i > 0, "native fused_mlp: empty dims");
        assert_eq!(w.norm_w.len(), h, "native fused_mlp: norm_w.len() != H");
        assert_eq!(w.w1.len(), i * h, "native fused_mlp: w1.len() != I·H");
        assert_eq!(w.w3.len(), i * h, "native fused_mlp: w3.len() != I·H");
        assert_eq!(w.w2.len(), h * i, "native fused_mlp: w2.len() != H·I");
        assert_eq!(out.len(), h, "native fused_mlp: out.len() != H");
        // The compatibility lock that makes `Sync` true: this wrapper serializes
        // its borrowed Rust slices through exact completion even though native
        // continuations use the second request/scratch slot independently.
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: slice extents checked above; the call blocks until the pass
        // completes, so every pointer outlives its use.
        let rc = unsafe {
            lfm_engine_mlp(
                self.ptr,
                x.as_ptr(),
                w.norm_w.as_ptr(),
                w.w1.as_ptr(),
                w.w3.as_ptr(),
                w.w2.as_ptr(),
                out.as_mut_ptr(),
                h,
                i,
                w.eps,
                lanes,
            )
        };
        // rc != 0 = native-side failure (e.g. scratch growth failed): report it so
        // the caller can take the bit-identical threadgroup path instead of dying.
        rc == 0
    }

    /// Advance one conversation-owned CSPRNG stream through a typed native pass.
    /// Sampling calls this primitive inside its token pass once mounted; this
    /// standalone entry pins assembly and kcoro lifecycle behavior independently.
    #[cfg(test)]
    #[must_use = "false = native pass rejected; the stream was not advanced"]
    pub(crate) fn prng_fill(&self, state: &mut PrngState, out: &mut [u64]) -> bool {
        if out.is_empty() {
            return true;
        }
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: exclusive borrows keep state/output live and unaliased until the
        // blocking SQ/CQ pass completes.
        unsafe { lfm_engine_prng_fill(self.ptr, state, out.as_mut_ptr(), out.len()) == 0 }
    }

    fn sample(
        &self,
        logits: *const c_void,
        count: usize,
        dtype: u32,
        config: &SampleConfig,
        state: &mut PrngState,
    ) -> Result<u32, i32> {
        const F32: u32 = 1;
        const BF16: u32 = 2;
        assert!(dtype == F32 || dtype == BF16);
        let _pass = self.pass_lock.lock().unwrap();
        let mut token = 0;
        // SAFETY: the typed wrappers below supply a live input extent; exclusive
        // state/output borrows last through the blocking SQ/CQ completion.
        let rc =
            unsafe { lfm_engine_sample(self.ptr, logits, count, dtype, config, state, &mut token) };
        if rc != 0 {
            return Err(rc);
        }
        Ok(token)
    }

    pub(crate) fn sample_f32(
        &self,
        logits: &[f32],
        config: &SampleConfig,
        state: &mut PrngState,
    ) -> Result<u32, i32> {
        self.sample(logits.as_ptr().cast(), logits.len(), 1, config, state)
    }

    pub(crate) fn sample_bf16(
        &self,
        logits: &[u16],
        config: &SampleConfig,
        state: &mut PrngState,
    ) -> Result<u32, i32> {
        self.sample(logits.as_ptr().cast(), logits.len(), 2, config, state)
    }

    /// Borrow two bf16 matrices and write one f32 result through a single fixed-team
    /// ticket. `rhs_nk` selects checkpoint-native `[N,K]`; false selects `[K,N]`.
    pub(crate) fn bf16_gemm_f32(
        &self,
        a: &[u16],
        rhs: &[u16],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        rhs_nk: bool,
    ) -> bool {
        const KN: u32 = 0;
        const NK: u32 = 1;
        let a_count = m.checked_mul(k).expect("bf16 gemm A extent overflow");
        let rhs_count = n.checked_mul(k).expect("bf16 gemm RHS extent overflow");
        let out_count = m.checked_mul(n).expect("bf16 gemm output extent overflow");
        assert_eq!(a.len(), a_count, "bf16 gemm A extent");
        assert_eq!(rhs.len(), rhs_count, "bf16 gemm RHS extent");
        assert_eq!(out.len(), out_count, "bf16 gemm output extent");
        if m == 0 || n == 0 || k == 0 {
            return true;
        }
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: exact extents are checked above. Both inputs and the exclusive
        // destination remain borrowed until the blocking CQ completion arrives.
        unsafe {
            lfm_engine_bf16_gemm_f32(
                self.ptr,
                a.as_ptr(),
                a.len(),
                rhs.as_ptr(),
                rhs.len(),
                out.as_mut_ptr(),
                out.len(),
                m,
                n,
                k,
                if rhs_nk { NK } else { KN },
            ) == 0
        }
    }

    #[cfg(test)]
    fn bf16_gemm_nt_direct_f32(
        &self,
        a: &[u16],
        weights: &[u16],
        out: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> bool {
        assert_eq!(a.len(), m.checked_mul(k).expect("direct GEMM A overflow"));
        assert_eq!(
            weights.len(),
            n.checked_mul(k).expect("direct GEMM W overflow")
        );
        assert_eq!(
            out.len(),
            m.checked_mul(n).expect("direct GEMM output overflow")
        );
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: exact extents are asserted and all borrows outlive the
        // blocking completion returned by the native ticket.
        unsafe {
            lfm_engine_bf16_gemm_nt_direct_f32(
                self.ptr,
                a.as_ptr(),
                a.len(),
                weights.as_ptr().cast(),
                weights.len(),
                out.as_mut_ptr(),
                out.len(),
                m,
                n,
                k,
            ) == 0
        }
    }

    /// Run the complete double-double FFT convolution grid as one native ticket.
    /// The fixed lane team shares one reusable work plane and fences every radix-2 stage.
    #[cfg(test)]
    pub(crate) fn fft_conv_dd(
        &self,
        input: &[f32],
        kernel: &[f32],
        skip: &[f32],
        out: &mut [f32],
        batch: usize,
        channels: usize,
        steps: usize,
        fft_size: usize,
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: the C boundary validates every extent. Safe borrows keep all
        // inputs and the exclusive destination alive through exact completion.
        unsafe {
            lfm_engine_fft_conv_dd(
                self.ptr,
                input.as_ptr(),
                input.len(),
                kernel.as_ptr(),
                kernel.len(),
                skip.as_ptr(),
                skip.len(),
                out.as_mut_ptr(),
                out.len(),
                batch,
                channels,
                steps,
                fft_size,
            ) == 0
        }
    }

    #[cfg(test)]
    pub(crate) fn irfft_dd(
        &self,
        real: &[f32],
        imag: &[f32],
        out: &mut [f32],
        rows: usize,
        fft_size: usize,
        scale: super::dd::Dd,
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: the native boundary validates all row/frequency extents and
        // consumes the two scale limbs by value before returning.
        unsafe {
            lfm_engine_irfft_dd(
                self.ptr,
                real.as_ptr(),
                real.len(),
                imag.as_ptr(),
                imag.len(),
                out.as_mut_ptr(),
                out.len(),
                rows,
                fft_size,
                scale.hi,
                scale.lo,
            ) == 0
        }
    }

    /// Streaming depthwise convolution on the CPU kernel team.
    /// The prior state and incoming chunk remain separate borrowed planes; native
    /// lanes write output and next-state planes under one SQ/CQ ticket.
    pub(crate) fn depthwise_stream_bf16(
        &self,
        x: &[u16],
        cache: Option<&[u16]>,
        weights: &[u16],
        out: &mut [u16],
        next: &mut [u16],
        batch: usize,
        channels: usize,
        steps: usize,
        kernel: usize,
    ) -> bool {
        if !depthwise_stream_available() {
            return false;
        }
        assert!(batch > 0 && channels > 0 && steps > 0 && kernel > 0);
        let rows = batch
            .checked_mul(channels)
            .expect("depthwise rows overflow");
        let prior = kernel - 1;
        assert_eq!(x.len(), rows * steps, "depthwise stream input extent");
        assert_eq!(
            weights.len(),
            channels * kernel,
            "depthwise stream weight extent"
        );
        if let Some(cache) = cache {
            assert_eq!(cache.len(), rows * prior, "depthwise stream cache extent");
        }
        assert_eq!(out.len(), rows * steps, "depthwise stream output extent");
        assert_eq!(
            next.len(),
            rows * prior,
            "depthwise stream next-state extent"
        );
        let (cache_ptr, cache_count) = cache
            .filter(|values| !values.is_empty())
            .map(|values| (values.as_ptr(), values.len()))
            .unwrap_or((std::ptr::null(), 0));
        let next_ptr = if next.is_empty() {
            std::ptr::null_mut()
        } else {
            next.as_mut_ptr()
        };
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: all extents were checked above; borrowed inputs and exclusive
        // output remain live until the exact blocking completion returns.
        unsafe {
            lfm_engine_depthwise_stream_bf16(
                self.ptr,
                x.as_ptr(),
                x.len(),
                cache_ptr,
                cache_count,
                weights.as_ptr().cast(),
                weights.len(),
                out.as_mut_ptr(),
                out.len(),
                next_ptr,
                next.len(),
                batch,
                channels,
                steps,
                kernel,
            ) == 0
        }
    }

    pub(crate) fn depth_build(&self, plan: &super::decode::DepthPlan) -> Option<u64> {
        let _pass = self.pass_lock.lock().unwrap();
        let mut id = 0;
        // SAFETY: native code copies every descriptor before returning. The
        // pointed-to weight payloads remain owned by the model.
        let rc = unsafe { lfm_engine_depth_build(self.ptr, plan, &mut id) };
        (rc == 0).then_some(id)
    }

    pub(crate) fn depth_frame(
        &self,
        id: u64,
        hidden: &[u16],
        sampler: &SampleConfig,
        state: &mut PrngState,
        out: &mut [u32],
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: all slices and the exclusive sampler state borrow remain live
        // until the blocking typed pass receives its exact completion.
        unsafe {
            lfm_engine_depth_frame(
                self.ptr,
                id,
                hidden.as_ptr(),
                hidden.len(),
                sampler,
                state,
                out.as_mut_ptr(),
                out.len(),
            ) == 0
        }
    }

    pub(crate) fn depth_clear(&self, id: u64) {
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: pass_lock excludes all frame use; id ownership is checked natively.
        let rc = unsafe { lfm_engine_depth_clear(self.ptr, id) };
        assert_eq!(rc, 0, "depth plan clear raced a native pass");
    }
}

impl NativeEngine {
    /// Install the resident backbone layer table. The pointers must stay valid until
    /// [`Self::ctx_clear`] — the [`BackboneCtxGuard`] enforces clear-before-drop.
    /// Multiple immutable plans may coexist; the returned identity selects one
    /// plan for each native pass while the executor and scratch arena stay shared.
    fn ctx_build(&self, descs: &[LayerDesc], h: usize, ffn: usize, max_ctx: usize) -> Option<u64> {
        let _pass = self.pass_lock.lock().unwrap();
        let mut id = 0u64;
        // SAFETY: descs copied by the C side before return; dims checked there.
        let rc = unsafe {
            lfm_ctx_build(
                self.ptr,
                descs.as_ptr(),
                descs.len(),
                h,
                ffn,
                max_ctx,
                &mut id,
            )
        };
        (rc == 0).then_some(id)
    }

    /// Install the head tables (text embed / audio embed / final norm / tied logits).
    unsafe fn set_heads(
        &self,
        id: u64,
        embed_w: *const u16,
        embed_len: usize,
        vocab: usize,
        audio_embed_w: *const u16,
        audio_embed_len: usize,
        audio_rows: usize,
        emb_norm_w: *const u16,
        emb_norm_len: usize,
        emb_norm_eps: f32,
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        let rc = unsafe {
            lfm_ctx_set_heads(
                self.ptr,
                id,
                embed_w,
                embed_len,
                vocab,
                audio_embed_w,
                audio_embed_len,
                audio_rows,
                emb_norm_w,
                emb_norm_len,
                emb_norm_eps,
            )
        };
        rc == 0
    }

    /// ONE token through the whole backbone — embed, every layer, final norm, and
    /// (when `out_logits` is `Some`) the tied logits head. Sampling stays with the
    /// caller. `states[l]` carries fresh per-generation pointers; the caller ensured
    /// plane capacity BEFORE capture and advances its cursors on success.
    #[must_use = "false = native pass did not run; caller must take the fallback"]
    #[allow(clippy::too_many_arguments)]
    unsafe fn token_pass(
        &self,
        id: u64,
        ids: &[u32],
        embed_kind: u32,
        states: &[LayerState],
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out_hidden: &mut [u16],
        out_logits: Option<&mut [f32]>,
        sampler: Option<&SampleConfig>,
        sample_state: Option<&mut PrngState>,
        out_token: Option<&mut u32>,
        lanes: usize,
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        let (logits_ptr, logits_len) = out_logits
            .map(|l| (l.as_mut_ptr(), l.len()))
            .unwrap_or((std::ptr::null_mut(), 0));
        let sampler_ptr = sampler.map_or(std::ptr::null(), std::ptr::from_ref);
        let state_ptr = sample_state.map_or(std::ptr::null_mut(), std::ptr::from_mut);
        let token_ptr = out_token.map_or(std::ptr::null_mut(), std::ptr::from_mut);
        let rc = unsafe {
            lfm_engine_token_pass(
                self.ptr,
                id,
                ids.as_ptr(),
                ids.len(),
                embed_kind,
                states.as_ptr(),
                states.len(),
                pos,
                cos_base,
                sin_base,
                rope_len,
                out_hidden.as_mut_ptr(),
                out_hidden.len(),
                logits_ptr,
                logits_len,
                sampler_ptr,
                state_ptr,
                token_ptr,
                lanes,
                // Rust decode never provides an embedding — token/audio-out ids
                // embed via the native tables. The `embed_kind == 2` path (native
                // audio-in prefill) is driven by C++ `lfm_conversation_prefill`,
                // not this rim. This arg only keeps the decode call ABI-correct.
                std::ptr::null(),
            )
        };
        rc == 0
    }

    /// One whole attention+MLP layer in a single doorbell. The engine appends the
    /// step's K/V rows at `pos` into the caller's planes (capacity pre-grown by the
    /// caller) and attends over pos+1 entries. Caller advances its cursor on success.
    #[must_use = "false = native pass did not run; caller must take the fallback"]
    #[allow(clippy::too_many_arguments)]
    unsafe fn attn_layer(
        &self,
        id: u64,
        layer: usize,
        x: &[u16],
        k_plane: *mut u16,
        k_len: usize,
        v_plane: *mut u16,
        v_len: usize,
        head_stride: usize,
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
        assert_eq!(out.len(), x.len(), "native attn_layer: out.len() != H");
        let _pass = self.pass_lock.lock().unwrap();
        let rc = unsafe {
            lfm_engine_attn_layer(
                self.ptr,
                id,
                layer,
                x.as_ptr(),
                x.len(),
                k_plane,
                k_len,
                v_plane,
                v_len,
                head_stride,
                pos,
                cos_base,
                sin_base,
                rope_len,
                out.as_mut_ptr(),
                out.len(),
                lanes,
            )
        };
        rc == 0
    }

    fn ctx_clear(&self, id: u64) {
        // Serialized against passes: no pass can be in flight while we clear. The id
        // keys ownership — a stale guard's clear is a no-op on the C side.
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: engine pointer valid for the process lifetime.
        let rc = unsafe { lfm_ctx_clear(self.ptr, id) };
        assert_eq!(
            rc, 0,
            "ctx_clear raced another raw engine operation; retained weight pointers cannot be dropped"
        );
    }

    /// The team's lane count — the ONE authority for lane-uniform program sizing.
    /// (Previously programs asked rayon, i.e. a foreign pool, for our kernel's width.)
    pub fn lanes_total(&self) -> usize {
        // SAFETY: engine pointer valid for the process lifetime; pure read.
        unsafe { lfm_engine_lanes(self.ptr) as usize }
    }

    /// One whole shortconv+MLP layer in a single doorbell, parity-pinned against the
    /// composed test-only references at the same `lanes`.
    #[must_use = "false = native pass did not run; caller must preserve state"]
    fn conv_layer(
        &self,
        id: u64,
        layer: usize,
        x: &[u16],
        state_in: &[u16],
        state_out: &mut [u16],
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
        let h = x.len();
        assert_eq!(out.len(), h, "native conv_layer: out.len() != H");
        assert_eq!(
            state_in.len(),
            state_out.len(),
            "native conv_layer: state extents differ"
        );
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: slice extents checked; the call blocks until the pass completes; the
        // layer-table pointers are guarded live by BackboneCtxGuard.
        let rc = unsafe {
            lfm_engine_conv_layer(
                self.ptr,
                id,
                layer,
                x.as_ptr(),
                x.len(),
                state_in.as_ptr(),
                state_in.len(),
                state_out.as_mut_ptr(),
                state_out.len(),
                out.as_mut_ptr(),
                out.len(),
                lanes,
            )
        };
        rc == 0
    }
}

/// Cache-resident scratch table of per-layer engine states — allocated once, entries
/// rewritten every token. The raw pointers inside are meaningful ONLY during the
/// single blocking `token_pass` call that follows their capture; between passes they
/// are stale by contract and never dereferenced.
///
/// SAFETY (Send): moving the container between threads moves dead pointers; every
/// live use happens on the capturing thread inside the blocking call.
#[derive(Default)]
pub struct StateTable(pub Vec<LayerState>);
unsafe impl Send for StateTable {}

/// Keeps the resident layer table's backing alive and clears the table before it dies.
/// `held` owns tensors DERIVED at capture (e.g. the squeezed-contiguous conv weight);
/// the undived model weights are owned by the model, which must own this guard so the
/// guard drops (and clears the C table) before those weights do. The install id keys
/// ownership: this drop can only clear ITS OWN install, never a later model's.
pub struct BackboneCtxGuard {
    id: u64,
    _held: Vec<candle_core::Tensor>,
}

impl BackboneCtxGuard {
    pub(crate) fn lanes_total(&self) -> usize {
        process_engine().lanes_total()
    }

    /// # Safety
    /// Every pointer must refer to the owning model's live, contiguous bf16 storage
    /// for at least the supplied extent. This guard keeps the owning context selected;
    /// the model keeps the underlying tensors alive.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn set_heads(
        &self,
        embed_w: *const u16,
        embed_len: usize,
        vocab: usize,
        audio_embed_w: *const u16,
        audio_embed_len: usize,
        audio_rows: usize,
        emb_norm_w: *const u16,
        emb_norm_len: usize,
        emb_norm_eps: f32,
    ) -> bool {
        unsafe {
            process_engine().set_heads(
                self.id,
                embed_w,
                embed_len,
                vocab,
                audio_embed_w,
                audio_embed_len,
                audio_rows,
                emb_norm_w,
                emb_norm_len,
                emb_norm_eps,
            )
        }
    }

    /// # Safety
    /// Raw state and rope pointers must remain live, correctly aligned, and exclusively
    /// mutable where written for the duration of this blocking native pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn token_pass(
        &self,
        ids: &[u32],
        embed_kind: u32,
        states: &[LayerState],
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out_hidden: &mut [u16],
        out_logits: Option<&mut [f32]>,
        sampler: Option<&SampleConfig>,
        sample_state: Option<&mut PrngState>,
        out_token: Option<&mut u32>,
        lanes: usize,
    ) -> bool {
        unsafe {
            process_engine().token_pass(
                self.id,
                ids,
                embed_kind,
                states,
                pos,
                cos_base,
                sin_base,
                rope_len,
                out_hidden,
                out_logits,
                sampler,
                sample_state,
                out_token,
                lanes,
            )
        }
    }

    /// # Safety
    /// The K/V and rope pointers must name the supplied live extents. K/V must be
    /// exclusively mutable for the blocking call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attn_layer(
        &self,
        layer: usize,
        x: &[u16],
        k_plane: *mut u16,
        k_len: usize,
        v_plane: *mut u16,
        v_len: usize,
        head_stride: usize,
        pos: usize,
        cos_base: *const u16,
        sin_base: *const u16,
        rope_len: usize,
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
        unsafe {
            process_engine().attn_layer(
                self.id,
                layer,
                x,
                k_plane,
                k_len,
                v_plane,
                v_len,
                head_stride,
                pos,
                cos_base,
                sin_base,
                rope_len,
                out,
                lanes,
            )
        }
    }

    pub(crate) fn conv_layer(
        &self,
        layer: usize,
        x: &[u16],
        state_in: &[u16],
        state_out: &mut [u16],
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
        process_engine().conv_layer(self.id, layer, x, state_in, state_out, out, lanes)
    }
}

impl Drop for BackboneCtxGuard {
    fn drop(&mut self) {
        process_engine().ctx_clear(self.id);
    }
}

/// Build + install the backbone layer table on the process engine. Returns the guard
/// the MODEL must own (declared before its weight fields so it drops first), or `None`
/// when that model cannot be represented by the native ABI. Any number of immutable
/// plans may coexist; one pass ticket selects one plan. Engine presence is unconditional.
pub fn install_backbone_ctx(
    descs: &[LayerDesc],
    h: usize,
    ffn: usize,
    max_ctx: usize,
    held: Vec<candle_core::Tensor>,
) -> Option<BackboneCtxGuard> {
    process_engine()
        .ctx_build(descs, h, ffn, max_ctx)
        .map(|id| BackboneCtxGuard { id, _held: held })
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        let _pass = self
            .pass_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // SAFETY: pass_lock excludes every safe Rust call. Native stop closes bridge
        // admission, wakes the dispatcher, joins it and the lane team, then destroys
        // the fully drained retained-descriptor rings.
        unsafe { lfm_engine_request_stop(self.ptr) };
        unsafe { lfm_engine_free(self.ptr) };
    }
}

/// The process-resident engine for the model hot path, built on first use. Initialization
/// failure is fatal: this is the decode substrate, not an optional acceleration with a
/// second scheduler hidden behind it.
///
/// Lifetime is deliberately process-long: `OnceLock` never drops, so the team's
/// threads live until exit — the daemon shape this crate ships in. Workers are sized
/// by the crate's torch-parity thread policy (`threads::intraop_default_num_threads`:
/// P-cores only on Apple Silicon via `hw.perflevel0.physicalcpu`) — NOT
/// `available_parallelism`, which counts E-cores and reintroduces the tail-latency
/// imbalance the runtime documents as harmful.
/// INFALLIBLE (her substrate rule): the kcoro engine is part of the whole
/// thing, not an optional acceleration. If the team can't stand up, the
/// process is broken — panic with the reason instead of handing callers an
/// Option whose `None` arm ships a decode path that shouldn't exist.
pub fn process_engine() -> &'static NativeEngine {
    use std::sync::OnceLock;
    static ENGINE: OnceLock<NativeEngine> = OnceLock::new();
    ENGINE.get_or_init(|| {
        let workers = crate::threads::intraop_default_num_threads().clamp(1, 16);
        NativeEngine::new(workers).unwrap_or_else(|| {
            panic!(
                "kcoro native engine failed to initialize ({workers} lanes) — \
                 the engine is the decode substrate; there is no fallback path"
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    // The process engine holds ONE resident layer table; tests that build/clear it
    // must not interleave. (Each individual call is pass_lock-serialized; this guards
    // the build→use→clear SEQUENCE.)
    static CTX_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn flashkern_control_progress_is_owned_by_retained_kcoro_services() {
        let source = include_str!("../../../../liquid-audio/native/src/engine/flashkern_engine.cpp");
        assert!(!source.contains("pthread_create"));
        assert!(!source.contains("pthread_join"));
        assert!(!source.contains("route_word"));
        assert!(source.contains("kc_runtime_create(&runtime_config"));
        assert!(source.contains("kc_service_create(e->control_runtime"));
        assert!(source.contains("kc_service_notifier_create(e->bridge_service"));
        assert!(source.contains("kc_service_notifier_create(e->route_service"));
        assert!(source.contains("kc_service_notifier_notify(notifier)"));
        assert!(!source.contains("kc_service_notify("));
        assert!(source.contains("kc_team_dispatch_notify"));

        let free = source
            .split("void lfm_engine_free(void *ep) {")
            .nth(1)
            .expect("engine teardown");
        let team = free.find("kc_team_join(e->team)").expect("team join");
        let notifier = free
            .find("kc_service_notifier_destroy(e->route_notifier)")
            .expect("notifier destroy");
        let service = free
            .find("kc_service_destroy(e->route_service)")
            .expect("service destroy");
        let runtime = free
            .find("kc_runtime_destroy(e->control_runtime)")
            .expect("runtime destroy");
        assert!(team < notifier && notifier < service && service < runtime);

        let bridge = source
            .split("static void bridge_service_main")
            .nth(1)
            .expect("bridge service callback")
            .split("static uint64_t next_sequence")
            .next()
            .expect("bridge service callback boundary");
        assert!(!bridge.contains("kc_port_wait_u32"));
        assert!(!bridge.contains("kc_team_wait"));

        let route = source
            .split("static void audio_route_service_main")
            .nth(1)
            .expect("route service callback")
            .split("} // namespace")
            .next()
            .expect("route service callback boundary");
        assert!(!route.contains("kc_port_wait_u32"));
        assert!(!route.contains("kc_team_wait"));
    }

    #[test]
    fn scalar_assembly_math_abi_is_bit_exact_without_simd_feature_gates() {
        let values = [1.0f32, 2.0, 3.0];
        let bf16 = [0x3f80u16, 0x4000, 0x4040, 0x4080];
        let mut bias = [1.0f32, 2.0];
        let bias_bits = [0x3f00u16, 0xbf80];
        let mut rope = bf16;
        let cosine = [0x3f80u16, 0x3f00];
        let sine = [0u16, 0x3f00];

        // SAFETY: every pointer names the complete fixed fixture for the declared count.
        unsafe {
            assert_eq!(lfm_rsqrt_size(4).to_bits(), 0x3f00_0000);
            assert_eq!(lfm_inv_rms_f32(30.0, 4, 0.5).to_bits(), 0x3eb5_04f3);
            assert_eq!(
                lfm_sum_f32(values.as_ptr(), values.len()).to_bits(),
                0x40c0_0000
            );
            assert_eq!(
                lfm_bf16_sumsq_stride_f32(bf16.as_ptr(), bf16.len(), 0, 2).to_bits(),
                0x4120_0000
            );
            assert_eq!(
                lfm_bf16_sumsq_stride_f32(bf16.as_ptr(), bf16.len(), 1, 2).to_bits(),
                0x41a0_0000
            );
            lfm_bf16_bias_add_f32(bias.as_mut_ptr(), bias_bits.as_ptr().cast(), bias.len());
            lfm_bf16_rope_neox(
                rope.as_mut_ptr(),
                cosine.as_ptr(),
                sine.as_ptr(),
                rope.len(),
            );
        }
        assert_eq!(bias.map(f32::to_bits), [0x3fc0_0000, 0x3f80_0000]);
        assert_eq!(rope, [0x3f80, 0xbf80, 0x4040, 0x4040]);

        // Products land exactly halfway between adjacent bf16 values. The
        // low candidate is odd in lane 0 and even in lane 1, proving both
        // halves of round-to-nearest-even rather than merely truncation.
        let mut ties = [0x3f81u16, 0x3f82, 0, 0];
        let tie_cosine = [0x3fc0u16, 0x3fa0];
        let tie_sine = [0u16; 2];
        unsafe {
            lfm_bf16_rope_neox(
                ties.as_mut_ptr(),
                tie_cosine.as_ptr(),
                tie_sine.as_ptr(),
                ties.len(),
            );
        }
        assert_eq!(ties, [0x3fc2, 0x3fa2, 0, 0]);
    }

    #[test]
    fn bf16_checkpoint_words_copy_bit_exactly_from_an_unaligned_view() {
        let expected = [0x0000u16, 0x8000, 0x0001, 0x7f80, 0xff80, 0x7fc1];
        let mut image = vec![0xa5u8; expected.len() * 2 + 1];
        for (index, word) in expected.iter().enumerate() {
            image[1 + index * 2..1 + index * 2 + 2].copy_from_slice(&word.to_le_bytes());
        }
        let mut actual = [0u16; 6];
        // SAFETY: the deliberately odd source address still names `expected.len()`
        // complete little-endian BF16 words; the assembly leaf uses unaligned
        // halfword loads and writes the aligned activation destination.
        unsafe {
            lfm_bf16_copy_bytes(
                image.as_ptr().add(1).cast(),
                actual.as_mut_ptr(),
                actual.len(),
            );
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn sampler_assembly_exponential_is_a_fixed_cross_arch_fixture() {
        let values = [0.0f32, -0.5, -1.0, -2.0, -4.0, -8.0, -16.0, -100.0];
        let bf16 = [
            0x0000u16, 0xbf00, 0xbf80, 0xc000, 0xc080, 0xc100, 0xc180, 0xc2c8,
        ];
        let mut f32_weights = [0.0f32; 8];
        let mut bf16_weights = [0.0f32; 8];

        // SAFETY: every pointer names the full fixed fixture and both destinations
        // have exactly `values.len()` writable elements.
        let f32_sum = unsafe {
            lfm_sampler_exp_sum_f32(
                values.as_ptr(),
                f32_weights.as_mut_ptr(),
                values.len(),
                1.0,
                0.0,
                f32::NEG_INFINITY,
            )
        };
        let bf16_sum = unsafe {
            lfm_sampler_exp_sum_bf16(
                bf16.as_ptr(),
                bf16_weights.as_mut_ptr(),
                bf16.len(),
                0x3f80,
                0.0,
                f32::NEG_INFINITY,
            )
        };

        let expected = [
            0x3f80_0000,
            0x3f1b_4598,
            0x3ebc_5ab2,
            0x3e0a_9555,
            0x3c96_0aae,
            0x39af_e108,
            0x33f1_aad7,
            0x0000_0000,
        ];
        assert_eq!(f32_weights.map(f32::to_bits), expected);
        assert_eq!(bf16_weights.map(f32::to_bits), expected);
        assert_eq!(f32_sum.to_bits(), 0x4008_37a5);
        assert_eq!(bf16_sum.to_bits(), 0x4008_37a5);
    }

    #[test]
    fn native_prng_matches_chacha20_and_replays_snapshot_through_kcoro() {
        let engine = NativeEngine::new(4).expect("native engine init");
        let mut state = PrngState::from_material(&[0; 32], &[0; 8]).expect("material seed");
        let snapshot = state;
        let engine_before = engine.snapshot();

        // Original ChaCha20, zero key/nonce, counters 0 and 1. These are fixed
        // published block vectors interpreted as little-endian u64 draws.
        let expected = [
            0x903d_f1a0_ade0_b876,
            0x28bd_8653_e56a_5d40,
            0x1aed_8da0_b819_d2bd,
            0xc70d_778b_ccef_36a8,
            0x8d48_5751_7c59_41da,
            0x374a_d8b8_3fe0_2477,
            0x1ca1_1815_f4b8_436a,
            0x8665_eeb2_69b6_87c3,
            0x7a38_5155_bee7_079f,
            0x0d08_2d73_7c97_ba98,
            0x6965_e348_a029_0fcb,
            0xed7a_ee32_3e53_c612,
            0x434e_e69c_7621_b729,
            0xd539_d874_b033_71d5,
            0x45fb_0a51_281f_ed31,
            0x6f4d_794b_1f0a_e1ac,
        ];
        let mut got = [0u64; 16];
        assert!(engine.prng_fill(&mut state, &mut got));
        assert_eq!(got, expected);
        assert_eq!(state.cursor, 64, "two complete blocks must be consumed");
        assert_eq!(state.core[12], 2, "next block counter");
        assert_eq!(state.core[13], 0);

        // A quiescent byte-copy of the state is the complete replay boundary.
        let mut replay = snapshot;
        let mut replayed = [0u64; 16];
        assert!(engine.prng_fill(&mut replay, &mut replayed));
        assert_eq!(replayed, got);
        assert_eq!(replay.core, state.core);
        assert_eq!(replay.block, state.block);
        assert_eq!(replay.cursor, state.cursor);

        // Both fills traversed the real retained-descriptor and kcoro SQ/CQ path.
        let engine_after = engine.snapshot();
        assert_eq!(
            engine_after.pass_submissions - engine_before.pass_submissions,
            2
        );
        assert_eq!(
            engine_after.pass_completions - engine_before.pass_completions,
            2
        );
        assert_eq!(
            engine_after.bridge_dispatches - engine_before.bridge_dispatches,
            2
        );
        assert_eq!(
            engine_after.fence_generations - engine_before.fence_generations,
            4
        );
        assert_eq!(engine_after.descriptors_live, 0);
        assert_eq!(engine_after.pass_claimed, 0);
    }

    #[test]
    fn native_sampler_is_deterministic_thresholded_and_snapshotable() {
        use half::bf16;

        let engine = NativeEngine::new(4).expect("native engine init");
        let logits = [0.1f32, 5.0, 0.2, 3.0, -2.0, 3.0];
        let greedy = SampleConfig::new(None, None);
        let stochastic = SampleConfig::new(Some(1.0), Some(2));
        let mut state = PrngState::from_seed(123).expect("seed");
        let untouched = state;
        assert_eq!(engine.sample_f32(&logits, &greedy, &mut state), Ok(1));
        assert_eq!(
            state.cursor, untouched.cursor,
            "greedy must not consume RNG"
        );
        assert_eq!(state.core, untouched.core);

        let tied = [f32::NAN, 5.0, 5.0, -3.0].map(|value| bf16::from_f32(value).to_bits());
        let mut tied_state = PrngState::from_seed(456).expect("seed");
        assert_eq!(
            engine.sample_bf16(&tied, &greedy, &mut tied_state),
            Ok(1),
            "assembly argmax must ignore NaN and preserve the earliest maximum tie"
        );

        let snapshot = state;
        let first = (0..32)
            .map(|_| engine.sample_f32(&logits, &stochastic, &mut state).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            first,
            [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1,
            ],
            "seeded sampler sequence is an ABI-format fixture"
        );
        assert_eq!(state.cursor, 64);
        assert_eq!(
            state.core,
            [
                1634760805, 857760878, 2036477234, 1797285236, 1658732843, 3034356692, 4033853308,
                4194450665, 3547450344, 3692221201, 2425913855, 2949776396, 4, 0, 2014243122,
                2946713300,
            ]
        );
        assert_eq!(
            state.block,
            [
                2383642325, 43725645, 1286118100, 2469055513, 777813414, 2667126283, 129165302,
                847218681, 4067806552, 303737036, 3222962614, 2784967409, 746176814, 3114314420,
                2338605642, 352298425,
            ]
        );
        assert!(
            first.iter().all(|token| matches!(*token, 1 | 3 | 5)),
            "threshold top-k must retain boundary ties and exclude lower logits"
        );
        let mut replay = snapshot;
        let second = (0..32)
            .map(|_| {
                engine
                    .sample_f32(&logits, &stochastic, &mut replay)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(first, second, "snapshot replay must preserve draw order");
        assert_eq!(state.cursor, replay.cursor);
        assert_eq!(state.core, replay.core);
        assert_eq!(state.block, replay.block);
    }

    #[test]
    fn typed_depthwise_stream_matches_split_buffer_oracle_and_uses_one_ticket() {
        use half::bf16;

        if !depthwise_stream_available() {
            eprintln!("depthwise stream opcodes unavailable on this runner - skipping");
            return;
        }
        let engine = NativeEngine::new(4).expect("native engine init");
        let before = engine.snapshot();
        let cases = [
            (1usize, 3usize, 1usize, 3usize, false),
            (2, 5, 4, 3, true),
            (1, 2, 2, 5, true),
            (1, 4, 9, 1, false),
        ];
        for (batch, channels, steps, kernel, resumed) in cases {
            let rows = batch * channels;
            let prior = kernel - 1;
            let bits = |i: usize, salt: usize| {
                bf16::from_f32((((i * 17 + salt * 11) % 41) as f32 - 20.0) / 13.0).to_bits()
            };
            let x = (0..rows * steps).map(|i| bits(i, 1)).collect::<Vec<_>>();
            let weights = (0..channels * kernel)
                .map(|i| bits(i, 2))
                .collect::<Vec<_>>();
            let cache = resumed.then(|| (0..rows * prior).map(|i| bits(i, 3)).collect::<Vec<_>>());
            let mut got = vec![0u16; rows * steps];
            let mut got_state = vec![0u16; rows * prior];
            assert!(engine.depthwise_stream_bf16(
                &x,
                cache.as_deref(),
                &weights,
                &mut got,
                &mut got_state,
                batch,
                channels,
                steps,
                kernel,
            ));

            let mut expected = vec![0u16; got.len()];
            let mut expected_state = vec![0u16; got_state.len()];
            for row in 0..rows {
                let channel = row % channels;
                for t in 0..steps {
                    let mut acc = 0.0f32;
                    for j in 0..kernel {
                        let source = t + j;
                        let value = if source < prior {
                            cache
                                .as_ref()
                                .map(|values| {
                                    bf16::from_bits(values[row * prior + source]).to_f32()
                                })
                                .unwrap_or(0.0)
                        } else {
                            bf16::from_bits(x[row * steps + source - prior]).to_f32()
                        };
                        let weight = bf16::from_bits(weights[channel * kernel + j]).to_f32();
                        acc = value.mul_add(weight, acc);
                    }
                    expected[row * steps + t] = bf16::from_f32(acc).to_bits();
                }
                for i in 0..prior {
                    let source = steps + i;
                    expected_state[row * prior + i] = if source < prior {
                        cache
                            .as_ref()
                            .map(|values| values[row * prior + source])
                            .unwrap_or(0)
                    } else {
                        x[row * steps + source - prior]
                    };
                }
            }
            assert_eq!(
                got, expected,
                "split stream mismatch for B={batch} D={channels} T={steps} K={kernel}"
            );
            assert_eq!(
                got_state, expected_state,
                "next-state mismatch for B={batch} D={channels} T={steps} K={kernel}"
            );
        }
        let after = engine.snapshot();
        assert_eq!(
            after.pass_submissions - before.pass_submissions,
            cases.len() as u64
        );
        assert_eq!(
            after.pass_completions - before.pass_completions,
            cases.len() as u64
        );
        assert_eq!(
            after.bridge_dispatches - before.bridge_dispatches,
            cases.len() as u64
        );
        assert!(after.fence_generations - before.fence_generations >= cases.len() as u64);
    }

    #[test]
    fn typed_gemm_layouts_and_gemv_use_one_ticket_each() {
        if !bf16_gemm_available() {
            eprintln!("native bf16 GEMM opcodes unavailable - skipping");
            return;
        }
        let engine = NativeEngine::new(4).expect("native engine init");
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| half::bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>()
        };
        let a = bits(&[1.0, 2.0, 3.0, 4.0]);
        let b_kn = bits(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let w_nk = bits(&[5.0, 8.0, 6.0, 9.0, 7.0, 10.0]);
        let expected = [21.0, 24.0, 27.0, 47.0, 54.0, 61.0];
        let before = engine.snapshot();

        let mut kn = [0.0f32; 6];
        assert!(engine.bf16_gemm_f32(&a, &b_kn, &mut kn, 2, 3, 2, false));
        assert_eq!(kn, expected);

        let mut nk = [0.0f32; 6];
        assert!(engine.bf16_gemm_f32(&a, &w_nk, &mut nk, 2, 3, 2, true));
        assert_eq!(nk, expected);

        let mut gemv = [0.0f32; 3];
        assert!(engine.bf16_gemm_f32(&a[..2], &b_kn, &mut gemv, 1, 3, 2, false));
        assert_eq!(gemv, expected[..3]);

        let after = engine.snapshot();
        assert_eq!(after.pass_submissions - before.pass_submissions, 3);
        assert_eq!(after.pass_completions - before.pass_completions, 3);
        assert_eq!(after.bridge_dispatches - before.bridge_dispatches, 3);
        assert_eq!(after.descriptors_live, 0);
    }

    #[test]
    fn direct_nk_gemm_streams_checkpoint_words_without_an_isa_fallback_copy() {
        let engine = NativeEngine::new(4).expect("native engine init");
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| half::bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>()
        };
        let a = bits(&[1.0, 2.0, 3.0, 4.0]);
        let w_nk = bits(&[5.0, 8.0, 6.0, 9.0, 7.0, 10.0]);
        let before = engine.snapshot();
        let mut out = [0.0f32; 6];
        assert!(engine.bf16_gemm_nt_direct_f32(&a, &w_nk, &mut out, 2, 3, 2));
        assert_eq!(out, [21.0, 24.0, 27.0, 47.0, 54.0, 61.0]);
        let after = engine.snapshot();
        assert_eq!(after.pass_submissions - before.pass_submissions, 1);
        assert_eq!(after.pass_completions - before.pass_completions, 1);
        assert_eq!(after.bridge_dispatches - before.bridge_dispatches, 1);
        assert_eq!(after.descriptors_live, 0);
    }

    #[test]
    fn direct_nk_scalar_leaf_accepts_byte_unaligned_checkpoint_views() {
        let pack = |values: &[f32]| {
            let mut bytes = vec![0xa5];
            for value in values {
                bytes.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
            }
            bytes
        };
        let a = pack(&[1.0, 2.0, 3.0, 4.0]);
        let weights = pack(&[5.0, 8.0, 6.0, 9.0, 7.0, 10.0]);
        let mut out = [0.0f32; 6];
        // SAFETY: the leaf contract explicitly permits unaligned bf16 byte
        // views. Both prefixed byte buffers contain the complete 2x2 and 3x2
        // little-endian matrices after byte zero.
        unsafe {
            lfm_bf16_gemm_nt_f32_scalar(
                a.as_ptr().add(1).cast(),
                weights.as_ptr().add(1).cast(),
                out.as_mut_ptr(),
                2,
                3,
                2,
            );
        }
        assert_eq!(out, [21.0, 24.0, 27.0, 47.0, 54.0, 61.0]);
    }

    #[test]
    fn direct_nk_ticket_accepts_byte_unaligned_checkpoint_weights() {
        let engine = NativeEngine::new(4).expect("native engine init");
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| half::bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>()
        };
        let a = bits(&[1.0, 2.0, 3.0, 4.0]);
        let mut storage = vec![0x5a];
        for word in bits(&[5.0, 8.0, 6.0, 9.0, 7.0, 10.0]) {
            storage.extend_from_slice(&word.to_le_bytes());
        }
        let mut out = [0.0f32; 6];
        let _pass = engine.pass_lock.lock().unwrap();
        // SAFETY: byte one starts a complete unaligned 3x2 little-endian BF16
        // checkpoint view. The ticket retains all buffers until completion.
        let status = unsafe {
            lfm_engine_bf16_gemm_nt_direct_f32(
                engine.ptr,
                a.as_ptr(),
                a.len(),
                storage.as_ptr().add(1).cast(),
                6,
                out.as_mut_ptr(),
                out.len(),
                2,
                3,
                2,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(out, [21.0, 24.0, 27.0, 47.0, 54.0, 61.0]);
    }

        #[test]
    fn direct_bf16_epilogues_match_materialized_primitives() {
        use half::bf16;

        const M: usize = 4;
        const N: usize = 5;
        const K: usize = 11;
        const STRIDE: usize = N + 2;
        const SENTINEL: u16 = 0xdead;

        assert_ne!(N % 4, 0, "fixture must exercise the N%4 tail");
        assert_ne!(K % 8, 0, "fixture must exercise the K%8 tail");

        let words = |count: usize, salt: usize| {
            (0..count)
                .map(|index| {
                    let value = ((index * 7 + salt) % 19 + 1) as f32 / 16.0;
                    bf16::from_f32(value).to_bits()
                })
                .collect::<Vec<_>>()
        };
        let pack = |words: &[u16], fill: u8| {
            let mut bytes = vec![fill; words.len() * 2 + 1];
            let start = usize::from((bytes.as_ptr() as usize) & 1 == 0);
            for (index, word) in words.iter().enumerate() {
                let offset = start + index * 2;
                bytes[offset..offset + 2].copy_from_slice(&word.to_le_bytes());
            }
            assert_eq!((unsafe { bytes.as_ptr().add(start) } as usize) & 1, 1);
            (bytes, start)
        };
        let gemm = |activation: &[u16],
                    weight: *const c_void,
                    out: &mut [f32],
                    rows: usize,
                    columns: usize,
                    inner: usize| {
            assert_eq!(activation.len(), rows * inner);
            assert_eq!(out.len(), rows * columns);
            let rows = i32::try_from(rows).expect("oracle rows fit i32");
            let columns = i32::try_from(columns).expect("oracle columns fit i32");
            let inner = i32::try_from(inner).expect("oracle inner fits i32");
            // Rosetta deliberately withholds AVX state. Its production leaves use
            // this scalar materialized primitive; native ARM and AVX-capable x86
            // use the tuned primitive. That selection keeps the reduction order
            // identical on both sides of the differential test without skipping.
            unsafe {
                if cfg!(target_arch = "x86_64") && !bf16_gemm_available() {
                    lfm_bf16_gemm_nt_f32_scalar(
                        activation.as_ptr(),
                        weight,
                        out.as_mut_ptr(),
                        rows,
                        columns,
                        inner,
                    );
                    return;
                }
                lfm_bf16_gemm_nt_f32(
                    activation.as_ptr(),
                    weight,
                    out.as_mut_ptr(),
                    rows,
                    columns,
                    inner,
                );
            }
        };
        let round = |values: &[f32]| {
            values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>()
        };
        let written = |name: &str, values: &[u16]| {
            assert!(
                values.iter().all(|value| *value & 0x7fff != 0),
                "{name} fixture must produce nonzero outputs: {values:04x?}"
            );
        };

        let activation = words(M * K, 3);
        let weights = pack(&words(N * K, 5), 0xa5);
        let gate = pack(&words(N * K, 9), 0xb6);
        let up = pack(&words(N * K, 13), 0xc7);
        let residual = pack(&words(N, 2), 0xd8);
        let bias = pack(&words(N, 7), 0xe9);
        let weight_ptr = unsafe { weights.0.as_ptr().add(weights.1).cast() };
        let gate_ptr = unsafe { gate.0.as_ptr().add(gate.1).cast() };
        let up_ptr = unsafe { up.0.as_ptr().add(up.1).cast() };
        let residual_ptr = unsafe { residual.0.as_ptr().add(residual.1).cast() };
        let bias_ptr = unsafe { bias.0.as_ptr().add(bias.1).cast() };

        // Prefill has no projection bias. Cover every admitted row count, an
        // unaligned checkpoint view, both vector tails, row-stride gaps, and
        // canaries before and after the complete destination plane.
        let mut sums = vec![0.0f32; M * N];
        gemm(&activation, weight_ptr, &mut sums, M, N, K);
        for rows in 1..=M {
            const PREFIX: usize = 3;
            const SUFFIX: usize = 4;
            let mut direct = vec![SENTINEL; PREFIX + rows * STRIDE + SUFFIX];
            let output = &mut direct[PREFIX..PREFIX + rows * STRIDE];
            unsafe {
                lfm_bf16_gemm_nt_bias_bf16(
                    activation.as_ptr(),
                    weight_ptr,
                    std::ptr::null(),
                    output.as_mut_ptr(),
                    rows as i32,
                    N as i32,
                    K as i32,
                    STRIDE as i32,
                );
            }
            let actual = output
                .chunks_exact(STRIDE)
                .flat_map(|row| row[..N].iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(actual, round(&sums[..rows * N]), "prefill M={rows}");
            written("direct unbiased prefill BF16 output", &actual);
            assert!(output
                .chunks_exact(STRIDE)
                .all(|row| row[N..].iter().all(|value| *value == SENTINEL)));
            assert!(direct[..PREFIX].iter().all(|value| *value == SENTINEL));
            assert!(direct[PREFIX + rows * STRIDE..]
                .iter()
                .all(|value| *value == SENTINEL));
        }

        let mut dots = vec![0.0f32; N];
        gemm(&activation[..K], weight_ptr, &mut dots, 1, N, K);
        let expected = round(&dots);
        let mut direct = vec![SENTINEL; N];
        unsafe {
            lfm_bf16_gemv_rne_bf16(
                activation.as_ptr().cast(),
                weight_ptr,
                direct.as_mut_ptr(),
                N,
                K,
            );
        }
        assert_eq!(direct, expected, "direct RNE GEMV");
        written("direct RNE GEMV", &direct);

        let projected = round(&dots);
        let mut added = projected
            .iter()
            .map(|value| bf16::from_bits(*value).to_f32())
            .collect::<Vec<_>>();
        unsafe {
            lfm_bf16_bias_add_f32(added.as_mut_ptr(), residual_ptr, N);
        }
        let expected = round(&added);
        let mut direct = vec![SENTINEL; N];
        unsafe {
            lfm_bf16_gemv_rne_add_bf16(
                activation.as_ptr().cast(),
                weight_ptr,
                residual_ptr,
                direct.as_mut_ptr(),
                N,
                K,
            );
        }
        assert_eq!(direct, expected, "direct RNE GEMV residual add");
        written("direct RNE GEMV residual add", &direct);

        let mut gates = vec![0.0f32; N];
        let mut ups = vec![0.0f32; N];
        gemm(&activation[..K], gate_ptr, &mut gates, 1, N, K);
        gemm(&activation[..K], up_ptr, &mut ups, 1, N, K);
        let mut expected = vec![0u16; N];
        unsafe {
            lfm_swiglu_bf16(
                gates.as_ptr(),
                ups.as_ptr(),
                expected.as_mut_ptr(),
                N as i32,
            );
        }
        let mut direct = vec![SENTINEL; N];
        unsafe {
            lfm_bf16_gemv_pair_swiglu_bf16(
                activation.as_ptr().cast(),
                gate_ptr,
                up_ptr,
                direct.as_mut_ptr(),
                N,
                K,
            );
        }
        assert_eq!(direct, expected, "paired direct SwiGLU");
        written("paired direct SwiGLU", &direct);

        for row in sums.chunks_exact_mut(N) {
            unsafe {
                lfm_bf16_bias_add_f32(row.as_mut_ptr(), bias_ptr, N);
            }
        }
        let expected = round(&sums);
        let mut direct = vec![SENTINEL; M * STRIDE];
        unsafe {
            lfm_bf16_gemm_nt_bias_bf16(
                activation.as_ptr(),
                weight_ptr,
                bias_ptr,
                direct.as_mut_ptr(),
                M as i32,
                N as i32,
                K as i32,
                STRIDE as i32,
            );
        }
        let actual = direct
            .chunks_exact(STRIDE)
            .flat_map(|row| row[..N].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "direct bias BF16 output");
        written("direct bias BF16 output", &actual);
        assert!(direct
            .chunks_exact(STRIDE)
            .all(|row| row[N..].iter().all(|value| *value == SENTINEL)));
    }

    #[test]
    fn small_prefill_linears_publish_directly_at_the_bf16_boundary() {
        let source = include_str!("../../../../liquid-audio/native/src/engine/flashkern_engine.cpp");
        let workspace = source
            .split("struct PrefillWorkspace")
            .nth(1)
            .expect("prefill workspace")
            .split("struct PrefillReq")
            .next()
            .expect("prefill workspace boundary");
        let prefill = source
            .split("static void prefill_band")
            .nth(1)
            .expect("prefill implementation")
            .split("static void run_prng_pass")
            .next()
            .expect("prefill implementation boundary");
        assert!(prefill.contains("static void prefill_linear_bf16"));
        assert_eq!(prefill.matches("prefill_linear_bf16(e, lane").count(), 7);
        assert!(!prefill.contains("prefill_round"));
        assert!(!workspace.contains("bcxf"));
        assert!(!workspace.contains("qkvf"));
        assert!(!workspace.contains("projf"));
    }

    #[test]
    fn typed_fft_grids_use_one_ticket_each() {
        let engine = NativeEngine::new(4).expect("native engine init");
        let (batch, channels, steps, fft) = (1usize, 2usize, 4usize, 8usize);
        let input = [1.0f32, -2.0, 3.0, -4.0, 2.0, 4.0, -6.0, -8.0];
        let kernel = vec![0.0f32; channels * (fft / 2 + 1) * 2];
        let skip = [1.0f32, -0.5];
        let mut conv = [0.0f32; 8];
        let before = engine.snapshot();
        assert!(engine.fft_conv_dd(&input, &kernel, &skip, &mut conv, batch, channels, steps, fft,));
        assert_eq!(conv, [1.0, -2.0, 3.0, -4.0, -1.0, -2.0, 3.0, 4.0]);

        let rows = 3usize;
        let frequency = fft / 2 + 1;
        let real = vec![0.0f32; rows * frequency];
        let imag = vec![0.0f32; rows * frequency];
        let mut inverse = vec![1.0f32; rows * fft];
        assert!(engine.irfft_dd(
            &real,
            &imag,
            &mut inverse,
            rows,
            fft,
            crate::flashkern::dd::Dd::from_f32(1.0 / fft as f32),
        ));
        assert!(inverse.iter().all(|value| *value == 0.0));

        let after = engine.snapshot();
        assert_eq!(after.pass_submissions - before.pass_submissions, 2);
        assert_eq!(after.pass_completions - before.pass_completions, 2);
        assert_eq!(after.bridge_dispatches - before.bridge_dispatches, 2);
        assert_eq!(after.descriptors_live, 0);
        // fft=8 has three radix-2 stages in each direction. Every signal crosses
        // init, bit-reversal/stages, product/mirror, inverse, and output barriers;
        // lane_program adds the one pass-completion fence.
        // The typed IRFFT pass below contributes its final pass fence as well.
        assert_eq!(after.fence_generations - before.fence_generations, 30);
    }

    #[test]
    fn typed_fft_is_bit_exact_across_physical_lane_counts() {
        let single = NativeEngine::new(1).expect("single-lane native engine init");
        let group = NativeEngine::new(4).expect("four-lane native engine init");
        let (batch, channels, steps, fft) = (2usize, 2usize, 7usize, 16usize);
        let input = (0..batch * channels * steps)
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.0625 - 0.75)
            .collect::<Vec<_>>();
        let kernel = (0..channels * (fft / 2 + 1) * 2)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let skip = [0.375f32, -0.625];
        let mut expected = vec![0.0f32; input.len()];
        let mut actual = vec![0.0f32; input.len()];
        assert!(single.fft_conv_dd(
            &input,
            &kernel,
            &skip,
            &mut expected,
            batch,
            channels,
            steps,
            fft,
        ));
        let before = group.snapshot();
        assert!(group.fft_conv_dd(
            &input,
            &kernel,
            &skip,
            &mut actual,
            batch,
            channels,
            steps,
            fft,
        ));
        let after = group.snapshot();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(after.pass_submissions - before.pass_submissions, 1);
        assert_eq!(after.pass_completions - before.pass_completions, 1);
        assert_eq!(after.bridge_dispatches - before.bridge_dispatches, 1);
        // Four signals, four radix-2 stages in each direction, plus the final
        // pass fence. This proves the physical lane team executed the staged grid.
        assert_eq!(after.fence_generations - before.fence_generations, 65);
        assert_eq!(after.descriptors_live, 0);
    }

    #[test]
    fn typed_depth_plans_coexist_and_use_one_ticket_per_frame() {
        use crate::flashkern::decode::{DepthHead, DepthLayer, DepthPlan, PtrLen};

        let engine = NativeEngine::new(4).expect("native engine init");
        let (dim, ffn, codebooks, backbone) = (4usize, 4usize, 2usize, 4usize);
        let (heads, kv_heads, head_dim, vocab) = (1usize, 1usize, 4usize, 3usize);
        let qkv = dim + 2 * kv_heads * head_dim;
        let zeros_qkv = vec![0u16; qkv * dim];
        let zeros_square = vec![0u16; dim * dim];
        let ones = vec![0x3f80u16; dim];
        let zeros_ffn = vec![0u16; ffn * dim];
        let layer = DepthLayer {
            qkv_w: PtrLen::from_u16(&zeros_qkv),
            out_w: PtrLen::from_u16(&zeros_square),
            q_ln: PtrLen::from_u16(&ones),
            k_ln: PtrLen::from_u16(&ones),
            opnorm: PtrLen::from_u16(&ones),
            ffnnorm: PtrLen::from_u16(&ones),
            w1: PtrLen::from_u16(&zeros_ffn),
            w3: PtrLen::from_u16(&zeros_ffn),
            w2: PtrLen::from_u16(&zeros_ffn),
        };
        let layers = [layer];
        let table = vec![0u16; vocab * dim];
        let depth_heads = [
            DepthHead {
                emb: PtrLen::from_u16(&table),
                norm: PtrLen::from_u16(&ones),
                logits: PtrLen::from_u16(&table),
                vocab,
            },
            DepthHead {
                emb: PtrLen::from_u16(&table),
                norm: PtrLen::from_u16(&ones),
                logits: PtrLen::from_u16(&table),
                vocab,
            },
        ];
        let depth_w = vec![0u16; codebooks * dim * backbone];
        let depth_b = vec![0u16; codebooks * dim];
        let rope_cos = vec![1.0f32; codebooks * head_dim / 2];
        let rope_sin = vec![0.0f32; rope_cos.len()];
        let plan = DepthPlan {
            size: std::mem::size_of::<DepthPlan>() as u32,
            abi_version: 1,
            dim: dim as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            ffn_dim: ffn as u32,
            codebooks: codebooks as u32,
            backbone_dim: backbone as u32,
            eps: 1e-5,
            depth_linear_w: PtrLen::from_u16(&depth_w),
            depth_linear_b: PtrLen::from_u16(&depth_b),
            rope_cos: PtrLen::from_f32(&rope_cos),
            rope_sin: PtrLen::from_f32(&rope_sin),
            layers: layers.as_ptr(),
            layer_count: layers.len(),
            codebook_heads: depth_heads.as_ptr(),
            codebook_head_count: depth_heads.len(),
        };

        let first = engine.depth_build(&plan).expect("first depth plan");
        let second = engine.depth_build(&plan).expect("second depth plan");
        assert_ne!(first, second, "depth plan identities must be unique");
        let hidden = [0u16; 4];
        let config = SampleConfig::new(None, None);
        let mut state = PrngState::from_seed(17).expect("seed");
        let mut tokens = [u32::MAX; 2];
        let before = engine.snapshot();
        assert!(engine.depth_frame(first, &hidden, &config, &mut state, &mut tokens));
        let after = engine.snapshot();
        assert_eq!(tokens, [0, 0]);
        assert_eq!(after.pass_submissions - before.pass_submissions, 1);
        assert_eq!(after.pass_completions - before.pass_completions, 1);
        assert_eq!(after.bridge_dispatches - before.bridge_dispatches, 1);
        assert!(after.fence_generations > before.fence_generations);
        assert_eq!(after.descriptors_live, 0);

        engine.depth_clear(first);
        assert!(!engine.depth_frame(first, &hidden, &config, &mut state, &mut tokens));
        assert!(engine.depth_frame(second, &hidden, &config, &mut state, &mut tokens));
        engine.depth_clear(second);
        assert!(!engine.depth_frame(second, &hidden, &config, &mut state, &mut tokens));
    }

    #[test]
    fn audio_recurrence_route_matches_split_passes_and_commits_before_depth_failure() {
        use crate::flashkern::decode::{DepthHead, DepthLayer, DepthPlan, PtrLen};
        use half::bf16;

        fn weights(count: usize, seed: usize, scale: f32) -> Vec<u16> {
            (0..count)
                .map(|index| {
                    let value = ((index.wrapping_mul(17).wrapping_add(seed * 11)) % 19) as i32 - 9;
                    bf16::from_f32(value as f32 * scale).to_bits()
                })
                .collect()
        }

        let _guard = CTX_TEST_LOCK.lock().unwrap();
        let (hidden, ffn, kernel, max_ctx, vocab) = (4usize, 4usize, 2usize, 8usize, 3usize);
        let norm = vec![0x3f80u16; hidden];
        let input = weights(3 * hidden * hidden, 1, 0.03125);
        let conv = weights(hidden * kernel, 2, 0.0625);
        let output = weights(hidden * hidden, 3, 0.03125);
        let w1 = weights(ffn * hidden, 4, 0.03125);
        let w3 = weights(ffn * hidden, 5, 0.03125);
        let w2 = weights(hidden * ffn, 6, 0.03125);
        let embed = weights(vocab * hidden, 7, 0.125);
        let descriptor = LayerDesc {
            kind: 0,
            k: kernel as u32,
            op_eps: 1e-5,
            ffn_eps: 1e-5,
            op_norm_w: norm.as_ptr(),
            ffn_norm_w: norm.as_ptr(),
            in_w: input.as_ptr(),
            conv_w: conv.as_ptr(),
            out_w: output.as_ptr(),
            w1: w1.as_ptr(),
            w3: w3.as_ptr(),
            w2: w2.as_ptr(),
            ..LayerDesc::attn_placeholder()
        };

        let (dim, depth_ffn, codebooks, depth_vocab) = (4usize, 4usize, 8usize, 5usize);
        let (heads, kv_heads, head_dim) = (1usize, 1usize, 4usize);
        let qkv = vec![0u16; (dim + 2 * kv_heads * head_dim) * dim];
        let square = vec![0u16; dim * dim];
        let feed = vec![0u16; depth_ffn * dim];
        let depth_layer = DepthLayer {
            qkv_w: PtrLen::from_u16(&qkv),
            out_w: PtrLen::from_u16(&square),
            q_ln: PtrLen::from_u16(&norm),
            k_ln: PtrLen::from_u16(&norm),
            opnorm: PtrLen::from_u16(&norm),
            ffnnorm: PtrLen::from_u16(&norm),
            w1: PtrLen::from_u16(&feed),
            w3: PtrLen::from_u16(&feed),
            w2: PtrLen::from_u16(&feed),
        };
        let depth_layers = [depth_layer];
        let table = vec![0u16; depth_vocab * dim];
        let depth_heads = (0..codebooks)
            .map(|_| DepthHead {
                emb: PtrLen::from_u16(&table),
                norm: PtrLen::from_u16(&norm),
                logits: PtrLen::from_u16(&table),
                vocab: depth_vocab,
            })
            .collect::<Vec<_>>();
        let depth_w = vec![0u16; codebooks * dim * hidden];
        let depth_b = vec![0u16; codebooks * dim];
        let rope_cos = vec![1.0f32; codebooks * head_dim / 2];
        let rope_sin = vec![0.0f32; rope_cos.len()];
        let depth_plan = DepthPlan {
            size: std::mem::size_of::<DepthPlan>() as u32,
            abi_version: 1,
            dim: dim as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            ffn_dim: depth_ffn as u32,
            codebooks: codebooks as u32,
            backbone_dim: hidden as u32,
            eps: 1e-5,
            depth_linear_w: PtrLen::from_u16(&depth_w),
            depth_linear_b: PtrLen::from_u16(&depth_b),
            rope_cos: PtrLen::from_f32(&rope_cos),
            rope_sin: PtrLen::from_f32(&rope_sin),
            layers: depth_layers.as_ptr(),
            layer_count: depth_layers.len(),
            codebook_heads: depth_heads.as_ptr(),
            codebook_head_count: depth_heads.len(),
        };

        let split = NativeEngine::new(4).expect("split engine init");
        let routed = NativeEngine::new(4).expect("routed engine init");
        let split_model = split
            .ctx_build(&[descriptor], hidden, ffn, max_ctx)
            .expect("split backbone plan");
        let routed_model = routed
            .ctx_build(&[descriptor], hidden, ffn, max_ctx)
            .expect("routed backbone plan");
        for (engine, id) in [(&split, split_model), (&routed, routed_model)] {
            assert!(unsafe {
                engine.set_heads(
                    id,
                    embed.as_ptr(),
                    embed.len(),
                    vocab,
                    std::ptr::null(),
                    0,
                    0,
                    norm.as_ptr(),
                    norm.len(),
                    1e-5,
                )
            });
        }
        let split_depth = split.depth_build(&depth_plan).expect("split depth plan");
        let routed_depth = routed.depth_build(&depth_plan).expect("routed depth plan");

        let ids = [1u32];
        let initial_carry = weights(hidden * (kernel - 1), 8, 0.0625);
        let mut stale_keys = vec![0x3555u16; max_ctx * hidden];
        let mut stale_values = vec![0xb555u16; max_ctx * hidden];
        let mut stale_carry = initial_carry.clone();
        let stale_states = [LayerState {
            k_plane: stale_keys.as_mut_ptr(),
            v_plane: stale_values.as_mut_ptr(),
            head_stride: max_ctx * hidden,
            k_len: stale_keys.len(),
            v_len: stale_values.len(),
            conv_state: stale_carry.as_mut_ptr(),
            conv_len: stale_carry.len(),
        }];
        let expected_keys = stale_keys.clone();
        let expected_values = stale_values.clone();
        let expected_carry = stale_carry.clone();
        let mut stale_hidden = vec![0x7fc1u16; hidden];
        let expected_hidden = stale_hidden.clone();
        let sampler = SampleConfig::new(Some(1.0), None);
        let mut stale_prng = PrngState::from_seed(0x51eed).expect("stale seed");
        let expected_prng = stale_prng;
        let mut stale_codes = vec![u32::MAX; codebooks];
        let expected_codes = stale_codes.clone();
        let mut stale_window = ContextWindow {
            capacity: max_ctx as u64,
            runway: 4,
            position: 0,
            start: 0,
            cursor: 0,
            rope_base: 0,
        };
        let stale_before = routed.snapshot();
        for (position, expected_start) in [(1usize, 0u64), (0, 1)] {
            let mut token_completed = 0u32;
            let mut token_committed = 0u32;
            let commit = TokenCommitRecord {
                window: &mut stale_window,
                expected_position: stale_window.position,
                expected_start,
                expected_cursor: stale_window.cursor,
                expected_rope_base: stale_window.rope_base,
                token_committed: &mut token_committed,
            };
            let status = {
                let _pass = routed.pass_lock.lock().unwrap();
                unsafe {
                    lfm_engine_audio_recurrence(
                        routed.ptr,
                        routed_model,
                        routed_depth,
                        ids.as_ptr(),
                        ids.len(),
                        0,
                        stale_states.as_ptr(),
                        stale_states.len(),
                        position,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        stale_hidden.as_mut_ptr(),
                        stale_hidden.len(),
                        &sampler,
                        &mut stale_prng,
                        stale_codes.as_mut_ptr(),
                        stale_codes.len(),
                        4,
                        &commit,
                        &mut token_completed,
                    )
                }
            };
            assert_eq!(status, -libc::ESTALE);
            assert_eq!((token_completed, token_committed), (0, 0));
        }
        let stale_after = routed.snapshot();
        assert_eq!(
            stale_keys, expected_keys,
            "KV keys changed before admission"
        );
        assert_eq!(
            stale_values, expected_values,
            "KV values changed before admission"
        );
        assert_eq!(
            stale_carry, expected_carry,
            "ShortConv carry changed before admission"
        );
        assert_eq!(
            stale_hidden, expected_hidden,
            "hidden output changed before admission"
        );
        assert_eq!(
            stale_codes, expected_codes,
            "Depth output changed before admission"
        );
        assert_eq!(stale_prng.cursor, expected_prng.cursor);
        assert_eq!(stale_prng.core, expected_prng.core);
        assert_eq!(stale_prng.block, expected_prng.block);
        assert_eq!(
            (
                stale_window.position,
                stale_window.start,
                stale_window.cursor,
                stale_window.rope_base,
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(stale_after.pass_submissions, stale_before.pass_submissions);
        assert_eq!(stale_after.pass_completions, stale_before.pass_completions);
        assert_eq!(
            stale_after.bridge_dispatches,
            stale_before.bridge_dispatches
        );
        assert_eq!(stale_after.pass_slots_live, stale_before.pass_slots_live);
        assert_eq!(stale_after.descriptors_live, stale_before.descriptors_live);

        let mut split_carry = initial_carry.clone();
        let split_states = [LayerState {
            conv_state: split_carry.as_mut_ptr(),
            conv_len: split_carry.len(),
            ..LayerState::none()
        }];
        let mut routed_carry = initial_carry.clone();
        let routed_states = [LayerState {
            conv_state: routed_carry.as_mut_ptr(),
            conv_len: routed_carry.len(),
            ..LayerState::none()
        }];
        let mut split_hidden = vec![0u16; hidden];
        let mut routed_hidden = vec![0u16; hidden];
        let mut split_prng = PrngState::from_seed(0x51eed).expect("split seed");
        let mut routed_prng = PrngState::from_seed(0x51eed).expect("routed seed");
        let mut split_codes = vec![u32::MAX; codebooks];
        let mut routed_codes = vec![u32::MAX; codebooks];

        let split_before = split.snapshot();
        assert!(unsafe {
            split.token_pass(
                split_model,
                &ids,
                0,
                &split_states,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut split_hidden,
                None,
                None,
                None,
                None,
                4,
            )
        });
        assert!(split.depth_frame(
            split_depth,
            &split_hidden,
            &sampler,
            &mut split_prng,
            &mut split_codes,
        ));
        let split_after = split.snapshot();

        let mut window = ContextWindow {
            capacity: max_ctx as u64,
            runway: 4,
            position: 0,
            start: 0,
            cursor: 0,
            rope_base: 0,
        };
        let mut token_completed = 0u32;
        let mut token_committed = 0u32;
        let commit = TokenCommitRecord {
            window: &mut window,
            expected_position: window.position,
            expected_start: window.start,
            expected_cursor: window.cursor,
            expected_rope_base: window.rope_base,
            token_committed: &mut token_committed,
        };
        let route_before = routed.snapshot();
        let route_status = {
            let _pass = routed.pass_lock.lock().unwrap();
            unsafe {
                lfm_engine_audio_recurrence(
                    routed.ptr,
                    routed_model,
                    routed_depth,
                    ids.as_ptr(),
                    ids.len(),
                    0,
                    routed_states.as_ptr(),
                    routed_states.len(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    routed_hidden.as_mut_ptr(),
                    routed_hidden.len(),
                    &sampler,
                    &mut routed_prng,
                    routed_codes.as_mut_ptr(),
                    routed_codes.len(),
                    4,
                    &commit,
                    &mut token_completed,
                )
            }
        };
        let route_after = routed.snapshot();
        assert_eq!(route_status, 0);
        assert_eq!((token_completed, token_committed), (1, 1));
        assert_eq!((window.position, window.cursor), (1, 1));
        assert_eq!(routed_hidden, split_hidden);
        assert_eq!(routed_carry, split_carry);
        assert_eq!(routed_codes, split_codes);
        assert_eq!(routed_prng.cursor, split_prng.cursor);
        assert_eq!(routed_prng.core, split_prng.core);
        assert_eq!(routed_prng.block, split_prng.block);
        assert_eq!(
            split_after.pass_submissions - split_before.pass_submissions,
            2
        );
        assert_eq!(
            split_after.pass_completions - split_before.pass_completions,
            2
        );
        assert_eq!(
            split_after.continuation_submissions - split_before.continuation_submissions,
            0
        );
        assert_eq!(
            route_after.pass_submissions - route_before.pass_submissions,
            2
        );
        assert_eq!(
            route_after.pass_completions - route_before.pass_completions,
            2
        );
        assert_eq!(
            route_after.bridge_dispatches - route_before.bridge_dispatches,
            2
        );
        assert_eq!(
            route_after.continuation_submissions - route_before.continuation_submissions,
            2
        );
        assert_eq!(route_after.max_pass_slots_live, 1);
        assert_eq!(route_after.pass_slots_live, 0);
        assert_eq!(route_after.route_capacity, 64);
        assert_eq!((route_after.routes_live, route_after.routes_ready), (0, 0));
        assert_eq!(
            route_after.route_dispatches - route_before.route_dispatches,
            2
        );
        assert_eq!(route_after.descriptors_live, 0);
        assert_eq!(
            route_after.descriptor_acquires - route_before.descriptor_acquires,
            2
        );
        assert_eq!(
            route_after.descriptor_retains - route_before.descriptor_retains,
            2
        );
        assert_eq!(
            route_after.descriptor_releases - route_before.descriptor_releases,
            4
        );

        // Text recurrence is the same broker with a terminal TOKEN node. Its
        // callback is only a doorbell edge; the caller performs exact-handle
        // collection and observes the committed state afterward.
        extern "C" fn text_notify(context: *mut c_void) {
            let notified = unsafe { &*(context as *const std::sync::atomic::AtomicU32) };
            notified.store(1, std::sync::atomic::Ordering::Release);
        }
        let mut split_text_hidden = vec![0u16; hidden];
        let mut routed_text_hidden = vec![0u16; hidden];
        let mut split_text_prng = PrngState::from_seed(0x91eed).expect("split text seed");
        let mut routed_text_prng = PrngState::from_seed(0x91eed).expect("route text seed");
        let text_sampler = SampleConfig::new(None, None);
        let mut split_token = u32::MAX;
        assert!(unsafe {
            split.token_pass(
                split_model,
                &ids,
                0,
                &split_states,
                1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut split_text_hidden,
                None,
                Some(&text_sampler),
                Some(&mut split_text_prng),
                Some(&mut split_token),
                4,
            )
        });
        let mut text_window = ContextWindow {
            capacity: max_ctx as u64,
            runway: 4,
            position: 1,
            start: 0,
            cursor: 1,
            rope_base: 0,
        };
        let mut text_completed = 0u32;
        let mut text_committed = 0u32;
        let mut routed_token = u32::MAX;
        let text_commit = TokenCommitRecord {
            window: &mut text_window,
            expected_position: 1,
            expected_start: 0,
            expected_cursor: 1,
            expected_rope_base: 0,
            token_committed: &mut text_committed,
        };
        let text_notified = std::sync::atomic::AtomicU32::new(0);
        let mut text_handle = AudioRouteHandle::default();
        assert_eq!(
            unsafe {
                lfm_engine_token_route_submit(
                    routed.ptr,
                    routed_model,
                    ids.as_ptr(),
                    ids.len(),
                    0,
                    routed_states.as_ptr(),
                    routed_states.len(),
                    1,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    routed_text_hidden.as_mut_ptr(),
                    routed_text_hidden.len(),
                    &text_sampler,
                    &mut routed_text_prng,
                    &mut routed_token,
                    4,
                    &text_commit,
                    &mut text_completed,
                    text_notify,
                    (&text_notified as *const std::sync::atomic::AtomicU32)
                        .cast_mut()
                        .cast(),
                    &mut text_handle,
                )
            },
            0
        );
        let text_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while text_notified.load(std::sync::atomic::Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < text_deadline,
                "text route timed out"
            );
            std::thread::yield_now();
        }
        assert_eq!(
            unsafe { lfm_engine_audio_route_collect(routed.ptr, &mut text_handle) },
            0
        );
        assert_eq!((text_completed, text_committed), (1, 1));
        assert_eq!((text_window.position, text_window.cursor), (2, 2));
        assert_eq!(routed_text_hidden, split_text_hidden);
        assert_eq!(routed_carry, split_carry);
        assert_eq!(routed_token, split_token);
        assert_eq!(routed_text_prng.cursor, split_text_prng.cursor);

        let mut failed_carry = initial_carry;
        let failed_states = [LayerState {
            conv_state: failed_carry.as_mut_ptr(),
            conv_len: failed_carry.len(),
            ..LayerState::none()
        }];
        let mut failed_hidden = vec![0u16; hidden];
        let mut failed_prng = PrngState::from_seed(0x51eed).expect("failed seed");
        let mut failed_codes = vec![u32::MAX; codebooks];
        let mut failed_window = ContextWindow {
            capacity: max_ctx as u64,
            runway: 4,
            position: 0,
            start: 0,
            cursor: 0,
            rope_base: 0,
        };
        let mut failed_completed = 0u32;
        let mut failed_committed = 0u32;
        let failed_commit = TokenCommitRecord {
            window: &mut failed_window,
            expected_position: failed_window.position,
            expected_start: failed_window.start,
            expected_cursor: failed_window.cursor,
            expected_rope_base: failed_window.rope_base,
            token_committed: &mut failed_committed,
        };
        assert_eq!(
            unsafe { lfm_internal_engine_fail_audio_route_depth_for_test(routed.ptr, -libc::EIO) },
            0
        );
        let failed_before = routed.snapshot();
        let failed_status = {
            let _pass = routed.pass_lock.lock().unwrap();
            unsafe {
                lfm_engine_audio_recurrence(
                    routed.ptr,
                    routed_model,
                    routed_depth,
                    ids.as_ptr(),
                    ids.len(),
                    0,
                    failed_states.as_ptr(),
                    failed_states.len(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    failed_hidden.as_mut_ptr(),
                    failed_hidden.len(),
                    &sampler,
                    &mut failed_prng,
                    failed_codes.as_mut_ptr(),
                    failed_codes.len(),
                    4,
                    &failed_commit,
                    &mut failed_completed,
                )
            }
        };
        let failed_after = routed.snapshot();
        assert_eq!(failed_status, -libc::EIO);
        assert_eq!((failed_completed, failed_committed), (1, 1));
        assert_eq!((failed_window.position, failed_window.cursor), (1, 1));
        assert_eq!(failed_hidden, split_hidden);
        assert_eq!(failed_carry, split_carry);
        assert_eq!(
            failed_after.pass_submissions - failed_before.pass_submissions,
            2
        );
        assert_eq!(
            failed_after.pass_completions - failed_before.pass_completions,
            2
        );
        assert_eq!(failed_after.pass_slots_live, 0);
        assert_eq!(failed_after.descriptors_live, 0);

        // Drive the real third SQ/lane/CQ node without a Mimi checkpoint. The
        // one-shot fault is consumed by REQ_MIMI_DECODE before its opaque state
        // is dereferenced, proving exact-slot routing and terminal cleanup.
        let mut mimi_carry = weights(hidden * (kernel - 1), 8, 0.0625);
        let mimi_states = [LayerState {
            conv_state: mimi_carry.as_mut_ptr(),
            conv_len: mimi_carry.len(),
            ..LayerState::none()
        }];
        let mut mimi_hidden = vec![0u16; hidden];
        let mut mimi_prng = PrngState::from_seed(0x51eed).expect("mimi seed");
        let mut mimi_window = ContextWindow {
            capacity: max_ctx as u64,
            runway: 4,
            position: 0,
            start: 0,
            cursor: 0,
            rope_base: 0,
        };
        let mut result = AudioRouteResult::default();
        let commit = TokenCommitRecord {
            window: &mut mimi_window,
            expected_position: 0,
            expected_start: 0,
            expected_cursor: 0,
            expected_rope_base: 0,
            token_committed: &mut result.token_committed,
        };
        let epoch = unsafe { lfm_internal_audio_route_epoch_new_for_test(7) };
        assert!(!epoch.is_null());
        let mut pcm = vec![0.0f32; 3840];
        let target = AudioRouteTarget {
            epoch,
            expected_epoch: 7,
            pcm: pcm.as_mut_ptr(),
            pcm_capacity: pcm.len(),
            codec_pcm: std::ptr::null_mut(),
            codec_pcm_capacity: 0,
            resampler_stream: std::ptr::null_mut(),
        };
        assert_eq!(
            unsafe { lfm_internal_engine_fail_audio_route_mimi_for_test(routed.ptr, -libc::EIO) },
            0
        );
        let mimi_before = routed.snapshot();
        extern "C" fn notify(context: *mut c_void) {
            let notified = unsafe { &*(context as *const std::sync::atomic::AtomicU32) };
            notified.store(1, std::sync::atomic::Ordering::Release);
        }
        let notified = std::sync::atomic::AtomicU32::new(0);
        let mut handle = AudioRouteHandle::default();
        let submit_status = unsafe {
            lfm_engine_audio_route_submit(
                routed.ptr,
                routed_model,
                routed_depth,
                ids.as_ptr(),
                ids.len(),
                0,
                mimi_states.as_ptr(),
                mimi_states.len(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                mimi_hidden.as_mut_ptr(),
                mimi_hidden.len(),
                &sampler,
                &mut mimi_prng,
                1usize as *mut c_void,
                &target,
                &mut result,
                4,
                &commit,
                notify,
                (&notified as *const std::sync::atomic::AtomicU32)
                    .cast_mut()
                    .cast(),
                &mut handle,
            )
        };
        assert_eq!(submit_status, 0);
        assert!(!handle.record.is_null());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while notified.load(std::sync::atomic::Ordering::Acquire) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "route notification timed out"
            );
            std::thread::yield_now();
        }
        let mimi_status = unsafe { lfm_engine_audio_route_collect(routed.ptr, &mut handle) };
        let mimi_after = routed.snapshot();
        unsafe { lfm_internal_audio_route_epoch_free_for_test(epoch) };
        assert_eq!(mimi_status, -libc::EIO);
        assert!(handle.record.is_null());
        assert_eq!(result.status, -libc::EIO);
        assert_eq!((result.token_completed, result.token_committed), (1, 1));
        assert_eq!(result.depth_completed, 1);
        assert_eq!(
            (result.mimi_completed, result.eoaudio, result.pcm_samples),
            (0, 0, 0)
        );
        assert_eq!(mimi_window.position, 1);
        assert_eq!(
            mimi_after.pass_submissions - mimi_before.pass_submissions,
            3
        );
        assert_eq!(
            mimi_after.pass_completions - mimi_before.pass_completions,
            3
        );
        assert_eq!(mimi_after.pass_slots_live, 0);
        assert_eq!((mimi_after.routes_live, mimi_after.routes_ready), (0, 0));
        assert_eq!(
            mimi_after.route_dispatches - mimi_before.route_dispatches,
            3
        );
        assert_eq!(mimi_after.descriptors_live, 0);
        assert_eq!(
            mimi_after.descriptor_acquires - mimi_before.descriptor_acquires,
            3
        );
        assert_eq!(
            mimi_after.descriptor_retains - mimi_before.descriptor_retains,
            3
        );
        assert_eq!(
            mimi_after.descriptor_releases - mimi_before.descriptor_releases,
            6
        );

        split.depth_clear(split_depth);
        routed.depth_clear(routed_depth);
        split.ctx_clear(split_model);
        routed.ctx_clear(routed_model);
    }

    #[test]
    fn native_prng_replays_a_partially_consumed_block() {
        let engine = NativeEngine::new(3).expect("native engine init");
        let mut state = PrngState::from_material(&[0; 32], &[0; 8]).expect("material seed");
        let mut prefix = [0u64; 3];
        assert!(engine.prng_fill(&mut state, &mut prefix));
        assert_eq!(
            prefix,
            [
                0x903d_f1a0_ade0_b876,
                0x28bd_8653_e56a_5d40,
                0x1aed_8da0_b819_d2bd,
            ]
        );
        assert_eq!(state.cursor, 24);
        assert_eq!(state.core[12], 1);

        // The snapshot lands inside the cached block. Continuation consumes its
        // remaining five draws, then crosses into the next assembly block.
        let mut replay = state;
        let mut continued = [0u64; 11];
        let mut replayed = [0u64; 11];
        assert!(engine.prng_fill(&mut state, &mut continued));
        assert!(engine.prng_fill(&mut replay, &mut replayed));
        assert_eq!(replayed, continued);
        assert_eq!(replay.size, state.size);
        assert_eq!(replay.abi_version, state.abi_version);
        assert_eq!(replay.cursor, state.cursor);
        assert_eq!(replay.flags, state.flags);
        assert_eq!(replay.core, state.core);
        assert_eq!(replay.block, state.block);
        assert_eq!(replay.reserved, state.reserved);
        assert_eq!(state.cursor, 48);
        assert_eq!(state.core[12], 2);
    }

    #[test]
    fn native_prng_accepts_system_entropy() {
        let engine = NativeEngine::new(2).expect("native engine init");
        let mut state = PrngState::from_system().expect("platform CSPRNG seed");
        assert_eq!(state.size as usize, std::mem::size_of::<PrngState>());
        assert_eq!(state.abi_version, 1);
        assert_eq!(state.cursor, 64);
        assert_eq!(state.flags & 1, 1, "system-seeded provenance bit");
        let mut draws = [0u64; 8];
        assert!(engine.prng_fill(&mut state, &mut draws));
        assert_eq!(state.cursor, 64);
        assert_eq!(state.core[12], 1);
    }

    #[test]
        fn probe_candle_bf16_sum_ladder() {
        use candle_core::{DType, Device, Tensor};
        use half::bf16;
        let dev = Device::Cpu;
        let (rows, h) = (8usize, 2048usize);
        let vals: Vec<f32> = (0..rows * h)
            .map(|j| (((j * 2654435761usize.wrapping_add(7)) % 2000) as f32 / 700.0) - 1.4)
            .collect();
        let t = Tensor::from_vec(vals.clone(), (rows, h), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let want: Vec<u16> = t
            .sum(0)
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap()
            .iter()
            .map(|b| b.to_bits())
            .collect();
        let bits: Vec<u16> = t
            .flatten_all()
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap()
            .iter()
            .map(|b| b.to_bits())
            .collect();
        // ladder A: sequential bf16 rounds (what the engine does today)
        let mut seq_bf = vec![0u16; h];
        for r in 0..rows {
            for j in 0..h {
                let a = f32::from_bits((seq_bf[j] as u32) << 16);
                let b = f32::from_bits((bits[r * h + j] as u32) << 16);
                let sum = a + b;
                let u = sum.to_bits();
                seq_bf[j] = ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16;
            }
        }
        // ladder B: f32 accumulate, one final round
        let mut f32acc = vec![0u16; h];
        for j in 0..h {
            let mut acc = 0f32;
            for r in 0..rows {
                acc += f32::from_bits((bits[r * h + j] as u32) << 16);
            }
            let u = acc.to_bits();
            f32acc[j] = ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16;
        }
        let a_match = seq_bf == want;
        let b_match = f32acc == want;
        eprintln!("candle sum(0) bf16: sequential-bf16 ladder match = {a_match}, f32-accumulate match = {b_match}");
        assert!(a_match || b_match, "neither ladder matches candle sum(0)");
    }

    #[test]
        fn native_engine_attn_layer_bit_parity() {
        let _ctx = CTX_TEST_LOCK.lock().unwrap();
        use candle_core::{DType, Device, Tensor, D};
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable — skipping");
            return;
        }
        let engine = process_engine();
        let dev = Device::Cpu;
        let (h, nh, nkv, hd, ffn, max_pos) = (256usize, 4usize, 2usize, 64usize, 512usize, 64usize);
        let pos = 3usize; // three rows already resident; this step appends row 3
        let cap = 8usize;
        let rnd = |i: usize, seed: usize| -> f32 {
            (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0
        };
        let bf = |v: Vec<f32>, shape: Vec<usize>| -> Tensor {
            Tensor::from_vec(v, shape, &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let mk = |n: usize, seed: usize| -> Vec<f32> { (0..n).map(|j| rnd(j, seed)).collect() };

        // Weights (bf16 tensors — the engine captures their storages).
        let op_norm = bf(mk(h, 1), vec![h]);
        let ffn_norm = bf(mk(h, 2), vec![h]);
        let q_w = bf(mk(nh * hd * h, 3), vec![nh * hd, h]);
        let k_w = bf(mk(nkv * hd * h, 4), vec![nkv * hd, h]);
        let v_w = bf(mk(nkv * hd * h, 5), vec![nkv * hd, h]);
        let o_w = bf(mk(h * nh * hd, 6), vec![h, nh * hd]);
        let qn_w = bf(mk(hd, 7), vec![hd]);
        let kn_w = bf(mk(hd, 8), vec![hd]);
        let w1 = bf(mk(ffn * h, 9), vec![ffn, h]);
        let w3 = bf(mk(ffn * h, 10), vec![ffn, h]);
        let w2 = bf(mk(h * ffn, 11), vec![h, ffn]);
        // rope tables [max_pos, hd/2] bf16 (both sides share the same tables).
        let angles: Vec<f32> = (0..max_pos * hd / 2).map(|j| rnd(j, 12) * 3.0).collect();
        let cos = bf(
            angles.iter().map(|a| a.cos()).collect(),
            vec![max_pos, hd / 2],
        );
        let sin = bf(
            angles.iter().map(|a| a.sin()).collect(),
            vec![max_pos, hd / 2],
        );
        // input + pre-existing plane rows
        let x = bf(mk(h, 13), vec![1, 1, h]);
        let mut kplane_init = vec![0f32; nkv * cap * hd];
        let mut vplane_init = vec![0f32; nkv * cap * hd];
        for kh in 0..nkv {
            for r in 0..pos {
                for j in 0..hd {
                    kplane_init[kh * cap * hd + r * hd + j] = rnd(kh * 131 + r * 17 + j, 14);
                    vplane_init[kh * cap * hd + r * hd + j] = rnd(kh * 131 + r * 17 + j, 15);
                }
            }
        }
        let k_plane_ref = bf(kplane_init.clone(), vec![1, nkv, cap, hd]);
        let v_plane_ref = bf(vplane_init.clone(), vec![1, nkv, cap, hd]);
        let k_plane_eng = bf(kplane_init, vec![1, nkv, cap, hd]);
        let v_plane_eng = bf(vplane_init, vec![1, nkv, cap, hd]);

        let bits_of = |t: &Tensor| -> Vec<u16> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<bf16>()
                .unwrap()
                .iter()
                .map(|b| b.to_bits())
                .collect()
        };

        // ---- Reference: the exact op sequence Attention::forward runs on the live
        // mixed path (candle wrappers + attn_decode_bf16 core + fused MLP block). ----
        let rms = |t: &Tensor, w: &Tensor, eps: f64| -> Tensor {
            let tf = t.to_dtype(DType::F32).unwrap();
            let mean_sq = tf.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
            let rsqrt = (mean_sq + eps).unwrap().sqrt().unwrap().recip().unwrap();
            let normed = tf.broadcast_mul(&rsqrt).unwrap();
            normed
                .broadcast_mul(&w.to_dtype(DType::F32).unwrap())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let linear = |wt: &Tensor, t: &Tensor| -> Tensor {
            crate::model::linear::linear_forward(&candle_nn::Linear::new(wt.clone(), None), t)
                .unwrap()
        };
        let xn = rms(&x, &op_norm, 1e-5);
        let q = linear(&q_w, &xn)
            .reshape((1, 1, nh, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let kk = linear(&k_w, &xn)
            .reshape((1, 1, nkv, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let vv = linear(&v_w, &xn)
            .reshape((1, 1, nkv, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        let q = rms(&q.contiguous().unwrap(), &qn_w, 1e-5);
        let kk = rms(&kk.contiguous().unwrap(), &kn_w, 1e-5);
        let cos_row = cos.narrow(0, pos, 1).unwrap();
        let sin_row = sin.narrow(0, pos, 1).unwrap();
        let q =
            candle_nn::rotary_emb::rope_slow(&q.contiguous().unwrap(), &cos_row, &sin_row).unwrap();
        let kk = candle_nn::rotary_emb::rope_slow(&kk.contiguous().unwrap(), &cos_row, &sin_row)
            .unwrap();
        // append at cursor (slice_set — the append_kv mechanism)
        k_plane_ref.slice_set(&kk, 2, pos).unwrap();
        v_plane_ref.slice_set(&vv, 2, pos).unwrap();
        // attention core over the planes
        let q_bits = bits_of(&q);
        let kp_ref = bits_of(&k_plane_ref);
        let vp_ref = bits_of(&v_plane_ref);
        let mut y_bits = vec![0u16; nh * hd];
        unsafe {
            crate::flashkern::decode::attn_decode_bf16(
                &q_bits,
                kp_ref.as_ptr(),
                vp_ref.as_ptr(),
                cap * hd,
                pos + 1,
                nh,
                nkv,
                hd,
                &mut y_bits,
            );
        }
        let y = Tensor::from_vec(
            y_bits
                .iter()
                .map(|&b| bf16::from_bits(b))
                .collect::<Vec<_>>(),
            (1, nh, 1, hd),
            &dev,
        )
        .unwrap();
        let y = y.transpose(1, 2).unwrap().reshape((1, 1, nh * hd)).unwrap();
        let attn_out = linear(&o_w, &y);
        let mid = (attn_out + &x).unwrap();
        // MLP block (the engine's MLP is already parity-pinned to this)
        let mid_bits = bits_of(&mid);
        let mlpw = crate::flashkern::decode::FusedMlpWeights {
            norm_w: &bits_of(&ffn_norm),
            w1: &bits_of(&w1),
            w3: &bits_of(&w3),
            w2: &bits_of(&w2),
            eps: 1e-5,
        };
        let lanes = 8usize;
        let mut out_ref = vec![0u16; h];
        crate::flashkern::decode::fused_mlp_reference(&mid_bits, &mlpw, &mut out_ref, lanes);

        // ---- Engine: install the table, run the layer, compare bits. ----
        use crate::flashkern::decode::PtrLen;
        let cap_ptr = |t: &Tensor| PtrLen::bf16(t).unwrap().addr() as *const u16;
        let descs = [LayerDesc {
            kind: 1,
            op_eps: 1e-5,
            ffn_eps: 1e-5,
            op_norm_w: cap_ptr(&op_norm),
            ffn_norm_w: cap_ptr(&ffn_norm),
            w1: cap_ptr(&w1),
            w3: cap_ptr(&w3),
            w2: cap_ptr(&w2),
            n_head: nh as u32,
            n_kv: nkv as u32,
            hd: hd as u32,
            qk_eps: 1e-5,
            q_w: cap_ptr(&q_w),
            k_w: cap_ptr(&k_w),
            v_w: cap_ptr(&v_w),
            o_w: cap_ptr(&o_w),
            qn_w: cap_ptr(&qn_w),
            kn_w: cap_ptr(&kn_w),
            ..LayerDesc::attn_placeholder()
        }];
        let ctx_id = engine
            .ctx_build(&descs, h, ffn, max_pos)
            .expect("ctx build failed");
        let x_bits = bits_of(&x);
        let kp_eng = PtrLen::bf16(&k_plane_eng).unwrap().addr() as *mut u16;
        let vp_eng = PtrLen::bf16(&v_plane_eng).unwrap().addr() as *mut u16;
        let mut out_got = vec![0u16; h];
        assert!(
            unsafe {
                engine.attn_layer(
                    ctx_id,
                    0,
                    &x_bits,
                    kp_eng,
                    nkv * cap * hd,
                    vp_eng,
                    nkv * cap * hd,
                    cap * hd,
                    pos,
                    cap_ptr(&cos),
                    cap_ptr(&sin),
                    max_pos * hd / 2,
                    &mut out_got,
                    lanes,
                )
            },
            "engine refused attn_layer"
        );
        assert_eq!(out_got, out_ref, "layer output");
        assert_eq!(bits_of(&k_plane_eng), kp_ref, "K plane after append");
        assert_eq!(bits_of(&v_plane_eng), vp_ref, "V plane after append");
        engine.ctx_clear(ctx_id);
    }

    #[test]
    fn attention_scratch_uses_maximum_geometry_across_every_layer() {
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable - skipping");
            return;
        }
        let engine = NativeEngine::new(4).expect("native engine init");
        let (h, ffn, max_ctx) = (8usize, 8usize, 8usize);
        let norm = vec![0x3f80u16; h];
        let matrix = vec![0x3d00u16; h * h * 2];
        let square = vec![0x3c80u16; h * h];
        let desc = |heads: u32| LayerDesc {
            kind: 1,
            op_eps: 1e-5,
            ffn_eps: 1e-5,
            op_norm_w: norm.as_ptr(),
            ffn_norm_w: norm.as_ptr(),
            w1: square.as_ptr(),
            w3: square.as_ptr(),
            w2: square.as_ptr(),
            n_head: heads,
            n_kv: 1,
            hd: 4,
            qk_eps: 1e-5,
            q_w: matrix.as_ptr(),
            k_w: matrix.as_ptr(),
            v_w: matrix.as_ptr(),
            o_w: square.as_ptr(),
            qn_w: norm.as_ptr(),
            kn_w: norm.as_ptr(),
            ..LayerDesc::attn_placeholder()
        };
        // The larger layer deliberately precedes the smaller one. The old builder
        // sized shared planes from the final descriptor and overran them on layer 0.
        let descriptors = [desc(2), desc(1)];
        let id = engine
            .ctx_build(&descriptors, h, ffn, max_ctx)
            .expect("mixed attention context");
        let snapshot = engine.snapshot();
        assert!(snapshot.attention_qkv_capacity as usize >= 16);
        assert!(snapshot.attention_y_capacity as usize >= 8);
        assert!(snapshot.attention_score_capacity as usize >= 2 * max_ctx);

        let x = vec![0x3f00u16; h];
        let rope_cos = vec![0x3f80u16; max_ctx * 2];
        let rope_sin = vec![0u16; max_ctx * 2];
        for (layer, heads) in [2usize, 1].into_iter().enumerate() {
            let head_stride = max_ctx * 4;
            let mut keys = vec![0u16; head_stride];
            let mut values = vec![0u16; head_stride];
            let mut out = vec![0u16; h];
            assert!(unsafe {
                engine.attn_layer(
                    id,
                    layer,
                    &x,
                    keys.as_mut_ptr(),
                    keys.len(),
                    values.as_mut_ptr(),
                    values.len(),
                    head_stride,
                    0,
                    rope_cos.as_ptr(),
                    rope_sin.as_ptr(),
                    rope_cos.len(),
                    &mut out,
                    heads,
                )
            });
        }
        engine.ctx_clear(id);
    }

    #[test]
    fn native_engine_conv_layer_bit_parity() {
        let _ctx = CTX_TEST_LOCK.lock().unwrap();
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable — skipping");
            return;
        }
        let engine = process_engine();
        let rnd = |i: usize, seed: usize| -> u16 {
            bf16::from_f32(
                (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0,
            )
            .to_bits()
        };
        for &(h, i, k) in &[(256usize, 512usize, 3usize), (1024, 2048, 3)] {
            // Synthetic layer weights, held alive for the table's lifetime.
            let op_norm: Vec<u16> = (0..h).map(|j| rnd(j, 1)).collect();
            let ffn_norm: Vec<u16> = (0..h).map(|j| rnd(j, 2)).collect();
            let in_w: Vec<u16> = (0..3 * h * h).map(|j| rnd(j, 3)).collect();
            let conv_w: Vec<u16> = (0..h * k).map(|j| rnd(j, 4)).collect();
            let out_w: Vec<u16> = (0..h * h).map(|j| rnd(j, 5)).collect();
            let w1: Vec<u16> = (0..i * h).map(|j| rnd(j, 6)).collect();
            let w3: Vec<u16> = (0..i * h).map(|j| rnd(j, 7)).collect();
            let w2: Vec<u16> = (0..h * i).map(|j| rnd(j, 8)).collect();
            let x: Vec<u16> = (0..h).map(|j| rnd(j, 9)).collect();
            let state: Vec<u16> = (0..h * (k - 1)).map(|j| rnd(j, 10)).collect();

            // Table with an attention placeholder at 0 so the index path is exercised.
            let descs = [
                LayerDesc::attn_placeholder(),
                LayerDesc {
                    kind: 0,
                    k: k as u32,
                    op_eps: 1e-5,
                    ffn_eps: 1e-5,
                    op_norm_w: op_norm.as_ptr(),
                    ffn_norm_w: ffn_norm.as_ptr(),
                    in_w: in_w.as_ptr(),
                    conv_w: conv_w.as_ptr(),
                    out_w: out_w.as_ptr(),
                    w1: w1.as_ptr(),
                    w3: w3.as_ptr(),
                    w2: w2.as_ptr(),
                    ..LayerDesc::attn_placeholder()
                },
            ];
            let ctx_id = engine
                .ctx_build(&descs, h, i, 64)
                .expect("ctx build failed");

            for lanes in [1usize, 3, 8] {
                // Composed reference: the two fused blocks the layer runs today.
                let scw = crate::flashkern::decode::FusedShortConvWeights {
                    norm_w: &op_norm,
                    in_w: &in_w,
                    conv_w: &conv_w,
                    out_w: &out_w,
                    eps: 1e-5,
                    k,
                };
                let mlpw = crate::flashkern::decode::FusedMlpWeights {
                    norm_w: &ffn_norm,
                    w1: &w1,
                    w3: &w3,
                    w2: &w2,
                    eps: 1e-5,
                };
                let mut state_ref = vec![0u16; h * (k - 1)];
                let mut mid = vec![0u16; h];
                crate::flashkern::decode::fused_shortconv_reference(
                    &x,
                    &scw,
                    &state,
                    &mut state_ref,
                    &mut mid,
                    lanes,
                );
                let mut out_ref = vec![0u16; h];
                crate::flashkern::decode::fused_mlp_reference(&mid, &mlpw, &mut out_ref, lanes);

                let mut state_got = vec![0u16; h * (k - 1)];
                let mut out_got = vec![0u16; h];
                assert!(
                    engine.conv_layer(ctx_id, 1, &x, &state, &mut state_got, &mut out_got, lanes,),
                    "engine refused conv_layer"
                );
                assert_eq!(state_got, state_ref, "state H={h} I={i} lanes={lanes}");
                assert_eq!(out_got, out_ref, "out H={h} I={i} lanes={lanes}");
            }
            engine.ctx_clear(ctx_id);
        }
    }

    #[test]
    fn native_engine_keeps_multiple_model_plans_resident() {
        // Different geometries force every pass to resolve its own plan. Alternating
        // them catches both the old single-slot refusal and accidental reuse of the
        // most recently selected descriptor table.
        let _ctx = CTX_TEST_LOCK.lock().unwrap();
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable — skipping");
            return;
        }
        let engine = process_engine();
        let (h1, i1, k1) = (64usize, 96usize, 3usize);
        let op1 = vec![0x3f80u16; h1];
        let fn1 = vec![0x3f80u16; h1];
        let in1 = vec![0u16; 3 * h1 * h1];
        let conv1 = vec![0u16; h1 * k1];
        let out1w = vec![0u16; h1 * h1];
        let w11 = vec![0u16; i1 * h1];
        let w31 = vec![0u16; i1 * h1];
        let w21 = vec![0u16; h1 * i1];
        let desc1 = [LayerDesc {
            kind: 0,
            k: k1 as u32,
            op_eps: 1e-5,
            ffn_eps: 1e-5,
            op_norm_w: op1.as_ptr(),
            ffn_norm_w: fn1.as_ptr(),
            in_w: in1.as_ptr(),
            conv_w: conv1.as_ptr(),
            out_w: out1w.as_ptr(),
            w1: w11.as_ptr(),
            w3: w31.as_ptr(),
            w2: w21.as_ptr(),
            ..LayerDesc::attn_placeholder()
        }];
        let first = engine.ctx_build(&desc1, h1, i1, 64).expect("first build");

        let (h2, i2, k2) = (96usize, 128usize, 2usize);
        let op2 = vec![0x3f80u16; h2];
        let fn2 = vec![0x3f80u16; h2];
        let in2 = vec![0u16; 3 * h2 * h2];
        let conv2 = vec![0u16; h2 * k2];
        let out2w = vec![0u16; h2 * h2];
        let w12 = vec![0u16; i2 * h2];
        let w32 = vec![0u16; i2 * h2];
        let w22 = vec![0u16; h2 * i2];
        let desc2 = [LayerDesc {
            kind: 0,
            k: k2 as u32,
            op_eps: 1e-5,
            ffn_eps: 1e-5,
            op_norm_w: op2.as_ptr(),
            ffn_norm_w: fn2.as_ptr(),
            in_w: in2.as_ptr(),
            conv_w: conv2.as_ptr(),
            out_w: out2w.as_ptr(),
            w1: w12.as_ptr(),
            w3: w32.as_ptr(),
            w2: w22.as_ptr(),
            ..LayerDesc::attn_placeholder()
        }];
        let second = engine.ctx_build(&desc2, h2, i2, 96).expect("second build");
        assert_ne!(first, second, "resident plan identities must be unique");

        const VOCAB: usize = 4;
        let embed1 = vec![0u16; VOCAB * h1];
        let norm1 = vec![0x3f80u16; h1];
        let embed2 = vec![0u16; VOCAB * h2];
        let norm2 = vec![0x3f80u16; h2];
        assert!(unsafe {
            engine.set_heads(
                first,
                embed1.as_ptr(),
                embed1.len(),
                VOCAB,
                std::ptr::null(),
                0,
                0,
                norm1.as_ptr(),
                norm1.len(),
                1e-5,
            )
        });
        assert!(unsafe {
            engine.set_heads(
                second,
                embed2.as_ptr(),
                embed2.len(),
                VOCAB,
                std::ptr::null(),
                0,
                0,
                norm2.as_ptr(),
                norm2.len(),
                1e-5,
            )
        });

        let x1 = vec![0u16; h1];
        let state1 = vec![0u16; h1 * (k1 - 1)];
        let mut next1 = vec![0u16; state1.len()];
        let mut y1 = vec![0u16; h1];
        let x2 = vec![0u16; h2];
        let state2 = vec![0u16; h2 * (k2 - 1)];
        let mut next2 = vec![0u16; state2.len()];
        let mut y2 = vec![0u16; h2];
        assert!(engine.conv_layer(first, 0, &x1, &state1, &mut next1, &mut y1, 1));
        assert!(engine.conv_layer(second, 0, &x2, &state2, &mut next2, &mut y2, 1));
        assert!(engine.conv_layer(first, 0, &x1, &state1, &mut next1, &mut y1, 1));
        assert!(
            !engine.conv_layer(first, 0, &x2, &state2, &mut next2, &mut y2, 1),
            "ticket identity must select the first plan's geometry"
        );

        engine.ctx_clear(u64::MAX);
        let mut token_state = state2.clone();
        let states = [LayerState {
            conv_state: token_state.as_mut_ptr(),
            conv_len: token_state.len(),
            ..LayerState::none()
        }];
        let mut hidden = vec![0u16; h2];
        let config = SampleConfig::new(None, None);
        let mut sampling = PrngState::from_seed(9).expect("seed");
        let mut token = u32::MAX;
        let mut logits = vec![f32::from_bits(0x7fc0_0001); VOCAB + 2];
        assert!(unsafe {
            engine.token_pass(
                second,
                &[0],
                0,
                &states,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                &mut hidden,
                Some(&mut logits),
                Some(&config),
                Some(&mut sampling),
                Some(&mut token),
                1,
            )
        });
        assert_eq!(token, 0, "flat tied logits must choose the first argmax");
        assert_eq!(logits[..VOCAB], [0.0; VOCAB]);
        assert_eq!(
            logits[VOCAB..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0x7fc0_0001; 2],
            "the final logit destination must not write beyond the vocabulary"
        );

        engine.ctx_clear(first);
        assert!(
            !engine.conv_layer(first, 0, &x1, &state1, &mut next1, &mut y1, 1),
            "cleared plan must refuse passes"
        );
        assert!(
            engine.conv_layer(second, 0, &x2, &state2, &mut next2, &mut y2, 1),
            "clearing one plan must leave the other resident"
        );
        engine.ctx_clear(second);
    }

    #[test]
    fn multirow_prefill_matches_sequential_hidden_kv_and_shortconv_carry() {
        use half::bf16;

        fn weights(count: usize, seed: usize, scale: f32) -> Vec<u16> {
            (0..count)
                .map(|index| {
                    let value = ((index.wrapping_mul(37).wrapping_add(seed * 17)) % 29) as i32 - 14;
                    bf16::from_f32(value as f32 * scale).to_bits()
                })
                .collect()
        }

        fn norms(count: usize, seed: usize) -> Vec<u16> {
            (0..count)
                .map(|index| {
                    let value = ((index + seed) % 7) as f32 - 3.0;
                    bf16::from_f32(1.0 + value * 0.015625).to_bits()
                })
                .collect()
        }

        struct Memory {
            keys: Vec<u16>,
            values: Vec<u16>,
            carry: Vec<u16>,
            stride: usize,
        }

        impl Memory {
            fn new(hidden: usize, kernel: usize, stride: usize) -> Self {
                Self {
                    keys: vec![0; stride],
                    values: vec![0; stride],
                    carry: vec![0; hidden * (kernel - 1)],
                    stride,
                }
            }

            fn views(&mut self) -> [LayerState; 2] {
                [
                    LayerState {
                        k_plane: self.keys.as_mut_ptr(),
                        v_plane: self.values.as_mut_ptr(),
                        head_stride: self.stride,
                        k_len: self.keys.len(),
                        v_len: self.values.len(),
                        ..LayerState::none()
                    },
                    LayerState {
                        conv_state: self.carry.as_mut_ptr(),
                        conv_len: self.carry.len(),
                        ..LayerState::none()
                    },
                ]
            }
        }

        let _guard = CTX_TEST_LOCK.lock().unwrap();
        let engine = NativeEngine::new(4).expect("native engine init");
        let (h, ffn, max_ctx, kernel) = (32usize, 48usize, 16usize, 3usize);
        let (nh, nkv, hd, vocab) = (4usize, 1usize, 8usize, 8usize);
        let qrows = nh * hd;
        let kvrows = nkv * hd;

        let op0 = norms(h, 1);
        let fn0 = norms(h, 2);
        let q = weights(qrows * h, 3, 0.03125);
        let k = weights(kvrows * h, 4, 0.03125);
        let v = weights(kvrows * h, 5, 0.03125);
        let o = weights(h * qrows, 6, 0.03125);
        let qn = norms(hd, 3);
        let kn = norms(hd, 4);
        let aw1 = weights(ffn * h, 7, 0.0234375);
        let aw3 = weights(ffn * h, 8, 0.0234375);
        let aw2 = weights(h * ffn, 9, 0.0234375);

        let op1 = norms(h, 5);
        let fn1 = norms(h, 6);
        let input = weights(3 * h * h, 10, 0.0234375);
        let conv = weights(h * kernel, 11, 0.0625);
        let output = weights(h * h, 12, 0.03125);
        let cw1 = weights(ffn * h, 13, 0.0234375);
        let cw3 = weights(ffn * h, 14, 0.0234375);
        let cw2 = weights(h * ffn, 15, 0.0234375);

        let descs = [
            LayerDesc {
                kind: 1,
                op_eps: 1e-5,
                ffn_eps: 1e-5,
                op_norm_w: op0.as_ptr(),
                ffn_norm_w: fn0.as_ptr(),
                w1: aw1.as_ptr(),
                w3: aw3.as_ptr(),
                w2: aw2.as_ptr(),
                n_head: nh as u32,
                n_kv: nkv as u32,
                hd: hd as u32,
                qk_eps: 1e-5,
                q_w: q.as_ptr(),
                k_w: k.as_ptr(),
                v_w: v.as_ptr(),
                o_w: o.as_ptr(),
                qn_w: qn.as_ptr(),
                kn_w: kn.as_ptr(),
                ..LayerDesc::attn_placeholder()
            },
            LayerDesc {
                kind: 0,
                k: kernel as u32,
                op_eps: 1e-5,
                ffn_eps: 1e-5,
                op_norm_w: op1.as_ptr(),
                ffn_norm_w: fn1.as_ptr(),
                in_w: input.as_ptr(),
                conv_w: conv.as_ptr(),
                out_w: output.as_ptr(),
                w1: cw1.as_ptr(),
                w3: cw3.as_ptr(),
                w2: cw2.as_ptr(),
                ..LayerDesc::attn_placeholder()
            },
        ];
        let id = engine
            .ctx_build(&descs, h, ffn, max_ctx)
            .expect("mixed plan build");
        let embed = weights(vocab * h, 16, 0.078125);
        let final_norm = norms(h, 7);
        assert!(unsafe {
            engine.set_heads(
                id,
                embed.as_ptr(),
                embed.len(),
                vocab,
                std::ptr::null(),
                0,
                0,
                final_norm.as_ptr(),
                final_norm.len(),
                1e-5,
            )
        });

        let mut workspace = std::ptr::null_mut();
        assert_eq!(
            unsafe { lfm_engine_prefill_workspace_create(engine.ptr, id, &mut workspace) },
            0
        );
        assert!(!workspace.is_null());

        let cosine = (0..max_ctx * hd / 2)
            .map(|index| bf16::from_f32(0.875 + (index % 3) as f32 * 0.03125).to_bits())
            .collect::<Vec<_>>();
        let sine = (0..max_ctx * hd / 2)
            .map(|index| bf16::from_f32((index % 5) as f32 * 0.015625).to_bits())
            .collect::<Vec<_>>();
        let ids = [1u32, 3, 2, 5, 4, 6, 0];
        let stride = max_ctx * hd;

        let mut sequential = Memory::new(h, kernel, stride);
        let sequential_states = sequential.views();
        let mut hidden = vec![0u16; h];
        let mut hidden_steps = Vec::new();
        let mut state_steps = Vec::new();
        let sampler = SampleConfig::new(Some(0.8), Some(5));
        let mut sequential_prng = PrngState::from_seed(0x5eed).expect("sequential seed");
        let mut sequential_token = u32::MAX;
        for (position, token) in ids.iter().enumerate() {
            let sample = position + 1 == ids.len();
            assert_eq!(
                unsafe {
                    lfm_engine_token_pass(
                        engine.ptr,
                        id,
                        token,
                        1,
                        0,
                        sequential_states.as_ptr(),
                        sequential_states.len(),
                        position,
                        cosine.as_ptr(),
                        sine.as_ptr(),
                        cosine.len(),
                        hidden.as_mut_ptr(),
                        hidden.len(),
                        std::ptr::null_mut(),
                        0,
                        if sample { &sampler } else { std::ptr::null() },
                        if sample {
                            &mut sequential_prng
                        } else {
                            std::ptr::null_mut()
                        },
                        if sample {
                            &mut sequential_token
                        } else {
                            std::ptr::null_mut()
                        },
                        4,
                        std::ptr::null(),
                    )
                },
                0
            );
            hidden_steps.push(hidden.clone());
            state_steps.push((
                sequential.keys.clone(),
                sequential.values.clone(),
                sequential.carry.clone(),
            ));
        }

        for count in 1..=4 {
            let mut batched = Memory::new(h, kernel, stride);
            let batched_states = batched.views();
            const CANARY: u16 = 0xdead;
            let mut got = vec![CANARY; h + 5];
            assert_eq!(
                unsafe {
                    lfm_engine_prefill(
                        engine.ptr,
                        id,
                        workspace,
                        ids.as_ptr(),
                        std::ptr::null(),
                        count,
                        0,
                        batched_states.as_ptr(),
                        batched_states.len(),
                        0,
                        cosine.as_ptr(),
                        sine.as_ptr(),
                        cosine.len(),
                        got.as_mut_ptr(),
                        h,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        4,
                    )
                },
                0
            );
            assert_eq!(
                &got[..h],
                hidden_steps[count - 1].as_slice(),
                "text hidden, M={count}"
            );
            assert!(got[h..].iter().all(|value| *value == CANARY));
            assert_eq!(batched.keys, state_steps[count - 1].0, "text K, M={count}");
            assert_eq!(
                batched.values,
                state_steps[count - 1].1,
                "text V, M={count}"
            );
            assert_eq!(
                batched.carry,
                state_steps[count - 1].2,
                "text carry, M={count}"
            );
        }

        let mut chunked = Memory::new(h, kernel, stride);
        let chunked_states = chunked.views();
        let mut chunked_hidden = vec![0u16; h];
        let mut chunked_prng = PrngState::from_seed(0x5eed).expect("chunked seed");
        let mut chunked_token = u32::MAX;
        let submissions = engine.snapshot().pass_submissions;
        for (position, chunk) in [(0usize, &ids[..4]), (4, &ids[4..])] {
            let sample = position + chunk.len() == ids.len();
            assert_eq!(
                unsafe {
                    lfm_engine_prefill(
                        engine.ptr,
                        id,
                        workspace,
                        chunk.as_ptr(),
                        std::ptr::null(),
                        chunk.len(),
                        0,
                        chunked_states.as_ptr(),
                        chunked_states.len(),
                        position,
                        cosine.as_ptr(),
                        sine.as_ptr(),
                        cosine.len(),
                        chunked_hidden.as_mut_ptr(),
                        chunked_hidden.len(),
                        if sample { &sampler } else { std::ptr::null() },
                        if sample {
                            &mut chunked_prng
                        } else {
                            std::ptr::null_mut()
                        },
                        if sample {
                            &mut chunked_token
                        } else {
                            std::ptr::null_mut()
                        },
                        4,
                    )
                },
                0
            );
        }
        assert_eq!(engine.snapshot().pass_submissions, submissions + 2);
        assert_eq!(
            chunked_hidden,
            *hidden_steps.last().unwrap(),
            "text 4+3 hidden"
        );
        assert_eq!(chunked.keys, state_steps.last().unwrap().0, "text 4+3 K");
        assert_eq!(chunked.values, state_steps.last().unwrap().1, "text 4+3 V");
        assert_eq!(
            chunked.carry,
            state_steps.last().unwrap().2,
            "text 4+3 carry"
        );
        assert_eq!(chunked_token, sequential_token, "text 4+3 sampled token");

        let provided = weights(ids.len() * h, 23, 0.09375);
        let mut sequential = Memory::new(h, kernel, stride);
        let sequential_states = sequential.views();
        let mut want = vec![0u16; h];
        for position in 0..ids.len() {
            assert_eq!(
                unsafe {
                    lfm_engine_token_pass(
                        engine.ptr,
                        id,
                        ids.as_ptr(),
                        1,
                        2,
                        sequential_states.as_ptr(),
                        sequential_states.len(),
                        position,
                        cosine.as_ptr(),
                        sine.as_ptr(),
                        cosine.len(),
                        want.as_mut_ptr(),
                        want.len(),
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        4,
                        provided.as_ptr().add(position * h),
                    )
                },
                0
            );
        }
        let mut batched = Memory::new(h, kernel, stride);
        let batched_states = batched.views();
        const CANARY: u16 = 0xdead;
        let mut got = vec![CANARY; h + 5];
        let submissions = engine.snapshot().pass_submissions;
        for (position, range) in [(0usize, 0..4), (4, 4..ids.len())] {
            assert_eq!(
                unsafe {
                    lfm_engine_prefill(
                        engine.ptr,
                        id,
                        workspace,
                        std::ptr::null(),
                        provided.as_ptr().add(range.start * h),
                        range.len(),
                        2,
                        batched_states.as_ptr(),
                        batched_states.len(),
                        position,
                        cosine.as_ptr(),
                        sine.as_ptr(),
                        cosine.len(),
                        got.as_mut_ptr(),
                        h,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        4,
                    )
                },
                0
            );
        }
        assert_eq!(
            engine.snapshot().pass_submissions,
            submissions + 2,
            "seven provided rows must use exactly an M=4 ticket plus its tail"
        );
        assert_eq!(&got[..h], want.as_slice(), "provided-row hidden");
        assert!(got[h..].iter().all(|value| *value == CANARY));
        assert_eq!(batched.keys, sequential.keys, "provided-row K");
        assert_eq!(batched.values, sequential.values, "provided-row V");
        assert_eq!(batched.carry, sequential.carry, "provided-row carry");

        unsafe { lfm_engine_prefill_workspace_destroy(workspace) };
        engine.ctx_clear(id);
    }

    #[test]
    fn native_engine_mlp_bit_parity() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused mlp kernel unavailable — skipping");
            return;
        }
        let engine = NativeEngine::new(8).expect(
            "native engine init failed on a target with fused kernels; check kcoro link/init",
        );
        let rnd = |i: usize, seed: usize| -> u16 {
            bf16::from_f32(
                (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0,
            )
            .to_bits()
        };
        for &(h, i) in &[(64usize, 96usize), (256, 512), (1024, 2048)] {
            let x: Vec<u16> = (0..h).map(|j| rnd(j, 1)).collect();
            let w = crate::flashkern::decode::FusedMlpWeights {
                norm_w: &(0..h).map(|j| rnd(j, 2)).collect::<Vec<_>>(),
                w1: &(0..i * h).map(|j| rnd(j, 3)).collect::<Vec<_>>(),
                w3: &(0..i * h).map(|j| rnd(j, 4)).collect::<Vec<_>>(),
                w2: &(0..h * i).map(|j| rnd(j, 5)).collect::<Vec<_>>(),
                eps: 1e-5,
            };
            for lanes in [1usize, 3, 8] {
                let mut want = vec![0u16; h];
                crate::flashkern::decode::fused_mlp_reference(&x, &w, &mut want, lanes);
                let mut got = vec![0u16; h];
                assert!(engine.fused_mlp(&x, &w, &mut got, lanes));
                assert_eq!(got, want, "H={h} I={i} lanes={lanes}");
            }
        }
    }

    #[test]
    #[ignore = "local performance measurement; run explicitly with --ignored --nocapture"]
    fn native_engine_mlp_benchmark() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused mlp kernel unavailable — skipping");
            return;
        }
        let engine = NativeEngine::new(8).expect(
            "native engine init failed on a target with fused kernels; check kcoro link/init",
        );
        let rnd = |i: usize, seed: usize| -> u16 {
            bf16::from_f32(
                (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0,
            )
            .to_bits()
        };
        // Timing at the real decode shape: resident native engine versus the portable
        // scoped-thread parity path. This is intentionally local-only, not a CI assertion.
        let (h, i) = (1024usize, 4096usize);
        let x: Vec<u16> = (0..h).map(|j| rnd(j, 1)).collect();
        let norm_w: Vec<u16> = (0..h).map(|j| rnd(j, 2)).collect();
        let w1: Vec<u16> = (0..i * h).map(|j| rnd(j, 3)).collect();
        let w3: Vec<u16> = (0..i * h).map(|j| rnd(j, 4)).collect();
        let w2: Vec<u16> = (0..h * i).map(|j| rnd(j, 5)).collect();
        let w = crate::flashkern::decode::FusedMlpWeights {
            norm_w: &norm_w,
            w1: &w1,
            w3: &w3,
            w2: &w2,
            eps: 1e-5,
        };
        let mut out = vec![0u16; h];
        let lanes = 8;
        for _ in 0..20 {
            assert!(engine.fused_mlp(&x, &w, &mut out, lanes));
            crate::flashkern::decode::fused_mlp_reference(&x, &w, &mut out, lanes);
        }

        const SAMPLES: usize = 1_000;
        let mut native = Vec::with_capacity(SAMPLES);
        let mut reference = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let mut measure = |native_path| {
                let start = std::time::Instant::now();
                if native_path {
                    assert!(engine.fused_mlp(&x, &w, &mut out, lanes));
                } else {
                    crate::flashkern::decode::fused_mlp_reference(&x, &w, &mut out, lanes);
                }
                start.elapsed().as_secs_f64() * 1e3
            };
            if sample % 2 == 0 {
                native.push(measure(true));
                reference.push(measure(false));
            } else {
                reference.push(measure(false));
                native.push(measure(true));
            }
        }
        native.sort_by(f64::total_cmp);
        reference.sort_by(f64::total_cmp);
        let percentile =
            |samples: &[f64], percent: usize| samples[(samples.len() * percent).div_ceil(100) - 1];
        eprintln!(
            "native engine fused_mlp p50/p95/p99 {:.3}/{:.3}/{:.3} ms ({:.3}-{:.3}) vs scoped reference {:.3}/{:.3}/{:.3} ms ({:.3}-{:.3}) over {SAMPLES} passes (H=1024 I=4096, lanes=8)",
            percentile(&native, 50), percentile(&native, 95), percentile(&native, 99),
            native[0], native[SAMPLES - 1], percentile(&reference, 50),
            percentile(&reference, 95), percentile(&reference, 99), reference[0], reference[SAMPLES - 1]
        );
    }

    #[test]
    fn engine_protocol_selectors_are_closed_sets() {
        // SAFETY: the private probe calls the exact closed-set predicate used by
        // both native submission and descriptor admission without dispatching an
        // uninitialized typed payload.
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        for kind in [0, 5, 16, u32::MAX] {
            assert_eq!(
                unsafe { lfm_internal_engine_request_kind_valid_for_test(kind) },
                0,
                "request kind {kind}"
            );
        }
        for kind in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
            assert_eq!(
                unsafe { lfm_internal_engine_request_kind_valid_for_test(kind) },
                1,
                "request kind {kind}"
            );
        }

        let mut descriptor = LayerDesc::attn_placeholder();
        descriptor.kind = 2;
        let mut id = 0;
        assert_eq!(
            unsafe { lfm_ctx_build(raw, &descriptor, 1, 1, 1, 1, &mut id) },
            -libc::ENOTSUP
        );
        assert_eq!(id, 0);

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.pass_submissions, 0);
        assert_eq!(snapshot.bridge_dispatches, 0);
        assert_eq!(snapshot.pass_slots_live, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn audio_route_forwarding_table_is_total_and_bounds_checked() {
        const TOKEN: u32 = 0;
        const DEPTH: u32 = 1;
        const MIMI: u32 = 2;
        const TERMINAL: u32 = 3;
        const SUCCESS: u32 = 0;
        const FAILURE: u32 = 1;
        const EOAUDIO: u32 = 2;
        const STALE: u32 = 3;
        for (node, outcome, expected) in [
            (TOKEN, SUCCESS, DEPTH),
            (TOKEN, FAILURE, TERMINAL),
            (TOKEN, EOAUDIO, TERMINAL),
            (TOKEN, STALE, TERMINAL),
            (DEPTH, SUCCESS, MIMI),
            (DEPTH, FAILURE, TERMINAL),
            (DEPTH, EOAUDIO, TERMINAL),
            (DEPTH, STALE, TERMINAL),
            (MIMI, SUCCESS, TERMINAL),
            (MIMI, FAILURE, TERMINAL),
            (MIMI, EOAUDIO, TERMINAL),
            (MIMI, STALE, TERMINAL),
        ] {
            let mut target = u32::MAX;
            assert_eq!(
                unsafe {
                    lfm_internal_engine_audio_route_edge_for_test(node, outcome, &mut target)
                },
                0
            );
            assert_eq!(target, expected);
        }
        let mut target = 0xfeed_beef;
        assert_eq!(
            unsafe { lfm_internal_engine_audio_route_edge_for_test(3, SUCCESS, &mut target) },
            -libc::EINVAL
        );
        assert_eq!(target, 0xfeed_beef);
        assert_eq!(
            unsafe { lfm_internal_engine_audio_route_edge_for_test(TOKEN, 4, &mut target) },
            -libc::EINVAL
        );
        assert_eq!(target, 0xfeed_beef);
        assert_eq!(
            unsafe { lfm_internal_engine_audio_token_class_for_test(0) },
            0
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_token_class_for_test(2047) },
            0
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_token_class_for_test(2048) },
            1
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_token_class_for_test(2049) },
            2
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_token_class_for_test(u32::MAX) },
            2
        );
        const DEADLINE: u32 = 1;
        const INTERACTIVE: u32 = 2;
        const BACKGROUND: u32 = 3;
        assert_eq!(
            unsafe { lfm_internal_engine_audio_route_service_for_test(100, 101, BACKGROUND) },
            BACKGROUND,
            "a route enqueued after the broker snapshot is new, not starved"
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_route_service_for_test(164, 100, BACKGROUND) },
            DEADLINE,
            "64 missed broker enqueue epochs promote genuinely waiting work"
        );
        assert_eq!(
            unsafe { lfm_internal_engine_audio_route_service_for_test(163, 100, INTERACTIVE) },
            INTERACTIVE
        );
    }

    #[test]
    fn monarch_longconv_selector_is_explicitly_unsupported() {
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        let mut descriptor = LayerDesc::attn_placeholder();
        descriptor.kind = 2;
        let mut id = 0u64;
        assert_eq!(
            unsafe { lfm_ctx_build(raw, &descriptor, 1, 4, 4, 8, &mut id) },
            -libc::ENOTSUP
        );
        assert_eq!(id, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn engine_wait_words_occupy_distinct_128_byte_lines() {
        let raw = unsafe { lfm_engine_new(8) };
        assert!(!raw.is_null());
        assert_eq!(
            unsafe { lfm_internal_engine_wait_word_layout_for_test(raw) },
            0
        );
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn eight_lane_gang_requires_both_block_completions() {
        let raw = unsafe { lfm_engine_new(8) };
        assert!(!raw.is_null());
        let mut state = PrngState::from_seed(19).expect("seed");
        let mut value = 0u64;
        assert_eq!(
            unsafe { lfm_engine_prng_fill(raw, &mut state, &mut value, 1) },
            0
        );

        let mut blocks = 0u32;
        let mut completions = 0u64;
        let mut generations = 0u64;
        let mut lease = u64::MAX;
        assert_eq!(
            unsafe {
                lfm_internal_engine_grid_snapshot_for_test(
                    raw,
                    &mut blocks,
                    &mut completions,
                    &mut generations,
                    &mut lease,
                )
            },
            0
        );
        assert_eq!(blocks, 2);
        assert_eq!(completions, 2);
        assert_eq!(generations, 1);
        assert_eq!(lease, 0, "publication must retire the exact gang lease");
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn lane_counts_are_rejected_instead_of_clamped() {
        assert!(NativeEngine::new(0).is_none());
        assert!(NativeEngine::new(17).is_none());
        assert!(NativeEngine::new(usize::MAX).is_none());
        assert!(unsafe { lfm_engine_new(-1) }.is_null());
        assert!(unsafe { lfm_engine_new(17) }.is_null());

        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        let one = [0x3f80u16];
        let mut out = [0xdead];
        for lanes in [0, 17] {
            assert_eq!(
                unsafe {
                    lfm_engine_mlp(
                        raw,
                        one.as_ptr(),
                        one.as_ptr(),
                        one.as_ptr(),
                        one.as_ptr(),
                        one.as_ptr(),
                        out.as_mut_ptr(),
                        1,
                        1,
                        1e-5,
                        lanes,
                    )
                },
                -libc::EINVAL
            );
        }
        assert_eq!(out, [0xdead]);
        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.pass_submissions, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn logical_reduction_width_is_independent_of_physical_lane_count() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused mlp kernel unavailable — skipping");
            return;
        }
        let four = NativeEngine::new(4).expect("four-lane engine");
        let eight = NativeEngine::new(8).expect("eight-lane engine");
        assert_eq!(four.lanes_total(), 4);
        assert_eq!(eight.lanes_total(), 8);

        const H: usize = 64;
        const I: usize = 96;
        const LOGICAL: usize = 8;
        let bits = |index: usize, seed: usize| {
            bf16::from_f32(
                (((index.wrapping_mul(2_654_435_761).wrapping_add(seed)) % 2000) as f32 / 1000.0)
                    - 1.0,
            )
            .to_bits()
        };
        let x = (0..H).map(|index| bits(index, 1)).collect::<Vec<_>>();
        let norm = (0..H).map(|index| bits(index, 2)).collect::<Vec<_>>();
        let w1 = (0..I * H).map(|index| bits(index, 3)).collect::<Vec<_>>();
        let w3 = (0..I * H).map(|index| bits(index, 4)).collect::<Vec<_>>();
        let w2 = (0..H * I).map(|index| bits(index, 5)).collect::<Vec<_>>();
        let weights = crate::flashkern::decode::FusedMlpWeights {
            norm_w: &norm,
            w1: &w1,
            w3: &w3,
            w2: &w2,
            eps: 1e-5,
        };
        let mut want = vec![0; H];
        let mut got4 = vec![0; H];
        let mut got8 = vec![0; H];
        crate::flashkern::decode::fused_mlp_reference(&x, &weights, &mut want, LOGICAL);
        assert!(four.fused_mlp(&x, &weights, &mut got4, LOGICAL));
        assert!(eight.fused_mlp(&x, &weights, &mut got8, LOGICAL));
        assert_eq!(got4, want);
        assert_eq!(got8, want);
    }

    #[test]
    fn raw_engine_rejects_concurrent_request_before_payload_write() {
        let engine = NativeEngine::new(2).expect("native engine init");
        let engine_address = engine.ptr as usize;
        const N: usize = 1024;
        const ROWS: usize = 8;
        let frequency = N / 2 + 1;
        let real = Arc::new(vec![0.25f32; ROWS * frequency]);
        let imag = Arc::new(vec![-0.125f32; ROWS * frequency]);
        let start = Arc::new(Barrier::new(3));
        let calls = (0..2)
            .map(|_| {
                let real = Arc::clone(&real);
                let imag = Arc::clone(&imag);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let mut out = vec![0.0f32; ROWS * N];
                    start.wait();
                    // SAFETY: this deliberately bypasses the Rust pass lock to prove
                    // the native claim protects request storage before payload writes.
                    unsafe {
                        lfm_engine_irfft_dd(
                            engine_address as *mut c_void,
                            real.as_ptr(),
                            real.len(),
                            imag.as_ptr(),
                            imag.len(),
                            out.as_mut_ptr(),
                            out.len(),
                            ROWS,
                            N,
                            1.0 / N as f32,
                            0.0,
                        )
                    }
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        let mut results = calls
            .into_iter()
            .map(|call| call.join().unwrap())
            .collect::<Vec<_>>();
        results.sort_unstable();
        assert_eq!(results, [-libc::EBUSY, 0]);
    }

    #[test]
    fn raw_engine_owns_its_sq_cq_without_rust_progress() {
        // SAFETY: this test deliberately exercises the unpublished C constructor.
        // A successful pass proves native progress has no Rust callback dependency.
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        let mut state = PrngState::from_seed(7).expect("seed");
        let mut value = 0u64;
        let rc = unsafe { lfm_engine_prng_fill(raw, &mut state, &mut value, 1) };
        assert_eq!(rc, 0);

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.pass_submissions, 1);
        assert_eq!(snapshot.pass_completions, 1);
        assert_eq!(snapshot.bridge_dispatches, 1);
        assert_eq!(snapshot.descriptor_acquires, 1);
        assert_eq!(snapshot.descriptor_retains, 1);
        assert_eq!(snapshot.descriptor_releases, 2);
        assert_eq!(snapshot.descriptors_live, 0);
        assert_eq!(snapshot.bridge_capacity, 2);
        assert_eq!(snapshot.pass_slot_capacity, 2);
        assert_eq!(snapshot.pass_slots_live, 0);
        assert_eq!(snapshot.max_pass_slots_live, 1);
        // SAFETY: the accepted bridge ticket completed and both leases are settled.
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn exact_cq_continuation_refills_the_second_slot_without_rust_progress() {
        const PASSES: usize = 64;
        let oracle = NativeEngine::new(2).expect("oracle engine init");
        let mut oracle_state = PrngState::from_seed(0x5eed).expect("oracle seed");
        let mut expected = [0u64; PASSES];
        assert!(oracle.prng_fill(&mut oracle_state, &mut expected));

        // SAFETY: the private test seam borrows these fixed buffers until its
        // terminal expected-value doorbell. No Rust callback advances the chain.
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        let mut state = PrngState::from_seed(0x5eed).expect("chain seed");
        let mut actual = [0u64; PASSES];
        assert_eq!(
            unsafe {
                lfm_internal_engine_prng_continuation_for_test(
                    raw,
                    &mut state,
                    actual.as_mut_ptr(),
                    actual.len(),
                )
            },
            0
        );
        assert_eq!(actual, expected);
        assert_eq!(state.cursor, oracle_state.cursor);
        assert_eq!(state.core, oracle_state.core);
        assert_eq!(state.block, oracle_state.block);

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.bridge_capacity, 2);
        assert_eq!(snapshot.pass_slot_capacity, 2);
        assert_eq!(snapshot.pass_submissions, PASSES as u64);
        assert_eq!(snapshot.pass_completions, PASSES as u64);
        assert_eq!(snapshot.bridge_dispatches, PASSES as u64);
        assert_eq!(snapshot.continuation_submissions, PASSES as u64);
        assert_eq!(snapshot.max_pass_slots_live, 1);
        assert_eq!(snapshot.pass_slots_live, 0);
        assert_eq!(snapshot.pass_claimed, 0);
        assert_eq!(snapshot.descriptors_live, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn exact_cq_handoff_keeps_its_slot_from_a_competing_compatibility_call() {
        const PASSES: usize = 32;
        const CALLBACK: u32 = 2;
        const ARM: u32 = 1;
        const WAIT: u32 = 2;
        const RELEASE: u32 = 3;

        let oracle = NativeEngine::new(2).expect("oracle engine init");
        let mut chain_oracle = PrngState::from_seed(0xabc1).expect("chain oracle");
        let mut chain_expected = [0u64; PASSES];
        assert!(oracle.prng_fill(&mut chain_oracle, &mut chain_expected));
        let mut peer_oracle = PrngState::from_seed(0xabc2).expect("peer oracle");
        let mut peer_expected = 0;
        assert!(oracle.prng_fill(&mut peer_oracle, std::slice::from_mut(&mut peer_expected)));

        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        assert_eq!(
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, ARM) },
            0
        );
        let address = raw as usize;
        let chain = std::thread::spawn(move || {
            let mut state = PrngState::from_seed(0xabc1).expect("chain seed");
            let mut out = [0u64; PASSES];
            let rc = unsafe {
                lfm_internal_engine_prng_continuation_for_test(
                    address as *mut c_void,
                    &mut state,
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            (rc, state, out)
        });
        let callback_wait =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, WAIT) };

        let peer_address = raw as usize;
        let peer = std::thread::spawn(move || {
            let mut state = PrngState::from_seed(0xabc2).expect("peer seed");
            let mut out = 0;
            let rc = unsafe {
                lfm_engine_prng_fill(peer_address as *mut c_void, &mut state, &mut out, 1)
            };
            (rc, state, out)
        });
        let live_wait = unsafe { lfm_internal_engine_wait_pass_slots_for_test(raw, 2) };
        let release =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, RELEASE) };
        let (peer_rc, peer_state, peer_value) = peer.join().expect("peer join");
        let (chain_rc, chain_state, chain_actual) = chain.join().expect("chain join");

        assert_eq!(callback_wait, 0);
        assert_eq!(live_wait, 0);
        assert_eq!(release, 0);
        assert_eq!(peer_rc, 0);
        assert_eq!(peer_value, peer_expected);
        assert_eq!(peer_state.cursor, peer_oracle.cursor);
        assert_eq!(chain_rc, 0);
        assert_eq!(chain_actual, chain_expected);
        assert_eq!(chain_state.cursor, chain_oracle.cursor);

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.max_pass_slots_live, 2);
        assert_eq!(snapshot.pass_slots_live, 0);
        assert_eq!(snapshot.pass_claimed, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn stale_claim_destructor_cannot_release_a_reowned_continuation_slot() {
        const PASSES: usize = 16;
        const CLAIM_RETURN: u32 = 1;
        const CALLBACK: u32 = 2;
        const ARM: u32 = 1;
        const WAIT: u32 = 2;
        const RELEASE: u32 = 3;

        let oracle = NativeEngine::new(2).expect("oracle engine init");
        let mut expected_state = PrngState::from_seed(0xdef1).expect("oracle seed");
        let mut expected = [0u64; PASSES];
        assert!(oracle.prng_fill(&mut expected_state, &mut expected));

        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        assert_eq!(
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CLAIM_RETURN, ARM) },
            0
        );
        let address = raw as usize;
        let old = std::thread::spawn(move || {
            let mut state = PrngState::from_seed(0xdef0).expect("old seed");
            let mut out = 0;
            let rc =
                unsafe { lfm_engine_prng_fill(address as *mut c_void, &mut state, &mut out, 1) };
            (rc, out)
        });
        let old_wait =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CLAIM_RETURN, WAIT) };

        assert_eq!(
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, ARM) },
            0
        );
        let chain_address = raw as usize;
        let chain = std::thread::spawn(move || {
            let mut state = PrngState::from_seed(0xdef1).expect("chain seed");
            let mut out = [0u64; PASSES];
            let rc = unsafe {
                lfm_internal_engine_prng_continuation_for_test(
                    chain_address as *mut c_void,
                    &mut state,
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            (rc, state, out)
        });
        let callback_wait =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, WAIT) };
        let old_release =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CLAIM_RETURN, RELEASE) };
        let (old_rc, _) = old.join().expect("old claim join");

        let mut paused = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        let paused_snapshot = unsafe { lfm_engine_snapshot(raw, &mut paused) };
        let callback_release =
            unsafe { lfm_internal_engine_pause_boundary_for_test(raw, CALLBACK, RELEASE) };
        let (chain_rc, chain_state, actual) = chain.join().expect("chain join");

        assert_eq!(old_wait, 0);
        assert_eq!(callback_wait, 0);
        assert_eq!(old_release, 0);
        assert_eq!(old_rc, 0);
        assert_eq!(paused_snapshot, 0);
        assert_eq!(paused.pass_claimed, 0);
        assert_eq!(paused.pass_slots_live, 1);
        assert_eq!(callback_release, 0);
        assert_eq!(chain_rc, 0);
        assert_eq!(actual, expected);
        assert_eq!(chain_state.cursor, expected_state.cursor);

        let mut settled = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut settled) }, 0);
        assert_eq!(settled.pass_slots_live, 0);
        assert_eq!(settled.pass_claimed, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn stop_during_an_active_pass_drains_exact_cq_before_unmounting_scratch() {
        const N: usize = 1024;
        const ROWS: usize = 8;
        let frequency = N / 2 + 1;
        // SAFETY: the worker owns every borrowed input/output until the blocking
        // pass returns. The main thread only observes atomic engine accounting.
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        assert_eq!(
            unsafe { lfm_internal_engine_arm_lane_pause_for_test(raw) },
            0
        );
        let address = raw as usize;
        let call = std::thread::spawn(move || {
            let real = vec![0.25f32; ROWS * frequency];
            let imag = vec![-0.125f32; ROWS * frequency];
            let mut out = vec![0.0f32; ROWS * N];
            let rc = unsafe {
                lfm_engine_irfft_dd(
                    address as *mut c_void,
                    real.as_ptr(),
                    real.len(),
                    imag.as_ptr(),
                    imag.len(),
                    out.as_mut_ptr(),
                    out.len(),
                    ROWS,
                    N,
                    1.0 / N as f32,
                    0.0,
                )
            };
            (rc, out)
        });

        // This is an expected-value wait, not a polling probe. Lane 0 publishes
        // the pause only after the bridge mounted this ticket's scratch and rang
        // the fixed team; peer lanes may already be consuming it.
        assert_eq!(
            unsafe { lfm_internal_engine_wait_lane_pause_for_test(raw) },
            0
        );
        unsafe { lfm_engine_request_stop(raw) };
        let (rc, out) = call.join().expect("active pass thread");
        assert_eq!(rc, 0, "an accepted pass must drain after stop");
        assert!(out.iter().all(|value| value.is_finite()));

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.pass_completions, 1);
        assert_eq!(snapshot.pass_slots_live, 0);
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn native_engine_bridge_and_fence_soak() {
        let engine = NativeEngine::new(8).expect("native engine init");
        const PASSES: u64 = 10_000;
        const LANES: u64 = 8;
        const FENCES_PER_PASS: u64 = 2;
        let start = std::time::Instant::now();
        let mut state = PrngState::from_seed(11).expect("seed");
        let mut value = [0u64; 1];
        for pass in 0..PASSES {
            assert!(
                engine.prng_fill(&mut state, &mut value),
                "pass {pass} did not complete"
            );
        }
        let stats = engine.snapshot();
        assert_eq!(stats.pass_submissions, PASSES);
        assert_eq!(stats.pass_completions, PASSES);
        assert_eq!(stats.bridge_dispatches, PASSES);
        assert_eq!(stats.dispatch_wakes, PASSES);
        assert_eq!(stats.fence_generations, PASSES * FENCES_PER_PASS);
        assert!(stats.fence_wake_calls > 0);
        assert!(stats.fence_wake_calls <= stats.fence_generations);
        assert!(stats.fence_wakes >= stats.fence_wake_calls);
        assert!(stats.fence_wakes > 0);
        assert!(stats.fence_wakes <= PASSES * FENCES_PER_PASS * (LANES - 1));
        assert_eq!(stats.descriptor_capacity, 8);
        assert_eq!(stats.descriptors_live, 0);
        assert_eq!(stats.descriptor_acquires, PASSES);
        assert_eq!(stats.descriptor_retains, PASSES);
        assert_eq!(stats.descriptor_releases, PASSES * 2);
        assert_eq!(stats.descriptor_callbacks, 0);
        assert_eq!(stats.max_descriptor_generation, PASSES as u32);
        assert_eq!(stats.pass_claimed, 0);
        assert_eq!(stats.bridge_capacity, 2);
        assert_eq!(stats.pass_slot_capacity, 2);
        assert_eq!(stats.pass_slots_live, 0);
        eprintln!(
            "native bridge/fence soak: {PASSES} passes, {} fence syscalls for {} waiters in {:.3}s",
            stats.fence_wake_calls,
            stats.fence_wakes,
            start.elapsed().as_secs_f64(),
        );
    }
}
