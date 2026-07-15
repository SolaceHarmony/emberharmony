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
    pass_claimed: u32,
}

extern "C" {
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
    fn lfm_bf16_bias_add_f32(values: *mut f32, bias: *const u16, count: usize);
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
}

pub(crate) fn depthwise_stream_available() -> bool {
    // SAFETY: capability query accepts no pointers and mutates no state.
    unsafe { lfm_depthwise_stream_bf16_available() != 0 }
}

/// Handle to the persistent native engine. One per process is the intended shape
/// (decode is sequential). The C side is a SINGLE-SLOT machine — one Pass, one
/// scratch arena, one request word — so the wrapper serializes the entire native
/// call under `pass_lock`; that lock makes the `Sync` below true. The raw C ABI
/// independently claims the slot before touching shared payload state.
pub struct NativeEngine {
    ptr: *mut c_void,
    pass_lock: Mutex<()>,
}

// SAFETY: Send — the handle is an opaque pointer to a C-heap object with no thread
// affinity. Sync — provided by `pass_lock` above serializing every call into the
// SINGLE-SLOT C engine (one Pass, one scratch arena, one request word). The C side's
// atomic claim rejects unsafe concurrent callers before request setup, but it does not
// make two safe Rust borrows of the same output buffer legal or define queue ordering.
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    pub fn new(workers: usize) -> Option<Self> {
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
        if !crate::bf16_gemm::bf16_gemm_nt_available() {
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
        // The lock that makes `Sync` true: the C engine is single-slot, so the whole
        // native call — request setup through completion — is serialized here.
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
                weights.as_ptr(),
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
            lfm_bf16_bias_add_f32(bias.as_mut_ptr(), bias_bits.as_ptr(), bias.len());
            lfm_bf16_rope_neox(
                rope.as_mut_ptr(),
                cosine.as_ptr(),
                sine.as_ptr(),
                rope.len(),
            );
        }
        assert_eq!(bias.map(f32::to_bits), [0x3fc0_0000, 0x3f80_0000]);
        assert_eq!(rope, [0x3f80, 0xbf80, 0x4040, 0x4040]);
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
        if !crate::bf16_gemm::bf16_gemm_available() {
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
                None,
                Some(&config),
                Some(&mut sampling),
                Some(&mut token),
                1,
            )
        });
        assert_eq!(token, 0, "flat tied logits must choose the first argmax");

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
        // SAFETY: the accepted bridge ticket completed and both leases are settled.
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
        eprintln!(
            "native bridge/fence soak: {PASSES} passes, {} fence syscalls for {} waiters in {:.3}s",
            stats.fence_wake_calls,
            stats.fence_wakes,
            start.elapsed().as_secs_f64(),
        );
    }
}
