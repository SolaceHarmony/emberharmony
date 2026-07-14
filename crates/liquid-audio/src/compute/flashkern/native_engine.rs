//! The Rust rim of the resident native decode engine (native/src/engine/flashkern_engine.cpp).
//!
//! Everything below the ABI line is C++: the persistent fixed-lane team, the block
//! schedules, the stage kernels. Rust's per-pass surface is one blocking call while
//! borrowed tensor pointers remain in use. Internally, the callback-driven coordinator
//! is the sole SQ producer and its ingress thread is the sole CQ consumer. No Rust runs
//! numerical stages and no application event loop makes progress for the kernel.

use super::coordinator::{self, Coordinator};
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
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_free(e: *mut c_void);
    fn lfm_engine_bridge(e: *mut c_void) -> *mut c_void;
    fn lfm_engine_set_submitter(
        e: *mut c_void,
        submitter: unsafe extern "C" fn(
            *mut c_void,
            *const kcoro::Submission,
            *mut kcoro::Completion,
        ) -> i32,
        context: *mut c_void,
    ) -> i32;
    fn lfm_engine_clear_submitter(e: *mut c_void, context: *mut c_void) -> i32;
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
    fn lfm_engine_call(
        e: *mut c_void,
        f: unsafe extern "C" fn(*mut c_void, u32, u32),
        ctx: *mut c_void,
    ) -> i32;
    fn lfm_engine_lanes(e: *mut c_void) -> u32;
    #[cfg(test)]
    fn lfm_engine_snapshot(e: *mut c_void, out: *mut EngineSnapshot) -> i32;
    #[cfg(test)]
    fn lfm_lane_fence(e: *mut c_void, lane: u32);
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
        lanes: usize,
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
}

/// Handle to the persistent native engine. One per process is the intended shape
/// (decode is sequential). The C side is a SINGLE-SLOT machine — one Pass, one
/// scratch arena, one request word — so the wrapper serializes the entire native
/// call under `pass_lock`; that lock makes the `Sync` below true. The raw C ABI
/// independently claims the slot before touching shared payload state.
pub struct NativeEngine {
    ptr: *mut c_void,
    coordinator: Coordinator,
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
        // SAFETY: `p` names the just-created engine and therefore its live bridge.
        let bridge = unsafe { lfm_engine_bridge(p) };
        let mut coordinator = match Coordinator::new(bridge, 8) {
            Ok(coordinator) => coordinator,
            Err(error) => {
                eprintln!("[flashkern] coordinator init failed: {error}");
                // SAFETY: constructor rollback owns the unpublished engine.
                unsafe { lfm_engine_free(p) };
                return None;
            }
        };
        let context = coordinator.context();
        // SAFETY: the context points into `coordinator`'s stable Arc allocation. It
        // remains live until Drop clears the callback and joins both endpoint owners.
        let rc = unsafe { lfm_engine_set_submitter(p, coordinator::submit, context) };
        if rc != 0 {
            // SAFETY: stop closes bridge admission before coordinator teardown.
            unsafe { lfm_engine_request_stop(p) };
            coordinator.shutdown();
            // SAFETY: all Rust endpoint owners have joined.
            unsafe { lfm_engine_free(p) };
            return None;
        }
        Some(Self {
            ptr: p,
            coordinator,
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
}

impl NativeEngine {
    /// Install the resident backbone layer table. The pointers must stay valid until
    /// [`Self::ctx_clear`] — the [`BackboneCtxGuard`] enforces clear-before-drop.
    /// Single-tenant: returns the install id, or `None` while another install is
    /// live (that caller keeps its bit-identical candle path).
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
        if rc == -4 {
            // Observability for the one legitimate refusal: a CPU→CPU model swap
            // where the previous model is still alive. That model decodes on the
            // bit-identical candle path until the old install drops.
            eprintln!(
                "[flashkern] ctx install refused: another model's table is live; \
                 this model decodes on the candle path"
            );
        }
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
        lanes: usize,
    ) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        let (logits_ptr, logits_len) = out_logits
            .map(|l| (l.as_mut_ptr(), l.len()))
            .unwrap_or((std::ptr::null_mut(), 0));
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

    /// Run a lane-uniform program on the whole team: `f(lane)` executes concurrently
    /// on every lane (0..lanes_total). Blocks until every lane completes (the
    /// engine's program-final fence). One doorbell in, one completion out.
    ///
    /// Each logical lane is pinned to one fixed pthread for the engine lifetime, so
    /// this transitional Rust callback cannot migrate a live frame. New numerical
    /// programs still belong in C++/assembly; this surface remains only until the
    /// depthformer callback is ported.
    /// A panic in `f` aborts the process (it cannot unwind across the C boundary).
    #[must_use = "false = engine refused; caller must take the fallback dispatch"]
    pub fn run_lanes<F: Fn(usize) + Sync>(&self, f: F) -> bool {
        unsafe extern "C" fn trampoline<F: Fn(usize) + Sync>(
            ctx: *mut c_void,
            lane: u32,
            _lanes: u32,
        ) {
            let call = || {
                // SAFETY: ctx is &F, valid for the blocking duration of run_lanes.
                let f = unsafe { &*(ctx as *const F) };
                f(lane as usize);
            };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).is_err() {
                // Unwinding into kcoro/C++ frames is UB; die loudly instead.
                eprintln!("[flashkern] panic in lane program (lane {lane}); aborting");
                std::process::abort();
            }
        }
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: single-slot engine serialized by pass_lock; &f outlives the
        // blocking call; trampoline::<F> matches the C ABI.
        let rc =
            unsafe { lfm_engine_call(self.ptr, trampoline::<F>, &f as *const F as *mut c_void) };
        rc == 0
    }

    /// Fan a flat grid of `n` INDEPENDENT items across the lane team: `f(item)`
    /// runs for every `item` in `0..n`, each on whichever lane claims it via a
    /// strided partition (`lane, lane+lanes, …`). The items must be genuinely
    /// disjoint (no shared mutable state, no cross-item ordering) — then the
    /// result is byte-identical to any serial or foreign-pool fan-out, which is
    /// what lets this REPLACE rayon `par_chunks` grids without moving a bit.
    ///
    /// This is the prefill/grid analog of [`run_lanes`](Self::run_lanes): same
    /// team, same doorbell, zero-spin when idle. Like `run_lanes`, `f` must not
    /// call into kcoro, and — because this itself dispatches on the team — it
    /// must NOT be called from inside a lane program (that would re-enter the
    /// single-slot engine and deadlock). All current callers are candle
    /// CustomOps at graph scope; none run inside a lane program.
    pub fn grid<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        if n == 0 {
            return;
        }
        let lanes = self.lanes_total().max(1);
        let dispatched = self.run_lanes(|lane| {
            let mut item = lane;
            while item < n {
                f(item);
                item += lanes;
            }
        });
        // The engine is the substrate — run_lanes only returns false on a hard
        // C-side failure, which for a resident team is a broken invariant, not
        // a slow path. Surface it; do NOT silently drop the grid's work.
        assert!(dispatched, "engine.grid: lane-team dispatch refused");
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
        lanes: usize,
    ) -> bool {
        unsafe {
            process_engine().token_pass(
                self.id, ids, embed_kind, states, pos, cos_base, sin_base, rope_len, out_hidden,
                out_logits, lanes,
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
/// when the build fails or another model's install is live (single-tenant) —
/// callers keep the per-block path. Engine presence is unconditional.
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
        let context = self.coordinator.context();
        // SAFETY: pass_lock excludes every safe Rust call. Clearing the callback
        // prevents C++ from retaining a pointer into the coordinator during teardown.
        let rc = unsafe { lfm_engine_clear_submitter(self.ptr, context) };
        if rc != 0 {
            std::process::abort();
        }
        // SAFETY: closes native admission and wakes both endpoint doorbells.
        unsafe { lfm_engine_request_stop(self.ptr) };
        self.coordinator.shutdown();
        // SAFETY: Rust SQ/CQ owners are joined; this joins the native dispatcher and
        // fixed lane team, then destroys the fully drained bridge.
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
    use std::sync::{Arc, Condvar};

    // The process engine holds ONE resident layer table; tests that build/clear it
    // must not interleave. (Each individual call is pass_lock-serialized; this guards
    // the build→use→clear SEQUENCE.)
    static CTX_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct RawGate {
        state: Mutex<(usize, bool)>,
        ready: Condvar,
    }

    unsafe extern "C" fn block_raw_lane(ctx: *mut c_void, _lane: u32, _lanes: u32) {
        // SAFETY: the test holds the Arc until both raw calls have returned.
        let gate = unsafe { &*(ctx.cast::<RawGate>()) };
        let mut state = gate.state.lock().unwrap();
        state.0 += 1;
        gate.ready.notify_all();
        while !state.1 {
            state = gate.ready.wait(state).unwrap();
        }
    }

    unsafe extern "C" fn noop_raw_lane(_: *mut c_void, _: u32, _: u32) {}

    unsafe extern "C" fn fence_raw_lane(ctx: *mut c_void, lane: u32, _: u32) {
        // SAFETY: `ctx` is this live engine and the callback runs on one of its lanes.
        unsafe { lfm_lane_fence(ctx, lane) };
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
    fn native_engine_ctx_single_tenant() {
        // Two installs cannot coexist (the two-model clobber): the second build is
        // refused while the first is live; a refused/stale id cannot clear the
        // owner's table; releasing the owner reopens the slot.
        let _ctx = CTX_TEST_LOCK.lock().unwrap();
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable — skipping");
            return;
        }
        let engine = process_engine();
        let h = 64usize;
        let k = 3usize;
        let i = 96usize;
        let op_norm = vec![0x3f80u16; h];
        let ffn_norm = vec![0x3f80u16; h];
        let in_w = vec![0u16; 3 * h * h];
        let conv_w = vec![0u16; h * k];
        let out_w = vec![0u16; h * h];
        let w1 = vec![0u16; i * h];
        let w3 = vec![0u16; i * h];
        let w2 = vec![0u16; h * i];
        let descs = [LayerDesc {
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
        }];
        let first = engine.ctx_build(&descs, h, i, 64).expect("first build");
        assert!(
            engine.ctx_build(&descs, h, i, 64).is_none(),
            "second install must be refused while the first is live"
        );
        // A stale/foreign id must not release the owner's install.
        engine.ctx_clear(first + 1);
        let x = vec![0u16; h];
        let state = vec![0u16; h * (k - 1)];
        let mut state_out = vec![0u16; h * (k - 1)];
        let mut out = vec![0u16; h];
        assert!(
            engine.conv_layer(first, 0, &x, &state, &mut state_out, &mut out, 1),
            "owner's table must survive a stale clear"
        );
        assert!(
            !engine.conv_layer(first + 1, 0, &x, &state, &mut state_out, &mut out, 1),
            "a foreign context id must not execute against the live table"
        );
        let short_input = vec![0u16; state.len() - 1];
        let mut short_state = vec![0u16; state.len() - 1];
        assert!(
            !engine.conv_layer(first, 0, &x, &short_input, &mut short_state, &mut out, 1,),
            "the native boundary must reject an undersized conv state"
        );

        let vocab = 4usize;
        let embed = vec![0u16; vocab * h];
        let norm = vec![0x3f80u16; h];
        assert!(!unsafe {
            engine.set_heads(
                first + 1,
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
        assert!(unsafe {
            engine.set_heads(
                first,
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
        let mut token_state = state.clone();
        let states = [LayerState {
            conv_state: token_state.as_mut_ptr(),
            conv_len: token_state.len(),
            ..LayerState::none()
        }];
        let mut hidden = vec![0u16; h];
        let mut short_logits = vec![0f32; vocab - 1];
        assert!(
            !unsafe {
                engine.token_pass(
                    first,
                    &[0],
                    0,
                    &states,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &mut hidden,
                    Some(&mut short_logits),
                    1,
                )
            },
            "the native boundary must reject an undersized logits buffer"
        );
        // The owner's clear releases the slot; a new install then succeeds.
        engine.ctx_clear(first);
        assert!(
            !engine.conv_layer(first, 0, &x, &state, &mut state_out, &mut out, 1),
            "cleared table must refuse passes"
        );
        let second = engine
            .ctx_build(&descs, h, i, 64)
            .expect("post-release build");
        assert_ne!(first, second, "install ids must be unique");
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
        let gate = Arc::new(RawGate {
            state: Mutex::new((0, false)),
            ready: Condvar::new(),
        });
        let engine_address = engine.ptr as usize;
        let gate_address = Arc::as_ptr(&gate) as usize;
        let first = std::thread::spawn(move || unsafe {
            lfm_engine_call(
                engine_address as *mut c_void,
                block_raw_lane,
                gate_address as *mut c_void,
            )
        });

        let mut state = gate.state.lock().unwrap();
        while state.0 != 2 {
            state = gate.ready.wait(state).unwrap();
        }
        drop(state);

        let (send, recv) = std::sync::mpsc::channel();
        let contender = std::thread::spawn(move || {
            let rc = unsafe {
                lfm_engine_call(
                    engine_address as *mut c_void,
                    noop_raw_lane,
                    std::ptr::null_mut(),
                )
            };
            send.send(rc).unwrap();
        });
        let immediate = recv.recv_timeout(std::time::Duration::from_millis(100));

        let mut state = gate.state.lock().unwrap();
        state.1 = true;
        gate.ready.notify_all();
        drop(state);

        let rc = immediate.unwrap_or_else(|_| {
            recv.recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
        });
        assert_eq!(
            rc,
            -libc::EBUSY,
            "raw contender entered the single-slot engine"
        );
        assert_eq!(first.join().unwrap(), 0);
        contender.join().unwrap();
    }

    #[test]
    fn raw_engine_requires_a_coordinator_without_leaking_its_descriptor() {
        // SAFETY: this test deliberately exercises the unpublished C constructor so
        // the no-submitter failure path cannot hide behind NativeEngine::new.
        let raw = unsafe { lfm_engine_new(2) };
        assert!(!raw.is_null());
        let rc = unsafe { lfm_engine_call(raw, noop_raw_lane, std::ptr::null_mut()) };
        assert_eq!(rc, -libc::ENOTCONN);

        let mut snapshot = EngineSnapshot {
            size: std::mem::size_of::<EngineSnapshot>() as u32,
            abi_version: 1,
            ..EngineSnapshot::default()
        };
        assert_eq!(unsafe { lfm_engine_snapshot(raw, &mut snapshot) }, 0);
        assert_eq!(snapshot.pass_submissions, 0);
        assert_eq!(snapshot.bridge_dispatches, 0);
        assert_eq!(snapshot.descriptor_acquires, 1);
        assert_eq!(snapshot.descriptor_retains, 0);
        assert_eq!(snapshot.descriptor_releases, 1);
        assert_eq!(snapshot.descriptors_live, 0);
        // SAFETY: no bridge ticket was accepted and the descriptor pool is settled.
        unsafe { lfm_engine_free(raw) };
    }

    #[test]
    fn native_engine_bridge_and_fence_soak() {
        let engine = NativeEngine::new(8).expect("native engine init");
        const PASSES: u64 = 10_000;
        const LANES: u64 = 8;
        const FENCES_PER_PASS: u64 = 2;
        let start = std::time::Instant::now();
        for pass in 0..PASSES {
            // SAFETY: the engine is live, no other caller shares it, and `ctx` is the
            // same engine pointer consumed synchronously by every fixed lane.
            let rc = unsafe { lfm_engine_call(engine.ptr, fence_raw_lane, engine.ptr) };
            assert_eq!(rc, 0, "pass {pass} did not complete");
        }
        let stats = engine.snapshot();
        let coordinator = engine.coordinator.snapshot();
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
        assert_eq!(coordinator.admitted, PASSES);
        assert_eq!(coordinator.native_submissions, PASSES);
        assert_eq!(coordinator.native_completions, PASSES);
        assert_eq!(coordinator.resolved, PASSES);
        assert_eq!(coordinator.failed, 0);
        assert_eq!(coordinator.edge_signals, PASSES);
        assert_eq!(coordinator.live, 0);
        assert_eq!(coordinator.max_generation, PASSES as u32);
        assert!(coordinator.executor_polls > PASSES);
        assert!(coordinator.executor_wakes > 0);
        assert!(!coordinator.fault);
        eprintln!(
            "native bridge/fence soak: {PASSES} passes, {} fence syscalls for {} waiters in {:.3}s",
            stats.fence_wake_calls,
            stats.fence_wakes,
            start.elapsed().as_secs_f64(),
        );
    }
}
