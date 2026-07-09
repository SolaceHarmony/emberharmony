//! The Rust rim of the resident native decode engine (csrc/flashkern_engine.cpp).
//!
//! Everything below the ABI line is C++: the persistent kcoro team, the block
//! schedules, the stage kernels. Rust's per-pass surface is one blocking call —
//! internally: write the request slot, `kcoro_unpark` the parked coordinator (the
//! doorbell), park on a condvar until the pass boundary. No Rust between stages.

#![cfg(all(
    has_kcoro,
    has_native_engine,
    any(
        all(target_arch = "aarch64", has_flashkern_neon),
        all(target_arch = "x86_64", has_flashkern_x86)
    )
))]

use std::ffi::c_void;
use std::sync::Mutex;

/// Mirror of the C `LfmConvLayerDesc` (flashkern_engine.cpp) — one per backbone block,
/// indexed by block_idx. Field order/types must match the C struct exactly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvLayerDesc {
    pub kind: u32, // 0 = shortconv+mlp, 1 = attention (placeholder until rung 2)
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
}

impl ConvLayerDesc {
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
        }
    }
}

extern "C" {
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_free(e: *mut c_void);
    fn lfm_ctx_build(
        e: *mut c_void,
        descs: *const ConvLayerDesc,
        n_layers: usize,
        h: usize,
        ffn: usize,
    ) -> i32;
    fn lfm_ctx_clear(e: *mut c_void);
    fn lfm_engine_conv_layer(
        e: *mut c_void,
        layer: usize,
        x: *const u16,
        state_in: *const u16,
        state_out: *mut u16,
        out: *mut u16,
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
/// call under `pass_lock`; that lock is what makes the `Sync` below true.
pub struct NativeEngine {
    ptr: *mut c_void,
    pass_lock: Mutex<()>,
}

// SAFETY: Send — the handle is an opaque pointer to a C-heap object with no thread
// affinity. Sync — provided by `pass_lock` above serializing every call into the
// SINGLE-SLOT C engine (one Pass, one scratch arena, one request word); the C side's
// own mutex only covers the completion handshake, NOT concurrent request setup.
// Removing the lock reintroduces the data race, whatever the C side looks like.
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    pub fn new(workers: usize) -> Option<Self> {
        // SAFETY: plain constructor call; null = failure.
        let p = unsafe { lfm_engine_new(workers as i32) };
        if p.is_null() {
            None
        } else {
            Some(Self {
                ptr: p,
                pass_lock: Mutex::new(()),
            })
        }
    }

    /// One fused-MLP decode block, entirely native — bit-identical to
    /// [`super::decode::fused_mlp_decode`] at the same `lanes`.
    #[must_use = "false = native pass did not run; caller must take the fallback"]
    pub fn fused_mlp(
        &self,
        x: &[u16],
        w: &super::decode::FusedMlpWeights,
        out: &mut [u16],
        lanes: usize,
    ) -> bool {
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
    fn ctx_build(&self, descs: &[ConvLayerDesc], h: usize, ffn: usize) -> bool {
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: descs copied by the C side before return; dims checked there.
        let rc = unsafe { lfm_ctx_build(self.ptr, descs.as_ptr(), descs.len(), h, ffn) };
        rc == 0
    }

    fn ctx_clear(&self) {
        // Serialized against passes: no pass can be in flight while we clear.
        let _pass = self.pass_lock.lock().unwrap();
        // SAFETY: engine pointer valid for the process lifetime.
        unsafe { lfm_ctx_clear(self.ptr) };
    }

    /// One whole shortconv+MLP layer in a single doorbell — bit-identical to the
    /// composed `fused_shortconv_decode` + `fused_mlp_decode` at the same `lanes`.
    #[must_use = "false = native pass did not run; caller must take the fallback"]
    pub fn conv_layer(
        &self,
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
                layer,
                x.as_ptr(),
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                out.as_mut_ptr(),
                lanes,
            )
        };
        rc == 0
    }
}

/// Keeps the resident layer table's backing alive and clears the table before it dies.
/// `held` owns tensors DERIVED at capture (e.g. the squeezed-contiguous conv weight);
/// the undived model weights are owned by the model, which must own this guard so the
/// guard drops (and clears the C table) before those weights do.
pub struct BackboneCtxGuard {
    _held: Vec<candle_core::Tensor>,
}

impl Drop for BackboneCtxGuard {
    fn drop(&mut self) {
        if let Some(engine) = process_engine() {
            engine.ctx_clear();
        }
    }
}

/// Build + install the backbone layer table on the process engine. Returns the guard
/// the MODEL must own (declared before its weight fields so it drops first), or `None`
/// when the engine is unavailable or the build fails — callers keep the per-block path.
pub fn install_backbone_ctx(
    descs: &[ConvLayerDesc],
    h: usize,
    ffn: usize,
    held: Vec<candle_core::Tensor>,
) -> Option<BackboneCtxGuard> {
    let engine = process_engine()?;
    if engine.ctx_build(descs, h, ffn) {
        Some(BackboneCtxGuard { _held: held })
    } else {
        None
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        // SAFETY: shuts the coordinator down, joins the team, releases the handles.
        unsafe { lfm_engine_free(self.ptr) };
    }
}

/// The process-resident engine for the model hot path (the same residency pattern as
/// rayon's global pool): built on first use, `None` when the runtime cannot come up —
/// callers fall back to the threadgroup port, which is bit-identical by the parity
/// test, so the fallback changes scheduling only, never numerics.
///
/// Lifetime is deliberately process-long: `OnceLock` never drops, so the team's
/// threads live until exit — the daemon shape this crate ships in. Workers are sized
/// by the crate's torch-parity thread policy (`threads::intraop_default_num_threads`:
/// P-cores only on Apple Silicon via `hw.perflevel0.physicalcpu`) — NOT
/// `available_parallelism`, which counts E-cores and reintroduces the tail-latency
/// imbalance the runtime documents as harmful.
pub fn process_engine() -> Option<&'static NativeEngine> {
    use std::sync::OnceLock;
    static ENGINE: OnceLock<Option<NativeEngine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let workers = crate::threads::intraop_default_num_threads().clamp(1, 16);
            NativeEngine::new(workers)
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_engine_conv_layer_bit_parity() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused kernels unavailable — skipping");
            return;
        }
        let Some(engine) = process_engine() else {
            eprintln!("native engine init failed — skipping");
            return;
        };
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
                ConvLayerDesc::attn_placeholder(),
                ConvLayerDesc {
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
                },
            ];
            assert!(engine.ctx_build(&descs, h, i), "ctx build failed");

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
                crate::flashkern::decode::fused_shortconv_decode(
                    &x, &scw, &state, &mut state_ref, &mut mid, lanes,
                );
                let mut out_ref = vec![0u16; h];
                crate::flashkern::decode::fused_mlp_decode(&mid, &mlpw, &mut out_ref, lanes);

                let mut state_got = vec![0u16; h * (k - 1)];
                let mut out_got = vec![0u16; h];
                assert!(
                    engine.conv_layer(1, &x, &state, &mut state_got, &mut out_got, lanes),
                    "engine refused conv_layer"
                );
                assert_eq!(state_got, state_ref, "state H={h} I={i} lanes={lanes}");
                assert_eq!(out_got, out_ref, "out H={h} I={i} lanes={lanes}");
            }
            engine.ctx_clear();
        }
    }

    #[test]
    fn native_engine_mlp_bit_parity() {
        use half::bf16;
        if !crate::flashkern::decode::fused_mlp_available() {
            eprintln!("fused mlp kernel unavailable — skipping");
            return;
        }
        let Some(engine) = NativeEngine::new(8) else {
            eprintln!("native engine init failed — skipping");
            return;
        };
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
                crate::flashkern::decode::fused_mlp_decode(&x, &w, &mut want, lanes);
                let mut got = vec![0u16; h];
                assert!(engine.fused_mlp(&x, &w, &mut got, lanes));
                assert_eq!(got, want, "H={h} I={i} lanes={lanes}");
            }
        }

        // Timing at the real decode shape: native engine vs Rust-dispatched kcoro
        // engine vs the rayon threadgroup port.
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
        let t = std::time::Instant::now();
        for _ in 0..50 {
            assert!(engine.fused_mlp(&x, &w, &mut out, lanes));
        }
        let native_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        let t = std::time::Instant::now();
        for _ in 0..50 {
            crate::flashkern::decode::fused_mlp_decode(&x, &w, &mut out, lanes);
        }
        let tg_ms = t.elapsed().as_secs_f64() * 1e3 / 50.0;
        eprintln!(
            "native engine fused_mlp {native_ms:.3} ms vs threadgroup+spin {tg_ms:.3} ms (H=1024 I=4096, lanes=8)"
        );
    }
}
