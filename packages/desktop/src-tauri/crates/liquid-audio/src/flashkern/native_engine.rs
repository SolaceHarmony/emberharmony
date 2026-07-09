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

extern "C" {
    fn lfm_engine_new(workers: i32) -> *mut c_void;
    fn lfm_engine_free(e: *mut c_void);
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
