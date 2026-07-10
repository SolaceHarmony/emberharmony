//! Fused decode blocks on the GPU dispatch model — the point of the exercise.
//!
//! The per-op execution the model otherwise uses on CPU (candle op → rayon fork/join →
//! tensor alloc → bf16↔f32 cast, ×~240 ops per decode token) is exactly what a GPU never
//! does: a GPU enters ONE dispatch per fused region and the data flows through threadgroup
//! memory between barrier-fenced stages. This module runs the decode step that way on CPU:
//!
//! | Metal                       | here |
//! |-----------------------------|------|
//! | `dispatch_thread_groups`    | one rayon scope over the persistent pool — the dispatch |
//! | simdgroup lanes             | exactly `lanes` concurrent workers, one per pool thread |
//! | `threadgroup float* shared` | the activation scratch the lanes co-own |
//! | `threadgroup_barrier`       | [`SpinBarrier`] — spinning, like the GPU's, not parking |
//!
//! Numerics: activations round through bf16 at EXACTLY the points the candle op chain
//! rounds (linear outputs, silu, gating mul, the norm's single output round, the residual
//! add) — the fused block changes *where the time goes*, not the trained-regime arithmetic.
//! Weights are read zero-copy in their checkpoint-native `[N,K]` layout via the nt dot
//! kernel; each lane owns a contiguous slice of output rows.

use super::Shared;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A spinning generation barrier — `threadgroup_barrier(mem_threadgroup)` semantics. GPU
/// barriers spin; parking (`std::sync::Barrier`) costs ~1-2 µs a crossing, which at
/// hundreds of crossings per token is real money. AcqRel on the generation flip publishes
/// each stage's shared-memory writes to every lane, same fence contract as the GPU.
/// One threadgroup dispatch at a time, process-wide. A spin-barrier dispatch has a
/// HARD scheduling requirement: all `lanes` tasks must run concurrently on the shared
/// rayon pool. Two overlapping dispatches can fill the pool with spinners that each
/// wait for the other's unstarted lanes — a livelock that burns every occupied core
/// (observed: 249 CPU-minutes in 33 wall-minutes across two concurrent release-suite
/// parity tests). Production decode is sequential, so this lock is uncontended there;
/// it exists to make the requirement structural instead of hoped-for. The resident
/// native stage machine has no such requirement — parked workers occupy nothing.
pub(crate) static DISPATCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) struct SpinBarrier {
    lanes: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    pub(crate) fn new(lanes: usize) -> Self {
        Self {
            lanes,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub(crate) fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.lanes {
            self.count.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::AcqRel); // release the cohort
        } else {
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
        }
    }
}

/// Stage fence for lane-uniform programs: the engine team's fence when the program
/// rides the kcoro lanes (bounded spin, then a precise parked wake — the engine's
/// own barrier), or the spinning [`SpinBarrier`] on the rayon threadgroup fallback.
/// Numerics never depend on which — banding and ladders are identical.
pub(crate) trait LaneFence: Sync {
    fn wait(&self, lane: usize);
}

impl LaneFence for SpinBarrier {
    fn wait(&self, _lane: usize) {
        SpinBarrier::wait(self);
    }
}


/// `true` when the fused decode blocks can run: the STRICT nt-kernel gate — the looser
/// [`bf16_gemm_available`](crate::bf16_gemm::bf16_gemm_available) is also satisfied by the
/// reference-kernel-only aarch64 build, which has no nt kernel and would panic in
/// [`fused_mlp_decode`]'s lane bodies.
pub fn fused_mlp_available() -> bool {
    crate::bf16_gemm::bf16_gemm_nt_available()
}

#[inline]
pub(crate) fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16 bits (the storage rounding the candle op chain applies
/// after every op on the bf16 CPU path).
#[inline]
pub(crate) fn rb_bits(f: f32) -> u16 {
    let u = f.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

// The nt dot kernel over a lane's row range, arch-dispatched.
// SAFETY: caller guarantees a=[K], w=n·K rows at `w`, c=n f32 at `c`, availability checked.
#[allow(unused_variables)]
pub(crate) unsafe fn nt_rows(a: *const u16, w: *const u16, c: *mut f32, n: usize, k: usize) {
    #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
    super::neon::bf16_gemm_nt_raw(a, w, c, n, k);
    #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
    super::x86::bf16_gemm_nt_raw(a, w, c, n, k);
    #[cfg(not(any(
        all(target_arch = "aarch64", has_flashkern_neon),
        all(target_arch = "x86_64", has_flashkern_x86)
    )))]
    unreachable!("fused decode called without a flashkern kernel — gate on fused_mlp_available()");
}

/// The LFM2 FFN residual block's weights, zero-copy bf16 bit slices in checkpoint layout:
/// `norm_w [H]`, `w1/w3 [I,H]` (gate/up), `w2 [H,I]` (down).
pub struct FusedMlpWeights<'a> {
    pub norm_w: &'a [u16],
    pub w1: &'a [u16],
    pub w3: &'a [u16],
    pub w2: &'a [u16],
    pub eps: f32,
}

/// One decode step of the FFN residual block — `out = rb(x + w2·(silu(w1·xn) ⊙ w3·xn))`,
/// `xn = rms_norm(x)·norm_w` — as a single threadgroup dispatch (3 barriers), replacing the
/// eight candle ops (norm, 3 linears + casts, silu, mul, add) the per-op path runs.
///
/// `x`/`out` are `[H]` bf16 bits. Stage map (each lane grid-strides or owns row slices):
/// 1. Σx² partials → all lanes reduce the `lanes` partials serially (same order — deterministic)
///    → `xn = rb(x · rsqrt(mean+eps) · w_norm)` (the real `RmsNorm::forward`'s single round).
/// 2. Lane's gate/up rows: `t[r] = rb(rb(silu(rb(g_r))) · rb(u_r))` — the op chain's rounds.
/// 3. Lane's down rows + residual: `out[r] = rb(rb(y_r) + x[r])`.
pub fn fused_mlp_decode(x: &[u16], w: &FusedMlpWeights, out: &mut [u16], lanes: usize) {
    let h = x.len();
    let i = w.w1.len() / h;
    assert!(h > 0 && i > 0, "fused_mlp_decode: empty dims");
    assert_eq!(w.norm_w.len(), h, "fused_mlp_decode: norm_w.len() != H");
    assert_eq!(w.w1.len(), i * h, "fused_mlp_decode: w1.len() != I·H");
    assert_eq!(w.w3.len(), i * h, "fused_mlp_decode: w3.len() != I·H");
    assert_eq!(w.w2.len(), h * i, "fused_mlp_decode: w2.len() != H·I");
    assert_eq!(out.len(), h, "fused_mlp_decode: out.len() != H");
    assert!(
        fused_mlp_available(),
        "fused_mlp_decode requires the flashkern nt kernel (gate on fused_mlp_available())"
    );
    let lanes = lanes.clamp(1, h.min(i));

    // Threadgroup scratch: sumsq partials, normed activation (bf16), gate/up dots (f32),
    // gated intermediate (bf16). Lane-disjoint writes, barrier-fenced reads.
    let mut partials = vec![0f32; lanes];
    let mut xn = vec![0u16; h];
    let mut gu = vec![0f32; 2 * i]; // g = gu[0..i], u = gu[i..2i]
    let mut t = vec![0u16; i];
    let _dispatch = DISPATCH_LOCK.lock().unwrap();
    let sh_part = Shared(partials.as_mut_ptr());
    let sh_xn = Shared(xn.as_mut_ptr());
    let sh_gu = Shared(gu.as_mut_ptr());
    let sh_t = Shared(t.as_mut_ptr());
    let sh_out = Shared(out.as_mut_ptr());
    let barrier = SpinBarrier::new(lanes);
    let barrier = &barrier;

    // Row ownership: contiguous slices so each lane's nt call streams contiguous weight rows.
    let i_chunk = i.div_ceil(lanes);
    let h_chunk = h.div_ceil(lanes);

    // dispatch_thread_groups == one rayon scope over the persistent pool; `lanes` concurrent
    // workers spin-sync inside it. Nothing else runs on the pool during a decode step.
    rayon::scope(|scope| {
        for lane in 0..lanes {
            scope.spawn(move |_| {
                // stage 1a: Σx² partial (grid-stride, f32).
                let mut s = 0f32;
                let mut idx = lane;
                while idx < h {
                    let v = bf16_f32(x[idx]);
                    s += v * v;
                    idx += lanes;
                }
                // SAFETY: partials[lane] is this lane's private slot.
                unsafe { sh_part.set(lane, s) };
                barrier.wait(); // threadgroup_barrier — partials visible

                // stage 1b: every lane folds the partials in the same serial order
                // (deterministic), then normalizes its grid-stride cells. Matches
                // RmsNorm::forward: f32 throughout, ONE bf16 round after ·w_norm.
                let mut total = 0f32;
                for l in 0..lanes {
                    // SAFETY: post-barrier read-only view of the partials.
                    total += unsafe { sh_part.get(l) };
                }
                let rs = 1.0f32 / (total / h as f32 + w.eps).sqrt();
                let mut idx = lane;
                while idx < h {
                    let v = bf16_f32(x[idx]) * rs * bf16_f32(w.norm_w[idx]);
                    // SAFETY: cell idx is this lane's private grid-stride slot.
                    unsafe { sh_xn.set(idx, rb_bits(v)) };
                    idx += lanes;
                }
                barrier.wait(); // threadgroup_barrier — xn visible before the gate/up dots

                // stage 2: this lane's contiguous gate/up rows — two nt dots over xn, then
                // the op chain's exact rounding ladder into the gated intermediate.
                let r0 = (lane * i_chunk).min(i);
                let r1 = ((lane + 1) * i_chunk).min(i);
                if r1 > r0 {
                    let n = r1 - r0;
                    // SAFETY: xn is post-barrier read-only; g/u row ranges are lane-private;
                    // w1/w3 row slices are in-bounds by the entry asserts.
                    unsafe {
                        nt_rows(
                            sh_xn.ptr(),
                            w.w1.as_ptr().add(r0 * h),
                            sh_gu.ptr().add(r0),
                            n,
                            h,
                        );
                        nt_rows(
                            sh_xn.ptr(),
                            w.w3.as_ptr().add(r0 * h),
                            sh_gu.ptr().add(i + r0),
                            n,
                            h,
                        );
                        for r in r0..r1 {
                            let g = bf16_f32(rb_bits(sh_gu.get(r))); // linear-out round
                            let sg = rb_bits(g / (1.0 + (-g).exp())); // silu round
                            let u = rb_bits(sh_gu.get(i + r)); // linear-out round
                            sh_t.set(r, rb_bits(bf16_f32(sg) * bf16_f32(u))); // gating-mul round
                        }
                    }
                }
                barrier.wait(); // threadgroup_barrier — t visible before the down dots

                // stage 3: this lane's contiguous down rows + residual, straight to out.
                let r0 = (lane * h_chunk).min(h);
                let r1 = ((lane + 1) * h_chunk).min(h);
                if r1 > r0 {
                    let n = r1 - r0;
                    let mut y = vec![0f32; n]; // lane-private accumulator
                                               // SAFETY: t is post-barrier read-only; w2 row slice in-bounds; out rows
                                               // are lane-private.
                    unsafe {
                        nt_rows(sh_t.ptr(), w.w2.as_ptr().add(r0 * i), y.as_mut_ptr(), n, i);
                        for (j, &yv) in y.iter().enumerate() {
                            let d = bf16_f32(rb_bits(yv)); // linear-out round
                            let r = rb_bits(d + bf16_f32(x[r0 + j])); // residual-add round
                            sh_out.set(r0 + j, r);
                        }
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::linear::linear_forward;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Linear;
    use half::bf16;

    fn rnd(i: usize, seed: usize) -> f32 {
        (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 2000) as f32 / 1000.0) - 1.0
    }

    // The production op chain, op for op: RmsNorm::forward's math (f32 norm, ·w, one bf16
    // round), then silu(linear)·linear → linear through the REAL linear_forward, then the
    // bf16 residual add — the exact path DecoderLayer runs today.
    fn reference(x: &[u16], w: &FusedMlpWeights, h: usize, i: usize) -> Vec<u16> {
        let dev = Device::Cpu;
        let up = |bits: &[u16]| -> Vec<f32> { bits.iter().map(|&b| bf16_f32(b)).collect() };
        let t_bf16 = |v: Vec<f32>, shape: (usize, usize)| {
            Tensor::from_vec(v, shape, &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let xt = t_bf16(up(x), (1, h));
        // RmsNorm::forward: f32 → mean(x²) → rsqrt via sqrt+recip → ·x → ·w → bf16.
        let xf = xt.to_dtype(DType::F32).unwrap();
        let mean_sq = xf.sqr().unwrap().mean_keepdim(1).unwrap();
        let rs = (mean_sq + w.eps as f64)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = xf.broadcast_mul(&rs).unwrap();
        let wn = Tensor::from_vec(up(w.norm_w), (h,), &dev).unwrap();
        let xn = normed
            .broadcast_mul(&wn)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // SwiGLU through the real linear path.
        let lin = |wbits: &[u16], o: usize, k: usize| Linear::new(t_bf16(up(wbits), (o, k)), None);
        let gate = linear_forward(&lin(w.w1, i, h), &xn).unwrap();
        let gate = candle_nn::ops::silu(&gate).unwrap();
        let upv = linear_forward(&lin(w.w3, i, h), &xn).unwrap();
        let tmid = (gate * upv).unwrap();
        let down = linear_forward(&lin(w.w2, h, i), &tmid).unwrap();
        let out = (down + xt).unwrap();
        out.flatten_all()
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn fused_mlp_matches_candle_op_chain() {
        if !fused_mlp_available() {
            eprintln!("fused MLP unavailable on this CPU — skipping");
            return;
        }
        let (h, i) = (128usize, 342usize); // odd I exercises ragged row ownership
        let mk = |n: usize, seed: usize| -> Vec<u16> {
            (0..n)
                .map(|j| bf16::from_f32(rnd(j, seed) * 0.25).to_bits())
                .collect()
        };
        let x = mk(h, 3);
        let (norm_w, w1, w3, w2) = (mk(h, 7), mk(i * h, 11), mk(i * h, 13), mk(h * i, 17));
        let w = FusedMlpWeights {
            norm_w: &norm_w,
            w1: &w1,
            w3: &w3,
            w2: &w2,
            eps: 1e-5,
        };
        let want = reference(&x, &w, h, i);
        for lanes in [1usize, 3, 8] {
            let mut got = vec![0u16; h];
            fused_mlp_decode(&x, &w, &mut got, lanes);
            let (mut md, mut sc) = (0f32, 1e-3f32);
            for (g, r) in got.iter().zip(&want) {
                md = md.max((bf16_f32(*g) - bf16_f32(*r)).abs());
                sc = sc.max(bf16_f32(*r).abs());
            }
            // Everything rounds through bf16 at the same points; the only latitude is dot
            // summation order, so parity sits at bf16 resolution.
            assert!(
                md / sc < 3e-2,
                "lanes={lanes}: fused vs op chain rel {}",
                md / sc
            );
        }
    }

    #[test]
    fn fused_mlp_lane_determinism() {
        // Same lane count twice → bit-identical (fixed ownership, fixed reduce order).
        if !fused_mlp_available() {
            return;
        }
        let (h, i) = (64usize, 96usize);
        let mk = |n: usize, seed: usize| -> Vec<u16> {
            (0..n)
                .map(|j| bf16::from_f32(rnd(j, seed) * 0.25).to_bits())
                .collect()
        };
        let x = mk(h, 23);
        let (norm_w, w1, w3, w2) = (mk(h, 29), mk(i * h, 31), mk(i * h, 37), mk(h * i, 41));
        let w = FusedMlpWeights {
            norm_w: &norm_w,
            w1: &w1,
            w3: &w3,
            w2: &w2,
            eps: 1e-5,
        };
        let (mut a, mut b) = (vec![0u16; h], vec![0u16; h]);
        fused_mlp_decode(&x, &w, &mut a, 4);
        fused_mlp_decode(&x, &w, &mut b, 4);
        assert_eq!(a, b, "same dispatch shape must be bit-identical");
    }
}

// ======================================================================================
// Pure-NEON depthformer decode — candle stripped from the audio-frame hot loop.
//
// Profiling showed `sample_audio_frame` (8 sequential codebook steps × 6 StandardBlocks,
// per audio frame) dominating decode, every op a candle dispatch. This section runs the
// whole frame as ONE threadgroup dispatch: lanes walk the codebook steps and layers with
// spin barriers between stages; weights are read zero-copy from the checkpoint tensors;
// KV lives in tiny resident f32 planes (cursor reset per frame — zero allocation); every
// bf16 round sits exactly where the candle op chain rounds, so the flash path is
// value-equivalent at the same tier as the fused MLP block.
// ======================================================================================

#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
extern "C" {
    fn lfm_bf16_sumsq_f32(x: *const u16, n: i32) -> f32;
    fn lfm_bf16_sumsq_seq_f32(x: *const u16, n: i32) -> f32;
    fn lfm_bf16_sumsq_candle_f32(x: *const u16, n: i32) -> f32;
    fn lfm_bf16_rmsnorm(x: *const u16, w: *const u16, out: *mut u16, n: i32, inv_rms: f32);
    fn lfm_bf16_add(a: *const u16, b: *const u16, out: *mut u16, n: i32);
    fn lfm_swiglu_bf16(g: *const f32, u: *const f32, out: *mut u16, n: i32);
    fn lfm_softmax_scaled_f32(x: *mut f32, n: i32, scale: f32);
    fn lfm_attn_av_f32(att: *const f32, v: *const f32, out: *mut f32, len: i32, hd: i32);
    fn lfm_attn_qk_f32(q: *const f32, k: *const f32, att: *mut f32, len: i32, hd: i32);
    fn lfm_attn_qk_bf16(q: *const f32, k: *const u16, att: *mut f32, len: i32, hd: i32);
    fn lfm_attn_av_bf16(att: *const f32, v: *const u16, out: *mut f32, len: i32, hd: i32);
    fn lfm_rope_i_f32(x: *mut f32, cos_p: *const f32, sin_p: *const f32, hd: i32);
    fn lfm_bf16_to_f32(x: *const u16, out: *mut f32, n: i32);
    fn lfm_f32_to_bf16(x: *const f32, out: *mut u16, n: i32);
}

/// A raw view of a checkpoint tensor's storage, stored as `usize` so the ctx stays `Send`.
/// SAFETY CONTRACT: the owning model outlives the ctx (both live in `LFM2AudioModel`), and
/// candle storages are `Arc`-heap — moves of the model don't move the data.
#[derive(Clone, Copy)]
pub struct PtrLen {
    ptr: usize,
    len: usize,
}

impl PtrLen {
    /// Capture a contiguous CPU bf16 tensor as raw bits. `None` if it isn't one.
    pub fn bf16(t: &candle_core::Tensor) -> Option<Self> {
        use candle_core::Storage;
        let (s, l) = t.storage_and_layout();
        match &*s {
            Storage::Cpu(candle_core::CpuStorage::BF16(v)) => {
                let (a, b) = l.contiguous_offsets()?;
                Some(Self {
                    ptr: v[a..b].as_ptr() as usize,
                    len: b - a,
                })
            }
            _ => None,
        }
    }
    /// The raw address (usize-stored; see the safety contract above).
    pub(crate) fn addr(&self) -> usize {
        self.ptr
    }
    /// Element count of the captured view.
    pub(crate) fn size(&self) -> usize {
        self.len
    }
    /// Capture a contiguous CPU f32 tensor.
    pub fn f32(t: &candle_core::Tensor) -> Option<Self> {
        use candle_core::Storage;
        let (s, l) = t.storage_and_layout();
        match &*s {
            Storage::Cpu(candle_core::CpuStorage::F32(v)) => {
                let (a, b) = l.contiguous_offsets()?;
                Some(Self {
                    ptr: v[a..b].as_ptr() as usize,
                    len: b - a,
                })
            }
            _ => None,
        }
    }
    #[inline]
    fn u16_ptr(&self) -> *const u16 {
        self.ptr as *const u16
    }
    #[inline]
    fn f32_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }
}

/// One depthformer StandardBlock's weights (bf16 bits, checkpoint layout).
pub struct DepthLayer {
    pub qkv_w: PtrLen,   // [dim + 2·kvh·hd, dim]
    pub out_w: PtrLen,   // [dim, dim]
    pub q_ln: PtrLen,    // [hd]
    pub k_ln: PtrLen,    // [hd]
    pub opnorm: PtrLen,  // [dim]
    pub ffnnorm: PtrLen, // [dim]
    pub w1: PtrLen,      // [ff, dim]
    pub w3: PtrLen,      // [ff, dim]
    pub w2: PtrLen,      // [dim, ff]
}

/// One codebook's `SharedEmbedding` (embed table + pre-logits norm + tied head).
pub struct DepthHead {
    pub emb: PtrLen,    // [vocab, dim]
    pub norm: PtrLen,   // [dim]
    pub logits: PtrLen, // [vocab, dim]
    pub vocab: usize,
}

struct DepthScratch {
    x: Vec<u16>,        // [dim] running hidden (bf16 bits)
    h: Vec<u16>,        // [dim] post-attention residual
    xn: Vec<u16>,       // [dim] normed input to qkv / glu
    qkv_f: Vec<f32>,    // [qkv_out] GEMV accumulators
    qkv_b: Vec<u16>,    // [qkv_out] rounded linear outputs
    u_f: Vec<f32>,      // [ff] up-projection plane (w3)
    y_b: Vec<u16>,      // [max plane] rounded-bits staging for residual adds
    q_f: Vec<f32>,      // [heads·hd] post-rope f32 queries
    attn_f: Vec<f32>,   // [dim] attention output (f32, per-head slices)
    attn_b: Vec<u16>,   // [dim]
    proj_f: Vec<f32>,   // [max(dim, ff, vocab)] general GEMV plane
    t_b: Vec<u16>,      // [ff] gated intermediate
    kplane: Vec<u16>,   // [layers][kvh][cap][hd] bf16 bits — torch's cache dtype
    vplane: Vec<u16>,   // same
    logits_b: Vec<u16>, // [vocab]
    din_b: Vec<u16>,    // [codebooks·dim]
    df_b: Vec<u16>,     // [dim] running depth token embedding
    partials: Vec<f32>, // [lanes]
}

/// The pure-NEON depthformer frame decoder. Built once from the model's tensors; per
/// frame it runs one dispatch with zero allocation and zero candle ops.
pub struct DepthDecode {
    pub dim: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
    pub ff: usize,
    pub codebooks: usize,
    pub backbone_dim: usize,
    pub eps: f32,
    pub layers: Vec<DepthLayer>,
    pub heads_w: Vec<DepthHead>,
    pub depth_lin_w: PtrLen, // [codebooks·dim, backbone_dim]
    pub depth_lin_b: PtrLen, // [codebooks·dim]
    pub cos: PtrLen,         // [max_seq, hd/2] f32 rope table (layer-shared)
    pub sin: PtrLen,
    scratch: std::sync::Mutex<DepthScratch>,
}

// Send/Sync are compiler-derived: PtrLen stores addresses as usize, scratch sits behind a
// Mutex (frame() locks it — concurrent frame() calls serialize instead of racing a RefCell
// borrow flag, per review). Dereferencing the PtrLen views remains the documented contract:
// the owning model outlives the ctx and candle storages are Arc-heap.

#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
impl DepthDecode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        heads: usize,
        kv_heads: usize,
        ff: usize,
        codebooks: usize,
        backbone_dim: usize,
        eps: f32,
        layers: Vec<DepthLayer>,
        heads_w: Vec<DepthHead>,
        depth_lin_w: PtrLen,
        depth_lin_b: PtrLen,
        cos: PtrLen,
        sin: PtrLen,
    ) -> Self {
        let hd = dim / heads;
        let qkv_out = dim + 2 * kv_heads * hd;
        let vocab_max = heads_w.iter().map(|h| h.vocab).max().unwrap_or(0);
        // proj_f serves every GEMV in the program — including stage 0's depth_linear whose
        // output is codebooks·dim rows, the largest plane in the frame.
        let plane = dim.max(ff).max(vocab_max).max(qkv_out).max(codebooks * dim);
        let scratch = DepthScratch {
            x: vec![0; dim],
            h: vec![0; dim],
            xn: vec![0; dim],
            qkv_f: vec![0.0; qkv_out],
            qkv_b: vec![0; qkv_out],
            u_f: vec![0.0; ff],
            y_b: vec![0; plane],
            q_f: vec![0.0; heads * hd],
            attn_f: vec![0.0; dim],
            attn_b: vec![0; dim],
            proj_f: vec![0.0; plane],
            t_b: vec![0; ff],
            kplane: vec![0; layers.len() * kv_heads * codebooks * hd],
            vplane: vec![0; layers.len() * kv_heads * codebooks * hd],
            logits_b: vec![0; vocab_max],
            din_b: vec![0; codebooks * dim],
            df_b: vec![0; dim],
            partials: vec![0.0; 64],
        };
        Self {
            dim,
            heads,
            kv_heads,
            hd,
            ff,
            codebooks,
            backbone_dim,
            eps,
            layers,
            heads_w,
            depth_lin_w,
            depth_lin_b,
            cos,
            sin,
            scratch: std::sync::Mutex::new(scratch),
        }
    }

    /// One audio frame: backbone hidden (bf16 bits, `[backbone_dim]`) → `codebooks` tokens.
    /// `sample` is called once per codebook with the rounded bf16 logits bits and must
    /// return the chosen token (the caller wraps its seeded Sampler — same RNG stream as
    /// the candle path). ONE dispatch; lanes walk steps/layers with spin barriers.
    pub fn frame(
        &self,
        emb_bits: &[u16],
        mut sample: impl FnMut(&[u16]) -> u32 + Send,
    ) -> Vec<u32> {
        assert_eq!(
            emb_bits.len(),
            self.backbone_dim,
            "depth frame: bad hidden size"
        );
        assert!(
            self.codebooks <= 64,
            "depth frame: att score buffer holds ≤64 steps"
        );
        assert!(
            self.hd <= 128,
            "depth frame: per-head buffers hold ≤128 lanes"
        );
        let (dim, heads, kvh, hd, cb_n) =
            (self.dim, self.heads, self.kv_heads, self.hd, self.codebooks);
        let group = heads / kvh;
        let qkv_out = dim + 2 * kvh * hd;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let mut s = self.scratch.lock().expect("depth scratch poisoned");
        let s = &mut *s;
        // Lane count comes from OUR team when the engine is up — a foreign pool has
        // no authority over this kernel's width. Non-engine builds keep rayon sizing.
        // Both size from the P-core count, so banding (and therefore bits) agrees.
        #[cfg(all(
        has_kcoro,
        has_native_engine,
        any(
            all(target_arch = "aarch64", has_flashkern_neon),
            all(target_arch = "x86_64", has_flashkern_x86)
        )
        ))]
        let lanes = crate::flashkern::native_engine::process_engine()
            .map(|e| e.lanes_total())
            .unwrap_or_else(|| rayon::current_num_threads())
            .clamp(1, 16);
        #[cfg(not(all(
        has_kcoro,
        has_native_engine,
        any(
            all(target_arch = "aarch64", has_flashkern_neon),
            all(target_arch = "x86_64", has_flashkern_x86)
        )
        )))]
        let lanes = rayon::current_num_threads().clamp(1, 16);

        // din = rb(depth_linear(emb) + bias): one GEMV over the backbone hidden, rounded
        // per element — the linear_forward ladder. Fanned over lanes below (row slices).
        let sh_din = Shared(s.din_b.as_mut_ptr());
        let sh_x = Shared(s.x.as_mut_ptr());
        let sh_h = Shared(s.h.as_mut_ptr());
        let sh_xn = Shared(s.xn.as_mut_ptr());
        let sh_qkvf = Shared(s.qkv_f.as_mut_ptr());
        let sh_qkvb = Shared(s.qkv_b.as_mut_ptr());
        let sh_uf = Shared(s.u_f.as_mut_ptr());
        let sh_yb = Shared(s.y_b.as_mut_ptr());
        let sh_qf = Shared(s.q_f.as_mut_ptr());
        let sh_attnf = Shared(s.attn_f.as_mut_ptr());
        let sh_attnb = Shared(s.attn_b.as_mut_ptr());
        let sh_projf = Shared(s.proj_f.as_mut_ptr());
        let sh_tb = Shared(s.t_b.as_mut_ptr());
        let sh_k = Shared(s.kplane.as_mut_ptr());
        let sh_v = Shared(s.vplane.as_mut_ptr());
        let sh_log = Shared(s.logits_b.as_mut_ptr());
        let sh_df = Shared(s.df_b.as_mut_ptr());
        let sh_part = Shared(s.partials.as_mut_ptr());

        let tokens: Vec<std::sync::atomic::AtomicU32> = (0..cb_n)
            .map(|_| std::sync::atomic::AtomicU32::new(u32::MAX))
            .collect();
        let tokens = &tokens;
        // The sampler runs on lane 0 between barriers; hand it the &mut via a take-once slot.
        let sampler_cell = std::sync::Mutex::new(Some(&mut sample));
        let this = &*self;

        // The lane-uniform frame program: every lane runs the whole walk, `fence`
        // separates the stages. On the engine team this is ONE doorbell per frame on
        // the same kcoro lanes as the backbone — no rayon, no SpinBarrier, no
        // DISPATCH_LOCK in the production path.
        let run_lane = |lane: usize, fence: &dyn LaneFence| {
                    let mut sampler_slot = if lane == 0 {
                        Some(
                            sampler_cell
                                .lock()
                                .unwrap()
                                .take()
                                .expect("sampler taken once"),
                        )
                    } else {
                        None
                    };
                    let own = |n: usize, l: usize| -> (usize, usize) {
                        let c = n.div_ceil(lanes);
                        ((l * c).min(n), ((l + 1) * c).min(n))
                    };
                    // GEMV helper over this lane's contiguous output rows: rows [r0,r1) of
                    // W[n,k] dotted with a bf16 bits vector, into an f32 plane.
                    let gemv = |w: &PtrLen,
                                x_bits: *const u16,
                                out: Shared<f32>,
                                n: usize,
                                k: usize,
                                l: usize| {
                        let (r0, r1) = {
                            let c = n.div_ceil(lanes);
                            ((l * c).min(n), ((l + 1) * c).min(n))
                        };
                        if r1 > r0 {
                            // SAFETY: lane-private output rows; weight rows in bounds.
                            unsafe {
                                nt_rows(
                                    x_bits,
                                    w.u16_ptr().add(r0 * k),
                                    out.ptr().add(r0),
                                    r1 - r0,
                                    k,
                                );
                            }
                        }
                    };

                    // ---- stage 0: din = rb(depth_linear·emb + bias) ----
                    gemv(
                        &this.depth_lin_w,
                        emb_bits.as_ptr(),
                        sh_projf,
                        cb_n * dim,
                        this.backbone_dim,
                        lane,
                    );
                    {
                        let (r0, r1) = own(cb_n * dim, lane);
                        // bias add in f32 (once per frame), then ONE slice round — the
                        // linear_forward ladder: rb(dot + bias), a single rounding.
                        for r in r0..r1 {
                            // SAFETY: lane-private rows.
                            unsafe {
                                let b = f32::from_bits(
                                    (*this.depth_lin_b.u16_ptr().add(r) as u32) << 16,
                                );
                                sh_projf.set(r, sh_projf.get(r) + b);
                            }
                        }
                        if r1 > r0 {
                            // SAFETY: lane-private slice.
                            unsafe {
                                lfm_f32_to_bf16(
                                    sh_projf.ptr().add(r0),
                                    sh_din.ptr().add(r0),
                                    (r1 - r0) as i32,
                                )
                            };
                        }
                        // df_token starts at zero.
                        let (d0, d1) = own(dim, lane);
                        for i in d0..d1 {
                            unsafe { sh_df.set(i, 0) };
                        }
                    }
                    fence.wait(lane);

                    for cb in 0..cb_n {
                        // ---- cur = rb(din[cb] + df_token) → x ----
                        {
                            let (d0, d1) = own(dim, lane);
                            if d1 > d0 {
                                // SAFETY: lane-private cells; din row cb read-only.
                                unsafe {
                                    lfm_bf16_add(
                                        sh_din.ptr().add(cb * dim + d0),
                                        sh_df.ptr().add(d0),
                                        sh_x.ptr().add(d0),
                                        (d1 - d0) as i32,
                                    );
                                }
                            }
                        }
                        fence.wait(lane);

                        for (li, lw) in this.layers.iter().enumerate() {
                            let kbase = (li * kvh) * cb_n * hd;
                            // ---- operator_norm(x) → xn ----
                            this.norm_stage(
                                sh_x.ptr(),
                                lw.opnorm,
                                sh_xn,
                                sh_part,
                                dim,
                                lane,
                                lanes,
                                fence,
                            );
                            // ---- qkv GEMV + per-row rb ----
                            gemv(&lw.qkv_w, sh_xn.ptr(), sh_qkvf, qkv_out, dim, lane);
                            {
                                let (r0, r1) = own(qkv_out, lane);
                                if r1 > r0 {
                                    // SAFETY: lane-private rows.
                                    unsafe {
                                        lfm_f32_to_bf16(
                                            sh_qkvf.ptr().add(r0),
                                            sh_qkvb.ptr().add(r0),
                                            (r1 - r0) as i32,
                                        )
                                    };
                                }
                            }
                            fence.wait(lane);
                            // ---- per-head qk-norm + rope; K/V rows into the resident planes ----
                            {
                                let total_heads = heads + kvh; // q heads then k heads
                                let (h0, h1) = own(total_heads, lane);
                                for hh in h0..h1 {
                                    // SAFETY: each head's slices are lane-private this stage.
                                    unsafe {
                                        if hh < heads {
                                            let src = sh_qkvb.ptr().add(hh * hd);
                                            let mut bits = [0u16; 128];
                                            this.qk_head_bits(
                                                src,
                                                lw.q_ln,
                                                bits.as_mut_ptr(),
                                                hd,
                                                cb,
                                            );
                                            // q is consumed in f32 (the sdpa upcast point).
                                            lfm_bf16_to_f32(
                                                bits.as_ptr(),
                                                sh_qf.ptr().add(hh * hd),
                                                hd as i32,
                                            );
                                        } else {
                                            let kh = hh - heads;
                                            let src = sh_qkvb.ptr().add(dim + kh * hd);
                                            this.qk_head_bits(
                                                src,
                                                lw.k_ln,
                                                sh_k.ptr().add(kbase + (kh * cb_n + cb) * hd),
                                                hd,
                                                cb,
                                            );
                                            // V: the rb'd projection bits, verbatim (cache dtype).
                                            let vsrc = sh_qkvb.ptr().add(dim + kvh * hd + kh * hd);
                                            std::ptr::copy_nonoverlapping(
                                                vsrc,
                                                sh_v.ptr().add(kbase + (kh * cb_n + cb) * hd),
                                                hd,
                                            );
                                        }
                                    }
                                }
                            }
                            fence.wait(lane);
                            // ---- attention per q-head over the planes ----
                            {
                                let (h0, h1) = own(heads, lane);
                                let len = cb + 1;
                                let mut att = [0f32; 64];
                                for qh in h0..h1 {
                                    let kh = qh / group;
                                    // SAFETY: score buf lane-local; planes read-only post-barrier;
                                    // out slice lane-private.
                                    unsafe {
                                        lfm_attn_qk_bf16(
                                            sh_qf.ptr().add(qh * hd),
                                            sh_k.ptr().add(kbase + kh * cb_n * hd).cast_const(),
                                            att.as_mut_ptr(),
                                            len as i32,
                                            hd as i32,
                                        );
                                        lfm_softmax_scaled_f32(att.as_mut_ptr(), len as i32, scale);
                                        lfm_attn_av_bf16(
                                            att.as_ptr(),
                                            sh_v.ptr().add(kbase + kh * cb_n * hd).cast_const(),
                                            sh_attnf.ptr().add(qh * hd),
                                            len as i32,
                                            hd as i32,
                                        );
                                    }
                                }
                            }
                            fence.wait(lane);
                            {
                                let (d0, d1) = own(dim, lane);
                                if d1 > d0 {
                                    // SAFETY: lane-private cells; attn_f read-only post-barrier.
                                    unsafe {
                                        lfm_f32_to_bf16(
                                            sh_attnf.ptr().add(d0),
                                            sh_attnb.ptr().add(d0),
                                            (d1 - d0) as i32,
                                        )
                                    };
                                }
                            }
                            fence.wait(lane);
                            // ---- out_proj + residual → h ----
                            gemv(&lw.out_w, sh_attnb.ptr(), sh_projf, dim, dim, lane);
                            {
                                let (d0, d1) = own(dim, lane);
                                if d1 > d0 {
                                    // SAFETY: lane-private slices; ladder: rb(out_proj), rb(+x).
                                    unsafe {
                                        lfm_f32_to_bf16(
                                            sh_projf.ptr().add(d0),
                                            sh_yb.ptr().add(d0),
                                            (d1 - d0) as i32,
                                        );
                                        lfm_bf16_add(
                                            sh_yb.ptr().add(d0).cast_const(),
                                            sh_x.ptr().add(d0).cast_const(),
                                            sh_h.ptr().add(d0),
                                            (d1 - d0) as i32,
                                        );
                                    }
                                }
                            }
                            fence.wait(lane);
                            // ---- ffn_norm(h) → xn; w1/w3 → swiglu → t; w2 + residual → x ----
                            this.norm_stage(
                                sh_h.ptr(),
                                lw.ffnnorm,
                                sh_xn,
                                sh_part,
                                dim,
                                lane,
                                lanes,
                                fence,
                            );
                            gemv(&lw.w1, sh_xn.ptr(), sh_projf, this.ff, dim, lane);
                            gemv(&lw.w3, sh_xn.ptr(), sh_uf, this.ff, dim, lane);
                            {
                                let (r0, r1) = own(this.ff, lane);
                                if r1 > r0 {
                                    // SAFETY: lane-private rows.
                                    unsafe {
                                        lfm_swiglu_bf16(
                                            sh_projf.ptr().add(r0).cast_const(),
                                            sh_uf.ptr().add(r0).cast_const(),
                                            sh_tb.ptr().add(r0),
                                            (r1 - r0) as i32,
                                        );
                                    }
                                }
                            }
                            fence.wait(lane);
                            gemv(&lw.w2, sh_tb.ptr(), sh_projf, dim, this.ff, lane);
                            {
                                let (d0, d1) = own(dim, lane);
                                if d1 > d0 {
                                    // SAFETY: lane-private slices; ladder: rb(w2·t), rb(+h).
                                    unsafe {
                                        lfm_f32_to_bf16(
                                            sh_projf.ptr().add(d0),
                                            sh_yb.ptr().add(d0),
                                            (d1 - d0) as i32,
                                        );
                                        lfm_bf16_add(
                                            sh_yb.ptr().add(d0).cast_const(),
                                            sh_h.ptr().add(d0).cast_const(),
                                            sh_x.ptr().add(d0),
                                            (d1 - d0) as i32,
                                        );
                                    }
                                }
                            }
                            fence.wait(lane);
                        }

                        // ---- get_logits: embedding_norm(x) → to_logits GEMV → rb → sample ----
                        let hw = &this.heads_w[cb];
                        this.norm_stage(
                            sh_x.ptr(),
                            hw.norm,
                            sh_xn,
                            sh_part,
                            dim,
                            lane,
                            lanes,
                            fence,
                        );
                        gemv(&hw.logits, sh_xn.ptr(), sh_projf, hw.vocab, dim, lane);
                        {
                            let (r0, r1) = own(hw.vocab, lane);
                            if r1 > r0 {
                                // SAFETY: lane-private rows.
                                unsafe {
                                    lfm_f32_to_bf16(
                                        sh_projf.ptr().add(r0),
                                        sh_log.ptr().add(r0),
                                        (r1 - r0) as i32,
                                    )
                                };
                            }
                        }
                        fence.wait(lane);
                        if let Some(sampler) = sampler_slot.as_mut() {
                            // SAFETY: post-barrier read-only logits view on lane 0.
                            let logits = unsafe {
                                std::slice::from_raw_parts(sh_log.ptr().cast_const(), hw.vocab)
                            };
                            tokens[cb]
                                .store((sampler)(logits), std::sync::atomic::Ordering::Release);
                        }
                        fence.wait(lane);
                        // ---- df_token = embed row of the sampled token ----
                        let tok = tokens[cb].load(std::sync::atomic::Ordering::Acquire) as usize;
                        {
                            let (d0, d1) = own(dim, lane);
                            for i in d0..d1 {
                                // SAFETY: lane-private cells; embed row read-only.
                                unsafe { sh_df.set(i, *hw.emb.u16_ptr().add(tok * dim + i)) };
                            }
                        }
                        fence.wait(lane);
                    }
        };

        // Engine team first: the depthformer rides the SAME lanes as the backbone.
        // Fallback (engine absent): the original rayon + SpinBarrier threadgroup,
        // bit-identical banding, still caged by DISPATCH_LOCK.
        let mut dispatched = false;
        #[cfg(all(
            has_kcoro,
            has_native_engine,
            any(
                all(target_arch = "aarch64", has_flashkern_neon),
                all(target_arch = "x86_64", has_flashkern_x86)
            )
        ))]
        if let Some(engine) = crate::flashkern::native_engine::process_engine() {
            if engine.lanes_total() == lanes {
                // SPIN-ONLY fences under Rust frames (review P1): a lane that PARKS
                // inside a fence lands on kc_sched's global ready queue and can be
                // resumed on a DIFFERENT worker pthread — migrating a live Rust
                // frame (sampler included) across threads, the exact TLS hazard
                // class kcoro patch 0002 fixed for the runtime's own C frames.
                // Native (C++) engine programs are written to tolerate that; Rust
                // programs are not, so they never park mid-frame: the engine
                // provides the dispatch, the stage barrier stays a pure spin —
                // exactly the pre-fold SpinBarrier semantics, on our lanes.
                let barrier = SpinBarrier::new(lanes);
                let barrier = &barrier;
                dispatched = engine.run_lanes(|lane| run_lane(lane, barrier));
            }
        }
        if !dispatched {
            let _dispatch = DISPATCH_LOCK.lock().unwrap();
            let barrier = SpinBarrier::new(lanes);
            let barrier = &barrier;
            let run_lane = &run_lane;
            rayon::scope(|scope| {
                for lane in 0..lanes {
                    scope.spawn(move |_| run_lane(lane, barrier));
                }
            });
        }

        tokens
            .iter()
            .map(|t| t.load(std::sync::atomic::Ordering::Acquire))
            .collect()
    }

    /// RMSNorm stage on the lane team: sumsq partials → same-order fold on every lane →
    /// per-slice apply with ONE bf16 round (the transformer RmsNorm ladder: 1/sqrt(mean+eps)
    /// as sqrt-then-divide, matching `recip(sqrt(z))`).
    #[allow(clippy::too_many_arguments)]
    fn norm_stage(
        &self,
        x: *mut u16,
        w: PtrLen,
        out: Shared<u16>,
        part: Shared<f32>,
        n: usize,
        lane: usize,
        lanes: usize,
        fence: &dyn LaneFence,
    ) {
        let c = n.div_ceil(lanes);
        let (r0, r1) = ((lane * c).min(n), ((lane + 1) * c).min(n));
        // SAFETY: lane-private slice sumsq; partials slot lane-private.
        let p = if r1 > r0 {
            unsafe { lfm_bf16_sumsq_f32(x.add(r0).cast_const(), (r1 - r0) as i32) }
        } else {
            0.0
        };
        unsafe { part.set(lane, p) };
        fence.wait(lane);
        let mut total = 0f32;
        for l in 0..lanes {
            // SAFETY: post-barrier read-only partials.
            total += unsafe { part.get(l) };
        }
        let inv_rms = 1.0f32 / (total / n as f32 + self.eps).sqrt();
        if r1 > r0 {
            // SAFETY: lane-private slices.
            unsafe {
                lfm_bf16_rmsnorm(
                    x.add(r0).cast_const(),
                    w.u16_ptr().add(r0),
                    out.ptr().add(r0),
                    (r1 - r0) as i32,
                    inv_rms,
                );
            }
        }
        fence.wait(lane);
    }

    /// One q/k head: qk RMSNorm over `hd` (per-head, the BoundedAttention ladder) then
    /// interleaved rope at position `pos` — output the rb'd bf16 BITS of the rotated head
    /// (`apply_rotary_emb`'s type_as round; the cache stores exactly these bits, torch's
    /// cache dtype). SAFETY: caller guarantees src/dst head slices are lane-private.
    unsafe fn qk_head_bits(
        &self,
        src: *const u16,
        ln: PtrLen,
        dst_bits: *mut u16,
        hd: usize,
        pos: usize,
    ) {
        let mut normed = [0u16; 128];
        let mut rot = [0f32; 128];
        let ss = lfm_bf16_sumsq_f32(src, hd as i32);
        let inv_rms = 1.0f32 / (ss / hd as f32 + self.eps).sqrt();
        lfm_bf16_rmsnorm(src, ln.u16_ptr(), normed.as_mut_ptr(), hd as i32, inv_rms);
        lfm_bf16_to_f32(normed.as_ptr(), rot.as_mut_ptr(), hd as i32);
        let half = hd / 2;
        lfm_rope_i_f32(
            rot.as_mut_ptr(),
            self.cos.f32_ptr().add(pos * half),
            self.sin.f32_ptr().add(pos * half),
            hd as i32,
        );
        lfm_f32_to_bf16(rot.as_ptr(), dst_bits, hd as i32);
    }
}

/// Backbone decode attention over the RESIDENT bf16 KV planes — the flashkern path for
/// `seq_len == 1` (no mask): per q-head, dot the f32-widened query against the shared
/// kv-head's live K rows (widened in registers), scaled softmax (backbone `hd = 64` ⇒
/// `1/√hd = 0.125`, exact — multiply ≡ divide), then the V gather; ONE bf16 round at the
/// per-head store (the sdpa output `to_dtype` point). K/V never leave checkpoint dtype.
///
/// SAFETY: `k_base`/`v_base` point at the slot planes (`[1, n_kv, cap, hd]` bf16 bits,
/// head stride `cap·hd`); the first `len` rows of each head are live; `q_bits`/`out_bits`
/// are `n_head·hd`. Caller holds the storage borrow for the duration.
#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_decode_bf16(
    q_bits: &[u16],
    k_base: *const u16,
    v_base: *const u16,
    head_stride: usize,
    len: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    out_bits: &mut [u16],
) {
    assert_eq!(
        q_bits.len(),
        n_head * hd,
        "attn_decode: q_bits.len() != n_head·hd"
    );
    assert_eq!(
        out_bits.len(),
        n_head * hd,
        "attn_decode: out_bits.len() != n_head·hd"
    );
    assert!(len > 0 && n_head % n_kv == 0, "attn_decode: bad geometry");
    let group = n_head / n_kv;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut qf = vec![0f32; hd];
    let mut att = vec![0f32; len];
    let mut of = vec![0f32; hd];
    for qh in 0..n_head {
        let kh = qh / group;
        lfm_bf16_to_f32(q_bits.as_ptr().add(qh * hd), qf.as_mut_ptr(), hd as i32);
        lfm_attn_qk_bf16(
            qf.as_ptr(),
            k_base.add(kh * head_stride),
            att.as_mut_ptr(),
            len as i32,
            hd as i32,
        );
        lfm_softmax_scaled_f32(att.as_mut_ptr(), len as i32, scale);
        lfm_attn_av_bf16(
            att.as_ptr(),
            v_base.add(kh * head_stride),
            of.as_mut_ptr(),
            len as i32,
            hd as i32,
        );
        lfm_f32_to_bf16(of.as_ptr(), out_bits.as_mut_ptr().add(qh * hd), hd as i32);
    }
}

/// The ShortConv residual block's weights, zero-copy bf16 bit slices: `norm_w [H]` (the
/// layer's operator norm), `in_w [3H, H]` (in_proj), `conv_w [H, K]` (depthwise taps,
/// squeezed), `out_w [H, H]` (out_proj).
pub struct FusedShortConvWeights<'a> {
    pub norm_w: &'a [u16],
    pub in_w: &'a [u16],
    pub conv_w: &'a [u16],
    pub out_w: &'a [u16],
    pub eps: f32,
    pub k: usize,
}

/// One decode step of the ShortConv residual block — `out = rb(x + out_proj(C ⊙
/// conv1d_causal(B ⊙ x_proj, w, state)))` with `xn = rms_norm(x)·norm_w` and the carried
/// state advanced — as ONE threadgroup dispatch replacing the candle chain (norm, in_proj
/// + transposes, the conv CustomOp, out_proj, residual). The conv itself is the existing
/// `lfm_conv1d_update_bf16` T=1 channel-vectorized kernel, run by lane 0 (it is ~0.1% of
/// the block's work; the GEMVs dominate and they are lane-split).
///
/// `x`/`out` are `[H]` bf16 bits; `state_in`/`state_out` are `[H, K-1]` bf16 bits (the
/// carried Bx window — same contract as the candle op's functional state).
#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
pub fn fused_shortconv_decode(
    x: &[u16],
    w: &FusedShortConvWeights,
    state_in: &[u16],
    state_out: &mut [u16],
    out: &mut [u16],
    lanes: usize,
) {
    let h = x.len();
    let k = w.k;
    assert!(h > 0 && k >= 1 && k <= 8, "fused_shortconv: bad dims");
    assert_eq!(w.norm_w.len(), h, "fused_shortconv: norm_w.len() != H");
    assert_eq!(
        w.in_w.len(),
        3 * h * h,
        "fused_shortconv: in_w.len() != 3H·H"
    );
    assert_eq!(
        w.conv_w.len(),
        h * k,
        "fused_shortconv: conv_w.len() != H·K"
    );
    assert_eq!(w.out_w.len(), h * h, "fused_shortconv: out_w.len() != H·H");
    assert_eq!(
        state_in.len(),
        h * (k - 1),
        "fused_shortconv: state_in.len() != H·(K-1)"
    );
    assert_eq!(
        state_out.len(),
        h * (k - 1),
        "fused_shortconv: state_out.len() != H·(K-1)"
    );
    assert_eq!(out.len(), h, "fused_shortconv: out.len() != H");
    assert!(
        fused_mlp_available(),
        "fused_shortconv_decode requires the flashkern nt kernel"
    );
    let lanes = lanes.clamp(1, h);

    // Threadgroup scratch: normed input, in_proj rows (f32 + rounded bits, [3H] in the
    // conv kernel's B|C|x row order), the conv output [H·K] = [y | new_state], out_proj
    // accumulators + rounded bits.
    let mut partials = vec![0f32; lanes];
    let mut xn = vec![0u16; h];
    let mut bcx_f = vec![0f32; 3 * h];
    let mut bcx_b = vec![0u16; 3 * h];
    let mut conv_out = vec![0u16; h * k];
    let mut proj_f = vec![0f32; h];
    let mut proj_b = vec![0u16; h];
    let sh_part = Shared(partials.as_mut_ptr());
    let sh_xn = Shared(xn.as_mut_ptr());
    let sh_bcxf = Shared(bcx_f.as_mut_ptr());
    let sh_bcxb = Shared(bcx_b.as_mut_ptr());
    let sh_conv = Shared(conv_out.as_mut_ptr());
    let sh_projf = Shared(proj_f.as_mut_ptr());
    let sh_projb = Shared(proj_b.as_mut_ptr());
    let sh_out = Shared(out.as_mut_ptr());
    let sh_state_out = Shared(state_out.as_mut_ptr());
    let _dispatch = DISPATCH_LOCK.lock().unwrap();
    let barrier = SpinBarrier::new(lanes);
    let barrier = &barrier;

    rayon::scope(|scope| {
        for lane in 0..lanes {
            scope.spawn(move |_| {
                let own = |n: usize, l: usize| -> (usize, usize) {
                    let c = n.div_ceil(lanes);
                    ((l * c).min(n), ((l + 1) * c).min(n))
                };
                // stage 1: operator RMSNorm. The reduction runs on lane 0 in CANDLE's
                // exact order (cpu/{neon,avx}.rs vec_sum: 4-register lanes + pairwise tree
                // + horizontal, sequential leftovers) so this block stays TOKEN-EXACT vs
                // the composed op chain — the fused_conv_decode A/B contract. NOT the
                // sequential form: candle's own reduction is vectorized.
                let (r0, r1) = own(h, lane);
                if lane == 0 {
                    // SAFETY: read-only x; slot 0 is lane 0's.
                    unsafe { sh_part.set(0, lfm_bf16_sumsq_candle_f32(x.as_ptr(), h as i32)) };
                }
                barrier.wait();
                // SAFETY: post-barrier read-only.
                let total = unsafe { sh_part.get(0) };
                let inv_rms = 1.0f32 / (total / h as f32 + w.eps).sqrt();
                if r1 > r0 {
                    // SAFETY: lane-private slices.
                    unsafe {
                        lfm_bf16_rmsnorm(
                            x.as_ptr().add(r0),
                            w.norm_w.as_ptr().add(r0),
                            sh_xn.ptr().add(r0),
                            (r1 - r0) as i32,
                            inv_rms,
                        );
                    }
                }
                barrier.wait();

                // stage 2: in_proj — this lane's rows of [3H, H], rounded to bits (the
                // linear_forward ladder). Row order IS the conv kernel's B|C|x chunks.
                let (p0, p1) = own(3 * h, lane);
                if p1 > p0 {
                    // SAFETY: lane-private rows; xn read-only post-barrier.
                    unsafe {
                        nt_rows(
                            sh_xn.ptr(),
                            w.in_w.as_ptr().add(p0 * h),
                            sh_bcxf.ptr().add(p0),
                            p1 - p0,
                            h,
                        );
                        lfm_f32_to_bf16(
                            sh_bcxf.ptr().add(p0),
                            sh_bcxb.ptr().add(p0),
                            (p1 - p0) as i32,
                        );
                    }
                }
                barrier.wait();

                // stage 3: the fused conv update (lane 0; trivially small next to the GEMVs).
                if lane == 0 {
                    // SAFETY: post-barrier read-only bcx; conv_out is exclusively written here.
                    unsafe {
                        conv1d_update_bf16_raw(
                            sh_bcxb.ptr().cast_const(),
                            state_in.as_ptr(),
                            w.conv_w.as_ptr(),
                            sh_conv.ptr(),
                            h,
                            k,
                        );
                    }
                }
                barrier.wait();

                // stage 4: out_proj over the conv's y rows ([H·K] = [y | new_state] per
                // channel: y is column 0 of each channel's K-wide row) — gather y bits,
                // then this lane's out rows + residual. Also copy this lane's slice of the
                // carried state out (columns 1..K of each channel row).
                // y gather: conv_out layout is [H][K] with y at [c][0].
                let (c0, c1) = own(h, lane);
                for c in c0..c1 {
                    // SAFETY: lane-private cells; conv_out read-only post-barrier.
                    unsafe {
                        sh_projb.set(c, sh_conv.get(c * k));
                        for j in 0..k - 1 {
                            sh_state_out.set(c * (k - 1) + j, sh_conv.get(c * k + 1 + j));
                        }
                    }
                }
                barrier.wait();
                if r1 > r0 {
                    // SAFETY: lane-private rows; y bits read-only post-barrier.
                    unsafe {
                        nt_rows(
                            sh_projb.ptr().cast_const(),
                            w.out_w.as_ptr().add(r0 * h),
                            sh_projf.ptr().add(r0),
                            r1 - r0,
                            h,
                        );
                        let mut yb = [0u16; 0];
                        let _ = &mut yb;
                        // rb(out_proj) then rb(+residual), slice-wide (reuse xn as staging).
                        lfm_f32_to_bf16(
                            sh_projf.ptr().add(r0),
                            sh_xn.ptr().add(r0),
                            (r1 - r0) as i32,
                        );
                        lfm_bf16_add(
                            sh_xn.ptr().add(r0).cast_const(),
                            x.as_ptr().add(r0),
                            sh_out.ptr().add(r0),
                            (r1 - r0) as i32,
                        );
                    }
                }
            });
        }
    });
}

// Raw single-step call into the existing fused conv kernel: bcx is [1, 3H, 1] (== the
// contiguous [3H] plane in B|C|x row order), state [1, H, K-1], w [H, K], out [1, H, K].
// SAFETY: caller guarantees the plane sizes and availability.
#[cfg(any(
    all(target_arch = "aarch64", has_flashkern_neon),
    all(target_arch = "x86_64", has_flashkern_x86)
))]
unsafe fn conv1d_update_bf16_raw(
    bcx: *const u16,
    state: *const u16,
    w: *const u16,
    out: *mut u16,
    h: usize,
    k: usize,
) {
    #[cfg(all(target_arch = "aarch64", has_flashkern_neon))]
    super::neon::conv1d_update_bf16_ptr(bcx, state, w, out, h, k);
    #[cfg(all(target_arch = "x86_64", has_flashkern_x86))]
    super::x86::conv1d_update_bf16_ptr(bcx, state, w, out, h, k);
}
