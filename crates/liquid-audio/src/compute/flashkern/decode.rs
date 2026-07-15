//! Fused decode blocks on the GPU dispatch model — the point of the exercise.
//!
//! The per-op execution the model otherwise uses on CPU (candle op → scheduler fork/join →
//! tensor alloc → bf16↔f32 cast, ×~240 ops per decode token) is exactly what a GPU never
//! does: a GPU enters ONE dispatch per fused region and the data flows through threadgroup
//! memory between barrier-fenced stages. This module runs the decode step that way on CPU:
//!
//! | Metal                       | native engine |
//! |-----------------------------|---------------|
//! | `dispatch_thread_groups`    | one doorbell to the resident fixed lane team |
//! | simdgroup lanes             | exactly `lanes` native workers |
//! | `threadgroup float* shared` | the activation scratch the lanes co-own |
//! | `threadgroup_barrier`       | the engine's generation fence |
//!
//! Numerics: activations round through bf16 at EXACTLY the points the candle op chain
//! rounds (linear outputs, silu, gating mul, the norm's single output round, the residual
//! add) — the fused block changes *where the time goes*, not the trained-regime arithmetic.
//! Weights are read zero-copy in their checkpoint-native `[N,K]` layout via the nt dot
//! kernel; each lane owns a contiguous slice of output rows.

#[cfg(test)]
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

// SAFETY: reference lanes write disjoint indices between real blocking barriers.
#[cfg(test)]
unsafe impl<T> Send for Shared<T> {}
#[cfg(test)]
unsafe impl<T> Sync for Shared<T> {}

#[cfg(test)]
impl<T: Copy> Shared<T> {
    #[inline]
    fn ptr(self) -> *mut T {
        self.0
    }

    #[inline]
    unsafe fn get(self, index: usize) -> T {
        *self.0.add(index)
    }

    #[inline]
    unsafe fn set(self, index: usize, value: T) {
        *self.0.add(index) = value;
    }
}

/// `true` when the fused decode blocks can run through the native-layout kernel.
pub fn fused_mlp_available() -> bool {
    crate::bf16_gemm::bf16_gemm_nt_available()
}

#[inline]
#[cfg(test)]
pub(crate) fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16 bits (the storage rounding the candle op chain applies
/// after every op on the bf16 CPU path).
#[inline]
#[cfg(test)]
pub(crate) fn rb_bits(f: f32) -> u16 {
    let u = f.to_bits();
    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

// The nt dot kernel over a lane's row range, arch-dispatched.
// SAFETY: caller guarantees a=[K], w=n·K rows at `w`, c=n f32 at `c`, availability checked.
#[allow(unused_variables)]
#[cfg(test)]
pub(crate) unsafe fn nt_rows(a: *const u16, w: *const u16, c: *mut f32, n: usize, k: usize) {
    #[cfg(target_arch = "aarch64")]
    super::neon::bf16_gemm_nt_raw(a, w, c, n, k);
    #[cfg(target_arch = "x86_64")]
    super::x86::bf16_gemm_nt_raw(a, w, c, n, k);
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

/// Test-only parity oracle for one FFN residual block. Production inference enters the
/// resident C++ engine and never schedules this Rust lane program.
///
/// `x`/`out` are `[H]` bf16 bits. Stage map (each lane grid-strides or owns row slices):
/// 1. Σx² partials → all lanes reduce the `lanes` partials serially (same order — deterministic)
///    → `xn = rb(x · rsqrt(mean+eps) · w_norm)` (the real `RmsNorm::forward`'s single round).
/// 2. Lane's gate/up rows: `t[r] = rb(rb(silu(rb(g_r))) · rb(u_r))` — the op chain's rounds.
/// 3. Lane's down rows + residual: `out[r] = rb(rb(y_r) + x[r])`.
#[cfg(test)]
pub(crate) fn fused_mlp_reference(x: &[u16], w: &FusedMlpWeights, out: &mut [u16], lanes: usize) {
    let h = x.len();
    let i = w.w1.len() / h;
    assert!(h > 0 && i > 0, "fused_mlp_reference: empty dims");
    assert_eq!(w.norm_w.len(), h, "fused_mlp_reference: norm_w.len() != H");
    assert_eq!(w.w1.len(), i * h, "fused_mlp_reference: w1.len() != I·H");
    assert_eq!(w.w3.len(), i * h, "fused_mlp_reference: w3.len() != I·H");
    assert_eq!(w.w2.len(), h * i, "fused_mlp_reference: w2.len() != H·I");
    assert_eq!(out.len(), h, "fused_mlp_reference: out.len() != H");
    assert!(
        fused_mlp_available(),
        "fused_mlp_reference requires the flashkern nt kernel (gate on fused_mlp_available())"
    );
    let lanes = lanes.clamp(1, h.min(i));

    // Threadgroup scratch: sumsq partials, normed activation (bf16), gate/up dots (f32),
    // gated intermediate (bf16). Lane-disjoint writes, barrier-fenced reads.
    let mut partials = vec![0f32; lanes];
    let mut xn = vec![0u16; h];
    let mut gu = vec![0f32; 2 * i]; // g = gu[0..i], u = gu[i..2i]
    let mut t = vec![0u16; i];
    let sh_part = Shared(partials.as_mut_ptr());
    let sh_xn = Shared(xn.as_mut_ptr());
    let sh_gu = Shared(gu.as_mut_ptr());
    let sh_t = Shared(t.as_mut_ptr());
    let sh_out = Shared(out.as_mut_ptr());
    let barrier = std::sync::Barrier::new(lanes);
    let barrier = &barrier;

    // Row ownership: contiguous slices so each lane's nt call streams contiguous weight rows.
    let i_chunk = i.div_ceil(lanes);
    let h_chunk = h.div_ceil(lanes);

    // This is the portable parity/reference path, not the resident inference path. Give every
    // barrier participant a real scoped thread and park at stage boundaries. A work-stealing
    // pool cannot guarantee that all participants run concurrently and can deadlock when its
    // first workers wait for tasks that are still queued.
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            scope.spawn(move || {
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
            fused_mlp_reference(&x, &w, &mut got, lanes);
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
        fused_mlp_reference(&x, &w, &mut a, 4);
        fused_mlp_reference(&x, &w, &mut b, 4);
        assert_eq!(a, b, "same dispatch shape must be bit-identical");
    }
}

// ======================================================================================
// Typed native Depthformer descriptors — Candle stripped from the audio-frame hot loop.
//
// Profiling showed `sample_audio_frame` (8 sequential codebook steps × 6 StandardBlocks,
// per audio frame) dominating decode, every op a Candle dispatch. Rust now installs only
// immutable pointer descriptors; C++ owns the frame program, scratch, KV planes, zero-spin
// generation fences, and integrated sampler under one typed kcoro ticket.
// ======================================================================================

extern "C" {
    #[cfg(test)]
    fn lfm_bf16_sumsq_candle_f32(x: *const u16, n: i32) -> f32;
    #[cfg(test)]
    fn lfm_bf16_rmsnorm(x: *const u16, w: *const u16, out: *mut u16, n: i32, inv_rms: f32);
    #[cfg(test)]
    fn lfm_bf16_add(a: *const u16, b: *const u16, out: *mut u16, n: i32);
    fn lfm_softmax_scaled_f32(x: *mut f32, n: i32, scale: f32);
    fn lfm_attn_qk_bf16(q: *const f32, k: *const u16, att: *mut f32, len: i32, hd: i32);
    fn lfm_attn_av_bf16(att: *const f32, v: *const u16, out: *mut f32, len: i32, hd: i32);
    fn lfm_bf16_to_f32(x: *const u16, out: *mut f32, n: i32);
    fn lfm_f32_to_bf16(x: *const f32, out: *mut u16, n: i32);
}

/// A raw view of a checkpoint tensor's storage, stored as `usize` so the ctx stays `Send`.
/// SAFETY CONTRACT: the owning model outlives the ctx (both live in `LFM2AudioModel`), and
/// candle storages are `Arc`-heap — moves of the model don't move the data.
#[repr(C)]
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

    #[cfg(test)]
    pub(crate) fn from_u16(values: &[u16]) -> Self {
        Self {
            ptr: values.as_ptr() as usize,
            len: values.len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_f32(values: &[f32]) -> Self {
        Self {
            ptr: values.as_ptr() as usize,
            len: values.len(),
        }
    }
}

/// One depthformer StandardBlock's zero-copy weight descriptors. Layout mirrors
/// `LfmDepthLayerV1`; the native build copies these descriptors, never payloads.
#[repr(C)]
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
#[repr(C)]
pub struct DepthHead {
    pub emb: PtrLen,    // [vocab, dim]
    pub norm: PtrLen,   // [dim]
    pub logits: PtrLen, // [vocab, dim]
    pub vocab: usize,
}

#[repr(C)]
pub(crate) struct DepthPlan {
    pub(crate) size: u32,
    pub(crate) abi_version: u32,
    pub(crate) dim: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) ffn_dim: u32,
    pub(crate) codebooks: u32,
    pub(crate) backbone_dim: u32,
    pub(crate) eps: f32,
    pub(crate) depth_linear_w: PtrLen,
    pub(crate) depth_linear_b: PtrLen,
    pub(crate) rope_cos: PtrLen,
    pub(crate) rope_sin: PtrLen,
    pub(crate) layers: *const DepthLayer,
    pub(crate) layer_count: usize,
    pub(crate) codebook_heads: *const DepthHead,
    pub(crate) codebook_head_count: usize,
}

const _: [(); 16] = [(); std::mem::size_of::<PtrLen>()];
const _: [(); 144] = [(); std::mem::size_of::<DepthLayer>()];
const _: [(); 56] = [(); std::mem::size_of::<DepthHead>()];
const _: [(); 136] = [(); std::mem::size_of::<DepthPlan>()];

/// Resident typed Depthformer plan. C++ owns its mutable planes and copied
/// descriptors; the model keeps the immutable checkpoint storage alive.
pub struct DepthDecode {
    id: u64,
    backbone_dim: usize,
    codebooks: usize,
}

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
    ) -> Option<Self> {
        const ABI: u32 = 1;
        if heads == 0 || dim % heads != 0 || heads_w.len() != codebooks {
            return None;
        }
        let plan = DepthPlan {
            size: std::mem::size_of::<DepthPlan>() as u32,
            abi_version: ABI,
            dim: dim.try_into().ok()?,
            heads: heads.try_into().ok()?,
            kv_heads: kv_heads.try_into().ok()?,
            head_dim: (dim / heads).try_into().ok()?,
            ffn_dim: ff.try_into().ok()?,
            codebooks: codebooks.try_into().ok()?,
            backbone_dim: backbone_dim.try_into().ok()?,
            eps,
            depth_linear_w: depth_lin_w,
            depth_linear_b: depth_lin_b,
            rope_cos: cos,
            rope_sin: sin,
            layers: layers.as_ptr(),
            layer_count: layers.len(),
            codebook_heads: heads_w.as_ptr(),
            codebook_head_count: heads_w.len(),
        };
        crate::flashkern::native_engine::process_engine()
            .depth_build(&plan)
            .map(|id| Self {
                id,
                backbone_dim,
                codebooks,
            })
    }

    /// One typed native frame ticket. Hidden, sampler state, and the fixed token
    /// result span remain borrowed until the exact kcoro completion resolves.
    pub(crate) fn frame(
        &self,
        emb_bits: &[u16],
        config: &crate::flashkern::native_engine::SampleConfig,
        state: &mut crate::flashkern::native_engine::PrngState,
    ) -> Vec<u32> {
        assert_eq!(
            emb_bits.len(),
            self.backbone_dim,
            "depth frame: bad hidden size"
        );
        let mut tokens = [u32::MAX; 64];
        let out = &mut tokens[..self.codebooks];
        assert!(
            crate::flashkern::native_engine::process_engine()
                .depth_frame(self.id, emb_bits, config, state, out),
            "DepthDecode::frame: native typed pass rejected"
        );
        out.to_vec()
    }
}

impl Drop for DepthDecode {
    fn drop(&mut self) {
        crate::flashkern::native_engine::process_engine().depth_clear(self.id);
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
#[cfg(test)]
pub(crate) struct FusedShortConvWeights<'a> {
    pub norm_w: &'a [u16],
    pub in_w: &'a [u16],
    pub conv_w: &'a [u16],
    pub out_w: &'a [u16],
    pub eps: f32,
    pub k: usize,
}

/// Test-only parity oracle for one ShortConv residual block — `out = rb(x + out_proj(C ⊙
/// conv1d_causal(B ⊙ x_proj, w, state)))` with `xn = rms_norm(x)·norm_w` and the carried
/// state advanced — as ONE threadgroup dispatch replacing the candle chain (norm, in_proj
/// + transposes, the conv CustomOp, out_proj, residual). The conv itself is the existing
/// `lfm_conv1d_update_bf16` T=1 channel-vectorized kernel, run by lane 0 (it is ~0.1% of
/// the block's work; the GEMVs dominate and they are lane-split).
///
/// `x`/`out` are `[H]` bf16 bits; `state_in`/`state_out` are `[H, K-1]` bf16 bits (the
/// carried Bx window — same contract as the candle op's functional state).
#[cfg(test)]
pub(crate) fn fused_shortconv_reference(
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
        "fused_shortconv_reference requires the flashkern nt kernel"
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
    let barrier = std::sync::Barrier::new(lanes);
    let barrier = &barrier;

    // Portable parity/reference path. Scoped threads plus a blocking barrier preserve the
    // requested lane partition without depending on the capacity of Rayon's global pool.
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            scope.spawn(move || {
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
#[cfg(test)]
unsafe fn conv1d_update_bf16_raw(
    bcx: *const u16,
    state: *const u16,
    w: *const u16,
    out: *mut u16,
    h: usize,
    k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    super::neon::conv1d_update_bf16_ptr(bcx, state, w, out, h, k);
    #[cfg(target_arch = "x86_64")]
    super::x86::conv1d_update_bf16_ptr(bcx, state, w, out, h, k);
}
