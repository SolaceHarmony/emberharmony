//! GPU dispatch model, simulated on the CPU. flashkern's single-stage ops (GEMM, reductions)
//! only need SIMD lanes + a rayon fan-out. The *multi-stage* Metal kernels — `FFTConv.metal`'s
//! `fft_radix2`/`fft_conv` (a `threadgroup_barrier` between every stage), the double-double
//! `fft_conv_dd`, and `fused_monarch`'s forward and full-convolution pipelines (barriers +
//! ping-pong through `threadgroup` memory) — need the full model. This module reproduces it
//! faithfully:
//!
//! | Metal                                   | here |
//! |-----------------------------------------|------|
//! | `dispatch_thread_groups(grid=(B,1,1))`  | rayon fan-out, one task per threadgroup |
//! | one threadgroup = one simdgroup (lanes) | a scoped **thread team** of `lanes` workers |
//! | `threadgroup float* shared`             | a shared `&mut [f32]` the lanes co-own |
//! | `thread_position_in_threadgroup` (tid)  | the lane index; `for i=lane; i<N; i+=lanes` grid-stride |
//! | `threadgroup_barrier(mem_threadgroup)`  | `std::sync::Barrier` — a **real** barrier, not a sequence point |
//!
//! The barriers are genuine: the lanes run concurrently, write **disjoint** shared indices in a
//! stage, and the `Barrier` (with its Acquire/Release fence) guarantees a stage's writes are
//! visible before the next stage reads them — the exact contract of `threadgroup_barrier`.

use super::dd::{self, CDd, Dd};
use std::sync::Barrier;

/// A `*mut T` shared across the lane team — the CPU analog of `threadgroup` memory. Lanes
/// write disjoint indices within a barrier-delimited stage, so aliasing never occurs; the
/// `Barrier` between stages provides the fence that makes writes visible (as on the GPU).
/// (`T` is `f32` for the f32 kernels, [`CDd`] for the double-double ones.)
pub(crate) struct Shared<T>(pub(crate) *mut T);
// SAFETY: sharing is sound because (a) within a stage every lane touches a disjoint index set
// and (b) `Barrier` fences separate stages, so there is never a concurrent read+write or
// write+write to the same location — identical to GPU threadgroup-memory discipline.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}

impl<T: Copy> Shared<T> {
    /// The raw plane pointer. Use THIS in lane closures, not the `.0` field: Rust 2021
    /// precise capture would otherwise capture the bare `*mut` field (not `Shared`) and
    /// lose the Send/Sync impls this wrapper exists to carry.
    #[inline]
    pub(crate) fn ptr(self) -> *mut T {
        self.0
    }
    #[inline]
    pub(crate) unsafe fn get(self, i: usize) -> T {
        *self.0.add(i)
    }
    #[inline]
    pub(crate) unsafe fn set(self, i: usize, v: T) {
        *self.0.add(i) = v;
    }
}

// The per-lane body of the radix-2 FFT over interleaved [re,im] threadgroup memory — the CPU
// port of `fft_radix2` in FFTConv.metal (bit-reverse + butterfly stages), callable from inside
// a larger lane team (the fused conv kernels) as well as from [`fft_threadgroup`]. Every
// `barrier.wait()` is a `threadgroup_barrier`; `sign` is the twiddle sign (−1 forward, +1 for
// the sign-flipped inverse form [`fused_fft`] uses).
fn fft_lane(shared: Shared<f32>, n: usize, sign: f32, lane: usize, lanes: usize, barrier: &Barrier) {
    let log2n = n.trailing_zeros();
    // Bit-reverse permutation (grid-stride over lanes; each pair swapped once by the
    // lower-index lane — disjoint, exactly as the Metal `if (tid < rev)` guard).
    let mut i = lane;
    while i < n {
        let mut rev = 0usize;
        for bit in 0..log2n {
            rev |= ((i >> bit) & 1) << (log2n - 1 - bit);
        }
        if i < rev {
            // SAFETY: only lane `i` touches the (i,rev) pair; no other lane aliases it.
            unsafe {
                let (r0, i0, r1, i1) = (
                    shared.get(2 * i),
                    shared.get(2 * i + 1),
                    shared.get(2 * rev),
                    shared.get(2 * rev + 1),
                );
                shared.set(2 * i, r1);
                shared.set(2 * i + 1, i1);
                shared.set(2 * rev, r0);
                shared.set(2 * rev + 1, i0);
            }
        }
        i += lanes;
    }
    barrier.wait(); // threadgroup_barrier — bit-reverse visible before the butterflies

    // Butterfly stages. Within a stage each of the n/2 butterflies owns a disjoint
    // (a,b) index pair, so lanes stride over them freely; the barrier fences stages.
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = sign * 2.0 * std::f32::consts::PI / (len as f32);
        let mut bf = lane;
        while bf < n / 2 {
            let g = bf / half; // butterfly group
            let j = bf % half; // position in group
            let a = g * len + j;
            let b = a + half;
            let (wr, wi) = ((ang * j as f32).cos(), (ang * j as f32).sin());
            // SAFETY: (a,b) is this butterfly's private pair for this stage.
            unsafe {
                let (xr, xi) = (shared.get(2 * b), shared.get(2 * b + 1));
                let tr = wr * xr - wi * xi;
                let ti = wr * xi + wi * xr;
                let (ur, ui) = (shared.get(2 * a), shared.get(2 * a + 1));
                shared.set(2 * a, ur + tr);
                shared.set(2 * a + 1, ui + ti);
                shared.set(2 * b, ur - tr);
                shared.set(2 * b + 1, ui - ti);
            }
            bf += lanes;
        }
        barrier.wait(); // threadgroup_barrier between butterfly stages
        len <<= 1;
    }
}

// The per-lane inverse FFT in FFTConv.metal's exact form (`ifft_radix2`): conjugate the data,
// run the FORWARD butterflies, conjugate + scale by 1/n. (Mathematically the same as the
// sign-flipped form; kept op-for-op so the fused conv reproduces the Metal kernel's roundings.)
fn ifft_lane(shared: Shared<f32>, n: usize, lane: usize, lanes: usize, barrier: &Barrier) {
    let mut i = lane;
    while i < n {
        // SAFETY: each lane conjugates only its own grid-stride cells.
        unsafe { shared.set(2 * i + 1, -shared.get(2 * i + 1)) };
        i += lanes;
    }
    barrier.wait(); // threadgroup_barrier — conjugation visible before the forward FFT

    fft_lane(shared, n, -1.0, lane, lanes, barrier);

    let scale = 1.0f32 / n as f32;
    let mut i = lane;
    while i < n {
        // SAFETY: each lane scales only its own grid-stride cells.
        unsafe {
            shared.set(2 * i, shared.get(2 * i) * scale);
            shared.set(2 * i + 1, -shared.get(2 * i + 1) * scale);
        }
        i += lanes;
    }
    barrier.wait(); // threadgroup_barrier closing ifft_radix2
}

// One threadgroup: run the whole radix-2 FFT of `data` (interleaved [re,im], length 2*n) with a
// team of `lanes` workers synchronising at `threadgroup_barrier`s between every stage — the
// standalone-dispatch wrapper over [`fft_lane`]. `inverse` uses the sign-flipped twiddles +
// 1/n scale (equivalent to, but not op-for-op, FFTConv.metal's conjugate-form `ifft_radix2`,
// which the fused conv path uses via [`ifft_lane`]).
fn fft_threadgroup(data: &mut [f32], n: usize, inverse: bool, lanes: usize) {
    debug_assert_eq!(data.len(), 2 * n);
    if n <= 1 {
        return;
    }
    let shared = Shared(data.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
                let sign = if inverse { 1.0f32 } else { -1.0f32 };
                fft_lane(shared, n, sign, lane, lanes, barrier);
                if inverse {
                    let inv = 1.0f32 / n as f32;
                    let mut k = lane;
                    while k < 2 * n {
                        // SAFETY: each element scaled by exactly one lane.
                        unsafe { shared.set(k, shared.get(k) * inv) };
                        k += lanes;
                    }
                    barrier.wait();
                }
            });
        }
    });
}

/// Fused batched radix-2 FFT with the full GPU dispatch model: `data` is `batch` interleaved
/// `[re,im]` signals of length `2*n` (n a power of two) laid end to end. Fans out over the batch
/// grid with rayon (one **threadgroup** per signal), each threadgroup running the staged FFT on a
/// team of `lanes` workers synchronised by `threadgroup_barrier`s. `inverse` scales by `1/n`.
///
/// `lanes` is the simulated simdgroup width (Metal uses 32); 1 collapses to the serial kernel.
pub fn fused_fft(data: &mut [f32], batch: usize, inverse: bool, lanes: usize) {
    use rayon::prelude::*;
    assert!(batch > 0 && data.len() % batch == 0, "data.len() must be batch·2n");
    let per = data.len() / batch;
    let n = per / 2;
    assert!(per % 2 == 0 && (n == 0 || n.is_power_of_two()), "each signal must be 2·(power of two)");
    let lanes = lanes.clamp(1, n.max(1));
    // Grid dispatch: one rayon task = one threadgroup (one (batch) index).
    data.par_chunks_mut(per).for_each(|sig| {
        fft_threadgroup(sig, n, inverse, lanes);
    });
}

// One threadgroup: the complete FFT convolution for one (batch, channel) pair — a line-for-line
// CPU port of `fft_conv` in FFTConv.metal: load+zero-pad into threadgroup memory, forward FFT,
// half-spectrum × k_f (fixed-order `cmul`, no FMA), Hermitian mirror of the UPDATED first half,
// conjugate-form inverse FFT, then `y = irfft + u·D`. Every `barrier.wait()` is one of the
// kernel's `threadgroup_barrier`s.
fn fft_conv_threadgroup(
    u_bc: &[f32],
    kf_c: &[f32],
    d_c: f32,
    y_bc: &mut [f32],
    seqlen: usize,
    fft_size: usize,
    lanes: usize,
) {
    let half_sz = fft_size / 2 + 1;
    // threadgroup Complex* shared — interleaved [re,im], fft_size elements.
    let mut shared = vec![0f32; 2 * fft_size];
    let sh = Shared(shared.as_mut_ptr());
    let y_sh = Shared(y_bc.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
                // Load input signal (real FFT: zero imaginary), zero-pad to fft_size.
                let mut i = lane;
                while i < fft_size {
                    // SAFETY: cell i is this lane's private grid-stride slot.
                    unsafe {
                        sh.set(2 * i, if i < seqlen { u_bc[i] } else { 0.0 });
                        sh.set(2 * i + 1, 0.0);
                    }
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — staging visible before the FFT

                // Forward FFT.
                fft_lane(sh, fft_size, -1.0, lane, lanes, barrier);

                // Complex multiply with the kernel spectrum (first half + Nyquist only) —
                // the Metal `cmul`'s fixed evaluation order, deliberately no FMA.
                let mut i = lane;
                while i < half_sz {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let (zr, zi) = (sh.get(2 * i), sh.get(2 * i + 1));
                        let (kr, ki) = (kf_c[2 * i], kf_c[2 * i + 1]);
                        sh.set(2 * i, (zr * kr) - (zi * ki));
                        sh.set(2 * i + 1, (zr * ki) + (zi * kr));
                    }
                    i += lanes;
                }
                barrier.wait(); // first-half writes visible before mirroring

                // Mirror Hermitian symmetry into the second half (reads the already-updated
                // first half — disjoint from this stage's writes, which are all ≥ half_sz).
                let mut i = half_sz + lane;
                while i < fft_size {
                    let mirror = fft_size - i; // in 1..half-1
                    // SAFETY: reads first half (read-only this stage), writes own cell i.
                    unsafe {
                        sh.set(2 * i, sh.get(2 * mirror));
                        sh.set(2 * i + 1, -sh.get(2 * mirror + 1));
                    }
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — full spectrum visible before the iFFT

                // Inverse FFT (Metal's conjugate form).
                ifft_lane(sh, fft_size, lane, lanes, barrier);

                // Write output: truncate to seqlen, add the u·D skip.
                let mut t = lane;
                while t < seqlen {
                    // SAFETY: output cell t is this lane's private grid-stride slot.
                    unsafe { y_sh.set(t, sh.get(2 * t) + u_bc[t] * d_c) };
                    t += lanes;
                }
            });
        }
    });
}

/// Fused FFT convolution with the full GPU dispatch model — the CPU port of `fft_conv` in
/// FFTConv.metal (the FlashFFTConv product kernel): `y = irfft(rfft(u) ⊙ k_f) + u·D`, all of it
/// inside one threadgroup per `(batch, channel)` pair.
///
/// * `u` — `[batch, channels, seqlen]` real input.
/// * `k_f` — `[channels, fft_size/2+1, 2]` interleaved pre-computed half-spectrum.
/// * `d` — `[channels]` per-channel skip term.
/// * `y` — `[batch, channels, seqlen]` output.
///
/// `fft_size` must be a power of two ≥ `seqlen` (typically `2·seqlen` for linear conv). The
/// Metal 1024-thread threadgroup ceiling does not apply here — lanes grid-stride.
/// `lanes` is the simulated simdgroup width; 1 collapses to the serial kernel.
#[allow(clippy::too_many_arguments)]
pub fn fused_fft_conv(
    u: &[f32],
    k_f: &[f32],
    d: &[f32],
    y: &mut [f32],
    batch: usize,
    channels: usize,
    seqlen: usize,
    fft_size: usize,
    lanes: usize,
) {
    use rayon::prelude::*;
    let half_sz = fft_size / 2 + 1;
    assert!(batch > 0 && channels > 0 && seqlen > 0, "fused_fft_conv: empty dims");
    assert!(fft_size.is_power_of_two(), "fused_fft_conv: fft_size must be a power of two");
    assert!(fft_size >= seqlen, "fused_fft_conv: fft_size {fft_size} < seqlen {seqlen}");
    assert_eq!(u.len(), batch * channels * seqlen, "fused_fft_conv: u.len() != B·C·seqlen");
    assert_eq!(k_f.len(), channels * half_sz * 2, "fused_fft_conv: k_f.len() != C·(fft/2+1)·2");
    assert_eq!(d.len(), channels, "fused_fft_conv: d.len() != C");
    assert_eq!(y.len(), u.len(), "fused_fft_conv: y.len() != u.len()");
    let lanes = lanes.clamp(1, fft_size);
    // Grid dispatch: one rayon task == one threadgroup == one (b,c) index.
    y.par_chunks_mut(seqlen)
        .zip(u.par_chunks(seqlen))
        .enumerate()
        .for_each(|(bc, (y_bc, u_bc))| {
            let c = bc % channels;
            let kf_c = &k_f[c * half_sz * 2..(c + 1) * half_sz * 2];
            fft_conv_threadgroup(u_bc, kf_c, d[c], y_bc, seqlen, fft_size, lanes);
        });
}

/// Standalone real-input FFT — the CPU port of `rfft_kernel` in FFTConv.metal. `input` is
/// `seqlen` reals (zero-padded to `n`); `out` receives the `n/2+1` interleaved complex bins.
pub fn rfft(input: &[f32], out: &mut [f32], seqlen: usize, n: usize, lanes: usize) {
    assert!(n.is_power_of_two() && n >= seqlen, "rfft: n must be a power of two ≥ seqlen");
    assert_eq!(input.len(), seqlen, "rfft: input.len() != seqlen");
    let half_sz = n / 2 + 1;
    assert_eq!(out.len(), half_sz * 2, "rfft: out.len() != (n/2+1)·2");
    let lanes = lanes.clamp(1, n);
    let mut shared = vec![0f32; 2 * n];
    for (i, &v) in input.iter().enumerate() {
        shared[2 * i] = v;
    }
    fft_threadgroup(&mut shared, n, false, lanes);
    out.copy_from_slice(&shared[..half_sz * 2]);
}

/// Standalone inverse real FFT — the CPU port of `irfft_kernel` in FFTConv.metal. `input` is
/// the `n/2+1` interleaved complex half-spectrum; the second half is mirrored from the INPUT
/// (Hermitian), the conjugate-form inverse FFT runs, and `out` gets the first `seqlen` reals.
pub fn irfft(input: &[f32], out: &mut [f32], n: usize, seqlen: usize, lanes: usize) {
    let half_sz = n / 2 + 1;
    assert!(n.is_power_of_two() && seqlen <= n, "irfft: n must be a power of two ≥ seqlen");
    assert_eq!(input.len(), half_sz * 2, "irfft: input.len() != (n/2+1)·2");
    assert_eq!(out.len(), seqlen, "irfft: out.len() != seqlen");
    let lanes = lanes.clamp(1, n);
    let mut shared = vec![0f32; 2 * n];
    shared[..half_sz * 2].copy_from_slice(input);
    for i in half_sz..n {
        let mirror = n - i; // in 1..half-1
        shared[2 * i] = input[2 * mirror];
        shared[2 * i + 1] = -input[2 * mirror + 1];
    }
    // Conjugate-form inverse: conjugate, forward FFT, conjugate + 1/n (as ifft_radix2 does).
    let sh = Shared(shared.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || ifft_lane(sh, n, lane, lanes, barrier));
        }
    });
    for (t, o) in out.iter_mut().enumerate() {
        *o = shared[2 * t];
    }
}

// The per-lane radix-2 FFT in DOUBLE-DOUBLE — the CPU port of `fft_radix2_dd` in
// FFTConvDd.metal. Same butterfly structure as [`fft_lane`], but the state is [`CDd`]
// threadgroup memory, the arithmetic is the dd toolkit's (`cdd_mul`/`cdd_add`/`cdd_sub`),
// and the twiddles come from the host-precomputed f64→dd table (`tw[j] = exp(−2πi·j/n)`,
// j < n/2) instead of in-kernel f32 cos/sin — index `k·(n >> (stage+1))` == `k·(n/len)`.
fn fft_dd_lane(shared: Shared<CDd>, tw: &[CDd], n: usize, lane: usize, lanes: usize, barrier: &Barrier) {
    let log2n = n.trailing_zeros();
    let mut i = lane;
    while i < n {
        let mut rev = 0usize;
        for bit in 0..log2n {
            rev |= ((i >> bit) & 1) << (log2n - 1 - bit);
        }
        if i < rev {
            // SAFETY: only lane `i` touches the (i,rev) pair; no other lane aliases it.
            unsafe {
                let (a, b) = (shared.get(i), shared.get(rev));
                shared.set(i, b);
                shared.set(rev, a);
            }
        }
        i += lanes;
    }
    barrier.wait(); // threadgroup_barrier — bit-reverse visible before the butterflies

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let mut bf = lane;
        while bf < n / 2 {
            let g = bf / half;
            let j = bf % half;
            let a = g * len + j;
            let b = a + half;
            // twiddle(j, len) = exp(−2πi·j/len) = tw[j·(n/len)].
            let w = tw[j * (n / len)];
            // SAFETY: (a,b) is this butterfly's private pair for this stage.
            unsafe {
                let t = dd::cdd_mul(w, shared.get(b));
                let u = shared.get(a);
                shared.set(a, dd::cdd_add(u, t));
                shared.set(b, dd::cdd_sub(u, t));
            }
            bf += lanes;
        }
        barrier.wait(); // threadgroup_barrier between butterfly stages
        len <<= 1;
    }
}

// The per-lane dd inverse FFT — `ifft_radix2_dd`: conjugate, forward dd FFT, conjugate +
// dd-scale by 1/n (exact in f32: n is a power of two).
fn ifft_dd_lane(shared: Shared<CDd>, tw: &[CDd], n: usize, lane: usize, lanes: usize, barrier: &Barrier) {
    let mut i = lane;
    while i < n {
        // SAFETY: each lane conjugates only its own grid-stride cells.
        unsafe {
            let z = shared.get(i);
            shared.set(i, CDd::new(z.re, dd::dd_neg(z.im)));
        }
        i += lanes;
    }
    barrier.wait();

    fft_dd_lane(shared, tw, n, lane, lanes, barrier);

    let scale = Dd::from_f32(1.0 / n as f32);
    let mut i = lane;
    while i < n {
        // SAFETY: each lane scales only its own grid-stride cells.
        unsafe {
            let z = shared.get(i);
            shared.set(
                i,
                CDd::new(dd::dd_mul(z.re, scale), dd::dd_neg(dd::dd_mul(z.im, scale))),
            );
        }
        i += lanes;
    }
    barrier.wait();
}

// One threadgroup: the double-double FFT convolution for one (batch, channel) — the CPU port
// of `fft_conv_dd` in FFTConvDd.metal. Identical staging to [`fft_conv_threadgroup`], but the
// whole rfft → ⊙k_f → irfft pipeline runs in dd ([`super::dd`]) with the f64→dd twiddle
// table, and rounds ONCE at the store: `y = dd_to_float(re) + u·D`.
#[allow(clippy::too_many_arguments)]
fn fft_conv_dd_threadgroup(
    u_bc: &[f32],
    kf_c: &[f32],
    d_c: f32,
    y_bc: &mut [f32],
    tw: &[CDd],
    seqlen: usize,
    fft_size: usize,
    lanes: usize,
) {
    let half_sz = fft_size / 2 + 1;
    // threadgroup complex_dd* shared.
    let mut shared = vec![CDd::default(); fft_size];
    let sh = Shared(shared.as_mut_ptr());
    let y_sh = Shared(y_bc.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
                let mut i = lane;
                while i < fft_size {
                    // SAFETY: cell i is this lane's private grid-stride slot.
                    unsafe {
                        sh.set(
                            i,
                            if i < seqlen { CDd::from_f32(u_bc[i], 0.0) } else { CDd::default() },
                        );
                    }
                    i += lanes;
                }
                barrier.wait();

                fft_dd_lane(sh, tw, fft_size, lane, lanes, barrier);

                let mut i = lane;
                while i < half_sz {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let z = sh.get(i);
                        sh.set(i, dd::cdd_mul(z, CDd::from_f32(kf_c[2 * i], kf_c[2 * i + 1])));
                    }
                    i += lanes;
                }
                barrier.wait();

                let mut i = half_sz + lane;
                while i < fft_size {
                    let mirror = fft_size - i;
                    // SAFETY: reads first half (read-only this stage), writes own cell i.
                    unsafe { sh.set(i, dd::cdd_conj(sh.get(mirror))) };
                    i += lanes;
                }
                barrier.wait();

                ifft_dd_lane(sh, tw, fft_size, lane, lanes, barrier);

                let mut t = lane;
                while t < seqlen {
                    // SAFETY: output cell t is this lane's private grid-stride slot.
                    unsafe { y_sh.set(t, dd::dd_to_f32(sh.get(t).re) + u_bc[t] * d_c) };
                    t += lanes;
                }
            });
        }
    });
}

/// Double-double fused FFT convolution — the CPU port of `fft_conv_dd` in FFTConvDd.metal:
/// same contract as [`fused_fft_conv`], but every butterfly, spectrum multiply, and the final
/// normalization run in double-double with host-f64 twiddles, so the f32 output is the
/// once-rounded ~f64 result (the truth tier) instead of accumulated f32 roundings.
#[allow(clippy::too_many_arguments)]
pub fn fused_fft_conv_dd(
    u: &[f32],
    k_f: &[f32],
    d: &[f32],
    y: &mut [f32],
    batch: usize,
    channels: usize,
    seqlen: usize,
    fft_size: usize,
    lanes: usize,
) {
    use rayon::prelude::*;
    let half_sz = fft_size / 2 + 1;
    assert!(batch > 0 && channels > 0 && seqlen > 0, "fused_fft_conv_dd: empty dims");
    assert!(fft_size.is_power_of_two(), "fused_fft_conv_dd: fft_size must be a power of two");
    assert!(fft_size >= seqlen, "fused_fft_conv_dd: fft_size {fft_size} < seqlen {seqlen}");
    assert_eq!(u.len(), batch * channels * seqlen, "fused_fft_conv_dd: u.len() != B·C·seqlen");
    assert_eq!(k_f.len(), channels * half_sz * 2, "fused_fft_conv_dd: k_f.len() != C·(fft/2+1)·2");
    assert_eq!(d.len(), channels, "fused_fft_conv_dd: d.len() != C");
    assert_eq!(y.len(), u.len(), "fused_fft_conv_dd: y.len() != u.len()");
    let lanes = lanes.clamp(1, fft_size);
    // Host-side dd twiddle table (the "DD Taylor series" TODO, done in f64), shared read-only.
    let tw = dd::fft_twiddles_dd(fft_size);
    y.par_chunks_mut(seqlen)
        .zip(u.par_chunks(seqlen))
        .enumerate()
        .for_each(|(bc, (y_bc, u_bc))| {
            let c = bc % channels;
            let kf_c = &k_f[c * half_sz * 2..(c + 1) * half_sz * 2];
            fft_conv_dd_threadgroup(u_bc, kf_c, d[c], y_bc, &tw, seqlen, fft_size, lanes);
        });
}

/// Double-double inverse real FFT — the CPU port of `irfft_dd` in IrfftDd.metal (torch's
/// `irfft` c2r contract, one thread per output sample, no barriers — a flat grid kernel):
///
/// `y[r][j] = scale · Σ_k a_k · (Re[r][k]·cos(2πkj/n) − Im[r][k]·sin(2πkj/n))`
///
/// with `a_0 = a_{n/2} = 1` (even `n`), else 2 — every product and the accumulation in
/// double-double, rounded once per sample. `re`/`im` are `[m, n/2+1]`; `out` is `[m, n]`.
/// `scale` is the dd norm factor (backward = `dd_from_f64(1/n)`).
pub fn irfft_dd(re: &[f32], im: &[f32], out: &mut [f32], m: usize, n: usize, scale: Dd) {
    use rayon::prelude::*;
    let freq = n / 2 + 1;
    assert!(m > 0 && n > 0, "irfft_dd: empty dims");
    assert_eq!(re.len(), m * freq, "irfft_dd: re.len() != M·(n/2+1)");
    assert_eq!(im.len(), m * freq, "irfft_dd: im.len() != M·(n/2+1)");
    assert_eq!(out.len(), m * n, "irfft_dd: out.len() != M·n");
    let n_even = n % 2 == 0;
    let nyq = n / 2;
    // tw[mm] = (cos, sin) of +2π·mm/n in dd; angle 2πkj/n folds to index (k·j) mod n.
    let tw = dd::irfft_twiddles_dd(n);
    out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for (j, o) in row.iter_mut().enumerate() {
            let mut acc = Dd::default();
            for k in 0..freq {
                let idx = (k * j) % n;
                let cs = tw[idx];
                let a = if k == 0 || (n_even && k == nyq) { 1.0f32 } else { 2.0f32 };
                let re_dd = Dd::from_f32(re[r * freq + k]);
                let im_dd = Dd::from_f32(im[r * freq + k]);
                let mut t = dd::dd_sub(dd::dd_mul(re_dd, cs.re), dd::dd_mul(im_dd, cs.im));
                t = dd::dd_mul(t, Dd::from_f32(a));
                acc = dd::dd_add(acc, t);
            }
            *o = dd::dd_to_f32(dd::dd_mul(acc, scale));
        }
    });
}

// One threadgroup: the fused forward Monarch butterfly for a single (batch,head) —
// row-DFT along L → twiddle → col-DFT along N — a line-for-line CPU port of
// `monarch_fused_fwd_f32` in `fused_monarch.rs` (stages 1-3). The `[N,L]` complex
// intermediate lives entirely in threadgroup memory (`sxr`/`sxi`, here the two halves of
// `scratch`) and never leaves it between stages; a `barrier.wait()` (== `threadgroup_barrier`)
// fences every stage, exactly as the three `threadgroup_barrier`s do on the GPU. The lanes
// grid-stride over the N·L cells and each computes a disjoint set (the CPU analog of the
// simdgroup tiling — same dispatch/shared-memory/barrier structure, dot-product form instead
// of `simdgroup_multiply_accumulate`). `xb` is `[N,L]` real; `dLr/dLi` are the real/imag
// planes of the L-point DFT matrix `[L,L]`, `dNr/dNi` the N-point `[N,N]`; `tw` is `[N,L,2]`
// interleaved; `ob` is the `[N,L,2]` complex output.
#[allow(clippy::too_many_arguments)]
fn monarch_fwd_threadgroup(
    xb: &[f32],
    dlr: &[f32],
    dli: &[f32],
    dnr: &[f32],
    dni: &[f32],
    tw: &[f32],
    ob: &mut [f32],
    n: usize,
    l: usize,
    lanes: usize,
) {
    let nl = n * l;
    // threadgroup shared memory: sxr = scratch[0..nl], sxi = scratch[nl..2*nl].
    let mut scratch = vec![0f32; 2 * nl];
    let sh = Shared(scratch.as_mut_ptr());
    let ob_sh = Shared(ob.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
                // stage 1: row DFT along L. Y[ni,lp] = Σ_k X[ni,k]·dL[k,lp] (x real → 2 real
                // GEMMs). Each grid-stride cell i is written by exactly one lane — disjoint.
                let mut i = lane;
                while i < nl {
                    let (ni, lp) = (i / l, i % l);
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..l {
                        let xv = xb[ni * l + k];
                        sr += xv * dlr[k * l + lp];
                        si += xv * dli[k * l + lp];
                    }
                    // SAFETY: cell i (sxr[i], sxi[i]) is this lane's private output this stage.
                    unsafe {
                        sh.set(i, sr);
                        sh.set(nl + i, si);
                    }
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — row-DFT visible before twiddle

                // stage 2: twiddle (elementwise complex) in threadgroup memory.
                let mut i = lane;
                while i < nl {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let (zr, zi) = (sh.get(i), sh.get(nl + i));
                        let (twr, twi) = (tw[i * 2], tw[i * 2 + 1]);
                        sh.set(i, zr * twr - zi * twi);
                        sh.set(nl + i, zr * twi + zi * twr);
                    }
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — twiddle visible before col-DFT

                // stage 3: col DFT along N. O[np,li] = Σ_k dN[np,k]·Z[k,li] (complex = 4 real
                // GEMMs). Reads of sxr/sxi are all-visible post-barrier and read-only now.
                let mut i = lane;
                while i < nl {
                    let (np, li) = (i / l, i % l);
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (dnr[np * n + k], dni[np * n + k]);
                        // SAFETY: post-barrier read-only view of the twiddled intermediate.
                        let (zr, zi) = unsafe { (sh.get(k * l + li), sh.get(nl + k * l + li)) };
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    // SAFETY: output pair (2i, 2i+1) is this lane's private (np,li) cell.
                    unsafe {
                        ob_sh.set(2 * i, sr);
                        ob_sh.set(2 * i + 1, si);
                    }
                    i += lanes;
                }
            });
        }
    });
}

/// Fused forward Monarch butterfly FFT with the full GPU dispatch model — the CPU port of
/// `fused_monarch.rs`'s `monarch_fused_fwd_f32`. `u` is `bh` real signals of shape `[N,L]`
/// (`bh = B·H`) laid end to end; the output `out` is the matching `bh × [N,L,2]` complex.
/// Fans out over the `(b,h)` grid with rayon (one **threadgroup** per signal), each threadgroup
/// running the staged row-DFT → twiddle → col-DFT on a team of `lanes` workers synchronised by
/// `threadgroup_barrier`s. The DFT matrices are given as separated real/imag planes: `dlr/dli`
/// `[L,L]`, `dnr/dni` `[N,N]`; `tw` is the `[N,L,2]` interleaved twiddles.
///
/// `lanes` is the simulated simdgroup width (Metal uses 32); 1 collapses to the serial kernel.
#[allow(clippy::too_many_arguments)]
pub fn fused_monarch_fwd(
    u: &[f32],
    dlr: &[f32],
    dli: &[f32],
    dnr: &[f32],
    dni: &[f32],
    tw: &[f32],
    out: &mut [f32],
    bh: usize,
    n: usize,
    l: usize,
    lanes: usize,
) {
    use rayon::prelude::*;
    let nl = n * l;
    assert!(bh > 0 && n > 0 && l > 0, "fused_monarch_fwd: B·H, N, L must be > 0");
    assert_eq!(u.len(), bh * nl, "fused_monarch_fwd: u.len() != B·H·N·L");
    assert_eq!(out.len(), bh * nl * 2, "fused_monarch_fwd: out.len() != B·H·N·L·2");
    assert_eq!(dlr.len(), l * l, "fused_monarch_fwd: dlr.len() != L·L");
    assert_eq!(dli.len(), l * l, "fused_monarch_fwd: dli.len() != L·L");
    assert_eq!(dnr.len(), n * n, "fused_monarch_fwd: dnr.len() != N·N");
    assert_eq!(dni.len(), n * n, "fused_monarch_fwd: dni.len() != N·N");
    assert_eq!(tw.len(), nl * 2, "fused_monarch_fwd: tw.len() != N·L·2");
    let lanes = lanes.clamp(1, nl.max(1));
    // Grid dispatch: one rayon task == one threadgroup == one (b,h) index.
    out.par_chunks_mut(nl * 2)
        .zip(u.par_chunks(nl))
        .for_each(|(ob, xb)| {
            monarch_fwd_threadgroup(xb, dlr, dli, dnr, dni, tw, ob, n, l, lanes);
        });
}

/// The ten Monarch conv operands the GPU packs into one padded buffer
/// (`dLr|dLi|dNr|dNi|tw|idNr|idNi|idLr|idLi|itw`) — here as unpadded slices, since the CPU
/// dot-product form needs no 8×8 tile alignment: `dlr/dli` `[L,L]`, `dnr/dni`/`idnr/idni`
/// `[N,N]`, `idlr/idli` `[L,L]`, `tw`/`itw` `[N,L,2]` interleaved.
pub struct MonarchConvMats<'a> {
    pub dlr: &'a [f32],
    pub dli: &'a [f32],
    pub dnr: &'a [f32],
    pub dni: &'a [f32],
    pub tw: &'a [f32],
    pub idnr: &'a [f32],
    pub idni: &'a [f32],
    pub idlr: &'a [f32],
    pub idli: &'a [f32],
    pub itw: &'a [f32],
}

impl MonarchConvMats<'_> {
    fn validate(&self, n: usize, l: usize) {
        let (ll, nn, twn) = (l * l, n * n, n * l * 2);
        assert_eq!(self.dlr.len(), ll, "monarch conv: dlr.len() != L·L");
        assert_eq!(self.dli.len(), ll, "monarch conv: dli.len() != L·L");
        assert_eq!(self.dnr.len(), nn, "monarch conv: dnr.len() != N·N");
        assert_eq!(self.dni.len(), nn, "monarch conv: dni.len() != N·N");
        assert_eq!(self.tw.len(), twn, "monarch conv: tw.len() != N·L·2");
        assert_eq!(self.idnr.len(), nn, "monarch conv: idnr.len() != N·N");
        assert_eq!(self.idni.len(), nn, "monarch conv: idni.len() != N·N");
        assert_eq!(self.idlr.len(), ll, "monarch conv: idlr.len() != L·L");
        assert_eq!(self.idli.len(), ll, "monarch conv: idli.len() != L·L");
        assert_eq!(self.itw.len(), twn, "monarch conv: itw.len() != N·L·2");
    }
}

// How one threadgroup's signal enters and leaves the Monarch conv pipeline — the only two
// places the plain and padded kernels differ (stages 1-6 are identical):
// * `Plain`: `xb`/`ob` are the `[N,L]` grid row-major, every cell live.
// * `Padded`: the length-T signal maps through the column-major flatten `t = c·N + r`
//   (zero past T), with optional input gate at load, and output gate / u·D skip at store.
enum MonarchIo<'a> {
    Plain,
    Padded {
        t_len: usize,
        ig: Option<&'a [f32]>,
        og: Option<&'a [f32]>,
        d_h: Option<f32>, // dvec[bh % H] when the skip gate is set
    },
}

// One threadgroup: the fused Monarch CONVOLUTION for a single (batch,head) — the CPU port of
// `monarch_fused_conv_f32` / `monarch_fused_conv_padded_f32` (fused_monarch.rs) stages 1-7:
// stage → row-DFT → twiddle → col-DFT → ×k_f → col-IDFT → conjugate twiddle → row-IDFT real
// × 1/(N·L). The staged input `ux` and both complex ping-pong planes (A = axr/axi,
// B = bxr/bxi) live in one shared scratch co-owned by the lane team; a `barrier.wait()`
// (== `threadgroup_barrier`) fences every stage. Lanes grid-stride the N·L cells, each
// writing a disjoint set — the dot-product form of the simdgroup tiling, with the GPU's
// four separate MMA accumulators kept as four separate sums combined after each K loop
// (`bxr = m0 − m1`, `bxi = m2 + m3`), so the summation structure matches the kernel's.
#[allow(clippy::too_many_arguments)]
fn monarch_conv_threadgroup(
    xb: &[f32],
    mats: &MonarchConvMats,
    kfb: &[f32],
    ob: &mut [f32],
    io: &MonarchIo,
    n: usize,
    l: usize,
    lanes: usize,
) {
    let nl = n * l;
    // threadgroup planes: ux | axr | axi | bxr | bxi.
    let mut scratch = vec![0f32; 5 * nl];
    let sh = Shared(scratch.as_mut_ptr());
    let (ux, axr, axi, bxr, bxi) = (0usize, nl, 2 * nl, 3 * nl, 4 * nl);
    let ob_sh = Shared(ob.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
                // preamble: stage the input into ux (gated + t-mapped for the padded kernel).
                let mut i = lane;
                while i < nl {
                    let v = match io {
                        MonarchIo::Plain => xb[i],
                        MonarchIo::Padded { t_len, ig, .. } => {
                            let (r, c) = (i / l, i % l);
                            let t = c * n + r;
                            if t < *t_len {
                                let mut v = xb[t];
                                if let Some(ig) = ig {
                                    v *= ig[t];
                                }
                                v
                            } else {
                                0.0
                            }
                        }
                    };
                    // SAFETY: cell i is this lane's private grid-stride slot.
                    unsafe { sh.set(ux + i, v) };
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — staging visible

                // stage 1: row DFT / L.  A[r,c] = Σ_k ux[r,k]·dL[k,c] (x real → 2 real sums).
                let mut i = lane;
                while i < nl {
                    let (r, c) = (i / l, i % l);
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..l {
                        // SAFETY: post-barrier read-only view of ux.
                        let xv = unsafe { sh.get(ux + r * l + k) };
                        sr += xv * mats.dlr[k * l + c];
                        si += xv * mats.dli[k * l + c];
                    }
                    // SAFETY: cell i of A is this lane's private output this stage.
                    unsafe {
                        sh.set(axr + i, sr);
                        sh.set(axi + i, si);
                    }
                    i += lanes;
                }
                barrier.wait(); // threadgroup_barrier — row-DFT visible

                // stage 2: forward twiddle (A *= tw), elementwise.
                let mut i = lane;
                while i < nl {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let (zr, zi) = (sh.get(axr + i), sh.get(axi + i));
                        let (twr, twi) = (mats.tw[i * 2], mats.tw[i * 2 + 1]);
                        sh.set(axr + i, zr * twr - zi * twi);
                        sh.set(axi + i, zr * twi + zi * twr);
                    }
                    i += lanes;
                }
                barrier.wait();

                // stage 3: col DFT / N.  B[r,c] = Σ_k dN[r,k]·A[k,c] — four separate sums
                // (the GPU's m0..m3 accumulators), combined after the K loop.
                let mut i = lane;
                while i < nl {
                    let (r, c) = (i / l, i % l);
                    let (mut m0, mut m1, mut m2, mut m3) = (0f32, 0f32, 0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (mats.dnr[r * n + k], mats.dni[r * n + k]);
                        // SAFETY: post-barrier read-only view of A.
                        let (zr, zi) = unsafe { (sh.get(axr + k * l + c), sh.get(axi + k * l + c)) };
                        m0 += dr * zr;
                        m1 += di * zi;
                        m2 += dr * zi;
                        m3 += di * zr;
                    }
                    // SAFETY: cell i of B is this lane's private output this stage.
                    unsafe {
                        sh.set(bxr + i, m0 - m1);
                        sh.set(bxi + i, m2 + m3);
                    }
                    i += lanes;
                }
                barrier.wait();

                // stage 4: × k_f (B *= k_f), elementwise over the [N,L] grid.
                let mut i = lane;
                while i < nl {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let (zr, zi) = (sh.get(bxr + i), sh.get(bxi + i));
                        let (kr, ki) = (kfb[i * 2], kfb[i * 2 + 1]);
                        sh.set(bxr + i, zr * kr - zi * ki);
                        sh.set(bxi + i, zr * ki + zi * kr);
                    }
                    i += lanes;
                }
                barrier.wait();

                // stage 5: col IDFT / N.  A[r,c] = Σ_k idN[r,k]·B[k,c] (four sums again).
                let mut i = lane;
                while i < nl {
                    let (r, c) = (i / l, i % l);
                    let (mut m0, mut m1, mut m2, mut m3) = (0f32, 0f32, 0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (mats.idnr[r * n + k], mats.idni[r * n + k]);
                        // SAFETY: post-barrier read-only view of B.
                        let (zr, zi) = unsafe { (sh.get(bxr + k * l + c), sh.get(bxi + k * l + c)) };
                        m0 += dr * zr;
                        m1 += di * zi;
                        m2 += dr * zi;
                        m3 += di * zr;
                    }
                    // SAFETY: cell i of A is this lane's private output this stage.
                    unsafe {
                        sh.set(axr + i, m0 - m1);
                        sh.set(axi + i, m2 + m3);
                    }
                    i += lanes;
                }
                barrier.wait();

                // stage 6: conjugate twiddle (A *= itw).
                let mut i = lane;
                while i < nl {
                    // SAFETY: lane reads+writes only its own cell i this stage.
                    unsafe {
                        let (zr, zi) = (sh.get(axr + i), sh.get(axi + i));
                        let (twr, twi) = (mats.itw[i * 2], mats.itw[i * 2 + 1]);
                        sh.set(axr + i, zr * twr - zi * twi);
                        sh.set(axi + i, zr * twi + zi * twr);
                    }
                    i += lanes;
                }
                barrier.wait();

                // stage 7: row IDFT / L, real part × 1/(N·L) — two sums (Ar·idLr, Ai·idLi)
                // combined then scaled; store through the io mapping (T-truncate, skip, gate).
                let scale = 1.0f32 / (n * l) as f32;
                let mut i = lane;
                while i < nl {
                    let (r, c) = (i / l, i % l);
                    let (mut m0, mut m1) = (0f32, 0f32);
                    for k in 0..l {
                        // SAFETY: post-barrier read-only view of A.
                        let (ar, ai) = unsafe { (sh.get(axr + r * l + k), sh.get(axi + r * l + k)) };
                        m0 += ar * mats.idlr[k * l + c];
                        m1 += ai * mats.idli[k * l + c];
                    }
                    match io {
                        MonarchIo::Plain => {
                            // SAFETY: output cell i is this lane's private (r,c) slot.
                            unsafe { ob_sh.set(i, (m0 - m1) * scale) };
                        }
                        MonarchIo::Padded { t_len, og, d_h, .. } => {
                            let t = c * n + r;
                            if t < *t_len {
                                let mut v = (m0 - m1) * scale;
                                if let Some(d_h) = d_h {
                                    // u·D delta-tap skip: ux still holds the STAGED (gated)
                                    // input — untouched since the preamble.
                                    // SAFETY: post-barrier read-only view of ux.
                                    v += unsafe { sh.get(ux + r * l + c) } * d_h;
                                }
                                if let Some(og) = og {
                                    v *= og[t];
                                }
                                // SAFETY: output cell t is owned by exactly one (r,c) — the
                                // flatten t = c·N + r is a bijection on the valid grid.
                                unsafe { ob_sh.set(t, v) };
                            }
                        }
                    }
                    i += lanes;
                }
            });
        }
    });
}

/// Fused Monarch convolution with the full GPU dispatch model — the CPU port of
/// `monarch_fused_conv_f32`: row-DFT → twiddle → col-DFT → ×`kf` → col-IDFT → conjugate
/// twiddle → real row-IDFT × 1/(N·L), one threadgroup per `(b,h)`. `u` is `bh` real `[N,L]`
/// signals; `kf` is the matching `bh × [N,L,2]` spectrum (broadcast on the host if shared);
/// `out` gets the `bh × [N,L]` real result.
#[allow(clippy::too_many_arguments)]
pub fn fused_monarch_conv(
    u: &[f32],
    mats: &MonarchConvMats,
    kf: &[f32],
    out: &mut [f32],
    bh: usize,
    n: usize,
    l: usize,
    lanes: usize,
) {
    use rayon::prelude::*;
    let nl = n * l;
    assert!(bh > 0 && n > 0 && l > 0, "fused_monarch_conv: B·H, N, L must be > 0");
    mats.validate(n, l);
    assert_eq!(u.len(), bh * nl, "fused_monarch_conv: u.len() != B·H·N·L");
    assert_eq!(kf.len(), bh * nl * 2, "fused_monarch_conv: kf.len() != B·H·N·L·2");
    assert_eq!(out.len(), bh * nl, "fused_monarch_conv: out.len() != B·H·N·L");
    let lanes = lanes.clamp(1, nl.max(1));
    out.par_chunks_mut(nl)
        .zip(u.par_chunks(nl).zip(kf.par_chunks(nl * 2)))
        .for_each(|(ob, (xb, kfb))| {
            monarch_conv_threadgroup(xb, mats, kfb, ob, &MonarchIo::Plain, n, l, lanes);
        });
}

/// Padded/gated fused Monarch convolution — the CPU port of `monarch_fused_conv_padded_f32`:
/// the length-`t_len` signal enters via the column-major flatten `t = c·N + r` (zero-fill past
/// `t_len`), with the optional gates fused: bit 0 of `gates` multiplies the input gate at load,
/// bit 1 the output gate at store, bit 2 adds the `u·D` skip (staged gated input × `dvec[h]`).
///
/// `u_ext` is `[G, B·H, t_len]` — slot 0 the signal, then the input gate (iff bit 0), then the
/// output gate (iff bit 1). `dvec` is `[H]`, required iff bit 2. `out` is `[B·H, t_len]`.
#[allow(clippy::too_many_arguments)]
pub fn fused_monarch_conv_padded(
    u_ext: &[f32],
    mats: &MonarchConvMats,
    kf: &[f32],
    out: &mut [f32],
    b: usize,
    h: usize,
    t_len: usize,
    gates: u32,
    dvec: Option<&[f32]>,
    n: usize,
    l: usize,
    lanes: usize,
) {
    use rayon::prelude::*;
    let (bh, nl) = (b * h, n * l);
    assert!(bh > 0 && n > 0 && l > 0 && t_len > 0, "monarch conv padded: empty dims");
    assert!(t_len <= nl, "monarch conv padded: t_len {t_len} > N·L {nl}");
    mats.validate(n, l);
    let slots = 1 + (gates & 1) as usize + ((gates >> 1) & 1) as usize;
    assert_eq!(u_ext.len(), slots * bh * t_len, "monarch conv padded: u_ext.len() != G·B·H·T");
    assert_eq!(kf.len(), bh * nl * 2, "monarch conv padded: kf.len() != B·H·N·L·2");
    assert_eq!(out.len(), bh * t_len, "monarch conv padded: out.len() != B·H·T");
    let d = if gates & 4 != 0 {
        let d = dvec.expect("monarch conv padded: gates bit 2 set but no dvec");
        assert_eq!(d.len(), h, "monarch conv padded: dvec.len() != H");
        Some(d)
    } else {
        None
    };
    let lanes = lanes.clamp(1, nl.max(1));
    out.par_chunks_mut(t_len)
        .zip(kf.par_chunks(nl * 2))
        .enumerate()
        .for_each(|(bhi, (ob, kfb))| {
            let xb = &u_ext[bhi * t_len..(bhi + 1) * t_len];
            let ig = (gates & 1 != 0).then(|| &u_ext[(bh + bhi) * t_len..(bh + bhi + 1) * t_len]);
            let og_slot = if gates & 1 != 0 { 2 } else { 1 };
            let og = (gates & 2 != 0)
                .then(|| &u_ext[(og_slot * bh + bhi) * t_len..(og_slot * bh + bhi + 1) * t_len]);
            let io = MonarchIo::Padded {
                t_len,
                ig,
                og,
                d_h: d.map(|d| d[bhi % h]),
            };
            monarch_conv_threadgroup(xb, mats, kfb, ob, &io, n, l, lanes);
        });
}

/// Real row-IDFT — the CPU port of `butterfly_row_idft_real_f32` (butterfly.rs), the closing
/// stage of the UNFUSED butterfly path: `out[·,n,l] = Re(Σ_k x[·,n,k]·idL[l,k]) / (N·L)`, one
/// flat grid cell per output (no threadgroup state → plain rayon fan-out). `x` is `[BH,N,L,2]`,
/// `id_f_l` `[L,L,2]` (indexed `[l_out, k]`), `out` `[BH,N,L]`; the interleaved
/// multiply-subtract and the final division are kept as the kernel writes them.
pub fn row_idft_real(x: &[f32], id_f_l: &[f32], out: &mut [f32], bh: usize, n: usize, l: usize) {
    use rayon::prelude::*;
    assert_eq!(x.len(), bh * n * l * 2, "row_idft_real: x.len() != B·H·N·L·2");
    assert_eq!(id_f_l.len(), l * l * 2, "row_idft_real: id_f_l.len() != L·L·2");
    assert_eq!(out.len(), bh * n * l, "row_idft_real: out.len() != B·H·N·L");
    out.par_chunks_mut(l).enumerate().for_each(|(row, orow)| {
        let x_base = row * l; // (bh·N + n) row of complex length-L cells
        for (l_out, o) in orow.iter_mut().enumerate() {
            let mut sr = 0f32;
            for k in 0..l {
                let df = (l_out * l + k) * 2;
                let (dr, di) = (id_f_l[df], id_f_l[df + 1]);
                let xi = (x_base + k) * 2;
                sr += dr * x[xi] - di * x[xi + 1];
            }
            *o = sr / (n * l) as f32;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // scalar radix-2 reference (single lane, no barriers) — the parity target.
    fn ref_fft(data: &mut [f32], n: usize, inverse: bool) {
        fft_threadgroup(data, n, inverse, 1);
    }

    #[test]
    fn barrier_synchronized_fft_matches_serial_across_lane_counts() {
        // The multi-lane, barrier-synchronised threadgroup must produce bit-identical results to
        // the single-lane serial run — proving the simulated threadgroup_barriers correctly order
        // the stages under real concurrency (and the disjoint-write discipline holds).
        for &n in &[8usize, 16, 64, 256] {
            let orig: Vec<f32> = (0..2 * n).map(|i| ((i * 37 % 11) as f32 / 11.0) - 0.5).collect();
            let mut reference = orig.clone();
            ref_fft(&mut reference, n, false);
            for &lanes in &[2usize, 4, 8, 16] {
                let mut d = orig.clone();
                fft_threadgroup(&mut d, n, false, lanes);
                for (g, r) in d.iter().zip(&reference) {
                    assert_eq!(g.to_bits(), r.to_bits(), "n={n} lanes={lanes}: barrier race?");
                }
            }
        }
    }

    // Independent oracle for the fused forward Monarch of one [N,L] signal, via candle matmul
    // (a different implementation path than the hand-rolled dot-product kernel) — row-DFT,
    // twiddle, col-DFT. Returns the [N,L,2] interleaved complex output.
    #[allow(clippy::too_many_arguments)]
    fn ref_monarch_fwd(
        xb: &[f32],
        dlr: &[f32],
        dli: &[f32],
        dnr: &[f32],
        dni: &[f32],
        tw: &[f32],
        n: usize,
        l: usize,
    ) -> Vec<f32> {
        use candle_core::{Device, Tensor};
        let d = Device::Cpu;
        let t = |v: &[f32], r: usize, c: usize| Tensor::from_slice(v, (r, c), &d).unwrap();
        let x = t(xb, n, l);
        // stage 1: Y = X @ dL (real/imag).
        let yr: Vec<f32> = x.matmul(&t(dlr, l, l)).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let yi: Vec<f32> = x.matmul(&t(dli, l, l)).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        // stage 2: twiddle.
        let (mut zr, mut zi) = (vec![0f32; n * l], vec![0f32; n * l]);
        for i in 0..n * l {
            let (twr, twi) = (tw[i * 2], tw[i * 2 + 1]);
            zr[i] = yr[i] * twr - yi[i] * twi;
            zi[i] = yr[i] * twi + yi[i] * twr;
        }
        // stage 3: O = dN @ Z (complex).
        let (zr_t, zi_t) = (t(&zr, n, l), t(&zi, n, l));
        let (dnr_t, dni_t) = (t(dnr, n, n), t(dni, n, n));
        let or = (dnr_t.matmul(&zr_t).unwrap() - dni_t.matmul(&zi_t).unwrap()).unwrap();
        let oi = (dnr_t.matmul(&zi_t).unwrap() + dni_t.matmul(&zr_t).unwrap()).unwrap();
        let orv: Vec<f32> = or.flatten_all().unwrap().to_vec1().unwrap();
        let oiv: Vec<f32> = oi.flatten_all().unwrap().to_vec1().unwrap();
        let mut ob = vec![0f32; n * l * 2];
        for i in 0..n * l {
            ob[2 * i] = orv[i];
            ob[2 * i + 1] = oiv[i];
        }
        ob
    }

    // deterministic pseudo-random f32 in [-0.5, 0.5).
    fn rnd(seed: usize) -> f32 {
        ((seed.wrapping_mul(2654435761) % 1009) as f32 / 1009.0) - 0.5
    }

    #[test]
    fn monarch_barrier_matches_serial_across_lane_counts() {
        // The multi-lane, barrier-synchronised threadgroup Monarch must be bit-identical to the
        // single-lane serial run — proving the two threadgroup_barriers (row-DFT→twiddle→col-DFT)
        // correctly order the stages under real concurrency and the disjoint-write discipline holds.
        for &(n, l) in &[(4usize, 4usize), (8, 6), (3, 5), (6, 8)] {
            let xb: Vec<f32> = (0..n * l).map(|i| rnd(i + 1)).collect();
            let dlr: Vec<f32> = (0..l * l).map(|i| rnd(i + 7)).collect();
            let dli: Vec<f32> = (0..l * l).map(|i| rnd(i + 13)).collect();
            let dnr: Vec<f32> = (0..n * n).map(|i| rnd(i + 19)).collect();
            let dni: Vec<f32> = (0..n * n).map(|i| rnd(i + 23)).collect();
            let tw: Vec<f32> = (0..n * l * 2).map(|i| rnd(i + 29)).collect();

            let mut serial = vec![0f32; n * l * 2];
            fused_monarch_fwd(&xb, &dlr, &dli, &dnr, &dni, &tw, &mut serial, 1, n, l, 1);
            for &lanes in &[2usize, 3, 4, 8] {
                let mut got = vec![0f32; n * l * 2];
                fused_monarch_fwd(&xb, &dlr, &dli, &dnr, &dni, &tw, &mut got, 1, n, l, lanes);
                for (g, r) in got.iter().zip(&serial) {
                    assert_eq!(g.to_bits(), r.to_bits(), "n={n} l={l} lanes={lanes}: barrier race?");
                }
            }
        }
    }

    #[test]
    fn monarch_fwd_matches_candle_reference() {
        // The full grid dispatch (multiple threadgroups) vs the independent candle-matmul oracle,
        // per (b,h) — validates the three-stage math, not just concurrency.
        let (bh, n, l) = (4usize, 8usize, 6usize);
        let nl = n * l;
        let u: Vec<f32> = (0..bh * nl).map(|i| rnd(i + 3)).collect();
        let dlr: Vec<f32> = (0..l * l).map(|i| rnd(i + 7)).collect();
        let dli: Vec<f32> = (0..l * l).map(|i| rnd(i + 13)).collect();
        let dnr: Vec<f32> = (0..n * n).map(|i| rnd(i + 19)).collect();
        let dni: Vec<f32> = (0..n * n).map(|i| rnd(i + 23)).collect();
        let tw: Vec<f32> = (0..nl * 2).map(|i| rnd(i + 29)).collect();

        let mut got = vec![0f32; bh * nl * 2];
        fused_monarch_fwd(&u, &dlr, &dli, &dnr, &dni, &tw, &mut got, bh, n, l, 8);
        for b in 0..bh {
            let xb = &u[b * nl..(b + 1) * nl];
            let want = ref_monarch_fwd(xb, &dlr, &dli, &dnr, &dni, &tw, n, l);
            let g = &got[b * nl * 2..(b + 1) * nl * 2];
            let (mut md, mut sc) = (0f32, 1e-6f32);
            for (a, e) in g.iter().zip(&want) {
                md = md.max((a - e).abs());
                sc = sc.max(e.abs());
            }
            assert!(md / sc < 1e-4, "bh={b}: monarch vs candle rel {}", md / sc);
        }
    }

    // f64 naive-DFT reference for the fused conv — the same oracle the JIT wrapper's
    // `FusedFftConv::cpu_fwd` uses: full DFT of the zero-padded input, × k_f with Hermitian
    // mirroring of the half-spectrum, inverse DFT, truncate, + u·D.
    fn ref_fft_conv_f64(
        u: &[f32],
        k_f: &[f32],
        d: &[f32],
        batch: usize,
        channels: usize,
        seqlen: usize,
        fft_size: usize,
    ) -> Vec<f64> {
        let half = fft_size / 2 + 1;
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut y = vec![0f64; batch * channels * seqlen];
        for bi in 0..batch {
            for ci in 0..channels {
                let base = (bi * channels + ci) * seqlen;
                let (mut sre, mut sim) = (vec![0f64; fft_size], vec![0f64; fft_size]);
                for k in 0..fft_size {
                    let (mut sr, mut si) = (0f64, 0f64);
                    for t in 0..seqlen {
                        let ang = -two_pi * (k as f64) * (t as f64) / fft_size as f64;
                        sr += u[base + t] as f64 * ang.cos();
                        si += u[base + t] as f64 * ang.sin();
                    }
                    let (kr, ki) = if k < half {
                        (k_f[(ci * half + k) * 2] as f64, k_f[(ci * half + k) * 2 + 1] as f64)
                    } else {
                        let m = fft_size - k;
                        (k_f[(ci * half + m) * 2] as f64, -(k_f[(ci * half + m) * 2 + 1] as f64))
                    };
                    sre[k] = sr * kr - si * ki;
                    sim[k] = sr * ki + si * kr;
                }
                let scale = 1.0 / fft_size as f64;
                for t in 0..seqlen {
                    let mut acc = 0f64;
                    for k in 0..fft_size {
                        let ang = two_pi * (k as f64) * (t as f64) / fft_size as f64;
                        acc += sre[k] * ang.cos() - sim[k] * ang.sin();
                    }
                    y[base + t] = acc * scale + u[base + t] as f64 * d[ci] as f64;
                }
            }
        }
        y
    }

    fn conv_case(
        batch: usize,
        channels: usize,
        seqlen: usize,
        fft_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let half = fft_size / 2 + 1;
        let u: Vec<f32> = (0..batch * channels * seqlen).map(|i| rnd(i + 3) * 4.0).collect();
        let kf: Vec<f32> = (0..channels * half * 2).map(|i| rnd(i + 41) * 2.0).collect();
        let d: Vec<f32> = (0..channels).map(|i| rnd(i + 97)).collect();
        (u, kf, d)
    }

    #[test]
    fn fft_conv_matches_f64_dft_reference() {
        // The full grid dispatch vs the f64 naive-DFT oracle (independent implementation path):
        // validates staging, FFT, fixed-order spectrum multiply, Hermitian mirror, conjugate-form
        // iFFT, truncation, and the u·D skip.
        let (batch, channels, seqlen, fft_size) = (2usize, 3usize, 13usize, 32usize);
        let (u, kf, d) = conv_case(batch, channels, seqlen, fft_size);
        let mut y = vec![0f32; u.len()];
        fused_fft_conv(&u, &kf, &d, &mut y, batch, channels, seqlen, fft_size, 8);
        let want = ref_fft_conv_f64(&u, &kf, &d, batch, channels, seqlen, fft_size);
        let (mut md, mut sc) = (0f64, 1e-6f64);
        for (g, w) in y.iter().zip(&want) {
            md = md.max((*g as f64 - w).abs());
            sc = sc.max(w.abs());
        }
        assert!(md / sc < 1e-5, "fft_conv vs f64 DFT rel {}", md / sc);
    }

    #[test]
    fn fft_conv_lane_parity() {
        // Multi-lane runs must be bit-identical to the serial one — the barrier discipline
        // across all six stages (stage/fft/cmul/mirror/ifft/store) under real concurrency.
        let (batch, channels, seqlen, fft_size) = (1usize, 2usize, 24usize, 64usize);
        let (u, kf, d) = conv_case(batch, channels, seqlen, fft_size);
        let mut serial = vec![0f32; u.len()];
        fused_fft_conv(&u, &kf, &d, &mut serial, batch, channels, seqlen, fft_size, 1);
        for &lanes in &[2usize, 4, 8, 32] {
            let mut got = vec![0f32; u.len()];
            fused_fft_conv(&u, &kf, &d, &mut got, batch, channels, seqlen, fft_size, lanes);
            for (g, r) in got.iter().zip(&serial) {
                assert_eq!(g.to_bits(), r.to_bits(), "lanes={lanes}: barrier race?");
            }
        }
    }

    #[test]
    fn rfft_irfft_round_trip_and_dft_parity() {
        // rfft against the f64 DFT half-spectrum, then irfft recovers the (zero-padded) signal.
        let (seqlen, n) = (12usize, 32usize);
        let x: Vec<f32> = (0..seqlen).map(|i| rnd(i + 11) * 3.0).collect();
        let half = n / 2 + 1;
        let mut spec = vec![0f32; half * 2];
        rfft(&x, &mut spec, seqlen, n, 4);
        let two_pi = 2.0 * std::f64::consts::PI;
        for k in 0..half {
            let (mut sr, mut si) = (0f64, 0f64);
            for (t, &v) in x.iter().enumerate() {
                let ang = -two_pi * (k as f64) * (t as f64) / n as f64;
                sr += v as f64 * ang.cos();
                si += v as f64 * ang.sin();
            }
            assert!((spec[2 * k] as f64 - sr).abs() < 1e-4, "rfft re k={k}");
            assert!((spec[2 * k + 1] as f64 - si).abs() < 1e-4, "rfft im k={k}");
        }
        let mut back = vec![0f32; seqlen];
        irfft(&spec, &mut back, n, seqlen, 4);
        for (g, o) in back.iter().zip(&x) {
            assert!((g - o).abs() < 1e-4, "round-trip {g} vs {o}");
        }
    }

    #[test]
    fn fft_conv_dd_beats_f32_and_tracks_f64() {
        // The dd conv (host-f64 twiddles, dd butterflies, one rounding) must sit at the f32
        // output-rounding floor vs the f64 oracle — and never be worse than the f32 kernel.
        let (batch, channels, seqlen, fft_size) = (2usize, 3usize, 13usize, 32usize);
        let (u, kf, d) = conv_case(batch, channels, seqlen, fft_size);
        let want = ref_fft_conv_f64(&u, &kf, &d, batch, channels, seqlen, fft_size);
        let sc = want.iter().fold(1e-6f64, |m, w| m.max(w.abs()));

        let mut y32 = vec![0f32; u.len()];
        fused_fft_conv(&u, &kf, &d, &mut y32, batch, channels, seqlen, fft_size, 4);
        let mut ydd = vec![0f32; u.len()];
        fused_fft_conv_dd(&u, &kf, &d, &mut ydd, batch, channels, seqlen, fft_size, 4);

        let err = |y: &[f32]| {
            y.iter().zip(&want).fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()))
        };
        let (e32, edd) = (err(&y32), err(&ydd));
        assert!(edd <= e32, "dd err {edd:e} must not exceed f32 err {e32:e}");
        assert!(edd / sc < 1e-6, "dd err {edd:e} above the f32 rounding floor (scale {sc:e})");
    }

    #[test]
    fn fft_conv_dd_lane_parity() {
        let (batch, channels, seqlen, fft_size) = (1usize, 2usize, 20usize, 32usize);
        let (u, kf, d) = conv_case(batch, channels, seqlen, fft_size);
        let mut serial = vec![0f32; u.len()];
        fused_fft_conv_dd(&u, &kf, &d, &mut serial, batch, channels, seqlen, fft_size, 1);
        for &lanes in &[2usize, 4, 16] {
            let mut got = vec![0f32; u.len()];
            fused_fft_conv_dd(&u, &kf, &d, &mut got, batch, channels, seqlen, fft_size, lanes);
            for (g, r) in got.iter().zip(&serial) {
                assert_eq!(g.to_bits(), r.to_bits(), "dd lanes={lanes}: barrier race?");
            }
        }
    }

    #[test]
    fn irfft_dd_matches_f64_reference() {
        // vs the exact f64 inverse real DFT (irfft.rs's irfft_cpu_f64 basis: a·cos, −a·sin,
        // backward 1/n scale), for both even and odd n.
        for &n in &[16usize, 15] {
            let (m, freq) = (3usize, n / 2 + 1);
            let re: Vec<f32> = (0..m * freq).map(|i| rnd(i + 5) * 2.0).collect();
            let im: Vec<f32> = (0..m * freq).map(|i| rnd(i + 31) * 2.0).collect();
            let scale64 = 1.0 / n as f64;
            let mut got = vec![0f32; m * n];
            irfft_dd(&re, &im, &mut got, m, n, crate::flashkern::dd::dd_from_f64(scale64));
            let two_pi = 2.0 * std::f64::consts::PI;
            for r in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for k in 0..freq {
                        let a = if k == 0 || (n % 2 == 0 && k == n / 2) { 1.0 } else { 2.0 };
                        let ang = two_pi * k as f64 * j as f64 / n as f64;
                        acc += re[r * freq + k] as f64 * a * ang.cos()
                            - im[r * freq + k] as f64 * a * ang.sin();
                    }
                    let want = acc * scale64;
                    let g = got[r * n + j] as f64;
                    assert!(
                        (g - want).abs() < 1e-6 * want.abs().max(1.0),
                        "n={n} r={r} j={j}: dd {g} vs f64 {want}"
                    );
                }
            }
        }
    }

    // f64 straight-loop oracle for the 7-stage Monarch conv pipeline over ONE staged [N,L]
    // signal (independent of the kernel's f32 four-sum structure). Returns the [N,L] real grid
    // BEFORE any store-side mapping (truncation/skip/output gate).
    fn ref_monarch_conv_f64(ux: &[f64], m: &MonarchConvMats, kfb: &[f32], n: usize, l: usize) -> Vec<f64> {
        let nl = n * l;
        let (mut ar, mut ai) = (vec![0f64; nl], vec![0f64; nl]);
        for r in 0..n {
            for c in 0..l {
                let (mut sr, mut si) = (0f64, 0f64);
                for k in 0..l {
                    sr += ux[r * l + k] * m.dlr[k * l + c] as f64;
                    si += ux[r * l + k] * m.dli[k * l + c] as f64;
                }
                ar[r * l + c] = sr;
                ai[r * l + c] = si;
            }
        }
        for i in 0..nl {
            let (tr, ti) = (m.tw[i * 2] as f64, m.tw[i * 2 + 1] as f64);
            let (zr, zi) = (ar[i], ai[i]);
            ar[i] = zr * tr - zi * ti;
            ai[i] = zr * ti + zi * tr;
        }
        let (mut br, mut bi) = (vec![0f64; nl], vec![0f64; nl]);
        for r in 0..n {
            for c in 0..l {
                let (mut sr, mut si) = (0f64, 0f64);
                for k in 0..n {
                    let (dr, di) = (m.dnr[r * n + k] as f64, m.dni[r * n + k] as f64);
                    let (zr, zi) = (ar[k * l + c], ai[k * l + c]);
                    sr += dr * zr - di * zi;
                    si += dr * zi + di * zr;
                }
                br[r * l + c] = sr;
                bi[r * l + c] = si;
            }
        }
        for i in 0..nl {
            let (kr, ki) = (kfb[i * 2] as f64, kfb[i * 2 + 1] as f64);
            let (zr, zi) = (br[i], bi[i]);
            br[i] = zr * kr - zi * ki;
            bi[i] = zr * ki + zi * kr;
        }
        for r in 0..n {
            for c in 0..l {
                let (mut sr, mut si) = (0f64, 0f64);
                for k in 0..n {
                    let (dr, di) = (m.idnr[r * n + k] as f64, m.idni[r * n + k] as f64);
                    let (zr, zi) = (br[k * l + c], bi[k * l + c]);
                    sr += dr * zr - di * zi;
                    si += dr * zi + di * zr;
                }
                ar[r * l + c] = sr;
                ai[r * l + c] = si;
            }
        }
        for i in 0..nl {
            let (tr, ti) = (m.itw[i * 2] as f64, m.itw[i * 2 + 1] as f64);
            let (zr, zi) = (ar[i], ai[i]);
            ar[i] = zr * tr - zi * ti;
            ai[i] = zr * ti + zi * tr;
        }
        let mut out = vec![0f64; nl];
        let scale = 1.0 / (n * l) as f64;
        for r in 0..n {
            for c in 0..l {
                let mut sr = 0f64;
                for k in 0..l {
                    sr += ar[r * l + k] * m.idlr[k * l + c] as f64
                        - ai[r * l + k] * m.idli[k * l + c] as f64;
                }
                out[r * l + c] = sr * scale;
            }
        }
        out
    }

    // Random (non-DFT) conv operand set — math parity doesn't require true DFT matrices,
    // and random operands keep the oracle independent of any FFT identity.
    fn conv_mats(n: usize, l: usize, seed: usize) -> Vec<Vec<f32>> {
        let sizes = [l * l, l * l, n * n, n * n, n * l * 2, n * n, n * n, l * l, l * l, n * l * 2];
        sizes
            .iter()
            .enumerate()
            .map(|(s, &len)| (0..len).map(|i| rnd(i + seed + s * 101)).collect())
            .collect()
    }

    fn mats_view(v: &[Vec<f32>]) -> MonarchConvMats<'_> {
        MonarchConvMats {
            dlr: &v[0],
            dli: &v[1],
            dnr: &v[2],
            dni: &v[3],
            tw: &v[4],
            idnr: &v[5],
            idni: &v[6],
            idlr: &v[7],
            idli: &v[8],
            itw: &v[9],
        }
    }

    #[test]
    fn monarch_conv_matches_f64_reference() {
        let (bh, n, l) = (3usize, 8usize, 6usize);
        let nl = n * l;
        let mv = conv_mats(n, l, 7);
        let mats = mats_view(&mv);
        let u: Vec<f32> = (0..bh * nl).map(|i| rnd(i + 3)).collect();
        let kf: Vec<f32> = (0..bh * nl * 2).map(|i| rnd(i + 501)).collect();
        let mut got = vec![0f32; bh * nl];
        fused_monarch_conv(&u, &mats, &kf, &mut got, bh, n, l, 4);
        for b in 0..bh {
            let ux: Vec<f64> = u[b * nl..(b + 1) * nl].iter().map(|&x| x as f64).collect();
            let want = ref_monarch_conv_f64(&ux, &mats, &kf[b * nl * 2..(b + 1) * nl * 2], n, l);
            let (mut md, mut sc) = (0f64, 1e-6f64);
            for (g, w) in got[b * nl..(b + 1) * nl].iter().zip(&want) {
                md = md.max((*g as f64 - w).abs());
                sc = sc.max(w.abs());
            }
            assert!(md / sc < 1e-4, "bh={b}: monarch conv vs f64 rel {}", md / sc);
        }
    }

    #[test]
    fn monarch_conv_lane_parity() {
        let (bh, n, l) = (2usize, 6usize, 4usize);
        let nl = n * l;
        let mv = conv_mats(n, l, 13);
        let mats = mats_view(&mv);
        let u: Vec<f32> = (0..bh * nl).map(|i| rnd(i + 9)).collect();
        let kf: Vec<f32> = (0..bh * nl * 2).map(|i| rnd(i + 601)).collect();
        let mut serial = vec![0f32; bh * nl];
        fused_monarch_conv(&u, &mats, &kf, &mut serial, bh, n, l, 1);
        for &lanes in &[2usize, 3, 8] {
            let mut got = vec![0f32; bh * nl];
            fused_monarch_conv(&u, &mats, &kf, &mut got, bh, n, l, lanes);
            for (g, r) in got.iter().zip(&serial) {
                assert_eq!(g.to_bits(), r.to_bits(), "conv lanes={lanes}: barrier race?");
            }
        }
    }

    #[test]
    fn monarch_conv_padded_gates_and_skip_match_f64() {
        // All three gate bits on: input gate at load, u·D skip from the STAGED (gated) input,
        // output gate at store, T < N·L truncation through the t = c·N + r flatten.
        let (b, h, n, l, t_len) = (2usize, 2usize, 6usize, 4usize, 20usize);
        let (bh, nl) = (b * h, n * l);
        let gates = 0b111u32;
        let mv = conv_mats(n, l, 29);
        let mats = mats_view(&mv);
        let u_ext: Vec<f32> = (0..3 * bh * t_len).map(|i| rnd(i + 17)).collect();
        let kf: Vec<f32> = (0..bh * nl * 2).map(|i| rnd(i + 701)).collect();
        let dvec: Vec<f32> = (0..h).map(|i| rnd(i + 811)).collect();
        let mut got = vec![0f32; bh * t_len];
        fused_monarch_conv_padded(
            &u_ext, &mats, &kf, &mut got, b, h, t_len, gates, Some(&dvec), n, l, 8,
        );
        for bhi in 0..bh {
            let xb = &u_ext[bhi * t_len..(bhi + 1) * t_len];
            let ig = &u_ext[(bh + bhi) * t_len..(bh + bhi + 1) * t_len];
            let og = &u_ext[(2 * bh + bhi) * t_len..(2 * bh + bhi + 1) * t_len];
            // Stage in f64 through the same contract: t = c·N + r, gated, zero past T.
            let mut ux = vec![0f64; nl];
            for r in 0..n {
                for c in 0..l {
                    let t = c * n + r;
                    if t < t_len {
                        ux[r * l + c] = xb[t] as f64 * ig[t] as f64;
                    }
                }
            }
            let grid = ref_monarch_conv_f64(&ux, &mats, &kf[bhi * nl * 2..(bhi + 1) * nl * 2], n, l);
            let (mut md, mut sc) = (0f64, 1e-6f64);
            for r in 0..n {
                for c in 0..l {
                    let t = c * n + r;
                    if t < t_len {
                        let want =
                            (grid[r * l + c] + ux[r * l + c] * dvec[bhi % h] as f64) * og[t] as f64;
                        md = md.max((got[bhi * t_len + t] as f64 - want).abs());
                        sc = sc.max(want.abs());
                    }
                }
            }
            assert!(md / sc < 1e-4, "bh={bhi}: padded conv vs f64 rel {}", md / sc);
        }
    }

    #[test]
    fn row_idft_real_matches_f64() {
        let (bh, n, l) = (2usize, 3usize, 5usize);
        let x: Vec<f32> = (0..bh * n * l * 2).map(|i| rnd(i + 23)).collect();
        let idl: Vec<f32> = (0..l * l * 2).map(|i| rnd(i + 37)).collect();
        let mut got = vec![0f32; bh * n * l];
        row_idft_real(&x, &idl, &mut got, bh, n, l);
        for row in 0..bh * n {
            for lo in 0..l {
                let mut sr = 0f64;
                for k in 0..l {
                    sr += idl[(lo * l + k) * 2] as f64 * x[(row * l + k) * 2] as f64
                        - idl[(lo * l + k) * 2 + 1] as f64 * x[(row * l + k) * 2 + 1] as f64;
                }
                let want = sr / (n * l) as f64;
                let g = got[row * l + lo] as f64;
                assert!((g - want).abs() < 1e-6, "row={row} l={lo}: {g} vs {want}");
            }
        }
    }

    #[test]
    fn fused_batched_grid_dispatch_and_round_trip() {
        // Grid fan-out over a batch, then forward∘inverse recovers each signal — exercises the
        // rayon grid dispatch + the per-threadgroup barrier'd FFT end to end.
        let (batch, n) = (5usize, 32usize);
        let orig: Vec<f32> = (0..batch * 2 * n)
            .map(|i| ((i * 53 % 17) as f32 / 17.0) - 0.5)
            .collect();
        let mut d = orig.clone();
        fused_fft(&mut d, batch, false, 8); // forward, 8-lane simdgroups
        fused_fft(&mut d, batch, true, 4); // inverse, different lane count
        for (g, o) in d.iter().zip(&orig) {
            assert!((g - o).abs() < 1e-3, "round-trip drift {g} vs {o}");
        }
    }
}
