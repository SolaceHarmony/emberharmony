//! GPU dispatch model, simulated on the CPU. The zoo's single-stage ops (GEMM, reductions)
//! only need SIMD lanes + a rayon fan-out. The *multi-stage* Metal kernels — `FFTConv.metal`'s
//! `fft_radix2` (a `threadgroup_barrier` between every butterfly stage) and `fused_monarch`'s
//! row-DFT → twiddle → col-DFT (barriers + ping-pong through `threadgroup` memory) — need the
//! full model. This module reproduces it faithfully:
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

use std::sync::Barrier;

/// A `*mut f32` shared across the lane team — the CPU analog of `threadgroup` memory. Lanes
/// write disjoint indices within a barrier-delimited stage, so aliasing never occurs; the
/// `Barrier` between stages provides the fence that makes writes visible (as on the GPU).
#[derive(Clone, Copy)]
struct Shared(*mut f32);
// SAFETY: sharing is sound because (a) within a stage every lane touches a disjoint index set
// and (b) `Barrier` fences separate stages, so there is never a concurrent read+write or
// write+write to the same location — identical to GPU threadgroup-memory discipline.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    #[inline]
    unsafe fn get(self, i: usize) -> f32 {
        *self.0.add(i)
    }
    #[inline]
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

// One threadgroup: run the whole radix-2 FFT of `data` (interleaved [re,im], length 2*n) with a
// team of `lanes` workers synchronising at `threadgroup_barrier`s between every stage. This is a
// line-for-line CPU port of the `fft_radix2` in FFTConv.metal (bit-reverse, butterfly stages,
// optional inverse-scale) — each `barrier.wait()` is a `threadgroup_barrier`.
fn fft_threadgroup(data: &mut [f32], n: usize, inverse: bool, lanes: usize) {
    debug_assert_eq!(data.len(), 2 * n);
    if n <= 1 {
        return;
    }
    let log2n = n.trailing_zeros();
    let shared = Shared(data.as_mut_ptr());
    let barrier = Barrier::new(lanes);
    std::thread::scope(|scope| {
        for lane in 0..lanes {
            let barrier = &barrier;
            scope.spawn(move || {
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
                let sign = if inverse { 1.0f32 } else { -1.0f32 };
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
