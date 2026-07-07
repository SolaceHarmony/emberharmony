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
