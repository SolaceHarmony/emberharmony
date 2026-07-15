//! Fused tensor-core Monarch butterfly — a **single** Metal dispatch doing all three
//! forward phases (row-DFT → twiddle → col-DFT) per `(batch, head)` with
//! `simdgroup_matrix` sub-DFTs and the intermediate held in `threadgroup` memory (no
//! global round-trips), the Metal counterpart of the fused MLX
//! `monarch_metal/butterfly_forward_fused.py` and the FFT half of CUDA's
//! `monarch_conv_cuda` single launch.
//!
//! vs the three-CustomOp [`crate::butterfly_fft_forward`] (row/twiddle/col dispatched
//! separately, intermediates through device memory): same math, one launch, one
//! threadgroup per `(b,h)` tiling the `[N,L]` grid in `simdgroup_float8x8` 8×8 tiles —
//! so the command graph is encoded once and the `[N,L]` intermediate never leaves
//! threadgroup memory. f32; `N` and `L` must be multiples of 8 (the simdgroup tile).
//!
//! The full fused conv (`×k_f` + the inverse half) lands next on top of this; this
//! file proves the multi-stage threadgroup + tensor-core fusion against the crate's
//! already-verified un-fused forward.

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

#[cfg(feature = "metal")]
const SRC_FUSED_FWD: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Params {
    uint B; uint H; uint N; uint L;
    uint off_dLr; uint off_dLi; uint off_dNr; uint off_dNi; uint off_tw;
};

// One threadgroup (one simdgroup, 32 lanes) per (b,h). packed holds the separated
// real/imag DFT matrices dL[L,L], dN[N,N] and the interleaved twiddles tw[N,L,2].
kernel void monarch_fused_fwd_f32(
    constant Params&    p       [[buffer(0)]],
    device const float* u       [[buffer(1)]],   // [B,H,N,L] real
    device const float* packed  [[buffer(2)]],   // dLr|dLi|dNr|dNi|tw(interleaved)
    device float*       out      [[buffer(3)]],  // [B,H,N,L,2] complex
    threadgroup float*  sxr      [[threadgroup(0)]],   // [N*L]
    threadgroup float*  sxi      [[threadgroup(1)]],   // [N*L]
    threadgroup float*  scratch  [[threadgroup(2)]],   // [4*64]
    uint bh   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint N = p.N, L = p.L, NL = N * L;
    device const float* xb  = u + bh * NL;
    device float*       ob  = out + bh * NL * 2u;
    device const float* dLr = packed + p.off_dLr;
    device const float* dLi = packed + p.off_dLi;
    device const float* dNr = packed + p.off_dNr;
    device const float* dNi = packed + p.off_dNi;
    device const float* tw  = packed + p.off_tw;

    // stage 1: row DFT along L.  Y[N,L] = xb[N,L] @ dL[L,L]   (x real -> 2 real GEMMs)
    for (uint pr = 0u; pr < N; pr += 8u) {
        for (uint qc = 0u; qc < L; qc += 8u) {
            simdgroup_float8x8 ar = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 ai = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kt = 0u; kt < L; kt += 8u) {
                simdgroup_float8x8 xt, dr, di;
                simdgroup_load(xt, xb  + pr * L + kt, L);
                simdgroup_load(dr, dLr + kt * L + qc, L);
                simdgroup_load(di, dLi + kt * L + qc, L);
                simdgroup_multiply_accumulate(ar, xt, dr, ar);
                simdgroup_multiply_accumulate(ai, xt, di, ai);
            }
            simdgroup_store(ar, sxr + pr * L + qc, L);
            simdgroup_store(ai, sxi + pr * L + qc, L);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 2: twiddle (elementwise complex), in threadgroup memory
    for (uint i = lane; i < NL; i += 32u) {
        uint n = i / L, l = i % L;
        float zr = sxr[i], zi = sxi[i];
        float twr = tw[(n * L + l) * 2u], twi = tw[(n * L + l) * 2u + 1u];
        sxr[i] = zr * twr - zi * twi;
        sxi[i] = zr * twi + zi * twr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 3: col DFT along N.  O[N,L] = dN[N,N] @ Z[N,L]   (complex = 4 real GEMMs)
    for (uint pr = 0u; pr < N; pr += 8u) {
        for (uint qc = 0u; qc < L; qc += 8u) {
            simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // dNr@Zr
            simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // dNi@Zi
            simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // dNr@Zi
            simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // dNi@Zr
            for (uint kt = 0u; kt < N; kt += 8u) {
                simdgroup_float8x8 dr, di, zr, zi;
                simdgroup_load(dr, dNr + pr * N + kt, N);
                simdgroup_load(di, dNi + pr * N + kt, N);
                simdgroup_load(zr, sxr + kt * L + qc, L);
                simdgroup_load(zi, sxi + kt * L + qc, L);
                simdgroup_multiply_accumulate(m0, dr, zr, m0);
                simdgroup_multiply_accumulate(m1, di, zi, m1);
                simdgroup_multiply_accumulate(m2, dr, zi, m2);
                simdgroup_multiply_accumulate(m3, di, zr, m3);
            }
            simdgroup_store(m0, scratch + 0u,   8);
            simdgroup_store(m1, scratch + 64u,  8);
            simdgroup_store(m2, scratch + 128u, 8);
            simdgroup_store(m3, scratch + 192u, 8);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 64u; i += 32u) {
                uint r = i / 8u, c = i % 8u;
                uint o = ((pr + r) * L + qc + c) * 2u;
                ob[o]      = scratch[i]        - scratch[64u + i];    // O_r = dNr@Zr - dNi@Zi
                ob[o + 1u] = scratch[128u + i] + scratch[192u + i];   // O_i = dNr@Zi + dNi@Zr
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
"#;

/// Offsets (in f32 elements) of each block inside the `packed` buffer, given `n,l`.
/// Layout: `dLr[L,L] | dLi[L,L] | dNr[N,N] | dNi[N,N] | tw[N,L,2]` (tw interleaved).
const fn packed_offsets(n: usize, l: usize) -> (usize, usize, usize, usize, usize, usize) {
    let off_dlr = 0;
    let off_dli = l * l;
    let off_dnr = 2 * l * l;
    let off_dni = 2 * l * l + n * n;
    let off_tw = 2 * l * l + 2 * n * n;
    let total = off_tw + 2 * n * l;
    (off_dlr, off_dli, off_dnr, off_dni, off_tw, total)
}

fn contig_f32<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let data = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((start, end)) => Ok(&data[start..end]),
        None => candle_core::bail!("monarch fused fwd expects contiguous f32 inputs"),
    }
}

/// Fused forward Monarch butterfly. Inputs: `u` `[B,H,N,L]` (real) and `packed`
/// (the separated DFT matrices + interleaved twiddles, see [`pack_forward`]). Output:
/// `[B,H,N,L,2]` (complex) — identical to [`crate::butterfly_fft_forward`].
struct MonarchFusedForward;

impl CustomOp2 for MonarchFusedForward {
    fn name(&self) -> &'static str {
        "monarch_fused_fwd"
    }

    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ps: &CpuStorage,
        pl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l) = ul.shape().dims4()?;
        let u = contig_f32(us, ul)?;
        let packed = contig_f32(ps, pl)?;
        let (o_dlr, o_dli, o_dnr, o_dni, o_tw, total) = packed_offsets(n, l);
        if packed.len() != total {
            candle_core::bail!(
                "monarch fused fwd: packed len {} != expected {total}",
                packed.len()
            );
        }
        let (dlr, dli) = (&packed[o_dlr..o_dlr + l * l], &packed[o_dli..o_dli + l * l]);
        let (dnr, dni) = (&packed[o_dnr..o_dnr + n * n], &packed[o_dni..o_dni + n * n]);
        let tw = &packed[o_tw..o_tw + 2 * n * l];

        let mut out = vec![0f32; b * h * n * l * 2];
        let mut yr = vec![0f32; n * l];
        let mut yi = vec![0f32; n * l];
        for bh in 0..b * h {
            let xb = &u[bh * n * l..(bh + 1) * n * l];
            // stage 1: Y[ni,lp] = Σ_k xb[ni,k]·dL[k,lp]
            for ni in 0..n {
                for lp in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..l {
                        let xv = xb[ni * l + k];
                        sr += xv * dlr[k * l + lp];
                        si += xv * dli[k * l + lp];
                    }
                    yr[ni * l + lp] = sr;
                    yi[ni * l + lp] = si;
                }
            }
            // stage 2: twiddle
            for ni in 0..n {
                for li in 0..l {
                    let idx = ni * l + li;
                    let (zr, zi) = (yr[idx], yi[idx]);
                    let (twr, twi) = (tw[(ni * l + li) * 2], tw[(ni * l + li) * 2 + 1]);
                    yr[idx] = zr * twr - zi * twi;
                    yi[idx] = zr * twi + zi * twr;
                }
            }
            // stage 3: O[np,l] = Σ_k dN[np,k]·Z[k,l]
            for np in 0..n {
                for li in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (dnr[np * n + k], dni[np * n + k]);
                        let (zr, zi) = (yr[k * l + li], yi[k * l + li]);
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    let o = (bh * n * l + np * l + li) * 2;
                    out[o] = sr;
                    out[o + 1] = si;
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l, 2))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ps: &candle_core::MetalStorage,
        pl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::MTLSize;

        let (b, h, n, l) = ul.shape().dims4()?;
        if n % 8 != 0 || l % 8 != 0 {
            candle_core::bail!(
                "monarch fused fwd metal: N,L must be multiples of 8 (got N={n} L={l})"
            );
        }
        let (o_dlr, o_dli, o_dnr, o_dni, o_tw, _total) = packed_offsets(n, l);

        #[repr(C)]
        struct Params {
            b: u32,
            h: u32,
            n: u32,
            l: u32,
            off_dlr: u32,
            off_dli: u32,
            off_dnr: u32,
            off_dni: u32,
            off_tw: u32,
        }
        let params = Params {
            b: b as u32,
            h: h as u32,
            n: n as u32,
            l: l as u32,
            off_dlr: o_dlr as u32,
            off_dli: o_dli as u32,
            off_dnr: o_dnr as u32,
            off_dni: o_dni as u32,
            off_tw: o_tw as u32,
        };

        let dev = us.device();
        let p = crate::metal_util::pipeline(dev, "monarch_fused_fwd_f32", SRC_FUSED_FWD)?;
        let out_el = b * h * n * l * 2;
        let out = dev.new_buffer(out_el, DType::F32, "monarch_fused_fwd")?;
        let dts = DType::F32.size_in_bytes();

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), ul.start_offset() * dts);
        enc.set_buffer(2, Some(ps.buffer()), pl.start_offset() * dts);
        enc.set_buffer(3, Some(&*out), 0);
        // dynamic threadgroup memory: sxr[N*L], sxi[N*L], scratch[4*64]
        enc.set_threadgroup_memory_length(0, n * l * dts);
        enc.set_threadgroup_memory_length(1, n * l * dts);
        enc.set_threadgroup_memory_length(2, 4 * 64 * dts);
        // one threadgroup per (b,h); one simdgroup (32 lanes) each.
        enc.dispatch_thread_groups(
            MTLSize {
                width: b * h,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), out_el, DType::F32),
            Shape::from((b, h, n, l, 2)),
        ))
    }
}

/// Build the `packed` buffer for [`butterfly_fft_forward_fused`] from the DFT matrices
/// `d_f_n` `[N,N,2]`, `d_f_l` `[L,L,2]` and twiddles `[N,L,2]` (same convention as
/// [`crate::fft_matrix`] / [`crate::twiddle_factors_fft`]): the matrices are split into
/// contiguous real/imag planes and concatenated, twiddles kept interleaved.
fn pack_forward(d_f_n: &Tensor, d_f_l: &Tensor, twiddles: &Tensor) -> Result<Tensor> {
    let (n, _, _) = d_f_n.dims3()?;
    let (l, _, _) = d_f_l.dims3()?;
    let dlr = d_f_l.narrow(2, 0, 1)?.reshape((l * l,))?;
    let dli = d_f_l.narrow(2, 1, 1)?.reshape((l * l,))?;
    let dnr = d_f_n.narrow(2, 0, 1)?.reshape((n * n,))?;
    let dni = d_f_n.narrow(2, 1, 1)?.reshape((n * n,))?;
    let tw = twiddles.reshape((2 * n * l,))?;
    Tensor::cat(&[&dlr, &dli, &dnr, &dni, &tw], 0)
}

/// Fused forward Monarch butterfly FFT — drop-in for [`crate::butterfly_fft_forward`]
/// (`x` `[B,H,N,L]` real → `[B,H,N,L,2]` complex), but the whole transform is one
/// tiled `simdgroup_matrix` Metal dispatch instead of three. `N,L` must be multiples
/// of 8 on Metal; the CPU reference has no such restriction.
pub fn butterfly_fft_forward_fused(
    x: &Tensor,
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let packed = pack_forward(
        &d_f_n.contiguous()?,
        &d_f_l.contiguous()?,
        &twiddles.contiguous()?,
    )?;
    x.apply_op2(&packed, MonarchFusedForward)
}

// ===========================================================================
// Full fused Monarch convolution: FFT → ×k_f → IFFT in ONE tiled kernel.
//
// One threadgroup per (b,h) runs all seven stages in threadgroup memory with
// simdgroup_matrix GEMMs (fp32 accumulate), ping-ponging between complex buffers A
// and B so the col-DFT stages (which read every row) never overwrite their own input:
//   1 row-DFT/L (u@dL)→A   2 twiddle A   3 col-DFT/N (dN@A)→B   4 ×k_f on B
//   5 col-IDFT/N (idN@B)→A 6 conj-twiddle A (itw)   7 row-IDFT/L real ×1/(N·L)→out
// Edge tiles: the matrices are zero-padded to multiples of 8 at pack time and the
// [N,L] intermediate lives in a padded [Np,Lp] threadgroup space, so every 8×8
// simdgroup_load is in-bounds; only the u-stage, ×k_f read, and output write touch
// the ragged [N,L] boundary. Drop-in for the un-fused `monarch_conv`.
// ===========================================================================

#[cfg(feature = "metal")]
const SRC_FUSED_CONV: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct ConvParams {
    uint B; uint H; uint N; uint L; uint Np; uint Lp;
    uint off_dLr; uint off_dLi; uint off_dNr; uint off_dNi; uint off_tw;
    uint off_idNr; uint off_idNi; uint off_idLr; uint off_idLi; uint off_itw;
};

kernel void monarch_fused_conv_f32(
    constant ConvParams& p    [[buffer(0)]],
    device const float*  u    [[buffer(1)]],   // [B,H,N,L] real
    device const float*  packed [[buffer(2)]], // dLr|dLi|dNr|dNi|tw|idNr|idNi|idLr|idLi|itw (padded)
    device const float*  kf   [[buffer(3)]],   // [B,H,N,L,2] complex (broadcast on host)
    device float*        out   [[buffer(4)]],  // [B,H,N,L] real
    threadgroup float*   ux   [[threadgroup(0)]],   // [Np*Lp] staged real input
    threadgroup float*   axr  [[threadgroup(1)]],   // [Np*Lp] buffer A real
    threadgroup float*   axi  [[threadgroup(2)]],   // [Np*Lp] buffer A imag
    threadgroup float*   bxr  [[threadgroup(3)]],   // [Np*Lp] buffer B real
    threadgroup float*   bxi  [[threadgroup(4)]],   // [Np*Lp] buffer B imag
    threadgroup float*   scratch [[threadgroup(5)]],// [4*64] tile scratch
    uint bh   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint N = p.N, L = p.L, Np = p.Np, Lp = p.Lp, NL = N * L, NpLp = Np * Lp;
    device const float* xb  = u  + bh * NL;
    device const float* kfb = kf + bh * NL * 2u;
    device float*       ob  = out + bh * NL;
    device const float* dLr = packed + p.off_dLr;
    device const float* dLi = packed + p.off_dLi;
    device const float* dNr = packed + p.off_dNr;
    device const float* dNi = packed + p.off_dNi;
    device const float* tw  = packed + p.off_tw;
    device const float* idNr= packed + p.off_idNr;
    device const float* idNi= packed + p.off_idNi;
    device const float* idLr= packed + p.off_idLr;
    device const float* idLi= packed + p.off_idLi;
    device const float* itw = packed + p.off_itw;

    // preamble: stage u[N,L] -> ux[Np,Lp] with zero-fill in the padding.
    for (uint i = lane; i < NpLp; i += 32u) {
        uint r = i / Lp, c = i % Lp;
        ux[i] = (r < N && c < L) ? xb[r * L + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 1: row DFT / L.  A[Np,Lp] = ux @ dL
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 ar = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 ai = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Lp; kt += 8u) {
            simdgroup_float8x8 xt, dr, di;
            simdgroup_load(xt, ux  + pr * Lp + kt, Lp);
            simdgroup_load(dr, dLr + kt * Lp + qc, Lp);
            simdgroup_load(di, dLi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(ar, xt, dr, ar);
            simdgroup_multiply_accumulate(ai, xt, di, ai);
        }
        simdgroup_store(ar, axr + pr * Lp + qc, Lp);
        simdgroup_store(ai, axi + pr * Lp + qc, Lp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 2: forward twiddle (A *= tw) over the padded grid (padding tw=0 keeps 0).
    for (uint i = lane; i < NpLp; i += 32u) {
        uint n = i / Lp, l = i % Lp;
        float zr = axr[i], zi = axi[i];
        float twr = tw[(n * Lp + l) * 2u], twi = tw[(n * Lp + l) * 2u + 1u];
        axr[i] = zr * twr - zi * twi;
        axi[i] = zr * twi + zi * twr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 3: col DFT / N.  B[Np,Lp] = dN @ A  (complex = 4 real GEMMs)
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Np; kt += 8u) {
            simdgroup_float8x8 dr, di, zr, zi;
            simdgroup_load(dr, dNr + pr * Np + kt, Np);
            simdgroup_load(di, dNi + pr * Np + kt, Np);
            simdgroup_load(zr, axr + kt * Lp + qc, Lp);
            simdgroup_load(zi, axi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, dr, zr, m0);
            simdgroup_multiply_accumulate(m1, di, zi, m1);
            simdgroup_multiply_accumulate(m2, dr, zi, m2);
            simdgroup_multiply_accumulate(m3, di, zr, m3);
        }
        simdgroup_store(m0, scratch + 0u,   8); simdgroup_store(m1, scratch + 64u,  8);
        simdgroup_store(m2, scratch + 128u, 8); simdgroup_store(m3, scratch + 192u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint o = (pr + r) * Lp + qc + c;
            bxr[o] = scratch[i] - scratch[64u + i];
            bxi[o] = scratch[128u + i] + scratch[192u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // stage 4: × k_f  (B *= k_f), valid [N,L] positions only (padding of B stays 0).
    for (uint i = lane; i < NL; i += 32u) {
        uint n = i / L, l = i % L; uint bo = n * Lp + l;
        float zr = bxr[bo], zi = bxi[bo];
        float kr = kfb[i * 2u], ki = kfb[i * 2u + 1u];
        bxr[bo] = zr * kr - zi * ki;
        bxi[bo] = zr * ki + zi * kr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 5: col IDFT / N.  A[Np,Lp] = idN @ B
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Np; kt += 8u) {
            simdgroup_float8x8 dr, di, zr, zi;
            simdgroup_load(dr, idNr + pr * Np + kt, Np);
            simdgroup_load(di, idNi + pr * Np + kt, Np);
            simdgroup_load(zr, bxr + kt * Lp + qc, Lp);
            simdgroup_load(zi, bxi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, dr, zr, m0);
            simdgroup_multiply_accumulate(m1, di, zi, m1);
            simdgroup_multiply_accumulate(m2, dr, zi, m2);
            simdgroup_multiply_accumulate(m3, di, zr, m3);
        }
        simdgroup_store(m0, scratch + 0u,   8); simdgroup_store(m1, scratch + 64u,  8);
        simdgroup_store(m2, scratch + 128u, 8); simdgroup_store(m3, scratch + 192u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint o = (pr + r) * Lp + qc + c;
            axr[o] = scratch[i] - scratch[64u + i];
            axi[o] = scratch[128u + i] + scratch[192u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // stage 6: conjugate twiddle (A *= itw) over the padded grid.
    for (uint i = lane; i < NpLp; i += 32u) {
        uint n = i / Lp, l = i % Lp;
        float zr = axr[i], zi = axi[i];
        float twr = itw[(n * Lp + l) * 2u], twi = itw[(n * Lp + l) * 2u + 1u];
        axr[i] = zr * twr - zi * twi;
        axi[i] = zr * twi + zi * twr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 7: row IDFT / L, real part × 1/(N·L).  out = Re(A @ idL) · scale
    float scale = 1.0f / (float)(N * L);
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // Ar@idLr
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); // Ai@idLi
        for (uint kt = 0u; kt < Lp; kt += 8u) {
            simdgroup_float8x8 ar, ai, dr, di;
            simdgroup_load(ar, axr  + pr * Lp + kt, Lp);
            simdgroup_load(ai, axi  + pr * Lp + kt, Lp);
            simdgroup_load(dr, idLr + kt * Lp + qc, Lp);
            simdgroup_load(di, idLi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, ar, dr, m0);
            simdgroup_multiply_accumulate(m1, ai, di, m1);
        }
        simdgroup_store(m0, scratch + 0u, 8); simdgroup_store(m1, scratch + 64u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint gr = pr + r, gc = qc + c;
            if (gr < N && gc < L) { ob[gr * L + gc] = (scratch[i] - scratch[64u + i]) * scale; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Padded-`packed` layout for the full conv: every matrix zero-padded to a multiple of
/// 8 (`Np=ceil8(N)`, `Lp=ceil8(L)`) so the in-kernel `simdgroup` tiles are always
/// in-bounds. Block order: `dLr|dLi|dNr|dNi|tw|idNr|idNi|idLr|idLi|itw` (matrices
/// real/imag separated; twiddles interleaved).
struct PackLayout {
    np: usize,
    lp: usize,
    dlr: usize,
    dli: usize,
    dnr: usize,
    dni: usize,
    tw: usize,
    idnr: usize,
    idni: usize,
    idlr: usize,
    idli: usize,
    itw: usize,
    total: usize,
}

fn pack_layout(n: usize, l: usize) -> PackLayout {
    let np = n.div_ceil(8) * 8;
    let lp = l.div_ceil(8) * 8;
    let (ll, nn, twn) = (lp * lp, np * np, np * lp * 2);
    let dlr = 0;
    let dli = dlr + ll;
    let dnr = dli + ll;
    let dni = dnr + nn;
    let tw = dni + nn;
    let idnr = tw + twn;
    let idni = idnr + nn;
    let idlr = idni + nn;
    let idli = idlr + ll;
    let itw = idli + ll;
    let total = itw + twn;
    PackLayout {
        np,
        lp,
        dlr,
        dli,
        dnr,
        dni,
        tw,
        idnr,
        idni,
        idlr,
        idli,
        itw,
        total,
    }
}

/// Zero-pad a `[d,d]` matrix tensor up to `[dp,dp]` (bottom/right).
fn pad_square(m: &Tensor, dp: usize) -> Result<Tensor> {
    let (r, c) = m.dims2()?;
    m.pad_with_zeros(0, 0, dp - r)?.pad_with_zeros(1, 0, dp - c)
}

/// Build the padded `packed` buffer from the six DFT/twiddle matrices (same convention
/// as [`crate::fft_matrix`]/[`crate::ifft_matrix`]/[`crate::twiddle_factors_fft`]/`_ifft`).
fn pack_full(
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
    id_f_n: &Tensor,
    id_f_l: &Tensor,
    ifft_twiddles: &Tensor,
) -> Result<Tensor> {
    let (n, _, _) = d_f_n.dims3()?;
    let (l, _, _) = d_f_l.dims3()?;
    let lay = pack_layout(n, l);
    let (np, lp) = (lay.np, lay.lp);
    // split real/imag plane of an [d,d,2] matrix, pad to [dp,dp], flatten.
    let mat = |m: &Tensor, idx: usize, d: usize, dp: usize| -> Result<Tensor> {
        pad_square(&m.narrow(2, idx, 1)?.contiguous()?.reshape((d, d))?, dp)?.reshape((dp * dp,))
    };
    // pad [N,L,2] twiddles to [Np,Lp,2], flatten (interleaved).
    let tw = |t: &Tensor| -> Result<Tensor> {
        t.pad_with_zeros(0, 0, np - n)?
            .pad_with_zeros(1, 0, lp - l)?
            .contiguous()?
            .reshape((np * lp * 2,))
    };
    let blocks = [
        mat(d_f_l, 0, l, lp)?,
        mat(d_f_l, 1, l, lp)?,
        mat(d_f_n, 0, n, np)?,
        mat(d_f_n, 1, n, np)?,
        tw(twiddles)?,
        mat(id_f_n, 0, n, np)?,
        mat(id_f_n, 1, n, np)?,
        mat(id_f_l, 0, l, lp)?,
        mat(id_f_l, 1, l, lp)?,
        tw(ifft_twiddles)?,
    ];
    Tensor::cat(&blocks.iter().collect::<Vec<_>>(), 0)
}

struct MonarchFusedConv;

impl CustomOp3 for MonarchFusedConv {
    fn name(&self) -> &'static str {
        "monarch_fused_conv"
    }

    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ps: &CpuStorage,
        pl: &Layout,
        ks: &CpuStorage,
        kl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, h, n, l) = ul.shape().dims4()?;
        let u = contig_f32(us, ul)?;
        let packed = contig_f32(ps, pl)?;
        let kf = contig_f32(ks, kl)?;
        let lay = pack_layout(n, l);
        let (np, lp) = (lay.np, lay.lp);
        if packed.len() != lay.total {
            candle_core::bail!(
                "monarch fused conv: packed len {} != expected {}",
                packed.len(),
                lay.total
            );
        }
        let scale = 1.0f32 / (n * l) as f32;
        let mut out = vec![0f32; b * h * n * l];
        let (mut ar, mut ai) = (vec![0f32; n * l], vec![0f32; n * l]); // buffer A
        let (mut br, mut bi) = (vec![0f32; n * l], vec![0f32; n * l]); // buffer B
        for bh in 0..b * h {
            let xb = &u[bh * n * l..(bh + 1) * n * l];
            let kfb = &kf[bh * n * l * 2..(bh + 1) * n * l * 2];
            // 1: A = u @ dL
            for ni in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..l {
                        let xv = xb[ni * l + k];
                        sr += xv * packed[lay.dlr + k * lp + lo];
                        si += xv * packed[lay.dli + k * lp + lo];
                    }
                    ar[ni * l + lo] = sr;
                    ai[ni * l + lo] = si;
                }
            }
            // 2: forward twiddle
            for ni in 0..n {
                for lo in 0..l {
                    let idx = ni * l + lo;
                    let (zr, zi) = (ar[idx], ai[idx]);
                    let (twr, twi) = (
                        packed[lay.tw + (ni * lp + lo) * 2],
                        packed[lay.tw + (ni * lp + lo) * 2 + 1],
                    );
                    ar[idx] = zr * twr - zi * twi;
                    ai[idx] = zr * twi + zi * twr;
                }
            }
            // 3: B = dN @ A
            for np_ in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (
                            packed[lay.dnr + np_ * np + k],
                            packed[lay.dni + np_ * np + k],
                        );
                        let (zr, zi) = (ar[k * l + lo], ai[k * l + lo]);
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    br[np_ * l + lo] = sr;
                    bi[np_ * l + lo] = si;
                }
            }
            // 4: × k_f
            for i in 0..n * l {
                let (zr, zi) = (br[i], bi[i]);
                let (kr, ki) = (kfb[i * 2], kfb[i * 2 + 1]);
                br[i] = zr * kr - zi * ki;
                bi[i] = zr * ki + zi * kr;
            }
            // 5: A = idN @ B
            for np_ in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (
                            packed[lay.idnr + np_ * np + k],
                            packed[lay.idni + np_ * np + k],
                        );
                        let (zr, zi) = (br[k * l + lo], bi[k * l + lo]);
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    ar[np_ * l + lo] = sr;
                    ai[np_ * l + lo] = si;
                }
            }
            // 6: conjugate twiddle (itw)
            for ni in 0..n {
                for lo in 0..l {
                    let idx = ni * l + lo;
                    let (zr, zi) = (ar[idx], ai[idx]);
                    let (twr, twi) = (
                        packed[lay.itw + (ni * lp + lo) * 2],
                        packed[lay.itw + (ni * lp + lo) * 2 + 1],
                    );
                    ar[idx] = zr * twr - zi * twi;
                    ai[idx] = zr * twi + zi * twr;
                }
            }
            // 7: out = Re(A @ idL) × scale
            for ni in 0..n {
                for lo in 0..l {
                    let mut sr = 0f32;
                    for k in 0..l {
                        sr += ar[ni * l + k] * packed[lay.idlr + k * lp + lo]
                            - ai[ni * l + k] * packed[lay.idli + k * lp + lo];
                    }
                    out[bh * n * l + ni * l + lo] = sr * scale;
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, n, l))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ps: &candle_core::MetalStorage,
        pl: &Layout,
        ks: &candle_core::MetalStorage,
        kl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::MTLSize;

        let (b, h, n, l) = ul.shape().dims4()?;
        let lay = pack_layout(n, l);
        let (np, lp) = (lay.np, lay.lp);
        let dts = DType::F32.size_in_bytes();
        let tg_bytes = (5 * np * lp + 4 * 64) * dts;
        if tg_bytes > 32768 {
            candle_core::bail!("monarch fused conv: threadgroup mem {tg_bytes}B exceeds 32KB (N={n} L={l}); factor smaller");
        }

        #[repr(C)]
        struct ConvParams {
            b: u32,
            h: u32,
            n: u32,
            l: u32,
            np: u32,
            lp: u32,
            off_dlr: u32,
            off_dli: u32,
            off_dnr: u32,
            off_dni: u32,
            off_tw: u32,
            off_idnr: u32,
            off_idni: u32,
            off_idlr: u32,
            off_idli: u32,
            off_itw: u32,
        }
        let params = ConvParams {
            b: b as u32,
            h: h as u32,
            n: n as u32,
            l: l as u32,
            np: np as u32,
            lp: lp as u32,
            off_dlr: lay.dlr as u32,
            off_dli: lay.dli as u32,
            off_dnr: lay.dnr as u32,
            off_dni: lay.dni as u32,
            off_tw: lay.tw as u32,
            off_idnr: lay.idnr as u32,
            off_idni: lay.idni as u32,
            off_idlr: lay.idlr as u32,
            off_idli: lay.idli as u32,
            off_itw: lay.itw as u32,
        };

        let dev = us.device();
        let p = crate::metal_util::pipeline(dev, "monarch_fused_conv_f32", SRC_FUSED_CONV)?;
        let out_el = b * h * n * l;
        let out = dev.new_buffer(out_el, DType::F32, "monarch_fused_conv")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), ul.start_offset() * dts);
        enc.set_buffer(2, Some(ps.buffer()), pl.start_offset() * dts);
        enc.set_buffer(3, Some(ks.buffer()), kl.start_offset() * dts);
        enc.set_buffer(4, Some(&*out), 0);
        let nplp = np * lp * dts;
        enc.set_threadgroup_memory_length(0, nplp); // ux
        enc.set_threadgroup_memory_length(1, nplp); // axr
        enc.set_threadgroup_memory_length(2, nplp); // axi
        enc.set_threadgroup_memory_length(3, nplp); // bxr
        enc.set_threadgroup_memory_length(4, nplp); // bxi
        enc.set_threadgroup_memory_length(5, 4 * 64 * dts); // scratch
        enc.dispatch_thread_groups(
            MTLSize {
                width: b * h,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), out_el, DType::F32),
            Shape::from((b, h, n, l)),
        ))
    }
}

/// Full fused Monarch convolution `IFFT(FFT(u) ⊙ k_f)` in ONE tiled `simdgroup_matrix`
/// dispatch — drop-in for [`crate::monarch_conv`]. `u` `[B,H,N,L]` real; `k_f`
/// `[…,N,L,2]` is the filter's forward FFT (broadcast over the batch). Any `N,L` (edge
/// tiles are zero-filled in-kernel, no caller padding); fp32.
#[allow(clippy::too_many_arguments)]
pub fn monarch_conv_fused(
    u: &Tensor,
    k_f: &Tensor,
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
    id_f_n: &Tensor,
    id_f_l: &Tensor,
    ifft_twiddles: &Tensor,
) -> Result<Tensor> {
    let (b, h, n, l) = u.dims4()?;
    let u = u.contiguous()?;
    let k_f = k_f.broadcast_as((b, h, n, l, 2))?.contiguous()?;
    let packed = pack_full(
        &d_f_n.contiguous()?,
        &d_f_l.contiguous()?,
        &twiddles.contiguous()?,
        &id_f_n.contiguous()?,
        &id_f_l.contiguous()?,
        &ifft_twiddles.contiguous()?,
    )?;
    u.apply_op3(&packed, &k_f, MonarchFusedConv)
}

// ===========================================================================
// PADDED (+ optionally GATED) fused Monarch convolution — the causal-linear-conv
// variant of `monarch_conv_fused`, on the same simdgroup_matrix core.
//
// FFT convolution is CIRCULAR at the transform length M = N·L. For the causal
// LINEAR convolution a sequence model needs (CUDA monarch_conv's padded path /
// the MLX `butterfly_*_padded*` oracle family), the length-T signal occupies the
// FIRST T positions of the M-point transform (T ≤ M, typically M ≥ 2T so wrap
// never reaches a valid output), the rest is zero-fill, and only the first T
// outputs are stored. Gating (the Hyena/H3 elementwise gates the CUDA kernels
// fuse) is applied at the same two touch points: input gate at load, output gate
// at store — so neither gate ever materializes a gated tensor in device memory.
//
// Time order: t = c·N + r over the [N,L] grid — the 4-step COLUMN-major
// flattening (`reshape(transpose(x))` in the MLX oracle's accuracy test), the
// same layout `monarch_conv`'s circular equivalence holds in. Pinned down by a
// delta-shift probe and the linear-conv ground-truth test below.
// ===========================================================================

#[cfg(feature = "metal")]
const SRC_FUSED_CONV_PADDED: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct PaddedConvParams {
    uint B; uint H; uint N; uint L; uint Np; uint Lp;
    uint off_dLr; uint off_dLi; uint off_dNr; uint off_dNi; uint off_tw;
    uint off_idNr; uint off_idNi; uint off_idLr; uint off_idLi; uint off_itw;
    uint T; uint gates; // bit0 input gate, bit1 output gate, bit2 u·D skip term
    uint off_d;         // packed offset of D[H] when gates bit2 is set
};

kernel void monarch_fused_conv_padded_f32(
    constant PaddedConvParams& p [[buffer(0)]],
    device const float*  u_ext  [[buffer(1)]],  // [G,B,H,T]: u | in_gate? | out_gate?
    device const float*  packed [[buffer(2)]],  // dLr|dLi|dNr|dNi|tw|idNr|idNi|idLr|idLi|itw (padded)
    device const float*  kf     [[buffer(3)]],  // [B,H,N,L,2] complex (broadcast on host)
    device float*        out    [[buffer(4)]],  // [B,H,T] real
    threadgroup float*   ux   [[threadgroup(0)]],
    threadgroup float*   axr  [[threadgroup(1)]],
    threadgroup float*   axi  [[threadgroup(2)]],
    threadgroup float*   bxr  [[threadgroup(3)]],
    threadgroup float*   bxi  [[threadgroup(4)]],
    threadgroup float*   scratch [[threadgroup(5)]],
    uint bh   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint N = p.N, L = p.L, Np = p.Np, Lp = p.Lp, NL = N * L, NpLp = Np * Lp;
    uint T = p.T, BH = p.B * p.H;
    device const float* xb  = u_ext + bh * T; // slot 0
    device const float* ig  = u_ext + (BH + bh) * T; // slot 1 (valid iff gates&1)
    uint og_slot = (p.gates & 1u) ? 2u : 1u;
    device const float* og  = u_ext + (og_slot * BH + bh) * T; // valid iff gates&2
    device const float* kfb = kf + bh * NL * 2u;
    device float*       ob  = out + bh * T;
    device const float* dLr = packed + p.off_dLr;
    device const float* dLi = packed + p.off_dLi;
    device const float* dNr = packed + p.off_dNr;
    device const float* dNi = packed + p.off_dNi;
    device const float* tw  = packed + p.off_tw;
    device const float* idNr= packed + p.off_idNr;
    device const float* idNi= packed + p.off_idNi;
    device const float* idLr= packed + p.off_idLr;
    device const float* idLi= packed + p.off_idLi;
    device const float* itw = packed + p.off_itw;
    device const float* dvec = packed + p.off_d; // D[H], valid iff gates bit2

    // preamble: stage the length-T signal into ux[Np,Lp]. Time order is the
    // 4-step column-major flattening t = c*N + r (the transpose-flatten the
    // oracle accuracy test uses); input gate fused at load, zero-fill past T
    // and in tile padding.
    for (uint i = lane; i < NpLp; i += 32u) {
        uint r = i / Lp, c = i % Lp;
        float v = 0.0f;
        if (r < N && c < L) {
            uint t = c * N + r;
            if (t < T) {
                v = xb[t];
                if (p.gates & 1u) { v *= ig[t]; }
            }
        }
        ux[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 1: row DFT / L.  A[Np,Lp] = ux @ dL
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 ar = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 ai = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Lp; kt += 8u) {
            simdgroup_float8x8 xt, dr, di;
            simdgroup_load(xt, ux  + pr * Lp + kt, Lp);
            simdgroup_load(dr, dLr + kt * Lp + qc, Lp);
            simdgroup_load(di, dLi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(ar, xt, dr, ar);
            simdgroup_multiply_accumulate(ai, xt, di, ai);
        }
        simdgroup_store(ar, axr + pr * Lp + qc, Lp);
        simdgroup_store(ai, axi + pr * Lp + qc, Lp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 2: forward twiddle
    for (uint i = lane; i < NpLp; i += 32u) {
        uint n = i / Lp, l = i % Lp;
        float zr = axr[i], zi = axi[i];
        float twr = tw[(n * Lp + l) * 2u], twi = tw[(n * Lp + l) * 2u + 1u];
        axr[i] = zr * twr - zi * twi;
        axi[i] = zr * twi + zi * twr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 3: col DFT / N.  B[Np,Lp] = dN @ A
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Np; kt += 8u) {
            simdgroup_float8x8 dr, di, zr, zi;
            simdgroup_load(dr, dNr + pr * Np + kt, Np);
            simdgroup_load(di, dNi + pr * Np + kt, Np);
            simdgroup_load(zr, axr + kt * Lp + qc, Lp);
            simdgroup_load(zi, axi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, dr, zr, m0);
            simdgroup_multiply_accumulate(m1, di, zi, m1);
            simdgroup_multiply_accumulate(m2, dr, zi, m2);
            simdgroup_multiply_accumulate(m3, di, zr, m3);
        }
        simdgroup_store(m0, scratch + 0u,   8); simdgroup_store(m1, scratch + 64u,  8);
        simdgroup_store(m2, scratch + 128u, 8); simdgroup_store(m3, scratch + 192u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint o = (pr + r) * Lp + qc + c;
            bxr[o] = scratch[i] - scratch[64u + i];
            bxi[o] = scratch[128u + i] + scratch[192u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // stage 4: × k_f, valid [N,L] positions only
    for (uint i = lane; i < NL; i += 32u) {
        uint n = i / L, l = i % L; uint bo = n * Lp + l;
        float zr = bxr[bo], zi = bxi[bo];
        float kr = kfb[i * 2u], ki = kfb[i * 2u + 1u];
        bxr[bo] = zr * kr - zi * ki;
        bxi[bo] = zr * ki + zi * kr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 5: col IDFT / N.  A[Np,Lp] = idN @ B
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Np; kt += 8u) {
            simdgroup_float8x8 dr, di, zr, zi;
            simdgroup_load(dr, idNr + pr * Np + kt, Np);
            simdgroup_load(di, idNi + pr * Np + kt, Np);
            simdgroup_load(zr, bxr + kt * Lp + qc, Lp);
            simdgroup_load(zi, bxi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, dr, zr, m0);
            simdgroup_multiply_accumulate(m1, di, zi, m1);
            simdgroup_multiply_accumulate(m2, dr, zi, m2);
            simdgroup_multiply_accumulate(m3, di, zr, m3);
        }
        simdgroup_store(m0, scratch + 0u,   8); simdgroup_store(m1, scratch + 64u,  8);
        simdgroup_store(m2, scratch + 128u, 8); simdgroup_store(m3, scratch + 192u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint o = (pr + r) * Lp + qc + c;
            axr[o] = scratch[i] - scratch[64u + i];
            axi[o] = scratch[128u + i] + scratch[192u + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // stage 6: conjugate twiddle
    for (uint i = lane; i < NpLp; i += 32u) {
        uint n = i / Lp, l = i % Lp;
        float zr = axr[i], zi = axi[i];
        float twr = itw[(n * Lp + l) * 2u], twi = itw[(n * Lp + l) * 2u + 1u];
        axr[i] = zr * twr - zi * twi;
        axi[i] = zr * twi + zi * twr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 7: row IDFT / L, real × 1/(N·L); store ONLY t < T, output gate fused.
    float scale = 1.0f / (float)(N * L);
    for (uint pr = 0u; pr < Np; pr += 8u) for (uint qc = 0u; qc < Lp; qc += 8u) {
        simdgroup_float8x8 m0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 m1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint kt = 0u; kt < Lp; kt += 8u) {
            simdgroup_float8x8 ar, ai, dr, di;
            simdgroup_load(ar, axr  + pr * Lp + kt, Lp);
            simdgroup_load(ai, axi  + pr * Lp + kt, Lp);
            simdgroup_load(dr, idLr + kt * Lp + qc, Lp);
            simdgroup_load(di, idLi + kt * Lp + qc, Lp);
            simdgroup_multiply_accumulate(m0, ar, dr, m0);
            simdgroup_multiply_accumulate(m1, ai, di, m1);
        }
        simdgroup_store(m0, scratch + 0u, 8); simdgroup_store(m1, scratch + 64u, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u; uint gr = pr + r, gc = qc + c;
            if (gr < N && gc < L) {
                uint t = gc * N + gr;
                if (t < T) {
                    float v = (scratch[i] - scratch[64u + i]) * scale;
                    // u·D delta-tap skip: ux still holds the STAGED (gated) input —
                    // untouched since the preamble — so the skip is one multiply.
                    if (p.gates & 4u) { v += ux[gr * Lp + gc] * dvec[bh % p.H]; }
                    if (p.gates & 2u) { v *= og[t]; }
                    ob[t] = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

struct MonarchFusedConvPadded {
    n: usize,
    l: usize,
    gates: u32,
}

impl CustomOp3 for MonarchFusedConvPadded {
    fn name(&self) -> &'static str {
        "monarch_fused_conv_padded"
    }

    fn cpu_fwd(
        &self,
        us: &CpuStorage,
        ul: &Layout,
        ps: &CpuStorage,
        pl: &Layout,
        ks: &CpuStorage,
        kl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (g, b, h, t_len) = ul.shape().dims4()?;
        let (n, l) = (self.n, self.l);
        let u_ext = contig_f32(us, ul)?;
        let packed = contig_f32(ps, pl)?;
        let kf = contig_f32(ks, kl)?;
        let lay = pack_layout(n, l);
        let (np, lp) = (lay.np, lay.lp);
        let expected_packed = lay.total + if self.gates & 4 != 0 { h } else { 0 };
        if packed.len() != expected_packed {
            candle_core::bail!(
                "monarch fused conv padded: packed len {} != expected {expected_packed}",
                packed.len()
            );
        }
        let expected_g = 1 + (self.gates & 1) as usize + ((self.gates >> 1) & 1) as usize;
        if g != expected_g {
            candle_core::bail!(
                "monarch fused conv padded: {g} input slots, gates flags expect {expected_g}"
            );
        }
        let has_d = self.gates & 4 != 0;
        let d_off = lay.total;
        let bh_stride = b * h * t_len;
        let og_slot = if self.gates & 1 != 0 { 2 } else { 1 };
        let scale = 1.0f32 / (n * l) as f32;
        let mut out = vec![0f32; b * h * t_len];
        let mut x = vec![0f32; n * l]; // padded, gated input
        let (mut ar, mut ai) = (vec![0f32; n * l], vec![0f32; n * l]);
        let (mut br, mut bi) = (vec![0f32; n * l], vec![0f32; n * l]);
        for bh in 0..b * h {
            let xb = &u_ext[bh * t_len..(bh + 1) * t_len];
            let kfb = &kf[bh * n * l * 2..(bh + 1) * n * l * 2];
            // preamble: pad to [n*l]; time order t = c*n + r (4-step column-major),
            // so time t lands at grid position (r = t % n, c = t / n); input gate
            // fused at load.
            x.iter_mut().for_each(|v| *v = 0.0);
            for t in 0..t_len.min(n * l) {
                let mut v = xb[t];
                if self.gates & 1 != 0 {
                    v *= u_ext[bh_stride + bh * t_len + t];
                }
                x[(t % n) * l + (t / n)] = v;
            }
            // 1: A = x @ dL
            for ni in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..l {
                        let xv = x[ni * l + k];
                        sr += xv * packed[lay.dlr + k * lp + lo];
                        si += xv * packed[lay.dli + k * lp + lo];
                    }
                    ar[ni * l + lo] = sr;
                    ai[ni * l + lo] = si;
                }
            }
            // 2: forward twiddle
            for ni in 0..n {
                for lo in 0..l {
                    let idx = ni * l + lo;
                    let (zr, zi) = (ar[idx], ai[idx]);
                    let (twr, twi) = (
                        packed[lay.tw + (ni * lp + lo) * 2],
                        packed[lay.tw + (ni * lp + lo) * 2 + 1],
                    );
                    ar[idx] = zr * twr - zi * twi;
                    ai[idx] = zr * twi + zi * twr;
                }
            }
            // 3: B = dN @ A
            for np_ in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (
                            packed[lay.dnr + np_ * np + k],
                            packed[lay.dni + np_ * np + k],
                        );
                        let (zr, zi) = (ar[k * l + lo], ai[k * l + lo]);
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    br[np_ * l + lo] = sr;
                    bi[np_ * l + lo] = si;
                }
            }
            // 4: × k_f
            for i in 0..n * l {
                let (zr, zi) = (br[i], bi[i]);
                let (kr, ki) = (kfb[i * 2], kfb[i * 2 + 1]);
                br[i] = zr * kr - zi * ki;
                bi[i] = zr * ki + zi * kr;
            }
            // 5: A = idN @ B
            for np_ in 0..n {
                for lo in 0..l {
                    let (mut sr, mut si) = (0f32, 0f32);
                    for k in 0..n {
                        let (dr, di) = (
                            packed[lay.idnr + np_ * np + k],
                            packed[lay.idni + np_ * np + k],
                        );
                        let (zr, zi) = (br[k * l + lo], bi[k * l + lo]);
                        sr += dr * zr - di * zi;
                        si += dr * zi + di * zr;
                    }
                    ar[np_ * l + lo] = sr;
                    ai[np_ * l + lo] = si;
                }
            }
            // 6: conjugate twiddle
            for ni in 0..n {
                for lo in 0..l {
                    let idx = ni * l + lo;
                    let (zr, zi) = (ar[idx], ai[idx]);
                    let (twr, twi) = (
                        packed[lay.itw + (ni * lp + lo) * 2],
                        packed[lay.itw + (ni * lp + lo) * 2 + 1],
                    );
                    ar[idx] = zr * twr - zi * twi;
                    ai[idx] = zr * twi + zi * twr;
                }
            }
            // 7: out = Re(A @ idL) × scale, first T only (time t = lo*n + ni),
            // output gate fused at store.
            for ni in 0..n {
                for lo in 0..l {
                    let t = lo * n + ni;
                    if t >= t_len {
                        continue;
                    }
                    let mut sr = 0f32;
                    for k in 0..l {
                        sr += ar[ni * l + k] * packed[lay.idlr + k * lp + lo]
                            - ai[ni * l + k] * packed[lay.idli + k * lp + lo];
                    }
                    let mut v = sr * scale;
                    if has_d {
                        // Delta-tap skip on the staged (gated) input, per head.
                        v += x[ni * l + lo] * packed[d_off + bh % h];
                    }
                    if self.gates & 2 != 0 {
                        v *= u_ext[og_slot * bh_stride + bh * t_len + t];
                    }
                    out[bh * t_len + t] = v;
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, h, t_len))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        us: &candle_core::MetalStorage,
        ul: &Layout,
        ps: &candle_core::MetalStorage,
        pl: &Layout,
        ks: &candle_core::MetalStorage,
        kl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::{DType, MetalStorage};
        use objc2_metal::MTLSize;

        let (_g, b, h, t_len) = ul.shape().dims4()?;
        let (n, l) = (self.n, self.l);
        let lay = pack_layout(n, l);
        let (np, lp) = (lay.np, lay.lp);
        let dts = DType::F32.size_in_bytes();
        let tg_bytes = (5 * np * lp + 4 * 64) * dts;
        if tg_bytes > 32768 {
            candle_core::bail!("monarch fused conv padded: threadgroup mem {tg_bytes}B exceeds 32KB (N={n} L={l}); factor smaller");
        }

        #[repr(C)]
        struct PaddedConvParams {
            b: u32,
            h: u32,
            n: u32,
            l: u32,
            np: u32,
            lp: u32,
            off_dlr: u32,
            off_dli: u32,
            off_dnr: u32,
            off_dni: u32,
            off_tw: u32,
            off_idnr: u32,
            off_idni: u32,
            off_idlr: u32,
            off_idli: u32,
            off_itw: u32,
            t: u32,
            gates: u32,
            off_d: u32,
        }
        let params = PaddedConvParams {
            b: b as u32,
            h: h as u32,
            n: n as u32,
            l: l as u32,
            np: np as u32,
            lp: lp as u32,
            off_dlr: lay.dlr as u32,
            off_dli: lay.dli as u32,
            off_dnr: lay.dnr as u32,
            off_dni: lay.dni as u32,
            off_tw: lay.tw as u32,
            off_idnr: lay.idnr as u32,
            off_idni: lay.idni as u32,
            off_idlr: lay.idlr as u32,
            off_idli: lay.idli as u32,
            off_itw: lay.itw as u32,
            t: t_len as u32,
            gates: self.gates,
            off_d: lay.total as u32,
        };

        let dev = us.device();
        let p = crate::metal_util::pipeline(
            dev,
            "monarch_fused_conv_padded_f32",
            SRC_FUSED_CONV_PADDED,
        )?;
        let out_el = b * h * t_len;
        let out = dev.new_buffer(out_el, DType::F32, "monarch_fused_conv_padded")?;

        let enc = dev.command_encoder()?;
        enc.set_compute_pipeline_state(&p);
        enc.set_bytes(0, &params);
        enc.set_buffer(1, Some(us.buffer()), ul.start_offset() * dts);
        enc.set_buffer(2, Some(ps.buffer()), pl.start_offset() * dts);
        enc.set_buffer(3, Some(ks.buffer()), kl.start_offset() * dts);
        enc.set_buffer(4, Some(&*out), 0);
        let nplp = np * lp * dts;
        enc.set_threadgroup_memory_length(0, nplp); // ux
        enc.set_threadgroup_memory_length(1, nplp); // axr
        enc.set_threadgroup_memory_length(2, nplp); // axi
        enc.set_threadgroup_memory_length(3, nplp); // bxr
        enc.set_threadgroup_memory_length(4, nplp); // bxi
        enc.set_threadgroup_memory_length(5, 4 * 64 * dts); // scratch
        enc.dispatch_thread_groups(
            MTLSize {
                width: b * h,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(out, dev.clone(), out_el, DType::F32),
            Shape::from((b, h, t_len)),
        ))
    }
}

/// PADDED fused Monarch convolution — causal LINEAR convolution of a length-`T`
/// signal in one tiled `simdgroup_matrix` dispatch, with the Hyena/H3 elementwise
/// gates fused at their natural touch points (input gate at load, output gate at
/// store — no gated tensor ever materializes).
///
/// `u`/`in_gate`/`out_gate` are `[B,H,T]` real with `T ≤ N·L`; `k_f` `[…,N,L,2]`
/// is the filter's forward Monarch FFT at the transform size (zero-pad the filter
/// to `N·L` before `butterfly_fft_forward`, laid out in the same column-major time
/// order). `d` `[H]` is the optional per-head delta-tap skip `y += (u·in_gate)·D`
/// (`fused_fft_conv`'s `u·D`, applied to the same input the conv sees), added
/// before the output gate. FFT conv is circular at `N·L`: choose
/// `N·L ≥ T + filter_len − 1` (e.g. `T = N·L/2` with a length-`T` filter) so wrap
/// never reaches a stored output. Returns `[B,H,T]`.
#[allow(clippy::too_many_arguments)]
pub fn monarch_conv_fused_padded(
    u: &Tensor,
    k_f: &Tensor,
    in_gate: Option<&Tensor>,
    out_gate: Option<&Tensor>,
    d: Option<&Tensor>,
    d_f_n: &Tensor,
    d_f_l: &Tensor,
    twiddles: &Tensor,
    id_f_n: &Tensor,
    id_f_l: &Tensor,
    ifft_twiddles: &Tensor,
) -> Result<Tensor> {
    let (b, h, t_len) = u.dims3()?;
    let (n, _, _) = d_f_n.dims3()?;
    let (l, _, _) = d_f_l.dims3()?;
    if t_len > n * l {
        candle_core::bail!(
            "monarch_conv_fused_padded: signal length {t_len} exceeds transform {n}*{l}"
        );
    }
    let mut gates = 0u32;
    let mut parts: Vec<Tensor> = vec![u.contiguous()?];
    for (gate, bit) in [(in_gate, 1u32), (out_gate, 2u32)] {
        if let Some(g) = gate {
            if g.dims3()? != (b, h, t_len) {
                candle_core::bail!(
                    "monarch_conv_fused_padded: gate dims {:?} != signal dims {:?}",
                    g.dims(),
                    (b, h, t_len)
                );
            }
            gates |= bit;
            parts.push(g.contiguous()?);
        }
    }
    let u_ext = Tensor::stack(&parts, 0)?.contiguous()?; // [G,B,H,T]
    let k_f = k_f.broadcast_as((b, h, n, l, 2))?.contiguous()?;
    let mut packed = pack_full(
        &d_f_n.contiguous()?,
        &d_f_l.contiguous()?,
        &twiddles.contiguous()?,
        &id_f_n.contiguous()?,
        &id_f_l.contiguous()?,
        &ifft_twiddles.contiguous()?,
    )?;
    if let Some(dv) = d {
        if dv.dims1()? != h {
            candle_core::bail!(
                "monarch_conv_fused_padded: d has {} entries, expected H = {h}",
                dv.dims1()?
            );
        }
        gates |= 4;
        packed = Tensor::cat(&[&packed, &dv.contiguous()?], 0)?;
    }
    u_ext.apply_op3(&packed, &k_f, MonarchFusedConvPadded { n, l, gates })
}

/// Pre-compile the fused tensor-core kernels into the global pipeline cache so the first
/// realtime dispatch never pays the MSL compile — the candle analog of holding the MLX
/// kernel object at startup. Compile-once is process-wide (shared across threads), so one
/// call at engine init covers every worker thread. No-op on a non-Metal device.
pub fn warmup(device: &candle_core::Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let candle_core::Device::Metal(md) = device {
        crate::metal_util::pipeline(md, "monarch_fused_fwd_f32", SRC_FUSED_FWD)?;
        crate::metal_util::pipeline(md, "monarch_fused_conv_f32", SRC_FUSED_CONV)?;
        crate::metal_util::pipeline(md, "monarch_fused_conv_padded_f32", SRC_FUSED_CONV_PADDED)?;
    }
    let _ = device;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        butterfly_fft_forward, fft_matrix, ifft_matrix, monarch_conv, twiddle_factors_fft,
        twiddle_factors_ifft,
    };
    use candle_core::Device;

    // Build the six DFT/twiddle matrices for (n,l) on `dev`.
    #[allow(clippy::type_complexity)]
    fn mats(n: usize, l: usize, dev: &Device) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        (
            fft_matrix(n, dev).unwrap(),
            fft_matrix(l, dev).unwrap(),
            twiddle_factors_fft(n, l, dev).unwrap(),
            ifft_matrix(n, dev).unwrap(),
            ifft_matrix(l, dev).unwrap(),
            twiddle_factors_ifft(n, l, dev).unwrap(),
        )
    }

    /// Padded fused conv (CPU reference) == compose-from-verified-parts: zero-pad
    /// the signal to [N,L], run the un-fused `monarch_conv`, trim to T, apply the
    /// gates as plain tensor multiplies. Odd T exercises the ragged tile boundary.
    #[test]
    fn fused_padded_cpu_matches_composition() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (2usize, 2, 16, 8);
        let m = n * l;
        for t_len in [m / 2, 45usize, m] {
            let u: Vec<f32> = (0..b * h * t_len)
                .map(|i| (i as f32 * 0.11).sin())
                .collect();
            let gi: Vec<f32> = (0..b * h * t_len)
                .map(|i| (i as f32 * 0.05).cos())
                .collect();
            let go: Vec<f32> = (0..b * h * t_len)
                .map(|i| 1.0 + 0.3 * (i as f32 * 0.03).sin())
                .collect();
            let ut = Tensor::from_vec(u, (b, h, t_len), &dev).unwrap();
            let git = Tensor::from_vec(gi, (b, h, t_len), &dev).unwrap();
            let got = Tensor::from_vec(go, (b, h, t_len), &dev).unwrap();
            let kf_sig: Vec<f32> = (0..m).map(|i| (-(i as f32) * 0.02).exp()).collect();
            let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, &dev);
            let kt = Tensor::from_vec(kf_sig, (1, 1, n, l), &dev).unwrap();
            let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();

            let fused: Vec<f32> = monarch_conv_fused_padded(
                &ut,
                &k_f,
                Some(&git),
                Some(&got),
                None,
                &dfn,
                &dfl,
                &tw,
                &idfn,
                &idfl,
                &itw,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

            // Composition reference from already-verified pieces. Time is the
            // 4-step column-major flattening of [N,L]: place the padded signal
            // via reshape[L,N] + transpose, and read the output back the same way.
            let gated_in = ut.mul(&git).unwrap();
            let padded = gated_in
                .reshape((b, h, t_len))
                .unwrap()
                .pad_with_zeros(2, 0, m - t_len)
                .unwrap()
                .reshape((b, h, l, n))
                .unwrap()
                .transpose(2, 3)
                .unwrap()
                .contiguous()
                .unwrap();
            let full = monarch_conv(&padded, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap();
            let reference: Vec<f32> = full
                .transpose(2, 3)
                .unwrap()
                .reshape((b, h, m))
                .unwrap()
                .narrow(2, 0, t_len)
                .unwrap()
                .mul(&got)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            let max = fused
                .iter()
                .zip(&reference)
                .map(|(a, r)| (a - r).abs())
                .fold(0f32, f32::max);
            assert!(max < 1e-4, "T={t_len}: fused padded vs composition {max}");
        }
    }

    /// Ground truth for the FULL fused block: with T = N·L/2 and a length-T filter,
    /// `out[t] = ( Σ_{s≤t} k[s]·ug[t−s] + ug[t]·D_h ) · out_gate[t]` with
    /// `ug = u·in_gate` — the complete Hyena/H3 operator, one dispatch.
    #[test]
    fn fused_padded_full_block_is_causal_linear_conv() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (1usize, 2, 16, 8);
        let m = n * l;
        let t_len = m / 2;
        let u: Vec<f32> = (0..b * h * t_len)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let filt: Vec<f32> = (0..t_len).map(|i| (-(i as f32) * 0.1).exp()).collect();
        let ut = Tensor::from_vec(u.clone(), (b, h, t_len), &dev).unwrap();
        let mut k_pad = filt.clone();
        k_pad.resize(m, 0.0);
        let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, &dev);
        // The filter's time layout must match the signal's: column-major over
        // [N,L] (t = c*N + r), i.e. reshape [L,N] then transpose.
        let kt = Tensor::from_vec(k_pad, (1, 1, l, n), &dev)
            .unwrap()
            .transpose(2, 3)
            .unwrap()
            .contiguous()
            .unwrap();
        let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();

        let got: Vec<f32> = monarch_conv_fused_padded(
            &ut, &k_f, None, None, None, &dfn, &dfl, &tw, &idfn, &idfl, &itw,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

        let mut max = 0f32;
        for bh in 0..b * h {
            for t in 0..t_len {
                let mut acc = 0f32;
                for s in 0..=t {
                    acc += filt[s] * u[bh * t_len + (t - s)];
                }
                max = max.max((got[bh * t_len + t] - acc).abs());
            }
        }
        assert!(max < 1e-3, "fused padded vs direct causal conv: {max}");
    }

    /// metal == cpu for the padded+gated kernel, ragged T.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_padded_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let dev = Device::new_metal(0).unwrap();
        let (b, h, n, l) = (2usize, 3, 16, 8);
        let t_len = 45usize;
        let u: Vec<f32> = (0..b * h * t_len)
            .map(|i| (i as f32 * 0.09).sin())
            .collect();
        let gi: Vec<f32> = (0..b * h * t_len)
            .map(|i| (i as f32 * 0.04).cos())
            .collect();
        let go: Vec<f32> = (0..b * h * t_len)
            .map(|i| 1.0 + 0.2 * (i as f32 * 0.06).sin())
            .collect();
        let kf_sig: Vec<f32> = (0..n * l).map(|i| (-(i as f32) * 0.03).exp()).collect();

        let run = |d: &Device| -> Vec<f32> {
            let ut = Tensor::from_vec(u.clone(), (b, h, t_len), d).unwrap();
            let git = Tensor::from_vec(gi.clone(), (b, h, t_len), d).unwrap();
            let got = Tensor::from_vec(go.clone(), (b, h, t_len), d).unwrap();
            let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, d);
            let kt = Tensor::from_vec(kf_sig.clone(), (1, 1, n, l), d).unwrap();
            let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();
            let dt = Tensor::from_vec(
                (0..h).map(|i| 0.1 + 0.05 * i as f32).collect::<Vec<f32>>(),
                (h,),
                d,
            )
            .unwrap();
            monarch_conv_fused_padded(
                &ut,
                &k_f,
                Some(&git),
                Some(&got),
                Some(&dt),
                &dfn,
                &dfl,
                &tw,
                &idfn,
                &idfl,
                &itw,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
        };
        let (c, g) = (run(&cpu), run(&dev));
        let max = c
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-3, "padded fused metal vs cpu: {max}");
    }

    #[test]
    fn fused_forward_cpu_matches_unfused() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (2usize, 3, 16, 8);
        let x: Vec<f32> = (0..b * h * n * l)
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let xt = Tensor::from_vec(x, (b, h, n, l), &dev).unwrap();
        let (dfn, dfl, tw) = (
            fft_matrix(n, &dev).unwrap(),
            fft_matrix(l, &dev).unwrap(),
            twiddle_factors_fft(n, l, &dev).unwrap(),
        );
        let fused: Vec<f32> = butterfly_fft_forward_fused(&xt, &dfn, &dfl, &tw)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let unfused: Vec<f32> = butterfly_fft_forward(&xt, &dfn, &dfl, &tw)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let maxd = fused
            .iter()
            .zip(&unfused)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(maxd < 1e-4, "fused fwd (cpu) vs un-fused max diff {maxd}");
        eprintln!("fused forward cpu == un-fused, max diff {maxd:.2e}");
    }

    // ---- full fused conv ----

    #[test]
    fn fused_conv_cpu_matches_monarch_conv() {
        let dev = Device::Cpu;
        let (b, h, n, l) = (2usize, 2, 16, 8);
        let u: Vec<f32> = (0..b * h * n * l)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let ut = Tensor::from_vec(u, (b, h, n, l), &dev).unwrap();
        let kf: Vec<f32> = (0..b * h * n * l * 2)
            .map(|i| ((i * 7 % 13) as f32 * 0.03) - 0.2)
            .collect();
        let kft = Tensor::from_vec(kf, (b, h, n, l, 2), &dev).unwrap();
        let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, &dev);
        let fused: Vec<f32> = monarch_conv_fused(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let oracle: Vec<f32> = monarch_conv(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let maxd = fused
            .iter()
            .zip(&oracle)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(
            maxd < 1e-3,
            "fused conv (cpu) vs monarch_conv max diff {maxd}"
        );
        eprintln!("fused conv cpu == monarch_conv, max diff {maxd:.2e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_conv_metal_matches_cpu_and_oracle() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, h, n, l) = (2usize, 2, 16, 16);
        let u: Vec<f32> = (0..b * h * n * l)
            .map(|i| ((i * 13 % 19) as f32 * 0.05) - 0.4)
            .collect();
        let kf: Vec<f32> = (0..b * h * n * l * 2)
            .map(|i| ((i * 7 % 11) as f32 * 0.03) - 0.15)
            .collect();
        let run = |dev: &Device, fused: bool| -> Vec<f32> {
            let ut = Tensor::from_vec(u.clone(), (b, h, n, l), dev).unwrap();
            let kft = Tensor::from_vec(kf.clone(), (b, h, n, l, 2), dev).unwrap();
            let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, dev);
            let y = if fused {
                monarch_conv_fused(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap()
            } else {
                monarch_conv(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap()
            };
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let fused_met = run(&mdev, true);
        let oracle_met = run(&mdev, false);
        let fused_cpu = run(&Device::Cpu, true);
        let d_oracle = fused_met
            .iter()
            .zip(&oracle_met)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        let d_cpu = fused_met
            .iter()
            .zip(&fused_cpu)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(
            d_oracle < 1e-3,
            "fused-metal vs monarch_conv max diff {d_oracle}"
        );
        assert!(d_cpu < 1e-4, "fused-metal vs fused-cpu max diff {d_cpu}");
        eprintln!("fused conv: metal==oracle {d_oracle:.2e}, metal==cpu {d_cpu:.2e}");
    }

    // The edge-tile gate: non-multiples-of-8 N,L must match the un-fused oracle.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_conv_edge_dims() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        for (n, l) in [(6usize, 10usize), (12, 20), (8, 24), (10, 6)] {
            let (b, h) = (1usize, 2usize);
            let u: Vec<f32> = (0..b * h * n * l)
                .map(|i| ((i * 5 % 17) as f32 * 0.05) - 0.3)
                .collect();
            let kf: Vec<f32> = (0..b * h * n * l * 2)
                .map(|i| ((i * 3 % 7) as f32 * 0.04) - 0.1)
                .collect();
            let run = |dev: &Device, fused: bool| -> Vec<f32> {
                let ut = Tensor::from_vec(u.clone(), (b, h, n, l), dev).unwrap();
                let kft = Tensor::from_vec(kf.clone(), (b, h, n, l, 2), dev).unwrap();
                let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, dev);
                let y = if fused {
                    monarch_conv_fused(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap()
                } else {
                    monarch_conv(&ut, &kft, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap()
                };
                y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            };
            let fm = run(&mdev, true);
            let om = run(&mdev, false);
            let d = fm
                .iter()
                .zip(&om)
                .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
            assert!(
                d < 1e-3,
                "edge dims N={n} L={l}: fused-metal vs oracle max diff {d}"
            );
            eprintln!("fused conv edge N={n} L={l}: metal==oracle {d:.2e}");
        }
    }

    #[test]
    fn fused_conv_matches_circular() {
        let dev = Device::Cpu;
        let (n, l) = (4usize, 4);
        let m = n * l;
        let u_time: Vec<f32> = (0..m).map(|i| (i as f32 * 0.21).sin()).collect();
        let k_time: Vec<f32> = (0..m)
            .map(|i| (i as f32 * 0.11 + 1.0).cos() * 0.5)
            .collect();
        // Monarch reads input column-major: tensor[ni*L+li] holds time index li*N+ni.
        let lay = |t: &[f32]| -> Vec<f32> {
            let mut v = vec![0f32; m];
            for ni in 0..n {
                for li in 0..l {
                    v[ni * l + li] = t[li * n + ni];
                }
            }
            v
        };
        let ut = Tensor::from_vec(lay(&u_time), (1, 1, n, l), &dev).unwrap();
        let kt = Tensor::from_vec(lay(&k_time), (1, 1, n, l), &dev).unwrap();
        let (dfn, dfl, tw, idfn, idfl, itw) = mats(n, l, &dev);
        let k_f = butterfly_fft_forward(&kt, &dfn, &dfl, &tw).unwrap();
        let y = monarch_conv_fused(&ut, &k_f, &dfn, &dfl, &tw, &idfn, &idfl, &itw).unwrap();
        let out_flat: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let mut y_time = vec![0f32; m];
        for ni in 0..n {
            for li in 0..l {
                y_time[li * n + ni] = out_flat[ni * l + li];
            }
        }
        // direct length-M circular convolution.
        let mut exp = vec![0f32; m];
        for nn in 0..m {
            let mut acc = 0f64;
            for j in 0..m {
                acc += u_time[j] as f64 * k_time[(nn + m - j) % m] as f64;
            }
            exp[nn] = acc as f32;
        }
        let maxd = y_time
            .iter()
            .zip(&exp)
            .fold(0f32, |mm, (a, e)| mm.max((a - e).abs()));
        assert!(maxd < 1e-3, "fused conv != circular conv, max diff {maxd}");
        eprintln!("fused conv == circular conv (col-major time order), max diff {maxd:.2e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_forward_compiles_once_then_reuses() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, h, n, l) = (1usize, 1, 16, 16);
        let x: Vec<f32> = (0..b * h * n * l).map(|i| (i as f32 * 0.1).sin()).collect();
        let go = || {
            let xt = Tensor::from_vec(x.clone(), (b, h, n, l), &mdev).unwrap();
            let (dfn, dfl, tw) = (
                fft_matrix(n, &mdev).unwrap(),
                fft_matrix(l, &mdev).unwrap(),
                twiddle_factors_fft(n, l, &mdev).unwrap(),
            );
            let _ = butterfly_fft_forward_fused(&xt, &dfn, &dfl, &tw).unwrap();
        };
        go(); // warm: compiles the pipeline at most once
        let before = crate::metal_util::pipeline_compiles();
        for _ in 0..8 {
            go();
        }
        let after = crate::metal_util::pipeline_compiles();
        assert_eq!(
            after,
            before,
            "fused fwd kernel recompiled on reuse: {} compiles over 8 dispatches",
            after - before
        );
        eprintln!("fused forward: 0 recompiles over 8 reuses (cached pipeline, total compiles seen = {after})");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_forward_compiled_once_shared_across_threads() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        // A nested fn (not a closure) so the worker thread's body stays `'static`.
        fn run(dev: &Device, x: &[f32], n: usize, l: usize) -> Vec<f32> {
            let xt = Tensor::from_vec(x.to_vec(), (1, 1, n, l), dev).unwrap();
            let (dfn, dfl, tw) = (
                fft_matrix(n, dev).unwrap(),
                fft_matrix(l, dev).unwrap(),
                twiddle_factors_fft(n, l, dev).unwrap(),
            );
            butterfly_fft_forward_fused(&xt, &dfn, &dfl, &tw)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        }
        let (n, l) = (16usize, 16);
        let x: Vec<f32> = (0..n * l).map(|i| (i as f32 * 0.1).sin()).collect();

        // Compile once on the main thread (into the global cache).
        let main_res = run(&mdev, &x, n, l);

        // A worker thread sharing the SAME Metal device must reuse that compiled kernel:
        // 0 compiles on its own (thread-local) counter, and an identical result.
        let dev2 = mdev.clone();
        let x2 = x.clone();
        let (worker_res, worker_compiles) = std::thread::spawn(move || {
            let before = crate::metal_util::pipeline_compiles(); // this thread: 0 so far
            let r = run(&dev2, &x2, n, l);
            let after = crate::metal_util::pipeline_compiles();
            (r, after - before)
        })
        .join()
        .unwrap();

        assert_eq!(
            worker_compiles, 0,
            "worker thread recompiled the shared kernel {worker_compiles}x instead of reusing it"
        );
        let maxd = main_res
            .iter()
            .zip(&worker_res)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "cross-thread result mismatch {maxd}");
        eprintln!("fused forward: compiled once on main, reused on worker thread (0 recompiles), result identical ({maxd:.1e})");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_forward_metal_matches_unfused() {
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let (b, h, n, l) = (2usize, 3, 16, 16);
        let x: Vec<f32> = (0..b * h * n * l)
            .map(|i| ((i * 11 % 17) as f32 * 0.05) - 0.4)
            .collect();
        // fused on metal vs the crate's already-verified un-fused forward on metal.
        let run = |dev: &Device, fused: bool| -> Vec<f32> {
            let xt = Tensor::from_vec(x.clone(), (b, h, n, l), dev).unwrap();
            let (dfn, dfl, tw) = (
                fft_matrix(n, dev).unwrap(),
                fft_matrix(l, dev).unwrap(),
                twiddle_factors_fft(n, l, dev).unwrap(),
            );
            let y = if fused {
                butterfly_fft_forward_fused(&xt, &dfn, &dfl, &tw).unwrap()
            } else {
                butterfly_fft_forward(&xt, &dfn, &dfl, &tw).unwrap()
            };
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let fused_met = run(&mdev, true);
        let unfused_met = run(&mdev, false);
        let fused_cpu = run(&Device::Cpu, true);
        let d_mm = fused_met
            .iter()
            .zip(&unfused_met)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        let d_mc = fused_met
            .iter()
            .zip(&fused_cpu)
            .fold(0f32, |m, (a, e)| m.max((a - e).abs()));
        assert!(d_mm < 1e-4, "fused-metal vs unfused-metal max diff {d_mm}");
        assert!(d_mc < 1e-4, "fused-metal vs fused-cpu max diff {d_mc}");
        eprintln!("fused forward: metal==unfused {d_mm:.2e}, metal==cpu {d_mc:.2e}");
    }
}

// --- simdgroup_matrix compile+run probe (kept as a minimal regression test). ---
#[cfg(all(test, feature = "metal"))]
mod sg_probe {
    use candle_core::backend::BackendStorage;
    use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape};

    const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void sg_probe_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    simdgroup_float8x8 a, b, acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, C, 8);
}
"#;

    struct SgProbe;

    impl CustomOp2 for SgProbe {
        fn name(&self) -> &'static str {
            "sg_probe"
        }
        fn cpu_fwd(
            &self,
            as_: &CpuStorage,
            al: &Layout,
            bs: &CpuStorage,
            bl: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = as_.as_slice::<f32>()?;
            let b = bs.as_slice::<f32>()?;
            let (a0, _) = al
                .contiguous_offsets()
                .ok_or_else(|| candle_core::Error::Msg("sg_probe: A not contiguous".into()))?;
            let (b0, _) = bl
                .contiguous_offsets()
                .ok_or_else(|| candle_core::Error::Msg("sg_probe: B not contiguous".into()))?;
            let mut c = vec![0f32; 64];
            for i in 0..8 {
                for j in 0..8 {
                    let mut s = 0f32;
                    for k in 0..8 {
                        s += a[a0 + i * 8 + k] * b[b0 + k * 8 + j];
                    }
                    c[i * 8 + j] = s;
                }
            }
            Ok((CpuStorage::F32(c), Shape::from((8, 8))))
        }
        fn metal_fwd(
            &self,
            as_: &MetalStorage,
            al: &Layout,
            bs: &MetalStorage,
            bl: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            use objc2_metal::MTLSize;
            let dev = as_.device();
            let p = crate::metal_util::pipeline(dev, "sg_probe_f32", SRC)?;
            let out = dev.new_buffer(64, DType::F32, "sg_probe")?;
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(as_.buffer()), al.start_offset() * 4);
            enc.set_buffer(1, Some(bs.buffer()), bl.start_offset() * 4);
            enc.set_buffer(2, Some(&*out), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            Ok((
                MetalStorage::new(out, dev.clone(), 64, DType::F32),
                Shape::from((8, 8)),
            ))
        }
    }

    #[test]
    fn sg_probe_metal_matches_cpu() {
        use candle_core::{Device, Tensor};
        let mdev = match Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no metal device; skipping");
                return;
            }
        };
        let a: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32) * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..64)
            .map(|i| ((i * 5 % 11) as f32) * 0.07 - 0.2)
            .collect();
        let run = |dev: &Device| -> Vec<f32> {
            let at = Tensor::from_vec(a.clone(), (8, 8), dev).unwrap();
            let bt = Tensor::from_vec(b.clone(), (8, 8), dev).unwrap();
            at.apply_op2(&bt, SgProbe)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let cpu = run(&Device::Cpu);
        let met = run(&mdev);
        let maxd = cpu
            .iter()
            .zip(&met)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            maxd < 1e-4,
            "simdgroup_matrix probe metal vs cpu max diff {maxd}"
        );
        eprintln!("simdgroup_matrix probe: metal == cpu, max diff {maxd:.2e}");
    }
}
