//! NEON `BFMMLA` bf16 GEMM — closes candle 0.9.2's CPU bf16-matmul gap.
//!
//! candle's CPU matmul allowlist is `F16 | F32 | F64` (`cpu_backend/mod.rs`); bf16 falls
//! through to `UnsupportedDTypeForOp`, so BF16 CPU linears route through this bridge instead
//! of stock candle matmul. The Arm BFloat16 extension (FEAT_BF16) has `BFMMLA`, which does
//! a 2×4·4×2 bf16 matmul with **f32 accumulate**. We compile a small C micro-kernel
//! (`csrc/bf16_gemm.c`, via build.rs `cc` with `-march=armv8.2-a+bf16`) and call it here.
//! Build-gated on aarch64 (`cfg(has_bf16_kernel)`) and **runtime**-gated on
//! FEAT_BF16 (BFMMLA `SIGILL`s without it), so a binary stays portable.

use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

// Only wired in as the fallback when the tightened zoo GEMM was not built (see cpu_fwd).
#[cfg(all(target_arch = "aarch64", has_bf16_kernel, not(has_neon_zoo)))]
extern "C" {
    /// `C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16`, all row-major. bf16 crosses as raw u16.
    fn lfm_bf16_gemm_f32(a: *const u16, b: *const u16, c: *mut f32, m: i32, n: i32, k: i32);
}

/// Whether the running CPU has a native bf16 tensor extension: Arm FEAT_BF16 (BFMMLA) on
/// aarch64, AVX-512-BF16 (VDPBF16PS) on x86-64. Cached probe. (On x86 the GEMM still runs
/// without it via the AVX2 upconvert path — see [`bf16_gemm_available`].)
pub fn has_feat_bf16() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon_zoo::neon_features().bf16
    }
    #[cfg(target_arch = "x86_64")]
    {
        crate::x86_zoo::x86_features().avx512bf16
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// `true` when a hardware bf16 GEMM is both **built in** and **usable** on this CPU — i.e.
/// [`bf16_matmul`] takes the SIMD path rather than returning `None`. aarch64 requires
/// FEAT_BF16 (BFMMLA is bf16-only); x86-64 requires just AVX2 + FMA (the kernel upconverts,
/// and additionally uses VDPBF16PS when AVX-512-BF16 is present).
pub fn bf16_gemm_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        cfg!(all(target_arch = "aarch64", has_bf16_kernel)) && crate::neon_zoo::neon_features().bf16
    }
    #[cfg(target_arch = "x86_64")]
    {
        crate::x86_zoo::bf16_gemm_available()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// candle `CustomOp2` over the kernel: `bf16 (M,K) ⊗ bf16 (K,N) → f32 (M,N)` on CPU.
/// The single FFI call site. Use as `a16.apply_op2_no_bwd(&b16, &Bf16Gemm)` with both
/// inputs bf16+contiguous (see [`bf16_matmul`] for the cast/guard wrapper). Backward and
/// the GPU paths intentionally bail — this op exists only to fill candle's CPU gap.
pub struct Bf16Gemm;

impl CustomOp2 for Bf16Gemm {
    fn name(&self) -> &'static str {
        "bf16-gemm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if !bf16_gemm_available() {
            candle_core::bail!("bf16-gemm: FEAT_BF16 kernel unavailable on this target");
        }
        let (d1, d2) = (l1.dims(), l2.dims());
        if d1.len() != 2 || d2.len() != 2 || d1[1] != d2[0] {
            candle_core::bail!("bf16-gemm expects 2-D (M,K)·(K,N), got {d1:?}·{d2:?}");
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("bf16-gemm requires contiguous inputs");
        }
        let (m, k, n) = (d1[0], d1[1], d2[1]);
        let a = match s1 {
            CpuStorage::BF16(v) => v,
            _ => candle_core::bail!("bf16-gemm: lhs must be bf16"),
        };
        let b = match s2 {
            CpuStorage::BF16(v) => v,
            _ => candle_core::bail!("bf16-gemm: rhs must be bf16"),
        };
        let a = &a[l1.start_offset()..l1.start_offset() + m * k];
        let b = &b[l2.start_offset()..l2.start_offset() + k * n];
        #[allow(unused_mut)] // `c` is mutated only on the SIMD kernel paths below
        let mut c = vec![0f32; m * n];
        // Preferred aarch64 path: the tightened NEON "zoo" GEMM (8×8 BFMMLA multi-accumulator +
        // rayon row-block dispatch, or a BFDOT GEMV when M==1). Same bf16-exact-product /
        // f32-accumulate numerics as the reference kernel; only the summation order differs.
        #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
        {
            // half::bf16 is repr(transparent) over u16, so the bit-slice view is sound.
            let ab = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u16, a.len()) };
            let bb = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len()) };
            crate::neon_zoo::bf16_gemm_into(ab, bb, &mut c, m, n, k);
        }
        // x86-64 path: the AVX-512-BF16 (VDPBF16PS) / AVX2 zoo GEMM, fanned out over M-row
        // blocks with rayon. Same f32-accumulate numerics.
        #[cfg(all(target_arch = "x86_64", has_x86_zoo))]
        {
            let ab = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u16, a.len()) };
            let bb = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len()) };
            crate::x86_zoo::bf16_gemm_into(ab, bb, &mut c, m, n, k);
        }
        // Fallback: the original single-file BFMMLA kernel (only reachable if the zoo TU failed
        // to build but the reference kernel did).
        #[cfg(all(target_arch = "aarch64", has_bf16_kernel, not(has_neon_zoo)))]
        // SAFETY: a/b are M*K / K*N contiguous bf16 (==u16) lanes; c is M*N f32; FEAT_BF16
        // verified above; the kernel reads/writes exactly those bounds.
        unsafe {
            lfm_bf16_gemm_f32(
                a.as_ptr() as *const u16,
                b.as_ptr() as *const u16,
                c.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
            );
        }
        let _ = (a, b);
        Ok((CpuStorage::F32(c), Shape::from((m, n))))
    }
}

/// bf16 matmul on the CPU via `BFMMLA`: `A(M,K) · B(K,N) → f32(M,N)`. Inputs are cast to
/// bf16; the accumulate is f32 (torch's bf16-matmul numerics). 2-D, CPU only. Returns
/// `Ok(None)` when the kernel/feature is unavailable, so callers fall back to candle's
/// f32 path (e.g. `a.to_dtype(F32)?.matmul(&b.to_dtype(F32)?)`).
pub fn bf16_matmul(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if !bf16_gemm_available() || !a.device().is_cpu() || !b.device().is_cpu() {
        return Ok(None);
    }
    let a16 = a.to_dtype(DType::BF16)?.contiguous()?;
    let b16 = b.to_dtype(DType::BF16)?.contiguous()?;
    Ok(Some(a16.apply_op2_no_bwd(&b16, &Bf16Gemm)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn bf16_gemm_matches_f32_reference() {
        if !bf16_gemm_available() {
            eprintln!("FEAT_BF16 / kernel unavailable on this target — skipping");
            return;
        }
        let dev = Device::Cpu;
        // Non-aligned dims exercise the M%2 / N%2 / K%4 zero-padded edges.
        let (m, k, n) = (5usize, 13usize, 7usize);
        let av: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 / 23.0 - 0.5) * 2.0)
            .collect();
        let bv: Vec<f32> = (0..k * n)
            .map(|i| ((i * 5 % 19) as f32 / 19.0 - 0.5) * 2.0)
            .collect();
        let a = Tensor::from_vec(av, (m, k), &dev).unwrap();
        let b = Tensor::from_vec(bv, (k, n), &dev).unwrap();

        // Reference: round inputs to bf16, then an f32 matmul — BFMMLA's exact-product
        // f32-accumulate numerics, modulo accumulation order.
        let a_ref = a
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let b_ref = b
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let cref: Vec<f32> = a_ref
            .matmul(&b_ref)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let cgot: Vec<f32> = bf16_matmul(&a, &b)
            .unwrap()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(cgot.len(), cref.len());
        let maxd = cgot
            .iter()
            .zip(&cref)
            .fold(0f32, |m, (g, r)| m.max((g - r).abs()));
        let scale = cref.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        eprintln!(
            "BFMMLA bf16 GEMM vs f32(bf16-inputs) ref: max {maxd:.3e} (rel {:.3e})",
            maxd / scale
        );
        assert!(
            maxd / scale < 1e-2,
            "BFMMLA vs ref rel {} too large",
            maxd / scale
        );
    }
}
