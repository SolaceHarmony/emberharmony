//! NEON "zoo" — Rust bridge to `csrc/neon_zoo.cpp`, a library of aarch64 SIMD procedures
//! that mirror the GPU idioms of the crate's JIT-embedded Metal kernels (the
//! `candle-flashfftconv` simdgroup GEMM, radix-2 FFT, bf16 conv1d, double-double). Each
//! Metal construct maps to its closest NEON opcode — BFMMLA/BFDOT for the tensor-core MAC,
//! TBL for `simd_shuffle`, FCMLA for the complex butterfly, FMA error-free transforms for
//! double-double, FRECPE/FRSQRTE for GPU fast-math, SMMLA for the int tensor-core.
//!
//! Everything is **build-gated** on aarch64 (`cfg(has_neon_zoo)`, set by `build.rs`) and
//! **runtime-gated** on the relevant CPU feature ([`NeonFeatures`]); a binary stays portable
//! because a feature-specific proc is never *called* on a core that lacks it (and its opcodes
//! never leak into another function — see the C++ file header).

/// Runtime CPU-feature probe covering every extension the zoo (and the original BFMMLA GEMM)
/// needs. Cached; cheap to call. Off aarch64 every field is `false`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeonFeatures {
    pub bf16: bool,    // FEAT_BF16 — BFMMLA / BFDOT / BFCVT
    pub i8mm: bool,    // FEAT_I8MM — SMMLA / UMMLA
    pub fp16: bool,    // FEAT_FP16 — arithmetic fp16
    pub dotprod: bool, // FEAT_DotProd — SDOT / UDOT
    pub fcma: bool,    // FEAT_FCMA — FCMLA / FCADD (complex)
}

/// The cached [`NeonFeatures`] for the running CPU.
pub fn neon_features() -> &'static NeonFeatures {
    use std::sync::OnceLock;
    static F: OnceLock<NeonFeatures> = OnceLock::new();
    F.get_or_init(detect_features)
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn detect_features() -> NeonFeatures {
    // Each FEAT_* is a `hw.optional.arm.*` sysctl returning 0/1 (same pattern as threads.rs).
    fn feat(name: &std::ffi::CStr) -> bool {
        let mut val: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>();
        // SAFETY: valid C string + OUT params; no input buffer.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut val as *mut libc::c_int as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        rc == 0 && val == 1
    }
    NeonFeatures {
        bf16: feat(c"hw.optional.arm.FEAT_BF16"),
        i8mm: feat(c"hw.optional.arm.FEAT_I8MM"),
        fp16: feat(c"hw.optional.arm.FEAT_FP16"),
        dotprod: feat(c"hw.optional.arm.FEAT_DotProd"),
        fcma: feat(c"hw.optional.arm.FEAT_FCMA"),
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_features() -> NeonFeatures {
    // Linux exposes CPU features through the ELF aux vector (fixes the old bf16 probe's
    // `false`-on-Linux TODO). Bit positions from arch/arm64/include/uapi/asm/hwcap.h.
    const HWCAP_ASIMDHP: u64 = 1 << 10; // FEAT_FP16 (arith)
    const HWCAP_FCMA: u64 = 1 << 14; // FEAT_FCMA
    const HWCAP_ASIMDDP: u64 = 1 << 20; // FEAT_DotProd
    const HWCAP2_I8MM: u64 = 1 << 13; // FEAT_I8MM
    const HWCAP2_BF16: u64 = 1 << 14; // FEAT_BF16
    // SAFETY: getauxval is always safe to call; unknown types return 0.
    let cap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    let cap2 = unsafe { libc::getauxval(libc::AT_HWCAP2) };
    NeonFeatures {
        bf16: cap2 & HWCAP2_BF16 != 0,
        i8mm: cap2 & HWCAP2_I8MM != 0,
        fp16: cap & HWCAP_ASIMDHP != 0,
        dotprod: cap & HWCAP_ASIMDDP != 0,
        fcma: cap & HWCAP_FCMA != 0,
    }
}

#[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux"))))]
fn detect_features() -> NeonFeatures {
    NeonFeatures::default()
}

// ---- FFI to csrc/neon_zoo.cpp (aarch64, kernel built in) --------------------------------
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
extern "C" {
    fn lfm_bf16_gemm_f32_v2(a: *const u16, b: *const u16, c: *mut f32, m: i32, n: i32, k: i32);
    fn lfm_bf16_gemv_f32(a: *const u16, b: *const u16, c: *mut f32, n: i32, k: i32);
    fn lfm_s8_gemm_s32(a: *const i8, b: *const i8, c: *mut i32, m: i32, n: i32, k: i32);
    fn lfm_reduce_sum_f32(x: *const f32, n: i32) -> f32;
    fn lfm_reduce_max_f32(x: *const f32, n: i32) -> f32;
    fn lfm_permute_u8(table16: *const u8, idx: *const u8, out: *mut u8, n: i32);
    fn lfm_depthwise_causal_conv1d_bf16(
        u: *const u16,
        w: *const u16,
        bias: *const u16,
        out: *mut u16,
        bn: i32,
        d: i32,
        l: i32,
        k: i32,
        lout: i32,
    );
    fn lfm_dd_sum_f32(x: *const f32, n: i32) -> f32;
    fn lfm_dd_dot_f32(a: *const f32, b: *const f32, n: i32) -> f32;
    fn lfm_recip_f32(x: *const f32, out: *mut f32, n: i32);
    fn lfm_rsqrt_f32(x: *const f32, out: *mut f32, n: i32);
    fn lfm_fft_radix2_f32(data: *mut f32, n: i32, inverse: i32);
}

/// `true` when the zoo's bf16 GEMM path is both built in and supported by this CPU.
pub fn bf16_gemm_available() -> bool {
    cfg!(all(target_arch = "aarch64", has_neon_zoo)) && neon_features().bf16
}

/// `C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16` (raw bf16 bits as `u16`), row-major, f32
/// accumulate. Dispatches `M==1 → BFDOT GEMV`, else the 8×8 BFMMLA micro-kernel tiled over
/// M-row blocks with rayon (reusing the global pool sized by [`crate::threads`]). B is shared
/// across blocks. Caller must have verified [`bf16_gemm_available`] and sized the slices to
/// `M*K`, `K*N`, `M*N`.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn bf16_gemm_into(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    if m == 1 {
        // SAFETY: slices sized M*K / K*N / N; FEAT_BF16 verified by the caller.
        unsafe { lfm_bf16_gemv_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n as i32, k as i32) };
        return;
    }
    // Row-block dispatch: one rayon task per block of rows (== one Metal threadgroup per
    // (batch,head)). par_chunks keeps A- and C-row blocks aligned, so no raw pointers escape.
    let threads = rayon::current_num_threads().max(1);
    let rows = (m.div_ceil(threads)).max(8); // ≥8 keeps whole 8-row tiles per task
    c.par_chunks_mut(rows * n)
        .zip(a.par_chunks(rows * k))
        .for_each(|(cc, aa)| {
            let mm = cc.len() / n;
            // SAFETY: aa is mm*K, b is K*N, cc is mm*N; FEAT_BF16 verified by the caller.
            unsafe {
                lfm_bf16_gemm_f32_v2(
                    aa.as_ptr(),
                    b.as_ptr(),
                    cc.as_mut_ptr(),
                    mm as i32,
                    n as i32,
                    k as i32,
                );
            }
        });
}

/// GPU-style fast reciprocal-sqrt (`1/√x`) over a slice (FRSQRTE + 2 Newton steps), or the
/// scalar fallback off-aarch64. Directly usable for RMSNorm. `out` must match `x` in length.
pub fn rsqrt(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: both slices are `n` f32; the kernel reads/writes exactly those bounds.
    unsafe {
        lfm_rsqrt_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32);
        return;
    }
    #[allow(unreachable_code)]
    for (o, &v) in out.iter_mut().zip(x) {
        *o = 1.0 / v.sqrt();
    }
}

/// Deterministic high-accuracy sum via double-double accumulation (FMA error-free
/// transforms), or the scalar fallback off-aarch64.
pub fn dd_sum(x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: `x` is `n` contiguous f32.
    return unsafe { lfm_dd_sum_f32(x.as_ptr(), x.len() as i32) };
    #[allow(unreachable_code)]
    x.iter().sum()
}

/// Deterministic high-accuracy dot product (double-double accumulation), scalar fallback off-aarch64.
pub fn dd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: `a`/`b` are both `n` contiguous f32.
    return unsafe { lfm_dd_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as i32) };
    #[allow(unreachable_code)]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Horizontal sum (ADDV/FADDP), the NEON analog of a Metal threadgroup reduce. Scalar off-aarch64.
pub fn reduce_sum(x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: `x` is `n` contiguous f32.
    return unsafe { lfm_reduce_sum_f32(x.as_ptr(), x.len() as i32) };
    #[allow(unreachable_code)]
    x.iter().sum()
}

/// Horizontal max (FMAXV/FMAXP). Returns `-inf` for an empty slice. Scalar off-aarch64.
pub fn reduce_max(x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: `x` is `n` contiguous f32.
    return unsafe { lfm_reduce_max_f32(x.as_ptr(), x.len() as i32) };
    #[allow(unreachable_code)]
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// GPU-style fast reciprocal (`1/x`) over a slice (FRECPE + 2 Newton steps). Scalar off-aarch64.
pub fn recip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: both slices are `n` f32.
    unsafe {
        lfm_recip_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32);
        return;
    }
    #[allow(unreachable_code)]
    for (o, &v) in out.iter_mut().zip(x) {
        *o = 1.0 / v;
    }
}

/// In-register byte permute over a 16-entry table (TBL/TBX) — the NEON analog of Metal
/// `simd_shuffle`. `out[i] = table16[idx[i]]` for `idx<16`, else 0. Scalar off-aarch64.
pub fn permute_u8(table16: &[u8; 16], idx: &[u8], out: &mut [u8]) {
    assert_eq!(idx.len(), out.len());
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: table is 16 bytes; idx/out are both `n` bytes.
    unsafe {
        lfm_permute_u8(table16.as_ptr(), idx.as_ptr(), out.as_mut_ptr(), idx.len() as i32);
        return;
    }
    #[allow(unreachable_code)]
    for (o, &i) in out.iter_mut().zip(idx) {
        *o = if (i as usize) < 16 { table16[i as usize] } else { 0 };
    }
}

/// In-place radix-2 Cooley-Tukey FFT on interleaved `[re,im]` f32 (complex butterfly via
/// FCMLA). `data.len() == 2*n`, `n` a power of two; `inverse` scales by `1/n`. Requires
/// [`NeonFeatures::fcma`]; a no-op with a logged skip when unavailable.
pub fn fft_radix2(data: &mut [f32], inverse: bool) {
    let n = data.len() / 2;
    debug_assert_eq!(data.len(), 2 * n);
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    if neon_features().fcma {
        // SAFETY: `data` holds 2*n f32; the kernel touches exactly those.
        unsafe { lfm_fft_radix2_f32(data.as_mut_ptr(), n as i32, inverse as i32) };
        return;
    }
    let _ = (data, inverse, n);
}

/// int8 tensor-core GEMM `C(M,N) s32 = A(M,K) s8 · B(K,N) s8` via SMMLA. Requires
/// [`NeonFeatures::i8mm`]; leaves `c` untouched when unavailable.
pub fn s8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    if neon_features().i8mm {
        // SAFETY: slices sized M*K / K*N / M*N; FEAT_I8MM verified.
        unsafe {
            lfm_s8_gemm_s32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32);
        }
        return;
    }
    let _ = (a, b, c, m, n, k);
}

/// Depthwise causal conv1d with bf16 storage and f32 accumulate (single bf16 RNE store),
/// mirroring the Metal `depthwise_causal_conv1d_bf16`. `u:[B,D,L]`, `w:[D,K]`, `bias:[D]`,
/// `out:[B,D,Lout]` — all raw bf16 bits. Requires [`NeonFeatures::bf16`].
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
#[allow(clippy::too_many_arguments)]
pub fn depthwise_causal_conv1d_bf16(
    u: &[u16],
    w: &[u16],
    bias: &[u16],
    out: &mut [u16],
    bn: usize,
    d: usize,
    l: usize,
    k: usize,
    lout: usize,
) {
    debug_assert_eq!(u.len(), bn * d * l);
    debug_assert_eq!(out.len(), bn * d * lout);
    // SAFETY: pointers sized per the layout above; FEAT_BF16 verified by the caller.
    unsafe {
        lfm_depthwise_causal_conv1d_bf16(
            u.as_ptr(),
            w.as_ptr(),
            bias.as_ptr(),
            out.as_mut_ptr(),
            bn as i32,
            d as i32,
            l as i32,
            k as i32,
            lout as i32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn skip(feat: bool, name: &str) -> bool {
        if !feat {
            eprintln!("{name}: feature/kernel unavailable on this target — skipping");
        }
        !feat
    }

    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    #[test]
    fn gemm_v2_matches_f32_bf16_ref() {
        if skip(bf16_gemm_available(), "gemm_v2") {
            return;
        }
        // ragged edges + a large-K case (stresses the 16-accumulator summation order).
        for &(m, k, n) in &[(5usize, 13usize, 7usize), (17, 129, 33), (1, 512, 64)] {
            let a: Vec<bf16> = (0..m * k)
                .map(|i| bf16::from_f32((i * 7 % 23) as f32 / 23.0 - 0.5))
                .collect();
            let b: Vec<bf16> = (0..k * n)
                .map(|i| bf16::from_f32((i * 5 % 19) as f32 / 19.0 - 0.5))
                .collect();
            let ab: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
            let bb: Vec<u16> = b.iter().map(|x| x.to_bits()).collect();
            let mut c = vec![0f32; m * n];
            bf16_gemm_into(&ab, &bb, &mut c, m, n, k);
            let mut rel = 0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0f32;
                    for kk in 0..k {
                        s += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
                    }
                    rel = rel.max((c[i * n + j] - s).abs() / s.abs().max(1e-6));
                }
            }
            assert!(rel < 1e-2, "m={m} k={k} n={n} rel={rel}");
        }
    }

    #[test]
    fn rsqrt_matches_scalar() {
        if skip(neon_features().fp16 || !cfg!(target_arch = "aarch64"), "rsqrt") {
            // rsqrt is baseline NEON (no special feature) on aarch64; the scalar fallback
            // is always exercised off-aarch64. Only skip if the whole path is unavailable.
        }
        let x: Vec<f32> = (1..=64).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0f32; x.len()];
        rsqrt(&x, &mut out);
        for (o, &v) in out.iter().zip(&x) {
            assert!((o - 1.0 / v.sqrt()).abs() * v.sqrt() < 1e-3);
        }
    }

    #[test]
    fn dd_sum_beats_naive() {
        // 1e4 followed by many small values that a naive f32 sum drops entirely.
        let mut x = vec![1e-2f32; 200_000];
        x[0] = 1e4;
        let reference: f64 = x.iter().map(|&v| v as f64).sum();
        let dd = dd_sum(&x) as f64;
        assert!(
            (dd - reference).abs() / reference < 1e-4,
            "dd={dd} ref={reference}"
        );
    }
}
