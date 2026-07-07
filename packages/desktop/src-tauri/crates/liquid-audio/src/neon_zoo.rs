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
/// across blocks. Self-gates on FEAT_BF16 (falling back to a scalar f32-accumulate matmul when
/// absent), so it is safe to call on any aarch64 CPU. Slices must be sized `M*K`, `K*N`, `M*N`.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn bf16_gemm_into(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    // Real asserts (not debug_assert): the kernel reads m*k / k*n and writes m*n through raw
    // pointers, so a size mismatch would be an out-of-bounds FFI access in release builds.
    assert_eq!(a.len(), m * k, "bf16_gemm_into: a.len() != m*k");
    assert_eq!(b.len(), k * n, "bf16_gemm_into: b.len() != k*n");
    assert_eq!(c.len(), m * n, "bf16_gemm_into: c.len() != m*n");
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    // The BFMMLA/BFDOT kernels SIGILL without FEAT_BF16; gate the FFI so this public wrapper is
    // safe even if a caller skips `bf16_gemm_available()`. Scalar fallback keeps it correct.
    if !neon_features().bf16 {
        use half::bf16;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += bf16::from_bits(a[i * k + kk]).to_f32()
                        * bf16::from_bits(b[kk * n + j]).to_f32();
                }
                c[i * n + j] = s;
            }
        }
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
    // Fallback: compensated (Kahan) sum, so the high-accuracy contract holds off-aarch64 too.
    #[allow(unreachable_code)]
    {
        let (mut sum, mut c) = (0f32, 0f32);
        for &v in x {
            let y = v - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }
}

/// Deterministic high-accuracy dot product (double-double accumulation), scalar fallback off-aarch64.
pub fn dd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    // SAFETY: `a`/`b` are both `n` contiguous f32.
    return unsafe { lfm_dd_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as i32) };
    // Fallback: compensated (Kahan) dot, matching the high-accuracy contract off-aarch64.
    #[allow(unreachable_code)]
    {
        let (mut sum, mut c) = (0f32, 0f32);
        for (x, y) in a.iter().zip(b) {
            let p = x * y;
            let yy = p - c;
            let t = sum + yy;
            c = (t - sum) - yy;
            sum = t;
        }
        sum
    }
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
/// FCMLA on aarch64). `data.len()` must be even and `n = data.len()/2` a power of two;
/// `inverse` scales by `1/n`. Non-power-of-two `n` is a no-op (radix-2 would index out of
/// bounds). Falls back to a scalar radix-2 off-aarch64 / without FEAT_FCMA.
pub fn fft_radix2(data: &mut [f32], inverse: bool) {
    let n = data.len() / 2;
    // Radix-2 requires a power-of-two n; refuse other sizes rather than read/write OOB.
    if data.len() % 2 != 0 || n <= 1 || !n.is_power_of_two() {
        return;
    }
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    if neon_features().fcma {
        // SAFETY: n is a checked power of two and data holds 2*n f32; the kernel stays in bounds.
        unsafe { lfm_fft_radix2_f32(data.as_mut_ptr(), n as i32, inverse as i32) };
        return;
    }
    // Scalar fallback (same math as the FCMLA kernel).
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
    let sign = if inverse { 1.0f32 } else { -1.0f32 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * std::f32::consts::PI / (len as f32);
        let mut i = 0;
        while i < n {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f32).cos(), (ang * k as f32).sin());
                let (a, b) = (i + k, i + k + len / 2);
                let (xr, xi) = (data[2 * b], data[2 * b + 1]);
                let (tr, ti) = (wr * xr - wi * xi, wr * xi + wi * xr);
                let (ur, ui) = (data[2 * a], data[2 * a + 1]);
                data[2 * a] = ur + tr;
                data[2 * a + 1] = ui + ti;
                data[2 * b] = ur - tr;
                data[2 * b + 1] = ui - ti;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv = 1.0f32 / n as f32;
        for v in data.iter_mut() {
            *v *= inv;
        }
    }
}

/// int8 tensor-core GEMM `C(M,N) s32 = A(M,K) s8 · B(K,N) s8` via SMMLA. Requires
/// [`NeonFeatures::i8mm`]; leaves `c` untouched when unavailable.
pub fn s8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // Real asserts: the kernel indexes m*k / k*n / m*n through raw pointers.
    assert_eq!(a.len(), m * k, "s8_gemm: a.len() != m*k");
    assert_eq!(b.len(), k * n, "s8_gemm: b.len() != k*n");
    assert_eq!(c.len(), m * n, "s8_gemm: c.len() != m*n");
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    if neon_features().i8mm {
        // SAFETY: slices sized M*K / K*N / M*N; FEAT_I8MM verified.
        unsafe {
            lfm_s8_gemm_s32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32);
        }
        return;
    }
    // Scalar fallback (off-aarch64 or no FEAT_I8MM) so callers get a correct result, not a no-op.
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
                sum += a[i * k + kk] as i32 * b[kk * n + j] as i32;
            }
            c[i * n + j] = sum;
        }
    }
}

/// Depthwise causal conv1d with bf16 storage and f32 accumulate (single bf16 RNE store),
/// mirroring the Metal `depthwise_causal_conv1d_bf16`. `u:[B,D,L]`, `w:[D,K]`, `bias:[D]`,
/// `out:[B,D,Lout]` — all raw bf16 bits. Uses the NEON kernel when FEAT_BF16 is present,
/// else an equivalent scalar (`half::bf16`) fallback — never SIGILLs on a non-bf16 core.
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
    // Real asserts: the kernel indexes these extents through raw pointers.
    assert_eq!(u.len(), bn * d * l, "conv1d: u.len() != B*D*L");
    assert_eq!(w.len(), d * k, "conv1d: w.len() != D*K");
    assert_eq!(bias.len(), d, "conv1d: bias.len() != D");
    assert_eq!(out.len(), bn * d * lout, "conv1d: out.len() != B*D*Lout");
    #[cfg(all(target_arch = "aarch64", has_neon_zoo))]
    if neon_features().bf16 {
        // SAFETY: pointers sized per the asserts above; FEAT_BF16 verified here (the kernel
        // uses BF16/BFCVT instructions and would SIGILL otherwise).
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
        return;
    }
    // Scalar fallback: bf16 load → f32 accumulate → bf16 RNE store (same regime as the kernel).
    use half::bf16;
    for b in 0..bn {
        for di in 0..d {
            let u_off = (b * d + di) * l;
            let o_off = (b * d + di) * lout;
            let bias_f = bf16::from_bits(bias[di]).to_f32();
            for t in 0..lout {
                let mut acc = bias_f;
                for j in 0..k {
                    let idx = t as isize - (k as isize - 1) + j as isize;
                    if idx >= 0 && (idx as usize) < l {
                        acc += bf16::from_bits(u[u_off + idx as usize]).to_f32()
                            * bf16::from_bits(w[di * k + j]).to_f32();
                    }
                }
                out[o_off + t] = bf16::from_f32(acc).to_bits();
            }
        }
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

    #[test]
    fn fft_round_trips_and_ignores_non_pow2() {
        // forward then inverse recovers the input (kernel on aarch64, scalar fallback elsewhere).
        let orig: Vec<f32> = (0..16).map(|i| ((i * 37 % 11) as f32 / 11.0) - 0.5).collect();
        let mut d = orig.clone();
        fft_radix2(&mut d, false);
        fft_radix2(&mut d, true);
        for (g, o) in d.iter().zip(&orig) {
            assert!((g - o).abs() < 1e-3, "round-trip drift g={g} o={o}");
        }
        // non-power-of-two n (len=12 → n=6) must be a safe no-op, not an OOB.
        let mut bad = vec![1.0f32; 12];
        fft_radix2(&mut bad, false);
        assert!(bad.iter().all(|&v| v == 1.0), "non-pow2 FFT must not touch data");
    }

    #[test]
    fn s8_gemm_matches_scalar() {
        let (m, k, n) = (6usize, 19usize, 5usize);
        let a: Vec<i8> = (0..m * k).map(|i| (i as i8 % 15) - 7).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i as i8 % 13) - 6).collect();
        let mut c = vec![0i32; m * n];
        s8_gemm(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let s: i32 = (0..k).map(|kk| a[i * k + kk] as i32 * b[kk * n + j] as i32).sum();
                assert_eq!(c[i * n + j], s, "s8_gemm[{i}][{j}]");
            }
        }
    }

    #[test]
    fn conv1d_bf16_matches_scalar() {
        use half::bf16;
        let (bn, d, l, k) = (2usize, 3usize, 12usize, 4usize);
        let mk = |i: usize| bf16::from_f32(((i * 7 % 17) as f32 / 17.0) - 0.5);
        let u: Vec<u16> = (0..bn * d * l).map(|i| mk(i).to_bits()).collect();
        let w: Vec<u16> = (0..d * k).map(|i| mk(i + 3).to_bits()).collect();
        let bias: Vec<u16> = (0..d).map(|i| mk(i + 5).to_bits()).collect();
        let mut out = vec![0u16; bn * d * l];
        depthwise_causal_conv1d_bf16(&u, &w, &bias, &mut out, bn, d, l, k, l);
        for b in 0..bn {
            for di in 0..d {
                for t in 0..l {
                    let mut acc = bf16::from_bits(bias[di]).to_f32();
                    for j in 0..k {
                        let idx = t as isize - (k as isize - 1) + j as isize;
                        if idx >= 0 && (idx as usize) < l {
                            acc += bf16::from_bits(u[(b * d + di) * l + idx as usize]).to_f32()
                                * bf16::from_bits(w[di * k + j]).to_f32();
                        }
                    }
                    let want = bf16::from_f32(acc).to_bits();
                    let got = out[(b * d + di) * l + t];
                    assert_eq!(got, want, "conv1d[{b}][{di}][{t}]");
                }
            }
        }
    }
}
