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
/// across blocks. **Precondition:** the running CPU has FEAT_BF16 — verify [`bf16_gemm_available`]
/// (or [`neon_features`]) first; the BFMMLA/BFDOT kernels `SIGILL` without it. Slices must be
/// sized `M*K`, `K*N`, `M*N`.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn bf16_gemm_into(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    // Real asserts (not debug_assert): the kernel reads m*k / k*n and writes m*n through raw
    // pointers, so a size mismatch would be an out-of-bounds FFI access in release builds.
    assert_eq!(a.len(), m * k, "bf16_gemm_into: a.len() != m*k");
    assert_eq!(b.len(), k * n, "bf16_gemm_into: b.len() != k*n");
    assert_eq!(c.len(), m * n, "bf16_gemm_into: c.len() != m*n");
    // Fail loudly (panic) rather than SIGILL if a caller reaches this without FEAT_BF16. Not a
    // fallback — the precondition simply must hold; the live path checks bf16_gemm_available().
    assert!(
        neon_features().bf16,
        "bf16_gemm_into requires FEAT_BF16 (check bf16_gemm_available() first)"
    );
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

// The remaining zoo procedures are aarch64 + zoo only — there is no scalar fallback. Off the
// hardware happy path a caller should use a different code path entirely, not a silent scalar
// substitute that would mask a missing feature. Feature-specific ops (FFT→FCMA, s8_gemm→I8MM,
// conv1d→BF16) document their precondition; verify [`neon_features`] before calling.

/// GPU-style fast reciprocal-sqrt (`1/√x`) over a slice (FRSQRTE + 2 Newton steps). Directly
/// usable for RMSNorm. `out` must match `x` in length.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn rsqrt(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    // SAFETY: both slices are `n` f32; the kernel reads/writes exactly those bounds.
    unsafe { lfm_rsqrt_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// GPU-style fast reciprocal (`1/x`) over a slice (FRECPE + 2 Newton steps).
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn recip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    // SAFETY: both slices are `n` f32.
    unsafe { lfm_recip_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// Deterministic high-accuracy sum via double-double accumulation (FMA error-free transforms).
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn dd_sum(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_dd_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Deterministic high-accuracy dot product (double-double accumulation).
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn dd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: `a`/`b` are both `n` contiguous f32.
    unsafe { lfm_dd_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as i32) }
}

/// Horizontal sum (ADDV/FADDP), the NEON analog of a Metal threadgroup reduce.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn reduce_sum(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Horizontal max (FMAXV/FMAXP). Returns `-inf` for an empty slice.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn reduce_max(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_max_f32(x.as_ptr(), x.len() as i32) }
}

/// In-register byte permute over a 16-entry table (TBL/TBX) — the NEON analog of Metal
/// `simd_shuffle`. `out[i] = table16[idx[i]]` for `idx<16`, else 0.
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn permute_u8(table16: &[u8; 16], idx: &[u8], out: &mut [u8]) {
    assert_eq!(idx.len(), out.len());
    // SAFETY: table is 16 bytes; idx/out are both `n` bytes.
    unsafe { lfm_permute_u8(table16.as_ptr(), idx.as_ptr(), out.as_mut_ptr(), idx.len() as i32) };
}

/// In-place radix-2 Cooley-Tukey FFT on interleaved `[re,im]` f32 (complex butterfly via FCMLA).
/// `data.len()` must be even and `n = data.len()/2` a power of two — asserted, since radix-2 has
/// no meaning otherwise (and would index out of bounds). `inverse` scales by `1/n`.
/// **Precondition:** FEAT_FCMA (check [`neon_features`]).
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn fft_radix2(data: &mut [f32], inverse: bool) {
    let n = data.len() / 2;
    assert!(
        data.len() % 2 == 0 && n >= 1 && n.is_power_of_two(),
        "fft_radix2: n must be a power of two, got data.len()={}",
        data.len()
    );
    // Fail loudly rather than SIGILL: the FCMLA butterfly kernel needs FEAT_FCMA.
    assert!(neon_features().fcma, "fft_radix2 requires FEAT_FCMA");
    // SAFETY: n is an asserted power of two and data holds 2*n f32; the kernel stays in bounds.
    unsafe { lfm_fft_radix2_f32(data.as_mut_ptr(), n as i32, inverse as i32) };
}

/// int8 tensor-core GEMM `C(M,N) s32 = A(M,K) s8 · B(K,N) s8` via SMMLA. Slices sized `M*K`,
/// `K*N`, `M*N`. **Precondition:** FEAT_I8MM (check [`neon_features`]).
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
pub fn s8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // Real asserts: the kernel indexes m*k / k*n / m*n through raw pointers.
    assert_eq!(a.len(), m * k, "s8_gemm: a.len() != m*k");
    assert_eq!(b.len(), k * n, "s8_gemm: b.len() != k*n");
    assert_eq!(c.len(), m * n, "s8_gemm: c.len() != m*n");
    // Fail loudly rather than SIGILL: SMMLA needs FEAT_I8MM (absent on e.g. M1).
    assert!(neon_features().i8mm, "s8_gemm requires FEAT_I8MM");
    // SAFETY: slices sized M*K / K*N / M*N; FEAT_I8MM asserted above.
    unsafe {
        lfm_s8_gemm_s32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32)
    };
}

/// Depthwise causal conv1d with bf16 storage and f32 accumulate (single bf16 RNE store),
/// mirroring the Metal `depthwise_causal_conv1d_bf16`. `u:[B,D,L]`, `w:[D,K]`, `bias:[D]`,
/// `out:[B,D,Lout]` — all raw bf16 bits. **Precondition:** FEAT_BF16 (check [`neon_features`]).
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
    // Real asserts: the kernel indexes these extents through raw pointers.
    assert_eq!(u.len(), bn * d * l, "conv1d: u.len() != B*D*L");
    assert_eq!(w.len(), d * k, "conv1d: w.len() != D*K");
    assert_eq!(bias.len(), d, "conv1d: bias.len() != D");
    assert_eq!(out.len(), bn * d * lout, "conv1d: out.len() != B*D*Lout");
    // Fail loudly rather than SIGILL: the bf16/BFCVT kernel needs FEAT_BF16.
    assert!(neon_features().bf16, "depthwise_causal_conv1d_bf16 requires FEAT_BF16");
    // SAFETY: pointers sized per the asserts; FEAT_BF16 asserted above.
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

// The zoo procedures only exist on aarch64 with the kernel built in, so the tests do too —
// they run on the macOS arm64 CI leg (rust-voice.yml), where the hardware actually executes
// BFMMLA/FCMLA/SMMLA. On x86 CI the crate still builds; there is simply nothing here to run.
#[cfg(test)]
#[cfg(all(target_arch = "aarch64", has_neon_zoo))]
mod tests {
    use super::*;
    use half::bf16;

    /// Skip a feature-gated test when the running CPU lacks the extension — e.g. an M1 CI
    /// runner has FCMA but not FEAT_BF16/I8MM. The baseline ops (rsqrt/recip/reduce/dd) always run.
    fn skip(feat: bool, name: &str) -> bool {
        if !feat {
            eprintln!("{name}: CPU feature unavailable on this runner — skipping");
        }
        !feat
    }

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
        // FRSQRTE + 2 Newton steps is baseline NEON (no feature gate) — always available here.
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
    fn fft_round_trips() {
        if skip(neon_features().fcma, "fft") {
            return;
        }
        // forward then inverse (via the FCMLA butterfly kernel) recovers the input.
        let orig: Vec<f32> = (0..16).map(|i| ((i * 37 % 11) as f32 / 11.0) - 0.5).collect();
        let mut d = orig.clone();
        fft_radix2(&mut d, false);
        fft_radix2(&mut d, true);
        for (g, o) in d.iter().zip(&orig) {
            assert!((g - o).abs() < 1e-3, "round-trip drift g={g} o={o}");
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn fft_rejects_non_pow2() {
        // n=6 (data.len()=12) is not a radix-2 size; the wrapper must reject it loudly rather
        // than silently no-op or index out of bounds.
        let mut bad = vec![1.0f32; 12];
        fft_radix2(&mut bad, false);
    }

    #[test]
    fn s8_gemm_matches_scalar() {
        if skip(neon_features().i8mm, "s8_gemm") {
            return;
        }
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
        if skip(neon_features().bf16, "conv1d") {
            return;
        }
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

    // The FFI size checks are real `assert_eq!` (not debug_assert), so a mismatched slice
    // length panics before any raw-pointer FFI access — even in release. These fire on the size
    // assert, which precedes the feature assert, so they hold regardless of the runner's CPU.
    #[test]
    #[should_panic(expected = "a.len() != m*k")]
    fn gemm_into_rejects_mismatched_dims() {
        let a = vec![0u16; 3]; // wrong: m*k = 2*4 = 8
        let b = vec![0u16; 8];
        let mut c = vec![0f32; 4];
        bf16_gemm_into(&a, &b, &mut c, 2, 2, 4);
    }

    #[test]
    #[should_panic(expected = "a.len() != m*k")]
    fn s8_gemm_rejects_mismatched_dims() {
        let a = vec![0i8; 3]; // wrong: m*k = 2*4 = 8
        let b = vec![0i8; 8];
        let mut c = vec![0i32; 4];
        s8_gemm(&a, &b, &mut c, 2, 2, 4);
    }

    #[test]
    #[should_panic(expected = "u.len() != B*D*L")]
    fn conv1d_rejects_mismatched_dims() {
        let u = vec![0u16; 5]; // wrong: B*D*L = 1*2*6 = 12
        let w = vec![0u16; 2 * 3];
        let bias = vec![0u16; 2];
        let mut out = vec![0u16; 1 * 2 * 6];
        depthwise_causal_conv1d_bf16(&u, &w, &bias, &mut out, 1, 2, 6, 3, 6);
    }
}
