//! NEON flashkern — Rust bridge to `native/kernels/aarch64/flashkern_neon.cpp`, a library of aarch64 SIMD procedures
//! that mirror the GPU idioms of the crate's JIT-embedded Metal kernels (the
//! `candle-flashfftconv` simdgroup GEMM, radix-2 FFT, bf16 conv1d, double-double). Each
//! Metal construct maps to its closest NEON opcode — BFMMLA/BFDOT for the tensor-core MAC,
//! TBL for `simd_shuffle`, FCMLA for the complex butterfly, FMA error-free transforms for
//! double-double, FRECPE/FRSQRTE for GPU fast-math, SMMLA for the int tensor-core.
//!
//! Everything is **build-gated** on aarch64 (`cfg(target_arch = "aarch64")`, set by `build.rs`) and
//! **runtime-gated** on the relevant CPU feature ([`NeonFeatures`]); a binary stays portable
//! because a feature-specific proc is never *called* on a core that lacks it (and its opcodes
//! never leak into another function — see the C++ file header).

/// Runtime CPU-feature probe covering every extension flashkern (and the original BFMMLA GEMM)
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

// ---- FFI to native/kernels/aarch64/flashkern_neon.cpp (aarch64, kernel built in) --------------------------------
#[cfg(target_arch = "aarch64")]
extern "C" {
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
    fn lfm_complex_mul_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32);
    fn lfm_depthwise3_f32(x: *const f32, k: *const f32, y: *mut f32, bn: i32, c: i32, l: i32);
    fn lfm_depthwise3_causal_f32(
        x: *const f32,
        k: *const f32,
        y: *mut f32,
        bn: i32,
        c: i32,
        l: i32,
    );
    fn lfm_conv1d_update_f32(
        bcx: *const f32,
        state: *const f32,
        w: *const f32,
        out: *mut f32,
        bn: i32,
        d: i32,
        t: i32,
        k: i32,
    );
    fn lfm_conv1d_update_bf16(
        bcx: *const u16,
        state: *const u16,
        w: *const u16,
        out: *mut u16,
        bn: i32,
        d: i32,
        t: i32,
        k: i32,
    );
}

// The remaining flashkern procedures are aarch64 + flashkern only — there is no scalar fallback. Off the
// hardware happy path a caller should use a different code path entirely, not a silent scalar
// substitute that would mask a missing feature. Feature-specific ops (FFT→FCMA, s8_gemm→I8MM,
// conv1d→BF16) document their precondition; verify [`neon_features`] before calling.

/// GPU-style fast reciprocal-sqrt (`1/√x`) over a slice (FRSQRTE + 2 Newton steps). Directly
/// usable for RMSNorm. `out` must match `x` in length.
#[cfg(target_arch = "aarch64")]
pub fn rsqrt(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    // SAFETY: both slices are `n` f32; the kernel reads/writes exactly those bounds.
    unsafe { lfm_rsqrt_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// GPU-style fast reciprocal (`1/x`) over a slice (FRECPE + 2 Newton steps).
#[cfg(target_arch = "aarch64")]
pub fn recip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    // SAFETY: both slices are `n` f32.
    unsafe { lfm_recip_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// Deterministic high-accuracy sum via double-double accumulation (FMA error-free transforms).
#[cfg(target_arch = "aarch64")]
pub fn dd_sum(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_dd_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Deterministic high-accuracy dot product (double-double accumulation).
#[cfg(target_arch = "aarch64")]
pub fn dd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    // SAFETY: `a`/`b` are both `n` contiguous f32.
    unsafe { lfm_dd_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as i32) }
}

/// Horizontal sum (ADDV/FADDP), the NEON analog of a Metal threadgroup reduce.
#[cfg(target_arch = "aarch64")]
pub fn reduce_sum(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Horizontal max (FMAXV/FMAXP). Returns `-inf` for an empty slice.
#[cfg(target_arch = "aarch64")]
pub fn reduce_max(x: &[f32]) -> f32 {
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_max_f32(x.as_ptr(), x.len() as i32) }
}

/// In-register byte permute over a 16-entry table (TBL/TBX) — the NEON analog of Metal
/// `simd_shuffle`. `out[i] = table16[idx[i]]` for `idx<16`, else 0.
#[cfg(target_arch = "aarch64")]
pub fn permute_u8(table16: &[u8; 16], idx: &[u8], out: &mut [u8]) {
    assert_eq!(idx.len(), out.len());
    // SAFETY: table is 16 bytes; idx/out are both `n` bytes.
    unsafe {
        lfm_permute_u8(
            table16.as_ptr(),
            idx.as_ptr(),
            out.as_mut_ptr(),
            idx.len() as i32,
        )
    };
}

/// In-place radix-2 Cooley-Tukey FFT on interleaved `[re,im]` f32 (complex butterfly via FCMLA).
/// `data.len()` must be even and `n = data.len()/2` a power of two — asserted, since radix-2 has
/// no meaning otherwise (and would index out of bounds). `inverse` scales by `1/n`.
/// **Precondition:** FEAT_FCMA (check [`neon_features`]).
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
pub fn s8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // Real asserts: the kernel indexes m*k / k*n / m*n through raw pointers.
    assert_eq!(a.len(), m * k, "s8_gemm: a.len() != m*k");
    assert_eq!(b.len(), k * n, "s8_gemm: b.len() != k*n");
    assert_eq!(c.len(), m * n, "s8_gemm: c.len() != m*n");
    // Fail loudly rather than SIGILL: SMMLA needs FEAT_I8MM (absent on e.g. M1).
    assert!(neon_features().i8mm, "s8_gemm requires FEAT_I8MM");
    // SAFETY: slices sized M*K / K*N / M*N; FEAT_I8MM asserted above.
    unsafe {
        lfm_s8_gemm_s32(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as i32,
            n as i32,
            k as i32,
        )
    };
}

/// Depthwise causal conv1d with bf16 storage and f32 accumulate (single bf16 RNE store),
/// mirroring the Metal `depthwise_causal_conv1d_bf16`. `u:[B,D,L]`, `w:[D,K]`, `bias:[D]`,
/// `out:[B,D,Lout]` — all raw bf16 bits. **Precondition:** FEAT_BF16 (check [`neon_features`]).
#[cfg(target_arch = "aarch64")]
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
    assert!(
        neon_features().bf16,
        "depthwise_causal_conv1d_bf16 requires FEAT_BF16"
    );
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

/// Elementwise complex multiply in ComplexMul.metal's FIXED evaluation order (no FMA):
/// `out = ((ar·br) − (ai·bi), (ar·bi) + (ai·br))`, each rounding separate — deterministic,
/// bit-identical to the same-order scalar. `a`/`b`/`out` are interleaved `[re,im]`, equal even
/// lengths. Baseline NEON (no feature gate).
#[cfg(target_arch = "aarch64")]
pub fn complex_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert!(
        a.len() % 2 == 0,
        "complex_mul: interleaved [re,im] needs an even length"
    );
    assert_eq!(a.len(), b.len(), "complex_mul: a.len() != b.len()");
    assert_eq!(a.len(), out.len(), "complex_mul: out.len() != a.len()");
    // SAFETY: all three slices hold n interleaved complex pairs.
    unsafe {
        lfm_complex_mul_f32(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            (a.len() / 2) as i32,
        )
    };
}

/// Deterministic 3-tap depthwise conv1d, forward window (`depthwise3` in Depthwise3.metal):
/// `y[t] = x[t]·w0 + x[t+1]·w1 + x[t+2]·w2` with zero-pad on the right — fixed multiply-add
/// order, no FMA. `x`/`y` are `[B,C,L]`, `k` is `[C,3]`. Baseline NEON.
#[cfg(target_arch = "aarch64")]
pub fn depthwise3(x: &[f32], k: &[f32], y: &mut [f32], bn: usize, c: usize, l: usize) {
    assert_eq!(x.len(), bn * c * l, "depthwise3: x.len() != B*C*L");
    assert_eq!(k.len(), c * 3, "depthwise3: k.len() != C*3");
    assert_eq!(y.len(), x.len(), "depthwise3: y.len() != x.len()");
    // SAFETY: slices sized per the asserts.
    unsafe {
        lfm_depthwise3_f32(
            x.as_ptr(),
            k.as_ptr(),
            y.as_mut_ptr(),
            bn as i32,
            c as i32,
            l as i32,
        )
    };
}

/// Deterministic 3-tap depthwise conv1d, causal window (`depthwise3_causal`): `y[t] =
/// x[t−2]·w0 + x[t−1]·w1 + x[t]·w2` (left-pad 2) — the LFM2 short-conv orientation, fixed
/// order, no FMA: this is the bit-exactness instrument; the FMA path is [`conv1d_update_f32`].
#[cfg(target_arch = "aarch64")]
pub fn depthwise3_causal(x: &[f32], k: &[f32], y: &mut [f32], bn: usize, c: usize, l: usize) {
    assert_eq!(x.len(), bn * c * l, "depthwise3_causal: x.len() != B*C*L");
    assert_eq!(k.len(), c * 3, "depthwise3_causal: k.len() != C*3");
    assert_eq!(y.len(), x.len(), "depthwise3_causal: y.len() != x.len()");
    // SAFETY: slices sized per the asserts.
    unsafe {
        lfm_depthwise3_causal_f32(
            x.as_ptr(),
            k.as_ptr(),
            y.as_mut_ptr(),
            bn as i32,
            c as i32,
            l as i32,
        )
    };
}

/// Fused LFM2 ShortConv decode-step update (conv1d_update.rs): `y = C ⊙ conv1d_causal(B ⊙ x,
/// w, state)` in one call, state advanced functionally. `bcx` `[B,3D,T]` in HF chunk order
/// (B-gate | C-gate | x), `state` `[B,D,K−1]`, `w` `[D,K]`, `out` `[B,D,T+K−1]` = `[y |
/// new_state]`. Multiply-adds are FMA-contracted — the trained regime (Tri Dao's CUDA kernel);
/// use [`depthwise3_causal`] when bit-exact strict order matters. `K ≤ 8` (the register-window
/// bound the GPU kernel shares).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_f32(
    bcx: &[f32],
    state: &[f32],
    w: &[f32],
    out: &mut [f32],
    bn: usize,
    d: usize,
    t: usize,
    k: usize,
) {
    assert!(
        (1..=8).contains(&k),
        "conv1d_update: K={k} outside the register window 1..=8"
    );
    assert_eq!(
        bcx.len(),
        bn * 3 * d * t,
        "conv1d_update: bcx.len() != B*3D*T"
    );
    assert_eq!(
        state.len(),
        bn * d * (k - 1),
        "conv1d_update: state.len() != B*D*(K-1)"
    );
    assert_eq!(w.len(), d * k, "conv1d_update: w.len() != D*K");
    assert_eq!(
        out.len(),
        bn * d * (t + k - 1),
        "conv1d_update: out.len() != B*D*(T+K-1)"
    );
    // SAFETY: slices sized per the asserts; kernel stays in those bounds.
    unsafe {
        lfm_conv1d_update_f32(
            bcx.as_ptr(),
            state.as_ptr(),
            w.as_ptr(),
            out.as_mut_ptr(),
            bn as i32,
            d as i32,
            t as i32,
            k as i32,
        );
    }
}

/// bf16-storage variant of [`conv1d_update_f32`] (raw bf16 bits): compute in f32, with `B⊙x`
/// and the conv output rounded through bf16 exactly where torch materializes them — the
/// trained regime's rounding points. Baseline NEON (RNE via integer round, no FEAT_BF16 needed).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_bf16(
    bcx: &[u16],
    state: &[u16],
    w: &[u16],
    out: &mut [u16],
    bn: usize,
    d: usize,
    t: usize,
    k: usize,
) {
    assert!(
        (1..=8).contains(&k),
        "conv1d_update: K={k} outside the register window 1..=8"
    );
    assert_eq!(
        bcx.len(),
        bn * 3 * d * t,
        "conv1d_update: bcx.len() != B*3D*T"
    );
    assert_eq!(
        state.len(),
        bn * d * (k - 1),
        "conv1d_update: state.len() != B*D*(K-1)"
    );
    assert_eq!(w.len(), d * k, "conv1d_update: w.len() != D*K");
    assert_eq!(
        out.len(),
        bn * d * (t + k - 1),
        "conv1d_update: out.len() != B*D*(T+K-1)"
    );
    // SAFETY: slices sized per the asserts; kernel stays in those bounds.
    unsafe {
        lfm_conv1d_update_bf16(
            bcx.as_ptr(),
            state.as_ptr(),
            w.as_ptr(),
            out.as_mut_ptr(),
            bn as i32,
            d as i32,
            t as i32,
            k as i32,
        );
    }
}

/// Raw single-step form of the fused conv update for the lane-team ShortConv block
/// ([`super::decode`]): bcx `[1,3H,1]` == a contiguous `[3H]` B|C|x plane, T==1.
/// SAFETY: caller guarantees plane sizes (bcx 3H, state H·(K-1), w H·K, out H·K) and
/// kernel availability.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) unsafe fn conv1d_update_bf16_ptr(
    bcx: *const u16,
    state: *const u16,
    w: *const u16,
    out: *mut u16,
    d: usize,
    k: usize,
) {
    lfm_conv1d_update_bf16(bcx, state, w, out, 1, d as i32, 1, k as i32);
}

// flashkern procedures only exist on aarch64 with the kernel built in, so the tests do too —
// they run on the macOS arm64 CI leg (rust-voice.yml), where the hardware actually executes
// BFMMLA/FCMLA/SMMLA. On x86 CI the crate still builds; there is simply nothing here to run.
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
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
        if skip(
            crate::flashkern::native_engine::bf16_gemm_available(),
            "gemm_v2",
        ) {
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
            assert!(crate::flashkern::native_engine::process_engine()
                .bf16_gemm_f32(&ab, &bb, &mut c, m, n, k, false,));
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
        let orig: Vec<f32> = (0..16)
            .map(|i| ((i * 37 % 11) as f32 / 11.0) - 0.5)
            .collect();
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
                let s: i32 = (0..k)
                    .map(|kk| a[i * k + kk] as i32 * b[kk * n + j] as i32)
                    .sum();
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

    #[test]
    fn complex_mul_bitexact_vs_fixed_order_scalar() {
        // The kernel promises ComplexMul.metal's exact op order (no FMA), so the SIMD result
        // must be BIT-identical to the same-order scalar — including near-cancellation inputs.
        let n = 37usize; // ragged tail exercises the scalar edge
        let mut a = vec![0f32; n * 2];
        let mut b = vec![0f32; n * 2];
        for i in 0..n {
            let t = i as f32;
            a[i * 2] = 1.0 + t * 1e-3;
            a[i * 2 + 1] = 1.0 + t * 1.0001e-3;
            b[i * 2] = 1.0 + t * 1.0002e-3;
            b[i * 2 + 1] = 1.0 + t * 0.9999e-3;
        }
        let mut out = vec![0f32; n * 2];
        complex_mul(&a, &b, &mut out);
        for i in 0..n {
            let (ar, ai, br, bi) = (a[2 * i], a[2 * i + 1], b[2 * i], b[2 * i + 1]);
            let wr = (ar * br) - (ai * bi);
            let wi = (ar * bi) + (ai * br);
            assert_eq!(out[2 * i].to_bits(), wr.to_bits(), "re i={i}");
            assert_eq!(out[2 * i + 1].to_bits(), wi.to_bits(), "im i={i}");
        }
    }

    #[test]
    fn depthwise3_both_windows_bitexact_vs_scalar() {
        // Fixed order + no FMA ⇒ SIMD body must match the same-order scalar bit-for-bit,
        // across lengths that hit the head/vector/tail splits.
        for &l in &[1usize, 2, 3, 7, 9, 64] {
            let (bn, c) = (2usize, 3usize);
            let x: Vec<f32> = (0..bn * c * l)
                .map(|i| ((i * 31 % 17) as f32 / 17.0) - 0.5)
                .collect();
            let k: Vec<f32> = (0..c * 3)
                .map(|i| ((i * 13 % 7) as f32 / 7.0) - 0.5)
                .collect();
            let mut fwd = vec![0f32; x.len()];
            let mut cau = vec![0f32; x.len()];
            depthwise3(&x, &k, &mut fwd, bn, c, l);
            depthwise3_causal(&x, &k, &mut cau, bn, c, l);
            for b in 0..bn {
                for ci in 0..c {
                    let row = &x[(b * c + ci) * l..][..l];
                    let (w0, w1, w2) = (k[ci * 3], k[ci * 3 + 1], k[ci * 3 + 2]);
                    for t in 0..l {
                        let xf = |i: isize| {
                            if i >= 0 && (i as usize) < l {
                                row[i as usize]
                            } else {
                                0.0
                            }
                        };
                        let f = {
                            let acc = (xf(t as isize) * w0) + (xf(t as isize + 1) * w1);
                            acc + (xf(t as isize + 2) * w2)
                        };
                        let ca = {
                            let acc = (xf(t as isize - 2) * w0) + (xf(t as isize - 1) * w1);
                            acc + (xf(t as isize) * w2)
                        };
                        let o = (b * c + ci) * l + t;
                        assert_eq!(fwd[o].to_bits(), f.to_bits(), "fwd L={l} t={t}");
                        assert_eq!(cau[o].to_bits(), ca.to_bits(), "causal L={l} t={t}");
                    }
                }
            }
        }
    }

    // Scalar reference for the fused update — a straight port of conv1d_update.rs's `cpu_ref`
    // (non-FMA), optionally with the bf16 rounding points of `cpu_ref_bf16_bx` + bf16 stores.
    fn update_ref(
        bcx: &[f32],
        state: &[f32],
        w: &[f32],
        bn: usize,
        d: usize,
        t_len: usize,
        k: usize,
        bf16_regime: bool,
    ) -> Vec<f32> {
        let km1 = k - 1;
        let mut out = vec![0f32; bn * d * (t_len + km1)];
        for bi in 0..bn {
            for c in 0..d {
                let brow = &bcx[((bi * 3) * d + c) * t_len..][..t_len];
                let crow = &bcx[((bi * 3 + 1) * d + c) * t_len..][..t_len];
                let xrow = &bcx[((bi * 3 + 2) * d + c) * t_len..][..t_len];
                let srow = &state[(bi * d + c) * km1..][..km1];
                let orow = &mut out[(bi * d + c) * (t_len + km1)..][..t_len + km1];
                let mut win = [0f32; 8];
                win[..km1].copy_from_slice(srow);
                for t in 0..t_len {
                    let bx = brow[t] * xrow[t];
                    win[k - 1] = if bf16_regime {
                        bf16::from_f32(bx).to_f32()
                    } else {
                        bx
                    };
                    let mut acc = 0f32;
                    for j in 0..k {
                        acc += w[c * k + j] * win[j];
                    }
                    let cv = if bf16_regime {
                        bf16::from_f32(acc).to_f32()
                    } else {
                        acc
                    };
                    let y = crow[t] * cv;
                    orow[t] = if bf16_regime {
                        bf16::from_f32(y).to_f32()
                    } else {
                        y
                    };
                    for j in 0..km1 {
                        win[j] = win[j + 1];
                    }
                }
                orow[t_len..].copy_from_slice(&win[..km1]);
            }
        }
        out
    }

    #[test]
    fn conv1d_update_f32_matches_reference() {
        // The FIR rewrite must reproduce the register-window kernel: y within FMA-vs-not
        // tolerance, the carried state BIT-exact (it's a plain product, no accumulation).
        for &(bn, d, t_len, k) in &[
            (1usize, 8usize, 1usize, 3usize),
            (2, 5, 12, 4),
            (1, 3, 2, 3),
        ] {
            let bcx: Vec<f32> = (0..bn * 3 * d * t_len)
                .map(|i| (i as f32 * 0.13).sin())
                .collect();
            let st: Vec<f32> = (0..bn * d * (k - 1))
                .map(|i| (i as f32 * 0.07).cos())
                .collect();
            let w: Vec<f32> = (0..d * k).map(|i| 0.1 + 0.02 * i as f32).collect();
            let mut out = vec![0f32; bn * d * (t_len + k - 1)];
            conv1d_update_f32(&bcx, &st, &w, &mut out, bn, d, t_len, k);
            let want = update_ref(&bcx, &st, &w, bn, d, t_len, k, false);
            for (row, (g, r)) in out.iter().zip(&want).enumerate() {
                let pos = row % (t_len + k - 1);
                if pos < t_len {
                    assert!((g - r).abs() < 1e-5, "y b/d/t={row}: {g} vs {r}");
                } else {
                    assert_eq!(g.to_bits(), r.to_bits(), "state {row}: {g} vs {r}");
                }
            }
        }
    }

    #[test]
    fn conv1d_update_bf16_matches_reference() {
        // bf16 regime: Bx and the conv output round through bf16 at the torch-materialized
        // points. State must be bit-exact; y within an ulp of the bf16 reference (FMA order).
        let (bn, d, t_len, k) = (2usize, 6usize, 9usize, 3usize);
        let mk = |i: usize| bf16::from_f32(((i * 7 % 23) as f32 / 23.0) - 0.5);
        let bcx_b: Vec<u16> = (0..bn * 3 * d * t_len).map(|i| mk(i).to_bits()).collect();
        let st_b: Vec<u16> = (0..bn * d * (k - 1))
            .map(|i| mk(i + 11).to_bits())
            .collect();
        let w_b: Vec<u16> = (0..d * k).map(|i| mk(i + 29).to_bits()).collect();
        let mut out_b = vec![0u16; bn * d * (t_len + k - 1)];
        conv1d_update_bf16(&bcx_b, &st_b, &w_b, &mut out_b, bn, d, t_len, k);
        let up =
            |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| bf16::from_bits(b).to_f32()).collect() };
        let want = update_ref(&up(&bcx_b), &up(&st_b), &up(&w_b), bn, d, t_len, k, true);
        for (row, (g, r)) in out_b.iter().zip(&want).enumerate() {
            let got = bf16::from_bits(*g).to_f32();
            let pos = row % (t_len + k - 1);
            if pos < t_len {
                assert!(
                    (got - r).abs() <= 1e-2 * r.abs().max(0.1),
                    "y {row}: {got} vs {r}"
                );
            } else {
                assert_eq!(got.to_bits(), r.to_bits(), "state {row}: {got} vs {r}");
            }
        }
    }

    #[test]
    fn gemm_nt_matches_f32_bf16_ref() {
        if skip(
            crate::flashkern::native_engine::bf16_gemm_available(),
            "gemm_nt",
        ) {
            return;
        }
        // Native [N,K] weight layout, decode-side row counts, ragged K for the scalar tail.
        for &(m, k, n) in &[
            (1usize, 13usize, 7usize),
            (1, 2048, 512),
            (3, 129, 33),
            (4, 511, 64),
        ] {
            let a: Vec<bf16> = (0..m * k)
                .map(|i| bf16::from_f32((i * 7 % 23) as f32 / 23.0 - 0.5))
                .collect();
            let w: Vec<bf16> = (0..n * k)
                .map(|i| bf16::from_f32((i * 5 % 19) as f32 / 19.0 - 0.5))
                .collect();
            let ab: Vec<u16> = a.iter().map(|x| x.to_bits()).collect();
            let wb: Vec<u16> = w.iter().map(|x| x.to_bits()).collect();
            let mut c = vec![0f32; m * n];
            assert!(crate::flashkern::native_engine::process_engine()
                .bf16_gemm_f32(&ab, &wb, &mut c, m, n, k, true,));
            let mut rel = 0f32;
            for mi in 0..m {
                for j in 0..n {
                    let mut s = 0f32;
                    for kk in 0..k {
                        s += a[mi * k + kk].to_f32() * w[j * k + kk].to_f32();
                    }
                    rel = rel.max((c[mi * n + j] - s).abs() / s.abs().max(1e-6));
                }
            }
            assert!(rel < 1e-2, "nt m={m} k={k} n={n} rel={rel}");
        }
    }

    // Decode-step GEMV throughput at LFM2 linear shapes (M=1 is every decode matmul).
    // #[ignore] as a CI gate only — run explicitly:
    //   cargo test --release --lib gemv_decode_shapes_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gemv_decode_shapes_bench() {
        if skip(
            crate::flashkern::native_engine::bf16_gemm_available(),
            "gemv bench",
        ) {
            return;
        }
        let engine = crate::flashkern::native_engine::process_engine();
        for &(k, n) in &[(2048usize, 8192usize), (8192, 2048), (2048, 2048)] {
            let a: Vec<u16> = (0..k).map(|i| bf16::from_f32(rndf(i)).to_bits()).collect();
            let b: Vec<u16> = (0..k * n)
                .map(|i| bf16::from_f32(rndf(i + 7)).to_bits())
                .collect();
            let mut c = vec![0f32; n];
            assert!(engine.bf16_gemm_f32(&a, &b, &mut c, 1, n, k, false)); // warm
            let t0 = std::time::Instant::now();
            let iters = 20;
            for _ in 0..iters {
                assert!(engine.bf16_gemm_f32(&a, &b, &mut c, 1, n, k, false));
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let gbs = (k * n * 2) as f64 / (ms * 1e-3) / 1e9;
            eprintln!("gemv K={k} N={n}: {ms:.3} ms/call ({gbs:.1} GB/s effective)");
        }
    }

    fn rndf(i: usize) -> f32 {
        ((i * 37 % 23) as f32 / 23.0) - 0.5
    }

    // The engine size checks are real `assert_eq!` (not debug_assert), so a mismatched slice
    // length panics before any raw-pointer FFI access — even in release. These fire on the size
    // assert, which precedes the feature assert, so they hold regardless of the runner's CPU.
    #[test]
    #[should_panic(expected = "bf16 gemm A extent")]
    fn gemm_into_rejects_mismatched_dims() {
        let a = vec![0u16; 3]; // wrong: m*k = 2*4 = 8
        let b = vec![0u16; 8];
        let mut c = vec![0f32; 4];
        crate::flashkern::native_engine::process_engine()
            .bf16_gemm_f32(&a, &b, &mut c, 2, 2, 4, false);
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
