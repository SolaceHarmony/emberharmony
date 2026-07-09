//! x86-64 "zoo" — the Intel/AMD sibling of [`crate::neon_zoo`], bridging `csrc/x86_zoo.cpp`.
//! Same `extern "C"` kernels (the C symbol names match the NEON ones), same public API, but
//! the SIMD is SSE/AVX2/AVX-512 instead of NEON — VDPBF16PS for the bf16 tensor MAC, PSHUFB
//! for `simd_shuffle`, RCPPS/RSQRTPS for GPU fast-math, VPMADDWD for the int tensor MAC.
//!
//! **Fan-out.** A CPU's analog of the GPU threadgroup grid is its cores. As on the NEON side,
//! [`bf16_gemm_into`] fans the GEMM out over M-row blocks with rayon (reusing the global pool
//! sized by [`crate::threads`] to torch's physical-core policy) — one task per block, each
//! running the single-threaded SIMD micro-kernel. B is shared across blocks. Wider machines
//! simply get more blocks in flight.
//!
//! Build-gated on x86_64 (`cfg(has_x86_zoo)`, set by `build.rs`) and runtime-gated on the CPU
//! feature via [`X86Features`] (`is_x86_feature_detected!`). The C kernel additionally
//! dispatches internally: the bf16 GEMM takes VDPBF16PS when AVX-512-BF16 is present, else an
//! AVX2 upconvert+FMA micro-kernel (baseline on essentially all x86-64).

/// Runtime CPU-feature probe (cached). Off x86_64 every field is `false`.
#[derive(Clone, Copy, Debug, Default)]
pub struct X86Features {
    pub avx2: bool,       // AVX2 + the 256-bit baseline the zoo needs
    pub fma: bool,        // FMA3 (fmadd/fmsub — double-double + GEMM)
    pub avx512f: bool,    // AVX-512 Foundation
    pub avx512bw: bool,   // AVX-512 Byte/Word — VPMADDWD int MAC
    pub avx512vl: bool,   // AVX-512 Vector Length
    pub avx512bf16: bool, // AVX-512 BF16 — VDPBF16PS tensor MAC
}

/// The cached [`X86Features`] for the running CPU.
pub fn x86_features() -> &'static X86Features {
    use std::sync::OnceLock;
    static F: OnceLock<X86Features> = OnceLock::new();
    F.get_or_init(detect_features)
}

#[cfg(target_arch = "x86_64")]
fn detect_features() -> X86Features {
    X86Features {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
        avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_features() -> X86Features {
    X86Features::default()
}

// ---- FFI to csrc/x86_zoo.cpp (x86_64, kernel built in) ----------------------------------
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
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

/// `true` when the zoo's bf16 GEMM is built in and the CPU meets its baseline (AVX2 + FMA).
pub fn bf16_gemm_available() -> bool {
    cfg!(all(target_arch = "x86_64", has_x86_zoo)) && x86_features().avx2 && x86_features().fma
}

/// `C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16` (raw bf16 bits as `u16`), row-major, f32
/// accumulate. Fans out over M-row blocks with rayon (the CPU fan-out); each block runs the
/// SIMD micro-kernel (VDPBF16PS when AVX-512-BF16 is present, else AVX2). B is shared.
/// **Precondition:** AVX2 + FMA — verify [`bf16_gemm_available`] first. Slices sized `M*K`,
/// `K*N`, `M*N`.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn bf16_gemm_into(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;
    assert_eq!(a.len(), m * k, "bf16_gemm_into: a.len() != m*k");
    assert_eq!(b.len(), k * n, "bf16_gemm_into: b.len() != k*n");
    assert_eq!(c.len(), m * n, "bf16_gemm_into: c.len() != m*n");
    let f = x86_features();
    assert!(f.avx2 && f.fma, "bf16_gemm_into requires AVX2 + FMA");
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    if m == 1 {
        // SAFETY: slices sized K / K*N / N; AVX2+FMA asserted above.
        unsafe { lfm_bf16_gemv_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n as i32, k as i32) };
        return;
    }
    // CPU fan-out: one rayon task per block of rows (== one GPU threadgroup per (batch,head)).
    // par_chunks keeps A- and C-row blocks aligned, so no raw pointers escape a task.
    let threads = rayon::current_num_threads().max(1);
    let rows = (m.div_ceil(threads)).max(1);
    c.par_chunks_mut(rows * n)
        .zip(a.par_chunks(rows * k))
        .for_each(|(cc, aa)| {
            let mm = cc.len() / n;
            // SAFETY: aa is mm*K, b is K*N, cc is mm*N; AVX2+FMA asserted above.
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

/// GPU-style fast reciprocal-sqrt (`1/√x`) (RSQRTPS + 2 Newton steps). **Precondition:** AVX2 +
/// FMA — the C kernel is compiled `target("avx2,fma")` and the Newton steps use `_mm256_fnmadd_ps`.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn rsqrt(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let f = x86_features();
    assert!(f.avx2 && f.fma, "rsqrt requires AVX2 + FMA");
    // SAFETY: both slices are `n` f32; AVX2+FMA asserted above.
    unsafe { lfm_rsqrt_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// GPU-style fast reciprocal (`1/x`) (RCPPS + 2 Newton steps). **Precondition:** AVX2 + FMA —
/// the C kernel is compiled `target("avx2,fma")` and the Newton steps use `_mm256_fnmadd_ps`.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn recip(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let f = x86_features();
    assert!(f.avx2 && f.fma, "recip requires AVX2 + FMA");
    // SAFETY: both slices are `n` f32; AVX2+FMA asserted above.
    unsafe { lfm_recip_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as i32) };
}

/// Deterministic high-accuracy sum via double-double (FMA error-free transforms). **Precondition:** AVX2+FMA.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn dd_sum(x: &[f32]) -> f32 {
    assert!(x86_features().avx2 && x86_features().fma, "dd_sum requires AVX2+FMA");
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_dd_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Deterministic high-accuracy dot product (double-double). **Precondition:** AVX2+FMA.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn dd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(x86_features().avx2 && x86_features().fma, "dd_dot requires AVX2+FMA");
    // SAFETY: `a`/`b` are both `n` contiguous f32.
    unsafe { lfm_dd_dot_f32(a.as_ptr(), b.as_ptr(), a.len() as i32) }
}

/// Horizontal sum (AVX reduce), the analog of a Metal threadgroup reduce. **Precondition:** AVX2.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn reduce_sum(x: &[f32]) -> f32 {
    assert!(x86_features().avx2, "reduce_sum requires AVX2");
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_sum_f32(x.as_ptr(), x.len() as i32) }
}

/// Horizontal max. Returns `-inf` for an empty slice. **Precondition:** AVX2.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn reduce_max(x: &[f32]) -> f32 {
    assert!(x86_features().avx2, "reduce_max requires AVX2");
    // SAFETY: `x` is `n` contiguous f32.
    unsafe { lfm_reduce_max_f32(x.as_ptr(), x.len() as i32) }
}

/// In-register byte permute over a 16-entry table (PSHUFB) — the x86 analog of NEON TBL.
/// `out[i] = table16[idx[i]]` for `idx<16`, else 0. **Precondition:** AVX2.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn permute_u8(table16: &[u8; 16], idx: &[u8], out: &mut [u8]) {
    assert_eq!(idx.len(), out.len());
    assert!(x86_features().avx2, "permute_u8 requires AVX2");
    // SAFETY: table is 16 bytes; idx/out are both `n` bytes.
    unsafe { lfm_permute_u8(table16.as_ptr(), idx.as_ptr(), out.as_mut_ptr(), idx.len() as i32) };
}

/// In-place radix-2 Cooley-Tukey FFT on interleaved `[re,im]` f32. `data.len()` even and
/// `n = data.len()/2` a power of two (asserted). `inverse` scales by `1/n`.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn fft_radix2(data: &mut [f32], inverse: bool) {
    let n = data.len() / 2;
    assert!(
        data.len() % 2 == 0 && n >= 1 && n.is_power_of_two(),
        "fft_radix2: n must be a power of two, got data.len()={}",
        data.len()
    );
    // SAFETY: n is an asserted power of two and data holds 2*n f32.
    unsafe { lfm_fft_radix2_f32(data.as_mut_ptr(), n as i32, inverse as i32) };
}

/// int8 tensor GEMM `C(M,N) s32 = A(M,K) s8 · B(K,N) s8` via VPMADDWD. Slices sized `M*K`,
/// `K*N`, `M*N`. **Precondition:** AVX-512F + AVX-512BW + AVX-512VL (check [`x86_features`]) —
/// the C kernel is compiled `target("avx512f,avx512bw,avx512vl")` and may emit VL-width EVEX.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
pub fn s8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "s8_gemm: a.len() != m*k");
    assert_eq!(b.len(), k * n, "s8_gemm: b.len() != k*n");
    assert_eq!(c.len(), m * n, "s8_gemm: c.len() != m*n");
    let f = x86_features();
    assert!(
        f.avx512f && f.avx512bw && f.avx512vl,
        "s8_gemm requires AVX-512F + AVX-512BW + AVX-512VL"
    );
    // SAFETY: slices sized M*K / K*N / M*N; AVX-512F/BW/VL asserted above.
    unsafe {
        lfm_s8_gemm_s32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32)
    };
}

/// Depthwise causal conv1d, bf16 storage / f32 accumulate / bf16 store. `u:[B,D,L]`,
/// `w:[D,K]`, `bias:[D]`, `out:[B,D,Lout]` — raw bf16 bits. **Precondition:** AVX2+FMA.
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
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
    assert_eq!(u.len(), bn * d * l, "conv1d: u.len() != B*D*L");
    assert_eq!(w.len(), d * k, "conv1d: w.len() != D*K");
    assert_eq!(bias.len(), d, "conv1d: bias.len() != D");
    assert_eq!(out.len(), bn * d * lout, "conv1d: out.len() != B*D*Lout");
    assert!(x86_features().avx2 && x86_features().fma, "conv1d requires AVX2+FMA");
    // SAFETY: pointers sized per the asserts; AVX2+FMA asserted above.
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

// Tests run NATIVELY (this crate builds on x86-64), unlike the NEON side which needs a device
// or an emulator. Feature-specific tests skip when the runner CPU lacks the extension.
#[cfg(test)]
#[cfg(all(target_arch = "x86_64", has_x86_zoo))]
mod tests {
    use super::*;
    use half::bf16;

    fn skip(feat: bool, name: &str) -> bool {
        if !feat {
            eprintln!("{name}: CPU feature unavailable on this runner — skipping");
        }
        !feat
    }

    #[test]
    fn gemm_matches_f32_bf16_ref() {
        if skip(bf16_gemm_available(), "gemm") {
            return;
        }
        // ragged edges, large-K, and the M==1 GEMV path; fan-out exercised at M=17.
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
        if skip(x86_features().avx2 && x86_features().fma, "rsqrt") {
            return;
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
        if skip(x86_features().avx2 && x86_features().fma, "dd_sum") {
            return;
        }
        let mut x = vec![1e-2f32; 200_000];
        x[0] = 1e4;
        let reference: f64 = x.iter().map(|&v| v as f64).sum();
        let dd = dd_sum(&x) as f64;
        assert!((dd - reference).abs() / reference < 1e-4, "dd={dd} ref={reference}");
    }

    #[test]
    fn dd_sum_and_dot_handle_ragged_tail() {
        // Codex example: a large value then a non-multiple-of-8 tail of tiny values, each below
        // the running sum's f32 ULP. A plain-f32 tail add drops them (returns exactly 1e4); the
        // double-double tail must retain them. n = 1 + 7 = 8... use 13 (one full lane + 5 tail).
        if skip(x86_features().avx2 && x86_features().fma, "dd_tail") {
            return;
        }
        let mut x = vec![3e-4f32; 13];
        x[0] = 1e4;
        let want: f64 = x.iter().map(|&v| v as f64).sum(); // ≈ 10000.0036
        let got = dd_sum(&x) as f64;
        assert!((got - want).abs() < 1e-2, "dd_sum tail dropped: got={got} want={want}");
        // the naive f32 running sum loses the tail entirely — confirm we beat it.
        let naive: f32 = x.iter().fold(0f32, |a, &v| a + v);
        assert!(
            (got - want).abs() < (naive as f64 - want).abs(),
            "dd_sum ({got}) no better than naive f32 ({naive}) vs {want}"
        );
        // dd_dot with a ragged tail: Σ x·1 == Σ x, same accuracy requirement.
        let ones = vec![1f32; x.len()];
        let dot = dd_dot(&x, &ones) as f64;
        assert!((dot - want).abs() < 1e-2, "dd_dot tail dropped: got={dot} want={want}");
    }

    #[test]
    fn permute_matches_scalar() {
        if skip(x86_features().avx2, "permute") {
            return;
        }
        let table: [u8; 16] = std::array::from_fn(|i| (i * 3) as u8);
        let idx: Vec<u8> = (0..40u8).map(|i| i % 20).collect();
        let mut out = vec![0u8; idx.len()];
        permute_u8(&table, &idx, &mut out);
        for (o, &i) in out.iter().zip(&idx) {
            let want = if (i as usize) < 16 { table[i as usize] } else { 0 };
            assert_eq!(*o, want);
        }
    }

    #[test]
    fn s8_gemm_matches_scalar() {
        if skip(
            x86_features().avx512f && x86_features().avx512bw && x86_features().avx512vl,
            "s8_gemm",
        ) {
            return;
        }
        let (m, k, n) = (6usize, 40usize, 5usize);
        let a: Vec<i8> = (0..m * k).map(|i| (i as i8 % 15) - 7).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i as i8 % 13) - 6).collect();
        let mut c = vec![0i32; m * n];
        s8_gemm(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let s: i32 = (0..k).map(|kk| a[i * k + kk] as i32 * b[kk * n + j] as i32).sum();
                assert_eq!(c[i * n + j], s);
            }
        }
    }

    #[test]
    fn fft_round_trips_and_rejects_non_pow2() {
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
        let mut bad = vec![1.0f32; 12]; // n=6
        fft_radix2(&mut bad, false);
    }

    #[test]
    #[should_panic(expected = "a.len() != m*k")]
    fn gemm_rejects_mismatched_dims() {
        let a = vec![0u16; 3];
        let b = vec![0u16; 8];
        let mut c = vec![0f32; 4];
        bf16_gemm_into(&a, &b, &mut c, 2, 2, 4);
    }
}
