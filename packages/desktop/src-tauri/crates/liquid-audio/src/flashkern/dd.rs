//! Double-double (DD) arithmetic — the CPU port of the vendored `double_double.metal`
//! (candle-flashfftconv), Sydney's extended-precision toolkit for the Metal JIT kernels:
//! two f32 limbs giving ~2×24-bit precision where the GPU has no f64.
//!
//! This is a **formulation-faithful** port, not a textbook reimplementation: the error-free
//! transforms (`two_sum` Knuth form, `two_prod` via FMA), `dd_add`'s double renormalization,
//! and `dd_mul`'s single combined cross-term are kept exactly as the Metal source writes
//! them, so the CPU dd kernels ([`super::fanout::fused_fft_conv_dd`], `irfft_dd`) compute the
//! same sequence of f32 roundings the GPU kernels do. Sources cited in the Metal file:
//! Dekker (1971), Knuth TAOCP v2, Shewchuk (1997).
//!
//! The one thing the Metal file could not do is done here on the host exactly as the JIT
//! wrappers do it: twiddles come from **f64** `cos`/`sin` split into dd limbs
//! ([`dd_from_f64`], [`fft_twiddles_dd`], [`irfft_twiddles_dd`]) — replacing the f32
//! `twiddle_dd` placeholder (the "DD Taylor series" TODO) so the exact dd butterflies
//! actually resolve below the f32 floor.

/// A double-double value: `hi` carries the f32 rounding of the true value, `lo` the residual.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f32,
    pub lo: f32,
}

impl Dd {
    #[inline]
    pub fn new(hi: f32, lo: f32) -> Self {
        Dd { hi, lo }
    }
    /// The single-float constructor (`double_double(x)` in Metal).
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Dd { hi: x, lo: 0.0 }
    }
}

/// Complex double-double: two [`Dd`] planes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CDd {
    pub re: Dd,
    pub im: Dd,
}

impl CDd {
    #[inline]
    pub fn new(re: Dd, im: Dd) -> Self {
        CDd { re, im }
    }
    /// `complex_dd(float2 z)`: lift an (re, im) f32 pair.
    #[inline]
    pub fn from_f32(re: f32, im: f32) -> Self {
        CDd {
            re: Dd::from_f32(re),
            im: Dd::from_f32(im),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Error-free transformations (verbatim op order from double_double.metal)
// ---------------------------------------------------------------------------------------

/// Quick-Two-Sum (assumes |a| ≥ |b|): `s = fl(a+b)`, `e` the exact residual.
#[inline]
pub fn quick_two_sum(a: f32, b: f32) -> Dd {
    let s = a + b;
    let e = b - (s - a);
    Dd::new(s, e)
}

/// Two-Sum (Knuth, no ordering assumption): exact `a + b` as (rounded, residual).
#[inline]
pub fn two_sum(a: f32, b: f32) -> Dd {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    Dd::new(s, e)
}

/// Two-Product via FMA: `p = fl(a·b)`, `e = fma(a, b, -p)` the exact residual.
#[inline]
pub fn two_prod(a: f32, b: f32) -> Dd {
    let p = a * b;
    let e = a.mul_add(b, -p);
    Dd::new(p, e)
}

// ---------------------------------------------------------------------------------------
// Double-double arithmetic
// ---------------------------------------------------------------------------------------

/// dd + dd, with the Metal source's two-pass renormalization.
#[inline]
pub fn dd_add(a: Dd, b: Dd) -> Dd {
    let mut s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    s = quick_two_sum(s.hi, s.lo);
    s
}

/// dd + f32 (`dd_add(double_double, float)` overload).
#[inline]
pub fn dd_add_f(a: Dd, b: f32) -> Dd {
    let mut s = two_sum(a.hi, b);
    s.lo += a.lo;
    quick_two_sum(s.hi, s.lo)
}

/// dd − dd (negate-and-add, as in the Metal source).
#[inline]
pub fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, Dd::new(-b.hi, -b.lo))
}

/// dd × dd: exact head product + one combined cross-term correction.
#[inline]
pub fn dd_mul(a: Dd, b: Dd) -> Dd {
    let mut p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p.hi, p.lo)
}

/// dd × f32 (`dd_mul(double_double, float)` overload).
#[inline]
pub fn dd_mul_f(a: Dd, b: f32) -> Dd {
    let mut p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    quick_two_sum(p.hi, p.lo)
}

/// dd ÷ f32 (one Newton-style correction, as in the Metal source).
#[inline]
pub fn dd_div_f(a: Dd, b: f32) -> Dd {
    let q = a.hi / b;
    let p = two_prod(q, b);
    let e = (a.hi - p.hi - p.lo + a.lo) / b;
    quick_two_sum(q, e)
}

/// −dd.
#[inline]
pub fn dd_neg(a: Dd) -> Dd {
    Dd::new(-a.hi, -a.lo)
}

/// Round to a single f32 — the one place precision is spent.
#[inline]
pub fn dd_to_f32(a: Dd) -> f32 {
    a.hi + a.lo
}

// ---------------------------------------------------------------------------------------
// Complex double-double
// ---------------------------------------------------------------------------------------

#[inline]
pub fn cdd_add(a: CDd, b: CDd) -> CDd {
    CDd::new(dd_add(a.re, b.re), dd_add(a.im, b.im))
}

#[inline]
pub fn cdd_sub(a: CDd, b: CDd) -> CDd {
    CDd::new(dd_sub(a.re, b.re), dd_sub(a.im, b.im))
}

/// (a+bi)(c+di) — "the CRITICAL operation for FFT frequency-domain multiply".
#[inline]
pub fn cdd_mul(a: CDd, b: CDd) -> CDd {
    let ac = dd_mul(a.re, b.re);
    let bd = dd_mul(a.im, b.im);
    let re = dd_sub(ac, bd);
    let ad = dd_mul(a.re, b.im);
    let bc = dd_mul(a.im, b.re);
    let im = dd_add(ad, bc);
    CDd::new(re, im)
}

#[inline]
pub fn cdd_conj(a: CDd) -> CDd {
    CDd::new(a.re, dd_neg(a.im))
}

#[inline]
pub fn cdd_to_f32(a: CDd) -> (f32, f32) {
    (dd_to_f32(a.re), dd_to_f32(a.im))
}

// ---------------------------------------------------------------------------------------
// Host-side f64 → dd (what the JIT wrappers do before dispatch)
// ---------------------------------------------------------------------------------------

/// Split an f64 into dd limbs: `hi = fl32(x)`, `lo = fl32(x − hi)` — the split the
/// `fused_fft_conv_dd` host wrapper uses for its twiddle table and `irfft_dd` for its scale.
#[inline]
pub fn dd_from_f64(x: f64) -> Dd {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    Dd::new(hi, lo)
}

/// The `fft_conv_dd` twiddle table: `tw[j] = exp(−2πi·j/fft_size)` for `j ∈ [0, fft_size/2)`,
/// cos/sin computed in f64 and split into dd (re, im) — `float4(re.hi, re.lo, im.hi, im.lo)`
/// on the Metal side.
pub fn fft_twiddles_dd(fft_size: usize) -> Vec<CDd> {
    let n = fft_size as f64;
    (0..fft_size / 2)
        .map(|j| {
            let ang = -2.0 * std::f64::consts::PI * (j as f64) / n;
            CDd::new(dd_from_f64(ang.cos()), dd_from_f64(ang.sin()))
        })
        .collect()
}

/// The `irfft_dd` twiddle table: `tw[m] = (cos(2πm/n), sin(2πm/n))` for `m ∈ [0, n)` as dd —
/// the POSITIVE angle, exactly as the `IrfftDd` host wrapper builds it (its kernel computes
/// `re·cos − im·sin` from this table; the `.metal` header's `exp(−2πi·m/n)` comment is off by
/// a conjugate). The kernel folds the angle `2πkj/n` to index `(k·j) mod n`.
pub fn irfft_twiddles_dd(n: usize) -> Vec<CDd> {
    let nf = n as f64;
    (0..n)
        .map(|m| {
            let ang = 2.0 * std::f64::consts::PI * (m as f64) / nf;
            CDd::new(dd_from_f64(ang.cos()), dd_from_f64(ang.sin()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: usize) -> f32 {
        (((seed.wrapping_mul(2654435761)) % 100_003) as f32 / 100_003.0) * 2.0 - 1.0
    }

    #[test]
    fn transforms_are_error_free() {
        // two_sum / two_prod must be EXACT: hi+lo (in f64) equals the true f64 result.
        // (f32 sums fit f64 trivially; f32 products are exact in f64: 24+24 ≤ 53 bits.)
        for i in 0..1000 {
            let (a, b) = (rnd(i) * 1e3, rnd(i + 7919) * 1e-3);
            let s = two_sum(a, b);
            assert_eq!(s.hi as f64 + s.lo as f64, a as f64 + b as f64, "two_sum {a} {b}");
            let p = two_prod(a, b);
            assert_eq!(p.hi as f64 + p.lo as f64, a as f64 * b as f64, "two_prod {a} {b}");
        }
    }

    #[test]
    fn dd_accumulation_tracks_f64() {
        // Summing 10k mixed-magnitude terms: dd must stay within a few f32 ulps of the f64
        // truth while the naive f32 sum drifts orders of magnitude further.
        let terms: Vec<f32> = (0..10_000).map(|i| rnd(i) * 10f32.powi((i % 7) as i32 - 3)).collect();
        let f64_sum: f64 = terms.iter().map(|&x| x as f64).sum();
        let f32_sum: f32 = terms.iter().sum();
        let mut acc = Dd::default();
        for &t in &terms {
            acc = dd_add_f(acc, t);
        }
        let dd_err = (dd_to_f32(acc) as f64 - f64_sum).abs();
        let f32_err = (f32_sum as f64 - f64_sum).abs();
        let ulp = (f64_sum.abs() as f32).max(1e-30) * f32::EPSILON;
        assert!(dd_err <= 4.0 * ulp as f64, "dd err {dd_err:e} vs ulp {ulp:e}");
        assert!(dd_err < f32_err, "dd {dd_err:e} must beat naive f32 {f32_err:e}");
    }

    #[test]
    fn cdd_mul_is_correctly_rounded_under_cancellation() {
        // The dd_complex_mul.rs test data: ar·br ≈ ai·bi so the real part cancels
        // catastrophically in f32. cdd_mul must land on the f64-computed, once-rounded value.
        for i in 0..64 {
            let t = i as f32;
            let a = CDd::from_f32(1.0 + t * 1e-3, 1.0 + t * 1.0001e-3);
            let b = CDd::from_f32(1.0 + t * 1.0002e-3, 1.0 + t * 0.9999e-3);
            let (gr, gi) = cdd_to_f32(cdd_mul(a, b));
            let (ar, ai) = (a.re.hi as f64, a.im.hi as f64);
            let (br, bi) = (b.re.hi as f64, b.im.hi as f64);
            let want_r = (ar * br - ai * bi) as f32;
            let want_i = (ar * bi + ai * br) as f32;
            assert_eq!(gr.to_bits(), want_r.to_bits(), "re i={i}: {gr:e} vs {want_r:e}");
            assert_eq!(gi.to_bits(), want_i.to_bits(), "im i={i}: {gi:e} vs {want_i:e}");
        }
    }

    #[test]
    fn twiddle_tables_carry_f64_precision() {
        for n in [8usize, 64, 256] {
            let tw = fft_twiddles_dd(n);
            assert_eq!(tw.len(), n / 2);
            for (j, t) in tw.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
                let (c, s) = (ang.cos(), ang.sin());
                assert!((t.re.hi as f64 + t.re.lo as f64 - c).abs() < 1e-14, "cos n={n} j={j}");
                assert!((t.im.hi as f64 + t.im.lo as f64 - s).abs() < 1e-14, "sin n={n} j={j}");
            }
            let itw = irfft_twiddles_dd(n);
            assert_eq!(itw.len(), n);
        }
    }
}
