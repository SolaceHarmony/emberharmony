// Double-double inverse real FFT (torch.fft.irfft, c2r) on Metal.
//
// torch's MPS irfft is MPSGraph `HermiteanToRealFFTWithTensor` — f32-only (MPSGraph
// FFT has no f64), which is the precision wall this kernel removes. Same contract
// (onesided complex spectrum -> real signal, `scalingMode` = norm, Hermitian DC/Nyquist
// handling), but the inverse DFT is accumulated in DOUBLE-DOUBLE using the vendored
// `double_double.metal` primitives (prepended at compile time), so the GPU result
// tracks the true f64 inverse FFT instead of the f32 floor.
//
//   y[r][j] = scale · Σ_{k=0}^{freq-1} a_k · ( Re[r][k]·cos(2πkj/n) − Im[r][k]·sin(2πkj/n) )
//
// with a_0 = a_{n/2} = 1 (even n), else 2, and scale = norm.inverse_scale(n). The
// twiddles `exp(-2πi·m/n)` for m∈[0,n) are precomputed in f64 on the host and passed
// as double-double float4(cos.hi,cos.lo,sin.hi,sin.lo); the angle 2πkj/n folds to
// twiddle index (k·j) mod n. `scale` is passed as a dd pair so the 1/n normalization
// is exact to dd precision too.

struct IrfftDdParams {
    uint m;        // number of spectra (rows)
    uint n;        // output length
    uint freq;     // n/2 + 1
    uint n_even;   // 1 if n even (Nyquist bin weight 1), else 0
    float scale_hi;
    float scale_lo;
};

kernel void irfft_dd(
    constant IrfftDdParams& p   [[buffer(0)]],
    const device float* re      [[buffer(1)]],   // [M, freq]
    const device float* im      [[buffer(2)]],   // [M, freq]
    device float* out           [[buffer(3)]],   // [M, n]
    const device float4* tw     [[buffer(4)]],   // [n] dd twiddles cos/sin
    uint gid                    [[thread_position_in_grid]]
) {
    uint total = p.m * p.n;
    if (gid >= total) { return; }
    uint r = gid / p.n;
    uint j = gid % p.n;

    double_double acc = double_double(0.0f);
    uint nyq = p.n / 2;
    for (uint k = 0; k < p.freq; k++) {
        // angle 2πkj/n  ->  twiddle index (k·j) mod n  (ulong to avoid overflow).
        uint idx = uint((ulong(k) * ulong(j)) % ulong(p.n));
        complex_dd cs = unpack_cdd(tw[idx]);     // cs.re = cos_dd, cs.im = sin_dd
        float a = (k == 0u || (p.n_even == 1u && k == nyq)) ? 1.0f : 2.0f;
        double_double re_dd = double_double(re[r * p.freq + k]);
        double_double im_dd = double_double(im[r * p.freq + k]);
        // Re·cos − Im·sin, all in dd.
        double_double t = dd_sub(dd_mul(re_dd, cs.re), dd_mul(im_dd, cs.im));
        t = dd_mul(t, double_double(a));
        acc = dd_add(acc, t);
    }
    // × scale (1/n for "backward") in dd, then a single rounding to f32.
    double_double scale = double_double(p.scale_hi, p.scale_lo);
    out[gid] = dd_to_float(dd_mul(acc, scale));
}
