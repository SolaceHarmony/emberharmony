// Double-double FFT convolution.
//
// Your FFTConv.metal algorithm (rfft -> multiply -> irfft -> +u·D, radix-2 in
// threadgroup memory), re-expressed in DOUBLE-DOUBLE precision using your
// double_double.metal primitives: complex_dd state, cdd_mul / cdd_add / cdd_sub /
// cdd_conj for the butterflies and the frequency-domain multiply, twiddle_dd for
// the twiddles, dd_to_float for the single final rounding. This realizes the
// "strict FP32 -> double-double" extended-precision plan so the whole convolution
// stays ~f64-accurate on the GPU (Metal has no f64). `double_double.metal` is
// prepended at compile time, so all the arithmetic here is yours.

inline uint bit_reverse_dd(uint x, uint bits) {
    uint r = 0;
    for (uint i = 0; i < bits; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

struct FFTConvParams { uint batch; uint channels; uint seqlen; uint fft_size; };

// `tw` is the host-precomputed double-double twiddle table: tw[j] = exp(-2πi·j/fft_size)
// packed as float4(re.hi, re.lo, im.hi, im.lo) for j in [0, fft_size/2). Computing the
// twiddles in f64 on the host (instead of your f32 `twiddle_dd` placeholder — the
// "DD Taylor series" TODO) is what lets the exact dd butterflies actually go below f32.
inline void fft_radix2_dd(threadgroup complex_dd* data, const device float4* tw, uint fft_size, uint tid) {
    uint log2n = 0, temp = fft_size;
    while (temp > 1) { temp >>= 1; log2n++; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < fft_size) {
        uint rev = bit_reverse_dd(tid, log2n);
        if (tid < rev) { complex_dd t = data[tid]; data[tid] = data[rev]; data[rev] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1u << stage;
        uint m2 = m << 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < fft_size) {
            uint k = tid & (m - 1);
            uint j = (tid >> stage) << (stage + 1);
            uint idx = j + k;
            uint idx_pair = idx + m;
            if (idx_pair < fft_size) {
                // twiddle(k, m2) = exp(-2πi·k/m2) = tw[k·(fft_size/m2)].
                complex_dd w = unpack_cdd(tw[k * (fft_size >> (stage + 1))]);
                complex_dd t = cdd_mul(w, data[idx_pair]);
                complex_dd u = data[idx];
                data[idx] = cdd_add(u, t);
                data[idx_pair] = cdd_sub(u, t);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void ifft_radix2_dd(threadgroup complex_dd* data, const device float4* tw, uint fft_size, uint tid) {
    if (tid < fft_size) { data[tid].im = dd_neg(data[tid].im); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_radix2_dd(data, tw, fft_size, tid);
    if (tid < fft_size) {
        // 1/fft_size is exact in f32 (fft_size is a power of two).
        double_double scale = double_double(1.0f / float(fft_size));
        data[tid].re = dd_mul(data[tid].re, scale);
        data[tid].im = dd_neg(dd_mul(data[tid].im, scale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void fft_conv_dd(
    constant FFTConvParams& p       [[buffer(0)]],
    const device float* u           [[buffer(1)]],   // (B, C, seqlen)
    const device float2* k_f        [[buffer(2)]],   // (C, fft_size/2+1) half-spectrum
    const device float* D           [[buffer(3)]],   // (C,) skip term
    device float* y                 [[buffer(4)]],   // (B, C, seqlen)
    const device float4* tw         [[buffer(5)]],   // dd twiddle table [fft_size/2]
    threadgroup complex_dd* shared  [[threadgroup(0)]],
    uint2 gid                       [[threadgroup_position_in_grid]],
    uint2 tid_v                     [[thread_position_in_threadgroup]]
) {
    uint b = gid.x;
    uint c = gid.y;
    uint tid = tid_v.x;
    if (b >= p.batch || c >= p.channels) { return; }

    if (tid < p.seqlen) {
        uint ui = (b * p.channels + c) * p.seqlen + tid;
        shared[tid] = complex_dd(double_double(u[ui]), double_double(0.0f));
    } else if (tid < p.fft_size) {
        shared[tid] = complex_dd(double_double(0.0f), double_double(0.0f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    fft_radix2_dd(shared, tw, p.fft_size, tid);

    uint half_sz = p.fft_size / 2 + 1;
    if (tid < half_sz) {
        uint ki = c * half_sz + tid;
        shared[tid] = cdd_mul(shared[tid], complex_dd(k_f[ki]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= half_sz && tid < p.fft_size) {
        uint mir = p.fft_size - tid;
        shared[tid] = cdd_conj(shared[mir]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ifft_radix2_dd(shared, tw, p.fft_size, tid);

    if (tid < p.seqlen) {
        uint ui = (b * p.channels + c) * p.seqlen + tid;
        y[ui] = dd_to_float(shared[tid].re) + u[ui] * D[c];
    }
}
