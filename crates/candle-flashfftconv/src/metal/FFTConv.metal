// Complete FFT convolution in Metal
// No host round-trips - entire rfft -> multiply -> irfft -> bias on GPU
#include <metal_stdlib>
using namespace metal;

// Complex number for FFT
struct Complex {
    float real;
    float imag;
};

// Twiddle factor for FFT butterfly
inline Complex twiddle(uint k, uint n) {
    float angle = -6.283185307179586f * float(k) / float(n);
    return Complex{cos(angle), sin(angle)};
}

// Complex multiply with fixed evaluation order
inline Complex cmul(Complex a, Complex b) {
    float real = (a.real * b.real) - (a.imag * b.imag);
    float imag = (a.real * b.imag) + (a.imag * b.real);
    return Complex{real, imag};
}

// Complex add
inline Complex cadd(Complex a, Complex b) {
    return Complex{a.real + b.real, a.imag + b.imag};
}

// Complex subtract
inline Complex csub(Complex a, Complex b) {
    return Complex{a.real - b.real, a.imag - b.imag};
}

// Bit-reverse permutation for FFT
inline uint bit_reverse(uint x, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Radix-2 Cooley-Tukey FFT (in-place, threadgroup memory)
// Assumes fft_size is power of 2
inline void fft_radix2(threadgroup Complex* data, uint fft_size, uint tid) {
    uint log2n = 0;
    uint temp = fft_size;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    // Bit-reverse permutation
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < fft_size) {
        uint rev = bit_reverse(tid, log2n);
        if (tid < rev) {
            Complex temp = data[tid];
            data[tid] = data[rev];
            data[rev] = temp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages
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
                Complex w = twiddle(k, m2);
                Complex t = cmul(w, data[idx_pair]);
                Complex u = data[idx];

                data[idx] = cadd(u, t);
                data[idx_pair] = csub(u, t);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Inverse FFT (same as forward but conjugate twiddles and normalize)
inline void ifft_radix2(threadgroup Complex* data, uint fft_size, uint tid) {
    // Conjugate input
    if (tid < fft_size) {
        data[tid].imag = -data[tid].imag;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward FFT
    fft_radix2(data, fft_size, tid);

    // Conjugate output and normalize
    if (tid < fft_size) {
        float scale = 1.0f / float(fft_size);
        data[tid].real = data[tid].real * scale;
        data[tid].imag = -data[tid].imag * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct FFTConvParams {
    uint batch;
    uint channels;
    uint seqlen;
    uint fft_size;  // Must be power of 2, typically 2*seqlen
};

// Complete FFT convolution: rfft(u) * k_f -> irfft -> add bias
// One kernel does the entire pipeline for one (batch, channel) pair
kernel void fft_conv(
    constant FFTConvParams& p       [[buffer(0)]],
    const device float* u           [[buffer(1)]], // (batch, channels, seqlen)
    const device float2* k_f        [[buffer(2)]], // (channels, fft_size/2+1) pre-computed spectrum
    const device float* D           [[buffer(3)]], // (channels,) bias per channel
    device float* y                 [[buffer(4)]], // (batch, channels, seqlen) output
    threadgroup Complex* shared     [[threadgroup(0)]],
    uint2 gid                       [[threadgroup_position_in_grid]],   // candle mod: was thread_position_in_grid
    uint2 tid_v                     [[thread_position_in_threadgroup]]) // candle mod: arity must match gid (uint2)
{
    uint b = gid.x;  // batch index
    uint c = gid.y;  // channel index
    uint tid = tid_v.x;

    if (b >= p.batch || c >= p.channels) return;

    // Load input signal into threadgroup (real FFT: zero imaginary)
    if (tid < p.seqlen) {
        uint u_idx = (b * p.channels + c) * p.seqlen + tid;
        shared[tid] = Complex{u[u_idx], 0.0f};
    } else if (tid < p.fft_size) {
        shared[tid] = Complex{0.0f, 0.0f};  // Zero-pad
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward FFT
    fft_radix2(shared, p.fft_size, tid);

    // Complex multiply with kernel spectrum (only need first half + Nyquist for real FFT)
    uint half_sz = p.fft_size / 2 + 1;   // candle mod: `half` is a reserved Metal type name
    if (tid < half_sz) {
        uint k_idx = c * half_sz + tid;
        Complex k = Complex{k_f[k_idx].x, k_f[k_idx].y};
        shared[tid] = cmul(shared[tid], k);
    }

    // Ensure first-half writes are visible before mirroring to second half
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Mirror Hermitian symmetry for second half (real IFFT)
    if (tid >= half_sz && tid < p.fft_size) {
        uint mirror = p.fft_size - tid;  // mirror in (1..half-1)
        // Read from already-updated first half; imaginary part is conjugated
        shared[tid] = Complex{shared[mirror].real, -shared[mirror].imag};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse FFT
    ifft_radix2(shared, p.fft_size, tid);

    // Write output (truncate to seqlen, add bias)
    if (tid < p.seqlen) {
        uint u_idx = (b * p.channels + c) * p.seqlen + tid;
        float bias = D[c];
        float u_val = u[u_idx];

        // y = irfft_result + u * D
        y[u_idx] = shared[tid].real + (u_val * bias);
    }
}

// Simpler version: just RFFT (for testing)
kernel void rfft_kernel(
    constant uint& n                [[buffer(0)]], // fft_size
    constant uint& seqlen           [[buffer(1)]], // input length
    const device float* input       [[buffer(2)]], // (seqlen,)
    device float2* output           [[buffer(3)]], // (fft_size/2+1,) complex output
    threadgroup Complex* shared     [[threadgroup(0)]],
    uint tid                        [[thread_position_in_threadgroup]])
{
    // Load real input, zero-pad imaginary
    if (tid < seqlen) {
        shared[tid] = Complex{input[tid], 0.0f};
    } else if (tid < n) {
        shared[tid] = Complex{0.0f, 0.0f};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // FFT
    fft_radix2(shared, n, tid);

    // Write first half + Nyquist
    uint half_sz = n / 2 + 1;   // candle mod: `half` is a reserved Metal type name
    if (tid < half_sz) {
        output[tid] = float2(shared[tid].real, shared[tid].imag);
    }
}

// Simpler version: just IRFFT (for testing)
kernel void irfft_kernel(
    constant uint& n                [[buffer(0)]], // fft_size
    constant uint& seqlen           [[buffer(1)]], // output length
    const device float2* input      [[buffer(2)]], // (fft_size/2+1,) complex input
    device float* output            [[buffer(3)]], // (seqlen,) real output
    threadgroup Complex* shared     [[threadgroup(0)]],
    uint tid                        [[thread_position_in_threadgroup]])
{
    uint half_sz = n / 2 + 1;   // candle mod: `half` is a reserved Metal type name

    // Load complex input with Hermitian symmetry
    if (tid < half_sz) {
        shared[tid] = Complex{input[tid].x, input[tid].y};
    } else if (tid < n) {
        // Mirror from the provided half-spectrum INPUT (not shared) to avoid races
        uint mirror = n - tid;  // in 1..half-1
        shared[tid] = Complex{input[mirror].x, -input[mirror].y};
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse FFT
    ifft_radix2(shared, n, tid);

    // Write real part (truncate to seqlen)
    if (tid < seqlen) {
        output[tid] = shared[tid].real;
    }
}
