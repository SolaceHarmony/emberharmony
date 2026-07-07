// x86-64 "zoo" — the Intel/AMD sibling of csrc/neon_zoo.cpp. Same public `extern "C"` API
// and the same GPU-idiom → SIMD-opcode mapping, but expressed in SSE/AVX2/AVX-512 instead of
// NEON. build.rs compiles exactly one of the two per target arch, so the Rust FFI is arch-
// agnostic. Each function runtime-dispatches on CPUID (`__builtin_cpu_supports`) and is
// confined behind a per-function `target(...)` attribute so no gated opcode leaks into an
// ungated function.
//
//   ARM (neon_zoo)                         x86 (this file)
//   BFMMLA / BFDOT (vbfmmlaq/vbfdotq)   ->  VDPBF16PS (_mm512_dpbf16_ps), AVX512-BF16
//   (bf16 store) BFCVT                  ->  VCVTNEPS2BF16 (_mm512_cvtneps_pbh)
//   TBL/TBX (vqtbl1q_u8)               ->  PSHUFB (_mm256_shuffle_epi8)
//   ADDV/FADDP (vaddvq_f32)            ->  AVX horizontal reduce
//   FRECPE/FRSQRTE (+Newton)           ->  RCPPS/RSQRTPS (_mm256_rcp_ps/_mm256_rsqrt_ps)+Newton
//   FCMLA complex butterfly            ->  AVX f32 mul/add (no native complex here)
//   double_double two_prod/two_sum     ->  FMA error-free transforms (_mm256_fmadd/ fmsub)
//   SMMLA (vmmlaq_s32)                 ->  VPMADDWD (_mm512_madd_epi16), AVX512-BW
//
// The bf16 GEMM has two real kernels dispatched by CPUID: AVX-512-BF16 (VDPBF16PS, the
// tensor MAC) when present, else an AVX2 upconvert+FMA microkernel (baseline on all x86-64).
// Both compute bf16 products with f32 accumulate — torch's CPU bf16-matmul numerics.

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <cmath>

// Every opcode-bearing function is confined to its own ISA via a per-function `target(...)`
// attribute, so nothing above AVX2 ever leaks into an ungated function (notably the AVX2
// bf16 fallback microkernel). This holds for BOTH gcc and clang on x86: clang honours
// per-function target attributes and declares the immintrin intrinsics unconditionally — so,
// unlike the aarch64 zoo where clang needs the feature in the base -march, here neither
// compiler needs a raised base ISA (build.rs compiles this TU with NO global AVX-512 flags).
// That is exactly what stops a clang binary from emitting zmm codegen inside gemm_bf16_avx2
// and then SIGILL-ing on an AVX2-only CPU that legitimately passed the AVX2 feature gate.
// MSVC understands none of this (no __attribute__, no __builtin_cpu_supports), so it is
// deliberately excluded in build.rs; this #error is the backstop if that gate ever regresses.
#if defined(_MSC_VER) && !defined(__clang__)
#error "x86_zoo.cpp requires GCC/Clang target attributes; build.rs must not compile it with MSVC"
#endif
#define X86_TGT_AVX2 __attribute__((target("avx2,fma")))
#define X86_TGT_BF16 __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))
#define X86_TGT_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl")))

// bf16 (upper 16 bits of the f32) -> f32, scalar.
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
// f32 -> bf16 bits, round-to-nearest-even (scalar, matches half::bf16 / hardware BFCVT).
static inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7fff + lsb;
    return (uint16_t)(u >> 16);
}

// =====================================================================================
// Group A — bf16 GEMM. C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16, row-major, f32 accumulate.
// =====================================================================================
namespace {

// --- AVX-512-BF16 path: VDPBF16PS microkernel, 16 output columns per row. ---
// Ap[m][kpair] = u32 packing (A[m][2p] | A[m][2p+1]<<16), broadcast to 16 lanes.
// Bp per 16-col block, per k-pair: 16 lanes × (B[2p][col] | B[2p+1][col]<<16).
X86_TGT_BF16
static void gemm_bf16_avx512(const uint16_t *A, const uint16_t *B, float *C,
                             int M, int N, int K) {
    const int Kp = (K + 1) & ~1, kp = Kp / 2;
    const int Nb = (N + 15) / 16;
    static thread_local std::vector<uint32_t> Ap; // [M][kp]
    static thread_local std::vector<uint32_t> Bp; // [Nb][kp][16]
    Ap.assign((size_t)M * kp, 0);
    Bp.assign((size_t)Nb * kp * 16, 0);
    for (int m = 0; m < M; m++)
        for (int p = 0; p < kp; p++) {
            uint32_t lo = (2 * p < K) ? A[(size_t)m * K + 2 * p] : 0;
            uint32_t hi = (2 * p + 1 < K) ? A[(size_t)m * K + 2 * p + 1] : 0;
            Ap[(size_t)m * kp + p] = lo | (hi << 16);
        }
    for (int nb = 0; nb < Nb; nb++)
        for (int p = 0; p < kp; p++)
            for (int c = 0; c < 16; c++) {
                int n = nb * 16 + c;
                if (n >= N) continue;
                uint32_t lo = (2 * p < K) ? B[(size_t)(2 * p) * N + n] : 0;
                uint32_t hi = (2 * p + 1 < K) ? B[(size_t)(2 * p + 1) * N + n] : 0;
                Bp[((size_t)nb * kp + p) * 16 + c] = lo | (hi << 16);
            }
    for (int m = 0; m < M; m++) {
        for (int nb = 0; nb < Nb; nb++) {
            __m512 acc = _mm512_setzero_ps();
            const uint32_t *bp = &Bp[((size_t)nb * kp) * 16];
            const uint32_t *ap = &Ap[(size_t)m * kp];
            for (int p = 0; p < kp; p++) {
                __m512bh a = (__m512bh)_mm512_set1_epi32((int)ap[p]);
                __m512bh b = (__m512bh)_mm512_loadu_si512((const void *)(bp + (size_t)p * 16));
                acc = _mm512_dpbf16_ps(acc, a, b);
            }
            int n0 = nb * 16, cols = N - n0 < 16 ? N - n0 : 16;
            if (cols == 16) {
                _mm512_storeu_ps(&C[(size_t)m * N + n0], acc);
            } else {
                float tmp[16];
                _mm512_storeu_ps(tmp, acc);
                for (int c = 0; c < cols; c++) C[(size_t)m * N + n0 + c] = tmp[c];
            }
        }
    }
}

// --- AVX2 baseline: upconvert bf16->f32 + FMA, 8 output columns per row. ---
X86_TGT_AVX2
static inline __m256 upconv8(const uint16_t *p) { // 8 bf16 -> 8 f32
    __m128i u16 = _mm_loadu_si128((const __m128i *)p);
    __m256i u32 = _mm256_cvtepu16_epi32(u16);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16));
}
X86_TGT_AVX2
static void gemm_bf16_avx2(const uint16_t *A, const uint16_t *B, float *C,
                           int M, int N, int K) {
    const int Nb = (N + 7) / 8;
    static thread_local std::vector<uint16_t> Brow; // padded row of B: [K][Nb*8]
    const int Np = Nb * 8;
    Brow.assign((size_t)K * Np, 0);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) Brow[(size_t)k * Np + n] = B[(size_t)k * N + n];
    for (int m = 0; m < M; m++) {
        for (int nb = 0; nb < Nb; nb++) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a = _mm256_set1_ps(bf16_to_f32(A[(size_t)m * K + k]));
                __m256 b = upconv8(&Brow[(size_t)k * Np + nb * 8]);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            int n0 = nb * 8, cols = N - n0 < 8 ? N - n0 : 8;
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            for (int c = 0; c < cols; c++) C[(size_t)m * N + n0 + c] = tmp[c];
        }
    }
}
} // namespace

extern "C" void lfm_bf16_gemm_f32_v2(const uint16_t *A, const uint16_t *B, float *C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (__builtin_cpu_supports("avx512bf16"))
        gemm_bf16_avx512(A, B, C, M, N, K);
    else
        gemm_bf16_avx2(A, B, C, M, N, K);
}

// GEMV (M==1) — just the M==1 case of the GEMM; reuse it.
extern "C" void lfm_bf16_gemv_f32(const uint16_t *A, const uint16_t *B, float *C, int N, int K) {
    lfm_bf16_gemm_f32_v2(A, B, C, 1, N, K);
}

// int8 tensor MAC via VPMADDWD (AVX-512-BW): C(M,N) s32 = A(M,K) s8 · B(K,N) s8.
namespace {
X86_TGT_AVX512
static void s8_gemm_avx512(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K) {
    // Simple per-(m,n) dot: widen s8->s16, madd pairs into s32, reduce. Kp padded to 32.
    const int Kp = (K + 31) & ~31;
    static thread_local std::vector<int16_t> Aw, Btw; // A row widened, B col-major widened
    Aw.assign((size_t)M * Kp, 0);
    Btw.assign((size_t)N * Kp, 0);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) Aw[(size_t)m * Kp + k] = A[(size_t)m * K + k];
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) Btw[(size_t)n * Kp + k] = B[(size_t)k * N + n];
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < Kp; k += 32) {
                __m512i a = _mm512_loadu_si512((const void *)&Aw[(size_t)m * Kp + k]);
                __m512i b = _mm512_loadu_si512((const void *)&Btw[(size_t)n * Kp + k]);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(a, b));
            }
            C[(size_t)m * N + n] = _mm512_reduce_add_epi32(acc);
        }
}
} // namespace
extern "C" void lfm_s8_gemm_s32(const int8_t *A, const int8_t *B, int32_t *C,
                                int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    s8_gemm_avx512(A, B, C, M, N, K);
}

// =====================================================================================
// Group B — reductions & permute
// =====================================================================================
X86_TGT_AVX2
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
extern "C" X86_TGT_AVX2 float lfm_reduce_sum_f32(const float *x, int n) {
    __m256 a = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) a = _mm256_add_ps(a, _mm256_loadu_ps(x + i));
    float acc = hsum256(a);
    for (; i < n; i++) acc += x[i];
    return acc;
}
extern "C" X86_TGT_AVX2 float lfm_reduce_max_f32(const float *x, int n) {
    if (n <= 0) return -INFINITY;
    __m256 m = _mm256_set1_ps(x[0]);
    int i = 0;
    for (; i + 8 <= n; i += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(x + i));
    float tmp[8];
    _mm256_storeu_ps(tmp, m);
    float acc = tmp[0];
    for (int j = 1; j < 8; j++) acc = acc > tmp[j] ? acc : tmp[j];
    for (; i < n; i++) acc = acc > x[i] ? acc : x[i];
    return acc;
}
// PSHUFB permute over a 16-entry table — the x86 analog of NEON TBL. out[i]=table16[idx[i]]
// for idx<16, else 0 (PSHUFB zeroes lanes whose index has the high bit set).
extern "C" X86_TGT_AVX2 void lfm_permute_u8(const uint8_t *table16, const uint8_t *idx,
                                            uint8_t *out, int n) {
    __m128i t = _mm_loadu_si128((const __m128i *)table16);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i id = _mm_loadu_si128((const __m128i *)(idx + i));
        // PSHUFB uses bits[3:0] for the index and zeroes the lane if bit7 set. Map idx>=16 -> 0x80.
        __m128i ge16 = _mm_cmpgt_epi8(id, _mm_set1_epi8(15));
        __m128i sel = _mm_or_si128(id, _mm_and_si128(ge16, _mm_set1_epi8((char)0x80)));
        _mm_storeu_si128((__m128i *)(out + i), _mm_shuffle_epi8(t, sel));
    }
    for (; i < n; i++) out[i] = idx[i] < 16 ? table16[idx[i]] : 0;
}

// =====================================================================================
// Group C — depthwise causal conv1d, bf16 storage / f32 accumulate / bf16 store
// out[b,d,t] = bias[d] + Σ_j w[d,j]·u[b,d, t-(K-1)+j]  (out-of-range taps contribute 0).
// =====================================================================================
extern "C" X86_TGT_AVX2 void lfm_depthwise_causal_conv1d_bf16(
    const uint16_t *u, const uint16_t *w, const uint16_t *bias, uint16_t *out,
    int Bn, int D, int L, int K, int Lout) {
    for (int b = 0; b < Bn; b++)
        for (int d = 0; d < D; d++) {
            const uint16_t *urow = u + ((size_t)b * D + d) * L;
            uint16_t *orow = out + ((size_t)b * D + d) * Lout;
            float biasf = bf16_to_f32(bias[d]);
            const int lo = K - 1, hi = Lout < L ? Lout : L;
            int t = 0;
            for (; t < lo && t < Lout; t++) {
                float acc = biasf;
                for (int j = 0; j < K; j++) {
                    int idx = t - (K - 1) + j;
                    if (idx >= 0 && idx < L) acc += bf16_to_f32(urow[idx]) * bf16_to_f32(w[d * K + j]);
                }
                orow[t] = f32_to_bf16_bits(acc);
            }
            for (; t + 8 <= hi; t += 8) { // 8 outputs at a time, all taps in-bounds
                __m256 acc = _mm256_set1_ps(biasf);
                for (int j = 0; j < K; j++) {
                    int idx = t - (K - 1) + j;
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(bf16_to_f32(w[d * K + j])),
                                          upconv8(urow + idx), acc);
                }
                float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                for (int c = 0; c < 8; c++) orow[t + c] = f32_to_bf16_bits(tmp[c]);
            }
            for (; t < Lout; t++) {
                float acc = biasf;
                for (int j = 0; j < K; j++) {
                    int idx = t - (K - 1) + j;
                    if (idx >= 0 && idx < L) acc += bf16_to_f32(urow[idx]) * bf16_to_f32(w[d * K + j]);
                }
                orow[t] = f32_to_bf16_bits(acc);
            }
        }
}

// =====================================================================================
// Group D — complex radix-2 FFT (scalar-structured; AVX not needed for correctness parity
// with the NEON FCMLA kernel). In-place interleaved [re,im] f32, n a power of two.
// =====================================================================================
extern "C" void lfm_fft_radix2_f32(float *data, int n, int inverse) {
    if (n <= 1) return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2 * i], ti = data[2 * i + 1];
            data[2 * i] = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j] = tr;
            data[2 * j + 1] = ti;
        }
    }
    const float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.0f * (float)M_PI / (float)len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                float wr = cosf(ang * k), wi = sinf(ang * k);
                int a = i + k, b = i + k + len / 2;
                float xr = data[2 * b], xi = data[2 * b + 1];
                float tr = wr * xr - wi * xi, ti = wr * xi + wi * xr;
                float ur = data[2 * a], ui = data[2 * a + 1];
                data[2 * a] = ur + tr;
                data[2 * a + 1] = ui + ti;
                data[2 * b] = ur - tr;
                data[2 * b + 1] = ui - ti;
            }
    }
    if (inverse) {
        float inv = 1.0f / (float)n;
        for (int i = 0; i < 2 * n; i++) data[i] *= inv;
    }
}

// =====================================================================================
// Group E — double-double (FMA error-free transforms), vectorized over __m256.
// =====================================================================================
namespace {
struct ddv {
    __m256 hi, lo;
};
X86_TGT_AVX2 static inline ddv two_sum(__m256 a, __m256 b) {
    __m256 s = _mm256_add_ps(a, b);
    __m256 v = _mm256_sub_ps(s, a);
    __m256 e = _mm256_add_ps(_mm256_sub_ps(a, _mm256_sub_ps(s, v)), _mm256_sub_ps(b, v));
    return {s, e};
}
X86_TGT_AVX2 static inline ddv two_prod(__m256 a, __m256 b) {
    __m256 p = _mm256_mul_ps(a, b);
    __m256 e = _mm256_fmsub_ps(a, b, p); // a*b - p, exact
    return {p, e};
}
X86_TGT_AVX2 static inline ddv dd_add(ddv a, ddv b) {
    ddv s = two_sum(a.hi, b.hi);
    s.lo = _mm256_add_ps(s.lo, _mm256_add_ps(a.lo, b.lo));
    __m256 hi = _mm256_add_ps(s.hi, s.lo);
    __m256 lo = _mm256_sub_ps(s.lo, _mm256_sub_ps(hi, s.hi));
    return {hi, lo};
}
X86_TGT_AVX2 static inline float dd_hreduce(ddv acc) {
    float hi[8], lo[8];
    _mm256_storeu_ps(hi, acc.hi);
    _mm256_storeu_ps(lo, acc.lo);
    float shi = 0.0f, slo = 0.0f; // serial dd sum across the 8 lanes (deterministic)
    for (int i = 0; i < 8; i++) {
        float s = shi + hi[i], v = s - shi;
        float e = (shi - (s - v)) + (hi[i] - v);
        shi = s;
        slo += e + lo[i];
        float t = shi + slo;
        slo = slo - (t - shi);
        shi = t;
    }
    return shi + slo;
}
} // namespace
extern "C" X86_TGT_AVX2 float lfm_dd_sum_f32(const float *x, int n) {
    ddv acc = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = dd_add(acc, {_mm256_loadu_ps(x + i), _mm256_setzero_ps()});
    float r = dd_hreduce(acc);
    for (; i < n; i++) r += x[i];
    return r;
}
extern "C" X86_TGT_AVX2 float lfm_dd_dot_f32(const float *a, const float *b, int n) {
    ddv acc = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = dd_add(acc, two_prod(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    float r = dd_hreduce(acc);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}

// =====================================================================================
// Group F — GPU-style fast-math (RCPPS/RSQRTPS estimate + Newton refinement).
// =====================================================================================
extern "C" X86_TGT_AVX2 void lfm_recip_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 r = _mm256_rcp_ps(v);                        // ~12-bit estimate
        r = _mm256_mul_ps(r, _mm256_fnmadd_ps(v, r, _mm256_set1_ps(2.0f))); // Newton
        r = _mm256_mul_ps(r, _mm256_fnmadd_ps(v, r, _mm256_set1_ps(2.0f)));
        _mm256_storeu_ps(out + i, r);
    }
    for (; i < n; i++) out[i] = 1.0f / x[i];
}
extern "C" X86_TGT_AVX2 void lfm_rsqrt_f32(const float *x, float *out, int n) {
    int i = 0;
    const __m256 half = _mm256_set1_ps(0.5f), three = _mm256_set1_ps(3.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 r = _mm256_rsqrt_ps(v);                      // ~12-bit estimate
        // Newton: r = r*0.5*(3 - v*r*r)
        r = _mm256_mul_ps(_mm256_mul_ps(half, r),
                          _mm256_fnmadd_ps(_mm256_mul_ps(v, r), r, three));
        r = _mm256_mul_ps(_mm256_mul_ps(half, r),
                          _mm256_fnmadd_ps(_mm256_mul_ps(v, r), r, three));
        _mm256_storeu_ps(out + i, r);
    }
    for (; i < n; i++) out[i] = 1.0f / sqrtf(x[i]);
}
