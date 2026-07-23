// x86-64 flashkern — the Intel/AMD sibling of
// native/kernels/aarch64/flashkern_neon.cpp. The mounted LFM2.5 path is one
// AVX2/FMA implementation. Engine readiness establishes that contract once;
// numerical calls never dispatch to a second implementation.
//
//   ARM (flashkern_neon)                         x86 (this file)
//   TBL/TBX (vqtbl1q_u8)               ->  PSHUFB (_mm256_shuffle_epi8)
//   ADDV/FADDP (vaddvq_f32)            ->  AVX horizontal reduce
//   FRECPE/FRSQRTE (+Newton)           ->  RCPPS/RSQRTPS (_mm256_rcp_ps/_mm256_rsqrt_ps)+Newton
//   FCMLA complex butterfly            ->  AVX f32 mul/add (no native complex here)
//   double_double two_prod/two_sum     ->  FMA error-free transforms (_mm256_fmadd/ fmsub)
//   SMMLA (vmmlaq_s32)                 ->  VPMADDWD (_mm512_madd_epi16), AVX512-BW
//
// BF16 GEMM widens checkpoint halfwords only in registers and accumulates in
// F32 through the sole AVX2/FMA leaf.

#include "flashkern_gemm.h"

#include <immintrin.h>
#include <cpuid.h>
#include <climits>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <cmath>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// Opcode-bearing functions carry explicit target attributes so setup and
// readiness code stays on the base ISA. This holds for both GCC and Clang.
// MSVC understands none of this (no __attribute__), so it is
// deliberately excluded in build.rs; this #error is the backstop if that gate ever regresses.
#if defined(_MSC_VER) && !defined(__clang__)
#error "flashkern_x86.cpp requires GCC/Clang target attributes; build.rs must not compile it with MSVC"
#endif
#define X86_TGT_AVX2 __attribute__((target("avx2,fma")))
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

static inline uint64_t xgetbv0() {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

// Do not use __builtin_cpu_supports here: Apple clang lowers its AVX-512 feature
// query to GCC's ___cpu_features2 runtime symbol, which Darwin does not provide.
// Check both hardware support and the OS-owned extended register state directly.
static bool cpu_has_avx2_fma() {
    static const bool available = [] {
        if (__get_cpuid_max(0, nullptr) < 7) return false;

        uint32_t eax, ebx, ecx, edx;
        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        constexpr uint32_t fma = 1u << 12;
        constexpr uint32_t osxsave = 1u << 27;
        constexpr uint32_t avx = 1u << 28;
        if ((ecx & (fma | osxsave | avx)) != (fma | osxsave | avx)) return false;

        // XMM and YMM state must both be owned by the host OS before an AVX
        // opcode can execute. Rosetta currently fails this gate by design.
        if ((xgetbv0() & 0x6) != 0x6) return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        constexpr uint32_t avx2 = 1u << 5;
        return (ebx & avx2) != 0;
    }();
    return available;
}

static bool rosetta_translates_avx2() {
#if defined(__APPLE__)
    int translated = 0;
    size_t size = sizeof(translated);
    return sysctlbyname("sysctl.proc_translated", &translated, &size,
                        nullptr, 0) == 0 &&
           translated == 1;
#else
    return false;
#endif
}

// --- AVX2 baseline: upconvert bf16->f32 + FMA, 8 output columns per row. ---
X86_TGT_AVX2
static inline __m256 upconv8(const uint16_t *p) { // 8 bf16 -> 8 f32
    __m128i u16 = _mm_loadu_si128((const __m128i *)p);
    __m256i u32 = _mm256_cvtepu16_epi32(u16);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16));
}

X86_TGT_AVX2
static inline __m256 upconv8_bytes(const unsigned char *p) {
    __m128i u16 = _mm_loadu_si128((const __m128i *)p);
    __m256i u32 = _mm256_cvtepu16_epi32(u16);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16));
}

static inline uint16_t load_bf16_word(const unsigned char *bytes) {
    uint16_t word;
    memcpy(&word, bytes, sizeof(word));
    return word;
}
X86_TGT_AVX2
static void gemm_bf16_avx2(const uint16_t *A, const uint16_t *B, float *C,
                           int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        float *row = C + (size_t)m * N;
        memset(row, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < K; ++k) {
            const __m256 a =
                _mm256_set1_ps(bf16_to_f32(A[(size_t)m * K + k]));
            const uint16_t *weights = B + (size_t)k * N;
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                const __m256 acc = _mm256_loadu_ps(row + n);
                _mm256_storeu_ps(row + n,
                                 _mm256_fmadd_ps(a, upconv8(weights + n), acc));
            }
            for (; n < N; ++n) {
                row[n] = fmaf(bf16_to_f32(A[(size_t)m * K + k]),
                              bf16_to_f32(weights[n]), row[n]);
            }
        }
    }
}
} // namespace

extern "C" void lfm_bf16_gemm_f32(const uint16_t *A, const uint16_t *B, float *C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    gemm_bf16_avx2(A, B, C, M, N, K);
}

// GEMV (M==1) — row-streaming "axpy" form. The former GEMM packed B per call,
// making decode pay a full K×N repack per token; both entry points now stream
// each contiguous B row directly, widening only the current SIMD registers.
namespace {
X86_TGT_AVX2
static void gemv_axpy(const uint16_t *A, const uint16_t *B, float *C, int N, int K) {
    memset(C, 0, (size_t)N * sizeof(float));
    int k = 0;
    for (; k + 2 <= K; k += 2) {
        const __m256 a0 = _mm256_set1_ps(bf16_to_f32(A[k]));
        const __m256 a1 = _mm256_set1_ps(bf16_to_f32(A[k + 1]));
        const uint16_t *r0 = B + (size_t)k * N;
        const uint16_t *r1 = r0 + N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 c = _mm256_loadu_ps(C + n);
            c = _mm256_fmadd_ps(a0, upconv8(r0 + n), c);
            c = _mm256_fmadd_ps(a1, upconv8(r1 + n), c);
            _mm256_storeu_ps(C + n, c);
        }
        for (; n < N; n++) { // same per-column op order as the vector body: k, then k+1
            float t = fmaf(bf16_to_f32(A[k]), bf16_to_f32(r0[n]), C[n]);
            C[n] = fmaf(bf16_to_f32(A[k + 1]), bf16_to_f32(r1[n]), t);
        }
    }
    if (k < K) {
        const __m256 a0 = _mm256_set1_ps(bf16_to_f32(A[k]));
        const uint16_t *r0 = B + (size_t)k * N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 c = _mm256_loadu_ps(C + n);
            c = _mm256_fmadd_ps(a0, upconv8(r0 + n), c);
            _mm256_storeu_ps(C + n, c);
        }
        for (; n < N; n++) C[n] = fmaf(bf16_to_f32(A[k]), bf16_to_f32(r0[n]), C[n]);
    }
}
} // namespace

extern "C" void lfm_bf16_gemv_f32(const uint16_t *A, const uint16_t *B, float *C, int N, int K) {
    if (N <= 0 || K <= 0) return;
    gemv_axpy(A, B, C, N, K);
}

// Forward decl at file scope (matching hsum256's definition scope): it is defined with the
// reductions (Group B) further down, but the decode GEMM's tail reduction below is its first
// use. Declaring it inside the anonymous namespace would name a different symbol and fail to
// link, so it must sit at file scope like the definition.
X86_TGT_AVX2 static inline float hsum256(__m256 v);

// Native-layout small-M matmul: C(M,N) = A(M,K) · W(N,K)ᵀ with W in its checkpoint
// row-major layout — each output dots a CONTIGUOUS W row; no transpose anywhere (see the
// NEON twin for the decode-path rationale). W rows stream once, reused across the M rows.
namespace {
X86_TGT_AVX2
static void gemm_nt_impl(const uint16_t *A, const void *W, float *C,
                         int M, int N, int K, int ldc) {
    const unsigned char *weight_bytes = static_cast<const unsigned char *>(W);
    for (int n = 0; n < N; n++) {
        const unsigned char *wr = weight_bytes + (size_t)n * K * sizeof(uint16_t);
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int rows = M - m0 < 4 ? M - m0 : 4;
            __m256 acc[4];
            for (int row = 0; row < rows; ++row)
                acc[row] = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                const __m256 weights =
                    upconv8_bytes(wr + (size_t)k * sizeof(uint16_t));
                for (int row = 0; row < rows; ++row) {
                    const uint16_t *ar = A + (size_t)(m0 + row) * K;
                    acc[row] = _mm256_fmadd_ps(upconv8(ar + k), weights,
                                               acc[row]);
                }
            }
            float sums[4];
            for (int row = 0; row < rows; ++row) sums[row] = hsum256(acc[row]);
            for (; k < K; ++k) {
                const float weight = bf16_to_f32(load_bf16_word(
                    wr + (size_t)k * sizeof(uint16_t)));
                for (int row = 0; row < rows; ++row) {
                    const uint16_t *ar = A + (size_t)(m0 + row) * K;
                    sums[row] = fmaf(bf16_to_f32(ar[k]), weight, sums[row]);
                }
            }
            for (int row = 0; row < rows; ++row)
                C[(size_t)(m0 + row) * ldc + n] = sums[row];
        }
    }
}
} // namespace

extern "C" void lfm_bf16_gemm_nt_f32(const uint16_t *A, const void *W, float *C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    gemm_nt_impl(A, W, C, M, N, K, N);
}

extern "C" void lfm_bf16_gemm_nt_strided_f32(const uint16_t *A, const void *W,
                                               float *C, int M, int N, int K,
                                               int ldc) {
    if (M <= 0 || N <= 0 || K <= 0 || ldc < N) return;
    gemm_nt_impl(A, W, C, M, N, K, ldc);
}

X86_TGT_AVX2
static void gemm_nt_bias_bf16_avx2(
    const uint16_t *A, const unsigned char *weights,
    const unsigned char *bias, uint16_t *output, int M, int N, int K,
    int output_stride) {
    for (int n = 0; n < N; ++n) {
        const unsigned char *wr =
            weights + (size_t)n * K * sizeof(uint16_t);
        for (int m0 = 0; m0 < M; m0 += 4) {
            const int rows = M - m0 < 4 ? M - m0 : 4;
            __m256 acc[4];
            for (int row = 0; row < rows; ++row)
                acc[row] = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                const __m256 weight =
                    upconv8_bytes(wr + (size_t)k * sizeof(uint16_t));
                for (int row = 0; row < rows; ++row)
                    acc[row] = _mm256_fmadd_ps(
                        upconv8(A + (size_t)(m0 + row) * K + k), weight,
                        acc[row]);
            }
            float sums[4];
            for (int row = 0; row < rows; ++row)
                sums[row] = hsum256(acc[row]);
            for (; k < K; ++k) {
                const float weight = bf16_to_f32(load_bf16_word(
                    wr + (size_t)k * sizeof(uint16_t)));
                for (int row = 0; row < rows; ++row)
                    sums[row] = fmaf(
                        bf16_to_f32(A[(size_t)(m0 + row) * K + k]),
                        weight, sums[row]);
            }
            const float offset = bias
                ? bf16_to_f32(load_bf16_word(
                      bias + (size_t)n * sizeof(uint16_t)))
                : 0.0f;
            for (int row = 0; row < rows; ++row)
                output[(size_t)(m0 + row) * output_stride + n] =
                    f32_to_bf16_bits(bias ? sums[row] + offset : sums[row]);
        }
    }
}

extern "C" void lfm_bf16_gemm_nt_bias_bf16(
    const uint16_t *A, const void *W, const void *bias_storage,
    uint16_t *output, int M, int N, int K, int output_stride) {
    if (!A || !W || !output || M <= 0 || N <= 0 || K <= 0 ||
        output_stride < N)
        return;
    const auto *weights = static_cast<const unsigned char *>(W);
    const auto *bias = static_cast<const unsigned char *>(bias_storage);
    gemm_nt_bias_bf16_avx2(A, weights, bias, output, M, N, K,
                           output_stride);
}

extern "C" void lfm_bf16_gemv_rne_add_bf16(
    const void *input_storage, const void *weight_storage,
    const void *residual_storage, uint16_t *output, size_t rows,
    size_t depth) {
    if (!input_storage || !weight_storage || !residual_storage || !output ||
        rows == 0 || depth == 0 || rows > static_cast<size_t>(INT_MAX) ||
        depth > static_cast<size_t>(INT_MAX))
        return;
    const auto *input = static_cast<const uint16_t *>(input_storage);
    const auto *weights = static_cast<const unsigned char *>(weight_storage);
    const auto *residual = static_cast<const unsigned char *>(residual_storage);
    for (size_t row = 0; row < rows; ++row) {
        const void *weight = weights + row * depth * sizeof(uint16_t);
        float dot = 0.0f;
        gemm_nt_impl(input, weight, &dot, 1, 1,
                     static_cast<int>(depth), 1);
        const uint16_t projected = f32_to_bf16_bits(dot);
        const float sum = bf16_to_f32(projected) +
                          bf16_to_f32(load_bf16_word(
                              residual + row * sizeof(uint16_t)));
        output[row] = f32_to_bf16_bits(sum);
    }
}

extern "C" void lfm_bf16_gemv_rne_bf16(
    const void *input_storage, const void *weight_storage, uint16_t *output,
    size_t rows, size_t depth) {
    if (!input_storage || !weight_storage || !output || rows == 0 ||
        depth == 0 || rows > static_cast<size_t>(INT_MAX) ||
        depth > static_cast<size_t>(INT_MAX))
        return;
    const auto *input = static_cast<const uint16_t *>(input_storage);
    const auto *weights = static_cast<const unsigned char *>(weight_storage);
    for (size_t row = 0; row < rows; ++row) {
        const void *weight = weights + row * depth * sizeof(uint16_t);
        float dot = 0.0f;
        gemm_nt_impl(input, weight, &dot, 1, 1,
                     static_cast<int>(depth), 1);
        output[row] = f32_to_bf16_bits(dot);
    }
}

extern "C" int lfm_bf16_gemm_available(void) {
    /* Rosetta executes the AVX2/FMA opcodes used by this one x86 path even
     * though its virtual CPUID/XCR0 contract does not advertise host-owned
     * YMM state. This is readiness validation only; it never selects another
     * numerical implementation. */
    return cpu_has_avx2_fma() || rosetta_translates_avx2() ? 1 : 0;
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

namespace {
X86_TGT_AVX2
static void depthwise_stream_copy(const uint16_t *src, uint16_t *dst, int count) {
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i values = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(dst + i), values);
    }
    volatile uint16_t *tail = dst;
    for (; i < count; ++i) tail[i] = src[i];
}

X86_TGT_AVX2
static void depthwise_stream_zero(uint16_t *dst, int count) {
    volatile uint16_t *tail = dst;
    for (int i = 0; i < count; ++i) tail[i] = 0;
}

X86_TGT_AVX2
static void depthwise_stream_state(const uint16_t *xrow, const uint16_t *crow,
                                   uint16_t *next_row, int T, int P) {
    if (P == 0) return;
    if (T >= P) {
        depthwise_stream_copy(xrow + T - P, next_row, P);
        return;
    }
    const int retained = P - T;
    if (crow)
        depthwise_stream_copy(crow + T, next_row, retained);
    else
        depthwise_stream_zero(next_row, retained);
    depthwise_stream_copy(xrow, next_row + retained, T);
}

} // namespace

extern "C" int lfm_depthwise_stream_bf16_available(void) {
    return lfm_bf16_gemm_available();
}

// CPU translation of the streaming depthwise grid. The x86 contract requires
// AVX2/FMA; Rosetta does not advertise those opcodes and must skip this leaf.
extern "C" X86_TGT_AVX2 void lfm_depthwise_stream_bf16(
    const uint16_t *x, const uint16_t *cache, const uint16_t *weights,
    uint16_t *out, uint16_t *next, int Bn, int D, int T, int K) {
    const int P = K - 1;
    const int rows = Bn * D;
    for (int row = 0; row < rows; ++row) {
        const int channel = row % D;
        const uint16_t *xrow = x + (size_t)row * T;
        const uint16_t *crow = cache ? cache + (size_t)row * P : nullptr;
        const uint16_t *wrow = weights + (size_t)channel * K;
        uint16_t *orow = out + (size_t)row * T;
        int t = 0;

        for (; t < T && t < P; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                const int source = t + j;
                const float value = source < P
                                        ? (crow ? bf16_to_f32(crow[source]) : 0.0f)
                                        : bf16_to_f32(xrow[source - P]);
                acc = std::fma(value, bf16_to_f32(wrow[j]), acc);
            }
            orow[t] = f32_to_bf16_bits(acc);
        }
        for (; t + 8 <= T; t += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int j = 0; j < K; ++j) {
                const int source = t - P + j;
                acc = _mm256_fmadd_ps(upconv8(xrow + source),
                                      _mm256_set1_ps(bf16_to_f32(wrow[j])), acc);
            }
            float values[8];
            _mm256_storeu_ps(values, acc);
            for (int i = 0; i < 8; ++i)
                orow[t + i] = f32_to_bf16_bits(values[i]);
        }
        for (; t < T; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                acc = std::fma(bf16_to_f32(xrow[t - P + j]),
                               bf16_to_f32(wrow[j]), acc);
            }
            orow[t] = f32_to_bf16_bits(acc);
        }

        depthwise_stream_state(xrow, crow,
                               P ? next + (size_t)row * P : nullptr, T, P);
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
// one scalar double-double accumulation step: fold (hi_i, lo_i) into the running (shi, slo).
X86_TGT_AVX2 static inline void dd_step(float &shi, float &slo, float hi_i, float lo_i) {
    float s = shi + hi_i, v = s - shi;
    float e = (shi - (s - v)) + (hi_i - v);
    shi = s;
    slo += e + lo_i;
    float t = shi + slo;
    slo = slo - (t - shi);
    shi = t;
}
// horizontal double-double reduce of the 8 lanes into a scalar (shi, slo) pair (deterministic
// serial order). Returns the pair rather than a collapsed float so a ragged tail can keep
// accumulating in double-double instead of falling back to lossy plain-f32 adds.
X86_TGT_AVX2 static inline void dd_hreduce2(ddv acc, float &shi, float &slo) {
    float hi[8], lo[8];
    _mm256_storeu_ps(hi, acc.hi);
    _mm256_storeu_ps(lo, acc.lo);
    shi = 0.0f;
    slo = 0.0f;
    for (int i = 0; i < 8; i++) dd_step(shi, slo, hi[i], lo[i]);
}
} // namespace
extern "C" X86_TGT_AVX2 float lfm_dd_sum_f32(const float *x, int n) {
    ddv acc = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = dd_add(acc, {_mm256_loadu_ps(x + i), _mm256_setzero_ps()});
    float shi, slo;
    dd_hreduce2(acc, shi, slo);
    // ragged tail: keep folding into the double-double accumulator. A plain-f32 `r += x[i]`
    // would drop any tail element below r's ULP, defeating the high-accuracy contract.
    for (; i < n; i++) dd_step(shi, slo, x[i], 0.0f);
    return shi + slo;
}
extern "C" X86_TGT_AVX2 float lfm_dd_dot_f32(const float *a, const float *b, int n) {
    ddv acc = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = dd_add(acc, two_prod(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    float shi, slo;
    dd_hreduce2(acc, shi, slo);
    // ragged tail: exact product (two_prod via fmaf) folded into the double-double accumulator.
    for (; i < n; i++) {
        float p = a[i] * b[i];
        float e = fmaf(a[i], b[i], -p); // exact a*b - p
        dd_step(shi, slo, p, 0.0f);
        dd_step(shi, slo, e, 0.0f);
    }
    return shi + slo;
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

// =====================================================================================
// Group G — flat-grid conv kernels (ComplexMul.metal, Depthwise3.metal, conv1d_update.rs).
// One thread per output on the GPU -> a plain SIMD loop here (no threadgroup state).
// =====================================================================================

// Elementwise complex multiply — ComplexMul.metal's FIXED evaluation order, deliberately
// NO FMA. Interleaved layout stays interleaved: b_re/b_im lane-duplicated (MOVELDUP /
// MOVEHDUP), one ADDSUBPS pairs the (−, +) — each lane is one separately-rounded product
// and one separately-rounded add/sub, exactly the kernel's ((ar·br)−(ai·bi), (ar·bi)+(ai·br))
// (the imag add commutes lanes but IEEE addition is commutative, so bits match).
extern "C" X86_TGT_AVX2 void lfm_complex_mul_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) { // 4 complexes = 8 floats per iteration
        __m256 av = _mm256_loadu_ps(a + 2 * i);
        __m256 bv = _mm256_loadu_ps(b + 2 * i);
        __m256 bre = _mm256_moveldup_ps(bv);                     // [br,br,...]
        __m256 bim = _mm256_movehdup_ps(bv);                     // [bi,bi,...]
        __m256 aswap = _mm256_permute_ps(av, 0xB1);              // [ai,ar,...]
        __m256 t1 = _mm256_mul_ps(av, bre);                      // [ar·br, ai·br,...]
        __m256 t2 = _mm256_mul_ps(aswap, bim);                   // [ai·bi, ar·bi,...]
        _mm256_storeu_ps(out + 2 * i, _mm256_addsub_ps(t1, t2)); // [−, +] per pair
    }
    for (; i < n; i++) {
        float ar = a[2 * i], ai = a[2 * i + 1], br = b[2 * i], bi = b[2 * i + 1];
        out[2 * i] = (ar * br) - (ai * bi);
        out[2 * i + 1] = (ar * bi) + (ai * br);
    }
}

// Deterministic 3-tap depthwise conv1d — Depthwise3.metal, both window directions, fixed
// multiply-add order ((x0·w0) + (x1·w1), then + (x2·w2)) and NO FMA, so the SIMD body is
// bit-identical to the scalar edges. x[B,C,L], k[C,3], y[B,C,L].
namespace {
X86_TGT_AVX2
static void dw3_row(const float *x, const float *w, float *y, int L) {
    const __m256 w0 = _mm256_set1_ps(w[0]), w1 = _mm256_set1_ps(w[1]), w2 = _mm256_set1_ps(w[2]);
    int t = 0;
    for (; t + 10 <= L; t += 8) { // outputs t..t+7 read x[t..t+9], all in-bounds
        __m256 acc = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + t), w0),
                                   _mm256_mul_ps(_mm256_loadu_ps(x + t + 1), w1));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(x + t + 2), w2));
        _mm256_storeu_ps(y + t, acc);
    }
    for (; t < L; t++) {
        float x0 = x[t];
        float x1 = (t + 1 < L) ? x[t + 1] : 0.0f;
        float x2 = (t + 2 < L) ? x[t + 2] : 0.0f;
        float acc = (x0 * w[0]) + (x1 * w[1]);
        y[t] = acc + (x2 * w[2]);
    }
}
X86_TGT_AVX2
static void dw3_causal_row(const float *x, const float *w, float *y, int L) {
    const __m256 w0 = _mm256_set1_ps(w[0]), w1 = _mm256_set1_ps(w[1]), w2 = _mm256_set1_ps(w[2]);
    int t = 0;
    for (; t < 2 && t < L; t++) {
        float x0 = (t >= 2) ? x[t - 2] : 0.0f;
        float x1 = (t >= 1) ? x[t - 1] : 0.0f;
        float acc = (x0 * w[0]) + (x1 * w[1]);
        y[t] = acc + (x[t] * w[2]);
    }
    for (; t + 8 <= L; t += 8) { // outputs t..t+7 read x[t-2..t+7], in-bounds for t >= 2
        __m256 acc = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x + t - 2), w0),
                                   _mm256_mul_ps(_mm256_loadu_ps(x + t - 1), w1));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(x + t), w2));
        _mm256_storeu_ps(y + t, acc);
    }
    for (; t < L; t++) {
        float acc = (x[t - 2] * w[0]) + (x[t - 1] * w[1]);
        y[t] = acc + (x[t] * w[2]);
    }
}
} // namespace

extern "C" X86_TGT_AVX2 void lfm_depthwise3_f32(const float *x, const float *k, float *y,
                                                int Bn, int C, int L) {
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < C; c++)
            dw3_row(x + ((size_t)b * C + c) * L, k + (size_t)c * 3,
                    y + ((size_t)b * C + c) * L, L);
}

extern "C" X86_TGT_AVX2 void lfm_depthwise3_causal_f32(const float *x, const float *k, float *y,
                                                       int Bn, int C, int L) {
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < C; c++)
            dw3_causal_row(x + ((size_t)b * C + c) * L, k + (size_t)c * 3,
                           y + ((size_t)b * C + c) * L, L);
}

// Fused LFM2 ShortConv decode-step update — conv1d_update.rs: `(B⊙x) → K-tap causal FIR over
// [state | Bx] → C⊙`, one dispatch, state advanced functionally. FMA is deliberate here
// (contractible IS the trained regime); the strict-order instrument is depthwise3_causal.
// bcx[B,3D,T] rows B|C|x, state[B,D,K-1], w[D,K], out[B,D,T+K-1] = [y | new_state].
namespace {
thread_local std::vector<float> g_bx; // [K-1 + T] extended conv-input window, per worker

X86_TGT_AVX2
static void update_row_f32(const float *brow, const float *crow, const float *xrow,
                           const float *srow, const float *wrow, float *orow, int T, int K) {
    const int km1 = K - 1;
    g_bx.resize((size_t)km1 + T);
    float *bx = g_bx.data();
    for (int j = 0; j < km1; j++) bx[j] = srow[j];
    int t = 0;
    for (; t + 8 <= T; t += 8)
        _mm256_storeu_ps(bx + km1 + t,
                         _mm256_mul_ps(_mm256_loadu_ps(brow + t), _mm256_loadu_ps(xrow + t)));
    for (; t < T; t++) bx[km1 + t] = brow[t] * xrow[t];
    t = 0;
    for (; t + 8 <= T; t += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < K; j++)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(wrow[j]), _mm256_loadu_ps(bx + t + j)));
        _mm256_storeu_ps(orow + t, _mm256_mul_ps(_mm256_loadu_ps(crow + t), acc));
    }
    for (; t < T; t++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) acc = acc + wrow[j] * bx[t + j];
        orow[t] = crow[t] * acc;
    }
    for (int j = 0; j < km1; j++) orow[T + j] = bx[T + j];
}

// bf16 storage regime: Bx rounds through bf16 before entering the window; the conv output
// rounds through bf16 before the C gate (both torch-materialized tensors). RNE via the same
// integer trick as f32_to_bf16_bits, vectorized.
X86_TGT_AVX2 static inline __m256 round_bf16_ps(__m256 v) {
    __m256i u = _mm256_castps_si256(v);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    u = _mm256_add_epi32(u, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff)));
    u = _mm256_and_si256(u, _mm256_set1_epi32((int)0xFFFF0000));
    return _mm256_castsi256_ps(u);
}
X86_TGT_AVX2 static inline __m128i bf16_bits_x8(__m256 v) { // 8 f32 -> 8 bf16 bit patterns
    __m256i u = _mm256_castps_si256(v);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    u = _mm256_add_epi32(u, _mm256_add_epi32(lsb, _mm256_set1_epi32(0x7fff)));
    u = _mm256_srli_epi32(u, 16); // 8×u32, each ≤ 0xFFFF
    __m128i lo = _mm256_castsi256_si128(u), hi = _mm256_extracti128_si256(u, 1);
    return _mm_packus_epi32(lo, hi); // exact: values fit u16
}
static inline float round_bf16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xFFFF0000u;
    memcpy(&f, &u, sizeof(f));
    return f;
}

X86_TGT_AVX2
static void update_row_bf16(const uint16_t *brow, const uint16_t *crow, const uint16_t *xrow,
                            const uint16_t *srow, const float *wf, uint16_t *orow, int T, int K) {
    const int km1 = K - 1;
    g_bx.resize((size_t)km1 + T);
    float *bx = g_bx.data();
    for (int j = 0; j < km1; j++) bx[j] = bf16_to_f32(srow[j]);
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        __m256 prod = _mm256_mul_ps(upconv8(brow + t), upconv8(xrow + t));
        _mm256_storeu_ps(bx + km1 + t, round_bf16_ps(prod));
    }
    for (; t < T; t++) bx[km1 + t] = round_bf16_scalar(bf16_to_f32(brow[t]) * bf16_to_f32(xrow[t]));
    t = 0;
    for (; t + 8 <= T; t += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < K; j++)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(wf[j]), _mm256_loadu_ps(bx + t + j)));
        acc = round_bf16_ps(acc);
        __m256 y = _mm256_mul_ps(upconv8(crow + t), acc);
        _mm_storeu_si128((__m128i *)(orow + t), bf16_bits_x8(y));
    }
    for (; t < T; t++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) acc = acc + wf[j] * bx[t + j];
        acc = round_bf16_scalar(acc);
        orow[t] = f32_to_bf16_bits(bf16_to_f32(crow[t]) * acc);
    }
    // new_state values are already bf16-rounded, so the store round-trips exactly.
    for (int j = 0; j < km1; j++) orow[T + j] = f32_to_bf16_bits(bx[T + j]);
}
} // namespace

extern "C" X86_TGT_AVX2 void lfm_conv1d_update_f32(const float *bcx, const float *state,
                                                   const float *w, float *out,
                                                   int Bn, int D, int T, int K) {
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < D; c++) {
            const float *brow = bcx + (((size_t)b * 3 + 0) * D + c) * T;
            const float *crow = bcx + (((size_t)b * 3 + 1) * D + c) * T;
            const float *xrow = bcx + (((size_t)b * 3 + 2) * D + c) * T;
            update_row_f32(brow, crow, xrow, state + ((size_t)b * D + c) * (K - 1),
                           w + (size_t)c * K, out + ((size_t)b * D + c) * (T + K - 1), T, K);
        }
}

extern "C" X86_TGT_AVX2 void lfm_conv1d_update_bf16(const uint16_t *bcx, const uint16_t *state,
                                                    const void *weight_storage, uint16_t *out,
                                                    int Bn, int D, int T, int K) {
    const unsigned char *weights = static_cast<const unsigned char *>(weight_storage);
    float wf[16];
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < D; c++) {
            for (int j = 0; j < K; j++) {
                const size_t index = (size_t)c * K + j;
                wf[j] = bf16_to_f32(load_bf16_word(
                    weights + index * sizeof(uint16_t)));
            }
            const uint16_t *brow = bcx + (((size_t)b * 3 + 0) * D + c) * T;
            const uint16_t *crow = bcx + (((size_t)b * 3 + 1) * D + c) * T;
            const uint16_t *xrow = bcx + (((size_t)b * 3 + 2) * D + c) * T;
            update_row_bf16(brow, crow, xrow, state + ((size_t)b * D + c) * (K - 1), wf,
                            out + ((size_t)b * D + c) * (T + K - 1), T, K);
        }
}

extern "C" void lfm_shortconv_update_split_bf16(
    const uint16_t *ball, const uint16_t *call, const uint16_t *xall,
    const uint16_t *state, const void *weight_storage, uint16_t *y,
    uint16_t *next, int channels, int kernel) {
    if (!ball || !call || !xall || !state || !weight_storage || !y || !next ||
        channels <= 0 || kernel <= 1 || kernel > 16)
        return;
    const auto *weights = static_cast<const unsigned char *>(weight_storage);
    for (int channel = 0; channel < channels; ++channel) {
        const int base = channel * (kernel - 1);
        const float bx = bf16_to_f32(f32_to_bf16_bits(
            bf16_to_f32(ball[channel]) * bf16_to_f32(xall[channel])));
        const unsigned char *row =
            weights + (size_t)channel * kernel * sizeof(uint16_t);
        float acc = 0.0f;
        for (int tap = 0; tap + 1 < kernel; ++tap)
            acc = acc + bf16_to_f32(load_bf16_word(
                                row + (size_t)tap * sizeof(uint16_t))) *
                            bf16_to_f32(state[base + tap]);
        acc = acc + bf16_to_f32(load_bf16_word(
                            row + (size_t)(kernel - 1) * sizeof(uint16_t))) *
                        bx;
        acc = bf16_to_f32(f32_to_bf16_bits(acc));
        y[channel] = f32_to_bf16_bits(bf16_to_f32(call[channel]) * acc);
        for (int tap = 0; tap + 2 < kernel; ++tap)
            next[base + tap] = state[base + tap + 1];
        next[base + kernel - 2] = f32_to_bf16_bits(bx);
    }
}

// =====================================================================================
// Group H — decode stage kernels (see the NEON twin for the ladder contracts).
// =====================================================================================

extern "C" X86_TGT_AVX2 float lfm_bf16_sumsq_f32(const uint16_t *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = upconv8(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float s = hsum256(acc);
    for (; i < n; i++) {
        float v = bf16_to_f32(x[i]);
        s = fmaf(v, v, s);
    }
    return s;
}

extern "C" X86_TGT_AVX2 void lfm_bf16_rmsnorm(const void *x_storage,
                                              const void *weight_storage,
                                              uint16_t *out, int n, float inv_rms) {
    const unsigned char *x = static_cast<const unsigned char *>(x_storage);
    const unsigned char *w = static_cast<const unsigned char *>(weight_storage);
    const __m256 rs = _mm256_set1_ps(inv_rms);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(
            _mm256_mul_ps(upconv8_bytes(x + (size_t)i * sizeof(uint16_t)), rs),
            upconv8_bytes(w + (size_t)i * sizeof(uint16_t)));
        _mm_storeu_si128((__m128i *)(out + i), bf16_bits_x8(v));
    }
    for (; i < n; i++)
        out[i] = f32_to_bf16_bits(
            bf16_to_f32(load_bf16_word(x + (size_t)i * sizeof(uint16_t))) * inv_rms *
            bf16_to_f32(load_bf16_word(w + (size_t)i * sizeof(uint16_t))));
}

extern "C" X86_TGT_AVX2 void lfm_bf16_add(const void *a_storage, const void *b_storage,
                                          uint16_t *out, int n) {
    const unsigned char *a = static_cast<const unsigned char *>(a_storage);
    const unsigned char *b = static_cast<const unsigned char *>(b_storage);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_add_ps(
            upconv8_bytes(a + (size_t)i * sizeof(uint16_t)),
            upconv8_bytes(b + (size_t)i * sizeof(uint16_t)));
        _mm_storeu_si128((__m128i *)(out + i), bf16_bits_x8(v));
    }
    for (; i < n; i++) {
        out[i] = f32_to_bf16_bits(
            bf16_to_f32(load_bf16_word(a + (size_t)i * sizeof(uint16_t))) +
            bf16_to_f32(load_bf16_word(b + (size_t)i * sizeof(uint16_t))));
    }
}

extern "C" void lfm_swiglu_bf16(const float *g, const float *u, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        float gv = bf16_to_f32(f32_to_bf16_bits(g[i]));
        float sg = bf16_to_f32(f32_to_bf16_bits(gv / (1.0f + expf(-gv))));
        float uv = bf16_to_f32(f32_to_bf16_bits(u[i]));
        out[i] = f32_to_bf16_bits(sg * uv);
    }
}

X86_TGT_AVX2
static void gemv_pair_swiglu_avx2(
    const unsigned char *input, const unsigned char *gate,
    const unsigned char *up, uint16_t *output, size_t rows, size_t depth) {
    const size_t row_bytes = depth * sizeof(uint16_t);
    for (size_t row = 0; row < rows; row += 2) {
        const size_t count = rows - row < 2 ? rows - row : 2;
        const unsigned char *g0 = gate + row * row_bytes;
        const unsigned char *u0 = up + row * row_bytes;
        const unsigned char *g1 = g0 + row_bytes;
        const unsigned char *u1 = u0 + row_bytes;
        __m256 ga0 = _mm256_setzero_ps(), ua0 = _mm256_setzero_ps();
        __m256 ga1 = _mm256_setzero_ps(), ua1 = _mm256_setzero_ps();
        size_t k = 0;
        for (; k + 8 <= depth; k += 8) {
            const __m256 x = upconv8_bytes(input + k * sizeof(uint16_t));
            ga0 = _mm256_fmadd_ps(
                x, upconv8_bytes(g0 + k * sizeof(uint16_t)), ga0);
            ua0 = _mm256_fmadd_ps(
                x, upconv8_bytes(u0 + k * sizeof(uint16_t)), ua0);
            if (count == 1) continue;
            ga1 = _mm256_fmadd_ps(
                x, upconv8_bytes(g1 + k * sizeof(uint16_t)), ga1);
            ua1 = _mm256_fmadd_ps(
                x, upconv8_bytes(u1 + k * sizeof(uint16_t)), ua1);
        }
        float gates[2] = {hsum256(ga0), hsum256(ga1)};
        float ups[2] = {hsum256(ua0), hsum256(ua1)};
        for (; k < depth; ++k) {
            const float value = bf16_to_f32(load_bf16_word(
                input + k * sizeof(uint16_t)));
            gates[0] = fmaf(value, bf16_to_f32(load_bf16_word(
                                         g0 + k * sizeof(uint16_t))),
                            gates[0]);
            ups[0] = fmaf(value, bf16_to_f32(load_bf16_word(
                                       u0 + k * sizeof(uint16_t))),
                          ups[0]);
            if (count == 1) continue;
            gates[1] = fmaf(value, bf16_to_f32(load_bf16_word(
                                         g1 + k * sizeof(uint16_t))),
                            gates[1]);
            ups[1] = fmaf(value, bf16_to_f32(load_bf16_word(
                                       u1 + k * sizeof(uint16_t))),
                          ups[1]);
        }
        for (size_t lane = 0; lane < count; ++lane) {
            const float g = bf16_to_f32(f32_to_bf16_bits(gates[lane]));
            const float silu = bf16_to_f32(f32_to_bf16_bits(
                g / (1.0f + expf(-g))));
            const float u = bf16_to_f32(f32_to_bf16_bits(ups[lane]));
            output[row + lane] = f32_to_bf16_bits(silu * u);
        }
    }
}

extern "C" void lfm_bf16_gemv_pair_swiglu_bf16(
    const void *input_storage, const void *gate_weight_storage,
    const void *up_weight_storage, uint16_t *output, size_t rows,
    size_t depth) {
    if (!input_storage || !gate_weight_storage || !up_weight_storage ||
        !output || rows == 0 || depth == 0 ||
        rows > static_cast<size_t>(INT_MAX) ||
        depth > static_cast<size_t>(INT_MAX))
        return;
    const auto *input = static_cast<const unsigned char *>(input_storage);
    const auto *gate = static_cast<const unsigned char *>(gate_weight_storage);
    const auto *up = static_cast<const unsigned char *>(up_weight_storage);
    gemv_pair_swiglu_avx2(input, gate, up, output, rows, depth);
}

X86_TGT_AVX2
static void shortconv_project_update_avx2(
    const unsigned char *input, const unsigned char *projection,
    const uint16_t *state, const unsigned char *conv, uint16_t *y,
    uint16_t *next, size_t hidden, size_t channel_begin,
    size_t channel_count, size_t kernel) {
    const size_t row_bytes = hidden * sizeof(uint16_t);
    for (size_t channel = channel_begin;
         channel < channel_begin + channel_count; ++channel) {
        const unsigned char *brow = projection + channel * row_bytes;
        const unsigned char *crow = projection + (hidden + channel) * row_bytes;
        const unsigned char *xrow = projection + (2 * hidden + channel) * row_bytes;
        __m256 ba = _mm256_setzero_ps(), ca = _mm256_setzero_ps();
        __m256 xa = _mm256_setzero_ps();
        size_t k = 0;
        for (; k + 8 <= hidden; k += 8) {
            const __m256 value =
                upconv8_bytes(input + k * sizeof(uint16_t));
            ba = _mm256_fmadd_ps(
                value, upconv8_bytes(brow + k * sizeof(uint16_t)), ba);
            ca = _mm256_fmadd_ps(
                value, upconv8_bytes(crow + k * sizeof(uint16_t)), ca);
            xa = _mm256_fmadd_ps(
                value, upconv8_bytes(xrow + k * sizeof(uint16_t)), xa);
        }
        float b = hsum256(ba), c = hsum256(ca), x = hsum256(xa);
        for (; k < hidden; ++k) {
            const float value = bf16_to_f32(load_bf16_word(
                input + k * sizeof(uint16_t)));
            b = fmaf(value, bf16_to_f32(load_bf16_word(
                                brow + k * sizeof(uint16_t))), b);
            c = fmaf(value, bf16_to_f32(load_bf16_word(
                                crow + k * sizeof(uint16_t))), c);
            x = fmaf(value, bf16_to_f32(load_bf16_word(
                                xrow + k * sizeof(uint16_t))), x);
        }
        b = bf16_to_f32(f32_to_bf16_bits(b));
        c = bf16_to_f32(f32_to_bf16_bits(c));
        x = bf16_to_f32(f32_to_bf16_bits(x));
        const float bx = bf16_to_f32(f32_to_bf16_bits(b * x));
        const size_t base = channel * (kernel - 1);
        const unsigned char *taps =
            conv + channel * kernel * sizeof(uint16_t);
        float acc = 0.0f;
        for (size_t tap = 0; tap + 1 < kernel; ++tap)
            acc = acc + bf16_to_f32(load_bf16_word(
                                taps + tap * sizeof(uint16_t))) *
                            bf16_to_f32(state[base + tap]);
        acc = acc + bf16_to_f32(load_bf16_word(
                            taps + (kernel - 1) * sizeof(uint16_t))) *
                        bx;
        acc = bf16_to_f32(f32_to_bf16_bits(acc));
        y[channel] = f32_to_bf16_bits(c * acc);
        for (size_t tap = 0; tap + 2 < kernel; ++tap)
            next[base + tap] = state[base + tap + 1];
        next[base + kernel - 2] = f32_to_bf16_bits(bx);
    }
}

extern "C" void lfm_shortconv_project_update_bf16(
    const void *input_storage, const void *projection_weight_storage,
    const uint16_t *state, const void *conv_weight_storage, uint16_t *y,
    uint16_t *next, size_t hidden, size_t channel_begin,
    size_t channel_count, size_t kernel) {
    if (!input_storage || !projection_weight_storage || !state ||
        !conv_weight_storage || !y || !next || hidden == 0 ||
        channel_count == 0 || kernel <= 1 || kernel > 16 ||
        channel_begin > hidden || channel_count > hidden - channel_begin)
        return;
    const auto *input = static_cast<const unsigned char *>(input_storage);
    const auto *projection =
        static_cast<const unsigned char *>(projection_weight_storage);
    const auto *conv = static_cast<const unsigned char *>(conv_weight_storage);
    shortconv_project_update_avx2(
        input, projection, state, conv, y, next, hidden, channel_begin,
        channel_count, kernel);
}

extern "C" void lfm_softmax_scaled_f32(float *x, int n, float scale) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
        if (x[i] > mx) mx = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

extern "C" X86_TGT_AVX2 void lfm_attn_av_f32(const float *att, const float *v, float *out,
                                             int len, int hd) {
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < len; t++) {
        const __m256 a = _mm256_set1_ps(att[t]);
        const float *row = v + (size_t)t * hd;
        int i = 0;
        for (; i + 8 <= hd; i += 8)
            _mm256_storeu_ps(out + i,
                             _mm256_fmadd_ps(a, _mm256_loadu_ps(row + i), _mm256_loadu_ps(out + i)));
        for (; i < hd; i++) out[i] = fmaf(att[t], row[i], out[i]);
    }
}

extern "C" X86_TGT_AVX2 void lfm_attn_qk_f32(const float *q, const float *k, float *att,
                                             int len, int hd) {
    for (int t = 0; t < len; t++) {
        const float *row = k + (size_t)t * hd;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hd; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(row + i), acc);
        float s = hsum256(acc);
        for (; i < hd; i++) s = fmaf(q[i], row[i], s);
        att[t] = s;
    }
}

extern "C" void lfm_rope_i_f32(float *x, const float *cos_p, const float *sin_p, int hd) {
    for (int i = 0; i + 1 < hd; i += 2) {
        float c = cos_p[i / 2], s = sin_p[i / 2];
        float x0 = x[i], x1 = x[i + 1];
        x[i] = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

extern "C" X86_TGT_AVX2 void lfm_bf16_to_f32(const uint16_t *x, float *out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, upconv8(x + i));
    for (; i < n; i++) out[i] = bf16_to_f32(x[i]);
}
extern "C" X86_TGT_AVX2 void lfm_f32_to_bf16(const float *x, uint16_t *out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(out + i), bf16_bits_x8(_mm256_loadu_ps(x + i)));
    for (; i < n; i++) out[i] = f32_to_bf16_bits(x[i]);
}

// bf16-plane attention — see the NEON twins. K/V widen in registers via upconv8.
extern "C" X86_TGT_AVX2 void lfm_attn_qk_bf16(const float *q, const uint16_t *k, float *att,
                                              int len, int hd) {
    for (int t = 0; t < len; t++) {
        const uint16_t *row = k + (size_t)t * hd;
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hd; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), upconv8(row + i), acc);
        float s = hsum256(acc);
        for (; i < hd; i++) s = fmaf(q[i], bf16_to_f32(row[i]), s);
        att[t] = s;
    }
}

extern "C" X86_TGT_AVX2 void lfm_attn_av_bf16(const float *att, const uint16_t *v, float *out,
                                              int len, int hd) {
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < len; t++) {
        const __m256 a = _mm256_set1_ps(att[t]);
        const uint16_t *row = v + (size_t)t * hd;
        int i = 0;
        for (; i + 8 <= hd; i += 8)
            _mm256_storeu_ps(out + i,
                             _mm256_fmadd_ps(a, upconv8(row + i), _mm256_loadu_ps(out + i)));
        for (; i < hd; i++) out[i] = fmaf(att[t], bf16_to_f32(row[i]), out[i]);
    }
}

// Sequential-order sumsq — see the NEON twin (token-exact norm reduction, no FMA).
extern "C" float lfm_bf16_sumsq_seq_f32(const uint16_t *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = bf16_to_f32(x[i]);
        float sq = v * v;
        acc = acc + sq;
    }
    return acc;
}

// Sumsq in CANDLE's exact f32 reduction order (cpu/avx.rs vec_sum over a sqr() tensor):
// four __m256 accumulators over 32-element steps, pairwise tree, then candle's exact
// horizontal (low128+high128, hadd, hadd), sequential leftovers. See the NEON twin.
extern "C" X86_TGT_AVX2 float lfm_bf16_sumsq_ordered_f32(const void *storage, int n) {
    const unsigned char *x = static_cast<const unsigned char *>(storage);
    const int np = n & ~31;
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
    for (int i = 0; i < np; i += 32) {
        __m256 x0 = upconv8_bytes(x + (size_t)i * sizeof(uint16_t));
        __m256 x1 = upconv8_bytes(x + (size_t)(i + 8) * sizeof(uint16_t));
        __m256 x2 = upconv8_bytes(x + (size_t)(i + 16) * sizeof(uint16_t));
        __m256 x3 = upconv8_bytes(x + (size_t)(i + 24) * sizeof(uint16_t));
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x0, x0));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x1, x1));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x2, x2));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x3, x3));
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum0), _mm256_extractf128_ps(sum0, 1));
    __m128 t1 = _mm_hadd_ps(t0, t0);
    float acc = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    for (int i = np; i < n; i++) {
        float v = bf16_to_f32(load_bf16_word(x + (size_t)i * sizeof(uint16_t)));
        acc = acc + v * v;
    }
    return acc;
}
