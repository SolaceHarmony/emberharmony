// x86-64 flashkern — the Intel/AMD sibling of native/kernels/aarch64/flashkern_neon.cpp. Same public `extern "C"` API
// and the same GPU-idiom → SIMD-opcode mapping, but expressed in SSE/AVX2/AVX-512 instead of
// NEON. build.rs compiles exactly one of the two per target arch, so the Rust FFI is arch-
// agnostic. Each function runtime-dispatches on CPUID/XCR0 and is
// confined behind a per-function `target(...)` attribute so no gated opcode leaks into an
// ungated function.
//
//   ARM (flashkern_neon)                         x86 (this file)
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
#include <cpuid.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <cmath>

// Every opcode-bearing function is confined to its own ISA via a per-function `target(...)`
// attribute, so nothing above AVX2 ever leaks into an ungated function (notably the AVX2
// bf16 fallback microkernel). This holds for BOTH gcc and clang on x86: clang honours
// per-function target attributes and declares the immintrin intrinsics unconditionally — so,
// unlike the aarch64 flashkern where clang needs the feature in the base -march, here neither
// compiler needs a raised base ISA (build.rs compiles this TU with NO global AVX-512 flags).
// That is exactly what stops a clang binary from emitting zmm codegen inside gemm_bf16_avx2
// and then SIGILL-ing on an AVX2-only CPU that legitimately passed the AVX2 feature gate.
// MSVC understands none of this (no __attribute__), so it is
// deliberately excluded in build.rs; this #error is the backstop if that gate ever regresses.
#if defined(_MSC_VER) && !defined(__clang__)
#error "flashkern_x86.cpp requires GCC/Clang target attributes; build.rs must not compile it with MSVC"
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

static inline uint64_t xgetbv0() {
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

// Do not use __builtin_cpu_supports here: Apple clang lowers its AVX-512 feature
// query to GCC's ___cpu_features2 runtime symbol, which Darwin does not provide.
// Check both hardware support and the OS-owned extended register state directly.
static bool cpu_has_avx512_bf16() {
    static const bool available = [] {
        if (__get_cpuid_max(0, nullptr) < 7) return false;

        uint32_t eax, ebx, ecx, edx;
        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        constexpr uint32_t osxsave = 1u << 27;
        constexpr uint32_t avx = 1u << 28;
        if ((ecx & (osxsave | avx)) != (osxsave | avx)) return false;

        // XMM, YMM, opmask, ZMM_hi256, and hi16_ZMM must all be OS-managed.
        constexpr uint64_t zmm_state = 0xe6;
        if ((xgetbv0() & zmm_state) != zmm_state) return false;

        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        const uint32_t max_subleaf = eax;
        constexpr uint32_t avx512f = 1u << 16;
        constexpr uint32_t avx512bw = 1u << 30;
        constexpr uint32_t avx512vl = 1u << 31;
        if ((ebx & (avx512f | avx512bw | avx512vl)) !=
            (avx512f | avx512bw | avx512vl) || max_subleaf < 1) {
            return false;
        }

        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        constexpr uint32_t avx512bf16 = 1u << 5;
        return (eax & avx512bf16) != 0;
    }();
    return available;
}

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
    if (cpu_has_avx512_bf16())
        gemm_bf16_avx512(A, B, C, M, N, K);
    else
        gemm_bf16_avx2(A, B, C, M, N, K);
}

// GEMV (M==1) — row-streaming "axpy" form, NOT the GEMM: the GEMM packs B per call, which
// at M==1 (every decode-step matmul) is a full K×N repack per token — the repack costs ~100×
// the dot products (the NEON side measured 0.6 GB/s effective before this form). Here each
// contiguous weight row is upconverted and FMA'd into the f32 accumulator with the broadcast
// scalar A[k]; B is read once, contiguously, no staging.
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
static void gemm_nt_impl(const uint16_t *A, const uint16_t *W, float *C, int M, int N, int K) {
    for (int n = 0; n < N; n++) {
        const uint16_t *wr = W + (size_t)n * K;
        for (int m = 0; m < M; m++) {
            const uint16_t *ar = A + (size_t)m * K;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K; k += 8)
                acc = _mm256_fmadd_ps(upconv8(ar + k), upconv8(wr + k), acc);
            float s = hsum256(acc);
            for (; k < K; k++) s = fmaf(bf16_to_f32(ar[k]), bf16_to_f32(wr[k]), s);
            C[(size_t)m * N + n] = s;
        }
    }
}
} // namespace

extern "C" void lfm_bf16_gemm_nt_f32(const uint16_t *A, const uint16_t *W, float *C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    gemm_nt_impl(A, W, C, M, N, K);
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
                                                    const uint16_t *w, uint16_t *out,
                                                    int Bn, int D, int T, int K) {
    float wf[16];
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < D; c++) {
            for (int j = 0; j < K; j++) wf[j] = bf16_to_f32(w[(size_t)c * K + j]);
            const uint16_t *brow = bcx + (((size_t)b * 3 + 0) * D + c) * T;
            const uint16_t *crow = bcx + (((size_t)b * 3 + 1) * D + c) * T;
            const uint16_t *xrow = bcx + (((size_t)b * 3 + 2) * D + c) * T;
            update_row_bf16(brow, crow, xrow, state + ((size_t)b * D + c) * (K - 1), wf,
                            out + ((size_t)b * D + c) * (T + K - 1), T, K);
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

extern "C" X86_TGT_AVX2 void lfm_bf16_rmsnorm(const uint16_t *x, const uint16_t *w,
                                              uint16_t *out, int n, float inv_rms) {
    const __m256 rs = _mm256_set1_ps(inv_rms);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_mul_ps(upconv8(x + i), rs), upconv8(w + i));
        _mm_storeu_si128((__m128i *)(out + i), bf16_bits_x8(v));
    }
    for (; i < n; i++)
        out[i] = f32_to_bf16_bits(bf16_to_f32(x[i]) * inv_rms * bf16_to_f32(w[i]));
}

extern "C" X86_TGT_AVX2 void lfm_bf16_add(const uint16_t *a, const uint16_t *b,
                                          uint16_t *out, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_add_ps(upconv8(a + i), upconv8(b + i));
        _mm_storeu_si128((__m128i *)(out + i), bf16_bits_x8(v));
    }
    for (; i < n; i++) out[i] = f32_to_bf16_bits(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

extern "C" void lfm_swiglu_bf16(const float *g, const float *u, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        float gv = bf16_to_f32(f32_to_bf16_bits(g[i]));
        float sg = bf16_to_f32(f32_to_bf16_bits(gv / (1.0f + expf(-gv))));
        float uv = bf16_to_f32(f32_to_bf16_bits(u[i]));
        out[i] = f32_to_bf16_bits(sg * uv);
    }
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
extern "C" X86_TGT_AVX2 float lfm_bf16_sumsq_candle_f32(const uint16_t *x, int n) {
    const int np = n & ~31;
    __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps();
    for (int i = 0; i < np; i += 32) {
        __m256 x0 = upconv8(x + i);
        __m256 x1 = upconv8(x + i + 8);
        __m256 x2 = upconv8(x + i + 16);
        __m256 x3 = upconv8(x + i + 24);
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
        float v = bf16_to_f32(x[i]);
        acc = acc + v * v;
    }
    return acc;
}
