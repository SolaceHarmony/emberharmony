// NEON flashkern — a library of aarch64 SIMD procedures that mirror the GPU idioms of the
// crate's JIT-embedded Metal kernels (crates/candle-flashfftconv),
// mapping each Metal construct to its closest — deliberately obscure — NEON opcode:
//
//   Metal simdgroup_multiply_accumulate (simdgroup_float8x8, fp32 accum)  ->  BFMMLA (vbfmmlaq_f32)
//   Metal skinny GEMV / dot                                               ->  BFDOT  (vbfdotq_f32)
//   Metal simd_shuffle / gather                                           ->  TBL/TBX (vqtbl1q_u8)
//   Metal threadgroup reduce                                              ->  ADDV/FADDP (vaddvq_f32)
//   Metal complex butterfly (cmul)                                        ->  FCMLA (vcmla_f32 + rot90)
//   Metal double_double two_prod/two_sum                                  ->  FMA error-free transforms
//   GPU rcp / rsqrt fast-math                                             ->  FRECPE/FRSQRTE + Newton
//   Metal int tensor-core (stretch)                                       ->  SMMLA (vmmlaq_s32)
//   Metal `bfloat(acc)` RNE store                                         ->  BFCVT (vcvth_bf16_f32)
//   Metal threadgroup shared memory + barrier + grid dispatch             ->  packed panels + PRFM;
//                                                                             rayon tiling (Rust side)
//
// All entry points are `extern "C"` (flat FFI to src/compute/flashkern/neon.rs). C++17 internally for the
// double-double struct. Each feature-specific block is confined to a function carrying a
// per-compiler target attribute so no gated opcode leaks into an ungated function; callers
// runtime-gate on NeonFeatures (src/compute/flashkern/neon.rs). Verified with aarch64-linux-gnu-g++ +
// qemu-aarch64 -cpu max; ships compiled by build.rs on aarch64 (clang on macOS).

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <cmath>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif

// --- Per-function target gating -------------------------------------------------------
// clang exposes the ACLE intrinsics only when the *base* -march enables the feature, so on
// the clang build (build.rs) the base march carries every feature and these macros are
// empty. GCC always declares the intrinsics (target-pragma-wrapped in arm_neon.h) and honours
// a per-function `target("arch=...")`, keeping the base march low and each opcode isolated.
#if defined(__clang__)
#define FK_TGT_BF16
#define FK_TGT_I8MM
#define FK_TGT_FCMA
#else
#define FK_TGT_BF16 __attribute__((target("arch=armv8.2-a+bf16")))
#define FK_TGT_I8MM __attribute__((target("arch=armv8.2-a+i8mm")))
#define FK_TGT_FCMA __attribute__((target("arch=armv8.3-a")))
#endif

// bf16 (upper 16 bits of the f32) -> f32, scalar. No dedicated bf16->f32 instruction exists.
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// f32 -> bf16 *bit pattern* via hardware BFCVT (round-to-nearest-even). Returned as raw
// uint16_t (not the __bf16 type — assigning __bf16 to uint16_t would convert numerically).
FK_TGT_BF16
static inline uint16_t f32_to_bf16_bits(float f) {
    bfloat16_t b = vcvth_bf16_f32(f);
    uint16_t u;
    memcpy(&u, &b, sizeof(u));
    return u;
}

// =====================================================================================
// Group A — GEMM (mirrors fused_monarch.rs simdgroup_float8x8 + simdgroup_multiply_accumulate)
// =====================================================================================
//
// C(M,N) f32 = A(M,K) bf16 · B(K,N) bf16, all row-major, f32 accumulate — torch's CPU
// bf16-matmul numerics. 8×8 output tile == a 4×4 grid of BFMMLA 2×2 sub-tiles => 16
// independent `float32x4_t` accumulators (the ILP the 2×2 reference kernel lacks). A and B
// are packed ONCE into thread-local scratch (reused across calls of matching size, no
// per-call heap alloc after warmup) in BFMMLA tile order, zero-padded to 8×8×(K→4).

namespace {

// Thread-local packed panels (each rayon worker gets its own — no contention).
thread_local std::vector<bfloat16_t> g_Ap;
thread_local std::vector<bfloat16_t> g_Bp;

// Pack A(M,K) row-major -> Ap: per 8-row panel, per 4-deep K block, 4 row-pairs ×
// 8 bf16 = [row(2rp)[k..k+3], row(2rp+1)[k..k+3]]. Padded rows/cols are +0.0 (contribute 0).
static void pack_a(const bfloat16_t *A, int M, int K, int Mp, int Kp, bfloat16_t *Ap) {
    const int kb = Kp / 4;
    memset(Ap, 0, (size_t)Mp * Kp * sizeof(bfloat16_t));
    for (int i = 0; i < M; i++) {
        const int panel = i / 8, rp = (i % 8) / 2, ir = i & 1;
        bfloat16_t *base = Ap + ((size_t)panel * kb) * 32 + (size_t)rp * 8; // 8 bf16 per rowpair
        for (int k = 0; k < K; k++) {
            base[(size_t)(k / 4) * 32 + ir * 4 + (k & 3)] = A[(size_t)i * K + k];
        }
    }
}

// Pack B(K,N) row-major -> Bp: per 8-col panel, per 4-deep K block, 4 col-pairs ×
// 8 bf16 = [col(2cp)[k..k+3], col(2cp+1)[k..k+3]].
static void pack_b(const bfloat16_t *B, int K, int N, int Np, int Kp, bfloat16_t *Bp) {
    const int kb = Kp / 4;
    memset(Bp, 0, (size_t)Np * Kp * sizeof(bfloat16_t));
    for (int j = 0; j < N; j++) {
        const int panel = j / 8, cp = (j % 8) / 2, jc = j & 1;
        bfloat16_t *base = Bp + ((size_t)panel * kb) * 32 + (size_t)cp * 8;
        for (int k = 0; k < K; k++) {
            base[(size_t)(k / 4) * 32 + jc * 4 + (k & 3)] = B[(size_t)k * N + j];
        }
    }
}

FK_TGT_BF16
static void gemm_tiles(const bfloat16_t *Ap, const bfloat16_t *Bp, float *C,
                       int M, int N, int Mp, int Np, int Kp) {
    const int kb = Kp / 4;
    for (int ip = 0; ip < Mp; ip += 8) {
        const bfloat16_t *apanel = Ap + ((size_t)(ip / 8) * kb) * 32;
        for (int jp = 0; jp < Np; jp += 8) {
            const bfloat16_t *bpanel = Bp + ((size_t)(jp / 8) * kb) * 32;
            float32x4_t acc[4][4];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) acc[r][c] = vdupq_n_f32(0.0f);
            const bfloat16_t *ap = apanel, *bp = bpanel;
            for (int b = 0; b < kb; b++) {
                __builtin_prefetch(ap + 32 * 4, 0, 3);
                __builtin_prefetch(bp + 32 * 4, 0, 3);
                bfloat16x8_t av[4], bv[4];
                for (int r = 0; r < 4; r++) av[r] = vld1q_bf16(ap + r * 8);
                for (int c = 0; c < 4; c++) bv[c] = vld1q_bf16(bp + c * 8);
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        acc[r][c] = vbfmmlaq_f32(acc[r][c], av[r], bv[c]);
                ap += 32;
                bp += 32;
            }
            // scatter 2×2 sub-tiles: acc lane order [c00,c01,c10,c11]
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    float out[4];
                    vst1q_f32(out, acc[r][c]);
                    const int r0 = ip + 2 * r, c0 = jp + 2 * c;
                    if (r0 + 0 < M && c0 + 0 < N) C[(size_t)(r0 + 0) * N + c0 + 0] = out[0];
                    if (r0 + 0 < M && c0 + 1 < N) C[(size_t)(r0 + 0) * N + c0 + 1] = out[1];
                    if (r0 + 1 < M && c0 + 0 < N) C[(size_t)(r0 + 1) * N + c0 + 0] = out[2];
                    if (r0 + 1 < M && c0 + 1 < N) C[(size_t)(r0 + 1) * N + c0 + 1] = out[3];
                }
            }
        }
    }
}

} // namespace

extern "C" {

// Full single-threaded 8×8 BFMMLA GEMM. Rust parallelizes by calling this over M-row blocks
// (rayon), which is why packing lives inside (each block packs its own rows + a B copy).
void lfm_bf16_gemm_f32_v2(const uint16_t *A_, const uint16_t *B_, float *C,
                          int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const bfloat16_t *A = (const bfloat16_t *)A_;
    const bfloat16_t *B = (const bfloat16_t *)B_;
    const int Mp = (M + 7) & ~7, Np = (N + 7) & ~7, Kp = (K + 3) & ~3;
    g_Ap.resize((size_t)Mp * Kp);
    g_Bp.resize((size_t)Np * Kp);
    pack_a(A, M, K, Mp, Kp, g_Ap.data());
    pack_b(B, K, N, Np, Kp, g_Bp.data());
    gemm_tiles(g_Ap.data(), g_Bp.data(), C, M, N, Mp, Np, Kp);
}

// GEMV (M==1): C[0..N) = Σ_k A[k]·B[k,N]. BFDOT over K, one column at a time. B is packed
// column-major (bf16) so each column is contiguous; K padded to a multiple of 8.
void lfm_bf16_gemv_f32(const uint16_t *A_, const uint16_t *B_, float *C, int N, int K);

} // extern "C"

// Row-streaming "axpy" GEMV: for each k the CONTIGUOUS weight row B[k,·] is widened
// (bf16 = the top 16 bits of the f32, so widen is a shift — baseline NEON, no FEAT_BF16
// opcode needed) and FMA'd into the f32 accumulator vector C with the broadcast scalar
// A[k]. B is read exactly once, contiguously, with NO per-call transpose or staging copy.
//
// The previous form transposed the whole K×N weight into a thread-local column-major
// buffer on EVERY call to feed BFDOT — a cache-hostile scalar repack ~100× the cost of the
// dot products themselves. M==1 is every decode-step matmul, so that repack was the entire
// CPU decode budget (measured 0.6 GB/s effective; this form is bandwidth-bound).
// Numerics: bf16 products exact in f32; per-column accumulation ascending k (one FMA
// rounding per row) — within the kernel's documented summation-order latitude.
static inline float32x4_t bf16_row_lo(uint16x8_t b) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
}
static inline float32x4_t bf16_row_hi(uint16x8_t b) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(b, 16));
}

static void gemv_impl(const uint16_t *A, const uint16_t *B, float *C, int N, int K) {
    memset(C, 0, (size_t)N * sizeof(float));
    int k = 0;
    for (; k + 2 <= K; k += 2) { // two rows per pass for ILP on the C read-modify-write
        const float a0 = bf16_to_f32(A[k]), a1 = bf16_to_f32(A[k + 1]);
        const uint16_t *r0 = B + (size_t)k * N;
        const uint16_t *r1 = r0 + N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __builtin_prefetch(r1 + n + 256, 0, 0);
            float32x4_t c0 = vld1q_f32(C + n), c1 = vld1q_f32(C + n + 4);
            uint16x8_t b0 = vld1q_u16(r0 + n);
            c0 = vfmaq_n_f32(c0, bf16_row_lo(b0), a0);
            c1 = vfmaq_n_f32(c1, bf16_row_hi(b0), a0);
            uint16x8_t b1 = vld1q_u16(r1 + n);
            c0 = vfmaq_n_f32(c0, bf16_row_lo(b1), a1);
            c1 = vfmaq_n_f32(c1, bf16_row_hi(b1), a1);
            vst1q_f32(C + n, c0);
            vst1q_f32(C + n + 4, c1);
        }
        for (; n < N; n++) { // same per-column op order as the vector body: k, then k+1
            float t = fmaf(a0, bf16_to_f32(r0[n]), C[n]);
            C[n] = fmaf(a1, bf16_to_f32(r1[n]), t);
        }
    }
    if (k < K) { // odd trailing row
        const float a0 = bf16_to_f32(A[k]);
        const uint16_t *r0 = B + (size_t)k * N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            float32x4_t c0 = vld1q_f32(C + n), c1 = vld1q_f32(C + n + 4);
            uint16x8_t b0 = vld1q_u16(r0 + n);
            c0 = vfmaq_n_f32(c0, bf16_row_lo(b0), a0);
            c1 = vfmaq_n_f32(c1, bf16_row_hi(b0), a0);
            vst1q_f32(C + n, c0);
            vst1q_f32(C + n + 4, c1);
        }
        for (; n < N; n++) C[n] = fmaf(a0, bf16_to_f32(r0[n]), C[n]);
    }
}

extern "C" void lfm_bf16_gemv_f32(const uint16_t *A_, const uint16_t *B_, float *C,
                                  int N, int K) {
    if (N <= 0 || K <= 0) return;
    gemv_impl(A_, B_, C, N, K);
}

// Native-layout small-M matmul: C(M,N) f32 = A(M,K) bf16 · W(N,K)ᵀ, with W kept in its
// checkpoint row-major [N,K] layout — each output C[m][n] is a dot over the CONTIGUOUS row
// W[n,·], so NO transpose exists anywhere on this path. (The candle-side alternative was
// `w.t().contiguous()`: a full strided weight copy per linear per call — measured as ~97%
// of CPU decode time.) Intended for decode-side small M (1 per decode step, ≤4 for suffix
// chunks); W rows stream once, reused across the M activation rows. Baseline NEON.
static void gemm_nt_impl(const uint16_t *A, const uint16_t *W, float *C, int M, int N, int K) {
    for (int n = 0; n < N; n++) {
        const uint16_t *wr = W + (size_t)n * K;
        __builtin_prefetch(wr + K, 0, 0);
        for (int m = 0; m < M; m++) {
            const uint16_t *ar = A + (size_t)m * K;
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                uint16x8_t wb = vld1q_u16(wr + k);
                uint16x8_t ab = vld1q_u16(ar + k);
                acc0 = vfmaq_f32(acc0, bf16_row_lo(ab), bf16_row_lo(wb));
                acc1 = vfmaq_f32(acc1, bf16_row_hi(ab), bf16_row_hi(wb));
            }
            float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
            for (; k < K; k++) acc = fmaf(bf16_to_f32(ar[k]), bf16_to_f32(wr[k]), acc);
            C[(size_t)m * N + n] = acc;
        }
    }
}

extern "C" void lfm_bf16_gemm_nt_f32(const uint16_t *A, const uint16_t *W, float *C,
                                     int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    gemm_nt_impl(A, W, C, M, N, K);
}

extern "C" int lfm_bf16_gemm_available(void) {
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname("hw.optional.arm.FEAT_BF16", &value, &size, nullptr, 0) == 0 &&
           value == 1;
#elif defined(__linux__)
    constexpr unsigned long HWCAP2_BF16 = 1ul << 14;
    return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#else
    return 0;
#endif
}

// Stretch: int8 tensor-core MAC — same 8×8 idiom via SMMLA, showing the pattern generalizes
// across dtypes. C(M,N) s32 = A(M,K) s8 · B(K,N) s8. Reference-quality (2×2 SMMLA tile).
extern "C" void lfm_s8_gemm_s32(const int8_t *A, const int8_t *B, int32_t *C,
                                int M, int N, int K);

FK_TGT_I8MM
static void s8_gemm_impl(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K) {
    const int Mp = (M + 1) & ~1, Np = (N + 1) & ~1, Kp = (K + 7) & ~7, kb = Kp / 8;
    static thread_local std::vector<int8_t> Ap, Bp;
    Ap.assign((size_t)Mp * Kp, 0);
    Bp.assign((size_t)Np * Kp, 0);
    // SMMLA: a is 2×8, b is 2×8; result[i][j] = Σ_k a[i][k]·b[j][k]. Pack row-pairs / col-pairs.
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            Ap[((size_t)(i / 2) * kb + k / 8) * 16 + (i & 1) * 8 + (k & 7)] = A[(size_t)i * K + k];
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            Bp[((size_t)(j / 2) * kb + k / 8) * 16 + (j & 1) * 8 + (k & 7)] = B[(size_t)k * N + j];
    for (int it = 0; it < Mp; it += 2) {
        for (int jt = 0; jt < Np; jt += 2) {
            const int8_t *ap = Ap.data() + (size_t)(it / 2) * kb * 16;
            const int8_t *bp = Bp.data() + (size_t)(jt / 2) * kb * 16;
            int32x4_t acc = vdupq_n_s32(0);
            for (int b = 0; b < kb; b++) {
                acc = vmmlaq_s32(acc, vld1q_s8(ap), vld1q_s8(bp));
                ap += 16;
                bp += 16;
            }
            int32_t o[4];
            vst1q_s32(o, acc);
            if (it + 0 < M && jt + 0 < N) C[(size_t)(it + 0) * N + jt + 0] = o[0];
            if (it + 0 < M && jt + 1 < N) C[(size_t)(it + 0) * N + jt + 1] = o[1];
            if (it + 1 < M && jt + 0 < N) C[(size_t)(it + 1) * N + jt + 0] = o[2];
            if (it + 1 < M && jt + 1 < N) C[(size_t)(it + 1) * N + jt + 1] = o[3];
        }
    }
}

extern "C" void lfm_s8_gemm_s32(const int8_t *A, const int8_t *B, int32_t *C,
                                int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    s8_gemm_impl(A, B, C, M, N, K);
}

// =====================================================================================
// Group B — reductions & permute (grid-stride lane loop + the would-be simd_shuffle)
// =====================================================================================

extern "C" float lfm_reduce_sum_f32(const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {           // two accumulators for ILP (mirrors grid-stride)
        a0 = vaddq_f32(a0, vld1q_f32(x + i));
        a1 = vaddq_f32(a1, vld1q_f32(x + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(a0, a1));   // ADDV/FADDP horizontal reduce
    for (; i < n; i++) acc += x[i];
    return acc;
}

extern "C" float lfm_reduce_max_f32(const float *x, int n) {
    if (n <= 0) return -INFINITY;
    float32x4_t m = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 4 <= n; i += 4) m = vmaxq_f32(m, vld1q_f32(x + i));
    float acc = vmaxvq_f32(m);
    for (; i < n; i++) acc = acc > x[i] ? acc : x[i];
    return acc;
}

// TBL/TBX arbitrary in-register permute — the closest NEON has to Metal `simd_shuffle`.
// out[i] = table[idx[i]] for idx<16, else 0 (TBL zeroes out-of-range indices). 16-byte table.
extern "C" void lfm_permute_u8(const uint8_t *table16, const uint8_t *idx, uint8_t *out, int n) {
    uint8x16_t t = vld1q_u8(table16);
    int i = 0;
    for (; i + 16 <= n; i += 16) vst1q_u8(out + i, vqtbl1q_u8(t, vld1q_u8(idx + i)));
    for (; i < n; i++) out[i] = idx[i] < 16 ? table16[idx[i]] : 0;
}

// =====================================================================================
// Group C — depthwise causal conv1d, bf16 storage / f32 accumulate / bf16 store
// (mirrors conv1d.rs depthwise_causal_conv1d_bf16). u[B,D,L], w[D,K], bias[D] -> out[B,D,Lout].
// out[b,d,t] = bias[d] + Σ_j w[d,j]·u[b,d, t-(K-1)+j]  (out-of-range taps contribute 0).
// =====================================================================================

extern "C" void lfm_depthwise_causal_conv1d_bf16(const uint16_t *u, const uint16_t *w,
                                                 const uint16_t *bias, uint16_t *out,
                                                 int Bn, int D, int L, int K, int Lout);

FK_TGT_BF16
static void conv1d_channel(const uint16_t *urow, const uint16_t *wrow, float biasf,
                           uint16_t *orow, int L, int K, int Lout) {
    // interior t in [K-1, min(Lout,L)) has all taps in-bounds -> vectorize 4 outputs at a time
    const int lo = K - 1;
    const int hi = Lout < L ? Lout : L;
    int t = 0;
    for (; t < lo && t < Lout; t++) {
        float acc = biasf;
        for (int j = 0; j < K; j++) {
            int idx = t - (K - 1) + j;
            if (idx >= 0 && idx < L) acc += bf16_to_f32(urow[idx]) * bf16_to_f32(wrow[j]);
        }
        orow[t] = f32_to_bf16_bits(acc);
    }
    for (; t + 4 <= hi; t += 4) {
        float32x4_t acc = vdupq_n_f32(biasf);
        for (int j = 0; j < K; j++) {
            int idx = t - (K - 1) + j;              // urow[idx .. idx+3] contiguous, in-bounds
            uint16x4_t bits = vld1_u16(urow + idx);
            float32x4_t uf = vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
            acc = vfmaq_n_f32(acc, uf, bf16_to_f32(wrow[j]));
        }
        bfloat16x4_t ob = vcvt_bf16_f32(acc);
        vst1_bf16((bfloat16_t *)(orow + t), ob);
    }
    for (; t < Lout; t++) {
        float acc = biasf;
        for (int j = 0; j < K; j++) {
            int idx = t - (K - 1) + j;
            if (idx >= 0 && idx < L) acc += bf16_to_f32(urow[idx]) * bf16_to_f32(wrow[j]);
        }
        orow[t] = f32_to_bf16_bits(acc);
    }
}

extern "C" void lfm_depthwise_causal_conv1d_bf16(const uint16_t *u, const uint16_t *w,
                                                 const uint16_t *bias, uint16_t *out,
                                                 int Bn, int D, int L, int K, int Lout) {
    for (int b = 0; b < Bn; b++) {
        for (int d = 0; d < D; d++) {
            const uint16_t *urow = u + ((size_t)b * D + d) * L;
            uint16_t *orow = out + ((size_t)b * D + d) * Lout;
            conv1d_channel(urow, w + (size_t)d * K, bf16_to_f32(bias[d]), orow, L, K, Lout);
        }
    }
}

static inline void depthwise_stream_copy(const uint16_t *src, uint16_t *dst, int count) {
    int i = 0;
    for (; i + 8 <= count; i += 8)
        vst1q_u16(dst + i, vld1q_u16(src + i));
    volatile uint16_t *tail = dst;
    for (; i < count; ++i) tail[i] = src[i];
}

static inline void depthwise_stream_zero(uint16_t *dst, int count) {
    volatile uint16_t *tail = dst;
    for (int i = 0; i < count; ++i) tail[i] = 0;
}

// CPU streaming depthwise grid. The virtual input row is
// `[cache | x]`, but the two payloads remain split: no staging concat is constructed.
// Each architecture call owns complete rows, vectorizing the all-x interior while
// preserving explicit FMA accumulation and one final bf16 rounding per output cell.
extern "C" int lfm_depthwise_stream_bf16_available(void) {
    return lfm_bf16_gemm_available();
}

FK_TGT_BF16
extern "C" void lfm_depthwise_stream_bf16(const uint16_t *x, const uint16_t *cache,
                                           const uint16_t *weights, uint16_t *out,
                                           uint16_t *next,
                                           int Bn, int D, int T, int K) {
    const int P = K - 1;
    const int rows = Bn * D;
    for (int row = 0; row < rows; ++row) {
        const int channel = row % D;
        const uint16_t *xrow = x + (size_t)row * T;
        const uint16_t *crow = cache ? cache + (size_t)row * P : nullptr;
        const uint16_t *wrow = weights + (size_t)channel * K;
        uint16_t *orow = out + (size_t)row * T;
        int t = 0;

        // Boundary cells still read prior-stream state.
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

        // Once t >= P every tap is a contiguous read from the incoming chunk.
        for (; t + 4 <= T; t += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int j = 0; j < K; ++j) {
                const int source = t - P + j;
                const uint16x4_t bits = vld1_u16(xrow + source);
                const float32x4_t values =
                    vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
                acc = vfmaq_n_f32(acc, values, bf16_to_f32(wrow[j]));
            }
            const bfloat16x4_t rounded = vcvt_bf16_f32(acc);
            vst1_bf16((bfloat16_t *)(orow + t), rounded);
        }
        for (; t < T; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                acc = std::fma(bf16_to_f32(xrow[t - P + j]),
                               bf16_to_f32(wrow[j]), acc);
            }
            orow[t] = f32_to_bf16_bits(acc);
        }

        // The only state movement: K-1 cells, written directly into the result plane.
        if (P == 0) continue;
        uint16_t *next_row = next + (size_t)row * P;
        if (T >= P) {
            depthwise_stream_copy(xrow + T - P, next_row, P);
            continue;
        }
        const int retained = P - T;
        if (crow)
            depthwise_stream_copy(crow + T, next_row, retained);
        else
            depthwise_stream_zero(next_row, retained);
        depthwise_stream_copy(xrow, next_row + retained, T);
    }
}

// =====================================================================================
// Group D — complex radix-2 FFT via FCMLA (mirrors FFTConv.metal fft_radix2).
// In-place, interleaved [re,im] f32, n a power of two. `inverse`!=0 -> IFFT (conj+scale).
// The per-butterfly complex multiply w·x uses the FCMLA opcode (vcmla_f32 + rot90).
// =====================================================================================

// single complex multiply a*b via two FCMLA (rot0 then rot90-accumulate)
FK_TGT_FCMA
static inline float32x2_t cmul_fcma(float32x2_t a, float32x2_t b) {
    float32x2_t acc = vcmla_f32(vdup_n_f32(0.0f), a, b);
    return vcmla_rot90_f32(acc, a, b);
}

FK_TGT_FCMA
static void fft_impl(float *data, int n, int inverse) {
    // bit-reverse permutation
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
    const float sign = inverse ? 1.0f : -1.0f;   // e^{sign·2πi k/len}
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = sign * 2.0f * (float)M_PI / (float)len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float wr = cosf(ang * k), wi = sinf(ang * k);
                float32x2_t w = {wr, wi};
                int a = i + k, b = i + k + len / 2;
                float32x2_t x = vld1_f32(data + 2 * b);
                float32x2_t t = cmul_fcma(w, x);           // t = w · data[b]
                float32x2_t u = vld1_f32(data + 2 * a);
                vst1_f32(data + 2 * a, vadd_f32(u, t));
                vst1_f32(data + 2 * b, vsub_f32(u, t));
            }
        }
    }
    if (inverse) {
        float32x4_t inv = vdupq_n_f32(1.0f / (float)n);
        int i = 0;
        for (; i + 4 <= 2 * n; i += 4) vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), inv));
        for (; i < 2 * n; i++) data[i] /= (float)n;
    }
}

// Radix-2 only: `n` must be a power of two (the Rust wrapper asserts this before calling).
// n<=1 is the trivial identity base case.
extern "C" void lfm_fft_radix2_f32(float *data, int n, int inverse) {
    if (n <= 1) return;
    fft_impl(data, n, inverse);
}

// =====================================================================================
// Group E — double-double extended precision (mirrors double_double.metal two_sum/two_prod).
// Two f32 limbs give ~2× the mantissa; the error-free transforms use FMA. Vectorized over
// float32x4 lanes, then reduced in dd. Ships as deterministic high-accuracy sum/dot.
// =====================================================================================

namespace {
struct ddv {
    float32x4_t hi, lo;
};
static inline ddv two_sum(float32x4_t a, float32x4_t b) {
    float32x4_t s = vaddq_f32(a, b);
    float32x4_t v = vsubq_f32(s, a);
    float32x4_t e = vaddq_f32(vsubq_f32(a, vsubq_f32(s, v)), vsubq_f32(b, v));
    return {s, e};
}
static inline ddv two_prod(float32x4_t a, float32x4_t b) {
    float32x4_t p = vmulq_f32(a, b);
    float32x4_t e = vfmaq_f32(vnegq_f32(p), a, b);   // a·b - p, exact via FMA
    return {p, e};
}
static inline ddv dd_add(ddv a, ddv b) {
    ddv s = two_sum(a.hi, b.hi);
    s.lo = vaddq_f32(s.lo, vaddq_f32(a.lo, b.lo));
    float32x4_t hi = vaddq_f32(s.hi, s.lo);          // renormalize
    float32x4_t lo = vsubq_f32(s.lo, vsubq_f32(hi, s.hi));
    return {hi, lo};
}
// one scalar double-double accumulation step: fold (hi_i, lo_i) into the running (shi, slo).
static inline void dd_step(float &shi, float &slo, float hi_i, float lo_i) {
    float s = shi + hi_i;
    float v = s - shi;
    float e = (shi - (s - v)) + (hi_i - v);
    shi = s;
    slo += e + lo_i;
    float t = shi + slo;
    slo = slo - (t - shi);
    shi = t;
}
// horizontal dd reduction of the 4-lane accumulator to a scalar (shi, slo) pair (deterministic
// serial order). Returns the pair rather than a collapsed float so a ragged tail can keep
// accumulating in double-double instead of falling back to lossy plain-f32 adds.
static inline void dd_hreduce2(ddv acc, float &shi, float &slo) {
    float hi[4], lo[4];
    vst1q_f32(hi, acc.hi);
    vst1q_f32(lo, acc.lo);
    shi = 0.0f;
    slo = 0.0f;
    for (int i = 0; i < 4; i++) dd_step(shi, slo, hi[i], lo[i]);
}
} // namespace

extern "C" float lfm_dd_sum_f32(const float *x, int n) {
    ddv acc = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        ddv v = {vld1q_f32(x + i), vdupq_n_f32(0.0f)};
        acc = dd_add(acc, v);
    }
    float shi, slo;
    dd_hreduce2(acc, shi, slo);
    // ragged tail: keep folding into the double-double accumulator. A plain-f32 `r += x[i]`
    // would drop any tail element below r's ULP, defeating the high-accuracy contract.
    for (; i < n; i++) dd_step(shi, slo, x[i], 0.0f);
    return shi + slo;
}

extern "C" float lfm_dd_dot_f32(const float *a, const float *b, int n) {
    ddv acc = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = dd_add(acc, two_prod(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
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
// Group F — GPU-style fast-math (FRECPE/FRSQRTE estimate + Newton refinement).
// =====================================================================================

extern "C" void lfm_recip_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t r = vrecpeq_f32(v);
        r = vmulq_f32(r, vrecpsq_f32(v, r));
        r = vmulq_f32(r, vrecpsq_f32(v, r));
        vst1q_f32(out + i, r);
    }
    for (; i < n; i++) out[i] = 1.0f / x[i];
}

extern "C" void lfm_rsqrt_f32(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t r = vrsqrteq_f32(v);
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(v, r), r));
        vst1q_f32(out + i, r);
    }
    for (; i < n; i++) out[i] = 1.0f / sqrtf(x[i]);
}

// =====================================================================================
// Group G — flat-grid conv kernels (ComplexMul.metal, Depthwise3.metal, conv1d_update.rs).
// One thread per output on the GPU -> a plain SIMD loop here (no threadgroup state).
// =====================================================================================

// Elementwise complex multiply — ComplexMul.metal's FIXED evaluation order, deliberately
// NO FMA: out = ((ar·br) − (ai·bi), (ar·bi) + (ai·br)), every product and sum rounded
// separately. vld2q de-interleaves 4 complexes; vmulq/vsubq/vaddq keep the separate
// roundings (the fused FCMLA path in Group D is exactly what this kernel must NOT use).
// a/b/out are n interleaved [re,im] pairs.
extern "C" void lfm_complex_mul_f32(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t av = vld2q_f32(a + 2 * i);
        float32x4x2_t bv = vld2q_f32(b + 2 * i);
        float32x4x2_t ov;
        ov.val[0] = vsubq_f32(vmulq_f32(av.val[0], bv.val[0]), vmulq_f32(av.val[1], bv.val[1]));
        ov.val[1] = vaddq_f32(vmulq_f32(av.val[0], bv.val[1]), vmulq_f32(av.val[1], bv.val[0]));
        vst2q_f32(out + 2 * i, ov);
    }
    for (; i < n; i++) {
        float ar = a[2 * i], ai = a[2 * i + 1], br = b[2 * i], bi = b[2 * i + 1];
        out[2 * i] = (ar * br) - (ai * bi);
        out[2 * i + 1] = (ar * bi) + (ai * br);
    }
}

// Deterministic 3-tap depthwise conv1d — Depthwise3.metal, both window directions, with the
// kernel's fixed multiply-add order ((x0·w0) + (x1·w1), then + (x2·w2)) and NO FMA, so the
// SIMD body is bit-identical to the scalar edges. x[B,C,L], k[C,3], y[B,C,L].
namespace {
// forward window (zero-pad right): y[t] = x[t]·w0 + x[t+1]·w1 + x[t+2]·w2.
static void dw3_row(const float *x, const float *w, float *y, int L) {
    const float32x4_t w0 = vdupq_n_f32(w[0]), w1 = vdupq_n_f32(w[1]), w2 = vdupq_n_f32(w[2]);
    int t = 0;
    for (; t + 6 <= L; t += 4) { // outputs t..t+3 read x[t..t+5], all in-bounds
        float32x4_t acc = vaddq_f32(vmulq_f32(vld1q_f32(x + t), w0),
                                    vmulq_f32(vld1q_f32(x + t + 1), w1));
        acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(x + t + 2), w2));
        vst1q_f32(y + t, acc);
    }
    for (; t < L; t++) {
        float x0 = x[t];
        float x1 = (t + 1 < L) ? x[t + 1] : 0.0f;
        float x2 = (t + 2 < L) ? x[t + 2] : 0.0f;
        float acc = (x0 * w[0]) + (x1 * w[1]);
        y[t] = acc + (x2 * w[2]);
    }
}
// causal window (left-pad K-1=2): y[t] = x[t-2]·w0 + x[t-1]·w1 + x[t]·w2 — the LFM2
// short-conv orientation (Depthwise3.metal `depthwise3_causal`).
static void dw3_causal_row(const float *x, const float *w, float *y, int L) {
    const float32x4_t w0 = vdupq_n_f32(w[0]), w1 = vdupq_n_f32(w[1]), w2 = vdupq_n_f32(w[2]);
    int t = 0;
    for (; t < 2 && t < L; t++) {
        float x0 = (t >= 2) ? x[t - 2] : 0.0f;
        float x1 = (t >= 1) ? x[t - 1] : 0.0f;
        float acc = (x0 * w[0]) + (x1 * w[1]);
        y[t] = acc + (x[t] * w[2]);
    }
    for (; t + 4 <= L; t += 4) { // outputs t..t+3 read x[t-2..t+3], in-bounds for t >= 2
        float32x4_t acc = vaddq_f32(vmulq_f32(vld1q_f32(x + t - 2), w0),
                                    vmulq_f32(vld1q_f32(x + t - 1), w1));
        acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(x + t), w2));
        vst1q_f32(y + t, acc);
    }
    for (; t < L; t++) {
        float acc = (x[t - 2] * w[0]) + (x[t - 1] * w[1]);
        y[t] = acc + (x[t] * w[2]);
    }
}
} // namespace

extern "C" void lfm_depthwise3_f32(const float *x, const float *k, float *y,
                                   int Bn, int C, int L) {
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < C; c++)
            dw3_row(x + ((size_t)b * C + c) * L, k + (size_t)c * 3,
                    y + ((size_t)b * C + c) * L, L);
}

extern "C" void lfm_depthwise3_causal_f32(const float *x, const float *k, float *y,
                                          int Bn, int C, int L) {
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < C; c++)
            dw3_causal_row(x + ((size_t)b * C + c) * L, k + (size_t)c * 3,
                           y + ((size_t)b * C + c) * L, L);
}

// Fused LFM2 ShortConv decode-step update — conv1d_update.rs's kernel: per (b,c) row,
// `y[t] = C[t] · Σ_j w[j]·win[t+j]` over the extended window `win = [state | B⊙x]`, with the
// carried state advanced functionally (`out = [y | new_state]`, new_state = the last K-1 conv
// inputs). Rewritten from the GPU's per-t register shift into a K-tap FIR over the extended
// buffer — same ascending-tap accumulation, same values. The multiply-adds are FMA
// (contractible IS the trained regime — Tri Dao's CUDA kernel compiles to FMA; the strict-order
// instrument is `lfm_depthwise3_causal_f32` above). bcx[B,3D,T] rows B|C|x, state[B,D,K-1],
// w[D,K], out[B,D,T+K-1].
namespace {
thread_local std::vector<float> g_bx; // [K-1 + T] extended conv-input window, per worker

// f32: gate, FIR, gate — no storage rounding anywhere (float is the storage dtype).
static void update_row_f32(const float *brow, const float *crow, const float *xrow,
                           const float *srow, const float *wrow, float *orow, int T, int K) {
    const int km1 = K - 1;
    g_bx.resize((size_t)km1 + T);
    float *bx = g_bx.data();
    for (int j = 0; j < km1; j++) bx[j] = srow[j];
    int t = 0;
    for (; t + 4 <= T; t += 4)
        vst1q_f32(bx + km1 + t, vmulq_f32(vld1q_f32(brow + t), vld1q_f32(xrow + t)));
    for (; t < T; t++) bx[km1 + t] = brow[t] * xrow[t];
    t = 0;
    for (; t + 4 <= T; t += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int j = 0; j < K; j++)
            acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(bx + t + j), vdupq_n_f32(wrow[j])));
        vst1q_f32(orow + t, vmulq_f32(vld1q_f32(crow + t), acc));
    }
    for (; t < T; t++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) acc = acc + wrow[j] * bx[t + j];
        orow[t] = crow[t] * acc;
    }
    for (int j = 0; j < km1; j++) orow[T + j] = bx[T + j];
}

// bf16 storage: compute in f32, but round B⊙x through bf16 BEFORE it enters the window and
// the conv output through bf16 BEFORE the C gate — torch materializes both as bf16 tensors,
// so the trained regime includes those roundings (conv1d_update.rs kernel_source).
// Round-to-nearest-even via the integer trick (same as f32_to_bf16_bits), vectorized.
static inline float32x4_t round_bf16_f32x4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    u = vaddq_u32(u, vaddq_u32(lsb, vdupq_n_u32(0x7fff)));
    u = vandq_u32(u, vdupq_n_u32(0xFFFF0000u));
    return vreinterpretq_f32_u32(u);
}
static inline uint16x4_t bf16_bits_f32x4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    u = vaddq_u32(u, vaddq_u32(lsb, vdupq_n_u32(0x7fff)));
    return vmovn_u32(vshrq_n_u32(u, 16));
}
static inline float32x4_t bf16_widen4(const uint16_t *p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
// scalar RNE round kept-as-f32 (for edges) — same integer trick.
static inline float round_bf16_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    u &= 0xFFFF0000u;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static inline uint16_t bf16_bits_scalar(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static void update_row_bf16(const uint16_t *brow, const uint16_t *crow, const uint16_t *xrow,
                            const uint16_t *srow, const float *wf, uint16_t *orow, int T, int K) {
    const int km1 = K - 1;
    g_bx.resize((size_t)km1 + T);
    float *bx = g_bx.data();
    for (int j = 0; j < km1; j++) bx[j] = bf16_to_f32(srow[j]);
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        float32x4_t prod = vmulq_f32(bf16_widen4(brow + t), bf16_widen4(xrow + t));
        vst1q_f32(bx + km1 + t, round_bf16_f32x4(prod)); // Bx rounds through bf16 storage
    }
    for (; t < T; t++) bx[km1 + t] = round_bf16_scalar(bf16_to_f32(brow[t]) * bf16_to_f32(xrow[t]));
    t = 0;
    for (; t + 4 <= T; t += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int j = 0; j < K; j++)
            acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(bx + t + j), vdupq_n_f32(wf[j])));
        acc = round_bf16_f32x4(acc); // conv output rounds through bf16 before the C gate
        float32x4_t y = vmulq_f32(bf16_widen4(crow + t), acc);
        vst1_u16(orow + t, bf16_bits_f32x4(y));
    }
    for (; t < T; t++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) acc = acc + wf[j] * bx[t + j];
        acc = round_bf16_scalar(acc);
        orow[t] = bf16_bits_scalar(bf16_to_f32(crow[t]) * acc);
    }
    // new_state values are already bf16-rounded, so the store round-trips exactly.
    for (int j = 0; j < km1; j++) orow[T + j] = bf16_bits_scalar(bx[T + j]);
}
} // namespace

namespace {
// T==1, K==3 — the LFM2 decode step, vectorized ACROSS CHANNELS (t-vectorization is
// degenerate at T==1). With T==1 the B/C/x rows are contiguous across channels; the state
// [c][2] de-interleaves with vld2q, the taps [c][3] with vld3q, and each channel's output
// triple [y | s1 | bx] stores as one interleaved vst3q. Same ascending-tap FMA accumulation
// as the general row kernel.
static void update_step_k3_f32(const float *ball, const float *call, const float *xall,
                               const float *state, const float *w, float *out, int D) {
    int c = 0;
    for (; c + 4 <= D; c += 4) {
        float32x4x2_t s = vld2q_f32(state + 2 * c);
        float32x4x3_t wv = vld3q_f32(w + 3 * c);
        float32x4_t bx = vmulq_f32(vld1q_f32(ball + c), vld1q_f32(xall + c));
        float32x4_t acc = vmulq_f32(s.val[0], wv.val[0]);
        acc = vaddq_f32(acc, vmulq_f32(s.val[1], wv.val[1]));
        acc = vaddq_f32(acc, vmulq_f32(bx, wv.val[2]));
        float32x4x3_t o;
        o.val[0] = vmulq_f32(vld1q_f32(call + c), acc); // y
        o.val[1] = s.val[1];                            // new_state[0] = old state[1]
        o.val[2] = bx;                                  // new_state[1] = this step's Bx
        vst3q_f32(out + 3 * c, o);
    }
    for (; c < D; c++) {
        float bx = ball[c] * xall[c];
        float acc = w[3 * c + 0] * state[2 * c + 0];
        acc = acc + w[3 * c + 1] * state[2 * c + 1];
        acc = acc + w[3 * c + 2] * bx;
        out[3 * c + 0] = call[c] * acc;
        out[3 * c + 1] = state[2 * c + 1];
        out[3 * c + 2] = bx;
    }
}
} // namespace

extern "C" void lfm_conv1d_update_f32(const float *bcx, const float *state, const float *w,
                                      float *out, int Bn, int D, int T, int K) {
    if (T == 1 && K == 3) {
        for (int b = 0; b < Bn; b++) {
            const float *base = bcx + (size_t)b * 3 * D;
            update_step_k3_f32(base, base + D, base + 2 * D, state + (size_t)b * D * 2, w,
                               out + (size_t)b * D * 3, D);
        }
        return;
    }
    for (int b = 0; b < Bn; b++)
        for (int c = 0; c < D; c++) {
            const float *brow = bcx + (((size_t)b * 3 + 0) * D + c) * T;
            const float *crow = bcx + (((size_t)b * 3 + 1) * D + c) * T;
            const float *xrow = bcx + (((size_t)b * 3 + 2) * D + c) * T;
            update_row_f32(brow, crow, xrow, state + ((size_t)b * D + c) * (K - 1),
                           w + (size_t)c * K, out + ((size_t)b * D + c) * (T + K - 1), T, K);
        }
}

namespace {
// bf16 widen for a 4-lane bit vector already in a register.
static inline float32x4_t bf16_widen_bits4(uint16x4_t bits) {
    return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}
// bits of an ALREADY-bf16-rounded f32 vector (low 16 bits zero) — pure truncation, no rounding.
static inline uint16x4_t bits_of_rounded4(float32x4_t v) {
    return vmovn_u32(vshrq_n_u32(vreinterpretq_u32_f32(v), 16));
}

// T==1, K==3 decode step, bf16 storage — channel-vectorized like the f32 twin, with the
// trained-regime rounding points (Bx and conv-out round through bf16) and the carried
// state's old s1 passed through as RAW bits (already bf16 — exact).
static void update_step_k3_bf16(const uint16_t *ball, const uint16_t *call, const uint16_t *xall,
                                const uint16_t *state, const uint16_t *w, uint16_t *out, int D) {
    int c = 0;
    for (; c + 4 <= D; c += 4) {
        uint16x4x2_t sb = vld2_u16(state + 2 * c);
        uint16x4x3_t wb = vld3_u16(w + 3 * c);
        float32x4_t bx = round_bf16_f32x4(
            vmulq_f32(bf16_widen4(ball + c), bf16_widen4(xall + c)));
        float32x4_t acc = vmulq_f32(bf16_widen_bits4(sb.val[0]), bf16_widen_bits4(wb.val[0]));
        acc = vaddq_f32(acc, vmulq_f32(bf16_widen_bits4(sb.val[1]), bf16_widen_bits4(wb.val[1])));
        acc = vaddq_f32(acc, vmulq_f32(bx, bf16_widen_bits4(wb.val[2])));
        acc = round_bf16_f32x4(acc);
        float32x4_t y = vmulq_f32(bf16_widen4(call + c), acc);
        uint16x4x3_t o;
        o.val[0] = bf16_bits_f32x4(y);
        o.val[1] = sb.val[1];            // raw pass-through: already bf16
        o.val[2] = bits_of_rounded4(bx);
        vst3_u16(out + 3 * c, o);
    }
    for (; c < D; c++) {
        float bx = round_bf16_scalar(bf16_to_f32(ball[c]) * bf16_to_f32(xall[c]));
        float acc = bf16_to_f32(w[3 * c + 0]) * bf16_to_f32(state[2 * c + 0]);
        acc = acc + bf16_to_f32(w[3 * c + 1]) * bf16_to_f32(state[2 * c + 1]);
        acc = acc + bf16_to_f32(w[3 * c + 2]) * bx;
        acc = round_bf16_scalar(acc);
        out[3 * c + 0] = bf16_bits_scalar(bf16_to_f32(call[c]) * acc);
        out[3 * c + 1] = state[2 * c + 1];
        out[3 * c + 2] = bf16_bits_scalar(bx);
    }
}
} // namespace

extern "C" void lfm_conv1d_update_bf16(const uint16_t *bcx, const uint16_t *state,
                                       const uint16_t *w, uint16_t *out,
                                       int Bn, int D, int T, int K) {
    if (T == 1 && K == 3) {
        for (int b = 0; b < Bn; b++) {
            const uint16_t *base = bcx + (size_t)b * 3 * D;
            update_step_k3_bf16(base, base + D, base + 2 * D, state + (size_t)b * D * 2, w,
                                out + (size_t)b * D * 3, D);
        }
        return;
    }
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
// Group H — decode stage kernels: the per-stage device functions of the pure-NEON decode
// step (no candle op in the token loop). Each consumes/produces bf16 bit planes or f32
// scratch exactly at the torch rounding points; lane teams (Rust side) slice rows and
// barrier between stages, these do the math.
// =====================================================================================

// Σ f32(x)² over a bf16 plane — the RMSNorm reduction.
extern "C" float lfm_bf16_sumsq_f32(const uint16_t *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t b = vld1q_u16(x + i);
        float32x4_t lo = bf16_row_lo(b), hi = bf16_row_hi(b);
        a0 = vfmaq_f32(a0, lo, lo);
        a1 = vfmaq_f32(a1, hi, hi);
    }
    float acc = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; i++) {
        float v = bf16_to_f32(x[i]);
        acc = fmaf(v, v, acc);
    }
    return acc;
}

// RMSNorm apply: out = rb(f32(x) · inv_rms · f32(w)) — f32 throughout, ONE bf16 round
// (transformer.rs RmsNorm::forward's ladder).
extern "C" void lfm_bf16_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *out,
                                 int n, float inv_rms) {
    const float32x4_t rs = vdupq_n_f32(inv_rms);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = bf16_widen4(x + i);
        float32x4_t wv = bf16_widen4(w + i);
        vst1_u16(out + i, bf16_bits_f32x4(vmulq_f32(vmulq_f32(xv, rs), wv)));
    }
    for (; i < n; i++)
        out[i] = bf16_bits_scalar(bf16_to_f32(x[i]) * inv_rms * bf16_to_f32(w[i]));
}

// bf16 elementwise add (the residual ladder): out = rb(f32(a) + f32(b)).
extern "C" void lfm_bf16_add(const uint16_t *a, const uint16_t *b, uint16_t *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1_u16(out + i, bf16_bits_f32x4(vaddq_f32(bf16_widen4(a + i), bf16_widen4(b + i))));
    for (; i < n; i++) out[i] = bf16_bits_scalar(bf16_to_f32(a[i]) + bf16_to_f32(b[i]));
}

// SwiGLU gate ladder over post-GEMV f32 planes: out = rb(rb(silu(rb(g))) · rb(u)) — the
// candle op chain's rounds (linear-out, silu, linear-out, gating mul). expf is libm
// (matches candle's per-element silu).
extern "C" void lfm_swiglu_bf16(const float *g, const float *u, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        float gv = bf16_to_f32(bf16_bits_scalar(g[i]));
        float sg = bf16_to_f32(bf16_bits_scalar(gv / (1.0f + expf(-gv))));
        float uv = bf16_to_f32(bf16_bits_scalar(u[i]));
        out[i] = bf16_bits_scalar(sg * uv);
    }
}

// Softmax with folded pre-scale, in place, f32: x = softmax(x · scale). Exact libm expf
// (candle's softmax path), max-subtracted for stability.
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

// Attention value gather: out[hd] += att[t] · V[t,·] for t in [0,len) — axpy over the
// resident f32 V plane rows (V row-major [len, hd]).
extern "C" void lfm_attn_av_f32(const float *att, const float *v, float *out, int len, int hd) {
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < len; t++) {
        const float a = att[t];
        const float *row = v + (size_t)t * hd;
        int i = 0;
        for (; i + 4 <= hd; i += 4)
            vst1q_f32(out + i, vfmaq_n_f32(vld1q_f32(out + i), vld1q_f32(row + i), a));
        for (; i < hd; i++) out[i] = fmaf(a, row[i], out[i]);
    }
}

// Attention scores: att[t] = dot(q, K[t,·]) for t in [0,len) over the resident f32 K
// plane — the q·Kᵀ row of scaled-dot-product attention (scale folds into softmax).
extern "C" void lfm_attn_qk_f32(const float *q, const float *k, float *att, int len, int hd) {
    for (int t = 0; t < len; t++) {
        const float *row = k + (size_t)t * hd;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= hd; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(q + i), vld1q_f32(row + i));
        float s = vaddvq_f32(acc);
        for (; i < hd; i++) s = fmaf(q[i], row[i], s);
        att[t] = s;
    }
}

// Interleaved (GPT-J) rotary on ONE head row, f32 in place: pairs (2i, 2i+1) rotated by
// (cos[i], sin[i]) — `apply_rotary_emb`'s real-valued rope_i. cos/sin are the model's
// precomputed f32 tables at this position, hd/2 entries.
extern "C" void lfm_rope_i_f32(float *x, const float *cos_p, const float *sin_p, int hd) {
    for (int i = 0; i + 1 < hd; i += 2) {
        float c = cos_p[i / 2], s = sin_p[i / 2];
        float x0 = x[i], x1 = x[i + 1];
        x[i] = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

// bf16 plane → f32 plane (the fp32-upcast points of the torch ladder), and back with RNE.
extern "C" void lfm_bf16_to_f32(const uint16_t *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, bf16_widen4(x + i));
    for (; i < n; i++) out[i] = bf16_to_f32(x[i]);
}
extern "C" void lfm_f32_to_bf16(const float *x, uint16_t *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1_u16(out + i, bf16_bits_f32x4(vld1q_f32(x + i)));
    for (; i < n; i++) out[i] = bf16_bits_scalar(x[i]);
}

// bf16-plane attention: K/V stay in checkpoint dtype (torch's cache dtype — half the
// bytes, half the read bandwidth of f32 planes); rows widen to f32 IN REGISTERS during
// the dot (a shift — free). q is f32 (the sdpa upcast point). Same math as the f32 forms.
extern "C" void lfm_attn_qk_bf16(const float *q, const uint16_t *k, float *att, int len, int hd) {
    for (int t = 0; t < len; t++) {
        const uint16_t *row = k + (size_t)t * hd;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= hd; i += 8) {
            uint16x8_t b = vld1q_u16(row + i);
            a0 = vfmaq_f32(a0, vld1q_f32(q + i), bf16_row_lo(b));
            a1 = vfmaq_f32(a1, vld1q_f32(q + i + 4), bf16_row_hi(b));
        }
        float s = vaddvq_f32(vaddq_f32(a0, a1));
        for (; i < hd; i++) s = fmaf(q[i], bf16_to_f32(row[i]), s);
        att[t] = s;
    }
}

extern "C" void lfm_attn_av_bf16(const float *att, const uint16_t *v, float *out, int len, int hd) {
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < len; t++) {
        const float a = att[t];
        const uint16_t *row = v + (size_t)t * hd;
        int i = 0;
        for (; i + 8 <= hd; i += 8) {
            uint16x8_t b = vld1q_u16(row + i);
            vst1q_f32(out + i, vfmaq_n_f32(vld1q_f32(out + i), bf16_row_lo(b), a));
            vst1q_f32(out + i + 4, vfmaq_n_f32(vld1q_f32(out + i + 4), bf16_row_hi(b), a));
        }
        for (; i < hd; i++) out[i] = fmaf(a, bf16_to_f32(row[i]), out[i]);
    }
}

// Sequential-order sumsq matching candle's `sqr().sum()` ladder EXACTLY: each square
// rounded (separate mul), each add rounded (no FMA, no vector partials). This is the
// token-exact norm reduction for fused blocks that must bit-match the composed op chain;
// the FMA/partials form (lfm_bf16_sumsq_f32) is the fast ulp-tier form.
extern "C" float lfm_bf16_sumsq_seq_f32(const uint16_t *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = bf16_to_f32(x[i]);
        float sq = v * v;
        acc = acc + sq;
    }
    return acc;
}

// Sumsq in CANDLE's exact f32 reduction order (cpu/neon.rs vec_sum over a sqr() tensor):
// four float32x4 accumulators over 16-element steps, pairwise tree (x0+=x1, x2+=x3,
// x0+=x2), ADDV, then sequential leftovers. Each square rounds before accumulating (the
// sqr() tensor's values). This is the token-exact norm reduction for fused blocks that
// must bit-match the composed candle chain on aarch64.
extern "C" float lfm_bf16_sumsq_candle_f32(const uint16_t *x, int n) {
    const int np = n & ~15;
    float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);
    for (int i = 0; i < np; i += 16) {
        float32x4_t x0 = bf16_widen4(x + i);
        float32x4_t x1 = bf16_widen4(x + i + 4);
        float32x4_t x2 = bf16_widen4(x + i + 8);
        float32x4_t x3 = bf16_widen4(x + i + 12);
        sum0 = vaddq_f32(sum0, vmulq_f32(x0, x0));
        sum1 = vaddq_f32(sum1, vmulq_f32(x1, x1));
        sum2 = vaddq_f32(sum2, vmulq_f32(x2, x2));
        sum3 = vaddq_f32(sum3, vmulq_f32(x3, x3));
    }
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    float acc = vaddvq_f32(sum0);
    for (int i = np; i < n; i++) {
        float v = bf16_to_f32(x[i]);
        acc = acc + v * v;
    }
    return acc;
}

// Sampling leaves. Vocabulary bands stay in the checkpoint/logit plane; only
// derived weights are written. NEON handles the comparison sweeps while expf
// remains the platform scalar primitive used by Candle's CPU softmax.
extern "C" uint32_t lfm_sampler_argmax_f32(const float *x, size_t count) {
    if (count == 0) return 0;
    float maximum = -INFINITY;
    size_t i = 0;
    for (; i + 4 <= count; i += 4) maximum = vmaxvq_f32(vmaxq_f32(vdupq_n_f32(maximum), vld1q_f32(x + i)));
    for (; i < count; ++i)
        if (x[i] > maximum) maximum = x[i];
    for (i = 0; i < count; ++i)
        if (x[i] == maximum) return (uint32_t)i;
    return 0;
}

extern "C" uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count) {
    if (count == 0) return 0;
    float maximum = -INFINITY;
    size_t i = 0;
    for (; i + 4 <= count; i += 4) maximum = vmaxvq_f32(vmaxq_f32(vdupq_n_f32(maximum), bf16_widen4(x + i)));
    for (; i < count; ++i)
        if (bf16_to_f32(x[i]) > maximum) maximum = bf16_to_f32(x[i]);
    for (i = 0; i < count; ++i)
        if (bf16_to_f32(x[i]) == maximum) return (uint32_t)i;
    return 0;
}

extern "C" float lfm_sampler_exp_sum_f32(const float *x, float *weights,
                                           size_t count, float scale,
                                           float maximum, float threshold) {
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t scaled = vmulq_n_f32(vld1q_f32(x + i), scale);
        float values[4];
        vst1q_f32(values, scaled);
        for (size_t j = 0; j < 4; ++j) {
            float weight = values[j] >= threshold ? expf(values[j] - maximum) : 0.0f;
            weights[i + j] = weight;
            sum += weight;
        }
    }
    for (; i < count; ++i) {
        float value = x[i] * scale;
        float weight = value >= threshold ? expf(value - maximum) : 0.0f;
        weights[i] = weight;
        sum += weight;
    }
    return sum;
}

extern "C" float lfm_sampler_exp_sum_bf16(const uint16_t *x, float *weights,
                                            size_t count, uint16_t bf16_scale,
                                            float maximum, float threshold) {
    float scale = bf16_to_f32(bf16_scale);
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float value = bf16_to_f32(bf16_bits_scalar(bf16_to_f32(x[i]) * scale));
        float weight = value >= threshold ? expf(value - maximum) : 0.0f;
        weights[i] = weight;
        sum += weight;
    }
    return sum;
}

extern "C" uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count,
                                              float target) {
    float prefix = 0.0f;
    uint32_t last = 0;
    for (size_t i = 0; i < count; ++i) {
        if (weights[i] > 0.0f) last = (uint32_t)i;
        prefix += weights[i];
        if (target < prefix) return (uint32_t)i;
    }
    return last;
}
