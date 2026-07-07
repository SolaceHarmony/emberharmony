// NEON "zoo" — a library of aarch64 SIMD procedures that mirror the GPU idioms of the
// crate's JIT-embedded Metal kernels (experiments/lfm2-audio-voice/candle-flashfftconv),
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
// All entry points are `extern "C"` (flat FFI to src/neon_zoo.rs). C++17 internally for the
// double-double struct. Each feature-specific block is confined to a function carrying a
// per-compiler target attribute so no gated opcode leaks into an ungated function; callers
// runtime-gate on NeonFeatures (src/neon_zoo.rs). Verified with aarch64-linux-gnu-g++ +
// qemu-aarch64 -cpu max; ships compiled by build.rs on aarch64 (clang on macOS).

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <cmath>

// --- Per-function target gating -------------------------------------------------------
// clang exposes the ACLE intrinsics only when the *base* -march enables the feature, so on
// the clang build (build.rs) the base march carries every feature and these macros are
// empty. GCC always declares the intrinsics (target-pragma-wrapped in arm_neon.h) and honours
// a per-function `target("arch=...")`, keeping the base march low and each opcode isolated.
#if defined(__clang__)
#define ZOO_TGT_BF16
#define ZOO_TGT_I8MM
#define ZOO_TGT_FCMA
#else
#define ZOO_TGT_BF16 __attribute__((target("arch=armv8.2-a+bf16")))
#define ZOO_TGT_I8MM __attribute__((target("arch=armv8.2-a+i8mm")))
#define ZOO_TGT_FCMA __attribute__((target("arch=armv8.3-a")))
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
ZOO_TGT_BF16
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

ZOO_TGT_BF16
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

ZOO_TGT_BF16
static void gemv_impl(const bfloat16_t *A, const bfloat16_t *B, float *C, int N, int K) {
    const int Kp = (K + 7) & ~7;
    static thread_local std::vector<bfloat16_t> Bt;   // [N][Kp] col-major, zero-padded
    static thread_local std::vector<bfloat16_t> Aa;   // [Kp] zero-padded copy of A
    Bt.assign((size_t)N * Kp, (bfloat16_t)0);
    Aa.assign((size_t)Kp, (bfloat16_t)0);
    for (int k = 0; k < K; k++) Aa[k] = A[k];
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) Bt[(size_t)n * Kp + k] = B[(size_t)k * N + n];
    for (int n = 0; n < N; n++) {
        const bfloat16_t *bcol = Bt.data() + (size_t)n * Kp;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = 0; k < Kp; k += 8) {
            bfloat16x8_t a8 = vld1q_bf16(Aa.data() + k);
            bfloat16x8_t b8 = vld1q_bf16(bcol + k);
            acc = vbfdotq_f32(acc, a8, b8);
        }
        C[n] = vaddvq_f32(acc);
    }
}

extern "C" void lfm_bf16_gemv_f32(const uint16_t *A_, const uint16_t *B_, float *C,
                                  int N, int K) {
    if (N <= 0 || K <= 0) return;
    gemv_impl((const bfloat16_t *)A_, (const bfloat16_t *)B_, C, N, K);
}

// Stretch: int8 tensor-core MAC — same 8×8 idiom via SMMLA, showing the pattern generalizes
// across dtypes. C(M,N) s32 = A(M,K) s8 · B(K,N) s8. Reference-quality (2×2 SMMLA tile).
extern "C" void lfm_s8_gemm_s32(const int8_t *A, const int8_t *B, int32_t *C,
                                int M, int N, int K);

ZOO_TGT_I8MM
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

ZOO_TGT_BF16
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

// =====================================================================================
// Group D — complex radix-2 FFT via FCMLA (mirrors FFTConv.metal fft_radix2).
// In-place, interleaved [re,im] f32, n a power of two. `inverse`!=0 -> IFFT (conj+scale).
// The per-butterfly complex multiply w·x uses the FCMLA opcode (vcmla_f32 + rot90).
// =====================================================================================

// single complex multiply a*b via two FCMLA (rot0 then rot90-accumulate)
ZOO_TGT_FCMA
static inline float32x2_t cmul_fcma(float32x2_t a, float32x2_t b) {
    float32x2_t acc = vcmla_f32(vdup_n_f32(0.0f), a, b);
    return vcmla_rot90_f32(acc, a, b);
}

ZOO_TGT_FCMA
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
// horizontal dd reduction of a 4-lane dd accumulator to a scalar dd, then hi+lo
static inline float dd_hreduce(ddv acc) {
    float hi[4], lo[4];
    vst1q_f32(hi, acc.hi);
    vst1q_f32(lo, acc.lo);
    // serial dd sum across the 4 lanes (deterministic)
    float shi = 0.0f, slo = 0.0f;
    for (int i = 0; i < 4; i++) {
        float s = shi + hi[i];
        float v = s - shi;
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

extern "C" float lfm_dd_sum_f32(const float *x, int n) {
    ddv acc = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        ddv v = {vld1q_f32(x + i), vdupq_n_f32(0.0f)};
        acc = dd_add(acc, v);
    }
    float r = dd_hreduce(acc);
    for (; i < n; i++) r += x[i];
    return r;
}

extern "C" float lfm_dd_dot_f32(const float *a, const float *b, int n) {
    ddv acc = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = dd_add(acc, two_prod(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    float r = dd_hreduce(acc);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
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
