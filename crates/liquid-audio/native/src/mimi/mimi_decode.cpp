// mimi_decode.cpp — Unit #6 of the Mimi decoder port (see docs/MIMI_PORT.md).
//
// Faithful C++/NEON port of moshi 0.6.4 `Mimi::decode_step` (mimi.rs:214) plus
// the decoder-half of `Mimi::reset_state` (mimi.rs:224), plus the shared
// infrastructure the arbiter header (mimi_kernel.h) assigns to this unit:
//   - mimi_arena_alloc   (bump allocator, 64-byte aligned, abort on overflow)
//   - mimi_weight_find   (init-time weight-table lookup)
//   - mimi_gemv_f32 / mimi_gemm_f32 / mimi_softmax_f32 / mimi_gelu_erf_f32 /
//     mimi_elu_f32 / mimi_layer_norm_f32   (deterministic math primitives)
//   - mimi_decoder_new / _step / _reset / _free   (top-level orchestration)
//
// The StreamTensor Option-ness of the Rust (streaming.rs) is dissolved here into
// explicit per-stage frame counts (n_in/n_out). A stage handed 0 frames returns
// 0, exactly as `StreamTensor::empty()` propagates through each `.step` in the
// Rust chain. See the NOTES block at the bottom for the full mapping, the
// per-stage count contract, the arena sizing breakdown, the primitive loop
// orders, and the open items for the arbiter.
//
// This file compiles standalone against mimi_kernel.h alone: the unit
// entry points it calls (mimi_quant_*, mimi_upsample_*, mimi_transformer_*,
// mimi_seanet_*) are *declared* in the header and *defined* by the sibling
// units (1–5), which may not exist yet — link-time resolution, not compile-time.

#include "mimi_kernel.h"

#include <cmath>    // erff, expf, sqrtf
#include <cstdio>   // snprintf, fprintf, stderr
#include <cstdlib>  // aligned_alloc, free, abort
#include <cstring>  // strcmp, memset

// GEMM/GEMV run on Apple's AMX matrix coprocessor via Accelerate's cblas. The
// header is include-only at compile time (declarations); LINKING requires
// -framework Accelerate. Guarded so the file still builds off-Apple (falling
// back to the scalar _ref path).
#ifdef __APPLE__
// Opt into the modern (non-deprecated) CBLAS interface. LP64 only — do NOT set
// ACCELERATE_LAPACK_ILP64, so cblas args stay 32-bit `int` and match the
// header's int M/K/N ABI. Without this, cblas_sgemm/sgemv warn as deprecated on
// macOS 13.3+.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

// NEON is the primary path for the activation/elementwise/reduction sweeps and
// for layer-norm; the scalar bodies below are the _ref parity-bisect path,
// selected on non-NEON targets or with -DMIMI_SCALAR_REF.
#if (defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64)) && defined(__ARM_NEON)
#define MIMI_HAVE_NEON 1
#include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// Local sizing constants (documented estimate — the arbiter tightens later).
// ---------------------------------------------------------------------------

// Worst-case 25 Hz frames flowing through a single decode_step. One latent
// (12.5 Hz) code frame -> upsample (stride 2) yields 2 frames in steady state;
// the header reserves pcm_out at MIMI_FRAME_OUT*2 (= 4*960) of "drain
// headroom", so we size the latent-side inter-stage buffers for 4 frames to
// stay consistent with that ceiling. Real steady value is 2.
enum { MIMI_MAX_LATENT = 4 };

// Generous arena ceiling. Under the "weight-norm folds ONCE at init into the
// arena" discipline, the SEANET decoder's folded conv weights alone approach
// ~59 MiB worst-case (if the checkpoint stores weight_g/weight_v rather than a
// pre-folded "weight"), and a transformer GEMM re-arm (if any unit chooses one)
// could add ~100 MiB. 256 MiB leaves comfortable slack over the ~170 MiB worst
// case; the arbiter drops this to ~16 MiB once the fold/re-arm decisions lock
// (see NOTES (c)). The bump allocator aborts on overflow, so an undersized
// ceiling surfaces immediately as a sizing bug rather than corruption.
static const size_t MIMI_ARENA_BYTES = (size_t)256 * 1024 * 1024;
static const size_t MIMI_ARENA_ALIGN = 64;
static const size_t MIMI_ARENA_HEADROOM_MIN = (size_t)1 * 1024 * 1024;

#define MIMI_ERR(...)                              \
    do {                                           \
        if (err && errlen) {                       \
            snprintf(err, errlen, __VA_ARGS__);    \
        }                                          \
    } while (0)

// ===========================================================================
// (c) Shared infrastructure — arena
// ===========================================================================

extern "C" void *mimi_arena_alloc(MimiArena *a, size_t bytes) {
    // Bump allocator: align the current watermark up to 64 bytes, hand back the
    // block, advance. Init-time only; steady state never calls this. The
    // subtraction form of the bounds check avoids overflow in `off + bytes`.
    size_t off = (a->used + (MIMI_ARENA_ALIGN - 1)) & ~(MIMI_ARENA_ALIGN - 1);
    if (off > a->size || bytes > a->size - off) {
        fprintf(stderr,
                "mimi_arena_alloc: overflow (used=%zu req=%zu size=%zu) — arena "
                "sizing bug, raise MIMI_ARENA_BYTES\n",
                a->used, bytes, a->size);
        abort();
    }
    uint8_t *p = a->base + off;
    // Zero every carved block so unit state starts in a defined (== reset)
    // condition — POD/hibernation-friendly, and matches reset_state() semantics.
    memset(p, 0, bytes);
    a->used = off + bytes;
    return p;
}

// ===========================================================================
// (c) Shared infrastructure — weight table lookup
// ===========================================================================

extern "C" const MimiWeight *mimi_weight_find(const MimiWeightTable *t,
                                              const char *name) {
    // Init-time only: linear scan by safetensors key. NULL if absent — callers
    // (unit inits) hard-fail on a missing REQUIRED weight (no fallbacks).
    if (!t || !name) {
        return NULL;
    }
    for (uint32_t i = 0; i < t->count; ++i) {
        const MimiWeight *e = &t->entries[i];
        if (e->name && strcmp(e->name, name) == 0) {
            return e;
        }
    }
    return NULL;
}

// ===========================================================================
// (c) Shared infrastructure — math primitives
//
// Numerics tier: "faithful" (mimi_kernel.h / MIMI_PORT.md). f32 in, f32
// accumulate, documented loop order. "Math is assembly at every step":
//   - GEMM/GEMV : Apple Accelerate cblas (AMX matrix coprocessor).
//   - sweeps / softmax / layer-norm : NEON intrinsics, transcendentals applied
//     LANE-WISE with libm erff/expf (no polynomial vector approximations — that
//     would move the numerics off the faithful tier).
// Parity-bisect: build -DMIMI_SCALAR_REF to force the scalar reference bodies
// (and, off-Apple, the scalar gemm/gemv _ref siblings) and diff against them.
// ===========================================================================

// -------- gemv: y[m] = sum_k w[m*k + k] * x[k] (+ bias[m]); W row-major [M,K] --
// Scalar reference (parity-bisect path + off-Apple fallback).
[[maybe_unused]] static void mimi_gemv_f32_ref(const float *w, const float *x,
                                               const float *bias, float *y,
                                               int m, int k) {
    for (int i = 0; i < m; ++i) {
        const float *wr = w + (size_t)i * (size_t)k;
        float s = 0.0f;
        for (int j = 0; j < k; ++j) {
            s += wr[j] * x[j];  // sequential accumulation, low index -> high
        }
        if (bias) {
            s += bias[i];
        }
        y[i] = s;
    }
}

extern "C" void mimi_gemv_f32(const float *w, const float *x,
                              const float *bias_or_null, float *y, int m, int k) {
#if defined(__APPLE__) && !defined(MIMI_SCALAR_REF)
    // W is row-major [M,K] == an M-by-K cblas matrix, lda = K, no transpose.
    // beta 0 => cblas overwrites y with W*x (y is not read). Bias is a separate
    // explicit loop AFTER the cblas call (the AMX matmul carries no bias term).
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, 1.0f, w, k, x, 1, 0.0f, y, 1);
    if (bias_or_null) {
        for (int i = 0; i < m; ++i) {
            y[i] += bias_or_null[i];
        }
    }
#else
    mimi_gemv_f32_ref(w, x, bias_or_null, y, m, k);
#endif
}

// -------- gemm: C[M,N] = A[M,K]*B[K,N] (beta 0) or += (beta 1); row-major -----
// Scalar reference (parity-bisect path + off-Apple fallback), loop order i-k-j.
[[maybe_unused]] static void mimi_gemm_f32_ref(const float *a, const float *b,
                                               float *c, int m, int k, int n,
                                               int beta) {
    for (int i = 0; i < m; ++i) {
        float *cr = c + (size_t)i * (size_t)n;
        if (beta == 0) {
            for (int j = 0; j < n; ++j) {
                cr[j] = 0.0f;
            }
        }
        for (int p = 0; p < k; ++p) {
            const float aval = a[(size_t)i * (size_t)k + (size_t)p];
            const float *br = b + (size_t)p * (size_t)n;
            for (int j = 0; j < n; ++j) {
                cr[j] += aval * br[j];
            }
        }
    }
}

extern "C" void mimi_gemm_f32(const float *a, const float *b, float *c, int m,
                              int k, int n, int beta) {
#if defined(__APPLE__) && !defined(MIMI_SCALAR_REF)
    // Direct row-major mapping, NO transpose (weights are a buffer, movement is
    // theft): A[M,K] lda=K, B[K,N] ldb=N, C[M,N] ldc=N; beta 0 overwrite / 1 acc.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a,
                k, b, n, (float)beta, c, n);
#else
    mimi_gemm_f32_ref(a, b, c, m, k, n, beta);
#endif
}

// -------- scalar per-element helpers (tail/lane + _ref building blocks) -------

extern "C" float mimi_gelu_erf_f32(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2))); candle's `gelu_erf`. 1/sqrt(2) constant.
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

extern "C" float mimi_elu_f32(float x, float alpha) {
    // candle Elu(alpha): x > 0 ? x : alpha * (exp(x) - 1).
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

// -------- gelu sweep: y[i] = gelu_erf(x[i]) -----------------------------------
// NEON vectorizes the 0.5*x*(1+e) arithmetic; erff is applied lane-wise (no
// vector-poly substitution). Tail via the scalar helper.
extern "C" void mimi_gelu_erf_vec_f32(const float *x, float *y, int n) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(0.70710678118654752440f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t arg = vmulq_f32(vx, inv_sqrt2);
        float lanes[4];
        vst1q_f32(lanes, arg);
        lanes[0] = erff(lanes[0]);
        lanes[1] = erff(lanes[1]);
        lanes[2] = erff(lanes[2]);
        lanes[3] = erff(lanes[3]);
        float32x4_t e = vld1q_f32(lanes);
        // 0.5 * x * (1 + e)
        float32x4_t res = vmulq_f32(vmulq_f32(half, vx), vaddq_f32(one, e));
        vst1q_f32(y + i, res);
    }
    for (; i < n; ++i) {
        y[i] = mimi_gelu_erf_f32(x[i]);
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] = mimi_gelu_erf_f32(x[i]);
    }
#endif
}

// -------- elu sweep: y[i] = elu(x[i], alpha) ----------------------------------
// NEON select between the x>0 and alpha*(exp(x)-1) branches; expf lane-wise.
extern "C" void mimi_elu_vec_f32(const float *x, float *y, int n, float alpha) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float lanes[4];
        vst1q_f32(lanes, vx);
        lanes[0] = expf(lanes[0]);
        lanes[1] = expf(lanes[1]);
        lanes[2] = expf(lanes[2]);
        lanes[3] = expf(lanes[3]);
        float32x4_t ve = vld1q_f32(lanes);
        // alpha * (exp(x) - 1)  for the negative branch
        float32x4_t neg = vmulq_n_f32(vsubq_f32(ve, one), alpha);
        uint32x4_t gt = vcgtq_f32(vx, zero);  // x > 0 mask
        float32x4_t res = vbslq_f32(gt, vx, neg);
        vst1q_f32(y + i, res);
    }
    for (; i < n; ++i) {
        y[i] = mimi_elu_f32(x[i], alpha);
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] = mimi_elu_f32(x[i], alpha);
    }
#endif
}

// -------- add sweep: y[i] = a[i] + b[i] (streaming skip / residual add) --------
extern "C" void mimi_add_vec_f32(const float *a, const float *b, float *y,
                                 int n) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
#endif
}

// -------- scale sweep: y[i] = x[i] * s[i] elementwise (LayerScale) -------------
extern "C" void mimi_scale_vec_f32(const float *x, const float *s, float *y,
                                   int n) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(x + i), vld1q_f32(s + i)));
    }
    for (; i < n; ++i) {
        y[i] = x[i] * s[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * s[i];
    }
#endif
}

// -------- softmax: in place, max-subtracted, f32 sum --------------------------
// NEON max reduction, NEON sum reduction, lane-wise expf.
extern "C" void mimi_softmax_f32(float *x, int n) {
    if (n <= 0) {
        return;
    }
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    // pass 1: max
    float32x4_t vmax = vdupq_n_f32(x[0]);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
    }
    float mx = vmaxvq_f32(vmax);
    for (; i < n; ++i) {
        if (x[i] > mx) {
            mx = x[i];
        }
    }
    // pass 2: e = expf(x - mx) (lane-wise), store, accumulate sum (NEON)
    const float32x4_t vmx = vdupq_n_f32(mx);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmx);
        float lanes[4];
        vst1q_f32(lanes, d);
        lanes[0] = expf(lanes[0]);
        lanes[1] = expf(lanes[1]);
        lanes[2] = expf(lanes[2]);
        lanes[3] = expf(lanes[3]);
        float32x4_t ve = vld1q_f32(lanes);
        vst1q_f32(x + i, ve);
        vsum = vaddq_f32(vsum, ve);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) {
        float e = expf(x[i] - mx);
        x[i] = e;
        sum += e;
    }
    // pass 3: DIVIDE by sum — candle's SoftmaxLastDim CPU kernel does
    // `*d /= sum_exp` per element (ops.rs); reciprocal-multiply rounds
    // differently (two roundings vs one). vdivq_f32 divides per lane with
    // scalar-division rounding. (Arbiter fix: was *= 1/sum.)
    const float32x4_t vsumq = vdupq_n_f32(sum);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(x + i, vdivq_f32(vld1q_f32(x + i), vsumq));
    }
    for (; i < n; ++i) {
        x[i] /= sum;
    }
#else
    float mx = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > mx) {
            mx = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = expf(x[i] - mx);
        x[i] = e;
        sum += e;
    }
    for (int i = 0; i < n; ++i) {
        x[i] /= sum;
    }
#endif
}

// -------- layer norm: ONE-pass sum/sum² (candle's CPU fast kernel) -----------
extern "C" void mimi_layer_norm_f32(const float *x, const float *w,
                                    const float *b, float *y, int n, float eps) {
    // candle_nn::ops::layer_norm CPU kernel (ops.rs LayerNorm::cpu_fwd) —
    // the path that ACTUALLY runs (bias present + contiguous f32); the
    // tensor-op fallback in layer_norm.rs is the slow path and rounds
    // differently. Matched term-for-term (arbiter fix: was two-pass centered):
    //   sum  = Σx ; sum2 = Σ x·x            (ONE pass, unfused mul-then-add)
    //   mean = sum/n ; var = sum2/n − mean² (naive variance, biased)
    //   inv_std = recip(sqrt(var + eps))
    //   y = (x−mean)·inv_std·w + b          (unfused: mul, mul, separate add)
    // NEON lane-blocking of the two sums is the declared faithful-tier
    // freedom; per-element operations stay unfused (rustc never contracts —
    // build with -ffp-contract=off so scalar tails/_ref match too).
    // w/b may be NULL -> treated as 1 / 0 (affine off).
    if (n <= 0) {
        return;
    }
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    // ONE pass: sum and sum-of-squares (unfused: vmul then vadd, no vmla —
    // candle's `sum2 += v*v` rounds the product before accumulating).
    float32x4_t vs = vdupq_n_f32(0.0f);
    float32x4_t vs2 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vs = vaddq_f32(vs, v);
        vs2 = vaddq_f32(vs2, vmulq_f32(v, v));
    }
    float sum = vaddvq_f32(vs);
    float sum2 = vaddvq_f32(vs2);
    for (; i < n; ++i) {
        float v = x[i];
        sum += v;
        float vv = v * v;
        sum2 += vv;
    }
    const float mean = sum / (float)n;
    const float var = sum2 / (float)n - mean * mean;
    const float inv_std = 1.0f / sqrtf(var + eps);
    const float32x4_t vmean = vdupq_n_f32(mean);
    const float32x4_t vinv = vdupq_n_f32(inv_std);
    // apply: y = ((x−mean)·inv_std)·w + b, unfused adds (vaddq, not vmlaq).
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmean);
        float32x4_t normed = vmulq_f32(d, vinv);
        float32x4_t vw = w ? vld1q_f32(w + i) : vdupq_n_f32(1.0f);
        float32x4_t vb = b ? vld1q_f32(b + i) : vdupq_n_f32(0.0f);
        vst1q_f32(y + i, vaddq_f32(vmulq_f32(normed, vw), vb));
    }
    for (; i < n; ++i) {
        float normed = (x[i] - mean) * inv_std;
        float wi = w ? w[i] : 1.0f;
        float bi = b ? b[i] : 0.0f;
        float t = normed * wi;
        y[i] = t + bi;
    }
#else
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        sum += v;
        float vv = v * v;
        sum2 += vv;
    }
    float mean = sum / (float)n;
    float var = sum2 / (float)n - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) {
        float normed = (x[i] - mean) * inv_std;
        float wi = w ? w[i] : 1.0f;
        float bi = b ? b[i] : 0.0f;
        float t = normed * wi;
        y[i] = t + bi;
    }
#endif
}

// ===========================================================================
// (d) Top level — MimiDecoder: owns the arena + the decoder chain
// ===========================================================================

struct MimiDecoder {
    MimiArena arena;

    // Unit states (all carved from `arena`; no per-unit heap ownership).
    MimiQuantState *quant;
    MimiUpsampleState *upsample;
    MimiTransformerState *transformer;
    MimiSeanetState *seanet;

    // Inter-stage latent buffers, conv layout [MIMI_DIM, MIMI_MAX_LATENT].
    // Distinct buffers so transformer's "y == distinct buf" contract holds and
    // no stage aliases its input.
    float *emb_buf;  // quantizer.decode output
    float *up_buf;   // upsample.step output
    float *tr_buf;   // decoder_transformer.step output (seanet input)
    // The final PCM lands directly in the caller's pcm_out (capacity
    // MIMI_FRAME_OUT*2), so no pcm scratch is carved here.
};

extern "C" int mimi_decoder_new(MimiDecoder **d_out, const MimiWeightTable *w,
                                char *err, size_t errlen) {
    if (!d_out) {
        return -1;
    }
    *d_out = NULL;
    if (!w) {
        MIMI_ERR("mimi_decoder_new: null weight table");
        return -1;
    }

    MimiDecoder *d = (MimiDecoder *)calloc(1, sizeof(MimiDecoder));
    if (!d) {
        MIMI_ERR("mimi_decoder_new: OOM allocating decoder struct");
        return -1;
    }

    void *base = aligned_alloc(MIMI_ARENA_ALIGN, MIMI_ARENA_BYTES);
    if (!base) {
        MIMI_ERR("mimi_decoder_new: OOM allocating %zu-byte arena",
                 MIMI_ARENA_BYTES);
        free(d);
        return -1;
    }
    d->arena.base = (uint8_t *)base;
    d->arena.size = MIMI_ARENA_BYTES;
    d->arena.used = 0;

    // Inter-stage buffers first (deterministic base offsets, easy to reason
    // about in a hibernation dump).
    const size_t latent_floats = (size_t)MIMI_DIM * (size_t)MIMI_MAX_LATENT;
    d->emb_buf = (float *)mimi_arena_alloc(&d->arena, latent_floats * sizeof(float));
    d->up_buf = (float *)mimi_arena_alloc(&d->arena, latent_floats * sizeof(float));
    d->tr_buf = (float *)mimi_arena_alloc(&d->arena, latent_floats * sizeof(float));

    // Unit inits in decode order (quant -> upsample -> transformer -> seanet),
    // each carving its own state (and folding weights into the arena as needed)
    // and propagating its error string verbatim on failure.
    int rc = mimi_quant_init(&d->quant, w, &d->arena, err, errlen);
    if (rc) {
        free(base);
        free(d);
        return rc;
    }
    rc = mimi_upsample_init(&d->upsample, w, &d->arena, err, errlen);
    if (rc) {
        free(base);
        free(d);
        return rc;
    }
    rc = mimi_transformer_init(&d->transformer, w, &d->arena, err, errlen);
    if (rc) {
        free(base);
        free(d);
        return rc;
    }
    rc = mimi_seanet_init(&d->seanet, w, &d->arena, err, errlen);
    if (rc) {
        free(base);
        free(d);
        return rc;
    }

    // Assert headroom: if we came within MIMI_ARENA_HEADROOM_MIN of the ceiling
    // the estimate is too tight — fail loudly so the constant gets raised rather
    // than risking a steady-state abort under a differently-sized checkpoint.
    size_t headroom = d->arena.size - d->arena.used;  // used <= size (bump guards)
    if (headroom < MIMI_ARENA_HEADROOM_MIN) {
        MIMI_ERR("mimi_decoder_new: arena headroom %zu < min %zu (used %zu/%zu) — "
                 "raise MIMI_ARENA_BYTES",
                 headroom, MIMI_ARENA_HEADROOM_MIN, d->arena.used, d->arena.size);
        free(base);
        free(d);
        return -2;
    }

    *d_out = d;
    return 0;
}

extern "C" int mimi_decoder_step(MimiDecoder *d, const uint32_t *codes,
                                 float *pcm_out) {
    // Faithful port of Mimi::decode_step (mimi.rs:214). In our streaming path
    // (audio_out.rs) `codes` is always a present single latent frame, so the
    // Rust `codes.as_option()` is always Some and the quantizer always emits one
    // embedding frame. Each stage is still invoked with the previous stage's
    // reported count, so a 0 (priming) propagates 0 onward exactly as
    // StreamTensor::empty() does through the Rust `.step` chain.
    if (!d || !codes || !pcm_out) {
        return 0;
    }

    // quantizer.decode: codes[MIMI_NQ] -> emb[MIMI_DIM, 1]. Pure per-frame RVQ
    // dequantize (no streaming state), so exactly 1 frame out.
    mimi_quant_decode(d->quant, codes, d->emb_buf);
    const int n_emb = 1;
    // --- fence F0 (post-quant): emb_buf[MIMI_DIM, n_emb], see NOTES (f) ---

    // upsample.step: [MIMI_DIM, n_emb] -> [MIMI_DIM, n_up]. Stride 2 => 2 frames
    // out per input frame in steady state (n_up == 2).
    int n_up = mimi_upsample_step(d->upsample, d->emb_buf, n_emb, d->up_buf);
    // --- fence F1 (post-upsample): up_buf[MIMI_DIM, n_up] ---

    // decoder_transformer.step: [MIMI_DIM, n_up] -> [MIMI_DIM, n_tr]. Causal KV
    // transformer preserves the frame count (n_tr == n_up). Intra: per-layer F2.
    int n_tr = mimi_transformer_step(d->transformer, d->up_buf, n_up, d->tr_buf);
    // --- fence F2 (post-transformer): tr_buf[MIMI_DIM, n_tr] ---

    // decoder(seanet).step: [MIMI_DIM, n_tr] -> pcm[1, n_pcm]. x960 upsample =>
    // n_pcm == n_tr * 960 in steady state (== MIMI_FRAME_OUT for n_tr == 2).
    // Intra: per {upsample+resnet} layer F3.
    int n_pcm = mimi_seanet_step(d->seanet, d->tr_buf, n_tr, pcm_out);
    // --- fence F4 (post-seanet): pcm_out[1, n_pcm] — pass-boundary doorbell ---

    return n_pcm;
}

extern "C" void mimi_decoder_reset(MimiDecoder *d) {
    // Decoder-half of Mimi::reset_state (mimi.rs:224). The Rust also resets the
    // encoder, encoder_transformer, and downsample — all out of scope here. The
    // quantizer has no streaming state (RVQ decode is stateless), so it has no
    // reset entry point. Order mirrors the Rust (decoder, transformer, upsample).
    if (!d) {
        return;
    }
    mimi_seanet_reset(d->seanet);
    mimi_transformer_reset(d->transformer);
    mimi_upsample_reset(d->upsample);
}

extern "C" void mimi_decoder_free(MimiDecoder *d) {
    if (!d) {
        return;
    }
    // Every unit state and folded weight lives in the single arena, so one free
    // reclaims all of it; the units own no separate heap.
    free(d->arena.base);
    free(d);
}

/* NOTES
 * ============================================================================
 * FAITHFUL PORT NOTES — mimi_decode.cpp (Unit #6), moshi 0.6.4
 * ============================================================================
 *
 * (a) RUST -> C++ MAPPING
 * ------------------------
 *  Rust (moshi 0.6.4)                         | C++ (this file)
 *  -------------------------------------------|--------------------------------
 *  Mimi::decode_step (mimi.rs:214)            | mimi_decoder_step
 *    codes.as_option()/quantizer.decode       |   mimi_quant_decode (always 1 fr)
 *    upsample.step (ConvTrUpsample1d)         |   mimi_upsample_step
 *    decoder_transformer.step (Projected...)  |   mimi_transformer_step
 *    decoder.step (SeaNetDecoder)             |   mimi_seanet_step
 *  Mimi::reset_state, decoder half (mimi.rs:224)| mimi_decoder_reset
 *    decoder / decoder_transformer / upsample |   seanet / transformer / upsample
 *    (encoder, encoder_transformer, downsample|   SKIPPED — encoder-side / OOS)
 *  Mimi::new_ construction order              | mimi_decoder_new (unit inits)
 *  StreamTensor(Option<Tensor>) (streaming.rs)| explicit int n_in/n_out counts
 *  StreamTensor::empty() propagation          | a stage handed 0 returns 0
 *  candle weight-norm fold (conv.rs:27,133)   | units fold into arena at init
 *  candle matmul / linear                     | mimi_gemm_f32 / mimi_gemv_f32 (AMX cblas)
 *  candle_nn::LayerNorm                       | mimi_layer_norm_f32 (NEON)
 *  candle gelu_erf / Elu / softmax_last_dim   | mimi_gelu_erf_vec_f32 / _elu_vec_ / _softmax_
 *  candle add (residual/skip) / LayerScale mul| mimi_add_vec_f32 / mimi_scale_vec_f32
 *  candle gelu_erf / Elu (per element)        | mimi_gelu_erf_f32 / mimi_elu_f32 (lane/tail)
 *
 *  The Rust ALWAYS calls each `.step` regardless of emptiness; this file mirrors
 *  that by always invoking each stage with the prior stage's count. `StreamMask`
 *  is empty on our path (audio_out.rs passes StreamMask::empty()), so every
 *  mask-conditioned branch in the Rust `.step`s is the None arm — nothing to
 *  port here; the units own their own (mask-free) state carry.
 *
 * (b) FRAME-COUNT CONTRACT (in -> out), verified against the Rust `.step`s
 * ------------------------------------------------------------------------
 *  Stage        | in (frames) | out                 | basis
 *  -------------|-------------|---------------------|-----------------------------
 *  quantizer    | 1 code fr   | 1 emb fr [DIM,1]    | pure RVQ decode, stateless;
 *               |             |                     | codes always present here
 *  upsample     | n           | 2n (n=1 -> 2)       | StreamableConvTranspose1d::
 *               |             |                     | step, stride 2 kernel 4:
 *               |             |                     | ot=(n)*2+2, emits ot-invalid,
 *               |             |                     | invalid=k-s=2 => 2 per frame,
 *               |             |                     | from frame 1 (no priming)
 *  transformer  | n           | n                   | StreamingTransformer::step:
 *               |             |                     | None->empty else forward,
 *               |             |                     | forward preserves T (25 Hz)
 *  seanet       | n           | n*960 (n=2 -> 1920) | SeaNetDecoder::step: ratios
 *               |             |                     | 8*6*5*4; each stream conv is
 *               |             |                     | self-priming via left-pad
 *  end-to-end   | 1 latent    | 1920 (=MIMI_FRAME_OUT)| steady state
 *
 *  PRIMING FINDING (per the "verify against the Rust" gotcha): for Config
 *  v0_1(8) with one code frame per call, EVERY stage is self-priming — the
 *  transpose convs emit from step 1 (invalid tail is *carried*, not withheld)
 *  and the stride-1 convs left-pad on their first step. So Mimi's FIRST
 *  decode_step already emits a full 1920 samples; there is no 0-output warm-up
 *  in this config. The consumer's "first call(s) may yield None" (audio_out.rs)
 *  is the generic streaming-codec disclaimer, not a v0_1(8) behavior. I still
 *  plumb the 0-propagation faithfully (mimi_decoder_step returns whatever
 *  mimi_seanet_step reports, 0 legal) so the contract holds if a unit author or
 *  the parity harness feeds partial frames, and because the header mandates it.
 *
 *  I do NOT hardcode 1920: mimi_decoder_step returns the seanet's reported
 *  n_out, and feeds each stage the prior stage's actual count. This is robust to
 *  whatever emit schedule units 2–3 land on, and is why the buffers are sized
 *  for the doubled worst case rather than the steady value.
 *
 * (c) ARENA SIZING ESTIMATE (MIMI_ARENA_BYTES = 256 MiB)
 * ------------------------------------------------------
 *  The arena holds ALL mutable state + scratch + any folded/re-armed weights
 *  (per the header discipline "weight-norm folds ONCE at init into the arena").
 *  Worst-case breakdown (checkpoint stores weight_g/weight_v => convs folded):
 *    - SEANET decoder folded conv weights .............. ~59 MiB
 *        init_conv 1024x512x7 (14.7M f) dominant, + 4 transpose upsample convs
 *        (ratios 8/6/5/4) + 4 resnet blocks (k3,k1) + final_conv; ~14.7M floats.
 *    - Transformer, IF a unit re-arms GEMM weights into arena  ~100 MiB
 *        8 layers x (in_proj 3*512*512 + out_proj 512*512 + mlp 2*512*2048)
 *        = ~25.2M floats. If instead used zero-copy (cblas can take checkpoint
 *        layout), this term is 0.
 *    - Transformer KV cache ............................ ~8.2 MiB
 *        2(k,v) x 8 layers x 250 context x 512 = ~2.05M floats.
 *    - SEANET streaming conv left-context state ........ < 1 MiB
 *        sum over convs of (kernel_eff-1)*channels.
 *    - Quant scratch + in/out projections .............. < 1 MiB
 *        codebook embeddings are zero-copy weights (not arena).
 *    - Inter-stage latent buffers ...................... 24 KiB
 *        3 x MIMI_DIM x MIMI_MAX_LATENT x 4B = 3*512*4*4.
 *  Worst-worst total ~170 MiB; 256 MiB gives >80 MiB slack, checked by the
 *  MIMI_ARENA_HEADROOM_MIN (1 MiB) assertion after init. NOTE for the arbiter:
 *  the header example (64 MiB) is INSUFFICIENT if seanet convs fold into the
 *  arena AND the transformer re-arms; it is ample if the checkpoint stores
 *  pre-folded "weight" tensors (zero-copy) and the transformer stays zero-copy
 *  (true need ~16 MiB). Tighten once those two decisions lock. The bump
 *  allocator aborts on overflow, so an undersized constant fails at init.
 *
 * (d) PRIMITIVE-KERNEL LOOP ORDERS  ("math is assembly at every step")
 * --------------------------------------------------------------------
 *  gemv (y[M] = W[M,K]*x + b): Accelerate cblas_sgemv(CblasRowMajor,
 *    CblasNoTrans, m, k, 1, W, lda=k, x, 1, beta=0, y, 1) — routes to the AMX
 *    coprocessor. No transpose (row-major maps 1:1; movement is theft). Bias is
 *    a SEPARATE explicit loop after the call (AMX carries no bias term). Scalar
 *    `_ref` (outer m / inner k, sequential accumulate) is the parity path.
 *  gemm (C[M,N] = A[M,K]*B[K,N], beta 0=overwrite / 1=accumulate): cblas_sgemm(
 *    CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, lda=k, B, ldb=n,
 *    beta, C, ldc=n) — AMX, direct row-major, no transpose copies. Scalar `_ref`
 *    is i-k-j. Off-Apple or -DMIMI_SCALAR_REF selects the _ref siblings.
 *  softmax (in place): NEON. pass 1 vmaxq + vmaxvq max reduction; pass 2
 *    expf(x-max) LANE-WISE with a vaddq/vaddvq f32 sum reduction; pass 3 NEON
 *    multiply by 1/sum. Max-subtracted for stability.
 *  gelu sweep (mimi_gelu_erf_vec_f32): NEON 0.5*x*(1+e); erff applied lane-wise
 *    (store arg, 4x erff, reload) — NO vector-poly. Tail via mimi_gelu_erf_f32.
 *  elu sweep (mimi_elu_vec_f32): NEON vbslq select on (x>0); expf lane-wise on
 *    the negative branch; alpha via vmulq_n_f32. Tail via mimi_elu_f32.
 *  add sweep (mimi_add_vec_f32): NEON vaddq, scalar tail.
 *  scale sweep (mimi_scale_vec_f32): NEON vmulq, elementwise s[i] (LayerScale),
 *    scalar tail.
 *  gelu_erf (scalar/lane): 0.5*x*(1+erff(x*(1/sqrt2))), 1/sqrt2=0.7071067811865.
 *  elu (scalar/lane): x>0 ? x : alpha*(expf(x)-1).
 *  layer_norm: NEON two-pass. pass 1 mean=sum/n (vaddq/vaddvq); pass 2
 *    var=sum((x-mean)^2)/n via vmlaq (BIASED, /n — matches candle_nn::LayerNorm);
 *    pass 3 y=(x-mean)/sqrt(var+eps)*w+b via vmlaq(vb, normed, vw), eps added to
 *    var BEFORE sqrt, eps supplied by the caller. NULL w/b => 1/0.
 *  LINKING: gemm/gemv need `-framework Accelerate`. Compiling this .cpp does NOT
 *    (cblas is header-only decls); the arbiter's build.rs adds the framework.
 *  Every scalar body above is gated to the _ref/-DMIMI_SCALAR_REF path or a
 *  sub-vector tail; the hot path is cblas (AMX) or NEON only.
 *
 * (e) UNCERTAINTIES / ARBITER RECONCILIATION
 * ------------------------------------------
 *  1. Arena ceiling vs the header's 64 MiB example — see (c). Depends on whether
 *     the checkpoint stores folded "weight" (zero-copy) or weight_g/weight_v
 *     (fold into arena), and whether the transformer re-arms into the arena.
 *     I chose 256 MiB generous; arbiter tightens.
 *  2. Step return convention. I read every *_step's int return as n_out (frames
 *     for upsample/transformer, samples for seanet), per the header's "reports
 *     n_out frames; 0 is legal" and conv's "returns n_out (>=0)". If a unit ever
 *     returns a negative error code from a step, this file passes it through
 *     unchanged (no steady-state error channel is defined). Please keep steps
 *     infallible (>=0) or the arbiter must add an error convention.
 *  3. MIMI_MAX_LATENT = 4 (not the steady 2). Sized to the header's pcm_out
 *     capacity MIMI_FRAME_OUT*2 = 4*960 "drain headroom", so a hypothetical
 *     double-emit step cannot overflow the latent buffers or pcm_out. If the
 *     arbiter proves emit is bounded at 2, this can drop to 2 (buffers shrink,
 *     pcm capacity stays per header).
 *  4. Accumulation order for parity. gemm/gemv run on AMX via cblas (its
 *     internal reduction order is opaque and differs from candle's blocked gemm
 *     — the manifest already accepts this: "candle's blocked gemm is not
 *     economically bit-reproducible", ulp-band tier). NEON reductions
 *     (softmax/layernorm lane-sums) also differ from a strict sequential sum.
 *     Bisect with -DMIMI_SCALAR_REF (forces scalar gemm/gemv _ref + scalar
 *     sweep bodies). Transcendentals are libm erff/expf lane-wise (NOT vector
 *     polynomials) to stay on the faithful tier. Softmax is max-subtracted
 *     3-pass matching candle's softmax_last_dim formula.
 *  5. Quantizer emptiness. mimi_decoder_step assumes `codes` is a present single
 *     frame (true on our path). There is no ABI way to pass "empty codes" to
 *     mimi_quant_decode, so the None-codes arm of Rust decode_step is not
 *     reachable here; if a future caller needs it, add an n_in to the quant ABI.
 *  6. reset ordering. Reset order among decoder/transformer/upsample is
 *     immaterial (independent state) but mirrors the Rust for auditability.
 *
 * (f) LANE-TEAM INTEGRATION MAP (kcoro engine, native C++ lane program)
 * --------------------------------------------------------------------
 *  This kernel is meant to ride the same kcoro lane team as the backbone /
 *  depthformer, so the arbiter needs to know where the natural fences fall if
 *  each stage becomes a lane-team stage under its own REQ kind.
 *
 *  Stateless / sub-range safe: the sweep + reduction primitives
 *  (mimi_add_vec_f32, mimi_scale_vec_f32, mimi_elu_vec_f32,
 *  mimi_gelu_erf_vec_f32, and the mimi_gemm/gemv cblas calls) hold NO global or
 *  static state — they operate purely on (pointer, length). So the integration
 *  layer may BAND any of them across lanes by slicing [base+off, len] per lane
 *  with no cross-lane hazard. The NEON idiom is the house one: full float32x4_t
 *  register chunks + scalar tail, so a band boundary that isn't 4-aligned still
 *  computes correctly (each lane runs its own tail).
 *
 *  Fence points inside mimi_decoder_step (doorbell / hand-back boundaries),
 *  in execution order:
 *    F0  post-quant     : after mimi_quant_decode -> emb_buf[MIMI_DIM, 1].
 *                         RVQ decode is embarrassingly bandable over the 8
 *                         codebooks (accumulate into emb); fence before upsample.
 *    F1  post-upsample  : after mimi_upsample_step -> up_buf[MIMI_DIM, n_up].
 *                         Frame count expands 1->2 here; fence carries n_up.
 *    F2  per-transformer-layer : the transformer is 8 residual layers; the
 *                         natural intra-stage fences are per layer (attn + MLP),
 *                         matching the depthformer's per-layer doorbell cadence.
 *                         Coarser option: one fence F2 before / after the whole
 *                         transformer (post-upsample -> post-transformer).
 *    F3  per-seanet-layer : the SEANET decoder is init_conv, then 4 {upsample +
 *                         resnet} layers (ratios 8/6/5/4), then final_conv. Each
 *                         {upsample, resnet} pair is a natural fence; the frame
 *                         count multiplies by the ratio at each, so a lane band
 *                         re-widens per stage (2 -> 16 -> 96 -> 480 -> 1920).
 *    F4  post-seanet    : pcm_out[1, n_pcm]; hand back the sample count. This is
 *                         the pass-boundary doorbell (one latent frame consumed,
 *                         n_pcm samples produced) — the engine's REQ retire point.
 *  The COUNTS contract in (b) is what each fence must carry (frames on the
 *  latent side, samples after F4). Cross-frame state (conv left-context, KV
 *  ring) lives in the arena and is owned by the unit between passes — a fence is
 *  a data hand-off, never a state copy.
 * ============================================================================
 */
