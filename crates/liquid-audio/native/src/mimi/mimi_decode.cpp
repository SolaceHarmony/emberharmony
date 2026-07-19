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
#include "lfm_safetensors.h"

#include <cerrno>
#include <cmath>    // erff, expf, sqrtf
#include <cstdio>   // snprintf, fprintf, stderr
#include <cstdlib>  // aligned_alloc, free, abort
#include <cstring>  // strcmp, memset

// GEMM/GEMV whose operands are mutable activation storage may run on Apple's
// AMX matrix coprocessor via Accelerate's cblas. Resident weights never cross
// that typed API: their variants load checkpoint bytes into NEON/SSE registers.
// The header is include-only at compile time; linking requires Accelerate.
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
#elif defined(__x86_64__) || defined(_M_X64)
#define MIMI_HAVE_SSE2 1
#include <immintrin.h>
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

// Plan construction uses generous temporary ceilings, records the exact mutable
// state and formula-derived footprints, then replaces both probes with their
// exact published allocations. Neither ceiling is retained by a live model.
static const size_t MIMI_STATE_ARENA_MAX = (size_t)256 * 1024 * 1024;
static const size_t MIMI_DERIVED_ARENA_MAX = (size_t)128 * 1024 * 1024;
static const size_t MIMI_ARENA_ALIGN = 64;
static const size_t MIMI_ARENA_HEADROOM_MIN = (size_t)1 * 1024 * 1024;

static bool mimi_align_checked(size_t bytes, size_t *out) {
    if (!out || bytes > SIZE_MAX - (MIMI_ARENA_ALIGN - 1)) return false;
    *out = (bytes + (MIMI_ARENA_ALIGN - 1)) & ~(MIMI_ARENA_ALIGN - 1);
    return true;
}

struct MimiDerivedArena {
    uint8_t *base;
    size_t size;
    size_t used;
    int sealed;
};

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
                "sizing bug, raise the state probe ceiling\n",
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

extern "C" void *mimi_arena_alloc_derived(MimiArena *a, size_t bytes) {
    MimiDerivedArena *derived = a ? a->derived : nullptr;
    if (!derived) {
        fprintf(stderr, "mimi_arena_alloc_derived: no plan arena\n");
        abort();
    }
    const size_t off = (a->derived_cursor + (MIMI_ARENA_ALIGN - 1)) &
                       ~(MIMI_ARENA_ALIGN - 1);
    if (off > derived->size || bytes > derived->size - off) {
        fprintf(stderr,
                "mimi_arena_alloc_derived: overflow (cursor=%zu req=%zu size=%zu)\n",
                a->derived_cursor, bytes, derived->size);
        abort();
    }
    const size_t end = off + bytes;
    if (derived->sealed) {
        if (end > derived->used) {
            fprintf(stderr, "mimi_arena_alloc_derived: replay exceeds sealed plan\n");
            abort();
        }
    } else {
        std::memset(derived->base + off, 0, bytes);
        derived->used = end;
    }
    a->derived_cursor = end;
    return derived->base + off;
}

extern "C" int mimi_arena_building_derived(const MimiArena *a) {
    return a && a->derived && !a->derived->sealed;
}

extern "C" float mimi_weight_load_f32(const uint8_t *bytes, uint64_t index) {
    const uint8_t *p = bytes + index * 4;
    const uint32_t bits = static_cast<uint32_t>(p[0]) |
                          (static_cast<uint32_t>(p[1]) << 8) |
                          (static_cast<uint32_t>(p[2]) << 16) |
                          (static_cast<uint32_t>(p[3]) << 24);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Resident checkpoint storage remains byte-addressed even when its address
// happens to satisfy float alignment. Architecture loads consume bytes and
// reinterpret only the register bits; scalar tails assemble little-endian u32
// explicitly in mimi_weight_load_f32. No C++ float object is ever manufactured
// inside the sealed image.
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
static inline float32x4_t mimi_weight_load4_f32(const uint8_t *bytes) {
    return vreinterpretq_f32_u8(vld1q_u8(bytes));
}

static inline float mimi_weight_sum4(float32x4_t values) {
    return vaddvq_f32(values);
}
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
static inline __m128 mimi_weight_load4_f32(const uint8_t *bytes) {
    __m128 values;
    std::memcpy(&values, bytes, sizeof(values));
    return values;
}

static inline float mimi_weight_sum4(__m128 values) {
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, values);
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
#endif

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
            if (t->bound) t->bound[i] = 1;
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
//   - activation GEMM/GEMV: Apple Accelerate cblas (AMX matrix coprocessor).
//   - resident-weight GEMM/GEMV: byte-load NEON/SSE, scalar LE tails.
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

static inline float mimi_weight_gemv_row_f32(const uint8_t *w,
                                              const float *x,
                                              const uint8_t *bias, int i,
                                              int k) {
    float sum = 0.0f;
    int j = 0;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    const uint8_t *row = w + static_cast<size_t>(i) * k * sizeof(float);
    for (; j + 16 <= k; j += 16) {
        acc0 = vaddq_f32(
            acc0, vmulq_f32(mimi_weight_load4_f32(row + j * sizeof(float)),
                            vld1q_f32(x + j)));
        acc1 = vaddq_f32(
            acc1, vmulq_f32(mimi_weight_load4_f32(row + (j + 4) * sizeof(float)),
                            vld1q_f32(x + j + 4)));
        acc2 = vaddq_f32(
            acc2, vmulq_f32(mimi_weight_load4_f32(row + (j + 8) * sizeof(float)),
                            vld1q_f32(x + j + 8)));
        acc3 = vaddq_f32(
            acc3, vmulq_f32(mimi_weight_load4_f32(row + (j + 12) * sizeof(float)),
                            vld1q_f32(x + j + 12)));
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    sum = mimi_weight_sum4(vaddq_f32(acc0, acc2));
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    __m128 acc2 = _mm_setzero_ps();
    __m128 acc3 = _mm_setzero_ps();
    const uint8_t *row = w + static_cast<size_t>(i) * k * sizeof(float);
    for (; j + 16 <= k; j += 16) {
        acc0 = _mm_add_ps(
            acc0, _mm_mul_ps(mimi_weight_load4_f32(row + j * sizeof(float)),
                             _mm_loadu_ps(x + j)));
        acc1 = _mm_add_ps(
            acc1, _mm_mul_ps(mimi_weight_load4_f32(row + (j + 4) * sizeof(float)),
                             _mm_loadu_ps(x + j + 4)));
        acc2 = _mm_add_ps(
            acc2, _mm_mul_ps(mimi_weight_load4_f32(row + (j + 8) * sizeof(float)),
                             _mm_loadu_ps(x + j + 8)));
        acc3 = _mm_add_ps(
            acc3, _mm_mul_ps(mimi_weight_load4_f32(row + (j + 12) * sizeof(float)),
                             _mm_loadu_ps(x + j + 12)));
    }
    acc0 = _mm_add_ps(acc0, acc1);
    acc2 = _mm_add_ps(acc2, acc3);
    sum = mimi_weight_sum4(_mm_add_ps(acc0, acc2));
#endif
    for (; j < k; ++j) {
        sum += mimi_weight_load_f32(w, static_cast<uint64_t>(i) * k + j) * x[j];
    }
    if (bias) sum += mimi_weight_load_f32(bias, i);
    return sum;
}

static void mimi_weight_gemv_range_f32(
    const uint8_t *w, const float *x, const uint8_t *bias, float *y,
    int row_begin, int row_end, int output_begin, int k, int accumulate) {
    for (int i = row_begin; i < row_end; ++i) {
        const float sum = mimi_weight_gemv_row_f32(w, x, bias, i, k);
        float *out = y + (i - output_begin);
        if (accumulate) {
            *out += sum;
        } else {
            *out = sum;
        }
    }
}

extern "C" void mimi_weight_gemv_rows_f32(
    const uint8_t *w, const float *x, const uint8_t *bias, float *y,
    int row_begin, int row_end, int k, int accumulate) {
    mimi_weight_gemv_range_f32(w, x, bias, y, row_begin, row_end,
                               /*output_begin*/ 0, k, accumulate);
}

extern "C" void mimi_weight_gemv_span_f32(
    const uint8_t *w, const float *x, const uint8_t *bias, float *y,
    int row_begin, int row_end, int k) {
    mimi_weight_gemv_range_f32(w, x, bias, y, row_begin, row_end,
                               /*output_begin*/ row_begin, k,
                               /*accumulate*/ 0);
}

extern "C" void mimi_weight_gemv_scale_residual_rows_f32(
    const uint8_t *w, const float *x, const uint8_t *scale, float *residual,
    int row_begin, int row_end, int k) {
    for (int i = row_begin; i < row_end; ++i) {
        const float sum = mimi_weight_gemv_row_f32(w, x, nullptr, i, k);
        const float scaled = sum * mimi_weight_load_f32(scale, i);
        const float prior = residual[i];
        residual[i] = prior + scaled;
    }
}

extern "C" void mimi_weight_gemv_f32(const uint8_t *w, const float *x,
                                       const uint8_t *bias, float *y, int m,
                                       int k) {
    mimi_weight_gemv_rows_f32(w, x, bias, y, 0, m, k, 0);
}

extern "C" void mimi_weight_gemm_f32(const uint8_t *w, const float *b,
                                       float *c, int m, int k, int n,
                                       int beta) {
    if (n == 1 && !beta) {
        mimi_weight_gemv_f32(w, b, nullptr, c, m, k);
        return;
    }
    for (int i = 0; i < m; ++i) {
        float *row = c + static_cast<size_t>(i) * n;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
        if (n == 2) {
            float32x2_t acc = beta ? vld1_f32(row) : vdup_n_f32(0.0f);
            for (int p = 0; p < k; ++p) {
                const float weight =
                    mimi_weight_load_f32(w, static_cast<uint64_t>(i) * k + p);
                acc = vadd_f32(acc, vmul_n_f32(vld1_f32(b + static_cast<size_t>(p) * 2),
                                               weight));
            }
            vst1_f32(row, acc);
            continue;
        }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
        if (n == 2) {
            __m128 acc = beta ? _mm_setr_ps(row[0], row[1], 0.0f, 0.0f)
                              : _mm_setzero_ps();
            for (int p = 0; p < k; ++p) {
                const float weight =
                    mimi_weight_load_f32(w, static_cast<uint64_t>(i) * k + p);
                const float *input = b + static_cast<size_t>(p) * 2;
                acc = _mm_add_ps(
                    acc, _mm_mul_ps(_mm_set1_ps(weight),
                                    _mm_setr_ps(input[0], input[1], 0.0f, 0.0f)));
            }
            alignas(16) float lanes[4];
            _mm_store_ps(lanes, acc);
            row[0] = lanes[0];
            row[1] = lanes[1];
            continue;
        }
#endif
        if (!beta) std::memset(row, 0, static_cast<size_t>(n) * sizeof(float));
        for (int p = 0; p < k; ++p) {
            const float weight =
                mimi_weight_load_f32(w, static_cast<uint64_t>(i) * k + p);
            const float *input = b + static_cast<size_t>(p) * n;
            int j = 0;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
            const float32x4_t vw = vdupq_n_f32(weight);
            for (; j + 4 <= n; j += 4) {
                vst1q_f32(row + j,
                          vaddq_f32(vld1q_f32(row + j),
                                    vmulq_f32(vw, vld1q_f32(input + j))));
            }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
            const __m128 vw = _mm_set1_ps(weight);
            for (; j + 4 <= n; j += 4) {
                _mm_storeu_ps(row + j,
                              _mm_add_ps(_mm_loadu_ps(row + j),
                                         _mm_mul_ps(vw, _mm_loadu_ps(input + j))));
            }
#endif
            for (; j < n; ++j) row[j] += weight * input[j];
        }
    }
}

extern "C" void mimi_weight_gemm_tn_f32(const uint8_t *w, const float *b,
                                          float *c, int rows, int k, int n) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    // Decode's wide transposed-convolution route has n==2. Vectorize across
    // four checkpoint rows so every resident load stays contiguous even though
    // the row-major output columns are short.
    if (n > 0 && n < 4) {
        int row = 0;
        for (; row + 4 <= rows; row += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            for (int p = 0; p < k; ++p) {
                const float32x4_t weights = mimi_weight_load4_f32(
                    w + (static_cast<size_t>(p) * rows + row) * sizeof(float));
                const float *input = b + static_cast<size_t>(p) * n;
                acc0 = vaddq_f32(acc0, vmulq_n_f32(weights, input[0]));
                if (n > 1) acc1 = vaddq_f32(acc1, vmulq_n_f32(weights, input[1]));
                if (n > 2) acc2 = vaddq_f32(acc2, vmulq_n_f32(weights, input[2]));
            }
            alignas(16) float lanes0[4], lanes1[4], lanes2[4];
            vst1q_f32(lanes0, acc0);
            if (n > 1) vst1q_f32(lanes1, acc1);
            if (n > 2) vst1q_f32(lanes2, acc2);
            for (int lane = 0; lane < 4; ++lane) {
                float *output = c + static_cast<size_t>(row + lane) * n;
                output[0] = lanes0[lane];
                if (n > 1) output[1] = lanes1[lane];
                if (n > 2) output[2] = lanes2[lane];
            }
        }
        for (; row < rows; ++row) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < k; ++p) {
                    sum += mimi_weight_load_f32(
                               w, static_cast<uint64_t>(p) * rows + row) *
                           b[static_cast<size_t>(p) * n + j];
                }
                c[static_cast<size_t>(row) * n + j] = sum;
            }
        }
        return;
    }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
    if (n > 0 && n < 4) {
        int row = 0;
        for (; row + 4 <= rows; row += 4) {
            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            for (int p = 0; p < k; ++p) {
                const __m128 weights = mimi_weight_load4_f32(
                    w + (static_cast<size_t>(p) * rows + row) * sizeof(float));
                const float *input = b + static_cast<size_t>(p) * n;
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(weights, _mm_set1_ps(input[0])));
                if (n > 1)
                    acc1 = _mm_add_ps(acc1, _mm_mul_ps(weights, _mm_set1_ps(input[1])));
                if (n > 2)
                    acc2 = _mm_add_ps(acc2, _mm_mul_ps(weights, _mm_set1_ps(input[2])));
            }
            alignas(16) float lanes0[4], lanes1[4], lanes2[4];
            _mm_store_ps(lanes0, acc0);
            if (n > 1) _mm_store_ps(lanes1, acc1);
            if (n > 2) _mm_store_ps(lanes2, acc2);
            for (int lane = 0; lane < 4; ++lane) {
                float *output = c + static_cast<size_t>(row + lane) * n;
                output[0] = lanes0[lane];
                if (n > 1) output[1] = lanes1[lane];
                if (n > 2) output[2] = lanes2[lane];
            }
        }
        for (; row < rows; ++row) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < k; ++p) {
                    sum += mimi_weight_load_f32(
                               w, static_cast<uint64_t>(p) * rows + row) *
                           b[static_cast<size_t>(p) * n + j];
                }
                c[static_cast<size_t>(row) * n + j] = sum;
            }
        }
        return;
    }
#endif
    for (int row = 0; row < rows; ++row) {
        float *output = c + static_cast<size_t>(row) * n;
        std::memset(output, 0, static_cast<size_t>(n) * sizeof(float));
        for (int p = 0; p < k; ++p) {
            const float weight = mimi_weight_load_f32(
                w, static_cast<uint64_t>(p) * rows + row);
            const float *input = b + static_cast<size_t>(p) * n;
            int j = 0;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
            const float32x4_t vw = vdupq_n_f32(weight);
            for (; j + 4 <= n; j += 4) {
                vst1q_f32(output + j,
                          vaddq_f32(vld1q_f32(output + j),
                                    vmulq_f32(vw, vld1q_f32(input + j))));
            }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
            const __m128 vw = _mm_set1_ps(weight);
            for (; j + 4 <= n; j += 4) {
                _mm_storeu_ps(output + j,
                              _mm_add_ps(_mm_loadu_ps(output + j),
                                         _mm_mul_ps(vw, _mm_loadu_ps(input + j))));
            }
#endif
            for (; j < n; ++j) output[j] += weight * input[j];
        }
    }
}

// -------- scalar per-element helpers (tail/lane + _ref building blocks) -------

/* ---- Rust-libm verbatim ports (bit-parity with candle) ----------------------
 * candle's gelu_erf (candle-core op.rs GeluErf::f32) calls
 * crate::cpu::erf::erf_f32 == libm::erff — the RUST libm crate (FreeBSD msun
 * s_erff.c float port), NOT the system libm. Apple's erff is a different
 * polynomial; using it is a systematic ulp bias at every activation. erff's
 * erfc2 branch internally calls libm::expf (e_expf.c) — ported too, with its
 * scalbnf. Verbatim structure and constants; do not "improve".
 *
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ==================================================== */
static inline float rl_scalbnf(float x, int n) {
    /* musl scalbnf semantics (libm generic/scalbn.rs reduces to this for f32) */
    if (n > 127) {
        x *= 0x1p127f;
        n -= 127;
        if (n > 127) {
            x *= 0x1p127f;
            n -= 127;
            if (n > 127) n = 127;
        }
    } else if (n < -126) {
        x *= 0x1p-126f * 0x1p24f;
        n += 126 - 24;
        if (n < -126) {
            x *= 0x1p-126f * 0x1p24f;
            n += 126 - 24;
            if (n < -126) n = -126;
        }
    }
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)(0x7f + n) << 23;
    return x * u.f;
}

static float rl_expf(float x) {
    /* libm 0.2.16 expf (FreeBSD e_expf.c) */
    static const float half_arr[2] = {0.5f, -0.5f};
    const float ln2_hi = 6.9314575195e-01f;  /* 0x3f317200 */
    const float ln2_lo = 1.4286067653e-06f;  /* 0x35bfbe8e */
    const float inv_ln2 = 1.4426950216e+00f; /* 0x3fb8aa3b */
    const float p1 = 1.6666625440e-1f;
    const float p2 = -2.7667332906e-3f;

    union { float f; uint32_t i; } ux;
    ux.f = x;
    uint32_t hx = ux.i;
    int sign = (int)(hx >> 31);
    hx &= 0x7fffffff;

    if (hx >= 0x42aeac50) { /* |x| >= -87.33655f or NaN */
        if (hx > 0x7f800000) return x; /* NaN */
        if (hx >= 0x42b17218 && !sign) { /* x >= 88.722839f: overflow */
            return x * 0x1p127f;
        }
        if (sign && hx >= 0x42cff1b5) { /* x <= -103.972084f: underflow to 0 */
            return 0.0f;
        }
    }

    int k;
    float hi, lo;
    if (hx > 0x3eb17218) { /* |x| > 0.5 ln2 */
        if (hx > 0x3f851592) { /* |x| > 1.5 ln2 */
            k = (int)(inv_ln2 * x + half_arr[sign]);
        } else {
            k = 1 - sign - sign;
        }
        float kf = (float)k;
        hi = x - kf * ln2_hi; /* k*ln2hi is exact here */
        lo = kf * ln2_lo;
        x = hi - lo;
    } else if (hx > 0x39000000) { /* |x| > 2**-14 */
        k = 0;
        hi = x;
        lo = 0.0f;
    } else {
        return 1.0f + x;
    }

    float xx = x * x;
    float c = x - xx * (p1 + xx * p2);
    float y = 1.0f + (x * c / (2.0f - c) - lo + hi);
    return (k == 0) ? y : rl_scalbnf(y, k);
}

/* erff coefficients (s_erff.c) */
static const float RL_ERX = 8.4506291151e-01f;
static const float RL_EFX8 = 1.0270333290e+00f;
static const float RL_PP0 = 1.2837916613e-01f, RL_PP1 = -3.2504209876e-01f,
                   RL_PP2 = -2.8481749818e-02f, RL_PP3 = -5.7702702470e-03f,
                   RL_PP4 = -2.3763017452e-05f;
static const float RL_QQ1 = 3.9791721106e-01f, RL_QQ2 = 6.5022252500e-02f,
                   RL_QQ3 = 5.0813062117e-03f, RL_QQ4 = 1.3249473704e-04f,
                   RL_QQ5 = -3.9602282413e-06f;
static const float RL_PA0 = -2.3621185683e-03f, RL_PA1 = 4.1485610604e-01f,
                   RL_PA2 = -3.7220788002e-01f, RL_PA3 = 3.1834661961e-01f,
                   RL_PA4 = -1.1089469492e-01f, RL_PA5 = 3.5478305072e-02f,
                   RL_PA6 = -2.1663755178e-03f;
static const float RL_QA1 = 1.0642088205e-01f, RL_QA2 = 5.4039794207e-01f,
                   RL_QA3 = 7.1828655899e-02f, RL_QA4 = 1.2617121637e-01f,
                   RL_QA5 = 1.3637083583e-02f, RL_QA6 = 1.1984500103e-02f;
static const float RL_RA0 = -9.8649440333e-03f, RL_RA1 = -6.9385856390e-01f,
                   RL_RA2 = -1.0558626175e+01f, RL_RA3 = -6.2375331879e+01f,
                   RL_RA4 = -1.6239666748e+02f, RL_RA5 = -1.8460508728e+02f,
                   RL_RA6 = -8.1287437439e+01f, RL_RA7 = -9.8143291473e+00f;
static const float RL_SA1 = 1.9651271820e+01f, RL_SA2 = 1.3765776062e+02f,
                   RL_SA3 = 4.3456588745e+02f, RL_SA4 = 6.4538726807e+02f,
                   RL_SA5 = 4.2900814819e+02f, RL_SA6 = 1.0863500214e+02f,
                   RL_SA7 = 6.5702495575e+00f, RL_SA8 = -6.0424413532e-02f;
static const float RL_RB0 = -9.8649431020e-03f, RL_RB1 = -7.9928326607e-01f,
                   RL_RB2 = -1.7757955551e+01f, RL_RB3 = -1.6063638306e+02f,
                   RL_RB4 = -6.3756646729e+02f, RL_RB5 = -1.0250950928e+03f,
                   RL_RB6 = -4.8351919556e+02f;
static const float RL_SB1 = 3.0338060379e+01f, RL_SB2 = 3.2579251099e+02f,
                   RL_SB3 = 1.5367296143e+03f, RL_SB4 = 3.1998581543e+03f,
                   RL_SB5 = 2.5530502930e+03f, RL_SB6 = 4.7452853394e+02f,
                   RL_SB7 = -2.2440952301e+01f;

static float rl_erfc1(float x) {
    float s = fabsf(x) - 1.0f;
    float p = RL_PA0 + s * (RL_PA1 + s * (RL_PA2 + s * (RL_PA3 + s * (RL_PA4 + s * (RL_PA5 + s * RL_PA6)))));
    float q = 1.0f + s * (RL_QA1 + s * (RL_QA2 + s * (RL_QA3 + s * (RL_QA4 + s * (RL_QA5 + s * RL_QA6)))));
    return 1.0f - RL_ERX - p / q;
}

static float rl_erfc2(uint32_t ix, float x) {
    if (ix < 0x3fa00000) { /* |x| < 1.25 */
        return rl_erfc1(x);
    }
    x = fabsf(x);
    float s = 1.0f / (x * x);
    float r, big_s;
    if (ix < 0x4036db6d) { /* |x| < 1/0.35 */
        r = RL_RA0 + s * (RL_RA1 + s * (RL_RA2 + s * (RL_RA3 + s * (RL_RA4 + s * (RL_RA5 + s * (RL_RA6 + s * RL_RA7))))));
        big_s = 1.0f + s * (RL_SA1 + s * (RL_SA2 + s * (RL_SA3 + s * (RL_SA4 + s * (RL_SA5 + s * (RL_SA6 + s * (RL_SA7 + s * RL_SA8)))))));
    } else { /* |x| >= 1/0.35 */
        r = RL_RB0 + s * (RL_RB1 + s * (RL_RB2 + s * (RL_RB3 + s * (RL_RB4 + s * (RL_RB5 + s * RL_RB6)))));
        big_s = 1.0f + s * (RL_SB1 + s * (RL_SB2 + s * (RL_SB3 + s * (RL_SB4 + s * (RL_SB5 + s * (RL_SB6 + s * RL_SB7))))));
    }
    union { float f; uint32_t i; } ux;
    ux.f = x;
    union { float f; uint32_t i; } uz;
    uz.i = ux.i & 0xffffe000;
    float z = uz.f;
    return rl_expf(-z * z - 0.5625f) * rl_expf((z - x) * (z + x) + r / big_s) / x;
}

static float rl_erff(float x) {
    union { float f; uint32_t i; } ux;
    ux.f = x;
    uint32_t ix = ux.i;
    int sign = (int)(ix >> 31);
    ix &= 0x7fffffff;
    if (ix >= 0x7f800000) { /* erf(nan)=nan, erf(+-inf)=+-1 */
        return 1.0f - 2.0f * (float)sign + 1.0f / x;
    }
    if (ix < 0x3f580000) { /* |x| < 0.84375 */
        if (ix < 0x31800000) { /* |x| < 2**-28 */
            return 0.125f * (8.0f * x + RL_EFX8 * x);
        }
        float z = x * x;
        float r = RL_PP0 + z * (RL_PP1 + z * (RL_PP2 + z * (RL_PP3 + z * RL_PP4)));
        float s = 1.0f + z * (RL_QQ1 + z * (RL_QQ2 + z * (RL_QQ3 + z * (RL_QQ4 + z * RL_QQ5))));
        float y = r / s;
        return x + x * y;
    }
    float y;
    if (ix < 0x40c00000) { /* |x| < 6 */
        y = 1.0f - rl_erfc2(ix, x);
    } else {
        const float x1p_120 = 0x1p-120f;
        y = 1.0f - x1p_120;
    }
    return sign ? -y : y;
}

extern "C" float mimi_gelu_erf_f32(float x) {
    // candle GeluErf::f32 EXACTLY (op.rs):
    //   (erf_f32(v * FRAC_1_SQRT_2) + 1.) * 0.5 * v
    // Left-associated ((erf+1)*0.5)*v — NOT 0.5*x*(1+erf), which rounds
    // differently. erf_f32 == Rust libm erff (ported above), not Apple erff.
    const float frac_1_sqrt_2 = 0.70710678118654752440f; /* rounds to 0x3f3504f3 */
    return (rl_erff(x * frac_1_sqrt_2) + 1.0f) * 0.5f * x;
}

extern "C" float mimi_elu_f32(float x, float alpha) {
    // candle cpu_backend elu EXACTLY: is_sign_positive() selects (true for
    // +0.0, false for -0.0 — not `x > 0`), else (exp(x) - 1) * alpha.
    // v.exp() is Rust std = the system expf — NOT rl_expf.
    return !std::signbit(x) ? x : (expf(x) - 1.0f) * alpha;
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
        lanes[0] = rl_erff(lanes[0]);
        lanes[1] = rl_erff(lanes[1]);
        lanes[2] = rl_erff(lanes[2]);
        lanes[3] = rl_erff(lanes[3]);
        float32x4_t e = vld1q_f32(lanes);
        // candle order: ((e + 1) * 0.5) * x — same association as the scalar.
        float32x4_t res = vmulq_f32(vmulq_f32(vaddq_f32(e, one), half), vx);
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
        // candle: (exp(x) - 1) * alpha for the sign-negative branch, selected
        // by is_sign_positive() — a SIGN-BIT test, not x > 0 (+0.0 takes the
        // identity branch, -0.0 the exp branch).
        float32x4_t neg = vmulq_n_f32(vsubq_f32(ve, one), alpha);
        uint32x4_t signbits =
            vtstq_u32(vreinterpretq_u32_f32(vx), vdupq_n_u32(0x80000000u));
        float32x4_t res = vbslq_f32(signbits, neg, vx);
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
    // pass 2: e = expf(x - mx), store ALL first (candle exps the whole row,
    // THEN reduces — the sum is a separate pass over dst, not fused).
    // expf = system libm: candle's `.exp()` is Rust std, which lowers to the
    // platform expf on aarch64-darwin — unlike erf, which is Rust-libm.
    const float32x4_t vmx = vdupq_n_f32(mx);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x + i), vmx);
        float lanes[4];
        vst1q_f32(lanes, d);
        lanes[0] = expf(lanes[0]);
        lanes[1] = expf(lanes[1]);
        lanes[2] = expf(lanes[2]);
        lanes[3] = expf(lanes[3]);
        vst1q_f32(x + i, vld1q_f32(lanes));
    }
    for (; i < n; ++i) {
        x[i] = expf(x[i] - mx);
    }
    // pass 2b: sum with candle's EXACT vec_sum blocking (cpu/mod.rs vec_sum +
    // neon.rs CurrentCpu: STEP=16, four q-register accumulators, tree reduce
    // x0+=x1 / x2+=x3 / x0+=x2, vaddvq, then SCALAR leftovers appended after).
    // A single-accumulator sum rounds differently — this is the bit-matched
    // reduction for the softmax row.
    const int np = n & ~15;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (i = 0; i + 16 <= n; i += 16) {
        acc0 = vaddq_f32(acc0, vld1q_f32(x + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(x + i + 4));
        acc2 = vaddq_f32(acc2, vld1q_f32(x + i + 8));
        acc3 = vaddq_f32(acc3, vld1q_f32(x + i + 12));
    }
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    float sum = vaddvq_f32(acc0);
    for (i = np; i < n; ++i) {
        sum += x[i];
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
    // The two accumulations are STRICTLY SEQUENTIAL SCALAR — not a shortcut,
    // a numerical requirement (review P1, probe-proven): var = sum2/n − mean²
    // is cancellation-critical on near-constant rows; a 512-wide row of
    // ~10 ± 1e-5 sat on the var ≈ −eps knife edge where candle's sequential
    // rounding stayed finite and a NEON lane-blocked reduction produced NaN
    // (another case diverged by 0.11 — unbounded, not ulp-band). Bit-matching
    // candle's exact accumulation order inherits the reference's survival
    // characteristics; the elementwise apply below stays NEON (order-exact).
    // Per-element ops stay unfused (rustc never contracts — build with
    // -ffp-contract=off). w/b may be NULL -> treated as 1 / 0 (affine off).
    if (n <= 0) {
        return;
    }
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int j = 0; j < n; ++j) {
        float v = x[j];
        sum += v;
        float vv = v * v;
        sum2 += vv;
    }
    const float mean = sum / (float)n;
    const float var = sum2 / (float)n - mean * mean;
    const float inv_std = 1.0f / sqrtf(var + eps);
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    const float32x4_t vmean = vdupq_n_f32(mean);
    const float32x4_t vinv = vdupq_n_f32(inv_std);
    // apply: y = ((x−mean)·inv_std)·w + b, unfused adds (vaddq, not vmlaq).
    int i = 0;
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
    for (int i = 0; i < n; ++i) {
        float normed = (x[i] - mean) * inv_std;
        float wi = w ? w[i] : 1.0f;
        float bi = b ? b[i] : 0.0f;
        float t = normed * wi;
        y[i] = t + bi;
    }
#endif
}

extern "C" void mimi_weight_scale_vec_f32(const float *x, const uint8_t *s,
                                            float *y, int n) {
    int i = 0;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(y + i,
                  vmulq_f32(vld1q_f32(x + i),
                            mimi_weight_load4_f32(s + i * sizeof(float))));
    }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(y + i,
                      _mm_mul_ps(_mm_loadu_ps(x + i),
                                 mimi_weight_load4_f32(s + i * sizeof(float))));
    }
#endif
    for (; i < n; ++i) y[i] = x[i] * mimi_weight_load_f32(s, i);
}

extern "C" void mimi_weight_layer_norm_f32(const float *x, const uint8_t *w,
                                             const uint8_t *b, float *y, int n,
                                             float eps) {
    if (n <= 0) return;
    float sum = 0.0f;
    float sum2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
        sum2 += x[i] * x[i];
    }
    const float mean = sum / static_cast<float>(n);
    const float variance = sum2 / static_cast<float>(n) - mean * mean;
    const float inv = 1.0f / sqrtf(variance + eps);
    int i = 0;
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    const float32x4_t vmean = vdupq_n_f32(mean);
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t normed =
            vmulq_f32(vsubq_f32(vld1q_f32(x + i), vmean), vinv);
        const float32x4_t vw =
            w ? mimi_weight_load4_f32(w + i * sizeof(float)) : vdupq_n_f32(1.0f);
        const float32x4_t vb =
            b ? mimi_weight_load4_f32(b + i * sizeof(float)) : vdupq_n_f32(0.0f);
        vst1q_f32(y + i, vaddq_f32(vmulq_f32(normed, vw), vb));
    }
#elif defined(MIMI_HAVE_SSE2) && !defined(MIMI_SCALAR_REF)
    const __m128 vmean = _mm_set1_ps(mean);
    const __m128 vinv = _mm_set1_ps(inv);
    for (; i + 4 <= n; i += 4) {
        const __m128 normed =
            _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(x + i), vmean), vinv);
        const __m128 vw = w ? mimi_weight_load4_f32(w + i * sizeof(float))
                            : _mm_set1_ps(1.0f);
        const __m128 vb = b ? mimi_weight_load4_f32(b + i * sizeof(float))
                            : _mm_setzero_ps();
        _mm_storeu_ps(y + i, _mm_add_ps(_mm_mul_ps(normed, vw), vb));
    }
#endif
    for (; i < n; ++i) {
        const float wi = w ? mimi_weight_load_f32(w, i) : 1.0f;
        const float bi = b ? mimi_weight_load_f32(b, i) : 0.0f;
        y[i] = ((x[i] - mean) * inv) * wi + bi;
    }
}

// ===========================================================================
// (d) Top level — immutable model plan + conversation-owned decode state
// ===========================================================================

struct MimiDecodePlan {
    MimiWeight *entries;
    uint8_t *bound;
    MimiWeightTable table;
    MimiDerivedArena derived;
    size_t state_bytes;
    uint64_t bound_weight_bytes;
    // Weight bytes MATERIALIZED rather than bound as a view — staging, a
    // transpose/repack, an alignment copy, or any re-laid weight buffer.
    // Doctrine requires 0 in production, and the model-level
    // `compatibility_copied_bytes` gate reads it. A real tally, not a constant:
    // ANY code that materializes a weight MUST add its bytes here, exactly as
    // binding adds to `bound_weight_bytes` — otherwise the gate silently stops
    // being able to fail. (Weight-norm folds are DERIVED, not materialized, and
    // belong in `derived`.) Zeroed by the plan's calloc.
    uint64_t compatibility_copied_bytes;
};

struct MimiDecodeState {
    const MimiDecodePlan *plan;
    MimiArena arena;
    MimiQuantState *quant;
    MimiUpsampleState *upsample;
    MimiTransformerState *transformer;
    MimiSeanetState *seanet;

    // Inter-stage latent buffers. Quantizer decode always emits exactly one
    // frame. The transformer's documented in-place contract lets up_buf carry
    // both upsample output and final transformer output into SeaNet.
    float *emb_buf;  // [MIMI_DIM], quantizer.decode output
    float *up_buf;   // [MIMI_DIM, MIMI_MAX_LATENT], upsample -> transformer
    // The final PCM lands directly in the caller's pcm_out (capacity
    // MIMI_FRAME_OUT*2), so no pcm scratch is carved here.
};

#ifdef LFM_BUILD_ORACLE
struct MimiDecoder {
    LfmWeightImage *weights;
    MimiDecodePlan *plan;
    MimiDecodeState *state;
};
#endif

static size_t mimi_align_size(size_t bytes) {
    if (bytes > SIZE_MAX - (MIMI_ARENA_ALIGN - 1)) return 0;
    return (bytes + (MIMI_ARENA_ALIGN - 1)) & ~(MIMI_ARENA_ALIGN - 1);
}

static int mimi_state_init(MimiDecodeState **out, const MimiDecodePlan *plan,
                           size_t capacity, char *err, size_t errlen) {
    if (!out || !plan || !plan->table.entries || capacity == 0) return -1;
    *out = nullptr;
    MimiDecodeState *state =
        static_cast<MimiDecodeState *>(calloc(1, sizeof(MimiDecodeState)));
    if (!state) {
        MIMI_ERR("mimi state: OOM allocating state descriptor");
        return -1;
    }
    void *base = aligned_alloc(MIMI_ARENA_ALIGN, capacity);
    if (!base) {
        MIMI_ERR("mimi state: OOM allocating %zu-byte arena", capacity);
        free(state);
        return -1;
    }
    state->plan = plan;
    state->arena = {static_cast<uint8_t *>(base), capacity, 0,
                    const_cast<MimiDerivedArena *>(&plan->derived), 0};

    state->emb_buf = static_cast<float *>(
        mimi_arena_alloc(&state->arena, (size_t)MIMI_DIM * sizeof(float)));
    const size_t latent_floats = (size_t)MIMI_DIM * (size_t)MIMI_MAX_LATENT;
    state->up_buf = static_cast<float *>(
        mimi_arena_alloc(&state->arena, latent_floats * sizeof(float)));

    int rc = mimi_quant_init(&state->quant, &plan->table, &state->arena, err, errlen);
    if (rc) {
        free(base);
        free(state);
        return rc;
    }
    rc = mimi_upsample_init(&state->upsample, &plan->table, &state->arena,
                            err, errlen);
    if (rc) {
        free(base);
        free(state);
        return rc;
    }
    rc = mimi_transformer_init(&state->transformer, &plan->table,
                               &state->arena, err, errlen);
    if (rc) {
        free(base);
        free(state);
        return rc;
    }
    rc = mimi_seanet_init(&state->seanet, &plan->table, &state->arena,
                          err, errlen);
    if (rc) {
        free(base);
        free(state);
        return rc;
    }
    if (plan->derived.sealed &&
        state->arena.derived_cursor != plan->derived.used) {
        MIMI_ERR("mimi state: derived-plan replay mismatch (%zu != %zu)",
                 state->arena.derived_cursor, plan->derived.used);
        free(base);
        free(state);
        return -2;
    }
    *out = state;
    return 0;
}

static void mimi_state_destroy(MimiDecodeState *state) {
    if (!state) return;
    free(state->arena.base);
    free(state);
}

static int mimi_plan_new(MimiDecodePlan **out, const MimiWeightTable *weights,
                         char *err, size_t errlen) {
    if (!out || !weights || !weights->entries || weights->count == 0) return -1;
    *out = nullptr;
    MimiDecodePlan *plan =
        static_cast<MimiDecodePlan *>(calloc(1, sizeof(MimiDecodePlan)));
    if (!plan) return -1;
    plan->entries = static_cast<MimiWeight *>(
        calloc(weights->count, sizeof(MimiWeight)));
    if (!plan->entries) {
        free(plan);
        return -1;
    }
    std::memcpy(plan->entries, weights->entries,
                static_cast<size_t>(weights->count) * sizeof(MimiWeight));
    plan->bound = static_cast<uint8_t *>(calloc(weights->count, sizeof(uint8_t)));
    if (!plan->bound) {
        free(plan->entries);
        free(plan);
        return -1;
    }
    plan->table = {plan->entries, weights->count, plan->bound};
    plan->derived.base = static_cast<uint8_t *>(
        aligned_alloc(MIMI_ARENA_ALIGN, MIMI_DERIVED_ARENA_MAX));
    if (!plan->derived.base) {
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return -1;
    }
    plan->derived.size = MIMI_DERIVED_ARENA_MAX;

    MimiDecodeState *probe = nullptr;
    int rc = mimi_state_init(&probe, plan, MIMI_STATE_ARENA_MAX, err, errlen);
    if (rc != 0) {
        free(plan->derived.base);
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return rc;
    }

    for (uint32_t index = 0; index < weights->count; ++index) {
        if (!plan->bound[index]) continue;
        const uint64_t elements = plan->entries[index].len;
        if (elements > UINT64_MAX / sizeof(float) ||
            elements * sizeof(float) > UINT64_MAX - plan->bound_weight_bytes) {
            MIMI_ERR("mimi plan: bound-weight byte accounting overflow");
            mimi_state_destroy(probe);
            free(plan->derived.base);
            free(plan->bound);
            free(plan->entries);
            free(plan);
            return -3;
        }
        plan->bound_weight_bytes += elements * sizeof(float);
    }
    /* Binding discovery is plan-construction-only. Conversation creation may
     * happen concurrently, so published plan lookups must be read-only rather
     * than racing on the temporary bitmap. */
    free(plan->bound);
    plan->bound = nullptr;
    plan->table.bound = nullptr;

    const size_t headroom = probe->arena.size - probe->arena.used;
    if (headroom < MIMI_ARENA_HEADROOM_MIN) {
        MIMI_ERR("mimi plan: state probe headroom %zu < %zu", headroom,
                 MIMI_ARENA_HEADROOM_MIN);
        mimi_state_destroy(probe);
        free(plan->derived.base);
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return -2;
    }
    plan->state_bytes = mimi_align_size(probe->arena.used);
    size_t derived_bytes = 0;
    if (!mimi_align_checked(plan->derived.used, &derived_bytes) ||
        derived_bytes == 0) {
        MIMI_ERR("mimi plan: invalid derived footprint %zu", plan->derived.used);
        mimi_state_destroy(probe);
        free(plan->derived.base);
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return -2;
    }
    uint8_t *derived = static_cast<uint8_t *>(
        aligned_alloc(MIMI_ARENA_ALIGN, derived_bytes));
    if (!derived) {
        mimi_state_destroy(probe);
        free(plan->derived.base);
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return -1;
    }
    std::memcpy(derived, plan->derived.base, plan->derived.used);
    free(plan->derived.base);
    plan->derived.base = derived;
    plan->derived.size = derived_bytes;
    plan->derived.sealed = 1;
    mimi_state_destroy(probe);
    if (plan->state_bytes == 0) {
        free(plan->derived.base);
        free(plan->bound);
        free(plan->entries);
        free(plan);
        return -2;
    }
    *out = plan;
    return 0;
}

static int mimi_plan_new_from_component(MimiDecodePlan **out,
                                        const LfmWeightImage *image,
                                        uint32_t component, char *err,
                                        size_t errlen) {
    if (!out || !image) return -1;
    *out = nullptr;
    const size_t count = lfm_weights_component_count(image, component);
    if (count > UINT32_MAX) {
        MIMI_ERR("mimi decoder: too many tensors (%zu)", count);
        return -3;
    }

    MimiWeight *entries = (MimiWeight *)calloc(count, sizeof(MimiWeight));
    if (count != 0 && !entries) {
        MIMI_ERR("mimi decoder: OOM allocating %zu descriptors", count);
        return -4;
    }

    for (size_t i = 0; i < count; ++i) {
        LfmTensorView view = {};
        int rc = lfm_weights_at_component(image, component, i, &view);
        if (rc != LFM_WEIGHT_OK) {
            MIMI_ERR("mimi decoder: tensor lookup %zu failed", i);
            free(entries);
            return rc;
        }
        if (view.dtype != LFM_DTYPE_F32) {
            MIMI_ERR("mimi decoder: tensor '%s' is %s, expected F32",
                     view.name, lfm_weights_dtype_name(view.dtype));
            free(entries);
            return -3;
        }
        entries[i] = MimiWeight{
            view.name,
            static_cast<const uint8_t *>(view.data),
            view.shape,
            view.rank,
            view.elements,
        };
    }

    const MimiWeightTable table = {entries, (uint32_t)count, nullptr};
    const int rc = mimi_plan_new(out, &table, err, errlen);
    free(entries);
    return rc;
}

extern "C" int mimi_decode_plan_new_from_image(MimiDecodePlan **out,
                                                const LfmWeightImage *image,
                                                char *err, size_t errlen) {
    return mimi_plan_new_from_component(out, image, LFM_WEIGHT_COMPONENT_CODEC,
                                        err, errlen);
}

extern "C" void mimi_decode_plan_free(MimiDecodePlan *plan) {
    if (!plan) return;
    free(plan->derived.base);
    free(plan->bound);
    free(plan->entries);
    free(plan);
}

extern "C" uint64_t mimi_decode_plan_derived_bytes(const MimiDecodePlan *plan) {
    return plan ? static_cast<uint64_t>(plan->derived.size) : 0;
}

extern "C" uint64_t
mimi_decode_plan_bound_weight_bytes(const MimiDecodePlan *plan) {
    return plan ? plan->bound_weight_bytes : 0;
}

extern "C" uint64_t mimi_decode_plan_compatibility_copied_bytes(
    const MimiDecodePlan *plan) {
    return plan ? plan->compatibility_copied_bytes : 0;
}

extern "C" int mimi_decode_state_new(MimiDecodeState **out,
                                      const MimiDecodePlan *plan, char *err,
                                      size_t errlen) {
    if (!plan || !plan->derived.sealed) return -1;
    return mimi_state_init(out, plan, plan->state_bytes, err, errlen);
}

extern "C" void mimi_decode_state_free(MimiDecodeState *state) {
    mimi_state_destroy(state);
}

extern "C" uint64_t mimi_decode_state_bytes(const MimiDecodeState *state) {
    return state ? static_cast<uint64_t>(sizeof(MimiDecodeState) +
                                         state->arena.size) : 0;
}

#ifdef LFM_BUILD_ORACLE
extern "C" int mimi_decoder_new(MimiDecoder **out, const MimiWeightTable *weights,
                                 char *err, size_t errlen) {
    if (!out) return -1;
    *out = nullptr;
    MimiDecoder *decoder =
        static_cast<MimiDecoder *>(calloc(1, sizeof(MimiDecoder)));
    if (!decoder) return -1;
    int rc = mimi_plan_new(&decoder->plan, weights, err, errlen);
    if (rc == 0) {
        rc = mimi_decode_state_new(&decoder->state, decoder->plan, err, errlen);
    }
    if (rc != 0) {
        mimi_decode_plan_free(decoder->plan);
        free(decoder);
        return rc;
    }
    *out = decoder;
    return 0;
}

extern "C" int mimi_decoder_new_from_image(MimiDecoder **d_out,
                                             const LfmWeightImage *image,
                                             char *err, size_t errlen) {
    if (!d_out || !image) return -1;
    *d_out = nullptr;
    MimiDecoder *decoder =
        static_cast<MimiDecoder *>(calloc(1, sizeof(MimiDecoder)));
    if (!decoder) return -1;
    int rc = mimi_decode_plan_new_from_image(&decoder->plan, image, err, errlen);
    if (rc == 0) {
        rc = mimi_decode_state_new(&decoder->state, decoder->plan, err, errlen);
    }
    if (rc != 0) {
        mimi_decode_plan_free(decoder->plan);
        free(decoder);
        return rc;
    }
    *d_out = decoder;
    return 0;
}

extern "C" int mimi_decoder_new_from_file(MimiDecoder **d_out,
                                           const char *checkpoint,
                                           char *err, size_t errlen) {
    if (!d_out) return -1;
    *d_out = NULL;
    if (!checkpoint || checkpoint[0] == '\0') {
        MIMI_ERR("mimi_decoder_new_from_file: empty checkpoint path");
        return -1;
    }
    LfmWeightImage *image = NULL;
    int rc = lfm_weights_open(checkpoint, &image, err, errlen);
    if (rc != LFM_WEIGHT_OK) return rc;
    MimiDecoder *decoder =
        static_cast<MimiDecoder *>(calloc(1, sizeof(MimiDecoder)));
    if (!decoder) {
        lfm_weights_close(image);
        return -1;
    }
    rc = mimi_plan_new_from_component(&decoder->plan, image,
                                      LFM_WEIGHT_COMPONENT_MAIN, err, errlen);
    if (rc == 0) {
        rc = mimi_decode_state_new(&decoder->state, decoder->plan, err, errlen);
    }
    if (rc != 0) {
        mimi_decode_plan_free(decoder->plan);
        free(decoder);
        lfm_weights_close(image);
        return rc;
    }
    decoder->weights = image;
    *d_out = decoder;
    return 0;
}
#endif

extern "C" int mimi_decode_state_step(MimiDecodeState *d,
                                       const uint32_t *codes, float *pcm_out) {
    // Faithful port of Mimi::decode_step (mimi.rs:214). In our streaming path
    // (audio_out.rs) `codes` is always a present single latent frame, so the
    // Rust `codes.as_option()` is always Some and the quantizer always emits one
    // embedding frame. Each stage is still invoked with the previous stage's
    // reported count, so a 0 (priming) propagates 0 onward exactly as
    // StreamTensor::empty() does through the Rust `.step` chain.
    if (!d || !codes || !pcm_out) return -EINVAL;

    // quantizer.decode: codes[MIMI_NQ] -> emb[MIMI_DIM, 1]. Pure per-frame RVQ
    // dequantize (no streaming state), so exactly 1 frame out.
    mimi_quant_decode(d->quant, codes, d->emb_buf);
    const int n_emb = 1;
    // --- fence F0 (post-quant): emb_buf[MIMI_DIM, n_emb], see NOTES (f) ---

    // upsample.step: [MIMI_DIM, n_emb] -> [MIMI_DIM, n_up]. Stride 2 => 2 frames
    // out per input frame in steady state (n_up == 2).
    int n_up = mimi_upsample_step(d->upsample, d->emb_buf, n_emb, d->up_buf);
    if (n_up < 0) return n_up;  // stage misuse/bounds error — NEVER folded into
                                // "0 = priming" (review P2: a failed decode must
                                // not read as valid empty output)
    // --- fence F1 (post-upsample): up_buf[MIMI_DIM, n_up] ---

    // decoder_transformer.step: [MIMI_DIM, n_up] -> [MIMI_DIM, n_tr]. Causal KV
    // transformer preserves the frame count (n_tr == n_up). Intra: per-layer F2.
    int n_tr = mimi_transformer_step(d->transformer, d->up_buf, n_up, d->up_buf);
    if (n_tr < 0) return n_tr;  // propagate stage error (see above)
    // --- fence F2 (post-transformer): up_buf[MIMI_DIM, n_tr] ---

    // decoder(seanet).step: [MIMI_DIM, n_tr] -> pcm[1, n_pcm]. x960 upsample =>
    // n_pcm == n_tr * 960 in steady state (== MIMI_FRAME_OUT for n_tr == 2).
    // Intra: per {upsample+resnet} layer F3.
    int n_pcm = mimi_seanet_step(d->seanet, d->up_buf, n_tr, pcm_out);
    // --- fence F4 (post-seanet): pcm_out[1, n_pcm] — pass-boundary doorbell ---

    return n_pcm;
}

extern "C" void mimi_decode_state_reset(MimiDecodeState *d) {
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

#ifdef LFM_BUILD_ORACLE
extern "C" int mimi_decoder_step(MimiDecoder *decoder, const uint32_t *codes,
                                   float *pcm_out) {
    return decoder ? mimi_decode_state_step(decoder->state, codes, pcm_out)
                   : -EINVAL;
}

extern "C" void mimi_decoder_reset(MimiDecoder *decoder) {
    if (decoder) mimi_decode_state_reset(decoder->state);
}

extern "C" void mimi_decoder_free(MimiDecoder *d) {
    if (!d) {
        return;
    }
    mimi_decode_state_free(d->state);
    mimi_decode_plan_free(d->plan);
    lfm_weights_close(d->weights);
    free(d);
}

extern "C" uint64_t mimi_decoder_derived_bytes(const MimiDecoder *d) {
    return d ? mimi_decode_plan_derived_bytes(d->plan) : 0;
}

extern "C" uint64_t mimi_decoder_compatibility_copied_bytes(
    const MimiDecoder *d) {
    return d ? mimi_decode_plan_compatibility_copied_bytes(d->plan) : 0;
}
#endif

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
 *  candle activation matmul                   | mimi_gemm_f32 / mimi_gemv_f32 (AMX cblas)
 *  candle checkpoint linear                   | mimi_weight_* (byte-load NEON/SSE)
 *  candle_nn::LayerNorm                       | mimi_layer_norm_f32 (NEON)
 *  candle gelu_erf / Elu / softmax_last_dim   | mimi_gelu_erf_vec_f32 / _elu_vec_ / _softmax_
 *  transformer projection + LayerScale + add | mimi_weight_gemv_scale_residual_rows_f32
 *  other residual/skip and scale sweeps       | mimi_add_vec_f32 / mimi_scale_vec_f32
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
 * (c) PLAN / STATE ARENAS
 * -----------------------
 *  `MimiDecodePlan` owns only validated descriptors and formula-changing
 *  immutable tables. The production checkpoint derives eight 2048x256
 *  codebooks plus 32 RoPE inverse frequencies: 16,777,344 bytes total.
 *  Layout, alignment, dtype, and transpose copies are absent.
 *
 *  Plan construction performs one temporary 256 MiB state probe while building
 *  derived tables into a separate arena. It records the state high-water mark,
 *  seals the derived arena, and discards the probe. Each conversation then gets
 *  an exact-sized mutable arena. `mimi_decode_state_bytes` reports the sealed
 *  high-water mark including the state descriptor, KV rings, conv carry, and
 *  activation scratch; no hard-coded total is authoritative as scratch planes
 *  are deleted. State initialization replays derived offsets and never
 *  recomputes or writes the sealed tables. Thus two conversations share one
 *  image and one derived plan while retaining independent mutable recurrence.
 *
 * (d) PRIMITIVE-KERNEL LOOP ORDERS  ("math is assembly at every step")
 * --------------------------------------------------------------------
 *  resident gemv (y[M] = W[M,K]*x + b): W remains `uint8_t*`; sixteen values
 *    are consumed as four byte-loaded NEON/SSE register blocks, then reduced.
 *    Bias and scalar tails use explicit little-endian loads.
 *  transformer residual epilogue: that exact resident row reducer feeds a
 *    separately-rounded sum*LayerScale followed by residual+scaled, with
 *    residual as the left operand. It writes only the final activation row;
 *    -ffp-contract=off forbids fma and no branch plane exists.
 *  resident gemm: row-major W streams one scalar register value per K while
 *    the mutable activation columns are vectorized. The transposed form uses
 *    contiguous four-row checkpoint loads for the hot n=2 route. No typed
 *    checkpoint pointer, transpose copy, packed panel, or alignment path exists.
 *  activation-only gemv/gemm retain the cblas AMX route because their storage
 *    consists of live C++ float objects rather than checkpoint bytes.
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
 *  sub-vector tail; resident hot paths are byte-load NEON/SSE.
 *
 * (e) UNCERTAINTIES / ARBITER RECONCILIATION
 * ------------------------------------------
 *  1. Step return convention. I read every *_step's int return as n_out (frames
 *     for upsample/transformer, samples for seanet), per the header's "reports
 *     n_out frames; 0 is legal" and conv's "returns n_out (>=0)". If a unit ever
 *     returns a negative error code from a step, this file passes it through
 *     unchanged (no steady-state error channel is defined). Please keep steps
 *     infallible (>=0) or the arbiter must add an error convention.
 *  2. MIMI_MAX_LATENT = 4 (not the steady 2). Sized to the header's pcm_out
 *     capacity MIMI_FRAME_OUT*2 = 4*960 "drain headroom", so a hypothetical
 *     double-emit step cannot overflow the latent buffers or pcm_out. If the
 *     arbiter proves emit is bounded at 2, this can drop to 2 (buffers shrink,
 *     pcm capacity stays per header).
 *  4. Accumulation order for parity. Resident gemv uses a fixed four-register
 *     reduction tree; matrix variants preserve ascending-K accumulation per
 *     output. This differs from candle's blocked gemm (ulp-band tier). NEON reductions
 *     (softmax/layernorm lane-sums) also differ from a strict sequential sum.
 *     Bisect with -DMIMI_SCALAR_REF (forces scalar gemm/gemv + scalar
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
 *  mimi_gelu_erf_vec_f32, and the matrix leaves) hold NO global or
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
