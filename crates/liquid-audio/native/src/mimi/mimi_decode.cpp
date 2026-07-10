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
// accumulate, documented loop order. NEON inner loops on aarch64 with scalar
// `_ref` siblings for parity bisecting: build with -DMIMI_SCALAR_REF to force
// the reference path and diff against the NEON path.
// ===========================================================================

// -------- gemv: y[m] = sum_k w[m*k + k] * x[k] (+ bias[m]); W row-major [M,K] --

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

#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
static void mimi_gemv_f32_neon(const float *w, const float *x, const float *bias,
                               float *y, int m, int k) {
    for (int i = 0; i < m; ++i) {
        const float *wr = w + (size_t)i * (size_t)k;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 4 <= k; j += 4) {
            acc = vmlaq_f32(acc, vld1q_f32(wr + j), vld1q_f32(x + j));
        }
        float s = vaddvq_f32(acc);  // horizontal sum of the 4 lanes
        for (; j < k; ++j) {
            s += wr[j] * x[j];
        }
        if (bias) {
            s += bias[i];
        }
        y[i] = s;
    }
}
#endif

extern "C" void mimi_gemv_f32(const float *w, const float *x,
                              const float *bias_or_null, float *y, int m, int k) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    mimi_gemv_f32_neon(w, x, bias_or_null, y, m, k);
#else
    mimi_gemv_f32_ref(w, x, bias_or_null, y, m, k);
#endif
}

// -------- gemm: C[M,N] = A[M,K]*B[K,N] (beta 0) or += (beta 1); row-major -----
// Loop order i-k-j: stream B[k,:] and C[i,:] contiguously, vectorize over j.

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

#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
static void mimi_gemm_f32_neon(const float *a, const float *b, float *c, int m,
                               int k, int n, int beta) {
    for (int i = 0; i < m; ++i) {
        float *cr = c + (size_t)i * (size_t)n;
        if (beta == 0) {
            memset(cr, 0, (size_t)n * sizeof(float));
        }
        for (int p = 0; p < k; ++p) {
            const float aval = a[(size_t)i * (size_t)k + (size_t)p];
            const float32x4_t va = vdupq_n_f32(aval);
            const float *br = b + (size_t)p * (size_t)n;
            int j = 0;
            for (; j + 4 <= n; j += 4) {
                float32x4_t cv = vld1q_f32(cr + j);
                cv = vmlaq_f32(cv, va, vld1q_f32(br + j));
                vst1q_f32(cr + j, cv);
            }
            for (; j < n; ++j) {
                cr[j] += aval * br[j];
            }
        }
    }
}
#endif

extern "C" void mimi_gemm_f32(const float *a, const float *b, float *c, int m,
                              int k, int n, int beta) {
#if defined(MIMI_HAVE_NEON) && !defined(MIMI_SCALAR_REF)
    mimi_gemm_f32_neon(a, b, c, m, k, n, beta);
#else
    mimi_gemm_f32_ref(a, b, c, m, k, n, beta);
#endif
}

// -------- softmax: in place, max-subtracted, f32 sum -------------------------

extern "C" void mimi_softmax_f32(float *x, int n) {
    if (n <= 0) {
        return;
    }
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
    // sum > 0 always (at least the max element contributes expf(0) == 1).
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

// -------- gelu (erf form) & elu ---------------------------------------------

extern "C" float mimi_gelu_erf_f32(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2))); candle's `gelu_erf`. 1/sqrt(2) constant.
    const float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}

extern "C" float mimi_elu_f32(float x, float alpha) {
    // candle Elu(alpha): x > 0 ? x : alpha * (exp(x) - 1).
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

// -------- layer norm: two-pass mean/var --------------------------------------

extern "C" void mimi_layer_norm_f32(const float *x, const float *w,
                                    const float *b, float *y, int n, float eps) {
    // candle_nn::LayerNorm forward, matched term-for-term:
    //   mean = sum(x)/n
    //   var  = sum((x-mean)^2)/n          (BIASED — divide by n, not n-1)
    //   y    = (x-mean)/sqrt(var+eps) * w + b   (eps added to var BEFORE sqrt)
    // `eps` is supplied by the consumer (unit 4 passes the transformer's LN
    // eps). f32 accumulation, sequential order, to track candle's CPU reduce.
    // w/b may be NULL -> treated as 1 / 0 (elementwise_affine off).
    if (n <= 0) {
        return;
    }
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) {
        float normed = (x[i] - mean) * inv_std;
        float wi = w ? w[i] : 1.0f;
        float bi = b ? b[i] : 0.0f;
        y[i] = normed * wi + bi;
    }
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

    // upsample.step: [MIMI_DIM, n_emb] -> [MIMI_DIM, n_up]. Stride 2 => 2 frames
    // out per input frame in steady state (n_up == 2).
    int n_up = mimi_upsample_step(d->upsample, d->emb_buf, n_emb, d->up_buf);

    // decoder_transformer.step: [MIMI_DIM, n_up] -> [MIMI_DIM, n_tr]. Causal KV
    // transformer preserves the frame count (n_tr == n_up).
    int n_tr = mimi_transformer_step(d->transformer, d->up_buf, n_up, d->tr_buf);

    // decoder(seanet).step: [MIMI_DIM, n_tr] -> pcm[1, n_pcm]. x960 upsample =>
    // n_pcm == n_tr * 960 in steady state (== MIMI_FRAME_OUT for n_tr == 2).
    int n_pcm = mimi_seanet_step(d->seanet, d->tr_buf, n_tr, pcm_out);

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
 *  candle_nn::LayerNorm                       | mimi_layer_norm_f32
 *  candle gelu_erf / Elu / softmax_last_dim   | mimi_gelu_erf_f32 / _elu_ / _softmax_
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
 * (d) PRIMITIVE-KERNEL LOOP ORDERS
 * --------------------------------
 *  gemv (y[M] = W[M,K]*x + b): outer m, inner k. NEON accumulates 4 lanes with
 *    vmlaq_f32 then vaddvq_f32 horizontal sum + scalar k-remainder + bias.
 *    Scalar `_ref` sums sequentially low->high k. -DMIMI_SCALAR_REF forces ref.
 *  gemm (C[M,N] = A[M,K]*B[K,N], beta 0=overwrite / 1=accumulate): i-k-j.
 *    B[k,:] and C[i,:] stream contiguously; NEON vectorizes j (vld/vmla/vst),
 *    beta==0 zeros the C row first (memset). Not a blocked/tiled gemm — first
 *    pass favors a documented, reproducible order over cache tricks; the AMX/
 *    cblas path (if adopted for the transformer) is a unit-4 decision.
 *  softmax (in place): pass 1 max, pass 2 expf(x-max) with f32 running sum,
 *    pass 3 multiply by 1/sum. Max-subtracted for stability.
 *  gelu_erf: 0.5*x*(1+erff(x*(1/sqrt2))), 1/sqrt2 = 0.70710678118654752440f.
 *  elu: x>0 ? x : alpha*(expf(x)-1).
 *  layer_norm: two-pass. pass 1 mean=sum/n; pass 2 var=sum((x-mean)^2)/n
 *    (BIASED, /n — matches candle_nn::LayerNorm); y=(x-mean)/sqrt(var+eps)*w+b,
 *    eps added to var BEFORE sqrt, eps supplied by the caller. NULL w/b => 1/0.
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
 *  4. Accumulation order for parity. gemv/gemm/layernorm use sequential f32
 *     accumulation; candle's CPU reduce order (and any NEON lane-sum vs scalar
 *     order in gemv) will differ at ULP level — expected under the "faithful"
 *     tier, bisect with -DMIMI_SCALAR_REF. The transformer softmax uses
 *     softmax_last_dim in candle; my max-subtracted 3-pass matches its formula.
 *  5. Quantizer emptiness. mimi_decoder_step assumes `codes` is a present single
 *     frame (true on our path). There is no ABI way to pass "empty codes" to
 *     mimi_quant_decode, so the None-codes arm of Rust decode_step is not
 *     reachable here; if a future caller needs it, add an n_in to the quant ABI.
 *  6. reset ordering. Reset order among decoder/transformer/upsample is
 *     immaterial (independent state) but mirrors the Rust for auditability.
 * ============================================================================
 */
