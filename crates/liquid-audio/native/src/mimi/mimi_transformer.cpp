// mimi_transformer.cpp — unit 4 of the Mimi decode port (docs/MIMI_PORT.md).
// Faithful C++/NEON port of the streaming DECODER transformer from
// moshi 0.6.4 src/transformer.rs, config (mimi.rs Config::v0_1):
//   d_model=512, heads=8 (head_dim 64), layers=8, causal=true, norm_first=true,
//   context=250, max_period=10000, LayerNorm (eps 1e-5), LayerScale (trained),
//   MLP NoGating linear1 -> gelu_erf -> linear2 (ff 2048), bias_ff=false,
//   bias_attn=false, kv_repeat=1, positional_embedding=Rope (rope_i,
//   interleaved), conv_layout=true, cross_attention=None, gating=None.
// Ported: LayerScale, RotaryEmbedding+Rope, StreamingMultiheadAttention
//   ::forward (streaming step), Mlp::NoGating, Norm::LayerNorm,
//   StreamingTransformerLayer::forward, StreamingTransformer::step (positional
//   bookkeeping + mask), ProjectedTransformer::step (projs None at 512<->512),
//   reset_state.
// Skipped (config says so): cross-attention, gating, RmsNorm, conv-block,
//   Sin positional embedding, batched paths, CaSrc, flash-attn.
// KV cache: the Standard (non-batched) path uses
//   candle_nn::kv_cache::RotatingKvCache (KvCache::new(2, context) in
//   transformer.rs:432) — NOT moshi's ScatteredKvCache (batched-only). This
//   unit owns a self-contained ring reproducing RotatingCache semantics
//   exactly; see /* NOTES */ (d) for the proof and the unit-5 reconciliation.
//
// Discipline: math is assembly at every step — GEMM/GEMV ride AMX via the
// header's Accelerate-backed mimi_gemv_f32/mimi_gemm_f32 (projections,
// attention scores, ws@v), every other sweep is NEON intrinsics as the
// primary path (rope rotation, score scale+mask locally; layernorm, softmax,
// gelu sweep, layer-scale, residual adds via the header NEON primitives).
// Scalar code exists only in the _ref parity siblings (MIMI_SCALAR_REF) and
// sub-vector tails. Zero allocation in steady state (state + scratch carved
// from MimiArena at init), POD state, f32 accumulate, documented orders.

#include "mimi_kernel.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(MIMI_SCALAR_REF)
#define MIMI_TR_NEON 1
#include <arm_neon.h>
#else
#define MIMI_TR_NEON 0
#endif

/* ---- fixed dimensions (transformer.rs Config via mimi.rs v0_1) ---------- */
enum {
    TR_D = MIMI_DIM,                    /* 512  d_model                      */
    TR_H = MIMI_TR_HEADS,               /* 8    num_heads                    */
    TR_HD = MIMI_DIM / MIMI_TR_HEADS,   /* 64   head_dim                     */
    TR_HD2 = TR_HD / 2,                 /* 32   rope pairs per head          */
    TR_L = MIMI_TR_LAYERS,              /* 8    num_layers                   */
    TR_CTX = MIMI_TR_CONTEXT,           /* 250  context == KV ring capacity  */
    TR_FF = MIMI_TR_FF,                 /* 2048 dim_feedforward              */
    TR_QKV = 3 * MIMI_DIM,              /* 1536 packed in_proj output        */
    TR_MAX_N = 8,                       /* scratch headroom; steady n is 1|2 */
};
static const float TR_EPS = 1e-5f;      /* Norm::new_shortcut LayerNorm eps  */
static const float TR_ATTN_SCALE = 0.125f; /* head_dim^-0.5 = 64^-0.5, exact */

/* ---- per-layer weights (zero-copy views into the checkpoint table) ------ */
struct TrLayer {
    /* self_attn: in_proj packed [1536,512] rows = [q(512) | k(512) | v(512)],
     * each block head-major (h*64+dd). transformer.rs:421 + reshape :458. */
    const uint8_t *in_proj_w; /* [1536, 512] f32 bytes */
    const uint8_t *out_proj_w;/* [512, 512], bias_attn=false */
    const uint8_t *norm1_w;   /* [512] */
    const uint8_t *norm1_b;   /* [512] */
    const uint8_t *norm2_w;   /* [512] */
    const uint8_t *norm2_b;   /* [512] */
    const uint8_t *ls1;       /* [512] layer_scale_1.scale */
    const uint8_t *ls2;       /* [512] layer_scale_2.scale */
    const uint8_t *lin1_w;    /* [2048, 512], bias_ff=false */
    const uint8_t *lin2_w;    /* [512, 2048], bias_ff=false */
    /* KV ring, RotatingCache slot order: [head][slot][dd], slots 0..TR_CTX-1.
     * K stored POST-rope (transformer.rs:469-474: rope applied, then append). */
    float *k_ring;            /* [TR_H][TR_CTX][TR_HD] */
    float *v_ring;            /* [TR_H][TR_CTX][TR_HD] */
};

/* ---- state (POD, carved from the arena) ---------------------------------
 * RotatingCache bookkeeping is SHARED across layers: every layer's cache in
 * the Rust model starts equal and appends the same t each step (lockstep), so
 * one (seq_len, ring_offset) pair represents all 8 — the mask/positions are
 * likewise computed once from layer 0's cache in forward_ca (:832-841). */
struct MimiTransformerState {
    TrLayer layers[TR_L];
    uint64_t seq_len;        /* RotatingCache.current_seq_len (pre-step)     */
    uint32_t ring_offset;    /* RotatingCache.offset — next write slot       */
    uint64_t last_reset_pos; /* StreamingTransformer.last_reset_pos, b=1;
                              * only reset_batch_idx ever raises it and the
                              * decode path never calls that; kept for the
                              * faithful mask formula (:851-854).            */
    /* rope */
    float *inv_freq;         /* [TR_HD2] 1/theta^(2j/64)                     */
    /* scratch (steady-state, no allocation) */
    float *xt;               /* [TR_MAX_N][TR_D] working activations [t,c]   */
    float *normb;            /* [TR_MAX_N][TR_D] norm1/norm2 output          */
    float *qkv;              /* [TR_MAX_N][TR_QKV] packed q|k|v, post-rope   */
    float *attn_cat;         /* [TR_MAX_N][TR_D] heads concatenated          */
    float *branch;           /* [TR_MAX_N][TR_D] out_proj / linear2 output   */
    float *mlp_hidden;       /* [TR_MAX_N][TR_FF]                            */
    float *scores;           /* [TR_H][TR_MAX_N][TR_CTX] score rows — banded
                              * per head so lanes never share a scratch row  */
    float *maskv;            /* [TR_MAX_N][TR_CTX] additive mask 0 / -inf    */
    float *rope_cos;         /* [TR_MAX_N][TR_HD2]                           */
    float *rope_sin;         /* [TR_MAX_N][TR_HD2]                           */
};

/* ---- local NEON kernels (with _ref parity siblings) --------------------- */

/* rope_i interleaved rotation over one 64-float head block, in place:
 *   y[2j]   = x[2j]*cos[j] - x[2j+1]*sin[j]
 *   y[2j+1] = x[2j]*sin[j] + x[2j+1]*cos[j]
 * Scalar ref keeps the four products in separate statements so clang cannot
 * contract into fma (rustc computes them unfused); the NEON path uses
 * explicit vmul/vsub/vadd (unfused) for the same reason. */
[[maybe_unused]] static void tr_rope_block_ref(float *hp, const float *crow,
                                               const float *srow) {
    for (int j = 0; j < TR_HD2; j++) {
        const float x0 = hp[2 * j];
        const float x1 = hp[2 * j + 1];
        const float c = crow[j];
        const float sn = srow[j];
        const float x0c = x0 * c;
        const float x1s = x1 * sn;
        const float x0s = x0 * sn;
        const float x1c = x1 * c;
        hp[2 * j] = x0c - x1s;
        hp[2 * j + 1] = x0s + x1c;
    }
}

static inline void tr_rope_block(float *hp, const float *crow, const float *srow) {
#if MIMI_TR_NEON
    for (int j = 0; j < TR_HD2; j += 4) { /* 32 pairs, no tail */
        float32x4x2_t x01 = vld2q_f32(hp + 2 * j); /* deinterleave x0|x1 */
        const float32x4_t c = vld1q_f32(crow + j);
        const float32x4_t s = vld1q_f32(srow + j);
        const float32x4_t x0c = vmulq_f32(x01.val[0], c);
        const float32x4_t x1s = vmulq_f32(x01.val[1], s);
        const float32x4_t x0s = vmulq_f32(x01.val[0], s);
        const float32x4_t x1c = vmulq_f32(x01.val[1], c);
        float32x4x2_t y;
        y.val[0] = vsubq_f32(x0c, x1s);
        y.val[1] = vaddq_f32(x0s, x1c);
        vst2q_f32(hp + 2 * j, y);
    }
#else
    tr_rope_block_ref(hp, crow, srow);
#endif
}

/* attention score row: sc[i] = sc[i] * 0.125f + mask[i] (0 / -inf).
 * candle: affine mul (exact, power of two) then broadcast_add — kept as
 * separate vmul/vadd, tail scalar. */
[[maybe_unused]] static void tr_scale_mask_ref(float *sc, const float *mrow,
                                               int k_len) {
    for (int i = 0; i < k_len; i++) {
        const float scaled = sc[i] * TR_ATTN_SCALE;
        sc[i] = scaled + mrow[i];
    }
}

static inline void tr_scale_mask(float *sc, const float *mrow, int k_len) {
#if MIMI_TR_NEON
    int i = 0;
    for (; i + 4 <= k_len; i += 4) {
        const float32x4_t scaled = vmulq_n_f32(vld1q_f32(sc + i), TR_ATTN_SCALE);
        vst1q_f32(sc + i, vaddq_f32(scaled, vld1q_f32(mrow + i)));
    }
    for (; i < k_len; i++) { /* sub-vector tail */
        const float scaled = sc[i] * TR_ATTN_SCALE;
        sc[i] = scaled + mrow[i];
    }
#else
    tr_scale_mask_ref(sc, mrow, k_len);
#endif
}

/* conv_layout boundary, n == 2 fast paths: [C,2] <-> [2,C] is a 512-wide
 * de/interleave (vld2q/vst2q). n == 1 is a straight copy; other n (cold,
 * priming-only shapes) fall back to the scalar movement loop. */
[[maybe_unused]] static void tr_transpose_in_ref(const float *x, float *xt,
                                                 int n, int t) {
    for (int c = 0; c < TR_D; c++) {
        const float *row = x + (size_t)c * (size_t)n;
        for (int tp = 0; tp < t; tp++) xt[(size_t)tp * TR_D + c] = row[tp];
    }
}

[[maybe_unused]] static void tr_transpose_out_ref(const float *xt, float *y,
                                                  int n, int t) {
    for (int c = 0; c < TR_D; c++) {
        float *row = y + (size_t)c * (size_t)n;
        for (int tp = 0; tp < t; tp++) row[tp] = xt[(size_t)tp * TR_D + c];
    }
}

static inline void tr_transpose_in(const float *x, float *xt, int n, int t) {
#if MIMI_TR_NEON
    if (n == 1) {
        memcpy(xt, x, TR_D * sizeof(float));
        return;
    }
    if (n == 2) {
        float *r0 = xt;
        float *r1 = xt + TR_D;
        for (int c = 0; c < TR_D; c += 4) {
            const float32x4x2_t v = vld2q_f32(x + 2 * c);
            vst1q_f32(r0 + c, v.val[0]);
            vst1q_f32(r1 + c, v.val[1]);
        }
        return;
    }
#endif
    tr_transpose_in_ref(x, xt, n, t);
}

static inline void tr_transpose_out(const float *xt, float *y, int n, int t) {
#if MIMI_TR_NEON
    if (n == 1) {
        memcpy(y, xt, TR_D * sizeof(float));
        return;
    }
    if (n == 2) {
        const float *r0 = xt;
        const float *r1 = xt + TR_D;
        for (int c = 0; c < TR_D; c += 4) {
            float32x4x2_t v;
            v.val[0] = vld1q_f32(r0 + c);
            v.val[1] = vld1q_f32(r1 + c);
            vst2q_f32(y + 2 * c, v);
        }
        return;
    }
#endif
    tr_transpose_out_ref(xt, y, n, t);
}

/* ---- init helpers -------------------------------------------------------- */

static void tr_err(char *err, size_t errlen, const char *msg, const char *name) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen, "mimi_transformer: %s %s", msg, name);
    }
}

/* find + shape-check a checkpoint tensor; d1 < 0 means 1-D [d0]. */
static const uint8_t *tr_find(const MimiWeightTable *w, const char *name,
                              int64_t d0, int64_t d1, char *err,
                              size_t errlen) {
    const MimiWeight *mw = mimi_weight_find(w, name);
    if (mw == NULL) {
        tr_err(err, errlen, "missing weight", name);
        return NULL;
    }
    int ok;
    if (d1 < 0) {
        ok = (mw->shape && mw->ndim == 1 && mw->shape[0] == d0 &&
              mw->len == (uint64_t)d0);
    } else {
        ok = (mw->shape && mw->ndim == 2 && mw->shape[0] == d0 &&
              mw->shape[1] == d1 &&
              mw->len == (uint64_t)d0 * (uint64_t)d1);
    }
    if (!ok || mw->bytes == NULL) {
        tr_err(err, errlen, "bad shape for weight", name);
        return NULL;
    }
    return mw->bytes;
}

/* ---- init ---------------------------------------------------------------- */

extern "C" int mimi_transformer_init(MimiTransformerState **st,
                                     const MimiWeightTable *w, MimiArena *a,
                                     char *err, size_t errlen) {
    if (st == NULL || w == NULL || a == NULL) {
        tr_err(err, errlen, "null argument", "");
        return 1;
    }
    MimiTransformerState *s =
        (MimiTransformerState *)mimi_arena_alloc(a, sizeof(MimiTransformerState));
    memset(s, 0, sizeof(*s));

    /* ProjectedTransformer::new (transformer.rs:970-1002): input_proj and
     * output_projs[0] are None because input_dim == output_dim == d_model
     * (512<->512); verified absent from the checkpoint. The identity hooks
     * live in step() at the conv_layout boundaries. */
    char nm[160];
    static const char *pfx = "decoder_transformer.transformer.layers";
    for (int l = 0; l < TR_L; l++) {
        TrLayer *L = &s->layers[l];
        snprintf(nm, sizeof nm, "%s.%d.self_attn.in_proj_weight", pfx, l);
        L->in_proj_w = tr_find(w, nm, TR_QKV, TR_D, err, errlen);
        if (L->in_proj_w == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.self_attn.out_proj.weight", pfx, l);
        L->out_proj_w = tr_find(w, nm, TR_D, TR_D, err, errlen);
        if (L->out_proj_w == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.norm1.weight", pfx, l);
        L->norm1_w = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->norm1_w == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.norm1.bias", pfx, l);
        L->norm1_b = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->norm1_b == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.norm2.weight", pfx, l);
        L->norm2_w = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->norm2_w == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.norm2.bias", pfx, l);
        L->norm2_b = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->norm2_b == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.layer_scale_1.scale", pfx, l);
        L->ls1 = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->ls1 == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.layer_scale_2.scale", pfx, l);
        L->ls2 = tr_find(w, nm, TR_D, -1, err, errlen);
        if (L->ls2 == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.linear1.weight", pfx, l);
        L->lin1_w = tr_find(w, nm, TR_FF, TR_D, err, errlen);
        if (L->lin1_w == NULL) return 1;
        snprintf(nm, sizeof nm, "%s.%d.linear2.weight", pfx, l);
        L->lin2_w = tr_find(w, nm, TR_D, TR_FF, err, errlen);
        if (L->lin2_w == NULL) return 1;

        const size_t ring = (size_t)TR_H * TR_CTX * TR_HD * sizeof(float);
        L->k_ring = (float *)mimi_arena_alloc(a, ring);
        L->v_ring = (float *)mimi_arena_alloc(a, ring);
        memset(L->k_ring, 0, ring);
        memset(L->v_ring, 0, ring);
    }

    /* RotaryEmbedding::new (transformer.rs:368-374): dim = head_dim = 64,
     * theta = max_period as f32 = 10000.0f;
     * inv_freq[j] = 1f32 / theta.powf((2j) as f32 / 64f32), j = 0..32.
     * (2j)/64 is exact in f32 (power-of-two divisor); powf is the platform
     * f32 pow, matching Rust f32::powf on this target. Init-time only. */
    s->inv_freq =
        (float *)mimi_arena_alloc_derived(a, TR_HD2 * sizeof(float));
    if (mimi_arena_building_derived(a)) {
        for (int j = 0; j < TR_HD2; j++) {
            s->inv_freq[j] = 1.0f /
                powf((float)MIMI_TR_MAX_PERIOD,
                     (float)(2 * j) / (float)TR_HD);
        }
    }

    s->xt = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->normb = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->qkv = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_QKV * sizeof(float));
    s->attn_cat = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->branch = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->mlp_hidden = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_FF * sizeof(float));
    s->scores = (float *)mimi_arena_alloc(
        a, (size_t)TR_H * TR_MAX_N * TR_CTX * sizeof(float));
    s->maskv = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_CTX * sizeof(float));
    s->rope_cos = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_HD2 * sizeof(float));
    s->rope_sin = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_HD2 * sizeof(float));

    s->seq_len = 0;
    s->ring_offset = 0;
    s->last_reset_pos = 0;
    *st = s;
    return 0;
}

/* ---- step -----------------------------------------------------------------
 * ProjectedTransformer::step (transformer.rs:1028-1046), conv_layout=true:
 *   x [C,T] -> transpose -> [T,C] -> (input_proj: None, identity)
 *   -> StreamingTransformer::step == forward == forward_ca(xs, None)
 *   -> (output_projs[0]: None, identity) -> transpose -> y [C,T].
 * StreamTensor None (n == 0) steps to None (0 frames, no state change). */
extern "C" int mimi_transformer_step(MimiTransformerState *st, const float *x,
                                     int n, float *y) {
    if (st == NULL || (n > 0 && (x == NULL || y == NULL))) return -1;
    if (n == 0) return 0; /* streaming.rs: empty in -> empty out */
    /* Steady-state n is 1 or 2 (upsample emits <=2 frames @25Hz per latent
     * frame). Scratch is sized TR_MAX_N; anything larger is a wiring bug.
     * n < TR_CTX also keeps the RotatingCache seq_len<max_seq_len branch and
     * the k-trim in transformer.rs:479-486 provably dead (see NOTES d). */
    if (n < 0 || n > TR_MAX_N) return -1;
    const int t = n;

    /* conv_layout boundary: [C,T] -> [T,C] (transpose(1,2), batch=1). */
    tr_transpose_in(x, st->xt, n, t);

    /* forward_ca positional bookkeeping (transformer.rs:824-876), computed
     * ONCE from the pre-step cache state and shared by all layers. */
    const uint64_t csl = st->seq_len;          /* current_seq_len, pre-append */
    const uint32_t off = st->ring_offset;      /* ring write index, pre-append */
    const uint64_t csl_after = csl + (uint64_t)t;
    const int k_len = csl_after < (uint64_t)TR_CTX ? (int)csl_after : TR_CTX;
    const uint32_t upd_off = (uint32_t)((off + (uint32_t)t) % TR_CTX);

    /* RotatingCache::positions(t) (candle-nn kv_cache.rs): absolute position
     * held by each cache slot AFTER this step's append, in slot order.
     * Bookkeeping (branchy, once per step) — candle builds this with a scalar
     * Vec loop too; not a math sweep. */
    int64_t ks[TR_CTX];
    for (int i = 0; i < k_len; i++) {
        int64_t pos = (int64_t)csl + t + i - (int64_t)upd_off;
        if ((uint32_t)i >= upd_off) pos -= TR_CTX;
        ks[i] = pos;
    }

    /* Mask (transformer.rs:836-868): slot allowed for query at absolute
     * position q_abs = csl + t_pos iff
     *   last_reset_pos <= ks[i] && ks[i] <= q_abs && q_abs <= ks[i] + context.
     * Rust skips building the mask when t == 1 && last_reset_pos <= min(ks);
     * in that case the predicate is all-true (ring holds positions
     * [csl_after - k_len, csl] only), so evaluating it uniformly is exact
     * (see NOTES d). Built once per step, shared by all layers/heads.
     * NEON int32-lane sweep; positions fit int32 until csl ~ 2^31 (guarded —
     * the rope f32 cast degrades far earlier anyway, NOTES b). */
    int mask_scalar = 1;
#if MIMI_TR_NEON
    if (csl_after + (uint64_t)TR_CTX < (uint64_t)INT32_MAX) {
        int32_t ks32[TR_CTX];
        for (int i = 0; i < k_len; i++) ks32[i] = (int32_t)ks[i];
        const int32x4_t lrp = vdupq_n_s32((int32_t)st->last_reset_pos);
        const int32x4_t ctx = vdupq_n_s32(TR_CTX);
        const float32x4_t zero = vdupq_n_f32(0.0f);
        const float32x4_t ninf = vdupq_n_f32(-INFINITY);
        for (int tp = 0; tp < t; tp++) {
            const int32x4_t q = vdupq_n_s32((int32_t)(csl + (uint64_t)tp));
            float *mrow = st->maskv + (size_t)tp * TR_CTX;
            int i = 0;
            for (; i + 4 <= k_len; i += 4) {
                const int32x4_t kv = vld1q_s32(ks32 + i);
                uint32x4_t ok = vandq_u32(vcgeq_s32(kv, lrp), vcleq_s32(kv, q));
                ok = vandq_u32(ok, vcgeq_s32(vaddq_s32(kv, ctx), q));
                vst1q_f32(mrow + i, vbslq_f32(ok, zero, ninf));
            }
            for (; i < k_len; i++) { /* sub-vector tail */
                const int64_t q_abs = (int64_t)csl + tp;
                const int allowed = (int64_t)st->last_reset_pos <= ks[i] &&
                                    ks[i] <= q_abs && q_abs <= ks[i] + TR_CTX;
                mrow[i] = allowed ? 0.0f : -INFINITY;
            }
        }
        mask_scalar = 0;
    }
#endif
    if (mask_scalar) { /* _ref / overflow fallback */
        for (int tp = 0; tp < t; tp++) {
            const int64_t q_abs = (int64_t)csl + tp;
            float *mrow = st->maskv + (size_t)tp * TR_CTX;
            for (int i = 0; i < k_len; i++) {
                const int allowed = (int64_t)st->last_reset_pos <= ks[i] &&
                                    ks[i] <= q_abs && q_abs <= ks[i] + TR_CTX;
                mrow[i] = allowed ? 0.0f : -INFINITY;
            }
        }
    }

    /* Rope tables (transformer.rs:871-876 + RotaryEmbedding::rope):
     * pos = arange(csl, csl+t) as u32 -> f32; freqs[tp][j] = pos * inv_freq[j];
     * cos/sin in f32. Table layout [t, 32] flat == rope_i's i_over_2 index.
     * NEON vmul for the angles, lane-wise libm cosf/sinf (faithful tier —
     * per the header, vector-polynomial transcendentals only enter later
     * behind the parity gate). */
    for (int tp = 0; tp < t; tp++) {
        const float p = (float)(uint32_t)(csl + (uint64_t)tp);
        float *crow = st->rope_cos + (size_t)tp * TR_HD2;
        float *srow = st->rope_sin + (size_t)tp * TR_HD2;
#if MIMI_TR_NEON
        for (int j = 0; j < TR_HD2; j += 4) { /* 32 lanes, no tail */
            float ang[4];
            vst1q_f32(ang, vmulq_n_f32(vld1q_f32(st->inv_freq + j), p));
            crow[j] = cosf(ang[0]);
            crow[j + 1] = cosf(ang[1]);
            crow[j + 2] = cosf(ang[2]);
            crow[j + 3] = cosf(ang[3]);
            srow[j] = sinf(ang[0]);
            srow[j + 1] = sinf(ang[1]);
            srow[j + 2] = sinf(ang[2]);
            srow[j + 3] = sinf(ang[3]);
        }
#else
        for (int j = 0; j < TR_HD2; j++) {
            const float ang = p * st->inv_freq[j];
            crow[j] = cosf(ang);
            srow[j] = sinf(ang);
        }
#endif
    }

    /* ---- layers (StreamingTransformerLayer::forward, norm_first=true) ---- */
    for (int l = 0; l < TR_L; l++) {
        TrLayer *L = &st->layers[l];

        /* norm1 (candle_nn LayerNorm fast path, see NOTES f) */
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_layer_norm_f32(
                st->xt + (size_t)tp * TR_D, L->norm1_w, L->norm1_b,
                st->normb + (size_t)tp * TR_D, TR_D, TR_EPS);
        }

        /* in_proj (AMX gemv): packed qkv = norm1(x) @ W^T, rows [q|k|v]
         * head-major. qkv[tp][s*512 + h*64 + dd] == candle reshape (b,t,3,h,d). */
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_gemv_f32(
                L->in_proj_w, st->normb + (size_t)tp * TR_D, NULL,
                st->qkv + (size_t)tp * TR_QKV, TR_QKV, TR_D);
        }

        /* rope_i on q and k blocks, in place (NEON, see tr_rope_block).
         * v untouched. */
        for (int tp = 0; tp < t; tp++) {
            const float *crow = st->rope_cos + (size_t)tp * TR_HD2;
            const float *srow = st->rope_sin + (size_t)tp * TR_HD2;
            for (int qk = 0; qk < 2; qk++) { /* 0: q block, 1: k block */
                float *blk = st->qkv + (size_t)tp * TR_QKV + (size_t)qk * TR_D;
                for (int h = 0; h < TR_H; h++) {
                    tr_rope_block(blk + (size_t)h * TR_HD, crow, srow);
                }
            }
        }

        /* kv_cache.append (RotatingCache::append, seq_len < max_seq_len
         * branch): write each new frame at slot (off + i) % TR_CTX. All
         * layers use the same pre-step offset (lockstep caches). */
        for (int i = 0; i < t; i++) {
            const uint32_t slot = (off + (uint32_t)i) % TR_CTX;
            const float *krow = st->qkv + (size_t)i * TR_QKV + TR_D;
            const float *vrow = st->qkv + (size_t)i * TR_QKV + 2 * TR_D;
            for (int h = 0; h < TR_H; h++) {
                memcpy(L->k_ring + ((size_t)h * TR_CTX + slot) * TR_HD,
                       krow + (size_t)h * TR_HD, TR_HD * sizeof(float));
                memcpy(L->v_ring + ((size_t)h * TR_CTX + slot) * TR_HD,
                       vrow + (size_t)h * TR_HD, TR_HD * sizeof(float));
            }
        }

        /* attention (transformer.rs:495-511):
         *   pre_ws = q @ k^T * head_dim^-0.5 (+ mask) ; softmax ; ws @ v.
         * k slots 0..k_len-1 in storage order == candle's ad.narrow(0,k_len)
         * (its k-trim :479-486 is dead for t < context, see NOTES d). */
        for (int h = 0; h < TR_H; h++) {
            const float *kr = L->k_ring + (size_t)h * TR_CTX * TR_HD;
            const float *vr = L->v_ring + (size_t)h * TR_CTX * TR_HD;
            for (int tp = 0; tp < t; tp++) {
                const float *qh = st->qkv + (size_t)tp * TR_QKV + (size_t)h * TR_HD;
                float *sc = st->scores + ((size_t)h * TR_MAX_N + (size_t)tp) * TR_CTX;
                /* scores row = K[0..k_len) @ q — AMX gemv over the
                 * contiguous ring rows. */
                mimi_gemv_f32(kr, qh, NULL, sc, k_len, TR_HD);
                /* scale + additive mask (NEON sweep) */
                tr_scale_mask(sc, st->maskv + (size_t)tp * TR_CTX, k_len);
                /* softmax_last_dim over the k_len window (header NEON) */
                mimi_softmax_f32(sc, k_len);
                /* ws @ v == C[1x64] = A[1xk_len] B[k_len x 64] — AMX gemm,
                 * beta=0 writes the head section of attn_cat directly. */
                mimi_gemm_f32(sc, vr,
                              st->attn_cat + (size_t)tp * TR_D + (size_t)h * TR_HD,
                              1, k_len, TR_HD, 0);
            }
        }

        /* out_proj (AMX gemv, no bias), then
         * x = x + layer_scale_1 * attn (header NEON sweeps; candle order:
         * broadcast_mul then add — two separate sweeps, no fma). */
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_gemv_f32(
                L->out_proj_w, st->attn_cat + (size_t)tp * TR_D, NULL,
                st->branch + (size_t)tp * TR_D, TR_D, TR_D);
        }
        for (int tp = 0; tp < t; tp++) {
            float *xr = st->xt + (size_t)tp * TR_D;
            float *br = st->branch + (size_t)tp * TR_D;
            mimi_weight_scale_vec_f32(br, L->ls1, br, TR_D);
            mimi_add_vec_f32(xr, br, xr, TR_D);       /* residual, in place  */
        }

        /* mlp branch: x = x + layer_scale_2 * linear2(gelu_erf(linear1(norm2(x)))) */
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_layer_norm_f32(
                st->xt + (size_t)tp * TR_D, L->norm2_w, L->norm2_b,
                st->normb + (size_t)tp * TR_D, TR_D, TR_EPS);
        }
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_gemv_f32(
                L->lin1_w, st->normb + (size_t)tp * TR_D, NULL,
                st->mlp_hidden + (size_t)tp * TR_FF, TR_FF, TR_D);
        }
        /* gelu_erf sweep (header NEON, lane-wise erff) */
        mimi_gelu_erf_vec_f32(st->mlp_hidden, st->mlp_hidden, t * TR_FF);
        for (int tp = 0; tp < t; tp++) {
            mimi_weight_gemv_f32(
                L->lin2_w, st->mlp_hidden + (size_t)tp * TR_FF, NULL,
                st->branch + (size_t)tp * TR_D, TR_D, TR_FF);
        }
        for (int tp = 0; tp < t; tp++) {
            float *xr = st->xt + (size_t)tp * TR_D;
            float *br = st->branch + (size_t)tp * TR_D;
            mimi_weight_scale_vec_f32(br, L->ls2, br, TR_D);
            mimi_add_vec_f32(xr, br, xr, TR_D);
        }
    }

    /* advance the shared cache bookkeeping once (all layers appended t) */
    st->seq_len = csl_after;
    st->ring_offset = upd_off;

    /* conv_layout boundary out: [T,C] -> [C,T]. y may alias x. */
    tr_transpose_out(st->xt, y, n, t);
    return n;
}

/* ---- reset ----------------------------------------------------------------
 * StreamingTransformer::reset_state (transformer.rs:946-949): clears
 * last_reset_pos and resets every layer's RotatingKvCache (offset=0,
 * current_seq_len=0, data dropped — recreated zeroed on next append; we zero
 * in place so hibernated state is deterministic). */
extern "C" void mimi_transformer_reset(MimiTransformerState *st) {
    if (st == NULL) return;
    st->seq_len = 0;
    st->ring_offset = 0;
    st->last_reset_pos = 0;
    const size_t ring = (size_t)TR_H * TR_CTX * TR_HD * sizeof(float);
    for (int l = 0; l < TR_L; l++) {
        memset(st->layers[l].k_ring, 0, ring);
        memset(st->layers[l].v_ring, 0, ring);
    }
}

/* NOTES
 * =====
 *
 * (a) Rust -> C++ mapping (moshi 0.6.4 src/transformer.rs unless noted)
 *   LayerScale::forward (:90-94)            -> mimi_scale_vec_f32(branch, ls,
 *                                              branch) — separate sweep before
 *                                              the residual add (= candle's
 *                                              broadcast_mul then add, no fma).
 *   RotaryEmbedding::new (:368-374)         -> inv_freq[] in init.
 *   RotaryEmbedding::rope (:376-384)        -> per-step rope_cos/rope_sin
 *                                              tables ([t,32], 1-D pos branch).
 *   Rope::apply_rotary_emb (:355-359)       -> tr_rope_block on q,k head
 *                                              blocks in place (input already
 *                                              f32; the to_dtype(F32)
 *                                              round-trip is identity here).
 *   StreamingMultiheadAttention::forward
 *     (:445-513)                            -> per-layer attention section of
 *                                              mimi_transformer_step.
 *   KvCache::append / positions
 *     (kv_cache.rs:242-252 -> candle-nn
 *      kv_cache.rs RotatingCache)           -> k_ring/v_ring + ks[] block (see d).
 *   Mlp::NoGating::forward (:565)           -> linear1 gemv ->
 *                                              mimi_gelu_erf_vec_f32 sweep ->
 *                                              linear2 gemv.
 *   Norm::LayerNorm forward (:601-618,
 *     eps 1e-5 from Norm::new_shortcut :639) -> mimi_layer_norm_f32(..., 1e-5f).
 *   StreamingTransformerLayer::forward
 *     (:731-758, norm_first)                -> layer loop body (attn residual,
 *                                              then mlp residual; cross_attn
 *                                              arm dead, config None).
 *   StreamingTransformer::step/forward/
 *     forward_ca (:816-898, :951-957)       -> positional bookkeeping + mask +
 *                                              rope tables + layer loop.
 *   ProjectedTransformer::step (:1028-1046) -> tr_transpose_in/out at the
 *                                              boundaries; input_proj /
 *                                              output_projs[0] are None
 *                                              (512<->512) — identity hooks
 *                                              are the [T,C] buffer itself.
 *   reset_state (:946-949)                  -> mimi_transformer_reset.
 *   Skipped: XaGate, StreamingMultiheadCrossAttention, Mlp::Gating, RmsNorm,
 *   PositionalEmbedding::Sin, CaSrc, flash-attn, batched Transformer::Batched,
 *   copy_state, reset_batch_idx (unreachable from the non-batched decode path;
 *   last_reset_pos kept in state so the mask formula stays literal).
 *
 * (b) rope_i exact formula implemented (candle-nn 0.9.2 rotary_emb.rs, CPU
 *     path of the RotaryEmbI custom op — INTERLEAVED variant):
 *       for pair j in 0..head_dim/2, position index tp:
 *         y[2j]   = x[2j]*cos[tp][j] - x[2j+1]*sin[tp][j]
 *         y[2j+1] = x[2j]*sin[tp][j] + x[2j+1]*cos[tp][j]
 *     cos/sin table (RotaryEmbedding::rope, 1-D pos):
 *         cos[tp][j] = cosf(pos_tp * inv_freq[j]),
 *         inv_freq[j] = 1.0f / powf(10000.0f, (2j)/64.0f)   (all f32)
 *         pos_tp = (float)(uint32_t)(current_seq_len + tp)  (Rust arange u32
 *                  -> F32 cast; exact while positions < 2^24 ~ 7.7 days of
 *                  25 Hz stream).
 *     cos/sin here is 2-D [t, 32] so rope_i's flat index i_over_2 =
 *     tp*32 + j (the "unbatched_rope" 3-D branch is not taken). Applied to q
 *     and k only, before the KV append, exactly as transformer.rs:469-474.
 *     NEON: vld2q deinterleaves the (x0,x1) pairs; four vmulq then
 *     vsub/vadd — deliberately UNFUSED (no vfma) because rustc computes the
 *     rotation unfused; scalar _ref keeps the products in separate statements
 *     for the same reason.
 *
 * (c) conv_layout boundary handling (ProjectedTransformer::step, conv_layout
 *     = true): input [B,C,T] -> transpose(1,2) -> [B,T,C] before the
 *     transformer, transpose back after the (identity) output proj. With the
 *     header's [MIMI_DIM, n] conv layout and batch 1 this is
 *       xt[tp*512 + c] = x[c*n + tp]  on entry,
 *       y[c*n + tp] = xt[tp*512 + c]  on exit.
 *     Data movement, not math: n==1 is memcpy, n==2 is a NEON
 *     vld2q/vst2q de/interleave (the steady-state shapes); other n (cold
 *     priming shapes only) take the scalar movement loop. All internal math
 *     is [t, c] row-major (candle's b,t,c with b=1).
 *
 * (d) KV interface: THIS UNIT OWNS THE CACHE (arbiter-settled; the mimi_kv_*
 *     ABI is parked and never called from here).
 *     Why: transformer.rs:17 imports crate::kv_cache::KvCache, and
 *     kv_cache.rs:220 defines it as an enum wrapping
 *     candle_nn::kv_cache::RotatingKvCache; transformer.rs:432 builds
 *     `KvCache::new(2, context)` — dim 2 is the seq axis of candle's
 *     [b,h,t,hd]. moshi's ScatteredKvCache is only reachable via
 *     crate::batched_transformer (out of scope). MimiDetokenizer builds the
 *     non-batched Mimi (Transformer::new with batch_size None -> Standard).
 *     Layout mapping: candle's per-layer K/V [1, 8, slot<=250, 64] with the
 *     ring on dim 2 becomes k_ring/v_ring [head][slot][dd] (head-major so a
 *     head is one contiguous 250x64 band; slot == candle's dim-2 index,
 *     unchanged).
 *     Semantics reproduced from candle-nn 0.9.2 kv_cache.rs::RotatingCache
 *     (:180-335, wrapped by RotatingKvCache :336; append :372,
 *     current_seq_len :382), specialized to per-step t < 250:
 *       - append: write t frames at slots (offset+i) % 250; offset advances
 *         mod 250; current_seq_len += t. Returned k/v = first
 *         min(current_seq_len, 250) slots in SLOT (storage) order — scores,
 *         softmax and ws@v all run in slot order, matching the tensor candle
 *         hands to matmul/softmax_last_dim. (candle's two-copy boundary
 *         split in append is the same writes as our per-frame modulo slots.)
 *       - ORDERING (transformer.rs:824-876): current_seq_len is read BEFORE
 *         append; rope positions arange(csl, csl+t) and the mask both come
 *         from that pre-append read; appends happen inside the layer loop.
 *         This file mirrors that: csl/off snapshot -> rope tables + mask ->
 *         per-layer appends -> advance (seq_len, ring_offset) once at end.
 *       - positions(t) (candle :296-312, called pre-append, describes
 *         post-append state): upd_offset = (offset+t) % 250;
 *         out_len = min(csl+t, 250); slot i holds absolute position
 *         csl + t + i - upd_offset, minus 250 when i >= upd_offset.
 *         PRE-FILL (csl+t <= 250, offset == csl): upd_offset == csl+t, every
 *         i < upd_offset, so ks[i] = i — identity, slots in chronological
 *         order. POST-WRAP (csl+t > 250): slots split into a young run
 *         [0, upd_offset) holding positions csl+t-upd_offset .. csl+t-1 and
 *         an old run [upd_offset, 250) holding csl+t-250+... — worked example
 *         around the 250 boundary: csl=249, off=249, t=2 -> csl_after=251,
 *         upd_offset=(249+2)%250=1, k_len=250;
 *           slot 0 (i=0 < 1):  ks[0] = 249+2+0-1       = 250 (new frame #2)
 *           slot 1 (i=1 >= 1): ks[1] = 249+2+1-1-250   = 1
 *           ...
 *           slot 249:          ks[249] = 249+2+249-1-250 = 249 (frame #1 of
 *           this step landed at slot 249, position 249's slot)
 *         i.e. position 0's slot was overwritten by position 250; the window
 *         is [csl+t-250, csl+t-1] = [1, 250]. Ring writes for that step:
 *         frame i=0 -> slot (249+0)%250 = 249, frame i=1 -> slot 0.
 *       - the seq_len >= max_seq_len append branch and the k-trim in
 *         transformer.rs:479-486 are dead: k_len = min(csl+t, 250) gives
 *         k_len - t <= 250 - t < context, so k_target_len == k_len; guarded
 *         by n <= TR_MAX_N (8).
 *       - mask (forward_ca :836-868): allowed iff last_reset_pos <= ks[i] &&
 *         ks[i] <= csl+t_pos && csl+t_pos <= ks[i]+250, materialized as
 *         additive 0/-inf and added AFTER the 0.125 scale (broadcast_add
 *         order preserved). t==1 FAST PATH as implemented: Rust returns
 *         mask=None iff t == 1 && all(last_reset_pos[b] <= min(ks)) (:842);
 *         at batch=1 that is `t == 1 && last_reset_pos <= min(ks)`. This
 *         port evaluates the predicate uniformly instead of special-casing:
 *         when the fast-path condition holds the predicate is provably
 *         all-true (window is [csl+t-k_len, csl] with every ks[i] <= q_abs =
 *         csl and q_abs <= ks[i]+250), and an all-zero additive mask is
 *         bit-identical to no mask. last_reset_pos: reset_state clears it to
 *         0 (via last_reset_pos.clear() + resize on next forward); only
 *         reset_batch_idx(:933-942) raises it (to current_seq_len) and the
 *         decode path never calls that — kept as a scalar in state so the
 *         formula stays literal.
 *       - bookkeeping (seq_len, ring_offset) is stored ONCE for all 8 layers:
 *         the per-layer Rust caches start equal and append the same t every
 *         step (lockstep); forward_ca already reads only layer 0's cache for
 *         positions/mask/rope.
 *     reset(): offset=0, current_seq_len=0, rings zeroed (candle drops the
 *     buffer and re-zeros on next append; never exposed pre-fill either way).
 *
 * (e) weight names + shapes (verified against
 *     tokenizer-e351c8d8-checkpoint125.safetensors, prefix
 *     decoder_transformer.transformer.layers.{0..7}.):
 *       self_attn.in_proj_weight [1536, 512]  packed rows q|k|v, each block
 *                                             head-major (h*64+dd); split by
 *                                             reshape (b,t,3,h,64) — one
 *                                             packed weight, NOT separate
 *                                             q/k/v (transformer.rs:421,458).
 *       self_attn.out_proj.weight [512, 512]  (bias_attn=false: no biases)
 *       norm1.weight/.bias [512], norm2.weight/.bias [512]
 *                                             (LayerNorm::new takes the
 *                                             "weight" key — no "alpha" in
 *                                             this checkpoint)
 *       linear1.weight [2048, 512], linear2.weight [512, 2048] (bias_ff=false)
 *       layer_scale_1.scale [512], layer_scale_2.scale [512]
 *     decoder_transformer.input_proj / output_projs absent (512<->512) —
 *     confirmed not in the checkpoint listing. No repack/transpose at init:
 *     all weights consumed in checkpoint layout by the AMX gemv.
 *
 * (f) reduction orders / primitive semantics this unit depends on
 *     - LayerNorm: candle takes ops::layer_norm's CPU fast path (contiguous
 *       f32, remove_mean, bias present — candle-nn 0.9.2 ops.rs LayerNorm
 *       CustomOp3): ONE sequential pass accumulating sum and sum2 in f32
 *       (index order), mean = sum/512, var = sum2/512 - mean*mean (naive, NOT
 *       Welford/two-pass), inv_std = 1/sqrtf(var + 1e-5f), out =
 *       (x-mean)*inv_std*w + b with that exact op order. mimi_layer_norm_f32
 *       (unit 6) MUST implement this formula; its NEON sum/sum2 lane blocking
 *       is the faithful-tier freedom.
 *     - softmax (candle-nn ops.rs SoftmaxLastDim cpu_fwd): max scan (Rust
 *       f32::max == fmaxf semantics), d = expf(s - max) lane-wise, sum via
 *       candle's NEON-blocked vec_sum, then d /= sum (division, NOT
 *       multiply-by-reciprocal). mimi_softmax_f32 (unit 6) must follow.
 *     - attention scores: one dot per kv slot via mimi_gemv_f32
 *       (K[0..k_len) x q over contiguous ring rows) — AMX/cblas; candle runs
 *       the gemm crate (blocked, fma). Scale-then-mask per element as a NEON
 *       sweep: (dot * 0.125f) + mask with separate vmul/vadd — 0.125 exact,
 *       adding 0/-inf exact, so contraction would be harmless, but unfused
 *       matches candle's two tensor ops literally.
 *     - ws @ v: mimi_gemm_f32(sc [1 x k_len], V[k_len x 64], out, beta=0) —
 *       AMX/cblas accumulation order (unit 6's ledger); candle: gemm crate.
 *     - projections/MLP: per-token mimi_gemv_f32 (y[m] = sum_k w[m,k]x[k]);
 *       candle Linear is xs.matmul(w.t()) via the gemm crate — blocked and
 *       fma'd, so the cblas order is the faithful-tier freedom.
 *     - gelu_erf (candle-core op.rs GeluErf::f32): ((erff(x * (1/sqrt(2)))
 *       + 1.0f) * 0.5f) * x, all f32, candle calls the Rust libm crate's
 *       erff. mimi_gelu_erf_vec_f32 (unit 6) must be that exact expression
 *       lane-wise (system erff — see g.1); in-place x==y must be legal.
 *     - residual/LayerScale: mimi_scale_vec_f32 then mimi_add_vec_f32 — two
 *       separate elementwise NEON sweeps in that order (candle: broadcast_mul
 *       kernel, then add kernel; rustc does not contract, so the sweeps must
 *       not fuse mul+add into one fma either). Same-index in-place aliasing
 *       (y == x for scale, y == a for add) must be legal for these prims —
 *       ABI friction note for the arbiter: the header doesn't state it.
 *     - rope: local tr_rope_block (NEON vld2q/vst2q, unfused vmul/vsub/vadd),
 *       _ref sibling under MIMI_SCALAR_REF; table build is NEON vmul +
 *       lane-wise cosf/sinf.
 *     - mask fill: NEON int32-lane sweep (vcge/vcle/vbsl selecting 0/-inf),
 *       scalar tail; positions are cast to int32 under a
 *       csl+t+250 < INT32_MAX guard (scalar int64 fallback beyond — the
 *       rope f32 cast has degraded ~2^24 long before that matters). The
 *       ks[]/positions bookkeeping itself stays scalar: branchy integer
 *       control-plane work that Rust also does with a scalar Vec loop.
 *
 * (g) uncertainties
 *     1. Transcendental ulp drift: Rust f32::sin/cos/powf lower to the
 *        platform libm on aarch64-apple-darwin, so rope tables should match
 *        bit-for-bit in practice, but candle's erff is the Rust `libm` crate
 *        (portable Musl-derived polynomial) while ours is Apple libm erff —
 *        expect <=1-2 ulp differences feeding gelu; covered by the ulp-band
 *        harness, flagged here for the threshold ledger.
 *     2. AMX/cblas vs candle's gemm crate: different blocking and fma
 *        schedules — per-GEMM ulp-band drift is expected and is the accepted
 *        faithful-tier cost (manifest: "candle's blocked gemm is not
 *        economically bit-reproducible"). Bisect path: MIMI_SCALAR_REF
 *        builds + unit 6's _ref gemv/gemm.
 *     3. This unit's parity also depends on unit 6 honoring (f) exactly —
 *        especially layer_norm's naive one-pass sum/sum2 and softmax's
 *        divide-by-sum. If unit 6 shipped Welford or two-pass mean/var,
 *        layer 0's output already drifts.
 *     4. Scale factor: candle multiplies pre_ws by (head_dim as f64)^-0.5
 *        as an affine op (f64 scalar cast to f32) = 0.125 exactly — no
 *        uncertainty, noted only because the f64 detour looks lossy but isn't.
 *     5. TR_MAX_N = 8 headroom: steady-state n is 1 or 2 from the stride-2
 *        upsampler; if some priming path ever hands >8 frames in one step the
 *        unit returns -1 rather than silently splitting (no-fallbacks rule).
 *        The Rust path would handle it; revisit if the harness ever trips it.
 *     6. mimi_gemm_f32 with m=1 per (head, t_pos) means 16-32 tiny cblas
 *        calls per layer; if Accelerate dispatch overhead shows up in the
 *        profile, the batched alternative (one [t*? x k_len] gemm per head)
 *        changes accumulation shape — numerics re-measure required, so it
 *        waits for the parity harness.
 *     7. y aliasing x in mimi_transformer_step is safe (output written from
 *        the xt scratch after all reads), matching the header's in-place
 *        allowance.
 *
 * (h) kcoro lane-banding map (arbiter's integration cut; the single-call step
 *     API is unchanged). Written with no incidental cross-band sequential
 *     dependence — per stage, the natural band axis and the shared state:
 *       norm1/norm2:      band = token row tp (<=2) or split the 512-lane
 *                         channel sweep inside mimi_layer_norm_f32 (its
 *                         reduction is the constraint); reads xt row, writes
 *                         disjoint normb rows.
 *       in_proj gemv:     band = output rows m of [1536,512] (ST_-style row
 *                         bands), or keep each whole gemv on one lane per
 *                         token — AMX calls likely stay whole per lane (one
 *                         cblas call per band is the sane cut; splitting an
 *                         AMX gemv across lanes buys nothing). Reads normb
 *                         (shared, read-only), writes disjoint qkv spans.
 *       rope:             band = head h (x2 for q|k blocks x t rows = up to
 *                         32 independent tr_rope_block calls); disjoint
 *                         64-float blocks in qkv, rope tables read-only.
 *       kv append:        band = head h — disjoint ring bands
 *                         (k_ring/v_ring are head-major precisely so a head
 *                         band is one contiguous 250x64 region per lane).
 *       attention:        band = head h (8 heads -> 8 lanes, the natural
 *                         cut): each head touches only its ring band, its
 *                         own scores row (scores is [head][row][ctx] — NOT
 *                         shared across heads, privatized already), and its
 *                         own 64-wide attn_cat section. maskv/rope tables/
 *                         qkv are read-only shared.
 *       out_proj gemv:    band = output rows of [512,512] or whole-per-lane
 *                         (AMX, same call as in_proj); reads attn_cat.
 *       LayerScale+resid: band = token row (or channel sub-ranges of the
 *                         sweeps); xt row is the only read-write target.
 *       MLP:              band = output rows (linear1 [2048,512] rows band
 *                         like the engine's ST_ stages; gelu sweep bands on
 *                         element ranges; linear2 [512,2048] rows likewise).
 *     Per-lane privatization needed if banded: NONE of the persistent state
 *     (rings are written disjointly by head; seq_len/ring_offset advance is
 *     single-writer at the pass boundary — keep it on the closing lane after
 *     the barrier, exactly the two-barrier doctrine). Scratch is already
 *     banded: scores per head; normb/qkv/attn_cat/branch/mlp_hidden are
 *     written in disjoint spans per band. The only sequential dependences are
 *     the REAL ones: layer l -> l+1, and within a layer norm -> proj -> rope
 *     -> append -> attn -> out_proj -> residual -> norm2 -> mlp -> residual
 *     (stage fences, not lane fences).
 */
