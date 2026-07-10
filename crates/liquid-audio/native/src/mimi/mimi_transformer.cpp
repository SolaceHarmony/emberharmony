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
// Discipline: zero allocation in steady state (all state + scratch carved
// from MimiArena at init), POD state, f32 accumulate, documented reduction
// orders, NEON with scalar _ref siblings under MIMI_SCALAR_REF.

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
    const float *in_proj_w;   /* [1536, 512] */
    const float *out_proj_w;  /* [512, 512], bias_attn=false */
    const float *norm1_w;     /* [512] */
    const float *norm1_b;     /* [512] */
    const float *norm2_w;     /* [512] */
    const float *norm2_b;     /* [512] */
    const float *ls1;         /* [512] layer_scale_1.scale */
    const float *ls2;         /* [512] layer_scale_2.scale */
    const float *lin1_w;      /* [2048, 512], bias_ff=false */
    const float *lin2_w;      /* [512, 2048], bias_ff=false */
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
    float *scores;           /* [TR_MAX_N][TR_CTX] per-head attn row (reused)*/
    float *maskv;            /* [TR_MAX_N][TR_CTX] additive mask 0 / -inf    */
    float *rope_cos;         /* [TR_MAX_N][TR_HD2]                           */
    float *rope_sin;         /* [TR_MAX_N][TR_HD2]                           */
};

/* ---- helpers ------------------------------------------------------------ */

static void tr_err(char *err, size_t errlen, const char *msg, const char *name) {
    if (err != NULL && errlen > 0) {
        snprintf(err, errlen, "mimi_transformer: %s %s", msg, name);
    }
}

/* find + shape-check a checkpoint tensor; d1 < 0 means 1-D [d0]. */
static const float *tr_find(const MimiWeightTable *w, const char *name,
                            int64_t d0, int64_t d1, char *err, size_t errlen) {
    const MimiWeight *mw = mimi_weight_find(w, name);
    if (mw == NULL) {
        tr_err(err, errlen, "missing weight", name);
        return NULL;
    }
    int ok;
    if (d1 < 0) {
        ok = (mw->ndim == 1 && mw->shape[0] == d0);
    } else {
        ok = (mw->ndim == 2 && mw->shape[0] == d0 && mw->shape[1] == d1);
    }
    if (!ok || mw->data == NULL) {
        tr_err(err, errlen, "bad shape for weight", name);
        return NULL;
    }
    return mw->data;
}

/* acc[0..64) += s * v[0..64) — the ws@v accumulation (per kv slot).
 * Scalar reference: sequential over dd; per-element mul then add. */
#if defined(MIMI_SCALAR_REF) || !MIMI_TR_NEON
static void tr_axpy64_ref(float *acc, float s, const float *v) {
    for (int dd = 0; dd < TR_HD; dd++) {
        float p = s * v[dd];
        acc[dd] = acc[dd] + p;
    }
}
#endif

static inline void tr_axpy64(float *acc, float s, const float *v) {
#if MIMI_TR_NEON
    /* 4x unrolled q-register fma: lanes dd, dd+1, dd+2, dd+3 accumulate
     * independently; slots accumulate sequentially (outer caller loop). */
    float32x4_t vs = vdupq_n_f32(s);
    for (int dd = 0; dd < TR_HD; dd += 16) {
        float32x4_t a0 = vld1q_f32(acc + dd);
        float32x4_t a1 = vld1q_f32(acc + dd + 4);
        float32x4_t a2 = vld1q_f32(acc + dd + 8);
        float32x4_t a3 = vld1q_f32(acc + dd + 12);
        a0 = vfmaq_f32(a0, vs, vld1q_f32(v + dd));
        a1 = vfmaq_f32(a1, vs, vld1q_f32(v + dd + 4));
        a2 = vfmaq_f32(a2, vs, vld1q_f32(v + dd + 8));
        a3 = vfmaq_f32(a3, vs, vld1q_f32(v + dd + 12));
        vst1q_f32(acc + dd, a0);
        vst1q_f32(acc + dd + 4, a1);
        vst1q_f32(acc + dd + 8, a2);
        vst1q_f32(acc + dd + 12, a3);
    }
#else
    tr_axpy64_ref(acc, s, v);
#endif
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

        size_t ring = (size_t)TR_H * TR_CTX * TR_HD * sizeof(float);
        L->k_ring = (float *)mimi_arena_alloc(a, ring);
        L->v_ring = (float *)mimi_arena_alloc(a, ring);
        memset(L->k_ring, 0, ring);
        memset(L->v_ring, 0, ring);
    }

    /* RotaryEmbedding::new (transformer.rs:368-374): dim = head_dim = 64,
     * theta = max_period as f32 = 10000.0f;
     * inv_freq[j] = 1f32 / theta.powf((2j) as f32 / 64f32), j = 0..32.
     * (2j)/64 is exact in f32 (power-of-two divisor); powf is the platform
     * f32 pow, matching Rust f32::powf on this target. */
    s->inv_freq = (float *)mimi_arena_alloc(a, TR_HD2 * sizeof(float));
    for (int j = 0; j < TR_HD2; j++) {
        s->inv_freq[j] =
            1.0f / powf((float)MIMI_TR_MAX_PERIOD, (float)(2 * j) / (float)TR_HD);
    }

    s->xt = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->normb = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->qkv = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_QKV * sizeof(float));
    s->attn_cat = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->branch = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_D * sizeof(float));
    s->mlp_hidden = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_FF * sizeof(float));
    s->scores = (float *)mimi_arena_alloc(a, (size_t)TR_MAX_N * TR_CTX * sizeof(float));
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
    for (int c = 0; c < TR_D; c++) {
        const float *row = x + (size_t)c * (size_t)n;
        for (int tp = 0; tp < t; tp++) st->xt[(size_t)tp * TR_D + c] = row[tp];
    }

    /* forward_ca positional bookkeeping (transformer.rs:824-876), computed
     * ONCE from the pre-step cache state and shared by all layers. */
    const uint64_t csl = st->seq_len;          /* current_seq_len, pre-append */
    const uint32_t off = st->ring_offset;      /* ring write index, pre-append */
    const uint64_t csl_after = csl + (uint64_t)t;
    const int k_len = csl_after < (uint64_t)TR_CTX ? (int)csl_after : TR_CTX;
    const uint32_t upd_off = (uint32_t)((off + (uint32_t)t) % TR_CTX);

    /* RotatingCache::positions(t) (candle-nn kv_cache.rs): absolute position
     * held by each cache slot AFTER this step's append, in slot order. */
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
     * [csl_after - k_len, csl] only), so evaluating it uniformly is exact. */
    for (int tp = 0; tp < t; tp++) {
        const int64_t q_abs = (int64_t)csl + tp;
        float *mrow = st->maskv + (size_t)tp * TR_CTX;
        for (int i = 0; i < k_len; i++) {
            const int allowed = (int64_t)st->last_reset_pos <= ks[i] &&
                                ks[i] <= q_abs && q_abs <= ks[i] + TR_CTX;
            mrow[i] = allowed ? 0.0f : -INFINITY;
        }
    }

    /* Rope tables (transformer.rs:871-876 + RotaryEmbedding::rope):
     * pos = arange(csl, csl+t) as u32 -> f32; freqs[tp][j] = pos * inv_freq[j];
     * cos/sin in f32. Table layout [t, 32] flat == rope_i's i_over_2 index. */
    for (int tp = 0; tp < t; tp++) {
        const float p = (float)(uint32_t)(csl + (uint64_t)tp);
        float *crow = st->rope_cos + (size_t)tp * TR_HD2;
        float *srow = st->rope_sin + (size_t)tp * TR_HD2;
        for (int j = 0; j < TR_HD2; j++) {
            const float ang = p * st->inv_freq[j];
            crow[j] = cosf(ang);
            srow[j] = sinf(ang);
        }
    }

    /* ---- layers (StreamingTransformerLayer::forward, norm_first=true) ---- */
    for (int l = 0; l < TR_L; l++) {
        TrLayer *L = &st->layers[l];

        /* norm1 (candle_nn LayerNorm fast path, see NOTES f) */
        for (int tp = 0; tp < t; tp++) {
            mimi_layer_norm_f32(st->xt + (size_t)tp * TR_D, L->norm1_w,
                                L->norm1_b, st->normb + (size_t)tp * TR_D,
                                TR_D, TR_EPS);
        }

        /* in_proj: packed qkv = norm1(x) @ W^T, rows [q|k|v] head-major.
         * qkv[tp][s*512 + h*64 + dd] == candle reshape (b,t,3,h,d). */
        for (int tp = 0; tp < t; tp++) {
            mimi_gemv_f32(L->in_proj_w, st->normb + (size_t)tp * TR_D, NULL,
                          st->qkv + (size_t)tp * TR_QKV, TR_QKV, TR_D);
        }

        /* rope_i on q and k, in place (candle_nn rotary_emb::rope_i):
         *   y[2j]   = x[2j]*cos - x[2j+1]*sin
         *   y[2j+1] = x[2j]*sin + x[2j+1]*cos
         * pairs interleaved on head_dim; cos/sin indexed [tp][j]. v untouched.
         * Products kept in separate statements so clang cannot contract them
         * into fma (rustc does not contract). */
        for (int tp = 0; tp < t; tp++) {
            const float *crow = st->rope_cos + (size_t)tp * TR_HD2;
            const float *srow = st->rope_sin + (size_t)tp * TR_HD2;
            for (int qk = 0; qk < 2; qk++) { /* 0: q block, 1: k block */
                float *blk = st->qkv + (size_t)tp * TR_QKV + (size_t)qk * TR_D;
                for (int h = 0; h < TR_H; h++) {
                    float *hp = blk + (size_t)h * TR_HD;
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
                float *sc = st->scores + (size_t)tp * TR_CTX;
                const float *mrow = st->maskv + (size_t)tp * TR_CTX;
                /* scores row = K[0..k_len) @ q — a gemv over contiguous ring
                 * rows (dot order owned by mimi_gemv_f32, unit 6). */
                mimi_gemv_f32(kr, qh, NULL, sc, k_len, TR_HD);
                for (int i = 0; i < k_len; i++) {
                    const float scaled = sc[i] * TR_ATTN_SCALE;
                    sc[i] = scaled + mrow[i]; /* additive -inf mask */
                }
                /* softmax_last_dim over the k_len window, f32, in place */
                mimi_softmax_f32(sc, k_len);
                /* ws @ v: accumulate slots sequentially, lanes independent */
                float *out = st->attn_cat + (size_t)tp * TR_D + (size_t)h * TR_HD;
                memset(out, 0, TR_HD * sizeof(float));
                for (int i = 0; i < k_len; i++) {
                    tr_axpy64(out, sc[i], vr + (size_t)i * TR_HD);
                }
            }
        }

        /* out_proj (no bias) then x = x + layer_scale_1 * attn */
        for (int tp = 0; tp < t; tp++) {
            mimi_gemv_f32(L->out_proj_w, st->attn_cat + (size_t)tp * TR_D, NULL,
                          st->branch + (size_t)tp * TR_D, TR_D, TR_D);
        }
        for (int tp = 0; tp < t; tp++) {
            float *xr = st->xt + (size_t)tp * TR_D;
            const float *br = st->branch + (size_t)tp * TR_D;
            for (int c = 0; c < TR_D; c++) {
                /* candle: separate broadcast_mul then add — keep two
                 * statements so no fma contraction. */
                const float scaled = L->ls1[c] * br[c];
                xr[c] = xr[c] + scaled;
            }
        }

        /* mlp branch: x = x + layer_scale_2 * linear2(gelu_erf(linear1(norm2(x)))) */
        for (int tp = 0; tp < t; tp++) {
            mimi_layer_norm_f32(st->xt + (size_t)tp * TR_D, L->norm2_w,
                                L->norm2_b, st->normb + (size_t)tp * TR_D,
                                TR_D, TR_EPS);
        }
        for (int tp = 0; tp < t; tp++) {
            mimi_gemv_f32(L->lin1_w, st->normb + (size_t)tp * TR_D, NULL,
                          st->mlp_hidden + (size_t)tp * TR_FF, TR_FF, TR_D);
        }
        for (int i = 0; i < t * TR_FF; i++) {
            st->mlp_hidden[i] = mimi_gelu_erf_f32(st->mlp_hidden[i]);
        }
        for (int tp = 0; tp < t; tp++) {
            mimi_gemv_f32(L->lin2_w, st->mlp_hidden + (size_t)tp * TR_FF, NULL,
                          st->branch + (size_t)tp * TR_D, TR_D, TR_FF);
        }
        for (int tp = 0; tp < t; tp++) {
            float *xr = st->xt + (size_t)tp * TR_D;
            const float *br = st->branch + (size_t)tp * TR_D;
            for (int c = 0; c < TR_D; c++) {
                const float scaled = L->ls2[c] * br[c];
                xr[c] = xr[c] + scaled;
            }
        }
    }

    /* advance the shared cache bookkeeping once (all layers appended t) */
    st->seq_len = csl_after;
    st->ring_offset = upd_off;

    /* conv_layout boundary out: [T,C] -> [C,T]. y may alias x. */
    for (int c = 0; c < TR_D; c++) {
        float *row = y + (size_t)c * (size_t)n;
        for (int tp = 0; tp < t; tp++) row[tp] = st->xt[(size_t)tp * TR_D + c];
    }
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
 *   LayerScale::forward (:90-94)            -> fused "scaled = ls[c]*br[c]; x += scaled"
 *                                              loops after each branch (two
 *                                              statements = candle's separate
 *                                              broadcast_mul + add, no fma).
 *   RotaryEmbedding::new (:368-374)         -> inv_freq[] in init.
 *   RotaryEmbedding::rope (:376-384)        -> per-step rope_cos/rope_sin
 *                                              tables ([t,32], 1-D pos branch).
 *   Rope::apply_rotary_emb (:355-359)       -> in-place rope loop on q,k blocks
 *                                              (input already f32; the
 *                                              to_dtype(F32) round-trip is
 *                                              identity here).
 *   StreamingMultiheadAttention::forward
 *     (:445-513)                            -> per-layer attention section of
 *                                              mimi_transformer_step.
 *   KvCache::append / positions
 *     (kv_cache.rs:242-252 -> candle-nn
 *      kv_cache.rs RotatingCache)           -> k_ring/v_ring + ks[] block (see d).
 *   Mlp::NoGating::forward (:565)           -> linear1 gemv -> mimi_gelu_erf_f32
 *                                              -> linear2 gemv.
 *   Norm::LayerNorm forward (:601-618,
 *     eps 1e-5 from Norm::new_shortcut :639) -> mimi_layer_norm_f32(..., 1e-5f).
 *   StreamingTransformerLayer::forward
 *     (:731-758, norm_first)                -> layer loop body (attn residual,
 *                                              then mlp residual; cross_attn
 *                                              arm dead, config None).
 *   StreamingTransformer::step/forward/
 *     forward_ca (:816-898, :951-957)       -> positional bookkeeping + mask +
 *                                              rope tables + layer loop.
 *   ProjectedTransformer::step (:1028-1046) -> mimi_transformer_step outer
 *                                              transposes; input_proj /
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
 *     Muls are kept in separate statements from the add/sub so clang -O2
 *     (ffp-contract=on) cannot fuse what rustc computes unfused.
 *
 * (c) conv_layout boundary handling (ProjectedTransformer::step, conv_layout
 *     = true): input [B,C,T] -> transpose(1,2) -> [B,T,C] before the
 *     transformer, transpose back after the (identity) output proj. With the
 *     header's [MIMI_DIM, n] conv layout and batch 1 this is
 *       xt[tp*512 + c] = x[c*n + tp]  on entry,
 *       y[c*n + tp] = xt[tp*512 + c]  on exit.
 *     All internal math is [t, c] row-major (candle's b,t,c with b=1).
 *
 * (d) KV interface: OWNED RING, not the unit-5 ScatteredKvCache ABI.
 *     Why: transformer.rs:432 builds `KvCache::new(2, context)` ==
 *     KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache). moshi's
 *     ScatteredKvCache/ScatteredCacheBuilder (kv_cache.rs) is only reachable
 *     via crate::batched_transformer (Transformer::Batched), which is out of
 *     scope per the manifest ("batching > 1"); MimiDetokenizer builds the
 *     non-batched Mimi (mimi.rs new_/Transformer::new with batch_size None
 *     -> Transformer::Standard). Coding this unit against a hypothetical
 *     ScatteredKvCache C ABI would port the wrong cache. Proposed
 *     reconciliation for the arbiter: unit 5's real job for the decode chain
 *     is RotatingKvCache semantics; if mimi_kv.cpp ships ScatteredKvCache
 *     faithfully it serves only a future batched path and unit 4 keeps its
 *     ring (for batch=1 the two produce identical attention windows, but the
 *     Standard path literally runs Rotating, so that is what I match).
 *     Semantics reproduced from candle-nn 0.9.2 kv_cache.rs::RotatingCache
 *     (dim=2, max_seq_len=250), specialized to per-step t < 250:
 *       - append: write t frames at slots (offset+i) % 250; offset advances
 *         mod 250; current_seq_len += t. Returned k/v = first
 *         min(current_seq_len, 250) slots in SLOT (storage) order — scores,
 *         softmax and ws@v all run in slot order, matching the tensor candle
 *         hands to matmul/softmax_last_dim.
 *       - positions(t) (computed from PRE-append state, exactly as
 *         forward_ca does): upd_offset = (offset+t)%250; slot i holds
 *         absolute position csl+t+i-upd_offset, minus 250 when i >= upd_offset.
 *       - the seq_len >= max_seq_len append branch and the k-trim in
 *         transformer.rs:479-486 are dead: k_len = min(csl+t, 250) gives
 *         k_len - t <= 250 - t < context, so k_target_len == k_len; guarded
 *         by n <= TR_MAX_N (8).
 *       - mask (forward_ca :836-868): allowed iff last_reset_pos <= ks[i] &&
 *         ks[i] <= csl+t_pos && csl+t_pos <= ks[i]+250, materialized as
 *         additive 0/-inf and added AFTER the 0.125 scale (broadcast_add
 *         order preserved). Rust's mask=None shortcut for t==1 &&
 *         last_reset_pos <= min(ks) equals the all-allowed predicate (ring
 *         holds only positions [csl+t-k_len, csl+t-1], all causal-visible at
 *         t==1), so evaluating the predicate uniformly is exact.
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
 *     confirmed not in the checkpoint listing.
 *
 * (f) reduction orders
 *     - LayerNorm: candle takes ops::layer_norm's CPU fast path (contiguous
 *       f32, remove_mean, bias present — candle-nn 0.9.2 ops.rs LayerNorm
 *       CustomOp3): ONE sequential pass accumulating sum and sum2 in f32
 *       (index order), mean = sum/512, var = sum2/512 - mean*mean (naive, NOT
 *       Welford/two-pass), inv_std = 1/sqrtf(var + 1e-5f), out =
 *       (x-mean)*inv_std*w + b with that exact op order. mimi_layer_norm_f32
 *       (unit 6) MUST implement this formula/order.
 *     - softmax (candle-nn ops.rs SoftmaxLastDim cpu_fwd): max via sequential
 *       f32 max scan (Rust f32::max == fmaxf semantics), d = expf(s - max),
 *       sum via candle's vec_sum (NEON blocked tree on aarch64 — not
 *       sequential), then d /= sum (division, not multiply-by-reciprocal).
 *       mimi_softmax_f32 (unit 6) should follow; its sum-block shape is the
 *       one place candle is not economically bit-reproducible (gemm/NEON
 *       blocking) — faithful tier, harness measures the band.
 *     - attention scores: one dot per kv slot via mimi_gemv_f32 over the
 *       contiguous ring rows (K[0..k_len) x q); candle runs the gemm crate
 *       (blocked, fma). Scale-then-mask per element: (dot*0.125f) + mask —
 *       0.125 is exact, add of 0/-inf exact.
 *     - ws @ v: slots accumulated SEQUENTIALLY in slot order (tr_axpy64), 64
 *       lanes independent; NEON path uses vfmaq (4x4 unroll), scalar _ref is
 *       mul-then-add per element.
 *     - projections/MLP: per-token mimi_gemv_f32 (y[m] = sum_k w[m,k]x[k]);
 *       candle Linear is xs.matmul(w.t()) via the gemm crate — blocked and
 *       fma'd, so gemv order is unit 6's documented choice; faithful tier.
 *     - gelu_erf (candle-core op.rs GeluErf::f32): ((erff(x * (1/sqrt(2)))
 *       + 1.0f) * 0.5f) * x, all f32 — candle calls the Rust libm crate's
 *       erff; mimi_gelu_erf_f32 (unit 6, per header comment) uses system
 *       erff. Same algorithm family, may differ by ulps (see g).
 *     - residual/LayerScale: elementwise, two statements (mul, then add) to
 *       forbid fma contraction — rustc does not contract fp ops.
 *
 * (g) uncertainties
 *     1. Transcendental ulp drift: Rust f32::sin/cos/powf lower to the
 *        platform libm on aarch64-apple-darwin, so rope tables should match
 *        bit-for-bit in practice, but candle's erff is the Rust `libm` crate
 *        (portable Musl-derived polynomial) while ours is Apple libm erff —
 *        expect <=1-2 ulp differences feeding gelu; covered by the ulp-band
 *        harness, flagged here for the threshold ledger.
 *     2. mimi_gemv_f32 / mimi_softmax_f32 / mimi_layer_norm_f32 /
 *        mimi_gelu_erf_f32 are unit-6-owned; this unit's parity depends on
 *        them honoring the semantics in (f) — especially layer_norm's naive
 *        sum/sum2 single pass and softmax's divide-by-sum. If unit 6 shipped
 *        Welford or two-pass mean/var, layer 0's output already drifts.
 *     3. Scale factor: candle multiplies pre_ws by (head_dim as f64)^-0.5
 *        as an affine op (f64 scalar cast to f32) = 0.125 exactly — no
 *        uncertainty, noted only because the f64 detour looks lossy but isn't.
 *     4. TR_MAX_N = 8 headroom: steady-state n is 1 or 2 from the stride-2
 *        upsampler; if some priming path ever hands >8 frames in one step the
 *        unit returns -1 rather than silently splitting (no-fallbacks rule).
 *        The Rust path would handle it; revisit if the harness ever trips it.
 *     5. fp contraction: guarded the spots that matter (rope, layer-scale
 *        residuals) by statement splitting; clang could still in principle
 *        contract inside mimi_gemv_f32 et al. (unit 6's ledger). If exact
 *        rustc parity of the in-unit loops is ever required, compile this TU
 *        with -ffp-contract=off and re-measure — the manifest command leaves
 *        contraction on.
 *     6. y aliasing x in mimi_transformer_step is safe (output written from
 *        the xt scratch after all reads), matching the header's in-place
 *        allowance.
 */
