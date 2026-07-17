// mimi_kernel.h — the ABI contract for the Mimi decode kernel port.
// Faithful C++/NEON port of the moshi 0.6.4 Mimi DECODER path (decode_step):
//   RVQ dequantize -> ConvTr upsample (x2) -> 8-layer streaming transformer
//   -> SEANet decoder (ratios 8*6*5*4) -> 1920 f32 samples @ 24 kHz per frame.
// Encoder half, batching > 1, quantized weights: out of scope (see
// docs/MIMI_PORT.md). Arbiter-owned: unit implementations must code against
// THIS header; propose changes in NOTES, do not fork the types.
//
// Discipline (engine rules apply verbatim):
//   - Weights are a buffer: flat name -> {bytes, len} table, zero-copy views
//     into the native resident safetensors image. Weight-norm folds ONCE at
//     plan construction into shared derived storage; nothing repacks.
//   - Zero allocation in steady state: every stream state and scratch lives
//     in one conversation arena sized at plan construction. State is POD.
//   - f32 math, f32 accumulation, documented loop order. MATH IS ASSEMBLY AT
//     EVERY STEP (her rule): no tensor-op thinking, no scalar C++ loops as a
//     primary path — every sweep/reduction/activation is aarch64 NEON
//     intrinsics (float32x4_t: fmla/vmax/vsub, vectorized loads/stores), and
//     GEMM/GEMV ride AMX via Accelerate. Scalar code exists in exactly two
//     places: the `..._ref` parity siblings under MIMI_SCALAR_REF, and
//     sub-vector tail remainders. Transcendentals (erff/expf): lane-wise libm
//     calls INSIDE the NEON sweep on the first pass (faithful tier — a
//     polynomial vector exp/erf changes numerics; it enters later, behind the
//     parity gate, as a fast-tier variant).
//   - No exceptions across this ABI. No candle. Return codes, not throws.
//   - ENGINE PLACEMENT (her directive): this kernel runs INSIDE the same kcoro
//     engine as the backbone/depthformer (flashkern_engine.cpp) — same
//     persistent lane team, same doorbell, a REQ kind at the pass boundary.
//     Because this is a native C++ program (no Rust frames), its lane fences
//     PARK precisely after the bounded spin (two-barrier doctrine) — unlike
//     the Rust depthformer program, which must spin pure. First-pass unit
//     APIs are single-call; write inner loops BAND-SPLITTABLE (channel/row
//     bands with no incidental cross-band sequential dependence) so the
//     arbiter integration step can cut them across lanes without re-deriving
//     the math.
//
// House NEON idiom (arm_neon.h, hers verbatim — chunks of a full register,
// tail handled scalar after):
//   for (int i = 0; i + 4 <= n; i += 4) {
//       float32x4_t va = vld1q_f32(&a[i]);
//       float32x4_t vb = vld1q_f32(&b[i]);
//       vst1q_f32(&r[i], vaddq_f32(va, vb));
//   }
#ifndef MIMI_KERNEL_H
#define MIMI_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- weight table -------------------------------------------------------
 * The native safetensors loader (or the parity harness) captures every decoder
 * tensor as a named f32 span in checkpoint layout and hands the table down once.
 * Names are the safetensors keys (e.g. "decoder.model.0.conv.weight",
 * "decoder_transformer.transformer.layers.3.self_attn.in_proj_weight",
 * "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum", ...).
 * Lookup is init-time only — steady state touches raw pointers. */
typedef struct MimiWeight {
    const char *name;   /* safetensors key, NUL-terminated */
    const uint8_t *bytes; /* little-endian f32 bytes; may be unaligned */
    const uint64_t *shape; /* dims, length ndim */
    uint32_t ndim;
    uint64_t len;       /* total element count */
} MimiWeight;

typedef struct MimiWeightTable {
    const MimiWeight *entries;
    uint32_t count;
} MimiWeightTable;

/* init-time helper (mimi_decode.cpp owns the impl): NULL if absent.
 * REQUIRED weights hard-fail init (no-fallbacks) — return code, message out. */
const MimiWeight *mimi_weight_find(const MimiWeightTable *t, const char *name);

/* ---- fixed config (mimi.rs Config::v0_1(8)) ----------------------------- */
enum {
    MIMI_DIM = 512,          /* seanet.dimension == transformer d_model */
    MIMI_QUANT_DIM = 256,    /* quantizer codebook dim */
    MIMI_BINS = 2048,        /* codebook size */
    MIMI_NQ = 8,             /* rvq_first 1 + rvq_rest 7 */
    MIMI_TR_LAYERS = 8,
    MIMI_TR_HEADS = 8,
    MIMI_TR_CONTEXT = 250,   /* KV window */
    MIMI_TR_FF = 2048,
    MIMI_TR_MAX_PERIOD = 10000,
    MIMI_UPSAMPLE_STRIDE = 2,  /* 12.5 Hz -> 25 Hz */
    MIMI_N_FILTERS = 64,
    MIMI_KERNEL = 7,
    MIMI_RES_KERNEL = 3,
    MIMI_LAST_KERNEL = 3,
    MIMI_FRAME_OUT = 1920,   /* samples @ 24 kHz per latent frame */
};
static const int MIMI_RATIOS[4] = {8, 6, 5, 4};
static const float MIMI_LAYER_SCALE_INIT = 0.01f; /* trained values come from weights */
static const float MIMI_ELU_ALPHA = 1.0f;

/* ---- arena --------------------------------------------------------------
 * One exact-sized block owns one conversation's mutable state + scratch. A
 * separate sealed plan arena supplies shared formula-derived immutable spans.
 * Each unit carves its POD state at init; steady state never allocates. */
typedef struct MimiArena {
    uint8_t *base;
    size_t size;
    size_t used;
    struct MimiDerivedArena *derived;
    size_t derived_cursor;
} MimiArena;
void *mimi_arena_alloc(MimiArena *a, size_t bytes); /* aborts on overflow: sizing bug */
void *mimi_arena_alloc_derived(MimiArena *a, size_t bytes);
int mimi_arena_building_derived(const MimiArena *a);

/* Safe checkpoint load. This never forms a typed pointer for unaligned
 * storage. Direct matrix calls specialize once on alignment and otherwise use
 * the byte-load path without staging or repacking weights. */
float mimi_weight_load_f32(const uint8_t *bytes, uint64_t index);

/* ---- shared math primitives (mimi_decode.cpp owns impls; units call) -----
 * GEMM/GEMV are AMX-backed on Apple: implement over Accelerate cblas_sgemm /
 * cblas_sgemv (-framework Accelerate) — never a hand-rolled vanilla GEMM; the
 * machine has a matrix coprocessor and prefill already runs on it (E4).
 * Scalar _ref siblings under MIMI_SCALAR_REF remain the parity-bisect path.
 *   y[m] = sum_k w[m*k_stride + k] * x[k] (+ b[m])  — row-major W [M,K] */
void mimi_gemv_f32(const float *w, const float *x, const float *bias_or_null,
                   float *y, int m, int k);
/* C[MxN] += / = A[MxK] * B[KxN], row-major, f32 accumulate. beta 0 or 1. */
void mimi_gemm_f32(const float *a, const float *b, float *c,
                   int m, int k, int n, int beta);
void mimi_weight_gemv_f32(const uint8_t *w, const float *x,
                          const uint8_t *bias_or_null, float *y, int m, int k);
void mimi_weight_gemm_f32(const uint8_t *w, const float *b, float *c,
                          int m, int k, int n, int beta);
void mimi_weight_gemm_tn_f32(const uint8_t *w, const float *b, float *c,
                             int rows, int k, int n);
void mimi_softmax_f32(float *x, int n);            /* in place, max-subtracted;
                                                      NEON max/sum reductions,
                                                      lane-wise expf */
/* scalar per-element forms: tail/lane helpers + _ref building blocks ONLY —
 * hot loops call the NEON sweep forms below, never these in a loop. */
float mimi_gelu_erf_f32(float x);                  /* 0.5x(1+erf(x/sqrt(2))) — erff */
float mimi_elu_f32(float x, float alpha);          /* x>0 ? x : alpha*(expf(x)-1) */
/* NEON sweep forms (primary path for activations/elementwise): */
void mimi_gelu_erf_vec_f32(const float *x, float *y, int n);
void mimi_elu_vec_f32(const float *x, float *y, int n, float alpha);
void mimi_add_vec_f32(const float *a, const float *b, float *y, int n);
void mimi_scale_vec_f32(const float *x, const float *s, float *y, int n); /* y=x*s elementwise (LayerScale) */
void mimi_layer_norm_f32(const float *x, const float *w, const float *b,
                         float *y, int n, float eps); /* NEON mean/var/apply */
void mimi_weight_scale_vec_f32(const float *x, const uint8_t *s,
                               float *y, int n);
void mimi_weight_layer_norm_f32(const float *x, const uint8_t *w,
                                const uint8_t *b, float *y, int n, float eps);

/* ---- unit entry points ---------------------------------------------------
 * Streaming convention (replaces StreamTensor): every step takes
 * n_in frames and reports n_out frames; 0 is legal (module buffering).
 * Layout is conv layout throughout: [C, T] channel-major per frame batch=1.
 * Each unit: *_init carves state from the arena + captures weight pointers
 * (binding shared plan folds if needed), *_step runs frames, *_reset re-arms state.
 * Init returns 0 on success, nonzero + msg on missing/misshaped weights. */

/* 1. quantization: codes [MIMI_NQ] u32 -> emb [MIMI_DIM, 1] */
typedef struct MimiQuantState MimiQuantState;
int  mimi_quant_init(MimiQuantState **st, const MimiWeightTable *w,
                     MimiArena *a, char *err, size_t errlen);
void mimi_quant_decode(MimiQuantState *st, const uint32_t *codes, float *emb_out);

/* 2. conv primitives: used by upsample + seanet (state structs in mimi_conv.cpp) */
typedef struct MimiConvState MimiConvState;       /* StreamableConv1d */
typedef struct MimiConvTrState MimiConvTrState;   /* StreamableConvTranspose1d */
int  mimi_conv_init(MimiConvState **st, const MimiWeightTable *w,
                    const char *prefix, int in_c, int out_c, int ksize,
                    int stride, int dilation, int groups, int causal,
                    MimiArena *a, char *err, size_t errlen);
int  mimi_convtr_init(MimiConvTrState **st, const MimiWeightTable *w,
                      const char *prefix, int in_c, int out_c, int ksize,
                      int stride, int causal, MimiArena *a,
                      char *err, size_t errlen);
/* frames in [in_c, n_in] -> out [out_c, n_out]; returns n_out (>=0) */
int  mimi_conv_step(MimiConvState *st, const float *x, int n_in, float *y);
int  mimi_convtr_step(MimiConvTrState *st, const float *x, int n_in, float *y);
void mimi_conv_reset(MimiConvState *st);
void mimi_convtr_reset(MimiConvTrState *st);

/* upsample wrapper (ConvTrUpsample1d: stride 2, dim 512, causal, learnt) */
typedef struct MimiUpsampleState MimiUpsampleState;
int  mimi_upsample_init(MimiUpsampleState **st, const MimiWeightTable *w,
                        MimiArena *a, char *err, size_t errlen);
int  mimi_upsample_step(MimiUpsampleState *st, const float *x, int n_in, float *y);
void mimi_upsample_reset(MimiUpsampleState *st);

/* 4+5. streaming decoder transformer (KV ring inside, context 250) */
typedef struct MimiTransformerState MimiTransformerState;
int  mimi_transformer_init(MimiTransformerState **st, const MimiWeightTable *w,
                           MimiArena *a, char *err, size_t errlen);
/* x [MIMI_DIM, n] in conv layout; in place is allowed via y == distinct buf */
int  mimi_transformer_step(MimiTransformerState *st, const float *x, int n, float *y);
void mimi_transformer_reset(MimiTransformerState *st);

/* 3. seanet decoder: latent [MIMI_DIM, n] -> pcm [1, n*960] */
typedef struct MimiSeanetState MimiSeanetState;
int  mimi_seanet_init(MimiSeanetState **st, const MimiWeightTable *w,
                      MimiArena *a, char *err, size_t errlen);
int  mimi_seanet_step(MimiSeanetState *st, const float *x, int n_in, float *pcm);
void mimi_seanet_reset(MimiSeanetState *st);

/* 6. top level: model-lifetime plan + conversation-lifetime state */
typedef struct MimiDecoder MimiDecoder;
typedef struct MimiDecodePlan MimiDecodePlan;
typedef struct MimiDecodeState MimiDecodeState;
typedef struct LfmWeightImage LfmWeightImage;
int mimi_decode_plan_new_from_image(MimiDecodePlan **plan,
                                    const LfmWeightImage *image,
                                    char *err, size_t errlen);
void mimi_decode_plan_free(MimiDecodePlan *plan);
uint64_t mimi_decode_plan_derived_bytes(const MimiDecodePlan *plan);
uint64_t mimi_decode_plan_compatibility_copied_bytes(const MimiDecodePlan *plan);
int mimi_decode_state_new(MimiDecodeState **state, const MimiDecodePlan *plan,
                          char *err, size_t errlen);
void mimi_decode_state_free(MimiDecodeState *state);
int mimi_decode_state_step(MimiDecodeState *state, const uint32_t *codes,
                           float *pcm_out);
void mimi_decode_state_reset(MimiDecodeState *state);
uint64_t mimi_decode_state_bytes(const MimiDecodeState *state);
int  mimi_decoder_new(MimiDecoder **d, const MimiWeightTable *w,
                      char *err, size_t errlen);
/* Transitional single-state wrappers used by parity tests. */
int  mimi_decoder_new_from_image(MimiDecoder **d, const LfmWeightImage *image,
                                 char *err, size_t errlen);
int  mimi_decoder_new_from_file(MimiDecoder **d, const char *checkpoint,
                                char *err, size_t errlen);
/* one latent frame of codes [MIMI_NQ] -> n_out samples (0 while priming);
 * pcm_out capacity MIMI_FRAME_OUT * 2 (drain headroom). */
int  mimi_decoder_step(MimiDecoder *d, const uint32_t *codes, float *pcm_out);
void mimi_decoder_reset(MimiDecoder *d);
void mimi_decoder_free(MimiDecoder *d);
uint64_t mimi_decoder_derived_bytes(const MimiDecoder *d);
uint64_t mimi_decoder_compatibility_copied_bytes(const MimiDecoder *d);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* MIMI_KERNEL_H */
