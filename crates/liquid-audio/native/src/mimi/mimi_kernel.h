// mimi_kernel.h — the ABI contract for the Mimi decode kernel port.
// Faithful C++/NEON port of the moshi 0.6.4 Mimi DECODER path (decode_step):
//   RVQ dequantize -> ConvTr upsample (x2) -> 8-layer streaming transformer
//   -> SEANet decoder (ratios 8*6*5*4) -> 1920 f32 samples @ 24 kHz per frame.
// Encoder half, batching > 1, quantized weights: out of scope (see
// docs/MIMI_PORT.md). Arbiter-owned: unit implementations must code against
// THIS header; propose changes in NOTES, do not fork the types.
//
// Discipline (engine rules apply verbatim):
//   - Weights are a buffer: flat name -> {f32*, len} table, zero-copy views
//     into the mmap'd safetensors. Weight-norm folds ONCE at init into the
//     arena; nothing repacks per step.
//   - Zero allocation in steady state: every stream state and scratch lives
//     in ONE arena sized at init. State is POD (hibernatable).
//   - f32 math, f32 accumulation, documented loop order. NEON encouraged in
//     inner loops, but EVERY kernel keeps a scalar reference sibling
//     (`..._ref`) compiled under MIMI_SCALAR_REF for parity bisecting.
//   - No exceptions across this ABI. No candle. Return codes, not throws.
#ifndef MIMI_KERNEL_H
#define MIMI_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- weight table -------------------------------------------------------
 * The Rust rim (or the parity harness) captures every decoder tensor as a
 * named f32 span in checkpoint layout and hands the whole table down once.
 * Names are the safetensors keys (e.g. "decoder.model.0.conv.weight",
 * "decoder_transformer.transformer.layers.3.self_attn.in_proj_weight",
 * "quantizer.rvq_first.vq.layers.0._codebook.embedding_sum", ...).
 * Lookup is init-time only — steady state touches raw pointers. */
typedef struct MimiWeight {
    const char *name;   /* safetensors key, NUL-terminated */
    const float *data;  /* f32, checkpoint layout, read-only, process-long */
    const int64_t *shape; /* dims, length ndim */
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
 * One block, init-sized, owns ALL mutable state + scratch. Each unit declares
 * its state struct (POD) in its .cpp and carves it from the arena at init via
 * mimi_arena_alloc (bump allocator, 64-byte aligned, init-time only).
 * Steady state NEVER allocates. reset_state() zeroes states in place. */
typedef struct MimiArena {
    uint8_t *base;
    size_t size;
    size_t used;
} MimiArena;
void *mimi_arena_alloc(MimiArena *a, size_t bytes); /* aborts on overflow: sizing bug */

/* ---- shared math primitives (mimi_decode.cpp owns impls; units call) -----
 * First pass: correct + documented accumulation order beats clever.
 *   y[m] = sum_k w[m*k_stride + k] * x[k] (+ b[m])  — row-major W [M,K] */
void mimi_gemv_f32(const float *w, const float *x, const float *bias_or_null,
                   float *y, int m, int k);
/* C[MxN] += / = A[MxK] * B[KxN], row-major, f32 accumulate. beta 0 or 1. */
void mimi_gemm_f32(const float *a, const float *b, float *c,
                   int m, int k, int n, int beta);
void mimi_softmax_f32(float *x, int n);            /* in place, max-subtracted */
float mimi_gelu_erf_f32(float x);                  /* 0.5x(1+erf(x/sqrt(2))) — erff */
float mimi_elu_f32(float x, float alpha);          /* x>0 ? x : alpha*(expf(x)-1) */
void mimi_layer_norm_f32(const float *x, const float *w, const float *b,
                         float *y, int n, float eps);

/* ---- unit entry points ---------------------------------------------------
 * Streaming convention (replaces StreamTensor): every step takes
 * n_in frames and reports n_out frames; 0 is legal (module buffering).
 * Layout is conv layout throughout: [C, T] channel-major per frame batch=1.
 * Each unit: *_init carves state from the arena + captures weight pointers
 * (folding into arena if needed), *_step runs frames, *_reset re-arms state.
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

/* 6. top level (mimi_decode.cpp): owns the arena + the chain */
typedef struct MimiDecoder MimiDecoder;
int  mimi_decoder_new(MimiDecoder **d, const MimiWeightTable *w,
                      char *err, size_t errlen);
/* one latent frame of codes [MIMI_NQ] -> n_out samples (0 while priming);
 * pcm_out capacity MIMI_FRAME_OUT * 2 (drain headroom). */
int  mimi_decoder_step(MimiDecoder *d, const uint32_t *codes, float *pcm_out);
void mimi_decoder_reset(MimiDecoder *d);
void mimi_decoder_free(MimiDecoder *d);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* MIMI_KERNEL_H */
