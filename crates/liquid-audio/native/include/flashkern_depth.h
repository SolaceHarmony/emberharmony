#ifndef FLASHKERN_DEPTH_H
#define FLASHKERN_DEPTH_H

#include <stddef.h>
#include <stdint.h>

#include "flashkern_prng.h"
#include "flashkern_sampler.h"

#ifdef __cplusplus
extern "C" {
#define LFM_DEPTH_STATIC_ASSERT(test, message) static_assert(test, message)
#else
#define LFM_DEPTH_STATIC_ASSERT(test, message) _Static_assert(test, message)
#endif


/* Counted immutable view into the resident checkpoint image. The build call
 * copies descriptors only; payload bytes remain at their original addresses. */
typedef struct LfmDepthBuffer {
    uintptr_t address;
    size_t count;
} LfmDepthBuffer;

typedef struct LfmDepthLayer {
    LfmDepthBuffer qkv_w;
    LfmDepthBuffer out_w;
    LfmDepthBuffer q_ln;
    LfmDepthBuffer k_ln;
    LfmDepthBuffer op_norm;
    LfmDepthBuffer ffn_norm;
    LfmDepthBuffer w1;
    LfmDepthBuffer w3;
    LfmDepthBuffer w2;
} LfmDepthLayer;

typedef struct LfmDepthHead {
    LfmDepthBuffer embedding;
    LfmDepthBuffer norm;
    LfmDepthBuffer logits;
    size_t vocab;
} LfmDepthHead;

/* Construction-only descriptor. Flashkern copies the layer/head tables and
 * reserves every mutable plane before publishing the returned plan identity. */
typedef struct LfmDepthPlan {
    uint32_t dim;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t codebooks;
    uint32_t backbone_dim;
    float eps;
    LfmDepthBuffer depth_linear_w;
    LfmDepthBuffer depth_linear_b;
    LfmDepthBuffer rope_cos;
    LfmDepthBuffer rope_sin;
    const LfmDepthLayer *layers;
    size_t layer_count;
    const LfmDepthHead *codebook_heads;
    size_t codebook_head_count;
} LfmDepthPlan;

LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthBuffer) == 16,
                        "LfmDepthBuffer layout must stay compact");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthLayer) == 144,
                        "LfmDepthLayer layout must stay compact");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthHead) == 56,
                        "LfmDepthHead layout must stay compact");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthPlan) == 128,
                        "LfmDepthPlan layout must stay compact");

int lfm_engine_depth_build(void *engine, const LfmDepthPlan *plan,
                           uint64_t *out_id);
int lfm_engine_depth_clear(void *engine, uint64_t id);

#undef LFM_DEPTH_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif /* FLASHKERN_DEPTH_H */
