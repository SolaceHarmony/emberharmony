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

#define LFM_DEPTH_ABI_VERSION 1u

/* Counted immutable view into the resident checkpoint image. The build call
 * copies descriptors only; payload bytes remain at their original addresses. */
typedef struct LfmDepthBufferV1 {
    uintptr_t address;
    size_t count;
} LfmDepthBufferV1;

typedef struct LfmDepthLayerV1 {
    LfmDepthBufferV1 qkv_w;
    LfmDepthBufferV1 out_w;
    LfmDepthBufferV1 q_ln;
    LfmDepthBufferV1 k_ln;
    LfmDepthBufferV1 op_norm;
    LfmDepthBufferV1 ffn_norm;
    LfmDepthBufferV1 w1;
    LfmDepthBufferV1 w3;
    LfmDepthBufferV1 w2;
} LfmDepthLayerV1;

typedef struct LfmDepthHeadV1 {
    LfmDepthBufferV1 embedding;
    LfmDepthBufferV1 norm;
    LfmDepthBufferV1 logits;
    size_t vocab;
} LfmDepthHeadV1;

/* Construction-only descriptor. Flashkern copies the layer/head tables and
 * reserves every mutable plane before publishing the returned plan identity. */
typedef struct LfmDepthPlanV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t dim;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t codebooks;
    uint32_t backbone_dim;
    float eps;
    LfmDepthBufferV1 depth_linear_w;
    LfmDepthBufferV1 depth_linear_b;
    LfmDepthBufferV1 rope_cos;
    LfmDepthBufferV1 rope_sin;
    const LfmDepthLayerV1 *layers;
    size_t layer_count;
    const LfmDepthHeadV1 *codebook_heads;
    size_t codebook_head_count;
} LfmDepthPlanV1;

LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthBufferV1) == 16,
                        "LfmDepthBufferV1 ABI changed");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthLayerV1) == 144,
                        "LfmDepthLayerV1 ABI changed");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthHeadV1) == 56,
                        "LfmDepthHeadV1 ABI changed");
LFM_DEPTH_STATIC_ASSERT(sizeof(LfmDepthPlanV1) == 136,
                        "LfmDepthPlanV1 ABI changed");

int lfm_engine_depth_build(void *engine, const LfmDepthPlanV1 *plan,
                           uint64_t *out_id);
int lfm_engine_depth_clear(void *engine, uint64_t id);

#undef LFM_DEPTH_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif /* FLASHKERN_DEPTH_H */
