#ifndef LFM_MODEL_PLAN_H
#define LFM_MODEL_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "flashkern_prng.h"
#include "flashkern_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private native plan ABI shared by the model binder and fixed executor.
 * Rust never constructs either structure in production. */
typedef struct LfmLayerDesc {
    uint32_t kind;
    uint32_t k;
    float op_eps;
    float ffn_eps;
    const uint16_t *op_norm_w;
    const uint16_t *ffn_norm_w;
    const uint16_t *in_w;
    const uint16_t *conv_w;
    const uint16_t *out_w;
    const uint16_t *w1;
    const uint16_t *w3;
    const uint16_t *w2;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t hd;
    float qk_eps;
    const uint16_t *q_w;
    const uint16_t *k_w;
    const uint16_t *v_w;
    const uint16_t *o_w;
    const uint16_t *qn_w;
    const uint16_t *kn_w;
} LfmLayerDesc;

typedef struct LfmLayerState {
    uint16_t *k_plane;
    uint16_t *v_plane;
    size_t head_stride;
    size_t k_len;
    size_t v_len;
    uint16_t *conv_state;
    size_t conv_len;
} LfmLayerState;

int lfm_ctx_build(void *engine, const LfmLayerDesc *descs, size_t layers,
                  size_t hidden, size_t ffn, size_t max_context,
                  uint64_t *out_id);
int lfm_ctx_set_heads(void *engine, uint64_t id,
                      const uint16_t *text_embedding, size_t text_elements,
                      size_t vocab, const uint16_t *audio_embedding,
                      size_t audio_elements, size_t audio_rows,
                      const uint16_t *final_norm, size_t final_norm_elements,
                      float final_norm_eps);
int lfm_ctx_clear(void *engine, uint64_t id);
uint32_t lfm_engine_lanes(void *engine);
int lfm_engine_token_pass(void *engine, uint64_t id,
                          const uint32_t *ids, size_t id_count,
                          uint32_t embedding_kind,
                          const LfmLayerState *states, size_t state_count,
                          size_t position, const uint16_t *rope_cos,
                          const uint16_t *rope_sin, size_t rope_elements,
                          uint16_t *out_hidden, size_t hidden_elements,
                          float *out_logits, size_t logit_elements,
                          const LfmSamplerConfigV1 *sampler,
                          LfmPrngStateV1 *prng, uint32_t *out_token,
                          size_t lanes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MODEL_PLAN_H */
