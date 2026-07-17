#ifndef LFM_MODEL_PLAN_H
#define LFM_MODEL_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "flashkern_prng.h"
#include "flashkern_sampler.h"
#include "lfm_mimi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { LFM_PREFILL_MAX_ROWS = 4 };

/* Private pointer-free context state shared by the model and the eager native
 * recurrence route. It is not part of the product lifecycle ABI. */
typedef struct LfmContextWindowState {
    uint64_t capacity;
    uint64_t runway;
    uint64_t position;
    uint64_t start;
    uint64_t cursor;
    uint64_t rope_base;
} LfmContextWindowState;

typedef struct LfmTokenCommitRecord {
    LfmContextWindowState *window;
    uint64_t expected_position;
    uint64_t expected_start;
    uint64_t expected_cursor;
    uint64_t expected_rope_base;
    uint32_t *token_committed;
} LfmTokenCommitRecord;

typedef struct LfmRouteEpoch LfmRouteEpoch;

/* Private, fixed conversation-owned result for the bounded audio route. */
typedef struct LfmAudioRouteResult {
    int32_t status;
    uint32_t token_completed;
    uint32_t token_committed;
    uint32_t depth_completed;
    uint32_t mimi_completed;
    uint32_t eoaudio;
    uint32_t reserved;
    size_t pcm_samples;
    uint32_t codes[LFM_MIMI_CODEBOOKS];
} LfmAudioRouteResult;

typedef struct LfmAudioRouteTarget {
    const LfmRouteEpoch *epoch;
    uint64_t expected_epoch;
    float *pcm;
    size_t pcm_capacity;
} LfmAudioRouteTarget;

int lfm_context_window_can_commit(const LfmContextWindowState *window);
int lfm_context_window_commit(LfmContextWindowState *window);

/* Private native plan ABI shared by the model binder and fixed executor.
 * Rust never constructs either structure in production. */
typedef struct LfmLayerDesc {
    uint32_t kind;
    uint32_t k;
    float op_eps;
    float ffn_eps;
    /* Resident checkpoint storage is byte-addressed. Safetensors permits an
     * odd tensor start, so constructing a uint16_t pointer here would already
     * promise alignment that the image does not provide. Architecture leaves
     * perform unaligned loads and unlift each little-endian word in registers. */
    const uint8_t *op_norm_w;
    const uint8_t *ffn_norm_w;
    const uint8_t *in_w;
    const uint8_t *conv_w;
    const uint8_t *out_w;
    const uint8_t *w1;
    const uint8_t *w3;
    const uint8_t *w2;
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t hd;
    float qk_eps;
    const uint8_t *q_w;
    const uint8_t *k_w;
    const uint8_t *v_w;
    const uint8_t *o_w;
    const uint8_t *qn_w;
    const uint8_t *kn_w;
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
                      const uint8_t *text_embedding, size_t text_elements,
                      size_t vocab, const uint8_t *audio_embedding,
                      size_t audio_elements, size_t audio_rows,
                      const uint8_t *final_norm, size_t final_norm_elements,
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
                          size_t lanes, const uint16_t *provided_embed);

/* Private eager two-node route used by interleaved audio generation. It runs
 * TOKEN_PASS without sampling, commits the exact predeclared context record,
 * then tail-submits DEPTH_FRAME from the token CQ on the same retained slot.
 * The outward call blocks only on the terminal slot completion. */
int lfm_engine_audio_recurrence(
    void *engine, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    uint32_t *out_codes, size_t code_count, size_t lanes,
    const struct LfmTokenCommitRecord *commit,
    uint32_t *out_token_completed);

/* Production bounded route. PCM points directly into an already-retained
 * playback reservation; no dock or publication operation occurs in-engine. */
int lfm_engine_audio_route(
    void *engine, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    MimiDecodeState *mimi, const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, size_t lanes,
    const struct LfmTokenCommitRecord *commit);

/* Private native prefill seam. The workspace is conversation-owned and fully
 * sized before readiness; production never exposes it or the row/state planes
 * through the Rust ABI. A pass accepts at most four consecutive text embedding
 * ids (kind 0) or borrowed BF16 rows (kind 2). */
int lfm_engine_prefill_workspace_create(void *engine, uint64_t id,
                                        void **out_workspace);
void lfm_engine_prefill_workspace_destroy(void *workspace);
int lfm_engine_prefill(void *engine, uint64_t id, void *workspace,
                       const uint32_t *ids, const uint16_t *provided_rows,
                       size_t row_count, uint32_t embedding_kind,
                       const LfmLayerState *states, size_t state_count,
                       size_t position, const uint16_t *rope_cos,
                       const uint16_t *rope_sin, size_t rope_elements,
                       uint16_t *out_hidden, size_t hidden_elements,
                       const LfmSamplerConfigV1 *sampler,
                       LfmPrngStateV1 *prng, uint32_t *out_token,
                       size_t lanes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MODEL_PLAN_H */
