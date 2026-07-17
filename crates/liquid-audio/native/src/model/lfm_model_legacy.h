#ifndef LFM_MODEL_LEGACY_H
#define LFM_MODEL_LEGACY_H

/*
 * Private transitional numerical ABI.
 *
 * These declarations exist only so the native implementation and the
 * non-release Candle oracle can exercise the cutover seam.  They are not a
 * product interface: production callers use lfm_runtime.h, lfm_session.h, and
 * opaque owner handles.  Keep this header under native/src so tensor-shaped
 * inputs and numerical results cannot become an installed ABI by accident.
 */

#include <stddef.h>
#include <stdint.h>

#include "flashkern_sampler.h"
#include "lfm_runtime.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_MODEL_CAP_DEPTHFORMER 1u
#define LFM_MODEL_CAP_FRONTEND 2u
#define LFM_MODEL_CAP_CONFORMER 4u
#define LFM_MODEL_CAP_MIMI 8u
#define LFM_INPUT_MAX_IDS 8u
#define LFM_AUDIO_TOKEN_CAPACITY 64u

typedef struct LfmConversationConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t seed;
    LfmSamplerConfigV1 text_sampler;
    LfmSamplerConfigV1 audio_sampler;
    uint64_t reserved[4];
} LfmConversationConfigV1;

/* Transitional oracle input record. Production modality assembly stays
 * behind the native session boundary. */
typedef struct LfmInputV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t embedding_kind;
    uint32_t id_count;
    uint32_t ids[LFM_INPUT_MAX_IDS];
    uint64_t reserved[2];
} LfmInputV1;

typedef struct LfmTokenResultV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t position;
    uint32_t sampled_token;
    uint32_t input_count;
    uint32_t embedding_kind;
    uint32_t flags;
    uint64_t reserved[4];
} LfmTokenResultV1;

typedef struct LfmAudioResultV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t source_position;
    uint32_t token_count;
    uint32_t flags;
    uint32_t tokens[LFM_AUDIO_TOKEN_CAPACITY];
    uint64_t reserved[4];
} LfmAudioResultV1;

typedef struct LfmModelInfoV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t resident_bytes;
    uint64_t plan_id;
    uint64_t depth_plan_id;
    uint32_t hidden;
    uint32_t ffn;
    uint32_t layers;
    uint32_t vocab;
    uint32_t max_context;
    uint32_t codebooks;
    uint32_t capabilities;
    uint32_t reserved[5];
} LfmModelInfoV1;

LFM_ORACLE_API int lfm_model_open(void *engine, const char *path,
                                  LfmModel **out, char *error,
                                  size_t error_length);
LFM_ORACLE_API int lfm_model_close(LfmModel *model);
LFM_ORACLE_API int lfm_model_info(const LfmModel *model, LfmModelInfoV1 *out);
LFM_ORACLE_API int lfm_model_memory(const LfmModel *model,
                                    LfmModelMemoryV1 *out);

LFM_ORACLE_API int lfm_conversation_create(
    LfmModel *model, const LfmConversationConfigV1 *config,
    LfmConversation **out, char *error, size_t error_length);
LFM_ORACLE_API int lfm_conversation_step(LfmConversation *conversation,
                                         const uint32_t *ids, size_t id_count,
                                         uint32_t embedding_kind,
                                         LfmTokenResultV1 *out);
LFM_ORACLE_API int lfm_conversation_prefill(
    LfmConversation *conversation, const LfmInputV1 *inputs,
    size_t input_count, uint64_t *out_position);
LFM_ORACLE_API int lfm_conversation_prefill_audio(
    LfmConversation *conversation, const uint16_t *rows, size_t element_count,
    uint64_t *out_position);
LFM_ORACLE_API int lfm_conversation_prefill_pcm_f32(
    LfmConversation *conversation, const float *pcm, size_t sample_count,
    uint32_t sample_rate, uint64_t *out_position);
LFM_ORACLE_API int lfm_conversation_audio_frame(
    LfmConversation *conversation, LfmAudioResultV1 *out);
LFM_ORACLE_API int lfm_conversation_reset(LfmConversation *conversation);
LFM_ORACLE_API int lfm_conversation_close(LfmConversation *conversation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MODEL_LEGACY_H */
