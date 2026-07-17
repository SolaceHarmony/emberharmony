#ifndef LFM_MODEL_H
#define LFM_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "flashkern_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_MODEL_ABI_VERSION 2u

typedef struct LfmModel LfmModel;
typedef struct LfmConversation LfmConversation;

#define LFM_CONVERSATION_SEED_SYSTEM 1u
#define LFM_MODEL_CAP_DEPTHFORMER 1u
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

/* Compact host-to-kernel input record. Payload tensors never cross this ABI:
 * one text record carries one vocabulary ID; one audio record carries the
 * already-offset codebook IDs whose resident embedding rows are summed. */
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

/* Open config.json and the resident safetensors image below `path`, validate
 * every backbone weight, and publish one immutable plan to `engine`. */
int lfm_model_open(void *engine, const char *path, LfmModel **out,
                   char *error, size_t error_length);

/* Clears the executor plan before releasing its resident weight image. A busy
 * executor returns -EBUSY and leaves the handle fully live for a later retry. */
int lfm_model_close(LfmModel *model);

int lfm_model_info(const LfmModel *model, LfmModelInfoV1 *out);

/* Native conversation state owns KV, short-convolution carry, rotary tables,
 * hidden/logit scratch, and PRNG state. Rust supplies only compact token IDs. */
int lfm_conversation_create(LfmModel *model,
                            const LfmConversationConfigV1 *config,
                            LfmConversation **out,
                            char *error, size_t error_length);
int lfm_conversation_step(LfmConversation *conversation,
                          const uint32_t *ids, size_t id_count,
                          uint32_t embedding_kind,
                          LfmTokenResultV1 *out);
int lfm_conversation_prefill(LfmConversation *conversation,
                             const LfmInputV1 *inputs, size_t input_count,
                             uint64_t *out_position);
/* Audio-in prefill: `rows` is a borrowed [row_count, hidden] bf16 view (the
 * Conformer/adapter output); each row is prefilled via the provided-embedding
 * pass. No payload crosses the ABI. */
int lfm_conversation_prefill_audio(LfmConversation *conversation,
                                   const uint16_t *rows, size_t row_count,
                                   uint64_t *out_position);
int lfm_conversation_audio_frame(LfmConversation *conversation,
                                 LfmAudioResultV1 *out);
int lfm_conversation_reset(LfmConversation *conversation);
int lfm_conversation_close(LfmConversation *conversation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MODEL_H */
