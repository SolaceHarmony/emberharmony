#ifndef LFM_RUNTIME_H
#define LFM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_types.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/* `payload_read_coverage` says which setup-time payload sources contribute to
 * the counters below. `LFM_MODEL_ACCOUNTING_PAYLOAD_READS_COMPLETE` is a
 * separate claim: it must remain clear until every possible source is routed
 * through the rejecting owner-scoped recorder. This prevents partial zero
 * counters from masquerading as a complete post-publication read gate. */
#define LFM_MODEL_PAYLOAD_READ_CONFIG (1u << 0)
#define LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE (1u << 1)
#define LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX (1u << 2)
#define LFM_MODEL_PAYLOAD_READ_TOKENIZER (1u << 3)
#define LFM_MODEL_ACCOUNTING_PAYLOAD_READS_COMPLETE (1u << 0)

typedef struct LfmModelMemory {
    uint64_t source_bytes;
    uint64_t segment_bytes;
    uint64_t segment_constructed_bytes;
    uint64_t attached_shared_bytes;
    uint64_t wired_bytes;
    uint64_t process_resident_bytes;
    uint64_t directly_bound_bytes;
    uint64_t derived_immutable_bytes;
    uint64_t materialized_weight_bytes;
    uint64_t compatibility_copied_bytes;
    uint64_t payload_read_calls;
    uint64_t payload_read_bytes;
    /* Attempted after publication and rejected before the recorder performs
     * its payload operation. These are attempts, not completed reads. */
    uint64_t post_publication_read_calls;
    uint64_t post_publication_read_bytes;
    uint64_t post_publication_materialization_attempts;
    uint64_t post_publication_materialization_bytes;
    /* Zero while the model is private construction state; one after its only
     * successful publication. The generation belongs to this model object. */
    uint64_t publication_generation;
    uint64_t weight_build_ns;
    uint64_t weight_attach_ns;
    uint64_t weight_generation;
    uint64_t load_ns;
    uint32_t load_workers;
    uint32_t load_tasks;
    uint32_t payload_read_coverage;
    uint32_t accounting_flags;
    uint32_t weight_flags;
    uint32_t weight_source_count;
    uint64_t weight_payload_read_calls;
    uint64_t weight_payload_read_bytes;
    /* A conversation seals its numerical allocation geometry after its first
     * complete capture-plus-playback preparation. These counters aggregate
     * later rejected attempts across the model's conversations. Bytes are the
     * logical numerical capacity requested at the rejecting boundary, not
     * allocator metadata or an estimate of opaque object overhead. */
    uint64_t post_readiness_allocation_attempts;
    uint64_t post_readiness_allocation_bytes;
    uint8_t weight_identity_digest[32];
    uint8_t weight_content_digest[32];
} LfmModelMemory;

typedef struct LfmRuntimeConfig {
    uint32_t coordination_workers;
    uint32_t kernel_lanes;
    uint32_t event_capacity;
    uint32_t session_capacity;
    uint64_t flags;
} LfmRuntimeConfig;

typedef struct LfmRuntimeSnapshot {
    uint64_t runtime_epoch;
    uint32_t state;
    uint32_t kernel_lanes;
    uint32_t live_models;
    uint32_t live_sessions;
} LfmRuntimeSnapshot;

/* Sampling is control policy, not a numerical model surface. Logits, sampler
 * scratch, and PRNG state stay inside the opaque conversation. */
typedef struct LfmSamplingPolicy {
    uint32_t flags;
    uint32_t top_k;
    double temperature;
} LfmSamplingPolicy;

#define LFM_SAMPLING_GREEDY 1u

typedef struct LfmConversationOptions {
    uint32_t flags;
    uint64_t seed;
    LfmSamplingPolicy text;
    LfmSamplingPolicy audio;
} LfmConversationOptions;

#define LFM_CONVERSATION_SEED_SYSTEM 1u

#define LFM_RUNTIME_CREATED 0u
#define LFM_RUNTIME_STARTED 1u
#define LFM_RUNTIME_STOPPING 2u
#define LFM_RUNTIME_JOINED 3u

LFM_PUBLIC_API int lfm_runtime_create(const LfmRuntimeConfig *config,
                                      LfmRuntime **out);
LFM_PUBLIC_API int lfm_runtime_start(LfmRuntime *runtime);
LFM_PUBLIC_API void lfm_runtime_request_stop(LfmRuntime *runtime);
LFM_PUBLIC_API int lfm_runtime_join(LfmRuntime *runtime);
LFM_PUBLIC_API int lfm_runtime_snapshot(const LfmRuntime *runtime,
                                        LfmRuntimeSnapshot *out);
LFM_PUBLIC_API int lfm_runtime_destroy(LfmRuntime *runtime);

/* Product lifecycle open. It publishes a model child only after validating
 * complete native LFM2 voice ownership and zero compatibility-copied weights;
 * the executor, tensor schema, and resident image remain private. */
LFM_PUBLIC_API int lfm_runtime_model_open(
    LfmRuntime *runtime, const char *path, LfmModel **out, char *error,
    size_t error_length);
/* Lifecycle-only accounting query. The runtime must own `model`; no tensor,
 * shape, vocabulary, or numerical-plan metadata crosses this boundary. */
LFM_PUBLIC_API int lfm_runtime_model_memory(const LfmRuntime *runtime,
                                            const LfmModel *model,
                                            LfmModelMemory *out);
LFM_PUBLIC_API int lfm_runtime_model_close(LfmRuntime *runtime,
                                           LfmModel *model);

/* Runtime-scoped opaque conversation lifecycle. The runtime/model ownership
 * check is part of both calls, and an attached conversation cannot be closed
 * or attached to a second session. No token, tensor, cache, or codec state is
 * exposed through this boundary. */
LFM_PUBLIC_API int lfm_runtime_conversation_create(
    LfmRuntime *runtime, LfmModel *model,
    const LfmConversationOptions *options, LfmConversation **out,
    char *error, size_t error_length);
LFM_PUBLIC_API int
lfm_runtime_conversation_close(LfmRuntime *runtime,
                               LfmConversation *conversation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_RUNTIME_H */
