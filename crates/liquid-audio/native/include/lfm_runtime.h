#ifndef LFM_RUNTIME_H
#define LFM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_types.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The immutable native model/accounting ABI remains versioned independently
 * from the runtime lifecycle ABI. Numerical model metadata is intentionally not
 * part of this header. */
#define LFM_MODEL_ABI_VERSION 4u

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

typedef struct LfmModelMemoryV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t source_bytes;
    uint64_t resident_image_bytes;
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
    uint64_t load_ns;
    uint32_t load_workers;
    uint32_t load_tasks;
    uint32_t payload_read_coverage;
    uint32_t accounting_flags;
    /* A conversation seals its numerical allocation geometry after its first
     * complete capture-plus-playback preparation. These counters aggregate
     * later rejected attempts across the model's conversations. Bytes are the
     * logical numerical capacity requested at the rejecting boundary, not
     * allocator metadata or an estimate of opaque object overhead. */
    uint64_t post_readiness_allocation_attempts;
    uint64_t post_readiness_allocation_bytes;
    uint64_t reserved[2];
} LfmModelMemoryV1;

typedef struct LfmRuntimeConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t coordination_workers;
    uint32_t kernel_lanes;
    uint32_t event_capacity;
    uint32_t session_capacity;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t flags;
    uint64_t reserved[4];
} LfmRuntimeConfigV1;

typedef struct LfmRuntimeSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t runtime_epoch;
    uint32_t state;
    uint32_t kernel_lanes;
    uint32_t live_models;
    uint32_t live_sessions;
    uint64_t reserved[4];
} LfmRuntimeSnapshotV1;

/* Sampling is control policy, not a numerical model surface. Logits, sampler
 * scratch, and PRNG state stay inside the opaque conversation. */
typedef struct LfmSamplingPolicyV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t top_k;
    double temperature;
    uint64_t reserved;
} LfmSamplingPolicyV1;

#define LFM_SAMPLING_GREEDY 1u

typedef struct LfmConversationOptionsV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t seed;
    LfmSamplingPolicyV1 text;
    LfmSamplingPolicyV1 audio;
    uint64_t reserved[4];
} LfmConversationOptionsV1;

#define LFM_CONVERSATION_SEED_SYSTEM 1u

#define LFM_RUNTIME_CREATED 0u
#define LFM_RUNTIME_STARTED 1u
#define LFM_RUNTIME_STOPPING 2u
#define LFM_RUNTIME_JOINED 3u

LFM_PUBLIC_API int lfm_runtime_create(const LfmRuntimeConfigV1 *config,
                                      LfmRuntime **out);
LFM_PUBLIC_API int lfm_runtime_start(LfmRuntime *runtime);
LFM_PUBLIC_API void lfm_runtime_request_stop(LfmRuntime *runtime);
LFM_PUBLIC_API int lfm_runtime_join(LfmRuntime *runtime);
LFM_PUBLIC_API int lfm_runtime_snapshot(const LfmRuntime *runtime,
                                        LfmRuntimeSnapshotV1 *out);
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
                                            LfmModelMemoryV1 *out);
LFM_PUBLIC_API int lfm_runtime_model_close(LfmRuntime *runtime,
                                           LfmModel *model);

/* Runtime-scoped opaque conversation lifecycle. The runtime/model ownership
 * check is part of both calls, and an attached conversation cannot be closed
 * or attached to a second session. No token, tensor, cache, or codec state is
 * exposed through this boundary. */
LFM_PUBLIC_API int lfm_runtime_conversation_create(
    LfmRuntime *runtime, LfmModel *model,
    const LfmConversationOptionsV1 *options, LfmConversation **out,
    char *error, size_t error_length);
LFM_PUBLIC_API int
lfm_runtime_conversation_close(LfmRuntime *runtime,
                               LfmConversation *conversation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_RUNTIME_H */
