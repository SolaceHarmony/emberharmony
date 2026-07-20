#ifndef LFM_KERNEL_BRIDGE_H
#define LFM_KERNEL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "kc_identity.h"

#ifdef __cplusplus
extern "C" {
#define LFM_KERNEL_ALIGNAS(n) alignas(n)
#define LFM_KERNEL_ALIGNOF(type) alignof(type)
#define LFM_KERNEL_STATIC_ASSERT(test, message) static_assert(test, message)
#else
#define LFM_KERNEL_ALIGNAS(n) __attribute__((aligned(n)))
#define LFM_KERNEL_ALIGNOF(type) _Alignof(type)
#define LFM_KERNEL_STATIC_ASSERT(test, message) _Static_assert(test, message)
#endif

#define KC_COORD_ABI_VERSION 1u
#define KC_COORD_MAX_RESULTS 8u

#define KC_COORD_TICKET_SESSION KC_TICKET_KIND_SESSION
#define KC_COORD_TICKET_TURN KC_TICKET_KIND_TURN
#define KC_COORD_TICKET_FRAME KC_TICKET_KIND_FRAME
#define KC_COORD_TICKET_PASS KC_TICKET_KIND_PASS
#define KC_COORD_TICKET_CONTEXT_SWITCH KC_TICKET_KIND_CONTEXT_SWITCH
#define KC_COORD_TICKET_CHECKPOINT KC_TICKET_KIND_CHECKPOINT
#define KC_COORD_TICKET_WORKFLOW KC_TICKET_KIND_WORKFLOW

#define KC_COORD_COMMAND_RUN_PASS 1u
#define KC_COORD_COMMAND_RUN_STANDING_ORDER 2u
#define KC_COORD_COMMAND_SET_ATTENTION 3u
#define KC_COORD_COMMAND_PAUSE 4u
#define KC_COORD_COMMAND_RESUME 5u
#define KC_COORD_COMMAND_CANCEL 6u
#define KC_COORD_COMMAND_STOP 7u

#define KC_COORD_SERVICE_REALTIME 1u
#define KC_COORD_SERVICE_INTERACTIVE 2u
#define KC_COORD_SERVICE_BACKGROUND 3u

#define KC_COORD_EXECUTION_NOT_DISPATCHED 0u
#define KC_COORD_EXECUTION_COMPLETED 1u
#define KC_COORD_EXECUTION_FAILED 2u

#define KC_COORD_STATE_NONE 0u
#define KC_COORD_STATE_COMMITTED 1u
#define KC_COORD_STATE_ROLLED_BACK 2u
#define KC_COORD_STATE_POISONED 3u

#define KC_COORD_PUBLICATION_NONE 0u
#define KC_COORD_PUBLICATION_COMMITTED 1u
#define KC_COORD_PUBLICATION_STALE 2u

#define KC_COORD_CAUSE_SUCCESS 0u
#define KC_COORD_CAUSE_REJECTED 1u
#define KC_COORD_CAUSE_CANCELED 2u
#define KC_COORD_CAUSE_TIMED_OUT 3u
#define KC_COORD_CAUSE_STALE_EPOCH 4u
#define KC_COORD_CAUSE_STOP 5u
#define KC_COORD_CAUSE_FAULT 6u

#define KC_COORD_RESULT_NONE 0u
#define KC_COORD_RESULT_TEXT_TOKEN 1u
#define KC_COORD_RESULT_AUDIO_CODES 2u
#define KC_COORD_RESULT_FRAME 3u
#define KC_COORD_RESULT_CONTROL 4u

typedef kc_ticket_id KcTicketIdV1;

/* ABI name retained, but this is not a generic descriptor or owning object.
 * It is the exact fixed pass-slot index plus ticket generation. The slot owns
 * the typed byte views and durable continuation state. */
typedef struct KcDescriptorIdV1 {
    uint32_t slot;
    uint32_t generation;
} KcDescriptorIdV1;

/* ABI-v1 values require 64-byte alignment. Internal SQ/CQ storage wraps each
 * value in a 128-byte-aligned cell on Apple instead of strengthening this
 * caller-facing contract in place. */
typedef struct LFM_KERNEL_ALIGNAS(64) KcSubmissionV1 {
    uint32_t size;
    uint32_t abi_version;
    KcTicketIdV1 ticket;
    KcTicketIdV1 parent;
    uint64_t conversation_id;
    uint64_t epoch;
    KcDescriptorIdV1 descriptor;
    uint32_t command;
    uint32_t service_class;
    uint32_t flags;
    uint32_t pass_budget;
    uint64_t reserved0;
    uint64_t reserved[3];
} KcSubmissionV1;

typedef struct LFM_KERNEL_ALIGNAS(64) KcCompletionV1 {
    uint32_t size;
    uint32_t abi_version;
    KcTicketIdV1 ticket;
    uint64_t conversation_id;
    uint64_t epoch;
    uint64_t pass_id;
    uint32_t execution;
    uint32_t state;
    uint32_t publication;
    uint32_t cause;
    int32_t status;
    uint32_t flags;
    uint32_t result_kind;
    uint32_t result_count;
    uint32_t results[KC_COORD_MAX_RESULTS];
    uint64_t reserved;
} KcCompletionV1;

typedef struct LfmKernelBridge LfmKernelBridge;

typedef struct LfmKernelBridgeConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t capacity;
    uint32_t reserved;
} LfmKernelBridgeConfigV1;

typedef struct LfmKernelBridgeSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t capacity;
    uint32_t stopping;
    uint64_t submissions_accepted;
    uint64_t submissions_consumed;
    uint64_t completions_published;
    uint64_t completions_consumed;
    uint32_t reserved[2];
} LfmKernelBridgeSnapshotV1;

/*
 * Native submission side. The model runtime is the sole SQ producer and CQ
 * consumer in production. Rust bindings exercise this protocol only in tests.
 * Accepted submissions reserve one CQ cell until consumed. Queue operations
 * never wait: the owner resumes its retained service on publication and drains
 * with the `try` operations below.
 */
int lfm_kernel_bridge_create(const LfmKernelBridgeConfigV1 *config,
                             LfmKernelBridge **out);
int lfm_kernel_bridge_submit(LfmKernelBridge *bridge,
                             const KcSubmissionV1 *submission);
/* Nonblocking CQ receive. Returns -EAGAIN while the queue is empty and
 * -ECANCELED only after stop has closed submission admission and every
 * accepted ticket has settled. */
int lfm_kernel_bridge_try_completion(LfmKernelBridge *bridge,
                                     KcCompletionV1 *out);

/*
 * Native side. One executor is the sole SQ consumer and logical CQ producer.
 * These functions perform no policy work and never wait.
 */
int lfm_kernel_bridge_try_submission(LfmKernelBridge *bridge,
                                     KcSubmissionV1 *out);
int lfm_kernel_bridge_publish_completion(LfmKernelBridge *bridge,
                                         const KcCompletionV1 *completion);

void lfm_kernel_bridge_request_stop(LfmKernelBridge *bridge);
int lfm_kernel_bridge_snapshot(LfmKernelBridge *bridge,
                               LfmKernelBridgeSnapshotV1 *out);
/* All four endpoint owners must be joined and every accepted ticket drained. */
int lfm_kernel_bridge_destroy(LfmKernelBridge *bridge);

LFM_KERNEL_STATIC_ASSERT(sizeof(KcTicketIdV1) == 24, "KcTicketIdV1 size");
LFM_KERNEL_STATIC_ASSERT(sizeof(KcDescriptorIdV1) == 8, "KcDescriptorIdV1 size");
LFM_KERNEL_STATIC_ASSERT(sizeof(KcSubmissionV1) == 128, "KcSubmissionV1 size");
LFM_KERNEL_STATIC_ASSERT(LFM_KERNEL_ALIGNOF(KcSubmissionV1) == 64,
                         "KcSubmissionV1 alignment");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, ticket) == 8,
                         "KcSubmissionV1 ticket offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, parent) == 32,
                         "KcSubmissionV1 parent offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, conversation_id) == 56,
                         "KcSubmissionV1 conversation offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, descriptor) == 72,
                         "KcSubmissionV1 descriptor offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, reserved0) == 96,
                         "KcSubmissionV1 reserved offset");
LFM_KERNEL_STATIC_ASSERT(sizeof(KcCompletionV1) == 128, "KcCompletionV1 size");
LFM_KERNEL_STATIC_ASSERT(LFM_KERNEL_ALIGNOF(KcCompletionV1) == 64,
                         "KcCompletionV1 alignment");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcCompletionV1, ticket) == 8,
                         "KcCompletionV1 ticket offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcCompletionV1, conversation_id) == 32,
                         "KcCompletionV1 conversation offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcCompletionV1, execution) == 56,
                         "KcCompletionV1 execution offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcCompletionV1, results) == 88,
                         "KcCompletionV1 results offset");
LFM_KERNEL_STATIC_ASSERT(offsetof(KcCompletionV1, reserved) == 120,
                         "KcCompletionV1 reserved offset");
LFM_KERNEL_STATIC_ASSERT(sizeof(LfmKernelBridgeSnapshotV1) == 56,
                         "LfmKernelBridgeSnapshotV1 size");

#undef LFM_KERNEL_ALIGNAS
#undef LFM_KERNEL_ALIGNOF
#undef LFM_KERNEL_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif
