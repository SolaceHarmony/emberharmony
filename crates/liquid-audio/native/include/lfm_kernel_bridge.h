#ifndef LFM_KERNEL_BRIDGE_H
#define LFM_KERNEL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

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

typedef struct KcTicketIdV1 {
    uint64_t runtime_epoch;
    uint64_t sequence;
    uint32_t generation;
    uint32_t kind;
} KcTicketIdV1;

typedef struct KcDescriptorIdV1 {
    uint32_t slot;
    uint32_t generation;
} KcDescriptorIdV1;

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
    uint64_t deadline_ns;
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
    uint32_t active_waits;
    uint32_t reserved;
} LfmKernelBridgeSnapshotV1;

/*
 * Rust side. One broker is the sole SQ producer and one ingress thread is the
 * sole CQ consumer. Accepted submissions reserve one CQ cell until consumed.
 * A nonzero deadline is an absolute monotonic timestamp; zero waits forever.
 */
int lfm_kernel_bridge_create(const LfmKernelBridgeConfigV1 *config,
                             LfmKernelBridge **out);
int lfm_kernel_bridge_submit(LfmKernelBridge *bridge,
                             const KcSubmissionV1 *submission);
int lfm_kernel_bridge_wait_completion(LfmKernelBridge *bridge,
                                      KcCompletionV1 *out,
                                      uint64_t deadline_ns);

/*
 * Native side. One executor is the sole SQ consumer and logical CQ producer.
 * These functions perform no policy work. A nonzero deadline is absolute
 * monotonic time; zero waits forever.
 */
int lfm_kernel_bridge_wait_submission(LfmKernelBridge *bridge,
                                      KcSubmissionV1 *out,
                                      uint64_t deadline_ns);
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
LFM_KERNEL_STATIC_ASSERT(offsetof(KcSubmissionV1, deadline_ns) == 96,
                         "KcSubmissionV1 deadline offset");
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

#undef LFM_KERNEL_ALIGNAS
#undef LFM_KERNEL_ALIGNOF
#undef LFM_KERNEL_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif
