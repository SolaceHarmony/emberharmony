#include "lfm_kernel_bridge.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

constexpr uint32_t ADMISSION_PUBLISHER = UINT32_C(1);
constexpr uint32_t ADMISSION_STOP = UINT32_C(1) << 31;
/* Apple arm64 and Rosetta execute on the same 128-byte cache-line hardware. */
constexpr size_t HOT_ATOMIC_BYTES = 128;

struct alignas(HOT_ATOMIC_BYTES) Cursor {
    std::atomic<uint64_t> value{0};
};
static_assert(alignof(Cursor) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Cursor) == HOT_ATOMIC_BYTES,
              "adjacent queue cursors must not share an Apple cache line");

template <typename T>
struct alignas(HOT_ATOMIC_BYTES) QueueCell {
    T value{};
};
static_assert(alignof(QueueCell<KcSubmissionV1>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(QueueCell<KcSubmissionV1>) == HOT_ATOMIC_BYTES);
static_assert(alignof(QueueCell<KcCompletionV1>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(QueueCell<KcCompletionV1>) == HOT_ATOMIC_BYTES,
              "SQ/CQ storage must isolate ABI-v1 values without changing their alignment");

bool ticket_equal(const KcTicketIdV1 &a, const KcTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

bool ticket_valid(const KcTicketIdV1 &ticket) {
    return ticket.runtime_epoch != 0 && ticket.sequence != 0 &&
           ticket.generation != 0 &&
           ticket.kind >= KC_COORD_TICKET_SESSION &&
           ticket.kind <= KC_COORD_TICKET_WORKFLOW;
}

bool ticket_none(const KcTicketIdV1 &ticket) {
    return ticket.runtime_epoch == 0 && ticket.sequence == 0 &&
           ticket.generation == 0 && ticket.kind == 0;
}

bool descriptor_required(uint32_t command) {
    return command == KC_COORD_COMMAND_RUN_PASS ||
           command == KC_COORD_COMMAND_RUN_STANDING_ORDER;
}

bool submission_valid(const KcSubmissionV1 *submission, uint32_t capacity) {
    if (!submission || submission->size != sizeof(*submission) ||
        submission->abi_version != KC_COORD_ABI_VERSION ||
        !ticket_valid(submission->ticket) ||
        (!ticket_none(submission->parent) && !ticket_valid(submission->parent)) ||
        submission->command < KC_COORD_COMMAND_RUN_PASS ||
        submission->command > KC_COORD_COMMAND_STOP ||
        submission->service_class < KC_COORD_SERVICE_REALTIME ||
        submission->service_class > KC_COORD_SERVICE_BACKGROUND ||
        submission->flags != 0 || submission->pass_budget == 0 ||
        submission->reserved0 != 0 || submission->reserved[0] != 0 ||
        submission->reserved[1] != 0 || submission->reserved[2] != 0) {
        return false;
    }
    if (!descriptor_required(submission->command)) return true;
    return submission->descriptor.slot < capacity &&
           submission->descriptor.generation != 0;
}

bool completion_valid(const KcCompletionV1 *completion) {
    return completion && completion->size == sizeof(*completion) &&
           completion->abi_version == KC_COORD_ABI_VERSION &&
           ticket_valid(completion->ticket) &&
           completion->execution <= KC_COORD_EXECUTION_FAILED &&
           completion->state <= KC_COORD_STATE_POISONED &&
           completion->publication <= KC_COORD_PUBLICATION_STALE &&
           completion->cause <= KC_COORD_CAUSE_FAULT &&
           completion->result_kind <= KC_COORD_RESULT_CONTROL &&
           completion->result_count <= KC_COORD_MAX_RESULTS &&
           completion->reserved == 0;
}

} // namespace

struct LfmKernelBridge {
    uint32_t capacity = 0;
    QueueCell<KcSubmissionV1> *submissions = nullptr;
    QueueCell<KcCompletionV1> *completions = nullptr;
    KcTicketIdV1 *ledger = nullptr;
    Cursor submission_head;
    Cursor submission_tail;
    Cursor completion_head;
    Cursor completion_tail;
    /* One structural SQ publisher plus a stop bit. This is an admission gate,
     * not a lock: a competing publisher receives backpressure immediately. */
    std::atomic<uint32_t> admission{0};
};

namespace {

bool stopping(const LfmKernelBridge *bridge) {
    return (bridge->admission.load(std::memory_order_seq_cst) & ADMISSION_STOP) != 0;
}

bool submissions_settled(const LfmKernelBridge *bridge) {
    return (bridge->admission.load(std::memory_order_seq_cst) &
            ADMISSION_PUBLISHER) == 0;
}

bool enter_submission(LfmKernelBridge *bridge) {
    uint32_t idle = 0;
    return bridge->admission.compare_exchange_strong(
        idle, ADMISSION_PUBLISHER, std::memory_order_seq_cst,
        std::memory_order_seq_cst);
}

void leave_submission(LfmKernelBridge *bridge) {
    const uint32_t prior = bridge->admission.fetch_and(
        ~ADMISSION_PUBLISHER, std::memory_order_seq_cst);
    if ((prior & ADMISSION_PUBLISHER) == 0) std::abort();
}

int take_submission(LfmKernelBridge *bridge, KcSubmissionV1 *out) {
    const uint64_t head =
        bridge->submission_head.value.load(std::memory_order_relaxed);
    const uint64_t tail =
        bridge->submission_tail.value.load(std::memory_order_acquire);
    if (head == tail) return -EAGAIN;
    *out = bridge->submissions[head % bridge->capacity].value;
    bridge->submission_head.value.store(head + 1, std::memory_order_release);
    return 0;
}

int take_completion(LfmKernelBridge *bridge, KcCompletionV1 *out) {
    const uint64_t head =
        bridge->completion_head.value.load(std::memory_order_relaxed);
    const uint64_t tail =
        bridge->completion_tail.value.load(std::memory_order_acquire);
    if (head == tail) return -EAGAIN;
    const size_t index = head % bridge->capacity;
    *out = bridge->completions[index].value;
    bridge->ledger[index] = {};
    bridge->completion_head.value.store(head + 1, std::memory_order_release);
    return 0;
}

} // namespace

extern "C" {

int lfm_kernel_bridge_create(const LfmKernelBridgeConfigV1 *config,
                             LfmKernelBridge **out) {
    if (!config || !out || config->size != sizeof(*config) ||
        config->abi_version != KC_COORD_ABI_VERSION || config->capacity == 0 ||
        config->reserved != 0) {
        return -EINVAL;
    }

    LfmKernelBridge *bridge = new (std::nothrow) LfmKernelBridge();
    if (!bridge) return -ENOMEM;
    bridge->capacity = config->capacity;
    bridge->submissions =
        new (std::nothrow) QueueCell<KcSubmissionV1>[bridge->capacity];
    bridge->completions =
        new (std::nothrow) QueueCell<KcCompletionV1>[bridge->capacity];
    bridge->ledger = new (std::nothrow) KcTicketIdV1[bridge->capacity];
    if (!bridge->submissions || !bridge->completions || !bridge->ledger) {
        delete[] bridge->ledger;
        delete[] bridge->completions;
        delete[] bridge->submissions;
        delete bridge;
        return -ENOMEM;
    }
    std::memset(bridge->submissions, 0,
                sizeof(*bridge->submissions) * bridge->capacity);
    std::memset(bridge->completions, 0,
                sizeof(*bridge->completions) * bridge->capacity);
    std::memset(bridge->ledger, 0, sizeof(*bridge->ledger) * bridge->capacity);
    *out = bridge;
    return 0;
}

int lfm_kernel_bridge_submit(LfmKernelBridge *bridge,
                             const KcSubmissionV1 *submission) {
    if (!bridge || !submission_valid(submission, bridge->capacity)) return -EINVAL;
    if (!enter_submission(bridge)) return stopping(bridge) ? -ECANCELED : -EAGAIN;

    const uint64_t tail =
        bridge->submission_tail.value.load(std::memory_order_relaxed);
    const uint64_t completed =
        bridge->completion_head.value.load(std::memory_order_acquire);
    const uint64_t head =
        bridge->submission_head.value.load(std::memory_order_acquire);
    if (tail - completed >= bridge->capacity ||
        tail - head >= bridge->capacity) {
        leave_submission(bridge);
        return -EAGAIN;
    }

    const size_t index = tail % bridge->capacity;
    bridge->ledger[index] = submission->ticket;
    bridge->submissions[index].value = *submission;
    bridge->submission_tail.value.store(tail + 1, std::memory_order_release);
    leave_submission(bridge);
    return 0;
}

int lfm_kernel_bridge_try_submission(LfmKernelBridge *bridge,
                                     KcSubmissionV1 *out) {
    if (!bridge || !out) return -EINVAL;
    const int status = take_submission(bridge, out);
    if (status != -EAGAIN) return status;
    return stopping(bridge) && submissions_settled(bridge) &&
                   bridge->submission_head.value.load(std::memory_order_acquire) ==
                       bridge->submission_tail.value.load(std::memory_order_acquire)
        ? -ECANCELED
        : -EAGAIN;
}

int lfm_kernel_bridge_publish_completion(LfmKernelBridge *bridge,
                                         const KcCompletionV1 *completion) {
    if (!bridge || !completion_valid(completion)) return -EINVAL;
    const uint64_t tail =
        bridge->completion_tail.value.load(std::memory_order_relaxed);
    const uint64_t dispatched =
        bridge->submission_head.value.load(std::memory_order_acquire);
    if (tail == dispatched) return -EAGAIN;
    const uint64_t head =
        bridge->completion_head.value.load(std::memory_order_acquire);
    if (tail - head >= bridge->capacity) return -EOVERFLOW;

    const size_t index = tail % bridge->capacity;
    if (!ticket_equal(bridge->ledger[index], completion->ticket)) return -ESTALE;
    bridge->completions[index].value = *completion;
    bridge->completion_tail.value.store(tail + 1, std::memory_order_release);
    return 0;
}

int lfm_kernel_bridge_try_completion(LfmKernelBridge *bridge,
                                     KcCompletionV1 *out) {
    if (!bridge || !out) return -EINVAL;
    const int status = take_completion(bridge, out);
    if (status != -EAGAIN) return status;
    return stopping(bridge) && submissions_settled(bridge) &&
                   bridge->completion_head.value.load(std::memory_order_acquire) ==
                       bridge->submission_tail.value.load(std::memory_order_acquire)
        ? -ECANCELED
        : -EAGAIN;
}

void lfm_kernel_bridge_request_stop(LfmKernelBridge *bridge) {
    if (!bridge) return;
    bridge->admission.fetch_or(ADMISSION_STOP, std::memory_order_seq_cst);
}

int lfm_kernel_bridge_snapshot(LfmKernelBridge *bridge,
                               LfmKernelBridgeSnapshotV1 *out) {
    if (!bridge || !out || out->size < sizeof(*out) ||
        out->abi_version != KC_COORD_ABI_VERSION) {
        return -EINVAL;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = KC_COORD_ABI_VERSION,
        .capacity = bridge->capacity,
        .stopping = stopping(bridge) ? 1u : 0u,
        .submissions_accepted =
            bridge->submission_tail.value.load(std::memory_order_acquire),
        .submissions_consumed =
            bridge->submission_head.value.load(std::memory_order_acquire),
        .completions_published =
            bridge->completion_tail.value.load(std::memory_order_acquire),
        .completions_consumed =
            bridge->completion_head.value.load(std::memory_order_acquire),
        .reserved = {0, 0},
    };
    return 0;
}

int lfm_kernel_bridge_destroy(LfmKernelBridge *bridge) {
    if (!bridge) return -EINVAL;
    if (!stopping(bridge) || !submissions_settled(bridge) ||
        bridge->submission_head.value.load(std::memory_order_acquire) !=
            bridge->submission_tail.value.load(std::memory_order_acquire) ||
        bridge->completion_head.value.load(std::memory_order_acquire) !=
            bridge->completion_tail.value.load(std::memory_order_acquire) ||
        bridge->completion_head.value.load(std::memory_order_acquire) !=
            bridge->submission_tail.value.load(std::memory_order_acquire)) {
        return -EBUSY;
    }
    delete[] bridge->ledger;
    delete[] bridge->completions;
    delete[] bridge->submissions;
    delete bridge;
    return 0;
}

} // extern "C"
