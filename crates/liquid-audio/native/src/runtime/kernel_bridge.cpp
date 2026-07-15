#include "lfm_kernel_bridge.h"

#include "kc_atomic.h"
#include "kc_port.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>

namespace {

constexpr uint32_t ADMISSION_STOP = UINT32_C(1) << 31;
constexpr uint32_t ADMISSION_COUNT = ADMISSION_STOP - 1;
constexpr uint32_t DESCRIPTOR_FREE = 0;
constexpr uint32_t DESCRIPTOR_LIVE = 1;
constexpr uint32_t DESCRIPTOR_RELEASING = 2;
constexpr uint32_t DESCRIPTOR_RETIRED = 3;

struct alignas(64) Cursor {
    std::atomic<uint64_t> value{0};
};

struct alignas(64) Doorbell {
    uint32_t value = 0;
    kc_port_wait_word *wait = nullptr;
};

bool ticket_equal(const KcTicketIdV1 &a, const KcTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

bool ticket_valid(const KcTicketIdV1 &ticket) {
    return ticket.runtime_epoch != 0 && ticket.sequence != 0 && ticket.generation != 0 &&
           ticket.kind >= KC_COORD_TICKET_SESSION && ticket.kind <= KC_COORD_TICKET_WORKFLOW;
}

bool ticket_none(const KcTicketIdV1 &ticket) {
    return ticket.runtime_epoch == 0 && ticket.sequence == 0 && ticket.generation == 0 &&
           ticket.kind == 0;
}

bool descriptor_required(uint32_t command) {
    return command == KC_COORD_COMMAND_RUN_PASS ||
           command == KC_COORD_COMMAND_RUN_STANDING_ORDER;
}

KcDescriptorIdV1 descriptor_none() {
    return {.slot = UINT32_MAX, .generation = 0};
}

bool descriptor_spec_valid(const LfmKernelDescriptorSpecV1 *spec) {
    return spec && spec->size == sizeof(*spec) &&
           spec->abi_version == KC_COORD_ABI_VERSION && spec->kind != 0 &&
           spec->payload && spec->reserved[0] == 0 && spec->reserved[1] == 0 &&
           spec->reserved[2] == 0;
}

bool submission_valid(const KcSubmissionV1 *submission) {
    if (!submission || submission->size != sizeof(*submission) ||
        submission->abi_version != KC_COORD_ABI_VERSION ||
        !ticket_valid(submission->ticket) ||
        (!ticket_none(submission->parent) && !ticket_valid(submission->parent)) ||
        submission->command < KC_COORD_COMMAND_RUN_PASS ||
        submission->command > KC_COORD_COMMAND_STOP ||
        submission->service_class < KC_COORD_SERVICE_DEADLINE ||
        submission->service_class > KC_COORD_SERVICE_BACKGROUND ||
        submission->pass_budget == 0 ||
        submission->reserved[0] != 0 || submission->reserved[1] != 0 ||
        submission->reserved[2] != 0) {
        return false;
    }
    if (descriptor_required(submission->command)) {
        return submission->descriptor.slot != UINT32_MAX &&
               submission->descriptor.generation != 0;
    }
    return true;
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
           completion->result_count <= KC_COORD_MAX_RESULTS && completion->reserved == 0;
}

void ring(Doorbell *doorbell, bool all) {
    kc_atomic_u32_fetch_add_release(&doorbell->value, 1);
    if (all) {
        kc_port_wake_u32_all(doorbell->wait);
        return;
    }
    kc_port_wake_u32_one(doorbell->wait);
}

} // namespace

struct LfmKernelBridge {
    struct DescriptorSlot {
        uint32_t generation = 0;
        uint32_t refs = 0;
        uint32_t state = DESCRIPTOR_FREE;
        uint32_t kind = 0;
        uint32_t flags = 0;
        void *payload = nullptr;
        void *context = nullptr;
        LfmKernelDescriptorReleaseFn release = nullptr;
    };

    uint32_t capacity = 0;
    uint32_t descriptor_capacity = 0;
    KcSubmissionV1 *submissions = nullptr;
    KcCompletionV1 *completions = nullptr;
    KcTicketIdV1 *ledger = nullptr;
    KcDescriptorIdV1 *descriptor_ledger = nullptr;
    DescriptorSlot *descriptors = nullptr;
    uint32_t *descriptor_free = nullptr;
    uint32_t descriptor_free_count = 0;
    uint32_t descriptors_live = 0;
    uint32_t descriptors_retired = 0;
    uint32_t descriptor_max_generation = 0;
    uint64_t descriptor_acquires = 0;
    uint64_t descriptor_retains = 0;
    uint64_t descriptor_releases = 0;
    uint64_t descriptor_callbacks = 0;
    std::mutex descriptor_mutex;
    Cursor submission_head;
    Cursor submission_tail;
    Cursor completion_head;
    Cursor completion_tail;
    Doorbell submission_doorbell;
    Doorbell completion_doorbell;
    std::atomic<uint32_t> admission{0};
    std::atomic<uint32_t> active_waits{0};
};

namespace {

bool descriptor_matches(const LfmKernelBridge *bridge, KcDescriptorIdV1 descriptor,
                        const LfmKernelBridge::DescriptorSlot &slot) {
    return descriptor.slot < bridge->descriptor_capacity && descriptor.generation != 0 &&
           slot.state == DESCRIPTOR_LIVE && slot.generation == descriptor.generation &&
           slot.refs != 0;
}

bool descriptors_settled(LfmKernelBridge *bridge) {
    std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
    return bridge->descriptors_live == 0;
}

int retain_descriptor(LfmKernelBridge *bridge, KcDescriptorIdV1 descriptor) {
    if (descriptor.slot >= bridge->descriptor_capacity || descriptor.generation == 0) {
        return -EINVAL;
    }
    std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
    LfmKernelBridge::DescriptorSlot &slot = bridge->descriptors[descriptor.slot];
    if (!descriptor_matches(bridge, descriptor, slot)) return -ESTALE;
    if (slot.refs == UINT32_MAX) return -EOVERFLOW;
    slot.refs++;
    bridge->descriptor_retains++;
    return 0;
}

bool enter_submission(LfmKernelBridge *bridge) {
    uint32_t state = bridge->admission.load(std::memory_order_acquire);
    for (;;) {
        if (state & ADMISSION_STOP || (state & ADMISSION_COUNT) == ADMISSION_COUNT) {
            return false;
        }
        if (bridge->admission.compare_exchange_weak(
                state, state + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return true;
        }
    }
}

bool stopping(const LfmKernelBridge *bridge) {
    return bridge->admission.load(std::memory_order_acquire) & ADMISSION_STOP;
}

bool submissions_settled(const LfmKernelBridge *bridge) {
    return (bridge->admission.load(std::memory_order_acquire) & ADMISSION_COUNT) == 0;
}

void leave_submission(LfmKernelBridge *bridge) {
    uint32_t previous = bridge->admission.fetch_sub(1, std::memory_order_acq_rel);
    if ((previous & ADMISSION_STOP) && (previous & ADMISSION_COUNT) == 1) {
        ring(&bridge->submission_doorbell, true);
        ring(&bridge->completion_doorbell, true);
    }
}

int take_submission(LfmKernelBridge *bridge, KcSubmissionV1 *out) {
    uint64_t head = bridge->submission_head.value.load(std::memory_order_relaxed);
    uint64_t tail = bridge->submission_tail.value.load(std::memory_order_acquire);
    if (head == tail) return -EAGAIN;
    *out = bridge->submissions[head % bridge->capacity];
    bridge->submission_head.value.store(head + 1, std::memory_order_release);
    return 0;
}

int take_completion(LfmKernelBridge *bridge, KcCompletionV1 *out) {
    uint64_t head = bridge->completion_head.value.load(std::memory_order_relaxed);
    uint64_t tail = bridge->completion_tail.value.load(std::memory_order_acquire);
    if (head == tail) return -EAGAIN;
    size_t index = head % bridge->capacity;
    *out = bridge->completions[index];
    KcDescriptorIdV1 descriptor = bridge->descriptor_ledger[index];
    bridge->descriptor_ledger[index] = descriptor_none();
    bridge->completion_head.value.store(head + 1, std::memory_order_release);
    if (descriptor.slot != UINT32_MAX &&
        lfm_kernel_bridge_descriptor_release(bridge, descriptor) != 0) {
        std::abort();
    }
    return 0;
}

template <typename Take, typename Done>
int wait_for_edge(LfmKernelBridge *bridge, Doorbell *doorbell, uint64_t deadline_ns,
                  Take &&take, Done &&done) {
    bridge->active_waits.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        int rc = take();
        if (rc == 0) {
            bridge->active_waits.fetch_sub(1, std::memory_order_acq_rel);
            return 0;
        }
        if (done()) {
            bridge->active_waits.fetch_sub(1, std::memory_order_acq_rel);
            return -ECANCELED;
        }
        uint32_t expected = kc_atomic_u32_load_acquire(&doorbell->value);
        rc = take();
        if (rc == 0) {
            bridge->active_waits.fetch_sub(1, std::memory_order_acq_rel);
            return 0;
        }
        if (done()) {
            bridge->active_waits.fetch_sub(1, std::memory_order_acq_rel);
            return -ECANCELED;
        }
        rc = kc_port_wait_u32(doorbell->wait, expected, deadline_ns);
        if (rc != 0) {
            bridge->active_waits.fetch_sub(1, std::memory_order_acq_rel);
            return rc;
        }
    }
}

} // namespace

extern "C" {

int lfm_kernel_bridge_create(const LfmKernelBridgeConfigV1 *config,
                             LfmKernelBridge **out) {
    if (!config || !out || config->size != sizeof(*config) ||
        config->abi_version != KC_COORD_ABI_VERSION || config->capacity == 0) {
        return -EINVAL;
    }

    LfmKernelBridge *bridge = new (std::nothrow) LfmKernelBridge();
    if (!bridge) return -ENOMEM;
    bridge->capacity = config->capacity;
    bridge->descriptor_capacity =
        config->descriptor_capacity == 0 ? config->capacity : config->descriptor_capacity;
    bridge->submissions = new (std::nothrow) KcSubmissionV1[bridge->capacity];
    bridge->completions = new (std::nothrow) KcCompletionV1[bridge->capacity];
    bridge->ledger = new (std::nothrow) KcTicketIdV1[bridge->capacity];
    bridge->descriptor_ledger = new (std::nothrow) KcDescriptorIdV1[bridge->capacity];
    bridge->descriptors =
        new (std::nothrow) LfmKernelBridge::DescriptorSlot[bridge->descriptor_capacity];
    bridge->descriptor_free = new (std::nothrow) uint32_t[bridge->descriptor_capacity];
    if (!bridge->submissions || !bridge->completions || !bridge->ledger ||
        !bridge->descriptor_ledger || !bridge->descriptors || !bridge->descriptor_free) {
        delete[] bridge->descriptor_free;
        delete[] bridge->descriptors;
        delete[] bridge->descriptor_ledger;
        delete[] bridge->ledger;
        delete[] bridge->completions;
        delete[] bridge->submissions;
        delete bridge;
        return -ENOMEM;
    }
    std::memset(bridge->submissions, 0, sizeof(*bridge->submissions) * bridge->capacity);
    std::memset(bridge->completions, 0, sizeof(*bridge->completions) * bridge->capacity);
    std::memset(bridge->ledger, 0, sizeof(*bridge->ledger) * bridge->capacity);
    for (uint32_t index = 0; index < bridge->capacity; ++index) {
        bridge->descriptor_ledger[index] = descriptor_none();
    }
    for (uint32_t index = 0; index < bridge->descriptor_capacity; ++index) {
        bridge->descriptor_free[index] = bridge->descriptor_capacity - index - 1;
    }
    bridge->descriptor_free_count = bridge->descriptor_capacity;

    int rc = kc_port_wait_u32_prepare(&bridge->submission_doorbell.value,
                                      &bridge->submission_doorbell.wait);
    if (rc == 0) {
        rc = kc_port_wait_u32_prepare(&bridge->completion_doorbell.value,
                                      &bridge->completion_doorbell.wait);
    }
    if (rc != 0) {
        if (bridge->submission_doorbell.wait) {
            kc_port_wait_u32_release(bridge->submission_doorbell.wait);
        }
        delete[] bridge->descriptor_free;
        delete[] bridge->descriptors;
        delete[] bridge->descriptor_ledger;
        delete[] bridge->ledger;
        delete[] bridge->completions;
        delete[] bridge->submissions;
        delete bridge;
        return rc;
    }
    *out = bridge;
    return 0;
}

int lfm_kernel_bridge_descriptor_create(
    LfmKernelBridge *bridge, const LfmKernelDescriptorSpecV1 *spec,
    KcDescriptorIdV1 *out) {
    if (!bridge || !out || !descriptor_spec_valid(spec)) return -EINVAL;
    if (!enter_submission(bridge)) return -ECANCELED;

    int rc = 0;
    {
        std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
        if (bridge->descriptor_free_count == 0) {
            rc = -ENOSPC;
        } else {
            uint32_t index =
                bridge->descriptor_free[--bridge->descriptor_free_count];
            LfmKernelBridge::DescriptorSlot &slot = bridge->descriptors[index];
            if (slot.state != DESCRIPTOR_FREE || slot.generation == UINT32_MAX) {
                std::abort();
            }
            slot.generation++;
            if (slot.generation == 0) std::abort();
            slot.refs = 1;
            slot.state = DESCRIPTOR_LIVE;
            slot.kind = spec->kind;
            slot.flags = spec->flags;
            slot.payload = spec->payload;
            slot.context = spec->context;
            slot.release = spec->release;
            bridge->descriptors_live++;
            bridge->descriptor_acquires++;
            if (slot.generation > bridge->descriptor_max_generation) {
                bridge->descriptor_max_generation = slot.generation;
            }
            *out = {.slot = index, .generation = slot.generation};
        }
    }
    leave_submission(bridge);
    return rc;
}

int lfm_kernel_bridge_descriptor_retain(LfmKernelBridge *bridge,
                                        KcDescriptorIdV1 descriptor) {
    if (!bridge) return -EINVAL;
    if (!enter_submission(bridge)) return -ECANCELED;
    int rc = retain_descriptor(bridge, descriptor);
    leave_submission(bridge);
    return rc;
}

int lfm_kernel_bridge_descriptor_get(LfmKernelBridge *bridge,
                                     KcDescriptorIdV1 descriptor,
                                     LfmKernelDescriptorViewV1 *out) {
    if (!bridge || !out || descriptor.slot >= bridge->descriptor_capacity ||
        descriptor.generation == 0 || out->size < sizeof(*out) ||
        out->abi_version != KC_COORD_ABI_VERSION) {
        return -EINVAL;
    }
    std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
    const LfmKernelBridge::DescriptorSlot &slot = bridge->descriptors[descriptor.slot];
    if (!descriptor_matches(bridge, descriptor, slot)) return -ESTALE;
    *out = {
        .size = sizeof(*out),
        .abi_version = KC_COORD_ABI_VERSION,
        .kind = slot.kind,
        .flags = slot.flags,
        .payload = slot.payload,
        .reserved = 0,
    };
    return 0;
}

int lfm_kernel_bridge_descriptor_release(LfmKernelBridge *bridge,
                                         KcDescriptorIdV1 descriptor) {
    if (!bridge || descriptor.slot >= bridge->descriptor_capacity ||
        descriptor.generation == 0) {
        return -EINVAL;
    }

    LfmKernelDescriptorReleaseFn release = nullptr;
    void *payload = nullptr;
    void *context = nullptr;
    {
        std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
        LfmKernelBridge::DescriptorSlot &slot = bridge->descriptors[descriptor.slot];
        if (!descriptor_matches(bridge, descriptor, slot)) return -ESTALE;
        bridge->descriptor_releases++;
        slot.refs--;
        if (slot.refs != 0) return 0;
        slot.state = DESCRIPTOR_RELEASING;
        release = slot.release;
        payload = slot.payload;
        context = slot.context;
    }

    if (release) release(payload, context);

    {
        std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
        LfmKernelBridge::DescriptorSlot &slot = bridge->descriptors[descriptor.slot];
        if (slot.state != DESCRIPTOR_RELEASING ||
            slot.generation != descriptor.generation || slot.refs != 0) {
            std::abort();
        }
        slot.kind = 0;
        slot.flags = 0;
        slot.payload = nullptr;
        slot.context = nullptr;
        slot.release = nullptr;
        bridge->descriptors_live--;
        if (release) bridge->descriptor_callbacks++;
        if (slot.generation == UINT32_MAX) {
            slot.state = DESCRIPTOR_RETIRED;
            bridge->descriptors_retired++;
        } else {
            slot.state = DESCRIPTOR_FREE;
            bridge->descriptor_free[bridge->descriptor_free_count++] = descriptor.slot;
        }
    }
    return 0;
}

int lfm_kernel_bridge_descriptor_snapshot(
    LfmKernelBridge *bridge, LfmKernelDescriptorSnapshotV1 *out) {
    if (!bridge || !out || out->size < sizeof(*out) ||
        out->abi_version != KC_COORD_ABI_VERSION) {
        return -EINVAL;
    }
    std::lock_guard<std::mutex> guard(bridge->descriptor_mutex);
    *out = {
        .size = sizeof(*out),
        .abi_version = KC_COORD_ABI_VERSION,
        .capacity = bridge->descriptor_capacity,
        .live = bridge->descriptors_live,
        .acquired = bridge->descriptor_acquires,
        .retained = bridge->descriptor_retains,
        .released = bridge->descriptor_releases,
        .callbacks = bridge->descriptor_callbacks,
        .max_generation = bridge->descriptor_max_generation,
        .retired = bridge->descriptors_retired,
    };
    return 0;
}

int lfm_kernel_bridge_submit(LfmKernelBridge *bridge,
                             const KcSubmissionV1 *submission) {
    if (!bridge || !submission_valid(submission)) return -EINVAL;
    if (!enter_submission(bridge)) return -ECANCELED;

    uint64_t tail = bridge->submission_tail.value.load(std::memory_order_relaxed);
    uint64_t completed = bridge->completion_head.value.load(std::memory_order_acquire);
    if (tail - completed >= bridge->capacity) {
        leave_submission(bridge);
        return -EAGAIN;
    }
    uint64_t head = bridge->submission_head.value.load(std::memory_order_acquire);
    if (tail - head >= bridge->capacity) {
        leave_submission(bridge);
        return -EAGAIN;
    }

    if (descriptor_required(submission->command)) {
        int rc = retain_descriptor(bridge, submission->descriptor);
        if (rc != 0) {
            leave_submission(bridge);
            return rc;
        }
    }

    size_t index = tail % bridge->capacity;
    bridge->ledger[index] = submission->ticket;
    bridge->descriptor_ledger[index] = descriptor_required(submission->command)
                                           ? submission->descriptor
                                           : descriptor_none();
    bridge->submissions[index] = *submission;
    bridge->submission_tail.value.store(tail + 1, std::memory_order_release);
    ring(&bridge->submission_doorbell, false);
    leave_submission(bridge);
    return 0;
}

int lfm_kernel_bridge_wait_submission(LfmKernelBridge *bridge, KcSubmissionV1 *out,
                                      uint64_t deadline_ns) {
    if (!bridge || !out) return -EINVAL;
    return wait_for_edge(
        bridge, &bridge->submission_doorbell, deadline_ns,
        [&] { return take_submission(bridge, out); },
        [&] {
            return stopping(bridge) && submissions_settled(bridge) &&
                   bridge->submission_head.value.load(std::memory_order_acquire) ==
                       bridge->submission_tail.value.load(std::memory_order_acquire);
        });
}

int lfm_kernel_bridge_publish_completion(LfmKernelBridge *bridge,
                                         const KcCompletionV1 *completion) {
    if (!bridge || !completion_valid(completion)) return -EINVAL;
    uint64_t tail = bridge->completion_tail.value.load(std::memory_order_relaxed);
    uint64_t dispatched = bridge->submission_head.value.load(std::memory_order_acquire);
    if (tail == dispatched) return -EAGAIN;
    uint64_t head = bridge->completion_head.value.load(std::memory_order_acquire);
    if (tail - head >= bridge->capacity) return -EOVERFLOW;

    size_t index = tail % bridge->capacity;
    if (!ticket_equal(bridge->ledger[index], completion->ticket)) return -ESTALE;
    bridge->completions[index] = *completion;
    bridge->completion_tail.value.store(tail + 1, std::memory_order_release);
    ring(&bridge->completion_doorbell, false);
    return 0;
}

int lfm_kernel_bridge_wait_completion(LfmKernelBridge *bridge, KcCompletionV1 *out,
                                      uint64_t deadline_ns) {
    if (!bridge || !out) return -EINVAL;
    return wait_for_edge(
        bridge, &bridge->completion_doorbell, deadline_ns,
        [&] { return take_completion(bridge, out); },
        [&] {
            return stopping(bridge) && submissions_settled(bridge) &&
                   bridge->completion_head.value.load(std::memory_order_acquire) ==
                       bridge->submission_tail.value.load(std::memory_order_acquire);
        });
}

void lfm_kernel_bridge_request_stop(LfmKernelBridge *bridge) {
    if (!bridge) return;
    uint32_t previous = bridge->admission.fetch_or(ADMISSION_STOP, std::memory_order_acq_rel);
    if (previous & ADMISSION_STOP) return;
    ring(&bridge->submission_doorbell, true);
    ring(&bridge->completion_doorbell, true);
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
        .active_waits = bridge->active_waits.load(std::memory_order_acquire),
        .reserved = 0,
    };
    return 0;
}

int lfm_kernel_bridge_destroy(LfmKernelBridge *bridge) {
    if (!bridge) return -EINVAL;
    if (!stopping(bridge) || !submissions_settled(bridge) ||
        bridge->active_waits.load(std::memory_order_acquire) != 0 ||
        bridge->submission_head.value.load(std::memory_order_acquire) !=
            bridge->submission_tail.value.load(std::memory_order_acquire) ||
        bridge->completion_head.value.load(std::memory_order_acquire) !=
            bridge->completion_tail.value.load(std::memory_order_acquire) ||
        bridge->completion_head.value.load(std::memory_order_acquire) !=
            bridge->submission_tail.value.load(std::memory_order_acquire) ||
        !descriptors_settled(bridge)) {
        return -EBUSY;
    }
    kc_port_wait_u32_release(bridge->completion_doorbell.wait);
    kc_port_wait_u32_release(bridge->submission_doorbell.wait);
    delete[] bridge->descriptor_free;
    delete[] bridge->descriptors;
    delete[] bridge->descriptor_ledger;
    delete[] bridge->ledger;
    delete[] bridge->completions;
    delete[] bridge->submissions;
    delete bridge;
    return 0;
}

} // extern "C"
