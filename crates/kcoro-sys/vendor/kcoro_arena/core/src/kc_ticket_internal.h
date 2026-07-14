// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_ticket.h"

#include <stdatomic.h>

struct kc_runtime;

struct kc_ticket {
    atomic_uint refs;
    struct kc_runtime *runtime;
    uint32_t slot;
    uint32_t generation;
    uint32_t free_next;
    uint32_t in_use;
    kc_ticket_state state;
    uint32_t cancel_requested;
    kc_ticket_terminal_cause cancel_cause;
    uint32_t target_consumed;
    uint32_t delivery_queued;
    kc_ticket_deadline_mode deadline_mode;
    uint64_t deadline_ns;
    kc_descriptor_t *descriptor;
    kc_descriptor_t *result;
    kc_ticket_callback_fn callback;
    void *callback_context;
    kc_ticket_context_release_fn context_release;
    kc_ticket_event_v1 event;
    struct kc_ticket *completion_next;
};

int kc_ticket_runtime_init(struct kc_runtime *runtime, uint32_t capacity);
void kc_ticket_runtime_destroy(struct kc_runtime *runtime);
kc_ticket_t *kc_ticket_runtime_dequeue_locked(struct kc_runtime *runtime);
void kc_ticket_runtime_deliver(kc_ticket_t *ticket);
void kc_ticket_runtime_stop_locked(struct kc_runtime *runtime);
