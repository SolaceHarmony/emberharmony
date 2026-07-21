// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kcoro_stackless.h"

#include <stdatomic.h>
#include <stdlib.h>

/* The logical stack frame is retained independently of physical workers.  No
 * field below points at a worker stack or a parked operation. */
struct koro_cont {
    uint32_t state;
    koro_step_fn next_step;
    void *frame;
    size_t frame_size;
    void *user_arg;
    struct kc_runtime *runtime;
    atomic_int run_state;
    atomic_uint refs;
    atomic_uint current_worker;
    uint64_t worker_mask;
    kc_ticket_id identity;
    uint32_t slot;
    uint32_t suspend_kind;
    int completed;
    int tracked;
    atomic_uint registered;
    koro_completion_fn completion;
    void *completion_context;
    /* Internal post-DONE edge. Its context must carry an explicit lifetime
     * gate that cannot open until this callback returns. */
    koro_completion_fn settled;
    void *settled_context;
};

/* The callback deposits KORO_WAKE_BIT with one fetch-or and returns.  Only a
 * runtime worker arbitrates base-state transitions with compare/exchange.
 * Internal states and the wake bit must never be published as API states. */
enum {
    KORO_SUSPENDING = KORO_DONE + 1,
    KORO_COMPLETING,
    KORO_STATE_MASK = 0xff,
    KORO_WAKE_BIT = 0x100,
};

static inline int koro_run_base(int state)
{
    return state & KORO_STATE_MASK;
}

static inline int koro_run_has_wake(int state)
{
    return (state & KORO_WAKE_BIT) != 0;
}

static inline uint32_t koro_run_public(int state)
{
    switch (koro_run_base(state)) {
    case KORO_NEW:
    case KORO_QUEUED:
    case KORO_RUNNING:
    case KORO_SUSPENDED:
    case KORO_DONE:
        return (uint32_t)koro_run_base(state);
    case KORO_SUSPENDING:
    case KORO_COMPLETING:
        return KORO_RUNNING;
    default:
        abort();
    }
}

void koro_cont_retain_internal(koro_cont_t *continuation);
void koro_cont_release_internal(koro_cont_t *continuation);
int koro_cont_resume_internal(koro_cont_t *continuation);
int koro_cont_start_internal(koro_cont_t *continuation);

static inline void *koro_cont_step(koro_cont_t *continuation)
{
    if (continuation->completed || !continuation->next_step) return (void *)1;
    return continuation->next_step(continuation);
}
