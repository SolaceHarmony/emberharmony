// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kcoro_stackless.h"

#include <stdatomic.h>

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
    atomic_uint wake_pending;
    atomic_uint refs;
    atomic_uint current_worker;
    uint64_t worker_mask;
    kc_ticket_id identity;
    uint32_t slot;
    uint32_t suspend_kind;
    int completed;
    int tracked;
    int registered;
    koro_completion_fn completion;
    void *completion_context;
};

void koro_cont_retain_internal(koro_cont_t *continuation);
void koro_cont_release_internal(koro_cont_t *continuation);
int koro_cont_resume_internal(koro_cont_t *continuation);
int koro_cont_start_internal(koro_cont_t *continuation);

static inline void *koro_cont_step(koro_cont_t *continuation)
{
    if (continuation->completed || !continuation->next_step) return (void *)1;
    return continuation->next_step(continuation);
}
