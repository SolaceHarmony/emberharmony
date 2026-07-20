// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

struct kc_runtime;
struct koro_cont;

typedef void *(*koro_step_fn)(struct koro_cont *continuation);

typedef enum koro_run_state {
    KORO_NEW = 0,
    KORO_QUEUED,
    KORO_RUNNING,
    KORO_DORMANT,
    KORO_DONE,
} koro_run_state;

/* Private retained continuation record. Numerical programs store their state
 * in ticket/route/conversation records; this object only owns service dispatch
 * identity and queue linkage. It is not a public coroutine programming API. */
typedef struct koro_cont {
    koro_step_fn next_step;
    void *user_arg;
    struct kc_runtime *runtime;
    atomic_int run_state;
    atomic_uint refs;
    uint32_t owner_worker;
    uint32_t owner_slot;
    int completed;
    int tracked;
} koro_cont_t;

koro_cont_t *koro_cont_create_on(struct kc_runtime *runtime,
                                 koro_step_fn step, void *arg,
                                 size_t reserved_size);
void koro_cont_retain_internal(koro_cont_t *continuation);
void koro_cont_release_internal(koro_cont_t *continuation);

static inline void *koro_cont_step(koro_cont_t *continuation)
{
    if (continuation->completed || !continuation->next_step) return (void *)1;
    return continuation->next_step(continuation);
}
