// SPDX-License-Identifier: BSD-3-Clause
#include "koro_internal.h"
#include "kc_runtime_internal.h"

#include <stdlib.h>

koro_cont_t *koro_cont_create_on(kc_runtime_t *runtime, koro_step_fn step,
                                 void *arg, size_t reserved_size)
{
    if (!runtime || !step || reserved_size != 0) return NULL;
    koro_cont_t *continuation = calloc(1, sizeof(*continuation));
    if (!continuation) return NULL;
    continuation->next_step = step;
    continuation->user_arg = arg;
    continuation->runtime = runtime;
    kc_runtime_retain_internal(runtime);
    atomic_init(&continuation->run_state, KORO_NEW);
    atomic_init(&continuation->refs, 1);
    continuation->owner_worker = UINT32_MAX;
    continuation->owner_slot = UINT32_MAX;
    return continuation;
}

void koro_cont_retain_internal(koro_cont_t *continuation)
{
    if (continuation) {
        atomic_fetch_add_explicit(&continuation->refs, 1,
                                  memory_order_relaxed);
    }
}

void koro_cont_release_internal(koro_cont_t *continuation)
{
    if (!continuation ||
        atomic_fetch_sub_explicit(&continuation->refs, 1,
                                  memory_order_acq_rel) != 1) return;
    kc_runtime_release_internal(continuation->runtime);
    free(continuation);
}
