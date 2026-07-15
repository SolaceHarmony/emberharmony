// SPDX-License-Identifier: BSD-3-Clause
#include "kc_timer_internal.h"
#include "kc_runtime_internal.h"

#include <errno.h>
#include <stdlib.h>

int kc_timer_create(kc_runtime_t *runtime, const kc_timer_config *config,
                    kc_timer_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->deadline_ns) return -EINVAL;
    kc_timer_t *timer = calloc(1, sizeof(*timer));
    if (!timer) return -ENOMEM;
    atomic_init(&timer->refs, 1);
    timer->runtime = runtime;
    timer->deadline_ns = config->deadline_ns;
    timer->id = (kc_id){ runtime->epoch, kc_runtime_next_sequence(runtime) };
    kc_runtime_retain_internal(runtime);
    int rc = kc_cancel_create(&timer->cancel, config->parent_cancel);
    if (rc != 0) {
        kc_runtime_release_internal(runtime);
        free(timer);
        return rc;
    }
    *out = timer;
    return 0;
}

void kc_timer_retain(kc_timer_t *timer)
{
    if (timer) atomic_fetch_add_explicit(&timer->refs, 1, memory_order_relaxed);
}

void kc_timer_cancel(kc_timer_t *timer)
{
    if (timer) kc_cancel_trigger(timer->cancel);
}

void kc_timer_release(kc_timer_t *timer)
{
    if (!timer) return;
    if (atomic_fetch_sub_explicit(&timer->refs, 1, memory_order_acq_rel) != 1) return;
    kc_cancel_release(timer->cancel);
    kc_runtime_release_internal(timer->runtime);
    free(timer);
}

int kc_timer_snapshot_get(const kc_timer_t *timer, kc_timer_snapshot *out)
{
    if (!timer || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_timer_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = timer->id, .deadline_ns = timer->deadline_ns,
        .canceled = (unsigned)kc_cancel_is_triggered(timer->cancel),
    };
    return 0;
}
