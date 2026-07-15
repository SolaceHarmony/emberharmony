// SPDX-License-Identifier: BSD-3-Clause
#include "koro_sched_stackless.h"
#include "kc_runtime_internal.h"

#include <errno.h>

int koro_sched_init(void)
{
    return kc_runtime_default_get() ? 0 : -ENOMEM;
}

int koro_sched_start_workers(int n)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    if (!runtime) return -ENOMEM;
    if (n > 0 && !runtime->started) runtime->worker_count = (unsigned)n;
    int rc = kc_runtime_start(runtime);
    return rc == 0 ? (int)runtime->worker_count : rc;
}

int koro_sched_worker_count(void)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    return runtime && runtime->started ? (int)runtime->worker_count : 0;
}

void koro_sched_enqueue_ready(koro_cont_t *cont)
{
    if (!cont) return;
    kc_runtime_t *runtime = cont->runtime ? cont->runtime : kc_runtime_default_get();
    if (!runtime) return;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!cont->tracked) { cont->tracked = 1; runtime->active++; }
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (kc_runtime_enqueue_internal(runtime, cont, KORO_NEW)) return;
    kc_runtime_wake_internal(cont);
}

void koro_sched_wake(koro_cont_t *cont) { kc_runtime_wake_internal(cont); }

int koro_run(void)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    return runtime ? kc_runtime_join_all(runtime) : -ENOMEM;
}

void koro_stop(void) { kc_runtime_legacy_break(kc_runtime_default_get()); }

void koro_sched_shutdown(void)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    if (!runtime) return;
    kc_runtime_request_stop(runtime);
    if (kc_runtime_join(runtime) == 0) (void)kc_runtime_destroy(runtime);
}

int koro_go(void *(*func)(koro_cont_t *), void *arg, size_t size)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    return runtime ? kc_runtime_spawn(runtime, func, arg, size) : -ENOMEM;
}
