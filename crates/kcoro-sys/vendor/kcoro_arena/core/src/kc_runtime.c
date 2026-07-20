// SPDX-License-Identifier: BSD-3-Clause
#include "kc_runtime_internal.h"
#include "kc_service_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdlib.h>

static _Thread_local kc_runtime_t *current_worker_runtime;
static _Thread_local koro_cont_t *current_worker_continuation;

static void destroy_workers(kc_runtime_t *runtime)
{
    if (!runtime || !runtime->workers) return;
    for (unsigned index = 0; index < runtime->worker_count; ++index)
        kc_doorbell_destroy(runtime->workers[index].idle_doorbell);
    free(runtime->workers);
    runtime->workers = NULL;
}

static void runtime_free(kc_runtime_t *runtime)
{
    if (!runtime) return;
    destroy_workers(runtime);
    kc_doorbell_destroy(runtime->lifecycle_doorbell);
    KC_MUTEX_DESTROY(&runtime->mu);
    free(runtime);
}

void kc_runtime_retain_internal(kc_runtime_t *runtime)
{
    if (runtime)
        atomic_fetch_add_explicit(&runtime->refs, 1, memory_order_relaxed);
}

void kc_runtime_release_internal(kc_runtime_t *runtime)
{
    if (!runtime) return;
    if (atomic_fetch_sub_explicit(&runtime->refs, 1, memory_order_acq_rel) == 1)
        runtime_free(runtime);
}

static int create_workers(kc_runtime_t *runtime)
{
    runtime->workers = calloc(runtime->worker_count, sizeof(*runtime->workers));
    if (!runtime->workers) return -ENOMEM;
    for (unsigned index = 0; index < runtime->worker_count; ++index) {
        kc_runtime_worker *worker = &runtime->workers[index];
        worker->runtime = runtime;
        worker->index = index;
        atomic_init(&worker->ready_services, 0);
        for (unsigned slot = 0; slot < KC_RUNTIME_SERVICES_PER_WORKER; ++slot)
            atomic_init(&worker->services[slot], NULL);
        int status = kc_doorbell_create(&worker->idle_doorbell);
        if (status != 0) {
            runtime->worker_count = index;
            destroy_workers(runtime);
            return status;
        }
    }
    return 0;
}

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out)
{
    if (!out) return -EINVAL;
    if (config && (config->size < sizeof(*config) ||
                   config->abi_version != KC_ABI_VERSION ||
                   config->reserved != 0)) return -EINVAL;
    kc_runtime_t *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return -ENOMEM;
    atomic_init(&runtime->refs, 1);
    atomic_init(&runtime->wake_requests, 0);
    atomic_init(&runtime->resumes, 0);
    atomic_init(&runtime->queued, 0);
    atomic_init(&runtime->running, 0);
    atomic_init(&runtime->active, 0);
    atomic_init(&runtime->lifecycle_waiters, 0);
    atomic_init(&runtime->progress, 1);
    atomic_init(&runtime->worker_stop, 0);
    runtime->worker_count = config && config->worker_count
        ? config->worker_count : 1;
    if (KC_MUTEX_INIT(&runtime->mu) != 0) {
        free(runtime);
        return -ENOMEM;
    }
    int status = kc_doorbell_create(&runtime->lifecycle_doorbell);
    if (status != 0) {
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return status;
    }
    status = create_workers(runtime);
    if (status != 0) {
        kc_doorbell_destroy(runtime->lifecycle_doorbell);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return status;
    }
    runtime->accepting = 1;
    *out = runtime;
    return 0;
}

void kc_runtime_signal_lifecycle_internal(kc_runtime_t *runtime)
{
    if (!runtime) return;
    atomic_fetch_add_explicit(&runtime->progress, 1, memory_order_release);
    if (atomic_load_explicit(&runtime->lifecycle_waiters,
                             memory_order_seq_cst) != 0)
        kc_doorbell_ring_all(runtime->lifecycle_doorbell);
}

void kc_runtime_ring_workers_internal(kc_runtime_t *runtime)
{
    if (!runtime || !runtime->workers) return;
    for (unsigned index = 0; index < runtime->worker_count; ++index)
        kc_doorbell_ring_all(runtime->workers[index].idle_doorbell);
}

int kc_runtime_work_realtime_safe_internal(const kc_runtime_t *runtime)
{
    if (!runtime || !runtime->workers) return 0;
    for (unsigned index = 0; index < runtime->worker_count; ++index) {
        if (!kc_doorbell_realtime_safe(runtime->workers[index].idle_doorbell))
            return 0;
    }
    return 1;
}

int kc_runtime_is_current_worker_internal(const kc_runtime_t *runtime)
{
    return runtime && current_worker_runtime == runtime;
}

int kc_runtime_is_current_cont_internal(const koro_cont_t *continuation)
{
    return continuation && current_worker_continuation == continuation;
}

int kc_runtime_bind_service_locked_internal(kc_runtime_t *runtime,
                                            struct kc_service *service,
                                            koro_cont_t *continuation)
{
    if (!runtime || !service || !continuation || !runtime->accepting)
        return -ECANCELED;
    for (unsigned offset = 0; offset < runtime->worker_count; ++offset) {
        unsigned owner = (runtime->next_service_owner + offset) %
                         runtime->worker_count;
        kc_runtime_worker *worker = &runtime->workers[owner];
        if (worker->service_slots == UINT64_MAX) continue;
        unsigned slot = 0;
        while (slot < KC_RUNTIME_SERVICES_PER_WORKER &&
               (worker->service_slots & (UINT64_C(1) << slot))) ++slot;
        if (slot == KC_RUNTIME_SERVICES_PER_WORKER) continue;
        worker->service_slots |= UINT64_C(1) << slot;
        continuation->owner_worker = owner;
        continuation->owner_slot = slot;
        atomic_store_explicit(&worker->services[slot], service,
                              memory_order_release);
        runtime->next_service_owner = (owner + 1) % runtime->worker_count;
        return 0;
    }
    return -ENOSPC;
}

void kc_runtime_unbind_service_locked_internal(kc_runtime_t *runtime,
                                               struct kc_service *service,
                                               const koro_cont_t *continuation)
{
    if (!runtime || !service || !continuation ||
        continuation->owner_worker >= runtime->worker_count ||
        continuation->owner_slot >= KC_RUNTIME_SERVICES_PER_WORKER) return;
    kc_runtime_worker *worker = &runtime->workers[continuation->owner_worker];
    struct kc_service *current = atomic_load_explicit(
        &worker->services[continuation->owner_slot], memory_order_acquire);
    if (current != service) return;
    atomic_store_explicit(&worker->services[continuation->owner_slot], NULL,
                          memory_order_release);
    worker->service_slots &= ~(UINT64_C(1) << continuation->owner_slot);
}

void kc_runtime_publish_service_internal(kc_runtime_t *runtime,
                                         const koro_cont_t *continuation)
{
    if (!runtime || !continuation ||
        continuation->owner_worker >= runtime->worker_count ||
        continuation->owner_slot >= KC_RUNTIME_SERVICES_PER_WORKER) return;
    kc_runtime_worker *worker = &runtime->workers[continuation->owner_worker];
    const uint64_t bit = UINT64_C(1) << continuation->owner_slot;
    uint64_t prior = atomic_fetch_or_explicit(&worker->ready_services, bit,
                                              memory_order_acq_rel);
    if ((prior & bit) != 0) return;
    atomic_fetch_add_explicit(&runtime->wake_requests, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&runtime->queued, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&runtime->progress, 1, memory_order_release);
    kc_doorbell_ring_one(worker->idle_doorbell);
}

void kc_runtime_retire_service_internal(kc_runtime_t *runtime,
                                        const koro_cont_t *continuation)
{
    if (!runtime || !continuation ||
        continuation->owner_worker >= runtime->worker_count ||
        continuation->owner_slot >= KC_RUNTIME_SERVICES_PER_WORKER) return;
    kc_runtime_worker *worker = &runtime->workers[continuation->owner_worker];
    const uint64_t bit = UINT64_C(1) << continuation->owner_slot;
    uint64_t prior = atomic_fetch_and_explicit(&worker->ready_services, ~bit,
                                               memory_order_acq_rel);
    if ((prior & bit) != 0)
        atomic_fetch_sub_explicit(&runtime->queued, 1, memory_order_relaxed);
}

static void run_services(kc_runtime_worker *worker, uint64_t ready)
{
    for (unsigned slot = 0; slot < KC_RUNTIME_SERVICES_PER_WORKER; ++slot) {
        if ((ready & (UINT64_C(1) << slot)) == 0) continue;
        atomic_fetch_sub_explicit(&worker->runtime->queued, 1,
                                  memory_order_relaxed);
        struct kc_service *service = atomic_load_explicit(
            &worker->services[slot], memory_order_acquire);
        if (!service) continue;
        koro_cont_t *continuation = kc_service_continuation_internal(service);
        if (!continuation || continuation->owner_worker != worker->index ||
            continuation->owner_slot != slot) abort();
        current_worker_continuation = continuation;
        kc_service_runtime_execute_internal(service);
        current_worker_continuation = NULL;
    }
}

static void *worker_main(void *arg)
{
    kc_runtime_worker *worker = arg;
    kc_runtime_t *runtime = worker->runtime;
    current_worker_runtime = runtime;
    for (;;) {
        /* This is the resident dispatch machine, not a readiness poll. It
         * consumes one bounded inbound bitmap, then becomes dormant on its
         * private idle doorbell when no ticket or service edge exists. */
        uint32_t observed = kc_doorbell_observe(worker->idle_doorbell);
        uint64_t ready = atomic_exchange_explicit(&worker->ready_services, 0,
                                                  memory_order_acq_rel);
        if (ready) {
            run_services(worker, ready);
            continue;
        }
        if (atomic_load_explicit(&runtime->worker_stop,
                                 memory_order_acquire) != 0 &&
            atomic_load_explicit(&worker->ready_services,
                                 memory_order_acquire) == 0) {
            current_worker_runtime = NULL;
            return NULL;
        }
        if (atomic_load_explicit(&worker->ready_services,
                                 memory_order_acquire) != 0) continue;
        if (kc_doorbell_park(worker->idle_doorbell, observed) != 0) abort();
    }
}

int kc_runtime_start(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->starting || runtime->joining) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (runtime->started) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    runtime->starting = 1;
    atomic_store_explicit(&runtime->worker_stop, 0, memory_order_release);
    runtime->joined = 0;
    KC_MUTEX_UNLOCK(&runtime->mu);

    unsigned started = 0;
    for (; started < runtime->worker_count; ++started) {
        kc_runtime_worker *worker = &runtime->workers[started];
        if (kc_port_thread_create(&worker->thread, worker_main, worker) != 0)
            break;
    }
    if (started != runtime->worker_count) {
        atomic_store_explicit(&runtime->worker_stop, 1, memory_order_release);
        kc_runtime_ring_workers_internal(runtime);
        for (unsigned index = 0; index < started; ++index)
            kc_port_thread_join(runtime->workers[index].thread);
        KC_MUTEX_LOCK(&runtime->mu);
        runtime->starting = 0;
        runtime->joined = 1;
        KC_MUTEX_UNLOCK(&runtime->mu);
        kc_runtime_signal_lifecycle_internal(runtime);
        return -EAGAIN;
    }
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->starting = 0;
    runtime->started = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_join_all(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    if (kc_runtime_is_current_worker_internal(runtime)) return -EDEADLK;
    int status = kc_runtime_start(runtime);
    if (status != 0) return status;
    for (;;) {
        uint32_t observed = kc_doorbell_observe(runtime->lifecycle_doorbell);
        uint64_t progress = atomic_load_explicit(&runtime->progress,
                                                 memory_order_acquire);
        KC_MUTEX_LOCK(&runtime->mu);
        int done = atomic_load_explicit(&runtime->active,
                                        memory_order_acquire) == 0;
        KC_MUTEX_UNLOCK(&runtime->mu);
        if (done && atomic_load_explicit(&runtime->progress,
                                         memory_order_acquire) == progress)
            return 0;
        atomic_fetch_add_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
        if (atomic_load_explicit(&runtime->progress, memory_order_acquire) ==
            progress) {
            if (kc_doorbell_park(runtime->lifecycle_doorbell, observed) != 0)
                abort();
        }
        atomic_fetch_sub_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
    }
}

void kc_runtime_request_stop(kc_runtime_t *runtime)
{
    if (!runtime) return;
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->accepting = 0;
    runtime->stop_requested = 1;
    kc_service_runtime_stop_locked(runtime);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_ring_workers_internal(runtime);
    kc_runtime_signal_lifecycle_internal(runtime);
}

int kc_runtime_join(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    if (kc_runtime_is_current_worker_internal(runtime)) return -EDEADLK;
    KC_MUTEX_LOCK(&runtime->mu);
    int busy = runtime->starting || runtime->joining ||
               atomic_load_explicit(&runtime->active, memory_order_acquire) != 0 ||
               atomic_load_explicit(&runtime->queued, memory_order_acquire) != 0 ||
               atomic_load_explicit(&runtime->running, memory_order_acquire) != 0;
    if (busy) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (!runtime->started || runtime->joined) {
        runtime->joined = 1;
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    runtime->joining = 1;
    atomic_store_explicit(&runtime->worker_stop, 1, memory_order_release);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_ring_workers_internal(runtime);
    for (unsigned index = 0; index < runtime->worker_count; ++index)
        kc_port_thread_join(runtime->workers[index].thread);
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->started = 0;
    runtime->joining = 0;
    runtime->joined = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_destroy(kc_runtime_t *runtime)
{
    if (!runtime) return 0;
    KC_MUTEX_LOCK(&runtime->mu);
    int busy = runtime->starting || runtime->joining ||
               atomic_load_explicit(&runtime->active, memory_order_acquire) != 0 ||
               atomic_load_explicit(&runtime->queued, memory_order_acquire) != 0 ||
               atomic_load_explicit(&runtime->running, memory_order_acquire) != 0 ||
               runtime->live_services != 0 ||
               (runtime->started && !runtime->joined);
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (busy) return -EBUSY;
    kc_runtime_release_internal(runtime);
    return 0;
}

int kc_runtime_snapshot_get(kc_runtime_t *runtime, kc_runtime_snapshot *out)
{
    if (!runtime || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    const size_t active = atomic_load_explicit(&runtime->active,
                                               memory_order_acquire);
    const size_t queued = atomic_load_explicit(&runtime->queued,
                                               memory_order_acquire);
    const size_t running = atomic_load_explicit(&runtime->running,
                                                memory_order_acquire);
    *out = (kc_runtime_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .active = active, .queued = queued, .running = running,
        .dormant = active > queued + running ? active - queued - running : 0,
        .workers = runtime->worker_count,
        .accepting = (unsigned)runtime->accepting,
        .started = (unsigned)runtime->started,
        .stop_requested = (unsigned)runtime->stop_requested,
        .wake_requests = atomic_load_explicit(&runtime->wake_requests,
                                              memory_order_relaxed),
        .resumes = atomic_load_explicit(&runtime->resumes, memory_order_relaxed),
    };
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}
