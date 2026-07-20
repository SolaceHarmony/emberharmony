// SPDX-License-Identifier: BSD-3-Clause
#include "kc_service.h"

#include "kc_runtime_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

enum kc_service_phase {
    KC_SERVICE_CREATED = 0,
    KC_SERVICE_STARTED,
    KC_SERVICE_RETIRING,
    KC_SERVICE_JOINED,
};

struct kc_service {
    kc_runtime_t *runtime;
    koro_cont_t *continuation;
    kc_service_fn callback;
    kc_service_fn owner_init;
    kc_service_fn owner_fini;
    void *context;
    struct kc_service *registry_prev;
    struct kc_service *registry_next;
    atomic_uint_fast64_t notifications;
    atomic_uint_fast64_t handled_notifications;
    atomic_uint_fast64_t callbacks;
    atomic_uint phase;
    atomic_uint ever_started;
    atomic_uint stop_requested;
    /* Realtime admission is a bounded two-observation lease, never a CAS
     * retry: closed is checked before and after incrementing publishers. Stop
     * closes first, then retirement waits for already-admitted publishers. */
    atomic_uint realtime_closed;
    atomic_uint realtime_publishers;
    size_t realtime_notifiers;
    int realtime_capable;
    /* Touched only by the permanent owner worker. They deliberately are not
     * atomics: service affinity, rather than shared synchronization, is the
     * ownership proof. */
    int owner_initialized;
    int owner_finalized;
};

struct kc_service_notifier {
    kc_service_t *service;
};

static int notify_realtime(kc_service_t *service);

static void *service_step(koro_cont_t *continuation)
{
    kc_service_t *service = continuation->user_arg;
    if (!service->owner_initialized) {
        service->owner_initialized = 1;
        if (service->owner_init) service->owner_init(service->context);
    }
    uint64_t handled = atomic_load_explicit(
        &service->handled_notifications, memory_order_relaxed);
    uint64_t notified = atomic_load_explicit(
        &service->notifications, memory_order_acquire);

    if (notified != handled) {
        service->callback(service->context);
        atomic_fetch_add_explicit(&service->callbacks, 1, memory_order_relaxed);
        atomic_store_explicit(&service->handled_notifications, notified,
                              memory_order_release);
        handled = notified;
    }

    /*
     * Stop closes admission before publishing RETIRING. A successful notify
     * therefore remains visible here. Do not retire until every accepted edge
     * has received its drain callback.
     */
    if (atomic_load_explicit(&service->phase, memory_order_acquire) ==
            KC_SERVICE_RETIRING &&
        atomic_load_explicit(&service->realtime_publishers,
                             memory_order_seq_cst) == 0 &&
        atomic_load_explicit(&service->notifications, memory_order_acquire) ==
            handled) {
        if (!service->owner_finalized) {
            service->owner_finalized = 1;
            if (service->owner_fini) service->owner_fini(service->context);
        }
        continuation->completed = 1;
        return (void *)1;
    }
    return NULL;
}

static void stop_service(kc_service_t *service)
{
    if (!service) return;
    atomic_store_explicit(&service->realtime_closed, 1,
                          memory_order_seq_cst);
    unsigned phase = atomic_load_explicit(&service->phase,
                                          memory_order_acquire);
    if (phase != KC_SERVICE_CREATED && phase != KC_SERVICE_STARTED) return;
    int won = atomic_compare_exchange_strong_explicit(
        &service->phase, &phase, KC_SERVICE_RETIRING,
        memory_order_acq_rel, memory_order_acquire);
    if (!won && phase == KC_SERVICE_STARTED) {
        won = atomic_compare_exchange_strong_explicit(
            &service->phase, &phase, KC_SERVICE_RETIRING,
            memory_order_acq_rel, memory_order_acquire);
    }
    if (!won) return;
    atomic_store_explicit(&service->stop_requested, 1, memory_order_release);
    if (atomic_load_explicit(&service->ever_started, memory_order_acquire))
        kc_runtime_publish_service_internal(service->runtime,
                                            service->continuation);
}

int kc_service_create(kc_runtime_t *runtime, const kc_service_config *config,
                      kc_service_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->callback ||
        config->reserved != 0) return -EINVAL;

    kc_service_t *service = calloc(1, sizeof(*service));
    if (!service) return -ENOMEM;
    service->runtime = runtime;
    service->callback = config->callback;
    service->owner_init = config->owner_init;
    service->owner_fini = config->owner_fini;
    service->context = config->context;
    atomic_init(&service->notifications, 0);
    atomic_init(&service->handled_notifications, 0);
    atomic_init(&service->callbacks, 0);
    atomic_init(&service->phase, KC_SERVICE_CREATED);
    atomic_init(&service->ever_started, 0);
    atomic_init(&service->stop_requested, 0);
    atomic_init(&service->realtime_closed, 1);
    atomic_init(&service->realtime_publishers, 0);
    service->realtime_capable =
        atomic_is_lock_free(&service->notifications) &&
        atomic_is_lock_free(&service->realtime_closed) &&
        atomic_is_lock_free(&service->realtime_publishers) &&
        atomic_is_lock_free(&runtime->wake_requests) &&
        kc_runtime_work_realtime_safe_internal(runtime);
    service->continuation = koro_cont_create_on(runtime, service_step, service, 0);
    if (!service->continuation) {
        free(service);
        return -ENOMEM;
    }

    KC_MUTEX_LOCK(&runtime->mu);
    if (!runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        koro_cont_release_internal(service->continuation);
        free(service);
        return -ECANCELED;
    }
    int bind_status = kc_runtime_bind_service_locked_internal(
        runtime, service, service->continuation);
    if (bind_status != 0) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        koro_cont_release_internal(service->continuation);
        free(service);
        return bind_status;
    }
    service->registry_next = runtime->services_head;
    if (runtime->services_head)
        runtime->services_head->registry_prev = service;
    runtime->services_head = service;
    runtime->live_services++;
    KC_MUTEX_UNLOCK(&runtime->mu);

    *out = service;
    return 0;
}

int kc_service_start(kc_service_t *service)
{
    if (!service) return -EINVAL;
    kc_runtime_t *runtime = service->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    unsigned phase = atomic_load_explicit(&service->phase, memory_order_acquire);
    if (phase == KC_SERVICE_STARTED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    if (phase != KC_SERVICE_CREATED || !runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }

    unsigned expected = KC_SERVICE_CREATED;
    if (!atomic_compare_exchange_strong_explicit(
            &service->phase, &expected, KC_SERVICE_STARTED,
            memory_order_acq_rel, memory_order_acquire)) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    service->continuation->tracked = 1;
    atomic_store_explicit(&service->continuation->run_state, KORO_QUEUED,
                          memory_order_release);
    atomic_fetch_add_explicit(&runtime->active, 1, memory_order_relaxed);
    atomic_store_explicit(&service->ever_started, 1, memory_order_release);
    atomic_store_explicit(&service->realtime_closed, 0,
                          memory_order_seq_cst);
    if (atomic_load_explicit(&service->phase, memory_order_acquire) !=
        KC_SERVICE_STARTED)
        atomic_store_explicit(&service->realtime_closed, 1,
                              memory_order_seq_cst);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_publish_service_internal(runtime, service->continuation);
    return 0;
}

int kc_service_notify(kc_service_t *service)
{
    if (!service) return -EINVAL;
    if (atomic_load_explicit(&service->phase, memory_order_acquire) !=
        KC_SERVICE_STARTED) return -ECANCELED;
    return notify_realtime(service);
}

static void realtime_leave(kc_service_t *service, int published)
{
    (void)published;
    const unsigned prior = atomic_fetch_sub_explicit(
        &service->realtime_publishers, 1, memory_order_seq_cst);
    if (prior == 0) abort();
    /* A retiring owner may consume the producer's first ready edge while this
     * publication lease is still live. In that case it must remain dormant,
     * because the producer may still be installing its notification. The
     * final publisher is therefore the causal successor: once 1 -> 0 makes
     * every admitted publication visible, it republishes the fixed-owner bit
     * so the continuation can observe quiescence and retire. If this release
     * linearizes before close, stop_service publishes the successor instead. */
    if (prior == 1 &&
        atomic_load_explicit(&service->realtime_closed,
                             memory_order_seq_cst)) {
        kc_runtime_publish_service_internal(service->runtime,
                                            service->continuation);
    }
}

static int realtime_enter(kc_service_t *service)
{
    /* These three operations share the sequentially-consistent order with the
     * close-and-inspect path. Without that one order, a closing service and a
     * publisher may legally observe the old value of different atomics and
     * both conclude that the other has not arrived. */
    if (atomic_load_explicit(&service->phase, memory_order_acquire) !=
            KC_SERVICE_STARTED ||
        atomic_load_explicit(&service->realtime_closed,
                             memory_order_seq_cst)) return -ECANCELED;
    atomic_fetch_add_explicit(&service->realtime_publishers, 1,
                              memory_order_seq_cst);
    if (!atomic_load_explicit(&service->realtime_closed,
                              memory_order_seq_cst) &&
        atomic_load_explicit(&service->phase, memory_order_acquire) ==
            KC_SERVICE_STARTED) return 0;
    /* Publish retirement work before dropping the admission lease. Once the
     * owner observes zero publishers, every admitted producer has already
     * installed its fixed-owner bit. */
    kc_runtime_publish_service_internal(service->runtime,
                                        service->continuation);
    realtime_leave(service, 0);
    return -ECANCELED;
}

static int notify_realtime(kc_service_t *service)
{
    if (!service->realtime_capable) return -ENOTSUP;
    int status = realtime_enter(service);
    if (status != 0) return status;

    /* Publish the callback predicate first, then drop this publisher's
     * admission lease. The final doorbell ring makes both releases visible to
     * the worker; if stop raced this publisher, that final edge is what lets a
     * service that previously observed a non-zero count run once more and
     * retire. */
    atomic_fetch_add_explicit(&service->notifications, 1,
                              memory_order_release);
    kc_runtime_publish_service_internal(service->runtime,
                                        service->continuation);
    realtime_leave(service, 1);
    return 0;
}

int kc_service_notifier_create(kc_service_t *service,
                               kc_service_notifier_t **out)
{
    if (!service || !out) return -EINVAL;
    if (!service->realtime_capable) return -ENOTSUP;
    kc_service_notifier_t *notifier = calloc(1, sizeof(*notifier));
    if (!notifier) return -ENOMEM;

    kc_runtime_t *runtime = service->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    unsigned phase = atomic_load_explicit(&service->phase,
                                           memory_order_acquire);
    if (phase == KC_SERVICE_RETIRING || phase == KC_SERVICE_JOINED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        free(notifier);
        return -ECANCELED;
    }
    service->realtime_notifiers++;
    notifier->service = service;
    KC_MUTEX_UNLOCK(&runtime->mu);
    *out = notifier;
    return 0;
}

int kc_service_notifier_notify(kc_service_notifier_t *notifier)
{
    return notifier ? notify_realtime(notifier->service) : -EINVAL;
}

int kc_service_notifier_destroy(kc_service_notifier_t *notifier)
{
    if (!notifier) return 0;
    kc_service_t *service = notifier->service;
    kc_runtime_t *runtime = service->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!service->realtime_notifiers) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EINVAL;
    }
    service->realtime_notifiers--;
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_signal_lifecycle_internal(runtime);
    free(notifier);
    return 0;
}

int kc_service_ready_again(kc_service_t *service)
{
    if (!service) return -EINVAL;
    if (!kc_runtime_is_current_cont_internal(service->continuation))
        return -EPERM;

    /* Use the same bounded admission lease as the realtime producer edge, so a
     * local reschedule linearizes cleanly against stop. Unlike an external
     * notification, it republishes the same fixed-owner slot. The current
     * callback returns before its owner consumes that next edge. */
    int status = realtime_enter(service);
    if (status != 0) return status;

    atomic_fetch_add_explicit(&service->notifications, 1,
                              memory_order_release);
    kc_runtime_publish_service_internal(service->runtime,
                                        service->continuation);
    realtime_leave(service, 1);
    return 0;
}

int kc_service_complete_current(kc_service_t *service)
{
    if (!service) return -EINVAL;
    if (!kc_runtime_is_current_cont_internal(service->continuation))
        return -EPERM;

    /* Natural completion is a continuation state transition, not a runtime
     * stop and not a request for another worker. Close producer admission
     * before publishing RETIRING. A realtime publisher that already owns an
     * admission lease retains its count and notification; its final edge re-enters
     * this continuation so the accepted generation is drained before DONE. */
    atomic_store_explicit(&service->realtime_closed, 1,
                          memory_order_seq_cst);
    unsigned expected = KC_SERVICE_STARTED;
    if (!atomic_compare_exchange_strong_explicit(
            &service->phase, &expected, KC_SERVICE_RETIRING,
            memory_order_release, memory_order_acquire)) {
        return expected == KC_SERVICE_RETIRING ? 0 : -ECANCELED;
    }
    return 0;
}

void kc_service_request_stop(kc_service_t *service)
{
    stop_service(service);
}

int kc_service_join(kc_service_t *service)
{
    if (!service) return -EINVAL;
    kc_runtime_t *runtime = service->runtime;
    if (kc_runtime_is_current_worker_internal(runtime)) return -EDEADLK;
    KC_MUTEX_LOCK(&runtime->mu);
    unsigned phase = atomic_load_explicit(&service->phase, memory_order_acquire);
    if (phase == KC_SERVICE_JOINED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    if (phase == KC_SERVICE_CREATED) {
        atomic_store_explicit(&service->phase, KC_SERVICE_JOINED,
                              memory_order_release);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    if (phase != KC_SERVICE_RETIRING) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (!atomic_load_explicit(&service->ever_started, memory_order_acquire)) {
        atomic_store_explicit(&service->phase, KC_SERVICE_JOINED,
                              memory_order_release);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }

    while (atomic_load_explicit(&service->continuation->run_state,
                                memory_order_acquire) != KORO_DONE) {
        if (!runtime->started) {
            KC_MUTEX_UNLOCK(&runtime->mu);
            return -EBUSY;
        }
        KC_MUTEX_UNLOCK(&runtime->mu);
        uint32_t observed = kc_doorbell_observe(runtime->lifecycle_doorbell);
        uint64_t progress = atomic_load_explicit(&runtime->progress,
                                                 memory_order_acquire);
        atomic_fetch_add_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
        if (atomic_load_explicit(&service->continuation->run_state,
                                 memory_order_acquire) != KORO_DONE &&
            atomic_load_explicit(&runtime->progress, memory_order_acquire) ==
                progress) {
            if (kc_doorbell_park(runtime->lifecycle_doorbell, observed) != 0)
                abort();
        }
        atomic_fetch_sub_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
        KC_MUTEX_LOCK(&runtime->mu);
    }
    atomic_store_explicit(&service->phase, KC_SERVICE_JOINED,
                          memory_order_release);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_service_snapshot_get(kc_service_t *service, kc_service_snapshot *out)
{
    if (!service || !out || out->size < sizeof(*out)) return -EINVAL;
    unsigned phase = atomic_load_explicit(&service->phase, memory_order_acquire);
    *out = (kc_service_snapshot){
        .size = sizeof(*out),
        .abi_version = KC_ABI_VERSION,
        .notifications = atomic_load_explicit(&service->notifications,
                                              memory_order_acquire),
        .handled_notifications = atomic_load_explicit(
            &service->handled_notifications, memory_order_acquire),
        .callbacks = atomic_load_explicit(&service->callbacks,
                                          memory_order_acquire),
        .run_state = (uint32_t)atomic_load_explicit(
            &service->continuation->run_state, memory_order_acquire),
        .started = atomic_load_explicit(&service->ever_started,
                                        memory_order_acquire),
        .stop_requested = atomic_load_explicit(&service->stop_requested,
                                               memory_order_acquire),
        .joined = phase == KC_SERVICE_JOINED,
    };
    return 0;
}

int kc_service_destroy(kc_service_t *service)
{
    if (!service) return 0;
    kc_runtime_t *runtime = service->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    unsigned phase = atomic_load_explicit(&service->phase, memory_order_acquire);
    int run_state = atomic_load_explicit(&service->continuation->run_state,
                                         memory_order_acquire);
    if (service->realtime_notifiers) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (phase == KC_SERVICE_CREATED && run_state == KORO_NEW) {
        atomic_store_explicit(&service->phase, KC_SERVICE_JOINED,
                              memory_order_release);
    } else if (phase != KC_SERVICE_JOINED ||
               (run_state != KORO_NEW && run_state != KORO_DONE)) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }

    if (service->registry_prev)
        service->registry_prev->registry_next = service->registry_next;
    else if (runtime->services_head == service)
        runtime->services_head = service->registry_next;
    if (service->registry_next)
        service->registry_next->registry_prev = service->registry_prev;
    if (runtime->live_services) runtime->live_services--;
    kc_runtime_unbind_service_locked_internal(runtime, service,
                                              service->continuation);
    KC_MUTEX_UNLOCK(&runtime->mu);

    kc_runtime_signal_lifecycle_internal(runtime);
    koro_cont_release_internal(service->continuation);
    free(service);
    return 0;
}

void kc_service_runtime_stop_locked(kc_runtime_t *runtime)
{
    for (kc_service_t *service = runtime->services_head; service;
         service = service->registry_next) stop_service(service);
}

koro_cont_t *kc_service_continuation_internal(kc_service_t *service)
{
    return service ? service->continuation : NULL;
}

void kc_service_runtime_execute_internal(kc_service_t *service)
{
    if (!service) return;
    kc_runtime_t *runtime = service->runtime;
    koro_cont_t *continuation = service->continuation;
    int state = atomic_load_explicit(&continuation->run_state,
                                     memory_order_acquire);
    if (state != KORO_QUEUED && state != KORO_DORMANT) return;
    atomic_store_explicit(&continuation->run_state, KORO_RUNNING,
                          memory_order_release);
    atomic_fetch_add_explicit(&runtime->running, 1,
                              memory_order_relaxed);
    if (state == KORO_DORMANT)
        atomic_fetch_add_explicit(&runtime->resumes, 1,
                                  memory_order_relaxed);

    void *result = koro_cont_step(continuation);
    atomic_fetch_sub_explicit(&runtime->running, 1,
                              memory_order_relaxed);
    if (result || continuation->completed) {
        kc_runtime_retire_service_internal(runtime, continuation);
        if (continuation->tracked) {
            continuation->tracked = 0;
            atomic_fetch_sub_explicit(&runtime->active, 1,
                                      memory_order_relaxed);
        }
        /* DONE is the retirement acknowledgement. No owner bit remains and
         * no service field is touched after this release publication. */
        atomic_store_explicit(&continuation->run_state, KORO_DONE,
                              memory_order_release);
    } else {
        atomic_store_explicit(&continuation->run_state, KORO_DORMANT,
                              memory_order_release);
    }
    kc_runtime_signal_lifecycle_internal(runtime);
}
