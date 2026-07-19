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
    KC_SERVICE_STOPPING,
    KC_SERVICE_JOINED,
};

#define KC_SERVICE_RT_CLOSED UINT32_C(0x80000000)
#define KC_SERVICE_RT_COUNT UINT32_C(0x7fffffff)

struct kc_service {
    kc_runtime_t *runtime;
    koro_cont_t *continuation;
    kc_service_fn callback;
    void *context;
    struct kc_service *registry_prev;
    struct kc_service *registry_next;
    atomic_uint_fast64_t notifications;
    atomic_uint_fast64_t handled_notifications;
    atomic_uint_fast64_t callbacks;
    atomic_uint phase;
    atomic_uint ever_started;
    atomic_uint stop_requested;
    /* High bit closes admission; low bits count publishers that crossed the
     * gate before stop. One CAS makes that decision indivisible. */
    atomic_uint realtime_gate;
    size_t realtime_notifiers;
    int realtime_capable;
};

struct kc_service_notifier {
    kc_service_t *service;
};

static void *service_step(koro_cont_t *continuation)
{
    kc_service_t *service = continuation->user_arg;
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
     * Stop closes admission under runtime->mu. A successful notify therefore
     * precedes STOPPING and is visible here. Do not retire until every accepted
     * notification generation has received its drain callback.
     */
    if (atomic_load_explicit(&service->phase, memory_order_acquire) ==
            KC_SERVICE_STOPPING &&
        (atomic_load_explicit(&service->realtime_gate,
                              memory_order_acquire) &
         KC_SERVICE_RT_COUNT) == 0 &&
        atomic_load_explicit(&service->notifications, memory_order_acquire) ==
            handled) {
        continuation->completed = 1;
        return (void *)1;
    }
    return NULL;
}

static void stop_locked(kc_service_t *service)
{
    unsigned phase = atomic_load_explicit(&service->phase, memory_order_acquire);
    if (phase == KC_SERVICE_CREATED || phase == KC_SERVICE_STARTED) {
        atomic_fetch_or_explicit(&service->realtime_gate,
                                 KC_SERVICE_RT_CLOSED,
                                 memory_order_acq_rel);
        atomic_store_explicit(&service->phase, KC_SERVICE_STOPPING,
                              memory_order_release);
        atomic_store_explicit(&service->stop_requested, 1,
                              memory_order_release);
        atomic_fetch_add_explicit(&service->runtime->wake_requests, 1,
                                  memory_order_relaxed);
        kc_runtime_wake_locked_internal(service->continuation);
    }
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
    service->context = config->context;
    atomic_init(&service->notifications, 0);
    atomic_init(&service->handled_notifications, 0);
    atomic_init(&service->callbacks, 0);
    atomic_init(&service->phase, KC_SERVICE_CREATED);
    atomic_init(&service->ever_started, 0);
    atomic_init(&service->stop_requested, 0);
    atomic_init(&service->realtime_gate, KC_SERVICE_RT_CLOSED);
    service->realtime_capable =
        atomic_is_lock_free(&service->notifications) &&
        atomic_is_lock_free(&service->realtime_gate) &&
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

    atomic_store_explicit(&service->phase, KC_SERVICE_STARTED,
                          memory_order_release);
    service->continuation->tracked = 1;
    runtime->active++;
    if (kc_runtime_enqueue_locked_internal(runtime, service->continuation,
                                           KORO_NEW)) {
        atomic_store_explicit(&service->ever_started, 1, memory_order_release);
        atomic_store_explicit(&service->realtime_gate, 0,
                              memory_order_release);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }

    service->continuation->tracked = 0;
    if (runtime->active) runtime->active--;
    atomic_store_explicit(&service->phase, KC_SERVICE_CREATED,
                          memory_order_release);
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return -EAGAIN;
}

int kc_service_notify(kc_service_t *service)
{
    if (!service) return -EINVAL;
    kc_runtime_t *runtime = service->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (atomic_load_explicit(&service->phase, memory_order_acquire) !=
        KC_SERVICE_STARTED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    atomic_fetch_add_explicit(&service->notifications, 1, memory_order_release);
    atomic_fetch_add_explicit(&runtime->wake_requests, 1, memory_order_relaxed);
    kc_runtime_wake_locked_internal(service->continuation);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

static int notify_realtime(kc_service_t *service)
{
    if (!service->realtime_capable) return -ENOTSUP;

    unsigned gate = atomic_load_explicit(&service->realtime_gate,
                                         memory_order_acquire);
    for (;;) {
        if (gate & KC_SERVICE_RT_CLOSED) return -ECANCELED;
        if ((gate & KC_SERVICE_RT_COUNT) == KC_SERVICE_RT_COUNT)
            return -EAGAIN;
        if (atomic_compare_exchange_weak_explicit(
                &service->realtime_gate, &gate, gate + 1,
                memory_order_acquire, memory_order_acquire)) break;
    }

    /* Publish the callback predicate first, then drop this publisher's
     * admission lease. The final doorbell ring makes both releases visible to
     * the worker; if stop raced this publisher, that final edge is what lets a
     * service that previously observed a non-zero count run once more and
     * retire. */
    atomic_fetch_add_explicit(&service->notifications, 1,
                              memory_order_release);
    atomic_fetch_sub_explicit(&service->realtime_gate, 1,
                              memory_order_release);
    atomic_fetch_add_explicit(&service->runtime->wake_requests, 1,
                              memory_order_relaxed);
    kc_runtime_ring_work_internal(service->runtime, 0);
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
    if (phase == KC_SERVICE_STOPPING || phase == KC_SERVICE_JOINED) {
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
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    free(notifier);
    return 0;
}

int kc_service_ready_again(kc_service_t *service)
{
    if (!service) return -EINVAL;
    if (!kc_runtime_is_current_cont_internal(service->continuation))
        return -EPERM;

    /* Use the same packed admission gate as the realtime producer edge, so a
     * local reschedule linearizes cleanly against stop. Unlike an external
     * notification, the current continuation needs no doorbell: publishing
     * wake_pending before releasing the admission lease makes suspend_cont
     * requeue it directly after this bounded callback returns. */
    unsigned gate = atomic_load_explicit(&service->realtime_gate,
                                         memory_order_acquire);
    for (;;) {
        if (gate & KC_SERVICE_RT_CLOSED) return -ECANCELED;
        if ((gate & KC_SERVICE_RT_COUNT) == KC_SERVICE_RT_COUNT)
            return -EAGAIN;
        if (atomic_compare_exchange_weak_explicit(
                &service->realtime_gate, &gate, gate + 1,
                memory_order_acquire, memory_order_acquire)) break;
    }

    atomic_fetch_add_explicit(&service->notifications, 1,
                              memory_order_release);
    atomic_fetch_add_explicit(&service->runtime->wake_requests, 1,
                              memory_order_relaxed);
    atomic_store_explicit(&service->continuation->wake_pending, 1,
                          memory_order_release);
    atomic_fetch_sub_explicit(&service->realtime_gate, 1,
                              memory_order_release);
    return 0;
}

void kc_service_request_stop(kc_service_t *service)
{
    if (!service) return;
    KC_MUTEX_LOCK(&service->runtime->mu);
    stop_locked(service);
    KC_MUTEX_UNLOCK(&service->runtime->mu);
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
    if (phase != KC_SERVICE_STOPPING) {
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
        KC_COND_WAIT(&runtime->lifecycle_cv, &runtime->mu);
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
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);

    koro_cont_release_internal(service->continuation);
    free(service);
    return 0;
}

void kc_service_runtime_stop_locked(kc_runtime_t *runtime)
{
    for (kc_service_t *service = runtime->services_head; service;
         service = service->registry_next) stop_locked(service);
}

void kc_service_runtime_drain_realtime_locked(kc_runtime_t *runtime)
{
    for (kc_service_t *service = runtime->services_head; service;
         service = service->registry_next) {
        const uint64_t notified = atomic_load_explicit(
            &service->notifications, memory_order_acquire);
        const uint64_t handled = atomic_load_explicit(
            &service->handled_notifications, memory_order_acquire);
        const unsigned phase = atomic_load_explicit(
            &service->phase, memory_order_acquire);
        const unsigned gate = atomic_load_explicit(
            &service->realtime_gate, memory_order_acquire);
        if (notified == handled &&
            !(phase == KC_SERVICE_STOPPING &&
              (gate & KC_SERVICE_RT_COUNT) == 0)) continue;
        kc_runtime_wake_locked_internal(service->continuation);
    }
}
