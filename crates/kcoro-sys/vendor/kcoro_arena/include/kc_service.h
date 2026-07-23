// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_SERVICE_H
#define KC_SERVICE_H

#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A retained service is one stackless continuation mounted on an explicit
 * runtime. It creates no thread. Normal services are resumable by any free
 * runtime worker; owner hooks narrow eligibility to one worker only for a
 * genuinely thread-affine host resource. Notifications are coalesced while
 * the callback drains its own predicate. The continuation state
 * closes notify-before-dormancy and notify-during-callback races. Realtime
 * notify takes one bounded admission lease, with no compare/exchange retry:
 * a successful notify
 * receives a callback before the service retires, while notifications after
 * stopping return -ECANCELED. Every value that survives a callback return
 * belongs to the retained context or its owned predicate; no blocked C stack
 * is part of the service state machine.
 */
typedef struct kc_service kc_service_t;
typedef struct kc_service_notifier kc_service_notifier_t;

typedef void (*kc_service_fn)(void *context);

typedef struct kc_service_config {
    kc_service_fn callback;
    void *context;
    /* Optional owner-affine lifecycle hooks. owner_init runs exactly once on
     * the service's eligible worker before its first callback can run.
     * owner_fini runs exactly once on that same worker after every admitted
     * edge has drained and before DONE is published. They are the lifetime
     * boundary for resources that may neither migrate nor be destroyed by an
     * administrative joiner. Neither hook may block or wait for another edge. */
    kc_service_fn owner_init;
    kc_service_fn owner_fini;
} kc_service_config;

typedef struct kc_service_snapshot {
    uint64_t notifications;
    uint64_t handled_notifications;
    uint64_t callbacks;
    uint32_t run_state;
    uint32_t started;
    uint32_t stop_requested;
    uint32_t joined;
} kc_service_snapshot;

int kc_service_create(kc_runtime_t *runtime, const kc_service_config *config,
                      kc_service_t **out);
int kc_service_start(kc_service_t *service);
/* Atomics-only MPSC control edge. The caller owns its predicate publication;
 * this function allocates nothing, takes no mutex, and invokes no callback. */
int kc_service_notify(kc_service_t *service);
/* Setup-time retained realtime edge. Creation may allocate and lock; notify
 * performs no mutex, allocation, retry loop, deadline, or callback. A burst
 * publishes only the continuation's ready transition. The producer publishes
 * its owned predicate before notify. Stop closes service admission. The host
 * must disconnect and quiesce every producer before notifier_destroy; this is
 * the same ownership boundary used before releasing a hardware callback
 * context. A live notifier makes service_destroy return -EBUSY. Creation
 * returns -ENOTSUP when the host lacks a direct address-wake backend. */
int kc_service_notifier_create(kc_service_t *service,
                               kc_service_notifier_t **out);
int kc_service_notifier_notify(kc_service_notifier_t *notifier);
int kc_service_notifier_destroy(kc_service_notifier_t *notifier);
/* Bounded-callback continuation edge. Callable only from this service's active
 * callback. It publishes one coalescible local-ready generation and causes the
 * same logical continuation to re-enter after yielding, without a mutex, timer,
 * external producer, or wait-word syscall. This is the quota boundary for a
 * callback whose owned predicate still contains work. */
int kc_service_ready_again(kc_service_t *service);
/* Natural terminal edge for the currently-running retained callback. Closes
 * future notification admission and retires after every already-admitted edge
 * drains. It neither stops nor joins the owning runtime. */
int kc_service_complete_current(kc_service_t *service);
void kc_service_request_stop(kc_service_t *service);
/* A runtime callback may request stop but must return before joining; join
 * returns -EDEADLK when called from any callback on the owning runtime. */
int kc_service_join(kc_service_t *service);
int kc_service_snapshot_get(kc_service_t *service, kc_service_snapshot *out);
int kc_service_destroy(kc_service_t *service);

#ifdef __cplusplus
}
#endif

#endif
