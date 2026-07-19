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
 * runtime. It creates no thread. Notifications are edge-coalesced while the
 * callback drains its own predicate, and the runtime's continuation state
 * closes notify-before-park and notify-during-callback races. Notify and stop
 * linearize at the service's packed admission gate: a successful notify
 * receives a callback before the service retires, while notifications after
 * stopping return -ECANCELED.
 */
typedef struct kc_service kc_service_t;
typedef struct kc_service_notifier kc_service_notifier_t;

typedef void (*kc_service_fn)(void *context);

typedef struct kc_service_config {
    uint32_t size;
    uint32_t abi_version;
    kc_service_fn callback;
    void *context;
    uint64_t reserved;
} kc_service_config;

typedef struct kc_service_snapshot {
    uint32_t size;
    uint32_t abi_version;
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
/* General control-plane notification. This may acquire runtime->mu and is not
 * admissible from a realtime callback. */
int kc_service_notify(kc_service_t *service);
/* Setup-time retained realtime edge. Creation may allocate and lock; notify
 * performs no mutex, allocation, deadline, or callback. The producer publishes
 * its owned predicate before notify. Stop closes service admission. The host
 * must disconnect and quiesce every producer before notifier_destroy; this is
 * the same ownership boundary used before releasing a hardware callback
 * context. A live notifier makes service_destroy return -EBUSY. Creation
 * returns -ENOTSUP when the host lacks a direct address-wake backend. */
int kc_service_notifier_create(kc_service_t *service,
                               kc_service_notifier_t **out);
int kc_service_notifier_notify(kc_service_notifier_t *notifier);
int kc_service_notifier_destroy(kc_service_notifier_t *notifier);
void kc_service_request_stop(kc_service_t *service);
int kc_service_join(kc_service_t *service);
int kc_service_snapshot_get(kc_service_t *service, kc_service_snapshot *out);
int kc_service_destroy(kc_service_t *service);

#ifdef __cplusplus
}
#endif

#endif
