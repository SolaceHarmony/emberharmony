// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_cancel.h"
#include "kcoro_port.h"

#include <stdatomic.h>
#include <stdint.h>

typedef void (*kc_cancel_callback)(void *context);

typedef struct kc_cancel_subscription {
    struct kc_cancel_subscription *prev;
    struct kc_cancel_subscription *next;
    _Atomic(kc_cancel_t *) cancel;
    atomic_int linked;
    kc_cancel_callback callback;
    void *context;
} kc_cancel_subscription;

struct kc_cancel {
    atomic_uint refs;
    atomic_int triggered;
    KC_MUTEX_T mu;
    KC_COND_T cv;
    _Atomic(struct kc_cancel *) parent;
    struct kc_cancel *child_head;
    struct kc_cancel *sibling_prev;
    struct kc_cancel *sibling_next;
    kc_cancel_subscription *subscriptions;
};

int kc_cancel_subscribe(kc_cancel_t *cancel,
                        kc_cancel_subscription *subscription,
                        kc_cancel_callback callback, void *context);
int kc_cancel_unsubscribe(kc_cancel_subscription *subscription);
