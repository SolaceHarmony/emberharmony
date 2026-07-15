// SPDX-License-Identifier: BSD-3-Clause
#include "kc_cancel_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

static void cancel_free(kc_cancel_t *cancel)
{
    KC_COND_DESTROY(&cancel->cv);
    KC_MUTEX_DESTROY(&cancel->mu);
    free(cancel);
}

void kc_cancel_retain(kc_cancel_t *cancel)
{
    if (cancel) atomic_fetch_add_explicit(&cancel->refs, 1, memory_order_relaxed);
}

static void unlink_child_locked(kc_cancel_t *parent, kc_cancel_t *child)
{
    if (child->sibling_prev) child->sibling_prev->sibling_next = child->sibling_next;
    else parent->child_head = child->sibling_next;
    if (child->sibling_next) child->sibling_next->sibling_prev = child->sibling_prev;
    atomic_store_explicit(&child->parent, NULL, memory_order_release);
    child->sibling_prev = NULL;
    child->sibling_next = NULL;
}

void kc_cancel_release(kc_cancel_t *cancel)
{
    if (!cancel) return;
    for (;;) {
        kc_cancel_t *parent = atomic_load_explicit(&cancel->parent,
                                                   memory_order_acquire);
        if (!parent) {
            if (atomic_fetch_sub_explicit(&cancel->refs, 1,
                                          memory_order_acq_rel) == 1) {
                cancel_free(cancel);
            }
            return;
        }
        KC_MUTEX_LOCK(&parent->mu);
        if (atomic_load_explicit(&cancel->parent, memory_order_acquire) != parent) {
            KC_MUTEX_UNLOCK(&parent->mu);
            continue;
        }
        unsigned previous = atomic_fetch_sub_explicit(&cancel->refs, 1,
                                                      memory_order_acq_rel);
        if (previous != 1) { KC_MUTEX_UNLOCK(&parent->mu); return; }
        unlink_child_locked(parent, cancel);
        KC_MUTEX_UNLOCK(&parent->mu);
        kc_cancel_release(parent);
        cancel_free(cancel);
        return;
    }
}

int kc_cancel_add_child(kc_cancel_t *parent, kc_cancel_t *child)
{
    if (!parent || !child || parent == child) return -EINVAL;
    KC_MUTEX_LOCK(&parent->mu);
    if (atomic_load_explicit(&child->parent, memory_order_acquire)) {
        KC_MUTEX_UNLOCK(&parent->mu);
        return -EALREADY;
    }
    if (atomic_load_explicit(&parent->triggered, memory_order_acquire)) {
        KC_MUTEX_UNLOCK(&parent->mu);
        kc_cancel_trigger(child);
        return 0;
    }
    child->sibling_next = parent->child_head;
    if (parent->child_head) parent->child_head->sibling_prev = child;
    parent->child_head = child;
    atomic_store_explicit(&child->parent, parent, memory_order_release);
    kc_cancel_retain(parent);
    KC_MUTEX_UNLOCK(&parent->mu);
    return 0;
}

void kc_cancel_remove_child(kc_cancel_t *parent, kc_cancel_t *child)
{
    if (!parent || !child) return;
    int removed = 0;
    KC_MUTEX_LOCK(&parent->mu);
    if (atomic_load_explicit(&child->parent, memory_order_acquire) == parent) {
        unlink_child_locked(parent, child);
        removed = 1;
    }
    KC_MUTEX_UNLOCK(&parent->mu);
    if (removed) kc_cancel_release(parent);
}

int kc_cancel_create(kc_cancel_t **out, kc_cancel_t *parent)
{
    if (!out) return -EINVAL;
    kc_cancel_t *cancel = calloc(1, sizeof(*cancel));
    if (!cancel) return -ENOMEM;
    atomic_init(&cancel->refs, 1);
    atomic_init(&cancel->triggered, 0);
    atomic_init(&cancel->parent, NULL);
    if (KC_MUTEX_INIT(&cancel->mu) != 0) { free(cancel); return -ENOMEM; }
    if (KC_COND_INIT(&cancel->cv) != 0) {
        KC_MUTEX_DESTROY(&cancel->mu);
        free(cancel);
        return -ENOMEM;
    }
    int rc = parent ? kc_cancel_add_child(parent, cancel) : 0;
    if (rc != 0) { kc_cancel_release(cancel); return rc; }
    *out = cancel;
    return 0;
}

int kc_cancel_subscribe(kc_cancel_t *cancel,
                        kc_cancel_subscription *subscription,
                        kc_cancel_callback callback, void *context)
{
    if (!cancel || !subscription || !callback) return -EINVAL;
    KC_MUTEX_LOCK(&cancel->mu);
    if (atomic_load_explicit(&cancel->triggered, memory_order_acquire)) {
        KC_MUTEX_UNLOCK(&cancel->mu);
        return -ECANCELED;
    }
    subscription->callback = callback;
    subscription->context = context;
    subscription->prev = NULL;
    subscription->next = cancel->subscriptions;
    if (cancel->subscriptions) cancel->subscriptions->prev = subscription;
    cancel->subscriptions = subscription;
    atomic_store_explicit(&subscription->cancel, cancel, memory_order_release);
    atomic_store_explicit(&subscription->linked, 1, memory_order_release);
    KC_MUTEX_UNLOCK(&cancel->mu);
    return 0;
}

int kc_cancel_unsubscribe(kc_cancel_subscription *subscription)
{
    if (!subscription) return 0;
    kc_cancel_t *cancel = atomic_load_explicit(&subscription->cancel,
                                               memory_order_acquire);
    if (!cancel) return 0;
    KC_MUTEX_LOCK(&cancel->mu);
    if (!atomic_load_explicit(&subscription->linked, memory_order_acquire) ||
        atomic_load_explicit(&subscription->cancel, memory_order_acquire) != cancel) {
        KC_MUTEX_UNLOCK(&cancel->mu);
        return 0;
    }
    if (subscription->prev) subscription->prev->next = subscription->next;
    else cancel->subscriptions = subscription->next;
    if (subscription->next) subscription->next->prev = subscription->prev;
    subscription->prev = NULL;
    subscription->next = NULL;
    atomic_store_explicit(&subscription->linked, 0, memory_order_release);
    atomic_store_explicit(&subscription->cancel, NULL, memory_order_release);
    KC_MUTEX_UNLOCK(&cancel->mu);
    return 1;
}

void kc_cancel_trigger(kc_cancel_t *cancel)
{
    if (!cancel) return;
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&cancel->triggered, &expected, 1,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) return;

    KC_MUTEX_LOCK(&cancel->mu);
    KC_COND_BROADCAST(&cancel->cv);
    kc_cancel_subscription *subscriptions = cancel->subscriptions;
    cancel->subscriptions = NULL;
    for (kc_cancel_subscription *item = subscriptions; item; item = item->next) {
        item->prev = NULL;
        atomic_store_explicit(&item->linked, 0, memory_order_release);
        atomic_store_explicit(&item->cancel, NULL, memory_order_release);
    }
    KC_MUTEX_UNLOCK(&cancel->mu);

    while (subscriptions) {
        kc_cancel_subscription *next = subscriptions->next;
        subscriptions->next = NULL;
        subscriptions->callback(subscriptions->context);
        subscriptions = next;
    }

    for (;;) {
        KC_MUTEX_LOCK(&cancel->mu);
        kc_cancel_t *child = cancel->child_head;
        while (child && kc_cancel_is_triggered(child)) child = child->sibling_next;
        if (child) kc_cancel_retain(child);
        KC_MUTEX_UNLOCK(&cancel->mu);
        if (!child) break;
        kc_cancel_trigger(child);
        kc_cancel_release(child);
    }
}

int kc_cancel_is_triggered(const kc_cancel_t *cancel)
{
    return cancel && atomic_load_explicit(&cancel->triggered, memory_order_acquire);
}

int kc_cancel_wait(const kc_cancel_t *value, long timeout_ms)
{
    if (!value) return -EINVAL;
    kc_cancel_t *cancel = (kc_cancel_t *)value;
    if (kc_cancel_is_triggered(cancel)) return 0;
    KC_MUTEX_LOCK(&cancel->mu);
    if (timeout_ms < 0) {
        while (!kc_cancel_is_triggered(cancel)) KC_COND_WAIT(&cancel->cv, &cancel->mu);
        KC_MUTEX_UNLOCK(&cancel->mu);
        return 0;
    }
    uint64_t now = kc_port_monotonic_ns();
    uint64_t delta = (uint64_t)timeout_ms * UINT64_C(1000000);
    uint64_t deadline = UINT64_MAX - now < delta ? UINT64_MAX : now + delta;
    int rc = 0;
    while (!kc_cancel_is_triggered(cancel)) {
        rc = KC_COND_TIMEDWAIT_NS(&cancel->cv, &cancel->mu, deadline);
        if (rc == -ETIMEDOUT) break;
    }
    KC_MUTEX_UNLOCK(&cancel->mu);
    return rc;
}
