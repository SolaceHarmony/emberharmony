// SPDX-License-Identifier: BSD-3-Clause
#include "kc_actor.h"
#include "kc_runtime_internal.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

struct kc_actor {
    atomic_uint refs;
    atomic_int closed;
    atomic_uint_fast64_t messages;
    atomic_uint_fast64_t failures;
    atomic_uint_fast64_t callback_ns;
    atomic_uint_fast64_t max_callback_ns;
    kc_runtime_t *runtime;
    kc_channel_t *mailbox;
    kc_scope_t *scope;
    kc_actor_receive_fn receive;
    void *context;
    kc_id id;
};

static void update_max(atomic_uint_fast64_t *value, uint64_t sample)
{
    uint64_t current = atomic_load_explicit(value, memory_order_relaxed);
    while (current < sample &&
           !atomic_compare_exchange_weak_explicit(value, &current, sample,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) { }
}

static void *actor_step(koro_cont_t *cont)
{
    kc_actor_t *actor = cont->user_arg;
    KORO_BEGIN(cont);
    for (;;) {
        KORO_RECV_EX(cont, actor->mailbox, kc_scope_cancel_token(actor->scope), 0);
        if (cont->last_park_result == -EPIPE ||
            cont->last_park_result == -ECANCELED) break;
        if (cont->last_park_result != 0) {
            atomic_fetch_add_explicit(&actor->failures, 1, memory_order_relaxed);
            break;
        }
        uint64_t start = kc_port_monotonic_ns();
        int rc = actor->receive(actor->context, cont->arena_payload,
                                cont->arena_payload_len);
        uint64_t elapsed = kc_port_monotonic_ns() - start;
        atomic_fetch_add_explicit(&actor->callback_ns, elapsed, memory_order_relaxed);
        update_max(&actor->max_callback_ns, elapsed);
        atomic_fetch_add_explicit(&actor->messages, 1, memory_order_relaxed);
        if (rc != 0) atomic_fetch_add_explicit(&actor->failures, 1,
                                               memory_order_relaxed);
    }
    KORO_END(cont);
}

int kc_actor_create(kc_runtime_t *runtime, const kc_actor_config *config,
                    kc_actor_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->element_size ||
        !config->receive) return -EINVAL;
    kc_actor_t *actor = calloc(1, sizeof(*actor));
    if (!actor) return -ENOMEM;
    atomic_init(&actor->refs, 1);
    atomic_init(&actor->closed, 0);
    atomic_init(&actor->messages, 0);
    atomic_init(&actor->failures, 0);
    atomic_init(&actor->callback_ns, 0);
    atomic_init(&actor->max_callback_ns, 0);
    actor->runtime = runtime;
    actor->receive = config->receive;
    actor->context = config->context;
    actor->id = (kc_id){ runtime->epoch, kc_runtime_next_sequence(runtime) };

    kc_scope_config scope_config = {
        .size = sizeof(scope_config), .abi_version = KC_ABI_VERSION,
        .parent_cancel = config->parent_cancel,
    };
    int rc = kc_scope_create(runtime, &scope_config, &actor->scope);
    if (rc != 0) { free(actor); return rc; }
    kc_channel_config channel_config = {
        .size = sizeof(channel_config), .abi_version = KC_ABI_VERSION,
        .kind = config->mailbox_kind, .element_size = config->element_size,
        .capacity = config->mailbox_capacity,
    };
    rc = kc_channel_create(runtime, &channel_config, &actor->mailbox);
    if (rc != 0) {
        kc_scope_release(actor->scope);
        free(actor);
        return rc;
    }
    rc = kc_scope_spawn(actor->scope, actor_step, actor, 0);
    if (rc != 0) {
        kc_channel_release(actor->mailbox);
        kc_scope_release(actor->scope);
        free(actor);
        return rc;
    }
    *out = actor;
    return 0;
}

void kc_actor_retain(kc_actor_t *actor)
{
    if (actor) atomic_fetch_add_explicit(&actor->refs, 1, memory_order_relaxed);
}

void kc_actor_close(kc_actor_t *actor)
{
    if (!actor || atomic_exchange_explicit(&actor->closed, 1,
                                           memory_order_acq_rel)) return;
    kc_channel_close(actor->mailbox);
    kc_scope_close(actor->scope);
}

int kc_actor_join(kc_actor_t *actor, uint64_t deadline_ns)
{
    return actor ? kc_scope_join(actor->scope, deadline_ns) : -EINVAL;
}

void kc_actor_release(kc_actor_t *actor)
{
    if (!actor) return;
    if (atomic_fetch_sub_explicit(&actor->refs, 1, memory_order_acq_rel) != 1) return;
    kc_actor_close(actor);
    (void)kc_actor_join(actor, 0);
    kc_channel_release(actor->mailbox);
    kc_scope_release(actor->scope);
    free(actor);
}

kc_channel_t *kc_actor_mailbox(kc_actor_t *actor)
{
    return actor ? actor->mailbox : NULL;
}

int kc_actor_snapshot_get(kc_actor_t *actor, kc_actor_snapshot *out)
{
    if (!actor || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_actor_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = actor->id,
        .messages = atomic_load_explicit(&actor->messages, memory_order_relaxed),
        .failures = atomic_load_explicit(&actor->failures, memory_order_relaxed),
        .callback_ns = atomic_load_explicit(&actor->callback_ns, memory_order_relaxed),
        .max_callback_ns = atomic_load_explicit(&actor->max_callback_ns,
                                                memory_order_relaxed),
        .mailbox_depth = kc_channel_length(actor->mailbox),
        .closed = (unsigned)atomic_load_explicit(&actor->closed, memory_order_acquire),
    };
    return 0;
}
