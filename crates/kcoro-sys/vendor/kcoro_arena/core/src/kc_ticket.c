// SPDX-License-Identifier: BSD-3-Clause
#include "kc_ticket_internal.h"
#include "kc_descriptor_internal.h"
#include "kc_runtime_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static int id_is_zero(kc_id id)
{
    return id.epoch == 0 && id.sequence == 0;
}

static int deadline_expired(const kc_ticket_t *ticket, uint64_t now)
{
    return ticket->deadline_ns && now >= ticket->deadline_ns;
}

int kc_ticket_runtime_init(kc_runtime_t *runtime, uint32_t capacity)
{
    if (!runtime || capacity == 0) return -EINVAL;
    runtime->tickets = calloc(capacity, sizeof(*runtime->tickets));
    if (!runtime->tickets) return -ENOMEM;
    runtime->ticket_capacity = capacity;
    runtime->ticket_free_head = 1;
    for (uint32_t slot = 0; slot < capacity; slot++) {
        runtime->tickets[slot].slot = slot;
        runtime->tickets[slot].free_next = slot + 1 < capacity ? slot + 2 : 0;
    }
    return 0;
}

void kc_ticket_runtime_destroy(kc_runtime_t *runtime)
{
    if (!runtime) return;
    free(runtime->tickets);
    runtime->tickets = NULL;
    runtime->ticket_capacity = 0;
    runtime->ticket_free_head = 0;
}

static void queue_completion_locked(kc_ticket_t *ticket)
{
    kc_runtime_t *runtime = ticket->runtime;
    ticket->completion_next = NULL;
    ticket->delivery_queued = 1;
    if (runtime->completion_tail) {
        runtime->completion_tail->completion_next = ticket;
    } else {
        runtime->completion_head = ticket;
    }
    runtime->completion_tail = ticket;
    runtime->completion_queued++;
    KC_COND_SIGNAL(&runtime->work_cv);
}

static int publish_locked(kc_ticket_t *ticket,
                          kc_ticket_execution_status execution,
                          kc_ticket_state_status state,
                          kc_ticket_publication_status publication,
                          kc_ticket_terminal_cause cause,
                          int status,
                          kc_descriptor_t *result)
{
    if (ticket->state == KC_TICKET_PUBLISHED) return 0;
    uint64_t now = kc_port_monotonic_ns();
    if (result) kc_descriptor_retain(result);
    ticket->result = result;
    ticket->event.execution_status = execution;
    ticket->event.state_status = state;
    ticket->event.publication_status = publication;
    ticket->event.terminal_cause = cause;
    ticket->event.status_code = status;
    ticket->event.result = kc_descriptor_id_get(result);
    ticket->event.completed_ns = now;
    ticket->event.published_ns = now;
    ticket->state = KC_TICKET_PUBLISHED;
    queue_completion_locked(ticket);
    return 1;
}

int kc_ticket_create(kc_runtime_t *runtime,
                     const kc_ticket_config_v1 *config,
                     kc_ticket_t **out)
{
    if (!runtime || !config || !out ||
        config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->callback ||
        config->deadline_mode < KC_TICKET_DEADLINE_NONE ||
        config->deadline_mode > KC_TICKET_DEADLINE_SOFT) return -EINVAL;
    if (config->deadline_mode == KC_TICKET_DEADLINE_NONE && config->deadline_ns != 0)
        return -EINVAL;
    if ((config->callback_context &&
         (!config->context_retain || !config->context_release)) ||
        (!config->callback_context &&
         (config->context_retain || config->context_release))) return -EINVAL;

    if (config->context_retain) config->context_retain(config->callback_context);

    KC_MUTEX_LOCK(&runtime->mu);
    if (!runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        if (config->context_release) config->context_release(config->callback_context);
        return -ECANCELED;
    }
    if (!runtime->ticket_free_head) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        if (config->context_release) config->context_release(config->callback_context);
        return -EAGAIN;
    }

    uint32_t slot = runtime->ticket_free_head - 1;
    kc_ticket_t *ticket = &runtime->tickets[slot];
    uint32_t next = ticket->free_next;
    uint32_t generation = ticket->generation + 1;
    if (generation == 0) generation = 1;
    memset(ticket, 0, sizeof(*ticket));
    atomic_init(&ticket->refs, 2); /* caller + reserved terminal delivery */
    ticket->runtime = runtime;
    ticket->slot = slot;
    ticket->generation = generation;
    ticket->in_use = 1;
    ticket->state = KC_TICKET_CREATED;
    ticket->deadline_mode = (kc_ticket_deadline_mode)config->deadline_mode;
    ticket->deadline_ns = config->deadline_ns;
    ticket->callback = config->callback;
    ticket->callback_context = config->callback_context;
    ticket->context_release = config->context_release;
    ticket->event = (kc_ticket_event_v1){
        .size = sizeof(kc_ticket_event_v1),
        .abi_version = KC_ABI_VERSION,
        .flags = config->flags,
        .ticket = {
            .runtime_epoch = runtime->epoch,
            .sequence = kc_runtime_next_sequence(runtime),
            .slot = slot,
            .generation = generation,
            .kind = config->kind,
        },
        .parent = config->parent,
        .correlation = config->correlation,
        .trace = config->trace,
        .context_id = config->context_id,
        .epoch = config->epoch,
        .execution_status = KC_TICKET_NOT_DISPATCHED,
        .state_status = KC_TICKET_STATE_NONE,
        .publication_status = KC_TICKET_PUBLICATION_NONE,
        .terminal_cause = KC_TICKET_CAUSE_REJECTED,
    };
    if (id_is_zero(ticket->event.correlation)) {
        ticket->event.correlation.epoch = runtime->epoch;
        ticket->event.correlation.sequence = ticket->event.ticket.sequence;
    }
    if (id_is_zero(ticket->event.trace)) ticket->event.trace = ticket->event.correlation;
    runtime->ticket_free_head = next;
    runtime->live_tickets++;
    kc_runtime_retain_internal(runtime);
    KC_MUTEX_UNLOCK(&runtime->mu);
    *out = ticket;
    return 0;
}

void kc_ticket_retain(kc_ticket_t *ticket)
{
    if (ticket) atomic_fetch_add_explicit(&ticket->refs, 1, memory_order_relaxed);
}

void kc_ticket_release(kc_ticket_t *ticket)
{
    if (!ticket) return;
    if (atomic_fetch_sub_explicit(&ticket->refs, 1, memory_order_acq_rel) != 1) return;

    kc_runtime_t *runtime = ticket->runtime;
    kc_descriptor_t *descriptor = ticket->descriptor;
    kc_descriptor_t *result = ticket->result;
    ticket->descriptor = NULL;
    ticket->result = NULL;

    /*
     * A slot is not reusable while any lease from its previous incarnation is
     * still being released. Host-owned region release callbacks may block, and
     * join_all must continue to observe this ticket until those callbacks have
     * returned.
     */
    kc_descriptor_release(descriptor);
    kc_descriptor_release(result);

    KC_MUTEX_LOCK(&runtime->mu);
    ticket->in_use = 0;
    ticket->free_next = runtime->ticket_free_head;
    runtime->ticket_free_head = ticket->slot + 1;
    if (runtime->live_tickets) runtime->live_tickets--;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_release_internal(runtime);
}

int kc_ticket_attach_descriptor(kc_ticket_t *ticket,
                                kc_descriptor_t *descriptor)
{
    if (!ticket || !descriptor || descriptor->runtime != ticket->runtime) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!ticket->in_use || ticket->state != KC_TICKET_CREATED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EALREADY;
    }
    kc_descriptor_retain(descriptor);
    kc_descriptor_t *previous = ticket->descriptor;
    ticket->descriptor = descriptor;
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_descriptor_release(previous);
    return 0;
}

int kc_ticket_accept(kc_ticket_t *ticket)
{
    if (!ticket) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!ticket->in_use || ticket->state != KC_TICKET_CREATED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EALREADY;
    }
    if (!runtime->accepting) {
        (void)publish_locked(ticket, KC_TICKET_NOT_DISPATCHED,
                             KC_TICKET_STATE_NONE, KC_TICKET_PUBLICATION_NONE,
                             KC_TICKET_CAUSE_STOP, -ECANCELED, NULL);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    if ((ticket->deadline_mode == KC_TICKET_DEADLINE_QUEUE ||
         ticket->deadline_mode == KC_TICKET_DEADLINE_HARD_PUBLICATION) &&
        deadline_expired(ticket, kc_port_monotonic_ns())) {
        (void)publish_locked(ticket, KC_TICKET_NOT_DISPATCHED,
                             KC_TICKET_STATE_NONE, KC_TICKET_PUBLICATION_NONE,
                             KC_TICKET_CAUSE_TIMED_OUT, -ETIMEDOUT, NULL);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ETIMEDOUT;
    }
    ticket->state = KC_TICKET_ACCEPTED;
    ticket->event.accepted_ns = kc_port_monotonic_ns();
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_ticket_dispatch(kc_ticket_t *ticket)
{
    if (!ticket) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!ticket->in_use || ticket->state != KC_TICKET_ACCEPTED) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EALREADY;
    }
    if ((ticket->deadline_mode == KC_TICKET_DEADLINE_QUEUE ||
         ticket->deadline_mode == KC_TICKET_DEADLINE_HARD_PUBLICATION) &&
        deadline_expired(ticket, kc_port_monotonic_ns())) {
        (void)publish_locked(ticket, KC_TICKET_NOT_DISPATCHED,
                             KC_TICKET_STATE_NONE, KC_TICKET_PUBLICATION_NONE,
                             KC_TICKET_CAUSE_TIMED_OUT, -ETIMEDOUT, NULL);
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ETIMEDOUT;
    }
    ticket->state = KC_TICKET_DISPATCHED;
    ticket->event.dispatched_ns = kc_port_monotonic_ns();
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

static int completion_valid(const kc_ticket_completion_v1 *completion)
{
    return completion && completion->size >= sizeof(*completion) &&
        completion->abi_version == KC_ABI_VERSION &&
        completion->execution_status >= KC_TICKET_EXECUTION_COMPLETED &&
        completion->execution_status <= KC_TICKET_EXECUTION_FAILED &&
        completion->state_status >= KC_TICKET_STATE_NONE &&
        completion->state_status <= KC_TICKET_STATE_POISONED &&
        completion->publication_status >= KC_TICKET_PUBLICATION_NONE &&
        completion->publication_status <= KC_TICKET_PUBLICATION_STALE &&
        completion->terminal_cause >= KC_TICKET_CAUSE_SUCCESS &&
        completion->terminal_cause <= KC_TICKET_CAUSE_FAULT;
}

static kc_ticket_t *ticket_from_id_locked(kc_runtime_t *runtime, kc_ticket_id id)
{
    if (id.runtime_epoch != runtime->epoch || id.slot >= runtime->ticket_capacity)
        return NULL;
    kc_ticket_t *ticket = &runtime->tickets[id.slot];
    kc_ticket_id current = ticket->event.ticket;
    if (!ticket->in_use || current.sequence != id.sequence ||
        current.generation != id.generation || current.kind != id.kind)
        return NULL;
    return ticket;
}

static int complete_locked(kc_ticket_t *ticket,
                           const kc_ticket_completion_v1 *completion)
{
    if (completion->result && completion->result->runtime != ticket->runtime)
        return -EXDEV;
    if (!ticket->in_use || ticket->state != KC_TICKET_DISPATCHED) return 0;

    kc_ticket_state_status state = (kc_ticket_state_status)completion->state_status;
    kc_ticket_publication_status publication =
        (kc_ticket_publication_status)completion->publication_status;
    kc_ticket_terminal_cause cause =
        (kc_ticket_terminal_cause)completion->terminal_cause;
    int status = completion->status_code;
    if (completion->execution_status == KC_TICKET_EXECUTION_FAILED) {
        cause = KC_TICKET_CAUSE_FAULT;
        publication = KC_TICKET_PUBLICATION_NONE;
    } else if (ticket->cancel_requested) {
        cause = ticket->cancel_cause;
        publication = KC_TICKET_PUBLICATION_STALE;
        status = -ECANCELED;
    } else if (ticket->deadline_mode == KC_TICKET_DEADLINE_HARD_PUBLICATION &&
               deadline_expired(ticket, kc_port_monotonic_ns())) {
        cause = KC_TICKET_CAUSE_TIMED_OUT;
        publication = KC_TICKET_PUBLICATION_STALE;
        status = -ETIMEDOUT;
    }
    int won = publish_locked(ticket,
                             (kc_ticket_execution_status)completion->execution_status,
                             state, publication, cause, status, completion->result);
    return won;
}

int kc_ticket_complete(kc_ticket_t *ticket,
                       const kc_ticket_completion_v1 *completion)
{
    if (!ticket || !completion_valid(completion)) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    int won = complete_locked(ticket, completion);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return won;
}

static int cancel_locked(kc_ticket_t *ticket)
{
    if (!ticket->in_use || ticket->state == KC_TICKET_PUBLISHED) return 0;
    if (ticket->state == KC_TICKET_DISPATCHED) {
        if (ticket->cancel_requested) return 0;
        ticket->cancel_requested = 1;
        ticket->cancel_cause = KC_TICKET_CAUSE_CANCELED;
        return 1;
    }
    int won = publish_locked(ticket, KC_TICKET_NOT_DISPATCHED,
                             KC_TICKET_STATE_NONE, KC_TICKET_PUBLICATION_NONE,
                             KC_TICKET_CAUSE_CANCELED, -ECANCELED, NULL);
    return won;
}

int kc_ticket_cancel(kc_ticket_t *ticket)
{
    if (!ticket) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    int won = cancel_locked(ticket);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return won;
}

int kc_ticket_complete_id(kc_runtime_t *runtime, kc_ticket_id id,
                          const kc_ticket_completion_v1 *completion)
{
    if (!runtime || !completion_valid(completion)) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    kc_ticket_t *ticket = ticket_from_id_locked(runtime, id);
    int won = ticket ? complete_locked(ticket, completion) : -ESTALE;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return won;
}

int kc_ticket_cancel_id(kc_runtime_t *runtime, kc_ticket_id id)
{
    if (!runtime) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    kc_ticket_t *ticket = ticket_from_id_locked(runtime, id);
    int won = ticket ? cancel_locked(ticket) : -ESTALE;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return won;
}

kc_ticket_id kc_ticket_id_get(const kc_ticket_t *ticket)
{
    return ticket ? ticket->event.ticket : (kc_ticket_id){0};
}

int kc_ticket_snapshot_get(const kc_ticket_t *ticket,
                           kc_ticket_snapshot_v1 *out)
{
    if (!ticket || !out || out->size < sizeof(*out)) return -EINVAL;
    kc_runtime_t *runtime = ticket->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!ticket->in_use) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ENOENT;
    }
    *out = (kc_ticket_snapshot_v1){
        .size = sizeof(*out),
        .abi_version = KC_ABI_VERSION,
        .state = ticket->state,
        .cancel_requested = ticket->cancel_requested,
        .target_consumed = ticket->target_consumed,
        .event = ticket->event,
    };
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

kc_ticket_t *kc_ticket_runtime_dequeue_locked(kc_runtime_t *runtime)
{
    kc_ticket_t *ticket = runtime->completion_head;
    if (!ticket) return NULL;
    runtime->completion_head = ticket->completion_next;
    if (!runtime->completion_head) runtime->completion_tail = NULL;
    ticket->completion_next = NULL;
    ticket->delivery_queued = 0;
    if (runtime->completion_queued) runtime->completion_queued--;
    runtime->completion_running++;
    return ticket;
}

void kc_ticket_runtime_deliver(kc_ticket_t *ticket)
{
    if (!ticket) return;
    kc_runtime_t *runtime = ticket->runtime;
    ticket->callback(ticket->callback_context, &ticket->event);
    if (ticket->context_release) {
        ticket->context_release(ticket->callback_context);
        ticket->context_release = NULL;
        ticket->callback_context = NULL;
    }
    KC_MUTEX_LOCK(&runtime->mu);
    ticket->target_consumed = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_ticket_release(ticket); /* reserved terminal-delivery reference */
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->completion_running) runtime->completion_running--;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_ticket_runtime_stop_locked(kc_runtime_t *runtime)
{
    if (!runtime || !runtime->tickets) return;
    for (uint32_t slot = 0; slot < runtime->ticket_capacity; slot++) {
        kc_ticket_t *ticket = &runtime->tickets[slot];
        if (!ticket->in_use || ticket->state == KC_TICKET_PUBLISHED) continue;
        if (ticket->state == KC_TICKET_DISPATCHED) {
            ticket->cancel_requested = 1;
            ticket->cancel_cause = KC_TICKET_CAUSE_STOP;
            continue;
        }
        (void)publish_locked(ticket, KC_TICKET_NOT_DISPATCHED,
                             KC_TICKET_STATE_NONE, KC_TICKET_PUBLICATION_NONE,
                             KC_TICKET_CAUSE_STOP, -ECANCELED, NULL);
    }
}
