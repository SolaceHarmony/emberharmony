// SPDX-License-Identifier: BSD-3-Clause
#include "kc_deadline.h"

#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

enum kc_deadline_slot_state {
    KC_DEADLINE_SLOT_IDLE = 0,
    KC_DEADLINE_SLOT_WRITING = 1,
    KC_DEADLINE_SLOT_ARMED = 2,
    KC_DEADLINE_SLOT_FIRING = 3,
    KC_DEADLINE_SLOT_EVENT = 4,
    KC_DEADLINE_SLOT_CANCELING = 5,
    KC_DEADLINE_SLOT_CANCELED = 6,
    KC_DEADLINE_SLOT_RETIRING = 7,
};

enum { KC_DEADLINE_STATE_BITS = 8 };
#define KC_DEADLINE_GENERATION_MAX (UINT64_MAX >> KC_DEADLINE_STATE_BITS)

typedef struct kc_deadline_event_atomic {
    atomic_uint_fast64_t sequence;
    atomic_uint kind;
    atomic_uint_fast64_t scheduled_arm_generation;
    atomic_uint_fast64_t current_arm_generation;
    atomic_uint_fast64_t child_runtime_epoch;
    atomic_uint_fast64_t child_sequence;
    atomic_uint child_generation;
    atomic_uint child_kind;
    atomic_uint_fast64_t parent_runtime_epoch;
    atomic_uint_fast64_t parent_sequence;
    atomic_uint parent_generation;
    atomic_uint parent_kind;
    atomic_uint_fast64_t scope_generation;
    atomic_uint_fast64_t epoch;
    atomic_uint_fast64_t domain;
    atomic_uint_fast64_t team_generation;
} kc_deadline_event_atomic;

typedef struct kc_deadline_slot {
    struct kc_deadline_source *source;
    uint32_t index;
    atomic_uint_fast64_t control;
    atomic_uint_fast64_t due_ns;
    atomic_uint cancel_acked;
    kc_deadline_arm scheduled;
    kc_deadline_event_atomic event;
#if defined(__APPLE__)
    dispatch_source_t timer;
#endif
} kc_deadline_slot;

struct kc_deadline_source {
    uint32_t capacity;
    int manual;
    kc_deadline_notify_fn notify;
    void *context;
    kc_deadline_slot *slots;
    atomic_uint phase;
    atomic_uint closed;
    atomic_uint publishers;
    atomic_uint cancellation_started;
    atomic_uint cancellation_walk_done;
    atomic_uint cancellation_acks;
    atomic_uint active_handlers;
    atomic_uint_fast64_t published_events;
    atomic_uint_fast64_t stale_events;
    atomic_uint_fast64_t notifications;
    atomic_uint_fast64_t manual_now_ns;
#if defined(__APPLE__)
    dispatch_queue_t queue;
#endif
};

static uint64_t control_word(uint64_t generation, uint32_t state)
{
    return (generation << KC_DEADLINE_STATE_BITS) | state;
}

static uint64_t control_generation(uint64_t control)
{
    return control >> KC_DEADLINE_STATE_BITS;
}

static uint32_t control_state(uint64_t control)
{
    return (uint32_t)(control & ((UINT64_C(1) << KC_DEADLINE_STATE_BITS) - 1));
}

static int ticket_valid(const kc_ticket_id *ticket)
{
    return ticket && ticket->runtime_epoch != 0 && ticket->sequence != 0 &&
           ticket->generation != 0 && ticket->kind >= KC_TICKET_KIND_SESSION &&
           ticket->kind <= KC_TICKET_KIND_DEADLINE;
}

static void cancel_ack(kc_deadline_slot *slot);

/* STOPPED is the logical terminal edge: all native cancel acknowledgements
 * have arrived and the one cancellation-submission owner has finished issuing
 * every cancel. A handler may still be returning from the retained notify
 * call; that stack lifetime is administrative and is drained by destroy. */
static int publish_stopped(kc_deadline_source_t *source)
{
    if (!atomic_load_explicit(&source->cancellation_walk_done,
                              memory_order_acquire) ||
        atomic_load_explicit(&source->cancellation_acks,
                             memory_order_acquire) != source->capacity)
        return 0;
    unsigned expected = KC_DEADLINE_SOURCE_STOPPING;
    return atomic_compare_exchange_strong_explicit(
        &source->phase, &expected, KC_DEADLINE_SOURCE_STOPPED,
        memory_order_acq_rel, memory_order_acquire);
}

static uint64_t source_now(const kc_deadline_source_t *source)
{
#if defined(__APPLE__)
    if (!source->manual)
        return (uint64_t)dispatch_time(DISPATCH_TIME_NOW, 0);
#endif
    return atomic_load_explicit(&source->manual_now_ns,
                                memory_order_acquire);
}

static int source_due(const kc_deadline_source_t *source, uint64_t delay_ns,
                      uint64_t *out)
{
#if defined(__APPLE__)
    if (!source->manual) {
        dispatch_time_t due = dispatch_time(DISPATCH_TIME_NOW,
                                            (int64_t)delay_ns);
        if (due == DISPATCH_TIME_FOREVER) return -EOVERFLOW;
        *out = (uint64_t)due;
        return 0;
    }
#endif
    uint64_t now = source_now(source);
    if (delay_ns > UINT64_MAX - now) return -EOVERFLOW;
    *out = now + delay_ns;
    return 0;
}

static void event_atomic_init(kc_deadline_event_atomic *event)
{
    atomic_init(&event->sequence, 0);
    atomic_init(&event->kind, 0);
    atomic_init(&event->scheduled_arm_generation, 0);
    atomic_init(&event->current_arm_generation, 0);
    atomic_init(&event->child_runtime_epoch, 0);
    atomic_init(&event->child_sequence, 0);
    atomic_init(&event->child_generation, 0);
    atomic_init(&event->child_kind, 0);
    atomic_init(&event->parent_runtime_epoch, 0);
    atomic_init(&event->parent_sequence, 0);
    atomic_init(&event->parent_generation, 0);
    atomic_init(&event->parent_kind, 0);
    atomic_init(&event->scope_generation, 0);
    atomic_init(&event->epoch, 0);
    atomic_init(&event->domain, 0);
    atomic_init(&event->team_generation, 0);
}

static void publish_event_record(kc_deadline_slot *slot, uint32_t kind,
                                 uint64_t current_generation)
{
    kc_deadline_source_t *source = slot->source;
    kc_deadline_event_atomic *event = &slot->event;
    uint64_t sequence = atomic_load_explicit(&event->sequence,
                                             memory_order_relaxed) + 1;
    if (!sequence) sequence = 1;
    atomic_store_explicit(&event->kind, kind, memory_order_relaxed);
    atomic_store_explicit(&event->scheduled_arm_generation,
                          slot->scheduled.arm_generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->current_arm_generation, current_generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->child_runtime_epoch,
                          slot->scheduled.child.runtime_epoch,
                          memory_order_relaxed);
    atomic_store_explicit(&event->child_sequence,
                          slot->scheduled.child.sequence,
                          memory_order_relaxed);
    atomic_store_explicit(&event->child_generation,
                          slot->scheduled.child.generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->child_kind, slot->scheduled.child.kind,
                          memory_order_relaxed);
    atomic_store_explicit(&event->parent_runtime_epoch,
                          slot->scheduled.parent.runtime_epoch,
                          memory_order_relaxed);
    atomic_store_explicit(&event->parent_sequence,
                          slot->scheduled.parent.sequence,
                          memory_order_relaxed);
    atomic_store_explicit(&event->parent_generation,
                          slot->scheduled.parent.generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->parent_kind, slot->scheduled.parent.kind,
                          memory_order_relaxed);
    atomic_store_explicit(&event->scope_generation,
                          slot->scheduled.scope_generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->epoch, slot->scheduled.epoch,
                          memory_order_relaxed);
    atomic_store_explicit(&event->domain, slot->scheduled.domain,
                          memory_order_relaxed);
    atomic_store_explicit(&event->team_generation,
                          slot->scheduled.team_generation,
                          memory_order_relaxed);
    atomic_store_explicit(&event->sequence, sequence, memory_order_release);
    atomic_fetch_add_explicit(&source->published_events, 1,
                              memory_order_relaxed);
    if (kind == KC_DEADLINE_EVENT_STALE)
        atomic_fetch_add_explicit(&source->stale_events, 1,
                                  memory_order_relaxed);
}

/* Native timer callbacks carry no trusted arm identity. They only re-evaluate
 * the current slot predicate, so a callback queued for an older arm cannot
 * expire a later arm before that later arm's own due time. */
static int deliver_hint(kc_deadline_slot *slot)
{
    kc_deadline_source_t *source = slot->source;
    atomic_fetch_add_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
    uint64_t control = atomic_load_explicit(&slot->control,
                                            memory_order_acquire);
    if (control_state(control) != KC_DEADLINE_SLOT_ARMED) {
        atomic_fetch_sub_explicit(&source->active_handlers, 1,
                                  memory_order_acq_rel);
        return -ESTALE;
    }
    uint64_t due = atomic_load_explicit(&slot->due_ns,
                                        memory_order_acquire);
    if (source_now(source) < due) {
        atomic_fetch_sub_explicit(&source->active_handlers, 1,
                                  memory_order_acq_rel);
        return -EAGAIN;
    }

    uint64_t current = control_generation(control);
    uint64_t firing = control_word(current, KC_DEADLINE_SLOT_FIRING);
    if (!atomic_compare_exchange_strong_explicit(
            &slot->control, &control, firing,
            memory_order_acq_rel, memory_order_acquire)) {
        atomic_fetch_sub_explicit(&source->active_handlers, 1,
                                  memory_order_acq_rel);
        return -ESTALE;
    }
    publish_event_record(slot, KC_DEADLINE_EVENT_EXPIRED, current);
    atomic_store_explicit(&slot->control,
                          control_word(current, KC_DEADLINE_SLOT_EVENT),
                          memory_order_release);
    if (source->manual &&
        atomic_load_explicit(&source->phase, memory_order_acquire) ==
            KC_DEADLINE_SOURCE_STOPPING)
        cancel_ack(slot);
    kc_deadline_notify_fn notify = source->notify;
    void *context = source->context;
    atomic_fetch_add_explicit(&source->notifications, 1,
                              memory_order_relaxed);
    notify(context);
    atomic_fetch_sub_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
    return 0;
}

static void cancel_ack(kc_deadline_slot *slot)
{
    kc_deadline_source_t *source = slot->source;
    int publish = 0;
    atomic_fetch_add_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
    if (!atomic_exchange_explicit(&slot->cancel_acked, 1,
                                  memory_order_acq_rel)) {
        uint64_t control = atomic_load_explicit(&slot->control,
                                                memory_order_acquire);
        /* A terminal deadline record is reliable. Source stop closes new arm
         * admission, but it cannot overwrite an expiry that already won the
         * slot CAS. The consumer acknowledges that record into CANCELED. */
        if (control_state(control) != KC_DEADLINE_SLOT_EVENT) {
            atomic_store_explicit(
                &slot->control,
                control_word(control_generation(control),
                             KC_DEADLINE_SLOT_CANCELED),
                memory_order_release);
        }
        (void)atomic_fetch_add_explicit(&source->cancellation_acks, 1,
                                        memory_order_acq_rel);
        (void)publish_stopped(source);
        publish = 1;
    }
    kc_deadline_notify_fn notify = source->notify;
    void *context = source->context;
    if (publish)
        atomic_fetch_add_explicit(&source->notifications, 1,
                                  memory_order_relaxed);
    if (publish) notify(context);
    atomic_fetch_sub_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
}

#if defined(__APPLE__)
static void native_timer_handler(void *context)
{
    kc_deadline_slot *slot = context;
    (void)deliver_hint(slot);
}

static void native_cancel_handler(void *context)
{
    cancel_ack(context);
}

static void dispatch_release_object(dispatch_object_t object)
{
#if !OS_OBJECT_USE_OBJC
    dispatch_release(object);
#else
    (void)object;
#endif
}

static void drain_deadline_queue(void *context)
{
    (void)context;
}
#endif

static void start_cancellation(kc_deadline_source_t *source)
{
    unsigned expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &source->cancellation_started, &expected, 1,
            memory_order_acq_rel, memory_order_acquire)) return;
    /* The submission walk itself owns retained source storage. Manual cancel
     * acknowledgements can make STOPPED visible inline, and native cancel
     * handlers can run on the dispatch queue while this loop is still issuing
     * later cancels. Neither may make destruction legal before this owner
     * leaves. */
    atomic_fetch_add_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
    for (uint32_t index = 0; index < source->capacity; ++index) {
        kc_deadline_slot *slot = &source->slots[index];
        uint64_t control = atomic_load_explicit(&slot->control,
                                                memory_order_acquire);
        uint32_t state = control_state(control);
        /* EVENT is a reliable correlated record. Closing admission may cancel
         * its dispatch source, but it must not overwrite an expiry that has
         * already won the slot CAS. The owner still has to acknowledge that
         * exact sequence before the slot/storage can retire. */
        if (state != KC_DEADLINE_SLOT_FIRING &&
            state != KC_DEADLINE_SLOT_EVENT &&
            state != KC_DEADLINE_SLOT_CANCELED) {
            uint64_t canceled = control_word(control_generation(control),
                                             KC_DEADLINE_SLOT_CANCELING);
            (void)atomic_compare_exchange_strong_explicit(
                &slot->control, &control, canceled,
                memory_order_acq_rel, memory_order_acquire);
        }
#if defined(__APPLE__)
        if (!source->manual && slot->timer) {
            dispatch_source_cancel(slot->timer);
            continue;
        }
#endif
        if (state == KC_DEADLINE_SLOT_FIRING) continue;
        cancel_ack(slot);
    }
    atomic_store_explicit(&source->cancellation_walk_done, 1,
                          memory_order_release);
    int stopped = publish_stopped(source);
    kc_deadline_notify_fn notify = source->notify;
    void *context = source->context;
    if (stopped)
        atomic_fetch_add_explicit(&source->notifications, 1,
                                  memory_order_relaxed);
    if (stopped) notify(context);
    atomic_fetch_sub_explicit(&source->active_handlers, 1,
                              memory_order_acq_rel);
}

static int arm_enter(kc_deadline_source_t *source)
{
    if (atomic_load_explicit(&source->closed, memory_order_seq_cst) ||
        atomic_load_explicit(&source->phase, memory_order_acquire) !=
            KC_DEADLINE_SOURCE_OPEN) return -ECANCELED;
    atomic_fetch_add_explicit(&source->publishers, 1,
                              memory_order_seq_cst);
    if (!atomic_load_explicit(&source->closed, memory_order_seq_cst) &&
        atomic_load_explicit(&source->phase, memory_order_acquire) ==
            KC_DEADLINE_SOURCE_OPEN) return 0;
    unsigned prior = atomic_fetch_sub_explicit(&source->publishers, 1,
                                                memory_order_seq_cst);
    if (prior == 1) start_cancellation(source);
    return -ECANCELED;
}

static void arm_leave(kc_deadline_source_t *source)
{
    unsigned prior = atomic_fetch_sub_explicit(&source->publishers, 1,
                                                memory_order_seq_cst);
    if (prior == 1 &&
        atomic_load_explicit(&source->closed, memory_order_seq_cst))
        start_cancellation(source);
}

static int source_create(const kc_deadline_source_config *config, int manual,
                         kc_deadline_source_t **out)
{
    if (!out) return -EINVAL;
    *out = NULL;
    if (!config || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || config->capacity == 0 ||
        config->reserved != 0 || !config->notify ||
        sizeof(kc_deadline_slot) > SIZE_MAX / config->capacity) return -EINVAL;
    kc_deadline_source_t *source = calloc(1, sizeof(*source));
    if (!source) return -ENOMEM;
    source->slots = calloc(config->capacity, sizeof(*source->slots));
    if (!source->slots) {
        free(source);
        return -ENOMEM;
    }
    source->capacity = config->capacity;
    source->manual = manual;
    source->notify = config->notify;
    source->context = config->context;
    atomic_init(&source->phase, KC_DEADLINE_SOURCE_OPEN);
    atomic_init(&source->closed, 0);
    atomic_init(&source->publishers, 0);
    atomic_init(&source->cancellation_started, 0);
    atomic_init(&source->cancellation_walk_done, 0);
    atomic_init(&source->cancellation_acks, 0);
    atomic_init(&source->active_handlers, 0);
    atomic_init(&source->published_events, 0);
    atomic_init(&source->stale_events, 0);
    atomic_init(&source->notifications, 0);
    atomic_init(&source->manual_now_ns, 1);
    for (uint32_t index = 0; index < source->capacity; ++index) {
        kc_deadline_slot *slot = &source->slots[index];
        slot->source = source;
        slot->index = index;
        atomic_init(&slot->control, control_word(0, KC_DEADLINE_SLOT_IDLE));
        atomic_init(&slot->due_ns, UINT64_MAX);
        atomic_init(&slot->cancel_acked, 0);
        event_atomic_init(&slot->event);
    }
    if (!atomic_is_lock_free(&source->slots[0].control) ||
        !atomic_is_lock_free(&source->slots[0].due_ns) ||
        !atomic_is_lock_free(&source->slots[0].event.sequence) ||
        !atomic_is_lock_free(&source->cancellation_walk_done) ||
        !atomic_is_lock_free(&source->active_handlers)) {
        free(source->slots);
        free(source);
        return -ENOTSUP;
    }

#if defined(__APPLE__)
    if (!manual) {
        source->queue = dispatch_queue_create(
            "kcoro.deadline.monotonic", DISPATCH_QUEUE_SERIAL);
        if (!source->queue) {
            free(source->slots);
            free(source);
            return -ENOMEM;
        }
        for (uint32_t index = 0; index < source->capacity; ++index) {
            kc_deadline_slot *slot = &source->slots[index];
            slot->timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER,
                                                 0, 0, source->queue);
            if (!slot->timer) {
                for (uint32_t prior = 0; prior < index; ++prior)
                    dispatch_release_object(source->slots[prior].timer);
                dispatch_release_object(source->queue);
                free(source->slots);
                free(source);
                return -ENOMEM;
            }
            dispatch_set_context(slot->timer, slot);
            dispatch_source_set_event_handler_f(slot->timer,
                                                native_timer_handler);
            dispatch_source_set_cancel_handler_f(slot->timer,
                                                 native_cancel_handler);
        }
        for (uint32_t index = 0; index < source->capacity; ++index)
            dispatch_activate(source->slots[index].timer);
    }
#endif
    *out = source;
    return 0;
}

int kc_deadline_source_create(const kc_deadline_source_config *config,
                              kc_deadline_source_t **out)
{
    return source_create(config, 0, out);
}

int kc_deadline_source_create_manual_test(
    const kc_deadline_source_config *config, kc_deadline_source_t **out)
{
    return source_create(config, 1, out);
}

int kc_deadline_source_arm(kc_deadline_source_t *source,
                           const kc_deadline_arm_config *config,
                           kc_deadline_arm *out_arm)
{
    if (!source || !config || !out_arm ||
        config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || config->reserved != 0 ||
        config->slot >= source->capacity || !ticket_valid(&config->child) ||
        config->child.kind != KC_TICKET_KIND_DEADLINE ||
        !ticket_valid(&config->parent) || config->scope_generation == 0 ||
        config->epoch == 0 || config->domain == 0 ||
        config->team_generation == 0 || config->delay_ns > INT64_MAX)
        return -EINVAL;
#if !defined(__APPLE__)
    if (!source->manual) return -ENOTSUP;
#endif
    int admission = arm_enter(source);
    if (admission != 0) return admission;
    uint64_t due = 0;
    int due_status = source_due(source, config->delay_ns, &due);
    if (due_status != 0) {
        arm_leave(source);
        return due_status;
    }

    kc_deadline_slot *slot = &source->slots[config->slot];
    uint64_t control = atomic_load_explicit(&slot->control,
                                            memory_order_acquire);
    uint64_t generation = control_generation(control);
    if (control_state(control) != KC_DEADLINE_SLOT_IDLE) {
        arm_leave(source);
        return -EBUSY;
    }
    if (generation == KC_DEADLINE_GENERATION_MAX) {
        arm_leave(source);
        return -EOVERFLOW;
    }
    uint64_t writing = control_word(generation, KC_DEADLINE_SLOT_WRITING);
    if (!atomic_compare_exchange_strong_explicit(
            &slot->control, &control, writing,
            memory_order_acq_rel, memory_order_acquire)) {
        arm_leave(source);
        return -EBUSY;
    }

    uint64_t armed_generation = generation + 1;
    slot->scheduled = (kc_deadline_arm){
        .size = sizeof(slot->scheduled),
        .abi_version = KC_ABI_VERSION,
        .slot = config->slot,
        .reserved = 0,
        .arm_generation = armed_generation,
        .child = config->child,
        .parent = config->parent,
        .scope_generation = config->scope_generation,
        .epoch = config->epoch,
        .domain = config->domain,
        .team_generation = config->team_generation,
    };
    *out_arm = slot->scheduled;
    atomic_store_explicit(&slot->due_ns, due, memory_order_relaxed);
    atomic_store_explicit(
        &slot->control,
        control_word(armed_generation, KC_DEADLINE_SLOT_ARMED),
        memory_order_release);
#if defined(__APPLE__)
    if (!source->manual) {
        dispatch_source_set_timer(
            slot->timer, (dispatch_time_t)due,
            DISPATCH_TIME_FOREVER, 0);
    }
#endif
    arm_leave(source);
    return 0;
}

int kc_deadline_source_retire(kc_deadline_source_t *source, uint32_t slot,
                              uint64_t arm_generation)
{
    if (!source || slot >= source->capacity || arm_generation == 0)
        return -EINVAL;
    if (arm_generation == KC_DEADLINE_GENERATION_MAX) return -EOVERFLOW;
    int admission = arm_enter(source);
    if (admission != 0) return admission;

    kc_deadline_slot *record = &source->slots[slot];
    uint64_t observed = control_word(arm_generation,
                                     KC_DEADLINE_SLOT_ARMED);
    uint64_t retired_generation = arm_generation + 1;
    if (!atomic_compare_exchange_strong_explicit(
            &record->control, &observed,
            control_word(retired_generation, KC_DEADLINE_SLOT_RETIRING),
            memory_order_acq_rel, memory_order_acquire)) {
        uint64_t current_generation = control_generation(observed);
        uint32_t state = control_state(observed);
        arm_leave(source);
        if (current_generation != arm_generation) return -ESTALE;
        if (state == KC_DEADLINE_SLOT_FIRING ||
            state == KC_DEADLINE_SLOT_EVENT ||
            state == KC_DEADLINE_SLOT_IDLE)
            return KC_DEADLINE_RETIRE_EXPIRY_WON;
        if (state == KC_DEADLINE_SLOT_CANCELING ||
            state == KC_DEADLINE_SLOT_CANCELED)
            return -ECANCELED;
        return -EALREADY;
    }

    /* RETIRING prevents a successor arm from publishing a new due time before
     * the old native source and due predicate have both been disabled. */
#if defined(__APPLE__)
    if (!source->manual)
        dispatch_source_set_timer(record->timer, DISPATCH_TIME_FOREVER,
                                  DISPATCH_TIME_FOREVER, 0);
#endif
    atomic_store_explicit(&record->due_ns, UINT64_MAX,
                          memory_order_release);
    atomic_store_explicit(
        &record->control,
        control_word(retired_generation, KC_DEADLINE_SLOT_IDLE),
        memory_order_release);
    arm_leave(source);
    return KC_DEADLINE_RETIRE_RETIRED;
}

int kc_deadline_source_disarm(kc_deadline_source_t *source, uint32_t slot,
                              uint64_t arm_generation)
{
    if (!source || slot >= source->capacity || arm_generation == 0)
        return -EINVAL;
    int admission = arm_enter(source);
    if (admission != 0) return admission;
    kc_deadline_slot *record = &source->slots[slot];
    uint64_t expected = control_word(arm_generation,
                                     KC_DEADLINE_SLOT_ARMED);
    if (arm_generation == KC_DEADLINE_GENERATION_MAX) {
        arm_leave(source);
        return -EOVERFLOW;
    }
    if (!atomic_compare_exchange_strong_explicit(
            &record->control, &expected,
            control_word(arm_generation + 1, KC_DEADLINE_SLOT_FIRING),
            memory_order_acq_rel, memory_order_acquire)) {
        arm_leave(source);
        return control_generation(expected) != arm_generation
                   ? -ESTALE : -EALREADY;
    }
#if defined(__APPLE__)
    if (!source->manual)
        dispatch_source_set_timer(record->timer, DISPATCH_TIME_FOREVER,
                                  DISPATCH_TIME_FOREVER, 0);
#endif
    atomic_store_explicit(&record->due_ns, UINT64_MAX,
                          memory_order_release);
    publish_event_record(record, KC_DEADLINE_EVENT_STALE,
                         arm_generation + 1);
    atomic_store_explicit(
        &record->control,
        control_word(arm_generation + 1, KC_DEADLINE_SLOT_EVENT),
        memory_order_release);
    kc_deadline_notify_fn notify = source->notify;
    void *context = source->context;
    atomic_fetch_add_explicit(&source->notifications, 1,
                              memory_order_relaxed);
    notify(context);
    arm_leave(source);
    return 0;
}

int kc_deadline_source_event_get(const kc_deadline_source_t *source,
                                 uint32_t slot, kc_deadline_event *out)
{
    if (!source || !out || out->size < sizeof(*out) ||
        slot >= source->capacity) return -EINVAL;
    const kc_deadline_slot *record = &source->slots[slot];
    uint64_t control = atomic_load_explicit(&record->control,
                                            memory_order_acquire);
    if (control_state(control) != KC_DEADLINE_SLOT_EVENT) return -EAGAIN;
    const kc_deadline_event_atomic *event = &record->event;
    uint64_t sequence = atomic_load_explicit(&event->sequence,
                                             memory_order_acquire);
    *out = (kc_deadline_event){
        .size = sizeof(*out),
        .abi_version = KC_ABI_VERSION,
        .slot = slot,
        .kind = atomic_load_explicit(&event->kind, memory_order_relaxed),
        .sequence = sequence,
        .scheduled_arm_generation = atomic_load_explicit(
            &event->scheduled_arm_generation, memory_order_relaxed),
        .current_arm_generation = atomic_load_explicit(
            &event->current_arm_generation, memory_order_relaxed),
        .child = {
            .runtime_epoch = atomic_load_explicit(
                &event->child_runtime_epoch, memory_order_relaxed),
            .sequence = atomic_load_explicit(
                &event->child_sequence, memory_order_relaxed),
            .generation = atomic_load_explicit(
                &event->child_generation, memory_order_relaxed),
            .kind = atomic_load_explicit(&event->child_kind,
                                         memory_order_relaxed),
        },
        .parent = {
            .runtime_epoch = atomic_load_explicit(
                &event->parent_runtime_epoch, memory_order_relaxed),
            .sequence = atomic_load_explicit(
                &event->parent_sequence, memory_order_relaxed),
            .generation = atomic_load_explicit(
                &event->parent_generation, memory_order_relaxed),
            .kind = atomic_load_explicit(&event->parent_kind,
                                         memory_order_relaxed),
        },
        .scope_generation = atomic_load_explicit(
            &event->scope_generation, memory_order_relaxed),
        .epoch = atomic_load_explicit(&event->epoch, memory_order_relaxed),
        .domain = atomic_load_explicit(&event->domain,
                                       memory_order_relaxed),
        .team_generation = atomic_load_explicit(
            &event->team_generation, memory_order_relaxed),
    };
    if (atomic_load_explicit(&event->sequence, memory_order_acquire) !=
        sequence) return -EAGAIN;
    return 0;
}

static int event_identity_equal(const kc_deadline_event *left,
                                const kc_deadline_event *right)
{
    return left->slot == right->slot && left->kind == right->kind &&
           left->sequence == right->sequence &&
           left->scheduled_arm_generation ==
               right->scheduled_arm_generation &&
           left->current_arm_generation == right->current_arm_generation &&
           left->scope_generation == right->scope_generation &&
           left->epoch == right->epoch && left->domain == right->domain &&
           left->team_generation == right->team_generation &&
           left->child.runtime_epoch == right->child.runtime_epoch &&
           left->child.sequence == right->child.sequence &&
           left->child.generation == right->child.generation &&
           left->child.kind == right->child.kind &&
           left->parent.runtime_epoch == right->parent.runtime_epoch &&
           left->parent.sequence == right->parent.sequence &&
           left->parent.generation == right->parent.generation &&
           left->parent.kind == right->parent.kind;
}

int kc_deadline_source_event_ack(kc_deadline_source_t *source,
                                 const kc_deadline_event *event)
{
    if (!source || !event || event->size < sizeof(*event) ||
        event->abi_version != KC_ABI_VERSION ||
        event->slot >= source->capacity || event->sequence == 0 ||
        event->current_arm_generation == 0) return -EINVAL;
    kc_deadline_slot *record = &source->slots[event->slot];
    kc_deadline_event stored = {
        .size = sizeof(stored),
        .abi_version = KC_ABI_VERSION,
    };
    int observed = kc_deadline_source_event_get(source, event->slot, &stored);
    if (observed != 0) return observed == -EAGAIN ? -ESTALE : observed;
    if (!event_identity_equal(&stored, event)) return -ESTALE;
    uint32_t target = atomic_load_explicit(&source->phase,
                                           memory_order_acquire) ==
                              KC_DEADLINE_SOURCE_OPEN
                          ? KC_DEADLINE_SLOT_IDLE
                          : KC_DEADLINE_SLOT_CANCELED;
    uint64_t control = control_word(event->current_arm_generation,
                                    KC_DEADLINE_SLOT_EVENT);
    if (!atomic_compare_exchange_strong_explicit(
            &record->control, &control,
            control_word(event->current_arm_generation, target),
            memory_order_acq_rel, memory_order_acquire))
        return control_generation(control) != event->current_arm_generation
                   ? -ESTALE : -EALREADY;
    return 0;
}

int kc_deadline_source_advance_manual_test(kc_deadline_source_t *source,
                                           uint64_t elapsed_ns)
{
    if (!source || !source->manual) return -EINVAL;
    uint64_t now = atomic_load_explicit(&source->manual_now_ns,
                                        memory_order_acquire);
    for (;;) {
        if (elapsed_ns > UINT64_MAX - now) return -EOVERFLOW;
        if (atomic_compare_exchange_weak_explicit(
                &source->manual_now_ns, &now, now + elapsed_ns,
                memory_order_acq_rel, memory_order_acquire)) return 0;
    }
}

int kc_deadline_source_fire_manual_test(kc_deadline_source_t *source,
                                        uint32_t slot)
{
    if (!source || !source->manual || slot >= source->capacity)
        return -EINVAL;
    return deliver_hint(&source->slots[slot]);
}

void kc_deadline_source_request_stop(kc_deadline_source_t *source)
{
    if (!source) return;
    atomic_store_explicit(&source->closed, 1, memory_order_seq_cst);
    unsigned expected = KC_DEADLINE_SOURCE_OPEN;
    if (!atomic_compare_exchange_strong_explicit(
            &source->phase, &expected, KC_DEADLINE_SOURCE_STOPPING,
            memory_order_acq_rel, memory_order_acquire)) return;
    if (atomic_load_explicit(&source->publishers,
                             memory_order_seq_cst) == 0)
        start_cancellation(source);
}

int kc_deadline_source_snapshot_get(const kc_deadline_source_t *source,
                                    kc_deadline_source_snapshot *out)
{
    if (!source || !out || out->size < sizeof(*out)) return -EINVAL;
    uint32_t idle = 0;
    uint32_t armed = 0;
    uint32_t events = 0;
    for (uint32_t index = 0; index < source->capacity; ++index) {
        uint32_t state = control_state(atomic_load_explicit(
            &source->slots[index].control, memory_order_acquire));
        idle += state == KC_DEADLINE_SLOT_IDLE;
        armed += state == KC_DEADLINE_SLOT_ARMED;
        events += state == KC_DEADLINE_SLOT_EVENT;
    }
    *out = (kc_deadline_source_snapshot){
        .size = sizeof(*out),
        .abi_version = KC_ABI_VERSION,
        .capacity = source->capacity,
        .phase = atomic_load_explicit(&source->phase, memory_order_acquire),
        .idle = idle,
        .armed = armed,
        .pending_events = events,
        .reserved = 0,
        .published_events = atomic_load_explicit(
            &source->published_events, memory_order_acquire),
        .stale_events = atomic_load_explicit(&source->stale_events,
                                             memory_order_acquire),
        .notifications = atomic_load_explicit(&source->notifications,
                                               memory_order_acquire),
        .cancellation_acks = atomic_load_explicit(
            &source->cancellation_acks, memory_order_acquire),
        .active_handlers = atomic_load_explicit(
            &source->active_handlers, memory_order_acquire),
    };
    return 0;
}

int kc_deadline_source_destroy(kc_deadline_source_t *source)
{
    if (!source) return 0;
    if (atomic_load_explicit(&source->phase, memory_order_acquire) !=
            KC_DEADLINE_SOURCE_STOPPED ||
        atomic_load_explicit(&source->cancellation_acks,
                             memory_order_acquire) != source->capacity ||
        !atomic_load_explicit(&source->cancellation_walk_done,
                              memory_order_acquire) ||
        atomic_load_explicit(&source->publishers,
                             memory_order_acquire) != 0) return -EBUSY;
    for (uint32_t index = 0; index < source->capacity; ++index) {
        uint64_t control = atomic_load_explicit(&source->slots[index].control,
                                                memory_order_acquire);
        if (control_state(control) == KC_DEADLINE_SLOT_EVENT) return -EBUSY;
    }
#if defined(__APPLE__)
    if (!source->manual) {
        /* STOPPED is published from the final cancel handler before that
         * handler returns. Administrative destruction drains the source's
         * serial queue so no caller has to poll a handler counter and no final
         * zero-transition wake can be lost. Destroy is forbidden from a
         * deadline handler by the public callback contract. */
        dispatch_sync_f(source->queue, NULL, drain_deadline_queue);
        if (atomic_load_explicit(&source->active_handlers,
                                 memory_order_acquire) != 0)
            return -EBUSY;
        for (uint32_t index = 0; index < source->capacity; ++index)
            dispatch_release_object(source->slots[index].timer);
        dispatch_release_object(source->queue);
    }
#endif
    if (atomic_load_explicit(&source->active_handlers,
                             memory_order_acquire) != 0)
        return -EBUSY;
    free(source->slots);
    free(source);
    return 0;
}
