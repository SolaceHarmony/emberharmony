// SPDX-License-Identifier: BSD-3-Clause
#include "kc_fixed_scope.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

enum kc_scope_child_state {
    KC_SCOPE_SLOT_CONFIGURED = 1,
    KC_SCOPE_SLOT_ACTIVE = 2,
    KC_SCOPE_SLOT_CANCELLING = 3,
    KC_SCOPE_SLOT_TERMINAL = 4,
};

typedef struct kc_atomic_ticket {
    atomic_uint_fast64_t runtime_epoch;
    atomic_uint_fast64_t sequence;
    atomic_uint generation;
    atomic_uint kind;
} kc_atomic_ticket;

typedef struct kc_fixed_scope_child {
    atomic_uint_fast64_t word;
    kc_atomic_ticket ticket;
    uint32_t child_class;
    kc_scope_child_cancel_fn cancel;
    void *context;
} kc_fixed_scope_child;

struct kc_fixed_scope {
    uint32_t capacity;
    uint32_t children;
    uint32_t functional_children;
    uint32_t telemetry_children;
    atomic_uint_fast64_t generation;
    kc_atomic_ticket parent;
    kc_scope_ready_fn ready;
    void *context;
    kc_fixed_scope_child *slots;
    atomic_uint_fast64_t control;
    atomic_uint_fast64_t publishers;
    atomic_uint active_ready;
    atomic_uint terminal_children;
    atomic_uint cause;
    atomic_uint cause_slot;
    atomic_uint ready_edges;
    atomic_uint propagators;
};

enum { KC_SCOPE_PHASE_BITS = 8 };
#define KC_SCOPE_PUBLISHERS_CLOSED UINT64_C(1)
#define KC_SCOPE_PUBLISHER UINT64_C(2)

static void publish_ready(kc_fixed_scope_t *scope);
static void claim_terminal(kc_fixed_scope_t *scope);
static void publisher_leave(kc_fixed_scope_t *scope);

static int publisher_enter(kc_fixed_scope_t *scope)
{
    /* One RMW is the admission linearization point. The low bit is closure and
     * every publisher owns one of the upper-bit units. Unlike a CAS loop this
     * has bounded callback cost under contention: an entrant is ordered wholly
     * before or wholly after close in this atomic's modification order. */
    uint64_t gate = atomic_fetch_add_explicit(
        &scope->publishers, KC_SCOPE_PUBLISHER, memory_order_acq_rel);
    if (!(gate & KC_SCOPE_PUBLISHERS_CLOSED)) return 0;
    publisher_leave(scope);
    return -ECANCELED;
}

static void publisher_leave(kc_fixed_scope_t *scope)
{
    /* Keep destruction closed across the last-count handoff. Once the count is
     * decremented to CLOSED, another owner may observe an otherwise-terminal
     * scope; this hold covers the decision and any resulting ready publication. */
    atomic_fetch_add_explicit(&scope->active_ready, 1,
                              memory_order_acq_rel);
    uint64_t prior = atomic_fetch_sub_explicit(
        &scope->publishers, KC_SCOPE_PUBLISHER, memory_order_acq_rel);
    if ((prior & ~KC_SCOPE_PUBLISHERS_CLOSED) == 0) abort();
    if ((prior & KC_SCOPE_PUBLISHERS_CLOSED) &&
        (prior >> 1) == 1) {
        publish_ready(scope);
    } else if (!(prior & KC_SCOPE_PUBLISHERS_CLOSED) &&
               (prior >> 1) == 1 &&
               atomic_load_explicit(&scope->terminal_children,
                                    memory_order_acquire) == scope->children) {
        /* Every child/cancel transition owns this same admission counter. The
         * 1 -> 0 publisher is therefore the convergence owner and has acquired
         * every earlier publisher's state writes through the RMW chain. This
         * avoids a two-atomic store-buffering hole where a cancelling parent
         * could miss the final child while that child still observed an active
         * propagator, leaving a complete scope stranded in CANCELLING. */
        claim_terminal(scope);
    }
    unsigned active = atomic_fetch_sub_explicit(&scope->active_ready, 1,
                                                 memory_order_acq_rel);
    if (active == 0) abort();
}

static int publisher_return(kc_fixed_scope_t *scope, int status)
{
    publisher_leave(scope);
    return status;
}

static uint64_t scope_word(uint32_t cycle, uint32_t phase)
{
    return ((uint64_t)cycle << KC_SCOPE_PHASE_BITS) | phase;
}

static uint32_t scope_cycle(uint64_t control)
{
    return (uint32_t)(control >> KC_SCOPE_PHASE_BITS);
}

static uint32_t scope_phase(uint64_t control)
{
    return (uint32_t)(control & UINT64_C(0xff));
}

static int ticket_valid(const kc_ticket_id *ticket)
{
    return ticket && ticket->runtime_epoch != 0 && ticket->sequence != 0 &&
           ticket->generation != 0 && ticket->kind >= KC_TICKET_KIND_SESSION &&
           ticket->kind <= KC_TICKET_KIND_DEADLINE;
}

static int ticket_equal(const kc_ticket_id *left, const kc_ticket_id *right)
{
    return left->runtime_epoch == right->runtime_epoch &&
           left->sequence == right->sequence &&
           left->generation == right->generation &&
           left->kind == right->kind;
}

static void atomic_ticket_init(kc_atomic_ticket *target)
{
    atomic_init(&target->runtime_epoch, 0);
    atomic_init(&target->sequence, 0);
    atomic_init(&target->generation, 0);
    atomic_init(&target->kind, 0);
}

static void atomic_ticket_store(kc_atomic_ticket *target,
                                const kc_ticket_id *ticket)
{
    atomic_store_explicit(&target->runtime_epoch, ticket->runtime_epoch,
                          memory_order_relaxed);
    atomic_store_explicit(&target->sequence, ticket->sequence,
                          memory_order_relaxed);
    atomic_store_explicit(&target->generation, ticket->generation,
                          memory_order_relaxed);
    atomic_store_explicit(&target->kind, ticket->kind, memory_order_relaxed);
}

static kc_ticket_id atomic_ticket_load(const kc_atomic_ticket *source)
{
    return (kc_ticket_id){
        .runtime_epoch = atomic_load_explicit(&source->runtime_epoch,
                                              memory_order_relaxed),
        .sequence = atomic_load_explicit(&source->sequence,
                                         memory_order_relaxed),
        .generation = atomic_load_explicit(&source->generation,
                                            memory_order_relaxed),
        .kind = atomic_load_explicit(&source->kind, memory_order_relaxed),
    };
}

static uint64_t child_word(uint32_t generation, uint32_t state,
                           uint32_t cause)
{
    return ((uint64_t)generation << 32) | ((uint64_t)cause << 8) | state;
}

static uint32_t child_state(uint64_t word)
{
    return (uint32_t)(word & UINT64_C(0xff));
}

static uint32_t child_generation(uint64_t word)
{
    return (uint32_t)(word >> 32);
}

static kc_scope_child_lease child_lease(const kc_fixed_scope_t *scope,
                                        uint32_t slot, uint64_t generation,
                                        uint32_t lease_generation)
{
    const kc_fixed_scope_child *child = &scope->slots[slot];
    return (kc_scope_child_lease){
        .slot = slot,
        .child_class = child->child_class,
        .scope_generation = generation,
        .child_generation = lease_generation,
        .parent = atomic_ticket_load(&scope->parent),
        .child = atomic_ticket_load(&child->ticket),
    };
}

/* FINALIZING closes admission. TERMINAL and its edge are withheld until every
 * child/cancel publisher admitted before that close has left. */
static void publish_ready(kc_fixed_scope_t *scope)
{
    if (atomic_load_explicit(&scope->publishers, memory_order_acquire) !=
        KC_SCOPE_PUBLISHERS_CLOSED) return;
    uint64_t expected = atomic_load_explicit(&scope->control,
                                              memory_order_acquire);
    uint32_t cycle = scope_cycle(expected);
    if (scope_phase(expected) != KC_FIXED_SCOPE_FINALIZING) return;

    unsigned cause = atomic_load_explicit(&scope->cause,
                                          memory_order_acquire);
    uint64_t generation = atomic_load_explicit(&scope->generation,
                                                memory_order_acquire);
    kc_scope_ready_fn ready = scope->ready;
    void *context = scope->context;
    atomic_fetch_add_explicit(&scope->active_ready, 1,
                              memory_order_acq_rel);
    if (!atomic_compare_exchange_strong_explicit(
            &scope->control, &expected,
            scope_word(cycle, KC_FIXED_SCOPE_TERMINAL),
            memory_order_release, memory_order_acquire)) {
        atomic_fetch_sub_explicit(&scope->active_ready, 1,
                                  memory_order_acq_rel);
        return;
    }
    atomic_fetch_add_explicit(&scope->ready_edges, 1, memory_order_relaxed);
    ready(context, generation, cause);
    unsigned prior = atomic_fetch_sub_explicit(&scope->active_ready, 1,
                                                memory_order_acq_rel);
    if (prior == 0) abort();
}

static void claim_terminal(kc_fixed_scope_t *scope)
{
    uint64_t expected = atomic_load_explicit(&scope->control,
                                              memory_order_acquire);
    uint32_t cycle = scope_cycle(expected);
    uint32_t phase = scope_phase(expected);
    if (phase != KC_FIXED_SCOPE_RUNNING &&
        phase != KC_FIXED_SCOPE_CANCELLING) return;
    if (!atomic_compare_exchange_strong_explicit(
            &scope->control, &expected,
            scope_word(cycle, KC_FIXED_SCOPE_FINALIZING),
            memory_order_acq_rel, memory_order_acquire)) return;
    unsigned cause = atomic_load_explicit(&scope->cause,
                                          memory_order_acquire);
    if (cause == KC_SCOPE_CAUSE_NONE) {
        unsigned none = KC_SCOPE_CAUSE_NONE;
        atomic_compare_exchange_strong_explicit(
            &scope->cause, &none, KC_SCOPE_CAUSE_COMPLETE,
            memory_order_release, memory_order_acquire);
    }
    uint64_t gate = atomic_fetch_or_explicit(
        &scope->publishers, KC_SCOPE_PUBLISHERS_CLOSED,
        memory_order_acq_rel);
    if ((gate >> 1) == 0) publish_ready(scope);
}

static void record_terminals(kc_fixed_scope_t *scope, uint32_t count)
{
    if (!count) return;
    unsigned prior = atomic_fetch_add_explicit(&scope->terminal_children,
                                                count, memory_order_acq_rel);
    if (prior + count == scope->children &&
        atomic_load_explicit(&scope->propagators,
                             memory_order_acquire) == 0) {
        claim_terminal(scope);
        return;
    }
}

static uint32_t propagate_cancel(kc_fixed_scope_t *scope, uint32_t cause)
{
    uint32_t published = 0;
    uint64_t generation = atomic_load_explicit(&scope->generation,
                                                memory_order_acquire);
    for (uint32_t slot = 0; slot < scope->children; ++slot) {
        kc_fixed_scope_child *child = &scope->slots[slot];
        uint64_t expected = atomic_load_explicit(&child->word,
                                                 memory_order_acquire);
        uint32_t lease_generation = child_generation(expected);
        if (child_state(expected) != KC_SCOPE_SLOT_ACTIVE) continue;
        if (atomic_compare_exchange_strong_explicit(
                &child->word, &expected,
                child_word(lease_generation, KC_SCOPE_SLOT_CANCELLING,
                           cause),
                memory_order_acq_rel, memory_order_acquire)) {
            kc_scope_child_lease lease = child_lease(
                scope, slot, generation, lease_generation);
            ++published;
            child->cancel(child->context, &lease, cause);
        }
    }
    return published;
}

static int begin_closing(kc_fixed_scope_t *scope, uint32_t cycle,
                         uint32_t cause, uint32_t cause_slot)
{
    uint64_t expected = scope_word(cycle, KC_FIXED_SCOPE_RUNNING);
    if (!atomic_compare_exchange_strong_explicit(
            &scope->control, &expected,
            scope_word(cycle, KC_FIXED_SCOPE_CLOSING),
            memory_order_acq_rel, memory_order_acquire)) return 0;
    atomic_store_explicit(&scope->cause, cause, memory_order_release);
    atomic_store_explicit(&scope->cause_slot, cause_slot,
                          memory_order_release);
    /* Keep one propagation owner live until CANCELLING is visible. If the
     * final child retires while cancellation callbacks are running, it sees a
     * non-zero propagator and leaves terminal publication to this owner. If it
     * retires after the release below, it sees CANCELLING and publishes
     * itself. This closes the former gap where both sides could observe the
     * other's old atomic and leave a fully-terminal scope stranded in
     * CANCELLING. */
    atomic_fetch_add_explicit(&scope->propagators, 1, memory_order_acq_rel);
    (void)propagate_cancel(scope, cause == KC_SCOPE_CAUSE_FAULT
                                     ? KC_SCOPE_CAUSE_CANCELLED : cause);
    atomic_store_explicit(&scope->control,
                          scope_word(cycle, KC_FIXED_SCOPE_CANCELLING),
                          memory_order_release);
    unsigned prior = atomic_fetch_sub_explicit(&scope->propagators, 1,
                                                memory_order_acq_rel);
    if (prior == 0) abort();
    if (prior == 1 &&
        atomic_load_explicit(&scope->terminal_children,
                             memory_order_acquire) == scope->children) {
        claim_terminal(scope);
        return 1;
    }
    return 1;
}

int kc_fixed_scope_create(const kc_fixed_scope_config *config,
                          kc_fixed_scope_t **out)
{
    if (!out) return -EINVAL;
    *out = NULL;
    if (!config || config->child_capacity == 0 || !config->ready ||
        sizeof(kc_fixed_scope_child) > SIZE_MAX / config->child_capacity)
        return -EINVAL;

    kc_fixed_scope_t *scope = calloc(1, sizeof(*scope));
    if (!scope) return -ENOMEM;
    scope->slots = calloc(config->child_capacity, sizeof(*scope->slots));
    if (!scope->slots) {
        free(scope);
        return -ENOMEM;
    }
    scope->capacity = config->child_capacity;
    scope->ready = config->ready;
    scope->context = config->context;
    atomic_init(&scope->generation, 0);
    atomic_ticket_init(&scope->parent);
    atomic_init(&scope->control, scope_word(0, KC_FIXED_SCOPE_SETUP));
    atomic_init(&scope->publishers, KC_SCOPE_PUBLISHERS_CLOSED);
    atomic_init(&scope->active_ready, 0);
    atomic_init(&scope->terminal_children, 0);
    atomic_init(&scope->cause, KC_SCOPE_CAUSE_NONE);
    atomic_init(&scope->cause_slot, UINT32_MAX);
    atomic_init(&scope->ready_edges, 0);
    atomic_init(&scope->propagators, 0);
    for (uint32_t slot = 0; slot < scope->capacity; ++slot) {
        atomic_init(&scope->slots[slot].word,
                    child_word(0, KC_SCOPE_SLOT_CONFIGURED,
                               KC_SCOPE_CAUSE_NONE));
        atomic_ticket_init(&scope->slots[slot].ticket);
    }
    if (!atomic_is_lock_free(&scope->generation) ||
        !atomic_is_lock_free(&scope->control) ||
        !atomic_is_lock_free(&scope->publishers) ||
        !atomic_is_lock_free(&scope->active_ready) ||
        !atomic_is_lock_free(&scope->terminal_children) ||
        !atomic_is_lock_free(&scope->slots[0].word) ||
        !atomic_is_lock_free(&scope->slots[0].ticket.runtime_epoch)) {
        free(scope->slots);
        free(scope);
        return -ENOTSUP;
    }
    *out = scope;
    return 0;
}

int kc_fixed_scope_add_role(kc_fixed_scope_t *scope,
                            const kc_scope_child_config *config,
                            uint32_t *out_slot)
{
    if (!scope || !config || !out_slot || !config->cancel ||
        (config->child_class != KC_SCOPE_CHILD_FUNCTIONAL &&
         config->child_class != KC_SCOPE_CHILD_TELEMETRY)) return -EINVAL;
    if (scope_phase(atomic_load_explicit(&scope->control,
                                         memory_order_acquire)) !=
        KC_FIXED_SCOPE_SETUP) return -ECANCELED;
    if (scope->children == scope->capacity) return -ENOSPC;

    uint32_t slot = scope->children++;
    kc_fixed_scope_child *child = &scope->slots[slot];
    child->child_class = config->child_class;
    child->cancel = config->cancel;
    child->context = config->context;
    if (config->child_class == KC_SCOPE_CHILD_FUNCTIONAL)
        scope->functional_children++;
    else
        scope->telemetry_children++;
    *out_slot = slot;
    return 0;
}

int kc_fixed_scope_seal(kc_fixed_scope_t *scope)
{
    if (!scope) return -EINVAL;
    uint64_t expected = scope_word(0, KC_FIXED_SCOPE_SETUP);
    if (!atomic_compare_exchange_strong_explicit(
            &scope->control, &expected,
            scope_word(0, KC_FIXED_SCOPE_SEALED),
            memory_order_release, memory_order_acquire))
        return scope_phase(expected) == KC_FIXED_SCOPE_SEALED
                   ? 0 : -ECANCELED;
    return 0;
}

int kc_fixed_scope_cycle_begin(
    kc_fixed_scope_t *scope, const kc_fixed_scope_cycle_config *config,
    kc_scope_child_lease *out_leases, size_t lease_capacity)
{
    if (!scope || !config || config->generation == 0 ||
        !ticket_valid(&config->parent) ||
        config->child_count != scope->children ||
        lease_capacity < scope->children ||
        (scope->children && (!config->child_tickets || !out_leases)))
        return -EINVAL;
    uint64_t prior_generation = atomic_load_explicit(
        &scope->generation, memory_order_acquire);
    if (config->generation <= prior_generation) return -ESTALE;
    for (uint32_t slot = 0; slot < scope->children; ++slot) {
        if (!ticket_valid(&config->child_tickets[slot])) return -EINVAL;
        for (uint32_t prior = 0; prior < slot; ++prior) {
            if (ticket_equal(&config->child_tickets[slot],
                             &config->child_tickets[prior])) return -EEXIST;
        }
    }

    uint64_t control = atomic_load_explicit(&scope->control,
                                             memory_order_acquire);
    uint32_t phase = scope_phase(control);
    uint32_t prior_cycle = scope_cycle(control);
    if (phase != KC_FIXED_SCOPE_SEALED && phase != KC_FIXED_SCOPE_TERMINAL)
        return -EBUSY;
    if (atomic_load_explicit(&scope->publishers, memory_order_acquire) !=
        KC_SCOPE_PUBLISHERS_CLOSED) return -EBUSY;
    if (atomic_load_explicit(&scope->active_ready,
                             memory_order_acquire) != 0) return -EBUSY;
    if (prior_cycle == UINT32_MAX) return -EOVERFLOW;
    if (!atomic_compare_exchange_strong_explicit(
            &scope->control, &control,
            scope_word(prior_cycle, KC_FIXED_SCOPE_STARTING),
            memory_order_acq_rel, memory_order_acquire)) return -EBUSY;
    if (atomic_load_explicit(&scope->generation, memory_order_acquire) !=
        prior_generation) {
        atomic_store_explicit(&scope->control,
                              scope_word(prior_cycle, phase),
                              memory_order_release);
        return -ESTALE;
    }
    uint32_t cycle = prior_cycle + 1;

    atomic_store_explicit(&scope->terminal_children, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&scope->cause, KC_SCOPE_CAUSE_NONE,
                          memory_order_relaxed);
    atomic_store_explicit(&scope->cause_slot, UINT32_MAX,
                          memory_order_relaxed);
    atomic_store_explicit(&scope->propagators, 0, memory_order_relaxed);
    atomic_ticket_store(&scope->parent, &config->parent);
    for (uint32_t slot = 0; slot < scope->children; ++slot) {
        kc_fixed_scope_child *child = &scope->slots[slot];
        atomic_ticket_store(&child->ticket, &config->child_tickets[slot]);
        atomic_store_explicit(
            &child->word,
            child_word(cycle, KC_SCOPE_SLOT_ACTIVE,
                       KC_SCOPE_CAUSE_NONE),
            memory_order_relaxed);
        out_leases[slot] = (kc_scope_child_lease){
            .slot = slot,
            .child_class = child->child_class,
            .scope_generation = config->generation,
            .child_generation = cycle,
            .parent = config->parent,
            .child = config->child_tickets[slot],
        };
    }
    atomic_store_explicit(&scope->generation, config->generation,
                          memory_order_release);
    atomic_store_explicit(&scope->control,
                          scope_word(cycle, KC_FIXED_SCOPE_RUNNING),
                          memory_order_release);
    atomic_store_explicit(&scope->publishers, 0, memory_order_release);
    if (!scope->children) claim_terminal(scope);
    return 0;
}

int kc_fixed_scope_child_terminal(kc_fixed_scope_t *scope,
                                  const kc_scope_child_lease *lease,
                                  uint32_t cause)
{
    if (!scope || !lease ||
        cause < KC_SCOPE_CAUSE_COMPLETE || cause > KC_SCOPE_CAUSE_STOPPED)
        return -EINVAL;
    int admission = publisher_enter(scope);
    if (admission != 0) return admission;
    uint64_t generation = atomic_load_explicit(&scope->generation,
                                                memory_order_acquire);
    if (lease->scope_generation != generation)
        return publisher_return(scope, -ESTALE);
    if (lease->slot >= scope->children)
        return publisher_return(scope, -ESTALE);
    uint64_t control = atomic_load_explicit(&scope->control,
                                             memory_order_acquire);
    uint32_t phase = scope_phase(control);
    uint32_t cycle = scope_cycle(control);
    if (phase == KC_FIXED_SCOPE_STARTING)
        return publisher_return(scope, -ESTALE);
    if (phase != KC_FIXED_SCOPE_RUNNING &&
        phase != KC_FIXED_SCOPE_CLOSING &&
        phase != KC_FIXED_SCOPE_CANCELLING)
        return publisher_return(scope, -ECANCELED);

    kc_fixed_scope_child *child = &scope->slots[lease->slot];
    uint64_t observed = atomic_load_explicit(&child->word,
                                             memory_order_acquire);
    if (lease->child_generation != cycle ||
        lease->child_generation != child_generation(observed) ||
        lease->child_class != child->child_class)
        return publisher_return(scope, -ESTALE);
    kc_ticket_id parent = atomic_ticket_load(&scope->parent);
    kc_ticket_id ticket = atomic_ticket_load(&child->ticket);
    if (!ticket_equal(&lease->parent, &parent) ||
        !ticket_equal(&lease->child, &ticket) ||
        atomic_load_explicit(&scope->generation,
                             memory_order_acquire) != generation)
        return publisher_return(scope, -ESTALE);

    uint64_t expected = child_word(lease->child_generation,
                                   KC_SCOPE_SLOT_ACTIVE,
                                   KC_SCOPE_CAUSE_NONE);
    if (!atomic_compare_exchange_strong_explicit(
            &child->word, &expected,
            child_word(lease->child_generation, KC_SCOPE_SLOT_TERMINAL,
                       cause),
            memory_order_acq_rel, memory_order_acquire)) {
        if (child_generation(expected) != lease->child_generation)
            return publisher_return(scope, -ESTALE);
        if (child_state(expected) == KC_SCOPE_SLOT_CANCELLING) {
            uint64_t canceling = expected;
            if (!atomic_compare_exchange_strong_explicit(
                    &child->word, &canceling,
                    child_word(lease->child_generation,
                               KC_SCOPE_SLOT_TERMINAL, cause),
                    memory_order_acq_rel, memory_order_acquire)) {
                return child_state(canceling) == KC_SCOPE_SLOT_TERMINAL
                           ? publisher_return(scope, -EALREADY)
                           : publisher_return(scope, -ECANCELED);
            }
        } else {
            return child_state(expected) == KC_SCOPE_SLOT_TERMINAL
                       ? publisher_return(scope, -EALREADY)
                       : publisher_return(scope, -ECANCELED);
        }
    }

    if (child->child_class == KC_SCOPE_CHILD_FUNCTIONAL &&
        cause != KC_SCOPE_CAUSE_COMPLETE)
        (void)begin_closing(scope, cycle, cause, lease->slot);
    record_terminals(scope, 1);
    return publisher_return(scope, 0);
}

int kc_fixed_scope_cancel(kc_fixed_scope_t *scope, uint64_t generation,
                          const kc_ticket_id *parent, uint32_t cause)
{
    if (!scope || !ticket_valid(parent) || generation == 0 ||
        cause < KC_SCOPE_CAUSE_CANCELLED ||
        cause > KC_SCOPE_CAUSE_STOPPED) return -EINVAL;
    int admission = publisher_enter(scope);
    if (admission != 0) return admission;
    uint64_t current = atomic_load_explicit(&scope->generation,
                                            memory_order_acquire);
    if (generation != current) return publisher_return(scope, -ESTALE);
    kc_ticket_id current_parent = atomic_ticket_load(&scope->parent);
    if (!ticket_equal(parent, &current_parent) ||
        atomic_load_explicit(&scope->generation,
                             memory_order_acquire) != current)
        return publisher_return(scope, -ESTALE);
    uint64_t control = atomic_load_explicit(&scope->control,
                                             memory_order_acquire);
    if (scope_phase(control) != KC_FIXED_SCOPE_RUNNING)
        return publisher_return(scope, -EALREADY);
    if (atomic_load_explicit(&scope->generation,
                             memory_order_acquire) != current)
        return publisher_return(scope, -ESTALE);
    int status = begin_closing(scope, scope_cycle(control), cause,
                               UINT32_MAX)
                     ? 0 : -EALREADY;
    return publisher_return(scope, status);
}

int kc_fixed_scope_snapshot_get(const kc_fixed_scope_t *scope,
                                kc_fixed_scope_snapshot *out)
{
    if (!scope || !out) return -EINVAL;
    uint32_t cancelling = 0;
    for (uint32_t slot = 0; slot < scope->children; ++slot) {
        uint64_t word = atomic_load_explicit(&scope->slots[slot].word,
                                             memory_order_acquire);
        cancelling += child_state(word) == KC_SCOPE_SLOT_CANCELLING;
    }
    *out = (kc_fixed_scope_snapshot){
        .capacity = scope->capacity,
        .children = scope->children,
        .terminal_children = atomic_load_explicit(
            &scope->terminal_children, memory_order_acquire),
        .functional_children = scope->functional_children,
        .telemetry_children = scope->telemetry_children,
        .phase = scope_phase(atomic_load_explicit(
            &scope->control, memory_order_acquire)),
        .cause = atomic_load_explicit(&scope->cause, memory_order_acquire),
        .cause_slot = atomic_load_explicit(&scope->cause_slot,
                                           memory_order_acquire),
        .ready_edges = atomic_load_explicit(&scope->ready_edges,
                                             memory_order_acquire),
        .cancelling_children = cancelling,
        .generation = atomic_load_explicit(&scope->generation,
                                            memory_order_acquire),
        .parent = atomic_ticket_load(&scope->parent),
    };
    return 0;
}

int kc_fixed_scope_destroy(kc_fixed_scope_t *scope)
{
    if (!scope) return 0;
    uint32_t phase = scope_phase(atomic_load_explicit(
        &scope->control, memory_order_acquire));
    if (phase != KC_FIXED_SCOPE_SETUP && phase != KC_FIXED_SCOPE_SEALED &&
        phase != KC_FIXED_SCOPE_TERMINAL) return -EBUSY;
    if (atomic_load_explicit(&scope->publishers, memory_order_acquire) !=
        KC_SCOPE_PUBLISHERS_CLOSED) return -EBUSY;
    if (atomic_load_explicit(&scope->active_ready,
                             memory_order_acquire) != 0) return -EBUSY;
    free(scope->slots);
    free(scope);
    return 0;
}
