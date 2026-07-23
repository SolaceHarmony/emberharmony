// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_FIXED_SCOPE_H
#define KC_FIXED_SCOPE_H

#include "kc_identity.h"
#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A fixed scope is a setup-time ownership graph, not a task scheduler. Child
 * slots are populated once, the table is sealed, and every later transition is
 * an exact generation-stamped terminal edge. No child owns a blocked stack and
 * there is no join, waiter, queue, timer, or work-stealing surface.
 */
typedef struct kc_fixed_scope kc_fixed_scope_t;

typedef enum kc_scope_child_class {
    KC_SCOPE_CHILD_FUNCTIONAL = 1,
    KC_SCOPE_CHILD_TELEMETRY = 2,
} kc_scope_child_class;

typedef enum kc_scope_cause {
    KC_SCOPE_CAUSE_NONE = 0,
    KC_SCOPE_CAUSE_COMPLETE = 1,
    KC_SCOPE_CAUSE_CANCELLED = 2,
    KC_SCOPE_CAUSE_FAULT = 3,
    KC_SCOPE_CAUSE_DEADLINE = 4,
    KC_SCOPE_CAUSE_STOPPED = 5,
} kc_scope_cause;

typedef enum kc_fixed_scope_phase {
    KC_FIXED_SCOPE_SETUP = 1,
    KC_FIXED_SCOPE_SEALED = 2,
    KC_FIXED_SCOPE_STARTING = 3,
    KC_FIXED_SCOPE_RUNNING = 4,
    KC_FIXED_SCOPE_CLOSING = 5,
    KC_FIXED_SCOPE_CANCELLING = 6,
    KC_FIXED_SCOPE_FINALIZING = 7,
    KC_FIXED_SCOPE_TERMINAL = 8,
} kc_fixed_scope_phase;

typedef void (*kc_scope_ready_fn)(void *context, uint64_t generation,
                                  uint32_t cause);

typedef struct kc_scope_child_lease {
    uint32_t slot;
    uint32_t child_class;
    uint64_t scope_generation;
    uint32_t child_generation;
    kc_ticket_id parent;
    kc_ticket_id child;
} kc_scope_child_lease;

typedef void (*kc_scope_child_cancel_fn)(
    void *context, const kc_scope_child_lease *lease, uint32_t cause);

/* Both callbacks are bounded edge publishers. They must publish durable owner
 * state and return: no allocation, lock, wait, child/application work, inline
 * continuation execution, or destruction of the scope is permitted. */

typedef struct kc_fixed_scope_config {
    uint32_t child_capacity;
    kc_scope_ready_fn ready;
    void *context;
} kc_fixed_scope_config;

typedef struct kc_scope_child_config {
    uint32_t child_class;
    kc_scope_child_cancel_fn cancel;
    void *context;
} kc_scope_child_config;

/* Per-cycle identity is rebound into the sealed role table synchronously.
 * child_tickets is borrowed only for cycle_begin; no pointer is retained. */
typedef struct kc_fixed_scope_cycle_config {
    uint32_t child_count;
    uint64_t generation;
    kc_ticket_id parent;
    const kc_ticket_id *child_tickets;
} kc_fixed_scope_cycle_config;

typedef struct kc_fixed_scope_snapshot {
    uint32_t capacity;
    uint32_t children;
    uint32_t terminal_children;
    uint32_t functional_children;
    uint32_t telemetry_children;
    uint32_t phase;
    uint32_t cause;
    uint32_t cause_slot;
    uint32_t ready_edges;
    uint32_t cancelling_children;
    uint64_t generation;
    kc_ticket_id parent;
} kc_fixed_scope_snapshot;

int kc_fixed_scope_create(const kc_fixed_scope_config *config,
                          kc_fixed_scope_t **out);
int kc_fixed_scope_add_role(kc_fixed_scope_t *scope,
                            const kc_scope_child_config *config,
                            uint32_t *out_slot);
int kc_fixed_scope_seal(kc_fixed_scope_t *scope);

/* Starts the first or a later cycle on the same sealed storage. All identity
 * validation completes before STARTING is published. The caller supplies a
 * fixed output array; no allocation occurs after seal. Old leases are rejected
 * by both the scope generation and an internal per-role generation. */
int kc_fixed_scope_cycle_begin(
    kc_fixed_scope_t *scope, const kc_fixed_scope_cycle_config *config,
    kc_scope_child_lease *out_leases, size_t lease_capacity);

/* COMPLETE retires one child normally. A non-COMPLETE functional cause closes
 * the whole scope and structurally publishes cancellation to every still-live
 * sibling. A telemetry failure retires only that lossy child. Cancellation is
 * not completion: every child must acknowledge its own terminal edge before
 * the final child publishes exactly one parent-ready edge. */
int kc_fixed_scope_child_terminal(kc_fixed_scope_t *scope,
                                  const kc_scope_child_lease *lease,
                                  uint32_t cause);

/* Parent cancellation is an exact correlated edge. A delayed cancellation
 * from an earlier cycle is rejected before it can affect rebound roles. */
int kc_fixed_scope_cancel(kc_fixed_scope_t *scope, uint64_t generation,
                          const kc_ticket_id *parent, uint32_t cause);

int kc_fixed_scope_snapshot_get(const kc_fixed_scope_t *scope,
                                kc_fixed_scope_snapshot *out);

/* Destruction never waits. Closing first rejects new child/cancel publishers;
 * TERMINAL and the ready callback are withheld until every publisher admitted
 * before that close has drained. An active callback lifetime keeps destruction
 * busy through ready's return; its release is the publisher's final scope
 * access. */
int kc_fixed_scope_destroy(kc_fixed_scope_t *scope);

#ifdef __cplusplus
}
#endif

#endif
