// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_DEADLINE_H
#define KC_DEADLINE_H

#include "kc_identity.h"
#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A deadline is an externally generated, correlated edge. It may report a
 * liveness failure or release a policy gate, but it never polls and never
 * advances numerical work by itself. */
typedef struct kc_deadline_source kc_deadline_source_t;

typedef enum kc_deadline_event_kind {
    KC_DEADLINE_EVENT_EXPIRED = 1,
    KC_DEADLINE_EVENT_STALE = 2,
} kc_deadline_event_kind;

typedef enum kc_deadline_source_phase {
    KC_DEADLINE_SOURCE_OPEN = 1,
    KC_DEADLINE_SOURCE_STOPPING = 2,
    KC_DEADLINE_SOURCE_STOPPED = 3,
} kc_deadline_source_phase;

/* Non-negative outcomes from kc_deadline_source_retire. Negative values are
 * errno-style API/lifecycle failures. */
typedef enum kc_deadline_retire_result {
    KC_DEADLINE_RETIRE_RETIRED = 0,
    KC_DEADLINE_RETIRE_EXPIRY_WON = 1,
} kc_deadline_retire_result;

typedef void (*kc_deadline_notify_fn)(void *context);

/* notify is a retained edge sink, normally kc_service_notifier_notify behind a
 * signature adapter. It must only publish its owned predicate: it may not lock,
 * allocate, wait, run the consumer inline, or destroy this source from inside
 * the handler. The source retains handler lifetime through notify's return, and
 * context must be retained notifier/mailbox state, never a product route,
 * conversation, or numerical scratch owner. */

typedef struct kc_deadline_source_config {
    uint32_t size;
    uint32_t abi_version;
    uint32_t capacity;
    uint32_t reserved;
    kc_deadline_notify_fn notify;
    void *context;
} kc_deadline_source_config;

typedef struct kc_deadline_arm_config {
    uint32_t size;
    uint32_t abi_version;
    uint32_t slot;
    uint32_t reserved;
    uint64_t delay_ns;
    kc_ticket_id child;
    kc_ticket_id parent;
    uint64_t scope_generation;
    uint64_t epoch;
    uint64_t domain;
    uint64_t team_generation;
} kc_deadline_arm_config;

typedef struct kc_deadline_arm {
    uint32_t size;
    uint32_t abi_version;
    uint32_t slot;
    uint32_t reserved;
    uint64_t arm_generation;
    kc_ticket_id child;
    kc_ticket_id parent;
    uint64_t scope_generation;
    uint64_t epoch;
    uint64_t domain;
    uint64_t team_generation;
} kc_deadline_arm;

typedef struct kc_deadline_event {
    uint32_t size;
    uint32_t abi_version;
    uint32_t slot;
    uint32_t kind;
    uint64_t sequence;
    uint64_t scheduled_arm_generation;
    uint64_t current_arm_generation;
    kc_ticket_id child;
    kc_ticket_id parent;
    uint64_t scope_generation;
    uint64_t epoch;
    uint64_t domain;
    uint64_t team_generation;
} kc_deadline_event;

typedef struct kc_deadline_source_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint32_t capacity;
    uint32_t phase;
    uint32_t idle;
    uint32_t armed;
    uint32_t pending_events;
    uint32_t reserved;
    uint64_t published_events;
    uint64_t stale_events;
    uint64_t notifications;
    uint32_t cancellation_acks;
    uint32_t active_handlers;
} kc_deadline_source_snapshot;

/* Creates a fixed native source pool during readiness. On Apple platforms each
 * slot owns one GCD one-shot timer on a private serial queue and uses monotonic
 * dispatch_time. Non-Apple production construction returns -ENOTSUP so an
 * engine rejects the unsupported backend before admitting numerical work. */
int kc_deadline_source_create(const kc_deadline_source_config *config,
                              kc_deadline_source_t **out);

/* Deterministic backend for lifecycle tests only. Production code must call
 * kc_deadline_source_create and must not drive expiry itself. */
int kc_deadline_source_create_manual_test(
    const kc_deadline_source_config *config, kc_deadline_source_t **out);

int kc_deadline_source_arm(kc_deadline_source_t *source,
                           const kc_deadline_arm_config *config,
                           kc_deadline_arm *out_arm);

/* Silently retires exactly one still-ARMED generation after its supervised
 * work completed normally. Success advances the slot generation and makes it
 * reusable without publishing an event or invoking notify. This is the normal
 * completion path; disarm below deliberately retains its existing observable
 * STALE-event semantics for policy cancellation.
 *
 * KC_DEADLINE_RETIRE_EXPIRY_WON means the exact generation already crossed
 * the expiry terminal CAS (including an expiry already acknowledged by its
 * supervisor), so the caller must defer to that correlated supervisor path.
 * -ESTALE means a different generation owns the slot. No outcome waits, polls,
 * allocates, or runs a consumer inline. */
int kc_deadline_source_retire(kc_deadline_source_t *source, uint32_t slot,
                              uint64_t arm_generation);

/* Disarm races expiry through the same terminal CAS, advances generation, and
 * immediately publishes the exact scheduled identity as STALE. event_ack then
 * makes the permanent slot reusable. A queued native callback is only an
 * untrusted wake hint: it must observe a currently armed slot whose current
 * monotonic due time has actually arrived, so it cannot expire a later arm. */
int kc_deadline_source_disarm(kc_deadline_source_t *source, uint32_t slot,
                              uint64_t arm_generation);

int kc_deadline_source_event_get(const kc_deadline_source_t *source,
                                 uint32_t slot, kc_deadline_event *out);
/* Acknowledgement consumes the exact immutable observation returned by
 * event_get. Slot, complete ticket lineage, policy identity, sequence, and arm
 * generations must all match before the terminal slot CAS, so a corrupted slot
 * or delayed acknowledgement cannot clear an unrelated or successor EVENT. */
int kc_deadline_source_event_ack(kc_deadline_source_t *source,
                                 const kc_deadline_event *event);

/* Deterministic test backend only. Time advances without publishing work; fire
 * injects the same identity-free wake hint as a queued native callback. */
int kc_deadline_source_advance_manual_test(kc_deadline_source_t *source,
                                           uint64_t elapsed_ns);
int kc_deadline_source_fire_manual_test(kc_deadline_source_t *source,
                                        uint32_t slot);

/* Stop closes arm admission and asynchronously cancels every native source.
 * STOPPED is one correlated logical edge published only after the cancellation
 * submission walk is complete and every GCD cancel handler acknowledges. A
 * handler may still be returning from that retained notification; Apple
 * destruction drains the private serial queue administratively before freeing
 * storage. Consumers therefore never poll active_handlers or wait for a
 * handler-stack zero transition. Producers must be disconnected and quiesced
 * before destroy. There is no source join or waiter API. */
void kc_deadline_source_request_stop(kc_deadline_source_t *source);
int kc_deadline_source_snapshot_get(const kc_deadline_source_t *source,
                                    kc_deadline_source_snapshot *out);
int kc_deadline_source_destroy(kc_deadline_source_t *source);

#ifdef __cplusplus
}
#endif

#endif
