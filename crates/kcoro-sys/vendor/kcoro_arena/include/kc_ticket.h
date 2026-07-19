// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_TICKET_H
#define KC_TICKET_H

#include "kc_descriptor.h"
#include "kc_identity.h"
#include "kc_op.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_runtime kc_runtime_t;
typedef struct kc_ticket kc_ticket_t;

typedef enum kc_ticket_state {
    KC_TICKET_CREATED = 0,
    KC_TICKET_ACCEPTED,
    KC_TICKET_DISPATCHED,
    KC_TICKET_PUBLISHED,
} kc_ticket_state;

typedef enum kc_ticket_deadline_mode {
    KC_TICKET_DEADLINE_NONE = 0,
    KC_TICKET_DEADLINE_QUEUE,
    KC_TICKET_DEADLINE_HARD_PUBLICATION,
    KC_TICKET_DEADLINE_SOFT,
} kc_ticket_deadline_mode;

typedef enum kc_ticket_execution_status {
    KC_TICKET_NOT_DISPATCHED = 0,
    KC_TICKET_EXECUTION_COMPLETED,
    KC_TICKET_EXECUTION_FAILED,
} kc_ticket_execution_status;

typedef enum kc_ticket_state_status {
    KC_TICKET_STATE_NONE = 0,
    KC_TICKET_STATE_COMMITTED,
    KC_TICKET_STATE_ROLLED_BACK,
    KC_TICKET_STATE_POISONED,
} kc_ticket_state_status;

typedef enum kc_ticket_publication_status {
    KC_TICKET_PUBLICATION_NONE = 0,
    KC_TICKET_PUBLICATION_COMMITTED,
    KC_TICKET_PUBLICATION_STALE,
} kc_ticket_publication_status;

typedef enum kc_ticket_terminal_cause {
    KC_TICKET_CAUSE_SUCCESS = 0,
    KC_TICKET_CAUSE_REJECTED,
    KC_TICKET_CAUSE_CANCELED,
    KC_TICKET_CAUSE_TIMED_OUT,
    KC_TICKET_CAUSE_STALE_EPOCH,
    KC_TICKET_CAUSE_STOP,
    KC_TICKET_CAUSE_FAULT,
} kc_ticket_terminal_cause;

typedef struct kc_ticket_event_v1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    kc_ticket_id ticket;
    kc_ticket_id parent;
    kc_id correlation;
    kc_id trace;
    uint64_t context_id;
    uint64_t epoch;
    int32_t execution_status;
    int32_t state_status;
    int32_t publication_status;
    int32_t terminal_cause;
    int32_t status_code;
    uint32_t reserved1;
    kc_descriptor_id result;
    uint64_t accepted_ns;
    uint64_t dispatched_ns;
    uint64_t completed_ns;
    uint64_t published_ns;
} kc_ticket_event_v1;

typedef void (*kc_ticket_callback_fn)(void *context,
                                      const kc_ticket_event_v1 *event);
typedef void (*kc_ticket_context_retain_fn)(void *context);
typedef void (*kc_ticket_context_release_fn)(void *context);

typedef struct kc_ticket_config_v1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t kind;
    kc_ticket_id parent;
    kc_id correlation;
    kc_id trace;
    uint64_t context_id;
    uint64_t epoch;
    /* Zero means no deadline. NONE requires zero; other modes may use zero. */
    uint64_t deadline_ns;
    int32_t deadline_mode;
    uint32_t reserved;
    kc_ticket_callback_fn callback;
    void *callback_context;
    kc_ticket_context_retain_fn context_retain;
    kc_ticket_context_release_fn context_release;
} kc_ticket_config_v1;

typedef struct kc_ticket_completion_v1 {
    uint32_t size;
    uint32_t abi_version;
    int32_t execution_status;
    int32_t state_status;
    int32_t publication_status;
    int32_t terminal_cause;
    int32_t status_code;
    uint32_t reserved;
    kc_descriptor_t *result;
} kc_ticket_completion_v1;

typedef struct kc_ticket_snapshot_v1 {
    uint32_t size;
    uint32_t abi_version;
    kc_ticket_state state;
    uint32_t cancel_requested;
    uint32_t target_consumed;
    uint32_t reserved;
    kc_ticket_event_v1 event;
} kc_ticket_snapshot_v1;

int kc_ticket_create(kc_runtime_t *runtime,
                     const kc_ticket_config_v1 *config,
                     kc_ticket_t **out);
int kc_ticket_attach_descriptor(kc_ticket_t *ticket,
                                kc_descriptor_t *descriptor);
int kc_ticket_accept(kc_ticket_t *ticket);
int kc_ticket_dispatch(kc_ticket_t *ticket);
int kc_ticket_complete(kc_ticket_t *ticket,
                       const kc_ticket_completion_v1 *completion);
int kc_ticket_cancel(kc_ticket_t *ticket);
int kc_ticket_complete_id(kc_runtime_t *runtime, kc_ticket_id ticket,
                          const kc_ticket_completion_v1 *completion);
int kc_ticket_cancel_id(kc_runtime_t *runtime, kc_ticket_id ticket);
kc_ticket_id kc_ticket_id_get(const kc_ticket_t *ticket);
int kc_ticket_snapshot_get(const kc_ticket_t *ticket,
                           kc_ticket_snapshot_v1 *out);
void kc_ticket_retain(kc_ticket_t *ticket);
void kc_ticket_release(kc_ticket_t *ticket);

#ifdef __cplusplus
}
#endif

#endif
