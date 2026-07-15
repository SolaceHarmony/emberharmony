// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_OP_H
#define KC_OP_H

#include "kc_descriptor.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_op kc_op;

typedef struct kc_id {
    uint64_t epoch;
    uint64_t sequence;
} kc_id;

typedef enum kc_op_kind {
    KC_OP_SEND = 1,
    KC_OP_RECV,
    KC_OP_TIMER,
    KC_OP_SELECT,
    KC_OP_JOIN,
} kc_op_kind;

typedef enum kc_op_state {
    KC_OP_NEW = 0,
    KC_OP_REGISTERING,
    KC_OP_WAITING,
    KC_OP_COMPLETING,
    KC_OP_OK,
    KC_OP_CLOSED,
    KC_OP_TIMED_OUT,
    KC_OP_CANCELED,
    KC_OP_FAILED,
} kc_op_state;

typedef enum kc_op_cause {
    KC_CAUSE_NONE = 0,
    KC_CAUSE_MATCH,
    KC_CAUSE_CLOSE,
    KC_CAUSE_CANCEL,
    KC_CAUSE_TIMEOUT,
    KC_CAUSE_AGAIN,
    KC_CAUSE_FAILURE,
} kc_op_cause;

typedef struct kc_op_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id id;
    uint32_t generation;
    uint32_t reserved;
    kc_id trace_id;
    kc_op_kind kind;
    kc_op_state state;
    kc_op_cause cause;
    int result;
    kc_descriptor_id descriptor;
    uint64_t deadline_ns;
    size_t select_index;
} kc_op_snapshot;

void kc_op_retain(kc_op *op);
void kc_op_release(kc_op *op);
int kc_op_cancel(kc_op *op);
int kc_op_snapshot_get(const kc_op *op, kc_op_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
