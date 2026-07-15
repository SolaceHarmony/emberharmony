// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_scope.h"
#include "kcoro_port.h"

#include <stdatomic.h>

struct kc_op;

struct kc_scope {
    atomic_uint refs;
    kc_runtime_t *runtime;
    kc_cancel_t *cancel;
    kc_id id;
    KC_MUTEX_T mu;
    KC_COND_T cv;
    size_t children;
    size_t join_waiters;
    struct kc_op *join_head;
    struct kc_op *join_tail;
    struct kc_scope *registry_prev;
    struct kc_scope *registry_next;
    int closed;
};

int kc_scope_join_submit(kc_scope_t *scope, struct kc_op *op);
int kc_scope_cancel_op(struct kc_op *op, kc_op_cause cause);
