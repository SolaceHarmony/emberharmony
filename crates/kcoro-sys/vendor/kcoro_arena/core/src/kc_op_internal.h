// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_op.h"
#include "kc_descriptor_internal.h"
#include "kc_cancel_internal.h"
#include "kcoro_stackless.h"

#include <stdatomic.h>

struct kc_chan;

struct kc_op {
    atomic_uint refs;
    atomic_int state;
    atomic_int claimed;
    atomic_int published;
    atomic_int wake_mode;
    kc_op_kind kind;
    kc_op_cause cause;
    kc_id id;
    uint32_t generation;
    kc_id trace_id;
    koro_cont_t *cont;
    struct kc_chan *channel;
    kc_descriptor_t *descriptor;
    kc_descriptor_t *result_descriptor;
    kc_payload result;
    uint64_t deadline_ns;
    kc_cancel_t *cancel;
    kc_cancel_subscription cancel_subscription;
    struct kc_op *timer_prev;
    struct kc_op *timer_next;
    int timer_linked;
    struct kc_op *select_parent;
    struct kc_op **select_children;
    size_t select_count;
    size_t select_index;
    atomic_int select_cleaned;
    atomic_int select_building;
    struct kc_scope *scope;
    struct kc_op *prev;
    struct kc_op *next;
    struct kc_op *registry_prev;
    struct kc_op *registry_next;
    int linked;
};

kc_op *kc_op_create_internal(koro_cont_t *cont, struct kc_chan *channel,
                             kc_op_kind kind, kc_descriptor_t *descriptor);
int kc_op_claim_locked(kc_op *op, kc_op_cause cause, const kc_payload *payload);
int kc_op_claim_direct(kc_op *op, kc_op_cause cause, const kc_payload *payload);
int kc_op_claim_select(kc_op *op, kc_op_cause cause, const kc_payload *payload,
                       kc_descriptor_t *descriptor, size_t index);
int kc_op_arm(kc_op *op, kc_cancel_t *cancel, uint64_t deadline_ns);
int kc_op_prepare_suspend(kc_op *op);
int kc_op_cancel_cause(kc_op *op, kc_op_cause cause);
void kc_op_publish(kc_op *op);
int kc_op_is_terminal(const kc_op *op);
