// SPDX-License-Identifier: BSD-3-Clause
#include "kc_op_internal.h"
#include "kc_channel_internal.h"
#include "kc_runtime_internal.h"
#include "kc_scope_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdlib.h>

enum {
    KC_WAKE_UNDECIDED = 0,
    KC_WAKE_ARMED,
    KC_WAKE_PUBLISHED,
    KC_WAKE_CONSUMED,
};

static kc_op_state terminal_state(kc_op_cause cause)
{
    switch (cause) {
    case KC_CAUSE_MATCH: return KC_OP_OK;
    case KC_CAUSE_CLOSE: return KC_OP_CLOSED;
    case KC_CAUSE_CANCEL: return KC_OP_CANCELED;
    case KC_CAUSE_TIMEOUT: return KC_OP_TIMED_OUT;
    default: return KC_OP_FAILED;
    }
}

kc_op *kc_op_create_internal(koro_cont_t *cont, struct kc_chan *channel,
                             kc_op_kind kind, kc_descriptor_t *descriptor)
{
    if (!cont || (!channel && kind != KC_OP_TIMER && kind != KC_OP_SELECT &&
                  kind != KC_OP_JOIN)) return NULL;
    kc_op *op = calloc(1, sizeof(*op));
    if (!op) return NULL;
    atomic_init(&op->refs, 1);
    atomic_init(&op->state, KC_OP_REGISTERING);
    atomic_init(&op->claimed, 0);
    atomic_init(&op->published, 0);
    atomic_init(&op->wake_mode, KC_WAKE_UNDECIDED);
    atomic_init(&op->select_cleaned, 0);
    atomic_init(&op->select_building, 0);
    op->select_index = SIZE_MAX;
    op->kind = kind;
    op->cont = cont;
    op->channel = channel;
    op->descriptor = descriptor;
    op->id.epoch = cont->runtime->epoch;
    op->id.sequence = kc_runtime_next_sequence(cont->runtime);
    op->generation = 1;
    op->trace_id = op->id;
    koro_cont_retain(cont);
    if (channel) kc_channel_retain(channel);
    kc_runtime_register_op(cont->runtime, op);
    return op;
}

void kc_op_retain(kc_op *op)
{
    if (op) atomic_fetch_add_explicit(&op->refs, 1, memory_order_relaxed);
}

void kc_op_release(kc_op *op)
{
    if (!op) return;
    if (atomic_fetch_sub_explicit(&op->refs, 1, memory_order_acq_rel) != 1) return;
    kc_op *select_parent = op->select_parent;
    kc_scope_t *scope = op->scope;
    kc_runtime_unregister_op(op->cont->runtime, op);
    kc_descriptor_release(op->descriptor);
    kc_descriptor_release(op->result_descriptor);
    if (op->channel) kc_channel_release(op->channel);
    koro_cont_release_internal(op->cont);
    free(op);
    if (select_parent) kc_op_release(select_parent);
    if (scope) kc_scope_release(scope);
}

static int claim_result(kc_op *op, kc_op_cause cause, const kc_payload *payload,
                        kc_descriptor_t *descriptor, size_t index)
{
    if (!op) return 0;
    int state = atomic_load_explicit(&op->state, memory_order_acquire);
    if (state != KC_OP_REGISTERING && state != KC_OP_WAITING) return 0;
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(&op->claimed, &expected, 1,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) return 0;
    op->cause = cause;
    if (payload) op->result = *payload;
    if (descriptor) op->result_descriptor = descriptor;
    if (index != SIZE_MAX) op->select_index = index;
    atomic_store_explicit(&op->state, KC_OP_COMPLETING, memory_order_release);
    return 1;
}

int kc_op_claim_locked(kc_op *op, kc_op_cause cause, const kc_payload *payload)
{
    return claim_result(op, cause, payload, NULL, SIZE_MAX);
}

int kc_op_claim_direct(kc_op *op, kc_op_cause cause, const kc_payload *payload)
{
    return kc_op_claim_locked(op, cause, payload);
}

int kc_op_claim_select(kc_op *op, kc_op_cause cause, const kc_payload *payload,
                       kc_descriptor_t *descriptor, size_t index)
{
    return claim_result(op, cause, payload, descriptor, index);
}

static void cancel_operation(void *context)
{
    kc_op *op = context;
    (void)kc_op_cancel(op);
    kc_op_release(op);
}

int kc_op_arm(kc_op *op, kc_cancel_t *cancel, uint64_t deadline_ns)
{
    if (!op || (op->kind == KC_OP_TIMER && !deadline_ns)) return -EINVAL;
    if (cancel) {
        op->cancel = cancel;
        kc_cancel_retain(cancel);
        kc_op_retain(op);
        int rc = kc_cancel_subscribe(cancel, &op->cancel_subscription,
                                     cancel_operation, op);
        if (rc != 0) {
            kc_op_release(op);
            if (rc == -ECANCELED) {
                (void)kc_op_cancel(op);
                return 0;
            }
            kc_cancel_release(cancel);
            op->cancel = NULL;
            return rc;
        }
    }
    if (!deadline_ns) return 0;
    int rc = kc_runtime_timer_arm(op, deadline_ns);
    if (rc == 0) return 0;
    if (op->channel) (void)kc_channel_cancel_op(op, KC_CAUSE_FAILURE);
    else if (kc_op_claim_direct(op, KC_CAUSE_FAILURE,
                                &(kc_payload){ .status = rc })) kc_op_publish(op);
    return rc;
}

static void kc_op_disarm(kc_op *op)
{
    kc_runtime_timer_disarm(op);
    kc_cancel_t *cancel = op->cancel;
    if (!cancel) return;
    op->cancel = NULL;
    if (kc_cancel_unsubscribe(&op->cancel_subscription)) kc_op_release(op);
    kc_cancel_release(cancel);
}

static void kc_select_cleanup(kc_op *parent)
{
    if (!parent || parent->kind != KC_OP_SELECT ||
        atomic_exchange_explicit(&parent->select_cleaned, 1,
                                 memory_order_acq_rel)) return;
    kc_op **children = parent->select_children;
    size_t count = parent->select_count;
    parent->select_children = NULL;
    parent->select_count = 0;
    for (size_t index = 0; index < count; index++) {
        kc_op *child = children[index];
        if (!child) continue;
        if (!kc_op_is_terminal(child)) (void)kc_op_cancel(child);
        kc_op_release(child);
    }
    free(children);
}

static void kc_select_child_complete(kc_op *child)
{
    kc_op *parent = child->select_parent;
    if (!parent) return;
    if (!kc_op_claim_select(parent, child->cause, &child->result,
                            child->result_descriptor,
                            child->select_index)) return;
    child->result_descriptor = NULL;
    kc_op_publish(parent);
}

int kc_op_prepare_suspend(kc_op *op)
{
    if (!op) return 0;
    if (kc_op_is_terminal(op)) {
        int expected = KC_WAKE_UNDECIDED;
        (void)atomic_compare_exchange_strong_explicit(
            &op->wake_mode, &expected, KC_WAKE_CONSUMED,
            memory_order_acq_rel, memory_order_acquire);
        return 0;
    }
    int expected = KC_WAKE_UNDECIDED;
    if (atomic_compare_exchange_strong_explicit(
            &op->wake_mode, &expected, KC_WAKE_ARMED,
            memory_order_acq_rel, memory_order_acquire)) return 1;
    return expected == KC_WAKE_ARMED;
}

void kc_op_publish(kc_op *op)
{
    if (!op) return;
    if (atomic_load_explicit(&op->state, memory_order_acquire) != KC_OP_COMPLETING) return;
    if (op->kind == KC_OP_SELECT &&
        atomic_load_explicit(&op->select_building, memory_order_acquire)) return;
    int expected_publish = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &op->published, &expected_publish, 1,
            memory_order_acq_rel, memory_order_acquire)) return;
    int select_child = op->select_parent != NULL;
    if (select_child) kc_op_retain(op);
    if (op->kind == KC_OP_SELECT) kc_select_cleanup(op);
    kc_op_disarm(op);
    atomic_store_explicit(&op->state, terminal_state(op->cause), memory_order_release);
    if ((unsigned)op->cause <= KC_CAUSE_FAILURE) {
        atomic_fetch_add_explicit(
            &op->cont->runtime->terminal_causes[op->cause], 1,
            memory_order_relaxed);
    }
    if (select_child) {
        kc_select_child_complete(op);
        kc_op_release(op);
        return;
    }
    int expected = KC_WAKE_UNDECIDED;
    if (atomic_compare_exchange_strong_explicit(
            &op->wake_mode, &expected, KC_WAKE_PUBLISHED,
            memory_order_acq_rel, memory_order_acquire)) return;
    if (expected == KC_WAKE_ARMED) kc_runtime_wake_internal(op->cont);
}

int kc_op_is_terminal(const kc_op *op)
{
    if (!op) return 1;
    int state = atomic_load_explicit(&op->state, memory_order_acquire);
    return state >= KC_OP_OK;
}

int kc_op_cancel(kc_op *op)
{
    return kc_op_cancel_cause(op, KC_CAUSE_CANCEL);
}

int kc_op_cancel_cause(kc_op *op, kc_op_cause cause)
{
    if (!op) return -EINVAL;
    if (op->channel) return kc_channel_cancel_op(op, cause);
    if (op->scope) return kc_scope_cancel_op(op, cause);
    int status = cause == KC_CAUSE_TIMEOUT ? -ETIMEDOUT
        : cause == KC_CAUSE_FAILURE ? -EIO : -ECANCELED;
    if (!kc_op_claim_direct(op, cause,
                            &(kc_payload){ .status = status })) return -EALREADY;
    kc_op_publish(op);
    return 0;
}

int kc_op_snapshot_get(const kc_op *op, kc_op_snapshot *out)
{
    if (!op || !out || out->size < sizeof(*out)) return -EINVAL;
    kc_op_state state = (kc_op_state)atomic_load_explicit(&op->state,
                                                          memory_order_acquire);
    int terminal = state >= KC_OP_OK;
    *out = (kc_op_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = op->id, .generation = op->generation,
        .trace_id = op->trace_id, .kind = op->kind,
        .state = state,
        .cause = terminal ? op->cause : KC_CAUSE_NONE,
        .result = terminal ? op->result.status : 0,
        .descriptor = terminal ? kc_descriptor_id_get(op->result_descriptor)
                               : (kc_descriptor_id){0},
        .deadline_ns = state == KC_OP_REGISTERING ? 0 : op->deadline_ns,
        .select_index = terminal ? op->select_index : SIZE_MAX,
    };
    return 0;
}
