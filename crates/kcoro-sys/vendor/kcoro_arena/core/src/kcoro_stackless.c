// SPDX-License-Identifier: BSD-3-Clause
#include "kcoro_stackless.h"
#include "kc_channel_internal.h"
#include "kc_op_internal.h"
#include "kc_runtime_internal.h"
#include "kc_timer_internal.h"
#include "kcoro_port.h"

#include <errno.h>
#include <stdlib.h>

koro_cont_t *koro_cont_create_on(kc_runtime_t *runtime, koro_step_fn step,
                                 void *arg, size_t size)
{
    if (!runtime || !step) return NULL;
    koro_cont_t *cont = calloc(1, sizeof(*cont));
    if (!cont) return NULL;
    if (size) {
        cont->user_data = calloc(1, size);
        if (!cont->user_data) { free(cont); return NULL; }
    }
    cont->next_step = step;
    cont->user_arg = arg;
    cont->runtime = runtime;
    kc_runtime_retain_internal(runtime);
    cont->id = kc_runtime_next_sequence(runtime);
    cont->name = "stackless";
    cont->suspend_kind = KORO_SUSPEND_WAIT;
    atomic_init(&cont->run_state, KORO_NEW);
    atomic_init(&cont->wake_pending, 0);
    atomic_init(&cont->destroy_requested, 0);
    atomic_init(&cont->refs, 1);
    return cont;
}

koro_cont_t *koro_cont_create(koro_step_fn step, void *arg, size_t size)
{
    return koro_cont_create_on(kc_runtime_default_get(), step, arg, size);
}

void koro_cont_retain(koro_cont_t *cont)
{
    if (cont) atomic_fetch_add_explicit(&cont->refs, 1, memory_order_relaxed);
}

static void cont_release(koro_cont_t *cont)
{
    if (!cont) return;
    if (atomic_fetch_sub_explicit(&cont->refs, 1, memory_order_acq_rel) != 1) return;
    kc_descriptor_release(cont->arena_descriptor);
    free(cont->user_data);
    kc_runtime_release_internal(cont->runtime);
    free(cont);
}

void koro_cont_destroy(koro_cont_t *cont)
{
    if (!cont) return;
    atomic_store_explicit(&cont->destroy_requested, 1, memory_order_release);
    kc_op *op = NULL;
    KC_MUTEX_LOCK(&cont->runtime->mu);
    for (kc_op *candidate = cont->runtime->ops_head; candidate;
         candidate = candidate->registry_next) {
        if (candidate->cont == cont && !candidate->select_parent &&
            !kc_op_is_terminal(candidate)) {
            op = candidate;
            kc_op_retain(op);
            break;
        }
    }
    KC_MUTEX_UNLOCK(&cont->runtime->mu);
    if (op) {
        (void)kc_op_cancel(op);
        kc_op_release(op);
    }
    kc_runtime_wake_internal(cont);
    cont_release(cont);
}

void koro_cont_release_internal(koro_cont_t *cont) { cont_release(cont); }

static int begin_error(koro_cont_t *cont, int status)
{
    cont->last_park_result = status;
    return 1;
}

int koro_send_begin(koro_cont_t *cont, struct kc_chan *channel,
                    void *data, size_t len)
{
    return koro_send_begin_ex(cont, channel, data, len, NULL, 0);
}

int koro_send_begin_ex(koro_cont_t *cont, struct kc_chan *channel,
                       void *data, size_t len, kc_cancel_t *cancel,
                       uint64_t deadline_ns)
{
    if (!cont) return 1;
    if (!channel || (!data && len)) return begin_error(cont, -EINVAL);
    if (cont->arena_op) return kc_op_is_terminal(cont->arena_op);

    kc_descriptor_t *descriptor = NULL;
    if (kc_descriptor_create_copy(cont->runtime, data, len, &descriptor) != 0) {
        return begin_error(cont, -ENOMEM);
    }
    kc_op *op = kc_op_create_internal(cont, channel, KC_OP_SEND, descriptor);
    if (!op) { kc_descriptor_release(descriptor); return begin_error(cont, -ENOMEM); }
    cont->arena_op = op;
    cont->suspend_kind = KORO_SUSPEND_WAIT;
    if (atomic_load_explicit(&cont->destroy_requested, memory_order_acquire)) {
        (void)kc_op_cancel(op);
    }
    int rc = kc_op_arm(op, cancel, deadline_ns);
    if (rc == 0 && !kc_op_is_terminal(op)) rc = kc_channel_submit(channel, op);
    if (rc != 0 && !kc_op_is_terminal(op)) {
        (void)kc_channel_cancel_op(op, KC_CAUSE_FAILURE);
    }
    return !kc_op_prepare_suspend(op);
}

int koro_recv_begin(koro_cont_t *cont, struct kc_chan *channel)
{
    return koro_recv_begin_ex(cont, channel, NULL, 0);
}

int koro_recv_begin_ex(koro_cont_t *cont, struct kc_chan *channel,
                       kc_cancel_t *cancel, uint64_t deadline_ns)
{
    if (!cont) return 1;
    if (!channel) return begin_error(cont, -EINVAL);
    if (cont->arena_op) return kc_op_is_terminal(cont->arena_op);

    kc_op *op = kc_op_create_internal(cont, channel, KC_OP_RECV, 0);
    if (!op) return begin_error(cont, -ENOMEM);
    cont->arena_op = op;
    cont->suspend_kind = KORO_SUSPEND_WAIT;
    if (atomic_load_explicit(&cont->destroy_requested, memory_order_acquire)) {
        (void)kc_op_cancel(op);
    }
    int rc = kc_op_arm(op, cancel, deadline_ns);
    if (rc == 0 && !kc_op_is_terminal(op)) rc = kc_channel_submit(channel, op);
    if (rc != 0 && !kc_op_is_terminal(op)) {
        (void)kc_channel_cancel_op(op, KC_CAUSE_FAILURE);
    }
    return !kc_op_prepare_suspend(op);
}

int koro_sleep_begin(koro_cont_t *cont, uint64_t deadline_ns,
                     kc_cancel_t *cancel)
{
    if (!cont) return 1;
    if (!deadline_ns) return begin_error(cont, -EINVAL);
    if (cont->arena_op) return kc_op_is_terminal(cont->arena_op);
    kc_op *op = kc_op_create_internal(cont, NULL, KC_OP_TIMER, NULL);
    if (!op) return begin_error(cont, -ENOMEM);
    cont->arena_op = op;
    cont->suspend_kind = KORO_SUSPEND_WAIT;
    if (atomic_load_explicit(&cont->destroy_requested, memory_order_acquire)) {
        (void)kc_op_cancel(op);
    }
    int rc = kc_op_arm(op, cancel, deadline_ns);
    if (rc != 0 && !kc_op_is_terminal(op)) {
        if (kc_op_claim_direct(op, KC_CAUSE_FAILURE,
                               &(kc_payload){ .status = rc })) kc_op_publish(op);
    }
    return !kc_op_prepare_suspend(op);
}

int koro_timer_begin(koro_cont_t *cont, kc_timer_t *timer)
{
    if (!cont) return 1;
    if (!timer) return begin_error(cont, -EINVAL);
    return koro_sleep_begin(cont, timer->deadline_ns, timer->cancel);
}

int koro_select_begin(koro_cont_t *cont, const kc_select_clause *clauses,
                      size_t count, kc_cancel_t *cancel,
                      uint64_t deadline_ns)
{
    if (!cont) return 1;
    if (!clauses || !count) return begin_error(cont, -EINVAL);
    if (cont->arena_op) return kc_op_is_terminal(cont->arena_op);

    kc_op *parent = kc_op_create_internal(cont, NULL, KC_OP_SELECT, NULL);
    if (!parent) return begin_error(cont, -ENOMEM);
    parent->select_children = calloc(count, sizeof(*parent->select_children));
    if (!parent->select_children) {
        kc_op_release(parent);
        return begin_error(cont, -ENOMEM);
    }
    parent->select_count = count;
    atomic_store_explicit(&parent->select_building, 1, memory_order_release);
    cont->arena_op = parent;
    cont->select_index = SIZE_MAX;
    cont->suspend_kind = KORO_SUSPEND_WAIT;

    int error = 0;
    size_t default_index = SIZE_MAX;
    for (size_t index = 0; index < count; index++) {
        const kc_select_clause *clause = &clauses[index];
        if (clause->size < sizeof(*clause) ||
            clause->abi_version != KC_ABI_VERSION) { error = -EINVAL; break; }
        if (clause->kind == KC_SELECT_DEFAULT) {
            if (default_index != SIZE_MAX || clause->channel) {
                error = -EINVAL;
                break;
            }
            default_index = index;
            continue;
        }
        if ((clause->kind != KC_SELECT_SEND && clause->kind != KC_SELECT_RECV) ||
            !clause->channel || clause->channel->runtime != cont->runtime) {
            error = -EINVAL;
            break;
        }
        kc_descriptor_t *descriptor = NULL;
        if (clause->kind == KC_SELECT_SEND) {
            if ((!clause->data && clause->length) ||
                clause->length != clause->channel->element_size) {
                error = -EINVAL;
                break;
            }
            error = kc_descriptor_create_copy(cont->runtime, clause->data,
                                              clause->length, &descriptor);
            if (error != 0) break;
        }
        kc_op_kind kind = clause->kind == KC_SELECT_SEND ? KC_OP_SEND : KC_OP_RECV;
        kc_op *child = kc_op_create_internal(cont, clause->channel, kind, descriptor);
        if (!child) {
            kc_descriptor_release(descriptor);
            error = -ENOMEM;
            break;
        }
        child->select_parent = parent;
        child->select_index = index;
        kc_op_retain(parent);
        parent->select_children[index] = child;
    }

    if (!error) {
        for (size_t index = 0; index < count; index++) {
            kc_op *child = parent->select_children[index];
            if (!child) continue;
            int rc = kc_channel_submit(child->channel, child);
            if (rc != 0 && !kc_op_is_terminal(child)) {
                (void)kc_channel_cancel_op(child, KC_CAUSE_FAILURE);
            }
            if (atomic_load_explicit(&parent->state, memory_order_acquire) ==
                KC_OP_COMPLETING) break;
        }
    }
    if (!error && atomic_load_explicit(&parent->state, memory_order_acquire) !=
                  KC_OP_COMPLETING) {
        error = kc_op_arm(parent, cancel, deadline_ns);
    }
    if (!error && default_index != SIZE_MAX &&
        atomic_load_explicit(&parent->state, memory_order_acquire) != KC_OP_COMPLETING) {
        parent->select_index = default_index;
        if (kc_op_claim_direct(parent, KC_CAUSE_AGAIN,
                               &(kc_payload){ .status = -EAGAIN })) {
            kc_op_publish(parent);
        }
    }
    if (error && !kc_op_is_terminal(parent) &&
        atomic_load_explicit(&parent->state, memory_order_acquire) != KC_OP_COMPLETING) {
        if (kc_op_claim_direct(parent, KC_CAUSE_FAILURE,
                               &(kc_payload){ .status = error })) {
            kc_op_publish(parent);
        }
    }
    atomic_store_explicit(&parent->select_building, 0, memory_order_release);
    if (atomic_load_explicit(&cont->destroy_requested, memory_order_acquire) &&
        !kc_op_is_terminal(parent)) (void)kc_op_cancel(parent);
    if (atomic_load_explicit(&parent->state, memory_order_acquire) ==
        KC_OP_COMPLETING) kc_op_publish(parent);
    return !kc_op_prepare_suspend(parent);
}

int koro_op_finish(koro_cont_t *cont)
{
    kc_op *op = cont ? cont->arena_op : NULL;
    if (!op) return 1;
    if (!kc_op_is_terminal(op)) return 0;
    cont->last_park_result = op->result.status;
    if (op->kind == KC_OP_RECV || op->kind == KC_OP_SELECT) {
        kc_descriptor_release(cont->arena_descriptor);
        cont->arena_payload = op->result_descriptor ? op->result.ptr : NULL;
        cont->arena_payload_len = op->result_descriptor ? op->result.len : 0;
        cont->arena_desc_id = op->result_descriptor ? op->result.desc_id : 0;
        cont->arena_descriptor = op->result_descriptor;
        op->result_descriptor = NULL;
    }
    if (op->kind == KC_OP_SELECT) cont->select_index = op->select_index;
    cont->arena_op = NULL;
    kc_op_release(op);
    return 1;
}
