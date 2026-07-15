// SPDX-License-Identifier: BSD-3-Clause
#include "kc_scope_internal.h"
#include "kc_op_internal.h"
#include "kc_runtime_internal.h"

#include <errno.h>
#include <stdlib.h>

static void join_push_locked(kc_scope_t *scope, kc_op *op)
{
    kc_op_retain(op);
    op->prev = scope->join_tail;
    op->next = NULL;
    if (scope->join_tail) scope->join_tail->next = op;
    else scope->join_head = op;
    scope->join_tail = op;
    op->linked = 1;
    atomic_store_explicit(&op->state, KC_OP_WAITING, memory_order_release);
    scope->join_waiters++;
}

static kc_op *join_pop_locked(kc_scope_t *scope)
{
    kc_op *op = scope->join_head;
    if (!op) return NULL;
    scope->join_head = op->next;
    if (scope->join_head) scope->join_head->prev = NULL;
    else scope->join_tail = NULL;
    op->prev = NULL;
    op->next = NULL;
    op->linked = 0;
    if (scope->join_waiters) scope->join_waiters--;
    return op;
}

static void join_remove_locked(kc_scope_t *scope, kc_op *op)
{
    if (op->prev) op->prev->next = op->next;
    else scope->join_head = op->next;
    if (op->next) op->next->prev = op->prev;
    else scope->join_tail = op->prev;
    op->prev = NULL;
    op->next = NULL;
    op->linked = 0;
    if (scope->join_waiters) scope->join_waiters--;
}

void kc_scope_retain(kc_scope_t *scope)
{
    if (scope) atomic_fetch_add_explicit(&scope->refs, 1, memory_order_relaxed);
}

void kc_scope_release(kc_scope_t *scope)
{
    if (!scope) return;
    if (atomic_fetch_sub_explicit(&scope->refs, 1, memory_order_acq_rel) != 1) return;
    kc_runtime_unregister_scope(scope->runtime, scope);
    kc_cancel_release(scope->cancel);
    KC_COND_DESTROY(&scope->cv);
    KC_MUTEX_DESTROY(&scope->mu);
    kc_runtime_release_internal(scope->runtime);
    free(scope);
}

int kc_scope_create(kc_runtime_t *runtime, const kc_scope_config *config,
                    kc_scope_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION) return -EINVAL;
    kc_scope_t *scope = calloc(1, sizeof(*scope));
    if (!scope) return -ENOMEM;
    atomic_init(&scope->refs, 1);
    scope->runtime = runtime;
    scope->id = (kc_id){ runtime->epoch, kc_runtime_next_sequence(runtime) };
    kc_runtime_retain_internal(runtime);
    if (KC_MUTEX_INIT(&scope->mu) != 0) {
        kc_runtime_release_internal(runtime);
        free(scope);
        return -ENOMEM;
    }
    if (KC_COND_INIT(&scope->cv) != 0) {
        KC_MUTEX_DESTROY(&scope->mu);
        kc_runtime_release_internal(runtime);
        free(scope);
        return -ENOMEM;
    }
    int rc = kc_cancel_create(&scope->cancel, config->parent_cancel);
    if (rc != 0) {
        KC_COND_DESTROY(&scope->cv);
        KC_MUTEX_DESTROY(&scope->mu);
        kc_runtime_release_internal(runtime);
        free(scope);
        return rc;
    }
    kc_runtime_register_scope(runtime, scope);
    *out = scope;
    return 0;
}

static void scope_child_done(void *context)
{
    kc_scope_t *scope = context;
    KC_MUTEX_LOCK(&scope->mu);
    if (scope->children) scope->children--;
    int complete = scope->children == 0;
    KC_COND_BROADCAST(&scope->cv);
    KC_MUTEX_UNLOCK(&scope->mu);

    while (complete) {
        KC_MUTEX_LOCK(&scope->mu);
        kc_op *op = join_pop_locked(scope);
        if (op) (void)kc_op_claim_direct(op, KC_CAUSE_MATCH,
                                         &(kc_payload){ .status = 0 });
        KC_MUTEX_UNLOCK(&scope->mu);
        if (!op) break;
        kc_op_publish(op);
        kc_op_release(op);
    }
    kc_scope_release(scope);
}

int kc_scope_spawn(kc_scope_t *scope, kc_runtime_step_fn step, void *arg,
                   size_t local_size)
{
    if (!scope || !step) return -EINVAL;
    KC_MUTEX_LOCK(&scope->mu);
    if (scope->closed || kc_cancel_is_triggered(scope->cancel)) {
        KC_MUTEX_UNLOCK(&scope->mu);
        return -ECANCELED;
    }
    scope->children++;
    kc_scope_retain(scope);
    KC_MUTEX_UNLOCK(&scope->mu);
    int rc = kc_runtime_spawn_internal(scope->runtime, step, arg, local_size,
                                       scope_child_done, scope);
    if (rc != 0) scope_child_done(scope);
    return rc;
}

void kc_scope_close(kc_scope_t *scope)
{
    if (!scope) return;
    KC_MUTEX_LOCK(&scope->mu);
    scope->closed = 1;
    KC_MUTEX_UNLOCK(&scope->mu);
}

void kc_scope_cancel(kc_scope_t *scope)
{
    if (!scope) return;
    kc_scope_close(scope);
    kc_cancel_trigger(scope->cancel);
}

int kc_scope_join(kc_scope_t *scope, uint64_t deadline_ns)
{
    if (!scope) return -EINVAL;
    KC_MUTEX_LOCK(&scope->mu);
    int rc = 0;
    while (scope->children) {
        if (!deadline_ns) {
            KC_COND_WAIT(&scope->cv, &scope->mu);
            continue;
        }
        rc = KC_COND_TIMEDWAIT_NS(&scope->cv, &scope->mu, deadline_ns);
        if (rc == -ETIMEDOUT) break;
    }
    KC_MUTEX_UNLOCK(&scope->mu);
    return rc;
}

kc_cancel_t *kc_scope_cancel_token(kc_scope_t *scope)
{
    return scope ? scope->cancel : NULL;
}

int kc_scope_snapshot_get(kc_scope_t *scope, kc_scope_snapshot *out)
{
    if (!scope || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&scope->mu);
    *out = (kc_scope_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = scope->id, .children = scope->children,
        .join_waiters = scope->join_waiters,
        .closed = (unsigned)scope->closed,
        .canceled = (unsigned)kc_cancel_is_triggered(scope->cancel),
    };
    KC_MUTEX_UNLOCK(&scope->mu);
    return 0;
}

int kc_scope_join_submit(kc_scope_t *scope, kc_op *op)
{
    if (!scope || !op || op->scope != scope) return -EINVAL;
    KC_MUTEX_LOCK(&scope->mu);
    int completed = 0;
    if (scope->children == 0) {
        completed = kc_op_claim_direct(op, KC_CAUSE_MATCH,
                                       &(kc_payload){ .status = 0 });
    } else {
        join_push_locked(scope, op);
    }
    KC_MUTEX_UNLOCK(&scope->mu);
    if (completed) kc_op_publish(op);
    return 0;
}

int kc_scope_cancel_op(kc_op *op, kc_op_cause cause)
{
    if (!op || !op->scope) return -EINVAL;
    kc_scope_t *scope = op->scope;
    int queued = 0;
    KC_MUTEX_LOCK(&scope->mu);
    if (op->linked) {
        join_remove_locked(scope, op);
        queued = 1;
    }
    int status = cause == KC_CAUSE_TIMEOUT ? -ETIMEDOUT
        : cause == KC_CAUSE_FAILURE ? -EIO : -ECANCELED;
    int won = kc_op_claim_direct(op, cause, &(kc_payload){ .status = status });
    KC_MUTEX_UNLOCK(&scope->mu);
    if (won) kc_op_publish(op);
    if (queued) kc_op_release(op);
    return won ? 0 : -EALREADY;
}

int koro_scope_join_begin(koro_cont_t *cont, kc_scope_t *scope,
                          kc_cancel_t *cancel, uint64_t deadline_ns)
{
    if (!cont) return 1;
    if (!scope || scope->runtime != cont->runtime) {
        cont->last_park_result = -EINVAL;
        return 1;
    }
    if (cont->arena_op) return kc_op_is_terminal(cont->arena_op);
    kc_op *op = kc_op_create_internal(cont, NULL, KC_OP_JOIN, NULL);
    if (!op) { cont->last_park_result = -ENOMEM; return 1; }
    op->scope = scope;
    kc_scope_retain(scope);
    cont->arena_op = op;
    cont->suspend_kind = KORO_SUSPEND_WAIT;
    if (atomic_load_explicit(&cont->destroy_requested, memory_order_acquire)) {
        (void)kc_op_cancel(op);
    }
    int rc = kc_op_arm(op, cancel, deadline_ns);
    if (rc == 0 && !kc_op_is_terminal(op)) rc = kc_scope_join_submit(scope, op);
    if (rc != 0 && !kc_op_is_terminal(op)) {
        (void)kc_scope_cancel_op(op, KC_CAUSE_FAILURE);
    }
    return !kc_op_prepare_suspend(op);
}
