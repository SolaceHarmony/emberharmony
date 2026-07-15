// SPDX-License-Identifier: BSD-3-Clause
#include "kc_runtime_internal.h"
#include "kc_op_internal.h"
#include "kc_channel_internal.h"
#include "kc_scope_internal.h"
#include "kc_ticket_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdlib.h>

static kc_runtime_t *default_runtime;
static atomic_uint_fast64_t next_runtime_epoch = ATOMIC_VAR_INIT(1);
enum { KC_COMPLETION_DRAIN_BUDGET = 64 };

static void timer_remove_locked(kc_runtime_t *runtime, kc_op *op)
{
    if (op->timer_prev) op->timer_prev->timer_next = op->timer_next;
    else runtime->timer_head = op->timer_next;
    if (op->timer_next) op->timer_next->timer_prev = op->timer_prev;
    op->timer_prev = NULL;
    op->timer_next = NULL;
    op->timer_linked = 0;
    if (runtime->live_timers) runtime->live_timers--;
}

static void *timer_main(void *arg)
{
    kc_runtime_t *runtime = arg;
    KC_MUTEX_LOCK(&runtime->timer_mu);
    for (;;) {
        while (!runtime->timer_head && !runtime->timer_stop) {
            KC_COND_WAIT(&runtime->timer_cv, &runtime->timer_mu);
        }
        if (runtime->timer_stop && !runtime->timer_head) {
            KC_MUTEX_UNLOCK(&runtime->timer_mu);
            return NULL;
        }
        kc_op *op = runtime->timer_head;
        uint64_t now = kc_port_monotonic_ns();
        if (!runtime->timer_stop && op->deadline_ns > now) {
            (void)KC_COND_TIMEDWAIT_NS(&runtime->timer_cv, &runtime->timer_mu,
                                       op->deadline_ns);
            continue;
        }
        timer_remove_locked(runtime, op);
        KC_MUTEX_UNLOCK(&runtime->timer_mu);
        if (op->kind == KC_OP_TIMER && !runtime->timer_stop) {
            if (kc_op_claim_direct(op, KC_CAUSE_MATCH,
                                   &(kc_payload){ .status = 0 })) kc_op_publish(op);
        } else if (runtime->timer_stop) {
            (void)kc_op_cancel(op);
        } else {
            (void)kc_op_cancel_cause(op, KC_CAUSE_TIMEOUT);
        }
        kc_op_release(op);
        KC_MUTEX_LOCK(&runtime->timer_mu);
    }
}

int kc_runtime_timer_arm(kc_op *op, uint64_t deadline_ns)
{
    if (!op || !deadline_ns) return -EINVAL;
    kc_runtime_t *runtime = op->cont->runtime;
    kc_op_retain(op);
    KC_MUTEX_LOCK(&runtime->timer_mu);
    int rc = 0;
    if (runtime->timer_stop || op->timer_linked) {
        rc = runtime->timer_stop ? -ECANCELED : -EALREADY;
    } else if (!runtime->timer_started) {
        rc = kc_port_thread_create(&runtime->timer_thread, timer_main, runtime);
        if (rc == 0) runtime->timer_started = 1;
    }
    if (rc == 0) {
        op->deadline_ns = deadline_ns;
        kc_op **link = &runtime->timer_head;
        kc_op *previous = NULL;
        while (*link && (*link)->deadline_ns <= deadline_ns) {
            previous = *link;
            link = &(*link)->timer_next;
        }
        op->timer_prev = previous;
        op->timer_next = *link;
        if (*link) (*link)->timer_prev = op;
        *link = op;
        op->timer_linked = 1;
        runtime->live_timers++;
        KC_COND_SIGNAL(&runtime->timer_cv);
    }
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    if (rc != 0) kc_op_release(op);
    return rc;
}

void kc_runtime_timer_disarm(kc_op *op)
{
    if (!op || !op->deadline_ns) return;
    kc_runtime_t *runtime = op->cont->runtime;
    int removed = 0;
    KC_MUTEX_LOCK(&runtime->timer_mu);
    if (op->timer_linked) {
        timer_remove_locked(runtime, op);
        KC_COND_SIGNAL(&runtime->timer_cv);
        removed = 1;
    }
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    if (removed) kc_op_release(op);
}

static void timer_shutdown(kc_runtime_t *runtime)
{
    KC_MUTEX_LOCK(&runtime->timer_mu);
    runtime->timer_stop = 1;
    int started = runtime->timer_started;
    kc_port_thread *thread = runtime->timer_thread;
    KC_COND_BROADCAST(&runtime->timer_cv);
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    if (started) kc_port_thread_join(thread);
    KC_MUTEX_LOCK(&runtime->timer_mu);
    runtime->timer_thread = NULL;
    runtime->timer_started = 0;
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
}

static void runtime_free(kc_runtime_t *runtime)
{
    if (!runtime) return;
    kc_ticket_runtime_destroy(runtime);
    kc_descriptor_runtime_destroy(runtime);
    KC_COND_DESTROY(&runtime->timer_cv);
    KC_MUTEX_DESTROY(&runtime->timer_mu);
    KC_COND_DESTROY(&runtime->lifecycle_cv);
    KC_COND_DESTROY(&runtime->work_cv);
    KC_MUTEX_DESTROY(&runtime->mu);
    free(runtime);
}

void kc_runtime_retain_internal(kc_runtime_t *runtime)
{
    if (runtime) atomic_fetch_add_explicit(&runtime->refs, 1, memory_order_relaxed);
}

void kc_runtime_release_internal(kc_runtime_t *runtime)
{
    if (!runtime) return;
    if (atomic_fetch_sub_explicit(&runtime->refs, 1, memory_order_acq_rel) == 1) {
        runtime_free(runtime);
    }
}

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out)
{
    if (!out) return -EINVAL;
    if (config && (config->size < sizeof(*config) ||
                   config->abi_version != KC_ABI_VERSION)) return -EINVAL;
    kc_runtime_t *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return -ENOMEM;
    atomic_init(&runtime->refs, 1);
    atomic_init(&runtime->next_sequence, 1);
    atomic_init(&runtime->wake_requests, 0);
    atomic_init(&runtime->resumes, 0);
    for (size_t cause = 0; cause <= KC_CAUSE_FAILURE; cause++) {
        atomic_init(&runtime->terminal_causes[cause], 0);
    }
    if (KC_MUTEX_INIT(&runtime->mu) != 0) {
        free(runtime);
        return -ENOMEM;
    }
    if (KC_COND_INIT(&runtime->work_cv) != 0) {
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    if (KC_COND_INIT(&runtime->lifecycle_cv) != 0) {
        KC_COND_DESTROY(&runtime->work_cv);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    runtime->worker_count = config && config->worker_count ? config->worker_count : 1;
    runtime->arena_segment_size = config && config->arena_segment_size
        ? config->arena_segment_size : 1024u * 1024u;
    runtime->epoch = atomic_fetch_add_explicit(&next_runtime_epoch, 1,
                                               memory_order_relaxed);
    uint32_t ticket_capacity = config && config->ticket_capacity
        ? config->ticket_capacity : 256;
    if (kc_ticket_runtime_init(runtime, ticket_capacity) != 0) {
        KC_COND_DESTROY(&runtime->lifecycle_cv);
        KC_COND_DESTROY(&runtime->work_cv);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    if (kc_descriptor_runtime_init(runtime) != 0) {
        kc_ticket_runtime_destroy(runtime);
        KC_COND_DESTROY(&runtime->lifecycle_cv);
        KC_COND_DESTROY(&runtime->work_cv);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    if (KC_MUTEX_INIT(&runtime->timer_mu) != 0) {
        kc_descriptor_runtime_destroy(runtime);
        kc_ticket_runtime_destroy(runtime);
        KC_COND_DESTROY(&runtime->lifecycle_cv);
        KC_COND_DESTROY(&runtime->work_cv);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    if (KC_COND_INIT(&runtime->timer_cv) != 0) {
        KC_MUTEX_DESTROY(&runtime->timer_mu);
        kc_descriptor_runtime_destroy(runtime);
        kc_ticket_runtime_destroy(runtime);
        KC_COND_DESTROY(&runtime->lifecycle_cv);
        KC_COND_DESTROY(&runtime->work_cv);
        KC_MUTEX_DESTROY(&runtime->mu);
        free(runtime);
        return -ENOMEM;
    }
    runtime->accepting = 1;
    *out = runtime;
    return 0;
}

static void queue_locked(kc_runtime_t *runtime, koro_cont_t *cont)
{
    cont->next = NULL;
    if (runtime->tail) runtime->tail->next = cont;
    else runtime->head = cont;
    runtime->tail = cont;
    runtime->queued++;
    KC_COND_SIGNAL(&runtime->work_cv);
}

int kc_runtime_enqueue_internal(kc_runtime_t *runtime, koro_cont_t *cont,
                                int from_state)
{
    if (!runtime || !cont) return 0;
    int expected = from_state;
    if (!atomic_compare_exchange_strong_explicit(&cont->run_state, &expected,
                                                 KORO_QUEUED,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire)) return 0;
    koro_cont_retain(cont);
    KC_MUTEX_LOCK(&runtime->mu);
    if (from_state == KORO_WAITING && runtime->waiting) runtime->waiting--;
    queue_locked(runtime, cont);
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 1;
}

void kc_runtime_wake_internal(koro_cont_t *cont)
{
    if (!cont || !cont->runtime) return;
    kc_runtime_t *runtime = cont->runtime;
    atomic_fetch_add_explicit(&runtime->wake_requests, 1, memory_order_relaxed);
    KC_MUTEX_LOCK(&runtime->mu);
    int state = atomic_load_explicit(&cont->run_state, memory_order_acquire);
    if (state == KORO_RUNNING) {
        atomic_store_explicit(&cont->wake_pending, 1, memory_order_release);
    } else if (state == KORO_WAITING) {
        atomic_store_explicit(&cont->run_state, KORO_QUEUED, memory_order_release);
        atomic_store_explicit(&cont->wake_pending, 0, memory_order_release);
        atomic_fetch_add_explicit(&runtime->resumes, 1, memory_order_relaxed);
        if (runtime->waiting) runtime->waiting--;
        koro_cont_retain(cont);
        queue_locked(runtime, cont);
    }
    KC_MUTEX_UNLOCK(&runtime->mu);
}

static koro_cont_t *dequeue_locked(kc_runtime_t *runtime)
{
    koro_cont_t *cont = runtime->head;
    if (!cont) return NULL;
    runtime->head = cont->next;
    if (!runtime->head) runtime->tail = NULL;
    cont->next = NULL;
    if (runtime->queued) runtime->queued--;
    runtime->running++;
    return cont;
}

static void finish_cont(kc_runtime_t *runtime, koro_cont_t *cont)
{
    kc_op *op = cont->arena_op;
    if (op) {
        cont->arena_op = NULL;
        (void)kc_op_cancel(op);
        kc_op_release(op);
    }
    atomic_store_explicit(&cont->run_state, KORO_DONE, memory_order_release);
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->running) runtime->running--;
    if (cont->tracked) {
        cont->tracked = 0;
        if (runtime->active) runtime->active--;
    }
    int managed = cont->managed;
    void (*completion)(void *) = cont->completion;
    void *context = cont->completion_context;
    cont->completion = NULL;
    cont->completion_context = NULL;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (completion) completion(context);
    if (managed) koro_cont_release_internal(cont);
}

static void suspend_cont(kc_runtime_t *runtime, koro_cont_t *cont)
{
    KC_MUTEX_LOCK(&runtime->mu);
    atomic_store_explicit(&cont->run_state, KORO_WAITING, memory_order_release);
    if (runtime->running) runtime->running--;
    runtime->waiting++;
    if (cont->suspend_kind == KORO_SUSPEND_YIELD ||
        atomic_exchange_explicit(&cont->wake_pending, 0, memory_order_acq_rel)) {
        atomic_store_explicit(&cont->run_state, KORO_QUEUED, memory_order_release);
        runtime->waiting--;
        koro_cont_retain(cont);
        queue_locked(runtime, cont);
    }
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

static void *worker_main(void *arg)
{
    kc_runtime_t *runtime = arg;
    unsigned completion_streak = 0;
    for (;;) {
        KC_MUTEX_LOCK(&runtime->mu);
        while (!runtime->head && !runtime->completion_head &&
               !runtime->worker_stop) {
            KC_COND_WAIT(&runtime->work_cv, &runtime->mu);
        }
        if (runtime->worker_stop && !runtime->head &&
            !runtime->completion_head) {
            KC_MUTEX_UNLOCK(&runtime->mu);
            return NULL;
        }
        kc_ticket_t *ticket = NULL;
        if (runtime->completion_head &&
            (!runtime->head || completion_streak < KC_COMPLETION_DRAIN_BUDGET))
            ticket = kc_ticket_runtime_dequeue_locked(runtime);
        if (ticket) {
            KC_MUTEX_UNLOCK(&runtime->mu);
            kc_ticket_runtime_deliver(ticket);
            completion_streak++;
            continue;
        }
        koro_cont_t *cont = dequeue_locked(runtime);
        if (cont) completion_streak = 0;
        KC_MUTEX_UNLOCK(&runtime->mu);
        if (!cont) continue;

        int expected = KORO_QUEUED;
        if (!atomic_compare_exchange_strong_explicit(&cont->run_state, &expected,
                                                     KORO_RUNNING,
                                                     memory_order_acq_rel,
                                                     memory_order_acquire)) {
            KC_MUTEX_LOCK(&runtime->mu);
            if (runtime->running) runtime->running--;
            KC_COND_BROADCAST(&runtime->lifecycle_cv);
            KC_MUTEX_UNLOCK(&runtime->mu);
            koro_cont_release_internal(cont);
            continue;
        }
        atomic_store_explicit(&cont->wake_pending, 0, memory_order_release);
        cont->suspend_kind = KORO_SUSPEND_WAIT;
        void *result = atomic_load_explicit(&cont->destroy_requested,
                                            memory_order_acquire)
            ? (void *)1 : koro_cont_step(cont);
        if (result || cont->completed) finish_cont(runtime, cont);
        else suspend_cont(runtime, cont);
        koro_cont_release_internal(cont);
    }
}

int kc_runtime_start(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->started) { KC_MUTEX_UNLOCK(&runtime->mu); return 0; }
    runtime->workers = calloc(runtime->worker_count, sizeof(*runtime->workers));
    if (!runtime->workers) { KC_MUTEX_UNLOCK(&runtime->mu); return -ENOMEM; }
    runtime->worker_stop = 0;
    runtime->joined = 0;
    KC_MUTEX_UNLOCK(&runtime->mu);

    unsigned started = 0;
    for (; started < runtime->worker_count; started++) {
        if (kc_port_thread_create(&runtime->workers[started], worker_main,
                                  runtime) != 0) break;
    }
    if (started != runtime->worker_count) {
        KC_MUTEX_LOCK(&runtime->mu);
        runtime->worker_stop = 1;
        KC_COND_BROADCAST(&runtime->work_cv);
        KC_MUTEX_UNLOCK(&runtime->mu);
        for (unsigned i = 0; i < started; i++) kc_port_thread_join(runtime->workers[i]);
        free(runtime->workers);
        runtime->workers = NULL;
        return -EAGAIN;
    }
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->started = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_spawn(kc_runtime_t *runtime, kc_runtime_step_fn step,
                     void *arg, size_t local_size)
{
    return kc_runtime_spawn_internal(runtime, step, arg, local_size, NULL, NULL);
}

int kc_runtime_spawn_internal(kc_runtime_t *runtime, kc_runtime_step_fn step,
                              void *arg, size_t local_size,
                              void (*completion)(void *), void *context)
{
    if (!runtime || !step) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    int accepting = runtime->accepting;
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (!accepting) return -ECANCELED;
    koro_cont_t *cont = koro_cont_create_on(runtime, step, arg, local_size);
    if (!cont) return -ENOMEM;
    cont->managed = 1;
    cont->completion = completion;
    cont->completion_context = context;
    KC_MUTEX_LOCK(&runtime->mu);
    cont->tracked = 1;
    runtime->active++;
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (!kc_runtime_enqueue_internal(runtime, cont, KORO_NEW)) {
        KC_MUTEX_LOCK(&runtime->mu);
        cont->tracked = 0;
        runtime->active--;
        KC_MUTEX_UNLOCK(&runtime->mu);
        koro_cont_destroy(cont);
        return -EAGAIN;
    }
    return 0;
}

int kc_runtime_run_until_idle(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    int rc = kc_runtime_start(runtime);
    if (rc != 0) return rc;
    KC_MUTEX_LOCK(&runtime->mu);
    while ((runtime->queued || runtime->running || runtime->completion_queued ||
            runtime->completion_running) && !runtime->legacy_break) {
        KC_COND_WAIT(&runtime->lifecycle_cv, &runtime->mu);
    }
    runtime->legacy_break = 0;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_join_all(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    int rc = kc_runtime_start(runtime);
    if (rc != 0) return rc;
    KC_MUTEX_LOCK(&runtime->mu);
    while ((runtime->active || runtime->live_operations ||
            runtime->live_tickets || runtime->completion_queued ||
            runtime->completion_running) &&
           !runtime->legacy_break) {
        KC_COND_WAIT(&runtime->lifecycle_cv, &runtime->mu);
    }
    runtime->legacy_break = 0;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

void kc_runtime_request_stop(kc_runtime_t *runtime)
{
    if (!runtime) return;
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->accepting = 0;
    runtime->stop_requested = 1;
    kc_ticket_runtime_stop_locked(runtime);
    KC_COND_BROADCAST(&runtime->work_cv);
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    for (;;) {
        KC_MUTEX_LOCK(&runtime->mu);
        kc_op *op = runtime->ops_head;
        while (op && kc_op_is_terminal(op)) op = op->registry_next;
        if (op) kc_op_retain(op);
        KC_MUTEX_UNLOCK(&runtime->mu);
        if (!op) break;
        (void)kc_op_cancel(op);
        kc_op_release(op);
    }
}

int kc_runtime_join(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->active || runtime->queued || runtime->running ||
        runtime->live_tickets || runtime->completion_queued ||
        runtime->completion_running) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (!runtime->started || runtime->joined) {
        runtime->joined = 1;
        KC_MUTEX_UNLOCK(&runtime->mu);
        timer_shutdown(runtime);
        return 0;
    }
    runtime->worker_stop = 1;
    KC_COND_BROADCAST(&runtime->work_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
    for (unsigned i = 0; i < runtime->worker_count; i++) {
        kc_port_thread_join(runtime->workers[i]);
    }
    free(runtime->workers);
    runtime->workers = NULL;
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->started = 0;
    runtime->joined = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    timer_shutdown(runtime);
    return 0;
}

int kc_runtime_destroy(kc_runtime_t *runtime)
{
    if (!runtime) return 0;
    KC_MUTEX_LOCK(&runtime->mu);
    int busy = runtime->active || runtime->queued || runtime->running ||
               runtime->waiting || runtime->live_operations ||
               runtime->live_channels || runtime->live_scopes ||
               runtime->live_tickets || runtime->completion_queued ||
               runtime->completion_running ||
               (runtime->started && !runtime->joined);
    KC_MUTEX_UNLOCK(&runtime->mu);
    KC_MUTEX_LOCK(&runtime->descriptors.mu);
    busy = busy || runtime->descriptors.live_descriptors ||
           runtime->descriptors.live_regions;
    KC_MUTEX_UNLOCK(&runtime->descriptors.mu);
    KC_MUTEX_LOCK(&runtime->timer_mu);
    busy = busy || runtime->live_timers || runtime->timer_started;
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    if (busy) return -EBUSY;
    kc_runtime_default_clear(runtime);
    kc_runtime_release_internal(runtime);
    return 0;
}

int kc_runtime_snapshot_get(kc_runtime_t *runtime, kc_runtime_snapshot *out)
{
    if (!runtime || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    *out = (kc_runtime_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .epoch = runtime->epoch,
        .next_sequence = atomic_load_explicit(&runtime->next_sequence,
                                              memory_order_relaxed),
        .active = runtime->active, .queued = runtime->queued,
        .running = runtime->running, .waiting = runtime->waiting,
        .live_operations = runtime->live_operations,
        .live_channels = runtime->live_channels,
        .live_scopes = runtime->live_scopes,
        .live_tickets = runtime->live_tickets,
        .completion_queued = runtime->completion_queued,
        .completion_running = runtime->completion_running,
        .workers = runtime->worker_count, .accepting = (unsigned)runtime->accepting,
        .started = (unsigned)runtime->started,
        .stop_requested = (unsigned)runtime->stop_requested,
        .ticket_capacity = runtime->ticket_capacity,
        .wake_requests = atomic_load_explicit(&runtime->wake_requests,
                                              memory_order_relaxed),
        .resumes = atomic_load_explicit(&runtime->resumes, memory_order_relaxed),
    };
    KC_MUTEX_UNLOCK(&runtime->mu);
    KC_MUTEX_LOCK(&runtime->timer_mu);
    out->live_timers = runtime->live_timers;
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    KC_MUTEX_LOCK(&runtime->descriptors.mu);
    out->live_descriptors = runtime->descriptors.live_descriptors;
    out->live_regions = runtime->descriptors.live_regions;
    out->live_segments = runtime->descriptors.live_segments;
    out->reserved_bytes = runtime->descriptors.reserved_bytes;
    KC_MUTEX_UNLOCK(&runtime->descriptors.mu);
    return 0;
}

void kc_runtime_legacy_break(kc_runtime_t *runtime)
{
    if (!runtime) return;
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->legacy_break = 1;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

uint64_t kc_runtime_next_sequence(kc_runtime_t *runtime)
{
    return runtime ? atomic_fetch_add_explicit(&runtime->next_sequence, 1,
                                               memory_order_relaxed) : 0;
}

void kc_runtime_register_op(kc_runtime_t *runtime, kc_op *op)
{
    KC_MUTEX_LOCK(&runtime->mu);
    op->registry_next = runtime->ops_head;
    if (runtime->ops_head) runtime->ops_head->registry_prev = op;
    runtime->ops_head = op;
    runtime->live_operations++;
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_runtime_unregister_op(kc_runtime_t *runtime, kc_op *op)
{
    KC_MUTEX_LOCK(&runtime->mu);
    if (op->registry_prev) op->registry_prev->registry_next = op->registry_next;
    else if (runtime->ops_head == op) runtime->ops_head = op->registry_next;
    if (op->registry_next) op->registry_next->registry_prev = op->registry_prev;
    if (runtime->live_operations) runtime->live_operations--;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_runtime_register_channel(kc_runtime_t *runtime, struct kc_chan *channel)
{
    KC_MUTEX_LOCK(&runtime->mu);
    channel->registry_next = runtime->channels_head;
    if (runtime->channels_head) runtime->channels_head->registry_prev = channel;
    runtime->channels_head = channel;
    runtime->live_channels++;
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_runtime_unregister_channel(kc_runtime_t *runtime, struct kc_chan *channel)
{
    KC_MUTEX_LOCK(&runtime->mu);
    if (channel->registry_prev) channel->registry_prev->registry_next = channel->registry_next;
    else if (runtime->channels_head == channel) runtime->channels_head = channel->registry_next;
    if (channel->registry_next) channel->registry_next->registry_prev = channel->registry_prev;
    if (runtime->live_channels) runtime->live_channels--;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_runtime_register_scope(kc_runtime_t *runtime, struct kc_scope *scope)
{
    KC_MUTEX_LOCK(&runtime->mu);
    scope->registry_next = runtime->scopes_head;
    if (runtime->scopes_head) runtime->scopes_head->registry_prev = scope;
    runtime->scopes_head = scope;
    runtime->live_scopes++;
    KC_MUTEX_UNLOCK(&runtime->mu);
}

void kc_runtime_unregister_scope(kc_runtime_t *runtime, struct kc_scope *scope)
{
    KC_MUTEX_LOCK(&runtime->mu);
    if (scope->registry_prev) scope->registry_prev->registry_next = scope->registry_next;
    else if (runtime->scopes_head == scope) runtime->scopes_head = scope->registry_next;
    if (scope->registry_next) scope->registry_next->registry_prev = scope->registry_prev;
    if (runtime->live_scopes) runtime->live_scopes--;
    KC_COND_BROADCAST(&runtime->lifecycle_cv);
    KC_MUTEX_UNLOCK(&runtime->mu);
}

kc_runtime_t *kc_runtime_default_get(void)
{
    if (default_runtime) return default_runtime;
    kc_runtime_config config = {
        .size = sizeof(config), .abi_version = KC_ABI_VERSION,
        .worker_count = 1,
    };
    if (kc_runtime_create(&config, &default_runtime) != 0) return NULL;
    return default_runtime;
}

void kc_runtime_default_clear(kc_runtime_t *runtime)
{
    if (default_runtime == runtime) default_runtime = NULL;
}
