// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_runtime.h"
#include "kc_descriptor_internal.h"
#include "kcoro_port.h"
#include "kcoro_stackless.h"

#include <stdatomic.h>

struct kc_op;
struct kc_chan;
struct kc_scope;
struct kc_ticket;

struct kc_runtime {
    atomic_uint refs;
    KC_MUTEX_T mu;
    KC_COND_T work_cv;
    KC_COND_T lifecycle_cv;
    koro_cont_t *head;
    koro_cont_t *tail;
    size_t queued;
    size_t running;
    size_t waiting;
    size_t active;
    struct kc_op *ops_head;
    size_t live_operations;
    struct kc_chan *channels_head;
    size_t live_channels;
    struct kc_scope *scopes_head;
    size_t live_scopes;
    struct kc_ticket *tickets;
    struct kc_ticket *completion_head;
    struct kc_ticket *completion_tail;
    uint32_t ticket_capacity;
    uint32_t ticket_free_head;
    size_t live_tickets;
    size_t completion_queued;
    size_t completion_running;
    KC_MUTEX_T timer_mu;
    KC_COND_T timer_cv;
    KC_THREAD_T timer_thread;
    struct kc_op *timer_head;
    size_t live_timers;
    int timer_started;
    int timer_stop;
    unsigned worker_count;
    KC_THREAD_T *workers;
    uint64_t epoch;
    atomic_uint_fast64_t next_sequence;
    size_t arena_segment_size;
    kc_descriptor_registry descriptors;
    int accepting;
    int started;
    int stop_requested;
    int worker_stop;
    int joined;
    int legacy_break;
    atomic_uint_fast64_t wake_requests;
    atomic_uint_fast64_t resumes;
    atomic_uint_fast64_t terminal_causes[KC_CAUSE_FAILURE + 1];
};

kc_runtime_t *kc_runtime_default_get(void);
void kc_runtime_default_clear(kc_runtime_t *runtime);
void kc_runtime_retain_internal(kc_runtime_t *runtime);
void kc_runtime_release_internal(kc_runtime_t *runtime);
int kc_runtime_enqueue_internal(kc_runtime_t *runtime, koro_cont_t *cont,
                                int from_state);
void kc_runtime_wake_internal(koro_cont_t *cont);
void kc_runtime_legacy_break(kc_runtime_t *runtime);
uint64_t kc_runtime_next_sequence(kc_runtime_t *runtime);
void kc_runtime_register_op(kc_runtime_t *runtime, struct kc_op *op);
void kc_runtime_unregister_op(kc_runtime_t *runtime, struct kc_op *op);
void kc_runtime_register_channel(kc_runtime_t *runtime, struct kc_chan *channel);
void kc_runtime_unregister_channel(kc_runtime_t *runtime, struct kc_chan *channel);
void kc_runtime_register_scope(kc_runtime_t *runtime, struct kc_scope *scope);
void kc_runtime_unregister_scope(kc_runtime_t *runtime, struct kc_scope *scope);
int kc_runtime_spawn_internal(kc_runtime_t *runtime, kc_runtime_step_fn step,
                              void *arg, size_t local_size,
                              void (*completion)(void *), void *context);
int kc_runtime_timer_arm(struct kc_op *op, uint64_t deadline_ns);
void kc_runtime_timer_disarm(struct kc_op *op);
