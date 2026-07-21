// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_runtime.h"
#include "kc_doorbell.h"
#include "kcoro_port.h"
#include "koro_internal.h"

#include <stdatomic.h>

struct kc_service;

enum { KC_RUNTIME_CONTINUATIONS_PER_WORKER = 64 };

typedef struct kc_runtime_worker {
    struct kc_runtime *runtime;
    unsigned index;
    KC_THREAD_T thread;
} kc_runtime_worker;

struct kc_runtime {
    atomic_uint refs;
    KC_MUTEX_T mu;
    kc_doorbell_t *lifecycle_doorbell;
    kc_doorbell_t *work_doorbell;
    atomic_size_t queued;
    atomic_size_t running;
    atomic_size_t active;
    atomic_uint lifecycle_waiters;
    atomic_uint_fast64_t progress;
    struct kc_service *services_head;
    size_t live_services;
    size_t live_continuations;
    unsigned worker_count;
    kc_runtime_worker *workers;
    size_t continuation_capacity;
    size_t ready_word_count;
    _Atomic(koro_cont_t *) *continuations;
    atomic_uint_fast64_t *ready_words;
    uint32_t *slot_generations;
    atomic_uint next_ready_word;
    atomic_uint next_affinity_worker;
    uint64_t runtime_epoch;
    atomic_uint_fast64_t next_sequence;
    int accepting;
    int starting;
    int started;
    int stop_requested;
    atomic_uint worker_stop;
    int joining;
    int joined;
    atomic_uint_fast64_t wake_requests;
    atomic_uint_fast64_t resumes;
};

void kc_runtime_retain_internal(kc_runtime_t *runtime);
void kc_runtime_release_internal(kc_runtime_t *runtime);
int kc_runtime_register_continuation_internal(kc_runtime_t *runtime,
                                              koro_cont_t *continuation);
int kc_runtime_unregister_continuation_internal(kc_runtime_t *runtime,
                                                koro_cont_t *continuation);
int kc_runtime_start_continuation_internal(koro_cont_t *continuation);
int kc_runtime_resume_continuation_internal(koro_cont_t *continuation);
void kc_runtime_publish_service_internal(kc_runtime_t *runtime,
                                         const koro_cont_t *continuation);
void kc_runtime_retire_service_internal(kc_runtime_t *runtime,
                                        const koro_cont_t *continuation);
void kc_runtime_ring_workers_internal(kc_runtime_t *runtime);
int kc_runtime_work_realtime_safe_internal(const kc_runtime_t *runtime);
int kc_runtime_is_current_worker_internal(const kc_runtime_t *runtime);
int kc_runtime_is_current_cont_internal(const koro_cont_t *continuation);
uint64_t kc_runtime_affinity_mask_internal(kc_runtime_t *runtime);
void kc_runtime_signal_lifecycle_internal(kc_runtime_t *runtime);
