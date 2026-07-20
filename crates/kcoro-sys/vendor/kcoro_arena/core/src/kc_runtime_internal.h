// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_runtime.h"
#include "kc_doorbell.h"
#include "kcoro_port.h"
#include "koro_internal.h"

#include <stdatomic.h>

struct kc_service;

enum { KC_RUNTIME_SERVICES_PER_WORKER = 64 };

typedef struct kc_runtime_worker {
    struct kc_runtime *runtime;
    unsigned index;
    KC_THREAD_T thread;
    kc_doorbell_t *idle_doorbell;
    atomic_uint_fast64_t ready_services;
    _Atomic(struct kc_service *) services[KC_RUNTIME_SERVICES_PER_WORKER];
    uint64_t service_slots;
} kc_runtime_worker;

struct kc_runtime {
    atomic_uint refs;
    KC_MUTEX_T mu;
    kc_doorbell_t *lifecycle_doorbell;
    atomic_size_t queued;
    atomic_size_t running;
    atomic_size_t active;
    atomic_uint lifecycle_waiters;
    atomic_uint_fast64_t progress;
    struct kc_service *services_head;
    size_t live_services;
    unsigned worker_count;
    kc_runtime_worker *workers;
    unsigned next_service_owner;
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
/* runtime->mu is held by the caller for bind/unbind. */
int kc_runtime_bind_service_locked_internal(kc_runtime_t *runtime,
                                            struct kc_service *service,
                                            koro_cont_t *continuation);
void kc_runtime_unbind_service_locked_internal(kc_runtime_t *runtime,
                                               struct kc_service *service,
                                               const koro_cont_t *continuation);
void kc_runtime_publish_service_internal(kc_runtime_t *runtime,
                                         const koro_cont_t *continuation);
void kc_runtime_retire_service_internal(kc_runtime_t *runtime,
                                        const koro_cont_t *continuation);
void kc_runtime_ring_workers_internal(kc_runtime_t *runtime);
int kc_runtime_work_realtime_safe_internal(const kc_runtime_t *runtime);
int kc_runtime_is_current_worker_internal(const kc_runtime_t *runtime);
int kc_runtime_is_current_cont_internal(const koro_cont_t *continuation);
void kc_runtime_signal_lifecycle_internal(kc_runtime_t *runtime);
