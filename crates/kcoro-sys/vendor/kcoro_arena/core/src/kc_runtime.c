// SPDX-License-Identifier: BSD-3-Clause
#include "kc_runtime_internal.h"
#include "kc_service_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

static atomic_uint_fast64_t next_runtime_epoch = ATOMIC_VAR_INIT(1);
static _Thread_local kc_runtime_t *current_worker_runtime;
static _Thread_local koro_cont_t *current_worker_continuation;
static _Thread_local uint32_t current_worker_index = UINT32_MAX;

static uint64_t next_epoch(void)
{
    uint64_t epoch = atomic_fetch_add_explicit(&next_runtime_epoch, 1,
                                                memory_order_relaxed);
    if (epoch == 0)
        epoch = atomic_fetch_add_explicit(&next_runtime_epoch, 1,
                                          memory_order_relaxed);
    return epoch;
}

static void runtime_free(kc_runtime_t *runtime)
{
    if (!runtime) return;
    free(runtime->slot_generations);
    free(runtime->ready_words);
    free(runtime->continuations);
    free(runtime->workers);
    kc_doorbell_destroy(runtime->work_doorbell);
    kc_doorbell_destroy(runtime->lifecycle_doorbell);
    KC_MUTEX_DESTROY(&runtime->mu);
    free(runtime);
}

void kc_runtime_retain_internal(kc_runtime_t *runtime)
{
    if (runtime)
        atomic_fetch_add_explicit(&runtime->refs, 1, memory_order_relaxed);
}

void kc_runtime_release_internal(kc_runtime_t *runtime)
{
    if (!runtime) return;
    if (atomic_fetch_sub_explicit(&runtime->refs, 1,
                                  memory_order_acq_rel) == 1)
        runtime_free(runtime);
}

static int create_workers(kc_runtime_t *runtime)
{
    runtime->workers = calloc(runtime->worker_count, sizeof(*runtime->workers));
    if (!runtime->workers) return -ENOMEM;
    for (unsigned index = 0; index < runtime->worker_count; ++index) {
        runtime->workers[index].runtime = runtime;
        runtime->workers[index].index = index;
    }
    return 0;
}

static int create_continuation_board(kc_runtime_t *runtime)
{
    if (runtime->worker_count > 64) return -EINVAL;
    runtime->continuation_capacity =
        (size_t)runtime->worker_count * KC_RUNTIME_CONTINUATIONS_PER_WORKER;
    runtime->ready_word_count =
        (runtime->continuation_capacity + 63) / 64;
    runtime->continuations = calloc(runtime->continuation_capacity,
                                    sizeof(*runtime->continuations));
    runtime->ready_words = calloc(runtime->ready_word_count,
                                  sizeof(*runtime->ready_words));
    runtime->slot_generations = calloc(runtime->continuation_capacity,
                                       sizeof(*runtime->slot_generations));
    if (!runtime->continuations || !runtime->ready_words ||
        !runtime->slot_generations) return -ENOMEM;
    for (size_t slot = 0; slot < runtime->continuation_capacity; ++slot)
        atomic_init(&runtime->continuations[slot], NULL);
    for (size_t word = 0; word < runtime->ready_word_count; ++word)
        atomic_init(&runtime->ready_words[word], 0);
    return 0;
}

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out)
{
    if (!out) return -EINVAL;
    if (config && (config->size < sizeof(*config) ||
                   config->abi_version != KC_ABI_VERSION ||
                   config->reserved != 0)) return -EINVAL;
    kc_runtime_t *runtime = calloc(1, sizeof(*runtime));
    if (!runtime) return -ENOMEM;
    atomic_init(&runtime->refs, 1);
    atomic_init(&runtime->wake_requests, 0);
    atomic_init(&runtime->resumes, 0);
    atomic_init(&runtime->queued, 0);
    atomic_init(&runtime->running, 0);
    atomic_init(&runtime->active, 0);
    atomic_init(&runtime->lifecycle_waiters, 0);
    atomic_init(&runtime->progress, 1);
    atomic_init(&runtime->worker_stop, 0);
    atomic_init(&runtime->next_ready_word, 0);
    atomic_init(&runtime->next_affinity_worker, 0);
    atomic_init(&runtime->next_sequence, 1);
    runtime->runtime_epoch = next_epoch();
    runtime->worker_count = config && config->worker_count
        ? config->worker_count : 1;
    if (runtime->worker_count > 64) {
        free(runtime);
        return -EINVAL;
    }
    if (KC_MUTEX_INIT(&runtime->mu) != 0) {
        free(runtime);
        return -ENOMEM;
    }
    int status = kc_doorbell_create(&runtime->lifecycle_doorbell);
    if (status == 0) status = kc_doorbell_create(&runtime->work_doorbell);
    if (status == 0) status = create_workers(runtime);
    if (status == 0) status = create_continuation_board(runtime);
    if (status != 0) {
        runtime_free(runtime);
        return status;
    }
    runtime->accepting = 1;
    *out = runtime;
    return 0;
}

void kc_runtime_signal_lifecycle_internal(kc_runtime_t *runtime)
{
    if (!runtime) return;
    atomic_fetch_add_explicit(&runtime->progress, 1, memory_order_release);
    if (atomic_load_explicit(&runtime->lifecycle_waiters,
                             memory_order_seq_cst) != 0)
        kc_doorbell_ring_all(runtime->lifecycle_doorbell);
}

void kc_runtime_ring_workers_internal(kc_runtime_t *runtime)
{
    if (runtime) kc_doorbell_ring_all(runtime->work_doorbell);
}

int kc_runtime_work_realtime_safe_internal(const kc_runtime_t *runtime)
{
    return runtime && kc_doorbell_realtime_safe(runtime->work_doorbell);
}

int kc_runtime_is_current_worker_internal(const kc_runtime_t *runtime)
{
    return runtime && current_worker_runtime == runtime;
}

int kc_runtime_is_current_cont_internal(const koro_cont_t *continuation)
{
    return continuation && current_worker_continuation == continuation;
}

int kc_runtime_current_worker(const kc_runtime_t *runtime,
                              uint32_t *out_worker)
{
    if (!runtime || !out_worker) return -EINVAL;
    if (current_worker_runtime != runtime || current_worker_index == UINT32_MAX)
        return -EPERM;
    *out_worker = current_worker_index;
    return 0;
}

uint64_t kc_runtime_affinity_mask_internal(kc_runtime_t *runtime)
{
    if (!runtime || !runtime->worker_count) return 0;
    const unsigned worker = atomic_fetch_add_explicit(
        &runtime->next_affinity_worker, 1, memory_order_relaxed) %
        runtime->worker_count;
    return UINT64_C(1) << worker;
}

int kc_runtime_register_continuation_internal(kc_runtime_t *runtime,
                                              koro_cont_t *continuation)
{
    if (!runtime || !continuation) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    size_t slot = 0;
    while (slot < runtime->continuation_capacity &&
           atomic_load_explicit(&runtime->continuations[slot],
                                memory_order_acquire) != NULL) ++slot;
    if (slot == runtime->continuation_capacity) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ENOSPC;
    }
    uint32_t generation = ++runtime->slot_generations[slot];
    if (generation == 0) generation = ++runtime->slot_generations[slot];
    uint64_t sequence = atomic_fetch_add_explicit(&runtime->next_sequence, 1,
                                                   memory_order_relaxed);
    if (sequence == 0)
        sequence = atomic_fetch_add_explicit(&runtime->next_sequence, 1,
                                             memory_order_relaxed);
    continuation->slot = (uint32_t)slot;
    continuation->identity = (kc_ticket_id){
        .runtime_epoch = runtime->runtime_epoch,
        .sequence = sequence,
        .generation = generation,
        .kind = KC_TICKET_KIND_CONTROL,
    };
    continuation->registered = 1;
    atomic_store_explicit(&runtime->continuations[slot], continuation,
                          memory_order_release);
    runtime->live_continuations++;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

static void clear_ready(kc_runtime_t *runtime, uint32_t slot)
{
    const size_t word = slot / 64;
    const uint64_t bit = UINT64_C(1) << (slot % 64);
    const uint64_t prior = atomic_fetch_and_explicit(
        &runtime->ready_words[word], ~bit, memory_order_acq_rel);
    if (prior & bit)
        atomic_fetch_sub_explicit(&runtime->queued, 1,
                                  memory_order_relaxed);
}

int kc_runtime_unregister_continuation_internal(kc_runtime_t *runtime,
                                                koro_cont_t *continuation)
{
    if (!runtime || !continuation || !continuation->registered ||
        continuation->slot >= runtime->continuation_capacity) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    koro_cont_t *current = atomic_load_explicit(
        &runtime->continuations[continuation->slot], memory_order_acquire);
    if (current != continuation) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ESTALE;
    }
    clear_ready(runtime, continuation->slot);
    atomic_store_explicit(&runtime->continuations[continuation->slot], NULL,
                          memory_order_release);
    continuation->registered = 0;
    if (runtime->live_continuations) runtime->live_continuations--;
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_signal_lifecycle_internal(runtime);
    return 0;
}

static void publish_ready(kc_runtime_t *runtime, koro_cont_t *continuation)
{
    if (!runtime || !continuation || !continuation->registered) return;
    const size_t word = continuation->slot / 64;
    const uint64_t bit = UINT64_C(1) << (continuation->slot % 64);
    const uint64_t prior = atomic_fetch_or_explicit(
        &runtime->ready_words[word], bit, memory_order_acq_rel);
    if (prior & bit) return;
    atomic_fetch_add_explicit(&runtime->queued, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&runtime->progress, 1, memory_order_release);
    if (continuation->worker_mask)
        kc_doorbell_ring_all(runtime->work_doorbell);
    else
        kc_doorbell_ring_one(runtime->work_doorbell);
}

int kc_runtime_start_continuation_internal(koro_cont_t *continuation)
{
    if (!continuation || !continuation->runtime ||
        !continuation->registered) return -EINVAL;
    kc_runtime_t *runtime = continuation->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    int expected = KORO_NEW;
    if (!atomic_compare_exchange_strong_explicit(
            &continuation->run_state, &expected, KORO_QUEUED,
            memory_order_acq_rel, memory_order_acquire)) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return expected == KORO_QUEUED || expected == KORO_RUNNING ||
                       expected == KORO_SUSPENDED
                   ? 0 : -ECANCELED;
    }
    continuation->tracked = 1;
    atomic_fetch_add_explicit(&runtime->active, 1, memory_order_relaxed);
    KC_MUTEX_UNLOCK(&runtime->mu);
    publish_ready(runtime, continuation);
    return 0;
}

int kc_runtime_resume_continuation_internal(koro_cont_t *continuation)
{
    if (!continuation || !continuation->runtime ||
        !continuation->registered) return -EINVAL;
    kc_runtime_t *runtime = continuation->runtime;
    atomic_fetch_add_explicit(&runtime->wake_requests, 1,
                              memory_order_relaxed);
    atomic_store_explicit(&continuation->wake_pending, 1,
                          memory_order_release);
    int state = atomic_load_explicit(&continuation->run_state,
                                     memory_order_acquire);
    if (state == KORO_DONE) return -ECANCELED;
    if (state != KORO_SUSPENDED) return 0;
    int expected = KORO_SUSPENDED;
    if (!atomic_compare_exchange_strong_explicit(
            &continuation->run_state, &expected, KORO_QUEUED,
            memory_order_acq_rel, memory_order_acquire)) return 0;
    atomic_store_explicit(&continuation->wake_pending, 0,
                          memory_order_relaxed);
    atomic_fetch_add_explicit(&runtime->resumes, 1, memory_order_relaxed);
    publish_ready(runtime, continuation);
    return 0;
}

void kc_runtime_publish_service_internal(kc_runtime_t *runtime,
                                         const koro_cont_t *continuation)
{
    if (!runtime || !continuation || continuation->runtime != runtime) return;
    (void)kc_runtime_resume_continuation_internal((koro_cont_t *)continuation);
}

void kc_runtime_retire_service_internal(kc_runtime_t *runtime,
                                        const koro_cont_t *continuation)
{
    if (runtime && continuation && continuation->registered)
        clear_ready(runtime, continuation->slot);
}

static koro_cont_t *claim_ready(kc_runtime_worker *worker)
{
    kc_runtime_t *runtime = worker->runtime;
    const size_t words = runtime->ready_word_count;
    const size_t start = atomic_fetch_add_explicit(
        &runtime->next_ready_word, 1, memory_order_relaxed) % words;
    for (size_t offset = 0; offset < words; ++offset) {
        const size_t word = (start + offset) % words;
        uint64_t ready = atomic_load_explicit(&runtime->ready_words[word],
                                              memory_order_acquire);
        while (ready) {
            const unsigned bit_index = (unsigned)__builtin_ctzll(ready);
            const uint64_t bit = UINT64_C(1) << bit_index;
            const size_t slot = word * 64 + bit_index;
            if (slot >= runtime->continuation_capacity) break;
            koro_cont_t *continuation = atomic_load_explicit(
                &runtime->continuations[slot], memory_order_acquire);
            if (!continuation) {
                atomic_fetch_and_explicit(&runtime->ready_words[word], ~bit,
                                          memory_order_acq_rel);
                ready &= ~bit;
                continue;
            }
            if (continuation->worker_mask &&
                !(continuation->worker_mask &
                  (UINT64_C(1) << worker->index))) {
                ready &= ~bit;
                continue;
            }
            const uint64_t prior = atomic_fetch_and_explicit(
                &runtime->ready_words[word], ~bit, memory_order_acq_rel);
            if (!(prior & bit)) {
                ready = prior & ~bit;
                continue;
            }
            atomic_fetch_sub_explicit(&runtime->queued, 1,
                                      memory_order_relaxed);
            int expected = KORO_QUEUED;
            if (!atomic_compare_exchange_strong_explicit(
                    &continuation->run_state, &expected, KORO_RUNNING,
                    memory_order_acq_rel, memory_order_acquire)) {
                ready = prior & ~bit;
                continue;
            }
            koro_cont_retain_internal(continuation);
            return continuation;
        }
    }
    return NULL;
}

static void execute_continuation(kc_runtime_worker *worker,
                                 koro_cont_t *continuation)
{
    kc_runtime_t *runtime = worker->runtime;
    atomic_fetch_add_explicit(&runtime->running, 1, memory_order_relaxed);
    atomic_store_explicit(&continuation->current_worker, worker->index,
                          memory_order_release);
    atomic_store_explicit(&continuation->wake_pending, 0,
                          memory_order_relaxed);
    current_worker_continuation = continuation;
    void *result = koro_cont_step(continuation);
    current_worker_continuation = NULL;
    atomic_store_explicit(&continuation->current_worker, UINT32_MAX,
                          memory_order_release);
    atomic_fetch_sub_explicit(&runtime->running, 1, memory_order_relaxed);

    if (result || continuation->completed) {
        if (continuation->tracked) {
            continuation->tracked = 0;
            atomic_fetch_sub_explicit(&runtime->active, 1,
                                      memory_order_relaxed);
        }
        atomic_store_explicit(&continuation->run_state, KORO_DONE,
                              memory_order_release);
        koro_completion_fn completion = continuation->completion;
        void *context = continuation->completion_context;
        const kc_ticket_id identity = continuation->identity;
        if (completion) completion(context, &identity);
        kc_runtime_signal_lifecycle_internal(runtime);
        koro_cont_release_internal(continuation);
        return;
    }

    int expected = KORO_RUNNING;
    if (!atomic_compare_exchange_strong_explicit(
            &continuation->run_state, &expected, KORO_SUSPENDED,
            memory_order_acq_rel, memory_order_acquire)) abort();
    const int ready = continuation->suspend_kind == KORO_SUSPEND_YIELD ||
        atomic_exchange_explicit(&continuation->wake_pending, 0,
                                 memory_order_acq_rel) != 0;
    if (ready) {
        expected = KORO_SUSPENDED;
        if (atomic_compare_exchange_strong_explicit(
                &continuation->run_state, &expected, KORO_QUEUED,
                memory_order_acq_rel, memory_order_acquire))
            publish_ready(runtime, continuation);
    }
    kc_runtime_signal_lifecycle_internal(runtime);
    koro_cont_release_internal(continuation);
}

static void *worker_main(void *argument)
{
    kc_runtime_worker *worker = argument;
    kc_runtime_t *runtime = worker->runtime;
    current_worker_runtime = runtime;
    current_worker_index = worker->index;
    for (;;) {
        /* The bounded pool has no operation-owned waiter.  A worker that finds
         * the entire runnable board empty dehydrates itself on the runtime's
         * one infrastructure doorbell; any callback may make an exact frame
         * runnable and ring that doorbell. */
        const uint32_t observed = kc_doorbell_observe(runtime->work_doorbell);
        koro_cont_t *continuation = claim_ready(worker);
        if (continuation) {
            execute_continuation(worker, continuation);
            continue;
        }
        if (atomic_load_explicit(&runtime->worker_stop,
                                 memory_order_acquire) != 0) {
            current_worker_index = UINT32_MAX;
            current_worker_runtime = NULL;
            return NULL;
        }
        continuation = claim_ready(worker);
        if (continuation) {
            execute_continuation(worker, continuation);
            continue;
        }
        if (kc_doorbell_park(runtime->work_doorbell, observed) != 0) abort();
    }
}

int kc_runtime_start(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    if (runtime->starting || runtime->joining) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (runtime->started) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    runtime->starting = 1;
    atomic_store_explicit(&runtime->worker_stop, 0, memory_order_release);
    runtime->joined = 0;
    KC_MUTEX_UNLOCK(&runtime->mu);

    unsigned started = 0;
    for (; started < runtime->worker_count; ++started) {
        if (kc_port_thread_create(&runtime->workers[started].thread,
                                  worker_main,
                                  &runtime->workers[started]) != 0) break;
    }
    if (started != runtime->worker_count) {
        atomic_store_explicit(&runtime->worker_stop, 1, memory_order_release);
        kc_runtime_ring_workers_internal(runtime);
        for (unsigned index = 0; index < started; ++index)
            kc_port_thread_join(runtime->workers[index].thread);
        KC_MUTEX_LOCK(&runtime->mu);
        runtime->starting = 0;
        runtime->joined = 1;
        KC_MUTEX_UNLOCK(&runtime->mu);
        kc_runtime_signal_lifecycle_internal(runtime);
        return -EAGAIN;
    }
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->starting = 0;
    runtime->started = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_join_all(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    if (kc_runtime_is_current_worker_internal(runtime)) return -EDEADLK;
    int status = kc_runtime_start(runtime);
    if (status != 0) return status;
    for (;;) {
        const uint32_t observed = kc_doorbell_observe(
            runtime->lifecycle_doorbell);
        const uint64_t progress = atomic_load_explicit(&runtime->progress,
                                                       memory_order_acquire);
        if (atomic_load_explicit(&runtime->active,
                                 memory_order_acquire) == 0 &&
            atomic_load_explicit(&runtime->progress,
                                 memory_order_acquire) == progress) return 0;
        atomic_fetch_add_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
        if (atomic_load_explicit(&runtime->active, memory_order_acquire) != 0 &&
            atomic_load_explicit(&runtime->progress, memory_order_acquire) ==
                progress) {
            if (kc_doorbell_park(runtime->lifecycle_doorbell, observed) != 0)
                abort();
        }
        atomic_fetch_sub_explicit(&runtime->lifecycle_waiters, 1,
                                  memory_order_seq_cst);
    }
}

void kc_runtime_request_stop(kc_runtime_t *runtime)
{
    if (!runtime) return;
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->accepting = 0;
    runtime->stop_requested = 1;
    kc_service_runtime_stop_locked(runtime);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_ring_workers_internal(runtime);
    kc_runtime_signal_lifecycle_internal(runtime);
}

int kc_runtime_join(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    if (kc_runtime_is_current_worker_internal(runtime)) return -EDEADLK;
    KC_MUTEX_LOCK(&runtime->mu);
    const int busy = runtime->starting || runtime->joining ||
        atomic_load_explicit(&runtime->active, memory_order_acquire) != 0 ||
        atomic_load_explicit(&runtime->queued, memory_order_acquire) != 0 ||
        atomic_load_explicit(&runtime->running, memory_order_acquire) != 0;
    if (busy) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -EBUSY;
    }
    if (!runtime->started || runtime->joined) {
        runtime->joined = 1;
        KC_MUTEX_UNLOCK(&runtime->mu);
        return 0;
    }
    runtime->joining = 1;
    atomic_store_explicit(&runtime->worker_stop, 1, memory_order_release);
    KC_MUTEX_UNLOCK(&runtime->mu);
    kc_runtime_ring_workers_internal(runtime);
    for (unsigned index = 0; index < runtime->worker_count; ++index)
        kc_port_thread_join(runtime->workers[index].thread);
    KC_MUTEX_LOCK(&runtime->mu);
    runtime->started = 0;
    runtime->joining = 0;
    runtime->joined = 1;
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_runtime_destroy(kc_runtime_t *runtime)
{
    if (!runtime) return 0;
    KC_MUTEX_LOCK(&runtime->mu);
    const int busy = runtime->starting || runtime->joining ||
        atomic_load_explicit(&runtime->active, memory_order_acquire) != 0 ||
        atomic_load_explicit(&runtime->queued, memory_order_acquire) != 0 ||
        atomic_load_explicit(&runtime->running, memory_order_acquire) != 0 ||
        runtime->live_services != 0 || runtime->live_continuations != 0 ||
        (runtime->started && !runtime->joined);
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (busy) return -EBUSY;
    kc_runtime_release_internal(runtime);
    return 0;
}

int kc_runtime_snapshot_get(kc_runtime_t *runtime, kc_runtime_snapshot *out)
{
    if (!runtime || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    const size_t active = atomic_load_explicit(&runtime->active,
                                               memory_order_acquire);
    const size_t queued = atomic_load_explicit(&runtime->queued,
                                               memory_order_acquire);
    const size_t running = atomic_load_explicit(&runtime->running,
                                                memory_order_acquire);
    *out = (kc_runtime_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .active = active, .queued = queued, .running = running,
        .dormant = active > queued + running ? active - queued - running : 0,
        .workers = runtime->worker_count,
        .accepting = (unsigned)runtime->accepting,
        .started = (unsigned)runtime->started,
        .stop_requested = (unsigned)runtime->stop_requested,
        .wake_requests = atomic_load_explicit(&runtime->wake_requests,
                                              memory_order_relaxed),
        .resumes = atomic_load_explicit(&runtime->resumes,
                                        memory_order_relaxed),
    };
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

/* Private implementation-backed measurement hook. It observes the actual
 * native worker threads, not Rust executor activity or a scheduler counter. */
int kc_runtime_worker_cpu_ns_for_test(kc_runtime_t *runtime,
                                      uint64_t *out_ns)
{
    if (!runtime || !out_ns || !runtime->started || runtime->joined)
        return -EINVAL;
    uint64_t total = 0;
    for (unsigned index = 0; index < runtime->worker_count; ++index) {
        uint64_t value = 0;
        const int status = kc_port_thread_cpu_ns(
            runtime->workers[index].thread, &value);
        if (status != 0) return status;
        if (UINT64_MAX - total < value) return -EOVERFLOW;
        total += value;
    }
    *out_ns = total;
    return 0;
}
