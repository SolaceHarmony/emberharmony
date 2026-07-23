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

static const unsigned KC_RUNTIME_SLOT_CLOSED = 1u << 31;
static const unsigned KC_RUNTIME_SLOT_READERS = (1u << 31) - 1;

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
    free(runtime->slot_gates);
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
    runtime->slot_gates = calloc(runtime->continuation_capacity,
                                 sizeof(*runtime->slot_gates));
    runtime->slot_generations = calloc(runtime->continuation_capacity,
                                       sizeof(*runtime->slot_generations));
    if (!runtime->continuations || !runtime->slot_gates ||
        !runtime->ready_words ||
        !runtime->slot_generations) return -ENOMEM;
    for (size_t slot = 0; slot < runtime->continuation_capacity; ++slot) {
        atomic_init(&runtime->continuations[slot], NULL);
        atomic_init(&runtime->slot_gates[slot], KC_RUNTIME_SLOT_CLOSED);
    }
    for (size_t word = 0; word < runtime->ready_word_count; ++word)
        atomic_init(&runtime->ready_words[word], 0);
    return 0;
}

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out)
{
    if (!out) return -EINVAL;
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
    atomic_init(&runtime->test_claim_armed, 0);
    atomic_init(&runtime->test_register_armed, 0);
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
    for (; slot < runtime->continuation_capacity; ++slot) {
        if (atomic_load_explicit(&runtime->continuations[slot],
                                 memory_order_acquire) != NULL) continue;
        if (atomic_exchange_explicit(&runtime->test_register_armed, 0,
                                     memory_order_acq_rel) != 0) {
            if (!runtime->test_register_pause) abort();
            runtime->test_register_pause(runtime->test_register_context,
                                         runtime, (uint32_t)slot);
        }
        unsigned closed = KC_RUNTIME_SLOT_CLOSED;
        if (atomic_compare_exchange_strong_explicit(
                &runtime->slot_gates[slot], &closed, 0,
                memory_order_acq_rel, memory_order_acquire)) break;
    }
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
    /* The slot owns one lifetime lease until unregister closes admission and
     * every worker that entered the slot gate has released its read lease. */
    koro_cont_retain_internal(continuation);
    atomic_store_explicit(&continuation->registered, 1,
                          memory_order_release);
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
    if (!runtime || !continuation ||
        !atomic_load_explicit(&continuation->registered,
                              memory_order_acquire) ||
        continuation->slot >= runtime->continuation_capacity) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    koro_cont_t *current = atomic_load_explicit(
        &runtime->continuations[continuation->slot], memory_order_acquire);
    if (current != continuation) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ESTALE;
    }
    const unsigned readers = atomic_fetch_or_explicit(
        &runtime->slot_gates[continuation->slot], KC_RUNTIME_SLOT_CLOSED,
        memory_order_acq_rel) & KC_RUNTIME_SLOT_READERS;
    clear_ready(runtime, continuation->slot);
    atomic_store_explicit(&continuation->registered, 0,
                          memory_order_release);
    if (readers == 0) {
        atomic_store_explicit(
            &runtime->continuations[continuation->slot], NULL,
            memory_order_release);
    }
    if (runtime->live_continuations) runtime->live_continuations--;
    KC_MUTEX_UNLOCK(&runtime->mu);
    if (readers == 0) koro_cont_release_internal(continuation);
    kc_runtime_signal_lifecycle_internal(runtime);
    return 0;
}

static void slot_reader_leave(kc_runtime_t *runtime, size_t slot)
{
    const unsigned prior = atomic_fetch_sub_explicit(
        &runtime->slot_gates[slot], 1, memory_order_acq_rel);
    const unsigned readers = prior & KC_RUNTIME_SLOT_READERS;
    if (readers == 0) abort();
    if ((prior & KC_RUNTIME_SLOT_CLOSED) && readers == 1) {
        /* A publication admitted before close may have deposited its ready
         * bit after unregister's first clear. The final admitted reader owns
         * the causal second clear before the slot can be reused. */
        clear_ready(runtime, (uint32_t)slot);
        koro_cont_t *continuation = atomic_exchange_explicit(
            &runtime->continuations[slot], NULL, memory_order_acq_rel);
        if (continuation) koro_cont_release_internal(continuation);
    }
}

static int slot_reader_enter(kc_runtime_t *runtime, size_t slot)
{
    const unsigned prior = atomic_fetch_add_explicit(
        &runtime->slot_gates[slot], 1, memory_order_acq_rel);
    if ((prior & KC_RUNTIME_SLOT_READERS) == KC_RUNTIME_SLOT_READERS)
        abort();
    if ((prior & KC_RUNTIME_SLOT_CLOSED) == 0) return 1;
    /* Closed admission is not a wait or retry. Undo this non-admitted lease;
     * if it is the final observer, it may finish deferred slot retirement. */
    slot_reader_leave(runtime, slot);
    return 0;
}

static void publish_ready(kc_runtime_t *runtime, koro_cont_t *continuation)
{
    if (!runtime || !continuation ||
        !atomic_load_explicit(&continuation->registered,
                              memory_order_acquire)) return;
    const size_t slot = continuation->slot;
    if (slot >= runtime->continuation_capacity ||
        !slot_reader_enter(runtime, slot)) return;
    if (atomic_load_explicit(&runtime->continuations[slot],
                             memory_order_acquire) != continuation ||
        !atomic_load_explicit(&continuation->registered,
                              memory_order_acquire)) {
        slot_reader_leave(runtime, slot);
        return;
    }
    const size_t word = continuation->slot / 64;
    const uint64_t bit = UINT64_C(1) << (continuation->slot % 64);
    /* Account before visibility: a worker may consume a ready bit as soon as
     * fetch-or publishes it. Publishing the bit first lets that worker
     * decrement zero and transiently wrap the lifecycle counter. */
    atomic_fetch_add_explicit(&runtime->queued, 1, memory_order_release);
    const uint64_t prior = atomic_fetch_or_explicit(
        &runtime->ready_words[word], bit, memory_order_acq_rel);
    if (prior & bit) {
        atomic_fetch_sub_explicit(&runtime->queued, 1,
                                  memory_order_acq_rel);
        slot_reader_leave(runtime, slot);
        return;
    }
    atomic_fetch_add_explicit(&runtime->progress, 1, memory_order_release);
    if (continuation->worker_mask)
        kc_doorbell_ring_all(runtime->work_doorbell);
    else
        kc_doorbell_ring_one(runtime->work_doorbell);
    slot_reader_leave(runtime, slot);
}

int kc_runtime_start_continuation_internal(koro_cont_t *continuation)
{
    if (!continuation || !continuation->runtime ||
        !atomic_load_explicit(&continuation->registered,
                              memory_order_acquire)) return -EINVAL;
    kc_runtime_t *runtime = continuation->runtime;
    KC_MUTEX_LOCK(&runtime->mu);
    if (!runtime->accepting) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        return -ECANCELED;
    }
    int expected = atomic_load_explicit(&continuation->run_state,
                                        memory_order_acquire);
    int started = 0;
    if (koro_run_base(expected) == KORO_NEW) {
        started = atomic_compare_exchange_strong_explicit(
            &continuation->run_state, &expected, KORO_QUEUED,
            memory_order_acq_rel, memory_order_acquire);
        /* The only same-base interference is the callback's one-way wake-bit
         * deposit. Once present, another callback cannot change the word, so
         * one bounded retry consumes it without placing a loop on either path. */
        if (!started && koro_run_base(expected) == KORO_NEW &&
            koro_run_has_wake(expected))
            started = atomic_compare_exchange_strong_explicit(
                &continuation->run_state, &expected, KORO_QUEUED,
                memory_order_acq_rel, memory_order_acquire);
    }
    if (!started) {
        KC_MUTEX_UNLOCK(&runtime->mu);
        const int state = koro_run_base(expected);
        return state == KORO_QUEUED || state == KORO_RUNNING ||
                       state == KORO_SUSPENDED
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
        !atomic_load_explicit(&continuation->registered,
                              memory_order_acquire)) return -EINVAL;
    kc_runtime_t *runtime = continuation->runtime;
    atomic_fetch_add_explicit(&runtime->wake_requests, 1,
                              memory_order_relaxed);
    /* Realtime producers perform exactly one state RMW.  The wake bit is the
     * callback deposit; workers, never callbacks, arbitrate the base state.
     * A suspended frame additionally receives one idempotent ready-board edge. */
    const int prior = atomic_fetch_or_explicit(
        &continuation->run_state, KORO_WAKE_BIT, memory_order_acq_rel);
    const int state = koro_run_base(prior);
    if (state == KORO_DONE || state == KORO_COMPLETING) return -ECANCELED;
    if (state != KORO_NEW && state != KORO_QUEUED &&
        state != KORO_RUNNING && state != KORO_SUSPENDED &&
        state != KORO_SUSPENDING) abort();
    if (state == KORO_SUSPENDED && !koro_run_has_wake(prior)) {
        atomic_fetch_add_explicit(&runtime->resumes, 1,
                                  memory_order_relaxed);
        publish_ready(runtime, continuation);
    }
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
    if (runtime && continuation &&
        atomic_load_explicit(&continuation->registered,
                             memory_order_acquire))
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
            if (!slot_reader_enter(runtime, slot)) {
                ready &= ~bit;
                continue;
            }
            koro_cont_t *continuation = atomic_load_explicit(
                &runtime->continuations[slot], memory_order_acquire);
            if (!continuation) {
                slot_reader_leave(runtime, slot);
                ready &= ~bit;
                continue;
            }
            if (atomic_exchange_explicit(&runtime->test_claim_armed, 0,
                                         memory_order_acq_rel) != 0) {
                if (!runtime->test_claim_pause) abort();
                runtime->test_claim_pause(runtime->test_claim_context,
                                          worker->index, (uint32_t)slot);
            }
            if (continuation->worker_mask &&
                !(continuation->worker_mask &
                  (UINT64_C(1) << worker->index))) {
                slot_reader_leave(runtime, slot);
                ready &= ~bit;
                continue;
            }
            /* Claim the publication before the frame. A worker may carry a
             * stale local copy of this word past a complete invocation; only
             * fetch-and returning the bit proves that this is the current
             * publication. Clearing it first is safe because a callback that
             * overlaps the claim deposits KORO_WAKE_BIT in the same state
             * word. This worker's CAS either consumes that deposit with the
             * queued invocation or observes it on the active frame. */
            const uint64_t prior = atomic_fetch_and_explicit(
                &runtime->ready_words[word], ~bit, memory_order_acq_rel);
            ready = prior & ~bit;
            if (!(prior & bit)) {
                slot_reader_leave(runtime, slot);
                continue;
            }
            atomic_fetch_sub_explicit(&runtime->queued, 1,
                                      memory_order_relaxed);
            for (;;) {
                int state = atomic_load_explicit(&continuation->run_state,
                                                 memory_order_acquire);
                const int base = koro_run_base(state);
                if (base != KORO_QUEUED &&
                    !(base == KORO_SUSPENDED &&
                      koro_run_has_wake(state))) abort();
                if (atomic_compare_exchange_weak_explicit(
                        &continuation->run_state, &state, KORO_RUNNING,
                        memory_order_acq_rel, memory_order_acquire)) break;
            }
            koro_cont_retain_internal(continuation);
            slot_reader_leave(runtime, slot);
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
    current_worker_continuation = continuation;
    void *result = koro_cont_step(continuation);
    current_worker_continuation = NULL;
    atomic_store_explicit(&continuation->current_worker, UINT32_MAX,
                          memory_order_release);
    atomic_fetch_sub_explicit(&runtime->running, 1, memory_order_relaxed);

    if (result || continuation->completed) {
        for (;;) {
            int state = atomic_load_explicit(&continuation->run_state,
                                             memory_order_acquire);
            const int base = koro_run_base(state);
            if (base == KORO_RUNNING && koro_run_has_wake(state)) {
                if (!atomic_compare_exchange_weak_explicit(
                        &continuation->run_state, &state, KORO_RUNNING,
                        memory_order_acq_rel, memory_order_acquire)) continue;
                continuation->completed = 0;
                continuation->state = 0;
                continuation->suspend_kind = KORO_SUSPEND_YIELD;
                result = NULL;
                break;
            }
            if (base == KORO_RUNNING) {
                if (!atomic_compare_exchange_weak_explicit(
                        &continuation->run_state, &state, KORO_COMPLETING,
                        memory_order_acq_rel, memory_order_acquire)) continue;
                break;
            }
            if (base != KORO_COMPLETING) abort();
            break;
        }
    }

    if (result || continuation->completed) {
        koro_completion_fn completion = continuation->completion;
        void *context = continuation->completion_context;
        const kc_ticket_id identity = continuation->identity;
        if (completion) completion(context, &identity);
        /* DONE is the callback-lifetime acknowledgement. A destroyer cannot
         * release completion_context while the terminal callback is active. */
        atomic_store_explicit(&continuation->run_state, KORO_DONE,
                              memory_order_release);
        koro_completion_fn settled = continuation->settled;
        void *settled_context = continuation->settled_context;
        if (settled) settled(settled_context, &identity);
        if (continuation->tracked) {
            continuation->tracked = 0;
            atomic_fetch_sub_explicit(&runtime->active, 1,
                                      memory_order_relaxed);
        }
        kc_runtime_signal_lifecycle_internal(runtime);
        koro_cont_release_internal(continuation);
        return;
    }

    /* Seal the logical frame before another physical worker can claim it.
     * A callback deposits KORO_WAKE_BIT with one RMW; this worker owns all
     * base-state arbitration. */
    for (;;) {
        int state = atomic_load_explicit(&continuation->run_state,
                                         memory_order_acquire);
        if (koro_run_base(state) != KORO_RUNNING) abort();
        const int desired = KORO_SUSPENDING |
            (state & KORO_WAKE_BIT);
        if (atomic_compare_exchange_weak_explicit(
                &continuation->run_state, &state, desired,
                memory_order_acq_rel, memory_order_acquire)) break;
    }

    for (;;) {
        int state = atomic_load_explicit(&continuation->run_state,
                                         memory_order_acquire);
        const int ready =
            continuation->suspend_kind == KORO_SUSPEND_YIELD ||
            koro_run_has_wake(state);
        if (koro_run_base(state) != KORO_SUSPENDING) abort();
        const int desired = ready ? KORO_QUEUED : KORO_SUSPENDED;
        if (!atomic_compare_exchange_weak_explicit(
                &continuation->run_state, &state, desired,
                memory_order_acq_rel, memory_order_acquire)) continue;
        if (ready) publish_ready(runtime, continuation);
        break;
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
    if (!runtime || !out) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    const size_t active = atomic_load_explicit(&runtime->active,
                                               memory_order_acquire);
    const size_t queued = atomic_load_explicit(&runtime->queued,
                                               memory_order_acquire);
    const size_t running = atomic_load_explicit(&runtime->running,
                                                memory_order_acquire);
    *out = (kc_runtime_snapshot){
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

int kc_runtime_inject_claim_pause_for_test(
    kc_runtime_t *runtime, kc_runtime_test_claim_fn pause, void *context)
{
    if (!runtime || !pause) return -EINVAL;
    if (atomic_load_explicit(&runtime->test_claim_armed,
                             memory_order_acquire) != 0) return -EALREADY;
    runtime->test_claim_pause = pause;
    runtime->test_claim_context = context;
    atomic_store_explicit(&runtime->test_claim_armed, 1,
                          memory_order_release);
    return 0;
}

int kc_runtime_inject_register_pause_for_test(
    kc_runtime_t *runtime, kc_runtime_test_register_fn pause, void *context)
{
    if (!runtime || !pause) return -EINVAL;
    if (atomic_load_explicit(&runtime->test_register_armed,
                             memory_order_acquire) != 0) return -EALREADY;
    runtime->test_register_pause = pause;
    runtime->test_register_context = context;
    atomic_store_explicit(&runtime->test_register_armed, 1,
                          memory_order_release);
    return 0;
}

int kc_runtime_hold_closed_slot_reader_for_test(kc_runtime_t *runtime,
                                                uint32_t slot)
{
    if (!runtime || slot >= runtime->continuation_capacity) return -EINVAL;
    unsigned closed = KC_RUNTIME_SLOT_CLOSED;
    if (!atomic_compare_exchange_strong_explicit(
            &runtime->slot_gates[slot], &closed,
            KC_RUNTIME_SLOT_CLOSED | 1u, memory_order_acq_rel,
            memory_order_acquire)) return -EBUSY;
    return 0;
}

int kc_runtime_release_closed_slot_reader_for_test(kc_runtime_t *runtime,
                                                   uint32_t slot)
{
    if (!runtime || slot >= runtime->continuation_capacity) return -EINVAL;
    const unsigned gate = atomic_load_explicit(&runtime->slot_gates[slot],
                                               memory_order_acquire);
    if (gate != (KC_RUNTIME_SLOT_CLOSED | 1u)) return -EINVAL;
    slot_reader_leave(runtime, slot);
    return 0;
}
