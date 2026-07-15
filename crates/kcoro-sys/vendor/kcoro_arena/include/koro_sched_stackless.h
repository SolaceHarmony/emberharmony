// SPDX-License-Identifier: BSD-3-Clause
/* koro_sched_stackless.h - Stackless coroutine scheduler API
 *
 * This is the public interface for the stackless scheduler.
 * It provides the Go-like "koro_go()" API for spawning coroutines
 * and "koro_run()" for running the event loop.
 */
#ifndef KORO_SCHED_STACKLESS_H
#define KORO_SCHED_STACKLESS_H

#include "kcoro_stackless.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the stackless scheduler.
 * Must be called before any koro_go() or koro_run(). */
int koro_sched_init(void);

/* Optional: start background worker threads for parallel continuation execution.
 * If n <= 0, uses number of online processors. Safe to call after koro_sched_init().
 * Returns the number of workers started (>=0) or negative on error. */
int koro_sched_start_workers(int n);

/* Returns current worker count (0 means single-threaded cooperative run). */
int koro_sched_worker_count(void);

/* Spawn a new stackless coroutine.
 * - func: Entry point function with signature void* (*)(koro_cont_t*)
 * - arg: User argument passed to func
 * - local_size: Bytes to allocate for local variables struct
 * Returns 0 on success, negative on error. */
int koro_go(void* (*func)(koro_cont_t*), void* arg, size_t local_size);

/* Run the scheduler until all coroutines complete.
 * This is the main event loop—it does not return until done.
 * All coroutines execute on this thread's stack. */
int koro_run(void);

/* Stop the scheduler (can be called from a coroutine to exit). */
void koro_stop(void);

/* Stop workers, join them, and reset the process-default scheduler. */
void koro_sched_shutdown(void);

/* Enqueue a continuation to the ready queue (used by callbacks). */
void koro_sched_enqueue_ready(koro_cont_t* k);

/* Lost-wake-safe completion doorbell used by operations and external threads. */
void koro_sched_wake(koro_cont_t *k);

#ifdef __cplusplus
}
#endif

#endif /* KORO_SCHED_STACKLESS_H */
