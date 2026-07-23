// SPDX-License-Identifier: BSD-3-Clause
#ifndef KCORO_STACKLESS_H
#define KCORO_STACKLESS_H

#include "kc_identity.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kc_runtime;
typedef struct koro_cont koro_cont_t;

/* One invocation advances a logical coroutine until it either suspends or
 * completes.  NULL means that the frame dehydrated itself.  Any non-NULL value
 * means that the frame is terminal. */
typedef void *(*koro_step_fn)(koro_cont_t *continuation);
typedef void (*koro_completion_fn)(void *context,
                                   const kc_ticket_id *continuation);

typedef enum koro_run_state {
    KORO_NEW = 0,
    KORO_QUEUED = 1,
    KORO_RUNNING = 2,
    KORO_SUSPENDED = 3,
    KORO_DONE = 4,
} koro_run_state;

typedef enum koro_suspend_kind {
    /* Only a correlated callback may make this frame runnable again. */
    KORO_SUSPEND_CALLBACK = 0,
    /* Cooperative scheduling point.  The runtime republishes the frame.
     * Execution resumes immediately after KORO_YIELD.  A predicate-draining
     * loop must recheck its owner buffer before callback suspension because
     * a callback edge may legally coalesce with the self-publication. */
    KORO_SUSPEND_YIELD = 1,
} koro_suspend_kind;

typedef struct koro_cont_config {
    koro_step_fn step;
    void *argument;
    size_t frame_size;
    /* Zero admits every runtime worker.  A non-zero mask is reserved for a
     * genuinely owner-affine host resource; it is eligibility, not ownership
     * of the logical coroutine. */
    uint64_t worker_mask;
    koro_completion_fn completion;
    void *completion_context;
} koro_cont_config;

/* Read-only control evidence for native supervision and fatal test gates. It
 * contains no frame bytes and does not retain the continuation. The owner
 * must keep its setup-time lease alive while reading this view. */
typedef struct koro_cont_snapshot {
    kc_ticket_id identity;
    uint32_t run_state;
    uint32_t wake_pending;
    uint32_t current_worker;
} koro_cont_snapshot;

/* Creation is a setup-time allocation and registration operation.  The frame
 * is fixed for its lifetime; starting and resuming allocate nothing. */
int koro_cont_create_on(struct kc_runtime *runtime,
                        const koro_cont_config *config,
                        koro_cont_t **out);
int koro_cont_start(koro_cont_t *continuation);

/* A callback resumes one exact logical coroutine. The complete identity is
 * its correlation key; stale or unrelated callbacks are rejected. */
int koro_cont_resume(koro_cont_t *continuation,
                     const kc_ticket_id *identity);
kc_ticket_id koro_cont_identity(const koro_cont_t *continuation);

/* The caller's setup-time lease may be released only after DONE (or while the
 * continuation is still NEW).  Runtime and callback leases keep the frame
 * alive through terminal publication. */
int koro_cont_destroy(koro_cont_t *continuation);
void koro_cont_retain(koro_cont_t *continuation);
void koro_cont_release(koro_cont_t *continuation);

void *koro_cont_frame(koro_cont_t *continuation);
void *koro_cont_argument(koro_cont_t *continuation);
uint32_t koro_cont_current_worker(const koro_cont_t *continuation);
int koro_cont_snapshot_get(const koro_cont_t *continuation,
                           koro_cont_snapshot *out);
uint32_t koro_cont_state_get(const koro_cont_t *continuation);
void koro_cont_state_set(koro_cont_t *continuation, uint32_t state,
                         uint32_t suspend_kind);
/* Claims terminal ownership against a concurrent callback resume. Returns one
 * when terminal publication may proceed. Returns zero when a correlated edge
 * won first; the caller must return dormant and let that successor invocation
 * run. */
int koro_cont_finish(koro_cont_t *continuation);

/* The source position is only a private resume label inside one compiled
 * function.  It is never serialized or exposed as product protocol state. */
#define KORO_BEGIN(k) switch (koro_cont_state_get((k))) { case 0:

#define KORO_SUSPEND(k)                                                   \
    do {                                                                  \
        koro_cont_state_set((k), (uint32_t)__LINE__,                       \
                            KORO_SUSPEND_CALLBACK);                        \
        return NULL;                                                       \
        case __LINE__:;                                                    \
    } while (0)

#define KORO_YIELD(k)                                                     \
    do {                                                                  \
        koro_cont_state_set((k), (uint32_t)__LINE__,                       \
                            KORO_SUSPEND_YIELD);                           \
        return NULL;                                                       \
        case __LINE__:;                                                    \
    } while (0)

#define KORO_END(k)                                                       \
    }                                                                     \
    return koro_cont_finish((k)) ? (void *)1 : NULL

#ifdef __cplusplus
}
#endif

#endif
