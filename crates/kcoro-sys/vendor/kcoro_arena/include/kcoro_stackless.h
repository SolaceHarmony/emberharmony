// SPDX-License-Identifier: BSD-3-Clause
/* kcoro_stackless.h - Stackless coroutine primitives
 *
 * This header defines the stackless coroutine model for kcoro_arena.
 * Unlike stackful coroutines that allocate separate stacks, stackless
 * coroutines maintain state in heap-allocated continuation records.
 *
 * Key benefits:
 * - Memory efficiency: ~100 bytes per coroutine vs 64KB+ stacks
 * - No assembly required: pure C implementation
 * - Portable: works on any architecture
 * - Cache-friendly: better locality
 *
 * Design inspired by Protothreads and continuation-passing style (CPS).
 */
#ifndef KCORO_STACKLESS_H
#define KCORO_STACKLESS_H

#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include "kcoro_desc.h"
#include "kc_op.h"
#include "kc_cancel.h"
#include "kc_timer.h"
#include "kc_select.h"
#include "kc_scope.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct koro_cont;
struct koro_scheduler;
struct kc_chan;
struct kc_runtime;
struct kc_descriptor;

/* Continuation function pointer type.
 * Returns NULL when suspended, non-NULL when complete.
 * The scheduler calls this repeatedly until it returns non-NULL. */
typedef void* (*koro_step_fn)(struct koro_cont* k);

typedef enum koro_run_state {
    KORO_NEW = 0,
    KORO_QUEUED,
    KORO_RUNNING,
    KORO_WAITING,
    KORO_DONE,
} koro_run_state;

typedef enum koro_suspend_kind {
    KORO_SUSPEND_WAIT = 0,
    KORO_SUSPEND_YIELD = 1,
} koro_suspend_kind;

/* Coroutine continuation record.
 * This is the heap-allocated "stack frame" for a stackless coroutine.
 * All state that must survive suspension is stored here. */
typedef struct koro_cont {
    /* Core state machine */
    int state;              /* Current resumption point (line number) */
    koro_step_fn next_step; /* Function to call on next resume */

    /* User state */
    void* user_data;        /* Points to user's local variables struct */
    void* user_arg;         /* Original argument passed to koro_go */
    struct kc_runtime *runtime;

    /* Scheduler linkage */
    struct koro_cont* next; /* Next in ready queue */
    atomic_int run_state;
    atomic_int wake_pending;
    atomic_int destroy_requested;
    int suspend_kind;

    /* Identity and lifecycle */
    uint64_t id;            /* Unique coroutine ID */
    const char* name;       /* Debug name */
    atomic_uint refs;
    int completed;          /* True when coroutine has finished */
    int managed;            /* Scheduler owns lifetime (set by koro_go) */
    int tracked;            /* Scheduler has counted this coroutine as active */
    void (*completion)(void *context);
    void *completion_context;

    /* Arena integration */
    int last_park_result;   /* Result from last suspension (arena status) */
    void* arena_payload;    /* Cached arena payload pointer */
    size_t arena_payload_len; /* Cached arena payload length */
    uint64_t arena_desc_id; /* Descriptor ID for zero-copy payloads */
    struct kc_descriptor *arena_descriptor;
    size_t select_index;
    kc_op *arena_op;
} koro_cont_t;

/* Protothread-style macros for user code.
 * These expand into a switch statement-based state machine. */

/* Begin a stackless coroutine function.
 * Place at the start of the function, before any locals. */
#define KORO_BEGIN(k) \
    switch ((k)->state) { \
        case 0:

/* End a stackless coroutine function.
 * Place at the end. Sets completed flag and returns. */
#define KORO_END(k) \
    } \
    (k)->state = 0; \
    (k)->completed = 1; \
    return (void*)1;

/* Suspend execution and yield to scheduler.
 * Saves current line as resumption point. */
#define KORO_YIELD(k) \
    do { \
        (k)->suspend_kind = KORO_SUSPEND_YIELD; \
        (k)->state = __LINE__; \
        return NULL; \
        case __LINE__:; \
    } while (0)

/* Suspend while waiting for a condition.
 * Yields repeatedly until condition becomes true. */
#define KORO_WAIT_UNTIL(k, condition) \
    do { \
        (k)->suspend_kind = KORO_SUSPEND_WAIT; \
        (k)->state = __LINE__; \
        case __LINE__: \
        if (!(condition)) return NULL; \
    } while (0)

/* Public API functions */

/* Create a new stackless coroutine.
 * - initial_step: first function to execute
 * - user_arg: argument passed to user code
 * - user_data_size: bytes to allocate for local variables
 * Returns continuation record or NULL on failure. */
koro_cont_t* koro_cont_create(koro_step_fn initial_step,
                               void* user_arg,
                               size_t user_data_size);
koro_cont_t* koro_cont_create_on(struct kc_runtime *runtime,
                                 koro_step_fn initial_step,
                                 void *user_arg,
                                 size_t user_data_size);

/* Free a continuation record and its user_data.
 * Should only be called when coroutine is complete. */
void koro_cont_destroy(koro_cont_t* k);
void koro_cont_retain(koro_cont_t *k);

/* Execute one step of a coroutine.
 * Returns NULL if suspended, non-NULL if complete.
 * This is what the scheduler calls. */
static inline void* koro_cont_step(koro_cont_t* k) {
    if (k->completed) return (void*)1;
    if (!k->next_step) return (void*)1;
    return k->next_step(k);
}

/* Check if a coroutine is complete. */
static inline int koro_cont_is_done(koro_cont_t* k) {
    return k->completed;
}

/* Stackless arena primitives.
 * These are CPS versions of the arena operations. */

/* Attempt to send to arena channel.
 * Returns immediately if successful.
 * Suspends and returns NULL if blocked. */
int koro_send_begin(koro_cont_t* k, struct kc_chan* ch, void* data, size_t len);
int koro_recv_begin(koro_cont_t* k, struct kc_chan* ch);
int koro_send_begin_ex(koro_cont_t *k, struct kc_chan *ch, void *data,
                       size_t len, kc_cancel_t *cancel, uint64_t deadline_ns);
int koro_recv_begin_ex(koro_cont_t *k, struct kc_chan *ch,
                       kc_cancel_t *cancel, uint64_t deadline_ns);
int koro_sleep_begin(koro_cont_t *k, uint64_t deadline_ns,
                     kc_cancel_t *cancel);
int koro_timer_begin(koro_cont_t *k, kc_timer_t *timer);
int koro_select_begin(koro_cont_t *k, const kc_select_clause *clauses,
                      size_t count, kc_cancel_t *cancel,
                      uint64_t deadline_ns);
int koro_scope_join_begin(koro_cont_t *k, kc_scope_t *scope,
                          kc_cancel_t *cancel, uint64_t deadline_ns);
int koro_op_finish(koro_cont_t *k);

/* Macros for arena operations in user code */

/* Send data through arena channel, suspending if necessary.
 * After resume, check k->last_park_result for status. */
#define KORO_SEND_EX(k, ch, data, len, cancel, deadline_ns) \
    do { \
        if (!koro_send_begin_ex((k), (ch), (data), (len), (cancel), (deadline_ns))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#define KORO_SEND(k, ch, data, len) \
    KORO_SEND_EX((k), (ch), (data), (len), NULL, 0)

/* Receive data from arena channel, suspending if necessary.
 * After resume, data is in k->arena_payload with length k->arena_payload_len.
 * For zero-copy descriptors, k->arena_desc_id contains the descriptor ID. */
#define KORO_RECV_EX(k, ch, cancel, deadline_ns) \
    do { \
        if (!koro_recv_begin_ex((k), (ch), (cancel), (deadline_ns))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#define KORO_RECV(k, ch) KORO_RECV_EX((k), (ch), NULL, 0)

#define KORO_SLEEP_UNTIL(k, deadline_ns, cancel) \
    do { \
        if (!koro_sleep_begin((k), (deadline_ns), (cancel))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#define KORO_TIMER(k, timer) \
    do { \
        if (!koro_timer_begin((k), (timer))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#define KORO_SELECT(k, clauses, count, cancel, deadline_ns) \
    do { \
        if (!koro_select_begin((k), (clauses), (count), (cancel), (deadline_ns))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#define KORO_JOIN_SCOPE(k, scope, cancel, deadline_ns) \
    do { \
        if (!koro_scope_join_begin((k), (scope), (cancel), (deadline_ns))) { \
            (k)->state = __LINE__; \
            return NULL; \
            case __LINE__:; \
        } \
        if (!koro_op_finish((k))) return NULL; \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* KCORO_STACKLESS_H */
