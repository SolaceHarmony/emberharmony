// SPDX-License-Identifier: BSD-3-Clause
/**
 * @file kcoro_core.h
 * @brief Core coroutine primitives (create/destroy, resume/yield, park/unpark).
 *
 * -----------------------------------------------------------------------------
 * Header Surface & Optional Items
 * -----------------------------------------------------------------------------
 * Purpose
 *   Small, focused primitives for user‑space coroutines with an ARM64 switcher.
 *   Assembly keeps context hops fast; scheduling and policy remain in C.
 *
 * Optional items
 *   - This header intentionally has no optional tunables. The primitives here
 *     are foundational and stable.
 *
 * Install guidance
 *   - This header is part of the production public API and should be installed.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* C++ consumers (e.g. the flashkern engine TU): gcc's <stdatomic.h> does not
 * provide ::atomic_int in C++ mode before C++23 (clang's does — which is how
 * this went unnoticed on macOS). Alias the one atomic type this header's structs
 * use to std::atomic<int>, which is layout-compatible with C11 atomic_int on
 * every ABI we target (asserted below). C++ code must not touch these fields
 * directly — they are the runtime's; the alias exists only so the struct parses. */
#ifdef __cplusplus
#include <atomic>
typedef std::atomic<int> kc_atomic_int;
static_assert(sizeof(kc_atomic_int) == sizeof(int) && alignof(kc_atomic_int) == alignof(int),
              "std::atomic<int> must be layout-compatible with C11 atomic_int");
#else
#include <stdatomic.h>
typedef atomic_int kc_atomic_int;
_Static_assert(sizeof(kc_atomic_int) == sizeof(int) && _Alignof(kc_atomic_int) == _Alignof(int),
               "atomic_int must be layout-compatible with int (struct kcoro ABI)");
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct kcoro kcoro_t;
typedef struct kcoro_sched kcoro_sched_t;

/* Patch 0009: atomic owner-scheduler pointer. kcoro_unpark reads it UNLOCKED
 * from external threads (the doorbell path) while workers reassign it at every
 * resume and enqueue — a formal data race as a plain pointer (TSan: worker_main
 * write vs kc_sched_enqueue_ready write vs kcoro_unpark read). Relaxed
 * everywhere: same plain load/store codegen, no timing change; the writes are
 * same-value in a single-scheduler process, and under multiple dispatchers any
 * observed value is a live owner per patch 0004's semantics. */
#ifdef __cplusplus
typedef std::atomic<kcoro_sched_t*> kc_atomic_sched_ptr;
static_assert(sizeof(kc_atomic_sched_ptr) == sizeof(kcoro_sched_t*) &&
              alignof(kc_atomic_sched_ptr) == alignof(kcoro_sched_t*),
              "std::atomic<T*> must be layout-compatible with T* (struct kcoro ABI)");
#else
typedef _Atomic(kcoro_sched_t*) kc_atomic_sched_ptr;
_Static_assert(sizeof(kc_atomic_sched_ptr) == sizeof(kcoro_sched_t*) &&
               _Alignof(kc_atomic_sched_ptr) == _Alignof(kcoro_sched_t*),
               "_Atomic(T*) must be layout-compatible with T* (struct kcoro ABI)");
#endif

/* Coroutine function type */
typedef void (*kcoro_fn_t)(void* arg);

/* Coroutine state */
typedef enum {
    KCORO_CREATED,           /* Created but not started */
    KCORO_READY,             /* Ready to run */
    KCORO_RUNNING,           /* Currently executing */
    KCORO_SUSPENDED,         /* Suspended (yielded) */
    KCORO_PARKED,            /* Parked (not runnable until explicitly unparked) */
    KCORO_FINISHED           /* Completed execution */
} kcoro_state_t;

/* Core coroutine structure - matches ARM64 assembly requirements */
struct kcoro {
    /* Register save area - MUST be first field for assembly */
    void* reg[32];               /* ARM64: x19-x28, x30, sp, x29 at specific indices */
    
    /* Coroutine metadata */
    kc_atomic_int state;         /* Current execution state (holds kcoro_state_t values).
                                    Atomic since patch 0009: kcoro_is_parked reads it from
                                    OTHER threads (kc_chan's rendezvous direct-handoff
                                    probe), a C data race while this was a plain enum. All
                                    runtime accesses are RELAXED — on our targets that is
                                    the same plain load/store codegen as the old enum, so
                                    every timing edge of the park/handoff protocol is
                                    untouched (patch 0008 lesson: is_parked's PARKED edge
                                    is load-bearing). Ordering is carried by park_notify,
                                    running_flag, and the scheduler locks — never by state. */
    kcoro_fn_t fn;               /* Task function */
    void* arg;                   /* Task argument */
    uint64_t id;                 /* Unique coroutine ID */

    /* Execution context */
    kcoro_t* main_co;            /* Main coroutine (yield target) */
    kc_atomic_sched_ptr scheduler; /* Owning scheduler (relaxed atomic — patch 0009) */
    bool ready_enqueued;         /* Scheduler ready-queue flag */
    kc_atomic_int running_flag;     /* 0 = idle, 1 = running */
    kc_atomic_int park_notify;      /* park gate: 0 empty / 1 notified / 2 parked. The whole
                                    park/unpark handshake serializes on this one atomic so
                                    a wake can never race the park switch (liquid-audio
                                    patch 0001, see crates/kcoro-sys/vendor/kcoro/PATCHES.md) */
    kc_atomic_int refcount;         /* Reference count for lifetime management */

    /* Stack management */
    void* stack_ptr;             /* Private stack (if not using shared) */
    size_t stack_size;           /* Stack size */
    
    /* Scheduler linkage */
    kcoro_t* next;               /* Next in queue */
    kcoro_t* prev;               /* Previous in queue */
    
    /* Debug info */
    const char* name;            /* Optional name for debugging */

    /* Lightweight rendezvous handshake hint:
     * When a parked sender is directly handed off by a receiver, the receiver
     * marks this flag on the sender coroutine before waking it. The sender
     * checks and clears the flag after kcoro_park() to return success without
     * re-enqueueing. This avoids "success without transfer" and keeps the
     * core lock structure unchanged. */
    int last_send_delivered;     /* 1 if last parked send was delivered by recv */
};

/* Patch 0009 (runtime-internal, C only): the ONLY sanctioned accessors for
 * co->state. Relaxed, deliberately — state never carries ordering; substituting
 * anything stronger (or a different source, like the park gate) changes the
 * protocol's timing edges. C++ consumers must not touch the field at all. */
#ifndef __cplusplus
#define kc_co_state(co)        ((kcoro_state_t)atomic_load_explicit(&(co)->state, memory_order_relaxed))
#define kc_co_state_set(co, v) atomic_store_explicit(&(co)->state, (v), memory_order_relaxed)
#define kc_co_sched(co)        atomic_load_explicit(&(co)->scheduler, memory_order_relaxed)
#define kc_co_sched_set(co, v) atomic_store_explicit(&(co)->scheduler, (v), memory_order_relaxed)
#endif

/** ARM64 assembly context switching primitive (internal). */
extern void* kcoro_switch(kcoro_t* from_co, kcoro_t* to_co);

/* Function protector for proper stack cleanup */
extern void kcoro_funcp_protector_asm(void);
void kcoro_funcp_protector(void);

/**
 * @name Core coroutine API
 * Create/destroy coroutines and control execution.
 * @{ */
kcoro_t* kcoro_create(kcoro_fn_t fn, void* arg, size_t stack_size);
void kcoro_destroy(kcoro_t* co);

/* Set optional name for debugging */
void kcoro_set_name(kcoro_t* co, const char* name);

/** Execution control */
void kcoro_resume(kcoro_t* co);
kcoro_t* kcoro_current(void);
kcoro_t* kcoro_thread_main(void);
void kcoro_retain(kcoro_t* co);
void kcoro_release(kcoro_t* co);
void kcoro_yield(void);
void kcoro_yield_to(kcoro_t* target_co);

/** Parking (no fairness requeue) */
void kcoro_park(void);              /* Park current coroutine (must be running in scheduler context) */
void kcoro_unpark(kcoro_t* co);     /* Make a parked coroutine ready again */
int  kcoro_is_parked(const kcoro_t* co);

/** Main coroutine setup */
kcoro_t* kcoro_create_main(void);
void kcoro_set_thread_main(kcoro_t* main_co);
/** @} */

#ifdef __cplusplus
}
#endif
