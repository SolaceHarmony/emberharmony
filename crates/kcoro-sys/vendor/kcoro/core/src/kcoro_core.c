// SPDX-License-Identifier: BSD-3-Clause
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>

#ifndef MAP_ANON
#define MAP_ANON 0x1000
#endif
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#include "kcoro_core.h"
#include "kcoro_sched.h"

/* Thread-local current coroutine */
static __thread kcoro_t* current_kcoro = NULL;
/* Thread-local main coroutine (yield target) */
static __thread kcoro_t* main_kcoro = NULL;

/* Coroutine ID counter */
static uint64_t next_kcoro_id = 1;

/* Default stack size */
#define KCORO_DEFAULT_STACK_SIZE (64 * 1024)  /* 64KB */

/* Patch 0001: park-gate states. All transitions are seq_cst exchanges on
 * co->park_notify, the single source of truth for the park/unpark handshake —
 * co->state (atomic since patch 0009, but always RELAXED) cannot order a wake
 * against the park switch. */
#define KC_PARK_EMPTY    0
#define KC_PARK_NOTIFIED 1
#define KC_PARK_PARKED   2

/* Patch 0002: fiber-safe TLS access. A compiler may legally cache the address of a
 * __thread variable for the lifetime of a stack frame (C assumes a frame never changes
 * threads). A coroutine frame DOES change threads across kcoro_switch under the M:N
 * scheduler, so any direct TLS access after a switch in the same frame may read or
 * write the OLD thread's slot — poisoning that thread's current_kcoro and misdirecting
 * every wake it later performs. Post-switch code must go through this noinline helper,
 * whose fresh frame recomputes the TLS address on the thread actually executing it. */
__attribute__((noinline)) static kcoro_t* kc_tls_main_fresh(void)
{
    __asm__ volatile("" ::: "memory");
    return main_kcoro;
}

/* Function protector implementation */
void kcoro_funcp_protector(void)
{
    kcoro_t *current = current_kcoro;
    int state = current ? (int)kc_co_state(current) : -1;
    fprintf(stderr,
            "kcoro: coroutine function returned unexpectedly (co=%p state=%d main=%p fn=%p)\n",
            (void*)current,
            state,
            current ? (void*)current->main_co : NULL,
            current ? (void*)current->fn : NULL);
    abort();
}

/* Internal function: coroutine trampoline */
static void kcoro_trampoline(void);

kcoro_t* kcoro_create_main(void)
{
    kcoro_t* main_co = (kcoro_t*)calloc(1, sizeof(kcoro_t));
    if (!main_co) return NULL;

    /* Initialize main coroutine */
    memset(main_co->reg, 0, sizeof(main_co->reg));
    atomic_init(&main_co->state, KCORO_RUNNING);
    main_co->fn = NULL;  /* Main has no function */
    main_co->arg = NULL;
    main_co->id = 0;     /* Special ID for main */
    main_co->main_co = NULL;  /* Main has no parent */
    main_co->name = "main";
    main_co->ready_enqueued = false;
    atomic_init(&main_co->running_flag, 0);
    atomic_init(&main_co->park_notify, 0);
    atomic_init(&main_co->refcount, 1);
    atomic_init(&main_co->scheduler, NULL);
    main_co->last_send_delivered = 0;
    
    /* Set as current */
    current_kcoro = main_co;
    main_kcoro = main_co;

    return main_co;
}

void kcoro_set_thread_main(kcoro_t* main_co)
{
    main_kcoro = main_co;
    current_kcoro = main_co;
}

kcoro_t* kcoro_create(kcoro_fn_t fn, void* arg, size_t stack_size)
{
    if (!fn) return NULL;
    if (stack_size == 0) stack_size = KCORO_DEFAULT_STACK_SIZE;
    
    kcoro_t* co = (kcoro_t*)calloc(1, sizeof(kcoro_t));
    if (!co) return NULL;
    
    /* Allocate stack with mmap for guard page support */
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0) page_size = 4096;
    
    /* Align stack size to page boundary */
    size_t total_size = (stack_size + page_size - 1) & ~(page_size - 1);
    
    void* stack_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (stack_mem == MAP_FAILED) {
        free(co);
        return NULL;
    }
    
    /* Initialize coroutine */
    memset(co->reg, 0, sizeof(co->reg));
    atomic_init(&co->state, KCORO_CREATED);
    co->fn = fn;
    co->arg = arg;
    co->id = __sync_fetch_and_add(&next_kcoro_id, 1);
    co->main_co = main_kcoro;     /* Default yield target */
    co->stack_ptr = stack_mem;
    co->stack_size = total_size;
    co->ready_enqueued = false;
    atomic_init(&co->running_flag, 0);
    atomic_init(&co->park_notify, 0);
    atomic_init(&co->refcount, 1);
    atomic_init(&co->scheduler, NULL);
    co->last_send_delivered = 0;
    
    /* Set up stack and entry point */
    uintptr_t stack_top = (uintptr_t)stack_mem + total_size;
    stack_top = stack_top & ~0xFUL;  /* 16-byte align */
    stack_top -= 16;  /* Leave space */

#if defined(__x86_64__)
    /* SysV x86-64 function entry expects %rsp % 16 == 8 (as if a `call` just
     * pushed the return address). The switcher jumps straight to reg[13] with
     * rsp = reg[14], so seed that entry state here; rbp = 0 roots the frame
     * chain for backtraces. (AArch64 below wants SP % 16 == 0 at all times —
     * the split is the ABI difference, not a style choice.) */
    co->reg[14] = (void*)(stack_top - 8);     /* RSP at reg[14] - entry state */
    co->reg[15] = NULL;                       /* RBP at reg[15] */
    co->reg[13] = (void*)kcoro_trampoline;    /* RIP at reg[13] - entry point */
#else
    co->reg[14] = (void*)stack_top;           /* SP at reg[14] */
    co->reg[15] = (void*)stack_top;           /* FP at reg[15] */
    co->reg[13] = (void*)kcoro_trampoline;    /* LR at reg[13] - entry point */
#endif
    
    return co;
}

static void kcoro_free(kcoro_t* co)
{
    if (!co) return;
    if (co->stack_ptr && co->stack_size > 0) {
        munmap(co->stack_ptr, co->stack_size);
    }
    if (current_kcoro == co) {
        current_kcoro = NULL;
    }
    if (main_kcoro == co) {
        main_kcoro = NULL;
    }
    free(co);
}

static int kcoro_ref_debug_enabled(void)
{
    /* Patch 0009: atomic (relaxed) — every worker thread races through here on
     * first use. Two threads may both compute the value; getenv is stable, so
     * the double-store is idempotent. */
    static kc_atomic_int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (__builtin_expect(v == -1, 0)) {
        const char *env = getenv("KCORO_REF_DEBUG");
        v = (env && *env && env[0] != '0');
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
}

void kcoro_destroy(kcoro_t* co)
{
    kcoro_release(co);
}

void kcoro_set_name(kcoro_t* co, const char* name)
{
    if (co) {
        co->name = name;
    }
}

kcoro_t* kcoro_current(void)
{
    return current_kcoro;
}

kcoro_t* kcoro_thread_main(void)
{
    return main_kcoro;
}

void kcoro_retain(kcoro_t* co)
{
    if (!co) return;
    int prev = atomic_fetch_add_explicit(&co->refcount, 1, memory_order_relaxed);
    if (kcoro_ref_debug_enabled()) {
        fprintf(stderr, "[kcoro][ref] retain co=%p -> %d\n", (void*)co, prev + 1);
    }
}

void kcoro_release(kcoro_t* co)
{
    if (!co) return;
    int prev = atomic_fetch_sub_explicit(&co->refcount, 1, memory_order_acq_rel);
    if (kcoro_ref_debug_enabled()) {
        fprintf(stderr, "[kcoro][ref] release co=%p prev=%d\n", (void*)co, prev);
    }
    if (prev == 1) {
        kcoro_free(co);
    }
}

void kcoro_resume(kcoro_t* co)
{
    if (!co || kc_co_state(co) == KCORO_FINISHED) return;
    
    kcoro_t* yield_co = current_kcoro;
    kcoro_t* from_co = yield_co ? yield_co : main_kcoro;
    if (!from_co) {
        from_co = co->main_co;
    }
    
    /* Update states */
    if (yield_co && yield_co != co) {
        kc_co_state_set(yield_co, KCORO_SUSPENDED);
    }
    kc_co_state_set(co, KCORO_RUNNING);
    current_kcoro = co;
    
    /* Context switch */
    kcoro_switch(from_co, co);

    /* Returned from context switch - restore current */
    current_kcoro = yield_co ? yield_co : main_kcoro;
    if (yield_co) {
        kc_co_state_set(yield_co, KCORO_RUNNING);
    } else if (main_kcoro) {
        kc_co_state_set(main_kcoro, KCORO_RUNNING);
    }
}

void kcoro_yield(void)
{
    kcoro_t* current = current_kcoro;
    kcoro_t* main_co = main_kcoro ? main_kcoro : (current ? current->main_co : NULL);
    if (!current || !main_co) {
        /* No main coroutine to yield to - this might be in a different context */
        return;
    }

    /* Update states */
    kc_co_state_set(current, KCORO_SUSPENDED);
    kc_co_state_set(main_co, KCORO_RUNNING);
    current_kcoro = main_co;
    
    /* Context switch back to main */
    kcoro_switch(current, main_co);
    
    /* When resumed, we'll be back here. Patch 0002: do NOT touch current_kcoro in
     * this frame — the resuming thread's kcoro_resume already set its own TLS to this
     * coroutine before switching in, and this frame's cached TLS address may belong to
     * the thread we parked on, not the one we resumed on. */
    kc_co_state_set(current, KCORO_RUNNING);
}

void kcoro_yield_to(kcoro_t* target_co)
{
    if (!target_co) return;
    
    kcoro_t* current = current_kcoro;
    
    /* Update states */
    if (current) {
        kc_co_state_set(current, KCORO_SUSPENDED);
    }
    kc_co_state_set(target_co, KCORO_RUNNING);
    current_kcoro = target_co;
    
    /* Context switch */
    kcoro_switch(current, target_co);
    
    /* When resumed, restore our state (Patch 0002: no TLS writes post-switch — see
     * kcoro_yield). */
    if (current) {
        kc_co_state_set(current, KCORO_RUNNING);
    }
}

/* Park current coroutine: transitions to KCORO_PARKED and switches to main */
void kcoro_park(void)
{
    kcoro_t* current = current_kcoro;
    kcoro_t* main_co = main_kcoro ? main_kcoro : (current ? current->main_co : NULL);
    if (!current || !main_co) return;
    if (kc_co_state(current) == KCORO_FINISHED) return;
    /* Patch 0001: publish park intent on the gate FIRST. If a notification already
     * landed, consume it and do not park — every park caller loops and re-checks its
     * condition, so a spurious return is safe. */
    if (atomic_exchange(&current->park_notify, KC_PARK_PARKED) == KC_PARK_NOTIFIED) {
        atomic_store(&current->park_notify, KC_PARK_EMPTY);
        return;
    }
    kc_co_state_set(current, KCORO_PARKED);
    kc_co_state_set(main_co, KCORO_RUNNING);
    current_kcoro = main_co;
    kcoro_switch(current, main_co);
    /* Patch 0001: resumed — retire this park cycle. A notification that arrived while
     * we were parked coalesces into this resume (all park sites re-check under lock). */
    atomic_store(&current->park_notify, KC_PARK_EMPTY);
    /* When unparked & resumed, state will be set by kcoro_unpark before scheduling */
    if (kc_co_state(current) == KCORO_PARKED) {
        /* Defensive: if resumed without state change, mark running */
        kc_co_state_set(current, KCORO_RUNNING);
    }
    /* Patch 0002: no TLS writes post-switch — see kcoro_yield. */
}

void kcoro_unpark(kcoro_t* co)
{
    if (!co) return;
    /* Patch 0001: the gate decides, not co->state (relaxed-atomic since 0009 —
     * still cannot order a wake against the park switch). Exchange to NOTIFIED:
     *   prev EMPTY/NOTIFIED — the target is running (or already has a wake pending):
     *     the token is stored; kcoro_park consumes it on entry. Nothing to schedule.
     *   prev PARKED — the target parked (or is mid-switch): ready it. If it is still
     *     mid-switch, the scheduler's running_flag CAS keeps re-queueing it until the
     *     switch completes, so the wake cannot be lost. */
    if (atomic_exchange(&co->park_notify, KC_PARK_NOTIFIED) != KC_PARK_PARKED) {
        return;
    }
    /* Patch 0008: do NOT write co->state here — and keep it that way even now
     * that state is atomic (0009). The write was pure redundancy: the worker
     * promotes PARKED -> READY itself when it resumes the coroutine (kc_sched.c,
     * the NOTIFIED-gate check), and the wake is carried by the atomic gate +
     * this enqueue + the running_flag CAS. Writers of state are the owning
     * worker plus kc_sched_enqueue_ready's PARKED->READY promotion (under
     * rq_mu); a bare cross-thread store HERE would bypass both the mutex and
     * the enqueue's state checks. Atomicity (0009) exists for the cross-thread
     * readers (kcoro_is_parked), not to license new writers. */
    /* Patch 0004: enqueue to the coroutine's OWN scheduler first. The caller's
     * scheduler (kc_sched_current) is NULL on external threads and may be a different
     * instance under multiple dispatchers — either way the wake belongs to the
     * scheduler that owns the coroutine. This is what makes an external-thread
     * `kcoro_unpark` a legal handoff (the engine's per-token doorbell: write the
     * request slot, unpark the parked coordinator). No auto-created default scheduler
     * — manually-driven coroutines (co->scheduler == NULL off-runtime) keep the old
     * behavior of not enqueueing. */
    kc_sched_t* s = (kc_sched_t*)kc_co_sched(co);
    if (!s) s = kc_sched_current();
    if (s) {
        kc_sched_enqueue_ready(s, co);
    }
}

int kcoro_is_parked(const kcoro_t* co)
{
    /* Patch 0009 (closes the 0008 residual): this cross-thread READ was the
     * remaining half of the state data race; `state` is atomic now, and this
     * relaxed load is value-identical to the old plain read. Do NOT "improve"
     * this to the park gate (park_notify == KC_PARK_PARKED) — that was tried
     * and REVERTED: the gate goes PARKED a beat before the coroutine switches
     * out, widening the "is parked" window enough to fire a rendezvous
     * direct-handoff early and DROP the final message (test_chan_rv_metrics:
     * sends=800 recvs=799). is_parked's timing edge is load-bearing in
     * kc_chan's handoff; only value-AND-timing-neutral changes are safe here. */
    return co && kc_co_state(co) == KCORO_PARKED;
}

/* Internal coroutine trampoline function */
static void kcoro_trampoline(void)
{
    kcoro_t* current = current_kcoro;
    assert(current && current->fn);
    
    /* Mark as running and call the function */
    kc_co_state_set(current, KCORO_RUNNING);
    current->fn(current->arg);
    
    /* Function completed - mark as finished */
    kc_co_state_set(current, KCORO_FINISHED);
    
    /* Yield back to main coroutine. Patch 0002: this frame started on the FIRST
     * thread that ever ran the coroutine and may be finishing on a different one, so
     * main_kcoro must be re-read through a fresh frame and the TLS write dropped
     * (kcoro_resume's post-switch tail restores the worker's current_kcoro). */
    {
        kcoro_t* main_co = kc_tls_main_fresh();
        if (main_co) {
            kc_co_state_set(main_co, KCORO_RUNNING);
            kcoro_switch(current, main_co);
            return;
        }
    }

    /* Should never reach here, but if we do, call protector */
    kcoro_funcp_protector();
}
