// SPDX-License-Identifier: BSD-3-Clause
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#endif

#include "kc_port.h"
#include "kc_atomic.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <os/os_sync_wait_on_address.h>
#elif defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

struct kc_port_mutex { pthread_mutex_t value; };
struct kc_port_cond { pthread_cond_t value; };
struct kc_port_thread { pthread_t value; };

enum kc_wait_backend {
    KC_WAIT_PTHREAD = 0,
    KC_WAIT_DARWIN,
    KC_WAIT_FUTEX,
};

struct kc_port_wait_word {
    uint32_t *address;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    atomic_uint active;
    atomic_int closing;
    enum kc_wait_backend backend;
};

static int kc_cond_init(pthread_cond_t *cond)
{
    pthread_condattr_t attr;
    int rc = pthread_condattr_init(&attr);
    if (rc != 0) return rc;
#if !defined(__APPLE__) && defined(CLOCK_MONOTONIC)
    (void)pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
#endif
    rc = pthread_cond_init(cond, &attr);
    pthread_condattr_destroy(&attr);
    return rc;
}

static struct timespec kc_deadline_timespec(uint64_t deadline_ns)
{
#if defined(__APPLE__)
    uint64_t now = kc_port_monotonic_ns();
    uint64_t delta = deadline_ns > now ? deadline_ns - now : 0;
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += (time_t)(delta / UINT64_C(1000000000));
    deadline.tv_nsec += (long)(delta % UINT64_C(1000000000));
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    return deadline;
#else
    struct timespec deadline = {
        .tv_sec = (time_t)(deadline_ns / UINT64_C(1000000000)),
        .tv_nsec = (long)(deadline_ns % UINT64_C(1000000000)),
    };
    return deadline;
#endif
}

static void kc_wait_leave(kc_port_wait_word *word)
{
    if (atomic_fetch_sub_explicit(&word->active, 1, memory_order_acq_rel) != 1) return;
    if (!atomic_load_explicit(&word->closing, memory_order_acquire)) return;
    pthread_mutex_lock(&word->mutex);
    pthread_cond_broadcast(&word->cond);
    pthread_mutex_unlock(&word->mutex);
}

static int kc_wait_enter(kc_port_wait_word *word)
{
    if (!word) return -EINVAL;
    atomic_fetch_add_explicit(&word->active, 1, memory_order_acquire);
    if (!atomic_load_explicit(&word->closing, memory_order_acquire)) return 0;
    kc_wait_leave(word);
    return -ECANCELED;
}

int kc_port_mutex_create(kc_port_mutex **out)
{
    if (!out) return -EINVAL;
    kc_port_mutex *mutex = calloc(1, sizeof(*mutex));
    if (!mutex) return -ENOMEM;
    int rc = pthread_mutex_init(&mutex->value, NULL);
    if (rc != 0) { free(mutex); return -rc; }
    *out = mutex;
    return 0;
}

void kc_port_mutex_destroy(kc_port_mutex *mutex)
{
    if (!mutex) return;
    pthread_mutex_destroy(&mutex->value);
    free(mutex);
}

void kc_port_mutex_lock(kc_port_mutex *mutex) { pthread_mutex_lock(&mutex->value); }
void kc_port_mutex_unlock(kc_port_mutex *mutex) { pthread_mutex_unlock(&mutex->value); }

int kc_port_cond_create(kc_port_cond **out)
{
    if (!out) return -EINVAL;
    kc_port_cond *cond = calloc(1, sizeof(*cond));
    if (!cond) return -ENOMEM;
    int rc = kc_cond_init(&cond->value);
    if (rc != 0) { free(cond); return -rc; }
    *out = cond;
    return 0;
}

void kc_port_cond_destroy(kc_port_cond *cond)
{
    if (!cond) return;
    pthread_cond_destroy(&cond->value);
    free(cond);
}

void kc_port_cond_wait(kc_port_cond *cond, kc_port_mutex *mutex)
{
    pthread_cond_wait(&cond->value, &mutex->value);
}

int kc_port_cond_timedwait(kc_port_cond *cond, kc_port_mutex *mutex,
                           uint64_t deadline_ns)
{
    if (deadline_ns <= kc_port_monotonic_ns()) return -ETIMEDOUT;
    struct timespec deadline = kc_deadline_timespec(deadline_ns);
    int rc = pthread_cond_timedwait(&cond->value, &mutex->value, &deadline);
    return rc == 0 ? 0 : -rc;
}

void kc_port_cond_signal(kc_port_cond *cond) { pthread_cond_signal(&cond->value); }
void kc_port_cond_broadcast(kc_port_cond *cond) { pthread_cond_broadcast(&cond->value); }

int kc_port_wait_u32_prepare(uint32_t *address, kc_port_wait_word **out)
{
    if (!address || !out || ((uintptr_t)address & (sizeof(*address) - 1)) != 0)
        return -EINVAL;
    if (!kc_atomic_u32_is_lock_free(address)) return -ENOTSUP;

    kc_port_wait_word *word = calloc(1, sizeof(*word));
    if (!word) return -ENOMEM;
    int rc = pthread_mutex_init(&word->mutex, NULL);
    if (rc != 0) { free(word); return -rc; }
    rc = kc_cond_init(&word->cond);
    if (rc != 0) {
        pthread_mutex_destroy(&word->mutex);
        free(word);
        return -rc;
    }
    word->address = address;
    atomic_init(&word->active, 0);
    atomic_init(&word->closing, 0);
#if defined(__APPLE__)
    if (__builtin_available(macOS 14.4, *)) word->backend = KC_WAIT_DARWIN;
#elif defined(__linux__)
    word->backend = KC_WAIT_FUTEX;
#endif
    *out = word;
    return 0;
}

static int kc_wait_pthread(kc_port_wait_word *word, uint32_t expected,
                           uint64_t deadline_ns)
{
    int result = 0;
    pthread_mutex_lock(&word->mutex);
    while (!atomic_load_explicit(&word->closing, memory_order_acquire) &&
           kc_atomic_u32_load_acquire(word->address) == expected) {
        int rc;
        if (deadline_ns == 0) rc = pthread_cond_wait(&word->cond, &word->mutex);
        else {
            if (deadline_ns <= kc_port_monotonic_ns()) {
                result = -ETIMEDOUT;
                break;
            }
            struct timespec deadline = kc_deadline_timespec(deadline_ns);
            rc = pthread_cond_timedwait(&word->cond, &word->mutex, &deadline);
        }
        if (rc == ETIMEDOUT &&
            kc_atomic_u32_load_acquire(word->address) == expected) {
            result = -ETIMEDOUT;
            break;
        }
        if (rc != 0 && rc != ETIMEDOUT) {
            result = -rc;
            break;
        }
    }
    if (kc_atomic_u32_load_acquire(word->address) != expected) result = 0;
    else if (atomic_load_explicit(&word->closing, memory_order_acquire))
        result = -ECANCELED;
    pthread_mutex_unlock(&word->mutex);
    return result;
}

#if defined(__APPLE__)
static int kc_wait_darwin(kc_port_wait_word *word, uint32_t expected,
                          uint64_t deadline_ns)
{
    if (__builtin_available(macOS 14.4, *)) {
        for (;;) {
            if (kc_atomic_u32_load_acquire(word->address) != expected) return 0;
            if (atomic_load_explicit(&word->closing, memory_order_acquire))
                return -ECANCELED;
            int rc;
            if (deadline_ns == 0) {
                rc = os_sync_wait_on_address(word->address, expected,
                                             sizeof(*word->address),
                                             OS_SYNC_WAIT_ON_ADDRESS_NONE);
            } else {
                uint64_t now = kc_port_monotonic_ns();
                if (deadline_ns <= now) return -ETIMEDOUT;
                rc = os_sync_wait_on_address_with_timeout(
                    word->address, expected, sizeof(*word->address),
                    OS_SYNC_WAIT_ON_ADDRESS_NONE, OS_CLOCK_MACH_ABSOLUTE_TIME,
                    deadline_ns - now);
            }
            if (rc >= 0) continue;
            int error = errno;
            if (error == EINTR) continue;
            if (error == ETIMEDOUT &&
                kc_atomic_u32_load_acquire(word->address) != expected) continue;
            return -error;
        }
    }
    return -ENOTSUP;
}
#endif

#if defined(__linux__)
static int kc_wait_futex(kc_port_wait_word *word, uint32_t expected,
                         uint64_t deadline_ns)
{
    for (;;) {
        if (kc_atomic_u32_load_acquire(word->address) != expected) return 0;
        if (atomic_load_explicit(&word->closing, memory_order_acquire))
            return -ECANCELED;
        struct timespec timeout;
        struct timespec *timeout_ptr = NULL;
        if (deadline_ns) {
            uint64_t now = kc_port_monotonic_ns();
            if (deadline_ns <= now) return -ETIMEDOUT;
            uint64_t remaining = deadline_ns - now;
            timeout.tv_sec = (time_t)(remaining / UINT64_C(1000000000));
            timeout.tv_nsec = (long)(remaining % UINT64_C(1000000000));
            timeout_ptr = &timeout;
        }
        int rc = (int)syscall(SYS_futex, word->address, FUTEX_WAIT_PRIVATE,
                              expected, timeout_ptr, NULL, 0);
        if (rc == 0) continue;
        int error = errno;
        if (error == EAGAIN || error == EINTR) continue;
        if (error == ETIMEDOUT &&
            kc_atomic_u32_load_acquire(word->address) != expected) continue;
        return -error;
    }
}
#endif

int kc_port_wait_u32(kc_port_wait_word *word, uint32_t expected,
                     uint64_t deadline_ns)
{
    int entered = kc_wait_enter(word);
    if (entered != 0) return entered;
    int result;
    switch (word->backend) {
#if defined(__APPLE__)
    case KC_WAIT_DARWIN:
        result = kc_wait_darwin(word, expected, deadline_ns);
        break;
#endif
#if defined(__linux__)
    case KC_WAIT_FUTEX:
        result = kc_wait_futex(word, expected, deadline_ns);
        break;
#endif
    default:
        result = kc_wait_pthread(word, expected, deadline_ns);
        break;
    }
    kc_wait_leave(word);
    return result;
}

static void kc_wait_wake_native(kc_port_wait_word *word, int all)
{
    switch (word->backend) {
#if defined(__APPLE__)
    case KC_WAIT_DARWIN:
        if (__builtin_available(macOS 14.4, *)) {
            if (all) {
                (void)os_sync_wake_by_address_all(word->address,
                                                  sizeof(*word->address),
                                                  OS_SYNC_WAKE_BY_ADDRESS_NONE);
            } else {
                (void)os_sync_wake_by_address_any(word->address,
                                                  sizeof(*word->address),
                                                  OS_SYNC_WAKE_BY_ADDRESS_NONE);
            }
        }
        break;
#endif
#if defined(__linux__)
    case KC_WAIT_FUTEX:
        (void)syscall(SYS_futex, word->address, FUTEX_WAKE_PRIVATE,
                      all ? INT_MAX : 1, NULL, NULL, 0);
        break;
#endif
    default:
        pthread_mutex_lock(&word->mutex);
        if (all) pthread_cond_broadcast(&word->cond);
        else pthread_cond_signal(&word->cond);
        pthread_mutex_unlock(&word->mutex);
        break;
    }
}

static void kc_port_wake_u32(kc_port_wait_word *word, int all)
{
    if (kc_wait_enter(word) != 0) return;
    kc_wait_wake_native(word, all);
    kc_wait_leave(word);
}

void kc_port_wake_u32_one(kc_port_wait_word *word) { kc_port_wake_u32(word, 0); }
void kc_port_wake_u32_all(kc_port_wait_word *word) { kc_port_wake_u32(word, 1); }

void kc_port_wait_u32_release(kc_port_wait_word *word)
{
    if (!word) return;
    if (atomic_exchange_explicit(&word->closing, 1, memory_order_acq_rel)) return;
    kc_atomic_u32_fetch_add_release(word->address, 1);
    kc_wait_wake_native(word, 1);

    pthread_mutex_lock(&word->mutex);
    pthread_cond_broadcast(&word->cond);
    while (atomic_load_explicit(&word->active, memory_order_acquire) != 0) {
        pthread_cond_wait(&word->cond, &word->mutex);
    }
    pthread_mutex_unlock(&word->mutex);
    pthread_cond_destroy(&word->cond);
    pthread_mutex_destroy(&word->mutex);
    free(word);
}

int kc_port_thread_create(kc_port_thread **out, kc_port_thread_fn fn, void *arg)
{
    if (!out || !fn) return -EINVAL;
    kc_port_thread *thread = calloc(1, sizeof(*thread));
    if (!thread) return -ENOMEM;
    int rc = pthread_create(&thread->value, NULL, fn, arg);
    if (rc != 0) { free(thread); return -rc; }
    *out = thread;
    return 0;
}

void kc_port_thread_join(kc_port_thread *thread)
{
    if (!thread) return;
    pthread_join(thread->value, NULL);
    free(thread);
}

unsigned kc_port_cpu_count(void)
{
#ifdef _SC_NPROCESSORS_ONLN
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    if (count < 1) return 1;
    if (count > 256) return 256;
    return (unsigned)count;
#else
    return 1;
#endif
}

uint64_t kc_port_monotonic_ns(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

void kc_port_thread_yield(void) { sched_yield(); }
