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
    /* One atomic linearizes operation admission against release. The high bit
     * closes admission; the low bits are operations that crossed the gate.
     * Separate `active` and `closing` atomics let release observe zero between
     * an entrant's two updates and free the registration underneath it. */
    atomic_uint gate;
    enum kc_wait_backend backend;
};

#define KC_WAIT_CLOSED UINT32_C(0x80000000)
#define KC_WAIT_COUNT UINT32_C(0x7fffffff)

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

static void kc_wait_leave(kc_port_wait_word *word)
{
    unsigned before = atomic_fetch_sub_explicit(&word->gate, 1,
                                                memory_order_release);
    if ((before & KC_WAIT_COUNT) != 1 || !(before & KC_WAIT_CLOSED)) return;
    pthread_mutex_lock(&word->mutex);
    pthread_cond_broadcast(&word->cond);
    pthread_mutex_unlock(&word->mutex);
}

static int kc_wait_enter(kc_port_wait_word *word)
{
    if (!word) return -EINVAL;
    /* The registration owner bounds callers to its resident worker set, far
     * below KC_WAIT_COUNT. One fetch-add admits or rejects this operation;
     * idle entry and realtime wake never retry an atomic under contention. */
    unsigned gate = atomic_fetch_add_explicit(&word->gate, 1,
                                               memory_order_acquire);
    if (!(gate & KC_WAIT_CLOSED) &&
        (gate & KC_WAIT_COUNT) != KC_WAIT_COUNT) return 0;
    kc_wait_leave(word);
    if (gate & KC_WAIT_CLOSED) return -ECANCELED;
    return -EAGAIN;
}

static int kc_wait_is_closing(const kc_port_wait_word *word)
{
    return (atomic_load_explicit(&word->gate, memory_order_acquire) &
            KC_WAIT_CLOSED) != 0;
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
    atomic_init(&word->gate, 0);
#if defined(__APPLE__)
    if (__builtin_available(macOS 14.4, *)) word->backend = KC_WAIT_DARWIN;
#elif defined(__linux__)
    word->backend = KC_WAIT_FUTEX;
#endif
    *out = word;
    return 0;
}

static int kc_wait_pthread(kc_port_wait_word *word, uint32_t expected)
{
    int result = 0;
    pthread_mutex_lock(&word->mutex);
    while (!kc_wait_is_closing(word) &&
           kc_atomic_u32_load_acquire(word->address) == expected) {
        int rc = pthread_cond_wait(&word->cond, &word->mutex);
        if (rc != 0) {
            result = -rc;
            break;
        }
    }
    if (kc_atomic_u32_load_acquire(word->address) != expected) result = 0;
    else if (kc_wait_is_closing(word))
        result = -ECANCELED;
    pthread_mutex_unlock(&word->mutex);
    return result;
}

#if defined(__APPLE__)
static int kc_wait_darwin(kc_port_wait_word *word, uint32_t expected)
{
    if (__builtin_available(macOS 14.4, *)) {
        for (;;) {
            if (kc_atomic_u32_load_acquire(word->address) != expected) return 0;
            if (kc_wait_is_closing(word))
                return -ECANCELED;
            int rc = os_sync_wait_on_address(word->address, expected,
                                             sizeof(*word->address),
                                             OS_SYNC_WAIT_ON_ADDRESS_NONE);
            if (rc >= 0) continue;
            int error = errno;
            if (error == EINTR) continue;
            return -error;
        }
    }
    return -ENOTSUP;
}
#endif

#if defined(__linux__)
static int kc_wait_futex(kc_port_wait_word *word, uint32_t expected)
{
    for (;;) {
        if (kc_atomic_u32_load_acquire(word->address) != expected) return 0;
        if (kc_wait_is_closing(word))
            return -ECANCELED;
        int rc = (int)syscall(SYS_futex, word->address, FUTEX_WAIT_PRIVATE,
                              expected, NULL, NULL, 0);
        if (rc == 0) continue;
        int error = errno;
        if (error == EAGAIN || error == EINTR) continue;
        return -error;
    }
}
#endif

int kc_port_wait_u32(kc_port_wait_word *word, uint32_t expected)
{
    int entered = kc_wait_enter(word);
    if (entered != 0) return entered;
    int result;
    switch (word->backend) {
#if defined(__APPLE__)
    case KC_WAIT_DARWIN:
        result = kc_wait_darwin(word, expected);
        break;
#endif
#if defined(__linux__)
    case KC_WAIT_FUTEX:
        result = kc_wait_futex(word, expected);
        break;
#endif
    default:
        result = kc_wait_pthread(word, expected);
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

int kc_port_wait_u32_wake_is_realtime_safe(const kc_port_wait_word *word)
{
    if (!word || !atomic_is_lock_free(&word->gate)) return 0;
    switch (word->backend) {
#if defined(__APPLE__)
    case KC_WAIT_DARWIN:
        return 1;
#endif
#if defined(__linux__)
    case KC_WAIT_FUTEX:
        return 1;
#endif
    default:
        return 0;
    }
}

void kc_port_wait_u32_release(kc_port_wait_word *word)
{
    if (!word) return;
    unsigned before = atomic_fetch_or_explicit(&word->gate, KC_WAIT_CLOSED,
                                               memory_order_acq_rel);
    if (before & KC_WAIT_CLOSED) return;
    kc_atomic_u32_fetch_add_release(word->address, 1);
    kc_wait_wake_native(word, 1);

    pthread_mutex_lock(&word->mutex);
    pthread_cond_broadcast(&word->cond);
    while ((atomic_load_explicit(&word->gate, memory_order_acquire) &
            KC_WAIT_COUNT) != 0) {
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
