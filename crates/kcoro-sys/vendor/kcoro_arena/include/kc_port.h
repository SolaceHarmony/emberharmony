// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_PORT_H
#define KC_PORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_port_mutex kc_port_mutex;
typedef struct kc_port_cond kc_port_cond;
typedef struct kc_port_thread kc_port_thread;
typedef struct kc_port_wait_word kc_port_wait_word;
typedef void *(*kc_port_thread_fn)(void *arg);

int kc_port_mutex_create(kc_port_mutex **out);
void kc_port_mutex_destroy(kc_port_mutex *mutex);
void kc_port_mutex_lock(kc_port_mutex *mutex);
void kc_port_mutex_unlock(kc_port_mutex *mutex);

int kc_port_cond_create(kc_port_cond **out);
void kc_port_cond_destroy(kc_port_cond *cond);
void kc_port_cond_wait(kc_port_cond *cond, kc_port_mutex *mutex);
int kc_port_cond_timedwait(kc_port_cond *cond, kc_port_mutex *mutex,
                           uint64_t deadline_ns);
void kc_port_cond_signal(kc_port_cond *cond);
void kc_port_cond_broadcast(kc_port_cond *cond);

/*
 * Expected-value wait words are the portable doorbell boundary for fixed
 * compute teams. Preparation may allocate and selects the host's direct address
 * wait backend once; wait and wake never search a registry or allocate. The
 * caller release-publishes a changed value before an ordinary wake. Wait returns
 * immediately when the value differs, parks without polling while it is equal,
 * and returns -ETIMEDOUT for an unchanged deadline. Release is exactly once; it
 * publishes one terminal increment and wakes operations that already entered the
 * wait object. Entry and release linearize through one packed closed/count
 * gate. No new operation may begin after release starts, and the owner must
 * quiesce every potential caller before releasing the registration.
 */
int kc_port_wait_u32_prepare(uint32_t *address, kc_port_wait_word **out);
int kc_port_wait_u32(kc_port_wait_word *word, uint32_t expected,
                     uint64_t deadline_ns);
void kc_port_wake_u32_one(kc_port_wait_word *word);
void kc_port_wake_u32_all(kc_port_wait_word *word);
/* True only when wake uses the host's direct address primitive and cannot
 * acquire the pthread fallback mutex. The result is immutable after prepare. */
int kc_port_wait_u32_wake_is_realtime_safe(const kc_port_wait_word *word);
void kc_port_wait_u32_release(kc_port_wait_word *word);

int kc_port_thread_create(kc_port_thread **out, kc_port_thread_fn fn, void *arg);
void kc_port_thread_join(kc_port_thread *thread);
unsigned kc_port_cpu_count(void);
uint64_t kc_port_monotonic_ns(void);
void kc_port_thread_yield(void);

#ifdef __cplusplus
}
#endif

#endif
