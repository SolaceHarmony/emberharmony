// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_PORT_H
#define KC_PORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_port_mutex kc_port_mutex;
typedef struct kc_port_thread kc_port_thread;
typedef struct kc_port_wait_word kc_port_wait_word;
typedef void *(*kc_port_thread_fn)(void *arg);

int kc_port_mutex_create(kc_port_mutex **out);
void kc_port_mutex_destroy(kc_port_mutex *mutex);
void kc_port_mutex_lock(kc_port_mutex *mutex);
void kc_port_mutex_unlock(kc_port_mutex *mutex);

/*
 * Expected-value address dormancy is the private idle boundary for resident
 * runtime/team workers. Preparation may allocate and selects the host's direct
 * address backend once; park and wake never search a registry or allocate. The
 * caller release-publishes a changed value before an ordinary wake. Park returns
 * immediately when the value differs and becomes dormant without polling while
 * it is equal. There is deliberately no deadline form. Release is exactly once;
 * it publishes one terminal increment and wakes workers that already entered the
 * registration. Entry and release linearize through one packed closed/count
 * gate. No new worker may enter after release starts, and the owner must quiesce
 * every potential caller before releasing the registration.
 */
int kc_port_wait_u32_prepare(uint32_t *address, kc_port_wait_word **out);
int kc_port_wait_u32(kc_port_wait_word *word, uint32_t expected);
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

#ifdef __cplusplus
}
#endif

#endif
