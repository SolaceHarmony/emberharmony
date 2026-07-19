// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_DOORBELL_H
#define KC_DOORBELL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cache-isolated expected-value edge. The value is only a sequence used to
 * close the observe/recheck/park race; callers retain the actual predicate.
 */
typedef struct kc_doorbell kc_doorbell_t;

int kc_doorbell_create(kc_doorbell_t **out);
uint32_t kc_doorbell_observe(const kc_doorbell_t *doorbell);
void kc_doorbell_ring_one(kc_doorbell_t *doorbell);
void kc_doorbell_ring_all(kc_doorbell_t *doorbell);
int kc_doorbell_wait(kc_doorbell_t *doorbell, uint32_t expected,
                     uint64_t deadline_ns);
/* Ring is allocation-free for every backend, but only direct address-wake
 * backends are mutex-free and therefore admissible from realtime callbacks. */
int kc_doorbell_realtime_safe(const kc_doorbell_t *doorbell);
void kc_doorbell_destroy(kc_doorbell_t *doorbell);

#ifdef __cplusplus
}
#endif

#endif
