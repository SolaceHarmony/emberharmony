// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_COLLECTIVE_H
#define KC_COLLECTIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_collective kc_collective_t;
typedef void (*kc_collective_final_fn)(void *context);

typedef struct kc_collective_snapshot {
    uint32_t size;
    uint32_t members;
    uint64_t generation;
    uint64_t wake_calls;
    uint64_t wakes;
    uint32_t arrived;
    uint32_t parked;
} kc_collective_snapshot;

int kc_collective_create(uint32_t members, kc_collective_t **out);
/*
 * Every fixed member calls exactly once for a boundary. The last arrival runs
 * `final` before release-publishing the next generation. Other members park on
 * the collective doorbell and recheck the generation; there is no spin tier.
 */
int kc_collective_arrive(kc_collective_t *collective, uint32_t member,
                         kc_collective_final_fn final, void *context);
uint64_t kc_collective_generation(const kc_collective_t *collective);
int kc_collective_snapshot_get(const kc_collective_t *collective,
                               kc_collective_snapshot *out);
void kc_collective_destroy(kc_collective_t *collective);

#ifdef __cplusplus
}
#endif

#endif
