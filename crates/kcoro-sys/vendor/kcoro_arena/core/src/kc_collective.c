// SPDX-License-Identifier: BSD-3-Clause
#include "kc_collective.h"

#include "kc_doorbell.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

struct kc_collective {
    uint32_t members;
    atomic_uint arrived;
    atomic_uint park_mask;
    atomic_uint_fast64_t generation;
    atomic_uint_fast64_t wake_calls;
    atomic_uint_fast64_t wakes;
    kc_doorbell_t *doorbell;
};

int kc_collective_create(uint32_t members, kc_collective_t **out)
{
    if (!out || members == 0 || members > 32) return -EINVAL;
    kc_collective_t *collective = calloc(1, sizeof(*collective));
    if (!collective) return -ENOMEM;
    collective->members = members;
    atomic_init(&collective->arrived, 0);
    atomic_init(&collective->park_mask, 0);
    atomic_init(&collective->generation, 0);
    atomic_init(&collective->wake_calls, 0);
    atomic_init(&collective->wakes, 0);
    int status = kc_doorbell_create(&collective->doorbell);
    if (status != 0) {
        free(collective);
        return status;
    }
    *out = collective;
    return 0;
}

int kc_collective_arrive(kc_collective_t *collective, uint32_t member,
                         kc_collective_final_fn final, void *context)
{
    if (!collective || member >= collective->members) return -EINVAL;
    const uint64_t generation = atomic_load_explicit(
        &collective->generation, memory_order_relaxed);
    if (atomic_fetch_add_explicit(&collective->arrived, 1,
                                  memory_order_acq_rel) + 1 ==
        collective->members) {
        if (final) final(context);
        atomic_store_explicit(&collective->arrived, 0, memory_order_relaxed);
        atomic_store_explicit(&collective->generation, generation + 1,
                              memory_order_release);
        const uint32_t parked = atomic_exchange_explicit(
            &collective->park_mask, 0, memory_order_acq_rel);
        if (parked) {
            atomic_fetch_add_explicit(&collective->wake_calls, 1,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&collective->wakes,
                                      (uint32_t)__builtin_popcount(parked),
                                      memory_order_relaxed);
            kc_doorbell_ring_all(collective->doorbell);
        }
        return 1;
    }

    const uint32_t bit = UINT32_C(1) << member;
    uint32_t observed = kc_doorbell_observe(collective->doorbell);
    atomic_fetch_or_explicit(&collective->park_mask, bit, memory_order_acq_rel);
    if (atomic_load_explicit(&collective->generation, memory_order_acquire) !=
        generation) {
        atomic_fetch_and_explicit(&collective->park_mask, ~bit,
                                  memory_order_acq_rel);
        return 0;
    }
    while (atomic_load_explicit(&collective->generation,
                                memory_order_acquire) == generation) {
        int status = kc_doorbell_wait(collective->doorbell, observed, 0);
        observed = kc_doorbell_observe(collective->doorbell);
        if (status != 0) return status;
    }
    atomic_fetch_and_explicit(&collective->park_mask, ~bit,
                              memory_order_acq_rel);
    return 0;
}

uint64_t kc_collective_generation(const kc_collective_t *collective)
{
    return collective
        ? atomic_load_explicit(&collective->generation, memory_order_acquire)
        : 0;
}

int kc_collective_snapshot_get(const kc_collective_t *collective,
                               kc_collective_snapshot *out)
{
    if (!collective || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_collective_snapshot){
        .size = sizeof(*out),
        .members = collective->members,
        .generation = atomic_load_explicit(&collective->generation,
                                           memory_order_acquire),
        .wake_calls = atomic_load_explicit(&collective->wake_calls,
                                           memory_order_relaxed),
        .wakes = atomic_load_explicit(&collective->wakes,
                                      memory_order_relaxed),
        .arrived = atomic_load_explicit(&collective->arrived,
                                        memory_order_acquire),
        .parked = atomic_load_explicit(&collective->park_mask,
                                       memory_order_acquire),
    };
    return 0;
}

void kc_collective_destroy(kc_collective_t *collective)
{
    if (!collective) return;
    kc_doorbell_destroy(collective->doorbell);
    free(collective);
}
