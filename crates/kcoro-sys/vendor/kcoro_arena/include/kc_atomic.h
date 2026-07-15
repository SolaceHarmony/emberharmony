// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_ATOMIC_H
#define KC_ATOMIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Raw-word atomics for ABI-visible doorbells. Keeping the storage as uint32_t
 * lets C, C++, assembly, and host bindings share one naturally aligned word.
 * The supported native toolchains provide these operations as lock-free
 * compiler intrinsics; callers can verify that property before registration.
 */
static inline int kc_atomic_u32_is_lock_free(const uint32_t *address)
{
    return address && __atomic_is_lock_free(sizeof(*address), address);
}

static inline uint32_t kc_atomic_u32_load_relaxed(const uint32_t *address)
{
    return __atomic_load_n(address, __ATOMIC_RELAXED);
}

static inline uint32_t kc_atomic_u32_load_acquire(const uint32_t *address)
{
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

static inline void kc_atomic_u32_store_relaxed(uint32_t *address, uint32_t value)
{
    __atomic_store_n(address, value, __ATOMIC_RELAXED);
}

static inline uint32_t kc_atomic_u32_fetch_add_release(uint32_t *address,
                                                       uint32_t value)
{
    return __atomic_fetch_add(address, value, __ATOMIC_RELEASE);
}

static inline uint32_t kc_atomic_u32_fetch_add_acq_rel(uint32_t *address,
                                                       uint32_t value)
{
    return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
}

static inline uint32_t kc_atomic_u32_fetch_or_acq_rel(uint32_t *address,
                                                      uint32_t value)
{
    return __atomic_fetch_or(address, value, __ATOMIC_ACQ_REL);
}

static inline uint32_t kc_atomic_u32_exchange_acq_rel(uint32_t *address,
                                                      uint32_t value)
{
    return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
}

#ifdef __cplusplus
}
#endif

#endif
