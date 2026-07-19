// SPDX-License-Identifier: BSD-3-Clause
#include "kc_doorbell.h"

#include "kc_atomic.h"
#include "kc_port.h"

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>

enum { KC_DOORBELL_CACHELINE = 128 };

struct kc_doorbell {
    _Alignas(KC_DOORBELL_CACHELINE) uint32_t value;
    kc_port_wait_word *wait;
};

_Static_assert(_Alignof(struct kc_doorbell) == KC_DOORBELL_CACHELINE,
               "doorbell must begin on its own Apple cache line");
_Static_assert(sizeof(struct kc_doorbell) == KC_DOORBELL_CACHELINE,
               "adjacent doorbells must not false-share");

int kc_doorbell_create(kc_doorbell_t **out)
{
    if (!out) return -EINVAL;
    kc_doorbell_t *doorbell = aligned_alloc(KC_DOORBELL_CACHELINE,
                                            sizeof(*doorbell));
    if (!doorbell) return -ENOMEM;
    doorbell->value = 0;
    doorbell->wait = NULL;
    if (!kc_atomic_u32_is_lock_free(&doorbell->value) ||
        kc_port_wait_u32_prepare(&doorbell->value, &doorbell->wait) != 0) {
        free(doorbell);
        return -ENOTSUP;
    }
    *out = doorbell;
    return 0;
}

uint32_t kc_doorbell_observe(const kc_doorbell_t *doorbell)
{
    return doorbell ? kc_atomic_u32_load_acquire(&doorbell->value) : 0;
}

void kc_doorbell_ring_one(kc_doorbell_t *doorbell)
{
    if (!doorbell) return;
    kc_atomic_u32_fetch_add_release(&doorbell->value, 1);
    kc_port_wake_u32_one(doorbell->wait);
}

void kc_doorbell_ring_all(kc_doorbell_t *doorbell)
{
    if (!doorbell) return;
    kc_atomic_u32_fetch_add_release(&doorbell->value, 1);
    kc_port_wake_u32_all(doorbell->wait);
}

int kc_doorbell_wait(kc_doorbell_t *doorbell, uint32_t expected,
                     uint64_t deadline_ns)
{
    return doorbell
        ? kc_port_wait_u32(doorbell->wait, expected, deadline_ns)
        : -EINVAL;
}

void kc_doorbell_destroy(kc_doorbell_t *doorbell)
{
    if (!doorbell) return;
    kc_port_wait_u32_release(doorbell->wait);
    free(doorbell);
}
