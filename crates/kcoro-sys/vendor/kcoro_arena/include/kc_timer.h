// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_TIMER_H
#define KC_TIMER_H

#include "kc_cancel.h"
#include "kc_op.h"
#include "kc_runtime.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_timer kc_timer_t;

typedef struct kc_timer_config {
    uint32_t size;
    uint32_t abi_version;
    uint64_t deadline_ns;
    kc_cancel_t *parent_cancel;
} kc_timer_config;

typedef struct kc_timer_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id id;
    uint64_t deadline_ns;
    unsigned canceled;
} kc_timer_snapshot;

int kc_timer_create(kc_runtime_t *runtime, const kc_timer_config *config,
                    kc_timer_t **out);
void kc_timer_retain(kc_timer_t *timer);
void kc_timer_cancel(kc_timer_t *timer);
void kc_timer_release(kc_timer_t *timer);
int kc_timer_snapshot_get(const kc_timer_t *timer, kc_timer_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
