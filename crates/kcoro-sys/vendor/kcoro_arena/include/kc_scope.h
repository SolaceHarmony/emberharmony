// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_SCOPE_H
#define KC_SCOPE_H

#include "kc_cancel.h"
#include "kc_op.h"
#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_scope kc_scope_t;

typedef struct kc_scope_config {
    uint32_t size;
    uint32_t abi_version;
    kc_cancel_t *parent_cancel;
} kc_scope_config;

typedef struct kc_scope_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id id;
    size_t children;
    size_t join_waiters;
    unsigned closed;
    unsigned canceled;
} kc_scope_snapshot;

int kc_scope_create(kc_runtime_t *runtime, const kc_scope_config *config,
                    kc_scope_t **out);
void kc_scope_retain(kc_scope_t *scope);
void kc_scope_release(kc_scope_t *scope);
int kc_scope_spawn(kc_scope_t *scope, kc_runtime_step_fn step, void *arg,
                   size_t local_size);
void kc_scope_close(kc_scope_t *scope);
void kc_scope_cancel(kc_scope_t *scope);
int kc_scope_join(kc_scope_t *scope, uint64_t deadline_ns);
kc_cancel_t *kc_scope_cancel_token(kc_scope_t *scope);
int kc_scope_snapshot_get(kc_scope_t *scope, kc_scope_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
