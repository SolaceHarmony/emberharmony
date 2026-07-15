// SPDX-License-Identifier: BSD-3-Clause
#include "kc_admin.h"
#include "kc_channel_internal.h"
#include "kc_descriptor_internal.h"
#include "kc_op_internal.h"
#include "kc_runtime_internal.h"
#include "kc_scope_internal.h"

#include <errno.h>

uint64_t kc_runtime_capabilities(const kc_runtime_t *runtime)
{
    if (!runtime) return 0;
    return KC_CAP_STACKLESS | KC_CAP_CHANNEL_POLICIES |
           KC_CAP_DESCRIPTOR_LEASES | KC_CAP_CANCELLATION |
           KC_CAP_DEADLINES | KC_CAP_SELECT | KC_CAP_SCOPES |
           KC_CAP_ACTORS | KC_CAP_HOST_PORT | KC_CAP_TICKETS | KC_CAP_DURABLE_STORE |
           KC_CAP_TRANSPORT | KC_CAP_WORKFLOWS | KC_CAP_SHARED_REGIONS;
}

int kc_admin_snapshot_get(kc_runtime_t *runtime, kc_admin_snapshot *out)
{
    if (!runtime || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_admin_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .capabilities = kc_runtime_capabilities(runtime),
        .runtime = { .size = sizeof(kc_runtime_snapshot),
                     .abi_version = KC_ABI_VERSION },
        .memory = { .size = sizeof(kc_memory_snapshot),
                    .abi_version = KC_ABI_VERSION },
    };
    int rc = kc_runtime_snapshot_get(runtime, &out->runtime);
    if (rc != 0) return rc;
    rc = kc_memory_snapshot_get(runtime, &out->memory);
    if (rc != 0) return rc;
    for (size_t cause = 0; cause <= KC_CAUSE_FAILURE; cause++) {
        out->terminal_causes[cause] = atomic_load_explicit(
            &runtime->terminal_causes[cause], memory_order_relaxed);
    }
    return 0;
}

static int list_args(const void *out, size_t capacity, size_t *written,
                     size_t *total)
{
    if ((capacity && !out) || !written || !total) return -EINVAL;
    *written = 0;
    *total = 0;
    return 0;
}

int kc_admin_list_operations(kc_runtime_t *runtime, kc_op_snapshot *out,
                             size_t capacity, size_t *written, size_t *total)
{
    if (!runtime || list_args(out, capacity, written, total) != 0) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    for (kc_op *op = runtime->ops_head; op; op = op->registry_next) {
        (*total)++;
        if (*written == capacity) continue;
        out[*written] = (kc_op_snapshot){
            .size = sizeof(kc_op_snapshot), .abi_version = KC_ABI_VERSION,
        };
        if (kc_op_snapshot_get(op, &out[*written]) == 0) (*written)++;
    }
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_admin_list_channels(kc_runtime_t *runtime, kc_channel_snapshot *out,
                           size_t capacity, size_t *written, size_t *total)
{
    if (!runtime || list_args(out, capacity, written, total) != 0) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    for (struct kc_chan *channel = runtime->channels_head; channel;
         channel = channel->registry_next) {
        (*total)++;
        if (*written == capacity) continue;
        out[*written] = (kc_channel_snapshot){
            .size = sizeof(kc_channel_snapshot), .abi_version = KC_ABI_VERSION,
        };
        if (kc_channel_snapshot_get(channel, &out[*written]) == 0) (*written)++;
    }
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}

int kc_admin_list_descriptors(kc_runtime_t *runtime, kc_descriptor_snapshot *out,
                              size_t capacity, size_t *written, size_t *total)
{
    if (!runtime || list_args(out, capacity, written, total) != 0) return -EINVAL;
    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    for (uint32_t slot = 0; slot < registry->capacity; slot++) {
        kc_descriptor_t *descriptor = registry->slots[slot].descriptor;
        if (!descriptor) continue;
        (*total)++;
        if (*written == capacity) continue;
        out[*written] = (kc_descriptor_snapshot){
            .size = sizeof(kc_descriptor_snapshot), .abi_version = KC_ABI_VERSION,
        };
        if (kc_descriptor_snapshot_get(descriptor, &out[*written]) == 0) (*written)++;
    }
    KC_MUTEX_UNLOCK(&registry->mu);
    return 0;
}

int kc_admin_list_timers(kc_runtime_t *runtime, kc_op_snapshot *out,
                         size_t capacity, size_t *written, size_t *total)
{
    if (!runtime || list_args(out, capacity, written, total) != 0) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->timer_mu);
    for (kc_op *op = runtime->timer_head; op; op = op->timer_next) {
        (*total)++;
        if (*written == capacity) continue;
        out[*written] = (kc_op_snapshot){
            .size = sizeof(kc_op_snapshot), .abi_version = KC_ABI_VERSION,
        };
        if (kc_op_snapshot_get(op, &out[*written]) == 0) (*written)++;
    }
    KC_MUTEX_UNLOCK(&runtime->timer_mu);
    return 0;
}

int kc_admin_list_scopes(kc_runtime_t *runtime, kc_scope_snapshot *out,
                         size_t capacity, size_t *written, size_t *total)
{
    if (!runtime || list_args(out, capacity, written, total) != 0) return -EINVAL;
    KC_MUTEX_LOCK(&runtime->mu);
    for (kc_scope_t *scope = runtime->scopes_head; scope;
         scope = scope->registry_next) {
        (*total)++;
        if (*written == capacity) continue;
        out[*written] = (kc_scope_snapshot){
            .size = sizeof(kc_scope_snapshot), .abi_version = KC_ABI_VERSION,
        };
        if (kc_scope_snapshot_get(scope, &out[*written]) == 0) (*written)++;
    }
    KC_MUTEX_UNLOCK(&runtime->mu);
    return 0;
}
