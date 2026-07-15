// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_ADMIN_H
#define KC_ADMIN_H

#include "kc_channel.h"
#include "kc_descriptor.h"
#include "kc_op.h"
#include "kc_runtime.h"
#include "kc_scope.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum kc_capability {
    KC_CAP_STACKLESS = UINT64_C(1) << 0,
    KC_CAP_CHANNEL_POLICIES = UINT64_C(1) << 1,
    KC_CAP_DESCRIPTOR_LEASES = UINT64_C(1) << 2,
    KC_CAP_CANCELLATION = UINT64_C(1) << 3,
    KC_CAP_DEADLINES = UINT64_C(1) << 4,
    KC_CAP_SELECT = UINT64_C(1) << 5,
    KC_CAP_SCOPES = UINT64_C(1) << 6,
    KC_CAP_ACTORS = UINT64_C(1) << 7,
    KC_CAP_HOST_PORT = UINT64_C(1) << 8,
    KC_CAP_TICKETS = UINT64_C(1) << 9,
    KC_CAP_DURABLE_STORE = UINT64_C(1) << 32,
    KC_CAP_TRANSPORT = UINT64_C(1) << 33,
    KC_CAP_WORKFLOWS = UINT64_C(1) << 34,
    KC_CAP_SHARED_REGIONS = UINT64_C(1) << 35,
};

typedef struct kc_admin_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t capabilities;
    kc_runtime_snapshot runtime;
    kc_memory_snapshot memory;
    uint64_t terminal_causes[KC_CAUSE_FAILURE + 1];
    uint64_t dropped_telemetry;
} kc_admin_snapshot;

uint64_t kc_runtime_capabilities(const kc_runtime_t *runtime);
int kc_admin_snapshot_get(kc_runtime_t *runtime, kc_admin_snapshot *out);
int kc_admin_list_operations(kc_runtime_t *runtime, kc_op_snapshot *out,
                             size_t capacity, size_t *written, size_t *total);
int kc_admin_list_channels(kc_runtime_t *runtime, kc_channel_snapshot *out,
                           size_t capacity, size_t *written, size_t *total);
int kc_admin_list_descriptors(kc_runtime_t *runtime, kc_descriptor_snapshot *out,
                              size_t capacity, size_t *written, size_t *total);
int kc_admin_list_timers(kc_runtime_t *runtime, kc_op_snapshot *out,
                         size_t capacity, size_t *written, size_t *total);
int kc_admin_list_scopes(kc_runtime_t *runtime, kc_scope_snapshot *out,
                         size_t capacity, size_t *written, size_t *total);

#ifdef __cplusplus
}
#endif

#endif
