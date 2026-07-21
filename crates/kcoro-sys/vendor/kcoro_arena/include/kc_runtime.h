// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_RUNTIME_H
#define KC_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KC_ABI_VERSION 1u

typedef struct kc_runtime kc_runtime_t;

typedef struct kc_runtime_config {
    uint32_t size;
    uint32_t abi_version;
    unsigned worker_count;
    uint32_t reserved;
} kc_runtime_config;

typedef struct kc_runtime_snapshot {
    uint32_t size;
    uint32_t abi_version;
    size_t active;
    size_t queued;
    size_t running;
    size_t dormant;
    uint64_t wake_requests;
    uint64_t resumes;
    unsigned workers;
    unsigned accepting;
    unsigned started;
    unsigned stop_requested;
} kc_runtime_snapshot;

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out);
int kc_runtime_start(kc_runtime_t *runtime);
/* Administrative join for retained services and their notifier leases.
 * Returns -EDEADLK from a callback executing on this runtime. */
int kc_runtime_join_all(kc_runtime_t *runtime);
void kc_runtime_request_stop(kc_runtime_t *runtime);
/* Returns -EDEADLK from a callback executing on this runtime. */
int kc_runtime_join(kc_runtime_t *runtime);
int kc_runtime_destroy(kc_runtime_t *runtime);
int kc_runtime_snapshot_get(kc_runtime_t *runtime, kc_runtime_snapshot *out);
/* Valid only from a continuation callback running on this runtime. */
int kc_runtime_current_worker(const kc_runtime_t *runtime,
                              uint32_t *out_worker);

#ifdef __cplusplus
}
#endif

#endif
