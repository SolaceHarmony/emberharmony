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
struct koro_cont;
typedef void *(*kc_runtime_step_fn)(struct koro_cont *cont);

typedef struct kc_runtime_config {
    uint32_t size;
    uint32_t abi_version;
    unsigned worker_count;
    size_t arena_segment_size;
    uint32_t ticket_capacity;
    uint32_t reserved;
} kc_runtime_config;

typedef struct kc_runtime_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t epoch;
    uint64_t next_sequence;
    size_t active;
    size_t queued;
    size_t running;
    size_t waiting;
    size_t live_operations;
    size_t live_timers;
    size_t live_channels;
    size_t live_scopes;
    size_t live_tickets;
    size_t completion_queued;
    size_t completion_running;
    size_t live_descriptors;
    size_t live_regions;
    size_t live_segments;
    size_t reserved_bytes;
    uint64_t wake_requests;
    uint64_t resumes;
    unsigned workers;
    unsigned accepting;
    unsigned started;
    unsigned stop_requested;
    unsigned ticket_capacity;
} kc_runtime_snapshot;

int kc_runtime_create(const kc_runtime_config *config, kc_runtime_t **out);
int kc_runtime_start(kc_runtime_t *runtime);
int kc_runtime_spawn(kc_runtime_t *runtime, kc_runtime_step_fn step,
                     void *arg, size_t local_size);
/* Wait until no work is runnable. Parked continuations may remain active. */
int kc_runtime_run_until_idle(kc_runtime_t *runtime);
/* Wait until tracked continuations and all operation leases are released. */
int kc_runtime_join_all(kc_runtime_t *runtime);
void kc_runtime_request_stop(kc_runtime_t *runtime);
int kc_runtime_join(kc_runtime_t *runtime);
int kc_runtime_destroy(kc_runtime_t *runtime);
int kc_runtime_snapshot_get(kc_runtime_t *runtime, kc_runtime_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
