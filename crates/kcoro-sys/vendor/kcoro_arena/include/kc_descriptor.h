// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_DESCRIPTOR_H
#define KC_DESCRIPTOR_H

#include "kc_payload.h"
#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_descriptor kc_descriptor_t;
typedef struct kc_region kc_region_t;

typedef struct kc_descriptor_id {
    uint64_t runtime_epoch;
    uint32_t slot;
    uint32_t generation;
} kc_descriptor_id;

typedef enum kc_descriptor_kind {
    KC_DESCRIPTOR_COPY = 1,
    KC_DESCRIPTOR_REGION = 2,
} kc_descriptor_kind;

typedef enum kc_region_ownership {
    KC_REGION_BORROWED = 1,
    KC_REGION_HOST_OWNED = 2,
} kc_region_ownership;

typedef void (*kc_region_release_fn)(void *context, void *base, size_t length);

typedef struct kc_region_config {
    uint32_t size;
    uint32_t abi_version;
    void *base;
    size_t length;
    uint64_t provider_id;
    uint64_t region_id;
    uint32_t generation;
    kc_region_ownership ownership;
    kc_region_release_fn release;
    void *release_context;
} kc_region_config;

typedef struct kc_descriptor_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_descriptor_id id;
    kc_descriptor_kind kind;
    size_t length;
    unsigned references;
    uint64_t provider_id;
    uint64_t region_id;
    uint32_t region_generation;
} kc_descriptor_snapshot;

typedef struct kc_memory_snapshot {
    uint32_t size;
    uint32_t abi_version;
    size_t live_descriptors;
    size_t live_regions;
    size_t live_segments;
    size_t logical_bytes;
    size_t reserved_bytes;
    uint64_t cumulative_bytes;
    uint64_t reclaimed_bytes;
} kc_memory_snapshot;

int kc_region_create(kc_runtime_t *runtime, const kc_region_config *config,
                     kc_region_t **out);
void kc_region_retain(kc_region_t *region);
void kc_region_release(kc_region_t *region);

int kc_descriptor_create_copy(kc_runtime_t *runtime, const void *source,
                              size_t length, kc_descriptor_t **out);
int kc_descriptor_create_region(kc_runtime_t *runtime, kc_region_t *region,
                                size_t offset, size_t length,
                                kc_descriptor_t **out);
void kc_descriptor_retain(kc_descriptor_t *descriptor);
void kc_descriptor_release(kc_descriptor_t *descriptor);
kc_descriptor_id kc_descriptor_id_get(const kc_descriptor_t *descriptor);
int kc_descriptor_lookup(kc_runtime_t *runtime, kc_descriptor_id id,
                         kc_descriptor_t **out);
int kc_descriptor_payload(const kc_descriptor_t *descriptor, kc_payload *out);
int kc_descriptor_snapshot_get(const kc_descriptor_t *descriptor,
                               kc_descriptor_snapshot *out);
int kc_memory_snapshot_get(kc_runtime_t *runtime, kc_memory_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
