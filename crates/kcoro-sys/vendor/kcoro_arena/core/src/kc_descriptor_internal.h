// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_descriptor.h"
#include "kcoro_port.h"

#include <stdatomic.h>
#include <stdint.h>

typedef struct kc_segment kc_segment;
typedef struct kc_descriptor_slot kc_descriptor_slot;

struct kc_segment {
    kc_segment *next;
    unsigned char *base;
    size_t size;
    size_t used;
    size_t live;
    int sealed;
};

struct kc_region {
    atomic_uint refs;
    kc_runtime_t *runtime;
    struct kc_region *prev;
    struct kc_region *next;
    void *base;
    size_t length;
    uint64_t sequence;
    uint64_t provider_id;
    uint64_t region_id;
    uint32_t generation;
    kc_region_ownership ownership;
    kc_region_release_fn release;
    void *release_context;
};

struct kc_descriptor {
    atomic_uint refs;
    kc_runtime_t *runtime;
    kc_descriptor_id id;
    kc_descriptor_kind kind;
    void *data;
    size_t length;
    kc_segment *segment;
    kc_region_t *region;
};

struct kc_descriptor_slot {
    kc_descriptor_t *descriptor;
    uint32_t generation;
    uint32_t next_free;
};

typedef struct kc_descriptor_registry {
    KC_MUTEX_T mu;
    kc_descriptor_slot *slots;
    uint32_t capacity;
    uint32_t free_head;
    kc_segment *segments;
    kc_segment *active;
    kc_region_t *regions;
    size_t live_descriptors;
    size_t live_regions;
    size_t live_segments;
    size_t logical_bytes;
    size_t reserved_bytes;
    uint64_t cumulative_bytes;
    uint64_t reclaimed_bytes;
} kc_descriptor_registry;

int kc_descriptor_runtime_init(kc_runtime_t *runtime);
void kc_descriptor_runtime_destroy(kc_runtime_t *runtime);
uint64_t kc_descriptor_legacy_id(const kc_descriptor_t *descriptor);
