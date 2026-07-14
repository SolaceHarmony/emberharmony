// SPDX-License-Identifier: BSD-3-Clause
#include "kc_descriptor_internal.h"
#include "kc_runtime_internal.h"
#include "kcoro_desc.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define KC_DESCRIPTOR_NONE UINT32_MAX
#define KC_SEGMENT_ALIGNMENT 64u

static int aligned_size(size_t value, size_t *out)
{
    if (value > SIZE_MAX - (KC_SEGMENT_ALIGNMENT - 1u)) return -EOVERFLOW;
    *out = (value + KC_SEGMENT_ALIGNMENT - 1u) & ~(size_t)(KC_SEGMENT_ALIGNMENT - 1u);
    return 0;
}

static uint32_t generation_seed(const kc_runtime_t *runtime)
{
    uint32_t seed = (uint32_t)(runtime->epoch ^ (runtime->epoch >> 32));
    return seed ? seed : 1u;
}

static int grow_slots_locked(kc_runtime_t *runtime)
{
    kc_descriptor_registry *registry = &runtime->descriptors;
    uint32_t old = registry->capacity;
    uint32_t capacity = old ? old * 2u : 64u;
    if (capacity < old) return -EOVERFLOW;
    kc_descriptor_slot *slots = realloc(registry->slots,
                                        (size_t)capacity * sizeof(*slots));
    if (!slots) return -ENOMEM;
    registry->slots = slots;
    uint32_t seed = generation_seed(runtime);
    for (uint32_t slot = old; slot < capacity; slot++) {
        slots[slot].descriptor = NULL;
        slots[slot].generation = seed;
        slots[slot].next_free = slot + 1u < capacity
            ? slot + 1u : registry->free_head;
    }
    registry->capacity = capacity;
    registry->free_head = old;
    return 0;
}

static int register_descriptor_locked(kc_runtime_t *runtime,
                                      kc_descriptor_t *descriptor)
{
    kc_descriptor_registry *registry = &runtime->descriptors;
    if (registry->free_head == KC_DESCRIPTOR_NONE) {
        int rc = grow_slots_locked(runtime);
        if (rc != 0) return rc;
    }
    uint32_t slot = registry->free_head;
    kc_descriptor_slot *entry = &registry->slots[slot];
    registry->free_head = entry->next_free;
    entry->descriptor = descriptor;
    entry->next_free = KC_DESCRIPTOR_NONE;
    descriptor->id = (kc_descriptor_id){
        .runtime_epoch = runtime->epoch,
        .slot = slot,
        .generation = entry->generation,
    };
    return 0;
}

static void unregister_descriptor_locked(kc_descriptor_t *descriptor)
{
    kc_descriptor_registry *registry = &descriptor->runtime->descriptors;
    kc_descriptor_slot *entry = &registry->slots[descriptor->id.slot];
    entry->descriptor = NULL;
    entry->generation++;
    if (!entry->generation) entry->generation = 1;
    entry->next_free = registry->free_head;
    registry->free_head = descriptor->id.slot;
}

static void unlink_segment_locked(kc_descriptor_registry *registry,
                                  kc_segment *segment)
{
    kc_segment **link = &registry->segments;
    while (*link && *link != segment) link = &(*link)->next;
    if (*link) *link = segment->next;
    if (registry->active == segment) registry->active = NULL;
    registry->reserved_bytes -= segment->size;
    registry->live_segments--;
    free(segment->base);
    free(segment);
}

static kc_segment *new_segment_locked(kc_runtime_t *runtime, size_t need)
{
    kc_descriptor_registry *registry = &runtime->descriptors;
    size_t capacity = runtime->arena_segment_size;
    if (capacity < need) capacity = need;
    if (aligned_size(capacity, &capacity) != 0) return NULL;
    kc_segment *segment = calloc(1, sizeof(*segment));
    if (!segment) return NULL;
    segment->base = aligned_alloc(KC_SEGMENT_ALIGNMENT, capacity);
    if (!segment->base) { free(segment); return NULL; }
    segment->size = capacity;
    segment->sealed = need > runtime->arena_segment_size;
    segment->next = registry->segments;
    registry->segments = segment;
    registry->live_segments++;
    registry->reserved_bytes += capacity;
    if (!segment->sealed) registry->active = segment;
    return segment;
}

static kc_segment *segment_for_copy_locked(kc_runtime_t *runtime, size_t need)
{
    kc_descriptor_registry *registry = &runtime->descriptors;
    kc_segment *segment = registry->active;
    if (segment && segment->live == 0) {
        registry->reclaimed_bytes += segment->used;
        segment->used = 0;
    }
    if (segment && need <= segment->size - segment->used) return segment;
    if (segment) {
        segment->sealed = 1;
        registry->active = NULL;
        if (segment->live == 0) unlink_segment_locked(registry, segment);
    }
    return new_segment_locked(runtime, need);
}

int kc_descriptor_runtime_init(kc_runtime_t *runtime)
{
    if (!runtime) return -EINVAL;
    runtime->descriptors.free_head = KC_DESCRIPTOR_NONE;
    return KC_MUTEX_INIT(&runtime->descriptors.mu) == 0 ? 0 : -ENOMEM;
}

void kc_descriptor_runtime_destroy(kc_runtime_t *runtime)
{
    if (!runtime || !runtime->descriptors.mu) return;
    kc_descriptor_registry *registry = &runtime->descriptors;
    kc_segment *segment = registry->segments;
    while (segment) {
        kc_segment *next = segment->next;
        free(segment->base);
        free(segment);
        segment = next;
    }
    free(registry->slots);
    KC_MUTEX_DESTROY(&registry->mu);
    memset(registry, 0, sizeof(*registry));
}

int kc_region_create(kc_runtime_t *runtime, const kc_region_config *config,
                     kc_region_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION ||
        (!config->base && config->length) ||
        (config->ownership != KC_REGION_BORROWED &&
         config->ownership != KC_REGION_HOST_OWNED) ||
        (config->ownership == KC_REGION_HOST_OWNED && !config->release) ||
        (config->ownership == KC_REGION_BORROWED && config->release)) return -EINVAL;
    kc_region_t *region = calloc(1, sizeof(*region));
    if (!region) return -ENOMEM;
    atomic_init(&region->refs, 1);
    region->runtime = runtime;
    region->base = config->base;
    region->length = config->length;
    region->provider_id = config->provider_id;
    region->region_id = config->region_id;
    region->generation = config->generation;
    region->ownership = config->ownership;
    region->release = config->release;
    region->release_context = config->release_context;
    region->sequence = kc_runtime_next_sequence(runtime);
    kc_runtime_retain_internal(runtime);

    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    region->next = registry->regions;
    if (registry->regions) registry->regions->prev = region;
    registry->regions = region;
    registry->live_regions++;
    KC_MUTEX_UNLOCK(&registry->mu);
    *out = region;
    return 0;
}

void kc_region_retain(kc_region_t *region)
{
    if (region) atomic_fetch_add_explicit(&region->refs, 1, memory_order_relaxed);
}

void kc_region_release(kc_region_t *region)
{
    if (!region) return;
    kc_runtime_t *runtime = region->runtime;
    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    unsigned previous = atomic_fetch_sub_explicit(&region->refs, 1,
                                                  memory_order_acq_rel);
    if (previous != 1) { KC_MUTEX_UNLOCK(&registry->mu); return; }
    if (region->prev) region->prev->next = region->next;
    else registry->regions = region->next;
    if (region->next) region->next->prev = region->prev;
    registry->live_regions--;
    KC_MUTEX_UNLOCK(&registry->mu);

    if (region->ownership == KC_REGION_HOST_OWNED) {
        region->release(region->release_context, region->base, region->length);
    }
    free(region);
    kc_runtime_release_internal(runtime);
}

int kc_descriptor_create_copy(kc_runtime_t *runtime, const void *source,
                              size_t length, kc_descriptor_t **out)
{
    if (!runtime || !out || (!source && length)) return -EINVAL;
    size_t aligned = 0;
    if (length && aligned_size(length, &aligned) != 0) return -EOVERFLOW;
    kc_descriptor_t *descriptor = calloc(1, sizeof(*descriptor));
    if (!descriptor) return -ENOMEM;
    atomic_init(&descriptor->refs, 1);
    descriptor->runtime = runtime;
    descriptor->kind = KC_DESCRIPTOR_COPY;
    descriptor->length = length;

    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    if (length) {
        descriptor->segment = segment_for_copy_locked(runtime, aligned);
        if (!descriptor->segment) {
            KC_MUTEX_UNLOCK(&registry->mu);
            free(descriptor);
            return -ENOMEM;
        }
        descriptor->data = descriptor->segment->base + descriptor->segment->used;
        descriptor->segment->used += aligned;
        descriptor->segment->live++;
        memcpy(descriptor->data, source, length);
    }
    int rc = register_descriptor_locked(runtime, descriptor);
    if (rc != 0) {
        if (descriptor->segment) {
            descriptor->segment->used -= aligned;
            descriptor->segment->live--;
            if (descriptor->segment->sealed && descriptor->segment->live == 0) {
                unlink_segment_locked(registry, descriptor->segment);
            }
        }
        KC_MUTEX_UNLOCK(&registry->mu);
        free(descriptor);
        return rc;
    }
    registry->live_descriptors++;
    registry->logical_bytes += length;
    registry->cumulative_bytes += length;
    KC_MUTEX_UNLOCK(&registry->mu);
    kc_runtime_retain_internal(runtime);
    *out = descriptor;
    return 0;
}

int kc_descriptor_create_region(kc_runtime_t *runtime, kc_region_t *region,
                                size_t offset, size_t length,
                                kc_descriptor_t **out)
{
    if (!runtime || !region || !out || region->runtime != runtime ||
        offset > region->length || length > region->length - offset) return -EINVAL;
    kc_descriptor_t *descriptor = calloc(1, sizeof(*descriptor));
    if (!descriptor) return -ENOMEM;
    atomic_init(&descriptor->refs, 1);
    descriptor->runtime = runtime;
    descriptor->kind = KC_DESCRIPTOR_REGION;
    descriptor->region = region;
    descriptor->data = offset ? (unsigned char *)region->base + offset
                              : region->base;
    descriptor->length = length;
    kc_region_retain(region);

    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    int rc = register_descriptor_locked(runtime, descriptor);
    if (rc == 0) {
        registry->live_descriptors++;
        registry->logical_bytes += length;
        registry->cumulative_bytes += length;
    }
    KC_MUTEX_UNLOCK(&registry->mu);
    if (rc != 0) {
        kc_region_release(region);
        free(descriptor);
        return rc;
    }
    kc_runtime_retain_internal(runtime);
    *out = descriptor;
    return 0;
}

void kc_descriptor_retain(kc_descriptor_t *descriptor)
{
    if (descriptor) atomic_fetch_add_explicit(&descriptor->refs, 1,
                                              memory_order_relaxed);
}

void kc_descriptor_release(kc_descriptor_t *descriptor)
{
    if (!descriptor) return;
    kc_runtime_t *runtime = descriptor->runtime;
    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    unsigned previous = atomic_fetch_sub_explicit(&descriptor->refs, 1,
                                                  memory_order_acq_rel);
    if (previous != 1) { KC_MUTEX_UNLOCK(&registry->mu); return; }
    unregister_descriptor_locked(descriptor);
    registry->live_descriptors--;
    registry->logical_bytes -= descriptor->length;
    if (descriptor->segment) {
        descriptor->segment->live--;
        if (descriptor->segment->live == 0) {
            registry->reclaimed_bytes += descriptor->segment->used;
            if (descriptor->segment == registry->active) {
                descriptor->segment->used = 0;
            } else {
                unlink_segment_locked(registry, descriptor->segment);
            }
        }
    }
    kc_region_t *region = descriptor->region;
    KC_MUTEX_UNLOCK(&registry->mu);
    if (region) kc_region_release(region);
    free(descriptor);
    kc_runtime_release_internal(runtime);
}

kc_descriptor_id kc_descriptor_id_get(const kc_descriptor_t *descriptor)
{
    return descriptor ? descriptor->id : (kc_descriptor_id){0};
}

int kc_descriptor_lookup(kc_runtime_t *runtime, kc_descriptor_id id,
                         kc_descriptor_t **out)
{
    if (!runtime || !out || id.runtime_epoch != runtime->epoch) return -ENOENT;
    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    if (id.slot >= registry->capacity) { KC_MUTEX_UNLOCK(&registry->mu); return -ENOENT; }
    kc_descriptor_slot *entry = &registry->slots[id.slot];
    if (!entry->descriptor || entry->generation != id.generation) {
        KC_MUTEX_UNLOCK(&registry->mu);
        return -ENOENT;
    }
    kc_descriptor_retain(entry->descriptor);
    *out = entry->descriptor;
    KC_MUTEX_UNLOCK(&registry->mu);
    return 0;
}

uint64_t kc_descriptor_legacy_id(const kc_descriptor_t *descriptor)
{
    if (!descriptor) return 0;
    return ((uint64_t)descriptor->id.generation << 32) |
           ((uint64_t)descriptor->id.slot + 1u);
}

int kc_descriptor_payload(const kc_descriptor_t *descriptor, kc_payload *out)
{
    if (!descriptor || !out) return -EINVAL;
    *out = (kc_payload){
        .ptr = descriptor->data,
        .len = descriptor->length,
        .status = 0,
        .desc_id = kc_descriptor_legacy_id(descriptor),
    };
    return 0;
}

int kc_descriptor_snapshot_get(const kc_descriptor_t *descriptor,
                               kc_descriptor_snapshot *out)
{
    if (!descriptor || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_descriptor_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = descriptor->id, .kind = descriptor->kind,
        .length = descriptor->length,
        .references = atomic_load_explicit(&descriptor->refs, memory_order_acquire),
        .provider_id = descriptor->region ? descriptor->region->provider_id : 0,
        .region_id = descriptor->region ? descriptor->region->region_id : 0,
        .region_generation = descriptor->region ? descriptor->region->generation : 0,
    };
    return 0;
}

int kc_memory_snapshot_get(kc_runtime_t *runtime, kc_memory_snapshot *out)
{
    if (!runtime || !out || out->size < sizeof(*out)) return -EINVAL;
    kc_descriptor_registry *registry = &runtime->descriptors;
    KC_MUTEX_LOCK(&registry->mu);
    *out = (kc_memory_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .live_descriptors = registry->live_descriptors,
        .live_regions = registry->live_regions,
        .live_segments = registry->live_segments,
        .logical_bytes = registry->logical_bytes,
        .reserved_bytes = registry->reserved_bytes,
        .cumulative_bytes = registry->cumulative_bytes,
        .reclaimed_bytes = registry->reclaimed_bytes,
    };
    KC_MUTEX_UNLOCK(&registry->mu);
    return 0;
}

static kc_descriptor_id legacy_id(kc_runtime_t *runtime, kc_desc_id id)
{
    return (kc_descriptor_id){
        .runtime_epoch = runtime ? runtime->epoch : 0,
        .slot = id ? (uint32_t)id - 1u : 0,
        .generation = (uint32_t)(id >> 32),
    };
}

int kc_desc_global_init(void) { return kc_runtime_default_get() ? 0 : -ENOMEM; }
void kc_desc_global_shutdown(void) { }

kc_desc_id kc_desc_make_alias(void *ptr, size_t length)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    if (!runtime) return 0;
    kc_region_config config = {
        .size = sizeof(config), .abi_version = KC_ABI_VERSION,
        .base = ptr, .length = length, .ownership = KC_REGION_BORROWED,
    };
    kc_region_t *region = NULL;
    kc_descriptor_t *descriptor = NULL;
    if (kc_region_create(runtime, &config, &region) != 0 ||
        kc_descriptor_create_region(runtime, region, 0, length, &descriptor) != 0) {
        kc_region_release(region);
        return 0;
    }
    kc_region_release(region);
    return kc_descriptor_legacy_id(descriptor);
}

kc_desc_id kc_desc_make_copy(const void *source, size_t length)
{
    kc_descriptor_t *descriptor = NULL;
    if (kc_descriptor_create_copy(kc_runtime_default_get(), source, length,
                                  &descriptor) != 0) return 0;
    return kc_descriptor_legacy_id(descriptor);
}

void kc_desc_retain(kc_desc_id id)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    kc_descriptor_t *descriptor = NULL;
    (void)kc_descriptor_lookup(runtime, legacy_id(runtime, id), &descriptor);
}

void kc_desc_release(kc_desc_id id)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    kc_descriptor_t *descriptor = NULL;
    if (kc_descriptor_lookup(runtime, legacy_id(runtime, id), &descriptor) != 0) return;
    kc_descriptor_release(descriptor);
    kc_descriptor_release(descriptor);
}

int kc_desc_payload(kc_desc_id id, kc_payload *out)
{
    kc_runtime_t *runtime = kc_runtime_default_get();
    kc_descriptor_t *descriptor = NULL;
    int rc = kc_descriptor_lookup(runtime, legacy_id(runtime, id), &descriptor);
    if (rc != 0) return rc;
    rc = kc_descriptor_payload(descriptor, out);
    kc_descriptor_release(descriptor);
    return rc;
}
