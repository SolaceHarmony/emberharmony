// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_SHARED_H
#define KC_SHARED_H

#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KC_SHARED_PAYLOAD_WIRE_SIZE 56u

typedef struct kc_region_provider kc_region_provider;
typedef struct kc_shared_lease kc_shared_lease_t;

typedef struct kc_shared_payload {
    uint32_t size;
    uint32_t abi_version;
    uint64_t provider_id;
    uint64_t region_id;
    uint64_t offset;
    uint64_t length;
    uint32_t format;
    uint32_t generation;
} kc_shared_payload;

/* Direct link-time host contract. The core ships no provider implementation. */
int kc_region_provider_acquire(kc_region_provider *provider,
                               const kc_shared_payload *payload,
                               const void **data, void **host_lease);
void kc_region_provider_release(kc_region_provider *provider, void *host_lease);

int kc_shared_payload_encode(const kc_shared_payload *payload,
                             unsigned char wire[KC_SHARED_PAYLOAD_WIRE_SIZE]);
int kc_shared_payload_decode(const unsigned char *wire, size_t length,
                             kc_shared_payload *payload);
int kc_shared_payload_acquire(kc_region_provider *provider,
                              const kc_shared_payload *payload,
                              kc_shared_lease_t **lease,
                              const void **data, size_t *length);
void kc_shared_payload_release(kc_shared_lease_t *lease);

#ifdef __cplusplus
}
#endif

#endif
