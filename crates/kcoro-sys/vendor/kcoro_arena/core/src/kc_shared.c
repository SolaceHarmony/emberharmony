// SPDX-License-Identifier: BSD-3-Clause
#include "kc_shared.h"
#include "kc_codec_internal.h"

#include <errno.h>
#include <stdlib.h>

#define SHARED_MAGIC UINT32_C(0x4853434b)
#define SHARED_FORMAT_VERSION 1u

struct kc_shared_lease {
    kc_region_provider *provider;
    void *host_lease;
};

static int payload_valid(const kc_shared_payload *payload)
{
    if (!payload || payload->size < sizeof(*payload) ||
        payload->abi_version != KC_ABI_VERSION || !payload->provider_id ||
        !payload->region_id || !payload->generation ||
        payload->length > SIZE_MAX) return 0;
    return payload->offset <= UINT64_MAX - payload->length;
}

int kc_shared_payload_encode(const kc_shared_payload *payload,
                             unsigned char wire[KC_SHARED_PAYLOAD_WIRE_SIZE])
{
    if (!wire || !payload_valid(payload)) return -EINVAL;
    for (size_t index = 0; index < KC_SHARED_PAYLOAD_WIRE_SIZE; index++) {
        wire[index] = 0;
    }
    kc_put_u32(wire, SHARED_MAGIC);
    kc_put_u16(wire + 4, SHARED_FORMAT_VERSION);
    kc_put_u16(wire + 6, KC_SHARED_PAYLOAD_WIRE_SIZE);
    kc_put_u64(wire + 8, payload->provider_id);
    kc_put_u64(wire + 16, payload->region_id);
    kc_put_u64(wire + 24, payload->offset);
    kc_put_u64(wire + 32, payload->length);
    kc_put_u32(wire + 40, payload->format);
    kc_put_u32(wire + 44, payload->generation);
    return 0;
}

int kc_shared_payload_decode(const unsigned char *wire, size_t length,
                             kc_shared_payload *payload)
{
    if (!wire || !payload || payload->size < sizeof(*payload)) return -EINVAL;
    if (length != KC_SHARED_PAYLOAD_WIRE_SIZE ||
        kc_get_u32(wire) != SHARED_MAGIC ||
        kc_get_u16(wire + 4) != SHARED_FORMAT_VERSION ||
        kc_get_u16(wire + 6) != KC_SHARED_PAYLOAD_WIRE_SIZE) return -EBADMSG;
    *payload = (kc_shared_payload){
        .size = sizeof(*payload), .abi_version = KC_ABI_VERSION,
        .provider_id = kc_get_u64(wire + 8),
        .region_id = kc_get_u64(wire + 16),
        .offset = kc_get_u64(wire + 24),
        .length = kc_get_u64(wire + 32),
        .format = kc_get_u32(wire + 40),
        .generation = kc_get_u32(wire + 44),
    };
    return payload_valid(payload) ? 0 : -EBADMSG;
}

int kc_shared_payload_acquire(kc_region_provider *provider,
                              const kc_shared_payload *payload,
                              kc_shared_lease_t **lease,
                              const void **data, size_t *length)
{
    if (!provider || !lease || !data || !length || !payload_valid(payload)) {
        return -EINVAL;
    }
    kc_shared_lease_t *owned = calloc(1, sizeof(*owned));
    if (!owned) return -ENOMEM;
    const void *mapped = NULL;
    void *host_lease = NULL;
    int rc = kc_region_provider_acquire(provider, payload, &mapped, &host_lease);
    if (rc != 0) { free(owned); return rc; }
    if (payload->length && !mapped) {
        kc_region_provider_release(provider, host_lease);
        free(owned);
        return -EFAULT;
    }
    owned->provider = provider;
    owned->host_lease = host_lease;
    *lease = owned;
    *data = mapped;
    *length = (size_t)payload->length;
    return 0;
}

void kc_shared_payload_release(kc_shared_lease_t *lease)
{
    if (!lease) return;
    kc_region_provider_release(lease->provider, lease->host_lease);
    free(lease);
}
