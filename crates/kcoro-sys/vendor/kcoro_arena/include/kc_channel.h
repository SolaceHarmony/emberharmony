// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_CHANNEL_H
#define KC_CHANNEL_H

#include "kc_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_chan kc_channel_t;

typedef enum kc_channel_kind {
    KC_CHANNEL_RENDEZVOUS = 0,
    KC_CHANNEL_BUFFERED = 1,
    KC_CHANNEL_CONFLATED = -1,
    KC_CHANNEL_UNLIMITED = -2,
} kc_channel_kind;

typedef struct kc_channel_config {
    uint32_t size;
    uint32_t abi_version;
    kc_channel_kind kind;
    size_t element_size;
    size_t capacity;
} kc_channel_config;

typedef struct kc_channel_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t runtime_epoch;
    uint64_t sequence;
    kc_channel_kind kind;
    size_t element_size;
    size_t depth;
    size_t send_waiters;
    size_t receive_waiters;
    size_t logical_bytes;
    unsigned closed;
} kc_channel_snapshot;

int kc_channel_create(kc_runtime_t *runtime, const kc_channel_config *config,
                      kc_channel_t **out);
void kc_channel_retain(kc_channel_t *channel);
void kc_channel_close(kc_channel_t *channel);
void kc_channel_release(kc_channel_t *channel);
size_t kc_channel_length(kc_channel_t *channel);
int kc_channel_snapshot_get(kc_channel_t *channel, kc_channel_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
