// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_ACTOR_H
#define KC_ACTOR_H

#include "kc_channel.h"
#include "kc_scope.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_actor kc_actor_t;
typedef int (*kc_actor_receive_fn)(void *context, const void *data, size_t length);

typedef struct kc_actor_config {
    uint32_t size;
    uint32_t abi_version;
    kc_channel_kind mailbox_kind;
    size_t mailbox_capacity;
    size_t element_size;
    kc_cancel_t *parent_cancel;
    kc_actor_receive_fn receive;
    void *context;
} kc_actor_config;

typedef struct kc_actor_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id id;
    uint64_t messages;
    uint64_t failures;
    uint64_t callback_ns;
    uint64_t max_callback_ns;
    size_t mailbox_depth;
    unsigned closed;
} kc_actor_snapshot;

int kc_actor_create(kc_runtime_t *runtime, const kc_actor_config *config,
                    kc_actor_t **out);
void kc_actor_retain(kc_actor_t *actor);
void kc_actor_release(kc_actor_t *actor);
kc_channel_t *kc_actor_mailbox(kc_actor_t *actor);
void kc_actor_close(kc_actor_t *actor);
int kc_actor_join(kc_actor_t *actor, uint64_t deadline_ns);
int kc_actor_snapshot_get(kc_actor_t *actor, kc_actor_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
