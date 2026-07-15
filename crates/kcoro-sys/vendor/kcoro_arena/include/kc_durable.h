// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_DURABLE_H
#define KC_DURABLE_H

#include "kc_op.h"
#include "kc_wal.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_durable kc_durable_t;

typedef enum kc_message_state {
    KC_MESSAGE_PENDING = 1,
    KC_MESSAGE_IN_FLIGHT,
    KC_MESSAGE_ACKNOWLEDGED,
    KC_MESSAGE_DEAD_LETTERED,
} kc_message_state;

typedef struct kc_durable_config {
    uint32_t size;
    uint32_t abi_version;
    kc_wal_t *wal;
    size_t dedupe_capacity;
    uint32_t max_delivery_attempts;
    uint32_t reserved;
} kc_durable_config;

typedef struct kc_publish {
    uint32_t size;
    uint32_t abi_version;
    uint64_t route;
    kc_id correlation_id;
    kc_id trace_id;
    kc_id idempotency_key;
    const void *payload;
    size_t payload_length;
} kc_publish;

typedef struct kc_message {
    uint32_t size;
    uint32_t abi_version;
    kc_id id;
    kc_id correlation_id;
    kc_id trace_id;
    kc_id idempotency_key;
    uint64_t route;
    uint32_t delivery_attempt;
    kc_message_state state;
    int32_t terminal_reason;
    uint32_t reserved;
    const void *payload;
    size_t payload_length;
} kc_message;

typedef struct kc_durable_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t runtime_epoch;
    uint64_t next_message_sequence;
    size_t pending;
    size_t in_flight;
    size_t acknowledged;
    size_t dead_lettered;
    size_t dedupe_entries;
    size_t logical_bytes;
    uint64_t publications;
    uint64_t delivery_attempts;
    uint64_t redeliveries;
} kc_durable_snapshot;

int kc_durable_create(const kc_durable_config *config, kc_durable_t **out);
void kc_durable_destroy(kc_durable_t *durable);
int kc_durable_publish(kc_durable_t *durable, const kc_publish *publish,
                       kc_id *message_id);
int kc_durable_next(kc_durable_t *durable, uint64_t route, kc_message *out);
int kc_durable_acknowledge(kc_durable_t *durable, kc_id message_id);
int kc_durable_retry(kc_durable_t *durable, kc_id message_id);
int kc_durable_dead_letter(kc_durable_t *durable, kc_id message_id, int reason);
/* The returned payload is borrowed and remains valid until durable is
 * destroyed. State fields are a point-in-time snapshot. */
int kc_durable_lookup(kc_durable_t *durable, kc_id message_id, kc_message *out);
int kc_durable_checkpoint(kc_durable_t *durable);
int kc_durable_snapshot_get(kc_durable_t *durable, kc_durable_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
