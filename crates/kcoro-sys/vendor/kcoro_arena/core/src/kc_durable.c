// SPDX-License-Identifier: BSD-3-Clause
#include "kc_durable.h"
#include "kc_checkpoint_internal.h"
#include "kc_codec_internal.h"
#include "kc_durable_internal.h"
#include "kcoro_port.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum {
    DURABLE_FORMAT_VERSION = 1,
    DURABLE_PUBLISH_RECORD = 0x200,
    DURABLE_ATTEMPT_RECORD,
    DURABLE_ACK_RECORD,
    DURABLE_RETRY_RECORD,
    DURABLE_DEAD_RECORD,
    PUBLISH_FIXED_SIZE = 80,
    EVENT_SIZE = 40,
    DURABLE_SNAPSHOT_HEADER_SIZE = 32,
    DURABLE_SNAPSHOT_MESSAGE_SIZE = 96,
};

#define DURABLE_SNAPSHOT_MAGIC UINT32_C(0x4d44434b)

typedef struct message_node {
    kc_id id;
    kc_id correlation_id;
    kc_id trace_id;
    kc_id idempotency_key;
    uint64_t route;
    uint32_t attempt;
    kc_message_state state;
    int32_t terminal_reason;
    void *payload;
    size_t payload_length;
    struct message_node *next;
} message_node;

typedef struct dedupe_entry {
    kc_id key;
    kc_id message_id;
} dedupe_entry;

struct kc_durable {
    KC_MUTEX_T mu;
    kc_wal_t *wal;
    message_node *head;
    message_node *tail;
    dedupe_entry *dedupe;
    size_t dedupe_capacity;
    size_t dedupe_count;
    size_t dedupe_next;
    uint64_t epoch;
    uint64_t next_message_sequence;
    uint32_t max_delivery_attempts;
    size_t pending;
    size_t in_flight;
    size_t acknowledged;
    size_t dead_lettered;
    size_t logical_bytes;
    uint64_t publications;
    uint64_t delivery_attempts;
    uint64_t redeliveries;
    size_t workflow_users;
};

struct kc_durable_batch {
    kc_durable_t *durable;
    kc_wal_tx_t *transaction;
    message_node **messages;
    size_t message_count;
    size_t message_capacity;
    message_node **acknowledgements;
    size_t acknowledgement_count;
    size_t acknowledgement_capacity;
};

static int id_equal(kc_id left, kc_id right)
{
    return left.epoch == right.epoch && left.sequence == right.sequence;
}

static int id_empty(kc_id id)
{
    return id.epoch == 0 && id.sequence == 0;
}

static void encode_id(unsigned char *data, kc_id id)
{
    kc_put_u64(data, id.epoch);
    kc_put_u64(data + 8, id.sequence);
}

static kc_id decode_id(const unsigned char *data)
{
    return (kc_id){ kc_get_u64(data), kc_get_u64(data + 8) };
}

static message_node *message_find(kc_durable_t *durable, kc_id id)
{
    for (message_node *message = durable->head; message; message = message->next) {
        if (id_equal(message->id, id)) return message;
    }
    return NULL;
}

static void message_append(kc_durable_t *durable, message_node *message)
{
    message->next = NULL;
    if (durable->tail) durable->tail->next = message;
    else durable->head = message;
    durable->tail = message;
    durable->pending++;
    durable->logical_bytes += message->payload_length;
    durable->publications++;
    if (message->id.sequence >= durable->next_message_sequence) {
        durable->next_message_sequence = message->id.sequence + 1;
    }
}

static void dedupe_add(kc_durable_t *durable, kc_id key, kc_id message_id)
{
    if (id_empty(key) || !durable->dedupe_capacity) return;
    for (size_t index = 0; index < durable->dedupe_count; index++) {
        if (id_equal(durable->dedupe[index].key, key)) {
            durable->dedupe[index].message_id = message_id;
            return;
        }
    }
    size_t index;
    if (durable->dedupe_count < durable->dedupe_capacity) {
        index = durable->dedupe_count++;
    } else {
        index = durable->dedupe_next;
        durable->dedupe_next = (durable->dedupe_next + 1) %
                               durable->dedupe_capacity;
    }
    durable->dedupe[index] = (dedupe_entry){ key, message_id };
}

static int dedupe_find(kc_durable_t *durable, kc_id key, kc_id *message_id)
{
    if (id_empty(key)) return 0;
    for (size_t index = 0; index < durable->dedupe_count; index++) {
        if (id_equal(durable->dedupe[index].key, key)) {
            *message_id = durable->dedupe[index].message_id;
            return 1;
        }
    }
    return 0;
}

static message_node *message_create(kc_id id, kc_id correlation, kc_id trace,
                                    kc_id key, uint64_t route,
                                    const void *payload, size_t length)
{
    message_node *message = calloc(1, sizeof(*message));
    if (!message) return NULL;
    if (length) {
        message->payload = malloc(length);
        if (!message->payload) { free(message); return NULL; }
        memcpy(message->payload, payload, length);
    }
    message->id = id;
    message->correlation_id = correlation;
    message->trace_id = trace;
    message->idempotency_key = key;
    message->route = route;
    message->state = KC_MESSAGE_PENDING;
    message->payload_length = length;
    return message;
}

static void message_destroy(message_node *message)
{
    if (!message) return;
    free(message->payload);
    free(message);
}

static int encode_publish(const message_node *message, unsigned char **out,
                          size_t *length)
{
    if (message->payload_length > UINT32_MAX ||
        message->payload_length > SIZE_MAX - PUBLISH_FIXED_SIZE) return -E2BIG;
    size_t total = PUBLISH_FIXED_SIZE + message->payload_length;
    unsigned char *data = calloc(1, total);
    if (!data) return -ENOMEM;
    kc_put_u16(data, DURABLE_FORMAT_VERSION);
    kc_put_u32(data + 4, (uint32_t)message->payload_length);
    encode_id(data + 8, message->id);
    encode_id(data + 24, message->correlation_id);
    encode_id(data + 40, message->idempotency_key);
    kc_put_u64(data + 56, message->route);
    encode_id(data + 64, message->trace_id);
    if (message->payload_length) {
        memcpy(data + PUBLISH_FIXED_SIZE, message->payload,
               message->payload_length);
    }
    *out = data;
    *length = total;
    return 0;
}

static void encode_event(unsigned char data[EVENT_SIZE], message_node *message,
                         uint32_t value)
{
    memset(data, 0, EVENT_SIZE);
    kc_put_u16(data, DURABLE_FORMAT_VERSION);
    kc_put_u32(data + 4, value);
    encode_id(data + 8, message->id);
    encode_id(data + 24, message->trace_id);
}

static int persist(kc_durable_t *durable, uint16_t type,
                   const void *payload, size_t length)
{
    kc_wal_tx_t *transaction = NULL;
    int rc = kc_wal_transaction_begin(durable->wal, &transaction);
    if (rc != 0) return rc;
    rc = kc_wal_transaction_append(transaction, type, payload, length);
    if (rc != 0) {
        kc_wal_transaction_abort(transaction);
        return rc;
    }
    return kc_wal_transaction_commit(transaction);
}

static int decode_publish(kc_durable_t *durable, const kc_wal_record *record)
{
    if (record->payload_length < PUBLISH_FIXED_SIZE) return -EBADMSG;
    const unsigned char *data = record->payload;
    uint32_t length = kc_get_u32(data + 4);
    if (kc_get_u16(data) != DURABLE_FORMAT_VERSION ||
        record->payload_length != PUBLISH_FIXED_SIZE + (uint64_t)length) {
        return -EBADMSG;
    }
    kc_id id = decode_id(data + 8);
    if (id.epoch != durable->epoch || !id.sequence ||
        id.sequence == UINT64_MAX || message_find(durable, id)) {
        return -EBADMSG;
    }
    message_node *message = message_create(
        id, decode_id(data + 24), decode_id(data + 64), decode_id(data + 40),
        kc_get_u64(data + 56), data + PUBLISH_FIXED_SIZE, length);
    if (!message) return -ENOMEM;
    message_append(durable, message);
    dedupe_add(durable, message->idempotency_key, message->id);
    return 0;
}

static int decode_event(kc_durable_t *durable, const kc_wal_record *record)
{
    if (record->payload_length != EVENT_SIZE) return -EBADMSG;
    const unsigned char *data = record->payload;
    if (kc_get_u16(data) != DURABLE_FORMAT_VERSION) return -EBADMSG;
    message_node *message = message_find(durable, decode_id(data + 8));
    if (!message || !id_equal(message->trace_id, decode_id(data + 24))) {
        return -EBADMSG;
    }
    uint32_t value = kc_get_u32(data + 4);
    switch (record->type) {
    case DURABLE_ATTEMPT_RECORD:
        if (message->state == KC_MESSAGE_ACKNOWLEDGED ||
            message->state == KC_MESSAGE_DEAD_LETTERED ||
            value <= message->attempt) return -EBADMSG;
        if (message->state == KC_MESSAGE_PENDING) {
            durable->pending--;
            durable->in_flight++;
        }
        message->attempt = value;
        message->state = KC_MESSAGE_IN_FLIGHT;
        message->terminal_reason = 0;
        durable->delivery_attempts++;
        return 0;
    case DURABLE_ACK_RECORD:
        if (message->state == KC_MESSAGE_ACKNOWLEDGED) return 0;
        if (message->state == KC_MESSAGE_DEAD_LETTERED) return -EBADMSG;
        if (message->state == KC_MESSAGE_PENDING) durable->pending--;
        else durable->in_flight--;
        durable->acknowledged++;
        message->state = KC_MESSAGE_ACKNOWLEDGED;
        message->terminal_reason = 0;
        return 0;
    case DURABLE_RETRY_RECORD:
        if (message->state == KC_MESSAGE_ACKNOWLEDGED ||
            message->state == KC_MESSAGE_DEAD_LETTERED) return -EBADMSG;
        if (message->state == KC_MESSAGE_IN_FLIGHT) {
            durable->in_flight--;
            durable->pending++;
        }
        message->state = KC_MESSAGE_PENDING;
        message->terminal_reason = 0;
        return 0;
    case DURABLE_DEAD_RECORD:
        if (message->state == KC_MESSAGE_DEAD_LETTERED) return 0;
        if (message->state == KC_MESSAGE_ACKNOWLEDGED) return -EBADMSG;
        if (message->state == KC_MESSAGE_PENDING) durable->pending--;
        else durable->in_flight--;
        durable->dead_lettered++;
        message->state = KC_MESSAGE_DEAD_LETTERED;
        message->terminal_reason = kc_get_i32(data + 4);
        return 0;
    default:
        return 0;
    }
}

static int recover_record(void *context, const kc_wal_record *record)
{
    kc_durable_t *durable = context;
    if (record->type == DURABLE_PUBLISH_RECORD) {
        return decode_publish(durable, record);
    }
    if (record->type >= DURABLE_ATTEMPT_RECORD &&
        record->type <= DURABLE_DEAD_RECORD) return decode_event(durable, record);
    return 0;
}

static int snapshot_decode(kc_durable_t *durable, const void *payload,
                           size_t length)
{
    if (length < DURABLE_SNAPSHOT_HEADER_SIZE) return -EBADMSG;
    const unsigned char *data = payload;
    if (kc_get_u32(data) != DURABLE_SNAPSHOT_MAGIC ||
        kc_get_u16(data + 4) != DURABLE_FORMAT_VERSION ||
        kc_get_u16(data + 6) != DURABLE_SNAPSHOT_HEADER_SIZE ||
        kc_get_u64(data + 8) != durable->epoch) return -EBADMSG;
    uint64_t next_sequence = kc_get_u64(data + 16);
    uint32_t count = kc_get_u32(data + 24);
    size_t offset = DURABLE_SNAPSHOT_HEADER_SIZE;
    for (uint32_t index = 0; index < count; index++) {
        if (length - offset < DURABLE_SNAPSHOT_MESSAGE_SIZE) return -EBADMSG;
        const unsigned char *encoded = data + offset;
        uint32_t record_size = kc_get_u32(encoded);
        uint16_t state = kc_get_u16(encoded + 4);
        uint32_t attempt = kc_get_u32(encoded + 8);
        uint32_t payload_length = kc_get_u32(encoded + 12);
        int32_t terminal_reason = kc_get_i32(encoded + 88);
        if (record_size != DURABLE_SNAPSHOT_MESSAGE_SIZE +
                           (uint64_t)payload_length ||
            record_size > length - offset || state < KC_MESSAGE_PENDING ||
            state > KC_MESSAGE_DEAD_LETTERED ||
            (state != KC_MESSAGE_DEAD_LETTERED && terminal_reason != 0)) {
            return -EBADMSG;
        }
        kc_id id = decode_id(encoded + 24);
        if (id.epoch != durable->epoch || !id.sequence ||
            id.sequence == UINT64_MAX ||
            message_find(durable, id)) return -EBADMSG;
        message_node *message = message_create(
            id, decode_id(encoded + 40), decode_id(encoded + 56),
            decode_id(encoded + 72), kc_get_u64(encoded + 16),
            encoded + DURABLE_SNAPSHOT_MESSAGE_SIZE, payload_length);
        if (!message) return -ENOMEM;
        message->attempt = attempt;
        message->terminal_reason = terminal_reason;
        message_append(durable, message);
        dedupe_add(durable, message->idempotency_key, message->id);
        if (state == KC_MESSAGE_IN_FLIGHT) {
            durable->pending--;
            durable->in_flight++;
        } else if (state == KC_MESSAGE_ACKNOWLEDGED) {
            durable->pending--;
            durable->acknowledged++;
        } else if (state == KC_MESSAGE_DEAD_LETTERED) {
            durable->pending--;
            durable->dead_lettered++;
        }
        message->state = (kc_message_state)state;
        durable->delivery_attempts += attempt;
        offset += record_size;
    }
    if (offset != length || !next_sequence ||
        next_sequence < durable->next_message_sequence) return -EBADMSG;
    durable->next_message_sequence = next_sequence;
    return 0;
}

static int snapshot_encode(kc_durable_t *durable, void **payload, size_t *length)
{
    size_t total = DURABLE_SNAPSHOT_HEADER_SIZE;
    uint32_t count = 0;
    for (message_node *message = durable->head; message; message = message->next) {
        if (count == UINT32_MAX ||
            message->payload_length > UINT32_MAX - DURABLE_SNAPSHOT_MESSAGE_SIZE ||
            message->payload_length > SIZE_MAX - DURABLE_SNAPSHOT_MESSAGE_SIZE ||
            total > SIZE_MAX - DURABLE_SNAPSHOT_MESSAGE_SIZE -
                    message->payload_length) return -E2BIG;
        total += DURABLE_SNAPSHOT_MESSAGE_SIZE + message->payload_length;
        count++;
    }
    unsigned char *data = calloc(1, total);
    if (!data) return -ENOMEM;
    kc_put_u32(data, DURABLE_SNAPSHOT_MAGIC);
    kc_put_u16(data + 4, DURABLE_FORMAT_VERSION);
    kc_put_u16(data + 6, DURABLE_SNAPSHOT_HEADER_SIZE);
    kc_put_u64(data + 8, durable->epoch);
    kc_put_u64(data + 16, durable->next_message_sequence);
    kc_put_u32(data + 24, count);
    size_t offset = DURABLE_SNAPSHOT_HEADER_SIZE;
    for (message_node *message = durable->head; message; message = message->next) {
        unsigned char *encoded = data + offset;
        size_t record_size = DURABLE_SNAPSHOT_MESSAGE_SIZE +
                             message->payload_length;
        kc_put_u32(encoded, (uint32_t)record_size);
        kc_put_u16(encoded + 4, (uint16_t)message->state);
        kc_put_u32(encoded + 8, message->attempt);
        kc_put_u32(encoded + 12, (uint32_t)message->payload_length);
        kc_put_u64(encoded + 16, message->route);
        encode_id(encoded + 24, message->id);
        encode_id(encoded + 40, message->correlation_id);
        encode_id(encoded + 56, message->trace_id);
        encode_id(encoded + 72, message->idempotency_key);
        kc_put_i32(encoded + 88, message->terminal_reason);
        if (message->payload_length) {
            memcpy(encoded + DURABLE_SNAPSHOT_MESSAGE_SIZE, message->payload,
                   message->payload_length);
        }
        offset += record_size;
    }
    *payload = data;
    *length = total;
    return 0;
}

static void message_view(const message_node *message, kc_message *out)
{
    *out = (kc_message){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .id = message->id, .correlation_id = message->correlation_id,
        .trace_id = message->trace_id,
        .idempotency_key = message->idempotency_key, .route = message->route,
        .delivery_attempt = message->attempt, .state = message->state,
        .terminal_reason = message->terminal_reason,
        .payload = message->payload, .payload_length = message->payload_length,
    };
}

int kc_durable_create(const kc_durable_config *config, kc_durable_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->wal) return -EINVAL;
    kc_wal_snapshot wal_snapshot = {
        .size = sizeof(wal_snapshot), .abi_version = KC_ABI_VERSION,
    };
    int rc = kc_wal_snapshot_get(config->wal, &wal_snapshot);
    if (rc != 0) return rc;
    kc_durable_t *durable = calloc(1, sizeof(*durable));
    if (!durable) return -ENOMEM;
    durable->wal = config->wal;
    durable->epoch = wal_snapshot.runtime_epoch;
    durable->next_message_sequence = 1;
    durable->max_delivery_attempts = config->max_delivery_attempts
        ? config->max_delivery_attempts : 16;
    durable->dedupe_capacity = config->dedupe_capacity
        ? config->dedupe_capacity : 1024;
    durable->dedupe = calloc(durable->dedupe_capacity, sizeof(*durable->dedupe));
    if (!durable->dedupe) { free(durable); return -ENOMEM; }
    if (KC_MUTEX_INIT(&durable->mu) != 0) {
        free(durable->dedupe);
        free(durable);
        return -ENOMEM;
    }
    if (wal_snapshot.snapshot_valid) {
        size_t snapshot_length = 0;
        uint64_t snapshot_sequence = 0;
        rc = kc_wal_snapshot_load(durable->wal, NULL, 0, &snapshot_length,
                                  &snapshot_sequence);
        if (rc != -ENOSPC) {
            kc_durable_destroy(durable);
            return rc == 0 ? -EBADMSG : rc;
        }
        void *snapshot = malloc(snapshot_length);
        if (!snapshot) { kc_durable_destroy(durable); return -ENOMEM; }
        rc = kc_wal_snapshot_load(durable->wal, snapshot, snapshot_length,
                                  &snapshot_length, &snapshot_sequence);
        if (rc == 0 && snapshot_length >= sizeof(uint32_t) &&
            kc_get_u32(snapshot) == DURABLE_SNAPSHOT_MAGIC) {
            rc = snapshot_decode(durable, snapshot, snapshot_length);
        } else if (rc == 0) {
            const void *section = NULL;
            size_t section_length = 0;
            rc = kc_checkpoint_find(snapshot, snapshot_length,
                                    KC_CHECKPOINT_DURABLE,
                                    &section, &section_length);
            if (rc == 0) rc = snapshot_decode(durable, section, section_length);
        }
        free(snapshot);
        if (rc != 0) { kc_durable_destroy(durable); return rc; }
    }
    rc = kc_wal_recover(durable->wal, recover_record, durable);
    if (rc != 0) {
        kc_durable_destroy(durable);
        return rc;
    }
    for (message_node *message = durable->head; message; message = message->next) {
        if (message->state != KC_MESSAGE_IN_FLIGHT) continue;
        message->state = KC_MESSAGE_PENDING;
        durable->in_flight--;
        durable->pending++;
        durable->redeliveries++;
    }
    *out = durable;
    return 0;
}

void kc_durable_destroy(kc_durable_t *durable)
{
    if (!durable) return;
    message_node *message = durable->head;
    while (message) {
        message_node *next = message->next;
        message_destroy(message);
        message = next;
    }
    KC_MUTEX_DESTROY(&durable->mu);
    free(durable->dedupe);
    free(durable);
}

int kc_durable_publish(kc_durable_t *durable, const kc_publish *publish,
                       kc_id *message_id)
{
    if (!durable || !publish || !message_id || publish->size < sizeof(*publish) ||
        publish->abi_version != KC_ABI_VERSION ||
        (publish->payload_length && !publish->payload)) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    if (dedupe_find(durable, publish->idempotency_key, message_id)) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return 0;
    }
    if (durable->next_message_sequence == UINT64_MAX) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return -EOVERFLOW;
    }
    kc_id id = { durable->epoch, durable->next_message_sequence };
    kc_id trace = id_empty(publish->trace_id) ? id : publish->trace_id;
    message_node *message = message_create(
        id, publish->correlation_id, trace, publish->idempotency_key,
        publish->route, publish->payload, publish->payload_length);
    if (!message) { KC_MUTEX_UNLOCK(&durable->mu); return -ENOMEM; }
    unsigned char *encoded = NULL;
    size_t encoded_length = 0;
    int rc = encode_publish(message, &encoded, &encoded_length);
    if (rc == 0) rc = persist(durable, DURABLE_PUBLISH_RECORD,
                              encoded, encoded_length);
    free(encoded);
    if (rc == 0) {
        message_append(durable, message);
        dedupe_add(durable, message->idempotency_key, message->id);
        *message_id = message->id;
    } else message_destroy(message);
    KC_MUTEX_UNLOCK(&durable->mu);
    return rc;
}

static int transition(kc_durable_t *durable, message_node *message,
                      uint16_t type, uint32_t value)
{
    unsigned char encoded[EVENT_SIZE];
    encode_event(encoded, message, value);
    return persist(durable, type, encoded, sizeof(encoded));
}

int kc_durable_next(kc_durable_t *durable, uint64_t route, kc_message *out)
{
    if (!durable || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    for (message_node *message = durable->head; message; message = message->next) {
        if (message->route != route || message->state != KC_MESSAGE_PENDING) continue;
        if (message->attempt >= durable->max_delivery_attempts) {
            int rc = transition(durable, message, DURABLE_DEAD_RECORD,
                                (uint32_t)(-ELOOP));
            if (rc != 0) { KC_MUTEX_UNLOCK(&durable->mu); return rc; }
            durable->pending--;
            durable->dead_lettered++;
            message->state = KC_MESSAGE_DEAD_LETTERED;
            message->terminal_reason = -ELOOP;
            continue;
        }
        uint32_t attempt = message->attempt + 1;
        int rc = transition(durable, message, DURABLE_ATTEMPT_RECORD, attempt);
        if (rc != 0) { KC_MUTEX_UNLOCK(&durable->mu); return rc; }
        durable->pending--;
        durable->in_flight++;
        durable->delivery_attempts++;
        message->attempt = attempt;
        message->state = KC_MESSAGE_IN_FLIGHT;
        message_view(message, out);
        KC_MUTEX_UNLOCK(&durable->mu);
        return 0;
    }
    KC_MUTEX_UNLOCK(&durable->mu);
    return -ENOENT;
}

int kc_durable_acknowledge(kc_durable_t *durable, kc_id message_id)
{
    if (!durable) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    message_node *message = message_find(durable, message_id);
    if (!message) { KC_MUTEX_UNLOCK(&durable->mu); return -ENOENT; }
    if (message->state == KC_MESSAGE_ACKNOWLEDGED) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return 0;
    }
    if (message->state == KC_MESSAGE_DEAD_LETTERED) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return -EALREADY;
    }
    int rc = transition(durable, message, DURABLE_ACK_RECORD, 0);
    if (rc == 0) {
        if (message->state == KC_MESSAGE_PENDING) durable->pending--;
        else durable->in_flight--;
        durable->acknowledged++;
        message->state = KC_MESSAGE_ACKNOWLEDGED;
        message->terminal_reason = 0;
    }
    KC_MUTEX_UNLOCK(&durable->mu);
    return rc;
}

int kc_durable_retry(kc_durable_t *durable, kc_id message_id)
{
    if (!durable) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    message_node *message = message_find(durable, message_id);
    if (!message) { KC_MUTEX_UNLOCK(&durable->mu); return -ENOENT; }
    if (message->state == KC_MESSAGE_ACKNOWLEDGED ||
        message->state == KC_MESSAGE_DEAD_LETTERED) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return -EALREADY;
    }
    if (message->state == KC_MESSAGE_PENDING) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return 0;
    }
    int rc = transition(durable, message, DURABLE_RETRY_RECORD, 0);
    if (rc == 0) {
        durable->in_flight--;
        durable->pending++;
        message->state = KC_MESSAGE_PENDING;
        message->terminal_reason = 0;
    }
    KC_MUTEX_UNLOCK(&durable->mu);
    return rc;
}

int kc_durable_dead_letter(kc_durable_t *durable, kc_id message_id, int reason)
{
    if (!durable) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    message_node *message = message_find(durable, message_id);
    if (!message) { KC_MUTEX_UNLOCK(&durable->mu); return -ENOENT; }
    if (message->state == KC_MESSAGE_DEAD_LETTERED) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return 0;
    }
    if (message->state == KC_MESSAGE_ACKNOWLEDGED) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return -EALREADY;
    }
    int rc = transition(durable, message, DURABLE_DEAD_RECORD,
                        (uint32_t)reason);
    if (rc == 0) {
        if (message->state == KC_MESSAGE_PENDING) durable->pending--;
        else durable->in_flight--;
        durable->dead_lettered++;
        message->state = KC_MESSAGE_DEAD_LETTERED;
        message->terminal_reason = reason;
    }
    KC_MUTEX_UNLOCK(&durable->mu);
    return rc;
}

int kc_durable_lookup(kc_durable_t *durable, kc_id message_id, kc_message *out)
{
    if (!durable || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    message_node *message = message_find(durable, message_id);
    if (!message) { KC_MUTEX_UNLOCK(&durable->mu); return -ENOENT; }
    message_view(message, out);
    KC_MUTEX_UNLOCK(&durable->mu);
    return 0;
}

int kc_durable_checkpoint(kc_durable_t *durable)
{
    if (!durable) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    if (durable->workflow_users) {
        KC_MUTEX_UNLOCK(&durable->mu);
        return -EBUSY;
    }
    void *payload = NULL;
    size_t length = 0;
    int rc = snapshot_encode(durable, &payload, &length);
    void *checkpoint = NULL;
    size_t checkpoint_length = 0;
    if (rc == 0) {
        kc_checkpoint_section section = {
            .type = KC_CHECKPOINT_DURABLE,
            .payload = payload,
            .payload_length = length,
        };
        rc = kc_checkpoint_encode(&section, 1, &checkpoint,
                                  &checkpoint_length);
    }
    if (rc == 0) {
        rc = kc_wal_snapshot_write(durable->wal, checkpoint,
                                   checkpoint_length);
    }
    free(checkpoint);
    free(payload);
    KC_MUTEX_UNLOCK(&durable->mu);
    return rc;
}

void kc_durable_lock_internal(kc_durable_t *durable)
{
    if (durable) KC_MUTEX_LOCK(&durable->mu);
}

void kc_durable_unlock_internal(kc_durable_t *durable)
{
    if (durable) KC_MUTEX_UNLOCK(&durable->mu);
}

int kc_durable_snapshot_encode_internal(kc_durable_t *durable, void **payload,
                                        size_t *length)
{
    if (!durable || !payload || !length) return -EINVAL;
    return snapshot_encode(durable, payload, length);
}

kc_wal_t *kc_durable_wal_internal(kc_durable_t *durable)
{
    return durable ? durable->wal : NULL;
}

void kc_durable_workflow_attach(kc_durable_t *durable)
{
    if (!durable) return;
    KC_MUTEX_LOCK(&durable->mu);
    durable->workflow_users++;
    KC_MUTEX_UNLOCK(&durable->mu);
}

void kc_durable_workflow_detach(kc_durable_t *durable)
{
    if (!durable) return;
    KC_MUTEX_LOCK(&durable->mu);
    if (durable->workflow_users) durable->workflow_users--;
    KC_MUTEX_UNLOCK(&durable->mu);
}

int kc_durable_snapshot_get(kc_durable_t *durable, kc_durable_snapshot *out)
{
    if (!durable || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&durable->mu);
    *out = (kc_durable_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .runtime_epoch = durable->epoch,
        .next_message_sequence = durable->next_message_sequence,
        .pending = durable->pending, .in_flight = durable->in_flight,
        .acknowledged = durable->acknowledged,
        .dead_lettered = durable->dead_lettered,
        .dedupe_entries = durable->dedupe_count,
        .logical_bytes = durable->logical_bytes,
        .publications = durable->publications,
        .delivery_attempts = durable->delivery_attempts,
        .redeliveries = durable->redeliveries,
    };
    KC_MUTEX_UNLOCK(&durable->mu);
    return 0;
}

static int batch_push(message_node ***items, size_t *count, size_t *capacity,
                      message_node *message)
{
    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 4;
        if (next < *capacity || next > SIZE_MAX / sizeof(**items)) return -E2BIG;
        message_node **grown = realloc(*items, next * sizeof(**items));
        if (!grown) return -ENOMEM;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = message;
    return 0;
}

int kc_durable_batch_begin(kc_durable_t *durable, kc_durable_batch **out)
{
    if (!durable || !out) return -EINVAL;
    kc_durable_batch *batch = calloc(1, sizeof(*batch));
    if (!batch) return -ENOMEM;
    KC_MUTEX_LOCK(&durable->mu);
    int rc = kc_wal_transaction_begin(durable->wal, &batch->transaction);
    if (rc != 0) {
        KC_MUTEX_UNLOCK(&durable->mu);
        free(batch);
        return rc;
    }
    batch->durable = durable;
    *out = batch;
    return 0;
}

int kc_durable_batch_ack(kc_durable_batch *batch, kc_id message_id)
{
    if (!batch) return -EINVAL;
    message_node *message = message_find(batch->durable, message_id);
    if (!message) return -ENOENT;
    if (message->state == KC_MESSAGE_ACKNOWLEDGED ||
        message->state == KC_MESSAGE_DEAD_LETTERED) return -EALREADY;
    for (size_t index = 0; index < batch->acknowledgement_count; index++) {
        if (batch->acknowledgements[index] == message) return -EALREADY;
    }
    unsigned char encoded[EVENT_SIZE];
    encode_event(encoded, message, 0);
    int rc = kc_wal_transaction_append(batch->transaction, DURABLE_ACK_RECORD,
                                       encoded, sizeof(encoded));
    if (rc != 0) return rc;
    return batch_push(&batch->acknowledgements,
                      &batch->acknowledgement_count,
                      &batch->acknowledgement_capacity, message);
}

int kc_durable_batch_record(kc_durable_batch *batch, uint16_t type,
                            const void *payload, size_t length)
{
    if (!batch) return -EINVAL;
    return kc_wal_transaction_append(batch->transaction, type, payload, length);
}

int kc_durable_batch_publish(kc_durable_batch *batch, const kc_publish *publish,
                             kc_id *message_id)
{
    if (!batch || !publish || !message_id || publish->size < sizeof(*publish) ||
        publish->abi_version != KC_ABI_VERSION ||
        (publish->payload_length && !publish->payload)) return -EINVAL;
    kc_durable_t *durable = batch->durable;
    if (dedupe_find(durable, publish->idempotency_key, message_id)) return 0;
    for (size_t index = 0; index < batch->message_count; index++) {
        if (id_equal(batch->messages[index]->idempotency_key,
                     publish->idempotency_key) &&
            !id_empty(publish->idempotency_key)) {
            *message_id = batch->messages[index]->id;
            return 0;
        }
    }
    if (durable->next_message_sequence == UINT64_MAX ||
        batch->message_count > UINT64_MAX - UINT64_C(1) -
                               durable->next_message_sequence) {
        return -EOVERFLOW;
    }
    kc_id id = {
        durable->epoch,
        durable->next_message_sequence + batch->message_count,
    };
    kc_id trace = id_empty(publish->trace_id) ? id : publish->trace_id;
    message_node *message = message_create(
        id, publish->correlation_id, trace, publish->idempotency_key,
        publish->route, publish->payload, publish->payload_length);
    if (!message) return -ENOMEM;
    unsigned char *encoded = NULL;
    size_t encoded_length = 0;
    int rc = encode_publish(message, &encoded, &encoded_length);
    if (rc == 0) {
        rc = kc_wal_transaction_append(batch->transaction,
                                       DURABLE_PUBLISH_RECORD,
                                       encoded, encoded_length);
    }
    free(encoded);
    if (rc == 0) {
        rc = batch_push(&batch->messages, &batch->message_count,
                        &batch->message_capacity, message);
    }
    if (rc != 0) { message_destroy(message); return rc; }
    *message_id = id;
    return 0;
}

static void batch_free(kc_durable_batch *batch, int committed)
{
    if (!batch) return;
    if (!committed) {
        for (size_t index = 0; index < batch->message_count; index++) {
            message_destroy(batch->messages[index]);
        }
    }
    free(batch->messages);
    free(batch->acknowledgements);
    KC_MUTEX_UNLOCK(&batch->durable->mu);
    free(batch);
}

int kc_durable_batch_commit(kc_durable_batch *batch)
{
    if (!batch) return -EINVAL;
    int rc = kc_wal_transaction_commit(batch->transaction);
    batch->transaction = NULL;
    if (rc != 0) { batch_free(batch, 0); return rc; }
    kc_durable_t *durable = batch->durable;
    for (size_t index = 0; index < batch->acknowledgement_count; index++) {
        message_node *message = batch->acknowledgements[index];
        if (message->state == KC_MESSAGE_PENDING) durable->pending--;
        else durable->in_flight--;
        durable->acknowledged++;
        message->state = KC_MESSAGE_ACKNOWLEDGED;
        message->terminal_reason = 0;
    }
    for (size_t index = 0; index < batch->message_count; index++) {
        message_node *message = batch->messages[index];
        message_append(durable, message);
        dedupe_add(durable, message->idempotency_key, message->id);
    }
    batch_free(batch, 1);
    return 0;
}

void kc_durable_batch_abort(kc_durable_batch *batch)
{
    if (!batch) return;
    if (batch->transaction) kc_wal_transaction_abort(batch->transaction);
    batch->transaction = NULL;
    batch_free(batch, 0);
}
