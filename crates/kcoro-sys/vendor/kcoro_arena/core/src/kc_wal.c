// SPDX-License-Identifier: BSD-3-Clause
#include "kc_wal.h"
#include "kc_codec_internal.h"
#include "kcoro_port.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum {
    WAL_HEADER_SIZE = 48,
    SNAPSHOT_HEADER_SIZE = 40,
    WAL_RECORD_BEGIN = 1,
    WAL_RECORD_COMMIT = 2,
};

#define WAL_MAGIC UINT32_C(0x4c57434b)
#define SNAPSHOT_MAGIC UINT32_C(0x4e53434b)
#define DEFAULT_MAX_RECORD (16u * 1024u * 1024u)
#define DEFAULT_MAX_TRANSACTION (64u * 1024u * 1024u)

typedef struct pending_record {
    uint16_t type;
    uint32_t length;
    uint64_t sequence;
    void *payload;
} pending_record;

typedef struct scan_result {
    uint64_t bytes;
    uint64_t valid_end;
    uint64_t highest_sequence;
    uint64_t committed_sequence;
} scan_result;

struct kc_wal {
    KC_MUTEX_T mu;
    kc_store_handle *store;
    kc_store_handle *snapshot_store;
    uint64_t epoch;
    uint64_t next_sequence;
    uint64_t committed_sequence;
    uint64_t snapshot_sequence;
    uint64_t wal_bytes;
    size_t max_record_size;
    size_t max_transaction_bytes;
    int poisoned;
    int snapshot_valid;
    int active_transaction;
};

struct kc_wal_tx {
    kc_wal_t *wal;
    uint64_t id;
    uint64_t last_sequence;
    size_t bytes;
    int active;
};

static uint32_t crc32c_update(uint32_t crc, const void *data, size_t length)
{
    const unsigned char *bytes = data;
    for (size_t index = 0; index < length; index++) {
        crc ^= bytes[index];
        for (unsigned bit = 0; bit < 8; bit++) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (UINT32_C(0x82f63b78) & mask);
        }
    }
    return crc;
}

static uint32_t record_crc(const unsigned char *header, size_t header_length,
                           const void *payload, size_t payload_length,
                           size_t crc_offset)
{
    unsigned char copy[WAL_HEADER_SIZE];
    memcpy(copy, header, header_length);
    kc_put_u32(copy + crc_offset, 0);
    uint32_t crc = crc32c_update(UINT32_MAX, copy, header_length);
    if (payload_length) crc = crc32c_update(crc, payload, payload_length);
    return ~crc;
}

static void pending_clear(pending_record *records, size_t count)
{
    for (size_t index = 0; index < count; index++) free(records[index].payload);
    free(records);
}

static int pending_push(pending_record **records, size_t *count, size_t *capacity,
                        uint16_t type, uint64_t sequence, void *payload,
                        uint32_t length)
{
    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 8;
        if (next < *capacity || next > SIZE_MAX / sizeof(**records)) {
            return -EOVERFLOW;
        }
        pending_record *grown = realloc(*records, next * sizeof(**records));
        if (!grown) return -ENOMEM;
        *records = grown;
        *capacity = next;
    }
    (*records)[*count] = (pending_record){
        .type = type, .length = length, .sequence = sequence,
        .payload = payload,
    };
    (*count)++;
    return 0;
}

static int wal_scan(kc_wal_t *wal, kc_wal_visit_fn visit, void *context,
                    scan_result *result)
{
    uint64_t size = 0;
    int rc = kc_store_size(wal->store, &size);
    if (rc != 0) return rc;
    uint64_t offset = 0;
    uint64_t previous_sequence = 0;
    uint64_t transaction = 0;
    uint64_t transaction_start = 0;
    size_t transaction_bytes = 0;
    pending_record *records = NULL;
    size_t count = 0;
    size_t capacity = 0;
    *result = (scan_result){ .bytes = size, .valid_end = 0,
                            .committed_sequence = wal->snapshot_sequence };

    while (offset < size) {
        uint64_t record_start = offset;
        if (size - offset < WAL_HEADER_SIZE) break;
        unsigned char header[WAL_HEADER_SIZE];
        rc = kc_store_read(wal->store, offset, header, sizeof(header));
        if (rc != 0) goto done;
        uint32_t magic = kc_get_u32(header);
        uint16_t version = kc_get_u16(header + 4);
        uint16_t type = kc_get_u16(header + 6);
        uint32_t length = kc_get_u32(header + 8);
        uint32_t header_length = kc_get_u32(header + 12);
        uint64_t epoch = kc_get_u64(header + 16);
        uint64_t sequence = kc_get_u64(header + 24);
        uint64_t record_transaction = kc_get_u64(header + 32);
        uint32_t expected_crc = kc_get_u32(header + 40);
        if (magic != WAL_MAGIC || version != KC_WAL_FORMAT_VERSION ||
            header_length != WAL_HEADER_SIZE || epoch != wal->epoch ||
            length > wal->max_record_size || !sequence ||
            sequence == UINT64_MAX ||
            sequence <= previous_sequence) {
            rc = -EBADMSG;
            goto done;
        }
        if ((type == WAL_RECORD_BEGIN || type == WAL_RECORD_COMMIT) && length) {
            rc = -EBADMSG;
            goto done;
        }
        if (type != WAL_RECORD_BEGIN && type != WAL_RECORD_COMMIT &&
            type < KC_WAL_RECORD_USER_BASE) {
            rc = -EBADMSG;
            goto done;
        }
        if ((uint64_t)length > size - offset - WAL_HEADER_SIZE) break;
        void *payload = NULL;
        if (length) {
            payload = malloc(length);
            if (!payload) { rc = -ENOMEM; goto done; }
            rc = kc_store_read(wal->store, offset + WAL_HEADER_SIZE,
                               payload, length);
            if (rc != 0) { free(payload); goto done; }
        }
        uint32_t actual_crc = record_crc(header, sizeof(header), payload,
                                         length, 40);
        if (actual_crc != expected_crc) {
            free(payload);
            rc = -EBADMSG;
            goto done;
        }
        offset += WAL_HEADER_SIZE + length;
        previous_sequence = sequence;
        result->highest_sequence = sequence;

        if (type == WAL_RECORD_BEGIN) {
            if (record_transaction != sequence) {
                free(payload);
                rc = -EBADMSG;
                goto done;
            }
            free(payload);
            pending_clear(records, count);
            records = NULL;
            count = 0;
            capacity = 0;
            transaction_bytes = 0;
            transaction = record_transaction;
            transaction_start = record_start;
        } else if (type == WAL_RECORD_COMMIT) {
            free(payload);
            if (!transaction || record_transaction != transaction) {
                rc = -EBADMSG;
                goto done;
            }
            if (sequence > wal->snapshot_sequence && visit) {
                for (size_t index = 0; index < count; index++) {
                    kc_wal_record record = {
                        .size = sizeof(record), .abi_version = KC_ABI_VERSION,
                        .type = records[index].type,
                        .payload_length = records[index].length,
                        .runtime_epoch = wal->epoch,
                        .sequence = records[index].sequence,
                        .transaction = transaction,
                        .payload = records[index].payload,
                    };
                    rc = visit(context, &record);
                    if (rc != 0) goto done;
                }
            }
            if (sequence > result->committed_sequence) {
                result->committed_sequence = sequence;
            }
            pending_clear(records, count);
            records = NULL;
            count = 0;
            capacity = 0;
            transaction_bytes = 0;
            transaction = 0;
            transaction_start = 0;
        } else {
            size_t record_bytes = WAL_HEADER_SIZE + (size_t)length;
            if (type < KC_WAL_RECORD_USER_BASE || !transaction ||
                record_transaction != transaction ||
                record_bytes > wal->max_transaction_bytes ||
                transaction_bytes > wal->max_transaction_bytes - record_bytes) {
                free(payload);
                rc = type < KC_WAL_RECORD_USER_BASE ? -EBADMSG : -E2BIG;
                goto done;
            }
            rc = pending_push(&records, &count, &capacity, type, sequence,
                              payload, length);
            if (rc != 0) { free(payload); goto done; }
            transaction_bytes += record_bytes;
        }
        result->valid_end = offset;
    }

    if (transaction) result->valid_end = transaction_start;
    rc = 0;

done:
    pending_clear(records, count);
    return rc;
}

static int snapshot_read(kc_wal_t *wal, void **payload, size_t *length,
                         uint64_t *sequence, uint64_t *valid_end,
                         uint64_t *stored_bytes)
{
    if (valid_end) *valid_end = 0;
    if (stored_bytes) *stored_bytes = 0;
    if (!wal->snapshot_store) return -ENODATA;
    uint64_t size = 0;
    int rc = kc_store_size(wal->snapshot_store, &size);
    if (rc != 0) return rc;
    if (stored_bytes) *stored_bytes = size;
    if (!size) return -ENODATA;
    uint64_t offset = 0;
    uint64_t latest_sequence = 0;
    void *latest = NULL;
    size_t latest_length = 0;
    int found = 0;
    while (offset < size) {
        if (size - offset < SNAPSHOT_HEADER_SIZE) break;
        unsigned char header[SNAPSHOT_HEADER_SIZE];
        rc = kc_store_read(wal->snapshot_store, offset, header,
                           sizeof(header));
        if (rc != 0) goto fail;
        uint32_t data_length = kc_get_u32(header + 8);
        uint64_t frame_sequence = kc_get_u64(header + 24);
        if (kc_get_u32(header) != SNAPSHOT_MAGIC ||
            kc_get_u16(header + 4) != KC_WAL_FORMAT_VERSION ||
            kc_get_u16(header + 6) != SNAPSHOT_HEADER_SIZE ||
            kc_get_u64(header + 16) != wal->epoch ||
            data_length > wal->max_transaction_bytes ||
            frame_sequence == UINT64_MAX ||
            (found && frame_sequence < latest_sequence)) {
            rc = -EBADMSG;
            goto fail;
        }
        if ((uint64_t)data_length > size - offset - SNAPSHOT_HEADER_SIZE) break;
        void *data = NULL;
        if (data_length) {
            data = malloc(data_length);
            if (!data) { rc = -ENOMEM; goto fail; }
            rc = kc_store_read(wal->snapshot_store,
                               offset + SNAPSHOT_HEADER_SIZE,
                               data, data_length);
            if (rc != 0) { free(data); goto fail; }
        }
        uint32_t expected_crc = kc_get_u32(header + 32);
        uint32_t actual_crc = record_crc(header, sizeof(header), data,
                                         data_length, 32);
        if (expected_crc != actual_crc) {
            free(data);
            rc = -EBADMSG;
            goto fail;
        }
        free(latest);
        latest = data;
        latest_length = data_length;
        latest_sequence = frame_sequence;
        found = 1;
        offset += SNAPSHOT_HEADER_SIZE + data_length;
    }
    if (valid_end) *valid_end = offset;
    if (!found) { free(latest); return -EBADMSG; }
    *payload = latest;
    *length = latest_length;
    *sequence = latest_sequence;
    return 0;

fail:
    free(latest);
    return rc;
}

static int append_record(kc_wal_t *wal, uint16_t type, uint64_t transaction,
                         const void *payload, size_t length,
                         uint64_t *sequence_out)
{
    if (length > wal->max_record_size || (length && !payload)) return -E2BIG;
    if (!wal->next_sequence || wal->next_sequence == UINT64_MAX) {
        return -EOVERFLOW;
    }
    unsigned char header[WAL_HEADER_SIZE] = {0};
    uint64_t sequence = wal->next_sequence;
    kc_put_u32(header, WAL_MAGIC);
    kc_put_u16(header + 4, KC_WAL_FORMAT_VERSION);
    kc_put_u16(header + 6, type);
    kc_put_u32(header + 8, (uint32_t)length);
    kc_put_u32(header + 12, WAL_HEADER_SIZE);
    kc_put_u64(header + 16, wal->epoch);
    kc_put_u64(header + 24, sequence);
    kc_put_u64(header + 32, transaction);
    kc_put_u32(header + 40, record_crc(header, sizeof(header), payload, length, 40));
    uint64_t offset = 0;
    int rc = kc_store_append(wal->store, header, sizeof(header), &offset);
    if (rc != 0) { wal->poisoned = 1; return rc; }
    wal->wal_bytes = offset + sizeof(header);
    if (length) {
        rc = kc_store_append(wal->store, payload, length, NULL);
        if (rc != 0) { wal->poisoned = 1; return rc; }
        wal->wal_bytes += length;
    }
    wal->next_sequence++;
    if (sequence_out) *sequence_out = sequence;
    return 0;
}

int kc_wal_create(const kc_wal_config *config, kc_wal_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->store ||
        !config->runtime_epoch || config->snapshot_store == config->store) {
        return -EINVAL;
    }
    kc_wal_t *wal = calloc(1, sizeof(*wal));
    if (!wal) return -ENOMEM;
    wal->store = config->store;
    wal->snapshot_store = config->snapshot_store;
    wal->epoch = config->runtime_epoch;
    wal->max_record_size = config->max_record_size
        ? config->max_record_size : DEFAULT_MAX_RECORD;
    wal->max_transaction_bytes = config->max_transaction_bytes
        ? config->max_transaction_bytes : DEFAULT_MAX_TRANSACTION;
    wal->next_sequence = 1;
    if (wal->max_record_size > UINT32_MAX ||
        wal->max_record_size > SIZE_MAX - WAL_HEADER_SIZE ||
        wal->max_transaction_bytes <
            wal->max_record_size + WAL_HEADER_SIZE) {
        free(wal);
        return -EINVAL;
    }
    if (KC_MUTEX_INIT(&wal->mu) != 0) { free(wal); return -ENOMEM; }

    void *snapshot = NULL;
    size_t snapshot_length = 0;
    uint64_t snapshot_sequence = 0;
    uint64_t snapshot_valid_end = 0;
    uint64_t snapshot_bytes = 0;
    int rc = snapshot_read(wal, &snapshot, &snapshot_length,
                           &snapshot_sequence, &snapshot_valid_end,
                           &snapshot_bytes);
    free(snapshot);
    if (rc == 0) {
        wal->snapshot_valid = 1;
        wal->snapshot_sequence = snapshot_sequence;
        wal->committed_sequence = snapshot_sequence;
        wal->next_sequence = snapshot_sequence + 1;
    } else if (rc != -ENODATA) {
        KC_MUTEX_DESTROY(&wal->mu);
        free(wal);
        return rc;
    }
    if (snapshot_valid_end < snapshot_bytes) {
        rc = kc_store_truncate(wal->snapshot_store, snapshot_valid_end);
        if (rc == 0) rc = kc_store_sync(wal->snapshot_store);
        if (rc != 0) {
            KC_MUTEX_DESTROY(&wal->mu);
            free(wal);
            return rc;
        }
    }

    scan_result scan;
    rc = wal_scan(wal, NULL, NULL, &scan);
    if (rc != 0) {
        KC_MUTEX_DESTROY(&wal->mu);
        free(wal);
        return rc;
    }
    if (scan.valid_end < scan.bytes) {
        rc = kc_store_truncate(wal->store, scan.valid_end);
        if (rc == 0) rc = kc_store_sync(wal->store);
        if (rc != 0) {
            KC_MUTEX_DESTROY(&wal->mu);
            free(wal);
            return rc;
        }
    }
    wal->wal_bytes = scan.valid_end;
    if (scan.highest_sequence >= wal->next_sequence) {
        wal->next_sequence = scan.highest_sequence + 1;
    }
    if (scan.committed_sequence > wal->committed_sequence) {
        wal->committed_sequence = scan.committed_sequence;
    }
    *out = wal;
    return 0;
}

void kc_wal_destroy(kc_wal_t *wal)
{
    if (!wal) return;
    KC_MUTEX_DESTROY(&wal->mu);
    free(wal);
}

int kc_wal_recover(kc_wal_t *wal, kc_wal_visit_fn visit, void *context)
{
    if (!wal || !visit) return -EINVAL;
    KC_MUTEX_LOCK(&wal->mu);
    if (wal->active_transaction) {
        KC_MUTEX_UNLOCK(&wal->mu);
        return -EBUSY;
    }
    scan_result scan;
    int rc = wal_scan(wal, visit, context, &scan);
    KC_MUTEX_UNLOCK(&wal->mu);
    return rc;
}

int kc_wal_transaction_begin(kc_wal_t *wal, kc_wal_tx_t **out)
{
    if (!wal || !out) return -EINVAL;
    kc_wal_tx_t *transaction = calloc(1, sizeof(*transaction));
    if (!transaction) return -ENOMEM;
    KC_MUTEX_LOCK(&wal->mu);
    if (wal->poisoned || wal->active_transaction) {
        KC_MUTEX_UNLOCK(&wal->mu);
        free(transaction);
        return wal->poisoned ? -EIO : -EBUSY;
    }
    wal->active_transaction = 1;
    transaction->wal = wal;
    transaction->id = wal->next_sequence;
    transaction->active = 1;
    int rc = append_record(wal, WAL_RECORD_BEGIN, transaction->id,
                           NULL, 0, &transaction->last_sequence);
    if (rc != 0) {
        wal->active_transaction = 0;
        KC_MUTEX_UNLOCK(&wal->mu);
        free(transaction);
        return rc;
    }
    *out = transaction;
    return 0;
}

int kc_wal_transaction_append(kc_wal_tx_t *transaction, uint16_t type,
                              const void *payload, size_t length)
{
    if (!transaction || !transaction->active || type < KC_WAL_RECORD_USER_BASE ||
        (length && !payload)) return -EINVAL;
    kc_wal_t *wal = transaction->wal;
    if (length > wal->max_record_size) return -E2BIG;
    size_t record_bytes = WAL_HEADER_SIZE + length;
    if (record_bytes > wal->max_transaction_bytes ||
        transaction->bytes > wal->max_transaction_bytes - record_bytes) {
        return -E2BIG;
    }
    int rc = append_record(wal, type, transaction->id, payload, length,
                           &transaction->last_sequence);
    if (rc == 0) transaction->bytes += record_bytes;
    return rc;
}

static void transaction_finish(kc_wal_tx_t *transaction)
{
    kc_wal_t *wal = transaction->wal;
    transaction->active = 0;
    wal->active_transaction = 0;
    KC_MUTEX_UNLOCK(&wal->mu);
    free(transaction);
}

int kc_wal_transaction_commit(kc_wal_tx_t *transaction)
{
    if (!transaction || !transaction->active) return -EINVAL;
    kc_wal_t *wal = transaction->wal;
    uint64_t sequence = 0;
    int rc = append_record(wal, WAL_RECORD_COMMIT, transaction->id,
                           NULL, 0, &sequence);
    if (rc == 0) rc = kc_store_sync(wal->store);
    if (rc == 0) wal->committed_sequence = sequence;
    else wal->poisoned = 1;
    transaction_finish(transaction);
    return rc;
}

void kc_wal_transaction_abort(kc_wal_tx_t *transaction)
{
    if (!transaction || !transaction->active) return;
    transaction_finish(transaction);
}

int kc_wal_snapshot_write(kc_wal_t *wal, const void *payload, size_t length)
{
    if (!wal || !wal->snapshot_store || (length && !payload)) return -EINVAL;
    if (length > wal->max_transaction_bytes || length > UINT32_MAX) return -E2BIG;
    KC_MUTEX_LOCK(&wal->mu);
    if (wal->poisoned || wal->active_transaction) {
        KC_MUTEX_UNLOCK(&wal->mu);
        return wal->poisoned ? -EIO : -EBUSY;
    }
    unsigned char header[SNAPSHOT_HEADER_SIZE] = {0};
    kc_put_u32(header, SNAPSHOT_MAGIC);
    kc_put_u16(header + 4, KC_WAL_FORMAT_VERSION);
    kc_put_u16(header + 6, SNAPSHOT_HEADER_SIZE);
    kc_put_u32(header + 8, (uint32_t)length);
    kc_put_u64(header + 16, wal->epoch);
    kc_put_u64(header + 24, wal->committed_sequence);
    kc_put_u32(header + 32, record_crc(header, sizeof(header), payload, length, 32));

    uint64_t start = 0;
    int rc = kc_store_size(wal->snapshot_store, &start);
    uint64_t offset = 0;
    int appended = 0;
    if (rc == 0) {
        rc = kc_store_append(wal->snapshot_store, header,
                             sizeof(header), &offset);
        if (rc == 0) appended = 1;
    }
    if (rc == 0 && offset != start) rc = -EIO;
    if (rc == 0 && length) {
        rc = kc_store_append(wal->snapshot_store, payload, length, NULL);
    }
    if (rc == 0) rc = kc_store_sync(wal->snapshot_store);
    if (rc != 0 && appended) {
        int rollback = kc_store_truncate(wal->snapshot_store, start);
        if (rollback != 0) wal->poisoned = 1;
    }
    if (rc == 0) {
        wal->snapshot_valid = 1;
        wal->snapshot_sequence = wal->committed_sequence;
        rc = kc_store_truncate(wal->store, 0);
        if (rc == 0) rc = kc_store_sync(wal->store);
        if (rc == 0) wal->wal_bytes = 0;
        else wal->poisoned = 1;
    }
    KC_MUTEX_UNLOCK(&wal->mu);
    return rc;
}

int kc_wal_snapshot_load(kc_wal_t *wal, void *payload, size_t capacity,
                         size_t *written, uint64_t *sequence)
{
    if (!wal || !written || !sequence || (capacity && !payload)) return -EINVAL;
    KC_MUTEX_LOCK(&wal->mu);
    void *data = NULL;
    size_t length = 0;
    uint64_t loaded_sequence = 0;
    int rc = snapshot_read(wal, &data, &length, &loaded_sequence, NULL, NULL);
    if (rc == 0 && capacity < length) rc = -ENOSPC;
    if (rc == 0 && length) memcpy(payload, data, length);
    *written = length;
    *sequence = loaded_sequence;
    free(data);
    KC_MUTEX_UNLOCK(&wal->mu);
    return rc;
}

int kc_wal_snapshot_get(kc_wal_t *wal, kc_wal_snapshot *out)
{
    if (!wal || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&wal->mu);
    *out = (kc_wal_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .runtime_epoch = wal->epoch, .next_sequence = wal->next_sequence,
        .committed_sequence = wal->committed_sequence,
        .snapshot_sequence = wal->snapshot_sequence,
        .wal_bytes = wal->wal_bytes, .poisoned = (unsigned)wal->poisoned,
        .snapshot_valid = (unsigned)wal->snapshot_valid,
    };
    KC_MUTEX_UNLOCK(&wal->mu);
    return 0;
}
