// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_WAL_H
#define KC_WAL_H

#include "kc_runtime.h"
#include "kc_store.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KC_WAL_FORMAT_VERSION 1u
#define KC_WAL_RECORD_USER_BASE 0x100u

typedef struct kc_wal kc_wal_t;
typedef struct kc_wal_tx kc_wal_tx_t;

typedef struct kc_wal_config {
    uint32_t size;
    uint32_t abi_version;
    kc_store_handle *store;
    kc_store_handle *snapshot_store;
    uint64_t runtime_epoch;
    size_t max_record_size;
    size_t max_transaction_bytes;
} kc_wal_config;

typedef struct kc_wal_record {
    uint32_t size;
    uint32_t abi_version;
    uint16_t type;
    uint16_t reserved;
    uint32_t payload_length;
    uint64_t runtime_epoch;
    uint64_t sequence;
    uint64_t transaction;
    const void *payload;
} kc_wal_record;

typedef int (*kc_wal_visit_fn)(void *context, const kc_wal_record *record);

typedef struct kc_wal_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t runtime_epoch;
    uint64_t next_sequence;
    uint64_t committed_sequence;
    uint64_t snapshot_sequence;
    uint64_t wal_bytes;
    unsigned poisoned;
    unsigned snapshot_valid;
} kc_wal_snapshot;

int kc_wal_create(const kc_wal_config *config, kc_wal_t **out);
void kc_wal_destroy(kc_wal_t *wal);
int kc_wal_recover(kc_wal_t *wal, kc_wal_visit_fn visit, void *context);

int kc_wal_transaction_begin(kc_wal_t *wal, kc_wal_tx_t **out);
int kc_wal_transaction_append(kc_wal_tx_t *transaction, uint16_t type,
                              const void *payload, size_t length);
int kc_wal_transaction_commit(kc_wal_tx_t *transaction);
void kc_wal_transaction_abort(kc_wal_tx_t *transaction);

int kc_wal_snapshot_write(kc_wal_t *wal, const void *payload, size_t length);
int kc_wal_snapshot_load(kc_wal_t *wal, void *payload, size_t capacity,
                         size_t *written, uint64_t *sequence);
int kc_wal_snapshot_get(kc_wal_t *wal, kc_wal_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
