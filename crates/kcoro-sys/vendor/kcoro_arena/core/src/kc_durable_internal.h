// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_durable.h"

typedef struct kc_durable_batch kc_durable_batch;

int kc_durable_batch_begin(kc_durable_t *durable, kc_durable_batch **out);
int kc_durable_batch_ack(kc_durable_batch *batch, kc_id message_id);
int kc_durable_batch_record(kc_durable_batch *batch, uint16_t type,
                            const void *payload, size_t length);
int kc_durable_batch_publish(kc_durable_batch *batch, const kc_publish *publish,
                             kc_id *message_id);
int kc_durable_batch_commit(kc_durable_batch *batch);
void kc_durable_batch_abort(kc_durable_batch *batch);
void kc_durable_lock_internal(kc_durable_t *durable);
void kc_durable_unlock_internal(kc_durable_t *durable);
int kc_durable_snapshot_encode_internal(kc_durable_t *durable, void **payload,
                                        size_t *length);
kc_wal_t *kc_durable_wal_internal(kc_durable_t *durable);
void kc_durable_workflow_attach(kc_durable_t *durable);
void kc_durable_workflow_detach(kc_durable_t *durable);
