// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_WORKFLOW_H
#define KC_WORKFLOW_H

#include "kc_durable.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_workflows kc_workflows_t;

typedef enum kc_workflow_command_kind {
    KC_WORKFLOW_EMIT = 1,
    KC_WORKFLOW_RETRY,
    KC_WORKFLOW_COMPENSATE,
    KC_WORKFLOW_JOIN,
    KC_WORKFLOW_SUBSCRIBE,
} kc_workflow_command_kind;

enum kc_workflow_flags {
    KC_WORKFLOW_COMPLETE = 1u << 0,
    KC_WORKFLOW_FAULTED = 1u << 1,
    KC_WORKFLOW_COMPENSATED = 1u << 2,
};

typedef struct kc_workflow_command {
    uint32_t size;
    uint32_t abi_version;
    kc_workflow_command_kind kind;
    uint32_t reserved;
    uint64_t route;
    kc_id target_instance_id;
    kc_id correlation_id;
    kc_id idempotency_key;
    uint64_t state_id;
    uint64_t deadline_ns;
    const void *payload;
    size_t payload_length;
} kc_workflow_command;

typedef struct kc_workflow_step {
    uint32_t size;
    uint32_t abi_version;
    uint64_t state_id;
    const void *state;
    size_t state_length;
    const kc_workflow_command *commands;
    size_t command_count;
    uint32_t flags;
    int32_t fault_code;
    void *owner;
} kc_workflow_step;

typedef int (*kc_workflow_initialize_fn)(void *context,
                                          const kc_message *input,
                                          kc_workflow_step *step);
typedef int (*kc_workflow_transition_fn)(void *context, uint64_t state_id,
                                         const void *state, size_t state_length,
                                         const kc_message *input,
                                         kc_workflow_step *step);
typedef int (*kc_workflow_fault_fn)(void *context, uint64_t state_id,
                                    const void *state, size_t state_length,
                                    const kc_message *input, int error,
                                    kc_workflow_step *step);
typedef int (*kc_workflow_encode_fn)(void *context, const void *state,
                                     size_t state_length, void **encoded,
                                     size_t *encoded_length);
typedef int (*kc_workflow_decode_fn)(void *context, const void *encoded,
                                     size_t encoded_length, void **state,
                                     size_t *state_length);
typedef int (*kc_workflow_migrate_fn)(void *context, uint32_t from_version,
                                      uint32_t to_version,
                                      const void *encoded,
                                      size_t encoded_length,
                                      void **migrated,
                                      size_t *migrated_length);
typedef void (*kc_workflow_release_fn)(void *context, void *allocation);

typedef struct kc_workflow_definition {
    uint32_t size;
    uint32_t abi_version;
    uint64_t type_id;
    uint32_t workflow_version;
    uint32_t reserved;
    kc_workflow_initialize_fn initialize;
    kc_workflow_transition_fn transition;
    kc_workflow_transition_fn compensate;
    kc_workflow_fault_fn fault;
    kc_workflow_encode_fn encode;
    kc_workflow_decode_fn decode;
    kc_workflow_migrate_fn migrate;
    kc_workflow_release_fn release;
    void *context;
} kc_workflow_definition;

typedef struct kc_workflows_config {
    uint32_t size;
    uint32_t abi_version;
    kc_wal_t *wal;
    kc_durable_t *durable;
} kc_workflows_config;

typedef struct kc_workflow_start {
    uint32_t size;
    uint32_t abi_version;
    uint64_t type_id;
    uint32_t workflow_version;
    uint32_t reserved;
    kc_id instance_id;
    kc_id parent_instance_id;
    kc_id trace_id;
    kc_id input_message_id;
} kc_workflow_start;

typedef struct kc_workflow_instance_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id instance_id;
    kc_id parent_instance_id;
    kc_id trace_id;
    kc_id last_input_id;
    uint64_t type_id;
    uint32_t workflow_version;
    uint32_t flags;
    uint64_t state_id;
    uint32_t retry_count;
    int32_t fault_code;
    const void *encoded_state;
    size_t encoded_state_length;
    const kc_workflow_command *commands;
    size_t command_count;
} kc_workflow_instance_snapshot;

typedef struct kc_workflows_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t runtime_epoch;
    uint64_t next_instance_sequence;
    size_t definitions;
    size_t instances;
    size_t completed;
    size_t faulted;
    size_t subscriptions;
    size_t joins;
    unsigned recovered;
} kc_workflows_snapshot;

int kc_workflows_create(const kc_workflows_config *config, kc_workflows_t **out);
void kc_workflows_destroy(kc_workflows_t *workflows);
int kc_workflow_register(kc_workflows_t *workflows,
                         const kc_workflow_definition *definition);
int kc_workflows_recover(kc_workflows_t *workflows);
int kc_workflow_start_instance(kc_workflows_t *workflows,
                               const kc_workflow_start *start,
                               kc_id *instance_id);
int kc_workflow_dispatch(kc_workflows_t *workflows, kc_id instance_id,
                         kc_id input_message_id);
int kc_workflow_compensate(kc_workflows_t *workflows, kc_id instance_id,
                           kc_id input_message_id);
int kc_workflow_dispatch_correlation(kc_workflows_t *workflows,
                                     kc_id correlation_id,
                                     kc_id input_message_id,
                                     kc_id *instance_id);
/* Encoded state and command pointers are borrowed until the next mutation of
 * this workflow engine. Callers must serialize their use with mutations. */
int kc_workflow_lookup(kc_workflows_t *workflows, kc_id instance_id,
                       kc_workflow_instance_snapshot *out);
int kc_workflows_checkpoint(kc_workflows_t *workflows);
int kc_workflows_snapshot_get(kc_workflows_t *workflows,
                              kc_workflows_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
