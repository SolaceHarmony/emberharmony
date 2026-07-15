// SPDX-License-Identifier: BSD-3-Clause
#include "kc_workflow.h"
#include "kc_checkpoint_internal.h"
#include "kc_codec_internal.h"
#include "kc_durable_internal.h"
#include "kcoro_port.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum {
    WORKFLOW_FORMAT_VERSION = 1,
    WORKFLOW_STATE_RECORD = 0x300,
    WORKFLOW_STATE_SIZE = 112,
    WORKFLOW_COMMAND_SIZE = 88,
    WORKFLOW_SNAPSHOT_HEADER_SIZE = 32,
};

#define WORKFLOW_SNAPSHOT_MAGIC UINT32_C(0x5357434b)

typedef struct definition_node {
    kc_workflow_definition definition;
    struct definition_node *next;
} definition_node;

typedef struct workflow_instance {
    kc_id id;
    kc_id parent_id;
    kc_id trace_id;
    kc_id last_input_id;
    uint64_t type_id;
    uint32_t version;
    uint32_t flags;
    uint64_t state_id;
    uint32_t retry_count;
    int32_t fault_code;
    void *encoded;
    size_t encoded_length;
    kc_workflow_command *commands;
    size_t command_count;
    struct workflow_instance *next;
} workflow_instance;

struct kc_workflows {
    KC_MUTEX_T mu;
    kc_wal_t *wal;
    kc_durable_t *durable;
    definition_node *definitions;
    workflow_instance *instances;
    uint64_t epoch;
    uint64_t next_instance_sequence;
    size_t definition_count;
    size_t instance_count;
    int recovered;
};

static int id_equal(kc_id left, kc_id right)
{
    return left.epoch == right.epoch && left.sequence == right.sequence;
}

static int id_empty(kc_id id)
{
    return !id.epoch && !id.sequence;
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

static definition_node *definition_exact(kc_workflows_t *workflows,
                                         uint64_t type_id, uint32_t version)
{
    for (definition_node *node = workflows->definitions; node; node = node->next) {
        if (node->definition.type_id == type_id &&
            node->definition.workflow_version == version) return node;
    }
    return NULL;
}

static definition_node *definition_latest(kc_workflows_t *workflows,
                                          uint64_t type_id)
{
    definition_node *latest = NULL;
    for (definition_node *node = workflows->definitions; node; node = node->next) {
        if (node->definition.type_id != type_id) continue;
        if (!latest || node->definition.workflow_version >
                       latest->definition.workflow_version) latest = node;
    }
    return latest;
}

static workflow_instance *instance_find(kc_workflows_t *workflows, kc_id id)
{
    for (workflow_instance *instance = workflows->instances; instance;
         instance = instance->next) {
        if (id_equal(instance->id, id)) return instance;
    }
    return NULL;
}

static void commands_free(kc_workflow_command *commands, size_t count)
{
    if (!commands) return;
    for (size_t index = 0; index < count; index++) {
        free((void *)commands[index].payload);
    }
    free(commands);
}

static void instance_content_free(workflow_instance *instance)
{
    free(instance->encoded);
    commands_free(instance->commands, instance->command_count);
    instance->encoded = NULL;
    instance->commands = NULL;
    instance->encoded_length = 0;
    instance->command_count = 0;
}

static void instance_destroy(workflow_instance *instance)
{
    if (!instance) return;
    instance_content_free(instance);
    free(instance);
}

static void instances_clear(kc_workflows_t *workflows)
{
    workflow_instance *instance = workflows->instances;
    while (instance) {
        workflow_instance *next = instance->next;
        instance_destroy(instance);
        instance = next;
    }
    workflows->instances = NULL;
    workflows->instance_count = 0;
    workflows->next_instance_sequence = 1;
}

static int command_copy(kc_workflow_command *out,
                        const kc_workflow_command *source)
{
    if (!source || source->size < sizeof(*source) ||
        source->abi_version != KC_ABI_VERSION ||
        source->kind < KC_WORKFLOW_EMIT ||
        source->kind > KC_WORKFLOW_SUBSCRIBE ||
        (source->payload_length && !source->payload)) return -EINVAL;
    *out = *source;
    out->size = sizeof(*out);
    out->abi_version = KC_ABI_VERSION;
    out->payload = NULL;
    if (source->payload_length) {
        void *payload = malloc(source->payload_length);
        if (!payload) return -ENOMEM;
        memcpy(payload, source->payload, source->payload_length);
        out->payload = payload;
    }
    return 0;
}

static int commands_copy(const kc_workflow_command *commands, size_t count,
                         kc_workflow_command **out)
{
    if (count && !commands) return -EINVAL;
    if (!count) { *out = NULL; return 0; }
    if (count > SIZE_MAX / sizeof(**out)) return -E2BIG;
    kc_workflow_command *copy = calloc(count, sizeof(*copy));
    if (!copy) return -ENOMEM;
    for (size_t index = 0; index < count; index++) {
        int rc = command_copy(&copy[index], &commands[index]);
        if (rc != 0) {
            commands_free(copy, index);
            return rc;
        }
    }
    *out = copy;
    return 0;
}

static int instance_apply(kc_workflows_t *workflows, workflow_instance *prepared)
{
    uint64_t sequence = prepared->id.sequence;
    workflow_instance *current = instance_find(workflows, prepared->id);
    if (!current) {
        prepared->next = workflows->instances;
        workflows->instances = prepared;
        workflows->instance_count++;
    } else {
        workflow_instance *next = current->next;
        instance_content_free(current);
        *current = *prepared;
        current->next = next;
        free(prepared);
    }
    if (sequence >= workflows->next_instance_sequence) {
        workflows->next_instance_sequence = sequence + 1;
    }
    return 0;
}

static int instance_encode(const workflow_instance *instance,
                           void **payload, size_t *length)
{
    if (instance->encoded_length > UINT32_MAX) return -E2BIG;
    size_t total = WORKFLOW_STATE_SIZE + instance->encoded_length;
    if (total < instance->encoded_length) return -E2BIG;
    for (size_t index = 0; index < instance->command_count; index++) {
        size_t command_length = instance->commands[index].payload_length;
        if (command_length > UINT32_MAX - WORKFLOW_COMMAND_SIZE ||
            total > SIZE_MAX - WORKFLOW_COMMAND_SIZE - command_length) return -E2BIG;
        total += WORKFLOW_COMMAND_SIZE + command_length;
    }
    if (total > UINT32_MAX || instance->command_count > UINT32_MAX) return -E2BIG;
    unsigned char *data = calloc(1, total);
    if (!data) return -ENOMEM;
    kc_put_u16(data, WORKFLOW_FORMAT_VERSION);
    kc_put_u16(data + 2, (uint16_t)instance->flags);
    kc_put_u32(data + 4, (uint32_t)total);
    kc_put_u64(data + 8, instance->type_id);
    kc_put_u32(data + 16, instance->version);
    kc_put_u32(data + 20, instance->retry_count);
    encode_id(data + 24, instance->id);
    encode_id(data + 40, instance->parent_id);
    encode_id(data + 56, instance->trace_id);
    encode_id(data + 72, instance->last_input_id);
    kc_put_u64(data + 88, instance->state_id);
    kc_put_i32(data + 96, instance->fault_code);
    kc_put_u32(data + 100, (uint32_t)instance->encoded_length);
    kc_put_u32(data + 104, (uint32_t)instance->command_count);
    if (instance->encoded_length) {
        memcpy(data + WORKFLOW_STATE_SIZE, instance->encoded,
               instance->encoded_length);
    }
    size_t offset = WORKFLOW_STATE_SIZE + instance->encoded_length;
    for (size_t index = 0; index < instance->command_count; index++) {
        const kc_workflow_command *command = &instance->commands[index];
        size_t command_size = WORKFLOW_COMMAND_SIZE + command->payload_length;
        kc_put_u32(data + offset, (uint32_t)command_size);
        kc_put_u16(data + offset + 4, (uint16_t)command->kind);
        kc_put_u64(data + offset + 8, command->route);
        encode_id(data + offset + 16, command->target_instance_id);
        encode_id(data + offset + 32, command->correlation_id);
        encode_id(data + offset + 48, command->idempotency_key);
        kc_put_u64(data + offset + 64, command->state_id);
        kc_put_u64(data + offset + 72, command->deadline_ns);
        kc_put_u32(data + offset + 80, (uint32_t)command->payload_length);
        if (command->payload_length) {
            memcpy(data + offset + WORKFLOW_COMMAND_SIZE, command->payload,
                   command->payload_length);
        }
        offset += command_size;
    }
    *payload = data;
    *length = total;
    return 0;
}

static int migrate_instance(kc_workflows_t *workflows,
                            workflow_instance *instance)
{
    definition_node *latest = definition_latest(workflows, instance->type_id);
    if (!latest) return -EPROTONOSUPPORT;
    if (latest->definition.workflow_version == instance->version) return 0;
    if (latest->definition.workflow_version < instance->version) {
        return -EPROTONOSUPPORT;
    }
    void *migrated = NULL;
    size_t migrated_length = 0;
    int rc = latest->definition.migrate(
        latest->definition.context, instance->version,
        latest->definition.workflow_version, instance->encoded,
        instance->encoded_length, &migrated, &migrated_length);
    if (rc != 0 || (migrated_length && !migrated)) {
        if (migrated) latest->definition.release(latest->definition.context,
                                                 migrated);
        return rc != 0 ? rc : -EBADMSG;
    }
    void *copy = NULL;
    if (migrated_length) {
        copy = malloc(migrated_length);
        if (!copy) {
            latest->definition.release(latest->definition.context, migrated);
            return -ENOMEM;
        }
        memcpy(copy, migrated, migrated_length);
    }
    if (migrated) latest->definition.release(latest->definition.context, migrated);
    free(instance->encoded);
    instance->encoded = copy;
    instance->encoded_length = migrated_length;
    instance->version = latest->definition.workflow_version;
    return 0;
}

static int instance_decode(kc_workflows_t *workflows,
                           const kc_wal_record *record)
{
    if (record->payload_length < WORKFLOW_STATE_SIZE) return -EBADMSG;
    const unsigned char *data = record->payload;
    uint32_t total = kc_get_u32(data + 4);
    uint32_t state_length = kc_get_u32(data + 100);
    uint32_t command_count = kc_get_u32(data + 104);
    if (kc_get_u16(data) != WORKFLOW_FORMAT_VERSION ||
        total != record->payload_length ||
        state_length > total - WORKFLOW_STATE_SIZE) return -EBADMSG;
    workflow_instance *instance = calloc(1, sizeof(*instance));
    if (!instance) return -ENOMEM;
    instance->flags = kc_get_u16(data + 2);
    instance->type_id = kc_get_u64(data + 8);
    instance->version = kc_get_u32(data + 16);
    instance->retry_count = kc_get_u32(data + 20);
    instance->id = decode_id(data + 24);
    instance->parent_id = decode_id(data + 40);
    instance->trace_id = decode_id(data + 56);
    instance->last_input_id = decode_id(data + 72);
    instance->state_id = kc_get_u64(data + 88);
    instance->fault_code = kc_get_i32(data + 96);
    if (instance->id.epoch != workflows->epoch || !instance->id.sequence ||
        instance->id.sequence == UINT64_MAX ||
        !instance->type_id || !instance->version ||
        (instance->flags & ~(KC_WORKFLOW_COMPLETE | KC_WORKFLOW_FAULTED |
                             KC_WORKFLOW_COMPENSATED))) {
        instance_destroy(instance);
        return -EBADMSG;
    }
    if (state_length) {
        instance->encoded = malloc(state_length);
        if (!instance->encoded) { instance_destroy(instance); return -ENOMEM; }
        memcpy(instance->encoded, data + WORKFLOW_STATE_SIZE, state_length);
    }
    instance->encoded_length = state_length;
    size_t command_bytes = total - WORKFLOW_STATE_SIZE - state_length;
    if ((uint64_t)command_count > command_bytes / WORKFLOW_COMMAND_SIZE) {
        instance_destroy(instance);
        return -EBADMSG;
    }
    if (command_count) {
        if (SIZE_MAX / command_count < sizeof(*instance->commands)) {
            instance_destroy(instance);
            return -E2BIG;
        }
        instance->commands = calloc(command_count, sizeof(*instance->commands));
        if (!instance->commands) { instance_destroy(instance); return -ENOMEM; }
    }
    size_t offset = WORKFLOW_STATE_SIZE + state_length;
    for (uint32_t index = 0; index < command_count; index++) {
        if (total - offset < WORKFLOW_COMMAND_SIZE) {
            instance_destroy(instance);
            return -EBADMSG;
        }
        uint32_t command_size = kc_get_u32(data + offset);
        uint32_t payload_length = kc_get_u32(data + offset + 80);
        uint16_t kind = kc_get_u16(data + offset + 4);
        if (command_size != WORKFLOW_COMMAND_SIZE + (uint64_t)payload_length ||
            command_size > total - offset || kind < KC_WORKFLOW_EMIT ||
            kind > KC_WORKFLOW_SUBSCRIBE) {
            instance_destroy(instance);
            return -EBADMSG;
        }
        kc_workflow_command *command = &instance->commands[index];
        *command = (kc_workflow_command){
            .size = sizeof(*command), .abi_version = KC_ABI_VERSION,
            .kind = (kc_workflow_command_kind)kind,
            .route = kc_get_u64(data + offset + 8),
            .target_instance_id = decode_id(data + offset + 16),
            .correlation_id = decode_id(data + offset + 32),
            .idempotency_key = decode_id(data + offset + 48),
            .state_id = kc_get_u64(data + offset + 64),
            .deadline_ns = kc_get_u64(data + offset + 72),
            .payload_length = payload_length,
        };
        if (payload_length) {
            void *copy = malloc(payload_length);
            if (!copy) { instance_destroy(instance); return -ENOMEM; }
            memcpy(copy, data + offset + WORKFLOW_COMMAND_SIZE, payload_length);
            command->payload = copy;
        }
        instance->command_count++;
        offset += command_size;
    }
    if (offset != total) { instance_destroy(instance); return -EBADMSG; }
    int rc = migrate_instance(workflows, instance);
    if (rc != 0) { instance_destroy(instance); return rc; }
    return instance_apply(workflows, instance);
}

static int recover_record(void *context, const kc_wal_record *record)
{
    if (record->type != WORKFLOW_STATE_RECORD) return 0;
    return instance_decode(context, record);
}

static int workflow_snapshot_encode(kc_workflows_t *workflows, void **payload,
                                    size_t *length)
{
    size_t count = workflows->instance_count;
    if (count > UINT32_MAX || count > SIZE_MAX / sizeof(void *)) return -E2BIG;
    void **records = count ? calloc(count, sizeof(*records)) : NULL;
    size_t *lengths = count ? calloc(count, sizeof(*lengths)) : NULL;
    if (count && (!records || !lengths)) {
        free(records);
        free(lengths);
        return -ENOMEM;
    }
    size_t total = WORKFLOW_SNAPSHOT_HEADER_SIZE;
    size_t index = 0;
    int rc = 0;
    for (workflow_instance *instance = workflows->instances; instance;
         instance = instance->next) {
        if (index >= count) { rc = -EBADMSG; break; }
        rc = instance_encode(instance, &records[index], &lengths[index]);
        if (rc != 0 || lengths[index] > UINT32_MAX ||
            total > SIZE_MAX - sizeof(uint32_t) - lengths[index]) {
            if (rc == 0) rc = -E2BIG;
            break;
        }
        total += sizeof(uint32_t) + lengths[index];
        index++;
    }
    if (rc == 0 && index != count) rc = -EBADMSG;
    unsigned char *data = NULL;
    if (rc == 0) {
        data = calloc(1, total);
        if (!data) rc = -ENOMEM;
    }
    if (rc == 0) {
        kc_put_u32(data, WORKFLOW_SNAPSHOT_MAGIC);
        kc_put_u16(data + 4, WORKFLOW_FORMAT_VERSION);
        kc_put_u16(data + 6, WORKFLOW_SNAPSHOT_HEADER_SIZE);
        kc_put_u64(data + 8, workflows->epoch);
        kc_put_u64(data + 16, workflows->next_instance_sequence);
        kc_put_u32(data + 24, (uint32_t)count);
        size_t offset = WORKFLOW_SNAPSHOT_HEADER_SIZE;
        for (size_t record = 0; record < count; record++) {
            kc_put_u32(data + offset, (uint32_t)lengths[record]);
            memcpy(data + offset + sizeof(uint32_t), records[record],
                   lengths[record]);
            offset += sizeof(uint32_t) + lengths[record];
        }
        *payload = data;
        *length = total;
    }
    for (size_t record = 0; record < count; record++) free(records[record]);
    free(records);
    free(lengths);
    return rc;
}

static int workflow_snapshot_decode(kc_workflows_t *workflows,
                                    const void *payload, size_t length)
{
    if (length < WORKFLOW_SNAPSHOT_HEADER_SIZE) return -EBADMSG;
    const unsigned char *data = payload;
    if (kc_get_u32(data) != WORKFLOW_SNAPSHOT_MAGIC ||
        kc_get_u16(data + 4) != WORKFLOW_FORMAT_VERSION ||
        kc_get_u16(data + 6) != WORKFLOW_SNAPSHOT_HEADER_SIZE ||
        kc_get_u64(data + 8) != workflows->epoch) return -EBADMSG;
    uint64_t next_sequence = kc_get_u64(data + 16);
    uint32_t count = kc_get_u32(data + 24);
    size_t offset = WORKFLOW_SNAPSHOT_HEADER_SIZE;
    for (uint32_t index = 0; index < count; index++) {
        if (length - offset < sizeof(uint32_t)) return -EBADMSG;
        uint32_t record_length = kc_get_u32(data + offset);
        offset += sizeof(uint32_t);
        if (record_length > length - offset) return -EBADMSG;
        kc_wal_record record = {
            .size = sizeof(record), .abi_version = KC_ABI_VERSION,
            .type = WORKFLOW_STATE_RECORD,
            .payload_length = record_length,
            .runtime_epoch = workflows->epoch,
            .payload = data + offset,
        };
        int rc = instance_decode(workflows, &record);
        if (rc != 0) return rc;
        offset += record_length;
    }
    if (offset != length || !next_sequence ||
        next_sequence < workflows->next_instance_sequence) return -EBADMSG;
    workflows->next_instance_sequence = next_sequence;
    return 0;
}

static int joins_ready(kc_workflows_t *workflows,
                       const workflow_instance *instance)
{
    for (size_t index = 0; index < instance->command_count; index++) {
        const kc_workflow_command *command = &instance->commands[index];
        if (command->kind != KC_WORKFLOW_JOIN) continue;
        workflow_instance *target = instance_find(workflows,
                                                  command->target_instance_id);
        if (!target || !(target->flags & KC_WORKFLOW_COMPLETE)) return 0;
    }
    return 1;
}

static int step_prepare(kc_workflows_t *workflows, definition_node *definition,
                        const workflow_instance *current, kc_id id,
                        kc_id parent_id, kc_id trace_id, kc_id input_id,
                        const kc_message *input, int mode,
                        workflow_instance **out)
{
    void *decoded = NULL;
    size_t decoded_length = 0;
    int rc = 0;
    if (current) {
        rc = definition->definition.decode(
            definition->definition.context, current->encoded,
            current->encoded_length, &decoded, &decoded_length);
        if (rc != 0 || (decoded_length && !decoded)) {
            if (decoded) definition->definition.release(
                definition->definition.context, decoded);
            return rc != 0 ? rc : -EBADMSG;
        }
    }
    kc_workflow_step step = {
        .size = sizeof(step), .abi_version = KC_ABI_VERSION,
    };
    if (!current) {
        rc = definition->definition.initialize(definition->definition.context,
                                               input, &step);
    } else if (mode == 1) {
        if (!definition->definition.compensate) rc = -ENOTSUP;
        else rc = definition->definition.compensate(
            definition->definition.context, current->state_id, decoded,
            decoded_length, input, &step);
    } else {
        rc = definition->definition.transition(
            definition->definition.context, current->state_id, decoded,
            decoded_length, input, &step);
        if (rc != 0 && definition->definition.fault) {
            if (step.owner) definition->definition.release(
                definition->definition.context, step.owner);
            step = (kc_workflow_step){
                .size = sizeof(step), .abi_version = KC_ABI_VERSION,
            };
            rc = definition->definition.fault(
                definition->definition.context, current->state_id, decoded,
                decoded_length, input, rc, &step);
        }
    }
    if (decoded) definition->definition.release(definition->definition.context,
                                                decoded);
    if (rc != 0) {
        if (step.owner) definition->definition.release(
            definition->definition.context, step.owner);
        return rc;
    }
    if (step.size < sizeof(step) || step.abi_version != KC_ABI_VERSION ||
        (step.state_length && !step.state) ||
        (step.command_count && !step.commands) ||
        (step.flags & ~(KC_WORKFLOW_COMPLETE | KC_WORKFLOW_FAULTED |
                        KC_WORKFLOW_COMPENSATED))) {
        if (step.owner) definition->definition.release(
            definition->definition.context, step.owner);
        return -EINVAL;
    }
    void *encoded = NULL;
    size_t encoded_length = 0;
    rc = definition->definition.encode(
        definition->definition.context, step.state, step.state_length,
        &encoded, &encoded_length);
    if (rc != 0 || (encoded_length && !encoded)) {
        if (encoded) definition->definition.release(
            definition->definition.context, encoded);
        if (step.owner) definition->definition.release(
            definition->definition.context, step.owner);
        return rc != 0 ? rc : -EBADMSG;
    }
    workflow_instance *prepared = calloc(1, sizeof(*prepared));
    if (!prepared) rc = -ENOMEM;
    if (rc == 0 && encoded_length) {
        prepared->encoded = malloc(encoded_length);
        if (!prepared->encoded) rc = -ENOMEM;
        else memcpy(prepared->encoded, encoded, encoded_length);
    }
    if (encoded) definition->definition.release(definition->definition.context,
                                                encoded);
    if (rc == 0) {
        rc = commands_copy(step.commands, step.command_count,
                           &prepared->commands);
    }
    if (rc == 0) {
        prepared->id = id;
        prepared->parent_id = parent_id;
        prepared->trace_id = trace_id;
        prepared->last_input_id = input_id;
        prepared->type_id = definition->definition.type_id;
        prepared->version = definition->definition.workflow_version;
        prepared->flags = step.flags;
        if (mode == 1) prepared->flags |= KC_WORKFLOW_COMPENSATED;
        prepared->state_id = step.state_id;
        prepared->fault_code = step.fault_code;
        prepared->encoded_length = encoded_length;
        prepared->command_count = step.command_count;
        prepared->retry_count = current ? current->retry_count : 0;
        for (size_t index = 0; index < prepared->command_count; index++) {
            if (prepared->commands[index].kind == KC_WORKFLOW_RETRY) {
                if (prepared->retry_count == UINT32_MAX) {
                    rc = -EOVERFLOW;
                    break;
                }
                prepared->retry_count++;
            }
        }
    }
    if (step.owner) definition->definition.release(definition->definition.context,
                                                   step.owner);
    if (rc != 0) { instance_destroy(prepared); return rc; }
    *out = prepared;
    (void)workflows;
    return 0;
}

static int transition_commit(kc_workflows_t *workflows,
                             workflow_instance *prepared,
                             kc_id input_message_id)
{
    void *encoded = NULL;
    size_t encoded_length = 0;
    int rc = instance_encode(prepared, &encoded, &encoded_length);
    if (rc != 0) return rc;
    kc_durable_batch *batch = NULL;
    rc = kc_durable_batch_begin(workflows->durable, &batch);
    if (rc == 0) rc = kc_durable_batch_ack(batch, input_message_id);
    if (rc == 0) rc = kc_durable_batch_record(batch, WORKFLOW_STATE_RECORD,
                                              encoded, encoded_length);
    for (size_t index = 0; rc == 0 && index < prepared->command_count; index++) {
        const kc_workflow_command *command = &prepared->commands[index];
        if (command->kind != KC_WORKFLOW_EMIT) continue;
        kc_publish publish = {
            .size = sizeof(publish), .abi_version = KC_ABI_VERSION,
            .route = command->route,
            .correlation_id = id_empty(command->correlation_id)
                ? prepared->id : command->correlation_id,
            .trace_id = prepared->trace_id,
            .idempotency_key = command->idempotency_key,
            .payload = command->payload,
            .payload_length = command->payload_length,
        };
        kc_id ignored;
        rc = kc_durable_batch_publish(batch, &publish, &ignored);
    }
    free(encoded);
    if (rc == 0) rc = kc_durable_batch_commit(batch);
    else if (batch) kc_durable_batch_abort(batch);
    if (rc != 0) return rc;
    return instance_apply(workflows, prepared);
}

int kc_workflows_create(const kc_workflows_config *config, kc_workflows_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->wal ||
        !config->durable) return -EINVAL;
    if (kc_durable_wal_internal(config->durable) != config->wal) return -EXDEV;
    kc_wal_snapshot snapshot = {
        .size = sizeof(snapshot), .abi_version = KC_ABI_VERSION,
    };
    int rc = kc_wal_snapshot_get(config->wal, &snapshot);
    if (rc != 0) return rc;
    kc_workflows_t *workflows = calloc(1, sizeof(*workflows));
    if (!workflows) return -ENOMEM;
    if (KC_MUTEX_INIT(&workflows->mu) != 0) { free(workflows); return -ENOMEM; }
    workflows->wal = config->wal;
    workflows->durable = config->durable;
    workflows->epoch = snapshot.runtime_epoch;
    workflows->next_instance_sequence = 1;
    kc_durable_workflow_attach(workflows->durable);
    *out = workflows;
    return 0;
}

void kc_workflows_destroy(kc_workflows_t *workflows)
{
    if (!workflows) return;
    kc_durable_workflow_detach(workflows->durable);
    definition_node *definition = workflows->definitions;
    while (definition) {
        definition_node *next = definition->next;
        free(definition);
        definition = next;
    }
    instances_clear(workflows);
    KC_MUTEX_DESTROY(&workflows->mu);
    free(workflows);
}

int kc_workflow_register(kc_workflows_t *workflows,
                         const kc_workflow_definition *definition)
{
    if (!workflows || !definition || definition->size < sizeof(*definition) ||
        definition->abi_version != KC_ABI_VERSION || !definition->type_id ||
        !definition->workflow_version || !definition->initialize ||
        !definition->transition || !definition->encode || !definition->decode ||
        !definition->migrate || !definition->release) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    if (workflows->recovered) { KC_MUTEX_UNLOCK(&workflows->mu); return -EBUSY; }
    if (definition_exact(workflows, definition->type_id,
                         definition->workflow_version)) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EEXIST;
    }
    definition_node *node = calloc(1, sizeof(*node));
    if (!node) { KC_MUTEX_UNLOCK(&workflows->mu); return -ENOMEM; }
    node->definition = *definition;
    node->next = workflows->definitions;
    workflows->definitions = node;
    workflows->definition_count++;
    KC_MUTEX_UNLOCK(&workflows->mu);
    return 0;
}

int kc_workflows_recover(kc_workflows_t *workflows)
{
    if (!workflows) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    if (workflows->recovered) { KC_MUTEX_UNLOCK(&workflows->mu); return 0; }
    if (!workflows->definition_count) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EPROTONOSUPPORT;
    }
    instances_clear(workflows);
    kc_wal_snapshot wal_snapshot = {
        .size = sizeof(wal_snapshot), .abi_version = KC_ABI_VERSION,
    };
    int rc = kc_wal_snapshot_get(workflows->wal, &wal_snapshot);
    if (rc == 0 && wal_snapshot.snapshot_valid) {
        size_t snapshot_length = 0;
        uint64_t snapshot_sequence = 0;
        rc = kc_wal_snapshot_load(workflows->wal, NULL, 0, &snapshot_length,
                                  &snapshot_sequence);
        if (rc == -ENOSPC) {
            void *snapshot = malloc(snapshot_length);
            if (!snapshot) rc = -ENOMEM;
            else {
                rc = kc_wal_snapshot_load(workflows->wal, snapshot,
                                          snapshot_length, &snapshot_length,
                                          &snapshot_sequence);
                if (rc == 0) {
                    const void *section = NULL;
                    size_t section_length = 0;
                    rc = kc_checkpoint_find(snapshot, snapshot_length,
                                            KC_CHECKPOINT_WORKFLOWS,
                                            &section, &section_length);
                    if (rc == -ENOENT) rc = 0;
                    else if (rc == 0) {
                        rc = workflow_snapshot_decode(workflows, section,
                                                      section_length);
                    }
                }
                free(snapshot);
            }
        } else if (rc == 0) rc = -EBADMSG;
    }
    if (rc == 0) rc = kc_wal_recover(workflows->wal, recover_record, workflows);
    if (rc == 0) workflows->recovered = 1;
    else instances_clear(workflows);
    KC_MUTEX_UNLOCK(&workflows->mu);
    return rc;
}

int kc_workflow_start_instance(kc_workflows_t *workflows,
                               const kc_workflow_start *start,
                               kc_id *instance_id)
{
    if (!workflows || !start || !instance_id || start->size < sizeof(*start) ||
        start->abi_version != KC_ABI_VERSION || !start->type_id ||
        id_empty(start->input_message_id)) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    if (!workflows->recovered) { KC_MUTEX_UNLOCK(&workflows->mu); return -EAGAIN; }
    definition_node *definition = start->workflow_version
        ? definition_exact(workflows, start->type_id, start->workflow_version)
        : definition_latest(workflows, start->type_id);
    if (!definition) { KC_MUTEX_UNLOCK(&workflows->mu); return -EPROTONOSUPPORT; }
    kc_id id = id_empty(start->instance_id)
        ? (kc_id){ workflows->epoch, workflows->next_instance_sequence }
        : start->instance_id;
    if (id.sequence == UINT64_MAX) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EOVERFLOW;
    }
    if (id.epoch != workflows->epoch || !id.sequence) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EINVAL;
    }
    workflow_instance *existing = instance_find(workflows, id);
    if (existing) {
        int same = id_equal(existing->last_input_id, start->input_message_id);
        *instance_id = id;
        KC_MUTEX_UNLOCK(&workflows->mu);
        return same ? 0 : -EEXIST;
    }
    kc_message input = { .size = sizeof(input), .abi_version = KC_ABI_VERSION };
    int rc = kc_durable_lookup(workflows->durable, start->input_message_id, &input);
    workflow_instance *prepared = NULL;
    kc_id trace = id_empty(start->trace_id) ? id : start->trace_id;
    if (rc == 0) rc = step_prepare(
        workflows, definition, NULL, id, start->parent_instance_id, trace,
        start->input_message_id, &input, 0, &prepared);
    if (rc == 0) rc = transition_commit(workflows, prepared,
                                        start->input_message_id);
    if (rc != 0) instance_destroy(prepared);
    else *instance_id = id;
    KC_MUTEX_UNLOCK(&workflows->mu);
    return rc;
}

static int dispatch_mode(kc_workflows_t *workflows, kc_id instance_id,
                         kc_id input_message_id, int mode)
{
    if (!workflows || id_empty(instance_id) || id_empty(input_message_id)) {
        return -EINVAL;
    }
    KC_MUTEX_LOCK(&workflows->mu);
    if (!workflows->recovered) { KC_MUTEX_UNLOCK(&workflows->mu); return -EAGAIN; }
    workflow_instance *current = instance_find(workflows, instance_id);
    if (!current) { KC_MUTEX_UNLOCK(&workflows->mu); return -ENOENT; }
    if (id_equal(current->last_input_id, input_message_id)) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return 0;
    }
    if (!mode && (current->flags & (KC_WORKFLOW_COMPLETE | KC_WORKFLOW_FAULTED))) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EALREADY;
    }
    if (!mode && !joins_ready(workflows, current)) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EAGAIN;
    }
    definition_node *definition = definition_exact(
        workflows, current->type_id, current->version);
    if (!definition) { KC_MUTEX_UNLOCK(&workflows->mu); return -EPROTONOSUPPORT; }
    kc_message input = { .size = sizeof(input), .abi_version = KC_ABI_VERSION };
    int rc = kc_durable_lookup(workflows->durable, input_message_id, &input);
    workflow_instance *prepared = NULL;
    if (rc == 0) rc = step_prepare(
        workflows, definition, current, current->id, current->parent_id,
        current->trace_id, input_message_id, &input, mode, &prepared);
    if (rc == 0) rc = transition_commit(workflows, prepared, input_message_id);
    if (rc != 0) instance_destroy(prepared);
    KC_MUTEX_UNLOCK(&workflows->mu);
    return rc;
}

int kc_workflow_dispatch(kc_workflows_t *workflows, kc_id instance_id,
                         kc_id input_message_id)
{
    return dispatch_mode(workflows, instance_id, input_message_id, 0);
}

int kc_workflow_compensate(kc_workflows_t *workflows, kc_id instance_id,
                           kc_id input_message_id)
{
    return dispatch_mode(workflows, instance_id, input_message_id, 1);
}

int kc_workflow_dispatch_correlation(kc_workflows_t *workflows,
                                     kc_id correlation_id,
                                     kc_id input_message_id,
                                     kc_id *instance_id)
{
    if (!workflows || !instance_id || id_empty(correlation_id)) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    kc_id found = {0};
    for (workflow_instance *instance = workflows->instances; instance;
         instance = instance->next) {
        for (size_t index = 0; index < instance->command_count; index++) {
            kc_workflow_command *command = &instance->commands[index];
            if (command->kind != KC_WORKFLOW_SUBSCRIBE ||
                !id_equal(command->correlation_id, correlation_id)) continue;
            if (!id_empty(found) && !id_equal(found, instance->id)) {
                KC_MUTEX_UNLOCK(&workflows->mu);
                return -EEXIST;
            }
            found = instance->id;
        }
    }
    KC_MUTEX_UNLOCK(&workflows->mu);
    if (id_empty(found)) return -ENOENT;
    int rc = kc_workflow_dispatch(workflows, found, input_message_id);
    if (rc == 0) *instance_id = found;
    return rc;
}

int kc_workflow_lookup(kc_workflows_t *workflows, kc_id instance_id,
                       kc_workflow_instance_snapshot *out)
{
    if (!workflows || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    workflow_instance *instance = instance_find(workflows, instance_id);
    if (!instance) { KC_MUTEX_UNLOCK(&workflows->mu); return -ENOENT; }
    *out = (kc_workflow_instance_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .instance_id = instance->id, .parent_instance_id = instance->parent_id,
        .trace_id = instance->trace_id, .last_input_id = instance->last_input_id,
        .type_id = instance->type_id, .workflow_version = instance->version,
        .flags = instance->flags, .state_id = instance->state_id,
        .retry_count = instance->retry_count, .fault_code = instance->fault_code,
        .encoded_state = instance->encoded,
        .encoded_state_length = instance->encoded_length,
        .commands = instance->commands, .command_count = instance->command_count,
    };
    KC_MUTEX_UNLOCK(&workflows->mu);
    return 0;
}

int kc_workflows_checkpoint(kc_workflows_t *workflows)
{
    if (!workflows) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    if (!workflows->recovered) {
        KC_MUTEX_UNLOCK(&workflows->mu);
        return -EAGAIN;
    }
    kc_durable_lock_internal(workflows->durable);
    void *durable_payload = NULL;
    size_t durable_length = 0;
    void *workflow_payload = NULL;
    size_t workflow_length = 0;
    int rc = kc_durable_snapshot_encode_internal(
        workflows->durable, &durable_payload, &durable_length);
    if (rc == 0) {
        rc = workflow_snapshot_encode(workflows, &workflow_payload,
                                      &workflow_length);
    }
    void *checkpoint = NULL;
    size_t checkpoint_length = 0;
    if (rc == 0) {
        kc_checkpoint_section sections[2] = {
            { KC_CHECKPOINT_DURABLE, durable_payload, durable_length },
            { KC_CHECKPOINT_WORKFLOWS, workflow_payload, workflow_length },
        };
        rc = kc_checkpoint_encode(sections, 2, &checkpoint,
                                  &checkpoint_length);
    }
    if (rc == 0) {
        rc = kc_wal_snapshot_write(workflows->wal, checkpoint,
                                   checkpoint_length);
    }
    free(checkpoint);
    free(workflow_payload);
    free(durable_payload);
    kc_durable_unlock_internal(workflows->durable);
    KC_MUTEX_UNLOCK(&workflows->mu);
    return rc;
}

int kc_workflows_snapshot_get(kc_workflows_t *workflows,
                              kc_workflows_snapshot *out)
{
    if (!workflows || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&workflows->mu);
    *out = (kc_workflows_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .runtime_epoch = workflows->epoch,
        .next_instance_sequence = workflows->next_instance_sequence,
        .definitions = workflows->definition_count,
        .instances = workflows->instance_count,
        .recovered = (unsigned)workflows->recovered,
    };
    for (workflow_instance *instance = workflows->instances; instance;
         instance = instance->next) {
        if (instance->flags & KC_WORKFLOW_COMPLETE) out->completed++;
        if (instance->flags & KC_WORKFLOW_FAULTED) out->faulted++;
        for (size_t index = 0; index < instance->command_count; index++) {
            if (instance->commands[index].kind == KC_WORKFLOW_SUBSCRIBE) {
                out->subscriptions++;
            } else if (instance->commands[index].kind == KC_WORKFLOW_JOIN) {
                out->joins++;
            }
        }
    }
    KC_MUTEX_UNLOCK(&workflows->mu);
    return 0;
}
