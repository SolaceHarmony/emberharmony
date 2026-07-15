// SPDX-License-Identifier: BSD-3-Clause
#include "kc_channel_internal.h"
#include "kc_runtime_internal.h"
#include "kc_chan_api.h"

#include <errno.h>
#include <stdlib.h>

static void queue_push(kc_op **head, kc_op **tail, size_t *count, kc_op *op)
{
    kc_op_retain(op);
    op->prev = *tail;
    op->next = NULL;
    if (*tail) (*tail)->next = op;
    else *head = op;
    *tail = op;
    op->linked = 1;
    (*count)++;
    atomic_store_explicit(&op->state, KC_OP_WAITING, memory_order_release);
}

static kc_op *queue_pop(kc_op **head, kc_op **tail, size_t *count)
{
    kc_op *op = *head;
    if (!op) return NULL;
    *head = op->next;
    if (*head) (*head)->prev = NULL;
    else *tail = NULL;
    op->prev = NULL;
    op->next = NULL;
    op->linked = 0;
    if (*count) (*count)--;
    return op;
}

static void queue_remove(kc_op **head, kc_op **tail, size_t *count, kc_op *op)
{
    if (op->prev) op->prev->next = op->next;
    else *head = op->next;
    if (op->next) op->next->prev = op->prev;
    else *tail = op->prev;
    op->prev = NULL;
    op->next = NULL;
    op->linked = 0;
    if (*count) (*count)--;
}

static int ring_grow(struct kc_chan *channel)
{
    size_t capacity = channel->capacity ? channel->capacity * 2 : 64;
    if (capacity < channel->capacity) return -EOVERFLOW;
    kc_descriptor_t **ring = calloc(capacity, sizeof(*ring));
    if (!ring) return -ENOMEM;
    for (size_t i = 0; i < channel->count; i++) {
        ring[i] = channel->ring[(channel->head + i) % channel->capacity];
    }
    free(channel->ring);
    channel->ring = ring;
    channel->capacity = capacity;
    channel->head = 0;
    return 0;
}

static void channel_free(struct kc_chan *channel)
{
    if (!channel) return;
    for (size_t i = 0; i < channel->count; i++) {
        kc_descriptor_release(channel->ring[(channel->head + i) % channel->capacity]);
    }
    kc_descriptor_release(channel->conflated);
    free(channel->ring);
    KC_MUTEX_DESTROY(&channel->mu);
    kc_runtime_release_internal(channel->runtime);
    free(channel);
}

int kc_channel_create(kc_runtime_t *runtime, const kc_channel_config *config,
                      kc_channel_t **out)
{
    if (!runtime || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->element_size ||
        (config->kind != KC_CHANNEL_RENDEZVOUS &&
         config->kind != KC_CHANNEL_BUFFERED &&
         config->kind != KC_CHANNEL_UNLIMITED &&
         config->kind != KC_CHANNEL_CONFLATED)) return -EINVAL;
    struct kc_chan *channel = calloc(1, sizeof(*channel));
    if (!channel) return -ENOMEM;
    atomic_init(&channel->refs, 1);
    if (KC_MUTEX_INIT(&channel->mu) != 0) { free(channel); return -ENOMEM; }
    channel->runtime = runtime;
    kc_runtime_retain_internal(runtime);
    channel->id = (kc_id){ runtime->epoch, kc_runtime_next_sequence(runtime) };
    channel->kind = config->kind;
    channel->element_size = config->element_size;
    if (config->kind == KC_CHANNEL_BUFFERED || config->kind == KC_CHANNEL_UNLIMITED) {
        channel->capacity = config->capacity ? config->capacity : 64;
        channel->ring = calloc(channel->capacity, sizeof(*channel->ring));
        if (!channel->ring) { channel_free(channel); return -ENOMEM; }
    }
    kc_runtime_register_channel(runtime, channel);
    *out = channel;
    return 0;
}

void kc_channel_retain(kc_channel_t *channel)
{
    if (channel) atomic_fetch_add_explicit(&channel->refs, 1, memory_order_relaxed);
}

void kc_channel_release(kc_channel_t *channel)
{
    if (!channel) return;
    if (atomic_fetch_sub_explicit(&channel->refs, 1, memory_order_acq_rel) == 1) {
        kc_runtime_unregister_channel(channel->runtime, channel);
        channel_free(channel);
    }
}

int kc_channel_snapshot_get(kc_channel_t *channel, kc_channel_snapshot *out)
{
    if (!channel || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&channel->mu);
    size_t depth = channel->kind == KC_CHANNEL_CONFLATED
        ? (channel->conflated != NULL) : channel->count;
    *out = (kc_channel_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .runtime_epoch = channel->id.epoch, .sequence = channel->id.sequence,
        .kind = channel->kind, .element_size = channel->element_size,
        .depth = depth, .send_waiters = channel->send_waiters,
        .receive_waiters = channel->receive_waiters,
        .logical_bytes = depth * channel->element_size,
        .closed = (unsigned)channel->closed,
    };
    KC_MUTEX_UNLOCK(&channel->mu);
    return 0;
}

size_t kc_channel_length(kc_channel_t *channel)
{
    if (!channel) return 0;
    KC_MUTEX_LOCK(&channel->mu);
    size_t count = channel->kind == KC_CHANNEL_CONFLATED
        ? (channel->conflated != 0) : channel->count;
    KC_MUTEX_UNLOCK(&channel->mu);
    return count;
}

static int match_pair(kc_op *send, kc_op *recv)
{
    kc_payload payload = {0};
    if (kc_descriptor_payload(send->descriptor, &payload) != 0) {
        int send_won = kc_op_claim_locked(send, KC_CAUSE_FAILURE,
                                          &(kc_payload){ .status = -EPIPE });
        int recv_won = kc_op_claim_locked(recv, KC_CAUSE_FAILURE,
                                          &(kc_payload){ .status = -EPIPE });
        return send_won && recv_won;
    }
    if (!kc_op_claim_locked(send, KC_CAUSE_MATCH, &(kc_payload){ .status = 0 }) ||
        !kc_op_claim_locked(recv, KC_CAUSE_MATCH, &payload)) return 0;
    recv->result_descriptor = send->descriptor;
    send->descriptor = NULL;
    return 1;
}

int kc_channel_submit(kc_channel_t *channel, kc_op *op)
{
    if (!channel || !op || op->channel != channel) return -EINVAL;
    kc_op *paired = NULL;
    kc_op *promoted = NULL;
    kc_descriptor_t *released = NULL;
    int completed = 0;
    int promoted_completed = 0;

    KC_MUTEX_LOCK(&channel->mu);
    if (atomic_load_explicit(&op->state, memory_order_acquire) != KC_OP_REGISTERING) {
        KC_MUTEX_UNLOCK(&channel->mu);
        return -ECANCELED;
    }
    if (channel->closed && !(op->kind == KC_OP_RECV &&
        (channel->count || channel->conflated))) {
        (void)kc_op_claim_locked(op, KC_CAUSE_CLOSE,
                                 &(kc_payload){ .status = -EPIPE });
        KC_MUTEX_UNLOCK(&channel->mu);
        kc_op_publish(op);
        return 0;
    }

    if (op->kind == KC_OP_SEND) {
        paired = queue_pop(&channel->recv_head, &channel->recv_tail,
                           &channel->receive_waiters);
        if (paired) {
            completed = match_pair(op, paired);
        } else if (channel->kind == KC_CHANNEL_RENDEZVOUS) {
            queue_push(&channel->send_head, &channel->send_tail,
                       &channel->send_waiters, op);
        } else if (channel->kind == KC_CHANNEL_CONFLATED) {
            released = channel->conflated;
            channel->conflated = op->descriptor;
            op->descriptor = NULL;
            completed = kc_op_claim_locked(op, KC_CAUSE_MATCH,
                                           &(kc_payload){ .status = 0 });
        } else {
            if (channel->count == channel->capacity &&
                channel->kind == KC_CHANNEL_UNLIMITED && ring_grow(channel) != 0) {
                completed = kc_op_claim_locked(op, KC_CAUSE_FAILURE,
                                               &(kc_payload){ .status = -ENOMEM });
            } else if (channel->count == channel->capacity) {
                queue_push(&channel->send_head, &channel->send_tail,
                           &channel->send_waiters, op);
            } else {
                size_t index = (channel->head + channel->count) % channel->capacity;
                channel->ring[index] = op->descriptor;
                channel->count++;
                op->descriptor = NULL;
                completed = kc_op_claim_locked(op, KC_CAUSE_MATCH,
                                               &(kc_payload){ .status = 0 });
            }
        }
    } else if (op->kind == KC_OP_RECV) {
        kc_descriptor_t *descriptor = NULL;
        if (channel->kind == KC_CHANNEL_CONFLATED && channel->conflated) {
            descriptor = channel->conflated;
            channel->conflated = NULL;
        } else if (channel->count) {
            descriptor = channel->ring[channel->head];
            channel->ring[channel->head] = NULL;
            channel->head = (channel->head + 1) % channel->capacity;
            channel->count--;
        }
        if (descriptor) {
            kc_payload payload = {0};
            if (kc_descriptor_payload(descriptor, &payload) == 0) {
                completed = kc_op_claim_locked(op, KC_CAUSE_MATCH, &payload);
                if (completed) op->result_descriptor = descriptor;
                promoted = queue_pop(&channel->send_head, &channel->send_tail,
                                     &channel->send_waiters);
                if (promoted) {
                    size_t index = (channel->head + channel->count) % channel->capacity;
                    channel->ring[index] = promoted->descriptor;
                    channel->count++;
                    promoted->descriptor = NULL;
                    promoted_completed = kc_op_claim_locked(
                        promoted, KC_CAUSE_MATCH, &(kc_payload){ .status = 0 });
                }
            } else {
                kc_descriptor_release(descriptor);
                (void)kc_op_claim_locked(op, KC_CAUSE_FAILURE,
                                         &(kc_payload){ .status = -EPIPE });
            }
        } else {
            paired = queue_pop(&channel->send_head, &channel->send_tail,
                               &channel->send_waiters);
            if (paired) completed = match_pair(paired, op);
            else if (channel->closed) {
                completed = kc_op_claim_locked(op, KC_CAUSE_CLOSE,
                                               &(kc_payload){ .status = -EPIPE });
            } else queue_push(&channel->recv_head, &channel->recv_tail,
                              &channel->receive_waiters, op);
        }
    } else {
        completed = kc_op_claim_locked(op, KC_CAUSE_FAILURE,
                                       &(kc_payload){ .status = -EINVAL });
    }
    KC_MUTEX_UNLOCK(&channel->mu);

    kc_descriptor_release(released);
    if (completed) kc_op_publish(op);
    if (paired) {
        if (completed) kc_op_publish(paired);
        kc_op_release(paired);
    }
    if (promoted) {
        if (promoted_completed) kc_op_publish(promoted);
        kc_op_release(promoted);
    }
    return 0;
}

int kc_channel_cancel_op(kc_op *op, kc_op_cause cause)
{
    if (!op || !op->channel) return -EINVAL;
    struct kc_chan *channel = op->channel;
    int queued = 0;
    int won;
    KC_MUTEX_LOCK(&channel->mu);
    if (op->linked) {
        if (op->kind == KC_OP_SEND) queue_remove(&channel->send_head,
                                                 &channel->send_tail,
                                                 &channel->send_waiters, op);
        else queue_remove(&channel->recv_head, &channel->recv_tail,
                          &channel->receive_waiters, op);
        queued = 1;
    }
    int status = cause == KC_CAUSE_CLOSE ? -EPIPE
        : cause == KC_CAUSE_TIMEOUT ? -ETIMEDOUT
        : cause == KC_CAUSE_FAILURE ? -EIO : -ECANCELED;
    won = kc_op_claim_locked(op, cause, &(kc_payload){ .status = status });
    KC_MUTEX_UNLOCK(&channel->mu);
    if (won) kc_op_publish(op);
    if (queued) kc_op_release(op);
    return won ? 0 : -EALREADY;
}

void kc_channel_close(kc_channel_t *channel)
{
    if (!channel) return;
    KC_MUTEX_LOCK(&channel->mu);
    if (channel->closed) { KC_MUTEX_UNLOCK(&channel->mu); return; }
    channel->closed = 1;
    KC_MUTEX_UNLOCK(&channel->mu);
    for (;;) {
        KC_MUTEX_LOCK(&channel->mu);
        kc_op *op = queue_pop(&channel->send_head, &channel->send_tail,
                              &channel->send_waiters);
        if (op) (void)kc_op_claim_locked(op, KC_CAUSE_CLOSE,
                                         &(kc_payload){ .status = -EPIPE });
        KC_MUTEX_UNLOCK(&channel->mu);
        if (!op) break;
        kc_op_publish(op);
        kc_op_release(op);
    }
    for (;;) {
        KC_MUTEX_LOCK(&channel->mu);
        kc_op *op = queue_pop(&channel->recv_head, &channel->recv_tail,
                              &channel->receive_waiters);
        if (op) (void)kc_op_claim_locked(op, KC_CAUSE_CLOSE,
                                         &(kc_payload){ .status = -EPIPE });
        KC_MUTEX_UNLOCK(&channel->mu);
        if (!op) break;
        kc_op_publish(op);
        kc_op_release(op);
    }
}

int kc_chan_make(kc_chan_t **out, int kind, size_t element_size, size_t capacity)
{
    kc_channel_config config = {
        .size = sizeof(config), .abi_version = KC_ABI_VERSION,
        .kind = (kc_channel_kind)kind, .element_size = element_size,
        .capacity = capacity,
    };
    return kc_channel_create(kc_runtime_default_get(), &config,
                             (kc_channel_t **)out);
}

void kc_chan_destroy(kc_chan_t *channel)
{
    kc_channel_close((kc_channel_t *)channel);
    kc_channel_release((kc_channel_t *)channel);
}

void kc_chan_close(kc_chan_t *channel) { kc_channel_close((kc_channel_t *)channel); }
unsigned kc_chan_len(kc_chan_t *channel) { return (unsigned)kc_channel_length((kc_channel_t *)channel); }
