// SPDX-License-Identifier: BSD-3-Clause
#include "kc_transport.h"
#include "kcoro_port.h"

#include <errno.h>
#include <stdlib.h>

typedef struct delivery_entry {
    kc_id message_id;
    struct delivery_entry *next;
} delivery_entry;

struct kc_delivery {
    KC_MUTEX_T mu;
    kc_durable_t *durable;
    kc_transport_handle *transport;
    delivery_entry *in_flight;
    kc_id connection_id;
    uint64_t route;
    size_t in_flight_count;
    uint64_t sends;
    uint64_t acknowledgements;
    uint64_t reconnects;
    uint64_t backpressure;
    uint64_t failures;
    int closed;
    int flushed;
};

static int id_equal(kc_id left, kc_id right)
{
    return left.epoch == right.epoch && left.sequence == right.sequence;
}

static delivery_entry **entry_find(kc_delivery_t *delivery, kc_id message_id)
{
    delivery_entry **link = &delivery->in_flight;
    while (*link && !id_equal((*link)->message_id, message_id)) {
        link = &(*link)->next;
    }
    return link;
}

static int requeue_all(kc_delivery_t *delivery)
{
    int result = 0;
    delivery_entry **link = &delivery->in_flight;
    while (*link) {
        delivery_entry *entry = *link;
        int rc = kc_durable_retry(delivery->durable, entry->message_id);
        if (rc != 0) {
            if (!result) result = rc;
            link = &entry->next;
            continue;
        }
        *link = entry->next;
        delivery->in_flight_count--;
        free(entry);
    }
    return result;
}

int kc_delivery_create(const kc_delivery_config *config, kc_delivery_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->durable ||
        !config->transport) return -EINVAL;
    kc_delivery_t *delivery = calloc(1, sizeof(*delivery));
    if (!delivery) return -ENOMEM;
    if (KC_MUTEX_INIT(&delivery->mu) != 0) { free(delivery); return -ENOMEM; }
    delivery->durable = config->durable;
    delivery->transport = config->transport;
    delivery->route = config->route;
    int rc = kc_transport_connection(delivery->transport,
                                     &delivery->connection_id);
    if (rc != 0) {
        KC_MUTEX_DESTROY(&delivery->mu);
        free(delivery);
        return rc;
    }
    *out = delivery;
    return 0;
}

int kc_delivery_send_next(kc_delivery_t *delivery, kc_id *message_id)
{
    if (!delivery || !message_id) return -EINVAL;
    KC_MUTEX_LOCK(&delivery->mu);
    if (delivery->closed) { KC_MUTEX_UNLOCK(&delivery->mu); return -EPIPE; }
    int ready = kc_transport_ready(delivery->transport);
    if (ready <= 0) {
        if (!ready) delivery->backpressure++;
        else delivery->failures++;
        KC_MUTEX_UNLOCK(&delivery->mu);
        return ready < 0 ? ready : -EAGAIN;
    }
    delivery_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) { KC_MUTEX_UNLOCK(&delivery->mu); return -ENOMEM; }
    kc_message message = {
        .size = sizeof(message), .abi_version = KC_ABI_VERSION,
    };
    int rc = kc_durable_next(delivery->durable, delivery->route, &message);
    if (rc != 0) {
        free(entry);
        KC_MUTEX_UNLOCK(&delivery->mu);
        return rc;
    }
    kc_id connection = {0};
    rc = kc_transport_connection(delivery->transport, &connection);
    if (rc == 0) {
        kc_transport_frame frame = {
            .size = sizeof(frame), .abi_version = KC_ABI_VERSION,
            .connection_id = connection, .message_id = message.id,
            .correlation_id = message.correlation_id,
            .trace_id = message.trace_id, .route = message.route,
            .delivery_attempt = message.delivery_attempt,
            .payload = message.payload, .payload_length = message.payload_length,
        };
        rc = kc_transport_send(delivery->transport, &frame);
    }
    if (rc != 0) {
        int retry = kc_durable_retry(delivery->durable, message.id);
        if (rc == -EAGAIN) delivery->backpressure++;
        else delivery->failures++;
        free(entry);
        KC_MUTEX_UNLOCK(&delivery->mu);
        return retry != 0 ? retry : rc;
    }
    entry->message_id = message.id;
    entry->next = delivery->in_flight;
    delivery->in_flight = entry;
    delivery->in_flight_count++;
    delivery->connection_id = connection;
    delivery->sends++;
    *message_id = message.id;
    KC_MUTEX_UNLOCK(&delivery->mu);
    return 0;
}

int kc_delivery_poll(kc_delivery_t *delivery, kc_transport_event *event)
{
    if (!delivery || !event || event->size < sizeof(*event)) return -EINVAL;
    KC_MUTEX_LOCK(&delivery->mu);
    kc_transport_event received = {
        .size = sizeof(received), .abi_version = KC_ABI_VERSION,
    };
    int rc = kc_transport_next_event(delivery->transport, &received);
    if (rc != 0) { KC_MUTEX_UNLOCK(&delivery->mu); return rc; }
    if (received.size < sizeof(received) ||
        received.abi_version != KC_ABI_VERSION ||
        received.kind < KC_TRANSPORT_WRITABLE ||
        received.kind > KC_TRANSPORT_CLOSED) {
        KC_MUTEX_UNLOCK(&delivery->mu);
        return -EBADMSG;
    }
    if (received.kind == KC_TRANSPORT_ACKNOWLEDGED) {
        delivery_entry **link = entry_find(delivery, received.message_id);
        if (!*link) rc = -ENOENT;
        else {
            rc = received.status == 0
                ? kc_durable_acknowledge(delivery->durable,
                                         received.message_id)
                : kc_durable_retry(delivery->durable,
                                   received.message_id);
            if (rc == 0) {
                delivery_entry *entry = *link;
                *link = entry->next;
                delivery->in_flight_count--;
                if (received.status == 0) delivery->acknowledgements++;
                else delivery->failures++;
                free(entry);
            }
        }
    } else if (received.kind == KC_TRANSPORT_RECONNECTED ||
               received.kind == KC_TRANSPORT_CLOSED) {
        rc = requeue_all(delivery);
        if (received.kind == KC_TRANSPORT_RECONNECTED) {
            delivery->connection_id = received.connection_id;
            delivery->reconnects++;
        }
    }
    if (rc == 0) *event = received;
    KC_MUTEX_UNLOCK(&delivery->mu);
    return rc;
}

int kc_delivery_close(kc_delivery_t *delivery)
{
    if (!delivery) return -EINVAL;
    KC_MUTEX_LOCK(&delivery->mu);
    if (delivery->closed && !delivery->in_flight && delivery->flushed) {
        KC_MUTEX_UNLOCK(&delivery->mu);
        return 0;
    }
    delivery->closed = 1;
    int rc = requeue_all(delivery);
    int flush = delivery->flushed ? 0
        : kc_transport_flush(delivery->transport);
    if (flush == 0) delivery->flushed = 1;
    KC_MUTEX_UNLOCK(&delivery->mu);
    return rc != 0 ? rc : flush;
}

void kc_delivery_destroy(kc_delivery_t *delivery)
{
    if (!delivery) return;
    (void)kc_delivery_close(delivery);
    delivery_entry *entry = delivery->in_flight;
    while (entry) {
        delivery_entry *next = entry->next;
        free(entry);
        entry = next;
    }
    KC_MUTEX_DESTROY(&delivery->mu);
    free(delivery);
}

int kc_delivery_snapshot_get(kc_delivery_t *delivery, kc_delivery_snapshot *out)
{
    if (!delivery || !out || out->size < sizeof(*out)) return -EINVAL;
    KC_MUTEX_LOCK(&delivery->mu);
    *out = (kc_delivery_snapshot){
        .size = sizeof(*out), .abi_version = KC_ABI_VERSION,
        .connection_id = delivery->connection_id, .route = delivery->route,
        .in_flight = delivery->in_flight_count, .sends = delivery->sends,
        .acknowledgements = delivery->acknowledgements,
        .reconnects = delivery->reconnects,
        .backpressure = delivery->backpressure,
        .failures = delivery->failures,
    };
    KC_MUTEX_UNLOCK(&delivery->mu);
    return 0;
}
