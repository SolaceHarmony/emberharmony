// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_TRANSPORT_H
#define KC_TRANSPORT_H

#include "kc_durable.h"
#include "kc_shared.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_transport_handle kc_transport_handle;
typedef struct kc_delivery kc_delivery_t;

typedef enum kc_transport_event_kind {
    KC_TRANSPORT_WRITABLE = 1,
    KC_TRANSPORT_ACKNOWLEDGED,
    KC_TRANSPORT_RECONNECTED,
    KC_TRANSPORT_CLOSED,
} kc_transport_event_kind;

typedef struct kc_transport_frame {
    uint32_t size;
    uint32_t abi_version;
    kc_id connection_id;
    kc_id message_id;
    kc_id correlation_id;
    kc_id trace_id;
    uint64_t route;
    uint32_t delivery_attempt;
    uint32_t flags;
    const void *payload;
    size_t payload_length;
} kc_transport_frame;

typedef struct kc_transport_event {
    uint32_t size;
    uint32_t abi_version;
    kc_transport_event_kind kind;
    uint32_t reserved;
    kc_id connection_id;
    kc_id message_id;
    int status;
} kc_transport_event;

/* Direct link-time host contract. Framing, readiness, and reconnect events are
 * supplied by the adapter; the core ships no OS transport. */
int kc_transport_connection(kc_transport_handle *transport, kc_id *connection);
int kc_transport_ready(kc_transport_handle *transport);
int kc_transport_send(kc_transport_handle *transport,
                      const kc_transport_frame *frame);
int kc_transport_next_event(kc_transport_handle *transport,
                            kc_transport_event *event);
int kc_transport_flush(kc_transport_handle *transport);

typedef struct kc_delivery_config {
    uint32_t size;
    uint32_t abi_version;
    kc_durable_t *durable;
    kc_transport_handle *transport;
    uint64_t route;
} kc_delivery_config;

typedef struct kc_delivery_snapshot {
    uint32_t size;
    uint32_t abi_version;
    kc_id connection_id;
    uint64_t route;
    size_t in_flight;
    uint64_t sends;
    uint64_t acknowledgements;
    uint64_t reconnects;
    uint64_t backpressure;
    uint64_t failures;
} kc_delivery_snapshot;

int kc_delivery_create(const kc_delivery_config *config, kc_delivery_t **out);
int kc_delivery_send_next(kc_delivery_t *delivery, kc_id *message_id);
int kc_delivery_poll(kc_delivery_t *delivery, kc_transport_event *event);
int kc_delivery_close(kc_delivery_t *delivery);
void kc_delivery_destroy(kc_delivery_t *delivery);
int kc_delivery_snapshot_get(kc_delivery_t *delivery, kc_delivery_snapshot *out);

#ifdef __cplusplus
}
#endif

#endif
