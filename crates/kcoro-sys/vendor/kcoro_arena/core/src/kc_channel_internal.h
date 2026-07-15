// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_channel.h"
#include "kc_op_internal.h"
#include "kcoro_port.h"

#include <stdatomic.h>

struct kc_chan {
    atomic_uint refs;
    kc_runtime_t *runtime;
    KC_MUTEX_T mu;
    kc_id id;
    kc_channel_kind kind;
    size_t element_size;
    size_t capacity;
    kc_descriptor_t **ring;
    size_t head;
    size_t count;
    kc_descriptor_t *conflated;
    kc_op *send_head;
    kc_op *send_tail;
    kc_op *recv_head;
    kc_op *recv_tail;
    size_t send_waiters;
    size_t receive_waiters;
    struct kc_chan *registry_prev;
    struct kc_chan *registry_next;
    int closed;
};

int kc_channel_submit(kc_channel_t *channel, kc_op *op);
int kc_channel_cancel_op(kc_op *op, kc_op_cause cause);
