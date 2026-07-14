// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_PAYLOAD_H
#define KC_PAYLOAD_H

#include <stddef.h>
#include <stdint.h>

typedef struct kc_payload {
    void *ptr;
    size_t len;
    int status;
    uint64_t desc_id;
} kc_payload;

#endif
