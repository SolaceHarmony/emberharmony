// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_SELECT_H
#define KC_SELECT_H

#include "kc_cancel.h"
#include "kc_channel.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kc_select_clause_kind {
    KC_SELECT_RECV = 1,
    KC_SELECT_SEND = 2,
    KC_SELECT_DEFAULT = 3,
} kc_select_clause_kind;

typedef struct kc_select_clause {
    uint32_t size;
    uint32_t abi_version;
    kc_select_clause_kind kind;
    kc_channel_t *channel;
    const void *data;
    size_t length;
} kc_select_clause;

#ifdef __cplusplus
}
#endif

#endif
