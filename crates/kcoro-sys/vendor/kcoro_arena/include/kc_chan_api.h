// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque channel handle */
typedef struct kc_chan kc_chan_t;

/* Channel kinds */
enum kc_kind {
    KC_RENDEZVOUS = 0,
    KC_BUFFERED   = 1,
    KC_CONFLATED  = -1,
    KC_UNLIMITED  = -2
};

/* Lifecycle */
int  kc_chan_make(kc_chan_t **out, int kind, size_t elem_sz, size_t capacity);
void kc_chan_destroy(kc_chan_t *ch);
void kc_chan_close(kc_chan_t *ch);
unsigned kc_chan_len(kc_chan_t *ch);

#ifdef __cplusplus
}
#endif
