// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_IDENTITY_H
#define KC_IDENTITY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Correlation and trace identity carried by exact tickets. */
typedef struct kc_id {
    uint64_t epoch;
    uint64_t sequence;
} kc_id;

/* Canonical ticket identity shared by kcoro, Flashkern, Rust, and product ABI. */
typedef struct kc_ticket_id {
    uint64_t runtime_epoch;
    uint64_t sequence;
    uint32_t generation;
    uint32_t kind;
} kc_ticket_id;

#define KC_TICKET_KIND_SESSION 1u
#define KC_TICKET_KIND_TURN 2u
#define KC_TICKET_KIND_FRAME 3u
#define KC_TICKET_KIND_PASS 4u
#define KC_TICKET_KIND_CONTEXT_SWITCH 5u
#define KC_TICKET_KIND_CHECKPOINT 6u
#define KC_TICKET_KIND_WORKFLOW 7u
#define KC_TICKET_KIND_CONTROL 8u
#define KC_TICKET_KIND_DEADLINE 9u

#ifdef __cplusplus
}
#endif

#endif
