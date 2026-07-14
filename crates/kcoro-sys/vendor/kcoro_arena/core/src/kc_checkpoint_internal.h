// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stddef.h>
#include <stdint.h>

enum {
    KC_CHECKPOINT_DURABLE = 1,
    KC_CHECKPOINT_WORKFLOWS = 2,
};

typedef struct kc_checkpoint_section {
    uint32_t type;
    const void *payload;
    size_t payload_length;
} kc_checkpoint_section;

int kc_checkpoint_encode(const kc_checkpoint_section *sections, size_t count,
                         void **payload, size_t *payload_length);
int kc_checkpoint_find(const void *payload, size_t payload_length, uint32_t type,
                       const void **section, size_t *section_length);
