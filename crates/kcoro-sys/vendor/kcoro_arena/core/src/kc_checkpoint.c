// SPDX-License-Identifier: BSD-3-Clause
#include "kc_checkpoint_internal.h"
#include "kc_codec_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

enum {
    CHECKPOINT_VERSION = 1,
    CHECKPOINT_HEADER_SIZE = 16,
    CHECKPOINT_SECTION_SIZE = 16,
};

#define CHECKPOINT_MAGIC UINT32_C(0x5043434b)

int kc_checkpoint_encode(const kc_checkpoint_section *sections, size_t count,
                         void **payload, size_t *payload_length)
{
    if (!sections || !count || !payload || !payload_length || count > UINT32_MAX) {
        return -EINVAL;
    }
    size_t total = CHECKPOINT_HEADER_SIZE;
    for (size_t index = 0; index < count; index++) {
        if (!sections[index].type ||
            (sections[index].payload_length && !sections[index].payload) ||
            sections[index].payload_length > UINT32_MAX ||
            total > SIZE_MAX - CHECKPOINT_SECTION_SIZE -
                    sections[index].payload_length) return -E2BIG;
        for (size_t prior = 0; prior < index; prior++) {
            if (sections[prior].type == sections[index].type) return -EINVAL;
        }
        total += CHECKPOINT_SECTION_SIZE + sections[index].payload_length;
    }
    unsigned char *data = calloc(1, total);
    if (!data) return -ENOMEM;
    kc_put_u32(data, CHECKPOINT_MAGIC);
    kc_put_u16(data + 4, CHECKPOINT_VERSION);
    kc_put_u16(data + 6, CHECKPOINT_HEADER_SIZE);
    kc_put_u32(data + 8, (uint32_t)count);
    size_t offset = CHECKPOINT_HEADER_SIZE;
    for (size_t index = 0; index < count; index++) {
        kc_put_u32(data + offset, sections[index].type);
        kc_put_u32(data + offset + 4,
                   (uint32_t)sections[index].payload_length);
        if (sections[index].payload_length) {
            memcpy(data + offset + CHECKPOINT_SECTION_SIZE,
                   sections[index].payload, sections[index].payload_length);
        }
        offset += CHECKPOINT_SECTION_SIZE + sections[index].payload_length;
    }
    *payload = data;
    *payload_length = total;
    return 0;
}

int kc_checkpoint_find(const void *payload, size_t payload_length, uint32_t type,
                       const void **section, size_t *section_length)
{
    if (!payload || !section || !section_length || !type) return -EINVAL;
    if (payload_length < CHECKPOINT_HEADER_SIZE) return -EBADMSG;
    const unsigned char *data = payload;
    if (kc_get_u32(data) != CHECKPOINT_MAGIC ||
        kc_get_u16(data + 4) != CHECKPOINT_VERSION ||
        kc_get_u16(data + 6) != CHECKPOINT_HEADER_SIZE) return -EBADMSG;
    uint32_t count = kc_get_u32(data + 8);
    size_t offset = CHECKPOINT_HEADER_SIZE;
    int found = 0;
    for (uint32_t index = 0; index < count; index++) {
        if (payload_length - offset < CHECKPOINT_SECTION_SIZE) return -EBADMSG;
        uint32_t current_type = kc_get_u32(data + offset);
        uint32_t length = kc_get_u32(data + offset + 4);
        if (!current_type || length > payload_length - offset -
                                      CHECKPOINT_SECTION_SIZE) return -EBADMSG;
        if (current_type == type) {
            if (found) return -EBADMSG;
            *section = data + offset + CHECKPOINT_SECTION_SIZE;
            *section_length = length;
            found = 1;
        }
        offset += CHECKPOINT_SECTION_SIZE + length;
    }
    if (offset != payload_length) return -EBADMSG;
    return found ? 0 : -ENOENT;
}
