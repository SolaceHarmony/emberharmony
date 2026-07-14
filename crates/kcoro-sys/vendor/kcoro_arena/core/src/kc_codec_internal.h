// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stdint.h>

static inline uint16_t kc_get_u16(const unsigned char *data)
{
    return (uint16_t)data[0] | (uint16_t)((uint16_t)data[1] << 8);
}

static inline uint32_t kc_get_u32(const unsigned char *data)
{
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static inline uint64_t kc_get_u64(const unsigned char *data)
{
    return (uint64_t)kc_get_u32(data) |
           ((uint64_t)kc_get_u32(data + 4) << 32);
}

static inline int32_t kc_get_i32(const unsigned char *data)
{
    uint32_t value = kc_get_u32(data);
    if (value <= INT32_MAX) return (int32_t)value;
    return -1 - (int32_t)(UINT32_MAX - value);
}

static inline void kc_put_u16(unsigned char *data, uint16_t value)
{
    data[0] = (unsigned char)value;
    data[1] = (unsigned char)(value >> 8);
}

static inline void kc_put_u32(unsigned char *data, uint32_t value)
{
    data[0] = (unsigned char)value;
    data[1] = (unsigned char)(value >> 8);
    data[2] = (unsigned char)(value >> 16);
    data[3] = (unsigned char)(value >> 24);
}

static inline void kc_put_u64(unsigned char *data, uint64_t value)
{
    kc_put_u32(data, (uint32_t)value);
    kc_put_u32(data + 4, (uint32_t)(value >> 32));
}

static inline void kc_put_i32(unsigned char *data, int32_t value)
{
    kc_put_u32(data, (uint32_t)value);
}
