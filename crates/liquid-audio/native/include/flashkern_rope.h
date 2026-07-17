#ifndef FLASHKERN_ROPE_H
#define FLASHKERN_ROPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture leaf for interleaved rotary tables. The caller owns aligned
 * output planes of positions * (head_dim / 2) floats. */
int lfm_rope_table_f32(size_t positions, size_t head_dim, float theta,
                       float *cosine, float *sine);

/* Build a contiguous absolute-position range. This is the sliding-context
 * companion to the zero-based table leaf above: retained KV rows keep their
 * original rotary phase, while newly exposed runway rows are generated for
 * monotonically increasing positions. */
int lfm_rope_range_f32(uint64_t first_position, size_t positions,
                       size_t head_dim, float theta, float *cosine,
                       float *sine);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FLASHKERN_ROPE_H */
