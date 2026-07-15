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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FLASHKERN_ROPE_H */
