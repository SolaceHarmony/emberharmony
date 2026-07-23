#ifndef LFM_FLASHKERN_MATH_H
#define LFM_FLASHKERN_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture assembly leaves. C++ may route operands but never evaluate them. */
float lfm_rsqrt_size(size_t value);
float lfm_inv_rms_f32(float sum, size_t count, float epsilon);
float lfm_sum_f32(const float *values, size_t count);
float lfm_bf16_sumsq_stride_f32(const unsigned short *values, size_t count,
                                size_t start, size_t stride);
void lfm_bf16_bias_add_f32(float *values, const void *bias_bytes,
                           size_t count);
void lfm_bf16_copy_bytes(const void *source_bytes, unsigned short *destination,
                         size_t count);
/* One unaligned checkpoint word to the exact f32 bit pattern used by scalar
 * tails. The value stays integer bits across the ABI; no f32 weight plane is
 * ever materialized. */
unsigned int lfm_bf16_unlift_bits(const void *source_bytes);
void lfm_bf16_rope_neox(unsigned short *values, const unsigned short *cosine,
                        const unsigned short *sine, size_t head_dim);

#ifdef __cplusplus
}
#endif

#endif
