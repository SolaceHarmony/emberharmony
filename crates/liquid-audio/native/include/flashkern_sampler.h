#ifndef FLASHKERN_SAMPLER_H
#define FLASHKERN_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define LFM_SAMPLE_STATIC_ASSERT(test, message) static_assert(test, message)
#else
#define LFM_SAMPLE_STATIC_ASSERT(test, message) _Static_assert(test, message)
#endif

#define LFM_SAMPLE_ABI_VERSION 1u
#define LFM_SAMPLE_FLAG_GREEDY 1u

/* Policy is an inline control record. Logits, probability scratch, and the
 * conversation PRNG remain pointer-referenced native planes. */
typedef struct LfmSamplerConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t top_k;
    double temperature;
    uint64_t reserved;
} LfmSamplerConfigV1;

/* Architecture leaves. The engine divides the vocabulary into contiguous
 * lane-owned bands and calls these over disjoint slices between generation
 * fences. `scale` is the f32 form of Candle's affine 1/temperature factor;
 * `bf16_scale` is that same factor rounded to bf16 for bf16 input parity. */
uint32_t lfm_sampler_argmax_f32(const float *x, size_t count);
uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count);
float lfm_sampler_exp_sum_f32(const float *x, float *weights, size_t count,
                              float scale, float maximum, float threshold);
float lfm_sampler_exp_sum_bf16(const uint16_t *x, float *weights, size_t count,
                               uint16_t bf16_scale, float maximum,
                               float threshold);
uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count,
                                 float target);

LFM_SAMPLE_STATIC_ASSERT(sizeof(LfmSamplerConfigV1) == 32,
                         "LfmSamplerConfigV1 ABI changed");

#undef LFM_SAMPLE_STATIC_ASSERT

#ifdef __cplusplus
}
#endif

#endif /* FLASHKERN_SAMPLER_H */
