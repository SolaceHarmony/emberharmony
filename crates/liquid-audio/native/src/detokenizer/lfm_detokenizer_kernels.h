#ifndef LFM_DETOKENIZER_KERNELS_H
#define LFM_DETOKENIZER_KERNELS_H

/* Private buffer-view ABI shared by C++23 orchestration and the paired
 * architecture leaves. These records own no storage. */

#define LFM_DETOK_EMBED_ROWS 0
#define LFM_DETOK_EMBED_OUTPUT 64
#define LFM_DETOK_EMBED_COUNT 72

#define LFM_DETOK_RMS_INPUT 0
#define LFM_DETOK_RMS_WEIGHT 8
#define LFM_DETOK_RMS_OUTPUT 16
#define LFM_DETOK_RMS_COLUMNS 24
#define LFM_DETOK_RMS_EPSILON 32

#define LFM_DETOK_CONV_PROJECTED 0
#define LFM_DETOK_CONV_CARRY 8
#define LFM_DETOK_CONV_WEIGHT 16
#define LFM_DETOK_CONV_OUTPUT 24
#define LFM_DETOK_CONV_COUNT 32

#define LFM_DETOK_WEIGHTED_VALUES 0
#define LFM_DETOK_WEIGHTED_SCORES 8
#define LFM_DETOK_WEIGHTED_OUTPUT 16
#define LFM_DETOK_WEIGHTED_FIRST 24
#define LFM_DETOK_WEIGHTED_COUNT 32
#define LFM_DETOK_WEIGHTED_HEAD 40

#define LFM_DETOK_POLAR_MAGNITUDE 0
#define LFM_DETOK_POLAR_SINE 8
#define LFM_DETOK_POLAR_COSINE 16
#define LFM_DETOK_POLAR_REAL 24
#define LFM_DETOK_POLAR_IMAGINARY 32
#define LFM_DETOK_POLAR_COUNT 40

#define LFM_DETOK_OVERLAP_SIGNAL 0
#define LFM_DETOK_OVERLAP_WINDOW 8
#define LFM_DETOK_OVERLAP_OUTPUT 16
#define LFM_DETOK_OVERLAP_ENVELOPE 24
#define LFM_DETOK_OVERLAP_COUNT 32

#define LFM_DETOK_EMIT_OUTPUT 0
#define LFM_DETOK_EMIT_ENVELOPE 8
#define LFM_DETOK_EMIT_PCM 16
#define LFM_DETOK_EMIT_COUNT 24
#define LFM_DETOK_EMIT_EPSILON 32

#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

typedef struct LfmDetokEmbedArgs {
    const void *rows[8];
    float *output;
    uint64_t count;
} LfmDetokEmbedArgs;

typedef struct LfmDetokRmsArgs {
    const float *input;
    const void *weight;
    float *output;
    uint64_t columns;
    float epsilon;
    uint32_t reserved;
} LfmDetokRmsArgs;

typedef struct LfmDetokConvArgs {
    const float *projected;
    float *carry;
    const void *weight;
    float *output;
    uint64_t count;
} LfmDetokConvArgs;

typedef struct LfmDetokWeightedArgs {
    const float *values;
    const float *scores;
    float *output;
    uint64_t ring_first;
    uint64_t count;
    uint64_t kv_head;
} LfmDetokWeightedArgs;

typedef struct LfmDetokPolarArgs {
    const float *magnitude;
    const float *sine;
    const float *cosine;
    float *real;
    float *imaginary;
    uint64_t count;
} LfmDetokPolarArgs;

typedef struct LfmDetokOverlapArgs {
    const float *signal;
    const void *window;
    float *output;
    float *envelope;
    uint64_t count;
} LfmDetokOverlapArgs;

typedef struct LfmDetokEmitArgs {
    const float *output;
    const float *envelope;
    float *pcm;
    uint64_t count;
    float epsilon;
    uint32_t reserved;
} LfmDetokEmitArgs;

#ifdef __cplusplus
extern "C" {
#endif

void lfm_detok_copy_f32(const float *source, float *destination,
                        uint64_t count);
void lfm_detok_add_f32(float *destination, const float *source,
                       uint64_t count);
void lfm_detok_embed_f32(const LfmDetokEmbedArgs *args);
void lfm_detok_rms_f32(const LfmDetokRmsArgs *args);
void lfm_detok_swiglu_f32(float *gate, const float *up,
                          const float *negative_exp, uint64_t count);
float lfm_detok_dot32_scaled_f32(const float *left, const float *right,
                                 float scale);
float lfm_detok_max_f32(const float *values, uint64_t count);
void lfm_detok_subtract_f32(float *values, uint64_t count, float value);
float lfm_detok_sum_f32(const float *values, uint64_t count);
void lfm_detok_normalize_f32(float *values, uint64_t count, float sum);
void lfm_detok_rope_angles_f32(const float *inverse, float *angles,
                               uint64_t count, uint64_t position);
void lfm_detok_rope_f32(float *values, const float *cosine,
                        const float *sine);
int lfm_detok_rope_inverse_f32(float *inverse, uint64_t count, float theta,
                               uint64_t head);
int lfm_detok_ifft_basis_f32(float *basis, uint64_t bins, uint64_t fft);
void lfm_detok_conv_f32(const LfmDetokConvArgs *args);
void lfm_detok_weighted_f32(const LfmDetokWeightedArgs *args);
void lfm_detok_polar_f32(const LfmDetokPolarArgs *args);
void lfm_detok_overlap_f32(const LfmDetokOverlapArgs *args);
int lfm_detok_emit_f32(const LfmDetokEmitArgs *args);

#ifdef __cplusplus
} /* extern "C" */

static_assert(offsetof(LfmDetokEmbedArgs, output) == LFM_DETOK_EMBED_OUTPUT);
static_assert(offsetof(LfmDetokEmbedArgs, count) == LFM_DETOK_EMBED_COUNT);
static_assert(offsetof(LfmDetokRmsArgs, epsilon) == LFM_DETOK_RMS_EPSILON);
static_assert(offsetof(LfmDetokConvArgs, count) == LFM_DETOK_CONV_COUNT);
static_assert(offsetof(LfmDetokWeightedArgs, kv_head) ==
              LFM_DETOK_WEIGHTED_HEAD);
static_assert(offsetof(LfmDetokPolarArgs, count) == LFM_DETOK_POLAR_COUNT);
static_assert(offsetof(LfmDetokOverlapArgs, count) ==
              LFM_DETOK_OVERLAP_COUNT);
static_assert(offsetof(LfmDetokEmitArgs, epsilon) ==
              LFM_DETOK_EMIT_EPSILON);
#endif

#endif /* !__ASSEMBLER__ */

#endif /* LFM_DETOKENIZER_KERNELS_H */
