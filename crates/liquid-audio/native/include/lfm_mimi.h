#ifndef LFM_MIMI_H
#define LFM_MIMI_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmWeightImage LfmWeightImage;
typedef struct MimiDecodePlan MimiDecodePlan;
typedef struct MimiDecodeState MimiDecodeState;

enum {
    LFM_MIMI_CODEBOOKS = 8,
    LFM_MIMI_CODE_VALUES = 2048,
    LFM_MIMI_SAMPLE_RATE = 24000,
    LFM_MIMI_FRAME_SAMPLES = 1920,
    LFM_MIMI_PCM_CAPACITY = 3840,
};

/* Production ownership split: one immutable plan per model image, one mutable
 * state per conversation. The plan must outlive every state created from it. */
LFM_ORACLE_API int mimi_decode_plan_new_from_image(
    MimiDecodePlan **plan, const LfmWeightImage *image, char *error,
    size_t error_length);
LFM_ORACLE_API void mimi_decode_plan_free(MimiDecodePlan *plan);
LFM_ORACLE_API uint64_t
mimi_decode_plan_derived_bytes(const MimiDecodePlan *plan);
LFM_ORACLE_API uint64_t
mimi_decode_plan_bound_weight_bytes(const MimiDecodePlan *plan);
LFM_ORACLE_API uint64_t
mimi_decode_plan_compatibility_copied_bytes(const MimiDecodePlan *plan);

LFM_ORACLE_API int mimi_decode_state_new(MimiDecodeState **state,
                                         const MimiDecodePlan *plan,
                                         char *error, size_t error_length);
LFM_ORACLE_API void mimi_decode_state_free(MimiDecodeState *state);
LFM_ORACLE_API int mimi_decode_state_step(MimiDecodeState *state,
                                          const uint32_t *codes,
                                          float *pcm_out);
LFM_ORACLE_API void mimi_decode_state_reset(MimiDecodeState *state);
LFM_ORACLE_API uint64_t mimi_decode_state_bytes(const MimiDecodeState *state);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MIMI_H */
