#ifndef LFM_DETOKENIZER_H
#define LFM_DETOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmAudioDetokenizerPlan LfmAudioDetokenizerPlan;
typedef struct LfmAudioDetokenizerState LfmAudioDetokenizerState;
typedef struct LfmWeightImage LfmWeightImage;

enum {
    LFM_DETOKENIZER_CODEBOOKS = 8,
    LFM_DETOKENIZER_CODE_VALUES = 2048,
    LFM_DETOKENIZER_SAMPLE_RATE = 24000,
    LFM_DETOKENIZER_FRAME_SAMPLES = 1920,
    LFM_DETOKENIZER_MAX_STEP_SAMPLES = 1920,
};

/* Bind the released LFM2.5 audio_detokenizer component from the model-owned,
 * read-only resident image. The plan retains only byte/shape views plus
 * formula-derived immutable Fourier coefficients; it never widens, aligns,
 * transposes, or otherwise materializes checkpoint weights. */
LFM_ORACLE_API int lfm_detokenizer_plan_new_from_image(
    LfmAudioDetokenizerPlan **out, const LfmWeightImage *image, char *error,
    size_t error_length);
LFM_ORACLE_API void
lfm_detokenizer_plan_free(LfmAudioDetokenizerPlan *plan);
LFM_ORACLE_API uint64_t
lfm_detokenizer_plan_bound_weight_bytes(const LfmAudioDetokenizerPlan *plan);
LFM_ORACLE_API uint64_t
lfm_detokenizer_plan_derived_bytes(const LfmAudioDetokenizerPlan *plan);
LFM_ORACLE_API uint64_t lfm_detokenizer_plan_compatibility_copied_bytes(
    const LfmAudioDetokenizerPlan *plan);

/* Conversation-owned causal state and one preallocated liveness arena. One
 * input code frame produces 1,440 samples on the first call and 1,920 samples
 * thereafter. Flush publishes the final 480 samples removed from the future
 * overlap dependency by the reference's same-padding contract. */
LFM_ORACLE_API int lfm_detokenizer_state_new(
    LfmAudioDetokenizerState **out, const LfmAudioDetokenizerPlan *plan,
    char *error, size_t error_length);
LFM_ORACLE_API void
lfm_detokenizer_state_free(LfmAudioDetokenizerState *state);
LFM_ORACLE_API void
lfm_detokenizer_state_reset(LfmAudioDetokenizerState *state);
LFM_ORACLE_API uint64_t
lfm_detokenizer_state_bytes(const LfmAudioDetokenizerState *state);
LFM_ORACLE_API int lfm_detokenizer_state_step(
    LfmAudioDetokenizerState *state,
    const uint32_t codes[LFM_DETOKENIZER_CODEBOOKS], float *pcm,
    size_t pcm_capacity, size_t *out_samples);
LFM_ORACLE_API int lfm_detokenizer_state_flush(
    LfmAudioDetokenizerState *state, float *pcm, size_t pcm_capacity,
    size_t *out_samples);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_DETOKENIZER_H */
