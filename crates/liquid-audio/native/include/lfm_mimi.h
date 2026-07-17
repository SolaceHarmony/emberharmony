#ifndef LFM_MIMI_H
#define LFM_MIMI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmWeightImage LfmWeightImage;
typedef struct MimiDecoder MimiDecoder;
typedef struct MimiDecodePlan MimiDecodePlan;
typedef struct MimiDecodeState MimiDecodeState;

enum {
    LFM_MIMI_CODEBOOKS = 8,
    LFM_MIMI_FRAME_SAMPLES = 1920,
    LFM_MIMI_PCM_CAPACITY = 3840,
};

/* Transitional single-state wrapper for parity tests. Production uses the
 * plan/state split below. The caller owns and retains the image. */
int mimi_decoder_new_from_image(MimiDecoder **decoder,
                                const LfmWeightImage *image, char *error,
                                 size_t error_length);

/* Production ownership split: one immutable plan per model image, one mutable
 * state per conversation. The plan must outlive every state created from it. */
int mimi_decode_plan_new_from_image(MimiDecodePlan **plan,
                                    const LfmWeightImage *image, char *error,
                                    size_t error_length);
void mimi_decode_plan_free(MimiDecodePlan *plan);
uint64_t mimi_decode_plan_derived_bytes(const MimiDecodePlan *plan);
uint64_t mimi_decode_plan_compatibility_copied_bytes(const MimiDecodePlan *plan);

int mimi_decode_state_new(MimiDecodeState **state, const MimiDecodePlan *plan,
                          char *error, size_t error_length);
void mimi_decode_state_free(MimiDecodeState *state);
int mimi_decode_state_step(MimiDecodeState *state, const uint32_t *codes,
                           float *pcm_out);
void mimi_decode_state_reset(MimiDecodeState *state);
uint64_t mimi_decode_state_bytes(const MimiDecodeState *state);

/* Transitional from-file parity constructor. */
int mimi_decoder_new_from_file(MimiDecoder **decoder, const char *checkpoint,
                               char *error, size_t error_length);

int mimi_decoder_step(MimiDecoder *decoder, const uint32_t *codes,
                      float *pcm_out);
void mimi_decoder_reset(MimiDecoder *decoder);
void mimi_decoder_free(MimiDecoder *decoder);

uint64_t mimi_decoder_derived_bytes(const MimiDecoder *decoder);
uint64_t mimi_decoder_compatibility_copied_bytes(const MimiDecoder *decoder);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_MIMI_H */
