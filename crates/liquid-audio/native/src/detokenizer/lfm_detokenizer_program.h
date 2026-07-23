#ifndef LFM_DETOKENIZER_PROGRAM_H
#define LFM_DETOKENIZER_PROGRAM_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_detokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private Flashkern program cursor.  The resident weights and conversation
 * state remain owned by the detokenizer; this record is only the durable
 * continuation frame carried by one admitted numerical ticket. */
enum {
    LFM_DETOKENIZER_PHASE_EMBED = 0,
    LFM_DETOKENIZER_PHASE_OPERATOR_NORM = 1,
    LFM_DETOKENIZER_PHASE_OPERATOR_PROJECT = 2,
    LFM_DETOKENIZER_PHASE_OPERATOR_MIX = 3,
    LFM_DETOKENIZER_PHASE_OPERATOR_OUT = 4,
    LFM_DETOKENIZER_PHASE_OPERATOR_RESIDUAL_NORM = 5,
    LFM_DETOKENIZER_PHASE_FFN_PROJECT = 6,
    LFM_DETOKENIZER_PHASE_FFN_ACTIVATE = 7,
    LFM_DETOKENIZER_PHASE_FFN_DOWN = 8,
    LFM_DETOKENIZER_PHASE_FFN_RESIDUAL_NORM = 9,
    LFM_DETOKENIZER_PHASE_FINAL_PROJECT = 10,
    LFM_DETOKENIZER_PHASE_IFFT = 11,
    LFM_DETOKENIZER_PHASE_OVERLAP_EMIT = 12,
    LFM_DETOKENIZER_PHASE_EMIT = 13,
    LFM_DETOKENIZER_PHASE_DONE = 14,
};

typedef struct LfmAudioDetokenizerProgram {
    LfmAudioDetokenizerState *state;
    float *pcm;
    size_t pcm_capacity;
    size_t produced;
    uint64_t emit_end;
    uint32_t codes[LFM_DETOKENIZER_CODEBOOKS];
    uint32_t phase;
    uint32_t layer;
    uint32_t flush;
    uint32_t active;
} LfmAudioDetokenizerProgram;

/* Begin validates and retains one state transition.  No allocation occurs.
 * `run` is called once by every fixed-team member for the current phase.
 * Only the final-return callback may call `advance`; 1 requests the next team
 * generation, 0 is terminal, and a negative value is a numerical failure. */
int lfm_detokenizer_program_begin(
    LfmAudioDetokenizerProgram *program, LfmAudioDetokenizerState *state,
    const uint32_t codes[LFM_DETOKENIZER_CODEBOOKS], float *pcm,
    size_t pcm_capacity, uint32_t flush);
int lfm_detokenizer_program_run(LfmAudioDetokenizerProgram *program,
                                uint32_t lane, uint32_t lanes);
int lfm_detokenizer_program_advance(LfmAudioDetokenizerProgram *program);
void lfm_detokenizer_program_cancel(LfmAudioDetokenizerProgram *program);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_DETOKENIZER_PROGRAM_H */
