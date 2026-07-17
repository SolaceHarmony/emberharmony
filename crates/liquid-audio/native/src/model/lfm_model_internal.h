#ifndef LFM_MODEL_INTERNAL_H
#define LFM_MODEL_INTERNAL_H

#include "lfm_model_legacy.h"

#include <stddef.h>
#include <stdint.h>

enum LfmNativeEmissionKind : uint32_t {
    LFM_NATIVE_EMISSION_NONE = 0,
    LFM_NATIVE_EMISSION_TEXT = 1,
    LFM_NATIVE_EMISSION_AUDIO_CODES = 2,
    LFM_NATIVE_EMISSION_FINISHED = 3,
};

struct LfmNativeEmission {
    uint32_t kind;
    uint32_t text_bytes;
    uint32_t code_count;
    uint32_t flags;
    uint64_t position;
    uint8_t text[512];
    uint32_t codes[LFM_AUDIO_TOKEN_CAPACITY];
};

/* Private, pointer-free window state shared with focused native tests. `position`
 * is the live cache length; `cursor` is the monotonic number of committed model
 * passes. `start` is the physical row offset inside capacity+runway storage and
 * `rope_base` is the absolute position represented by logical row zero. */
struct LfmContextWindowState {
    uint64_t capacity;
    uint64_t runway;
    uint64_t position;
    uint64_t start;
    uint64_t cursor;
    uint64_t rope_base;
};

struct LfmContextWindowMove {
    uint64_t dropped;
    uint64_t source;
    uint64_t retained;
    uint32_t compact;
    uint32_t reserved;
};

/* Private pointer-free admission plan used by the mixed-turn implementation
 * and focused native tests. Offsets describe prefix -> text -> audio ->
 * assistant ordering in logical context rows. */
struct LfmMixedTurnPlan {
    size_t text_offset;
    size_t audio_offset;
    size_t assistant_offset;
    size_t total;
};

extern "C" int lfm_context_window_reserve(LfmContextWindowState *window,
                                           size_t needed,
                                           LfmContextWindowMove *move);
extern "C" int lfm_context_window_commit(LfmContextWindowState *window);
extern "C" int lfm_context_compact_bf16(uint16_t *plane, size_t heads,
                                         size_t head_stride, size_t head_dim,
                                         size_t source_row,
                                         size_t retained_rows);
extern "C" int lfm_mixed_turn_plan(size_t capacity, size_t prefix_tokens,
                                     size_t text_tokens, size_t audio_rows,
                                     size_t assistant_tokens,
                                     LfmMixedTurnPlan *out);

/* Private session/model seam. No declaration in the product or Rust ABI. */
int lfm_conversation_prepare_pcm_native(LfmConversation *conversation,
                                        size_t max_sample_count,
                                        uint32_t sample_rate);
int lfm_conversation_begin_pcm_native(LfmConversation *conversation,
                                      const float *pcm, size_t sample_count,
                                      uint32_t sample_rate,
                                      LfmNativeEmission *out);
int lfm_conversation_begin_text_native(LfmConversation *conversation,
                                       const char *text, size_t text_bytes,
                                       LfmNativeEmission *out);
int lfm_conversation_begin_mixed_native(LfmConversation *conversation,
                                        const char *text, size_t text_bytes,
                                        const float *pcm, size_t sample_count,
                                        uint32_t sample_rate,
                                        LfmNativeEmission *out);
int lfm_conversation_next_native(LfmConversation *conversation,
                                 LfmNativeEmission *out);
int lfm_conversation_interrupt_native(LfmConversation *conversation);
int lfm_conversation_decode_native(LfmConversation *conversation,
                                   const uint32_t *codes, size_t code_count,
                                   float *pcm, size_t pcm_capacity,
                                   size_t *out_samples);
int lfm_conversation_belongs_to(const LfmConversation *conversation,
                                const LfmModel *model);

#endif /* LFM_MODEL_INTERNAL_H */
