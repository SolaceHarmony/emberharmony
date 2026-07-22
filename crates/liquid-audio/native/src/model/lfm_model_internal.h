#ifndef LFM_MODEL_INTERNAL_H
#define LFM_MODEL_INTERNAL_H

#include "lfm_model_plan.h"
#include "lfm_runtime.h"
#include "lfm_visibility.h"

#include <stddef.h>
#include <stdint.h>

/* Private native owner ABI. These are lifecycle/configuration records used
 * only between the opaque product runtime and the native model owner. They do
 * not expose numerical operations, activations, or checkpoint views. */
#define LFM_MODEL_CAP_DEPTHFORMER 1u
#define LFM_MODEL_CAP_FRONTEND 2u
#define LFM_MODEL_CAP_CONFORMER 4u
#define LFM_MODEL_CAP_DETOKENIZER 8u
#define LFM_INPUT_MAX_IDS 8u
#define LFM_AUDIO_TOKEN_CAPACITY 64u

typedef struct LfmConversationConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t seed;
    LfmSamplerConfigV1 text_sampler;
    LfmSamplerConfigV1 audio_sampler;
    uint64_t reserved[4];
} LfmConversationConfigV1;

typedef struct LfmModelInfoV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t resident_bytes;
    uint64_t plan_id;
    uint64_t depth_plan_id;
    uint32_t hidden;
    uint32_t ffn;
    uint32_t layers;
    uint32_t vocab;
    uint32_t max_context;
    uint32_t codebooks;
    uint32_t capabilities;
    uint32_t reserved[5];
} LfmModelInfoV1;

#ifdef __cplusplus
extern "C" {
#endif

LFM_INTERNAL_API int lfm_model_open(void *engine, const char *path,
                                  LfmModel **out, char *error,
                                  size_t error_length);
LFM_INTERNAL_API int lfm_model_close(LfmModel *model);
LFM_INTERNAL_API int lfm_model_info(const LfmModel *model,
                                  LfmModelInfoV1 *out);
LFM_INTERNAL_API int lfm_model_memory(const LfmModel *model,
                                    LfmModelMemoryV1 *out);
LFM_INTERNAL_API int lfm_conversation_create(
    LfmModel *model, const LfmConversationConfigV1 *config,
    LfmConversation **out, char *error, size_t error_length);
LFM_INTERNAL_API int lfm_conversation_reset(LfmConversation *conversation);
LFM_INTERNAL_API int lfm_conversation_close(LfmConversation *conversation);

#ifdef __cplusplus
} /* extern "C" */
#endif

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

struct LfmAudioDetokenizerState;

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

/* Private workflow identity for one retained conversation admission. The
 * record is conversation-owned; `ticket` is the first workflow ticket and
 * remains the external correlation identity while child pass tickets advance
 * the state machine. */
struct LfmConversationAdmissionHandle {
    void *record;
    uint64_t generation;
    KcTicketIdV1 ticket;
};

extern "C" int lfm_context_window_reserve(LfmContextWindowState *window,
                                           size_t needed,
                                           LfmContextWindowMove *move);
extern "C" int lfm_context_window_admit(const LfmContextWindowState *window,
                                         size_t needed);
extern "C" int lfm_context_window_prefill_chunk(
    const LfmContextWindowState *window, size_t remaining, size_t max_rows,
    size_t *out_rows);
extern "C" int lfm_context_window_commit(LfmContextWindowState *window);
extern "C" int lfm_context_compact_bf16(uint16_t *plane, size_t heads,
                                         size_t head_stride, size_t head_dim,
                                         size_t source_row,
                                         size_t retained_rows);
extern "C" int lfm_mixed_turn_plan(size_t capacity, size_t prefix_tokens,
                                     size_t text_tokens, size_t audio_rows,
                                     size_t assistant_tokens,
                                     LfmMixedTurnPlan *out);
/* Private publication decision kept testable without exposing codec codes in
 * the product ABI: 1 = decode/publish PCM, 0 = recurrence-only EOAudio. */
extern "C" LFM_INTERNAL_API int lfm_native_emission_needs_pcm(
    const LfmNativeEmission *emission);

/* Test-scoped inner-voice listening probe. Feeds one user audio turn through
 * the production admission prefill seam ONE adapted row per pass, sampling
 * the greedy text head at every row into `out_tokens` and recording per-row
 * wall time into `out_row_ns`. `out_readouts` is optional: when non-null it
 * must hold `row_capacity` records and every row pass also reports top-k ids
 * with natural-log probabilities plus full-distribution entropy for the text
 * head and the Depthformer codebook-0 head (see LfmListenReadoutForTest).
 * Sampled ids and readouts are reported only, never committed; context
 * commits match a production admission over the same prefix and rows.
 * Submit rings `notify` once at terminal; collect returns -EINPROGRESS until
 * then and releases the probe record on any terminal status. */
extern "C" LFM_INTERNAL_API int
lfm_internal_conversation_listen_probe_submit_for_test(
    LfmConversation *conversation, const float *pcm, size_t sample_count,
    uint32_t sample_rate, uint32_t *out_tokens, uint64_t *out_row_ns,
    LfmListenReadoutForTest *out_readouts, size_t row_capacity,
    LfmAudioRouteNotify notify, void *notify_context, void **out_probe);
extern "C" LFM_INTERNAL_API int
lfm_internal_conversation_listen_probe_collect_for_test(
    LfmConversation *conversation, void *probe, uint64_t *out_rows,
    uint64_t *out_encode_ns);

/* Private session/model seam. No declaration in the product or Rust ABI. */
LFM_INTERNAL_API int lfm_conversation_prepare_pcm_native(
    LfmConversation *conversation, size_t max_sample_count,
    uint32_t capture_rate, uint32_t playback_rate,
    size_t *out_playback_frames);
LFM_INTERNAL_API int lfm_conversation_begin_pcm_submit_native(
    LfmConversation *conversation, const float *pcm, size_t sample_count,
    uint32_t sample_rate, LfmNativeEmission *out,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmConversationAdmissionHandle *out_handle);
/* Circular-arena dock seam. `spans` is copied before return; its pointed
 * sample storage remains a retained read-only lease until admission collect. */
LFM_INTERNAL_API int lfm_conversation_begin_pcm_spans_submit_native(
    LfmConversation *conversation, const LfmF32Span *spans,
    uint32_t span_count, uint32_t sample_rate, LfmNativeEmission *out,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmConversationAdmissionHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_begin_text_submit_native(
    LfmConversation *conversation, const char *text, size_t text_bytes,
    LfmNativeEmission *out, LfmAudioRouteNotify notify,
    void *notify_context, LfmConversationAdmissionHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_begin_mixed_submit_native(
    LfmConversation *conversation, const char *text, size_t text_bytes,
    const float *pcm, size_t sample_count, uint32_t sample_rate,
    LfmNativeEmission *out, LfmAudioRouteNotify notify,
    void *notify_context, LfmConversationAdmissionHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_begin_collect_native(
    LfmConversation *conversation, LfmConversationAdmissionHandle *handle);
LFM_INTERNAL_API int lfm_conversation_next_requires_playback_native(
    LfmConversation *conversation);
LFM_INTERNAL_API int lfm_conversation_next_submit_native(
    LfmConversation *conversation, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_next_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle,
    LfmNativeEmission *out);
LFM_INTERNAL_API int lfm_conversation_next_into_submit_native(
    LfmConversation *conversation, const LfmAudioRouteTarget *target,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_next_into_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle,
    LfmNativeEmission *out, size_t *out_samples);
LFM_INTERNAL_API int lfm_conversation_interrupt_submit_native(
    LfmConversation *conversation, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle);
LFM_INTERNAL_API int lfm_conversation_interrupt_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle);
LFM_INTERNAL_API int
lfm_conversation_belongs_to(const LfmConversation *conversation,
                            const LfmModel *model);

#endif /* LFM_MODEL_INTERNAL_H */
