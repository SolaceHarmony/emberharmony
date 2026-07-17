// Private native audio-encode pass. This is not part of the product ABI: it is
// the typed SQ/CQ boundary between a conversation's retained PCM lease and its
// pre-reserved adapted-row plane.
//
// Every pointer below is a borrowed view. Plans own immutable tables/weight
// views; workspaces and activation spans are conversation-owned and must remain
// live until exact completion. The descriptor owns no storage and performs no
// conversion, repack, alignment copy, or relocation.
#ifndef LFM_AUDIO_PASS_H
#define LFM_AUDIO_PASS_H

#include "lfm_conformer.h"
#include "lfm_frontend.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_AUDIO_PASS_ABI 1u

typedef struct LfmAudioEncodePassV1 {
    uint32_t size;
    uint32_t abi_version;

    const LfmResampler *resampler;
    LfmResamplerWorkspace *resampler_workspace;
    const LfmFrontend *frontend;
    LfmFrontendWorkspace *frontend_workspace;
    const LfmConformer *conformer;
    LfmConformerWorkspace *conformer_workspace;

    const float *pcm;
    uint64_t sample_count;
    float *resampled;
    uint64_t resampled_capacity;
    uint16_t *mel;
    uint64_t mel_capacity;
    uint16_t *adapted;
    uint64_t adapted_capacity;
    uint64_t *out_adapted_values;
} LfmAudioEncodePassV1;

// Submit one retained PCM span through resample -> frontend -> Conformer. The
// complete chain is one typed bridge ticket. `model_id` selects the resident
// backbone plan solely for lifetime/correlation validation; weight bytes remain
// reachable through `pass->conformer` as immutable views.
LFM_INTERNAL_API int lfm_engine_audio_encode(
    void *engine, uint64_t model_id, const LfmAudioEncodePassV1 *pass);

// Private execution witness used by native integration tests.
LFM_INTERNAL_API uint64_t lfm_engine_audio_encode_passes(const void *engine);

// Called only by the audio pass's lane-0 Conformer sequencer. It publishes one
// checkpoint-layout GEMM substage to the peer lanes already inside the same
// ticket, then joins them at the fixed-team fence. It never submits recursively.
LFM_INTERNAL_API int lfm_engine_conformer_gemm_team(
    void *engine, const uint16_t *activation, size_t activation_count,
    const void *weight_bytes, size_t weight_count, float *out,
    size_t out_count, size_t rows, size_t columns, size_t inner);

// Same numerical program as lfm_conformer_forward, but its direct BF16 linears
// use the in-ticket fixed-team substage above instead of nested SQ submissions.
LFM_INTERNAL_API int lfm_conformer_forward_engine_team(
    const LfmConformer *conformer, LfmConformerWorkspace *workspace,
    const uint16_t *mel, uint64_t mel_frames, uint16_t *out_rows,
    uint64_t out_capacity_values);

#ifdef __cplusplus
}
#endif

#endif // LFM_AUDIO_PASS_H
