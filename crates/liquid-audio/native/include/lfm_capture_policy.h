// Private observability for the native Sesame turn-policy continuation.
// This is not part of the product ABI: Rust owns only opaque PCM endpoints.
#ifndef LFM_CAPTURE_POLICY_H
#define LFM_CAPTURE_POLICY_H

#include <stdint.h>

#include "lfm_types.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_CAPTURE_POLICY_ABI 1u

#define LFM_CAPTURE_LISTENING 0u
#define LFM_CAPTURE_CANDIDATE 1u
#define LFM_CAPTURE_SPEAKING 2u
#define LFM_CAPTURE_PAUSE 3u

typedef struct LfmCapturePolicySnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t sample_rate;
    uint32_t state;
    uint32_t last_voice;
    uint32_t detector_backlog;
    uint64_t evidence_updates;
    uint64_t last_evidence_cursor;
    uint64_t turn_start_cursor;
    uint64_t last_voiced_cursor;
    uint64_t voiced_frames;
    uint64_t silence_frames;
    uint64_t pause_generation;
    uint64_t prepare_sample_generation;
    uint64_t commit_sample_generation;
    uint64_t forced_sample_generation;
    double last_score;
    uint32_t adaptive_min;
    uint32_t adaptive_max;
    uint64_t discarded_silence_frames;
    uint64_t reserved[3];
} LfmCapturePolicySnapshotV1;

LFM_INTERNAL_API int lfm_session_capture_policy_snapshot(
    const LfmSession *session, LfmCapturePolicySnapshotV1 *out);

#ifdef __cplusplus
}
#endif

#endif // LFM_CAPTURE_POLICY_H
