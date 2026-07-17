#ifndef LFM_AUDIO_DOCK_H
#define LFM_AUDIO_DOCK_H

/* Private Rust/native PCM dock ABI. The product header intentionally does not
 * include this file. Lease cells contain identity and bounds, never pointers;
 * a pointer exists only during a generation-checked resolve call. */

#include <stddef.h>
#include <stdint.h>

#include "lfm_session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_PCM_FORMAT_F32 1u
#define LFM_PCM_LEASE_CAPTURE 1u
#define LFM_PCM_LEASE_PLAYBACK 2u
#define LFM_PCM_LEASE_DIRECTION_MASK 3u
#define LFM_PCM_LEASE_TURN_END (1u << 8)
/* Private bring-up mode for dock/lifecycle verification before a model is
 * attached. Production session creation rejects null owners without it. */
#define LFM_SESSION_FLAG_DOCK_ONLY (UINT64_C(1) << 63)

typedef struct LfmPcmLeaseV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t lease_id;
    uint64_t stream_epoch;
    uint64_t buffer_generation;
    LfmTicketIdV1 ticket;
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
    uint32_t format;
    uint32_t offset_bytes;
    uint32_t length_bytes;
    uint32_t flags;
    uint32_t reserved;
} LfmPcmLeaseV1;

/* Compatibility try-reserve. Returns WOULD_BLOCK when every slot is live. */
LFM_PUBLIC_API int lfm_audio_dock_reserve(
    LfmSession *session, uint32_t direction, uint32_t frames,
    uint32_t sample_rate, LfmPcmLeaseV1 *out);
/* Expected-value blocking reserve. It never probes on a timer and aborts an
 * admission wait when interrupt advances the stream epoch. */
LFM_PUBLIC_API int lfm_audio_dock_wait_reserve(
    LfmSession *session, uint32_t direction, uint32_t frames,
    uint32_t sample_rate, LfmPcmLeaseV1 *out);
/* Atomically bind one bounded UTF-8 input and one RESERVED capture lease to
 * the lease's existing action ticket. On success the command owns the lease;
 * the caller must not publish or release it. On any failure the lease remains
 * RESERVED and caller-owned. The blocking form parks on command-ring space and
 * aborts when interrupt advances the lease epoch or the session stops. */
LFM_PUBLIC_API int lfm_session_submit_mixed(
    LfmSession *session, const char *utf8, size_t utf8_bytes,
    const LfmPcmLeaseV1 *capture, LfmTicketIdV1 *out_ticket);
LFM_PUBLIC_API int lfm_session_wait_submit_mixed(
    LfmSession *session, const char *utf8, size_t utf8_bytes,
    const LfmPcmLeaseV1 *capture, LfmTicketIdV1 *out_ticket);
LFM_PUBLIC_API int lfm_audio_dock_resolve_mut(
    LfmSession *session, const LfmPcmLeaseV1 *lease, float **out_samples,
    size_t *out_sample_capacity);
LFM_PUBLIC_API int lfm_audio_dock_resolve(
    const LfmSession *session, const LfmPcmLeaseV1 *lease,
    const float **out_samples, size_t *out_sample_count);
LFM_PUBLIC_API int lfm_audio_dock_publish(
    LfmSession *session, const LfmPcmLeaseV1 *lease);
/* Expected-value blocking playback receive. It never probes on a timer. */
LFM_PUBLIC_API int lfm_audio_dock_wait_playback(LfmSession *session,
                                                LfmPcmLeaseV1 *out);
LFM_PUBLIC_API int lfm_audio_dock_release(
    LfmSession *session, const LfmPcmLeaseV1 *lease);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_AUDIO_DOCK_H */
