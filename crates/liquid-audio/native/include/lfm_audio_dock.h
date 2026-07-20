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

typedef struct LfmCaptureProducer LfmCaptureProducer;
typedef struct LfmPlaybackConsumer LfmPlaybackConsumer;
typedef struct LfmSessionControl LfmSessionControl;

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

/* Setup-time structural ownership for the hardware capture edge. Exactly one
 * non-cloneable host endpoint owns this handle for one device stream. The one
 * endpoint may retain a bounded ping-pong set of WRITING leases; generation-
 * checked pool cells, not cloned producers, distinguish them. It retains the
 * session/storage until the device has stopped and joined and every writing
 * lease has been published or released. None of its realtime methods lock,
 * allocate, wait, or retry on contention. */
LFM_PUBLIC_API int lfm_capture_producer_create(
    LfmSession *session, LfmCaptureProducer **out);
LFM_PUBLIC_API int lfm_capture_producer_reserve(
    LfmCaptureProducer *producer, uint32_t frames, uint32_t sample_rate,
    LfmPcmLeaseV1 *out);
LFM_PUBLIC_API int lfm_capture_producer_resolve_mut(
    LfmCaptureProducer *producer, const LfmPcmLeaseV1 *lease,
    float **out_samples, size_t *out_sample_capacity);
LFM_PUBLIC_API int lfm_capture_producer_finalize(
    LfmCaptureProducer *producer, LfmPcmLeaseV1 *lease,
    uint32_t offset_frames, uint32_t used_frames);
LFM_PUBLIC_API int lfm_capture_producer_publish(
    LfmCaptureProducer *producer, const LfmPcmLeaseV1 *lease);
LFM_PUBLIC_API int lfm_capture_producer_release(
    LfmCaptureProducer *producer, const LfmPcmLeaseV1 *lease);
/* Administrative teardown after the hardware stream has stopped and joined. */
LFM_PUBLIC_API int lfm_capture_producer_destroy(
    LfmCaptureProducer *producer);

/* Setup-time structural ownership for the hardware playback edge. Exactly one
 * non-cloneable device endpoint owns this handle. Claim/resolve/release are
 * bounded callback operations: no allocation, mutex, wait, timer, retry, or
 * Rust callback occurs. The ticket identity supplied by the reliable
 * PLAYBACK_READY record is validated before the lease becomes active. Claim is
 * not a polling API: a missing or mismatched head is STALE and never consumes
 * another ticket. Session join refuses to retire its continuation notifiers
 * until this endpoint has been disconnected from the device and destroyed. */
LFM_PUBLIC_API int lfm_playback_consumer_create(
    LfmSession *session, LfmPlaybackConsumer **out);
LFM_PUBLIC_API int lfm_playback_consumer_claim(
    LfmPlaybackConsumer *consumer, const LfmTicketIdV1 *ticket,
    uint64_t stream_epoch, uint64_t lease_id, uint64_t buffer_generation,
    LfmPcmLeaseV1 *out);
LFM_PUBLIC_API int lfm_playback_consumer_resolve(
    const LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    const float **out_samples, size_t *out_sample_count);
LFM_PUBLIC_API int lfm_playback_consumer_release(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease);
/* Administrative teardown after the device callback has been disconnected and
 * joined. Returns BUSY while an exact playback lease is still active. */
LFM_PUBLIC_API int lfm_playback_consumer_destroy(
    LfmPlaybackConsumer *consumer);

/* Interrupt is a separate retained control edge. It never borrows the capture
 * producer and therefore cannot accidentally turn a second source into an
 * SPSC producer. Creation/destruction are administrative; interrupt itself is
 * one lock-free, coalescible epoch transition. */
LFM_PUBLIC_API int lfm_session_control_create(
    LfmSession *session, LfmSessionControl **out);
LFM_PUBLIC_API int lfm_session_control_interrupt(
    LfmSessionControl *control, uint64_t *out_epoch);
LFM_PUBLIC_API int lfm_session_control_destroy(LfmSessionControl *control);

/* Bounded nonblocking try-reserve for native and administrative callers.
 * Hardware capture uses the producer API above: each device callback makes
 * one constant-work admission attempt and an occupied target is an explicit
 * WOULD_BLOCK/XRUN outcome, never a slot hunt or a phantom capacity waiter. */
LFM_PUBLIC_API int lfm_audio_dock_reserve(
    LfmSession *session, uint32_t direction, uint32_t frames,
    uint32_t sample_rate, LfmPcmLeaseV1 *out);
/* Atomically bind one bounded UTF-8 input and one RESERVED capture lease to
 * the lease's existing action ticket. On success the command owns the lease;
 * the caller must not publish or release it. On any failure the lease remains
 * RESERVED and caller-owned. Full-ring admission returns WOULD_BLOCK; no
 * thread or borrowed caller pointer survives the attempt. */
LFM_PUBLIC_API int lfm_session_submit_mixed(
    LfmSession *session, const char *utf8, size_t utf8_bytes,
    const LfmPcmLeaseV1 *capture, LfmTicketIdV1 *out_ticket);
LFM_PUBLIC_API int lfm_audio_dock_resolve_mut(
    LfmSession *session, const LfmPcmLeaseV1 *lease, float **out_samples,
    size_t *out_sample_capacity);
/* Select the exact {offset, length} subspan of one RESERVED capture lease at
 * the completed utterance's sample-clock boundary. The same generation and
 * storage remain authoritative and no data moves. This is a state transition
 * over the view, not a publication edge; publish exactly once afterward. */
LFM_PUBLIC_API int lfm_audio_dock_finalize_capture(
    LfmSession *session, LfmPcmLeaseV1 *lease, uint32_t offset_frames,
    uint32_t used_frames);
LFM_PUBLIC_API int lfm_audio_dock_publish(
    LfmSession *session, const LfmPcmLeaseV1 *lease);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_AUDIO_DOCK_H */
