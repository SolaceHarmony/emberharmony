#ifndef LFM_AUDIO_DOCK_H
#define LFM_AUDIO_DOCK_H

/* Private native PCM dock ABI. The product header intentionally does not
 * include this file. Lease cells contain identity and bounds, never pointers;
 * a pointer exists only during a generation-checked resolve call. */

#include <stddef.h>
#include <stdint.h>

#include "lfm_session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_PCM_FORMAT_F32 1u
#define LFM_PCM_LEASE_PLAYBACK 2u
#define LFM_PCM_LEASE_DIRECTION_MASK 3u
#define LFM_CAPTURE_CHUNK_GAP (1u << 0)
#define LFM_CAPTURE_CHUNK_XRUN (1u << 1)
#define LFM_CAPTURE_CHUNK_MUTED (1u << 2)
#define LFM_CAPTURE_INPUT_F32 1u
#define LFM_CAPTURE_INPUT_I16 2u
#define LFM_CAPTURE_INPUT_U16 3u
#define LFM_CAPTURE_WRITE_GAP_PUBLISHED (1u << 0)
#define LFM_CAPTURE_DEADLINE_PREPARE 0u
#define LFM_CAPTURE_DEADLINE_COMMIT 1u
#define LFM_CAPTURE_DEADLINE_FORCED 2u
#define LFM_CAPTURE_DEADLINE_COUNT 3u
#define LFM_PLAYBACK_EVIDENCE_RENDERED (1u << 0)
#define LFM_PLAYBACK_EVIDENCE_SILENCE (1u << 1)
#define LFM_PLAYBACK_EVIDENCE_FLUSH (1u << 2)
#define LFM_PLAYBACK_EVIDENCE_DISCONTINUITY (1u << 3)
typedef struct LfmCaptureProducer LfmCaptureProducer;
typedef struct LfmPlaybackConsumer LfmPlaybackConsumer;
typedef struct LfmSessionControl LfmSessionControl;

typedef struct LfmPcmLease {
    uint64_t lease_id;
    uint64_t stream_epoch;
    uint64_t buffer_generation;
    LfmTicketId ticket;
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
    uint32_t format;
    uint32_t offset_bytes;
    uint32_t length_bytes;
    uint32_t flags;
} LfmPcmLease;

/* A callback publication is metadata over one logical subrange of the native
 * circular arena. It never carries a process pointer. `first_sample_cursor` advances for
 * both stored frames and explicit gaps, so a consumer can distinguish a true
 * contiguous turn from an XRUN without consulting wall time. */
typedef struct LfmCaptureChunk {
    uint64_t stream;
    uint32_t lane;
    uint32_t flags;
    uint64_t chunk_sequence;
    uint64_t first_sample_cursor;
    uint64_t stream_epoch;
    LfmTicketId turn_ticket;
    uint64_t lease_id;
    uint64_t buffer_generation;
    uint32_t offset_frames;
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
} LfmCaptureChunk;

/* Result of one hardware callback publication. `status` is the original
 * admission/conversion/commit outcome. A nonzero status is a dropped block;
 * GAP_PUBLISHED proves the same call also installed a sequenced XRUN record. */
typedef struct LfmCaptureWrite {
    uint32_t admitted_frames;
    uint32_t dropped_frames;
    uint32_t flags;
    int32_t status;
} LfmCaptureWrite;

/* A callback reservation can wrap the fixed native circular arena once. The
 * two writable views are the complete logical destination in array order;
 * neither view owns storage and their combined count equals chunk.frames. */
typedef struct LfmMutableF32Span {
    float *data;
    size_t count;
} LfmMutableF32Span;

/* One playback callback edge. PCM remains in the immutable native lease; this
 * result contains correlation and sample-clock accounting only. */
typedef struct LfmPlaybackRender {
    uint64_t session_id;
    uint64_t stream_epoch;
    LfmTicketId ticket;
    uint64_t lease_id;
    uint64_t buffer_generation;
    uint32_t source_offset_frames;
    uint32_t rendered_frames;
    uint64_t first_playback_sample_cursor;
    uint64_t capture_sample_cursor_snapshot;
    uint32_t flags;
} LfmPlaybackRender;

/* Production capture dock. Creation binds exactly one stream/lane endpoint to
 * the fixed native circular arena. Claim/resolve/commit are constant-work realtime
 * operations: no allocation, mutex, wait, retry loop, slot scan, callback, or
 * timer. Claim returns a generation-checked view descriptor; only resolve
 * exposes its exact writable subspan for the duration of that callback. Turn
 * boundaries are exclusively owned by native detector policy; this transport
 * has no manual end-of-turn operation or flag. One callback publication is
 * bounded by the platform contract sealed in
 * `LfmSessionConfig::capture_max_callback_frames`; a larger device block is
 * rejected as one block and becomes an explicit XRUN rather than an unbounded
 * WRITING lease. The sole physical producer is created while the session is
 * CREATED; readiness permanently closes endpoint allocation. */
LFM_PUBLIC_API int lfm_capture_chunk_producer_create(
    LfmSession *session, uint64_t stream, uint32_t lane,
    LfmCaptureProducer **out);
LFM_PUBLIC_API int lfm_capture_producer_claim_chunk(
    LfmCaptureProducer *producer, uint32_t frames, uint32_t sample_rate,
    uint32_t source_channels, uint32_t flags, LfmCaptureChunk *out);
LFM_PUBLIC_API int lfm_capture_producer_resolve_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunk *chunk,
    LfmMutableF32Span out_spans[2], uint32_t *out_span_count);
LFM_PUBLIC_API int lfm_capture_producer_commit_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunk *chunk);
/* One bounded realtime callback transition. Source PCM is borrowed only for
 * this synchronous call. Native code claims the exact destination subspan,
 * invokes one architecture-specific format/downmix leaf directly into it, and
 * commits one metadata record. No staging allocation or Rust numerical loop is
 * involved. `sample_count` is the number of interleaved scalar source values. */
LFM_PUBLIC_API int lfm_capture_producer_write_interleaved(
    LfmCaptureProducer *producer, const void *samples, size_t sample_count,
    uint32_t channels, uint32_t sample_rate, uint32_t format, uint32_t flags,
    LfmCaptureWrite *out);
/* Cancel one claimed subspan before publication. This retires no samples and
 * publishes no record; hardware callers use it only when their device callback
 * itself fails before filling the promised block. */
LFM_PUBLIC_API int lfm_capture_producer_abort_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunk *chunk);
/* Publish a known discontinuity. Dropped frames advance the sample cursor but
 * never stored PCM, and force a turn boundary before later PCM is admitted.
 * GAP is mandatory; XRUN distinguishes device overflow from an intentional
 * discontinuity. */
LFM_PUBLIC_API int lfm_capture_producer_publish_gap(
    LfmCaptureProducer *producer, uint32_t dropped_frames,
    uint32_t source_channels, uint32_t flags, LfmCaptureChunk *out);
/* Administrative teardown after the hardware stream has stopped and joined. */
LFM_PUBLIC_API int lfm_capture_producer_destroy(
    LfmCaptureProducer *producer);

/* Setup-time structural ownership for the hardware playback edge. Exactly one
 * non-cloneable device endpoint owns this handle. Claim/render/release are
 * bounded callback operations: no allocation, mutex, wait, timer, retry, or
 * Rust callback occurs. The ticket identity supplied by the reliable
 * PLAYBACK_READY record is validated before the lease becomes active. Claim is
 * not a polling API: a missing or mismatched head is STALE and never consumes
 * another ticket. The sole physical consumer is created while the session is
 * CREATED; readiness permanently closes endpoint allocation. Session join
 * refuses to retire its continuation notifiers until this endpoint has been
 * disconnected from the device and destroyed. */
LFM_PUBLIC_API int lfm_playback_consumer_create(
    LfmSession *session, LfmPlaybackConsumer **out);
LFM_PUBLIC_API int lfm_playback_consumer_claim(
    LfmPlaybackConsumer *consumer, const LfmTicketId *ticket,
    uint64_t stream_epoch, uint64_t lease_id, uint64_t buffer_generation,
    LfmPcmLease *out);
/* Lease-aware realtime render operations. The source pointer never crosses the
 * ABI: native code validates the exact active lease, fans its requested range
 * directly into the device callback buffer, and publishes one metadata-only
 * playback-evidence record. No allocation, mutex, wait, retry, or callback is
 * performed. destination_capacity is measured in interleaved scalar samples. */
LFM_PUBLIC_API int lfm_playback_consumer_render_f32(
    LfmPlaybackConsumer *consumer, const LfmPcmLease *lease,
    uint32_t source_offset_frames, float *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRender *out);
LFM_PUBLIC_API int lfm_playback_consumer_render_i16(
    LfmPlaybackConsumer *consumer, const LfmPcmLease *lease,
    uint32_t source_offset_frames, int16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRender *out);
LFM_PUBLIC_API int lfm_playback_consumer_render_u16(
    LfmPlaybackConsumer *consumer, const LfmPcmLease *lease,
    uint32_t source_offset_frames, uint16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRender *out);
/* Publish callback-rendered logical silence or a zero-frame flush/
 * discontinuity edge. A null lease uses the most recently claimed exact
 * playback lineage; observation before any successful claim is STALE. */
LFM_PUBLIC_API int lfm_playback_consumer_observe(
    LfmPlaybackConsumer *consumer, const LfmPcmLease *lease,
    uint32_t source_offset_frames, uint32_t frames, uint32_t flags,
    LfmPlaybackRender *out);
LFM_PUBLIC_API int lfm_playback_consumer_release(
    LfmPlaybackConsumer *consumer, const LfmPcmLease *lease);
/* Administrative teardown after the device callback has been disconnected and
 * joined. Returns BUSY while an exact playback lease is still active. */
LFM_PUBLIC_API int lfm_playback_consumer_destroy(
    LfmPlaybackConsumer *consumer);

/* Interrupt is a separate retained control edge. It never borrows the capture
 * producer and therefore cannot accidentally turn a second source into an
 * SPSC producer. Control handles are allocated only while the session is
 * CREATED; destruction is administrative, and interrupt itself is one
 * lock-free, coalescible epoch transition. */
LFM_PUBLIC_API int lfm_session_control_create(
    LfmSession *session, LfmSessionControl **out);
LFM_PUBLIC_API int lfm_session_control_interrupt(
    LfmSessionControl *control, uint64_t *out_epoch);
LFM_PUBLIC_API int lfm_session_control_destroy(LfmSessionControl *control);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_AUDIO_DOCK_H */
