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
/* Private bring-up mode for dock/lifecycle verification before a model is
 * attached. Production session creation rejects null owners without it. */
#define LFM_SESSION_FLAG_DOCK_ONLY (UINT64_C(1) << 63)
/* Deterministic private lifecycle-test backend. It is accepted only together
 * with DOCK_ONLY and cannot enter a production model session. */
#define LFM_SESSION_FLAG_MANUAL_DEADLINES (UINT64_C(1) << 62)

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

/* A callback publication is metadata over one logical subrange of the native
 * circular arena. It never carries a process pointer. `first_sample_cursor` advances for
 * both stored frames and explicit gaps, so a consumer can distinguish a true
 * contiguous turn from an XRUN without consulting wall time. */
typedef struct LfmCaptureChunkV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t stream;
    uint32_t lane;
    uint32_t flags;
    uint64_t chunk_sequence;
    uint64_t first_sample_cursor;
    uint64_t stream_epoch;
    LfmTicketIdV1 turn_ticket;
    uint64_t lease_id;
    uint64_t buffer_generation;
    uint32_t offset_frames;
    uint32_t frames;
    uint32_t channels;
    uint32_t sample_rate;
    uint64_t reserved[2];
} LfmCaptureChunkV1;

/* Result of one hardware callback publication. `status` is the original
 * admission/conversion/commit outcome. A nonzero status is a dropped block;
 * GAP_PUBLISHED proves the same call also installed a sequenced XRUN record. */
typedef struct LfmCaptureWriteV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t admitted_frames;
    uint32_t dropped_frames;
    uint32_t flags;
    int32_t status;
    uint64_t reserved[2];
} LfmCaptureWriteV1;

/* A callback reservation can wrap the fixed native circular arena once. The
 * two writable views are the complete logical destination in array order;
 * neither view owns storage and their combined count equals chunk.frames. */
typedef struct LfmMutableF32SpanV1 {
    float *data;
    size_t count;
} LfmMutableF32SpanV1;

/* One playback callback edge. PCM remains in the immutable native lease; this
 * result contains correlation and sample-clock accounting only. */
typedef struct LfmPlaybackRenderV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t stream_epoch;
    LfmTicketIdV1 ticket;
    uint64_t lease_id;
    uint64_t buffer_generation;
    uint32_t source_offset_frames;
    uint32_t rendered_frames;
    uint64_t first_playback_sample_cursor;
    uint64_t capture_sample_cursor_snapshot;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved[2];
} LfmPlaybackRenderV1;

/* Private playback-detector observability. The detector is session-owned and
 * created at playback_sample_rate; these counters are release-published by the
 * coordinator after it consumes playback evidence. */
typedef struct LfmPlaybackPolicySnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t sample_rate;
    uint32_t last_voice;
    uint32_t detector_backlog;
    uint32_t retained_observers;
    uint64_t evidence_records;
    uint64_t evidence_updates;
    uint64_t last_evidence_cursor;
    uint64_t discontinuities;
    uint64_t stream_epoch;
    LfmTicketIdV1 ticket;
    uint64_t capture_sample_cursor_snapshot;
    double last_score;
    uint32_t adaptive_min;
    uint32_t adaptive_max;
    uint64_t echo_start_capture_cursor;
    uint64_t last_voice_capture_cursor;
    uint64_t echo_tail_capture_cursor;
    uint64_t barge_voiced_frames;
    uint64_t barge_interrupts;
    uint64_t barge_source_epoch;
    uint64_t barge_interrupt_epoch;
    LfmTicketIdV1 barge_playback_ticket;
    uint64_t reserved[1];
} LfmPlaybackPolicySnapshotV1;

typedef struct LfmCaptureDeadlineSlotSnapshotV1 {
    uint32_t slot;
    uint32_t armed;
    uint32_t terminal;
    uint32_t cancel_cause;
    uint64_t arm_generation;
    uint64_t expiry_generation;
    uint64_t scope_generation;
    uint64_t epoch;
    uint64_t domain;
    uint64_t pause_generation;
    LfmTicketIdV1 child;
    LfmTicketIdV1 parent;
} LfmCaptureDeadlineSlotSnapshotV1;

typedef struct LfmCaptureSupervisionSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t cycle_active;
    uint32_t scope_phase;
    uint32_t source_phase;
    uint32_t source_pending_events;
    uint32_t policy_state;
    uint32_t reserved0;
    uint64_t scope_generation;
    uint64_t epoch;
    uint64_t domain;
    uint64_t pause_generation;
    uint64_t prepare_ready_generation;
    uint64_t commit_ready_generation;
    uint64_t forced_ready_generation;
    uint64_t prepare_sample_generation;
    uint64_t commit_sample_generation;
    uint64_t forced_sample_generation;
    uint64_t turn_start_cursor;
    uint64_t last_evidence_cursor;
    uint64_t silence_frames;
    LfmTicketIdV1 parent;
    LfmCaptureDeadlineSlotSnapshotV1 slots[LFM_CAPTURE_DEADLINE_COUNT];
} LfmCaptureSupervisionSnapshotV1;

/* Production capture dock. Creation binds exactly one stream/lane endpoint to
 * the fixed native circular arena. Claim/resolve/commit are constant-work realtime
 * operations: no allocation, mutex, wait, retry loop, slot scan, callback, or
 * timer. Claim returns a generation-checked view descriptor; only resolve
 * exposes its exact writable subspan for the duration of that callback. Turn
 * boundaries are exclusively owned by native detector policy; this transport
 * has no manual end-of-turn operation or flag. One callback publication is
 * bounded by the platform contract sealed in
 * `LfmSessionConfigV1::capture_max_callback_frames`; a larger device block is
 * rejected as one block and becomes an explicit XRUN rather than an unbounded
 * WRITING lease. */
LFM_PUBLIC_API int lfm_capture_chunk_producer_create(
    LfmSession *session, uint64_t stream, uint32_t lane,
    LfmCaptureProducer **out);
LFM_PUBLIC_API int lfm_capture_producer_claim_chunk(
    LfmCaptureProducer *producer, uint32_t frames, uint32_t sample_rate,
    uint32_t source_channels, uint32_t flags, LfmCaptureChunkV1 *out);
LFM_PUBLIC_API int lfm_capture_producer_resolve_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunkV1 *chunk,
    LfmMutableF32SpanV1 out_spans[2], uint32_t *out_span_count);
LFM_PUBLIC_API int lfm_capture_producer_commit_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunkV1 *chunk);
/* One bounded realtime callback transition. Source PCM is borrowed only for
 * this synchronous call. Native code claims the exact destination subspan,
 * invokes one architecture-specific format/downmix leaf directly into it, and
 * commits one metadata record. No staging allocation or Rust numerical loop is
 * involved. `sample_count` is the number of interleaved scalar source values. */
LFM_PUBLIC_API int lfm_capture_producer_write_interleaved(
    LfmCaptureProducer *producer, const void *samples, size_t sample_count,
    uint32_t channels, uint32_t sample_rate, uint32_t format, uint32_t flags,
    LfmCaptureWriteV1 *out);
/* Cancel one claimed subspan before publication. This retires no samples and
 * publishes no record; hardware callers use it only when their device callback
 * itself fails before filling the promised block. */
LFM_PUBLIC_API int lfm_capture_producer_abort_chunk(
    LfmCaptureProducer *producer, const LfmCaptureChunkV1 *chunk);
/* Publish a known discontinuity. Dropped frames advance the sample cursor but
 * never stored PCM, and force a turn boundary before later PCM is admitted.
 * GAP is mandatory; XRUN distinguishes device overflow from an intentional
 * discontinuity. */
LFM_PUBLIC_API int lfm_capture_producer_publish_gap(
    LfmCaptureProducer *producer, uint32_t dropped_frames,
    uint32_t source_channels, uint32_t flags, LfmCaptureChunkV1 *out);
/* Administrative teardown after the hardware stream has stopped and joined. */
LFM_PUBLIC_API int lfm_capture_producer_destroy(
    LfmCaptureProducer *producer);

/* Setup-time structural ownership for the hardware playback edge. Exactly one
 * non-cloneable device endpoint owns this handle. Claim/render/release are
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
/* Lease-aware realtime render operations. The source pointer never crosses the
 * ABI: native code validates the exact active lease, fans its requested range
 * directly into the device callback buffer, and publishes one metadata-only
 * playback-evidence record. No allocation, mutex, wait, retry, or callback is
 * performed. destination_capacity is measured in interleaved scalar samples. */
LFM_PUBLIC_API int lfm_playback_consumer_render_f32(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, float *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out);
LFM_PUBLIC_API int lfm_playback_consumer_render_i16(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, int16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out);
LFM_PUBLIC_API int lfm_playback_consumer_render_u16(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, uint16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out);
/* Publish callback-rendered logical silence or a zero-frame flush/
 * discontinuity edge. A null lease uses the most recently claimed exact
 * playback lineage; observation before any successful claim is STALE. */
LFM_PUBLIC_API int lfm_playback_consumer_observe(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, uint32_t frames, uint32_t flags,
    LfmPlaybackRenderV1 *out);
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

/* Private deterministic supervision diagnostics. Manual advance/fire are
 * rejected unless the session was created with MANUAL_DEADLINES. They inject
 * only the deadline source's ordinary wake hint; the native coordinator still
 * performs the exact identity check, immutable event acknowledgement, scope
 * transition, and turn commit. */
LFM_INTERNAL_API int lfm_session_capture_supervision_snapshot(
    const LfmSession *session, LfmCaptureSupervisionSnapshotV1 *out);
LFM_INTERNAL_API int lfm_session_capture_deadline_advance_manual_test(
    LfmSession *session, uint64_t elapsed_ns);
LFM_INTERNAL_API int lfm_session_capture_deadline_fire_manual_test(
    LfmSession *session, uint32_t slot);
LFM_INTERNAL_API int lfm_session_capture_deadline_identity_test(
    const LfmSession *session, uint32_t slot,
    const LfmCaptureDeadlineSlotSnapshotV1 *identity);
LFM_PUBLIC_API int lfm_session_playback_policy_snapshot(
    const LfmSession *session, LfmPlaybackPolicySnapshotV1 *out);
/* Private implementation-backed dock-test source. It enters the ordinary
 * reserve/fill/publish path and is rejected outside DOCK_ONLY sessions. */
LFM_INTERNAL_API int lfm_session_publish_playback_f32_test(
    LfmSession *session, const float *samples, uint32_t frames,
    LfmPcmLeaseV1 *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_AUDIO_DOCK_H */
