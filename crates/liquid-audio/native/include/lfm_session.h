#ifndef LFM_SESSION_H
#define LFM_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_types.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmSessionConfigV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t session_id;
    uint32_t capture_slots;
    uint32_t playback_slots;
    uint32_t capture_frames_per_slot;
    uint32_t playback_frames_per_slot;
    uint32_t pcm_channels;
    uint32_t pcm_sample_rate;
    uint32_t command_capacity;
    uint32_t max_new_tokens;
    uint32_t reserved0;
    uint64_t flags;
    uint64_t reserved[4];
} LfmSessionConfigV1;

typedef enum LfmEventKindV1 {
    LFM_EVENT_STATE = 1,
    LFM_EVENT_TEXT = 2,
    LFM_EVENT_TURN = 3,
    LFM_EVENT_ERROR = 4,
    LFM_EVENT_STOPPED = 5,
} LfmEventKindV1;

#define LFM_EVENT_FLAG_HAS_AUDIO (1u << 0)
#define LFM_EVENT_FLAG_TRUNCATED (1u << 1)

typedef struct LfmTurnEventV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t playback_leases;
    uint32_t emitted_items;
} LfmTurnEventV1;

typedef struct LfmEventV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t kind;
    uint32_t flags;
    uint64_t session_id;
    uint64_t epoch;
    LfmTicketIdV1 ticket;
    const void *payload;
    uint32_t payload_bytes;
    int32_t status;
} LfmEventV1;

typedef int (*LfmOnEventV1)(void *context, const LfmEventV1 *event);

typedef struct LfmCallbacksV1 {
    uint32_t size;
    uint32_t abi_version;
    void *context;
    LfmOnEventV1 on_event;
} LfmCallbacksV1;

typedef struct LfmSessionSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t session_id;
    uint64_t epoch;
    uint32_t state;
    int32_t terminal_status;
    uint64_t coordinator_parks;
    uint64_t coordinator_wakes;
    uint64_t notification_parks;
    uint64_t callbacks_entered;
    uint64_t capture_consumed;
    uint64_t capture_stale;
    uint64_t playback_published;
    uint64_t playback_consumed;
    uint64_t text_commands_accepted;
    uint64_t text_commands_consumed;
    uint64_t text_commands_stale;
    uint32_t live_capture_leases;
    uint32_t live_playback_leases;
    uint32_t reliable_event_depth;
    uint32_t reliable_event_capacity;
    uint64_t reserved[4];
} LfmSessionSnapshotV1;

#define LFM_SESSION_CREATED 0u
#define LFM_SESSION_RUNNING 1u
#define LFM_SESSION_STOPPING 2u
#define LFM_SESSION_THREADS_JOINED 3u
#define LFM_SESSION_JOINED 4u

LFM_PUBLIC_API int lfm_session_create(
    LfmRuntime *runtime, LfmModel *model, LfmConversation *conversation,
    const LfmSessionConfigV1 *config, const LfmCallbacksV1 *callbacks,
    LfmSession **out);
LFM_PUBLIC_API int lfm_session_start(LfmSession *session);
/* Compatibility try-submit. Returns WOULD_BLOCK when the fixed control ring is
 * full. New callers should use the expected-value blocking entry point below. */
LFM_PUBLIC_API int lfm_session_submit_text(LfmSession *session,
                                           const char *utf8,
                                           size_t utf8_bytes,
                                           LfmTicketIdV1 *out_ticket);
/* Copies one bounded UTF-8 command into the fixed control ring and returns its
 * native action ticket. Full-ring admission parks without polling until the
 * consumer opens space, the session is interrupted, or the session stops. */
LFM_PUBLIC_API int lfm_session_wait_submit_text(
    LfmSession *session, const char *utf8, size_t utf8_bytes,
    LfmTicketIdV1 *out_ticket);
LFM_PUBLIC_API int lfm_session_interrupt(LfmSession *session,
                                         uint64_t *out_epoch);
LFM_PUBLIC_API void lfm_session_request_stop(LfmSession *session);
LFM_PUBLIC_API int lfm_session_join(LfmSession *session);
LFM_PUBLIC_API int lfm_session_snapshot(const LfmSession *session,
                                        LfmSessionSnapshotV1 *out);
LFM_PUBLIC_API int lfm_session_destroy(LfmSession *session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_SESSION_H */
