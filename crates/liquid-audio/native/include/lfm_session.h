#ifndef LFM_SESSION_H
#define LFM_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_types.h"
#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmSessionConfig {
    uint64_t session_id;
    uint32_t playback_slots;
    /* Maximum complete frame block promised by the platform capture device.
     * Readiness must fail when this contract is unknown. Each callback is one
     * indivisible capture chunk; native never guesses or splits this bound. */
    uint32_t capture_max_callback_frames;
    /* Zero lets a model-backed session derive its exact codec/rate capacity.
     * Dock-only sessions must provide an explicit capacity. */
    uint32_t playback_frames_per_slot;
    uint32_t pcm_channels;
    /* Capture and playback clocks are sealed independently at readiness.
     * Capture leases must retain capture_sample_rate; generated PCM leases
     * always carry playback_sample_rate. */
    uint32_t capture_sample_rate;
    uint32_t playback_sample_rate;
    uint32_t command_capacity;
    uint32_t max_new_tokens;
    uint64_t flags;
} LfmSessionConfig;

typedef enum LfmEventKind {
    LFM_EVENT_STATE = 1,
    LFM_EVENT_TEXT = 2,
    LFM_EVENT_TURN = 3,
    LFM_EVENT_ERROR = 4,
    LFM_EVENT_STOPPED = 5,
    LFM_EVENT_PLAYBACK_READY = 6,
    /* Reliable ownership edge for a native-admitted audio turn. Rust uses the
     * exact ticket only to correlate outward records; it never creates or
     * advances the turn. */
    LFM_EVENT_TURN_STARTED = 7,
} LfmEventKind;

#define LFM_EVENT_FLAG_HAS_AUDIO (1u << 0)
#define LFM_EVENT_FLAG_TRUNCATED (1u << 1)

typedef struct LfmTurnEvent {
    uint32_t playback_leases;
    uint32_t emitted_items;
} LfmTurnEvent;

/* Identity carried by one reliable PLAYBACK_READY edge. The corresponding
 * lease remains native-owned until the session's sole PlaybackConsumer claims
 * the complete {event ticket, epoch, lease_id, buffer_generation} identity. */
typedef struct LfmPlaybackReadyEvent {
    uint64_t lease_id;
    uint64_t buffer_generation;
} LfmPlaybackReadyEvent;

typedef struct LfmEvent {
    uint32_t kind;
    uint32_t flags;
    uint64_t session_id;
    uint64_t epoch;
    LfmTicketId ticket;
    const void *payload;
    uint32_t payload_bytes;
    int32_t status;
} LfmEvent;

/* The callback must copy the bounded record and return immediately. Returning
 * WOULD_BLOCK retains the exact record in the native output continuation;
 * after freeing host capacity, call lfm_session_host_capacity once. No pointer
 * in the callback remains valid after a successful return. Any other nonzero
 * status is a terminal host-sink failure. */
typedef int (*LfmOnEvent)(void *context, const LfmEvent *event);

typedef struct LfmCallbacks {
    void *context;
    LfmOnEvent on_event;
} LfmCallbacks;

#define LFM_SESSION_CREATED 0u
#define LFM_SESSION_RUNNING 1u
#define LFM_SESSION_STOPPING 2u
#define LFM_SESSION_SERVICES_JOINED 3u
#define LFM_SESSION_JOINED 4u

LFM_PUBLIC_API int lfm_session_create(
    LfmRuntime *runtime, LfmModel *model, LfmConversation *conversation,
    const LfmSessionConfig *config, const LfmCallbacks *callbacks,
    LfmSession **out);
LFM_PUBLIC_API int lfm_session_start(LfmSession *session);
/* Copies one bounded UTF-8 command into the fixed control ring and returns its
 * native action ticket. Returns WOULD_BLOCK immediately when the ring is full;
 * callers never park on behalf of admission. */
LFM_PUBLIC_API int lfm_session_submit_text(LfmSession *session,
                                           const char *utf8,
                                           size_t utf8_bytes,
                                           LfmTicketId *out_ticket);
LFM_PUBLIC_API int lfm_session_interrupt(LfmSession *session,
                                         uint64_t *out_epoch);
/* Nonblocking host-capacity edge for a callback that previously returned
 * WOULD_BLOCK. It makes the retained output continuation runnable; it never
 * invokes the callback on the caller's stack. */
LFM_PUBLIC_API int lfm_session_host_capacity(LfmSession *session);
LFM_PUBLIC_API void lfm_session_request_stop(LfmSession *session);
LFM_PUBLIC_API int lfm_session_join(LfmSession *session);
LFM_PUBLIC_API int lfm_session_destroy(LfmSession *session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_SESSION_H */
