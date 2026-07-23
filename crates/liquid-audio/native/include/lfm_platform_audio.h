#ifndef LFM_PLATFORM_AUDIO_H
#define LFM_PLATFORM_AUDIO_H

/* Native platform-audio ownership. The operating system invokes the hardware
 * callbacks; those callbacks publish directly into the session's retained PCM
 * docks. No Rust PCM object, generic channel, polling loop, or device thread is
 * part of this path. */

#include <stdint.h>

#include "lfm_session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LfmPlatformAudio LfmPlatformAudio;

typedef struct LfmPlatformAudioConfig {
    uint32_t capture_device;
    uint32_t playback_device;
    uint32_t capture_sample_rate;
    uint32_t playback_sample_rate;
    uint32_t capture_callback_frames;
    uint32_t playback_callback_frames;
    uint32_t flags;
} LfmPlatformAudioConfig;

/* Query the current default input/output devices without opening a model or
 * allocating callback buffers. The returned identity is passed back to create;
 * device/rate drift between the two operations is a setup failure. */
LFM_PUBLIC_API int lfm_platform_audio_default_config(
    LfmPlatformAudioConfig *out);

/* Create both CoreAudio callback units and every callback buffer while the
 * session is CREATED. This call also creates the session's sole capture
 * producer and playback consumer and installs the correlated playback edge.
 * No hardware callback is admitted before start. */
LFM_PUBLIC_API int lfm_platform_audio_create(
    LfmSession *session, const LfmPlatformAudioConfig *config,
    LfmPlatformAudio **out);
LFM_PUBLIC_API int lfm_platform_audio_start(LfmPlatformAudio *audio);

/* Control operations are atomic publications observed by a later hardware
 * callback. They never call into CoreAudio, wait, allocate, or run PCM math on
 * the caller's stack. */
LFM_PUBLIC_API int lfm_platform_audio_set_capture_enabled(
    LfmPlatformAudio *audio, uint32_t enabled);

/* Terminal teardown requests session stop, closes playback-ready admission,
 * and disconnects CoreAudio. If a callback or start operation was already
 * admitted, that operation is the causal successor that retires its owned
 * resources; this call returns LFM_STATUS_WOULD_BLOCK until that edge lands.
 * The small binding object remains session-owned until lfm_session_destroy,
 * preventing late-callback UAF without assigning a thread to observe it. */
LFM_PUBLIC_API int lfm_platform_audio_retire(LfmPlatformAudio *audio);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_PLATFORM_AUDIO_H */
