#ifndef LFM_PLATFORM_AUDIO_INTERNAL_H
#define LFM_PLATFORM_AUDIO_INTERNAL_H

#include "lfm_audio_dock.h"
#include "lfm_platform_audio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Stable session-owned hook. The context remains allocated until the session
 * itself is destroyed, even after the physical device endpoints retire. */
typedef struct LfmPlatformAudioBindingV1 {
    uint32_t size;
    uint32_t abi_version;
    void *context;
    int (*playback_ready)(void *context, const LfmPcmLeaseV1 *lease);
    int (*playback_flush)(void *context, uint64_t stream_epoch);
    int (*retire_context)(void *context);
    int (*finish_retirement)(void *context);
    void (*destroy_context)(void *context);
} LfmPlatformAudioBindingV1;

int lfm_internal_session_bind_platform_audio(
    LfmSession *session, const LfmPlatformAudioConfigV1 *config,
    const LfmPlatformAudioBindingV1 *binding);
void lfm_internal_session_platform_fault(LfmSession *session,
                                         int32_t status);
void lfm_internal_session_platform_retirement_ready(LfmSession *session);
uint64_t lfm_internal_session_epoch(const LfmSession *session);
int lfm_internal_playback_consumer_discard_all(
    LfmPlaybackConsumer *consumer, uint64_t *out_frames);
int lfm_internal_playback_consumer_publish_flush(
    LfmPlaybackConsumer *consumer, uint64_t stream_epoch,
    LfmPlaybackRenderV1 *out);
int lfm_internal_platform_audio_callback_retirement_test(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_PLATFORM_AUDIO_INTERNAL_H */
