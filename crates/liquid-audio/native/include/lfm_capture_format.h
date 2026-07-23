// Realtime-safe device-format boundary for native mono capture/playback views.
//
// Capture borrows the callback's complete interleaved block and writes directly
// into an already-claimed native mono destination. Playback borrows a native
// mono lease and writes directly into the callback's interleaved destination.
// Neither direction allocates storage or performs synchronization. C++ validates
// view geometry; every value-producing conversion and fan-out operation runs
// in an architecture assembly leaf.
#ifndef LFM_CAPTURE_FORMAT_H
#define LFM_CAPTURE_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// destination_capacity is measured in mono float samples. A zero-frame call
// is valid and does not dereference either pointer. f32 is averaged in channel
// order. i16 preserves the existing device contract, sum/(channels*32767).
// u16 is centered at 32768, scaled by 32768, then channel-averaged.
LFM_INTERNAL_API int lfm_capture_downmix_f32(const float *source,
                                             float *destination,
                                             size_t frames,
                                             uint32_t channels,
                                             size_t destination_capacity);
LFM_INTERNAL_API int lfm_capture_downmix_i16(const int16_t *source,
                                             float *destination,
                                             size_t frames,
                                             uint32_t channels,
                                             size_t destination_capacity);
LFM_INTERNAL_API int lfm_capture_downmix_u16(const uint16_t *source,
                                             float *destination,
                                             size_t frames,
                                             uint32_t channels,
                                             size_t destination_capacity);

// Fan-out destinations are interleaved device samples and
// destination_capacity is measured in those device samples, not frames. The
// mono source is borrowed directly from a native playback lease.
// f32 is replicated unchanged. Integer formats preserve the shipped contract:
// unordered input maps to zero; otherwise clamp [-1,1], multiply by 32767,
// truncate toward zero; u16 then adds 32768.
LFM_INTERNAL_API int lfm_playback_fanout_f32(
    const float *source, float *destination, size_t frames, uint32_t channels,
    size_t destination_capacity);
LFM_INTERNAL_API int lfm_playback_fanout_i16(
    const float *source, int16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity);
LFM_INTERNAL_API int lfm_playback_fanout_u16(
    const float *source, uint16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity);

#ifdef __cplusplus
}
#endif

#endif // LFM_CAPTURE_FORMAT_H
