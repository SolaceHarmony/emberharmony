// Realtime-safe device-format boundary for native mono capture/playback views.
//
// Capture borrows the callback's complete interleaved block and writes directly
// into an already-claimed native mono destination. Playback borrows a native
// mono lease and writes directly into the callback's interleaved destination.
// Neither direction allocates storage or performs synchronization. C++ validates
// view geometry; every value-producing conversion, reduction, fan-out, and RMS
// operation runs in an architecture assembly leaf.
#ifndef LFM_CAPTURE_FORMAT_H
#define LFM_CAPTURE_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_PLAYBACK_METER_ABI 1u

// Compact callback-local signal state. One meter is reset at the beginning of
// a hardware output callback and then passed through every contiguous native
// lease rendered into that callback. The assembly leaf carries the f32
// reduction across leases and publishes the aggregate RMS; Rust never reduces
// signal values or takes a square root.
typedef struct LfmPlaybackMeterV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t rendered_frames;
    float sum_squares;
    float rms;
    uint64_t reserved[3];
} LfmPlaybackMeterV1;

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

// Reset a callback-local meter. Render destinations are interleaved device
// samples and destination_capacity is measured in those device samples, not
// frames. The mono source is borrowed directly from a native playback lease.
// f32 is replicated unchanged. Integer formats preserve the shipped contract:
// clamp [-1,1], multiply by 32767, truncate toward zero; u16 then adds 32768.
LFM_INTERNAL_API int lfm_playback_meter_reset(LfmPlaybackMeterV1 *meter);
LFM_INTERNAL_API int lfm_playback_render_f32(
    const float *source, float *destination, size_t frames, uint32_t channels,
    size_t destination_capacity, LfmPlaybackMeterV1 *meter);
LFM_INTERNAL_API int lfm_playback_render_i16(
    const float *source, int16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackMeterV1 *meter);
LFM_INTERNAL_API int lfm_playback_render_u16(
    const float *source, uint16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackMeterV1 *meter);

#ifdef __cplusplus
}
#endif

#endif // LFM_CAPTURE_FORMAT_H
