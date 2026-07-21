// O(1) geometry validation for realtime capture conversion. Numerical work is
// architecture assembly only; there is intentionally no scalar C++ fallback.

#include "lfm_capture_format.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>

extern "C" void lfm_capture_downmix_f32_dd(const float *source,
                                            float *destination, size_t frames,
                                            uint32_t channels);
extern "C" void lfm_capture_downmix_i16_dd(const int16_t *source,
                                            float *destination, size_t frames,
                                            uint32_t channels);
extern "C" void lfm_capture_downmix_u16_dd(const uint16_t *source,
                                            float *destination, size_t frames,
                                            uint32_t channels);
extern "C" void lfm_playback_fanout_f32_dd(const float *source,
                                            float *destination, size_t frames,
                                            uint32_t channels);
extern "C" void lfm_playback_fanout_i16_dd(const float *source,
                                            int16_t *destination,
                                            size_t frames,
                                            uint32_t channels);
extern "C" void lfm_playback_fanout_u16_dd(const float *source,
                                            uint16_t *destination,
                                            size_t frames,
                                            uint32_t channels);

namespace {

template <typename Sample>
int validate(const Sample *source, float *destination, size_t frames,
             uint32_t channels, size_t destination_capacity) {
    if (channels == 0) {
        return -EINVAL;
    }
    if (frames > destination_capacity) {
        return -ENOSPC;
    }
    if (frames > std::numeric_limits<size_t>::max() /
                     static_cast<size_t>(channels)) {
        return -EOVERFLOW;
    }
    if (frames != 0 && (!source || !destination)) {
        return -EINVAL;
    }
    return 0;
}

template <typename Sample>
int validate_fanout(const float *source, Sample *destination, size_t frames,
                    uint32_t channels, size_t destination_capacity) {
    if (channels == 0) {
        return -EINVAL;
    }
    if (frames > std::numeric_limits<size_t>::max() /
                     static_cast<size_t>(channels)) {
        return -EOVERFLOW;
    }
    const size_t samples = frames * static_cast<size_t>(channels);
    if (samples > destination_capacity) {
        return -ENOSPC;
    }
    if (frames != 0 && (!source || !destination)) {
        return -EINVAL;
    }
    return 0;
}

} // namespace

extern "C" int lfm_capture_downmix_f32(const float *source,
                                         float *destination, size_t frames,
                                         uint32_t channels,
                                         size_t destination_capacity) {
    const int status =
        validate(source, destination, frames, channels, destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_capture_downmix_f32_dd(source, destination, frames, channels);
    return 0;
}

extern "C" int lfm_capture_downmix_i16(const int16_t *source,
                                         float *destination, size_t frames,
                                         uint32_t channels,
                                         size_t destination_capacity) {
    const int status =
        validate(source, destination, frames, channels, destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_capture_downmix_i16_dd(source, destination, frames, channels);
    return 0;
}

extern "C" int lfm_capture_downmix_u16(const uint16_t *source,
                                         float *destination, size_t frames,
                                         uint32_t channels,
                                         size_t destination_capacity) {
    const int status =
        validate(source, destination, frames, channels, destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_capture_downmix_u16_dd(source, destination, frames, channels);
    return 0;
}

extern "C" int lfm_playback_fanout_f32(
    const float *source, float *destination, size_t frames, uint32_t channels,
    size_t destination_capacity) {
    const int status = validate_fanout(source, destination, frames, channels,
                                      destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_playback_fanout_f32_dd(source, destination, frames, channels);
    return 0;
}

extern "C" int lfm_playback_fanout_i16(
    const float *source, int16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity) {
    const int status = validate_fanout(source, destination, frames, channels,
                                      destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_playback_fanout_i16_dd(source, destination, frames, channels);
    return 0;
}

extern "C" int lfm_playback_fanout_u16(
    const float *source, uint16_t *destination, size_t frames,
    uint32_t channels, size_t destination_capacity) {
    const int status = validate_fanout(source, destination, frames, channels,
                                      destination_capacity);
    if (status != 0 || frames == 0) {
        return status;
    }
    lfm_playback_fanout_u16_dd(source, destination, frames, channels);
    return 0;
}
