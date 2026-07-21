#include "lfm_platform_audio.h"

#include "lfm_audio_dock.h"
#include "lfm_platform_audio_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(__APPLE__)
#include <AudioUnit/AudioUnit.h>
#include <CoreAudio/CoreAudio.h>
#endif

namespace {

constexpr uint32_t READY_CAPACITY = 64;
constexpr size_t HOT_ATOMIC_BYTES = 128;
constexpr uint64_t CALLBACK_CLOSED = UINT64_C(1) << 63;
constexpr uint64_t CALLBACK_COUNT_MASK = CALLBACK_CLOSED - 1;
constexpr uint32_t ENDPOINTS_LIVE = 0;
constexpr uint32_t ENDPOINTS_RETIRING = 1;
constexpr uint32_t ENDPOINTS_RETIRED = 2;
constexpr uint32_t PLATFORM_CREATED = 0;
constexpr uint32_t PLATFORM_STARTING = 1;
constexpr uint32_t PLATFORM_STARTED = 2;
constexpr uint32_t PLATFORM_RETIRE_REQUESTED = 3;
constexpr uint32_t PLATFORM_RETIRING = 4;
constexpr uint32_t PLATFORM_RETIRED = 5;

template <typename T>
struct alignas(HOT_ATOMIC_BYTES) AudioCursor {
    std::atomic<T> value{0};
};

struct alignas(HOT_ATOMIC_BYTES) ReadyCell {
    LfmPcmLeaseV1 lease{};
};

struct ReadyRing {
    ReadyCell cells[READY_CAPACITY]{};
    AudioCursor<uint64_t> head;
    AudioCursor<uint64_t> tail;
};

static_assert(sizeof(AudioCursor<uint64_t>) == HOT_ATOMIC_BYTES);
static_assert(alignof(ReadyCell) == HOT_ATOMIC_BYTES);

bool ready_push(ReadyRing *ring, const LfmPcmLeaseV1 &lease) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == READY_CAPACITY) return false;
    ring->cells[tail % READY_CAPACITY].lease = lease;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool ready_pop(ReadyRing *ring, LfmPcmLeaseV1 *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->cells[head % READY_CAPACITY].lease;
    ring->cells[head % READY_CAPACITY].lease = {};
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

bool ready_peek(const ReadyRing *ring, LfmPcmLeaseV1 *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->cells[head % READY_CAPACITY].lease;
    return true;
}

} // namespace

struct LfmPlatformAudio {
    LfmSession *session = nullptr;
    LfmCaptureProducer *capture = nullptr;
    LfmPlaybackConsumer *playback = nullptr;
    LfmPlatformAudioConfigV1 config{};
    ReadyRing ready;
    LfmPcmLeaseV1 active{};
    uint32_t active_offset = 0;
    std::atomic<bool> started{false};
    std::atomic<bool> capture_enabled{true};
    std::atomic<bool> flush_pending{false};
    std::atomic<bool> retired{false};
    std::atomic<uint32_t> physical_state{PLATFORM_CREATED};
    AudioCursor<uint64_t> callback_gate;
    std::atomic<uint32_t> endpoints_state{ENDPOINTS_LIVE};
    std::atomic<int32_t> terminal_status{0};
    std::atomic<uint64_t> captured_frames{0};
    std::atomic<uint64_t> dropped_capture_frames{0};
    std::atomic<uint64_t> played_frames{0};
    std::atomic<uint64_t> silent_playback_frames{0};
    std::atomic<uint64_t> playback_leases{0};
    std::atomic<uint64_t> playback_releases{0};
    std::atomic<uint64_t> claimed_playback_frames{0};
    std::atomic<uint64_t> dropped_playback_frames{0};
    float *capture_discard = nullptr;
#if defined(__APPLE__)
    struct Listener {
        AudioObjectID object = kAudioObjectUnknown;
        AudioObjectPropertyAddress address{};
    };
    Listener listeners[12]{};
    uint32_t listener_count = 0;
    AudioUnit input = nullptr;
    AudioUnit output = nullptr;
#endif

    ~LfmPlatformAudio() {
        delete[] capture_discard;
    }
};

namespace {

int retire_endpoints_once(LfmPlatformAudio *audio);
void close_callback_admission(LfmPlatformAudio *audio);

int publish_physical_started(LfmPlatformAudio *audio) {
    uint32_t expected = PLATFORM_STARTING;
    if (audio->physical_state.compare_exchange_strong(
            expected, PLATFORM_STARTED, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return 0;
    }
    return expected == PLATFORM_RETIRE_REQUESTED
        ? LFM_STATUS_CANCELLED
        : LFM_STATUS_INTERNAL;
}

void signal_callback_drained(LfmPlatformAudio *audio) {
    if (audio) {
        lfm_internal_session_platform_retirement_ready(audio->session);
    }
}

bool enter_playback_callback(LfmPlatformAudio *audio) {
    if (!audio) return false;
    const uint64_t prior = audio->callback_gate.value.fetch_add(
        1, std::memory_order_acq_rel);
    if ((prior & CALLBACK_CLOSED) == 0) {
        if ((prior & CALLBACK_COUNT_MASK) == CALLBACK_COUNT_MASK) {
            std::abort();
        }
        return true;
    }
    audio->callback_gate.value.fetch_sub(1, std::memory_order_release);
    return false;
}

void leave_playback_callback(LfmPlatformAudio *audio) {
    const uint64_t prior = audio->callback_gate.value.fetch_sub(
        1, std::memory_order_acq_rel);
    const uint64_t count = prior & CALLBACK_COUNT_MASK;
    if (count == 0) std::abort();
    if ((prior & CALLBACK_CLOSED) != 0 && count == 1) {
        /* A hardware callback publishes only the successor edge. Endpoint
         * destruction takes lifecycle locks and therefore belongs to the
         * resumed coordinator, never this realtime stack. */
        signal_callback_drained(audio);
    }
}

struct CallbackLease {
    LfmPlatformAudio *audio;
    bool admitted;

    explicit CallbackLease(LfmPlatformAudio *value)
        : audio(value), admitted(enter_playback_callback(value)) {}

    ~CallbackLease() {
        if (admitted) leave_playback_callback(audio);
    }

    explicit operator bool() const { return admitted; }
};

void close_callback_admission(LfmPlatformAudio *audio) {
    if (!audio) return;
    audio->started.store(false, std::memory_order_release);
    audio->retired.store(true, std::memory_order_release);
    const uint64_t prior = audio->callback_gate.value.fetch_or(
        CALLBACK_CLOSED, std::memory_order_acq_rel);
    if ((prior & CALLBACK_COUNT_MASK) == 0) {
        signal_callback_drained(audio);
    }
}

void platform_fault(LfmPlatformAudio *audio, int32_t status) {
    if (!audio) return;
    int32_t expected = 0;
    audio->terminal_status.compare_exchange_strong(
        expected, status, std::memory_order_acq_rel,
        std::memory_order_acquire);
    lfm_internal_session_platform_fault(audio->session, status);
    close_callback_admission(audio);
}

int accept_playback(void *context, const LfmPcmLeaseV1 *lease) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!audio || !lease || !enter_playback_callback(audio)) {
        return LFM_STATUS_CANCELLED;
    }
    const bool published = ready_push(&audio->ready, *lease);
    if (!published) {
        audio->dropped_playback_frames.fetch_add(
            lease->frames, std::memory_order_relaxed);
        platform_fault(audio, LFM_STATUS_INTERNAL);
    }
    leave_playback_callback(audio);
    return published ? 0 : LFM_STATUS_INTERNAL;
}

int flush_context(void *context, uint64_t stream_epoch) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!audio || stream_epoch == 0) return LFM_STATUS_INVALID_ARGUMENT;
    /* Interrupt epochs are monotonic and concurrent interrupts coalesce in the
     * session before reaching this edge. One release publication is enough;
     * the next hardware callback owns the correlated queue transition. */
    audio->flush_pending.store(true, std::memory_order_release);
    return 0;
}

int retire_context(void *context) {
    return lfm_platform_audio_retire(
        static_cast<LfmPlatformAudio *>(context));
}

int finish_retirement_context(void *context) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!audio) return LFM_STATUS_INVALID_ARGUMENT;
    if (audio->callback_gate.value.load(std::memory_order_acquire) !=
        CALLBACK_CLOSED) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    return retire_endpoints_once(audio);
}

void destroy_context(void *context) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!audio) return;
    /* Session destruction owns the final handle. A concurrent STARTING or
     * RETIRING owner here is a lifecycle violation; deleting through it would
     * be a deterministic UAF, so fail hard rather than forging completion. */
    if (lfm_platform_audio_retire(audio) != 0) std::abort();
    delete audio;
}

int claim_next(LfmPlatformAudio *audio) {
    LfmPcmLeaseV1 ready{};
    if (!ready_pop(&audio->ready, &ready)) return LFM_STATUS_WOULD_BLOCK;
    LfmPcmLeaseV1 claimed{};
    const int status = lfm_playback_consumer_claim(
        audio->playback, &ready.ticket, ready.stream_epoch, ready.lease_id,
        ready.buffer_generation, &claimed);
    if (status != 0) {
        audio->dropped_playback_frames.fetch_add(
            ready.frames, std::memory_order_relaxed);
        return status;
    }
    audio->active = claimed;
    audio->active_offset = 0;
    audio->playback_leases.fetch_add(1, std::memory_order_relaxed);
    audio->claimed_playback_frames.fetch_add(
        claimed.frames, std::memory_order_relaxed);
    return 0;
}

void release_active(LfmPlatformAudio *audio, bool dropped = false) {
    if (!audio || audio->active.lease_id == 0) return;
    if (dropped && audio->active_offset < audio->active.frames) {
        audio->dropped_playback_frames.fetch_add(
            audio->active.frames - audio->active_offset,
            std::memory_order_relaxed);
    }
    const int status = lfm_playback_consumer_release(
        audio->playback, &audio->active);
    if (status != 0 && status != LFM_STATUS_STALE &&
        status != LFM_STATUS_CANCELLED) {
        platform_fault(audio, status);
    }
    audio->active = {};
    audio->active_offset = 0;
    audio->playback_releases.fetch_add(1, std::memory_order_relaxed);
}

#if defined(__APPLE__)

int os_status(OSStatus status) {
    return status == noErr ? 0 : LFM_STATUS_HOST_SINK;
}

int default_device(AudioObjectPropertySelector selector,
                   AudioDeviceID *out) {
    AudioObjectPropertyAddress address = {
        selector,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    UInt32 bytes = sizeof(*out);
    const OSStatus status = AudioObjectGetPropertyData(
        kAudioObjectSystemObject, &address, 0, nullptr, &bytes, out);
    if (status != noErr || *out == kAudioObjectUnknown) {
        return LFM_STATUS_UNSUPPORTED;
    }
    return 0;
}

int device_rate(AudioDeviceID device, bool input, uint32_t *out) {
    AudioObjectPropertyAddress address = {
        kAudioDevicePropertyNominalSampleRate,
        input ? kAudioDevicePropertyScopeInput
              : kAudioDevicePropertyScopeOutput,
        kAudioObjectPropertyElementMain,
    };
    Float64 rate = 0;
    UInt32 bytes = sizeof(rate);
    const OSStatus status = AudioObjectGetPropertyData(
        device, &address, 0, nullptr, &bytes, &rate);
    if (status != noErr || !std::isfinite(rate) || rate < 8000.0 ||
        rate > 192000.0 || rate > UINT32_MAX) {
        return LFM_STATUS_UNSUPPORTED;
    }
    *out = static_cast<uint32_t>(rate + 0.5);
    return 0;
}

int device_frames(AudioDeviceID device, uint32_t *out) {
    AudioObjectPropertyAddress address = {
        kAudioDevicePropertyBufferFrameSize,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    UInt32 frames = 0;
    UInt32 bytes = sizeof(frames);
    const OSStatus status = AudioObjectGetPropertyData(
        device, &address, 0, nullptr, &bytes, &frames);
    if (status != noErr || frames == 0) return LFM_STATUS_UNSUPPORTED;
    const AudioObjectPropertyAddress variable = {
        kAudioDevicePropertyUsesVariableBufferFrameSizes,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    if (AudioObjectHasProperty(device, &variable)) {
        UInt32 maximum = 0;
        bytes = sizeof(maximum);
        const OSStatus variable_status = AudioObjectGetPropertyData(
            device, &variable, 0, nullptr, &bytes, &maximum);
        if (variable_status != noErr || maximum == 0) {
            return LFM_STATUS_UNSUPPORTED;
        }
        frames = std::max(frames, maximum);
    }
    *out = frames;
    return 0;
}

bool same_address(const AudioObjectPropertyAddress &left,
                  const AudioObjectPropertyAddress &right) {
    return left.mSelector == right.mSelector &&
           left.mScope == right.mScope &&
           left.mElement == right.mElement;
}

OSStatus property_changed(AudioObjectID, UInt32, const AudioObjectPropertyAddress *,
                          void *context) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    CallbackLease callback(audio);
    if (!callback) return noErr;
    platform_fault(audio, LFM_STATUS_HOST_SINK);
    return noErr;
}

int add_listener(LfmPlatformAudio *audio, AudioObjectID object,
                 AudioObjectPropertyAddress address, bool required) {
    if (!AudioObjectHasProperty(object, &address)) {
        return required ? LFM_STATUS_UNSUPPORTED : 0;
    }
    for (uint32_t index = 0; index < audio->listener_count; ++index) {
        if (audio->listeners[index].object == object &&
            same_address(audio->listeners[index].address, address)) {
            return 0;
        }
    }
    constexpr size_t capacity =
        sizeof(audio->listeners) / sizeof(audio->listeners[0]);
    if (audio->listener_count == capacity) {
        return LFM_STATUS_INTERNAL;
    }
    const OSStatus status = AudioObjectAddPropertyListener(
        object, &address, property_changed, audio);
    if (status != noErr) return os_status(status);
    audio->listeners[audio->listener_count++] = {object, address};
    return 0;
}

void remove_listeners(LfmPlatformAudio *audio) {
    while (audio && audio->listener_count != 0) {
        const LfmPlatformAudio::Listener listener =
            audio->listeners[--audio->listener_count];
        (void)AudioObjectRemovePropertyListener(
            listener.object, &listener.address, property_changed, audio);
    }
}

int install_listeners(LfmPlatformAudio *audio) {
    const AudioObjectPropertyAddress default_input = {
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress default_output = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress alive = {
        kAudioDevicePropertyDeviceIsAlive,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress stopped = {
        kAudioDevicePropertyIOStoppedAbnormally,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress rate = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress frames = {
        kAudioDevicePropertyBufferFrameSize,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    const AudioObjectPropertyAddress variable = {
        kAudioDevicePropertyUsesVariableBufferFrameSizes,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    int status = add_listener(audio, kAudioObjectSystemObject,
                              default_input, true);
    if (status == 0) {
        status = add_listener(audio, kAudioObjectSystemObject,
                              default_output, true);
    }
    for (AudioDeviceID device : {audio->config.capture_device,
                                 audio->config.playback_device}) {
        if (status == 0) status = add_listener(audio, device, alive, true);
        if (status == 0) status = add_listener(audio, device, stopped, false);
        if (status == 0) status = add_listener(audio, device, rate, true);
        if (status == 0) status = add_listener(audio, device, frames, true);
        if (status == 0) status = add_listener(audio, device, variable, false);
    }
    if (status != 0) remove_listeners(audio);
    return status;
}

AudioStreamBasicDescription mono_f32(uint32_t rate) {
    AudioStreamBasicDescription format{};
    format.mSampleRate = rate;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsFloat |
                          kAudioFormatFlagIsPacked |
                          kAudioFormatFlagsNativeEndian;
    format.mBytesPerPacket = sizeof(float);
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = sizeof(float);
    format.mChannelsPerFrame = 1;
    format.mBitsPerChannel = 32;
    return format;
}

void output_fill(LfmPlatformAudio *audio, float *destination,
                 uint32_t frames) {
    if (!audio || !destination || frames == 0) return;
    std::memset(destination, 0, static_cast<size_t>(frames) * sizeof(float));

    const bool flush =
        audio->flush_pending.exchange(false, std::memory_order_acq_rel);
    if (flush) {
        const uint64_t flush_epoch =
            lfm_internal_session_epoch(audio->session);
        if (audio->active.lease_id != 0 &&
            audio->active.stream_epoch < flush_epoch) {
            release_active(audio, true);
        }
        LfmPcmLeaseV1 queued{};
        while (ready_peek(&audio->ready, &queued) &&
               queued.stream_epoch < flush_epoch) {
            const int status = claim_next(audio);
            if (status == 0) {
                release_active(audio, true);
                continue;
            }
            if (status == LFM_STATUS_STALE ||
                status == LFM_STATUS_CANCELLED) {
                continue;
            }
            platform_fault(audio, status);
            break;
        }
        LfmPlaybackRenderV1 report{};
        const int published = lfm_internal_playback_consumer_publish_flush(
            audio->playback, flush_epoch, &report);
        /* A newer concurrent interrupt may advance the epoch between the
         * exchange and this publication. Its monotonic flush value remains
         * armed for the next callback, so this older record is simply stale. */
        if (published != 0 && published != LFM_STATUS_STALE &&
            published != LFM_STATUS_CANCELLED) {
            platform_fault(audio, published);
        }
    }

    uint32_t written = 0;
    uint32_t transitions = 0;
    while (written < frames && transitions <= READY_CAPACITY) {
        if (audio->active.lease_id == 0) {
            const int claim = claim_next(audio);
            transitions++;
            if (claim == LFM_STATUS_WOULD_BLOCK ||
                claim == LFM_STATUS_CANCELLED) {
                break;
            }
            if (claim == LFM_STATUS_STALE) continue;
            if (claim != 0) {
                platform_fault(audio, claim);
                break;
            }
        }
        const uint32_t remaining =
            audio->active.frames - audio->active_offset;
        const uint32_t count = std::min(frames - written, remaining);
        LfmPlaybackRenderV1 report{};
        const int status = lfm_playback_consumer_render_f32(
            audio->playback, &audio->active, audio->active_offset,
            destination + written, count, 1, frames - written, &report);
        if (status != 0) {
            /* Fanout precedes evidence publication. If the correlated edge
             * loses an epoch/stop/capacity race, scrub that copied prefix so
             * the device and telemetry agree that it was dropped, not played. */
            std::memset(destination + written, 0,
                        static_cast<size_t>(count) * sizeof(float));
            if (status != LFM_STATUS_STALE &&
                status != LFM_STATUS_CANCELLED) {
                platform_fault(audio, status);
            }
            release_active(audio, true);
            break;
        }
        written += count;
        audio->active_offset += count;
        audio->played_frames.fetch_add(count, std::memory_order_relaxed);
        if (audio->active_offset == audio->active.frames) {
            release_active(audio);
        }
    }
    if (written < frames) {
        LfmPlaybackRenderV1 report{};
        (void)lfm_playback_consumer_observe(
            audio->playback, nullptr, 0, frames - written,
            LFM_PLAYBACK_EVIDENCE_SILENCE, &report);
        audio->silent_playback_frames.fetch_add(
            frames - written, std::memory_order_relaxed);
    }
}

OSStatus render_input(LfmPlatformAudio *audio,
                      AudioUnitRenderActionFlags *flags,
                      const AudioTimeStamp *timestamp, UInt32 frames,
                      float *destination) {
    AudioBufferList list{};
    list.mNumberBuffers = 1;
    list.mBuffers[0].mNumberChannels = 1;
    list.mBuffers[0].mDataByteSize = frames * sizeof(float);
    list.mBuffers[0].mData = destination;
    return AudioUnitRender(audio->input, flags, timestamp, 1, frames, &list);
}

OSStatus input_callback(void *context, AudioUnitRenderActionFlags *flags,
                        const AudioTimeStamp *timestamp, UInt32,
                        UInt32 frames, AudioBufferList *) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!audio || frames == 0) {
        return noErr;
    }
    CallbackLease callback(audio);
    if (!callback || !audio->started.load(std::memory_order_acquire))
        return noErr;
    if (frames > audio->config.capture_callback_frames) {
        platform_fault(audio, LFM_STATUS_HOST_SINK);
        return kAudio_ParamError;
    }

    if (!audio->capture_enabled.load(std::memory_order_acquire)) {
        LfmCaptureChunkV1 gap{};
        const OSStatus rendered = render_input(
            audio, flags, timestamp, frames, audio->capture_discard);
        const int status = lfm_capture_producer_publish_gap(
            audio->capture, frames, 1,
            LFM_CAPTURE_CHUNK_GAP | LFM_CAPTURE_CHUNK_MUTED, &gap);
        if (rendered != noErr || (status != 0 &&
                                  status != LFM_STATUS_WOULD_BLOCK &&
                                  status != LFM_STATUS_CANCELLED)) {
            platform_fault(audio, rendered != noErr
                                      ? os_status(rendered)
                                      : status);
        }
        return rendered;
    }

    LfmCaptureChunkV1 chunk{};
    int status = lfm_capture_producer_claim_chunk(
        audio->capture, frames, audio->config.capture_sample_rate, 1, 0,
        &chunk);
    if (status != 0) {
        const OSStatus rendered = render_input(
            audio, flags, timestamp, frames, audio->capture_discard);
        LfmCaptureChunkV1 gap{};
        const int gap_status = lfm_capture_producer_publish_gap(
            audio->capture, frames, 1,
            LFM_CAPTURE_CHUNK_GAP | LFM_CAPTURE_CHUNK_XRUN, &gap);
        audio->dropped_capture_frames.fetch_add(
            frames, std::memory_order_relaxed);
        if (rendered != noErr ||
            (gap_status != 0 && gap_status != LFM_STATUS_WOULD_BLOCK &&
             gap_status != LFM_STATUS_CANCELLED)) {
            platform_fault(audio, rendered != noErr
                                      ? os_status(rendered)
                                      : gap_status);
        }
        return rendered;
    }

    LfmMutableF32SpanV1 spans[2]{};
    uint32_t span_count = 0;
    status = lfm_capture_producer_resolve_chunk(
        audio->capture, &chunk, spans, &span_count);
    if (status != 0 || span_count != 1 || spans[0].count != frames) {
        (void)lfm_capture_producer_abort_chunk(audio->capture, &chunk);
        platform_fault(audio, status != 0 ? status : LFM_STATUS_INTERNAL);
        return kAudio_ParamError;
    }

    const OSStatus rendered = render_input(
        audio, flags, timestamp, frames, spans[0].data);
    if (rendered != noErr) {
        (void)lfm_capture_producer_abort_chunk(audio->capture, &chunk);
        LfmCaptureChunkV1 gap{};
        (void)lfm_capture_producer_publish_gap(
            audio->capture, frames, 1,
            LFM_CAPTURE_CHUNK_GAP | LFM_CAPTURE_CHUNK_XRUN, &gap);
        audio->dropped_capture_frames.fetch_add(
            frames, std::memory_order_relaxed);
        platform_fault(audio, os_status(rendered));
        return rendered;
    }

    status = lfm_capture_producer_commit_chunk(audio->capture, &chunk);
    if (status != 0) {
        (void)lfm_capture_producer_abort_chunk(audio->capture, &chunk);
        if (status == LFM_STATUS_CANCELLED || status == LFM_STATUS_STALE ||
            audio->retired.load(std::memory_order_acquire)) {
            return noErr;
        }
        platform_fault(audio, status);
        return kAudio_ParamError;
    }
    audio->captured_frames.fetch_add(frames, std::memory_order_relaxed);
    return noErr;
}

OSStatus output_callback(void *context, AudioUnitRenderActionFlags *,
                         const AudioTimeStamp *, UInt32, UInt32 frames,
                         AudioBufferList *buffers) {
    auto *audio = static_cast<LfmPlatformAudio *>(context);
    if (!buffers) return kAudio_ParamError;
    for (UInt32 index = 0; index < buffers->mNumberBuffers; ++index) {
        AudioBuffer &buffer = buffers->mBuffers[index];
        if (buffer.mData && buffer.mDataByteSize != 0) {
            std::memset(buffer.mData, 0, buffer.mDataByteSize);
        }
    }
    CallbackLease callback(audio);
    if (!callback || !audio->started.load(std::memory_order_acquire)) {
        return noErr;
    }
    if (frames == 0 || frames > audio->config.playback_callback_frames ||
        buffers->mNumberBuffers != 1 ||
        buffers->mBuffers[0].mNumberChannels != 1 ||
        !buffers->mBuffers[0].mData ||
        buffers->mBuffers[0].mDataByteSize < frames * sizeof(float)) {
        platform_fault(audio, LFM_STATUS_HOST_SINK);
        return kAudio_ParamError;
    }
    output_fill(audio, static_cast<float *>(buffers->mBuffers[0].mData),
                frames);
    buffers->mBuffers[0].mDataByteSize = frames * sizeof(float);
    return noErr;
}

int unit_property(AudioUnit unit, AudioUnitPropertyID property,
                  AudioUnitScope scope, AudioUnitElement element,
                  const void *value, UInt32 bytes) {
    return os_status(AudioUnitSetProperty(
        unit, property, scope, element, value, bytes));
}

int create_input_unit(LfmPlatformAudio *audio) {
    const AudioComponentDescription description = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_HALOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent component = AudioComponentFindNext(nullptr, &description);
    if (!component) return LFM_STATUS_UNSUPPORTED;
    OSStatus status = AudioComponentInstanceNew(component, &audio->input);
    if (status != noErr) return os_status(status);
    const UInt32 enabled = 1;
    const UInt32 disabled = 0;
    int result = unit_property(
        audio->input, kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Input, 1, &enabled, sizeof(enabled));
    if (result == 0) {
        result = unit_property(
            audio->input, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output, 0, &disabled, sizeof(disabled));
    }
    if (result == 0) {
        result = unit_property(
            audio->input, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0, &audio->config.capture_device,
            sizeof(audio->config.capture_device));
    }
    AudioStreamBasicDescription format =
        mono_f32(audio->config.capture_sample_rate);
    if (result == 0) {
        result = unit_property(
            audio->input, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output, 1, &format, sizeof(format));
    }
    const AURenderCallbackStruct callback = {
        .inputProc = input_callback,
        .inputProcRefCon = audio,
    };
    if (result == 0) {
        result = unit_property(
            audio->input, kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global, 0, &callback, sizeof(callback));
    }
    if (result == 0) result = os_status(AudioUnitInitialize(audio->input));
    return result;
}

int create_output_unit(LfmPlatformAudio *audio) {
    const AudioComponentDescription description = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_HALOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent component = AudioComponentFindNext(nullptr, &description);
    if (!component) return LFM_STATUS_UNSUPPORTED;
    OSStatus status = AudioComponentInstanceNew(component, &audio->output);
    if (status != noErr) return os_status(status);
    const UInt32 enabled = 1;
    const UInt32 disabled = 0;
    int result = unit_property(
        audio->output, kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Output, 0, &enabled, sizeof(enabled));
    if (result == 0) {
        result = unit_property(
            audio->output, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input, 1, &disabled, sizeof(disabled));
    }
    if (result == 0) {
        result = unit_property(
            audio->output, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0, &audio->config.playback_device,
            sizeof(audio->config.playback_device));
    }
    AudioStreamBasicDescription format =
        mono_f32(audio->config.playback_sample_rate);
    if (result == 0) {
        result = unit_property(
            audio->output, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input, 0, &format, sizeof(format));
    }
    const AURenderCallbackStruct callback = {
        .inputProc = output_callback,
        .inputProcRefCon = audio,
    };
    if (result == 0) {
        result = unit_property(
            audio->output, kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input, 0, &callback, sizeof(callback));
    }
    if (result == 0) result = os_status(AudioUnitInitialize(audio->output));
    return result;
}

void dispose_units(LfmPlatformAudio *audio) {
    if (audio->input) {
        (void)AudioOutputUnitStop(audio->input);
        (void)AudioUnitUninitialize(audio->input);
        (void)AudioComponentInstanceDispose(audio->input);
        audio->input = nullptr;
    }
    if (audio->output) {
        (void)AudioOutputUnitStop(audio->output);
        (void)AudioUnitUninitialize(audio->output);
        (void)AudioComponentInstanceDispose(audio->output);
        audio->output = nullptr;
    }
}

int create_units(LfmPlatformAudio *audio) {
    int status = create_input_unit(audio);
    if (status == 0) status = create_output_unit(audio);
    return status;
}

#endif /* __APPLE__ */

int retire_endpoints(LfmPlatformAudio *audio) {
    if (!audio) return LFM_STATUS_INVALID_ARGUMENT;
    release_active(audio, true);
    LfmPcmLeaseV1 ready{};
    while (ready_pop(&audio->ready, &ready)) {
        /* Callback admission is closed and no hardware consumer can observe
         * these records now. The authoritative FIFO is drained below so both
         * accepted and acceptance-race publications retire in one order. */
    }
    uint64_t dropped = 0;
    if (audio->playback) {
        const int status = lfm_internal_playback_consumer_discard_all(
            audio->playback, &dropped);
        if (status != 0) return status;
        audio->dropped_playback_frames.fetch_add(
            dropped, std::memory_order_relaxed);
    }
    if (audio->capture) {
        const int status = lfm_capture_producer_destroy(audio->capture);
        if (status != 0) return status;
        audio->capture = nullptr;
    }
    if (audio->playback) {
        const int status = lfm_playback_consumer_destroy(audio->playback);
        if (status != 0) return status;
        audio->playback = nullptr;
    }
    return 0;
}

int retire_endpoints_once(LfmPlatformAudio *audio) {
    if (!audio) return LFM_STATUS_INVALID_ARGUMENT;
    uint32_t expected = ENDPOINTS_LIVE;
    if (!audio->endpoints_state.compare_exchange_strong(
            expected, ENDPOINTS_RETIRING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return expected == ENDPOINTS_RETIRED ? 0 : LFM_STATUS_WOULD_BLOCK;
    }
    const int status = retire_endpoints(audio);
    audio->endpoints_state.store(
        status == 0 ? ENDPOINTS_RETIRED : ENDPOINTS_LIVE,
        std::memory_order_release);
    return status;
}

} // namespace

extern "C" {

int lfm_platform_audio_default_config(LfmPlatformAudioConfigV1 *out) {
    if (!out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = {
        .size = sizeof(LfmPlatformAudioConfigV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
    };
#if defined(__APPLE__)
    AudioDeviceID input = kAudioObjectUnknown;
    AudioDeviceID output = kAudioObjectUnknown;
    int status = default_device(kAudioHardwarePropertyDefaultInputDevice,
                                &input);
    if (status != 0) return status;
    status = default_device(kAudioHardwarePropertyDefaultOutputDevice,
                            &output);
    if (status != 0) return status;
    uint32_t input_rate = 0;
    uint32_t output_rate = 0;
    status = device_rate(input, true, &input_rate);
    if (status != 0) return status;
    status = device_rate(output, false, &output_rate);
    if (status != 0) return status;
    uint32_t capture_frames = 0;
    uint32_t playback_frames = 0;
    status = device_frames(input, &capture_frames);
    if (status != 0) return status;
    status = device_frames(output, &playback_frames);
    if (status != 0) return status;
    out->capture_device = input;
    out->playback_device = output;
    out->capture_sample_rate = input_rate;
    out->playback_sample_rate = output_rate;
    out->capture_callback_frames = capture_frames;
    out->playback_callback_frames = playback_frames;
    return 0;
#else
    return LFM_STATUS_UNSUPPORTED;
#endif
}

int lfm_platform_audio_create(
    LfmSession *session, const LfmPlatformAudioConfigV1 *config,
    LfmPlatformAudio **out) {
    if (!session || !config || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (config->size != sizeof(*config) ||
        config->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
#if !defined(__APPLE__)
    return LFM_STATUS_UNSUPPORTED;
#else
    LfmPlatformAudioConfigV1 current{};
    int status = lfm_platform_audio_default_config(&current);
    if (status != 0) return status;
    if (current.capture_device != config->capture_device ||
        current.playback_device != config->playback_device ||
        current.capture_sample_rate != config->capture_sample_rate ||
        current.playback_sample_rate != config->playback_sample_rate ||
        current.capture_callback_frames !=
            config->capture_callback_frames ||
        current.playback_callback_frames !=
            config->playback_callback_frames) {
        return LFM_STATUS_STALE;
    }
    auto *audio = new (std::nothrow) LfmPlatformAudio();
    if (!audio) return LFM_STATUS_OUT_OF_MEMORY;
    audio->session = session;
    audio->config = *config;
    audio->capture_discard = new (std::nothrow)
        float[config->capture_callback_frames];
    if (!audio->capture_discard) {
        delete audio;
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    status = lfm_capture_chunk_producer_create(
        session, 1, 0, &audio->capture);
    if (status == 0) {
        status = lfm_playback_consumer_create(session, &audio->playback);
    }
    if (status == 0) status = create_units(audio);
    const LfmPlatformAudioBindingV1 binding = {
        .size = sizeof(LfmPlatformAudioBindingV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .context = audio,
        .playback_ready = accept_playback,
        .playback_flush = flush_context,
        .retire_context = retire_context,
        .finish_retirement = finish_retirement_context,
        .destroy_context = destroy_context,
    };
    if (status == 0) {
        status = lfm_internal_session_bind_platform_audio(
            session, config, &binding);
    }
    if (status != 0) {
        lfm_session_request_stop(session);
        dispose_units(audio);
        retire_endpoints_once(audio);
        delete audio;
        return status;
    }
    *out = audio;
    return 0;
#endif
}

int lfm_platform_audio_start(LfmPlatformAudio *audio) {
    if (!audio || audio->retired.load(std::memory_order_acquire)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
#if !defined(__APPLE__)
    return LFM_STATUS_UNSUPPORTED;
#else
    uint32_t expected = PLATFORM_CREATED;
    if (!audio->physical_state.compare_exchange_strong(
            expected, PLATFORM_STARTING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_BUSY;
    }
    int setup = install_listeners(audio);
    LfmPlatformAudioConfigV1 current{};
    if (setup == 0) setup = lfm_platform_audio_default_config(&current);
    if (setup == 0 &&
        (current.capture_device != audio->config.capture_device ||
         current.playback_device != audio->config.playback_device ||
         current.capture_sample_rate != audio->config.capture_sample_rate ||
         current.playback_sample_rate != audio->config.playback_sample_rate ||
         current.capture_callback_frames !=
             audio->config.capture_callback_frames ||
         current.playback_callback_frames !=
             audio->config.playback_callback_frames)) {
        setup = LFM_STATUS_STALE;
    }
    if (setup != 0 || audio->retired.load(std::memory_order_acquire)) {
        platform_fault(audio, setup != 0 ? setup : LFM_STATUS_HOST_SINK);
#if defined(__APPLE__)
        remove_listeners(audio);
        dispose_units(audio);
#endif
        audio->physical_state.store(PLATFORM_RETIRED,
                                    std::memory_order_release);
        return setup != 0 ? setup : LFM_STATUS_HOST_SINK;
    }
    OSStatus status = AudioOutputUnitStart(audio->output);
    if (status == noErr) status = AudioOutputUnitStart(audio->input);
    if (status != noErr) {
        (void)AudioOutputUnitStop(audio->output);
        (void)AudioOutputUnitStop(audio->input);
        platform_fault(audio, os_status(status));
        remove_listeners(audio);
        dispose_units(audio);
        audio->physical_state.store(PLATFORM_RETIRED,
                                    std::memory_order_release);
        return os_status(status);
    }
    audio->started.store(true, std::memory_order_release);
    const int published = publish_physical_started(audio);
    if (published != 0) {
        /* Retirement claims STARTING by changing it to RETIRE_REQUESTED.
         * That state change is the terminal edge: start may finish the
         * CoreAudio call already on its stack, but it cannot publish STARTED.
         * The start owner also owns physical cleanup, so no second thread can
         * dispose units underneath it. */
        audio->started.store(false, std::memory_order_release);
        (void)AudioOutputUnitStop(audio->output);
        (void)AudioOutputUnitStop(audio->input);
        remove_listeners(audio);
        dispose_units(audio);
        if (published != LFM_STATUS_CANCELLED) {
            platform_fault(audio, LFM_STATUS_INTERNAL);
        }
        audio->physical_state.store(PLATFORM_RETIRED,
                                    std::memory_order_release);
        return published;
    }
    return 0;
#endif
}

int lfm_platform_audio_set_capture_enabled(LfmPlatformAudio *audio,
                                           uint32_t enabled) {
    if (!audio || enabled > 1 ||
        audio->retired.load(std::memory_order_acquire)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    audio->capture_enabled.store(enabled != 0, std::memory_order_release);
    return 0;
}

int lfm_platform_audio_retire(LfmPlatformAudio *audio) {
    if (!audio) return LFM_STATUS_INVALID_ARGUMENT;
    /* Retiring the only physical capture/playback endpoint is a terminal
     * session transition. Publish that edge here so callers cannot close the
     * hardware and accidentally leave a live coordinator with no successor. */
    lfm_session_request_stop(audio->session);
    close_callback_admission(audio);
    uint32_t state = audio->physical_state.load(std::memory_order_acquire);
    if (state == PLATFORM_RETIRED) return 0;
    if (state == PLATFORM_RETIRE_REQUESTED || state == PLATFORM_RETIRING) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (state == PLATFORM_STARTING) {
        if (audio->physical_state.compare_exchange_strong(
                state, PLATFORM_RETIRE_REQUESTED,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            /* The in-flight start call is the causal successor and owns unit
             * disposal. No thread waits for it and no competing disposer is
             * admitted. */
            return LFM_STATUS_WOULD_BLOCK;
        }
        if (state == PLATFORM_RETIRED) return 0;
        if (state == PLATFORM_RETIRE_REQUESTED ||
            state == PLATFORM_RETIRING) {
            return LFM_STATUS_WOULD_BLOCK;
        }
    }
    if (state != PLATFORM_CREATED && state != PLATFORM_STARTED) {
        return LFM_STATUS_INTERNAL;
    }
    if (!audio->physical_state.compare_exchange_strong(
            state, PLATFORM_RETIRING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return state == PLATFORM_RETIRED ? 0 : LFM_STATUS_WOULD_BLOCK;
    }
#if defined(__APPLE__)
    remove_listeners(audio);
    dispose_units(audio);
#endif
    audio->physical_state.store(PLATFORM_RETIRED,
                                std::memory_order_release);
    return 0;
}

int lfm_internal_platform_audio_callback_retirement_test(void) {
    LfmPlatformAudio audio{};
    if (!enter_playback_callback(&audio)) return LFM_STATUS_INTERNAL;
    const int status = lfm_platform_audio_retire(&audio);
    if (status != 0 ||
        audio.endpoints_state.load(std::memory_order_acquire) !=
            ENDPOINTS_LIVE) {
        return LFM_STATUS_INTERNAL;
    }
    leave_playback_callback(&audio);
    if (audio.endpoints_state.load(std::memory_order_acquire) !=
            ENDPOINTS_LIVE ||
        audio.callback_gate.value.load(std::memory_order_acquire) !=
            CALLBACK_CLOSED) {
        return LFM_STATUS_INTERNAL;
    }
    if (finish_retirement_context(&audio) != 0 ||
        audio.endpoints_state.load(std::memory_order_acquire) !=
            ENDPOINTS_RETIRED) {
        return LFM_STATUS_INTERNAL;
    }
    LfmPlatformAudio starting{};
    starting.physical_state.store(PLATFORM_STARTING,
                                  std::memory_order_release);
    if (lfm_platform_audio_retire(&starting) != LFM_STATUS_WOULD_BLOCK ||
        starting.physical_state.load(std::memory_order_acquire) !=
            PLATFORM_RETIRE_REQUESTED ||
        publish_physical_started(&starting) != LFM_STATUS_CANCELLED ||
        starting.physical_state.load(std::memory_order_acquire) !=
            PLATFORM_RETIRE_REQUESTED) {
        return LFM_STATUS_INTERNAL;
    }
    /* The in-flight start call owns this terminal publication after disposing
     * its physical units. A later administrative retire observes completion. */
    starting.physical_state.store(PLATFORM_RETIRED,
                                  std::memory_order_release);
    if (lfm_platform_audio_retire(&starting) != 0) {
        return LFM_STATUS_INTERNAL;
    }
    return 0;
}

int lfm_platform_audio_snapshot(const LfmPlatformAudio *audio,
                                LfmPlatformAudioSnapshotV1 *out) {
    if (!audio || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = {
        .size = sizeof(LfmPlatformAudioSnapshotV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .started = audio->started.load(std::memory_order_acquire) ? 1u : 0u,
        .capture_enabled =
            audio->capture_enabled.load(std::memory_order_acquire) ? 1u : 0u,
        .terminal_status =
            audio->terminal_status.load(std::memory_order_acquire),
        .reserved0 = 0,
        .captured_frames =
            audio->captured_frames.load(std::memory_order_relaxed),
        .dropped_capture_frames =
            audio->dropped_capture_frames.load(std::memory_order_relaxed),
        .played_frames =
            audio->played_frames.load(std::memory_order_relaxed),
        .silent_playback_frames =
            audio->silent_playback_frames.load(std::memory_order_relaxed),
        .playback_leases =
            audio->playback_leases.load(std::memory_order_relaxed),
        .playback_releases =
            audio->playback_releases.load(std::memory_order_relaxed),
        .claimed_playback_frames =
            audio->claimed_playback_frames.load(std::memory_order_relaxed),
        .dropped_playback_frames =
            audio->dropped_playback_frames.load(std::memory_order_relaxed),
    };
    return 0;
}

} /* extern "C" */
