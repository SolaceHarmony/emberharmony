#include "kcoro_stackless.h"
#include "lfm_audio_dock.h"
#include "lfm_detokenizer.h"
#include "lfm_detokenizer_program.h"
#include "lfm_runtime.h"
#include "../src/runtime/lfm_runtime_diagnostics.hpp"
#include "lfm_runtime_internal.h"
#include "lfm_safetensors.h"
#include "lfm_session.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>

#if defined(__APPLE__)
#include <AudioUnit/AudioUnit.h>
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#endif

namespace {

constexpr uint32_t ABI = LFM_RUNTIME_ABI_VERSION;
constexpr uint32_t RATE = 24000;
constexpr uint32_t CALLBACK_FRAMES = 480;
constexpr uint32_t EVENT_CAPACITY = 128;
constexpr uint32_t EVENT_PAYLOAD = 512;
constexpr uint32_t EVENT_BUDGET = 24;
/* Production interleaved-turn budget. This is deliberately much larger than
 * the vendor demo's 1024-step guard: the model owns its terminal, and this
 * gate must expose—not manufacture—a cutoff before that terminal. The bound
 * remains below the model card's 32,768-token conversation context. */
constexpr uint32_t MAX_TOKENS = 8192;
constexpr uint64_t CLOSED_LOOP_CAPACITY = UINT64_C(30) * RATE;
constexpr uint64_t MONITOR_CAPACITY = UINT64_C(1) << 20;
constexpr uint64_t MONITOR_CALLBACK_CLOSED = UINT64_C(1) << 63;
constexpr uint64_t MONITOR_CALLBACK_COUNT = MONITOR_CALLBACK_CLOSED - 1;
constexpr uint64_t WATCHDOG_NS = UINT64_C(120) * UINT64_C(1000000000);
constexpr uint64_t FNV_OFFSET = UINT64_C(1469598103934665603);
constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);
constexpr uint32_t GATE_CONTINUATION_DONE = 1u << 0;
constexpr uint32_t GATE_WATCHDOG_DONE = 1u << 1;
constexpr uint32_t GATE_ALL_DONE =
    GATE_CONTINUATION_DONE | GATE_WATCHDOG_DONE;

void copy_error(char *destination, size_t capacity, const char *source);

struct Gate;

struct AudibleMonitor {
    Gate *gate = nullptr;
    float *samples = nullptr;
    alignas(128) std::atomic<uint64_t> head{0};
    alignas(128) std::atomic<uint64_t> tail{0};
    alignas(128) std::atomic<uint64_t> callback_gate{0};
    std::atomic<uint64_t> underflow_frames{0};
    std::atomic<uint64_t> underflow_callbacks{0};
    std::atomic<bool> closing{false};
    std::atomic<bool> drained{false};
    std::atomic<bool> active{false};
    std::atomic<bool> started{false};
#if defined(__APPLE__)
    AudioUnit output = nullptr;
#endif
    uint32_t source = 0;
    uint32_t source_transitions = 0;
    bool streaming = false;
    bool enabled = false;
};

struct GateEvent {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t session_id = 0;
    uint64_t epoch = 0;
    LfmTicketIdV1 ticket{};
    uint32_t payload_bytes = 0;
    int32_t status = 0;
    unsigned char payload[EVENT_PAYLOAD]{};
};

struct alignas(128) EventCursor {
    std::atomic<uint64_t> value{0};
};

struct EventRing {
    GateEvent records[EVENT_CAPACITY]{};
    EventCursor head;
    EventCursor tail;
};

struct SessionEdge {
    Gate *gate = nullptr;
    LfmSession *session = nullptr;
    EventRing events;
    std::atomic<bool> blocked{false};
    uint32_t index = 0;
};

struct PcmEvidence {
    uint64_t frames = 0;
    uint64_t nonzero = 0;
    uint64_t hash = FNV_OFFSET;
    double peak = 0.0;
};

struct GateFrame {
    PcmEvidence first_pcm;
    PcmEvidence second_pcm;
    LfmTicketIdV1 second_ticket{};
    char first_text[4096]{};
    char second_text[4096]{};
    uint32_t first_text_bytes = 0;
    uint32_t second_text_bytes = 0;
    uint32_t first_terminals = 0;
    uint32_t second_terminals = 0;
    uint32_t first_playback_leases = 0;
    uint32_t second_playback_leases = 0;
    int32_t status = 0;
    uint32_t outcome = 0;
    bool second_ticket_bound = false;
    bool second_stop_requested = false;
    bool stop_requested = false;
    bool first_stopped = false;
    bool second_stopped = false;
    char error[512]{};
};

struct GateEvidence {
    uint64_t first_hash = 0;
    uint64_t second_hash = 0;
    uint64_t first_frames = 0;
    uint64_t second_frames = 0;
    uint64_t first_nonzero = 0;
    uint64_t second_nonzero = 0;
    uint64_t monitor_underflow_frames = 0;
    uint64_t monitor_underflow_callbacks = 0;
    uint32_t monitor_source_transitions = 0;
    char first_text[4096]{};
    char second_text[4096]{};
};

struct alignas(128) GateDiagnosticState {
    std::atomic<uint64_t> publications{0};
    std::atomic<uint32_t> outcome{0};
    std::atomic<int32_t> status{0};
    std::atomic<uint32_t> first_terminals{0};
    std::atomic<uint32_t> second_terminals{0};
    std::atomic<uint32_t> first_stopped{0};
    std::atomic<uint32_t> second_stopped{0};
    std::atomic<uint32_t> stop_requested{0};
};

struct Gate {
    LfmRuntime *runtime = nullptr;
    LfmModel *model = nullptr;
    LfmConversation *first_conversation = nullptr;
    LfmConversation *second_conversation = nullptr;
    LfmSession *first_session = nullptr;
    LfmSession *second_session = nullptr;
    LfmPlaybackConsumer *first_playback = nullptr;
    LfmPlaybackConsumer *second_playback = nullptr;
    SessionEdge first_edge;
    SessionEdge second_edge;
    koro_cont_t *continuation = nullptr;
    kc_ticket_id identity{};
    LfmTicketIdV1 first_ticket{};
    std::atomic<bool> submitted{false};
    std::atomic<int32_t> external_failure{0};
    std::atomic<uint32_t> terminal_edges{0};
    GateDiagnosticState diagnostics;
    LfmRuntimeDiagnosticView runtime_diagnostics{};
    AudibleMonitor monitor;
    float *closed_loop_pcm = nullptr;
    uint64_t closed_loop_frames = 0;
    bool audible = false;
    LfmModelMemoryV2 before{};
    LfmModelMemoryV2 after{};
    float sink[CALLBACK_FRAMES]{};
#if defined(__APPLE__)
    CFRunLoopRef runloop = nullptr;
    CFRunLoopSourceRef runloop_source = nullptr;
    std::atomic<dispatch_source_t> watchdog{nullptr};
#endif
};

void resume_gate(Gate *gate);

bool ticket_equal(const LfmTicketIdV1 &a, const LfmTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

bool ring_push(EventRing *ring, const GateEvent &event) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == EVENT_CAPACITY) return false;
    ring->records[tail % EVENT_CAPACITY] = event;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool ring_pop(EventRing *ring, GateEvent *event) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *event = ring->records[head % EVENT_CAPACITY];
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

bool ring_ready(const EventRing &ring) {
    return ring.head.value.load(std::memory_order_acquire) !=
           ring.tail.value.load(std::memory_order_acquire);
}

void fail(GateFrame *frame, int32_t status, const char *message) {
    if (!frame || frame->status != 0) return;
    frame->status = status == 0 ? LFM_STATUS_INTERNAL : status;
    std::snprintf(frame->error, sizeof(frame->error), "%s", message);
}

void fail_status(GateFrame *frame, int32_t status, const char *operation) {
    if (!frame || frame->status != 0) return;
    frame->status = status == 0 ? LFM_STATUS_INTERNAL : status;
    std::snprintf(frame->error, sizeof(frame->error), "%s failed: %d",
                  operation, status);
}

void evidence_add(PcmEvidence *evidence, const float *samples,
                  uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        const float sample = samples[index];
        if (!std::isfinite(sample)) {
            evidence->peak = std::numeric_limits<double>::infinity();
            continue;
        }
        const double magnitude = std::fabs(static_cast<double>(sample));
        evidence->peak = std::max(evidence->peak, magnitude);
        if (magnitude > 1e-6) evidence->nonzero++;
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        evidence->hash ^= bits;
        evidence->hash *= FNV_PRIME;
    }
    evidence->frames += count;
}

bool append_text(char *destination, uint32_t *used,
                 const GateEvent &event) {
    if (event.payload_bytes > 4095 - *used) return false;
    std::memcpy(destination + *used, event.payload, event.payload_bytes);
    *used += event.payload_bytes;
    destination[*used] = '\0';
    return true;
}

void resume_gate(Gate *gate) {
    if (!gate || !gate->continuation) return;
    const int status = koro_cont_resume(gate->continuation, &gate->identity);
    if (status != 0 && status != -ECANCELED) {
        int32_t expected = 0;
        gate->external_failure.compare_exchange_strong(
            expected, status, std::memory_order_release,
            std::memory_order_relaxed);
    }
}

int monitor_push(Gate *gate, const float *samples, uint32_t count,
                 uint32_t source) {
    AudibleMonitor *monitor = gate ? &gate->monitor : nullptr;
    if (!monitor || !monitor->enabled || count == 0) return 0;
    if (!samples || monitor->closing.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (monitor->source != 0 && monitor->source != source) {
        monitor->source_transitions++;
    }
    monitor->source = source;
    const uint64_t tail = monitor->tail.load(std::memory_order_relaxed);
    const uint64_t head = monitor->head.load(std::memory_order_acquire);
    if (tail - head + count > MONITOR_CAPACITY) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    const uint64_t slot = tail % MONITOR_CAPACITY;
    const uint32_t first = static_cast<uint32_t>(
        std::min<uint64_t>(count, MONITOR_CAPACITY - slot));
    std::memcpy(monitor->samples + slot, samples,
                static_cast<size_t>(first) * sizeof(float));
    if (first != count) {
        std::memcpy(monitor->samples, samples + first,
                    static_cast<size_t>(count - first) * sizeof(float));
    }
    monitor->tail.store(tail + count, std::memory_order_release);
    return 0;
}

bool monitor_close(Gate *gate) {
    AudibleMonitor *monitor = gate ? &gate->monitor : nullptr;
    if (!monitor || !monitor->enabled) return true;
    monitor->closing.store(true, std::memory_order_release);
    const uint64_t head = monitor->head.load(std::memory_order_acquire);
    const uint64_t tail = monitor->tail.load(std::memory_order_acquire);
    if (head == tail) {
        monitor->drained.store(true, std::memory_order_release);
        const uint64_t prior = monitor->callback_gate.fetch_or(
            MONITOR_CALLBACK_CLOSED, std::memory_order_acq_rel);
        if ((prior & MONITOR_CALLBACK_COUNT) != 0) return false;
    }
    return monitor->drained.load(std::memory_order_acquire) &&
           (monitor->callback_gate.load(std::memory_order_acquire) &
            MONITOR_CALLBACK_COUNT) == 0;
}

#if defined(__APPLE__)

bool monitor_callback_enter(AudibleMonitor *monitor) {
    if (!monitor) return false;
    const uint64_t prior =
        monitor->callback_gate.fetch_add(1, std::memory_order_acq_rel);
    if ((prior & MONITOR_CALLBACK_CLOSED) == 0) {
        if ((prior & MONITOR_CALLBACK_COUNT) == MONITOR_CALLBACK_COUNT) {
            std::abort();
        }
        return true;
    }
    monitor->callback_gate.fetch_sub(1, std::memory_order_release);
    return false;
}

void monitor_callback_leave(AudibleMonitor *monitor) {
    Gate *gate = monitor ? monitor->gate : nullptr;
    const uint64_t prior =
        monitor->callback_gate.fetch_sub(1, std::memory_order_acq_rel);
    const uint64_t count = prior & MONITOR_CALLBACK_COUNT;
    if (count == 0) std::abort();
    if ((prior & MONITOR_CALLBACK_CLOSED) != 0 && count == 1 && gate) {
        /* The callback releases admission before publishing the successor.
         * Resumption is its final operation; teardown may begin immediately. */
        resume_gate(gate);
    }
}

struct MonitorCallbackLease {
    AudibleMonitor *monitor;
    bool admitted;

    explicit MonitorCallbackLease(AudibleMonitor *value)
        : monitor(value), admitted(monitor_callback_enter(value)) {}

    ~MonitorCallbackLease() {
        if (admitted) monitor_callback_leave(monitor);
    }

    explicit operator bool() const { return admitted; }
};

OSStatus monitor_output_callback(void *context, AudioUnitRenderActionFlags *,
                                 const AudioTimeStamp *, UInt32, UInt32 frames,
                                 AudioBufferList *buffers) {
    auto *monitor = static_cast<AudibleMonitor *>(context);
    if (!monitor || !buffers) return kAudio_ParamError;
    for (UInt32 index = 0; index < buffers->mNumberBuffers; ++index) {
        AudioBuffer &buffer = buffers->mBuffers[index];
        if (buffer.mData && buffer.mDataByteSize != 0) {
            std::memset(buffer.mData, 0, buffer.mDataByteSize);
        }
    }
    MonitorCallbackLease callback(monitor);
    if (!callback || !monitor->enabled || frames == 0) return noErr;
    if (buffers->mNumberBuffers != 1 ||
        buffers->mBuffers[0].mNumberChannels != 1 ||
        !buffers->mBuffers[0].mData ||
        buffers->mBuffers[0].mDataByteSize < frames * sizeof(float)) {
        Gate *gate = monitor->gate;
        if (gate) {
            int32_t expected = 0;
            gate->external_failure.compare_exchange_strong(
                expected, LFM_STATUS_HOST_SINK, std::memory_order_release,
                std::memory_order_relaxed);
        }
        monitor->closing.store(true, std::memory_order_release);
        monitor->drained.store(true, std::memory_order_release);
        monitor->callback_gate.fetch_or(MONITOR_CALLBACK_CLOSED,
                                        std::memory_order_acq_rel);
        return kAudio_ParamError;
    }
    auto *destination =
        static_cast<float *>(buffers->mBuffers[0].mData);
    const uint64_t head = monitor->head.load(std::memory_order_relaxed);
    const uint64_t tail = monitor->tail.load(std::memory_order_acquire);
    const uint32_t count = static_cast<uint32_t>(
        std::min<uint64_t>(frames, tail - head));
    const bool active = monitor->active.load(std::memory_order_relaxed);
    if (count != 0) monitor->active.store(true, std::memory_order_relaxed);
    if (count < frames && (active || count != 0) &&
        !monitor->closing.load(std::memory_order_acquire)) {
        monitor->underflow_frames.fetch_add(frames - count,
                                            std::memory_order_relaxed);
        monitor->underflow_callbacks.fetch_add(1,
                                               std::memory_order_relaxed);
    }
    const uint64_t slot = head % MONITOR_CAPACITY;
    const uint32_t first = static_cast<uint32_t>(
        std::min<uint64_t>(count, MONITOR_CAPACITY - slot));
    if (first != 0) {
        std::memcpy(destination, monitor->samples + slot,
                    static_cast<size_t>(first) * sizeof(float));
    }
    if (first != count) {
        std::memcpy(destination + first, monitor->samples,
                    static_cast<size_t>(count - first) * sizeof(float));
    }
    const uint64_t consumed = head + count;
    monitor->head.store(consumed, std::memory_order_release);
    buffers->mBuffers[0].mDataByteSize = frames * sizeof(float);
    if (monitor->closing.load(std::memory_order_acquire) &&
        consumed == monitor->tail.load(std::memory_order_acquire) &&
        !monitor->drained.exchange(true, std::memory_order_acq_rel)) {
        /* MonitorCallbackLease publishes the successor after this callback
         * has released its retained admission. */
        monitor->callback_gate.fetch_or(MONITOR_CALLBACK_CLOSED,
                                        std::memory_order_acq_rel);
    }
    return noErr;
}

int monitor_create(Gate *gate) {
    if (!gate || !gate->audible) return 0;
    AudibleMonitor *monitor = &gate->monitor;
    monitor->gate = gate;
    monitor->samples = new (std::nothrow) float[MONITOR_CAPACITY];
    if (!monitor->samples) return LFM_STATUS_OUT_OF_MEMORY;
    const AudioComponentDescription description = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_HALOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent component = AudioComponentFindNext(nullptr, &description);
    if (!component) return LFM_STATUS_UNSUPPORTED;
    OSStatus status = AudioComponentInstanceNew(component, &monitor->output);
    if (status != noErr) return static_cast<int>(status);
    const UInt32 enabled = 1;
    const UInt32 disabled = 0;
    status = AudioUnitSetProperty(
        monitor->output, kAudioOutputUnitProperty_EnableIO,
        kAudioUnitScope_Output, 0, &enabled, sizeof(enabled));
    if (status == noErr) {
        status = AudioUnitSetProperty(
            monitor->output, kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input, 1, &disabled, sizeof(disabled));
    }
    AudioDeviceID device = kAudioObjectUnknown;
    const AudioObjectPropertyAddress address = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain,
    };
    UInt32 device_bytes = sizeof(device);
    if (status == noErr) {
        status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject, &address, 0, nullptr, &device_bytes,
            &device);
    }
    if (status == noErr && device == kAudioObjectUnknown) {
        return LFM_STATUS_UNSUPPORTED;
    }
    if (status == noErr) {
        status = AudioUnitSetProperty(
            monitor->output, kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global, 0, &device, sizeof(device));
    }
    AudioStreamBasicDescription format{};
    format.mSampleRate = RATE;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsFloat |
                          kAudioFormatFlagIsPacked |
                          kAudioFormatFlagsNativeEndian;
    format.mBytesPerPacket = sizeof(float);
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = sizeof(float);
    format.mChannelsPerFrame = 1;
    format.mBitsPerChannel = 32;
    if (status == noErr) {
        status = AudioUnitSetProperty(
            monitor->output, kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input, 0, &format, sizeof(format));
    }
    const AURenderCallbackStruct callback = {
        .inputProc = monitor_output_callback,
        .inputProcRefCon = monitor,
    };
    if (status == noErr) {
        status = AudioUnitSetProperty(
            monitor->output, kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input, 0, &callback, sizeof(callback));
    }
    if (status == noErr) status = AudioUnitInitialize(monitor->output);
    if (status != noErr) return static_cast<int>(status);
    monitor->enabled = true;
    return 0;
}

int monitor_start(Gate *gate) {
    if (!gate || !gate->audible) return 0;
    bool expected = false;
    if (!gate->monitor.started.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return 0;
    }
    const int status =
        static_cast<int>(AudioOutputUnitStart(gate->monitor.output));
    if (status != 0) {
        gate->monitor.started.store(false, std::memory_order_release);
    }
    return status;
}

void monitor_destroy(Gate *gate) {
    if (!gate) return;
    AudibleMonitor *monitor = &gate->monitor;
    monitor->enabled = false;
    const uint64_t callbacks = monitor->callback_gate.fetch_or(
        MONITOR_CALLBACK_CLOSED, std::memory_order_acq_rel);
    if ((callbacks & MONITOR_CALLBACK_COUNT) != 0) std::abort();
    if (monitor->output) {
        (void)AudioOutputUnitStop(monitor->output);
        (void)AudioUnitUninitialize(monitor->output);
        (void)AudioComponentInstanceDispose(monitor->output);
        monitor->output = nullptr;
    }
    delete[] monitor->samples;
    monitor->samples = nullptr;
    monitor->gate = nullptr;
}

#endif

int event_callback(void *context, const LfmEventV1 *source) {
    auto *edge = static_cast<SessionEdge *>(context);
    if (!edge || !edge->gate || !source ||
        source->size != sizeof(*source) || source->abi_version != ABI ||
        source->payload_bytes > EVENT_PAYLOAD ||
        (source->payload_bytes != 0 && !source->payload)) {
        return LFM_STATUS_HOST_SINK;
    }
    GateEvent event{};
    event.kind = source->kind;
    event.flags = source->flags;
    event.session_id = source->session_id;
    event.epoch = source->epoch;
    event.ticket = source->ticket;
    event.payload_bytes = source->payload_bytes;
    event.status = source->status;
    if (source->payload_bytes != 0) {
        std::memcpy(event.payload, source->payload, source->payload_bytes);
    }
    if (!ring_push(&edge->events, event)) {
        edge->blocked.store(true, std::memory_order_release);
        resume_gate(edge->gate);
        return LFM_STATUS_WOULD_BLOCK;
    }
    resume_gate(edge->gate);
    return 0;
}

int drain_first_playback(Gate *gate, GateFrame *frame,
                         const GateEvent &event) {
    if (event.payload_bytes != sizeof(LfmPlaybackReadyEventV1)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPlaybackReadyEventV1 ready{};
    std::memcpy(&ready, event.payload, sizeof(ready));
    if (ready.size != sizeof(ready) || ready.abi_version != ABI) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    LfmPcmLeaseV1 lease{};
    int status = lfm_playback_consumer_claim(
        gate->first_playback, &event.ticket, event.epoch, ready.lease_id,
        ready.buffer_generation, &lease);
    if (status != 0) return status;
    if (!gate->closed_loop_pcm ||
        gate->closed_loop_frames > CLOSED_LOOP_CAPACITY ||
        lease.frames > CLOSED_LOOP_CAPACITY - gate->closed_loop_frames) {
        status = LFM_STATUS_WOULD_BLOCK;
    }
    if (status == 0) {
        float *destination =
            gate->closed_loop_pcm + gate->closed_loop_frames;
        LfmPlaybackRenderV1 rendered{};
        status = lfm_playback_consumer_render_f32(
            gate->first_playback, &lease, 0, destination, lease.frames, 1,
            CLOSED_LOOP_CAPACITY - gate->closed_loop_frames, &rendered);
        if (status == 0) {
            status = monitor_push(gate, destination, lease.frames, 1);
        }
        if (status == 0) {
            evidence_add(&frame->first_pcm, destination, lease.frames);
            gate->closed_loop_frames += lease.frames;
        }
    }
    const int released =
        lfm_playback_consumer_release(gate->first_playback, &lease);
    return status != 0 ? status : released;
}

int drain_second_playback(Gate *gate, GateFrame *frame,
                          const GateEvent &event) {
    if (event.payload_bytes != sizeof(LfmPlaybackReadyEventV1)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPlaybackReadyEventV1 ready{};
    std::memcpy(&ready, event.payload, sizeof(ready));
    if (ready.size != sizeof(ready) || ready.abi_version != ABI) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    LfmPcmLeaseV1 lease{};
    int status = lfm_playback_consumer_claim(
        gate->second_playback, &event.ticket, event.epoch, ready.lease_id,
        ready.buffer_generation, &lease);
    if (status != 0) return status;
    uint32_t offset = 0;
    while (offset < lease.frames && status == 0) {
        const uint32_t count =
            std::min(CALLBACK_FRAMES, lease.frames - offset);
        LfmPlaybackRenderV1 rendered{};
        status = lfm_playback_consumer_render_f32(
            gate->second_playback, &lease, offset, gate->sink, count, 1,
            CALLBACK_FRAMES, &rendered);
        if (status == 0) {
            status = monitor_push(gate, gate->sink, count, 2);
        }
        if (status == 0) {
            evidence_add(&frame->second_pcm, gate->sink, count);
            offset += count;
        }
    }
    const int released =
        lfm_playback_consumer_release(gate->second_playback, &lease);
    return status != 0 ? status : released;
}

int process_event(Gate *gate, GateFrame *frame, uint32_t endpoint,
                  const GateEvent &event) {
    const bool first = endpoint == 0;
    const LfmTicketIdV1 &ticket =
        first ? gate->first_ticket : frame->second_ticket;
    if (event.kind == LFM_EVENT_STATE) return 0;
    if (event.kind == LFM_EVENT_STOPPED) {
        if (first) frame->first_stopped = true;
        else frame->second_stopped = true;
        return event.status;
    }
    if (event.kind == LFM_EVENT_ERROR) {
        char message[EVENT_PAYLOAD + 1]{};
        std::memcpy(message, event.payload, event.payload_bytes);
        fail(frame, event.status, message[0] ? message : "native error event");
        return 0;
    }
    if (!first && frame->second_terminals != 0) return 0;
    if (event.kind == LFM_EVENT_TURN_STARTED) {
        if (first) {
            if (!ticket_equal(event.ticket, gate->first_ticket)) {
                return LFM_STATUS_STALE;
            }
        } else if (!frame->second_ticket_bound) {
            frame->second_ticket = event.ticket;
            frame->second_ticket_bound = true;
        } else if (!ticket_equal(event.ticket, frame->second_ticket)) {
            return LFM_STATUS_STALE;
        }
        return 0;
    }
    if ((first || frame->second_ticket_bound) &&
        !ticket_equal(event.ticket, ticket)) {
        return LFM_STATUS_STALE;
    }
    if (event.kind == LFM_EVENT_TEXT) {
        const bool appended = first
            ? append_text(frame->first_text, &frame->first_text_bytes, event)
            : append_text(frame->second_text, &frame->second_text_bytes,
                          event);
        return appended ? 0 : LFM_STATUS_WOULD_BLOCK;
    }
    if (event.kind == LFM_EVENT_PLAYBACK_READY) {
        const int status = first
            ? drain_first_playback(gate, frame, event)
            : drain_second_playback(gate, frame, event);
        if (status == 0) {
            if (first) frame->first_playback_leases++;
            else frame->second_playback_leases++;
        }
        return status;
    }
    if (event.kind != LFM_EVENT_TURN ||
        event.payload_bytes != sizeof(LfmTurnEventV1)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmTurnEventV1 turn{};
    std::memcpy(&turn, event.payload, sizeof(turn));
    if (turn.size != sizeof(turn) || turn.abi_version != ABI ||
        event.status != 0) {
        return event.status != 0 ? event.status : LFM_STATUS_ABI_MISMATCH;
    }
    if ((event.flags & LFM_EVENT_FLAG_TRUNCATED) != 0) {
        fail(frame, LFM_STATUS_INTERNAL,
             first
                 ? "first native agent exhausted max_new_tokens before its terminal"
                 : "second native agent exhausted max_new_tokens before its terminal");
        return 0;
    }
    if (first) {
        frame->first_terminals++;
        if (frame->first_terminals != 1 ||
            (event.flags & LFM_EVENT_FLAG_HAS_AUDIO) == 0 ||
            turn.playback_leases != frame->first_playback_leases) {
            return LFM_STATUS_INTERNAL;
        }
        if (!gate->closed_loop_pcm || gate->closed_loop_frames == 0 ||
            gate->closed_loop_frames != frame->first_pcm.frames) {
            return LFM_STATUS_INTERNAL;
        }
        const LfmF32Span pcm = {
            .data = gate->closed_loop_pcm,
            .length = gate->closed_loop_frames,
        };
        const int submitted = lfm_internal_session_submit_pcm_spans(
            gate->second_session, &pcm, 1, RATE, &event.ticket,
            &frame->second_ticket);
        if (submitted != 0) return submitted;
        frame->second_ticket_bound = true;
        /* The complete source turn is now buffered and sealed. Start physical
         * playback at this model-derived edge while B computes from the same
         * PCM view; the FIFO already contains all of A, so B can only follow
         * it and can never splice into the middle of the source utterance. */
        const int monitor = monitor_start(gate);
        if (monitor != 0) return monitor;
    } else {
        frame->second_terminals++;
        if (frame->second_terminals != 1 ||
            (event.flags & LFM_EVENT_FLAG_HAS_AUDIO) == 0 ||
            turn.playback_leases != frame->second_playback_leases) {
            return LFM_STATUS_INTERNAL;
        }
    }
    return 0;
}

enum GateOutcome : uint32_t {
    GATE_SUSPEND = 0,
    GATE_YIELD = 1,
    GATE_DONE = 2,
};

uint32_t advance_gate(Gate *gate, GateFrame *frame) {
    const int32_t external =
        gate->external_failure.load(std::memory_order_acquire);
    if (external != 0) fail_status(frame, external, "native gate watchdog");

    uint32_t drained = 0;
    bool progressed = true;
    while (drained < EVENT_BUDGET && progressed) {
        progressed = false;
        for (SessionEdge *edge : {&gate->first_edge, &gate->second_edge}) {
            GateEvent event{};
            if (drained == EVENT_BUDGET ||
                !ring_pop(&edge->events, &event)) {
                continue;
            }
            progressed = true;
            const int status = process_event(gate, frame, edge->index, event);
            if (status != 0 && frame->status == 0) {
                const LfmTicketIdV1 expected = edge->index == 0
                    ? gate->first_ticket : frame->second_ticket;
                const LfmSessionDiagnosticState *owner = edge->index == 0
                    ? gate->runtime_diagnostics.first.state
                    : gate->runtime_diagnostics.second.state;
                const uint32_t cause = owner
                    ? owner->terminal_cause.load(std::memory_order_acquire)
                    : 0;
                const uint32_t operation = owner
                    ? owner->terminal_operation.load(
                          std::memory_order_acquire)
                    : 0;
                const int32_t owner_status = owner
                    ? owner->terminal_status.load(std::memory_order_acquire)
                    : 0;
                frame->status = status;
                std::snprintf(
                    frame->error, sizeof(frame->error),
                    "native event failed: status=%d endpoint=%u kind=%u "
                    "cause=%u operation=%u owner_status=%d "
                    "ticket={%llu,%llu,%u,%u} expected={%llu,%llu,%u,%u}",
                    status, edge->index, event.kind, cause, operation,
                    owner_status,
                    static_cast<unsigned long long>(event.ticket.runtime_epoch),
                    static_cast<unsigned long long>(event.ticket.sequence),
                    event.ticket.generation, event.ticket.kind,
                    static_cast<unsigned long long>(expected.runtime_epoch),
                    static_cast<unsigned long long>(expected.sequence),
                    expected.generation, expected.kind);
            }
            drained++;
        }
    }
    for (SessionEdge *edge : {&gate->first_edge, &gate->second_edge}) {
        if (edge->blocked.exchange(false, std::memory_order_acq_rel)) {
            const int status = lfm_session_host_capacity(edge->session);
            if (status != 0 && status != LFM_STATUS_CANCELLED) {
                fail_status(frame, status, "resume native event capacity");
            }
        }
    }

    if (frame->second_terminals == 1 &&
        !frame->second_stop_requested) {
        frame->second_stop_requested = true;
        lfm_session_request_stop(gate->second_session);
    }

    if (frame->status == 0 && frame->first_terminals == 1 &&
        frame->second_terminals == 1 && !frame->stop_requested) {
        if (frame->first_text_bytes == 0 || frame->second_text_bytes == 0 ||
            frame->first_pcm.frames == 0 || frame->second_pcm.frames == 0 ||
            frame->first_pcm.nonzero == 0 || frame->second_pcm.nonzero == 0 ||
            !std::isfinite(frame->first_pcm.peak) ||
            !std::isfinite(frame->second_pcm.peak)) {
            fail(frame, LFM_STATUS_INTERNAL,
                 "native agents did not produce finite text plus spoken PCM");
        }
    }

    if ((frame->status != 0 ||
         (frame->first_terminals == 1 && frame->second_terminals == 1)) &&
        !frame->stop_requested) {
        frame->stop_requested = true;
        lfm_session_request_stop(gate->first_session);
        if (!frame->second_stop_requested) {
            frame->second_stop_requested = true;
            lfm_session_request_stop(gate->second_session);
        }
    }
    if (frame->stop_requested && frame->first_stopped &&
        frame->second_stopped) {
        const int status = monitor_start(gate);
        if (status != 0) {
            fail_status(frame, status, "start native speaker monitor");
            return GATE_DONE;
        }
        return monitor_close(gate) ? GATE_DONE : GATE_SUSPEND;
    }
    if (ring_ready(gate->first_edge.events) ||
        ring_ready(gate->second_edge.events) ||
        drained == EVENT_BUDGET) {
        return GATE_YIELD;
    }
    return GATE_SUSPEND;
}

void publish_gate_diagnostics(Gate *gate, const GateFrame *frame) {
    GateDiagnosticState &state = gate->diagnostics;
    state.outcome.store(frame->outcome, std::memory_order_relaxed);
    state.status.store(frame->status, std::memory_order_relaxed);
    state.first_terminals.store(frame->first_terminals,
                                std::memory_order_relaxed);
    state.second_terminals.store(frame->second_terminals,
                                 std::memory_order_relaxed);
    state.first_stopped.store(frame->first_stopped ? 1u : 0u,
                              std::memory_order_relaxed);
    state.second_stopped.store(frame->second_stopped ? 1u : 0u,
                               std::memory_order_relaxed);
    state.stop_requested.store(frame->stop_requested ? 1u : 0u,
                               std::memory_order_relaxed);
    state.publications.fetch_add(1, std::memory_order_release);
}

void *gate_step(koro_cont_t *continuation) {
    auto *gate = static_cast<Gate *>(koro_cont_argument(continuation));
    auto *frame = static_cast<GateFrame *>(koro_cont_frame(continuation));
    if (!gate || !frame) std::abort();
    KORO_BEGIN(continuation);
    for (;;) {
        if (!gate->submitted.load(std::memory_order_acquire)) {
            KORO_SUSPEND(continuation);
        }
        frame->outcome = advance_gate(gate, frame);
        publish_gate_diagnostics(gate, frame);
        if (frame->outcome == GATE_DONE) break;
        if (frame->outcome == GATE_YIELD) {
            KORO_YIELD(continuation);
            /* A callback may have been coalesced with the self-publication
             * that resumed this yield.  Re-enter the predicate drain before
             * the frame is allowed to suspend on a future callback. */
            continue;
        }
        KORO_SUSPEND(continuation);
    }
    KORO_END(continuation);
}

#if defined(__APPLE__)

void publish_terminal_edge(Gate *gate, uint32_t edge) {
    CFRunLoopRef runloop = gate->runloop;
    if (runloop) CFRetain(runloop);
    const bool failed =
        gate->external_failure.load(std::memory_order_acquire) != 0;
    /* This fetch-or is the publisher's final Gate access. The second edge
     * owns the run-loop wake using its separately-retained local handle. */
    const uint32_t prior = gate->terminal_edges.fetch_or(
        edge, std::memory_order_acq_rel);
    if (((prior | edge) == GATE_ALL_DONE || failed) && runloop) {
        CFRunLoopStop(runloop);
        CFRunLoopWakeUp(runloop);
    }
    if (runloop) CFRelease(runloop);
}

void gate_retired(void *context, const kc_ticket_id *identity) {
    auto *gate = static_cast<Gate *>(context);
    if (!gate || !identity || !ticket_equal(*identity, gate->identity)) {
        std::abort();
    }
    dispatch_source_t watchdog =
        gate->watchdog.load(std::memory_order_acquire);
    if (watchdog) dispatch_source_cancel(watchdog);
    publish_terminal_edge(gate, GATE_CONTINUATION_DONE);
}

void print_session_watchdog(const char *name,
                            const LfmSessionDiagnosticView &view) {
    kc_service_snapshot coordinator = {
    };
    kc_service_snapshot delivery = {
    };
    const int coordinator_status =
        kc_service_snapshot_get(view.coordinator, &coordinator);
    const int delivery_status =
        kc_service_snapshot_get(view.delivery, &delivery);
    const LfmSessionDiagnosticState *state = view.state;
    std::fprintf(
        stderr,
        "native speech watchdog: %s session={pub=%llu progress=%u "
        "coord_phase=%u action={active=%u phase=%u admission=%u route=%u "
        "playback=%u ticket=%llu route_ticket=%llu} result={active=%u "
        "next=%u count=%u} queues={event=%llu command=%llu pcm=%llu} "
        "delivery_pending=%u stop=%u event_done=%u operation=%u "
        "terminal=%d cause=%u terminal_operation=%u} "
        "coordinator={rc=%d state=%u notify=%llu handled=%llu callbacks=%llu} "
        "delivery={rc=%d state=%u notify=%llu handled=%llu callbacks=%llu}\n",
        name,
        static_cast<unsigned long long>(
            state->publications.load(std::memory_order_acquire)),
        state->progress.load(std::memory_order_acquire),
        state->coordinator_phase.load(std::memory_order_acquire),
        state->action_active.load(std::memory_order_acquire),
        state->action_phase.load(std::memory_order_acquire),
        state->admission_pending.load(std::memory_order_acquire),
        state->route_pending.load(std::memory_order_acquire),
        state->playback_active.load(std::memory_order_acquire),
        static_cast<unsigned long long>(
            state->action_ticket_sequence.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            state->action_route_sequence.load(std::memory_order_acquire)),
        state->result_active.load(std::memory_order_acquire),
        state->result_next.load(std::memory_order_acquire),
        state->result_count.load(std::memory_order_acquire),
        static_cast<unsigned long long>(
            state->event_depth.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            state->command_depth.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            state->pcm_depth.load(std::memory_order_acquire)),
        state->delivery_pending.load(std::memory_order_acquire),
        state->stop.load(std::memory_order_acquire),
        state->event_done.load(std::memory_order_acquire),
        state->conversation_operation.load(std::memory_order_acquire),
        state->terminal_status.load(std::memory_order_acquire),
        state->terminal_cause.load(std::memory_order_acquire),
        state->terminal_operation.load(std::memory_order_acquire),
        coordinator_status, coordinator.run_state,
        static_cast<unsigned long long>(coordinator.notifications),
        static_cast<unsigned long long>(coordinator.handled_notifications),
        static_cast<unsigned long long>(coordinator.callbacks), delivery_status,
        delivery.run_state,
        static_cast<unsigned long long>(delivery.notifications),
        static_cast<unsigned long long>(delivery.handled_notifications),
        static_cast<unsigned long long>(delivery.callbacks));
}

void print_native_watchdog(Gate *gate) {
    const LfmRuntimeDiagnosticView &view = gate->runtime_diagnostics;
    koro_cont_snapshot gate_cont{};
    koro_cont_snapshot bridge_cont{};
    kc_runtime_snapshot coordination = {
    };
    kc_runtime_snapshot engine_runtime = {
    };
    kc_service_snapshot route = {
    };
    kc_service_snapshot supervisor = {
    };
    kc_team_snapshot team = {
    };
    kc_team_quorum_snapshot quorum = {
    };
    LfmKernelBridgeSnapshotV1 bridge = {
        .size = sizeof(LfmKernelBridgeSnapshotV1),
        .abi_version = KC_COORD_ABI_VERSION,
    };
    LfmEngineDiagnosticCounts counts{};
    const int gate_status =
        koro_cont_snapshot_get(gate->continuation, &gate_cont);
    const int bridge_cont_status = koro_cont_snapshot_get(
        view.engine.bridge_continuation, &bridge_cont);
    const int coordination_status =
        kc_runtime_snapshot_get(view.coordination, &coordination);
    const int engine_runtime_status =
        kc_runtime_snapshot_get(view.engine.runtime, &engine_runtime);
    const int route_status =
        kc_service_snapshot_get(view.engine.route_service, &route);
    const int supervisor_status =
        kc_service_snapshot_get(view.engine.supervisor_service, &supervisor);
    const int team_status = kc_team_snapshot_get(view.engine.team, &team);
    const int quorum_status = team_status == 0 &&
            team.dispatched_generation != 0
        ? kc_team_quorum_snapshot_get(view.engine.team,
                                      team.dispatched_generation, &quorum)
        : -EINVAL;
    const int queue_status =
        lfm_kernel_bridge_snapshot(view.engine.bridge, &bridge);
    const int counts_status =
        lfm_internal_engine_diagnostic_counts(&view.engine, &counts);
    const LfmEngineDiagnosticState *engine = view.engine.state;
    const uint64_t gate_first_head =
        gate->first_edge.events.head.value.load(std::memory_order_acquire);
    const uint64_t gate_first_tail =
        gate->first_edge.events.tail.value.load(std::memory_order_acquire);
    const uint64_t gate_second_head =
        gate->second_edge.events.head.value.load(std::memory_order_acquire);
    const uint64_t gate_second_tail =
        gate->second_edge.events.tail.value.load(std::memory_order_acquire);

    const char *classification = "unclassified";
    if (quorum_status == 0 &&
        quorum.returned_mask != quorum.expected_mask) {
        classification = "hung-team-generation";
    } else if (engine->team_completion_edge.load(std::memory_order_acquire) >
               engine->team_completion_consumed.load(
                   std::memory_order_acquire)) {
        classification = "lost-team-completion-resume";
    } else if (queue_status == 0 &&
               bridge.completions_published > bridge.completions_consumed) {
        classification = "lost-bridge-completion-resume";
    } else if (counts_status == 0 && counts.routes_ready != 0 &&
               route_status == 0 && route.run_state == KORO_SUSPENDED &&
               route.notifications == route.handled_notifications) {
        classification = "orphaned-ready-route";
    } else if ((gate_first_tail != gate_first_head ||
                gate_second_tail != gate_second_head) &&
               gate_status == 0 && gate_cont.run_state == KORO_SUSPENDED &&
               gate_cont.wake_pending == 0) {
        classification = "lost-gate-event-resume";
    }

    std::fprintf(
        stderr,
        "native speech watchdog: classification=%s gate={rc=%d state=%u "
        "wake=%u worker=%u pub=%llu outcome=%u status=%d terminals=%u/%u "
        "stopped=%u/%u stop=%u rings=%llu:%llu/%llu:%llu edges=%u} "
        "coord_runtime={rc=%d active=%zu queued=%zu running=%zu dormant=%zu "
        "wake=%llu resumes=%llu}\n",
        classification, gate_status, gate_cont.run_state,
        gate_cont.wake_pending, gate_cont.current_worker,
        static_cast<unsigned long long>(
            gate->diagnostics.publications.load(std::memory_order_acquire)),
        gate->diagnostics.outcome.load(std::memory_order_acquire),
        gate->diagnostics.status.load(std::memory_order_acquire),
        gate->diagnostics.first_terminals.load(std::memory_order_acquire),
        gate->diagnostics.second_terminals.load(std::memory_order_acquire),
        gate->diagnostics.first_stopped.load(std::memory_order_acquire),
        gate->diagnostics.second_stopped.load(std::memory_order_acquire),
        gate->diagnostics.stop_requested.load(std::memory_order_acquire),
        static_cast<unsigned long long>(gate_first_head),
        static_cast<unsigned long long>(gate_first_tail),
        static_cast<unsigned long long>(gate_second_head),
        static_cast<unsigned long long>(gate_second_tail),
        gate->terminal_edges.load(std::memory_order_acquire),
        coordination_status, coordination.active, coordination.queued,
        coordination.running, coordination.dormant,
        static_cast<unsigned long long>(coordination.wake_requests),
        static_cast<unsigned long long>(coordination.resumes));
    std::fprintf(
        stderr,
        "native speech watchdog: engine={pub=%llu bridge={rc=%d state=%u "
        "wake=%u worker=%u phase=%u valid=%u activations=%llu} request=%u "
        "stage=%u program_phase=%u active_status=%d team_gen=%llu "
        "completion_edge=%llu consumed=%llu route_callbacks=%llu} "
        "runtime={rc=%d active=%zu queued=%zu running=%zu dormant=%zu "
        "wake=%llu resumes=%llu} route={rc=%d state=%u notify=%llu "
        "handled=%llu callbacks=%llu} supervisor={rc=%d state=%u "
        "notify=%llu handled=%llu callbacks=%llu}\n",
        static_cast<unsigned long long>(
            engine->publications.load(std::memory_order_acquire)),
        bridge_cont_status, bridge_cont.run_state, bridge_cont.wake_pending,
        bridge_cont.current_worker,
        engine->bridge_phase.load(std::memory_order_acquire),
        engine->bridge_valid.load(std::memory_order_acquire),
        static_cast<unsigned long long>(
            engine->bridge_activations.load(std::memory_order_acquire)),
        engine->request.load(std::memory_order_acquire),
        engine->stage.load(std::memory_order_acquire),
        engine->program_phase.load(std::memory_order_acquire),
        engine->active_status.load(std::memory_order_acquire),
        static_cast<unsigned long long>(
            engine->team_generation.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            engine->team_completion_edge.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            engine->team_completion_consumed.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            engine->route_callbacks.load(std::memory_order_acquire)),
        engine_runtime_status, engine_runtime.active, engine_runtime.queued,
        engine_runtime.running, engine_runtime.dormant,
        static_cast<unsigned long long>(engine_runtime.wake_requests),
        static_cast<unsigned long long>(engine_runtime.resumes), route_status,
        route.run_state, static_cast<unsigned long long>(route.notifications),
        static_cast<unsigned long long>(route.handled_notifications),
        static_cast<unsigned long long>(route.callbacks), supervisor_status,
        supervisor.run_state,
        static_cast<unsigned long long>(supervisor.notifications),
        static_cast<unsigned long long>(supervisor.handled_notifications),
        static_cast<unsigned long long>(supervisor.callbacks));
    std::fprintf(
        stderr,
        "native speech watchdog: team={rc=%d members=%u started=%u "
        "dispatched=%llu completed=%llu returned=%u quorum_rc=%d "
        "masks=%016llx/%016llx/%016llx} queue={rc=%d submit=%llu/%llu "
        "complete=%llu/%llu} counts={rc=%d pass=%llu/%llu dispatch=%llu "
        "wake=%llu route=%llu deferral=%llu slots=%u "
        "routes=%u/%u/%u/%u/%u/%u bridge_gen=%llu edge=%llu retired=%llu "
        "lease=%llu terminal=%llu}\n",
        team_status, team.member_count, team.started_members,
        static_cast<unsigned long long>(team.dispatched_generation),
        static_cast<unsigned long long>(team.completed_generation),
        team.completed_members, quorum_status,
        static_cast<unsigned long long>(quorum.expected_mask),
        static_cast<unsigned long long>(quorum.entered_mask),
        static_cast<unsigned long long>(quorum.returned_mask), queue_status,
        static_cast<unsigned long long>(bridge.submissions_consumed),
        static_cast<unsigned long long>(bridge.submissions_accepted),
        static_cast<unsigned long long>(bridge.completions_consumed),
        static_cast<unsigned long long>(bridge.completions_published),
        counts_status,
        static_cast<unsigned long long>(counts.pass_completions),
        static_cast<unsigned long long>(counts.pass_submissions),
        static_cast<unsigned long long>(counts.bridge_dispatches),
        static_cast<unsigned long long>(counts.dispatch_wakes),
        static_cast<unsigned long long>(counts.route_dispatches),
        static_cast<unsigned long long>(counts.route_admission_deferrals),
        counts.pass_slots_live, counts.routes_free, counts.routes_claimed,
        counts.routes_ready, counts.routes_dispatching, counts.routes_running,
        counts.routes_done,
        static_cast<unsigned long long>(counts.bridge_team_generation),
        static_cast<unsigned long long>(counts.bridge_team_completion),
        static_cast<unsigned long long>(counts.bridge_retired_generation),
        static_cast<unsigned long long>(counts.gang_lease),
        static_cast<unsigned long long>(counts.team_terminal));
    print_session_watchdog("first", view.first);
    print_session_watchdog("second", view.second);
}

void watchdog_fired(void *context) {
    auto *gate = static_cast<Gate *>(context);
    if (gate) {
        const uint64_t head =
            gate->monitor.head.load(std::memory_order_acquire);
        const uint64_t tail =
            gate->monitor.tail.load(std::memory_order_acquire);
        const uint64_t callbacks =
            gate->monitor.callback_gate.load(std::memory_order_acquire);
        std::fprintf(
            stderr,
            "native speech watchdog: monitor={enabled=%u started=%u "
            "closing=%u drained=%u head=%llu tail=%llu callbacks=%llu} "
            "terminal_edges=%u\n",
            gate->monitor.enabled ? 1u : 0u,
            gate->monitor.started.load(std::memory_order_acquire) ? 1u : 0u,
            gate->monitor.closing.load(std::memory_order_acquire) ? 1u : 0u,
            gate->monitor.drained.load(std::memory_order_acquire) ? 1u : 0u,
            static_cast<unsigned long long>(head),
            static_cast<unsigned long long>(tail),
            static_cast<unsigned long long>(callbacks),
            gate->terminal_edges.load(std::memory_order_acquire));
        print_native_watchdog(gate);
        int32_t expected = 0;
        gate->external_failure.compare_exchange_strong(
            expected, -ETIMEDOUT, std::memory_order_release,
            std::memory_order_relaxed);
    }
    /* A watchdog is not an inference successor. Returning to close_gate would
     * immediately enter administrative joins and let the deadlock that fired
     * this watchdog defeat its bound. Terminate the test process here: no
     * continuation is resumed, no model state advances, and no callback can
     * outlive stack-owned Gate storage. */
    std::abort();
}

void watchdog_cancelled(void *context) {
    auto *gate = static_cast<Gate *>(context);
    publish_terminal_edge(gate, GATE_WATCHDOG_DONE);
}

#endif

LfmConversationOptionsV1 conversation_options(uint64_t seed) {
    return {
        .size = sizeof(LfmConversationOptionsV1),
        .abi_version = ABI,
        .flags = 0,
        .reserved0 = 0,
        .seed = seed,
        .text = {
            .size = sizeof(LfmSamplingPolicyV1),
            .abi_version = ABI,
            .flags = LFM_SAMPLING_GREEDY,
            .top_k = 1,
            .temperature = 0.0,
            .reserved = 0,
        },
        .audio = {
            .size = sizeof(LfmSamplingPolicyV1),
            .abi_version = ABI,
            .flags = 0,
            .top_k = 4,
            .temperature = 1.0,
            .reserved = 0,
        },
        .reserved = {},
    };
}

LfmSessionConfigV1 session_config(uint64_t id) {
    return {
        .size = sizeof(LfmSessionConfigV1),
        .abi_version = ABI,
        .session_id = id,
        .playback_slots = 8,
        .capture_max_callback_frames = CALLBACK_FRAMES,
        .playback_frames_per_slot = 0,
        .pcm_channels = 1,
        .capture_sample_rate = RATE,
        .playback_sample_rate = RATE,
        .command_capacity = 8,
        .max_new_tokens = MAX_TOKENS,
        .flags = 0,
        .reserved = {},
    };
}

void copy_error(char *destination, size_t capacity, const char *source) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", source ? source : "unknown");
}

int validate_native_diagnostics(Gate *gate, char *error,
                                size_t error_length) {
    if (!gate || !gate->runtime_diagnostics.engine.state) {
        copy_error(error, error_length,
                   "native diagnostic pointer view was not retained");
        return LFM_STATUS_INTERNAL;
    }
    const LfmRuntimeDiagnosticView &view = gate->runtime_diagnostics;
    kc_team_snapshot team = {
    };
    LfmKernelBridgeSnapshotV1 bridge = {
        .size = sizeof(LfmKernelBridgeSnapshotV1),
        .abi_version = KC_COORD_ABI_VERSION,
    };
    LfmEngineDiagnosticCounts counts{};
    const int team_status = kc_team_snapshot_get(view.engine.team, &team);
    const int bridge_status =
        lfm_kernel_bridge_snapshot(view.engine.bridge, &bridge);
    const int counts_status =
        lfm_internal_engine_diagnostic_counts(&view.engine, &counts);
    const LfmEngineDiagnosticState *state = view.engine.state;
    const uint64_t completion_edge =
        state->team_completion_edge.load(std::memory_order_acquire);
    const uint64_t completion_consumed =
        state->team_completion_consumed.load(std::memory_order_acquire);
    const uint32_t live_routes =
        counts.routes_claimed + counts.routes_ready +
        counts.routes_dispatching + counts.routes_running + counts.routes_done;
    const bool settled =
        team_status == 0 && bridge_status == 0 && counts_status == 0 &&
        team.dispatched_generation == team.completed_generation &&
        bridge.submissions_accepted == bridge.submissions_consumed &&
        bridge.completions_published == bridge.completions_consumed &&
        counts.pass_submissions == counts.pass_completions &&
        counts.pass_slots_live == 0 && live_routes == 0 &&
        counts.gang_lease == 0 && completion_edge == completion_consumed &&
        view.first.state->event_done.load(std::memory_order_acquire) != 0 &&
        view.second.state->event_done.load(std::memory_order_acquire) != 0;
    if (settled) return 0;
    std::snprintf(
        error, error_length,
        "native diagnostic settlement failed: rc=%d/%d/%d team=%llu/%llu "
        "sq=%llu/%llu cq=%llu/%llu pass=%llu/%llu slots=%u routes=%u "
        "lease=%llu edge=%llu/%llu done=%u/%u",
        team_status, bridge_status, counts_status,
        static_cast<unsigned long long>(team.completed_generation),
        static_cast<unsigned long long>(team.dispatched_generation),
        static_cast<unsigned long long>(bridge.submissions_consumed),
        static_cast<unsigned long long>(bridge.submissions_accepted),
        static_cast<unsigned long long>(bridge.completions_consumed),
        static_cast<unsigned long long>(bridge.completions_published),
        static_cast<unsigned long long>(counts.pass_completions),
        static_cast<unsigned long long>(counts.pass_submissions),
        counts.pass_slots_live, live_routes,
        static_cast<unsigned long long>(counts.gang_lease),
        static_cast<unsigned long long>(completion_consumed),
        static_cast<unsigned long long>(completion_edge),
        view.first.state->event_done.load(std::memory_order_acquire),
        view.second.state->event_done.load(std::memory_order_acquire));
    return LFM_STATUS_INTERNAL;
}

int close_gate(Gate *gate, char *error, size_t error_length) {
    int result = 0;
    const char *failure = nullptr;
    auto record = [&](int status, const char *operation) {
        if (result != 0 || status == 0) return;
        result = status;
        failure = operation;
    };
    auto playback = [&](LfmPlaybackConsumer **consumer,
                        const char *operation) {
        if (!*consumer) return;
        const int status = lfm_playback_consumer_destroy(*consumer);
        record(status, operation);
        *consumer = nullptr;
    };
    playback(&gate->first_playback, "destroy first playback consumer");
    playback(&gate->second_playback, "destroy second playback consumer");
    struct SessionClose {
        LfmSession **session;
        const char *join;
        const char *destroy;
    };
    for (const SessionClose close : {
             SessionClose{&gate->first_session, "join first session",
                          "destroy first session"},
             SessionClose{&gate->second_session, "join second session",
                          "destroy second session"},
         }) {
        LfmSession **session = close.session;
        if (!*session) continue;
        lfm_session_request_stop(*session);
        int status = lfm_session_join(*session);
        record(status, close.join);
        status = lfm_session_destroy(*session);
        record(status, close.destroy);
        *session = nullptr;
    }
    gate->runtime_diagnostics = {};
    if (gate->continuation) {
        /* Public completion deliberately precedes DONE so its callback context
         * remains retained. Once the sessions' own continuations have retired,
         * this administrative latch proves the gate worker returned and
         * published DONE before unregistering its frame. */
        const int status = kc_runtime_join_all(
            lfm_internal_runtime_coordination(gate->runtime));
        record(status, "drain coordination runtime");
    }
    if (gate->continuation) {
        const int status = koro_cont_destroy(gate->continuation);
        record(status, "destroy gate continuation");
        gate->continuation = nullptr;
    }
    struct ConversationClose {
        LfmConversation **conversation;
        const char *operation;
    };
    for (const ConversationClose close : {
             ConversationClose{&gate->first_conversation,
                               "close first conversation"},
             ConversationClose{&gate->second_conversation,
                               "close second conversation"},
         }) {
        LfmConversation **conversation = close.conversation;
        if (!*conversation) continue;
        const int status = lfm_runtime_conversation_close(
            gate->runtime, *conversation);
        record(status, close.operation);
        *conversation = nullptr;
    }
    delete[] gate->closed_loop_pcm;
    gate->closed_loop_pcm = nullptr;
    gate->closed_loop_frames = 0;
    if (result != 0 && error && error_length != 0 && error[0] == '\0') {
        char message[128]{};
        std::snprintf(message, sizeof(message),
                      "native gate teardown failed during %s: %d",
                      failure ? failure : "unknown operation", result);
        copy_error(error, error_length, message);
    }
    return result;
}

int run_once(Gate *gate, uint64_t run, uint32_t audible, GateEvidence *evidence,
             char *error, size_t error_length) {
#if !defined(__APPLE__)
    (void)gate;
    (void)run;
    (void)audible;
    (void)evidence;
    copy_error(error, error_length,
               "native speech gate currently requires macOS GCD deadlines");
    return LFM_STATUS_UNSUPPORTED;
#else
    gate->first_edge.events.head.value.store(0, std::memory_order_relaxed);
    gate->first_edge.events.tail.value.store(0, std::memory_order_relaxed);
    gate->first_edge.blocked.store(false, std::memory_order_relaxed);
    gate->second_edge.events.head.value.store(0, std::memory_order_relaxed);
    gate->second_edge.events.tail.value.store(0, std::memory_order_relaxed);
    gate->second_edge.blocked.store(false, std::memory_order_relaxed);
    gate->first_edge.gate = gate;
    gate->first_edge.index = 0;
    gate->second_edge.gate = gate;
    gate->second_edge.index = 1;
    gate->submitted.store(false, std::memory_order_relaxed);
    gate->external_failure.store(0, std::memory_order_relaxed);
    gate->terminal_edges.store(0, std::memory_order_relaxed);
    gate->diagnostics.publications.store(0, std::memory_order_relaxed);
    gate->diagnostics.outcome.store(0, std::memory_order_relaxed);
    gate->diagnostics.status.store(0, std::memory_order_relaxed);
    gate->diagnostics.first_terminals.store(0, std::memory_order_relaxed);
    gate->diagnostics.second_terminals.store(0, std::memory_order_relaxed);
    gate->diagnostics.first_stopped.store(0, std::memory_order_relaxed);
    gate->diagnostics.second_stopped.store(0, std::memory_order_relaxed);
    gate->diagnostics.stop_requested.store(0, std::memory_order_relaxed);
    gate->runtime_diagnostics = {};
    gate->first_ticket = {};
    gate->audible = audible != 0;
    gate->monitor.head.store(0, std::memory_order_relaxed);
    gate->monitor.tail.store(0, std::memory_order_relaxed);
    gate->monitor.callback_gate.store(0, std::memory_order_relaxed);
    gate->monitor.underflow_frames.store(0, std::memory_order_relaxed);
    gate->monitor.underflow_callbacks.store(0, std::memory_order_relaxed);
    gate->monitor.closing.store(false, std::memory_order_relaxed);
    gate->monitor.drained.store(false, std::memory_order_relaxed);
    gate->monitor.active.store(false, std::memory_order_relaxed);
    gate->monitor.started.store(false, std::memory_order_relaxed);
    gate->monitor.source = 0;
    gate->monitor.source_transitions = 0;
    gate->monitor.streaming = audible == 2;
    gate->closed_loop_frames = 0;

    char native_error[512]{};
    LfmConversationOptionsV1 first_options = conversation_options(0x51d7u);
    LfmConversationOptionsV1 second_options = conversation_options(0x7a11u);
    gate->closed_loop_pcm = new (std::nothrow) float[CLOSED_LOOP_CAPACITY];
    int status = gate->closed_loop_pcm ? 0 : LFM_STATUS_OUT_OF_MEMORY;
    if (status == 0) status = monitor_create(gate);
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            gate->runtime, gate->model, &first_options,
            &gate->first_conversation, native_error, sizeof(native_error));
    }
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            gate->runtime, gate->model, &second_options,
            &gate->second_conversation, native_error, sizeof(native_error));
    }
    const LfmCallbacksV1 first_callbacks = {
        .size = sizeof(LfmCallbacksV1),
        .abi_version = ABI,
        .context = &gate->first_edge,
        .on_event = event_callback,
    };
    const LfmCallbacksV1 second_callbacks = {
        .size = sizeof(LfmCallbacksV1),
        .abi_version = ABI,
        .context = &gate->second_edge,
        .on_event = event_callback,
    };
    LfmSessionConfigV1 first_config = session_config(run * 2 + 1);
    LfmSessionConfigV1 second_config = session_config(run * 2 + 2);
    if (status == 0) {
        status = lfm_session_create(
            gate->runtime, gate->model, gate->first_conversation,
            &first_config, &first_callbacks, &gate->first_session);
    }
    if (status == 0) {
        status = lfm_session_create(
            gate->runtime, gate->model, gate->second_conversation,
            &second_config, &second_callbacks, &gate->second_session);
    }
    gate->first_edge.session = gate->first_session;
    gate->second_edge.session = gate->second_session;
    if (status == 0) {
        status = lfm_playback_consumer_create(
            gate->first_session, &gate->first_playback);
    }
    if (status == 0) {
        status = lfm_playback_consumer_create(
            gate->second_session, &gate->second_playback);
    }
    if (status == 0) {
        status = lfm_internal_runtime_diagnostic_view(
            gate->runtime, gate->first_session, gate->second_session,
            &gate->runtime_diagnostics);
    }

    const koro_cont_config continuation = {
        .step = gate_step,
        .argument = gate,
        .frame_size = sizeof(GateFrame),
        .worker_mask = 0,
        .completion = gate_retired,
        .completion_context = gate,
    };
    if (status == 0) {
        status = koro_cont_create_on(
            lfm_internal_runtime_coordination(gate->runtime), &continuation,
            &gate->continuation);
    }
    GateFrame *frame = gate->continuation
        ? static_cast<GateFrame *>(koro_cont_frame(gate->continuation))
        : nullptr;
    if (status == 0 && !frame) status = LFM_STATUS_INTERNAL;
    if (status == 0) gate->identity = koro_cont_identity(gate->continuation);
    if (status == 0) status = lfm_session_start(gate->first_session);
    if (status == 0) status = lfm_session_start(gate->second_session);
    if (status == 0 && gate->monitor.streaming) {
        status = monitor_start(gate);
    }

    gate->runloop = CFRunLoopGetCurrent();
    if (gate->runloop) CFRetain(gate->runloop);
    if (status == 0 && !gate->runloop) status = LFM_STATUS_INTERNAL;
    if (status == 0) {
        CFRunLoopSourceContext source{};
        gate->runloop_source =
            CFRunLoopSourceCreate(nullptr, 0, &source);
        if (!gate->runloop_source) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (status == 0) {
        CFRunLoopAddSource(gate->runloop, gate->runloop_source,
                           kCFRunLoopDefaultMode);
    }
    static constexpr char prompt[] =
        "Greet another voice assistant in one short spoken sentence.";
    if (status == 0) {
        status = lfm_session_submit_text(
            gate->first_session, prompt, sizeof(prompt) - 1,
            &gate->first_ticket);
    }
    gate->submitted.store(status == 0, std::memory_order_release);
    if (status == 0) status = koro_cont_start(gate->continuation);
    const bool continuation_started = status == 0;
    dispatch_source_t watchdog = nullptr;
    if (continuation_started) {
        watchdog = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
        if (!watchdog) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (watchdog) {
        dispatch_set_context(watchdog, gate);
        dispatch_source_set_event_handler_f(watchdog, watchdog_fired);
        dispatch_source_set_cancel_handler_f(watchdog, watchdog_cancelled);
        dispatch_source_set_timer(
            watchdog,
            dispatch_time(DISPATCH_TIME_NOW,
                          static_cast<int64_t>(WATCHDOG_NS)),
            DISPATCH_TIME_FOREVER, 0);
        dispatch_resume(watchdog);
        gate->watchdog.store(watchdog, std::memory_order_release);
        if ((gate->terminal_edges.load(std::memory_order_acquire) &
             GATE_CONTINUATION_DONE) != 0) {
            dispatch_source_cancel(watchdog);
        }
    }
    if (status != 0) {
        if (!continuation_started) {
            gate->terminal_edges.store(GATE_ALL_DONE,
                                       std::memory_order_release);
        } else {
            int32_t expected = 0;
            gate->external_failure.compare_exchange_strong(
                expected, status, std::memory_order_release,
                std::memory_order_relaxed);
            if (!watchdog) {
                publish_terminal_edge(gate, GATE_WATCHDOG_DONE);
            }
            resume_gate(gate);
        }
    }

    if (continuation_started &&
        gate->terminal_edges.load(std::memory_order_acquire) !=
            GATE_ALL_DONE) {
        CFRunLoopRun();
    }
    const int32_t external =
        gate->external_failure.load(std::memory_order_acquire);
    if (status == 0 && external != 0) {
        status = external;
        copy_error(error, error_length,
                   external == -ETIMEDOUT
                       ? "native speech gate watchdog expired"
                       : "native speech gate external callback failed");
    }
    if (status == 0 && continuation_started &&
        gate->terminal_edges.load(std::memory_order_acquire) !=
            GATE_ALL_DONE) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native gate event loop returned before terminal callbacks");
    }
    if (status == 0 && frame->status != 0) {
        status = frame->status;
        copy_error(error, error_length, frame->error);
    }
    if (status == 0) {
        status = validate_native_diagnostics(gate, error, error_length);
    }
    if (status == 0) {
        evidence->first_hash = frame->first_pcm.hash;
        evidence->second_hash = frame->second_pcm.hash;
        evidence->first_frames = frame->first_pcm.frames;
        evidence->second_frames = frame->second_pcm.frames;
        evidence->first_nonzero = frame->first_pcm.nonzero;
        evidence->second_nonzero = frame->second_pcm.nonzero;
        evidence->monitor_underflow_frames =
            gate->monitor.underflow_frames.load(std::memory_order_acquire);
        evidence->monitor_underflow_callbacks =
            gate->monitor.underflow_callbacks.load(std::memory_order_acquire);
        evidence->monitor_source_transitions =
            gate->monitor.source_transitions;
        std::memcpy(evidence->first_text, frame->first_text,
                    sizeof(evidence->first_text));
        std::memcpy(evidence->second_text, frame->second_text,
                    sizeof(evidence->second_text));
    }
    if (status != 0 && gate->runtime_diagnostics.engine.state) {
        print_native_watchdog(gate);
    }
    const int closed = close_gate(gate, error, error_length);
    monitor_destroy(gate);
    watchdog = gate->watchdog.exchange(nullptr, std::memory_order_acq_rel);
    if (watchdog) {
#if !OS_OBJECT_USE_OBJC
        dispatch_release(watchdog);
#endif
    }
    if (gate->runloop) {
        if (gate->runloop_source) {
            CFRunLoopRemoveSource(gate->runloop, gate->runloop_source,
                                  kCFRunLoopDefaultMode);
            CFRunLoopSourceInvalidate(gate->runloop_source);
            CFRelease(gate->runloop_source);
            gate->runloop_source = nullptr;
        }
        CFRelease(gate->runloop);
        gate->runloop = nullptr;
    }
    return status != 0 ? status : closed;
#endif
}

bool accounting_equal(const LfmModelMemoryV2 &a,
                      const LfmModelMemoryV2 &b) {
    return a.source_bytes == b.source_bytes &&
           a.segment_bytes == b.segment_bytes &&
           a.segment_constructed_bytes == b.segment_constructed_bytes &&
           a.attached_shared_bytes == b.attached_shared_bytes &&
           a.wired_bytes == b.wired_bytes &&
           a.process_resident_bytes == b.process_resident_bytes &&
           a.directly_bound_bytes == b.directly_bound_bytes &&
           a.derived_immutable_bytes == b.derived_immutable_bytes &&
           a.materialized_weight_bytes == b.materialized_weight_bytes &&
           a.compatibility_copied_bytes == b.compatibility_copied_bytes &&
           a.payload_read_calls == b.payload_read_calls &&
           a.payload_read_bytes == b.payload_read_bytes &&
           a.post_publication_read_calls == b.post_publication_read_calls &&
           a.post_publication_read_bytes == b.post_publication_read_bytes &&
           a.post_publication_materialization_attempts ==
               b.post_publication_materialization_attempts &&
           a.post_publication_materialization_bytes ==
               b.post_publication_materialization_bytes &&
           a.publication_generation == b.publication_generation &&
           a.weight_build_ns == b.weight_build_ns &&
           a.weight_attach_ns == b.weight_attach_ns &&
           a.weight_generation == b.weight_generation &&
           a.weight_flags == b.weight_flags &&
           a.weight_source_count == b.weight_source_count &&
           a.weight_payload_read_calls == b.weight_payload_read_calls &&
           a.weight_payload_read_bytes == b.weight_payload_read_bytes &&
           std::memcmp(a.weight_identity_digest, b.weight_identity_digest,
                       sizeof(a.weight_identity_digest)) == 0 &&
           std::memcmp(a.weight_content_digest, b.weight_content_digest,
                       sizeof(a.weight_content_digest)) == 0 &&
           a.post_readiness_allocation_attempts ==
               b.post_readiness_allocation_attempts &&
           a.post_readiness_allocation_bytes ==
               b.post_readiness_allocation_bytes;
}

} // namespace

extern "C" int lfm_native_speech_to_speech_gate(
    const char *model_path, uint32_t audible, uint32_t kernel_lanes, char *error,
    size_t error_length) {
    if (!model_path || !*model_path || audible > 2 || kernel_lanes == 0 || !error ||
        error_length == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    error[0] = '\0';
    int status = 0;
    Gate gate{};
    const LfmRuntimeConfigV1 config = {
        .size = sizeof(LfmRuntimeConfigV1),
        .abi_version = ABI,
        .coordination_workers = 2,
        .kernel_lanes = kernel_lanes,
        .event_capacity = 64,
        .session_capacity = 2,
        .reserved0 = 0,
        .reserved1 = 0,
        .flags = 0,
        .reserved = {},
    };
    status = lfm_runtime_create(&config, &gate.runtime);
    if (status == 0) status = lfm_runtime_start(gate.runtime);
    if (status == 0) {
        status = lfm_runtime_model_open(
            gate.runtime, model_path, &gate.model, error, error_length);
    }
    if (status == 0) {
        gate.before = {
            .size = sizeof(LfmModelMemoryV2),
            .abi_version = LFM_MODEL_ABI_VERSION,
        };
        status = lfm_runtime_model_memory(gate.runtime, gate.model,
                                          &gate.before);
    }
    if (status == 0 &&
        (gate.before.compatibility_copied_bytes != 0 ||
         gate.before.materialized_weight_bytes != 0 ||
         gate.before.post_publication_read_calls != 0 ||
         gate.before.post_publication_materialization_attempts != 0)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native model accounting was dirty before generation");
    }

    GateEvidence first{};
    GateEvidence second{};
    if (status == 0) {
        status = run_once(&gate, 1, audible, &first, error, error_length);
    }
    if (status == 0) {
        status = run_once(&gate, 2, 0, &second, error, error_length);
    }
    if (status == 0 &&
        (first.first_hash != second.first_hash ||
         first.second_hash != second.second_hash ||
         first.first_frames != second.first_frames ||
         first.second_frames != second.second_frames ||
         std::strcmp(first.first_text, second.first_text) != 0 ||
         std::strcmp(first.second_text, second.second_text) != 0)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "fixed-seed native speech trace was not deterministic");
    }
    if (status == 0) {
        gate.after = {
            .size = sizeof(LfmModelMemoryV2),
            .abi_version = LFM_MODEL_ABI_VERSION,
        };
        status = lfm_runtime_model_memory(gate.runtime, gate.model,
                                          &gate.after);
    }
    if (status == 0 && !accounting_equal(gate.before, gate.after)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "model reads, weights, or allocation accounting changed after readiness");
    }
    if (status == 0) {
        std::fprintf(stderr,
                     "native speech gate: lanes=%u A=%llu frames/%llu nonzero "
                     "hash=%016llx, B=%llu frames/%llu nonzero hash=%016llx "
                     "weights=%s generation=%llu payload_reads=%llu/%llu\n"
                     "A: %s\nB: %s\n",
                     kernel_lanes,
                     static_cast<unsigned long long>(first.first_frames),
                     static_cast<unsigned long long>(first.first_nonzero),
                     static_cast<unsigned long long>(first.first_hash),
                     static_cast<unsigned long long>(first.second_frames),
                     static_cast<unsigned long long>(first.second_nonzero),
                     static_cast<unsigned long long>(first.second_hash),
                     (gate.before.weight_flags & LFM_WEIGHT_LOAD_BUILT)
                         ? "built"
                         : "attached",
                     static_cast<unsigned long long>(
                         gate.before.weight_generation),
                     static_cast<unsigned long long>(
                         gate.before.weight_payload_read_calls),
                     static_cast<unsigned long long>(
                         gate.before.weight_payload_read_bytes),
                     first.first_text, first.second_text);
        if (audible != 0) {
            std::fprintf(
                stderr,
                "speaker monitor: mode=%s underflow=%llu frames/%llu "
                "callbacks source-transitions=%u\n",
                audible == 2 ? "stream" : "buffered",
                static_cast<unsigned long long>(
                    first.monitor_underflow_frames),
                static_cast<unsigned long long>(
                    first.monitor_underflow_callbacks),
                first.monitor_source_transitions);
        }
    }

    if (gate.model) {
        const int closed = lfm_runtime_model_close(gate.runtime, gate.model);
        if (status == 0 && closed != 0) status = closed;
        gate.model = nullptr;
    }
    if (gate.runtime) {
        lfm_runtime_request_stop(gate.runtime);
        const int joined = lfm_runtime_join(gate.runtime);
        if (status == 0 && joined != 0) status = joined;
        const int destroyed = lfm_runtime_destroy(gate.runtime);
        if (status == 0 && destroyed != 0) status = destroyed;
        gate.runtime = nullptr;
    }
    if (status != 0 && error[0] == '\0') {
        char message[128]{};
        std::snprintf(message, sizeof(message),
                      "native speech-to-speech gate failed: %d", status);
        copy_error(error, error_length, message);
    }
    return status;
}
