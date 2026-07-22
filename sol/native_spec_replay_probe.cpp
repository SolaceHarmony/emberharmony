/* Native prefix-stability experiment.
 *
 * One native LFM2 conversation speaks a short utterance into a retained PCM
 * buffer.  A second, independently reset native conversation hears either the
 * complete buffer or one callback-aligned prefix and executes the ordinary
 * interleaved recurrence: text sampling, Depthformer audio-code sampling, and
 * the checkpoint audio detokenizer.  A prefix matches the oracle only when the
 * complete emitted text/audio-code trajectory and decoded PCM for one full
 * 6-text/12-audio cycle are identical.
 *
 * This deliberately re-encodes each PCM prefix.  The Conformer is
 * bidirectional over the samples it can see, so truncating already-computed
 * adapted rows would leak future audio into the supposed prefix.  Resetting a
 * preallocated conversation restores KV, ShortConv, PRNG, cursor, and codec
 * state before every attempt; only the immutable model image is shared.
 *
 * There is no operation waiter.  Route completion resumes this exact logical
 * coroutine by ticket.  The process run loop is only the outer event pump, and
 * the monotonic GCD one-shot is only a fatal test watchdog: neither one
 * advances inference.  PCM never leaves memory. */

#include "kcoro_stackless.h"
#include "lfm_detokenizer.h"
#include "lfm_model_internal.h"
#include "lfm_route_epoch.h"
#include "lfm_runtime.h"
#include "lfm_runtime_internal.h"

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

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#endif

namespace {

constexpr uint32_t ABI = LFM_RUNTIME_ABI_VERSION;
constexpr uint32_t RATE = 24000;
constexpr uint32_t CALLBACK_FRAMES = 480;       // 20 ms at 24 kHz.
constexpr uint32_t TRACE_EVENTS = 18;           // One 6-text/12-audio cycle.
constexpr uint32_t MAX_PREFIXES = 12;
constexpr uint32_t SOURCE_EVENT_CAP = 8192;
constexpr size_t SOURCE_CAPACITY = size_t{30} * RATE;
constexpr size_t MIN_PREFIX = size_t{2} * RATE / 5; // 400 ms.
constexpr size_t BASE_STEP = RATE / 5;              // 200 ms.
constexpr uint64_t WATCHDOG_NS = UINT64_C(180) * UINT64_C(1000000000);
constexpr uint64_t FNV_OFFSET = UINT64_C(1469598103934665603);
constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);
constexpr uint32_t CONTINUATION_DONE = 1u << 0;
constexpr uint32_t WATCHDOG_DONE = 1u << 1;
constexpr uint32_t ALL_DONE = CONTINUATION_DONE | WATCHDOG_DONE;

constexpr char SOURCE_PROMPT[] =
    "Speak one short sentence to another voice assistant. Tell it that the "
    "doves came back to the balcony this morning.";

enum Run : uint32_t {
    RUN_SOURCE = 0,
    RUN_ORACLE_A = 1,
    RUN_ORACLE_B = 2,
    RUN_PREFIX = 3,
    RUN_FINISHED = 4,
};

enum Pending : uint32_t {
    PENDING_NONE = 0,
    PENDING_ADMISSION = 1,
    PENDING_TEXT = 2,
    PENDING_AUDIO = 3,
    PENDING_INTERRUPT = 4,
};

struct PcmEvidence {
    uint64_t frames;
    uint64_t nonzero;
    uint64_t hash;
    double peak;
};

struct TraceEvent {
    uint32_t kind;
    uint32_t flags;
    uint32_t text_bytes;
    uint32_t code_count;
    uint8_t text[512];
    uint32_t codes[LFM_AUDIO_TOKEN_CAPACITY];
};

struct Trace {
    TraceEvent events[TRACE_EVENTS];
    PcmEvidence pcm;
    uint32_t count;
    uint32_t text_events;
    uint32_t audio_events;
    uint32_t terminal_events;
    uint32_t text_bytes;
    char text[2048];
};

struct ProbeFrame {
    LfmConversationAdmissionHandle admission;
    LfmAudioRouteHandle route;
    LfmNativeEmission emission;
    Trace oracle_a;
    Trace oracle_b;
    Trace current;
    PcmEvidence source_pcm;
    size_t prefixes[MAX_PREFIXES];
    uint8_t matches[MAX_PREFIXES];
    size_t prefix_count;
    size_t prefix_index;
    size_t earliest;
    size_t collected_samples;
    uint32_t source_events;
    uint32_t source_text_bytes;
    char source_text[2048];
    uint32_t run;
    uint32_t pending;
    uint32_t started;
    uint32_t done;
    int32_t status;
    char error[512];
};

struct Probe {
    LfmRuntime *runtime = nullptr;
    LfmModel *model = nullptr;
    LfmConversation *source = nullptr;
    LfmConversation *branch = nullptr;
    koro_cont_t *continuation = nullptr;
    kc_ticket_id identity{};
    LfmRouteEpoch epoch;
    float *source_pcm = nullptr;
    float *branch_pcm = nullptr;
    size_t source_frames = 0;
    size_t source_step_frames = 0;
    size_t branch_step_frames = 0;
    std::atomic<uint32_t> terminal_edges{0};
#if defined(__APPLE__)
    CFRunLoopRef runloop = nullptr;
    CFRunLoopSourceRef runloop_source = nullptr;
    std::atomic<dispatch_source_t> watchdog{nullptr};
#endif
};

void copy_error(char *destination, size_t capacity, const char *source) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", source ? source : "unknown");
}

void fail(ProbeFrame *frame, int status, const char *operation) {
    if (!frame || frame->status != 0) return;
    frame->status = status == 0 ? LFM_STATUS_INTERNAL : status;
    std::snprintf(frame->error, sizeof(frame->error), "%s failed: %d",
                  operation ? operation : "native prefix experiment", status);
}

bool same_identity(const kc_ticket_id &left, const kc_ticket_id &right) {
    return left.runtime_epoch == right.runtime_epoch &&
        left.sequence == right.sequence &&
        left.generation == right.generation && left.kind == right.kind;
}

void reset_pcm(PcmEvidence *evidence) {
    if (!evidence) return;
    *evidence = {};
    evidence->hash = FNV_OFFSET;
}

int add_pcm(PcmEvidence *evidence, const float *samples, size_t count) {
    if (!evidence || (!samples && count != 0)) return -EINVAL;
    for (size_t index = 0; index < count; ++index) {
        const float sample = samples[index];
        if (!std::isfinite(sample)) return -EDOM;
        const double magnitude = std::fabs(static_cast<double>(sample));
        evidence->peak = std::max(evidence->peak, magnitude);
        if (magnitude > 1e-6) ++evidence->nonzero;
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        evidence->hash ^= bits;
        evidence->hash *= FNV_PRIME;
    }
    evidence->frames += count;
    return 0;
}

void reset_trace(Trace *trace) {
    if (!trace) return;
    *trace = {};
    reset_pcm(&trace->pcm);
}

int append_text(char *destination, uint32_t *used, size_t capacity,
                const uint8_t *text, uint32_t bytes) {
    if (!destination || !used || (!text && bytes != 0) || *used >= capacity ||
        bytes > capacity - 1 - *used) {
        return -ENOBUFS;
    }
    if (bytes != 0) std::memcpy(destination + *used, text, bytes);
    *used += bytes;
    destination[*used] = '\0';
    return 0;
}

int add_trace_event(Trace *trace, const LfmNativeEmission &emission) {
    if (!trace || emission.kind == LFM_NATIVE_EMISSION_NONE ||
        trace->count >= TRACE_EVENTS || emission.text_bytes > sizeof(emission.text) ||
        emission.code_count > LFM_AUDIO_TOKEN_CAPACITY) {
        return -EPROTO;
    }
    TraceEvent &event = trace->events[trace->count++];
    event.kind = emission.kind;
    event.flags = emission.flags;
    event.text_bytes = emission.text_bytes;
    event.code_count = emission.code_count;
    if (emission.text_bytes != 0) {
        std::memcpy(event.text, emission.text, emission.text_bytes);
    }
    if (emission.code_count != 0) {
        std::copy_n(emission.codes, emission.code_count, event.codes);
    }
    if (emission.kind == LFM_NATIVE_EMISSION_TEXT) {
        ++trace->text_events;
        return append_text(trace->text, &trace->text_bytes,
                           sizeof(trace->text), emission.text,
                           emission.text_bytes);
    }
    if (emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES) {
        ++trace->audio_events;
        return 0;
    }
    if (emission.kind == LFM_NATIVE_EMISSION_FINISHED) {
        ++trace->terminal_events;
        return 0;
    }
    return -EPROTO;
}

bool same_trace(const Trace &left, const Trace &right) {
    if (left.count != right.count || left.pcm.frames != right.pcm.frames ||
        left.pcm.hash != right.pcm.hash ||
        left.text_events != right.text_events ||
        left.audio_events != right.audio_events ||
        left.terminal_events != right.terminal_events) {
        return false;
    }
    for (uint32_t index = 0; index < left.count; ++index) {
        const TraceEvent &a = left.events[index];
        const TraceEvent &b = right.events[index];
        if (a.kind != b.kind || a.flags != b.flags ||
            a.text_bytes != b.text_bytes || a.code_count != b.code_count ||
            std::memcmp(a.text, b.text, a.text_bytes) != 0 ||
            std::memcmp(a.codes, b.codes,
                        a.code_count * sizeof(uint32_t)) != 0) {
            return false;
        }
    }
    return true;
}

size_t align_up(size_t value, size_t alignment) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return std::numeric_limits<size_t>::max();
    }
    return (value + alignment - 1) / alignment * alignment;
}

int build_prefixes(Probe *probe, ProbeFrame *frame) {
    if (!probe || !frame || probe->source_frames <= MIN_PREFIX) return -ENODATA;
    const size_t last = (probe->source_frames - 1) / CALLBACK_FRAMES *
        CALLBACK_FRAMES;
    if (last < MIN_PREFIX) return -ENODATA;
    const size_t span = last - MIN_PREFIX;
    const size_t ideal = (span + 7) / 8;
    const size_t step = align_up(std::max(BASE_STEP, ideal), CALLBACK_FRAMES);
    if (step == std::numeric_limits<size_t>::max()) return -EOVERFLOW;
    for (size_t prefix = align_up(MIN_PREFIX, CALLBACK_FRAMES);
         prefix <= last && frame->prefix_count < MAX_PREFIXES;) {
        frame->prefixes[frame->prefix_count++] = prefix;
        if (prefix > last - std::min(step, last)) break;
        prefix += step;
    }
    if (frame->prefix_count == 0) return -ENODATA;
    if (frame->prefixes[frame->prefix_count - 1] != last) {
        if (frame->prefix_count == MAX_PREFIXES) --frame->prefix_count;
        frame->prefixes[frame->prefix_count++] = last;
    }
    return 0;
}

LfmConversationOptionsV1 options(uint64_t seed) {
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

void route_complete(void *context) {
    auto *probe = static_cast<Probe *>(context);
    if (!probe || !probe->continuation) std::abort();
    const int status = koro_cont_resume(probe->continuation, &probe->identity);
    if (status != 0 && status != -ECANCELED) std::abort();
}

int submit_begin(Probe *probe, ProbeFrame *frame) {
    frame->admission = {};
    frame->emission = {};
    int status = 0;
    if (frame->run == RUN_SOURCE) {
        status = lfm_conversation_reset(probe->source);
        if (status == 0) {
            status = lfm_conversation_begin_text_submit_native(
                probe->source, SOURCE_PROMPT, sizeof(SOURCE_PROMPT) - 1,
                &frame->emission, route_complete, probe, &frame->admission);
        }
    } else {
        status = lfm_conversation_reset(probe->branch);
        const size_t samples = frame->run == RUN_PREFIX
            ? frame->prefixes[frame->prefix_index]
            : probe->source_frames;
        reset_trace(&frame->current);
        if (status == 0) {
            status = lfm_conversation_begin_pcm_submit_native(
                probe->branch, probe->source_pcm, samples, RATE,
                &frame->emission, route_complete, probe, &frame->admission);
        }
    }
    if (status != 0) return status;
    frame->pending = PENDING_ADMISSION;
    frame->started = 1;
    return 0;
}

int submit_next(Probe *probe, ProbeFrame *frame) {
    LfmConversation *conversation = frame->run == RUN_SOURCE
        ? probe->source
        : probe->branch;
    const int playback =
        lfm_conversation_next_requires_playback_native(conversation);
    if (playback < 0) return playback;
    frame->route = {};
    if (playback == 0) {
        const int status = lfm_conversation_next_submit_native(
            conversation, route_complete, probe, &frame->route);
        if (status == 0) frame->pending = PENDING_TEXT;
        return status;
    }
    float *pcm = probe->branch_pcm;
    size_t capacity = probe->branch_step_frames;
    if (frame->run == RUN_SOURCE) {
        if (probe->source_frames > SOURCE_CAPACITY ||
            probe->source_step_frames > SOURCE_CAPACITY - probe->source_frames) {
            return -ENOBUFS;
        }
        pcm = probe->source_pcm + probe->source_frames;
        capacity = SOURCE_CAPACITY - probe->source_frames;
    }
    const LfmAudioRouteTarget target = {
        .epoch = &probe->epoch,
        .expected_epoch = 1,
        .pcm = pcm,
        .pcm_capacity = capacity,
    };
    const int status = lfm_conversation_next_into_submit_native(
        conversation, &target, route_complete, probe, &frame->route);
    if (status == 0) frame->pending = PENDING_AUDIO;
    return status;
}

int submit_interrupt(Probe *probe, ProbeFrame *frame) {
    frame->route = {};
    const int status = lfm_conversation_interrupt_submit_native(
        probe->branch, route_complete, probe, &frame->route);
    if (status == 0) frame->pending = PENDING_INTERRUPT;
    return status;
}

int finish_source(Probe *probe, ProbeFrame *frame) {
    if (probe->source_frames == 0 || frame->source_pcm.nonzero == 0 ||
        !std::isfinite(frame->source_pcm.peak) ||
        frame->source_text_bytes == 0) {
        return -ENODATA;
    }
    const int status = build_prefixes(probe, frame);
    if (status != 0) return status;
    frame->run = RUN_ORACLE_A;
    frame->started = 0;
    return 0;
}

int finish_attempt(ProbeFrame *frame) {
    if (frame->run == RUN_ORACLE_A) {
        frame->oracle_a = frame->current;
        frame->run = RUN_ORACLE_B;
        frame->started = 0;
        return 0;
    }
    if (frame->run == RUN_ORACLE_B) {
        frame->oracle_b = frame->current;
        if (!same_trace(frame->oracle_a, frame->oracle_b) ||
            frame->oracle_a.count != TRACE_EVENTS ||
            frame->oracle_a.text_events == 0 ||
            frame->oracle_a.audio_events == 0 ||
            frame->oracle_a.pcm.frames == 0 ||
            frame->oracle_a.pcm.nonzero == 0) {
            return -EDOM;
        }
        frame->run = RUN_PREFIX;
        frame->prefix_index = 0;
        frame->started = 0;
        return 0;
    }
    if (frame->run != RUN_PREFIX || frame->prefix_index >= frame->prefix_count) {
        return -EPROTO;
    }
    frame->matches[frame->prefix_index] =
        same_trace(frame->current, frame->oracle_a) ? 1u : 0u;
    ++frame->prefix_index;
    frame->started = 0;
    if (frame->prefix_index < frame->prefix_count) return 0;

    bool stable = true;
    frame->earliest = SIZE_MAX;
    for (size_t index = frame->prefix_count; index-- > 0;) {
        stable = stable && frame->matches[index] != 0;
        if (stable) frame->earliest = index;
    }
    frame->run = RUN_FINISHED;
    frame->done = 1;
    return 0;
}

int process_emission(Probe *probe, ProbeFrame *frame) {
    if (frame->run == RUN_SOURCE) {
        ++frame->source_events;
        if (frame->source_events > SOURCE_EVENT_CAP) return -EOVERFLOW;
        if (frame->emission.kind == LFM_NATIVE_EMISSION_TEXT) {
            const int status = append_text(
                frame->source_text, &frame->source_text_bytes,
                sizeof(frame->source_text), frame->emission.text,
                frame->emission.text_bytes);
            if (status != 0) return status;
        }
        if (frame->emission.kind == LFM_NATIVE_EMISSION_FINISHED) {
            return finish_source(probe, frame);
        }
        return frame->emission.kind == LFM_NATIVE_EMISSION_TEXT ||
                frame->emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES
            ? 0
            : -EPROTO;
    }
    const int status = add_trace_event(&frame->current, frame->emission);
    if (status != 0) return status;
    if (frame->emission.kind == LFM_NATIVE_EMISSION_FINISHED) {
        return finish_attempt(frame);
    }
    if (frame->current.count == TRACE_EVENTS) {
        return submit_interrupt(probe, frame);
    }
    return 0;
}

int collect_pending(Probe *probe, ProbeFrame *frame) {
    int status = 0;
    const uint32_t pending = frame->pending;
    LfmConversation *conversation = frame->run == RUN_SOURCE
        ? probe->source
        : probe->branch;
    if (pending == PENDING_ADMISSION) {
        status = lfm_conversation_begin_collect_native(
            conversation, &frame->admission);
    } else if (pending == PENDING_TEXT) {
        status = lfm_conversation_next_collect_native(
            conversation, &frame->route, &frame->emission);
    } else if (pending == PENDING_AUDIO) {
        frame->collected_samples = 0;
        status = lfm_conversation_next_into_collect_native(
            conversation, &frame->route, &frame->emission,
            &frame->collected_samples);
        if (status == 0 && frame->run == RUN_SOURCE) {
            status = add_pcm(&frame->source_pcm,
                             probe->source_pcm + probe->source_frames,
                             frame->collected_samples);
            if (status == 0) probe->source_frames += frame->collected_samples;
        } else if (status == 0) {
            status = add_pcm(&frame->current.pcm, probe->branch_pcm,
                             frame->collected_samples);
        }
    } else if (pending == PENDING_INTERRUPT) {
        status = lfm_conversation_interrupt_collect_native(
            probe->branch, &frame->route);
    } else {
        return -EPROTO;
    }
    if (status == -EINPROGRESS) return -EPROTO;
    frame->pending = PENDING_NONE;
    if (status != 0) return status;
    if (pending == PENDING_INTERRUPT) return finish_attempt(frame);
    return process_emission(probe, frame);
}

int advance(Probe *probe, ProbeFrame *frame) {
    for (uint32_t transition = 0; transition < 64; ++transition) {
        if (frame->status != 0 || frame->done != 0) return 1;
        if (frame->pending != PENDING_NONE) {
            const int status = collect_pending(probe, frame);
            if (status != 0) {
                fail(frame, status, "collect native recurrence edge");
                return 1;
            }
            if (frame->pending != PENDING_NONE) return 0;
            continue;
        }
        if (frame->run == RUN_FINISHED) {
            frame->done = 1;
            return 1;
        }
        if (frame->started == 0) {
            const int status = submit_begin(probe, frame);
            if (status != 0) {
                fail(frame, status, "submit native admission");
                return 1;
            }
            return 0;
        }
        const int status = submit_next(probe, frame);
        if (status != 0) {
            fail(frame, status, "submit native recurrence edge");
            return 1;
        }
        return 0;
    }
    fail(frame, -ELOOP, "advance native prefix experiment");
    return 1;
}

void *probe_step(koro_cont_t *continuation) {
    auto *probe = static_cast<Probe *>(koro_cont_argument(continuation));
    auto *frame = static_cast<ProbeFrame *>(koro_cont_frame(continuation));
    if (!probe || !frame) std::abort();
    KORO_BEGIN(continuation);
    for (;;) {
        if (advance(probe, frame) != 0) break;
        KORO_SUSPEND(continuation);
    }
    KORO_END(continuation);
}

#if defined(__APPLE__)

void publish_terminal(Probe *probe, uint32_t edge) {
    CFRunLoopRef runloop = probe->runloop;
    if (runloop) CFRetain(runloop);
    const uint32_t prior = probe->terminal_edges.fetch_or(
        edge, std::memory_order_acq_rel);
    if ((prior | edge) == ALL_DONE && runloop) {
        CFRunLoopStop(runloop);
        CFRunLoopWakeUp(runloop);
    }
    if (runloop) CFRelease(runloop);
}

void probe_retired(void *context, const kc_ticket_id *identity) {
    auto *probe = static_cast<Probe *>(context);
    if (!probe || !identity || !same_identity(*identity, probe->identity)) {
        std::abort();
    }
    dispatch_source_t watchdog =
        probe->watchdog.load(std::memory_order_acquire);
    if (watchdog) dispatch_source_cancel(watchdog);
    publish_terminal(probe, CONTINUATION_DONE);
}

void watchdog_fired(void *context) {
    auto *probe = static_cast<Probe *>(context);
    auto *frame = probe && probe->continuation
        ? static_cast<ProbeFrame *>(koro_cont_frame(probe->continuation))
        : nullptr;
    std::fprintf(stderr,
                 "native prefix experiment watchdog: run=%u pending=%u "
                 "prefix=%zu/%zu source_frames=%zu status=%d\n",
                 frame ? frame->run : UINT32_MAX,
                 frame ? frame->pending : UINT32_MAX,
                 frame ? frame->prefix_index : 0,
                 frame ? frame->prefix_count : 0,
                 probe ? probe->source_frames : 0,
                 frame ? frame->status : -ETIMEDOUT);
    /* A watchdog is evidence of a lost edge or hung numerical generation.  It
     * never resumes the inference continuation and teardown cannot safely
     * reclaim potentially live scratch, so the focused test process dies. */
    std::abort();
}

void watchdog_cancelled(void *context) {
    publish_terminal(static_cast<Probe *>(context), WATCHDOG_DONE);
}

#endif

bool same_accounting(const LfmModelMemoryV1 &left,
                     const LfmModelMemoryV1 &right) {
    return left.source_bytes == right.source_bytes &&
        left.resident_image_bytes == right.resident_image_bytes &&
        left.directly_bound_bytes == right.directly_bound_bytes &&
        left.derived_immutable_bytes == right.derived_immutable_bytes &&
        left.materialized_weight_bytes == right.materialized_weight_bytes &&
        left.compatibility_copied_bytes == right.compatibility_copied_bytes &&
        left.payload_read_calls == right.payload_read_calls &&
        left.payload_read_bytes == right.payload_read_bytes &&
        left.post_publication_read_calls == right.post_publication_read_calls &&
        left.post_publication_read_bytes == right.post_publication_read_bytes &&
        left.post_publication_materialization_attempts ==
            right.post_publication_materialization_attempts &&
        left.post_publication_materialization_bytes ==
            right.post_publication_materialization_bytes &&
        left.publication_generation == right.publication_generation &&
        left.post_readiness_allocation_attempts ==
            right.post_readiness_allocation_attempts &&
        left.post_readiness_allocation_bytes ==
            right.post_readiness_allocation_bytes;
}

int close_probe(Probe *probe, char *error, size_t error_length) {
    int result = 0;
    const char *operation = nullptr;
    const auto record = [&](int status, const char *name) {
        if (result != 0 || status == 0) return;
        result = status;
        operation = name;
    };
    if (probe->continuation) {
        record(koro_cont_destroy(probe->continuation),
               "destroy experiment continuation");
        probe->continuation = nullptr;
    }
    if (probe->branch) {
        record(lfm_runtime_conversation_close(probe->runtime, probe->branch),
               "close branch conversation");
        probe->branch = nullptr;
    }
    if (probe->source) {
        record(lfm_runtime_conversation_close(probe->runtime, probe->source),
               "close source conversation");
        probe->source = nullptr;
    }
    delete[] probe->branch_pcm;
    probe->branch_pcm = nullptr;
    delete[] probe->source_pcm;
    probe->source_pcm = nullptr;
    if (result != 0 && error && error_length != 0 && error[0] == '\0') {
        char message[160]{};
        std::snprintf(message, sizeof(message), "%s failed: %d",
                      operation ? operation : "native experiment teardown",
                      result);
        copy_error(error, error_length, message);
    }
    return result;
}

} // namespace

extern "C" int lfm_native_spec_replay_probe_gate(
    const char *model_path, uint32_t kernel_lanes, char *evidence,
    size_t evidence_length, char *error, size_t error_length) {
    if (!model_path || !*model_path || kernel_lanes == 0 || !evidence ||
        evidence_length == 0 || !error || error_length == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    evidence[0] = '\0';
    error[0] = '\0';
#if !defined(__APPLE__)
    copy_error(error, error_length,
               "native prefix experiment requires macOS monotonic watchdogs");
    return LFM_STATUS_UNSUPPORTED;
#else
    Probe probe{};
    const LfmRuntimeConfigV1 config = {
        .size = sizeof(LfmRuntimeConfigV1),
        .abi_version = ABI,
        .coordination_workers = 2,
        .kernel_lanes = kernel_lanes,
        .event_capacity = 64,
        .session_capacity = 1,
        .reserved0 = 0,
        .reserved1 = 0,
        .flags = 0,
        .reserved = {},
    };
    int status = lfm_runtime_create(&config, &probe.runtime);
    if (status == 0) status = lfm_runtime_start(probe.runtime);
    if (status == 0) {
        status = lfm_runtime_model_open(probe.runtime, model_path, &probe.model,
                                        error, error_length);
    }
    LfmModelMemoryV1 before = {
        .size = sizeof(LfmModelMemoryV1),
        .abi_version = LFM_MODEL_ABI_VERSION,
    };
    LfmModelMemoryV1 after = before;
    const LfmConversationOptionsV1 source_options = options(0x51d7u);
    const LfmConversationOptionsV1 branch_options = options(0x7a11u);
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            probe.runtime, probe.model, &source_options, &probe.source,
            error, error_length);
    }
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            probe.runtime, probe.model, &branch_options, &probe.branch,
            error, error_length);
    }
    if (status == 0) {
        status = lfm_conversation_prepare_pcm_native(
            probe.source, SOURCE_CAPACITY, RATE, RATE,
            &probe.source_step_frames);
    }
    if (status == 0) {
        status = lfm_conversation_prepare_pcm_native(
            probe.branch, SOURCE_CAPACITY, RATE, RATE,
            &probe.branch_step_frames);
    }
    if (status == 0 &&
        (probe.source_step_frames == 0 || probe.branch_step_frames == 0)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native conversations exposed no detokenizer output span");
    }
    if (status == 0) {
        probe.source_pcm = new (std::nothrow) float[SOURCE_CAPACITY];
        probe.branch_pcm =
            new (std::nothrow) float[probe.branch_step_frames];
        if (!probe.source_pcm || !probe.branch_pcm) {
            status = LFM_STATUS_OUT_OF_MEMORY;
        }
    }
    if (status == 0) {
        status = lfm_runtime_model_memory(probe.runtime, probe.model, &before);
    }
    if (status == 0 &&
        (before.compatibility_copied_bytes != 0 ||
         before.materialized_weight_bytes != 0 ||
         before.post_publication_read_calls != 0 ||
         before.post_publication_materialization_attempts != 0)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native model image was dirty before the experiment");
    }

    const koro_cont_config continuation = {
        .size = sizeof(koro_cont_config),
        .abi_version = KC_ABI_VERSION,
        .step = probe_step,
        .argument = &probe,
        .frame_size = sizeof(ProbeFrame),
        .worker_mask = 0,
        .completion = probe_retired,
        .completion_context = &probe,
    };
    if (status == 0) {
        status = koro_cont_create_on(
            lfm_internal_runtime_coordination(probe.runtime), &continuation,
            &probe.continuation);
    }
    ProbeFrame *frame = probe.continuation
        ? static_cast<ProbeFrame *>(koro_cont_frame(probe.continuation))
        : nullptr;
    if (status == 0 && !frame) status = LFM_STATUS_INTERNAL;
    if (status == 0) {
        std::memset(frame, 0, sizeof(*frame));
        frame->run = RUN_SOURCE;
        frame->earliest = SIZE_MAX;
        reset_pcm(&frame->source_pcm);
        probe.identity = koro_cont_identity(probe.continuation);
    }

    probe.runloop = CFRunLoopGetCurrent();
    if (probe.runloop) CFRetain(probe.runloop);
    if (status == 0 && !probe.runloop) status = LFM_STATUS_INTERNAL;
    if (status == 0) {
        CFRunLoopSourceContext source{};
        probe.runloop_source = CFRunLoopSourceCreate(nullptr, 0, &source);
        if (!probe.runloop_source) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (status == 0) {
        CFRunLoopAddSource(probe.runloop, probe.runloop_source,
                           kCFRunLoopDefaultMode);
    }
    if (status == 0) status = koro_cont_start(probe.continuation);
    const bool started = status == 0;
    dispatch_source_t watchdog = nullptr;
    if (started) {
        watchdog = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
        if (!watchdog) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (watchdog) {
        dispatch_set_context(watchdog, &probe);
        dispatch_source_set_event_handler_f(watchdog, watchdog_fired);
        dispatch_source_set_cancel_handler_f(watchdog, watchdog_cancelled);
        dispatch_source_set_timer(
            watchdog,
            dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(WATCHDOG_NS)),
            DISPATCH_TIME_FOREVER, 0);
        dispatch_resume(watchdog);
        probe.watchdog.store(watchdog, std::memory_order_release);
        if ((probe.terminal_edges.load(std::memory_order_acquire) &
             CONTINUATION_DONE) != 0) {
            dispatch_source_cancel(watchdog);
        }
    }
    if (started && !watchdog) {
        publish_terminal(&probe, WATCHDOG_DONE);
    }
    if (started &&
        probe.terminal_edges.load(std::memory_order_acquire) != ALL_DONE) {
        CFRunLoopRun();
    }
    if (status == 0 && started &&
        probe.terminal_edges.load(std::memory_order_acquire) != ALL_DONE) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native event pump returned before terminal callbacks");
    }
    if (status == 0 && frame && frame->status != 0) {
        status = frame->status;
        copy_error(error, error_length, frame->error);
    }
    if (status == 0) {
        status = lfm_runtime_model_memory(probe.runtime, probe.model, &after);
    }
    if (status == 0 && !same_accounting(before, after)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "model image accounting changed during prefix replay");
    }
    if (status == 0 && frame) {
        char pattern[96]{};
        size_t used = 0;
        for (size_t index = 0; index < frame->prefix_count; ++index) {
            const int wrote = std::snprintf(
                pattern + used, sizeof(pattern) - used, "%s%zu:%c",
                index == 0 ? "" : " ",
                frame->prefixes[index] * 1000 / RATE,
                frame->matches[index] ? 'Y' : 'N');
            if (wrote < 0 || static_cast<size_t>(wrote) >=
                    sizeof(pattern) - used) {
                break;
            }
            used += static_cast<size_t>(wrote);
        }
        const long long earliest_ms = frame->earliest == SIZE_MAX
            ? -1
            : static_cast<long long>(
                  frame->prefixes[frame->earliest] * 1000 / RATE);
        std::snprintf(
            evidence, evidence_length,
            "source=%zums/%zu_frames candidates=%zu earliest_stable_ms=%lld "
            "oracle={events=%u text=%u audio=%u pcm=%llu hash=%016llx} "
            "prefixes=[%s full@%zums:Y] source_text=\"%s\" "
            "oracle_text=\"%s\"",
            probe.source_frames * 1000 / RATE, probe.source_frames,
            frame->prefix_count, earliest_ms, frame->oracle_a.count,
            frame->oracle_a.text_events, frame->oracle_a.audio_events,
            static_cast<unsigned long long>(frame->oracle_a.pcm.frames),
            static_cast<unsigned long long>(frame->oracle_a.pcm.hash), pattern,
            probe.source_frames * 1000 / RATE, frame->source_text,
            frame->oracle_a.text);
    }

    const int closed = close_probe(&probe, error, error_length);
    if (status == 0) status = closed;
    watchdog = probe.watchdog.exchange(nullptr, std::memory_order_acq_rel);
    if (watchdog) {
#if !OS_OBJECT_USE_OBJC
        dispatch_release(watchdog);
#endif
    }
    if (probe.runloop) {
        if (probe.runloop_source) {
            CFRunLoopRemoveSource(probe.runloop, probe.runloop_source,
                                  kCFRunLoopDefaultMode);
            CFRunLoopSourceInvalidate(probe.runloop_source);
            CFRelease(probe.runloop_source);
            probe.runloop_source = nullptr;
        }
        CFRelease(probe.runloop);
        probe.runloop = nullptr;
    }
    if (probe.model) {
        const int model_status =
            lfm_runtime_model_close(probe.runtime, probe.model);
        if (status == 0) status = model_status;
        probe.model = nullptr;
    }
    if (probe.runtime) {
        lfm_runtime_request_stop(probe.runtime);
        const int joined = lfm_runtime_join(probe.runtime);
        if (status == 0) status = joined;
        const int destroyed = lfm_runtime_destroy(probe.runtime);
        if (status == 0) status = destroyed;
        probe.runtime = nullptr;
    }
    if (status != 0 && error[0] == '\0') {
        char message[96]{};
        std::snprintf(message, sizeof(message),
                      "native prefix experiment failed: %d", status);
        copy_error(error, error_length, message);
    }
    return status;
#endif
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : std::getenv("LFM_MODEL_DIR");
    if (!model_path || !*model_path) {
        std::fprintf(stderr,
                     "usage: %s MODEL_DIRECTORY [KERNEL_LANES]\n",
                     argc > 0 && argv[0] ? argv[0]
                                              : "native_spec_replay_probe");
        return 2;
    }
    uint32_t lanes = 8;
    const char *lane_text = argc > 2
        ? argv[2]
        : std::getenv("LFM_SPEECH_GATE_LANES");
    if (lane_text && *lane_text) {
        char *end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(lane_text, &end, 10);
        if (errno != 0 || !end || *end != '\0' || parsed == 0 ||
            parsed > std::numeric_limits<uint32_t>::max()) {
            std::fprintf(stderr, "invalid kernel lane count: %s\n", lane_text);
            return 2;
        }
        lanes = static_cast<uint32_t>(parsed);
    }
    char evidence[4096]{};
    char error[1024]{};
    const int status = lfm_native_spec_replay_probe_gate(
        model_path, lanes, evidence, sizeof(evidence), error, sizeof(error));
    if (status != 0) {
        std::fprintf(stderr, "native prefix experiment failed (%d): %s\n",
                     status, error[0] ? error : "no diagnostic");
        return 1;
    }
    std::fprintf(stdout, "%s\n", evidence);
    return 0;
}
