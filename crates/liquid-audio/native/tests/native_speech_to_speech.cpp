#include "kcoro_stackless.h"
#include "lfm_audio_dock.h"
#include "lfm_detokenizer.h"
#include "lfm_detokenizer_program.h"
#include "lfm_runtime.h"
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
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#endif

namespace {

constexpr uint32_t RATE = 24000;
constexpr uint32_t CALLBACK_FRAMES = 480;
constexpr uint32_t EVENT_CAPACITY = 128;
constexpr uint32_t EVENT_PAYLOAD = 512;
constexpr uint32_t EVENT_BUDGET = 24;
constexpr uint32_t KERNEL_LANES = 8;
/* Production interleaved-turn budget. This is deliberately much larger than
 * the vendor demo's 1024-step guard: the model owns its terminal, and this
 * test must expose—not manufacture—a cutoff before that terminal. The bound
 * remains below the model card's 32,768-token conversation context. */
constexpr uint32_t MAX_TOKENS = 8192;
constexpr uint64_t CLOSED_LOOP_CAPACITY = UINT64_C(30) * RATE;
constexpr uint64_t WATCHDOG_NS = UINT64_C(120) * UINT64_C(1000000000);
constexpr uint64_t FNV_OFFSET = UINT64_C(1469598103934665603);
constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);
constexpr uint32_t TEST_CONTINUATION_DONE = 1u << 0;
constexpr uint32_t TEST_WATCHDOG_DONE = 1u << 1;
constexpr uint32_t TEST_ALL_DONE =
    TEST_CONTINUATION_DONE | TEST_WATCHDOG_DONE;

void copy_error(char *destination, size_t capacity, const char *source);

struct SpeechTest;

struct SpeechEvent {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t session_id = 0;
    uint64_t epoch = 0;
    LfmTicketId ticket{};
    uint32_t payload_bytes = 0;
    int32_t status = 0;
    unsigned char payload[EVENT_PAYLOAD]{};
};

struct alignas(128) EventCursor {
    std::atomic<uint64_t> value{0};
};

struct EventRing {
    SpeechEvent records[EVENT_CAPACITY]{};
    EventCursor head;
    EventCursor tail;
};

struct SessionEdge {
    SpeechTest *test = nullptr;
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

struct SpeechFrame {
    PcmEvidence first_pcm;
    PcmEvidence second_pcm;
    LfmTicketId second_ticket{};
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

struct SpeechEvidence {
    uint64_t first_hash = 0;
    uint64_t second_hash = 0;
    uint64_t first_frames = 0;
    uint64_t second_frames = 0;
    uint64_t first_nonzero = 0;
    uint64_t second_nonzero = 0;
    char first_text[4096]{};
    char second_text[4096]{};
};

struct SpeechTest {
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
    LfmTicketId first_ticket{};
    std::atomic<bool> submitted{false};
    std::atomic<int32_t> external_failure{0};
    std::atomic<uint32_t> terminal_edges{0};
    float *closed_loop_pcm = nullptr;
    uint64_t closed_loop_frames = 0;
    LfmModelMemory before{};
    LfmModelMemory after{};
    float sink[CALLBACK_FRAMES]{};
#if defined(__APPLE__)
    CFRunLoopRef runloop = nullptr;
    CFRunLoopSourceRef runloop_source = nullptr;
    std::atomic<dispatch_source_t> watchdog{nullptr};
#endif
};

void resume_test(SpeechTest *test);

bool ticket_equal(const LfmTicketId &a, const LfmTicketId &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

bool ring_push(EventRing *ring, const SpeechEvent &event) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == EVENT_CAPACITY) return false;
    ring->records[tail % EVENT_CAPACITY] = event;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool ring_pop(EventRing *ring, SpeechEvent *event) {
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

void fail(SpeechFrame *frame, int32_t status, const char *message) {
    if (!frame || frame->status != 0) return;
    frame->status = status == 0 ? LFM_STATUS_INTERNAL : status;
    std::snprintf(frame->error, sizeof(frame->error), "%s", message);
}

void fail_status(SpeechFrame *frame, int32_t status, const char *operation) {
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
                 const SpeechEvent &event) {
    if (event.payload_bytes > 4095 - *used) return false;
    std::memcpy(destination + *used, event.payload, event.payload_bytes);
    *used += event.payload_bytes;
    destination[*used] = '\0';
    return true;
}

void resume_test(SpeechTest *test) {
    if (!test || !test->continuation) return;
    const int status = koro_cont_resume(test->continuation, &test->identity);
    if (status != 0 && status != -ECANCELED) {
        int32_t expected = 0;
        test->external_failure.compare_exchange_strong(
            expected, status, std::memory_order_release,
            std::memory_order_relaxed);
    }
}

int event_callback(void *context, const LfmEvent *source) {
    auto *edge = static_cast<SessionEdge *>(context);
    if (!edge || !edge->test || !source ||
        source->payload_bytes > EVENT_PAYLOAD ||
        (source->payload_bytes != 0 && !source->payload)) {
        return LFM_STATUS_HOST_SINK;
    }
    SpeechEvent event{};
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
        resume_test(edge->test);
        return LFM_STATUS_WOULD_BLOCK;
    }
    resume_test(edge->test);
    return 0;
}

int drain_first_playback(SpeechTest *test, SpeechFrame *frame,
                         const SpeechEvent &event) {
    if (event.payload_bytes != sizeof(LfmPlaybackReadyEvent)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPlaybackReadyEvent ready{};
    std::memcpy(&ready, event.payload, sizeof(ready));
    LfmPcmLease lease{};
    int status = lfm_playback_consumer_claim(
        test->first_playback, &event.ticket, event.epoch, ready.lease_id,
        ready.buffer_generation, &lease);
    if (status != 0) return status;
    if (!test->closed_loop_pcm ||
        test->closed_loop_frames > CLOSED_LOOP_CAPACITY ||
        lease.frames > CLOSED_LOOP_CAPACITY - test->closed_loop_frames) {
        status = LFM_STATUS_WOULD_BLOCK;
    }
    if (status == 0) {
        float *destination =
            test->closed_loop_pcm + test->closed_loop_frames;
        LfmPlaybackRender rendered{};
        status = lfm_playback_consumer_render_f32(
            test->first_playback, &lease, 0, destination, lease.frames, 1,
            CLOSED_LOOP_CAPACITY - test->closed_loop_frames, &rendered);
        if (status == 0) {
            evidence_add(&frame->first_pcm, destination, lease.frames);
            test->closed_loop_frames += lease.frames;
        }
    }
    const int released =
        lfm_playback_consumer_release(test->first_playback, &lease);
    return status != 0 ? status : released;
}

int drain_second_playback(SpeechTest *test, SpeechFrame *frame,
                          const SpeechEvent &event) {
    if (event.payload_bytes != sizeof(LfmPlaybackReadyEvent)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPlaybackReadyEvent ready{};
    std::memcpy(&ready, event.payload, sizeof(ready));
    LfmPcmLease lease{};
    int status = lfm_playback_consumer_claim(
        test->second_playback, &event.ticket, event.epoch, ready.lease_id,
        ready.buffer_generation, &lease);
    if (status != 0) return status;
    uint32_t offset = 0;
    while (offset < lease.frames && status == 0) {
        const uint32_t count =
            std::min(CALLBACK_FRAMES, lease.frames - offset);
        LfmPlaybackRender rendered{};
        status = lfm_playback_consumer_render_f32(
            test->second_playback, &lease, offset, test->sink, count, 1,
            CALLBACK_FRAMES, &rendered);
        if (status == 0) {
            evidence_add(&frame->second_pcm, test->sink, count);
            offset += count;
        }
    }
    const int released =
        lfm_playback_consumer_release(test->second_playback, &lease);
    return status != 0 ? status : released;
}

int process_event(SpeechTest *test, SpeechFrame *frame, uint32_t endpoint,
                  const SpeechEvent &event) {
    const bool first = endpoint == 0;
    const LfmTicketId &ticket =
        first ? test->first_ticket : frame->second_ticket;
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
            if (!ticket_equal(event.ticket, test->first_ticket)) {
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
            ? drain_first_playback(test, frame, event)
            : drain_second_playback(test, frame, event);
        if (status == 0) {
            if (first) frame->first_playback_leases++;
            else frame->second_playback_leases++;
        }
        return status;
    }
    if (event.kind != LFM_EVENT_TURN ||
        event.payload_bytes != sizeof(LfmTurnEvent)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmTurnEvent turn{};
    std::memcpy(&turn, event.payload, sizeof(turn));
    if (event.status != 0) return event.status;
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
        if (!test->closed_loop_pcm || test->closed_loop_frames == 0 ||
            test->closed_loop_frames != frame->first_pcm.frames) {
            return LFM_STATUS_INTERNAL;
        }
        const LfmF32Span pcm = {
            .data = test->closed_loop_pcm,
            .length = test->closed_loop_frames,
        };
        const int submitted = lfm_internal_session_submit_pcm_spans(
            test->second_session, &pcm, 1, RATE, &event.ticket,
            &frame->second_ticket);
        if (submitted != 0) return submitted;
        frame->second_ticket_bound = true;
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

enum SpeechOutcome : uint32_t {
    TEST_SUSPEND = 0,
    TEST_YIELD = 1,
    TEST_DONE = 2,
};

uint32_t advance_test(SpeechTest *test, SpeechFrame *frame) {
    const int32_t external =
        test->external_failure.load(std::memory_order_acquire);
    if (external != 0) fail_status(frame, external, "native test watchdog");

    uint32_t drained = 0;
    bool progressed = true;
    while (drained < EVENT_BUDGET && progressed) {
        progressed = false;
        for (SessionEdge *edge : {&test->first_edge, &test->second_edge}) {
            SpeechEvent event{};
            if (drained == EVENT_BUDGET ||
                !ring_pop(&edge->events, &event)) {
                continue;
            }
            progressed = true;
            const int status = process_event(test, frame, edge->index, event);
            if (status != 0 && frame->status == 0) {
                const LfmTicketId expected = edge->index == 0
                    ? test->first_ticket : frame->second_ticket;
                frame->status = status;
                std::snprintf(
                    frame->error, sizeof(frame->error),
                    "native event failed: status=%d endpoint=%u kind=%u "
                    "ticket={%llu,%llu,%u,%u} expected={%llu,%llu,%u,%u}",
                    status, edge->index, event.kind,
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
    for (SessionEdge *edge : {&test->first_edge, &test->second_edge}) {
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
        lfm_session_request_stop(test->second_session);
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
        lfm_session_request_stop(test->first_session);
        if (!frame->second_stop_requested) {
            frame->second_stop_requested = true;
            lfm_session_request_stop(test->second_session);
        }
    }
    if (frame->stop_requested && frame->first_stopped &&
        frame->second_stopped) {
        return TEST_DONE;
    }
    if (ring_ready(test->first_edge.events) ||
        ring_ready(test->second_edge.events) ||
        drained == EVENT_BUDGET) {
        return TEST_YIELD;
    }
    return TEST_SUSPEND;
}

void *test_step(koro_cont_t *continuation) {
    auto *test = static_cast<SpeechTest *>(koro_cont_argument(continuation));
    auto *frame = static_cast<SpeechFrame *>(koro_cont_frame(continuation));
    if (!test || !frame) std::abort();
    KORO_BEGIN(continuation);
    for (;;) {
        if (!test->submitted.load(std::memory_order_acquire)) {
            KORO_SUSPEND(continuation);
        }
        frame->outcome = advance_test(test, frame);
        if (frame->outcome == TEST_DONE) break;
        if (frame->outcome == TEST_YIELD) {
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

void publish_terminal_edge(SpeechTest *test, uint32_t edge) {
    CFRunLoopRef runloop = test->runloop;
    if (runloop) CFRetain(runloop);
    const bool failed =
        test->external_failure.load(std::memory_order_acquire) != 0;
    /* This fetch-or is the publisher's final SpeechTest access. The second edge
     * owns the run-loop wake using its separately-retained local handle. */
    const uint32_t prior = test->terminal_edges.fetch_or(
        edge, std::memory_order_acq_rel);
    if (((prior | edge) == TEST_ALL_DONE || failed) && runloop) {
        CFRunLoopStop(runloop);
        CFRunLoopWakeUp(runloop);
    }
    if (runloop) CFRelease(runloop);
}

void test_retired(void *context, const kc_ticket_id *identity) {
    auto *test = static_cast<SpeechTest *>(context);
    if (!test || !identity || !ticket_equal(*identity, test->identity)) {
        std::abort();
    }
    dispatch_source_t watchdog =
        test->watchdog.load(std::memory_order_acquire);
    if (watchdog) dispatch_source_cancel(watchdog);
    publish_terminal_edge(test, TEST_CONTINUATION_DONE);
}

void print_native_watchdog(SpeechTest *test) {
    koro_cont_snapshot gate_cont{};
    const int gate_status =
        koro_cont_snapshot_get(test->continuation, &gate_cont);
    const uint64_t gate_first_head =
        test->first_edge.events.head.value.load(std::memory_order_acquire);
    const uint64_t gate_first_tail =
        test->first_edge.events.tail.value.load(std::memory_order_acquire);
    const uint64_t gate_second_head =
        test->second_edge.events.head.value.load(std::memory_order_acquire);
    const uint64_t gate_second_tail =
        test->second_edge.events.tail.value.load(std::memory_order_acquire);

    std::fprintf(
        stderr,
        "native speech watchdog: test={rc=%d state=%u wake=%u worker=%u "
        "rings=%llu:%llu/%llu:%llu edges=%u}\n",
        gate_status, gate_cont.run_state,
        gate_cont.wake_pending, gate_cont.current_worker,
        static_cast<unsigned long long>(gate_first_head),
        static_cast<unsigned long long>(gate_first_tail),
        static_cast<unsigned long long>(gate_second_head),
        static_cast<unsigned long long>(gate_second_tail),
        test->terminal_edges.load(std::memory_order_acquire));
}

void watchdog_fired(void *context) {
    auto *test = static_cast<SpeechTest *>(context);
    if (test) {
        std::fprintf(
            stderr,
            "native speech watchdog: terminal_edges=%u\n",
            test->terminal_edges.load(std::memory_order_acquire));
        print_native_watchdog(test);
        int32_t expected = 0;
        test->external_failure.compare_exchange_strong(
            expected, -ETIMEDOUT, std::memory_order_release,
            std::memory_order_relaxed);
    }
    /* A watchdog is not an inference successor. Returning to close_test would
     * immediately enter administrative joins and let the deadlock that fired
     * this watchdog defeat its bound. Terminate the test process here: no
     * continuation is resumed, no model state advances, and no callback can
     * outlive stack-owned SpeechTest storage. */
    std::abort();
}

void watchdog_cancelled(void *context) {
    auto *test = static_cast<SpeechTest *>(context);
    publish_terminal_edge(test, TEST_WATCHDOG_DONE);
}

#endif

LfmConversationOptions conversation_options(uint64_t seed) {
    return {
        .flags = 0,
        .seed = seed,
        .text = {
            .flags = LFM_SAMPLING_GREEDY,
            .top_k = 1,
            .temperature = 0.0,
        },
        .audio = {
            .flags = 0,
            .top_k = 4,
            .temperature = 1.0,
        },
    };
}

LfmSessionConfig session_config(uint64_t id) {
    return {
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
    };
}

void copy_error(char *destination, size_t capacity, const char *source) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", source ? source : "unknown");
}

int close_test(SpeechTest *test, char *error, size_t error_length) {
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
    playback(&test->first_playback, "destroy first playback consumer");
    playback(&test->second_playback, "destroy second playback consumer");
    struct SessionClose {
        LfmSession **session;
        const char *join;
        const char *destroy;
    };
    for (const SessionClose close : {
             SessionClose{&test->first_session, "join first session",
                          "destroy first session"},
             SessionClose{&test->second_session, "join second session",
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
    if (test->continuation) {
        /* Public completion deliberately precedes DONE so its callback context
         * remains retained. Once the sessions' own continuations have retired,
         * this administrative latch proves the test worker returned and
         * published DONE before unregistering its frame. */
        const int status = kc_runtime_join_all(
            lfm_internal_runtime_coordination(test->runtime));
        record(status, "drain coordination runtime");
    }
    if (test->continuation) {
        const int status = koro_cont_destroy(test->continuation);
        record(status, "destroy test continuation");
        test->continuation = nullptr;
    }
    struct ConversationClose {
        LfmConversation **conversation;
        const char *operation;
    };
    for (const ConversationClose close : {
             ConversationClose{&test->first_conversation,
                               "close first conversation"},
             ConversationClose{&test->second_conversation,
                               "close second conversation"},
         }) {
        LfmConversation **conversation = close.conversation;
        if (!*conversation) continue;
        const int status = lfm_runtime_conversation_close(
            test->runtime, *conversation);
        record(status, close.operation);
        *conversation = nullptr;
    }
    delete[] test->closed_loop_pcm;
    test->closed_loop_pcm = nullptr;
    test->closed_loop_frames = 0;
    if (result != 0 && error && error_length != 0 && error[0] == '\0') {
        char message[128]{};
        std::snprintf(message, sizeof(message),
                      "native test teardown failed during %s: %d",
                      failure ? failure : "unknown operation", result);
        copy_error(error, error_length, message);
    }
    return result;
}

int run_once(SpeechTest *test, uint64_t run, SpeechEvidence *evidence, char *error,
             size_t error_length) {
#if !defined(__APPLE__)
    (void)test;
    (void)run;
    (void)evidence;
    copy_error(error, error_length,
               "native speech test requires macOS GCD deadlines");
    return LFM_STATUS_UNSUPPORTED;
#else
    test->first_edge.events.head.value.store(0, std::memory_order_relaxed);
    test->first_edge.events.tail.value.store(0, std::memory_order_relaxed);
    test->first_edge.blocked.store(false, std::memory_order_relaxed);
    test->second_edge.events.head.value.store(0, std::memory_order_relaxed);
    test->second_edge.events.tail.value.store(0, std::memory_order_relaxed);
    test->second_edge.blocked.store(false, std::memory_order_relaxed);
    test->first_edge.test = test;
    test->first_edge.index = 0;
    test->second_edge.test = test;
    test->second_edge.index = 1;
    test->submitted.store(false, std::memory_order_relaxed);
    test->external_failure.store(0, std::memory_order_relaxed);
    test->terminal_edges.store(0, std::memory_order_relaxed);
    test->first_ticket = {};
    test->closed_loop_frames = 0;

    char native_error[512]{};
    LfmConversationOptions first_options = conversation_options(0x51d7u);
    LfmConversationOptions second_options = conversation_options(0x7a11u);
    test->closed_loop_pcm = new (std::nothrow) float[CLOSED_LOOP_CAPACITY];
    int status = test->closed_loop_pcm ? 0 : LFM_STATUS_OUT_OF_MEMORY;
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            test->runtime, test->model, &first_options,
            &test->first_conversation, native_error, sizeof(native_error));
    }
    if (status == 0) {
        status = lfm_runtime_conversation_create(
            test->runtime, test->model, &second_options,
            &test->second_conversation, native_error, sizeof(native_error));
    }
    const LfmCallbacks first_callbacks = {
        .context = &test->first_edge,
        .on_event = event_callback,
    };
    const LfmCallbacks second_callbacks = {
        .context = &test->second_edge,
        .on_event = event_callback,
    };
    LfmSessionConfig first_config = session_config(run * 2 + 1);
    LfmSessionConfig second_config = session_config(run * 2 + 2);
    if (status == 0) {
        status = lfm_session_create(
            test->runtime, test->model, test->first_conversation,
            &first_config, &first_callbacks, &test->first_session);
    }
    if (status == 0) {
        status = lfm_session_create(
            test->runtime, test->model, test->second_conversation,
            &second_config, &second_callbacks, &test->second_session);
    }
    test->first_edge.session = test->first_session;
    test->second_edge.session = test->second_session;
    if (status == 0) {
        status = lfm_playback_consumer_create(
            test->first_session, &test->first_playback);
    }
    if (status == 0) {
        status = lfm_playback_consumer_create(
            test->second_session, &test->second_playback);
    }
    const koro_cont_config continuation = {
        .step = test_step,
        .argument = test,
        .frame_size = sizeof(SpeechFrame),
        .worker_mask = 0,
        .completion = test_retired,
        .completion_context = test,
    };
    if (status == 0) {
        status = koro_cont_create_on(
            lfm_internal_runtime_coordination(test->runtime), &continuation,
            &test->continuation);
    }
    SpeechFrame *frame = test->continuation
        ? static_cast<SpeechFrame *>(koro_cont_frame(test->continuation))
        : nullptr;
    if (status == 0 && !frame) status = LFM_STATUS_INTERNAL;
    if (status == 0) test->identity = koro_cont_identity(test->continuation);
    if (status == 0) status = lfm_session_start(test->first_session);
    if (status == 0) status = lfm_session_start(test->second_session);
    test->runloop = CFRunLoopGetCurrent();
    if (test->runloop) CFRetain(test->runloop);
    if (status == 0 && !test->runloop) status = LFM_STATUS_INTERNAL;
    if (status == 0) {
        CFRunLoopSourceContext source{};
        test->runloop_source =
            CFRunLoopSourceCreate(nullptr, 0, &source);
        if (!test->runloop_source) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (status == 0) {
        CFRunLoopAddSource(test->runloop, test->runloop_source,
                           kCFRunLoopDefaultMode);
    }
    static constexpr char prompt[] =
        "Greet another voice assistant in one short spoken sentence.";
    if (status == 0) {
        status = lfm_session_submit_text(
            test->first_session, prompt, sizeof(prompt) - 1,
            &test->first_ticket);
    }
    test->submitted.store(status == 0, std::memory_order_release);
    if (status == 0) status = koro_cont_start(test->continuation);
    const bool continuation_started = status == 0;
    dispatch_source_t watchdog = nullptr;
    if (continuation_started) {
        watchdog = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
        if (!watchdog) status = LFM_STATUS_OUT_OF_MEMORY;
    }
    if (watchdog) {
        dispatch_set_context(watchdog, test);
        dispatch_source_set_event_handler_f(watchdog, watchdog_fired);
        dispatch_source_set_cancel_handler_f(watchdog, watchdog_cancelled);
        dispatch_source_set_timer(
            watchdog,
            dispatch_time(DISPATCH_TIME_NOW,
                          static_cast<int64_t>(WATCHDOG_NS)),
            DISPATCH_TIME_FOREVER, 0);
        dispatch_resume(watchdog);
        test->watchdog.store(watchdog, std::memory_order_release);
        if ((test->terminal_edges.load(std::memory_order_acquire) &
             TEST_CONTINUATION_DONE) != 0) {
            dispatch_source_cancel(watchdog);
        }
    }
    if (status != 0) {
        if (!continuation_started) {
            test->terminal_edges.store(TEST_ALL_DONE,
                                       std::memory_order_release);
        } else {
            int32_t expected = 0;
            test->external_failure.compare_exchange_strong(
                expected, status, std::memory_order_release,
                std::memory_order_relaxed);
            if (!watchdog) {
                publish_terminal_edge(test, TEST_WATCHDOG_DONE);
            }
            resume_test(test);
        }
    }

    if (continuation_started &&
        test->terminal_edges.load(std::memory_order_acquire) !=
            TEST_ALL_DONE) {
        CFRunLoopRun();
    }
    const int32_t external =
        test->external_failure.load(std::memory_order_acquire);
    if (status == 0 && external != 0) {
        status = external;
        copy_error(error, error_length,
                   external == -ETIMEDOUT
                       ? "native speech test watchdog expired"
                       : "native speech test external callback failed");
    }
    if (status == 0 && continuation_started &&
        test->terminal_edges.load(std::memory_order_acquire) !=
            TEST_ALL_DONE) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native test event loop returned before terminal callbacks");
    }
    if (status == 0 && frame->status != 0) {
        status = frame->status;
        copy_error(error, error_length, frame->error);
    }
    if (status == 0) {
        evidence->first_hash = frame->first_pcm.hash;
        evidence->second_hash = frame->second_pcm.hash;
        evidence->first_frames = frame->first_pcm.frames;
        evidence->second_frames = frame->second_pcm.frames;
        evidence->first_nonzero = frame->first_pcm.nonzero;
        evidence->second_nonzero = frame->second_pcm.nonzero;
        std::memcpy(evidence->first_text, frame->first_text,
                    sizeof(evidence->first_text));
        std::memcpy(evidence->second_text, frame->second_text,
                    sizeof(evidence->second_text));
    }
    if (status != 0 && test->continuation) print_native_watchdog(test);
    const int closed = close_test(test, error, error_length);
    watchdog = test->watchdog.exchange(nullptr, std::memory_order_acq_rel);
    if (watchdog) {
#if !OS_OBJECT_USE_OBJC
        dispatch_release(watchdog);
#endif
    }
    if (test->runloop) {
        if (test->runloop_source) {
            CFRunLoopRemoveSource(test->runloop, test->runloop_source,
                                  kCFRunLoopDefaultMode);
            CFRunLoopSourceInvalidate(test->runloop_source);
            CFRelease(test->runloop_source);
            test->runloop_source = nullptr;
        }
        CFRelease(test->runloop);
        test->runloop = nullptr;
    }
    return status != 0 ? status : closed;
#endif
}

bool accounting_equal(const LfmModelMemory &a,
                      const LfmModelMemory &b) {
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

int run_speech_test(const char *model_path, char *error,
                    size_t error_length) {
    if (!model_path || !*model_path || !error || error_length == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    error[0] = '\0';
    int status = 0;
    SpeechTest test{};
    const LfmRuntimeConfig config = {
        .coordination_workers = 2,
        .kernel_lanes = KERNEL_LANES,
        .event_capacity = 64,
        .session_capacity = 2,
        .flags = 0,
    };
    status = lfm_runtime_create(&config, &test.runtime);
    if (status == 0) status = lfm_runtime_start(test.runtime);
    if (status == 0) {
        status = lfm_runtime_model_open(
            test.runtime, model_path, &test.model, error, error_length);
    }
    if (status == 0) {
        test.before = {
        };
        status = lfm_runtime_model_memory(test.runtime, test.model,
                                          &test.before);
    }
    if (status == 0 &&
        (test.before.compatibility_copied_bytes != 0 ||
         test.before.materialized_weight_bytes != 0 ||
         test.before.post_publication_read_calls != 0 ||
         test.before.post_publication_materialization_attempts != 0)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "native model accounting was dirty before generation");
    }

    SpeechEvidence first{};
    SpeechEvidence second{};
    if (status == 0) {
        status = run_once(&test, 1, &first, error, error_length);
    }
    if (status == 0) {
        status = run_once(&test, 2, &second, error, error_length);
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
        test.after = {
        };
        status = lfm_runtime_model_memory(test.runtime, test.model,
                                          &test.after);
    }
    if (status == 0 && !accounting_equal(test.before, test.after)) {
        status = LFM_STATUS_INTERNAL;
        copy_error(error, error_length,
                   "model reads, weights, or allocation accounting changed after readiness");
    }
    if (status == 0) {
        std::fprintf(stderr,
                     "native speech test: lanes=%u A=%llu frames/%llu nonzero "
                     "hash=%016llx, B=%llu frames/%llu nonzero hash=%016llx "
                     "weights=%s generation=%llu payload_reads=%llu/%llu\n"
                     "A: %s\nB: %s\n",
                     KERNEL_LANES,
                     static_cast<unsigned long long>(first.first_frames),
                     static_cast<unsigned long long>(first.first_nonzero),
                     static_cast<unsigned long long>(first.first_hash),
                     static_cast<unsigned long long>(first.second_frames),
                     static_cast<unsigned long long>(first.second_nonzero),
                     static_cast<unsigned long long>(first.second_hash),
                     (test.before.weight_flags & LFM_WEIGHT_LOAD_BUILT)
                         ? "built"
                         : "attached",
                     static_cast<unsigned long long>(
                         test.before.weight_generation),
                     static_cast<unsigned long long>(
                         test.before.weight_payload_read_calls),
                     static_cast<unsigned long long>(
                         test.before.weight_payload_read_bytes),
                     first.first_text, first.second_text);
    }

    if (test.model) {
        const int closed = lfm_runtime_model_close(test.runtime, test.model);
        if (status == 0 && closed != 0) status = closed;
        test.model = nullptr;
    }
    if (test.runtime) {
        lfm_runtime_request_stop(test.runtime);
        const int joined = lfm_runtime_join(test.runtime);
        if (status == 0 && joined != 0) status = joined;
        const int destroyed = lfm_runtime_destroy(test.runtime);
        if (status == 0 && destroyed != 0) status = destroyed;
        test.runtime = nullptr;
    }
    if (status != 0 && error[0] == '\0') {
        char message[128]{};
        std::snprintf(message, sizeof(message),
                      "native speech-to-speech test failed: %d", status);
        copy_error(error, error_length, message);
    }
    return status;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s CHECKPOINT\n", argv[0]);
        return EXIT_FAILURE;
    }
    char error[1024]{};
    const int status = run_speech_test(argv[1], error, sizeof(error));
    if (status != 0) {
        std::fprintf(stderr, "native speech test failed (%d): %s\n", status,
                     error[0] ? error : "no diagnostic");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
