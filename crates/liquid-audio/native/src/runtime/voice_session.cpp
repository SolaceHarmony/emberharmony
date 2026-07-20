#include "lfm_audio_dock.h"
#include "lfm_runtime.h"
#include "lfm_session.h"

#include "kc_runtime.h"
#include "kc_service.h"
#include "lfm_mimi.h"
#include "lfm_model_internal.h"
#include "../model/lfm_route_epoch.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

extern "C" {
void *lfm_engine_new(int workers);
void lfm_engine_request_stop(void *engine);
void lfm_engine_free(void *engine);
/* Private pool operations used by the native coordinator and implementation
 * tests. Hardware callbacks enter only through the tracked producer/consumer
 * endpoints declared in lfm_audio_dock.h. */
int lfm_audio_dock_resolve(const LfmSession *session,
                           const LfmPcmLeaseV1 *lease,
                           const float **out_samples,
                           size_t *out_sample_count);
int lfm_audio_dock_try_playback(LfmSession *session, LfmPcmLeaseV1 *out);
int lfm_audio_dock_release(LfmSession *session,
                           const LfmPcmLeaseV1 *lease);
}

namespace {

constexpr uint32_t MAX_RUNTIME_SESSIONS = 64;
constexpr uint32_t MAX_EVENT_CAPACITY = 64;
constexpr uint32_t MAX_PCM_SLOTS = 64;
constexpr uint32_t SLOT_FREE = 0;
constexpr uint32_t SLOT_RESERVED = 1;
constexpr uint32_t SLOT_PUBLISHED = 2;
constexpr uint32_t SLOT_CONSUMING = 3;
constexpr uint32_t SLOT_RELEASING = 4;
constexpr uint32_t SLOT_RETIRED = 5;
constexpr uint32_t COMMAND_TEXT = 1;
constexpr uint32_t COMMAND_MIXED = 2;
constexpr uint32_t EMISSION_AUDIO_END = 1;
constexpr size_t EVENT_PAYLOAD_CAPACITY = 512;
constexpr uint32_t MAX_KERNEL_LANES = 16;
constexpr uint32_t SESSION_STEP_BUDGET = 16;
constexpr uint32_t ACTION_TRANSITION_BUDGET = 8;
constexpr uint32_t ACTION_PHASE_EMIT = 1;
constexpr uint32_t ACTION_PHASE_TEXT_PUBLISHED = 2;
constexpr uint32_t ACTION_PHASE_TERMINAL_PUBLISHED = 3;
constexpr uint32_t ACTION_PHASE_FAILURE_PUBLISHED = 4;
constexpr uint32_t ACTION_PHASE_NEED_ROUTE = 5;
constexpr uint32_t ACTION_PHASE_PLAYBACK_CAPACITY_PENDING = 6;
constexpr uint32_t ACTION_PHASE_ROUTE_PENDING = 7;
constexpr uint32_t ACTION_PHASE_PLAYBACK_PUBLISHED = 8;
constexpr uint32_t ACTION_PHASE_ADMISSION_PENDING = 9;
constexpr uint32_t ACTION_PHASE_INTERRUPT_PENDING = 10;
constexpr uint32_t COORDINATOR_STARTING = 0;
constexpr uint32_t COORDINATOR_RUNNING = 1;
constexpr uint32_t COORDINATOR_STOPPING = 2;
constexpr uint32_t COORDINATOR_DONE = 3;
constexpr uint64_t PUBLICATION_CLOSED = UINT64_C(1) << 63;
constexpr uint64_t PUBLICATION_COUNT_MASK = PUBLICATION_CLOSED - 1;
/* Apple arm64 and Rosetta execute on the same 128-byte cache-line hardware. */
constexpr size_t HOT_ATOMIC_BYTES = 128;

std::atomic<uint64_t> next_runtime_epoch{1};
std::atomic<uint64_t> next_session_id{1};
std::atomic<uint64_t> next_lease_nonce{1};

constexpr uint32_t LEASE_INDEX_BITS = 6;
constexpr uint32_t LEASE_DIRECTION_SHIFT = LEASE_INDEX_BITS;
constexpr uint32_t LEASE_NONCE_SHIFT = 8;
constexpr uint64_t LEASE_INDEX_MASK = (UINT64_C(1) << LEASE_INDEX_BITS) - 1;
constexpr uint64_t LEASE_NONCE_MAX = UINT64_MAX >> LEASE_NONCE_SHIFT;

template <typename T>
struct alignas(HOT_ATOMIC_BYTES) Cursor {
    std::atomic<T> value{0};
};
static_assert(alignof(Cursor<uint32_t>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Cursor<uint32_t>) == HOT_ATOMIC_BYTES);
static_assert(alignof(Cursor<uint64_t>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Cursor<uint64_t>) == HOT_ATOMIC_BYTES,
              "adjacent queue cursors must not share an Apple cache line");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "realtime ingress publication requires a lock-free packed gate");

struct PcmSlot {
    std::atomic<uint32_t> state{SLOT_FREE};
    std::atomic<uint64_t> generation{1};
    std::atomic<uint64_t> identity{0};
    float *samples = nullptr;
    uint32_t reserved_frames = 0;
    uint32_t frames = 0;
    uint32_t offset_frames = 0;
    uint32_t channels = 0;
    uint32_t sample_rate = 0;
    uint64_t stream_epoch = 0;
    LfmTicketIdV1 ticket{};
};

struct alignas(HOT_ATOMIC_BYTES) PcmRecordCell {
    std::atomic<uint64_t> sequence{0};
    LfmPcmLeaseV1 lease{};
};
static_assert(alignof(PcmRecordCell) == HOT_ATOMIC_BYTES);

struct PcmPool {
    PcmSlot *slots = nullptr;
    PcmRecordCell *ring = nullptr;
    uint32_t capacity = 0;
    uint32_t samples_per_slot = 0;
    uint32_t direction = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
    Cursor<uint32_t> cursor;
};

struct EventRecord {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t epoch = 0;
    LfmTicketIdV1 ticket{};
    int32_t status = 0;
    uint32_t payload_bytes = 0;
    uint8_t payload[EVENT_PAYLOAD_CAPACITY]{};
};

struct EventRing {
    EventRecord *records = nullptr;
    uint32_t capacity = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
};

struct TextCommand {
    LfmTicketIdV1 ticket{};
    uint64_t epoch = 0;
    uint32_t bytes = 0;
    uint32_t kind = COMMAND_TEXT;
    LfmPcmLeaseV1 capture{};
    char text[LFM_TEXT_COMMAND_MAX_BYTES]{};
};

struct alignas(HOT_ATOMIC_BYTES) TextRecordCell {
    std::atomic<uint64_t> sequence{0};
    TextCommand command{};
};
static_assert(alignof(TextRecordCell) == HOT_ATOMIC_BYTES);

struct TextRing {
    TextRecordCell *ring = nullptr;
    uint32_t capacity = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
};

bool checked_samples(uint32_t frames, uint32_t channels, size_t *out) {
    if (frames == 0 || channels == 0) return false;
    size_t count = static_cast<size_t>(frames) * static_cast<size_t>(channels);
    if (count / channels != frames || count > SIZE_MAX / sizeof(float)) return false;
    *out = count;
    return true;
}

uint64_t lease_id(uint32_t direction, uint32_t index) {
    const uint64_t nonce = next_lease_nonce.fetch_add(1, std::memory_order_relaxed);
    if (nonce == 0 || nonce > LEASE_NONCE_MAX || index > LEASE_INDEX_MASK) return 0;
    return (nonce << LEASE_NONCE_SHIFT) |
           (static_cast<uint64_t>(direction) << LEASE_DIRECTION_SHIFT) | index;
}

bool decode_lease_id(uint64_t id, uint32_t *direction, uint32_t *index) {
    const uint64_t nonce = id >> LEASE_NONCE_SHIFT;
    const uint32_t decoded_direction =
        static_cast<uint32_t>((id >> LEASE_DIRECTION_SHIFT) & 3u);
    if (nonce == 0 || (decoded_direction != LFM_PCM_LEASE_CAPTURE &&
                       decoded_direction != LFM_PCM_LEASE_PLAYBACK)) {
        return false;
    }
    *direction = decoded_direction;
    *index = static_cast<uint32_t>(id & LEASE_INDEX_MASK);
    return true;
}

bool ticket_equal(const LfmTicketIdV1 &a, const LfmTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

void pool_push(PcmPool *pool, const LfmPcmLeaseV1 &lease) {
    /* Slot reservation is the capacity lease: at most `capacity` records can
     * be published or retained at once. Publication therefore needs no second
     * contested admission gate. fetch_add gives every producer one unique FIFO
     * cell in bounded time; an earlier producer may finish later, but the
     * single consumer simply remains at that durable sequence until its final
     * publication edge arrives. */
    const uint64_t tail =
        pool->tail.value.fetch_add(1, std::memory_order_relaxed);
    PcmRecordCell *cell = &pool->ring[tail % pool->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != tail * 2) {
        /* A reserved PCM slot makes this cell ownership structural. Reaching
         * an occupied sequence is an internal accounting violation, not
         * backpressure that a realtime producer could repair by retrying. */
        std::abort();
    }
    cell->lease = lease;
    cell->sequence.store(tail * 2 + 1, std::memory_order_release);
}

bool pool_pop(PcmPool *pool, LfmPcmLeaseV1 *out) {
    const uint64_t head = pool->head.value.load(std::memory_order_relaxed);
    PcmRecordCell *cell = &pool->ring[head % pool->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != head * 2 + 1) return false;
    *out = cell->lease;
    cell->sequence.store((head + pool->capacity) * 2,
                         std::memory_order_release);
    pool->head.value.store(head + 1, std::memory_order_relaxed);
    return true;
}

bool pool_peek(const PcmPool *pool, LfmPcmLeaseV1 *out,
               uint64_t *out_head) {
    const uint64_t head = pool->head.value.load(std::memory_order_relaxed);
    const PcmRecordCell *cell = &pool->ring[head % pool->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != head * 2 + 1) {
        return false;
    }
    *out = cell->lease;
    *out_head = head;
    return true;
}

void pool_retire_peeked(PcmPool *pool, uint64_t head) {
    /* Playback has one structurally-owned consumer. The exact head observed by
     * that consumer therefore cannot move before this retirement. */
    if (pool->head.value.load(std::memory_order_relaxed) != head) std::abort();
    PcmRecordCell *cell = &pool->ring[head % pool->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != head * 2 + 1) {
        std::abort();
    }
    cell->sequence.store((head + pool->capacity) * 2,
                         std::memory_order_release);
    pool->head.value.store(head + 1, std::memory_order_relaxed);
}

uint32_t pool_live(const PcmPool &pool) {
    uint32_t live = 0;
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        uint32_t state = pool.slots[i].state.load(std::memory_order_acquire);
        if (state >= SLOT_RESERVED && state <= SLOT_RELEASING) live++;
    }
    return live;
}

void pool_destroy(PcmPool *pool) {
    if (pool->slots) {
        for (uint32_t i = 0; i < pool->capacity; ++i) delete[] pool->slots[i].samples;
    }
    delete[] pool->slots;
    delete[] pool->ring;
    pool->slots = nullptr;
    pool->ring = nullptr;
}

int pool_create(PcmPool *pool, uint32_t capacity, uint32_t samples_per_slot,
                uint32_t direction) {
    pool->slots = new (std::nothrow) PcmSlot[capacity];
    pool->ring = new (std::nothrow) PcmRecordCell[capacity];
    if (!pool->slots || !pool->ring) return LFM_STATUS_OUT_OF_MEMORY;
    pool->capacity = capacity;
    pool->samples_per_slot = samples_per_slot;
    pool->direction = direction;
    for (uint32_t i = 0; i < capacity; ++i) {
        pool->ring[i].sequence.store(static_cast<uint64_t>(i) * 2,
                                     std::memory_order_relaxed);
        pool->slots[i].samples = new (std::nothrow) float[samples_per_slot];
        if (!pool->slots[i].samples) return LFM_STATUS_OUT_OF_MEMORY;
    }
    return 0;
}

int pool_slot(PcmPool *pool, const LfmPcmLeaseV1 *lease, PcmSlot **out,
              uint32_t *out_index) {
    if (!lease || lease->size != sizeof(*lease) ||
        lease->abi_version != LFM_RUNTIME_ABI_VERSION || lease->reserved != 0 ||
        lease->format != LFM_PCM_FORMAT_F32) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t direction = 0;
    uint32_t index = 0;
    if (!decode_lease_id(lease->lease_id, &direction, &index) ||
        direction != pool->direction ||
        (lease->flags & LFM_PCM_LEASE_DIRECTION_MASK) != direction ||
        (lease->flags & ~(LFM_PCM_LEASE_DIRECTION_MASK | LFM_PCM_LEASE_TURN_END)) != 0 ||
        index >= pool->capacity) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmSlot *slot = &pool->slots[index];
    if (slot->identity.load(std::memory_order_acquire) != lease->lease_id) {
        return LFM_STATUS_STALE;
    }
    if (slot->generation.load(std::memory_order_acquire) != lease->buffer_generation) {
        return LFM_STATUS_STALE;
    }
    if (slot->stream_epoch != lease->stream_epoch) {
        return LFM_STATUS_STALE;
    }
    const uint32_t state = slot->state.load(std::memory_order_acquire);
    if ((pool->direction == LFM_PCM_LEASE_CAPTURE ||
         state != SLOT_RESERVED) &&
        !ticket_equal(slot->ticket, lease->ticket)) {
        return LFM_STATUS_STALE;
    }
    if (lease->channels != slot->channels ||
        lease->sample_rate != slot->sample_rate || lease->frames == 0 ||
        lease->frames > slot->reserved_frames ||
        (pool->direction == LFM_PCM_LEASE_CAPTURE &&
         (lease->frames != slot->frames ||
          lease->offset_bytes !=
              static_cast<uint64_t>(slot->offset_frames) *
                  slot->channels * sizeof(float)))) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    size_t samples = 0;
    const size_t offset = lease->offset_bytes / sizeof(float);
    if (!checked_samples(lease->frames, lease->channels, &samples) ||
        lease->offset_bytes % sizeof(float) != 0 ||
        offset > pool->samples_per_slot ||
        samples > pool->samples_per_slot - offset ||
        (pool->direction == LFM_PCM_LEASE_PLAYBACK && offset != 0) ||
        lease->length_bytes != samples * sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out = slot;
    if (out_index) *out_index = index;
    return 0;
}

int release_slot(PcmPool *pool, const LfmPcmLeaseV1 *lease,
                 uint32_t allowed_states = UINT32_MAX) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    uint32_t state = slot->state.load(std::memory_order_acquire);
    if (state == SLOT_FREE || state == SLOT_RELEASING || state == SLOT_RETIRED) {
        return LFM_STATUS_STALE;
    }
    if ((allowed_states & (UINT32_C(1) << state)) == 0) {
        return LFM_STATUS_BUSY;
    }
    if (!slot->state.compare_exchange_strong(
            state, SLOT_RELEASING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    slot->reserved_frames = 0;
    slot->frames = 0;
    slot->offset_frames = 0;
    slot->channels = 0;
    slot->sample_rate = 0;
    slot->stream_epoch = 0;
    slot->ticket = {};
    slot->identity.store(0, std::memory_order_relaxed);
    const uint64_t generation =
        slot->generation.load(std::memory_order_relaxed);
    if (generation == std::numeric_limits<uint64_t>::max()) {
        slot->state.store(SLOT_RETIRED, std::memory_order_release);
        return 0;
    }
    slot->generation.store(generation + 1, std::memory_order_relaxed);
    slot->state.store(SLOT_FREE, std::memory_order_release);
    return 0;
}

int claim_published(PcmPool *pool, const LfmPcmLeaseV1 *lease, PcmSlot **out) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    uint32_t expected = SLOT_PUBLISHED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_CONSUMING,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    *out = slot;
    return 0;
}

bool event_push(EventRing *ring, const EventRecord &record) {
    uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == ring->capacity) return false;
    ring->records[tail % ring->capacity] = record;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool event_pop(EventRing *ring, EventRecord *out) {
    uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->records[head % ring->capacity];
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

uint32_t event_depth(const EventRing &ring) {
    uint64_t head = ring.head.value.load(std::memory_order_acquire);
    uint64_t tail = ring.tail.value.load(std::memory_order_acquire);
    uint64_t depth = tail - head;
    return depth > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(depth);
}

bool text_push(TextRing *ring, const TextCommand &command) {
    uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    TextRecordCell *cell = &ring->ring[tail % ring->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != tail * 2) return false;
    if (!ring->tail.value.compare_exchange_strong(
            tail, tail + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
        return false;
    }
    cell->command = command;
    cell->sequence.store(tail * 2 + 1, std::memory_order_release);
    return true;
}

bool text_pop(TextRing *ring, TextCommand *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    TextRecordCell *cell = &ring->ring[head % ring->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != head * 2 + 1) return false;
    *out = cell->command;
    cell->sequence.store((head + ring->capacity) * 2,
                         std::memory_order_release);
    ring->head.value.store(head + 1, std::memory_order_relaxed);
    return true;
}

} // namespace

struct LfmRuntime {
    void *engine = nullptr;
    kc_runtime_t *coordination = nullptr;
    uint64_t epoch = 0;
    uint32_t kernel_lanes = 0;
    uint32_t event_capacity = 0;
    uint32_t session_capacity = 0;
    std::atomic<uint32_t> state{LFM_RUNTIME_CREATED};
    mutable std::mutex children_mutex;
    LfmModel *model = nullptr;
    LfmSession *sessions[MAX_RUNTIME_SESSIONS]{};
    uint32_t session_count = 0;
};

struct PreparedPlayback {
    LfmPcmLeaseV1 lease{};
    size_t samples = 0;
    bool active = false;
};

struct SessionAction {
    LfmNativeEmission emission{};
    LfmAudioRouteHandle route{};
    LfmConversationAdmissionHandle admission{};
    PreparedPlayback playback{};
    LfmPcmLeaseV1 capture{};
    LfmTicketIdV1 ticket{};
    uint64_t epoch = 0;
    uint32_t playback_count = 0;
    uint32_t emitted = 0;
    uint32_t interrupt_flags = 0;
    int32_t terminal_status = 0;
    int32_t interrupt_status = 0;
    uint32_t phase = 0;
    bool active = false;
    bool admission_pending = false;
    bool capture_active = false;
    bool route_pending = false;
    bool route_audio = false;
};

struct ResultRecord {
    EventRecord records[2]{};
    uint32_t count = 0;
    uint32_t next = 0;
    int32_t stop_after = 0;
    bool active = false;
    bool gate_epoch = false;
};

struct LfmSession {
    LfmRuntime *runtime = nullptr;
    LfmModel *model = nullptr;
    LfmConversation *conversation = nullptr;
    LfmCallbacksV1 callbacks{};
    uint64_t id = 0;
    uint32_t sample_rate = 0;
    uint32_t playback_frames = 0;
    uint32_t channels = 0;
    uint32_t max_new_tokens = 0;
    uint32_t generation = 1;
    bool dock_only = false;
    std::atomic<uint32_t> state{LFM_SESSION_CREATED};
    LfmRouteEpoch epoch{};
    std::atomic<uint64_t> sequence{1};
    std::atomic<bool> stop{false};
    /* A publication is a bounded ingress transition, not a waiter. Stop may
     * retire command/PCM queues only after every transition admitted before
     * its close edge has published or cancelled. The last publisher supplies
     * the causal edge that resumes a dormant coordinator. */
    Cursor<uint64_t> publication_gate;
    std::atomic<bool> event_done{false};
    std::atomic<bool> sink_failed{false};
    std::atomic<int32_t> terminal_status{0};
    std::atomic<uint64_t> callbacks_entered{0};
    std::atomic<uint64_t> capture_consumed{0};
    std::atomic<uint64_t> capture_stale{0};
    std::atomic<uint64_t> playback_published{0};
    std::atomic<uint64_t> playback_consumed{0};
    std::atomic<uint64_t> text_commands_accepted{0};
    std::atomic<uint64_t> text_commands_consumed{0};
    std::atomic<uint64_t> text_commands_stale{0};
    PcmPool capture;
    PcmPool playback;
    EventRing events;
    TextRing commands;
    SessionAction action;
    ResultRecord result;
    TextCommand pending_command{};
    LfmPcmLeaseV1 pending_capture{};
    bool command_pending = false;
    bool capture_pending = false;
    LfmAudioRouteHandle interrupt_route{};
    bool interrupt_pending = false;
    EventRecord delivery_record{};
    bool delivery_pending = false;
    bool stopped_staged = false;
    uint64_t applied_epoch = 1;
    uint32_t coordinator_phase = 0;
    kc_service_t *coordinator = nullptr;
    kc_service_notifier_t *coordinator_notifier = nullptr;
    kc_service_t *delivery = nullptr;
    kc_service_notifier_t *delivery_notifier = nullptr;
    bool coordinator_started = false;
    bool delivery_started = false;
    bool coordinator_done = false;
    bool delivery_done = false;
    bool services_joined = false;
    bool start_cleanup = false;
    uint32_t capture_producers = 0;
    uint32_t playback_consumers = 0;
    uint32_t control_handles = 0;
    /* Lock order is runtime.children_mutex -> lifecycle_mutex. join_mutex is
     * outermost only for concurrent join callers and is never acquired by
     * start or stop. No retained-service join holds lifecycle_mutex. */
    mutable std::mutex lifecycle_mutex;
    mutable std::condition_variable lifecycle_cv;
    mutable std::mutex join_mutex;

    ~LfmSession() {
        pool_destroy(&playback);
        pool_destroy(&capture);
        delete[] events.records;
        delete[] commands.ring;
    }
};

struct LfmCaptureProducer {
    LfmSession *session = nullptr;
    /* One device endpoint may have a bounded ping-pong set of WRITING leases.
     * The pool generation is the authoritative identity for each lease; this
     * count only prevents endpoint retirement while the callback still owns
     * any reservation. No per-callback allocation or lease table is needed. */
    std::atomic<uint32_t> active_leases{0};
};

struct LfmPlaybackConsumer {
    LfmSession *session = nullptr;
    LfmPcmLeaseV1 lease{};
    bool active = false;
};

struct LfmSessionControl {
    LfmSession *session = nullptr;
};

namespace {

PcmPool *select_pool(LfmSession *session, uint32_t direction) {
    if (direction == LFM_PCM_LEASE_CAPTURE) return &session->capture;
    if (direction == LFM_PCM_LEASE_PLAYBACK) return &session->playback;
    return nullptr;
}

const PcmPool *select_pool(const LfmSession *session, uint32_t direction) {
    if (direction == LFM_PCM_LEASE_CAPTURE) return &session->capture;
    if (direction == LFM_PCM_LEASE_PLAYBACK) return &session->playback;
    return nullptr;
}

bool producer_matches(const LfmCaptureProducer *producer,
                      const LfmPcmLeaseV1 *lease) {
    uint32_t direction = 0;
    uint32_t index = 0;
    return producer && producer->session &&
           producer->active_leases.load(std::memory_order_acquire) != 0 &&
           lease && decode_lease_id(lease->lease_id, &direction, &index) &&
           direction == LFM_PCM_LEASE_CAPTURE;
}

bool consumer_matches(const LfmPlaybackConsumer *consumer,
                      const LfmPcmLeaseV1 *lease) {
    return consumer && consumer->active && lease && consumer->session &&
           consumer->lease.lease_id == lease->lease_id &&
           consumer->lease.buffer_generation == lease->buffer_generation &&
           consumer->lease.stream_epoch == lease->stream_epoch &&
           ticket_equal(consumer->lease.ticket, lease->ticket);
}

int mixed_push(LfmSession *session, const TextCommand &command) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(&session->capture, &command.capture, &slot, nullptr);
    if (rc != 0) return rc;
    if (command.capture.stream_epoch != command.epoch ||
        !ticket_equal(command.capture.ticket, command.ticket) ||
        command.capture.sample_rate != session->sample_rate) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }

    uint32_t expected = SLOT_RESERVED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_PUBLISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    if (!text_push(&session->commands, command)) {
        expected = SLOT_PUBLISHED;
        if (!slot->state.compare_exchange_strong(
                expected, SLOT_RESERVED, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            std::abort();
        }
        return LFM_STATUS_WOULD_BLOCK;
    }
    return 0;
}

LfmTicketIdV1 next_ticket(LfmSession *session, uint32_t kind) {
    return {
        .runtime_epoch = session->runtime->epoch,
        .sequence = session->sequence.fetch_add(1, std::memory_order_relaxed),
        .generation = session->generation,
        .kind = kind,
    };
}

int prepare_reservation(LfmSession *session, uint32_t direction,
                        uint32_t frames, uint32_t sample_rate,
                        PcmPool **out_pool, uint32_t *out_rate,
                        size_t *out_samples) {
    if (!session || frames == 0 || !out_pool || !out_rate || !out_samples) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    const uint32_t rate = sample_rate == 0 ? session->sample_rate : sample_rate;
    if (rate < 8000 || rate > 192000 ||
        (direction == LFM_PCM_LEASE_CAPTURE &&
         rate != session->sample_rate)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmPool *pool = select_pool(session, direction);
    size_t samples = 0;
    if (!pool || !checked_samples(frames, session->channels, &samples) ||
        samples > pool->samples_per_slot) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out_pool = pool;
    *out_rate = rate;
    *out_samples = samples;
    return 0;
}

int reserve_slot_at(LfmSession *session, PcmPool *pool, uint32_t direction,
                    uint32_t frames, uint32_t rate, size_t samples,
                    uint32_t index, LfmPcmLeaseV1 *out) {
    PcmSlot &slot = pool->slots[index];
    uint32_t expected = SLOT_FREE;
    if (!slot.state.compare_exchange_strong(expected, SLOT_RESERVED,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    const uint64_t identity = lease_id(direction, index);
    if (identity == 0) {
        slot.state.store(SLOT_RETIRED, std::memory_order_release);
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    slot.identity.store(identity, std::memory_order_release);
    slot.reserved_frames = frames;
    slot.frames = frames;
    slot.offset_frames = 0;
    slot.channels = session->channels;
    slot.sample_rate = rate;
    slot.stream_epoch = session->epoch.load(std::memory_order_acquire);
    slot.ticket = direction == LFM_PCM_LEASE_CAPTURE
                      ? next_ticket(session, LFM_TICKET_TURN)
                      : LfmTicketIdV1{};
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .lease_id = identity,
        .stream_epoch = slot.stream_epoch,
        .buffer_generation = slot.generation.load(std::memory_order_acquire),
        .ticket = slot.ticket,
        .frames = frames,
        .channels = session->channels,
        .sample_rate = rate,
        .format = LFM_PCM_FORMAT_F32,
        .offset_bytes = 0,
        .length_bytes = static_cast<uint32_t>(samples * sizeof(float)),
        .flags = direction,
        .reserved = 0,
    };
    return 0;
}

int reserve_one(LfmSession *session, uint32_t direction, uint32_t frames,
                uint32_t sample_rate, LfmPcmLeaseV1 *out) {
    if (!out) return LFM_STATUS_INVALID_ARGUMENT;
    PcmPool *pool = nullptr;
    uint32_t rate = 0;
    size_t samples = 0;
    const int status = prepare_reservation(session, direction, frames,
                                           sample_rate, &pool, &rate,
                                           &samples);
    if (status != 0) return status;
    const uint32_t index =
        pool->cursor.value.fetch_add(1, std::memory_order_relaxed) %
        pool->capacity;
    return reserve_slot_at(session, pool, direction, frames, rate, samples,
                           index, out);
}

void set_error(char *error, size_t error_length, const char *message) {
    if (!error || error_length == 0) return;
    size_t bytes = std::strlen(message);
    if (bytes >= error_length) bytes = error_length - 1;
    if (bytes != 0) std::memcpy(error, message, bytes);
    error[bytes] = '\0';
}

int validate_voice_model(const LfmModel *model, char *error,
                         size_t error_length) {
    LfmModelInfoV1 info = {
        .size = sizeof(LfmModelInfoV1),
        .abi_version = LFM_MODEL_ABI_VERSION,
    };
    int rc = lfm_model_info(model, &info);
    if (rc != 0) {
        set_error(error, error_length,
                  "native voice model metadata validation failed");
        return rc;
    }
    constexpr uint32_t required =
        LFM_MODEL_CAP_DEPTHFORMER | LFM_MODEL_CAP_FRONTEND |
        LFM_MODEL_CAP_CONFORMER | LFM_MODEL_CAP_MIMI;
    if ((info.capabilities & required) != required || info.codebooks == 0) {
        set_error(error, error_length,
                  "checkpoint is not a complete native LFM2 voice model");
        return LFM_STATUS_INVALID_ARGUMENT;
    }

    LfmModelMemoryV1 memory = {
        .size = sizeof(LfmModelMemoryV1),
        .abi_version = LFM_MODEL_ABI_VERSION,
    };
    rc = lfm_model_memory(model, &memory);
    if (rc != 0) {
        set_error(error, error_length,
                  "native voice model accounting validation failed");
        return rc;
    }
    if (memory.compatibility_copied_bytes != 0) {
        set_error(error, error_length,
                  "native voice model contains compatibility-copied weights");
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    return 0;
}

void close_publications(LfmSession *session) {
    session->publication_gate.value.fetch_or(PUBLICATION_CLOSED,
                                             std::memory_order_acq_rel);
}

void notify_session(LfmSession *session) {
    if (!session || !session->coordinator_notifier) return;
    const int status =
        kc_service_notifier_notify(session->coordinator_notifier);
    if (status != 0 && status != -ECANCELED) {
        int32_t expected = 0;
        session->terminal_status.compare_exchange_strong(
            expected, status, std::memory_order_acq_rel,
            std::memory_order_acquire);
        close_publications(session);
        session->stop.store(true, std::memory_order_release);
    }
}

int notify_delivery(LfmSession *session) {
    if (!session || !session->delivery_notifier) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const int status =
        kc_service_notifier_notify(session->delivery_notifier);
    if (status != 0 && status != -ECANCELED) {
        int32_t expected = 0;
        session->terminal_status.compare_exchange_strong(
            expected, status, std::memory_order_acq_rel,
            std::memory_order_acquire);
        close_publications(session);
        session->stop.store(true, std::memory_order_release);
        notify_session(session);
    }
    return status;
}

void request_stop(LfmSession *session, int32_t status) {
    if (status != 0) {
        int32_t expected = 0;
        session->terminal_status.compare_exchange_strong(expected, status,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire);
    }
    close_publications(session);
    bool first = !session->stop.exchange(true, std::memory_order_acq_rel);
    if (first) {
        uint64_t epoch = session->epoch.load(std::memory_order_relaxed);
        if (epoch != std::numeric_limits<uint64_t>::max()) {
            (void)session->epoch.value.compare_exchange_strong(
                epoch, epoch + 1, std::memory_order_release,
                std::memory_order_relaxed);
        }
    }
    uint32_t state = session->state.load(std::memory_order_acquire);
    if (state == LFM_SESSION_RUNNING) {
        session->state.compare_exchange_strong(state, LFM_SESSION_STOPPING,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire);
    }
    notify_session(session);
}

bool enter_publication(LfmSession *session) {
    const uint64_t previous = session->publication_gate.value.fetch_add(
        1, std::memory_order_acq_rel);
    const uint64_t count = previous & PUBLICATION_COUNT_MASK;
    if (count == PUBLICATION_COUNT_MASK) std::abort();
    if ((previous & PUBLICATION_CLOSED) == 0) return true;
    const uint64_t released = session->publication_gate.value.fetch_sub(
        1, std::memory_order_acq_rel);
    if ((released & PUBLICATION_COUNT_MASK) == 0) std::abort();
    /* A rejected post-close entry can be the transient count observed by the
     * coordinator. Its compensating release is therefore also a successor. */
    notify_session(session);
    return false;
}

void leave_publication(LfmSession *session) {
    const uint64_t previous = session->publication_gate.value.fetch_sub(
        1, std::memory_order_acq_rel);
    const uint64_t count = previous & PUBLICATION_COUNT_MASK;
    if (count == 0) std::abort();
    if (count == 1 && (previous & PUBLICATION_CLOSED) != 0) {
        notify_session(session);
    }
}

EventRecord make_event(uint32_t kind, uint64_t epoch, LfmTicketIdV1 ticket,
                       int32_t status, const void *payload,
                       size_t payload_bytes, uint32_t flags = 0) {
    EventRecord record{};
    record.kind = kind;
    record.flags = flags;
    record.epoch = epoch;
    record.ticket = ticket;
    record.status = status;
    record.payload_bytes = static_cast<uint32_t>(payload_bytes);
    if (payload_bytes != 0) std::memcpy(record.payload, payload, payload_bytes);
    return record;
}

EventRecord make_turn(uint64_t epoch, LfmTicketIdV1 ticket,
                      uint32_t playback_count, uint32_t emitted,
                      uint32_t flags, int32_t status) {
    const LfmTurnEventV1 turn = {
        .size = sizeof(LfmTurnEventV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .playback_leases = playback_count,
        .emitted_items = emitted,
    };
    if (playback_count != 0) flags |= LFM_EVENT_FLAG_HAS_AUDIO;
    return make_event(LFM_EVENT_TURN, epoch, ticket, status, &turn,
                      sizeof(turn), flags);
}

bool stage_results(LfmSession *session, const EventRecord *records,
                   uint32_t count, bool gate_epoch = false,
                   int32_t stop_after = 0) {
    if (!session || !records || count == 0 || count > 2 ||
        session->result.active) {
        if (session) request_stop(session, LFM_STATUS_INTERNAL);
        return false;
    }
    session->result = {};
    for (uint32_t index = 0; index < count; ++index) {
        session->result.records[index] = records[index];
    }
    session->result.count = count;
    session->result.stop_after = stop_after;
    session->result.active = true;
    session->result.gate_epoch = gate_epoch;
    return true;
}

bool stage_event(LfmSession *session, uint32_t kind, uint64_t epoch,
                 LfmTicketIdV1 ticket, int32_t status, const void *payload,
                 size_t payload_bytes, uint32_t flags = 0,
                 bool gate_epoch = false, int32_t stop_after = 0) {
    if (payload_bytes > EVENT_PAYLOAD_CAPACITY) {
        request_stop(session, LFM_STATUS_INTERNAL);
        return false;
    }
    const EventRecord record =
        make_event(kind, epoch, ticket, status, payload, payload_bytes, flags);
    return stage_results(session, &record, 1, gate_epoch, stop_after);
}

bool stage_turn(LfmSession *session, uint64_t action_epoch,
                LfmTicketIdV1 ticket, uint32_t playback_count,
                uint32_t emitted, uint32_t flags, int32_t status = 0,
                int32_t stop_after = 0) {
    const EventRecord record = make_turn(action_epoch, ticket, playback_count,
                                         emitted, flags, status);
    const bool gate_epoch = status != LFM_STATUS_STALE &&
                            status != LFM_STATUS_CANCELLED;
    return stage_results(session, &record, 1, gate_epoch, stop_after);
}

bool stage_playback_ready(LfmSession *session,
                          const LfmPcmLeaseV1 &lease) {
    const LfmPlaybackReadyEventV1 ready = {
        .size = sizeof(LfmPlaybackReadyEventV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .lease_id = lease.lease_id,
        .buffer_generation = lease.buffer_generation,
    };
    return stage_event(session, LFM_EVENT_PLAYBACK_READY,
                       lease.stream_epoch, lease.ticket, 0, &ready,
                       sizeof(ready));
}

bool stage_action_failure(LfmSession *session, uint64_t action_epoch,
                          LfmTicketIdV1 ticket, int32_t status,
                          const char *message, uint32_t playback_count = 0,
                          uint32_t emitted = 0, bool stop_after = true) {
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        return stage_turn(session, action_epoch, ticket, playback_count,
                          emitted, 0, LFM_STATUS_STALE);
    }
    size_t bytes = std::strlen(message);
    if (bytes > EVENT_PAYLOAD_CAPACITY) bytes = EVENT_PAYLOAD_CAPACITY;
    const EventRecord records[2] = {
        make_event(LFM_EVENT_ERROR, action_epoch, ticket, status, message,
                   bytes),
        make_turn(action_epoch, ticket, playback_count, emitted, 0, status),
    };
    return stage_results(session, records, 2, true,
                         stop_after ? status : 0);
}

void stage_error(LfmSession *session, int32_t status, const char *message) {
    size_t bytes = std::strlen(message);
    if (bytes > EVENT_PAYLOAD_CAPACITY) bytes = EVENT_PAYLOAD_CAPACITY;
    (void)stage_event(session, LFM_EVENT_ERROR,
                      session->epoch.load(std::memory_order_acquire),
                      next_ticket(session, LFM_TICKET_CONTROL), status,
                      message, bytes, 0, false, status);
}

void release_prepared(LfmSession *session, PreparedPlayback *playback) {
    if (!playback || !playback->active) return;
    (void)lfm_audio_dock_release(session, &playback->lease);
    playback->active = false;
    playback->samples = 0;
}

int reserve_playback(LfmSession *session, uint64_t action_epoch,
                     LfmPcmLeaseV1 *out) {
    if (session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        return LFM_STATUS_STALE;
    }
    return lfm_audio_dock_reserve(session, LFM_PCM_LEASE_PLAYBACK,
                                  session->playback_frames,
                                  session->sample_rate, out);
}

void route_notify(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    notify_session(session);
}

void release_action_capture(LfmSession *session, SessionAction *action) {
    if (!action || !action->capture_active) return;
    (void)release_slot(&session->capture, &action->capture);
    action->capture = {};
    action->capture_active = false;
}

void clear_action(LfmSession *session) {
    if (session->action.admission_pending || session->action.route_pending) {
        std::abort();
    }
    release_action_capture(session, &session->action);
    release_prepared(session, &session->action.playback);
    session->action = {};
}

int submit_action_interrupt(LfmSession *session, int32_t status,
                            uint32_t flags) {
    SessionAction &action = session->action;
    action.route = {};
    const int rc = lfm_conversation_interrupt_submit_native(
        session->conversation, route_notify, session, &action.route);
    if (rc != 0) return rc;
    action.route_pending = true;
    action.route_audio = false;
    action.interrupt_status = status;
    action.interrupt_flags = flags;
    action.phase = ACTION_PHASE_INTERRUPT_PENDING;
    return 0;
}

enum ResultProgress : uint32_t {
    RESULT_EMPTY = 0,
    RESULT_DRAINED = 1,
    RESULT_BLOCKED = 2,
};

ResultProgress drain_result(LfmSession *session) {
    ResultRecord &result = session->result;
    if (!result.active) return RESULT_EMPTY;
    if (session->sink_failed.load(std::memory_order_acquire)) {
        result = {};
        request_stop(session, LFM_STATUS_HOST_SINK);
        return RESULT_DRAINED;
    }
    if (result.gate_epoch && result.next == 0 &&
        session->epoch.load(std::memory_order_acquire) !=
            result.records[0].epoch) {
        const int32_t terminal =
            session->stop.load(std::memory_order_acquire)
                ? LFM_STATUS_CANCELLED
                : LFM_STATUS_STALE;
        const EventRecord &tail = result.records[result.count - 1];
        if (tail.kind == LFM_EVENT_TURN) {
            EventRecord stale = tail;
            stale.status = terminal;
            stale.flags &= ~LFM_EVENT_FLAG_TRUNCATED;
            result = {};
            result.records[0] = stale;
            result.count = 1;
            result.active = true;
        } else if (session->action.active &&
                   ticket_equal(result.records[0].ticket,
                                session->action.ticket)) {
            const SessionAction &action = session->action;
            const EventRecord stale =
                make_turn(action.epoch, action.ticket, action.playback_count,
                          action.emitted, 0, terminal);
            result = {};
            result.records[0] = stale;
            result.count = 1;
            result.active = true;
            session->action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
        } else {
            result = {};
            return RESULT_DRAINED;
        }
    }
    while (result.next < result.count) {
        if (!event_push(&session->events, result.records[result.next])) {
            return RESULT_BLOCKED;
        }
        result.next++;
        (void)notify_delivery(session);
    }
    const int32_t stop_after = result.stop_after;
    result = {};
    if (stop_after != 0) request_stop(session, stop_after);
    return RESULT_DRAINED;
}

void fail_action(LfmSession *session, int status, const char *message) {
    SessionAction &action = session->action;
    release_action_capture(session, &action);
    release_prepared(session, &action.playback);
    if (session->stop.load(std::memory_order_acquire)) {
        (void)stage_turn(session, action.epoch, action.ticket,
                         action.playback_count, action.emitted, 0,
                         LFM_STATUS_CANCELLED);
        action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
        return;
    }
    if (session->epoch.load(std::memory_order_acquire) != action.epoch) {
        (void)stage_turn(session, action.epoch, action.ticket,
                         action.playback_count, action.emitted, 0,
                         LFM_STATUS_STALE);
        action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
        return;
    }
    (void)stage_action_failure(session, action.epoch, action.ticket, status,
                               message, action.playback_count, action.emitted);
    action.terminal_status = status;
    action.phase = ACTION_PHASE_FAILURE_PUBLISHED;
}

enum ActionProgress : uint32_t {
    ACTION_IDLE = 0,
    ACTION_PROGRESS = 1,
    ACTION_BLOCKED_RESULT = 2,
    ACTION_BLOCKED_ROUTE = 3,
    ACTION_BLOCKED_PLAYBACK = 4,
};

ActionProgress advance_action(LfmSession *session) {
    SessionAction &action = session->action;
    if (!action.active) return ACTION_IDLE;
    if (session->result.active) return ACTION_BLOCKED_RESULT;
    for (uint32_t transition = 0; transition < ACTION_TRANSITION_BUDGET;
         ++transition) {
        if (action.phase == ACTION_PHASE_ADMISSION_PENDING) {
            const int rc = lfm_conversation_begin_collect_native(
                session->conversation, &action.admission);
            if (rc == -EINPROGRESS) return ACTION_BLOCKED_ROUTE;
            action.admission_pending = false;
            release_action_capture(session, &action);
            if (rc != 0) {
                fail_action(session, rc, "native turn admission failed");
                return ACTION_PROGRESS;
            }
            action.phase = ACTION_PHASE_EMIT;
        }
        if (action.phase == ACTION_PHASE_TEXT_PUBLISHED ||
            action.phase == ACTION_PHASE_PLAYBACK_PUBLISHED) {
            action.emission = {};
            action.phase = ACTION_PHASE_NEED_ROUTE;
            continue;
        }
        if (action.phase == ACTION_PHASE_TERMINAL_PUBLISHED) {
            clear_action(session);
            return ACTION_PROGRESS;
        }
        if (action.phase == ACTION_PHASE_FAILURE_PUBLISHED) {
            const int status = action.terminal_status;
            clear_action(session);
            request_stop(session, status);
            return ACTION_PROGRESS;
        }
        if (action.phase == ACTION_PHASE_PLAYBACK_CAPACITY_PENDING) {
            action.phase = ACTION_PHASE_NEED_ROUTE;
        }
        if (action.phase == ACTION_PHASE_ROUTE_PENDING) {
            action.emission = {};
            int rc = action.route_audio
                ? lfm_conversation_next_into_collect_native(
                      session->conversation, &action.route, &action.emission,
                      &action.playback.samples)
                : lfm_conversation_next_collect_native(
                      session->conversation, &action.route, &action.emission);
            if (rc == -EINPROGRESS) return ACTION_BLOCKED_ROUTE;
            action.route_pending = false;
            if (rc != 0) {
                fail_action(session, rc, "native recurrence failed");
                return ACTION_PROGRESS;
            }
            action.phase = ACTION_PHASE_EMIT;
        }
        if (action.phase == ACTION_PHASE_INTERRUPT_PENDING) {
            const int rc = lfm_conversation_interrupt_collect_native(
                session->conversation, &action.route);
            if (rc == -EINPROGRESS) return ACTION_BLOCKED_ROUTE;
            action.route_pending = false;
            if (rc != 0) {
                fail_action(session, rc, "native action interrupt failed");
                return ACTION_PROGRESS;
            }
            (void)stage_turn(session, action.epoch, action.ticket,
                             action.playback_count, action.emitted,
                             action.interrupt_flags,
                             action.interrupt_status);
            action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
            return ACTION_PROGRESS;
        }
        if (session->stop.load(std::memory_order_acquire)) {
            const int rc = submit_action_interrupt(
                session, LFM_STATUS_CANCELLED, 0);
            if (rc != 0) {
                fail_action(session, rc,
                            "native cancellation interrupt failed");
                return ACTION_PROGRESS;
            }
            return ACTION_BLOCKED_ROUTE;
        }
        if (session->epoch.load(std::memory_order_acquire) != action.epoch) {
            const int rc = submit_action_interrupt(session, LFM_STATUS_STALE,
                                                   0);
            if (rc != 0) {
                fail_action(session, rc, "native epoch interrupt failed");
                return ACTION_PROGRESS;
            }
            return ACTION_BLOCKED_ROUTE;
        }
        if (action.phase == ACTION_PHASE_EMIT) {
            const LfmNativeEmission &emission = action.emission;
            if (emission.kind == LFM_NATIVE_EMISSION_TEXT ||
                (emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES &&
                 (emission.flags & EMISSION_AUDIO_END) == 0)) {
                action.emitted++;
            }
            if (emission.kind == LFM_NATIVE_EMISSION_NONE) {
                if (action.playback.active) {
                    fail_action(session, LFM_STATUS_INTERNAL,
                                "audio route returned no emission");
                    return ACTION_PROGRESS;
                }
                action.phase = ACTION_PHASE_NEED_ROUTE;
                continue;
            }
            if (emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES) {
                const int needs_pcm = lfm_native_emission_needs_pcm(&emission);
                if (needs_pcm < 0) {
                    fail_action(session, LFM_STATUS_INTERNAL,
                                "invalid native audio emission");
                    return ACTION_PROGRESS;
                }
                if (needs_pcm != 0) {
                    if (emission.code_count != LFM_MIMI_CODEBOOKS ||
                        !action.playback.active || action.playback.samples == 0 ||
                        action.playback.samples > UINT32_MAX) {
                        fail_action(session, LFM_STATUS_INTERNAL,
                                    "native Mimi route produced invalid PCM");
                        return ACTION_PROGRESS;
                    }
                    action.playback.lease.ticket = action.ticket;
                    action.playback.lease.frames =
                        static_cast<uint32_t>(action.playback.samples);
                    action.playback.lease.length_bytes =
                        static_cast<uint32_t>(action.playback.samples *
                                              sizeof(float));
                    action.playback.lease.flags = LFM_PCM_LEASE_PLAYBACK;
                    const int rc = lfm_audio_dock_publish(
                        session, &action.playback.lease);
                    if (rc != 0) {
                        fail_action(session, rc,
                                    "playback publication failed");
                        return ACTION_PROGRESS;
                    }
                    const LfmPcmLeaseV1 published = action.playback.lease;
                    action.playback.active = false;
                    action.playback.samples = 0;
                    action.playback_count++;
                    if (!stage_playback_ready(session, published)) {
                        action.terminal_status = LFM_STATUS_INTERNAL;
                        action.phase = ACTION_PHASE_FAILURE_PUBLISHED;
                        return ACTION_PROGRESS;
                    }
                    action.phase = ACTION_PHASE_PLAYBACK_PUBLISHED;
                    return ACTION_PROGRESS;
                } else {
                    release_prepared(session, &action.playback);
                }
                action.emission = {};
                action.phase = ACTION_PHASE_NEED_ROUTE;
                continue;
            }
            if (emission.kind == LFM_NATIVE_EMISSION_TEXT) {
                if (action.playback.active ||
                    emission.text_bytes > sizeof(emission.text)) {
                    fail_action(session, LFM_STATUS_INTERNAL,
                                action.playback.active
                                    ? "audio route returned text"
                                    : "native text emission exceeds bound");
                    return ACTION_PROGRESS;
                }
                (void)stage_event(session, LFM_EVENT_TEXT, action.epoch,
                                  action.ticket, 0, emission.text,
                                  emission.text_bytes, 0, true);
                action.phase = ACTION_PHASE_TEXT_PUBLISHED;
                return ACTION_PROGRESS;
            }
            if (emission.kind == LFM_NATIVE_EMISSION_FINISHED) {
                if (action.playback.active) {
                    fail_action(session, LFM_STATUS_INTERNAL,
                                "audio route finished with a live lease");
                    return ACTION_PROGRESS;
                }
                (void)stage_turn(session, action.epoch, action.ticket,
                                 action.playback_count, action.emitted, 0);
                action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
                return ACTION_PROGRESS;
            }
            fail_action(session, LFM_STATUS_INTERNAL,
                        "unknown native emission kind");
            return ACTION_PROGRESS;
        }
        if (action.phase != ACTION_PHASE_NEED_ROUTE) {
            fail_action(session, LFM_STATUS_INTERNAL,
                        "invalid native action phase");
            return ACTION_PROGRESS;
        }
        if (action.emitted >= session->max_new_tokens) {
            const int rc = submit_action_interrupt(
                session, 0, LFM_EVENT_FLAG_TRUNCATED);
            if (rc != 0) {
                fail_action(session, rc,
                            "native generation limit interrupt failed");
                return ACTION_PROGRESS;
            }
            return ACTION_BLOCKED_ROUTE;
        }
        int needs_playback =
            lfm_conversation_next_requires_playback_native(session->conversation);
        if (needs_playback < 0) {
            fail_action(session, needs_playback,
                        "native route requirement failed");
            return ACTION_PROGRESS;
        }
        int rc = 0;
        if (needs_playback != 0) {
            rc = reserve_playback(session, action.epoch,
                                  &action.playback.lease);
            if (rc == LFM_STATUS_WOULD_BLOCK) {
                action.phase = ACTION_PHASE_PLAYBACK_CAPACITY_PENDING;
                return ACTION_BLOCKED_PLAYBACK;
            }
            if (rc != 0) {
                fail_action(session, rc, "playback reservation failed");
                return ACTION_PROGRESS;
            }
            action.playback.active = true;
            float *pcm = nullptr;
            size_t capacity = 0;
            rc = lfm_audio_dock_resolve_mut(
                session, &action.playback.lease, &pcm, &capacity);
            if (rc == 0) {
                const LfmAudioRouteTarget target = {
                    .epoch = &session->epoch,
                    .expected_epoch = action.epoch,
                    .pcm = pcm,
                    .pcm_capacity = capacity,
                };
                rc = lfm_conversation_next_into_submit_native(
                    session->conversation, &target, route_notify, session,
                    &action.route);
                if (rc == 0) {
                    action.route_audio = true;
                    action.route_pending = true;
                    action.phase = ACTION_PHASE_ROUTE_PENDING;
                    return ACTION_BLOCKED_ROUTE;
                }
            }
        } else {
            rc = lfm_conversation_next_submit_native(
                session->conversation, route_notify, session, &action.route);
            if (rc == 0) {
                action.route_audio = false;
                action.route_pending = true;
                action.phase = ACTION_PHASE_ROUTE_PENDING;
                return ACTION_BLOCKED_ROUTE;
            }
        }
        if (rc != 0) {
            fail_action(session, rc, "native recurrence failed");
            return ACTION_PROGRESS;
        }
    }
    fail_action(session, LFM_STATUS_INTERNAL,
                "native action transition budget exhausted");
    return ACTION_PROGRESS;
}

SessionAction *prepare_action(LfmSession *session, uint64_t action_epoch,
                              LfmTicketIdV1 ticket,
                              const LfmPcmLeaseV1 *capture = nullptr) {
    if (session->action.active) {
        stage_action_failure(session, action_epoch, ticket, LFM_STATUS_BUSY,
                               "conversation already has a mutating route");
        return nullptr;
    }
    session->action = {};
    session->action.ticket = ticket;
    session->action.epoch = action_epoch;
    session->action.phase = ACTION_PHASE_ADMISSION_PENDING;
    session->action.active = true;
    session->action.admission_pending = true;
    if (capture) {
        session->action.capture = *capture;
        session->action.capture_active = true;
    }
    return &session->action;
}

void flush_published(PcmPool *pool) {
    for (uint32_t i = 0; i < pool->capacity; ++i) {
        PcmSlot &slot = pool->slots[i];
        uint32_t expected = SLOT_PUBLISHED;
        if (slot.state.compare_exchange_strong(expected, SLOT_RELEASING,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
            slot.reserved_frames = 0;
            slot.frames = 0;
            slot.offset_frames = 0;
            slot.channels = 0;
            slot.sample_rate = 0;
            slot.stream_epoch = 0;
            slot.ticket = {};
            slot.identity.store(0, std::memory_order_relaxed);
            uint64_t generation = slot.generation.load(std::memory_order_relaxed);
            if (generation == std::numeric_limits<uint64_t>::max()) {
                slot.state.store(SLOT_RETIRED, std::memory_order_release);
                continue;
            }
            slot.generation.store(generation + 1, std::memory_order_relaxed);
            slot.state.store(SLOT_FREE, std::memory_order_release);
        }
    }
}

int drive_conversation_interrupt(LfmSession *session) {
    if (session->dock_only) return 0;
    if (!session->interrupt_pending) {
        session->interrupt_route = {};
        const int rc = lfm_conversation_interrupt_submit_native(
            session->conversation, route_notify, session,
            &session->interrupt_route);
        if (rc != 0) return rc;
        session->interrupt_pending = true;
        return -EINPROGRESS;
    }
    const int rc = lfm_conversation_interrupt_collect_native(
        session->conversation, &session->interrupt_route);
    if (rc == -EINPROGRESS) return rc;
    session->interrupt_pending = false;
    session->interrupt_route = {};
    return rc;
}

bool apply_epoch(LfmSession *session, uint64_t epoch) {
    const int rc = drive_conversation_interrupt(session);
    if (rc == 0) return true;
    if (rc == -EINPROGRESS) return false;
    stage_error(session, rc, "native conversation interrupt failed");
    (void)epoch;
    return false;
}

bool synchronize_epoch(LfmSession *session) {
    const uint64_t current_epoch =
        session->epoch.load(std::memory_order_acquire);
    if (current_epoch == session->applied_epoch) return true;
    if (!apply_epoch(session, current_epoch)) return false;
    session->applied_epoch = current_epoch;
    static constexpr char interrupted[] = "interrupted";
    (void)stage_event(session, LFM_EVENT_STATE, current_epoch,
                      next_ticket(session, LFM_TICKET_CONTROL), 0,
                      interrupted, sizeof(interrupted) - 1);
    return false;
}

void process_capture(LfmSession *session, const LfmPcmLeaseV1 &lease) {
    PcmSlot *slot = nullptr;
    int rc = claim_published(&session->capture, &lease, &slot);
    if (rc != 0) return;
    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (lease.stream_epoch != current_epoch) {
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        release_slot(&session->capture, &lease);
        stage_turn(session, lease.stream_epoch, lease.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }
    session->capture_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        release_slot(&session->capture, &lease);
        stage_turn(session, current_epoch, lease.ticket, 0, 0, 0);
        return;
    }

    size_t samples = lease.length_bytes / sizeof(float);
    const size_t offset = lease.offset_bytes / sizeof(float);
    SessionAction *action = prepare_action(
        session, current_epoch, lease.ticket, &lease);
    if (!action) {
        (void)release_slot(&session->capture, &lease);
        return;
    }
    rc = lfm_conversation_begin_pcm_submit_native(
        session->conversation, slot->samples + offset, samples,
        lease.sample_rate,
        &action->emission, route_notify, session, &action->admission);
    if (rc != 0) {
        action->admission_pending = false;
        fail_action(session, rc, "native PCM admission failed");
        return;
    }
}

void process_text(LfmSession *session, const TextCommand &command) {
    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (command.epoch != current_epoch) {
        session->text_commands_stale.fetch_add(1, std::memory_order_relaxed);
        stage_turn(session, command.epoch, command.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }
    session->text_commands_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        stage_turn(session, current_epoch, command.ticket, 0, 0, 0);
        return;
    }
    SessionAction *action = prepare_action(
        session, current_epoch, command.ticket);
    if (!action) return;
    int rc = lfm_conversation_begin_text_submit_native(
        session->conversation, command.text, command.bytes,
        &action->emission, route_notify, session, &action->admission);
    if (rc != 0) {
        action->admission_pending = false;
        fail_action(session, rc, "native typed-input admission failed");
        return;
    }
}

void process_mixed(LfmSession *session, const TextCommand &command) {
    PcmSlot *slot = nullptr;
    int rc = claim_published(&session->capture, &command.capture, &slot);
    if (rc != 0) {
        stage_action_failure(session, command.epoch, command.ticket, rc,
                               "mixed capture lease claim failed");
        return;
    }

    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (command.epoch != current_epoch ||
        command.capture.stream_epoch != current_epoch) {
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        session->text_commands_stale.fetch_add(1, std::memory_order_relaxed);
        release_slot(&session->capture, &command.capture);
        stage_turn(session, command.epoch, command.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }

    session->capture_consumed.fetch_add(1, std::memory_order_relaxed);
    session->text_commands_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        release_slot(&session->capture, &command.capture);
        stage_turn(session, current_epoch, command.ticket, 0, 0, 0);
        return;
    }

    const size_t samples = command.capture.length_bytes / sizeof(float);
    const size_t offset = command.capture.offset_bytes / sizeof(float);
    SessionAction *action = prepare_action(
        session, current_epoch, command.ticket, &command.capture);
    if (!action) {
        (void)release_slot(&session->capture, &command.capture);
        return;
    }
    rc = lfm_conversation_begin_mixed_submit_native(
        session->conversation, command.text, command.bytes,
        slot->samples + offset, samples, command.capture.sample_rate,
        &action->emission,
        route_notify, session, &action->admission);
    if (rc != 0) {
        action->admission_pending = false;
        fail_action(session, rc, "native mixed text/PCM admission failed");
        return;
    }
}

void process_command(LfmSession *session, const TextCommand &command) {
    if (command.kind == COMMAND_TEXT) {
        process_text(session, command);
        return;
    }
    if (command.kind == COMMAND_MIXED) {
        process_mixed(session, command);
        return;
    }
    stage_action_failure(session, command.epoch, command.ticket,
                           LFM_STATUS_INTERNAL, "unknown native command kind");
}

enum SessionProgress : uint32_t {
    SESSION_READY = 0,
    SESSION_IDLE = 1,
    SESSION_BLOCKED_RESULT = 2,
    SESSION_BLOCKED_ROUTE = 3,
    SESSION_BLOCKED_PLAYBACK = 4,
    SESSION_DONE = 5,
};

SessionProgress session_step(LfmSession *session) {
    for (uint32_t quantum = 0; quantum < SESSION_STEP_BUDGET; ++quantum) {
        const ResultProgress result = drain_result(session);
        if (result == RESULT_BLOCKED) return SESSION_BLOCKED_RESULT;
        if (result == RESULT_DRAINED) continue;

        if (session->action.active) {
            const ActionProgress action = advance_action(session);
            if (session->result.active || action == ACTION_PROGRESS) continue;
            if (action == ACTION_BLOCKED_ROUTE) return SESSION_BLOCKED_ROUTE;
            if (action == ACTION_BLOCKED_PLAYBACK) {
                return SESSION_BLOCKED_PLAYBACK;
            }
            if (action == ACTION_BLOCKED_RESULT) {
                return SESSION_BLOCKED_RESULT;
            }
            return SESSION_IDLE;
        }

        if (session->stop.load(std::memory_order_acquire)) {
            session->coordinator_phase = COORDINATOR_STOPPING;
            /* Closing and draining are one Rube-Goldberg transition: once the
             * packed gate is CLOSED|0, no producer can add another record.
             * Until then this retained state goes dormant and the releasing
             * publisher provides its sole successor edge. Check before the
             * final queue scan so a just-published record cannot be skipped. */
            if (session->publication_gate.value.load(
                    std::memory_order_acquire) != PUBLICATION_CLOSED) {
                return SESSION_IDLE;
            }
            if (session->command_pending) {
                const TextCommand command = session->pending_command;
                session->pending_command = {};
                session->command_pending = false;
                if (command.kind == COMMAND_MIXED) {
                    PcmSlot *slot = nullptr;
                    if (claim_published(&session->capture, &command.capture,
                                        &slot) == 0) {
                        (void)slot;
                        (void)release_slot(&session->capture,
                                           &command.capture);
                    }
                }
                (void)stage_turn(session, command.epoch, command.ticket, 0, 0,
                                 0, LFM_STATUS_CANCELLED);
                continue;
            }
            if (session->capture_pending) {
                const LfmPcmLeaseV1 lease = session->pending_capture;
                session->pending_capture = {};
                session->capture_pending = false;
                PcmSlot *slot = nullptr;
                if (claim_published(&session->capture, &lease, &slot) == 0) {
                    (void)slot;
                    (void)release_slot(&session->capture, &lease);
                    (void)stage_turn(session, lease.stream_epoch, lease.ticket,
                                     0, 0, 0, LFM_STATUS_CANCELLED);
                }
                continue;
            }
            TextCommand command{};
            if (text_pop(&session->commands, &command)) {
                if (command.kind == COMMAND_MIXED) {
                    PcmSlot *slot = nullptr;
                    if (claim_published(&session->capture, &command.capture,
                                        &slot) == 0) {
                        (void)slot;
                        (void)release_slot(&session->capture,
                                           &command.capture);
                    }
                }
                (void)stage_turn(session, command.epoch, command.ticket, 0, 0,
                                 0, LFM_STATUS_CANCELLED);
                continue;
            }
            LfmPcmLeaseV1 lease{};
            if (pool_pop(&session->capture, &lease)) {
                PcmSlot *slot = nullptr;
                if (claim_published(&session->capture, &lease, &slot) == 0) {
                    (void)slot;
                    (void)release_slot(&session->capture, &lease);
                    (void)stage_turn(session, lease.stream_epoch, lease.ticket,
                                     0, 0, 0, LFM_STATUS_CANCELLED);
                }
                continue;
            }
            if (!session->dock_only) {
                const int teardown = drive_conversation_interrupt(session);
                if (teardown == -EINPROGRESS) {
                    return SESSION_BLOCKED_ROUTE;
                }
                if (teardown != 0) {
                    int32_t expected = 0;
                    session->terminal_status.compare_exchange_strong(
                        expected, teardown, std::memory_order_acq_rel,
                        std::memory_order_acquire);
                }
            }
            flush_published(&session->capture);
            flush_published(&session->playback);
            session->event_done.store(true, std::memory_order_release);
            session->coordinator_phase = COORDINATOR_DONE;
            (void)notify_delivery(session);
            {
                std::lock_guard<std::mutex> guard(session->lifecycle_mutex);
                session->coordinator_done = true;
            }
            session->lifecycle_cv.notify_all();
            kc_service_request_stop(session->coordinator);
            return SESSION_DONE;
        }

        if (session->coordinator_phase == COORDINATOR_STARTING) {
            session->applied_epoch =
                session->epoch.load(std::memory_order_acquire);
            static constexpr char running[] = "running";
            (void)stage_event(session, LFM_EVENT_STATE,
                              session->applied_epoch,
                              next_ticket(session, LFM_TICKET_SESSION), 0,
                              running, sizeof(running) - 1);
            session->coordinator_phase = COORDINATOR_RUNNING;
            continue;
        }

        if (!synchronize_epoch(session)) {
            if (session->result.active ||
                session->stop.load(std::memory_order_acquire)) {
                continue;
            }
            return SESSION_IDLE;
        }

        if (session->command_pending) {
            if (!synchronize_epoch(session)) continue;
            process_command(session, session->pending_command);
            session->pending_command = {};
            session->command_pending = false;
            continue;
        }
        if (text_pop(&session->commands, &session->pending_command)) {
            session->command_pending = true;
            continue;
        }

        if (session->capture_pending) {
            if (!synchronize_epoch(session)) continue;
            process_capture(session, session->pending_capture);
            session->pending_capture = {};
            session->capture_pending = false;
            continue;
        }
        if (pool_pop(&session->capture, &session->pending_capture)) {
            session->capture_pending = true;
            continue;
        }
        return SESSION_IDLE;
    }
    return SESSION_READY;
}

void coordinator_main(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    if (session_step(session) != SESSION_READY) return;
    const int status = kc_service_ready_again(session->coordinator);
    if (status != 0 && status != -ECANCELED) {
        request_stop(session, status);
    }
}

int invoke_callback(LfmSession *session, const EventRecord &record) {
    if (!session->callbacks.on_event) return 0;
    LfmEventV1 event = {
        .size = sizeof(LfmEventV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .kind = record.kind,
        .flags = record.flags,
        .session_id = session->id,
        .epoch = record.epoch,
        .ticket = record.ticket,
        .payload = record.payload_bytes == 0 ? nullptr : record.payload,
        .payload_bytes = record.payload_bytes,
        .status = record.status,
    };
    session->callbacks_entered.fetch_add(1, std::memory_order_relaxed);
    return session->callbacks.on_event(session->callbacks.context, &event);
}

enum DeliveryProgress : uint32_t {
    DELIVERY_READY = 0,
    DELIVERY_IDLE = 1,
    DELIVERY_BLOCKED_HOST = 2,
    DELIVERY_DONE = 3,
};

void finish_delivery(LfmSession *session) {
    {
        std::lock_guard<std::mutex> guard(session->lifecycle_mutex);
        session->delivery_done = true;
    }
    session->lifecycle_cv.notify_all();
    kc_service_request_stop(session->delivery);
}

DeliveryProgress delivery_step(LfmSession *session) {
    for (uint32_t quantum = 0; quantum < SESSION_STEP_BUDGET; ++quantum) {
        if (session->delivery_pending) {
            const bool stopped =
                session->delivery_record.kind == LFM_EVENT_STOPPED;
            if (session->sink_failed.load(std::memory_order_acquire) &&
                !stopped) {
                session->delivery_record = {};
                session->delivery_pending = false;
                continue;
            }
            const int status =
                invoke_callback(session, session->delivery_record);
            if (status == LFM_STATUS_WOULD_BLOCK) {
                return DELIVERY_BLOCKED_HOST;
            }
            session->delivery_record = {};
            session->delivery_pending = false;
            if (status != 0 && !stopped) {
                session->sink_failed.store(true, std::memory_order_release);
                request_stop(session, LFM_STATUS_HOST_SINK);
                continue;
            }
            if (stopped) {
                finish_delivery(session);
                return DELIVERY_DONE;
            }
            continue;
        }

        EventRecord record{};
        if (event_pop(&session->events, &record)) {
            /* The ring cell is free as soon as its bounded value is copied
             * into delivery_record. Resume the exact producer; host
             * backpressure retains this record outside the ring. */
            notify_session(session);
            session->delivery_record = record;
            session->delivery_pending = true;
            continue;
        }
        if (!session->event_done.load(std::memory_order_acquire)) {
            return DELIVERY_IDLE;
        }
        if (!session->stopped_staged) {
            session->delivery_record = {};
            session->delivery_record.kind = LFM_EVENT_STOPPED;
            session->delivery_record.epoch =
                session->epoch.load(std::memory_order_acquire);
            session->delivery_record.ticket =
                next_ticket(session, LFM_TICKET_SESSION);
            session->delivery_record.status =
                session->terminal_status.load(std::memory_order_acquire);
            static constexpr char payload[] = "stopped";
            session->delivery_record.payload_bytes = sizeof(payload) - 1;
            std::memcpy(session->delivery_record.payload, payload,
                        sizeof(payload) - 1);
            session->delivery_pending = true;
            session->stopped_staged = true;
            continue;
        }
        finish_delivery(session);
        return DELIVERY_DONE;
    }
    return DELIVERY_READY;
}

void delivery_main(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    if (delivery_step(session) != DELIVERY_READY) return;
    const int status = kc_service_ready_again(session->delivery);
    if (status != 0 && status != -ECANCELED) {
        request_stop(session, status);
    }
}

bool register_session_locked(LfmRuntime *runtime, LfmSession *session) {
    if (runtime->session_count >= runtime->session_capacity) return false;
    for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
        if (!runtime->sessions[i]) {
            runtime->sessions[i] = session;
            runtime->session_count++;
            return true;
        }
    }
    return false;
}

void unregister_session(LfmRuntime *runtime, LfmSession *session) {
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
        if (runtime->sessions[i] == session) {
            runtime->sessions[i] = nullptr;
            runtime->session_count--;
            return;
        }
    }
}

int submit_text(LfmSession *session, const char *utf8, size_t utf8_bytes,
                LfmTicketIdV1 *out_ticket) {
    if (!session || !utf8 || utf8_bytes == 0 ||
        utf8_bytes > LFM_TEXT_COMMAND_MAX_BYTES || !out_ticket) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (!enter_publication(session)) return LFM_STATUS_CANCELLED;
    const auto finish = [session](int status) {
        leave_publication(session);
        return status;
    };
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_CANCELLED);
    }

    TextCommand command{};
    command.ticket = next_ticket(session, LFM_TICKET_TURN);
    command.epoch = session->epoch.load(std::memory_order_acquire);
    command.bytes = static_cast<uint32_t>(utf8_bytes);
    std::memcpy(command.text, utf8, utf8_bytes);
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_CANCELLED);
    }
    const int rc = text_push(&session->commands, command)
                       ? 0
                       : LFM_STATUS_WOULD_BLOCK;
    if (rc != 0) return finish(rc);
    session->text_commands_accepted.fetch_add(1, std::memory_order_relaxed);
    *out_ticket = command.ticket;
    notify_session(session);
    return finish(0);
}

int submit_mixed(LfmSession *session, const char *utf8, size_t utf8_bytes,
                 const LfmPcmLeaseV1 *capture,
                 LfmTicketIdV1 *out_ticket) {
    if (!session || !utf8 || utf8_bytes == 0 ||
        utf8_bytes > LFM_TEXT_COMMAND_MAX_BYTES || !capture || !out_ticket ||
        capture->flags != LFM_PCM_LEASE_CAPTURE) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (!enter_publication(session)) return LFM_STATUS_CANCELLED;
    const auto finish = [session](int status) {
        leave_publication(session);
        return status;
    };
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_CANCELLED);
    }

    TextCommand command{};
    command.ticket = capture->ticket;
    command.epoch = capture->stream_epoch;
    command.bytes = static_cast<uint32_t>(utf8_bytes);
    command.kind = COMMAND_MIXED;
    command.capture = *capture;
    std::memcpy(command.text, utf8, utf8_bytes);
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_CANCELLED);
    }
    if (session->epoch.load(std::memory_order_acquire) != command.epoch) {
        return finish(LFM_STATUS_STALE);
    }
    const int rc = mixed_push(session, command);
    if (rc != 0) return finish(rc);
    session->text_commands_accepted.fetch_add(1,
                                               std::memory_order_relaxed);
    *out_ticket = command.ticket;
    notify_session(session);
    return finish(0);
}

} // namespace

extern "C" {

int lfm_native_emission_needs_pcm(const LfmNativeEmission *emission) {
    if (!emission || emission->kind != LFM_NATIVE_EMISSION_AUDIO_CODES ||
        emission->code_count != LFM_MIMI_CODEBOOKS ||
        (emission->flags & ~EMISSION_AUDIO_END) != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const bool end = (emission->flags & EMISSION_AUDIO_END) != 0;
    for (uint32_t index = 0; index < emission->code_count; ++index) {
        if ((end && emission->codes[index] != LFM_MIMI_CODE_VALUES) ||
            (!end && emission->codes[index] >= LFM_MIMI_CODE_VALUES)) {
            return LFM_STATUS_INVALID_ARGUMENT;
        }
    }
    return end ? 0 : 1;
}

int lfm_runtime_create(const LfmRuntimeConfigV1 *config, LfmRuntime **out) {
    if (!config || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (config->size != sizeof(*config) ||
        config->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    if (config->coordination_workers != 1 || config->kernel_lanes == 0 ||
        config->kernel_lanes > MAX_KERNEL_LANES || config->event_capacity < 2 ||
        config->event_capacity > MAX_EVENT_CAPACITY || config->session_capacity == 0 ||
        config->session_capacity > MAX_RUNTIME_SESSIONS || config->reserved0 != 0 ||
        config->reserved1 != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmRuntime *runtime = new (std::nothrow) LfmRuntime();
    if (!runtime) return LFM_STATUS_OUT_OF_MEMORY;
    runtime->epoch = next_runtime_epoch.fetch_add(1, std::memory_order_acq_rel);
    if (runtime->epoch == 0) {
        runtime->epoch = next_runtime_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    runtime->kernel_lanes = config->kernel_lanes;
    runtime->event_capacity = config->event_capacity;
    runtime->session_capacity = config->session_capacity;
    runtime->engine = lfm_engine_new(static_cast<int>(config->kernel_lanes));
    if (!runtime->engine) {
        delete runtime;
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    const kc_runtime_config coordination = {
        .size = sizeof(kc_runtime_config),
        .abi_version = KC_ABI_VERSION,
        .worker_count = config->coordination_workers,
        .reserved = 0,
    };
    if (kc_runtime_create(&coordination, &runtime->coordination) != 0) {
        lfm_engine_free(runtime->engine);
        runtime->engine = nullptr;
        delete runtime;
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    *out = runtime;
    return 0;
}

int lfm_runtime_start(LfmRuntime *runtime) {
    if (!runtime) return LFM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    uint32_t expected = LFM_RUNTIME_CREATED;
    if (!runtime->state.compare_exchange_strong(expected, LFM_RUNTIME_STARTED,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        return LFM_STATUS_BUSY;
    }
    const int status = kc_runtime_start(runtime->coordination);
    if (status != 0) {
        runtime->state.store(LFM_RUNTIME_CREATED, std::memory_order_release);
        return status;
    }
    return 0;
}

void lfm_runtime_request_stop(LfmRuntime *runtime) {
    if (!runtime) return;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (runtime->state.load(std::memory_order_acquire) <
        LFM_RUNTIME_STOPPING) {
        runtime->state.store(LFM_RUNTIME_STOPPING,
                             std::memory_order_release);
    }
    for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
        if (runtime->sessions[i]) lfm_session_request_stop(runtime->sessions[i]);
    }
    if (runtime->engine) lfm_engine_request_stop(runtime->engine);
}

int lfm_runtime_join(LfmRuntime *runtime) {
    if (!runtime) return LFM_STATUS_INVALID_ARGUMENT;
    if (runtime->state.load(std::memory_order_acquire) < LFM_RUNTIME_STOPPING) {
        return LFM_STATUS_BUSY;
    }
    {
        std::lock_guard<std::mutex> guard(runtime->children_mutex);
        if (runtime->session_count != 0 || runtime->model != nullptr) {
            return LFM_STATUS_BUSY;
        }
    }
    if (runtime->engine) {
        lfm_engine_free(runtime->engine);
        runtime->engine = nullptr;
    }
    if (runtime->coordination) {
        kc_runtime_request_stop(runtime->coordination);
        const int joined = kc_runtime_join(runtime->coordination);
        if (joined != 0) return joined;
        const int destroyed = kc_runtime_destroy(runtime->coordination);
        if (destroyed != 0) return destroyed;
        runtime->coordination = nullptr;
    }
    runtime->state.store(LFM_RUNTIME_JOINED, std::memory_order_release);
    return 0;
}

int lfm_runtime_snapshot(const LfmRuntime *runtime, LfmRuntimeSnapshotV1 *out) {
    if (!runtime || !out) return LFM_STATUS_INVALID_ARGUMENT;
    if (out->size != sizeof(*out) || out->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .runtime_epoch = runtime->epoch,
        .state = runtime->state.load(std::memory_order_acquire),
        .kernel_lanes = runtime->kernel_lanes,
        .live_models = runtime->model ? 1u : 0u,
        .live_sessions = runtime->session_count,
        .reserved = {},
    };
    return 0;
}

int lfm_runtime_destroy(LfmRuntime *runtime) {
    if (!runtime) return LFM_STATUS_INVALID_ARGUMENT;
    if (runtime->state.load(std::memory_order_acquire) != LFM_RUNTIME_JOINED) {
        return LFM_STATUS_BUSY;
    }
    {
        std::lock_guard<std::mutex> guard(runtime->children_mutex);
        if (runtime->session_count != 0 || runtime->model != nullptr) {
            return LFM_STATUS_BUSY;
        }
    }
    delete runtime;
    return 0;
}

int lfm_runtime_model_open(LfmRuntime *runtime, const char *path,
                           LfmModel **out, char *error, size_t error_length) {
    if (!runtime || !path || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING) {
        return LFM_STATUS_CANCELLED;
    }
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING) {
        return LFM_STATUS_CANCELLED;
    }
    if (runtime->model) return LFM_STATUS_BUSY;
    LfmModel *model = nullptr;
    int rc = lfm_model_open(runtime->engine, path, &model, error, error_length);
    if (rc != 0) return rc;
    rc = validate_voice_model(model, error, error_length);
    if (rc != 0) {
        int close = lfm_model_close(model);
        if (close != 0) {
            set_error(error, error_length,
                      "incomplete native voice model could not be released");
            return close;
        }
        return rc;
    }
    runtime->model = model;
    *out = model;
    return 0;
}

int lfm_runtime_model_memory(const LfmRuntime *runtime,
                             const LfmModel *model,
                             LfmModelMemoryV1 *out) {
    if (!runtime || !model || !out) return LFM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (runtime->model != model) return LFM_STATUS_INVALID_ARGUMENT;
    return lfm_model_memory(model, out);
}

int lfm_runtime_model_close(LfmRuntime *runtime, LfmModel *model) {
    if (!runtime || !model) return LFM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (runtime->model != model || runtime->session_count != 0) return LFM_STATUS_BUSY;
    int rc = lfm_model_close(model);
    if (rc == 0) runtime->model = nullptr;
    return rc;
}

int lfm_runtime_conversation_create(LfmRuntime *runtime, LfmModel *model,
                                    const LfmConversationOptionsV1 *options,
                                    LfmConversation **out, char *error,
                                    size_t error_length) {
    if (!runtime || !model || !options || !out) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (options->size != sizeof(*options) ||
        options->abi_version != LFM_RUNTIME_ABI_VERSION ||
        options->reserved0 != 0 ||
        (options->flags & ~LFM_CONVERSATION_SEED_SYSTEM) != 0 ||
        options->text.size != sizeof(options->text) ||
        options->text.abi_version != LFM_RUNTIME_ABI_VERSION ||
        options->audio.size != sizeof(options->audio) ||
        options->audio.abi_version != LFM_RUNTIME_ABI_VERSION ||
        (options->text.flags & ~LFM_SAMPLING_GREEDY) != 0 ||
        (options->audio.flags & ~LFM_SAMPLING_GREEDY) != 0 ||
        options->text.reserved != 0 || options->audio.reserved != 0) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    for (uint64_t reserved : options->reserved) {
        if (reserved != 0) return LFM_STATUS_INVALID_ARGUMENT;
    }
    const auto policy_valid = [](const LfmSamplingPolicyV1 &policy) {
        return (policy.flags & LFM_SAMPLING_GREEDY) != 0 ||
               (std::isfinite(policy.temperature) && policy.temperature > 0.0);
    };
    if (!policy_valid(options->text) || !policy_valid(options->audio)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const auto policy = [](const LfmSamplingPolicyV1 &source) {
        return LfmSamplerConfigV1{
            .size = sizeof(LfmSamplerConfigV1),
            .abi_version = LFM_SAMPLE_ABI_VERSION,
            .flags = (source.flags & LFM_SAMPLING_GREEDY) != 0
                         ? LFM_SAMPLE_FLAG_GREEDY
                         : 0u,
            .top_k = source.top_k,
            .temperature = source.temperature,
            .reserved = 0,
        };
    };
    const LfmConversationConfigV1 config = {
        .size = sizeof(LfmConversationConfigV1),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .flags = (options->flags & LFM_CONVERSATION_SEED_SYSTEM) != 0
                     ? LFM_CONVERSATION_SEED_SYSTEM
                     : 0u,
        .reserved0 = 0,
        .seed = options->seed,
        .text_sampler = policy(options->text),
        .audio_sampler = policy(options->audio),
        .reserved = {},
    };

    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING) {
        return LFM_STATUS_CANCELLED;
    }
    if (runtime->model != model) return LFM_STATUS_INVALID_ARGUMENT;
    return lfm_conversation_create(model, &config, out, error, error_length);
}

int lfm_runtime_conversation_close(LfmRuntime *runtime,
                                   LfmConversation *conversation) {
    if (!runtime || !conversation) return LFM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    if (!runtime->model ||
        lfm_conversation_belongs_to(conversation, runtime->model) != 1) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
        if (runtime->sessions[i] &&
            runtime->sessions[i]->conversation == conversation) {
            return LFM_STATUS_BUSY;
        }
    }
    return lfm_conversation_close(conversation);
}

int lfm_session_create(LfmRuntime *runtime, LfmModel *model,
                       LfmConversation *conversation,
                       const LfmSessionConfigV1 *config,
                       const LfmCallbacksV1 *callbacks, LfmSession **out) {
    if (!runtime || !config || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    if (config->size != sizeof(*config) ||
        config->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    const bool dock_only =
        (config->flags & LFM_SESSION_FLAG_DOCK_ONLY) != 0;
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING ||
        config->capture_slots == 0 || config->capture_slots > MAX_PCM_SLOTS ||
        config->playback_slots == 0 || config->playback_slots > MAX_PCM_SLOTS ||
        config->capture_frames_per_slot == 0 ||
        (dock_only && config->playback_frames_per_slot == 0) ||
        config->pcm_channels != 1 ||
        config->pcm_sample_rate < 8000 || config->pcm_sample_rate > 192000 ||
        config->command_capacity == 0 || config->command_capacity > 64 ||
        config->max_new_tokens == 0 || config->reserved0 != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (dock_only && (model || conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (!dock_only && (!model || !conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (callbacks && (callbacks->size != sizeof(*callbacks) ||
                      callbacks->abi_version != LFM_RUNTIME_ABI_VERSION)) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    size_t capture_samples = 0;
    if (!checked_samples(config->capture_frames_per_slot, config->pcm_channels,
                         &capture_samples) ||
        capture_samples > UINT32_MAX / sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> owner(runtime->children_mutex);
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING) {
        return LFM_STATUS_CANCELLED;
    }
    if (!dock_only) {
        if (runtime->model != model ||
            lfm_conversation_belongs_to(conversation, model) != 1) {
            return LFM_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
            if (runtime->sessions[i] &&
                runtime->sessions[i]->conversation == conversation) {
                return LFM_STATUS_BUSY;
            }
        }
    }
    if (runtime->session_count >= runtime->session_capacity) {
        return LFM_STATUS_BUSY;
    }
    size_t playback_frames = config->playback_frames_per_slot;
    size_t playback_capacity = config->playback_frames_per_slot;
    if (!dock_only) {
        int prepare = lfm_conversation_prepare_pcm_native(
            conversation, capture_samples, config->pcm_sample_rate,
            &playback_frames);
        if (prepare != 0) return prepare;
        if (playback_frames == 0 || playback_frames > UINT32_MAX ||
            (playback_capacity != 0 && playback_frames > playback_capacity)) {
            return LFM_STATUS_INVALID_ARGUMENT;
        }
        if (playback_capacity == 0) playback_capacity = playback_frames;
    }
    size_t playback_samples = 0;
    if (!checked_samples(static_cast<uint32_t>(playback_capacity),
                         config->pcm_channels,
                         &playback_samples) ||
        playback_samples > UINT32_MAX / sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }

    LfmSession *session = new (std::nothrow) LfmSession();
    if (!session) return LFM_STATUS_OUT_OF_MEMORY;
    session->runtime = runtime;
    session->model = model;
    session->conversation = conversation;
    session->dock_only = dock_only;
    session->id = config->session_id == 0
                      ? next_session_id.fetch_add(1, std::memory_order_relaxed)
                      : config->session_id;
    if (session->id == 0) session->id = next_session_id.fetch_add(1);
    session->sample_rate = config->pcm_sample_rate;
    session->playback_frames = static_cast<uint32_t>(playback_frames);
    session->channels = config->pcm_channels;
    session->max_new_tokens = config->max_new_tokens;
    if (callbacks) session->callbacks = *callbacks;
    session->events.capacity = runtime->event_capacity;
    session->events.records = new (std::nothrow) EventRecord[runtime->event_capacity];
    session->commands.capacity = config->command_capacity;
    session->commands.ring =
        new (std::nothrow) TextRecordCell[config->command_capacity];
    int rc = session->events.records && session->commands.ring
                 ? 0
                 : LFM_STATUS_OUT_OF_MEMORY;
    if (rc == 0) {
        for (uint32_t index = 0; index < config->command_capacity; ++index) {
            session->commands.ring[index].sequence.store(
                static_cast<uint64_t>(index) * 2,
                std::memory_order_relaxed);
        }
    }
    if (rc == 0) {
        rc = pool_create(&session->capture, config->capture_slots,
                         static_cast<uint32_t>(capture_samples), LFM_PCM_LEASE_CAPTURE);
    }
    if (rc == 0) {
        rc = pool_create(&session->playback, config->playback_slots,
                         static_cast<uint32_t>(playback_samples), LFM_PCM_LEASE_PLAYBACK);
    }
    const kc_service_config coordinator = {
        .size = sizeof(kc_service_config),
        .abi_version = KC_ABI_VERSION,
        .callback = coordinator_main,
        .context = session,
        .reserved = 0,
    };
    if (rc == 0 &&
        kc_service_create(runtime->coordination, &coordinator,
                          &session->coordinator) != 0) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        kc_service_notifier_create(session->coordinator,
                                   &session->coordinator_notifier) != 0) {
        rc = LFM_STATUS_INTERNAL;
    }
    const kc_service_config delivery = {
        .size = sizeof(kc_service_config),
        .abi_version = KC_ABI_VERSION,
        .callback = delivery_main,
        .context = session,
        .reserved = 0,
    };
    if (rc == 0 &&
        kc_service_create(runtime->coordination, &delivery,
                          &session->delivery) != 0) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        kc_service_notifier_create(session->delivery,
                                   &session->delivery_notifier) != 0) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 && !register_session_locked(runtime, session)) {
        rc = LFM_STATUS_BUSY;
    }
    if (rc != 0) {
        if (session->delivery_notifier) {
            (void)kc_service_notifier_destroy(session->delivery_notifier);
            session->delivery_notifier = nullptr;
        }
        if (session->delivery) {
            (void)kc_service_destroy(session->delivery);
            session->delivery = nullptr;
        }
        if (session->coordinator_notifier) {
            (void)kc_service_notifier_destroy(
                session->coordinator_notifier);
            session->coordinator_notifier = nullptr;
        }
        if (session->coordinator) {
            (void)kc_service_destroy(session->coordinator);
            session->coordinator = nullptr;
        }
        delete session;
        return rc;
    }
    *out = session;
    return 0;
}

int lfm_session_start(LfmSession *session) {
    if (!session) return LFM_STATUS_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> owner(session->runtime->children_mutex);
    std::unique_lock<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->runtime->state.load(std::memory_order_acquire) != LFM_RUNTIME_STARTED) {
        return LFM_STATUS_BUSY;
    }
    if (session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    uint32_t expected = LFM_SESSION_CREATED;
    if (!session->state.compare_exchange_strong(expected, LFM_SESSION_RUNNING,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        return LFM_STATUS_BUSY;
    }
    int rc = kc_service_start(session->delivery);
    if (rc != 0) {
        session->state.store(LFM_SESSION_CREATED, std::memory_order_release);
        return rc;
    }
    session->delivery_started = true;
    rc = kc_service_start(session->coordinator);
    if (rc != 0) {
        request_stop(session, rc);
        kc_service_request_stop(session->delivery);
        session->start_cleanup = true;
        owner.unlock();
        lifecycle.unlock();
        (void)kc_service_join(session->delivery);
        (void)kc_service_join(session->coordinator);
        lifecycle.lock();
        session->delivery_started = false;
        session->services_joined = true;
        session->start_cleanup = false;
        session->state.store(LFM_SESSION_SERVICES_JOINED,
                             std::memory_order_release);
        return rc;
    }
    session->coordinator_started = true;
    notify_session(session);
    return 0;
}

int lfm_session_submit_text(LfmSession *session, const char *utf8,
                            size_t utf8_bytes, LfmTicketIdV1 *out_ticket) {
    return submit_text(session, utf8, utf8_bytes, out_ticket);
}

int lfm_session_submit_mixed(LfmSession *session, const char *utf8,
                             size_t utf8_bytes,
                             const LfmPcmLeaseV1 *capture,
                             LfmTicketIdV1 *out_ticket) {
    return submit_mixed(session, utf8, utf8_bytes, capture, out_ticket);
}

int lfm_session_interrupt(LfmSession *session, uint64_t *out_epoch) {
    if (!session || !out_epoch) return LFM_STATUS_INVALID_ARGUMENT;
    if (session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    uint64_t current = session->epoch.load(std::memory_order_relaxed);
    if (current == std::numeric_limits<uint64_t>::max()) return -EOVERFLOW;
    if (session->epoch.value.compare_exchange_strong(
            current, current + 1, std::memory_order_release,
            std::memory_order_relaxed)) {
        *out_epoch = current + 1;
    } else {
        /* Interrupt edges coalesce. A concurrent publisher already advanced
         * the epoch, which is the entire state transition this edge needed. */
        *out_epoch = current;
    }
    notify_session(session);
    return 0;
}

int lfm_session_host_capacity(LfmSession *session) {
    if (!session) return LFM_STATUS_INVALID_ARGUMENT;
    const int status = notify_delivery(session);
    return status == -ECANCELED ? LFM_STATUS_CANCELLED : status;
}

void lfm_session_request_stop(LfmSession *session) {
    if (!session) return;
    std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
    request_stop(session, 0);
}

int lfm_session_join(LfmSession *session) {
    if (!session) return LFM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> join(session->join_mutex);
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        const uint32_t state = session->state.load(std::memory_order_acquire);
        /* Callback endpoints are lifetime leases over the session and its
         * notifier pointers. Teardown must reject before retiring either
         * retained service; checking live PCM cells afterward is too late for a
         * device callback concurrently publishing the release edge. */
        if (session->capture_producers != 0 ||
            session->playback_consumers != 0 ||
            session->control_handles != 0) {
            return LFM_STATUS_BUSY;
        }
        if (state == LFM_SESSION_JOINED) {
            return session->terminal_status.load(std::memory_order_acquire);
        }
        if (session->start_cleanup) return LFM_STATUS_BUSY;
        if (state == LFM_SESSION_CREATED) {
            // A never-started session still owns admission docks. Closing them
            // under the same transition lock as start makes that choice final.
            request_stop(session, 0);
        }
        if (!session->stop.load(std::memory_order_acquire) &&
            state != LFM_SESSION_CREATED) {
            return LFM_STATUS_BUSY;
        }
    }

    /* Worker failure paths can call request_stop. Do not hold lifecycle_mutex
     * while waiting for their terminal administrative edge; stop/state already
     * make a later start impossible. This latch does not drive execution. */
    if (!session->services_joined) {
        if (session->coordinator_started) {
            std::unique_lock<std::mutex> lifecycle(session->lifecycle_mutex);
            session->lifecycle_cv.wait(
                lifecycle, [&] {
                    return session->coordinator_done &&
                           session->delivery_done;
                });
            lifecycle.unlock();
            int status = kc_service_join(session->coordinator);
            if (status != 0) return status;
            status = kc_service_join(session->delivery);
            if (status != 0) return status;
            session->coordinator_started = false;
            session->delivery_started = false;
        } else {
            int status = kc_service_join(session->coordinator);
            if (status != 0) return status;
            status = kc_service_join(session->delivery);
            if (status != 0) return status;
        }
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        session->services_joined = true;
        session->state.store(LFM_SESSION_SERVICES_JOINED,
                             std::memory_order_release);
    }
    if (session->coordinator_notifier) {
        const int status =
            kc_service_notifier_destroy(session->coordinator_notifier);
        if (status != 0) return status;
        session->coordinator_notifier = nullptr;
    }
    if (session->delivery_notifier) {
        const int status =
            kc_service_notifier_destroy(session->delivery_notifier);
        if (status != 0) return status;
        session->delivery_notifier = nullptr;
    }
    if (session->coordinator) {
        const int status = kc_service_destroy(session->coordinator);
        if (status != 0) return status;
        session->coordinator = nullptr;
    }
    if (session->delivery) {
        const int status = kc_service_destroy(session->delivery);
        if (status != 0) return status;
        session->delivery = nullptr;
    }
    if (pool_live(session->capture) != 0 || pool_live(session->playback) != 0) {
        return LFM_STATUS_BUSY;
    }
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        session->state.store(LFM_SESSION_JOINED, std::memory_order_release);
    }
    return session->terminal_status.load(std::memory_order_acquire);
}

int lfm_session_snapshot(const LfmSession *session, LfmSessionSnapshotV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    if (out->size != sizeof(*out) || out->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .session_id = session->id,
        .epoch = session->epoch.load(std::memory_order_acquire),
        .state = session->state.load(std::memory_order_acquire),
        .terminal_status = session->terminal_status.load(std::memory_order_acquire),
        .reserved_coordinator = {},
        .reserved_delivery = 0,
        .callbacks_entered = session->callbacks_entered.load(std::memory_order_relaxed),
        .capture_consumed = session->capture_consumed.load(std::memory_order_relaxed),
        .capture_stale = session->capture_stale.load(std::memory_order_relaxed),
        .playback_published = session->playback_published.load(std::memory_order_relaxed),
        .playback_consumed = session->playback_consumed.load(std::memory_order_relaxed),
        .text_commands_accepted =
            session->text_commands_accepted.load(std::memory_order_relaxed),
        .text_commands_consumed =
            session->text_commands_consumed.load(std::memory_order_relaxed),
        .text_commands_stale =
            session->text_commands_stale.load(std::memory_order_relaxed),
        .live_capture_leases = pool_live(session->capture),
        .live_playback_leases = pool_live(session->playback),
        .reliable_event_depth = event_depth(session->events),
        .reliable_event_capacity = session->events.capacity,
        .reserved = {},
    };
    return 0;
}

int lfm_session_destroy(LfmSession *session) {
    if (!session) return LFM_STATUS_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_JOINED ||
        pool_live(session->capture) != 0 || pool_live(session->playback) != 0 ||
        session->capture_producers != 0 || session->playback_consumers != 0 ||
        session->control_handles != 0) {
        return LFM_STATUS_BUSY;
    }
    lifecycle.unlock();
    unregister_session(session->runtime, session);
    delete session;
    return 0;
}

int lfm_capture_producer_create(LfmSession *session,
                                LfmCaptureProducer **out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    LfmCaptureProducer *producer =
        new (std::nothrow) LfmCaptureProducer();
    if (!producer) return LFM_STATUS_OUT_OF_MEMORY;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        const uint32_t state = session->state.load(std::memory_order_acquire);
        if ((state != LFM_SESSION_CREATED && state != LFM_SESSION_RUNNING) ||
            session->stop.load(std::memory_order_acquire)) {
            delete producer;
            return LFM_STATUS_CANCELLED;
        }
        /* The current dock is one mono/interleaved device source and its pool
         * has one producer cursor. Make SPSC ownership structural. Additional
         * independent lanes require distinct pools and explicit lane metadata,
         * not cloneable handles over this cursor. */
        if (session->capture_producers != 0) {
            delete producer;
            return LFM_STATUS_BUSY;
        }
        session->capture_producers = 1;
    }
    producer->session = session;
    *out = producer;
    return 0;
}

int lfm_capture_producer_reserve(LfmCaptureProducer *producer,
                                 uint32_t frames, uint32_t sample_rate,
                                 LfmPcmLeaseV1 *out) {
    if (!producer || !producer->session || !out) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPcmLeaseV1 lease{};
    const int status = reserve_one(producer->session, LFM_PCM_LEASE_CAPTURE,
                                   frames, sample_rate, &lease);
    if (status != 0) return status;
    producer->active_leases.fetch_add(1, std::memory_order_relaxed);
    *out = lease;
    return 0;
}

int lfm_capture_producer_resolve_mut(LfmCaptureProducer *producer,
                                     const LfmPcmLeaseV1 *lease,
                                     float **out_samples,
                                     size_t *out_sample_capacity) {
    if (!producer_matches(producer, lease)) return LFM_STATUS_STALE;
    return lfm_audio_dock_resolve_mut(producer->session, lease, out_samples,
                                      out_sample_capacity);
}

int lfm_capture_producer_finalize(LfmCaptureProducer *producer,
                                  LfmPcmLeaseV1 *lease,
                                  uint32_t offset_frames,
                                  uint32_t used_frames) {
    if (!producer_matches(producer, lease)) return LFM_STATUS_STALE;
    return lfm_audio_dock_finalize_capture(producer->session, lease,
                                           offset_frames, used_frames);
}

int lfm_capture_producer_publish(LfmCaptureProducer *producer,
                                 const LfmPcmLeaseV1 *lease) {
    if (!producer_matches(producer, lease)) return LFM_STATUS_STALE;
    const int status = lfm_audio_dock_publish(producer->session, lease);
    if (status == 0 &&
        producer->active_leases.fetch_sub(1, std::memory_order_acq_rel) == 0) {
        std::abort();
    }
    return status;
}

int lfm_capture_producer_release(LfmCaptureProducer *producer,
                                 const LfmPcmLeaseV1 *lease) {
    if (!producer_matches(producer, lease)) return LFM_STATUS_STALE;
    const int status = lfm_audio_dock_release(producer->session, lease);
    if (status == 0 &&
        producer->active_leases.fetch_sub(1, std::memory_order_acq_rel) == 0) {
        std::abort();
    }
    return status;
}

int lfm_capture_producer_destroy(LfmCaptureProducer *producer) {
    if (!producer || !producer->session) return LFM_STATUS_INVALID_ARGUMENT;
    if (producer->active_leases.load(std::memory_order_acquire) != 0) {
        return LFM_STATUS_BUSY;
    }
    LfmSession *session = producer->session;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        if (session->capture_producers == 0) std::abort();
        session->capture_producers--;
    }
    producer->session = nullptr;
    delete producer;
    return 0;
}

int lfm_playback_consumer_create(LfmSession *session,
                                 LfmPlaybackConsumer **out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    LfmPlaybackConsumer *consumer =
        new (std::nothrow) LfmPlaybackConsumer();
    if (!consumer) return LFM_STATUS_OUT_OF_MEMORY;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        const uint32_t state = session->state.load(std::memory_order_acquire);
        if ((state != LFM_SESSION_CREATED && state != LFM_SESSION_RUNNING) ||
            session->stop.load(std::memory_order_acquire)) {
            delete consumer;
            return LFM_STATUS_CANCELLED;
        }
        /* PcmPool::head is a single-consumer cursor. Make that ownership a
         * lifecycle invariant instead of trusting callers not to clone the
         * hardware endpoint. */
        if (session->playback_consumers != 0) {
            delete consumer;
            return LFM_STATUS_BUSY;
        }
        session->playback_consumers = 1;
    }
    consumer->session = session;
    *out = consumer;
    return 0;
}

int lfm_playback_consumer_claim(LfmPlaybackConsumer *consumer,
                                const LfmTicketIdV1 *ticket,
                                uint64_t stream_epoch, uint64_t lease_id,
                                uint64_t buffer_generation,
                                LfmPcmLeaseV1 *out) {
    if (!consumer || !consumer->session || !ticket || !out ||
        stream_epoch == 0 || lease_id == 0 || buffer_generation == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (consumer->active) return LFM_STATUS_WOULD_BLOCK;

    LfmPcmLeaseV1 lease{};
    uint64_t head = 0;
    if (!pool_peek(&consumer->session->playback, &lease, &head)) {
        return consumer->session->stop.load(std::memory_order_acquire)
                   ? LFM_STATUS_CANCELLED
                   : LFM_STATUS_STALE;
    }
    const bool exact = lease.lease_id == lease_id &&
                       lease.buffer_generation == buffer_generation &&
                       lease.stream_epoch == stream_epoch &&
                       ticket_equal(lease.ticket, *ticket);
    /* A duplicate, corrupt, or out-of-order reliable record must not consume
     * the true FIFO head. Only the exact ticket is authorized to move it. */
    if (!exact) return LFM_STATUS_STALE;

    PcmSlot *slot = nullptr;
    const int claim = claim_published(&consumer->session->playback, &lease,
                                      &slot);
    if (claim != 0) return claim;
    (void)slot;
    pool_retire_peeked(&consumer->session->playback, head);
    *out = lease;
    if (lease.stream_epoch !=
        consumer->session->epoch.load(std::memory_order_acquire)) {
        const int release = lfm_audio_dock_release(consumer->session, &lease);
        if (release != 0) return release;
        return LFM_STATUS_STALE;
    }
    consumer->session->playback_consumed.fetch_add(
        1, std::memory_order_relaxed);
    consumer->lease = lease;
    consumer->active = true;
    return 0;
}

int lfm_playback_consumer_resolve(const LfmPlaybackConsumer *consumer,
                                  const LfmPcmLeaseV1 *lease,
                                  const float **out_samples,
                                  size_t *out_sample_count) {
    if (!consumer_matches(consumer, lease)) return LFM_STATUS_STALE;
    return lfm_audio_dock_resolve(consumer->session, lease, out_samples,
                                  out_sample_count);
}

int lfm_playback_consumer_release(LfmPlaybackConsumer *consumer,
                                  const LfmPcmLeaseV1 *lease) {
    if (!consumer_matches(consumer, lease)) return LFM_STATUS_STALE;
    const int status = lfm_audio_dock_release(consumer->session, lease);
    if (status == 0 || status == LFM_STATUS_STALE ||
        status == LFM_STATUS_CANCELLED) {
        consumer->lease = {};
        consumer->active = false;
    }
    return status;
}

int lfm_playback_consumer_destroy(LfmPlaybackConsumer *consumer) {
    if (!consumer || !consumer->session) return LFM_STATUS_INVALID_ARGUMENT;
    if (consumer->active) return LFM_STATUS_BUSY;
    LfmSession *session = consumer->session;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        if (session->playback_consumers == 0) std::abort();
        session->playback_consumers--;
    }
    consumer->session = nullptr;
    delete consumer;
    return 0;
}

int lfm_session_control_create(LfmSession *session,
                               LfmSessionControl **out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    LfmSessionControl *control = new (std::nothrow) LfmSessionControl();
    if (!control) return LFM_STATUS_OUT_OF_MEMORY;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        const uint32_t state = session->state.load(std::memory_order_acquire);
        if ((state != LFM_SESSION_CREATED && state != LFM_SESSION_RUNNING) ||
            session->stop.load(std::memory_order_acquire)) {
            delete control;
            return LFM_STATUS_CANCELLED;
        }
        if (session->control_handles == UINT32_MAX) {
            delete control;
            return LFM_STATUS_OUT_OF_MEMORY;
        }
        session->control_handles++;
    }
    control->session = session;
    *out = control;
    return 0;
}

int lfm_session_control_interrupt(LfmSessionControl *control,
                                  uint64_t *out_epoch) {
    if (!control || !control->session) return LFM_STATUS_INVALID_ARGUMENT;
    return lfm_session_interrupt(control->session, out_epoch);
}

int lfm_session_control_destroy(LfmSessionControl *control) {
    if (!control || !control->session) return LFM_STATUS_INVALID_ARGUMENT;
    LfmSession *session = control->session;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        if (session->control_handles == 0) std::abort();
        session->control_handles--;
    }
    control->session = nullptr;
    delete control;
    return 0;
}

int lfm_audio_dock_reserve(LfmSession *session, uint32_t direction,
                           uint32_t frames, uint32_t sample_rate,
                           LfmPcmLeaseV1 *out) {
    if (!out) return LFM_STATUS_INVALID_ARGUMENT;
    PcmPool *pool = nullptr;
    uint32_t rate = 0;
    size_t samples = 0;
    const int prepared = prepare_reservation(session, direction, frames,
                                             sample_rate, &pool, &rate,
                                             &samples);
    if (prepared != 0) return prepared;
    const uint32_t start =
        pool->cursor.value.fetch_add(1, std::memory_order_relaxed) % pool->capacity;
    for (uint32_t offset = 0; offset < pool->capacity; ++offset) {
        const uint32_t index = (start + offset) % pool->capacity;
        const int status = reserve_slot_at(session, pool, direction, frames,
                                           rate, samples, index, out);
        if (status == 0 || status != LFM_STATUS_WOULD_BLOCK) return status;
    }
    return LFM_STATUS_WOULD_BLOCK;
}

int lfm_audio_dock_resolve_mut(LfmSession *session,
                               const LfmPcmLeaseV1 *lease,
                               float **out_samples, size_t *out_sample_capacity) {
    if (!session || !lease || !out_samples || !out_sample_capacity) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t direction = 0;
    uint32_t index = 0;
    if (!decode_lease_id(lease->lease_id, &direction, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmPool *pool = select_pool(session, direction);
    PcmSlot *slot = nullptr;
    int rc = pool ? pool_slot(pool, lease, &slot, nullptr) : LFM_STATUS_INVALID_ARGUMENT;
    if (rc != 0) return rc;
    if (slot->state.load(std::memory_order_acquire) != SLOT_RESERVED) {
        return LFM_STATUS_STALE;
    }
    *out_samples = slot->samples + lease->offset_bytes / sizeof(float);
    *out_sample_capacity = lease->length_bytes / sizeof(float);
    return 0;
}

int lfm_audio_dock_finalize_capture(LfmSession *session,
                                    LfmPcmLeaseV1 *lease,
                                    uint32_t offset_frames,
                                    uint32_t used_frames) {
    if (!session || !lease || used_frames == 0 ||
        (lease->flags & LFM_PCM_LEASE_DIRECTION_MASK) !=
            LFM_PCM_LEASE_CAPTURE ||
        offset_frames > lease->frames ||
        used_frames > lease->frames - offset_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmSlot *slot = nullptr;
    const int rc = pool_slot(&session->capture, lease, &slot, nullptr);
    if (rc != 0) return rc;
    if (slot->state.load(std::memory_order_acquire) != SLOT_RESERVED) {
        return LFM_STATUS_STALE;
    }
    size_t samples = 0;
    size_t offset = 0;
    if ((offset_frames != 0 &&
         !checked_samples(offset_frames, lease->channels, &offset)) ||
        !checked_samples(used_frames, lease->channels, &samples) ||
        offset > session->capture.samples_per_slot ||
        samples > session->capture.samples_per_slot ||
        samples > session->capture.samples_per_slot - offset ||
        samples > UINT32_MAX / sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (offset > UINT32_MAX / sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    slot->offset_frames = offset_frames;
    slot->frames = used_frames;
    lease->frames = used_frames;
    lease->offset_bytes = static_cast<uint32_t>(offset * sizeof(float));
    lease->length_bytes = static_cast<uint32_t>(samples * sizeof(float));
    return 0;
}

int lfm_audio_dock_resolve(const LfmSession *session,
                           const LfmPcmLeaseV1 *lease,
                           const float **out_samples,
                           size_t *out_sample_count) {
    if (!session || !lease || !out_samples || !out_sample_count) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t direction = 0;
    uint32_t index = 0;
    if (!decode_lease_id(lease->lease_id, &direction, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (direction == LFM_PCM_LEASE_PLAYBACK &&
        lease->stream_epoch != session->epoch.load(std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    const PcmPool *selected = select_pool(session, direction);
    if (!selected) return LFM_STATUS_INVALID_ARGUMENT;
    PcmPool *pool = const_cast<PcmPool *>(selected);
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    if (slot->state.load(std::memory_order_acquire) != SLOT_CONSUMING) {
        return LFM_STATUS_STALE;
    }
    *out_samples = slot->samples + lease->offset_bytes / sizeof(float);
    *out_sample_count = lease->length_bytes / sizeof(float);
    return 0;
}

int lfm_audio_dock_publish(LfmSession *session, const LfmPcmLeaseV1 *lease) {
    if (!session || !lease) return LFM_STATUS_INVALID_ARGUMENT;
    if (!enter_publication(session)) return LFM_STATUS_CANCELLED;
    const auto finish = [session](int status) {
        leave_publication(session);
        return status;
    };
    uint32_t direction = 0;
    uint32_t index = 0;
    if (!decode_lease_id(lease->lease_id, &direction, &index)) {
        return finish(LFM_STATUS_INVALID_ARGUMENT);
    }
    PcmPool *pool = select_pool(session, direction);
    PcmSlot *slot = nullptr;
    int rc = pool ? pool_slot(pool, lease, &slot, nullptr) : LFM_STATUS_INVALID_ARGUMENT;
    if (rc != 0) return finish(rc);
    if (direction == LFM_PCM_LEASE_PLAYBACK) {
        if (session->stop.load(std::memory_order_acquire)) {
            return finish(LFM_STATUS_CANCELLED);
        }
        if (lease->stream_epoch != session->epoch.load(std::memory_order_acquire)) {
            return finish(LFM_STATUS_STALE);
        }
    }
    if (direction == LFM_PCM_LEASE_CAPTURE &&
        lease->sample_rate != session->sample_rate) {
        return finish(LFM_STATUS_INVALID_ARGUMENT);
    }
    uint32_t expected = SLOT_RESERVED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_PUBLISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return finish(LFM_STATUS_STALE);
    }
    if (direction == LFM_PCM_LEASE_PLAYBACK) {
        slot->ticket = lease->ticket;
    }
    pool_push(pool, *lease);
    if (direction == LFM_PCM_LEASE_CAPTURE) {
        notify_session(session);
        return finish(0);
    }
    session->playback_published.fetch_add(1, std::memory_order_relaxed);
    return finish(0);
}

int lfm_audio_dock_try_playback(LfmSession *session, LfmPcmLeaseV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    LfmPcmLeaseV1 lease{};
    if (pool_pop(&session->playback, &lease)) {
        PcmSlot *slot = nullptr;
        const int claim =
            claim_published(&session->playback, &lease, &slot);
        if (claim != 0) {
            *out = lease;
            return claim;
        }
        (void)slot;
        if (lease.stream_epoch != session->epoch.load(std::memory_order_acquire)) {
            const int release = lfm_audio_dock_release(session, &lease);
            if (release != 0) return release;
            *out = lease;
            return LFM_STATUS_STALE;
        }
        session->playback_consumed.fetch_add(1, std::memory_order_relaxed);
        *out = lease;
        return 0;
    }
    return session->stop.load(std::memory_order_acquire)
               ? LFM_STATUS_CANCELLED
               : LFM_STATUS_WOULD_BLOCK;
}

int lfm_audio_dock_release(LfmSession *session, const LfmPcmLeaseV1 *lease) {
    if (!session || !lease) return LFM_STATUS_INVALID_ARGUMENT;
    uint32_t direction = 0;
    uint32_t index = 0;
    if (!decode_lease_id(lease->lease_id, &direction, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmPool *pool = select_pool(session, direction);
    uint32_t allowed = direction == LFM_PCM_LEASE_CAPTURE
                           ? (UINT32_C(1) << SLOT_RESERVED)
                           : ((UINT32_C(1) << SLOT_RESERVED) |
                              (UINT32_C(1) << SLOT_CONSUMING));
    if (!pool) return LFM_STATUS_INVALID_ARGUMENT;
    const int status = release_slot(pool, lease, allowed);
    if (status == 0 && direction == LFM_PCM_LEASE_PLAYBACK) {
        notify_session(session);
    }
    return status;
}

} /* extern "C" */
