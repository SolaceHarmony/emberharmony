#include "lfm_audio_dock.h"
#include "lfm_runtime.h"
#include "lfm_session.h"

#include "kc_atomic.h"
#include "kc_port.h"
#include "lfm_mimi.h"
#include "lfm_model_internal.h"
#include "../model/lfm_route_epoch.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

extern "C" {
void *lfm_engine_new(int workers);
void lfm_engine_request_stop(void *engine);
void lfm_engine_free(void *engine);
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

struct alignas(HOT_ATOMIC_BYTES) Doorbell {
    uint32_t value = 0;
    kc_port_wait_word *wait = nullptr;
};
static_assert(alignof(Doorbell) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Doorbell) == HOT_ATOMIC_BYTES,
              "adjacent doorbells must not share an Apple cache line");

template <typename T>
struct alignas(HOT_ATOMIC_BYTES) Cursor {
    std::atomic<T> value{0};
};
static_assert(alignof(Cursor<uint32_t>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Cursor<uint32_t>) == HOT_ATOMIC_BYTES);
static_assert(alignof(Cursor<uint64_t>) == HOT_ATOMIC_BYTES);
static_assert(sizeof(Cursor<uint64_t>) == HOT_ATOMIC_BYTES,
              "adjacent queue cursors must not share an Apple cache line");

void ring(Doorbell *doorbell, bool all) {
    kc_atomic_u32_fetch_add_release(&doorbell->value, 1);
    if (all) {
        kc_port_wake_u32_all(doorbell->wait);
        return;
    }
    kc_port_wake_u32_one(doorbell->wait);
}

struct PcmSlot {
    std::atomic<uint32_t> state{SLOT_FREE};
    std::atomic<uint64_t> generation{1};
    std::atomic<uint64_t> identity{0};
    float *samples = nullptr;
    uint32_t frames = 0;
    uint32_t channels = 0;
    uint32_t sample_rate = 0;
    LfmTicketIdV1 ticket{};
};

struct PcmPool {
    PcmSlot *slots = nullptr;
    LfmPcmLeaseV1 *ring = nullptr;
    uint32_t capacity = 0;
    uint32_t samples_per_slot = 0;
    uint32_t direction = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
    Cursor<uint32_t> cursor;
    std::mutex push_mutex;
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

struct TextRing {
    TextCommand *records = nullptr;
    uint32_t capacity = 0;
    uint64_t head = 0;
    uint64_t tail = 0;
    std::mutex mutex;
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

bool pool_push(PcmPool *pool, const LfmPcmLeaseV1 &lease) {
    std::lock_guard<std::mutex> guard(pool->push_mutex);
    uint64_t tail = pool->tail.value.load(std::memory_order_relaxed);
    uint64_t head = pool->head.value.load(std::memory_order_acquire);
    if (tail - head == pool->capacity) return false;
    pool->ring[tail % pool->capacity] = lease;
    pool->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool pool_pop(PcmPool *pool, LfmPcmLeaseV1 *out) {
    uint64_t head = pool->head.value.load(std::memory_order_relaxed);
    uint64_t tail = pool->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = pool->ring[head % pool->capacity];
    pool->head.value.store(head + 1, std::memory_order_release);
    return true;
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
    pool->ring = new (std::nothrow) LfmPcmLeaseV1[capacity]();
    if (!pool->slots || !pool->ring) return LFM_STATUS_OUT_OF_MEMORY;
    pool->capacity = capacity;
    pool->samples_per_slot = samples_per_slot;
    pool->direction = direction;
    for (uint32_t i = 0; i < capacity; ++i) {
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
    if (pool->direction == LFM_PCM_LEASE_CAPTURE &&
        !ticket_equal(slot->ticket, lease->ticket)) {
        return LFM_STATUS_STALE;
    }
    if (lease->channels != slot->channels || lease->sample_rate != slot->sample_rate ||
        lease->frames == 0 || lease->frames > slot->frames ||
        (pool->direction == LFM_PCM_LEASE_CAPTURE && lease->frames != slot->frames)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    size_t samples = 0;
    if (!checked_samples(lease->frames, lease->channels, &samples) ||
        samples > pool->samples_per_slot || lease->offset_bytes != 0 ||
        lease->length_bytes != samples * sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out = slot;
    if (out_index) *out_index = index;
    return 0;
}

int release_slot(PcmPool *pool, const LfmPcmLeaseV1 *lease, Doorbell *space,
                 uint32_t allowed_states = UINT32_MAX) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    uint32_t state = slot->state.load(std::memory_order_acquire);
    for (;;) {
        if (state == SLOT_FREE || state == SLOT_RELEASING || state == SLOT_RETIRED) {
            return LFM_STATUS_STALE;
        }
        if ((allowed_states & (UINT32_C(1) << state)) == 0) {
            return LFM_STATUS_BUSY;
        }
        if (slot->state.compare_exchange_weak(state, SLOT_RELEASING,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            slot->frames = 0;
            slot->channels = 0;
            slot->sample_rate = 0;
            slot->ticket = {};
            slot->identity.store(0, std::memory_order_relaxed);
            uint64_t generation = slot->generation.load(std::memory_order_relaxed);
            if (generation == std::numeric_limits<uint64_t>::max()) {
                slot->state.store(SLOT_RETIRED, std::memory_order_release);
                return 0;
            }
            slot->generation.store(generation + 1, std::memory_order_relaxed);
            slot->state.store(SLOT_FREE, std::memory_order_release);
            if (space) ring(space, false);
            return 0;
        }
    }
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
    std::lock_guard<std::mutex> guard(ring->mutex);
    if (ring->tail - ring->head == ring->capacity) return false;
    ring->records[ring->tail % ring->capacity] = command;
    ring->tail++;
    return true;
}

bool text_pop(TextRing *queue, TextCommand *out, Doorbell *space) {
    {
        std::lock_guard<std::mutex> guard(queue->mutex);
        if (queue->head == queue->tail) return false;
        *out = queue->records[queue->head % queue->capacity];
        queue->head++;
    }
    ring(space, false);
    return true;
}

bool text_empty(TextRing *ring) {
    std::lock_guard<std::mutex> guard(ring->mutex);
    return ring->head == ring->tail;
}

} // namespace

struct LfmRuntime {
    void *engine = nullptr;
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

struct LfmSession {
    LfmRuntime *runtime = nullptr;
    LfmModel *model = nullptr;
    LfmConversation *conversation = nullptr;
    LfmCallbacksV1 callbacks{};
    uint64_t id = 0;
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    uint32_t max_new_tokens = 0;
    uint32_t generation = 1;
    bool dock_only = false;
    std::atomic<uint32_t> state{LFM_SESSION_CREATED};
    LfmRouteEpoch epoch{};
    std::atomic<uint64_t> sequence{1};
    std::atomic<bool> stop{false};
    std::atomic<bool> event_done{false};
    std::atomic<bool> sink_failed{false};
    std::atomic<int32_t> terminal_status{0};
    std::atomic<uint64_t> coordinator_parks{0};
    std::atomic<uint64_t> coordinator_wakes{0};
    std::atomic<uint64_t> notification_parks{0};
    std::atomic<uint64_t> callbacks_entered{0};
    std::atomic<uint64_t> capture_consumed{0};
    std::atomic<uint64_t> capture_stale{0};
    std::atomic<uint64_t> playback_published{0};
    std::atomic<uint64_t> playback_consumed{0};
    std::atomic<uint64_t> text_commands_accepted{0};
    std::atomic<uint64_t> text_commands_consumed{0};
    std::atomic<uint64_t> text_commands_stale{0};
    Doorbell work_doorbell;
    Doorbell event_doorbell;
    Doorbell event_space_doorbell;
    Doorbell playback_doorbell;
    Doorbell playback_space_doorbell;
    Doorbell capture_space_doorbell;
    Doorbell command_space_doorbell;
    PcmPool capture;
    PcmPool playback;
    EventRing events;
    TextRing commands;
    kc_port_thread *coordinator = nullptr;
    kc_port_thread *notification = nullptr;
    bool coordinator_started = false;
    bool notification_started = false;
    bool threads_joined = false;
    bool start_cleanup = false;
    /* Lock order is runtime.children_mutex -> lifecycle_mutex. join_mutex is
     * outermost only for concurrent join callers and is never acquired by
     * start or stop. No native thread join holds lifecycle_mutex. */
    mutable std::mutex lifecycle_mutex;
    mutable std::mutex join_mutex;
    mutable std::mutex publication_mutex;

    ~LfmSession() {
        if (command_space_doorbell.wait) {
            kc_port_wait_u32_release(command_space_doorbell.wait);
        }
        if (capture_space_doorbell.wait) {
            kc_port_wait_u32_release(capture_space_doorbell.wait);
        }
        if (playback_space_doorbell.wait) {
            kc_port_wait_u32_release(playback_space_doorbell.wait);
        }
        if (playback_doorbell.wait) kc_port_wait_u32_release(playback_doorbell.wait);
        if (event_space_doorbell.wait) {
            kc_port_wait_u32_release(event_space_doorbell.wait);
        }
        if (event_doorbell.wait) kc_port_wait_u32_release(event_doorbell.wait);
        if (work_doorbell.wait) kc_port_wait_u32_release(work_doorbell.wait);
        pool_destroy(&playback);
        pool_destroy(&capture);
        delete[] events.records;
        delete[] commands.records;
    }
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

int mixed_push(LfmSession *session, const TextCommand &command) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(&session->capture, &command.capture, &slot, nullptr);
    if (rc != 0) return rc;
    if (command.capture.stream_epoch != command.epoch ||
        !ticket_equal(command.capture.ticket, command.ticket) ||
        command.capture.sample_rate != session->sample_rate) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> guard(session->commands.mutex);
    if (session->commands.tail - session->commands.head ==
        session->commands.capacity) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    uint32_t expected = SLOT_RESERVED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_PUBLISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    session->commands.records[session->commands.tail %
                              session->commands.capacity] = command;
    session->commands.tail++;
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

void request_stop(LfmSession *session, int32_t status) {
    if (status != 0) {
        int32_t expected = 0;
        session->terminal_status.compare_exchange_strong(expected, status,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire);
    }
    bool first = !session->stop.exchange(true, std::memory_order_acq_rel);
    if (first) {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        uint64_t epoch = session->epoch.load(std::memory_order_relaxed);
        if (epoch != std::numeric_limits<uint64_t>::max()) {
            session->epoch.store(epoch + 1, std::memory_order_release);
        }
    }
    uint32_t state = session->state.load(std::memory_order_acquire);
    if (state == LFM_SESSION_RUNNING) {
        session->state.compare_exchange_strong(state, LFM_SESSION_STOPPING,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire);
    }
    ring(&session->work_doorbell, true);
    ring(&session->event_space_doorbell, true);
    ring(&session->playback_doorbell, true);
    ring(&session->playback_space_doorbell, true);
    ring(&session->capture_space_doorbell, true);
    ring(&session->command_space_doorbell, true);
}

bool publish_event(LfmSession *session, uint32_t kind, uint64_t epoch,
                   LfmTicketIdV1 ticket, int32_t status,
                   const void *payload, size_t payload_bytes, uint32_t flags = 0,
                   bool gate_epoch = false) {
    if (payload_bytes > EVENT_PAYLOAD_CAPACITY) {
        request_stop(session, LFM_STATUS_INTERNAL);
        return false;
    }
    EventRecord record{};
    record.kind = kind;
    record.flags = flags;
    record.epoch = epoch;
    record.ticket = ticket;
    record.status = status;
    record.payload_bytes = static_cast<uint32_t>(payload_bytes);
    if (payload_bytes != 0) std::memcpy(record.payload, payload, payload_bytes);
    for (;;) {
        {
            std::lock_guard<std::mutex> guard(session->publication_mutex);
            if (session->sink_failed.load(std::memory_order_acquire) ||
                (gate_epoch && session->epoch.load(std::memory_order_acquire) != epoch)) {
                return false;
            }
            if (event_push(&session->events, record)) {
                ring(&session->event_doorbell, false);
                return true;
            }
        }
        uint32_t expected =
            kc_atomic_u32_load_acquire(&session->event_space_doorbell.value);
        {
            std::lock_guard<std::mutex> guard(session->publication_mutex);
            if (session->sink_failed.load(std::memory_order_acquire) ||
                (gate_epoch && session->epoch.load(std::memory_order_acquire) != epoch)) {
                return false;
            }
            if (event_push(&session->events, record)) {
                ring(&session->event_doorbell, false);
                return true;
            }
        }
        int rc = kc_port_wait_u32(session->event_space_doorbell.wait, expected, 0);
        if (rc != 0 && !session->sink_failed.load(std::memory_order_acquire)) {
            request_stop(session, rc);
            return false;
        }
    }
}

void publish_error(LfmSession *session, int32_t status, const char *message) {
    size_t bytes = std::strlen(message);
    if (bytes > EVENT_PAYLOAD_CAPACITY) bytes = EVENT_PAYLOAD_CAPACITY;
    publish_event(session, LFM_EVENT_ERROR, session->epoch.load(std::memory_order_acquire),
                  next_ticket(session, LFM_TICKET_CONTROL), status, message, bytes);
    request_stop(session, status);
}

bool publish_turn(LfmSession *session, uint64_t action_epoch,
                  LfmTicketIdV1 ticket, uint32_t playback_count,
                  uint32_t emitted, uint32_t flags, int32_t status);

bool publish_action_failure(LfmSession *session, uint64_t action_epoch,
                            LfmTicketIdV1 ticket, int32_t status,
                            const char *message, uint32_t playback_count = 0,
                            uint32_t emitted = 0) {
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        return publish_turn(session, action_epoch, ticket, playback_count,
                            emitted, 0, LFM_STATUS_STALE);
    }
    size_t bytes = std::strlen(message);
    if (bytes > EVENT_PAYLOAD_CAPACITY) bytes = EVENT_PAYLOAD_CAPACITY;
    if (!publish_event(session, LFM_EVENT_ERROR, action_epoch, ticket, status,
                       message, bytes, 0, true)) {
        if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
            return publish_turn(session, action_epoch, ticket, playback_count,
                                emitted, 0, LFM_STATUS_STALE);
        }
        return false;
    }
    if (publish_turn(session, action_epoch, ticket, playback_count, emitted, 0,
                     status)) {
        return true;
    }
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        return publish_turn(session, action_epoch, ticket, playback_count,
                            emitted, 0, LFM_STATUS_STALE);
    }
    return false;
}

bool publish_turn(LfmSession *session, uint64_t action_epoch,
                  LfmTicketIdV1 ticket, uint32_t playback_count,
                  uint32_t emitted, uint32_t flags, int32_t status = 0) {
    LfmTurnEventV1 turn = {
        .size = sizeof(LfmTurnEventV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .playback_leases = playback_count,
        .emitted_items = emitted,
    };
    if (playback_count != 0) flags |= LFM_EVENT_FLAG_HAS_AUDIO;
    bool gate_epoch = status != LFM_STATUS_STALE &&
                      status != LFM_STATUS_CANCELLED;
    return publish_event(session, LFM_EVENT_TURN, action_epoch, ticket, status,
                         &turn, sizeof(turn), flags, gate_epoch);
}

struct PreparedPlayback {
    LfmPcmLeaseV1 lease{};
    size_t samples = 0;
    bool active = false;
};

void release_prepared(LfmSession *session, PreparedPlayback *playback) {
    if (!playback || !playback->active) return;
    (void)lfm_audio_dock_release(session, &playback->lease);
    playback->active = false;
    playback->samples = 0;
}

bool reserve_playback(LfmSession *session, uint64_t action_epoch,
                      LfmTicketIdV1 ticket, uint32_t playback_count,
                      uint32_t emitted, LfmPcmLeaseV1 *out) {
    const auto reserve = [&]() -> int {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        if (session->stop.load(std::memory_order_acquire)) {
            return LFM_STATUS_CANCELLED;
        }
        if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
            return LFM_STATUS_STALE;
        }
        return lfm_audio_dock_reserve(session, LFM_PCM_LEASE_PLAYBACK,
                                      LFM_MIMI_PCM_CAPACITY, 24000, out);
    };
    for (;;) {
        int rc = reserve();
        if (rc == 0) return true;
        if (rc != LFM_STATUS_WOULD_BLOCK) {
            if (rc != LFM_STATUS_CANCELLED && rc != LFM_STATUS_STALE) {
                publish_action_failure(session, action_epoch, ticket, rc,
                                       "playback reservation failed",
                                       playback_count, emitted);
            }
            return false;
        }
        uint32_t expected =
            kc_atomic_u32_load_acquire(&session->playback_space_doorbell.value);
        if (session->stop.load(std::memory_order_acquire) ||
            session->epoch.load(std::memory_order_acquire) != action_epoch) {
            return false;
        }
        rc = reserve();
        if (rc == 0) return true;
        if (rc != LFM_STATUS_WOULD_BLOCK) {
            if (rc != LFM_STATUS_CANCELLED && rc != LFM_STATUS_STALE) {
                publish_action_failure(session, action_epoch, ticket, rc,
                                       "playback reservation failed",
                                       playback_count, emitted);
            }
            return false;
        }
        rc = kc_port_wait_u32(session->playback_space_doorbell.wait, expected, 0);
        if (rc != 0 && !session->stop.load(std::memory_order_acquire) &&
            session->epoch.load(std::memory_order_acquire) == action_epoch) {
            publish_action_failure(session, action_epoch, ticket, rc,
                                   "playback-space wait failed", playback_count,
                                   emitted);
            return false;
        }
    }
}

bool publish_audio(LfmSession *session, const LfmNativeEmission &emission,
                   uint64_t action_epoch, LfmTicketIdV1 ticket,
                   uint32_t playback_count, uint32_t emitted,
                   PreparedPlayback *playback) {
    if (emission.code_count != LFM_MIMI_CODEBOOKS) {
        release_prepared(session, playback);
        publish_action_failure(session, action_epoch, ticket, LFM_STATUS_INTERNAL,
                               "native audio code count mismatch", playback_count,
                               emitted);
        return false;
    }
    if (!playback || !playback->active || playback->samples == 0 ||
        playback->samples > UINT32_MAX) {
        release_prepared(session, playback);
        publish_action_failure(session, action_epoch, ticket,
                               LFM_STATUS_INTERNAL,
                               "native Mimi route produced invalid PCM",
                               playback_count,
                               emitted);
        return false;
    }
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        release_prepared(session, playback);
        return false;
    }
    playback->lease.ticket = ticket;
    playback->lease.frames = static_cast<uint32_t>(playback->samples);
    playback->lease.length_bytes =
        static_cast<uint32_t>(playback->samples * sizeof(float));
    playback->lease.flags = LFM_PCM_LEASE_PLAYBACK;
    const int rc = lfm_audio_dock_publish(session, &playback->lease);
    if (rc != 0) {
        release_prepared(session, playback);
        if (rc != LFM_STATUS_STALE && rc != LFM_STATUS_CANCELLED) {
            publish_action_failure(session, action_epoch, ticket, rc,
                                   "playback publication failed", playback_count,
                                   emitted);
        }
        return false;
    }
    playback->active = false;
    playback->samples = 0;
    return true;
}

bool handle_emission(LfmSession *session, const LfmNativeEmission &emission,
                     uint64_t action_epoch, LfmTicketIdV1 ticket,
                     uint32_t *playback_count, uint32_t emitted,
                     bool *finished, PreparedPlayback *playback) {
    if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
        release_prepared(session, playback);
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        *finished = true;
        return publish_turn(session, action_epoch, ticket, *playback_count, emitted,
                            0, LFM_STATUS_STALE);
    }
    if (emission.kind == LFM_NATIVE_EMISSION_NONE) {
        if (!playback || !playback->active) return true;
        release_prepared(session, playback);
        publish_action_failure(session, action_epoch, ticket,
                               LFM_STATUS_INTERNAL,
                               "audio route returned no emission",
                               *playback_count, emitted);
        return false;
    }
    if (emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES) {
        const int needs_pcm = lfm_native_emission_needs_pcm(&emission);
        if (needs_pcm < 0) {
            release_prepared(session, playback);
            publish_action_failure(session, action_epoch, ticket,
                                   LFM_STATUS_INTERNAL,
                                   "invalid native audio emission",
                                   *playback_count, emitted);
            return false;
        }
        // EOAudio is a recurrence/context sentinel. It must reach the next
        // native token pass, but it is not a codec frame and never reaches Mimi.
        if (needs_pcm == 0) {
            release_prepared(session, playback);
            return true;
        }
        if (!publish_audio(session, emission, action_epoch, ticket,
                           *playback_count, emitted, playback)) {
            return false;
        }
        (*playback_count)++;
        return true;
    }
    if (emission.kind == LFM_NATIVE_EMISSION_TEXT) {
        if (playback && playback->active) {
            release_prepared(session, playback);
            publish_action_failure(session, action_epoch, ticket,
                                   LFM_STATUS_INTERNAL,
                                   "audio route returned text",
                                   *playback_count, emitted);
            return false;
        }
        if (emission.text_bytes > sizeof(emission.text)) {
            publish_action_failure(session, action_epoch, ticket,
                                   LFM_STATUS_INTERNAL,
                                   "native text emission exceeds bound",
                                   *playback_count, emitted);
            return false;
        }
        return publish_event(session, LFM_EVENT_TEXT, action_epoch, ticket, 0,
                             emission.text, emission.text_bytes, 0, true);
    }
    if (emission.kind == LFM_NATIVE_EMISSION_FINISHED) {
        if (playback && playback->active) {
            release_prepared(session, playback);
            publish_action_failure(session, action_epoch, ticket,
                                   LFM_STATUS_INTERNAL,
                                   "audio route finished with a live lease",
                                   *playback_count, emitted);
            return false;
        }
        *finished = true;
        return publish_turn(session, action_epoch, ticket, *playback_count, emitted, 0);
    }
    publish_action_failure(session, action_epoch, ticket, LFM_STATUS_INTERNAL,
                           "unknown native emission kind", *playback_count,
                           emitted);
    release_prepared(session, playback);
    return false;
}

void run_action(LfmSession *session, LfmNativeEmission emission,
                uint64_t action_epoch, LfmTicketIdV1 ticket) {
    bool finished = false;
    uint32_t playback_count = 0;
    uint32_t emitted = 0;
    PreparedPlayback playback{};
    for (;;) {
        if (emission.kind == LFM_NATIVE_EMISSION_TEXT ||
            (emission.kind == LFM_NATIVE_EMISSION_AUDIO_CODES &&
             (emission.flags & EMISSION_AUDIO_END) == 0)) {
            emitted++;
        }
        if (!handle_emission(session, emission, action_epoch, ticket,
                             &playback_count, emitted, &finished, &playback)) {
            if (session->stop.load(std::memory_order_acquire)) {
                publish_turn(session, action_epoch, ticket, playback_count, emitted,
                             0, LFM_STATUS_CANCELLED);
            } else if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
                publish_turn(session, action_epoch, ticket, playback_count, emitted,
                             0, LFM_STATUS_STALE);
            } else {
                /* A reliable text/PCM publication or codec failure cannot be
                 * skipped while preserving the turn stream. Make the action
                 * terminal; coordinator teardown commits any already-emitted
                 * pending token before retiring the conversation. */
                request_stop(session, LFM_STATUS_INTERNAL);
            }
            return;
        }
        if (finished) {
            return;
        }
        if (emitted >= session->max_new_tokens) {
            int rc = lfm_conversation_interrupt_native(session->conversation);
            if (rc != 0) {
                publish_action_failure(session, action_epoch, ticket, rc,
                                       "native generation limit interrupt failed",
                                       playback_count, emitted);
                request_stop(session, rc);
                return;
            }
            publish_turn(session, action_epoch, ticket, playback_count, emitted,
                         LFM_EVENT_FLAG_TRUNCATED);
            return;
        }
        if (session->stop.load(std::memory_order_acquire)) {
            publish_turn(session, action_epoch, ticket, playback_count, emitted,
                         0, LFM_STATUS_CANCELLED);
            return;
        }
        if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
            publish_turn(session, action_epoch, ticket, playback_count, emitted,
                         0, LFM_STATUS_STALE);
            return;
        }
        emission = {};
        int needs_playback =
            lfm_conversation_next_requires_playback_native(session->conversation);
        if (needs_playback < 0) {
            publish_action_failure(session, action_epoch, ticket, needs_playback,
                                   "native route requirement failed",
                                   playback_count, emitted);
            request_stop(session, needs_playback);
            return;
        }
        int rc = 0;
        if (needs_playback != 0) {
            if (!reserve_playback(session, action_epoch, ticket,
                                  playback_count, emitted, &playback.lease)) {
                if (session->epoch.load(std::memory_order_acquire) !=
                    action_epoch) {
                    publish_turn(session, action_epoch, ticket, playback_count,
                                 emitted, 0, LFM_STATUS_STALE);
                }
                return;
            }
            playback.active = true;
            float *pcm = nullptr;
            size_t capacity = 0;
            rc = lfm_audio_dock_resolve_mut(session, &playback.lease, &pcm,
                                            &capacity);
            if (rc == 0) {
                const LfmAudioRouteTarget target = {
                    .epoch = &session->epoch,
                    .expected_epoch = action_epoch,
                    .pcm = pcm,
                    .pcm_capacity = capacity,
                };
                rc = lfm_conversation_next_into_native(
                    session->conversation, &target, &emission,
                    &playback.samples);
            }
        } else {
            rc = lfm_conversation_next_native(session->conversation,
                                              &emission);
        }
        if (rc != 0) {
            release_prepared(session, &playback);
            if (session->epoch.load(std::memory_order_acquire) != action_epoch) {
                publish_turn(session, action_epoch, ticket, playback_count,
                             emitted, 0, LFM_STATUS_STALE);
                return;
            }
            publish_action_failure(session, action_epoch, ticket, rc,
                                   "native recurrence failed", playback_count,
                                   emitted);
            request_stop(session, rc);
            return;
        }
    }
}

void flush_capture(LfmSession *session) {
    LfmPcmLeaseV1 lease{};
    while (pool_pop(&session->capture, &lease)) {
        PcmSlot *slot = nullptr;
        if (claim_published(&session->capture, &lease, &slot) == 0) {
            (void)slot;
            release_slot(&session->capture, &lease,
                         &session->capture_space_doorbell);
            publish_turn(session, lease.stream_epoch, lease.ticket, 0, 0, 0,
                         LFM_STATUS_CANCELLED);
        }
    }
}

void flush_commands(LfmSession *session) {
    TextCommand command{};
    while (text_pop(&session->commands, &command,
                    &session->command_space_doorbell)) {
        if (command.kind == COMMAND_MIXED) {
            PcmSlot *slot = nullptr;
            if (claim_published(&session->capture, &command.capture, &slot) == 0) {
                (void)slot;
                release_slot(&session->capture, &command.capture,
                             &session->capture_space_doorbell);
            }
        }
        publish_turn(session, command.epoch, command.ticket, 0, 0, 0,
                     LFM_STATUS_CANCELLED);
    }
}

void flush_published(PcmPool *pool) {
    for (uint32_t i = 0; i < pool->capacity; ++i) {
        PcmSlot &slot = pool->slots[i];
        uint32_t expected = SLOT_PUBLISHED;
        if (slot.state.compare_exchange_strong(expected, SLOT_RELEASING,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
            slot.frames = 0;
            slot.channels = 0;
            slot.sample_rate = 0;
            slot.ticket = {};
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

bool apply_epoch(LfmSession *session, uint64_t epoch) {
    if (session->dock_only) return true;
    int rc = lfm_conversation_interrupt_native(session->conversation);
    if (rc == 0) return true;
    publish_error(session, rc, "native conversation interrupt failed");
    (void)epoch;
    return false;
}

bool synchronize_epoch(LfmSession *session, uint64_t *applied_epoch) {
    for (;;) {
        const uint64_t current_epoch =
            session->epoch.load(std::memory_order_acquire);
        if (current_epoch == *applied_epoch) return true;
        if (!apply_epoch(session, current_epoch)) return false;
        *applied_epoch = current_epoch;
        static constexpr char interrupted[] = "interrupted";
        if (!publish_event(session, LFM_EVENT_STATE, current_epoch,
                           next_ticket(session, LFM_TICKET_CONTROL), 0,
                           interrupted, sizeof(interrupted) - 1)) {
            return false;
        }
        /* Publication can park behind reliable backpressure. Recheck the
         * expected value before any command is allowed to reach inference. */
    }
}

void process_capture(LfmSession *session, const LfmPcmLeaseV1 &lease) {
    PcmSlot *slot = nullptr;
    int rc = claim_published(&session->capture, &lease, &slot);
    if (rc != 0) return;
    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (lease.stream_epoch != current_epoch) {
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        release_slot(&session->capture, &lease,
                     &session->capture_space_doorbell);
        publish_turn(session, lease.stream_epoch, lease.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }
    session->capture_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        release_slot(&session->capture, &lease,
                     &session->capture_space_doorbell);
        publish_turn(session, current_epoch, lease.ticket, 0, 0, 0);
        return;
    }

    LfmNativeEmission emission{};
    size_t samples = lease.length_bytes / sizeof(float);
    rc = lfm_conversation_begin_pcm_native(session->conversation, slot->samples, samples,
                                           lease.sample_rate, &emission);
    release_slot(&session->capture, &lease,
                 &session->capture_space_doorbell);
    if (rc != 0) {
        publish_action_failure(session, current_epoch, lease.ticket, rc,
                               "native PCM prefill failed");
        request_stop(session, rc);
        return;
    }
    run_action(session, emission, current_epoch, lease.ticket);
}

void process_text(LfmSession *session, const TextCommand &command) {
    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (command.epoch != current_epoch) {
        session->text_commands_stale.fetch_add(1, std::memory_order_relaxed);
        publish_turn(session, command.epoch, command.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }
    session->text_commands_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        publish_turn(session, current_epoch, command.ticket, 0, 0, 0);
        return;
    }
    LfmNativeEmission emission{};
    int rc = lfm_conversation_begin_text_native(session->conversation,
                                                command.text, command.bytes,
                                                &emission);
    if (rc != 0) {
        publish_action_failure(session, current_epoch, command.ticket, rc,
                               "native typed-input prefill failed");
        request_stop(session, rc);
        return;
    }
    run_action(session, emission, current_epoch, command.ticket);
}

void process_mixed(LfmSession *session, const TextCommand &command) {
    PcmSlot *slot = nullptr;
    int rc = claim_published(&session->capture, &command.capture, &slot);
    if (rc != 0) {
        publish_action_failure(session, command.epoch, command.ticket, rc,
                               "mixed capture lease claim failed");
        return;
    }

    uint64_t current_epoch = session->epoch.load(std::memory_order_acquire);
    if (command.epoch != current_epoch ||
        command.capture.stream_epoch != current_epoch) {
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        session->text_commands_stale.fetch_add(1, std::memory_order_relaxed);
        release_slot(&session->capture, &command.capture,
                     &session->capture_space_doorbell);
        publish_turn(session, command.epoch, command.ticket, 0, 0, 0,
                     LFM_STATUS_STALE);
        return;
    }

    session->capture_consumed.fetch_add(1, std::memory_order_relaxed);
    session->text_commands_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        release_slot(&session->capture, &command.capture,
                     &session->capture_space_doorbell);
        publish_turn(session, current_epoch, command.ticket, 0, 0, 0);
        return;
    }

    LfmNativeEmission emission{};
    size_t samples = command.capture.length_bytes / sizeof(float);
    rc = lfm_conversation_begin_mixed_native(
        session->conversation, command.text, command.bytes, slot->samples,
        samples, command.capture.sample_rate, &emission);
    int release = release_slot(&session->capture, &command.capture,
                               &session->capture_space_doorbell);
    if (release != 0) {
        publish_action_failure(session, current_epoch, command.ticket, release,
                               "mixed capture lease release failed");
        request_stop(session, release);
        return;
    }
    if (rc != 0) {
        publish_action_failure(session, current_epoch, command.ticket, rc,
                               "native mixed text/PCM prefill failed");
        request_stop(session, rc);
        return;
    }
    run_action(session, emission, current_epoch, command.ticket);
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
    publish_action_failure(session, command.epoch, command.ticket,
                           LFM_STATUS_INTERNAL, "unknown native command kind");
}

void *coordinator_main(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    uint64_t applied_epoch = session->epoch.load(std::memory_order_acquire);
    static constexpr char running[] = "running";
    publish_event(session, LFM_EVENT_STATE, applied_epoch,
                  next_ticket(session, LFM_TICKET_SESSION), 0,
                  running, sizeof(running) - 1);

    for (;;) {
        if (!synchronize_epoch(session, &applied_epoch)) break;

        bool progressed = false;
        TextCommand command{};
        while (text_pop(&session->commands, &command,
                        &session->command_space_doorbell)) {
            progressed = true;
            /* An interrupt may race between the drain predicate and this pop.
             * Apply it before the popped record can touch conversation state. */
            if (!synchronize_epoch(session, &applied_epoch)) break;
            process_command(session, command);
            if (session->stop.load(std::memory_order_acquire) ||
                session->epoch.load(std::memory_order_acquire) != applied_epoch) {
                break;
            }
        }
        if (session->stop.load(std::memory_order_acquire)) break;
        if (session->epoch.load(std::memory_order_acquire) != applied_epoch) continue;
        LfmPcmLeaseV1 lease{};
        while (pool_pop(&session->capture, &lease)) {
            progressed = true;
            if (!synchronize_epoch(session, &applied_epoch)) break;
            process_capture(session, lease);
            if (session->stop.load(std::memory_order_acquire) ||
                session->epoch.load(std::memory_order_acquire) != applied_epoch) {
                break;
            }
        }
        if (session->stop.load(std::memory_order_acquire)) break;
        if (session->epoch.load(std::memory_order_acquire) != applied_epoch) continue;
        if (progressed) continue;

        uint32_t expected = kc_atomic_u32_load_acquire(&session->work_doorbell.value);
        if (session->stop.load(std::memory_order_acquire) ||
            session->capture.head.value.load(std::memory_order_acquire) !=
                session->capture.tail.value.load(std::memory_order_acquire) ||
            !text_empty(&session->commands) ||
            session->epoch.load(std::memory_order_acquire) != applied_epoch) {
            continue;
        }
        session->coordinator_parks.fetch_add(1, std::memory_order_relaxed);
        int rc = kc_port_wait_u32(session->work_doorbell.wait, expected, 0);
        session->coordinator_wakes.fetch_add(1, std::memory_order_relaxed);
        if (rc != 0 && !session->stop.load(std::memory_order_acquire)) {
            publish_error(session, rc, "coordinator wait failed");
            break;
        }
    }

    if (!session->dock_only) {
        const int teardown =
            lfm_conversation_interrupt_native(session->conversation);
        if (teardown != 0) request_stop(session, teardown);
    }
    flush_commands(session);
    flush_capture(session);
    flush_published(&session->capture);
    flush_published(&session->playback);
    session->event_done.store(true, std::memory_order_release);
    ring(&session->event_doorbell, true);
    ring(&session->playback_doorbell, true);
    return nullptr;
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

void *notification_main(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    for (;;) {
        EventRecord record{};
        if (event_pop(&session->events, &record)) {
            ring(&session->event_space_doorbell, false);
            if (!session->sink_failed.load(std::memory_order_acquire) &&
                invoke_callback(session, record) != 0) {
                session->sink_failed.store(true, std::memory_order_release);
                request_stop(session, LFM_STATUS_HOST_SINK);
            }
            continue;
        }
        if (session->event_done.load(std::memory_order_acquire)) break;
        uint32_t expected = kc_atomic_u32_load_acquire(&session->event_doorbell.value);
        if (session->events.head.value.load(std::memory_order_acquire) !=
                session->events.tail.value.load(std::memory_order_acquire) ||
            session->event_done.load(std::memory_order_acquire)) {
            continue;
        }
        session->notification_parks.fetch_add(1, std::memory_order_relaxed);
        int rc = kc_port_wait_u32(session->event_doorbell.wait, expected, 0);
        if (rc != 0 && !session->event_done.load(std::memory_order_acquire)) {
            request_stop(session, rc);
        }
    }

    EventRecord stopped{};
    stopped.kind = LFM_EVENT_STOPPED;
    stopped.epoch = session->epoch.load(std::memory_order_acquire);
    stopped.ticket = next_ticket(session, LFM_TICKET_SESSION);
    stopped.status = session->terminal_status.load(std::memory_order_acquire);
    static constexpr char payload[] = "stopped";
    stopped.payload_bytes = sizeof(payload) - 1;
    std::memcpy(stopped.payload, payload, sizeof(payload) - 1);
    (void)invoke_callback(session, stopped);
    return nullptr;
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
                LfmTicketIdV1 *out_ticket, bool wait_for_space) {
    if (!session || !utf8 || utf8_bytes == 0 ||
        utf8_bytes > LFM_TEXT_COMMAND_MAX_BYTES || !out_ticket) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }

    TextCommand command{};
    command.ticket = next_ticket(session, LFM_TICKET_TURN);
    command.epoch = session->epoch.load(std::memory_order_acquire);
    command.bytes = static_cast<uint32_t>(utf8_bytes);
    std::memcpy(command.text, utf8, utf8_bytes);
    auto admit = [&]() -> int {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
            session->stop.load(std::memory_order_acquire)) {
            return LFM_STATUS_CANCELLED;
        }
        if (session->epoch.load(std::memory_order_acquire) != command.epoch) {
            return LFM_STATUS_STALE;
        }
        return text_push(&session->commands, command) ? 0 : LFM_STATUS_WOULD_BLOCK;
    };

    for (;;) {
        int rc = admit();
        if (rc == 0) {
            session->text_commands_accepted.fetch_add(1, std::memory_order_relaxed);
            *out_ticket = command.ticket;
            ring(&session->work_doorbell, false);
            return 0;
        }
        if (rc != LFM_STATUS_WOULD_BLOCK || !wait_for_space) return rc;

        uint32_t expected =
            kc_atomic_u32_load_acquire(&session->command_space_doorbell.value);
        rc = admit();
        if (rc == 0) {
            session->text_commands_accepted.fetch_add(1, std::memory_order_relaxed);
            *out_ticket = command.ticket;
            ring(&session->work_doorbell, false);
            return 0;
        }
        if (rc != LFM_STATUS_WOULD_BLOCK) return rc;

        rc = kc_port_wait_u32(session->command_space_doorbell.wait, expected, 0);
        if (rc != 0) {
            if (session->stop.load(std::memory_order_acquire)) {
                return LFM_STATUS_CANCELLED;
            }
            if (session->epoch.load(std::memory_order_acquire) != command.epoch) {
                return LFM_STATUS_STALE;
            }
            return rc;
        }
    }
}

int submit_mixed(LfmSession *session, const char *utf8, size_t utf8_bytes,
                 const LfmPcmLeaseV1 *capture,
                 LfmTicketIdV1 *out_ticket, bool wait_for_space) {
    if (!session || !utf8 || utf8_bytes == 0 ||
        utf8_bytes > LFM_TEXT_COMMAND_MAX_BYTES || !capture || !out_ticket ||
        capture->flags != LFM_PCM_LEASE_CAPTURE) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }

    TextCommand command{};
    command.ticket = capture->ticket;
    command.epoch = capture->stream_epoch;
    command.bytes = static_cast<uint32_t>(utf8_bytes);
    command.kind = COMMAND_MIXED;
    command.capture = *capture;
    std::memcpy(command.text, utf8, utf8_bytes);
    auto admit = [&]() -> int {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        if (session->state.load(std::memory_order_acquire) != LFM_SESSION_RUNNING ||
            session->stop.load(std::memory_order_acquire)) {
            return LFM_STATUS_CANCELLED;
        }
        if (session->epoch.load(std::memory_order_acquire) != command.epoch) {
            return LFM_STATUS_STALE;
        }
        return mixed_push(session, command);
    };

    for (;;) {
        int rc = admit();
        if (rc == 0) {
            session->text_commands_accepted.fetch_add(1,
                                                       std::memory_order_relaxed);
            *out_ticket = command.ticket;
            ring(&session->work_doorbell, false);
            return 0;
        }
        if (rc != LFM_STATUS_WOULD_BLOCK || !wait_for_space) return rc;

        uint32_t expected =
            kc_atomic_u32_load_acquire(&session->command_space_doorbell.value);
        rc = admit();
        if (rc == 0) {
            session->text_commands_accepted.fetch_add(1,
                                                       std::memory_order_relaxed);
            *out_ticket = command.ticket;
            ring(&session->work_doorbell, false);
            return 0;
        }
        if (rc != LFM_STATUS_WOULD_BLOCK) return rc;

        rc = kc_port_wait_u32(session->command_space_doorbell.wait, expected, 0);
        if (rc != 0) {
            if (session->stop.load(std::memory_order_acquire)) {
                return LFM_STATUS_CANCELLED;
            }
            if (session->epoch.load(std::memory_order_acquire) != command.epoch) {
                return LFM_STATUS_STALE;
            }
            return rc;
        }
    }
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
    return 0;
}

void lfm_runtime_request_stop(LfmRuntime *runtime) {
    if (!runtime) return;
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    uint32_t state = runtime->state.load(std::memory_order_acquire);
    while (state < LFM_RUNTIME_STOPPING &&
           !runtime->state.compare_exchange_weak(state, LFM_RUNTIME_STOPPING,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
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
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING ||
        config->capture_slots == 0 || config->capture_slots > MAX_PCM_SLOTS ||
        config->playback_slots == 0 || config->playback_slots > MAX_PCM_SLOTS ||
        config->capture_frames_per_slot == 0 ||
        config->playback_frames_per_slot == 0 || config->pcm_channels != 1 ||
        config->pcm_sample_rate < 8000 || config->pcm_sample_rate > 192000 ||
        config->command_capacity == 0 || config->command_capacity > 64 ||
        config->max_new_tokens == 0 || config->reserved0 != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    bool dock_only = (config->flags & LFM_SESSION_FLAG_DOCK_ONLY) != 0;
    if (!dock_only && config->playback_frames_per_slot < LFM_MIMI_PCM_CAPACITY) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (dock_only && (model || conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (!dock_only && (!model || !conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (callbacks && (callbacks->size != sizeof(*callbacks) ||
                      callbacks->abi_version != LFM_RUNTIME_ABI_VERSION)) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    size_t capture_samples = 0;
    size_t playback_samples = 0;
    if (!checked_samples(config->capture_frames_per_slot, config->pcm_channels,
                         &capture_samples) ||
        capture_samples > UINT32_MAX / sizeof(float) ||
        !checked_samples(config->playback_frames_per_slot, config->pcm_channels,
                         &playback_samples) ||
        playback_samples > UINT32_MAX / sizeof(float)) {
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
    if (!dock_only) {
        int prepare = lfm_conversation_prepare_pcm_native(
            conversation, capture_samples, config->pcm_sample_rate);
        if (prepare != 0) return prepare;
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
    session->channels = config->pcm_channels;
    session->max_new_tokens = config->max_new_tokens;
    if (callbacks) session->callbacks = *callbacks;
    session->events.capacity = runtime->event_capacity;
    session->events.records = new (std::nothrow) EventRecord[runtime->event_capacity];
    session->commands.capacity = config->command_capacity;
    session->commands.records = new (std::nothrow) TextCommand[config->command_capacity];
    int rc = session->events.records && session->commands.records
                 ? 0
                 : LFM_STATUS_OUT_OF_MEMORY;
    if (rc == 0) {
        rc = pool_create(&session->capture, config->capture_slots,
                         static_cast<uint32_t>(capture_samples), LFM_PCM_LEASE_CAPTURE);
    }
    if (rc == 0) {
        rc = pool_create(&session->playback, config->playback_slots,
                         static_cast<uint32_t>(playback_samples), LFM_PCM_LEASE_PLAYBACK);
    }
    if (rc == 0 && (!kc_atomic_u32_is_lock_free(&session->work_doorbell.value) ||
                    kc_port_wait_u32_prepare(&session->work_doorbell.value,
                                             &session->work_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 && (!kc_atomic_u32_is_lock_free(&session->event_doorbell.value) ||
                    kc_port_wait_u32_prepare(&session->event_doorbell.value,
                                             &session->event_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        (!kc_atomic_u32_is_lock_free(&session->event_space_doorbell.value) ||
         kc_port_wait_u32_prepare(&session->event_space_doorbell.value,
                                  &session->event_space_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 && (!kc_atomic_u32_is_lock_free(&session->playback_doorbell.value) ||
                    kc_port_wait_u32_prepare(&session->playback_doorbell.value,
                                             &session->playback_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        (!kc_atomic_u32_is_lock_free(&session->playback_space_doorbell.value) ||
         kc_port_wait_u32_prepare(&session->playback_space_doorbell.value,
                                  &session->playback_space_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        (!kc_atomic_u32_is_lock_free(&session->capture_space_doorbell.value) ||
         kc_port_wait_u32_prepare(&session->capture_space_doorbell.value,
                                  &session->capture_space_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 &&
        (!kc_atomic_u32_is_lock_free(&session->command_space_doorbell.value) ||
         kc_port_wait_u32_prepare(&session->command_space_doorbell.value,
                                  &session->command_space_doorbell.wait) != 0)) {
        rc = LFM_STATUS_INTERNAL;
    }
    if (rc == 0 && !register_session_locked(runtime, session)) {
        rc = LFM_STATUS_BUSY;
    }
    if (rc != 0) {
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
    int rc = kc_port_thread_create(&session->notification, notification_main, session);
    if (rc != 0) {
        session->state.store(LFM_SESSION_CREATED, std::memory_order_release);
        return rc;
    }
    session->notification_started = true;
    rc = kc_port_thread_create(&session->coordinator, coordinator_main, session);
    if (rc != 0) {
        request_stop(session, rc);
        session->event_done.store(true, std::memory_order_release);
        ring(&session->event_doorbell, true);
        session->start_cleanup = true;
        owner.unlock();
        lifecycle.unlock();
        kc_port_thread_join(session->notification);
        lifecycle.lock();
        session->notification = nullptr;
        session->notification_started = false;
        session->threads_joined = true;
        session->start_cleanup = false;
        session->state.store(LFM_SESSION_THREADS_JOINED, std::memory_order_release);
        return rc;
    }
    session->coordinator_started = true;
    return 0;
}

int lfm_session_submit_text(LfmSession *session, const char *utf8,
                            size_t utf8_bytes, LfmTicketIdV1 *out_ticket) {
    return submit_text(session, utf8, utf8_bytes, out_ticket, false);
}

int lfm_session_wait_submit_text(LfmSession *session, const char *utf8,
                                 size_t utf8_bytes,
                                 LfmTicketIdV1 *out_ticket) {
    return submit_text(session, utf8, utf8_bytes, out_ticket, true);
}

int lfm_session_submit_mixed(LfmSession *session, const char *utf8,
                             size_t utf8_bytes,
                             const LfmPcmLeaseV1 *capture,
                             LfmTicketIdV1 *out_ticket) {
    return submit_mixed(session, utf8, utf8_bytes, capture, out_ticket, false);
}

int lfm_session_wait_submit_mixed(LfmSession *session, const char *utf8,
                                  size_t utf8_bytes,
                                  const LfmPcmLeaseV1 *capture,
                                  LfmTicketIdV1 *out_ticket) {
    return submit_mixed(session, utf8, utf8_bytes, capture, out_ticket, true);
}

int lfm_session_interrupt(LfmSession *session, uint64_t *out_epoch) {
    if (!session || !out_epoch) return LFM_STATUS_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        if (session->stop.load(std::memory_order_acquire)) return LFM_STATUS_CANCELLED;
        uint64_t current = session->epoch.load(std::memory_order_relaxed);
        if (current == std::numeric_limits<uint64_t>::max()) return -EOVERFLOW;
        session->epoch.store(current + 1, std::memory_order_release);
        *out_epoch = current + 1;
    }
    ring(&session->work_doorbell, true);
    ring(&session->playback_doorbell, true);
    ring(&session->playback_space_doorbell, true);
    ring(&session->capture_space_doorbell, true);
    ring(&session->command_space_doorbell, true);
    return 0;
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
     * while waiting for them; stop/state already make a later start impossible. */
    if (!session->threads_joined) {
        if (session->coordinator_started) {
            kc_port_thread_join(session->coordinator);
            session->coordinator = nullptr;
            session->coordinator_started = false;
        }
        if (session->notification_started) {
            kc_port_thread_join(session->notification);
            session->notification = nullptr;
            session->notification_started = false;
        }
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        session->threads_joined = true;
        session->state.store(LFM_SESSION_THREADS_JOINED,
                             std::memory_order_release);
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
        .coordinator_parks = session->coordinator_parks.load(std::memory_order_relaxed),
        .coordinator_wakes = session->coordinator_wakes.load(std::memory_order_relaxed),
        .notification_parks = session->notification_parks.load(std::memory_order_relaxed),
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
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_JOINED ||
        pool_live(session->capture) != 0 || pool_live(session->playback) != 0) {
        return LFM_STATUS_BUSY;
    }
    unregister_session(session->runtime, session);
    delete session;
    return 0;
}

int lfm_audio_dock_reserve(LfmSession *session, uint32_t direction,
                           uint32_t frames, uint32_t sample_rate,
                           LfmPcmLeaseV1 *out) {
    if (!session || !out || frames == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (session->stop.load(std::memory_order_acquire)) return LFM_STATUS_CANCELLED;
    uint32_t rate = sample_rate == 0 ? session->sample_rate : sample_rate;
    if (rate < 8000 || rate > 192000) return LFM_STATUS_INVALID_ARGUMENT;
    if (direction == LFM_PCM_LEASE_CAPTURE && rate != session->sample_rate) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmPool *pool = select_pool(session, direction);
    if (!pool) return LFM_STATUS_INVALID_ARGUMENT;
    size_t samples = 0;
    if (!checked_samples(frames, session->channels, &samples) ||
        samples > pool->samples_per_slot) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t start =
        pool->cursor.value.fetch_add(1, std::memory_order_relaxed) % pool->capacity;
    for (uint32_t offset = 0; offset < pool->capacity; ++offset) {
        uint32_t index = (start + offset) % pool->capacity;
        PcmSlot &slot = pool->slots[index];
        uint32_t expected = SLOT_FREE;
        if (!slot.state.compare_exchange_strong(expected, SLOT_RESERVED,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            continue;
        }
        const uint64_t identity = lease_id(direction, index);
        if (identity == 0) {
            slot.state.store(SLOT_RETIRED, std::memory_order_release);
            continue;
        }
        slot.identity.store(identity, std::memory_order_release);
        slot.frames = frames;
        slot.channels = session->channels;
        slot.sample_rate = rate;
        slot.ticket = direction == LFM_PCM_LEASE_CAPTURE
                          ? next_ticket(session, LFM_TICKET_TURN)
                          : LfmTicketIdV1{};
        *out = {
            .size = sizeof(*out),
            .abi_version = LFM_RUNTIME_ABI_VERSION,
            .lease_id = identity,
            .stream_epoch = session->epoch.load(std::memory_order_acquire),
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
    return LFM_STATUS_WOULD_BLOCK;
}

int lfm_audio_dock_wait_reserve(LfmSession *session, uint32_t direction,
                                uint32_t frames, uint32_t sample_rate,
                                LfmPcmLeaseV1 *out) {
    if (!session || !out || frames == 0) return LFM_STATUS_INVALID_ARGUMENT;
    Doorbell *space = direction == LFM_PCM_LEASE_CAPTURE
                          ? &session->capture_space_doorbell
                          : direction == LFM_PCM_LEASE_PLAYBACK
                                ? &session->playback_space_doorbell
                                : nullptr;
    if (!space) return LFM_STATUS_INVALID_ARGUMENT;
    uint64_t admission_epoch = session->epoch.load(std::memory_order_acquire);
    auto reserve = [&]() -> int {
        std::lock_guard<std::mutex> guard(session->publication_mutex);
        if (session->stop.load(std::memory_order_acquire)) {
            return LFM_STATUS_CANCELLED;
        }
        if (session->epoch.load(std::memory_order_acquire) != admission_epoch) {
            return LFM_STATUS_STALE;
        }
        return lfm_audio_dock_reserve(session, direction, frames, sample_rate, out);
    };

    for (;;) {
        int rc = reserve();
        if (rc != LFM_STATUS_WOULD_BLOCK) return rc;

        uint32_t expected = kc_atomic_u32_load_acquire(&space->value);
        rc = reserve();
        if (rc != LFM_STATUS_WOULD_BLOCK) return rc;

        rc = kc_port_wait_u32(space->wait, expected, 0);
        if (rc != 0) {
            if (session->stop.load(std::memory_order_acquire)) {
                return LFM_STATUS_CANCELLED;
            }
            if (session->epoch.load(std::memory_order_acquire) != admission_epoch) {
                return LFM_STATUS_STALE;
            }
            return rc;
        }
    }
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
    *out_samples = slot->samples;
    *out_sample_capacity = pool->samples_per_slot;
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
    const PcmPool *selected = select_pool(session, direction);
    if (!selected) return LFM_STATUS_INVALID_ARGUMENT;
    PcmPool *pool = const_cast<PcmPool *>(selected);
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    if (slot->state.load(std::memory_order_acquire) != SLOT_CONSUMING) {
        return LFM_STATUS_STALE;
    }
    *out_samples = slot->samples;
    *out_sample_count = lease->length_bytes / sizeof(float);
    return 0;
}

int lfm_audio_dock_publish(LfmSession *session, const LfmPcmLeaseV1 *lease) {
    if (!session || !lease || session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
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
    std::unique_lock<std::mutex> publication;
    if (direction == LFM_PCM_LEASE_PLAYBACK) {
        publication = std::unique_lock<std::mutex>(session->publication_mutex);
        if (session->stop.load(std::memory_order_acquire)) return LFM_STATUS_CANCELLED;
        if (lease->stream_epoch != session->epoch.load(std::memory_order_acquire)) {
            return LFM_STATUS_STALE;
        }
    }
    if (direction == LFM_PCM_LEASE_CAPTURE &&
        lease->sample_rate != session->sample_rate) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t expected = SLOT_RESERVED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_PUBLISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    if (!pool_push(pool, *lease)) {
        slot->state.store(SLOT_RESERVED, std::memory_order_release);
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (direction == LFM_PCM_LEASE_CAPTURE) {
        ring(&session->work_doorbell, false);
        return 0;
    }
    session->playback_published.fetch_add(1, std::memory_order_relaxed);
    ring(&session->playback_doorbell, false);
    return 0;
}

int lfm_audio_dock_wait_playback(LfmSession *session, LfmPcmLeaseV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    for (;;) {
        LfmPcmLeaseV1 lease{};
        while (pool_pop(&session->playback, &lease)) {
            PcmSlot *slot = nullptr;
            if (claim_published(&session->playback, &lease, &slot) != 0) continue;
            (void)slot;
            if (lease.stream_epoch != session->epoch.load(std::memory_order_acquire)) {
                lfm_audio_dock_release(session, &lease);
                continue;
            }
            session->playback_consumed.fetch_add(1, std::memory_order_relaxed);
            *out = lease;
            return 0;
        }
        if (session->stop.load(std::memory_order_acquire)) return LFM_STATUS_CANCELLED;
        uint32_t expected = kc_atomic_u32_load_acquire(&session->playback_doorbell.value);
        if (session->playback.head.value.load(std::memory_order_acquire) !=
                session->playback.tail.value.load(std::memory_order_acquire) ||
            session->stop.load(std::memory_order_acquire)) {
            continue;
        }
        int rc = kc_port_wait_u32(session->playback_doorbell.wait, expected, 0);
        if (rc != 0 && !session->stop.load(std::memory_order_acquire)) return rc;
    }
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
    Doorbell *space = direction == LFM_PCM_LEASE_CAPTURE
                          ? &session->capture_space_doorbell
                          : direction == LFM_PCM_LEASE_PLAYBACK
                                ? &session->playback_space_doorbell
                                : nullptr;
    return pool ? release_slot(pool, lease, space, allowed)
                : LFM_STATUS_INVALID_ARGUMENT;
}

} /* extern "C" */
