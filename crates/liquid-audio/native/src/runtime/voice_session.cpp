#include "lfm_audio_dock.h"
#include "lfm_runtime.h"
#include "lfm_session.h"

#include "kc_runtime.h"
#include "kc_deadline.h"
#include "kc_fixed_scope.h"
#include "kc_service.h"
#include "lfm_capture_policy.h"
#include "lfm_capture_format.h"
#include "lfm_mimi.h"
#include "lfm_model_internal.h"
#include "lfm_sesame_detector.h"
#include "../model/lfm_route_epoch.h"

#include <algorithm>
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
void *lfm_engine_new_status(int workers, int *out_status);
void *lfm_internal_engine_new_manual_deadlines_for_test(int workers);
void lfm_engine_request_stop(void *engine);
void lfm_engine_free(void *engine);
}

namespace {

int playback_reserve(LfmSession *session, uint32_t frames,
                     uint32_t sample_rate, LfmPcmLeaseV1 *out);
int playback_resolve_mut(LfmSession *session, const LfmPcmLeaseV1 *lease,
                         float **out_samples, size_t *out_sample_capacity);
int playback_resolve(const LfmSession *session,
                     const LfmPcmLeaseV1 *lease,
                     const float **out_samples, size_t *out_sample_count);
int playback_publish(LfmSession *session, const LfmPcmLeaseV1 *lease);
int playback_release(LfmSession *session, const LfmPcmLeaseV1 *lease);

constexpr uint32_t MAX_RUNTIME_SESSIONS = 64;
constexpr uint32_t MAX_EVENT_CAPACITY = 64;
constexpr uint32_t MAX_PCM_SLOTS = 64;
constexpr uint32_t CAPTURE_IDENTITY_DIRECTION = 1;
constexpr uint32_t CAPTURE_CHUNK_CAPACITY = 512;
constexpr uint32_t CAPTURE_RANGE_CAPACITY = 2;
constexpr uint32_t SLOT_FREE = 0;
constexpr uint32_t SLOT_RESERVED = 1;
constexpr uint32_t SLOT_PUBLISHED = 2;
constexpr uint32_t SLOT_CONSUMING = 3;
constexpr uint32_t SLOT_RELEASING = 4;
constexpr uint32_t SLOT_FINALIZING = 5;
constexpr uint32_t SLOT_RETIRED = 6;
constexpr uint32_t EMISSION_AUDIO_END = 1;
constexpr size_t EVENT_PAYLOAD_CAPACITY = 512;
constexpr uint32_t MAX_KERNEL_LANES = 16;
constexpr uint32_t SESSION_STEP_BUDGET = 16;
constexpr uint32_t ACTION_TRANSITION_BUDGET = 8;
constexpr uint32_t ACTION_CAPTURE_DRAIN_BUDGET = 8;
constexpr uint32_t ACTION_PLAYBACK_EVIDENCE_DRAIN_BUDGET = 16;
constexpr uint32_t PLAYBACK_EVIDENCE_CAPACITY = 512;
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
constexpr uint32_t ACTION_PHASE_TURN_STARTED_PUBLISHED = 11;
constexpr uint32_t COORDINATOR_STARTING = 0;
constexpr uint32_t COORDINATOR_RUNNING = 1;
constexpr uint32_t COORDINATOR_STOPPING = 2;
constexpr uint32_t COORDINATOR_DONE = 3;
constexpr uint32_t CAPTURE_WRITER_IDLE = 0;
constexpr uint32_t CAPTURE_WRITER_ACTIVE = 1;
constexpr uint32_t CAPTURE_RANGE_FREE = 0;
constexpr uint32_t CAPTURE_RANGE_RESERVED = 1;
constexpr uint32_t CAPTURE_RANGE_PUBLISHED = 2;
constexpr uint32_t CAPTURE_RANGE_CONSUMING = 3;
constexpr uint32_t CAPTURE_RANGE_RETIRED = 4;
constexpr uint32_t CAPTURE_POLICY_LISTENING = LFM_CAPTURE_LISTENING;
constexpr uint32_t CAPTURE_POLICY_CANDIDATE = LFM_CAPTURE_CANDIDATE;
constexpr uint32_t CAPTURE_POLICY_SPEAKING = LFM_CAPTURE_SPEAKING;
constexpr uint32_t CAPTURE_POLICY_PAUSE = LFM_CAPTURE_PAUSE;
constexpr uint32_t CAPTURE_DEADLINE_PREPARE = 0;
constexpr uint32_t CAPTURE_DEADLINE_COMMIT = 1;
constexpr uint32_t CAPTURE_DEADLINE_FORCED = 2;
constexpr uint32_t CAPTURE_DEADLINE_COUNT = 3;
constexpr uint64_t CAPTURE_PREPARE_DELAY_NS = UINT64_C(200'000'000);
constexpr uint64_t CAPTURE_COMMIT_DELAY_NS = UINT64_C(500'000'000);
constexpr uint32_t CAPTURE_BARGE_SUSTAIN_MS = 400;
constexpr uint32_t CAPTURE_ECHO_TAIL_MS = 700;
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
    std::atomic<uint32_t> observers{0};
    float *samples = nullptr;
    uint32_t reserved_frames = 0;
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

struct PlaybackPool {
    PcmSlot *slots = nullptr;
    PcmRecordCell *ring = nullptr;
    uint32_t capacity = 0;
    uint32_t samples_per_slot = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
    Cursor<uint32_t> cursor;
};

struct PlaybackEvidenceRecord {
    uint64_t session_id = 0;
    uint64_t stream_epoch = 0;
    LfmTicketIdV1 ticket{};
    uint64_t lease_id = 0;
    uint64_t buffer_generation = 0;
    uint32_t source_offset_frames = 0;
    uint32_t rendered_frames = 0;
    uint64_t first_playback_sample_cursor = 0;
    uint64_t capture_sample_cursor_snapshot = 0;
    uint32_t sample_rate = 0;
    uint32_t flags = 0;
};

struct PlaybackEvidenceRing {
    PlaybackEvidenceRecord *records = nullptr;
    uint32_t capacity = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
};

struct PlaybackEvidenceHistory {
    PlaybackEvidenceRecord *records = nullptr;
    uint32_t capacity = 0;
    uint64_t head = 0;
    uint64_t tail = 0;
};

struct alignas(HOT_ATOMIC_BYTES) PlaybackPolicySnapshotSlot {
    mutable std::atomic<uint32_t> state{0};
    LfmPlaybackPolicySnapshotV1 value{};
};
static_assert(alignof(PlaybackPolicySnapshotSlot) == HOT_ATOMIC_BYTES);

struct PlaybackPolicy {
    LfmSesameDetector *detector = nullptr;
    PlaybackEvidenceRing incoming;
    PlaybackEvidenceHistory history;
    LfmSesameDecisionV1 decision{};
    LfmTicketIdV1 last_ticket{};
    uint64_t last_epoch = 0;
    uint64_t last_capture_cursor = 0;
    uint64_t next_evidence_cursor = 0;
    uint64_t last_evidence_cursor = 0;
    uint64_t available_cursor = 0;
    uint64_t evidence_records = 0;
    uint64_t evidence_updates = 0;
    uint64_t discontinuities = 0;
    uint64_t echo_start_capture_cursor = 0;
    uint64_t last_voice_capture_cursor = 0;
    uint64_t echo_tail_capture_cursor = 0;
    uint64_t echo_epoch = 0;
    LfmTicketIdV1 echo_ticket{};
    uint32_t cadence_remainder = 49;
    std::atomic<uint32_t> published_snapshot{0};
    PlaybackPolicySnapshotSlot snapshots[2]{};
};

struct CaptureChunkRing {
    /* PCM stays in the fixed circular arena. This ring contains identity and
     * absolute bounds only, and its one producer/one consumer cursors are
     * structural. */
    LfmCaptureChunkV1 *records = nullptr;
    uint32_t capacity = 0;
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
};

struct alignas(HOT_ATOMIC_BYTES) CaptureWriter {
    /* Exactly one non-cloneable hardware endpoint may own ACTIVE. The
     * coordinator never takes this gate: committed absolute ranges remain
     * immutable through reader-floor reclamation, so callback and model work
     * can proceed concurrently without rotating storage ownership. */
    std::atomic<uint32_t> gate{CAPTURE_WRITER_IDLE};
    LfmCaptureChunkV1 pending{};
};
static_assert(alignof(CaptureWriter) == HOT_ATOMIC_BYTES);

struct CaptureRangeLease {
    uint64_t lease_id = 0;
    uint64_t buffer_generation = 0;
    uint64_t first_sample_cursor = 0;
    uint64_t stream_epoch = 0;
    LfmTicketIdV1 ticket{};
    uint32_t frames = 0;
    uint32_t sample_rate = 0;
    uint32_t slot = 0;
};

struct alignas(HOT_ATOMIC_BYTES) CaptureRangeSlot {
    std::atomic<uint32_t> state{CAPTURE_RANGE_FREE};
    std::atomic<uint64_t> generation{1};
    std::atomic<uint64_t> identity{0};
    CaptureRangeLease lease{};
};
static_assert(alignof(CaptureRangeSlot) == HOT_ATOMIC_BYTES);

struct CaptureRangeRing {
    CaptureRangeLease records[CAPTURE_RANGE_CAPACITY]{};
    Cursor<uint64_t> head;
    Cursor<uint64_t> tail;
};

struct CaptureArena {
    float *samples = nullptr;
    uint64_t capacity_frames = 0;
    Cursor<uint64_t> reclaim_cursor;
    Cursor<uint32_t> range_cursor;
    CaptureRangeSlot ranges[CAPTURE_RANGE_CAPACITY]{};
    CaptureRangeRing ready;
};

struct CaptureSupervision;

struct CaptureDeadlineRole {
    CaptureSupervision *owner = nullptr;
    uint32_t slot = 0;
    std::atomic<uint64_t> cancel_scope_generation{0};
    std::atomic<uint32_t> cancel_child_generation{0};
    std::atomic<uint32_t> cancel_cause{0};
    kc_scope_child_lease lease{};
    kc_deadline_arm arm{};
    uint64_t domain_generation = 0;
    uint64_t expiry_generation = 0;
    bool grace_armed = false;
    bool terminal = false;
};

constexpr uint32_t CAPTURE_SNAPSHOT_WRITING = UINT32_C(1) << 31;

struct alignas(HOT_ATOMIC_BYTES) CaptureSupervisionSnapshotSlot {
    mutable std::atomic<uint32_t> state{0};
    LfmCaptureSupervisionSnapshotV1 value{};
};
static_assert(alignof(CaptureSupervisionSnapshotSlot) == HOT_ATOMIC_BYTES);

struct CaptureSupervision {
    kc_deadline_source_t *source = nullptr;
    kc_fixed_scope_t *scope = nullptr;
    kc_service_notifier_t *notifier = nullptr;
    CaptureDeadlineRole roles[CAPTURE_DEADLINE_COUNT]{};
    LfmTicketIdV1 parent{};
    LfmTicketIdV1 restart_parent{};
    uint64_t scope_generation = 0;
    uint64_t next_scope_generation = 1;
    uint64_t epoch = 0;
    uint64_t domain = 0;
    uint64_t pause_generation = 0;
    uint64_t commit_cursor = 0;
    uint64_t commit_lease_id = 0;
    bool manual = false;
    bool cycle_active = false;
    bool restart_after_cancel = false;
    bool commit_after_cancel = false;
    bool freeze_pending = false;
    std::atomic<bool> device_loss_pending{false};
    bool device_loss_after_cancel = false;
    bool device_loss_ready = false;
    LfmTicketIdV1 device_loss_parent{};
    uint64_t device_loss_epoch = 0;
    bool stop_requested = false;
    bool source_stop_requested = false;
    std::atomic<uint32_t> published_snapshot{0};
    CaptureSupervisionSnapshotSlot snapshots[2]{};
};

struct CapturePolicy {
    LfmSesameDetector *detector = nullptr;
    LfmCaptureChunkV1 chunk{};
    LfmSesameDecisionV1 decision{};
    LfmTicketIdV1 turn_ticket{};
    uint64_t segment_cursor = 0;
    uint64_t next_evidence_cursor = 0;
    uint64_t last_evidence_cursor = 0;
    uint64_t evidence_updates = 0;
    uint64_t turn_start_cursor = 0;
    uint64_t last_voiced_cursor = 0;
    uint64_t voiced_frames = 0;
    uint64_t silence_frames = 0;
    uint64_t pause_generation = 0;
    uint64_t prepare_sample_generation = 0;
    uint64_t commit_sample_generation = 0;
    uint64_t forced_sample_generation = 0;
    uint64_t prepare_expiry_generation = 0;
    uint64_t commit_expiry_generation = 0;
    uint64_t forced_expiry_generation = 0;
    uint64_t prepare_ready_generation = 0;
    uint64_t commit_ready_generation = 0;
    uint64_t forced_ready_generation = 0;
    uint64_t discarded_silence_frames = 0;
    uint64_t barge_voiced_frames = 0;
    uint64_t barge_candidate_epoch = 0;
    uint64_t barge_source_epoch = 0;
    uint64_t barge_interrupt_epoch = 0;
    uint64_t barge_interrupts = 0;
    LfmTicketIdV1 barge_candidate_ticket{};
    LfmTicketIdV1 barge_playback_ticket{};
    uint64_t segment_epoch = 0;
    uint32_t cadence_remainder = 49;
    uint32_t state = CAPTURE_POLICY_LISTENING;
    bool chunk_pending = false;
    bool turn_active = false;
    bool barge_triggered = false;
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

bool decode_playback_lease_id(uint64_t id, uint32_t *index) {
    const uint64_t nonce = id >> LEASE_NONCE_SHIFT;
    const uint32_t decoded_direction =
        static_cast<uint32_t>((id >> LEASE_DIRECTION_SHIFT) & 3u);
    if (nonce == 0 || decoded_direction != LFM_PCM_LEASE_PLAYBACK) {
        return false;
    }
    *index = static_cast<uint32_t>(id & LEASE_INDEX_MASK);
    return true;
}

bool ticket_equal(const LfmTicketIdV1 &a, const LfmTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

void pool_push(PlaybackPool *pool, const LfmPcmLeaseV1 &lease) {
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

bool pool_peek(const PlaybackPool *pool, LfmPcmLeaseV1 *out,
               uint64_t *out_head) {
    if (!pool || pool->capacity == 0) return false;
    const uint64_t head = pool->head.value.load(std::memory_order_relaxed);
    const PcmRecordCell *cell = &pool->ring[head % pool->capacity];
    if (cell->sequence.load(std::memory_order_acquire) != head * 2 + 1) {
        return false;
    }
    *out = cell->lease;
    *out_head = head;
    return true;
}

void pool_retire_peeked(PlaybackPool *pool, uint64_t head) {
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

uint32_t pool_live(const PlaybackPool &pool) {
    uint32_t live = 0;
    for (uint32_t i = 0; i < pool.capacity; ++i) {
        uint32_t state = pool.slots[i].state.load(std::memory_order_acquire);
        if (state >= SLOT_RESERVED && state <= SLOT_FINALIZING) live++;
    }
    return live;
}

void pool_destroy(PlaybackPool *pool) {
    if (pool->slots) {
        for (uint32_t i = 0; i < pool->capacity; ++i) {
            if (pool->slots[i].observers.load(std::memory_order_acquire) != 0) {
                std::abort();
            }
            delete[] pool->slots[i].samples;
        }
    }
    delete[] pool->slots;
    delete[] pool->ring;
    pool->slots = nullptr;
    pool->ring = nullptr;
}

int pool_create(PlaybackPool *pool, uint32_t capacity,
                uint32_t samples_per_slot) {
    pool->slots = new (std::nothrow) PcmSlot[capacity];
    pool->ring = new (std::nothrow) PcmRecordCell[capacity];
    if (!pool->slots || !pool->ring) return LFM_STATUS_OUT_OF_MEMORY;
    pool->capacity = capacity;
    pool->samples_per_slot = samples_per_slot;
    for (uint32_t i = 0; i < capacity; ++i) {
        pool->ring[i].sequence.store(static_cast<uint64_t>(i) * 2,
                                     std::memory_order_relaxed);
        pool->slots[i].samples = new (std::nothrow) float[samples_per_slot];
        if (!pool->slots[i].samples) return LFM_STATUS_OUT_OF_MEMORY;
    }
    return 0;
}

int pool_slot(PlaybackPool *pool, const LfmPcmLeaseV1 *lease, PcmSlot **out,
              uint32_t *out_index) {
    if (!lease || lease->size != sizeof(*lease) ||
        lease->abi_version != LFM_RUNTIME_ABI_VERSION || lease->reserved != 0 ||
        lease->format != LFM_PCM_FORMAT_F32) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t index = 0;
    if (!decode_playback_lease_id(lease->lease_id, &index) ||
        (lease->flags & LFM_PCM_LEASE_DIRECTION_MASK) !=
            LFM_PCM_LEASE_PLAYBACK ||
        (lease->flags & ~LFM_PCM_LEASE_DIRECTION_MASK) != 0 ||
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
    if (state != SLOT_RESERVED &&
        !ticket_equal(slot->ticket, lease->ticket)) {
        return LFM_STATUS_STALE;
    }
    if (lease->channels != slot->channels ||
        lease->sample_rate != slot->sample_rate || lease->frames == 0 ||
        lease->frames > slot->reserved_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    size_t samples = 0;
    const size_t offset = lease->offset_bytes / sizeof(float);
    if (!checked_samples(lease->frames, lease->channels, &samples) ||
        lease->offset_bytes % sizeof(float) != 0 ||
        offset > pool->samples_per_slot ||
        samples > pool->samples_per_slot - offset ||
        offset != 0 ||
        lease->length_bytes != samples * sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out = slot;
    if (out_index) *out_index = index;
    return 0;
}

bool finalize_slot(PcmSlot *slot, std::atomic<uint64_t> *consumed) {
    if (slot->observers.load(std::memory_order_acquire) != 0) return false;
    uint32_t expected = SLOT_RELEASING;
    if (!slot->state.compare_exchange_strong(
            expected, SLOT_FINALIZING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return false;
    }
    slot->reserved_frames = 0;
    slot->channels = 0;
    slot->sample_rate = 0;
    slot->stream_epoch = 0;
    slot->ticket = {};
    slot->identity.store(0, std::memory_order_relaxed);
    const uint64_t generation =
        slot->generation.load(std::memory_order_relaxed);
    if (consumed) consumed->fetch_add(1, std::memory_order_relaxed);
    if (generation == std::numeric_limits<uint64_t>::max()) {
        slot->state.store(SLOT_RETIRED, std::memory_order_release);
        return true;
    }
    slot->generation.store(generation + 1, std::memory_order_relaxed);
    slot->state.store(SLOT_FREE, std::memory_order_release);
    return true;
}

void retire_slot_observer(PcmSlot *slot,
                          std::atomic<uint64_t> *consumed) {
    const uint32_t prior =
        slot->observers.fetch_sub(1, std::memory_order_acq_rel);
    if (prior == 0) std::abort();
    if (prior == 1) (void)finalize_slot(slot, consumed);
}

int release_slot(PlaybackPool *pool, const LfmPcmLeaseV1 *lease,
                 std::atomic<uint64_t> *consumed,
                 uint32_t allowed_states = UINT32_MAX) {
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    uint32_t state = slot->state.load(std::memory_order_acquire);
    if (state == SLOT_FREE || state == SLOT_RELEASING ||
        state == SLOT_FINALIZING || state == SLOT_RETIRED) {
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
    /* `playback_consumed` is the retirement side of
     * `playback_published`; an unused reservation never entered that
     * accounting domain. Generation can reserve one final Mimi buffer and
     * release it when no PCM remains, so counting SLOT_RESERVED here makes a
     * clean turn report consumed > published. */
    const bool published = state == SLOT_PUBLISHED || state == SLOT_CONSUMING;
    (void)finalize_slot(slot, published ? consumed : nullptr);
    return 0;
}

int claim_published(PlaybackPool *pool, const LfmPcmLeaseV1 *lease,
                    PcmSlot **out) {
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
    std::atomic<uint64_t> ticket_sequence{1};
    uint32_t kernel_lanes = 0;
    uint32_t event_capacity = 0;
    uint32_t session_capacity = 0;
    bool manual_deadlines = false;
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
    CaptureRangeLease capture_range{};
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
    bool capture_range_active = false;
    bool announce_start = false;
    bool turn_started_required = false;
    bool turn_started_visible = false;
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
    uint32_t capture_rate = 0;
    uint32_t capture_callback_frames = 0;
    uint32_t capture_turn_frames = 0;
    uint32_t playback_rate = 0;
    uint32_t playback_frames = 0;
    uint32_t channels = 0;
    uint32_t max_new_tokens = 0;
    uint32_t generation = 1;
    bool dock_only = false;
    bool manual_deadlines = false;
    std::atomic<uint32_t> state{LFM_SESSION_CREATED};
    LfmRouteEpoch epoch{};
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
    std::atomic<uint64_t> capture_evidence_updates{0};
    std::atomic<uint64_t> capture_evidence_cursor{0};
    std::atomic<uint64_t> playback_evidence_records{0};
    std::atomic<uint64_t> playback_evidence_updates{0};
    std::atomic<uint64_t> playback_evidence_cursor{0};
    std::atomic<uint64_t> playback_sample_cursor{0};
    std::atomic<uint32_t> playback_evidence_voice{0};
    std::atomic<uint32_t> playback_evidence_backlog{0};
    std::atomic<uint32_t> playback_retained_observers{0};
    PlaybackPool playback;
    PlaybackPolicy playback_policy;
    CaptureArena capture_arena;
    CaptureChunkRing capture_chunks;
    CapturePolicy capture_policy;
    CaptureSupervision capture_supervision;
    std::atomic<LfmCaptureProducer *> chunk_producer{nullptr};
    LfmCaptureProducer *retired_chunk_producer = nullptr;
    EventRing events;
    TextRing commands;
    SessionAction action;
    ResultRecord result;
    TextCommand pending_command{};
    CaptureRangeLease pending_range{};
    bool command_pending = false;
    bool range_pending = false;
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
    std::atomic<uint32_t> capture_producers{0};
    std::atomic<uint32_t> playback_consumers{0};
    uint32_t control_handles = 0;
    /* Lock order is runtime.children_mutex -> lifecycle_mutex. join_mutex is
     * outermost only for concurrent join callers and is never acquired by
     * start or stop. No retained-service join holds lifecycle_mutex. */
    mutable std::mutex lifecycle_mutex;
    mutable std::condition_variable lifecycle_cv;
    mutable std::mutex join_mutex;

    ~LfmSession() {
        if (capture_policy.detector) {
            (void)lfm_sesame_detector_destroy(capture_policy.detector);
        }
        if (playback_policy.detector) {
            (void)lfm_sesame_detector_destroy(playback_policy.detector);
        }
        pool_destroy(&playback);
        delete[] playback_policy.incoming.records;
        delete[] playback_policy.history.records;
        delete[] capture_arena.samples;
        delete[] capture_chunks.records;
        delete[] events.records;
        delete[] commands.ring;
    }
};

struct LfmCaptureProducer {
    LfmSession *session = nullptr;
    uint64_t stream = 0;
    uint32_t lane = 0;
    uint32_t sample_rate = 0;
    CaptureWriter writer{};
    /* Runtime/session/kind are immutable, so the sequence is the only
     * cross-owner coordinate needed to publish the next turn identity. The
     * coordinator rotates it at an exact committed boundary; the hardware
     * producer reads it once when stamping a callback record. */
    std::atomic<uint64_t> transport_sequence{0};
    std::atomic<uint64_t> transport_epoch{0};
    std::atomic<bool> closing{false};
    /* A failed XRUN publication is durable producer state, never an omitted
     * record. The sole hardware producer adds each dropped callback block
     * exactly once. Later PCM admission remains closed until one sequenced GAP
     * record pays the complete debt. */
    std::atomic<uint64_t> gap_debt_frames{0};
    std::atomic<uint32_t> gap_debt_channels{0};
    std::atomic<uint32_t> gap_debt_flags{0};
    uint64_t chunk_sequence = 1;
    std::atomic<uint64_t> sample_cursor{0};
};

struct LfmPlaybackConsumer {
    LfmSession *session = nullptr;
    LfmPcmLeaseV1 lease{};
    LfmPcmLeaseV1 lineage{};
    uint64_t sample_cursor = 0;
    bool active = false;
    bool faulted = false;
};

struct LfmSessionControl {
    LfmSession *session = nullptr;
};

namespace {

bool capture_chunk_push(CaptureChunkRing *ring,
                        const LfmCaptureChunkV1 &chunk) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == ring->capacity) return false;
    ring->records[tail % ring->capacity] = chunk;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool capture_chunk_pop(CaptureChunkRing *ring, LfmCaptureChunkV1 *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->records[head % ring->capacity];
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

bool capture_chunk_has_space(const CaptureChunkRing &ring) {
    const uint64_t tail = ring.tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring.head.value.load(std::memory_order_acquire);
    return tail - head < ring.capacity;
}

bool capture_chunk_empty(const CaptureChunkRing &ring) {
    return ring.head.value.load(std::memory_order_acquire) ==
           ring.tail.value.load(std::memory_order_acquire);
}

bool capture_range_push(CaptureRangeRing *ring,
                        const CaptureRangeLease &lease) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == CAPTURE_RANGE_CAPACITY) return false;
    ring->records[tail % CAPTURE_RANGE_CAPACITY] = lease;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool capture_range_pop(CaptureRangeRing *ring, CaptureRangeLease *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->records[head % CAPTURE_RANGE_CAPACITY];
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

bool capture_range_empty(const CaptureRangeRing &ring) {
    return ring.head.value.load(std::memory_order_acquire) ==
           ring.tail.value.load(std::memory_order_acquire);
}

uint32_t capture_range_live(const CaptureArena &arena) {
    uint32_t live = 0;
    for (const CaptureRangeSlot &slot : arena.ranges) {
        const uint32_t state = slot.state.load(std::memory_order_acquire);
        if (state == CAPTURE_RANGE_RESERVED ||
            state == CAPTURE_RANGE_PUBLISHED ||
            state == CAPTURE_RANGE_CONSUMING) {
            ++live;
        }
    }
    return live;
}

int capture_arena_spans(const CaptureArena &arena, uint64_t start,
                        uint32_t frames, LfmF32Span out[2],
                        uint32_t *out_count) {
    if (!arena.samples || arena.capacity_frames == 0 || frames == 0 || !out ||
        !out_count || frames > arena.capacity_frames ||
        start > UINT64_MAX - frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t offset = start % arena.capacity_frames;
    const uint64_t first = std::min<uint64_t>(
        frames, arena.capacity_frames - offset);
    out[0] = {
        .data = arena.samples + offset,
        .length = first,
    };
    const uint64_t second = static_cast<uint64_t>(frames) - first;
    out[1] = second == 0
        ? LfmF32Span{}
        : LfmF32Span{.data = arena.samples, .length = second};
    *out_count = second == 0 ? 1u : 2u;
    return 0;
}

int capture_arena_mutable_spans(CaptureArena &arena, uint64_t start,
                                uint32_t frames,
                                LfmMutableF32SpanV1 out[2],
                                uint32_t *out_count) {
    LfmF32Span spans[2]{};
    const int status = capture_arena_spans(
        arena, start, frames, spans, out_count);
    if (status != 0) return status;
    out[0] = {
        .data = const_cast<float *>(spans[0].data),
        .count = static_cast<size_t>(spans[0].length),
    };
    out[1] = *out_count == 2
        ? LfmMutableF32SpanV1{
              .data = const_cast<float *>(spans[1].data),
              .count = static_cast<size_t>(spans[1].length),
          }
        : LfmMutableF32SpanV1{};
    return 0;
}

bool chunk_equal(const LfmCaptureChunkV1 &a,
                 const LfmCaptureChunkV1 &b) {
    return a.stream == b.stream && a.lane == b.lane &&
           a.flags == b.flags && a.chunk_sequence == b.chunk_sequence &&
           a.first_sample_cursor == b.first_sample_cursor &&
           a.stream_epoch == b.stream_epoch &&
           ticket_equal(a.turn_ticket, b.turn_ticket) &&
           a.lease_id == b.lease_id &&
           a.buffer_generation == b.buffer_generation &&
           a.offset_frames == b.offset_frames && a.frames == b.frames &&
           a.channels == b.channels && a.sample_rate == b.sample_rate;
}

bool valid_chunk(const LfmCaptureChunkV1 *chunk) {
    if (!chunk || chunk->size != sizeof(*chunk) ||
        chunk->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return false;
    }
    return chunk->reserved[0] == 0 && chunk->reserved[1] == 0;
}

void request_stop(LfmSession *session, int32_t status);
void notify_session(LfmSession *session);
bool enter_publication(LfmSession *session);
void leave_publication(LfmSession *session);
int release_capture_range(LfmSession *session,
                          const CaptureRangeLease &lease);

int add_gap_debt(LfmCaptureProducer *producer, uint32_t frames,
                 uint32_t channels, uint32_t flags, uint32_t *out_total,
                 uint32_t *out_channels, uint32_t *out_flags) {
    const uint64_t debt =
        producer->gap_debt_frames.load(std::memory_order_acquire);
    const uint32_t debt_channels =
        producer->gap_debt_channels.load(std::memory_order_acquire);
    const uint32_t debt_flags =
        producer->gap_debt_flags.load(std::memory_order_acquire);
    if (debt != 0 && debt_channels != channels) {
        request_stop(producer->session, LFM_STATUS_INTERNAL);
        return LFM_STATUS_INTERNAL;
    }
    if (debt > static_cast<uint64_t>(UINT32_MAX) - frames) {
        producer->gap_debt_frames.store(UINT32_MAX,
                                        std::memory_order_release);
        request_stop(producer->session, -EOVERFLOW);
        return -EOVERFLOW;
    }
    const uint32_t total = static_cast<uint32_t>(debt + frames);
    if (debt == 0) {
        producer->gap_debt_channels.store(channels,
                                          std::memory_order_relaxed);
    }
    producer->gap_debt_flags.store(debt_flags | flags,
                                   std::memory_order_relaxed);
    producer->gap_debt_frames.store(total, std::memory_order_release);
    *out_total = total;
    *out_channels = debt == 0 ? channels : debt_channels;
    *out_flags = debt_flags | flags;
    return 0;
}

void capture_write_result(LfmCaptureWriteV1 *out, uint32_t admitted,
                          uint32_t dropped, uint32_t flags, int32_t status) {
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .admitted_frames = admitted,
        .dropped_frames = dropped,
        .flags = flags,
        .status = status,
        .reserved = {},
    };
}

int capture_write_drop(LfmCaptureProducer *producer, uint32_t frames,
                       uint32_t channels, int32_t status,
                       LfmCaptureWriteV1 *out) {
    LfmCaptureChunkV1 gap{};
    const int published = frames == 0
        ? LFM_STATUS_INVALID_ARGUMENT
        : lfm_capture_producer_publish_gap(
              producer, frames, channels,
              LFM_CAPTURE_CHUNK_GAP | LFM_CAPTURE_CHUNK_XRUN, &gap);
    capture_write_result(
        out, 0, frames,
        published == 0 ? LFM_CAPTURE_WRITE_GAP_PUBLISHED : 0, status);
    return 0;
}

bool consumer_matches(const LfmPlaybackConsumer *consumer,
                      const LfmPcmLeaseV1 *lease) {
    return consumer && consumer->active && lease && consumer->session &&
           consumer->lease.lease_id == lease->lease_id &&
           consumer->lease.buffer_generation == lease->buffer_generation &&
           consumer->lease.stream_epoch == lease->stream_epoch &&
           ticket_equal(consumer->lease.ticket, lease->ticket);
}

bool playback_evidence_empty(const PlaybackEvidenceRing &ring) {
    return ring.head.value.load(std::memory_order_acquire) ==
           ring.tail.value.load(std::memory_order_acquire);
}

bool playback_evidence_push(PlaybackEvidenceRing *ring,
                            const PlaybackEvidenceRecord &record) {
    const uint64_t tail = ring->tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring->head.value.load(std::memory_order_acquire);
    if (tail - head == ring->capacity) return false;
    ring->records[tail % ring->capacity] = record;
    ring->tail.value.store(tail + 1, std::memory_order_release);
    return true;
}

bool playback_evidence_pop(PlaybackEvidenceRing *ring,
                           PlaybackEvidenceRecord *out) {
    const uint64_t head = ring->head.value.load(std::memory_order_relaxed);
    const uint64_t tail = ring->tail.value.load(std::memory_order_acquire);
    if (head == tail) return false;
    *out = ring->records[head % ring->capacity];
    ring->head.value.store(head + 1, std::memory_order_release);
    return true;
}

uint64_t playback_capture_cursor_snapshot(const LfmSession *session) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    return producer
        ? producer->sample_cursor.load(std::memory_order_acquire)
        : session->capture_evidence_cursor.load(std::memory_order_acquire);
}

void fill_playback_render(const PlaybackEvidenceRecord &record,
                          LfmPlaybackRenderV1 *out) {
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .session_id = record.session_id,
        .stream_epoch = record.stream_epoch,
        .ticket = record.ticket,
        .lease_id = record.lease_id,
        .buffer_generation = record.buffer_generation,
        .source_offset_frames = record.source_offset_frames,
        .rendered_frames = record.rendered_frames,
        .first_playback_sample_cursor =
            record.first_playback_sample_cursor,
        .capture_sample_cursor_snapshot =
            record.capture_sample_cursor_snapshot,
        .flags = record.flags,
        .reserved0 = 0,
        .reserved = {},
    };
}

int publish_playback_evidence(LfmPlaybackConsumer *consumer,
                              const LfmPcmLeaseV1 *lease,
                              uint32_t source_offset_frames, uint32_t frames,
                              uint32_t flags, LfmPlaybackRenderV1 *out) {
    if (!consumer || !consumer->session || !out) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (consumer->faulted ||
        consumer->session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    constexpr uint32_t supported = LFM_PLAYBACK_EVIDENCE_RENDERED |
                                   LFM_PLAYBACK_EVIDENCE_SILENCE |
                                   LFM_PLAYBACK_EVIDENCE_FLUSH |
                                   LFM_PLAYBACK_EVIDENCE_DISCONTINUITY;
    if (flags == 0 || (flags & ~supported) != 0 ||
        ((flags & LFM_PLAYBACK_EVIDENCE_RENDERED) != 0 &&
         ((flags & ~LFM_PLAYBACK_EVIDENCE_RENDERED) != 0 || frames == 0)) ||
        ((flags & LFM_PLAYBACK_EVIDENCE_SILENCE) != 0 &&
         ((flags & ~LFM_PLAYBACK_EVIDENCE_SILENCE) != 0 || frames == 0)) ||
        ((flags & (LFM_PLAYBACK_EVIDENCE_FLUSH |
                   LFM_PLAYBACK_EVIDENCE_DISCONTINUITY)) != 0 &&
         frames != 0)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const bool rendered = (flags & LFM_PLAYBACK_EVIDENCE_RENDERED) != 0;
    const LfmPcmLeaseV1 *lineage = lease;
    if (!lineage) {
        lineage = consumer->lineage.lease_id == 0 ? nullptr
                                                  : &consumer->lineage;
    }
    if (!lineage || !ticket_equal(lineage->ticket,
                                  consumer->lineage.ticket)) {
        return LFM_STATUS_STALE;
    }
    LfmSession *session = consumer->session;
    if (!enter_publication(session)) return LFM_STATUS_CANCELLED;
    const auto finish = [session](int status) {
        leave_publication(session);
        return status;
    };
    const uint64_t epoch = session->epoch.load(std::memory_order_acquire);
    if (lineage->stream_epoch != epoch) return finish(LFM_STATUS_STALE);
    if (frames != 0 && consumer->sample_cursor > UINT64_MAX - frames) {
        request_stop(session, -EOVERFLOW);
        return finish(-EOVERFLOW);
    }
    PlaybackEvidenceRing &ring = session->playback_policy.incoming;
    const uint64_t tail = ring.tail.value.load(std::memory_order_relaxed);
    const uint64_t head = ring.head.value.load(std::memory_order_acquire);
    if (tail - head == ring.capacity) {
        request_stop(session, LFM_STATUS_INTERNAL);
        return finish(LFM_STATUS_INTERNAL);
    }

    PcmSlot *slot = nullptr;
    if (rendered) {
        if (!consumer_matches(consumer, lineage) ||
            source_offset_frames > lineage->frames ||
            frames > lineage->frames - source_offset_frames) {
            return finish(LFM_STATUS_STALE);
        }
        const int resolved = pool_slot(&session->playback, lineage, &slot,
                                       nullptr);
        if (resolved != 0 ||
            slot->state.load(std::memory_order_acquire) != SLOT_CONSUMING) {
            return finish(resolved == 0 ? LFM_STATUS_STALE : resolved);
        }
        const uint32_t prior =
            slot->observers.fetch_add(1, std::memory_order_acq_rel);
        if (prior == UINT32_MAX) std::abort();
        session->playback_retained_observers.fetch_add(
            1, std::memory_order_release);
    }

    const PlaybackEvidenceRecord record = {
        .session_id = session->id,
        .stream_epoch = epoch,
        .ticket = lineage->ticket,
        .lease_id = rendered ? lineage->lease_id : 0,
        .buffer_generation = rendered ? lineage->buffer_generation : 0,
        .source_offset_frames = rendered ? source_offset_frames : 0,
        .rendered_frames = frames,
        .first_playback_sample_cursor = consumer->sample_cursor,
        .capture_sample_cursor_snapshot =
            playback_capture_cursor_snapshot(session),
        .sample_rate = session->playback_rate,
        .flags = flags,
    };
    if (!playback_evidence_push(&ring, record)) std::abort();
    consumer->sample_cursor += frames;
    session->playback_sample_cursor.store(consumer->sample_cursor,
                                          std::memory_order_release);
    fill_playback_render(record, out);
    notify_session(session);
    return finish(0);
}

using PlaybackFanout = int (*)(const float *, void *, size_t, uint32_t,
                               size_t);

int render_playback_evidence(LfmPlaybackConsumer *consumer,
                             const LfmPcmLeaseV1 *lease,
                             uint32_t source_offset_frames, void *destination,
                             uint32_t frames, uint32_t channels,
                             size_t destination_capacity,
                             PlaybackFanout fanout,
                             LfmPlaybackRenderV1 *out) {
    if (!consumer || !consumer->session || !lease || !destination ||
        !fanout || !out || frames == 0 || channels == 0 ||
        source_offset_frames > lease->frames ||
        frames > lease->frames - source_offset_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (consumer->faulted ||
        consumer->session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (!consumer_matches(consumer, lease)) return LFM_STATUS_STALE;
    const float *samples = nullptr;
    size_t count = 0;
    const int resolved = playback_resolve(consumer->session, lease, &samples,
                                          &count);
    if (resolved != 0) return resolved;
    if (!samples || count != lease->frames) return LFM_STATUS_INTERNAL;
    const int rendered = fanout(samples + source_offset_frames, destination,
                                frames, channels, destination_capacity);
    if (rendered != 0) return rendered;
    const int published = publish_playback_evidence(
        consumer, lease, source_offset_frames, frames,
        LFM_PLAYBACK_EVIDENCE_RENDERED, out);
    if (published == 0) return 0;
    /* The device prefix is already visible. It cannot be replayed after its
     * correlated evidence edge loses an epoch/stop/capacity race. Poison this
     * endpoint and make the session failure explicit. */
    consumer->faulted = true;
    if (consumer->sample_cursor <= UINT64_MAX - frames) {
        consumer->sample_cursor += frames;
        consumer->session->playback_sample_cursor.store(
            consumer->sample_cursor, std::memory_order_release);
    }
    request_stop(consumer->session, LFM_STATUS_HOST_SINK);
    return published;
}

int fanout_f32_erased(const float *source, void *destination, size_t frames,
                      uint32_t channels, size_t capacity) {
    return lfm_playback_fanout_f32(source, static_cast<float *>(destination),
                                   frames, channels, capacity);
}

int fanout_i16_erased(const float *source, void *destination, size_t frames,
                      uint32_t channels, size_t capacity) {
    return lfm_playback_fanout_i16(
        source, static_cast<int16_t *>(destination), frames, channels,
        capacity);
}

int fanout_u16_erased(const float *source, void *destination, size_t frames,
                      uint32_t channels, size_t capacity) {
    return lfm_playback_fanout_u16(
        source, static_cast<uint16_t *>(destination), frames, channels,
        capacity);
}

LfmTicketIdV1 next_ticket(LfmSession *session, uint32_t kind) {
    const uint64_t sequence = session->runtime->ticket_sequence.fetch_add(
        1, std::memory_order_relaxed);
    if (sequence == 0) std::abort();
    return {
        .runtime_epoch = session->runtime->epoch,
        .sequence = sequence,
        .generation = session->generation,
        .kind = kind,
    };
}

LfmTicketIdV1 capture_ticket_from_sequence(
    const LfmCaptureProducer *producer, uint64_t sequence) {
    if (!producer || !producer->session || sequence == 0) std::abort();
    return {
        .runtime_epoch = producer->session->runtime->epoch,
        .sequence = sequence,
        .generation = producer->session->generation,
        .kind = LFM_TICKET_TURN,
    };
}

LfmTicketIdV1 rotate_capture_ticket(LfmCaptureProducer *producer,
                                    uint64_t epoch) {
    if (!producer || !producer->session || epoch == 0) std::abort();
    const LfmTicketIdV1 ticket =
        next_ticket(producer->session, LFM_TICKET_TURN);
    producer->transport_epoch.store(epoch, std::memory_order_relaxed);
    producer->transport_sequence.store(ticket.sequence,
                                       std::memory_order_release);
    return ticket;
}

LfmTicketIdV1 current_capture_ticket(LfmCaptureProducer *producer,
                                     uint64_t epoch) {
    const uint64_t sequence = producer->transport_sequence.load(
        std::memory_order_acquire);
    const uint64_t ticket_epoch = producer->transport_epoch.load(
        std::memory_order_relaxed);
    if (sequence == 0 || ticket_epoch != epoch) {
        return rotate_capture_ticket(producer, epoch);
    }
    return capture_ticket_from_sequence(producer, sequence);
}

int prepare_reservation(LfmSession *session, uint32_t frames,
                        uint32_t sample_rate, PlaybackPool **out_pool,
                        uint32_t *out_rate,
                        size_t *out_samples) {
    if (!session || frames == 0 || !out_pool || !out_rate || !out_samples) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    const uint32_t configured = session->playback_rate;
    const uint32_t rate = sample_rate == 0 ? configured : sample_rate;
    if (rate < 8000 || rate > 192000 || rate != configured) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PlaybackPool *pool = &session->playback;
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

int reserve_slot_at(LfmSession *session, PlaybackPool *pool,
                    uint32_t frames, uint32_t rate, size_t samples,
                    uint32_t index, LfmPcmLeaseV1 *out) {
    PcmSlot &slot = pool->slots[index];
    uint32_t expected = SLOT_FREE;
    if (!slot.state.compare_exchange_strong(expected, SLOT_RESERVED,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (slot.observers.load(std::memory_order_acquire) != 0) std::abort();
    const uint64_t identity = lease_id(LFM_PCM_LEASE_PLAYBACK, index);
    if (identity == 0) {
        slot.state.store(SLOT_RETIRED, std::memory_order_release);
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    slot.identity.store(identity, std::memory_order_release);
    slot.reserved_frames = frames;
    slot.channels = session->channels;
    slot.sample_rate = rate;
    slot.stream_epoch = session->epoch.load(std::memory_order_acquire);
    slot.ticket = {};
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
        .flags = LFM_PCM_LEASE_PLAYBACK,
        .reserved = 0,
    };
    return 0;
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

void reset_capture_policy(LfmSession *session, uint64_t cursor,
                          bool reset_detector);

void capture_supervision_notify(void *context) {
    auto *notifier = static_cast<kc_service_notifier_t *>(context);
    if (notifier) (void)kc_service_notifier_notify(notifier);
}

void capture_scope_ready(void *context, uint64_t, uint32_t) {
    capture_supervision_notify(context);
}

void capture_scope_cancel(void *context,
                          const kc_scope_child_lease *lease,
                          uint32_t cause) {
    auto *role = static_cast<CaptureDeadlineRole *>(context);
    if (!role || !lease || !role->owner) return;
    role->cancel_scope_generation.store(
        lease->scope_generation, std::memory_order_relaxed);
    role->cancel_child_generation.store(
        lease->child_generation, std::memory_order_relaxed);
    role->cancel_cause.store(cause, std::memory_order_release);
    capture_supervision_notify(role->owner->notifier);
}

void publish_capture_supervision_snapshot(LfmSession *session);

int capture_supervision_create(LfmSession *session) {
    CaptureSupervision &supervision = session->capture_supervision;
    if (supervision.scope || supervision.source) return LFM_STATUS_BUSY;
    supervision.notifier = session->coordinator_notifier;
    supervision.manual = session->manual_deadlines;
    for (uint32_t slot = 0; slot < CAPTURE_DEADLINE_COUNT; ++slot) {
        supervision.roles[slot].owner = &supervision;
        supervision.roles[slot].slot = slot;
    }

    const kc_fixed_scope_config scope_config = {
        .size = sizeof(kc_fixed_scope_config),
        .abi_version = KC_ABI_VERSION,
        .child_capacity = CAPTURE_DEADLINE_COUNT,
        .reserved = 0,
        .ready = capture_scope_ready,
        .context = supervision.notifier,
    };
    int status = kc_fixed_scope_create(&scope_config, &supervision.scope);
    if (status != 0) return status;
    for (uint32_t slot = 0; slot < CAPTURE_DEADLINE_COUNT; ++slot) {
        const kc_scope_child_config child = {
            .size = sizeof(kc_scope_child_config),
            .abi_version = KC_ABI_VERSION,
            .child_class = KC_SCOPE_CHILD_FUNCTIONAL,
            .reserved = 0,
            .cancel = capture_scope_cancel,
            .context = &supervision.roles[slot],
        };
        uint32_t added = UINT32_MAX;
        status = kc_fixed_scope_add_role(supervision.scope, &child, &added);
        if (status != 0 || added != slot) {
            (void)kc_fixed_scope_destroy(supervision.scope);
            supervision.scope = nullptr;
            return status != 0 ? status : LFM_STATUS_INTERNAL;
        }
    }
    status = kc_fixed_scope_seal(supervision.scope);
    if (status != 0) {
        (void)kc_fixed_scope_destroy(supervision.scope);
        supervision.scope = nullptr;
        return status;
    }

    const kc_deadline_source_config source_config = {
        .size = sizeof(kc_deadline_source_config),
        .abi_version = KC_ABI_VERSION,
        .capacity = CAPTURE_DEADLINE_COUNT,
        .reserved = 0,
        .notify = capture_supervision_notify,
        .context = supervision.notifier,
    };
    status = supervision.manual
                 ? kc_deadline_source_create_manual_test(
                       &source_config, &supervision.source)
                 : kc_deadline_source_create(
                       &source_config, &supervision.source);
    if (status != 0) {
        (void)kc_fixed_scope_destroy(supervision.scope);
        supervision.scope = nullptr;
        return status;
    }
    publish_capture_supervision_snapshot(session);
    return 0;
}

int build_capture_supervision_snapshot(
    const LfmSession *session, LfmCaptureSupervisionSnapshotV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    const CaptureSupervision &supervision = session->capture_supervision;
    kc_fixed_scope_snapshot scope = {
        .size = sizeof(kc_fixed_scope_snapshot),
        .abi_version = KC_ABI_VERSION,
    };
    kc_deadline_source_snapshot source = {
        .size = sizeof(kc_deadline_source_snapshot),
        .abi_version = KC_ABI_VERSION,
    };
    if (supervision.scope) {
        const int status = kc_fixed_scope_snapshot_get(
            supervision.scope, &scope);
        if (status != 0) return status;
    }
    if (supervision.source) {
        const int status = kc_deadline_source_snapshot_get(
            supervision.source, &source);
        if (status != 0) return status;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .cycle_active = supervision.cycle_active ? 1u : 0u,
        .scope_phase = scope.phase,
        .source_phase = source.phase,
        .source_pending_events = source.pending_events,
        .policy_state = session->capture_policy.state,
        .reserved0 = 0,
        .scope_generation = supervision.scope_generation,
        .epoch = supervision.epoch,
        .domain = supervision.domain,
        .pause_generation = session->capture_policy.pause_generation,
        .prepare_ready_generation =
            session->capture_policy.prepare_ready_generation,
        .commit_ready_generation =
            session->capture_policy.commit_ready_generation,
        .forced_ready_generation =
            session->capture_policy.forced_ready_generation,
        .prepare_sample_generation =
            session->capture_policy.prepare_sample_generation,
        .commit_sample_generation =
            session->capture_policy.commit_sample_generation,
        .forced_sample_generation =
            session->capture_policy.forced_sample_generation,
        .turn_start_cursor = session->capture_policy.turn_start_cursor,
        .last_evidence_cursor =
            session->capture_policy.last_evidence_cursor,
        .silence_frames = session->capture_policy.silence_frames,
        .parent = supervision.parent,
        .slots = {},
    };
    for (uint32_t slot = 0; slot < CAPTURE_DEADLINE_COUNT; ++slot) {
        const CaptureDeadlineRole &role = supervision.roles[slot];
        out->slots[slot] = {
            .slot = slot,
            .armed = role.arm.arm_generation != 0 ? 1u : 0u,
            .terminal = role.terminal ? 1u : 0u,
            .cancel_cause = role.cancel_cause.load(
                std::memory_order_acquire),
            .arm_generation = role.arm.arm_generation,
            .expiry_generation = role.expiry_generation,
            .scope_generation = role.lease.scope_generation,
            .epoch = supervision.epoch,
            .domain = supervision.domain,
            .pause_generation = role.domain_generation,
            .child = role.lease.child,
            .parent = role.lease.parent,
        };
    }
    return 0;
}

void publish_capture_supervision_snapshot(LfmSession *session) {
    if (!session) return;
    CaptureSupervision &supervision = session->capture_supervision;
    const uint32_t current = supervision.published_snapshot.load(
        std::memory_order_relaxed);
    const uint32_t target = current ^ 1u;
    CaptureSupervisionSnapshotSlot &slot = supervision.snapshots[target];
    uint32_t expected = 0;
    if (!slot.state.compare_exchange_strong(
            expected, CAPTURE_SNAPSHOT_WRITING,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        /* Diagnostics may retain the inactive slot briefly. Progress never
         * waits for a reader; the last coherent publication remains valid. */
        return;
    }
    LfmCaptureSupervisionSnapshotV1 snapshot{};
    const int status = build_capture_supervision_snapshot(
        session, &snapshot);
    if (status == 0) slot.value = snapshot;
    slot.state.store(0, std::memory_order_release);
    if (status == 0) {
        supervision.published_snapshot.store(target,
                                              std::memory_order_release);
    }
}

int read_capture_supervision_snapshot(
    const LfmSession *session, LfmCaptureSupervisionSnapshotV1 *out) {
    const CaptureSupervision &supervision = session->capture_supervision;
    /* There are exactly two immutable snapshot generations. A writer may
     * rotate after the first index load and win the inactive-slot claim just
     * before this reader. Re-read the newly published slot once instead of
     * exposing that completed rotation as artificial backpressure. This is a
     * bounded two-generation transaction, not a progress wait. */
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        const uint32_t index = supervision.published_snapshot.load(
            std::memory_order_acquire);
        const CaptureSupervisionSnapshotSlot &slot =
            supervision.snapshots[index];
        uint32_t readers = slot.state.load(std::memory_order_acquire);
        if ((readers & CAPTURE_SNAPSHOT_WRITING) != 0 ||
            readers == CAPTURE_SNAPSHOT_WRITING - 1) {
            continue;
        }
        if (!slot.state.compare_exchange_strong(
                readers, readers + 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        if (supervision.published_snapshot.load(std::memory_order_acquire) !=
            index) {
            slot.state.fetch_sub(1, std::memory_order_release);
            continue;
        }
        *out = slot.value;
        slot.state.fetch_sub(1, std::memory_order_release);
        return 0;
    }
    return LFM_STATUS_WOULD_BLOCK;
}

uint64_t capture_delay_ns(uint64_t frames, uint32_t sample_rate) {
    if (sample_rate == 0 ||
        frames > (UINT64_MAX - sample_rate + 1) / UINT64_C(1'000'000'000)) {
        return UINT64_MAX;
    }
    return (frames * UINT64_C(1'000'000'000) + sample_rate - 1) /
           sample_rate;
}

int capture_deadline_arm(LfmSession *session, uint32_t slot,
                         uint64_t delay_ns, uint64_t domain_generation) {
    CaptureSupervision &supervision = session->capture_supervision;
    if (!supervision.cycle_active || slot >= CAPTURE_DEADLINE_COUNT ||
        delay_ns == UINT64_MAX) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    CaptureDeadlineRole &role = supervision.roles[slot];
    if (role.arm.arm_generation != 0 || role.terminal) {
        return LFM_STATUS_BUSY;
    }
    role.domain_generation = domain_generation;
    const kc_deadline_arm_config config = {
        .size = sizeof(kc_deadline_arm_config),
        .abi_version = KC_ABI_VERSION,
        .slot = slot,
        .reserved = 0,
        .delay_ns = delay_ns,
        .child = role.lease.child,
        .parent = role.lease.parent,
        .scope_generation = role.lease.scope_generation,
        .epoch = supervision.epoch,
        .domain = supervision.domain,
        .team_generation = domain_generation,
    };
    return kc_deadline_source_arm(
        supervision.source, &config, &role.arm);
}

int capture_supervision_begin(LfmSession *session, uint64_t cursor) {
    CaptureSupervision &supervision = session->capture_supervision;
    CapturePolicy &policy = session->capture_policy;
    if (!supervision.scope || !supervision.source ||
        supervision.cycle_active || !policy.turn_active ||
        supervision.next_scope_generation == 0) {
        return LFM_STATUS_INTERNAL;
    }
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer) return LFM_STATUS_STALE;
    const bool same_lease_restart =
        supervision.restart_parent.sequence != 0;
    const LfmTicketIdV1 expected_parent = same_lease_restart
        ? supervision.restart_parent
        : policy.turn_ticket;
    const uint64_t epoch = session->epoch.load(std::memory_order_acquire);
    if (expected_parent.sequence == 0 || epoch == 0) {
        return LFM_STATUS_STALE;
    }

    const uint64_t generation = supervision.next_scope_generation++;
    if (supervision.next_scope_generation == 0) return -EOVERFLOW;
    LfmTicketIdV1 tickets[CAPTURE_DEADLINE_COUNT]{};
    for (LfmTicketIdV1 &ticket : tickets) {
        ticket = next_ticket(session, LFM_TICKET_DEADLINE);
    }
    kc_scope_child_lease leases[CAPTURE_DEADLINE_COUNT]{};
    const kc_fixed_scope_cycle_config cycle = {
        .size = sizeof(kc_fixed_scope_cycle_config),
        .abi_version = KC_ABI_VERSION,
        .child_count = CAPTURE_DEADLINE_COUNT,
        .reserved = 0,
        .generation = generation,
        .parent = expected_parent,
        .child_tickets = tickets,
    };
    const int begun = kc_fixed_scope_cycle_begin(
        supervision.scope, &cycle, leases, CAPTURE_DEADLINE_COUNT);
    if (begun != 0) return begun;

    supervision.parent = expected_parent;
    supervision.restart_parent = {};
    supervision.scope_generation = generation;
    supervision.epoch = epoch;
    supervision.domain = session->id;
    supervision.pause_generation = policy.pause_generation;
    supervision.commit_cursor = 0;
    supervision.commit_lease_id = 0;
    supervision.cycle_active = true;
    supervision.restart_after_cancel = false;
    supervision.commit_after_cancel = false;
    supervision.freeze_pending = false;
    for (uint32_t slot = 0; slot < CAPTURE_DEADLINE_COUNT; ++slot) {
        CaptureDeadlineRole &role = supervision.roles[slot];
        role.lease = leases[slot];
        role.arm = {};
        role.domain_generation = 0;
        role.expiry_generation = 0;
        role.grace_armed = false;
        role.terminal = false;
        role.cancel_scope_generation.store(0, std::memory_order_relaxed);
        role.cancel_child_generation.store(0, std::memory_order_relaxed);
        role.cancel_cause.store(0, std::memory_order_release);
    }

    const uint64_t forced_frames =
        (static_cast<uint64_t>(session->capture_rate) * 30'000 + 999) / 1000;
    const uint64_t elapsed = cursor > policy.turn_start_cursor
                                 ? cursor - policy.turn_start_cursor
                                 : 0;
    const uint64_t remaining = elapsed < forced_frames
                                   ? forced_frames - elapsed
                                   : 0;
    policy.forced_sample_generation =
        elapsed >= forced_frames ? generation : 0;
    policy.forced_expiry_generation = 0;
    policy.forced_ready_generation = 0;
    const int armed = capture_deadline_arm(
        session, CAPTURE_DEADLINE_FORCED,
        capture_delay_ns(remaining, session->capture_rate),
        generation);
    if (armed == 0) return 0;
    (void)kc_fixed_scope_cancel(
        supervision.scope, generation, &supervision.parent,
        KC_SCOPE_CAUSE_FAULT);
    return armed;
}

int capture_supervision_arm_pause(LfmSession *session) {
    CaptureSupervision &supervision = session->capture_supervision;
    const uint64_t generation = session->capture_policy.pause_generation;
    if (!supervision.cycle_active || generation == 0) {
        return LFM_STATUS_INTERNAL;
    }
    int status = capture_deadline_arm(
        session, CAPTURE_DEADLINE_PREPARE,
        CAPTURE_PREPARE_DELAY_NS, generation);
    if (status != 0) return status;
    status = capture_deadline_arm(
        session, CAPTURE_DEADLINE_COMMIT,
        CAPTURE_COMMIT_DELAY_NS, generation);
    if (status != 0) {
        (void)kc_fixed_scope_cancel(
            supervision.scope, supervision.scope_generation,
            &supervision.parent, KC_SCOPE_CAUSE_FAULT);
        return status;
    }
    supervision.pause_generation = generation;
    return 0;
}

int capture_supervision_cancel(LfmSession *session, uint32_t cause,
                               bool restart, bool commit) {
    CaptureSupervision &supervision = session->capture_supervision;
    supervision.restart_after_cancel = restart;
    supervision.commit_after_cancel = commit;
    supervision.restart_parent = {};
    if (restart && supervision.cycle_active) {
        if (!ticket_equal(session->capture_policy.turn_ticket,
                          supervision.parent) ||
            session->epoch.load(std::memory_order_acquire) !=
                supervision.epoch) {
            return LFM_STATUS_STALE;
        }
        supervision.restart_parent = supervision.parent;
    }
    if (!supervision.cycle_active) return 0;
    const int status = kc_fixed_scope_cancel(
        supervision.scope, supervision.scope_generation,
        &supervision.parent, cause);
    return status == -EALREADY || status == -ECANCELED ? 0 : status;
}

bool capture_deadline_event_matches(
    const LfmSession *session, const CaptureDeadlineRole &role,
    const kc_deadline_event &event) {
    const CaptureSupervision &supervision = session->capture_supervision;
    if (event.slot != role.slot || event.sequence == 0 ||
        role.arm.arm_generation == 0 ||
        event.scheduled_arm_generation != role.arm.arm_generation ||
        !ticket_equal(event.child, role.arm.child) ||
        !ticket_equal(event.parent, role.arm.parent) ||
        !ticket_equal(event.child, role.lease.child) ||
        !ticket_equal(event.parent, role.lease.parent) ||
        event.scope_generation != role.lease.scope_generation ||
        event.scope_generation != supervision.scope_generation ||
        event.epoch != supervision.epoch ||
        event.domain != supervision.domain || event.domain != session->id ||
        event.team_generation != role.domain_generation) {
        return false;
    }
    if (event.kind == KC_DEADLINE_EVENT_EXPIRED) {
        return event.current_arm_generation == role.arm.arm_generation;
    }
    return event.kind == KC_DEADLINE_EVENT_STALE &&
           role.arm.arm_generation != UINT64_MAX &&
           event.current_arm_generation == role.arm.arm_generation + 1;
}

int capture_scope_child_terminal(LfmSession *session,
                                 CaptureDeadlineRole *role,
                                 uint32_t cause) {
    if (role->terminal) return 0;
    const int status = kc_fixed_scope_child_terminal(
        session->capture_supervision.scope, &role->lease, cause);
    if (status != 0 && status != -EALREADY) return status;
    role->terminal = true;
    role->cancel_cause.store(0, std::memory_order_release);
    return 0;
}

int capture_supervision_cancel_children(LfmSession *session) {
    CaptureSupervision &supervision = session->capture_supervision;
    for (CaptureDeadlineRole &role : supervision.roles) {
        const uint32_t cause =
            role.cancel_cause.load(std::memory_order_acquire);
        if (cause == 0) continue;
        if (role.cancel_scope_generation.load(std::memory_order_relaxed) !=
                role.lease.scope_generation ||
            role.cancel_child_generation.load(std::memory_order_relaxed) !=
                role.lease.child_generation) {
            return LFM_STATUS_STALE;
        }
        if (role.arm.arm_generation == 0) {
            const int terminal = capture_scope_child_terminal(
                session, &role, cause);
            if (terminal != 0) return terminal;
            continue;
        }
        const int disarmed = kc_deadline_source_disarm(
            supervision.source, role.slot, role.arm.arm_generation);
        if (disarmed != 0 && disarmed != -EALREADY &&
            disarmed != -ESTALE && disarmed != -ECANCELED) {
            return disarmed;
        }
    }
    return 0;
}

void record_capture_expiry(CapturePolicy *policy, uint32_t slot,
                           uint64_t generation) {
    if (slot == CAPTURE_DEADLINE_PREPARE) {
        policy->prepare_expiry_generation = generation;
        return;
    }
    if (slot == CAPTURE_DEADLINE_COMMIT) {
        policy->commit_expiry_generation = generation;
        return;
    }
    policy->forced_expiry_generation = generation;
}

int capture_supervision_drain_events(LfmSession *session, bool *progress) {
    CaptureSupervision &supervision = session->capture_supervision;
    for (CaptureDeadlineRole &role : supervision.roles) {
        kc_deadline_event event = {
            .size = sizeof(kc_deadline_event),
            .abi_version = KC_ABI_VERSION,
        };
        const int observed = kc_deadline_source_event_get(
            supervision.source, role.slot, &event);
        if (observed == -EAGAIN) continue;
        if (observed != 0) return observed;
        *progress = true;
        const bool matches = capture_deadline_event_matches(
            session, role, event);
        uint32_t cause = role.cancel_cause.load(std::memory_order_acquire);
        if (!matches) cause = KC_SCOPE_CAUSE_FAULT;
        if (cause == 0 && event.kind == KC_DEADLINE_EVENT_EXPIRED) {
            role.expiry_generation = event.team_generation;
            record_capture_expiry(
                &session->capture_policy, role.slot,
                event.team_generation);
        }
        const int acknowledged = kc_deadline_source_event_ack(
            supervision.source, &event);
        role.arm = {};
        if (acknowledged != 0) return acknowledged;
        if (cause != 0) {
            const int terminal = capture_scope_child_terminal(
                session, &role, cause);
            if (terminal != 0) return terminal;
        }
        if (!matches) return LFM_STATUS_STALE;
        if (cause == 0 && role.slot == CAPTURE_DEADLINE_FORCED) {
            LfmCaptureProducer *producer =
                session->chunk_producer.load(std::memory_order_acquire);
            if (!producer) return LFM_STATUS_STALE;
            const uint64_t cursor = producer->sample_cursor.load(
                std::memory_order_acquire);
            const uint64_t start = session->capture_policy.turn_start_cursor;
            const uint64_t elapsed = cursor > start ? cursor - start : 0;
            const uint64_t forced =
                (static_cast<uint64_t>(session->capture_rate) * 30'000 +
                 999) /
                1000;
            const uint64_t cadence =
                (static_cast<uint64_t>(session->capture_rate) + 49) / 50;
            const bool writer = producer->writer.gate.load(
                                    std::memory_order_acquire) ==
                                CAPTURE_WRITER_ACTIVE;
            const uint64_t completion_bound =
                session->capture_callback_frames;
            if (elapsed + (writer ? completion_bound : cadence) < forced) {
                const int terminal = capture_scope_child_terminal(
                    session, &role, KC_SCOPE_CAUSE_FAULT);
                return terminal == 0 ? -ETIMEDOUT : terminal;
            }
            if (elapsed < forced) {
                if (role.grace_armed) {
                    const int terminal = capture_scope_child_terminal(
                        session, &role, KC_SCOPE_CAUSE_FAULT);
                    return terminal == 0 ? -ETIMEDOUT : terminal;
                }
                role.grace_armed = true;
                const int rearmed = capture_deadline_arm(
                    session, CAPTURE_DEADLINE_FORCED,
                    capture_delay_ns(
                        writer ? completion_bound : cadence,
                        session->capture_rate),
                    role.domain_generation);
                if (rearmed != 0) return rearmed;
            } else {
                role.grace_armed = false;
            }
        }
    }
    return 0;
}

int capture_supervision_gate(LfmSession *session) {
    CapturePolicy &policy = session->capture_policy;
    CaptureSupervision &supervision = session->capture_supervision;
    if (!supervision.cycle_active) return 0;
    if (supervision.epoch !=
        session->epoch.load(std::memory_order_acquire)) {
        return 0;
    }
    const uint64_t generation = policy.pause_generation;
    if (policy.state == CAPTURE_POLICY_PAUSE &&
        policy.prepare_sample_generation == generation &&
        policy.prepare_expiry_generation == generation &&
        policy.prepare_ready_generation == 0) {
        /* Candidate-owned model scratch does not exist yet. Readiness is
         * durable policy state only; committed recurrence remains untouched. */
        const int terminal = capture_scope_child_terminal(
            session, &supervision.roles[CAPTURE_DEADLINE_PREPARE],
            KC_SCOPE_CAUSE_COMPLETE);
        if (terminal != 0) return terminal;
        policy.prepare_ready_generation = generation;
    }
    const bool commit = policy.state == CAPTURE_POLICY_PAUSE &&
                        policy.commit_sample_generation == generation &&
                        policy.commit_expiry_generation == generation &&
                        policy.commit_ready_generation == 0;
    const uint64_t forced_generation =
        supervision.roles[CAPTURE_DEADLINE_FORCED].domain_generation;
    const bool forced =
        (policy.state == CAPTURE_POLICY_SPEAKING ||
        policy.state == CAPTURE_POLICY_PAUSE) &&
        forced_generation != 0 &&
        policy.forced_sample_generation == forced_generation &&
        policy.forced_expiry_generation == forced_generation &&
        policy.forced_ready_generation == 0;
    if (!commit && !forced) return 0;
    const uint32_t slot = commit ? CAPTURE_DEADLINE_COMMIT
                                 : CAPTURE_DEADLINE_FORCED;
    const int terminal = capture_scope_child_terminal(
        session, &supervision.roles[slot], KC_SCOPE_CAUSE_COMPLETE);
    if (terminal != 0) return terminal;
    if (commit) policy.commit_ready_generation = generation;
    if (forced) policy.forced_ready_generation = forced_generation;

    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer) return LFM_STATUS_STALE;
    if (!ticket_equal(policy.turn_ticket, supervision.parent) ||
        session->epoch.load(std::memory_order_acquire) !=
            supervision.epoch) {
        return LFM_STATUS_STALE;
    }
    const uint64_t forced_end = policy.turn_start_cursor >
            UINT64_MAX - session->capture_turn_frames
        ? UINT64_MAX
        : policy.turn_start_cursor + session->capture_turn_frames;
    supervision.commit_cursor = forced
        ? std::min(policy.last_evidence_cursor, forced_end)
        : policy.last_evidence_cursor;
    supervision.commit_lease_id = 0;
    return capture_supervision_cancel(
        session, KC_SCOPE_CAUSE_CANCELLED, false, true);
}

enum CaptureSupervisionProgress : int {
    CAPTURE_SUPERVISION_IDLE = 0,
    CAPTURE_SUPERVISION_PROGRESS = 1,
    CAPTURE_SUPERVISION_STOPPING = 2,
    CAPTURE_SUPERVISION_STOPPED = 3,
};

int step_capture_supervision(LfmSession *session) {
    CaptureSupervision &supervision = session->capture_supervision;
    if (!supervision.scope || !supervision.source) {
        return session->stop.load(std::memory_order_acquire)
                   ? CAPTURE_SUPERVISION_STOPPED
                   : CAPTURE_SUPERVISION_IDLE;
    }
    bool progress = false;
    if (supervision.cycle_active &&
        supervision.epoch !=
            session->epoch.load(std::memory_order_acquire) &&
        !supervision.stop_requested &&
        !supervision.restart_after_cancel) {
        const int cancelled = capture_supervision_cancel(
            session, KC_SCOPE_CAUSE_CANCELLED, false, false);
        if (cancelled != 0) return cancelled;
        progress = true;
    }
    if (session->stop.load(std::memory_order_acquire) &&
        !supervision.stop_requested) {
        supervision.stop_requested = true;
        supervision.restart_after_cancel = false;
        supervision.commit_after_cancel = false;
        const int cancelled = capture_supervision_cancel(
            session, KC_SCOPE_CAUSE_STOPPED, false, false);
        if (cancelled != 0) return cancelled;
        progress = true;
    }
    const int cancels = capture_supervision_cancel_children(session);
    if (cancels != 0) return cancels;
    const int events = capture_supervision_drain_events(session, &progress);
    if (events != 0) return events;

    kc_fixed_scope_snapshot scope = {
        .size = sizeof(kc_fixed_scope_snapshot),
        .abi_version = KC_ABI_VERSION,
    };
    int status = kc_fixed_scope_snapshot_get(supervision.scope, &scope);
    if (status != 0) return status;
    if (supervision.cycle_active &&
        scope.phase == KC_FIXED_SCOPE_TERMINAL) {
        supervision.cycle_active = false;
        progress = true;
        if (supervision.commit_after_cancel) {
            supervision.commit_after_cancel = false;
            supervision.freeze_pending = true;
        } else if (supervision.restart_after_cancel &&
                   session->capture_policy.state ==
                       CAPTURE_POLICY_SPEAKING) {
            supervision.restart_after_cancel = false;
            status = capture_supervision_begin(
                session, session->capture_policy.last_evidence_cursor);
            if (status != 0) return status;
        } else {
            supervision.restart_after_cancel = false;
            if (supervision.device_loss_after_cancel) {
                supervision.device_loss_after_cancel = false;
                supervision.device_loss_ready = true;
                LfmCaptureProducer *producer =
                    session->chunk_producer.load(std::memory_order_acquire);
                const uint64_t cursor = producer
                    ? producer->sample_cursor.load(std::memory_order_acquire)
                    : session->capture_policy.last_evidence_cursor;
                reset_capture_policy(session, cursor, true);
            }
            if (!session->stop.load(std::memory_order_acquire) &&
                supervision.epoch !=
                    session->epoch.load(std::memory_order_acquire)) {
                LfmCaptureProducer *producer =
                    session->chunk_producer.load(std::memory_order_acquire);
                const uint64_t cursor = producer
                    ? producer->sample_cursor.load(std::memory_order_acquire)
                    : session->capture_policy.last_evidence_cursor;
                reset_capture_policy(session, cursor, true);
                /* An old-epoch callback may still own ACTIVE and publish
                 * beyond this cursor after the reset. Leave the detector
                 * segment unbound until the first current-epoch record; that
                 * record establishes the only valid pre-roll boundary. */
                session->capture_policy.segment_epoch = 0;
            }
            supervision.parent = {};
        }
    }

    status = capture_supervision_gate(session);
    if (status != 0) return status;
    if (!session->stop.load(std::memory_order_acquire) &&
        supervision.device_loss_pending.exchange(
            false, std::memory_order_acq_rel)) {
        supervision.device_loss_parent = supervision.parent;
        supervision.device_loss_epoch =
            supervision.cycle_active || supervision.freeze_pending
                ? supervision.epoch
                : session->epoch.load(std::memory_order_acquire);
        if (supervision.cycle_active) {
            supervision.device_loss_after_cancel = true;
            /* Deadline events and their dual gate are drained first. A commit
             * that already won owns the scope cancellation; endpoint loss may
             * observe it but cannot replace its disposition. */
            if (!supervision.commit_after_cancel) {
                const int cancelled = capture_supervision_cancel(
                    session, KC_SCOPE_CAUSE_CANCELLED, false, false);
                if (cancelled != 0) return cancelled;
            }
        } else if (supervision.freeze_pending) {
            /* The range is committed but not yet mounted. Preserve its exact
             * policy state; freeze is the successor, and only then may device
             * loss reset capture for a replacement endpoint. */
            supervision.device_loss_after_cancel = true;
        } else {
            supervision.device_loss_ready = true;
            const uint64_t cursor = session->capture_policy.last_evidence_cursor;
            reset_capture_policy(session, cursor, true);
        }
        progress = true;
    }
    if (!session->stop.load(std::memory_order_acquire)) {
        if ((supervision.commit_after_cancel ||
             supervision.device_loss_after_cancel) &&
            supervision.cycle_active) {
            return CAPTURE_SUPERVISION_STOPPING;
        }
        return progress ? CAPTURE_SUPERVISION_PROGRESS
                        : CAPTURE_SUPERVISION_IDLE;
    }
    if (supervision.cycle_active) return CAPTURE_SUPERVISION_STOPPING;
    if (!supervision.source_stop_requested) {
        kc_deadline_source_request_stop(supervision.source);
        supervision.source_stop_requested = true;
        return CAPTURE_SUPERVISION_PROGRESS;
    }
    kc_deadline_source_snapshot source = {
        .size = sizeof(kc_deadline_source_snapshot),
        .abi_version = KC_ABI_VERSION,
    };
    status = kc_deadline_source_snapshot_get(supervision.source, &source);
    if (status != 0) return status;
    return source.phase == KC_DEADLINE_SOURCE_STOPPED &&
                   source.pending_events == 0
               ? CAPTURE_SUPERVISION_STOPPED
               : CAPTURE_SUPERVISION_STOPPING;
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
    (void)playback_release(session, &playback->lease);
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
    return playback_reserve(session, session->playback_frames,
                            session->playback_rate, out);
}

void route_notify(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    notify_session(session);
}

void release_action_capture_range(LfmSession *session, SessionAction *action) {
    if (!action) return;
    if (action->capture_range_active) {
        (void)release_capture_range(session, action->capture_range);
        action->capture_range = {};
        action->capture_range_active = false;
    }
}

void clear_action(LfmSession *session) {
    if (session->action.admission_pending || session->action.route_pending) {
        std::abort();
    }
    release_action_capture_range(session, &session->action);
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
    if (result.records[0].kind == LFM_EVENT_TURN_STARTED &&
        session->stop.load(std::memory_order_acquire)) {
        for (uint32_t index = result.next; index < result.count; ++index) {
            if (result.records[index].kind == LFM_EVENT_TURN) {
                result.records[index].status = LFM_STATUS_CANCELLED;
                result.records[index].flags &= ~LFM_EVENT_FLAG_TRUNCATED;
            }
        }
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
    /* Any outward record may be the host's causal observation edge. Publish
     * the coordinator-owned diagnostic image before making that edge visible. */
    publish_capture_supervision_snapshot(session);
    while (result.next < result.count) {
        const EventRecord &record = result.records[result.next];
        if (!event_push(&session->events, record)) {
            return RESULT_BLOCKED;
        }
        if (record.kind == LFM_EVENT_TURN_STARTED &&
            session->action.active &&
            ticket_equal(record.ticket, session->action.ticket)) {
            session->action.turn_started_visible = true;
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
    release_action_capture_range(session, &action);
    release_prepared(session, &action.playback);
    const bool visible = !action.turn_started_required ||
                         action.turn_started_visible;
    if (session->stop.load(std::memory_order_acquire)) {
        if (visible) {
            (void)stage_turn(session, action.epoch, action.ticket,
                             action.playback_count, action.emitted, 0,
                             LFM_STATUS_CANCELLED);
        }
        action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
        return;
    }
    if (session->epoch.load(std::memory_order_acquire) != action.epoch) {
        if (visible) {
            (void)stage_turn(session, action.epoch, action.ticket,
                             action.playback_count, action.emitted, 0,
                             LFM_STATUS_STALE);
        }
        action.phase = ACTION_PHASE_TERMINAL_PUBLISHED;
        return;
    }
    if (!visible) {
        stage_error(session, status, message);
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
            release_action_capture_range(session, &action);
            if (rc != 0) {
                fail_action(session, rc, "native turn admission failed");
                return ACTION_PROGRESS;
            }
            if (action.announce_start) {
                action.announce_start = false;
                action.phase = ACTION_PHASE_TURN_STARTED_PUBLISHED;
                if (!stage_event(session, LFM_EVENT_TURN_STARTED,
                                 action.epoch, action.ticket, 0, nullptr, 0,
                                 0, false)) {
                    fail_action(session, LFM_STATUS_INTERNAL,
                                "native turn-start publication failed");
                }
                return ACTION_PROGRESS;
            }
            action.phase = ACTION_PHASE_EMIT;
        }
        if (action.phase == ACTION_PHASE_TURN_STARTED_PUBLISHED) {
            action.phase = ACTION_PHASE_EMIT;
            continue;
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
            if (!action.turn_started_required ||
                action.turn_started_visible) {
                (void)stage_turn(session, action.epoch, action.ticket,
                                 action.playback_count, action.emitted,
                                 action.interrupt_flags,
                                 action.interrupt_status);
            }
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
                    const int rc = playback_publish(
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
            rc = playback_resolve_mut(
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
                              bool turn_started_required) {
    if (session->action.active) {
        if (!turn_started_required) {
            stage_action_failure(session, action_epoch, ticket,
                                 LFM_STATUS_BUSY,
                                 "conversation already has a mutating route");
        }
        return nullptr;
    }
    session->action = {};
    session->action.ticket = ticket;
    session->action.epoch = action_epoch;
    session->action.phase = ACTION_PHASE_ADMISSION_PENDING;
    session->action.active = true;
    session->action.admission_pending = true;
    session->action.turn_started_required = turn_started_required;
    session->action.announce_start = turn_started_required;
    return &session->action;
}

enum CaptureFreezeProgress : int {
    CAPTURE_FREEZE_NONE = 0,
    CAPTURE_FREEZE_PROGRESS = 1,
    CAPTURE_FREEZE_WRITER = 2,
    CAPTURE_FREEZE_CAPACITY = 3,
};

enum CaptureRetireProgress : int {
    CAPTURE_RETIRE_NONE = 0,
    CAPTURE_RETIRE_PROGRESS = 1,
    CAPTURE_RETIRE_BLOCKED = 2,
};

int refresh_capture_reclaim(LfmSession *session) {
    CaptureArena &arena = session->capture_arena;
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    uint64_t floor = producer
        ? producer->sample_cursor.load(std::memory_order_acquire)
        : session->capture_policy.last_evidence_cursor;
    const uint64_t policy_floor = session->capture_policy.turn_active
        ? session->capture_policy.turn_start_cursor
        : session->capture_policy.segment_cursor;
    floor = std::min(floor, policy_floor);
    for (const CaptureRangeSlot &slot : arena.ranges) {
        const uint32_t state = slot.state.load(std::memory_order_acquire);
        if (state == CAPTURE_RANGE_RESERVED ||
            state == CAPTURE_RANGE_PUBLISHED ||
            state == CAPTURE_RANGE_CONSUMING) {
            floor = std::min(floor, slot.lease.first_sample_cursor);
        }
    }
    const uint64_t prior =
        arena.reclaim_cursor.value.load(std::memory_order_relaxed);
    if (floor < prior) return LFM_STATUS_INTERNAL;
    arena.reclaim_cursor.value.store(floor, std::memory_order_release);
    return 0;
}

int claim_capture_range(LfmSession *session, uint64_t start, uint64_t end,
                        uint64_t epoch, const LfmTicketIdV1 &ticket,
                        CaptureRangeLease *out) {
    CaptureArena &arena = session->capture_arena;
    if (!out || end <= start || end - start > session->capture_turn_frames ||
        end - start > UINT32_MAX || arena.capacity_frames == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t first =
        arena.range_cursor.value.fetch_add(1, std::memory_order_relaxed) %
        CAPTURE_RANGE_CAPACITY;
    for (uint32_t attempt = 0; attempt < CAPTURE_RANGE_CAPACITY; ++attempt) {
        const uint32_t index = (first + attempt) % CAPTURE_RANGE_CAPACITY;
        CaptureRangeSlot &slot = arena.ranges[index];
        uint32_t expected = CAPTURE_RANGE_FREE;
        if (!slot.state.compare_exchange_strong(
                expected, CAPTURE_RANGE_RESERVED, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        const uint64_t identity = lease_id(CAPTURE_IDENTITY_DIRECTION, index);
        const uint64_t generation =
            slot.generation.load(std::memory_order_acquire);
        if (identity == 0 || generation == 0) {
            slot.state.store(CAPTURE_RANGE_RETIRED,
                             std::memory_order_release);
            return LFM_STATUS_OUT_OF_MEMORY;
        }
        const CaptureRangeLease lease = {
            .lease_id = identity,
            .buffer_generation = generation,
            .first_sample_cursor = start,
            .stream_epoch = epoch,
            .ticket = ticket,
            .frames = static_cast<uint32_t>(end - start),
            .sample_rate = session->capture_rate,
            .slot = index,
        };
        slot.lease = lease;
        slot.identity.store(identity, std::memory_order_release);
        if (!capture_range_push(&arena.ready, lease)) {
            slot.lease = {};
            slot.identity.store(0, std::memory_order_relaxed);
            slot.state.store(CAPTURE_RANGE_FREE, std::memory_order_release);
            return LFM_STATUS_WOULD_BLOCK;
        }
        slot.state.store(CAPTURE_RANGE_PUBLISHED, std::memory_order_release);
        *out = lease;
        return 0;
    }
    return LFM_STATUS_WOULD_BLOCK;
}

int take_capture_range(LfmSession *session, CaptureRangeLease *out) {
    CaptureRangeLease lease{};
    if (!capture_range_pop(&session->capture_arena.ready, &lease)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (lease.slot >= CAPTURE_RANGE_CAPACITY) return LFM_STATUS_INTERNAL;
    CaptureRangeSlot &slot = session->capture_arena.ranges[lease.slot];
    if (slot.identity.load(std::memory_order_acquire) != lease.lease_id ||
        slot.lease.buffer_generation != lease.buffer_generation ||
        slot.lease.first_sample_cursor != lease.first_sample_cursor) {
        return LFM_STATUS_STALE;
    }
    uint32_t expected = CAPTURE_RANGE_PUBLISHED;
    if (!slot.state.compare_exchange_strong(
            expected, CAPTURE_RANGE_CONSUMING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    *out = lease;
    return 0;
}

int release_capture_range(LfmSession *session,
                          const CaptureRangeLease &lease) {
    if (lease.slot >= CAPTURE_RANGE_CAPACITY) return LFM_STATUS_INVALID_ARGUMENT;
    CaptureRangeSlot &slot = session->capture_arena.ranges[lease.slot];
    if (slot.identity.load(std::memory_order_acquire) != lease.lease_id ||
        slot.lease.buffer_generation != lease.buffer_generation ||
        slot.lease.first_sample_cursor != lease.first_sample_cursor) {
        return LFM_STATUS_STALE;
    }
    uint32_t expected = CAPTURE_RANGE_CONSUMING;
    if (!slot.state.compare_exchange_strong(
            expected, CAPTURE_RANGE_RESERVED, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    slot.lease = {};
    slot.identity.store(0, std::memory_order_relaxed);
    const uint64_t generation =
        slot.generation.load(std::memory_order_relaxed);
    if (generation == UINT64_MAX) {
        slot.state.store(CAPTURE_RANGE_RETIRED, std::memory_order_release);
        return -EOVERFLOW;
    }
    slot.generation.store(generation + 1, std::memory_order_relaxed);
    slot.state.store(CAPTURE_RANGE_FREE, std::memory_order_release);
    return refresh_capture_reclaim(session);
}

int retire_closed_capture_producer(LfmSession *session) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer || !producer->closing.load(std::memory_order_acquire)) {
        return CAPTURE_RETIRE_NONE;
    }
    if (!capture_chunk_empty(session->capture_chunks) ||
        session->capture_policy.chunk_pending ||
        producer->writer.gate.load(std::memory_order_acquire) ==
            CAPTURE_WRITER_ACTIVE) {
        return CAPTURE_RETIRE_BLOCKED;
    }
    producer->gap_debt_frames.store(0, std::memory_order_release);
    producer->gap_debt_channels.store(0, std::memory_order_relaxed);
    producer->gap_debt_flags.store(0, std::memory_order_relaxed);
    session->chunk_producer.store(nullptr, std::memory_order_release);
    (void)refresh_capture_reclaim(session);
    return CAPTURE_RETIRE_PROGRESS;
}

int retire_unstarted_capture_producer(LfmSession *session) {
    const uint64_t gate =
        session->publication_gate.value.load(std::memory_order_acquire);
    if ((gate & PUBLICATION_COUNT_MASK) != 0) {
        return LFM_STATUS_BUSY;
    }
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer) return CAPTURE_RETIRE_NONE;
    if (!producer->closing.load(std::memory_order_acquire)) {
        return LFM_STATUS_BUSY;
    }

    /* A CREATED session has no runnable coordinator to consume the records
     * accepted before setup failed. The hardware endpoint is already joined,
     * publication admission is closed with no active publisher, and the ring
     * is fixed-size, so administrative join owns this bounded retirement. */
    LfmCaptureChunkV1 discarded{};
    for (uint32_t index = 0; index < CAPTURE_CHUNK_CAPACITY; ++index) {
        if (!capture_chunk_pop(&session->capture_chunks, &discarded)) break;
    }
    return retire_closed_capture_producer(session);
}

int freeze_capture_turn(LfmSession *session) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    CaptureSupervision &supervision = session->capture_supervision;
    if (!producer || !supervision.freeze_pending) {
        return CAPTURE_FREEZE_NONE;
    }
    CapturePolicy &policy = session->capture_policy;
    const uint64_t start = policy.turn_start_cursor;
    const uint64_t end = supervision.commit_cursor;
    const uint64_t published =
        producer->sample_cursor.load(std::memory_order_acquire);
    if (!policy.turn_active || end <= start || end > published ||
        end - start > session->capture_turn_frames) {
        return LFM_STATUS_INTERNAL;
    }
    CaptureRangeLease range{};
    const int claimed = claim_capture_range(
        session, start, end, supervision.epoch, supervision.parent, &range);
    if (claimed == LFM_STATUS_WOULD_BLOCK) return CAPTURE_FREEZE_CAPACITY;
    if (claimed != 0) return claimed;

    LfmCaptureChunkV1 suffix = policy.chunk;
    const bool suffix_pending = policy.chunk_pending &&
        suffix.stream_epoch == supervision.epoch &&
        suffix.stream_epoch == session->epoch.load(std::memory_order_acquire) &&
        end < suffix.first_sample_cursor + suffix.frames;
    supervision.freeze_pending = false;
    supervision.commit_cursor = 0;
    supervision.commit_lease_id = 0;
    supervision.parent = {};
    reset_capture_policy(session, end, false);
    const LfmTicketIdV1 next = rotate_capture_ticket(
        producer, session->epoch.load(std::memory_order_acquire));
    policy.turn_ticket = next;
    if (suffix_pending) {
        suffix.turn_ticket = next;
        policy.chunk = suffix;
        policy.chunk_pending = true;
    }
    if (supervision.device_loss_after_cancel) {
        /* A committed range is no longer a device-owned in-flight turn. Keep
         * it for model admission and report endpoint loss independently. */
        supervision.device_loss_after_cancel = false;
        supervision.device_loss_ready = true;
        supervision.device_loss_parent = {};
        supervision.device_loss_epoch =
            session->epoch.load(std::memory_order_acquire);
    }
    const int reclaim = refresh_capture_reclaim(session);
    if (reclaim != 0) return reclaim;
    return CAPTURE_FREEZE_PROGRESS;
}

int recycle_background_silence(LfmSession *session) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    CapturePolicy &policy = session->capture_policy;
    if (!producer || producer->closing.load(std::memory_order_acquire) ||
        policy.state != CAPTURE_POLICY_LISTENING || policy.chunk_pending ||
        !capture_chunk_empty(session->capture_chunks) ||
        policy.next_evidence_cursor <=
            producer->sample_cursor.load(std::memory_order_acquire)) {
        return CAPTURE_FREEZE_NONE;
    }
    const uint32_t recycle_frames = std::max(
        session->capture_callback_frames,
        static_cast<uint32_t>(2 * LFM_SESAME_FFT_SIZE));
    /* Background silence is dead after detector consumption. Rotate it after
     * one sealed maximum callback rather than letting the forced-turn-sized
     * arena capacity dictate retention. This bounds the prefix preceding a later voice
     * onset while preserving every complete callback publication. */
    const uint64_t cursor =
        producer->sample_cursor.load(std::memory_order_acquire);
    if (cursor < policy.segment_cursor ||
        cursor - policy.segment_cursor < recycle_frames) {
        return CAPTURE_FREEZE_NONE;
    }
    policy.discarded_silence_frames += cursor - policy.segment_cursor;
    policy.segment_cursor = cursor;
    const int reclaim = refresh_capture_reclaim(session);
    if (reclaim != 0) return reclaim;
    return CAPTURE_FREEZE_PROGRESS;
}

bool advance_capture_cadence(CapturePolicy *policy, uint32_t sample_rate) {
    const uint64_t scaled =
        static_cast<uint64_t>(policy->cadence_remainder) + sample_rate;
    const uint64_t step = scaled / 50;
    policy->cadence_remainder = static_cast<uint32_t>(scaled % 50);
    if (step == 0 ||
        policy->next_evidence_cursor > UINT64_MAX - step) {
        return false;
    }
    policy->next_evidence_cursor += step;
    return true;
}

void reset_capture_policy(LfmSession *session, uint64_t cursor,
                          bool reset_detector) {
    CapturePolicy &policy = session->capture_policy;
    if (reset_detector && policy.detector) {
        (void)lfm_sesame_detector_discontinuity(
            policy.detector, LFM_SESAME_STREAM_MIC);
    }
    policy.chunk = {};
    policy.decision = {};
    policy.turn_ticket = {};
    policy.segment_cursor = cursor;
    policy.next_evidence_cursor = cursor;
    policy.last_evidence_cursor = cursor;
    policy.turn_start_cursor = 0;
    policy.last_voiced_cursor = 0;
    policy.voiced_frames = 0;
    policy.silence_frames = 0;
    policy.barge_voiced_frames = 0;
    policy.barge_candidate_epoch = 0;
    policy.barge_candidate_ticket = {};
    policy.barge_triggered = false;
    policy.pause_generation++;
    if (policy.pause_generation == 0) policy.pause_generation = 1;
    policy.prepare_sample_generation = 0;
    policy.commit_sample_generation = 0;
    policy.forced_sample_generation = 0;
    policy.prepare_expiry_generation = 0;
    policy.commit_expiry_generation = 0;
    policy.forced_expiry_generation = 0;
    policy.prepare_ready_generation = 0;
    policy.commit_ready_generation = 0;
    policy.forced_ready_generation = 0;
    policy.segment_epoch = session->epoch.load(std::memory_order_acquire);
    policy.cadence_remainder = 49;
    policy.state = CAPTURE_POLICY_LISTENING;
    policy.chunk_pending = false;
    policy.turn_active = false;
    (void)advance_capture_cadence(&policy, session->capture_rate);
}

uint64_t capture_duration_frames(uint32_t sample_rate, uint32_t milliseconds) {
    return (static_cast<uint64_t>(sample_rate) * milliseconds + 999) / 1000;
}

void clear_barge_candidate(CapturePolicy *policy) {
    policy->barge_voiced_frames = 0;
    policy->barge_candidate_epoch = 0;
    policy->barge_candidate_ticket = {};
}

bool playback_echo_window(const LfmSession *session, uint64_t cursor) {
    const PlaybackPolicy &playback = session->playback_policy;
    const uint64_t epoch = session->epoch.load(std::memory_order_acquire);
    return playback.echo_epoch == epoch &&
           playback.echo_ticket.sequence != 0 &&
           cursor > playback.echo_start_capture_cursor &&
           cursor <= playback.echo_tail_capture_cursor;
}

int trigger_barge_interrupt(LfmSession *session) {
    CapturePolicy &policy = session->capture_policy;
    CaptureSupervision &supervision = session->capture_supervision;
    const PlaybackPolicy &playback = session->playback_policy;
    if (policy.barge_triggered) return 0;
    if (!policy.turn_active || policy.state != CAPTURE_POLICY_SPEAKING ||
        policy.turn_ticket.sequence == 0 || !supervision.cycle_active ||
        supervision.freeze_pending || playback.echo_ticket.sequence == 0 ||
        playback.echo_epoch !=
            session->epoch.load(std::memory_order_acquire)) {
        return LFM_STATUS_INTERNAL;
    }
    if (policy.barge_interrupts == UINT64_MAX) return -EOVERFLOW;

    /* Retire the old epoch's deadline children first, then restart the same
     * capture turn under the new epoch. The capture turn ticket deliberately
     * survives: its initial 400 ms of user speech is still the beginning of
     * this turn and must not be discarded merely because it interrupted the
     * assistant. */
    const int cancelled = capture_supervision_cancel(
        session, KC_SCOPE_CAUSE_CANCELLED, false, false);
    if (cancelled != 0) return cancelled;
    uint64_t epoch = 0;
    const int interrupted = lfm_session_interrupt(session, &epoch);
    if (interrupted != 0) return interrupted;

    supervision.restart_after_cancel = true;
    supervision.commit_after_cancel = false;
    supervision.restart_parent = policy.turn_ticket;
    policy.segment_epoch = epoch;
    if (policy.chunk_pending) {
        policy.chunk.stream_epoch = epoch;
        policy.chunk.turn_ticket = policy.turn_ticket;
    }
    policy.barge_triggered = true;
    policy.barge_source_epoch = playback.echo_epoch;
    policy.barge_interrupt_epoch = epoch;
    policy.barge_playback_ticket = playback.echo_ticket;
    policy.barge_interrupts++;
    return 0;
}

int apply_barge_decision(LfmSession *session, uint64_t cursor,
                         uint64_t interval,
                         const LfmSesameDecisionV1 &decision) {
    CapturePolicy &policy = session->capture_policy;
    const PlaybackPolicy &playback = session->playback_policy;
    if (decision.voice == 0 || !playback_echo_window(session, cursor)) {
        clear_barge_candidate(&policy);
        return 0;
    }
    if (policy.barge_candidate_epoch != playback.echo_epoch ||
        !ticket_equal(policy.barge_candidate_ticket,
                      playback.echo_ticket)) {
        policy.barge_candidate_epoch = playback.echo_epoch;
        policy.barge_candidate_ticket = playback.echo_ticket;
        policy.barge_voiced_frames = 0;
    }
    /* Playback metadata is drained before capture metadata. A playback edge
     * can therefore snapshot a producer cursor whose older microphone blocks
     * are still queued. Count only the portion of this detector interval that
     * is causally newer than the first classified playback voice for this
     * ticket; queued pre-playback speech must never become barge evidence. */
    const uint64_t interval_start = cursor >= interval ? cursor - interval : 0;
    const uint64_t eligible_start = std::max(
        interval_start, playback.echo_start_capture_cursor);
    const uint64_t eligible = cursor - eligible_start;
    if (policy.barge_voiced_frames > UINT64_MAX - eligible) {
        return -EOVERFLOW;
    }
    policy.barge_voiced_frames += eligible;
    const uint64_t sustained = capture_duration_frames(
        session->capture_rate, CAPTURE_BARGE_SUSTAIN_MS);
    if (policy.barge_voiced_frames < sustained) return 0;
    return trigger_barge_interrupt(session);
}

int apply_capture_decision(LfmSession *session, uint64_t cursor,
                           const LfmSesameDecisionV1 &decision) {
    CapturePolicy &policy = session->capture_policy;
    const uint32_t prior_state = policy.state;
    const uint64_t interval = cursor - policy.last_evidence_cursor;
    policy.last_evidence_cursor = cursor;
    policy.decision = decision;
    policy.evidence_updates++;

    const uint64_t minimum =
        capture_duration_frames(session->capture_rate, 300);
    const uint64_t prepare =
        capture_duration_frames(session->capture_rate, 200);
    const uint64_t commit =
        capture_duration_frames(session->capture_rate, 500);
    const uint64_t forced =
        capture_duration_frames(session->capture_rate, 30'000);

    if (decision.voice != 0) {
        if (policy.state == CAPTURE_POLICY_LISTENING) {
            policy.state = CAPTURE_POLICY_CANDIDATE;
            if (policy.turn_ticket.sequence == 0) {
                policy.turn_ticket = policy.chunk.turn_ticket;
            }
            policy.turn_start_cursor =
                std::max(policy.segment_cursor,
                         cursor - LFM_SESAME_FFT_SIZE);
            policy.turn_active = true;
            policy.voiced_frames = interval;
            const int begun = capture_supervision_begin(session, cursor);
            if (begun != 0) return begun;
        } else {
            policy.voiced_frames += interval;
        }
        if (prior_state == CAPTURE_POLICY_CANDIDATE) {
            /* A new positive decision closes a brief spectral valley without
             * discarding the retained candidate span. */
            policy.silence_frames = 0;
        }
        if (prior_state == CAPTURE_POLICY_PAUSE) {
            policy.state = CAPTURE_POLICY_SPEAKING;
            policy.pause_generation++;
            if (policy.pause_generation == 0) policy.pause_generation = 1;
            policy.prepare_sample_generation = 0;
            policy.commit_sample_generation = 0;
            policy.prepare_expiry_generation = 0;
            policy.commit_expiry_generation = 0;
            policy.forced_expiry_generation = 0;
            policy.prepare_ready_generation = 0;
            policy.commit_ready_generation = 0;
            policy.forced_ready_generation = 0;
            policy.silence_frames = 0;
            const int cancelled = capture_supervision_cancel(
                session, KC_SCOPE_CAUSE_CANCELLED, true, false);
            if (cancelled != 0) return cancelled;
        }
        if (policy.state == CAPTURE_POLICY_CANDIDATE &&
            cursor >= policy.turn_start_cursor &&
            cursor - policy.turn_start_cursor >= minimum) {
            /* Minimum utterance is the retained speech span from first
             * evidence through the latest voiced evidence. Detector-negative
             * valleys inside natural speech do not shorten that span, while
             * trailing silence alone can never promote a false start. */
            policy.state = CAPTURE_POLICY_SPEAKING;
        }
        policy.last_voiced_cursor = cursor;
    } else if (policy.state == CAPTURE_POLICY_CANDIDATE) {
        policy.silence_frames += interval;
        if (policy.silence_frames >= commit) {
            const int cancelled = capture_supervision_cancel(
                session, KC_SCOPE_CAUSE_CANCELLED, false, false);
            if (cancelled != 0) return cancelled;
            policy.discarded_silence_frames +=
                cursor - policy.turn_start_cursor;
            policy.state = CAPTURE_POLICY_LISTENING;
            policy.turn_start_cursor = 0;
            policy.turn_active = false;
            policy.last_voiced_cursor = 0;
            policy.voiced_frames = 0;
            policy.silence_frames = 0;
        }
    } else if (policy.state == CAPTURE_POLICY_SPEAKING) {
        policy.state = CAPTURE_POLICY_PAUSE;
        policy.pause_generation++;
        if (policy.pause_generation == 0) policy.pause_generation = 1;
        policy.prepare_sample_generation = 0;
        policy.commit_sample_generation = 0;
        policy.prepare_expiry_generation = 0;
        policy.commit_expiry_generation = 0;
        policy.prepare_ready_generation = 0;
        policy.commit_ready_generation = 0;
        policy.silence_frames = interval;
        const int armed = capture_supervision_arm_pause(session);
        if (armed != 0) return armed;
    } else if (policy.state == CAPTURE_POLICY_PAUSE) {
        policy.silence_frames += interval;
    }

    if (policy.state == CAPTURE_POLICY_PAUSE) {
        if (policy.silence_frames >= prepare &&
            policy.prepare_sample_generation == 0) {
            policy.prepare_sample_generation = policy.pause_generation;
        }
        if (policy.silence_frames >= commit &&
            policy.commit_sample_generation == 0) {
            policy.commit_sample_generation = policy.pause_generation;
        }
    }
    if (policy.turn_active &&
        (policy.state == CAPTURE_POLICY_SPEAKING ||
         policy.state == CAPTURE_POLICY_PAUSE) &&
        cursor - policy.turn_start_cursor >= forced &&
        policy.forced_sample_generation == 0) {
        policy.forced_sample_generation =
            session->capture_supervision
                .roles[CAPTURE_DEADLINE_FORCED]
                .domain_generation;
    }
    const int barge = apply_barge_decision(
        session, cursor, interval, decision);
    if (barge != 0) return barge;
    session->capture_evidence_cursor.store(cursor, std::memory_order_relaxed);
    session->capture_evidence_updates.store(
        policy.evidence_updates, std::memory_order_release);
    return 0;
}

int resolve_capture_window(LfmSession *session,
                           const LfmCaptureChunkV1 &chunk,
                           uint64_t cursor, LfmSesameWindowV1 *out) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer || !out || cursor < LFM_SESAME_FFT_SIZE) {
        return LFM_STATUS_STALE;
    }
    const uint64_t end = chunk.first_sample_cursor + chunk.frames;
    if (end < chunk.first_sample_cursor || cursor > end) {
        return LFM_STATUS_STALE;
    }
    const uint64_t window_cursor = cursor - LFM_SESAME_FFT_SIZE;
    const uint64_t reclaim = session->capture_arena.reclaim_cursor.value.load(
        std::memory_order_acquire);
    const uint64_t published =
        producer->sample_cursor.load(std::memory_order_acquire);
    if (window_cursor < reclaim || cursor > published) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    LfmF32Span spans[2]{};
    uint32_t count = 0;
    const int resolved = capture_arena_spans(
        session->capture_arena, window_cursor, LFM_SESAME_FFT_SIZE, spans,
        &count);
    if (resolved != 0) return resolved;
    *out = {
        .first = spans[0].data,
        .first_count = static_cast<size_t>(spans[0].length),
        .second = count == 2 ? spans[1].data : nullptr,
        .second_count = count == 2
            ? static_cast<size_t>(spans[1].length)
            : 0,
    };
    return 0;
}

enum CapturePolicyProgress : int {
    CAPTURE_POLICY_EMPTY = 0,
    CAPTURE_POLICY_PROGRESS = 1,
    CAPTURE_POLICY_YIELD = 2,
};

int advance_capture_policy(LfmSession *session) {
    CapturePolicy &policy = session->capture_policy;
    if (!policy.chunk_pending) return CAPTURE_POLICY_EMPTY;
    const uint64_t end = policy.chunk.first_sample_cursor +
                         policy.chunk.frames;
    if (end < policy.chunk.first_sample_cursor) {
        return -EOVERFLOW;
    }
    if (policy.next_evidence_cursor > end) {
        policy.chunk = {};
        policy.chunk_pending = false;
        return CAPTURE_POLICY_PROGRESS;
    }
    if (policy.next_evidence_cursor - policy.segment_cursor <
        LFM_SESAME_FFT_SIZE) {
        if (!advance_capture_cadence(&policy, session->capture_rate)) {
            return -EOVERFLOW;
        }
        return CAPTURE_POLICY_PROGRESS;
    }

    LfmSesameWindowV1 window{};
    const int resolved = resolve_capture_window(
        session, policy.chunk, policy.next_evidence_cursor, &window);
    if (resolved != 0) return resolved;
    LfmSesameDecisionV1 decision{};
    const int detected = lfm_sesame_detector_process_window(
        policy.detector, LFM_SESAME_STREAM_MIC, &window, nullptr, 0,
        &decision);
    if (detected != 0) return detected;
    const int applied = apply_capture_decision(
        session, policy.next_evidence_cursor, decision);
    if (applied != 0) return applied;
    if (!advance_capture_cadence(&policy, session->capture_rate)) {
        return -EOVERFLOW;
    }
    if (policy.next_evidence_cursor > end) {
        policy.chunk = {};
        policy.chunk_pending = false;
    }
    return CAPTURE_POLICY_YIELD;
}

int process_capture_chunk(LfmSession *session,
                          const LfmCaptureChunkV1 &chunk) {
    LfmCaptureProducer *producer =
        session->chunk_producer.load(std::memory_order_acquire);
    if (!producer) return CAPTURE_POLICY_PROGRESS;
    if (chunk.stream_epoch !=
        session->epoch.load(std::memory_order_acquire)) {
        /* Chunk records are evidence within one retained capture lease, not
         * independently consumed captures. The sealed lease records the one
         * stale turn; counting every metadata fragment makes the public
         * snapshot depend on device callback size. */
        return CAPTURE_POLICY_PROGRESS;
    }
    if (session->capture_policy.segment_epoch != chunk.stream_epoch) {
        /* Epoch is a hard detector boundary. A callback admitted before an
         * interrupt may retire afterward, but neither its suffix nor the FFT
         * pre-roll behind the next callback may seed the new turn. */
        reset_capture_policy(session, chunk.first_sample_cursor, true);
    }
    if ((chunk.flags & LFM_CAPTURE_CHUNK_GAP) != 0) {
        const int cancelled = capture_supervision_cancel(
            session, KC_SCOPE_CAUSE_CANCELLED, false, false);
        if (cancelled != 0) return cancelled;
        const uint64_t cursor = chunk.first_sample_cursor + chunk.frames;
        if (cursor < chunk.first_sample_cursor) return -EOVERFLOW;
        if (session->capture_policy.turn_active) {
            session->capture_policy.discarded_silence_frames +=
                cursor - session->capture_policy.turn_start_cursor;
        }
        reset_capture_policy(session, cursor, true);
        const int reclaim = refresh_capture_reclaim(session);
        if (reclaim != 0) return reclaim;
        return CAPTURE_POLICY_PROGRESS;
    }
    if (session->capture_policy.chunk_pending ||
        chunk.sample_rate != session->capture_rate) {
        return LFM_STATUS_INTERNAL;
    }
    session->capture_policy.chunk = chunk;
    session->capture_policy.chunk_pending = true;
    return advance_capture_policy(session);
}

int step_capture_policy(LfmSession *session, uint32_t *budget) {
    if (session->capture_policy.chunk_pending) {
        return advance_capture_policy(session);
    }
    if (!budget || *budget == 0) return CAPTURE_POLICY_EMPTY;
    LfmCaptureChunkV1 chunk{};
    if (!capture_chunk_pop(&session->capture_chunks, &chunk)) {
        return CAPTURE_POLICY_EMPTY;
    }
    --*budget;
    return process_capture_chunk(session, chunk);
}

alignas(64) const float playback_zeros[LFM_SESAME_FFT_SIZE]{};

int playback_record_slot(LfmSession *session,
                         const PlaybackEvidenceRecord &record,
                         PcmSlot **out) {
    if ((record.flags & LFM_PLAYBACK_EVIDENCE_RENDERED) == 0 ||
        record.rendered_frames == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t index = 0;
    if (!decode_playback_lease_id(record.lease_id, &index) ||
        index >= session->playback.capacity) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PcmSlot &slot = session->playback.slots[index];
    const uint32_t state = slot.state.load(std::memory_order_acquire);
    if (slot.identity.load(std::memory_order_acquire) != record.lease_id ||
        slot.generation.load(std::memory_order_acquire) !=
            record.buffer_generation ||
        slot.stream_epoch != record.stream_epoch ||
        !ticket_equal(slot.ticket, record.ticket) ||
        (state != SLOT_CONSUMING && state != SLOT_RELEASING) ||
        record.source_offset_frames > slot.reserved_frames ||
        record.rendered_frames >
            slot.reserved_frames - record.source_offset_frames) {
        return LFM_STATUS_STALE;
    }
    *out = &slot;
    return 0;
}

int retire_playback_record(LfmSession *session,
                           const PlaybackEvidenceRecord &record) {
    if ((record.flags & LFM_PLAYBACK_EVIDENCE_RENDERED) == 0) return 0;
    PcmSlot *slot = nullptr;
    const int status = playback_record_slot(session, record, &slot);
    if (status != 0) return status;
    retire_slot_observer(slot, &session->playback_consumed);
    const uint32_t prior = session->playback_retained_observers.fetch_sub(
        1, std::memory_order_acq_rel);
    if (prior == 0) std::abort();
    return 0;
}

int retire_playback_history(LfmSession *session) {
    PlaybackEvidenceHistory &history = session->playback_policy.history;
    while (history.head != history.tail) {
        const PlaybackEvidenceRecord record =
            history.records[history.head % history.capacity];
        history.head++;
        const int status = retire_playback_record(session, record);
        if (status != 0) return status;
    }
    return 0;
}

bool advance_playback_cadence(PlaybackPolicy *policy,
                              uint32_t sample_rate) {
    const uint64_t scaled =
        static_cast<uint64_t>(policy->cadence_remainder) + sample_rate;
    const uint64_t step = scaled / 50;
    policy->cadence_remainder = static_cast<uint32_t>(scaled % 50);
    if (step == 0 ||
        policy->next_evidence_cursor > UINT64_MAX - step) {
        return false;
    }
    policy->next_evidence_cursor += step;
    return true;
}

int reset_playback_policy(LfmSession *session, uint64_t cursor,
                          bool count_discontinuity,
                          bool preserve_echo_tail) {
    PlaybackPolicy &policy = session->playback_policy;
    const int retired = retire_playback_history(session);
    if (retired != 0) return retired;
    const int reset = lfm_sesame_detector_discontinuity(
        policy.detector, LFM_SESAME_STREAM_PLAYBACK);
    if (reset != 0) return reset;
    policy.decision = {};
    policy.next_evidence_cursor = cursor;
    policy.last_evidence_cursor = cursor;
    policy.available_cursor = cursor;
    policy.cadence_remainder = 49;
    if (!preserve_echo_tail) {
        policy.echo_start_capture_cursor = 0;
        policy.last_voice_capture_cursor = 0;
        policy.echo_tail_capture_cursor = 0;
        policy.echo_epoch = 0;
        policy.echo_ticket = {};
    }
    if (count_discontinuity) policy.discontinuities++;
    return advance_playback_cadence(&policy, session->playback_rate)
               ? 0
               : -EOVERFLOW;
}

int append_playback_history(LfmSession *session,
                            const PlaybackEvidenceRecord &record) {
    PlaybackPolicy &policy = session->playback_policy;
    PlaybackEvidenceHistory &history = policy.history;
    if (history.tail - history.head == history.capacity) {
        return LFM_STATUS_INTERNAL;
    }
    if (record.first_playback_sample_cursor != policy.available_cursor ||
        record.rendered_frames == 0 ||
        record.first_playback_sample_cursor >
            UINT64_MAX - record.rendered_frames) {
        return LFM_STATUS_INTERNAL;
    }
    history.records[history.tail % history.capacity] = record;
    history.tail++;
    policy.available_cursor += record.rendered_frames;
    return 0;
}

struct PlaybackWindowBuilder {
    LfmSesameSpanV1 *spans = nullptr;
    size_t capacity = 0;
    size_t count = 0;
    size_t filled = 0;
};

bool append_playback_window_span(PlaybackWindowBuilder *builder,
                                 const float *data, size_t frames,
                                 bool zero) {
    if (frames == 0) return true;
    if (!builder || !builder->spans || !data ||
        builder->filled > LFM_SESAME_FFT_SIZE ||
        frames > LFM_SESAME_FFT_SIZE - builder->filled) {
        return false;
    }
    if (builder->count != 0) {
        LfmSesameSpanV1 &prior = builder->spans[builder->count - 1];
        if ((zero && prior.samples == playback_zeros) ||
            (!zero && prior.samples != playback_zeros &&
             prior.samples + prior.count == data)) {
            prior.count += frames;
            builder->filled += frames;
            return true;
        }
    }
    if (builder->count == builder->capacity) return false;
    builder->spans[builder->count] = {
        .samples = data,
        .count = frames,
    };
    ++builder->count;
    builder->filled += frames;
    return true;
}

int resolve_playback_window(LfmSession *session, uint64_t cursor,
                            LfmSesameSpanV1 *spans, size_t capacity,
                            size_t *out_count) {
    if (!spans || capacity == 0 || !out_count ||
        cursor > session->playback_policy.available_cursor) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out_count = 0;
    const uint64_t real_start =
        cursor > LFM_SESAME_FFT_SIZE ? cursor - LFM_SESAME_FFT_SIZE : 0;
    const size_t prefix = cursor < LFM_SESAME_FFT_SIZE
                              ? LFM_SESAME_FFT_SIZE - cursor
                              : 0;
    PlaybackWindowBuilder builder = {
        .spans = spans,
        .capacity = capacity,
    };
    if (!append_playback_window_span(&builder, playback_zeros, prefix, true)) {
        return LFM_STATUS_INTERNAL;
    }
    uint64_t expected = real_start;
    PlaybackEvidenceHistory &history = session->playback_policy.history;
    for (uint64_t cursor_index = history.head;
         cursor_index != history.tail && expected < cursor; ++cursor_index) {
        const PlaybackEvidenceRecord &record =
            history.records[cursor_index % history.capacity];
        const uint64_t end = record.first_playback_sample_cursor +
                             record.rendered_frames;
        if (end <= real_start) continue;
        const uint64_t start =
            std::max(record.first_playback_sample_cursor, real_start);
        if (start != expected || end < start) return LFM_STATUS_INTERNAL;
        const uint64_t selected_end = std::min(end, cursor);
        const size_t frames = static_cast<size_t>(selected_end - start);
        const bool zero =
            (record.flags & LFM_PLAYBACK_EVIDENCE_SILENCE) != 0;
        const float *data = playback_zeros;
        if (!zero) {
            PcmSlot *slot = nullptr;
            const int status = playback_record_slot(session, record, &slot);
            if (status != 0) return status;
            const uint64_t offset = record.source_offset_frames +
                                    (start - record.first_playback_sample_cursor);
            if (offset > slot->reserved_frames ||
                frames > slot->reserved_frames - offset) {
                return LFM_STATUS_INTERNAL;
            }
            data = slot->samples + offset;
        }
        if (!append_playback_window_span(&builder, data, frames, zero)) {
            return LFM_STATUS_INTERNAL;
        }
        expected = selected_end;
    }
    if (expected != cursor || builder.count == 0 ||
        builder.filled != LFM_SESAME_FFT_SIZE) {
        return LFM_STATUS_INTERNAL;
    }
    *out_count = builder.count;
    return 0;
}

int trim_playback_history(LfmSession *session) {
    PlaybackPolicy &policy = session->playback_policy;
    PlaybackEvidenceHistory &history = policy.history;
    const uint64_t keep = policy.next_evidence_cursor > LFM_SESAME_FFT_SIZE
                              ? policy.next_evidence_cursor -
                                    LFM_SESAME_FFT_SIZE
                              : 0;
    while (history.head != history.tail) {
        const PlaybackEvidenceRecord &record =
            history.records[history.head % history.capacity];
        const uint64_t end = record.first_playback_sample_cursor +
                             record.rendered_frames;
        if (end > keep) break;
        const PlaybackEvidenceRecord retired = record;
        history.head++;
        const int status = retire_playback_record(session, retired);
        if (status != 0) return status;
    }
    return 0;
}

enum PlaybackPolicyProgress : int {
    PLAYBACK_POLICY_EMPTY = 0,
    PLAYBACK_POLICY_PROGRESS = 1,
};

int step_playback_policy(LfmSession *session, uint32_t *budget) {
    PlaybackPolicy &policy = session->playback_policy;
    if (policy.next_evidence_cursor <= policy.available_cursor) {
        LfmSesameSpanV1 spans[LFM_SESAME_FFT_SIZE];
        size_t span_count = 0;
        const int resolved = resolve_playback_window(
            session, policy.next_evidence_cursor, spans,
            LFM_SESAME_FFT_SIZE, &span_count);
        if (resolved != 0) return resolved;
        LfmSesameDecisionV1 decision{};
        int detected = 0;
        if (span_count <= 2) {
            const LfmSesameWindowV1 window = {
                .first = spans[0].samples,
                .first_count = spans[0].count,
                .second = span_count == 2 ? spans[1].samples : nullptr,
                .second_count = span_count == 2 ? spans[1].count : 0,
            };
            detected = lfm_sesame_detector_process_window(
                policy.detector, LFM_SESAME_STREAM_PLAYBACK, &window,
                nullptr, 0, &decision);
        } else {
            const LfmSesameScatterWindowV1 window = {
                .spans = spans,
                .span_count = span_count,
            };
            detected = lfm_sesame_detector_process_scatter_window(
                policy.detector, LFM_SESAME_STREAM_PLAYBACK, &window,
                nullptr, 0, &decision);
        }
        if (detected != 0) return detected;
        if (decision.voice != 0) {
            const uint64_t tail = capture_duration_frames(
                session->capture_rate, CAPTURE_ECHO_TAIL_MS);
            if (policy.last_capture_cursor > UINT64_MAX - tail) {
                return -EOVERFLOW;
            }
            const uint64_t until = policy.last_capture_cursor + tail;
            if (policy.echo_epoch != policy.last_epoch ||
                !ticket_equal(policy.echo_ticket, policy.last_ticket)) {
                policy.echo_start_capture_cursor =
                    policy.last_capture_cursor;
                policy.echo_tail_capture_cursor = until;
            } else {
                policy.echo_tail_capture_cursor = std::max(
                    policy.echo_tail_capture_cursor, until);
            }
            policy.last_voice_capture_cursor = policy.last_capture_cursor;
            policy.echo_epoch = policy.last_epoch;
            policy.echo_ticket = policy.last_ticket;
        }
        policy.decision = decision;
        policy.last_evidence_cursor = policy.next_evidence_cursor;
        policy.evidence_updates++;
        session->playback_evidence_updates.store(
            policy.evidence_updates, std::memory_order_relaxed);
        session->playback_evidence_cursor.store(
            policy.last_evidence_cursor, std::memory_order_relaxed);
        session->playback_evidence_voice.store(decision.voice,
                                               std::memory_order_relaxed);
        if (!advance_playback_cadence(&policy, session->playback_rate)) {
            return -EOVERFLOW;
        }
        const int trimmed = trim_playback_history(session);
        if (trimmed != 0) return trimmed;
        return PLAYBACK_POLICY_PROGRESS;
    }
    if (!budget || *budget == 0) {
        return playback_evidence_empty(policy.incoming)
                   ? PLAYBACK_POLICY_EMPTY
                   : PLAYBACK_POLICY_PROGRESS;
    }
    PlaybackEvidenceRecord record{};
    if (!playback_evidence_pop(&policy.incoming, &record)) {
        session->playback_evidence_backlog.store(0,
                                                std::memory_order_release);
        return PLAYBACK_POLICY_EMPTY;
    }
    --*budget;
    if (record.session_id != session->id ||
        record.sample_rate != session->playback_rate ||
        record.first_playback_sample_cursor >
            UINT64_MAX - record.rendered_frames) {
        (void)retire_playback_record(session, record);
        return LFM_STATUS_INTERNAL;
    }
    policy.evidence_records++;
    policy.last_ticket = record.ticket;
    policy.last_epoch = record.stream_epoch;
    policy.last_capture_cursor = record.capture_sample_cursor_snapshot;
    session->playback_evidence_records.store(
        policy.evidence_records, std::memory_order_relaxed);
    const uint64_t current_epoch =
        session->epoch.load(std::memory_order_acquire);
    const bool control =
        (record.flags & (LFM_PLAYBACK_EVIDENCE_FLUSH |
                         LFM_PLAYBACK_EVIDENCE_DISCONTINUITY)) != 0;
    if (record.stream_epoch != current_epoch || control) {
        const int retired = retire_playback_record(session, record);
        if (retired != 0) return retired;
        const uint64_t cursor = record.first_playback_sample_cursor +
                                record.rendered_frames;
        const bool preserve_echo_tail =
            record.stream_epoch == current_epoch &&
            (record.flags & LFM_PLAYBACK_EVIDENCE_FLUSH) != 0;
        const int reset = reset_playback_policy(
            session, cursor, true, preserve_echo_tail);
        if (reset != 0) return reset;
        return PLAYBACK_POLICY_PROGRESS;
    }
    const int appended = append_playback_history(session, record);
    if (appended != 0) {
        (void)retire_playback_record(session, record);
        return appended;
    }
    session->playback_evidence_backlog.store(
        static_cast<uint32_t>(policy.history.tail - policy.history.head),
        std::memory_order_release);
    return PLAYBACK_POLICY_PROGRESS;
}

void publish_playback_policy_snapshot(LfmSession *session) {
    PlaybackPolicy &policy = session->playback_policy;
    const uint32_t current = policy.published_snapshot.load(
        std::memory_order_relaxed);
    const uint32_t target = current ^ 1u;
    PlaybackPolicySnapshotSlot &slot = policy.snapshots[target];
    uint32_t expected = 0;
    if (!slot.state.compare_exchange_strong(
            expected, CAPTURE_SNAPSHOT_WRITING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    const uint64_t incoming_head = policy.incoming.head.value.load(
        std::memory_order_acquire);
    const uint64_t incoming_tail = policy.incoming.tail.value.load(
        std::memory_order_acquire);
    slot.value = {
        .size = sizeof(LfmPlaybackPolicySnapshotV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .sample_rate = session->playback_rate,
        .last_voice = policy.decision.voice,
        .detector_backlog = static_cast<uint32_t>(
            std::min<uint64_t>(UINT32_MAX,
                               incoming_tail - incoming_head +
                                   policy.history.tail - policy.history.head)),
        .retained_observers = session->playback_retained_observers.load(
            std::memory_order_acquire),
        .evidence_records = policy.evidence_records,
        .evidence_updates = policy.evidence_updates,
        .last_evidence_cursor = policy.last_evidence_cursor,
        .discontinuities = policy.discontinuities,
        .stream_epoch = policy.last_epoch,
        .ticket = policy.last_ticket,
        .capture_sample_cursor_snapshot = policy.last_capture_cursor,
        .last_score = policy.decision.score,
        .adaptive_min = policy.decision.adaptive_min,
        .adaptive_max = policy.decision.adaptive_max,
        .echo_start_capture_cursor = policy.echo_start_capture_cursor,
        .last_voice_capture_cursor = policy.last_voice_capture_cursor,
        .echo_tail_capture_cursor = policy.echo_tail_capture_cursor,
        .barge_voiced_frames = session->capture_policy.barge_voiced_frames,
        .barge_interrupts = session->capture_policy.barge_interrupts,
        .barge_source_epoch =
            session->capture_policy.barge_source_epoch,
        .barge_interrupt_epoch =
            session->capture_policy.barge_interrupt_epoch,
        .barge_playback_ticket =
            session->capture_policy.barge_playback_ticket,
        .reserved = {},
    };
    slot.state.store(0, std::memory_order_release);
    policy.published_snapshot.store(target, std::memory_order_release);
}

int read_playback_policy_snapshot(
    const LfmSession *session, LfmPlaybackPolicySnapshotV1 *out) {
    const PlaybackPolicy &policy = session->playback_policy;
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        const uint32_t index = policy.published_snapshot.load(
            std::memory_order_acquire);
        const PlaybackPolicySnapshotSlot &slot = policy.snapshots[index];
        uint32_t readers = slot.state.load(std::memory_order_acquire);
        if ((readers & CAPTURE_SNAPSHOT_WRITING) != 0 ||
            readers == CAPTURE_SNAPSHOT_WRITING - 1) {
            continue;
        }
        if (!slot.state.compare_exchange_strong(
                readers, readers + 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        if (policy.published_snapshot.load(std::memory_order_acquire) !=
            index) {
            slot.state.fetch_sub(1, std::memory_order_release);
            continue;
        }
        *out = slot.value;
        slot.state.fetch_sub(1, std::memory_order_release);
        return 0;
    }
    return LFM_STATUS_WOULD_BLOCK;
}

void flush_published(LfmSession *session) {
    PlaybackPool *pool = &session->playback;
    for (uint32_t i = 0; i < pool->capacity; ++i) {
        PcmSlot &slot = pool->slots[i];
        uint32_t expected = SLOT_PUBLISHED;
        if (slot.state.compare_exchange_strong(expected, SLOT_RELEASING,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
            (void)finalize_slot(&slot, &session->playback_consumed);
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
    if (rc == 0) {
        /* This continuation owns available_cursor. Callback publication may
         * have advanced the hardware cursor while its SPSC record is still
         * pending; resetting to that producer cursor would jump over the FIFO
         * head. Old-epoch records advance/reset themselves when drained. */
        const uint64_t cursor = session->playback_policy.available_cursor;
        const int reset = reset_playback_policy(session, cursor, true, false);
        if (reset == 0) return true;
        stage_error(session, reset, "native playback epoch reset failed");
        return false;
    }
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

void process_capture_range(LfmSession *session,
                           const CaptureRangeLease &range) {
    const uint64_t current_epoch =
        session->epoch.load(std::memory_order_acquire);
    if (range.stream_epoch != current_epoch) {
        session->capture_stale.fetch_add(1, std::memory_order_relaxed);
        (void)release_capture_range(session, range);
        return;
    }
    session->capture_consumed.fetch_add(1, std::memory_order_relaxed);
    if (session->dock_only) {
        (void)release_capture_range(session, range);
        const EventRecord records[2] = {
            make_event(LFM_EVENT_TURN_STARTED, current_epoch, range.ticket,
                       0, nullptr, 0),
            make_turn(current_epoch, range.ticket, 0, 0, 0, 0),
        };
        (void)stage_results(session, records, 2);
        return;
    }

    LfmF32Span spans[2]{};
    uint32_t span_count = 0;
    int rc = capture_arena_spans(
        session->capture_arena, range.first_sample_cursor, range.frames,
        spans, &span_count);
    if (rc != 0) {
        (void)release_capture_range(session, range);
        stage_error(session, rc, "native capture range resolve failed");
        return;
    }
    SessionAction *action = prepare_action(
        session, current_epoch, range.ticket, true);
    if (!action) {
        (void)release_capture_range(session, range);
        return;
    }
    action->capture_range = range;
    action->capture_range_active = true;
    rc = lfm_conversation_begin_pcm_spans_submit_native(
        session->conversation, spans, span_count, range.sample_rate,
        &action->emission, route_notify, session, &action->admission);
    if (rc != 0) {
        action->admission_pending = false;
        fail_action(session, rc, "native PCM range admission failed");
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
        session, current_epoch, command.ticket, false);
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

void process_command(LfmSession *session, const TextCommand &command) {
    process_text(session, command);
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
    uint32_t capture_budget = ACTION_CAPTURE_DRAIN_BUDGET;
    uint32_t playback_budget = ACTION_PLAYBACK_EVIDENCE_DRAIN_BUDGET;
    for (uint32_t quantum = 0; quantum < SESSION_STEP_BUDGET; ++quantum) {
        /* Playback evidence is an input to turn policy, not outward telemetry.
         * Drain its callback-published metadata before any action, supervision,
         * or microphone decision can observe a later state. */
        const int playback = step_playback_policy(session, &playback_budget);
        if (playback < 0) {
            stage_error(session, playback,
                        "native playback detector failed");
            continue;
        }
        if (playback == PLAYBACK_POLICY_PROGRESS) continue;
        const int supervision = step_capture_supervision(session);
        if (supervision < 0) {
            stage_error(session, supervision,
                        "native capture supervision failed");
            continue;
        }
        if (supervision == CAPTURE_SUPERVISION_PROGRESS) continue;
        if (supervision == CAPTURE_SUPERVISION_STOPPING) {
            return SESSION_IDLE;
        }
        /* Once a dual gate wins, no later detector cadence may reinterpret a
         * buffered suffix before the exact committed boundary is mounted.
         * The freeze successor is a callback/capacity edge; the coordinator
         * becomes dormant rather than draining more capture metadata. */
        if (session->capture_supervision.freeze_pending) {
            const int freeze = freeze_capture_turn(session);
            if (freeze == CAPTURE_FREEZE_PROGRESS) continue;
            if (freeze == CAPTURE_FREEZE_WRITER ||
                freeze == CAPTURE_FREEZE_CAPACITY) {
                return SESSION_IDLE;
            }
            if (freeze < 0) {
                request_stop(session, freeze);
                continue;
            }
            request_stop(session, LFM_STATUS_INTERNAL);
            continue;
        }
        const ResultProgress result = drain_result(session);
        if (result == RESULT_BLOCKED) {
            if (!session->stop.load(std::memory_order_acquire)) {
                const int capture =
                    step_capture_policy(session, &capture_budget);
                if (capture == CAPTURE_POLICY_YIELD ||
                    capture == CAPTURE_POLICY_PROGRESS) {
                    return SESSION_READY;
                }
                if (capture < 0) {
                    request_stop(session, capture);
                    return SESSION_READY;
                }
                const int recycle = recycle_background_silence(session);
                if (recycle == CAPTURE_FREEZE_PROGRESS) return SESSION_READY;
                if (recycle < 0) {
                    request_stop(session, recycle);
                    return SESSION_READY;
                }
                const int freeze = freeze_capture_turn(session);
                if (freeze == CAPTURE_FREEZE_PROGRESS) return SESSION_READY;
                if (freeze < 0) {
                    request_stop(session, freeze);
                    return SESSION_READY;
                }
            }
            return SESSION_BLOCKED_RESULT;
        }
        if (result == RESULT_DRAINED) continue;

        if (session->capture_supervision.device_loss_ready) {
            CaptureSupervision &capture = session->capture_supervision;
            const uint64_t epoch = capture.device_loss_epoch;
            capture.device_loss_ready = false;
            capture.device_loss_parent = {};
            capture.device_loss_epoch = 0;
            static constexpr char lost[] = "capture-device-lost";
            (void)stage_event(
                session, LFM_EVENT_ERROR, epoch,
                next_ticket(session, LFM_TICKET_CONTROL),
                LFM_STATUS_CANCELLED, lost, sizeof(lost) - 1, 0, false,
                LFM_STATUS_CANCELLED);
            continue;
        }

        if (session->action.active) {
            /* Capture policy remains live while a reply is being generated.
             * Bounded metadata drain precedes recurrence so a callback edge can
             * detect barge-in and request an epoch transition. A completed
             * capture turn may publish an immutable arena range, but it is not admitted
             * as a second mutating action until the current action retires. */
            if (capture_budget != 0) {
                const int capture =
                    step_capture_policy(session, &capture_budget);
                if (capture == CAPTURE_POLICY_YIELD) return SESSION_READY;
                if (capture == CAPTURE_POLICY_PROGRESS) continue;
                if (capture < 0) {
                    stage_error(session, capture,
                                "native capture detector failed");
                    continue;
                }
            }
            const int recycle = recycle_background_silence(session);
            if (recycle == CAPTURE_FREEZE_PROGRESS) continue;
            if (recycle < 0) {
                stage_error(session, recycle,
                            "native silent capture recycle failed");
                continue;
            }
            const int retire = retire_closed_capture_producer(session);
            if (retire == CAPTURE_RETIRE_PROGRESS) continue;
            if (retire < 0) {
                stage_error(session, retire,
                            "native capture endpoint retirement failed");
                continue;
            }
            const int freeze = freeze_capture_turn(session);
            if (freeze == CAPTURE_FREEZE_PROGRESS) continue;
            if (freeze < 0) {
                stage_error(session, freeze,
                            "native capture range publication failed");
                continue;
            }
            const ActionProgress action = advance_action(session);
            if (session->result.active || action == ACTION_PROGRESS) continue;
            if (!capture_chunk_empty(session->capture_chunks)) {
                return SESSION_READY;
            }
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
            session->capture_policy.chunk = {};
            session->capture_policy.chunk_pending = false;
            /* Closing and draining are one Rube-Goldberg transition: once the
             * packed gate is CLOSED|0, no producer can add another record.
             * Until then this retained state goes dormant and the releasing
             * publisher provides its sole successor edge. Check before the
             * final queue scan so a just-published record cannot be skipped. */
            if (session->publication_gate.value.load(
                    std::memory_order_acquire) != PUBLICATION_CLOSED) {
                return SESSION_IDLE;
            }
            LfmCaptureChunkV1 chunk{};
            if (capture_chunk_pop(&session->capture_chunks, &chunk)) {
                continue;
            }
            const int retire = retire_closed_capture_producer(session);
            if (retire == CAPTURE_RETIRE_PROGRESS) continue;
            if (retire < 0) {
                int32_t expected = 0;
                session->terminal_status.compare_exchange_strong(
                    expected, retire, std::memory_order_acq_rel,
                    std::memory_order_acquire);
                continue;
            }
            if (session->command_pending) {
                const TextCommand command = session->pending_command;
                session->pending_command = {};
                session->command_pending = false;
                (void)stage_turn(session, command.epoch, command.ticket, 0, 0,
                                 0, LFM_STATUS_CANCELLED);
                continue;
            }
            if (session->range_pending) {
                const CaptureRangeLease range = session->pending_range;
                session->pending_range = {};
                session->range_pending = false;
                (void)release_capture_range(session, range);
                continue;
            }
            TextCommand command{};
            if (text_pop(&session->commands, &command)) {
                (void)stage_turn(session, command.epoch, command.ticket, 0, 0,
                                 0, LFM_STATUS_CANCELLED);
                continue;
            }
            CaptureRangeLease range{};
            if (take_capture_range(session, &range) == 0) {
                (void)release_capture_range(session, range);
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
            if (!playback_evidence_empty(
                    session->playback_policy.incoming)) {
                return SESSION_READY;
            }
            const int playback_retired = retire_playback_history(session);
            if (playback_retired != 0) {
                int32_t expected = 0;
                session->terminal_status.compare_exchange_strong(
                    expected, playback_retired, std::memory_order_acq_rel,
                    std::memory_order_acquire);
                continue;
            }
            flush_published(session);
            const bool first_terminal = !session->event_done.exchange(
                true, std::memory_order_acq_rel);
            if (first_terminal) (void)notify_delivery(session);

            /* STOPPED is the host's correlated device-teardown request, not
             * permission to abandon native ownership. A native-originated
             * failure can reach this point while the hardware endpoints still
             * retain capture ranges or a playback lease. Their close/release
             * operations publish the successor edge that re-enters this
             * continuation. Until then the coordinator is dormant but alive. */
            if (session->chunk_producer.load(std::memory_order_acquire) ||
                session->capture_producers.load(std::memory_order_acquire) != 0 ||
                session->playback_consumers.load(std::memory_order_acquire) != 0 ||
                session->playback_retained_observers.load(
                    std::memory_order_acquire) != 0 ||
                capture_range_live(session->capture_arena) != 0 ||
                !capture_range_empty(session->capture_arena.ready) ||
                pool_live(session->playback) != 0) {
                return SESSION_IDLE;
            }
            session->coordinator_phase = COORDINATOR_DONE;
            kc_service_request_stop(session->coordinator);
            {
                std::lock_guard<std::mutex> guard(session->lifecycle_mutex);
                session->coordinator_done = true;
            }
            session->lifecycle_cv.notify_all();
            return SESSION_DONE;
        }

        if (session->coordinator_phase == COORDINATOR_STARTING) {
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

        const int capture = step_capture_policy(session, &capture_budget);
        if (capture == CAPTURE_POLICY_YIELD) return SESSION_READY;
        if (capture == CAPTURE_POLICY_PROGRESS) continue;
        if (capture < 0) {
            stage_error(session, capture, "native capture detector failed");
            continue;
        }
        const int recycle = recycle_background_silence(session);
        if (recycle == CAPTURE_FREEZE_PROGRESS) continue;
        if (recycle == CAPTURE_FREEZE_WRITER) return SESSION_IDLE;
        if (recycle < 0) {
            stage_error(session, recycle,
                        "native silent capture recycle failed");
            continue;
        }
        const int retire = retire_closed_capture_producer(session);
        if (retire == CAPTURE_RETIRE_PROGRESS) continue;
        if (retire < 0) {
            stage_error(session, retire,
                        "native capture endpoint retirement failed");
            continue;
        }
        const int freeze = freeze_capture_turn(session);
        if (freeze == CAPTURE_FREEZE_PROGRESS) continue;
        if (freeze == CAPTURE_FREEZE_WRITER) return SESSION_IDLE;
        if (freeze < 0) {
            stage_error(session, freeze, "native capture range publication failed");
            continue;
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

        if (session->range_pending) {
            if (!synchronize_epoch(session)) continue;
            process_capture_range(session, session->pending_range);
            session->pending_range = {};
            session->range_pending = false;
            continue;
        }
        const int range = take_capture_range(
            session, &session->pending_range);
        if (range == 0) {
            session->range_pending = true;
            continue;
        }
        if (range != LFM_STATUS_WOULD_BLOCK) {
            stage_error(session, range, "native capture range claim failed");
            continue;
        }

        return SESSION_IDLE;
    }
    return SESSION_READY;
}

void coordinator_main(void *context) {
    LfmSession *session = static_cast<LfmSession *>(context);
    /* kc_service_start publishes its retained continuation once so owner
     * initialization can run. A pre-created realtime endpoint can also race a
     * notification into the narrow interval between service start and the
     * readiness publication. Neither edge may enter numerical/session work
     * until start publishes RUNNING; the explicit post-publication notify is
     * their only successor. Stop is admitted so a partially-started service
     * can still retire after a start failure. */
    if (session->state.load(std::memory_order_acquire) !=
            LFM_SESSION_RUNNING &&
        !session->stop.load(std::memory_order_acquire)) {
        return;
    }
    const SessionProgress progress = session_step(session);
    publish_playback_policy_snapshot(session);
    publish_capture_supervision_snapshot(session);
    if (progress != SESSION_READY) return;
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
    kc_service_request_stop(session->delivery);
    {
        std::lock_guard<std::mutex> guard(session->lifecycle_mutex);
        session->delivery_done = true;
    }
    session->lifecycle_cv.notify_all();
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
    if (session->state.load(std::memory_order_acquire) !=
            LFM_SESSION_RUNNING &&
        !session->stop.load(std::memory_order_acquire)) {
        return;
    }
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

void unregister_session_locked(LfmRuntime *runtime, LfmSession *session) {
    for (uint32_t i = 0; i < runtime->session_capacity; ++i) {
        if (runtime->sessions[i] == session) {
            runtime->sessions[i] = nullptr;
            runtime->session_count--;
            return;
        }
    }
}

void unregister_session(LfmRuntime *runtime, LfmSession *session) {
    std::lock_guard<std::mutex> guard(runtime->children_mutex);
    unregister_session_locked(runtime, session);
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

static int runtime_create_impl(const LfmRuntimeConfigV1 *config,
                               LfmRuntime **out, bool manual_deadlines) {
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
    runtime->manual_deadlines = manual_deadlines;
    int engine_status = 0;
    runtime->engine = manual_deadlines
        ? lfm_internal_engine_new_manual_deadlines_for_test(
              static_cast<int>(config->kernel_lanes))
        : lfm_engine_new_status(static_cast<int>(config->kernel_lanes),
                                &engine_status);
    if (!runtime->engine) {
        delete runtime;
        if (engine_status == -ENOTSUP) return LFM_STATUS_UNSUPPORTED;
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

int lfm_runtime_create(const LfmRuntimeConfigV1 *config, LfmRuntime **out) {
    return runtime_create_impl(config, out, false);
}

/* Private deterministic construction for non-Apple lifecycle tests. The
 * product ABI never exposes this symbol and production callers cannot select
 * a clock that only advances under test control. */
int lfm_internal_runtime_create_manual_deadlines_for_test(
    const LfmRuntimeConfigV1 *config, LfmRuntime **out) {
    return runtime_create_impl(config, out, true);
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
    const bool requested_manual_deadlines =
        (config->flags & LFM_SESSION_FLAG_MANUAL_DEADLINES) != 0;
    const bool manual_deadlines =
        runtime->manual_deadlines || requested_manual_deadlines;
    if (runtime->state.load(std::memory_order_acquire) >= LFM_RUNTIME_STOPPING ||
        (config->flags & ~(LFM_SESSION_FLAG_DOCK_ONLY |
                           LFM_SESSION_FLAG_MANUAL_DEADLINES)) != 0 ||
        (requested_manual_deadlines && !dock_only) ||
        config->playback_slots == 0 || config->playback_slots > MAX_PCM_SLOTS ||
        config->capture_max_callback_frames == 0 ||
        (dock_only && config->playback_frames_per_slot == 0) ||
        config->pcm_channels != 1 ||
        config->capture_sample_rate < 8000 ||
        config->capture_sample_rate > 192000 ||
        config->playback_sample_rate < 8000 ||
        config->playback_sample_rate > 192000 ||
        config->command_capacity == 0 || config->command_capacity > 64 ||
        config->max_new_tokens == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (dock_only && (model || conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (!dock_only && (!model || !conversation)) return LFM_STATUS_INVALID_ARGUMENT;
    if (callbacks && (callbacks->size != sizeof(*callbacks) ||
                      callbacks->abi_version != LFM_RUNTIME_ABI_VERSION)) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    const uint64_t cadence_frames =
        (static_cast<uint64_t>(config->capture_sample_rate) + 49) / 50;
    const uint64_t callback_frames = config->capture_max_callback_frames;
    const uint64_t forced_frames =
        static_cast<uint64_t>(config->capture_sample_rate) * 30;
    /* One already-admitted model range and one complete following turn may be
     * live together. Two detector/callback guards cover onset evidence and the
     * indivisible callback that crosses a boundary; those guards are arena
     * residency only and can never enlarge a model-admissible turn. */
    const uint64_t arena_frames =
        2 * forced_frames + 2 * cadence_frames + 2 * callback_frames;
    if (callback_frames == 0 || callback_frames > UINT32_MAX ||
        forced_frames > UINT32_MAX || arena_frames > UINT32_MAX ||
        arena_frames > SIZE_MAX / sizeof(float)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    size_t capture_prepare_samples = 0;
    size_t bounded = 0;
    if (!checked_samples(static_cast<uint32_t>(forced_frames),
                         config->pcm_channels, &bounded)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    capture_prepare_samples = bounded;
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
            conversation, capture_prepare_samples,
            config->capture_sample_rate,
            config->playback_sample_rate, &playback_frames);
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
    session->manual_deadlines = manual_deadlines;
    session->id = config->session_id == 0
                      ? next_session_id.fetch_add(1, std::memory_order_relaxed)
                      : config->session_id;
    if (session->id == 0) session->id = next_session_id.fetch_add(1);
    session->capture_rate = config->capture_sample_rate;
    session->capture_callback_frames =
        static_cast<uint32_t>(callback_frames);
    session->capture_turn_frames =
        static_cast<uint32_t>(forced_frames);
    session->playback_rate = config->playback_sample_rate;
    session->playback_frames = static_cast<uint32_t>(playback_frames);
    session->channels = config->pcm_channels;
    session->max_new_tokens = config->max_new_tokens;
    if (callbacks) session->callbacks = *callbacks;
    session->events.capacity = runtime->event_capacity;
    session->events.records = new (std::nothrow) EventRecord[runtime->event_capacity];
    session->commands.capacity = config->command_capacity;
    session->commands.ring =
        new (std::nothrow) TextRecordCell[config->command_capacity];
    session->capture_arena.capacity_frames = arena_frames;
    session->capture_arena.samples =
        new (std::nothrow) float[static_cast<size_t>(arena_frames)];
    session->capture_chunks.records =
        new (std::nothrow) LfmCaptureChunkV1[CAPTURE_CHUNK_CAPACITY];
    session->capture_chunks.capacity = CAPTURE_CHUNK_CAPACITY;
    session->playback_policy.incoming.records =
        new (std::nothrow) PlaybackEvidenceRecord[PLAYBACK_EVIDENCE_CAPACITY];
    session->playback_policy.incoming.capacity = PLAYBACK_EVIDENCE_CAPACITY;
    session->playback_policy.history.records =
        new (std::nothrow) PlaybackEvidenceRecord[PLAYBACK_EVIDENCE_CAPACITY];
    session->playback_policy.history.capacity = PLAYBACK_EVIDENCE_CAPACITY;
    int rc = session->events.records && session->commands.ring &&
                     session->capture_arena.samples &&
                     session->capture_chunks.records &&
                     session->playback_policy.incoming.records &&
                     session->playback_policy.history.records
                 ? 0
                 : LFM_STATUS_OUT_OF_MEMORY;
    if (rc == 0) {
        rc = lfm_sesame_detector_create(
            session->capture_rate, &session->capture_policy.detector);
    }
    if (rc == 0) reset_capture_policy(session, 0, false);
    if (rc == 0) {
        rc = lfm_sesame_detector_create(
            session->playback_rate, &session->playback_policy.detector);
    }
    if (rc == 0 &&
        !advance_playback_cadence(&session->playback_policy,
                                  session->playback_rate)) {
        rc = -EOVERFLOW;
    }
    if (rc == 0) publish_playback_policy_snapshot(session);
    if (rc == 0) {
        for (uint32_t index = 0; index < config->command_capacity; ++index) {
            session->commands.ring[index].sequence.store(
                static_cast<uint64_t>(index) * 2,
                std::memory_order_relaxed);
        }
    }
    if (rc == 0) {
        rc = pool_create(&session->playback, config->playback_slots,
                         static_cast<uint32_t>(playback_samples));
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
    bool registered = false;
    if (rc == 0) {
        registered = register_session_locked(runtime, session);
        if (!registered) rc = LFM_STATUS_BUSY;
    }
    /* Scope roles, native deadline sources, and every GCD timer are readiness
     * storage. Construct and seal them while the session is still private and
     * CREATED; start is consequently an allocation-free state publication. */
    if (rc == 0) rc = capture_supervision_create(session);
    if (rc != 0 && registered) {
        unregister_session_locked(runtime, session);
        registered = false;
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
    if (session->state.load(std::memory_order_acquire) !=
        LFM_SESSION_CREATED) {
        return LFM_STATUS_BUSY;
    }
    int rc = kc_service_start(session->delivery);
    if (rc != 0) return rc;
    session->delivery_started = true;
    rc = kc_service_start(session->coordinator);
    if (rc != 0) {
        request_stop(session, rc);
        kc_deadline_source_request_stop(
            session->capture_supervision.source);
        session->capture_supervision.source_stop_requested = true;
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
    /* Starting a retained service only publishes its owner-initialization
     * continuation; with zero notifications its callback remains dormant.
     * Both services and every setup allocation are therefore complete before
     * this release makes product work admissible. */
    session->state.store(LFM_SESSION_RUNNING, std::memory_order_release);
    notify_session(session);
    return 0;
}

int lfm_session_submit_text(LfmSession *session, const char *utf8,
                            size_t utf8_bytes, LfmTicketIdV1 *out_ticket) {
    return submit_text(session, utf8, utf8_bytes, out_ticket);
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
    /* Absolute sample storage is not rotated on interrupt. The epoch edge
     * cancels the policy scope; stale chunk records are discarded by identity,
     * and reader-floor reclamation advances only after their owners retire. */
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
    bool unstarted = false;
    bool stop_source = false;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        const uint32_t state = session->state.load(std::memory_order_acquire);
        /* Callback endpoints are lifetime leases over the session and its
         * notifier pointers. Teardown must reject before retiring either
         * retained service; checking live PCM cells afterward is too late for a
         * device callback concurrently publishing the release edge. */
        if (session->capture_producers.load(std::memory_order_acquire) != 0 ||
            session->playback_consumers.load(std::memory_order_acquire) != 0 ||
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
            unstarted = true;
            if (session->capture_supervision.source &&
                !session->capture_supervision.source_stop_requested) {
                session->capture_supervision.source_stop_requested = true;
                stop_source = true;
            }
        }
        if (!session->stop.load(std::memory_order_acquire) &&
            state != LFM_SESSION_CREATED) {
            return LFM_STATUS_BUSY;
        }
    }

    if (stop_source) {
        kc_deadline_source_request_stop(
            session->capture_supervision.source);
    }

    if (unstarted) {
        const int retire = retire_unstarted_capture_producer(session);
        if (retire < 0) return retire;
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
    if (session->capture_supervision.source_stop_requested) {
        const int status = kc_deadline_source_join(
            session->capture_supervision.source);
        if (status != 0) return status;
    }
    if (session->capture_supervision.source) {
        const int status = kc_deadline_source_destroy(
            session->capture_supervision.source);
        if (status != 0) return status;
        session->capture_supervision.source = nullptr;
    }
    if (session->capture_supervision.scope) {
        const int status = kc_fixed_scope_destroy(
            session->capture_supervision.scope);
        if (status != 0) return status;
        session->capture_supervision.scope = nullptr;
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
    if (pool_live(session->playback) != 0) {
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
        .live_playback_leases = pool_live(session->playback),
        .reliable_event_depth = event_depth(session->events),
        .reliable_event_capacity = session->events.capacity,
        .reserved = {},
    };
    return 0;
}

int lfm_session_capture_policy_snapshot(
    const LfmSession *session, LfmCapturePolicySnapshotV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    if (out->size != sizeof(*out) ||
        out->abi_version != LFM_CAPTURE_POLICY_ABI) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    const uint32_t state = session->state.load(std::memory_order_acquire);
    if (state != LFM_SESSION_JOINED) {
        /* A live diagnostic exposes only the release-published progress edge;
         * the compact classifier state remains coordinator-owned and lockless. */
        *out = {
            .size = sizeof(*out),
            .abi_version = LFM_CAPTURE_POLICY_ABI,
            .sample_rate = session->capture_rate,
            .evidence_updates = session->capture_evidence_updates.load(
                std::memory_order_acquire),
            .last_evidence_cursor = session->capture_evidence_cursor.load(
                std::memory_order_relaxed),
        };
        return 0;
    }
    const CapturePolicy &policy = session->capture_policy;
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_CAPTURE_POLICY_ABI,
        .sample_rate = session->capture_rate,
        .state = policy.state,
        .last_voice = policy.decision.voice,
        .detector_backlog = policy.chunk_pending ? 1u : 0u,
        .evidence_updates = policy.evidence_updates,
        .last_evidence_cursor = policy.last_evidence_cursor,
        .turn_start_cursor = policy.turn_start_cursor,
        .last_voiced_cursor = policy.last_voiced_cursor,
        .voiced_frames = policy.voiced_frames,
        .silence_frames = policy.silence_frames,
        .pause_generation = policy.pause_generation,
        .prepare_sample_generation = policy.prepare_sample_generation,
        .commit_sample_generation = policy.commit_sample_generation,
        .forced_sample_generation = policy.forced_sample_generation,
        .last_score = policy.decision.score,
        .adaptive_min = policy.decision.adaptive_min,
        .adaptive_max = policy.decision.adaptive_max,
        .discarded_silence_frames = policy.discarded_silence_frames,
        .reserved = {},
    };
    return 0;
}

int lfm_session_playback_policy_snapshot(
    const LfmSession *session, LfmPlaybackPolicySnapshotV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    if (out->size != sizeof(*out) ||
        out->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    return read_playback_policy_snapshot(session, out);
}

int lfm_session_capture_supervision_snapshot(
    const LfmSession *session, LfmCaptureSupervisionSnapshotV1 *out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    if (out->size != sizeof(*out) ||
        out->abi_version != LFM_RUNTIME_ABI_VERSION) {
        return LFM_STATUS_ABI_MISMATCH;
    }
    return read_capture_supervision_snapshot(session, out);
}

int lfm_session_capture_deadline_advance_manual_test(
    LfmSession *session, uint64_t elapsed_ns) {
    if (!session || !session->capture_supervision.manual ||
        !session->capture_supervision.source) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    return kc_deadline_source_advance_manual_test(
        session->capture_supervision.source, elapsed_ns);
}

int lfm_session_capture_deadline_fire_manual_test(
    LfmSession *session, uint32_t slot) {
    if (!session || !session->capture_supervision.manual ||
        !session->capture_supervision.source ||
        slot >= CAPTURE_DEADLINE_COUNT) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    return kc_deadline_source_fire_manual_test(
        session->capture_supervision.source, slot);
}

int lfm_session_capture_deadline_identity_test(
    const LfmSession *session, uint32_t slot,
    const LfmCaptureDeadlineSlotSnapshotV1 *identity) {
    if (!session || !identity || slot >= CAPTURE_DEADLINE_COUNT ||
        identity->slot != slot) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmCaptureSupervisionSnapshotV1 snapshot = {
        .size = sizeof(LfmCaptureSupervisionSnapshotV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
    };
    const int status = read_capture_supervision_snapshot(
        session, &snapshot);
    if (status != 0) return status;
    const LfmCaptureDeadlineSlotSnapshotV1 &published =
        snapshot.slots[slot];
    return published.slot == identity->slot &&
                   published.arm_generation == identity->arm_generation &&
                   published.scope_generation == identity->scope_generation &&
                   published.epoch == identity->epoch &&
                   published.domain == identity->domain &&
                   published.pause_generation == identity->pause_generation &&
                   ticket_equal(published.child, identity->child) &&
                   ticket_equal(published.parent, identity->parent)
               ? 0
               : LFM_STATUS_STALE;
}

int lfm_session_destroy(LfmSession *session) {
    if (!session) return LFM_STATUS_INVALID_ARGUMENT;
    std::unique_lock<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->state.load(std::memory_order_acquire) != LFM_SESSION_JOINED ||
        pool_live(session->playback) != 0 ||
        session->capture_producers.load(std::memory_order_acquire) != 0 ||
        session->playback_consumers.load(std::memory_order_acquire) != 0 ||
        session->control_handles != 0) {
        return LFM_STATUS_BUSY;
    }
    lifecycle.unlock();
    unregister_session(session->runtime, session);
    delete session->retired_chunk_producer;
    delete session;
    return 0;
}

int lfm_capture_chunk_producer_create(LfmSession *session, uint64_t stream,
                                      uint32_t lane,
                                      LfmCaptureProducer **out) {
    if (!session || !out || stream == 0 || lane >= MAX_KERNEL_LANES) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->state.load(std::memory_order_acquire) !=
            LFM_SESSION_CREATED ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (session->capture_producers.load(std::memory_order_acquire) != 0 ||
        session->retired_chunk_producer != nullptr ||
        !session->capture_arena.samples ||
        session->capture_arena.capacity_frames == 0 ||
        !capture_chunk_empty(session->capture_chunks) ||
        capture_range_live(session->capture_arena) != 0) {
        return LFM_STATUS_BUSY;
    }
    /* Allocation occurs inside CREATED admission. A racing start owns this
     * same mutex, so no physical endpoint can appear after readiness. */
    LfmCaptureProducer *producer =
        new (std::nothrow) LfmCaptureProducer();
    if (!producer) return LFM_STATUS_OUT_OF_MEMORY;
    producer->session = session;
    producer->stream = stream;
    producer->lane = lane;
    producer->sample_rate = session->capture_rate;
    /* Do not pre-mint a turn before the device publishes audio. A typed
     * command may legitimately run first; the first callback must receive a
     * ticket newer than every action already admitted to this runtime. */
    producer->transport_sequence.store(0, std::memory_order_relaxed);
    producer->transport_epoch.store(0, std::memory_order_relaxed);
    session->capture_producers.store(1, std::memory_order_release);
    session->chunk_producer.store(producer, std::memory_order_release);
    *out = producer;
    return 0;
}

void capture_writer_idle(LfmCaptureProducer *producer) {
    /* ACTIVE is a coordinator-visible retirement predicate. Every transition
     * back to IDLE is therefore a successor edge, including cancellation
     * after a concurrent stop/close has already rung its own earlier edge. */
    producer->writer.gate.store(CAPTURE_WRITER_IDLE,
                                std::memory_order_release);
    notify_session(producer->session);
}

int lfm_capture_producer_claim_chunk(LfmCaptureProducer *producer,
                                     uint32_t frames,
                                     uint32_t sample_rate,
                                     uint32_t source_channels, uint32_t flags,
                                     LfmCaptureChunkV1 *out) {
    if (!producer || !producer->session || !out || frames == 0 ||
        source_channels == 0 || flags != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmSession *session = producer->session;
    const uint32_t rate = sample_rate == 0 ? producer->sample_rate : sample_rate;
    if (rate != producer->sample_rate) return LFM_STATUS_INVALID_ARGUMENT;
    if (frames > session->capture_callback_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (producer->closing.load(std::memory_order_acquire) ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (producer->gap_debt_frames.load(std::memory_order_acquire) != 0 ||
        !capture_chunk_has_space(session->capture_chunks)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    uint32_t expected = CAPTURE_WRITER_IDLE;
    if (!producer->writer.gate.compare_exchange_strong(
            expected, CAPTURE_WRITER_ACTIVE, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (producer->closing.load(std::memory_order_acquire) ||
        session->stop.load(std::memory_order_acquire)) {
        capture_writer_idle(producer);
        return LFM_STATUS_CANCELLED;
    }
    const uint64_t start =
        producer->sample_cursor.load(std::memory_order_relaxed);
    if (start > UINT64_MAX - frames) {
        capture_writer_idle(producer);
        request_stop(session, -EOVERFLOW);
        return -EOVERFLOW;
    }
    const uint64_t end = start + frames;
    const uint64_t reclaim =
        session->capture_arena.reclaim_cursor.value.load(
            std::memory_order_acquire);
    if (start < reclaim || end - reclaim >
            session->capture_arena.capacity_frames) {
        capture_writer_idle(producer);
        return LFM_STATUS_WOULD_BLOCK;
    }
    const uint64_t cycle = start / session->capture_arena.capacity_frames;
    const uint64_t identity = lease_id(CAPTURE_IDENTITY_DIRECTION, 0);
    if (cycle == UINT64_MAX || identity == 0) {
        capture_writer_idle(producer);
        request_stop(session, -EOVERFLOW);
        return -EOVERFLOW;
    }
    const uint64_t epoch = session->epoch.load(std::memory_order_acquire);
    const LfmTicketIdV1 transport = current_capture_ticket(producer, epoch);
    producer->writer.pending = {
        .size = sizeof(LfmCaptureChunkV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .stream = producer->stream,
        .lane = producer->lane,
        .flags = flags,
        .chunk_sequence = producer->chunk_sequence,
        .first_sample_cursor = start,
        .stream_epoch = epoch,
        .turn_ticket = transport,
        .lease_id = identity,
        .buffer_generation = cycle + 1,
        .offset_frames = static_cast<uint32_t>(
            start % session->capture_arena.capacity_frames),
        .frames = frames,
        .channels = source_channels,
        .sample_rate = rate,
        .reserved = {},
    };
    *out = producer->writer.pending;
    return 0;
}

int lfm_capture_producer_resolve_chunk(LfmCaptureProducer *producer,
                                       const LfmCaptureChunkV1 *chunk,
                                       LfmMutableF32SpanV1 out_spans[2],
                                       uint32_t *out_span_count) {
    if (!producer || !producer->session || !out_spans ||
        !out_span_count || !valid_chunk(chunk) ||
        (chunk->flags & (LFM_CAPTURE_CHUNK_GAP |
                         LFM_CAPTURE_CHUNK_XRUN)) != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (producer->writer.gate.load(std::memory_order_acquire) !=
            CAPTURE_WRITER_ACTIVE ||
        !chunk_equal(producer->writer.pending, *chunk)) {
        return LFM_STATUS_STALE;
    }
    return capture_arena_mutable_spans(
        producer->session->capture_arena, chunk->first_sample_cursor,
        chunk->frames, out_spans, out_span_count);
}

int lfm_capture_producer_commit_chunk(LfmCaptureProducer *producer,
                                      const LfmCaptureChunkV1 *chunk) {
    if (!producer || !producer->session || !valid_chunk(chunk)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmSession *session = producer->session;
    if (producer->writer.gate.load(std::memory_order_acquire) !=
            CAPTURE_WRITER_ACTIVE ||
        !chunk_equal(producer->writer.pending, *chunk)) {
        return LFM_STATUS_STALE;
    }
    if (!enter_publication(session)) {
        producer->writer.pending = {};
        capture_writer_idle(producer);
        return LFM_STATUS_CANCELLED;
    }
    if (!capture_chunk_has_space(session->capture_chunks)) {
        producer->writer.pending = {};
        capture_writer_idle(producer);
        leave_publication(session);
        return LFM_STATUS_INTERNAL;
    }

    producer->chunk_sequence++;
    const uint64_t end = chunk->first_sample_cursor + chunk->frames;
    producer->sample_cursor.store(end, std::memory_order_release);
    const LfmCaptureChunkV1 published = producer->writer.pending;
    producer->writer.pending = {};
    if (!capture_chunk_push(&session->capture_chunks, published)) {
        std::abort();
    }
    capture_writer_idle(producer);
    leave_publication(session);
    return 0;
}

int lfm_capture_producer_write_interleaved(
    LfmCaptureProducer *producer, const void *samples, size_t sample_count,
    uint32_t channels, uint32_t sample_rate, uint32_t format, uint32_t flags,
    LfmCaptureWriteV1 *out) {
    if (!out) return LFM_STATUS_INVALID_ARGUMENT;
    capture_write_result(out, 0, 0, 0, LFM_STATUS_INVALID_ARGUMENT);
    if (!producer || !producer->session || channels == 0 ||
        (sample_count != 0 && !samples) || flags != 0 ||
        (format != LFM_CAPTURE_INPUT_F32 &&
         format != LFM_CAPTURE_INPUT_I16 &&
         format != LFM_CAPTURE_INPUT_U16)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (sample_count == 0) {
        capture_write_result(out, 0, 0, 0, 0);
        return 0;
    }

    const size_t whole_frames = sample_count / channels;
    const size_t remainder = sample_count % channels;
    const size_t rounded_frames = whole_frames + (remainder != 0 ? 1 : 0);
    if (rounded_frames > std::numeric_limits<uint32_t>::max()) {
        out->dropped_frames = UINT32_MAX;
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    const uint32_t frames = static_cast<uint32_t>(rounded_frames);
    if (producer->gap_debt_frames.load(std::memory_order_acquire) != 0) {
        return capture_write_drop(
            producer, frames, channels,
            remainder == 0 ? LFM_STATUS_WOULD_BLOCK
                           : LFM_STATUS_INVALID_ARGUMENT,
            out);
    }
    if (remainder != 0) {
        return capture_write_drop(producer, frames, channels,
                                  LFM_STATUS_INVALID_ARGUMENT, out);
    }

    LfmCaptureChunkV1 chunk{};
    int status = lfm_capture_producer_claim_chunk(
        producer, frames, sample_rate, channels, flags, &chunk);
    if (status != 0) {
        return capture_write_drop(producer, frames, channels, status, out);
    }

    LfmMutableF32SpanV1 spans[2]{};
    uint32_t span_count = 0;
    status = lfm_capture_producer_resolve_chunk(
        producer, &chunk, spans, &span_count);
    const size_t capacity = span_count == 0
        ? 0
        : spans[0].count + (span_count == 2 ? spans[1].count : 0);
    if (status == 0 &&
        (span_count == 0 || span_count > 2 || capacity != whole_frames)) {
        status = LFM_STATUS_INVALID_ARGUMENT;
    }
    if (status == 0 && format == LFM_CAPTURE_INPUT_F32) {
        status = lfm_capture_downmix_f32(
            static_cast<const float *>(samples), spans[0].data,
            spans[0].count, channels, spans[0].count);
        if (status == 0 && span_count == 2) {
            status = lfm_capture_downmix_f32(
                static_cast<const float *>(samples) + spans[0].count * channels,
                spans[1].data, spans[1].count, channels, spans[1].count);
        }
    }
    if (status == 0 && format == LFM_CAPTURE_INPUT_I16) {
        status = lfm_capture_downmix_i16(
            static_cast<const int16_t *>(samples), spans[0].data,
            spans[0].count, channels, spans[0].count);
        if (status == 0 && span_count == 2) {
            status = lfm_capture_downmix_i16(
                static_cast<const int16_t *>(samples) + spans[0].count * channels,
                spans[1].data, spans[1].count, channels, spans[1].count);
        }
    }
    if (status == 0 && format == LFM_CAPTURE_INPUT_U16) {
        status = lfm_capture_downmix_u16(
            static_cast<const uint16_t *>(samples), spans[0].data,
            spans[0].count, channels, spans[0].count);
        if (status == 0 && span_count == 2) {
            status = lfm_capture_downmix_u16(
                static_cast<const uint16_t *>(samples) + spans[0].count * channels,
                spans[1].data, spans[1].count, channels, spans[1].count);
        }
    }
    if (status != 0) {
        (void)lfm_capture_producer_abort_chunk(producer, &chunk);
        return capture_write_drop(producer, frames, channels, status, out);
    }

    status = lfm_capture_producer_commit_chunk(producer, &chunk);
    if (status != 0) {
        return capture_write_drop(producer, frames, channels, status, out);
    }
    capture_write_result(out, frames, 0, 0, 0);
    return 0;
}

int lfm_capture_producer_abort_chunk(LfmCaptureProducer *producer,
                                     const LfmCaptureChunkV1 *chunk) {
    if (!producer || !producer->session || !valid_chunk(chunk)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (producer->writer.gate.load(std::memory_order_acquire) !=
            CAPTURE_WRITER_ACTIVE ||
        !chunk_equal(producer->writer.pending, *chunk)) {
        return LFM_STATUS_STALE;
    }
    producer->writer.pending = {};
    capture_writer_idle(producer);
    return 0;
}

int lfm_capture_producer_publish_gap(LfmCaptureProducer *producer,
                                     uint32_t dropped_frames,
                                     uint32_t source_channels, uint32_t flags,
                                     LfmCaptureChunkV1 *out) {
    if (!producer || !producer->session || !out || dropped_frames == 0 ||
        source_channels == 0 ||
        (flags & LFM_CAPTURE_CHUNK_GAP) == 0 ||
        (flags & ~(LFM_CAPTURE_CHUNK_GAP | LFM_CAPTURE_CHUNK_XRUN |
                   LFM_CAPTURE_CHUNK_MUTED)) != 0 ||
        (flags & (LFM_CAPTURE_CHUNK_XRUN | LFM_CAPTURE_CHUNK_MUTED)) == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmSession *session = producer->session;
    if (producer->closing.load(std::memory_order_acquire) ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    uint32_t total_frames = 0;
    uint32_t debt_channels = 0;
    uint32_t debt_flags = 0;
    const int debt = add_gap_debt(
        producer, dropped_frames, source_channels, flags, &total_frames,
        &debt_channels, &debt_flags);
    if (debt != 0) return debt;
    if (!capture_chunk_has_space(session->capture_chunks)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    uint32_t expected = CAPTURE_WRITER_IDLE;
    if (!producer->writer.gate.compare_exchange_strong(
            expected, CAPTURE_WRITER_ACTIVE, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return LFM_STATUS_WOULD_BLOCK;
    }
    if (!enter_publication(session)) {
        capture_writer_idle(producer);
        return LFM_STATUS_CANCELLED;
    }
    const uint64_t start =
        producer->sample_cursor.load(std::memory_order_relaxed);
    if (start > UINT64_MAX - total_frames) {
        capture_writer_idle(producer);
        leave_publication(session);
        request_stop(session, -EOVERFLOW);
        return -EOVERFLOW;
    }
    const uint64_t cycle = start / session->capture_arena.capacity_frames;
    const uint64_t identity = lease_id(CAPTURE_IDENTITY_DIRECTION, 0);
    if (cycle == UINT64_MAX || identity == 0) {
        capture_writer_idle(producer);
        leave_publication(session);
        request_stop(session, -EOVERFLOW);
        return -EOVERFLOW;
    }
    const uint64_t epoch = session->epoch.load(std::memory_order_acquire);
    const LfmTicketIdV1 transport = current_capture_ticket(producer, epoch);
    const LfmCaptureChunkV1 gap = {
        .size = sizeof(LfmCaptureChunkV1),
        .abi_version = LFM_RUNTIME_ABI_VERSION,
        .stream = producer->stream,
        .lane = producer->lane,
        .flags = debt_flags,
        .chunk_sequence = producer->chunk_sequence,
        .first_sample_cursor = start,
        .stream_epoch = epoch,
        .turn_ticket = transport,
        .lease_id = identity,
        .buffer_generation = cycle + 1,
        .offset_frames = static_cast<uint32_t>(
            start % session->capture_arena.capacity_frames),
        .frames = total_frames,
        .channels = debt_channels,
        .sample_rate = producer->sample_rate,
        .reserved = {},
    };
    producer->chunk_sequence++;
    producer->sample_cursor.store(start + total_frames,
                                  std::memory_order_release);
    if (!capture_chunk_push(&session->capture_chunks, gap)) std::abort();
    producer->gap_debt_frames.store(0, std::memory_order_release);
    producer->gap_debt_channels.store(0, std::memory_order_relaxed);
    producer->gap_debt_flags.store(0, std::memory_order_relaxed);
    /* A gap terminates the logical capture turn even though the session epoch
     * remains live. The record above carries the old correlation identity;
     * the first callback after the discontinuity must belong to a new turn. */
    (void)rotate_capture_ticket(producer, epoch);
    capture_writer_idle(producer);
    *out = gap;
    leave_publication(session);
    return 0;
}

int lfm_capture_producer_destroy(LfmCaptureProducer *producer) {
    if (!producer || !producer->session) return LFM_STATUS_INVALID_ARGUMENT;
    LfmSession *session = producer->session;
    {
        std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
        if (session->capture_producers.load(std::memory_order_acquire) != 1 ||
            session->retired_chunk_producer != nullptr ||
            session->chunk_producer.load(std::memory_order_acquire) !=
                producer) {
            return LFM_STATUS_BUSY;
        }
        bool expected = false;
        if (!producer->closing.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return LFM_STATUS_BUSY;
        }
        session->capture_producers.store(0, std::memory_order_release);
        session->retired_chunk_producer = producer;
    }
    if (!session->stop.load(std::memory_order_acquire)) {
        session->capture_supervision.device_loss_pending.store(
            true, std::memory_order_release);
    }
    /* The hardware endpoint is now disconnected, so Rust may drop its opaque
     * handle. Native session ownership remains until the coordinator drains
     * every publication admitted before `closing`, retires the endpoint, and
     * session destruction finally releases this object. */
    notify_session(session);
    return 0;
}

int lfm_playback_consumer_create(LfmSession *session,
                                 LfmPlaybackConsumer **out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->state.load(std::memory_order_acquire) !=
            LFM_SESSION_CREATED ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    /* PlaybackPool::head is a single-consumer cursor. Make that ownership a
     * lifecycle invariant instead of trusting callers not to clone the
     * hardware endpoint. */
    if (session->playback_consumers.load(std::memory_order_acquire) != 0) {
        return LFM_STATUS_BUSY;
    }
    LfmPlaybackConsumer *consumer =
        new (std::nothrow) LfmPlaybackConsumer();
    if (!consumer) return LFM_STATUS_OUT_OF_MEMORY;
    consumer->session = session;
    session->playback_consumers.store(1, std::memory_order_release);
    *out = consumer;
    return 0;
}

int lfm_session_publish_playback_f32_test(LfmSession *session,
                                          const float *samples,
                                          uint32_t frames,
                                          LfmPcmLeaseV1 *out) {
    if (!session || !samples || !out || !session->dock_only || frames == 0 ||
        frames > session->playback_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPcmLeaseV1 lease{};
    int status = playback_reserve(session, frames, session->playback_rate,
                                  &lease);
    if (status != 0) return status;
    float *destination = nullptr;
    size_t capacity = 0;
    status = playback_resolve_mut(session, &lease, &destination, &capacity);
    if (status != 0 || !destination || capacity < frames) {
        (void)playback_release(session, &lease);
        return status == 0 ? LFM_STATUS_INTERNAL : status;
    }
    std::memcpy(destination, samples, static_cast<size_t>(frames) *
                                      sizeof(float));
    lease.ticket = next_ticket(session, LFM_TICKET_TURN);
    status = playback_publish(session, &lease);
    if (status != 0) {
        (void)playback_release(session, &lease);
        return status;
    }
    *out = lease;
    return 0;
}

int lfm_internal_session_release_unpublished_playback_for_test(
    LfmSession *session, uint32_t frames) {
    if (!session || !session->dock_only || frames == 0 ||
        frames > session->playback_frames) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    LfmPcmLeaseV1 lease{};
    const int status = playback_reserve(session, frames,
                                        session->playback_rate, &lease);
    if (status != 0) return status;
    return playback_release(session, &lease);
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
        const int release = playback_release(consumer->session, &lease);
        if (release != 0) return release;
        return LFM_STATUS_STALE;
    }
    consumer->lease = lease;
    consumer->lineage = lease;
    consumer->active = true;
    return 0;
}

int lfm_playback_consumer_render_f32(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, float *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out) {
    return render_playback_evidence(
        consumer, lease, source_offset_frames, destination, frames, channels,
        destination_capacity, fanout_f32_erased, out);
}

int lfm_playback_consumer_render_i16(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, int16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out) {
    return render_playback_evidence(
        consumer, lease, source_offset_frames, destination, frames, channels,
        destination_capacity, fanout_i16_erased, out);
}

int lfm_playback_consumer_render_u16(
    LfmPlaybackConsumer *consumer, const LfmPcmLeaseV1 *lease,
    uint32_t source_offset_frames, uint16_t *destination, uint32_t frames,
    uint32_t channels, size_t destination_capacity,
    LfmPlaybackRenderV1 *out) {
    return render_playback_evidence(
        consumer, lease, source_offset_frames, destination, frames, channels,
        destination_capacity, fanout_u16_erased, out);
}

int lfm_playback_consumer_observe(LfmPlaybackConsumer *consumer,
                                  const LfmPcmLeaseV1 *lease,
                                  uint32_t source_offset_frames,
                                  uint32_t frames, uint32_t flags,
                                  LfmPlaybackRenderV1 *out) {
    if ((flags & LFM_PLAYBACK_EVIDENCE_RENDERED) != 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    return publish_playback_evidence(consumer, lease, source_offset_frames,
                                     frames, flags, out);
}

int lfm_playback_consumer_release(LfmPlaybackConsumer *consumer,
                                  const LfmPcmLeaseV1 *lease) {
    if (!consumer_matches(consumer, lease)) return LFM_STATUS_STALE;
    const int status = playback_release(consumer->session, lease);
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
        if (session->playback_consumers.load(std::memory_order_acquire) == 0) {
            std::abort();
        }
        session->playback_consumers.store(0, std::memory_order_release);
        /* A live playback endpoint is a lifetime lease over the physical
         * sink. Losing it is terminal: allowing the native route to keep
         * producing would only fill the fixed pool and leave the session
         * dormant with no callback capable of draining it. Administrative
         * teardown sets stop while holding this same lifecycle lock first, so
         * it remains a clean close rather than forging a device-loss fault. */
        if (!session->stop.load(std::memory_order_acquire)) {
            request_stop(session, LFM_STATUS_HOST_SINK);
        } else {
            notify_session(session);
        }
        consumer->session = nullptr;
    }
    delete consumer;
    return 0;
}

int lfm_session_control_create(LfmSession *session,
                               LfmSessionControl **out) {
    if (!session || !out) return LFM_STATUS_INVALID_ARGUMENT;
    *out = nullptr;
    std::lock_guard<std::mutex> lifecycle(session->lifecycle_mutex);
    if (session->state.load(std::memory_order_acquire) !=
            LFM_SESSION_CREATED ||
        session->stop.load(std::memory_order_acquire)) {
        return LFM_STATUS_CANCELLED;
    }
    if (session->control_handles == UINT32_MAX) {
        return LFM_STATUS_OUT_OF_MEMORY;
    }
    LfmSessionControl *control = new (std::nothrow) LfmSessionControl();
    if (!control) return LFM_STATUS_OUT_OF_MEMORY;
    control->session = session;
    session->control_handles++;
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

} /* extern "C" */

namespace {

int playback_reserve(LfmSession *session, uint32_t frames,
                     uint32_t sample_rate, LfmPcmLeaseV1 *out) {
    if (!out) return LFM_STATUS_INVALID_ARGUMENT;
    PlaybackPool *pool = nullptr;
    uint32_t rate = 0;
    size_t samples = 0;
    const int prepared = prepare_reservation(session, frames, sample_rate,
                                             &pool, &rate,
                                             &samples);
    if (prepared != 0) return prepared;
    const uint32_t start =
        pool->cursor.value.fetch_add(1, std::memory_order_relaxed) % pool->capacity;
    for (uint32_t offset = 0; offset < pool->capacity; ++offset) {
        const uint32_t index = (start + offset) % pool->capacity;
        const int status = reserve_slot_at(session, pool, frames, rate,
                                           samples, index, out);
        if (status == 0 || status != LFM_STATUS_WOULD_BLOCK) return status;
    }
    return LFM_STATUS_WOULD_BLOCK;
}

int playback_resolve_mut(LfmSession *session, const LfmPcmLeaseV1 *lease,
                         float **out_samples,
                         size_t *out_sample_capacity) {
    if (!session || !lease || !out_samples || !out_sample_capacity) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t index = 0;
    if (!decode_playback_lease_id(lease->lease_id, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PlaybackPool *pool = &session->playback;
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return rc;
    if (slot->state.load(std::memory_order_acquire) != SLOT_RESERVED) {
        return LFM_STATUS_STALE;
    }
    *out_samples = slot->samples + lease->offset_bytes / sizeof(float);
    *out_sample_capacity = lease->length_bytes / sizeof(float);
    return 0;
}

int playback_resolve(const LfmSession *session,
                     const LfmPcmLeaseV1 *lease,
                     const float **out_samples,
                     size_t *out_sample_count) {
    if (!session || !lease || !out_samples || !out_sample_count) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    uint32_t index = 0;
    if (!decode_playback_lease_id(lease->lease_id, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    if (lease->stream_epoch != session->epoch.load(std::memory_order_acquire)) {
        return LFM_STATUS_STALE;
    }
    PlaybackPool *pool = const_cast<PlaybackPool *>(&session->playback);
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

int playback_publish(LfmSession *session, const LfmPcmLeaseV1 *lease) {
    if (!session || !lease) return LFM_STATUS_INVALID_ARGUMENT;
    if (!enter_publication(session)) return LFM_STATUS_CANCELLED;
    const auto finish = [session](int status) {
        leave_publication(session);
        return status;
    };
    uint32_t index = 0;
    if (!decode_playback_lease_id(lease->lease_id, &index)) {
        return finish(LFM_STATUS_INVALID_ARGUMENT);
    }
    PlaybackPool *pool = &session->playback;
    PcmSlot *slot = nullptr;
    int rc = pool_slot(pool, lease, &slot, nullptr);
    if (rc != 0) return finish(rc);
    if (session->stop.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_CANCELLED);
    }
    if (lease->stream_epoch != session->epoch.load(std::memory_order_acquire)) {
        return finish(LFM_STATUS_STALE);
    }
    uint32_t expected = SLOT_RESERVED;
    if (!slot->state.compare_exchange_strong(expected, SLOT_PUBLISHED,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return finish(LFM_STATUS_STALE);
    }
    slot->ticket = lease->ticket;
    pool_push(pool, *lease);
    session->playback_published.fetch_add(1, std::memory_order_relaxed);
    return finish(0);
}

int playback_release(LfmSession *session, const LfmPcmLeaseV1 *lease) {
    if (!session || !lease) return LFM_STATUS_INVALID_ARGUMENT;
    uint32_t index = 0;
    if (!decode_playback_lease_id(lease->lease_id, &index)) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    PlaybackPool *pool = &session->playback;
    const uint32_t allowed = (UINT32_C(1) << SLOT_RESERVED) |
                             (UINT32_C(1) << SLOT_CONSUMING);
    const int status = release_slot(pool, lease,
                                    &session->playback_consumed, allowed);
    /* Playback retirement is the sole successor edge after the coordinator
     * observed a live device lease and became dormant. */
    if (status == 0) notify_session(session);
    return status;
}

} // namespace
