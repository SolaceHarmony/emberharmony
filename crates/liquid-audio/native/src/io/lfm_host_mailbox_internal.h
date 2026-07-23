#ifndef LFM_HOST_MAILBOX_INTERNAL_H
#define LFM_HOST_MAILBOX_INTERNAL_H

#include "lfm_host_mailbox.h"

#include <cstddef>
#include <cstdint>

namespace lfm::host {

constexpr size_t kCacheLine = 128;
constexpr uint32_t kMailboxInvalid = 0;
constexpr uint32_t kMailboxInitializing = 1;
constexpr uint32_t kMailboxReady = 2;
constexpr uint32_t kMailboxStopping = 3;
constexpr uint32_t kMailboxDead = 4;
constexpr uint32_t kClientFree = 0;
constexpr uint32_t kClientClaiming = 1;
constexpr uint32_t kClientRegistering = 2;
constexpr uint32_t kClientActive = 3;
constexpr uint32_t kClientRetiring = 4;
constexpr uint32_t kClientDead = 5;
constexpr uint32_t kClientActivating = 6;
constexpr uint32_t kMailboxEvicted = 1u << 0;
constexpr uint8_t kMailboxMagic[8] = {'L', 'F', 'M', 'H', 'O', 'S', 'T', 0};

struct alignas(kCacheLine) Cursor {
    uint64_t value{0};
    uint64_t generation{0};
    uint8_t padding[112]{};
};
static_assert(sizeof(Cursor) == kCacheLine);

struct alignas(kCacheLine) RequestCell {
    uint64_t sequence{0};
    HostRequest record{};
};
static_assert(sizeof(RequestCell) == kCacheLine);

struct alignas(kCacheLine) CompletionCell {
    uint64_t sequence{0};
    HostCompletion record{};
};
static_assert(sizeof(CompletionCell) == kCacheLine);

struct alignas(kCacheLine) ClientControl {
    uint32_t state{0};
    uint32_t flags{0};
    uint64_t client_generation{0};
    uint64_t host_generation{0};
    uint64_t client_pid{0};
    uint64_t client_start_time{0};
    uint64_t client_uid{0};
    uint64_t active_lease_generation{0};
    uint32_t lease_count{0};
    uint32_t registered{0};
    uint64_t padding[8]{};
};
static_assert(sizeof(ClientControl) == kCacheLine);

struct alignas(kCacheLine) MailboxHeader {
    uint8_t magic[8]{};
    uint32_t state{0};
    uint32_t client_capacity{0};
    uint32_t ring_capacity{0};
    uint32_t flags{0};
    uint64_t host_generation{0};
    uint64_t host_pid{0};
    uint64_t host_start_time{0};
    uint64_t host_uid{0};
    uint64_t segment_generation{0};
    CheckpointIdentity checkpoint_identity{};
    CheckpointIdentity content_digest{};
    uint64_t active_clients{0};
    uint64_t active_leases{0};
    uint64_t client_events{0};
    uint64_t padding[13]{};
};
static_assert(sizeof(MailboxHeader) == 256);

constexpr size_t kControlOffset = 0;
constexpr size_t kRequestHeadOffset = kControlOffset + sizeof(ClientControl);
constexpr size_t kRequestTailOffset = kRequestHeadOffset + sizeof(Cursor);
constexpr size_t kCompletionHeadOffset = kRequestTailOffset + sizeof(Cursor);
constexpr size_t kCompletionTailOffset = kCompletionHeadOffset + sizeof(Cursor);
constexpr size_t kRequestsOffset = kCompletionTailOffset + sizeof(Cursor);
constexpr size_t kCompletionsOffset =
    kRequestsOffset + kRingCapacity * sizeof(RequestCell);
constexpr size_t kClientStride =
    kCompletionsOffset + kRingCapacity * sizeof(CompletionCell);
constexpr size_t kClientsOffset = sizeof(MailboxHeader);
constexpr size_t kMailboxBytes =
    kClientsOffset + kClientCapacity * kClientStride;
static_assert(kClientStride % kCacheLine == 0);
static_assert(kMailboxBytes % kCacheLine == 0);

/* A client slot is a non-owning view over one stride of the shared mapping.
 * Numerical storage never enters this region; records carry identity and
 * lease metadata only. */
struct ClientSlotView {
    std::byte *base{nullptr};

    ClientControl &control() const {
        return *reinterpret_cast<ClientControl *>(base + kControlOffset);
    }
    Cursor &request_head() const {
        return *reinterpret_cast<Cursor *>(base + kRequestHeadOffset);
    }
    Cursor &request_tail() const {
        return *reinterpret_cast<Cursor *>(base + kRequestTailOffset);
    }
    Cursor &completion_head() const {
        return *reinterpret_cast<Cursor *>(base + kCompletionHeadOffset);
    }
    Cursor &completion_tail() const {
        return *reinterpret_cast<Cursor *>(base + kCompletionTailOffset);
    }
    RequestCell &request(uint64_t sequence) const {
        const size_t index = static_cast<size_t>(sequence % kRingCapacity);
        return *reinterpret_cast<RequestCell *>(
            base + kRequestsOffset + index * sizeof(RequestCell));
    }
    CompletionCell &completion(uint64_t sequence) const {
        const size_t index = static_cast<size_t>(sequence % kRingCapacity);
        return *reinterpret_cast<CompletionCell *>(
            base + kCompletionsOffset + index * sizeof(CompletionCell));
    }
};

/* Root view of the mapped buffer. sizeof(Mailbox) is only the header;
 * kMailboxBytes is the validated mapping extent. */
struct alignas(kCacheLine) Mailbox {
    MailboxHeader header;

    ClientSlotView client(uint32_t index) {
        return {
            .base = reinterpret_cast<std::byte *>(this) + kClientsOffset +
                    static_cast<size_t>(index) * kClientStride,
        };
    }
};
static_assert(sizeof(Mailbox) == sizeof(MailboxHeader));

template <typename T>
T load(const T *value, int order = __ATOMIC_ACQUIRE) {
    return __atomic_load_n(value, order);
}

template <typename T>
void store(T *target, T value, int order = __ATOMIC_RELEASE) {
    __atomic_store_n(target, value, order);
}

template <typename T>
T fetch_add(T *target, T value, int order = __ATOMIC_ACQ_REL) {
    return __atomic_fetch_add(target, value, order);
}

template <typename T>
bool compare_exchange(T *target, T *expected, T desired) {
    return __atomic_compare_exchange_n(target, expected, desired, false,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

inline bool ticket_equal(const kc_ticket_id &left,
                         const kc_ticket_id &right) {
    return left.runtime_epoch == right.runtime_epoch &&
           left.sequence == right.sequence &&
           left.generation == right.generation && left.kind == right.kind;
}

inline bool ticket_valid(const kc_ticket_id &ticket) {
    return ticket.runtime_epoch != 0 && ticket.sequence != 0 &&
           ticket.generation != 0 && ticket.kind != 0;
}

inline bool identity_equal(const CheckpointIdentity &left,
                           const CheckpointIdentity &right) {
    return left.word0 == right.word0 && left.word1 == right.word1 &&
           left.word2 == right.word2 && left.word3 == right.word3;
}

inline CheckpointIdentity identity_from_bytes(const uint8_t *bytes) {
    CheckpointIdentity identity{};
    __builtin_memcpy(&identity, bytes, sizeof(identity));
    return identity;
}

inline const uint8_t *identity_bytes(const CheckpointIdentity &identity) {
    return reinterpret_cast<const uint8_t *>(&identity);
}

namespace test {

/* Native hostile gate only. The producer is quiescent when this is called;
 * the record still travels through the real shared completion ring and GCD
 * callback before the exact probe continuation resumes. */
Status inject_stale_completion(Client *client,
                               koro_cont_t *continuation);

} // namespace test

} // namespace lfm::host

#endif
