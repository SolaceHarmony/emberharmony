#include "lfm_host_mailbox.h"

#include "lfm_host_mailbox_internal.h"
#include "lfm_safetensors.h"

#include "kc_runtime.h"
#include "kcoro_stackless.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <string>

#ifdef __APPLE__
#include <bsm/libbsm.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <servers/bootstrap.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/proc.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
extern char **environ;
#endif
namespace fs = std::filesystem;
namespace lfm::host {

namespace {

void set_error(char *error, size_t length, const std::string &message) {
    if (!error || length == 0) return;
    const size_t bytes = std::min(length - 1, message.size());
    std::memcpy(error, message.data(), bytes);
    error[bytes] = '\0';
}

uint64_t generation() {
    uint64_t value = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#ifdef __APPLE__
    value ^= static_cast<uint64_t>(getpid()) << 19;
#endif
    return value == 0 ? 1 : value;
}

std::string mailbox_name(const char *service) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *cursor =
             reinterpret_cast<const unsigned char *>(service);
         cursor && *cursor; ++cursor) {
        hash ^= *cursor;
        hash *= UINT64_C(1099511628211);
    }
    char name[40]{};
    std::snprintf(name, sizeof(name), "/lfm-host-%016llx",
                  static_cast<unsigned long long>(hash));
    return name;
}

#ifdef __APPLE__
size_t mailbox_mapping_bytes() {
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return kMailboxBytes;
    const size_t width = static_cast<size_t>(page);
    return ((kMailboxBytes + width - 1) / width) * width;
}
#endif

int open_image(const fs::path &root, LfmWeightImage **out,
               char *error, size_t error_length) {
    const fs::path detokenizer = root / "audio_detokenizer";
    if (fs::is_regular_file(detokenizer / "model.safetensors")) {
        const std::string main = root.string();
        const std::string audio = detokenizer.string();
        return lfm_weights_open_bundle(main.c_str(), audio.c_str(), out,
                                       error, error_length);
    }
    const std::string path = root.string();
    return lfm_weights_open(path.c_str(), out, error, error_length);
}

bool request_ready(ClientSlotView slot) {
    const uint64_t head = load(&slot.request_head().value, __ATOMIC_RELAXED);
    const RequestCell &cell = slot.request(head);
    return load(&cell.sequence) == head + 1;
}

bool request_room(ClientSlotView slot) {
    const uint64_t tail = load(&slot.request_tail().value, __ATOMIC_RELAXED);
    const RequestCell &cell = slot.request(tail);
    return load(&cell.sequence) == tail;
}

/* These are fixed pointer-free control records. The copied bytes are the
 * correlation envelope itself; model weights, PCM, activations, and numerical
 * scratch remain in separately owned native buffers and are reached only by
 * validated views. */
bool request_push(ClientSlotView slot, const HostRequest &record) {
    const uint64_t tail = load(&slot.request_tail().value, __ATOMIC_RELAXED);
    RequestCell &cell = slot.request(tail);
    if (load(&cell.sequence) != tail) return false;
    std::memcpy(&cell.record, &record, sizeof(record));
    store(&cell.sequence, tail + 1);
    store(&slot.request_tail().value, tail + 1);
    return true;
}

bool request_pop(ClientSlotView slot, HostRequest *record) {
    const uint64_t head = load(&slot.request_head().value, __ATOMIC_RELAXED);
    RequestCell &cell = slot.request(head);
    if (load(&cell.sequence) != head + 1) return false;
    std::memcpy(record, &cell.record, sizeof(*record));
    store(&cell.sequence, head + kRingCapacity);
    store(&slot.request_head().value, head + 1);
    fetch_add(&slot.request_head().generation, UINT64_C(1));
    return true;
}

bool completion_push(ClientSlotView slot,
                     const HostCompletion &record) {
    const uint64_t tail = load(&slot.completion_tail().value,
                               __ATOMIC_RELAXED);
    CompletionCell &cell = slot.completion(tail);
    if (load(&cell.sequence) != tail) return false;
    std::memcpy(&cell.record, &record, sizeof(record));
    store(&cell.sequence, tail + 1);
    store(&slot.completion_tail().value, tail + 1);
    return true;
}

bool completion_pop(ClientSlotView slot, HostCompletion *record) {
    const uint64_t head = load(&slot.completion_head().value,
                               __ATOMIC_RELAXED);
    CompletionCell &cell = slot.completion(head);
    if (load(&cell.sequence) != head + 1) return false;
    std::memcpy(record, &cell.record, sizeof(*record));
    store(&cell.sequence, head + kRingCapacity);
    store(&slot.completion_head().value, head + 1);
    fetch_add(&slot.completion_head().generation, UINT64_C(1));
    return true;
}

void initialize_slot(ClientSlotView slot, uint64_t prior_generation = 0) {
    std::memset(slot.base, 0, kClientStride);
    slot.control().client_generation = prior_generation;
    for (uint64_t index = 0; index < kRingCapacity;
         ++index) {
        slot.request(index).sequence = index;
        slot.completion(index).sequence = index;
    }
}

void snapshot(const Mailbox *mailbox, HostSnapshot *out) {
    *out = {
        .state = load(&mailbox->header.state),
        .flags = load(&mailbox->header.flags),
        .host_generation = mailbox->header.host_generation,
        .host_pid = mailbox->header.host_pid,
        .segment_generation = mailbox->header.segment_generation,
        .active_clients = load(&mailbox->header.active_clients),
        .active_leases = load(&mailbox->header.active_leases),
        .client_events = load(&mailbox->header.client_events),
    };
    out->checkpoint_identity = mailbox->header.checkpoint_identity;
    out->content_digest = mailbox->header.content_digest;
}

#ifdef __APPLE__

constexpr mach_msg_id_t kMessageHello = 0x4c460101;
constexpr mach_msg_id_t kMessageRegister = 0x4c460102;
constexpr mach_msg_id_t kMessageHostEdge = 0x4c460103;
constexpr mach_msg_id_t kMessageAck = 0x4c460104;
constexpr mach_msg_id_t kMessageClientEdge = 0x4c460105;
constexpr uint32_t kOutstandingFree = 0;
constexpr uint32_t kOutstandingPending = 1;
constexpr uint32_t kOutstandingReady = 2;
constexpr uint32_t kOutstandingFaulted = 3;
constexpr uint32_t kOutstandingReserved = 4;
constexpr uint32_t kCapacityFree = 0;
constexpr uint32_t kCapacityArmed = 1;
constexpr uint32_t kCapacityInstalling = 2;
constexpr uint32_t kDrainBudget = 64;

struct RegisterMessage {
    mach_msg_header_t header{};
    mach_msg_body_t body{};
    mach_msg_port_descriptor_t wake{};
    uint32_t slot{UINT32_MAX};
    uint32_t flags{0};
    uint64_t host_generation{0};
    uint64_t client_generation{0};
};

struct AckMessage {
    mach_msg_header_t header{};
    mach_msg_body_t body{};
    mach_msg_port_descriptor_t liveness{};
    int32_t status{0};
    uint32_t slot{UINT32_MAX};
    uint64_t host_generation{0};
    uint64_t client_generation{0};
};

struct EdgeMessage {
    mach_msg_header_t header{};
    uint64_t host_generation{0};
    uint64_t client_generation{0};
};

struct alignas(16) ReceiveBuffer {
    uint8_t bytes[512]{};
};

bool process_info(uint64_t pid, kinfo_proc *out) {
    if (!out) return false;
    if (pid == 0 ||
        pid > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    int query[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID,
                    static_cast<int>(pid)};
    size_t bytes = sizeof(*out);
    if (sysctl(query, 4, out, &bytes, nullptr, 0) != 0 || bytes == 0) {
        return false;
    }
    return true;
}

uint64_t process_start_time(uint64_t pid) {
    kinfo_proc info{};
    if (!process_info(pid, &info)) return 0;
    const timeval started = info.kp_proc.p_starttime;
    return static_cast<uint64_t>(started.tv_sec) * UINT64_C(1'000'000) +
           static_cast<uint64_t>(started.tv_usec);
}

bool process_alive(uint64_t pid, uint64_t started) {
    if (pid == 0 || started == 0 ||
        pid > static_cast<uint64_t>(std::numeric_limits<pid_t>::max())) {
        return false;
    }
    kinfo_proc info{};
    if (!process_info(pid, &info) || info.kp_proc.p_stat == SZOMB) {
        return false;
    }
    const timeval current = info.kp_proc.p_starttime;
    const uint64_t current_start =
        static_cast<uint64_t>(current.tv_sec) * UINT64_C(1'000'000) +
        static_cast<uint64_t>(current.tv_usec);
    return current_start == started;
}

int send_message(mach_msg_header_t *message, bool coalesced) {
    const mach_msg_return_t status = mach_msg(
        message, MACH_SEND_MSG | MACH_SEND_TIMEOUT, message->msgh_size, 0,
        MACH_PORT_NULL, 0, MACH_PORT_NULL);
    if (status == MACH_MSG_SUCCESS) return Ok;
    /* A full port already contains a wake edge, so another edge is redundant.
     * Registration and acknowledgement carry unique rights and may not be
     * coalesced. */
    if (status == MACH_SEND_TIMED_OUT) return coalesced ? Ok : Capacity;
    if (status == MACH_SEND_INVALID_DEST) return HostDown;
    return IoError;
}

int send_edge(mach_port_t port, mach_msg_id_t id, uint64_t host_generation,
              uint64_t client_generation) {
    if (port == MACH_PORT_NULL) return HostDown;
    EdgeMessage message{};
    message.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);
    message.header.msgh_size = sizeof(message);
    message.header.msgh_remote_port = port;
    message.header.msgh_id = id;
    message.host_generation = host_generation;
    message.client_generation = client_generation;
    return send_message(&message.header, true);
}

int receive_one(mach_port_t port, ReceiveBuffer *buffer) {
    std::memset(buffer, 0, sizeof(*buffer));
    const mach_msg_option_t options =
        MACH_RCV_MSG | MACH_RCV_TIMEOUT |
        MACH_RCV_TRAILER_TYPE(MACH_MSG_TRAILER_FORMAT_0) |
        MACH_RCV_TRAILER_ELEMENTS(MACH_RCV_TRAILER_AUDIT);
    const mach_msg_return_t status = mach_msg(
        reinterpret_cast<mach_msg_header_t *>(buffer), options, 0,
        sizeof(*buffer), port, 0, MACH_PORT_NULL);
    if (status == MACH_MSG_SUCCESS) return 1;
    if (status == MACH_RCV_TIMED_OUT) return 0;
    return -1;
}

audit_token_t audit_token(const mach_msg_header_t *message) {
    const auto offset = round_msg(message->msgh_size);
    const auto *trailer = reinterpret_cast<const mach_msg_audit_trailer_t *>(
        reinterpret_cast<const uint8_t *>(message) + offset);
    return trailer->msgh_audit;
}

bool message_has_port(const mach_msg_header_t *message) {
    if ((message->msgh_bits & MACH_MSGH_BITS_COMPLEX) == 0 ||
        message->msgh_size != sizeof(RegisterMessage)) {
        return false;
    }
    const auto *registration =
        reinterpret_cast<const RegisterMessage *>(message);
    return registration->body.msgh_descriptor_count == 1 &&
           registration->wake.type == MACH_MSG_PORT_DESCRIPTOR &&
           registration->wake.name != MACH_PORT_NULL;
}

struct ServerClient {
    struct Server *server{nullptr};
    uint32_t slot{0};
    std::atomic<mach_port_t> wake_port{MACH_PORT_NULL};
    std::atomic<uint64_t> generation{0};
    std::atomic<uint32_t> privileged{0};
    std::atomic<dispatch_source_t> process_source{nullptr};
    std::atomic<uint32_t> process_cancelled{1};
    std::atomic<uint32_t> cancel_requested{0};
    bool pending{false};
    HostCompletion pending_completion{};
};

struct Outstanding {
    std::atomic<uint32_t> state{kOutstandingFree};
    kc_ticket_id ticket{};
    HostRequest request{};
    HostCompletion completion{};
    koro_cont_t *continuation{nullptr};
};

struct EdgeContinuation {
    std::atomic<uint32_t> state{kCapacityFree};
    uint64_t generation{0};
    kc_ticket_id identity{};
    koro_cont_t *continuation{nullptr};
};

template <typename T>
T *allocate_records(size_t count) {
    void *storage = ::operator new(sizeof(T) * count, std::nothrow);
    if (!storage) return nullptr;
    auto *records = static_cast<T *>(storage);
    size_t constructed = 0;
    try {
        for (; constructed < count; ++constructed) {
            std::construct_at(records + constructed);
        }
    } catch (...) {
        while (constructed != 0) {
            --constructed;
            std::destroy_at(records + constructed);
        }
        ::operator delete(storage);
        return nullptr;
    }
    return records;
}

template <typename T>
void release_records(T *records, size_t count) {
    if (!records) return;
    while (count != 0) {
        --count;
        std::destroy_at(records + count);
    }
    ::operator delete(records);
}

} // namespace

struct Server {
    std::string checkpoint;
    std::string service;
    std::string name;
    uint32_t flags{0};
    uint64_t privileged_pid{0};
    int readiness_fd{-1};
    int fd{-1};
    Mailbox *mailbox{nullptr};
    LfmWeightImage *image{nullptr};
    LfmWeightLoadStatsV2 weights{};
    kc_runtime_t *runtime{nullptr};
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
    mach_port_t service_port{MACH_PORT_NULL};
    mach_port_t liveness_port{MACH_PORT_NULL};
    dispatch_queue_t queue{nullptr};
    dispatch_source_t receive_source{nullptr};
    dispatch_source_t terminate_source{nullptr};
    dispatch_source_t interrupt_source{nullptr};
    std::atomic<uint32_t> stop_requested{0};
    std::atomic<uint32_t> retired{0};
    uint32_t exit_on_retire{0};
    std::atomic<uint64_t> dead_clients{0};
    ServerClient *clients{nullptr};

    ServerClient &client(uint32_t index) const {
        return clients[index];
    }
};

struct Client {
    std::string service;
    std::string name;
    int fd{-1};
    Mailbox *mailbox{nullptr};
    kc_runtime_t *runtime{nullptr};
    uint32_t flags{0};
    uint32_t slot{UINT32_MAX};
    uint64_t host_generation{0};
    uint64_t client_generation{0};
    mach_port_t service_port{MACH_PORT_NULL};
    mach_port_t wake_port{MACH_PORT_NULL};
    mach_port_t liveness_port{MACH_PORT_NULL};
    dispatch_queue_t queue{nullptr};
    dispatch_source_t receive_source{nullptr};
    dispatch_source_t death_source{nullptr};
    std::atomic<int32_t> ready_status{InProgress};
    std::atomic<uint32_t> hello_complete{0};
    std::atomic<uint32_t> host_dead{0};
    std::atomic<uint32_t> producer{0};
    std::atomic<uint64_t> request_generation{0};
    std::atomic<uint64_t> capacity_generation{0};
    koro_cont_t *readiness{nullptr};
    kc_ticket_id readiness_identity{};
    Outstanding *outstanding{nullptr};
    EdgeContinuation capacity{};
    EdgeContinuation events{};
    EdgeContinuation stale_probe{};
    EdgeContinuation retirement{};
    EdgeContinuation close{};
    std::atomic<uint32_t> close_started{0};
    std::atomic<uint32_t> cancel_pending{0};
    std::atomic<uint32_t> closed{0};
};

namespace {

void destroy_unstarted_server(Server *server) {
    if (!server) return;
    if (server->receive_source) dispatch_release(server->receive_source);
    if (server->queue) dispatch_release(server->queue);
    if (server->continuation) {
        (void)koro_cont_destroy(server->continuation);
    }
    if (server->runtime) (void)kc_runtime_destroy(server->runtime);
    if (server->service_port != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), server->service_port);
        (void)mach_port_mod_refs(mach_task_self(), server->service_port,
                                 MACH_PORT_RIGHT_RECEIVE, -1);
    }
    if (server->liveness_port != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), server->liveness_port);
        (void)mach_port_mod_refs(mach_task_self(), server->liveness_port,
                                 MACH_PORT_RIGHT_RECEIVE, -1);
    }
    if (server->mailbox) {
        store(&server->mailbox->header.state, kMailboxDead);
        (void)munmap(server->mailbox, mailbox_mapping_bytes());
    }
    if (!server->name.empty()) (void)shm_unlink(server->name.c_str());
    if (server->fd >= 0) (void)::close(server->fd);
    if (server->image) lfm_weights_close(server->image);
    release_records(server->clients, kClientCapacity);
    delete server;
}

int map_mailbox_client(Client *client, char *error,
                       size_t error_length) {
    client->name = mailbox_name(client->service.c_str());
    client->fd = shm_open(client->name.c_str(), O_RDWR, 0);
    if (client->fd < 0) {
        set_error(error, error_length,
                  "cannot open host mailbox shared memory: " +
                      std::string(std::strerror(errno)));
        return IoError;
    }
    struct stat info {};
    if (fstat(client->fd, &info) != 0) {
        set_error(error, error_length,
                  "cannot stat host mailbox shared memory: " +
                      std::string(std::strerror(errno)));
        return Rejected;
    }
    const size_t mapping_bytes = mailbox_mapping_bytes();
    if (info.st_uid != geteuid() ||
        info.st_size != static_cast<off_t>(mapping_bytes)) {
        set_error(error, error_length,
                  "host mailbox shared object failed size/uid validation "
                  "(size=" + std::to_string(info.st_size) +
                  " expected=" + std::to_string(mapping_bytes) +
                  " uid=" + std::to_string(info.st_uid) +
                  " expected_uid=" + std::to_string(geteuid()) + ")");
        return Rejected;
    }
    void *memory = mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, client->fd, 0);
    if (memory == MAP_FAILED) {
        client->mailbox = nullptr;
        set_error(error, error_length,
                  "cannot map host mailbox: " +
                      std::string(std::strerror(errno)));
        return IoError;
    }
    client->mailbox = static_cast<Mailbox *>(memory);
    const MailboxHeader &header = client->mailbox->header;
    if (std::memcmp(header.magic, kMailboxMagic, sizeof(kMailboxMagic)) != 0 ||
        header.size != kMailboxBytes ||
        header.layout_version != kLayoutVersion ||
        load(&header.state) != kMailboxReady ||
        header.host_generation != client->host_generation ||
        header.host_uid != static_cast<uint64_t>(geteuid()) ||
        !process_alive(header.host_pid, header.host_start_time)) {
        set_error(error, error_length,
                  "host mailbox readiness identity did not match its Mach acknowledgement");
        return Rejected;
    }
    return Ok;
}

int send_registration(Client *client, mach_msg_id_t id, uint32_t slot,
                      uint64_t host_generation,
                      uint64_t client_generation) {
    RegisterMessage message{};
    message.header.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) |
        MACH_MSGH_BITS_COMPLEX;
    message.header.msgh_size = sizeof(message);
    message.header.msgh_remote_port = client->service_port;
    message.header.msgh_id = id;
    message.body.msgh_descriptor_count = 1;
    message.wake.name = client->wake_port;
    message.wake.disposition = MACH_MSG_TYPE_MAKE_SEND;
    message.wake.type = MACH_MSG_PORT_DESCRIPTOR;
    message.slot = slot;
    message.flags = client->flags;
    message.host_generation = host_generation;
    message.client_generation = client_generation;
    return send_message(&message.header, false);
}

void resume_and_release(koro_cont_t *continuation,
                        const kc_ticket_id &identity) {
    if (!continuation) return;
    (void)koro_cont_resume(continuation, &identity);
    koro_cont_release(continuation);
}

void resume_edge(EdgeContinuation &edge) {
    uint32_t armed = kCapacityArmed;
    if (!edge.state.compare_exchange_strong(
            armed, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    koro_cont_t *continuation = edge.continuation;
    const kc_ticket_id identity = edge.identity;
    edge.continuation = nullptr;
    edge.state.store(kCapacityFree, std::memory_order_release);
    resume_and_release(continuation, identity);
}

bool discard_edge(EdgeContinuation &edge) {
    uint32_t armed = kCapacityArmed;
    if (!edge.state.compare_exchange_strong(
            armed, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return false;
    }
    koro_cont_t *continuation = edge.continuation;
    edge.continuation = nullptr;
    edge.state.store(kCapacityFree, std::memory_order_release);
    if (continuation) koro_cont_release(continuation);
    return true;
}

bool outstanding_room(const Client *client) {
    for (uint32_t index = 0; index < kRingCapacity; ++index) {
        if (client->outstanding[index].state.load(std::memory_order_acquire) ==
            kOutstandingFree) {
            return true;
        }
    }
    return false;
}

bool capacity_ready(const Client *client) {
    if (!client->mailbox || client->slot >= kClientCapacity) return false;
    return request_room(client->mailbox->client(client->slot)) &&
           outstanding_room(client);
}

void publish_capacity(Client *client) {
    client->capacity_generation.fetch_add(1, std::memory_order_acq_rel);
    resume_edge(client->capacity);
}

Status dehydrate_for_capacity(Client *client,
                              koro_cont_t *continuation) {
    uint32_t free = kCapacityFree;
    if (!client->capacity.state.compare_exchange_strong(
            free, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    const uint64_t observed =
        client->capacity_generation.load(std::memory_order_acquire);
    client->capacity.generation = observed;
    client->capacity.identity = koro_cont_identity(continuation);
    client->capacity.continuation = continuation;
    koro_cont_retain(continuation);
    client->capacity.state.store(kCapacityArmed,
                                 std::memory_order_release);
    if (capacity_ready(client) ||
        client->capacity_generation.load(std::memory_order_acquire) !=
            observed ||
        client->host_dead.load(std::memory_order_acquire) != 0) {
        resume_edge(client->capacity);
    }
    return Capacity;
}

void fault_client(Client *client) {
    const bool first = client->host_dead.exchange(
                           1, std::memory_order_acq_rel) == 0;
    if (first) {
        client->ready_status.store(HostDown, std::memory_order_release);
        if (client->readiness) {
            koro_cont_t *continuation = client->readiness;
            client->readiness = nullptr;
            resume_and_release(continuation, client->readiness_identity);
        }
        resume_edge(client->capacity);
        resume_edge(client->events);
        resume_edge(client->stale_probe);
        resume_edge(client->retirement);
    }
    for (uint32_t index = 0; index < kRingCapacity; ++index) {
        Outstanding &entry = client->outstanding[index];
        uint32_t expected = kOutstandingPending;
        if (!entry.state.compare_exchange_strong(
                expected, kOutstandingReserved, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        entry.completion = {
            .size = sizeof(HostCompletion),
            .layout_version = kLayoutVersion,
            .ticket = entry.request.ticket,
            .parent = entry.request.parent,
            .host_generation = entry.request.host_generation,
            .client_generation = entry.request.client_generation,
            .lease_generation = entry.request.lease_generation,
            .operation = entry.request.operation,
            .status = HostDown,
        };
        entry.completion.checkpoint_identity =
            entry.request.checkpoint_identity;
        koro_cont_t *continuation = entry.continuation;
        entry.continuation = nullptr;
        entry.state.store(kOutstandingFaulted, std::memory_order_release);
        resume_and_release(continuation, entry.ticket);
    }
}

void client_death_handler(void *context) {
    fault_client(static_cast<Client *>(context));
}

void client_drain(Client *client) {
    if (!client->mailbox || client->slot >= kClientCapacity) {
        return;
    }
    ClientSlotView slot = client->mailbox->client(client->slot);
    HostCompletion completion{};
    while (completion_pop(slot, &completion)) {
        const bool identity_valid =
            completion.size == sizeof(completion) &&
            completion.layout_version == kLayoutVersion &&
            completion.host_generation == client->host_generation &&
            completion.client_generation == client->client_generation &&
            identity_equal(completion.checkpoint_identity,
                           client->mailbox->header.checkpoint_identity);
        if (identity_valid) {
            for (uint32_t index = 0; index < kRingCapacity; ++index) {
                Outstanding &entry = client->outstanding[index];
                if (entry.state.load(std::memory_order_acquire) !=
                        kOutstandingPending ||
                    !ticket_equal(entry.ticket, completion.ticket)) {
                    continue;
                }
                uint32_t expected = kOutstandingPending;
                if (!entry.state.compare_exchange_strong(
                        expected, kOutstandingReserved,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    break;
                }
                entry.completion = completion;
                koro_cont_t *continuation = entry.continuation;
                entry.continuation = nullptr;
                entry.state.store(kOutstandingReady,
                                  std::memory_order_release);
                resume_and_release(continuation, entry.ticket);
                break;
            }
        } else {
            resume_edge(client->stale_probe);
        }
        (void)send_edge(client->service_port, kMessageHostEdge,
                        client->host_generation,
                        client->client_generation);
    }
    const uint64_t capacity = load(&slot.request_head().generation);
    if (client->request_generation.exchange(
            capacity, std::memory_order_acq_rel) != capacity) {
        publish_capacity(client);
    }
    const uint64_t events = load(&client->mailbox->header.client_events);
    if (client->events.state.load(std::memory_order_acquire) ==
            kCapacityArmed &&
        events != client->events.generation) {
        resume_edge(client->events);
    }
    const ClientControl &control = slot.control();
    if (load(&control.state) == kClientFree ||
        control.client_generation != client->client_generation) {
        resume_edge(client->retirement);
    }
}

void client_source_cancelled(void *context) {
    auto *client = static_cast<Client *>(context);
    if (client->cancel_pending.fetch_sub(
            1, std::memory_order_acq_rel) == 1) {
        client->closed.store(1, std::memory_order_release);
        resume_edge(client->close);
    }
}

void abandon_claim(Client *client) {
    if (!client->mailbox || client->slot >= kClientCapacity) return;
    ClientControl &control = client->mailbox->client(client->slot).control();
    uint32_t claiming = kClientRegistering;
    if (!compare_exchange(&control.state, &claiming, kClientDead)) {
        claiming = kClientClaiming;
    }
    if (claiming == kClientClaiming &&
        !compare_exchange(&control.state, &claiming, kClientDead)) {
        return;
    }
    if (claiming == kClientRegistering || claiming == kClientClaiming) {
        (void)send_edge(client->service_port, kMessageHostEdge,
                        client->host_generation,
                        client->client_generation);
    }
}

void client_ack(Client *client, AckMessage *ack) {
    if (ack->body.msgh_descriptor_count != 1 ||
        ack->liveness.type != MACH_MSG_PORT_DESCRIPTOR ||
        ack->host_generation == 0) {
        fault_client(client);
        return;
    }
    const mach_port_t liveness = ack->liveness.name;
    if (ack->status != Ok) {
        if (liveness != MACH_PORT_NULL) {
            (void)mach_port_deallocate(mach_task_self(), liveness);
        }
        if (client->hello_complete.load(std::memory_order_acquire) != 0) {
            abandon_claim(client);
        }
        client->ready_status.store(ack->status, std::memory_order_release);
        if (client->readiness) {
            koro_cont_t *continuation = client->readiness;
            client->readiness = nullptr;
            resume_and_release(continuation, client->readiness_identity);
        }
        return;
    }
    if (client->hello_complete.exchange(1, std::memory_order_acq_rel) == 0) {
        client->host_generation = ack->host_generation;
        client->liveness_port = liveness;
        client->death_source = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MACH_SEND, liveness,
            DISPATCH_MACH_SEND_DEAD, client->queue);
        if (!client->death_source) {
            fault_client(client);
            return;
        }
        dispatch_set_context(client->death_source, client);
        dispatch_source_set_event_handler_f(client->death_source,
                                            client_death_handler);
        dispatch_source_set_cancel_handler_f(client->death_source,
                                             client_source_cancelled);
        dispatch_activate(client->death_source);
        char error[256]{};
        if (ack->slot >= kClientCapacity ||
            ack->client_generation == 0) {
            fault_client(client);
            return;
        }
        client->slot = ack->slot;
        client->client_generation = ack->client_generation;
        int status = map_mailbox_client(client, error, sizeof(error));
        if (status == Ok) {
            const ClientControl &control =
                client->mailbox->client(client->slot).control();
            if (load(&control.state) != kClientRegistering ||
                control.client_generation != client->client_generation ||
                control.host_generation != client->host_generation ||
                control.client_pid != static_cast<uint64_t>(getpid()) ||
                control.client_start_time != process_start_time(getpid()) ||
                control.client_uid != static_cast<uint64_t>(geteuid())) {
                status = Stale;
            }
        }
        if (status == Ok) {
            client->request_generation.store(
                load(&client->mailbox->client(client->slot)
                          .request_head()
                          .generation),
                std::memory_order_release);
            status = send_registration(
                client, kMessageRegister, client->slot,
                client->host_generation, client->client_generation);
        }
        if (status == Ok) return;
        abandon_claim(client);
        client->ready_status.store(status, std::memory_order_release);
    } else {
        if (liveness != MACH_PORT_NULL) {
            (void)mach_port_deallocate(mach_task_self(), liveness);
        }
        if (ack->slot != client->slot ||
            ack->client_generation != client->client_generation ||
            ack->host_generation != client->host_generation) {
            abandon_claim(client);
            client->ready_status.store(Stale,
                                       std::memory_order_release);
        } else {
            client->ready_status.store(Ok,
                                       std::memory_order_release);
        }
    }
    if (client->readiness) {
        koro_cont_t *continuation = client->readiness;
        client->readiness = nullptr;
        resume_and_release(continuation, client->readiness_identity);
    }
}

void client_receive_handler(void *context) {
    auto *client = static_cast<Client *>(context);
    ReceiveBuffer buffer{};
    for (;;) {
        const int received = receive_one(client->wake_port, &buffer);
        if (received <= 0) break;
        auto *header = reinterpret_cast<mach_msg_header_t *>(&buffer);
        if (header->msgh_id == kMessageAck &&
            header->msgh_size == sizeof(AckMessage) &&
            (header->msgh_bits & MACH_MSGH_BITS_COMPLEX) != 0) {
            client_ack(client, reinterpret_cast<AckMessage *>(header));
            continue;
        }
        if (header->msgh_id == kMessageClientEdge &&
            header->msgh_size == sizeof(EdgeMessage)) {
            const auto *edge = reinterpret_cast<const EdgeMessage *>(header);
            if (edge->host_generation == client->host_generation &&
                edge->client_generation == client->client_generation) {
                client_drain(client);
            }
        }
    }
}

} // namespace

#else

struct Server {};
struct Client {};

#endif

#ifdef __APPLE__

namespace {

int register_service(Server *server) {
    if ((server->flags & kTestService) == 0) {
        const kern_return_t status = bootstrap_check_in(
            bootstrap_port, server->service.c_str(), &server->service_port);
        return status == KERN_SUCCESS ? Ok : IoError;
    }
    kern_return_t status = mach_port_allocate(
        mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &server->service_port);
    if (status == KERN_SUCCESS) {
        status = mach_port_insert_right(
            mach_task_self(), server->service_port, server->service_port,
            MACH_MSG_TYPE_MAKE_SEND);
    }
    if (status == KERN_SUCCESS) {
        /* Production uses launchd bootstrap_check_in. Dynamic registration is
         * confined to the native hostile-lifecycle executable so it can own a
         * fresh service name without installing a launchd job. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        status = bootstrap_register(
            bootstrap_port, const_cast<char *>(server->service.c_str()),
            server->service_port);
#pragma clang diagnostic pop
    }
    return status == KERN_SUCCESS ? Ok : IoError;
}

int create_liveness_port(Server *server) {
    kern_return_t status = mach_port_allocate(
        mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &server->liveness_port);
    if (status == KERN_SUCCESS) {
        status = mach_port_insert_right(
            mach_task_self(), server->liveness_port, server->liveness_port,
            MACH_MSG_TYPE_MAKE_SEND);
    }
    return status == KERN_SUCCESS ? Ok : IoError;
}

int create_mailbox(Server *server, char *error,
                   size_t error_length) {
    server->name = mailbox_name(server->service.c_str());
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
        server->fd = shm_open(server->name.c_str(),
                              O_RDWR | O_CREAT | O_EXCL, 0600);
        if (server->fd >= 0) break;
        if (errno != EEXIST || attempt != 0) {
            set_error(error, error_length,
                      "cannot create host mailbox: " +
                          std::string(std::strerror(errno)));
            return IoError;
        }
        const int stale = shm_open(server->name.c_str(), O_RDONLY, 0);
        if (stale < 0) continue;
        struct stat info {};
        MailboxHeader header{};
        bool readable = fstat(stale, &info) == 0 &&
                        info.st_uid == geteuid() &&
                        info.st_size >=
                            static_cast<off_t>(sizeof(MailboxHeader));
        if (readable) {
            void *view = mmap(nullptr, sizeof(MailboxHeader), PROT_READ,
                              MAP_SHARED, stale, 0);
            readable = view != MAP_FAILED;
            if (readable) {
                header = *static_cast<const MailboxHeader *>(view);
                (void)munmap(view, sizeof(MailboxHeader));
            }
        }
        (void)::close(stale);
        if (!readable ||
            process_alive(header.host_pid, header.host_start_time)) {
            set_error(error, error_length,
                      "host mailbox name is owned by a live or invalid host");
            return Rejected;
        }
        (void)shm_unlink(server->name.c_str());
    }
    if (server->fd < 0) return IoError;
    const size_t mapping_bytes = mailbox_mapping_bytes();
    if (ftruncate(server->fd, static_cast<off_t>(mapping_bytes)) != 0) {
        set_error(error, error_length,
                  "cannot size host mailbox: " +
                      std::string(std::strerror(errno)));
        return IoError;
    }
    void *memory = mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, server->fd, 0);
    if (memory == MAP_FAILED) {
        set_error(error, error_length,
                  "cannot map host mailbox: " +
                      std::string(std::strerror(errno)));
        return IoError;
    }
    std::memset(memory, 0, mapping_bytes);
    server->mailbox = ::new (memory) Mailbox();
    MailboxHeader &header = server->mailbox->header;
    header.host_generation = generation();
    header.host_pid = static_cast<uint64_t>(getpid());
    header.host_start_time = process_start_time(getpid());
    header.host_uid = static_cast<uint64_t>(geteuid());
    store(&header.state, kMailboxInitializing);
    std::memcpy(header.magic, kMailboxMagic, sizeof(kMailboxMagic));
    header.size = kMailboxBytes;
    header.layout_version = kLayoutVersion;
    header.client_capacity = kClientCapacity;
    header.ring_capacity = kRingCapacity;
    header.segment_generation = server->weights.generation;
    header.checkpoint_identity =
        identity_from_bytes(server->weights.identity_digest);
    header.content_digest = identity_from_bytes(server->weights.content_digest);
    for (uint32_t index = 0; index < kClientCapacity; ++index) {
        initialize_slot(server->mailbox->client(index));
    }
    return header.host_start_time == 0 ? IoError : Ok;
}

int send_ack(Server *server, mach_port_t port, int32_t status,
             uint32_t slot, uint64_t client_generation) {
    AckMessage message{};
    message.header.msgh_bits =
        MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) |
        MACH_MSGH_BITS_COMPLEX;
    message.header.msgh_size = sizeof(message);
    message.header.msgh_remote_port = port;
    message.header.msgh_id = kMessageAck;
    message.body.msgh_descriptor_count = 1;
    message.liveness.name = server->liveness_port;
    message.liveness.disposition = MACH_MSG_TYPE_COPY_SEND;
    message.liveness.type = MACH_MSG_PORT_DESCRIPTOR;
    message.status = status;
    message.slot = slot;
    message.host_generation = server->mailbox->header.host_generation;
    message.client_generation = client_generation;
    return send_message(&message.header, false);
}

void resume_server(Server *server) {
    if (server && server->continuation) {
        (void)koro_cont_resume(server->continuation, &server->identity);
    }
}

void server_process_exit(void *context) {
    auto *local = static_cast<ServerClient *>(context);
    Server *server = local->server;
    server->dead_clients.fetch_or(UINT64_C(1) << local->slot,
                                  std::memory_order_acq_rel);
    resume_server(server);
}

void server_process_cancelled(void *context) {
    auto *local = static_cast<ServerClient *>(context);
    const dispatch_source_t source =
        local->process_source.exchange(nullptr, std::memory_order_acq_rel);
    local->process_cancelled.store(1, std::memory_order_release);
    resume_server(local->server);
    if (source) dispatch_release(source);
}

bool install_process_source(ServerClient &local, pid_t pid) {
    const dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_PROC, static_cast<uintptr_t>(pid),
        DISPATCH_PROC_EXIT, local.server->queue);
    if (!source) return false;
    local.process_cancelled.store(0, std::memory_order_release);
    local.cancel_requested.store(0, std::memory_order_release);
    local.process_source.store(source, std::memory_order_release);
    dispatch_set_context(source, &local);
    dispatch_source_set_event_handler_f(source, server_process_exit);
    dispatch_source_set_cancel_handler_f(source, server_process_cancelled);
    dispatch_activate(source);
    return true;
}

void reject_registration(mach_port_t port) {
    if (port != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), port);
    }
}

int reserve_client(Server *server, const RegisterMessage &message,
                   pid_t pid, uid_t uid, uint32_t *slot_out,
                   uint64_t *generation_out) {
    if (load(&server->mailbox->header.state) != kMailboxReady) {
        return HostDown;
    }
    const uint64_t started = process_start_time(pid);
    if (uid != geteuid() || started == 0) return Denied;
    for (uint32_t index = 0; index < kClientCapacity; ++index) {
        ClientSlotView slot = server->mailbox->client(index);
        ClientControl &control = slot.control();
        ServerClient &local = server->client(index);
        if (local.wake_port.load(std::memory_order_acquire) !=
                MACH_PORT_NULL ||
            local.process_cancelled.load(std::memory_order_acquire) == 0) {
            continue;
        }
        uint32_t free = kClientFree;
        if (!compare_exchange(&control.state, &free, kClientClaiming)) {
            continue;
        }
        uint64_t next = load(&control.client_generation,
                             __ATOMIC_RELAXED) + 1;
        if (next == 0) next = 1;
        control.flags = message.flags;
        control.client_generation = next;
        control.host_generation = server->mailbox->header.host_generation;
        control.client_pid = static_cast<uint64_t>(pid);
        control.client_start_time = started;
        control.client_uid = static_cast<uint64_t>(uid);
        control.active_lease_generation = 0;
        control.lease_count = 0;
        control.registered = 0;
        local.generation.store(next, std::memory_order_release);
        const bool privileged =
            (message.flags & kPrivilegedClient) != 0 &&
            server->privileged_pid != 0 &&
            server->privileged_pid == static_cast<uint64_t>(pid);
        local.privileged.store(privileged ? 1u : 0u,
                               std::memory_order_release);
        local.wake_port.store(message.wake.name,
                              std::memory_order_release);
        if (!install_process_source(local, pid)) {
            local.wake_port.store(MACH_PORT_NULL,
                                  std::memory_order_release);
            local.generation.store(0, std::memory_order_release);
            local.privileged.store(0, std::memory_order_release);
            initialize_slot(slot, next);
            store(&slot.control().state, kClientFree);
            return IoError;
        }
        control.flags = privileged ? kPrivilegedClient : 0;
        store(&control.state, kClientRegistering);
        *slot_out = index;
        *generation_out = next;
        return Ok;
    }
    return Capacity;
}

void accept_registration(Server *server,
                         const RegisterMessage &message,
                         const audit_token_t &audit) {
    const mach_port_t port = message.wake.name;
    const pid_t pid = audit_token_to_pid(audit);
    const uid_t uid = audit_token_to_euid(audit);
    if (message.header.msgh_id == kMessageHello) {
        uint32_t slot = UINT32_MAX;
        uint64_t client_generation = 0;
        const int32_t status = reserve_client(
            server, message, pid, uid, &slot, &client_generation);
        const int sent = send_ack(server, port, status, slot,
                                  client_generation);
        if (status != Ok) {
            reject_registration(port);
            return;
        }
        if (sent != Ok) {
            server->dead_clients.fetch_or(UINT64_C(1) << slot,
                                          std::memory_order_acq_rel);
        }
        resume_server(server);
        return;
    }
    if (message.header.msgh_id != kMessageRegister ||
        message.slot >= kClientCapacity ||
        message.host_generation != server->mailbox->header.host_generation) {
        (void)send_ack(server, port, Stale, message.slot,
                       message.client_generation);
        reject_registration(port);
        return;
    }
    ClientSlotView slot = server->mailbox->client(message.slot);
    ClientControl &control = slot.control();
    const bool valid =
                       load(&server->mailbox->header.state) == kMailboxReady &&
                       load(&control.state) == kClientRegistering &&
                       control.client_generation ==
                           message.client_generation &&
                       control.host_generation ==
                           message.host_generation &&
                       control.client_pid == static_cast<uint64_t>(pid) &&
                       control.client_uid == static_cast<uint64_t>(uid) &&
                       uid == geteuid() &&
                       process_start_time(pid) == control.client_start_time;
    ServerClient &local = server->client(message.slot);
    const mach_port_t registered_port =
        local.wake_port.load(std::memory_order_acquire);
    if (!valid || registered_port == MACH_PORT_NULL ||
        local.generation.load(std::memory_order_acquire) !=
            message.client_generation) {
        if ((server->flags & kTestService) != 0) {
            std::fprintf(
                stderr,
                "registration rejected: valid=%u slot=%u state=%u "
                "pid=%d/%llu uid=%u/%llu start=%llu/%llu host=%llu/%llu "
                "generation=%llu/%llu local_port=%u\n",
                valid ? 1u : 0u, message.slot, load(&control.state), pid,
                static_cast<unsigned long long>(control.client_pid), uid,
                static_cast<unsigned long long>(control.client_uid),
                static_cast<unsigned long long>(process_start_time(pid)),
                static_cast<unsigned long long>(control.client_start_time),
                static_cast<unsigned long long>(message.host_generation),
                static_cast<unsigned long long>(control.host_generation),
                static_cast<unsigned long long>(message.client_generation),
                static_cast<unsigned long long>(control.client_generation),
                local.wake_port.load(std::memory_order_acquire));
            std::fflush(stderr);
        }
        (void)send_ack(server, port, Rejected, message.slot,
                       message.client_generation);
        reject_registration(port);
        return;
    }
    uint32_t registering = kClientRegistering;
    if (!compare_exchange(&control.state, &registering,
                          kClientActivating)) {
        (void)send_ack(server, registered_port, Stale, message.slot,
                       message.client_generation);
        reject_registration(port);
        resume_server(server);
        return;
    }
    store(&control.registered, 1u);
    fetch_add(&server->mailbox->header.active_clients, UINT64_C(1));
    fetch_add(&server->mailbox->header.client_events, UINT64_C(1));
    store(&control.state, kClientActive);
    if (send_ack(server, registered_port, Ok, message.slot,
                 message.client_generation) != Ok) {
        server->dead_clients.fetch_or(UINT64_C(1) << message.slot,
                                      std::memory_order_acq_rel);
    }
    reject_registration(port);
    resume_server(server);
}

void server_receive_handler(void *context) {
    auto *server = static_cast<Server *>(context);
    ReceiveBuffer buffer{};
    for (;;) {
        const int received = receive_one(server->service_port, &buffer);
        if (received <= 0) break;
        auto *header = reinterpret_cast<mach_msg_header_t *>(&buffer);
        if ((header->msgh_id == kMessageHello ||
             header->msgh_id == kMessageRegister) &&
            message_has_port(header)) {
            accept_registration(
                server, *reinterpret_cast<RegisterMessage *>(header),
                audit_token(header));
            continue;
        }
        if (header->msgh_id == kMessageHostEdge &&
            header->msgh_size == sizeof(EdgeMessage)) {
            resume_server(server);
            continue;
        }
        if ((header->msgh_bits & MACH_MSGH_BITS_COMPLEX) != 0) {
            mach_msg_destroy(header);
        }
    }
}

void notify_client(Server *server, uint32_t index) {
    ServerClient &local = server->client(index);
    const mach_port_t port =
        local.wake_port.load(std::memory_order_acquire);
    const uint64_t generation =
        local.generation.load(std::memory_order_acquire);
    if (send_edge(port, kMessageClientEdge,
                  server->mailbox->header.host_generation,
                  generation) == HostDown) {
        server->dead_clients.fetch_or(UINT64_C(1) << index,
                                      std::memory_order_acq_rel);
    }
}

void notify_all_clients(Server *server) {
    for (uint32_t index = 0; index < kClientCapacity;
         ++index) {
        if (server->client(index).wake_port.load(
                std::memory_order_acquire) != MACH_PORT_NULL) {
            notify_client(server, index);
        }
    }
}

void retire_client(Server *server, uint32_t index) {
    ClientSlotView slot = server->mailbox->client(index);
    ClientControl &control = slot.control();
    const uint32_t state = load(&control.state);
    if (state != kClientActive && state != kClientRetiring &&
        state != kClientDead && state != kClientClaiming &&
        state != kClientRegistering) {
        return;
    }
    if (state != kClientDead) store(&control.state, kClientDead);
    const uint64_t prior_generation = control.client_generation;
    if (control.lease_count != 0) {
        control.lease_count = 0;
        control.active_lease_generation = 0;
        fetch_add(&server->mailbox->header.active_leases,
                  UINT64_MAX);
    }
    ServerClient &local = server->client(index);
    local.pending = false;
    if (control.registered != 0) {
        control.registered = 0;
        fetch_add(&server->mailbox->header.active_clients, UINT64_MAX);
    }
    const dispatch_source_t source =
        local.process_source.load(std::memory_order_acquire);
    if (source &&
        local.process_cancelled.load(std::memory_order_acquire) == 0) {
        if (local.cancel_requested.exchange(
                1, std::memory_order_acq_rel) == 0) {
            dispatch_source_cancel(source);
        }
        return;
    }
    const mach_port_t port =
        local.wake_port.exchange(MACH_PORT_NULL, std::memory_order_acq_rel);
    const uint64_t generation =
        local.generation.exchange(0, std::memory_order_acq_rel);
    local.privileged.store(0, std::memory_order_release);
    local.cancel_requested.store(0, std::memory_order_release);
    initialize_slot(slot, prior_generation);
    store(&slot.control().state, kClientFree);
    fetch_add(&server->mailbox->header.client_events, UINT64_C(1));
    if (port != MACH_PORT_NULL) {
        (void)send_edge(port, kMessageClientEdge,
                        server->mailbox->header.host_generation,
                        generation);
        (void)mach_port_deallocate(mach_task_self(), port);
    }
    notify_all_clients(server);
}

bool request_valid(const Server *server,
                   uint32_t index, const HostRequest &request) {
    const ClientControl &control =
        server->mailbox->client(index).control();
    return request.size == sizeof(request) &&
           request.layout_version == kLayoutVersion &&
           ticket_valid(request.ticket) && ticket_valid(request.parent) &&
           request.host_generation ==
               server->mailbox->header.host_generation &&
           request.client_generation == control.client_generation &&
           identity_equal(request.checkpoint_identity,
                          server->mailbox->header.checkpoint_identity) &&
           request.operation >= Attach &&
           request.operation <= Evict;
}

HostCompletion execute_request(Server *server,
                                    uint32_t index,
                                    const HostRequest &request) {
    HostCompletion completion{
        .size = sizeof(completion),
        .layout_version = kLayoutVersion,
        .ticket = request.ticket,
        .parent = request.parent,
        .host_generation = request.host_generation,
        .client_generation = request.client_generation,
        .lease_generation = request.lease_generation,
        .operation = request.operation,
        .status = Ok,
    };
    completion.checkpoint_identity = request.checkpoint_identity;
    ClientControl &control = server->mailbox->client(index).control();
    if (!request_valid(server, index, request)) {
        completion.status = Stale;
        return completion;
    }
    switch (request.operation) {
    case Attach: {
        if ((load(&server->mailbox->header.flags) & kMailboxEvicted) != 0) {
            completion.status = Rejected;
            break;
        }
        if (control.lease_count != 0) {
            completion.status = Busy;
            completion.lease_generation =
                control.active_lease_generation;
            break;
        }
        uint64_t lease = control.active_lease_generation + 1;
        if (lease == 0) lease = 1;
        control.active_lease_generation = lease;
        control.lease_count = 1;
        completion.lease_generation = lease;
        fetch_add(&server->mailbox->header.active_leases, UINT64_C(1));
        break;
    }
    case Release:
        if (control.lease_count != 1 || request.lease_generation == 0 ||
            request.lease_generation !=
                control.active_lease_generation) {
            completion.status = Stale;
            break;
        }
        control.lease_count = 0;
        completion.lease_generation = control.active_lease_generation;
        fetch_add(&server->mailbox->header.active_leases, UINT64_MAX);
        break;
    case QueryStatus: {
        const uint64_t clients =
            load(&server->mailbox->header.active_clients);
        const uint64_t leases =
            load(&server->mailbox->header.active_leases);
        completion.lease_generation =
            control.lease_count ? control.active_lease_generation : 0;
        completion.result_flags =
            static_cast<uint32_t>(std::min<uint64_t>(clients, UINT16_MAX)) |
            (static_cast<uint32_t>(
                 std::min<uint64_t>(leases, UINT16_MAX))
             << 16);
        break;
    }
    case Evict: {
        if (server->client(index).privileged.load(
                std::memory_order_acquire) == 0) {
            completion.status = Denied;
            break;
        }
        if (load(&server->mailbox->header.active_leases) != 0) {
            completion.status = Busy;
            break;
        }
        char error[512]{};
        const int evicted = lfm_weights_evict(
            identity_bytes(server->mailbox->header.checkpoint_identity),
            error, sizeof(error));
        if (evicted != LFM_WEIGHT_OK) {
            completion.status = IoError;
            break;
        }
        __atomic_fetch_or(&server->mailbox->header.flags, kMailboxEvicted,
                          __ATOMIC_ACQ_REL);
        break;
    }
    default:
        completion.status = InvalidArgument;
        break;
    }
    return completion;
}

bool drain_server(Server *server) {
    uint32_t drained = 0;
    const uint64_t dead =
        server->dead_clients.exchange(0, std::memory_order_acq_rel);
    for (uint32_t index = 0; index < kClientCapacity;
         ++index) {
        ClientSlotView slot = server->mailbox->client(index);
        const uint32_t state = load(&slot.control().state);
        if ((dead & (UINT64_C(1) << index)) != 0 &&
            (state == kClientClaiming ||
             state == kClientActivating)) {
            server->dead_clients.fetch_or(UINT64_C(1) << index,
                                          std::memory_order_acq_rel);
            continue;
        }
        if ((dead & (UINT64_C(1) << index)) != 0 ||
            state == kClientRetiring || state == kClientDead ||
            (state == kClientRegistering &&
             !process_alive(slot.control().client_pid,
                            slot.control().client_start_time))) {
            retire_client(server, index);
            continue;
        }
        if (state != kClientActive) continue;
        ServerClient &local = server->client(index);
        if (local.pending) {
            if (!completion_push(slot, local.pending_completion)) continue;
            local.pending = false;
            notify_client(server, index);
        }
        while (drained < kDrainBudget && !local.pending) {
            HostRequest request{};
            if (!request_pop(slot, &request)) break;
            notify_client(server, index);
            HostCompletion completion =
                execute_request(server, index, request);
            if (!completion_push(slot, completion)) {
                local.pending = true;
                local.pending_completion = completion;
                break;
            }
            notify_client(server, index);
            ++drained;
        }
    }
    if (drained == kDrainBudget) return true;
    for (uint32_t index = 0; index < kClientCapacity;
         ++index) {
        const ClientSlotView slot = server->mailbox->client(index);
        if (load(&slot.control().state) == kClientActive &&
            (server->client(index).pending || request_ready(slot))) {
            return true;
        }
    }
    return false;
}

bool stop_server(Server *server) {
    store(&server->mailbox->header.state, kMailboxStopping);
    bool pending = false;
    for (uint32_t index = 0; index < kClientCapacity;
         ++index) {
        const uint32_t state =
            load(&server->mailbox->client(index).control().state);
        if (state == kClientClaiming || state == kClientActivating) {
            pending = true;
            continue;
        }
        if (state != kClientFree) {
            notify_client(server, index);
            retire_client(server, index);
        }
        if (load(&server->mailbox->client(index).control().state) !=
            kClientFree) {
            pending = true;
        }
    }
    if (pending) return false;
    store(&server->mailbox->header.state, kMailboxDead);
    (void)shm_unlink(server->name.c_str());
    return true;
}

void *server_step(koro_cont_t *continuation) {
    auto *server = static_cast<Server *>(
        koro_cont_argument(continuation));
    KORO_BEGIN(continuation);
    for (;;) {
        if (server->stop_requested.load(std::memory_order_acquire) != 0) {
            if (stop_server(server)) break;
            KORO_SUSPEND(continuation);
        }
        if (drain_server(server)) {
            KORO_YIELD(continuation);
        }
        KORO_SUSPEND(continuation);
    }
    KORO_END(continuation);
}

void server_retired(void *context, const kc_ticket_id *) {
    auto *server = static_cast<Server *>(context);
    if (server->image) {
        lfm_weights_close(server->image);
        server->image = nullptr;
    }
    server->retired.store(1, std::memory_order_release);
    if (server->exit_on_retire) std::_Exit(EXIT_SUCCESS);
}

void server_signal(void *context) {
    server_request_stop(
        static_cast<Server *>(context));
}

} // namespace

#endif

Status server_create(const ServerConfig &config, Server **out,
                     std::string *error) {
    if (error) error->clear();
    if (!out || config.checkpoint.empty() || config.service.empty() ||
        config.coordination_workers == 0) {
        if (error) *error = "invalid native host mailbox server config";
        return InvalidArgument;
    }
    *out = nullptr;
#ifndef __APPLE__
    if (error) {
        *error = "the production host mailbox requires macOS Mach services";
    }
    return Unsupported;
#else
    auto *server = new (std::nothrow) Server();
    if (!server) return IoError;
    server->clients = allocate_records<ServerClient>(kClientCapacity);
    if (!server->clients) {
        delete server;
        return IoError;
    }
    server->checkpoint = config.checkpoint;
    server->service = config.service;
    server->flags = config.flags;
    server->privileged_pid = config.privileged_pid;
    server->readiness_fd = config.readiness_fd;
    char detail[512]{};
    int status = open_image(server->checkpoint, &server->image,
                            detail, sizeof(detail));
    if (status != LFM_WEIGHT_OK) {
        if (error) *error = detail;
        destroy_unstarted_server(server);
        return IoError;
    }
    server->weights = {
        .size = sizeof(server->weights),
        .abi_version = LFM_WEIGHT_ABI_VERSION,
    };
    if (lfm_weights_load_stats(server->image, &server->weights) !=
        LFM_WEIGHT_OK) {
        if (error) *error = "cannot read native shared-weight metadata";
        destroy_unstarted_server(server);
        return IoError;
    }
    status = create_mailbox(server, detail, sizeof(detail));
    if (status == Ok) status = register_service(server);
    if (status == Ok) status = create_liveness_port(server);
    kc_runtime_config runtime_config{
        .size = sizeof(runtime_config),
        .abi_version = KC_ABI_VERSION,
        .worker_count = config.coordination_workers,
    };
    if (status == Ok &&
        kc_runtime_create(&runtime_config, &server->runtime) != 0) {
        status = IoError;
    }
    koro_cont_config continuation_config{
        .size = sizeof(continuation_config),
        .abi_version = KC_ABI_VERSION,
        .step = server_step,
        .argument = server,
        .frame_size = sizeof(uint64_t),
        .worker_mask = 0,
        .completion = server_retired,
        .completion_context = server,
    };
    if (status == Ok &&
        koro_cont_create_on(server->runtime, &continuation_config,
                            &server->continuation) != 0) {
        status = IoError;
    }
    if (status == Ok) {
        server->identity = koro_cont_identity(server->continuation);
        server->queue = dispatch_queue_create(
            "com.solaceharmony.lfm.host-mailbox", DISPATCH_QUEUE_SERIAL);
        if (!server->queue) status = IoError;
    }
    if (status == Ok) {
        server->receive_source = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MACH_RECV, server->service_port, 0,
            server->queue);
        if (!server->receive_source) status = IoError;
    }
    if (status != Ok) {
        if (error) {
            *error = detail[0] ? detail
                               : "cannot construct native host mailbox service";
        }
        destroy_unstarted_server(server);
        return static_cast<Status>(status);
    }
    for (uint32_t index = 0; index < kClientCapacity; ++index) {
        server->client(index).server = server;
        server->client(index).slot = index;
    }
    dispatch_set_context(server->receive_source, server);
    dispatch_source_set_event_handler_f(server->receive_source,
                                        server_receive_handler);
    *out = server;
    return Ok;
#endif
}

Status server_start(Server *server) {
#ifndef __APPLE__
    (void)server;
    return Unsupported;
#else
    if (!server || !server->runtime || !server->continuation ||
        !server->mailbox || load(&server->mailbox->header.state) !=
                                kMailboxInitializing) {
        return InvalidArgument;
    }
    int status = kc_runtime_start(server->runtime);
    if (status == 0) status = koro_cont_start(server->continuation);
    if (status != 0) return IoError;
    store(&server->mailbox->header.state, kMailboxReady);
    dispatch_activate(server->receive_source);
    if (server->readiness_fd >= 0) {
        char ready[128]{};
        const int bytes = std::snprintf(
            ready, sizeof(ready), "READY %llu\n",
            static_cast<unsigned long long>(
                server->mailbox->header.host_generation));
        if (bytes > 0) {
            (void)::write(server->readiness_fd, ready,
                          static_cast<size_t>(bytes));
        }
        (void)::close(server->readiness_fd);
        server->readiness_fd = -1;
    }
    return Ok;
#endif
}

void server_request_stop(Server *server) {
#ifdef __APPLE__
    if (!server || server->stop_requested.exchange(
                       1, std::memory_order_acq_rel) != 0) {
        return;
    }
    resume_server(server);
#else
    (void)server;
#endif
}

Status server_snapshot(const Server *server, HostSnapshot *out) {
#ifndef __APPLE__
    (void)server;
    (void)out;
    return Unsupported;
#else
    if (!server || !server->mailbox || !out) return InvalidArgument;
    snapshot(server->mailbox, out);
    return Ok;
#endif
}

Status client_create(const ClientConfig &config, koro_cont_t *readiness,
                     Client **out, std::string *error) {
    if (error) error->clear();
    if (!out || config.service.empty() || !config.runtime || !readiness) {
        if (error) *error = "invalid native host mailbox client config";
        return InvalidArgument;
    }
    *out = nullptr;
#ifndef __APPLE__
    if (error) {
        *error = "the production host mailbox requires macOS Mach services";
    }
    return Unsupported;
#else
    auto *client = new (std::nothrow) Client();
    if (!client) return IoError;
    client->outstanding = allocate_records<Outstanding>(kRingCapacity);
    if (!client->outstanding) {
        delete client;
        return IoError;
    }
    client->service = config.service;
    client->runtime = config.runtime;
    client->flags = config.flags;
    client->readiness = readiness;
    client->readiness_identity = koro_cont_identity(readiness);
    koro_cont_retain(readiness);
    kern_return_t mach = bootstrap_look_up(
        bootstrap_port, client->service.c_str(), &client->service_port);
    if (mach == KERN_SUCCESS) {
        mach = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE,
                                  &client->wake_port);
    }
    if (mach != KERN_SUCCESS) {
        if (client->service_port != MACH_PORT_NULL) {
            (void)mach_port_deallocate(mach_task_self(),
                                       client->service_port);
        }
        if (client->wake_port != MACH_PORT_NULL) {
            (void)mach_port_mod_refs(mach_task_self(), client->wake_port,
                                     MACH_PORT_RIGHT_RECEIVE, -1);
        }
        koro_cont_release(readiness);
        release_records(client->outstanding, kRingCapacity);
        delete client;
        if (error) *error = "cannot resolve native host Mach service";
        return HostDown;
    }
    client->queue = dispatch_queue_create(
        "com.solaceharmony.lfm.host-client", DISPATCH_QUEUE_SERIAL);
    if (client->queue) {
        client->receive_source = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MACH_RECV, client->wake_port, 0,
            client->queue);
    }
    if (!client->queue || !client->receive_source) {
        if (client->receive_source) dispatch_release(client->receive_source);
        if (client->queue) dispatch_release(client->queue);
        (void)mach_port_deallocate(mach_task_self(), client->service_port);
        (void)mach_port_mod_refs(mach_task_self(), client->wake_port,
                                 MACH_PORT_RIGHT_RECEIVE, -1);
        koro_cont_release(readiness);
        release_records(client->outstanding, kRingCapacity);
        delete client;
        return IoError;
    }
    dispatch_set_context(client->receive_source, client);
    dispatch_source_set_event_handler_f(client->receive_source,
                                        client_receive_handler);
    dispatch_source_set_cancel_handler_f(client->receive_source,
                                         client_source_cancelled);
    /* Publish ownership before activating any callback that can resume the
     * caller's readiness continuation. */
    *out = client;
    dispatch_activate(client->receive_source);
    if (send_registration(client, kMessageHello, UINT32_MAX, 0, 0) != Ok) {
        fault_client(client);
    }
    return InProgress;
#endif
}

Status client_ready(const Client *client) {
#ifndef __APPLE__
    (void)client;
    return Unsupported;
#else
    if (!client) return InvalidArgument;
    return static_cast<Status>(
        client->ready_status.load(std::memory_order_acquire));
#endif
}

Status client_submit(Client *client, Operation operation,
                     const kc_ticket_id &parent,
                     uint64_t lease_generation,
                     koro_cont_t *continuation) {
#ifndef __APPLE__
    (void)client;
    (void)operation;
    (void)parent;
    (void)lease_generation;
    (void)continuation;
    return Unsupported;
#else
    if (!client || !continuation || !ticket_valid(parent) ||
        operation < Attach || operation > Evict) {
        return InvalidArgument;
    }
    if (client->ready_status.load(std::memory_order_acquire) != Ok ||
        client->host_dead.load(std::memory_order_acquire) != 0 ||
        !client->mailbox || client->slot >= kClientCapacity) {
        return HostDown;
    }
    uint32_t idle = 0;
    if (!client->producer.compare_exchange_strong(
            idle, 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    ClientSlotView slot = client->mailbox->client(client->slot);
    const kc_ticket_id identity = koro_cont_identity(continuation);
    if (!request_room(slot)) {
        const Status status = dehydrate_for_capacity(client, continuation);
        client->producer.store(0, std::memory_order_release);
        return status;
    }
    Outstanding *entry = nullptr;
    for (uint32_t index = 0; index < kRingCapacity; ++index) {
        Outstanding &candidate = client->outstanding[index];
        uint32_t free = kOutstandingFree;
        if (candidate.state.compare_exchange_strong(
                free, kOutstandingReserved, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            entry = &candidate;
            break;
        }
    }
    if (!entry) {
        const Status status = dehydrate_for_capacity(client, continuation);
        client->producer.store(0, std::memory_order_release);
        return status;
    }
    HostRequest request{
        .size = sizeof(request),
        .layout_version = kLayoutVersion,
        .ticket = identity,
        .parent = parent,
        .host_generation = client->host_generation,
        .client_generation = client->client_generation,
        .lease_generation = lease_generation,
        .operation = static_cast<uint32_t>(operation),
        .flags = client->flags,
    };
    request.checkpoint_identity =
        client->mailbox->header.checkpoint_identity;
    entry->ticket = identity;
    entry->request = request;
    entry->completion = {};
    entry->continuation = continuation;
    koro_cont_retain(continuation);
    entry->state.store(kOutstandingPending, std::memory_order_release);
    if (!request_push(slot, request)) {
        entry->state.store(kOutstandingFree, std::memory_order_release);
        entry->continuation = nullptr;
        koro_cont_release(continuation);
        client->producer.store(0, std::memory_order_release);
        return Capacity;
    }
    if (send_edge(client->service_port, kMessageHostEdge,
                  client->host_generation,
                  client->client_generation) == HostDown ||
        client->host_dead.load(std::memory_order_acquire) != 0) {
        fault_client(client);
    }
    client->producer.store(0, std::memory_order_release);
    return InProgress;
#endif
}

Status client_take(Client *client, const kc_ticket_id &ticket,
                   HostCompletion *out) {
#ifndef __APPLE__
    (void)client;
    (void)ticket;
    (void)out;
    return Unsupported;
#else
    if (!client || !out) return InvalidArgument;
    for (uint32_t index = 0; index < kRingCapacity; ++index) {
        Outstanding &entry = client->outstanding[index];
        if (!ticket_equal(entry.ticket, ticket)) continue;
        const uint32_t state = entry.state.load(std::memory_order_acquire);
        if (state != kOutstandingReady && state != kOutstandingFaulted) {
            return InProgress;
        }
        *out = entry.completion;
        entry.ticket = {};
        entry.request = {};
        entry.completion = {};
        entry.state.store(kOutstandingFree, std::memory_order_release);
        publish_capacity(client);
        return Ok;
    }
    return Stale;
#endif
}

Status client_snapshot(const Client *client, HostSnapshot *out) {
#ifndef __APPLE__
    (void)client;
    (void)out;
    return Unsupported;
#else
    if (!client || !client->mailbox || !out) return InvalidArgument;
    snapshot(client->mailbox, out);
    return Ok;
#endif
}

Status client_watch(Client *client, uint64_t observed,
                    koro_cont_t *continuation) {
#ifndef __APPLE__
    (void)client;
    (void)observed;
    (void)continuation;
    return Unsupported;
#else
    if (!client || !continuation || !client->mailbox) {
        return InvalidArgument;
    }
    if (client->host_dead.load(std::memory_order_acquire) != 0) {
        return HostDown;
    }
    uint32_t free = kCapacityFree;
    if (!client->events.state.compare_exchange_strong(
            free, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    client->events.generation = observed;
    client->events.identity = koro_cont_identity(continuation);
    client->events.continuation = continuation;
    koro_cont_retain(continuation);
    client->events.state.store(kCapacityArmed, std::memory_order_release);
    const uint64_t current = load(&client->mailbox->header.client_events);
    if (current != observed ||
        client->host_dead.load(std::memory_order_acquire) != 0) {
        resume_edge(client->events);
    }
    return InProgress;
#endif
}

Status client_request_stop(Client *client,
                           koro_cont_t *continuation) {
#ifdef __APPLE__
    if (!client || !continuation || !client->mailbox ||
        client->slot >= kClientCapacity) {
        return InvalidArgument;
    }
    for (uint32_t index = 0; index < kRingCapacity; ++index) {
        if (client->outstanding[index].state.load(std::memory_order_acquire) !=
            kOutstandingFree) {
            return Busy;
        }
    }
    uint32_t free = kCapacityFree;
    if (!client->retirement.state.compare_exchange_strong(
            free, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    client->retirement.identity = koro_cont_identity(continuation);
    client->retirement.continuation = continuation;
    koro_cont_retain(continuation);
    client->retirement.state.store(kCapacityArmed,
                                   std::memory_order_release);
    ClientControl &control = client->mailbox->client(client->slot).control();
    uint32_t active = kClientActive;
    if (!compare_exchange(&control.state, &active, kClientRetiring)) {
        if (active == kClientFree ||
            client->host_dead.load(std::memory_order_acquire) != 0) {
            resume_edge(client->retirement);
            return InProgress;
        }
        return discard_edge(client->retirement) ? Stale : InProgress;
    }
    const int edge = send_edge(client->service_port, kMessageHostEdge,
                               client->host_generation,
                               client->client_generation);
    if (edge != Ok) fault_client(client);
    if (load(&control.state) == kClientFree ||
        control.client_generation != client->client_generation) {
        resume_edge(client->retirement);
    }
    return InProgress;
#else
    (void)client;
    (void)continuation;
    return Unsupported;
#endif
}

Status client_begin_close(Client *client,
                          koro_cont_t *continuation) {
#ifndef __APPLE__
    (void)client;
    (void)continuation;
    return Unsupported;
#else
    if (!client || !continuation) return InvalidArgument;
    if (client->host_dead.load(std::memory_order_acquire) == 0 &&
        client->mailbox && client->slot < kClientCapacity &&
        load(&client->mailbox->client(client->slot).control().state) !=
            kClientFree) {
        return Busy;
    }
    uint32_t idle = 0;
    if (!client->close_started.compare_exchange_strong(
            idle, 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    uint32_t free = kCapacityFree;
    if (!client->close.state.compare_exchange_strong(
            free, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        client->close_started.store(0, std::memory_order_release);
        return Busy;
    }
    client->close.identity = koro_cont_identity(continuation);
    client->close.continuation = continuation;
    koro_cont_retain(continuation);
    client->close.state.store(kCapacityArmed, std::memory_order_release);
    uint32_t pending = 0;
    if (client->receive_source) ++pending;
    if (client->death_source) ++pending;
    client->cancel_pending.store(pending, std::memory_order_release);
    if (client->receive_source) dispatch_source_cancel(client->receive_source);
    if (client->death_source) dispatch_source_cancel(client->death_source);
    if (pending == 0) {
        client->closed.store(1, std::memory_order_release);
        resume_edge(client->close);
    }
    return InProgress;
#endif
}

Status client_destroy(Client *client) {
#ifndef __APPLE__
    (void)client;
    return Unsupported;
#else
    if (!client || client->closed.load(std::memory_order_acquire) == 0 ||
        client->cancel_pending.load(std::memory_order_acquire) != 0) {
        return InvalidArgument;
    }
    if (client->receive_source) dispatch_release(client->receive_source);
    if (client->death_source) dispatch_release(client->death_source);
    if (client->queue) dispatch_release(client->queue);
    if (client->service_port != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), client->service_port);
    }
    if (client->liveness_port != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), client->liveness_port);
    }
    if (client->wake_port != MACH_PORT_NULL) {
        (void)mach_port_mod_refs(mach_task_self(), client->wake_port,
                                 MACH_PORT_RIGHT_RECEIVE, -1);
    }
    if (client->mailbox) {
        (void)munmap(client->mailbox, mailbox_mapping_bytes());
    }
    if (client->fd >= 0) (void)::close(client->fd);
    release_records(client->outstanding, kRingCapacity);
    delete client;
    return Ok;
#endif
}

namespace test {

Status inject_stale_completion(Client *client,
                               koro_cont_t *continuation) {
#ifndef __APPLE__
    (void)client;
    (void)continuation;
    return Unsupported;
#else
    if (!client || !continuation || !client->mailbox ||
        client->slot >= kClientCapacity) {
        return InvalidArgument;
    }
    uint32_t free = kCapacityFree;
    if (!client->stale_probe.state.compare_exchange_strong(
            free, kCapacityInstalling, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return Busy;
    }
    client->stale_probe.identity = koro_cont_identity(continuation);
    client->stale_probe.continuation = continuation;
    koro_cont_retain(continuation);
    client->stale_probe.state.store(kCapacityArmed,
                                    std::memory_order_release);
    const kc_ticket_id identity = koro_cont_identity(continuation);
    HostCompletion completion{
        .size = sizeof(completion),
        .layout_version = kLayoutVersion,
        .ticket = identity,
        .parent = identity,
        .host_generation = client->host_generation + 1,
        .client_generation = client->client_generation,
        .operation = QueryStatus,
        .status = Ok,
    };
    completion.checkpoint_identity =
        client->mailbox->header.checkpoint_identity;
    ClientSlotView slot = client->mailbox->client(client->slot);
    if (!completion_push(slot, completion)) {
        resume_edge(client->stale_probe);
        return Capacity;
    }
    const kern_return_t inserted = mach_port_insert_right(
        mach_task_self(), client->wake_port, client->wake_port,
        MACH_MSG_TYPE_MAKE_SEND);
    if (inserted != KERN_SUCCESS) {
        resume_edge(client->stale_probe);
        return IoError;
    }
    const int edge = send_edge(client->wake_port, kMessageClientEdge,
                               client->host_generation,
                               client->client_generation);
    (void)mach_port_deallocate(mach_task_self(), client->wake_port);
    if (edge == HostDown) fault_client(client);
    return edge == Ok ? InProgress : static_cast<Status>(edge);
#endif
}

} // namespace test

Status serve(const ServerConfig &config, std::string *error) {
#ifndef __APPLE__
    (void)config;
    if (error) {
        *error = "the production host mailbox requires macOS Mach services";
    }
    return Unsupported;
#else
    Server *server = nullptr;
    Status status = server_create(config, &server, error);
    if (status != Ok) return status;
    server->exit_on_retire = 1;
    const dispatch_queue_t queue = dispatch_get_global_queue(
        QOS_CLASS_USER_INITIATED, 0);
    ::signal(SIGTERM, SIG_IGN);
    ::signal(SIGINT, SIG_IGN);
    server->terminate_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0, queue);
    server->interrupt_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_SIGNAL, SIGINT, 0, queue);
    if (!server->terminate_source || !server->interrupt_source) {
        if (error) *error = "cannot create native host signal edges";
        return IoError;
    }
    dispatch_set_context(server->terminate_source, server);
    dispatch_set_context(server->interrupt_source, server);
    dispatch_source_set_event_handler_f(server->terminate_source,
                                        server_signal);
    dispatch_source_set_event_handler_f(server->interrupt_source,
                                        server_signal);
    dispatch_activate(server->terminate_source);
    dispatch_activate(server->interrupt_source);
    status = server_start(server);
    if (status != Ok) return status;
    dispatch_main();
    return Ok;
#endif
}

} // namespace lfm::host
