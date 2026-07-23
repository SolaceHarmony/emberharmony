#ifndef LFM_HOST_MAILBOX_H
#define LFM_HOST_MAILBOX_H

/* Private native host mailbox. This is a C++/kcoro ownership seam. Git carries
 * its history; the running tree has one shared-memory layout. Model pointers
 * and payloads never cross it; Mach messages carry wake and port registration
 * edges only. */

#include "kc_identity.h"
#include "kc_runtime.h"
#include "kcoro_stackless.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace lfm::host {

constexpr uint32_t kClientCapacity = 8;
constexpr uint32_t kRingCapacity = 8;
constexpr uint32_t kTestService = 1u << 0;
constexpr uint32_t kPrivilegedClient = 1u << 0;

enum Status : int32_t {
    Ok = 0,
    InvalidArgument = -1,
    IoError = -2,
    Rejected = -3,
    InProgress = -4,
    Capacity = -5,
    Stale = -6,
    HostDown = -7,
    Busy = -8,
    Denied = -9,
    Unsupported = -10,
    Cancelled = -11,
};

enum Operation : uint32_t {
    InvalidOperation = 0,
    Attach = 1,
    Release = 2,
    QueryStatus = 3,
    Evict = 4,
};

struct CheckpointIdentity {
    uint64_t word0{0};
    uint64_t word1{0};
    uint64_t word2{0};
    uint64_t word3{0};
};
static_assert(sizeof(CheckpointIdentity) == 32);

/* One sequence word plus one request fills one 128-byte Apple cache line. */
struct HostRequest {
    kc_ticket_id ticket;
    kc_ticket_id parent;
    uint64_t host_generation;
    uint64_t client_generation;
    CheckpointIdentity checkpoint_identity;
    uint64_t lease_generation;
    uint32_t operation;
    uint32_t flags;
};
static_assert(sizeof(HostRequest) == 112);
static_assert(std::is_trivially_copyable_v<HostRequest>);

/* QueryStatus packs active clients into result_flags[15:0] and active leases
 * into result_flags[31:16]. Every completion retains request lineage. */
struct HostCompletion {
    kc_ticket_id ticket;
    kc_ticket_id parent;
    uint64_t host_generation;
    uint64_t client_generation;
    CheckpointIdentity checkpoint_identity;
    uint64_t lease_generation;
    uint32_t operation;
    int32_t status;
    uint32_t result_flags;
};
static_assert(sizeof(HostCompletion) == 120);
static_assert(std::is_trivially_copyable_v<HostCompletion>);

struct HostSnapshot {
    uint32_t state;
    uint32_t flags;
    uint64_t host_generation;
    uint64_t host_pid;
    uint64_t segment_generation;
    uint64_t active_clients;
    uint64_t active_leases;
    uint64_t client_events;
    CheckpointIdentity checkpoint_identity;
    CheckpointIdentity content_digest;
};

struct Server;
struct Client;

struct ServerConfig {
    std::string checkpoint;
    std::string service;
    uint32_t coordination_workers{2};
    uint32_t flags{0};
    uint64_t privileged_pid{0};
    int readiness_fd{-1};
};

struct ClientConfig {
    std::string service;
    kc_runtime_t *runtime{nullptr};
    uint32_t flags{0};
};

Status server_create(const ServerConfig &config, Server **out,
                     std::string *error);
Status server_start(Server *server);
void server_request_stop(Server *server);
Status server_snapshot(const Server *server, HostSnapshot *out);

/* Creation claims no worker. The HELLO/registration Mach callbacks resume the
 * exact readiness continuation after the host's native layout is validated. */
Status client_create(const ClientConfig &config, koro_cont_t *readiness,
                     Client **out, std::string *error);
Status client_ready(const Client *client);
Status client_submit(Client *client, Operation operation,
                     const kc_ticket_id &parent, uint64_t lease_generation,
                     koro_cont_t *continuation);
Status client_take(Client *client, const kc_ticket_id &ticket,
                   HostCompletion *out);
Status client_snapshot(const Client *client, HostSnapshot *out);
/* Arms one exact continuation against the aggregate client-event generation.
 * A changed generation or host death publishes the callback edge. */
Status client_watch(Client *client, uint64_t observed,
                    koro_cont_t *continuation);
/* Stop and close are callback-shaped. The first continuation resumes when the
 * host has retired the shared slot; the second resumes only after every GCD
 * source has acknowledged cancellation, after which destroy cannot race a
 * queued callback. */
Status client_request_stop(Client *client, koro_cont_t *continuation);
Status client_begin_close(Client *client, koro_cont_t *continuation);
Status client_destroy(Client *client);

Status serve(const ServerConfig &config, std::string *error);

} // namespace lfm::host

#endif
