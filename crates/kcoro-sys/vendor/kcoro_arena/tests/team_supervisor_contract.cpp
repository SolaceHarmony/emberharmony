// SPDX-License-Identifier: BSD-3-Clause

#include "kc_team_supervisor.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

extern "C" int kc_team_inject_member_exit_for_test(
    kc_team_t *, std::uint64_t, std::uint32_t, std::uint32_t,
    void (*)(void *, std::uint64_t), void *);

namespace {

[[noreturn]] void fail(const char *message) {
    std::fprintf(
        stderr, "kcoro team-supervisor contract failed: %s\n",
        message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

struct Request {
    kc_ticket_id workflow{};
    kc_ticket_id pass{};
    std::uint64_t conversation = 0;
};

struct Work {
    std::uint32_t operation = 0;
    std::uint32_t stage = 0;
    std::uint64_t shape = 0;
};

struct alignas(128) Fatal {
    Request request{};
    Work work{};
    kc_deadline_event event{};
    kc_team_quorum_snapshot quorum{};
    std::uint64_t armed_ns = 0;
    std::uint64_t budget_ns = 0;
    std::uint64_t elapsed_ns = 0;
    std::int32_t quorum_status = 0;
    std::uint32_t published = 0;
};

using Supervisor = kc::TeamSupervisor<Request, Work, Fatal>;
using Store = kc::FatalStore<Fatal>;

struct Probe {
    Supervisor *supervisor = nullptr;
    std::mutex mutex;
    std::condition_variable changed;
    std::uint64_t completed = 0;
    std::uint32_t fatal_publications = 0;
};

void member(void *, std::uint32_t, std::uint32_t,
            std::uint64_t) noexcept {}

void fill_fatal(
    void *, Fatal &fatal, const Request &request, const Work &work,
    const kc_deadline_event &event, std::uint64_t armed_ns,
    std::uint64_t budget_ns, std::uint64_t elapsed_ns,
    int quorum_status,
    const kc_team_quorum_snapshot &quorum) noexcept {
    fatal = {
        .request = request,
        .work = work,
        .event = event,
        .quorum = quorum,
        .armed_ns = armed_ns,
        .budget_ns = budget_ns,
        .elapsed_ns = elapsed_ns,
        .quorum_status = quorum_status,
        .published = 1,
    };
}

void fatal_published(void *context, const Fatal &) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    probe->fatal_publications++;
}

void returned(void *context, std::uint64_t generation) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    kc::TeamCompletion completion{};
    if (!probe || !probe->supervisor ||
        probe->supervisor->complete(
            generation, &completion) != 0 ||
        completion != kc::TeamCompletion::Continue) {
        std::abort();
    }
    {
        std::lock_guard lock(probe->mutex);
        probe->completed = generation;
    }
    probe->changed.notify_all();
}

void wait_completion(Probe &probe, std::uint64_t generation) {
    /* Administrative test observation only. Numerical progress is driven by
     * the team's final-return callback above; no product worker waits here. */
    std::unique_lock lock(probe.mutex);
    check(probe.changed.wait_for(
              lock, std::chrono::seconds(5), [&] {
                  return probe.completed == generation;
              }),
          "normal final-return callback did not settle supervision");
}

Request request(kc::TicketSource &tickets) {
    return {
        .workflow = tickets.mint(KC_TICKET_KIND_WORKFLOW),
        .pass = tickets.mint(KC_TICKET_KIND_PASS),
        .conversation = 41,
    };
}

kc::TeamSupervisionIdentity identity(
    const Request &value, kc::TicketSource &tickets) {
    return {
        .parent = value.pass,
        .scope_generation = 19,
        .epoch = 23,
        .domain = tickets.epoch(),
    };
}

void terminal_contract() {
    kc::TeamTerminal first;
    check(first.begin(7) == 0, "terminal generation did not begin");
    check(first.claim(7, kc::TeamTerminalState::Completed),
          "normal completion did not claim terminal state");
    check(!first.claim(7, kc::TeamTerminalState::TimedOut),
          "timeout published after completion");
    check(kc::TeamTerminal::generation_of(first.word()) == 7 &&
              kc::TeamTerminal::state_of(first.word()) ==
                  kc::TeamTerminalState::Completed,
          "completed terminal word lost its generation");
    check(first.begin(8) == 0,
          "successor generation did not replace completion");

    kc::TeamCompletion completion{};
    check(first.settle(
              8, KC_DEADLINE_RETIRE_EXPIRY_WON,
              &completion) == 0 &&
              completion == kc::TeamCompletion::ExpiryWon,
          "expiry winner was not preserved");
    check(first.claim(8, kc::TeamTerminalState::TimedOut),
          "expiry path could not claim timeout");
    check(!first.claim(8, kc::TeamTerminalState::Completed),
          "completion published after timeout");

    kc::TeamTerminal retired;
    check(retired.begin(11) == 0,
          "retired generation did not begin");
    check(retired.settle(
              11, KC_DEADLINE_RETIRE_RETIRED,
              &completion) == 0 &&
              completion == kc::TeamCompletion::Continue,
          "authoritative deadline retirement did not permit completion");
}

void store_contract() {
#if !defined(_WIN32)
    std::array<char, 256> path{};
    const int length = std::snprintf(
        path.data(), path.size(),
        "/tmp/kcoro-fatal-store-%ld.bin",
        static_cast<long>(::getpid()));
    check(length > 0 &&
              static_cast<std::size_t>(length) < path.size(),
          "fatal store path overflow");
    (void)::unlink(path.data());

    Store store;
    check(store.initialize({
              .magic = UINT64_C(0x1122334455667788),
              .runtime_epoch = 91,
              .path = path.data(),
          }) == 0,
          "fatal store initialization failed");
    Fatal fatal = {
        .armed_ns = 101,
        .budget_ns = 202,
        .elapsed_ns = 303,
        .published = 1,
    };
    store.publish(fatal);

    std::array<unsigned char, Store::bytes> bytes{};
    const int descriptor = ::open(path.data(), O_RDONLY);
    check(descriptor >= 0, "published fatal store did not exist");
    const ssize_t count =
        ::pread(descriptor, bytes.data(), bytes.size(), 0);
    (void)::close(descriptor);
    check(count == static_cast<ssize_t>(bytes.size()),
          "fatal store record was truncated");

    Store::Header header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    check(header.magic == UINT64_C(0x1122334455667788) &&
              header.format_version == Store::format &&
              header.header_size == sizeof(Store::Header) &&
              header.record_size == sizeof(Fatal) &&
              header.publication == Store::committed &&
              header.runtime_epoch == 91,
          "fatal store header drifted");
    Fatal found{};
    std::memcpy(
        &found, bytes.data() + sizeof(Store::Header),
        sizeof(found));
    check(std::memcmp(&found, &fatal, sizeof(found)) == 0 &&
              header.checksum == Store::checksum(found),
          "fatal store payload or checksum drifted");
    store.destroy();
    check(::access(path.data(), F_OK) != 0,
          "normal fatal-store destruction did not retire its file");
#endif
}

void normal_supervision_contract() {
    kc::TicketSource tickets;
    Probe probe;
    const kc_runtime_config runtime_config = {
        .worker_count = 2,
    };
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&runtime_config, &runtime) == 0,
          "runtime creation failed");
    const kc_team_config team_config = {
        .member_count = 2,
        .member = member,
        .context = nullptr,
        .runtime = runtime,
        .retired = nullptr,
        .retired_context = nullptr,
    };
    kc_team_t *team = nullptr;
    check(kc_team_create(&team_config, &team) == 0,
          "team creation failed");

    Supervisor supervisor;
    probe.supervisor = &supervisor;
    check(supervisor.initialize({
              .runtime = runtime,
              .team = team,
              .tickets = &tickets,
              .ticket_kind = KC_TICKET_KIND_DEADLINE,
              .deadline_slot = 0,
              .expected_mask = 0b11,
              .fatal_magic = UINT64_C(0x2233445566778899),
              .fatal_path = nullptr,
              .manual_deadlines = true,
              .context = &probe,
              .operations = {
                  .fill_fatal = fill_fatal,
                  .fatal_published = fatal_published,
              },
          }) == 0,
          "supervisor initialization failed");
    check(kc_runtime_start(runtime) == 0,
          "runtime start failed");
    check(kc_team_start(team) == 0, "team start failed");
    check(supervisor.start() == 0, "supervisor start failed");

    const Request work_request = request(tickets);
    const Work work = {
        .operation = 5,
        .stage = 7,
        .shape = 4096,
    };
    check(supervisor.begin(
              work_request, work,
              identity(work_request, tickets), 1, 0) == -ENODATA,
          "an unbudgeted generation entered the team");
    check(supervisor.snapshot().terminal == 0,
          "unbudgeted work changed terminal state");
    check(supervisor.begin(
              work_request, work,
              identity(work_request, tickets), 1,
              UINT64_C(1000000000)) == 0,
          "budgeted generation did not arm");
    const kc::TeamSupervisorSnapshot armed =
        supervisor.snapshot();
    check(armed.generation == 1 &&
              armed.arm_generation != 0 &&
              armed.arm_team_generation == 1 &&
              armed.deadline_slot == 0,
          "armed deadline snapshot lost its generation");
    check(kc_team_dispatch_notify(
              team, 1, returned, &probe) == 0,
          "team generation dispatch failed");
    wait_completion(probe, 1);
    const std::uint64_t terminal = supervisor.snapshot().terminal;
    check(kc::TeamTerminal::generation_of(terminal) == 1 &&
              kc::TeamTerminal::state_of(terminal) ==
                  kc::TeamTerminalState::Completed,
          "normal quorum return did not win terminal state");
    check(probe.fatal_publications == 0,
          "normal completion published fatal evidence");

    supervisor.request_stop();
    kc_team_request_stop(team);
    check(kc_runtime_join_all(runtime) == 0,
          "retained supervisor/team continuations did not retire");
    check(supervisor.join() == 0,
          "supervisor administrative join failed");
    check(kc_team_destroy(team) == 0,
          "team destruction failed");
    check(supervisor.destroy() == 0,
          "supervisor setup leases did not destroy");
    kc_runtime_request_stop(runtime);
    check(kc_runtime_join(runtime) == 0,
          "runtime join failed");
    check(kc_runtime_destroy(runtime) == 0,
          "runtime destruction failed");
}

#if !defined(_WIN32)
struct Fault {
    Supervisor *supervisor = nullptr;
};

void fault_ready(void *context, std::uint64_t generation) {
    Fault *fault = static_cast<Fault *>(context);
    if (!fault || !fault->supervisor ||
        fault->supervisor->expire_manual_test(generation) != 0) {
        std::abort();
    }
}

[[noreturn]] void timeout_child(
    std::uint32_t point, const char *path) {
    (void)::alarm(5);
    kc::TicketSource tickets;
    Probe probe;
    const kc_runtime_config runtime_config = {
        .worker_count = 4,
    };
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&runtime_config, &runtime) == 0,
          "fault runtime creation failed");
    const kc_team_config team_config = {
        .member_count = 4,
        .member = member,
        .context = nullptr,
        .runtime = runtime,
        .retired = nullptr,
        .retired_context = nullptr,
    };
    kc_team_t *team = nullptr;
    check(kc_team_create(&team_config, &team) == 0,
          "fault team creation failed");
    Supervisor supervisor;
    probe.supervisor = &supervisor;
    check(supervisor.initialize({
              .runtime = runtime,
              .team = team,
              .tickets = &tickets,
              .ticket_kind = KC_TICKET_KIND_DEADLINE,
              .deadline_slot = 0,
              .expected_mask = 0b1111,
              .fatal_magic = UINT64_C(0x33445566778899aa),
              .fatal_path = path,
              .manual_deadlines = true,
              .context = &probe,
              .operations = {
                  .fill_fatal = fill_fatal,
                  .fatal_published = fatal_published,
              },
          }) == 0,
          "fault supervisor initialization failed");
    Fault fault = {
        .supervisor = &supervisor,
    };
    check(kc_team_inject_member_exit_for_test(
              team, 1, 1, point, fault_ready, &fault) == 0,
          "team fault injection failed");
    check(kc_runtime_start(runtime) == 0,
          "fault runtime start failed");
    check(kc_team_start(team) == 0,
          "fault team start failed");
    check(supervisor.start() == 0,
          "fault supervisor start failed");
    const Request work_request = request(tickets);
    check(supervisor.begin(
              work_request,
              {.operation = 9, .stage = point, .shape = 8192},
              identity(work_request, tickets), 1,
              UINT64_C(1000000000)) == 0,
          "fault generation did not arm");
    check(kc_team_dispatch_notify(team, 1, returned, &probe) == 0,
          "fault generation dispatch failed");
    for (;;) (void)::pause();
}

Fatal read_fatal(const char *path) {
    std::array<unsigned char, Store::bytes> bytes{};
    const int descriptor = ::open(path, O_RDONLY);
    check(descriptor >= 0, "fatal child left no durable record");
    const ssize_t count =
        ::pread(descriptor, bytes.data(), bytes.size(), 0);
    (void)::close(descriptor);
    check(count == static_cast<ssize_t>(bytes.size()),
          "fatal child record was truncated");
    Store::Header header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    check(header.magic == UINT64_C(0x33445566778899aa) &&
              header.publication == Store::committed,
          "fatal child header was not committed");
    Fatal fatal{};
    std::memcpy(
        &fatal, bytes.data() + sizeof(Store::Header),
        sizeof(fatal));
    check(header.checksum == Store::checksum(fatal),
          "fatal child checksum did not match");
    return fatal;
}

void fatal_supervision_contract() {
    for (const std::uint32_t point : {1u, 2u}) {
        std::array<char, 256> path{};
        const int length = std::snprintf(
            path.data(), path.size(),
            "/tmp/kcoro-team-supervisor-%ld-%u.bin",
            static_cast<long>(::getpid()), point);
        check(length > 0 &&
                  static_cast<std::size_t>(length) < path.size(),
              "fault path overflow");
        (void)::unlink(path.data());
        const pid_t child = ::fork();
        check(child >= 0, "fault child fork failed");
        if (child == 0) timeout_child(point, path.data());

        int status = 0;
        check(::waitpid(child, &status, 0) == child,
              "fault child reap failed");
        check(WIFSIGNALED(status) &&
                  WTERMSIG(status) == SIGABRT,
              "hard timeout did not abort exactly once");
        const Fatal fatal = read_fatal(path.data());
        (void)::unlink(path.data());
        check(fatal.published == 1 &&
                  fatal.request.conversation == 41 &&
                  fatal.work.operation == 9 &&
                  fatal.work.stage == point &&
                  fatal.event.kind == KC_DEADLINE_EVENT_EXPIRED &&
                  fatal.event.team_generation == 1 &&
                  fatal.quorum.expected_mask == 0b1111 &&
                  fatal.budget_ns == UINT64_C(1000000000) &&
                  fatal.elapsed_ns >= fatal.budget_ns,
              "fatal record lost lineage, work, or timing evidence");
        const std::uint64_t missing =
            fatal.quorum.expected_mask &
            ~fatal.quorum.entered_mask;
        const std::uint64_t hung =
            fatal.quorum.entered_mask &
            ~fatal.quorum.returned_mask;
        check((point == 1 && missing == 0b0010 && hung == 0) ||
                  (point == 2 && missing == 0 && hung == 0b0010),
              "fatal quorum masks did not classify the injected lane");
    }
}
#endif

} // namespace

int main() {
    terminal_contract();
    store_contract();
    normal_supervision_contract();
#if !defined(_WIN32)
    fatal_supervision_contract();
#endif
    return 0;
}
