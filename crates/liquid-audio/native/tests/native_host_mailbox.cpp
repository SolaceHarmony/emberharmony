// Native hostile-lifecycle gate for the shared weight host mailbox.
//
// The test itself follows the production rule: logical work dehydrates in a
// kcoro continuation and only a correlated Mach/GCD callback makes it
// runnable. Pipes carry child-readiness evidence only; operation records stay
// in shared memory. The one-shot timer can fail the executable, never advance
// its state machine.

#include "lfm_host_mailbox.h"
#include "lfm_host_mailbox_internal.h"
#include "lfm_safetensors.h"

#include "kc_runtime.h"
#include "kcoro_stackless.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <signal.h>
#include <spawn.h>
#include <unistd.h>
extern char **environ;
#endif

namespace {

using lfm::host::Attach;
using lfm::host::Busy;
using lfm::host::Capacity;
using lfm::host::Client;
using lfm::host::ClientConfig;
using lfm::host::Evict;
using lfm::host::HostCompletion;
using lfm::host::HostDown;
using lfm::host::HostSnapshot;
using lfm::host::InProgress;
using lfm::host::Ok;
using lfm::host::QueryStatus;
using lfm::host::Release;
using lfm::host::ServerConfig;
using lfm::host::Stale;

constexpr uint32_t kTaskCount = lfm::host::kRingCapacity + 1;
constexpr uint64_t kMarker = UINT64_C(0x6b636f726f686f73);
constexpr uint64_t kWatchdogSeconds = 60;

struct Gate;

#ifdef __APPLE__

struct Process {
    Gate *gate{nullptr};
    pid_t pid{0};
    int evidence_fd{-1};
    dispatch_source_t evidence_source{nullptr};
    dispatch_source_t process_source{nullptr};
    std::atomic<uint32_t> ready{0};
    std::atomic<uint32_t> attached{0};
    std::atomic<uint32_t> stop_edge{0};
    std::atomic<uint32_t> exited{0};
};

struct RequestTask {
    Gate *gate{nullptr};
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
    std::atomic<uint32_t> submitted{0};
    std::atomic<uint32_t> completed{0};
    uint32_t saw_capacity{0};
};

struct GateFrame {
    uint64_t marker{0};
    uint32_t first_worker{UINT32_MAX};
    uint32_t last_worker{UINT32_MAX};
    uint32_t resumes{0};
};

struct Gate {
    const char *program{nullptr};
    std::string checkpoint;
    std::string service;
    dispatch_queue_t queue{nullptr};
    dispatch_source_t watchdog{nullptr};
    kc_runtime_t *runtime{nullptr};
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
    Client *client{nullptr};
    Client *restart_client{nullptr};
    Process first_host{};
    Process second_host{};
    Process child{};
    RequestTask *tasks{nullptr};
    std::atomic<uint32_t> task_ready{0};
    std::atomic<uint32_t> task_done{0};
    std::atomic<uint32_t> stage{0};
    std::atomic<uint32_t> terminal{0};
    uint32_t capacity_edges{0};
    uint64_t host_generation{0};
    uint64_t lease_generation{0};
    uint64_t child_event_generation{0};
    lfm::host::CheckpointIdentity checkpoint_identity{};
};

template <typename T>
T *allocate_records(size_t count) {
    void *storage = ::operator new(sizeof(T) * count, std::nothrow);
    if (!storage) return nullptr;
    auto *records = static_cast<T *>(storage);
    for (size_t index = 0; index < count; ++index) {
        std::construct_at(records + index);
    }
    return records;
}

[[noreturn]] void finish(Gate *gate, const char *message, int code) {
    if (gate && gate->terminal.exchange(1, std::memory_order_acq_rel) != 0) {
        std::_Exit(code);
    }
    if (message) std::fprintf(stderr, "%s\n", message);
    if (gate) {
        if (gate->first_host.pid > 0 &&
            gate->first_host.exited.load(std::memory_order_acquire) == 0) {
            (void)::kill(gate->first_host.pid, SIGKILL);
        }
        if (gate->second_host.pid > 0 &&
            gate->second_host.exited.load(std::memory_order_acquire) == 0) {
            (void)::kill(gate->second_host.pid, SIGKILL);
        }
        if (gate->child.pid > 0 &&
            gate->child.exited.load(std::memory_order_acquire) == 0) {
            (void)::kill(gate->child.pid, SIGKILL);
        }
    }
    std::_Exit(code);
}

void resume_gate(Gate *gate) {
    if (gate && gate->continuation) {
        (void)koro_cont_resume(gate->continuation, &gate->identity);
    }
}

void evidence_ready(void *context) {
    auto *process = static_cast<Process *>(context);
    char evidence[128]{};
    const ssize_t count = ::read(process->evidence_fd, evidence,
                                 sizeof(evidence) - 1);
    if (count > 0) {
        evidence[count] = '\0';
        if (std::strstr(evidence, "READY") != nullptr) {
            process->ready.store(1, std::memory_order_release);
        }
        if (std::strstr(evidence, "ATTACHED") != nullptr) {
            process->attached.store(1, std::memory_order_release);
        }
        dispatch_source_cancel(process->evidence_source);
        (void)::close(process->evidence_fd);
        process->evidence_fd = -1;
        resume_gate(process->gate);
        return;
    }
    if (count == 0) {
        finish(process->gate,
               "child retired before publishing its native readiness edge",
               EXIT_FAILURE);
    }
    if (errno != EINTR && errno != EAGAIN) {
        finish(process->gate, "native readiness edge read failed",
               EXIT_FAILURE);
    }
}

void process_event(void *context) {
    auto *process = static_cast<Process *>(context);
    const unsigned long data =
        dispatch_source_get_data(process->process_source);
    if ((data & DISPATCH_PROC_SIGNAL) != 0) {
        process->stop_edge.store(1, std::memory_order_release);
    }
    if ((data & DISPATCH_PROC_EXIT) != 0) {
        process->exited.store(1, std::memory_order_release);
    }
    resume_gate(process->gate);
}

int publish_process_sources(Process *process, int fd) {
    process->evidence_fd = fd;
    process->evidence_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_READ, static_cast<uintptr_t>(fd), 0,
        process->gate->queue);
    process->process_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_PROC, static_cast<uintptr_t>(process->pid),
        DISPATCH_PROC_EXIT | DISPATCH_PROC_SIGNAL, process->gate->queue);
    if (!process->evidence_source || !process->process_source) return ENOMEM;
    dispatch_set_context(process->evidence_source, process);
    dispatch_source_set_event_handler_f(process->evidence_source,
                                        evidence_ready);
    dispatch_set_context(process->process_source, process);
    dispatch_source_set_event_handler_f(process->process_source,
                                        process_event);
    dispatch_activate(process->evidence_source);
    dispatch_activate(process->process_source);
    return 0;
}

int spawn_host(Gate *gate, Process *process) {
    int pipefd[2]{};
    if (pipe(pipefd) != 0) return errno;
    posix_spawn_file_actions_t actions{};
    int status = posix_spawn_file_actions_init(&actions);
    if (status == 0) {
        status = posix_spawn_file_actions_adddup2(&actions, pipefd[1], 3);
    }
    if (status == 0 && pipefd[0] != 3) {
        status = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    }
    if (status == 0 && pipefd[1] != 3) {
        status = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    }
    char owner[32]{};
    std::snprintf(owner, sizeof(owner), "%llu",
                  static_cast<unsigned long long>(getpid()));
    char *const arguments[] = {
        const_cast<char *>(gate->program), const_cast<char *>("host"),
        const_cast<char *>(gate->checkpoint.c_str()),
        const_cast<char *>(gate->service.c_str()), owner,
        const_cast<char *>("3"), nullptr,
    };
    if (status == 0) {
        status = posix_spawn(&process->pid, gate->program, &actions, nullptr,
                             arguments, environ);
    }
    (void)posix_spawn_file_actions_destroy(&actions);
    (void)::close(pipefd[1]);
    if (status != 0) {
        (void)::close(pipefd[0]);
        return status;
    }
    process->gate = gate;
    return publish_process_sources(process, pipefd[0]);
}

int spawn_client_child(Gate *gate, Process *process) {
    int pipefd[2]{};
    if (pipe(pipefd) != 0) return errno;
    posix_spawn_file_actions_t actions{};
    int status = posix_spawn_file_actions_init(&actions);
    if (status == 0) {
        status = posix_spawn_file_actions_adddup2(&actions, pipefd[1], 3);
    }
    if (status == 0 && pipefd[0] != 3) {
        status = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    }
    if (status == 0 && pipefd[1] != 3) {
        status = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    }
    char *const arguments[] = {
        const_cast<char *>(gate->program), const_cast<char *>("client"),
        const_cast<char *>(gate->service.c_str()),
        const_cast<char *>("3"), nullptr,
    };
    if (status == 0) {
        status = posix_spawn(&process->pid, gate->program, &actions, nullptr,
                             arguments, environ);
    }
    (void)posix_spawn_file_actions_destroy(&actions);
    (void)::close(pipefd[1]);
    if (status != 0) {
        (void)::close(pipefd[0]);
        return status;
    }
    process->gate = gate;
    return publish_process_sources(process, pipefd[0]);
}

void watchdog(void *context) {
    auto *gate = static_cast<Gate *>(context);
    char detail[384]{};
    std::snprintf(
        detail, sizeof(detail),
        "native host mailbox gate watchdog expired "
        "(stage=%u task_ready=%u task_done=%u first_ready=%u "
        "first_stop=%u first_exit=%u second_ready=%u second_exit=%u "
        "child_attached=%u child_exit=%u)",
        gate->stage.load(std::memory_order_acquire),
        gate->task_ready.load(std::memory_order_acquire),
        gate->task_done.load(std::memory_order_acquire),
        gate->first_host.ready.load(std::memory_order_acquire),
        gate->first_host.stop_edge.load(std::memory_order_acquire),
        gate->first_host.exited.load(std::memory_order_acquire),
        gate->second_host.ready.load(std::memory_order_acquire),
        gate->second_host.exited.load(std::memory_order_acquire),
        gate->child.attached.load(std::memory_order_acquire),
        gate->child.exited.load(std::memory_order_acquire));
    finish(gate, detail, EXIT_FAILURE);
}

void publish_task(RequestTask *task) {
    if (task->gate->task_done.fetch_add(1, std::memory_order_acq_rel) + 1 ==
        kTaskCount) {
        resume_gate(task->gate);
    }
}

void *request_step(koro_cont_t *continuation) {
    auto *task = static_cast<RequestTask *>(
        koro_cont_argument(continuation));
    const uint32_t state = koro_cont_state_get(continuation);
    if (state == 0) {
        koro_cont_state_set(continuation, 1, KORO_SUSPEND_CALLBACK);
        if (task->gate->task_ready.fetch_add(
                1, std::memory_order_acq_rel) + 1 == kTaskCount) {
            resume_gate(task->gate);
        }
        return nullptr;
    }
    if (state != 1) {
        finish(task->gate, "request continuation resumed at an invalid PC",
               EXIT_FAILURE);
    }
    if (task->submitted.load(std::memory_order_acquire) == 0) {
        const lfm::host::Status status = lfm::host::client_submit(
            task->gate->client, QueryStatus, task->gate->identity, 0,
            continuation);
        if (status == Capacity) {
            task->saw_capacity = 1;
            return nullptr;
        }
        if (status != InProgress) {
            finish(task->gate, "capacity continuation could not resubmit",
                   EXIT_FAILURE);
        }
        task->submitted.store(1, std::memory_order_release);
        return nullptr;
    }
    HostCompletion completion{};
    const lfm::host::Status taken = lfm::host::client_take(
        task->gate->client, task->identity, &completion);
    if (taken == InProgress) return nullptr;
    if (taken != Ok || completion.status != Ok ||
        completion.operation != QueryStatus) {
        finish(task->gate, "capacity request completion was not correlated",
               EXIT_FAILURE);
    }
    task->completed.store(1, std::memory_order_release);
    if (!koro_cont_finish(continuation)) return nullptr;
    publish_task(task);
    return reinterpret_cast<void *>(1);
}

bool take(Gate *gate, lfm::host::Status expected,
          uint32_t operation, HostCompletion *out = nullptr) {
    HostCompletion completion{};
    const lfm::host::Status status = lfm::host::client_take(
        gate->client, gate->identity, &completion);
    if (status != Ok || completion.status != expected ||
        completion.operation != operation) {
        return false;
    }
    if (out) *out = completion;
    return true;
}

void *suspend(koro_cont_t *continuation, uint32_t state) {
    koro_cont_state_set(continuation, state, KORO_SUSPEND_CALLBACK);
    return nullptr;
}

void *gate_step(koro_cont_t *continuation) {
    auto *gate = static_cast<Gate *>(koro_cont_argument(continuation));
    auto *frame = static_cast<GateFrame *>(koro_cont_frame(continuation));
    frame->last_worker = koro_cont_current_worker(continuation);
    ++frame->resumes;
    const uint32_t state = koro_cont_state_get(continuation);
    gate->stage.store(state, std::memory_order_release);
    switch (state) {
    case 0: {
        frame->marker = kMarker;
        frame->first_worker = frame->last_worker;
        if (spawn_host(gate, &gate->first_host) != 0) {
            finish(gate, "cannot spawn first native mailbox host",
                   EXIT_FAILURE);
        }
        return suspend(continuation, 1);
    }
    case 1: {
        if (gate->first_host.ready.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 1);
        }
        ClientConfig config{
            .service = gate->service,
            .runtime = gate->runtime,
            .flags = lfm::host::kPrivilegedClient,
        };
        koro_cont_state_set(continuation, 2, KORO_SUSPEND_CALLBACK);
        std::string error;
        if (lfm::host::client_create(config, continuation, &gate->client,
                                     &error) != InProgress) {
            finish(gate, error.c_str(), EXIT_FAILURE);
        }
        return nullptr;
    }
    case 2: {
        const lfm::host::Status ready =
            lfm::host::client_ready(gate->client);
        if (frame->marker != kMarker || ready != Ok) {
            char detail[192]{};
            std::snprintf(
                detail, sizeof(detail),
                "native client readiness lost its coroutine frame "
                "(marker=%llx expected=%llx status=%d)",
                static_cast<unsigned long long>(frame->marker),
                static_cast<unsigned long long>(kMarker),
                static_cast<int>(ready));
            finish(gate, detail, EXIT_FAILURE);
        }
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->client, &snapshot) != Ok) {
            finish(gate, "cannot read first native host snapshot",
                   EXIT_FAILURE);
        }
        gate->host_generation = snapshot.host_generation;
        gate->checkpoint_identity = snapshot.checkpoint_identity;
        koro_cont_state_set(continuation, 3, KORO_SUSPEND_CALLBACK);
        if (lfm::host::test::inject_stale_completion(
                gate->client, continuation) != InProgress) {
            finish(gate, "cannot inject stale shared completion",
                   EXIT_FAILURE);
        }
        return nullptr;
    }
    case 3: {
        HostCompletion ignored{};
        if (lfm::host::client_take(gate->client, gate->identity,
                                   &ignored) != Stale) {
            finish(gate, "stale completion escaped correlation rejection",
                   EXIT_FAILURE);
        }
        koro_cont_state_set(continuation, 4, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->client, Attach, gate->identity,
                                     0, continuation) != InProgress) {
            finish(gate, "cannot submit native ATTACH", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 4: {
        HostCompletion completion{};
        if (!take(gate, Ok, Attach, &completion) ||
            completion.lease_generation == 0) {
            finish(gate, "native ATTACH did not publish one lease",
                   EXIT_FAILURE);
        }
        gate->lease_generation = completion.lease_generation;
        koro_cont_state_set(continuation, 5, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->client, Evict, gate->identity,
                                     0, continuation) != InProgress) {
            finish(gate, "cannot submit live-lease EVICT", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 5: {
        if (!take(gate, Busy, Evict)) {
            finish(gate, "EVICT did not reject a live lease", EXIT_FAILURE);
        }
        gate->task_ready.store(0, std::memory_order_release);
        gate->task_done.store(0, std::memory_order_release);
        for (uint32_t index = 0; index < kTaskCount; ++index) {
            RequestTask &task = gate->tasks[index];
            task.gate = gate;
            koro_cont_config config{
                .step = request_step,
                .argument = &task,
                .frame_size = 0,
                .worker_mask = 0,
            };
            if (koro_cont_create_on(gate->runtime, &config,
                                    &task.continuation) != 0) {
                finish(gate, "cannot create capacity continuation",
                       EXIT_FAILURE);
            }
            task.identity = koro_cont_identity(task.continuation);
            if (koro_cont_start(task.continuation) != 0) {
                finish(gate, "cannot start capacity continuation",
                       EXIT_FAILURE);
            }
        }
        return suspend(continuation, 6);
    }
    case 6: {
        if (gate->task_ready.load(std::memory_order_acquire) != kTaskCount) {
            return suspend(continuation, 6);
        }
        gate->first_host.stop_edge.store(0, std::memory_order_release);
        if (::kill(gate->first_host.pid, SIGSTOP) != 0) {
            finish(gate, "cannot suspend host for capacity gate",
                   EXIT_FAILURE);
        }
        return suspend(continuation, 7);
    }
    case 7: {
        if (gate->first_host.stop_edge.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 7);
        }
        uint32_t capacity_count = 0;
        for (uint32_t index = 0; index < kTaskCount; ++index) {
            RequestTask &task = gate->tasks[index];
            const lfm::host::Status status = lfm::host::client_submit(
                gate->client, QueryStatus, gate->identity, 0,
                task.continuation);
            if (status == InProgress) {
                task.submitted.store(1, std::memory_order_release);
                continue;
            }
            if (status == Capacity) {
                task.saw_capacity = 1;
                ++capacity_count;
                continue;
            }
            finish(gate, "shared request ring admission was not bounded",
                   EXIT_FAILURE);
        }
        if (capacity_count != 1) {
            finish(gate, "request ring did not dehydrate exactly one caller",
                   EXIT_FAILURE);
        }
        gate->capacity_edges = capacity_count;
        if (::kill(gate->first_host.pid, SIGCONT) != 0) {
            finish(gate, "cannot resume host after capacity gate",
                   EXIT_FAILURE);
        }
        return suspend(continuation, 8);
    }
    case 8: {
        if (gate->task_done.load(std::memory_order_acquire) != kTaskCount) {
            return suspend(continuation, 8);
        }
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->client, &snapshot) != Ok) {
            finish(gate, "cannot snapshot before client-death gate",
                   EXIT_FAILURE);
        }
        gate->child_event_generation = snapshot.client_events;
        if (spawn_client_child(gate, &gate->child) != 0) {
            finish(gate, "cannot spawn attached native client",
                   EXIT_FAILURE);
        }
        return suspend(continuation, 9);
    }
    case 9: {
        if (gate->child.attached.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 9);
        }
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->client, &snapshot) != Ok) {
            finish(gate, "cannot observe second native client",
                   EXIT_FAILURE);
        }
        gate->child_event_generation = snapshot.client_events;
        return suspend(continuation, 10);
    }
    case 10: {
        if (gate->child.exited.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 10);
        }
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->client, &snapshot) != Ok) {
            finish(gate, "cannot observe client-death retirement",
                   EXIT_FAILURE);
        }
        if (snapshot.active_clients == 1 && snapshot.active_leases == 1) {
            koro_cont_state_set(continuation, 13,
                                KORO_SUSPEND_CALLBACK);
            if (lfm::host::client_submit(
                    gate->client, Release, gate->identity,
                    gate->lease_generation, continuation) != InProgress) {
                finish(gate, "cannot submit native RELEASE", EXIT_FAILURE);
            }
            return nullptr;
        }
        koro_cont_state_set(continuation, 11, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_watch(gate->client,
                                    gate->child_event_generation,
                                    continuation) != InProgress) {
            finish(gate, "cannot arm client-retirement continuation",
                   EXIT_FAILURE);
        }
        return nullptr;
    }
    case 11: {
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->client, &snapshot) != Ok ||
            snapshot.active_clients != 1 || snapshot.active_leases != 1) {
            finish(gate, "dead client retained its host lease",
                   EXIT_FAILURE);
        }
        [[fallthrough]];
    }
    case 12: {
        koro_cont_state_set(continuation, 13, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(
                gate->client, Release, gate->identity,
                gate->lease_generation, continuation) != InProgress) {
            finish(gate, "cannot submit native RELEASE", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 13: {
        if (!take(gate, Ok, Release)) {
            finish(gate, "native RELEASE lost its lease identity",
                   EXIT_FAILURE);
        }
        koro_cont_state_set(continuation, 14, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->client, Evict, gate->identity,
                                     0, continuation) != InProgress) {
            finish(gate, "cannot submit privileged EVICT", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 14: {
        if (!take(gate, Ok, Evict)) {
            finish(gate, "privileged EVICT did not retire the namespace",
                   EXIT_FAILURE);
        }
        koro_cont_state_set(continuation, 15, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->client, QueryStatus,
                                     gate->identity, 0,
                                     continuation) != InProgress) {
            finish(gate, "cannot query first host status", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 15: {
        HostCompletion completion{};
        if (!take(gate, Ok, QueryStatus, &completion) ||
            (completion.result_flags & UINT32_C(0xffff)) != 1 ||
            (completion.result_flags >> 16) != 0) {
            finish(gate, "first host status accounting was not exact",
                   EXIT_FAILURE);
        }
        gate->first_host.stop_edge.store(0, std::memory_order_release);
        if (::kill(gate->first_host.pid, SIGSTOP) != 0) {
            finish(gate, "cannot suspend host for death gate", EXIT_FAILURE);
        }
        return suspend(continuation, 16);
    }
    case 16: {
        if (gate->first_host.stop_edge.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 16);
        }
        koro_cont_state_set(continuation, 17, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->client, QueryStatus,
                                     gate->identity, 0,
                                     continuation) != InProgress) {
            finish(gate, "cannot queue host-death request", EXIT_FAILURE);
        }
        if (::kill(gate->first_host.pid, SIGKILL) != 0) {
            finish(gate, "cannot kill first mailbox host", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 17: {
        HostCompletion completion{};
        if (!take(gate, HostDown, QueryStatus, &completion)) {
            return suspend(continuation, 17);
        }
        koro_cont_state_set(continuation, 18, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_begin_close(
                gate->client, continuation) != InProgress) {
            finish(gate, "cannot close dead-host client", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 18: {
        if (lfm::host::client_destroy(gate->client) != Ok) {
            finish(gate, "dead-host client retained callback ownership",
                   EXIT_FAILURE);
        }
        gate->client = nullptr;
        if (spawn_host(gate, &gate->second_host) != 0) {
            finish(gate, "cannot restart native mailbox host", EXIT_FAILURE);
        }
        return suspend(continuation, 19);
    }
    case 19: {
        if (gate->second_host.ready.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 19);
        }
        ClientConfig config{
            .service = gate->service,
            .runtime = gate->runtime,
            .flags = lfm::host::kPrivilegedClient,
        };
        koro_cont_state_set(continuation, 20, KORO_SUSPEND_CALLBACK);
        std::string error;
        if (lfm::host::client_create(config, continuation,
                                     &gate->restart_client,
                                     &error) != InProgress) {
            finish(gate, error.c_str(), EXIT_FAILURE);
        }
        return nullptr;
    }
    case 20: {
        if (lfm::host::client_ready(gate->restart_client) != Ok) {
            finish(gate, "restarted native client did not become ready",
                   EXIT_FAILURE);
        }
        HostSnapshot snapshot{};
        if (lfm::host::client_snapshot(gate->restart_client,
                                       &snapshot) != Ok ||
            snapshot.host_generation == gate->host_generation ||
            !lfm::host::identity_equal(snapshot.checkpoint_identity,
                                       gate->checkpoint_identity)) {
            finish(gate, "host restart did not change only host generation",
                   EXIT_FAILURE);
        }
        koro_cont_state_set(continuation, 21, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(gate->restart_client, QueryStatus,
                                     gate->identity, 0,
                                     continuation) != InProgress) {
            finish(gate, "cannot query restarted host", EXIT_FAILURE);
        }
        return nullptr;
    }
    case 21: {
        HostCompletion completion{};
        if (lfm::host::client_take(gate->restart_client, gate->identity,
                                   &completion) != Ok ||
            completion.status != Ok || completion.operation != QueryStatus ||
            (completion.result_flags & UINT32_C(0xffff)) != 1 ||
            (completion.result_flags >> 16) != 0) {
            finish(gate, "restarted host replayed old numerical work",
                   EXIT_FAILURE);
        }
        koro_cont_state_set(continuation, 22, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_request_stop(
                gate->restart_client, continuation) != InProgress) {
            finish(gate, "cannot retire restarted native client",
                   EXIT_FAILURE);
        }
        return nullptr;
    }
    case 22: {
        koro_cont_state_set(continuation, 23, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_begin_close(
                gate->restart_client, continuation) != InProgress) {
            finish(gate, "cannot close restarted native client",
                   EXIT_FAILURE);
        }
        return nullptr;
    }
    case 23: {
        if (lfm::host::client_destroy(gate->restart_client) != Ok) {
            finish(gate, "restarted client retained callback ownership",
                   EXIT_FAILURE);
        }
        gate->restart_client = nullptr;
        if (::kill(gate->second_host.pid, SIGTERM) != 0) {
            finish(gate, "cannot retire restarted mailbox host",
                   EXIT_FAILURE);
        }
        return suspend(continuation, 24);
    }
    case 24: {
        if (gate->second_host.exited.load(std::memory_order_acquire) == 0) {
            return suspend(continuation, 24);
        }
        char error[512]{};
        const int evicted = lfm_weights_evict(
            lfm::host::identity_bytes(gate->checkpoint_identity), error,
            sizeof(error));
        if (evicted != LFM_WEIGHT_OK) {
            finish(gate, error, EXIT_FAILURE);
        }
        std::printf(
            "{\"state\":\"verified\",\"native_only\":true,"
            "\"ring_capacity\":%u,"
            "\"capacity_dehydrations\":%u,"
            "\"multi_client_attach\":true,"
            "\"client_death_reclaimed\":true,"
            "\"host_restart_generation_changed\":true,"
            "\"stale_completion_rejected\":true,"
            "\"old_work_replayed\":false,"
            "\"frame_marker\":\"%llx\","
            "\"first_worker\":%u,\"last_worker\":%u,"
            "\"resumes\":%u}\n",
            lfm::host::kRingCapacity,
            gate->capacity_edges,
            static_cast<unsigned long long>(frame->marker),
            frame->first_worker, frame->last_worker, frame->resumes);
        std::fflush(stdout);
        finish(gate, nullptr, EXIT_SUCCESS);
    }
    default:
        finish(gate, "native mailbox continuation resumed at an invalid PC",
               EXIT_FAILURE);
    }
}

struct Child {
    std::string service;
    int evidence_fd{-1};
    kc_runtime_t *runtime{nullptr};
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
    Client *client{nullptr};
};

[[noreturn]] void child_fail(const char *message) {
    if (message) std::fprintf(stderr, "%s\n", message);
    std::_Exit(EXIT_FAILURE);
}

void child_watchdog(void *) {
    child_fail("native child client watchdog expired");
}

void *child_step(koro_cont_t *continuation) {
    auto *child = static_cast<Child *>(koro_cont_argument(continuation));
    switch (koro_cont_state_get(continuation)) {
    case 0: {
        ClientConfig config{
            .service = child->service,
            .runtime = child->runtime,
        };
        koro_cont_state_set(continuation, 1, KORO_SUSPEND_CALLBACK);
        std::string error;
        if (lfm::host::client_create(config, continuation, &child->client,
                                     &error) != InProgress) {
            child_fail(error.c_str());
        }
        return nullptr;
    }
    case 1:
        if (lfm::host::client_ready(child->client) != Ok) {
            child_fail("native child client did not become ready");
        }
        koro_cont_state_set(continuation, 2, KORO_SUSPEND_CALLBACK);
        if (lfm::host::client_submit(child->client, Attach, child->identity,
                                     0, continuation) != InProgress) {
            child_fail("native child client could not attach");
        }
        return nullptr;
    case 2: {
        HostCompletion completion{};
        if (lfm::host::client_take(child->client, child->identity,
                                   &completion) != Ok ||
            completion.status != Ok || completion.operation != Attach ||
            completion.lease_generation == 0) {
            child_fail("native child ATTACH was not correlated");
        }
        char evidence[64]{};
        const int bytes = std::snprintf(
            evidence, sizeof(evidence), "ATTACHED %llu\n",
            static_cast<unsigned long long>(completion.lease_generation));
        if (bytes <= 0 || ::write(child->evidence_fd, evidence,
                                  static_cast<size_t>(bytes)) != bytes) {
            child_fail("native child could not publish attach evidence");
        }
        (void)::close(child->evidence_fd);
        std::_Exit(EXIT_SUCCESS);
    }
    default:
        child_fail("native child continuation resumed at an invalid PC");
    }
}

int run_host(int argc, char **argv) {
    if (argc != 6) return EXIT_FAILURE;
    char *end = nullptr;
    const uint64_t owner = std::strtoull(argv[4], &end, 10);
    if (!end || *end != '\0' || owner == 0) return EXIT_FAILURE;
    const long fd = std::strtol(argv[5], &end, 10);
    if (!end || *end != '\0' || fd < 0 || fd > INT32_MAX) {
        return EXIT_FAILURE;
    }
    ServerConfig config{
        .checkpoint = argv[2],
        .service = argv[3],
        .coordination_workers = 2,
        .flags = lfm::host::kTestService,
        .privileged_pid = owner,
        .readiness_fd = static_cast<int>(fd),
    };
    std::string error;
    const lfm::host::Status status = lfm::host::serve(config, &error);
    if (status != Ok) std::fprintf(stderr, "%s\n", error.c_str());
    return status == Ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_client(int argc, char **argv) {
    if (argc != 4) return EXIT_FAILURE;
    char *end = nullptr;
    const long fd = std::strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || fd < 0 || fd > INT32_MAX) {
        return EXIT_FAILURE;
    }
    auto *child = new (std::nothrow) Child();
    if (!child) return EXIT_FAILURE;
    child->service = argv[2];
    child->evidence_fd = static_cast<int>(fd);
    kc_runtime_config runtime_config{
        .worker_count = 2,
    };
    if (kc_runtime_create(&runtime_config, &child->runtime) != 0 ||
        kc_runtime_start(child->runtime) != 0) {
        return EXIT_FAILURE;
    }
    koro_cont_config config{
        .step = child_step,
        .argument = child,
        .frame_size = sizeof(uint64_t),
        .worker_mask = 0,
    };
    if (koro_cont_create_on(child->runtime, &config,
                            &child->continuation) != 0) {
        return EXIT_FAILURE;
    }
    child->identity = koro_cont_identity(child->continuation);
    const dispatch_source_t timer = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
    if (!timer) return EXIT_FAILURE;
    dispatch_set_context(timer, child);
    dispatch_source_set_event_handler_f(timer, child_watchdog);
    dispatch_source_set_timer(
        timer,
        dispatch_time(DISPATCH_TIME_NOW,
                      static_cast<int64_t>(kWatchdogSeconds * NSEC_PER_SEC)),
        DISPATCH_TIME_FOREVER, 0);
    dispatch_activate(timer);
    if (koro_cont_start(child->continuation) != 0) return EXIT_FAILURE;
    dispatch_main();
}

int run_gate(const char *program, const char *checkpoint) {
    auto *gate = new (std::nothrow) Gate();
    if (!gate) return EXIT_FAILURE;
    gate->tasks = allocate_records<RequestTask>(kTaskCount);
    if (!gate->tasks) return EXIT_FAILURE;
    gate->program = program;
    gate->checkpoint = checkpoint;
    const uint64_t stamp = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    char service[128]{};
    std::snprintf(service, sizeof(service),
                  "com.solaceharmony.lfm.host-gate.%d.%llx", getpid(),
                  static_cast<unsigned long long>(stamp));
    gate->service = service;
    gate->queue = dispatch_queue_create(
        "com.solaceharmony.lfm.host-gate", DISPATCH_QUEUE_SERIAL);
    if (!gate->queue) return EXIT_FAILURE;
    kc_runtime_config runtime_config{
        .worker_count = 2,
    };
    if (kc_runtime_create(&runtime_config, &gate->runtime) != 0 ||
        kc_runtime_start(gate->runtime) != 0) {
        return EXIT_FAILURE;
    }
    koro_cont_config config{
        .step = gate_step,
        .argument = gate,
        .frame_size = sizeof(GateFrame),
        .worker_mask = 0,
    };
    if (koro_cont_create_on(gate->runtime, &config,
                            &gate->continuation) != 0) {
        return EXIT_FAILURE;
    }
    gate->identity = koro_cont_identity(gate->continuation);
    gate->watchdog = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0, gate->queue);
    if (!gate->watchdog) return EXIT_FAILURE;
    dispatch_set_context(gate->watchdog, gate);
    dispatch_source_set_event_handler_f(gate->watchdog, watchdog);
    dispatch_source_set_timer(
        gate->watchdog,
        dispatch_time(DISPATCH_TIME_NOW,
                      static_cast<int64_t>(kWatchdogSeconds * NSEC_PER_SEC)),
        DISPATCH_TIME_FOREVER, 0);
    dispatch_activate(gate->watchdog);
    if (koro_cont_start(gate->continuation) != 0) return EXIT_FAILURE;
    dispatch_main();
}

#endif

} // namespace

int main(int argc, char **argv) {
#ifndef __APPLE__
    (void)argc;
    (void)argv;
    std::fprintf(stderr,
                 "native host mailbox production gate requires macOS\n");
    return EXIT_FAILURE;
#else
    if (argc >= 2 && std::strcmp(argv[1], "host") == 0) {
        return run_host(argc, argv);
    }
    if (argc >= 2 && std::strcmp(argv[1], "client") == 0) {
        return run_client(argc, argv);
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s CHECKPOINT\n", argv[0]);
        return EXIT_FAILURE;
    }
    return run_gate(argv[0], argv[1]);
#endif
}
