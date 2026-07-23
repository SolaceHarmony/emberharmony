// SPDX-License-Identifier: BSD-3-Clause

#include "kc_team_executor.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "kcoro team-executor contract failed: %s\n",
                 message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

struct Request {
    kc_ticket_id ticket{};
    std::uint32_t operation = 0;
    std::uint32_t value = 0;
};

struct Completion {
    kc_ticket_id ticket{};
    std::int32_t status = 0;
    std::uint32_t value = 0;
};

using Executor = kc::TeamExecutor<Request, Completion, 2>;

struct Probe {
    Executor *executor = nullptr;
    kc_ticket_id capacity_identity{};
    std::atomic_uint capacity_edges{0};
    std::atomic_uint member_calls{0};
    std::atomic_uint returned_edges{0};
    std::atomic_uint finishes{0};
    std::atomic_uint retirements{0};
    std::atomic_uint generations_in_request{0};
    std::atomic_uint completion_mask{0};
    std::mutex mutex;
    std::condition_variable changed;
    bool retired = false;
};

void publish_capacity(void *context,
                      const kc_ticket_id *identity) {
    Probe *probe = static_cast<Probe *>(context);
    if (!probe || !identity ||
        !kc::ticket_equal(*identity,
                          probe->capacity_identity)) {
        fail("capacity edge lost its correlation identity");
    }
    probe->capacity_edges.fetch_add(1, std::memory_order_release);
}

int begin(void *context, const Request &request) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    probe->generations_in_request.store(0,
                                        std::memory_order_release);
    return request.operation == 99 ? -EPROTO : 0;
}

void run_member(void *context, std::uint32_t member,
                std::uint32_t members,
                std::uint64_t generation) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    if (members != 4 || member >= members || generation == 0)
        std::abort();
    probe->member_calls.fetch_add(1, std::memory_order_relaxed);
}

void returned(void *context, const Request &,
              std::uint64_t) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    probe->returned_edges.fetch_add(1, std::memory_order_release);
}

kc::TeamAdvance advance(void *context, const Request &request,
                        std::uint64_t) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    const unsigned generation =
        probe->generations_in_request.fetch_add(
            1, std::memory_order_acq_rel) + 1;
    if (request.operation == 2 && generation == 1) {
        return {
            .disposition = kc::TeamDisposition::Next,
            .status = 0,
        };
    }
    return {
        .disposition = kc::TeamDisposition::Complete,
        .status = 0,
    };
}

void make_completion(void *, const Request &request, int status,
                     Completion *completion) noexcept {
    *completion = {
        .ticket = request.ticket,
        .status = status,
        .value = status == 0 ? request.value + 1 : 0,
    };
}

void finish(void *context, const Request &request,
            const Completion &completion) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    if (!kc::ticket_equal(request.ticket, completion.ticket))
        std::abort();
    const unsigned bit = request.operation == 2 ? 1u : 2u;
    if (request.operation == 2) {
        if (completion.status != 0 ||
            completion.value != request.value + 1) {
            std::abort();
        }
    } else if (completion.status != -EPROTO ||
               completion.value != 0) {
        std::abort();
    }
    probe->completion_mask.fetch_or(bit, std::memory_order_acq_rel);
    if (probe->finishes.fetch_add(1, std::memory_order_acq_rel) + 1 ==
        2) {
        probe->executor->request_stop();
    }
}

void retired(void *context, const kc_ticket_id *identity) {
    Probe *probe = static_cast<Probe *>(context);
    if (!probe || !identity ||
        !kc::ticket_equal(*identity,
                          probe->executor->identity())) {
        fail("retirement edge named the wrong continuation");
    }
    check(probe->retirements.fetch_add(
              1, std::memory_order_acq_rel) == 0,
          "retirement callback published twice");
    {
        std::lock_guard lock(probe->mutex);
        probe->retired = true;
    }
    probe->changed.notify_all();
}

} // namespace

int main() {
    kc::TicketSource tickets;
    Probe probe;
    probe.capacity_identity =
        tickets.mint(KC_TICKET_KIND_CONTROL);

    const kc_runtime_config runtime_config{
        .worker_count = 2,
    };
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&runtime_config, &runtime) == 0,
          "runtime creation failed");

    Executor executor;
    probe.executor = &executor;
    const Executor::Config config{
        .runtime = runtime,
        .member_count = 4,
        .context = &probe,
        .operations = {
            .begin = begin,
            .run_member = run_member,
            .generation_returned = returned,
            .advance = advance,
            .make_completion = make_completion,
            .finish = finish,
        },
        .capacity_ready = {
            publish_capacity, &probe, probe.capacity_identity},
        .retired = retired,
        .retired_context = &probe,
    };
    check(executor.initialize(config) == 0,
          "executor initialization failed");
    check(kc_runtime_start(runtime) == 0,
          "runtime start failed");
    check(executor.start() == 0,
          "executor start failed");

    const Request first{
        .ticket = tickets.mint(KC_TICKET_KIND_PASS),
        .operation = 2,
        .value = 41,
    };
    const Request rejected{
        .ticket = tickets.mint(KC_TICKET_KIND_PASS),
        .operation = 99,
        .value = 7,
    };
    check(executor.submit(first) == 0 &&
              executor.submit(rejected) == 0,
          "executor did not accept its fixed request capacity");

    {
        std::unique_lock lock(probe.mutex);
        check(probe.changed.wait_for(
                  lock, std::chrono::seconds(5),
                  [&] { return probe.retired; }),
              "executor retirement edge did not arrive");
    }

    const kc::TeamExecutorSnapshot snapshot = executor.snapshot();
    check(snapshot.phase ==
              static_cast<std::uint32_t>(
                  kc::TeamExecutorPhase::Done) &&
              snapshot.requests_started == 2 &&
              snapshot.requests_finished == 2 &&
              snapshot.generations_dispatched == 2 &&
              snapshot.generations_returned == 2 &&
              snapshot.returned_generation == 0 &&
              snapshot.generation == 0 &&
              snapshot.retired == 1,
          "executor snapshot lost terminal accounting");
    check(snapshot.mailbox.requests_published == 2 &&
              snapshot.mailbox.requests_consumed == 2 &&
              snapshot.mailbox.completions_published == 2 &&
              snapshot.mailbox.completions_consumed == 2,
          "executor mailbox did not retire exact records");
    check(probe.member_calls.load(std::memory_order_acquire) == 8 &&
              probe.returned_edges.load(std::memory_order_acquire) == 2 &&
              probe.finishes.load(std::memory_order_acquire) == 2 &&
              probe.completion_mask.load(std::memory_order_acquire) == 3,
          "executor lost a generation or exact completion");

    check(executor.destroy() == 0,
          "executor setup leases did not destroy");
    kc_runtime_request_stop(runtime);
    check(kc_runtime_join(runtime) == 0,
          "runtime join failed");
    check(kc_runtime_destroy(runtime) == 0,
          "runtime destroy failed");
    return 0;
}
