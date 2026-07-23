// SPDX-License-Identifier: BSD-3-Clause

#include "kc_permit_broker.hpp"

#include <array>
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
    std::fprintf(stderr, "kcoro permit-broker contract failed: %s\n",
                 message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

struct Record {
    std::uint32_t id = 0;
    std::uint32_t phase = 0;
};

using Broker = kc::PermitBroker<Record, 5>;

struct Probe {
    Broker *broker = nullptr;
    kc_ticket_id capacity_identity{};
    std::array<std::uint32_t, 16> order{};
    std::array<std::uint32_t, 16> workers{};
    std::atomic_uint order_count{0};
    std::atomic_uint finished{0};
    std::atomic_uint capacity_edges{0};
    std::mutex mutex;
    std::condition_variable changed;
    bool inside = false;
    bool release = false;
};

void capacity(void *context, const kc_ticket_id *identity) {
    Probe *probe = static_cast<Probe *>(context);
    if (!probe || !identity ||
        !kc::ticket_equal(*identity,
                          probe->capacity_identity)) {
        fail("capacity edge lost its correlation identity");
    }
    probe->capacity_edges.fetch_add(1, std::memory_order_release);
}

void reset(void *, Record &record, std::uint32_t) noexcept {
    record = {};
}

kc::PermitAdvance step(void *context, Broker::Lease lease,
                       Record &record, kc::PermitEvent event) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    if (event == kc::PermitEvent::Cancel) {
        return {
            .disposition = kc::PermitDisposition::Complete,
            .status = -ECANCELED,
        };
    }
    if (event == kc::PermitEvent::Completion) {
        if (record.phase != 1) std::abort();
        record.phase = 2;
        if (record.id != 7) {
            return {
                .disposition = kc::PermitDisposition::Complete,
                .status = 0,
            };
        }
        return {
            .disposition = kc::PermitDisposition::Requeue,
            .status = 0,
        };
    }
    if (event != kc::PermitEvent::Grant) std::abort();

    const unsigned index =
        probe->order_count.fetch_add(1, std::memory_order_acq_rel);
    if (index >= probe->order.size()) std::abort();
    probe->order[index] = record.id;
    probe->workers[index] =
        probe->broker->current_worker(lease);
    probe->changed.notify_all();

    if (record.id == 7 && record.phase == 0) {
        record.phase = 1;
        {
            std::unique_lock lock(probe->mutex);
            probe->inside = true;
            probe->changed.notify_all();
            probe->changed.wait(lock, [&] { return probe->release; });
        }
        return {
            .disposition = kc::PermitDisposition::Suspend,
            .status = 0,
        };
    }
    if ((record.id == 8 || record.id == 9) &&
        record.phase == 0) {
        record.phase = 1;
        return {
            .disposition = kc::PermitDisposition::Suspend,
            .status = 0,
        };
    }
    return {
        .disposition = kc::PermitDisposition::Complete,
        .status = 0,
    };
}

void finished(void *context, Broker::Lease, Record &record,
              int status) noexcept {
    Probe *probe = static_cast<Probe *>(context);
    if (status != 0 || record.id == 0) std::abort();
    probe->finished.fetch_add(1, std::memory_order_release);
    probe->changed.notify_all();
}

void wait_count(Probe &probe, unsigned count,
                const char *message) {
    std::unique_lock lock(probe.mutex);
    check(probe.changed.wait_for(
              lock, std::chrono::seconds(5), [&] {
                  return probe.finished.load(
                             std::memory_order_acquire) >= count;
              }),
          message);
}

void wait_grants(Probe &probe, unsigned count,
                 const char *message) {
    std::unique_lock lock(probe.mutex);
    check(probe.changed.wait_for(
              lock, std::chrono::seconds(5), [&] {
                  return probe.order_count.load(
                             std::memory_order_acquire) >= count;
              }),
          message);
}

} // namespace

int main() {
    check(Broker::age(7, 9) == 0,
          "snapshot-newer work did not receive age zero");

    kc::TicketSource tickets;
    Probe probe;
    probe.capacity_identity =
        tickets.mint(KC_TICKET_KIND_CONTROL);

    const kc_runtime_config runtime_config = {
        .worker_count = 2,
    };
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&runtime_config, &runtime) == 0,
          "runtime creation failed");

    Broker broker;
    probe.broker = &broker;
    const Broker::Config config = {
        .runtime = runtime,
        .tickets = &tickets,
        .ticket_kind = KC_TICKET_KIND_WORKFLOW,
        .age_promotion = 3,
        .context = &probe,
        .operations = {
            .reset = reset,
            .step = step,
            .finished = finished,
        },
        .capacity_ready = {
            capacity, &probe, probe.capacity_identity},
    };
    check(broker.initialize(config) == 0,
          "broker initialization failed");
    check(kc_runtime_start(runtime) == 0,
          "runtime start failed");

    std::array<Broker::Claim, 5> claims;
    std::array<Broker::Lease, 5> leases;
    const std::array<kc::ServiceClass, 5> classes = {
        kc::ServiceClass::Background,
        kc::ServiceClass::Deadline,
        kc::ServiceClass::Interactive,
        kc::ServiceClass::Background,
        kc::ServiceClass::Background,
    };
    for (std::size_t index = 0; index < claims.size(); ++index) {
        check(broker.claim(classes[index], &claims[index]) == 0,
              "fixed permit claim failed");
        claims[index].record()->id =
            static_cast<std::uint32_t>(index + 1);
        leases[index] = claims[index].lease();
        check(claims[index].publish() == 0,
              "fixed permit publication failed");
    }
    Broker::Claim overflow;
    check(broker.claim(kc::ServiceClass::Deadline, &overflow) ==
              -EBUSY,
          "saturated broker admitted a sixth route");

    check(broker.start() == 0, "broker start failed");
    wait_count(probe, 5, "fair routes did not complete");

    check(probe.order_count.load(std::memory_order_acquire) == 5 &&
              probe.order[0] == 1 && probe.order[1] == 2 &&
              probe.order[2] == 3 && probe.order[3] == 4 &&
              probe.order[4] == 5,
          "service class, FIFO, or age promotion order drifted");
    for (Broker::Lease lease : leases)
        check(broker.release(lease) == 0,
              "completed route release failed");

    std::array<Broker::Claim, 2> asynchronous;
    std::array<Broker::Lease, 2> asynchronous_leases;
    for (std::size_t index = 0; index < asynchronous.size(); ++index) {
        check(broker.claim(kc::ServiceClass::Interactive,
                           &asynchronous[index]) == 0,
              "asynchronous route claim failed");
        asynchronous[index].record()->id =
            static_cast<std::uint32_t>(index + 8);
        asynchronous_leases[index] = asynchronous[index].lease();
        check(asynchronous[index].publish() == 0,
              "asynchronous route publication failed");
    }
    wait_grants(
        probe, 7,
        "a suspended route did not publish the next broker edge");
    check(broker.snapshot().running == 2,
          "two asynchronous permits did not remain in flight");
    for (Broker::Lease lease : asynchronous_leases)
        check(broker.resume(lease) == 0,
              "asynchronous completion did not resume its route");
    wait_count(probe, 7,
               "asynchronous routes did not complete");
    for (Broker::Lease lease : asynchronous_leases)
        check(broker.release(lease) == 0,
              "asynchronous route release failed");

    Broker::Claim multihop;
    check(broker.claim(kc::ServiceClass::Interactive,
                       &multihop) == 0,
          "multihop claim failed");
    multihop.record()->id = 7;
    const Broker::Lease multihop_lease = multihop.lease();
    check(multihop.publish() == 0,
          "multihop publication failed");

    {
        std::unique_lock lock(probe.mutex);
        check(probe.changed.wait_for(
                  lock, std::chrono::seconds(5),
                  [&] { return probe.inside; }),
              "multihop grant did not enter its continuation");
    }
    Broker::Lease stale = multihop_lease;
    stale.ticket.generation++;
    check(broker.resume(stale) == -ESTALE,
          "stale completion resumed a route");
    check(broker.resume(multihop_lease) == 0,
          "exact completion did not resume its route");
    {
        std::lock_guard lock(probe.mutex);
        probe.release = true;
    }
    probe.changed.notify_all();

    wait_count(probe, 8,
               "resume-during-execution lost the completion edge");
    check(probe.order_count.load(std::memory_order_acquire) == 9 &&
              probe.order[7] == 7 && probe.order[8] == 7,
          "multihop continuation did not retain and requeue its frame");
    check(broker.release(multihop_lease) == 0,
          "multihop route release failed");

    const kc::PermitBrokerSnapshot snapshot = broker.snapshot();
    check(snapshot.grants == 9 && snapshot.resumes == 3 &&
              snapshot.completions == 8 && snapshot.live == 0 &&
              snapshot.free == 5 && snapshot.ready == 0 &&
              snapshot.running == 0 && snapshot.done == 0,
          "broker terminal accounting drifted");
    check(probe.capacity_edges.load(std::memory_order_acquire) == 8,
          "capacity callbacks did not match exact releases");
    check(!kc::ticket_equal(
              broker.continuation_identity(0),
              broker.continuation_identity(1)),
          "two permits shared one continuation identity");

    broker.request_stop();
    check(kc_runtime_join_all(runtime) == 0,
          "runtime did not observe broker retirement");
    check(broker.join() == 0,
          "broker service did not retire asynchronously");
    check(broker.destroy() == 0,
          "broker setup leases did not destroy");
    kc_runtime_request_stop(runtime);
    check(kc_runtime_join(runtime) == 0,
          "runtime join failed");
    check(kc_runtime_destroy(runtime) == 0,
          "runtime destroy failed");
    return 0;
}
