// SPDX-License-Identifier: BSD-3-Clause

#include "kc_coordination.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "kcoro coordination contract failed: %s\n", message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

void verify_ticket_source() {
    kc::TicketSource source;
    const kc_ticket_id parent = source.mint(KC_TICKET_KIND_WORKFLOW);
    const kc_ticket_id pass = source.mint(KC_TICKET_KIND_PASS);
    check(kc::ticket_valid(parent) && kc::ticket_valid(pass),
          "ticket source minted an invalid identity");
    check(parent.runtime_epoch == pass.runtime_epoch,
          "one source changed runtime epoch");
    check(parent.sequence != pass.sequence &&
              parent.generation != pass.generation,
          "ticket source reused identity coordinates");
    check(parent.kind == KC_TICKET_KIND_WORKFLOW &&
              pass.kind == KC_TICKET_KIND_PASS,
          "ticket source changed the requested kind");
    check(!kc::ticket_equal(parent, pass),
          "distinct tickets compare equal");
}

void notify(void *context) {
    static_cast<std::atomic_uint *>(context)->fetch_add(
        1, std::memory_order_release);
}

void verify_admission_gate() {
    std::atomic_uint edges{0};
    kc::AdmissionGate<2> gate(notify, &edges);
    check(gate.enter() == 0 && gate.enter() == 0,
          "admission rejected available capacity");
    check(gate.enter() == -EBUSY,
          "admission exceeded its fixed capacity");
    check(gate.active() == 2, "admission count is incorrect");
    check(gate.try_seal() == -EBUSY,
          "active admission was sealed");
    check(!gate.sealed(), "failed seal did not reopen admission");
    gate.leave();
    gate.leave();
    check(gate.try_seal() == 0, "idle admission did not seal");
    check(gate.sealed(), "admission did not report its seal");
    check(gate.enter() == -EBUSY,
          "sealed admission accepted a publisher");
    check(gate.unseal() == 0, "sealed admission did not reopen");
    check(gate.enter() == 0, "reopened admission rejected a publisher");
    gate.leave();
    gate.stop();
    check(gate.stopped(), "admission did not stop");
    check(gate.enter() == -ECANCELED,
          "stopped admission accepted a publisher");
    check(gate.active() == 0, "admission leases did not retire");
    check(edges.load(std::memory_order_acquire) == 6,
          "capacity and lifecycle edges were not published exactly");

    kc::AdmissionGate<1> stopped;
    check(stopped.try_seal() == 0, "idle stop-race gate did not seal");
    stopped.stop();
    check(stopped.unseal() == -ECANCELED,
          "temporary unseal reopened permanently stopped admission");
    check(stopped.enter() == -ECANCELED,
          "stopped admission accepted work after an unseal race");
}

struct Record {
    std::uint32_t index = UINT32_MAX;
    std::uint32_t resets = 0;
    std::uint64_t value = 0;
};

void verify_slot_pool() {
    kc::SlotPool<Record, 2> pool;
    const auto reset = [](Record &record, std::uint32_t index) {
        record.index = index;
        ++record.resets;
        record.value = 0;
    };
    const auto first = pool.acquire(1, reset);
    const auto second = pool.acquire(1, reset);
    const auto full = pool.acquire(1, reset);
    check(first && second && !full,
          "slot pool did not enforce fixed capacity");
    check(first.index != second.index,
          "two leases claimed the same slot");
    check(pool.get(first) != nullptr && pool.get(second) != nullptr,
          "exact leases did not resolve their records");
    pool.get(first)->value = 42;
    check(pool.transition(first, 1, 2),
          "exact lease could not advance state");
    check(!pool.transition(first, 1, 2),
          "slot state advanced twice");
    check(pool.begin_release(first, 2),
          "exact lease could not begin release");
    pool.finish_release(first);
    check(pool.get(first) == nullptr,
          "released lease still resolved its record");

    const auto replacement = pool.acquire(1, reset);
    check(replacement && replacement.index == first.index,
          "released slot was not reusable");
    check(replacement.generation != first.generation,
          "slot reuse did not change generation");
    check(!pool.transition(first, 1, 2) &&
              !pool.begin_release(first, 1),
          "stale lease mutated its successor");
    check(pool.get(replacement)->resets == 2 &&
              pool.get(replacement)->value == 0,
          "slot reset did not run before publication");

    check(pool.begin_release(replacement, 1),
          "replacement release failed");
    pool.finish_release(replacement);
    check(pool.begin_release(second, 1),
          "second release failed");
    pool.finish_release(second);
}

} // namespace

int main() {
    verify_ticket_source();
    verify_admission_gate();
    verify_slot_pool();
    return 0;
}
