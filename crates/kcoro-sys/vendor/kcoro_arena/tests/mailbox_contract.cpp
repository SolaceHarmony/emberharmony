// SPDX-License-Identifier: BSD-3-Clause

#include "kc_mailbox.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "kcoro mailbox contract failed: %s\n", message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

struct Request {
    kc_ticket_id ticket{};
    kc_ticket_id parent{};
    std::uint64_t lease_generation = 0;
    std::uint32_t slot = 0;
    std::uint32_t operation = 0;
};

struct Completion {
    kc_ticket_id ticket{};
    std::uint64_t lease_generation = 0;
    std::uint32_t slot = 0;
    std::int32_t status = 0;
};

struct Probe {
    kc_ticket_id identity{};
    std::atomic_uint edges{0};
};

void publish(void *context, const kc_ticket_id *identity) {
    Probe *probe = static_cast<Probe *>(context);
    if (!probe || !identity ||
        !kc::ticket_equal(probe->identity, *identity)) {
        fail("callback did not carry its exact correlation identity");
    }
    probe->edges.fetch_add(1, std::memory_order_release);
}

Request request(kc::TicketSource &source, const kc_ticket_id &parent,
                std::uint64_t generation, std::uint32_t slot) {
    return {
        .ticket = source.mint(KC_TICKET_KIND_PASS),
        .parent = parent,
        .lease_generation = generation,
        .slot = slot,
        .operation = 7,
    };
}

Completion completion(const Request &request) {
    return {
        .ticket = request.ticket,
        .lease_generation = request.lease_generation,
        .slot = request.slot,
        .status = 0,
    };
}

} // namespace

int main() {
    using Mailbox = kc::Mailbox<Request, Completion, 2>;
    kc::TicketSource tickets;
    Probe requests{.identity = tickets.mint(KC_TICKET_KIND_CONTROL)};
    Probe completions{.identity = tickets.mint(KC_TICKET_KIND_CONTROL)};
    Probe capacity{.identity = tickets.mint(KC_TICKET_KIND_CONTROL)};
    Mailbox mailbox;
    check(mailbox.bind({
              .request_ready = {publish, &requests, requests.identity},
              .completion_ready =
                  {publish, &completions, completions.identity},
              .capacity_ready = {publish, &capacity, capacity.identity},
          }) == 0,
          "mailbox did not bind exact callback edges");
    Mailbox::Endpoints endpoints;
    check(mailbox.open(&endpoints) == 0,
          "mailbox endpoints did not open");
    Mailbox::Endpoints duplicate;
    check(mailbox.open(&duplicate) == -EBUSY,
          "mailbox issued a second endpoint set");

    const kc_ticket_id workflow =
        tickets.mint(KC_TICKET_KIND_WORKFLOW);
    const Request first = request(tickets, workflow, 11, 0);
    const Request second = request(tickets, workflow, 12, 1);
    const Request third = request(tickets, workflow, 13, 0);
    check(endpoints.requests.publish(first) == 0 &&
              endpoints.requests.publish(second) == 0,
          "mailbox rejected available request capacity");
    check(endpoints.requests.publish(third) == -EAGAIN,
          "mailbox did not dehydrate a saturated producer");

    Request observed{};
    check(endpoints.request_consumer.consume(&observed) == 0 &&
              kc::ticket_equal(observed.ticket, first.ticket),
          "request consumer did not receive the first ticket");
    check(endpoints.request_consumer.consume(&observed) == 0 &&
              kc::ticket_equal(observed.ticket, second.ticket),
          "request consumer did not receive the second ticket");
    check(endpoints.request_consumer.consume(&observed) == -EAGAIN,
          "empty live request edge did not return EAGAIN");

    Completion wrong = completion(first);
    wrong.ticket = third.ticket;
    check(endpoints.completions.publish(wrong) == -ESTALE,
          "completion producer accepted the wrong ticket");
    check(endpoints.completions.publish(completion(first)) == 0 &&
              endpoints.completions.publish(completion(second)) == 0,
          "completion producer lost an accepted ticket");

    Completion result{};
    check(endpoints.completion_consumer.consume(&result) == 0 &&
              kc::ticket_equal(result.ticket, first.ticket),
          "completion consumer did not receive the first ticket");
    check(endpoints.requests.publish(third) == 0,
          "capacity callback did not correspond to reusable capacity");
    check(endpoints.completion_consumer.consume(&result) == 0 &&
              kc::ticket_equal(result.ticket, second.ticket),
          "completion consumer did not receive the second ticket");
    check(endpoints.request_consumer.consume(&observed) == 0 &&
              kc::ticket_equal(observed.ticket, third.ticket),
          "reused request cell did not carry its new generation");

    mailbox.stop();
    check(endpoints.requests.publish(first) == -ECANCELED,
          "stopped mailbox admitted a new request");
    check(endpoints.completions.publish(completion(third)) == 0,
          "stop discarded an already accepted completion");
    check(endpoints.completion_consumer.consume(&result) == 0 &&
              kc::ticket_equal(result.ticket, third.ticket),
          "final accepted completion did not drain");
    check(endpoints.request_consumer.consume(&observed) == -ECANCELED &&
              endpoints.completion_consumer.consume(&result) ==
                  -ECANCELED,
          "drained stopped endpoints did not retire");

    const kc::MailboxSnapshot snapshot = mailbox.snapshot();
    check(snapshot.capacity == 2 && snapshot.stopped == 1 &&
              snapshot.requests_published == 3 &&
              snapshot.requests_consumed == 3 &&
              snapshot.completions_published == 3 &&
              snapshot.completions_consumed == 3,
          "mailbox snapshot did not describe the exact exchange");
    check(requests.edges.load(std::memory_order_acquire) == 4 &&
              completions.edges.load(std::memory_order_acquire) == 4 &&
              capacity.edges.load(std::memory_order_acquire) == 4,
          "mailbox lost or duplicated a publication edge");
    check(mailbox.retire(&endpoints) == 0,
          "settled mailbox endpoints did not retire");
    check(endpoints.requests.publish(first) == -ESTALE,
          "retired endpoint still published a request");
    return 0;
}
