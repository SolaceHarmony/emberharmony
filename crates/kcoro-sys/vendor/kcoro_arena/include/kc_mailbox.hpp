// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_coordination.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace kc {

struct MailboxSnapshot {
    std::uint64_t requests_published = 0;
    std::uint64_t requests_consumed = 0;
    std::uint64_t completions_published = 0;
    std::uint64_t completions_consumed = 0;
    std::uint32_t capacity = 0;
    std::uint32_t stopped = 0;
};

/*
 * One fixed, in-process, bidirectional coordination mailbox.
 *
 * Request and completion records are control values. Numerical payloads remain
 * in owner-held buffers and are addressed by the caller's slot/generation
 * fields. An accepted request reserves its ordinal until the matching
 * completion is consumed, so completion publication cannot be dropped for lack
 * of capacity. Every producer and consumer role has one non-copyable setup-time
 * endpoint lease. Publication never allocates, waits, polls, or runs the
 * continuation: it stores one record and fires a prebound callback edge.
 */
template <typename Request, typename Completion, std::size_t Capacity,
          std::size_t Isolation = 128>
class Mailbox final {
    static_assert(Capacity != 0);
    static_assert((Isolation & (Isolation - 1)) == 0);
    static_assert(std::is_trivially_copyable_v<Request>);
    static_assert(std::is_trivially_copyable_v<Completion>);
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(std::declval<Request>().ticket)>,
                  kc_ticket_id>);
    static_assert(std::is_same_v<std::remove_cvref_t<
                      decltype(std::declval<Completion>().ticket)>,
                  kc_ticket_id>);

    template <typename Record>
    struct alignas(Isolation) Cell {
        std::atomic<std::uint64_t> sequence{0};
        Record record{};
    };

    struct alignas(Isolation) LedgerCell {
        std::atomic<std::uint64_t> sequence{0};
        kc_ticket_id ticket{};
    };

    struct alignas(Isolation) Cursor {
        std::atomic<std::uint64_t> value{0};
    };

    static_assert(sizeof(Cell<Request>) % Isolation == 0);
    static_assert(sizeof(Cell<Completion>) % Isolation == 0);
    static_assert(sizeof(LedgerCell) % Isolation == 0);
    static_assert(sizeof(Cursor) == Isolation);

    enum : std::uint32_t {
        request_producer_role_ = UINT32_C(1) << 0,
        request_consumer_role_ = UINT32_C(1) << 1,
        completion_producer_role_ = UINT32_C(1) << 2,
        completion_consumer_role_ = UINT32_C(1) << 3,
        all_roles_ = request_producer_role_ | request_consumer_role_ |
            completion_producer_role_ | completion_consumer_role_,
    };

  public:
    struct Config {
        CallbackEdge request_ready{};
        CallbackEdge completion_ready{};
        CallbackEdge capacity_ready{};
    };

    class RequestProducer final {
      public:
        RequestProducer() noexcept = default;
        RequestProducer(const RequestProducer &) = delete;
        RequestProducer &operator=(const RequestProducer &) = delete;

        [[nodiscard]] int publish(const Request &request) noexcept {
            return owner_
                ? owner_->publish_request(generation_, request)
                : -ESTALE;
        }

      private:
        friend class Mailbox;
        Mailbox *owner_ = nullptr;
        std::uint64_t generation_ = 0;
    };

    class RequestConsumer final {
      public:
        RequestConsumer() noexcept = default;
        RequestConsumer(const RequestConsumer &) = delete;
        RequestConsumer &operator=(const RequestConsumer &) = delete;

        [[nodiscard]] int consume(Request *request) noexcept {
            return owner_
                ? owner_->consume_request(generation_, request)
                : -ESTALE;
        }

      private:
        friend class Mailbox;
        Mailbox *owner_ = nullptr;
        std::uint64_t generation_ = 0;
    };

    class CompletionProducer final {
      public:
        CompletionProducer() noexcept = default;
        CompletionProducer(const CompletionProducer &) = delete;
        CompletionProducer &operator=(const CompletionProducer &) = delete;

        [[nodiscard]] int publish(const Completion &completion) noexcept {
            return owner_
                ? owner_->publish_completion(generation_, completion)
                : -ESTALE;
        }

      private:
        friend class Mailbox;
        Mailbox *owner_ = nullptr;
        std::uint64_t generation_ = 0;
    };

    class CompletionConsumer final {
      public:
        CompletionConsumer() noexcept = default;
        CompletionConsumer(const CompletionConsumer &) = delete;
        CompletionConsumer &operator=(const CompletionConsumer &) = delete;

        [[nodiscard]] int consume(Completion *completion) noexcept {
            return owner_
                ? owner_->consume_completion(generation_, completion)
                : -ESTALE;
        }

      private:
        friend class Mailbox;
        Mailbox *owner_ = nullptr;
        std::uint64_t generation_ = 0;
    };

    struct Endpoints final {
        Endpoints() noexcept = default;
        Endpoints(const Endpoints &) = delete;
        Endpoints &operator=(const Endpoints &) = delete;

        RequestProducer requests;
        RequestConsumer request_consumer;
        CompletionProducer completions;
        CompletionConsumer completion_consumer;
    };

    Mailbox() noexcept {
        for (std::size_t index = 0; index < Capacity; ++index) {
            requests_[index].sequence.store(index,
                                             std::memory_order_relaxed);
            completions_[index].sequence.store(index,
                                                std::memory_order_relaxed);
        }
    }

    Mailbox(const Mailbox &) = delete;
    Mailbox &operator=(const Mailbox &) = delete;

    [[nodiscard]] int bind(const Config &config) noexcept {
        if (bound_ || roles_.load(std::memory_order_relaxed) != 0 ||
            !config.request_ready.valid() ||
            !config.completion_ready.valid() ||
            !config.capacity_ready.valid()) {
            return -EINVAL;
        }
        config_ = config;
        bound_ = true;
        return 0;
    }

    [[nodiscard]] int open(Endpoints *endpoints) noexcept {
        if (!bound_ || !endpoints || stopped() ||
            endpoints->requests.owner_ ||
            endpoints->request_consumer.owner_ ||
            endpoints->completions.owner_ ||
            endpoints->completion_consumer.owner_) {
            return -EINVAL;
        }
        std::uint32_t expected = 0;
        if (!roles_.compare_exchange_strong(
                expected, all_roles_, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return -EBUSY;
        }
        std::uint64_t generation =
            endpoint_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (generation == 0) {
            generation =
                endpoint_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
        }
        if (generation == 0) std::abort();
        endpoint_generation_.store(generation, std::memory_order_release);
        endpoints->requests.owner_ = this;
        endpoints->requests.generation_ = generation;
        endpoints->request_consumer.owner_ = this;
        endpoints->request_consumer.generation_ = generation;
        endpoints->completions.owner_ = this;
        endpoints->completions.generation_ = generation;
        endpoints->completion_consumer.owner_ = this;
        endpoints->completion_consumer.generation_ = generation;
        return 0;
    }

    void stop() noexcept {
        const bool first = !request_admission_.stopped();
        request_admission_.stop();
        if (!first || !bound_) return;
        config_.request_ready.fire();
        config_.completion_ready.fire();
        config_.capacity_ready.fire();
    }

    [[nodiscard]] int retire(Endpoints *endpoints) noexcept {
        if (!endpoints) return -EINVAL;
        if (roles_.load(std::memory_order_acquire) == 0) return 0;
        const std::uint64_t generation =
            endpoint_generation_.load(std::memory_order_acquire);
        if (!endpoint_valid(endpoints->requests, generation,
                            request_producer_role_) ||
            !endpoint_valid(endpoints->request_consumer, generation,
                            request_consumer_role_) ||
            !endpoint_valid(endpoints->completions, generation,
                            completion_producer_role_) ||
            !endpoint_valid(endpoints->completion_consumer, generation,
                            completion_consumer_role_)) {
            return -ESTALE;
        }
        if (!settled()) return -EBUSY;
        endpoints->requests.owner_ = nullptr;
        endpoints->requests.generation_ = 0;
        endpoints->request_consumer.owner_ = nullptr;
        endpoints->request_consumer.generation_ = 0;
        endpoints->completions.owner_ = nullptr;
        endpoints->completions.generation_ = 0;
        endpoints->completion_consumer.owner_ = nullptr;
        endpoints->completion_consumer.generation_ = 0;
        roles_.store(0, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] MailboxSnapshot snapshot() const noexcept {
        return {
            .requests_published =
                request_tail_.value.load(std::memory_order_acquire),
            .requests_consumed =
                request_head_.value.load(std::memory_order_acquire),
            .completions_published =
                completion_tail_.value.load(std::memory_order_acquire),
            .completions_consumed =
                completion_head_.value.load(std::memory_order_acquire),
            .capacity = static_cast<std::uint32_t>(Capacity),
            .stopped = stopped() ? 1u : 0u,
        };
    }

    [[nodiscard]] bool settled() const noexcept {
        const std::uint64_t requests =
            request_tail_.value.load(std::memory_order_acquire);
        return stopped() && request_admission_.active() == 0 &&
            request_head_.value.load(std::memory_order_acquire) == requests &&
            completion_tail_.value.load(std::memory_order_acquire) ==
                requests &&
            completion_head_.value.load(std::memory_order_acquire) ==
                requests;
    }

    [[nodiscard]] bool stopped() const noexcept {
        return request_admission_.stopped();
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity;
    }

  private:
    template <typename Endpoint>
    [[nodiscard]] bool endpoint_valid(const Endpoint &endpoint,
                                      std::uint64_t generation,
                                      std::uint32_t role) const noexcept {
        return endpoint.owner_ == this &&
            endpoint.generation_ == generation &&
            (roles_.load(std::memory_order_acquire) & role) != 0;
    }

    [[nodiscard]] bool generation_valid(
        std::uint64_t generation, std::uint32_t role) const noexcept {
        return generation != 0 &&
            endpoint_generation_.load(std::memory_order_acquire) ==
                generation &&
            (roles_.load(std::memory_order_acquire) & role) != 0;
    }

    [[nodiscard]] int publish_request(std::uint64_t generation,
                                      const Request &request) noexcept {
        if (!generation_valid(generation, request_producer_role_))
            return -ESTALE;
        if (!ticket_valid(request.ticket)) return -EINVAL;
        const int admitted = request_admission_.enter();
        if (admitted != 0) return admitted;
        const std::uint64_t tail =
            request_tail_.value.load(std::memory_order_relaxed);
        const std::uint64_t completed =
            completion_head_.value.load(std::memory_order_acquire);
        if (tail - completed >= Capacity) {
            request_admission_.leave();
            return -EAGAIN;
        }
        const std::size_t index = tail % Capacity;
        Cell<Request> &cell = requests_[index];
        if (cell.sequence.load(std::memory_order_acquire) != tail ||
            ledger_[index].sequence.load(std::memory_order_acquire) != 0) {
            std::abort();
        }
        ledger_[index].ticket = request.ticket;
        ledger_[index].sequence.store(tail + 1, std::memory_order_release);
        cell.record = request;
        cell.sequence.store(tail + 1, std::memory_order_release);
        request_tail_.value.store(tail + 1, std::memory_order_release);
        request_admission_.leave();
        config_.request_ready.fire();
        return 0;
    }

    [[nodiscard]] int consume_request(std::uint64_t generation,
                                      Request *request) noexcept {
        if (!generation_valid(generation, request_consumer_role_))
            return -ESTALE;
        if (!request) return -EINVAL;
        const std::uint64_t head =
            request_head_.value.load(std::memory_order_relaxed);
        const std::uint64_t tail =
            request_tail_.value.load(std::memory_order_acquire);
        if (head == tail) {
            return stopped() && request_admission_.active() == 0
                ? -ECANCELED
                : -EAGAIN;
        }
        Cell<Request> &cell = requests_[head % Capacity];
        if (cell.sequence.load(std::memory_order_acquire) != head + 1)
            std::abort();
        *request = cell.record;
        cell.sequence.store(head + Capacity, std::memory_order_release);
        request_head_.value.store(head + 1, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] int publish_completion(
        std::uint64_t generation, const Completion &completion) noexcept {
        if (!generation_valid(generation, completion_producer_role_))
            return -ESTALE;
        if (!ticket_valid(completion.ticket)) return -EINVAL;
        const std::uint64_t tail =
            completion_tail_.value.load(std::memory_order_relaxed);
        const std::uint64_t dispatched =
            request_head_.value.load(std::memory_order_acquire);
        if (tail == dispatched) return -EAGAIN;
        const std::uint64_t head =
            completion_head_.value.load(std::memory_order_acquire);
        if (tail - head >= Capacity) return -EOVERFLOW;
        const std::size_t index = tail % Capacity;
        Cell<Completion> &cell = completions_[index];
        if (cell.sequence.load(std::memory_order_acquire) != tail)
            std::abort();
        if (ledger_[index].sequence.load(std::memory_order_acquire) !=
                tail + 1 ||
            !ticket_equal(ledger_[index].ticket, completion.ticket)) {
            return -ESTALE;
        }
        cell.record = completion;
        cell.sequence.store(tail + 1, std::memory_order_release);
        completion_tail_.value.store(tail + 1, std::memory_order_release);
        config_.completion_ready.fire();
        return 0;
    }

    [[nodiscard]] int consume_completion(
        std::uint64_t generation, Completion *completion) noexcept {
        if (!generation_valid(generation, completion_consumer_role_))
            return -ESTALE;
        if (!completion) return -EINVAL;
        const std::uint64_t head =
            completion_head_.value.load(std::memory_order_relaxed);
        const std::uint64_t tail =
            completion_tail_.value.load(std::memory_order_acquire);
        if (head == tail) {
            return stopped() && request_admission_.active() == 0 &&
                    head ==
                        request_tail_.value.load(std::memory_order_acquire)
                ? -ECANCELED
                : -EAGAIN;
        }
        const std::size_t index = head % Capacity;
        Cell<Completion> &cell = completions_[index];
        if (cell.sequence.load(std::memory_order_acquire) != head + 1)
            std::abort();
        *completion = cell.record;
        if (ledger_[index].sequence.load(std::memory_order_acquire) !=
                head + 1 ||
            !ticket_equal(ledger_[index].ticket, completion->ticket)) {
            std::abort();
        }
        ledger_[index].ticket = {};
        ledger_[index].sequence.store(0, std::memory_order_release);
        cell.sequence.store(head + Capacity, std::memory_order_release);
        completion_head_.value.store(head + 1, std::memory_order_release);
        config_.capacity_ready.fire();
        return 0;
    }

    Config config_{};
    bool bound_ = false;
    std::atomic<std::uint32_t> roles_{0};
    std::atomic<std::uint64_t> endpoint_sequence_{0};
    std::atomic<std::uint64_t> endpoint_generation_{0};
    AdmissionGate<1> request_admission_;
    std::array<Cell<Request>, Capacity> requests_{};
    std::array<Cell<Completion>, Capacity> completions_{};
    std::array<LedgerCell, Capacity> ledger_{};
    Cursor request_head_{};
    Cursor request_tail_{};
    Cursor completion_head_{};
    Cursor completion_tail_{};
};

} // namespace kc
