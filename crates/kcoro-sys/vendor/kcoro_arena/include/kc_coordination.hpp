// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_identity.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace kc {

enum class ServiceClass : std::uint32_t {
    Deadline = 1,
    Interactive = 2,
    Background = 3,
};

inline bool ticket_equal(const kc_ticket_id &left,
                         const kc_ticket_id &right) noexcept {
    return left.runtime_epoch == right.runtime_epoch &&
        left.sequence == right.sequence &&
        left.generation == right.generation && left.kind == right.kind;
}

inline bool ticket_valid(const kc_ticket_id &ticket) noexcept {
    return ticket.runtime_epoch != 0 && ticket.sequence != 0 &&
        ticket.generation != 0 && ticket.kind != 0;
}

class TicketSource final {
  public:
    TicketSource() noexcept : epoch_(next(epoch_source_)) {}

    explicit TicketSource(std::uint64_t epoch) noexcept : epoch_(epoch) {
        if (epoch_ == 0) std::abort();
    }

    TicketSource(const TicketSource &) = delete;
    TicketSource &operator=(const TicketSource &) = delete;

    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

    [[nodiscard]] kc_ticket_id mint(std::uint32_t kind) noexcept {
        if (kind == 0) std::abort();
        return {
            .runtime_epoch = epoch_,
            .sequence = next(sequence_),
            .generation = next(generation_),
            .kind = kind,
        };
    }

  private:
    template <typename Value>
    static Value next(std::atomic<Value> &source) noexcept {
        static_assert(std::is_unsigned_v<Value>);
        Value value = source.fetch_add(1, std::memory_order_relaxed) + 1;
        if (value != 0) return value;
        value = source.fetch_add(1, std::memory_order_relaxed) + 1;
        if (value == 0) std::abort();
        return value;
    }

    inline static std::atomic<std::uint64_t> epoch_source_{0};
    const std::uint64_t epoch_;
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<std::uint32_t> generation_{0};
};

template <std::uint32_t Capacity>
class AdmissionGate final {
    static_assert(Capacity != 0);
    enum : std::uint32_t {
        open_state_ = 0,
        sealed_state_ = 1,
        stopped_state_ = 2,
    };

  public:
    using Notify = void (*)(void *);

    AdmissionGate() noexcept = default;

    AdmissionGate(Notify notify, void *context) noexcept
        : notify_(notify), context_(context) {}

    AdmissionGate(const AdmissionGate &) = delete;
    AdmissionGate &operator=(const AdmissionGate &) = delete;

    void bind(Notify notify, void *context) noexcept {
        if (active_.load(std::memory_order_relaxed) != 0 ||
            state_.load(std::memory_order_relaxed) != open_state_) {
            std::abort();
        }
        notify_ = notify;
        context_ = context;
    }

    [[nodiscard]] int enter() noexcept {
        const std::uint32_t state =
            state_.load(std::memory_order_seq_cst);
        if (state != open_state_)
            return state == stopped_state_ ? -ECANCELED : -EBUSY;
        const std::uint32_t previous =
            active_.fetch_add(1, std::memory_order_seq_cst);
        if (previous >= Capacity) {
            active_.fetch_sub(1, std::memory_order_seq_cst);
            return -EBUSY;
        }
        const std::uint32_t after =
            state_.load(std::memory_order_seq_cst);
        if (after == open_state_) return 0;
        active_.fetch_sub(1, std::memory_order_seq_cst);
        return after == stopped_state_ ? -ECANCELED : -EBUSY;
    }

    void leave() noexcept {
        const std::uint32_t previous =
            active_.fetch_sub(1, std::memory_order_seq_cst);
        if (previous == 0 || previous > Capacity) std::abort();
        publish();
    }

    [[nodiscard]] int try_seal() noexcept {
        std::uint32_t expected = open_state_;
        if (!state_.compare_exchange_strong(
                expected, sealed_state_, std::memory_order_seq_cst,
                std::memory_order_seq_cst)) {
            return expected == stopped_state_ ? -ECANCELED : -EBUSY;
        }
        if (active_.load(std::memory_order_seq_cst) == 0) return 0;
        expected = sealed_state_;
        if (!state_.compare_exchange_strong(
                expected, open_state_, std::memory_order_seq_cst,
                std::memory_order_seq_cst)) {
            if (expected == stopped_state_) return -ECANCELED;
            std::abort();
        }
        publish();
        return -EBUSY;
    }

    [[nodiscard]] int unseal() noexcept {
        if (active_.load(std::memory_order_seq_cst) != 0) return -EBUSY;
        std::uint32_t expected = sealed_state_;
        if (!state_.compare_exchange_strong(
                expected, open_state_, std::memory_order_seq_cst,
                std::memory_order_seq_cst)) {
            return expected == stopped_state_ ? -ECANCELED : -EINVAL;
        }
        publish();
        return 0;
    }

    void stop() noexcept {
        if (state_.exchange(stopped_state_,
                            std::memory_order_seq_cst) != stopped_state_) {
            publish();
        }
    }

    [[nodiscard]] bool stopped() const noexcept {
        return state_.load(std::memory_order_seq_cst) == stopped_state_;
    }

    [[nodiscard]] bool sealed() const noexcept {
        return state_.load(std::memory_order_seq_cst) != open_state_;
    }

    [[nodiscard]] std::uint32_t active() const noexcept {
        return active_.load(std::memory_order_seq_cst);
    }

  private:
    void publish() noexcept {
        if (notify_) notify_(context_);
    }

    std::atomic<std::uint32_t> state_{open_state_};
    std::atomic<std::uint32_t> active_{0};
    Notify notify_ = nullptr;
    void *context_ = nullptr;
};

template <typename Record, std::size_t Capacity,
          std::size_t Isolation = 128>
class SlotPool final {
    static_assert(Capacity != 0);
    static_assert(Isolation >= alignof(std::atomic<std::uint64_t>));
    static_assert((Isolation & (Isolation - 1)) == 0);

    static constexpr std::uint64_t state_bits_ = 8;
    static constexpr std::uint64_t state_mask_ =
        (UINT64_C(1) << state_bits_) - 1;
    static constexpr std::uint32_t claiming_state_ =
        static_cast<std::uint32_t>(state_mask_);
    static constexpr std::uint32_t releasing_state_ =
        claiming_state_ - 1;
    static constexpr std::uint64_t maximum_generation_ =
        std::numeric_limits<std::uint64_t>::max() >> state_bits_;

    static constexpr std::uint64_t word(std::uint64_t generation,
                                        std::uint32_t state) noexcept {
        return (generation << state_bits_) | state;
    }

    struct alignas(Isolation) Cell {
        std::atomic<std::uint64_t> lease{word(0, 0)};
        std::atomic<std::uint64_t> sequence{0};
        Record record{};
    };

    static_assert(sizeof(Cell) % Isolation == 0);

  public:
    struct Lease {
        std::uint32_t index = UINT32_MAX;
        std::uint64_t generation = 0;

        [[nodiscard]] explicit operator bool() const noexcept {
            return index != UINT32_MAX && generation != 0;
        }
    };

    SlotPool() noexcept = default;
    SlotPool(const SlotPool &) = delete;
    SlotPool &operator=(const SlotPool &) = delete;

    template <typename Reset>
    [[nodiscard]] Lease acquire(std::uint32_t initial_state,
                                Reset &&reset) noexcept {
        if (initial_state == 0 || initial_state >= releasing_state_)
            std::abort();
        for (std::size_t index = 0; index < Capacity; ++index) {
            Cell &cell = cells_[index];
            std::uint64_t expected = word(0, 0);
            if (!cell.lease.compare_exchange_strong(
                    expected, word(0, claiming_state_),
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                continue;
            }
            std::uint64_t generation =
                (cell.sequence.fetch_add(1, std::memory_order_acq_rel) + 1) &
                maximum_generation_;
            if (generation == 0) {
                generation =
                    (cell.sequence.fetch_add(1,
                                             std::memory_order_acq_rel) + 1) &
                    maximum_generation_;
            }
            if (generation == 0) std::abort();
            reset(cell.record, static_cast<std::uint32_t>(index));
            cell.lease.store(word(generation, initial_state),
                             std::memory_order_release);
            return {
                .index = static_cast<std::uint32_t>(index),
                .generation = generation,
            };
        }
        return {};
    }

    [[nodiscard]] bool transition(Lease lease, std::uint32_t from,
                                  std::uint32_t to) noexcept {
        if (!valid(lease) || from == 0 || to == 0 ||
            from >= releasing_state_ || to >= releasing_state_) {
            return false;
        }
        std::uint64_t expected = word(lease.generation, from);
        return cells_[lease.index].lease.compare_exchange_strong(
            expected, word(lease.generation, to),
            std::memory_order_acq_rel, std::memory_order_acquire);
    }

    [[nodiscard]] bool begin_release(Lease lease,
                                     std::uint32_t from) noexcept {
        if (!valid(lease) || from == 0 || from >= releasing_state_)
            return false;
        std::uint64_t expected = word(lease.generation, from);
        return cells_[lease.index].lease.compare_exchange_strong(
            expected, word(lease.generation, releasing_state_),
            std::memory_order_acq_rel, std::memory_order_acquire);
    }

    void finish_release(Lease lease) noexcept {
        if (!valid(lease) ||
            cells_[lease.index].lease.load(std::memory_order_acquire) !=
                word(lease.generation, releasing_state_)) {
            std::abort();
        }
        cells_[lease.index].lease.store(word(0, 0),
                                        std::memory_order_release);
    }

    [[nodiscard]] Record *get(Lease lease) noexcept {
        if (!matches(lease)) return nullptr;
        return &cells_[lease.index].record;
    }

    [[nodiscard]] const Record *get(Lease lease) const noexcept {
        if (!matches(lease)) return nullptr;
        return &cells_[lease.index].record;
    }

    [[nodiscard]] Record &record(std::size_t index) noexcept {
        if (index >= Capacity) std::abort();
        return cells_[index].record;
    }

    [[nodiscard]] const Record &record(std::size_t index) const noexcept {
        if (index >= Capacity) std::abort();
        return cells_[index].record;
    }

    [[nodiscard]] std::uint32_t state(std::size_t index) const noexcept {
        if (index >= Capacity) return 0;
        return static_cast<std::uint32_t>(
            cells_[index].lease.load(std::memory_order_acquire) &
            state_mask_);
    }

    [[nodiscard]] std::uint64_t generation(
        std::size_t index) const noexcept {
        if (index >= Capacity) return 0;
        return cells_[index].lease.load(std::memory_order_acquire) >>
            state_bits_;
    }

    [[nodiscard]] bool matches(Lease lease) const noexcept {
        if (!valid(lease)) return false;
        const std::uint64_t current =
            cells_[lease.index].lease.load(std::memory_order_acquire);
        const std::uint32_t state =
            static_cast<std::uint32_t>(current & state_mask_);
        return (current >> state_bits_) == lease.generation &&
            state != 0 && state < releasing_state_;
    }

    [[nodiscard]] static constexpr std::size_t size() noexcept {
        return Capacity;
    }

  private:
    [[nodiscard]] static constexpr bool valid(Lease lease) noexcept {
        return lease.index < Capacity && lease.generation != 0;
    }

    std::array<Cell, Capacity> cells_{};
};

} // namespace kc
