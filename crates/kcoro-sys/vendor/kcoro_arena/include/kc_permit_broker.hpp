// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_coordination.hpp"
#include "kc_service.h"
#include "kcoro_stackless.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace kc {

enum class PermitEvent : std::uint32_t {
    Grant = 1,
    Completion = 2,
    Cancel = 3,
    Stop = 4,
};

enum class PermitDisposition : std::uint32_t {
    Suspend = 1,
    Requeue = 2,
    Preserve = 3,
    Complete = 4,
};

struct PermitAdvance {
    PermitDisposition disposition = PermitDisposition::Complete;
    int status = -EINVAL;
};

struct PermitBrokerSnapshot {
    std::uint64_t enqueue_sequence = 0;
    std::uint64_t grants = 0;
    std::uint64_t resumes = 0;
    std::uint64_t completions = 0;
    std::uint64_t deferrals = 0;
    std::uint32_t free = 0;
    std::uint32_t claimed = 0;
    std::uint32_t ready = 0;
    std::uint32_t running = 0;
    std::uint32_t done = 0;
    std::uint32_t live = 0;
    std::uint32_t stopped = 0;
};

/*
 * A fixed fair broker whose permits are retained logical continuations.
 *
 * Every slot has one setup-time continuation frame. The broker service selects
 * one READY lease by service class, FIFO sequence, and explicit age promotion,
 * then resumes that slot's exact continuation identity. Asynchronous work
 * publishes its result into the owner record and calls resume(); the same
 * logical continuation advances on any eligible runtime worker. No broker
 * callback runs numerical work, blocks a worker, polls a predicate, or creates
 * a thread.
 *
 * Record is owner-defined state. It may contain pointer/stride views and a
 * model-specific route cursor; kcoro owns only its lifetime, correlation,
 * fairness, and continuation transitions.
 */
template <typename Record, std::size_t Capacity,
          std::size_t Isolation = 128>
class PermitBroker {
    static_assert(Capacity != 0);
    static_assert(Capacity <= UINT32_MAX);
    static_assert((Isolation & (Isolation - 1)) == 0);

    enum : std::uint32_t {
        state_claimed_ = 1,
        state_ready_ = 2,
        state_running_ = 3,
        state_done_ = 4,
    };

    enum : std::uint32_t {
        event_none_ = 0,
    };

    using Pool = SlotPool<Record, Capacity, Isolation>;

  public:
    struct Lease {
        std::uint32_t index = UINT32_MAX;
        std::uint64_t generation = 0;
        kc_ticket_id ticket{};

        [[nodiscard]] explicit operator bool() const noexcept {
            return index != UINT32_MAX && generation != 0 &&
                ticket_valid(ticket);
        }
    };

    struct Operations {
        void (*reset)(void *, Record &, std::uint32_t) noexcept = nullptr;
        PermitAdvance (*step)(void *, Lease, Record &,
                              PermitEvent) noexcept = nullptr;
        void (*finished)(void *, Lease, Record &, int) noexcept = nullptr;
    };

    struct Config {
        kc_runtime_t *runtime = nullptr;
        TicketSource *tickets = nullptr;
        std::uint32_t ticket_kind = 0;
        std::uint64_t age_promotion = 0;
        void *context = nullptr;
        Operations operations{};
        CallbackEdge capacity_ready{};
    };

    class Claim final {
      public:
        Claim() noexcept = default;
        Claim(const Claim &) = delete;
        Claim &operator=(const Claim &) = delete;

        Claim(Claim &&other) noexcept { move(other); }

        Claim &operator=(Claim &&other) noexcept {
            if (this == &other) return *this;
            abandon();
            move(other);
            return *this;
        }

        ~Claim() { abandon(); }

        [[nodiscard]] explicit operator bool() const noexcept {
            return owner_ != nullptr && static_cast<bool>(lease_);
        }

        [[nodiscard]] Record *record() const noexcept {
            return owner_ ? owner_->get(lease_) : nullptr;
        }

        [[nodiscard]] Lease lease() const noexcept { return lease_; }

        [[nodiscard]] int publish() noexcept {
            if (!owner_) return -ESTALE;
            PermitBroker *owner = owner_;
            owner_ = nullptr;
            return owner->publish_claim(lease_);
        }

        void abandon() noexcept {
            if (!owner_) return;
            PermitBroker *owner = owner_;
            owner_ = nullptr;
            owner->abandon_claim(lease_);
            lease_ = {};
        }

      private:
        friend class PermitBroker;

        void move(Claim &other) noexcept {
            owner_ = other.owner_;
            lease_ = other.lease_;
            other.owner_ = nullptr;
            other.lease_ = {};
        }

        PermitBroker *owner_ = nullptr;
        Lease lease_{};
    };

    PermitBroker() noexcept = default;
    PermitBroker(const PermitBroker &) = delete;
    PermitBroker &operator=(const PermitBroker &) = delete;

    [[nodiscard]] int initialize(const Config &config) noexcept {
        if (initialized_.load(std::memory_order_relaxed) ||
            !config.runtime || !config.tickets ||
            config.ticket_kind == 0 || config.age_promotion == 0 ||
            !config.context || !config.operations.reset ||
            !config.operations.step || !config.operations.finished ||
            !config.capacity_ready.valid()) {
            return -EINVAL;
        }
        config_ = config;
        admission_.bind(admission_edge, this);

        const kc_service_config service_config = {
            .callback = broker_step,
            .context = this,
        };
        int status =
            kc_service_create(config.runtime, &service_config, &service_);
        if (status != 0) return status;

        for (std::size_t index = 0; index < Capacity; ++index) {
            Entry &entry = entries_[index];
            const koro_cont_config continuation_config = {
                .step = permit_step,
                .argument = &entry,
                .frame_size = sizeof(Frame),
                .worker_mask = 0,
                .completion = permit_retired,
                .completion_context = &entry,
            };
            status = koro_cont_create_on(
                config.runtime, &continuation_config,
                &entry.continuation);
            if (status != 0) {
                for (std::size_t prior = 0; prior < index; ++prior) {
                    (void)koro_cont_destroy(
                        entries_[prior].continuation);
                    entries_[prior].continuation = nullptr;
                }
                (void)kc_service_destroy(service_);
                service_ = nullptr;
                return status;
            }
            entry.owner = this;
            entry.index = static_cast<std::uint32_t>(index);
            entry.identity =
                koro_cont_identity(entry.continuation);
            Frame *frame = static_cast<Frame *>(
                koro_cont_frame(entry.continuation));
            *frame = {
                .owner = this,
                .index = static_cast<std::uint32_t>(index),
            };
        }
        initialized_.store(true, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] int start() noexcept {
        if (!initialized_.load(std::memory_order_acquire))
            return -EINVAL;
        bool expected = false;
        if (!started_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return expected ? 0 : -EINVAL;
        }

        for (std::size_t index = 0; index < Capacity; ++index) {
            const int status =
                koro_cont_start(entries_[index].continuation);
            if (status != 0) {
                stopping_.store(true, std::memory_order_release);
                for (std::size_t started = 0; started < index; ++started)
                    signal(entries_[started], PermitEvent::Stop);
                return status;
            }
        }
        const int status = kc_service_start(service_);
        if (status != 0) {
            stopping_.store(true, std::memory_order_release);
            for (Entry &entry : entries_)
                signal(entry, PermitEvent::Stop);
            return status;
        }
        service_started_.store(true, std::memory_order_release);
        notify();
        return 0;
    }

    [[nodiscard]] int claim(ServiceClass service,
                            Claim *claim) noexcept {
        if (!claim || claim->owner_ ||
            !initialized_.load(std::memory_order_acquire) ||
            service < ServiceClass::Deadline ||
            service > ServiceClass::Background)
            return -EINVAL;
        const int admitted = admission_.enter();
        if (admitted != 0) return admitted;
        const typename Pool::Lease slot = pool_.acquire(
            state_claimed_,
            [this](Record &record, std::uint32_t index) noexcept {
                config_.operations.reset(
                    config_.context, record, index);
            });
        if (!slot) {
            admission_.leave();
            return -EBUSY;
        }
        Entry &entry = entries_[slot.index];
        entry.service.store(
            static_cast<std::uint32_t>(service),
            std::memory_order_relaxed);
        entry.sequence.store(0, std::memory_order_relaxed);
        entry.event.store(event_none_, std::memory_order_relaxed);
        entry.ticket =
            config_.tickets->mint(config_.ticket_kind);
        live_.fetch_add(1, std::memory_order_relaxed);
        claim->owner_ = this;
        claim->lease_ = {
            .index = slot.index,
            .generation = slot.generation,
            .ticket = entry.ticket,
        };
        return 0;
    }

    [[nodiscard]] int resume(Lease lease) noexcept {
        Entry *entry = checked_entry(lease, state_running_);
        if (!entry) return -ESTALE;
        std::uint32_t empty = event_none_;
        if (!entry->event.compare_exchange_strong(
                empty,
                static_cast<std::uint32_t>(
                    PermitEvent::Completion),
                std::memory_order_release,
                std::memory_order_acquire)) {
            return -EBUSY;
        }
        resumes_.fetch_add(1, std::memory_order_relaxed);
        return resume_exact(*entry);
    }

    void notify() noexcept {
        if (!service_started_.load(std::memory_order_acquire))
            return;
        const int status = kc_service_notify(service_);
        if (status != 0 && status != -ECANCELED) std::abort();
    }

    void defer() noexcept {
        deferrals_.fetch_add(1, std::memory_order_relaxed);
    }

    void request_stop() noexcept {
        stopping_.store(true, std::memory_order_release);
        admission_.stop();
        notify();
    }

    [[nodiscard]] int join() noexcept {
        if (!service_) return 0;
        const int status = kc_service_join(service_);
        if (status == 0)
            joined_.store(true, std::memory_order_release);
        return status;
    }

    [[nodiscard]] int destroy() noexcept {
        if (!initialized_.load(std::memory_order_acquire)) return 0;
        if (started_.load(std::memory_order_acquire) &&
            !joined_.load(std::memory_order_acquire)) {
            return -EBUSY;
        }
        const int service_status = kc_service_destroy(service_);
        if (service_status != 0) return service_status;
        service_ = nullptr;
        for (Entry &entry : entries_) {
            const int status =
                koro_cont_destroy(entry.continuation);
            if (status != 0) return status;
            entry.continuation = nullptr;
            entry.identity = {};
        }
        initialized_.store(false, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] Record *get(Lease lease) noexcept {
        const typename Pool::Lease slot = slot_lease(lease);
        if (!ticket_matches(lease)) return nullptr;
        return pool_.get(slot);
    }

    [[nodiscard]] const Record *get(Lease lease) const noexcept {
        const typename Pool::Lease slot = slot_lease(lease);
        if (!ticket_matches(lease)) return nullptr;
        return pool_.get(slot);
    }

    [[nodiscard]] int locate(const Record *record,
                             std::uint64_t generation,
                             const kc_ticket_id &ticket,
                             Lease *lease) const noexcept {
        if (!record || generation == 0 || !ticket_valid(ticket) ||
            !lease) return -EINVAL;
        for (std::size_t index = 0; index < Capacity; ++index) {
            if (&pool_.record(index) != record) continue;
            const Lease found = {
                .index = static_cast<std::uint32_t>(index),
                .generation = generation,
                .ticket = ticket,
            };
            if (!get(found)) return -ESTALE;
            *lease = found;
            return 0;
        }
        return -ESTALE;
    }

    [[nodiscard]] int release(Lease lease) noexcept {
        Entry *entry = checked_entry(lease, state_done_);
        if (!entry) return -ESTALE;
        const typename Pool::Lease slot = slot_lease(lease);
        if (!pool_.begin_release(slot, state_done_))
            return -ESTALE;
        entry->ticket = {};
        entry->sequence.store(0, std::memory_order_relaxed);
        entry->service.store(
            static_cast<std::uint32_t>(
                ServiceClass::Interactive),
            std::memory_order_relaxed);
        pool_.finish_release(slot);
        const std::uint32_t prior =
            live_.fetch_sub(1, std::memory_order_acq_rel);
        if (prior == 0 || prior > Capacity) std::abort();
        config_.capacity_ready.fire();
        notify();
        return 0;
    }

    [[nodiscard]] std::uint32_t state(Lease lease) const noexcept {
        if (!ticket_matches(lease)) return 0;
        return pool_.state(lease.index);
    }

    [[nodiscard]] bool done(Lease lease) const noexcept {
        return state(lease) == state_done_;
    }

    [[nodiscard]] kc_ticket_id continuation_identity(
        std::uint32_t index) const noexcept {
        return index < Capacity ? entries_[index].identity
                                : kc_ticket_id{};
    }

    [[nodiscard]] std::uint32_t current_worker(
        Lease lease) const noexcept {
        return ticket_matches(lease)
            ? koro_cont_current_worker(
                  entries_[lease.index].continuation)
            : UINT32_MAX;
    }

    [[nodiscard]] kc_service_t *service() const noexcept {
        return service_;
    }

    [[nodiscard]] bool stopped() const noexcept {
        return stopping_.load(std::memory_order_acquire);
    }

    [[nodiscard]] PermitBrokerSnapshot snapshot() const noexcept {
        PermitBrokerSnapshot snapshot = {
            .enqueue_sequence =
                sequence_.load(std::memory_order_acquire),
            .grants = grants_.load(std::memory_order_acquire),
            .resumes = resumes_.load(std::memory_order_acquire),
            .completions =
                completions_.load(std::memory_order_acquire),
            .deferrals =
                deferrals_.load(std::memory_order_acquire),
            .live = live_.load(std::memory_order_acquire),
            .stopped = stopping_.load(std::memory_order_acquire)
                ? 1u : 0u,
        };
        for (std::size_t index = 0; index < Capacity; ++index) {
            switch (pool_.state(index)) {
            case 0:
                snapshot.free++;
                break;
            case state_claimed_:
                snapshot.claimed++;
                break;
            case state_ready_:
                snapshot.ready++;
                break;
            case state_running_:
                snapshot.running++;
                break;
            case state_done_:
                snapshot.done++;
                break;
            default:
                std::abort();
            }
        }
        return snapshot;
    }

    [[nodiscard]] static constexpr std::uint64_t age(
        std::uint64_t snapshot,
        std::uint64_t enqueued) noexcept {
        return snapshot >= enqueued ? snapshot - enqueued : 0;
    }

    [[nodiscard]] static constexpr ServiceClass classify(
        std::uint64_t snapshot, std::uint64_t enqueued,
        ServiceClass service,
        std::uint64_t promotion) noexcept {
        return age(snapshot, enqueued) >= promotion
            ? ServiceClass::Deadline
            : service;
    }

  private:
    struct Entry {
        PermitBroker *owner = nullptr;
        koro_cont_t *continuation = nullptr;
        kc_ticket_id identity{};
        kc_ticket_id ticket{};
        std::uint32_t index = UINT32_MAX;
        /*
         * Retirement is durable state, not a one-shot event. A Stop callback
         * can land after this continuation drains `event` but before it calls
         * koro_cont_finish(); in that ordering the runtime correctly preserves
         * the wake and re-enters the frame. The next invocation must still see
         * the terminal predicate after the event record has been consumed.
         */
        std::atomic<bool> stop_requested{false};
        std::atomic<std::uint64_t> sequence{0};
        std::atomic<std::uint32_t> service{
            static_cast<std::uint32_t>(
                ServiceClass::Interactive)};
        std::atomic<std::uint32_t> event{event_none_};
    };

    struct Frame {
        PermitBroker *owner = nullptr;
        std::uint32_t index = UINT32_MAX;
        std::uint32_t event = event_none_;
        std::uint64_t generation = 0;
    };

    enum class Step : std::uint32_t {
        Again = 1,
        Suspend = 2,
        Finish = 3,
    };

    [[nodiscard]] static typename Pool::Lease slot_lease(
        Lease lease) noexcept {
        return {
            .index = lease.index,
            .generation = lease.generation,
        };
    }

    [[nodiscard]] bool ticket_matches(
        Lease lease) const noexcept {
        if (!lease || lease.index >= Capacity ||
            !pool_.matches(slot_lease(lease))) {
            return false;
        }
        return ticket_equal(entries_[lease.index].ticket,
                            lease.ticket);
    }

    [[nodiscard]] Entry *checked_entry(
        Lease lease, std::uint32_t expected_state) noexcept {
        if (!ticket_matches(lease) ||
            pool_.state(lease.index) != expected_state) {
            return nullptr;
        }
        return &entries_[lease.index];
    }

    [[nodiscard]] Lease current_lease(
        const Entry &entry) const noexcept {
        return {
            .index = entry.index,
            .generation = pool_.generation(entry.index),
            .ticket = entry.ticket,
        };
    }

    static void admission_edge(void *context) noexcept {
        static_cast<PermitBroker *>(context)->notify();
    }

    static void broker_step(void *context) {
        static_cast<PermitBroker *>(context)->dispatch_once();
    }

    static void *permit_step(koro_cont_t *continuation) noexcept {
        Entry *entry = static_cast<Entry *>(
            koro_cont_argument(continuation));
        if (!entry || !entry->owner) std::abort();
        Frame *frame = static_cast<Frame *>(
            koro_cont_frame(continuation));
        KORO_BEGIN(continuation);
        for (;;) {
            frame->event = entry->event.exchange(
                event_none_, std::memory_order_acq_rel);
            if (entry->stop_requested.load(
                    std::memory_order_acquire)) {
                break;
            }
            if (frame->event == event_none_) {
                KORO_SUSPEND(continuation);
                continue;
            }
            if (frame->event ==
                static_cast<std::uint32_t>(
                    PermitEvent::Stop)) {
                break;
            }
            frame->generation =
                entry->owner->pool_.generation(entry->index);
            if (frame->generation == 0) std::abort();
            entry->owner->run_permit(
                *entry, static_cast<PermitEvent>(frame->event));
        }
        KORO_END(continuation);
    }

    static void permit_retired(
        void *context, const kc_ticket_id *identity) noexcept {
        Entry *entry = static_cast<Entry *>(context);
        if (!entry || !entry->owner || !identity ||
            !ticket_equal(entry->identity, *identity)) {
            std::abort();
        }
        entry->owner->retired_continuations_.fetch_add(
            1, std::memory_order_acq_rel);
        entry->owner->notify();
    }

    [[nodiscard]] int resume_exact(Entry &entry) noexcept {
        const int status =
            koro_cont_resume(entry.continuation,
                             &entry.identity);
        if (status != 0 && status != -ECANCELED)
            std::abort();
        return status;
    }

    void signal(Entry &entry, PermitEvent event) noexcept {
        if (event == PermitEvent::Stop) {
            entry.stop_requested.store(
                true, std::memory_order_release);
        }
        std::uint32_t empty = event_none_;
        if (!entry.event.compare_exchange_strong(
                empty, static_cast<std::uint32_t>(event),
                std::memory_order_release,
                std::memory_order_acquire)) {
            std::abort();
        }
        (void)resume_exact(entry);
    }

    [[nodiscard]] std::uint64_t next_sequence() noexcept {
        std::uint64_t sequence =
            sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (sequence != 0) return sequence;
        sequence =
            sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (sequence == 0) std::abort();
        return sequence;
    }

    [[nodiscard]] ServiceClass effective(
        const Entry &entry, std::uint64_t snapshot) const noexcept {
        const ServiceClass service =
            static_cast<ServiceClass>(
                entry.service.load(std::memory_order_relaxed));
        return classify(
            snapshot,
            entry.sequence.load(std::memory_order_relaxed),
            service, config_.age_promotion);
    }

    [[nodiscard]] Entry *select() noexcept {
        Entry *best = nullptr;
        ServiceClass best_class = ServiceClass::Background;
        std::uint64_t best_sequence = UINT64_MAX;
        const std::uint64_t snapshot =
            sequence_.load(std::memory_order_acquire);
        for (Entry &entry : entries_) {
            if (pool_.state(entry.index) != state_ready_)
                continue;
            const ServiceClass service =
                effective(entry, snapshot);
            const std::uint64_t sequence =
                entry.sequence.load(std::memory_order_relaxed);
            if (!best || service < best_class ||
                (service == best_class &&
                 sequence < best_sequence)) {
                best = &entry;
                best_class = service;
                best_sequence = sequence;
            }
        }
        if (!best) return nullptr;
        const Lease lease = current_lease(*best);
        if (!pool_.transition(
                slot_lease(lease), state_ready_,
                state_running_)) {
            return nullptr;
        }
        return best;
    }

    void dispatch_once() noexcept {
        if (stopping_.load(std::memory_order_acquire)) {
            if (admission_.active() != 0) return;
            for (Entry &entry : entries_) {
                if (pool_.state(entry.index) != state_ready_)
                    continue;
                const Lease lease = current_lease(entry);
                if (!pool_.transition(
                        slot_lease(lease), state_ready_,
                        state_running_)) {
                    return;
                }
                signal(entry, PermitEvent::Cancel);
                return;
            }
            if (live_.load(std::memory_order_acquire) != 0)
                return;
            bool expected = false;
            if (retirement_started_.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                for (Entry &entry : entries_)
                    signal(entry, PermitEvent::Stop);
                return;
            }
            if (retired_continuations_.load(
                    std::memory_order_acquire) != Capacity) {
                return;
            }
            const int status =
                kc_service_complete_current(service_);
            if (status != 0 && status != -ECANCELED)
                std::abort();
            return;
        }

        Entry *entry = select();
        if (!entry) return;
        grants_.fetch_add(1, std::memory_order_relaxed);
        signal(*entry, PermitEvent::Grant);
    }

    void run_permit(Entry &entry, PermitEvent event) noexcept {
        const Lease lease = current_lease(entry);
        Record *record = get(lease);
        if (!record ||
            pool_.state(entry.index) != state_running_) {
            std::abort();
        }
        const PermitAdvance advance =
            config_.operations.step(
                config_.context, lease, *record, event);
        if (advance.disposition ==
            PermitDisposition::Suspend) {
            if (advance.status != 0) std::abort();
            /*
             * The granted continuation has published its asynchronous work
             * and dehydrated. Republish the broker continuation exactly once
             * so another READY permit can use remaining downstream capacity.
             * This is a bounded successor edge, not a scan loop: a broker
             * activation with no READY permit simply returns dormant. The
             * route continuation is not the broker service continuation, so
             * it publishes through the service's realtime-safe edge.
             */
            notify();
            return;
        }
        if (advance.disposition ==
                PermitDisposition::Requeue ||
            advance.disposition ==
                PermitDisposition::Preserve) {
            if (advance.status != 0 ||
                !pool_.transition(
                    slot_lease(lease), state_running_,
                    state_ready_)) {
                std::abort();
            }
            if (advance.disposition ==
                PermitDisposition::Requeue) {
                entry.sequence.store(
                    next_sequence(), std::memory_order_release);
                notify();
            } else {
                deferrals_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            return;
        }
        if (advance.disposition !=
            PermitDisposition::Complete) {
            std::abort();
        }
        if (!pool_.transition(
                slot_lease(lease), state_running_,
                state_done_)) {
            std::abort();
        }
        completions_.fetch_add(1, std::memory_order_relaxed);
        config_.operations.finished(
            config_.context, lease, *record, advance.status);
        notify();
    }

    [[nodiscard]] int publish_claim(Lease lease) noexcept {
        Entry *entry = checked_entry(lease, state_claimed_);
        if (!entry) std::abort();
        entry->sequence.store(
            next_sequence(), std::memory_order_release);
        if (!pool_.transition(
                slot_lease(lease), state_claimed_,
                state_ready_)) {
            std::abort();
        }
        admission_.leave();
        notify();
        return 0;
    }

    void abandon_claim(Lease lease) noexcept {
        Entry *entry = checked_entry(lease, state_claimed_);
        if (!entry) std::abort();
        const typename Pool::Lease slot = slot_lease(lease);
        if (!pool_.begin_release(slot, state_claimed_))
            std::abort();
        entry->ticket = {};
        pool_.finish_release(slot);
        const std::uint32_t prior =
            live_.fetch_sub(1, std::memory_order_acq_rel);
        if (prior == 0 || prior > Capacity) std::abort();
        admission_.leave();
        config_.capacity_ready.fire();
        notify();
    }

    Config config_{};
    AdmissionGate<static_cast<std::uint32_t>(Capacity)> admission_{};
    Pool pool_{};
    std::array<Entry, Capacity> entries_{};
    kc_service_t *service_ = nullptr;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> service_started_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> retirement_started_{false};
    std::atomic<bool> joined_{false};
    std::atomic<std::uint32_t> retired_continuations_{0};
    std::atomic<std::uint32_t> live_{0};
    std::atomic<std::uint64_t> sequence_{0};
    std::atomic<std::uint64_t> grants_{0};
    std::atomic<std::uint64_t> resumes_{0};
    std::atomic<std::uint64_t> completions_{0};
    std::atomic<std::uint64_t> deferrals_{0};
};

} // namespace kc
