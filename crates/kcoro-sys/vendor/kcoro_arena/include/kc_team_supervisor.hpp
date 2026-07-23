// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_coordination.hpp"
#include "kc_deadline.h"
#include "kc_fatal_store.hpp"
#include "kc_port.h"
#include "kc_service.h"
#include "kc_team.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace kc {

enum class TeamTerminalState : std::uint32_t {
    Idle = 0,
    Active = 1,
    Completed = 2,
    TimedOut = 3,
};

enum class TeamCompletion : std::uint32_t {
    Continue = 1,
    ExpiryWon = 2,
};

/*
 * The one linearization point shared by normal quorum return and hard expiry.
 * A generation can publish exactly one terminal state.
 */
class TeamTerminal final {
  public:
    static constexpr std::uint32_t bits = 2;
    static constexpr std::uint64_t mask =
        (UINT64_C(1) << bits) - 1;
    static constexpr std::uint64_t max_generation =
        UINT64_MAX >> bits;

    [[nodiscard]] int begin(std::uint64_t generation) noexcept {
        if (generation == 0 || generation > max_generation)
            return -EINVAL;
        std::uint64_t current =
            word_.load(std::memory_order_acquire);
        const std::uint64_t prior = generation_of(current);
        const TeamTerminalState state = state_of(current);
        if (!((prior == 0 && state == TeamTerminalState::Idle) ||
              (prior < generation &&
               state == TeamTerminalState::Completed))) {
            return -ESTALE;
        }
        return word_.compare_exchange_strong(
                   current,
                   encode(generation, TeamTerminalState::Active),
                   std::memory_order_acq_rel,
                   std::memory_order_acquire)
            ? 0
            : -EAGAIN;
    }

    [[nodiscard]] bool claim(
        std::uint64_t generation,
        TeamTerminalState winner) noexcept {
        if (generation == 0 || generation > max_generation ||
            (winner != TeamTerminalState::Completed &&
             winner != TeamTerminalState::TimedOut)) {
            return false;
        }
        std::uint64_t expected =
            encode(generation, TeamTerminalState::Active);
        return word_.compare_exchange_strong(
            expected, encode(generation, winner),
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    [[nodiscard]] int settle(
        std::uint64_t generation, int retired,
        TeamCompletion *completion) noexcept {
        if (!completion || generation == 0 ||
            generation > max_generation) {
            return -EINVAL;
        }
        if (retired == KC_DEADLINE_RETIRE_RETIRED) {
            if (!claim(generation, TeamTerminalState::Completed))
                return -EFAULT;
            *completion = TeamCompletion::Continue;
            return 0;
        }
        if (retired != KC_DEADLINE_RETIRE_EXPIRY_WON)
            return -EINVAL;
        const std::uint64_t current =
            word_.load(std::memory_order_acquire);
        if (generation_of(current) != generation)
            return -EFAULT;
        const TeamTerminalState state = state_of(current);
        if (state != TeamTerminalState::Active &&
            state != TeamTerminalState::TimedOut) {
            return -EFAULT;
        }
        *completion = TeamCompletion::ExpiryWon;
        return 0;
    }

    [[nodiscard]] std::uint64_t word() const noexcept {
        return word_.load(std::memory_order_acquire);
    }

    [[nodiscard]] static constexpr std::uint64_t encode(
        std::uint64_t generation,
        TeamTerminalState state) noexcept {
        return (generation << bits) |
            static_cast<std::uint32_t>(state);
    }

    [[nodiscard]] static constexpr std::uint64_t generation_of(
        std::uint64_t word) noexcept {
        return word >> bits;
    }

    [[nodiscard]] static constexpr TeamTerminalState state_of(
        std::uint64_t word) noexcept {
        return static_cast<TeamTerminalState>(word & mask);
    }

  private:
    std::atomic<std::uint64_t> word_{0};
};

struct TeamSupervisionIdentity {
    kc_ticket_id parent{};
    std::uint64_t scope_generation = 0;
    std::uint64_t epoch = 0;
    std::uint64_t domain = 0;
};

struct TeamSupervisorSnapshot {
    std::uint64_t terminal = 0;
    std::uint64_t generation = 0;
    std::uint64_t armed_ns = 0;
    std::uint64_t budget_ns = 0;
    std::uint64_t arm_generation = 0;
    std::uint64_t arm_team_generation = 0;
    std::uint32_t deadline_slot = 0;
    std::uint32_t started = 0;
    std::uint32_t stopping = 0;
    std::uint32_t joined = 0;
};

/*
 * Hard supervision for one fixed team.
 *
 * begin() copies immutable request/work evidence and arms one correlated
 * monotonic deadline child before the numerical generation is dispatched.
 * complete() retires the exact arm and races expiry through TeamTerminal.
 * Expiry runs on a retained service, captures the generation-stamped quorum,
 * publishes one prefaulted fatal record, and aborts. There is no retry,
 * polling, worker wait, soft replay, or scratch reclamation after timeout.
 */
template <typename Request, typename Work, typename Fatal,
          std::size_t Isolation = 128>
class TeamSupervisor final {
    static_assert(std::is_trivially_copyable_v<Request>);
    static_assert(std::is_trivially_copyable_v<Work>);
    static_assert(std::is_trivially_copyable_v<Fatal>);

  public:
    using Store = FatalStore<Fatal, Isolation>;

    struct Operations {
        void (*fill_fatal)(
            void *, Fatal &, const Request &, const Work &,
            const kc_deadline_event &, std::uint64_t,
            std::uint64_t, std::uint64_t, int,
            const kc_team_quorum_snapshot &) noexcept = nullptr;
        void (*fatal_published)(void *, const Fatal &) noexcept = nullptr;
    };

    struct Config {
        kc_runtime_t *runtime = nullptr;
        kc_team_t *team = nullptr;
        TicketSource *tickets = nullptr;
        std::uint32_t ticket_kind = 0;
        std::uint32_t deadline_slot = 0;
        std::uint64_t expected_mask = 0;
        std::uint64_t fatal_magic = 0;
        const char *fatal_path = nullptr;
        bool manual_deadlines = false;
        void *context = nullptr;
        Operations operations{};
    };

    TeamSupervisor() noexcept = default;
    TeamSupervisor(const TeamSupervisor &) = delete;
    TeamSupervisor &operator=(const TeamSupervisor &) = delete;

    [[nodiscard]] int initialize(const Config &config) noexcept {
        if (initialized_.load(std::memory_order_relaxed) ||
            !config.runtime || !config.team || !config.tickets ||
            config.ticket_kind == 0 || config.deadline_slot >= 64 ||
            config.expected_mask == 0 || config.fatal_magic == 0 ||
            !config.context || !config.operations.fill_fatal) {
            return -EINVAL;
        }
        config_ = config;
        int status = store_.initialize({
            .magic = config.fatal_magic,
            .runtime_epoch = config.tickets->epoch(),
            .path = config.fatal_path,
        });
        if (status != 0) return status;

        const kc_service_config service_config = {
            .callback = service_step,
            .context = this,
        };
        status = kc_service_create(
            config.runtime, &service_config, &service_);
        if (status != 0) {
            store_.destroy();
            return status;
        }
        status = kc_service_notifier_create(service_, &notifier_);
        if (status != 0) {
            (void)kc_service_destroy(service_);
            service_ = nullptr;
            store_.destroy();
            return status;
        }
        const kc_deadline_source_config source_config = {
            .capacity = config.deadline_slot + 1,
            .notify = deadline_edge,
            .context = notifier_,
        };
        status = config.manual_deadlines
            ? kc_deadline_source_create_manual_test(
                  &source_config, &source_)
            : kc_deadline_source_create(&source_config, &source_);
        if (status != 0) {
            (void)kc_service_notifier_destroy(notifier_);
            notifier_ = nullptr;
            (void)kc_service_destroy(service_);
            service_ = nullptr;
            store_.destroy();
            return status;
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
        const int status = kc_service_start(service_);
        if (status != 0)
            started_.store(false, std::memory_order_release);
        return status;
    }

    [[nodiscard]] int begin(
        const Request &request, const Work &work,
        TeamSupervisionIdentity identity,
        std::uint64_t generation,
        std::uint64_t budget_ns) noexcept {
        if (!started_.load(std::memory_order_acquire))
            return -ECANCELED;
        if (stopping_.load(std::memory_order_acquire))
            return -ECANCELED;
        if (budget_ns == 0) return -ENODATA;
        if (!ticket_valid(identity.parent) ||
            identity.scope_generation == 0 ||
            identity.epoch == 0 || identity.domain == 0) {
            return -EINVAL;
        }
        const int terminal_status = terminal_.begin(generation);
        if (terminal_status != 0) return terminal_status;

        request_ = request;
        work_ = work;
        identity_ = identity;
        child_ = config_.tickets->mint(config_.ticket_kind);
        arm_generation_.store(0, std::memory_order_relaxed);
        arm_team_generation_.store(0, std::memory_order_relaxed);
        armed_ns_.store(
            kc_port_monotonic_ns(), std::memory_order_relaxed);
        budget_ns_.store(budget_ns, std::memory_order_relaxed);
        active_generation_.store(
            generation, std::memory_order_release);

        const kc_deadline_arm_config arm_config = {
            .slot = config_.deadline_slot,
            .delay_ns = budget_ns,
            .child = child_,
            .parent = identity.parent,
            .scope_generation = identity.scope_generation,
            .epoch = identity.epoch,
            .domain = identity.domain,
            .team_generation = generation,
        };
        kc_deadline_arm arm{};
        const int status =
            kc_deadline_source_arm(source_, &arm_config, &arm);
        if (status == 0) {
            arm_generation_.store(
                arm.arm_generation, std::memory_order_release);
            arm_team_generation_.store(
                arm.team_generation, std::memory_order_release);
            return 0;
        }
        if (!terminal_.claim(
                generation, TeamTerminalState::Completed)) {
            std::abort();
        }
        active_generation_.store(0, std::memory_order_release);
        return status;
    }

    [[nodiscard]] int complete(
        std::uint64_t generation,
        TeamCompletion *completion) noexcept {
        if (!completion || generation == 0 ||
            active_generation_.load(std::memory_order_acquire) !=
                generation ||
            arm_team_generation_.load(std::memory_order_acquire) !=
                generation ||
            arm_generation_.load(std::memory_order_acquire) == 0) {
            return -ESTALE;
        }
        const int retired = kc_deadline_source_retire(
            source_, config_.deadline_slot,
            arm_generation_.load(std::memory_order_acquire));
        return terminal_.settle(generation, retired, completion);
    }

    void request_stop() noexcept {
        stopping_.store(true, std::memory_order_release);
        if (source_) {
            kc_deadline_source_request_stop(source_);
            return;
        }
        if (service_) kc_service_request_stop(service_);
    }

    [[nodiscard]] int join() noexcept {
        if (!initialized_.load(std::memory_order_acquire))
            return 0;
        if (!stopping_.load(std::memory_order_acquire))
            return -EBUSY;
        int status = kc_deadline_source_join(source_);
        if (status == 0) status = kc_service_join(service_);
        if (status == 0)
            joined_.store(true, std::memory_order_release);
        return status;
    }

    [[nodiscard]] int destroy() noexcept {
        if (!initialized_.load(std::memory_order_acquire))
            return 0;
        if (!joined_.load(std::memory_order_acquire))
            return -EBUSY;
        int status = kc_deadline_source_destroy(source_);
        if (status != 0) return status;
        source_ = nullptr;
        status = kc_service_notifier_destroy(notifier_);
        if (status != 0) return status;
        notifier_ = nullptr;
        status = kc_service_destroy(service_);
        if (status != 0) return status;
        service_ = nullptr;
        store_.destroy();
        initialized_.store(false, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] int expire_manual_test(
        std::uint64_t generation) noexcept {
        if (!config_.manual_deadlines ||
            active_generation_.load(std::memory_order_acquire) !=
                generation) {
            return -EINVAL;
        }
        const std::uint64_t budget =
            budget_ns_.load(std::memory_order_acquire);
        const std::uint64_t now = kc_port_monotonic_ns();
        armed_ns_.store(
            now >= budget ? now - budget : 0,
            std::memory_order_release);
        int status =
            kc_deadline_source_advance_manual_test(source_, budget);
        if (status == 0) {
            status = kc_deadline_source_fire_manual_test(
                source_, config_.deadline_slot);
        }
        return status;
    }

    [[nodiscard]] TeamSupervisorSnapshot snapshot() const noexcept {
        return {
            .terminal = terminal_.word(),
            .generation =
                active_generation_.load(std::memory_order_acquire),
            .armed_ns =
                armed_ns_.load(std::memory_order_acquire),
            .budget_ns =
                budget_ns_.load(std::memory_order_acquire),
            .arm_generation =
                arm_generation_.load(std::memory_order_acquire),
            .arm_team_generation =
                arm_team_generation_.load(std::memory_order_acquire),
            .deadline_slot = config_.deadline_slot,
            .started = started_.load(std::memory_order_acquire)
                ? 1u : 0u,
            .stopping = stopping_.load(std::memory_order_acquire)
                ? 1u : 0u,
            .joined = joined_.load(std::memory_order_acquire)
                ? 1u : 0u,
        };
    }

    [[nodiscard]] kc_service_t *service() const noexcept {
        return service_;
    }

    [[nodiscard]] const Fatal &fatal_record() const noexcept {
        return fatal_;
    }

  private:
    static void deadline_edge(void *context) noexcept {
        kc_service_notifier_t *notifier =
            static_cast<kc_service_notifier_t *>(context);
        const int status = kc_service_notifier_notify(notifier);
        if (status != 0 && status != -ECANCELED)
            std::abort();
    }

    static void service_step(void *context) {
        TeamSupervisor *owner =
            static_cast<TeamSupervisor *>(context);
        if (!owner) std::abort();
        owner->consume_deadline();
    }

    [[nodiscard]] bool matches(
        const kc_deadline_event &event) const noexcept {
        const std::uint64_t generation =
            active_generation_.load(std::memory_order_acquire);
        return generation != 0 &&
            budget_ns_.load(std::memory_order_acquire) != 0 &&
            event.kind == KC_DEADLINE_EVENT_EXPIRED &&
            event.slot == config_.deadline_slot &&
            event.scheduled_arm_generation != 0 &&
            event.current_arm_generation ==
                event.scheduled_arm_generation &&
            event.team_generation == generation &&
            event.scope_generation == identity_.scope_generation &&
            event.epoch == identity_.epoch &&
            event.domain == identity_.domain &&
            ticket_equal(event.child, child_) &&
            ticket_equal(event.parent, identity_.parent);
    }

    [[noreturn]] void publish_fatal(
        const kc_deadline_event &event) noexcept {
        if (!terminal_.claim(
                event.team_generation,
                TeamTerminalState::TimedOut)) {
            std::abort();
        }
        kc_team_quorum_snapshot quorum = {
            .generation = event.team_generation,
            .expected_mask = config_.expected_mask,
            .entered_mask = 0,
            .returned_mask = 0,
        };
        const int quorum_status = kc_team_quorum_snapshot_get(
            config_.team, event.team_generation, &quorum);
        if (quorum_status != 0)
            quorum.expected_mask = config_.expected_mask;
        const std::uint64_t armed =
            armed_ns_.load(std::memory_order_acquire);
        const std::uint64_t budget =
            budget_ns_.load(std::memory_order_acquire);
        const std::uint64_t now = kc_port_monotonic_ns();
        const std::uint64_t elapsed =
            now >= armed ? now - armed : 0;
        config_.operations.fill_fatal(
            config_.context, fatal_, request_, work_, event,
            armed, budget, elapsed, quorum_status, quorum);
        store_.publish(fatal_);
        if (config_.operations.fatal_published) {
            config_.operations.fatal_published(
                config_.context, fatal_);
        }
        std::abort();
    }

    void consume_deadline() noexcept {
        kc_deadline_event event{};
        const int status = kc_deadline_source_event_get(
            source_, config_.deadline_slot, &event);
        if (status == 0) {
            if (!matches(event)) std::abort();
            publish_fatal(event);
        }
        if (status != -EAGAIN) std::abort();

        kc_deadline_source_snapshot snapshot{};
        if (kc_deadline_source_snapshot_get(
                source_, &snapshot) != 0) {
            std::abort();
        }
        if (snapshot.phase == KC_DEADLINE_SOURCE_STOPPED &&
            snapshot.pending_events == 0) {
            const int complete =
                kc_service_complete_current(service_);
            if (complete != 0 && complete != -ECANCELED)
                std::abort();
        }
    }

    Config config_{};
    Store store_{};
    TeamTerminal terminal_{};
    Request request_{};
    Work work_{};
    TeamSupervisionIdentity identity_{};
    Fatal fatal_{};
    kc_deadline_source_t *source_ = nullptr;
    kc_service_t *service_ = nullptr;
    kc_service_notifier_t *notifier_ = nullptr;
    kc_ticket_id child_{};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> joined_{false};
    std::atomic<std::uint64_t> active_generation_{0};
    std::atomic<std::uint64_t> armed_ns_{0};
    std::atomic<std::uint64_t> budget_ns_{0};
    std::atomic<std::uint64_t> arm_generation_{0};
    std::atomic<std::uint64_t> arm_team_generation_{0};
};

} // namespace kc
