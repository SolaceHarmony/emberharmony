// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_mailbox.hpp"
#include "kc_team.h"
#include "kcoro_stackless.h"

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace kc {

enum class TeamDisposition : std::uint32_t {
    Next = 1,
    Complete = 2,
    Fault = 3,
};

struct TeamAdvance {
    TeamDisposition disposition = TeamDisposition::Fault;
    int status = -EINVAL;
};

enum class TeamReturn : std::uint32_t {
    Continue = 1,
    Halt = 2,
};

enum class TeamExecutorPhase : std::uint32_t {
    Idle = 0,
    Generation = 1,
    Completion = 2,
    Halted = 3,
    Stopping = 4,
    Done = 5,
};

struct TeamExecutorSnapshot {
    MailboxSnapshot mailbox{};
    std::uint64_t activations = 0;
    std::uint64_t requests_started = 0;
    std::uint64_t requests_finished = 0;
    std::uint64_t generations_dispatched = 0;
    std::uint64_t generations_returned = 0;
    std::uint64_t generation = 0;
    std::uint64_t returned_generation = 0;
    std::uint64_t consumed_generation = 0;
    std::uint64_t retired_generation = 0;
    std::uint32_t phase = 0;
    std::uint32_t started = 0;
    std::uint32_t stopped = 0;
    std::uint32_t retired = 0;
};

/*
 * A fixed mailbox-to-team continuation.
 *
 * The caller supplies numerical operations; kcoro owns the execution protocol:
 * one retained orchestration frame, one fixed logical team, exact final-return
 * callbacks, request/completion publication, and asynchronous retirement.
 * Numerical member callbacks never suspend. No worker waits for a request,
 * generation, completion, capacity change, or stop: every successor is a
 * correlated callback edge that makes the retained continuation runnable.
 */
template <typename Request, typename Completion, std::size_t Capacity,
          std::size_t Isolation = 128>
class TeamExecutor final {
    static_assert(std::is_trivially_copyable_v<Request>);
    static_assert(std::is_trivially_copyable_v<Completion>);

  public:
    using MailboxType = Mailbox<Request, Completion, Capacity, Isolation>;
    using Retired = void (*)(void *, const kc_ticket_id *);

    struct Operations {
        int (*begin)(void *, const Request &) noexcept = nullptr;
        void (*run_member)(void *, std::uint32_t, std::uint32_t,
                           std::uint64_t) noexcept = nullptr;
        int (*before_generation)(void *, const Request &,
                                 std::uint64_t) noexcept = nullptr;
        void (*generation_returned)(void *, const Request &,
                                    std::uint64_t) noexcept = nullptr;
        TeamReturn (*accept_generation)(void *, const Request &,
                                        std::uint64_t) noexcept = nullptr;
        TeamAdvance (*advance)(void *, const Request &,
                               std::uint64_t) noexcept = nullptr;
        void (*make_completion)(void *, const Request &, int,
                                Completion *) noexcept = nullptr;
        void (*finish)(void *, const Request &,
                       const Completion &) noexcept = nullptr;
        void (*stopping)(void *) noexcept = nullptr;
    };

    struct Config {
        kc_runtime_t *runtime = nullptr;
        std::uint32_t member_count = 0;
        void *context = nullptr;
        Operations operations{};
        MailboxEdge capacity_ready{};
        Retired retired = nullptr;
        void *retired_context = nullptr;
    };

    TeamExecutor() noexcept = default;
    TeamExecutor(const TeamExecutor &) = delete;
    TeamExecutor &operator=(const TeamExecutor &) = delete;

    [[nodiscard]] int initialize(const Config &config) noexcept {
        if (initialized_.load(std::memory_order_relaxed) ||
            !config.runtime || config.member_count == 0 ||
            !config.context || !config.operations.begin ||
            !config.operations.run_member || !config.operations.advance ||
            !config.operations.make_completion ||
            !config.operations.finish ||
            !config.capacity_ready.valid()) {
            return -EINVAL;
        }
        config_ = config;
        const koro_cont_config continuation_config = {
            .step = continuation_step,
            .argument = this,
            .frame_size = sizeof(Frame),
            .worker_mask = 0,
            .completion = continuation_retired,
            .completion_context = this,
        };
        int status = koro_cont_create_on(
            config.runtime, &continuation_config, &continuation_);
        if (status != 0) return status;
        identity_ = koro_cont_identity(continuation_);
        Frame *frame = static_cast<Frame *>(
            koro_cont_frame(continuation_));
        *frame = {
            .owner = this,
        };

        const kc_team_config team_config = {
            .member_count = config.member_count,
            .member = team_member,
            .context = this,
            .runtime = config.runtime,
            .retired = team_retired,
            .retired_context = this,
        };
        status = kc_team_create(&team_config, &team_);
        if (status != 0) {
            (void)koro_cont_destroy(continuation_);
            continuation_ = nullptr;
            identity_ = {};
            return status;
        }
        const typename MailboxType::Config mailbox_config = {
            .request_ready = {
                continuation_edge, this, identity_},
            .completion_ready = {
                continuation_edge, this, identity_},
            .capacity_ready = config.capacity_ready,
        };
        status = mailbox_.bind(mailbox_config);
        if (status == 0) status = mailbox_.open(&endpoints_);
        if (status != 0) {
            (void)kc_team_destroy(team_);
            team_ = nullptr;
            (void)koro_cont_destroy(continuation_);
            continuation_ = nullptr;
            identity_ = {};
            return status;
        }
        mailbox_open_.store(true, std::memory_order_release);
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
        int status = kc_team_start(team_);
        if (status == 0) {
            team_started_.store(true, std::memory_order_release);
            status = koro_cont_start(continuation_);
        }
        if (status == 0) {
            continuation_started_.store(true, std::memory_order_release);
            return 0;
        }
        mailbox_.stop();
        if (team_started_.load(std::memory_order_acquire))
            kc_team_request_stop(team_);
        return status;
    }

    [[nodiscard]] int submit(const Request &request) noexcept {
        if (!continuation_started_.load(std::memory_order_acquire))
            return -ECANCELED;
        return endpoints_.requests.publish(request);
    }

    void request_stop() noexcept {
        stopped_.store(true, std::memory_order_release);
        if (mailbox_open_.load(std::memory_order_acquire)) {
            mailbox_.stop();
            return;
        }
        if (team_) kc_team_request_stop(team_);
    }

    /*
     * Setup-lease destruction only. It never blocks. The retained retirement
     * callback must already have proved that the mailbox, orchestration frame,
     * and logical team have all reached terminal state.
     */
    [[nodiscard]] int destroy() noexcept {
        if (!initialized_.load(std::memory_order_acquire)) return 0;
        if (continuation_started_.load(std::memory_order_acquire) &&
            !retired_.load(std::memory_order_acquire)) {
            return -EBUSY;
        }
        if (!continuation_started_.load(std::memory_order_acquire) &&
            mailbox_open_.load(std::memory_order_acquire)) {
            mailbox_.stop();
            const int status = mailbox_.retire(&endpoints_);
            if (status != 0) return status;
            mailbox_open_.store(false, std::memory_order_release);
        }
        const int team_status = kc_team_destroy(team_);
        if (team_status != 0) return team_status;
        team_ = nullptr;
        const int continuation_status =
            koro_cont_destroy(continuation_);
        if (continuation_status != 0) return continuation_status;
        continuation_ = nullptr;
        identity_ = {};
        initialized_.store(false, std::memory_order_release);
        return 0;
    }

    [[nodiscard]] TeamExecutorSnapshot snapshot() const noexcept {
        return {
            .mailbox = mailbox_.snapshot(),
            .activations =
                activations_.load(std::memory_order_acquire),
            .requests_started =
                requests_started_.load(std::memory_order_acquire),
            .requests_finished =
                requests_finished_.load(std::memory_order_acquire),
            .generations_dispatched =
                generations_dispatched_.load(std::memory_order_acquire),
            .generations_returned =
                generations_returned_.load(std::memory_order_acquire),
            .generation =
                active_generation_.load(std::memory_order_acquire),
            .returned_generation =
                returned_generation_.load(std::memory_order_acquire),
            .consumed_generation =
                consumed_generation_.load(std::memory_order_acquire),
            .retired_generation =
                retired_generation_.load(std::memory_order_acquire),
            .phase = phase_.load(std::memory_order_acquire),
            .started =
                continuation_started_.load(std::memory_order_acquire)
                    ? 1u : 0u,
            .stopped = stopped_.load(std::memory_order_acquire)
                ? 1u : 0u,
            .retired = retired_.load(std::memory_order_acquire)
                ? 1u : 0u,
        };
    }

    [[nodiscard]] kc_team_t *team() const noexcept { return team_; }
    [[nodiscard]] koro_cont_t *continuation() const noexcept {
        return continuation_;
    }
    [[nodiscard]] kc_ticket_id identity() const noexcept {
        return identity_;
    }

  private:
    struct Frame {
        TeamExecutor *owner = nullptr;
        Request request{};
        std::uint64_t last_generation = 0;
        std::uint32_t decision = 0;
        bool active = false;
    };

    enum class Step : std::uint32_t {
        Again = 1,
        Suspend = 2,
        Finish = 3,
    };

    static void continuation_edge(
        void *context, const kc_ticket_id *identity) noexcept {
        TeamExecutor *owner =
            static_cast<TeamExecutor *>(context);
        if (!owner || !identity ||
            !ticket_equal(owner->identity_, *identity)) {
            std::abort();
        }
        const int status =
            koro_cont_resume(owner->continuation_, identity);
        if (status != 0 && status != -ECANCELED) std::abort();
    }

    static void team_member(void *context, std::uint32_t member,
                            std::uint32_t members,
                            std::uint64_t generation) noexcept {
        TeamExecutor *owner =
            static_cast<TeamExecutor *>(context);
        if (!owner || members != owner->config_.member_count ||
            generation != owner->active_generation_.load(
                std::memory_order_acquire)) {
            std::abort();
        }
        owner->config_.operations.run_member(
            owner->config_.context, member, members, generation);
    }

    static void team_generation_returned(
        void *context, std::uint64_t generation) noexcept {
        TeamExecutor *owner =
            static_cast<TeamExecutor *>(context);
        if (!owner ||
            generation != owner->active_generation_.load(
                std::memory_order_acquire)) {
            std::abort();
        }
        std::uint64_t empty = 0;
        if (!owner->returned_generation_.compare_exchange_strong(
                empty, generation, std::memory_order_release,
                std::memory_order_acquire)) {
            std::abort();
        }
        owner->generations_returned_.fetch_add(
            1, std::memory_order_relaxed);
        if (owner->config_.operations.generation_returned) {
            Frame *frame = static_cast<Frame *>(
                koro_cont_frame(owner->continuation_));
            owner->config_.operations.generation_returned(
                owner->config_.context, frame->request, generation);
        }
        continuation_edge(owner, &owner->identity_);
    }

    static void team_retired(void *context,
                             std::uint64_t generation) noexcept {
        TeamExecutor *owner =
            static_cast<TeamExecutor *>(context);
        if (!owner ||
            generation >
                owner->generation_sequence_.load(
                    std::memory_order_acquire)) {
            std::abort();
        }
        owner->team_retired_.store(true, std::memory_order_release);
        continuation_edge(owner, &owner->identity_);
    }

    static void continuation_retired(
        void *context, const kc_ticket_id *identity) noexcept {
        TeamExecutor *owner =
            static_cast<TeamExecutor *>(context);
        if (!owner || !identity ||
            !ticket_equal(owner->identity_, *identity)) {
            std::abort();
        }
        owner->retired_.store(true, std::memory_order_release);
        if (owner->config_.retired) {
            owner->config_.retired(
                owner->config_.retired_context, identity);
        }
    }

    static void *continuation_step(
        koro_cont_t *continuation) noexcept {
        Frame *frame =
            static_cast<Frame *>(koro_cont_frame(continuation));
        if (!frame || !frame->owner) std::abort();
        TeamExecutor *owner = frame->owner;
        KORO_BEGIN(continuation);
        for (;;) {
            owner->activations_.fetch_add(
                1, std::memory_order_relaxed);
            frame->decision = static_cast<std::uint32_t>(
                owner->step_once(*frame));
            if (frame->decision ==
                static_cast<std::uint32_t>(Step::Again)) continue;
            if (frame->decision ==
                static_cast<std::uint32_t>(Step::Finish)) break;
            KORO_SUSPEND(continuation);
        }
        KORO_END(continuation);
    }

    [[nodiscard]] int dispatch(Frame &frame) noexcept {
        std::uint64_t generation =
            generation_sequence_.fetch_add(
                1, std::memory_order_relaxed) + 1;
        if (generation == 0) {
            generation = generation_sequence_.fetch_add(
                1, std::memory_order_relaxed) + 1;
        }
        if (generation == 0) std::abort();
        frame.last_generation = generation;
        if (config_.operations.before_generation) {
            const int status =
                config_.operations.before_generation(
                    config_.context, frame.request, generation);
            if (status != 0) return status;
        }
        returned_generation_.store(0, std::memory_order_relaxed);
        active_generation_.store(generation,
                                 std::memory_order_release);
        phase_.store(
            static_cast<std::uint32_t>(
                TeamExecutorPhase::Generation),
            std::memory_order_release);
        generations_dispatched_.fetch_add(
            1, std::memory_order_relaxed);
        const int status = kc_team_dispatch_notify(
            team_, generation, team_generation_returned, this);
        if (status != 0) {
            active_generation_.store(0,
                                     std::memory_order_release);
        }
        return status;
    }

    void publish_completion(Frame &frame, int status) noexcept {
        Completion completion{};
        config_.operations.make_completion(
            config_.context, frame.request, status, &completion);
        if (!ticket_equal(completion.ticket,
                          frame.request.ticket)) {
            std::abort();
        }
        retired_generation_.store(
            frame.last_generation, std::memory_order_release);
        active_generation_.store(0, std::memory_order_release);
        phase_.store(
            static_cast<std::uint32_t>(
                TeamExecutorPhase::Completion),
            std::memory_order_release);
        if (endpoints_.completions.publish(completion) != 0)
            std::abort();
    }

    [[nodiscard]] Step begin_request(Frame &frame,
                                     const Request &request) noexcept {
        frame.request = request;
        frame.last_generation = 0;
        frame.active = true;
        requests_started_.fetch_add(1, std::memory_order_relaxed);
        const int status =
            config_.operations.begin(config_.context, frame.request);
        if (status != 0) {
            publish_completion(frame, status);
            return Step::Again;
        }
        const int dispatched = dispatch(frame);
        if (dispatched != 0) {
            publish_completion(frame, dispatched);
            return Step::Again;
        }
        return Step::Suspend;
    }

    [[nodiscard]] Step consume_generation(Frame &frame) noexcept {
        const std::uint64_t generation =
            active_generation_.load(std::memory_order_acquire);
        std::uint64_t returned =
            returned_generation_.load(std::memory_order_acquire);
        if (returned == 0) return Step::Suspend;
        if (generation == 0 || returned != generation ||
            !returned_generation_.compare_exchange_strong(
                returned, 0, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            std::abort();
        }
        consumed_generation_.store(generation,
                                   std::memory_order_release);
        if (config_.operations.accept_generation &&
            config_.operations.accept_generation(
                config_.context, frame.request, generation) ==
                TeamReturn::Halt) {
            phase_.store(
                static_cast<std::uint32_t>(
                    TeamExecutorPhase::Halted),
                std::memory_order_release);
            return Step::Suspend;
        }
        const TeamAdvance advance =
            config_.operations.advance(
                config_.context, frame.request, generation);
        if (advance.disposition == TeamDisposition::Next) {
            if (advance.status != 0) std::abort();
            const int status = dispatch(frame);
            if (status != 0) {
                publish_completion(frame, status);
                return Step::Again;
            }
            return Step::Suspend;
        }
        if (advance.disposition == TeamDisposition::Complete) {
            publish_completion(frame, advance.status);
            return Step::Again;
        }
        if (advance.disposition == TeamDisposition::Fault &&
            advance.status != 0) {
            publish_completion(frame, advance.status);
            return Step::Again;
        }
        std::abort();
    }

    [[nodiscard]] Step consume_completion(Frame &frame) noexcept {
        Completion completion{};
        const int status =
            endpoints_.completion_consumer.consume(&completion);
        if (status == -EAGAIN) return Step::Suspend;
        if (status != 0 ||
            !ticket_equal(completion.ticket,
                          frame.request.ticket)) {
            std::abort();
        }
        config_.operations.finish(
            config_.context, frame.request, completion);
        requests_finished_.fetch_add(1, std::memory_order_relaxed);
        frame.request = {};
        frame.last_generation = 0;
        frame.active = false;
        phase_.store(
            static_cast<std::uint32_t>(
                TeamExecutorPhase::Idle),
            std::memory_order_release);
        return Step::Again;
    }

    [[nodiscard]] Step begin_stop() noexcept {
        phase_.store(
            static_cast<std::uint32_t>(
                TeamExecutorPhase::Stopping),
            std::memory_order_release);
        if (config_.operations.stopping)
            config_.operations.stopping(config_.context);
        kc_team_request_stop(team_);
        return team_retired_.load(std::memory_order_acquire)
            ? finish_stop() : Step::Suspend;
    }

    [[nodiscard]] Step finish_stop() noexcept {
        if (!team_retired_.load(std::memory_order_acquire))
            return Step::Suspend;
        if (!mailbox_.settled()) std::abort();
        if (mailbox_.retire(&endpoints_) != 0) std::abort();
        mailbox_open_.store(false, std::memory_order_release);
        phase_.store(
            static_cast<std::uint32_t>(
                TeamExecutorPhase::Done),
            std::memory_order_release);
        return Step::Finish;
    }

    [[nodiscard]] Step step_once(Frame &frame) noexcept {
        const TeamExecutorPhase phase =
            static_cast<TeamExecutorPhase>(
                phase_.load(std::memory_order_acquire));
        if (phase == TeamExecutorPhase::Generation)
            return consume_generation(frame);
        if (phase == TeamExecutorPhase::Completion)
            return consume_completion(frame);
        if (phase == TeamExecutorPhase::Halted)
            return Step::Suspend;
        if (phase == TeamExecutorPhase::Stopping)
            return finish_stop();
        if (phase == TeamExecutorPhase::Done)
            return Step::Finish;
        if (phase != TeamExecutorPhase::Idle || frame.active)
            std::abort();

        Request request{};
        const int status =
            endpoints_.request_consumer.consume(&request);
        if (status == 0) return begin_request(frame, request);
        if (status == -EAGAIN) return Step::Suspend;
        if (status == -ECANCELED) return begin_stop();
        std::abort();
    }

    Config config_{};
    MailboxType mailbox_{};
    typename MailboxType::Endpoints endpoints_{};
    kc_team_t *team_ = nullptr;
    koro_cont_t *continuation_ = nullptr;
    kc_ticket_id identity_{};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> started_{false};
    std::atomic<bool> team_started_{false};
    std::atomic<bool> continuation_started_{false};
    std::atomic<bool> mailbox_open_{false};
    std::atomic<bool> stopped_{false};
    std::atomic<bool> team_retired_{false};
    std::atomic<bool> retired_{false};
    std::atomic<std::uint64_t> activations_{0};
    std::atomic<std::uint64_t> requests_started_{0};
    std::atomic<std::uint64_t> requests_finished_{0};
    std::atomic<std::uint64_t> generations_dispatched_{0};
    std::atomic<std::uint64_t> generations_returned_{0};
    std::atomic<std::uint64_t> generation_sequence_{0};
    std::atomic<std::uint64_t> active_generation_{0};
    std::atomic<std::uint64_t> returned_generation_{0};
    std::atomic<std::uint64_t> consumed_generation_{0};
    std::atomic<std::uint64_t> retired_generation_{0};
    std::atomic<std::uint32_t> phase_{
        static_cast<std::uint32_t>(
            TeamExecutorPhase::Idle)};
};

} // namespace kc
