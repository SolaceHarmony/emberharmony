// SPDX-License-Identifier: BSD-3-Clause

#include "kcoro_arena.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "kcoro team contract failed: %s\n", message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

struct TeamState {
    kc_runtime_t *runtime = nullptr;
    kc_team_t *team = nullptr;
    std::atomic_uint member_mask{0};
    std::atomic_uint worker_mask{0};
    std::atomic_uint calls[4]{};
    std::atomic_uint completions{0};
    std::atomic_uint retirements{0};
    std::mutex mutex;
    std::condition_variable changed;
    bool retired = false;
};

void member_run(void *context, std::uint32_t member, std::uint32_t members,
                std::uint64_t generation) {
    auto *state = static_cast<TeamState *>(context);
    check(members == 4, "team changed its logical member count");
    check(member < members, "team published an invalid member identity");
    check(generation == 1, "member observed the wrong generation");
    std::uint32_t current_member = UINT32_MAX;
    std::uint32_t current_worker = UINT32_MAX;
    check(kc_team_current_member(state->team, &current_member) == 0,
          "team member identity is not active in its callback");
    check(kc_runtime_current_worker(state->runtime, &current_worker) == 0,
          "member callback is not running on a runtime worker");
    check(current_member == member, "logical member identity changed");
    check(current_worker < 2, "team escaped its bounded worker pool");
    state->member_mask.fetch_or(1u << member, std::memory_order_acq_rel);
    state->worker_mask.fetch_or(1u << current_worker,
                                std::memory_order_acq_rel);
    state->calls[member].fetch_add(1, std::memory_order_relaxed);
}

void generation_complete(void *context, std::uint64_t generation) {
    auto *state = static_cast<TeamState *>(context);
    check(generation == 1, "completion reported the wrong generation");
    check(state->completions.fetch_add(1, std::memory_order_acq_rel) == 0,
          "generation completion published more than once");
    kc_team_quorum_snapshot quorum{};
    check(kc_team_quorum_snapshot_get(state->team, generation,
                                      &quorum) == 0,
          "completed generation lost its quorum evidence");
    check(quorum.expected_mask == 0x0f &&
              quorum.entered_mask == quorum.expected_mask &&
              quorum.returned_mask == quorum.expected_mask,
          "completion published before the complete quorum returned");
    kc_team_request_stop(state->team);
}

void team_retired(void *context, std::uint64_t generation) {
    auto *state = static_cast<TeamState *>(context);
    check(generation == 1, "retirement reported the wrong generation");
    check(state->retirements.fetch_add(1, std::memory_order_acq_rel) == 0,
          "team retirement published more than once");
    {
        std::lock_guard lock(state->mutex);
        state->retired = true;
    }
    state->changed.notify_all();
}

void verify_logical_team_on_bounded_workers() {
    const kc_runtime_config runtime_config{.worker_count = 2};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&runtime_config, &runtime) == 0,
          "runtime creation failed");

    TeamState state;
    state.runtime = runtime;
    const kc_team_config team_config{
        .member_count = 4,
        .member = member_run,
        .context = &state,
        .runtime = runtime,
        .retired = team_retired,
        .retired_context = &state,
    };
    kc_team_t *team = nullptr;
    check(kc_team_create(&team_config, &team) == 0,
          "team creation failed");
    state.team = team;

    check(kc_runtime_start(runtime) == 0, "runtime start failed");
    check(kc_team_start(team) == 0, "team start failed");
    check(kc_team_dispatch_notify(team, 1, generation_complete,
                                  &state) == 0,
          "team dispatch failed");
    check(kc_runtime_join_all(runtime) == 0,
          "administrative team retirement observation failed");

    {
        std::unique_lock lock(state.mutex);
        check(state.changed.wait_for(lock, std::chrono::seconds(5),
                                     [&] { return state.retired; }),
              "team retirement edge did not arrive");
    }
    check(state.member_mask.load(std::memory_order_acquire) == 0x0f,
          "not every logical team member ran");
    check(state.worker_mask.load(std::memory_order_acquire) != 0 &&
              (state.worker_mask.load(std::memory_order_acquire) & ~0x03u) ==
                  0,
          "team used a worker outside the runtime");
    for (const std::atomic_uint &calls : state.calls)
        check(calls.load(std::memory_order_acquire) == 1,
              "a logical team member executed twice");
    check(state.completions.load(std::memory_order_acquire) == 1,
          "generation completion count is not exact");
    check(state.retirements.load(std::memory_order_acquire) == 1,
          "team retirement count is not exact");

    kc_runtime_snapshot runtime_snapshot{};
    kc_team_snapshot team_snapshot{};
    check(kc_runtime_snapshot_get(runtime, &runtime_snapshot) == 0,
          "runtime snapshot failed");
    check(kc_team_snapshot_get(team, &team_snapshot) == 0,
          "team snapshot failed");
    check(runtime_snapshot.workers == 2,
          "runtime worker count changed");
    check(team_snapshot.member_count == 4,
          "team member count changed");
    check(team_snapshot.completed_generation == 1 &&
              team_snapshot.joined == 1,
          "team terminal state is incomplete");

    check(kc_team_join(team) == 0,
          "terminal team acknowledgement failed");
    check(kc_team_destroy(team) == 0, "team destroy failed");
    kc_runtime_request_stop(runtime);
    check(kc_runtime_join(runtime) == 0, "runtime join failed");
    check(kc_runtime_destroy(runtime) == 0, "runtime destroy failed");
}

} // namespace

int main() {
    verify_logical_team_on_bounded_workers();
    return 0;
}
