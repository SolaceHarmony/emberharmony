// SPDX-License-Identifier: BSD-3-Clause

#include "kcoro_arena.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <dirent.h>
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define KCORO_TEST_THREAD_SANITIZER 1
#endif
#endif

extern "C" int kc_runtime_worker_cpu_ns_for_test(kc_runtime_t *runtime,
                                                  uint64_t *out_ns);

namespace {

using namespace std::chrono_literals;

constexpr std::uint64_t kMarker = UINT64_C(0x51a71e55c0def00d);
constexpr double kIdleLimitPercent = 0.5;

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "kcoro runtime contract failed: %s\n", message);
    std::abort();
}

void check(bool condition, const char *message) {
    if (!condition) fail(message);
}

class Signal {
  public:
    void publish(std::uint32_t value) {
        {
            std::lock_guard lock(mutex_);
            if (value > value_) value_ = value;
        }
        changed_.notify_all();
    }

    void await(std::uint32_t value, const char *message) {
        std::unique_lock lock(mutex_);
        if (!changed_.wait_for(lock, 5s,
                               [&] { return value_ >= value; })) {
            fail(message);
        }
    }

    std::uint32_t value() {
        std::lock_guard lock(mutex_);
        return value_;
    }

  private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::uint32_t value_ = 0;
};

std::size_t process_threads() {
#if defined(__APPLE__)
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t count = 0;
    check(task_threads(mach_task_self(), &threads, &count) == KERN_SUCCESS,
          "task_threads failed");
    check(vm_deallocate(mach_task_self(),
                        reinterpret_cast<vm_address_t>(threads),
                        count * sizeof(*threads)) == KERN_SUCCESS,
          "thread-list deallocation failed");
    return count;
#elif defined(__linux__)
    DIR *directory = opendir("/proc/self/task");
    check(directory != nullptr, "cannot open /proc/self/task");
    std::size_t count = 0;
    while (const dirent *entry = readdir(directory)) {
        if (entry->d_name[0] != '.') ++count;
    }
    closedir(directory);
    return count;
#else
    return 0;
#endif
}

void stop_runtime(kc_runtime_t *runtime) {
    kc_runtime_request_stop(runtime);
    check(kc_runtime_join(runtime) == 0, "runtime did not join");
    check(kc_runtime_destroy(runtime) == 0, "runtime did not destroy");
}

void *terminal_step(koro_cont_t *continuation) {
    return koro_cont_finish(continuation) ? reinterpret_cast<void *>(1)
                                         : nullptr;
}

void count_completion(void *context, const kc_ticket_id *identity) {
    auto *count = static_cast<std::atomic_uint *>(context);
    check(identity != nullptr, "terminal callback lost its identity");
    count->fetch_add(1, std::memory_order_release);
}

void verify_bounded_worker_pool() {
#if !defined(KCORO_TEST_THREAD_SANITIZER)
    const std::size_t baseline_threads = process_threads();
#endif
    const kc_runtime_config config{.worker_count = 3};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&config, &runtime) == 0,
          "runtime creation failed");
    check(kc_runtime_start(runtime) == 0, "runtime start failed");
    const std::size_t ready_threads = process_threads();
#if !defined(KCORO_TEST_THREAD_SANITIZER)
    if (baseline_threads != 0)
        check(ready_threads == baseline_threads + config.worker_count,
              "runtime did not create exactly its bounded OS-worker pool");
#endif

    std::atomic_uint completions{0};
    std::vector<koro_cont_t *> continuations;
    continuations.reserve(96);
    for (unsigned index = 0; index < 96; ++index) {
        const koro_cont_config continuation{
            .step = terminal_step,
            .argument = nullptr,
            .frame_size = 0,
            .worker_mask = 0,
            .completion = count_completion,
            .completion_context = &completions,
        };
        koro_cont_t *item = nullptr;
        check(koro_cont_create_on(runtime, &continuation, &item) == 0,
              "continuation creation failed");
        continuations.push_back(item);
    }
    for (koro_cont_t *item : continuations)
        check(koro_cont_start(item) == 0, "continuation start failed");
    check(kc_runtime_join_all(runtime) == 0,
          "administrative completion observation failed");
    check(completions.load(std::memory_order_acquire) == continuations.size(),
          "a logical continuation failed to complete");
    if (ready_threads != 0)
        check(process_threads() == ready_threads,
              "an operation created a physical thread after readiness");

    kc_runtime_snapshot snapshot{};
    check(kc_runtime_snapshot_get(runtime, &snapshot) == 0,
          "runtime snapshot failed");
    check(snapshot.workers == 3, "runtime changed its worker bound");
    check(snapshot.active == 0 && snapshot.queued == 0 &&
              snapshot.running == 0 && snapshot.dormant == 0,
          "completed continuations remained active");

    for (koro_cont_t *item : continuations)
        check(koro_cont_destroy(item) == 0,
              "completed continuation did not retire");
    stop_runtime(runtime);
}

struct MigratingFrame {
    std::uint64_t marker;
    std::uint32_t first;
    std::uint32_t second;
};

struct Migrating {
    Signal signal;
};

void migrate_complete(void *context, const kc_ticket_id *identity) {
    check(identity != nullptr, "migration callback lost its identity");
    static_cast<Migrating *>(context)->signal.publish(3);
}

void *migrate_step(koro_cont_t *continuation) {
    auto *context =
        static_cast<Migrating *>(koro_cont_argument(continuation));
    auto *frame =
        static_cast<MigratingFrame *>(koro_cont_frame(continuation));
    switch (koro_cont_state_get(continuation)) {
    case 0:
        frame->marker = kMarker;
        frame->first = koro_cont_current_worker(continuation);
        context->signal.publish(1);
        koro_cont_state_set(continuation, 1, KORO_SUSPEND_CALLBACK);
        return nullptr;
    case 1:
        check(frame->marker == kMarker,
              "logical frame did not survive dehydration");
        frame->second = koro_cont_current_worker(continuation);
        context->signal.publish(2);
        return koro_cont_finish(continuation)
            ? reinterpret_cast<void *>(1)
            : nullptr;
    default:
        fail("migration continuation resumed at an unknown position");
    }
}

struct Blocker {
    Signal signal;
    std::mutex mutex;
    std::condition_variable changed;
    bool release = false;
};

void blocker_complete(void *context, const kc_ticket_id *identity) {
    check(identity != nullptr, "blocker callback lost its identity");
    static_cast<Blocker *>(context)->signal.publish(2);
}

void *blocker_step(koro_cont_t *continuation) {
    auto *blocker =
        static_cast<Blocker *>(koro_cont_argument(continuation));
    blocker->signal.publish(1);
    std::unique_lock lock(blocker->mutex);
    blocker->changed.wait(lock, [&] { return blocker->release; });
    return koro_cont_finish(continuation) ? reinterpret_cast<void *>(1)
                                         : nullptr;
}

void verify_frame_migration() {
    const kc_runtime_config config{.worker_count = 2};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&config, &runtime) == 0,
          "migration runtime creation failed");
    check(kc_runtime_start(runtime) == 0, "migration runtime start failed");

    Migrating migrating;
    const koro_cont_config target_config{
        .step = migrate_step,
        .argument = &migrating,
        .frame_size = sizeof(MigratingFrame),
        .worker_mask = 0,
        .completion = migrate_complete,
        .completion_context = &migrating,
    };
    koro_cont_t *target = nullptr;
    check(koro_cont_create_on(runtime, &target_config, &target) == 0,
          "migration target creation failed");
    check(koro_cont_start(target) == 0, "migration target start failed");
    migrating.signal.await(1, "migration target did not dehydrate");
    const auto *frame =
        static_cast<const MigratingFrame *>(koro_cont_frame(target));
    check(frame->first < 2, "first worker identity is invalid");

    Blocker blocker;
    const koro_cont_config blocker_config{
        .step = blocker_step,
        .argument = &blocker,
        .frame_size = 0,
        .worker_mask = UINT64_C(1) << frame->first,
        .completion = blocker_complete,
        .completion_context = &blocker,
    };
    koro_cont_t *pinned = nullptr;
    check(koro_cont_create_on(runtime, &blocker_config, &pinned) == 0,
          "blocker creation failed");
    check(koro_cont_start(pinned) == 0, "blocker start failed");
    blocker.signal.await(1, "eligible worker did not enter blocker");

    kc_ticket_id stale = koro_cont_identity(target);
    ++stale.generation;
    if (stale.generation == 0) ++stale.generation;
    check(koro_cont_resume(target, &stale) == -ESTALE,
          "stale callback resumed the target");
    const kc_ticket_id identity = koro_cont_identity(target);
    check(koro_cont_resume(target, &identity) == 0,
          "exact callback did not resume the target");
    migrating.signal.await(3, "migrated continuation did not complete");
    check(frame->second < 2 && frame->second != frame->first,
          "eligible continuation remained pinned to its first worker");

    {
        std::lock_guard lock(blocker.mutex);
        blocker.release = true;
    }
    blocker.changed.notify_all();
    blocker.signal.await(2, "blocker did not retire");
    check(kc_runtime_join_all(runtime) == 0,
          "migration runtime did not become idle");
    check(koro_cont_destroy(pinned) == 0, "blocker destroy failed");
    check(koro_cont_destroy(target) == 0, "target destroy failed");
    stop_runtime(runtime);
}

struct Exact {
    Signal signal;
};

void exact_complete(void *context, const kc_ticket_id *identity) {
    check(identity != nullptr, "exact callback lost its identity");
    static_cast<Exact *>(context)->signal.publish(3);
}

void *exact_step(koro_cont_t *continuation) {
    auto *context = static_cast<Exact *>(koro_cont_argument(continuation));
    switch (koro_cont_state_get(continuation)) {
    case 0:
        context->signal.publish(1);
        koro_cont_state_set(continuation, 1, KORO_SUSPEND_CALLBACK);
        return nullptr;
    case 1:
        context->signal.publish(2);
        return koro_cont_finish(continuation)
            ? reinterpret_cast<void *>(1)
            : nullptr;
    default:
        fail("exact continuation resumed at an unknown position");
    }
}

void verify_exact_callback_routing() {
    const kc_runtime_config config{.worker_count = 2};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&config, &runtime) == 0,
          "routing runtime creation failed");
    check(kc_runtime_start(runtime) == 0, "routing runtime start failed");

    Exact left;
    Exact right;
    const koro_cont_config left_config{
        .step = exact_step,
        .argument = &left,
        .frame_size = 0,
        .worker_mask = 0,
        .completion = exact_complete,
        .completion_context = &left,
    };
    const koro_cont_config right_config{
        .step = exact_step,
        .argument = &right,
        .frame_size = 0,
        .worker_mask = 0,
        .completion = exact_complete,
        .completion_context = &right,
    };
    koro_cont_t *left_continuation = nullptr;
    koro_cont_t *right_continuation = nullptr;
    check(koro_cont_create_on(runtime, &left_config,
                              &left_continuation) == 0,
          "left continuation creation failed");
    check(koro_cont_create_on(runtime, &right_config,
                              &right_continuation) == 0,
          "right continuation creation failed");
    check(koro_cont_start(left_continuation) == 0,
          "left continuation start failed");
    check(koro_cont_start(right_continuation) == 0,
          "right continuation start failed");
    left.signal.await(1, "left continuation did not suspend");
    right.signal.await(1, "right continuation did not suspend");

    const kc_ticket_id right_identity =
        koro_cont_identity(right_continuation);
    check(koro_cont_resume(right_continuation, &right_identity) == 0,
          "right callback was rejected");
    right.signal.await(3, "right continuation did not complete");
    check(left.signal.value() == 1,
          "callback resumed FIFO order instead of its named continuation");

    const kc_ticket_id left_identity =
        koro_cont_identity(left_continuation);
    check(koro_cont_resume(left_continuation, &left_identity) == 0,
          "left callback was rejected");
    left.signal.await(3, "left continuation did not complete");
    check(kc_runtime_join_all(runtime) == 0,
          "routing runtime did not become idle");
    check(koro_cont_destroy(right_continuation) == 0,
          "right continuation destroy failed");
    check(koro_cont_destroy(left_continuation) == 0,
          "left continuation destroy failed");
    stop_runtime(runtime);
}

struct Race {
    Signal signal;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic_uint runs{0};
    bool release = false;
};

void race_complete(void *context, const kc_ticket_id *identity) {
    check(identity != nullptr, "race callback lost its identity");
    static_cast<Race *>(context)->signal.publish(3);
}

void *race_step(koro_cont_t *continuation) {
    auto *race = static_cast<Race *>(koro_cont_argument(continuation));
    const unsigned run =
        race->runs.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (run == 1) {
        race->signal.publish(1);
        std::unique_lock lock(race->mutex);
        race->changed.wait(lock, [&] { return race->release; });
    }
    if (run == 2) {
        race->signal.publish(2);
    }
    check(run <= 2, "terminal race produced a duplicate invocation");
    return koro_cont_finish(continuation)
        ? reinterpret_cast<void *>(1)
        : nullptr;
}

void verify_resume_during_step_runtime() {
    const kc_runtime_config config{.worker_count = 2};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&config, &runtime) == 0,
          "race runtime creation failed");
    check(kc_runtime_start(runtime) == 0, "race runtime start failed");

    for (unsigned iteration = 0; iteration < 1024; ++iteration) {
        Race race;
        const koro_cont_config continuation_config{
            .step = race_step,
            .argument = &race,
            .frame_size = 0,
            .worker_mask = 0,
            .completion = race_complete,
            .completion_context = &race,
        };
        koro_cont_t *continuation = nullptr;
        check(koro_cont_create_on(runtime, &continuation_config,
                                  &continuation) == 0,
              "race continuation creation failed");
        check(koro_cont_start(continuation) == 0,
              "race continuation start failed");
        race.signal.await(1, "race continuation did not enter");
        const kc_ticket_id identity = koro_cont_identity(continuation);
        check(koro_cont_resume(continuation, &identity) == 0,
              "callback during execution was rejected");
        {
            std::lock_guard lock(race.mutex);
            race.release = true;
        }
        race.changed.notify_all();
        race.signal.await(3, "callback during suspension was lost");
        check(race.runs.load(std::memory_order_acquire) == 2,
              "overlapping callback did not produce one successor invocation");
        check(kc_runtime_join_all(runtime) == 0,
              "race runtime did not become idle");
        check(koro_cont_destroy(continuation) == 0,
              "race continuation destroy failed");
    }
    stop_runtime(runtime);
}

void verify_resume_during_step() {
    std::vector<std::thread> controllers;
    controllers.reserve(8);
    for (unsigned index = 0; index < 8; ++index)
        controllers.emplace_back(verify_resume_during_step_runtime);
    for (std::thread &controller : controllers) controller.join();
}

double idle_percent(kc_runtime_t *runtime,
                    std::chrono::milliseconds window) {
    std::uint64_t before = 0;
    std::uint64_t after = 0;
    check(kc_runtime_worker_cpu_ns_for_test(runtime, &before) == 0,
          "worker CPU measurement failed");
    const auto started = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(window);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    check(kc_runtime_worker_cpu_ns_for_test(runtime, &after) == 0,
          "worker CPU remeasurement failed");
    const double wall = std::chrono::duration<double>(elapsed).count();
    return 100.0 * static_cast<double>(after - before) /
        (wall * 1'000'000'000.0);
}

void verify_idle_cpu() {
    const kc_runtime_config config{.worker_count = 4};
    kc_runtime_t *runtime = nullptr;
    check(kc_runtime_create(&config, &runtime) == 0,
          "idle runtime creation failed");
    check(kc_runtime_start(runtime) == 0, "idle runtime start failed");
    std::this_thread::sleep_for(250ms);
    const double cold = idle_percent(runtime, 1s);
    check(cold < kIdleLimitPercent, "cold worker pool is spinning");

    Exact edge;
    const koro_cont_config continuation_config{
        .step = exact_step,
        .argument = &edge,
        .frame_size = 0,
        .worker_mask = 0,
        .completion = exact_complete,
        .completion_context = &edge,
    };
    koro_cont_t *continuation = nullptr;
    check(koro_cont_create_on(runtime, &continuation_config,
                              &continuation) == 0,
          "idle probe continuation creation failed");
    check(koro_cont_start(continuation) == 0,
          "idle probe continuation start failed");
    edge.signal.await(1, "idle probe did not suspend");
    const kc_ticket_id identity = koro_cont_identity(continuation);
    check(koro_cont_resume(continuation, &identity) == 0,
          "idle probe callback was rejected");
    edge.signal.await(3, "idle probe did not complete");
    check(kc_runtime_join_all(runtime) == 0,
          "idle probe did not retire");
    check(koro_cont_destroy(continuation) == 0,
          "idle probe destroy failed");

    std::this_thread::sleep_for(250ms);
    const double reparked = idle_percent(runtime, 1s);
    check(reparked < kIdleLimitPercent,
          "worker pool did not repark after work");
    stop_runtime(runtime);
}

} // namespace

int main() {
    verify_bounded_worker_pool();
    verify_frame_migration();
    verify_exact_callback_routing();
    verify_resume_during_step();
    verify_idle_cpu();
    return 0;
}
