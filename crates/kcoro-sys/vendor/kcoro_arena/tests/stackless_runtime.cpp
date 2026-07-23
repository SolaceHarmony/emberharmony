// SPDX-License-Identifier: BSD-3-Clause

#include "kcoro_arena.h"

#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <vector>

namespace {

constexpr std::uint64_t kMarker = UINT64_C(0x51a71e55c0def00d);

void *terminal_step(koro_cont_t *continuation) {
    return koro_cont_finish(continuation) ? reinterpret_cast<void *>(1)
                                         : nullptr;
}

void verify_bounded_dormancy() {
    const kc_runtime_config config{
        .worker_count = 1,
    };
    kc_runtime_t *runtime = nullptr;
    assert(kc_runtime_create(&config, &runtime) == 0);

    const koro_cont_config continuation{
        .step = terminal_step,
        .argument = nullptr,
        .frame_size = 0,
        .worker_mask = 0,
        .completion = nullptr,
        .completion_context = nullptr,
    };
    std::vector<koro_cont_t *> admitted;
    for (;;) {
        koro_cont_t *candidate = nullptr;
        const int status =
            koro_cont_create_on(runtime, &continuation, &candidate);
        if (status == -ENOSPC) break;
        assert(status == 0);
        assert(candidate != nullptr);
        admitted.push_back(candidate);
    }

    assert(admitted.size() == 64);
    const kc_ticket_id retired = koro_cont_identity(admitted.front());
    assert(koro_cont_destroy(admitted.front()) == 0);
    admitted.erase(admitted.begin());

    koro_cont_t *replacement = nullptr;
    assert(koro_cont_create_on(runtime, &continuation, &replacement) == 0);
    const kc_ticket_id current = koro_cont_identity(replacement);
    assert(current.runtime_epoch == retired.runtime_epoch);
    assert(current.sequence != retired.sequence);
    assert(current.generation != retired.generation);
    assert(koro_cont_resume(replacement, &retired) == -ESTALE);

    assert(koro_cont_destroy(replacement) == 0);
    for (koro_cont_t *item : admitted) assert(koro_cont_destroy(item) == 0);
    assert(kc_runtime_destroy(runtime) == 0);
}

struct TargetFrame {
    std::uint64_t marker;
    std::uint32_t runs;
};

struct Target {
    std::atomic_uint completions{0};
};

void target_complete(void *context, const kc_ticket_id *identity) {
    auto *target = static_cast<Target *>(context);
    assert(identity != nullptr);
    target->completions.fetch_add(1, std::memory_order_release);
}

void *target_step(koro_cont_t *continuation) {
    auto *frame = static_cast<TargetFrame *>(koro_cont_frame(continuation));
    assert(frame != nullptr);
    KORO_BEGIN(continuation);
    frame->marker = kMarker;
    ++frame->runs;
    KORO_SUSPEND(continuation);
    assert(frame->marker == kMarker);
    ++frame->runs;
    KORO_END(continuation);
}

struct Trigger {
    koro_cont_t *target;
    kc_ticket_id identity;
};

void *trigger_step(koro_cont_t *continuation) {
    auto *trigger = static_cast<Trigger *>(
        koro_cont_argument(continuation));
    assert(trigger != nullptr);
    kc_ticket_id stale = trigger->identity;
    ++stale.generation;
    if (stale.generation == 0) ++stale.generation;
    assert(koro_cont_resume(trigger->target, &stale) == -ESTALE);
    assert(koro_cont_resume(trigger->target, &trigger->identity) == 0);
    return koro_cont_finish(continuation) ? reinterpret_cast<void *>(1)
                                         : nullptr;
}

void verify_correlated_trampoline() {
    const kc_runtime_config config{
        .worker_count = 1,
    };
    kc_runtime_t *runtime = nullptr;
    assert(kc_runtime_create(&config, &runtime) == 0);

    Target target;
    const koro_cont_config target_config{
        .step = target_step,
        .argument = &target,
        .frame_size = sizeof(TargetFrame),
        .worker_mask = 0,
        .completion = target_complete,
        .completion_context = &target,
    };
    koro_cont_t *target_continuation = nullptr;
    assert(koro_cont_create_on(runtime, &target_config,
                               &target_continuation) == 0);

    Trigger trigger{
        .target = target_continuation,
        .identity = koro_cont_identity(target_continuation),
    };
    const koro_cont_config trigger_config{
        .step = trigger_step,
        .argument = &trigger,
        .frame_size = 0,
        .worker_mask = 0,
        .completion = nullptr,
        .completion_context = nullptr,
    };
    koro_cont_t *trigger_continuation = nullptr;
    assert(koro_cont_create_on(runtime, &trigger_config,
                               &trigger_continuation) == 0);

    /*
     * Both records are published before the worker exists. Slot order makes
     * the target run and dehydrate first; the trigger then acts as the external
     * completion callback and publishes the target's exact successor edge.
     * No observer polls, sleeps, or waits beside either operation.
     */
    assert(koro_cont_start(target_continuation) == 0);
    assert(koro_cont_start(trigger_continuation) == 0);
    assert(kc_runtime_start(runtime) == 0);
    assert(kc_runtime_join_all(runtime) == 0);

    const auto *frame = static_cast<const TargetFrame *>(
        koro_cont_frame(target_continuation));
    assert(frame->marker == kMarker);
    assert(frame->runs == 2);
    assert(target.completions.load(std::memory_order_acquire) == 1);

    assert(koro_cont_destroy(trigger_continuation) == 0);
    assert(koro_cont_destroy(target_continuation) == 0);
    kc_runtime_request_stop(runtime);
    assert(kc_runtime_join(runtime) == 0);
    assert(kc_runtime_destroy(runtime) == 0);
}

} // namespace

int main() {
    verify_bounded_dormancy();
    verify_correlated_trampoline();
    return 0;
}
