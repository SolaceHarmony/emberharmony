// SPDX-License-Identifier: BSD-3-Clause
#include "kc_team.h"

#include "kcoro_stackless.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

enum {
    KC_TEAM_MAX_MEMBERS = 64,
    KC_TEAM_CACHELINE = 128,
    KC_TEAM_DISPATCH_PUBLISHER = 1u,
    KC_TEAM_DISPATCH_CLOSED = 2u,
    KC_TEAM_START_PUBLISHER = 1u,
    KC_TEAM_START_CLOSED = 2u,
    KC_TEAM_TEST_NEVER_ENTERED = 1u,
    KC_TEAM_TEST_ENTERED_NEVER_RETURNED = 2u,
};

typedef struct kc_team_member_progress {
    _Alignas(KC_TEAM_CACHELINE) atomic_uint_fast64_t entered_generation;
    atomic_uint_fast64_t returned_generation;
    unsigned char padding[KC_TEAM_CACHELINE -
        sizeof(atomic_uint_fast64_t) * 2];
} kc_team_member_progress;

_Static_assert(sizeof(kc_team_member_progress) == KC_TEAM_CACHELINE,
               "team member progress must own one Apple cache line");

typedef struct kc_team_member_frame {
    struct kc_team *team;
    kc_team_member_progress *progress;
    uint32_t index;
    uint32_t fault_held;
    uint64_t seen_generation;
    /* These values remain live across the fault-injection suspension points.
     * They belong to the logical coroutine frame, never the physical worker
     * stack: a callback may resume this member on a different worker. */
    uint64_t active_generation;
    uint32_t active_fault_point;
} kc_team_member_frame;

typedef void (*kc_team_test_fault_fn)(void *context, uint64_t generation);
typedef void (*kc_team_test_start_fn)(void *context, uint32_t member);

struct kc_team {
    kc_team_config config;
    koro_cont_t **continuations;
    kc_team_member_progress *progress;
    atomic_uint started_members;
    atomic_uint completed_members;
    atomic_uint retirement_arrivals;
    atomic_uint retired_members;
    atomic_uint retirement_published;
    atomic_uint started;
    atomic_uint stop_requested;
    atomic_uint start_gate;
    atomic_uint dispatch_gate;
    atomic_uint joined;
    atomic_uint_fast64_t dispatched_generation;
    atomic_uint_fast64_t completed_generation;
    atomic_uint_fast64_t retired_generation;
    kc_team_completion_fn completion_notify;
    void *completion_context;
    atomic_uint_fast64_t test_fault_generation;
    atomic_uint test_fault_member;
    atomic_uint test_fault_point;
    atomic_uint test_fault_settled;
    atomic_uint test_start_failure_after;
    atomic_uint test_start_pause_member;
    kc_team_test_start_fn test_start_pause;
    void *test_start_pause_context;
    kc_team_test_fault_fn test_fault_ready;
    void *test_fault_context;
};

static _Thread_local kc_team_t *current_team;
static _Thread_local uint32_t current_member;
static _Thread_local kc_team_t *current_completion_team;

static void publish_retirement_if_complete(kc_team_t *team)
{
    if (!team || !atomic_load_explicit(&team->stop_requested,
                                        memory_order_acquire) ||
        !atomic_load_explicit(&team->started, memory_order_acquire) ||
        (atomic_load_explicit(&team->start_gate, memory_order_acquire) &
         KC_TEAM_START_PUBLISHER)) return;
    const unsigned started = atomic_load_explicit(
        &team->started_members, memory_order_acquire);
    if (atomic_load_explicit(&team->retirement_arrivals,
                             memory_order_acquire) != started ||
        atomic_load_explicit(&team->retired_members,
                             memory_order_acquire) != started) return;
    unsigned expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->retirement_published, &expected, 1,
            memory_order_acq_rel, memory_order_acquire)) return;
    if (team->config.retired) {
        team->config.retired(
            team->config.retired_context,
            atomic_load_explicit(&team->retired_generation,
                                 memory_order_acquire));
    }
    atomic_store_explicit(&team->retirement_published, 2,
                          memory_order_release);
    atomic_store_explicit(&team->joined, 1, memory_order_release);
}

static void member_settled(void *context, const kc_ticket_id *identity)
{
    kc_team_t *team = context;
    if (!team || !identity) abort();
    atomic_fetch_add_explicit(&team->retirement_arrivals, 1,
                              memory_order_acq_rel);
    atomic_fetch_add_explicit(&team->retired_members, 1,
                              memory_order_acq_rel);
    /* started_members is not stable while the start publisher owns its gate.
     * A member may settle during that window; start_leave is then the causal
     * successor that evaluates the final admitted count. */
    publish_retirement_if_complete(team);
}

static void resume_started_members(kc_team_t *team)
{
    const uint32_t started = atomic_load_explicit(
        &team->started_members, memory_order_acquire);
    for (uint32_t index = 0; index < started; ++index)
        (void)koro_cont_resume_internal(team->continuations[index]);
}

static void resume_terminal_members(kc_team_t *team)
{
    if (!atomic_load_explicit(&team->stop_requested,
                              memory_order_acquire)) return;
    if (atomic_load_explicit(&team->start_gate, memory_order_acquire) &
        KC_TEAM_START_PUBLISHER) return;
    if (atomic_load_explicit(&team->dispatch_gate, memory_order_acquire) &
        KC_TEAM_DISPATCH_PUBLISHER) return;
    resume_started_members(team);
}

static int start_enter(kc_team_t *team)
{
    unsigned expected = atomic_load_explicit(&team->start_gate,
                                             memory_order_acquire);
    if (expected & KC_TEAM_START_PUBLISHER) return -EBUSY;
    if (expected & KC_TEAM_START_CLOSED) return -ECANCELED;
    const unsigned claimed = expected | KC_TEAM_START_PUBLISHER;
    if (atomic_compare_exchange_strong_explicit(
            &team->start_gate, &expected, claimed,
            memory_order_acquire, memory_order_acquire)) return 0;
    return (expected & KC_TEAM_START_CLOSED) ? -ECANCELED : -EBUSY;
}

static void start_leave(kc_team_t *team)
{
    const unsigned before = atomic_fetch_sub_explicit(
        &team->start_gate, KC_TEAM_START_PUBLISHER,
        memory_order_acq_rel);
    if ((before & KC_TEAM_START_PUBLISHER) == 0) abort();
    if (before & KC_TEAM_START_CLOSED) resume_terminal_members(team);
    publish_retirement_if_complete(team);
}

static int dispatch_enter(kc_team_t *team)
{
    unsigned expected = atomic_load_explicit(&team->dispatch_gate,
                                             memory_order_acquire);
    if (expected & KC_TEAM_DISPATCH_PUBLISHER) return -EBUSY;
    if (expected & KC_TEAM_DISPATCH_CLOSED) return -ECANCELED;
    const unsigned claimed = expected | KC_TEAM_DISPATCH_PUBLISHER;
    if (atomic_compare_exchange_strong_explicit(
            &team->dispatch_gate, &expected, claimed,
            memory_order_acquire, memory_order_acquire)) return 0;
    return (expected & KC_TEAM_DISPATCH_CLOSED) ? -ECANCELED : -EBUSY;
}

static void dispatch_leave(kc_team_t *team)
{
    const unsigned before = atomic_fetch_sub_explicit(
        &team->dispatch_gate, KC_TEAM_DISPATCH_PUBLISHER,
        memory_order_acq_rel);
    if ((before & KC_TEAM_DISPATCH_PUBLISHER) == 0) abort();
    /* Stop closes dispatch admission without racing the admitted publisher.
     * If close arrived while this publisher owned the gate, releasing the
     * gate becomes the causal edge that lets every logical member observe the
     * terminal state. A duplicate generation resume is harmless and
     * coalesces in the continuation's wake-bearing state. */
    if (before & KC_TEAM_DISPATCH_CLOSED) resume_terminal_members(team);
}

static void test_fault_settle(kc_team_t *team, uint64_t generation)
{
    if (atomic_load_explicit(&team->test_fault_generation,
                             memory_order_acquire) != generation) return;
    const unsigned settled = atomic_fetch_add_explicit(
        &team->test_fault_settled, 1, memory_order_acq_rel) + 1;
    if (settled != team->config.member_count) return;
    if (!team->test_fault_ready) abort();
    team->test_fault_ready(team->test_fault_context, generation);
}

static void finish_member_generation(kc_team_member_frame *member,
                                     uint64_t generation)
{
    kc_team_t *team = member->team;
    atomic_store_explicit(&member->progress->returned_generation, generation,
                          memory_order_release);
    test_fault_settle(team, generation);
    if (atomic_fetch_add_explicit(&team->completed_members, 1,
                                  memory_order_acq_rel) + 1 !=
        team->config.member_count) return;
    kc_team_completion_fn completion = team->completion_notify;
    void *context = team->completion_context;
    atomic_store_explicit(&team->completed_generation, generation,
                          memory_order_release);
    atomic_store_explicit(&team->retired_generation, generation,
                          memory_order_release);
    if (!completion) return;
    current_completion_team = team;
    completion(context, generation);
    current_completion_team = NULL;
}

static void *team_member_step(koro_cont_t *continuation)
{
    kc_team_member_frame *member = koro_cont_frame(continuation);
    kc_team_t *team = member->team;
    KORO_BEGIN(continuation);
    for (;;) {
        while (atomic_load_explicit(&team->dispatched_generation,
                                    memory_order_acquire) ==
               member->seen_generation) {
            if (atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire)) {
                return koro_cont_finish(continuation) ? (void *)1 : NULL;
            }
            KORO_SUSPEND(continuation);
        }

        member->seen_generation = atomic_load_explicit(
            &team->dispatched_generation, memory_order_acquire);
        member->active_generation = member->seen_generation;
        const int selected =
            atomic_load_explicit(&team->test_fault_generation,
                                 memory_order_acquire) ==
                member->active_generation &&
            atomic_load_explicit(&team->test_fault_member,
                                 memory_order_relaxed) == member->index;
        member->active_fault_point = selected
            ? atomic_load_explicit(&team->test_fault_point,
                                   memory_order_relaxed)
            : 0;
        if (member->active_fault_point == KC_TEAM_TEST_NEVER_ENTERED) {
            member->fault_held = 1;
            test_fault_settle(team, member->active_generation);
            while (!atomic_load_explicit(&team->stop_requested,
                                         memory_order_acquire))
                KORO_SUSPEND(continuation);
            return koro_cont_finish(continuation) ? (void *)1 : NULL;
        }

        atomic_store_explicit(&member->progress->entered_generation,
                              member->active_generation,
                              memory_order_release);
        if (member->active_fault_point ==
            KC_TEAM_TEST_ENTERED_NEVER_RETURNED) {
            member->fault_held = 1;
            test_fault_settle(team, member->active_generation);
            while (!atomic_load_explicit(&team->stop_requested,
                                         memory_order_acquire))
                KORO_SUSPEND(continuation);
            return koro_cont_finish(continuation) ? (void *)1 : NULL;
        }

        current_team = team;
        current_member = member->index;
        team->config.member(team->config.context, member->index,
                            team->config.member_count,
                            member->active_generation);
        current_team = NULL;
        finish_member_generation(member, member->active_generation);
    }
    KORO_END(continuation);
}

int kc_team_create(const kc_team_config *config, kc_team_t **out)
{
    if (!config || !out || !config->runtime ||
        !config->member || config->member_count == 0 ||
        config->member_count > KC_TEAM_MAX_MEMBERS) return -EINVAL;
    kc_team_t *team = calloc(1, sizeof(*team));
    if (!team) return -ENOMEM;
    team->config = *config;
    team->continuations = calloc(config->member_count,
                                 sizeof(*team->continuations));
    team->progress = aligned_alloc(
        KC_TEAM_CACHELINE,
        config->member_count * sizeof(*team->progress));
    if (!team->continuations || !team->progress) {
        free(team->progress);
        free(team->continuations);
        free(team);
        return -ENOMEM;
    }
    for (uint32_t index = 0; index < config->member_count; ++index) {
        atomic_init(&team->progress[index].entered_generation, 0);
        atomic_init(&team->progress[index].returned_generation, 0);
        const koro_cont_config member_config = {
            .step = team_member_step,
            .argument = team,
            .frame_size = sizeof(kc_team_member_frame),
            .worker_mask = 0,
            .completion = NULL,
            .completion_context = NULL,
        };
        const int status = koro_cont_create_on(config->runtime,
                                                &member_config,
                                                &team->continuations[index]);
        if (status != 0) {
            for (uint32_t prior = 0; prior < index; ++prior)
                (void)koro_cont_destroy(team->continuations[prior]);
            free(team->progress);
            free(team->continuations);
            free(team);
            return status;
        }
        team->continuations[index]->settled = member_settled;
        team->continuations[index]->settled_context = team;
        kc_team_member_frame *frame = koro_cont_frame(
            team->continuations[index]);
        *frame = (kc_team_member_frame){
            .team = team,
            .progress = &team->progress[index],
            .index = index,
        };
    }
    atomic_init(&team->started_members, 0);
    atomic_init(&team->completed_members, 0);
    atomic_init(&team->retirement_arrivals, 0);
    atomic_init(&team->retired_members, 0);
    atomic_init(&team->retirement_published, 0);
    atomic_init(&team->started, 0);
    atomic_init(&team->stop_requested, 0);
    atomic_init(&team->start_gate, 0);
    atomic_init(&team->dispatch_gate, 0);
    atomic_init(&team->joined, 0);
    atomic_init(&team->dispatched_generation, 0);
    atomic_init(&team->completed_generation, 0);
    atomic_init(&team->retired_generation, 0);
    atomic_init(&team->test_fault_generation, 0);
    atomic_init(&team->test_fault_member, 0);
    atomic_init(&team->test_fault_point, 0);
    atomic_init(&team->test_fault_settled, 0);
    atomic_init(&team->test_start_failure_after, UINT32_MAX);
    atomic_init(&team->test_start_pause_member, UINT32_MAX);
    *out = team;
    return 0;
}

int kc_team_start(kc_team_t *team)
{
    if (!team) return -EINVAL;
    const int admission = start_enter(team);
    if (admission != 0) return admission;
    unsigned expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->started, &expected, 1, memory_order_acq_rel,
            memory_order_acquire)) {
        start_leave(team);
        if (expected != 1) return -EINVAL;
        return atomic_load_explicit(&team->started_members,
                                    memory_order_acquire) ==
                       team->config.member_count &&
                   !atomic_load_explicit(&team->stop_requested,
                                         memory_order_acquire)
               ? 0 : -ECANCELED;
    }
    uint32_t started = 0;
    int failure = -EAGAIN;
    for (; started < team->config.member_count; ++started) {
        if (atomic_load_explicit(&team->stop_requested,
                                 memory_order_acquire)) {
            failure = -ECANCELED;
            break;
        }
        if (started == atomic_load_explicit(
                &team->test_start_failure_after, memory_order_relaxed))
            break;
        if (started == atomic_load_explicit(
                &team->test_start_pause_member, memory_order_relaxed) &&
            team->test_start_pause) {
            /* The test hook sits after the stop observation but before the
             * irreversible admission count. It exercises the exact edge where
             * an already-running member can settle while this next member is
             * still being published. */
            team->test_start_pause(team->test_start_pause_context, started);
        }
        /* Admission is counted before the frame can become runnable. Stop
         * closes start_gate and deposits intent; it cannot resume this member
         * until the complete admitted count is stable. */
        atomic_fetch_add_explicit(&team->started_members, 1,
                                  memory_order_release);
        const int status = koro_cont_start(team->continuations[started]);
        if (status != 0) {
            atomic_fetch_sub_explicit(&team->started_members, 1,
                                      memory_order_release);
            failure = status;
            break;
        }
    }
    const int complete = started == team->config.member_count;
    if (!complete) {
        atomic_store_explicit(&team->stop_requested, 1,
                              memory_order_release);
        atomic_fetch_or_explicit(&team->start_gate, KC_TEAM_START_CLOSED,
                                 memory_order_acq_rel);
        atomic_fetch_or_explicit(&team->dispatch_gate,
                                 KC_TEAM_DISPATCH_CLOSED,
                                 memory_order_acq_rel);
    }
    start_leave(team);
    if (!complete) return failure;
    return atomic_load_explicit(&team->stop_requested,
                                memory_order_acquire)
               ? -ECANCELED : 0;
}

int kc_team_dispatch(kc_team_t *team, uint64_t generation)
{
    return kc_team_dispatch_notify(team, generation, NULL, NULL);
}

int kc_team_dispatch_notify(kc_team_t *team, uint64_t generation,
                            kc_team_completion_fn completion, void *context)
{
    if (!team || generation == 0) return -EINVAL;
    if (!atomic_load_explicit(&team->started, memory_order_acquire) ||
        atomic_load_explicit(&team->stop_requested, memory_order_acquire) ||
        atomic_load_explicit(&team->started_members,
                             memory_order_acquire) !=
            team->config.member_count)
        return -ECANCELED;
    const int status = dispatch_enter(team);
    if (status != 0) return status;
    const uint64_t dispatched = atomic_load_explicit(
        &team->dispatched_generation, memory_order_acquire);
    const uint64_t retired = atomic_load_explicit(
        &team->retired_generation, memory_order_acquire);
    if (dispatched != retired) {
        dispatch_leave(team);
        return -EBUSY;
    }
    if (generation <= dispatched) {
        dispatch_leave(team);
        return -EINVAL;
    }
    team->completion_notify = completion;
    team->completion_context = context;
    atomic_store_explicit(&team->completed_members, 0, memory_order_relaxed);
    atomic_store_explicit(&team->dispatched_generation, generation,
                          memory_order_release);
    dispatch_leave(team);
    for (uint32_t index = 0; index < team->config.member_count; ++index) {
        const int resume = koro_cont_resume_internal(
            team->continuations[index]);
        if (resume != 0) return resume;
    }
    return 0;
}

void kc_team_request_stop(kc_team_t *team)
{
    if (!team) return;
    atomic_fetch_or_explicit(&team->start_gate, KC_TEAM_START_CLOSED,
                             memory_order_acq_rel);
    atomic_fetch_or_explicit(
        &team->dispatch_gate, KC_TEAM_DISPATCH_CLOSED,
        memory_order_acq_rel);
    atomic_store_explicit(&team->stop_requested, 1, memory_order_release);
    /* The last active publisher is the successor edge. If neither admission
     * publisher remains, stop itself may publish the exact resume edge. */
    resume_terminal_members(team);
    publish_retirement_if_complete(team);
}

int kc_team_join(kc_team_t *team)
{
    if (!team) return -EINVAL;
    if (current_team == team || current_completion_team == team)
        return -EDEADLK;
    if (!atomic_load_explicit(&team->started, memory_order_acquire)) return 0;
    if (atomic_load_explicit(&team->joined, memory_order_acquire)) return 0;
    return -EBUSY;
}

int kc_team_destroy(kc_team_t *team)
{
    if (!team) return 0;
    if (atomic_load_explicit(&team->started, memory_order_acquire) &&
        !atomic_load_explicit(&team->joined, memory_order_acquire)) return -EBUSY;
    for (uint32_t index = 0; index < team->config.member_count; ++index) {
        const int status = koro_cont_destroy(team->continuations[index]);
        if (status != 0) return status;
    }
    free(team->progress);
    free(team->continuations);
    free(team);
    return 0;
}

int kc_team_snapshot_get(kc_team_t *team, kc_team_snapshot *out)
{
    if (!team || !out) return -EINVAL;
    *out = (kc_team_snapshot){
        .member_count = team->config.member_count,
        .started_members = atomic_load_explicit(&team->started_members,
                                                memory_order_acquire),
        .dispatched_generation = atomic_load_explicit(
            &team->dispatched_generation, memory_order_acquire),
        .completed_generation = atomic_load_explicit(
            &team->completed_generation, memory_order_acquire),
        .completed_members = atomic_load_explicit(&team->completed_members,
                                                  memory_order_acquire),
        .started = atomic_load_explicit(&team->started, memory_order_acquire),
        .stop_requested = atomic_load_explicit(&team->stop_requested,
                                               memory_order_acquire),
        .joined = atomic_load_explicit(&team->joined, memory_order_acquire),
    };
    return 0;
}

int kc_team_quorum_snapshot_get(kc_team_t *team, uint64_t generation,
                                kc_team_quorum_snapshot *out)
{
    if (!team || !out || generation == 0)
        return -EINVAL;
    if (atomic_load_explicit(&team->dispatched_generation,
                             memory_order_acquire) != generation)
        return -ESTALE;
    uint64_t entered = 0;
    uint64_t returned = 0;
    for (uint32_t index = 0; index < team->config.member_count; ++index) {
        const uint64_t bit = UINT64_C(1) << index;
        const uint64_t member_returned = atomic_load_explicit(
            &team->progress[index].returned_generation, memory_order_acquire);
        if (member_returned == generation) {
            returned |= bit;
            entered |= bit;
            continue;
        }
        const uint64_t member_entered = atomic_load_explicit(
            &team->progress[index].entered_generation, memory_order_acquire);
        if (member_entered == generation) entered |= bit;
    }
    if (atomic_load_explicit(&team->dispatched_generation,
                             memory_order_acquire) != generation)
        return -EAGAIN;
    const uint64_t expected = team->config.member_count == KC_TEAM_MAX_MEMBERS
        ? UINT64_MAX : (UINT64_C(1) << team->config.member_count) - 1;
    *out = (kc_team_quorum_snapshot){
        .generation = generation,
        .expected_mask = expected,
        .entered_mask = entered,
        .returned_mask = returned,
    };
    return 0;
}

int kc_team_current_member(const kc_team_t *team, uint32_t *out_member)
{
    if (!team || !out_member) return -EINVAL;
    if (current_team != team) return -EPERM;
    *out_member = current_member;
    return 0;
}

int kc_team_inject_member_exit_for_test(
    kc_team_t *team, uint64_t generation, uint32_t member, uint32_t point,
    kc_team_test_fault_fn ready, void *context)
{
    if (!team || generation == 0 || member >= team->config.member_count ||
        (point != KC_TEAM_TEST_NEVER_ENTERED &&
         point != KC_TEAM_TEST_ENTERED_NEVER_RETURNED) || !ready)
        return -EINVAL;
    if (atomic_load_explicit(&team->started, memory_order_acquire))
        return -EBUSY;
    if (generation != 1) return -EINVAL;
    uint64_t expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->test_fault_generation, &expected, generation,
            memory_order_acq_rel, memory_order_acquire)) return -EALREADY;
    team->test_fault_ready = ready;
    team->test_fault_context = context;
    atomic_store_explicit(&team->test_fault_member, member,
                          memory_order_relaxed);
    atomic_store_explicit(&team->test_fault_point, point,
                          memory_order_relaxed);
    atomic_store_explicit(&team->test_fault_settled, 0,
                          memory_order_relaxed);
    return 0;
}

int kc_team_inject_start_failure_for_test(kc_team_t *team,
                                          uint32_t after_started)
{
    if (!team || after_started >= team->config.member_count) return -EINVAL;
    if (atomic_load_explicit(&team->started, memory_order_acquire))
        return -EBUSY;
    atomic_store_explicit(&team->test_start_failure_after, after_started,
                          memory_order_relaxed);
    return 0;
}

int kc_team_inject_start_pause_for_test(kc_team_t *team, uint32_t member,
                                        kc_team_test_start_fn pause,
                                        void *context)
{
    if (!team || member >= team->config.member_count || !pause)
        return -EINVAL;
    if (atomic_load_explicit(&team->started, memory_order_acquire))
        return -EBUSY;
    team->test_start_pause = pause;
    team->test_start_pause_context = context;
    atomic_store_explicit(&team->test_start_pause_member, member,
                          memory_order_relaxed);
    return 0;
}
