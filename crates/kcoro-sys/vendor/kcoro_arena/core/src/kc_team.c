// SPDX-License-Identifier: BSD-3-Clause
#include "kc_team.h"

#include "kc_doorbell.h"
#include "kc_port.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

/* Implementation-private deterministic fault injection. It is linked only so
 * subprocess tests can exercise the real fixed-member lifecycle; the public
 * kc_team header deliberately exposes no team-poison surface. */
typedef void (*kc_team_test_fault_fn)(void *context, uint64_t generation);

enum kc_team_test_fault_point {
    KC_TEAM_TEST_NEVER_ENTERED = 1,
    KC_TEAM_TEST_ENTERED_NEVER_RETURNED = 2,
};

int kc_team_inject_member_exit_for_test(
    kc_team_t *team, uint64_t generation, uint32_t member, uint32_t point,
    kc_team_test_fault_fn ready, void *context);

enum {
    KC_TEAM_ABI_VERSION = 1u,
    KC_TEAM_CACHELINE = 128u,
    KC_TEAM_MAX_MEMBERS = 64u,
};

#define KC_TEAM_DISPATCH_CLOSED UINT32_C(0x80000000)
#define KC_TEAM_DISPATCH_PUBLISHER UINT32_C(0x00000001)

typedef struct kc_team_member_progress {
    _Alignas(KC_TEAM_CACHELINE) atomic_uint_fast64_t entered_generation;
    atomic_uint_fast64_t returned_generation;
    unsigned char padding[KC_TEAM_CACHELINE -
                          2 * sizeof(atomic_uint_fast64_t)];
} kc_team_member_progress;

_Static_assert(_Alignof(kc_team_member_progress) == KC_TEAM_CACHELINE,
               "team member progress must be cache isolated");
_Static_assert(sizeof(kc_team_member_progress) == KC_TEAM_CACHELINE,
               "adjacent team progress cells must not share cache lines");

typedef struct kc_team_member {
    struct kc_team *team;
    kc_team_member_progress *progress;
    uint32_t index;
    uint64_t seen_generation;
} kc_team_member;

struct kc_team {
    kc_team_config config;
    kc_port_thread **threads;
    kc_team_member *members;
    kc_team_member_progress *progress;
    kc_doorbell_t *dispatch;
    kc_doorbell_t *readiness;
    atomic_uint started_members;
    atomic_uint completed_members;
    atomic_uint started;
    atomic_uint stop_requested;
    /* High bit closes dispatch admission; low bit is the one publisher that
     * crossed the gate. There is never a publisher-side spin or mutex. */
    atomic_uint dispatch_gate;
    atomic_uint joining;
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
    kc_team_test_fault_fn test_fault_ready;
    void *test_fault_context;
    kc_doorbell_t *test_fault_hold;
};

static _Thread_local kc_team_t *current_team;
static _Thread_local uint32_t current_member;
static _Thread_local kc_team_t *current_completion_team;

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
    unsigned before = atomic_fetch_sub_explicit(
        &team->dispatch_gate, KC_TEAM_DISPATCH_PUBLISHER,
        memory_order_release);
    if (before & KC_TEAM_DISPATCH_CLOSED)
        kc_doorbell_ring_all(team->dispatch);
}

static void test_fault_settle(kc_team_t *team, uint64_t generation)
{
    if (atomic_load_explicit(&team->test_fault_generation,
                             memory_order_acquire) != generation) return;
    unsigned settled = atomic_fetch_add_explicit(&team->test_fault_settled, 1,
                                                  memory_order_acq_rel) + 1;
    if (settled != team->config.member_count) return;
    kc_team_test_fault_fn ready = team->test_fault_ready;
    if (!ready) abort();
    ready(team->test_fault_context, generation);
}

static void *team_member_main(void *context)
{
    kc_team_member *member = context;
    kc_team_t *team = member->team;
    uint32_t observed = kc_doorbell_observe(team->dispatch);
    atomic_fetch_add_explicit(&team->started_members, 1, memory_order_release);
    kc_doorbell_ring_all(team->readiness);

    for (;;) {
        uint64_t generation = atomic_load_explicit(
            &team->dispatched_generation, memory_order_acquire);
        while (generation == member->seen_generation) {
            const unsigned gate = atomic_load_explicit(
                &team->dispatch_gate, memory_order_acquire);
            if (atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire) &&
                !(gate & KC_TEAM_DISPATCH_PUBLISHER)) return NULL;
            int status = kc_doorbell_park(team->dispatch, observed);
            observed = kc_doorbell_observe(team->dispatch);
            if (status != 0 &&
                atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire) &&
                !(atomic_load_explicit(&team->dispatch_gate,
                                       memory_order_acquire) &
                  KC_TEAM_DISPATCH_PUBLISHER)) return NULL;
            if (status != 0) abort();
            generation = atomic_load_explicit(
                &team->dispatched_generation, memory_order_acquire);
        }
        /* A generation admitted before stop remains authoritative. Every
         * member completes it and produces its one final-return edge; stop
         * only prevents the next admission. */
        member->seen_generation = generation;
        atomic_store_explicit(&member->progress->entered_generation, generation,
                              memory_order_release);
        current_team = team;
        current_member = member->index;
        team->config.member(team->config.context, member->index,
                            team->config.member_count, generation);
        current_team = NULL;
        atomic_store_explicit(&member->progress->returned_generation, generation,
                              memory_order_release);
        if (atomic_fetch_add_explicit(&team->completed_members, 1,
                                      memory_order_acq_rel) + 1 ==
            team->config.member_count) {
            kc_team_completion_fn completion = team->completion_notify;
            void *context = team->completion_context;
            atomic_store_explicit(&team->completed_generation, generation,
                                  memory_order_release);
            atomic_store_explicit(&team->retired_generation, generation,
                                  memory_order_release);
            if (completion) {
                current_completion_team = team;
                completion(context, generation);
                current_completion_team = NULL;
            }
        }
    }
}

static void *team_member_fault_main(void *context)
{
    kc_team_member *member = context;
    kc_team_t *team = member->team;
    uint32_t observed = kc_doorbell_observe(team->dispatch);
    atomic_fetch_add_explicit(&team->started_members, 1, memory_order_release);
    kc_doorbell_ring_all(team->readiness);

    for (;;) {
        uint64_t generation = atomic_load_explicit(
            &team->dispatched_generation, memory_order_acquire);
        while (generation == member->seen_generation) {
            const unsigned gate = atomic_load_explicit(
                &team->dispatch_gate, memory_order_acquire);
            if (atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire) &&
                !(gate & KC_TEAM_DISPATCH_PUBLISHER)) return NULL;
            int status = kc_doorbell_park(team->dispatch, observed);
            observed = kc_doorbell_observe(team->dispatch);
            if (status != 0 &&
                atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire) &&
                !(atomic_load_explicit(&team->dispatch_gate,
                                       memory_order_acquire) &
                  KC_TEAM_DISPATCH_PUBLISHER)) return NULL;
            if (status != 0) abort();
            generation = atomic_load_explicit(
                &team->dispatched_generation, memory_order_acquire);
        }

        member->seen_generation = generation;
        const int selected =
            atomic_load_explicit(&team->test_fault_generation,
                                 memory_order_acquire) == generation &&
            atomic_load_explicit(&team->test_fault_member,
                                 memory_order_relaxed) == member->index;
        const uint32_t point = selected
            ? atomic_load_explicit(&team->test_fault_point,
                                   memory_order_relaxed)
            : 0;
        if (point == KC_TEAM_TEST_NEVER_ENTERED) {
            test_fault_settle(team, generation);
            return NULL;
        }
        atomic_store_explicit(&member->progress->entered_generation, generation,
                              memory_order_release);
        if (point == KC_TEAM_TEST_ENTERED_NEVER_RETURNED) {
            test_fault_settle(team, generation);
            const uint32_t expected = kc_doorbell_observe(team->test_fault_hold);
            for (;;) (void)kc_doorbell_park(team->test_fault_hold, expected);
        }
        current_team = team;
        current_member = member->index;
        team->config.member(team->config.context, member->index,
                            team->config.member_count, generation);
        current_team = NULL;
        atomic_store_explicit(&member->progress->returned_generation, generation,
                              memory_order_release);
        test_fault_settle(team, generation);
        if (atomic_fetch_add_explicit(&team->completed_members, 1,
                                      memory_order_acq_rel) + 1 ==
            team->config.member_count) {
            kc_team_completion_fn completion = team->completion_notify;
            void *completion_context = team->completion_context;
            atomic_store_explicit(&team->completed_generation, generation,
                                  memory_order_release);
            atomic_store_explicit(&team->retired_generation, generation,
                                  memory_order_release);
            if (completion) {
                current_completion_team = team;
                completion(completion_context, generation);
                current_completion_team = NULL;
            }
        }
    }
}

int kc_team_create(const kc_team_config *config, kc_team_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_TEAM_ABI_VERSION || !config->member ||
        config->member_count == 0 ||
        config->member_count > KC_TEAM_MAX_MEMBERS ||
        config->reserved != 0) return -EINVAL;
    kc_team_t *team = calloc(1, sizeof(*team));
    if (!team) return -ENOMEM;
    team->config = *config;
    team->threads = calloc(config->member_count, sizeof(*team->threads));
    team->members = calloc(config->member_count, sizeof(*team->members));
    team->progress = aligned_alloc(
        KC_TEAM_CACHELINE,
        config->member_count * sizeof(*team->progress));
    if (!team->threads || !team->members || !team->progress) {
        free(team->progress);
        free(team->members);
        free(team->threads);
        free(team);
        return -ENOMEM;
    }
    for (uint32_t index = 0; index < config->member_count; ++index) {
        atomic_init(&team->progress[index].entered_generation, 0);
        atomic_init(&team->progress[index].returned_generation, 0);
    }
    atomic_init(&team->started_members, 0);
    atomic_init(&team->completed_members, 0);
    atomic_init(&team->started, 0);
    atomic_init(&team->stop_requested, 0);
    atomic_init(&team->dispatch_gate, 0);
    atomic_init(&team->joining, 0);
    atomic_init(&team->joined, 0);
    atomic_init(&team->dispatched_generation, 0);
    atomic_init(&team->completed_generation, 0);
    atomic_init(&team->retired_generation, 0);
    atomic_init(&team->test_fault_generation, 0);
    atomic_init(&team->test_fault_member, 0);
    atomic_init(&team->test_fault_point, 0);
    atomic_init(&team->test_fault_settled, 0);
    if (kc_doorbell_create(&team->dispatch) != 0 ||
        kc_doorbell_create(&team->readiness) != 0) {
        kc_doorbell_destroy(team->dispatch);
        free(team->progress);
        free(team->members);
        free(team->threads);
        free(team);
        return -ENOTSUP;
    }
    *out = team;
    return 0;
}

int kc_team_start(kc_team_t *team)
{
    if (!team) return -EINVAL;
    unsigned expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->started, &expected, 1, memory_order_acq_rel,
            memory_order_acquire)) return expected == 1 ? 0 : -EINVAL;
    const int inject = atomic_load_explicit(&team->test_fault_generation,
                                            memory_order_acquire) != 0;
    uint32_t started = 0;
    for (; started < team->config.member_count; ++started) {
        team->members[started] = (kc_team_member){
            .team = team,
            .progress = &team->progress[started],
            .index = started,
            .seen_generation = 0,
        };
        int status = kc_port_thread_create(&team->threads[started],
                                           inject ? team_member_fault_main
                                                  : team_member_main,
                                           &team->members[started]);
        if (status != 0) break;
    }
    if (started != team->config.member_count) {
        atomic_store_explicit(&team->stop_requested, 1, memory_order_release);
        kc_doorbell_ring_all(team->dispatch);
        for (uint32_t index = 0; index < started; ++index)
            kc_port_thread_join(team->threads[index]);
        atomic_store_explicit(&team->joined, 1, memory_order_release);
        return -EAGAIN;
    }
    uint32_t observed = kc_doorbell_observe(team->readiness);
    while (atomic_load_explicit(&team->started_members, memory_order_acquire) !=
           team->config.member_count) {
        int status = kc_doorbell_park(team->readiness, observed);
        observed = kc_doorbell_observe(team->readiness);
        if (status != 0) return status;
    }
    return 0;
}

int kc_team_dispatch(kc_team_t *team, uint64_t generation)
{
    return kc_team_dispatch_notify(team, generation, NULL, NULL);
}

int kc_team_dispatch_notify(kc_team_t *team, uint64_t generation,
                            kc_team_completion_fn completion, void *context)
{
    if (!team || generation == 0) return -EINVAL;
    if (!atomic_load_explicit(&team->started, memory_order_acquire))
        return -ECANCELED;
    int status = dispatch_enter(team);
    if (status != 0) return status;
    uint64_t dispatched = atomic_load_explicit(
        &team->dispatched_generation, memory_order_acquire);
    uint64_t retired = atomic_load_explicit(
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
    kc_doorbell_ring_all(team->dispatch);
    return 0;
}

void kc_team_request_stop(kc_team_t *team)
{
    if (!team) return;
    atomic_fetch_or_explicit(&team->dispatch_gate,
                             KC_TEAM_DISPATCH_CLOSED,
                             memory_order_acq_rel);
    atomic_store_explicit(&team->stop_requested, 1, memory_order_release);
    kc_doorbell_ring_all(team->dispatch);
    kc_doorbell_ring_all(team->readiness);
}

int kc_team_join(kc_team_t *team)
{
    if (!team) return -EINVAL;
    if (current_team == team || current_completion_team == team)
        return -EDEADLK;
    if (!atomic_load_explicit(&team->started, memory_order_acquire)) return 0;
    if (atomic_load_explicit(&team->joined, memory_order_acquire)) return 0;
    unsigned expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->joining, &expected, 1, memory_order_acq_rel,
            memory_order_acquire)) return -EBUSY;
    for (uint32_t index = 0; index < team->config.member_count; ++index)
        kc_port_thread_join(team->threads[index]);
    atomic_store_explicit(&team->joined, 1, memory_order_release);
    return 0;
}

int kc_team_destroy(kc_team_t *team)
{
    if (!team) return 0;
    if (atomic_load_explicit(&team->started, memory_order_acquire) &&
        !atomic_load_explicit(&team->joined, memory_order_acquire)) return -EBUSY;
    kc_doorbell_destroy(team->readiness);
    kc_doorbell_destroy(team->dispatch);
    kc_doorbell_destroy(team->test_fault_hold);
    free(team->progress);
    free(team->members);
    free(team->threads);
    free(team);
    return 0;
}

int kc_team_snapshot_get(kc_team_t *team, kc_team_snapshot *out)
{
    if (!team || !out || out->size < sizeof(*out)) return -EINVAL;
    *out = (kc_team_snapshot){
        .size = sizeof(*out),
        .abi_version = KC_TEAM_ABI_VERSION,
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
    if (!team || !out || out->size < sizeof(*out) || generation == 0)
        return -EINVAL;
    uint64_t dispatched = atomic_load_explicit(
        &team->dispatched_generation, memory_order_acquire);
    if (dispatched != generation) return -ESTALE;

    uint64_t entered = 0;
    uint64_t returned = 0;
    for (uint32_t index = 0; index < team->config.member_count; ++index) {
        const uint64_t bit = UINT64_C(1) << index;
        uint64_t member_returned = atomic_load_explicit(
            &team->progress[index].returned_generation, memory_order_acquire);
        if (member_returned == generation) {
            returned |= bit;
            entered |= bit;
            continue;
        }
        uint64_t member_entered = atomic_load_explicit(
            &team->progress[index].entered_generation, memory_order_acquire);
        if (member_entered == generation) entered |= bit;
    }

    if (atomic_load_explicit(&team->dispatched_generation,
                             memory_order_acquire) != generation)
        return -EAGAIN;
    const uint64_t expected = team->config.member_count == KC_TEAM_MAX_MEMBERS
                                  ? UINT64_MAX
                                  : (UINT64_C(1) << team->config.member_count) - 1;
    *out = (kc_team_quorum_snapshot){
        .size = sizeof(*out),
        .abi_version = KC_TEAM_ABI_VERSION,
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

    kc_doorbell_t *hold = NULL;
    if (kc_doorbell_create(&hold) != 0) return -ENOTSUP;
    uint64_t expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &team->test_fault_generation, &expected, UINT64_MAX,
            memory_order_acq_rel, memory_order_acquire)) {
        kc_doorbell_destroy(hold);
        return -EALREADY;
    }
    team->test_fault_hold = hold;
    team->test_fault_ready = ready;
    team->test_fault_context = context;
    atomic_store_explicit(&team->test_fault_member, member,
                          memory_order_relaxed);
    atomic_store_explicit(&team->test_fault_point, point,
                          memory_order_relaxed);
    atomic_store_explicit(&team->test_fault_settled, 0,
                          memory_order_relaxed);
    atomic_store_explicit(&team->test_fault_generation, generation,
                          memory_order_release);
    return 0;
}
