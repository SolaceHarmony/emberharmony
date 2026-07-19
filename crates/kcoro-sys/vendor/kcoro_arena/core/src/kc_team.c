// SPDX-License-Identifier: BSD-3-Clause
#include "kc_team.h"

#include "kc_doorbell.h"
#include "kc_port.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

enum { KC_TEAM_ABI_VERSION = 1u };

typedef struct kc_team_member {
    struct kc_team *team;
    uint32_t index;
} kc_team_member;

struct kc_team {
    kc_team_config config;
    kc_port_thread **threads;
    kc_team_member *members;
    kc_doorbell_t *dispatch;
    kc_doorbell_t *completion;
    atomic_uint started_members;
    atomic_uint completed_members;
    atomic_uint started;
    atomic_uint stop_requested;
    atomic_uint joined;
    atomic_uint_fast64_t dispatched_generation;
    atomic_uint_fast64_t completed_generation;
};

static _Thread_local kc_team_t *current_team;
static _Thread_local uint32_t current_member;

static void *team_member_main(void *context)
{
    kc_team_member *member = context;
    kc_team_t *team = member->team;
    uint64_t seen = 0;
    uint32_t observed = kc_doorbell_observe(team->dispatch);
    atomic_fetch_add_explicit(&team->started_members, 1, memory_order_release);
    kc_doorbell_ring_all(team->completion);

    for (;;) {
        uint64_t generation = atomic_load_explicit(
            &team->dispatched_generation, memory_order_acquire);
        while (generation == seen &&
               !atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire)) {
            int status = kc_doorbell_wait(team->dispatch, observed, 0);
            observed = kc_doorbell_observe(team->dispatch);
            if (status != 0 &&
                atomic_load_explicit(&team->stop_requested,
                                     memory_order_acquire)) return NULL;
            if (status != 0) abort();
            generation = atomic_load_explicit(
                &team->dispatched_generation, memory_order_acquire);
        }
        if (atomic_load_explicit(&team->stop_requested, memory_order_acquire))
            return NULL;
        seen = generation;
        current_team = team;
        current_member = member->index;
        team->config.member(team->config.context, member->index,
                            team->config.member_count, generation);
        current_team = NULL;
        if (atomic_fetch_add_explicit(&team->completed_members, 1,
                                      memory_order_acq_rel) + 1 ==
            team->config.member_count) {
            atomic_store_explicit(&team->completed_generation, generation,
                                  memory_order_release);
            kc_doorbell_ring_all(team->completion);
        }
    }
}

int kc_team_create(const kc_team_config *config, kc_team_t **out)
{
    if (!config || !out || config->size < sizeof(*config) ||
        config->abi_version != KC_TEAM_ABI_VERSION || !config->member ||
        config->member_count == 0 || config->reserved != 0) return -EINVAL;
    kc_team_t *team = calloc(1, sizeof(*team));
    if (!team) return -ENOMEM;
    team->config = *config;
    team->threads = calloc(config->member_count, sizeof(*team->threads));
    team->members = calloc(config->member_count, sizeof(*team->members));
    if (!team->threads || !team->members) {
        free(team->members);
        free(team->threads);
        free(team);
        return -ENOMEM;
    }
    atomic_init(&team->started_members, 0);
    atomic_init(&team->completed_members, 0);
    atomic_init(&team->started, 0);
    atomic_init(&team->stop_requested, 0);
    atomic_init(&team->joined, 0);
    atomic_init(&team->dispatched_generation, 0);
    atomic_init(&team->completed_generation, 0);
    if (kc_doorbell_create(&team->dispatch) != 0 ||
        kc_doorbell_create(&team->completion) != 0) {
        kc_doorbell_destroy(team->dispatch);
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
    uint32_t started = 0;
    for (; started < team->config.member_count; ++started) {
        team->members[started] = (kc_team_member){
            .team = team,
            .index = started,
        };
        int status = kc_port_thread_create(&team->threads[started],
                                           team_member_main,
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
    uint32_t observed = kc_doorbell_observe(team->completion);
    while (atomic_load_explicit(&team->started_members, memory_order_acquire) !=
           team->config.member_count) {
        int status = kc_doorbell_wait(team->completion, observed, 0);
        observed = kc_doorbell_observe(team->completion);
        if (status != 0) return status;
    }
    return 0;
}

int kc_team_dispatch(kc_team_t *team, uint64_t generation)
{
    if (!team || generation == 0) return -EINVAL;
    if (!atomic_load_explicit(&team->started, memory_order_acquire) ||
        atomic_load_explicit(&team->stop_requested, memory_order_acquire))
        return -ECANCELED;
    uint64_t dispatched = atomic_load_explicit(
        &team->dispatched_generation, memory_order_acquire);
    uint64_t completed = atomic_load_explicit(
        &team->completed_generation, memory_order_acquire);
    if (dispatched != completed) return -EBUSY;
    if (generation <= dispatched) return -EINVAL;
    atomic_store_explicit(&team->completed_members, 0, memory_order_relaxed);
    atomic_store_explicit(&team->dispatched_generation, generation,
                          memory_order_release);
    kc_doorbell_ring_all(team->dispatch);
    return 0;
}

int kc_team_wait(kc_team_t *team, uint64_t generation, uint64_t deadline_ns)
{
    if (!team || generation == 0) return -EINVAL;
    uint32_t observed = kc_doorbell_observe(team->completion);
    for (;;) {
        uint64_t completed = atomic_load_explicit(
            &team->completed_generation, memory_order_acquire);
        if (completed >= generation) return completed == generation ? 0 : -ESTALE;
        if (atomic_load_explicit(&team->stop_requested, memory_order_acquire))
            return -ECANCELED;
        int status = kc_doorbell_wait(team->completion, observed, deadline_ns);
        observed = kc_doorbell_observe(team->completion);
        if (status != 0) return status;
    }
}

void kc_team_request_stop(kc_team_t *team)
{
    if (!team) return;
    atomic_store_explicit(&team->stop_requested, 1, memory_order_release);
    kc_doorbell_ring_all(team->dispatch);
    kc_doorbell_ring_all(team->completion);
}

int kc_team_join(kc_team_t *team)
{
    if (!team) return -EINVAL;
    if (!atomic_load_explicit(&team->started, memory_order_acquire)) return 0;
    if (!atomic_load_explicit(&team->stop_requested, memory_order_acquire))
        return -EBUSY;
    if (atomic_exchange_explicit(&team->joined, 1, memory_order_acq_rel))
        return 0;
    for (uint32_t index = 0; index < team->config.member_count; ++index)
        kc_port_thread_join(team->threads[index]);
    return 0;
}

int kc_team_destroy(kc_team_t *team)
{
    if (!team) return 0;
    if (atomic_load_explicit(&team->started, memory_order_acquire) &&
        !atomic_load_explicit(&team->joined, memory_order_acquire)) return -EBUSY;
    kc_doorbell_destroy(team->completion);
    kc_doorbell_destroy(team->dispatch);
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

int kc_team_current_member(const kc_team_t *team, uint32_t *out_member)
{
    if (!team || !out_member) return -EINVAL;
    if (current_team != team) return -EPERM;
    *out_member = current_member;
    return 0;
}
