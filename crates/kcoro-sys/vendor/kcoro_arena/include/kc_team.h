// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_TEAM_H
#define KC_TEAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fixed-team execution is deliberately separate from the general continuation
 * executor.  Members never migrate or steal work.  One release-published
 * generation resumes every member, each member runs the supplied callback once,
 * and the final return completes that generation.
 */
typedef struct kc_team kc_team_t;

typedef void (*kc_team_member_fn)(void *context, uint32_t member,
                                  uint32_t members, uint64_t generation);
typedef void (*kc_team_completion_fn)(void *context, uint64_t generation);

typedef struct kc_team_config {
    uint32_t size;
    uint32_t abi_version;
    uint32_t member_count;
    uint32_t reserved;
    kc_team_member_fn member;
    void *context;
} kc_team_config;

typedef struct kc_team_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint32_t member_count;
    uint32_t started_members;
    uint64_t dispatched_generation;
    uint64_t completed_generation;
    uint32_t completed_members;
    uint32_t started;
    uint32_t stop_requested;
    uint32_t joined;
} kc_team_snapshot;

/*
 * A read-only observation of one exact dispatched generation. Team members
 * publish their own entry and return stamps; this mask is derived by scanning
 * those cache-isolated stamps and never adds a contended arrival bitmap to the
 * execution path. The requested generation must still be the team's current
 * dispatched generation. A return bit always implies the corresponding entry
 * bit.
 */
typedef struct kc_team_quorum_snapshot {
    uint32_t size;
    uint32_t abi_version;
    uint64_t generation;
    uint64_t expected_mask;
    uint64_t entered_mask;
    uint64_t returned_mask;
} kc_team_quorum_snapshot;

int kc_team_create(const kc_team_config *config, kc_team_t **out);
int kc_team_start(kc_team_t *team);
/* Exactly one generation may be active. Generations are non-zero and increasing. */
int kc_team_dispatch(kc_team_t *team, uint64_t generation);
/*
 * The completion callback runs exactly once after completed_generation is
 * release-published and the generation has retired. The callback is an edge:
 * a resumed continuation may immediately dispatch the next generation. Team
 * execution has no completion-wait API; orchestration state advances from this
 * edge instead of parking a thread on a generation.
 */
int kc_team_dispatch_notify(kc_team_t *team, uint64_t generation,
                            kc_team_completion_fn completion, void *context);
void kc_team_request_stop(kc_team_t *team);
/*
 * Terminal teardown only. The stop edge may already be published or may be
 * published by the terminal completion callback. Returns -EDEADLK from this
 * team's member or completion callback.
 */
int kc_team_join(kc_team_t *team);
int kc_team_destroy(kc_team_t *team);
int kc_team_snapshot_get(kc_team_t *team, kc_team_snapshot *out);
/*
 * Returns -ESTALE when generation is not the exact current dispatched
 * generation and -EAGAIN if a successor is dispatched during the scan.
 * Teams are bounded to 64 members so every lane has one stable mask bit.
 */
int kc_team_quorum_snapshot_get(kc_team_t *team, uint64_t generation,
                                kc_team_quorum_snapshot *out);
/* Returns zero only while the caller is executing this team's member callback. */
int kc_team_current_member(const kc_team_t *team, uint32_t *out_member);

#ifdef __cplusplus
}
#endif

#endif
