#ifndef LFM_RUNTIME_DIAGNOSTICS_HPP
#define LFM_RUNTIME_DIAGNOSTICS_HPP

#include "kc_runtime.h"
#include "kc_service.h"
#include "kc_team.h"
#include "kcoro_stackless.h"
#include "lfm_runtime.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

/*
 * Private native truth-gate instrumentation.  This is deliberately not a C
 * ABI and is never visible to Rust or a product client.  Owners retain the
 * records below from setup through retirement.  A diagnostic consumer receives
 * pointers into those records plus views of the existing kcoro objects; it
 * does not receive copied model state, PCM, weights, or numerical planes.
 */

struct alignas(128) LfmEngineDiagnosticState {
    std::atomic<uint64_t> publications{0};
    std::atomic<uint64_t> bridge_activations{0};
    std::atomic<uint64_t> team_completion_edge{0};
    std::atomic<uint64_t> team_completion_consumed{0};
    std::atomic<uint64_t> route_callbacks{0};
    std::atomic<uint64_t> team_generation{0};
    std::atomic<uint32_t> bridge_phase{0};
    std::atomic<uint32_t> request{0};
    std::atomic<uint32_t> stage{0};
    std::atomic<uint32_t> program_phase{0};
    std::atomic<uint32_t> bridge_valid{0};
    std::atomic<int32_t> active_status{0};
};

struct alignas(128) LfmSessionDiagnosticState {
    std::atomic<uint64_t> publications{0};
    std::atomic<uint64_t> action_ticket_sequence{0};
    std::atomic<uint64_t> action_route_sequence{0};
    std::atomic<uint64_t> event_depth{0};
    std::atomic<uint64_t> command_depth{0};
    std::atomic<uint64_t> pcm_depth{0};
    std::atomic<uint32_t> progress{0};
    std::atomic<uint32_t> coordinator_phase{0};
    std::atomic<uint32_t> action_phase{0};
    std::atomic<uint32_t> action_active{0};
    std::atomic<uint32_t> admission_pending{0};
    std::atomic<uint32_t> route_pending{0};
    std::atomic<uint32_t> playback_active{0};
    std::atomic<uint32_t> result_active{0};
    std::atomic<uint32_t> result_next{0};
    std::atomic<uint32_t> result_count{0};
    std::atomic<uint32_t> delivery_pending{0};
    std::atomic<uint32_t> stop{0};
    std::atomic<uint32_t> event_done{0};
    std::atomic<uint32_t> conversation_operation{0};
    std::atomic<uint32_t> terminal_cause{0};
    std::atomic<uint32_t> terminal_operation{0};
    std::atomic<int32_t> terminal_status{0};
};

struct LfmEngineDiagnosticCounts {
    uint64_t pass_submissions = 0;
    uint64_t pass_completions = 0;
    uint64_t bridge_dispatches = 0;
    uint64_t dispatch_wakes = 0;
    uint64_t route_dispatches = 0;
    uint64_t route_admission_deferrals = 0;
    uint64_t bridge_team_generation = 0;
    uint64_t bridge_team_completion = 0;
    uint64_t bridge_retired_generation = 0;
    uint64_t mailbox_requests_published = 0;
    uint64_t mailbox_requests_consumed = 0;
    uint64_t mailbox_completions_published = 0;
    uint64_t mailbox_completions_consumed = 0;
    uint64_t gang_lease = 0;
    uint64_t team_terminal = 0;
    uint32_t pass_slots_live = 0;
    uint32_t routes_free = 0;
    uint32_t routes_claimed = 0;
    uint32_t routes_ready = 0;
    uint32_t routes_dispatching = 0;
    uint32_t routes_running = 0;
    uint32_t routes_done = 0;
};

struct LfmEngineDiagnosticView {
    const LfmEngineDiagnosticState *state = nullptr;
    kc_runtime_t *runtime = nullptr;
    koro_cont_t *bridge_continuation = nullptr;
    kc_service_t *route_service = nullptr;
    kc_service_t *supervisor_service = nullptr;
    kc_team_t *team = nullptr;
    void *owner = nullptr;
};

struct LfmSessionDiagnosticView {
    const LfmSessionDiagnosticState *state = nullptr;
    kc_service_t *coordinator = nullptr;
    kc_service_t *delivery = nullptr;
};

struct LfmRuntimeDiagnosticView {
    kc_runtime_t *coordination = nullptr;
    LfmEngineDiagnosticView engine{};
    LfmSessionDiagnosticView first{};
    LfmSessionDiagnosticView second{};
};

int lfm_internal_engine_diagnostic_view(void *engine,
                                        LfmEngineDiagnosticView *out);
int lfm_internal_engine_diagnostic_counts(
    const LfmEngineDiagnosticView *view, LfmEngineDiagnosticCounts *out);
int lfm_internal_runtime_diagnostic_view(LfmRuntime *runtime,
                                         LfmSession *first,
                                         LfmSession *second,
                                         LfmRuntimeDiagnosticView *out);

#endif /* LFM_RUNTIME_DIAGNOSTICS_HPP */
