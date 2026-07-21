#ifndef LFM_RUNTIME_INTERNAL_H
#define LFM_RUNTIME_INTERNAL_H

#include "kc_runtime.h"
#include "lfm_frontend.h"
#include "lfm_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Native adapters and native-only integration gates mount their retained
 * continuations on the model runtime's bounded worker pool. This is deliberately
 * private: product hosts receive opaque lifecycle handles, never an executor. */
kc_runtime_t *lfm_internal_runtime_coordination(LfmRuntime *runtime);

/* Native model-to-model PCM handoff. This is not a capture-device seam:
 * `spans` is a retained read-only view over PCM already emitted by another
 * native conversation. The descriptor is copied into the bounded session
 * command ring; the pointed storage must remain alive through the correlated
 * TURN_STARTED edge, which is published only after native PCM admission has
 * consumed the view. `parent` records the source turn that sealed the PCM and
 * `out_ticket` names the destination continuation. No detector, timer, or
 * fabricated silence participates in this transition. */
int lfm_internal_session_submit_pcm_spans(
    LfmSession *session, const LfmF32Span *spans, uint32_t span_count,
    uint32_t sample_rate, const LfmTicketIdV1 *parent,
    LfmTicketIdV1 *out_ticket);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_RUNTIME_INTERNAL_H */
