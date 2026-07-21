#include "lfm_audio_dock.h"
#include "lfm_model.h"
#include "lfm_runtime.h"
#include "lfm_session.h"
#include "lfm_types.h"

#include <stddef.h>

_Static_assert(sizeof(LfmTicketIdV1) == 24, "LfmTicketIdV1 ABI");
_Static_assert(sizeof(LfmRuntimeConfigV1) == 72, "LfmRuntimeConfigV1 ABI");
_Static_assert(sizeof(LfmRuntimeSnapshotV1) == 64, "LfmRuntimeSnapshotV1 ABI");
_Static_assert(sizeof(LfmModelMemoryV1) == 168, "LfmModelMemoryV1 ABI");
_Static_assert(offsetof(LfmModelMemoryV1,
                        post_readiness_allocation_attempts) == 136,
               "LfmModelMemoryV1 allocation attempts offset");
_Static_assert(offsetof(LfmModelMemoryV1,
                        post_readiness_allocation_bytes) == 144,
               "LfmModelMemoryV1 allocation bytes offset");
_Static_assert(sizeof(LfmSamplingPolicyV1) == 32, "LfmSamplingPolicyV1 ABI");
_Static_assert(sizeof(LfmConversationOptionsV1) == 120,
               "LfmConversationOptionsV1 ABI");
_Static_assert(sizeof(LfmSessionConfigV1) == 88, "LfmSessionConfigV1 ABI");
_Static_assert(sizeof(LfmTurnEventV1) == 16, "LfmTurnEventV1 ABI");
_Static_assert(sizeof(LfmEventV1) == 72, "LfmEventV1 ABI");
_Static_assert(offsetof(LfmEventV1, ticket) == 32, "LfmEventV1 ticket offset");
_Static_assert(sizeof(LfmCallbacksV1) == 24, "LfmCallbacksV1 ABI");
_Static_assert(sizeof(LfmSessionSnapshotV1) == 168, "LfmSessionSnapshotV1 ABI");
_Static_assert(sizeof(LfmPcmLeaseV1) == 88, "LfmPcmLeaseV1 ABI");
_Static_assert(offsetof(LfmPcmLeaseV1, ticket) == 32,
               "LfmPcmLeaseV1 ticket offset");
