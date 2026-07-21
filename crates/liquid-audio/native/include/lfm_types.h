#ifndef LFM_TYPES_H
#define LFM_TYPES_H

#include <stdint.h>

#include "kc_identity.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_RUNTIME_ABI_VERSION 4u
#define LFM_TEXT_COMMAND_MAX_BYTES 2048u

typedef struct LfmRuntime LfmRuntime;
typedef struct LfmModel LfmModel;
typedef struct LfmConversation LfmConversation;
typedef struct LfmSession LfmSession;

typedef enum LfmStatusV1 {
    LFM_STATUS_OK = 0,
    LFM_STATUS_INVALID_ARGUMENT = -22,
    LFM_STATUS_OUT_OF_MEMORY = -12,
    LFM_STATUS_BUSY = -16,
    LFM_STATUS_WOULD_BLOCK = -11,
    LFM_STATUS_STALE = -116,
    LFM_STATUS_CANCELLED = -125,
    LFM_STATUS_ABI_MISMATCH = -1001,
    LFM_STATUS_HOST_SINK = -1002,
    LFM_STATUS_INTERNAL = -1003,
    LFM_STATUS_UNSUPPORTED = -1004,
} LfmStatusV1;

typedef kc_ticket_id LfmTicketIdV1;

#define LFM_TICKET_SESSION KC_TICKET_KIND_SESSION
#define LFM_TICKET_TURN KC_TICKET_KIND_TURN
#define LFM_TICKET_FRAME KC_TICKET_KIND_FRAME
#define LFM_TICKET_CONTROL KC_TICKET_KIND_CONTROL
#define LFM_TICKET_DEADLINE KC_TICKET_KIND_DEADLINE

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_TYPES_H */
