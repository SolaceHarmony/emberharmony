#ifndef LFM_TYPES_H
#define LFM_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_RUNTIME_ABI_VERSION 1u
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
} LfmStatusV1;

typedef struct LfmTicketIdV1 {
    uint64_t runtime_epoch;
    uint64_t sequence;
    uint32_t generation;
    uint32_t kind;
} LfmTicketIdV1;

#define LFM_TICKET_SESSION 1u
#define LFM_TICKET_TURN 2u
#define LFM_TICKET_FRAME 3u
#define LFM_TICKET_CONTROL 4u

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LFM_TYPES_H */
