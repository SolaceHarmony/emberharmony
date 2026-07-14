#include "lfm_kernel_bridge.h"

int lfm_kernel_protocol_c_anchor(void) {
    return (int)(sizeof(KcSubmissionV1) + sizeof(KcCompletionV1));
}
