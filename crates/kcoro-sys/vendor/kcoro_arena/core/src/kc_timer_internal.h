// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_timer.h"

#include <stdatomic.h>

struct kc_timer {
    atomic_uint refs;
    kc_runtime_t *runtime;
    kc_cancel_t *cancel;
    kc_id id;
    uint64_t deadline_ns;
};
