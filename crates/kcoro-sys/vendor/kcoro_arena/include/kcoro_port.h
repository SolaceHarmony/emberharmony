// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "kc_port.h"

/* Private core wrappers. The pointed-to implementations are supplied by a
 * separately linked port adapter. */
#define KC_MUTEX_T             kc_port_mutex *
#define KC_THREAD_T            kc_port_thread *

#define KC_MUTEX_INIT(m)       kc_port_mutex_create((m))
#define KC_MUTEX_DESTROY(m)    kc_port_mutex_destroy(*(m))
#define KC_MUTEX_LOCK(m)       kc_port_mutex_lock(*(m))
#define KC_MUTEX_UNLOCK(m)     kc_port_mutex_unlock(*(m))
