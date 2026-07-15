// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <errno.h>
#include <stdlib.h>

#include "kc_port.h"

/* Compatibility macros for the active core. The pointed-to implementations
 * are supplied by a separately linked port adapter. */
#define KC_MUTEX_T             kc_port_mutex *
#define KC_COND_T              kc_port_cond *
#define KC_THREAD_T            kc_port_thread *

#define KC_MUTEX_INIT(m)       kc_port_mutex_create((m))
#define KC_MUTEX_DESTROY(m)    kc_port_mutex_destroy(*(m))
#define KC_MUTEX_LOCK(m)       kc_port_mutex_lock(*(m))
#define KC_MUTEX_UNLOCK(m)     kc_port_mutex_unlock(*(m))

#define KC_COND_INIT(c)        kc_port_cond_create((c))
#define KC_COND_DESTROY(c)     kc_port_cond_destroy(*(c))
#define KC_COND_WAIT(c,m)      kc_port_cond_wait(*(c), *(m))
#define KC_COND_TIMEDWAIT_NS(c,m,deadline) \
    kc_port_cond_timedwait(*(c), *(m), (deadline))
#define KC_COND_SIGNAL(c)      kc_port_cond_signal(*(c))
#define KC_COND_BROADCAST(c)   kc_port_cond_broadcast(*(c))

#define KC_ALLOC(n)            malloc((n))
#define KC_FREE(p)             free((p))

#define KC_EAGAIN              (-EAGAIN)
#define KC_EPIPE               (-EPIPE)
#define KC_ETIME               (-ETIMEDOUT)
#define KC_ECANCELED           (-ECANCELED)
