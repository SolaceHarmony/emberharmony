// SPDX-License-Identifier: BSD-3-Clause
#pragma once

struct kc_runtime;

/* runtime->mu is held by the caller. */
void kc_service_runtime_stop_locked(struct kc_runtime *runtime);
/* runtime->mu is held by the caller. */
void kc_service_runtime_drain_realtime_locked(struct kc_runtime *runtime);
