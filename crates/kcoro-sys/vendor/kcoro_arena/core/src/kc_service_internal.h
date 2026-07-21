// SPDX-License-Identifier: BSD-3-Clause
#pragma once

struct kc_runtime;
struct kc_service;
struct koro_cont;

/* runtime->mu is held by the caller. */
void kc_service_runtime_stop_locked(struct kc_runtime *runtime);
struct koro_cont *kc_service_continuation_internal(struct kc_service *service);
