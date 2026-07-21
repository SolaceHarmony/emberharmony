// SPDX-License-Identifier: BSD-3-Clause
#include "kcoro_stackless.h"

#include "kc_runtime_internal.h"
#include "koro_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static int ticket_equal(const kc_ticket_id *left, const kc_ticket_id *right)
{
    return left && right &&
           left->runtime_epoch == right->runtime_epoch &&
           left->sequence == right->sequence &&
           left->generation == right->generation &&
           left->kind == right->kind;
}

int koro_cont_create_on(kc_runtime_t *runtime,
                        const koro_cont_config *config,
                        koro_cont_t **out)
{
    if (!runtime || !config || !out ||
        config->size < sizeof(*config) ||
        config->abi_version != KC_ABI_VERSION || !config->step) return -EINVAL;
    if (config->worker_mask &&
        (runtime->worker_count < 64 &&
         (config->worker_mask >> runtime->worker_count) != 0)) return -EINVAL;

    koro_cont_t *continuation = calloc(1, sizeof(*continuation));
    if (!continuation) return -ENOMEM;
    if (config->frame_size) {
        continuation->frame = calloc(1, config->frame_size);
        if (!continuation->frame) {
            free(continuation);
            return -ENOMEM;
        }
    }
    continuation->frame_size = config->frame_size;
    continuation->next_step = config->step;
    continuation->user_arg = config->argument;
    continuation->runtime = runtime;
    continuation->worker_mask = config->worker_mask;
    continuation->completion = config->completion;
    continuation->completion_context = config->completion_context;
    continuation->slot = UINT32_MAX;
    continuation->suspend_kind = KORO_SUSPEND_CALLBACK;
    atomic_init(&continuation->run_state, KORO_NEW);
    atomic_init(&continuation->wake_pending, 0);
    atomic_init(&continuation->refs, 1);
    atomic_init(&continuation->current_worker, UINT32_MAX);

    const int status = kc_runtime_register_continuation_internal(
        runtime, continuation);
    if (status != 0) {
        free(continuation->frame);
        free(continuation);
        return status;
    }
    kc_runtime_retain_internal(runtime);
    *out = continuation;
    return 0;
}

void koro_cont_retain_internal(koro_cont_t *continuation)
{
    if (continuation)
        atomic_fetch_add_explicit(&continuation->refs, 1,
                                  memory_order_relaxed);
}

void koro_cont_release_internal(koro_cont_t *continuation)
{
    if (!continuation ||
        atomic_fetch_sub_explicit(&continuation->refs, 1,
                                  memory_order_acq_rel) != 1) return;
    kc_runtime_release_internal(continuation->runtime);
    free(continuation->frame);
    free(continuation);
}

void koro_cont_retain(koro_cont_t *continuation)
{
    koro_cont_retain_internal(continuation);
}

void koro_cont_release(koro_cont_t *continuation)
{
    koro_cont_release_internal(continuation);
}

int koro_cont_start_internal(koro_cont_t *continuation)
{
    return kc_runtime_start_continuation_internal(continuation);
}

int koro_cont_start(koro_cont_t *continuation)
{
    return continuation ? koro_cont_start_internal(continuation) : -EINVAL;
}

int koro_cont_resume_internal(koro_cont_t *continuation)
{
    return kc_runtime_resume_continuation_internal(continuation);
}

int koro_cont_resume(koro_cont_t *continuation,
                     const kc_ticket_id *identity)
{
    if (!continuation || !identity) return -EINVAL;
    if (!ticket_equal(identity, &continuation->identity)) return -ESTALE;
    return koro_cont_resume_internal(continuation);
}

kc_ticket_id koro_cont_identity(const koro_cont_t *continuation)
{
    return continuation ? continuation->identity : (kc_ticket_id){0};
}

int koro_cont_destroy(koro_cont_t *continuation)
{
    if (!continuation) return 0;
    const int state = atomic_load_explicit(&continuation->run_state,
                                           memory_order_acquire);
    if (state != KORO_NEW && state != KORO_DONE) return -EBUSY;
    const int status = kc_runtime_unregister_continuation_internal(
        continuation->runtime, continuation);
    if (status != 0) return status;
    koro_cont_release_internal(continuation);
    return 0;
}

void *koro_cont_frame(koro_cont_t *continuation)
{
    return continuation ? continuation->frame : NULL;
}

void *koro_cont_argument(koro_cont_t *continuation)
{
    return continuation ? continuation->user_arg : NULL;
}

uint32_t koro_cont_current_worker(const koro_cont_t *continuation)
{
    return continuation
        ? atomic_load_explicit(&continuation->current_worker,
                               memory_order_acquire)
        : UINT32_MAX;
}

uint32_t koro_cont_state_get(const koro_cont_t *continuation)
{
    return continuation ? continuation->state : 0;
}

void koro_cont_state_set(koro_cont_t *continuation, uint32_t state,
                         uint32_t suspend_kind)
{
    if (!continuation ||
        (suspend_kind != KORO_SUSPEND_CALLBACK &&
         suspend_kind != KORO_SUSPEND_YIELD)) abort();
    continuation->state = state;
    continuation->suspend_kind = suspend_kind;
}

void koro_cont_finish(koro_cont_t *continuation)
{
    if (!continuation) return;
    continuation->state = 0;
    continuation->completed = 1;
}
