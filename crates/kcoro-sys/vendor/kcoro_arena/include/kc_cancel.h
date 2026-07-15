// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_CANCEL_H
#define KC_CANCEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_cancel kc_cancel_t;

int kc_cancel_create(kc_cancel_t **out, kc_cancel_t *parent);
void kc_cancel_retain(kc_cancel_t *cancel);
void kc_cancel_release(kc_cancel_t *cancel);
void kc_cancel_trigger(kc_cancel_t *cancel);
int kc_cancel_is_triggered(const kc_cancel_t *cancel);
int kc_cancel_wait(const kc_cancel_t *cancel, long timeout_ms);
int kc_cancel_add_child(kc_cancel_t *parent, kc_cancel_t *child);
void kc_cancel_remove_child(kc_cancel_t *parent, kc_cancel_t *child);

#ifdef __cplusplus
}
#endif

#endif
