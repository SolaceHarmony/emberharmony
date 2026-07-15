// SPDX-License-Identifier: BSD-3-Clause
#ifndef KC_STORE_H
#define KC_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque storage supplied by the host. The core intentionally provides no
 * implementation of these direct link-time functions. */
typedef struct kc_store_handle kc_store_handle;

int kc_store_size(kc_store_handle *store, uint64_t *size);
int kc_store_read(kc_store_handle *store, uint64_t offset, void *data,
                  size_t length);
int kc_store_append(kc_store_handle *store, const void *data, size_t length,
                    uint64_t *offset);
int kc_store_sync(kc_store_handle *store);
int kc_store_truncate(kc_store_handle *store, uint64_t size);

#ifdef __cplusplus
}
#endif

#endif
