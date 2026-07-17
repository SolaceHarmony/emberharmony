#ifndef LFM_MODEL_H
#define LFM_MODEL_H

/* Compatibility tombstone.
 *
 * The synchronous numerical model ABI formerly declared by this header is a
 * private oracle/cutover seam now. Product callers open and account an opaque
 * immutable model through lfm_runtime.h, then drive it through lfm_session.h.
 * Keeping this include-only file prevents stale consumers from silently
 * acquiring tensor-, shape-, or token-level operations again.
 */
#include "lfm_runtime.h"

#endif /* LFM_MODEL_H */
