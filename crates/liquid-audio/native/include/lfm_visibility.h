#ifndef LFM_VISIBILITY_H
#define LFM_VISIBILITY_H

/* Product builds publish only the opaque runtime/session lifecycle. Numerical
 * and checkpoint-view entry points remain linkable between native archives but
 * are not dynamic exports. The offline oracle deliberately restores default
 * visibility for its parity-only C ABI. */
#if defined(__GNUC__) || defined(__clang__)
#define LFM_DEFAULT_VISIBILITY __attribute__((visibility("default")))
#define LFM_HIDDEN_VISIBILITY __attribute__((visibility("hidden")))
#else
#define LFM_DEFAULT_VISIBILITY
#define LFM_HIDDEN_VISIBILITY
#endif

#if defined(LFM_BUILD_ORACLE)
#define LFM_ORACLE_API LFM_DEFAULT_VISIBILITY
#else
#define LFM_ORACLE_API LFM_HIDDEN_VISIBILITY
#endif

#define LFM_INTERNAL_API LFM_HIDDEN_VISIBILITY
#define LFM_PUBLIC_API LFM_DEFAULT_VISIBILITY

#endif /* LFM_VISIBILITY_H */
