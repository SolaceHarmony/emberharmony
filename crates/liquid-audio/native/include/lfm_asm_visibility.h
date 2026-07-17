#ifndef LFM_ASM_VISIBILITY_H
#define LFM_ASM_VISIBILITY_H

/* Assembly leaves must be globally resolvable while the static native archives
 * are linked, but they are not product ABI. Oracle builds keep their historical
 * default visibility for direct parity tests. */
#if defined(LFM_BUILD_ORACLE) || defined(_WIN32)
#define LFM_PRIVATE(name)
#elif defined(__APPLE__)
#define LFM_PRIVATE(name) .private_extern name
#else
#define LFM_PRIVATE(name) .hidden name
#endif

#endif /* LFM_ASM_VISIBILITY_H */
