#ifndef FLASHKERN_PRNG_H
#define FLASHKERN_PRNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define LFM_PRNG_STATIC_ASSERT(test, message) static_assert(test, message)
#define LFM_PRNG_ALIGNOF(type) alignof(type)
#else
#define LFM_PRNG_STATIC_ASSERT(test, message) _Static_assert(test, message)
#define LFM_PRNG_ALIGNOF(type) _Alignof(type)
#endif

#define LFM_PRNG_ABI_VERSION 1u
#define LFM_PRNG_BLOCK_BYTES 64u
#define LFM_PRNG_FLAG_SYSTEM_SEEDED 1u
#define LFM_PRNG_FLAG_EXHAUSTED 2u

/*
 * Conversation-owned ChaCha20 stream state. The core uses the original
 * 64-bit-counter/64-bit-nonce layout:
 *
 *   core[0..4]   constants
 *   core[4..12]  256-bit key
 *   core[12..14] 64-bit block counter, little endian
 *   core[14..16] 64-bit nonce
 *
 * `block` and `cursor` preserve partially consumed output. The complete object
 * is pointer-free and may be copied into a quiescent conversation snapshot.
 */
typedef struct __attribute__((aligned(64))) LfmPrngStateV1 {
    uint32_t size;
    uint32_t abi_version;
    uint32_t cursor;
    uint32_t flags;
    uint32_t core[16];
    uint32_t block[16];
    uint8_t reserved[48];
} LfmPrngStateV1;

/* Seed from the platform CSPRNG. Apple uses SecRandomCopyBytes through the
 * architecture assembly thunk; supported non-Apple hosts use their kernel RNG. */
int lfm_prng_seed_system(LfmPrngStateV1 *state);

/* Deterministic conformance/replay seed. `key` is 32 bytes and `nonce` is 8
 * bytes. Production entropy should normally use lfm_prng_seed_system. */
int lfm_prng_seed_material(LfmPrngStateV1 *state, const uint8_t *key,
                           const uint8_t *nonce);

/* Fill caller-owned output and advance `state` in place. ChaCha block expansion
 * is implemented by the selected architecture assembly kernel. */
int lfm_prng_fill_u64(LfmPrngStateV1 *state, uint64_t *out, size_t count);

/* Architecture assembly leaf. Exposed for ABI/link tests, not model policy. */
void lfm_chacha20_block(const uint32_t input[16], uint32_t output[16]);

LFM_PRNG_STATIC_ASSERT(sizeof(LfmPrngStateV1) == 192,
                       "LfmPrngStateV1 must remain snapshot-stable");
LFM_PRNG_STATIC_ASSERT(LFM_PRNG_ALIGNOF(LfmPrngStateV1) == 64,
                       "LfmPrngStateV1 must remain cache-line aligned");

#undef LFM_PRNG_STATIC_ASSERT
#undef LFM_PRNG_ALIGNOF

#ifdef __cplusplus
}
#endif

#endif /* FLASHKERN_PRNG_H */
