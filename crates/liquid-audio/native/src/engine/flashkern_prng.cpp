#include "flashkern_prng.h"

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__linux__)
#include <sys/random.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <bcrypt.h>
#include <windows.h>
#else
#include <cstdlib>
#endif

namespace {

constexpr uint32_t CHACHA_CONSTANTS[4] = {
    0x61707865u,
    0x3320646eu,
    0x79622d32u,
    0x6b206574u,
};

static uint32_t load_le32(const uint8_t *src) {
    return (uint32_t)src[0] | ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 24);
}

static void store_le64(uint8_t *dst, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) dst[i] = (uint8_t)(value >> (i * 8));
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static bool state_valid(const LfmPrngStateV1 *state) {
    if (!state || state->size != sizeof(*state) ||
        state->abi_version != LFM_PRNG_ABI_VERSION ||
        state->cursor > LFM_PRNG_BLOCK_BYTES || (state->cursor & 7u) != 0) {
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (state->core[i] != CHACHA_CONSTANTS[i]) return false;
    }
    return true;
}

static void seed_material(LfmPrngStateV1 *state, const uint8_t *key,
                          const uint8_t *nonce, uint32_t flags) {
    std::memset(state, 0, sizeof(*state));
    state->size = sizeof(*state);
    state->abi_version = LFM_PRNG_ABI_VERSION;
    state->cursor = LFM_PRNG_BLOCK_BYTES;
    state->flags = flags;
    for (size_t i = 0; i < 4; ++i) state->core[i] = CHACHA_CONSTANTS[i];
    for (size_t i = 0; i < 8; ++i) state->core[4 + i] = load_le32(key + i * 4);
    state->core[12] = 0;
    state->core[13] = 0;
    state->core[14] = load_le32(nonce);
    state->core[15] = load_le32(nonce + 4);
}

#if defined(__APPLE__)
extern "C" int lfm_apple_secure_random(void *bytes, size_t count);

static int system_entropy(void *bytes, size_t count) {
    return lfm_apple_secure_random(bytes, count) == 0 ? 0 : -EIO;
}
#elif defined(__linux__)
static int system_entropy(void *bytes, size_t count) {
    auto *dst = static_cast<uint8_t *>(bytes);
    size_t offset = 0;
    while (offset < count) {
        ssize_t got = getrandom(dst + offset, count - offset, 0);
        if (got > 0) {
            offset += (size_t)got;
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        return got < 0 ? -errno : -EIO;
    }
    return 0;
}
#elif defined(_WIN32)
static int system_entropy(void *bytes, size_t count) {
    if (count > ULONG_MAX) return -EOVERFLOW;
    NTSTATUS rc = BCryptGenRandom(nullptr, static_cast<PUCHAR>(bytes),
                                  static_cast<ULONG>(count),
                                  BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return rc == 0 ? 0 : -EIO;
}
#else
static int system_entropy(void *bytes, size_t count) {
    arc4random_buf(bytes, count);
    return 0;
}
#endif

static void erase(void *memory, size_t size) {
    volatile uint8_t *bytes = static_cast<volatile uint8_t *>(memory);
    while (size-- > 0) *bytes++ = 0;
}

} // namespace

extern "C" int lfm_prng_seed_material(LfmPrngStateV1 *state,
                                        const uint8_t *key,
                                        const uint8_t *nonce) {
    if (!state || !key || !nonce) return -EINVAL;
    seed_material(state, key, nonce, 0);
    return 0;
}

extern "C" int lfm_prng_seed_u64(LfmPrngStateV1 *state, uint64_t seed) {
    if (!state) return -EINVAL;
    alignas(8) uint8_t material[40];
    uint64_t stream = seed;
    for (size_t i = 0; i < 5; ++i) {
        store_le64(material + i * 8, splitmix64(&stream));
    }
    seed_material(state, material, material + 32, 0);
    erase(material, sizeof(material));
    return 0;
}

extern "C" int lfm_prng_seed_system(LfmPrngStateV1 *state) {
    if (!state) return -EINVAL;
    alignas(16) uint8_t entropy[40];
    int rc = system_entropy(entropy, sizeof(entropy));
    if (rc == 0) {
        seed_material(state, entropy, entropy + 32, LFM_PRNG_FLAG_SYSTEM_SEEDED);
    }
    erase(entropy, sizeof(entropy));
    return rc;
}

extern "C" int lfm_prng_fill_u64(LfmPrngStateV1 *state, uint64_t *out,
                                   size_t count) {
    if (!state_valid(state) || (!out && count != 0)) return -EINVAL;

    for (size_t i = 0; i < count; ++i) {
        if (state->cursor > LFM_PRNG_BLOCK_BYTES - sizeof(uint64_t)) {
            if ((state->flags & LFM_PRNG_FLAG_EXHAUSTED) != 0) return -EOVERFLOW;
            lfm_chacha20_block(state->core, state->block);
            uint64_t counter = (uint64_t)state->core[12] |
                               ((uint64_t)state->core[13] << 32);
            if (counter == UINT64_MAX) {
                state->flags |= LFM_PRNG_FLAG_EXHAUSTED;
            } else {
                ++counter;
                state->core[12] = (uint32_t)counter;
                state->core[13] = (uint32_t)(counter >> 32);
            }
            state->cursor = 0;
        }
        size_t word = state->cursor / sizeof(uint32_t);
        out[i] = (uint64_t)state->block[word] |
                 ((uint64_t)state->block[word + 1] << 32);
        state->cursor += sizeof(uint64_t);
    }
    return 0;
}
