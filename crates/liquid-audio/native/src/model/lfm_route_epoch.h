#ifndef LFM_ROUTE_EPOCH_H
#define LFM_ROUTE_EPOCH_H

#include <atomic>
#include <cstdint>

/* Private session publication epoch shared with the exact-CQ route callback.
 * A callback may only acquire-load this word; session control owns stores. */
struct alignas(128) LfmRouteEpoch {
    std::atomic<uint64_t> value{1};
    uint8_t padding[120]{};

    uint64_t load(std::memory_order order) const noexcept {
        return value.load(order);
    }
    void store(uint64_t next, std::memory_order order) noexcept {
        value.store(next, order);
    }
};
static_assert(sizeof(LfmRouteEpoch) == 128);
static_assert(alignof(LfmRouteEpoch) == 128);

#endif /* LFM_ROUTE_EPOCH_H */
