// register_cache_chain.cpp -- ground-truth probe for chained numerical work
// carried in NEON registers versus deliberately materialized intermediates.
//
// This is representative arithmetic, not the full LFM2 model. One invocation
// executes RMS normalization, a multiplicative gate, causal ShortConv-3, and a
// four-row projection over 16 F32 activations. The fused AArch64 leaf keeps all
// intermediates in caller-clobbered v0-v7/v16-v31 and writes only the four
// terminal projection values.
// The immediate controls execute identical instructions but store/reload three
// 64-byte stage planes through either one hot stack frame or rotating caller
// scratch. A second batch control delays each reuse across a complete plane so
// footprint and reuse distance can be swept independently of memcpy.
// The optimized batch contestant holds four independent tiles in register
// banks, interleaving their norm dependency chains before rotating each bank
// through the remaining stages without per-tile spills.
//
// Build from crates/liquid-audio/native/bench:
//
//   clang++ -O3 -std=c++23 -Wall -Wextra -Wpedantic -Werror \
//     -ffp-contract=off -mcpu=apple-m2 register_cache_chain.cpp \
//     register_cache_chain_aarch64.S -o /tmp/register_cache_chain
//
// The timed regions allocate nothing and perform no memcpy. Arena sizes are
// working-set labels, not proof of physical cache residency; use hardware
// counters before attributing a timing knee to a particular cache level.

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include <mach/mach_time.h>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "register_cache_chain requires native Apple AArch64."
#endif

namespace {

constexpr std::size_t kWidth = 16;
constexpr std::size_t kRows = 4;
constexpr std::size_t kScratch = 48;
constexpr std::size_t kOuts = 256;
constexpr std::size_t kCalls = 1u << 20;
constexpr std::size_t kRounds = 9;
constexpr std::uint32_t kPre = 0x13579bdfu;
constexpr std::uint32_t kPost = 0xfdb97531u;

struct alignas(32) Params {
    float eps;
    float inv;
    float one;
    float c0;
    float c1;
    float c2;
    float carry2;
    float carry1;
};

constexpr std::size_t kFixed = kWidth * sizeof(float) * 2 +
                               kRows * kWidth * sizeof(float) +
                               sizeof(Params) + kOuts * kRows * sizeof(float);
constexpr std::size_t kBatchFixed = kWidth * sizeof(float) * 2 +
                                    kRows * kWidth * sizeof(float) +
                                    sizeof(Params);
constexpr std::size_t kPlaneBytes = kWidth * sizeof(float);
constexpr std::size_t kTileBytes = kPlaneBytes * 3 +
                                   kRows * sizeof(float);

static_assert(offsetof(Params, inv) == 4);
static_assert(offsetof(Params, c0) == 12);
static_assert(offsetof(Params, carry2) == 24);

struct alignas(128) Slot {
    std::array<std::uint32_t, 8> pre;
    std::array<float, kScratch> data;
    std::array<std::uint32_t, 8> post;
};

static_assert(sizeof(Slot) == 256);
static_assert(offsetof(Slot, data) == 32);

struct alignas(128) Output {
    std::array<std::uint32_t, 8> pre;
    std::array<float, kRows> data;
    std::array<std::uint32_t, 8> post;
};

struct Batch {
    float *norm;
    float *gate;
    float *conv;
    float *out;
    std::size_t count;
};

static_assert(offsetof(Batch, norm) == 0);
static_assert(offsetof(Batch, gate) == 8);
static_assert(offsetof(Batch, conv) == 16);
static_assert(offsetof(Batch, out) == 24);
static_assert(offsetof(Batch, count) == 32);

struct alignas(128) Line {
    std::array<float, 32> values;
};

static_assert(sizeof(Line) == 128);

class Region {
  public:
    explicit Region(std::size_t bytes)
        : bytes_(bytes), capacity_((bytes + sizeof(Line) - 1) /
                                   sizeof(Line) * sizeof(Line)),
          lines_(capacity_ / sizeof(Line) + 2) {
        arm();
    }

    Region(const Region &) = delete;
    Region &operator=(const Region &) = delete;
    Region(Region &&) = default;
    Region &operator=(Region &&) = default;

    float *data() {
        return lines_[1].values.data();
    }

    const float *data() const {
        return lines_[1].values.data();
    }

    std::size_t bytes() const {
        return bytes_;
    }

    std::size_t capacity() const {
        return capacity_;
    }

    void arm() {
        lines_.front().values.fill(std::bit_cast<float>(kPre));
        lines_.back().values.fill(std::bit_cast<float>(kPost));
        for (std::size_t line = 1; line + 1 < lines_.size(); ++line) {
            lines_[line].values.fill(std::bit_cast<float>(0x7fc12345u));
        }
    }

    bool intact() const {
        if (!std::ranges::all_of(lines_.front().values,
                                 [](float value) {
                                     return std::bit_cast<std::uint32_t>(value) ==
                                            kPre;
                                 }) ||
            !std::ranges::all_of(lines_.back().values,
                                 [](float value) {
                                     return std::bit_cast<std::uint32_t>(value) ==
                                            kPost;
                                 })) {
            return false;
        }
        const std::size_t used = bytes_ / sizeof(std::uint32_t);
        const std::size_t total = capacity_ / sizeof(std::uint32_t);
        for (std::size_t i = used; i < total; ++i) {
            if (std::bit_cast<std::uint32_t>(data()[i]) != 0x7fc12345u) {
                return false;
            }
        }
        return true;
    }

  private:
    std::size_t bytes_;
    std::size_t capacity_;
    std::vector<Line> lines_;
};

struct Stats {
    double median;
    double p95;
};

struct BatchResult {
    Stats fused;
    Stats fifo;
    Stats planes;
    std::size_t count;
    std::size_t reuse;
    std::size_t fused_live;
    std::size_t planes_live;
    std::size_t planes_span;
};

using Fused = void (*)(const float *, const float *, const float *,
                       const Params *, float *);
using Arena = void (*)(const float *, const float *, const float *,
                       const Params *, float *, float *);
using BatchLeaf = void (*)(const float *, const float *, const float *,
                           const Params *, const Batch *);

extern "C" void register_cache_chain_fused(const float *, const float *,
                                            const float *, const Params *,
                                            float *);
extern "C" void register_cache_chain_stack(const float *, const float *,
                                            const float *, const Params *,
                                            float *);
extern "C" void register_cache_chain_arena(const float *, const float *,
                                            const float *, const Params *,
                                            float *, float *);
extern "C" void register_cache_chain_batch_fused(const float *, const float *,
                                                  const float *, const Params *,
                                                  const Batch *);
extern "C" void register_cache_chain_batch_fifo4(const float *, const float *,
                                                  const float *, const Params *,
                                                  const Batch *);
extern "C" void register_cache_chain_batch_planes(const float *, const float *,
                                                   const float *, const Params *,
                                                   const Batch *);

static std::uint64_t ticks() {
    return mach_continuous_time();
}

static double seconds(std::uint64_t delta) {
    static const mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    return static_cast<double>(delta) * static_cast<double>(info.numer) /
           static_cast<double>(info.denom) * 1e-9;
}

static Stats stats(std::array<double, kRounds> values) {
    std::ranges::sort(values);
    return Stats{values[kRounds / 2], values[(kRounds * 95 / 100)]};
}

static std::array<float, kRows> reference(
    const std::array<float, kWidth> &input,
    const std::array<float, kWidth> &gate,
    const std::array<float, kRows * kWidth> &weight, const Params &params) {
    double sum = 0.0;
    for (const float value : input) {
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(
        sum * static_cast<double>(params.inv) + params.eps));

    std::array<float, kWidth> gated{};
    for (std::size_t i = 0; i < kWidth; ++i) {
        gated[i] = input[i] * scale * gate[i];
    }

    std::array<float, kWidth> conv{};
    for (std::size_t i = 0; i < kWidth; ++i) {
        const float prev1 = i == 0 ? params.carry1 : gated[i - 1];
        const float prev2 = i == 0 ? params.carry2
            : (i == 1 ? params.carry1 : gated[i - 2]);
        conv[i] = gated[i] * params.c0 + prev1 * params.c1 +
                  prev2 * params.c2;
    }

    std::array<float, kRows> out{};
    for (std::size_t row = 0; row < kRows; ++row) {
        double value = 0.0;
        for (std::size_t i = 0; i < kWidth; ++i) {
            value += static_cast<double>(conv[i]) * weight[row * kWidth + i];
        }
        out[row] = static_cast<float>(value);
    }
    return out;
}

static bool same_bits(std::span<const float, kRows> left,
                      std::span<const float, kRows> right) {
    for (std::size_t i = 0; i < kRows; ++i) {
        if (std::bit_cast<std::uint32_t>(left[i]) !=
            std::bit_cast<std::uint32_t>(right[i])) {
            return false;
        }
    }
    return true;
}

static bool close_to(std::span<const float, kRows> got,
                     std::span<const float, kRows> want) {
    for (std::size_t i = 0; i < kRows; ++i) {
        const float limit = 3e-5f * std::max(1.0f, std::abs(want[i]));
        if (!std::isfinite(got[i]) || std::abs(got[i] - want[i]) > limit) {
            return false;
        }
    }
    return true;
}

static void arm(Slot &slot) {
    slot.pre.fill(kPre);
    slot.post.fill(kPost);
    slot.data.fill(std::bit_cast<float>(0x7fc12345u));
}

static bool intact(const Slot &slot) {
    return std::ranges::all_of(slot.pre, [](std::uint32_t value) {
               return value == kPre;
           }) &&
           std::ranges::all_of(slot.post, [](std::uint32_t value) {
               return value == kPost;
           });
}

static void arm(Output &out) {
    out.pre.fill(kPre);
    out.post.fill(kPost);
    out.data.fill(std::bit_cast<float>(0x7fc12345u));
}

static bool intact(const Output &out) {
    return std::ranges::all_of(out.pre, [](std::uint32_t value) {
               return value == kPre;
           }) &&
           std::ranges::all_of(out.post, [](std::uint32_t value) {
               return value == kPost;
           });
}

static void validate(const std::array<float, kWidth> &input,
                     const std::array<float, kWidth> &gate,
                     const std::array<float, kRows * kWidth> &weight,
                     const Params &params) {
    alignas(128) Output fused{};
    alignas(128) Output stack{};
    alignas(128) Output arena{};
    alignas(128) Slot slot{};
    arm(slot);
    arm(fused);
    arm(stack);
    arm(arena);

    register_cache_chain_fused(input.data(), gate.data(), weight.data(),
                               &params, fused.data.data());
    register_cache_chain_stack(input.data(), gate.data(), weight.data(),
                               &params, stack.data.data());
    register_cache_chain_arena(input.data(), gate.data(), weight.data(),
                               &params, slot.data.data(), arena.data.data());
    const auto oracle = reference(input, gate, weight, params);

    if (!same_bits(fused.data, stack.data) ||
        !same_bits(fused.data, arena.data)) {
        std::fputs("storage variants changed a numerical bit\n", stderr);
        std::exit(2);
    }
    if (!close_to(fused.data, oracle)) {
        std::fputs("assembly chain disagrees with independent oracle\n", stderr);
        std::exit(2);
    }
    if (!intact(slot) || !intact(fused) || !intact(stack) || !intact(arena)) {
        std::fputs("assembly leaf crossed a scratch/output reservation\n", stderr);
        std::exit(2);
    }
}

static float sample(std::uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const std::uint32_t bits = static_cast<std::uint32_t>(state >> 40);
    return static_cast<float>(bits) / static_cast<float>(0x00ffffffu) * 2.0f - 1.0f;
}

static void validate_sweep() {
    std::uint64_t state = 0x91e10da5c79e7b1dull;
    for (std::size_t trial = 0; trial < 257; ++trial) {
        alignas(128) std::array<float, kWidth> input{};
        alignas(128) std::array<float, kWidth> gate{};
        alignas(128) std::array<float, kRows * kWidth> weight{};
        for (float &value : input) {
            value = sample(state);
        }
        for (float &value : gate) {
            value = 0.75f + sample(state) * 0.5f;
        }
        for (float &value : weight) {
            value = sample(state) * 0.375f;
        }
        const Params params{1e-5f, 1.0f / static_cast<float>(kWidth), 1.0f,
                            0.625f, -0.25f, 0.125f,
                            sample(state) * 0.5f, sample(state) * 0.5f};
        validate(input, gate, weight, params);
    }
}

static double time_fused(Fused leaf,
                         const std::array<float, kWidth> &input,
                         const std::array<float, kWidth> &gate,
                         const std::array<float, kRows * kWidth> &weight,
                         const Params &params,
                         std::array<std::array<float, kRows>, kOuts> &outs) {
    const std::uint64_t begin = ticks();
    for (std::size_t i = 0; i < kCalls; ++i) {
        leaf(input.data(), gate.data(), weight.data(), &params,
             outs[i & (kOuts - 1)].data());
    }
    return seconds(ticks() - begin) * 1e9 / static_cast<double>(kCalls);
}

static std::array<Stats, 2> run_pair(
    const std::array<float, kWidth> &input,
    const std::array<float, kWidth> &gate,
    const std::array<float, kRows * kWidth> &weight, const Params &params,
    std::array<std::array<float, kRows>, kOuts> &outs) {
    std::array<double, kRounds> fused{};
    std::array<double, kRounds> stack{};
    static_cast<void>(time_fused(register_cache_chain_fused, input, gate,
                                 weight, params, outs));
    static_cast<void>(time_fused(register_cache_chain_stack, input, gate,
                                 weight, params, outs));

    std::uint64_t order = 0x713c5a4d9b18e2f7ull;
    for (std::size_t round = 0; round < kRounds; ++round) {
        order ^= order << 13;
        order ^= order >> 7;
        order ^= order << 17;
        if ((order & 1u) == 0) {
            fused[round] = time_fused(register_cache_chain_fused, input, gate,
                                      weight, params, outs);
            stack[round] = time_fused(register_cache_chain_stack, input, gate,
                                      weight, params, outs);
            continue;
        }
        stack[round] = time_fused(register_cache_chain_stack, input, gate,
                                  weight, params, outs);
        fused[round] = time_fused(register_cache_chain_fused, input, gate,
                                  weight, params, outs);
    }
    return {stats(fused), stats(stack)};
}

static Stats run_arena(Arena leaf, std::vector<Slot> &slots,
                       const std::array<float, kWidth> &input,
                       const std::array<float, kWidth> &gate,
                       const std::array<float, kRows * kWidth> &weight,
                       const Params &params,
                       std::array<std::array<float, kRows>, kOuts> &outs) {
    std::array<double, kRounds> samples{};
    for (std::size_t round = 0; round < kRounds + 1; ++round) {
        std::size_t done = 0;
        const std::uint64_t begin = ticks();
        while (done < kCalls) {
            const std::size_t count = std::min(slots.size(), kCalls - done);
            for (std::size_t i = 0; i < count; ++i) {
                leaf(input.data(), gate.data(), weight.data(), &params,
                     slots[i].data.data(), outs[(done + i) & (kOuts - 1)].data());
            }
            done += count;
        }
        const double elapsed = seconds(ticks() - begin);
        if (round != 0) {
            samples[round - 1] = elapsed * 1e9 / static_cast<double>(kCalls);
        }
    }
    return stats(samples);
}

static double time_batch(
    BatchLeaf leaf, std::size_t reps,
    const std::array<float, kWidth> &input,
    const std::array<float, kWidth> &gate,
    const std::array<float, kRows * kWidth> &weight, const Params &params,
    const Batch &batch) {
    const std::uint64_t begin = ticks();
    for (std::size_t rep = 0; rep < reps; ++rep) {
        leaf(input.data(), gate.data(), weight.data(), &params, &batch);
    }
    const double tiles = static_cast<double>(reps) *
                         static_cast<double>(batch.count);
    return seconds(ticks() - begin) * 1e9 / tiles;
}

static std::array<Stats, 3> run_batch_group(
    std::size_t reps, const std::array<float, kWidth> &input,
    const std::array<float, kWidth> &gate,
    const std::array<float, kRows * kWidth> &weight, const Params &params,
    const Batch &fused_batch, const Batch &fifo_batch,
    const Batch &planes_batch) {
    std::array<std::array<double, kRounds>, 3> samples{};
    constexpr std::array<BatchLeaf, 3> leaves{
        register_cache_chain_batch_fused,
        register_cache_chain_batch_fifo4,
        register_cache_chain_batch_planes,
    };
    const std::array<const Batch *, 3> batches{
        &fused_batch,
        &fifo_batch,
        &planes_batch,
    };
    static_cast<void>(time_batch(register_cache_chain_batch_fused, 1, input,
                                 gate, weight, params, fused_batch));
    static_cast<void>(time_batch(register_cache_chain_batch_fifo4, 1, input,
                                 gate, weight, params, fifo_batch));
    static_cast<void>(time_batch(register_cache_chain_batch_planes, 1, input,
                                 gate, weight, params, planes_batch));

    std::uint64_t order = 0xa0761d6478bd642full ^ fused_batch.count;
    for (std::size_t round = 0; round < kRounds; ++round) {
        std::array<std::size_t, 3> sequence{0, 1, 2};
        for (std::size_t i = sequence.size() - 1; i > 0; --i) {
            order ^= order << 13;
            order ^= order >> 7;
            order ^= order << 17;
            std::swap(sequence[i], sequence[order % (i + 1)]);
        }
        for (const std::size_t index : sequence) {
            samples[index][round] = time_batch(
                leaves[index], reps, input, gate, weight, params,
                *batches[index]);
        }
    }
    return {stats(samples[0]), stats(samples[1]), stats(samples[2])};
}

static void check_batch_outputs(const Region &fused, const Region &fifo,
                                const Region &planes,
                                std::size_t count,
                                const std::array<float, kRows> &want) {
    const float *left = fused.data();
    const float *middle = fifo.data();
    const float *right = planes.data();
    for (std::size_t tile = 0; tile < count; ++tile) {
        for (std::size_t row = 0; row < kRows; ++row) {
            const std::size_t at = tile * kRows + row;
            const std::uint32_t expected = std::bit_cast<std::uint32_t>(want[row]);
            if (std::bit_cast<std::uint32_t>(left[at]) != expected ||
                std::bit_cast<std::uint32_t>(middle[at]) != expected ||
                std::bit_cast<std::uint32_t>(right[at]) != expected ||
                std::bit_cast<std::uint32_t>(left[at]) !=
                    std::bit_cast<std::uint32_t>(middle[at]) ||
                std::bit_cast<std::uint32_t>(middle[at]) !=
                    std::bit_cast<std::uint32_t>(right[at])) {
                std::fprintf(stderr,
                             "batch terminal mismatch at tile=%zu row=%zu\n",
                             tile, row);
                std::exit(2);
            }
        }
    }
}

static BatchResult run_footprint(
    std::size_t target, const std::array<float, kWidth> &input,
    const std::array<float, kWidth> &gate,
    const std::array<float, kRows * kWidth> &weight, const Params &params,
    const std::array<float, kRows> &want) {
    const std::size_t count = std::max<std::size_t>(
        1, (target > kBatchFixed ? target - kBatchFixed : 0) / kTileBytes);
    const std::size_t plane_bytes = count * kPlaneBytes;
    const std::size_t out_bytes = count * kRows * sizeof(float);
    Region norm(plane_bytes);
    Region gated(plane_bytes);
    Region conv(plane_bytes);
    Region fused_out(out_bytes);
    Region fifo_out(out_bytes);
    Region planes_out(out_bytes);
    const Batch fused_batch{norm.data(), gated.data(), conv.data(),
                            fused_out.data(), count};
    const Batch fifo_batch{norm.data(), gated.data(), conv.data(),
                           fifo_out.data(), count};
    const Batch planes_batch{norm.data(), gated.data(), conv.data(),
                             planes_out.data(), count};

    register_cache_chain_batch_fused(input.data(), gate.data(), weight.data(),
                                     &params, &fused_batch);
    register_cache_chain_batch_fifo4(input.data(), gate.data(), weight.data(),
                                     &params, &fifo_batch);
    register_cache_chain_batch_planes(input.data(), gate.data(), weight.data(),
                                      &params, &planes_batch);
    check_batch_outputs(fused_out, fifo_out, planes_out, count, want);

    norm.arm();
    gated.arm();
    conv.arm();
    fused_out.arm();
    fifo_out.arm();
    planes_out.arm();
    constexpr std::size_t minimum = 64u << 20;
    const std::size_t live = kBatchFixed + count * kTileBytes;
    const std::size_t reps = std::max<std::size_t>(1, (minimum + live - 1) / live);
    const auto measured = run_batch_group(reps, input, gate, weight, params,
                                          fused_batch, fifo_batch,
                                          planes_batch);
    check_batch_outputs(fused_out, fifo_out, planes_out, count, want);
    if (!norm.intact() || !gated.intact() || !conv.intact() ||
        !fused_out.intact() || !fifo_out.intact() || !planes_out.intact()) {
        std::fputs("batch leaf crossed a guarded plane/output reservation\n",
                   stderr);
        std::exit(2);
    }

    return BatchResult{
        measured[0], measured[1], measured[2], count, plane_bytes,
        kBatchFixed + out_bytes, live,
        kBatchFixed + norm.capacity() + gated.capacity() + conv.capacity() +
            planes_out.capacity(),
    };
}

static void check_outputs(
    const std::array<std::array<float, kRows>, kOuts> &outs,
    const std::array<float, kRows> &want, std::string_view label) {
    for (const auto &out : outs) {
        if (!same_bits(out, want)) {
            std::fprintf(stderr, "%.*s produced a corrupt terminal output\n",
                         static_cast<int>(label.size()), label.data());
            std::exit(2);
        }
    }
}

static void print_row(std::string_view name, std::size_t live, std::size_t span,
                      const Stats &value, double base) {
    const double extra = value.median - base;
    std::printf("%-16.*s %12zu %12zu %9.3f %9.3f %9.3f %8.3fx\n",
                static_cast<int>(name.size()), name.data(), live, span,
                value.median, value.p95, extra, value.median / base);
}

} // namespace

int main() {
    validate_sweep();
    alignas(128) std::array<float, kWidth> input{};
    alignas(128) std::array<float, kWidth> gate{};
    alignas(128) std::array<float, kRows * kWidth> weight{};
    for (std::size_t i = 0; i < kWidth; ++i) {
        input[i] = std::sin(static_cast<float>(i) * 0.37f) * 0.8f +
                   static_cast<float>(i % 3) * 0.03125f;
        gate[i] = 0.7f + std::cos(static_cast<float>(i) * 0.19f) * 0.2f;
    }
    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] = std::sin(static_cast<float>(i) * 0.11f) * 0.25f +
                    std::cos(static_cast<float>(i) * 0.07f) * 0.125f;
    }
    const Params params{1e-5f, 1.0f / static_cast<float>(kWidth), 1.0f,
                        0.625f, -0.25f, 0.125f, -0.375f, 0.25f};
    validate(input, gate, weight, params);

    alignas(128) std::array<float, kRows> want{};
    register_cache_chain_fused(input.data(), gate.data(), weight.data(),
                               &params, want.data());
    alignas(128) std::array<std::array<float, kRows>, kOuts> outs{};

    const auto core = run_pair(input, gate, weight, params, outs);
    const Stats fused = core[0];
    const Stats stack = core[1];
    check_outputs(outs, want, "register/stack");

    std::puts("representative chain: norm -> gate -> ShortConv-3 -> projection");
    std::puts("257 randomized gates: bit-identical variants + bounded oracle parity");
    std::printf("fixed hot set: %zu B (input/gate/weights/params/output ring)\n", kFixed);
    std::puts("timed regions allocate/copy nothing; controls store+reload three 64 B planes");
    std::puts("variant          total live B address span B   p50 ns    p95 ns   +fused slowdown");
    print_row("register fused", kFixed, kFixed, fused, fused.median);
    print_row("stack 192 B", kFixed + 192, kFixed + 192, stack, fused.median);

    constexpr std::array<std::size_t, 8> sizes{
        4u << 10, 64u << 10, 256u << 10, 2u << 20,
        8u << 20, 16u << 20, 32u << 20, 64u << 20,
    };
    std::array<std::size_t, sizes.size()> order{};
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::uint64_t random = 0xd6e8feb86659fd93ull;
    for (std::size_t i = order.size() - 1; i > 0; --i) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        std::swap(order[i], order[random % (i + 1)]);
    }

    std::array<Stats, sizes.size()> results{};
    for (const std::size_t index : order) {
        const std::size_t bytes = sizes[index];
        std::vector<Slot> slots(bytes / sizeof(Slot));
        for (Slot &slot : slots) {
            arm(slot);
        }
        results[index] = run_arena(register_cache_chain_arena, slots, input,
                                   gate, weight, params, outs);
        check_outputs(outs, want, "arena");
        if (!std::ranges::all_of(slots, [](const Slot &slot) {
                return intact(slot);
            })) {
            std::fputs("arena canary corruption after timed run\n", stderr);
            return 2;
        }

    }

    for (std::size_t index = 0; index < sizes.size(); ++index) {
        const std::size_t bytes = sizes[index];
        const std::size_t slots = bytes / sizeof(Slot);
        std::array<char, 32> label{};
        if (bytes < (1u << 20)) {
            std::snprintf(label.data(), label.size(), "arena %zu KiB", bytes >> 10);
        } else {
            std::snprintf(label.data(), label.size(), "arena %zu MiB", bytes >> 20);
        }
        print_row(label.data(), kFixed + slots * kScratch * sizeof(float),
                  kFixed + slots * sizeof(Slot), results[index], fused.median);
    }

    constexpr std::array<std::size_t, 26> targets{
        64u << 10,  96u << 10,  112u << 10, 120u << 10, 124u << 10,
        128u << 10, 132u << 10, 136u << 10, 144u << 10, 160u << 10,
        192u << 10, 256u << 10,
        8u << 20,   12u << 20,  14u << 20,  15u << 20,  31u << 19,
        63u << 18,  16u << 20,  65u << 18,  33u << 19,  17u << 20,
        18u << 20,  20u << 20,  64u << 20,  128u << 20,
    };
    std::array<std::size_t, targets.size()> batch_order{};
    for (std::size_t i = 0; i < batch_order.size(); ++i) {
        batch_order[i] = i;
    }
    random = 0xe7037ed1a0b428dbull;
    for (std::size_t i = batch_order.size() - 1; i > 0; --i) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        std::swap(batch_order[i], batch_order[random % (i + 1)]);
    }

    std::array<BatchResult, targets.size()> batch_results{};
    for (const std::size_t index : batch_order) {
        batch_results[index] = run_footprint(targets[index], input, gate,
                                             weight, params, want);
    }

    std::puts("\ndelayed reuse: naive fused vs four-tile register FIFO vs full planes");
    std::puts("all paths write identical terminal spans; FIFO ABI save+restore is 128 B/batch");
    std::puts("phase_live_B   tiles abi_B/tile      reuse_B fused_live_B phase_span_B naive50 naive95 fifo50 fifo95 plane50 plane95 fifo/naive plane/fifo");
    for (const BatchResult &result : batch_results) {
        std::printf("%12zu %8zu %10.6f %12zu %12zu %12zu %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %9.3fx %9.3fx\n",
                    result.planes_live, result.count,
                    128.0 / static_cast<double>(result.count), result.reuse,
                    result.fused_live, result.planes_span,
                    result.fused.median, result.fused.p95,
                    result.fifo.median, result.fifo.p95,
                    result.planes.median, result.planes.p95,
                    result.fifo.median / result.fused.median,
                    result.planes.median / result.fifo.median);
    }

    std::puts("\nInterpret cache knees only with counters: stack and arenas are ordinary");
    std::puts("cache-backed virtual memory, not reserved L1/L2 storage. The batch table");
    std::puts("reports footprint and reuse distance only; it assigns no physical tier.");
    return 0;
}
