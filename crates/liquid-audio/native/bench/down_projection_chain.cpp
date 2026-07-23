// down_projection_chain.cpp -- product-grounded ST_DOWN materialization probe.
//
// This compares the same numerical program through three storage contracts:
//
//   1. the current three production leaves with 512-output stack scratch;
//   2. those same leaves and bands with one preallocated arena scratch slot;
//   3. one faithful AArch64 leaf that carries one dot and both RNE stages in
//      registers and stores only the terminal BF16 output;
//   4. a four-output register FIFO that loads each activation block once,
//      maintains eight exact accumulators, and terminal-stores four results.
//
// Build from crates/liquid-audio/native/bench (C++23 is intentional):
//
//   clang++ -O3 -std=c++23 -Wall -Wextra -Wpedantic -Werror \
//     -ffp-contract=off -mcpu=apple-m2 \
//     down_projection_chain.cpp down_projection_chain_aarch64.S \
//     ../kernels/aarch64/flashkern_neon.cpp -I../include \
//     -o /tmp/down_projection_chain
//   /tmp/down_projection_chain
//
// Prove the fused object has neither calls nor stack/intermediate stores:
//
//   clang -c -mcpu=apple-m2 down_projection_chain_aarch64.S \
//     -o /tmp/down_projection_chain_aarch64.o
//   otool -tvV /tmp/down_projection_chain_aarch64.o
//
// Neither symbol may contain `bl` or `[sp]`. The one-row symbol has exactly one
// store instruction (`strh`); the four-row symbol has only terminal `str d16`
// and tail `strh` sites. Loads are raw and tolerate byte-unaligned views.
//
// The fused dot loop is arithmetically faithful but scheduled differently from
// the generic-M production C++ leaf: it removes generic row machinery and
// consumes two eight-element blocks per branch. Its timing therefore measures
// combined fusion + specialized scheduling. Stack vs arena is the controlled
// scratch-location comparison; do not attribute the fused delta solely to the
// deleted intermediates. `fused4` is a separate cell: it additionally reuses
// each activation load across four weight rows and exposes eight independent
// accumulator chains, while preserving each row's arithmetic order.
//
// Production ST_DOWN currently uses its explicit integer RNE helper in both
// lfm_f32_to_bf16 and lfm_bf16_add, not BFCVT. The fused leaf implements that
// same `u + 0x7fff + lsb` operation. Probe operands are finite and bounded:
// |a|,|w|,|residual| <= 510/2048, so |dot| <= 508 at K=8192. NaN payload
// behavior is deliberately outside this benchmark's admitted input domain.
//
// Timed regions perform no allocation, initialization, copying, or comparison.
// The source planes are stable byte views and every contestant overwrites its
// destination. Reported `weight GB/s` counts only immutable checkpoint bytes;
// source-activation traffic and eliminated intermediate traffic are reported
// separately. Without hardware counters this benchmark does not claim that a
// byte was served by L1, L2, or DRAM.

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include <arm_neon.h>
#include <mach/mach_time.h>

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "down_projection_chain requires native Apple AArch64."
#endif

extern "C" void lfm_bf16_gemm_nt_f32(const uint16_t *input,
                                      const void *weights, float *output,
                                      int rows, int cols, int depth);
extern "C" void lfm_f32_to_bf16(const float *input, uint16_t *output,
                                 int count);
extern "C" void lfm_bf16_add(const void *left, const void *right,
                              uint16_t *output, int count);
extern "C" void lfm_down_projection_chain_fused_aarch64(
    const void *input, const void *weights, const void *residual, void *output,
    size_t cols, size_t depth);
extern "C" void lfm_down_projection_chain_fused4_aarch64(
    const void *input, const void *weights, const void *residual, void *output,
    size_t cols, size_t depth);

namespace {

constexpr size_t kBand = 512;
constexpr size_t kRounds = 17;
constexpr size_t kAlign = 128;
constexpr size_t kGuard = 128;

volatile uint64_t sink = 0;

class Block {
  public:
    Block(size_t bytes, size_t skew) : bytes_(bytes), skew_(skew) {
        if (posix_memalign(&base_, kAlign,
                          bytes + skew + kGuard * 2) != 0) {
            std::fprintf(stderr, "posix_memalign(%zu): %s\n", bytes,
                         std::strerror(errno));
            std::exit(2);
        }
        data_ = static_cast<uint8_t *>(base_) + kGuard + skew;
        std::fill_n(data_ - kGuard, kGuard, uint8_t{0xa5});
        std::fill_n(data_ + bytes_, kGuard, uint8_t{0x5a});
    }

    ~Block() { std::free(base_); }
    Block(const Block &) = delete;
    Block &operator=(const Block &) = delete;

    uint8_t *data() { return data_; }
    const uint8_t *data() const { return data_; }
    size_t size() const { return bytes_; }
    size_t skew() const { return skew_; }

    void require_guards(std::string_view name) const {
        for (size_t i = 0; i < kGuard; ++i) {
            if (data_[-static_cast<ptrdiff_t>(kGuard) +
                      static_cast<ptrdiff_t>(i)] == uint8_t{0xa5})
                continue;
            std::fprintf(stderr, "%.*s pre-guard overwritten at %zu\n",
                         static_cast<int>(name.size()), name.data(), i);
            std::exit(4);
        }
        for (size_t i = 0; i < kGuard; ++i) {
            if (data_[bytes_ + i] == uint8_t{0x5a}) continue;
            std::fprintf(stderr, "%.*s post-guard overwritten at %zu\n",
                         static_cast<int>(name.size()), name.data(), i);
            std::exit(4);
        }
    }

  private:
    void *base_ = nullptr;
    uint8_t *data_ = nullptr;
    size_t bytes_ = 0;
    size_t skew_ = 0;
};

struct Shape {
    size_t n;
    size_t k;
    const char *name;
};

struct Stats {
    double median;
    double p10;
    double p90;
};

struct Case {
    const char *name;
    void (*run)(const void *, const void *, const void *, void *, float *,
                uint16_t *, size_t, size_t);
    Block *output;
    std::vector<double> samples;
};

static uint64_t ticks() { return mach_continuous_time(); }

static double micros(uint64_t delta) {
    static const mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    return static_cast<double>(delta) * static_cast<double>(info.numer) /
           static_cast<double>(info.denom) * 1e-3;
}

static uint16_t to_bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static uint16_t bfcvt_bf16(float value) {
    return std::bit_cast<uint16_t>(vcvth_bf16_f32(value));
}

static void require_rne_contract() {
    const std::array<uint32_t, 16> patterns{{
        0x00000000u, 0x80000000u, 0x00000001u, 0x00007fffu,
        0x00008000u, 0x007fffffu, 0x00800000u, 0x3f7f7fffu,
        0x3f7f8000u, 0xbf7f8000u, 0x7f7fffffu, 0xff7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc12345u, 0x7f812345u,
    }};
    std::array<float, patterns.size()> values{};
    std::array<uint16_t, patterns.size()> production{};
    for (size_t i = 0; i < patterns.size(); ++i)
        values[i] = std::bit_cast<float>(patterns[i]);
    lfm_f32_to_bf16(values.data(), production.data(),
                     static_cast<int>(values.size()));

    size_t special_differences = 0;
    for (size_t i = 0; i < patterns.size(); ++i) {
        const uint16_t integer = to_bf16(values[i]);
        if (production[i] != integer) {
            std::fprintf(stderr,
                         "production integer-RNE mismatch for %08x: %04x/%04x\n",
                         patterns[i], production[i], integer);
            std::exit(5);
        }
        const uint16_t hardware = bfcvt_bf16(values[i]);
        if ((patterns[i] & 0x7f800000u) != 0x7f800000u &&
            hardware != integer) {
            std::fprintf(stderr,
                         "finite BFCVT mismatch for %08x: %04x/%04x\n",
                         patterns[i], hardware, integer);
            std::exit(5);
        }
        if (hardware != integer) ++special_differences;
    }
    std::printf("# RNE gate: production integer helper == fused rule; ");
    std::printf("finite integer-RNE == BFCVT; special-value differences=%zu\n",
                special_differences);
}

static void store_word(uint8_t *bytes, size_t index, uint16_t word) {
    bytes[index * 2] = static_cast<uint8_t>(word);
    bytes[index * 2 + 1] = static_cast<uint8_t>(word >> 8);
}

static uint16_t load_word(const uint8_t *bytes, size_t index) {
    return static_cast<uint16_t>(bytes[index * 2]) |
           static_cast<uint16_t>(bytes[index * 2 + 1]) << 8;
}

static uint32_t random_word(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_bf16(Block &block, size_t count, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        const int32_t raw = static_cast<int32_t>(random_word(state) % 1021u) - 510;
        store_word(block.data(), i, to_bf16(static_cast<float>(raw) / 2048.0f));
    }
}

__attribute__((noinline))
static void stack_chain(const void *input, const void *weights,
                        const void *residual, void *output, float *, uint16_t *,
                        size_t n, size_t k) {
    alignas(kAlign) float sums[kBand];
    alignas(kAlign) uint16_t rounded[kBand];
    const auto *input_words = static_cast<const uint16_t *>(input);
    const auto *weight_bytes = static_cast<const uint8_t *>(weights);
    const auto *residual_bytes = static_cast<const uint8_t *>(residual);
    auto *output_words = static_cast<uint16_t *>(output);
    for (size_t begin = 0; begin < n; begin += kBand) {
        const size_t count = std::min(kBand, n - begin);
        lfm_bf16_gemm_nt_f32(input_words, weight_bytes + begin * k * 2, sums,
                             1, static_cast<int>(count), static_cast<int>(k));
        lfm_f32_to_bf16(sums, rounded, static_cast<int>(count));
        lfm_bf16_add(rounded, residual_bytes + begin * 2,
                     output_words + begin, static_cast<int>(count));
    }
}

__attribute__((noinline))
static void arena_chain(const void *input, const void *weights,
                        const void *residual, void *output, float *sums,
                        uint16_t *rounded, size_t n, size_t k) {
    const auto *input_words = static_cast<const uint16_t *>(input);
    const auto *weight_bytes = static_cast<const uint8_t *>(weights);
    const auto *residual_bytes = static_cast<const uint8_t *>(residual);
    auto *output_words = static_cast<uint16_t *>(output);
    for (size_t begin = 0; begin < n; begin += kBand) {
        const size_t count = std::min(kBand, n - begin);
        lfm_bf16_gemm_nt_f32(input_words, weight_bytes + begin * k * 2, sums,
                             1, static_cast<int>(count), static_cast<int>(k));
        lfm_f32_to_bf16(sums, rounded, static_cast<int>(count));
        lfm_bf16_add(rounded, residual_bytes + begin * 2,
                     output_words + begin, static_cast<int>(count));
    }
}

__attribute__((noinline))
static void fused_chain(const void *input, const void *weights,
                        const void *residual, void *output, float *, uint16_t *,
                        size_t n, size_t k) {
    lfm_down_projection_chain_fused_aarch64(input, weights, residual, output, n,
                                            k);
}

__attribute__((noinline))
static void fused4_chain(const void *input, const void *weights,
                         const void *residual, void *output, float *,
                         uint16_t *, size_t n, size_t k) {
    lfm_down_projection_chain_fused4_aarch64(input, weights, residual, output,
                                             n, k);
}

static Stats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {
        .median = samples[samples.size() / 2],
        .p10 = samples[(samples.size() - 1) / 10],
        .p90 = samples[((samples.size() - 1) * 9) / 10],
    };
}

static void require_equal(const Block &expected, const Block &actual,
                          size_t count, std::string_view name) {
    for (size_t i = 0; i < count; ++i) {
        const uint16_t lhs = load_word(expected.data(), i);
        const uint16_t rhs = load_word(actual.data(), i);
        if (lhs == rhs) continue;
        std::fprintf(stderr,
                     "%.*s parity failed at %zu: expected=%04x actual=%04x\n",
                     static_cast<int>(name.size()), name.data(), i, lhs, rhs);
        std::exit(3);
    }
}

static void require_fused4_tail() {
    constexpr size_t n = 7;
    constexpr size_t k = 19;
    Block input(k * 2, 2);
    Block weights(n * k * 2, 1);
    Block residual(n * 2, 1);
    Block expected(n * 2, 0);
    Block actual(n * 2, 1);
    Block sums(kBand * sizeof(float), 0);
    Block rounded(kBand * sizeof(uint16_t), 0);
    fill_bf16(input, k, 0x0badf00du);
    fill_bf16(weights, n * k, 0x600df00du);
    fill_bf16(residual, n, 0x1234abcdu);
    stack_chain(input.data(), weights.data(), residual.data(), expected.data(),
                reinterpret_cast<float *>(sums.data()),
                reinterpret_cast<uint16_t *>(rounded.data()), n, k);
    fused4_chain(input.data(), weights.data(), residual.data(), actual.data(),
                 reinterpret_cast<float *>(sums.data()),
                 reinterpret_cast<uint16_t *>(rounded.data()), n, k);
    require_equal(expected, actual, n, "fused4 N/K tail");
    input.require_guards("tail input");
    weights.require_guards("tail weights");
    residual.require_guards("tail residual");
    expected.require_guards("tail expected");
    actual.require_guards("tail actual");
    sums.require_guards("tail sums");
    rounded.require_guards("tail rounded");
    std::printf("# fused4 tail gate: N=7 K=19 terminal BF16 bit-exact\n");
}

static void run_shape(const Shape &shape) {
    // Input remains 2-byte aligned because the production GEMM ABI still names
    // uint16_t. Checkpoint weights and residual views begin at odd byte offsets;
    // the fused destination is odd as well. No SIMD/cache-line alignment is
    // assumed by any contestant.
    Block input(shape.k * 2, 2);
    Block weights(shape.n * shape.k * 2, 1);
    Block residual(shape.n * 2, 1);
    Block stack_output(shape.n * 2, 0);
    Block arena_output(shape.n * 2, 0);
    Block fused_output(shape.n * 2, 1);
    Block fused4_output(shape.n * 2, 1);
    Block arena_f32(kBand * sizeof(float), 0);
    Block arena_bf16(kBand * sizeof(uint16_t), 0);

    fill_bf16(input, shape.k, 0x13579bdfu);
    fill_bf16(weights, shape.n * shape.k, 0x2468ace1u);
    fill_bf16(residual, shape.n, 0xdeadbeefu);

    auto *sums = reinterpret_cast<float *>(arena_f32.data());
    auto *rounded = reinterpret_cast<uint16_t *>(arena_bf16.data());

    stack_chain(input.data(), weights.data(), residual.data(),
                stack_output.data(), sums, rounded, shape.n, shape.k);
    arena_chain(input.data(), weights.data(), residual.data(),
                arena_output.data(), sums, rounded, shape.n, shape.k);
    fused_chain(input.data(), weights.data(), residual.data(),
                fused_output.data(), sums, rounded, shape.n, shape.k);
    fused4_chain(input.data(), weights.data(), residual.data(),
                 fused4_output.data(), sums, rounded, shape.n, shape.k);
    require_equal(stack_output, arena_output, shape.n, "arena");
    require_equal(stack_output, fused_output, shape.n, "fused");
    require_equal(stack_output, fused4_output, shape.n, "fused4");

    std::array<Case, 4> cases{{
        {"three-leaf stack", stack_chain, &stack_output, {}},
        {"three-leaf arena", arena_chain, &arena_output, {}},
        {"fused1 final-store", fused_chain, &fused_output, {}},
        {"fused4 reg-FIFO", fused4_chain, &fused4_output, {}},
    }};
    for (Case &entry : cases) {
        entry.samples.reserve(kRounds);
        entry.run(input.data(), weights.data(), residual.data(),
                  entry.output->data(), sums, rounded, shape.n, shape.k);
    }

    std::mt19937 random(0x5eedu + static_cast<uint32_t>(shape.n));
    std::array<size_t, 4> order{{0, 1, 2, 3}};
    for (size_t round = 0; round < kRounds; ++round) {
        std::shuffle(order.begin(), order.end(), random);
        for (size_t index : order) {
            Case &entry = cases[index];
            const uint64_t start = ticks();
            entry.run(input.data(), weights.data(), residual.data(),
                      entry.output->data(), sums, rounded, shape.n, shape.k);
            const uint64_t stop = ticks();
            entry.samples.push_back(micros(stop - start));
            sink ^= load_word(entry.output->data(),
                              (round * 131u + index * 17u) % shape.n);
        }
    }

    for (const Case &entry : cases)
        require_equal(stack_output, *entry.output, shape.n, entry.name);

    input.require_guards("input");
    weights.require_guards("weights");
    residual.require_guards("residual");
    stack_output.require_guards("stack output");
    arena_output.require_guards("arena output");
    fused_output.require_guards("fused output");
    fused4_output.require_guards("fused4 output");
    arena_f32.require_guards("arena f32");
    arena_bf16.require_guards("arena bf16");

    const size_t weight_bytes = shape.n * shape.k * 2;
    const size_t activation_bytes = shape.n * shape.k * 2;
    const size_t fifo_activation_bytes =
        (shape.n / 4 + shape.n % 4) * shape.k * 2;
    const size_t intermediate_bytes = shape.n * (sizeof(float) * 2 +
                                                  sizeof(uint16_t) * 2);
    std::printf("## %s: N=%zu K=%zu\n", shape.name, shape.n, shape.k);
    std::printf("   immutable weight stream : %zu bytes\n", weight_bytes);
    std::printf("   legacy/fused1 activation: %zu bytes\n", activation_bytes);
    std::printf("   fused4 activation stream: %zu bytes\n",
                fifo_activation_bytes);
    std::printf("   legacy intermediate R/W : %zu bytes (cache tier unclaimed)\n",
                intermediate_bytes);
    std::printf("   fused intermediate R/W  : 0 bytes; terminal store=%zu bytes\n",
                shape.n * sizeof(uint16_t));
    for (const Case &entry : cases) {
        const Stats stats = summarize(entry.samples);
        const double gbps = static_cast<double>(weight_bytes) / stats.median /
                            1000.0;
        std::printf("   %-18s %8.3f us  p10=%8.3f  p90=%8.3f  weight=%6.1f GB/s\n",
                    entry.name, stats.median, stats.p10, stats.p90, gbps);
    }
    std::printf("   parity: terminal BF16 bit-exact across all 17 rounds\n\n");
}

} // namespace

int main() {
    std::printf("# ST_DOWN register/cache chain probe\n");
    std::printf("# raw views: weights/residual/fused-output are deliberately +1 byte\n");
    std::printf("# randomized paired order; 17 samples; median/p10/p90\n");
    std::printf("# checked 128-byte pre/post guards around every allocated view\n");
    std::printf("# fused timing includes specialized dot scheduling + fusion\n");
    std::printf("# fused4 additionally measures four-row activation reuse + ILP\n");
    std::printf("# no cache-tier claim without hardware counters\n\n");
    require_rne_contract();
    require_fused4_tail();
    std::printf("\n");

    const std::array<Shape, 2> shapes{{
        {256, 8192, "single down band"},
        {2048, 8192, "full down projection"},
    }};
    for (const Shape &shape : shapes) run_shape(shape);
    std::printf("sink=%016llx\n", static_cast<unsigned long long>(sink));
    return 0;
}
