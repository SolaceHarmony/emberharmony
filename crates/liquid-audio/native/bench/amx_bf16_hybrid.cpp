// amx_bf16_hybrid.cpp -- ground-truth probe for the proposed AMX -> scratch ->
// NEON continuation seam on Apple M2.
//
// This deliberately uses the production checkpoint contract:
//
//   A: bf16 [M,K], W: immutable bf16 [N,K], C: f32 [M,N] = A * W^T
//
// There is no KxN weight image, widened weight plane, or packed weight panel.
// The direct AMX control therefore uses VECFP (32 pointwise bf16 FMAs) rather
// than MATFP (a 32x32 outer product). MATFP needs W[n0:n0+32,k] contiguous;
// those values are K-strided in the checkpoint image, and AMX has no memory
// gather or NEON-register input. Claiming a zero-pack MATFP kernel for this
// layout would be false.
//
// The probe times seven paths:
//   1. the exact product NEON checkpoint-layout leaf;
//   2. ordinary NEON FMA with four decode accumulator chains;
//   3. raw-layout NEON BFDOT with register-resident accumulator chains;
//   4. raw-layout AMX with a 512-byte tile-local scratch handoff;
//   5. raw-layout AMX fast-32 producing a complete partial plane, followed by a
//      separate NEON consumer (the proposed suspend/resume memory boundary);
//   6. the same split with the production-order exact-8 reduction;
//   7. that NEON scratch consumer alone.
//
// Build from crates/liquid-audio/native/bench:
//   clang++ -O3 -std=c++23 -ffp-contract=off -mcpu=apple-m2 \
//     amx_bf16_hybrid.cpp ../kernels/aarch64/flashkern_neon.cpp \
//     -I../include -framework Accelerate -o /tmp/amx_bf16_hybrid
//
// Run:
//   /tmp/amx_bf16_hybrid
//
// The AMX instruction encodings below follow the MIT-licensed corsix/amx
// documentation. They are private-ISA reconnaissance, not product code.

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

#include <arm_neon.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>

#include "flashkern_gemm.h"

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "This reconnaissance probe requires native Apple AArch64."
#endif

#define AMX_NOP_OP_IMM5(op, imm5)                                                \
    __asm__ volatile("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)"          \
                     : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr)                                                      \
    __asm__ volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))"     \
                     : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr)   AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr)   AMX_OP_GPR(1, gpr)
#define AMX_STZI(gpr)  AMX_OP_GPR(7, gpr)
#define AMX_SET()      AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()      AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECFP(gpr) AMX_OP_GPR(19, gpr)

namespace {

constexpr int kLanes = 32;
constexpr int kRounds = 17;

volatile uint64_t g_sink = 0;
bool g_raw = false;

struct Shape {
    int m;
    int n;
    int k;
    const char *name;
};

struct Stats {
    double median;
    double p10;
    double p90;
};

using Kernel = void (*)(const uint16_t *, const void *, float *, float *, int, int, int);

static void *aligned_alloc_or_die(size_t bytes, size_t extra = 0) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 128, bytes + extra) != 0) {
        std::fprintf(stderr, "posix_memalign(%zu): %s\n", bytes + extra,
                     std::strerror(errno));
        std::exit(2);
    }
    return ptr;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static uint64_t ticks() {
    return mach_continuous_time();
}

static double seconds(uint64_t delta) {
    static mach_timebase_info_data_t info = [] {
        mach_timebase_info_data_t value{};
        mach_timebase_info(&value);
        return value;
    }();
    return static_cast<double>(delta) * static_cast<double>(info.numer) /
           static_cast<double>(info.denom) * 1e-9;
}

// M2 VECFP lane mode 1: bf16 X/Y, f32 Z (an interleaved Z-register pair).
// ALU mode 10 overwrites with X*Y on the first K block; mode 0 accumulates.
static uint64_t vecfp_bf16_f32(int pair, int yreg, bool first) {
    return (1ull << 42) | (static_cast<uint64_t>(pair * 2) << 20) |
           (static_cast<uint64_t>(yreg) * 64) |
           (first ? (10ull << 47) : 0);
}

// Match gemm_nt_impl's two F32x4 accumulators exactly: four VECFP operations
// consume 8 lanes apiece from each 32-BF16 load and accumulate into the same
// first eight Z lanes. The enable mask is "first 8 lanes" (mode 4, value 8).
static uint64_t vecfp_bf16_f32_exact8(int pair, int yreg, int sub, bool first) {
    return (1ull << 42) | (static_cast<uint64_t>(pair * 2) << 20) |
           (static_cast<uint64_t>(sub * 2) << 10) |
           (static_cast<uint64_t>(yreg) * 64 + static_cast<uint64_t>(sub * 2)) |
           (4ull << 38) | (8ull << 32) |
           (first ? (10ull << 47) : 0);
}

static float reduce32(const float *values) {
    float32x4_t a0 = vld1q_f32(values + 0);
    float32x4_t a1 = vld1q_f32(values + 4);
    float32x4_t a2 = vld1q_f32(values + 8);
    float32x4_t a3 = vld1q_f32(values + 12);
    float32x4_t a4 = vld1q_f32(values + 16);
    float32x4_t a5 = vld1q_f32(values + 20);
    float32x4_t a6 = vld1q_f32(values + 24);
    float32x4_t a7 = vld1q_f32(values + 28);
    a0 = vaddq_f32(a0, a1);
    a2 = vaddq_f32(a2, a3);
    a4 = vaddq_f32(a4, a5);
    a6 = vaddq_f32(a6, a7);
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a4, a6)));
}


// STZI reconstructs the interleaved Z-register pair as contiguous logical F32
// lanes. Perform the product leaf's exact acc0+acc1 and vaddvq order.
static float reduce_exact8(const float *values) {
    return vaddvq_f32(vaddq_f32(vld1q_f32(values), vld1q_f32(values + 4)));
}

__attribute__((noinline))
static void product_neon(const uint16_t *a, const void *w, float *out,
                         float *, int m, int n, int k) {
    lfm_bf16_gemm_nt_f32(a, w, out, m, n, k);
}

static inline uint16x8_t load_words8(const unsigned char *bytes) {
    uint16x8_t words;
    std::memcpy(&words, bytes, sizeof(words));
    return words;
}

static inline float32x4_t widen_lo(uint16x8_t words) {
    return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(words), 16));
}

static inline float32x4_t widen_hi(uint16x8_t words) {
    return vreinterpretq_f32_u32(vshll_high_n_u16(words, 16));
}

// Isolate dependency-chain pressure without changing the FMA instruction.
// Rows 1..4 use four vector accumulators per output; wider controls use two so
// every live partial stays in the architectural register file.
template <int Rows>
__attribute__((always_inline))
static inline void product_neon_fma_rows(const uint16_t *a,
                                         const void *weight_bytes,
                                         float *out, int n, int k) {
    constexpr int vecs = Rows <= 4 ? 4 : 2;
    constexpr int blocks = vecs / 2;
    const auto *weights = static_cast<const unsigned char *>(weight_bytes);
    for (int col = 0; col < n; ++col) {
        const auto *wr = weights + static_cast<size_t>(col) * k * sizeof(uint16_t);
        float32x4_t acc[Rows][vecs];
        for (int row = 0; row < Rows; ++row) {
            for (int lane = 0; lane < vecs; ++lane) {
                acc[row][lane] = vdupq_n_f32(0.0f);
            }
        }
        int offset = 0;
        for (; offset + blocks * 8 <= k; offset += blocks * 8) {
            uint16x8_t wb[blocks];
            for (int block = 0; block < blocks; ++block) {
                wb[block] = load_words8(wr + (offset + block * 8) * 2);
            }
            for (int row = 0; row < Rows; ++row) {
                const auto *ar = reinterpret_cast<const unsigned char *>(
                    a + static_cast<size_t>(row) * k);
                for (int block = 0; block < blocks; ++block) {
                    const uint16x8_t ab =
                        load_words8(ar + (offset + block * 8) * 2);
                    acc[row][block * 2] = vfmaq_f32(
                        acc[row][block * 2], widen_lo(ab), widen_lo(wb[block]));
                    acc[row][block * 2 + 1] = vfmaq_f32(
                        acc[row][block * 2 + 1], widen_hi(ab), widen_hi(wb[block]));
                }
            }
        }
        for (; offset + 8 <= k; offset += 8) {
            const uint16x8_t wb = load_words8(wr + offset * 2);
            for (int row = 0; row < Rows; ++row) {
                const auto *ar = reinterpret_cast<const unsigned char *>(
                    a + static_cast<size_t>(row) * k);
                const uint16x8_t ab = load_words8(ar + offset * 2);
                acc[row][0] = vfmaq_f32(acc[row][0], widen_lo(ab), widen_lo(wb));
                acc[row][1] = vfmaq_f32(acc[row][1], widen_hi(ab), widen_hi(wb));
            }
        }
        float sums[Rows];
        for (int row = 0; row < Rows; ++row) {
            if constexpr (vecs == 4) {
                sums[row] = vaddvq_f32(vaddq_f32(
                    vaddq_f32(acc[row][0], acc[row][1]),
                    vaddq_f32(acc[row][2], acc[row][3])));
            } else {
                sums[row] =
                    vaddvq_f32(vaddq_f32(acc[row][0], acc[row][1]));
            }
        }
        for (; offset < k; ++offset) {
            uint16_t wb;
            std::memcpy(&wb, wr + offset * 2, sizeof(wb));
            const float weight = std::bit_cast<float>(static_cast<uint32_t>(wb) << 16);
            for (int row = 0; row < Rows; ++row) {
                const uint16_t ab = a[static_cast<size_t>(row) * k + offset];
                const float value =
                    std::bit_cast<float>(static_cast<uint32_t>(ab) << 16);
                sums[row] = std::fma(value, weight, sums[row]);
            }
        }
        for (int row = 0; row < Rows; ++row) {
            out[static_cast<size_t>(row) * n + col] = sums[row];
        }
    }
}

__attribute__((noinline))
static void product_neon_fma4(const uint16_t *a, const void *weight_bytes,
                              float *out, float *, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    switch (m) {
        case 1: product_neon_fma_rows<1>(a, weight_bytes, out, n, k); return;
        case 2: product_neon_fma_rows<2>(a, weight_bytes, out, n, k); return;
        case 3: product_neon_fma_rows<3>(a, weight_bytes, out, n, k); return;
        case 4: product_neon_fma_rows<4>(a, weight_bytes, out, n, k); return;
        case 5: product_neon_fma_rows<5>(a, weight_bytes, out, n, k); return;
        case 6: product_neon_fma_rows<6>(a, weight_bytes, out, n, k); return;
        case 7: product_neon_fma_rows<7>(a, weight_bytes, out, n, k); return;
        case 8: product_neon_fma_rows<8>(a, weight_bytes, out, n, k); return;
        default: lfm_bf16_gemm_nt_f32(a, weight_bytes, out, m, n, k); return;
    }
}

static inline bfloat16x8_t load_bfdot8(const unsigned char *bytes) {
    return vreinterpretq_bf16_u16(load_words8(bytes));
}

// The missing fast-NEON cell. BFDOT consumes eight adjacent BF16 values from
// each raw checkpoint row and accumulates four pairwise dot products in F32.
// Compile-time row counts keep the accumulator set in the vector register file:
// four chains per row through M=4 and two per row above that.
// This changes both the reduction tree and BFDOT's arithmetic semantics; the
// parity and cancellation gates below therefore treat it as a separate fast
// numerical contract, not as a faithful implementation.
template <int Rows>
__attribute__((always_inline, target("bf16,neon")))
static inline void product_neon_bfdot_rows(const uint16_t *a,
                                           const void *weight_bytes,
                                           float *out, int n, int k) {
    constexpr int chains = Rows <= 4 ? 4 : 2;
    const auto *weights = static_cast<const unsigned char *>(weight_bytes);
    for (int col = 0; col < n; ++col) {
        const auto *wr = weights + static_cast<size_t>(col) * k * sizeof(uint16_t);
        __builtin_prefetch(wr + static_cast<size_t>(k) * sizeof(uint16_t), 0, 0);
        float32x4_t acc[Rows][chains];
        for (int row = 0; row < Rows; ++row) {
            for (int chain = 0; chain < chains; ++chain) {
                acc[row][chain] = vdupq_n_f32(0.0f);
            }
        }

        int offset = 0;
        for (; offset + chains * 8 <= k; offset += chains * 8) {
            bfloat16x8_t wb[chains];
            for (int chain = 0; chain < chains; ++chain) {
                wb[chain] = load_bfdot8(wr + (offset + chain * 8) * 2);
            }
            for (int row = 0; row < Rows; ++row) {
                const auto *ar = reinterpret_cast<const unsigned char *>(
                    a + static_cast<size_t>(row) * k);
                for (int chain = 0; chain < chains; ++chain) {
                    acc[row][chain] = vbfdotq_f32(
                        acc[row][chain],
                        load_bfdot8(ar + (offset + chain * 8) * 2), wb[chain]);
                }
            }
        }
        for (; offset + 8 <= k; offset += 8) {
            const bfloat16x8_t wb = load_bfdot8(wr + offset * 2);
            for (int row = 0; row < Rows; ++row) {
                const auto *ar = reinterpret_cast<const unsigned char *>(
                    a + static_cast<size_t>(row) * k);
                acc[row][0] = vbfdotq_f32(acc[row][0],
                                          load_bfdot8(ar + offset * 2), wb);
            }
        }

        float sums[Rows];
        for (int row = 0; row < Rows; ++row) {
            if constexpr (chains == 4) {
                sums[row] = vaddvq_f32(vaddq_f32(
                    vaddq_f32(acc[row][0], acc[row][1]),
                    vaddq_f32(acc[row][2], acc[row][3])));
            } else {
                sums[row] =
                    vaddvq_f32(vaddq_f32(acc[row][0], acc[row][1]));
            }
        }
        for (; offset < k; ++offset) {
            uint16_t wb;
            std::memcpy(&wb, wr + offset * 2, sizeof(wb));
            const float weight = std::bit_cast<float>(static_cast<uint32_t>(wb) << 16);
            for (int row = 0; row < Rows; ++row) {
                const uint16_t ab = a[static_cast<size_t>(row) * k + offset];
                const float value =
                    std::bit_cast<float>(static_cast<uint32_t>(ab) << 16);
                sums[row] = std::fma(value, weight, sums[row]);
            }
        }
        for (int row = 0; row < Rows; ++row) {
            out[static_cast<size_t>(row) * n + col] = sums[row];
        }
    }
}

__attribute__((noinline, target("bf16,neon")))
static void product_neon_bfdot(const uint16_t *a, const void *weight_bytes,
                               float *out, float *, int m, int n, int k) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    switch (m) {
        case 1: product_neon_bfdot_rows<1>(a, weight_bytes, out, n, k); return;
        case 2: product_neon_bfdot_rows<2>(a, weight_bytes, out, n, k); return;
        case 3: product_neon_bfdot_rows<3>(a, weight_bytes, out, n, k); return;
        case 4: product_neon_bfdot_rows<4>(a, weight_bytes, out, n, k); return;
        case 5: product_neon_bfdot_rows<5>(a, weight_bytes, out, n, k); return;
        case 6: product_neon_bfdot_rows<6>(a, weight_bytes, out, n, k); return;
        case 7: product_neon_bfdot_rows<7>(a, weight_bytes, out, n, k); return;
        case 8: product_neon_bfdot_rows<8>(a, weight_bytes, out, n, k); return;
        default: lfm_bf16_gemm_nt_f32(a, weight_bytes, out, m, n, k); return;
    }
}

// The AMX state has eight 64-byte Y registers. Current production needs M<=4;
// the M=7 Conformer control also fits. Every output row uses one interleaved
// F32 Z pair. STZI reconstructs each 32-lane vector in two 64-byte stores.
__attribute__((noinline))
static void amx_inline(const uint16_t *a, const void *weight_bytes, float *out,
                       float *, int m, int n, int k) {
    if (m <= 0 || m > 8 || n <= 0 || k <= 0 || (k % kLanes) != 0) return;
    const auto *weights = static_cast<const unsigned char *>(weight_bytes);
    alignas(128) float partial[8][kLanes];

    AMX_SET();
    for (int col = 0; col < n; ++col) {
        const auto *row = weights + static_cast<size_t>(col) * k * sizeof(uint16_t);
        for (int offset = 0; offset < k; offset += kLanes) {
            AMX_LDX(reinterpret_cast<uint64_t>(row +
                                               static_cast<size_t>(offset) * 2));
            for (int lane = 0; lane < m; ++lane) {
                AMX_LDY(reinterpret_cast<uint64_t>(a +
                        static_cast<size_t>(lane) * k + offset) |
                        (static_cast<uint64_t>(lane) << 56));
            }
            for (int lane = 0; lane < m; ++lane) {
                AMX_VECFP(vecfp_bf16_f32(lane, lane, offset == 0));
            }
        }
        for (int lane = 0; lane < m; ++lane) {
            AMX_STZI(reinterpret_cast<uint64_t>(partial[lane]) |
                     (static_cast<uint64_t>(lane) << 57));
            AMX_STZI(reinterpret_cast<uint64_t>(partial[lane] + 16) |
                     (static_cast<uint64_t>(lane) << 57) | (1ull << 56));
            out[static_cast<size_t>(lane) * n + col] = reduce32(partial[lane]);
        }
    }
    AMX_CLR();
}

// Produce the complete partial plane and return. This is the strongest honest
// model of a suspend-at-AMX / resume-in-NEON seam available without installing
// a scheduler in the microbenchmark: all producer state has crossed memory and
// the consumer is a distinct noinline call.
__attribute__((noinline))
static void amx_plane(const uint16_t *a, const void *weight_bytes, float *,
                      float *scratch, int m, int n, int k) {
    if (m <= 0 || m > 8 || n <= 0 || k <= 0 || (k % kLanes) != 0) return;
    const auto *weights = static_cast<const unsigned char *>(weight_bytes);

    AMX_SET();
    for (int col = 0; col < n; ++col) {
        const auto *row = weights + static_cast<size_t>(col) * k * sizeof(uint16_t);
        for (int offset = 0; offset < k; offset += kLanes) {
            AMX_LDX(reinterpret_cast<uint64_t>(row +
                                               static_cast<size_t>(offset) * 2));
            for (int lane = 0; lane < m; ++lane) {
                AMX_LDY(reinterpret_cast<uint64_t>(a +
                        static_cast<size_t>(lane) * k + offset) |
                        (static_cast<uint64_t>(lane) << 56));
            }
            for (int lane = 0; lane < m; ++lane) {
                AMX_VECFP(vecfp_bf16_f32(lane, lane, offset == 0));
            }
        }
        for (int lane = 0; lane < m; ++lane) {
            float *dst = scratch + (static_cast<size_t>(lane) * n + col) * kLanes;
            AMX_STZI(reinterpret_cast<uint64_t>(dst) |
                     (static_cast<uint64_t>(lane) << 57));
            AMX_STZI(reinterpret_cast<uint64_t>(dst + 16) |
                     (static_cast<uint64_t>(lane) << 57) | (1ull << 56));
        }
    }
    AMX_CLR();
}

__attribute__((noinline))
static void amx_exact_plane(const uint16_t *a, const void *weight_bytes, float *,
                            float *scratch, int m, int n, int k) {
    if (m <= 0 || m > 8 || n <= 0 || k <= 0 || (k % kLanes) != 0) return;
    const auto *weights = static_cast<const unsigned char *>(weight_bytes);

    AMX_SET();
    for (int col = 0; col < n; ++col) {
        const auto *row = weights + static_cast<size_t>(col) * k * sizeof(uint16_t);
        for (int offset = 0; offset < k; offset += kLanes) {
            AMX_LDX(reinterpret_cast<uint64_t>(row +
                                               static_cast<size_t>(offset) * 2));
            for (int lane = 0; lane < m; ++lane) {
                AMX_LDY(reinterpret_cast<uint64_t>(a +
                        static_cast<size_t>(lane) * k + offset) |
                        (static_cast<uint64_t>(lane) << 56));
            }
            for (int sub = 0; sub < kLanes; sub += 8) {
                for (int lane = 0; lane < m; ++lane) {
                    AMX_VECFP(vecfp_bf16_f32_exact8(
                        lane, lane, sub, offset == 0 && sub == 0));
                }
            }
        }
        for (int lane = 0; lane < m; ++lane) {
            float *dst = scratch + (static_cast<size_t>(lane) * n + col) * kLanes;
            AMX_STZI(reinterpret_cast<uint64_t>(dst) |
                     (static_cast<uint64_t>(lane) << 57));
            AMX_STZI(reinterpret_cast<uint64_t>(dst + 16) |
                     (static_cast<uint64_t>(lane) << 57) | (1ull << 56));
        }
    }
    AMX_CLR();
}

__attribute__((noinline))
static void consume_plane(const uint16_t *, const void *, float *out,
                          float *scratch, int m, int n, int) {
    for (int lane = 0; lane < m; ++lane) {
        for (int col = 0; col < n; ++col) {
            const float *src = scratch +
                (static_cast<size_t>(lane) * n + col) * kLanes;
            out[static_cast<size_t>(lane) * n + col] = reduce32(src);
        }
    }
}

__attribute__((noinline))
static void consume_exact_plane(const uint16_t *, const void *, float *out,
                                float *scratch, int m, int n, int) {
    for (int lane = 0; lane < m; ++lane) {
        for (int col = 0; col < n; ++col) {
            const float *src = scratch +
                (static_cast<size_t>(lane) * n + col) * kLanes;
            out[static_cast<size_t>(lane) * n + col] = reduce_exact8(src);
        }
    }
}

__attribute__((noinline))
static void amx_split(const uint16_t *a, const void *w, float *out,
                      float *scratch, int m, int n, int k) {
    amx_plane(a, w, out, scratch, m, n, k);
    consume_plane(a, w, out, scratch, m, n, k);
}

__attribute__((noinline))
static void amx_exact_split(const uint16_t *a, const void *w, float *out,
                            float *scratch, int m, int n, int k) {
    amx_exact_plane(a, w, out, scratch, m, n, k);
    consume_exact_plane(a, w, out, scratch, m, n, k);
}

static uint64_t hash_output(const float *out, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count; ++i) {
        hash ^= std::bit_cast<uint32_t>(out[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

static Stats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto at = [&](double q) {
        return values[static_cast<size_t>(q * static_cast<double>(values.size() - 1))];
    };
    return {at(0.5), at(0.1), at(0.9)};
}

static int ordered_ulp(float lhs, float rhs) {
    const auto key = [](float value) {
        int32_t bits = std::bit_cast<int32_t>(value);
        return bits < 0 ? std::numeric_limits<int32_t>::min() - bits : bits;
    };
    const int64_t delta = static_cast<int64_t>(key(lhs)) - key(rhs);
    return static_cast<int>(std::min<int64_t>(std::llabs(delta),
                                              std::numeric_limits<int>::max()));
}

static void compare(const char *name, const float *reference, const float *actual,
                    size_t count) {
    double max_abs = 0;
    int max_ulp = 0;
    size_t f32_mismatch = 0;
    size_t bf16_mismatch = 0;
    for (size_t i = 0; i < count; ++i) {
        if (std::bit_cast<uint32_t>(reference[i]) !=
            std::bit_cast<uint32_t>(actual[i])) {
            ++f32_mismatch;
        }
        if (f32_to_bf16(reference[i]) != f32_to_bf16(actual[i])) {
            ++bf16_mismatch;
        }
        max_abs = std::max(max_abs,
                           std::abs(static_cast<double>(reference[i]) - actual[i]));
        max_ulp = std::max(max_ulp, ordered_ulp(reference[i], actual[i]));
    }
    std::printf("  parity %-16s f32-diff=%zu/%zu bf16-diff=%zu/%zu "
                "max-abs=%.3e max-ulp=%d\n",
                name, f32_mismatch, count, bf16_mismatch, count, max_abs, max_ulp);
}

static void fill_bf16(uint16_t *dst, size_t count, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const int value = static_cast<int>((state * 2685821657736338717ull) >> 53) - 1024;
        dst[i] = f32_to_bf16(static_cast<float>(value) / 2048.0f);
    }
}

static void run_shape(const Shape &shape) {
    const size_t acount = static_cast<size_t>(shape.m) * shape.k;
    const size_t wcount = static_cast<size_t>(shape.n) * shape.k;
    const size_t ccount = static_cast<size_t>(shape.m) * shape.n;
    const size_t scount = ccount * kLanes;

    auto *abase = static_cast<unsigned char *>(aligned_alloc_or_die(acount * 2 + 2));
    auto *wbase = static_cast<unsigned char *>(aligned_alloc_or_die(wcount * 2 + 2));
    auto *a = reinterpret_cast<uint16_t *>(abase + 2);
    auto *w = reinterpret_cast<uint16_t *>(wbase + 2);
    auto *reference = static_cast<float *>(aligned_alloc_or_die(ccount * 4));
    auto *out = static_cast<float *>(aligned_alloc_or_die(ccount * 4));
    auto *scratch = static_cast<float *>(aligned_alloc_or_die(scount * 4));

    fill_bf16(a, acount, 0x123456789abcdef0ull ^ wcount);
    fill_bf16(w, wcount, 0xfedcba9876543210ull ^ acount);
    std::fill(reference, reference + ccount, std::numeric_limits<float>::quiet_NaN());
    std::fill(out, out + ccount, std::numeric_limits<float>::quiet_NaN());
    std::fill(scratch, scratch + scount, std::numeric_limits<float>::quiet_NaN());

    std::printf("\n== %s M=%d N=%d K=%d  W=%.1f MiB  split-scratch=%.1f MiB ==\n",
                shape.name, shape.m, shape.n, shape.k,
                static_cast<double>(wcount * 2) / (1024.0 * 1024.0),
                static_cast<double>(scount * 4) / (1024.0 * 1024.0));

    product_neon(a, w, reference, scratch, shape.m, shape.n, shape.k);
    product_neon_fma4(a, w, out, scratch, shape.m, shape.n, shape.k);
    compare("NEON FMA fast", reference, out, ccount);
    product_neon_bfdot(a, w, out, scratch, shape.m, shape.n, shape.k);
    compare("NEON BFDOT", reference, out, ccount);
    amx_inline(a, w, out, scratch, shape.m, shape.n, shape.k);
    compare("AMX inline", reference, out, ccount);
    amx_split(a, w, out, scratch, shape.m, shape.n, shape.k);
    compare("AMX split", reference, out, ccount);
    amx_exact_split(a, w, out, scratch, shape.m, shape.n, shape.k);
    compare("AMX exact8", reference, out, ccount);

    struct Candidate { const char *name; Kernel fn; };
    const std::array<Candidate, 6> candidates{{
        {"NEON FMA faithful", product_neon},
        {"NEON FMA fast", product_neon_fma4},
        {"NEON BFDOT RO/FTZ", product_neon_bfdot},
        {"AMX tile-local", amx_inline},
        {"AMX fast32 split", amx_split},
        {"AMX exact8 split", amx_exact_split},
    }};
    std::array<std::vector<double>, 6> samples;
    std::mt19937 order(0x51d3u + shape.n + shape.k);

    // Warm each path and leave a valid partial plane for the consumer-only case.
    for (const Candidate &candidate : candidates) {
        candidate.fn(a, w, out, scratch, shape.m, shape.n, shape.k);
    }
    for (int round = 0; round < kRounds; ++round) {
        std::array<int, 6> indices{0, 1, 2, 3, 4, 5};
        std::shuffle(indices.begin(), indices.end(), order);
        for (int index : indices) {
            std::fill(out, out + ccount, std::numeric_limits<float>::quiet_NaN());
            Kernel volatile opaque = candidates[index].fn;
            const uint64_t start = ticks();
            opaque(a, w, out, scratch, shape.m, shape.n, shape.k);
            __asm__ volatile("" : : : "memory");
            const double elapsed = seconds(ticks() - start);
            samples[index].push_back(elapsed);
            g_sink ^= hash_output(out, ccount) + static_cast<uint64_t>(round + index);
        }
    }

    const double weight_gb = static_cast<double>(wcount * 2) / 1e9;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Stats stats = summarize(samples[i]);
        const double bandwidth = weight_gb / stats.median;
        std::printf("  %-21s median=%7.3f ms  p10=%7.3f  p90=%7.3f",
                    candidates[i].name, stats.median * 1e3,
                    stats.p10 * 1e3, stats.p90 * 1e3);
        std::printf("  checkpoint=%.1f GB/s", bandwidth);
        std::printf("\n");
        if (g_raw) {
            std::printf("    raw-us:");
            for (double sample : samples[i]) std::printf(" %.3f", sample * 1e6);
            std::printf("\n");
        }
    }

    // Diagnostic only, outside the contestant shuffle. Rebuild the fast32
    // plane before every sample so the consumer never observes a predecessor's
    // exact8 or stale scratch contents.
    std::vector<double> reduce_samples;
    for (int round = 0; round < kRounds; ++round) {
        amx_plane(a, w, out, scratch, shape.m, shape.n, shape.k);
        std::fill(out, out + ccount, std::numeric_limits<float>::quiet_NaN());
        const uint64_t start = ticks();
        consume_plane(a, w, out, scratch, shape.m, shape.n, shape.k);
        __asm__ volatile("" : : : "memory");
        reduce_samples.push_back(seconds(ticks() - start));
        g_sink ^= hash_output(out, ccount) + static_cast<uint64_t>(round);
    }
    const Stats reduce = summarize(reduce_samples);
    std::printf("  %-21s median=%7.3f ms  p10=%7.3f  p90=%7.3f\n",
                "NEON reduce32 hot", reduce.median * 1e3,
                reduce.p10 * 1e3, reduce.p90 * 1e3);
    if (g_raw) {
        std::printf("    raw-us:");
        for (double sample : reduce_samples) std::printf(" %.3f", sample * 1e6);
        std::printf("\n");
    }

    std::free(abase);
    std::free(wbase);
    std::free(reference);
    std::free(out);
    std::free(scratch);
}

static void cancellation_probe() {
    constexpr int m = 1;
    constexpr int n = 4;
    constexpr int k = 64;
    alignas(128) uint16_t a[m * k];
    alignas(128) uint16_t w[n * k];
    alignas(128) float reference[m * n];
    alignas(128) float fma4[m * n];
    alignas(128) float bfdot[m * n];
    alignas(128) float actual[m * n];
    alignas(128) float scratch[m * n * kLanes];
    std::fill(a, a + m * k, f32_to_bf16(1.0f));
    std::fill(w, w + n * k, f32_to_bf16(0.0f));

    // NEON's lane-0 accumulator observes big, tiny, tiny, tiny, -big in that
    // order. AMX keeps k mod 32 in separate lanes, so it cancels the big pair
    // before the horizontal reduction and preserves the tiny terms. Both are
    // valid F32 reductions; only one matches the production leaf.
    for (int row = 0; row < n; ++row) {
        const float tiny = std::ldexp(1.0f + static_cast<float>(row) / 8.0f, -14);
        w[row * k + 0] = f32_to_bf16(32768.0f);
        w[row * k + 8] = f32_to_bf16(tiny);
        w[row * k + 16] = f32_to_bf16(tiny);
        w[row * k + 24] = f32_to_bf16(tiny);
        w[row * k + 32] = f32_to_bf16(-32768.0f);
    }
    product_neon(a, w, reference, scratch, m, n, k);
    product_neon_fma4(a, w, fma4, scratch, m, n, k);
    product_neon_bfdot(a, w, bfdot, scratch, m, n, k);
    amx_split(a, w, actual, scratch, m, n, k);

    std::printf("# cancellation probe (reduction-order gate)\n");
    for (int row = 0; row < n; ++row) {
        std::printf("  row=%d faithful=% .9g FMA-fast=% .9g BFDOT=% .9g AMX=% .9g "
                    "bf16=%04x/%04x/%04x/%04x\n",
                    row, reference[row], fma4[row], bfdot[row], actual[row],
                    f32_to_bf16(reference[row]), f32_to_bf16(fma4[row]),
                    f32_to_bf16(bfdot[row]), f32_to_bf16(actual[row]));
    }
    compare("cancel FMA fast", reference, fma4, n);
    compare("cancel BFDOT", reference, bfdot, n);
    compare("cancel AMX", reference, actual, n);
    amx_exact_split(a, w, actual, scratch, m, n, k);
    compare("cancel exact8", reference, actual, n);
    std::printf("\n");
}

// BFDOT combines adjacent BF16 products inside one instruction; AMX fast32
// keeps them in distinct F32 lanes. Put a tiny value beside the large positive
// term so the pair operation can lose it before the later cancellation. This
// prevents the first adversary's matching BFDOT/AMX result from being mistaken
// for an identical fast numerical contract.
static void pair_probe() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 64;
    alignas(128) uint16_t a[m * k];
    alignas(128) uint16_t w[n * k];
    alignas(128) float faithful[m * n];
    alignas(128) float fma4[m * n];
    alignas(128) float bfdot[m * n];
    alignas(128) float amx[m * n];
    alignas(128) float exact[m * n];
    alignas(128) float scratch[m * n * kLanes];
    std::fill(a, a + m * k, f32_to_bf16(1.0f));
    std::fill(w, w + n * k, f32_to_bf16(0.0f));
    w[0] = f32_to_bf16(32768.0f);
    w[1] = f32_to_bf16(std::ldexp(1.0f, -14));
    w[32] = f32_to_bf16(-32768.0f);

    product_neon(a, w, faithful, scratch, m, n, k);
    product_neon_fma4(a, w, fma4, scratch, m, n, k);
    product_neon_bfdot(a, w, bfdot, scratch, m, n, k);
    amx_split(a, w, amx, scratch, m, n, k);
    amx_exact_split(a, w, exact, scratch, m, n, k);

    std::printf("# adjacent-pair probe (BFDOT arithmetic gate)\n");
    std::printf("  faithful=% .9g FMA-fast=% .9g BFDOT=% .9g AMX=% .9g exact8=% .9g\n",
                faithful[0], fma4[0], bfdot[0], amx[0], exact[0]);
    std::printf("  isolated-dot-bf16=%04x/%04x/%04x/%04x/%04x\n\n",
                f32_to_bf16(faithful[0]), f32_to_bf16(fma4[0]),
                f32_to_bf16(bfdot[0]), f32_to_bf16(amx[0]),
                f32_to_bf16(exact[0]));
}

} // namespace

int main(int argc, char **argv) {
    g_raw = argc == 2 && std::string_view(argv[1]) == "--raw";
    int bf16 = 0;
    size_t size = sizeof(bf16);
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &bf16, &size, nullptr, 0) != 0 ||
        bf16 != 1) {
        std::fprintf(stderr, "M2 BF16 support is required.\n");
        return 1;
    }

    std::printf("# immutable checkpoint view: BF16 W[N,K], deliberately +2-byte unaligned\n");
    std::printf("# effective checkpoint bandwidth counts exactly 2*N*K source bytes\n");
    std::printf("# MATFP excluded: it needs a forbidden KxN/packed weight view\n");
    std::printf("# bf16-diff rounds the isolated dot; full fused-stage parity is separate\n");
    std::printf("# all numerical paths use one execution thread; accumulator chains vary\n");
    std::printf("# samples: 17 randomized paired rounds; summary is median/p10/p90%s\n\n",
                g_raw ? "; raw microseconds follow" : "");

    cancellation_probe();
    pair_probe();

    const std::array<Shape, 4> shapes{{
        {1, 2048, 2048, "backbone decode"},
        {4, 8192, 2048, "backbone up"},
        {4, 2048, 8192, "backbone down"},
        {7, 2048, 512, "Conformer adapter"},
    }};
    for (const Shape &shape : shapes) run_shape(shape);
    std::printf("\nsink=%016llx\n", static_cast<unsigned long long>(g_sink));
    return 0;
}
