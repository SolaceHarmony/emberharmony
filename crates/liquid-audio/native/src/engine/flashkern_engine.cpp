// flashkern_engine.cpp — the resident native decode engine (ENGINE_DESIGN.md §2/§3),
// as a LANE-UNIFORM KERNEL: the engine owns all mutable state, and every lane runs
// the ENTIRE pass program — embed, every layer, final norm — exactly the way a GPU
// threadgroup runs a kernel. There is no coordinator publishing stages to workers:
// stages are separated by generation fences on fixed OS threads, tiles
// are claimed off a bare fetch_add counter (so an E-core straggler simply claims
// fewer), and each fence's last arriver runs that boundary's serial ladder work
// (sumsq folds, conv update, qk-norm/rope/append, embed) exactly once. The only
// runtime boundary is a fixed submission/completion bridge: a native dispatcher
// release-rings one pass descriptor, and lane 0 publishes one exact CQ record after
// the program-final fence.
//
// The compatibility Rust ABI still invokes blocking test/conformance calls while its
// borrowed buffers remain live. Native code owns SQ submission, CQ consumption, and
// pass recurrence; Rust is not a numerical-progress dependency. Stop remains a
// full-pass boundary decision and is never polled inside assembly operations.
//
// Numerics: stage bodies are line-for-line ports of src/compute/flashkern/decode.rs
// (fused_mlp_decode) — same RNE bf16 rounding ladder, same FIXED tile count and
// fixed-order partial fold (deterministic regardless of which worker runs which
// tile), same kernels (lfm_bf16_gemm_nt_f32, linked in-image). The Rust parity test
// pins this bit-identical to the threadgroup port, itself pinned to the candle chain.
//
// Build: -ffp-contract=off (the ladders promise separate roundings), C++23.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <pthread.h>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashkern_conv.h"
#include "flashkern_depth.h"
#include "flashkern_fft.h"
#include "flashkern_gemm.h"
#include "flashkern_math.h"
#include "flashkern_prng.h"
#include "flashkern_sampler.h"
#include "lfm_audio_pass.h"
#include "lfm_kernel_bridge.h"
#include "lfm_mimi.h"
#include "lfm_model_plan.h"
#include "../model/lfm_route_epoch.h"

extern "C" {
#include "kc_atomic.h"
#include "kc_collective.h"
#include "kc_port.h"
#include "kc_team.h"
}

// Stage kernels from the flashkern TU (same image, plain calls).
extern "C" float lfm_bf16_sumsq_candle_f32(const void *x_bytes, int n);
extern "C" float lfm_bf16_sumsq_f32(const uint16_t *x, int n);
extern "C" void lfm_bf16_rmsnorm(const void *x_bytes, const void *weight_bytes, uint16_t *out,
                                 int n, float inv_rms);
extern "C" void lfm_f32_to_bf16(const float *x, uint16_t *out, int n);
extern "C" void lfm_bf16_add(const void *a_bytes, const void *b_bytes,
                              uint16_t *out, int n);
extern "C" void lfm_conv1d_update_bf16(const uint16_t *bcx, const uint16_t *state,
                                       const void *weight_bytes, uint16_t *out, int bn, int d,
                                       int t, int k);
extern "C" void lfm_bf16_to_f32(const uint16_t *x, float *out, int n);
extern "C" void lfm_softmax_scaled_f32(float *x, int n, float scale);
extern "C" void lfm_attn_qk_bf16(const float *q, const uint16_t *k, float *att, int len,
                                 int hd);
extern "C" void lfm_attn_av_bf16(const float *att, const uint16_t *v, float *out,
                                 int len, int hd);
extern "C" void lfm_rope_i_f32(float *x, const float *cos_p, const float *sin_p, int hd);
extern "C" void lfm_swiglu_bf16(const float *g, const float *u, uint16_t *out, int n);

namespace {

constexpr int MAX_WORKERS = 16;
constexpr size_t PASS_CAPACITY = 2;
/* Runtime admission permits at most 64 sessions and each conversation admits
 * one mutating program. Matching that bound makes route admission lossless:
 * backpressure occurs at the session queues, never as an unobservable broker
 * retry after a playback lease has already been reserved. */
constexpr size_t ROUTE_CAPACITY = 64;
/* Starvation is measured in broker enqueue epochs, not pool occupancy. Keep the
 * policy independent from ROUTE_CAPACITY so changing admission memory cannot
 * silently retune service-class promotion. */
constexpr uint64_t ROUTE_AGE_PROMOTION = 64;
// Apple Silicon and every Apple-hosted slice (including Rosetta) run on
// 128-byte cache lines.  Keeping this conservative size on other targets is
// harmless and prevents adjacent expected-value words from sharing a line.
constexpr size_t ENGINE_CACHELINE = 128;
constexpr uint32_t PASS_ADMISSION_EXCLUSIVE = uint32_t{1} << 31;
constexpr uint32_t PASS_ADMISSION_COUNT = PASS_ADMISSION_EXCLUSIVE - 1;
constexpr size_t DOWN_BAND_CAP = 512; // worker-stack y[] extent
constexpr size_t PREFILL_ROWS = LFM_PREFILL_MAX_ROWS;
constexpr size_t TOKEN_INPUT_MAX_IDS = 8;
std::atomic<uint64_t> next_engine_epoch{1};

static bool checked_size_product(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) return false;
    *out = left * right;
    return true;
}

// ---- rounding helpers: exact ports of decode.rs ------------------------------------
static inline float bf16_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static inline uint16_t rb_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

using WeightBytes = const uint8_t *;

static inline WeightBytes weight_offset(WeightBytes base, size_t elements) {
    return base + elements * sizeof(uint16_t);
}

static inline WeightBytes activation_bytes(const uint16_t *values) {
    return reinterpret_cast<WeightBytes>(values);
}

// An input to the first backbone layer is either an aligned activation plane or
// a possibly byte-unaligned immutable embedding row. Keeping the two pointer
// types distinct prevents C++ from ever manufacturing a dereferenceable
// uint16_t* for checkpoint storage. Selection is once per stage, never per
// element and never per dtype.
struct Bf16Input {
    const uint16_t *activation = nullptr;
    WeightBytes resident = nullptr;

    static Bf16Input from_activation(const uint16_t *values) {
        return {.activation = values};
    }
    static Bf16Input from_resident(WeightBytes values) {
        return {.resident = values};
    }
    const void *data() const {
        return resident ? static_cast<const void *>(resident)
                        : static_cast<const void *>(activation);
    }
    Bf16Input offset(size_t elements) const {
        return resident ? from_resident(weight_offset(resident, elements))
                        : from_activation(activation + elements);
    }
};

// ---- the pass (engine-owned pointers; nothing here ever rides a message) ------------
struct Pass {
    const uint16_t *x;      // [h] bf16 bits
    WeightBytes norm_w;      // [h], resident checkpoint bytes
    WeightBytes w1;          // [i,h]
    WeightBytes w3;          // [i,h]
    WeightBytes w2;          // [h,i]
    uint16_t *out;          // [h]
    size_t h, i;
    size_t tiles; // FIXED — the deterministic partial/fold order
    float eps;
    // engine-owned scratch planes
    float *partials; // [tiles]
    uint16_t *xn;    // [h]
    float *gu;       // [2i]
    uint16_t *t;     // [i]
    std::atomic<uint32_t> rs_bits{0};
};

// Immutable-for-one-ticket MLP request metadata. The active `Pass` below also
// carries fence-published scalar state and pointers into the selected scratch
// bank, so it is assembled only when the dispatcher selects this ticket.
struct MlpReq {
    const uint16_t *x = nullptr;
    WeightBytes norm_w = nullptr;
    WeightBytes w1 = nullptr;
    WeightBytes w3 = nullptr;
    WeightBytes w2 = nullptr;
    uint16_t *out = nullptr;
    size_t h = 0;
    size_t i = 0;
    size_t tiles = 0;
    float eps = 0.0f;
};

// ---- the stage board ------------------------------------------------------------------
enum : uint32_t {
    ST_IDLE = 0,
    ST_SUMSQ = 1,
    ST_NORM = 2,
    ST_GATEUP = 3,
    ST_DOWN = 4,
    // ShortConv block stages (decode.rs fused_shortconv_decode, ported verbatim).
    ST_SC_NORM = 5,    // rmsnorm band via lfm_bf16_rmsnorm (inv_rms from Pass::rs_bits)
    ST_SC_INPROJ = 6,  // in_proj rows band: nt + f32→bf16 round
    ST_SC_GATHER = 7,  // y gather ([c][0]) + carried-state copy ([c][1..K]) band
    ST_SC_OUTPROJ = 8, // out_proj rows band: nt + round + residual add
    // Attention block stages (attn_decode_bf16 + its candle wrapper ops, ported).
    ST_AT_QKV = 9,    // q|k|v projection rows band (3-segment routing) + round
    ST_AT_HEAD = 10,  // one q head: qk dots over the K plane, softmax, av, round
    ST_AT_OPROJ = 11, // o_proj rows band (k = nh·hd) + round + residual add
    ST_LOGITS = 12,   // tied-head rows band: nt → bf16 round → exact f32 widen
};

struct Stage {
    // Bare tile-claim counter, reset in the opening fence's serial section. No epoch,
    // no completion count: fences guarantee no lane can straddle two stages, so a
    // claim can never be stale, and "all lanes arrived at the next fence" IS the
    // completion proof (a lane only arrives after finishing its claimed tiles).
    std::atomic<uint32_t> next{0};
    uint32_t kind = ST_IDLE; // written in the fence serial, read after fence exit
    uint32_t count = 0;      // number of tiles this stage
    uint32_t chunk = 0;      // band width for banded stages
};

enum : int {
    REQ_NONE = 0,
    REQ_MLP = 1,
    REQ_CONV_LAYER = 2,
    REQ_ATTN_LAYER = 3,
    REQ_TOKEN_PASS = 4,
    // Request kind 5 was the deleted caller-supplied Rust lane callback.
    // Conversation-owned ChaCha state advances exactly once in a collective
    // serial section. Production sampling consumes the same primitive inside
    // token/depthformer passes; this typed pass is its SQ/CQ conformance leaf.
    REQ_PRNG = 6,
    // Standalone sampling is the fallback/conformance leaf. Production token
    // and Depthformer passes call the same collective without another ticket.
    REQ_SAMPLE = 7,
    // Complete Depthformer frame: projection, all codebooks/layers, integrated
    // sampler, and sampled-embedding recurrence under one native ticket.
    REQ_DEPTH_FRAME = 8,
    // CPU depthwise-stream contract expressed as one fixed-team pass. The
    // prior state and incoming chunk stay as separate borrowed buffers.
    REQ_DEPTHWISE_STREAM = 9,
    // Architecture GEMM/GEMV leaves under one fixed-team launch. Rust owns
    // neither tile geometry nor a callback frame on the compute workers.
    REQ_GEMM = 10,
    REQ_FFT_CONV_DD = 11,
    REQ_IRFFT_DD = 12,
    REQ_PREFILL = 13,
    // One conversation-owned Mimi state step writes directly into a retained
    // playback reservation. Lane 0 runs the stateful graph while peer lanes
    // park at the pass fence; codec work therefore shares SQ/CQ ordering and
    // cannot oversubscribe the backbone/Depthformer executor.
    REQ_MIMI_DECODE = 14,
    // One retained PCM view through prepared resample, frontend, and Conformer
    // workspaces. Conformer GEMMs are fixed-team substages of this ticket and
    // never recurse through the bridge.
    REQ_AUDIO_ENCODE = 15,
};

static constexpr bool request_kind_valid(uint32_t kind) {
    switch (kind) {
    case REQ_MLP:
    case REQ_CONV_LAYER:
    case REQ_ATTN_LAYER:
    case REQ_TOKEN_PASS:
    case REQ_PRNG:
    case REQ_SAMPLE:
    case REQ_DEPTH_FRAME:
    case REQ_DEPTHWISE_STREAM:
    case REQ_GEMM:
    case REQ_FFT_CONV_DD:
    case REQ_IRFFT_DD:
    case REQ_PREFILL:
    case REQ_MIMI_DECODE:
    case REQ_AUDIO_ENCODE:
        return true;
    default:
        return false;
    }
}

static constexpr bool logical_lane_count_valid(size_t lanes) {
    return lanes >= 1 && lanes <= static_cast<size_t>(MAX_WORKERS);
}

struct PrngReq {
    LfmPrngStateV1 *state = nullptr;
    uint64_t *out = nullptr;
    size_t count = 0;
};

enum : uint32_t {
    SAMPLE_F32 = 1,
    SAMPLE_BF16 = 2,
};

struct SampleReq {
    const void *logits = nullptr;
    size_t count = 0;
    uint32_t dtype = 0;
    LfmSamplerConfigV1 config{};
    LfmPrngStateV1 *state = nullptr;
    uint32_t *out = nullptr;
};

struct DepthReq {
    const uint16_t *hidden = nullptr;
    LfmSamplerConfigV1 sampler{};
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_tokens = nullptr;
    int completion_status = 0; // private deterministic route-fault seam
};

struct DepthwiseStreamReq {
    const uint16_t *x = nullptr;
    const uint16_t *cache = nullptr;
    const uint16_t *weights = nullptr;
    uint16_t *out = nullptr;
    uint16_t *next = nullptr;
    size_t batch = 0;
    size_t channels = 0;
    size_t steps = 0;
    size_t kernel = 0;
};

struct GemmReq {
    const uint16_t *a = nullptr;
    const void *rhs = nullptr;
    float *out = nullptr;
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    uint32_t rhs_layout = LFM_GEMM_RHS_KN;
    bool direct = false;
};

struct Dd {
    float hi = 0.0f;
    float lo = 0.0f;
};

struct ComplexDd {
    Dd re{};
    Dd im{};
};

struct FftConvDdReq {
    const float *input = nullptr;
    const float *kernel = nullptr;
    const float *skip = nullptr;
    float *out = nullptr;
    size_t batch = 0;
    size_t channels = 0;
    size_t steps = 0;
    size_t fft_size = 0;
};

struct IrfftDdReq {
    const float *real = nullptr;
    const float *imag = nullptr;
    float *out = nullptr;
    size_t rows = 0;
    size_t fft_size = 0;
    Dd scale{};
};

struct DepthPlan {
    uint64_t id = 0;
    std::vector<LfmDepthLayerV1> layers;
    std::vector<LfmDepthHeadV1> heads;
    const uint8_t *depth_linear_w = nullptr;
    const uint8_t *depth_linear_b = nullptr;
    const float *cos = nullptr;
    const float *sin = nullptr;
    size_t dim = 0;
    size_t heads_total = 0;
    size_t kv_heads = 0;
    size_t hd = 0;
    size_t ffn = 0;
    size_t codebooks = 0;
    size_t backbone_dim = 0;
    float eps = 0.0f;
};

// Depthformer values are pass scratch, not model-plan state. Keeping them in
// the plan made a second accepted ticket alias the first ticket's mutable
// planes even though the resident weights themselves were immutable views.
struct DepthScratch {
    std::vector<uint16_t> x, h, xn, qkv_b, y_b, attn_b, t_b;
    std::vector<uint16_t> k_plane, v_plane, logits_b, din_b, df_b;
    std::vector<float> qkv_f, up_f, q_f, attn_f, proj_f;
    std::array<float, MAX_WORKERS> partials{};
};

enum class SequenceMixerKind : uint8_t {
    ShortConv = 0,
    Attention = 1,
    MonarchLongConv = 2,
};

struct SequenceMixerDesc {
    SequenceMixerKind kind = SequenceMixerKind::ShortConv;
    uint32_t layer = 0;
    uint32_t kernel = 0;
    uint32_t halo = 0;
};

struct BackbonePlan {
    uint64_t id = 0;
    std::vector<LfmLayerDesc> layers;
    std::vector<SequenceMixerDesc> mixers;
    WeightBytes embed_w = nullptr;
    WeightBytes audio_embed_w = nullptr;
    WeightBytes emb_norm_w = nullptr;
    float emb_norm_eps = 0.0f;
    size_t vocab = 0;
    size_t audio_rows = 0;
    size_t h = 0;
    size_t ffn = 0;
    size_t max_ctx = 0;
    size_t qkv_max = 0;
    size_t y_max = 0;
    size_t kmax = 1;
};

// Conv-layer request payload: the whole shortconv+MLP layer in one doorbell; the
// hidden state between the two blocks lives in the engine's `mid` plane.
struct ConvReq {
    size_t layer = 0;
    const uint16_t *x = nullptr;
    const uint16_t *state_in = nullptr;
    uint16_t *state_out = nullptr;
    uint16_t *out = nullptr;
    size_t lanes = 0;
};

// Shortconv stage pointers for the workers (set by the coordinator per conv pass).
struct ScPass {
    Bf16Input x{};                     // block input [H]
    WeightBytes norm_w = nullptr;      // operator norm [H]
    WeightBytes in_w = nullptr;        // [3H, H]
    WeightBytes out_w = nullptr;       // [H, H]
    uint16_t *state_out = nullptr;     // carried window out [H·(K-1)]
    size_t h = 0, k = 0;
    // planes
    uint16_t *xn = nullptr;    // normed input [H]
    float *bcxf = nullptr;     // in_proj f32 [3H]
    uint16_t *bcxb = nullptr;  // in_proj bits [3H]
    uint16_t *conv = nullptr;  // conv out [H·K] = per channel [y | new_state]
    float *projf = nullptr;    // out_proj f32 [H]
    uint16_t *projb = nullptr; // y bits [H]
    uint16_t *stage = nullptr; // rounded out_proj staging [H]
    uint16_t *mid = nullptr;   // block output = MLP input [H]
    std::atomic<uint32_t> rs_bits{0};
};

// Attention-layer request: per-generation state (KV planes, rope tables, cursor)
// rides HERE — it lives in the per-cache objects, not the load-time table. The engine
// appends the step's K/V rows at `pos` and attends over pos+1 entries.
struct AttnReq {
    size_t layer = 0;
    const uint16_t *x = nullptr;
    uint16_t *k_plane = nullptr; // [n_kv, cap, hd] bf16 bits, head stride = cap·hd
    uint16_t *v_plane = nullptr;
    size_t head_stride = 0;
    size_t pos = 0; // cursor: rows 0..pos live; this step appends row `pos`
    const uint16_t *cos_base = nullptr; // [max_pos, hd/2] bf16
    const uint16_t *sin_base = nullptr;
    uint16_t *out = nullptr;
    size_t lanes = 0;
};

// Attention stage pointers for the workers.
struct AtPass {
    WeightBytes o_w = nullptr;     // [H, nh·hd]
    uint16_t *qkvb = nullptr;      // rounded q|k|v rows [(nh+2·nkv)·hd]
    float *qkvf = nullptr;
    uint16_t *ybits = nullptr;     // attention output per q head [nh·hd]
    float *att = nullptr;          // per-head score scratch [nh · max_ctx]
    Bf16Input x{};                 // residual input [H]
    uint16_t *mid = nullptr;       // block output = MLP input [H]
    const uint16_t *k_plane = nullptr;
    const uint16_t *v_plane = nullptr;
    size_t head_stride = 0, att_len = 0, max_ctx = 0;
    size_t h = 0, n_head = 0, n_kv = 0, hd = 0;
};

// Token-pass request: ONE doorbell per token — embed → every layer → final norm →
// logits. Sampling stays at the rim (RNG-stream parity).
struct TokenReq {
    const uint32_t *ids = nullptr; // text: 1 id; audio: n pre-offset audio-embed rows
    size_t n_ids = 0;
    uint32_t embed_kind = 0; // 0 = text table, 1 = audio table (sum), 2 = provided
    const uint16_t *provided_embed = nullptr; // kind 2: [H] bf16 hidden fed verbatim
    const LfmLayerState *states = nullptr;
    size_t n_states = 0;
    size_t pos = 0;
    const uint16_t *cos_base = nullptr;
    const uint16_t *sin_base = nullptr;
    uint16_t *out_hidden = nullptr; // [H] post embedding-norm bits
    float *out_logits = nullptr;    // [vocab] f32 (bf16-rounded then widened — the
                                    // linear_logits ladder)
    const LfmSamplerConfigV1 *sampler = nullptr;
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_token = nullptr;
    size_t lanes = 0;
};

// Conversation-owned, fixed-capacity activation arena for one small-M prefill
// group. It contains no weights and never grows after construction.
struct PrefillWorkspace {
    uint64_t model_id = 0;
    size_t h = 0, ffn = 0, max_ctx = 0;
    size_t qkv_max = 0, y_max = 0, kmax = 0;
    size_t lane_count = 0;
    std::vector<uint16_t> h0, h1, xn, gate, stage, mid;
    std::vector<uint16_t> bcxb, projb;
    std::vector<uint16_t> qkvb, att_y;
    std::vector<float> gu, bcxf, projf, qkvf, scores, logits;
};

struct PrefillReq {
    PrefillWorkspace *workspace = nullptr;
    const uint32_t *ids = nullptr;
    const uint16_t *provided_rows = nullptr;
    size_t rows = 0;
    uint32_t embed_kind = 0;
    const LfmLayerState *states = nullptr;
    size_t n_states = 0;
    size_t pos = 0;
    const uint16_t *cos_base = nullptr;
    const uint16_t *sin_base = nullptr;
    uint16_t *out_hidden = nullptr;
    const LfmSamplerConfigV1 *sampler = nullptr;
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_token = nullptr;
    size_t lanes = 0;
};

struct MimiReq {
    MimiDecodeState *state = nullptr;
    const uint32_t *codes = nullptr;
    float *pcm = nullptr;
    size_t capacity = 0;
    size_t *out_samples = nullptr;
    int completion_status = 0; // private deterministic route-fault seam
};

struct AudioReq {
    LfmAudioEncodePassV1 pass{};
    uint64_t start_gemm_generation = 0;
    std::atomic<uint64_t> gemm_generation{0};
    std::atomic<bool> done{false};
};

// Each admitted ticket owns one activation/sampling scratch bank until its
// exact CQ record is consumed. The lane team remains single-pass, so dispatch
// swaps precisely one bank onto the stage board; a queued follow-on never
// aliases the completing ticket's values. These are activation buffers only --
// resident weights remain borrowed byte views and are never copied here.
struct ScratchBank {
    std::vector<float> sc_partials, sc_gu;
    std::vector<uint16_t> sc_xn, sc_t;
    std::vector<float> sc_bcxf, sc_projf;
    std::vector<uint16_t> sc_bcxb, sc_conv, sc_projb, sc_stage, sc_mid;
    std::vector<float> at_qkvf, at_att;
    std::vector<uint16_t> at_qkvb, at_y;
    std::vector<uint16_t> tk_h0, tk_h1;
    std::vector<float> tk_logf;
    std::vector<float> sample_weights;
    std::vector<float> sample_heap;
    std::array<float, MAX_WORKERS> sample_lane_value{};
    std::array<float, MAX_WORKERS> sample_lane_sum{};
    std::array<uint32_t, MAX_WORKERS> sample_lane_index{};
    float sample_maximum = 0.0f;
    float sample_threshold = 0.0f;
    float sample_target = 0.0f;
    uint32_t sample_winner_lane = 0;
    std::vector<ComplexDd> fft_twiddles;
    std::vector<ComplexDd> irfft_twiddles;
    std::vector<ComplexDd> fft_work;
    size_t fft_twiddle_size = 0;
    size_t irfft_twiddle_size = 0;
    DepthScratch depth;
};

struct Engine;
struct AudioRoutePool;
struct alignas(ENGINE_CACHELINE) WaitWord {
    uint32_t value = 0;
    uint32_t reserved = 0;
    kc_port_wait_word *wait = nullptr;
    uint8_t padding[ENGINE_CACHELINE - 16] = {};
};
static_assert(sizeof(WaitWord) == ENGINE_CACHELINE,
              "shared doorbells must occupy a complete cache line");
static_assert(alignof(WaitWord) == ENGINE_CACHELINE,
              "shared doorbells must start on a cache-line boundary");

constexpr size_t BLOCK_DOMAIN_COUNT = 2;
constexpr size_t BLOCK_CQ_CAPACITY = 2;
constexpr uint32_t BLOCK_LANES = 4;
constexpr uint32_t GRID_LANES = BLOCK_DOMAIN_COUNT * BLOCK_LANES;

struct BlockCompletion {
    uint64_t generation = 0;
    int32_t status = 0;
    uint32_t block = 0;
};

/* One soft four-lane block. No field asserts physical cluster residency:
 * macOS may scatter the fixed members and correctness remains address-based.
 * The CQ is SPSC: the block leader publishes and lane zero drains. */
struct alignas(ENGINE_CACHELINE) BlockDomain {
    uint32_t id = 0;
    uint32_t lane_begin = 0;
    uint32_t lane_count = 0;
    uint32_t reserved = 0;
    WaitWord completion_word;
    std::atomic<uint64_t> completion_head{0};
    std::atomic<uint64_t> completion_tail{0};
    std::array<BlockCompletion, BLOCK_CQ_CAPACITY> completions{};
};
static_assert(alignof(BlockDomain) == ENGINE_CACHELINE);
static_assert(sizeof(BlockDomain) % ENGINE_CACHELINE == 0);

enum : uint32_t {
    PASS_SLOT_FREE = 0,
    PASS_SLOT_CLAIMING = 1,
    PASS_SLOT_RESERVED = 2,
    PASS_SLOT_SUBMITTING = 3,
    PASS_SLOT_SUBMITTED = 4,
    PASS_SLOT_RUNNING = 5,
    PASS_SLOT_COMPLETING = 6,
    PASS_SLOT_COMPLETE = 7,
    PASS_SLOT_RELEASING = 8,
};

constexpr uint64_t PASS_SLOT_STATE_BITS = 8;
constexpr uint64_t PASS_SLOT_STATE_MASK =
    (UINT64_C(1) << PASS_SLOT_STATE_BITS) - 1;

static constexpr uint64_t pass_slot_lease(uint64_t generation,
                                          uint32_t state) {
    return (generation << PASS_SLOT_STATE_BITS) | state;
}

static constexpr uint32_t pass_slot_state(uint64_t lease) {
    return static_cast<uint32_t>(lease & PASS_SLOT_STATE_MASK);
}

static constexpr uint64_t pass_slot_generation(uint64_t lease) {
    return lease >> PASS_SLOT_STATE_BITS;
}

struct PassSlot;
struct PassContinuationPermit;
using PassContinuation = void (*)(PassContinuationPermit *,
                                  const KcCompletionV1 &, void *);

struct PassSlot {
    Engine *engine = nullptr;
    uint32_t index = 0;
    /* Generation and state are one CAS authority. Keeping them in separate
     * atomics permits a stale owner to validate an old generation and then
     * transition a newly recycled RESERVED state. */
    std::atomic<uint64_t> lease{pass_slot_lease(0, PASS_SLOT_FREE)};
    /* Every successful FREE -> RESERVED transition gets a distinct owner
     * generation. State alone is not an ownership proof: a synchronous claim
     * can release its completed slot, pause before its destructor, and observe
     * the same physical slot RESERVED by a continuation. */
    std::atomic<uint64_t> reservation_sequence{0};
    /* True only for the bounded audio route. Its caller owns the admission
     * high bit until this exact slot is FREE, so releasing the slot must not
     * decrement the ordinary low-bit lease count. */
    bool exclusive_admission = false;
    WaitWord completion_word;
    WaitWord audio_word;
    KcSubmissionV1 submission{};
    KcCompletionV1 completion{};
    int request = REQ_NONE;
    uint64_t context_id = 0;
    PassContinuation continuation = nullptr;
    void *continuation_context = nullptr;

    MlpReq mlp;
    ConvReq conv;
    AttnReq attn;
    PrngReq prng;
    SampleReq sample;
    DepthReq depth_req;
    DepthwiseStreamReq depthwise_stream;
    GemmReq gemm;
    FftConvDdReq fft_conv_dd;
    IrfftDdReq irfft_dd;
    BackbonePlan *model = nullptr;
    DepthPlan *depth = nullptr;
    TokenReq tok;
    PrefillReq prefill;
    MimiReq mimi;
    AudioReq audio;
    ScratchBank scratch;
};
static_assert(alignof(PassSlot) >= ENGINE_CACHELINE,
              "pass-slot wait words require cache-line-aligned array elements");
static_assert(sizeof(PassSlot) % ENGINE_CACHELINE == 0,
              "pass-slot array stride must preserve wait-word isolation");

/* Stack-scoped authority for the exact slot whose CQ record triggered a
 * continuation. The type never crosses a header or the product ABI. Keeping
 * the slot RESERVED under the same generation makes resubmission atomic with
 * respect to competing compatibility producers. */
struct PassContinuationPermit {
    Engine *engine = nullptr;
    PassSlot *slot = nullptr;
    uint64_t generation = 0;
    bool consumed = false;
};

static uint32_t slot_state(const PassSlot *slot) {
    return pass_slot_state(slot->lease.load(std::memory_order_acquire));
}

static uint64_t slot_generation(const PassSlot *slot) {
    return pass_slot_generation(slot->lease.load(std::memory_order_acquire));
}

static bool transition_slot(PassSlot *slot, uint64_t generation,
                            uint32_t from, uint32_t to) {
    uint64_t expected = pass_slot_lease(generation, from);
    return slot->lease.compare_exchange_strong(
        expected, pass_slot_lease(generation, to), std::memory_order_acq_rel,
        std::memory_order_acquire);
}

struct LfmEngineSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t pass_submissions;
    uint64_t pass_completions;
    uint64_t bridge_dispatches;
    uint64_t dispatch_wakes;
    uint64_t fence_wake_calls;
    uint64_t fence_wakes;
    uint64_t fence_generations;
    uint64_t descriptor_acquires;
    uint64_t descriptor_retains;
    uint64_t descriptor_releases;
    uint64_t descriptor_callbacks;
    uint32_t descriptor_capacity;
    uint32_t descriptors_live;
    uint32_t max_descriptor_generation;
    uint32_t attention_qkv_capacity;
    uint32_t attention_y_capacity;
    uint32_t attention_score_capacity;
    uint32_t pass_claimed;
    uint32_t bridge_capacity;
    uint32_t pass_slot_capacity;
    uint32_t pass_slots_live;
    uint32_t max_pass_slots_live;
    uint64_t continuation_submissions;
    uint32_t route_capacity;
    uint32_t routes_live;
    uint32_t routes_ready;
    uint32_t reserved0;
    uint64_t route_dispatches;
    uint64_t route_parks;
};

struct Engine {
    Pass pass;
    Stage stage;
    kc_collective_t *collective = nullptr;

    // Kcoro owns the fixed team and every resident lane thread. The SQ/CQ
    // dispatcher only mounts full generations; numerical call stacks never
    // migrate or enter the general continuation executor.
    kc_team_t *team = nullptr;
    pthread_t bridge_worker{};
    pthread_t route_worker{};
    WaitWord dispatch_word;
    WaitWord route_word;
    std::array<BlockDomain, BLOCK_DOMAIN_COUNT> blocks;
    int n_workers = 0;
    int wait_words_prepared = 0;
    int bridge_started = 0;
    int route_started = 0;
    int route_wait_prepared = 0;
    int route_done_waits_prepared = 0;
    int slot_waits_prepared = 0;
    int audio_waits_prepared = 0;
    int block_waits_prepared = 0;
    uint32_t block_count = 1;
    uint32_t lanes_total = 1;
    std::atomic<uint64_t> lane_gen{0};
    std::atomic<uint64_t> gang_lease{0};
    int cur_req = REQ_NONE;
    std::atomic<bool> retire{false};
    std::atomic<bool> route_retire{false};
    LfmKernelBridge *bridge = nullptr;
    AudioRoutePool *route_pool = nullptr;
    std::array<PassSlot, PASS_CAPACITY> slots;
    PassSlot *active_slot = nullptr;
    KcSubmissionV1 active_submission{};
    // Low bits count numerical ticket leases. The high bit is the exclusive
    // plan/table mutation lease. This prevents a plan install, clear, or
    // all-slot resize from racing a queued continuation that already borrowed
    // its plan pointer and scratch bank.
    std::atomic<uint32_t> pass_admission{0};
    std::atomic<bool> pass_claimed{false};
    std::atomic<int> active_status{0};
    uint64_t runtime_epoch = 0;
    std::atomic<uint64_t> submit_sequence{0};
    std::atomic<uint32_t> ticket_generation{0};
    std::mutex submission_mutex;
    std::atomic<uint64_t> pass_submissions{0};
    std::atomic<uint64_t> pass_completions{0};
    std::atomic<uint64_t> bridge_dispatches{0};
    std::atomic<uint64_t> dispatch_wakes{0};
    std::atomic<uint32_t> attention_qkv_capacity{0};
    std::atomic<uint32_t> attention_y_capacity{0};
    std::atomic<uint32_t> attention_score_capacity{0};
    std::atomic<uint32_t> pass_slots_live{0};
    std::atomic<uint32_t> max_pass_slots_live{0};
    std::atomic<uint64_t> continuation_submissions{0};
    std::atomic<uint64_t> block_completions{0};
    std::atomic<uint64_t> gang_generations{0};
    std::atomic<uint64_t> route_dispatches{0};
    std::atomic<uint64_t> route_parks{0};
    std::atomic<uint64_t> audio_encode_passes{0};
    // Private deterministic stop-race handshake for the IRFFT conformance
    // request only. Ordinary model passes never load or mutate this field.
    std::atomic<uint32_t> test_lane_pause{0};
    /* Private deterministic ownership-race seams. 0 idle, 1 armed, 2 parked,
     * 3 released. They are never read by ordinary model passes. */
    std::atomic<uint32_t> test_claim_return_pause{0};
    std::atomic<uint32_t> test_continuation_pause{0};
    std::atomic<uint32_t> test_live_target{0};
    std::atomic<int> test_audio_route_depth_status{0};
    std::atomic<int> test_audio_route_mimi_status{0};

    ConvReq conv;  // conv-layer request payload
    AttnReq attn;  // attention-layer request payload
    PrngReq prng;  // caller-owned CSPRNG state and destination
    SampleReq sample; // pointer-only logits/state handoff; policy is inline
    DepthReq depth_req; // complete typed Depthformer frame request
    DepthwiseStreamReq depthwise_stream; // split-state Metal conv translation
    GemmReq gemm; // borrowed matrices and exclusive destination
    FftConvDdReq fft_conv_dd;
    IrfftDdReq irfft_dd;
    ScPass sc;     // shortconv stage pointers
    AtPass at;     // attention stage pointers

    // Immutable model plans coexist. One in-flight ticket selects `model`; the
    // physical lane team and scratch arena remain singular.
    std::vector<std::unique_ptr<BackbonePlan>> models;
    BackbonePlan *model = nullptr;
    uint64_t model_seq = 0;
    TokenReq tok; // token-pass request payload
    PrefillReq prefill;
    MimiReq mimi;

    // Persistent scratch backing. With a ctx built everything is sized ONCE there
    // (fixed-arena: no allocation during passes); the legacy per-call MLP entry still
    // grows on first use at a new shape.
    std::vector<float> sc_partials, sc_gu;
    std::vector<uint16_t> sc_xn, sc_t;
    // shortconv planes (ctx build): see ScPass.
    std::vector<float> sc_bcxf, sc_projf;
    std::vector<uint16_t> sc_bcxb, sc_conv, sc_projb, sc_stage, sc_mid;
    // attention planes (ctx build): qkv f32/bits [(nh+2·nkv)·hd], y bits [nh·hd],
    // per-head score scratch [nh · max_ctx] f32
    std::vector<float> at_qkvf, at_att;
    std::vector<uint16_t> at_qkvb, at_y;
    std::vector<uint16_t> tk_h0, tk_h1; // token-pass hidden ping-pong [H]
    std::vector<float> tk_logf;         // logits GEMV accumulators [vocab] (staging)
    std::vector<float> sample_weights;  // derived exp weights [largest installed vocab]
    std::vector<float> sample_heap;     // top-k values only; no logit payload copy
    std::array<float, MAX_WORKERS> sample_lane_value{};
    std::array<float, MAX_WORKERS> sample_lane_sum{};
    std::array<uint32_t, MAX_WORKERS> sample_lane_index{};
    float sample_maximum = 0.0f;
    float sample_threshold = 0.0f;
    float sample_target = 0.0f;
    uint32_t sample_winner_lane = 0;
    std::vector<std::unique_ptr<DepthPlan>> depth_plans;
    DepthPlan *active_depth = nullptr;
    uint64_t depth_seq = 0;
    std::vector<ComplexDd> fft_twiddles;
    std::vector<ComplexDd> irfft_twiddles;
    std::vector<ComplexDd> fft_work;
    size_t fft_twiddle_size = 0;
    size_t irfft_twiddle_size = 0;
    DepthScratch depth_scratch;
};

static inline void signal_all(WaitWord *word);

static BackbonePlan *find_model(Engine *e, uint64_t id) {
    for (const std::unique_ptr<BackbonePlan> &model : e->models)
        if (model->id == id) return model.get();
    return nullptr;
}

static void update_slot_high_water(Engine *e, uint32_t live) {
    uint32_t high = e->max_pass_slots_live.load(std::memory_order_relaxed);
    while (high < live &&
           !e->max_pass_slots_live.compare_exchange_weak(
               high, live, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

static void update_capacity_high_water(std::atomic<uint32_t> *counter,
                                       size_t capacity) {
    if (capacity > UINT32_MAX) capacity = UINT32_MAX;
    const uint32_t requested = (uint32_t)capacity;
    uint32_t high = counter->load(std::memory_order_relaxed);
    while (high < requested &&
           !counter->compare_exchange_weak(
               high, requested, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

static bool enter_pass_admission(Engine *e) {
    uint32_t state = e->pass_admission.load(std::memory_order_acquire);
    for (;;) {
        if ((state & PASS_ADMISSION_EXCLUSIVE) != 0 ||
            (state & PASS_ADMISSION_COUNT) == PASS_ADMISSION_COUNT) {
            return false;
        }
        if (e->pass_admission.compare_exchange_weak(
                state, state + 1, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
    }
}

static void leave_pass_admission(Engine *e) {
    const uint32_t previous =
        e->pass_admission.fetch_sub(1, std::memory_order_acq_rel);
    if ((previous & PASS_ADMISSION_EXCLUSIVE) != 0 ||
        (previous & PASS_ADMISSION_COUNT) == 0) {
        std::abort();
    }
}

static void clear_slot_request(PassSlot *slot) {
    slot->submission = {};
    slot->completion = {};
    slot->request = REQ_NONE;
    slot->context_id = 0;
    slot->continuation = nullptr;
    slot->continuation_context = nullptr;
    slot->mlp = {};
    slot->conv = {};
    slot->attn = {};
    slot->prng = {};
    slot->sample = {};
    slot->depth_req = {};
    slot->depthwise_stream = {};
    slot->gemm = {};
    slot->fft_conv_dd = {};
    slot->irfft_dd = {};
    slot->model = nullptr;
    slot->depth = nullptr;
    slot->tok = {};
    slot->prefill = {};
    slot->mimi = {};
    slot->audio.pass = {};
    slot->audio.start_gemm_generation =
        slot->audio.gemm_generation.load(std::memory_order_relaxed);
    slot->audio.done.store(true, std::memory_order_relaxed);
}

static PassSlot *reserve_pass_slot(Engine *e,
                                   bool allow_exclusive = false) {
    if (!e) return nullptr;
    if (allow_exclusive) {
        if (e->pass_admission.load(std::memory_order_acquire) !=
            PASS_ADMISSION_EXCLUSIVE) {
            return nullptr;
        }
    } else {
        /* This early check is repeated after the physical reservation. The
         * admission CAS is authoritative; the two checks document and close
         * the route-acquisition boundary for future reservation changes. */
        if ((e->pass_admission.load(std::memory_order_acquire) &
             PASS_ADMISSION_EXCLUSIVE) != 0 ||
            !enter_pass_admission(e)) {
            return nullptr;
        }
    }
    for (PassSlot &slot : e->slots) {
        uint64_t expected = pass_slot_lease(0, PASS_SLOT_FREE);
        if (!slot.lease.compare_exchange_strong(
                expected, pass_slot_lease(0, PASS_SLOT_CLAIMING),
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        const uint32_t admission =
            e->pass_admission.load(std::memory_order_acquire);
        const bool admitted = allow_exclusive
            ? admission == PASS_ADMISSION_EXCLUSIVE
            : (admission & PASS_ADMISSION_EXCLUSIVE) == 0;
        if (!admitted) {
            slot.lease.store(pass_slot_lease(0, PASS_SLOT_FREE),
                             std::memory_order_release);
            if (!allow_exclusive) leave_pass_admission(e);
            return nullptr;
        }
        constexpr uint64_t max_generation =
            UINT64_MAX >> PASS_SLOT_STATE_BITS;
        uint64_t generation =
            (slot.reservation_sequence.fetch_add(1,
                                                 std::memory_order_acq_rel) +
             1) & max_generation;
        while (generation == 0) {
            generation =
                slot.reservation_sequence.fetch_add(1,
                                                    std::memory_order_acq_rel) + 1;
            generation &= max_generation;
        }
        clear_slot_request(&slot);
        slot.exclusive_admission = allow_exclusive;
        /* CLAIMING is deliberately non-releasable. A stale owner cannot see a
         * recycled RESERVED state until the new generation is published. */
        slot.lease.store(pass_slot_lease(generation, PASS_SLOT_RESERVED),
                         std::memory_order_release);
        const uint32_t live =
            e->pass_slots_live.fetch_add(1, std::memory_order_acq_rel) + 1;
        update_slot_high_water(e, live);
        if (e->test_live_target.load(std::memory_order_relaxed) != 0)
            signal_all(&e->dispatch_word);
        return &slot;
    }
    if (!allow_exclusive) leave_pass_admission(e);
    return nullptr;
}

static bool release_pass_slot(PassSlot *slot, uint64_t generation) {
    if (!slot || generation == 0) return false;
    Engine *e = slot->engine;
    if (!transition_slot(slot, generation, PASS_SLOT_RESERVED,
                         PASS_SLOT_RELEASING) &&
        !transition_slot(slot, generation, PASS_SLOT_COMPLETE,
                         PASS_SLOT_RELEASING)) {
        return false;
    }
    const bool exclusive = slot->exclusive_admission;
    clear_slot_request(slot);
    e->pass_slots_live.fetch_sub(1, std::memory_order_acq_rel);
    if (!exclusive) leave_pass_admission(e);
    slot->exclusive_admission = false;
    /* FREE is the final publication edge. Publishing it before the accounting
     * decrements lets a recycler increment live 2 -> 3 on a two-slot engine. */
    slot->lease.store(pass_slot_lease(0, PASS_SLOT_FREE),
                      std::memory_order_release);
    if (e->route_word.wait) signal_all(&e->route_word);
    if (e->test_live_target.load(std::memory_order_relaxed) != 0)
        signal_all(&e->dispatch_word);
    return true;
}

static void swap_depth_scratch(DepthScratch &left, DepthScratch &right) {
    left.x.swap(right.x);
    left.h.swap(right.h);
    left.xn.swap(right.xn);
    left.qkv_b.swap(right.qkv_b);
    left.y_b.swap(right.y_b);
    left.attn_b.swap(right.attn_b);
    left.t_b.swap(right.t_b);
    left.k_plane.swap(right.k_plane);
    left.v_plane.swap(right.v_plane);
    left.logits_b.swap(right.logits_b);
    left.din_b.swap(right.din_b);
    left.df_b.swap(right.df_b);
    left.qkv_f.swap(right.qkv_f);
    left.up_f.swap(right.up_f);
    left.q_f.swap(right.q_f);
    left.attn_f.swap(right.attn_f);
    left.proj_f.swap(right.proj_f);
    left.partials.swap(right.partials);
}

static void swap_scratch(Engine *e, ScratchBank &scratch) {
    e->sc_partials.swap(scratch.sc_partials);
    e->sc_gu.swap(scratch.sc_gu);
    e->sc_xn.swap(scratch.sc_xn);
    e->sc_t.swap(scratch.sc_t);
    e->sc_bcxf.swap(scratch.sc_bcxf);
    e->sc_projf.swap(scratch.sc_projf);
    e->sc_bcxb.swap(scratch.sc_bcxb);
    e->sc_conv.swap(scratch.sc_conv);
    e->sc_projb.swap(scratch.sc_projb);
    e->sc_stage.swap(scratch.sc_stage);
    e->sc_mid.swap(scratch.sc_mid);
    e->at_qkvf.swap(scratch.at_qkvf);
    e->at_att.swap(scratch.at_att);
    e->at_qkvb.swap(scratch.at_qkvb);
    e->at_y.swap(scratch.at_y);
    e->tk_h0.swap(scratch.tk_h0);
    e->tk_h1.swap(scratch.tk_h1);
    e->tk_logf.swap(scratch.tk_logf);
    e->sample_weights.swap(scratch.sample_weights);
    e->sample_heap.swap(scratch.sample_heap);
    e->sample_lane_value.swap(scratch.sample_lane_value);
    e->sample_lane_sum.swap(scratch.sample_lane_sum);
    e->sample_lane_index.swap(scratch.sample_lane_index);
    std::swap(e->sample_maximum, scratch.sample_maximum);
    std::swap(e->sample_threshold, scratch.sample_threshold);
    std::swap(e->sample_target, scratch.sample_target);
    std::swap(e->sample_winner_lane, scratch.sample_winner_lane);
    e->fft_twiddles.swap(scratch.fft_twiddles);
    e->irfft_twiddles.swap(scratch.irfft_twiddles);
    e->fft_work.swap(scratch.fft_work);
    std::swap(e->fft_twiddle_size, scratch.fft_twiddle_size);
    std::swap(e->irfft_twiddle_size, scratch.irfft_twiddle_size);
    swap_depth_scratch(e->depth_scratch, scratch.depth);
}

static void activate_slot(Engine *e, PassSlot *slot) {
    swap_scratch(e, slot->scratch);
    e->active_slot = slot;
    e->model = slot->model;
    e->active_depth = slot->depth;
    e->conv = slot->conv;
    e->attn = slot->attn;
    e->prng = slot->prng;
    e->sample = slot->sample;
    e->depth_req = slot->depth_req;
    e->depthwise_stream = slot->depthwise_stream;
    e->gemm = slot->gemm;
    e->fft_conv_dd = slot->fft_conv_dd;
    e->irfft_dd = slot->irfft_dd;
    e->tok = slot->tok;
    e->prefill = slot->prefill;
    e->mimi = slot->mimi;
    if (slot->request == REQ_MLP) {
        Pass *pass = &e->pass;
        pass->x = slot->mlp.x;
        pass->norm_w = slot->mlp.norm_w;
        pass->w1 = slot->mlp.w1;
        pass->w3 = slot->mlp.w3;
        pass->w2 = slot->mlp.w2;
        pass->out = slot->mlp.out;
        pass->h = slot->mlp.h;
        pass->i = slot->mlp.i;
        pass->tiles = slot->mlp.tiles;
        pass->eps = slot->mlp.eps;
        pass->partials = e->sc_partials.data();
        pass->xn = e->sc_xn.data();
        pass->gu = e->sc_gu.data();
        pass->t = e->sc_t.data();
        pass->rs_bits.store(0, std::memory_order_relaxed);
    }
}

static void deactivate_slot(Engine *e, PassSlot *slot) {
    e->active_slot = nullptr;
    e->model = nullptr;
    e->active_depth = nullptr;
    swap_scratch(e, slot->scratch);
}

// Compatibility calls remain mutually exclusive at the raw C ABI, but their
// request payload and scratch live in a capacity-2 PassSlot. Native
// continuations bypass this compatibility claim and enter pass_admission
// directly, so one synchronous caller may queue beside one native ticket.
class PassClaim {
  public:
    explicit PassClaim(Engine *engine) : engine_(engine) {
        bool expected = false;
        held_ = engine_ && engine_->pass_claimed.compare_exchange_strong(
                               expected, true, std::memory_order_acq_rel,
                               std::memory_order_acquire);
        if (held_) {
            slot_ = reserve_pass_slot(engine_);
            if (!slot_) {
                engine_->pass_claimed.store(false, std::memory_order_release);
                held_ = false;
            } else {
                generation_ = slot_generation(slot_);
            }
        }
    }

    ~PassClaim() {
        if (!held_) return;
        if (slot_) (void)release_pass_slot(slot_, generation_);
        engine_->pass_claimed.store(false, std::memory_order_release);
    }

    explicit operator bool() const { return held_; }
    PassSlot *slot() const { return slot_; }
    PassClaim(const PassClaim &) = delete;
    PassClaim &operator=(const PassClaim &) = delete;

  private:
    Engine *engine_ = nullptr;
    PassSlot *slot_ = nullptr;
    uint64_t generation_ = 0;
    bool held_ = false;
};

// Plan installation/removal and all-slot sizing need stronger exclusion than a
// compatibility PassClaim: an asynchronous continuation may exist without
// pass_claimed. The admission high bit excludes both queued and running slots
// and prevents a new slot from appearing until mutation is complete.
class PlanClaim {
  public:
    explicit PlanClaim(Engine *engine) : engine_(engine) {
        bool expected = false;
        if (!engine_ || !engine_->pass_claimed.compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
            return;
        }
        uint32_t idle = 0;
        held_ = engine_->pass_admission.compare_exchange_strong(
            idle, PASS_ADMISSION_EXCLUSIVE, std::memory_order_acq_rel,
            std::memory_order_acquire);
        if (!held_)
            engine_->pass_claimed.store(false, std::memory_order_release);
    }

    ~PlanClaim() {
        if (!held_) return;
        const uint32_t previous = engine_->pass_admission.exchange(
            0, std::memory_order_release);
        if (previous != PASS_ADMISSION_EXCLUSIVE) std::abort();
        engine_->pass_claimed.store(false, std::memory_order_release);
    }

    explicit operator bool() const { return held_; }
    PlanClaim(const PlanClaim &) = delete;
    PlanClaim &operator=(const PlanClaim &) = delete;

  private:
    Engine *engine_ = nullptr;
    bool held_ = false;
};

// ---- tile bodies (identical math to decode.rs) ----------------------------------------
static void run_tile(uint32_t kind, uint32_t idx, const Stage *st, Engine *e) {
    Pass *p = &e->pass;
    switch (kind) {
    case ST_SUMSQ: {
        p->partials[idx] =
            lfm_bf16_sumsq_stride_f32(p->x, p->h, idx, p->tiles);
        break;
    }
    case ST_NORM: {
        size_t chunk = (p->h + p->tiles - 1) / p->tiles;
        size_t begin = (size_t)idx * chunk;
        size_t end = std::min(begin + chunk, p->h);
        if (end <= begin) break;
        uint32_t rsb = p->rs_bits.load(std::memory_order_acquire);
        float rs;
        std::memcpy(&rs, &rsb, 4);
        lfm_bf16_rmsnorm(p->x + begin, weight_offset(p->norm_w, begin), p->xn + begin,
                         (int)(end - begin), rs);
        break;
    }
    case ST_GATEUP: {
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < p->i ? r0 + st->chunk : p->i;
        if (r1 <= r0) break;
        size_t n = r1 - r0;
        lfm_bf16_gemm_nt_f32(p->xn, weight_offset(p->w1, r0 * p->h), p->gu + r0,
                             1, (int)n, (int)p->h);
        lfm_bf16_gemm_nt_f32(p->xn, weight_offset(p->w3, r0 * p->h), p->gu + p->i + r0,
                             1, (int)n,
                             (int)p->h);
        lfm_swiglu_bf16(p->gu + r0, p->gu + p->i + r0, p->t + r0, (int)n);
        break;
    }
    case ST_DOWN: {
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < p->h ? r0 + st->chunk : p->h;
        if (r1 <= r0) break;
        size_t n = r1 - r0;
        float y[DOWN_BAND_CAP]; // per-worker accumulator; chunk capped at publish
        uint16_t rounded[DOWN_BAND_CAP];
        lfm_bf16_gemm_nt_f32(p->t, weight_offset(p->w2, r0 * p->i), y,
                             1, (int)n, (int)p->i);
        lfm_f32_to_bf16(y, rounded, (int)n);
        lfm_bf16_add(rounded, p->x + r0, p->out + r0, (int)n);
        break;
    }
    case ST_SC_NORM: {
        // Contiguous band — elementwise, so banding never changes a cell's value.
        ScPass *c = &e->sc;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < c->h ? r0 + st->chunk : c->h;
        if (r1 <= r0) break;
        uint32_t rsb = c->rs_bits.load(std::memory_order_acquire);
        float inv_rms;
        std::memcpy(&inv_rms, &rsb, 4);
        lfm_bf16_rmsnorm(c->x.offset(r0).data(), weight_offset(c->norm_w, r0),
                         c->xn + r0, (int)(r1 - r0), inv_rms);
        break;
    }
    case ST_SC_INPROJ: {
        ScPass *c = &e->sc;
        size_t rows = 3 * c->h;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < rows ? r0 + st->chunk : rows;
        if (r1 <= r0) break;
        lfm_bf16_gemm_nt_f32(c->xn, weight_offset(c->in_w, r0 * c->h), c->bcxf + r0, 1,
                             (int)(r1 - r0), (int)c->h);
        lfm_f32_to_bf16(c->bcxf + r0, c->bcxb + r0, (int)(r1 - r0));
        break;
    }
    case ST_SC_GATHER: {
        // conv plane layout is [H][K]: y at [ch][0], advanced window at [ch][1..K].
        ScPass *c = &e->sc;
        size_t c0 = (size_t)idx * st->chunk;
        size_t c1 = c0 + st->chunk < c->h ? c0 + st->chunk : c->h;
        for (size_t ch = c0; ch < c1; ++ch) {
            c->projb[ch] = c->conv[ch * c->k];
            for (size_t j = 0; j + 1 < c->k; ++j) {
                c->state_out[ch * (c->k - 1) + j] = c->conv[ch * c->k + 1 + j];
            }
        }
        break;
    }
    case ST_SC_OUTPROJ: {
        // rb(out_proj) then rb(+residual) — the linear_forward ladder, band-wise.
        ScPass *c = &e->sc;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < c->h ? r0 + st->chunk : c->h;
        if (r1 <= r0) break;
        lfm_bf16_gemm_nt_f32(c->projb, weight_offset(c->out_w, r0 * c->h), c->projf + r0, 1,
                             (int)(r1 - r0), (int)c->h);
        lfm_f32_to_bf16(c->projf + r0, c->stage + r0, (int)(r1 - r0));
        lfm_bf16_add(c->stage + r0, c->x.offset(r0).data(), c->mid + r0,
                     (int)(r1 - r0));
        break;
    }
    case ST_AT_QKV: {
        // One band over the concatenated q|k|v projection row space; segments route to
        // their own weight matrices. Each row is the same linear_forward ladder the
        // candle path runs: nt dot (f32) then one bf16 storage round.
        AtPass *a = &e->at;
        const LfmLayerDesc *d = &e->model->layers[e->attn.layer];
        size_t qrows = a->n_head * a->hd, kvrows = a->n_kv * a->hd;
        size_t total = qrows + 2 * kvrows;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < total ? r0 + st->chunk : total;
        size_t r = r0;
        while (r < r1) {
            WeightBytes w;
            size_t seg0, seglen;
            if (r < qrows) {
                w = d->q_w;
                seg0 = 0;
                seglen = qrows;
            } else if (r < qrows + kvrows) {
                w = d->k_w;
                seg0 = qrows;
                seglen = kvrows;
            } else {
                w = d->v_w;
                seg0 = qrows + kvrows;
                seglen = kvrows;
            }
            size_t seg_end = seg0 + seglen;
            size_t stop = r1 < seg_end ? r1 : seg_end;
            lfm_bf16_gemm_nt_f32(e->sc_xn.data(), weight_offset(w, (r - seg0) * a->h),
                                 a->qkvf + r, 1,
                                 (int)(stop - r), (int)a->h);
            lfm_f32_to_bf16(a->qkvf + r, a->qkvb + r, (int)(stop - r));
            r = stop;
        }
        break;
    }
    case ST_AT_HEAD: {
        // One q head, exactly attn_decode_bf16's per-head body: widen q, dots over the
        // K plane (grouped kv head), scaled softmax, weighted V sum, one round out.
        AtPass *a = &e->at;
        size_t qh = idx;
        if (qh >= a->n_head) break;
        size_t group = a->n_head / a->n_kv;
        size_t kh = qh / group;
        float scale = lfm_rsqrt_size(a->hd);
        float qf[512]; // hd cap (hd = 64 on this family; 512 is generous)
        lfm_bf16_to_f32(a->qkvb + qh * a->hd, qf, (int)a->hd);
        float *att = a->att + qh * a->max_ctx;
        lfm_attn_qk_bf16(qf, a->k_plane + kh * a->head_stride, att, (int)a->att_len,
                         (int)a->hd);
        lfm_softmax_scaled_f32(att, (int)a->att_len, scale);
        float of[512];
        lfm_attn_av_bf16(att, a->v_plane + kh * a->head_stride, of, (int)a->att_len,
                         (int)a->hd);
        lfm_f32_to_bf16(of, a->ybits + qh * a->hd, (int)a->hd);
        break;
    }
    case ST_AT_OPROJ: {
        // o_proj rows band over the attention output, then rb(+residual) — the same
        // ladder the candle path's linear_forward + residual add runs.
        AtPass *a = &e->at;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < a->h ? r0 + st->chunk : a->h;
        if (r1 <= r0) break;
        size_t kdim = a->n_head * a->hd;
        float yf[DOWN_BAND_CAP];
        (void)yf;
        ScPass *c = &e->sc; // reuse projf/stage planes (conv pass is never concurrent)
        lfm_bf16_gemm_nt_f32(a->ybits, weight_offset(a->o_w, r0 * kdim), c->projf + r0, 1,
                             (int)(r1 - r0), (int)kdim);
        lfm_f32_to_bf16(c->projf + r0, c->stage + r0, (int)(r1 - r0));
        lfm_bf16_add(c->stage + r0, a->x.offset(r0).data(), a->mid + r0,
                     (int)(r1 - r0));
        break;
    }
    case ST_LOGITS: {
        // linear_logits ladder EXACTLY: M==1 GEMV rows, f32 accumulate, RAW f32
        // out. The pinned Rust head (linear_logits -> Bf16GemmNt) emits the
        // kernel's f32 directly — the bf16 storage round this stage used to add
        // was an EXTRA round the reference never performs, and it is what
        // flipped the perf-chain hash when the head was first absorbed. Same
        // kernel, same per-row K-reduction (row banding cannot reorder a row's
        // accumulation), no round: bit-identical to the candle-head path — the
        // PERF oracle is the proof.
        Engine *ee = e;
        const TokenReq *t = &ee->tok;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < ee->model->vocab
                        ? r0 + st->chunk
                        : ee->model->vocab;
        if (r1 <= r0) break;
        float *acc = ee->tk_logf.data() + r0;
        lfm_bf16_gemm_nt_f32(t->out_hidden,
                             weight_offset(ee->model->embed_w, r0 * ee->model->h), acc, 1,
                             (int)(r1 - r0), (int)ee->model->h);
        if (t->out_logits)
            std::memcpy(t->out_logits + r0, ee->tk_logf.data() + r0,
                        (r1 - r0) * sizeof(float));
        break;
    }
    default:
        break;
    }
}

// ---- fixed-lane doorbells and fence ----------------------------------------------------
static inline void signal_all(WaitWord *word) {
    kc_atomic_u32_fetch_add_release(&word->value, 1);
    kc_port_wake_u32_all(word->wait);
}

static void pause_test_boundary(Engine *e, std::atomic<uint32_t> *state) {
    uint32_t armed = 1;
    if (!state->compare_exchange_strong(armed, 2, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        return;
    }
    signal_all(&e->dispatch_word);
    uint32_t observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
    while (state->load(std::memory_order_acquire) == 2 &&
           !e->retire.load(std::memory_order_acquire)) {
        (void)kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
    }
    state->store(0, std::memory_order_release);
    signal_all(&e->dispatch_word);
}

// One stage boundary. `serial` runs exactly once, on the last arriver, AFTER every
// lane's pre-fence work is complete and BEFORE any lane crosses — the collective
// serial section. Bit-determinism does not care which lane executes it: all operands
// live in engine-owned planes and every ladder has a fixed internal order.
template <typename F>
static inline void lane_fence(Engine *e, uint32_t lane, F &&serial) {
    using Callback = std::remove_reference_t<F>;
    const int status = kc_collective_arrive(
        e->collective, lane,
        [](void *context) { (*static_cast<Callback *>(context))(); },
        std::addressof(serial));
    if (status < 0) std::abort();
}

// One stage: fence in (serial section + claim-counter reset), then claim tiles off
// the bare counter until dry. Every lane calls this with IDENTICAL (kind, count,
// chunk) — all derived from shared engine state — so no lane needs a publish to
// learn the schedule. Claim order is bit-irrelevant: tiles write disjoint cells and
// every cross-tile fold happens serially in a later fence, in fixed tile order.
template <typename F>
static void run_stage(Engine *e, uint32_t lane, uint32_t kind, uint32_t count,
                      uint32_t chunk, F &&pre) {
    lane_fence(e, lane, [&] {
        pre();
        e->stage.kind = kind;
        e->stage.count = count;
        e->stage.chunk = chunk;
        e->stage.next.store(0, std::memory_order_relaxed);
    });
    for (;;) {
        uint32_t idx = e->stage.next.fetch_add(1, std::memory_order_relaxed);
        if (idx >= count) break;
        run_tile(kind, idx, &e->stage, e);
    }
}

static bool sample_config_valid(const LfmSamplerConfigV1 *config) {
    if (!config || config->size != sizeof(*config) ||
        config->abi_version != LFM_SAMPLE_ABI_VERSION || config->reserved != 0 ||
        (config->flags & ~LFM_SAMPLE_FLAG_GREEDY) != 0) {
        return false;
    }
    return (config->flags & LFM_SAMPLE_FLAG_GREEDY) != 0 ||
           (std::isfinite(config->temperature) && config->temperature > 0.0);
}

static inline float sample_raw(const SampleReq &sample, size_t index) {
    float value = sample.dtype == SAMPLE_F32
                      ? static_cast<const float *>(sample.logits)[index]
                      : bf16_f32(static_cast<const uint16_t *>(sample.logits)[index]);
    return std::isnan(value) ? -std::numeric_limits<float>::infinity() : value;
}

static inline float sample_scaled(const SampleReq &sample, size_t index,
                                  float scale, uint16_t bf16_scale) {
    if (sample.dtype == SAMPLE_F32) return sample_raw(sample, index) * scale;
    float value = sample_raw(sample, index) * bf16_f32(bf16_scale);
    return bf16_f32(rb_bits(value));
}

static float sample_topk_threshold(const SampleReq &sample, float scale,
                                   uint16_t bf16_scale, float *heap) {
    size_t k = sample.config.top_k;
    if (k == 0 || k >= sample.count) return -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < k; ++i) heap[i] = sample_scaled(sample, i, scale, bf16_scale);
    std::make_heap(heap, heap + k, std::greater<float>());
    for (size_t i = k; i < sample.count; ++i) {
        float value = sample_scaled(sample, i, scale, bf16_scale);
        if (value <= heap[0]) continue;
        std::pop_heap(heap, heap + k, std::greater<float>());
        heap[k - 1] = value;
        std::push_heap(heap, heap + k, std::greater<float>());
    }
    return heap[0];
}

// One collective categorical draw. It is deliberately shaped like one GPU
// threadgroup: lane-private vocabulary bands, three generation barriers, and
// one serial RNG mutation. The input payload never moves; `sample_weights` is
// derived scratch, not a copied logit/tensor plane.
static void run_sampler(Engine *e, uint32_t lane, const SampleReq &sample) {
    const uint32_t lanes = e->lanes_total;
    const size_t chunk = (sample.count + lanes - 1) / lanes;
    const size_t begin = std::min((size_t)lane * chunk, sample.count);
    const size_t end = std::min(begin + chunk, sample.count);
    const bool greedy = (sample.config.flags & LFM_SAMPLE_FLAG_GREEDY) != 0 ||
                        sample.config.top_k == 1;

    if (greedy) {
        uint32_t local = 0;
        if (end > begin) {
            local = sample.dtype == SAMPLE_F32
                        ? lfm_sampler_argmax_f32(
                              static_cast<const float *>(sample.logits) + begin, end - begin)
                        : lfm_sampler_argmax_bf16(
                              static_cast<const uint16_t *>(sample.logits) + begin, end - begin);
        }
        e->sample_lane_index[lane] = (uint32_t)(begin + local);
        e->sample_lane_value[lane] = end > begin
                                         ? sample_raw(sample, begin + local)
                                         : -std::numeric_limits<float>::infinity();
        lane_fence(e, lane, [&] {
            float best = -std::numeric_limits<float>::infinity();
            uint32_t index = 0;
            for (uint32_t l = 0; l < lanes; ++l) {
                float value = e->sample_lane_value[l];
                uint32_t candidate = e->sample_lane_index[l];
                if (value > best || (value == best && candidate < index)) {
                    best = value;
                    index = candidate;
                }
            }
            *sample.out = index;
        });
        return;
    }

    const float scale = (float)(1.0 / sample.config.temperature);
    const uint16_t bf16_scale = rb_bits(scale);
    float local_maximum = -std::numeric_limits<float>::infinity();
    uint32_t local_index = (uint32_t)begin;
    for (size_t i = begin; i < end; ++i) {
        float value = sample_scaled(sample, i, scale, bf16_scale);
        if (value > local_maximum) {
            local_maximum = value;
            local_index = (uint32_t)i;
        }
    }
    e->sample_lane_value[lane] = local_maximum;
    e->sample_lane_index[lane] = local_index;

    lane_fence(e, lane, [&] {
        e->sample_maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t l = 0; l < lanes; ++l) {
            if (e->sample_lane_value[l] > e->sample_maximum)
                e->sample_maximum = e->sample_lane_value[l];
        }
        e->sample_threshold = sample_topk_threshold(
            sample, scale, bf16_scale, e->sample_heap.data());
    });

    float *weights = e->sample_weights.data() + begin;
    e->sample_lane_sum[lane] = sample.dtype == SAMPLE_F32
                                  ? lfm_sampler_exp_sum_f32(
                                        static_cast<const float *>(sample.logits) + begin,
                                        weights, end - begin, scale, e->sample_maximum,
                                        e->sample_threshold)
                                  : lfm_sampler_exp_sum_bf16(
                                        static_cast<const uint16_t *>(sample.logits) + begin,
                                        weights, end - begin, bf16_scale, e->sample_maximum,
                                        e->sample_threshold);

    lane_fence(e, lane, [&] {
        float total = 0.0f;
        for (uint32_t l = 0; l < lanes; ++l) total += e->sample_lane_sum[l];
        if (!(total > 0.0f) || !std::isfinite(total)) {
            float best = -std::numeric_limits<float>::infinity();
            uint32_t index = 0;
            for (uint32_t l = 0; l < lanes; ++l) {
                if (e->sample_lane_value[l] > best) {
                    best = e->sample_lane_value[l];
                    index = e->sample_lane_index[l];
                }
            }
            *sample.out = index;
            e->sample_winner_lane = UINT32_MAX;
            return;
        }

        uint64_t draw = 0;
        if (lfm_prng_fill_u64(sample.state, &draw, 1) != 0) {
            *sample.out = e->sample_lane_index[0];
            e->sample_winner_lane = UINT32_MAX;
            return;
        }
        double unit = (double)(draw >> 11) * 0x1.0p-53;
        float target = (float)(unit * (double)total);
        if (target >= total) target = std::nextafter(total, 0.0f);
        float prefix = 0.0f;
        e->sample_winner_lane = lanes - 1;
        e->sample_target = target;
        for (uint32_t l = 0; l < lanes; ++l) {
            float next = prefix + e->sample_lane_sum[l];
            if (target < next) {
                e->sample_winner_lane = l;
                e->sample_target = target - prefix;
                break;
            }
            prefix = next;
        }
    });

    if (e->sample_winner_lane == lane) {
        e->sample_lane_index[lane] = (uint32_t)begin +
            lfm_sampler_prefix_pick(weights, end - begin, e->sample_target);
    }
    lane_fence(e, lane, [&] {
        if (e->sample_winner_lane != UINT32_MAX)
            *sample.out = e->sample_lane_index[e->sample_winner_lane];
    });
}

static inline const uint8_t *depth_bytes(const LfmDepthBufferV1 &view) {
    return reinterpret_cast<const uint8_t *>(view.address);
}

static inline const float *depth_f32(const LfmDepthBufferV1 &view) {
    return reinterpret_cast<const float *>(view.address);
}

static inline void depth_band(size_t count, uint32_t lane, uint32_t lanes,
                              size_t *begin, size_t *end) {
    const size_t chunk = (count + lanes - 1) / lanes;
    *begin = std::min((size_t)lane * chunk, count);
    *end = std::min(*begin + chunk, count);
}

static inline void depth_gemv(const LfmDepthBufferV1 &weight, const uint16_t *x,
                              float *out, size_t rows, size_t cols,
                              uint32_t lane, uint32_t lanes) {
    size_t begin = 0, end = 0;
    depth_band(rows, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_gemm_nt_f32(
            x, depth_bytes(weight) + begin * cols * sizeof(uint16_t),
            out + begin, 1, (int)(end - begin), (int)cols);
}

static void depth_norm(Engine *e, uint32_t lane, const uint16_t *x,
                       const LfmDepthBufferV1 &weight, uint16_t *out) {
    DepthPlan &d = *e->active_depth;
    DepthScratch &scratch = e->depth_scratch;
    size_t begin = 0, end = 0;
    depth_band(d.dim, lane, e->lanes_total, &begin, &end);
    scratch.partials[lane] =
        end > begin ? lfm_bf16_sumsq_f32(x + begin, (int)(end - begin))
                    : 0.0f;
    lane_fence(e, lane, [] {});
    float total = lfm_sum_f32(scratch.partials.data(), e->lanes_total);
    const float inv_rms = lfm_inv_rms_f32(total, d.dim, d.eps);
    if (end > begin)
        lfm_bf16_rmsnorm(
            x + begin, depth_bytes(weight) + begin * sizeof(uint16_t),
            out + begin, (int)(end - begin), inv_rms);
    lane_fence(e, lane, [] {});
}

static void depth_qk_head(const DepthPlan &d, const uint16_t *src,
                          const LfmDepthBufferV1 &weight, uint16_t *out,
                          size_t position) {
    uint16_t normed[128];
    float rotated[128];
    const float sum = lfm_bf16_sumsq_f32(src, (int)d.hd);
    const float inv_rms = lfm_inv_rms_f32(sum, d.hd, d.eps);
    lfm_bf16_rmsnorm(src, depth_bytes(weight), normed, (int)d.hd, inv_rms);
    lfm_bf16_to_f32(normed, rotated, (int)d.hd);
    const size_t half = d.hd / 2;
    lfm_rope_i_f32(rotated, d.cos + position * half, d.sin + position * half,
                   (int)d.hd);
    lfm_f32_to_bf16(rotated, out, (int)d.hd);
}

// Complete typed Depthformer frame. The former generic lane program is translated
// stage-for-stage: the same resident pointers, bf16 rounding points, lane bands,
// and recurrence, now using the zero-spin native fence.
static void run_depth_frame(Engine *e, uint32_t lane) {
    DepthPlan &d = *e->active_depth;
    DepthScratch &scratch = e->depth_scratch;
    const DepthReq &request = e->depth_req;
    const uint32_t lanes = e->lanes_total;
    const size_t qkv_rows = d.dim + 2 * d.kv_heads * d.hd;
    const size_t group = d.heads_total / d.kv_heads;
    const float attn_scale = lfm_rsqrt_size(d.hd);

    // depth_linear(hidden) + bias -> one row per codebook.
    depth_gemv({reinterpret_cast<uintptr_t>(d.depth_linear_w),
                d.codebooks * d.dim * d.backbone_dim},
               request.hidden, scratch.proj_f.data(), d.codebooks * d.dim,
               d.backbone_dim, lane, lanes);
    size_t begin = 0, end = 0;
    depth_band(d.codebooks * d.dim, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_bias_add_f32(scratch.proj_f.data() + begin,
                              d.depth_linear_b + begin * sizeof(uint16_t),
                              end - begin);
    if (end > begin)
        lfm_f32_to_bf16(scratch.proj_f.data() + begin,
                        scratch.din_b.data() + begin,
                        (int)(end - begin));
    depth_band(d.dim, lane, lanes, &begin, &end);
    std::fill(scratch.df_b.begin() + begin, scratch.df_b.begin() + end,
              (uint16_t)0);
    lane_fence(e, lane, [] {});

    for (size_t codebook = 0; codebook < d.codebooks; ++codebook) {
        depth_band(d.dim, lane, lanes, &begin, &end);
        if (end > begin)
            lfm_bf16_add(scratch.din_b.data() + codebook * d.dim + begin,
                         scratch.df_b.data() + begin,
                         scratch.x.data() + begin,
                         (int)(end - begin));
        lane_fence(e, lane, [] {});

        for (size_t layer = 0; layer < d.layers.size(); ++layer) {
            const LfmDepthLayerV1 &weights = d.layers[layer];
            const size_t cache_base = layer * d.kv_heads * d.codebooks * d.hd;

            depth_norm(e, lane, scratch.x.data(), weights.op_norm,
                       scratch.xn.data());
            depth_gemv(weights.qkv_w, scratch.xn.data(), scratch.qkv_f.data(),
                       qkv_rows,
                       d.dim, lane, lanes);
            depth_band(qkv_rows, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_f32_to_bf16(scratch.qkv_f.data() + begin,
                                scratch.qkv_b.data() + begin,
                                (int)(end - begin));
            lane_fence(e, lane, [] {});

            const size_t normalized_heads = d.heads_total + d.kv_heads;
            depth_band(normalized_heads, lane, lanes, &begin, &end);
            for (size_t head = begin; head < end; ++head) {
                if (head < d.heads_total) {
                    uint16_t bits[128];
                    depth_qk_head(d, scratch.qkv_b.data() + head * d.hd,
                                  weights.q_ln, bits, codebook);
                    lfm_bf16_to_f32(bits,
                                    scratch.q_f.data() + head * d.hd,
                                    (int)d.hd);
                    continue;
                }
                const size_t kv = head - d.heads_total;
                uint16_t *key = scratch.k_plane.data() + cache_base +
                                (kv * d.codebooks + codebook) * d.hd;
                depth_qk_head(d,
                              scratch.qkv_b.data() + d.dim + kv * d.hd,
                              weights.k_ln, key, codebook);
                const uint16_t *value = scratch.qkv_b.data() + d.dim +
                                        d.kv_heads * d.hd + kv * d.hd;
                std::memcpy(scratch.v_plane.data() + cache_base +
                                (kv * d.codebooks + codebook) * d.hd,
                            value, d.hd * sizeof(uint16_t));
            }
            lane_fence(e, lane, [] {});

            depth_band(d.heads_total, lane, lanes, &begin, &end);
            const int live = (int)(codebook + 1);
            for (size_t query = begin; query < end; ++query) {
                float attention[64];
                const size_t kv = query / group;
                lfm_attn_qk_bf16(scratch.q_f.data() + query * d.hd,
                                  scratch.k_plane.data() + cache_base +
                                      kv * d.codebooks * d.hd,
                                  attention, live, (int)d.hd);
                lfm_softmax_scaled_f32(attention, live, attn_scale);
                lfm_attn_av_bf16(attention,
                                  scratch.v_plane.data() + cache_base +
                                      kv * d.codebooks * d.hd,
                                  scratch.attn_f.data() + query * d.hd, live,
                                  (int)d.hd);
            }
            lane_fence(e, lane, [] {});

            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_f32_to_bf16(scratch.attn_f.data() + begin,
                                scratch.attn_b.data() + begin,
                                (int)(end - begin));
            lane_fence(e, lane, [] {});

            depth_gemv(weights.out_w, scratch.attn_b.data(),
                       scratch.proj_f.data(), d.dim,
                       d.dim, lane, lanes);
            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin) {
                lfm_f32_to_bf16(scratch.proj_f.data() + begin,
                                scratch.y_b.data() + begin,
                                (int)(end - begin));
                lfm_bf16_add(scratch.y_b.data() + begin,
                             scratch.x.data() + begin,
                             scratch.h.data() + begin, (int)(end - begin));
            }
            lane_fence(e, lane, [] {});

            depth_norm(e, lane, scratch.h.data(), weights.ffn_norm,
                       scratch.xn.data());
            depth_gemv(weights.w1, scratch.xn.data(), scratch.proj_f.data(),
                       d.ffn, d.dim,
                       lane, lanes);
            depth_gemv(weights.w3, scratch.xn.data(), scratch.up_f.data(),
                       d.ffn, d.dim,
                       lane, lanes);
            depth_band(d.ffn, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_swiglu_bf16(scratch.proj_f.data() + begin,
                                scratch.up_f.data() + begin,
                                scratch.t_b.data() + begin, (int)(end - begin));
            lane_fence(e, lane, [] {});

            depth_gemv(weights.w2, scratch.t_b.data(), scratch.proj_f.data(),
                       d.dim, d.ffn,
                       lane, lanes);
            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin) {
                lfm_f32_to_bf16(scratch.proj_f.data() + begin,
                                scratch.y_b.data() + begin,
                                (int)(end - begin));
                lfm_bf16_add(scratch.y_b.data() + begin,
                             scratch.h.data() + begin,
                             scratch.x.data() + begin, (int)(end - begin));
            }
            lane_fence(e, lane, [] {});
        }

        const LfmDepthHeadV1 &head = d.heads[codebook];
        depth_norm(e, lane, scratch.x.data(), head.norm, scratch.xn.data());
        depth_gemv(head.logits, scratch.xn.data(), scratch.proj_f.data(),
                   head.vocab, d.dim,
                   lane, lanes);
        depth_band(head.vocab, lane, lanes, &begin, &end);
        if (end > begin)
            lfm_f32_to_bf16(scratch.proj_f.data() + begin,
                            scratch.logits_b.data() + begin,
                            (int)(end - begin));
        lane_fence(e, lane, [] {});

        SampleReq sample = {
            .logits = scratch.logits_b.data(),
            .count = head.vocab,
            .dtype = SAMPLE_BF16,
            .config = request.sampler,
            .state = request.sample_state,
            .out = request.out_tokens + codebook,
        };
        run_sampler(e, lane, sample);

        const size_t token = request.out_tokens[codebook];
        depth_band(d.dim, lane, lanes, &begin, &end);
        lfm_bf16_copy_bytes(
            depth_bytes(head.embedding) +
                (token * d.dim + begin) * sizeof(uint16_t),
            scratch.df_b.data() + begin, end - begin);
        lane_fence(e, lane, [] {});
    }
    if (lane == 0 && request.completion_status != 0)
        e->active_status.store(request.completion_status,
                               std::memory_order_release);
}

// ---- the pass programs (every lane runs these, whole) ----------------------------------
// `tiles` is computed identically by every lane from shared immutable state (it is
// the partial-fold structure — pinned numerics); `first_pre` wires the Pass pointers
// exactly once (fence serial) before any ST_SUMSQ tile runs.
template <typename F>
static void run_mlp(Engine *e, uint32_t lane, uint32_t tiles, F &&first_pre) {
    Pass *p = &e->pass;

    run_stage(e, lane, ST_SUMSQ, tiles, 0, std::forward<F>(first_pre));

    run_stage(e, lane, ST_NORM, tiles, 0, [&] {
        // Serial fold in fixed tile order — matches the reference exactly.
        float total = lfm_sum_f32(p->partials, tiles);
        float rs = lfm_inv_rms_f32(total, p->h, p->eps);
        uint32_t rsb;
        std::memcpy(&rsb, &rs, 4);
        p->rs_bits.store(rsb, std::memory_order_release);
    });

    uint32_t i_chunk = (uint32_t)((p->i + tiles - 1) / tiles);
    run_stage(e, lane, ST_GATEUP, (uint32_t)((p->i + i_chunk - 1) / i_chunk), i_chunk,
              [] {});

    uint32_t h_chunk = (uint32_t)((p->h + tiles - 1) / tiles);
    if (h_chunk > DOWN_BAND_CAP) h_chunk = DOWN_BAND_CAP;
    run_stage(e, lane, ST_DOWN, (uint32_t)((p->h + h_chunk - 1) / h_chunk), h_chunk,
              [] {});
}

// One whole shortconv+MLP layer, lane-uniform. Stage bodies are
// decode.rs::fused_shortconv_decode ported verbatim: candle-order sumsq and the tiny
// conv update run in fence serial sections (the reference computes them once on lane
// 0), banded elementwise/GEMV stages claim off the counter, then the MLP block on
// the layer's ffn weights with the conv output as its input — all without leaving
// the engine. Every lane executes this whole function with identical arguments.
static void run_conv_block(Engine *e, uint32_t lane, const LfmLayerDesc *d,
                           Bf16Input x, const uint16_t *state_in,
                           uint16_t *state_out, uint16_t *out, size_t lanes) {
    ScPass *c = &e->sc;
    const size_t h = e->model->h;
    uint32_t sc_tiles = (uint32_t)(lanes > h ? h : lanes);
    uint32_t hc = (uint32_t)((h + sc_tiles - 1) / sc_tiles);
    uint32_t pc = (uint32_t)((3 * h + sc_tiles - 1) / sc_tiles);

    run_stage(e, lane, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc, [&] {
        // Wire the shortconv stage pointers for this pass, then candle's exact
        // serial reduction — the previous layer's writes are complete (fence).
        c->x = x;
        c->norm_w = d->op_norm_w;
        c->in_w = d->in_w;
        c->out_w = d->out_w;
        c->state_out = state_out;
        c->h = h;
        c->k = d->k;
        c->xn = e->sc_xn.data();
        c->bcxf = e->sc_bcxf.data();
        c->bcxb = e->sc_bcxb.data();
        c->conv = e->sc_conv.data();
        c->projf = e->sc_projf.data();
        c->projb = e->sc_projb.data();
        c->stage = e->sc_stage.data();
        c->mid = e->sc_mid.data();
        float total = lfm_bf16_sumsq_candle_f32(x.data(), (int)h);
        float inv_rms = lfm_inv_rms_f32(total, h, d->op_eps);
        uint32_t rsb;
        std::memcpy(&rsb, &inv_rms, 4);
        c->rs_bits.store(rsb, std::memory_order_release);
    });

    run_stage(e, lane, ST_SC_INPROJ, (uint32_t)((3 * h + pc - 1) / pc), pc, [] {});

    run_stage(e, lane, ST_SC_GATHER, (uint32_t)((h + hc - 1) / hc), hc, [&] {
        // Conv update (serial — ~0.1% of the block; reference: lane 0). In-place
        // carried state is safe: this reads state_in fully before any ST_SC_GATHER
        // tile writes state_out.
        lfm_conv1d_update_bf16(c->bcxb, state_in, d->conv_w, c->conv, 1, (int)h, 1,
                               (int)d->k);
    });

    run_stage(e, lane, ST_SC_OUTPROJ, (uint32_t)((h + hc - 1) / hc), hc, [] {});

    // MLP block on the layer's ffn weights: input = mid, output = out. The wiring
    // rides ST_SUMSQ's fence serial — the fence also proves ST_SC_OUTPROJ drained
    // (xn reuse: pipelining these blocks would corrupt the plane).
    size_t cap = h < e->model->ffn ? h : e->model->ffn;
    uint32_t m_tiles = (uint32_t)(lanes > cap ? cap : lanes);
    run_mlp(e, lane, m_tiles, [&] {
        Pass *m = &e->pass;
        m->x = c->mid;
        m->norm_w = d->ffn_norm_w;
        m->w1 = d->w1;
        m->w3 = d->w3;
        m->w2 = d->w2;
        m->out = out;
        m->h = h;
        m->i = e->model->ffn;
        m->eps = d->ffn_eps;
        m->tiles = m_tiles;
        m->partials = e->sc_partials.data();
        m->xn = e->sc_xn.data();
        m->gu = e->sc_gu.data();
        m->t = e->sc_t.data();
        m->rs_bits.store(0, std::memory_order_relaxed);
    });
}

// Serial per-head helpers for the attention pass (tiny next to the GEMVs; the
// reference computes these as whole-tensor candle ops — the math below is the exact
// per-element ladder those ops perform).

// candle RmsNorm::forward over one head row: ALL f32 arithmetic (upcast, mean via the
// candle-order sum, +eps, sqrt, recip, muls), ONE bf16 storage round at the end.
static void qk_norm_row(const uint16_t *x, WeightBytes w, uint16_t *out, size_t hd,
                        float eps) {
    float total = lfm_bf16_sumsq_candle_f32(x, (int)hd);
    float inv = lfm_inv_rms_f32(total, hd, eps);
    lfm_bf16_rmsnorm(x, w, out, (int)hd, inv);
}

// candle rotary_emb::rope_slow over one head row, NeoX half-split, computed in bf16
// exactly as the tensor ops do: cos2 = [cos|cos], out = x⊙cos2 + rotate_half(x)⊙sin2,
// where every bf16 multiply and the add each round once (half-crate semantics:
// f32 compute, RNE back to bf16). rotate_half = [-x2 | x1]; negation is exact.
static void rope_slow_row(uint16_t *x, const uint16_t *cos_row, const uint16_t *sin_row,
                          size_t hd) {
    lfm_bf16_rope_neox(x, cos_row, sin_row, hd);
}

// One whole attention+MLP layer, lane-uniform. Stage bodies are the candle wrapper
// ops and attn_decode_bf16 ported at the same rounding points; the serial section
// (qk-norm, rope, KV append) is per-head work two orders of magnitude below the
// GEMVs and rides ST_AT_HEAD's fence serial. Every lane executes this whole function
// with identical arguments.
static void run_attn_block(Engine *e, uint32_t lane, size_t layer_idx,
                           Bf16Input x, uint16_t *k_plane, uint16_t *v_plane,
                           size_t head_stride, size_t pos, const uint16_t *cos_base,
                           const uint16_t *sin_base, uint16_t *out, size_t lanes) {
    const LfmLayerDesc *d = &e->model->layers[layer_idx];
    ScPass *c = &e->sc;
    AtPass *a = &e->at;
    const size_t h = e->model->h;
    const size_t nh = d->n_head, nkv = d->n_kv, hd = d->hd;
    uint32_t tiles = (uint32_t)(lanes > h ? h : lanes);
    uint32_t hc = (uint32_t)((h + tiles - 1) / tiles);

    run_stage(e, lane, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc, [&] {
        // Wire stage pointers. The conv pass planes are reused where shapes allow —
        // a single request is in flight at a time, never both kinds at once.
        e->attn.layer = layer_idx; // ST_AT_QKV routes weights via this index
        c->x = x;
        c->norm_w = d->op_norm_w;
        c->h = h;
        c->xn = e->sc_xn.data();
        c->projf = e->sc_projf.data(); // ST_AT_OPROJ reuses the conv proj planes
        c->stage = e->sc_stage.data();
        a->o_w = d->o_w;
        a->qkvf = e->at_qkvf.data();
        a->qkvb = e->at_qkvb.data();
        a->ybits = e->at_y.data();
        a->att = e->at_att.data();
        a->x = x;
        a->mid = e->sc_mid.data();
        a->k_plane = k_plane;
        a->v_plane = v_plane;
        a->head_stride = head_stride;
        a->att_len = pos + 1;
        a->max_ctx = e->model->max_ctx;
        a->h = h;
        a->n_head = nh;
        a->n_kv = nkv;
        a->hd = hd;
        // operator norm: candle-order sumsq, serial.
        float total = lfm_bf16_sumsq_candle_f32(x.data(), (int)h);
        float inv_rms = lfm_inv_rms_f32(total, h, d->op_eps);
        uint32_t rsb;
        std::memcpy(&rsb, &inv_rms, 4);
        c->rs_bits.store(rsb, std::memory_order_release);
    });

    // q|k|v projections, banded over the concatenated row space.
    size_t total_rows = (nh + 2 * nkv) * hd;
    uint32_t qc = (uint32_t)((total_rows + tiles - 1) / tiles);
    run_stage(e, lane, ST_AT_QKV, (uint32_t)((total_rows + qc - 1) / qc), qc, [] {});

    // Attention: one tile per q head over the (then pos+1)-row planes; the fence
    // serial first runs per-head qk-norm + rope and appends this step's K/V rows.
    run_stage(e, lane, ST_AT_HEAD, (uint32_t)nh, 1, [&] {
        const uint16_t *cos_row = cos_base + pos * (hd / 2);
        const uint16_t *sin_row = sin_base + pos * (hd / 2);
        uint16_t *qrows = a->qkvb;
        uint16_t *krows = a->qkvb + nh * hd;
        const uint16_t *vrows = a->qkvb + (nh + nkv) * hd;
        for (size_t qh = 0; qh < nh; ++qh) {
            qk_norm_row(qrows + qh * hd, d->qn_w, qrows + qh * hd, hd, d->qk_eps);
            rope_slow_row(qrows + qh * hd, cos_row, sin_row, hd);
        }
        for (size_t kh = 0; kh < nkv; ++kh) {
            qk_norm_row(krows + kh * hd, d->kn_w, krows + kh * hd, hd, d->qk_eps);
            rope_slow_row(krows + kh * hd, cos_row, sin_row, hd);
            std::memcpy(k_plane + kh * head_stride + pos * hd, krows + kh * hd,
                        hd * sizeof(uint16_t));
            std::memcpy(v_plane + kh * head_stride + pos * hd, vrows + kh * hd,
                        hd * sizeof(uint16_t));
        }
    });

    // o_proj + residual → mid.
    run_stage(e, lane, ST_AT_OPROJ, (uint32_t)((h + hc - 1) / hc), hc, [] {});

    // MLP block on the layer's ffn weights: input = mid, output = the request's out.
    size_t cap = h < e->model->ffn ? h : e->model->ffn;
    uint32_t m_tiles = (uint32_t)(lanes > cap ? cap : lanes);
    run_mlp(e, lane, m_tiles, [&] {
        Pass *m = &e->pass;
        m->x = a->mid;
        m->norm_w = d->ffn_norm_w;
        m->w1 = d->w1;
        m->w3 = d->w3;
        m->w2 = d->w2;
        m->out = out;
        m->h = h;
        m->i = e->model->ffn;
        m->eps = d->ffn_eps;
        m->tiles = m_tiles;
        m->partials = e->sc_partials.data();
        m->xn = e->sc_xn.data();
        m->gu = e->sc_gu.data();
        m->t = e->sc_t.data();
        m->rs_bits.store(0, std::memory_order_relaxed);
    });
}

// THE token pass: embed → every layer (ping-pong hidden planes, per-layer state from
// the request) → final embedding-norm → tied logits. One doorbell per token; the
// doorbell (shutdown/cancel) is observed only between passes. Lane-uniform: every
// lane walks the identical layer program (control flow reads only shared state, so
// the lanes' h0/h1 locals swap in lockstep).
static void run_token_pass(Engine *e, uint32_t lane) {
    const TokenReq *t = &e->tok;
    BackbonePlan *model = e->model;
    const size_t h = model->h;
    const size_t lanes = t->lanes;
    uint16_t *plane0 = e->tk_h0.data();
    uint16_t *plane1 = e->tk_h1.data();
    Bf16Input hidden{};
    uint16_t *next = plane1;

    // Text and provided audio-in embeddings are immutable for the complete pass,
    // so layer zero consumes their resident/borrowed rows directly. Only the
    // multi-codebook audio embedding needs an activation plane because summation
    // is real computation rather than transport.
    if (t->embed_kind == 2) {
        hidden = Bf16Input::from_activation(t->provided_embed);
    } else if (t->embed_kind == 0) {
        hidden = Bf16Input::from_resident(
            weight_offset(model->embed_w, (size_t)t->ids[0] * h));
    } else if (t->embed_kind == 1) {
        hidden = Bf16Input::from_activation(plane0);
        lane_fence(e, lane, [&] {
            // Embed (serial — at most 8 rows). Audio matches candle's `.sum(0)`:
            // sequential BF16 adds from zero, with one RNE round per step.
            std::memset(plane0, 0, h * sizeof(uint16_t));
            for (size_t c = 0; c < t->n_ids; ++c) {
                WeightBytes row =
                    weight_offset(model->audio_embed_w, (size_t)t->ids[c] * h);
                lfm_bf16_add(plane0, row, plane0, (int)h);
            }
        });
    } else {
        e->active_status.store(-EINVAL, std::memory_order_release);
        return;
    }

    // The layer walk. The first input may be a resident weight row or borrowed
    // Conformer row; subsequent inputs alternate between the two activation
    // planes. No transport copy is required at either boundary.
    for (size_t l = 0; l < model->layers.size(); ++l) {
        const LfmLayerDesc *d = &model->layers[l];
        const LfmLayerState *st = &t->states[l];
        if (d->kind == 0) {
            run_conv_block(e, lane, d, hidden, st->conv_state, st->conv_state,
                           next, lanes);
        } else if (d->kind == 1) {
            run_attn_block(e, lane, l, hidden, st->k_plane, st->v_plane,
                           st->head_stride, t->pos, t->cos_base, t->sin_base,
                           next, lanes);
        } else {
            e->active_status.store(-EINVAL, std::memory_order_release);
            return;
        }
        hidden = Bf16Input::from_activation(next);
        next = next == plane0 ? plane1 : plane0;
    }

    // Final embedding-norm (candle RmsNorm: f32 arithmetic, one bf16 round), banded.
    ScPass *c = &e->sc;
    uint32_t tiles = (uint32_t)(lanes > h ? h : lanes);
    uint32_t hc = (uint32_t)((h + tiles - 1) / tiles);
    run_stage(e, lane, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc, [&] {
        float total = lfm_bf16_sumsq_candle_f32(hidden.data(), (int)h);
        float inv_rms = lfm_inv_rms_f32(total, h, model->emb_norm_eps);
        uint32_t rsb;
        std::memcpy(&rsb, &inv_rms, 4);
        c->rs_bits.store(rsb, std::memory_order_release);
        c->x = hidden;
        c->norm_w = model->emb_norm_w;
        c->h = h;
        c->xn = t->out_hidden;
    });

    // Tied logits head over the normed hidden — the heavy stage; real row bands.
    if ((t->out_logits || t->out_token) && model->vocab > 0) {
        uint32_t vc = (uint32_t)((model->vocab + (size_t)e->n_workers * 4 - 1) /
                                 ((size_t)e->n_workers * 4));
        run_stage(e, lane, ST_LOGITS,
                  (uint32_t)((model->vocab + vc - 1) / vc), vc, [] {});
    }
    if (t->out_token) {
        SampleReq sample = {
            .logits = e->tk_logf.data(),
            .count = model->vocab,
            .dtype = SAMPLE_F32,
            .config = *t->sampler,
            .state = t->sample_state,
            .out = t->out_token,
        };
        run_sampler(e, lane, sample);
    }
}

struct PrefillInput {
    WeightBytes embedding = nullptr;
    const uint32_t *ids = nullptr;
    const uint16_t *rows = nullptr;
    size_t h = 0;

    Bf16Input row(size_t index) const {
        if (embedding) {
            return Bf16Input::from_resident(
                weight_offset(embedding, (size_t)ids[index] * h));
        }
        return Bf16Input::from_activation(rows + index * h);
    }
};

static void prefill_band(size_t count, uint32_t lane, uint32_t lanes,
                         size_t *begin, size_t *end) {
    const size_t width = (count + lanes - 1) / lanes;
    *begin = std::min((size_t)lane * width, count);
    *end = std::min(*begin + width, count);
}

// C[M,N] = A[M,K] * W[N,K]^T. Each lane owns a disjoint W-row band, and the
// explicit output stride lets the architecture leaf reuse every checkpoint row
// across all M inputs without a scatter/copy plane.
static void prefill_linear(Engine *e, uint32_t lane, const uint16_t *a,
                           WeightBytes weight, float *out, size_t rows,
                           size_t n, size_t k, size_t stride) {
    size_t begin = 0, end = 0;
    prefill_band(n, lane, e->lanes_total, &begin, &end);
    if (end > begin) {
        WeightBytes band = weight_offset(weight, begin * k);
#if defined(__x86_64__) || defined(_M_X64)
        if (!lfm_bf16_gemm_available()) {
            lfm_bf16_gemm_nt_strided_f32_scalar(
                a, band, out + begin, (int)rows, (int)(end - begin),
                (int)k, (int)stride);
        } else
#endif
        {
            lfm_bf16_gemm_nt_strided_f32(
                a, band, out + begin, (int)rows, (int)(end - begin),
                (int)k, (int)stride);
        }
    }
    lane_fence(e, lane, [] {});
}

static void prefill_round(Engine *e, uint32_t lane, const float *input,
                          uint16_t *out, size_t count) {
    size_t begin = 0, end = 0;
    prefill_band(count, lane, e->lanes_total, &begin, &end);
    if (end > begin)
        lfm_f32_to_bf16(input + begin, out + begin, (int)(end - begin));
    lane_fence(e, lane, [] {});
}

static float prefill_weight(WeightBytes weights, size_t index) {
    const uint32_t bits = lfm_bf16_unlift_bits(weight_offset(weights, index));
    return std::bit_cast<float>(bits);
}

static void prefill_norm(Engine *e, uint32_t lane, const PrefillInput &input,
                         WeightBytes weight, float eps, uint16_t *out,
                         size_t rows) {
    for (size_t row = lane; row < rows; row += e->lanes_total) {
        const Bf16Input source = input.row(row);
        const float total = lfm_bf16_sumsq_candle_f32(source.data(), (int)input.h);
        const float inv = lfm_inv_rms_f32(total, input.h, eps);
        lfm_bf16_rmsnorm(source.data(), weight, out + row * input.h,
                         (int)input.h, inv);
    }
    lane_fence(e, lane, [] {});
}

// The MLP norm intentionally has a different reduction contract from the
// operator/final norms: decode partitions by a fixed logical tile count, then
// folds partials in tile order. Reproduce that order independently per row.
static void prefill_mlp_norm(Engine *e, uint32_t lane,
                             const PrefillInput &input, WeightBytes weight,
                             float eps, uint16_t *out, size_t rows) {
    const size_t cap = std::min(input.h, e->model->ffn);
    const size_t tiles = std::min(e->prefill.lanes, cap);
    for (size_t row = lane; row < rows; row += e->lanes_total) {
        const Bf16Input source = input.row(row);
        // MLP inputs are activation planes. Keeping the assertion structural
        // avoids manufacturing a typed pointer from resident checkpoint bytes.
        const uint16_t *values = source.activation;
        float partials[MAX_WORKERS] = {};
        for (size_t tile = 0; tile < tiles; ++tile) {
            partials[tile] = lfm_bf16_sumsq_stride_f32(
                values, input.h, tile, tiles);
        }
        const float total = lfm_sum_f32(partials, tiles);
        const float inv = lfm_inv_rms_f32(total, input.h, eps);
        lfm_bf16_rmsnorm(values, weight, out + row * input.h,
                         (int)input.h, inv);
    }
    lane_fence(e, lane, [] {});
}

static void prefill_add(Engine *e, uint32_t lane, const uint16_t *left,
                        const PrefillInput &right, uint16_t *out, size_t rows) {
    size_t begin = 0, end = 0;
    prefill_band(right.h, lane, e->lanes_total, &begin, &end);
    if (end > begin) {
        for (size_t row = 0; row < rows; ++row) {
            lfm_bf16_add(left + row * right.h + begin,
                         right.row(row).offset(begin).data(),
                         out + row * right.h + begin, (int)(end - begin));
        }
    }
    lane_fence(e, lane, [] {});
}

static void run_prefill_mlp(Engine *e, uint32_t lane, const LfmLayerDesc *desc,
                            const uint16_t *input, uint16_t *out, size_t rows) {
    PrefillWorkspace *w = e->prefill.workspace;
    const size_t h = e->model->h;
    const size_t ffn = e->model->ffn;
    const PrefillInput source = {.rows = input, .h = h};

    prefill_mlp_norm(e, lane, source, desc->ffn_norm_w, desc->ffn_eps,
                     w->xn.data(), rows);
    prefill_linear(e, lane, w->xn.data(), desc->w1, w->gu.data(), rows,
                   ffn, h, 2 * ffn);
    prefill_linear(e, lane, w->xn.data(), desc->w3, w->gu.data() + ffn,
                   rows, ffn, h, 2 * ffn);

    size_t begin = 0, end = 0;
    prefill_band(ffn, lane, e->lanes_total, &begin, &end);
    if (end > begin) {
        for (size_t row = 0; row < rows; ++row) {
            lfm_swiglu_bf16(w->gu.data() + row * 2 * ffn + begin,
                            w->gu.data() + row * 2 * ffn + ffn + begin,
                            w->gate.data() + row * ffn + begin,
                            (int)(end - begin));
        }
    }
    lane_fence(e, lane, [] {});

    prefill_linear(e, lane, w->gate.data(), desc->w2, w->projf.data(),
                   rows, h, ffn, h);
    prefill_round(e, lane, w->projf.data(), w->stage.data(), rows * h);
    prefill_add(e, lane, w->stage.data(), source, out, rows);
}

static void run_prefill_conv(Engine *e, uint32_t lane,
                             const LfmLayerDesc *desc,
                             const PrefillInput &input, uint16_t *state,
                             uint16_t *out, size_t rows) {
    PrefillWorkspace *w = e->prefill.workspace;
    const size_t h = e->model->h;
    const size_t kernel = desc->k;

    prefill_norm(e, lane, input, desc->op_norm_w, desc->op_eps,
                 w->xn.data(), rows);
    prefill_linear(e, lane, w->xn.data(), desc->in_w, w->bcxf.data(),
                   rows, 3 * h, h, 3 * h);
    prefill_round(e, lane, w->bcxf.data(), w->bcxb.data(), rows * 3 * h);

    // Each channel owns one causal carry chain. Load its <=8 resident taps once,
    // then advance rows in order with the exact Bx -> FIR -> C rounding ladder.
    // This removes the generic one-row leaf's thread-local growable window and
    // makes the complete prefill path allocation-free after workspace publish.
    size_t begin = 0, end = 0;
    prefill_band(h, lane, e->lanes_total, &begin, &end);
#if defined(__aarch64__) || defined(_M_ARM64)
    constexpr bool fast_k3 = true;
#else
    constexpr bool fast_k3 = false;
#endif
    for (size_t channel = begin; channel < end; ++channel) {
        uint16_t carry[7];
        const WeightBytes tap_row =
            weight_offset(desc->conv_w, channel * kernel);
        const float w0 = prefill_weight(tap_row, 0);
        const float w1 = kernel > 1 ? prefill_weight(tap_row, 1) : 0.0f;
        const float w2 = kernel > 2 ? prefill_weight(tap_row, 2) : 0.0f;
        const float w3 = kernel > 3 ? prefill_weight(tap_row, 3) : 0.0f;
        const float w4 = kernel > 4 ? prefill_weight(tap_row, 4) : 0.0f;
        const float w5 = kernel > 5 ? prefill_weight(tap_row, 5) : 0.0f;
        const float w6 = kernel > 6 ? prefill_weight(tap_row, 6) : 0.0f;
        const float w7 = kernel > 7 ? prefill_weight(tap_row, 7) : 0.0f;
        for (size_t tap = 0; tap + 1 < kernel; ++tap)
            carry[tap] = state[channel * (kernel - 1) + tap];
        for (size_t row = 0; row < rows; ++row) {
            const uint16_t *bcx = w->bcxb.data() + row * 3 * h;
            const uint16_t bx_bits = rb_bits(
                bf16_f32(bcx[channel]) * bf16_f32(bcx[2 * h + channel]));
            const float bx = bf16_f32(bx_bits);
            const float v0 = kernel == 1 ? bx : bf16_f32(carry[0]);
            // The AArch64 T=1,K=3 fast leaf starts from tap zero; the generic
            // twins add it to +0. Preserve that signed-zero/NaN distinction.
            float acc = fast_k3 && kernel == 3 ? w0 * v0 : 0.0f + w0 * v0;
            if (kernel > 1)
                acc = acc + w1 * (kernel == 2 ? bx : bf16_f32(carry[1]));
            if (kernel > 2)
                acc = acc + w2 * (kernel == 3 ? bx : bf16_f32(carry[2]));
            if (kernel > 3)
                acc = acc + w3 * (kernel == 4 ? bx : bf16_f32(carry[3]));
            if (kernel > 4)
                acc = acc + w4 * (kernel == 5 ? bx : bf16_f32(carry[4]));
            if (kernel > 5)
                acc = acc + w5 * (kernel == 6 ? bx : bf16_f32(carry[5]));
            if (kernel > 6)
                acc = acc + w6 * (kernel == 7 ? bx : bf16_f32(carry[6]));
            if (kernel > 7) acc = acc + w7 * bx;
            const uint16_t conv = rb_bits(acc);
            w->projb[row * h + channel] = rb_bits(
                bf16_f32(bcx[h + channel]) * bf16_f32(conv));
            if (kernel > 1) {
                for (size_t tap = 0; tap + 2 < kernel; ++tap)
                    carry[tap] = carry[tap + 1];
                carry[kernel - 2] = bx_bits;
            }
        }
        for (size_t tap = 0; tap + 1 < kernel; ++tap)
            state[channel * (kernel - 1) + tap] = carry[tap];
    }
    lane_fence(e, lane, [] {});

    prefill_linear(e, lane, w->projb.data(), desc->out_w,
                   w->projf.data(), rows, h, h, h);
    prefill_round(e, lane, w->projf.data(), w->stage.data(), rows * h);
    prefill_add(e, lane, w->stage.data(), input, w->mid.data(), rows);
    run_prefill_mlp(e, lane, desc, w->mid.data(), out, rows);
}

static void run_prefill_attention(Engine *e, uint32_t lane, size_t layer,
                                  const PrefillInput &input,
                                  const LfmLayerState *state, uint16_t *out,
                                  size_t rows) {
    PrefillWorkspace *w = e->prefill.workspace;
    const PrefillReq *request = &e->prefill;
    const LfmLayerDesc *desc = &e->model->layers[layer];
    const size_t h = e->model->h;
    const size_t nh = desc->n_head;
    const size_t nkv = desc->n_kv;
    const size_t hd = desc->hd;
    const size_t qrows = nh * hd;
    const size_t kvrows = nkv * hd;
    const size_t qkv = qrows + 2 * kvrows;

    prefill_norm(e, lane, input, desc->op_norm_w, desc->op_eps,
                 w->xn.data(), rows);
    prefill_linear(e, lane, w->xn.data(), desc->q_w, w->qkvf.data(),
                   rows, qrows, h, qkv);
    prefill_linear(e, lane, w->xn.data(), desc->k_w,
                   w->qkvf.data() + qrows, rows, kvrows, h, qkv);
    prefill_linear(e, lane, w->xn.data(), desc->v_w,
                   w->qkvf.data() + qrows + kvrows, rows, kvrows, h, qkv);
    prefill_round(e, lane, w->qkvf.data(), w->qkvb.data(), rows * qkv);

    const size_t norm_tasks = rows * (nh + nkv);
    for (size_t task = lane; task < norm_tasks; task += e->lanes_total) {
        const size_t row = task / (nh + nkv);
        const size_t head = task % (nh + nkv);
        const uint16_t *cos = request->cos_base +
                              (request->pos + row) * (hd / 2);
        const uint16_t *sin = request->sin_base +
                              (request->pos + row) * (hd / 2);
        uint16_t *base = w->qkvb.data() + row * qkv;
        if (head < nh) {
            uint16_t *query = base + head * hd;
            qk_norm_row(query, desc->qn_w, query, hd, desc->qk_eps);
            rope_slow_row(query, cos, sin, hd);
            continue;
        }
        const size_t kh = head - nh;
        uint16_t *key = base + qrows + kh * hd;
        const uint16_t *value = base + qrows + kvrows + kh * hd;
        qk_norm_row(key, desc->kn_w, key, hd, desc->qk_eps);
        rope_slow_row(key, cos, sin, hd);
        lfm_bf16_copy_bytes(key,
                            state->k_plane + kh * state->head_stride +
                                (request->pos + row) * hd,
                            hd);
        lfm_bf16_copy_bytes(value,
                            state->v_plane + kh * state->head_stride +
                                (request->pos + row) * hd,
                            hd);
    }
    lane_fence(e, lane, [] {});

    const size_t group = nh / nkv;
    for (size_t task = lane; task < rows * nh; task += e->lanes_total) {
        const size_t row = task / nh;
        const size_t qh = task % nh;
        const size_t kh = qh / group;
        const size_t length = request->pos + row + 1;
        const uint16_t *query = w->qkvb.data() + row * qkv + qh * hd;
        float *score = w->scores.data() + lane * w->max_ctx;
        float qf[512];
        float value[512];
        lfm_bf16_to_f32(query, qf, (int)hd);
        lfm_attn_qk_bf16(qf, state->k_plane + kh * state->head_stride,
                         score, (int)length, (int)hd);
        lfm_softmax_scaled_f32(score, (int)length, lfm_rsqrt_size(hd));
        lfm_attn_av_bf16(score, state->v_plane + kh * state->head_stride,
                         value, (int)length, (int)hd);
        lfm_f32_to_bf16(value,
                        w->att_y.data() + row * (nh * hd) + qh * hd,
                        (int)hd);
    }
    lane_fence(e, lane, [] {});

    prefill_linear(e, lane, w->att_y.data(), desc->o_w, w->projf.data(),
                   rows, h, nh * hd, h);
    prefill_round(e, lane, w->projf.data(), w->stage.data(), rows * h);
    prefill_add(e, lane, w->stage.data(), input, w->mid.data(), rows);
    run_prefill_mlp(e, lane, desc, w->mid.data(), out, rows);
}

static void run_prefill(Engine *e, uint32_t lane) {
    const PrefillReq *request = &e->prefill;
    PrefillWorkspace *w = request->workspace;
    const size_t rows = request->rows;
    const size_t h = e->model->h;
    PrefillInput first{};
    if (request->embed_kind == 0) {
        first = {.embedding = e->model->embed_w,
                 .ids = request->ids, .h = h};
    } else if (request->embed_kind == 2) {
        first = {.rows = request->provided_rows, .h = h};
    } else {
        e->active_status.store(-EINVAL, std::memory_order_release);
        return;
    }
    PrefillInput hidden = first;
    uint16_t *next = w->h1.data();

    for (size_t layer = 0; layer < e->model->layers.size(); ++layer) {
        const LfmLayerDesc *desc = &e->model->layers[layer];
        const LfmLayerState *state = &request->states[layer];
        if (desc->kind == 0) {
            run_prefill_conv(e, lane, desc, hidden, state->conv_state,
                             next, rows);
        } else if (desc->kind == 1) {
            run_prefill_attention(e, lane, layer, hidden, state, next, rows);
        } else {
            e->active_status.store(-EINVAL, std::memory_order_release);
            return;
        }
        hidden = {.rows = next, .h = h};
        next = next == w->h0.data() ? w->h1.data() : w->h0.data();
    }

    const PrefillInput last = {.rows = hidden.row(rows - 1).activation, .h = h};
    prefill_norm(e, lane, last, e->model->emb_norm_w,
                 e->model->emb_norm_eps, request->out_hidden, 1);

    if (request->out_token) {
        prefill_linear(e, lane, request->out_hidden, e->model->embed_w,
                       w->logits.data(), 1, e->model->vocab, h,
                       e->model->vocab);
        SampleReq sample = {
            .logits = w->logits.data(),
            .count = e->model->vocab,
            .dtype = SAMPLE_F32,
            .config = *request->sampler,
            .state = request->sample_state,
            .out = request->out_token,
        };
        run_sampler(e, lane, sample);
    }
}

static void run_prng_pass(Engine *e, uint32_t lane) {
    lane_fence(e, lane, [&] {
        int status = lfm_prng_fill_u64(e->prng.state, e->prng.out, e->prng.count);
        e->active_status.store(status, std::memory_order_release);
    });
}

static void run_sample_pass(Engine *e, uint32_t lane) {
    run_sampler(e, lane, e->sample);
}

static void run_depthwise_stream(Engine *e, uint32_t lane) {
    const DepthwiseStreamReq &request = e->depthwise_stream;
    const size_t prior = request.kernel - 1;
    const size_t rows = request.batch * request.channels;
    for (size_t row = lane; row < rows; row += e->lanes_total) {
        const size_t channel = row % request.channels;
        lfm_depthwise_stream_bf16(
            request.x + row * request.steps,
            request.cache ? request.cache + row * prior : nullptr,
            request.weights + channel * request.kernel,
            request.out + row * request.steps,
            prior ? request.next + row * prior : nullptr, 1, 1,
            (int)request.steps, (int)request.kernel);
    }
}

static void run_gemm(Engine *e, uint32_t lane) {
    const GemmReq &request = e->gemm;
    const size_t lanes = e->lanes_total;
    const bool scalar_nt = request.direct && !lfm_bf16_gemm_available();

    if (request.rhs_layout == LFM_GEMM_RHS_KN && request.m == 1) {
        if (lane == 0)
            lfm_bf16_gemv_f32(request.a,
                              static_cast<const uint16_t *>(request.rhs),
                              request.out,
                              (int)request.n, (int)request.k);
        return;
    }

    if (request.rhs_layout == LFM_GEMM_RHS_NK && request.m == 1) {
        const size_t cols = std::max<size_t>((request.n + lanes - 1) / lanes, 64);
        const size_t col = (size_t)lane * cols;
        if (col < request.n) {
            const size_t count = std::min(cols, request.n - col);
            const void *weights =
                static_cast<const unsigned char *>(request.rhs) +
                col * request.k * sizeof(uint16_t);
            if (scalar_nt)
                lfm_bf16_gemm_nt_f32_scalar(
                    request.a, weights, request.out + col, 1, (int)count,
                    (int)request.k);
            else
                lfm_bf16_gemm_nt_f32(request.a, weights,
                                     request.out + col, 1, (int)count,
                                     (int)request.k);
        }
        return;
    }

#if defined(__aarch64__)
    const size_t rows = std::max<size_t>((request.m + lanes - 1) / lanes, 8);
#else
    const size_t rows = std::max<size_t>((request.m + lanes - 1) / lanes, 1);
#endif
    const size_t row = (size_t)lane * rows;
    if (row >= request.m) return;
    const size_t count = std::min(rows, request.m - row);
    if (request.rhs_layout == LFM_GEMM_RHS_KN) {
        lfm_bf16_gemm_f32_v2(
                             request.a + row * request.k,
                             static_cast<const uint16_t *>(request.rhs),
                             request.out + row * request.n, (int)count,
                             (int)request.n, (int)request.k);
        return;
    }
    if (scalar_nt)
        lfm_bf16_gemm_nt_f32_scalar(
            request.a + row * request.k, request.rhs,
            request.out + row * request.n, (int)count, (int)request.n,
            (int)request.k);
    else
        lfm_bf16_gemm_nt_f32(request.a + row * request.k, request.rhs,
                             request.out + row * request.n, (int)count,
                             (int)request.n, (int)request.k);
}

static int execute_audio_encode(Engine *e, AudioReq &audio) {
    const LfmAudioEncodePassV1 &pass = audio.pass;
    *pass.out_adapted_values = 0;

    LfmF32Span samples{};
    int status = lfm_resampler_process(
        pass.resampler, pass.resampler_workspace, pass.pcm,
        pass.sample_count, pass.resampled, pass.resampled_capacity, &samples);
    if (status != 0) return status;

    const uint64_t frames = lfm_frontend_seq_len(pass.frontend, samples.length);
    if (frames == 0) return -EINVAL;
    status = lfm_frontend_forward_bf16_workspace(
        pass.frontend, pass.frontend_workspace, samples.data, samples.length,
        pass.mel, pass.mel_capacity);
    if (status != 0) return status;

    const uint64_t rows = lfm_conformer_out_rows(pass.conformer, frames);
    const uint64_t width = lfm_conformer_out_width(pass.conformer);
    if (rows == 0 || width == 0 || rows > UINT64_MAX / width) {
        return rows == 0 || width == 0 ? -EINVAL : -EOVERFLOW;
    }
    const uint64_t values = rows * width;
    if (values > pass.adapted_capacity) return -ENOBUFS;
    if (e->model && width != e->model->h) return -ESTALE;

    status = lfm_conformer_forward_engine_team(
        pass.conformer, pass.conformer_workspace, pass.mel, frames,
        pass.adapted, pass.adapted_capacity);
    if (status != 0) return status;
    *pass.out_adapted_values = values;
    return 0;
}

static void run_audio_encode(Engine *e, uint32_t lane) {
    PassSlot *slot = e->active_slot;
    if (!slot || slot->request != REQ_AUDIO_ENCODE) {
        if (lane == 0) e->active_status.store(-EFAULT, std::memory_order_release);
        return;
    }
    AudioReq &audio = slot->audio;
    if (lane == 0) {
        e->audio_encode_passes.fetch_add(1, std::memory_order_relaxed);
        const int status = execute_audio_encode(e, audio);
        e->active_status.store(status, std::memory_order_release);
        audio.done.store(true, std::memory_order_release);
        signal_all(&slot->audio_word);
        return;
    }

    uint64_t seen = audio.start_gemm_generation;
    uint32_t observed =
        kc_atomic_u32_load_acquire(&slot->audio_word.value);
    for (;;) {
        const uint64_t generation =
            audio.gemm_generation.load(std::memory_order_acquire);
        if (generation != seen) {
            seen = generation;
            run_gemm(e, lane);
            lane_fence(e, lane, [] {});
            continue;
        }
        if (audio.done.load(std::memory_order_acquire)) return;
        (void)kc_port_wait_u32(slot->audio_word.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&slot->audio_word.value);
    }
}

static inline Dd dd_from_f32(float value) { return {value, 0.0f}; }

static inline Dd dd_from_f64(double value) {
    const float hi = (float)value;
    return {hi, (float)(value - (double)hi)};
}

static inline Dd dd_quick_two_sum(float a, float b) {
    const float sum = a + b;
    return {sum, b - (sum - a)};
}

static inline Dd dd_two_sum(float a, float b) {
    const float sum = a + b;
    const float value = sum - a;
    return {sum, (a - (sum - value)) + (b - value)};
}

static inline Dd dd_two_prod(float a, float b) {
    const float product = a * b;
    return {product, std::fma(a, b, -product)};
}

static inline Dd dd_add(Dd a, Dd b) {
    Dd sum = dd_two_sum(a.hi, b.hi);
    const Dd tail = dd_two_sum(a.lo, b.lo);
    sum.lo += tail.hi;
    sum = dd_quick_two_sum(sum.hi, sum.lo);
    sum.lo += tail.lo;
    return dd_quick_two_sum(sum.hi, sum.lo);
}

static inline Dd dd_neg(Dd value) { return {-value.hi, -value.lo}; }

static inline Dd dd_sub(Dd a, Dd b) { return dd_add(a, dd_neg(b)); }

static inline Dd dd_mul(Dd a, Dd b) {
    Dd product = dd_two_prod(a.hi, b.hi);
    product.lo += a.hi * b.lo + a.lo * b.hi;
    return dd_quick_two_sum(product.hi, product.lo);
}

static inline float dd_to_f32(Dd value) { return value.hi + value.lo; }

static inline ComplexDd cdd_from_f32(float re, float im) {
    return {dd_from_f32(re), dd_from_f32(im)};
}

static inline ComplexDd cdd_add(ComplexDd a, ComplexDd b) {
    return {dd_add(a.re, b.re), dd_add(a.im, b.im)};
}

static inline ComplexDd cdd_sub(ComplexDd a, ComplexDd b) {
    return {dd_sub(a.re, b.re), dd_sub(a.im, b.im)};
}

static inline ComplexDd cdd_mul(ComplexDd a, ComplexDd b) {
    const Dd ac = dd_mul(a.re, b.re);
    const Dd bd = dd_mul(a.im, b.im);
    const Dd ad = dd_mul(a.re, b.im);
    const Dd bc = dd_mul(a.im, b.re);
    return {dd_sub(ac, bd), dd_add(ad, bc)};
}

static inline ComplexDd cdd_conj(ComplexDd value) {
    return {value.re, dd_neg(value.im)};
}

static size_t bit_reverse(size_t value, unsigned bits) {
    size_t reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit)
        reversed |= ((value >> bit) & size_t{1}) << (bits - 1 - bit);
    return reversed;
}

// One FFT is one cooperative lane program. Every butterfly stage maps to a
// threadgroup-equivalent fence: lanes mutate disjoint pairs in the shared work
// plane, then all lanes cross the generation word before the next stage reads it.
static void fft_dd_lanes(Engine *e, uint32_t lane, ComplexDd *work,
                         const ComplexDd *twiddles, size_t size) {
    if (size < 2) return;
    const size_t lanes = e->lanes_total;
    unsigned bits = 0;
    for (size_t value = size; value > 1; value >>= 1) ++bits;
    for (size_t i = lane; i < size; i += lanes) {
        const size_t reversed = bit_reverse(i, bits);
        if (i < reversed) std::swap(work[i], work[reversed]);
    }
    lane_fence(e, lane, [] {});

    for (size_t length = 2;; length <<= 1) {
        const size_t half = length / 2;
        for (size_t butterfly = lane; butterfly < size / 2; butterfly += lanes) {
            const size_t group = butterfly / half;
            const size_t offset = butterfly % half;
            const size_t a = group * length + offset;
            const size_t b = a + half;
            const ComplexDd product = cdd_mul(twiddles[offset * (size / length)], work[b]);
            const ComplexDd upper = work[a];
            work[a] = cdd_add(upper, product);
            work[b] = cdd_sub(upper, product);
        }
        lane_fence(e, lane, [] {});
        if (length == size) break;
    }
}

static void ifft_dd_lanes(Engine *e, uint32_t lane, ComplexDd *work,
                          const ComplexDd *twiddles, size_t size) {
    const size_t lanes = e->lanes_total;
    for (size_t i = lane; i < size; i += lanes) work[i] = cdd_conj(work[i]);
    lane_fence(e, lane, [] {});
    fft_dd_lanes(e, lane, work, twiddles, size);
    const Dd scale = dd_from_f32(1.0f / (float)size);
    for (size_t i = lane; i < size; i += lanes)
        work[i] = {dd_mul(work[i].re, scale), dd_neg(dd_mul(work[i].im, scale))};
    lane_fence(e, lane, [] {});
}

static void run_fft_conv_dd(Engine *e, uint32_t lane) {
    const FftConvDdReq &request = e->fft_conv_dd;
    const size_t half = request.fft_size / 2 + 1;
    const size_t signals = request.batch * request.channels;
    const size_t lanes = e->lanes_total;
    ComplexDd *work = e->fft_work.data();
    for (size_t signal = 0; signal < signals; ++signal) {
        const size_t channel = signal % request.channels;
        const float *input = request.input + signal * request.steps;
        const float *kernel = request.kernel + channel * half * 2;
        float *out = request.out + signal * request.steps;
        for (size_t i = lane; i < request.fft_size; i += lanes)
            work[i] = i < request.steps ? cdd_from_f32(input[i], 0.0f) : ComplexDd{};
        lane_fence(e, lane, [] {});
        fft_dd_lanes(e, lane, work, e->fft_twiddles.data(), request.fft_size);
        for (size_t i = lane; i < half; i += lanes)
            work[i] = cdd_mul(work[i], cdd_from_f32(kernel[2 * i], kernel[2 * i + 1]));
        lane_fence(e, lane, [] {});
        for (size_t i = half + lane; i < request.fft_size; i += lanes)
            work[i] = cdd_conj(work[request.fft_size - i]);
        lane_fence(e, lane, [] {});
        ifft_dd_lanes(e, lane, work, e->fft_twiddles.data(), request.fft_size);
        for (size_t i = lane; i < request.steps; i += lanes)
            out[i] = dd_to_f32(work[i].re) + input[i] * request.skip[channel];
        // The next signal reuses the same plane. This is the output/store barrier,
        // not a second ticket or a host-visible completion.
        lane_fence(e, lane, [] {});
    }
}

static void run_irfft_dd(Engine *e, uint32_t lane) {
    const IrfftDdReq &request = e->irfft_dd;
    const size_t frequency = request.fft_size / 2 + 1;
    const bool even = request.fft_size % 2 == 0;
    const size_t nyquist = request.fft_size / 2;
    for (size_t row = lane; row < request.rows; row += e->lanes_total) {
        for (size_t sample = 0; sample < request.fft_size; ++sample) {
            Dd sum{};
            for (size_t k = 0; k < frequency; ++k) {
                const ComplexDd twiddle =
                    e->irfft_twiddles[(uint64_t)k * sample % request.fft_size];
                Dd term = dd_sub(dd_mul(dd_from_f32(request.real[row * frequency + k]),
                                        twiddle.re),
                                 dd_mul(dd_from_f32(request.imag[row * frequency + k]),
                                        twiddle.im));
                term = dd_mul(term, dd_from_f32(k == 0 || (even && k == nyquist)
                                                   ? 1.0f : 2.0f));
                sum = dd_add(sum, term);
            }
            request.out[row * request.fft_size + sample] =
                dd_to_f32(dd_mul(sum, request.scale));
        }
    }
}

static void pause_irfft_lane_zero_for_test(Engine *e, uint32_t lane) {
    if (lane != 0) return;
    uint32_t armed = 1;
    if (!e->test_lane_pause.compare_exchange_strong(
            armed, 2, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    signal_all(&e->dispatch_word);
    while (e->test_lane_pause.load(std::memory_order_acquire) == 2 &&
           !e->retire.load(std::memory_order_acquire)) {
        const uint32_t observed =
            kc_atomic_u32_load_acquire(&e->dispatch_word.value);
        if (e->test_lane_pause.load(std::memory_order_acquire) != 2) break;
        (void)kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
    }
}

// The per-generation program, dispatched identically on every lane; the final fence
// proves ALL tiles landed before lane 0 signals the rim. Request payloads are written
// by the rim before its doorbell and read-only for the whole generation.
static void lane_program(Engine *e, uint32_t lane) {
    switch (e->cur_req) {
    case REQ_MLP:
        // The rim wired the Pass (including tiles) before the doorbell.
        run_mlp(e, lane, (uint32_t)e->pass.tiles, [] {});
        break;
    case REQ_CONV_LAYER: {
        const ConvReq *r = &e->conv;
        run_conv_block(e, lane, &e->model->layers[r->layer],
                       Bf16Input::from_activation(r->x), r->state_in,
                       r->state_out, r->out, r->lanes);
        break;
    }
    case REQ_ATTN_LAYER: {
        const AttnReq *r = &e->attn;
        run_attn_block(e, lane, r->layer, Bf16Input::from_activation(r->x),
                       r->k_plane, r->v_plane, r->head_stride,
                       r->pos, r->cos_base, r->sin_base, r->out, r->lanes);
        break;
    }
    case REQ_TOKEN_PASS:
        run_token_pass(e, lane);
        break;
    case REQ_PREFILL:
        run_prefill(e, lane);
        break;
    case REQ_PRNG:
        run_prng_pass(e, lane);
        break;
    case REQ_SAMPLE:
        run_sample_pass(e, lane);
        break;
    case REQ_DEPTH_FRAME:
        run_depth_frame(e, lane);
        break;
    case REQ_DEPTHWISE_STREAM:
        run_depthwise_stream(e, lane);
        break;
    case REQ_GEMM:
        run_gemm(e, lane);
        break;
    case REQ_FFT_CONV_DD:
        run_fft_conv_dd(e, lane);
        break;
    case REQ_IRFFT_DD:
        // Private deterministic shutdown-race seam. Ordinary model/audio/token
        // requests never touch its atomic; only this conformance leaf checks it.
        pause_irfft_lane_zero_for_test(e, lane);
        run_irfft_dd(e, lane);
        break;
    case REQ_MIMI_DECODE:
        if (lane == 0) {
            const int samples = e->mimi.completion_status != 0
                                    ? e->mimi.completion_status
                                    : mimi_decode_state_step(
                                          e->mimi.state, e->mimi.codes,
                                          e->mimi.pcm);
            if (samples < 0) {
                e->active_status.store(samples, std::memory_order_release);
            } else if (static_cast<size_t>(samples) > e->mimi.capacity) {
                e->active_status.store(-EOVERFLOW, std::memory_order_release);
            } else {
                *e->mimi.out_samples = static_cast<size_t>(samples);
            }
        }
        break;
    case REQ_AUDIO_ENCODE:
        run_audio_encode(e, lane);
        break;
    default:
        // A request selector is a closed protocol value.  This is a final
        // defense behind submission/descriptor validation: corruption must
        // become a failed completion, never a successful no-op.
        if (lane == 0)
            e->active_status.store(-EINVAL, std::memory_order_release);
        break;
    }
    lane_fence(e, lane, [] {});
}

static BlockDomain *block_for_lane(Engine *e, uint32_t lane) {
    for (uint32_t index = 0; index < e->block_count; ++index) {
        BlockDomain *block = &e->blocks[index];
        if (lane >= block->lane_begin &&
            lane < block->lane_begin + block->lane_count) {
            return block;
        }
    }
    return nullptr;
}

static void publish_block_completion(Engine *e, BlockDomain *block,
                                     uint64_t generation, int status) {
    const uint64_t head =
        block->completion_head.load(std::memory_order_relaxed);
    const uint64_t tail =
        block->completion_tail.load(std::memory_order_acquire);
    if (head - tail >= BLOCK_CQ_CAPACITY) std::abort();
    block->completions[head % BLOCK_CQ_CAPACITY] = {
        .generation = generation,
        .status = status,
        .block = block->id,
    };
    block->completion_head.store(head + 1, std::memory_order_release);
    e->block_completions.fetch_add(1, std::memory_order_relaxed);
    signal_all(&block->completion_word);
}

static BlockCompletion wait_block_completion(Engine *e, BlockDomain *block,
                                             uint64_t generation) {
    uint32_t observed =
        kc_atomic_u32_load_acquire(&block->completion_word.value);
    for (;;) {
        const uint64_t tail =
            block->completion_tail.load(std::memory_order_relaxed);
        const uint64_t head =
            block->completion_head.load(std::memory_order_acquire);
        if (tail < head) {
            const BlockCompletion completion =
                block->completions[tail % BLOCK_CQ_CAPACITY];
            block->completion_tail.store(tail + 1,
                                         std::memory_order_release);
            if (completion.generation != generation ||
                completion.block != block->id) {
                std::abort();
            }
            return completion;
        }
        const int status = kc_port_wait_u32(block->completion_word.wait,
                                            observed, 0);
        observed =
            kc_atomic_u32_load_acquire(&block->completion_word.value);
        if (status != 0 && !e->retire.load(std::memory_order_acquire)) {
            std::abort();
        }
    }
}

// Kcoro calls this once per stable member for each dispatched generation. It
// owns the resident thread, expected-value park, stop, and join; Flashkern owns
// only the lane-uniform numerical program and its model-specific completion.
static void lane_member(void *context, uint32_t lane, uint32_t members,
                        uint64_t generation) {
    Engine *e = static_cast<Engine *>(context);
    if (!e || members != e->lanes_total ||
        generation != e->lane_gen.load(std::memory_order_acquire)) {
        std::abort();
    }
    lane_program(e, lane);
    BlockDomain *block = block_for_lane(e, lane);
    if (!block) std::abort();
    if (lane == block->lane_begin) {
        publish_block_completion(
            e, block, generation,
            e->active_status.load(std::memory_order_acquire));
    }
    if (lane == 0) {
        int status = 0;
        // Drain in reverse block order deliberately. Completion order is
        // not an ownership or publication condition; exact generation is.
        for (uint32_t index = e->block_count; index > 0; --index) {
            const BlockCompletion block_completion =
                wait_block_completion(e, &e->blocks[index - 1], generation);
            if (status == 0 && block_completion.status != 0) {
                status = block_completion.status;
            }
        }
        uint64_t lease = generation;
        if (!e->gang_lease.compare_exchange_strong(
                lease, 0, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            std::abort();
        }
        const KcSubmissionV1 submission = e->active_submission;
        KcCompletionV1 completion{};
        completion.size = sizeof(completion);
        completion.abi_version = KC_COORD_ABI_VERSION;
        completion.ticket = submission.ticket;
        completion.conversation_id = submission.conversation_id;
        completion.epoch = submission.epoch;
        completion.pass_id = submission.ticket.sequence;
        completion.status = status;
        if (completion.status == 0) {
            completion.execution = KC_COORD_EXECUTION_COMPLETED;
            completion.state = KC_COORD_STATE_COMMITTED;
            completion.publication = KC_COORD_PUBLICATION_COMMITTED;
            completion.cause = KC_COORD_CAUSE_SUCCESS;
        } else {
            completion.execution = KC_COORD_EXECUTION_FAILED;
            completion.state = KC_COORD_STATE_NONE;
            completion.publication = KC_COORD_PUBLICATION_NONE;
            completion.cause = KC_COORD_CAUSE_FAULT;
        }
        e->pass_completions.fetch_add(1, std::memory_order_relaxed);
        if (lfm_kernel_bridge_publish_completion(e->bridge, &completion) != 0) {
            // The sole accepted ticket owns a reserved CQ cell. Failure here
            // would otherwise strand the caller forever, so surface the broken
            // executor invariant as a process fault.
            std::abort();
        }
    }
}

static bool ticket_equal(const KcTicketIdV1 &a, const KcTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
}

static PassSlot *slot_from_payload(Engine *e, void *payload) {
    if (!payload) return nullptr;
    const uintptr_t base = reinterpret_cast<uintptr_t>(e->slots.data());
    const uintptr_t address = reinterpret_cast<uintptr_t>(payload);
    if (address < base) return nullptr;
    const uintptr_t offset = address - base;
    if (offset % sizeof(PassSlot) != 0) return nullptr;
    const size_t index = offset / sizeof(PassSlot);
    return index < e->slots.size() ? &e->slots[index] : nullptr;
}

static void publish_rejected(Engine *e, const KcSubmissionV1 &submission, int status) {
    KcCompletionV1 completion{};
    completion.size = sizeof(completion);
    completion.abi_version = KC_COORD_ABI_VERSION;
    completion.ticket = submission.ticket;
    completion.conversation_id = submission.conversation_id;
    completion.epoch = submission.epoch;
    completion.execution = KC_COORD_EXECUTION_NOT_DISPATCHED;
    completion.state = KC_COORD_STATE_NONE;
    completion.publication = KC_COORD_PUBLICATION_NONE;
    completion.cause = KC_COORD_CAUSE_REJECTED;
    completion.status = status;
    if (lfm_kernel_bridge_publish_completion(e->bridge, &completion) != 0) std::abort();
}

// The bridge dispatcher is the sole SQ consumer and CQ consumer. It deliberately
// dispatches one complete pass at a time: SQ depth buys a queued continuation,
// never concurrent access to the lane board. Each descriptor names its immutable
// PassSlot; that slot's scratch bank is mounted only for its exact ticket and is
// returned before another ticket can execute.
static void *bridge_main(void *arg) {
    Engine *e = (Engine *)arg;
    for (;;) {
        KcSubmissionV1 submission{};
        int rc = lfm_kernel_bridge_wait_submission(e->bridge, &submission, 0);
        if (rc == -ECANCELED) return nullptr;
        if (rc != 0) std::abort();

        LfmKernelDescriptorViewV1 descriptor = {
            .size = sizeof(LfmKernelDescriptorViewV1),
            .abi_version = KC_COORD_ABI_VERSION,
        };
        const bool borrowed_descriptor =
            submission.flags == KC_COORD_SUBMISSION_BORROWED_DESCRIPTOR;
        int descriptor_rc = borrowed_descriptor
            ? lfm_kernel_bridge_descriptor_get_borrowed(
                  e->bridge, submission.descriptor, &descriptor)
            : lfm_kernel_bridge_descriptor_get(
                  e->bridge, submission.descriptor, &descriptor);
        PassSlot *slot = descriptor_rc == 0
            ? slot_from_payload(e, descriptor.payload)
            : nullptr;
        const uint64_t slot_owner = slot ? slot_generation(slot) : 0;
        // Acquiring the exact descriptor-named slot is what publishes its plain
        // request fields. Never scan the other slot: its producer may be filling
        // that independent record concurrently.
        bool valid = slot &&
                     slot_state(slot) == PASS_SLOT_SUBMITTED;
        if (valid) {
            valid =
                     request_kind_valid(descriptor.kind) &&
                     submission.command == KC_COORD_COMMAND_RUN_PASS &&
                     submission.pass_budget == 1 &&
                     submission.flags ==
                         (borrowed_descriptor
                              ? KC_COORD_SUBMISSION_BORROWED_DESCRIPTOR
                              : 0) &&
                     submission.ticket.kind == KC_COORD_TICKET_PASS &&
                     submission.epoch != 0 &&
                     descriptor.payload == slot && descriptor.flags == 0 &&
                     slot->engine == e && slot->request == (int)descriptor.kind &&
                     slot->context_id == submission.conversation_id &&
                     ticket_equal(slot->submission.ticket, submission.ticket);
        }
        if (valid) {
            switch (descriptor.kind) {
            case REQ_CONV_LAYER:
            case REQ_ATTN_LAYER:
            case REQ_TOKEN_PASS:
            case REQ_PREFILL:
            case REQ_MIMI_DECODE:
                valid = slot->model &&
                        submission.conversation_id == slot->model->id &&
                        submission.epoch == slot->model->id;
                break;
            case REQ_AUDIO_ENCODE:
                valid = slot->model
                    ? submission.conversation_id == slot->model->id &&
                          submission.epoch == slot->model->id
                    : submission.conversation_id == 0 &&
                          submission.epoch == 1;
                break;
            case REQ_DEPTH_FRAME:
                valid = slot->depth &&
                        submission.conversation_id == slot->depth->id &&
                        submission.epoch == slot->depth->id;
                break;
            default:
                valid = submission.conversation_id == 0 && submission.epoch == 1;
                break;
            }
        }
        if (!valid) {
            publish_rejected(e, submission, -ESTALE);
        } else {
            if (!transition_slot(slot, slot_owner, PASS_SLOT_SUBMITTED,
                                 PASS_SLOT_RUNNING)) {
                publish_rejected(e, submission, -ESTALE);
                valid = false;
            }
        }

        if (valid) {
            const uint64_t previous =
                e->lane_gen.load(std::memory_order_acquire);
            if (previous != 0 && kc_team_wait(e->team, previous, 0) != 0)
                std::abort();
            activate_slot(e, slot);
            e->active_status.store(0, std::memory_order_relaxed);
            e->cur_req = slot->request;
            e->active_submission = submission;
            e->bridge_dispatches.fetch_add(1, std::memory_order_relaxed);
            uint64_t generation = e->lane_gen.load(std::memory_order_relaxed) + 1;
            uint64_t idle = 0;
            if (!e->gang_lease.compare_exchange_strong(
                    idle, generation, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                std::abort();
            }
            e->gang_generations.fetch_add(1, std::memory_order_relaxed);
            e->lane_gen.store(generation, std::memory_order_release);
            e->dispatch_wakes.fetch_add(1, std::memory_order_relaxed);
            if (kc_team_dispatch(e->team, generation) != 0) std::abort();
        }

        KcCompletionV1 completion{};
        rc = lfm_kernel_bridge_wait_completion(e->bridge, &completion, 0);
        if (rc != 0) {
            // Once an SQ record is consumed, completion_head trails
            // submission_tail until this exact record is published. Therefore
            // bridge stop cannot satisfy wait_completion's cancellation
            // predicate here. Never unmount scratch on an error edge while a
            // lane could still hold its pointers.
            std::abort();
        }
        if (valid) deactivate_slot(e, slot);
        if (!slot || !ticket_equal(completion.ticket, submission.ticket) ||
            completion.conversation_id != submission.conversation_id ||
            completion.epoch != submission.epoch) {
            std::abort();
        }

        slot->completion = completion;
        const uint32_t completed_from = valid ? PASS_SLOT_RUNNING
                                              : PASS_SLOT_SUBMITTED;
        if (slot->continuation) {
            if (!transition_slot(slot, slot_owner, completed_from,
                                 PASS_SLOT_COMPLETING)) {
                std::abort();
            }
            PassContinuation continuation = slot->continuation;
            void *context = slot->continuation_context;
            const uint64_t generation = slot_owner;
            /* The CQ settles the old request, but not its slot lease. Reset the
             * payload and hand the exact RESERVED slot to the callback under
             * the same owner generation. No producer can steal it between CQ
             * and a follow-on SQ publication. */
            clear_slot_request(slot);
            slot->lease.store(pass_slot_lease(generation,
                                              PASS_SLOT_RESERVED),
                              std::memory_order_release);
            PassContinuationPermit permit = {
                .engine = e,
                .slot = slot,
                .generation = generation,
                .consumed = false,
            };
            try {
                continuation(&permit, completion, context);
            } catch (...) {
                if (!permit.consumed &&
                    !release_pass_slot(slot, generation)) {
                    std::abort();
                }
                std::abort();
            }
            if (!permit.consumed) {
                if (!release_pass_slot(slot, generation)) std::abort();
            }
            continue;
        }
        if (!transition_slot(slot, slot_owner, completed_from,
                             PASS_SLOT_COMPLETE)) {
            std::abort();
        }
        signal_all(&slot->completion_word);
    }
}

static uint64_t next_sequence(std::atomic<uint64_t> *counter) {
    uint64_t sequence = counter->fetch_add(1, std::memory_order_acq_rel) + 1;
    while (sequence == 0)
        sequence = counter->fetch_add(1, std::memory_order_acq_rel) + 1;
    return sequence;
}

static uint32_t next_generation(std::atomic<uint32_t> *counter) {
    uint32_t generation = counter->fetch_add(1, std::memory_order_acq_rel) + 1;
    while (generation == 0)
        generation = counter->fetch_add(1, std::memory_order_acq_rel) + 1;
    return generation;
}

static int submit_slot(Engine *e, PassSlot *slot, uint64_t generation,
                       int request, uint64_t context_id,
                       PassContinuation continuation,
                       void *continuation_context,
                       const KcDescriptorIdV1 *borrowed_descriptor = nullptr) {
    if (!e || !slot || slot->engine != e ||
        !request_kind_valid(static_cast<uint32_t>(request)) || generation == 0 ||
        slot->exclusive_admission != (borrowed_descriptor != nullptr) ||
        (borrowed_descriptor &&
         e->pass_admission.load(std::memory_order_acquire) !=
             PASS_ADMISSION_EXCLUSIVE) ||
        (borrowed_descriptor &&
         (borrowed_descriptor->slot == UINT32_MAX ||
          borrowed_descriptor->generation == 0)) ||
        slot->lease.load(std::memory_order_acquire) !=
            pass_slot_lease(generation, PASS_SLOT_RESERVED)) {
        return -EINVAL;
    }
    if (!transition_slot(slot, generation, PASS_SLOT_RESERVED,
                         PASS_SLOT_SUBMITTING)) {
        return -ESTALE;
    }

    const uint64_t sequence = next_sequence(&e->submit_sequence);
    const uint32_t ticket_generation = next_generation(&e->ticket_generation);

    LfmKernelDescriptorSpecV1 descriptor_spec = {
        .size = sizeof(LfmKernelDescriptorSpecV1),
        .abi_version = KC_COORD_ABI_VERSION,
        .kind = (uint32_t)request,
        .flags = 0,
        .payload = slot,
        .context = nullptr,
        .release = nullptr,
        .reserved = {0, 0, 0},
    };
    KcDescriptorIdV1 descriptor =
        borrowed_descriptor ? *borrowed_descriptor : KcDescriptorIdV1{};
    KcSubmissionV1 submission{};
    submission.size = sizeof(submission);
    submission.abi_version = KC_COORD_ABI_VERSION;
    submission.ticket.runtime_epoch = e->runtime_epoch;
    submission.ticket.sequence = sequence;
    submission.ticket.generation = ticket_generation;
    submission.ticket.kind = KC_COORD_TICKET_PASS;
    submission.conversation_id = context_id;
    submission.epoch = context_id == 0 ? 1 : context_id;
    submission.descriptor = descriptor;
    submission.command = KC_COORD_COMMAND_RUN_PASS;
    submission.service_class = KC_COORD_SERVICE_INTERACTIVE;
    submission.flags = borrowed_descriptor
        ? KC_COORD_SUBMISSION_BORROWED_DESCRIPTOR
        : 0;
    submission.pass_budget = 1;

    slot->request = request;
    slot->context_id = context_id;
    slot->continuation = continuation;
    slot->continuation_context = continuation_context;

    int rc = 0;
    if (borrowed_descriptor) {
        submission.descriptor = descriptor;
        slot->submission = submission;
        slot->lease.store(pass_slot_lease(generation,
                                          PASS_SLOT_SUBMITTED),
                          std::memory_order_release);
        rc = lfm_kernel_bridge_submit_borrowed(e->bridge, &submission);
        if (rc != 0)
            slot->lease.store(pass_slot_lease(generation,
                                              PASS_SLOT_RESERVED),
                              std::memory_order_release);
    } else {
        // The bridge SQ is SPSC. Native recurrence and compatibility callers
        // share this tiny admission critical section; no model value or scratch
        // is copied while it is held.
        std::lock_guard<std::mutex> submit(e->submission_mutex);
        rc = lfm_kernel_bridge_descriptor_create(e->bridge, &descriptor_spec,
                                                 &descriptor);
        if (rc == 0) {
            submission.descriptor = descriptor;
            slot->submission = submission;
            // This is the publication edge for every plain request field and
            // the final descriptor-bearing submission. The SQ consumer acquires
            // this exact state before reading any of them.
            slot->lease.store(pass_slot_lease(generation,
                                              PASS_SLOT_SUBMITTED),
                              std::memory_order_release);
            rc = lfm_kernel_bridge_submit(e->bridge, &submission);
            if (rc != 0)
                slot->lease.store(pass_slot_lease(generation,
                                                  PASS_SLOT_RESERVED),
                                  std::memory_order_release);
        }
    }
    if (!borrowed_descriptor && descriptor.generation != 0) {
        const int release_rc =
            lfm_kernel_bridge_descriptor_release(e->bridge, descriptor);
        if (release_rc != 0) std::abort();
    }
    if (rc != 0) {
        if (slot_state(slot) == PASS_SLOT_SUBMITTING) {
            slot->lease.store(pass_slot_lease(generation,
                                              PASS_SLOT_RESERVED),
                              std::memory_order_release);
        }
        return rc;
    }
    e->pass_submissions.fetch_add(1, std::memory_order_relaxed);
    if (continuation)
        e->continuation_submissions.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

static int submit_continuation(PassContinuationPermit *permit, int request,
                               uint64_t context_id,
                               PassContinuation continuation,
                               void *continuation_context,
                               const KcDescriptorIdV1 *borrowed_descriptor =
                                   nullptr) {
    if (!permit || permit->consumed || !permit->engine || !permit->slot ||
        permit->slot->engine != permit->engine || permit->generation == 0 ||
        permit->slot->lease.load(std::memory_order_acquire) !=
            pass_slot_lease(permit->generation, PASS_SLOT_RESERVED)) {
        return -ESTALE;
    }
    const int rc = submit_slot(permit->engine, permit->slot,
                               permit->generation, request,
                               context_id, continuation,
                               continuation_context, borrowed_descriptor);
    if (rc == 0) permit->consumed = true;
    return rc;
}

static bool release_continuation(PassContinuationPermit *permit) {
    if (!permit || permit->consumed || !permit->slot) return false;
    if (!release_pass_slot(permit->slot, permit->generation)) return false;
    permit->consumed = true;
    return true;
}

static int wait_submitted_slot(PassSlot *slot, uint64_t generation);

static int submit_pass(Engine *e, PassSlot *slot, int request,
                       uint64_t context_id = 0) {
    const uint64_t generation = slot ? slot_generation(slot) : 0;
    int rc = submit_slot(e, slot, generation, request, context_id, nullptr,
                         nullptr);
    if (rc != 0) return rc;
    rc = wait_submitted_slot(slot, generation);
    if (!release_pass_slot(slot, generation)) std::abort();
    /* The pause is after FREE publication but before the owning PassClaim's
     * destructor. It deterministically exposes the historical ABA window. */
    pause_test_boundary(e, &e->test_claim_return_pause);
    return rc;
}

static int wait_submitted_slot(PassSlot *slot, uint64_t generation) {
    if (!slot || generation == 0) return -EINVAL;
    int rc = 0;

    uint32_t observed =
        kc_atomic_u32_load_acquire(&slot->completion_word.value);
    while (slot->lease.load(std::memory_order_acquire) !=
           pass_slot_lease(generation, PASS_SLOT_COMPLETE)) {
        rc = kc_port_wait_u32(slot->completion_word.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&slot->completion_word.value);
        if (rc != 0 &&
            slot->lease.load(std::memory_order_acquire) !=
                pass_slot_lease(generation, PASS_SLOT_COMPLETE)) {
            // Infinite-deadline waits absorb interrupts. Any remaining wait
            // failure cannot safely return borrowed buffers while this ticket
            // may still be executing.
            std::abort();
        }
    }
    const KcCompletionV1 completion = slot->completion;

    if (!ticket_equal(completion.ticket, slot->submission.ticket) ||
        completion.conversation_id != slot->submission.conversation_id ||
        completion.epoch != slot->submission.epoch) {
        rc = -ESTALE;
    } else {
        rc = completion.status;
    }
    return rc;
}

// Private implementation-backed proof of the completion-continuation contract.
// One exact slot is handed directly from CQ to its callback and resubmitted
// without pass_claimed. Adversarial tests occupy the peer slot and delay an old
// compatibility claim destructor to prove both capacity-2 ownership edges.
struct PrngContinuationChain {
    WaitWord done;
    LfmPrngStateV1 *state = nullptr;
    uint64_t *out = nullptr;
    size_t count = 0;
    size_t next = 0;
    std::atomic<int> status{0};
    std::atomic<bool> finished{false};
};

static void finish_prng_chain(PrngContinuationChain *chain, int status) {
    if (status != 0) {
        int expected = 0;
        chain->status.compare_exchange_strong(
            expected, status, std::memory_order_acq_rel,
            std::memory_order_acquire);
    }
    chain->finished.store(true, std::memory_order_release);
    signal_all(&chain->done);
}

static void continue_prng_chain(PassContinuationPermit *permit,
                                const KcCompletionV1 &completion,
                                void *context) noexcept {
    PrngContinuationChain *chain =
        static_cast<PrngContinuationChain *>(context);
    if (!chain || completion.status != 0) {
        const int status = completion.status != 0 ? completion.status : -EFAULT;
        if (!release_continuation(permit)) std::abort();
        if (chain) finish_prng_chain(chain, status);
        return;
    }
    if (chain->next == chain->count) {
        if (!release_continuation(permit)) std::abort();
        finish_prng_chain(chain, 0);
        return;
    }

    if (!permit || !permit->engine || !permit->slot) {
        finish_prng_chain(chain, -ESTALE);
        return;
    }
    pause_test_boundary(permit->engine,
                        &permit->engine->test_continuation_pause);
    const size_t index = chain->next++;
    permit->slot->prng = {
        .state = chain->state,
        .out = chain->out + index,
        .count = 1,
    };
    const int rc = submit_continuation(permit, REQ_PRNG, 0,
                                       continue_prng_chain, chain);
    if (rc == 0) return;
    if (!release_continuation(permit)) std::abort();
    finish_prng_chain(chain, rc);
}

// The production audio route is deliberately smaller than a graph runtime:
// three trusted coarse nodes and one total immutable outcome table.
enum : uint32_t {
    AUDIO_ROUTE_TOKEN = 0,
    AUDIO_ROUTE_DEPTH = 1,
    AUDIO_ROUTE_MIMI = 2,
    AUDIO_ROUTE_NODE_COUNT = 3,
};
enum : uint32_t {
    AUDIO_ROUTE_SUCCESS = 0,
    AUDIO_ROUTE_FAILURE = 1,
    AUDIO_ROUTE_EOAUDIO = 2,
    AUDIO_ROUTE_STALE = 3,
    AUDIO_ROUTE_OUTCOME_COUNT = 4,
};
enum : uint32_t {
    AUDIO_ROUTE_TERMINAL = AUDIO_ROUTE_NODE_COUNT,
};
enum : uint32_t {
    AUDIO_TOKEN_CODEC = 0,
    AUDIO_TOKEN_END = 1,
    AUDIO_TOKEN_INVALID = 2,
};
enum : uint32_t {
    AUDIO_ROUTE_FREE = 0,
    AUDIO_ROUTE_CLAIMED = 1,
    AUDIO_ROUTE_READY = 2,
    AUDIO_ROUTE_DISPATCHING = 3,
    AUDIO_ROUTE_RUNNING = 4,
    AUDIO_ROUTE_DONE = 5,
};

static constexpr std::array<std::array<uint8_t, AUDIO_ROUTE_OUTCOME_COUNT>,
                            AUDIO_ROUTE_NODE_COUNT>
    AUDIO_ROUTE_TABLE = {{
        {{AUDIO_ROUTE_DEPTH, AUDIO_ROUTE_TERMINAL, AUDIO_ROUTE_TERMINAL,
          AUDIO_ROUTE_TERMINAL}},
        {{AUDIO_ROUTE_MIMI, AUDIO_ROUTE_TERMINAL, AUDIO_ROUTE_TERMINAL,
          AUDIO_ROUTE_TERMINAL}},
        {{AUDIO_ROUTE_TERMINAL, AUDIO_ROUTE_TERMINAL, AUDIO_ROUTE_TERMINAL,
          AUDIO_ROUTE_TERMINAL}},
    }};

static bool audio_route_next(uint32_t node, uint32_t outcome,
                             uint32_t *target) {
    if (!target || node >= AUDIO_ROUTE_NODE_COUNT ||
        outcome >= AUDIO_ROUTE_OUTCOME_COUNT) {
        return false;
    }
    const uint32_t next = AUDIO_ROUTE_TABLE[node][outcome];
    if (next > AUDIO_ROUTE_TERMINAL) return false;
    *target = next;
    return true;
}

static uint32_t audio_token_class(uint32_t token) {
    if (token < LFM_MIMI_CODE_VALUES) return AUDIO_TOKEN_CODEC;
    if (token == LFM_MIMI_CODE_VALUES) return AUDIO_TOKEN_END;
    return AUDIO_TOKEN_INVALID;
}

struct AudioRouteInstance {
    Engine *engine = nullptr;
    WaitWord done;
    std::atomic<uint32_t> state{AUDIO_ROUTE_FREE};
    std::atomic<uint64_t> generation{0};
    uint64_t enqueue_sequence = 0;
    uint32_t service_class = KC_COORD_SERVICE_INTERACTIVE;
    uint32_t node = AUDIO_ROUTE_TOKEN;
    uint64_t depth_id = 0;
    DepthPlan *depth = nullptr;
    DepthReq depth_req{};
    BackbonePlan *model = nullptr;
    uint64_t model_id = 0;
    TokenReq token_req{};
    MimiReq mimi_req{};
    const LfmRouteEpoch *epoch = nullptr;
    uint64_t expected_epoch = 0;
    LfmAudioRouteResult *result = nullptr;
    LfmAudioRouteNotify notify = nullptr;
    void *notify_context = nullptr;
    bool terminal_after_token = false;
    bool decode_mimi = false;
    LfmTokenCommitRecord commit{};
    uint32_t *token_completed = nullptr;
    int status = -EINPROGRESS;
};

struct AudioRoutePool {
    Engine *engine = nullptr;
    std::array<AudioRouteInstance, ROUTE_CAPACITY> routes;
    std::atomic<uint64_t> sequence{0};
};

static int preflight_audio_route_commit(const LfmTokenCommitRecord *commit,
                                        size_t position) {
    if (!commit || !commit->window || !commit->token_committed) return -EINVAL;
    const LfmContextWindowState *window = commit->window;
    if (position != commit->expected_position ||
        window->position != commit->expected_position ||
        window->start != commit->expected_start ||
        window->cursor != commit->expected_cursor ||
        window->rope_base != commit->expected_rope_base) {
        return -ESTALE;
    }
    return lfm_context_window_can_commit(window);
}

static int commit_audio_route_token(AudioRouteInstance *route) {
    if (!route || !route->token_completed || !route->commit.window ||
        !route->commit.token_committed ||
        *route->commit.token_committed != 0) {
        return -EINVAL;
    }
    *route->token_completed = 1;
    LfmContextWindowState *window = route->commit.window;
    if (window->position != route->commit.expected_position ||
        window->start != route->commit.expected_start ||
        window->cursor != route->commit.expected_cursor ||
        window->rope_base != route->commit.expected_rope_base) {
        return -ESTALE;
    }
    const int status = lfm_context_window_commit(window);
    if (status == 0) *route->commit.token_committed = 1;
    return status;
}

static void finish_audio_route(PassContinuationPermit *permit,
                               AudioRouteInstance *route, int status) {
    if (!permit || permit->consumed || !permit->slot || !route) std::abort();
    route->status = status;
    if (route->result) route->result->status = status;
    if (!release_continuation(permit)) std::abort();
    uint32_t running = AUDIO_ROUTE_RUNNING;
    if (!route->state.compare_exchange_strong(
            running, AUDIO_ROUTE_DONE, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        std::abort();
    }
    signal_all(&route->done);
    signal_all(&route->engine->route_word);
    if (route->notify) route->notify(route->notify_context);
}

static void continue_audio_route(PassContinuationPermit *permit,
                                 const KcCompletionV1 &completion,
                                 void *context) noexcept {
    AudioRouteInstance *route = static_cast<AudioRouteInstance *>(context);
    if (!permit || !route) std::abort();
    uint32_t outcome = completion.status == 0 ? AUDIO_ROUTE_SUCCESS
                                              : AUDIO_ROUTE_FAILURE;
    if (completion.status == 0 && route->node == AUDIO_ROUTE_TOKEN) {
        const int commit_status = commit_audio_route_token(route);
        if (commit_status != 0) {
            finish_audio_route(permit, route, commit_status);
            return;
        }
        if (route->terminal_after_token) {
            finish_audio_route(permit, route, 0);
            return;
        }
        if (route->decode_mimi &&
            route->epoch->load(std::memory_order_acquire) !=
                route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        }
    } else if (completion.status == 0 && route->node == AUDIO_ROUTE_DEPTH &&
               route->decode_mimi) {
        route->result->depth_completed = 1;
        const uint32_t first_class =
            audio_token_class(route->result->codes[0]);
        if (first_class == AUDIO_TOKEN_END) {
            route->result->eoaudio = 1;
            outcome = AUDIO_ROUTE_EOAUDIO;
        } else if (first_class == AUDIO_TOKEN_INVALID) {
            finish_audio_route(permit, route, -ERANGE);
            return;
        } else if (route->epoch->load(std::memory_order_acquire) !=
                   route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        } else {
            for (size_t index = 0; index < LFM_MIMI_CODEBOOKS; ++index) {
                if (audio_token_class(route->result->codes[index]) !=
                    AUDIO_TOKEN_CODEC) {
                    finish_audio_route(permit, route, -ERANGE);
                    return;
                }
            }
        }
    } else if (completion.status == 0 && route->node == AUDIO_ROUTE_MIMI &&
               route->decode_mimi) {
        route->result->mimi_completed = 1;
        if (route->epoch->load(std::memory_order_acquire) !=
            route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        }
    }
    if (completion.status == 0 &&
        route->engine->route_retire.load(std::memory_order_acquire)) {
        finish_audio_route(permit, route, -ECANCELED);
        return;
    }
    uint32_t target = AUDIO_ROUTE_TERMINAL;
    if (!audio_route_next(route->node, outcome, &target)) {
        finish_audio_route(permit, route, -EPROTO);
        return;
    }
    if (completion.status != 0) {
        finish_audio_route(permit, route, completion.status);
        return;
    }
    if (outcome == AUDIO_ROUTE_STALE) {
        finish_audio_route(permit, route, -ESTALE);
        return;
    }
    if (outcome == AUDIO_ROUTE_EOAUDIO) {
        finish_audio_route(permit, route, 0);
        return;
    }
    /* The retained two-node compatibility seam terminates after Depth. */
    if (!route->decode_mimi && route->node == AUDIO_ROUTE_DEPTH) {
        finish_audio_route(permit, route, 0);
        return;
    }
    if (target == AUDIO_ROUTE_TERMINAL) {
        finish_audio_route(permit, route, 0);
        return;
    }
    if (route->node == AUDIO_ROUTE_TOKEN && target == AUDIO_ROUTE_DEPTH &&
        route->depth && route->depth_id != 0) {
        route->node = AUDIO_ROUTE_DEPTH;
    } else if (route->node == AUDIO_ROUTE_DEPTH && target == AUDIO_ROUTE_MIMI &&
               route->decode_mimi && route->model && route->model_id != 0) {
        route->node = AUDIO_ROUTE_MIMI;
    } else {
        finish_audio_route(permit, route, -EPROTO);
        return;
    }
    if (!release_continuation(permit)) std::abort();
    route->enqueue_sequence =
        route->engine->route_pool->sequence.fetch_add(
            1, std::memory_order_acq_rel) + 1;
    uint32_t running = AUDIO_ROUTE_RUNNING;
    if (!route->state.compare_exchange_strong(
            running, AUDIO_ROUTE_READY, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        std::abort();
    }
    signal_all(&route->engine->route_word);
}

static int wait_audio_route(AudioRouteInstance *route, uint64_t generation) {
    if (!route || generation == 0) return -EINVAL;
    uint32_t observed =
        kc_atomic_u32_load_acquire(&route->done.value);
    while (route->state.load(std::memory_order_acquire) != AUDIO_ROUTE_DONE) {
        const int status =
            kc_port_wait_u32(route->done.wait, observed, 0);
        observed =
            kc_atomic_u32_load_acquire(&route->done.value);
        if (status != 0 &&
            route->state.load(std::memory_order_acquire) != AUDIO_ROUTE_DONE) {
            std::abort();
        }
    }
    if (route->generation.load(std::memory_order_acquire) != generation) {
        return -ESTALE;
    }
    return route->status;
}

static AudioRouteInstance *claim_audio_route(Engine *engine,
                                             uint64_t *generation) {
    if (!engine || !generation || !enter_pass_admission(engine)) return nullptr;
    for (AudioRouteInstance &route : engine->route_pool->routes) {
        uint32_t free = AUDIO_ROUTE_FREE;
        if (!route.state.compare_exchange_strong(
                free, AUDIO_ROUTE_CLAIMED, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            continue;
        }
        uint64_t next = route.generation.fetch_add(
                            1, std::memory_order_acq_rel) + 1;
        if (next == 0) {
            next = route.generation.fetch_add(
                       1, std::memory_order_acq_rel) + 1;
        }
        *generation = next;
        return &route;
    }
    leave_pass_admission(engine);
    return nullptr;
}

static void release_audio_route(AudioRouteInstance *route,
                                uint64_t generation) {
    if (!route || !route->engine || generation == 0 ||
        route->generation.load(std::memory_order_acquire) != generation) {
        std::abort();
    }
    uint32_t done = AUDIO_ROUTE_DONE;
    if (!route->state.compare_exchange_strong(
            done, AUDIO_ROUTE_FREE, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        std::abort();
    }
    leave_pass_admission(route->engine);
    signal_all(&route->engine->route_word);
}

class AudioRouteLease {
  public:
    explicit AudioRouteLease(Engine *engine) {
        route_ = claim_audio_route(engine, &generation_);
    }

    ~AudioRouteLease() {
        if (!route_) return;
        uint32_t state = route_->state.load(std::memory_order_acquire);
        if (state == AUDIO_ROUTE_CLAIMED) {
            route_->status = -ECANCELED;
            route_->state.store(AUDIO_ROUTE_DONE, std::memory_order_release);
            state = AUDIO_ROUTE_DONE;
        }
        if (state != AUDIO_ROUTE_DONE) std::abort();
        release_audio_route(route_, generation_);
    }

    explicit operator bool() const { return route_ != nullptr; }
    AudioRouteInstance *route() const { return route_; }
    uint64_t generation() const { return generation_; }
    void detach() { route_ = nullptr; }
    AudioRouteLease(const AudioRouteLease &) = delete;
    AudioRouteLease &operator=(const AudioRouteLease &) = delete;

  private:
    AudioRouteInstance *route_ = nullptr;
    uint64_t generation_ = 0;
};

static void settle_audio_route(AudioRouteInstance *route, uint32_t from,
                               int status) {
    route->status = status;
    if (route->result) route->result->status = status;
    if (!route->state.compare_exchange_strong(
            from, AUDIO_ROUTE_DONE, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        std::abort();
    }
    signal_all(&route->done);
    signal_all(&route->engine->route_word);
    if (route->notify) route->notify(route->notify_context);
}

static uint64_t audio_route_age(uint64_t snapshot, uint64_t enqueued) {
    return snapshot >= enqueued ? snapshot - enqueued : 0;
}

static uint32_t audio_route_service(uint64_t snapshot, uint64_t enqueued,
                                    uint32_t service) {
    return audio_route_age(snapshot, enqueued) >= ROUTE_AGE_PROMOTION
        ? KC_COORD_SERVICE_DEADLINE
        : service;
}

static AudioRouteInstance *select_audio_route(AudioRoutePool *pool) {
    AudioRouteInstance *best = nullptr;
    uint32_t best_class = UINT32_MAX;
    uint64_t best_sequence = UINT64_MAX;
    const uint64_t now = pool->sequence.load(std::memory_order_acquire);
    for (AudioRouteInstance &route : pool->routes) {
        if (route.state.load(std::memory_order_acquire) != AUDIO_ROUTE_READY)
            continue;
        /* A producer may enqueue after this scan latched `now`. That route is
         * newer than the snapshot, not infinitely old; age zero prevents a
         * fresh continuation from jumping genuinely-starved work. */
        const uint32_t service = audio_route_service(
            now, route.enqueue_sequence, route.service_class);
        if (!best || service < best_class ||
            (service == best_class &&
             route.enqueue_sequence < best_sequence)) {
            best = &route;
            best_class = service;
            best_sequence = route.enqueue_sequence;
        }
    }
    if (!best) return nullptr;
    uint32_t ready = AUDIO_ROUTE_READY;
    if (!best->state.compare_exchange_strong(
            ready, AUDIO_ROUTE_DISPATCHING, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return nullptr;
    }
    return best;
}

static int mount_audio_route(PassSlot *slot, AudioRouteInstance *route,
                             int *request, uint64_t *context) {
    if (!slot || !route || !request || !context) return -EINVAL;
    if (route->node == AUDIO_ROUTE_TOKEN) {
        slot->model = route->model;
        slot->tok = route->token_req;
        *request = REQ_TOKEN_PASS;
        *context = route->model_id;
        return 0;
    }
    if (route->node == AUDIO_ROUTE_DEPTH) {
        slot->depth = route->depth;
        slot->depth_req = route->depth_req;
        *request = REQ_DEPTH_FRAME;
        *context = route->depth_id;
        return 0;
    }
    if (route->node == AUDIO_ROUTE_MIMI && route->decode_mimi) {
        slot->model = route->model;
        slot->mimi = route->mimi_req;
        *request = REQ_MIMI_DECODE;
        *context = route->model_id;
        return 0;
    }
    return -EPROTO;
}

static void *audio_route_main(void *context) {
    Engine *engine = static_cast<Engine *>(context);
    AudioRoutePool *pool = engine->route_pool;
    uint32_t observed =
        kc_atomic_u32_load_acquire(&engine->route_word.value);
    for (;;) {
        if (engine->route_retire.load(std::memory_order_acquire)) {
            for (AudioRouteInstance &route : pool->routes) {
                uint32_t ready = AUDIO_ROUTE_READY;
                if (route.state.compare_exchange_strong(
                        ready, AUDIO_ROUTE_DONE, std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    route.status = -ECANCELED;
                    if (route.result) route.result->status = -ECANCELED;
                    signal_all(&route.done);
                    if (route.notify) route.notify(route.notify_context);
                }
            }
            return nullptr;
        }

        AudioRouteInstance *route = select_audio_route(pool);
        if (!route) {
            const int status = kc_port_wait_u32(
                engine->route_word.wait, observed, 0);
            observed = kc_atomic_u32_load_acquire(
                &engine->route_word.value);
            if (status != 0 &&
                !engine->route_retire.load(std::memory_order_acquire)) {
                std::abort();
            }
            continue;
        }

        PassSlot *slot = reserve_pass_slot(engine);
        if (!slot) {
            engine->route_parks.fetch_add(1, std::memory_order_relaxed);
            uint32_t dispatching = AUDIO_ROUTE_DISPATCHING;
            if (!route->state.compare_exchange_strong(
                    dispatching, AUDIO_ROUTE_READY,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                std::abort();
            }
            const int status = kc_port_wait_u32(
                engine->route_word.wait, observed, 0);
            observed = kc_atomic_u32_load_acquire(
                &engine->route_word.value);
            if (status != 0 &&
                !engine->route_retire.load(std::memory_order_acquire)) {
                std::abort();
            }
            continue;
        }

        int request = REQ_NONE;
        uint64_t request_context = 0;
        int status = mount_audio_route(slot, route, &request,
                                       &request_context);
        if (status != 0) {
            if (!release_pass_slot(slot, slot_generation(slot))) std::abort();
            settle_audio_route(route, AUDIO_ROUTE_DISPATCHING, status);
            continue;
        }
        uint32_t dispatching = AUDIO_ROUTE_DISPATCHING;
        if (!route->state.compare_exchange_strong(
                dispatching, AUDIO_ROUTE_RUNNING, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            std::abort();
        }
        const uint64_t slot_owner = slot_generation(slot);
        status = submit_slot(engine, slot, slot_owner, request,
                             request_context, continue_audio_route, route);
        if (status != 0) {
            if (!release_pass_slot(slot, slot_owner)) std::abort();
            settle_audio_route(route, AUDIO_ROUTE_RUNNING, status);
        } else {
            engine->route_dispatches.fetch_add(1,
                                               std::memory_order_relaxed);
        }
    }
}

} // namespace

// ---- the C ABI ------------------------------------------------------------------------
extern "C" {

void lfm_engine_free(void *ep);

// `workers` is the total fixed lane count. Every logical lane owns one pthread for
// the engine lifetime; one mechanical bridge dispatcher owns SQ consumption.
void *lfm_engine_new(int workers) {
    if (workers < 1 || workers > MAX_WORKERS) return nullptr;
    Engine *e = new (std::nothrow) Engine();
    if (!e) return nullptr;
    e->runtime_epoch = next_engine_epoch.fetch_add(1, std::memory_order_acq_rel);
    if (e->runtime_epoch == 0)
        e->runtime_epoch = next_engine_epoch.fetch_add(1, std::memory_order_acq_rel);
    e->lanes_total = (uint32_t)workers;
    e->n_workers = workers;
    e->block_count = workers == (int)GRID_LANES ? 2u : 1u;
    for (uint32_t index = 0; index < e->block_count; ++index) {
        BlockDomain &block = e->blocks[index];
        block.id = index;
        block.lane_begin = e->block_count == 2 ? index * BLOCK_LANES : 0;
        block.lane_count =
            e->block_count == 2 ? BLOCK_LANES : (uint32_t)workers;
        if (!kc_atomic_u32_is_lock_free(&block.completion_word.value) ||
            kc_port_wait_u32_prepare(&block.completion_word.value,
                                     &block.completion_word.wait) != 0) {
            lfm_engine_free(e);
            return nullptr;
        }
        e->block_waits_prepared++;
    }
    e->route_pool = new (std::nothrow) AudioRoutePool();
    if (!e->route_pool) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->route_pool->engine = e;
    for (AudioRouteInstance &route : e->route_pool->routes) {
        route.engine = e;
        if (!kc_atomic_u32_is_lock_free(&route.done.value) ||
            kc_port_wait_u32_prepare(&route.done.value,
                                     &route.done.wait) != 0) {
            lfm_engine_free(e);
            return nullptr;
        }
        e->route_done_waits_prepared++;
    }
    for (size_t index = 0; index < e->slots.size(); ++index) {
        PassSlot &slot = e->slots[index];
        slot.engine = e;
        slot.index = (uint32_t)index;
        if (!kc_atomic_u32_is_lock_free(&slot.completion_word.value) ||
            kc_port_wait_u32_prepare(&slot.completion_word.value,
                                     &slot.completion_word.wait) != 0) {
            lfm_engine_free(e);
            return nullptr;
        }
        e->slot_waits_prepared++;
        if (!kc_atomic_u32_is_lock_free(&slot.audio_word.value) ||
            kc_port_wait_u32_prepare(&slot.audio_word.value,
                                     &slot.audio_word.wait) != 0) {
            lfm_engine_free(e);
            return nullptr;
        }
        e->audio_waits_prepared++;
    }
    if (!kc_atomic_u32_is_lock_free(&e->dispatch_word.value) ||
        kc_port_wait_u32_prepare(&e->dispatch_word.value, &e->dispatch_word.wait) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->wait_words_prepared++;
    if (!kc_atomic_u32_is_lock_free(&e->route_word.value) ||
        kc_port_wait_u32_prepare(&e->route_word.value,
                                 &e->route_word.wait) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->route_wait_prepared = 1;
    if (kc_collective_create(static_cast<uint32_t>(workers),
                             &e->collective) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }

    LfmKernelBridgeConfigV1 bridge_config = {
        .size = sizeof(LfmKernelBridgeConfigV1),
        .abi_version = KC_COORD_ABI_VERSION,
        .capacity = (uint32_t)PASS_CAPACITY,
        .descriptor_capacity = 8,
    };
    if (lfm_kernel_bridge_create(&bridge_config, &e->bridge) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    if (pthread_create(&e->bridge_worker, nullptr, bridge_main, e) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->bridge_started = 1;
    if (pthread_create(&e->route_worker, nullptr, audio_route_main, e) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->route_started = 1;
    const kc_team_config team_config = {
        .size = sizeof(kc_team_config),
        .abi_version = 1,
        .member_count = static_cast<uint32_t>(workers),
        .reserved = 0,
        .member = lane_member,
        .context = e,
    };
    if (kc_team_create(&team_config, &e->team) != 0 ||
        kc_team_start(e->team) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    return e;
}

void lfm_engine_request_stop(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    e->route_retire.store(true, std::memory_order_release);
    if (e->route_wait_prepared) signal_all(&e->route_word);
    if (e->bridge) lfm_kernel_bridge_request_stop(e->bridge);
    const bool wake_test =
        e->test_lane_pause.exchange(0, std::memory_order_acq_rel) != 0 ||
        e->test_claim_return_pause.exchange(0, std::memory_order_acq_rel) != 0 ||
        e->test_continuation_pause.exchange(0,
                                            std::memory_order_acq_rel) != 0 ||
        e->test_live_target.exchange(0, std::memory_order_acq_rel) != 0;
    if (wake_test)
        signal_all(&e->dispatch_word);
}

int lfm_internal_engine_arm_lane_pause_for_test(void *ep) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e) return -EINVAL;
    uint32_t idle = 0;
    return e->test_lane_pause.compare_exchange_strong(
               idle, 1, std::memory_order_acq_rel,
               std::memory_order_acquire)
        ? 0
        : -EBUSY;
}

int lfm_internal_engine_wait_lane_pause_for_test(void *ep) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e) return -EINVAL;
    for (;;) {
        const uint32_t state =
            e->test_lane_pause.load(std::memory_order_acquire);
        if (state == 2) return 0;
        if (state == 0) return -ECANCELED;
        uint32_t observed =
            kc_atomic_u32_load_acquire(&e->dispatch_word.value);
        if (e->test_lane_pause.load(std::memory_order_acquire) != 1) continue;
        const int rc = kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
        if (rc != 0 &&
            e->test_lane_pause.load(std::memory_order_acquire) == 1) {
            return rc;
        }
    }
}

int lfm_internal_engine_pause_boundary_for_test(void *ep, uint32_t kind,
                                                uint32_t action) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || (kind != 1 && kind != 2) || action < 1 || action > 3) {
        return -EINVAL;
    }
    std::atomic<uint32_t> *state = kind == 1
        ? &e->test_claim_return_pause
        : &e->test_continuation_pause;
    if (action == 1) {
        uint32_t idle = 0;
        return state->compare_exchange_strong(
                   idle, 1, std::memory_order_acq_rel,
                   std::memory_order_acquire)
            ? 0
            : -EBUSY;
    }
    if (action == 3) {
        uint32_t parked = 2;
        if (!state->compare_exchange_strong(
                parked, 3, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return parked == 0 ? -ECANCELED : -EBUSY;
        }
        signal_all(&e->dispatch_word);
        return 0;
    }

    uint32_t observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
    for (;;) {
        const uint32_t value = state->load(std::memory_order_acquire);
        if (value == 2) return 0;
        if (value == 0 || value == 3) return -ECANCELED;
        const int rc = kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
        if (rc != 0 && state->load(std::memory_order_acquire) == 1) return rc;
    }
}

int lfm_internal_engine_wait_pass_slots_for_test(void *ep, uint32_t live) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || live == 0 || live > PASS_CAPACITY) return -EINVAL;
    uint32_t idle = 0;
    if (!e->test_live_target.compare_exchange_strong(
            idle, live, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return -EBUSY;
    }
    uint32_t observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
    while (e->pass_slots_live.load(std::memory_order_acquire) != live) {
        const int rc = kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
        if (rc != 0 &&
            e->pass_slots_live.load(std::memory_order_acquire) != live) {
            e->test_live_target.store(0, std::memory_order_release);
            return rc;
        }
    }
    e->test_live_target.store(0, std::memory_order_release);
    return 0;
}

// Private implementation-backed protocol probes. Selector membership is
// queried without dispatch: a valid request also requires a fully populated
// typed payload, so submitting an empty test slot would be an unsafe probe.
int lfm_internal_engine_request_kind_valid_for_test(uint32_t kind) {
    return request_kind_valid(kind) ? 1 : 0;
}

int lfm_internal_engine_wait_word_layout_for_test(void *ep) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e) return -EINVAL;
    constexpr size_t MAX_WAIT_WORD_COUNT =
        2 + PASS_CAPACITY * 2 + ROUTE_CAPACITY + BLOCK_DOMAIN_COUNT;
    std::array<const WaitWord *, MAX_WAIT_WORD_COUNT> words{};
    words[0] = &e->dispatch_word;
    words[1] = &e->route_word;
    for (size_t slot = 0; slot < PASS_CAPACITY; ++slot) {
        words[2 + slot * 2] = &e->slots[slot].completion_word;
        words[3 + slot * 2] = &e->slots[slot].audio_word;
    }
    for (size_t route = 0; route < ROUTE_CAPACITY; ++route) {
        words[2 + PASS_CAPACITY * 2 + route] =
            &e->route_pool->routes[route].done;
    }
    const size_t block_base = 2 + PASS_CAPACITY * 2 + ROUTE_CAPACITY;
    for (size_t block = 0; block < e->block_count; ++block) {
        words[block_base + block] = &e->blocks[block].completion_word;
    }
    const size_t word_count = block_base + e->block_count;
    for (size_t left = 0; left < word_count; ++left) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(words[left]);
        if (address % ENGINE_CACHELINE != 0) return -EFAULT;
        for (size_t right = left + 1; right < word_count; ++right) {
            const uintptr_t peer = reinterpret_cast<uintptr_t>(words[right]);
            if (address / ENGINE_CACHELINE == peer / ENGINE_CACHELINE)
                return -EFAULT;
        }
    }
    return 0;
}

int lfm_internal_engine_grid_snapshot_for_test(
    void *ep, uint32_t *blocks, uint64_t *completions,
    uint64_t *generations, uint64_t *lease) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !blocks || !completions || !generations || !lease) {
        return -EINVAL;
    }
    *blocks = e->block_count;
    *completions =
        e->block_completions.load(std::memory_order_acquire);
    *generations = e->gang_generations.load(std::memory_order_acquire);
    *lease = e->gang_lease.load(std::memory_order_acquire);
    return 0;
}

int lfm_internal_engine_audio_route_edge_for_test(uint32_t node,
                                                  uint32_t outcome,
                                                  uint32_t *target) {
    return audio_route_next(node, outcome, target) ? 0 : -EINVAL;
}

int lfm_internal_engine_audio_token_class_for_test(uint32_t token) {
    return (int)audio_token_class(token);
}

uint32_t lfm_internal_engine_audio_route_service_for_test(
    uint64_t snapshot, uint64_t enqueued, uint32_t service) {
    return audio_route_service(snapshot, enqueued, service);
}

int lfm_internal_engine_fail_audio_route_depth_for_test(void *ep, int status) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || status >= 0) return -EINVAL;
    int idle = 0;
    return e->test_audio_route_depth_status.compare_exchange_strong(
               idle, status, std::memory_order_acq_rel,
               std::memory_order_acquire)
        ? 0
        : -EBUSY;
}

int lfm_internal_engine_fail_audio_route_mimi_for_test(void *ep, int status) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || status >= 0) return -EINVAL;
    int idle = 0;
    return e->test_audio_route_mimi_status.compare_exchange_strong(
               idle, status, std::memory_order_acq_rel,
               std::memory_order_acquire)
        ? 0
        : -EBUSY;
}

void *lfm_internal_audio_route_epoch_new_for_test(uint64_t value) {
    if (value == 0) return nullptr;
    LfmRouteEpoch *epoch = new (std::nothrow) LfmRouteEpoch();
    if (!epoch) return nullptr;
    epoch->store(value, std::memory_order_release);
    return epoch;
}

void lfm_internal_audio_route_epoch_free_for_test(void *opaque) {
    delete static_cast<LfmRouteEpoch *>(opaque);
}

uint32_t lfm_engine_lanes(void *ep) {
    Engine *e = (Engine *)ep;
    return e ? e->lanes_total : 0;
}

uint64_t lfm_engine_audio_encode_passes(const void *ep) {
    const Engine *e = static_cast<const Engine *>(ep);
    return e ? e->audio_encode_passes.load(std::memory_order_relaxed) : 0;
}

int lfm_engine_conformer_gemm_team(
    void *ep, const uint16_t *activation, size_t activation_count,
    const void *weight_bytes, size_t weight_count, float *out,
    size_t out_count, size_t rows, size_t columns, size_t inner) {
    Engine *e = static_cast<Engine *>(ep);
    size_t activation_need = 0, weight_need = 0, output_need = 0;
    if (!e || !activation || !weight_bytes || !out || rows == 0 ||
        columns == 0 || inner == 0 || rows > INT_MAX || columns > INT_MAX ||
        inner > INT_MAX ||
        !checked_size_product(rows, inner, &activation_need) ||
        !checked_size_product(columns, inner, &weight_need) ||
        !checked_size_product(rows, columns, &output_need) ||
        activation_count != activation_need || weight_count != weight_need ||
        out_count != output_need) {
        return -EINVAL;
    }
    PassSlot *slot = e->active_slot;
    uint32_t member = UINT32_MAX;
    if (!slot || slot->request != REQ_AUDIO_ENCODE ||
        slot->audio.done.load(std::memory_order_acquire) ||
        kc_team_current_member(e->team, &member) != 0 || member != 0) {
        return -EBUSY;
    }

    slot->gemm = {
        .a = activation,
        .rhs = weight_bytes,
        .out = out,
        .m = rows,
        .n = columns,
        .k = inner,
        .rhs_layout = LFM_GEMM_RHS_NK,
        .direct = true,
    };
    e->gemm = slot->gemm;
    slot->audio.gemm_generation.fetch_add(1, std::memory_order_release);
    signal_all(&slot->audio_word);
    run_gemm(e, 0);
    lane_fence(e, 0, [] {});
    return 0;
}

int lfm_engine_audio_encode(void *ep, uint64_t model_id,
                            const LfmAudioEncodePassV1 *pass) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !pass || pass->size < sizeof(*pass) ||
        pass->abi_version != LFM_AUDIO_PASS_ABI || !pass->resampler ||
        !pass->resampler_workspace || !pass->frontend ||
        !pass->frontend_workspace || !pass->conformer ||
        !pass->conformer_workspace || !pass->pcm || pass->sample_count == 0 ||
        (!pass->resampled && pass->resampled_capacity != 0) || !pass->mel ||
        pass->mel_capacity == 0 || !pass->adapted ||
        pass->adapted_capacity == 0 || !pass->out_adapted_values) {
        return -EINVAL;
    }

    PassClaim claim(e);
    if (!claim) return -EBUSY;
    BackbonePlan *model = model_id == 0 ? nullptr : find_model(e, model_id);
    if (model_id != 0 && !model) return -ESTALE;
    if (model && lfm_conformer_out_width(pass->conformer) != model->h) {
        return -ESTALE;
    }

    PassSlot *slot = claim.slot();
    *pass->out_adapted_values = 0;
    slot->model = model;
    slot->audio.pass = *pass;
    slot->audio.start_gemm_generation =
        slot->audio.gemm_generation.load(std::memory_order_relaxed);
    slot->audio.done.store(false, std::memory_order_relaxed);
    return submit_pass(e, slot, REQ_AUDIO_ENCODE, model_id);
}

int lfm_engine_snapshot(void *ep, LfmEngineSnapshotV1 *out) {
    Engine *e = (Engine *)ep;
    if (!e || !out || out->size < sizeof(*out) || out->abi_version != 1) return -EINVAL;
    LfmKernelDescriptorSnapshotV1 descriptors = {
        .size = sizeof(LfmKernelDescriptorSnapshotV1),
        .abi_version = KC_COORD_ABI_VERSION,
    };
    if (lfm_kernel_bridge_descriptor_snapshot(e->bridge, &descriptors) != 0) return -EFAULT;
    LfmKernelBridgeSnapshotV1 bridge = {
        .size = sizeof(LfmKernelBridgeSnapshotV1),
        .abi_version = KC_COORD_ABI_VERSION,
    };
    if (lfm_kernel_bridge_snapshot(e->bridge, &bridge) != 0) return -EFAULT;
    uint32_t routes_live = 0;
    uint32_t routes_ready = 0;
    for (const AudioRouteInstance &route : e->route_pool->routes) {
        const uint32_t state = route.state.load(std::memory_order_acquire);
        if (state != AUDIO_ROUTE_FREE) routes_live++;
        if (state == AUDIO_ROUTE_READY) routes_ready++;
    }
    kc_collective_snapshot collective = {
        .size = sizeof(kc_collective_snapshot),
    };
    if (kc_collective_snapshot_get(e->collective, &collective) != 0)
        return -EFAULT;
    *out = {
        .size = sizeof(*out),
        .abi_version = 1,
        .pass_submissions = e->pass_submissions.load(std::memory_order_relaxed),
        .pass_completions = e->pass_completions.load(std::memory_order_relaxed),
        .bridge_dispatches = e->bridge_dispatches.load(std::memory_order_relaxed),
        .dispatch_wakes = e->dispatch_wakes.load(std::memory_order_relaxed),
        .fence_wake_calls = collective.wake_calls,
        .fence_wakes = collective.wakes,
        .fence_generations = collective.generation,
        .descriptor_acquires = descriptors.acquired,
        .descriptor_retains = descriptors.retained,
        .descriptor_releases = descriptors.released,
        .descriptor_callbacks = descriptors.callbacks,
        .descriptor_capacity = descriptors.capacity,
        .descriptors_live = descriptors.live,
        .max_descriptor_generation = descriptors.max_generation,
        .attention_qkv_capacity =
            e->attention_qkv_capacity.load(std::memory_order_relaxed),
        .attention_y_capacity =
            e->attention_y_capacity.load(std::memory_order_relaxed),
        .attention_score_capacity =
            e->attention_score_capacity.load(std::memory_order_relaxed),
        .pass_claimed = e->pass_claimed.load(std::memory_order_acquire) ? 1u : 0u,
        .bridge_capacity = bridge.capacity,
        .pass_slot_capacity = (uint32_t)e->slots.size(),
        .pass_slots_live = e->pass_slots_live.load(std::memory_order_acquire),
        .max_pass_slots_live =
            e->max_pass_slots_live.load(std::memory_order_acquire),
        .continuation_submissions =
            e->continuation_submissions.load(std::memory_order_relaxed),
        .route_capacity = (uint32_t)ROUTE_CAPACITY,
        .routes_live = routes_live,
        .routes_ready = routes_ready,
        .reserved0 = 0,
        .route_dispatches =
            e->route_dispatches.load(std::memory_order_relaxed),
        .route_parks = e->route_parks.load(std::memory_order_relaxed),
    };
    return 0;
}

void lfm_engine_free(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    e->route_retire.store(true, std::memory_order_release);
    if (e->route_wait_prepared) signal_all(&e->route_word);
    if (e->bridge) lfm_kernel_bridge_request_stop(e->bridge);
    if (e->route_started > 0) pthread_join(e->route_worker, nullptr);
    if (e->bridge_started > 0) pthread_join(e->bridge_worker, nullptr);
    e->retire.store(true, std::memory_order_release);
    if (e->wait_words_prepared > 0) signal_all(&e->dispatch_word);
    if (e->team) {
        kc_team_request_stop(e->team);
        if (kc_team_join(e->team) != 0 || kc_team_destroy(e->team) != 0)
            std::abort();
        e->team = nullptr;
    }
    if (e->bridge && lfm_kernel_bridge_destroy(e->bridge) != 0) std::abort();
    if (e->route_pool) {
        for (int index = 0; index < e->route_done_waits_prepared; ++index) {
            kc_port_wait_u32_release(
                e->route_pool->routes[(size_t)index].done.wait);
        }
    }
    for (int index = 0; index < e->audio_waits_prepared; ++index) {
        kc_port_wait_u32_release(e->slots[(size_t)index].audio_word.wait);
    }
    for (int index = 0; index < e->slot_waits_prepared; ++index) {
        kc_port_wait_u32_release(e->slots[(size_t)index].completion_word.wait);
    }
    for (int index = 0; index < e->block_waits_prepared; ++index) {
        kc_port_wait_u32_release(
            e->blocks[(size_t)index].completion_word.wait);
    }
    if (e->route_wait_prepared) kc_port_wait_u32_release(e->route_word.wait);
    if (e->wait_words_prepared > 0) kc_port_wait_u32_release(e->dispatch_word.wait);
    kc_collective_destroy(e->collective);
    delete e->route_pool;
    delete e;
}

// One fused-MLP decode block, entirely native: request slot → doorbell → park.
// Blocking; single pass in flight (decode is sequential). Returns 0 on success.
int lfm_engine_mlp(void *ep, const uint16_t *x, const uint16_t *norm_w,
                   const uint16_t *w1, const uint16_t *w3, const uint16_t *w2,
                   uint16_t *out, size_t h, size_t i, float eps, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || !x || !norm_w || !w1 || !w3 || !w2 || !out || h == 0 || i == 0 ||
        !logical_lane_count_valid(lanes))
        return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    size_t tiles = lanes;
    size_t cap = h < i ? h : i;
    if (tiles > cap) tiles = cap;

    // Grow the persistent scratch outside execution (no allocation once warm — the
    // first pass at a given shape sizes it, every later pass is allocation-free).
    // Allocation failure must NOT throw across the extern "C" boundary: report it
    // and let the Rust rim take the bit-identical threadgroup path.
    // GROW-ONLY: a ctx build (lfm_ctx_build) sizes these planes for the resident
    // model; a legacy per-call MLP with smaller dims must never shrink them — the
    // next conv/attention layer pass would write past the shrunken planes.
    try {
        auto grow_f = [](std::vector<float> &v, size_t n) {
            if (v.size() < n) v.resize(n);
        };
        auto grow_u = [](std::vector<uint16_t> &v, size_t n) {
            if (v.size() < n) v.resize(n);
        };
        grow_f(slot->scratch.sc_partials, tiles);
        grow_u(slot->scratch.sc_xn, h);
        grow_f(slot->scratch.sc_gu, 2 * i);
        grow_u(slot->scratch.sc_t, i);
    } catch (const std::bad_alloc &) {
        return -2;
    }

    slot->mlp = {
        .x = x,
        .norm_w = activation_bytes(norm_w),
        .w1 = activation_bytes(w1),
        .w3 = activation_bytes(w3),
        .w2 = activation_bytes(w2),
        .out = out,
        .h = h,
        .i = i,
        .tiles = tiles,
        .eps = eps,
    };

    return submit_pass(e, slot, REQ_MLP);
}

// Fill from one conversation-owned CSPRNG stream through the same retained
// descriptor -> kcoro SQ/CQ -> fixed-lane completion path as model passes.
// The random stream itself is serial by definition, so the fence's last arriver
// advances it exactly once; no lane races or per-draw tickets exist.
int lfm_engine_prng_fill(void *ep, LfmPrngStateV1 *state, uint64_t *out,
                         size_t count) {
    Engine *e = (Engine *)ep;
    if (!e || !state || !out || count == 0) return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    int valid = lfm_prng_fill_u64(state, nullptr, 0);
    if (valid != 0) return valid;
    slot->prng.state = state;
    slot->prng.out = out;
    slot->prng.count = count;
    return submit_pass(e, slot, REQ_PRNG);
}

extern "C" int lfm_internal_engine_prng_continuation_for_test(
    void *ep, LfmPrngStateV1 *state, uint64_t *out, size_t pass_count) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !state || !out || pass_count < PASS_CAPACITY ||
        lfm_prng_fill_u64(state, nullptr, 0) != 0) {
        return -EINVAL;
    }

    PrngContinuationChain chain{};
    if (!kc_atomic_u32_is_lock_free(&chain.done.value) ||
        kc_port_wait_u32_prepare(&chain.done.value, &chain.done.wait) != 0) {
        return -ENOTSUP;
    }

    /* Deliberately bypass PassClaim: this is the proof that a native CQ
     * callback retains its own exact slot while compatibility producers may
     * occupy the peer slot. */
    PassSlot *slot = reserve_pass_slot(e);
    if (!slot) {
        kc_port_wait_u32_release(chain.done.wait);
        return -EBUSY;
    }
    const uint64_t generation =
        slot_generation(slot);

    chain.state = state;
    chain.out = out;
    chain.count = pass_count;
    chain.next = 1;
    slot->prng = {.state = state, .out = out, .count = 1};

    const int rc = submit_slot(e, slot, generation, REQ_PRNG, 0,
                               continue_prng_chain, &chain);
    if (rc != 0) {
        if (!release_pass_slot(slot, generation)) std::abort();
        kc_port_wait_u32_release(chain.done.wait);
        return rc;
    }

    uint32_t observed = kc_atomic_u32_load_acquire(&chain.done.value);
    while (!chain.finished.load(std::memory_order_acquire)) {
        (void)kc_port_wait_u32(chain.done.wait, observed, 0);
        observed = kc_atomic_u32_load_acquire(&chain.done.value);
    }
    const int chain_rc = chain.status.load(std::memory_order_acquire);
    kc_port_wait_u32_release(chain.done.wait);
    return chain_rc;
}

// Standalone sampling is used for prefill/fallback logits and conformance. It
// still enters as one retained descriptor and one kcoro completion; integrated
// token/frame passes call run_sampler directly and pay no second ticket.
int lfm_engine_sample(void *ep, const void *logits, size_t count, uint32_t dtype,
                      const LfmSamplerConfigV1 *config, LfmPrngStateV1 *state,
                      uint32_t *out_token) {
    Engine *e = (Engine *)ep;
    if (!e || !logits || count == 0 || count > UINT32_MAX || !out_token ||
        (dtype != SAMPLE_F32 && dtype != SAMPLE_BF16) || !sample_config_valid(config))
        return -EINVAL;
    bool stochastic = (config->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 && config->top_k != 1;
    if (stochastic && (!state || lfm_prng_fill_u64(state, nullptr, 0) != 0)) return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    try {
        if (slot->scratch.sample_weights.size() < count)
            slot->scratch.sample_weights.resize(count);
        if (slot->scratch.sample_heap.size() < count)
            slot->scratch.sample_heap.resize(count);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    slot->sample = {
        .logits = logits,
        .count = count,
        .dtype = dtype,
        .config = *config,
        .state = state,
        .out = out_token,
    };
    return submit_pass(e, slot, REQ_SAMPLE);
}

static bool depth_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static int submit_bf16_gemm_f32(
    void *ep, const uint16_t *a, size_t a_count,
    const void *rhs, size_t rhs_count,
    float *out, size_t out_count,
    size_t m, size_t n, size_t k, uint32_t rhs_layout, bool direct) {
    Engine *e = (Engine *)ep;
    if (!e || !a || !rhs || !out || m == 0 || n == 0 || k == 0 ||
        m > INT_MAX || n > INT_MAX || k > INT_MAX ||
        (rhs_layout != LFM_GEMM_RHS_KN && rhs_layout != LFM_GEMM_RHS_NK))
        return -EINVAL;
    // The direct-NK contract has a baseline assembly fallback and therefore
    // does not depend on a SIMD runtime gate. Existing generic behavior stays
    // unchanged.
    if (!direct) {
#ifdef __APPLE__
        if (m == 1 && !lfm_bf16_gemm_available()) return -ENOTSUP;
#else
        if (!lfm_bf16_gemm_available()) return -ENOTSUP;
#endif
    }

    size_t a_need = 0, rhs_need = 0, out_need = 0;
    if (!depth_mul(m, k, &a_need) ||
        !depth_mul(rhs_layout == LFM_GEMM_RHS_KN ? k : n,
                   rhs_layout == LFM_GEMM_RHS_KN ? n : k, &rhs_need) ||
        !depth_mul(m, n, &out_need))
        return -EOVERFLOW;
    if (a_count != a_need || rhs_count != rhs_need || out_count != out_need)
        return -EINVAL;

    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    slot->gemm = {
        .a = a,
        .rhs = rhs,
        .out = out,
        .m = m,
        .n = n,
        .k = k,
        .rhs_layout = rhs_layout,
        .direct = direct,
    };
    return submit_pass(e, slot, REQ_GEMM);
}

int lfm_engine_bf16_gemm_f32(
    void *ep, const uint16_t *a, size_t a_count,
    const uint16_t *rhs, size_t rhs_count,
    float *out, size_t out_count,
    size_t m, size_t n, size_t k, uint32_t rhs_layout) {
    return submit_bf16_gemm_f32(ep, a, a_count, rhs, rhs_count, out,
                                out_count, m, n, k, rhs_layout, false);
}

int lfm_engine_bf16_gemm_nt_direct_f32(
    void *ep, const uint16_t *a, size_t a_count,
    const void *weights, size_t weight_count,
    float *out, size_t out_count, size_t m, size_t n, size_t k) {
    return submit_bf16_gemm_f32(ep, a, a_count, weights, weight_count, out,
                                out_count, m, n, k, LFM_GEMM_RHS_NK, true);
}

int lfm_engine_fft_conv_dd(
    void *ep, const float *input, size_t input_count,
    const float *kernel, size_t kernel_count,
    const float *skip, size_t skip_count,
    float *out, size_t out_count,
    size_t batch, size_t channels, size_t steps, size_t fft_size) {
    Engine *e = (Engine *)ep;
    if (!e || !input || !kernel || !skip || !out || batch == 0 ||
        channels == 0 || steps == 0 || fft_size == 0 || fft_size < steps ||
        (fft_size & (fft_size - 1)) != 0)
        return -EINVAL;

    const size_t half = fft_size / 2 + 1;
    size_t signals = 0, input_need = 0, kernel_need = 0;
    if (!depth_mul(batch, channels, &signals) ||
        !depth_mul(signals, steps, &input_need) ||
        !depth_mul(channels, half, &kernel_need) ||
        !depth_mul(kernel_need, 2, &kernel_need))
        return -EOVERFLOW;
    if (input_count != input_need || kernel_count != kernel_need ||
        skip_count != channels || out_count != input_need)
        return -EINVAL;

    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    try {
        if (slot->scratch.fft_twiddle_size != fft_size) {
            slot->scratch.fft_twiddles.resize(fft_size / 2);
            constexpr double pi = 3.141592653589793238462643383279502884;
            for (size_t i = 0; i < fft_size / 2; ++i) {
                const double angle = -2.0 * pi * (double)i / (double)fft_size;
                slot->scratch.fft_twiddles[i] = {
                    dd_from_f64(std::cos(angle)),
                    dd_from_f64(std::sin(angle))};
            }
            slot->scratch.fft_twiddle_size = fft_size;
        }
        if (slot->scratch.fft_work.size() < fft_size)
            slot->scratch.fft_work.resize(fft_size);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    slot->fft_conv_dd = {
        .input = input,
        .kernel = kernel,
        .skip = skip,
        .out = out,
        .batch = batch,
        .channels = channels,
        .steps = steps,
        .fft_size = fft_size,
    };
    return submit_pass(e, slot, REQ_FFT_CONV_DD);
}

int lfm_engine_irfft_dd(
    void *ep, const float *real, size_t real_count,
    const float *imag, size_t imag_count,
    float *out, size_t out_count,
    size_t rows, size_t fft_size, float scale_hi, float scale_lo) {
    Engine *e = (Engine *)ep;
    if (!e || !real || !imag || !out || rows == 0 || fft_size == 0 ||
        !std::isfinite(scale_hi) || !std::isfinite(scale_lo))
        return -EINVAL;

    const size_t frequency = fft_size / 2 + 1;
    size_t input_need = 0, output_need = 0;
    if (!depth_mul(rows, frequency, &input_need) ||
        !depth_mul(rows, fft_size, &output_need))
        return -EOVERFLOW;
    if (real_count != input_need || imag_count != input_need ||
        out_count != output_need)
        return -EINVAL;

    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    try {
        if (slot->scratch.irfft_twiddle_size != fft_size) {
            slot->scratch.irfft_twiddles.resize(fft_size);
            constexpr double pi = 3.141592653589793238462643383279502884;
            for (size_t i = 0; i < fft_size; ++i) {
                const double angle = 2.0 * pi * (double)i / (double)fft_size;
                slot->scratch.irfft_twiddles[i] = {
                    dd_from_f64(std::cos(angle)),
                    dd_from_f64(std::sin(angle))};
            }
            slot->scratch.irfft_twiddle_size = fft_size;
        }
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    slot->irfft_dd = {
        .real = real,
        .imag = imag,
        .out = out,
        .rows = rows,
        .fft_size = fft_size,
        .scale = {scale_hi, scale_lo},
    };
    return submit_pass(e, slot, REQ_IRFFT_DD);
}

static bool depth_view(const LfmDepthBufferV1 &view, size_t count) {
    return view.address != 0 && view.count >= count;
}

int lfm_engine_depthwise_stream_bf16(
    void *ep, const uint16_t *x, size_t x_count,
    const uint16_t *cache, size_t cache_count,
    const uint16_t *weights, size_t weight_count,
    uint16_t *out, size_t out_count,
    uint16_t *next, size_t next_count,
    size_t batch, size_t channels, size_t steps, size_t kernel) {
    Engine *e = (Engine *)ep;
    if (!e || !x || !weights || !out || batch == 0 || channels == 0 ||
        steps == 0 || kernel == 0 || steps > INT_MAX || kernel > INT_MAX)
        return -EINVAL;
    if (!lfm_depthwise_stream_bf16_available()) return -ENOTSUP;

    size_t rows = 0, input_need = 0, weight_need = 0, cache_need = 0;
    size_t output_need = 0;
    if (!depth_mul(batch, channels, &rows) ||
        !depth_mul(rows, steps, &input_need) ||
        !depth_mul(channels, kernel, &weight_need) ||
        !depth_mul(rows, kernel - 1, &cache_need) ||
        !depth_mul(rows, steps, &output_need))
        return -EOVERFLOW;

    const bool fresh = cache == nullptr && cache_count == 0;
    const bool resumed = cache_need != 0 && cache != nullptr && cache_count == cache_need;
    if (x_count != input_need || weight_count != weight_need ||
        out_count != output_need || next_count != cache_need ||
        (cache_need != 0 && !next) || (!fresh && !resumed))
        return -EINVAL;

    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    slot->depthwise_stream = {
        .x = x,
        .cache = cache,
        .weights = weights,
        .out = out,
        .next = next,
        .batch = batch,
        .channels = channels,
        .steps = steps,
        .kernel = kernel,
    };
    return submit_pass(e, slot, REQ_DEPTHWISE_STREAM);
}

// Install one resident Depthformer plan. Descriptor tables are copied; weights
// remain zero-copy views into the model image. All scratch is reserved before
// the plan identity becomes live, so frame passes cannot allocate.
int lfm_engine_depth_build(void *ep, const LfmDepthPlanV1 *plan, uint64_t *out_id) {
    Engine *e = (Engine *)ep;
    if (!e || !plan || !out_id || plan->size < sizeof(*plan) ||
        plan->abi_version != LFM_DEPTH_ABI_VERSION || !plan->layers ||
        !plan->codebook_heads || plan->layer_count == 0 || plan->dim == 0 ||
        plan->heads == 0 || plan->kv_heads == 0 || plan->head_dim == 0 ||
        plan->ffn_dim == 0 || plan->codebooks == 0 || plan->backbone_dim == 0 ||
        plan->codebooks > 64 || plan->head_dim > 128 || plan->head_dim % 2 != 0 ||
        plan->heads % plan->kv_heads != 0 ||
        (size_t)plan->heads * plan->head_dim != plan->dim ||
        plan->codebook_head_count != plan->codebooks || !std::isfinite(plan->eps) ||
        plan->eps < 0.0f)
        return -EINVAL;
    PlanClaim claim(e);
    if (!claim) return -EBUSY;
    const size_t dim = plan->dim;
    const size_t kv_heads = plan->kv_heads;
    const size_t hd = plan->head_dim;
    const size_t ffn = plan->ffn_dim;
    const size_t codebooks = plan->codebooks;
    const size_t backbone = plan->backbone_dim;
    size_t qkv_rows = 0, projection_rows = 0, count = 0;
    if (!depth_mul(2, kv_heads, &count) || !depth_mul(count, hd, &count) ||
        count > SIZE_MAX - dim)
        return -EOVERFLOW;
    qkv_rows = dim + count;
    if (dim > INT_MAX || ffn > INT_MAX || qkv_rows > INT_MAX || hd > INT_MAX ||
        codebooks > INT_MAX || backbone > INT_MAX ||
        !depth_mul(codebooks, dim, &projection_rows) || projection_rows > INT_MAX)
        return -EOVERFLOW;
    size_t depth_weight_count = 0;
    if (!depth_mul(projection_rows, backbone, &depth_weight_count) ||
        !depth_view(plan->depth_linear_w, depth_weight_count) ||
        !depth_view(plan->depth_linear_b, projection_rows))
        return -EINVAL;
    const size_t half = hd / 2;
    size_t rope_count = 0;
    if (!depth_mul(codebooks, half, &rope_count) ||
        !depth_view(plan->rope_cos, rope_count) || !depth_view(plan->rope_sin, rope_count))
        return -EINVAL;

    size_t qkv_weight_count = 0, square = 0, ffn_weight_count = 0;
    if (!depth_mul(qkv_rows, dim, &qkv_weight_count) || !depth_mul(dim, dim, &square) ||
        !depth_mul(ffn, dim, &ffn_weight_count))
        return -EOVERFLOW;
    for (size_t i = 0; i < plan->layer_count; ++i) {
        const LfmDepthLayerV1 &layer = plan->layers[i];
        if (!depth_view(layer.qkv_w, qkv_weight_count) ||
            !depth_view(layer.out_w, square) || !depth_view(layer.q_ln, hd) ||
            !depth_view(layer.k_ln, hd) || !depth_view(layer.op_norm, dim) ||
            !depth_view(layer.ffn_norm, dim) ||
            !depth_view(layer.w1, ffn_weight_count) ||
            !depth_view(layer.w3, ffn_weight_count) ||
            !depth_view(layer.w2, ffn_weight_count))
            return -EINVAL;
    }

    size_t vocab_max = 0;
    for (size_t i = 0; i < codebooks; ++i) {
        const LfmDepthHeadV1 &head = plan->codebook_heads[i];
        size_t table_count = 0;
        if (head.vocab == 0 || head.vocab > INT_MAX ||
            !depth_mul(head.vocab, dim, &table_count) ||
            !depth_view(head.embedding, table_count) || !depth_view(head.norm, dim) ||
            !depth_view(head.logits, table_count))
            return -EINVAL;
        vocab_max = std::max(vocab_max, head.vocab);
    }

    size_t cache_count = 0;
    if (!depth_mul(plan->layer_count, kv_heads, &cache_count) ||
        !depth_mul(cache_count, codebooks, &cache_count) ||
        !depth_mul(cache_count, hd, &cache_count))
        return -EOVERFLOW;
    const size_t plane = std::max({dim, ffn, vocab_max, qkv_rows, projection_rows});

    std::unique_ptr<DepthPlan> next(new (std::nothrow) DepthPlan());
    if (!next) return -ENOMEM;
    try {
        next->layers.assign(plan->layers, plan->layers + plan->layer_count);
        next->heads.assign(plan->codebook_heads, plan->codebook_heads + codebooks);
        for (PassSlot &slot : e->slots) {
            DepthScratch &scratch = slot.scratch.depth;
            scratch.x.resize(dim);
            scratch.h.resize(dim);
            scratch.xn.resize(dim);
            scratch.qkv_f.resize(qkv_rows);
            scratch.qkv_b.resize(qkv_rows);
            scratch.up_f.resize(ffn);
            scratch.y_b.resize(plane);
            scratch.q_f.resize((size_t)plan->heads * hd);
            scratch.attn_f.resize(dim);
            scratch.attn_b.resize(dim);
            scratch.proj_f.resize(plane);
            scratch.t_b.resize(ffn);
            scratch.k_plane.resize(cache_count);
            scratch.v_plane.resize(cache_count);
            scratch.logits_b.resize(vocab_max);
            scratch.din_b.resize(projection_rows);
            scratch.df_b.resize(dim);
            if (slot.scratch.sample_weights.size() < vocab_max)
                slot.scratch.sample_weights.resize(vocab_max);
            if (slot.scratch.sample_heap.size() < vocab_max)
                slot.scratch.sample_heap.resize(vocab_max);
        }
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }

    next->depth_linear_w = depth_bytes(plan->depth_linear_w);
    next->depth_linear_b = depth_bytes(plan->depth_linear_b);
    next->cos = depth_f32(plan->rope_cos);
    next->sin = depth_f32(plan->rope_sin);
    next->dim = dim;
    next->heads_total = plan->heads;
    next->kv_heads = kv_heads;
    next->hd = hd;
    next->ffn = ffn;
    next->codebooks = codebooks;
    next->backbone_dim = backbone;
    next->eps = plan->eps;
    next->id = ++e->depth_seq;
    if (next->id == 0) next->id = ++e->depth_seq;
    const uint64_t id = next->id;
    try {
        e->depth_plans.push_back(std::move(next));
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    *out_id = id;
    return 0;
}

int lfm_engine_depth_frame(void *ep, uint64_t id, const uint16_t *hidden,
                           size_t hidden_count, const LfmSamplerConfigV1 *sampler,
                           LfmPrngStateV1 *sample_state, uint32_t *out_tokens,
                           size_t out_token_count) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !hidden || !out_tokens || !sample_config_valid(sampler))
        return -EINVAL;
    const bool stochastic = (sampler->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 &&
                            sampler->top_k != 1;
    if (stochastic && (!sample_state || lfm_prng_fill_u64(sample_state, nullptr, 0) != 0))
        return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    DepthPlan *depth = nullptr;
    for (const std::unique_ptr<DepthPlan> &candidate : e->depth_plans)
        if (candidate->id == id) {
            depth = candidate.get();
            break;
        }
    if (!depth) return -ESTALE;
    if (hidden_count != depth->backbone_dim || out_token_count != depth->codebooks)
        return -EINVAL;
    slot->depth = depth;
    slot->depth_req = {
        .hidden = hidden,
        .sampler = *sampler,
        .sample_state = sample_state,
        .out_tokens = out_tokens,
    };
    return submit_pass(e, slot, REQ_DEPTH_FRAME, id);
}

static int run_audio_route(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    uint32_t *out_codes, size_t code_count, size_t lanes,
    const LfmTokenCommitRecord *commit, uint32_t *out_token_completed,
    MimiDecodeState *mimi, const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, LfmAudioRouteNotify notify = nullptr,
    void *notify_context = nullptr,
    LfmAudioRouteHandle *out_handle = nullptr,
    uint32_t *terminal_sampled = nullptr) {
    Engine *e = static_cast<Engine *>(ep);
    if (out_handle) *out_handle = {};
    const bool terminal_after_token = terminal_sampled != nullptr;
    const bool decode_mimi = mimi || target || result;
    if (terminal_after_token &&
        (decode_mimi || depth_id != 0 || out_codes != nullptr ||
         code_count != 0)) {
        return -EINVAL;
    }
    if (decode_mimi && (!mimi || !target || !result || !target->epoch ||
                        !target->pcm || target->expected_epoch == 0 ||
                        target->pcm_capacity < LFM_MIMI_PCM_CAPACITY ||
                        out_codes != result->codes ||
                        out_token_completed != &result->token_completed ||
                        !commit || commit->token_committed !=
                                       &result->token_committed ||
                        code_count != LFM_MIMI_CODEBOOKS)) {
        return -EINVAL;
    }
    if (result) {
        std::memset(result, 0, sizeof(*result));
        result->status = -EINPROGRESS;
    }
    if (!commit || !commit->token_committed || !out_token_completed) {
        return -EINVAL;
    }
    *commit->token_committed = 0;
    *out_token_completed = 0;
    const LfmTokenCommitRecord bound_commit = *commit;
    if (!e || model_id == 0 || (!terminal_after_token && depth_id == 0) ||
        !ids || id_count == 0 || !states || !out_hidden ||
        (!terminal_after_token && !out_codes) || !bound_commit.window ||
        !logical_lane_count_valid(lanes) ||
        !sample_config_valid(audio_sampler)) {
        return -EINVAL;
    }
    const bool stochastic =
        (audio_sampler->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 &&
        audio_sampler->top_k != 1;
    if (stochastic &&
        (!prng || lfm_prng_fill_u64(prng, nullptr, 0) != 0)) {
        return -EINVAL;
    }
    const int commit_status =
        preflight_audio_route_commit(&bound_commit, position);
    if (commit_status != 0) return commit_status;
    if (decode_mimi &&
        target->epoch->load(std::memory_order_acquire) !=
            target->expected_epoch) {
        return -ESTALE;
    }

    AudioRouteLease claim(e);
    if (!claim) return -EBUSY;
    AudioRouteInstance *route = claim.route();
    BackbonePlan *model = find_model(e, model_id);
    DepthPlan *depth = nullptr;
    if (!terminal_after_token) {
        for (const std::unique_ptr<DepthPlan> &candidate : e->depth_plans) {
            if (candidate->id == depth_id) {
                depth = candidate.get();
                break;
            }
        }
    }
    if (!model || !model->embed_w || !model->emb_norm_w ||
        (!terminal_after_token && !depth)) {
        return -ESTALE;
    }
    if (hidden_elements != model->h ||
        (!terminal_after_token && hidden_elements != depth->backbone_dim) ||
        (!terminal_after_token && code_count != depth->codebooks) ||
        state_count != model->layers.size() ||
        position >= model->max_ctx) {
        return -EINVAL;
    }
    if (embedding_kind == 0) {
        if (id_count != 1 || ids[0] >= model->vocab) return -ERANGE;
    } else if (embedding_kind == 1) {
        if (!model->audio_embed_w || id_count > TOKEN_INPUT_MAX_IDS) return -ERANGE;
        for (size_t index = 0; index < id_count; ++index) {
            if (ids[index] >= model->audio_rows) return -ERANGE;
        }
    } else {
        return -EINVAL;
    }
    for (size_t layer = 0; layer < model->layers.size(); ++layer) {
        const LfmLayerDesc &descriptor = model->layers[layer];
        const LfmLayerState &state = states[layer];
        if (descriptor.kind == 1) {
            if (!descriptor.q_w || !state.k_plane || !state.v_plane ||
                !rope_cos || !rope_sin || descriptor.hd == 0 ||
                descriptor.n_kv == 0 || position + 1 > SIZE_MAX / descriptor.hd) {
                return -ESTALE;
            }
            const size_t live = (position + 1) * descriptor.hd;
            const size_t prior_heads = descriptor.n_kv - 1;
            if (state.head_stride < live ||
                prior_heads > SIZE_MAX / state.head_stride ||
                prior_heads * state.head_stride > SIZE_MAX - live ||
                state.k_len < prior_heads * state.head_stride + live ||
                state.v_len < prior_heads * state.head_stride + live ||
                position + 1 > SIZE_MAX / (descriptor.hd / 2) ||
                rope_elements < (position + 1) * (descriptor.hd / 2)) {
                return -EINVAL;
            }
        } else if (descriptor.kind == 0) {
            const size_t tail = descriptor.k > 0 ? descriptor.k - 1 : 0;
            if (!state.conv_state || descriptor.k < 1 ||
                (tail > 0 && model->h > SIZE_MAX / tail) ||
                state.conv_len < model->h * tail) {
                return -ESTALE;
            }
        } else {
            return -EPROTO;
        }
    }

    route->engine = e;
    route->service_class = decode_mimi ? KC_COORD_SERVICE_DEADLINE
                                       : KC_COORD_SERVICE_INTERACTIVE;
    route->node = AUDIO_ROUTE_TOKEN;
    route->depth_id = depth_id;
    route->depth = depth;
    route->model = model;
    route->model_id = model_id;
    route->token_req = {
        .ids = ids,
        .n_ids = id_count,
        .embed_kind = embedding_kind,
        .provided_embed = nullptr,
        .states = states,
        .n_states = state_count,
        .pos = position,
        .cos_base = rope_cos,
        .sin_base = rope_sin,
        .out_hidden = out_hidden,
        .out_logits = nullptr,
        .sampler = terminal_after_token ? audio_sampler : nullptr,
        .sample_state = terminal_after_token ? prng : nullptr,
        .out_token = terminal_sampled,
        .lanes = lanes,
    };
    if (!terminal_after_token) route->depth_req = {
            .hidden = out_hidden,
            .sampler = *audio_sampler,
            .sample_state = prng,
            .out_tokens = out_codes,
            .completion_status =
                e->test_audio_route_depth_status.exchange(
                    0, std::memory_order_acq_rel),
        };
    route->mimi_req = {
            .state = mimi,
            .codes = out_codes,
            .pcm = decode_mimi ? target->pcm : nullptr,
            .capacity = decode_mimi ? target->pcm_capacity : 0,
            .out_samples = result ? &result->pcm_samples : nullptr,
            .completion_status = decode_mimi
                ? e->test_audio_route_mimi_status.exchange(
                      0, std::memory_order_acq_rel)
                : 0,
        };
    route->epoch = decode_mimi ? target->epoch : nullptr;
    route->expected_epoch = decode_mimi ? target->expected_epoch : 0;
    route->result = result;
    route->notify = notify;
    route->notify_context = notify_context;
    route->terminal_after_token = terminal_after_token;
    route->decode_mimi = decode_mimi;
    route->commit = bound_commit;
    route->token_completed = out_token_completed;
    route->status = -EINPROGRESS;
    route->enqueue_sequence =
        e->route_pool->sequence.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (out_handle) {
        out_handle->record = route;
        out_handle->generation = claim.generation();
    }
    uint32_t claimed = AUDIO_ROUTE_CLAIMED;
    if (!route->state.compare_exchange_strong(
            claimed, AUDIO_ROUTE_READY, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        if (out_handle) *out_handle = {};
        return -ESTALE;
    }
    if (out_handle) claim.detach();
    signal_all(&e->route_word);
    if (out_handle) return 0;
    return wait_audio_route(route, claim.generation());
}

int lfm_engine_audio_recurrence(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    uint32_t *out_codes, size_t code_count, size_t lanes,
    const LfmTokenCommitRecord *commit, uint32_t *out_token_completed) {
    return run_audio_route(
        ep, model_id, depth_id, ids, id_count, embedding_kind, states,
        state_count, position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, audio_sampler, prng, out_codes, code_count, lanes,
        commit, out_token_completed, nullptr, nullptr, nullptr);
}

int lfm_engine_audio_route(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    MimiDecodeState *mimi, const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, size_t lanes,
    const LfmTokenCommitRecord *commit) {
    if (!result) return -EINVAL;
    const int status = run_audio_route(
        ep, model_id, depth_id, ids, id_count, embedding_kind, states,
        state_count, position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, audio_sampler, prng, result->codes,
        LFM_MIMI_CODEBOOKS, lanes, commit, &result->token_completed, mimi,
        target, result);
    if (result->status == -EINPROGRESS) result->status = status;
    return status;
}

int lfm_engine_audio_route_submit(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    MimiDecodeState *mimi, const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, size_t lanes,
    const LfmTokenCommitRecord *commit, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    if (!result || !notify || !out_handle) return -EINVAL;
    return run_audio_route(
        ep, model_id, depth_id, ids, id_count, embedding_kind, states,
        state_count, position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, audio_sampler, prng, result->codes,
        LFM_MIMI_CODEBOOKS, lanes, commit, &result->token_completed, mimi,
        target, result, notify, notify_context, out_handle);
}

int lfm_engine_token_route_submit(
    void *ep, uint64_t model_id, const uint32_t *ids, size_t id_count,
    uint32_t embedding_kind, const LfmLayerState *states, size_t state_count,
    size_t position, const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *sampler, LfmPrngStateV1 *prng,
    uint32_t *out_token, size_t lanes,
    const LfmTokenCommitRecord *commit, uint32_t *out_token_completed,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *out_handle) {
    if (!out_token || !notify || !out_handle) return -EINVAL;
    return run_audio_route(
        ep, model_id, 0, ids, id_count, embedding_kind, states, state_count,
        position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, sampler, prng, nullptr, 0, lanes, commit,
        out_token_completed, nullptr, nullptr, nullptr, notify,
        notify_context, out_handle, out_token);
}

int lfm_engine_audio_route_collect(void *ep,
                                   LfmAudioRouteHandle *handle) {
    Engine *engine = static_cast<Engine *>(ep);
    if (!engine || !handle || !handle->record || handle->generation == 0 ||
        !engine->route_pool) {
        return -EINVAL;
    }
    AudioRouteInstance *route =
        static_cast<AudioRouteInstance *>(handle->record);
    bool owned = false;
    for (AudioRouteInstance &candidate : engine->route_pool->routes) {
        if (&candidate == route) {
            owned = true;
            break;
        }
    }
    if (!owned || route->engine != engine ||
        route->generation.load(std::memory_order_acquire) !=
            handle->generation) {
        return -ESTALE;
    }
    if (route->state.load(std::memory_order_acquire) != AUDIO_ROUTE_DONE) {
        return -EINPROGRESS;
    }
    const int status = route->status;
    release_audio_route(route, handle->generation);
    *handle = {};
    return status;
}

int lfm_engine_mimi_decode(void *ep, uint64_t model_id,
                           MimiDecodeState *state, const uint32_t *codes,
                           size_t code_count, float *pcm_out,
                           size_t pcm_capacity, size_t *out_samples) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || model_id == 0 || !state || !codes || !pcm_out || !out_samples ||
        code_count != LFM_MIMI_CODEBOOKS ||
        pcm_capacity < LFM_MIMI_PCM_CAPACITY) {
        return -EINVAL;
    }
    for (size_t index = 0; index < code_count; ++index) {
        if (codes[index] >= LFM_MIMI_CODE_VALUES) return -ERANGE;
    }
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    BackbonePlan *model = find_model(e, model_id);
    if (!model) return -ESTALE;
    *out_samples = 0;
    slot->model = model;
    slot->mimi = {
        .state = state,
        .codes = codes,
        .pcm = pcm_out,
        .capacity = pcm_capacity,
        .out_samples = out_samples,
    };
    return submit_pass(e, slot, REQ_MIMI_DECODE, model_id);
}

int lfm_engine_depth_clear(void *ep, uint64_t id) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0) return -EINVAL;
    PlanClaim claim(e);
    if (!claim) return -EBUSY;
    const auto found = std::find_if(
        e->depth_plans.begin(), e->depth_plans.end(),
        [id](const std::unique_ptr<DepthPlan> &candidate) { return candidate->id == id; });
    if (found == e->depth_plans.end()) return 0;
    e->depth_plans.erase(found);
    return 0;
}

// Build the resident layer table: one descriptor per backbone block (indexed by
// block_idx), plus the model dims. Sizes ALL pass scratch here — fixed-arena
// discipline: after a successful build, conv-layer passes allocate nothing.
// The Rust rim serializes this against passes (pass_lock); pointers must stay valid
// until lfm_ctx_clear (the model-side guard guarantees clear-before-drop). Plans
// coexist by identity; shared executor scratch grows to the largest live geometry.
int lfm_ctx_build(void *ep, const LfmLayerDesc *descs, size_t n_layers, size_t h,
                  size_t ffn, size_t max_ctx, uint64_t *out_id) {
    Engine *e = (Engine *)ep;
    if (!e || !descs || n_layers == 0 || h == 0 || ffn == 0 || max_ctx == 0 || !out_id)
        return -1;
    if (h > (size_t)INT_MAX / 3 || ffn > (size_t)INT_MAX ||
        max_ctx > (size_t)INT_MAX)
        return -EOVERFLOW;
    PlanClaim claim(e);
    if (!claim) return -EBUSY;
    size_t kmax = 1;
    size_t qkv_max = 0, y_max = 0, att_max = 0;
    for (size_t l = 0; l < n_layers; ++l) {
        if (descs[l].kind ==
            static_cast<uint32_t>(SequenceMixerKind::MonarchLongConv)) {
            return -ENOTSUP;
        }
        if (descs[l].kind != 0 && descs[l].kind != 1) return -EINVAL;
        if (descs[l].kind == 0) {
            if (!descs[l].op_norm_w || !descs[l].ffn_norm_w || !descs[l].in_w ||
                !descs[l].conv_w || !descs[l].out_w || !descs[l].w1 || !descs[l].w3 ||
                !descs[l].w2 || descs[l].k < 1 || descs[l].k > 8)
                return -1;
            if (descs[l].k > kmax) kmax = descs[l].k;
        } else if (descs[l].q_w) {
            // Attention slot with capture: all fields or nothing (q_w NULL = unserved).
            if (!descs[l].op_norm_w || !descs[l].ffn_norm_w || !descs[l].k_w ||
                !descs[l].v_w || !descs[l].o_w || !descs[l].qn_w || !descs[l].kn_w ||
                !descs[l].w1 || !descs[l].w3 || !descs[l].w2 || descs[l].n_head == 0 ||
                descs[l].n_kv == 0 || descs[l].hd == 0 || descs[l].hd > 512 ||
                descs[l].n_head % descs[l].n_kv != 0 || descs[l].hd % 2 != 0)
                return -1;
            const size_t nh = descs[l].n_head;
            const size_t nkv = descs[l].n_kv;
            const size_t hd = descs[l].hd;
            if (nkv > (SIZE_MAX - nh) / 2) return -EOVERFLOW;
            size_t qkv = 0, y = 0, att = 0;
            if (!depth_mul(nh + 2 * nkv, hd, &qkv) ||
                !depth_mul(nh, hd, &y) ||
                !depth_mul(nh, max_ctx, &att))
                return -EOVERFLOW;
            if (qkv > (size_t)INT_MAX || y > (size_t)INT_MAX)
                return -EOVERFLOW;
            qkv_max = std::max(qkv_max, qkv);
            y_max = std::max(y_max, y);
            att_max = std::max(att_max, att);
        }
    }
    std::unique_ptr<BackbonePlan> next(new (std::nothrow) BackbonePlan());
    if (!next) return -ENOMEM;
    try {
        next->layers.assign(descs, descs + n_layers);
        next->mixers.reserve(n_layers);
        for (size_t layer = 0; layer < n_layers; ++layer) {
            const bool shortconv = descs[layer].kind == 0;
            next->mixers.push_back({
                .kind = shortconv ? SequenceMixerKind::ShortConv
                                  : SequenceMixerKind::Attention,
                .layer = (uint32_t)layer,
                .kernel = shortconv ? (uint32_t)descs[layer].k : 0u,
                .halo = shortconv ? (uint32_t)(descs[layer].k - 1) : 0u,
            });
        }
        const auto grow = [](auto &values, size_t count) {
            if (values.size() < count) values.resize(count);
        };
        for (PassSlot &slot : e->slots) {
            ScratchBank &scratch = slot.scratch;
            grow(scratch.sc_partials, MAX_WORKERS);
            grow(scratch.sc_xn, h);
            grow(scratch.sc_gu, 2 * ffn);
            grow(scratch.sc_t, ffn);
            grow(scratch.sc_bcxf, 3 * h);
            grow(scratch.sc_bcxb, 3 * h);
            grow(scratch.sc_conv, h * kmax);
            grow(scratch.sc_projf, h);
            grow(scratch.sc_projb, h);
            grow(scratch.sc_stage, h);
            grow(scratch.sc_mid, h);
            if (qkv_max > 0) {
                grow(scratch.at_qkvf, qkv_max);
                grow(scratch.at_qkvb, qkv_max);
                grow(scratch.at_y, y_max);
                grow(scratch.at_att, att_max);
            }
            grow(scratch.tk_h0, h);
            grow(scratch.tk_h1, h);
        }
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    update_capacity_high_water(&e->attention_qkv_capacity, qkv_max);
    update_capacity_high_water(&e->attention_y_capacity, y_max);
    update_capacity_high_water(&e->attention_score_capacity, att_max);
    next->h = h;
    next->ffn = ffn;
    next->max_ctx = max_ctx;
    next->qkv_max = qkv_max;
    next->y_max = y_max;
    next->kmax = kmax;
    next->id = ++e->model_seq;
    if (next->id == 0) next->id = ++e->model_seq;
    const uint64_t id = next->id;
    try {
        e->models.push_back(std::move(next));
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    *out_id = id;
    return 0;
}

// Install the head tables (embed / audio-embed / final norm / tied logits) — the
// token pass needs them; the per-layer entries do not. Serialized by the rim.
int lfm_ctx_set_heads(void *ep, uint64_t id, const uint8_t *embed_w,
                      size_t embed_len, size_t vocab, const uint8_t *audio_embed_w,
                      size_t audio_embed_len, size_t audio_rows,
                      const uint8_t *emb_norm_w, size_t emb_norm_len,
                      float emb_norm_eps) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !embed_w || !emb_norm_w || vocab == 0) return -1;
    PlanClaim claim(e);
    if (!claim) return -EBUSY;
    BackbonePlan *model = find_model(e, id);
    if (!model) return -3;
    if (vocab > (size_t)INT_MAX || vocab > SIZE_MAX / model->h ||
        embed_len < vocab * model->h ||
        emb_norm_len < model->h)
        return -1;
    if (audio_rows > 0 &&
        (!audio_embed_w || audio_rows > SIZE_MAX / model->h ||
         audio_embed_len < audio_rows * model->h))
        return -1;
    try {
        for (PassSlot &slot : e->slots) {
            if (slot.scratch.tk_logf.size() < vocab)
                slot.scratch.tk_logf.resize(vocab);
            if (slot.scratch.sample_weights.size() < vocab)
                slot.scratch.sample_weights.resize(vocab);
            if (slot.scratch.sample_heap.size() < vocab)
                slot.scratch.sample_heap.resize(vocab);
        }
    } catch (const std::bad_alloc &) {
        return -2;
    }
    model->embed_w = embed_w;
    model->vocab = vocab;
    model->audio_embed_w = audio_embed_w;
    model->audio_rows = audio_rows;
    model->emb_norm_w = emb_norm_w;
    model->emb_norm_eps = emb_norm_eps;
    return 0;
}

// Clear one retained plan (its weight pointers are about to die with the model).
// Serialized by the Rust rim's pass lock, so no pass is in flight here. A stale or
// foreign id is a no-op and cannot disturb any other resident model.
int lfm_ctx_clear(void *ep, uint64_t id) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0) return -1;
    PlanClaim claim(e);
    if (!claim) return -EBUSY;
    const auto found = std::find_if(
        e->models.begin(), e->models.end(),
        [id](const std::unique_ptr<BackbonePlan> &model) { return model->id == id; });
    if (found == e->models.end()) return 0;
    e->models.erase(found);
    return 0;
}

// One whole shortconv+MLP layer: request slot → doorbell → park. Returns 0 on
// success; -3 when the plan is stale or the slot is not a conv layer.
int lfm_engine_conv_layer(void *ep, uint64_t id, size_t layer, const uint16_t *x,
                          size_t x_len, const uint16_t *state_in, size_t state_in_len,
                          uint16_t *state_out, size_t state_out_len, uint16_t *out,
                          size_t out_len, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !x || !state_in || !state_out || !out ||
        !logical_lane_count_valid(lanes))
        return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    BackbonePlan *model = find_model(e, id);
    if (!model || layer >= model->layers.size() || model->layers[layer].kind != 0)
        return -3;
    const size_t k = model->layers[layer].k;
    const size_t tail = k > 0 ? k - 1 : 0;
    if (k < 1 || (tail > 0 && model->h > SIZE_MAX / tail)) return -1;
    const size_t state_len = model->h * tail;
    if (x_len != model->h || out_len != model->h || state_in_len != state_len ||
        state_out_len != state_len)
        return -1;

    slot->model = model;
    slot->conv.layer = layer;
    slot->conv.x = x;
    slot->conv.state_in = state_in;
    slot->conv.state_out = state_out;
    slot->conv.out = out;
    slot->conv.lanes = lanes;

    return submit_pass(e, slot, REQ_CONV_LAYER, id);
}

// One whole attention+MLP layer. Per-generation state (planes, rope tables, cursor)
// arrives per request; the engine appends the step's K/V rows at `pos` and attends
// over pos+1 entries. Rows beyond `pos` must already fit the planes (the caller
// pre-grows capacity BEFORE capturing the plane pointers). Returns 0 on success;
// -3 when unserved (no ctx / not an attention slot / capture was null / pos over cap).
int lfm_engine_attn_layer(void *ep, uint64_t id, size_t layer, const uint16_t *x,
                          size_t x_len, uint16_t *k_plane, size_t k_len,
                          uint16_t *v_plane, size_t v_len, size_t head_stride,
                          size_t pos, const uint16_t *cos_base, const uint16_t *sin_base,
                          size_t rope_len, uint16_t *out, size_t out_len, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !x || !k_plane || !v_plane || !cos_base || !sin_base || !out ||
        !logical_lane_count_valid(lanes))
        return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    BackbonePlan *model = find_model(e, id);
    if (!model || layer >= model->layers.size() || model->layers[layer].kind != 1 ||
        !model->layers[layer].q_w || pos + 1 > model->max_ctx)
        return -3;
    const LfmLayerDesc *d = &model->layers[layer];
    if (d->hd == 0 || d->n_kv == 0 || pos + 1 > SIZE_MAX / d->hd) return -1;
    const size_t live = (pos + 1) * d->hd;
    const size_t prior_heads = d->n_kv - 1;
    if (x_len != model->h || out_len != model->h || head_stride < live ||
        prior_heads > SIZE_MAX / head_stride ||
        prior_heads * head_stride > SIZE_MAX - live ||
        k_len < prior_heads * head_stride + live ||
        v_len < prior_heads * head_stride + live ||
        pos + 1 > SIZE_MAX / (d->hd / 2) ||
        rope_len < (pos + 1) * (d->hd / 2))
        return -1;

    slot->model = model;
    slot->attn.layer = layer;
    slot->attn.x = x;
    slot->attn.k_plane = k_plane;
    slot->attn.v_plane = v_plane;
    slot->attn.head_stride = head_stride;
    slot->attn.pos = pos;
    slot->attn.cos_base = cos_base;
    slot->attn.sin_base = sin_base;
    slot->attn.out = out;
    slot->attn.lanes = lanes;

    return submit_pass(e, slot, REQ_ATTN_LAYER, id);
}

int lfm_engine_prefill_workspace_create(void *ep, uint64_t id,
                                        void **out_workspace) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !out_workspace) return -EINVAL;
    *out_workspace = nullptr;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    BackbonePlan *model = find_model(e, id);
    if (!model || !model->embed_w || !model->emb_norm_w) return -ESTALE;

    size_t rows_h = 0, rows_ffn = 0, rows_qkv = 0, rows_y = 0;
    size_t rows_3h = 0, rows_2ffn = 0, scores = 0;
    if (!checked_size_product(PREFILL_ROWS, model->h, &rows_h) ||
        !checked_size_product(PREFILL_ROWS, model->ffn, &rows_ffn) ||
        !checked_size_product(PREFILL_ROWS, model->qkv_max, &rows_qkv) ||
        !checked_size_product(PREFILL_ROWS, model->y_max, &rows_y) ||
        !checked_size_product(rows_h, 3, &rows_3h) ||
        !checked_size_product(rows_ffn, 2, &rows_2ffn) ||
        !checked_size_product(e->lanes_total, model->max_ctx, &scores)) {
        return -EOVERFLOW;
    }

    std::unique_ptr<PrefillWorkspace> workspace(
        new (std::nothrow) PrefillWorkspace());
    if (!workspace) return -ENOMEM;
    try {
        workspace->h0.resize(rows_h);
        workspace->h1.resize(rows_h);
        workspace->xn.resize(rows_h);
        workspace->gate.resize(rows_ffn);
        workspace->stage.resize(rows_h);
        workspace->mid.resize(rows_h);
        workspace->bcxb.resize(rows_3h);
        workspace->projb.resize(rows_h);
        workspace->qkvb.resize(rows_qkv);
        workspace->att_y.resize(rows_y);
        workspace->gu.resize(rows_2ffn);
        workspace->bcxf.resize(rows_3h);
        workspace->projf.resize(rows_h);
        workspace->qkvf.resize(rows_qkv);
        workspace->scores.resize(scores);
        workspace->logits.resize(model->vocab);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    workspace->model_id = id;
    workspace->h = model->h;
    workspace->ffn = model->ffn;
    workspace->max_ctx = model->max_ctx;
    workspace->qkv_max = model->qkv_max;
    workspace->y_max = model->y_max;
    workspace->kmax = model->kmax;
    workspace->lane_count = e->lanes_total;
    *out_workspace = workspace.release();
    return 0;
}

void lfm_engine_prefill_workspace_destroy(void *workspace) {
    delete static_cast<PrefillWorkspace *>(workspace);
}

int lfm_engine_prefill(void *ep, uint64_t id, void *workspace_pointer,
                       const uint32_t *ids, const uint16_t *provided_rows,
                       size_t row_count, uint32_t embed_kind,
                       const LfmLayerState *states, size_t state_count,
                       size_t pos, const uint16_t *cos_base,
                       const uint16_t *sin_base, size_t rope_len,
                       uint16_t *out_hidden, size_t out_hidden_len,
                       const LfmSamplerConfigV1 *sampler,
                       LfmPrngStateV1 *sample_state, uint32_t *out_token,
                       size_t lanes) {
    Engine *e = (Engine *)ep;
    PrefillWorkspace *workspace =
        static_cast<PrefillWorkspace *>(workspace_pointer);
    if (!e || id == 0 || !workspace || row_count == 0 ||
        row_count > PREFILL_ROWS || !states || !out_hidden ||
        !logical_lane_count_valid(lanes)) {
        return -EINVAL;
    }
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    BackbonePlan *model = find_model(e, id);
    if (!model || !model->embed_w || !model->emb_norm_w ||
        state_count != model->layers.size() || row_count > model->max_ctx ||
        pos > model->max_ctx - row_count) {
        return -ESTALE;
    }
    if (workspace->model_id != id || workspace->h != model->h ||
        workspace->ffn != model->ffn || workspace->max_ctx != model->max_ctx ||
        workspace->qkv_max != model->qkv_max ||
        workspace->y_max != model->y_max || workspace->kmax != model->kmax ||
        workspace->lane_count != e->lanes_total ||
        workspace->logits.size() < model->vocab ||
        out_hidden_len != model->h) {
        return -EINVAL;
    }
    if (embed_kind == 0) {
        if (!ids) return -EINVAL;
        for (size_t row = 0; row < row_count; ++row)
            if (ids[row] >= model->vocab) return -ERANGE;
    } else if (embed_kind == 2) {
        if (!provided_rows) return -EINVAL;
    } else {
        return -EINVAL;
    }
    if (out_token) {
        if (!sample_config_valid(sampler)) return -EINVAL;
        const bool stochastic =
            (sampler->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 &&
            sampler->top_k != 1;
        if (stochastic &&
            (!sample_state || lfm_prng_fill_u64(sample_state, nullptr, 0) != 0)) {
            return -EINVAL;
        }
    } else if (sampler || sample_state) {
        return -EINVAL;
    }

    const size_t end_pos = pos + row_count;
    for (size_t layer = 0; layer < model->layers.size(); ++layer) {
        const LfmLayerDesc *desc = &model->layers[layer];
        const LfmLayerState *state = &states[layer];
        if (desc->kind == 1) {
            if (!desc->q_w || !state->k_plane || !state->v_plane ||
                !cos_base || !sin_base || desc->hd == 0 || desc->n_kv == 0 ||
                end_pos > SIZE_MAX / desc->hd) {
                return -ESTALE;
            }
            const size_t live = end_pos * desc->hd;
            const size_t prior_heads = desc->n_kv - 1;
            if (state->head_stride < live ||
                prior_heads > SIZE_MAX / state->head_stride ||
                prior_heads * state->head_stride > SIZE_MAX - live ||
                state->k_len < prior_heads * state->head_stride + live ||
                state->v_len < prior_heads * state->head_stride + live ||
                end_pos > SIZE_MAX / (desc->hd / 2) ||
                rope_len < end_pos * (desc->hd / 2)) {
                return -EINVAL;
            }
            continue;
        }
        const size_t tail = desc->k > 0 ? desc->k - 1 : 0;
        if (!state->conv_state || desc->k < 1 ||
            (tail > 0 && model->h > SIZE_MAX / tail) ||
            state->conv_len < model->h * tail) {
            return -ESTALE;
        }
    }

    slot->model = model;
    slot->prefill.workspace = workspace;
    slot->prefill.ids = ids;
    slot->prefill.provided_rows = provided_rows;
    slot->prefill.rows = row_count;
    slot->prefill.embed_kind = embed_kind;
    slot->prefill.states = states;
    slot->prefill.n_states = state_count;
    slot->prefill.pos = pos;
    slot->prefill.cos_base = cos_base;
    slot->prefill.sin_base = sin_base;
    slot->prefill.out_hidden = out_hidden;
    slot->prefill.sampler = sampler;
    slot->prefill.sample_state = sample_state;
    slot->prefill.out_token = out_token;
    slot->prefill.lanes = lanes;
    return submit_pass(e, slot, REQ_PREFILL, id);
}

// ONE token through the whole backbone: embed → every layer → final norm → logits.
// `states` is one LfmLayerState per table slot (fresh pointers each token — the caller
// ensures plane capacity BEFORE capture). Returns 0 on success; -3 when unserved (no
// ctx/heads, an attention slot without capture, bad ids, or pos over capacity).
int lfm_engine_token_pass(void *ep, uint64_t id, const uint32_t *ids, size_t n_ids,
                          uint32_t embed_kind, const LfmLayerState *states,
                          size_t n_states, size_t pos, const uint16_t *cos_base,
                          const uint16_t *sin_base, size_t rope_len,
                          uint16_t *out_hidden, size_t out_hidden_len,
                          float *out_logits, size_t out_logits_len,
                          const LfmSamplerConfigV1 *sampler,
                          LfmPrngStateV1 *sample_state, uint32_t *out_token,
                          size_t lanes, const uint16_t *provided_embed) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !ids || n_ids == 0 || !states || !out_hidden ||
        !logical_lane_count_valid(lanes))
        return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    PassSlot *slot = claim.slot();
    BackbonePlan *model = find_model(e, id);
    if (!model || !model->embed_w || !model->emb_norm_w ||
        n_states != model->layers.size() || pos + 1 > model->max_ctx)
        return -3;
    if (out_hidden_len != model->h ||
        (out_logits && out_logits_len < model->vocab) ||
        (!out_logits && out_logits_len != 0))
        return -1;
    if (out_token) {
        if (!sample_config_valid(sampler)) return -1;
        bool stochastic = (sampler->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 &&
                          sampler->top_k != 1;
        if (stochastic &&
            (!sample_state || lfm_prng_fill_u64(sample_state, nullptr, 0) != 0))
            return -1;
    } else if (sampler || sample_state) {
        return -1;
    }
    if (embed_kind == 0) {
        if (ids[0] >= model->vocab) return -3;
    } else if (embed_kind == 1) {
        if (!model->audio_embed_w || n_ids > 8) return -3;
        for (size_t c = 0; c < n_ids; ++c)
            if (ids[c] >= model->audio_rows) return -3;
    } else if (embed_kind == 2) {
        if (!provided_embed) return -3; // native prefill audio-in: view required
    } else {
        return -3;
    }
    // Every attention slot must be served and carry planes; conv slots need state.
    for (size_t l = 0; l < model->layers.size(); ++l) {
        if (model->layers[l].kind == 1) {
            if (!model->layers[l].q_w || !states[l].k_plane || !states[l].v_plane ||
                !cos_base || !sin_base)
                return -3;
            const size_t hd = model->layers[l].hd;
            const size_t nkv = model->layers[l].n_kv;
            if (hd == 0 || nkv == 0 || pos + 1 > SIZE_MAX / hd) return -1;
            const size_t live = (pos + 1) * hd;
            const size_t prior_heads = nkv - 1;
            if (states[l].head_stride < live ||
                prior_heads > SIZE_MAX / states[l].head_stride ||
                prior_heads * states[l].head_stride > SIZE_MAX - live ||
                states[l].k_len < prior_heads * states[l].head_stride + live ||
                states[l].v_len < prior_heads * states[l].head_stride + live ||
                pos + 1 > SIZE_MAX / (hd / 2) || rope_len < (pos + 1) * (hd / 2))
                return -1;
        } else if (!states[l].conv_state) {
            return -3;
        } else {
            const size_t k = model->layers[l].k;
            const size_t tail = k > 0 ? k - 1 : 0;
            if (k < 1 || (tail > 0 && model->h > SIZE_MAX / tail) ||
                states[l].conv_len < model->h * tail)
                return -1;
        }
    }

    slot->model = model;
    slot->tok.ids = ids;
    slot->tok.n_ids = n_ids;
    slot->tok.embed_kind = embed_kind;
    slot->tok.provided_embed = provided_embed;
    slot->tok.states = states;
    slot->tok.n_states = n_states;
    slot->tok.pos = pos;
    slot->tok.cos_base = cos_base;
    slot->tok.sin_base = sin_base;
    slot->tok.out_hidden = out_hidden;
    slot->tok.out_logits = out_logits;
    slot->tok.sampler = sampler;
    slot->tok.sample_state = sample_state;
    slot->tok.out_token = out_token;
    slot->tok.lanes = lanes;

    return submit_pass(e, slot, REQ_TOKEN_PASS, id);
}

} // extern "C"
