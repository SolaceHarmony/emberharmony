// flashkern_engine.cpp — the resident native decode engine (ENGINE_DESIGN.md §2/§3),
// as a LANE-UNIFORM KERNEL: the engine owns all mutable state, and every lane runs
// the ENTIRE pass program — embed, every layer, final norm — exactly the way a GPU
// threadgroup runs a kernel. There is no host coordinator publishing stages:
// stages advance through fixed-team generation callbacks, tiles
// are claimed off a bare fetch_add counter (so an E-core straggler simply claims
// fewer). Each generation's final-return callback advances the bounded program
// cursor exactly once. Numerical serial phases are explicit generations owned
// by their selected logical member; they are not smuggled into a host-side
// barrier callback. The only runtime boundary is kcoro's fixed typed
// TeamExecutor: its retained frame mounts one pass descriptor, and the final
// member callback resumes that exact frame before it publishes one correlated
// completion record.
//
// Every operation enters through a retained workflow ticket. Completion makes
// the next continuation runnable; no caller thread waits on numerical progress.
// Native code owns SQ submission, CQ consumption, and pass recurrence. Stop is
// a full-pass boundary decision and is never polled inside assembly operations.
//
// Numerics: stage bodies preserve the model's RNE bf16 rounding ladder, fixed
// tile count, and fixed-order partial fold regardless of which member claims a
// tile. Architecture leaves are linked in-image and consume byte views directly.
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
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashkern_depth.h"
#include "flashkern_gemm.h"
#include "flashkern_math.h"
#include "flashkern_prng.h"
#include "flashkern_sampler.h"
#include "lfm_audio_pass.h"
#include "../model/lfm_conformer_program.h"
#include "lfm_frontend.h"
#include "lfm_detokenizer.h"
#include "lfm_detokenizer_program.h"
#include "lfm_model_plan.h"
#include "kc_coordination.hpp"
#include "kc_mailbox.hpp"
#include "kc_permit_broker.hpp"
#include "kc_team_executor.hpp"
#include "kc_team_supervisor.hpp"
#include "../runtime/lfm_runtime_diagnostics.hpp"
#include "../model/lfm_route_epoch.h"

extern "C" {
#include "kc_runtime.h"
#include "kc_team.h"
#include "kcoro_stackless.h"
}

// Stage kernels from the flashkern TU (same image, plain calls).
extern "C" float lfm_bf16_sumsq_ordered_f32(const void *x_bytes, int n);
extern "C" float lfm_bf16_sumsq_f32(const uint16_t *x, int n);
extern "C" void lfm_bf16_rmsnorm(const void *x_bytes, const void *weight_bytes, uint16_t *out,
                                 int n, float inv_rms);
extern "C" void lfm_f32_to_bf16(const float *x, uint16_t *out, int n);
extern "C" void lfm_bf16_add(const void *a_bytes, const void *b_bytes,
                              uint16_t *out, int n);
extern "C" void lfm_shortconv_project_update_bf16(
    const void *input, const void *projection_weight_bytes,
    const uint16_t *state, const void *conv_weight_bytes, uint16_t *y,
    uint16_t *next, size_t hidden, size_t channel_begin,
    size_t channel_count, size_t kernel);
extern "C" void lfm_bf16_to_f32(const uint16_t *x, float *out, int n);
extern "C" void lfm_softmax_scaled_f32(float *x, int n, float scale);
extern "C" void lfm_attn_qk_bf16(const float *q, const uint16_t *k, float *att, int len,
                                 int hd);
extern "C" void lfm_attn_av_bf16(const float *att, const uint16_t *v, float *out,
                                 int len, int hd);
extern "C" void lfm_rope_i_f32(float *x, const float *cos_p, const float *sin_p, int hd);
extern "C" void lfm_swiglu_bf16(const float *g, const float *u, uint16_t *out, int n);

/* Private implementation-backed supervision hook. This is intentionally not
 * declared by kc_team.h: product callers must not acquire a team-poison API. */
extern "C" int kc_team_inject_member_exit_for_test(
    kc_team_t *team, uint64_t generation, uint32_t member, uint32_t point,
    void (*ready)(void *, uint64_t), void *context);

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
constexpr uint32_t PASS_PUBLISHER_CAPACITY =
    static_cast<uint32_t>(ROUTE_CAPACITY + PASS_CAPACITY);
constexpr uint32_t TEAM_DEADLINE_SLOT = 0;
/* The closed production request table below assigns this conservative
 * generation ceiling to every currently mounted numerical family. Unknown
 * work receives no fallback and is rejected before team dispatch. */
constexpr uint64_t TEAM_PROBE_HARD_DEADLINE_NS = UINT64_C(1000000000);
constexpr size_t PREFILL_ROWS = LFM_PREFILL_MAX_ROWS;
constexpr size_t TOKEN_INPUT_MAX_IDS = 8;
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
    uint16_t *t;     // [i]
    std::atomic<uint32_t> rs_bits{0};
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
    ST_SC_INPROJ = 6,  // fused B/C/X projections + FIR/gate + state publication
    ST_SC_GATHER = 7,  // retired: kept as a closed internal selector value
    ST_SC_OUTPROJ = 8, // out_proj rows band: nt + round + residual add
    // Attention block stages with the accepted oracle operation order.
    ST_AT_QKV = 9,    // q|k|v projection rows band (3-segment routing) + round
    ST_AT_HEAD = 10,  // one q head: qk dots over the K plane, softmax, av, round
    ST_AT_OPROJ = 11, // o_proj rows band (k = nh·hd) + round + residual add
    ST_LOGITS = 12,   // tied-head rows band: nt → bf16 round → exact f32 widen
    ST_AT_PREP = 13,  // q/k norm + RoPE + exact-position K/V publication
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

// Durable control state for one admitted numerical ticket. A member callback
// executes exactly one stage and returns; only the fixed-team completion edge
// advances this cursor. Nothing needed after that return may live on a lane's
// C++ stack or in thread-local storage.
struct ProgramCursor {
    uint32_t phase = 0;
    uint32_t flags = 0;
    uint64_t outer = 0;
    uint64_t inner = 0;
};

enum : int {
    REQ_NONE = 0,
    REQ_CONV_LAYER = 2,
    REQ_ATTN_LAYER = 3,
    REQ_TOKEN_PASS = 4,
    // Complete Depthformer frame: projection, all codebooks/layers, integrated
    // sampler, and sampled-embedding recurrence under one native ticket.
    REQ_DEPTH_FRAME = 8,
    REQ_PREFILL = 13,
    // One conversation-owned detokenizer program writes directly into a
    // retained playback reservation. Dense AMX phases are explicit serial
    // resources; every separable phase is partitioned across the fixed team.
    REQ_AUDIO_DETOKENIZE = 14,
    // One retained PCM view through prepared resample, frontend, and Conformer
    // workspaces. Conformer GEMMs are fixed-team substages of this ticket and
    // never recurse through the bridge.
    REQ_AUDIO_ENCODE = 15,
};

static constexpr bool request_kind_valid(uint32_t kind) {
    switch (kind) {
    case REQ_CONV_LAYER:
    case REQ_ATTN_LAYER:
    case REQ_TOKEN_PASS:
    case REQ_DEPTH_FRAME:
    case REQ_PREFILL:
    case REQ_AUDIO_DETOKENIZE:
    case REQ_AUDIO_ENCODE:
        return true;
    default:
        return false;
    }
}

static constexpr bool logical_lane_count_valid(size_t lanes) {
    return lanes >= 1 && lanes <= static_cast<size_t>(MAX_WORKERS);
}

enum : uint32_t {
    SAMPLE_F32 = 1,
    SAMPLE_BF16 = 2,
};

enum : uint32_t {
    SAMPLE_PHASE_GREEDY = 0,
    SAMPLE_PHASE_MAXIMUM = 1,
    SAMPLE_PHASE_THRESHOLD = 2,
    SAMPLE_PHASE_EXP_SUM = 3,
    SAMPLE_PHASE_PICK = 4,
    SAMPLE_PHASE_DONE = 5,
};

struct SampleReq {
    const void *logits = nullptr;
    size_t count = 0;
    uint32_t dtype = 0;
    LfmSamplerConfigV1 config{};
    LfmPrngStateV1 *state = nullptr;
    uint32_t *out = nullptr;
    uint32_t phase = 0;
};

struct DepthReq {
    const uint16_t *hidden = nullptr;
    LfmSamplerConfigV1 sampler{};
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_tokens = nullptr;
    int completion_status = 0; // private deterministic route-fault seam
};

struct GemmReq {
    const uint16_t *a = nullptr;
    const void *rhs = nullptr;
    float *out = nullptr;
    const void *bias = nullptr;
    uint16_t *out_bf16 = nullptr;
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    size_t output_stride = 0;
    uint32_t rhs_layout = LFM_GEMM_RHS_KN;
    bool direct = false;
    bool bf16_epilogue = false;
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
    std::vector<uint16_t> x, h, xn, qkv_b, attn_b, t_b;
    std::vector<uint16_t> k_plane, v_plane, logits_b, din_b, df_b;
    std::vector<float> q_f, attn_f, proj_f;
    std::array<float, MAX_WORKERS> partials{};
    float inv_rms = 0.0f;
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
    Bf16Input x{};
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
    WeightBytes conv_w = nullptr;      // [H, K]
    const uint16_t *state_in = nullptr;// carried window in [H·(K-1)]
    uint16_t *state_out = nullptr;     // carried window out [H·(K-1)]
    size_t h = 0, k = 0;
    // planes
    uint16_t *xn = nullptr;    // normed input [H]
    uint16_t *projb = nullptr; // y bits [H]
    uint16_t *mid = nullptr;   // block output = MLP input [H]
    std::atomic<uint32_t> rs_bits{0};
};

// Attention-layer request: per-generation state (KV planes, rope tables, cursor)
// rides HERE — it lives in the per-cache objects, not the load-time table. The engine
// appends the step's K/V rows at `pos` and attends over pos+1 entries.
struct AttnReq {
    size_t layer = 0;
    Bf16Input x{};
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
    WeightBytes q_w = nullptr;     // [nh·hd, H]
    WeightBytes k_w = nullptr;     // [nkv·hd, H]
    WeightBytes v_w = nullptr;     // [nkv·hd, H]
    WeightBytes o_w = nullptr;     // [H, nh·hd]
    WeightBytes qn_w = nullptr;    // [hd]
    WeightBytes kn_w = nullptr;    // [hd]
    uint16_t *qkvb = nullptr;      // rounded q|k|v rows [(nh+2·nkv)·hd]
    uint16_t *ybits = nullptr;     // attention output per q head [nh·hd]
    float *att = nullptr;          // per-head score scratch [nh · max_ctx]
    Bf16Input x{};                 // residual input [H]
    uint16_t *mid = nullptr;       // block output = MLP input [H]
    uint16_t *k_plane = nullptr;
    uint16_t *v_plane = nullptr;
    const uint16_t *cos_row = nullptr;
    const uint16_t *sin_row = nullptr;
    size_t head_stride = 0, att_len = 0, max_ctx = 0;
    size_t h = 0, n_head = 0, n_kv = 0, hd = 0;
    float qk_eps = 0.0f;
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
    float *out_logits = nullptr;    // [vocab] final f32 linear-logits destination
    const LfmSamplerConfigV1 *sampler = nullptr;
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_token = nullptr;
    size_t lanes = 0;
};

enum : uint32_t {
    TOKEN_PROGRAM_EMBED = 0,
    TOKEN_PROGRAM_CONV = 1,
    TOKEN_PROGRAM_ATTN = 2,
    TOKEN_PROGRAM_FINAL_STATS = 3,
    TOKEN_PROGRAM_FINAL_NORM = 4,
    TOKEN_PROGRAM_LOGITS = 5,
    TOKEN_PROGRAM_SAMPLE = 6,
    TOKEN_PROGRAM_DONE = 7,
};

struct TokenProgram {
    Bf16Input hidden{};
    uint16_t *next = nullptr;
    uint32_t kind = TOKEN_PROGRAM_EMBED;
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
    std::vector<float> gu, scores, logits;
};

struct PrefillReq {
    PrefillWorkspace *workspace = nullptr;
    // Text IDs are tiny (at most PREFILL_ROWS) and caller storage is not part
    // of the ticket lease.  Keep the authoritative values in the pass record
    // so every continuation generation can reconstruct its input view without
    // retaining a caller stack address.
    std::array<uint32_t, PREFILL_ROWS> ids{};
    const uint16_t *provided_rows = nullptr;
    size_t provided_values = 0;
    size_t rows = 0;
    uint32_t embed_kind = 0;
    const LfmLayerState *states = nullptr;
    size_t n_states = 0;
    size_t pos = 0;
    const uint16_t *cos_base = nullptr;
    const uint16_t *sin_base = nullptr;
    size_t rope_len = 0;
    uint16_t *out_hidden = nullptr;
    size_t out_hidden_len = 0;
    LfmSamplerConfigV1 sampler{};
    bool sample = false;
    LfmPrngStateV1 *sample_state = nullptr;
    uint32_t *out_token = nullptr;
    /* Test-only dual-head readout target. Null on every production pass. */
    LfmListenReadoutForTest *readout = nullptr;
    size_t lanes = 0;
};

struct DetokenizerReq {
    LfmAudioDetokenizerState *state = nullptr;
    const uint32_t *codes = nullptr;
    float *pcm = nullptr;
    size_t capacity = 0;
    float *detokenizer_pcm = nullptr;
    size_t detokenizer_capacity = 0;
    LfmResamplerStream *resampler_stream = nullptr;
    size_t *out_samples = nullptr;
    LfmAudioDetokenizerProgram program{};
    bool flush = false;
    bool resample_pending = false;
    int completion_status = 0; // private deterministic route-fault seam
};

struct AudioReq {
    LfmAudioEncodePassV2 pass{};
    LfmConformerProgram conformer{};
    uint64_t adapted_values = 0;
    uint64_t frames = 0;
    uint32_t phase = 0;
};

// Each admitted ticket owns one activation/sampling scratch bank until its
// exact CQ record is consumed. The lane team remains single-pass, so dispatch
// swaps precisely one bank onto the stage board; a queued follow-on never
// aliases the completing ticket's values. These are activation buffers only --
// resident weights remain borrowed byte views and are never copied here.
struct ScratchBank {
    std::vector<float> sc_partials;
    std::vector<uint16_t> sc_xn, sc_t;
    std::vector<uint16_t> sc_projb, sc_mid;
    std::vector<float> at_att;
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
    DepthScratch depth;
};

struct Engine;
constexpr size_t BLOCK_DOMAIN_COUNT = 2;
constexpr uint32_t BLOCK_LANES = 4;
constexpr uint32_t GRID_LANES = BLOCK_DOMAIN_COUNT * BLOCK_LANES;

// The production audio route is deliberately smaller than a graph runtime:
// three trusted coarse nodes and one total immutable outcome table.
enum : uint32_t {
    AUDIO_ROUTE_TOKEN = 0,
    AUDIO_ROUTE_DEPTH = 1,
    AUDIO_ROUTE_DETOKENIZER = 2,
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
    AUDIO_TOKEN_VALUE = 0,
    AUDIO_TOKEN_END = 1,
    AUDIO_TOKEN_INVALID = 2,
};
enum : uint32_t {
    AUDIO_ROUTE_GENERATION = 0,
    AUDIO_ROUTE_ENCODE = 1,
    AUDIO_ROUTE_PREFILL = 2,
    AUDIO_ROUTE_CONTROL = 3,
};

enum : uint32_t {
    PASS_SLOT_FREE = 0,
    PASS_SLOT_RESERVED = 1,
    PASS_SLOT_SUBMITTING = 2,
    PASS_SLOT_SUBMITTED = 3,
    PASS_SLOT_RUNNING = 4,
    PASS_SLOT_COMPLETING = 5,
    PASS_SLOT_COMPLETE = 6,
};

enum : uint32_t {
    PASS_RESULT_NONE = 0,
    PASS_RESULT_TEXT_TOKEN = 1,
    PASS_RESULT_FRAME = 2,
};

constexpr size_t PASS_RESULT_CAPACITY = 8;

struct FlashkernRequest {
    kc_ticket_id ticket{};
    kc_ticket_id parent{};
    uint64_t conversation_id = 0;
    uint64_t epoch = 0;
    uint64_t lease_generation = 0;
    uint32_t slot = 0;
    uint32_t operation = REQ_NONE;
};

struct FlashkernCompletion {
    kc_ticket_id ticket{};
    kc_ticket_id parent{};
    uint64_t conversation_id = 0;
    uint64_t epoch = 0;
    uint64_t lease_generation = 0;
    uint32_t slot = 0;
    int32_t status = 0;
    uint32_t result_kind = PASS_RESULT_NONE;
    uint32_t result_count = 0;
    uint32_t results[PASS_RESULT_CAPACITY]{};
};

struct AudioRouteInstance {
    Engine *engine = nullptr;
    uint32_t lease_index = UINT32_MAX;
    uint64_t lease_generation = 0;
    kc_ticket_id ticket{};
    uint32_t kind = AUDIO_ROUTE_GENERATION;
    uint32_t node = AUDIO_ROUTE_TOKEN;
    uint64_t depth_id = 0;
    DepthPlan *depth = nullptr;
    DepthReq depth_req{};
    BackbonePlan *model = nullptr;
    uint64_t model_id = 0;
    TokenReq token_req{};
    PrefillReq prefill_req{};
    AudioReq audio_req{};
    uint64_t *adapted_values = nullptr;
    DetokenizerReq detokenizer_req{};
    const LfmRouteEpoch *epoch = nullptr;
    uint64_t expected_epoch = 0;
    LfmAudioRouteResult *result = nullptr;
    LfmAudioRouteNotify notify = nullptr;
    void *notify_context = nullptr;
    bool terminal_after_token = false;
    bool decode_detokenizer = false;
    LfmTokenCommitRecord commit{};
    uint32_t *token_completed = nullptr;
    FlashkernCompletion pass_completion{};
    int status = -EINPROGRESS;
};

using AudioRouteBroker =
    kc::PermitBroker<AudioRouteInstance, ROUTE_CAPACITY, ENGINE_CACHELINE>;

using EngineExecutor =
    kc::TeamExecutor<FlashkernRequest, FlashkernCompletion, PASS_CAPACITY,
                     ENGINE_CACHELINE>;

struct PassSlot;
struct PassContinuationPermit;
using PassContinuation = void (*)(PassContinuationPermit *,
                                  const FlashkernCompletion &, void *);

struct alignas(ENGINE_CACHELINE) PassSlot {
    Engine *engine = nullptr;
    uint32_t index = 0;
    FlashkernRequest submission{};
    FlashkernCompletion completion{};
    int request = REQ_NONE;
    uint64_t context_id = 0;
    PassContinuation continuation = nullptr;
    void *continuation_context = nullptr;
    ProgramCursor program{};
    Stage stage{};
    // Numerical state that survives a team return belongs to the ticket, not
    // the engine thread or a C++ call stack.  The dispatcher mounts these views
    // only for the lifetime of this slot generation.
    Pass pass{};
    ScPass sc{};
    AtPass at{};

    ConvReq conv;
    AttnReq attn;
    SampleReq sample;
    DepthReq depth_req;
    GemmReq gemm;
    BackbonePlan *model = nullptr;
    DepthPlan *depth = nullptr;
    TokenReq tok;
    TokenProgram token_program{};
    PrefillReq prefill;
    DetokenizerReq detokenizer;
    AudioReq audio;
    ScratchBank scratch;
};
static_assert(alignof(PassSlot) >= ENGINE_CACHELINE,
              "pass-slot lease words require cache-line-aligned elements");
static_assert(sizeof(PassSlot) % ENGINE_CACHELINE == 0,
              "pass-slot array stride must preserve lease-word isolation");
using PassSlotPool =
    kc::SlotPool<PassSlot, PASS_CAPACITY, ENGINE_CACHELINE>;

/* Stack-scoped authority for the exact slot whose CQ record triggered a
 * continuation. The type never crosses a header or the product ABI. Keeping
 * the slot RESERVED under the same generation makes completion retirement
 * atomic with respect to later route generations. */
struct PassContinuationPermit {
    Engine *engine = nullptr;
    PassSlot *slot = nullptr;
    uint64_t generation = 0;
    bool consumed = false;
};

static uint32_t slot_state(const PassSlot *slot);
static uint64_t slot_generation(const PassSlot *slot);
static bool transition_slot(PassSlot *slot, uint64_t generation,
                            uint32_t from, uint32_t to);

struct LfmEngineSnapshotV1 {
    uint32_t size;
    uint32_t abi_version;
    uint64_t pass_submissions;
    uint64_t pass_completions;
    uint64_t bridge_dispatches;
    uint64_t dispatch_wakes;
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
    uint64_t route_admission_deferrals;
};

/* Immutable identity of the exact numerical generation mounted on the fixed
 * team. A request selector alone is not enough: token, prefill, Depthformer,
 * Conformer, and sampling tickets each contain multiple independently
 * supervised phases. The descriptor is copied before the deadline is armed so
 * a fatal callback never dereferences a live pass slot. */
struct TeamWorkDescriptor {
    uint32_t request = REQ_NONE;
    uint32_t stage = ST_IDLE;
    uint32_t program_kind = 0;
    uint32_t program_phase = 0;
    uint32_t program_flags = 0;
    uint32_t reserved = 0;
    uint64_t program_outer = 0;
    uint64_t program_inner = 0;
    uint64_t shape0 = 0;
    uint64_t shape1 = 0;
    uint64_t shape2 = 0;
};
static_assert(std::is_trivially_copyable_v<TeamWorkDescriptor>);

/* Reserved crash evidence for one hard quorum failure. The capsule is wholly
 * pointer-free so platform crash reporting can preserve it without chasing an
 * engine, route, conversation, or scratch lifetime. It is never a CQ record:
 * timeout poisons the process and therefore cannot publish ordinary progress. */
struct alignas(ENGINE_CACHELINE) TeamFatalCapsule {
    uint32_t size = sizeof(TeamFatalCapsule);
    uint32_t abi_version = 2;
    uint32_t request = REQ_NONE;
    uint32_t stage = ST_IDLE;
    uint32_t program_kind = 0;
    uint32_t program_phase = 0;
    uint32_t program_flags = 0;
    int32_t quorum_status = 0;
    kc_ticket_id workflow{};
    kc_ticket_id pass{};
    kc_ticket_id deadline{};
    uint64_t conversation_id = 0;
    uint64_t epoch = 0;
    uint64_t scope_generation = 0;
    uint64_t team_generation = 0;
    uint64_t program_outer = 0;
    uint64_t program_inner = 0;
    uint64_t shape0 = 0;
    uint64_t shape1 = 0;
    uint64_t shape2 = 0;
    uint64_t expected_mask = 0;
    uint64_t entered_mask = 0;
    uint64_t returned_mask = 0;
    uint64_t never_entered_mask = 0;
    uint64_t entered_not_returned_mask = 0;
    uint64_t armed_ns = 0;
    uint64_t hard_budget_ns = 0;
    uint64_t elapsed_ns = 0;
    uint64_t deadline_event_sequence = 0;
    uint64_t scheduled_arm_generation = 0;
    uint64_t current_arm_generation = 0;
};
static_assert(std::is_trivially_copyable_v<TeamFatalCapsule>);

constexpr uint64_t TEAM_FATAL_SINK_MAGIC = UINT64_C(0x314c415441464b46);

struct TeamFaultTestConfig {
    uint32_t member = 0;
    uint32_t point = 0;
    const char *fatal_path = nullptr;
};

using EngineSupervisor =
    kc::TeamSupervisor<FlashkernRequest, TeamWorkDescriptor,
                       TeamFatalCapsule, ENGINE_CACHELINE>;

struct Engine {
    // Whole-pass programs use the engine records below; continuation-mounted
    // programs bind their PassSlot-owned records through these pointers. This
    // keeps run_tile shared while making cross-generation ownership explicit.
    Pass pass;
    Stage stage;
    Pass *pass_view = &pass;
    ScPass *sc_view = nullptr;
    AtPass *at_view = nullptr;

    // One bounded kcoro pool owns both orchestration frames and the logical
    // lane continuations. A lane identity is stable for quorum accounting,
    // but its physical worker is not: any free eligible worker may run its
    // uninterrupted numerical tile.
    kc_runtime_t *control_runtime = nullptr;
    EngineExecutor executor;
    AudioRouteBroker routes;
    EngineSupervisor supervisor;
    int n_workers = 0;
    int control_started = 0;
    int executor_started = 0;
    uint32_t block_count = 1;
    uint32_t lanes_total = 1;
    int cur_req = REQ_NONE;
    std::atomic<bool> retire{false};
    kc_ticket_id mailbox_capacity_identity{};
    kc_ticket_id route_capacity_identity{};
    PassSlotPool slots;
    PassSlot *active_slot = nullptr;
    FlashkernRequest active_submission{};
    /* Flashkern retains only the model-specific descriptor used for diagnostics
     * and budget selection. kc::TeamSupervisor owns deadline lineage,
     * terminal arbitration, quorum evidence, and fatal storage. */
    TeamWorkDescriptor supervised_work{};
    bool manual_deadlines = false;
    bool hard_timeout_probe = false;
    // Numerical publishers take a bounded, single-pass lease. Plan mutation
    // closes admission and succeeds only when every already-admitted publisher
    // has retired. No producer retries or waits for exclusivity.
    kc::AdmissionGate<PASS_PUBLISHER_CAPACITY> pass_admission;
    std::atomic<bool> pass_claimed{false};
    std::atomic<int> active_status{0};
    kc::TicketSource tickets;
    std::atomic<uint64_t> pass_submissions{0};
    std::atomic<uint64_t> pass_completions{0};
    std::atomic<uint32_t> attention_qkv_capacity{0};
    std::atomic<uint32_t> attention_y_capacity{0};
    std::atomic<uint32_t> attention_score_capacity{0};
    std::atomic<uint32_t> pass_slots_live{0};
    std::atomic<uint32_t> max_pass_slots_live{0};
    std::atomic<uint64_t> continuation_submissions{0};
    std::atomic<uint64_t> route_dispatches{0};
    std::atomic<uint64_t> audio_encode_passes{0};
    std::atomic<int> test_audio_route_depth_status{0};
    std::atomic<int> test_audio_route_detokenizer_status{0};
    /* Owner-published control evidence for the fatal outer truth-gate
     * watchdog. This is retained setup-time state, not a copied payload
     * snapshot and not part of the product ABI. */
    LfmEngineDiagnosticState diagnostics;

    ConvReq conv;  // conv-layer request payload
    AttnReq attn;  // attention-layer request payload
    SampleReq sample; // pointer-only logits/state handoff; policy is inline
    DepthReq depth_req; // complete typed Depthformer frame request
    GemmReq gemm; // borrowed matrices and exclusive destination
    ScPass sc;     // shortconv stage pointers
    AtPass at;     // attention stage pointers

    // Immutable model plans coexist. One in-flight ticket selects `model`; the
    // physical lane team and scratch arena remain singular.
    std::vector<std::unique_ptr<BackbonePlan>> models;
    BackbonePlan *model = nullptr;
    uint64_t model_seq = 0;
    TokenReq tok; // token-pass request payload

    // Persistent scratch backing is sized before numerical admission and is
    // swapped with the active ticket's private activation bank.
    std::vector<float> sc_partials;
    std::vector<uint16_t> sc_xn, sc_t;
    // shortconv planes (ctx build): see ScPass.
    std::vector<uint16_t> sc_projb, sc_mid;
    // attention planes (ctx build): qkv f32/bits [(nh+2·nkv)·hd], y bits [nh·hd],
    // per-head score scratch [nh · max_ctx] f32
    std::vector<float> at_att;
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
    DepthScratch depth_scratch;
};

static uint32_t slot_state(const PassSlot *slot) {
    return slot && slot->engine
        ? slot->engine->slots.state(slot->index)
        : PASS_SLOT_FREE;
}

static uint64_t slot_generation(const PassSlot *slot) {
    return slot && slot->engine
        ? slot->engine->slots.generation(slot->index)
        : 0;
}

static bool transition_slot(PassSlot *slot, uint64_t generation,
                            uint32_t from, uint32_t to) {
    if (!slot || !slot->engine) return false;
    return slot->engine->slots.transition(
        {.index = slot->index, .generation = generation}, from, to);
}

static void publish_engine_diagnostics(Engine *engine) {
    if (!engine) return;
    const kc::TeamExecutorSnapshot executor = engine->executor.snapshot();
    LfmEngineDiagnosticState &state = engine->diagnostics;
    state.bridge_activations.store(executor.activations,
                                   std::memory_order_relaxed);
    state.team_generation.store(executor.generation,
                                std::memory_order_relaxed);
    state.bridge_phase.store(executor.phase, std::memory_order_relaxed);
    state.request.store(static_cast<uint32_t>(engine->cur_req),
                        std::memory_order_relaxed);
    state.stage.store(engine->supervised_work.stage,
                      std::memory_order_relaxed);
    state.program_phase.store(engine->supervised_work.program_phase,
                              std::memory_order_relaxed);
    state.bridge_valid.store(engine->active_slot ? 1u : 0u,
                             std::memory_order_relaxed);
    state.active_status.store(
        engine->active_status.load(std::memory_order_acquire),
        std::memory_order_relaxed);
    state.publications.fetch_add(1, std::memory_order_release);
}


static void notify_route_broker(void *context) {
    Engine *engine = static_cast<Engine *>(context);
    if (engine) engine->routes.notify();
}

static void publish_mailbox_capacity_edge(void *context,
                                          const kc_ticket_id *identity) {
    Engine *engine = static_cast<Engine *>(context);
    if (!engine || !identity ||
        !kc::ticket_equal(*identity, engine->mailbox_capacity_identity)) {
        std::abort();
    }
    engine->routes.notify();
}

static void publish_route_capacity_edge(void *context,
                                        const kc_ticket_id *identity) {
    Engine *engine = static_cast<Engine *>(context);
    if (!engine || !identity ||
        !kc::ticket_equal(*identity, engine->route_capacity_identity)) {
        std::abort();
    }
    engine->routes.notify();
}

static BackbonePlan *find_model(Engine *e, uint64_t id) {
    for (const std::unique_ptr<BackbonePlan> &model : e->models)
        if (model->id == id) return model.get();
    return nullptr;
}

static void update_slot_high_water(Engine *e, uint32_t live) {
    if (live >= PASS_CAPACITY) {
        e->max_pass_slots_live.store(static_cast<uint32_t>(PASS_CAPACITY),
                                     std::memory_order_relaxed);
        return;
    }
    uint32_t high = e->max_pass_slots_live.load(std::memory_order_relaxed);
    if (high < live)
        e->max_pass_slots_live.compare_exchange_strong(
            high, live, std::memory_order_relaxed, std::memory_order_relaxed);
}

static void update_capacity_high_water(std::atomic<uint32_t> *counter,
                                       size_t capacity) {
    if (capacity > UINT32_MAX) capacity = UINT32_MAX;
    const uint32_t requested = (uint32_t)capacity;
    if (counter->load(std::memory_order_relaxed) < requested)
        counter->store(requested, std::memory_order_relaxed);
}

static bool enter_pass_admission(Engine *e) {
    return e && e->pass_admission.enter() == 0;
}

static void leave_pass_admission(Engine *e) {
    if (!e) std::abort();
    e->pass_admission.leave();
}

static void clear_slot_request(PassSlot *slot) {
    slot->submission = {};
    slot->completion = {};
    slot->request = REQ_NONE;
    slot->context_id = 0;
    slot->continuation = nullptr;
    slot->continuation_context = nullptr;
    slot->program = {};
    slot->stage.next.store(0, std::memory_order_relaxed);
    slot->stage.kind = ST_IDLE;
    slot->stage.count = 0;
    slot->stage.chunk = 0;
    slot->conv = {};
    slot->attn = {};
    slot->sample = {};
    slot->depth_req = {};
    slot->gemm = {};
    slot->model = nullptr;
    slot->depth = nullptr;
    slot->tok = {};
    slot->token_program = {};
    slot->prefill = {};
    slot->detokenizer = {};
    slot->audio = {};
}

static PassSlot *reserve_pass_slot(Engine *e) {
    if (!e) return nullptr;
    if (!enter_pass_admission(e)) return nullptr;
    const PassSlotPool::Lease lease = e->slots.acquire(
        PASS_SLOT_RESERVED,
        [](PassSlot &slot, uint32_t) { clear_slot_request(&slot); });
    if (!lease) {
        leave_pass_admission(e);
        return nullptr;
    }
    PassSlot *slot = e->slots.get(lease);
    if (!slot) std::abort();
    const uint32_t live =
        e->pass_slots_live.fetch_add(1, std::memory_order_acq_rel) + 1;
    update_slot_high_water(e, live);
    return slot;
}

static bool release_pass_slot(PassSlot *slot, uint64_t generation) {
    if (!slot || generation == 0) return false;
    Engine *e = slot->engine;
    const PassSlotPool::Lease lease = {
        .index = slot->index,
        .generation = generation,
    };
    if (!e->slots.begin_release(lease, PASS_SLOT_RESERVED) &&
        !e->slots.begin_release(lease, PASS_SLOT_COMPLETE)) {
        return false;
    }
    clear_slot_request(slot);
    e->pass_slots_live.fetch_sub(1, std::memory_order_acq_rel);
    /* FREE is published only after the old record and live accounting retire.
     * Admission release then publishes the correlated capacity edge. */
    e->slots.finish_release(lease);
    leave_pass_admission(e);
    return true;
}

static void swap_depth_scratch(DepthScratch &left, DepthScratch &right) {
    left.x.swap(right.x);
    left.h.swap(right.h);
    left.xn.swap(right.xn);
    left.qkv_b.swap(right.qkv_b);
    left.attn_b.swap(right.attn_b);
    left.t_b.swap(right.t_b);
    left.k_plane.swap(right.k_plane);
    left.v_plane.swap(right.v_plane);
    left.logits_b.swap(right.logits_b);
    left.din_b.swap(right.din_b);
    left.df_b.swap(right.df_b);
    left.q_f.swap(right.q_f);
    left.attn_f.swap(right.attn_f);
    left.proj_f.swap(right.proj_f);
    left.partials.swap(right.partials);
    std::swap(left.inv_rms, right.inv_rms);
}

static void swap_scratch(Engine *e, ScratchBank &scratch) {
    e->sc_partials.swap(scratch.sc_partials);
    e->sc_xn.swap(scratch.sc_xn);
    e->sc_t.swap(scratch.sc_t);
    e->sc_projb.swap(scratch.sc_projb);
    e->sc_mid.swap(scratch.sc_mid);
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
    swap_depth_scratch(e->depth_scratch, scratch.depth);
}

static void activate_slot(Engine *e, PassSlot *slot) {
    swap_scratch(e, slot->scratch);
    e->active_slot = slot;
    e->model = slot->model;
    e->active_depth = slot->depth;
    e->conv = slot->conv;
    e->attn = slot->attn;
    e->sample = slot->sample;
    e->depth_req = slot->depth_req;
    e->gemm = slot->gemm;
    e->tok = slot->tok;
    e->pass_view = &e->pass;
    e->sc_view = &e->sc;
    e->at_view = &e->at;
    if (slot->request == REQ_CONV_LAYER) {
        e->pass_view = &slot->pass;
        e->sc_view = &slot->sc;
    } else if (slot->request == REQ_ATTN_LAYER) {
        e->pass_view = &slot->pass;
        e->sc_view = &slot->sc;
        e->at_view = &slot->at;
    } else if (slot->request == REQ_TOKEN_PASS) {
        e->pass_view = &slot->pass;
        e->sc_view = &slot->sc;
        e->at_view = &slot->at;
    }
}

static void deactivate_slot(Engine *e, PassSlot *slot) {
    e->active_slot = nullptr;
    e->model = nullptr;
    e->active_depth = nullptr;
    e->pass_view = &e->pass;
    e->sc_view = &e->sc;
    e->at_view = &e->at;
    swap_scratch(e, slot->scratch);
}

// Plan installation/removal and all-slot sizing close publisher admission.
// They never wait: contention rejects the claim and the caller may submit a
// later control command after the in-flight ticket has produced its callback.
class PlanClaim {
  public:
    explicit PlanClaim(Engine *engine) : engine_(engine) {
        bool expected = false;
        if (!engine_ || !engine_->pass_claimed.compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
            return;
        }
        held_ = engine_->pass_admission.try_seal() == 0;
        if (!held_) {
            engine_->pass_claimed.store(false, std::memory_order_release);
        }
    }

    ~PlanClaim() {
        if (!held_) return;
        if (engine_->pass_admission.active() != 0)
            std::abort();
        engine_->pass_claimed.store(false, std::memory_order_release);
        const int status = engine_->pass_admission.unseal();
        if (status != 0 && status != -ECANCELED) std::abort();
    }

    explicit operator bool() const { return held_; }
    PlanClaim(const PlanClaim &) = delete;
    PlanClaim &operator=(const PlanClaim &) = delete;

  private:
    Engine *engine_ = nullptr;
    bool held_ = false;
};

static void qk_norm_row(const uint16_t *x, WeightBytes w, uint16_t *out,
                        size_t hd, float eps);
static void rope_slow_row(uint16_t *x, const uint16_t *cos_row,
                          const uint16_t *sin_row, size_t hd);

// ---- tile bodies (identical math to decode.rs) ----------------------------------------
static void run_tile(uint32_t kind, uint32_t idx, const Stage *st, Engine *e) {
    Pass *p = e->pass_view;
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
        lfm_bf16_gemv_pair_swiglu_bf16(
            p->xn, weight_offset(p->w1, r0 * p->h),
            weight_offset(p->w3, r0 * p->h), p->t + r0, r1 - r0,
            p->h);
        break;
    }
    case ST_DOWN: {
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < p->h ? r0 + st->chunk : p->h;
        if (r1 <= r0) break;
        const size_t n = r1 - r0;
        lfm_bf16_gemv_rne_add_bf16(
            p->t, weight_offset(p->w2, r0 * p->i), p->x + r0,
            p->out + r0, n, p->i);
        break;
    }
    case ST_SC_NORM: {
        // Contiguous band — elementwise, so banding never changes a cell's value.
        ScPass *c = e->sc_view;
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
        ScPass *c = e->sc_view;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < c->h ? r0 + st->chunk : c->h;
        if (r1 <= r0) break;
        lfm_shortconv_project_update_bf16(
            c->xn, c->in_w, c->state_in, c->conv_w, c->projb,
            c->state_out, c->h, r0, r1 - r0, c->k);
        break;
    }
    case ST_SC_OUTPROJ: {
        // rb(out_proj) then rb(+residual) — the linear_forward ladder, band-wise.
        ScPass *c = e->sc_view;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < c->h ? r0 + st->chunk : c->h;
        if (r1 <= r0) break;
        lfm_bf16_gemv_rne_add_bf16(
            c->projb, weight_offset(c->out_w, r0 * c->h),
            c->x.offset(r0).data(), c->mid + r0, r1 - r0, c->h);
        break;
    }
    case ST_AT_QKV: {
        // One band over the concatenated q|k|v projection row space; segments route to
        // their own weight matrices. Each row is the same linear_forward ladder the
        // Contract: NT dot in f32, followed by one bf16 storage round.
        AtPass *a = e->at_view;
        size_t qrows = a->n_head * a->hd, kvrows = a->n_kv * a->hd;
        size_t total = qrows + 2 * kvrows;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < total ? r0 + st->chunk : total;
        size_t r = r0;
        while (r < r1) {
            WeightBytes w;
            size_t seg0, seglen;
            if (r < qrows) {
                w = a->q_w;
                seg0 = 0;
                seglen = qrows;
            } else if (r < qrows + kvrows) {
                w = a->k_w;
                seg0 = qrows;
                seglen = kvrows;
            } else {
                w = a->v_w;
                seg0 = qrows + kvrows;
                seglen = kvrows;
            }
            size_t seg_end = seg0 + seglen;
            size_t stop = r1 < seg_end ? r1 : seg_end;
            lfm_bf16_gemv_rne_bf16(
                e->sc_view->xn, weight_offset(w, (r - seg0) * a->h),
                a->qkvb + r, stop - r, a->h);
            r = stop;
        }
        break;
    }
    case ST_AT_PREP: {
        // Heads are independent until attention reads the published K/V rows.
        // Keeping this as an ordinary team generation removes the former
        // lane-zero head loop and makes the final team return the publication
        // edge for the next stage.
        AtPass *a = e->at_view;
        const size_t head = idx;
        uint16_t *qrows = a->qkvb;
        uint16_t *krows = a->qkvb + a->n_head * a->hd;
        const uint16_t *vrows = a->qkvb +
                                (a->n_head + a->n_kv) * a->hd;
        if (head < a->n_head) {
            uint16_t *q = qrows + head * a->hd;
            qk_norm_row(q, a->qn_w, q, a->hd, a->qk_eps);
            rope_slow_row(q, a->cos_row, a->sin_row, a->hd);
        }
        if (head < a->n_kv) {
            uint16_t *k = krows + head * a->hd;
            qk_norm_row(k, a->kn_w, k, a->hd, a->qk_eps);
            rope_slow_row(k, a->cos_row, a->sin_row, a->hd);
            std::memcpy(a->k_plane + head * a->head_stride +
                            (a->att_len - 1) * a->hd,
                        k, a->hd * sizeof(uint16_t));
            std::memcpy(a->v_plane + head * a->head_stride +
                            (a->att_len - 1) * a->hd,
                        vrows + head * a->hd,
                        a->hd * sizeof(uint16_t));
        }
        break;
    }
    case ST_AT_HEAD: {
        // One q head, exactly attn_decode_bf16's per-head body: widen q, dots over the
        // K plane (grouped kv head), scaled softmax, weighted V sum, one round out.
        AtPass *a = e->at_view;
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
        // Preserve the accepted linear-forward plus residual-add ladder.
        AtPass *a = e->at_view;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < a->h ? r0 + st->chunk : a->h;
        if (r1 <= r0) break;
        size_t kdim = a->n_head * a->hd;
        lfm_bf16_gemv_rne_add_bf16(
            a->ybits, weight_offset(a->o_w, r0 * kdim),
            a->x.offset(r0).data(), a->mid + r0, r1 - r0, kdim);
        break;
    }
    case ST_LOGITS: {
        // linear_logits ladder EXACTLY: M==1 GEMV rows, f32 accumulate, RAW f32
        // out. The pinned Rust head (linear_logits -> Bf16GemmNt) emits the
        // kernel's f32 directly — the bf16 storage round this stage used to add
        // was an EXTRA round the reference never performs, and it is what
        // flipped the perf-chain hash when the head was first absorbed. Same
        // kernel, same per-row K-reduction (row banding cannot reorder a row's
        // accumulation), no round: bit-identical to the accepted head path — the
        // PERF oracle is the proof.
        Engine *ee = e;
        const TokenReq *t = ee->active_slot &&
                                  ee->active_slot->request == REQ_TOKEN_PASS
            ? &ee->active_slot->tok
            : &ee->tok;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < ee->model->vocab
                        ? r0 + st->chunk
                        : ee->model->vocab;
        if (r1 <= r0) break;
        // The public logit span is already the terminal value. Accumulate into
        // it directly when present instead of materializing the same rows in
        // engine scratch and copying them after the leaf returns. Token-only
        // production recurrence retains the preallocated private plane because
        // the sampler still needs the complete vocabulary after reconvergence.
        float *acc = t->out_logits ? t->out_logits + r0
                                   : ee->tk_logf.data() + r0;
        lfm_bf16_gemm_nt_f32(t->out_hidden,
                             weight_offset(ee->model->embed_w, r0 * ee->model->h), acc, 1,
                             (int)(r1 - r0), (int)ee->model->h);
        break;
    }
    default:
        break;
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

static inline void depth_gemv_rne(const LfmDepthBufferV1 &weight,
                                  const uint16_t *x, uint16_t *out,
                                  size_t rows, size_t cols, uint32_t lane,
                                  uint32_t lanes) {
    size_t begin = 0, end = 0;
    depth_band(rows, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_gemv_rne_bf16(
            x, depth_bytes(weight) + begin * cols * sizeof(uint16_t),
            out + begin, end - begin, cols);
}

static inline void depth_gemv_rne_add(
    const LfmDepthBufferV1 &weight, const uint16_t *x,
    const uint16_t *residual, uint16_t *out, size_t rows, size_t cols,
    uint32_t lane, uint32_t lanes) {
    size_t begin = 0, end = 0;
    depth_band(rows, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_gemv_rne_add_bf16(
            x, depth_bytes(weight) + begin * cols * sizeof(uint16_t),
            residual + begin, out + begin, end - begin, cols);
}

static inline void depth_gemv_pair_swiglu(
    const LfmDepthBufferV1 &gate, const LfmDepthBufferV1 &up,
    const uint16_t *x, uint16_t *out, size_t rows, size_t cols,
    uint32_t lane, uint32_t lanes) {
    size_t begin = 0, end = 0;
    depth_band(rows, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_gemv_pair_swiglu_bf16(
            x, depth_bytes(gate) + begin * cols * sizeof(uint16_t),
            depth_bytes(up) + begin * cols * sizeof(uint16_t), out + begin,
            end - begin, cols);
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

static void configure_serial_stage(Stage *stage);
static void run_sample_pass(Engine *e, uint32_t lane);
static bool advance_sample_program(Engine *e, PassSlot *slot);

enum : uint32_t {
    DEPTH_PHASE_PROJECT = 0,
    DEPTH_PHASE_CODEBOOK_ADD = 1,
    DEPTH_PHASE_OP_NORM_SUM = 2,
    DEPTH_PHASE_OP_NORM_APPLY = 3,
    DEPTH_PHASE_QKV = 4,
    DEPTH_PHASE_QK_PREP = 5,
    DEPTH_PHASE_ATTN = 6,
    DEPTH_PHASE_ATTN_ROUND = 7,
    DEPTH_PHASE_OUT = 8,
    DEPTH_PHASE_FFN_NORM_SUM = 9,
    DEPTH_PHASE_FFN_NORM_APPLY = 10,
    DEPTH_PHASE_FFN_GATE = 11,
    DEPTH_PHASE_FFN_DOWN = 12,
    DEPTH_PHASE_HEAD_NORM_SUM = 13,
    DEPTH_PHASE_HEAD_NORM_APPLY = 14,
    DEPTH_PHASE_HEAD_LOGITS = 15,
    DEPTH_PHASE_SAMPLE = 16,
    DEPTH_PHASE_EMBED = 17,
    DEPTH_PHASE_DONE = 18,
};

static void initialize_depth_program(PassSlot *slot) {
    if (!slot || slot->request != REQ_DEPTH_FRAME || !slot->depth) {
        std::abort();
    }
    slot->program.phase = DEPTH_PHASE_PROJECT;
    slot->program.outer = 0;
    slot->program.inner = 0;
    configure_serial_stage(&slot->stage);
}

static void depth_norm_sum_stage(Engine *e, uint32_t lane,
                                 const uint16_t *values) {
    DepthPlan &plan = *e->active_depth;
    DepthScratch &scratch = e->depth_scratch;
    size_t begin = 0, end = 0;
    depth_band(plan.dim, lane, e->lanes_total, &begin, &end);
    scratch.partials[lane] = end > begin
        ? lfm_bf16_sumsq_f32(values + begin,
                             static_cast<int>(end - begin))
        : 0.0f;
}

static void depth_norm_apply_stage(Engine *e, uint32_t lane,
                                   const uint16_t *values,
                                   const LfmDepthBufferV1 &weight,
                                   uint16_t *out) {
    DepthPlan &plan = *e->active_depth;
    DepthScratch &scratch = e->depth_scratch;
    size_t begin = 0, end = 0;
    depth_band(plan.dim, lane, e->lanes_total, &begin, &end);
    if (end > begin) {
        lfm_bf16_rmsnorm(
            values + begin,
            depth_bytes(weight) + begin * sizeof(uint16_t), out + begin,
            static_cast<int>(end - begin), scratch.inv_rms);
    }
}

static void run_depth_program_stage(Engine *e, uint32_t lane,
                                    PassSlot *slot) {
    DepthPlan &plan = *slot->depth;
    DepthScratch &scratch = e->depth_scratch;
    const DepthReq &request = slot->depth_req;
    const uint32_t lanes = e->lanes_total;
    const size_t codebook = static_cast<size_t>(slot->program.outer);
    const size_t layer = static_cast<size_t>(slot->program.inner);
    size_t begin = 0, end = 0;

    switch (slot->program.phase) {
    case DEPTH_PHASE_PROJECT: {
        depth_gemv({reinterpret_cast<uintptr_t>(plan.depth_linear_w),
                    plan.codebooks * plan.dim * plan.backbone_dim},
                   request.hidden, scratch.proj_f.data(),
                   plan.codebooks * plan.dim, plan.backbone_dim, lane, lanes);
        depth_band(plan.codebooks * plan.dim, lane, lanes, &begin, &end);
        if (end > begin) {
            lfm_bf16_bias_add_f32(
                scratch.proj_f.data() + begin,
                plan.depth_linear_b + begin * sizeof(uint16_t), end - begin);
            lfm_f32_to_bf16(scratch.proj_f.data() + begin,
                            scratch.din_b.data() + begin,
                            static_cast<int>(end - begin));
        }
        depth_band(plan.dim, lane, lanes, &begin, &end);
        std::fill(scratch.df_b.begin() + begin, scratch.df_b.begin() + end,
                  static_cast<uint16_t>(0));
        return;
    }
    case DEPTH_PHASE_CODEBOOK_ADD:
        depth_band(plan.dim, lane, lanes, &begin, &end);
        if (end > begin) {
            lfm_bf16_add(scratch.din_b.data() + codebook * plan.dim + begin,
                         scratch.df_b.data() + begin,
                         scratch.x.data() + begin,
                         static_cast<int>(end - begin));
        }
        return;
    case DEPTH_PHASE_OP_NORM_SUM:
        depth_norm_sum_stage(e, lane, scratch.x.data());
        return;
    case DEPTH_PHASE_OP_NORM_APPLY:
        depth_norm_apply_stage(e, lane, scratch.x.data(),
                               plan.layers[layer].op_norm,
                               scratch.xn.data());
        return;
    case DEPTH_PHASE_QKV: {
        const size_t rows = plan.dim + 2 * plan.kv_heads * plan.hd;
        depth_gemv_rne(plan.layers[layer].qkv_w, scratch.xn.data(),
                       scratch.qkv_b.data(), rows, plan.dim, lane, lanes);
        return;
    }
    case DEPTH_PHASE_QK_PREP: {
        const LfmDepthLayerV1 &weights = plan.layers[layer];
        const size_t heads = plan.heads_total + plan.kv_heads;
        const size_t cache =
            layer * plan.kv_heads * plan.codebooks * plan.hd;
        depth_band(heads, lane, lanes, &begin, &end);
        for (size_t head = begin; head < end; ++head) {
            if (head < plan.heads_total) {
                uint16_t bits[128];
                depth_qk_head(plan,
                              scratch.qkv_b.data() + head * plan.hd,
                              weights.q_ln, bits, codebook);
                lfm_bf16_to_f32(bits,
                                scratch.q_f.data() + head * plan.hd,
                                static_cast<int>(plan.hd));
                continue;
            }
            const size_t kv = head - plan.heads_total;
            uint16_t *key = scratch.k_plane.data() + cache +
                            (kv * plan.codebooks + codebook) * plan.hd;
            depth_qk_head(plan,
                          scratch.qkv_b.data() + plan.dim + kv * plan.hd,
                          weights.k_ln, key, codebook);
            const uint16_t *value = scratch.qkv_b.data() + plan.dim +
                                    plan.kv_heads * plan.hd + kv * plan.hd;
            std::memcpy(scratch.v_plane.data() + cache +
                            (kv * plan.codebooks + codebook) * plan.hd,
                        value, plan.hd * sizeof(uint16_t));
        }
        return;
    }
    case DEPTH_PHASE_ATTN: {
        const size_t cache =
            layer * plan.kv_heads * plan.codebooks * plan.hd;
        const size_t group = plan.heads_total / plan.kv_heads;
        const int live = static_cast<int>(codebook + 1);
        depth_band(plan.heads_total, lane, lanes, &begin, &end);
        for (size_t query = begin; query < end; ++query) {
            float attention[64];
            const size_t kv = query / group;
            lfm_attn_qk_bf16(scratch.q_f.data() + query * plan.hd,
                              scratch.k_plane.data() + cache +
                                  kv * plan.codebooks * plan.hd,
                              attention, live, static_cast<int>(plan.hd));
            lfm_softmax_scaled_f32(attention, live,
                                   lfm_rsqrt_size(plan.hd));
            lfm_attn_av_bf16(attention,
                              scratch.v_plane.data() + cache +
                                  kv * plan.codebooks * plan.hd,
                              scratch.attn_f.data() + query * plan.hd, live,
                              static_cast<int>(plan.hd));
        }
        return;
    }
    case DEPTH_PHASE_ATTN_ROUND:
        depth_band(plan.dim, lane, lanes, &begin, &end);
        if (end > begin) {
            lfm_f32_to_bf16(scratch.attn_f.data() + begin,
                            scratch.attn_b.data() + begin,
                            static_cast<int>(end - begin));
        }
        return;
    case DEPTH_PHASE_OUT:
        depth_gemv_rne_add(plan.layers[layer].out_w,
                           scratch.attn_b.data(), scratch.x.data(),
                           scratch.h.data(), plan.dim, plan.dim, lane, lanes);
        return;
    case DEPTH_PHASE_FFN_NORM_SUM:
        depth_norm_sum_stage(e, lane, scratch.h.data());
        return;
    case DEPTH_PHASE_FFN_NORM_APPLY:
        depth_norm_apply_stage(e, lane, scratch.h.data(),
                               plan.layers[layer].ffn_norm,
                               scratch.xn.data());
        return;
    case DEPTH_PHASE_FFN_GATE:
        depth_gemv_pair_swiglu(plan.layers[layer].w1,
                               plan.layers[layer].w3, scratch.xn.data(),
                               scratch.t_b.data(), plan.ffn, plan.dim, lane,
                               lanes);
        return;
    case DEPTH_PHASE_FFN_DOWN:
        depth_gemv_rne_add(plan.layers[layer].w2, scratch.t_b.data(),
                           scratch.h.data(), scratch.x.data(), plan.dim,
                           plan.ffn, lane, lanes);
        return;
    case DEPTH_PHASE_HEAD_NORM_SUM:
        depth_norm_sum_stage(e, lane, scratch.x.data());
        return;
    case DEPTH_PHASE_HEAD_NORM_APPLY:
        depth_norm_apply_stage(e, lane, scratch.x.data(),
                               plan.heads[codebook].norm,
                               scratch.xn.data());
        return;
    case DEPTH_PHASE_HEAD_LOGITS:
        depth_gemv_rne(plan.heads[codebook].logits, scratch.xn.data(),
                       scratch.logits_b.data(),
                       plan.heads[codebook].vocab, plan.dim, lane, lanes);
        return;
    case DEPTH_PHASE_SAMPLE:
        run_sample_pass(e, lane);
        return;
    case DEPTH_PHASE_EMBED: {
        const size_t token = request.out_tokens[codebook];
        depth_band(plan.dim, lane, lanes, &begin, &end);
        if (end > begin) {
            lfm_bf16_copy_bytes(
                depth_bytes(plan.heads[codebook].embedding) +
                    (token * plan.dim + begin) * sizeof(uint16_t),
                scratch.df_b.data() + begin, end - begin);
        }
        return;
    }
    case DEPTH_PHASE_DONE:
        return;
    default:
        if (lane == 0)
            e->active_status.store(-EPROTO, std::memory_order_release);
        return;
    }
}

static void fold_depth_norm(Engine *e) {
    DepthPlan &plan = *e->active_depth;
    DepthScratch &scratch = e->depth_scratch;
    const float total =
        lfm_sum_f32(scratch.partials.data(), e->lanes_total);
    scratch.inv_rms = lfm_inv_rms_f32(total, plan.dim, plan.eps);
}

static void configure_depth_sample(Engine *e, PassSlot *slot) {
    const size_t codebook = static_cast<size_t>(slot->program.outer);
    slot->sample = {
        .logits = e->depth_scratch.logits_b.data(),
        .count = slot->depth->heads[codebook].vocab,
        .dtype = SAMPLE_BF16,
        .config = slot->depth_req.sampler,
        .state = slot->depth_req.sample_state,
        .out = slot->depth_req.out_tokens + codebook,
    };
    const bool greedy =
        (slot->sample.config.flags & LFM_SAMPLE_FLAG_GREEDY) != 0 ||
        slot->sample.config.top_k == 1;
    slot->sample.phase = greedy ? SAMPLE_PHASE_GREEDY
                                : SAMPLE_PHASE_MAXIMUM;
}

static bool advance_depth_program(Engine *e, PassSlot *slot) {
    switch (slot->program.phase) {
    case DEPTH_PHASE_PROJECT:
        slot->program.phase = DEPTH_PHASE_CODEBOOK_ADD;
        return true;
    case DEPTH_PHASE_CODEBOOK_ADD:
        slot->program.inner = 0;
        slot->program.phase = slot->depth->layers.empty()
            ? DEPTH_PHASE_HEAD_NORM_SUM
            : DEPTH_PHASE_OP_NORM_SUM;
        return true;
    case DEPTH_PHASE_OP_NORM_SUM:
        fold_depth_norm(e);
        slot->program.phase = DEPTH_PHASE_OP_NORM_APPLY;
        return true;
    case DEPTH_PHASE_OP_NORM_APPLY:
        slot->program.phase = DEPTH_PHASE_QKV;
        return true;
    case DEPTH_PHASE_QKV:
        slot->program.phase = DEPTH_PHASE_QK_PREP;
        return true;
    case DEPTH_PHASE_QK_PREP:
        slot->program.phase = DEPTH_PHASE_ATTN;
        return true;
    case DEPTH_PHASE_ATTN:
        slot->program.phase = DEPTH_PHASE_ATTN_ROUND;
        return true;
    case DEPTH_PHASE_ATTN_ROUND:
        slot->program.phase = DEPTH_PHASE_OUT;
        return true;
    case DEPTH_PHASE_OUT:
        slot->program.phase = DEPTH_PHASE_FFN_NORM_SUM;
        return true;
    case DEPTH_PHASE_FFN_NORM_SUM:
        fold_depth_norm(e);
        slot->program.phase = DEPTH_PHASE_FFN_NORM_APPLY;
        return true;
    case DEPTH_PHASE_FFN_NORM_APPLY:
        slot->program.phase = DEPTH_PHASE_FFN_GATE;
        return true;
    case DEPTH_PHASE_FFN_GATE:
        slot->program.phase = DEPTH_PHASE_FFN_DOWN;
        return true;
    case DEPTH_PHASE_FFN_DOWN:
        ++slot->program.inner;
        slot->program.phase =
            slot->program.inner < slot->depth->layers.size()
                ? DEPTH_PHASE_OP_NORM_SUM
                : DEPTH_PHASE_HEAD_NORM_SUM;
        return true;
    case DEPTH_PHASE_HEAD_NORM_SUM:
        fold_depth_norm(e);
        slot->program.phase = DEPTH_PHASE_HEAD_NORM_APPLY;
        return true;
    case DEPTH_PHASE_HEAD_NORM_APPLY:
        slot->program.phase = DEPTH_PHASE_HEAD_LOGITS;
        return true;
    case DEPTH_PHASE_HEAD_LOGITS:
        configure_depth_sample(e, slot);
        slot->program.phase = DEPTH_PHASE_SAMPLE;
        return true;
    case DEPTH_PHASE_SAMPLE:
        if (advance_sample_program(e, slot)) return true;
        slot->program.phase = DEPTH_PHASE_EMBED;
        return true;
    case DEPTH_PHASE_EMBED:
        ++slot->program.outer;
        slot->program.inner = 0;
        if (slot->program.outer < slot->depth->codebooks) {
            slot->program.phase = DEPTH_PHASE_CODEBOOK_ADD;
            return true;
        }
        slot->program.phase = DEPTH_PHASE_DONE;
        if (slot->depth_req.completion_status != 0) {
            e->active_status.store(slot->depth_req.completion_status,
                                   std::memory_order_release);
        }
        return false;
    case DEPTH_PHASE_DONE:
        return false;
    default:
        std::abort();
    }
}

static void configure_stage(Stage *stage, uint32_t kind, uint32_t count,
                            uint32_t chunk) {
    if (!stage || kind == ST_IDLE || count == 0) std::abort();
    stage->kind = kind;
    stage->count = count;
    stage->chunk = chunk;
    stage->next.store(0, std::memory_order_release);
}

static void run_slot_stage(Engine *e, uint32_t lane, PassSlot *slot) {
    (void)lane;
    Stage *stage = &slot->stage;
    /* This loop performs one numerical tile per iteration. The fetch-add is a
     * work claim, not a readiness predicate: no lane retries a failed claim or
     * waits for another lane here. */
    for (;;) {
        const uint32_t tile =
            stage->next.fetch_add(1, std::memory_order_relaxed);
        if (tile >= stage->count) return;
        run_tile(stage->kind, tile, stage, e);
    }
}

enum : uint32_t {
    CONV_PHASE_STATS = 0,
    CONV_PHASE_NORM = 1,
    CONV_PHASE_INPROJ = 2,
    CONV_PHASE_OUTPROJ = 3,
    CONV_PHASE_MLP_SUMSQ = 4,
    CONV_PHASE_MLP_NORM = 5,
    CONV_PHASE_MLP_GATEUP = 6,
    CONV_PHASE_MLP_DOWN = 7,
    CONV_PHASE_DONE = 8,
};

enum : uint32_t {
    ATTN_PHASE_STATS = 0,
    ATTN_PHASE_NORM = 1,
    ATTN_PHASE_QKV = 2,
    ATTN_PHASE_PREP = 3,
    ATTN_PHASE_HEAD = 4,
    ATTN_PHASE_OPROJ = 5,
    ATTN_PHASE_MLP_SUMSQ = 6,
    ATTN_PHASE_MLP_NORM = 7,
    ATTN_PHASE_MLP_GATEUP = 8,
    ATTN_PHASE_MLP_DOWN = 9,
    ATTN_PHASE_DONE = 10,
};

static void configure_serial_stage(Stage *stage) {
    if (!stage) std::abort();
    stage->kind = ST_IDLE;
    stage->count = 0;
    stage->chunk = 0;
    stage->next.store(0, std::memory_order_release);
}

static void configure_embedded_mlp_stage(PassSlot *slot, uint32_t phase,
                                         uint32_t sumsq_phase,
                                         uint32_t norm_phase,
                                         uint32_t gateup_phase,
                                         uint32_t down_phase) {
    Pass *pass = &slot->pass;
    if (phase == sumsq_phase) {
        configure_stage(&slot->stage, ST_SUMSQ,
                        static_cast<uint32_t>(pass->tiles), 0);
        return;
    }
    if (phase == norm_phase) {
        configure_stage(&slot->stage, ST_NORM,
                        static_cast<uint32_t>(pass->tiles), 0);
        return;
    }
    if (phase == gateup_phase) {
        const uint32_t chunk = static_cast<uint32_t>(
            (pass->i + pass->tiles - 1) / pass->tiles);
        configure_stage(&slot->stage, ST_GATEUP,
                        static_cast<uint32_t>((pass->i + chunk - 1) / chunk),
                        chunk);
        return;
    }
    if (phase == down_phase) {
        const uint32_t chunk = static_cast<uint32_t>(
            (pass->h + pass->tiles - 1) / pass->tiles);
        configure_stage(&slot->stage, ST_DOWN,
                        static_cast<uint32_t>((pass->h + chunk - 1) / chunk),
                        chunk);
        return;
    }
    std::abort();
}

static void initialize_conv_program(Engine *e, PassSlot *slot) {
    if (!e || !slot || !slot->model ||
        slot->conv.layer >= slot->model->layers.size()) {
        std::abort();
    }
    const LfmLayerDesc *d = &slot->model->layers[slot->conv.layer];
    const size_t h = slot->model->h;
    const size_t lanes = slot->conv.lanes;
    const size_t sc_tiles = std::min(h, lanes);
    const size_t mlp_tiles = std::min(std::min(h, slot->model->ffn), lanes);
    if (sc_tiles == 0 || mlp_tiles == 0) std::abort();

    ScPass *conv = &slot->sc;
    conv->x = slot->conv.x;
    conv->norm_w = d->op_norm_w;
    conv->in_w = d->in_w;
    conv->out_w = d->out_w;
    conv->conv_w = d->conv_w;
    conv->state_in = slot->conv.state_in;
    conv->state_out = slot->conv.state_out;
    conv->h = h;
    conv->k = d->k;
    conv->xn = e->sc_xn.data();
    conv->projb = e->sc_projb.data();
    conv->mid = e->sc_mid.data();
    conv->rs_bits.store(0, std::memory_order_relaxed);

    Pass *mlp = &slot->pass;
    mlp->x = conv->mid;
    mlp->norm_w = d->ffn_norm_w;
    mlp->w1 = d->w1;
    mlp->w3 = d->w3;
    mlp->w2 = d->w2;
    mlp->out = slot->conv.out;
    mlp->h = h;
    mlp->i = slot->model->ffn;
    mlp->tiles = mlp_tiles;
    mlp->eps = d->ffn_eps;
    mlp->partials = e->sc_partials.data();
    mlp->xn = e->sc_xn.data();
    mlp->t = e->sc_t.data();
    mlp->rs_bits.store(0, std::memory_order_relaxed);

    slot->program.phase = CONV_PHASE_STATS;
    configure_serial_stage(&slot->stage);
}

static void configure_conv_stage(PassSlot *slot) {
    const size_t h = slot->model->h;
    const size_t tiles = std::min(h, slot->conv.lanes);
    const uint32_t chunk = static_cast<uint32_t>((h + tiles - 1) / tiles);
    switch (slot->program.phase) {
    case CONV_PHASE_STATS:
        configure_serial_stage(&slot->stage);
        return;
    case CONV_PHASE_NORM:
        configure_stage(&slot->stage, ST_SC_NORM,
                        static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
        return;
    case CONV_PHASE_INPROJ:
        configure_stage(&slot->stage, ST_SC_INPROJ,
                        static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
        return;
    case CONV_PHASE_OUTPROJ:
        configure_stage(&slot->stage, ST_SC_OUTPROJ,
                        static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
        return;
    default:
        configure_embedded_mlp_stage(
            slot, slot->program.phase, CONV_PHASE_MLP_SUMSQ,
            CONV_PHASE_MLP_NORM, CONV_PHASE_MLP_GATEUP,
            CONV_PHASE_MLP_DOWN);
    }
}

static void initialize_attn_program(Engine *e, PassSlot *slot) {
    if (!e || !slot || !slot->model ||
        slot->attn.layer >= slot->model->layers.size()) {
        std::abort();
    }
    const LfmLayerDesc *d = &slot->model->layers[slot->attn.layer];
    const size_t h = slot->model->h;
    const size_t lanes = slot->attn.lanes;
    const size_t mlp_tiles = std::min(std::min(h, slot->model->ffn), lanes);
    if (lanes == 0 || mlp_tiles == 0) std::abort();

    ScPass *norm = &slot->sc;
    norm->x = slot->attn.x;
    norm->norm_w = d->op_norm_w;
    norm->h = h;
    norm->xn = e->sc_xn.data();
    norm->rs_bits.store(0, std::memory_order_relaxed);

    AtPass *attn = &slot->at;
    attn->q_w = d->q_w;
    attn->k_w = d->k_w;
    attn->v_w = d->v_w;
    attn->o_w = d->o_w;
    attn->qn_w = d->qn_w;
    attn->kn_w = d->kn_w;
    attn->qkvb = e->at_qkvb.data();
    attn->ybits = e->at_y.data();
    attn->att = e->at_att.data();
    attn->x = norm->x;
    attn->mid = e->sc_mid.data();
    attn->k_plane = slot->attn.k_plane;
    attn->v_plane = slot->attn.v_plane;
    attn->cos_row = slot->attn.cos_base + slot->attn.pos * (d->hd / 2);
    attn->sin_row = slot->attn.sin_base + slot->attn.pos * (d->hd / 2);
    attn->head_stride = slot->attn.head_stride;
    attn->att_len = slot->attn.pos + 1;
    attn->max_ctx = slot->model->max_ctx;
    attn->h = h;
    attn->n_head = d->n_head;
    attn->n_kv = d->n_kv;
    attn->hd = d->hd;
    attn->qk_eps = d->qk_eps;

    Pass *mlp = &slot->pass;
    mlp->x = attn->mid;
    mlp->norm_w = d->ffn_norm_w;
    mlp->w1 = d->w1;
    mlp->w3 = d->w3;
    mlp->w2 = d->w2;
    mlp->out = slot->attn.out;
    mlp->h = h;
    mlp->i = slot->model->ffn;
    mlp->tiles = mlp_tiles;
    mlp->eps = d->ffn_eps;
    mlp->partials = e->sc_partials.data();
    mlp->xn = e->sc_xn.data();
    mlp->t = e->sc_t.data();
    mlp->rs_bits.store(0, std::memory_order_relaxed);

    slot->program.phase = ATTN_PHASE_STATS;
    configure_serial_stage(&slot->stage);
}

static void configure_attn_stage(PassSlot *slot) {
    const size_t h = slot->model->h;
    const size_t tiles = std::min(h, slot->attn.lanes);
    const uint32_t chunk = static_cast<uint32_t>((h + tiles - 1) / tiles);
    AtPass *attn = &slot->at;
    switch (slot->program.phase) {
    case ATTN_PHASE_STATS:
        configure_serial_stage(&slot->stage);
        return;
    case ATTN_PHASE_NORM:
        configure_stage(&slot->stage, ST_SC_NORM,
                        static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
        return;
    case ATTN_PHASE_QKV: {
        const size_t rows =
            (attn->n_head + 2 * attn->n_kv) * attn->hd;
        const uint32_t band = static_cast<uint32_t>(
            (rows + tiles - 1) / tiles);
        configure_stage(&slot->stage, ST_AT_QKV,
                        static_cast<uint32_t>((rows + band - 1) / band), band);
        return;
    }
    case ATTN_PHASE_PREP:
        configure_stage(&slot->stage, ST_AT_PREP,
                        static_cast<uint32_t>(
                            std::max(attn->n_head, attn->n_kv)), 1);
        return;
    case ATTN_PHASE_HEAD:
        configure_stage(&slot->stage, ST_AT_HEAD,
                        static_cast<uint32_t>(attn->n_head), 1);
        return;
    case ATTN_PHASE_OPROJ:
        configure_stage(&slot->stage, ST_AT_OPROJ,
                        static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
        return;
    default:
        configure_embedded_mlp_stage(
            slot, slot->program.phase, ATTN_PHASE_MLP_SUMSQ,
            ATTN_PHASE_MLP_NORM, ATTN_PHASE_MLP_GATEUP,
            ATTN_PHASE_MLP_DOWN);
    }
}

static void run_serial_program_stage(Engine *e, uint32_t lane, PassSlot *slot) {
    (void)e;
    if (lane != 0) return;
    const bool conv = slot->request == REQ_CONV_LAYER ||
        (slot->request == REQ_TOKEN_PASS &&
         slot->token_program.kind == TOKEN_PROGRAM_CONV);
    const bool attn = slot->request == REQ_ATTN_LAYER ||
        (slot->request == REQ_TOKEN_PASS &&
         slot->token_program.kind == TOKEN_PROGRAM_ATTN);
    if (conv &&
        slot->program.phase == CONV_PHASE_STATS) {
        const LfmLayerDesc *d = &slot->model->layers[slot->conv.layer];
        const float total =
            lfm_bf16_sumsq_ordered_f32(slot->sc.x.data(),
                                      static_cast<int>(slot->sc.h));
        const float inv = lfm_inv_rms_f32(total, slot->sc.h, d->op_eps);
        uint32_t bits = 0;
        std::memcpy(&bits, &inv, sizeof(bits));
        slot->sc.rs_bits.store(bits, std::memory_order_release);
        return;
    }
    if (attn &&
        slot->program.phase == ATTN_PHASE_STATS) {
        const LfmLayerDesc *d = &slot->model->layers[slot->attn.layer];
        const float total =
            lfm_bf16_sumsq_ordered_f32(slot->sc.x.data(),
                                      static_cast<int>(slot->sc.h));
        const float inv = lfm_inv_rms_f32(total, slot->sc.h, d->op_eps);
        uint32_t bits = 0;
        std::memcpy(&bits, &inv, sizeof(bits));
        slot->sc.rs_bits.store(bits, std::memory_order_release);
        return;
    }
    std::abort();
}

static void fold_embedded_mlp_sumsq(PassSlot *slot) {
    Pass *pass = &slot->pass;
    const float total = lfm_sum_f32(pass->partials, pass->tiles);
    const float inv = lfm_inv_rms_f32(total, pass->h, pass->eps);
    uint32_t bits = 0;
    std::memcpy(&bits, &inv, sizeof(bits));
    pass->rs_bits.store(bits, std::memory_order_release);
}

static bool advance_conv_program(PassSlot *slot) {
    switch (slot->program.phase) {
    case CONV_PHASE_STATS:
        slot->program.phase = CONV_PHASE_NORM;
        break;
    case CONV_PHASE_NORM:
        slot->program.phase = CONV_PHASE_INPROJ;
        break;
    case CONV_PHASE_INPROJ:
        slot->program.phase = CONV_PHASE_OUTPROJ;
        break;
    case CONV_PHASE_OUTPROJ:
        slot->program.phase = CONV_PHASE_MLP_SUMSQ;
        break;
    case CONV_PHASE_MLP_SUMSQ:
        fold_embedded_mlp_sumsq(slot);
        slot->program.phase = CONV_PHASE_MLP_NORM;
        break;
    case CONV_PHASE_MLP_NORM:
        slot->program.phase = CONV_PHASE_MLP_GATEUP;
        break;
    case CONV_PHASE_MLP_GATEUP:
        slot->program.phase = CONV_PHASE_MLP_DOWN;
        break;
    case CONV_PHASE_MLP_DOWN:
        slot->program.phase = CONV_PHASE_DONE;
        configure_serial_stage(&slot->stage);
        return false;
    default:
        std::abort();
    }
    configure_conv_stage(slot);
    return true;
}

static bool advance_attn_program(PassSlot *slot) {
    switch (slot->program.phase) {
    case ATTN_PHASE_STATS:
        slot->program.phase = ATTN_PHASE_NORM;
        break;
    case ATTN_PHASE_NORM:
        slot->program.phase = ATTN_PHASE_QKV;
        break;
    case ATTN_PHASE_QKV:
        slot->program.phase = ATTN_PHASE_PREP;
        break;
    case ATTN_PHASE_PREP:
        slot->program.phase = ATTN_PHASE_HEAD;
        break;
    case ATTN_PHASE_HEAD:
        slot->program.phase = ATTN_PHASE_OPROJ;
        break;
    case ATTN_PHASE_OPROJ:
        slot->program.phase = ATTN_PHASE_MLP_SUMSQ;
        break;
    case ATTN_PHASE_MLP_SUMSQ:
        fold_embedded_mlp_sumsq(slot);
        slot->program.phase = ATTN_PHASE_MLP_NORM;
        break;
    case ATTN_PHASE_MLP_NORM:
        slot->program.phase = ATTN_PHASE_MLP_GATEUP;
        break;
    case ATTN_PHASE_MLP_GATEUP:
        slot->program.phase = ATTN_PHASE_MLP_DOWN;
        break;
    case ATTN_PHASE_MLP_DOWN:
        slot->program.phase = ATTN_PHASE_DONE;
        configure_serial_stage(&slot->stage);
        return false;
    default:
        std::abort();
    }
    configure_attn_stage(slot);
    return true;
}

// Serial per-head helpers for the attention pass (tiny next to the GEMVs; the
// The oracle computes these as whole-plane operations; the math below preserves the exact
// per-element ladder those ops perform).

// RMSNorm over one head row: all f32 arithmetic (upcast, mean via the pinned
// ascending-order sum, +eps, sqrt, reciprocal, multiplies), then one bf16 round.
static void qk_norm_row(const uint16_t *x, WeightBytes w, uint16_t *out, size_t hd,
                        float eps) {
    float total = lfm_bf16_sumsq_ordered_f32(x, (int)hd);
    float inv = lfm_inv_rms_f32(total, hd, eps);
    lfm_bf16_rmsnorm(x, w, out, (int)hd, inv);
}

// Accepted slow RoPE order over one head row, NeoX half-split, computed in bf16
// exactly as the tensor ops do: cos2 = [cos|cos], out = x⊙cos2 + rotate_half(x)⊙sin2,
// where every bf16 multiply and the add each round once (half-crate semantics:
// f32 compute, RNE back to bf16). rotate_half = [-x2 | x1]; negation is exact.
static void rope_slow_row(uint16_t *x, const uint16_t *cos_row, const uint16_t *sin_row,
                          size_t hd) {
    lfm_bf16_rope_neox(x, cos_row, sin_row, hd);
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
        lfm_bf16_gemm_nt_strided_f32(
            a, band, out + begin, (int)rows, (int)(end - begin),
            (int)k, (int)stride);
    }
}

// The prefill rows are bounded by PREFILL_ROWS (four). This uses the same
// small-M dot reduction as prefill_linear, but keeps each completed F32 dot in
// registers through its one exact RNE BF16 storage boundary. Resident weights
// remain byte-addressed and each lane writes a disjoint destination-column band.
static void prefill_linear_bf16(Engine *e, uint32_t lane, const uint16_t *a,
                                WeightBytes weight, uint16_t *out, size_t rows,
                                size_t n, size_t k, size_t stride) {
    size_t begin = 0, end = 0;
    prefill_band(n, lane, e->lanes_total, &begin, &end);
    if (end > begin) {
        const WeightBytes band = weight_offset(weight, begin * k);
        lfm_bf16_gemm_nt_bias_bf16(
            a, band, nullptr, out + begin, (int)rows, (int)(end - begin),
            (int)k, (int)stride);
    }
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
        const float total = lfm_bf16_sumsq_ordered_f32(source.data(), (int)input.h);
        const float inv = lfm_inv_rms_f32(total, input.h, eps);
        lfm_bf16_rmsnorm(source.data(), weight, out + row * input.h,
                         (int)input.h, inv);
    }
}

// The MLP norm intentionally has a different reduction contract from the
// operator/final norms: decode partitions by a fixed logical tile count, then
// folds partials in tile order. Reproduce that order independently per row.
static void prefill_mlp_norm(Engine *e, uint32_t lane,
                             const PrefillInput &input, WeightBytes weight,
                             float eps, uint16_t *out, size_t rows,
                             size_t logical_lanes) {
    const size_t cap = std::min(input.h, e->model->ffn);
    const size_t tiles = std::min(logical_lanes, cap);
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
}

enum : uint32_t {
    PREFILL_PHASE_CONV_NORM = 0,
    PREFILL_PHASE_CONV_IN = 1,
    PREFILL_PHASE_CONV_FIR = 2,
    PREFILL_PHASE_CONV_OUT = 3,
    PREFILL_PHASE_CONV_ADD = 4,
    PREFILL_PHASE_ATTN_NORM = 5,
    PREFILL_PHASE_ATTN_Q = 6,
    PREFILL_PHASE_ATTN_K = 7,
    PREFILL_PHASE_ATTN_V = 8,
    PREFILL_PHASE_ATTN_PREP = 9,
    PREFILL_PHASE_ATTN_MIX = 10,
    PREFILL_PHASE_ATTN_OUT = 11,
    PREFILL_PHASE_ATTN_ADD = 12,
    PREFILL_PHASE_MLP_NORM = 13,
    PREFILL_PHASE_MLP_GATE = 14,
    PREFILL_PHASE_MLP_UP = 15,
    PREFILL_PHASE_MLP_SWIGLU = 16,
    PREFILL_PHASE_MLP_DOWN = 17,
    PREFILL_PHASE_MLP_ADD = 18,
    PREFILL_PHASE_FINAL_NORM = 19,
    PREFILL_PHASE_FINAL_LOGITS = 20,
    PREFILL_PHASE_SAMPLE = 21,
    PREFILL_PHASE_DONE = 22,
};

enum : uint32_t {
    PREFILL_FLAG_SAMPLE = 1u,
    /* Test-only: the pass is running the Depthformer codebook-0 readout tail
     * of a sampled prefill; program.phase holds DEPTH_PHASE_* values and the
     * ordinary depth program stages execute against this slot. */
    PREFILL_FLAG_READOUT_DEPTH = 2u,
};

static PrefillInput prefill_first_input(const PassSlot *slot) {
    const PrefillReq &request = slot->prefill;
    if (request.embed_kind == 0) {
        return {.embedding = slot->model->embed_w,
                .ids = request.ids.data(),
                .h = slot->model->h};
    }
    return {.rows = request.provided_rows, .h = slot->model->h};
}

static PrefillInput prefill_layer_input(const PassSlot *slot, size_t layer) {
    if (layer == 0) return prefill_first_input(slot);
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const uint16_t *rows = (layer - 1) % 2 == 0
        ? workspace->h1.data()
        : workspace->h0.data();
    return {.rows = rows, .h = slot->model->h};
}

static uint16_t *prefill_layer_output(const PassSlot *slot, size_t layer) {
    PrefillWorkspace *workspace = slot->prefill.workspace;
    return layer % 2 == 0 ? workspace->h1.data() : workspace->h0.data();
}

static PrefillInput prefill_final_input(const PassSlot *slot) {
    const PrefillReq &request = slot->prefill;
    const size_t h = slot->model->h;
    const size_t row = request.rows - 1;
    if (slot->model->layers.empty()) {
        if (request.embed_kind == 0) {
            return {.embedding = slot->model->embed_w,
                    .ids = request.ids.data() + row,
                    .h = h};
        }
        return {.rows = request.provided_rows + row * h, .h = h};
    }
    const uint16_t *rows = prefill_layer_output(
        slot, slot->model->layers.size() - 1);
    return {.rows = rows + row * h, .h = h};
}

static void prefill_conv_fir_stage(Engine *e, uint32_t lane,
                                   PassSlot *slot,
                                   const LfmLayerDesc *desc) {
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const LfmLayerState *state =
        &slot->prefill.states[slot->program.outer];
    const size_t rows = slot->prefill.rows;
    const size_t h = slot->model->h;
    const size_t kernel = desc->k;
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
        for (size_t tap = 0; tap + 1 < kernel; ++tap) {
            carry[tap] = state->conv_state[channel * (kernel - 1) + tap];
        }
        for (size_t row = 0; row < rows; ++row) {
            const uint16_t *bcx =
                workspace->bcxb.data() + row * 3 * h;
            const uint16_t bx_bits = rb_bits(
                bf16_f32(bcx[channel]) * bf16_f32(bcx[2 * h + channel]));
            const float bx = bf16_f32(bx_bits);
            const float v0 = kernel == 1 ? bx : bf16_f32(carry[0]);
            float acc = fast_k3 && kernel == 3
                ? w0 * v0
                : 0.0f + w0 * v0;
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
            workspace->projb[row * h + channel] = rb_bits(
                bf16_f32(bcx[h + channel]) * bf16_f32(conv));
            if (kernel > 1) {
                for (size_t tap = 0; tap + 2 < kernel; ++tap)
                    carry[tap] = carry[tap + 1];
                carry[kernel - 2] = bx_bits;
            }
        }
        for (size_t tap = 0; tap + 1 < kernel; ++tap) {
            state->conv_state[channel * (kernel - 1) + tap] = carry[tap];
        }
    }
}

static void prefill_attention_prep_stage(Engine *e, uint32_t lane,
                                         PassSlot *slot,
                                         const LfmLayerDesc *desc) {
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const PrefillReq &request = slot->prefill;
    const LfmLayerState &state = request.states[slot->program.outer];
    const size_t nh = desc->n_head;
    const size_t nkv = desc->n_kv;
    const size_t hd = desc->hd;
    const size_t qrows = nh * hd;
    const size_t kvrows = nkv * hd;
    const size_t qkv = qrows + 2 * kvrows;
    const size_t tasks = request.rows * (nh + nkv);
    for (size_t task = lane; task < tasks; task += e->lanes_total) {
        const size_t row = task / (nh + nkv);
        const size_t head = task % (nh + nkv);
        const uint16_t *cos = request.cos_base +
                              (request.pos + row) * (hd / 2);
        const uint16_t *sin = request.sin_base +
                              (request.pos + row) * (hd / 2);
        uint16_t *base = workspace->qkvb.data() + row * qkv;
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
        lfm_bf16_copy_bytes(
            key, state.k_plane + kh * state.head_stride +
                     (request.pos + row) * hd,
            hd);
        lfm_bf16_copy_bytes(
            value, state.v_plane + kh * state.head_stride +
                       (request.pos + row) * hd,
            hd);
    }
}

static void prefill_attention_mix_stage(Engine *e, uint32_t lane,
                                        PassSlot *slot,
                                        const LfmLayerDesc *desc) {
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const PrefillReq &request = slot->prefill;
    const LfmLayerState &state = request.states[slot->program.outer];
    const size_t nh = desc->n_head;
    const size_t nkv = desc->n_kv;
    const size_t hd = desc->hd;
    const size_t qrows = nh * hd;
    const size_t kvrows = nkv * hd;
    const size_t qkv = qrows + 2 * kvrows;
    const size_t group = nh / nkv;
    for (size_t task = lane; task < request.rows * nh;
         task += e->lanes_total) {
        const size_t row = task / nh;
        const size_t qh = task % nh;
        const size_t kh = qh / group;
        const size_t length = request.pos + row + 1;
        const uint16_t *query =
            workspace->qkvb.data() + row * qkv + qh * hd;
        float *score = workspace->scores.data() + lane * workspace->max_ctx;
        float qf[512];
        float value[512];
        lfm_bf16_to_f32(query, qf, static_cast<int>(hd));
        lfm_attn_qk_bf16(qf,
                         state.k_plane + kh * state.head_stride,
                         score, static_cast<int>(length),
                         static_cast<int>(hd));
        lfm_softmax_scaled_f32(score, static_cast<int>(length),
                               lfm_rsqrt_size(hd));
        lfm_attn_av_bf16(score,
                         state.v_plane + kh * state.head_stride,
                         value, static_cast<int>(length),
                         static_cast<int>(hd));
        lfm_f32_to_bf16(
            value, workspace->att_y.data() + row * (nh * hd) + qh * hd,
            static_cast<int>(hd));
    }
}

static void prefill_swiglu_stage(Engine *e, uint32_t lane,
                                 PassSlot *slot) {
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const size_t rows = slot->prefill.rows;
    const size_t ffn = slot->model->ffn;
    size_t begin = 0, end = 0;
    prefill_band(ffn, lane, e->lanes_total, &begin, &end);
    if (end <= begin) return;
    for (size_t row = 0; row < rows; ++row) {
        lfm_swiglu_bf16(workspace->gu.data() + row * 2 * ffn + begin,
                        workspace->gu.data() + row * 2 * ffn + ffn + begin,
                        workspace->gate.data() + row * ffn + begin,
                        static_cast<int>(end - begin));
    }
}

static bool begin_prefill_layer(Engine *e, PassSlot *slot) {
    const size_t layer = static_cast<size_t>(slot->program.outer);
    if (layer >= slot->model->layers.size()) {
        slot->program.phase = PREFILL_PHASE_FINAL_NORM;
        return true;
    }
    const uint32_t kind = slot->model->layers[layer].kind;
    if (kind == 0) {
        slot->program.phase = PREFILL_PHASE_CONV_NORM;
        return true;
    }
    if (kind == 1) {
        slot->program.phase = PREFILL_PHASE_ATTN_NORM;
        return true;
    }
    e->active_status.store(-EINVAL, std::memory_order_release);
    slot->program.phase = PREFILL_PHASE_DONE;
    return false;
}

static void initialize_prefill_program(Engine *e, PassSlot *slot) {
    if (!e || !slot || slot->request != REQ_PREFILL || !slot->model ||
        !slot->prefill.workspace || slot->prefill.rows == 0) {
        std::abort();
    }
    size_t provided_values = 0;
    const bool valid_values = checked_size_product(
        slot->prefill.rows, slot->model->h, &provided_values);
    bool valid = valid_values && slot->prefill.states &&
        slot->prefill.n_states == slot->model->layers.size() &&
        slot->prefill.out_hidden &&
        slot->prefill.out_hidden_len == slot->model->h &&
        ((slot->prefill.embed_kind == 0) ||
         (slot->prefill.embed_kind == 2 && slot->prefill.provided_rows &&
          slot->prefill.provided_values == provided_values)) &&
        ((!slot->prefill.sample && !slot->prefill.out_token) ||
         (slot->prefill.sample && slot->prefill.out_token)) &&
        (!slot->prefill.readout ||
         (slot->prefill.sample && slot->depth &&
          slot->depth_req.hidden == slot->prefill.out_hidden));
    const size_t end_pos = slot->prefill.pos + slot->prefill.rows;
    valid = valid && end_pos >= slot->prefill.pos &&
        end_pos <= slot->model->max_ctx;
    for (size_t layer = 0; valid && layer < slot->model->layers.size();
         ++layer) {
        const LfmLayerDesc &desc = slot->model->layers[layer];
        const LfmLayerState &state = slot->prefill.states[layer];
        if (desc.kind == 1) {
            size_t live = 0, rope = 0;
            const size_t half = desc.hd / 2;
            valid = desc.hd >= 2 && desc.hd % 2 == 0 && desc.n_kv > 0 &&
                slot->prefill.cos_base && slot->prefill.sin_base &&
                checked_size_product(end_pos, desc.hd, &live) &&
                checked_size_product(end_pos, half, &rope) &&
                slot->prefill.rope_len >= rope && state.k_plane &&
                state.v_plane && state.head_stride >= live;
            const size_t prior = desc.n_kv - 1;
            valid = valid &&
                prior <= SIZE_MAX / state.head_stride &&
                prior * state.head_stride <= SIZE_MAX - live &&
                state.k_len >= prior * state.head_stride + live &&
                state.v_len >= prior * state.head_stride + live;
            continue;
        }
        const size_t tail = desc.k > 0 ? desc.k - 1 : 0;
        valid = desc.kind == 0 && desc.k > 0 && state.conv_state &&
            (tail == 0 || slot->model->h <= SIZE_MAX / tail) &&
            state.conv_len >= slot->model->h * tail;
    }
    slot->program.outer = 0;
    slot->program.inner = 0;
    slot->program.flags = slot->prefill.sample ? PREFILL_FLAG_SAMPLE : 0u;
    if (!valid) {
        e->active_status.store(-ESTALE, std::memory_order_release);
        slot->program.phase = PREFILL_PHASE_DONE;
        return;
    }
    (void)begin_prefill_layer(e, slot);
}

static void configure_prefill_sample(Engine *e, PassSlot *slot) {
    slot->sample = {
        .logits = slot->prefill.workspace->logits.data(),
        .count = slot->model->vocab,
        .dtype = SAMPLE_F32,
        .config = slot->prefill.sampler,
        .state = slot->prefill.sample_state,
        .out = slot->prefill.out_token,
    };
    const bool greedy =
        (slot->sample.config.flags & LFM_SAMPLE_FLAG_GREEDY) != 0 ||
        slot->sample.config.top_k == 1;
    slot->sample.phase = greedy ? SAMPLE_PHASE_GREEDY
                                : SAMPLE_PHASE_MAXIMUM;
    e->sample = slot->sample;
}

/* Test-only readout fill. It scans the exact logits plane the sampler
 * consumed (softmax/logprob math lives here, on the test path only) and
 * mirrors the greedy fold's tie policy: equal values keep the lower id. */
static void fill_listen_readout_head(const void *logits, size_t count,
                                     uint32_t dtype, uint32_t *out_ids,
                                     float *out_logprobs,
                                     float *out_entropy) {
    const auto value_at = [logits, dtype](size_t index) {
        const float value = dtype == SAMPLE_F32
            ? static_cast<const float *>(logits)[index]
            : bf16_f32(static_cast<const uint16_t *>(logits)[index]);
        return std::isnan(value) ? -std::numeric_limits<float>::infinity()
                                 : value;
    };
    float top_values[LFM_LISTEN_READOUT_TOP_K];
    uint32_t top_ids[LFM_LISTEN_READOUT_TOP_K];
    size_t held = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < count; ++index) {
        const float value = value_at(index);
        if (value > maximum) maximum = value;
        size_t rank = held;
        while (rank > 0 && value > top_values[rank - 1]) --rank;
        if (rank >= LFM_LISTEN_READOUT_TOP_K) continue;
        const size_t tail =
            std::min(held, size_t{LFM_LISTEN_READOUT_TOP_K} - 1);
        for (size_t move = tail; move > rank; --move) {
            top_values[move] = top_values[move - 1];
            top_ids[move] = top_ids[move - 1];
        }
        top_values[rank] = value;
        top_ids[rank] = static_cast<uint32_t>(index);
        if (held < LFM_LISTEN_READOUT_TOP_K) ++held;
    }
    double exp_sum = 0.0;
    double weighted = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double centered =
            static_cast<double>(value_at(index)) -
            static_cast<double>(maximum);
        const double weight = std::exp(centered);
        exp_sum += weight;
        weighted += weight * centered;
    }
    const double log_z = std::log(exp_sum);
    for (size_t rank = 0; rank < LFM_LISTEN_READOUT_TOP_K; ++rank) {
        const bool live = rank < held;
        out_ids[rank] = live ? top_ids[rank] : UINT32_MAX;
        out_logprobs[rank] = live
            ? static_cast<float>(
                  static_cast<double>(top_values[rank]) -
                  static_cast<double>(maximum) - log_z)
            : -std::numeric_limits<float>::infinity();
    }
    *out_entropy = static_cast<float>(log_z - weighted / exp_sum);
}

static void run_prefill_program_stage(Engine *e, uint32_t lane,
                                      PassSlot *slot) {
    if ((slot->program.flags & PREFILL_FLAG_READOUT_DEPTH) != 0) {
        /* Test-only readout tail: the codebook-0 head evaluation runs the
         * ordinary depth program stages against this slot's depth scratch. */
        run_depth_program_stage(e, lane, slot);
        return;
    }
    PrefillWorkspace *workspace = slot->prefill.workspace;
    const size_t layer = static_cast<size_t>(slot->program.outer);
    const size_t rows = slot->prefill.rows;
    const size_t h = slot->model->h;
    const size_t ffn = slot->model->ffn;
    const LfmLayerDesc *desc = layer < slot->model->layers.size()
        ? &slot->model->layers[layer]
        : nullptr;
    const PrefillInput input = desc ? prefill_layer_input(slot, layer)
                                    : PrefillInput{};
    const PrefillInput mlp = {.rows = workspace->mid.data(), .h = h};
    switch (slot->program.phase) {
    case PREFILL_PHASE_CONV_NORM:
    case PREFILL_PHASE_ATTN_NORM:
        prefill_norm(e, lane, input, desc->op_norm_w, desc->op_eps,
                     workspace->xn.data(), rows);
        return;
    case PREFILL_PHASE_CONV_IN:
        prefill_linear_bf16(e, lane, workspace->xn.data(), desc->in_w,
                            workspace->bcxb.data(), rows, 3 * h, h, 3 * h);
        return;
    case PREFILL_PHASE_CONV_FIR:
        prefill_conv_fir_stage(e, lane, slot, desc);
        return;
    case PREFILL_PHASE_CONV_OUT:
        prefill_linear_bf16(e, lane, workspace->projb.data(), desc->out_w,
                            workspace->stage.data(), rows, h, h, h);
        return;
    case PREFILL_PHASE_CONV_ADD:
    case PREFILL_PHASE_ATTN_ADD:
        prefill_add(e, lane, workspace->stage.data(), input,
                    workspace->mid.data(), rows);
        return;
    case PREFILL_PHASE_ATTN_Q: {
        const size_t qrows = desc->n_head * desc->hd;
        const size_t kvrows = desc->n_kv * desc->hd;
        const size_t qkv = qrows + 2 * kvrows;
        prefill_linear_bf16(e, lane, workspace->xn.data(), desc->q_w,
                            workspace->qkvb.data(), rows, qrows, h, qkv);
        return;
    }
    case PREFILL_PHASE_ATTN_K: {
        const size_t qrows = desc->n_head * desc->hd;
        const size_t kvrows = desc->n_kv * desc->hd;
        const size_t qkv = qrows + 2 * kvrows;
        prefill_linear_bf16(e, lane, workspace->xn.data(), desc->k_w,
                            workspace->qkvb.data() + qrows, rows, kvrows,
                            h, qkv);
        return;
    }
    case PREFILL_PHASE_ATTN_V: {
        const size_t qrows = desc->n_head * desc->hd;
        const size_t kvrows = desc->n_kv * desc->hd;
        const size_t qkv = qrows + 2 * kvrows;
        prefill_linear_bf16(e, lane, workspace->xn.data(), desc->v_w,
                            workspace->qkvb.data() + qrows + kvrows, rows,
                            kvrows, h, qkv);
        return;
    }
    case PREFILL_PHASE_ATTN_PREP:
        prefill_attention_prep_stage(e, lane, slot, desc);
        return;
    case PREFILL_PHASE_ATTN_MIX:
        prefill_attention_mix_stage(e, lane, slot, desc);
        return;
    case PREFILL_PHASE_ATTN_OUT:
        prefill_linear_bf16(e, lane, workspace->att_y.data(), desc->o_w,
                            workspace->stage.data(), rows, h,
                            desc->n_head * desc->hd, h);
        return;
    case PREFILL_PHASE_MLP_NORM:
        prefill_mlp_norm(e, lane, mlp, desc->ffn_norm_w, desc->ffn_eps,
                         workspace->xn.data(), rows, slot->prefill.lanes);
        return;
    case PREFILL_PHASE_MLP_GATE:
        prefill_linear(e, lane, workspace->xn.data(), desc->w1,
                       workspace->gu.data(), rows, ffn, h, 2 * ffn);
        return;
    case PREFILL_PHASE_MLP_UP:
        prefill_linear(e, lane, workspace->xn.data(), desc->w3,
                       workspace->gu.data() + ffn, rows, ffn, h, 2 * ffn);
        return;
    case PREFILL_PHASE_MLP_SWIGLU:
        prefill_swiglu_stage(e, lane, slot);
        return;
    case PREFILL_PHASE_MLP_DOWN:
        prefill_linear_bf16(e, lane, workspace->gate.data(), desc->w2,
                            workspace->stage.data(), rows, h, ffn, h);
        return;
    case PREFILL_PHASE_MLP_ADD:
        prefill_add(e, lane, workspace->stage.data(), mlp,
                    prefill_layer_output(slot, layer), rows);
        return;
    case PREFILL_PHASE_FINAL_NORM:
        prefill_norm(e, lane, prefill_final_input(slot),
                     slot->model->emb_norm_w, slot->model->emb_norm_eps,
                     slot->prefill.out_hidden, 1);
        return;
    case PREFILL_PHASE_FINAL_LOGITS:
        prefill_linear(e, lane, slot->prefill.out_hidden,
                       slot->model->embed_w, workspace->logits.data(), 1,
                       slot->model->vocab, h, slot->model->vocab);
        return;
    case PREFILL_PHASE_SAMPLE:
        run_sample_pass(e, lane);
        return;
    case PREFILL_PHASE_DONE:
        return;
    default:
        if (lane == 0)
            e->active_status.store(-EPROTO, std::memory_order_release);
        return;
    }
}

static bool advance_prefill_program(Engine *e, PassSlot *slot) {
    if (e->active_status.load(std::memory_order_acquire) != 0) {
        slot->program.flags &= ~PREFILL_FLAG_READOUT_DEPTH;
        slot->program.phase = PREFILL_PHASE_DONE;
        return false;
    }
    if ((slot->program.flags & PREFILL_FLAG_READOUT_DEPTH) != 0) {
        /* Test-only readout tail. Halt the depth program at the codebook-0
         * logits: the readout wants the head distribution, never a sampled
         * code, so no sampler state is consumed and no embedding follows. */
        if (slot->program.phase == DEPTH_PHASE_HEAD_LOGITS) {
            LfmListenReadoutForTest *readout = slot->prefill.readout;
            fill_listen_readout_head(
                e->depth_scratch.logits_b.data(),
                slot->depth->heads[0].vocab, SAMPLE_BF16,
                readout->audio_ids, readout->audio_logprobs,
                &readout->audio_entropy);
            slot->program.flags &= ~PREFILL_FLAG_READOUT_DEPTH;
            slot->program.phase = PREFILL_PHASE_DONE;
            return false;
        }
        return advance_depth_program(e, slot);
    }
    switch (slot->program.phase) {
    case PREFILL_PHASE_CONV_NORM:
        slot->program.phase = PREFILL_PHASE_CONV_IN;
        return true;
    case PREFILL_PHASE_CONV_IN:
        slot->program.phase = PREFILL_PHASE_CONV_FIR;
        return true;
    case PREFILL_PHASE_CONV_FIR:
        slot->program.phase = PREFILL_PHASE_CONV_OUT;
        return true;
    case PREFILL_PHASE_CONV_OUT:
        slot->program.phase = PREFILL_PHASE_CONV_ADD;
        return true;
    case PREFILL_PHASE_CONV_ADD:
        slot->program.phase = PREFILL_PHASE_MLP_NORM;
        return true;
    case PREFILL_PHASE_ATTN_NORM:
        slot->program.phase = PREFILL_PHASE_ATTN_Q;
        return true;
    case PREFILL_PHASE_ATTN_Q:
        slot->program.phase = PREFILL_PHASE_ATTN_K;
        return true;
    case PREFILL_PHASE_ATTN_K:
        slot->program.phase = PREFILL_PHASE_ATTN_V;
        return true;
    case PREFILL_PHASE_ATTN_V:
        slot->program.phase = PREFILL_PHASE_ATTN_PREP;
        return true;
    case PREFILL_PHASE_ATTN_PREP:
        slot->program.phase = PREFILL_PHASE_ATTN_MIX;
        return true;
    case PREFILL_PHASE_ATTN_MIX:
        slot->program.phase = PREFILL_PHASE_ATTN_OUT;
        return true;
    case PREFILL_PHASE_ATTN_OUT:
        slot->program.phase = PREFILL_PHASE_ATTN_ADD;
        return true;
    case PREFILL_PHASE_ATTN_ADD:
        slot->program.phase = PREFILL_PHASE_MLP_NORM;
        return true;
    case PREFILL_PHASE_MLP_NORM:
        slot->program.phase = PREFILL_PHASE_MLP_GATE;
        return true;
    case PREFILL_PHASE_MLP_GATE:
        slot->program.phase = PREFILL_PHASE_MLP_UP;
        return true;
    case PREFILL_PHASE_MLP_UP:
        slot->program.phase = PREFILL_PHASE_MLP_SWIGLU;
        return true;
    case PREFILL_PHASE_MLP_SWIGLU:
        slot->program.phase = PREFILL_PHASE_MLP_DOWN;
        return true;
    case PREFILL_PHASE_MLP_DOWN:
        slot->program.phase = PREFILL_PHASE_MLP_ADD;
        return true;
    case PREFILL_PHASE_MLP_ADD:
        ++slot->program.outer;
        return begin_prefill_layer(e, slot);
    case PREFILL_PHASE_FINAL_NORM:
        if (!slot->prefill.sample) {
            slot->program.phase = PREFILL_PHASE_DONE;
            return false;
        }
        slot->program.phase = PREFILL_PHASE_FINAL_LOGITS;
        return true;
    case PREFILL_PHASE_FINAL_LOGITS:
        configure_prefill_sample(e, slot);
        slot->program.phase = PREFILL_PHASE_SAMPLE;
        return true;
    case PREFILL_PHASE_SAMPLE:
        if (advance_sample_program(e, slot)) return true;
        if (slot->prefill.readout) {
            /* Test-only: report the text head from the very plane the sample
             * fold consumed, then evaluate the codebook-0 head by running the
             * ordinary depth program on the just-published hidden. */
            LfmListenReadoutForTest *readout = slot->prefill.readout;
            fill_listen_readout_head(
                slot->prefill.workspace->logits.data(), slot->model->vocab,
                SAMPLE_F32, readout->text_ids, readout->text_logprobs,
                &readout->text_entropy);
            slot->program.outer = 0;
            slot->program.inner = 0;
            slot->program.flags |= PREFILL_FLAG_READOUT_DEPTH;
            slot->program.phase = DEPTH_PHASE_PROJECT;
            return true;
        }
        slot->program.phase = PREFILL_PHASE_DONE;
        return false;
    case PREFILL_PHASE_DONE:
        return false;
    default:
        std::abort();
    }
}

static void sample_band(const SampleReq &sample, uint32_t lane,
                        uint32_t lanes, size_t *begin, size_t *end) {
    const size_t chunk = (sample.count + lanes - 1) / lanes;
    *begin = std::min(static_cast<size_t>(lane) * chunk, sample.count);
    *end = std::min(*begin + chunk, sample.count);
}

static void run_sample_pass(Engine *e, uint32_t lane) {
    PassSlot *slot = e->active_slot;
    if (!slot || (slot->request != REQ_TOKEN_PASS &&
                  slot->request != REQ_DEPTH_FRAME &&
                  slot->request != REQ_PREFILL)) {
        if (lane == 0)
            e->active_status.store(-EFAULT, std::memory_order_release);
        return;
    }
    const SampleReq &sample = slot->sample;
    size_t begin = 0, end = 0;
    sample_band(sample, lane, e->lanes_total, &begin, &end);
    const float scale = sample.phase == SAMPLE_PHASE_GREEDY
        ? 1.0f
        : static_cast<float>(1.0 / sample.config.temperature);
    const uint16_t bf16_scale = rb_bits(scale);
    switch (sample.phase) {
    case SAMPLE_PHASE_GREEDY: {
        uint32_t local = 0;
        if (end > begin) {
            local = sample.dtype == SAMPLE_F32
                ? lfm_sampler_argmax_f32(
                      static_cast<const float *>(sample.logits) + begin,
                      end - begin)
                : lfm_sampler_argmax_bf16(
                      static_cast<const uint16_t *>(sample.logits) + begin,
                      end - begin);
        }
        e->sample_lane_index[lane] = static_cast<uint32_t>(begin + local);
        e->sample_lane_value[lane] = end > begin
            ? sample_raw(sample, begin + local)
            : -std::numeric_limits<float>::infinity();
        return;
    }
    case SAMPLE_PHASE_MAXIMUM: {
        float maximum = -std::numeric_limits<float>::infinity();
        uint32_t index = static_cast<uint32_t>(begin);
        for (size_t i = begin; i < end; ++i) {
            const float value = sample_scaled(sample, i, scale, bf16_scale);
            if (value > maximum) {
                maximum = value;
                index = static_cast<uint32_t>(i);
            }
        }
        e->sample_lane_value[lane] = maximum;
        e->sample_lane_index[lane] = index;
        return;
    }
    case SAMPLE_PHASE_THRESHOLD:
        if (lane == 0) {
            e->sample_threshold = sample_topk_threshold(
                sample, scale, bf16_scale, e->sample_heap.data());
        }
        return;
    case SAMPLE_PHASE_EXP_SUM: {
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
        return;
    }
    case SAMPLE_PHASE_PICK:
        if (e->sample_winner_lane == lane) {
            e->sample_lane_index[lane] = static_cast<uint32_t>(begin) +
                lfm_sampler_prefix_pick(e->sample_weights.data() + begin,
                                        end - begin, e->sample_target);
        }
        return;
    default:
        if (lane == 0)
            e->active_status.store(-EPROTO, std::memory_order_release);
        return;
    }
}

static bool advance_sample_program(Engine *e, PassSlot *slot) {
    if (!e || !slot || slot != e->active_slot ||
        (slot->request != REQ_TOKEN_PASS &&
         slot->request != REQ_DEPTH_FRAME &&
         slot->request != REQ_PREFILL)) {
        std::abort();
    }
    SampleReq &sample = slot->sample;
    const uint32_t lanes = e->lanes_total;
    switch (sample.phase) {
    case SAMPLE_PHASE_GREEDY: {
        float best = -std::numeric_limits<float>::infinity();
        uint32_t index = 0;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const float value = e->sample_lane_value[lane];
            const uint32_t candidate = e->sample_lane_index[lane];
            if (value > best || (value == best && candidate < index)) {
                best = value;
                index = candidate;
            }
        }
        *sample.out = index;
        sample.phase = SAMPLE_PHASE_DONE;
        return false;
    }
    case SAMPLE_PHASE_MAXIMUM:
        e->sample_maximum = -std::numeric_limits<float>::infinity();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            if (e->sample_lane_value[lane] > e->sample_maximum) {
                e->sample_maximum = e->sample_lane_value[lane];
            }
        }
        sample.phase = SAMPLE_PHASE_THRESHOLD;
        return true;
    case SAMPLE_PHASE_THRESHOLD:
        sample.phase = SAMPLE_PHASE_EXP_SUM;
        return true;
    case SAMPLE_PHASE_EXP_SUM: {
        float total = 0.0f;
        for (uint32_t lane = 0; lane < lanes; ++lane)
            total += e->sample_lane_sum[lane];
        if (!(total > 0.0f) || !std::isfinite(total)) {
            float best = -std::numeric_limits<float>::infinity();
            uint32_t index = 0;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                if (e->sample_lane_value[lane] > best) {
                    best = e->sample_lane_value[lane];
                    index = e->sample_lane_index[lane];
                }
            }
            *sample.out = index;
            e->sample_winner_lane = UINT32_MAX;
            sample.phase = SAMPLE_PHASE_DONE;
            return false;
        }
        uint64_t draw = 0;
        if (lfm_prng_fill_u64(sample.state, &draw, 1) != 0) {
            *sample.out = e->sample_lane_index[0];
            e->sample_winner_lane = UINT32_MAX;
            sample.phase = SAMPLE_PHASE_DONE;
            return false;
        }
        const double unit = static_cast<double>(draw >> 11) * 0x1.0p-53;
        float target = static_cast<float>(unit * static_cast<double>(total));
        if (target >= total) target = std::nextafter(total, 0.0f);
        float prefix = 0.0f;
        e->sample_winner_lane = lanes - 1;
        e->sample_target = target;
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const float next = prefix + e->sample_lane_sum[lane];
            if (target < next) {
                e->sample_winner_lane = lane;
                e->sample_target = target - prefix;
                break;
            }
            prefix = next;
        }
        sample.phase = SAMPLE_PHASE_PICK;
        return true;
    }
    case SAMPLE_PHASE_PICK:
        if (e->sample_winner_lane != UINT32_MAX) {
            *sample.out = e->sample_lane_index[e->sample_winner_lane];
        }
        sample.phase = SAMPLE_PHASE_DONE;
        return false;
    default:
        std::abort();
    }
}

static void configure_token_final_stats(Engine *e, PassSlot *slot) {
    ScPass *norm = &slot->sc;
    norm->x = slot->token_program.hidden;
    norm->norm_w = slot->model->emb_norm_w;
    norm->h = slot->model->h;
    norm->xn = slot->tok.out_hidden;
    norm->rs_bits.store(0, std::memory_order_relaxed);
    slot->token_program.kind = TOKEN_PROGRAM_FINAL_STATS;
    configure_serial_stage(&slot->stage);
    e->sc_view = norm;
}

static bool configure_token_layer(Engine *e, PassSlot *slot) {
    const size_t layer = static_cast<size_t>(slot->program.outer);
    if (layer >= slot->model->layers.size()) {
        configure_token_final_stats(e, slot);
        return true;
    }
    const LfmLayerDesc *desc = &slot->model->layers[layer];
    const LfmLayerState *state = &slot->tok.states[layer];
    if (desc->kind == 0) {
        slot->conv = {
            .layer = layer,
            .x = slot->token_program.hidden,
            .state_in = state->conv_state,
            .state_out = state->conv_state,
            .out = slot->token_program.next,
            .lanes = slot->tok.lanes,
        };
        slot->token_program.kind = TOKEN_PROGRAM_CONV;
        initialize_conv_program(e, slot);
        return true;
    }
    if (desc->kind == 1) {
        slot->attn = {
            .layer = layer,
            .x = slot->token_program.hidden,
            .k_plane = state->k_plane,
            .v_plane = state->v_plane,
            .head_stride = state->head_stride,
            .pos = slot->tok.pos,
            .cos_base = slot->tok.cos_base,
            .sin_base = slot->tok.sin_base,
            .out = slot->token_program.next,
            .lanes = slot->tok.lanes,
        };
        slot->token_program.kind = TOKEN_PROGRAM_ATTN;
        initialize_attn_program(e, slot);
        return true;
    }
    e->active_status.store(-EINVAL, std::memory_order_release);
    slot->token_program.kind = TOKEN_PROGRAM_DONE;
    configure_serial_stage(&slot->stage);
    return false;
}

static void initialize_token_program(Engine *e, PassSlot *slot) {
    if (!e || !slot || slot->request != REQ_TOKEN_PASS || !slot->model) {
        std::abort();
    }
    TokenProgram *program = &slot->token_program;
    program->next = e->tk_h1.data();
    slot->program.outer = 0;
    slot->program.inner = 0;
    if (slot->tok.embed_kind == 0) {
        program->hidden = Bf16Input::from_resident(weight_offset(
            slot->model->embed_w,
            static_cast<size_t>(slot->tok.ids[0]) * slot->model->h));
        configure_token_layer(e, slot);
        return;
    }
    if (slot->tok.embed_kind == 2) {
        program->hidden =
            Bf16Input::from_activation(slot->tok.provided_embed);
        configure_token_layer(e, slot);
        return;
    }
    if (slot->tok.embed_kind == 1) {
        program->hidden = Bf16Input::from_activation(e->tk_h0.data());
        program->kind = TOKEN_PROGRAM_EMBED;
        configure_serial_stage(&slot->stage);
        return;
    }
    e->active_status.store(-EINVAL, std::memory_order_release);
    program->kind = TOKEN_PROGRAM_DONE;
    configure_serial_stage(&slot->stage);
}

static void run_token_serial_stage(Engine *e, uint32_t lane, PassSlot *slot) {
    if (lane != 0) return;
    if (slot->token_program.kind == TOKEN_PROGRAM_EMBED) {
        uint16_t *hidden = e->tk_h0.data();
        std::memset(hidden, 0, slot->model->h * sizeof(uint16_t));
        for (size_t code = 0; code < slot->tok.n_ids; ++code) {
            const WeightBytes row = weight_offset(
                slot->model->audio_embed_w,
                static_cast<size_t>(slot->tok.ids[code]) * slot->model->h);
            lfm_bf16_add(hidden, row, hidden,
                         static_cast<int>(slot->model->h));
        }
        return;
    }
    if (slot->token_program.kind == TOKEN_PROGRAM_FINAL_STATS) {
        ScPass *norm = &slot->sc;
        const float total = lfm_bf16_sumsq_ordered_f32(
            norm->x.data(), static_cast<int>(norm->h));
        const float inv = lfm_inv_rms_f32(
            total, norm->h, slot->model->emb_norm_eps);
        uint32_t bits = 0;
        std::memcpy(&bits, &inv, sizeof(bits));
        norm->rs_bits.store(bits, std::memory_order_release);
        return;
    }
    std::abort();
}

static void configure_token_final_norm(PassSlot *slot) {
    const size_t h = slot->model->h;
    const size_t tiles = std::min(h, slot->tok.lanes);
    const uint32_t chunk = static_cast<uint32_t>((h + tiles - 1) / tiles);
    slot->token_program.kind = TOKEN_PROGRAM_FINAL_NORM;
    configure_stage(&slot->stage, ST_SC_NORM,
                    static_cast<uint32_t>((h + chunk - 1) / chunk), chunk);
}

static void configure_token_logits(Engine *e, PassSlot *slot) {
    const size_t workers = static_cast<size_t>(e->n_workers) * 4;
    const uint32_t chunk = static_cast<uint32_t>(
        (slot->model->vocab + workers - 1) / workers);
    slot->token_program.kind = TOKEN_PROGRAM_LOGITS;
    configure_stage(&slot->stage, ST_LOGITS,
                    static_cast<uint32_t>(
                        (slot->model->vocab + chunk - 1) / chunk),
                    chunk);
}

static void configure_token_sample(Engine *e, PassSlot *slot) {
    slot->sample = {
        .logits = slot->tok.out_logits ? slot->tok.out_logits
                                       : e->tk_logf.data(),
        .count = slot->model->vocab,
        .dtype = SAMPLE_F32,
        .config = *slot->tok.sampler,
        .state = slot->tok.sample_state,
        .out = slot->tok.out_token,
    };
    const bool greedy =
        (slot->sample.config.flags & LFM_SAMPLE_FLAG_GREEDY) != 0 ||
        slot->sample.config.top_k == 1;
    slot->sample.phase = greedy ? SAMPLE_PHASE_GREEDY
                                : SAMPLE_PHASE_MAXIMUM;
    slot->token_program.kind = TOKEN_PROGRAM_SAMPLE;
    configure_serial_stage(&slot->stage);
}

static void finish_token_layer(Engine *e, PassSlot *slot) {
    uint16_t *completed = slot->token_program.next;
    slot->token_program.hidden = Bf16Input::from_activation(completed);
    slot->token_program.next = completed == e->tk_h0.data()
        ? e->tk_h1.data()
        : e->tk_h0.data();
    ++slot->program.outer;
    configure_token_layer(e, slot);
}

static bool advance_token_program(Engine *e, PassSlot *slot) {
    switch (slot->token_program.kind) {
    case TOKEN_PROGRAM_EMBED:
        configure_token_layer(e, slot);
        return slot->token_program.kind != TOKEN_PROGRAM_DONE;
    case TOKEN_PROGRAM_CONV:
        if (advance_conv_program(slot)) return true;
        finish_token_layer(e, slot);
        return slot->token_program.kind != TOKEN_PROGRAM_DONE;
    case TOKEN_PROGRAM_ATTN:
        if (advance_attn_program(slot)) return true;
        finish_token_layer(e, slot);
        return slot->token_program.kind != TOKEN_PROGRAM_DONE;
    case TOKEN_PROGRAM_FINAL_STATS:
        configure_token_final_norm(slot);
        return true;
    case TOKEN_PROGRAM_FINAL_NORM:
        if (slot->tok.out_logits || slot->tok.out_token) {
            configure_token_logits(e, slot);
            return true;
        }
        slot->token_program.kind = TOKEN_PROGRAM_DONE;
        return false;
    case TOKEN_PROGRAM_LOGITS:
        if (slot->tok.out_token) {
            configure_token_sample(e, slot);
            return true;
        }
        slot->token_program.kind = TOKEN_PROGRAM_DONE;
        return false;
    case TOKEN_PROGRAM_SAMPLE:
        if (advance_sample_program(e, slot)) return true;
        slot->token_program.kind = TOKEN_PROGRAM_DONE;
        return false;
    case TOKEN_PROGRAM_DONE:
        return false;
    default:
        std::abort();
    }
}

static void run_token_program_stage(Engine *e, uint32_t lane, PassSlot *slot) {
    switch (slot->token_program.kind) {
    case TOKEN_PROGRAM_EMBED:
    case TOKEN_PROGRAM_FINAL_STATS:
        run_token_serial_stage(e, lane, slot);
        return;
    case TOKEN_PROGRAM_CONV:
        if (slot->program.phase == CONV_PHASE_STATS) {
            run_serial_program_stage(e, lane, slot);
        } else {
            run_slot_stage(e, lane, slot);
        }
        return;
    case TOKEN_PROGRAM_ATTN:
        if (slot->program.phase == ATTN_PHASE_STATS) {
            run_serial_program_stage(e, lane, slot);
        } else {
            run_slot_stage(e, lane, slot);
        }
        return;
    case TOKEN_PROGRAM_FINAL_NORM:
    case TOKEN_PROGRAM_LOGITS:
        run_slot_stage(e, lane, slot);
        return;
    case TOKEN_PROGRAM_SAMPLE:
        run_sample_pass(e, lane);
        return;
    case TOKEN_PROGRAM_DONE:
        return;
    default:
        if (lane == 0)
            e->active_status.store(-EPROTO, std::memory_order_release);
        return;
    }
}

static void run_gemm(Engine *e, uint32_t lane) {
    const GemmReq &request = e->gemm;
    const size_t lanes = e->lanes_total;

    if (request.bf16_epilogue) {
        const size_t columns = (request.n + lanes - 1) / lanes;
        const size_t column = (size_t)lane * columns;
        if (column >= request.n) return;
        const size_t count = std::min(columns, request.n - column);
        const auto *weights = static_cast<const unsigned char *>(request.rhs) +
            column * request.k * sizeof(uint16_t);
        const auto *bias = request.bias
            ? static_cast<const unsigned char *>(request.bias) +
                  column * sizeof(uint16_t)
            : nullptr;
        lfm_bf16_gemm_nt_bias_bf16(
            request.a, weights, bias, request.out_bf16 + column,
            (int)request.m, (int)count, (int)request.k,
            (int)request.output_stride);
        return;
    }

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
    lfm_bf16_gemm_nt_f32(request.a + row * request.k, request.rhs,
                         request.out + row * request.n, (int)count,
                         (int)request.n, (int)request.k);
}

enum : uint32_t {
    AUDIO_PHASE_FRONTEND = 0,
    AUDIO_PHASE_CONFORMER = 1,
    AUDIO_PHASE_DONE = 2,
};

static int configure_audio_gemm(Engine *e, PassSlot *slot) {
    if (!e || !slot || slot->request != REQ_AUDIO_ENCODE)
        return -EINVAL;
    if (lfm_conformer_program_stage(&slot->audio.conformer) !=
        LFM_CONFORMER_STAGE_GEMM) {
        return 0;
    }
    LfmConformerGemmStage stage{};
    const int status =
        lfm_conformer_program_gemm(&slot->audio.conformer, &stage);
    if (status != 0) return status;
    size_t activation_need = 0, weight_need = 0, output_need = 0;
    if (!stage.activation || !stage.weight_bytes || !stage.out ||
        stage.rows == 0 || stage.columns == 0 || stage.inner == 0 ||
        stage.rows > INT_MAX || stage.columns > INT_MAX ||
        stage.inner > INT_MAX ||
        !checked_size_product(stage.rows, stage.inner, &activation_need) ||
        !checked_size_product(stage.columns, stage.inner, &weight_need) ||
        !checked_size_product(stage.rows, stage.columns, &output_need) ||
        activation_need != stage.activation_count ||
        weight_need != stage.weight_count || output_need != stage.out_count ||
        ((stage.bias_bytes == nullptr) != (stage.bias_count == 0)) ||
        (stage.bias_bytes && stage.bias_count != stage.columns)) {
        return -EINVAL;
    }
    slot->gemm = {
        .a = stage.activation,
        .rhs = stage.weight_bytes,
        .bias = stage.bias_bytes,
        .out_bf16 = stage.out,
        .m = stage.rows,
        .n = stage.columns,
        .k = stage.inner,
        .output_stride = stage.columns,
        .rhs_layout = LFM_GEMM_RHS_NK,
        .direct = true,
        .bf16_epilogue = true,
    };
    e->gemm = slot->gemm;
    return 0;
}

static int prepare_audio_conformer(Engine *e, PassSlot *slot) {
    AudioReq &audio = slot->audio;
    const LfmAudioEncodePassV2 &pass = audio.pass;
    LfmF32SpanChain samples{};
    int status = lfm_resampler_process_spans(
        pass.resampler, pass.resampler_workspace, &pass.pcm, pass.resampled,
        pass.resampled_capacity, &samples);
    if (status != 0) return status;

    audio.frames = lfm_frontend_seq_len(pass.frontend, samples.length);
    if (audio.frames == 0) return -EINVAL;
    status = lfm_frontend_forward_bf16_spans_workspace(
        pass.frontend, pass.frontend_workspace, &samples, pass.mel,
        pass.mel_capacity);
    if (status != 0) return status;

    const uint64_t rows =
        lfm_conformer_out_rows(pass.conformer, audio.frames);
    const uint64_t width = lfm_conformer_out_width(pass.conformer);
    if (rows == 0 || width == 0 || rows > UINT64_MAX / width)
        return rows == 0 || width == 0 ? -EINVAL : -EOVERFLOW;
    audio.adapted_values = rows * width;
    if (audio.adapted_values > pass.adapted_capacity) return -ENOBUFS;
    if (e->model && width != e->model->h) return -ESTALE;

    return lfm_conformer_program_begin(
        &audio.conformer, pass.conformer, pass.conformer_workspace,
        pass.mel, audio.frames, pass.adapted, pass.adapted_capacity);
}

static void run_audio_program_stage(Engine *e, uint32_t lane) {
    PassSlot *slot = e->active_slot;
    if (!slot || slot->request != REQ_AUDIO_ENCODE) {
        if (lane == 0) e->active_status.store(-EFAULT, std::memory_order_release);
        return;
    }
    AudioReq &audio = slot->audio;
    if (audio.phase == AUDIO_PHASE_FRONTEND) {
        if (lane != 0) return;
        e->audio_encode_passes.fetch_add(1, std::memory_order_relaxed);
        const int status = prepare_audio_conformer(e, slot);
        if (status != 0)
            e->active_status.store(status, std::memory_order_release);
        return;
    }
    if (audio.phase == AUDIO_PHASE_CONFORMER) {
        const uint32_t stage =
            lfm_conformer_program_stage(&audio.conformer);
        if (stage == LFM_CONFORMER_STAGE_GEMM) {
            run_gemm(e, lane);
            return;
        }
        if (stage == LFM_CONFORMER_STAGE_SERIAL) {
            if (lane == 0) {
                const int status =
                    lfm_conformer_program_run_serial(&audio.conformer);
                if (status != 0)
                    e->active_status.store(status,
                                           std::memory_order_release);
            }
            return;
        }
        if (stage == LFM_CONFORMER_STAGE_DONE) return;
    }
    if (lane == 0)
        e->active_status.store(-EPROTO, std::memory_order_release);
}

static bool advance_audio_program(Engine *e, PassSlot *slot) {
    AudioReq &audio = slot->audio;
    if (e->active_status.load(std::memory_order_acquire) != 0)
        return false;
    if (audio.phase == AUDIO_PHASE_FRONTEND) {
        audio.phase = AUDIO_PHASE_CONFORMER;
    } else if (audio.phase == AUDIO_PHASE_CONFORMER) {
        const int status =
            lfm_conformer_program_advance(&audio.conformer);
        if (status < 0) {
            e->active_status.store(status, std::memory_order_release);
            return false;
        }
        if (status == 0) {
            audio.phase = AUDIO_PHASE_DONE;
            return false;
        }
    } else {
        e->active_status.store(-EPROTO, std::memory_order_release);
        return false;
    }
    const int configure_status = configure_audio_gemm(e, slot);
    if (configure_status != 0) {
        e->active_status.store(configure_status, std::memory_order_release);
        return false;
    }
    return true;
}

static void publish_detokenizer_error(Engine *e, int status) {
    if (status == 0) return;
    int expected = 0;
    e->active_status.compare_exchange_strong(
        expected, status, std::memory_order_release,
        std::memory_order_acquire);
}

static void initialize_detokenizer_program(Engine *e, PassSlot *slot) {
    DetokenizerReq &request = slot->detokenizer;
    request.resample_pending = false;
    request.program = {};
    if (request.completion_status != 0) {
        publish_detokenizer_error(e, request.completion_status);
        return;
    }
    float *decode_pcm = request.resampler_stream
        ? request.detokenizer_pcm
        : request.pcm;
    const size_t decode_capacity = request.resampler_stream
        ? request.detokenizer_capacity
        : request.capacity;
    const int status = lfm_detokenizer_program_begin(
        &request.program, request.state, request.codes, decode_pcm,
        decode_capacity, request.flush ? 1u : 0u);
    publish_detokenizer_error(e, status);
}

static void run_detokenizer_program_stage(Engine *e, uint32_t lane) {
    PassSlot *slot = e->active_slot;
    if (!slot || e->active_status.load(std::memory_order_acquire) != 0)
        return;
    DetokenizerReq &request = slot->detokenizer;
    if (request.resample_pending) {
        if (lane != 0) return;
        LfmF32Span span{};
        const int status = lfm_resampler_stream_process(
            request.resampler_stream, request.detokenizer_pcm,
            request.program.produced, request.pcm, request.capacity, &span);
        if (status != 0 || span.data != request.pcm ||
            span.length > request.capacity) {
            publish_detokenizer_error(
                e, status != 0 ? status : -EFAULT);
            return;
        }
        *request.out_samples = static_cast<size_t>(span.length);
        return;
    }
    publish_detokenizer_error(
        e, lfm_detokenizer_program_run(
               &request.program, lane,
               static_cast<uint32_t>(e->lanes_total)));
}

static bool advance_detokenizer_program(Engine *e, PassSlot *slot) {
    DetokenizerReq &request = slot->detokenizer;
    if (e->active_status.load(std::memory_order_acquire) != 0) {
        if (request.program.active)
            lfm_detokenizer_program_cancel(&request.program);
        request.resample_pending = false;
        return false;
    }
    if (request.resample_pending) {
        request.resample_pending = false;
        return false;
    }
    const int status = lfm_detokenizer_program_advance(&request.program);
    if (status < 0) {
        publish_detokenizer_error(e, status);
        lfm_detokenizer_program_cancel(&request.program);
        return false;
    }
    if (status > 0) return true;
    const size_t samples = request.program.produced;
    const size_t decode_capacity = request.resampler_stream
        ? request.detokenizer_capacity
        : request.capacity;
    if (samples > decode_capacity) {
        publish_detokenizer_error(e, -EOVERFLOW);
        return false;
    }
    if (request.resampler_stream) {
        request.resample_pending = true;
        return true;
    }
    *request.out_samples = samples;
    return false;
}

// The per-generation program is dispatched identically on every lane. Request
// payloads are release-published before dispatch and remain borrowed until the
// fixed team's final-return callback publishes the exact ticket completion.
// There is deliberately no trailing lane fence: the final return is the quorum.
static void lane_program(Engine *e, uint32_t lane) {
    switch (e->cur_req) {
    case REQ_CONV_LAYER:
        if (e->active_slot->program.phase == CONV_PHASE_STATS) {
            run_serial_program_stage(e, lane, e->active_slot);
        } else {
            run_slot_stage(e, lane, e->active_slot);
        }
        break;
    case REQ_ATTN_LAYER:
        if (e->active_slot->program.phase == ATTN_PHASE_STATS) {
            run_serial_program_stage(e, lane, e->active_slot);
        } else {
            run_slot_stage(e, lane, e->active_slot);
        }
        break;
    case REQ_TOKEN_PASS:
        run_token_program_stage(e, lane, e->active_slot);
        break;
    case REQ_PREFILL:
        run_prefill_program_stage(e, lane, e->active_slot);
        break;
    case REQ_DEPTH_FRAME:
        run_depth_program_stage(e, lane, e->active_slot);
        break;
    case REQ_AUDIO_DETOKENIZE:
        run_detokenizer_program_stage(e, lane);
        break;
    case REQ_AUDIO_ENCODE:
        run_audio_program_stage(e, lane);
        break;
    default:
        // A request selector is a closed protocol value.  This is a final
        // defense behind submission/descriptor validation: corruption must
        // become a failed completion, never a successful no-op.
        if (lane == 0)
            e->active_status.store(-EINVAL, std::memory_order_release);
        break;
    }
}

static bool ticket_equal(const kc_ticket_id &a, const kc_ticket_id &b) {
    return kc::ticket_equal(a, b);
}

static uint64_t expected_team_mask(const Engine *e) {
    if (!e || e->lanes_total == 0 || e->lanes_total > 64) std::abort();
    return e->lanes_total == 64
               ? UINT64_MAX
               : (UINT64_C(1) << e->lanes_total) - 1;
}

static TeamWorkDescriptor describe_active_generation(const Engine *e) {
    if (!e || !e->active_slot || e->cur_req == REQ_NONE ||
        e->active_slot->request != e->cur_req) {
        std::abort();
    }
    const PassSlot &slot = *e->active_slot;
    TeamWorkDescriptor work = {
        .request = static_cast<uint32_t>(slot.request),
        .stage = slot.stage.kind,
        .program_phase = slot.program.phase,
        .program_flags = slot.program.flags,
        .program_outer = slot.program.outer,
        .program_inner = slot.program.inner,
    };
    switch (slot.request) {
    case REQ_CONV_LAYER:
        work.shape0 = slot.model ? slot.model->h : 0;
        work.shape1 = slot.model ? slot.model->ffn : 0;
        work.shape2 = slot.conv.layer;
        break;
    case REQ_ATTN_LAYER:
        work.shape0 = slot.model ? slot.model->h : 0;
        work.shape1 = slot.attn.pos;
        work.shape2 = slot.attn.layer;
        break;
    case REQ_TOKEN_PASS:
        work.program_kind = slot.token_program.kind;
        work.shape0 = slot.model ? slot.model->h : 0;
        work.shape1 = slot.tok.pos;
        work.shape2 = slot.tok.n_ids;
        break;
    case REQ_PREFILL:
        work.shape0 = slot.prefill.rows;
        work.shape1 = slot.model ? slot.model->h : 0;
        work.shape2 = slot.prefill.pos;
        break;
    case REQ_DEPTH_FRAME:
        work.shape0 = slot.depth ? slot.depth->dim : 0;
        work.shape1 = slot.depth ? slot.depth->codebooks : 0;
        work.shape2 = slot.depth ? slot.depth->layers.size() : 0;
        break;
    case REQ_AUDIO_DETOKENIZE:
        work.program_kind = slot.detokenizer.resample_pending ? 1u : 0u;
        work.program_phase = slot.detokenizer.resample_pending
            ? LFM_DETOKENIZER_PHASE_DONE
            : slot.detokenizer.program.phase;
        work.program_outer = slot.detokenizer.program.layer;
        work.shape0 = slot.detokenizer.capacity;
        work.shape1 = slot.detokenizer.detokenizer_capacity;
        work.shape2 = slot.detokenizer.program.produced;
        break;
    case REQ_AUDIO_ENCODE:
        work.program_kind = slot.audio.phase;
        work.shape0 = slot.audio.frames;
        work.shape1 = slot.audio.adapted_values;
        work.shape2 = slot.audio.phase == AUDIO_PHASE_CONFORMER
            ? lfm_conformer_program_stage(&slot.audio.conformer)
            : 0;
        break;
    default:
        break;
    }
    return work;
}

/* Closed production budget table. Every mounted request family has an explicit
 * conservative hard ceiling; an unknown selector returns zero and
 * kc::TeamSupervisor rejects it before the team generation is dispatched. */
static uint64_t team_hard_budget_ns(bool hard_timeout_probe,
                                    const TeamWorkDescriptor &work) {
    if (hard_timeout_probe) return TEAM_PROBE_HARD_DEADLINE_NS;
    switch (work.request) {
    case REQ_CONV_LAYER:
    case REQ_ATTN_LAYER:
    case REQ_TOKEN_PASS:
    case REQ_DEPTH_FRAME:
    case REQ_PREFILL:
    case REQ_AUDIO_DETOKENIZE:
    case REQ_AUDIO_ENCODE:
        return TEAM_PROBE_HARD_DEADLINE_NS;
    default:
        return 0;
    }
}

static void fill_team_fatal_capsule(
    void *, TeamFatalCapsule &capsule,
    const FlashkernRequest &submission,
    const TeamWorkDescriptor &work,
    const kc_deadline_event &event, uint64_t armed_ns,
    uint64_t budget_ns, uint64_t elapsed_ns,
    int quorum_status,
    const kc_team_quorum_snapshot &quorum) noexcept {
    const uint64_t never_entered =
        quorum.expected_mask & ~quorum.entered_mask;
    const uint64_t entered_not_returned =
        quorum.entered_mask & ~quorum.returned_mask;
    capsule = {
        .size = sizeof(capsule),
        .abi_version = 2,
        .request = work.request,
        .stage = work.stage,
        .program_kind = work.program_kind,
        .program_phase = work.program_phase,
        .program_flags = work.program_flags,
        .quorum_status = quorum_status,
        .workflow = submission.parent,
        .pass = submission.ticket,
        .deadline = event.child,
        .conversation_id = submission.conversation_id,
        .epoch = submission.epoch,
        .scope_generation = event.scope_generation,
        .team_generation = event.team_generation,
        .program_outer = work.program_outer,
        .program_inner = work.program_inner,
        .shape0 = work.shape0,
        .shape1 = work.shape1,
        .shape2 = work.shape2,
        .expected_mask = quorum.expected_mask,
        .entered_mask = quorum.entered_mask,
        .returned_mask = quorum.returned_mask,
        .never_entered_mask = never_entered,
        .entered_not_returned_mask = entered_not_returned,
        .armed_ns = armed_ns,
        .hard_budget_ns = budget_ns,
        .elapsed_ns = elapsed_ns,
        .deadline_event_sequence = event.sequence,
        .scheduled_arm_generation = event.scheduled_arm_generation,
        .current_arm_generation = event.current_arm_generation,
    };
}

static void hard_timeout_probe_ready(void *context, uint64_t generation) {
    Engine *e = static_cast<Engine *>(context);
    const kc::TeamSupervisorSnapshot snapshot =
        e ? e->supervisor.snapshot() : kc::TeamSupervisorSnapshot{};
    if (!e || !e->hard_timeout_probe || !e->manual_deadlines ||
        generation != snapshot.generation ||
        snapshot.arm_team_generation != generation ||
        snapshot.budget_ns != TEAM_PROBE_HARD_DEADLINE_NS) {
        std::abort();
    }
    if (e->supervisor.expire_manual_test(generation) != 0) {
        std::abort();
    }
}

static int begin_team_supervision(
    Engine *e, uint64_t generation) noexcept {
    if (!e || !e->active_slot ||
        e->active_submission.lease_generation == 0) {
        return -EINVAL;
    }
    e->supervised_work = describe_active_generation(e);
    const uint64_t budget =
        team_hard_budget_ns(e->hard_timeout_probe, e->supervised_work);
    if (budget == 0) return -ENODATA;
    return e->supervisor.begin(
        e->active_submission, e->supervised_work,
        {
        .parent = e->active_submission.ticket,
        .scope_generation = e->active_submission.lease_generation,
        .epoch = e->active_submission.epoch,
        .domain = e->tickets.epoch(),
        },
        generation, budget);
}

static int executor_begin(void *context,
                          const FlashkernRequest &submission) noexcept {
    Engine *e = static_cast<Engine *>(context);
    const PassSlotPool::Lease lease = {
        .index = submission.slot,
        .generation = submission.lease_generation,
    };
    PassSlot *slot = e ? e->slots.get(lease) : nullptr;
    const uint64_t owner = submission.lease_generation;
    bool valid = slot && slot_state(slot) == PASS_SLOT_SUBMITTED;
    if (valid) {
        valid = request_kind_valid(submission.operation) &&
                slot->request == static_cast<int>(submission.operation) &&
                submission.ticket.kind == KC_TICKET_KIND_PASS &&
                submission.parent.runtime_epoch == e->tickets.epoch() &&
                submission.parent.sequence != 0 &&
                submission.parent.generation != 0 &&
                submission.parent.kind == KC_TICKET_KIND_WORKFLOW &&
                submission.epoch != 0 && slot->engine == e &&
                slot->context_id == submission.conversation_id &&
                slot->submission.slot == slot->index &&
                slot->submission.lease_generation ==
                    submission.lease_generation &&
                slot->submission.operation == submission.operation &&
                ticket_equal(slot->submission.ticket, submission.ticket) &&
                ticket_equal(slot->submission.parent, submission.parent);
    }
    if (valid) {
        switch (slot->request) {
        case REQ_CONV_LAYER:
        case REQ_ATTN_LAYER:
        case REQ_TOKEN_PASS:
        case REQ_PREFILL:
        case REQ_AUDIO_DETOKENIZE:
            valid = slot->model &&
                    submission.conversation_id == slot->model->id &&
                    submission.epoch == slot->model->id;
            break;
        case REQ_AUDIO_ENCODE:
            valid = slot->model
                ? submission.conversation_id == slot->model->id &&
                      submission.epoch == slot->model->id
                : submission.conversation_id == 0 && submission.epoch == 1;
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
    if (valid &&
        !transition_slot(slot, owner, PASS_SLOT_SUBMITTED,
                         PASS_SLOT_RUNNING)) {
        valid = false;
    }
    if (!valid) return -ESTALE;

    activate_slot(e, slot);
    e->active_status.store(0, std::memory_order_relaxed);
    e->cur_req = slot->request;
    e->active_submission = submission;
    if (slot->request == REQ_CONV_LAYER) {
        initialize_conv_program(e, slot);
    } else if (slot->request == REQ_ATTN_LAYER) {
        initialize_attn_program(e, slot);
    } else if (slot->request == REQ_TOKEN_PASS) {
        initialize_token_program(e, slot);
    } else if (slot->request == REQ_PREFILL) {
        initialize_prefill_program(e, slot);
    } else if (slot->request == REQ_DEPTH_FRAME) {
        initialize_depth_program(slot);
    } else if (slot->request == REQ_AUDIO_DETOKENIZE) {
        initialize_detokenizer_program(e, slot);
    } else if (slot->request == REQ_AUDIO_ENCODE) {
        if (!e->hard_timeout_probe ||
            slot->audio.phase != AUDIO_PHASE_DONE) {
            slot->audio.phase = AUDIO_PHASE_FRONTEND;
        }
    }
    publish_engine_diagnostics(e);
    return 0;
}

static void executor_run_member(void *context, uint32_t lane,
                                uint32_t members,
                                uint64_t) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (!e || members != e->lanes_total || lane >= members)
        std::abort();
    lane_program(e, lane);
}

static int executor_before_generation(
    void *context, const FlashkernRequest &submission,
    uint64_t generation) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (!e || !ticket_equal(submission.ticket,
                            e->active_submission.ticket)) {
        std::abort();
    }
    const int status = begin_team_supervision(e, generation);
    if (status != 0) return status;
    publish_engine_diagnostics(e);
    return 0;
}

static void executor_generation_returned(
    void *context, const FlashkernRequest &submission,
    uint64_t generation) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (!e || !ticket_equal(submission.ticket,
                            e->active_submission.ticket) ||
        e->supervisor.snapshot().generation != generation) {
        std::abort();
    }
    e->diagnostics.team_completion_edge.store(
        generation, std::memory_order_release);
}

static kc::TeamReturn executor_accept_generation(
    void *context, const FlashkernRequest &submission,
    uint64_t generation) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (!e || e->supervisor.snapshot().generation != generation ||
        !ticket_equal(submission.ticket,
                      e->active_submission.ticket)) {
        std::abort();
    }
    e->diagnostics.team_completion_consumed.store(
        generation, std::memory_order_release);
    kc::TeamCompletion completion{};
    const int status =
        e->supervisor.complete(generation, &completion);
    if (status != 0) std::abort();
    if (completion == kc::TeamCompletion::Continue)
        return kc::TeamReturn::Continue;
    if (completion == kc::TeamCompletion::ExpiryWon)
        return kc::TeamReturn::Halt;
    std::abort();
}

static kc::TeamAdvance executor_advance(
    void *context, const FlashkernRequest &submission,
    uint64_t) noexcept {
    Engine *e = static_cast<Engine *>(context);
    PassSlot *slot = e ? e->active_slot : nullptr;
    if (!slot || !ticket_equal(submission.ticket,
                               e->active_submission.ticket)) {
        std::abort();
    }
    bool next = false;
    if (slot->request == REQ_CONV_LAYER)
        next = advance_conv_program(slot);
    else if (slot->request == REQ_ATTN_LAYER)
        next = advance_attn_program(slot);
    else if (slot->request == REQ_TOKEN_PASS)
        next = advance_token_program(e, slot);
    else if (slot->request == REQ_PREFILL)
        next = advance_prefill_program(e, slot);
    else if (slot->request == REQ_DEPTH_FRAME)
        next = advance_depth_program(e, slot);
    else if (slot->request == REQ_AUDIO_DETOKENIZE)
        next = advance_detokenizer_program(e, slot);
    else if (slot->request == REQ_AUDIO_ENCODE)
        next = advance_audio_program(e, slot);
    if (next) {
        return {
            .disposition = kc::TeamDisposition::Next,
            .status = 0,
        };
    }
    return {
        .disposition = kc::TeamDisposition::Complete,
        .status = e->active_status.load(std::memory_order_acquire),
    };
}

static void executor_make_completion(
    void *context, const FlashkernRequest &submission, int status,
    FlashkernCompletion *completion) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (!e || !completion) std::abort();
    *completion = {
        .ticket = submission.ticket,
        .parent = submission.parent,
        .conversation_id = submission.conversation_id,
        .epoch = submission.epoch,
        .lease_generation = submission.lease_generation,
        .slot = submission.slot,
        .status = status,
    };
    PassSlot *slot = e->active_slot;
    if (!slot || !ticket_equal(submission.ticket,
                               e->active_submission.ticket)) {
        return;
    }
    if (status == 0 && slot->request == REQ_AUDIO_ENCODE) {
        completion->result_kind = PASS_RESULT_FRAME;
        completion->result_count = 2;
        completion->results[0] = static_cast<uint32_t>(
            slot->audio.adapted_values & UINT64_C(0xffffffff));
        completion->results[1] = static_cast<uint32_t>(
            slot->audio.adapted_values >> 32);
    } else if (status == 0 && slot->request == REQ_PREFILL &&
               slot->prefill.sample) {
        completion->result_kind = PASS_RESULT_TEXT_TOKEN;
        completion->result_count = 1;
        completion->results[0] = *slot->prefill.out_token;
    }
    e->pass_completions.fetch_add(1, std::memory_order_relaxed);
}

static void executor_finish(
    void *context, const FlashkernRequest &submission,
    const FlashkernCompletion &completion) noexcept {
    Engine *e = static_cast<Engine *>(context);
    const PassSlotPool::Lease lease = {
        .index = submission.slot,
        .generation = submission.lease_generation,
    };
    PassSlot *slot = e ? e->slots.get(lease) : nullptr;
    const uint64_t owner = submission.lease_generation;
    const uint32_t state = slot ? slot_state(slot) : PASS_SLOT_FREE;
    if (state == PASS_SLOT_RUNNING) {
        if (e->active_slot != slot) std::abort();
        deactivate_slot(e, slot);
        e->cur_req = REQ_NONE;
        e->active_submission = {};
    }
    if (!slot || !ticket_equal(completion.ticket, submission.ticket) ||
        !ticket_equal(completion.parent, submission.parent) ||
        completion.conversation_id != submission.conversation_id ||
        completion.epoch != submission.epoch ||
        completion.slot != submission.slot ||
        completion.lease_generation != submission.lease_generation ||
        !ticket_equal(slot->submission.ticket, submission.ticket) ||
        !ticket_equal(slot->submission.parent, submission.parent) ||
        (state != PASS_SLOT_RUNNING &&
         state != PASS_SLOT_SUBMITTED)) {
        std::abort();
    }

    slot->completion = completion;
    if (!slot->continuation ||
        !transition_slot(slot, owner, state,
                         PASS_SLOT_COMPLETING)) {
        std::abort();
    }
    PassContinuation continuation = slot->continuation;
    void *continuation_context = slot->continuation_context;
    clear_slot_request(slot);
    if (!transition_slot(slot, owner, PASS_SLOT_COMPLETING,
                         PASS_SLOT_RESERVED)) {
        std::abort();
    }
    PassContinuationPermit permit = {
        .engine = e,
        .slot = slot,
        .generation = owner,
        .consumed = false,
    };
    try {
        continuation(&permit, completion, continuation_context);
    } catch (...) {
        if (!permit.consumed &&
            !release_pass_slot(slot, owner)) {
            std::abort();
        }
        std::abort();
    }
    if (!permit.consumed && !release_pass_slot(slot, owner))
        std::abort();
    publish_engine_diagnostics(e);
}

static void executor_stopping(void *context) noexcept {
    Engine *e = static_cast<Engine *>(context);
    if (e) e->supervisor.request_stop();
}

static void executor_retired(void *context,
                             const kc_ticket_id *identity) {
    Engine *e = static_cast<Engine *>(context);
    if (!e || !identity ||
        !ticket_equal(*identity, e->executor.identity())) {
        std::abort();
    }
    publish_engine_diagnostics(e);
}

static int submit_slot(Engine *e, PassSlot *slot, uint64_t generation,
                       int request, uint64_t context_id,
                       const kc_ticket_id &parent,
                       PassContinuation continuation,
                       void *continuation_context) {
    if (!e || !slot || slot->engine != e ||
        !request_kind_valid(static_cast<uint32_t>(request)) || generation == 0 ||
        !continuation || parent.runtime_epoch != e->tickets.epoch() ||
        parent.sequence == 0 || parent.generation == 0 ||
        parent.kind != KC_TICKET_KIND_WORKFLOW ||
        slot_state(slot) != PASS_SLOT_RESERVED ||
        slot_generation(slot) != generation) {
        return -EINVAL;
    }
    if (!transition_slot(slot, generation, PASS_SLOT_RESERVED,
                         PASS_SLOT_SUBMITTING)) {
        return -ESTALE;
    }

    const kc_ticket_id ticket = e->tickets.mint(KC_TICKET_KIND_PASS);

    FlashkernRequest submission{};
    submission.ticket = ticket;
    submission.parent = parent;
    submission.conversation_id = context_id;
    submission.epoch = context_id == 0 ? 1 : context_id;
    submission.lease_generation = generation;
    submission.slot = slot->index;
    submission.operation = static_cast<uint32_t>(request);

    slot->request = request;
    slot->context_id = context_id;
    slot->continuation = continuation;
    slot->continuation_context = continuation_context;

    // The route continuation publishes one pointer-only request through
    // TeamExecutor's fixed producer endpoint. Its bounded admission lease
    // serializes concurrent route publishers without a mutex or retry loop.
    slot->submission = submission;
    // This is the publication edge for every typed request field and its exact
    // ticket locator. The SQ consumer acquires this state before reading them.
    if (!transition_slot(slot, generation, PASS_SLOT_SUBMITTING,
                         PASS_SLOT_SUBMITTED)) {
        std::abort();
    }
    const int rc = e->executor.submit(submission);
    if (rc != 0) {
        if (!transition_slot(slot, generation, PASS_SLOT_SUBMITTED,
                             PASS_SLOT_RESERVED)) {
            std::abort();
        }
    }
    if (rc != 0) {
        return rc;
    }
    e->pass_submissions.fetch_add(1, std::memory_order_relaxed);
    e->continuation_submissions.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

static bool release_continuation(PassContinuationPermit *permit) {
    if (!permit || permit->consumed || !permit->slot) return false;
    if (!release_pass_slot(permit->slot, permit->generation)) return false;
    permit->consumed = true;
    return true;
}

static constexpr std::array<std::array<uint8_t, AUDIO_ROUTE_OUTCOME_COUNT>,
                            AUDIO_ROUTE_NODE_COUNT>
    AUDIO_ROUTE_TABLE = {{
        {{AUDIO_ROUTE_DEPTH, AUDIO_ROUTE_TERMINAL, AUDIO_ROUTE_TERMINAL,
          AUDIO_ROUTE_TERMINAL}},
        {{AUDIO_ROUTE_DETOKENIZER, AUDIO_ROUTE_TERMINAL,
          AUDIO_ROUTE_DETOKENIZER, AUDIO_ROUTE_TERMINAL}},
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
    if (token < LFM_DETOKENIZER_CODE_VALUES) return AUDIO_TOKEN_VALUE;
    if (token == LFM_DETOKENIZER_CODE_VALUES) return AUDIO_TOKEN_END;
    return AUDIO_TOKEN_INVALID;
}

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

static kc::PermitAdvance complete_audio_route(int status) {
    return {
        .disposition = kc::PermitDisposition::Complete,
        .status = status,
    };
}

static kc::PermitAdvance continue_audio_route(
    AudioRouteInstance *route,
    const FlashkernCompletion &completion) noexcept {
    if (!route) std::abort();
    if (route->kind == AUDIO_ROUTE_ENCODE) {
        if (completion.status == 0) {
            if (!route->adapted_values ||
                completion.result_kind != PASS_RESULT_FRAME ||
                completion.result_count != 2) {
                return complete_audio_route(-EPROTO);
            }
            *route->adapted_values =
                static_cast<uint64_t>(completion.results[0]) |
                (static_cast<uint64_t>(completion.results[1]) << 32);
        }
        return complete_audio_route(completion.status);
    }
    if (route->kind == AUDIO_ROUTE_PREFILL) {
        return complete_audio_route(completion.status);
    }
    if (route->kind != AUDIO_ROUTE_GENERATION) {
        return complete_audio_route(-EPROTO);
    }
    uint32_t outcome = completion.status == 0 ? AUDIO_ROUTE_SUCCESS
                                              : AUDIO_ROUTE_FAILURE;
    if (completion.status == 0 && route->node == AUDIO_ROUTE_TOKEN) {
        const int commit_status = commit_audio_route_token(route);
        if (commit_status != 0)
            return complete_audio_route(commit_status);
        if (route->terminal_after_token) {
            return complete_audio_route(0);
        }
        if (route->decode_detokenizer &&
            route->epoch->load(std::memory_order_acquire) !=
                route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        }
    } else if (completion.status == 0 && route->node == AUDIO_ROUTE_DEPTH &&
               route->decode_detokenizer) {
        route->result->depth_completed = 1;
        const uint32_t first_class =
            audio_token_class(route->result->codes[0]);
        if (first_class == AUDIO_TOKEN_END) {
            route->result->eoaudio = 1;
            route->detokenizer_req.flush = true;
            outcome = AUDIO_ROUTE_EOAUDIO;
        } else if (first_class == AUDIO_TOKEN_INVALID) {
            return complete_audio_route(-ERANGE);
        } else if (route->epoch->load(std::memory_order_acquire) !=
                   route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        } else {
            for (size_t index = 0; index < LFM_DETOKENIZER_CODEBOOKS; ++index) {
                if (audio_token_class(route->result->codes[index]) !=
                    AUDIO_TOKEN_VALUE) {
                    return complete_audio_route(-ERANGE);
                }
            }
        }
    } else if (completion.status == 0 &&
               route->node == AUDIO_ROUTE_DETOKENIZER &&
               route->decode_detokenizer) {
        route->result->detokenizer_completed = 1;
        if (route->epoch->load(std::memory_order_acquire) !=
            route->expected_epoch) {
            outcome = AUDIO_ROUTE_STALE;
        }
    }
    if (completion.status == 0 && route->engine->routes.stopped())
        return complete_audio_route(-ECANCELED);
    uint32_t target = AUDIO_ROUTE_TERMINAL;
    if (!audio_route_next(route->node, outcome, &target))
        return complete_audio_route(-EPROTO);
    if (completion.status != 0)
        return complete_audio_route(completion.status);
    if (outcome == AUDIO_ROUTE_STALE)
        return complete_audio_route(-ESTALE);
    /* A codes-only route terminates after Depth. A playback route sends both a
     * normal code frame and EOAudio through the detokenizer: EOAudio flushes
     * the final same-padding overlap instead of manufacturing another code. */
    if (!route->decode_detokenizer && route->node == AUDIO_ROUTE_DEPTH)
        return complete_audio_route(0);
    if (target == AUDIO_ROUTE_TERMINAL)
        return complete_audio_route(0);
    if (route->node == AUDIO_ROUTE_TOKEN && target == AUDIO_ROUTE_DEPTH &&
        route->depth && route->depth_id != 0) {
        route->node = AUDIO_ROUTE_DEPTH;
    } else if (route->node == AUDIO_ROUTE_DEPTH &&
               target == AUDIO_ROUTE_DETOKENIZER &&
               route->decode_detokenizer && route->model &&
               route->model_id != 0) {
        route->node = AUDIO_ROUTE_DETOKENIZER;
    } else {
        return complete_audio_route(-EPROTO);
    }
    return {
        .disposition = kc::PermitDisposition::Requeue,
        .status = 0,
    };
}

class AudioRouteLease {
  public:
    explicit AudioRouteLease(
        Engine *engine,
        kc::ServiceClass service =
            kc::ServiceClass::Interactive) {
        if (!engine) {
            status_ = -EINVAL;
            return;
        }
        if (!enter_pass_admission(engine)) {
            status_ = -EBUSY;
            return;
        }
        engine_ = engine;
        pass_admitted_ = true;
        status_ = engine->routes.claim(service, &claim_);
        if (status_ != 0) {
            claim_.abandon();
            leave_pass_admission(engine_);
            pass_admitted_ = false;
            return;
        }
        lease_ = claim_.lease();
        route_ = claim_.record();
        if (!route_) std::abort();
        route_->lease_generation = lease_.generation;
        route_->ticket = lease_.ticket;
    }

    ~AudioRouteLease() {
        if (!pass_admitted_) return;
        claim_.abandon();
        leave_pass_admission(engine_);
    }

    explicit operator bool() const { return route_ != nullptr; }
    int status() const { return status_; }
    AudioRouteInstance *route() const { return route_; }
    uint64_t generation() const { return lease_.generation; }
    const kc_ticket_id &ticket() const { return lease_.ticket; }
    int publish() {
        const int status = claim_.publish();
        if (status == 0) {
            route_ = nullptr;
            pass_admitted_ = false;
        }
        return status;
    }
    AudioRouteLease(const AudioRouteLease &) = delete;
    AudioRouteLease &operator=(const AudioRouteLease &) = delete;

  private:
    AudioRouteBroker::Claim claim_;
    AudioRouteBroker::Lease lease_{};
    Engine *engine_ = nullptr;
    AudioRouteInstance *route_ = nullptr;
    bool pass_admitted_ = false;
    int status_ = -EINVAL;
};

static int mount_audio_route(PassSlot *slot, AudioRouteInstance *route,
                             int *request, uint64_t *context) {
    if (!slot || !route || !request || !context) return -EINVAL;
    if (route->kind == AUDIO_ROUTE_ENCODE) {
        slot->model = route->model;
        slot->audio = route->audio_req;
        *request = REQ_AUDIO_ENCODE;
        *context = route->model_id;
        return 0;
    }
    if (route->kind == AUDIO_ROUTE_PREFILL) {
        slot->model = route->model;
        slot->prefill = route->prefill_req;
        if (route->prefill_req.readout) {
            /* Test-only dual-head readout: the codebook-0 evaluation reuses
             * the depth program on this slot after the text sample. */
            slot->depth = route->depth;
            slot->depth_req = route->depth_req;
        }
        *request = REQ_PREFILL;
        *context = route->model_id;
        return 0;
    }
    if (route->kind != AUDIO_ROUTE_GENERATION) return -EPROTO;
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
    if (route->node == AUDIO_ROUTE_DETOKENIZER &&
        route->decode_detokenizer) {
        slot->model = route->model;
        slot->detokenizer = route->detokenizer_req;
        *request = REQ_AUDIO_DETOKENIZE;
        *context = route->model_id;
        return 0;
    }
    return -EPROTO;
}

static void reset_audio_route(void *context, AudioRouteInstance &route,
                              uint32_t index) noexcept {
    Engine *engine = static_cast<Engine *>(context);
    route = {};
    route.engine = engine;
    route.lease_index = index;
}

static void publish_audio_route_completion(
    PassContinuationPermit *permit,
    const FlashkernCompletion &completion,
    void *context) noexcept {
    AudioRouteInstance *route =
        static_cast<AudioRouteInstance *>(context);
    if (!permit || !route || !route->engine ||
        route->pass_completion.ticket.sequence != 0) {
        std::abort();
    }
    route->pass_completion = completion;
    if (!release_continuation(permit)) std::abort();
    const AudioRouteBroker::Lease lease = {
        .index = route->lease_index,
        .generation = route->lease_generation,
        .ticket = route->ticket,
    };
    if (route->engine->routes.resume(lease) != 0) {
        std::abort();
    }
}

static kc::PermitAdvance step_audio_route(
    void *context, AudioRouteBroker::Lease,
    AudioRouteInstance &route, kc::PermitEvent event) noexcept {
    Engine *engine = static_cast<Engine *>(context);
    engine->diagnostics.route_callbacks.fetch_add(
        1, std::memory_order_relaxed);
    if (event == kc::PermitEvent::Cancel)
        return complete_audio_route(-ECANCELED);
    if (event == kc::PermitEvent::Completion) {
        if (route.pass_completion.ticket.sequence == 0)
            std::abort();
        const FlashkernCompletion completion =
            route.pass_completion;
        route.pass_completion = {};
        return continue_audio_route(&route, completion);
    }
    if (event != kc::PermitEvent::Grant) std::abort();
    if (route.kind == AUDIO_ROUTE_CONTROL)
        return complete_audio_route(0);

    PassSlot *slot = reserve_pass_slot(engine);
    if (!slot) {
        return {
            .disposition = kc::PermitDisposition::Preserve,
            .status = 0,
        };
    }

    int request = REQ_NONE;
    uint64_t request_context = 0;
    int status = mount_audio_route(
        slot, &route, &request, &request_context);
    if (status != 0) {
        if (!release_pass_slot(slot, slot_generation(slot)))
            std::abort();
        return complete_audio_route(status);
    }
    const uint64_t slot_owner = slot_generation(slot);
    status = submit_slot(
        engine, slot, slot_owner, request, request_context,
        route.ticket, publish_audio_route_completion, &route);
    if (status != 0) {
        if (!release_pass_slot(slot, slot_owner)) std::abort();
        if (status == -EBUSY || status == -EAGAIN) {
            return {
                .disposition = kc::PermitDisposition::Preserve,
                .status = 0,
            };
        }
        return complete_audio_route(status);
    }
    engine->route_dispatches.fetch_add(
        1, std::memory_order_relaxed);
    return {
        .disposition = kc::PermitDisposition::Suspend,
        .status = 0,
    };
}

static void finish_audio_route(
    void *, AudioRouteBroker::Lease,
    AudioRouteInstance &route, int status) noexcept {
    route.status = status;
    if (route.result) route.result->status = status;
    if (!route.notify) std::abort();
    route.notify(route.notify_context);
}

} // namespace

// ---- the C ABI ------------------------------------------------------------------------
extern "C" {

void lfm_engine_free(void *ep);

// `workers` is the complete bounded kcoro worker pool. Numerical lane members,
// the bridge, the route broker, and the supervisor are logical continuations
// on that one pool; engine construction creates no second thread society.
static void *engine_new_impl(int workers, bool manual_deadlines,
                             const TeamFaultTestConfig *fault,
                             int *out_status) {
    if (out_status) *out_status = -ENOMEM;
    if (workers < 1 || workers > MAX_WORKERS) {
        if (out_status) *out_status = -EINVAL;
        return nullptr;
    }
    if (!lfm_bf16_gemm_available()) {
        if (out_status) *out_status = -ENOTSUP;
        return nullptr;
    }
    Engine *e = new (std::nothrow) Engine();
    if (!e) return nullptr;
    e->manual_deadlines = manual_deadlines;
    if (fault) e->hard_timeout_probe = true;
    e->lanes_total = (uint32_t)workers;
    e->n_workers = workers;
    e->block_count = workers == (int)GRID_LANES ? 2u : 1u;
    for (size_t index = 0; index < e->slots.size(); ++index) {
        PassSlot &slot = e->slots.record(index);
        slot.engine = e;
        slot.index = (uint32_t)index;
    }
    const kc_runtime_config runtime_config = {
        .worker_count = static_cast<uint32_t>(workers),
    };
    const size_t continuation_capacity =
        ROUTE_CAPACITY + static_cast<size_t>(workers) + 3;
    if (kc_runtime_create_capacity(
            runtime_config.worker_count, continuation_capacity,
            &e->control_runtime) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->mailbox_capacity_identity =
        e->tickets.mint(KC_TICKET_KIND_CONTROL);
    e->route_capacity_identity =
        e->tickets.mint(KC_TICKET_KIND_CONTROL);
    const AudioRouteBroker::Config route_config = {
        .runtime = e->control_runtime,
        .tickets = &e->tickets,
        .ticket_kind = KC_TICKET_KIND_WORKFLOW,
        .age_promotion = ROUTE_AGE_PROMOTION,
        .context = e,
        .operations = {
            .reset = reset_audio_route,
            .step = step_audio_route,
            .finished = finish_audio_route,
        },
        .capacity_ready = {
            publish_route_capacity_edge, e,
            e->route_capacity_identity},
    };
    if (e->routes.initialize(route_config) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    const EngineExecutor::Config executor_config = {
        .runtime = e->control_runtime,
        .member_count = static_cast<uint32_t>(workers),
        .context = e,
        .operations = {
            .begin = executor_begin,
            .run_member = executor_run_member,
            .before_generation = executor_before_generation,
            .generation_returned = executor_generation_returned,
            .accept_generation = executor_accept_generation,
            .advance = executor_advance,
            .make_completion = executor_make_completion,
            .finish = executor_finish,
            .stopping = executor_stopping,
        },
        .capacity_ready = {
            publish_mailbox_capacity_edge, e,
            e->mailbox_capacity_identity},
        .retired = executor_retired,
        .retired_context = e,
    };
    if (e->executor.initialize(executor_config) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    const EngineSupervisor::Config supervisor_config = {
        .runtime = e->control_runtime,
        .team = e->executor.team(),
        .tickets = &e->tickets,
        .ticket_kind = KC_TICKET_KIND_DEADLINE,
        .deadline_slot = TEAM_DEADLINE_SLOT,
        .expected_mask = expected_team_mask(e),
        .fatal_magic = TEAM_FATAL_SINK_MAGIC,
        .fatal_path = fault ? fault->fatal_path : nullptr,
        .manual_deadlines = manual_deadlines,
        .context = e,
        .operations = {
            .fill_fatal = fill_team_fatal_capsule,
            .fatal_published = nullptr,
        },
    };
    const int supervisor_status =
        e->supervisor.initialize(supervisor_config);
    if (supervisor_status != 0) {
        if (out_status) *out_status = supervisor_status;
        lfm_engine_free(e);
        return nullptr;
    }
    if (fault && kc_team_inject_member_exit_for_test(
                     e->executor.team(), 1, fault->member, fault->point,
                     hard_timeout_probe_ready, e) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->pass_admission.bind(notify_route_broker, e);
    if (kc_runtime_start(e->control_runtime) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->control_started = 1;
    if (e->supervisor.start() != 0 ||
        e->executor.start() != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->executor_started = 1;
    if (e->routes.start() != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    if (out_status) *out_status = 0;
    return e;
}

void *lfm_engine_new_status(int workers, int *out_status) {
    return engine_new_impl(workers, false, nullptr, out_status);
}

void *lfm_internal_engine_new_manual_deadlines_for_test(int workers) {
    return engine_new_impl(workers, true, nullptr, nullptr);
}

void *lfm_internal_engine_new_hard_timeout_probe_for_test(
    int workers, uint32_t member, uint32_t point, const char *fatal_path) {
    if (!fatal_path || fatal_path[0] == '\0') return nullptr;
    const TeamFaultTestConfig fault = {
        .member = member,
        .point = point,
        .fatal_path = fatal_path,
    };
    return engine_new_impl(workers, true, &fault, nullptr);
}

static void hard_timeout_probe_completion(
    PassContinuationPermit *, const FlashkernCompletion &, void *) {
    /* The hard deadline must abort before ordinary numerical publication. */
    std::abort();
}

int lfm_internal_engine_trigger_hard_timeout_probe_for_test(void *ep) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !e->hard_timeout_probe || !e->manual_deadlines ||
        e->active_slot || e->cur_req != REQ_NONE ||
        e->supervisor.snapshot().terminal != 0) {
        return -EINVAL;
    }
    PassSlot *slot = reserve_pass_slot(e);
    if (!slot) return -EAGAIN;
    const uint64_t generation = slot_generation(slot);
    /* AUDIO_ENCODE/DONE is a valid closed-protocol request whose peers only
     * publish -EPROTO; it touches no model, PCM, or scratch view. The selected
     * member alone owns the injected liveness fault. */
    slot->audio.phase = AUDIO_PHASE_DONE;
    slot->stage.kind = ST_IDLE;
    const kc_ticket_id parent =
        e->tickets.mint(KC_TICKET_KIND_WORKFLOW);
    const int status = submit_slot(
        e, slot, generation, REQ_AUDIO_ENCODE, 0, parent,
        hard_timeout_probe_completion, nullptr);
    if (status == 0) return 0;
    if (!release_pass_slot(slot, generation)) std::abort();
    return status;
}

void lfm_engine_request_stop(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    e->routes.request_stop();
    e->executor.request_stop();
    if (!e->executor_started) e->supervisor.request_stop();
}

// Private implementation-backed protocol probes. Selector membership is
// queried without dispatch: a valid request also requires a fully populated
// typed payload, so submitting an empty test slot would be an unsafe probe.
int lfm_internal_engine_request_kind_valid_for_test(uint32_t kind) {
    return request_kind_valid(kind) ? 1 : 0;
}

uint64_t lfm_internal_engine_hard_budget_for_test(uint32_t probe,
                                                  uint32_t request) {
    const TeamWorkDescriptor work = {
        .request = request,
    };
    return team_hard_budget_ns(probe != 0, work);
}

int lfm_internal_engine_team_terminal_race_for_test(
    uint64_t generation, uint32_t first, uint32_t second,
    uint32_t *winner_bits, uint32_t *terminal_state) {
    const uint32_t completed =
        static_cast<uint32_t>(kc::TeamTerminalState::Completed);
    const uint32_t timed_out =
        static_cast<uint32_t>(kc::TeamTerminalState::TimedOut);
    if (generation == 0 ||
        generation > kc::TeamTerminal::max_generation ||
        !winner_bits || !terminal_state ||
        (first != completed && first != timed_out) ||
        (second != completed && second != timed_out) ||
        first == second) {
        return -EINVAL;
    }
    kc::TeamTerminal terminal;
    if (terminal.begin(generation) != 0) return -EFAULT;
    uint32_t winners = 0;
    if (terminal.claim(
            generation,
            static_cast<kc::TeamTerminalState>(first))) {
        winners |= 1;
    }
    if (terminal.claim(
            generation,
            static_cast<kc::TeamTerminalState>(second))) {
        winners |= 2;
    }
    const uint64_t result = terminal.word();
    if (kc::TeamTerminal::generation_of(result) != generation)
        return -EFAULT;
    *winner_bits = winners;
    *terminal_state = static_cast<uint32_t>(
        kc::TeamTerminal::state_of(result));
    return 0;
}

int lfm_internal_engine_completion_decision_for_test(
    uint64_t generation, int retire_result, int *decision,
    uint32_t *terminal_state) {
    if (generation == 0 ||
        generation > kc::TeamTerminal::max_generation || !decision ||
        !terminal_state) {
        return -EINVAL;
    }
    kc::TeamTerminal terminal;
    if (terminal.begin(generation) != 0) return -EFAULT;
    kc::TeamCompletion completion{};
    const int status =
        terminal.settle(generation, retire_result, &completion);
    if (status != 0) return status;
    *decision = static_cast<int>(completion);
    const uint64_t observed = terminal.word();
    if (kc::TeamTerminal::generation_of(observed) != generation)
        return -EFAULT;
    *terminal_state = static_cast<uint32_t>(
        kc::TeamTerminal::state_of(observed));
    return 0;
}

int lfm_internal_engine_fatal_capsule_for_test(
    uint64_t generation, uint64_t expected_mask, uint64_t entered_mask,
    uint64_t returned_mask, TeamFatalCapsule *out) {
    if (!out || out->size < sizeof(*out) || generation == 0 ||
        (entered_mask & ~expected_mask) != 0 ||
        (returned_mask & ~entered_mask) != 0) {
        return -EINVAL;
    }
    const FlashkernRequest submission = {
        .ticket = {
            .runtime_epoch = 11,
            .sequence = 22,
            .generation = 32,
            .kind = KC_TICKET_KIND_PASS,
        },
        .parent = {
            .runtime_epoch = 11,
            .sequence = 21,
            .generation = 31,
            .kind = KC_TICKET_KIND_WORKFLOW,
        },
        .conversation_id = 44,
        .epoch = 55,
    };
    const kc_deadline_event event = {
        .slot = TEAM_DEADLINE_SLOT,
        .kind = KC_DEADLINE_EVENT_EXPIRED,
        .sequence = 77,
        .scheduled_arm_generation = 88,
        .current_arm_generation = 88,
        .child = {
            .runtime_epoch = 11,
            .sequence = 23,
            .generation = 33,
            .kind = KC_TICKET_KIND_DEADLINE,
        },
        .parent = submission.ticket,
        .scope_generation = 66,
        .epoch = submission.epoch,
        .domain = 99,
        .team_generation = generation,
    };
    const kc_team_quorum_snapshot quorum = {
        .generation = generation,
        .expected_mask = expected_mask,
        .entered_mask = entered_mask,
        .returned_mask = returned_mask,
    };
    const TeamWorkDescriptor work = {
        .request = REQ_TOKEN_PASS,
        .stage = ST_NORM,
        .program_kind = TOKEN_PROGRAM_CONV,
        .program_phase = CONV_PHASE_NORM,
        .program_flags = 0x5a,
        .program_outer = 12,
        .program_inner = 13,
        .shape0 = 2048,
        .shape1 = 8192,
        .shape2 = 7,
    };
    constexpr uint64_t armed_ns = 1000;
    constexpr uint64_t elapsed_ns = TEAM_PROBE_HARD_DEADLINE_NS + 123;
    fill_team_fatal_capsule(
        nullptr, *out, submission, work, event, armed_ns,
        TEAM_PROBE_HARD_DEADLINE_NS, elapsed_ns, 0, quorum);
    return 0;
}

int lfm_internal_engine_grid_snapshot_for_test(
    void *ep, uint32_t *blocks, uint64_t *completions,
    uint64_t *generations, uint64_t *lease) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !blocks || !completions || !generations || !lease) {
        return -EINVAL;
    }
    const kc::TeamExecutorSnapshot executor = e->executor.snapshot();
    *blocks = e->block_count;
    *completions =
        executor.generations_returned * e->block_count;
    *generations = executor.generations_dispatched;
    *lease = executor.generation;
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
    if (service < static_cast<uint32_t>(kc::ServiceClass::Deadline) ||
        service > static_cast<uint32_t>(kc::ServiceClass::Background)) {
        return 0;
    }
    return static_cast<uint32_t>(AudioRouteBroker::classify(
        snapshot, enqueued, static_cast<kc::ServiceClass>(service),
        ROUTE_AGE_PROMOTION));
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

int lfm_internal_engine_fail_audio_route_detokenizer_for_test(void *ep,
                                                              int status) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || status >= 0) return -EINVAL;
    int idle = 0;
    return e->test_audio_route_detokenizer_status.compare_exchange_strong(
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

int lfm_engine_audio_encode_submit(
    void *ep, uint64_t model_id, const LfmAudioEncodePassV2 *pass,
    uint64_t *out_adapted_values, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    Engine *e = static_cast<Engine *>(ep);
    if (!e || !pass || pass->size < sizeof(*pass) ||
        pass->abi_version != LFM_AUDIO_PASS_ABI || !pass->resampler ||
        !pass->resampler_workspace || !pass->frontend ||
        !pass->frontend_workspace || !pass->conformer ||
        !pass->conformer_workspace || pass->pcm.count == 0 ||
        pass->pcm.length == 0 ||
        (!pass->resampled && pass->resampled_capacity != 0) || !pass->mel ||
        pass->mel_capacity == 0 || !pass->adapted ||
        pass->adapted_capacity == 0 || !out_adapted_values || !notify ||
        !out_handle) {
        return -EINVAL;
    }
    LfmF32SpanChain checked{};
    if (lfm_f32_span_chain_init(pass->pcm.spans, pass->pcm.count, &checked) !=
            0 ||
        std::memcmp(&checked, &pass->pcm, sizeof(checked)) != 0) {
        return -EINVAL;
    }
    *out_handle = {};
    *out_adapted_values = 0;
    AudioRouteLease claim(e);
    if (!claim) return claim.status();
    BackbonePlan *model = model_id == 0 ? nullptr : find_model(e, model_id);
    if (model_id != 0 && !model) return -ESTALE;
    if (model && lfm_conformer_out_width(pass->conformer) != model->h) {
        return -ESTALE;
    }
    AudioRouteInstance *route = claim.route();
    route->engine = e;
    route->kind = AUDIO_ROUTE_ENCODE;
    route->node = AUDIO_ROUTE_TERMINAL;
    route->model = model;
    route->model_id = model_id;
    route->audio_req = {};
    route->audio_req.pass = *pass;
    route->audio_req.phase = AUDIO_PHASE_FRONTEND;
    route->adapted_values = out_adapted_values;
    route->result = nullptr;
    route->notify = notify;
    route->notify_context = notify_context;
    route->terminal_after_token = false;
    route->decode_detokenizer = false;
    route->status = -EINPROGRESS;
    out_handle->record = route;
    out_handle->generation = claim.generation();
    out_handle->ticket = route->ticket;
    const int published = claim.publish();
    if (published != 0) {
        *out_handle = {};
        return published;
    }
    return 0;
}

int lfm_engine_snapshot(void *ep, LfmEngineSnapshotV1 *out) {
    Engine *e = (Engine *)ep;
    if (!e || !out || out->size < sizeof(*out) || out->abi_version != 1) return -EINVAL;
    const kc::TeamExecutorSnapshot executor = e->executor.snapshot();
    const kc::MailboxSnapshot mailbox = executor.mailbox;
    const kc::PermitBrokerSnapshot routes = e->routes.snapshot();
    *out = {
        .size = sizeof(*out),
        .abi_version = 1,
        .pass_submissions = e->pass_submissions.load(std::memory_order_relaxed),
        .pass_completions = e->pass_completions.load(std::memory_order_relaxed),
        .bridge_dispatches = executor.requests_started,
        .dispatch_wakes = executor.generations_dispatched,
        .attention_qkv_capacity =
            e->attention_qkv_capacity.load(std::memory_order_relaxed),
        .attention_y_capacity =
            e->attention_y_capacity.load(std::memory_order_relaxed),
        .attention_score_capacity =
            e->attention_score_capacity.load(std::memory_order_relaxed),
        .pass_claimed = e->pass_claimed.load(std::memory_order_acquire) ? 1u : 0u,
        .bridge_capacity = mailbox.capacity,
        .pass_slot_capacity = (uint32_t)e->slots.size(),
        .pass_slots_live = e->pass_slots_live.load(std::memory_order_acquire),
        .max_pass_slots_live =
            e->max_pass_slots_live.load(std::memory_order_acquire),
        .continuation_submissions =
            e->continuation_submissions.load(std::memory_order_relaxed),
        .route_capacity = (uint32_t)ROUTE_CAPACITY,
        .routes_live = routes.live,
        .routes_ready = routes.ready,
        .reserved0 = 0,
        .route_dispatches =
            e->route_dispatches.load(std::memory_order_relaxed),
        .route_admission_deferrals = routes.deferrals,
    };
    return 0;
}

extern "C++" int lfm_internal_engine_diagnostic_view(
    void *ep, LfmEngineDiagnosticView *out) {
    Engine *engine = static_cast<Engine *>(ep);
    if (!engine || !out || !engine->control_runtime ||
        !engine->executor.continuation() || !engine->routes.service() ||
        !engine->supervisor.service() || !engine->executor.team()) {
        return -EINVAL;
    }
    *out = {
        .state = &engine->diagnostics,
        .runtime = engine->control_runtime,
        .bridge_continuation = engine->executor.continuation(),
        .route_service = engine->routes.service(),
        .supervisor_service = engine->supervisor.service(),
        .team = engine->executor.team(),
        .owner = engine,
    };
    return 0;
}

extern "C++" int lfm_internal_engine_diagnostic_counts(
    const LfmEngineDiagnosticView *view, LfmEngineDiagnosticCounts *out) {
    if (!view || !view->owner || !view->state || !out) return -EINVAL;
    Engine *engine = static_cast<Engine *>(view->owner);
    const kc::TeamExecutorSnapshot executor =
        engine->executor.snapshot();
    const kc::MailboxSnapshot mailbox = executor.mailbox;
    const kc::PermitBrokerSnapshot routes =
        engine->routes.snapshot();
    LfmEngineDiagnosticCounts counts = {
        .pass_submissions =
            engine->pass_submissions.load(std::memory_order_acquire),
        .pass_completions =
            engine->pass_completions.load(std::memory_order_acquire),
        .bridge_dispatches = executor.requests_started,
        .dispatch_wakes = executor.generations_dispatched,
        .route_dispatches =
            engine->route_dispatches.load(std::memory_order_acquire),
        .route_admission_deferrals = routes.deferrals,
        .bridge_team_generation = executor.generation,
        .bridge_team_completion = executor.returned_generation,
        .bridge_retired_generation = executor.retired_generation,
        .mailbox_requests_published = mailbox.requests_published,
        .mailbox_requests_consumed = mailbox.requests_consumed,
        .mailbox_completions_published = mailbox.completions_published,
        .mailbox_completions_consumed = mailbox.completions_consumed,
        .gang_lease = executor.generation,
        .team_terminal = engine->supervisor.snapshot().terminal,
        .pass_slots_live =
            engine->pass_slots_live.load(std::memory_order_acquire),
    };
    counts.routes_free = routes.free;
    counts.routes_claimed = routes.claimed;
    counts.routes_ready = routes.ready;
    counts.routes_dispatching = 0;
    counts.routes_running = routes.running;
    counts.routes_done = routes.done;
    *out = counts;
    return 0;
}

int lfm_internal_engine_stackless_runtime_for_test(
    void *ep, kc_runtime_snapshot *out_runtime, kc_ticket_id *out_bridge) {
    Engine *engine = static_cast<Engine *>(ep);
    if (!engine || !engine->control_runtime ||
        !engine->executor.continuation() ||
        !out_runtime || !out_bridge) {
        return -EINVAL;
    }
    const int status = kc_runtime_snapshot_get(engine->control_runtime,
                                               out_runtime);
    if (status != 0) return status;
    *out_bridge = engine->executor.identity();
    return 0;
}

void lfm_engine_free(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    lfm_engine_request_stop(e);
    /* Administrative teardown begins only after the retained frames have
     * acknowledged their authoritative terminal edges. No inference operation
     * owns this observation; the executor and team retire through callbacks. */
    if (e->control_runtime && e->control_started > 0 &&
        kc_runtime_join_all(e->control_runtime) != 0) {
        std::abort();
    }
    if (e->routes.join() != 0) std::abort();
    if (e->supervisor.join() != 0) std::abort();
    e->retire.store(true, std::memory_order_release);
    if (e->executor.destroy() != 0) std::abort();
    if (e->routes.destroy() != 0) std::abort();
    if (e->supervisor.destroy() != 0) std::abort();
    if (e->control_runtime) {
        kc_runtime_request_stop(e->control_runtime);
        if (kc_runtime_join(e->control_runtime) != 0 ||
            kc_runtime_destroy(e->control_runtime) != 0) {
            std::abort();
        }
        e->control_runtime = nullptr;
    }
    delete e;
}

static bool depth_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool depth_view(const LfmDepthBufferV1 &view, size_t count) {
    return view.address != 0 && view.count >= count;
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
    std::unique_ptr<DepthPlan> next(new (std::nothrow) DepthPlan());
    if (!next) return -ENOMEM;
    try {
        next->layers.assign(plan->layers, plan->layers + plan->layer_count);
        next->heads.assign(plan->codebook_heads, plan->codebook_heads + codebooks);
        const auto grow = [](auto &values, size_t count) {
            if (values.size() < count) values.resize(count);
        };
        for (size_t index = 0; index < e->slots.size(); ++index) {
            PassSlot &slot = e->slots.record(index);
            DepthScratch &scratch = slot.scratch.depth;
            grow(scratch.x, dim);
            grow(scratch.h, dim);
            grow(scratch.xn, dim);
            grow(scratch.qkv_b, qkv_rows);
            grow(scratch.q_f, (size_t)plan->heads * hd);
            grow(scratch.attn_f, dim);
            grow(scratch.attn_b, dim);
            grow(scratch.proj_f, projection_rows);
            grow(scratch.t_b, ffn);
            grow(scratch.k_plane, cache_count);
            grow(scratch.v_plane, cache_count);
            grow(scratch.logits_b, vocab_max);
            grow(scratch.din_b, projection_rows);
            grow(scratch.df_b, dim);
            grow(slot.scratch.sample_weights, vocab_max);
            grow(slot.scratch.sample_heap, vocab_max);
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

static int run_audio_route(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    uint32_t *out_codes, size_t code_count, size_t lanes,
    const LfmTokenCommitRecord *commit, uint32_t *out_token_completed,
    LfmAudioDetokenizerState *detokenizer,
    const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle,
    uint32_t *terminal_sampled = nullptr) {
    Engine *e = static_cast<Engine *>(ep);
    if (!notify || !out_handle) return -EINVAL;
    *out_handle = {};
    const bool terminal_after_token = terminal_sampled != nullptr;
    const bool decode_detokenizer = detokenizer || target || result;
    const bool commit_only = !terminal_after_token && !decode_detokenizer &&
                             depth_id == 0 && out_codes == nullptr &&
                             code_count == 0;
    if (terminal_after_token &&
        (decode_detokenizer || depth_id != 0 || out_codes != nullptr ||
         code_count != 0)) {
        return -EINVAL;
    }
    const bool resample_detokenizer =
        decode_detokenizer && target && target->resampler_stream;
    if (decode_detokenizer &&
        (!detokenizer || !target || !result || !target->epoch ||
                        !target->pcm || target->expected_epoch == 0 ||
                        target->pcm_capacity == 0 ||
                        (resample_detokenizer &&
                         (!target->detokenizer_pcm ||
                          target->detokenizer_pcm_capacity <
                              LFM_DETOKENIZER_MAX_STEP_SAMPLES)) ||
                        (!resample_detokenizer &&
                         target->pcm_capacity <
                             LFM_DETOKENIZER_MAX_STEP_SAMPLES) ||
                        (!resample_detokenizer &&
                         (target->detokenizer_pcm ||
                          target->detokenizer_pcm_capacity != 0 ||
                          target->resampler_stream)) ||
                        out_codes != result->codes ||
                        out_token_completed != &result->token_completed ||
                        !commit || commit->token_committed !=
                                       &result->token_committed ||
                        code_count != LFM_DETOKENIZER_CODEBOOKS)) {
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
    if (!e || model_id == 0 ||
        (!terminal_after_token && !commit_only && depth_id == 0) ||
        !ids || id_count == 0 || !states || !out_hidden ||
        (!terminal_after_token && !commit_only && !out_codes) ||
        !bound_commit.window || !logical_lane_count_valid(lanes) ||
        (!commit_only && !sample_config_valid(audio_sampler))) {
        return -EINVAL;
    }
    const bool stochastic = !commit_only &&
        (audio_sampler->flags & LFM_SAMPLE_FLAG_GREEDY) == 0 &&
        audio_sampler->top_k != 1;
    if (stochastic &&
        (!prng || lfm_prng_fill_u64(prng, nullptr, 0) != 0)) {
        return -EINVAL;
    }
    const int commit_status =
        preflight_audio_route_commit(&bound_commit, position);
    if (commit_status != 0) return commit_status;
    if (decode_detokenizer &&
        target->epoch->load(std::memory_order_acquire) !=
            target->expected_epoch) {
        return -ESTALE;
    }

    AudioRouteLease claim(
        e, decode_detokenizer
            ? kc::ServiceClass::Deadline
            : kc::ServiceClass::Interactive);
    if (!claim) return claim.status();
    AudioRouteInstance *route = claim.route();
    BackbonePlan *model = find_model(e, model_id);
    DepthPlan *depth = nullptr;
    if (!terminal_after_token && !commit_only) {
        for (const std::unique_ptr<DepthPlan> &candidate : e->depth_plans) {
            if (candidate->id == depth_id) {
                depth = candidate.get();
                break;
            }
        }
    }
    if (!model || !model->embed_w || !model->emb_norm_w ||
        (!terminal_after_token && !commit_only && !depth)) {
        return -ESTALE;
    }
    if (hidden_elements != model->h ||
        (!terminal_after_token && !commit_only &&
         hidden_elements != depth->backbone_dim) ||
        (!terminal_after_token && !commit_only &&
         code_count != depth->codebooks) ||
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
    route->kind = AUDIO_ROUTE_GENERATION;
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
    if (!terminal_after_token && !commit_only) route->depth_req = {
            .hidden = out_hidden,
            .sampler = *audio_sampler,
            .sample_state = prng,
            .out_tokens = out_codes,
            .completion_status =
                e->test_audio_route_depth_status.exchange(
                    0, std::memory_order_acq_rel),
        };
    route->detokenizer_req = {
            .state = detokenizer,
            .codes = out_codes,
            .pcm = decode_detokenizer ? target->pcm : nullptr,
            .capacity = decode_detokenizer ? target->pcm_capacity : 0,
            .detokenizer_pcm =
                resample_detokenizer ? target->detokenizer_pcm : nullptr,
            .detokenizer_capacity =
                resample_detokenizer
                    ? target->detokenizer_pcm_capacity
                    : 0,
            .resampler_stream =
                resample_detokenizer ? target->resampler_stream : nullptr,
            .out_samples = result ? &result->pcm_samples : nullptr,
            .flush = false,
            .completion_status = decode_detokenizer
                ? e->test_audio_route_detokenizer_status.exchange(
                      0, std::memory_order_acq_rel)
                : 0,
        };
    route->epoch = decode_detokenizer ? target->epoch : nullptr;
    route->expected_epoch = decode_detokenizer ? target->expected_epoch : 0;
    route->result = result;
    route->notify = notify;
    route->notify_context = notify_context;
    route->terminal_after_token = terminal_after_token || commit_only;
    route->decode_detokenizer = decode_detokenizer;
    route->commit = bound_commit;
    route->token_completed = out_token_completed;
    route->status = -EINPROGRESS;
    out_handle->record = route;
    out_handle->generation = claim.generation();
    out_handle->ticket = route->ticket;
    const int published = claim.publish();
    if (published != 0) {
        *out_handle = {};
        return published;
    }
    return 0;
}

int lfm_engine_audio_route_submit(
    void *ep, uint64_t model_id, uint64_t depth_id,
    const uint32_t *ids, size_t id_count, uint32_t embedding_kind,
    const LfmLayerState *states, size_t state_count, size_t position,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    size_t rope_elements, uint16_t *out_hidden, size_t hidden_elements,
    const LfmSamplerConfigV1 *audio_sampler, LfmPrngStateV1 *prng,
    LfmAudioDetokenizerState *detokenizer,
    const LfmAudioRouteTarget *target,
    LfmAudioRouteResult *result, size_t lanes,
    const LfmTokenCommitRecord *commit, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    if (!result || !notify || !out_handle) return -EINVAL;
    return run_audio_route(
        ep, model_id, depth_id, ids, id_count, embedding_kind, states,
        state_count, position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, audio_sampler, prng, result->codes,
        LFM_DETOKENIZER_CODEBOOKS, lanes, commit, &result->token_completed,
        detokenizer,
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

int lfm_engine_token_commit_route_submit(
    void *ep, uint64_t model_id, const uint32_t *ids, size_t id_count,
    uint32_t embedding_kind, const LfmLayerState *states,
    size_t state_count, size_t position, const uint16_t *rope_cos,
    const uint16_t *rope_sin, size_t rope_elements, uint16_t *out_hidden,
    size_t hidden_elements, size_t lanes,
    const LfmTokenCommitRecord *commit, uint32_t *out_token_completed,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *out_handle) {
    if (!notify || !out_handle || !out_token_completed) return -EINVAL;
    return run_audio_route(
        ep, model_id, 0, ids, id_count, embedding_kind, states, state_count,
        position, rope_cos, rope_sin, rope_elements, out_hidden,
        hidden_elements, nullptr, nullptr, nullptr, 0, lanes, commit,
        out_token_completed, nullptr, nullptr, nullptr, notify,
        notify_context, out_handle);
}

int lfm_engine_control_route_submit(
    void *ep, LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *out_handle) {
    Engine *engine = static_cast<Engine *>(ep);
    if (!engine || !notify || !out_handle) return -EINVAL;
    *out_handle = {};
    AudioRouteLease claim(engine);
    if (!claim) return claim.status();
    AudioRouteInstance *route = claim.route();
    route->engine = engine;
    route->kind = AUDIO_ROUTE_CONTROL;
    route->node = AUDIO_ROUTE_TERMINAL;
    route->model = nullptr;
    route->model_id = 0;
    route->result = nullptr;
    route->adapted_values = nullptr;
    route->notify = notify;
    route->notify_context = notify_context;
    route->terminal_after_token = false;
    route->decode_detokenizer = false;
    route->status = -EINPROGRESS;
    out_handle->record = route;
    out_handle->generation = claim.generation();
    out_handle->ticket = route->ticket;
    const int published = claim.publish();
    if (published != 0) {
        *out_handle = {};
        return published;
    }
    return 0;
}

int lfm_engine_audio_route_collect(void *ep,
                                   LfmAudioRouteHandle *handle) {
    Engine *engine = static_cast<Engine *>(ep);
    if (!engine || !handle || !handle->record ||
        handle->generation == 0) {
        return -EINVAL;
    }
    AudioRouteInstance *route =
        static_cast<AudioRouteInstance *>(handle->record);
    AudioRouteBroker::Lease lease{};
    if (engine->routes.locate(
            route, handle->generation, handle->ticket,
            &lease) != 0 ||
        route->engine != engine) {
        return -ESTALE;
    }
    if (!engine->routes.done(lease)) {
        return -EINPROGRESS;
    }
    const int status = route->status;
    if (engine->routes.release(lease) != 0) std::abort();
    leave_pass_admission(engine);
    *handle = {};
    return status;
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
        for (size_t index = 0; index < e->slots.size(); ++index) {
            PassSlot &slot = e->slots.record(index);
            ScratchBank &scratch = slot.scratch;
            grow(scratch.sc_partials, MAX_WORKERS);
            grow(scratch.sc_xn, h);
            grow(scratch.sc_t, ffn);
            grow(scratch.sc_projb, h);
            grow(scratch.sc_mid, h);
            if (qkv_max > 0) {
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
        for (size_t index = 0; index < e->slots.size(); ++index) {
            PassSlot &slot = e->slots.record(index);
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

int lfm_engine_prefill_workspace_create(void *ep, uint64_t id,
                                        void **out_workspace) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !out_workspace) return -EINVAL;
    *out_workspace = nullptr;
    PlanClaim claim(e);
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

int lfm_engine_prefill_submit(
    void *ep, uint64_t id, void *workspace_pointer, const uint32_t *ids,
    const uint16_t *provided_rows, size_t row_count, uint32_t embed_kind,
    const LfmLayerState *states, size_t state_count, size_t pos,
    const uint16_t *cos_base, const uint16_t *sin_base, size_t rope_len,
    uint16_t *out_hidden, size_t out_hidden_len,
    const LfmSamplerConfigV1 *sampler, LfmPrngStateV1 *sample_state,
    uint32_t *out_token, size_t lanes,
    LfmListenReadoutForTest *readout, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    Engine *e = (Engine *)ep;
    PrefillWorkspace *workspace =
        static_cast<PrefillWorkspace *>(workspace_pointer);
    if (!e || id == 0 || !workspace || row_count == 0 ||
        row_count > PREFILL_ROWS || !states || !out_hidden ||
        !logical_lane_count_valid(lanes) || !notify || !out_handle) {
        return -EINVAL;
    }
    *out_handle = {};
    AudioRouteLease claim(e);
    if (!claim) return claim.status();
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
    size_t provided_values = 0;
    if (!checked_size_product(row_count, model->h, &provided_values)) {
        return -EOVERFLOW;
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
    DepthPlan *readout_depth = nullptr;
    if (readout) {
        /* Test-only readout: it rides a sampled pass and evaluates the named
         * Depthformer's codebook-0 head on the pass's published hidden. */
        if (!out_token || readout->depth_id == 0) return -EINVAL;
        for (const std::unique_ptr<DepthPlan> &candidate : e->depth_plans) {
            if (candidate->id == readout->depth_id) {
                readout_depth = candidate.get();
                break;
            }
        }
        if (!readout_depth) return -ESTALE;
        if (readout_depth->backbone_dim != model->h ||
            readout_depth->heads.empty()) {
            return -EINVAL;
        }
    }

    const size_t end_pos = pos + row_count;
    for (size_t layer = 0; layer < model->layers.size(); ++layer) {
        const LfmLayerDesc *desc = &model->layers[layer];
        const LfmLayerState *state = &states[layer];
        if (desc->kind == 1) {
            if (!desc->q_w || !state->k_plane || !state->v_plane ||
                !cos_base || !sin_base || desc->hd < 2 ||
                desc->hd % 2 != 0 || desc->n_kv == 0 ||
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
        if (desc->kind != 0) return -ESTALE;
        const size_t tail = desc->k > 0 ? desc->k - 1 : 0;
        if (!state->conv_state || desc->k < 1 ||
            (tail > 0 && model->h > SIZE_MAX / tail) ||
            state->conv_len < model->h * tail) {
            return -ESTALE;
        }
    }

    AudioRouteInstance *route = claim.route();
    route->engine = e;
    route->kind = AUDIO_ROUTE_PREFILL;
    route->node = AUDIO_ROUTE_TERMINAL;
    route->model = model;
    route->model_id = id;
    route->prefill_req = {};
    route->prefill_req.workspace = workspace;
    if (ids) {
        std::copy_n(ids, row_count, route->prefill_req.ids.begin());
    }
    route->prefill_req.provided_rows = provided_rows;
    route->prefill_req.provided_values =
        embed_kind == 2 ? provided_values : 0;
    route->prefill_req.rows = row_count;
    route->prefill_req.embed_kind = embed_kind;
    route->prefill_req.states = states;
    route->prefill_req.n_states = state_count;
    route->prefill_req.pos = pos;
    route->prefill_req.cos_base = cos_base;
    route->prefill_req.sin_base = sin_base;
    route->prefill_req.rope_len = rope_len;
    route->prefill_req.out_hidden = out_hidden;
    route->prefill_req.out_hidden_len = out_hidden_len;
    if (sampler) route->prefill_req.sampler = *sampler;
    route->prefill_req.sample = out_token != nullptr;
    route->prefill_req.sample_state = sample_state;
    route->prefill_req.out_token = out_token;
    route->prefill_req.readout = readout;
    route->prefill_req.lanes = lanes;
    if (readout) {
        route->depth = readout_depth;
        route->depth_id = readout->depth_id;
        route->depth_req = {};
        route->depth_req.hidden = out_hidden;
    }
    route->adapted_values = nullptr;
    route->result = nullptr;
    route->notify = notify;
    route->notify_context = notify_context;
    route->terminal_after_token = false;
    route->decode_detokenizer = false;
    route->status = -EINPROGRESS;
    out_handle->record = route;
    out_handle->generation = claim.generation();
    out_handle->ticket = route->ticket;
    const int published = claim.publish();
    if (published != 0) {
        *out_handle = {};
        return published;
    }
    return 0;
}

} // extern "C"
