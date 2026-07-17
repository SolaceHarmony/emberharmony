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
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <pthread.h>
#include <utility>
#include <vector>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

#include "flashkern_conv.h"
#include "flashkern_depth.h"
#include "flashkern_fft.h"
#include "flashkern_gemm.h"
#include "flashkern_math.h"
#include "flashkern_prng.h"
#include "flashkern_sampler.h"
#include "lfm_kernel_bridge.h"
#include "lfm_model_plan.h"

extern "C" {
#include "kc_atomic.h"
#include "kc_port.h"
}

// Stage kernels from the flashkern TU (same image, plain calls).
extern "C" float lfm_bf16_sumsq_candle_f32(const uint16_t *x, int n);
extern "C" float lfm_bf16_sumsq_f32(const uint16_t *x, int n);
extern "C" void lfm_bf16_rmsnorm(const uint16_t *x, const uint16_t *w, uint16_t *out,
                                 int n, float inv_rms);
extern "C" void lfm_f32_to_bf16(const float *x, uint16_t *out, int n);
extern "C" void lfm_bf16_add(const uint16_t *a, const uint16_t *b, uint16_t *out, int n);
extern "C" void lfm_conv1d_update_bf16(const uint16_t *bcx, const uint16_t *state,
                                       const uint16_t *w, uint16_t *out, int bn, int d,
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
constexpr size_t DOWN_BAND_CAP = 512; // worker-stack y[] extent
std::atomic<uint64_t> next_engine_epoch{1};

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

// ---- the pass (engine-owned pointers; nothing here ever rides a message) ------------
struct Pass {
    const uint16_t *x;      // [h] bf16 bits
    const uint16_t *norm_w; // [h]
    const uint16_t *w1;     // [i,h]
    const uint16_t *w3;     // [i,h]
    const uint16_t *w2;     // [h,i]
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

// The generation fence — the GPU barrier idiom on fixed lanes. The last arriver runs
// the boundary's serial section, release-publishes the next generation, and rings one
// shared expected-value word. The host wakes only threads parked on that address, so
// one syscall fans out to the actual waiter set without polling.
struct Fence {
    std::atomic<uint32_t> arrived{0};
    std::atomic<uint32_t> park_mask{0};
    std::atomic<uint64_t> gen{0};
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
};

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
    const uint16_t *rhs = nullptr;
    float *amx_a = nullptr;
    float *amx_rhs = nullptr;
    float *out = nullptr;
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    uint32_t rhs_layout = LFM_GEMM_RHS_KN;
    bool use_amx = false;
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
    const uint16_t *depth_linear_w = nullptr;
    const uint16_t *depth_linear_b = nullptr;
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

    std::vector<uint16_t> x, h, xn, qkv_b, y_b, attn_b, t_b;
    std::vector<uint16_t> k_plane, v_plane, logits_b, din_b, df_b;
    std::vector<float> qkv_f, up_f, q_f, attn_f, proj_f;
    float partials[MAX_WORKERS] = {};
};

struct BackbonePlan {
    uint64_t id = 0;
    std::vector<LfmLayerDesc> layers;
    const uint16_t *embed_w = nullptr;
    const uint16_t *audio_embed_w = nullptr;
    const uint16_t *emb_norm_w = nullptr;
    float emb_norm_eps = 0.0f;
    size_t vocab = 0;
    size_t audio_rows = 0;
    size_t h = 0;
    size_t ffn = 0;
    size_t max_ctx = 0;
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
    const uint16_t *x = nullptr;       // block input [H]
    const uint16_t *norm_w = nullptr;  // operator norm [H]
    const uint16_t *in_w = nullptr;    // [3H, H]
    const uint16_t *out_w = nullptr;   // [H, H]
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
    const uint16_t *o_w = nullptr; // [H, nh·hd]
    uint16_t *qkvb = nullptr;      // rounded q|k|v rows [(nh+2·nkv)·hd]
    float *qkvf = nullptr;
    uint16_t *ybits = nullptr;     // attention output per q head [nh·hd]
    float *att = nullptr;          // per-head score scratch [nh · max_ctx]
    const uint16_t *x = nullptr;   // residual input [H]
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

struct Engine;
struct alignas(64) WaitWord {
    uint32_t value = 0;
    uint32_t reserved = 0;
    kc_port_wait_word *wait = nullptr;
    uint8_t padding[48] = {};
};
static_assert(sizeof(WaitWord) == 64, "shared doorbells must not share cache lines");

struct LaneArg {
    Engine *e;
    uint32_t lane;
};

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
};

struct Engine {
    Pass pass;
    Stage stage;
    Fence fence;

    // Stable logical lane i always runs on pthread i. The SQ/CQ dispatcher only
    // release-rings full passes; it never schedules or migrates numerical call stacks.
    pthread_t workers[MAX_WORKERS] = {};
    pthread_t bridge_worker{};
    WaitWord dispatch_word;
    WaitWord fence_word;
    LaneArg largs[MAX_WORKERS] = {};
    int n_workers = 0;
    int wait_words_prepared = 0;
    int workers_started = 0;
    int bridge_started = 0;
    uint32_t lanes_total = 1;
    std::atomic<uint64_t> lane_gen{0};
    int cur_req = REQ_NONE;
    std::atomic<bool> retire{false};
    LfmKernelBridge *bridge = nullptr;
    KcSubmissionV1 active_submission{};
    std::atomic<bool> pass_claimed{false};
    std::atomic<int> active_status{0};
    uint64_t runtime_epoch = 0;
    uint64_t submit_sequence = 0; // written only by the current pass claimant
    uint32_t ticket_generation = 0;
    std::atomic<uint64_t> pass_submissions{0};
    std::atomic<uint64_t> pass_completions{0};
    std::atomic<uint64_t> bridge_dispatches{0};
    std::atomic<uint64_t> dispatch_wakes{0};
    std::atomic<uint64_t> fence_wake_calls{0};
    std::atomic<uint64_t> fence_wakes{0};

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
    // Apple matrix-matrix staging. Capacity grows before admission; numerical
    // lanes only widen into these fixed planes, then one fence callback enters
    // Accelerate/AMX. No Rust buffer or allocation participates.
    std::vector<float> gemm_amx_a, gemm_amx_rhs;
    float sample_lane_value[MAX_WORKERS] = {};
    float sample_lane_sum[MAX_WORKERS] = {};
    uint32_t sample_lane_index[MAX_WORKERS] = {};
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
};

static BackbonePlan *find_model(Engine *e, uint64_t id) {
    for (const std::unique_ptr<BackbonePlan> &model : e->models)
        if (model->id == id) return model.get();
    return nullptr;
}

// Self-enforcing single-slot ownership at the raw C ABI. This claim must be acquired
// before a caller reads or writes any engine-owned request, context, or scratch state;
// the Rust pass_lock is an additional language-side guarantee, not the foundation.
class PassClaim {
  public:
    explicit PassClaim(Engine *engine) : engine_(engine) {
        bool expected = false;
        held_ = engine_ && engine_->pass_claimed.compare_exchange_strong(
                               expected, true, std::memory_order_acq_rel,
                               std::memory_order_acquire);
    }

    ~PassClaim() {
        if (held_) engine_->pass_claimed.store(false, std::memory_order_release);
    }

    explicit operator bool() const { return held_; }
    PassClaim(const PassClaim &) = delete;
    PassClaim &operator=(const PassClaim &) = delete;

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
        lfm_bf16_rmsnorm(p->x + begin, p->norm_w + begin, p->xn + begin,
                         (int)(end - begin), rs);
        break;
    }
    case ST_GATEUP: {
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < p->i ? r0 + st->chunk : p->i;
        if (r1 <= r0) break;
        size_t n = r1 - r0;
        lfm_bf16_gemm_nt_f32(p->xn, p->w1 + r0 * p->h, p->gu + r0, 1, (int)n, (int)p->h);
        lfm_bf16_gemm_nt_f32(p->xn, p->w3 + r0 * p->h, p->gu + p->i + r0, 1, (int)n,
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
        lfm_bf16_gemm_nt_f32(p->t, p->w2 + r0 * p->i, y, 1, (int)n, (int)p->i);
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
        lfm_bf16_rmsnorm(c->x + r0, c->norm_w + r0, c->xn + r0, (int)(r1 - r0), inv_rms);
        break;
    }
    case ST_SC_INPROJ: {
        ScPass *c = &e->sc;
        size_t rows = 3 * c->h;
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < rows ? r0 + st->chunk : rows;
        if (r1 <= r0) break;
        lfm_bf16_gemm_nt_f32(c->xn, c->in_w + r0 * c->h, c->bcxf + r0, 1,
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
        lfm_bf16_gemm_nt_f32(c->projb, c->out_w + r0 * c->h, c->projf + r0, 1,
                             (int)(r1 - r0), (int)c->h);
        lfm_f32_to_bf16(c->projf + r0, c->stage + r0, (int)(r1 - r0));
        lfm_bf16_add(c->stage + r0, c->x + r0, c->mid + r0, (int)(r1 - r0));
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
            const uint16_t *w;
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
            lfm_bf16_gemm_nt_f32(e->sc_xn.data(), w + (r - seg0) * a->h, a->qkvf + r, 1,
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
        lfm_bf16_gemm_nt_f32(a->ybits, a->o_w + r0 * kdim, c->projf + r0, 1,
                             (int)(r1 - r0), (int)kdim);
        lfm_f32_to_bf16(c->projf + r0, c->stage + r0, (int)(r1 - r0));
        lfm_bf16_add(c->stage + r0, a->x + r0, a->mid + r0, (int)(r1 - r0));
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
                             ee->model->embed_w + r0 * ee->model->h, acc, 1,
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

// One stage boundary. `serial` runs exactly once, on the last arriver, AFTER every
// lane's pre-fence work is complete and BEFORE any lane crosses — the collective
// serial section. Bit-determinism does not care which lane executes it: all operands
// live in engine-owned planes and every ladder has a fixed internal order.
template <typename F>
static inline void lane_fence(Engine *e, uint32_t lane, F &&serial) {
    Fence *f = &e->fence;
    uint64_t g = f->gen.load(std::memory_order_relaxed);
    if (f->arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == e->lanes_total) {
        serial();
        f->arrived.store(0, std::memory_order_relaxed); // before the release below
        f->gen.store(g + 1, std::memory_order_release);
        uint32_t parked = f->park_mask.exchange(0, std::memory_order_acq_rel);
        if (parked) {
            e->fence_wake_calls.fetch_add(1, std::memory_order_relaxed);
            e->fence_wakes.fetch_add((uint32_t)__builtin_popcount(parked),
                                     std::memory_order_relaxed);
            signal_all(&e->fence_word);
        }
        return;
    }
    uint32_t bit = 1u << lane;
    uint32_t expected = kc_atomic_u32_load_acquire(&e->fence_word.value);
    f->park_mask.fetch_or(bit, std::memory_order_acq_rel);
    if (f->gen.load(std::memory_order_acquire) != g) {
        f->park_mask.fetch_and(~bit, std::memory_order_acq_rel);
        return;
    }
    while (f->gen.load(std::memory_order_acquire) == g) {
        (void)kc_port_wait_u32(e->fence_word.wait, expected, 0);
        expected = kc_atomic_u32_load_acquire(&e->fence_word.value);
    }
    f->park_mask.fetch_and(~bit, std::memory_order_acq_rel);
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

static inline const uint16_t *depth_u16(const LfmDepthBufferV1 &view) {
    return reinterpret_cast<const uint16_t *>(view.address);
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
        lfm_bf16_gemm_nt_f32(x, depth_u16(weight) + begin * cols, out + begin,
                             1, (int)(end - begin), (int)cols);
}

static void depth_norm(Engine *e, uint32_t lane, const uint16_t *x,
                       const LfmDepthBufferV1 &weight, uint16_t *out) {
    DepthPlan &d = *e->active_depth;
    size_t begin = 0, end = 0;
    depth_band(d.dim, lane, e->lanes_total, &begin, &end);
    d.partials[lane] = end > begin
                           ? lfm_bf16_sumsq_f32(x + begin, (int)(end - begin))
                           : 0.0f;
    lane_fence(e, lane, [] {});
    float total = lfm_sum_f32(d.partials, e->lanes_total);
    const float inv_rms = lfm_inv_rms_f32(total, d.dim, d.eps);
    if (end > begin)
        lfm_bf16_rmsnorm(x + begin, depth_u16(weight) + begin, out + begin,
                         (int)(end - begin), inv_rms);
    lane_fence(e, lane, [] {});
}

static void depth_qk_head(const DepthPlan &d, const uint16_t *src,
                          const LfmDepthBufferV1 &weight, uint16_t *out,
                          size_t position) {
    uint16_t normed[128];
    float rotated[128];
    const float sum = lfm_bf16_sumsq_f32(src, (int)d.hd);
    const float inv_rms = lfm_inv_rms_f32(sum, d.hd, d.eps);
    lfm_bf16_rmsnorm(src, depth_u16(weight), normed, (int)d.hd, inv_rms);
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
    const DepthReq &request = e->depth_req;
    const uint32_t lanes = e->lanes_total;
    const size_t qkv_rows = d.dim + 2 * d.kv_heads * d.hd;
    const size_t group = d.heads_total / d.kv_heads;
    const float attn_scale = lfm_rsqrt_size(d.hd);

    // depth_linear(hidden) + bias -> one row per codebook.
    depth_gemv({reinterpret_cast<uintptr_t>(d.depth_linear_w),
                d.codebooks * d.dim * d.backbone_dim},
               request.hidden, d.proj_f.data(), d.codebooks * d.dim,
               d.backbone_dim, lane, lanes);
    size_t begin = 0, end = 0;
    depth_band(d.codebooks * d.dim, lane, lanes, &begin, &end);
    if (end > begin)
        lfm_bf16_bias_add_f32(d.proj_f.data() + begin,
                              d.depth_linear_b + begin, end - begin);
    if (end > begin)
        lfm_f32_to_bf16(d.proj_f.data() + begin, d.din_b.data() + begin,
                        (int)(end - begin));
    depth_band(d.dim, lane, lanes, &begin, &end);
    std::fill(d.df_b.begin() + begin, d.df_b.begin() + end, (uint16_t)0);
    lane_fence(e, lane, [] {});

    for (size_t codebook = 0; codebook < d.codebooks; ++codebook) {
        depth_band(d.dim, lane, lanes, &begin, &end);
        if (end > begin)
            lfm_bf16_add(d.din_b.data() + codebook * d.dim + begin,
                         d.df_b.data() + begin, d.x.data() + begin,
                         (int)(end - begin));
        lane_fence(e, lane, [] {});

        for (size_t layer = 0; layer < d.layers.size(); ++layer) {
            const LfmDepthLayerV1 &weights = d.layers[layer];
            const size_t cache_base = layer * d.kv_heads * d.codebooks * d.hd;

            depth_norm(e, lane, d.x.data(), weights.op_norm, d.xn.data());
            depth_gemv(weights.qkv_w, d.xn.data(), d.qkv_f.data(), qkv_rows,
                       d.dim, lane, lanes);
            depth_band(qkv_rows, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_f32_to_bf16(d.qkv_f.data() + begin, d.qkv_b.data() + begin,
                                (int)(end - begin));
            lane_fence(e, lane, [] {});

            const size_t normalized_heads = d.heads_total + d.kv_heads;
            depth_band(normalized_heads, lane, lanes, &begin, &end);
            for (size_t head = begin; head < end; ++head) {
                if (head < d.heads_total) {
                    uint16_t bits[128];
                    depth_qk_head(d, d.qkv_b.data() + head * d.hd,
                                  weights.q_ln, bits, codebook);
                    lfm_bf16_to_f32(bits, d.q_f.data() + head * d.hd, (int)d.hd);
                    continue;
                }
                const size_t kv = head - d.heads_total;
                uint16_t *key = d.k_plane.data() + cache_base +
                                (kv * d.codebooks + codebook) * d.hd;
                depth_qk_head(d, d.qkv_b.data() + d.dim + kv * d.hd,
                              weights.k_ln, key, codebook);
                const uint16_t *value = d.qkv_b.data() + d.dim +
                                        d.kv_heads * d.hd + kv * d.hd;
                std::memcpy(d.v_plane.data() + cache_base +
                                (kv * d.codebooks + codebook) * d.hd,
                            value, d.hd * sizeof(uint16_t));
            }
            lane_fence(e, lane, [] {});

            depth_band(d.heads_total, lane, lanes, &begin, &end);
            const int live = (int)(codebook + 1);
            for (size_t query = begin; query < end; ++query) {
                float attention[64];
                const size_t kv = query / group;
                lfm_attn_qk_bf16(d.q_f.data() + query * d.hd,
                                  d.k_plane.data() + cache_base +
                                      kv * d.codebooks * d.hd,
                                  attention, live, (int)d.hd);
                lfm_softmax_scaled_f32(attention, live, attn_scale);
                lfm_attn_av_bf16(attention,
                                  d.v_plane.data() + cache_base +
                                      kv * d.codebooks * d.hd,
                                  d.attn_f.data() + query * d.hd, live, (int)d.hd);
            }
            lane_fence(e, lane, [] {});

            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_f32_to_bf16(d.attn_f.data() + begin, d.attn_b.data() + begin,
                                (int)(end - begin));
            lane_fence(e, lane, [] {});

            depth_gemv(weights.out_w, d.attn_b.data(), d.proj_f.data(), d.dim,
                       d.dim, lane, lanes);
            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin) {
                lfm_f32_to_bf16(d.proj_f.data() + begin, d.y_b.data() + begin,
                                (int)(end - begin));
                lfm_bf16_add(d.y_b.data() + begin, d.x.data() + begin,
                             d.h.data() + begin, (int)(end - begin));
            }
            lane_fence(e, lane, [] {});

            depth_norm(e, lane, d.h.data(), weights.ffn_norm, d.xn.data());
            depth_gemv(weights.w1, d.xn.data(), d.proj_f.data(), d.ffn, d.dim,
                       lane, lanes);
            depth_gemv(weights.w3, d.xn.data(), d.up_f.data(), d.ffn, d.dim,
                       lane, lanes);
            depth_band(d.ffn, lane, lanes, &begin, &end);
            if (end > begin)
                lfm_swiglu_bf16(d.proj_f.data() + begin, d.up_f.data() + begin,
                                d.t_b.data() + begin, (int)(end - begin));
            lane_fence(e, lane, [] {});

            depth_gemv(weights.w2, d.t_b.data(), d.proj_f.data(), d.dim, d.ffn,
                       lane, lanes);
            depth_band(d.dim, lane, lanes, &begin, &end);
            if (end > begin) {
                lfm_f32_to_bf16(d.proj_f.data() + begin, d.y_b.data() + begin,
                                (int)(end - begin));
                lfm_bf16_add(d.y_b.data() + begin, d.h.data() + begin,
                             d.x.data() + begin, (int)(end - begin));
            }
            lane_fence(e, lane, [] {});
        }

        const LfmDepthHeadV1 &head = d.heads[codebook];
        depth_norm(e, lane, d.x.data(), head.norm, d.xn.data());
        depth_gemv(head.logits, d.xn.data(), d.proj_f.data(), head.vocab, d.dim,
                   lane, lanes);
        depth_band(head.vocab, lane, lanes, &begin, &end);
        if (end > begin)
            lfm_f32_to_bf16(d.proj_f.data() + begin, d.logits_b.data() + begin,
                            (int)(end - begin));
        lane_fence(e, lane, [] {});

        SampleReq sample = {
            .logits = d.logits_b.data(),
            .count = head.vocab,
            .dtype = SAMPLE_BF16,
            .config = request.sampler,
            .state = request.sample_state,
            .out = request.out_tokens + codebook,
        };
        run_sampler(e, lane, sample);

        const size_t token = request.out_tokens[codebook];
        depth_band(d.dim, lane, lanes, &begin, &end);
        const uint16_t *embedding = depth_u16(head.embedding) + token * d.dim;
        std::copy(embedding + begin, embedding + end, d.df_b.begin() + begin);
        lane_fence(e, lane, [] {});
    }
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
                           const uint16_t *x, const uint16_t *state_in,
                           uint16_t *state_out, uint16_t *out, size_t lanes) {
    ScPass *c = &e->sc;
    const size_t h = e->model->h;
    if (lanes < 1) lanes = 1;
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
        float total = lfm_bf16_sumsq_candle_f32(x, (int)h);
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
static void qk_norm_row(const uint16_t *x, const uint16_t *w, uint16_t *out, size_t hd,
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
                           const uint16_t *x, uint16_t *k_plane, uint16_t *v_plane,
                           size_t head_stride, size_t pos, const uint16_t *cos_base,
                           const uint16_t *sin_base, uint16_t *out, size_t lanes) {
    const LfmLayerDesc *d = &e->model->layers[layer_idx];
    ScPass *c = &e->sc;
    AtPass *a = &e->at;
    const size_t h = e->model->h;
    const size_t nh = d->n_head, nkv = d->n_kv, hd = d->hd;
    if (lanes < 1) lanes = 1;
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
        float total = lfm_bf16_sumsq_candle_f32(x, (int)h);
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
    size_t lanes = t->lanes < 1 ? 1 : t->lanes;
    uint16_t *h0 = e->tk_h0.data();
    uint16_t *h1 = e->tk_h1.data();

    lane_fence(e, lane, [&] {
        // Embed (serial — at most 8 rows). Text: one table row copied verbatim.
        // Audio: candle's `.sum(0)` over the gathered rows — sequential bf16 adds
        // from a bf16 zero, one RNE round per step (candle's in-dtype reduction).
        if (t->embed_kind == 2) {
            // Provided embedding (native audio-in prefill): the conformer/adapter
            // hidden row is fed verbatim as a view into the source buffer — no
            // table lookup. h0 is per-pass scratch, so loading it is not a copy of
            // any weight; the source stays borrowed until the pass completes.
            std::memcpy(h0, t->provided_embed, h * sizeof(uint16_t));
        } else if (t->embed_kind == 0) {
            std::memcpy(h0, model->embed_w + (size_t)t->ids[0] * h,
                        h * sizeof(uint16_t));
        } else {
            std::memset(h0, 0, h * sizeof(uint16_t));
            for (size_t c = 0; c < t->n_ids; ++c) {
                const uint16_t *row = model->audio_embed_w + (size_t)t->ids[c] * h;
                lfm_bf16_add(h0, row, h0, (int)h);
            }
        }
    });

    // The layer walk. x = h0, out = h1, swap — no Tensor, no Rust, no copies.
    for (size_t l = 0; l < model->layers.size(); ++l) {
        const LfmLayerDesc *d = &model->layers[l];
        const LfmLayerState *st = &t->states[l];
        if (d->kind == 0) {
            run_conv_block(e, lane, d, h0, st->conv_state, st->conv_state, h1, lanes);
        } else {
            run_attn_block(e, lane, l, h0, st->k_plane, st->v_plane, st->head_stride,
                           t->pos, t->cos_base, t->sin_base, h1, lanes);
        }
        uint16_t *tmp = h0;
        h0 = h1;
        h1 = tmp;
    }

    // Final embedding-norm (candle RmsNorm: f32 arithmetic, one bf16 round), banded.
    ScPass *c = &e->sc;
    uint32_t tiles = (uint32_t)(lanes > h ? h : lanes);
    uint32_t hc = (uint32_t)((h + tiles - 1) / tiles);
    run_stage(e, lane, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc, [&] {
        float total = lfm_bf16_sumsq_candle_f32(h0, (int)h);
        float inv_rms = lfm_inv_rms_f32(total, h, model->emb_norm_eps);
        uint32_t rsb;
        std::memcpy(&rsb, &inv_rms, 4);
        c->rs_bits.store(rsb, std::memory_order_release);
        c->x = h0;
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

#ifdef __APPLE__
    if (request.use_amx) {
        const size_t a_count = request.m * request.k;
        const size_t rhs_count = request.rhs_layout == LFM_GEMM_RHS_KN
                                     ? request.k * request.n
                                     : request.n * request.k;
        const auto widen = [lane, lanes](const uint16_t *src, float *dst, size_t count) {
            const size_t chunk = (count + lanes - 1) / lanes;
            const size_t begin = (size_t)lane * chunk;
            if (begin >= count) return;
            const size_t width = std::min(chunk, count - begin);
            lfm_bf16_to_f32(src + begin, dst + begin, (int)width);
        };
        widen(request.a, request.amx_a, a_count);
        widen(request.rhs, request.amx_rhs, rhs_count);
        lane_fence(e, lane, [&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans,
                        request.rhs_layout == LFM_GEMM_RHS_NK ? CblasTrans
                                                              : CblasNoTrans,
                        (int)request.m, (int)request.n, (int)request.k, 1.0f,
                        request.amx_a, (int)request.k, request.amx_rhs,
                        request.rhs_layout == LFM_GEMM_RHS_NK ? (int)request.k
                                                              : (int)request.n,
                        0.0f, request.out, (int)request.n);
        });
        return;
    }
#endif

    if (request.rhs_layout == LFM_GEMM_RHS_KN && request.m == 1) {
        if (lane == 0)
            lfm_bf16_gemv_f32(request.a, request.rhs, request.out,
                              (int)request.n, (int)request.k);
        return;
    }

    if (request.rhs_layout == LFM_GEMM_RHS_NK && request.m == 1) {
        const size_t cols = std::max<size_t>((request.n + lanes - 1) / lanes, 64);
        const size_t col = (size_t)lane * cols;
        if (col < request.n) {
            const size_t count = std::min(cols, request.n - col);
            lfm_bf16_gemm_nt_f32(request.a, request.rhs + col * request.k,
                                 request.out + col, 1, (int)count, (int)request.k);
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
        lfm_bf16_gemm_f32_v2(request.a + row * request.k, request.rhs,
                             request.out + row * request.n, (int)count,
                             (int)request.n, (int)request.k);
        return;
    }
    lfm_bf16_gemm_nt_f32(request.a + row * request.k, request.rhs,
                         request.out + row * request.n, (int)count,
                         (int)request.n, (int)request.k);
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
        run_conv_block(e, lane, &e->model->layers[r->layer], r->x, r->state_in,
                       r->state_out, r->out, r->lanes);
        break;
    }
    case REQ_ATTN_LAYER: {
        const AttnReq *r = &e->attn;
        run_attn_block(e, lane, r->layer, r->x, r->k_plane, r->v_plane, r->head_stride,
                       r->pos, r->cos_base, r->sin_base, r->out, r->lanes);
        break;
    }
    case REQ_TOKEN_PASS:
        run_token_pass(e, lane);
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
        run_irfft_dd(e, lane);
        break;
    default:
        break;
    }
    lane_fence(e, lane, [] {});
}

// Every fixed lane blocks on its own expected-value word between passes. The pass
// generation remains the predicate; the word is only the edge that makes it recheck.
static void *lane_main(void *arg) {
    LaneArg *la = (LaneArg *)arg;
    Engine *e = la->e;
    const uint32_t lane = la->lane;
    uint32_t observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
    uint64_t seen = 0;
    for (;;) {
        while (e->lane_gen.load(std::memory_order_acquire) == seen &&
               !e->retire.load(std::memory_order_acquire)) {
            int rc = kc_port_wait_u32(e->dispatch_word.wait, observed, 0);
            observed = kc_atomic_u32_load_acquire(&e->dispatch_word.value);
            if (rc != 0 && e->retire.load(std::memory_order_acquire)) return nullptr;
        }
        bool retire = e->retire.load(std::memory_order_acquire);
        uint64_t generation = e->lane_gen.load(std::memory_order_acquire);
        if (retire) return nullptr;
        seen = generation;
        lane_program(e, lane);
        if (lane == 0) {
            const KcSubmissionV1 submission = e->active_submission;
            KcCompletionV1 completion{};
            completion.size = sizeof(completion);
            completion.abi_version = KC_COORD_ABI_VERSION;
            completion.ticket = submission.ticket;
            completion.conversation_id = submission.conversation_id;
            completion.epoch = submission.epoch;
            completion.pass_id = submission.ticket.sequence;
            completion.status = e->active_status.load(std::memory_order_acquire);
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
}

static bool ticket_equal(const KcTicketIdV1 &a, const KcTicketIdV1 &b) {
    return a.runtime_epoch == b.runtime_epoch && a.sequence == b.sequence &&
           a.generation == b.generation && a.kind == b.kind;
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

// The bridge dispatcher is mechanical: consume one retained descriptor, validate
// its generation against the single request slot, and release-ring the lane team.
// Model recurrence remains inside the native session; the host only docks audio
// buffers and control tickets at the session boundary.
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
        int descriptor_rc = lfm_kernel_bridge_descriptor_get(
            e->bridge, submission.descriptor, &descriptor);
        bool valid = descriptor_rc == 0 &&
                     submission.command == KC_COORD_COMMAND_RUN_PASS &&
                     submission.pass_budget == 1 &&
                     submission.ticket.kind == KC_COORD_TICKET_PASS &&
                     submission.epoch != 0 &&
                     descriptor.payload == e && descriptor.flags == 0 &&
                     descriptor.kind > REQ_NONE &&
                     descriptor.kind <= REQ_IRFFT_DD;
        if (valid) {
            switch (descriptor.kind) {
            case REQ_CONV_LAYER:
            case REQ_ATTN_LAYER:
            case REQ_TOKEN_PASS:
                valid = e->model && submission.conversation_id == e->model->id &&
                        submission.epoch == e->model->id;
                break;
            case REQ_DEPTH_FRAME:
                valid = e->active_depth &&
                        submission.conversation_id == e->active_depth->id &&
                        submission.epoch == e->active_depth->id;
                break;
            default:
                valid = submission.conversation_id == 0 && submission.epoch == 1;
                break;
            }
        }
        if (!valid) {
            publish_rejected(e, submission, -ESTALE);
            continue;
        }

        e->active_status.store(0, std::memory_order_relaxed);
        e->cur_req = (int)descriptor.kind;
        e->active_submission = submission;
        e->bridge_dispatches.fetch_add(1, std::memory_order_relaxed);
        uint64_t generation = e->lane_gen.load(std::memory_order_relaxed) + 1;
        e->lane_gen.store(generation, std::memory_order_release);
        e->dispatch_wakes.fetch_add(1, std::memory_order_relaxed);
        signal_all(&e->dispatch_word);
    }
}

static int submit_pass(Engine *e, int request, uint64_t context_id = 0) {
    uint64_t sequence = ++e->submit_sequence;
    if (sequence == 0) sequence = ++e->submit_sequence;
    uint32_t ticket_generation = ++e->ticket_generation;
    if (ticket_generation == 0) ticket_generation = ++e->ticket_generation;

    LfmKernelDescriptorSpecV1 descriptor_spec = {
        .size = sizeof(LfmKernelDescriptorSpecV1),
        .abi_version = KC_COORD_ABI_VERSION,
        .kind = (uint32_t)request,
        .flags = 0,
        .payload = e,
        .context = nullptr,
        .release = nullptr,
        .reserved = {0, 0, 0},
    };
    KcDescriptorIdV1 descriptor{};
    int rc = lfm_kernel_bridge_descriptor_create(e->bridge, &descriptor_spec, &descriptor);
    if (rc != 0) return rc;

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
    submission.pass_budget = 1;

    KcCompletionV1 completion{};
    rc = lfm_kernel_bridge_submit(e->bridge, &submission);
    if (rc == 0) {
        e->pass_submissions.fetch_add(1, std::memory_order_relaxed);
        rc = lfm_kernel_bridge_wait_completion(e->bridge, &completion, 0);
    }
    int release_rc = lfm_kernel_bridge_descriptor_release(e->bridge, descriptor);
    if (release_rc != 0) std::abort();
    if (rc != 0) return rc;

    if (!ticket_equal(completion.ticket, submission.ticket) ||
        completion.conversation_id != submission.conversation_id ||
        completion.epoch != submission.epoch) {
        return -ESTALE;
    }
    return completion.status;
}

} // namespace

// ---- the C ABI ------------------------------------------------------------------------
extern "C" {

void lfm_engine_free(void *ep);

// `workers` is the total fixed lane count. Every logical lane owns one pthread for
// the engine lifetime; one mechanical bridge dispatcher owns SQ consumption.
void *lfm_engine_new(int workers) {
    if (workers < 1) workers = 1;
    if (workers > MAX_WORKERS) workers = MAX_WORKERS;
    Engine *e = new (std::nothrow) Engine();
    if (!e) return nullptr;
    e->runtime_epoch = next_engine_epoch.fetch_add(1, std::memory_order_acq_rel);
    if (e->runtime_epoch == 0)
        e->runtime_epoch = next_engine_epoch.fetch_add(1, std::memory_order_acq_rel);
    e->lanes_total = (uint32_t)workers;
    e->n_workers = workers;
    if (!kc_atomic_u32_is_lock_free(&e->dispatch_word.value) ||
        kc_port_wait_u32_prepare(&e->dispatch_word.value, &e->dispatch_word.wait) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->wait_words_prepared++;
    if (!kc_atomic_u32_is_lock_free(&e->fence_word.value) ||
        kc_port_wait_u32_prepare(&e->fence_word.value, &e->fence_word.wait) != 0) {
        lfm_engine_free(e);
        return nullptr;
    }
    e->wait_words_prepared++;

    LfmKernelBridgeConfigV1 bridge_config = {
        .size = sizeof(LfmKernelBridgeConfigV1),
        .abi_version = KC_COORD_ABI_VERSION,
        .capacity = 1,
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
    for (int lane = 0; lane < workers; ++lane) {
        e->largs[lane].e = e;
        e->largs[lane].lane = (uint32_t)lane;
        if (pthread_create(&e->workers[lane], nullptr, lane_main, &e->largs[lane]) != 0) {
            lfm_engine_free(e);
            return nullptr;
        }
        e->workers_started++;
    }
    return e;
}

void lfm_engine_request_stop(void *ep) {
    Engine *e = (Engine *)ep;
    if (e && e->bridge) lfm_kernel_bridge_request_stop(e->bridge);
}

uint32_t lfm_engine_lanes(void *ep) {
    Engine *e = (Engine *)ep;
    return e ? e->lanes_total : 0;
}

int lfm_engine_snapshot(void *ep, LfmEngineSnapshotV1 *out) {
    Engine *e = (Engine *)ep;
    if (!e || !out || out->size < sizeof(*out) || out->abi_version != 1) return -EINVAL;
    LfmKernelDescriptorSnapshotV1 descriptors = {
        .size = sizeof(LfmKernelDescriptorSnapshotV1),
        .abi_version = KC_COORD_ABI_VERSION,
    };
    if (lfm_kernel_bridge_descriptor_snapshot(e->bridge, &descriptors) != 0) return -EFAULT;
    *out = {
        .size = sizeof(*out),
        .abi_version = 1,
        .pass_submissions = e->pass_submissions.load(std::memory_order_relaxed),
        .pass_completions = e->pass_completions.load(std::memory_order_relaxed),
        .bridge_dispatches = e->bridge_dispatches.load(std::memory_order_relaxed),
        .dispatch_wakes = e->dispatch_wakes.load(std::memory_order_relaxed),
        .fence_wake_calls = e->fence_wake_calls.load(std::memory_order_relaxed),
        .fence_wakes = e->fence_wakes.load(std::memory_order_relaxed),
        .fence_generations = e->fence.gen.load(std::memory_order_acquire),
        .descriptor_acquires = descriptors.acquired,
        .descriptor_retains = descriptors.retained,
        .descriptor_releases = descriptors.released,
        .descriptor_callbacks = descriptors.callbacks,
        .descriptor_capacity = descriptors.capacity,
        .descriptors_live = descriptors.live,
        .max_descriptor_generation = descriptors.max_generation,
        .attention_qkv_capacity = (uint32_t)e->at_qkvf.size(),
        .attention_y_capacity = (uint32_t)e->at_y.size(),
        .attention_score_capacity = (uint32_t)e->at_att.size(),
        .pass_claimed = e->pass_claimed.load(std::memory_order_acquire) ? 1u : 0u,
    };
    return 0;
}

void lfm_engine_free(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    if (e->bridge) lfm_kernel_bridge_request_stop(e->bridge);
    if (e->bridge_started > 0) pthread_join(e->bridge_worker, nullptr);
    e->retire.store(true, std::memory_order_release);
    e->lane_gen.fetch_add(1, std::memory_order_release);
    if (e->wait_words_prepared > 0) signal_all(&e->dispatch_word);
    for (int lane = 0; lane < e->workers_started; ++lane) {
        pthread_join(e->workers[lane], nullptr);
    }
    if (e->bridge && lfm_kernel_bridge_destroy(e->bridge) != 0) std::abort();
    if (e->wait_words_prepared > 1) kc_port_wait_u32_release(e->fence_word.wait);
    if (e->wait_words_prepared > 0) kc_port_wait_u32_release(e->dispatch_word.wait);
    delete e;
}

// One fused-MLP decode block, entirely native: request slot → doorbell → park.
// Blocking; single pass in flight (decode is sequential). Returns 0 on success.
int lfm_engine_mlp(void *ep, const uint16_t *x, const uint16_t *norm_w,
                   const uint16_t *w1, const uint16_t *w3, const uint16_t *w2,
                   uint16_t *out, size_t h, size_t i, float eps, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || !x || !norm_w || !w1 || !w3 || !w2 || !out || h == 0 || i == 0)
        return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    size_t tiles = lanes;
    if (tiles < 1) tiles = 1;
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
        grow_f(e->sc_partials, tiles);
        grow_u(e->sc_xn, h);
        grow_f(e->sc_gu, 2 * i);
        grow_u(e->sc_t, i);
    } catch (const std::bad_alloc &) {
        return -2;
    }

    Pass *p = &e->pass;
    p->x = x;
    p->norm_w = norm_w;
    p->w1 = w1;
    p->w3 = w3;
    p->w2 = w2;
    p->out = out;
    p->h = h;
    p->i = i;
    p->eps = eps;
    p->tiles = tiles;
    p->partials = e->sc_partials.data();
    p->xn = e->sc_xn.data();
    p->gu = e->sc_gu.data();
    p->t = e->sc_t.data();
    p->rs_bits.store(0, std::memory_order_relaxed);

    return submit_pass(e, REQ_MLP);
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
    int valid = lfm_prng_fill_u64(state, nullptr, 0);
    if (valid != 0) return valid;
    e->prng.state = state;
    e->prng.out = out;
    e->prng.count = count;
    return submit_pass(e, REQ_PRNG);
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
    try {
        if (e->sample_weights.size() < count) e->sample_weights.resize(count);
        if (e->sample_heap.size() < count) e->sample_heap.resize(count);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    e->sample = {
        .logits = logits,
        .count = count,
        .dtype = dtype,
        .config = *config,
        .state = state,
        .out = out_token,
    };
    return submit_pass(e, REQ_SAMPLE);
}

static bool depth_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

int lfm_engine_bf16_gemm_f32(
    void *ep, const uint16_t *a, size_t a_count,
    const uint16_t *rhs, size_t rhs_count,
    float *out, size_t out_count,
    size_t m, size_t n, size_t k, uint32_t rhs_layout) {
    Engine *e = (Engine *)ep;
    if (!e || !a || !rhs || !out || m == 0 || n == 0 || k == 0 ||
        m > INT_MAX || n > INT_MAX || k > INT_MAX ||
        (rhs_layout != LFM_GEMM_RHS_KN && rhs_layout != LFM_GEMM_RHS_NK))
        return -EINVAL;
#ifdef __APPLE__
    if (m == 1 && !lfm_bf16_gemm_available()) return -ENOTSUP;
#else
    if (!lfm_bf16_gemm_available()) return -ENOTSUP;
#endif

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
    bool use_amx = false;
#ifdef __APPLE__
    if (m > 1) {
        try {
            e->gemm_amx_a.resize(a_need);
            e->gemm_amx_rhs.resize(rhs_need);
        } catch (const std::bad_alloc &) {
            return -ENOMEM;
        }
        use_amx = true;
    }
#endif
    e->gemm = {
        .a = a,
        .rhs = rhs,
        .amx_a = use_amx ? e->gemm_amx_a.data() : nullptr,
        .amx_rhs = use_amx ? e->gemm_amx_rhs.data() : nullptr,
        .out = out,
        .m = m,
        .n = n,
        .k = k,
        .rhs_layout = rhs_layout,
        .use_amx = use_amx,
    };
    return submit_pass(e, REQ_GEMM);
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
    try {
        if (e->fft_twiddle_size != fft_size) {
            e->fft_twiddles.resize(fft_size / 2);
            constexpr double pi = 3.141592653589793238462643383279502884;
            for (size_t i = 0; i < fft_size / 2; ++i) {
                const double angle = -2.0 * pi * (double)i / (double)fft_size;
                e->fft_twiddles[i] = {dd_from_f64(std::cos(angle)),
                                      dd_from_f64(std::sin(angle))};
            }
            e->fft_twiddle_size = fft_size;
        }
        if (e->fft_work.size() < fft_size) e->fft_work.resize(fft_size);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    e->fft_conv_dd = {
        .input = input,
        .kernel = kernel,
        .skip = skip,
        .out = out,
        .batch = batch,
        .channels = channels,
        .steps = steps,
        .fft_size = fft_size,
    };
    return submit_pass(e, REQ_FFT_CONV_DD);
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
    try {
        if (e->irfft_twiddle_size != fft_size) {
            e->irfft_twiddles.resize(fft_size);
            constexpr double pi = 3.141592653589793238462643383279502884;
            for (size_t i = 0; i < fft_size; ++i) {
                const double angle = 2.0 * pi * (double)i / (double)fft_size;
                e->irfft_twiddles[i] = {dd_from_f64(std::cos(angle)),
                                        dd_from_f64(std::sin(angle))};
            }
            e->irfft_twiddle_size = fft_size;
        }
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    e->irfft_dd = {
        .real = real,
        .imag = imag,
        .out = out,
        .rows = rows,
        .fft_size = fft_size,
        .scale = {scale_hi, scale_lo},
    };
    return submit_pass(e, REQ_IRFFT_DD);
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
    e->depthwise_stream = {
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
    return submit_pass(e, REQ_DEPTHWISE_STREAM);
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
    PassClaim claim(e);
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
        next->x.resize(dim);
        next->h.resize(dim);
        next->xn.resize(dim);
        next->qkv_f.resize(qkv_rows);
        next->qkv_b.resize(qkv_rows);
        next->up_f.resize(ffn);
        next->y_b.resize(plane);
        next->q_f.resize((size_t)plan->heads * hd);
        next->attn_f.resize(dim);
        next->attn_b.resize(dim);
        next->proj_f.resize(plane);
        next->t_b.resize(ffn);
        next->k_plane.resize(cache_count);
        next->v_plane.resize(cache_count);
        next->logits_b.resize(vocab_max);
        next->din_b.resize(projection_rows);
        next->df_b.resize(dim);
        if (e->sample_weights.size() < vocab_max) e->sample_weights.resize(vocab_max);
        if (e->sample_heap.size() < vocab_max) e->sample_heap.resize(vocab_max);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }

    next->depth_linear_w = depth_u16(plan->depth_linear_w);
    next->depth_linear_b = depth_u16(plan->depth_linear_b);
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
    DepthPlan *depth = nullptr;
    for (const std::unique_ptr<DepthPlan> &candidate : e->depth_plans)
        if (candidate->id == id) {
            depth = candidate.get();
            break;
        }
    if (!depth) return -ESTALE;
    if (hidden_count != depth->backbone_dim || out_token_count != depth->codebooks)
        return -EINVAL;
    e->active_depth = depth;
    e->depth_req = {
        .hidden = hidden,
        .sampler = *sampler,
        .sample_state = sample_state,
        .out_tokens = out_tokens,
    };
    return submit_pass(e, REQ_DEPTH_FRAME, id);
}

int lfm_engine_depth_clear(void *ep, uint64_t id) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0) return -EINVAL;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    const auto found = std::find_if(
        e->depth_plans.begin(), e->depth_plans.end(),
        [id](const std::unique_ptr<DepthPlan> &candidate) { return candidate->id == id; });
    if (found == e->depth_plans.end()) return 0;
    if (e->active_depth == found->get()) e->active_depth = nullptr;
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
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    size_t kmax = 1;
    size_t qkv_max = 0, y_max = 0, att_max = 0;
    for (size_t l = 0; l < n_layers; ++l) {
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
            qkv_max = std::max(qkv_max, qkv);
            y_max = std::max(y_max, y);
            att_max = std::max(att_max, att);
        }
    }
    std::unique_ptr<BackbonePlan> next(new (std::nothrow) BackbonePlan());
    if (!next) return -ENOMEM;
    try {
        next->layers.assign(descs, descs + n_layers);
        const auto grow = [](auto &values, size_t count) {
            if (values.size() < count) values.resize(count);
        };
        grow(e->sc_partials, MAX_WORKERS);
        grow(e->sc_xn, h);
        grow(e->sc_gu, 2 * ffn);
        grow(e->sc_t, ffn);
        grow(e->sc_bcxf, 3 * h);
        grow(e->sc_bcxb, 3 * h);
        grow(e->sc_conv, h * kmax);
        grow(e->sc_projf, h);
        grow(e->sc_projb, h);
        grow(e->sc_stage, h);
        grow(e->sc_mid, h);
        if (qkv_max > 0) {
            grow(e->at_qkvf, qkv_max);
            grow(e->at_qkvb, qkv_max);
            grow(e->at_y, y_max);
            grow(e->at_att, att_max);
        }
        grow(e->tk_h0, h);
        grow(e->tk_h1, h);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    next->h = h;
    next->ffn = ffn;
    next->max_ctx = max_ctx;
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
int lfm_ctx_set_heads(void *ep, uint64_t id, const uint16_t *embed_w,
                      size_t embed_len, size_t vocab, const uint16_t *audio_embed_w,
                      size_t audio_embed_len, size_t audio_rows,
                      const uint16_t *emb_norm_w, size_t emb_norm_len,
                      float emb_norm_eps) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !embed_w || !emb_norm_w || vocab == 0) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    BackbonePlan *model = find_model(e, id);
    if (!model) return -3;
    if (vocab > SIZE_MAX / model->h || embed_len < vocab * model->h ||
        emb_norm_len < model->h)
        return -1;
    if (audio_rows > 0 &&
        (!audio_embed_w || audio_rows > SIZE_MAX / model->h ||
         audio_embed_len < audio_rows * model->h))
        return -1;
    try {
        if (e->tk_logf.size() < vocab) e->tk_logf.resize(vocab);
        if (e->sample_weights.size() < vocab) e->sample_weights.resize(vocab);
        if (e->sample_heap.size() < vocab) e->sample_heap.resize(vocab);
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
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    const auto found = std::find_if(
        e->models.begin(), e->models.end(),
        [id](const std::unique_ptr<BackbonePlan> &model) { return model->id == id; });
    if (found == e->models.end()) return 0;
    if (e->model == found->get()) e->model = nullptr;
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
    if (!e || id == 0 || !x || !state_in || !state_out || !out) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
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

    e->model = model;
    e->conv.layer = layer;
    e->conv.x = x;
    e->conv.state_in = state_in;
    e->conv.state_out = state_out;
    e->conv.out = out;
    e->conv.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    return submit_pass(e, REQ_CONV_LAYER, id);
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
    if (!e || id == 0 || !x || !k_plane || !v_plane || !cos_base || !sin_base || !out)
        return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    BackbonePlan *model = find_model(e, id);
    if (!model || layer >= model->layers.size() || model->layers[layer].kind != 1 ||
        !model->layers[layer].q_w || pos + 1 > model->max_ctx)
        return -3;
    const LfmLayerDesc *d = &model->layers[layer];
    if (x_len != model->h || out_len != model->h || d->hd == 0 ||
        pos + 1 > SIZE_MAX / d->hd || head_stride < (pos + 1) * d->hd ||
        d->n_kv > SIZE_MAX / head_stride || k_len < d->n_kv * head_stride ||
        v_len < d->n_kv * head_stride || pos + 1 > SIZE_MAX / (d->hd / 2) ||
        rope_len < (pos + 1) * (d->hd / 2))
        return -1;

    e->model = model;
    e->attn.layer = layer;
    e->attn.x = x;
    e->attn.k_plane = k_plane;
    e->attn.v_plane = v_plane;
    e->attn.head_stride = head_stride;
    e->attn.pos = pos;
    e->attn.cos_base = cos_base;
    e->attn.sin_base = sin_base;
    e->attn.out = out;
    e->attn.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    return submit_pass(e, REQ_ATTN_LAYER, id);
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
    if (!e || id == 0 || !ids || n_ids == 0 || !states || !out_hidden) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
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
            if (hd == 0 || pos + 1 > SIZE_MAX / hd ||
                states[l].head_stride < (pos + 1) * hd ||
                nkv > SIZE_MAX / states[l].head_stride ||
                states[l].k_len < nkv * states[l].head_stride ||
                states[l].v_len < nkv * states[l].head_stride ||
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

    e->model = model;
    e->tok.ids = ids;
    e->tok.n_ids = n_ids;
    e->tok.embed_kind = embed_kind;
    e->tok.provided_embed = provided_embed;
    e->tok.states = states;
    e->tok.n_states = n_states;
    e->tok.pos = pos;
    e->tok.cos_base = cos_base;
    e->tok.sin_base = sin_base;
    e->tok.out_hidden = out_hidden;
    e->tok.out_logits = out_logits;
    e->tok.sampler = sampler;
    e->tok.sample_state = sample_state;
    e->tok.out_token = out_token;
    e->tok.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    return submit_pass(e, REQ_TOKEN_PASS, id);
}

} // extern "C"
