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
// The compatibility Rust ABI still invokes one blocking control call so its borrowed
// tensor pointers remain live. C++ claims the preallocated request slot, then invokes
// the registered Rust coordinator. That coordinator alone owns SQ submission and CQ
// ingress; the callback resolves only after the exact completion arrives. Stop remains
// a full-pass boundary decision and is never polled inside SIMD operations.
//
// Numerics: stage bodies are line-for-line ports of src/compute/flashkern/decode.rs
// (fused_mlp_decode) — same RNE bf16 rounding ladder, same FIXED tile count and
// fixed-order partial fold (deterministic regardless of which worker runs which
// tile), same kernels (lfm_bf16_gemm_nt_f32, linked in-image). The Rust parity test
// pins this bit-identical to the threadgroup port, itself pinned to the candle chain.
//
// Build: -ffp-contract=off (the ladders promise separate roundings), C++23.

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <pthread.h>
#include <vector>

#include "lfm_kernel_bridge.h"

extern "C" {
#include "kc_atomic.h"
#include "kc_port.h"
}

// Stage kernels from the flashkern TU (same image, plain calls).
extern "C" void lfm_bf16_gemm_nt_f32(const uint16_t *A, const uint16_t *W, float *C,
                                     int M, int N, int K);
extern "C" float lfm_bf16_sumsq_candle_f32(const uint16_t *x, int n);
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
    // Generic lane-uniform call: every lane runs fn(ctx, lane, lanes_total). The
    // program may use lfm_lane_fence: each logical lane remains on one pthread, so
    // ordinary nested C++/Rust frames and thread-local state never migrate. This
    // transitional call remains until the depthformer program is fully native.
    REQ_CALL = 5,
};

typedef void (*LfmLaneFn)(void *ctx, uint32_t lane, uint32_t lanes_total);
struct CallReq {
    LfmLaneFn fn = nullptr;
    void *ctx = nullptr;
};

// ---- the resident layer table (C ABI) ------------------------------------------------
// One entry per backbone block, indexed by block_idx. Pointers are PtrLen-style
// captures into the model's Arc-stable weight storages, built ONCE at load
// (lfm_ctx_build) and cleared before the model's weights drop (lfm_ctx_clear via the
// Rust-side guard). Rung 1 serves conv layers (kind 0); attention slots (kind 1) are
// placeholders until rung 2.
extern "C" {
struct LfmLayerDesc {
    uint32_t kind; // 0 = shortconv+mlp, 1 = attention (unserved this rung)
    uint32_t k;    // conv kernel size
    float op_eps;
    float ffn_eps;
    const uint16_t *op_norm_w;  // [H]
    const uint16_t *ffn_norm_w; // [H]
    const uint16_t *in_w;       // [3H, H] (B|C|x row order)
    const uint16_t *conv_w;     // [H, K]
    const uint16_t *out_w;      // [H, H]
    const uint16_t *w1;         // [I, H]
    const uint16_t *w3;         // [I, H]
    const uint16_t *w2;         // [H, I]
    // Attention fields (kind 1). q_w == NULL means "attention not served for this
    // slot" (capture failed at install): conv layers still run; attn requests bail.
    uint32_t n_head;
    uint32_t n_kv;
    uint32_t hd;
    float qk_eps;
    const uint16_t *q_w;  // [nh·hd, H]
    const uint16_t *k_w;  // [nkv·hd, H]
    const uint16_t *v_w;  // [nkv·hd, H]
    const uint16_t *o_w;  // [H, nh·hd]
    const uint16_t *qn_w; // [hd] per-head q RmsNorm
    const uint16_t *kn_w; // [hd]
};
}

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

// Per-layer per-generation state for the token pass (planes live in the per-cache
// objects; pointers are captured fresh each token AFTER capacity is ensured).
extern "C" {
struct LfmLayerState {
    uint16_t *k_plane; // attention layers; null for conv
    uint16_t *v_plane;
    size_t head_stride;
    size_t k_len;
    size_t v_len;
    uint16_t *conv_state; // conv layers: carried window, advanced IN PLACE; null for attn
    size_t conv_len;
};
}

// Token-pass request: ONE doorbell per token — embed → every layer → final norm →
// logits. Sampling stays at the rim (RNG-stream parity).
struct TokenReq {
    const uint32_t *ids = nullptr; // text: 1 id; audio: n pre-offset audio-embed rows
    size_t n_ids = 0;
    uint32_t embed_kind = 0; // 0 = text table, 1 = audio table (sum of rows)
    const LfmLayerState *states = nullptr;
    size_t n_states = 0;
    size_t pos = 0;
    const uint16_t *cos_base = nullptr;
    const uint16_t *sin_base = nullptr;
    uint16_t *out_hidden = nullptr; // [H] post embedding-norm bits
    float *out_logits = nullptr;    // [vocab] f32 (bf16-rounded then widened — the
                                    // linear_logits ladder)
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
    LfmKernelSubmitFn submitter = nullptr;
    void *submitter_context = nullptr;
    KcSubmissionV1 active_submission{};
    std::atomic<bool> pass_claimed{false};
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
    CallReq call;  // generic lane-uniform call payload
    ScPass sc;     // shortconv stage pointers
    AtPass at;     // attention stage pointers

    // Resident layer table + dims (lfm_ctx_build); cleared before model drop.
    std::vector<LfmLayerDesc> layers;
    // Head tables (lfm_ctx_set_heads): embed / audio-embed / final norm / tied logits.
    const uint16_t *embed_w = nullptr;      // [vocab, H]
    const uint16_t *audio_embed_w = nullptr; // [audio_rows, H]
    const uint16_t *emb_norm_w = nullptr;   // [H]
    float emb_norm_eps = 0.f;
    size_t vocab = 0, audio_rows = 0;
    TokenReq tok; // token-pass request payload
    size_t dim_h = 0, dim_ffn = 0, dim_kmax = 0;
    std::atomic<bool> ctx_live{false};
    // Single-tenant ctx ownership: a build claims the slot and mints an id; only the
    // matching clear releases it. A second model's install FAILS (that model keeps
    // its bit-identical candle path) instead of silently replacing a live model's
    // table — and a stale guard's drop can never clear the current owner's install.
    uint64_t ctx_id = 0;
    uint64_t ctx_seq = 0;

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
    size_t dim_maxctx = 0, dim_nh = 0, dim_nkv = 0, dim_hd = 0;
};

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
        float sum = 0.f;
        for (size_t j = idx; j < p->h; j += p->tiles) {
            float v = bf16_f32(p->x[j]);
            sum += v * v;
        }
        p->partials[idx] = sum;
        break;
    }
    case ST_NORM: {
        uint32_t rsb = p->rs_bits.load(std::memory_order_acquire);
        float rs;
        std::memcpy(&rs, &rsb, 4);
        for (size_t j = idx; j < p->h; j += p->tiles) {
            float v = bf16_f32(p->x[j]) * rs * bf16_f32(p->norm_w[j]);
            p->xn[j] = rb_bits(v);
        }
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
        for (size_t r = r0; r < r1; ++r) {
            float g = bf16_f32(rb_bits(p->gu[r]));            // linear-out round
            uint16_t sg = rb_bits(g / (1.0f + std::exp(-g))); // silu round
            uint16_t u = rb_bits(p->gu[p->i + r]);            // linear-out round
            p->t[r] = rb_bits(bf16_f32(sg) * bf16_f32(u));    // gating-mul round
        }
        break;
    }
    case ST_DOWN: {
        size_t r0 = (size_t)idx * st->chunk;
        size_t r1 = r0 + st->chunk < p->h ? r0 + st->chunk : p->h;
        if (r1 <= r0) break;
        size_t n = r1 - r0;
        float y[DOWN_BAND_CAP]; // per-worker accumulator; chunk capped at publish
        lfm_bf16_gemm_nt_f32(p->t, p->w2 + r0 * p->i, y, 1, (int)n, (int)p->i);
        for (size_t j = 0; j < n; ++j) {
            float d = bf16_f32(rb_bits(y[j]));                     // linear-out round
            p->out[r0 + j] = rb_bits(d + bf16_f32(p->x[r0 + j])); // residual round
        }
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
        const LfmLayerDesc *d = &e->layers[e->attn.layer];
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
        float scale = 1.0f / std::sqrt((float)a->hd);
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
        size_t r1 = r0 + st->chunk < ee->vocab ? r0 + st->chunk : ee->vocab;
        if (r1 <= r0) break;
        float *acc = ee->tk_logf.data() + r0;
        lfm_bf16_gemm_nt_f32(t->out_hidden, ee->embed_w + r0 * ee->dim_h, acc, 1,
                             (int)(r1 - r0), (int)ee->dim_h);
        for (size_t r = r0; r < r1; ++r) {
            t->out_logits[r] = ee->tk_logf[r];
        }
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
        float total = 0.f;
        for (uint32_t l = 0; l < tiles; ++l) total += p->partials[l];
        float rs = 1.0f / std::sqrt(total / (float)p->h + p->eps);
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
    const size_t h = e->dim_h;
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
        float inv_rms = 1.0f / std::sqrt(total / (float)h + d->op_eps);
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
    size_t cap = h < e->dim_ffn ? h : e->dim_ffn;
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
        m->i = e->dim_ffn;
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
    float inv = 1.0f / std::sqrt(total / (float)hd + eps);
    for (size_t j = 0; j < hd; ++j) {
        out[j] = rb_bits(bf16_f32(x[j]) * inv * bf16_f32(w[j]));
    }
}

// candle rotary_emb::rope_slow over one head row, NeoX half-split, computed in bf16
// exactly as the tensor ops do: cos2 = [cos|cos], out = x⊙cos2 + rotate_half(x)⊙sin2,
// where every bf16 multiply and the add each round once (half-crate semantics:
// f32 compute, RNE back to bf16). rotate_half = [-x2 | x1]; negation is exact.
static void rope_slow_row(uint16_t *x, const uint16_t *cos_row, const uint16_t *sin_row,
                          size_t hd) {
    size_t half = hd / 2;
    // In-place needs the original bits of both halves for the cross terms.
    uint16_t orig[512];
    std::memcpy(orig, x, hd * sizeof(uint16_t));
    for (size_t j = 0; j < half; ++j) {
        float c = bf16_f32(cos_row[j]);
        float sn = bf16_f32(sin_row[j]);
        // j < half: rotate_half[j] = -x[j+half]
        float p1 = bf16_f32(rb_bits(bf16_f32(orig[j]) * c));
        float p2 = bf16_f32(rb_bits(-bf16_f32(orig[j + half]) * sn));
        x[j] = rb_bits(p1 + p2);
        // j + half: cos2/sin2 reuse row [j]; rotate_half[j+half] = x[j]
        float q1 = bf16_f32(rb_bits(bf16_f32(orig[j + half]) * c));
        float q2 = bf16_f32(rb_bits(bf16_f32(orig[j]) * sn));
        x[j + half] = rb_bits(q1 + q2);
    }
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
    const LfmLayerDesc *d = &e->layers[layer_idx];
    ScPass *c = &e->sc;
    AtPass *a = &e->at;
    const size_t h = e->dim_h;
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
        a->max_ctx = e->dim_maxctx;
        a->h = h;
        a->n_head = nh;
        a->n_kv = nkv;
        a->hd = hd;
        // operator norm: candle-order sumsq, serial.
        float total = lfm_bf16_sumsq_candle_f32(x, (int)h);
        float inv_rms = 1.0f / std::sqrt(total / (float)h + d->op_eps);
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
    size_t cap = h < e->dim_ffn ? h : e->dim_ffn;
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
        m->i = e->dim_ffn;
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
    const size_t h = e->dim_h;
    size_t lanes = t->lanes < 1 ? 1 : t->lanes;
    uint16_t *h0 = e->tk_h0.data();
    uint16_t *h1 = e->tk_h1.data();

    lane_fence(e, lane, [&] {
        // Embed (serial — at most 8 rows). Text: one table row copied verbatim.
        // Audio: candle's `.sum(0)` over the gathered rows — sequential bf16 adds
        // from a bf16 zero, one RNE round per step (candle's in-dtype reduction).
        if (t->embed_kind == 0) {
            std::memcpy(h0, e->embed_w + (size_t)t->ids[0] * h, h * sizeof(uint16_t));
        } else {
            for (size_t j = 0; j < h; ++j) h0[j] = 0;
            for (size_t c = 0; c < t->n_ids; ++c) {
                const uint16_t *row = e->audio_embed_w + (size_t)t->ids[c] * h;
                for (size_t j = 0; j < h; ++j) {
                    h0[j] = rb_bits(bf16_f32(h0[j]) + bf16_f32(row[j]));
                }
            }
        }
    });

    // The layer walk. x = h0, out = h1, swap — no Tensor, no Rust, no copies.
    for (size_t l = 0; l < e->layers.size(); ++l) {
        const LfmLayerDesc *d = &e->layers[l];
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
        float inv_rms = 1.0f / std::sqrt(total / (float)h + e->emb_norm_eps);
        uint32_t rsb;
        std::memcpy(&rsb, &inv_rms, 4);
        c->rs_bits.store(rsb, std::memory_order_release);
        c->x = h0;
        c->norm_w = e->emb_norm_w;
        c->h = h;
        c->xn = t->out_hidden;
    });

    // Tied logits head over the normed hidden — the heavy stage; real row bands.
    if (t->out_logits && e->vocab > 0) {
        uint32_t vc = (uint32_t)((e->vocab + (size_t)e->n_workers * 4 - 1) /
                                 ((size_t)e->n_workers * 4));
        run_stage(e, lane, ST_LOGITS, (uint32_t)((e->vocab + vc - 1) / vc), vc, [] {});
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
        run_conv_block(e, lane, &e->layers[r->layer], r->x, r->state_in, r->state_out,
                       r->out, r->lanes);
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
    case REQ_CALL:
        e->call.fn(e->call.ctx, lane, e->lanes_total);
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
            completion.execution = KC_COORD_EXECUTION_COMPLETED;
            completion.state = KC_COORD_STATE_COMMITTED;
            completion.publication = KC_COORD_PUBLICATION_COMMITTED;
            completion.cause = KC_COORD_CAUSE_SUCCESS;
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
// Policy and recurrence remain above this boundary.
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
                     descriptor.kind > REQ_NONE && descriptor.kind <= REQ_CALL;
        if (!valid) {
            publish_rejected(e, submission, -ESTALE);
            continue;
        }

        e->cur_req = (int)descriptor.kind;
        e->active_submission = submission;
        e->bridge_dispatches.fetch_add(1, std::memory_order_relaxed);
        uint64_t generation = e->lane_gen.load(std::memory_order_relaxed) + 1;
        e->lane_gen.store(generation, std::memory_order_release);
        e->dispatch_wakes.fetch_add(1, std::memory_order_relaxed);
        signal_all(&e->dispatch_word);
    }
}

static int submit_pass(Engine *e, int request) {
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
    submission.conversation_id = e->ctx_id;
    submission.epoch = e->ctx_id == 0 ? 1 : e->ctx_id;
    submission.descriptor = descriptor;
    submission.command = KC_COORD_COMMAND_RUN_PASS;
    submission.service_class = KC_COORD_SERVICE_INTERACTIVE;
    submission.pass_budget = 1;

    KcCompletionV1 completion{};
    rc = e->submitter ? e->submitter(e->submitter_context, &submission, &completion)
                      : -ENOTCONN;
    int release_rc = lfm_kernel_bridge_descriptor_release(e->bridge, descriptor);
    if (release_rc != 0) std::abort();
    if (rc != 0) return rc;
    e->pass_submissions.fetch_add(1, std::memory_order_relaxed);

    if (!ticket_equal(completion.ticket, submission.ticket) ||
        completion.conversation_id != submission.conversation_id ||
        completion.epoch != submission.epoch) {
        return -ESTALE;
    }
    return completion.status;
}

} // namespace

// ---- the C ABI (the Rust rim) ---------------------------------------------------------
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

void *lfm_engine_bridge(void *ep) {
    Engine *e = (Engine *)ep;
    return e ? e->bridge : nullptr;
}

int lfm_engine_set_submitter(void *ep, LfmKernelSubmitFn submitter, void *context) {
    Engine *e = (Engine *)ep;
    if (!e || !submitter || !context) return -EINVAL;
    if (e->pass_claimed.load(std::memory_order_acquire) || e->submitter) return -EBUSY;
    e->submitter_context = context;
    e->submitter = submitter;
    return 0;
}

int lfm_engine_clear_submitter(void *ep, void *context) {
    Engine *e = (Engine *)ep;
    if (!e || !context) return -EINVAL;
    if (e->pass_claimed.load(std::memory_order_acquire)) return -EBUSY;
    if (e->submitter_context != context) return -ESTALE;
    e->submitter = nullptr;
    e->submitter_context = nullptr;
    return 0;
}

void lfm_engine_request_stop(void *ep) {
    Engine *e = (Engine *)ep;
    if (e && e->bridge) lfm_kernel_bridge_request_stop(e->bridge);
}

// Run a caller-supplied lane-uniform program on the whole team: fn(ctx, lane, lanes)
// on every lane. Logical lanes are thread-stable, so ordinary language TLS remains
// on its originating pthread. One ticket in, one exact completion out.
int lfm_engine_call(void *ep, LfmLaneFn fn, void *ctx) {
    Engine *e = (Engine *)ep;
    if (!e || !fn) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;

    e->call.fn = fn;
    e->call.ctx = ctx;
    return submit_pass(e, REQ_CALL);
}

// The team fence, exported for REQ_CALL programs: pure barrier (empty serial
// section). Callable ONLY from within a lane program on this engine's team.
void lfm_lane_fence(void *ep, uint32_t lane) {
    Engine *e = (Engine *)ep;
    lane_fence(e, lane, [] {});
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

// Build the resident layer table: one descriptor per backbone block (indexed by
// block_idx), plus the model dims. Sizes ALL pass scratch here — fixed-arena
// discipline: after a successful build, conv-layer passes allocate nothing.
// The Rust rim serializes this against passes (pass_lock); pointers must stay valid
// until lfm_ctx_clear (the model-side guard guarantees clear-before-drop).
// SINGLE-TENANT: fails with -4 while another install is live; the winning install's
// id lands in *out_id and is the only key that clears it.
int lfm_ctx_build(void *ep, const LfmLayerDesc *descs, size_t n_layers, size_t h,
                  size_t ffn, size_t max_ctx, uint64_t *out_id) {
    Engine *e = (Engine *)ep;
    if (!e || !descs || n_layers == 0 || h == 0 || ffn == 0 || max_ctx == 0 || !out_id)
        return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    if (e->ctx_live.load(std::memory_order_acquire)) return -4;
    size_t kmax = 1, nh = 0, nkv = 0, hd = 0;
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
            nh = descs[l].n_head;
            nkv = descs[l].n_kv;
            hd = descs[l].hd;
        }
    }
    try {
        e->layers.assign(descs, descs + n_layers);
        e->sc_partials.resize(MAX_WORKERS);
        e->sc_xn.resize(h);
        e->sc_gu.resize(2 * ffn);
        e->sc_t.resize(ffn);
        e->sc_bcxf.resize(3 * h);
        e->sc_bcxb.resize(3 * h);
        e->sc_conv.resize(h * kmax);
        e->sc_projf.resize(h);
        e->sc_projb.resize(h);
        e->sc_stage.resize(h);
        e->sc_mid.resize(h);
        if (nh > 0) {
            e->at_qkvf.resize((nh + 2 * nkv) * hd);
            e->at_qkvb.resize((nh + 2 * nkv) * hd);
            e->at_y.resize(nh * hd);
            e->at_att.resize(nh * max_ctx);
        }
        e->tk_h0.resize(h);
        e->tk_h1.resize(h);
    } catch (const std::bad_alloc &) {
        e->layers.clear();
        return -2;
    }
    e->dim_h = h;
    e->dim_ffn = ffn;
    e->dim_kmax = kmax;
    e->dim_maxctx = max_ctx;
    e->dim_nh = nh;
    e->dim_nkv = nkv;
    e->dim_hd = hd;
    e->ctx_id = ++e->ctx_seq;
    *out_id = e->ctx_id;
    e->ctx_live.store(true, std::memory_order_release);
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
    if (!e->ctx_live.load(std::memory_order_acquire) || id != e->ctx_id) return -3;
    if (vocab > SIZE_MAX / e->dim_h || embed_len < vocab * e->dim_h ||
        emb_norm_len < e->dim_h)
        return -1;
    if (audio_rows > 0 &&
        (!audio_embed_w || audio_rows > SIZE_MAX / e->dim_h ||
         audio_embed_len < audio_rows * e->dim_h))
        return -1;
    try {
        e->tk_logf.resize(vocab);
    } catch (const std::bad_alloc &) {
        return -2;
    }
    e->embed_w = embed_w;
    e->vocab = vocab;
    e->audio_embed_w = audio_embed_w;
    e->audio_rows = audio_rows;
    e->emb_norm_w = emb_norm_w;
    e->emb_norm_eps = emb_norm_eps;
    return 0;
}

// Clear the table (weight pointers are about to die with the model). Serialized by the
// Rust rim's pass lock, so no pass is in flight here. Only the owning install's id
// clears — a stale guard (its build was refused, or it was already superseded) is a
// no-op instead of clobbering the live owner's table.
int lfm_ctx_clear(void *ep, uint64_t id) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    if (id != e->ctx_id) return 0;
    e->ctx_id = 0;
    e->ctx_live.store(false, std::memory_order_release);
    e->layers.clear();
    e->embed_w = nullptr;
    e->audio_embed_w = nullptr;
    e->emb_norm_w = nullptr;
    e->vocab = 0;
    e->audio_rows = 0;
    return 0;
}

// One whole shortconv+MLP layer: request slot → doorbell → park. Returns 0 on
// success; -3 when no ctx is live or the slot is not a conv layer (caller takes the
// bit-identical per-block path).
int lfm_engine_conv_layer(void *ep, uint64_t id, size_t layer, const uint16_t *x,
                          size_t x_len, const uint16_t *state_in, size_t state_in_len,
                          uint16_t *state_out, size_t state_out_len, uint16_t *out,
                          size_t out_len, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !x || !state_in || !state_out || !out) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    if (!e->ctx_live.load(std::memory_order_acquire) || id != e->ctx_id ||
        layer >= e->layers.size() || e->layers[layer].kind != 0)
        return -3;
    const size_t k = e->layers[layer].k;
    const size_t tail = k > 0 ? k - 1 : 0;
    if (k < 1 || (tail > 0 && e->dim_h > SIZE_MAX / tail)) return -1;
    const size_t state_len = e->dim_h * tail;
    if (x_len != e->dim_h || out_len != e->dim_h || state_in_len != state_len ||
        state_out_len != state_len)
        return -1;

    e->conv.layer = layer;
    e->conv.x = x;
    e->conv.state_in = state_in;
    e->conv.state_out = state_out;
    e->conv.out = out;
    e->conv.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    return submit_pass(e, REQ_CONV_LAYER);
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
    if (!e->ctx_live.load(std::memory_order_acquire) || id != e->ctx_id ||
        layer >= e->layers.size() || e->layers[layer].kind != 1 ||
        !e->layers[layer].q_w ||
        pos + 1 > e->dim_maxctx)
        return -3;
    const LfmLayerDesc *d = &e->layers[layer];
    if (x_len != e->dim_h || out_len != e->dim_h || d->hd == 0 ||
        pos + 1 > SIZE_MAX / d->hd || head_stride < (pos + 1) * d->hd ||
        d->n_kv > SIZE_MAX / head_stride || k_len < d->n_kv * head_stride ||
        v_len < d->n_kv * head_stride || pos + 1 > SIZE_MAX / (d->hd / 2) ||
        rope_len < (pos + 1) * (d->hd / 2))
        return -1;

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

    return submit_pass(e, REQ_ATTN_LAYER);
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
                          float *out_logits, size_t out_logits_len, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || id == 0 || !ids || n_ids == 0 || !states || !out_hidden) return -1;
    PassClaim claim(e);
    if (!claim) return -EBUSY;
    if (!e->ctx_live.load(std::memory_order_acquire) || id != e->ctx_id ||
        !e->embed_w || !e->emb_norm_w || n_states != e->layers.size() ||
        pos + 1 > e->dim_maxctx)
        return -3;
    if (out_hidden_len != e->dim_h ||
        (out_logits && out_logits_len < e->vocab) ||
        (!out_logits && out_logits_len != 0))
        return -1;
    if (embed_kind == 0) {
        if (ids[0] >= e->vocab) return -3;
    } else {
        if (!e->audio_embed_w || n_ids > 8) return -3;
        for (size_t c = 0; c < n_ids; ++c)
            if (ids[c] >= e->audio_rows) return -3;
    }
    // Every attention slot must be served and carry planes; conv slots need state.
    for (size_t l = 0; l < e->layers.size(); ++l) {
        if (e->layers[l].kind == 1) {
            if (!e->layers[l].q_w || !states[l].k_plane || !states[l].v_plane ||
                !cos_base || !sin_base)
                return -3;
            const size_t hd = e->layers[l].hd;
            const size_t nkv = e->layers[l].n_kv;
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
            const size_t k = e->layers[l].k;
            const size_t tail = k > 0 ? k - 1 : 0;
            if (k < 1 || (tail > 0 && e->dim_h > SIZE_MAX / tail) ||
                states[l].conv_len < e->dim_h * tail)
                return -1;
        }
    }

    e->tok.ids = ids;
    e->tok.n_ids = n_ids;
    e->tok.embed_kind = embed_kind;
    e->tok.states = states;
    e->tok.n_states = n_states;
    e->tok.pos = pos;
    e->tok.cos_base = cos_base;
    e->tok.sin_base = sin_base;
    e->tok.out_hidden = out_hidden;
    e->tok.out_logits = out_logits;
    e->tok.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    return submit_pass(e, REQ_TOKEN_PASS);
}

} // extern "C"
