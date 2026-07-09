// flashkern_engine.cpp — the resident native decode engine (ENGINE_DESIGN.md §2/§3),
// as a RESIDENT STAGE MACHINE: the engine owns all mutable state; the schedule is an
// epoch plus an atomic tile index; workers pull indices, never messages. No channels,
// no descriptor staging, no malloc, no done tokens anywhere in the hot loop — the only
// runtime primitives are kcoro park/unpark, made sound by vendored patches 0001 (the
// three-state park gate: an unpark racing a park parks as a NOTIFIED token, never
// lost) and 0004 (unpark enqueues to the coroutine's OWNING scheduler, so both the
// Rust rim's request doorbell and the last worker's stage-done doorbell are legal from
// any context).
//
// Shape per pass: the Rust rim writes the request slot and unparks the coordinator
// (ONE doorbell). The coordinator publishes a stage — kind/count/chunk stores, tile
// counter to zero, remaining = workers, epoch bump (release) — and unparks the team;
// workers race the tile counter dry (fetch_add work-stealing), the last one out
// unparks the coordinator; repeat for the next stage; at the pass boundary the
// coordinator signals the rim's condvar. Stop/shutdown is observed at pass boundaries
// only — never polled inside ops.
//
// Numerics: stage bodies are line-for-line ports of src/flashkern/decode.rs
// (fused_mlp_decode) — same RNE bf16 rounding ladder, same FIXED tile count and
// fixed-order partial fold (deterministic regardless of which worker runs which
// tile), same kernels (lfm_bf16_gemm_nt_f32, linked in-image). The Rust parity test
// pins this bit-identical to the threadgroup port, itself pinned to the candle chain.
//
// Build: -ffp-contract=off (the ladders promise separate roundings), C++17.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <pthread.h>
#include <vector>

extern "C" {
#include "kcoro.h"
#include "kcoro_dispatch.h"
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
};

struct Stage {
    std::atomic<uint32_t> epoch{0};    // bumped (release) to publish kind/count/chunk
    std::atomic<uint32_t> next{0};     // tile index — workers fetch_add it dry
    std::atomic<uint32_t> remaining{0}; // participating workers still draining
    uint32_t kind = ST_IDLE;           // written before the epoch bump
    uint32_t count = 0;                // number of tiles this stage
    uint32_t chunk = 0;                // band width for GATEUP/DOWN
};

enum : int {
    REQ_NONE = 0,
    REQ_MLP = 1,
    REQ_CONV_LAYER = 2,
    REQ_ATTN_LAYER = 3,
    REQ_SHUTDOWN = -1
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

struct Engine {
    Pass pass;
    Stage stage;

    kcoro_t *coord = nullptr;
    kcoro_t *workers[MAX_WORKERS] = {};
    int n_workers = 0;
    kc_dispatcher_t *disp = nullptr;
    std::atomic<bool> retire{false}; // workers exit when set (observed while idle)

    // Rust-rim handshake: request slot + doorbell in, condvar back.
    std::atomic<int> req{REQ_NONE};
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
    int finished = 0;

    ConvReq conv;  // conv-layer request payload
    AttnReq attn;  // attention-layer request payload
    ScPass sc;     // shortconv stage pointers
    AtPass at;     // attention stage pointers

    // Resident layer table + dims (lfm_ctx_build); cleared before model drop.
    std::vector<LfmLayerDesc> layers;
    size_t dim_h = 0, dim_ffn = 0, dim_kmax = 0;
    std::atomic<bool> ctx_live{false};

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
    size_t dim_maxctx = 0, dim_nh = 0, dim_nkv = 0, dim_hd = 0;
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
    default:
        break;
    }
}

// ---- workers -----------------------------------------------------------------------------
static void worker_main(void *arg) {
    Engine *e = (Engine *)arg;
    uint32_t seen = 0;
    for (;;) {
        uint32_t ep = e->stage.epoch.load(std::memory_order_acquire);
        if (ep == seen) {
            if (e->retire.load(std::memory_order_acquire)) return;
            // Idle: park. The coordinator's epoch-bump + unpark wakes us; the park
            // gate turns an unpark that lands before this park into an immediate
            // (spurious-safe) return.
            kcoro_park();
            continue;
        }
        seen = ep;
        const uint32_t kind = e->stage.kind;
        const uint32_t count = e->stage.count;
        uint32_t idx;
        while ((idx = e->stage.next.fetch_add(1, std::memory_order_acq_rel)) < count) {
            run_tile(kind, idx, &e->stage, e);
        }
        // Last participant out closes the stage and rings the coordinator's doorbell.
        if (e->stage.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            kcoro_unpark(e->coord);
        }
    }
}

// ---- coordinator ---------------------------------------------------------------------------
static void publish_stage(Engine *e, uint32_t kind, uint32_t count, uint32_t chunk) {
    e->stage.kind = kind;
    e->stage.count = count;
    e->stage.chunk = chunk;
    e->stage.next.store(0, std::memory_order_relaxed);
    e->stage.remaining.store((uint32_t)e->n_workers, std::memory_order_relaxed);
    e->stage.epoch.fetch_add(1, std::memory_order_release);
    for (int w = 0; w < e->n_workers; ++w) kcoro_unpark(e->workers[w]);
}

static void wait_stage_done(Engine *e) {
    while (e->stage.remaining.load(std::memory_order_acquire) != 0) kcoro_park();
}

static void run_mlp(Engine *e) {
    Pass *p = &e->pass;
    const uint32_t tiles = (uint32_t)p->tiles;

    publish_stage(e, ST_SUMSQ, tiles, 0);
    wait_stage_done(e);

    // Serial fold in fixed tile order — matches the reference exactly.
    float total = 0.f;
    for (uint32_t l = 0; l < tiles; ++l) total += p->partials[l];
    float rs = 1.0f / std::sqrt(total / (float)p->h + p->eps);
    uint32_t rsb;
    std::memcpy(&rsb, &rs, 4);
    p->rs_bits.store(rsb, std::memory_order_release);

    publish_stage(e, ST_NORM, tiles, 0);
    wait_stage_done(e);

    uint32_t i_chunk = (uint32_t)((p->i + tiles - 1) / tiles);
    publish_stage(e, ST_GATEUP, (uint32_t)((p->i + i_chunk - 1) / i_chunk), i_chunk);
    wait_stage_done(e);

    uint32_t h_chunk = (uint32_t)((p->h + tiles - 1) / tiles);
    if (h_chunk > DOWN_BAND_CAP) h_chunk = DOWN_BAND_CAP;
    publish_stage(e, ST_DOWN, (uint32_t)((p->h + h_chunk - 1) / h_chunk), h_chunk);
    wait_stage_done(e);
}

// One whole shortconv+MLP layer. Stage bodies are decode.rs::fused_shortconv_decode
// ported verbatim: candle-order sumsq computed serially here (the reference computes it
// once on lane 0), banded elementwise/GEMV stages on the board, the tiny conv update
// serial here (reference: lane 0), then the MLP block on the layer's ffn weights with
// the conv output as its input — all without leaving the engine.
static void run_conv_layer(Engine *e) {
    const ConvReq *r = &e->conv;
    const LfmLayerDesc *d = &e->layers[r->layer];
    ScPass *c = &e->sc;
    const size_t h = e->dim_h;

    // Wire the shortconv stage pointers for this pass.
    c->x = r->x;
    c->norm_w = d->op_norm_w;
    c->in_w = d->in_w;
    c->out_w = d->out_w;
    c->state_out = r->state_out;
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

    size_t lanes = r->lanes < 1 ? 1 : r->lanes;
    uint32_t sc_tiles = (uint32_t)(lanes > h ? h : lanes);

    // Stage 1 (serial, candle's exact reduction — reference runs it once on lane 0).
    float total = lfm_bf16_sumsq_candle_f32(r->x, (int)h);
    float inv_rms = 1.0f / std::sqrt(total / (float)h + d->op_eps);
    uint32_t rsb;
    std::memcpy(&rsb, &inv_rms, 4);
    c->rs_bits.store(rsb, std::memory_order_release);

    uint32_t hc = (uint32_t)((h + sc_tiles - 1) / sc_tiles);
    publish_stage(e, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc);
    wait_stage_done(e);

    uint32_t pc = (uint32_t)((3 * h + sc_tiles - 1) / sc_tiles);
    publish_stage(e, ST_SC_INPROJ, (uint32_t)((3 * h + pc - 1) / pc), pc);
    wait_stage_done(e);

    // Conv update (serial — ~0.1% of the block; reference: lane 0).
    lfm_conv1d_update_bf16(c->bcxb, r->state_in, d->conv_w, c->conv, 1, (int)h, 1,
                           (int)d->k);

    publish_stage(e, ST_SC_GATHER, (uint32_t)((h + hc - 1) / hc), hc);
    wait_stage_done(e);

    publish_stage(e, ST_SC_OUTPROJ, (uint32_t)((h + hc - 1) / hc), hc);
    wait_stage_done(e);

    // MLP block on the layer's ffn weights: input = mid, output = the request's out.
    Pass *m = &e->pass;
    size_t cap = h < e->dim_ffn ? h : e->dim_ffn;
    m->x = c->mid;
    m->norm_w = d->ffn_norm_w;
    m->w1 = d->w1;
    m->w3 = d->w3;
    m->w2 = d->w2;
    m->out = r->out;
    m->h = h;
    m->i = e->dim_ffn;
    m->eps = d->ffn_eps;
    m->tiles = lanes > cap ? cap : lanes;
    m->partials = e->sc_partials.data();
    // xn reuse: SEQUENTIAL dependency — the MLP block must not start until
    // ST_SC_OUTPROJ drains (wait_stage_done above). Pipelining these blocks would
    // corrupt the plane.
    m->xn = e->sc_xn.data();
    m->gu = e->sc_gu.data();
    m->t = e->sc_t.data();
    m->rs_bits.store(0, std::memory_order_relaxed);
    run_mlp(e);
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

// One whole attention+MLP layer. Stage bodies are the candle wrapper ops and
// attn_decode_bf16 ported at the same rounding points; the serial section (qk-norm,
// rope, KV append) is per-head work two orders of magnitude below the GEMVs.
static void run_attn_layer(Engine *e) {
    const AttnReq *r = &e->attn;
    const LfmLayerDesc *d = &e->layers[r->layer];
    ScPass *c = &e->sc;
    AtPass *a = &e->at;
    const size_t h = e->dim_h;
    const size_t nh = d->n_head, nkv = d->n_kv, hd = d->hd;
    size_t lanes = r->lanes < 1 ? 1 : r->lanes;
    uint32_t tiles = (uint32_t)(lanes > h ? h : lanes);

    // Wire stage pointers. The conv pass planes are reused where shapes allow — a
    // single request is in flight at a time, never both kinds at once.
    c->x = r->x;
    c->norm_w = d->op_norm_w;
    c->h = h;
    c->xn = e->sc_xn.data();
    c->projf = e->sc_projf.data(); // ST_AT_OPROJ reuses the conv pass's proj planes
    c->stage = e->sc_stage.data();
    a->o_w = d->o_w;
    a->qkvf = e->at_qkvf.data();
    a->qkvb = e->at_qkvb.data();
    a->ybits = e->at_y.data();
    a->att = e->at_att.data();
    a->x = r->x;
    a->mid = e->sc_mid.data();
    a->k_plane = r->k_plane;
    a->v_plane = r->v_plane;
    a->head_stride = r->head_stride;
    a->att_len = r->pos + 1;
    a->max_ctx = e->dim_maxctx;
    a->h = h;
    a->n_head = nh;
    a->n_kv = nkv;
    a->hd = hd;

    // operator norm: candle-order sumsq (serial) + banded norm apply.
    float total = lfm_bf16_sumsq_candle_f32(r->x, (int)h);
    float inv_rms = 1.0f / std::sqrt(total / (float)h + d->op_eps);
    uint32_t rsb;
    std::memcpy(&rsb, &inv_rms, 4);
    c->rs_bits.store(rsb, std::memory_order_release);
    uint32_t hc = (uint32_t)((h + tiles - 1) / tiles);
    publish_stage(e, ST_SC_NORM, (uint32_t)((h + hc - 1) / hc), hc);
    wait_stage_done(e);

    // q|k|v projections, banded over the concatenated row space.
    size_t total_rows = (nh + 2 * nkv) * hd;
    uint32_t qc = (uint32_t)((total_rows + tiles - 1) / tiles);
    publish_stage(e, ST_AT_QKV, (uint32_t)((total_rows + qc - 1) / qc), qc);
    wait_stage_done(e);

    // Serial: per-head qk-norm + rope, then append this step's K/V rows at the cursor.
    const uint16_t *cos_row = r->cos_base + r->pos * (hd / 2);
    const uint16_t *sin_row = r->sin_base + r->pos * (hd / 2);
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
        std::memcpy(r->k_plane + kh * r->head_stride + r->pos * hd, krows + kh * hd,
                    hd * sizeof(uint16_t));
        std::memcpy(r->v_plane + kh * r->head_stride + r->pos * hd, vrows + kh * hd,
                    hd * sizeof(uint16_t));
    }

    // Attention: one tile per q head over the (now pos+1)-row planes.
    publish_stage(e, ST_AT_HEAD, (uint32_t)nh, 1);
    wait_stage_done(e);

    // o_proj + residual → mid.
    publish_stage(e, ST_AT_OPROJ, (uint32_t)((h + hc - 1) / hc), hc);
    wait_stage_done(e);

    // MLP block on the layer's ffn weights: input = mid, output = the request's out.
    Pass *m = &e->pass;
    size_t cap = h < e->dim_ffn ? h : e->dim_ffn;
    m->x = a->mid;
    m->norm_w = d->ffn_norm_w;
    m->w1 = d->w1;
    m->w3 = d->w3;
    m->w2 = d->w2;
    m->out = r->out;
    m->h = h;
    m->i = e->dim_ffn;
    m->eps = d->ffn_eps;
    m->tiles = lanes > cap ? cap : lanes;
    m->partials = e->sc_partials.data();
    m->xn = e->sc_xn.data();
    m->gu = e->sc_gu.data();
    m->t = e->sc_t.data();
    m->rs_bits.store(0, std::memory_order_relaxed);
    run_mlp(e);
}

static void coord_main(void *arg) {
    Engine *e = (Engine *)arg;
    for (;;) {
        int req = e->req.exchange(REQ_NONE, std::memory_order_acq_rel);
        if (req == REQ_SHUTDOWN) {
            // Retire the team: flag + wake so parked workers observe it while idle.
            e->retire.store(true, std::memory_order_release);
            for (int w = 0; w < e->n_workers; ++w) kcoro_unpark(e->workers[w]);
            return;
        }
        if (req == REQ_NONE) {
            kcoro_park(); // the Rust rim's doorbell (or a just-written request's
                          // NOTIFIED token) wakes us
            continue;
        }
        if (req == REQ_MLP) run_mlp(e);
        if (req == REQ_CONV_LAYER) run_conv_layer(e);
        if (req == REQ_ATTN_LAYER) run_attn_layer(e);
        // Pass boundary: hand back (signal from coroutine context never blocks).
        pthread_mutex_lock(&e->mu);
        e->finished = 1;
        pthread_cond_signal(&e->cv);
        pthread_mutex_unlock(&e->mu);
    }
}

} // namespace

// ---- the C ABI (the Rust rim) ---------------------------------------------------------
extern "C" {

void lfm_engine_free(void *ep);

void *lfm_engine_new(int workers) {
    if (workers < 1) workers = 1;
    if (workers > MAX_WORKERS) workers = MAX_WORKERS;
    Engine *e = new (std::nothrow) Engine();
    if (!e) return nullptr;
    e->n_workers = workers;
    e->disp = kc_dispatcher_new(workers + 1); // +1 lane so the coordinator never
                                              // starves the tile team
    if (!e->disp) {
        delete e;
        return nullptr;
    }
    for (int w = 0; w < workers; ++w) {
        if (kc_dispatcher_spawn_co(e->disp, worker_main, e, 128 * 1024,
                                   &e->workers[w]) != 0 ||
            !e->workers[w]) {
            lfm_engine_free(e);
            return nullptr;
        }
    }
    if (kc_dispatcher_spawn_co(e->disp, coord_main, e, 256 * 1024, &e->coord) != 0 ||
        !e->coord) {
        lfm_engine_free(e);
        return nullptr;
    }
    return e;
}

void lfm_engine_free(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    if (e->coord) {
        e->req.store(REQ_SHUTDOWN, std::memory_order_release);
        kcoro_unpark(e->coord);
    } else {
        // Coordinator never came up: retire workers directly.
        e->retire.store(true, std::memory_order_release);
        for (int w = 0; w < e->n_workers; ++w)
            if (e->workers[w]) kcoro_unpark(e->workers[w]);
    }
    if (e->disp) kc_dispatcher_release(e->disp); // joins the team's threads
    // Release the caller-owned coroutine handle refs (spawn_co's out_co retains for
    // us). Safe strictly after dispatcher release: threads are joined and the
    // scheduler's own queue refs are dropped (kcoro_destroy == kcoro_release —
    // refcounted, so this is the ref that lets the stacks actually unmap).
    for (int w = 0; w < e->n_workers; ++w)
        if (e->workers[w]) kcoro_release(e->workers[w]);
    if (e->coord) kcoro_release(e->coord);
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

    pthread_mutex_lock(&e->mu);
    e->finished = 0;
    pthread_mutex_unlock(&e->mu);

    e->req.store(REQ_MLP, std::memory_order_release);
    kcoro_unpark(e->coord); // the doorbell (patch 0004: legal from this thread)

    pthread_mutex_lock(&e->mu);
    while (!e->finished) pthread_cond_wait(&e->cv, &e->mu);
    pthread_mutex_unlock(&e->mu);
    return 0;
}

// Build the resident layer table: one descriptor per backbone block (indexed by
// block_idx), plus the model dims. Sizes ALL pass scratch here — fixed-arena
// discipline: after a successful build, conv-layer passes allocate nothing.
// The Rust rim serializes this against passes (pass_lock); pointers must stay valid
// until lfm_ctx_clear (the model-side guard guarantees clear-before-drop).
int lfm_ctx_build(void *ep, const LfmLayerDesc *descs, size_t n_layers, size_t h,
                  size_t ffn, size_t max_ctx) {
    Engine *e = (Engine *)ep;
    if (!e || !descs || n_layers == 0 || h == 0 || ffn == 0 || max_ctx == 0) return -1;
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
    e->ctx_live.store(true, std::memory_order_release);
    return 0;
}

// Clear the table (weight pointers are about to die with the model). Serialized by the
// Rust rim's pass lock, so no pass is in flight here.
void lfm_ctx_clear(void *ep) {
    Engine *e = (Engine *)ep;
    if (!e) return;
    e->ctx_live.store(false, std::memory_order_release);
    e->layers.clear();
}

// One whole shortconv+MLP layer: request slot → doorbell → park. Returns 0 on
// success; -3 when no ctx is live or the slot is not a conv layer (caller takes the
// bit-identical per-block path).
int lfm_engine_conv_layer(void *ep, size_t layer, const uint16_t *x,
                          const uint16_t *state_in, uint16_t *state_out, uint16_t *out,
                          size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || !x || !state_in || !state_out || !out) return -1;
    if (!e->ctx_live.load(std::memory_order_acquire) || layer >= e->layers.size() ||
        e->layers[layer].kind != 0)
        return -3;

    e->conv.layer = layer;
    e->conv.x = x;
    e->conv.state_in = state_in;
    e->conv.state_out = state_out;
    e->conv.out = out;
    e->conv.lanes = lanes < 1 ? 1 : (lanes > MAX_WORKERS ? MAX_WORKERS : lanes);

    pthread_mutex_lock(&e->mu);
    e->finished = 0;
    pthread_mutex_unlock(&e->mu);

    e->req.store(REQ_CONV_LAYER, std::memory_order_release);
    kcoro_unpark(e->coord);

    pthread_mutex_lock(&e->mu);
    while (!e->finished) pthread_cond_wait(&e->cv, &e->mu);
    pthread_mutex_unlock(&e->mu);
    return 0;
}

// One whole attention+MLP layer. Per-generation state (planes, rope tables, cursor)
// arrives per request; the engine appends the step's K/V rows at `pos` and attends
// over pos+1 entries. Rows beyond `pos` must already fit the planes (the caller
// pre-grows capacity BEFORE capturing the plane pointers). Returns 0 on success;
// -3 when unserved (no ctx / not an attention slot / capture was null / pos over cap).
int lfm_engine_attn_layer(void *ep, size_t layer, const uint16_t *x, uint16_t *k_plane,
                          uint16_t *v_plane, size_t head_stride, size_t pos,
                          const uint16_t *cos_base, const uint16_t *sin_base,
                          uint16_t *out, size_t lanes) {
    Engine *e = (Engine *)ep;
    if (!e || !x || !k_plane || !v_plane || !cos_base || !sin_base || !out) return -1;
    if (!e->ctx_live.load(std::memory_order_acquire) || layer >= e->layers.size() ||
        e->layers[layer].kind != 1 || !e->layers[layer].q_w ||
        pos + 1 > e->dim_maxctx)
        return -3;

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

    pthread_mutex_lock(&e->mu);
    e->finished = 0;
    pthread_mutex_unlock(&e->mu);

    e->req.store(REQ_ATTN_LAYER, std::memory_order_release);
    kcoro_unpark(e->coord);

    pthread_mutex_lock(&e->mu);
    while (!e->finished) pthread_cond_wait(&e->cv, &e->mu);
    pthread_mutex_unlock(&e->mu);
    return 0;
}

} // extern "C"
