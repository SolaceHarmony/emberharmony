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
};

struct Stage {
    std::atomic<uint32_t> epoch{0};    // bumped (release) to publish kind/count/chunk
    std::atomic<uint32_t> next{0};     // tile index — workers fetch_add it dry
    std::atomic<uint32_t> remaining{0}; // participating workers still draining
    uint32_t kind = ST_IDLE;           // written before the epoch bump
    uint32_t count = 0;                // number of tiles this stage
    uint32_t chunk = 0;                // band width for GATEUP/DOWN
};

enum : int { REQ_NONE = 0, REQ_MLP = 1, REQ_CONV_LAYER = 2, REQ_SHUTDOWN = -1 };

// ---- the resident layer table (C ABI) ------------------------------------------------
// One entry per backbone block, indexed by block_idx. Pointers are PtrLen-style
// captures into the model's Arc-stable weight storages, built ONCE at load
// (lfm_ctx_build) and cleared before the model's weights drop (lfm_ctx_clear via the
// Rust-side guard). Rung 1 serves conv layers (kind 0); attention slots (kind 1) are
// placeholders until rung 2.
extern "C" {
struct LfmConvLayerDesc {
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
    ScPass sc;     // shortconv stage pointers

    // Resident layer table + dims (lfm_ctx_build); cleared before model drop.
    std::vector<LfmConvLayerDesc> layers;
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
    const LfmConvLayerDesc *d = &e->layers[r->layer];
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
    m->xn = e->sc_xn.data(); // xn reuse after the conv block is done with it
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
    try {
        e->sc_partials.resize(tiles);
        e->sc_xn.resize(h);
        e->sc_gu.resize(2 * i);
        e->sc_t.resize(i);
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
int lfm_ctx_build(void *ep, const LfmConvLayerDesc *descs, size_t n_layers, size_t h,
                  size_t ffn) {
    Engine *e = (Engine *)ep;
    if (!e || !descs || n_layers == 0 || h == 0 || ffn == 0) return -1;
    size_t kmax = 1;
    for (size_t l = 0; l < n_layers; ++l) {
        if (descs[l].kind == 0) {
            if (!descs[l].op_norm_w || !descs[l].ffn_norm_w || !descs[l].in_w ||
                !descs[l].conv_w || !descs[l].out_w || !descs[l].w1 || !descs[l].w3 ||
                !descs[l].w2 || descs[l].k < 1 || descs[l].k > 8)
                return -1;
            if (descs[l].k > kmax) kmax = descs[l].k;
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
    } catch (const std::bad_alloc &) {
        e->layers.clear();
        return -2;
    }
    e->dim_h = h;
    e->dim_ffn = ffn;
    e->dim_kmax = kmax;
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

} // extern "C"
