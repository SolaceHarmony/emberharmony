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
};

struct Stage {
    std::atomic<uint32_t> epoch{0};    // bumped (release) to publish kind/count/chunk
    std::atomic<uint32_t> next{0};     // tile index — workers fetch_add it dry
    std::atomic<uint32_t> remaining{0}; // participating workers still draining
    uint32_t kind = ST_IDLE;           // written before the epoch bump
    uint32_t count = 0;                // number of tiles this stage
    uint32_t chunk = 0;                // band width for GATEUP/DOWN
};

enum : int { REQ_NONE = 0, REQ_MLP = 1, REQ_SHUTDOWN = -1 };

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

    // Persistent scratch backing, grown outside execution (arena discipline).
    std::vector<float> sc_partials, sc_gu;
    std::vector<uint16_t> sc_xn, sc_t;
};

// ---- tile bodies (identical math to decode.rs) ----------------------------------------
static void run_tile(uint32_t kind, uint32_t idx, const Stage *st, Pass *p) {
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
            run_tile(kind, idx, &e->stage, &e->pass);
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

    // Grow the persistent scratch outside execution (no allocation once warm).
    e->sc_partials.resize(tiles);
    e->sc_xn.resize(h);
    e->sc_gu.resize(2 * i);
    e->sc_t.resize(i);

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

} // extern "C"
