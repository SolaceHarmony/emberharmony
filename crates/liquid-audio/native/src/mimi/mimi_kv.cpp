// mimi_kv.cpp — Unit #5 of the Mimi decode port (see docs/MIMI_PORT.md).
// Faithful C++ port of moshi 0.6.4 src/kv_cache.rs:
//   ScatteredCacheBuilder (positions/indices bookkeeping, reset,
//   indices_and_mask + indices_and_mask_abs + get_mask_abs) and
//   ScatteredKvCache (append/scatter into the ring, k()/v() exposure),
//   plus the IndicesAndMask product the attention consumes.
// Specialized to batch = 1, but with FAITHFUL semantics: every index/mask
// value reproduces the Rust bit-for-bit (the mask is exact 0.0f / -inf; the
// ring arithmetic is the same modular walk). Verified against ALL SIX asserts
// in kv_cache.rs's test_scattered_kv_cache (see the selftest at the bottom,
// which replays the batch-0 trajectory of that test).
//
// IMPORTANT ARCHITECTURE NOTE (read the NOTES block, section (e)): the ACTUAL
// Mimi decoder (mimi.rs decode_step -> transformer::Transformer, transformer.rs)
// uses candle_nn RotatingKvCache, NOT this ScatteredKvCache. ScatteredKvCache
// is consumed only by batched_transformer.rs. The two are batch=1-equivalent
// (rotating ring + additive causal mask), but the arbiter must decide which
// cache unit #4 links. This file is the faithful port of the unit I was
// assigned (kv_cache.rs); the reconciliation is documented for the arbiter.
//
// ABI: mimi_kernel.h (arbiter-owned — code against it, do NOT edit it). This
// unit PROPOSES its own entry points (kv_cache has no slot in the header yet);
// exact signatures + the unit-#4 reconciliation are in the NOTES block.

#include "mimi_kernel.h"

#include <cmath>    // -INFINITY
#include <cstdint>
#include <cstdio>   // snprintf
#include <cstring>  // memcpy, memset

// This unit is pure control plane: modular index arithmetic, an additive mask
// fill, and a memcpy scatter. There is NO float reduction / GEMM / softmax
// here, so there is nothing to NEON-vectorize and no scalar-vs-NEON parity
// split to keep (unlike units 1-4). The mask values are produced exactly.

// ---------------------------------------------------------------------------
// Compile-time shape facts for this checkpoint (mirror mimi_kernel.h enums).
// ---------------------------------------------------------------------------
namespace {
// context is fixed at 250 for Mimi v0_1(8); the builder embeds an all_pos
// scratch array of this bound so the state stays fully POD (no interior
// pointer -> hibernation-friendly). Runtime context must be <= this.
constexpr int   kMaxContext = MIMI_TR_CONTEXT;  // 250
const     float kNegInf     = -INFINITY;        // f32::NEG_INFINITY

// Rust uses usize::MAX as the "empty ring slot" sentinel in all_pos; any
// sentinel strictly greater than every reachable absolute position works,
// because the mask test is `stored_pos <= my_pos`. INT64_MAX is never <= a
// real position (positions are bounded by the stream length).
constexpr int64_t kEmptySlot = INT64_MAX;
}  // namespace

// ---------------------------------------------------------------------------
// State structs (POD, carved from the arena at init; steady state never
// allocates). Two structs mirror the Rust split, which is LOAD-BEARING for
// correctness with 8 layers (see NOTES (b)/(d)):
//   - ScatteredCacheBuilder  -> MimiKvBuilder : ONE per transformer, shared by
//        all 8 layers. Owns positions/indices; indices_and_mask advances it
//        ONCE per step. The mask + indices are identical for every layer.
//   - ScatteredKvCache        -> MimiKvCache   : ONE per layer (8 total). Owns
//        the k/v ring; append scatters this step's k/v into the ring.
// Collapsing them into one per-layer state would advance the position counter
// 8x per step. So the split is faithful AND required.
// ---------------------------------------------------------------------------

// ScatteredCacheBuilder, batch=1 (positions[0], indices[0] -> scalars).
struct MimiKvBuilder {
    int     context;                 // == self.context (<= kMaxContext)
    int64_t position;                // positions[0]: stream pos, may exceed context
    int     index;                   // indices[0]: ring write ptr, 0..context-1
    // all_pos[slot] = absolute stream position currently stored in ring slot,
    // or kEmptySlot. Reconstructed fresh each indices_and_mask call, exactly
    // as the Rust `all_pos` local. Embedded (not arena) to keep POD/hibernation.
    int64_t all_pos[kMaxContext];
};

// ScatteredKvCache, batch=1. k/v rings, shape [heads, context, head_dim]
// row-major = candle (b=1, h, context, d) contiguous.
struct MimiKvCache {
    int    heads;
    int    head_dim;
    int    context;
    float *k_ring;  // [heads * context * head_dim]
    float *v_ring;  // [heads * context * head_dim]
};

// ===========================================================================
// Entry points.
// ===========================================================================
extern "C" {

// ---- Builder (ScatteredCacheBuilder) --------------------------------------

// ScatteredCacheBuilder::new (batch_size fixed to 1). Carves the builder from
// the arena, positions/indices zeroed. Returns 0 on success.
int mimi_kv_builder_init(MimiKvBuilder **st, MimiArena *a, int context,
                         char *err, size_t errlen) {
    if (context <= 0 || context > kMaxContext) {
        if (err && errlen)
            snprintf(err, errlen, "mimi_kv: context %d out of (0, %d]", context,
                     kMaxContext);
        return 1;
    }
    MimiKvBuilder *b =
        static_cast<MimiKvBuilder *>(mimi_arena_alloc(a, sizeof(MimiKvBuilder)));
    b->context  = context;
    b->position = 0;
    b->index    = 0;
    for (int i = 0; i < context; ++i) b->all_pos[i] = kEmptySlot;
    *st = b;
    return 0;
}

// ScatteredCacheBuilder::reset. Zeroes positions/indices only (the Rust reset
// does NOT touch the ring; stale ring data is masked out — see NOTES (a)).
void mimi_kv_builder_reset(MimiKvBuilder *b) {
    b->position = 0;
    b->index    = 0;
    // all_pos is rebuilt each call; nothing to reset here, but keep it tidy.
    for (int i = 0; i < b->context; ++i) b->all_pos[i] = kEmptySlot;
}

// ScatteredCacheBuilder::positions()[0]. At batch=1 this scalar is both the
// stream position and the effective current_seq_len. Unit #4 must read this
// BEFORE calling mimi_kv_indices_and_mask for the step to get the rope base
// position (matches transformer.rs current_seq_len, read pre-append). See
// NOTES (e) for the rope-ordering subtlety in batched_transformer.rs.
int64_t mimi_kv_positions(const MimiKvBuilder *b) { return b->position; }

// ScatteredCacheBuilder::indices_and_mask, batch=1.
//   n       : seq_len (# new query positions this step; 1 on the decode hot path)
//   active  : batch_mask[0]. Mimi always passes 1 (single stream, always live);
//             0 reproduces the frozen/all-zero-mask branch for a masked slot.
//   out_indices : [n]        ring slot each new k/v lands in (u32).
//   out_mask    : [n * klen] additive f32 mask, row-major [query, key], the
//                 tensor the attention broadcast_adds before softmax.
// Returns klen = the key (context) dimension of the mask / returned ring:
//   ring path (n <  context): klen == context.
//   abs  path (n >= context): klen == n (see mimi_kv_append abs branch).
// ADVANCES the builder's position/index by n on the active path (once/step).
int mimi_kv_indices_and_mask(MimiKvBuilder *b, int n, int active,
                             uint32_t *out_indices, float *out_mask) {
    const int context = b->context;

    // Rust: `if self.context <= seq_len { return indices_and_mask_abs(...) }`.
    if (context <= n) {
        // get_mask_abs(n, n): causal band mask, width = n. Computed once,
        // independent of `active` (Rust builds it outside the batch loop).
        //   mask[i][j] = (j > i || (i - j) > context) ? -inf : 0
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                bool masked = (j > i) || ((i - j) > context);
                out_mask[(size_t)i * n + j] = masked ? kNegInf : 0.0f;
            }
        }
        if (active) {
            // Cycle the ring pointer and advance the stream, recording the
            // slot each token would occupy (the ring itself is NOT written in
            // abs mode; append returns the raw k/v — see mimi_kv_append).
            for (int s = 0; s < n; ++s) {
                out_indices[s] = (uint32_t)b->index;
                b->index += 1;
                b->position += 1;
                if (b->index >= context) b->index = 0;
            }
        } else {
            // Inactive: frozen index repeated, no advance.
            for (int s = 0; s < n; ++s) out_indices[s] = (uint32_t)b->index;
        }
        return n;  // klen == seq_len for the abs path
    }

    // ---- Ring path (context > n): the decode hot path (n == 1). ----
    if (!active) {
        // Rust `!batch_mask` branch: all-zero mask over `context`, index frozen
        // and repeated, no bookkeeping advance.
        for (int s = 0; s < n; ++s) {
            out_indices[s] = (uint32_t)b->index;
            for (int j = 0; j < context; ++j) out_mask[(size_t)s * context + j] = 0.0f;
        }
        return context;
    }

    const int     start_index = b->index;     // ring ptr before this step
    const int64_t start_pos   = b->position;  // stream pos before this step
    int64_t      *all_pos     = b->all_pos;

    // Reconstruct the absolute position held in every ring slot.
    for (int i = 0; i < context; ++i) all_pos[i] = kEmptySlot;
    if (start_pos < (int64_t)context) {
        // Ring not yet wrapped: slots 0..start_pos hold positions 0..start_pos.
        for (int i = 0; i < (int)start_pos; ++i) all_pos[i] = i;
    } else {
        // Wrapped: undo the rotation. offset = start_pos - start_index.
        const int64_t offset = start_pos - (int64_t)start_index;
        for (int i = 0; i < context; ++i) {
            all_pos[i] = (i < start_index) ? (int64_t)i + offset
                                           : (int64_t)i + offset - (int64_t)context;
        }
    }

    // Write this step's n tokens into the ring, recording slots and updating
    // all_pos. Bookkeeping advances here (once per step, shared by all layers).
    // NOTE: all_pos is fully populated BEFORE the mask loop, so within a
    // multi-token chunk earlier queries still see (and correctly mask out as
    // future) the later tokens of the same chunk.
    for (int s = 0; s < n; ++s) {
        const int index = b->index;
        all_pos[index] = (int64_t)s + start_pos;
        out_indices[s] = (uint32_t)index;
        b->index += 1;
        b->position += 1;
        if (b->index >= context) b->index = 0;
    }

    // Additive causal mask: allow slot j for query s iff stored pos <= my_pos.
    for (int s = 0; s < n; ++s) {
        const int64_t my_pos = (int64_t)s + start_pos;
        float *row = out_mask + (size_t)s * context;
        for (int j = 0; j < context; ++j) {
            row[j] = (all_pos[j] <= my_pos) ? 0.0f : kNegInf;
        }
    }
    return context;  // klen == context for the ring path
}

// ---- Cache (ScatteredKvCache), one per layer ------------------------------

// ScatteredCacheBuilder::make_cache. Carves the k/v ring from the arena and
// zeroes it (Tensor::zeros). heads == num_kv (kv_repeat == 1 -> num_heads).
int mimi_kv_init(MimiKvCache **st, MimiArena *a, int heads, int head_dim,
                 int context, char *err, size_t errlen) {
    if (heads <= 0 || head_dim <= 0 || context <= 0 || context > kMaxContext) {
        if (err && errlen)
            snprintf(err, errlen,
                     "mimi_kv: bad cache dims heads=%d head_dim=%d context=%d",
                     heads, head_dim, context);
        return 1;
    }
    MimiKvCache *c =
        static_cast<MimiKvCache *>(mimi_arena_alloc(a, sizeof(MimiKvCache)));
    c->heads    = heads;
    c->head_dim = head_dim;
    c->context  = context;
    const size_t elems = (size_t)heads * context * head_dim;
    c->k_ring = static_cast<float *>(mimi_arena_alloc(a, elems * sizeof(float)));
    c->v_ring = static_cast<float *>(mimi_arena_alloc(a, elems * sizeof(float)));
    memset(c->k_ring, 0, elems * sizeof(float));
    memset(c->v_ring, 0, elems * sizeof(float));
    *st = c;
    return 0;
}

// ScatteredKvCache::append. Scatters this step's k/v into the ring at the
// slots from mimi_kv_indices_and_mask, then hands back the ring pointers the
// attention reads (k(), v()).
//   k_new, v_new : [heads, n, head_dim] row-major = candle (1,h,n,d) contiguous
//   indices      : [n]  ring slots (from mimi_kv_indices_and_mask)
//   out_k, out_v : receive the pointers the attention should q.matmul / .matmul
//   out_klen     : receive the key dimension (== context on the ring path)
// Returns 0 on success.
//
// Faithful detail: Rust append early-returns `(k.clone(), v.clone())` when
// `context <= k.dim(2)` (n >= context, the abs path). In that case NOTHING is
// scattered and the attention reads the RAW new k/v (length n), not the ring.
// The ring is left untouched (and thus stale), exactly as in Rust. Mimi's
// decode hot path is always n == 1 < context, so this branch never fires
// there; it is ported for faithfulness only.
int mimi_kv_append(MimiKvCache *c, const float *k_new, const float *v_new,
                   const uint32_t *indices, int n, const float **out_k,
                   const float **out_v, int *out_klen) {
    const int heads = c->heads, hd = c->head_dim, context = c->context;

    if (context <= n) {
        // Abs mode: no scatter; expose the raw inputs (length n).
        *out_k    = k_new;
        *out_v    = v_new;
        *out_klen = n;
        return 0;
    }

    // scatter_set(indices, k, dim=2): for every head, k_new[h, s, :] lands in
    // ring slot indices[s]. The candle indices tensor is broadcast from
    // (b, seq_len) across heads and head_dim, i.e. the SAME slot for all heads
    // and the whole head_dim vector — a plain per-vector copy.
    const size_t hd_bytes = (size_t)hd * sizeof(float);
    for (int h = 0; h < heads; ++h) {
        const float *ksrc = k_new + (size_t)h * n * hd;
        const float *vsrc = v_new + (size_t)h * n * hd;
        float       *kdst = c->k_ring + (size_t)h * context * hd;
        float       *vdst = c->v_ring + (size_t)h * context * hd;
        for (int s = 0; s < n; ++s) {
            const int slot = (int)indices[s];  // 0..context-1
            memcpy(kdst + (size_t)slot * hd, ksrc + (size_t)s * hd, hd_bytes);
            memcpy(vdst + (size_t)slot * hd, vsrc + (size_t)s * hd, hd_bytes);
        }
    }

    *out_k    = c->k_ring;
    *out_v    = c->v_ring;
    *out_klen = context;
    return 0;
}

// Accessors mirroring ScatteredKvCache::k()/v() — the full ring the attention
// reads (shape [heads, context, head_dim]).
const float *mimi_kv_k(const MimiKvCache *c) { return c->k_ring; }
const float *mimi_kv_v(const MimiKvCache *c) { return c->v_ring; }

// Re-arm the ring. Rust does NOT zero the ring on reset (only the builder is
// reset; the mask hides stale slots). We zero it anyway: the header's reset
// convention is "zero states in place", and it is provably harmless because
// every stale slot is -inf-masked until overwritten. See NOTES (a).
void mimi_kv_reset(MimiKvCache *c) {
    const size_t elems = (size_t)c->heads * c->context * c->head_dim;
    memset(c->k_ring, 0, elems * sizeof(float));
    memset(c->v_ring, 0, elems * sizeof(float));
}

}  // extern "C"

// ===========================================================================
// Selftest — replays the batch-0 trajectory of kv_cache.rs::test_scattered_kv_cache
// (context=5) plus the ring/append/wrap/reset worked examples from the NOTES.
//   clang++ -std=c++17 -O2 -DMIMI_KV_SELFTEST mimi_kv.cpp -o /tmp/mimi_kv_test
//   /tmp/mimi_kv_test
// ===========================================================================
#ifdef MIMI_KV_SELFTEST
#include <cstdlib>  // abort, malloc

// Standalone arena allocator (real build links mimi_decode.cpp's impl).
extern "C" void *mimi_arena_alloc(MimiArena *a, size_t bytes) {
    size_t off = (a->used + 63) & ~((size_t)63);
    if (off + bytes > a->size) {
        std::fprintf(stderr, "arena overflow: need %zu have %zu\n", off + bytes,
                     a->size);
        std::abort();
    }
    void *p = a->base + off;
    a->used = off + bytes;
    return p;
}

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); \
            g_fail = 1;                                                    \
        }                                                                  \
    } while (0)

static bool feq(float a, float b) {
    if (std::isinf(a) || std::isinf(b)) return a == b;  // -inf == -inf
    float d = a - b;
    return (d < 0 ? -d : d) < 1e-6f;
}

// Compare an [n][klen] mask row-major against an expected flat array.
static bool mask_eq(const float *got, const float *exp, int n, int klen) {
    for (int i = 0; i < n * klen; ++i)
        if (!feq(got[i], exp[i])) return false;
    return true;
}

int main() {
    const float I = kNegInf;
    uint8_t arena_buf[1 << 20];
    MimiArena arena{arena_buf, sizeof(arena_buf), 0};
    char err[128];

    // ---- Part 1: replay batch-0 of kv_cache.rs test (context = 5) ----------
    // Calls (batch_mask[0], seq_len): the Rust test's batch-0 view is
    //   (T,1) (T,1) (F,3) (T,3) (T,1) (T,2).
    {
        MimiKvBuilder *b = nullptr;
        int rc = mimi_kv_builder_init(&b, &arena, 5, err, sizeof(err));
        CHECK(rc == 0, "builder_init ctx=5");

        uint32_t idx[8];
        float    mask[8 * 5];
        int      klen;

        // call 1: (T, 1) -> idx [0], mask [[0,-inf,-inf,-inf,-inf]]
        klen = mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        {
            uint32_t ei[] = {0};
            float    em[] = {0, I, I, I, I};
            CHECK(klen == 5, "c1 klen");
            CHECK(idx[0] == ei[0], "c1 idx");
            CHECK(mask_eq(mask, em, 1, 5), "c1 mask");
            CHECK(mimi_kv_positions(b) == 1, "c1 pos");
        }
        // call 2: (T, 1) -> idx [1], mask [[0,0,-inf,-inf,-inf]]
        klen = mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        {
            float em[] = {0, 0, I, I, I};
            CHECK(idx[0] == 1, "c2 idx");
            CHECK(mask_eq(mask, em, 1, 5), "c2 mask");
        }
        // call 3: (F, 3) inactive -> idx [2,2,2], mask all zeros (3x5), no advance
        klen = mimi_kv_indices_and_mask(b, 3, 0, idx, mask);
        {
            float em[15] = {0};
            CHECK(idx[0] == 2 && idx[1] == 2 && idx[2] == 2, "c3 idx frozen");
            CHECK(mask_eq(mask, em, 3, 5), "c3 mask zeros");
            CHECK(mimi_kv_positions(b) == 2, "c3 pos unchanged");
        }
        // call 4: (T, 3) -> idx [2,3,4], causal-fill mask
        klen = mimi_kv_indices_and_mask(b, 3, 1, idx, mask);
        {
            float em[] = {0, 0, 0, I, I,  //
                          0, 0, 0, 0, I,  //
                          0, 0, 0, 0, 0};
            CHECK(idx[0] == 2 && idx[1] == 3 && idx[2] == 4, "c4 idx");
            CHECK(mask_eq(mask, em, 3, 5), "c4 mask");
            CHECK(mimi_kv_positions(b) == 5, "c4 pos");
        }
        // call 5: (T, 1) -> idx [0] (wrapped), mask all zeros (ring full)
        klen = mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        {
            float em[] = {0, 0, 0, 0, 0};
            CHECK(idx[0] == 0, "c5 idx wrap");
            CHECK(mask_eq(mask, em, 1, 5), "c5 mask");
        }
        // call 6: (T, 2) -> idx [1,2], mask [[0,0,-inf,0,0],[0,0,0,0,0]]
        klen = mimi_kv_indices_and_mask(b, 2, 1, idx, mask);
        {
            float em[] = {0, 0, I, 0, 0,  //
                          0, 0, 0, 0, 0};
            CHECK(idx[0] == 1 && idx[1] == 2, "c6 idx");
            CHECK(mask_eq(mask, em, 2, 5), "c6 mask");
        }
        (void)klen;
    }

    // ---- Part 2: NOTES ring/wrap worked examples (context = 250) ------------
    // positions 0, 249, 250, 251 -> slots 0, 249, 0, 1; eviction at wrap.
    {
        MimiKvBuilder *b = nullptr;
        mimi_kv_builder_init(&b, &arena, 250, err, sizeof(err));
        uint32_t idx[1];
        static float mask[250];

        // pos 0 -> slot 0, only slot 0 valid
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(idx[0] == 0, "p0 slot");
        CHECK(feq(mask[0], 0.0f) && feq(mask[1], I), "p0 mask head");

        // advance through positions 1..248 (248 steps) -> position 249
        for (int p = 1; p <= 248; ++p) mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(mimi_kv_positions(b) == 249, "pre-249 pos");

        // pos 249 -> slot 249; ring now full, mask all zeros
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(idx[0] == 249, "p249 slot");
        {
            bool all0 = true;
            for (int j = 0; j < 250; ++j) all0 &= feq(mask[j], 0.0f);
            CHECK(all0, "p249 mask all zero");
        }

        // pos 250 -> slot 0 (wrap, evicts pos 0), mask all zeros
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(idx[0] == 0, "p250 slot wrap");
        {
            bool all0 = true;
            for (int j = 0; j < 250; ++j) all0 &= feq(mask[j], 0.0f);
            CHECK(all0, "p250 mask all zero");
        }

        // pos 251 -> slot 1 (evicts pos 1)
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(idx[0] == 1, "p251 slot");
        CHECK(mimi_kv_positions(b) == 252, "p251 pos");
    }

    // ---- Part 3: append scatters into the ring at the right slots ----------
    // heads=2, head_dim=2, context=5. Feed distinctive vectors and verify the
    // ring slot each lands in, including a wrap overwrite.
    {
        MimiKvBuilder *b = nullptr;
        MimiKvCache   *c = nullptr;
        mimi_kv_builder_init(&b, &arena, 5, err, sizeof(err));
        int rc = mimi_kv_init(&c, &arena, 2, 2, 5, err, sizeof(err));
        CHECK(rc == 0, "cache init");

        uint32_t idx[4];
        float    mask[4 * 5];
        const float *ok, *ov;
        int          klen;

        // Step A: n=1. k_new[h,s,d]; encode value = 100*h + 10*(pos) + d.
        // pos here = 0. Layout [heads=2, n=1, hd=2].
        float kA[2 * 1 * 2] = {/*h0*/ 100 + 0, 100 + 1, /*h1*/ 200 + 0, 200 + 1};
        float vA[2 * 1 * 2] = {-1, -2, -3, -4};
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);  // idx[0]=0
        mimi_kv_append(c, kA, vA, idx, 1, &ok, &ov, &klen);
        CHECK(klen == 5, "append A klen");
        CHECK(ok == mimi_kv_k(c), "append A returns ring");
        // ring[h=0, slot=0, :] == kA[h0]
        CHECK(feq(ok[0 * 5 * 2 + 0 * 2 + 0], 100) &&
                  feq(ok[0 * 5 * 2 + 0 * 2 + 1], 101),
              "append A h0 slot0");
        CHECK(feq(ok[1 * 5 * 2 + 0 * 2 + 0], 200) &&
                  feq(ok[1 * 5 * 2 + 0 * 2 + 1], 201),
              "append A h1 slot0");
        CHECK(feq(ov[0 * 5 * 2 + 0 * 2 + 0], -1), "append A v h0 slot0");

        // Fill positions 1..4 with markers so the ring is full (slots 1..4).
        for (int p = 1; p <= 4; ++p) {
            float kk[2 * 1 * 2] = {100.f + 10 * p + 0, 100.f + 10 * p + 1,
                                   200.f + 10 * p + 0, 200.f + 10 * p + 1};
            float vv[2 * 1 * 2] = {0, 0, 0, 0};
            mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
            mimi_kv_append(c, kk, vv, idx, 1, &ok, &ov, &klen);
        }
        // slot 4 should hold pos-4 marker (140/141, 240/241)
        CHECK(feq(ok[0 * 5 * 2 + 4 * 2 + 0], 140), "ring slot4 h0");

        // Step wrap: pos 5 -> slot 0 overwrites the original pos-0 vector.
        float kW[2 * 1 * 2] = {150.f, 151.f, 250.f, 251.f};
        float vW[2 * 1 * 2] = {9, 9, 9, 9};
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);  // idx[0]=0 (wrap)
        CHECK(idx[0] == 0, "wrap slot 0");
        mimi_kv_append(c, kW, vW, idx, 1, &ok, &ov, &klen);
        CHECK(feq(ok[0 * 5 * 2 + 0 * 2 + 0], 150) &&
                  feq(ok[0 * 5 * 2 + 0 * 2 + 1], 151),
              "wrap overwrote slot0 h0");
        CHECK(feq(ok[1 * 5 * 2 + 0 * 2 + 0], 250), "wrap overwrote slot0 h1");

        // Step 4: reset the cache ring -> zeros; builder reset -> counters 0.
        mimi_kv_reset(c);
        CHECK(feq(mimi_kv_k(c)[0], 0.0f), "reset zeroed ring");
        mimi_kv_builder_reset(b);
        CHECK(mimi_kv_positions(b) == 0, "reset zeroed position");
        mimi_kv_indices_and_mask(b, 1, 1, idx, mask);
        CHECK(idx[0] == 0, "post-reset first slot 0");
    }

    // ---- Part 4: abs path (context <= seq_len) -----------------------------
    // context=3, n=4. get_mask_abs(4,4) causal band; append returns raw k/v.
    {
        MimiKvBuilder *b = nullptr;
        MimiKvCache   *c = nullptr;
        mimi_kv_builder_init(&b, &arena, 3, err, sizeof(err));
        mimi_kv_init(&c, &arena, 1, 2, 3, err, sizeof(err));

        uint32_t idx[4];
        float    mask[4 * 4];
        int klen = mimi_kv_indices_and_mask(b, 4, 1, idx, mask);
        CHECK(klen == 4, "abs klen == n");
        // mask[i][j] = (j>i || i-j>3) ? -inf : 0. With n=4, i-j max is 3, so
        // the lower band is never cut here; only strict upper triangle is -inf.
        float em[16] = {0, I, I, I,  //
                        0, 0, I, I,  //
                        0, 0, 0, I,  //
                        0, 0, 0, 0};
        CHECK(mask_eq(mask, em, 4, 4), "abs mask causal");
        CHECK(idx[0] == 0 && idx[1] == 1 && idx[2] == 2 && idx[3] == 0,
              "abs idx cycles mod context");

        // append abs: returns the raw inputs, ring untouched.
        float kR[1 * 4 * 2] = {1, 2, 3, 4, 5, 6, 7, 8};
        float vR[1 * 4 * 2] = {0};
        const float *ok, *ov;
        int          okl;
        mimi_kv_append(c, kR, vR, idx, 4, &ok, &ov, &okl);
        CHECK(ok == kR && okl == 4, "abs append returns raw k, klen n");
        CHECK(feq(mimi_kv_k(c)[0], 0.0f), "abs append left ring untouched");
    }

    // ---- Part 5: abs mask lower-band cut (i - j > context) -----------------
    // context=2, n=5 -> i-j>2 cuts the far-lower band as well.
    {
        MimiKvBuilder *b = nullptr;
        mimi_kv_builder_init(&b, &arena, 2, err, sizeof(err));
        uint32_t idx[5];
        float    mask[5 * 5];
        mimi_kv_indices_and_mask(b, 5, 1, idx, mask);
        // row i=4: j valid iff j<=4 and 4-j<=2 -> j in {2,3,4}; j=0,1 -> -inf.
        const float *r4 = mask + 4 * 5;
        CHECK(feq(r4[0], I) && feq(r4[1], I) && feq(r4[2], 0.0f) &&
                  feq(r4[3], 0.0f) && feq(r4[4], 0.0f),
              "abs lower-band cut row 4");
    }

    if (g_fail) {
        std::fprintf(stderr, "SELFTEST FAILED\n");
        return 1;
    }
    std::printf("mimi_kv selftest: ALL PASS\n");
    return 0;
}
#endif  // MIMI_KV_SELFTEST

/* NOTES =====================================================================

(a) RUST -> C++ MAPPING (moshi 0.6.4 src/kv_cache.rs)
----------------------------------------------------
Rust type / method                     -> C++ here
  ScatteredCacheBuilder                 -> MimiKvBuilder  (batch=1: positions[0]
                                           -> `position`, indices[0] -> `index`)
  ScatteredCacheBuilder::new            -> mimi_kv_builder_init
  ScatteredCacheBuilder::reset          -> mimi_kv_builder_reset
  ScatteredCacheBuilder::positions()    -> mimi_kv_positions (returns the scalar
                                           positions[0])
  ScatteredCacheBuilder::indices_and_mask     -> mimi_kv_indices_and_mask (ring
                                                 branch, `active` arg = batch_mask[0])
  ScatteredCacheBuilder::indices_and_mask_abs -> same fn, `context <= n` branch
  ScatteredCacheBuilder::get_mask_abs   -> inlined in the abs branch
  ScatteredCacheBuilder::make_cache     -> mimi_kv_init  (allocates + zeroes ring)
  ScatteredKvCache                      -> MimiKvCache (k/v rings)
  ScatteredKvCache::append              -> mimi_kv_append (scatter + return ring)
  ScatteredKvCache::k()/v()             -> mimi_kv_k / mimi_kv_v
  IndicesAndMask{indices, mask}         -> the (out_indices, out_mask) out-params
  reset_batch_index (multi-batch)       -> N/A at batch=1 (== builder_reset); SKIPPED

Local `all_pos: Vec<usize>` in indices_and_mask -> MimiKvBuilder::all_pos
(embedded, sized kMaxContext=250, rebuilt each call). usize::MAX sentinel ->
kEmptySlot = INT64_MAX. `f32::NEG_INFINITY` -> kNegInf = -INFINITY. positions ->
int64_t (stream length can far exceed context; fits i64 for any real session).

The candle tensor plumbing collapses out per the manifest:
  - `scatter_set(indices.broadcast_as(k.shape), k, dim=2)` -> a per-head,
    per-token memcpy of head_dim floats into ring slot indices[s]. The candle
    index broadcast means "same slot for every head and the whole d vector",
    i.e. a plain vector copy — implemented exactly.
  - `iam.indices.unsqueeze/broadcast_as/contiguous` -> nothing; we keep indices
    as a flat u32[n].
  - `Tensor::from_vec(mask, ((),1,seq_len,context))` -> flat f32[n*klen],
    row-major [query, key]. The `1` head-axis is a broadcast; the attention
    `broadcast_add`s it across all 8 heads, so we store one [n,klen] plane.

Reset semantics (faithful): Rust `reset()` only zeroes positions/indices; it
does NOT re-zero the k/v ring. Correctness does not depend on the ring being
zeroed because every not-yet-written slot is -inf-masked (its all_pos stays
kEmptySlot). mimi_kv_reset zeroes the ring anyway to honor the header's
"reset zeroes state" convention — provably harmless. mimi_kv_builder_reset is
the faithful Rust reset (counters only).

The consumer's post-append trim (batched_transformer.rs:97-105 /
transformer.rs:478-486): k_target_len = t + min(context, k_len - t). On this
ScatteredKvCache path k_len == context and t == n, so k_target_len == context
== k_len -> the `if k_target_len < k_len` trim NEVER fires. We therefore always
expose the full context-length ring and let the mask do all validity work. (The
trim only matters for the growable RotatingKvCache path, not this unit.)

(b) PROPOSED ABI (exact signatures) — for unit-#4 reconciliation
----------------------------------------------------------------
kv_cache has no slot in mimi_kernel.h yet; these are proposed additions. The
Rust builder/cache split is preserved because it is REQUIRED (see (d)).

  // Builder — ONE per transformer, shared by all 8 layers.
  int     mimi_kv_builder_init(MimiKvBuilder** st, MimiArena* a, int context,
                               char* err, size_t errlen);
  void    mimi_kv_builder_reset(MimiKvBuilder* b);
  int64_t mimi_kv_positions(const MimiKvBuilder* b);
  int     mimi_kv_indices_and_mask(MimiKvBuilder* b, int n, int active,
                                   uint32_t* out_indices,   // [n]
                                   float*    out_mask);     // [n*klen]
                                   // returns klen (context on ring path, n on abs)

  // Cache -- ONE per layer (8 total).
  int  mimi_kv_init(MimiKvCache** st, MimiArena* a, int heads, int head_dim,
                    int context, char* err, size_t errlen);
  int  mimi_kv_append(MimiKvCache* c, const float* k_new,   // [heads,n,head_dim]
                      const float* v_new, const uint32_t* indices, int n,  // indices [n]
                      const float** out_k, const float** out_v, int* out_klen);
  const float* mimi_kv_k(const MimiKvCache* c);
  const float* mimi_kv_v(const MimiKvCache* c);
  void  mimi_kv_reset(MimiKvCache* c);

Mapping to the task's requested 4-name ABI:
  mimi_kv_init      -> mimi_kv_init (the per-layer cache; heads/head_dim/context)
  mimi_kv_append    -> mimi_kv_append, PLUS mimi_kv_indices_and_mask supplies the
                       mask/indices the task asked append to "return" (split out
                       because the mask is computed once/step, not once/layer).
  mimi_kv_positions -> mimi_kv_positions (on the builder).
  mimi_kv_reset     -> mimi_kv_reset (ring) + mimi_kv_builder_reset (counters).

Per-step call sequence unit #4 should emit (8 layers):
  pos = mimi_kv_positions(builder);              // rope base (read BEFORE i&m)
  klen = mimi_kv_indices_and_mask(builder, n, 1, idx, mask);   // ONCE per step
  for layer L in 0..8:
      mimi_kv_append(cache[L], kL, vL, idx, n, &kL_ring, &vL_ring, &klen);
      // attention: prews[h,t,k] = (qL·kL_ring^T)*hd^-0.5; prews += mask (bcast h);
      //            softmax_last_dim; out = softmax·vL_ring
On turn boundary (reset_state): mimi_kv_builder_reset(builder) + optionally
mimi_kv_reset(cache[L]) for all L.

(c) RING / MASK WORKED EXAMPLES (context = 250, n = 1, active)
-------------------------------------------------------------
Start: position=0, index=0. Each step appends one token.
  pos 0   -> slot 0.   all_pos=[0,MAX,...]. mask=[0, then -inf x249]. (only self)
  pos 249 -> slot 249. ring just filled. all_pos=[0..249]. mask=[0 x250].
  pos 250 -> slot 0    (WRAP). offset=250; pre-write all_pos=[0..249];
             write all_pos[0]=250. Evicts pos 0. mask=[0 x250] (all remaining
             stored positions 1..250 are <= 250).
  pos 251 -> slot 1.   offset=250; pre-write all_pos[0]=250, [1..249]=1..249;
             write all_pos[1]=251. Evicts pos 1. mask=[0 x250].
Eviction is implicit: once full, each step overwrites the oldest slot; the ring
always holds exactly the last 250 positions, and (single-token steps) the mask
is all-zeros because every stored position <= the query position.

Multi-token chunk causality (context=5, from the Rust test, call 4, n=3,
start_pos=2): all_pos is fully populated ([0,1,2,3,4]) BEFORE the mask loop, so
query row 0 (my_pos=2) masks slots holding pos 3,4 as -inf ->
  row0=[0,0,0,-inf,-inf] row1=[0,0,0,0,-inf] row2=[0,0,0,0,0]. Matches Rust.

Wrapped mid-ring (context=5, call 6, n=2, start_pos=6, start_index=1):
offset=5; reconstruct all_pos=[5,1,2,3,4]; write slot1=6, slot2=7 ->
all_pos=[5,6,7,3,4]; masks row0(my_pos=6)=[0,0,-inf,0,0], row1(my_pos=7)=[0×5].
Matches Rust assert exactly.

ABS path (context <= seq_len; NOT on Mimi's hot path): mask = get_mask_abs(n,n)
= causal band, mask[i][j] = -inf iff (j>i) || (i-j>context); width n. append
returns the RAW k/v (no scatter, ring untouched); indices cycle mod context.

(d) BATCH=1 SPECIALIZATION DECISIONS
------------------------------------
- positions/indices Vec<usize> -> scalars `position`/`index` (batch slot 0).
- batch_size(), reset_batch_index(): dropped (multi-batch); builder_reset is the
  single-slot reset_batch_index.
- batch_mask: kept as the `active` arg. Mimi always passes active=1 (single live
  stream). active=0 reproduces the Rust `!batch_mask` branch (frozen index,
  all-zero mask, no advance) and the abs-inactive branch (frozen index, no
  advance, band mask still built) — ported for faithfulness though Mimi never
  hits them.
- Builder/cache SPLIT is retained (not merged) because it is load-bearing: the
  8 layers SHARE one builder (position advances once/step) but own 8 rings.
  A merged per-layer state would advance the stream counter 8x/step. This is
  the single most important correctness decision in the unit.
- all_pos embedded as a fixed [250] array (not arena) -> MimiKvBuilder stays
  fully POD with no interior pointer (hibernation-clean). MimiKvCache keeps
  arena pointers for the (large) rings, same as sibling units.

(e) UNCERTAINTIES / FLAGS FOR THE ARBITER
-----------------------------------------
1. WHICH CACHE DOES MIMI ACTUALLY USE? Big one. mimi.rs decode_step ->
   transformer::Transformer (transformer.rs), which uses candle_nn
   RotatingKvCache and builds its causal mask INLINE in forward_ca
   (transformer.rs:836-869), NOT via indices_and_mask. This ScatteredKvCache
   (kv_cache.rs) is consumed ONLY by batched_transformer.rs. So as written,
   unit #4 (a faithful transformer.rs port) will use RotatingKvCache semantics
   and would NOT call this unit. Options for the arbiter:
     (i)  unit #4 ports transformer.rs faithfully -> needs a RotatingKvCache
          port (different ring: append writes sequentially with a growable/
          rotating buffer; positions() and the inline mask differ in shape —
          mask is (b,1,t,k) with k = number of live positions, and the
          allow-rule is `last_reset_pos <= k_pos && k_pos <= t_pos &&
          t_pos <= k_pos + context`). This unit would then be unused by Mimi.
     (ii) unit #4 is retargeted to batched_transformer.rs (batch=1) -> THIS
          unit is exactly its cache. The two produce the same attention result
          at batch=1 (rotating ring + additive causal mask), so parity holds,
          but the port surface differs.
   RotatingKvCache vs ScatteredKvCache are batch=1-EQUIVALENT in effect (both:
   ring of `context` slots + additive 0/-inf causal mask, oldest evicted on
   wrap), but their index arithmetic and mask tensor SHAPE differ. I ported the
   unit I was assigned (kv_cache.rs). Flagging so the arbiter wires unit #4 to
   the matching cache and doesn't assume this file feeds a transformer.rs port.

2. ROPE POSITION ORDERING (unit #4 concern, not this unit's output). In
   batched_transformer.rs:438-449 indices_and_mask is called BEFORE positions()
   is read for rope, so rope sees the ALREADY-ADVANCED positions (+t). The
   non-batched transformer.rs reads current_seq_len BEFORE append (correct
   [pos..pos+t) base). To match the CORRECT (transformer.rs) behavior, unit #4
   should read mimi_kv_positions BEFORE mimi_kv_indices_and_mask. Documented so
   the ordering is a deliberate choice, not an accident.

3. ABS PATH (context <= seq_len) leaves the ring stale (append returns raw k/v)
   and desyncs indices vs ring — faithful to Rust, and unreachable on Mimi's
   n==1 decode path. If any caller ever streams a >=250-token chunk through
   this cache, the ring is bypassed exactly as in Rust; flagged in case a
   future prefill path relies on the ring being coherent afterwards (it isn't,
   in either language).

4. No NEON / scalar-ref sibling: this unit has no float reduction (only index
   math, a mask fill, and memcpy). The header's `..._ref under MIMI_SCALAR_REF`
   rule targets compute kernels; there is no arithmetic to bisect here.
=========================================================================== */
