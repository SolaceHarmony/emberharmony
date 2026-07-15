// mimi_quant.cpp — Unit #1 of the Mimi decode port (see docs/MIMI_PORT.md).
// Faithful C++/NEON port of the DECODE half of moshi 0.6.4 src/quantization.rs:
//   EuclideanCodebook::decode, VectorQuantization::decode,
//   ResidualVectorQuantization::decode, ResidualVectorQuantizer::decode
//   (incl. output_proj), SplitResidualVectorQuantizer::decode.
// Encode paths, CodebookEncode CustomOp2, and all training/EMA logic are OUT
// OF SCOPE and intentionally absent. Config is fixed to Mimi v0_1(8):
//   n_q = 8 (rvq_first 1 + rvq_rest 7), bins = 2048, codebook dim = 256,
//   model dim = 512. See the /* NOTES */ block at the bottom for the full
//   Rust->C++ mapping, weight list, and every interpretation made.
//
// ABI: mimi_kernel.h (arbiter-owned — code against it, do NOT edit it).

#include "mimi_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__ARM_NEON) && !defined(MIMI_SCALAR_REF)
#include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// Compile-time shape facts for this checkpoint (mirror mimi_kernel.h enums).
// ---------------------------------------------------------------------------
namespace {
constexpr int kBins     = MIMI_BINS;       // 2048 codebook entries
constexpr int kQDim     = MIMI_QUANT_DIM;  // 256 codebook / working dim
constexpr int kModelDim = MIMI_DIM;        // 512 model dim (output_proj out)
constexpr int kRestNq   = MIMI_NQ - 1;     // 7 codebooks in rvq_rest
constexpr float kCbEps  = 1e-5f;           // EuclideanCodebook epsilon (f32)
}  // namespace

// ---------------------------------------------------------------------------
// State (POD, carved from the arena at init; steady state never allocates).
// Folded codebook tables and the raw output_proj weight views live here, plus
// small scratch for the residual sum and the two per-RVQ 512-vectors.
// ---------------------------------------------------------------------------
struct MimiQuantState {
    // Folded effective embeddings, checkpoint row layout [bins, qdim] row-major.
    // emb[k*kQDim + d] = embedding_sum[k,d] / max(cluster_usage[k], eps).
    const float *emb_first;             // rvq_first codebook 0
    const float *emb_rest[kRestNq];     // rvq_rest codebooks 0..6

    // output_proj: conv1d(qdim -> model_dim, kernel 1, no bias). With kernel 1
    // and T=1 this is a pure gemv; weight [512,256,1] collapses to [512,256]
    // row-major, exactly mimi_gemv_f32's [M=512,K=256] layout.
    const float *out_proj_first;        // [kModelDim * kQDim]
    const float *out_proj_rest;         // [kModelDim * kQDim]

    // Scratch (arena).
    float *quant_rest;                  // [kQDim]  residual sum for rvq_rest
    float *emb_first_out;               // [kModelDim] projected rvq_first
    float *emb_rest_out;                // [kModelDim] projected rvq_rest
};

// ===========================================================================
// Small kernels. The only reduction here is the residual accumulation (an
// elementwise add across codebooks) — order-independent per lane. Everything
// else is copy or scalar division. mimi_gemv_f32 (shared, declared in the
// header, implemented elsewhere) owns the projection dot-product order.
// ===========================================================================
namespace {

#ifdef MIMI_SCALAR_REF
// Scalar reference sibling for the NEON accumulate (parity bisecting).
// acc[i] += x[i] for i in [0,n); lanes independent, no cross-lane reduction.
void vec_add_ref(float *acc, const float *x, int n) {
    for (int i = 0; i < n; ++i) acc[i] += x[i];
}
#endif

// acc[i] += x[i], i in [0,n). Elementwise add across a codebook contribution;
// accumulation order is per-index only (no reduction across i), so NEON and
// scalar agree bit-for-bit.
inline void vec_add(float *acc, const float *x, int n) {
#ifdef MIMI_SCALAR_REF
    vec_add_ref(acc, x, n);
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vld1q_f32(acc + i);
        float32x4_t b = vld1q_f32(x + i);
        vst1q_f32(acc + i, vaddq_f32(a, b));
    }
    for (; i < n; ++i) acc[i] += x[i];
#else
    for (int i = 0; i < n; ++i) acc[i] += x[i];
#endif
}

// Fold one EuclideanCodebook: dst[k,d] = embedding_sum[k,d] / max(usage[k],eps).
// Runs once at init (not steady state). Faithful to moshi: candle's
//   embedding = embedding_sum.broadcast_div(cluster_usage.maximum(1e-5))
// with Maximum defined as `if v1 < v2 { v2 } else { v1 }` (v1=usage, v2=eps,
// eps cast to f32 first) and Div as plain f32 v1/v2. Row loop k outer, d inner.
void fold_codebook(float *dst, const float *emb_sum, const float *usage) {
    for (int k = 0; k < kBins; ++k) {
        const float u = usage[k];
        const float denom = (u < kCbEps) ? kCbEps : u;  // candle Maximum tie rule
        const float *src = emb_sum + (size_t)k * kQDim;
        float *out = dst + (size_t)k * kQDim;
        for (int d = 0; d < kQDim; ++d) out[d] = src[d] / denom;
    }
}

// Clamp a code index into [0, bins). moshi's index_select would hard-error on
// an out-of-range index; the ABI decode returns void, so we clamp defensively
// rather than read OOB. Model-emitted codes are always in range; see NOTES.
inline uint32_t clamp_code(uint32_t c) {
    return (c < (uint32_t)kBins) ? c : (uint32_t)(kBins - 1);
}

// Look up + validate one required f32 weight span. Returns nullptr and writes
// err on missing/misshaped weight (no-fallbacks: init hard-fails upstream).
const float *find_req(const MimiWeightTable *w, const char *name,
                      uint64_t expect_len, char *err, size_t errlen) {
    const MimiWeight *e = mimi_weight_find(w, name);
    if (e == nullptr) {
        if (errlen) snprintf(err, errlen, "mimi_quant: missing weight '%s'", name);
        return nullptr;
    }
    if (e->len != expect_len) {
        if (errlen)
            snprintf(err, errlen,
                     "mimi_quant: weight '%s' has %llu elems, expected %llu",
                     name, (unsigned long long)e->len,
                     (unsigned long long)expect_len);
        return nullptr;
    }
    if (e->data == nullptr) {  // review P2: never hand back a null span
        if (errlen) snprintf(err, errlen, "mimi_quant: weight '%s' has null data", name);
        return nullptr;
    }
    return e->data;
}

// Fold one codebook by name-prefix into arena, returning the folded table (or
// nullptr on failure). prefix e.g. "quantizer.rvq_first.vq.layers.0".
const float *fold_codebook_by_prefix(const MimiWeightTable *w, MimiArena *a,
                                     const char *prefix, char *err,
                                     size_t errlen) {
    char name[256];
    snprintf(name, sizeof(name), "%s._codebook.embedding_sum", prefix);
    const float *emb_sum =
        find_req(w, name, (uint64_t)kBins * kQDim, err, errlen);
    if (!emb_sum) return nullptr;

    snprintf(name, sizeof(name), "%s._codebook.cluster_usage", prefix);
    const float *usage = find_req(w, name, (uint64_t)kBins, err, errlen);
    if (!usage) return nullptr;

    float *folded =
        (float *)mimi_arena_alloc(a, (size_t)kBins * kQDim * sizeof(float));
    fold_codebook(folded, emb_sum, usage);
    return folded;
}

}  // namespace

// ===========================================================================
// ABI entry points.
// ===========================================================================
extern "C" {

int mimi_quant_init(MimiQuantState **out, const MimiWeightTable *w,
                    MimiArena *a, char *err, size_t errlen) {
    if (out == nullptr || w == nullptr || a == nullptr) {
        if (errlen) snprintf(err, errlen, "mimi_quant: null argument to init");
        return 1;
    }

    MimiQuantState *st =
        (MimiQuantState *)mimi_arena_alloc(a, sizeof(MimiQuantState));

    // ---- rvq_first: 1 codebook + output_proj -----------------------------
    st->emb_first = fold_codebook_by_prefix(
        w, a, "quantizer.rvq_first.vq.layers.0", err, errlen);
    if (!st->emb_first) return 2;

    st->out_proj_first =
        find_req(w, "quantizer.rvq_first.output_proj.weight",
                 (uint64_t)kModelDim * kQDim, err, errlen);
    if (!st->out_proj_first) return 3;

    // ---- rvq_rest: 7 codebooks (layers 0..6) + output_proj ---------------
    // NOTE: the checkpoint stores 31 rest codebooks; at n_q=8 only 0..6 are
    // constructed by ResidualVectorQuantization::new(n_q-1=7), so we fold
    // exactly those seven and ignore layers 7..30.
    for (int j = 0; j < kRestNq; ++j) {
        char prefix[128];
        snprintf(prefix, sizeof(prefix),
                 "quantizer.rvq_rest.vq.layers.%d", j);
        st->emb_rest[j] = fold_codebook_by_prefix(w, a, prefix, err, errlen);
        if (!st->emb_rest[j]) return 4;
    }

    st->out_proj_rest =
        find_req(w, "quantizer.rvq_rest.output_proj.weight",
                 (uint64_t)kModelDim * kQDim, err, errlen);
    if (!st->out_proj_rest) return 5;

    // ---- scratch ---------------------------------------------------------
    st->quant_rest    = (float *)mimi_arena_alloc(a, (size_t)kQDim * sizeof(float));
    st->emb_first_out = (float *)mimi_arena_alloc(a, (size_t)kModelDim * sizeof(float));
    st->emb_rest_out  = (float *)mimi_arena_alloc(a, (size_t)kModelDim * sizeof(float));

    *out = st;
    return 0;
}

// Decode one latent frame (B=1, T=1). codes[MIMI_NQ]: codes[0] -> rvq_first
// codebook 0; codes[1..8] -> rvq_rest codebooks 0..6. emb_out: [MIMI_DIM].
//
// Faithful control flow of SplitResidualVectorQuantizer::decode specialized to
// B=1,T=1:
//   rvq_first.decode(codes[..1]):
//     ResidualVectorQuantization::decode over 1 layer -> codebook lookup of
//     code[0] (EuclideanCodebook::decode = embedding row, dim 256). No VQ-level
//     project_out (codebook_dim==dim). The Rust .t() on [1,256,1] is a no-op
//     for T=1. Then ResidualVectorQuantizer::output_proj (conv1d 256->512).
//   rvq_rest.decode(codes[1..]):
//     sum of 7 codebook lookups in 256-space, layer order 0,1,...,6 (matches
//     `quantized = layers[0]; for i in 1..7 { quantized += layers[i] }`), then
//     one output_proj (conv1d 256->512).
//   quantized = rvq_first_512 + rvq_rest_512   (two separate projections,
//     summed last — exactly the Rust `quantized + rvq_rest.decode(...)`).
void mimi_quant_decode(MimiQuantState *st, const uint32_t *codes,
                       float *emb_out) {
    // ---- rvq_first (single codebook, no residual sum) --------------------
    // The lone lookup row IS the 256-vec; project it straight through.
    const uint32_t c0 = clamp_code(codes[0]);
    const float *row_first = st->emb_first + (size_t)c0 * kQDim;
    mimi_gemv_f32(st->out_proj_first, row_first, /*bias*/ nullptr,
                  st->emb_first_out, kModelDim, kQDim);

    // ---- rvq_rest (sum 7 lookups in 256-space, then project) -------------
    // quant_rest = emb_rest[0][codes[1]], then += emb_rest[j][codes[1+j]].
    const uint32_t c_rest0 = clamp_code(codes[1]);
    std::memcpy(st->quant_rest,
                st->emb_rest[0] + (size_t)c_rest0 * kQDim,
                (size_t)kQDim * sizeof(float));
    for (int j = 1; j < kRestNq; ++j) {
        const uint32_t cj = clamp_code(codes[1 + j]);
        vec_add(st->quant_rest, st->emb_rest[j] + (size_t)cj * kQDim, kQDim);
    }
    mimi_gemv_f32(st->out_proj_rest, st->quant_rest, /*bias*/ nullptr,
                  st->emb_rest_out, kModelDim, kQDim);

    // ---- split sum in 512-space: first + rest ----------------------------
    std::memcpy(emb_out, st->emb_first_out, (size_t)kModelDim * sizeof(float));
    vec_add(emb_out, st->emb_rest_out, kModelDim);
}

}  // extern "C"

/* NOTES ---------------------------------------------------------------------

(a) Rust fn -> C++ fn mapping
  -------------------------------------------------------------------------
  moshi 0.6.4 quantization.rs (DECODE path)      | mimi_quant.cpp
  -------------------------------------------------------------------------
  EuclideanCodebook::new (embedding fold)        | fold_codebook / fold_codebook_by_prefix (at init)
  EuclideanCodebook::decode (index_select)       | pointer arithmetic `emb + code*kQDim` in mimi_quant_decode
  VectorQuantization::decode (project_out=None,  | (inlined) — project_out is None here, so the codebook
    then .t())                                   |   row is used directly; .t() is a no-op for T=1
  ResidualVectorQuantization::decode (residual   | mimi_quant_decode: memcpy layer0 then vec_add layers 1..6
    sum over layers)                             |
  ResidualVectorQuantizer::decode (vq.decode     | mimi_quant_decode: mimi_gemv_f32 with out_proj_{first,rest}
    then output_proj conv1d)                     |
  SplitResidualVectorQuantizer::decode           | mimi_quant_decode top level (first + rest)
  EuclideanCodebook::{encode,encode_slow,        | SKIPPED (encode-only / out of scope)
    encode_very_slow}, CodebookEncode CustomOp2  |
  VectorQuantization::encode, RVQ::encode,       | SKIPPED
    ResidualVectorQuantizer::encode, Split encode|
  input_proj (both RVQs)                         | SKIPPED — encode-only; decode never touches it

(b) Weights consumed (all F32 in this checkpoint), with shape:
  quantizer.rvq_first.vq.layers.0._codebook.embedding_sum   [2048, 256]  (folded)
  quantizer.rvq_first.vq.layers.0._codebook.cluster_usage   [2048]       (folded, divisor)
  quantizer.rvq_first.output_proj.weight                    [512, 256, 1] -> gemv [512,256]
  quantizer.rvq_rest.vq.layers.{0..6}._codebook.embedding_sum  [2048, 256] each (folded)
  quantizer.rvq_rest.vq.layers.{0..6}._codebook.cluster_usage  [2048]      each (folded, divisor)
  quantizer.rvq_rest.output_proj.weight                    [512, 256, 1] -> gemv [512,256]
  Deliberately NOT consumed:
    *._codebook._initialized                 (init flag; decode ignores it)
    *._codebook.embedding                     (NOT in checkpoint — derived at runtime; we fold it)
    quantizer.rvq_{first,rest}.input_proj.weight   (encode-only)
    quantizer.rvq_rest.vq.layers.{7..30}.*    (exist in checkpoint but unused at n_q=8)

(c) Interpretations / uncertainties (arbiter please re-check):
  1. Embedding fold formula. EuclideanCodebook::new computes
       embedding = embedding_sum.broadcast_div( cluster_usage.maximum(1e-5).unsqueeze(1) )
     I read candle-core 0.9.2: op.rs Maximum = `|v1,v2| if v1 < v2 { v2 } else { v1 }`
     with v1 = cluster_usage (lhs), v2 = scalar (rhs); the scalar 1e-5 (f64) is
     `to_dtype(f32)` FIRST (tensor.rs binary_op_scalar macro), so it is (float)1e-5.
     Reproduced as `denom = (u < 1e-5f) ? 1e-5f : u` (matches the tie/branch exactly),
     then plain f32 `emb_sum[k,d] / denom`. The `c2` tensor (encode-only) is not computed.
  2. output_proj is candle_nn::conv1d_no_bias(qdim, dim, kernel=1, Default cfg)
     => stride 1, padding 0, dilation 1, groups 1, no bias. Weight [out,in,1].
     For a length-1 kernel on a T=1 frame this is exactly y[o]=sum_i W[o,i]*x[i],
     delegated to mimi_gemv_f32(W[512x256], x[256], null, y[512], 512, 256).
     force_projection=true for BOTH rvqs => output_proj always exists (Some).
  3. VQ-level project_in/project_out are None (codebook_dim==dim==256), so there
     is no 256<->256 linear inside VectorQuantization; only the two conv1d
     output_projs (256->512) at the ResidualVectorQuantizer level exist on the
     decode path. Confirmed against checkpoint (no vq.layers.*.project_* tensors).
  4. Code range. index_select would hard-error on OOB; the void ABI can't return
     an error, so clamp_code() clamps to [0,2047] defensively. Model-emitted codes
     are always in range, so this never fires in practice — flag if the harness
     wants a hard-fault instead.
  5. B=1,T=1 specialization. The Rust threads [B,K,T] through transpose(0,1) and
     per-frame .t(); for B=1,T=1 all transposes are data-layout no-ops, so they
     are elided. The residual-sum ORDER and the first-then-rest sum order are
     preserved exactly (this is the part that must not rot).

(d) Accumulation order (every reduction):
  - Embedding fold: independent per (k,d); k outer, d inner. No reduction.
  - rvq_rest residual sum (vec_add): quant_rest = lookup(layer0); then
    += lookup(layer1), += lookup(layer2), ... += lookup(layer6). Sequential in
    layer index, elementwise per dim (no cross-dim reduction). Matches
    ResidualVectorQuantization::decode (`layers[0]` then `enumerate().skip(1)`).
  - output_proj dot products: owned by mimi_gemv_f32 (shared primitive); its
    documented order is y[m]=sum_{k=0..255} W[m,k]*x[k], f32 accumulate.
  - Split final sum: emb_out = emb_first_out; then += emb_rest_out, elementwise
    over the 512 dims. Matches `quantized + rvq_rest.decode(...)`. The two
    output_projs are applied to their own 256-vecs BEFORE this sum (proj-then-sum,
    exactly as the Rust does — NOT folded into one projection).

(e) For the arbiter to re-check:
  - candle broadcast_div / Maximum semantics quoted above are from candle-core
    0.9.2 (the registry copy in-tree). If the pinned candle version differs,
    re-verify the Maximum tie-break and the f32 scalar cast.
  - Whether the parity harness prefers a hard fault over clamp_code on OOB codes.
  - Arena budget: this unit folds 8 codebooks * 2048 * 256 * 4 B = 16 MiB into
    the arena at init, plus ~4 KiB scratch. Confirm the arena is sized for it.
  - NEON vec_add has a scalar sibling vec_add_ref under -DMIMI_SCALAR_REF; both
    are bit-identical (elementwise add). fold_codebook and clamp are scalar-only
    (init-time / trivial), so they need no _ref.
--------------------------------------------------------------------------- */
