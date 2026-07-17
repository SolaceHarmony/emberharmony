// mimi_conv.cpp — faithful C++/NEON port of moshi 0.6.4 streaming conv
// primitives (crates.io moshi-0.6.4/src/conv.rs), decoder path only.
//
// Ported (see /* NOTES */ at end for the full Rust->C++ map):
//   NormConv1d / NormConvTranspose1d forward MATH (the raw candle
//     conv1d / conv_transpose1d the streaming steps call) + WeightNorm fold.
//   StreamableConv1d::step         -> mimi_conv_step
//   StreamableConvTranspose1d::step-> mimi_convtr_step
//   ConvTrUpsample1d::step          -> mimi_upsample_step (depthwise, groups=dim)
//
// Deliberately NOT ported (out of the decode hot path / manifest SKIP):
//   the non-streaming Module::forward padding (get_extra_padding_for_conv1d,
//     symmetric pad, unpad1d) — not in the ABI, decode uses step only.
//   ConvDownsample1d (encode-only), batched StreamMask (batch==1: the mask is
//     always None here, so every mask where_cond branch collapses to identity).
//
// MATH IS ASSEMBLY (her rule): every reduction/sweep is aarch64 NEON
// (float32x4_t / vfmaq_f32) as the PRIMARY path. Scalar exists only in the
// MIMI_SCALAR_REF parity sibling build and in sub-vector tail remainders.
// Data marshalling (weight/carry gathers, strided scatters) is not "math" and
// stays scalar, mirroring candle's own inp_cont / k_cont staging copies.
//
// Layout: conv layout [C, T], channel-major, T contiguous, batch==1.
// f32 in, f32 accumulate. All state is POD, carved once from the arena;
// steady state never allocates.

#include "mimi_kernel.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(__aarch64__) && !defined(MIMI_SCALAR_REF)
#define MIMI_NEON 1
#define MIMI_SSE2 0
#include <arm_neon.h>
#elif (defined(__x86_64__) || defined(_M_X64)) && !defined(MIMI_SCALAR_REF)
#define MIMI_NEON 0
#define MIMI_SSE2 1
#include <immintrin.h>
#else
#define MIMI_NEON 0
#define MIMI_SSE2 0
#endif

// Cap on frames-per-step for the convtr time-axis reduction scratch. The
// largest input to any decoder transposed conv is 480 (ratio-4 layer); this
// gives 4x headroom. ENFORCED at the ABI (review P1): convtr/upsample steps
// refuse n_in beyond it instead of overrunning the arena-carved g scratch.
enum { MIMI_CONV_MAX_NIN = MIMI_FRAME_OUT };

// Matrix-route bound: steps with n_in (or num_frames) at or below this AND wide
// reductions use the byte-load SIMD matrix leaf instead of time-axis NEON. Rationale
// (review P1, measured): the widest decoder layers (init conv 512->1024 k7,
// convtr 1024->512 k16) receive only n=2 time samples, so a 4-lane time-axis
// AXPY runs entirely in its scalar tail — ~45 ms of the ~70 ms frame. The
// GEMM formulation moves the in_c*k reduction into a row/column SIMD kernel. It
// REORDERS the K reduction (SIMD blocking vs kk-outer/ic-inner) — faithful
// tier; re-measured against the proven ~4e-6 whole-chain parity bar.
enum { MIMI_CONV_GEMM_MAX_N = 512 };

/* ======================================================================== *
 *  NEON contiguous primitives (primary path; scalar under MIMI_SCALAR_REF)
 * ======================================================================== */

// y[i] += w * x[i]      — the vectorized MAC (conv1d time axis, convtr in-ch reduction)
static inline void vaxpy(float *y, const float *x, float w, int n) {
#if MIMI_NEON
    const float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), vld1q_f32(x + i), wv));
    for (; i < n; ++i) y[i] += w * x[i];      // sub-vector tail
#else
    for (int i = 0; i < n; ++i) y[i] += w * x[i];
#endif
}

// y[i] = s * x[i]       — depthwise upsample multiply (channel axis)
static inline void vscale(float *y, const float *x, float s, int n) {
#if MIMI_NEON
    const float32x4_t sv = vdupq_n_f32(s);
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vmulq_f32(vld1q_f32(x + i), sv));
    for (; i < n; ++i) y[i] = x[i] * s;
#else
    for (int i = 0; i < n; ++i) y[i] = x[i] * s;
#endif
}

// y[i] += c             — bias broadcast-add over a contiguous time run
static inline void vadd_scalar(float *y, float c, int n) {
#if MIMI_NEON
    const float32x4_t cv = vdupq_n_f32(c);
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), cv));
    for (; i < n; ++i) y[i] += c;
#else
    for (int i = 0; i < n; ++i) y[i] += c;
#endif
}

// y[i] += p[i] - c      — overlap-add of prior carry with bias removed
static inline void voverlap(float *y, const float *p, float c, int n) {
#if MIMI_NEON
    const float32x4_t cv = vdupq_n_f32(c);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vsubq_f32(vld1q_f32(p + i), cv)));
    for (; i < n; ++i) y[i] += p[i] - c;
#else
    for (int i = 0; i < n; ++i) y[i] += p[i] - c;
#endif
}

/* ======================================================================== *
 *  small scalar helpers (init-time / data marshalling only)
 * ======================================================================== */

static int fail(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return 1;
}

static const MimiWeight *wfind(const MimiWeightTable *w, const char *prefix,
                               const char *suffix) {
    char name[256];
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);
    return mimi_weight_find(w, name);
}

/* ------------------------------------------------------------------------ *
 *  WeightNorm fold  (conv1d_weight_norm / NormConvTranspose1d WeightNorm)
 *
 *  Rust (conv.rs:35-42 and 133-140):
 *     norm_v = weight_v.sqr().sum_keepdim((1,2)).sqrt()   // over dims 1 & 2
 *     weight = weight_v * weight_g / norm_v               // g,norm broadcast dim0
 *
 *  conv1d weight_v [out_c, in_c/groups, k], weight_g [out_c,1,1]: norm over
 *    (in,k) per OUTPUT channel oc.
 *  convtr weight_v [in_c, out_c/groups, k], weight_g [in_c,1,1]: norm over
 *    (out,k) per INPUT channel ic (dim0 is in_c for convtr).
 *
 *  This checkpoint (kyutai moshiko-candle-bf16) stores the already-folded
 *  "weight" (0 weight_g/weight_v tensors); the fold path runs only if a future
 *  export ships raw weight_g/weight_v. Folded ONCE here into the arena.
 *  (Reductions here are init-time, not steady-state; kept scalar-clear.)
 * ------------------------------------------------------------------------ */
static void weight_norm_fold(const uint8_t *v, const uint8_t *g, float *out,
                             int n0, int rest) {
    for (int c = 0; c < n0; ++c) {
        float ss = 0.f;
        for (int i = 0; i < rest; ++i) {
            const float value = mimi_weight_load_f32(
                v, static_cast<uint64_t>(c) * rest + i);
            ss += value * value;
        }
        const float scale = mimi_weight_load_f32(g, c) / sqrtf(ss);
        float *orow = out + (size_t)c * rest;
        for (int i = 0; i < rest; ++i) {
            orow[i] = mimi_weight_load_f32(
                          v, static_cast<uint64_t>(c) * rest + i) *
                      scale;
        }
    }
}

// Exact-shape check (review P2: the header promises rejection of misshaped
// weights, not just wrong element counts): 3-D, dims match, data non-null.
static int wcheck3(const MimiWeight *ww, int64_t d0, int64_t d1, int64_t d2) {
    return ww && ww->bytes && ww->ndim == 3 && ww->shape &&
           ww->shape[0] == d0 && ww->shape[1] == d1 && ww->shape[2] == d2;
}

// Resolve the (possibly weight-normed) weight under `base`. Zero-copy if a
// folded ".weight" exists; else fold weight_g/weight_v into the arena.
// Expected shape [d0, d1, d2]; d0 is the fold broadcast dim (out_c for conv1d,
// in_c for convtr).
static const uint8_t *resolve_weight(const MimiWeightTable *w, const char *base,
                                     int64_t d0, int64_t d1, int64_t d2,
                                     MimiArena *a, char *err, size_t errlen) {
    const int rest = (int)(d1 * d2);
    const MimiWeight *ww = wfind(w, base, ".weight");
    if (ww) {
        if (!wcheck3(ww, d0, d1, d2)) {
            fail(err, errlen, "conv weight shape mismatch (expect 3-D [d0,d1,d2])");
            return NULL;
        }
        return ww->bytes;
    }
    const MimiWeight *vg = wfind(w, base, ".weight_g");
    const MimiWeight *vv = wfind(w, base, ".weight_v");
    if (!vg || !vv) {
        fail(err, errlen, "conv weight (nor weight_g/weight_v) not found");
        return NULL;
    }
    if (!wcheck3(vv, d0, d1, d2) || !wcheck3(vg, d0, 1, 1)) {
        fail(err, errlen, "conv weight_v/weight_g shape mismatch");
        return NULL;
    }
    float *folded = (float *)mimi_arena_alloc_derived(
        a, (size_t)d0 * rest * sizeof(float));
    if (mimi_arena_building_derived(a)) {
        weight_norm_fold(vv->bytes, vg->bytes, folded, (int)d0, rest);
    }
    return reinterpret_cast<const uint8_t *>(folded);
}

/* ======================================================================== *
 *  1. StreamableConv1d  -> MimiConvState
 * ======================================================================== */
// Weights: "<prefix>.conv.conv.weight" [out_c, in_c/groups, k] (+ ".bias"
// [out_c]) — the StreamableConv1d owns the ".conv.conv" nesting (NormConv1d ->
// Conv1d). Pass `prefix` = the StreamableConv1d node, e.g. "decoder.model.0".
//
// Streaming state (Rust: state_prev_xs + left_pad_applied). We NEVER
// materialise cat(prev_xs, xs); output element (oc,f) reads its taps from the
// logical sequence [prev ++ xs]. Vectorization is over the OUTPUT TIME axis f:
// for stride==1 (every decoder conv1d) the tap contribution to y[oc, 0..nf) is
// a contiguous NEON AXPY of an input row scaled by one weight, split once at
// the prev/xs boundary. prev holds the left-context carry in [C, carry_cap].
//
// INVARIANT (proven from the Rust step arithmetic): after every step
//   prev_len < kernel_eff. num_frames = (seq_len+stride-kernel_eff)/stride
//   (floor; 0 when seq_len+stride<kernel_eff). Retained carry length
//   seq_len - num_frames*stride is in [kernel_eff-stride, kernel_eff) when
//   num_frames>0, and equals seq_len (< kernel_eff) when num_frames==0. The
//   first-step left pad = padding_total = kernel_eff-stride < kernel_eff. So
//   carry_cap = kernel_eff bounds prev for all time.
struct MimiConvState {
    int in_c, out_c, ksize, stride, dilation, groups;
    int causal;          // stored for parity; step is inherently causal (Rust
                         // step left-pads only and never reads self.causal)
    int kernel_eff;      // (ksize-1)*dilation + 1
    int padding_total;   // kernel_eff - stride  (first-step left pad, Constant)
    int cin_g;           // in_c / groups
    int cout_g;          // out_c / groups
    int carry_cap;       // = kernel_eff  (>= max carry, >= padding_total)
    const uint8_t *w;    // [out_c, in_c/groups, ksize] f32 bytes
    const uint8_t *bias; // [out_c] f32 bytes or NULL
    float *prev;         // [in_c, carry_cap] left-context carry
    float *cbuf;         // [in_c, carry_cap] scratch for the next carry gather
    int prev_len;        // # carried time steps currently in prev  (< kernel_eff)
    int left_pad_applied;
    // Matrix route (short-time wide layers): C[out_c,nf] = W[out_c,ic*k] x B.
    // W rows are ALREADY (ic,kk)-contiguous in checkpoint layout — zero-copy A;
    // only the im2col gather B is staged (activation marshalling, like candle).
    int route_gemm;      // groups==1 && cin_g*ksize wide enough
    float *im2col;       // [cin_g*ksize, MIMI_CONV_GEMM_MAX_N]
};

int mimi_conv_init(MimiConvState **st, const MimiWeightTable *w,
                   const char *prefix, int in_c, int out_c, int ksize,
                   int stride, int dilation, int groups, int causal,
                   MimiArena *a, char *err, size_t errlen) {
    if (ksize < stride) return fail(err, errlen, "conv1d: kernel < stride");
    if (in_c % groups || out_c % groups)
        return fail(err, errlen, "conv1d: channels not divisible by groups");

    MimiConvState *s = (MimiConvState *)mimi_arena_alloc(a, sizeof(MimiConvState));
    memset(s, 0, sizeof(*s));
    s->in_c = in_c; s->out_c = out_c; s->ksize = ksize; s->stride = stride;
    s->dilation = dilation; s->groups = groups; s->causal = causal;
    s->cin_g = in_c / groups; s->cout_g = out_c / groups;
    s->kernel_eff = (ksize - 1) * dilation + 1;
    s->padding_total = s->kernel_eff - stride;
    s->carry_cap = s->kernel_eff;

    char base[256];
    snprintf(base, sizeof(base), "%s.conv.conv", prefix);
    s->w = resolve_weight(w, base, out_c, s->cin_g, ksize, a, err, errlen);
    if (!s->w) return 1;
    const MimiWeight *b = wfind(w, base, ".bias");
    if (b) {
        if (!b->bytes || !b->shape || b->ndim != 1 ||
            b->shape[0] != (uint64_t)out_c || b->len != (uint64_t)out_c)
            return fail(err, errlen, "conv1d bias missing data or length mismatch");
        s->bias = b->bytes;
    }
    s->prev = (float *)mimi_arena_alloc(a, (size_t)in_c * s->carry_cap * sizeof(float));
    s->cbuf = (float *)mimi_arena_alloc(a, (size_t)in_c * s->carry_cap * sizeof(float));
    s->prev_len = 0;
    s->left_pad_applied = 0;
    // Matrix route for the wide layers: at n=2 (the seanet entry shapes) the
    // time-axis AXPY is all scalar tail; the GEMM moves the ic*k reduction
    // onto the matrix unit. Narrow convs (final 64->1 k3) stay NEON.
    s->route_gemm = (groups == 1) && ((size_t)s->cin_g * ksize >= 1024);
    if (s->route_gemm)
        s->im2col = (float *)mimi_arena_alloc(
            a, (size_t)s->cin_g * ksize * MIMI_CONV_GEMM_MAX_N * sizeof(float));
    *st = s;
    return 0;
}

void mimi_conv_reset(MimiConvState *s) {
    s->prev_len = 0;
    s->left_pad_applied = 0;
}

// logical input sample (channel c, logical time p) from [prev ++ xs].
static inline float conv_read(const MimiConvState *s, const float *xs, int n_in,
                              int c, int p) {
    if (p < s->prev_len) return s->prev[(size_t)c * s->carry_cap + p];
    return xs[(size_t)c * n_in + (p - s->prev_len)];
}

int mimi_conv_step(MimiConvState *s, const float *xs, int n_in, float *y) {
    // Empty StreamTensor propagation: 0 in => 0 out, no state change (the Rust
    // step short-circuits on None BEFORE the first-step pad).
    if (n_in <= 0) return 0;
    // First step: prepend padding_total zeros on the LEFT (PadMode::Constant),
    // modelled by pre-loading prev with zeros (cat2(empty, pad1d(xs,pt,0))==pad).
    if (!s->left_pad_applied) {
        s->left_pad_applied = 1;
        const int pt = s->padding_total;
        for (int c = 0; c < s->in_c; ++c)
            memset(s->prev + (size_t)c * s->carry_cap, 0, pt * sizeof(float));
        s->prev_len = pt;
    }

    const int stride = s->stride, dil = s->dilation, ke = s->kernel_eff;
    const int seq_len = s->prev_len + n_in;
    const int num_frames = (seq_len + stride >= ke) ? (seq_len + stride - ke) / stride : 0;

    if (num_frames > 0 && s->route_gemm && num_frames <= MIMI_CONV_GEMM_MAX_N) {
        // Matrix route: B[(ic*k + kk), f] = logical[ic][f*stride + kk*dil] gathered
        // once (row order matches W's (ic,kk) checkpoint layout — zero-copy A),
        // then C[out_c, nf] = W x B on the matrix unit, bias broadcast after.
        const int nf = num_frames, k = s->ksize;
        const int kdim = s->cin_g * k;
        for (int ic = 0; ic < s->cin_g; ++ic)
            for (int kk = 0; kk < k; ++kk) {
                float *brow = s->im2col + ((size_t)ic * k + kk) * nf;
                const int base = kk * dil;
                for (int f = 0; f < nf; ++f)
                    brow[f] = conv_read(s, xs, n_in, ic, f * stride + base);
            }
        mimi_weight_gemm_f32(s->w, s->im2col, y, s->out_c, kdim, nf, 0);
        if (s->bias)
            for (int oc = 0; oc < s->out_c; ++oc)
                vadd_scalar(y + (size_t)oc * nf,
                            mimi_weight_load_f32(s->bias, oc), nf);
        // carry update below is shared with the NEON route.
        const int offset = num_frames * stride;
        const int carry_len = seq_len - offset;
        for (int c = 0; c < s->in_c; ++c) {
            float *dst = s->cbuf + (size_t)c * s->carry_cap;
            for (int j = 0; j < carry_len; ++j)
                dst[j] = conv_read(s, xs, n_in, c, offset + j);
        }
        for (int c = 0; c < s->in_c; ++c)
            memcpy(s->prev + (size_t)c * s->carry_cap,
                   s->cbuf + (size_t)c * s->carry_cap, carry_len * sizeof(float));
        s->prev_len = carry_len;
        return nf;
    }

    if (num_frames > 0) {
        const int nf = num_frames;
        memset(y, 0, (size_t)s->out_c * nf * sizeof(float));
        // conv accumulate (from 0), candle order sum_kk( sum_ic ), vectorized
        // over the time axis f (contiguous NEON AXPY per (oc,kk,ic)).
        for (int oc = 0; oc < s->out_c; ++oc) {
            const int gbase = (oc / s->cout_g) * s->cin_g;
            float *yrow = y + (size_t)oc * nf;
            for (int kk = 0; kk < s->ksize; ++kk) {
                const int base = kk * dil;             // logical pos = f*stride + base
                for (int ic = 0; ic < s->cin_g; ++ic) {
                    const float wv = mimi_weight_load_f32(
                        s->w, (static_cast<uint64_t>(oc) * s->cin_g + ic) *
                                  s->ksize + kk);
                    const int lic = gbase + ic;
                    if (stride == 1) {
                        // p = f + base; contiguous run split at the prev/xs seam.
                        int fs = s->prev_len - base;   // first f whose p is in xs
                        if (fs < 0) fs = 0;
                        if (fs > nf) fs = nf;
                        if (fs > 0)
                            vaxpy(yrow, s->prev + (size_t)lic * s->carry_cap + base, wv, fs);
                        if (fs < nf)
                            vaxpy(yrow + fs, xs + (size_t)lic * n_in, wv, nf - fs);
                    } else {
                        // stride>1 (never on the decode path): stage strided reads
                        // into a stack tile, then NEON AXPY. Keeps the MAC in NEON.
                        float tile[256];
                        for (int f0 = 0; f0 < nf; f0 += 256) {
                            int m = nf - f0 < 256 ? nf - f0 : 256;
                            for (int j = 0; j < m; ++j)
                                tile[j] = conv_read(s, xs, n_in, lic, (f0 + j) * stride + base);
                            vaxpy(yrow + f0, tile, wv, m);
                        }
                    }
                }
            }
            if (s->bias)
                vadd_scalar(yrow, mimi_weight_load_f32(s->bias, oc), nf);
        }
        // new carry = logical[num_frames*stride .. seq_len]  (length < kernel_eff)
        const int offset = num_frames * stride;
        const int carry_len = seq_len - offset;
        for (int c = 0; c < s->in_c; ++c) {
            float *dst = s->cbuf + (size_t)c * s->carry_cap;
            for (int j = 0; j < carry_len; ++j)
                dst[j] = conv_read(s, xs, n_in, c, offset + j);
        }
        for (int c = 0; c < s->in_c; ++c)
            memcpy(s->prev + (size_t)c * s->carry_cap,
                   s->cbuf + (size_t)c * s->carry_cap, carry_len * sizeof(float));
        s->prev_len = carry_len;
        return nf;
    } else {
        // priming: emit nothing, carry the whole logical sequence (Rust:
        // state_prev_xs = cat2(prev, xs); ys = empty). seq_len < kernel_eff.
        for (int c = 0; c < s->in_c; ++c) {
            float *dst = s->cbuf + (size_t)c * s->carry_cap;
            for (int j = 0; j < seq_len; ++j)
                dst[j] = conv_read(s, xs, n_in, c, j);
        }
        for (int c = 0; c < s->in_c; ++c)
            memcpy(s->prev + (size_t)c * s->carry_cap,
                   s->cbuf + (size_t)c * s->carry_cap, seq_len * sizeof(float));
        s->prev_len = seq_len;
        return 0;
    }
}

/* ======================================================================== *
 *  2. StreamableConvTranspose1d  -> MimiConvTrState  (groups == 1)
 * ======================================================================== */
// Weights: "<prefix>.convtr.convtr.weight" [in_c, out_c, k] (+ ".bias"
// [out_c]). Pass `prefix` = the StreamableConvTranspose1d node, e.g.
// "decoder.model.2". (NormConvTranspose1d owns the ".convtr.convtr" nesting.)
//
// Per step (Rust conv.rs:448-501):
//   raw = conv_transpose1d(xs) + bias         // len ot = (n_in-1)*stride + k
//   overlap-add carry: combined[0..pt] = raw[0..pt] + (prev_ys - bias)
//   emit = combined[0 .. ot-invalid]  (= n_in*stride samples)
//   new carry prev_ys = combined[ot-invalid .. ot]  (invalid = k - stride)
//
// Vectorization: the transposed conv's in-channel reduction is done as a NEON
// AXPY over the CONTIGUOUS input-time axis. For fixed (oc,kk) we build
//   g[l] = sum_ic X[ic,l] * w[ic,oc,kk]   (l = 0..n_in)
// by accumulating, for each ic, wcoef * X[ic, :]  (X[ic,:] is contiguous in
// [C,T]). g is then scattered to output positions l*stride+kk. This keeps the
// heavy in_c reduction in NEON with NO weight repack. (A per-kk matrix pass over
// mimi_gemm_f32 is the perf-tier alternative; it needs an init-time weight
// transpose and changes the summation order, so it is deferred behind the
// parity gate.) The scatter/bias/overlap are the elementwise NEON sweeps below.
//
// INVARIANT: prev holds exactly `invalid = k - stride` output steps (WITH bias)
// once primed (prev_valid). Emits n_in*stride samples/step. causal is stored
// but unused in step (the Rust step never reads self.causal; the causal trim is
// implicit in the invalid-steps split, trim_right_ratio == 1).
struct MimiConvTrState {
    int in_c, out_c, ksize, stride, causal;
    int invalid;         // ksize - stride  (carry length once primed)
    const uint8_t *w;    // [in_c, out_c, ksize] f32 bytes
    const uint8_t *bias; // [out_c] f32 bytes or NULL
    float *prev;         // [out_c, invalid] output overlap carry (bias INCLUDED)
    float *carry_scratch;// [out_c, invalid] next-carry accumulator
    float *g;            // [MIMI_CONV_MAX_NIN] per-(oc,kk) time reduction
    int prev_valid;
    // Matrix route (short-time wide layers): G[k*out_c, n] = W_r x X in one pass
    // (X is the caller's [in_c, n] — zero-copy B), then the same per-oc
    // scatter/bias/overlap/commit. The GEMM reads checkpoint [ic][oc][kk]
    // directly as the transpose of [ic, oc*kk]; no re-arm exists.
    int route_gemm;
    float *g_gemm;       // [ksize*out_c, MIMI_CONV_GEMM_MAX_N]
};

int mimi_convtr_init(MimiConvTrState **st, const MimiWeightTable *w,
                     const char *prefix, int in_c, int out_c, int ksize,
                     int stride, int causal, MimiArena *a,
                     char *err, size_t errlen) {
    if (ksize < stride) return fail(err, errlen, "convtr: kernel < stride");
    MimiConvTrState *s = (MimiConvTrState *)mimi_arena_alloc(a, sizeof(MimiConvTrState));
    memset(s, 0, sizeof(*s));
    s->in_c = in_c; s->out_c = out_c; s->ksize = ksize; s->stride = stride;
    s->causal = causal;
    s->invalid = ksize - stride;

    char base[256];
    snprintf(base, sizeof(base), "%s.convtr.convtr", prefix);
    s->w = resolve_weight(w, base, in_c, out_c, ksize, a, err, errlen);  // dim0=in_c
    if (!s->w) return 1;
    const MimiWeight *b = wfind(w, base, ".bias");
    if (b) {
        if (!b->bytes || !b->shape || b->ndim != 1 ||
            b->shape[0] != (uint64_t)out_c || b->len != (uint64_t)out_c)
            return fail(err, errlen, "convtr bias missing data or length mismatch");
        s->bias = b->bytes;
    }
    const int inv = s->invalid > 0 ? s->invalid : 1;   // avoid 0-byte alloc
    s->prev = (float *)mimi_arena_alloc(a, (size_t)out_c * inv * sizeof(float));
    s->carry_scratch = (float *)mimi_arena_alloc(a, (size_t)out_c * inv * sizeof(float));
    s->g = (float *)mimi_arena_alloc(a, (size_t)MIMI_CONV_MAX_NIN * sizeof(float));
    s->prev_valid = 0;
    // Matrix route for the wide layers (the ratio-8 convtr receives n=2: its
    // time-axis AXPY was all scalar tail — the measured ~31 ms hot spot).
    s->route_gemm = (in_c >= 128);
    if (s->route_gemm) {
        s->g_gemm = (float *)mimi_arena_alloc(
            a, (size_t)ksize * out_c * MIMI_CONV_GEMM_MAX_N * sizeof(float));
    }
    *st = s;
    return 0;
}

void mimi_convtr_reset(MimiConvTrState *s) { s->prev_valid = 0; }

int mimi_convtr_step(MimiConvTrState *s, const float *xs, int n_in, float *y) {
    // ABI bounds (review P1): g/g_gemm scratch would overrun past
    // MIMI_CONV_MAX_NIN. 0 in = 0 out (empty StreamTensor propagation).
    if (n_in <= 0) return 0;
    if (n_in > MIMI_CONV_MAX_NIN) return -1;
    const int stride = s->stride, k = s->ksize, oc_n = s->out_c, in_c = s->in_c;
    const int emit_len = n_in * stride;
    const int invalid = s->invalid;                 // ot = emit_len + invalid
    const int inv_row = invalid > 0 ? invalid : 1;
    float *g = s->g;                                // per-lane-private (see NOTES g)

    // Matrix route: the whole (kk,oc) x ic reduction as one GEMM up front —
    // G[kk*oc_n + oc, l] = sum_ic W_r[kk*oc_n+oc, ic] * X[ic, l].
    const int use_gemm = s->route_gemm && n_in <= MIMI_CONV_GEMM_MAX_N;
    if (use_gemm)
        mimi_weight_gemm_tn_f32(s->w, xs, s->g_gemm, k * oc_n, in_c, n_in);

    // LANE-BAND AXIS = output channel oc: this loop body is fully independent
    // per oc (reduce -> scatter -> bias -> overlap -> commit its own carry row),
    // so the arbiter can cut [0,oc_n) into bands with no cross-oc dependence.
    for (int oc = 0; oc < oc_n; ++oc) {
        float *yrow = y + (size_t)oc * emit_len;
        float *crow = s->carry_scratch + (size_t)oc * inv_row;
        memset(yrow, 0, (size_t)emit_len * sizeof(float));
        if (invalid > 0) memset(crow, 0, (size_t)invalid * sizeof(float));

        // per kk: g[l] = sum_ic X[ic,l]*w[ic,oc,kk], then scatter to
        // out_t = l*stride+kk. GEMM route reads the SIMD-computed rows;
        // NEON route accumulates over the contiguous time axis. Per fixed oc
        // the kk-ascending scatter order matches candle either way.
        for (int kk = 0; kk < k; ++kk) {
            const float *grow;
            if (use_gemm) {
                grow = s->g_gemm + (((size_t)oc * k) + kk) * n_in;
            } else {
                memset(g, 0, (size_t)n_in * sizeof(float));
                for (int ic = 0; ic < in_c; ++ic)
                    vaxpy(g, xs + (size_t)ic * n_in,
                          mimi_weight_load_f32(
                              s->w,
                              (static_cast<uint64_t>(ic) * oc_n + oc) * k + kk),
                          n_in);
                grow = g;
            }
            for (int l = 0; l < n_in; ++l) {
                const int out_t = l * stride + kk;
                if (out_t < emit_len) yrow[out_t] += grow[l];
                else                  crow[out_t - emit_len] += grow[l];
            }
        }
        // bias broadcast-added to every raw position (emit region + tail carry).
        if (s->bias) {
            const float bias = mimi_weight_load_f32(s->bias, oc);
            vadd_scalar(yrow, bias, emit_len);
            if (invalid > 0) vadd_scalar(crow, bias, invalid);
        }
        // overlap-add prior carry (bias removed) into positions [0, invalid).
        // decode has invalid==stride<=emit_len so the whole overlap lands in y.
        if (s->prev_valid && invalid > 0) {
            const float bv = s->bias ? mimi_weight_load_f32(s->bias, oc) : 0.f;
            const float *prow = s->prev + (size_t)oc * invalid;
            const int nov = invalid < emit_len ? invalid : emit_len;
            voverlap(yrow, prow, bv, nov);
            for (int t = nov; t < invalid; ++t)     // rare tail (invalid>emit_len)
                crow[t - emit_len] += prow[t] - bv;
        }
        // commit this oc's carry row (kept for the next step, bias included).
        if (invalid > 0) memcpy(s->prev + (size_t)oc * invalid, crow, (size_t)invalid * sizeof(float));
    }
    s->prev_valid = 1;
    return emit_len;
}

/* ======================================================================== *
 *  3. ConvTrUpsample1d  -> MimiUpsampleState   (depthwise, groups == dim)
 * ======================================================================== */
// ConvTrUpsample1d: stride 2, dim 512, k=4, causal, learnt, NO bias, norm None.
// groups == out_c == in_c, so NormConvTranspose1d expands the stored [dim,1,k]
// weight to block-diagonal [dim,dim,k] via an identity multiply and runs
// groups=1 — i.e. a DEPTHWISE transposed conv: output channel c depends only on
// input channel c with kernel w[c,0,:]. We keep [dim,1,k] (no expansion).
//
// Vectorization: with n_in==1 (decode: one latent frame/step) the input column
// X[:,0] is CONTIGUOUS over channels, so for each tap kk we form the channel
// four tap vectors directly from checkpoint [channel, tap] using vld4q, then
// multiply them by X[:,0]. No weight transpose or repack exists. n_in>1 (not
// on the decode path) falls back to a per-channel NEON scale over time.
// Weight name (single instance): "upsample.convtr.convtr.convtr.weight".
struct MimiUpsampleState {
    int dim, ksize, stride, invalid;
    const uint8_t *w;    // [dim, 1, ksize] checkpoint-layout f32 bytes
    float *prev;         // [dim, invalid] overlap carry (no bias)
    float *carry_scratch;// [dim, invalid]
    float *g;            // [MIMI_CONV_MAX_NIN] n_in>1 fallback time reduction
    int prev_valid;
};

int mimi_upsample_init(MimiUpsampleState **st, const MimiWeightTable *w,
                       MimiArena *a, char *err, size_t errlen) {
    const int dim = MIMI_DIM, stride = MIMI_UPSAMPLE_STRIDE;
    const int ksize = 2 * stride;                   // ConvTrUpsample1d: k = 2*stride
    MimiUpsampleState *s = (MimiUpsampleState *)mimi_arena_alloc(a, sizeof(MimiUpsampleState));
    memset(s, 0, sizeof(*s));
    s->dim = dim; s->ksize = ksize; s->stride = stride;
    s->invalid = ksize - stride;
    const MimiWeight *ww = mimi_weight_find(w, "upsample.convtr.convtr.convtr.weight");
    if (!ww) return fail(err, errlen, "upsample weight not found");
    // Exact-shape + data validation (review P2: element count alone let a
    // null span reach the direct byte-load loop): [MIMI_DIM, 1, 2*stride],
    // data non-null.
    if (!wcheck3(ww, dim, 1, ksize))
        return fail(err, errlen,
                    "upsample weight shape mismatch (expect [dim,1,2*stride] with data)");
    s->w = ww->bytes;
    s->prev = (float *)mimi_arena_alloc(a, (size_t)dim * s->invalid * sizeof(float));
    s->carry_scratch = (float *)mimi_arena_alloc(a, (size_t)dim * s->invalid * sizeof(float));
    s->g = (float *)mimi_arena_alloc(a, (size_t)MIMI_CONV_MAX_NIN * sizeof(float));
    s->prev_valid = 0;
    *st = s;
    return 0;
}

void mimi_upsample_reset(MimiUpsampleState *s) { s->prev_valid = 0; }

int mimi_upsample_step(MimiUpsampleState *s, const float *xs, int n_in, float *y) {
    // ABI bounds (review P1): the n_in>1 fallback's g scratch would overrun
    // past MIMI_CONV_MAX_NIN. 0 in = 0 out (empty StreamTensor propagation).
    if (n_in <= 0) return 0;
    if (n_in > MIMI_CONV_MAX_NIN) return -1;
    const int stride = s->stride, k = s->ksize, dim = s->dim, invalid = s->invalid;
    const int emit_len = n_in * stride;             // ot = emit_len + invalid

    if (n_in == 1) {
        // BAND AXIS = channel. X[:,0] is contiguous over channels, so per tap kk
        // vld4 deinterleaves checkpoint [channel, tap] in registers. With
        // n_in==1 the k taps map bijectively onto the emit_len + invalid output
        // positions, so each y/carry slot is written ONCE (direct assign, no
        // pre-zero) -> per-channel rows are self-contained for lane banding.
#if MIMI_NEON
        if (k == 4 && emit_len == 2 && invalid == 2) {
            for (int c = 0; c < dim; c += 4) {
                const uint8_t *rows = s->w + (size_t)c * 4 * sizeof(float);
                float32x4_t row0 = vreinterpretq_f32_u8(vld1q_u8(rows));
                float32x4_t row1 = vreinterpretq_f32_u8(vld1q_u8(rows + 16));
                float32x4_t row2 = vreinterpretq_f32_u8(vld1q_u8(rows + 32));
                float32x4_t row3 = vreinterpretq_f32_u8(vld1q_u8(rows + 48));
                const float32x4x2_t pair01 = vtrnq_f32(row0, row1);
                const float32x4x2_t pair23 = vtrnq_f32(row2, row3);
                float32x4_t taps0 = vcombine_f32(vget_low_f32(pair01.val[0]),
                                                 vget_low_f32(pair23.val[0]));
                float32x4_t taps1 = vcombine_f32(vget_low_f32(pair01.val[1]),
                                                 vget_low_f32(pair23.val[1]));
                float32x4_t taps2 = vcombine_f32(vget_high_f32(pair01.val[0]),
                                                 vget_high_f32(pair23.val[0]));
                float32x4_t taps3 = vcombine_f32(vget_high_f32(pair01.val[1]),
                                                 vget_high_f32(pair23.val[1]));
                const float32x4_t input = vld1q_f32(xs + c);
                float32x4x2_t emit;
                emit.val[0] = vmulq_f32(input, taps0);
                emit.val[1] = vmulq_f32(input, taps1);
                vst2q_f32(y + (size_t)c * 2, emit);
                float32x4x2_t carry;
                carry.val[0] = vmulq_f32(input, taps2);
                carry.val[1] = vmulq_f32(input, taps3);
                vst2q_f32(s->carry_scratch + (size_t)c * 2, carry);
            }
        } else
#elif MIMI_SSE2
        if (k == 4 && emit_len == 2 && invalid == 2) {
            for (int c = 0; c < dim; c += 4) {
                const uint8_t *rows = s->w + (size_t)c * 4 * sizeof(float);
                __m128 taps0, taps1, taps2, taps3;
                memcpy(&taps0, rows, sizeof(taps0));
                memcpy(&taps1, rows + 16, sizeof(taps1));
                memcpy(&taps2, rows + 32, sizeof(taps2));
                memcpy(&taps3, rows + 48, sizeof(taps3));
                _MM_TRANSPOSE4_PS(taps0, taps1, taps2, taps3);
                const __m128 input = _mm_loadu_ps(xs + c);
                const __m128 emit0 = _mm_mul_ps(input, taps0);
                const __m128 emit1 = _mm_mul_ps(input, taps1);
                _mm_storeu_ps(y + (size_t)c * 2, _mm_unpacklo_ps(emit0, emit1));
                _mm_storeu_ps(y + (size_t)c * 2 + 4,
                              _mm_unpackhi_ps(emit0, emit1));
                const __m128 carry0 = _mm_mul_ps(input, taps2);
                const __m128 carry1 = _mm_mul_ps(input, taps3);
                _mm_storeu_ps(s->carry_scratch + (size_t)c * 2,
                              _mm_unpacklo_ps(carry0, carry1));
                _mm_storeu_ps(s->carry_scratch + (size_t)c * 2 + 4,
                              _mm_unpackhi_ps(carry0, carry1));
            }
        } else
#endif
        {
            for (int c = 0; c < dim; ++c) {
                const float input = xs[c];
                for (int kk = 0; kk < k; ++kk) {
                    const float product = input * mimi_weight_load_f32(
                        s->w, static_cast<uint64_t>(c) * k + kk);
                    if (kk < emit_len) y[(size_t)c * emit_len + kk] = product;
                    else s->carry_scratch[(size_t)c * invalid + kk - emit_len] = product;
                }
            }
        }
    } else {
        // n_in>1 (off decode path): per channel, NEON scale over the time axis.
        memset(y, 0, (size_t)dim * emit_len * sizeof(float));
        memset(s->carry_scratch, 0, (size_t)dim * invalid * sizeof(float));
        float *g = s->g;
        for (int c = 0; c < dim; ++c) {
            for (int kk = 0; kk < k; ++kk) {
                vscale(g, xs + (size_t)c * n_in,
                       mimi_weight_load_f32(
                           s->w, static_cast<uint64_t>(c) * k + kk),
                       n_in);
                float *yrow = y + (size_t)c * emit_len;
                float *crow = s->carry_scratch + (size_t)c * invalid;
                for (int l = 0; l < n_in; ++l) {
                    const int out_t = l * stride + kk;
                    if (out_t < emit_len) yrow[out_t] += g[l];
                    else                  crow[out_t - emit_len] += g[l];
                }
            }
        }
    }
    // no bias. overlap-add previous carry into [0, invalid).
    if (s->prev_valid) {
        for (int c = 0; c < dim; ++c) {
            const float *prow = s->prev + (size_t)c * invalid;
            float *yrow = y + (size_t)c * emit_len;
            float *crow = s->carry_scratch + (size_t)c * invalid;
            const int nov = invalid < emit_len ? invalid : emit_len;
            voverlap(yrow, prow, 0.f, nov);
            for (int t = nov; t < invalid; ++t) crow[t - emit_len] += prow[t];
        }
    }
    memcpy(s->prev, s->carry_scratch, (size_t)dim * invalid * sizeof(float));
    s->prev_valid = 1;
    return emit_len;
}

/* ========================================================================= *
 * NOTES
 * ========================================================================= *
 *
 * (a) Rust fn -> C++ fn mapping
 *   conv1d_weight_norm / NormConvTranspose1d WeightNorm branch
 *        -> weight_norm_fold + resolve_weight   (fold ONCE at init into arena)
 *   candle Conv1d::forward (conv1d + broadcast_add bias)
 *        -> mimi_conv_step (accumulate from 0, +bias last)
 *   candle conv_transpose1d + bias
 *        -> the g-reduction + scatter in mimi_convtr_step / mimi_upsample_step
 *   StreamableConv1d::step                 -> mimi_conv_step   / _init / _reset
 *   StreamableConvTranspose1d::step        -> mimi_convtr_step / _init / _reset
 *   ConvTrUpsample1d::step (depthwise)     -> mimi_upsample_step / _init / _reset
 *   NOT ported (not in ABI / decode path): Module::forward padding branches
 *     (get_extra_padding_for_conv1d, pad1d symmetric, unpad1d), ConvDownsample1d,
 *     reset_batch_idx, all StreamMask where_cond branches (mask is always None
 *     for batch==1 so they are identities), SpectralNorm/TimeGroupNorm (bail).
 *
 * (b) Per-struct pending/carry invariants (verify these hardest)
 *   MimiConvState (StreamableConv1d):
 *     * prev = [in_c, carry_cap] left-context carry; prev_len valid time steps.
 *       INVARIANT: prev_len < kernel_eff always, so carry_cap = kernel_eff =
 *       (ksize-1)*dilation+1 never overflows. Proof: num_frames =
 *       (seq_len+stride-kernel_eff)/stride (floor, 0 when seq_len+stride<ke);
 *       retained carry = seq_len - num_frames*stride is in [ke-stride, ke) if
 *       num_frames>0 else = seq_len (< ke). First-step left pad = ke-stride.
 *     * left_pad_applied: false only before the first step; the first step
 *       pre-loads prev with padding_total zeros (PadMode::Constant), reproducing
 *       cat2(empty, pad1d(xs, padding_total, 0)).
 *     * logical sequence per step = [prev(prev_len) ++ xs(n_in)]; output frame f
 *       channel oc reads taps at logical pos f*stride + kk*dilation. Emits
 *       [out_c, num_frames] (0 while priming). New carry =
 *       logical[num_frames*stride .. seq_len]; when num_frames==0 the WHOLE
 *       logical sequence is retained (Rust: state_prev_xs = cat2(prev,xs)).
 *     * DECODE FACT: every decoder StreamableConv1d has stride==1 (upsampling is
 *       done only by transposed convs), so num_frames = n_in each step and the
 *       priming (0-out) branch never actually fires in the decode graph — it is
 *       implemented faithfully for the general stride>1 case regardless.
 *   MimiConvTrState (StreamableConvTranspose1d, groups==1):
 *     * prev = [out_c, invalid] output-overlap carry WITH bias, invalid =
 *       ksize-stride. prev_valid false only before the first step. Emits
 *       emit_len = n_in*stride samples/step; ot = emit_len + invalid raw samples
 *       are produced, of which the last `invalid` become the next carry.
 *     * overlap-add: emitted[0..invalid] = raw[0..invalid] + (prev - bias); bias
 *       is subtracted from prev because the current raw re-adds it (Rust "Remove
 *       the bias as it will be applied multiple times"). Tail carry keeps bias.
 *   MimiUpsampleState (ConvTrUpsample1d): same overlap-add carry as convtr but
 *     depthwise (out ch c <- in ch c only), NO bias, dim=512, stride=2, k=4,
 *     invalid=2. Each latent frame (n_in=1) -> emit_len=2 upsampled frames.
 *
 * (c) Weight names + shapes consumed (verified vs the moshiko-candle-bf16
 *     checkpoint tokenizer-e351c8d8-checkpoint125.safetensors):
 *     conv1d   "<prefix>.conv.conv.weight"   [out_c, in_c/groups, ksize] f32
 *              "<prefix>.conv.conv.bias"      [out_c] (optional)
 *              e.g. decoder.model.0.conv.conv.weight [1024,512,7], .bias [1024]
 *     convtr   "<prefix>.convtr.convtr.weight" [in_c, out_c, ksize] f32
 *              "<prefix>.convtr.convtr.bias"   [out_c] (optional)
 *              e.g. decoder.model.2.convtr.convtr.weight [1024,512,16], .bias[512]
 *     upsample "upsample.convtr.convtr.convtr.weight" [512,1,4] f32, NO bias.
 *     0 weight_g/weight_v tensors in this checkpoint (pre-folded "weight"); the
 *     fold path is implemented but not exercised here. Prefix contract: caller
 *     passes the *streamable module* node (e.g. "decoder.model.0"); this unit
 *     appends the ".conv.conv"/".convtr.convtr" inner nesting, matching the
 *     moshi VarBuilder pp() chain exactly (upsample name is hardcoded, single
 *     instance, no prefix in the ABI).
 *
 * (d) Weight-norm fold formula and dims
 *     weight = weight_v * weight_g / ||weight_v||_2, norm over dims (1,2)
 *     (keepdim) with weight_g/norm broadcast over dim0:
 *       conv1d : dim0 = out_c, norm over (in_c/groups, ksize) per output channel.
 *       convtr : dim0 = in_c,  norm over (out_c, ksize) per INPUT channel.
 *     Folded once into arena buffers (weight_norm_fold, n0=dim0, rest = product
 *     of the remaining dims). ||.||_2 = sqrtf(sum of squares), f32.
 *
 * (e) Accumulation order + vectorization axis (f32 throughout; faithful tier)
 *     conv1d  : VECTORIZE OVER TIME (output frame axis f). Per output (oc,f) the
 *               sum is over taps kk (outer) then in-channels ic (inner):
 *               y = sum_kk( sum_ic in*w ), each (kk,ic) contribution a contiguous
 *               NEON vfmaq AXPY across all f at once. Because f is the innermost
 *               (vectorized) loop, the per-element (kk,ic) summation order is
 *               UNCHANGED vs candle's for-kk/vec_dot-ic. Bias broadcast-added
 *               last (candle_nn Conv1d::forward). stride==1 loads contiguously;
 *               stride>1 stages a strided tile then AXPYs (MAC stays NEON).
 *     convtr  : VECTORIZE OVER TIME (input frame axis l). For each (kk,oc):
 *               g[l] = sum_ic X[ic,l]*w[ic,oc,kk] accumulated by a NEON AXPY of
 *               each contiguous input row X[ic,:] scaled by one weight; ic
 *               ascending == candle vec_dot order. g scattered to out_t=l*stride+kk
 *               in kk-outer order == candle. bias/overlap are NEON sweeps.
 *     upsample: VECTORIZE OVER CHANNELS (n_in==1: X[:,0] contiguous). Per tap kk,
 *               checkpoint taps are deinterleaved in registers (NEON vld4),
 *               multiplied by X and scattered to out_t=kk. Depthwise
 *               has no in-channel reduction; each out sample = sum over the (<=2)
 *               taps hitting it, added in kk-outer order == candle. (n_in>1 falls
 *               back to a per-channel NEON scale over the time axis.)
 *
 * (f) Lane-banding (kcoro engine integration; banding is the arbiter's cut)
 *     Every step's per-output work is independent — no cross-output-channel
 *     sequential dependence — so the arbiter can split each conv across lanes
 *     as channel/row bands. Natural band axis + state disposition per conv:
 *     conv1d   : BAND AXIS = output channel oc (outer loop; each oc writes its
 *                own y[oc,:] row from read-only prev/xs/weights). The NEON axis
 *                is TIME (inner) — orthogonal to the band axis. Carry-update is a
 *                second phase banded over INPUT channel c (prev[c]/cbuf[c] rows
 *                disjoint). SHARED (read-only in compute): w, bias, prev, xs.
 *                PRIVATE: none — cbuf writes are per-input-channel disjoint, no
 *                cross-lane scratch. reset flags are step-boundary only.
 *     convtr   : BAND AXIS = output channel oc (outer loop; self-contained
 *                reduce->scatter->bias->overlap->commit per oc). NEON axis =
 *                TIME l (inner). PRIVATE per lane: the g[MIMI_CONV_MAX_NIN]
 *                reduction scratch (reused per oc) — one g per lane, or hoist g
 *                into per-lane scratch at the banding cut. SHARED: w, bias, xs;
 *                prev/carry_scratch rows are per-oc disjoint (shareable). Each
 *                oc reads only its own prev[oc] row and commits it, so the carry
 *                needs no cross-lane sync.
 *     upsample : BAND AXIS = channel c (depthwise: channel c is fully
 *                independent) — this COINCIDES with the NEON axis, so a lane
 *                band is a contiguous channel sub-range and vmul/vscale operate
 *                on that sub-range directly. PRIVATE: prod[dim] (per-channel
 *                disjoint writes — shareable. SHARED: checkpoint weights,
 *                xs; prev/carry_scratch rows per-channel disjoint.
 *     In all three the single-call step API is unchanged; per-lane privatization
 *     is limited to the convtr `g` (and optionally upsample `prod`) scratch.
 *
 * (g) Uncertainties / friction
 *   1. ABI friction (documented, not forked): mimi_conv_init/mimi_convtr_init
 *      take a *streamable node* prefix and this unit appends ".conv.conv" /
 *      ".convtr.convtr". If the arbiter instead intends prefix == the direct
 *      weight parent, drop the two snprintf(base, "%s.conv.conv"/"...") lines.
 *   2. Scalar code lives ONLY in: the MIMI_SCALAR_REF build (every vNNN helper
 *      degrades to a scalar loop -> that build IS the `_ref` parity sibling for
 *      bisecting), sub-vector tail remainders inside the NEON helpers, and pure
 *      activation/carry marshalling (strided gathers and scatters). Resident
 *      weights are never staged, repacked, transposed, widened, or aligned.
 *   3. Numerics are the *faithful* tier (ulp band), not bit-exact: NEON's 4-lane
 *      grouping of the AXPY accumulation differs from candle's f32 vec_dot lane
 *      order; the harness measures the band. A per-kk byte-load SIMD GEMM
 *      for convtr is the perf-tier alternative — deferred: it needs an init-time
 *      weight transpose (+memory) and reorders the K reduction.
 *   4. PadMode: only Constant (zeros) is implemented — the sole decode-path mode.
 *      Replicate (ConvDownsample1d, encode-only) is not needed here.
 *   5. `causal` is stored but never read in either step, matching the Rust step
 *      functions (both are inherently causal: conv1d left-pads only; convtr's
 *      trim is the invalid-steps split, trim_right_ratio == 1).
 *   6. Buffers are tiny: conv1d prev/cbuf are O(in_c * kernel_eff); convtr/
 *      upsample carries are O(channels * invalid); the only time-sized scratch
 *      is g[MIMI_CONV_MAX_NIN] (convtr/upsample n_in>1). Outputs go straight to
 *      the caller's y, so no worst-case frame-sized arena is needed.
 */
