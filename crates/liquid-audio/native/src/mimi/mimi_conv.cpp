// mimi_conv.cpp — faithful C++/NEON port of moshi 0.6.4 streaming conv
// primitives (crates.io moshi-0.6.4/src/conv.rs), decoder path only.
//
// Ported (see /* NOTES */ at end for the full Rust->C++ map):
//   NormConv1d / NormConvTranspose1d forward MATH (the raw candle
//     conv1d / conv_transpose1d the streaming steps call) + WeightNorm fold.
//   StreamableConv1d::step        -> mimi_conv_step
//   StreamableConvTranspose1d::step-> mimi_convtr_step
//   ConvTrUpsample1d::step         -> mimi_upsample_step (depthwise, groups=dim)
//
// Deliberately NOT ported (out of the decode hot path / manifest SKIP):
//   the non-streaming Module::forward padding (get_extra_padding_for_conv1d,
//     symmetric pad, unpad1d) — not in the ABI, decode uses step only.
//   ConvDownsample1d (encode-only), batched StreamMask (batch==1: mask is
//     always None here, so the mask where_cond branches collapse to identity).
//
// Layout: conv layout [C, T], channel-major, T contiguous, batch==1.
// f32 in, f32 accumulate. All state is POD, carved once from the arena;
// steady state never allocates.

#include "mimi_kernel.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* ======================================================================== *
 *  small helpers
 * ======================================================================== */

// y[i] += w * x[i], i in 0..n. Contiguous AXPY (conv1d stride-1 hot path).
static inline void axpy_f32(float *y, const float *x, float w, int n) {
#if defined(__aarch64__) && !defined(MIMI_SCALAR_REF)
    float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t yv = vld1q_f32(y + i);
        yv = vfmaq_f32(yv, vld1q_f32(x + i), wv);
        vst1q_f32(y + i, yv);
    }
    for (; i < n; ++i) y[i] += w * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] += w * x[i];
#endif
}

// strided dot: sum_i a[i*sa] * b[i*sb], i in 0..n. Order = ascending i
// (matches candle's vec_dot over c_in). Used by the transposed-conv scatter.
static inline float dot_strided(const float *a, int sa, const float *b, int sb, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i * sa] * b[i * sb];
    return s;
}

static int fail(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return 1;
}

// Look up "<prefix><suffix>" in the weight table. NULL if absent.
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
 *  conv1d weight_v shape [out_c, in_c/groups, k], weight_g [out_c,1,1]:
 *     norm is over (in,k) per output channel oc.
 *  convtr weight_v shape [in_c, out_c/groups, k], weight_g [in_c,1,1]:
 *     norm is over (out,k) per INPUT channel ic (dim0 is in_c for convtr).
 *
 *  This checkpoint (kyutai moshiko-candle-bf16) stores the already-folded
 *  "weight" (0 weight_g/weight_v tensors), so the fold path is exercised only
 *  if a future export ships raw weight_g/weight_v. Fold happens ONCE here.
 * ------------------------------------------------------------------------ */
// n0 = size of dim0 (the broadcast axis of weight_g / norm), rest = dim1*dim2.
static void weight_norm_fold(const float *v, const float *g, float *out,
                             int n0, int rest) {
    for (int c = 0; c < n0; ++c) {
        const float *vr = v + (size_t)c * rest;
        float ss = 0.f;
        for (int i = 0; i < rest; ++i) ss += vr[i] * vr[i];
        float scale = g[c] / sqrtf(ss);
        float *orow = out + (size_t)c * rest;
        for (int i = 0; i < rest; ++i) orow[i] = vr[i] * scale;
    }
}

// Resolve the (possibly weight-normed) weight tensor under `prefix`+`stem`.
// If "<...>.weight" exists -> zero-copy pointer. Else fold weight_g/weight_v
// into the arena. `n0` is the fold broadcast dim (out_c for conv1d, in_c for
// convtr). Returns pointer or NULL (+ err) on missing/misshaped.
static const float *resolve_weight(const MimiWeightTable *w, const char *base,
                                   int n0, int rest, MimiArena *a,
                                   char *err, size_t errlen) {
    const MimiWeight *ww = wfind(w, base, ".weight");
    if (ww) {
        if (ww->len != (uint64_t)n0 * (uint64_t)rest) {
            fail(err, errlen, "conv weight has unexpected element count");
            return NULL;
        }
        return ww->data;
    }
    const MimiWeight *vg = wfind(w, base, ".weight_g");
    const MimiWeight *vv = wfind(w, base, ".weight_v");
    if (!vg || !vv) {
        fail(err, errlen, "conv weight (nor weight_g/weight_v) not found");
        return NULL;
    }
    if (vv->len != (uint64_t)n0 * (uint64_t)rest || vg->len != (uint64_t)n0) {
        fail(err, errlen, "conv weight_v/weight_g shape mismatch");
        return NULL;
    }
    float *folded = (float *)mimi_arena_alloc(a, (size_t)n0 * rest * sizeof(float));
    weight_norm_fold(vv->data, vg->data, folded, n0, rest);
    return folded;
}

/* ======================================================================== *
 *  1. StreamableConv1d  -> MimiConvState
 * ======================================================================== */
// Weights: "<prefix>.conv.conv.weight" [out_c, in_c/groups, k] (+ ".bias"
// [out_c]) — the StreamableConv1d owns the ".conv.conv" nesting (NormConv1d ->
// Conv1d). Pass `prefix` = the StreamableConv1d node, e.g. "decoder.model.0".
//
// Streaming state (Rust: state_prev_xs + left_pad_applied). We NEVER
// materialise the cat(prev_xs, xs) window; conv output element (oc,f) gathers
// its taps from a logical sequence [prev ++ xs] read on the fly. prev holds
// the left-context carry in [C, carry_cap] layout.
//
// INVARIANT (proven from the Rust arithmetic, both padding modes of step):
//   after every step, prev_len < kernel_eff. Because num_frames =
//   (seq_len+stride-kernel_eff)/stride and the retained carry length
//   seq_len - num_frames*stride is in [kernel_eff-stride, kernel_eff) when
//   num_frames>0, and equals seq_len (< kernel_eff) when num_frames==0.
//   Also the first-step left pad = padding_total = kernel_eff-stride
//   < kernel_eff. So carry_cap = kernel_eff bounds prev for all time.
struct MimiConvState {
    int in_c, out_c, ksize, stride, dilation, groups;
    int causal;          // stored for parity; step is inherently causal (Rust
                         // step left-pads only and never reads self.causal)
    int kernel_eff;      // (ksize-1)*dilation + 1
    int padding_total;   // kernel_eff - stride  (first-step left pad, Constant)
    int cin_g;           // in_c / groups
    int cout_g;          // out_c / groups
    int carry_cap;       // = kernel_eff  (>= max carry, >= padding_total)
    const float *w;      // [out_c, in_c/groups, ksize], folded, checkpoint layout
    const float *bias;   // [out_c] or NULL
    float *prev;         // [in_c, carry_cap] left-context carry
    float *cbuf;         // [in_c, carry_cap] scratch for the next carry gather
    int prev_len;        // # carried time steps currently in prev  (< kernel_eff)
    int left_pad_applied;
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
    int rest = s->cin_g * ksize;
    s->w = resolve_weight(w, base, out_c, rest, a, err, errlen);
    if (!s->w) return 1;
    const MimiWeight *b = wfind(w, base, ".bias");
    if (b) {
        if (b->len != (uint64_t)out_c)
            return fail(err, errlen, "conv1d bias length mismatch");
        s->bias = b->data;
    }
    s->prev = (float *)mimi_arena_alloc(a, (size_t)in_c * s->carry_cap * sizeof(float));
    s->cbuf = (float *)mimi_arena_alloc(a, (size_t)in_c * s->carry_cap * sizeof(float));
    s->prev_len = 0;
    s->left_pad_applied = 0;
    *st = s;
    return 0;
}

void mimi_conv_reset(MimiConvState *s) {
    s->prev_len = 0;
    s->left_pad_applied = 0;
}

// read logical input sample (channel c, logical time p) from [prev ++ xs].
static inline float conv_read(const MimiConvState *s, const float *xs, int n_in,
                              int c, int p) {
    if (p < s->prev_len) return s->prev[(size_t)c * s->carry_cap + p];
    return xs[(size_t)c * n_in + (p - s->prev_len)];
}

int mimi_conv_step(MimiConvState *s, const float *xs, int n_in, float *y) {
    // First step: prepend padding_total zeros on the LEFT (PadMode::Constant).
    // Modelled by pre-loading prev with padding_total zeros (cat2(empty,pad)==pad).
    if (!s->left_pad_applied) {
        s->left_pad_applied = 1;
        int pt = s->padding_total;
        for (int c = 0; c < s->in_c; ++c)
            memset(s->prev + (size_t)c * s->carry_cap, 0, pt * sizeof(float));
        s->prev_len = pt;
    }

    const int stride = s->stride, dil = s->dilation, ke = s->kernel_eff;
    const int seq_len = s->prev_len + n_in;
    const int num_frames = (seq_len + stride >= ke) ? (seq_len + stride - ke) / stride : 0;

    if (num_frames > 0) {
        const int nf = num_frames;
        // zero output [out_c, nf]
        memset(y, 0, (size_t)s->out_c * nf * sizeof(float));
        // accumulate conv (from 0), matching candle order sum_kk( sum_ic ).
        for (int oc = 0; oc < s->out_c; ++oc) {
            const int g = oc / s->cout_g;
            const int gbase = g * s->cin_g;
            float *yrow = y + (size_t)oc * nf;
            const float *wrow = s->w + (size_t)oc * s->cin_g * s->ksize;
            for (int kk = 0; kk < s->ksize; ++kk) {
                const int base = kk * dil;            // logical pos = f*stride + base
                for (int ic = 0; ic < s->cin_g; ++ic) {
                    const float wv = wrow[(size_t)ic * s->ksize + kk];
                    const int lic = gbase + ic;
                    if (stride == 1) {
                        // contiguous run in f; split at the prev/xs boundary.
                        int f_split = s->prev_len - base;      // f where p crosses into xs
                        if (f_split < 0) f_split = 0;
                        if (f_split > nf) f_split = nf;
                        if (f_split > 0) {
                            const float *src = s->prev + (size_t)lic * s->carry_cap + base;
                            axpy_f32(yrow, src, wv, f_split);
                        }
                        if (f_split < nf) {
                            const float *src = xs + (size_t)lic * n_in; // xs index 0 at f_split
                            axpy_f32(yrow + f_split, src, wv, nf - f_split);
                        }
                    } else {
                        for (int f = 0; f < nf; ++f)
                            yrow[f] += wv * conv_read(s, xs, n_in, lic, f * stride + base);
                    }
                }
            }
            // bias broadcast-added after the conv (candle_nn Conv1d::forward).
            if (s->bias) {
                const float bv = s->bias[oc];
                for (int f = 0; f < nf; ++f) yrow[f] += bv;
            }
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
//   raw = conv_transpose1d(xs) + bias         // len ot=(n_in-1)*stride+k
//   overlap-add carry: combined[0..pt] = raw[0..pt] + (prev_ys - bias)
//   emit = combined[0 .. ot-invalid]  (= n_in*stride samples)
//   new carry prev_ys = combined[ot-invalid .. ot]  (invalid = k - stride)
//
// We never build the full `raw` buffer: the scatter writes emitted positions
// straight to y and the tail (>= emit_len) to a small carry_scratch, then the
// overlap-add folds the previous carry (bias removed) into positions [0,pt).
//
// INVARIANT: prev holds exactly `invalid = k - stride` output steps once
// primed (prev_len is 0 before the first step, `invalid` after). causal is
// stored but unused in step (the Rust step never reads self.causal; the causal
// trim is implicit in the invalid-steps split, trim_right_ratio == 1).
struct MimiConvTrState {
    int in_c, out_c, ksize, stride, causal;
    int invalid;         // ksize - stride  (carry length once primed)
    const float *w;      // [in_c, out_c, ksize] folded, checkpoint layout
    const float *bias;   // [out_c] or NULL
    float *prev;         // [out_c, invalid] output overlap carry (bias INCLUDED)
    float *carry_scratch;// [out_c, invalid] next-carry accumulator
    int prev_valid;      // 0 until the first step has run
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
    // convtr fold: dim0 = in_c, rest = out_c * k.
    s->w = resolve_weight(w, base, in_c, out_c * ksize, a, err, errlen);
    if (!s->w) return 1;
    const MimiWeight *b = wfind(w, base, ".bias");
    if (b) {
        if (b->len != (uint64_t)out_c)
            return fail(err, errlen, "convtr bias length mismatch");
        s->bias = b->data;
    }
    int inv = s->invalid > 0 ? s->invalid : 1;   // avoid 0-byte alloc
    s->prev = (float *)mimi_arena_alloc(a, (size_t)out_c * inv * sizeof(float));
    s->carry_scratch = (float *)mimi_arena_alloc(a, (size_t)out_c * inv * sizeof(float));
    s->prev_valid = 0;
    *st = s;
    return 0;
}

void mimi_convtr_reset(MimiConvTrState *s) { s->prev_valid = 0; }

int mimi_convtr_step(MimiConvTrState *s, const float *xs, int n_in, float *y) {
    const int stride = s->stride, k = s->ksize, oc_n = s->out_c;
    const int emit_len = n_in * stride;
    const int invalid = s->invalid;               // ot = emit_len + invalid
    const int in_c = s->in_c;

    memset(y, 0, (size_t)oc_n * emit_len * sizeof(float));
    if (invalid > 0)
        memset(s->carry_scratch, 0, (size_t)oc_n * invalid * sizeof(float));

    // scatter: candle order  for kk: for oc: for l:  d = dot_ic; dst[l*stride+kk]+=d
    for (int kk = 0; kk < k; ++kk) {
        for (int oc = 0; oc < oc_n; ++oc) {
            const float *wcol = s->w + (size_t)oc * k + kk; // w[ic,oc,kk], ic stride out_c*k
            float *yrow = y + (size_t)oc * emit_len;
            float *crow = s->carry_scratch + (size_t)oc * (invalid > 0 ? invalid : 1);
            for (int l = 0; l < n_in; ++l) {
                const float d = dot_strided(xs + l, n_in, wcol, oc_n * k, in_c);
                const int out_t = l * stride + kk;
                if (out_t < emit_len) yrow[out_t] += d;
                else                  crow[out_t - emit_len] += d;
            }
        }
    }
    // bias broadcast-added to every raw position (emit region + tail carry).
    if (s->bias) {
        for (int oc = 0; oc < oc_n; ++oc) {
            const float bv = s->bias[oc];
            float *yrow = y + (size_t)oc * emit_len;
            for (int t = 0; t < emit_len; ++t) yrow[t] += bv;
            if (invalid > 0) {
                float *crow = s->carry_scratch + (size_t)oc * invalid;
                for (int t = 0; t < invalid; ++t) crow[t] += bv;
            }
        }
    }
    // overlap-add previous carry (bias removed) into positions [0, invalid).
    // General placement: index t < emit_len -> y, else -> carry tail.
    if (s->prev_valid && invalid > 0) {
        for (int oc = 0; oc < oc_n; ++oc) {
            const float bv = s->bias ? s->bias[oc] : 0.f;
            const float *prow = s->prev + (size_t)oc * invalid;
            float *yrow = y + (size_t)oc * emit_len;
            float *crow = s->carry_scratch + (size_t)oc * invalid;
            for (int t = 0; t < invalid; ++t) {
                const float add = prow[t] - bv;
                if (t < emit_len) yrow[t] += add;
                else              crow[t - emit_len] += add;
            }
        }
    }
    // commit new carry.
    if (invalid > 0)
        memcpy(s->prev, s->carry_scratch, (size_t)oc_n * invalid * sizeof(float));
    s->prev_valid = 1;
    return emit_len;
}

/* ======================================================================== *
 *  3. ConvTrUpsample1d  -> MimiUpsampleState   (depthwise, groups == dim)
 * ======================================================================== */
// ConvTrUpsample1d: stride 2, dim 512, k=4, causal, learnt, NO bias, norm None.
// Because groups == out_c == in_c, NormConvTranspose1d expands the stored
// [dim,1,k] weight to a block-diagonal [dim,dim,k] via an identity multiply and
// runs groups=1. That is exactly a DEPTHWISE transposed conv: output channel c
// depends only on input channel c with kernel w[c,0,:]. We keep the weight as
// [dim,1,k] and compute per-channel (no expansion, identical math).
// Weight name (hardcoded, single instance): "upsample.convtr.convtr.convtr.weight".
struct MimiUpsampleState {
    int dim, ksize, stride, invalid;
    const float *w;      // [dim, 1, ksize] checkpoint layout (== [dim, ksize])
    float *prev;         // [dim, invalid] overlap carry (no bias)
    float *carry_scratch;// [dim, invalid]
    int prev_valid;
};

int mimi_upsample_init(MimiUpsampleState **st, const MimiWeightTable *w,
                       MimiArena *a, char *err, size_t errlen) {
    const int dim = MIMI_DIM, stride = MIMI_UPSAMPLE_STRIDE;
    const int ksize = 2 * stride;                 // ConvTrUpsample1d: k = 2*stride
    MimiUpsampleState *s = (MimiUpsampleState *)mimi_arena_alloc(a, sizeof(MimiUpsampleState));
    memset(s, 0, sizeof(*s));
    s->dim = dim; s->ksize = ksize; s->stride = stride;
    s->invalid = ksize - stride;
    const MimiWeight *ww = mimi_weight_find(w, "upsample.convtr.convtr.convtr.weight");
    if (!ww) return fail(err, errlen, "upsample weight not found");
    if (ww->len != (uint64_t)dim * ksize)
        return fail(err, errlen, "upsample weight shape mismatch (expect [dim,1,2*stride])");
    s->w = ww->data;
    s->prev = (float *)mimi_arena_alloc(a, (size_t)dim * s->invalid * sizeof(float));
    s->carry_scratch = (float *)mimi_arena_alloc(a, (size_t)dim * s->invalid * sizeof(float));
    s->prev_valid = 0;
    *st = s;
    return 0;
}

void mimi_upsample_reset(MimiUpsampleState *s) { s->prev_valid = 0; }

int mimi_upsample_step(MimiUpsampleState *s, const float *xs, int n_in, float *y) {
    const int stride = s->stride, k = s->ksize, dim = s->dim, invalid = s->invalid;
    const int emit_len = n_in * stride;           // ot = emit_len + invalid

    memset(y, 0, (size_t)dim * emit_len * sizeof(float));
    memset(s->carry_scratch, 0, (size_t)dim * invalid * sizeof(float));

    // depthwise scatter: order for kk: for c: for l  (matches candle groups=1
    // scatter on the identity-expanded weight, where only c_in==c_out survives).
    for (int kk = 0; kk < k; ++kk) {
        for (int c = 0; c < dim; ++c) {
            const float wv = s->w[(size_t)c * k + kk];
            float *yrow = y + (size_t)c * emit_len;
            float *crow = s->carry_scratch + (size_t)c * invalid;
            for (int l = 0; l < n_in; ++l) {
                const float d = xs[(size_t)c * n_in + l] * wv;
                const int out_t = l * stride + kk;
                if (out_t < emit_len) yrow[out_t] += d;
                else                  crow[out_t - emit_len] += d;
            }
        }
    }
    // no bias. overlap-add previous carry into [0, invalid).
    if (s->prev_valid) {
        for (int c = 0; c < dim; ++c) {
            const float *prow = s->prev + (size_t)c * invalid;
            float *yrow = y + (size_t)c * emit_len;
            float *crow = s->carry_scratch + (size_t)c * invalid;
            for (int t = 0; t < invalid; ++t) {
                const float add = prow[t];
                if (t < emit_len) yrow[t] += add;
                else              crow[t - emit_len] += add;
            }
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
 *        -> inline in mimi_conv_step (accumulate from 0, add bias last)
 *   candle conv_transpose1d + bias
 *        -> the scatter in mimi_convtr_step / mimi_upsample_step
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
 *     * prev = [in_c, carry_cap] holds the left-context carry; prev_len is the
 *       number of valid time steps. INVARIANT: prev_len < kernel_eff at all
 *       times, so carry_cap = kernel_eff = (ksize-1)*dilation+1 never overflows.
 *       Proof: num_frames = (seq_len+stride-kernel_eff)/stride (floor, 0 when
 *       seq_len+stride<kernel_eff). Retained carry = seq_len - num_frames*stride
 *       is in [kernel_eff-stride, kernel_eff) if num_frames>0, else = seq_len
 *       (< kernel_eff). First-step left pad = padding_total = kernel_eff-stride.
 *     * left_pad_applied: false only before the first step. First step pre-loads
 *       prev with padding_total zeros (PadMode::Constant) then proceeds; this
 *       reproduces cat2(empty, pad1d(xs, padding_total, 0)).
 *     * logical sequence for a step = [prev(prev_len) ++ xs(n_in)]; output frame
 *       f, channel oc reads taps at logical pos f*stride + kk*dilation. num_frames
 *       = # producible frames; emits [out_c, num_frames] (0 while priming). New
 *       carry = logical[num_frames*stride .. seq_len]; when num_frames==0 the
 *       whole logical sequence is retained (Rust: state_prev_xs = cat2(prev,xs)).
 *     * DECODE FACT: every decoder StreamableConv1d has stride==1 (upsampling is
 *       done only by transposed convs), so num_frames = seq_len-kernel_eff+1 = n_in
 *       on the first step and thereafter — priming (0 out) never actually fires in
 *       the decode graph, but is implemented faithfully for stride>1.
 *   MimiConvTrState (StreamableConvTranspose1d, groups==1):
 *     * prev = [out_c, invalid] output-overlap carry WITH bias included; invalid
 *       = ksize - stride. prev_valid false only before the first step. Emits
 *       emit_len = n_in*stride samples/step; ot = emit_len + invalid raw samples
 *       are produced, of which the last `invalid` become the next carry.
 *     * overlap-add: emitted[0..invalid] = raw[0..invalid] + (prev - bias); the
 *       bias is subtracted from prev because it is re-added by the current raw
 *       (Rust "Remove the bias as it will be applied multiple times"). The tail
 *       carry keeps bias (re-subtracted next step). causal is unused in step.
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
 *     The checkpoint stores pre-folded "weight" (0 weight_g/weight_v tensors);
 *     the weight_g/weight_v fold path is implemented but not exercised here.
 *     Prefix contract: caller passes the *streamable module* node (e.g.
 *     "decoder.model.0"); this unit appends the ".conv.conv"/".convtr.convtr"
 *     inner nesting, matching the moshi VarBuilder pp() chain exactly.
 *
 * (d) Weight-norm fold formula and dims
 *     weight = weight_v * weight_g / ||weight_v||_2, norm over dims (1,2)
 *     (keepdim) with weight_g / norm broadcast over dim0:
 *       conv1d : dim0 = out_c, norm over (in_c/groups, ksize) per output channel.
 *       convtr : dim0 = in_c,  norm over (out_c, ksize) per INPUT channel.
 *     Folded once into arena buffers (weight_norm_fold, n0=dim0, rest=product of
 *     the remaining dims). ||.||_2 = sqrtf(sum of squares), f32.
 *
 * (e) Accumulation order (f32 throughout, matches candle CPU kernels)
 *     conv1d  : per output (oc,f), sum over taps kk (outer) then in-channels ic
 *               (inner): y = sum_kk( sum_ic in*w ); bias broadcast-added last.
 *               The stride-1 fast path vectorises the frame axis f (contiguous
 *               NEON AXPY, split at the prev/xs boundary) WITHOUT changing the
 *               per-element (kk,ic) summation order.
 *     convtr / upsample : scatter in candle order  for kk: for oc(/c): for l,
 *               each contribution a dot over in-channels (ascending ic); bias
 *               broadcast-added to raw; then overlap-add of the prior carry.
 *
 * (f) Uncertainties / friction
 *   1. ABI friction (documented, not forked): mimi_conv_init/mimi_convtr_init
 *      take a *streamable node* prefix and this unit appends ".conv.conv" /
 *      ".convtr.convtr". If the arbiter instead intends the prefix to be the
 *      direct weight parent, drop the snprintf(base,"%s.conv.conv") lines. The
 *      upsample name is hardcoded (single instance, no prefix in the ABI).
 *   2. NEON is applied only to the conv1d stride-1 AXPY (the decode hot path).
 *      convtr/upsample scatter and the fold stay scalar (strided access);
 *      MIMI_SCALAR_REF forces the scalar AXPY too, giving a pure-scalar sibling
 *      for parity bisecting. Numerics are the *faithful* tier (ulp band), not
 *      bit-exact: candle's f32 vec_dot grouping and NEON's differ; the harness
 *      measures the band.
 *   3. PadMode: only Constant (zeros) is implemented — the sole mode on the
 *      decode path. Replicate (ConvDownsample1d, encode-only) is not needed here.
 *   4. `causal` is stored but never read in either step, matching the Rust step
 *      functions (both are inherently causal: conv1d left-pads only, convtr's
 *      trim is the invalid-steps split with trim_right_ratio == 1).
 *   5. Buffers are tiny (prev/scratch are O(channels * kernel) — no
 *      frame-sized scratch): outputs are written into the caller's y, so no
 *      worst-case max-frames arena sizing is required.
 */
