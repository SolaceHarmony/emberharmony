// Native Conformer encoder + audio adapter. Contract: lfm_conformer.h.
// Parity oracle: native/tests/fixtures/conformer/ (real checkpoint, BF16
// production ladder, captured from the deleted Rust).
//
// Law of this TU: C++ binds views, moves bytes (transposes, padding, im2col,
// rel_shift, head packing), and sequences stages. Every produced value comes
// from an assembly leaf (flashkern_conformer.S / flashkern_math.S), the
// engine's bf16 GEMM pass (the identical kernel+dispatch the deleted
// linear_forward ticket used), or the approved f32 matmul dispatch
// (Accelerate on Apple, lfm_sgemm_f32 leaf elsewhere) for stages the
// reference ran in f32 (convolutions, attention).
//
// Ladder (fixtures arbitrate): bf16 linears = engine GEMM -> f32 bias -> bf16
// round. Convs = widen -> f32 conv -> f32 bias -> bf16 round -> activation in
// bf16. LayerNorm = f32 stats, bf16 weight/bias tail (layer_norm_slow).
// BatchNorm eval = all-bf16 broadcast chain (denominators prefolded at create
// with explicit bf16 rounding). Attention = f32 scores/softmax/aggregation;
// probabilities never round to bf16. SiLU/gelu_erf round once; GLU rounds the
// sigmoid, then the product.

#include "lfm_conformer.h"

#include "flashkern_gemm.h"
#include "lfm_safetensors.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

extern "C" {
// flashkern_conformer.S (both architectures)
void lfm_bf16_widen_f32(const uint16_t *x, float *y, uint64_t n);
int lfm_ln_bf16(const uint16_t *x, const uint16_t *w, const uint16_t *b,
                uint16_t *y, uint64_t rows, uint64_t cols, float eps);
void lfm_bn_bf16(const uint16_t *x, const uint16_t *mean, const uint16_t *denom,
                 const uint16_t *w, const uint16_t *b, uint16_t *y,
                 uint64_t channels, uint64_t t);
void lfm_relu_bf16(uint16_t *x, uint64_t n);
int lfm_silu_bf16(uint16_t *x, uint64_t n);
int lfm_glu_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, uint64_t n);
int lfm_gelu_erf_bf16(uint16_t *x, uint64_t n);
void lfm_residual_half_bf16(uint16_t *residual, const uint16_t *h, uint64_t n);
void lfm_add_bf16(uint16_t *residual, const uint16_t *h, uint64_t n);
int lfm_softmax_rows_f32(float *x, uint64_t rows, uint64_t cols);
void lfm_sgemm_f32(const float *a, const float *b, float *c, uint64_t m,
                   uint64_t n, uint64_t k);
void lfm_sgemm_nt_f32(const float *a, const float *bt, float *c, uint64_t m,
                      uint64_t n, uint64_t k);
void lfm_dwconv_tap_f32(const float *xpad, const float *w, float *y,
                        uint64_t channels, uint64_t t_out, uint64_t t_pad,
                        uint64_t k, uint64_t stride);
void lfm_dwconv2d_k3s2_f32(const float *xpad, const float *w, float *y,
                           uint64_t channels, uint64_t h_pad, uint64_t w_pad,
                           uint64_t h_out, uint64_t w_out);
void lfm_bias_rows_f32(float *y, const uint16_t *bias, uint64_t rows, uint64_t n);
void lfm_bias_bcast_f32(float *y, const uint16_t *bias, uint64_t channels,
                        uint64_t t);
void lfm_add_scale_f32(float *acc, const float *addend, uint64_t rows,
                       uint64_t cols, uint64_t addend_stride, float scale);
void lfm_add_bias_hd_bf16(const uint16_t *x, const uint16_t *bias, uint16_t *y,
                          uint64_t t, uint64_t heads, uint64_t dk);
int lfm_pe_build_bf16(const float *div, uint64_t half, uint64_t t,
                      uint16_t *out);
// flashkern kernels (existing)
void lfm_f32_to_bf16(const float *input, uint16_t *output, int count);
}

namespace {

void sgemm_rm(size_t m, size_t n, size_t k, const float *a, const float *b,
              float *c) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n,
                (int)k, 1.0f, a, (int)k, b, (int)n, 0.0f, c, (int)n);
#else
    lfm_sgemm_f32(a, b, c, m, n, k);
#endif
}

void sgemm_ntx(size_t m, size_t n, size_t k, const float *a, const float *bt,
               float *c) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)m, (int)n,
                (int)k, 1.0f, a, (int)k, bt, (int)k, 0.0f, c, (int)n);
#else
    lfm_sgemm_nt_f32(a, bt, c, m, n, k);
#endif
}

inline float bf16_widen_one(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline uint16_t bf16_round_one(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t tie = (bits >> 16) & 1;
    bits += 0x7fffu + tie;
    return (uint16_t)(bits >> 16);
}

struct View {
    const uint16_t *bf16 = nullptr;
    uint64_t rows = 0, cols = 0, elements = 0;
};

struct LayerWeights {
    View norm_ff1_w, norm_ff1_b, ff1_l1_w, ff1_l1_b, ff1_l2_w, ff1_l2_b;
    View norm_att_w, norm_att_b;
    View q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, pos_w;
    View bias_u, bias_v;
    View norm_conv_w, norm_conv_b;
    View pw1_w, pw1_b, dw_w, dw_b, pw2_w, pw2_b;
    std::vector<uint16_t> bn_mean, bn_denom, bn_w, bn_b;
    View norm_ff2_w, norm_ff2_b, ff2_l1_w, ff2_l1_b, ff2_l2_w, ff2_l2_b;
    View norm_out_w, norm_out_b;
};

inline uint64_t conv_len(uint64_t l) { return l >= 1 ? (l + 2 - 3) / 2 + 1 : 0; }

} // namespace

struct LfmConformer {
    LfmConformerGeometry g;
    void *engine = nullptr;
    View stem_w, stem_b, dw1_w, dw1_b, pwa_w, pwa_b, dw2_w, dw2_b, pwb_w, pwb_b;
    View sub_out_w, sub_out_b;
    std::vector<LayerWeights> layers;
    View ad_ln_w, ad_ln_b, ad_l1_w, ad_l1_b, ad_l2_w, ad_l2_b;
    std::vector<float> pe_div;
};

// Bump arenas. Sized ONCE per forward to a safe high-water bound (reserve),
// then f()/b() only advance the offset — never resize mid-pass, because a
// resize would reallocate and dangle every pointer already handed out.
struct LfmConformerWorkspace {
    std::vector<float> f32;
    std::vector<uint16_t> u16;
    size_t fo = 0, bo = 0;
    bool overflow = false;
    void ensure(size_t need_f, size_t need_b) {
        if (f32.size() < need_f) f32.assign(need_f, 0.0f);
        if (u16.size() < need_b) u16.assign(need_b, 0);
        fo = bo = 0;
        overflow = false;
    }
    float *f(size_t n) {
        if (fo + n > f32.size()) {
            overflow = true;
            return f32.data(); // safe: caller checks overflow, forward bails
        }
        float *p = f32.data() + fo;
        fo += n;
        return p;
    }
    uint16_t *b(size_t n) {
        if (bo + n > u16.size()) {
            overflow = true;
            return u16.data();
        }
        uint16_t *p = u16.data() + bo;
        bo += n;
        return p;
    }
};

namespace {

int bind(const LfmWeightImage *img, const std::string &name, View &v,
         std::vector<uint64_t> expect, char *err, size_t errlen) {
    LfmTensorView tv{};
    tv.size = sizeof(tv);
    if (lfm_weights_find(img, name.c_str(), &tv) != 0) {
        std::snprintf(err, errlen, "conformer bind: missing '%s'", name.c_str());
        return -ENOENT;
    }
    if (tv.dtype != LFM_DTYPE_BF16) {
        std::snprintf(err, errlen, "conformer bind: '%s' not BF16", name.c_str());
        return -ENOENT;
    }
    if (!expect.empty()) {
        if (tv.rank != expect.size()) {
            std::snprintf(err, errlen, "conformer bind: '%s' rank %u != %zu",
                          name.c_str(), tv.rank, expect.size());
            return -ENOENT;
        }
        for (size_t i = 0; i < expect.size(); ++i)
            if (expect[i] != 0 && tv.shape[i] != expect[i]) {
                std::snprintf(err, errlen, "conformer bind: '%s' dim %zu mismatch",
                              name.c_str(), i);
                return -ENOENT;
            }
    }
    v.bf16 = (const uint16_t *)tv.data;
    v.elements = tv.elements;
    v.rows = tv.rank >= 1 ? tv.shape[0] : 1;
    v.cols = tv.rank >= 2 ? tv.elements / tv.shape[0] : 1;
    return 0;
}

// bf16 linear: X(rows x K) bf16 -> engine GEMM -> f32 -> +bias(f32) -> bf16.
int bf16_linear(void *engine, LfmConformerWorkspace *ws, const uint16_t *x,
                uint64_t rows, uint64_t k, const View &w, const View *bias,
                uint16_t *out) {
    const uint64_t n = w.rows;
    float *scratch = ws->f(rows * n);
    // Weight is checkpoint-native (N,K) row-major -> LFM_GEMM_RHS_NK (1).
    if (lfm_engine_bf16_gemm_f32(engine, x, rows * k, w.bf16, w.elements,
                                 scratch, rows * n, rows, n, k, 1) != 0)
        return -EIO;
    if (bias && bias->bf16) lfm_bias_rows_f32(scratch, bias->bf16, rows, n);
    lfm_f32_to_bf16(scratch, out, (int)(rows * n));
    ws->fo -= rows * n; // scratch is dead once rounded
    return 0;
}

// Widen a bf16 (C x T) plane into an f32 plane with a symmetric zero border of
// `pad` on the T axis only (1-D conv padding). Pure movement + leaf widen.
void widen_pad_1d(const uint16_t *x, uint64_t c, uint64_t t, uint64_t pad,
                  float *dst) {
    std::memset(dst, 0, (size_t)c * (t + 2 * pad) * sizeof(float));
    for (uint64_t ch = 0; ch < c; ++ch)
        lfm_bf16_widen_f32(x + ch * t, dst + ch * (t + 2 * pad) + pad, t);
}

// Widen a bf16 image (C x H x W) into f32 with a 1-px zero border per plane.
void widen_pad_2d(const uint16_t *x, uint64_t c, uint64_t h, uint64_t w,
                  float *dst) {
    const uint64_t hp = h + 2, wp = w + 2;
    std::memset(dst, 0, (size_t)c * hp * wp * sizeof(float));
    for (uint64_t ch = 0; ch < c; ++ch)
        for (uint64_t y = 0; y < h; ++y)
            lfm_bf16_widen_f32(x + (ch * h + y) * w,
                               dst + ch * hp * wp + (y + 1) * wp + 1, w);
}

} // namespace

extern "C" int lfm_conformer_create(void *engine, const void *weights,
                                    const LfmConformerGeometry *geometry,
                                    LfmConformer **out, char *error,
                                    size_t error_length) {
    if (!engine || !weights || !geometry || !out) return -EINVAL;
    if (geometry->size < sizeof(LfmConformerGeometry) ||
        geometry->abi_version != LFM_CONFORMER_ABI)
        return -EINVAL;
    const LfmWeightImage *img = (const LfmWeightImage *)weights;
    const LfmConformerGeometry &g = *geometry;
    if (g.d_model == 0 || g.n_layers == 0 || g.n_heads == 0 ||
        g.d_model % g.n_heads != 0 || g.subsampling != 8 || g.feat_in % 8 != 0)
        return -EINVAL;

    LfmConformer *c = new (std::nothrow) LfmConformer();
    if (!c) return -ENOMEM;
    c->g = g;
    c->engine = engine;

    char localerr[256] = {0};
    char *err = (error && error_length) ? error : localerr;
    size_t errlen = (error && error_length) ? error_length : sizeof(localerr);
    const uint64_t D = g.d_model, FF = g.d_ff, CC = g.conv_channels;

    int rc = 0;
    auto B = [&](const std::string &n, View &v, std::vector<uint64_t> e) {
        if (rc == 0) rc = bind(img, n, v, std::move(e), err, errlen);
    };

    B("conformer.pre_encode.conv.0.weight", c->stem_w, {CC, 1, 3, 3});
    B("conformer.pre_encode.conv.0.bias", c->stem_b, {CC});
    B("conformer.pre_encode.conv.2.weight", c->dw1_w, {CC, 1, 3, 3});
    B("conformer.pre_encode.conv.2.bias", c->dw1_b, {CC});
    B("conformer.pre_encode.conv.3.weight", c->pwa_w, {CC, CC, 1, 1});
    B("conformer.pre_encode.conv.3.bias", c->pwa_b, {CC});
    B("conformer.pre_encode.conv.5.weight", c->dw2_w, {CC, 1, 3, 3});
    B("conformer.pre_encode.conv.5.bias", c->dw2_b, {CC});
    B("conformer.pre_encode.conv.6.weight", c->pwb_w, {CC, CC, 1, 1});
    B("conformer.pre_encode.conv.6.bias", c->pwb_b, {CC});
    B("conformer.pre_encode.out.weight", c->sub_out_w, {D, CC * (g.feat_in / 8)});
    B("conformer.pre_encode.out.bias", c->sub_out_b, {D});

    c->layers.resize(g.n_layers);
    for (uint32_t i = 0; rc == 0 && i < g.n_layers; ++i) {
        const std::string p = "conformer.layers." + std::to_string(i) + ".";
        LayerWeights &L = c->layers[i];
        B(p + "norm_feed_forward1.weight", L.norm_ff1_w, {D});
        B(p + "norm_feed_forward1.bias", L.norm_ff1_b, {D});
        B(p + "feed_forward1.linear1.weight", L.ff1_l1_w, {FF, D});
        B(p + "feed_forward1.linear1.bias", L.ff1_l1_b, {FF});
        B(p + "feed_forward1.linear2.weight", L.ff1_l2_w, {D, FF});
        B(p + "feed_forward1.linear2.bias", L.ff1_l2_b, {D});
        B(p + "norm_self_att.weight", L.norm_att_w, {D});
        B(p + "norm_self_att.bias", L.norm_att_b, {D});
        B(p + "self_attn.linear_q.weight", L.q_w, {D, D});
        B(p + "self_attn.linear_q.bias", L.q_b, {D});
        B(p + "self_attn.linear_k.weight", L.k_w, {D, D});
        B(p + "self_attn.linear_k.bias", L.k_b, {D});
        B(p + "self_attn.linear_v.weight", L.v_w, {D, D});
        B(p + "self_attn.linear_v.bias", L.v_b, {D});
        B(p + "self_attn.linear_out.weight", L.out_w, {D, D});
        B(p + "self_attn.linear_out.bias", L.out_b, {D});
        B(p + "self_attn.linear_pos.weight", L.pos_w, {D, D});
        B(p + "self_attn.pos_bias_u", L.bias_u, {g.n_heads, D / g.n_heads});
        B(p + "self_attn.pos_bias_v", L.bias_v, {g.n_heads, D / g.n_heads});
        B(p + "norm_conv.weight", L.norm_conv_w, {D});
        B(p + "norm_conv.bias", L.norm_conv_b, {D});
        B(p + "conv.pointwise_conv1.weight", L.pw1_w, {2 * D, D, 1});
        B(p + "conv.pointwise_conv1.bias", L.pw1_b, {2 * D});
        B(p + "conv.depthwise_conv.weight", L.dw_w, {D, 1, g.conv_kernel});
        B(p + "conv.depthwise_conv.bias", L.dw_b, {D});
        B(p + "conv.pointwise_conv2.weight", L.pw2_w, {D, D, 1});
        B(p + "conv.pointwise_conv2.bias", L.pw2_b, {D});
        B(p + "norm_feed_forward2.weight", L.norm_ff2_w, {D});
        B(p + "norm_feed_forward2.bias", L.norm_ff2_b, {D});
        B(p + "feed_forward2.linear1.weight", L.ff2_l1_w, {FF, D});
        B(p + "feed_forward2.linear1.bias", L.ff2_l1_b, {FF});
        B(p + "feed_forward2.linear2.weight", L.ff2_l2_w, {D, FF});
        B(p + "feed_forward2.linear2.bias", L.ff2_l2_b, {D});
        B(p + "norm_out.weight", L.norm_out_w, {D});
        B(p + "norm_out.bias", L.norm_out_b, {D});

        View bn_w, bn_b, bn_m, bn_v;
        B(p + "conv.batch_norm.weight", bn_w, {D});
        B(p + "conv.batch_norm.bias", bn_b, {D});
        B(p + "conv.batch_norm.running_mean", bn_m, {D});
        B(p + "conv.batch_norm.running_var", bn_v, {D});
        if (rc == 0) {
            // BatchNorm eval prefold, candle op order with explicit bf16
            // rounding at each step: denom = bf16(sqrt(bf16(var + eps))).
            L.bn_mean.assign(bn_m.bf16, bn_m.bf16 + D);
            L.bn_w.assign(bn_w.bf16, bn_w.bf16 + D);
            L.bn_b.assign(bn_b.bf16, bn_b.bf16 + D);
            L.bn_denom.resize(D);
            for (uint64_t j = 0; j < D; ++j) {
                const float ve =
                    bf16_widen_one(bf16_round_one(bf16_widen_one(bn_v.bf16[j]) + 1e-5f));
                L.bn_denom[j] = bf16_round_one(std::sqrt(ve));
            }
        }
    }

    B("audio_adapter.model.0.weight", c->ad_ln_w, {D});
    B("audio_adapter.model.0.bias", c->ad_ln_b, {D});
    B("audio_adapter.model.1.weight", c->ad_l1_w, {g.adapter_hidden, D});
    B("audio_adapter.model.1.bias", c->ad_l1_b, {g.adapter_hidden});
    B("audio_adapter.model.3.weight", c->ad_l2_w, {g.adapter_out, g.adapter_hidden});
    B("audio_adapter.model.3.bias", c->ad_l2_b, {g.adapter_out});

    if (rc != 0) {
        delete c;
        return rc;
    }
    // Rel-pos inverse frequencies (create_pe): exp(2i * -(ln(10000)/d)), f64
    // math stored f32 — init-time table (mel-pass precedent).
    c->pe_div.resize(D / 2);
    for (uint64_t i = 0; i < D / 2; ++i)
        c->pe_div[i] =
            (float)std::exp(-(std::log(10000.0) / (double)D) * (double)(2 * i));
    *out = c;
    return 0;
}

extern "C" int lfm_conformer_destroy(LfmConformer *c) {
    if (!c) return -EINVAL;
    delete c;
    return 0;
}

extern "C" int lfm_conformer_workspace_create(LfmConformerWorkspace **out) {
    if (!out) return -EINVAL;
    auto *w = new (std::nothrow) LfmConformerWorkspace();
    if (!w) return -ENOMEM;
    *out = w;
    return 0;
}

extern "C" int lfm_conformer_workspace_destroy(LfmConformerWorkspace *w) {
    if (!w) return -EINVAL;
    delete w;
    return 0;
}

extern "C" uint64_t lfm_conformer_out_rows(const LfmConformer *c, uint64_t t) {
    return (c && t) ? conv_len(conv_len(conv_len(t))) : 0;
}

extern "C" int lfm_conformer_forward(const LfmConformer *c,
                                     LfmConformerWorkspace *ws,
                                     const uint16_t *mel, uint64_t mel_frames,
                                     uint16_t *out_rows_dst,
                                     uint64_t out_capacity_values) {
    if (!c || !ws || !mel || !out_rows_dst || mel_frames == 0) return -EINVAL;
    const LfmConformerGeometry &g = c->g;
    const uint64_t D = g.d_model, FF = g.d_ff, CC = g.conv_channels;
    const uint64_t H = g.n_heads, DK = D / H, K = g.conv_kernel;
    const uint64_t F0 = g.feat_in, T0 = mel_frames;
    const uint64_t T1 = conv_len(T0), F1 = conv_len(F0);
    const uint64_t T2 = conv_len(T1), F2 = conv_len(F1);
    const uint64_t T3 = conv_len(T2), F3 = conv_len(F2);
    const uint64_t Tp = T3, P = 2 * Tp - 1;
    if (Tp == 0 || T1 == 0 || T2 == 0) return -EINVAL;
    if (out_capacity_values < Tp * g.adapter_out) return -EINVAL;
    void *eng = c->engine;

    // High-water bound for both arenas, reserved once. Deliberately generous:
    // an underestimate would re-dangle pointers, so we bound every arena by a
    // comfortable multiple of the largest single plane class and sum them.
    const uint64_t maxTF = T1 * F1;                         // largest conv plane
    const uint64_t wideW = 2 * D * D;                       // largest weight widen
    const uint64_t bigrow = (Tp > P ? Tp : P) * (FF > 2 * D ? FF : 2 * D);
    const uint64_t attn = H * (Tp * Tp + 2 * Tp * P + 5 * Tp * DK + P * DK);
    const uint64_t FLAT = CC * F3;
    const uint64_t need_f =
        16 * CC * maxTF + 4 * wideW + 4 * bigrow + 2 * attn + 8 * D * Tp + 4096;
    const uint64_t need_b =
        6 * CC * maxTF + Tp * FLAT + 24 * Tp * (FF > 2 * D ? FF : 2 * D) +
        8 * P * D + 4 * Tp * g.adapter_hidden + 4096;
    ws->ensure(need_f, need_b);

    // ---- subsampling --------------------------------------------------------
    // Input arrives (F0, T0); the reference views it as an image (T0, F0).
    uint16_t *img0 = ws->b(T0 * F0);
    for (uint64_t f = 0; f < F0; ++f)
        for (uint64_t t = 0; t < T0; ++t) img0[t * F0 + f] = mel[f * T0 + t];

    // stem: full conv (1 -> CC), k3 s2 p1: im2col + f32 GEMM + bias -> bf16 -> ReLU.
    uint16_t *plane_a = ws->b(CC * T1 * F1);
    {
        float *xpad = ws->f((T0 + 2) * (F0 + 2));
        widen_pad_2d(img0, 1, T0, F0, xpad);
        float *cols = ws->f(9 * T1 * F1);
        // gather: cols[(dy*3+dx)][t*F1+f] = xpad[(2t+dy)*(F0+2) + (2f+dx)]
        for (uint64_t dy = 0; dy < 3; ++dy)
            for (uint64_t dx = 0; dx < 3; ++dx) {
                float *row = cols + (dy * 3 + dx) * (T1 * F1);
                for (uint64_t t = 0; t < T1; ++t)
                    for (uint64_t f = 0; f < F1; ++f)
                        row[t * F1 + f] = xpad[(2 * t + dy) * (F0 + 2) + (2 * f + dx)];
            }
        float *wf = ws->f(CC * 9);
        lfm_bf16_widen_f32(c->stem_w.bf16, wf, CC * 9);
        float *y = ws->f(CC * T1 * F1);
        sgemm_rm(CC, T1 * F1, 9, wf, cols, y);
        lfm_bias_bcast_f32(y, c->stem_b.bf16, CC, T1 * F1);
        lfm_f32_to_bf16(y, plane_a, (int)(CC * T1 * F1));
        lfm_relu_bf16(plane_a, CC * T1 * F1);
        ws->fo = 0;
    }

    // two dw+pw stages: depthwise k3 s2 p1 (leaf) -> bf16; pointwise k1 (f32
    // GEMM over channels) -> bf16 -> ReLU.
    uint16_t *plane_b = ws->b(CC * T2 * F2);
    uint16_t *plane_c = ws->b(CC * T3 * F3);
    struct Stage {
        const View *dw_w, *dw_b, *pw_w, *pw_b;
        uint16_t *in, *out;
        uint64_t hi, wi, ho, wo;
    } stages[2] = {
        {&c->dw1_w, &c->dw1_b, &c->pwa_w, &c->pwa_b, plane_a, plane_b, T1, F1, T2, F2},
        {&c->dw2_w, &c->dw2_b, &c->pwb_w, &c->pwb_b, plane_b, plane_c, T2, F2, T3, F3},
    };
    for (const Stage &s : stages) {
        float *xpad = ws->f(CC * (s.hi + 2) * (s.wi + 2));
        widen_pad_2d(s.in, CC, s.hi, s.wi, xpad);
        float *wdw = ws->f(CC * 9);
        lfm_bf16_widen_f32(s.dw_w->bf16, wdw, CC * 9);
        float *ydw = ws->f(CC * s.ho * s.wo);
        lfm_dwconv2d_k3s2_f32(xpad, wdw, ydw, CC, s.hi + 2, s.wi + 2, s.ho, s.wo);
        lfm_bias_bcast_f32(ydw, s.dw_b->bf16, CC, s.ho * s.wo);
        uint16_t *dwb = ws->b(CC * s.ho * s.wo);
        lfm_f32_to_bf16(ydw, dwb, (int)(CC * s.ho * s.wo));
        // pointwise k1: widen -> W(CC x CC) f32 GEMM -> bias -> bf16 -> ReLU.
        float *xin = ws->f(CC * s.ho * s.wo);
        lfm_bf16_widen_f32(dwb, xin, CC * s.ho * s.wo);
        float *wpw = ws->f(CC * CC);
        lfm_bf16_widen_f32(s.pw_w->bf16, wpw, CC * CC);
        float *ypw = ws->f(CC * s.ho * s.wo);
        sgemm_rm(CC, s.ho * s.wo, CC, wpw, xin, ypw);
        lfm_bias_bcast_f32(ypw, s.pw_b->bf16, CC, s.ho * s.wo);
        lfm_f32_to_bf16(ypw, s.out, (int)(CC * s.ho * s.wo));
        lfm_relu_bf16(s.out, CC * s.ho * s.wo);
        ws->fo = 0;
        ws->bo -= CC * s.ho * s.wo; // dwb dead
    }

    // flatten (CC, T3, F3) -> rows (T3, CC*F3): row t = [c-major][f]. FLAT was
    // computed for the arena bound above.
    uint16_t *flat_rows = ws->b(Tp * FLAT);
    for (uint64_t t = 0; t < Tp; ++t)
        for (uint64_t ch = 0; ch < CC; ++ch)
            for (uint64_t f = 0; f < F3; ++f)
                flat_rows[t * FLAT + ch * F3 + f] = plane_c[(ch * Tp + t) * F3 + f];

    // pre_encode.out: bf16 linear (Tp x FLAT) x (D, FLAT) -> x (Tp x D).
    uint16_t *x = ws->b(Tp * D);
    int rc = bf16_linear(eng, ws, flat_rows, Tp, FLAT, c->sub_out_w,
                         &c->sub_out_b, x);
    if (rc != 0) return rc;

    // rel-pos table (P x D) bf16 — sin/cos leaf; xscaling=false leaves x as-is.
    uint16_t *pe = ws->b(P * D);
    if (lfm_pe_build_bf16(c->pe_div.data(), D / 2, Tp, pe) != 0) return -EIO;

    // ---- layers -------------------------------------------------------------
    uint16_t *h = ws->b(Tp * (FF > 2 * D ? FF : 2 * D)); // stage output plane
    uint16_t *tmp = ws->b(Tp * (FF > 2 * D ? FF : 2 * D));
    uint16_t *qkv = ws->b(3 * Tp * D);
    uint16_t *pproj = ws->b(P * D);
    const float inv_sdk = 1.0f / std::sqrt((float)DK); // 1/8 exact for dk=64

    for (uint32_t li = 0; li < g.n_layers; ++li) {
        const LayerWeights &L = c->layers[li];
        const size_t f_mark = ws->fo, b_mark = ws->bo;

        // ff1 half-step: LN -> lin1 -> SiLU -> lin2 -> residual + 0.5*h.
        if (lfm_ln_bf16(x, L.norm_ff1_w.bf16, L.norm_ff1_b.bf16, tmp, Tp, D,
                        1e-5f) != 0)
            return -EIO;
        rc = bf16_linear(eng, ws, tmp, Tp, D, L.ff1_l1_w, &L.ff1_l1_b, h);
        if (rc != 0) return rc;
        if (lfm_silu_bf16(h, Tp * FF) != 0) return -EIO;
        rc = bf16_linear(eng, ws, h, Tp, FF, L.ff1_l2_w, &L.ff1_l2_b, tmp);
        if (rc != 0) return rc;
        lfm_residual_half_bf16(x, tmp, Tp * D);

        // attention: LN -> q,k,v -> (q+u)kT + rel_shift((q+v)pT) -> softmax ->
        // probs·v -> out linear -> residual.
        if (lfm_ln_bf16(x, L.norm_att_w.bf16, L.norm_att_b.bf16, tmp, Tp, D,
                        1e-5f) != 0)
            return -EIO;
        uint16_t *q = qkv, *k = qkv + Tp * D, *v = qkv + 2 * Tp * D;
        rc = bf16_linear(eng, ws, tmp, Tp, D, L.q_w, &L.q_b, q);
        if (rc == 0) rc = bf16_linear(eng, ws, tmp, Tp, D, L.k_w, &L.k_b, k);
        if (rc == 0) rc = bf16_linear(eng, ws, tmp, Tp, D, L.v_w, &L.v_b, v);
        if (rc == 0) rc = bf16_linear(eng, ws, pe, P, D, L.pos_w, nullptr, pproj);
        if (rc != 0) return rc;
        // q+u / q+v in bf16 (broadcast add per (h, dk)).
        uint16_t *qu = ws->b(Tp * D), *qv = ws->b(Tp * D);
        lfm_add_bias_hd_bf16(q, L.bias_u.bf16, qu, Tp, H, DK);
        lfm_add_bias_hd_bf16(q, L.bias_v.bf16, qv, Tp, H, DK);
        // widen + pack per head: (Tp, H, DK) -> head-major (H, Tp, DK) f32.
        float *quf = ws->f(H * Tp * DK), *qvf = ws->f(H * Tp * DK);
        float *kf = ws->f(H * Tp * DK), *vf = ws->f(H * Tp * DK);
        float *pf = ws->f(H * P * DK);
        for (uint64_t hh = 0; hh < H; ++hh)
            for (uint64_t t = 0; t < Tp; ++t) {
                lfm_bf16_widen_f32(qu + t * D + hh * DK, quf + (hh * Tp + t) * DK, DK);
                lfm_bf16_widen_f32(qv + t * D + hh * DK, qvf + (hh * Tp + t) * DK, DK);
                lfm_bf16_widen_f32(k + t * D + hh * DK, kf + (hh * Tp + t) * DK, DK);
                lfm_bf16_widen_f32(v + t * D + hh * DK, vf + (hh * Tp + t) * DK, DK);
            }
        for (uint64_t hh = 0; hh < H; ++hh)
            for (uint64_t p = 0; p < P; ++p)
                lfm_bf16_widen_f32(pproj + p * D + hh * DK, pf + (hh * P + p) * DK, DK);
        // scores.
        float *ac = ws->f(H * Tp * Tp);
        float *bd = ws->f(H * Tp * P);
        float *shifted = ws->f(H * Tp * P);
        for (uint64_t hh = 0; hh < H; ++hh) {
            sgemm_ntx(Tp, Tp, DK, quf + hh * Tp * DK, kf + hh * Tp * DK,
                      ac + hh * Tp * Tp);
            sgemm_ntx(Tp, P, DK, qvf + hh * Tp * DK, pf + hh * P * DK,
                      bd + hh * Tp * P);
        }
        // rel_shift (movement): pad left 1 -> reshape (P+1, Tp) -> drop row 0
        // -> reshape (Tp, P). Direct index algebra on the flat padded layout:
        // padded flat index of out[t][j] is (t*(P+1) + j + 1) - Tp... realized
        // literally below via the two reinterpretations.
        for (uint64_t hh = 0; hh < H; ++hh) {
            const float *src = bd + hh * Tp * P;
            float *dst = shifted + hh * Tp * P;
            // padded row-major (Tp, P+1) with col0 = 0: flat[t][0]=0,
            // flat[t][j+1]=src[t][j]. Reshape to (P+1, Tp), drop first row,
            // reshape (Tp, P): out_flat[i] = padded_flat[i + Tp].
            for (uint64_t i = 0; i < Tp * P; ++i) {
                const uint64_t padded_index = i + Tp;
                const uint64_t r = padded_index / (P + 1);
                const uint64_t col = padded_index % (P + 1);
                dst[i] = (col == 0) ? 0.0f : src[r * P + col - 1];
            }
        }
        // scores = (ac + shifted[:, :Tp]) * inv_sdk, then row softmax (f32).
        for (uint64_t hh = 0; hh < H; ++hh)
            lfm_add_scale_f32(ac + hh * Tp * Tp, shifted + hh * Tp * P, Tp, Tp,
                              P, inv_sdk);
        if (lfm_softmax_rows_f32(ac, H * Tp, Tp) != 0) return -EIO;
        // aggregation: probs (Tp x Tp) x v (Tp x DK) per head -> (Tp, D) f32.
        float *att = ws->f(Tp * D);
        for (uint64_t hh = 0; hh < H; ++hh) {
            float *outh = ws->f(Tp * DK);
            sgemm_rm(Tp, DK, Tp, ac + hh * Tp * Tp, vf + hh * Tp * DK, outh);
            for (uint64_t t = 0; t < Tp; ++t)
                std::memcpy(att + t * D + hh * DK, outh + t * DK,
                            DK * sizeof(float));
            ws->fo -= Tp * DK;
        }
        lfm_f32_to_bf16(att, tmp, (int)(Tp * D));
        rc = bf16_linear(eng, ws, tmp, Tp, D, L.out_w, &L.out_b, h);
        if (rc != 0) return rc;
        lfm_add_bf16(x, h, Tp * D);

        // conv module: LN -> transpose -> pw1(f32) -> GLU -> dw k9(f32) -> BN
        // -> SiLU -> pw2(f32) -> transpose -> residual.
        if (lfm_ln_bf16(x, L.norm_conv_w.bf16, L.norm_conv_b.bf16, tmp, Tp, D,
                        1e-5f) != 0)
            return -EIO;
        uint16_t *ct = ws->b(D * Tp); // (D, Tp) channel-major
        for (uint64_t t = 0; t < Tp; ++t)
            for (uint64_t d = 0; d < D; ++d) ct[d * Tp + t] = tmp[t * D + d];
        float *xin = ws->f(D * Tp);
        lfm_bf16_widen_f32(ct, xin, D * Tp);
        float *w1 = ws->f(2 * D * D);
        lfm_bf16_widen_f32(L.pw1_w.bf16, w1, 2 * D * D);
        float *y1 = ws->f(2 * D * Tp);
        sgemm_rm(2 * D, Tp, D, w1, xin, y1);
        lfm_bias_bcast_f32(y1, L.pw1_b.bf16, 2 * D, Tp);
        uint16_t *y1b = ws->b(2 * D * Tp);
        lfm_f32_to_bf16(y1, y1b, (int)(2 * D * Tp));
        uint16_t *glu = ws->b(D * Tp);
        if (lfm_glu_bf16(y1b, y1b + D * Tp, glu, D * Tp) != 0) return -EIO;
        // depthwise k9 symmetric pad (K-1)/2, stride 1, f32.
        const uint64_t pad = (K - 1) / 2;
        float *xdw = ws->f(D * (Tp + 2 * pad));
        widen_pad_1d(glu, D, Tp, pad, xdw);
        float *wdw = ws->f(D * K);
        lfm_bf16_widen_f32(L.dw_w.bf16, wdw, D * K);
        float *ydw = ws->f(D * Tp);
        lfm_dwconv_tap_f32(xdw, wdw, ydw, D, Tp, Tp + 2 * pad, K, 1);
        lfm_bias_bcast_f32(ydw, L.dw_b.bf16, D, Tp);
        uint16_t *dwb = ws->b(D * Tp);
        lfm_f32_to_bf16(ydw, dwb, (int)(D * Tp));
        uint16_t *bnb = ws->b(D * Tp);
        lfm_bn_bf16(dwb, L.bn_mean.data(), L.bn_denom.data(), L.bn_w.data(),
                    L.bn_b.data(), bnb, D, Tp);
        if (lfm_silu_bf16(bnb, D * Tp) != 0) return -EIO;
        float *xin2 = ws->f(D * Tp);
        lfm_bf16_widen_f32(bnb, xin2, D * Tp);
        float *w2 = ws->f(D * D);
        lfm_bf16_widen_f32(L.pw2_w.bf16, w2, D * D);
        float *y2 = ws->f(D * Tp);
        sgemm_rm(D, Tp, D, w2, xin2, y2);
        lfm_bias_bcast_f32(y2, L.pw2_b.bf16, D, Tp);
        uint16_t *y2b = ws->b(D * Tp);
        lfm_f32_to_bf16(y2, y2b, (int)(D * Tp));
        for (uint64_t t = 0; t < Tp; ++t)
            for (uint64_t d = 0; d < D; ++d) tmp[t * D + d] = y2b[d * Tp + t];
        lfm_add_bf16(x, tmp, Tp * D);

        // ff2 half-step + norm_out.
        if (lfm_ln_bf16(x, L.norm_ff2_w.bf16, L.norm_ff2_b.bf16, tmp, Tp, D,
                        1e-5f) != 0)
            return -EIO;
        rc = bf16_linear(eng, ws, tmp, Tp, D, L.ff2_l1_w, &L.ff2_l1_b, h);
        if (rc != 0) return rc;
        if (lfm_silu_bf16(h, Tp * FF) != 0) return -EIO;
        rc = bf16_linear(eng, ws, h, Tp, FF, L.ff2_l2_w, &L.ff2_l2_b, tmp);
        if (rc != 0) return rc;
        lfm_residual_half_bf16(x, tmp, Tp * D);
        if (lfm_ln_bf16(x, L.norm_out_w.bf16, L.norm_out_b.bf16, tmp, Tp, D,
                        1e-5f) != 0)
            return -EIO;
        std::memcpy(x, tmp, (size_t)Tp * D * sizeof(uint16_t));

        ws->fo = f_mark;
        ws->bo = b_mark;
    }

    // ---- adapter: LN -> lin(2048x512)+b -> gelu_erf -> lin(2048x2048)+b ----
    if (lfm_ln_bf16(x, c->ad_ln_w.bf16, c->ad_ln_b.bf16, tmp, Tp, D, 1e-5f) != 0)
        return -EIO;
    uint16_t *ah = ws->b(Tp * g.adapter_hidden);
    rc = bf16_linear(eng, ws, tmp, Tp, D, c->ad_l1_w, &c->ad_l1_b, ah);
    if (rc != 0) return rc;
    if (lfm_gelu_erf_bf16(ah, Tp * g.adapter_hidden) != 0) return -EIO;
    rc = bf16_linear(eng, ws, ah, Tp, g.adapter_hidden, c->ad_l2_w, &c->ad_l2_b,
                     out_rows_dst);
    if (rc != 0) return rc;
    // A workspace overflow means the reserved bound was too small — a sizing
    // bug (never silent corruption). It would have handed out an aliased
    // pointer, so the result is untrustworthy: fail loud.
    if (ws->overflow) return -ENOMEM;
    return 0;
}
