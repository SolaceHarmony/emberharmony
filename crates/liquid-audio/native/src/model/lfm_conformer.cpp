// Native Conformer encoder + audio adapter. Contract: lfm_conformer.h.
//
// Law of this TU: C++ binds views, moves bytes (transposes, padding, im2col,
// rel_shift, head packing), and sequences stages. Every produced value comes
// from an assembly leaf (flashkern_conformer.S / flashkern_math.S), the
// engine's bf16 GEMM pass, or the approved f32 matmul dispatch
// (Accelerate on Apple, lfm_sgemm_f32 leaf elsewhere) for stages the
// reference ran in f32 (convolutions, attention).
//
// Ladder (fixtures arbitrate): bf16 linears = direct checkpoint-layout GEMM ->
// f32 bias -> bf16 round. Convs unlift bf16 activations/taps in registers ->
// f32 bias -> bf16 round -> activation in bf16. No weight is widened into a
// workspace plane. LayerNorm = f32 stats, bf16 weight/bias tail (layer_norm_slow).
// BatchNorm eval = all-bf16 broadcast chain (denominators prefolded at create
// with explicit bf16 rounding). Attention = f32 scores/softmax/aggregation;
// probabilities never round to bf16. SiLU/gelu_erf round once; GLU rounds the
// sigmoid, then the product.

#include "lfm_conformer.h"

#include "lfm_conformer_program.h"
#include "lfm_safetensors.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
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
int lfm_ln_bf16(const uint16_t *x, const void *w, const void *b,
                uint16_t *y, uint64_t rows, uint64_t cols, float eps);
void lfm_bn_bf16(const uint16_t *x, const void *mean, const uint16_t *denom,
                 const void *w, const void *b, uint16_t *y,
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
void lfm_dwconv_tap_bf16_f32(const uint16_t *xpad, const void *w, float *y,
                             uint64_t channels, uint64_t t_out,
                             uint64_t t_pad, uint64_t k, uint64_t stride);
void lfm_dwconv2d_k3s2_bf16_f32(const uint16_t *xpad, const void *w,
                                float *y, uint64_t channels,
                                uint64_t h_pad, uint64_t w_pad,
                                uint64_t h_out, uint64_t w_out);
void lfm_conv2d_stem_k3s2_bf16_f32(const uint16_t *xpad, const void *w,
                                   float *y, uint64_t channels,
                                   uint64_t h_pad, uint64_t w_pad,
                                   uint64_t h_out, uint64_t w_out);
void lfm_bias_rows_f32(float *y, const void *bias, uint64_t rows, uint64_t n);
void lfm_bias_bcast_f32(float *y, const void *bias, uint64_t channels,
                        uint64_t t);
void lfm_add_scale_f32(float *acc, const float *addend, uint64_t rows,
                       uint64_t cols, uint64_t addend_stride, float scale);
void lfm_add_bias_hd_bf16(const uint16_t *x, const void *bias, uint16_t *y,
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
inline uint16_t load_bf16(const unsigned char *p) {
    uint16_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}
inline uint16_t bf16_round_one(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t tie = (bits >> 16) & 1;
    bits += 0x7fffu + tie;
    return (uint16_t)(bits >> 16);
}

/* Distilled immutable checkpoint view. The byte pointer borrows the model's
 * sealed resident image; this record cannot allocate, convert, or repack it. */
struct View {
    const unsigned char *bytes = nullptr;
    uint64_t rows = 0, cols = 0, elements = 0;
};

struct LayerWeights {
    View norm_ff1_w, norm_ff1_b, ff1_l1_w, ff1_l1_b, ff1_l2_w, ff1_l2_b;
    View norm_att_w, norm_att_b;
    View q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, pos_w;
    View bias_u, bias_v;
    View norm_conv_w, norm_conv_b;
    View pw1_w, pw1_b, dw_w, dw_b, pw2_w, pw2_b;
    View bn_mean, bn_w, bn_b;
    const uint16_t *bn_denom = nullptr;
    View norm_ff2_w, norm_ff2_b, ff2_l1_w, ff2_l1_b, ff2_l2_w, ff2_l2_b;
    View norm_out_w, norm_out_b;
};

inline uint64_t conv_len(uint64_t l) { return l >= 1 ? (l + 2 - 3) / 2 + 1 : 0; }

} // namespace

struct LfmConformer {
    static constexpr size_t DERIVED_ALIGNMENT = 64;
    LfmConformerGeometry g;
    void *engine = nullptr;
    View stem_w, stem_b, dw1_w, dw1_b, pwa_w, pwa_b, dw2_w, dw2_b, pwb_w, pwb_b;
    View sub_out_w, sub_out_b;
    std::vector<LayerWeights> layers;
    View ad_ln_w, ad_ln_b, ad_l1_w, ad_l1_b, ad_l2_w, ad_l2_b;
    unsigned char *derived_arena = nullptr;
    size_t derived_arena_bytes = 0;
    float *pe_div = nullptr;
    uint64_t bound_weight_bytes = 0;
    uint64_t derived_bytes = 0;
    // Weight bytes MATERIALIZED rather than bound as a view — F32 staging, a
    // transpose/repack, an alignment copy, or any re-laid weight buffer. The
    // doctrine requires this to stay 0 in production, and the model-level
    // `compatibility_copied_bytes` acceptance gate reads it. It is a real tally,
    // not a constant: ANY code that materializes a weight MUST add its bytes
    // here, exactly as binding adds to `bound_weight_bytes` — otherwise the gate
    // silently stops being able to fail.
    uint64_t materialized_weight_bytes = 0;
    mutable std::atomic<uint64_t> direct_gemm_calls{0};

    ~LfmConformer() {
        ::operator delete(derived_arena,
                          std::align_val_t(DERIVED_ALIGNMENT));
    }
};

// One byte arena, allocated once at readiness. Numerical "planes" are only
// typed offset views into these bytes; they never own storage and the arena is
// not zero-filled for values that the leaves completely overwrite. f()/b()
// only advance offsets during a pass. A sealed production workspace cannot
// grow, so every pointer remains stable across the complete audio ticket.
struct LfmConformerWorkspace {
    static constexpr size_t ALIGNMENT = 64;
    unsigned char *arena = nullptr;
    size_t arena_bytes = 0;
    float *f32 = nullptr;
    uint16_t *u16 = nullptr;
    size_t fcap = 0, bcap = 0;
    size_t fo = 0, bo = 0;
    bool overflow = false;
    bool sealed = false;

    ~LfmConformerWorkspace() {
        ::operator delete(arena, std::align_val_t(ALIGNMENT));
    }

    int begin(size_t need_f, size_t need_b) {
        if (need_f > fcap || need_b > bcap) {
            if (sealed) return -ENOBUFS;
            if (need_f > SIZE_MAX / sizeof(float) ||
                need_b > SIZE_MAX / sizeof(uint16_t))
                return -EOVERFLOW;
            const size_t fbytes = need_f * sizeof(float);
            if (fbytes > SIZE_MAX - (ALIGNMENT - 1)) return -EOVERFLOW;
            const size_t boffset =
                (fbytes + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
            const size_t bbytes = need_b * sizeof(uint16_t);
            if (boffset > SIZE_MAX - bbytes) return -EOVERFLOW;
            const size_t bytes = boffset + bbytes;
            auto *next = static_cast<unsigned char *>(::operator new(
                bytes, std::align_val_t(ALIGNMENT), std::nothrow));
            if (!next) return -ENOMEM;
            ::operator delete(arena, std::align_val_t(ALIGNMENT));
            arena = next;
            arena_bytes = bytes;
            f32 = reinterpret_cast<float *>(arena);
            u16 = reinterpret_cast<uint16_t *>(arena + boffset);
            fcap = need_f;
            bcap = need_b;
        }
        fo = bo = 0;
        overflow = false;
        return 0;
    }
    float *f(size_t n) {
        if (n > fcap - fo) {
            overflow = true;
            return f32; // sizing defect; forward rejects this pass
        }
        float *p = f32 + fo;
        fo += n;
        return p;
    }
    uint16_t *b(size_t n) {
        if (n > bcap - bo) {
            overflow = true;
            return u16;
        }
        uint16_t *p = u16 + bo;
        bo += n;
        return p;
    }
};

namespace {

int workspace_needs(const LfmConformer *c, uint64_t frames,
                    size_t *out_f, size_t *out_b) {
    if (!c || !out_f || !out_b || frames == 0) return -EINVAL;
    const LfmConformerGeometry &g = c->g;
    const __uint128_t D = g.d_model, FF = g.d_ff, CC = g.conv_channels;
    const __uint128_t H = g.n_heads;
    const __uint128_t T1 = conv_len(frames), F1 = conv_len(g.feat_in);
    const __uint128_t T2 = conv_len((uint64_t)T1), F2 = conv_len((uint64_t)F1);
    const __uint128_t Tp = conv_len((uint64_t)T2), F3 = conv_len((uint64_t)F2);
    if (T1 == 0 || T2 == 0 || Tp == 0 || H == 0) return -EINVAL;
    const __uint128_t P = 2 * Tp - 1;
    const __uint128_t DK = D / H;
    const __uint128_t max_tf = T1 * F1;
    const __uint128_t width = FF > 2 * D ? FF : 2 * D;
    const __uint128_t attn = H * (Tp * Tp + 2 * Tp * P + 5 * Tp * DK + P * DK);
    const __uint128_t flat = CC * F3;
    const __uint128_t need_f =
        4 * CC * max_tf + 2 * attn + 8 * D * Tp + 4096;
    const __uint128_t need_b =
        6 * CC * max_tf + Tp * flat + 24 * Tp * width + 8 * P * D +
        4 * Tp * g.adapter_hidden + 4096;
    if (need_f > SIZE_MAX || need_b > SIZE_MAX) {
        return -EOVERFLOW;
    }
    *out_f = (size_t)need_f;
    *out_b = (size_t)need_b;
    return 0;
}

int bind(const LfmWeightImage *img, const std::string &name, View &v,
         std::initializer_list<uint64_t> expect, char *err, size_t errlen) {
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
    if (expect.size() != 0) {
        if (tv.rank != expect.size()) {
            std::snprintf(err, errlen, "conformer bind: '%s' rank %u != %zu",
                          name.c_str(), tv.rank, expect.size());
            return -ENOENT;
        }
        size_t i = 0;
        for (const uint64_t dimension : expect) {
            if (dimension != 0 && tv.shape[i] != dimension) {
                std::snprintf(err, errlen, "conformer bind: '%s' dim %zu mismatch",
                              name.c_str(), i);
                return -ENOENT;
            }
            ++i;
        }
    }
    v.bytes = static_cast<const unsigned char *>(tv.data);
    v.elements = tv.elements;
    v.rows = tv.rank >= 1 ? tv.shape[0] : 1;
    v.cols = tv.rank >= 2 ? tv.elements / tv.shape[0] : 1;
    return 0;
}

// Pad activation planes without changing dtype. The direct convolution leaves
// unlift both activations and resident taps at the MAC.
void pad_1d(const uint16_t *x, uint64_t c, uint64_t t, uint64_t pad,
            uint16_t *dst) {
    std::memset(dst, 0, (size_t)c * (t + 2 * pad) * sizeof(uint16_t));
    for (uint64_t ch = 0; ch < c; ++ch)
        std::memcpy(dst + ch * (t + 2 * pad) + pad, x + ch * t,
                    (size_t)t * sizeof(uint16_t));
}

void pad_2d(const uint16_t *x, uint64_t c, uint64_t h, uint64_t w,
            uint16_t *dst) {
    const uint64_t hp = h + 2, wp = w + 2;
    std::memset(dst, 0, (size_t)c * hp * wp * sizeof(uint16_t));
    for (uint64_t ch = 0; ch < c; ++ch)
        for (uint64_t y = 0; y < h; ++y)
            std::memcpy(dst + ch * hp * wp + (y + 1) * wp + 1,
                        x + (ch * h + y) * w,
                        (size_t)w * sizeof(uint16_t));
}

void channels_to_rows(const uint16_t *channels, uint64_t c, uint64_t t,
                      uint16_t *rows) {
    for (uint64_t i = 0; i < t; ++i)
        for (uint64_t ch = 0; ch < c; ++ch)
            rows[i * c + ch] = channels[ch * t + i];
}

void rows_to_channels(const uint16_t *rows, uint64_t t, uint64_t c,
                      uint16_t *channels) {
    for (uint64_t i = 0; i < t; ++i)
        for (uint64_t ch = 0; ch < c; ++ch)
            channels[ch * t + i] = rows[i * c + ch];
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

    // Formula-derived BN denominators and relative-position divisors share one
    // immutable byte arena. Layer records retain pointer views into it; no
    // numerical vector owns one table per layer.
    const __uint128_t bn_bytes_wide =
        (__uint128_t)g.n_layers * g.d_model * sizeof(uint16_t);
    const __uint128_t pe_bytes_wide =
        (__uint128_t)(g.d_model / 2) * sizeof(float);
    if (bn_bytes_wide > SIZE_MAX || pe_bytes_wide > SIZE_MAX) {
        delete c;
        return -EOVERFLOW;
    }
    const size_t bn_bytes = (size_t)bn_bytes_wide;
    if (bn_bytes > SIZE_MAX - (LfmConformer::DERIVED_ALIGNMENT - 1)) {
        delete c;
        return -EOVERFLOW;
    }
    const size_t pe_offset =
        (bn_bytes + (LfmConformer::DERIVED_ALIGNMENT - 1)) &
        ~(LfmConformer::DERIVED_ALIGNMENT - 1);
    const size_t pe_bytes = (size_t)pe_bytes_wide;
    if (pe_offset > SIZE_MAX - pe_bytes) {
        delete c;
        return -EOVERFLOW;
    }
    c->derived_arena_bytes = pe_offset + pe_bytes;
    c->derived_arena = static_cast<unsigned char *>(::operator new(
        c->derived_arena_bytes,
        std::align_val_t(LfmConformer::DERIVED_ALIGNMENT), std::nothrow));
    if (!c->derived_arena) {
        delete c;
        return -ENOMEM;
    }
    c->pe_div = reinterpret_cast<float *>(c->derived_arena + pe_offset);

    char localerr[256] = {0};
    char *err = (error && error_length) ? error : localerr;
    size_t errlen = (error && error_length) ? error_length : sizeof(localerr);
    const uint64_t D = g.d_model, FF = g.d_ff, CC = g.conv_channels;

    int rc = 0;
    auto B = [&](const std::string &n, View &v,
                 std::initializer_list<uint64_t> shape) {
        if (rc != 0) return;
        rc = bind(img, n, v, shape, err, errlen);
        if (rc == 0) c->bound_weight_bytes += v.elements * sizeof(uint16_t);
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

        View bn_v;
        B(p + "conv.batch_norm.weight", L.bn_w, {D});
        B(p + "conv.batch_norm.bias", L.bn_b, {D});
        B(p + "conv.batch_norm.running_mean", L.bn_mean, {D});
        B(p + "conv.batch_norm.running_var", bn_v, {D});
        if (rc == 0) {
            // BatchNorm eval prefold, candle op order with explicit bf16
            // rounding at each step: denom = bf16(sqrt(bf16(var + eps))).
            auto *denom = reinterpret_cast<uint16_t *>(c->derived_arena) +
                          (size_t)i * D;
            L.bn_denom = denom;
            for (uint64_t j = 0; j < D; ++j) {
                const float ve = bf16_widen_one(bf16_round_one(
                    bf16_widen_one(load_bf16(
                        bn_v.bytes + j * sizeof(uint16_t))) +
                    1e-5f));
                denom[j] = bf16_round_one(std::sqrt(ve));
            }
            // running_var is consumed only to build the denominator table; it
            // is not retained as a pass-time checkpoint view.
            c->bound_weight_bytes -= bn_v.elements * sizeof(uint16_t);
            c->derived_bytes += D * sizeof(uint16_t);
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
    for (uint64_t i = 0; i < D / 2; ++i)
        c->pe_div[i] =
            (float)std::exp(-(std::log(10000.0) / (double)D) * (double)(2 * i));
    c->derived_bytes += (D / 2) * sizeof(float);
    *out = c;
    return 0;
}

extern "C" int lfm_conformer_destroy(LfmConformer *c) {
    if (!c) return -EINVAL;
    delete c;
    return 0;
}

extern "C" uint64_t lfm_conformer_bound_weight_bytes(const LfmConformer *c) {
    return c ? c->bound_weight_bytes : 0;
}

extern "C" uint64_t lfm_conformer_derived_bytes(const LfmConformer *c) {
    return c ? c->derived_bytes : 0;
}

extern "C" uint64_t lfm_conformer_materialized_weight_bytes(const LfmConformer *c) {
    return c ? c->materialized_weight_bytes : 0;
}

extern "C" uint64_t lfm_conformer_direct_gemm_calls(const LfmConformer *c) {
    return c ? c->direct_gemm_calls.load(std::memory_order_relaxed) : 0;
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

extern "C" int lfm_conformer_workspace_reserve(const LfmConformer *c,
                                                 LfmConformerWorkspace *w,
                                                 uint64_t max_mel_frames) {
    if (!c || !w || max_mel_frames == 0) return -EINVAL;
    size_t need_f = 0, need_b = 0;
    const int status = workspace_needs(c, max_mel_frames, &need_f, &need_b);
    if (status != 0) return status;
    w->sealed = false;
    const int reserve_status = w->begin(need_f, need_b);
    w->sealed = true;
    return reserve_status;
}

extern "C" uint64_t lfm_conformer_out_rows(const LfmConformer *c, uint64_t t) {
    return (c && t) ? conv_len(conv_len(conv_len(t))) : 0;
}

extern "C" uint64_t lfm_conformer_out_width(const LfmConformer *c) {
    return c ? c->g.adapter_out : 0;
}

namespace {

enum : uint32_t {
    CP_STEM = 0,
    CP_DW_PREP = 1,
    CP_DW_GEMM = 2,
    CP_DW_FINISH = 3,
    CP_FLATTEN = 4,
    CP_SUB_OUT_GEMM = 5,
    CP_POSITION = 6,
    CP_FF1_PRE = 7,
    CP_FF1_L1_GEMM = 8,
    CP_FF1_ACT = 9,
    CP_FF1_L2_GEMM = 10,
    CP_ATTN_PRE = 11,
    CP_ATTN_Q_GEMM = 12,
    CP_ATTN_K_GEMM = 13,
    CP_ATTN_V_GEMM = 14,
    CP_ATTN_POS_GEMM = 15,
    CP_ATTN_BODY = 16,
    CP_ATTN_OUT_GEMM = 17,
    CP_CONV_PRE = 18,
    CP_CONV_PW1_GEMM = 19,
    CP_CONV_BODY = 20,
    CP_CONV_PW2_GEMM = 21,
    CP_FF2_PRE = 22,
    CP_FF2_L1_GEMM = 23,
    CP_FF2_ACT = 24,
    CP_FF2_L2_GEMM = 25,
    CP_LAYER_FINISH = 26,
    CP_ADAPTER_PRE = 27,
    CP_ADAPTER_L1_GEMM = 28,
    CP_ADAPTER_ACT = 29,
    CP_ADAPTER_L2_GEMM = 30,
    CP_DONE = 31,
};

struct ConformerProgramState {
    const LfmConformer *c = nullptr;
    LfmConformerWorkspace *ws = nullptr;
    const uint16_t *mel = nullptr;
    uint16_t *out = nullptr;
    uint64_t out_capacity = 0;
    uint64_t frames = 0;
    uint64_t d = 0, ff = 0, cc = 0, heads = 0, dk = 0, kernel = 0;
    uint64_t f0 = 0, t0 = 0, t1 = 0, f1 = 0, t2 = 0, f2 = 0;
    uint64_t tp = 0, f3 = 0, p = 0, flat = 0;
    uint32_t phase = CP_DONE;
    uint32_t next = CP_DONE;
    uint32_t layer = 0;
    uint32_t dw = 0;
    size_t f_mark = 0;
    size_t b_mark = 0;
    uint16_t *plane_a = nullptr;
    uint16_t *plane_b = nullptr;
    uint16_t *plane_c = nullptr;
    uint16_t *flat_rows = nullptr;
    uint16_t *x = nullptr;
    uint16_t *pe = nullptr;
    uint16_t *h = nullptr;
    uint16_t *tmp = nullptr;
    uint16_t *qkv = nullptr;
    uint16_t *pproj = nullptr;
    uint16_t *in_rows = nullptr;
    uint16_t *out_rows = nullptr;
    uint16_t *qu = nullptr;
    uint16_t *qv = nullptr;
    uint16_t *y1b = nullptr;
    uint16_t *pw2_in = nullptr;
    uint16_t *ah = nullptr;
};

static_assert(sizeof(ConformerProgramState) <=
                  sizeof(LfmConformerProgram::storage),
              "retained Conformer cursor exceeds the pass-slot storage");
static_assert(alignof(ConformerProgramState) <= alignof(LfmConformerProgram),
              "retained Conformer cursor alignment exceeds its storage");

ConformerProgramState *program_state(LfmConformerProgram *program) {
    return std::launder(reinterpret_cast<ConformerProgramState *>(
        program->storage));
}

const ConformerProgramState *program_state(
    const LfmConformerProgram *program) {
    return std::launder(reinterpret_cast<const ConformerProgramState *>(
        program->storage));
}

bool conformer_gemm_phase(uint32_t phase) {
    switch (phase) {
    case CP_DW_GEMM:
    case CP_SUB_OUT_GEMM:
    case CP_FF1_L1_GEMM:
    case CP_FF1_L2_GEMM:
    case CP_ATTN_Q_GEMM:
    case CP_ATTN_K_GEMM:
    case CP_ATTN_V_GEMM:
    case CP_ATTN_POS_GEMM:
    case CP_ATTN_OUT_GEMM:
    case CP_CONV_PW1_GEMM:
    case CP_CONV_PW2_GEMM:
    case CP_FF2_L1_GEMM:
    case CP_FF2_L2_GEMM:
    case CP_ADAPTER_L1_GEMM:
    case CP_ADAPTER_L2_GEMM:
        return true;
    default:
        return false;
    }
}

int conformer_gemm_desc(const ConformerProgramState &s,
                        LfmConformerGemmStage *stage) {
    if (!stage || !s.c || !s.ws || !conformer_gemm_phase(s.phase))
        return -EINVAL;
    const LfmConformer *c = s.c;
    const LayerWeights *layer = s.layer < c->layers.size()
        ? &c->layers[s.layer]
        : nullptr;
    const View *weight = nullptr;
    const View *bias = nullptr;
    const uint16_t *activation = nullptr;
    uint16_t *out = nullptr;
    uint64_t rows = 0, inner = 0;

    switch (s.phase) {
    case CP_DW_GEMM:
        weight = s.dw == 0 ? &c->pwa_w : &c->pwb_w;
        bias = s.dw == 0 ? &c->pwa_b : &c->pwb_b;
        activation = s.in_rows;
        out = s.out_rows;
        rows = s.dw == 0 ? s.t2 * s.f2 : s.tp * s.f3;
        inner = s.cc;
        break;
    case CP_SUB_OUT_GEMM:
        weight = &c->sub_out_w;
        bias = &c->sub_out_b;
        activation = s.flat_rows;
        out = s.x;
        rows = s.tp;
        inner = s.flat;
        break;
    case CP_FF1_L1_GEMM:
        weight = &layer->ff1_l1_w; bias = &layer->ff1_l1_b;
        activation = s.tmp; out = s.h; rows = s.tp; inner = s.d;
        break;
    case CP_FF1_L2_GEMM:
        weight = &layer->ff1_l2_w; bias = &layer->ff1_l2_b;
        activation = s.h; out = s.tmp; rows = s.tp; inner = s.ff;
        break;
    case CP_ATTN_Q_GEMM:
        weight = &layer->q_w; bias = &layer->q_b;
        activation = s.tmp; out = s.qkv; rows = s.tp; inner = s.d;
        break;
    case CP_ATTN_K_GEMM:
        weight = &layer->k_w; bias = &layer->k_b;
        activation = s.tmp; out = s.qkv + s.tp * s.d;
        rows = s.tp; inner = s.d;
        break;
    case CP_ATTN_V_GEMM:
        weight = &layer->v_w; bias = &layer->v_b;
        activation = s.tmp; out = s.qkv + 2 * s.tp * s.d;
        rows = s.tp; inner = s.d;
        break;
    case CP_ATTN_POS_GEMM:
        weight = &layer->pos_w;
        activation = s.pe; out = s.pproj; rows = s.p; inner = s.d;
        break;
    case CP_ATTN_OUT_GEMM:
        weight = &layer->out_w; bias = &layer->out_b;
        activation = s.tmp; out = s.h; rows = s.tp; inner = s.d;
        break;
    case CP_CONV_PW1_GEMM:
        weight = &layer->pw1_w; bias = &layer->pw1_b;
        activation = s.tmp; out = s.y1b; rows = s.tp; inner = s.d;
        break;
    case CP_CONV_PW2_GEMM:
        weight = &layer->pw2_w; bias = &layer->pw2_b;
        activation = s.pw2_in; out = s.tmp; rows = s.tp; inner = s.d;
        break;
    case CP_FF2_L1_GEMM:
        weight = &layer->ff2_l1_w; bias = &layer->ff2_l1_b;
        activation = s.tmp; out = s.h; rows = s.tp; inner = s.d;
        break;
    case CP_FF2_L2_GEMM:
        weight = &layer->ff2_l2_w; bias = &layer->ff2_l2_b;
        activation = s.h; out = s.tmp; rows = s.tp; inner = s.ff;
        break;
    case CP_ADAPTER_L1_GEMM:
        weight = &c->ad_l1_w; bias = &c->ad_l1_b;
        activation = s.tmp; out = s.ah; rows = s.tp; inner = s.d;
        break;
    case CP_ADAPTER_L2_GEMM:
        weight = &c->ad_l2_w; bias = &c->ad_l2_b;
        activation = s.ah; out = s.out; rows = s.tp;
        inner = c->g.adapter_hidden;
        break;
    default:
        return -EPROTO;
    }

    if (!weight || !weight->bytes || !activation || !out || rows == 0 ||
        inner == 0 || weight->cols != inner || weight->rows == 0 ||
        weight->elements != weight->rows * inner ||
        rows > SIZE_MAX / inner || rows > SIZE_MAX / weight->rows ||
        weight->elements > SIZE_MAX) {
        return -EINVAL;
    }
    *stage = {
        .activation = activation,
        .activation_count = static_cast<size_t>(rows * inner),
        .weight_bytes = weight->bytes,
        .weight_count = static_cast<size_t>(weight->elements),
        .bias_bytes = bias ? bias->bytes : nullptr,
        .bias_count = bias ? static_cast<size_t>(weight->rows) : 0,
        .out = out,
        .out_count = static_cast<size_t>(rows * weight->rows),
        .rows = static_cast<size_t>(rows),
        .columns = static_cast<size_t>(weight->rows),
        .inner = static_cast<size_t>(inner),
    };
    return 0;
}

} // namespace

int lfm_conformer_program_begin(
    LfmConformerProgram *program, const LfmConformer *c,
    LfmConformerWorkspace *ws, const uint16_t *mel, uint64_t mel_frames,
    uint16_t *out_rows, uint64_t out_capacity_values) {
    if (!program || !c || !ws || !mel || !out_rows || mel_frames == 0)
        return -EINVAL;
    if (!ws->sealed) return -EPERM;
    const LfmConformerGeometry &g = c->g;
    const uint64_t t1 = conv_len(mel_frames), f1 = conv_len(g.feat_in);
    const uint64_t t2 = conv_len(t1), f2 = conv_len(f1);
    const uint64_t tp = conv_len(t2), f3 = conv_len(f2);
    if (tp == 0 || t1 == 0 || t2 == 0 || g.n_heads == 0 ||
        g.d_model % g.n_heads != 0 || tp > UINT64_MAX / g.adapter_out ||
        out_capacity_values < tp * g.adapter_out) {
        return -EINVAL;
    }
    size_t need_f = 0, need_b = 0;
    int status = workspace_needs(c, mel_frames, &need_f, &need_b);
    if (status != 0) return status;
    status = ws->begin(need_f, need_b);
    if (status != 0) return status;

    std::memset(program->storage, 0, sizeof(program->storage));
    ConformerProgramState *s =
        ::new (program->storage) ConformerProgramState{};
    s->c = c;
    s->ws = ws;
    s->mel = mel;
    s->out = out_rows;
    s->out_capacity = out_capacity_values;
    s->frames = mel_frames;
    s->d = g.d_model;
    s->ff = g.d_ff;
    s->cc = g.conv_channels;
    s->heads = g.n_heads;
    s->dk = g.d_model / g.n_heads;
    s->kernel = g.conv_kernel;
    s->f0 = g.feat_in;
    s->t0 = mel_frames;
    s->t1 = t1;
    s->f1 = f1;
    s->t2 = t2;
    s->f2 = f2;
    s->tp = tp;
    s->f3 = f3;
    s->p = 2 * tp - 1;
    s->flat = s->cc * s->f3;
    s->phase = CP_STEM;
    s->next = CP_STEM;
    return 0;
}

uint32_t lfm_conformer_program_stage(const LfmConformerProgram *program) {
    if (!program) return LFM_CONFORMER_STAGE_DONE;
    const ConformerProgramState *s = program_state(program);
    if (!s->c || s->phase == CP_DONE) return LFM_CONFORMER_STAGE_DONE;
    return conformer_gemm_phase(s->phase) ? LFM_CONFORMER_STAGE_GEMM
                                          : LFM_CONFORMER_STAGE_SERIAL;
}

int lfm_conformer_program_gemm(const LfmConformerProgram *program,
                               LfmConformerGemmStage *stage) {
    if (!program) return -EINVAL;
    return conformer_gemm_desc(*program_state(program), stage);
}

int lfm_conformer_program_run_serial(LfmConformerProgram *program) {
    if (!program) return -EINVAL;
    ConformerProgramState &s = *program_state(program);
    if (!s.c || !s.ws || conformer_gemm_phase(s.phase) || s.phase == CP_DONE)
        return -EPROTO;
    const LfmConformer *c = s.c;
    LfmConformerWorkspace *ws = s.ws;
    const LayerWeights *layer = s.layer < c->layers.size()
        ? &c->layers[s.layer]
        : nullptr;

    switch (s.phase) {
    case CP_STEM: {
        s.plane_a = ws->b(s.cc * s.t1 * s.f1);
        const size_t fm = ws->fo, bm = ws->bo;
        const uint64_t hp = s.t0 + 2, wp = s.f0 + 2;
        uint16_t *img = ws->b(hp * wp);
        std::memset(img, 0, static_cast<size_t>(hp * wp) * sizeof(uint16_t));
        for (uint64_t f = 0; f < s.f0; ++f)
            for (uint64_t t = 0; t < s.t0; ++t)
                img[(t + 1) * wp + f + 1] = s.mel[f * s.t0 + t];
        float *stem = ws->f(s.cc * s.t1 * s.f1);
        lfm_conv2d_stem_k3s2_bf16_f32(img, c->stem_w.bytes, stem, s.cc,
                                      hp, wp, s.t1, s.f1);
        lfm_bias_bcast_f32(stem, c->stem_b.bytes, s.cc, s.t1 * s.f1);
        lfm_f32_to_bf16(stem, s.plane_a,
                        static_cast<int>(s.cc * s.t1 * s.f1));
        lfm_relu_bf16(s.plane_a, s.cc * s.t1 * s.f1);
        ws->fo = fm;
        ws->bo = bm;
        s.plane_b = ws->b(s.cc * s.t2 * s.f2);
        s.plane_c = ws->b(s.cc * s.tp * s.f3);
        s.dw = 0;
        s.next = CP_DW_PREP;
        return ws->overflow ? -ENOMEM : 0;
    }
    case CP_DW_PREP: {
        const bool first = s.dw == 0;
        const View &dw_w = first ? c->dw1_w : c->dw2_w;
        const View &dw_b = first ? c->dw1_b : c->dw2_b;
        uint16_t *in = first ? s.plane_a : s.plane_b;
        const uint64_t hi = first ? s.t1 : s.t2;
        const uint64_t wi = first ? s.f1 : s.f2;
        const uint64_t ho = first ? s.t2 : s.tp;
        const uint64_t wo = first ? s.f2 : s.f3;
        s.f_mark = ws->fo;
        s.b_mark = ws->bo;
        uint16_t *xpad = ws->b(s.cc * (hi + 2) * (wi + 2));
        pad_2d(in, s.cc, hi, wi, xpad);
        float *ydw = ws->f(s.cc * ho * wo);
        lfm_dwconv2d_k3s2_bf16_f32(xpad, dw_w.bytes, ydw, s.cc,
                                   hi + 2, wi + 2, ho, wo);
        lfm_bias_bcast_f32(ydw, dw_b.bytes, s.cc, ho * wo);
        uint16_t *dwb = ws->b(s.cc * ho * wo);
        lfm_f32_to_bf16(ydw, dwb, static_cast<int>(s.cc * ho * wo));
        const uint64_t positions = ho * wo;
        s.in_rows = ws->b(positions * s.cc);
        s.out_rows = ws->b(positions * s.cc);
        channels_to_rows(dwb, s.cc, positions, s.in_rows);
        s.next = CP_DW_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    }
    case CP_DW_FINISH: {
        const uint64_t positions = s.dw == 0 ? s.t2 * s.f2 : s.tp * s.f3;
        uint16_t *out = s.dw == 0 ? s.plane_b : s.plane_c;
        lfm_relu_bf16(s.out_rows, positions * s.cc);
        rows_to_channels(s.out_rows, positions, s.cc, out);
        ws->fo = s.f_mark;
        ws->bo = s.b_mark;
        if (s.dw == 0) {
            s.dw = 1;
            s.next = CP_DW_PREP;
        } else {
            s.next = CP_FLATTEN;
        }
        return 0;
    }
    case CP_FLATTEN:
        s.flat_rows = ws->b(s.tp * s.flat);
        for (uint64_t t = 0; t < s.tp; ++t)
            for (uint64_t ch = 0; ch < s.cc; ++ch)
                for (uint64_t f = 0; f < s.f3; ++f)
                    s.flat_rows[t * s.flat + ch * s.f3 + f] =
                        s.plane_c[(ch * s.tp + t) * s.f3 + f];
        s.x = ws->b(s.tp * s.d);
        s.next = CP_SUB_OUT_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    case CP_POSITION: {
        const uint64_t width = s.ff > 2 * s.d ? s.ff : 2 * s.d;
        s.pe = ws->b(s.p * s.d);
        if (lfm_pe_build_bf16(c->pe_div, s.d / 2, s.tp, s.pe) != 0)
            return -EIO;
        s.h = ws->b(s.tp * width);
        s.tmp = ws->b(s.tp * width);
        s.qkv = ws->b(3 * s.tp * s.d);
        s.pproj = ws->b(s.p * s.d);
        s.layer = 0;
        s.next = c->g.n_layers == 0 ? CP_ADAPTER_PRE : CP_FF1_PRE;
        return ws->overflow ? -ENOMEM : 0;
    }
    case CP_FF1_PRE:
        if (!layer) return -EPROTO;
        s.f_mark = ws->fo;
        s.b_mark = ws->bo;
        if (lfm_ln_bf16(s.x, layer->norm_ff1_w.bytes,
                        layer->norm_ff1_b.bytes, s.tmp, s.tp, s.d,
                        1e-5f) != 0)
            return -EIO;
        s.next = CP_FF1_L1_GEMM;
        return 0;
    case CP_FF1_ACT:
        if (lfm_silu_bf16(s.h, s.tp * s.ff) != 0) return -EIO;
        s.next = CP_FF1_L2_GEMM;
        return 0;
    case CP_ATTN_PRE:
        if (!layer) return -EPROTO;
        lfm_residual_half_bf16(s.x, s.tmp, s.tp * s.d);
        if (lfm_ln_bf16(s.x, layer->norm_att_w.bytes,
                        layer->norm_att_b.bytes, s.tmp, s.tp, s.d,
                        1e-5f) != 0)
            return -EIO;
        s.next = CP_ATTN_Q_GEMM;
        return 0;
    case CP_ATTN_BODY: {
        if (!layer) return -EPROTO;
        uint16_t *q = s.qkv;
        uint16_t *k = s.qkv + s.tp * s.d;
        uint16_t *v = s.qkv + 2 * s.tp * s.d;
        s.qu = ws->b(s.tp * s.d);
        s.qv = ws->b(s.tp * s.d);
        lfm_add_bias_hd_bf16(q, layer->bias_u.bytes, s.qu,
                             s.tp, s.heads, s.dk);
        lfm_add_bias_hd_bf16(q, layer->bias_v.bytes, s.qv,
                             s.tp, s.heads, s.dk);
        float *quf = ws->f(s.heads * s.tp * s.dk);
        float *qvf = ws->f(s.heads * s.tp * s.dk);
        float *kf = ws->f(s.heads * s.tp * s.dk);
        float *vf = ws->f(s.heads * s.tp * s.dk);
        float *pf = ws->f(s.heads * s.p * s.dk);
        for (uint64_t head = 0; head < s.heads; ++head)
            for (uint64_t t = 0; t < s.tp; ++t) {
                lfm_bf16_widen_f32(s.qu + t * s.d + head * s.dk,
                                   quf + (head * s.tp + t) * s.dk, s.dk);
                lfm_bf16_widen_f32(s.qv + t * s.d + head * s.dk,
                                   qvf + (head * s.tp + t) * s.dk, s.dk);
                lfm_bf16_widen_f32(k + t * s.d + head * s.dk,
                                   kf + (head * s.tp + t) * s.dk, s.dk);
                lfm_bf16_widen_f32(v + t * s.d + head * s.dk,
                                   vf + (head * s.tp + t) * s.dk, s.dk);
            }
        for (uint64_t head = 0; head < s.heads; ++head)
            for (uint64_t p = 0; p < s.p; ++p)
                lfm_bf16_widen_f32(s.pproj + p * s.d + head * s.dk,
                                   pf + (head * s.p + p) * s.dk, s.dk);
        float *ac = ws->f(s.heads * s.tp * s.tp);
        float *bd = ws->f(s.heads * s.tp * s.p);
        float *shifted = ws->f(s.heads * s.tp * s.p);
        for (uint64_t head = 0; head < s.heads; ++head) {
            sgemm_ntx(s.tp, s.tp, s.dk,
                      quf + head * s.tp * s.dk,
                      kf + head * s.tp * s.dk,
                      ac + head * s.tp * s.tp);
            sgemm_ntx(s.tp, s.p, s.dk,
                      qvf + head * s.tp * s.dk,
                      pf + head * s.p * s.dk,
                      bd + head * s.tp * s.p);
        }
        for (uint64_t head = 0; head < s.heads; ++head) {
            const float *src = bd + head * s.tp * s.p;
            float *dst = shifted + head * s.tp * s.p;
            for (uint64_t i = 0; i < s.tp * s.p; ++i) {
                const uint64_t index = i + s.tp;
                const uint64_t row = index / (s.p + 1);
                const uint64_t col = index % (s.p + 1);
                dst[i] = col == 0 ? 0.0f : src[row * s.p + col - 1];
            }
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(s.dk));
        for (uint64_t head = 0; head < s.heads; ++head)
            lfm_add_scale_f32(ac + head * s.tp * s.tp,
                              shifted + head * s.tp * s.p,
                              s.tp, s.tp, s.p, scale);
        if (lfm_softmax_rows_f32(ac, s.heads * s.tp, s.tp) != 0)
            return -EIO;
        float *att = ws->f(s.tp * s.d);
        for (uint64_t head = 0; head < s.heads; ++head) {
            float *outh = ws->f(s.tp * s.dk);
            sgemm_rm(s.tp, s.dk, s.tp,
                     ac + head * s.tp * s.tp,
                     vf + head * s.tp * s.dk, outh);
            for (uint64_t t = 0; t < s.tp; ++t)
                std::memcpy(att + t * s.d + head * s.dk,
                            outh + t * s.dk,
                            static_cast<size_t>(s.dk) * sizeof(float));
            ws->fo -= s.tp * s.dk;
        }
        lfm_f32_to_bf16(att, s.tmp, static_cast<int>(s.tp * s.d));
        s.next = CP_ATTN_OUT_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    }
    case CP_CONV_PRE:
        if (!layer) return -EPROTO;
        lfm_add_bf16(s.x, s.h, s.tp * s.d);
        if (lfm_ln_bf16(s.x, layer->norm_conv_w.bytes,
                        layer->norm_conv_b.bytes, s.tmp, s.tp, s.d,
                        1e-5f) != 0)
            return -EIO;
        s.y1b = ws->b(s.tp * 2 * s.d);
        s.next = CP_CONV_PW1_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    case CP_CONV_BODY: {
        if (!layer) return -EPROTO;
        uint16_t *glu_rows = ws->b(s.tp * s.d);
        for (uint64_t t = 0; t < s.tp; ++t)
            if (lfm_glu_bf16(s.y1b + t * 2 * s.d,
                             s.y1b + t * 2 * s.d + s.d,
                             glu_rows + t * s.d, s.d) != 0)
                return -EIO;
        uint16_t *glu = ws->b(s.d * s.tp);
        rows_to_channels(glu_rows, s.tp, s.d, glu);
        const uint64_t pad = (s.kernel - 1) / 2;
        uint16_t *xdw = ws->b(s.d * (s.tp + 2 * pad));
        pad_1d(glu, s.d, s.tp, pad, xdw);
        float *ydw = ws->f(s.d * s.tp);
        lfm_dwconv_tap_bf16_f32(xdw, layer->dw_w.bytes, ydw, s.d, s.tp,
                                s.tp + 2 * pad, s.kernel, 1);
        lfm_bias_bcast_f32(ydw, layer->dw_b.bytes, s.d, s.tp);
        uint16_t *dwb = ws->b(s.d * s.tp);
        lfm_f32_to_bf16(ydw, dwb, static_cast<int>(s.d * s.tp));
        uint16_t *bnb = ws->b(s.d * s.tp);
        lfm_bn_bf16(dwb, layer->bn_mean.bytes, layer->bn_denom,
                    layer->bn_w.bytes, layer->bn_b.bytes, bnb, s.d, s.tp);
        if (lfm_silu_bf16(bnb, s.d * s.tp) != 0) return -EIO;
        s.pw2_in = ws->b(s.tp * s.d);
        channels_to_rows(bnb, s.d, s.tp, s.pw2_in);
        s.next = CP_CONV_PW2_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    }
    case CP_FF2_PRE:
        if (!layer) return -EPROTO;
        lfm_add_bf16(s.x, s.tmp, s.tp * s.d);
        if (lfm_ln_bf16(s.x, layer->norm_ff2_w.bytes,
                        layer->norm_ff2_b.bytes, s.tmp, s.tp, s.d,
                        1e-5f) != 0)
            return -EIO;
        s.next = CP_FF2_L1_GEMM;
        return 0;
    case CP_FF2_ACT:
        if (lfm_silu_bf16(s.h, s.tp * s.ff) != 0) return -EIO;
        s.next = CP_FF2_L2_GEMM;
        return 0;
    case CP_LAYER_FINISH:
        if (!layer) return -EPROTO;
        lfm_residual_half_bf16(s.x, s.tmp, s.tp * s.d);
        if (lfm_ln_bf16(s.x, layer->norm_out_w.bytes,
                        layer->norm_out_b.bytes, s.tmp, s.tp, s.d,
                        1e-5f) != 0)
            return -EIO;
        std::swap(s.x, s.tmp);
        ws->fo = s.f_mark;
        ws->bo = s.b_mark;
        ++s.layer;
        s.next = s.layer < c->g.n_layers ? CP_FF1_PRE : CP_ADAPTER_PRE;
        return 0;
    case CP_ADAPTER_PRE:
        if (lfm_ln_bf16(s.x, c->ad_ln_w.bytes, c->ad_ln_b.bytes,
                        s.tmp, s.tp, s.d, 1e-5f) != 0)
            return -EIO;
        s.ah = ws->b(s.tp * c->g.adapter_hidden);
        s.next = CP_ADAPTER_L1_GEMM;
        return ws->overflow ? -ENOMEM : 0;
    case CP_ADAPTER_ACT:
        if (lfm_gelu_erf_bf16(s.ah,
                              s.tp * c->g.adapter_hidden) != 0)
            return -EIO;
        s.next = CP_ADAPTER_L2_GEMM;
        return 0;
    default:
        return -EPROTO;
    }
}

int lfm_conformer_program_advance(LfmConformerProgram *program) {
    if (!program) return -EINVAL;
    ConformerProgramState &s = *program_state(program);
    if (!s.c || s.phase == CP_DONE) return -EPROTO;
    if (conformer_gemm_phase(s.phase)) {
        s.c->direct_gemm_calls.fetch_add(1, std::memory_order_relaxed);
        switch (s.phase) {
        case CP_DW_GEMM: s.next = CP_DW_FINISH; break;
        case CP_SUB_OUT_GEMM: s.next = CP_POSITION; break;
        case CP_FF1_L1_GEMM: s.next = CP_FF1_ACT; break;
        case CP_FF1_L2_GEMM: s.next = CP_ATTN_PRE; break;
        case CP_ATTN_Q_GEMM: s.next = CP_ATTN_K_GEMM; break;
        case CP_ATTN_K_GEMM: s.next = CP_ATTN_V_GEMM; break;
        case CP_ATTN_V_GEMM: s.next = CP_ATTN_POS_GEMM; break;
        case CP_ATTN_POS_GEMM: s.next = CP_ATTN_BODY; break;
        case CP_ATTN_OUT_GEMM: s.next = CP_CONV_PRE; break;
        case CP_CONV_PW1_GEMM: s.next = CP_CONV_BODY; break;
        case CP_CONV_PW2_GEMM: s.next = CP_FF2_PRE; break;
        case CP_FF2_L1_GEMM: s.next = CP_FF2_ACT; break;
        case CP_FF2_L2_GEMM: s.next = CP_LAYER_FINISH; break;
        case CP_ADAPTER_L1_GEMM: s.next = CP_ADAPTER_ACT; break;
        case CP_ADAPTER_L2_GEMM:
            if (s.ws->overflow) return -ENOMEM;
            s.next = CP_DONE;
            break;
        default: return -EPROTO;
        }
    }
    s.phase = s.next;
    return s.phase == CP_DONE ? 0 : 1;
}
