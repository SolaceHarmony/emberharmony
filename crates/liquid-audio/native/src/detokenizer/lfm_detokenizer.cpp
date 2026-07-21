// Native LFM2.5 audio detokenizer.
//
// Contract pinned under native/vendor/liquid_audio/reference/detokenizer.py:
// eight codebook rows -> mean embedding -> repeat x6 -> 8-layer F32 LFM2 ->
// complex STFT projection -> same-padded ISTFT. Checkpoint weights are borrowed
// byte views into the model-owned read-only image. No tensor object, weight
// widening, alignment repair, transpose, or packed copy exists in this file.

#include "lfm_detokenizer.h"

#include "lfm_safetensors.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <string>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

#if defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64)
#include <arm_neon.h>
#define LFM_DETOK_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LFM_DETOK_SSE 1
#endif

extern "C" void lfm_sgemm_f32(const float *a, const float *b, float *c,
                               uint64_t m, uint64_t n, uint64_t k);
extern "C" void lfm_sgemm_nt_f32(const float *a, const float *bt, float *c,
                                  uint64_t m, uint64_t n, uint64_t k);

namespace {

constexpr size_t kCodebooks = LFM_DETOKENIZER_CODEBOOKS;
constexpr size_t kCodeValues = LFM_DETOKENIZER_CODE_VALUES;
constexpr size_t kRows = 6;
constexpr size_t kHidden = 512;
constexpr size_t kFfn = 2304;
constexpr size_t kLayers = 8;
constexpr size_t kConvLayers = 5;
constexpr size_t kAttentionLayers = 3;
constexpr size_t kHeads = 16;
constexpr size_t kKvHeads = 8;
constexpr size_t kHead = 32;
constexpr size_t kWindow = 30;
constexpr size_t kConvWidth = 3;
constexpr size_t kBins = 641;
constexpr size_t kProjection = 1282;
constexpr size_t kFft = 1280;
constexpr size_t kHop = 320;
constexpr size_t kPad = 480;
constexpr size_t kRing = 4096;
constexpr float kEpsilon = 1.0e-5f;
constexpr float kTheta = 1000000.0f;
constexpr float kAttentionScale = 0.1767766952966369f;
constexpr size_t kAlign = 64;
constexpr std::array<bool, kLayers> kAttention = {
    false, false, true, false, true, false, true, false,
};

struct F32View {
    const std::byte *bytes = nullptr;
    uint64_t elements = 0;
    uint64_t rows = 0;
    uint64_t columns = 0;
};

struct LayerPlan {
    F32View operator_norm;
    F32View ffn_norm;
    F32View w1;
    F32View w2;
    F32View w3;
    F32View in_proj;
    F32View conv;
    F32View out_proj;
    F32View q_proj;
    F32View k_proj;
    F32View v_proj;
    F32View q_norm;
    F32View k_norm;
    uint32_t state_index = 0;
    bool attention = false;
};

void set_error(char *error, size_t length, const char *message) {
    if (!error || length == 0) return;
    std::snprintf(error, length, "%s", message ? message : "unknown error");
}

void set_error(char *error, size_t length, const std::string &message) {
    set_error(error, length, message.c_str());
}

size_t aligned_bytes(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max() - (kAlign - 1)) return 0;
    return (bytes + (kAlign - 1)) & ~(kAlign - 1);
}

float load_f32(const F32View &view, uint64_t index) {
    float value = 0.0f;
    std::memcpy(&value, view.bytes + index * sizeof(float), sizeof(value));
    return value;
}

const float *f32_pointer(const F32View &view) {
    return reinterpret_cast<const float *>(view.bytes);
}

#if defined(LFM_DETOK_NEON)
float32x4_t load4(const F32View &view, uint64_t index) {
    return vreinterpretq_f32_u8(
        vld1q_u8(reinterpret_cast<const uint8_t *>(view.bytes) +
                 index * sizeof(float)));
}

float sum4(float32x4_t value) { return vaddvq_f32(value); }
#elif defined(LFM_DETOK_SSE)
__m128 load4(const F32View &view, uint64_t index) {
    __m128 value;
    std::memcpy(&value, view.bytes + index * sizeof(float), sizeof(value));
    return value;
}

float sum4(__m128 value) {
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, value);
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
#endif

int bind_f32(const LfmWeightImage *image, const std::string &name,
             std::initializer_list<uint64_t> shape, F32View *out,
             uint64_t *bound, char *error, size_t error_length) {
    if (!image || !out || !bound) return -EINVAL;
    LfmTensorView tensor = {
        .size = sizeof(LfmTensorView),
        .abi_version = LFM_WEIGHT_ABI_VERSION,
    };
    const int status = lfm_weights_find_component(
        image, LFM_WEIGHT_COMPONENT_DETOKENIZER, name.c_str(), &tensor);
    if (status != LFM_WEIGHT_OK) {
        set_error(error, error_length,
                  "detokenizer: missing required view '" + name + "'");
        return status == LFM_WEIGHT_NOT_FOUND ? -ENOENT : -EINVAL;
    }
    if (tensor.dtype != LFM_DTYPE_F32 || tensor.rank != shape.size()) {
        set_error(error, error_length,
                  "detokenizer: wrong dtype or rank for '" + name + "'");
        return -EINVAL;
    }
    size_t axis = 0;
    for (const uint64_t expected : shape) {
        if (tensor.shape[axis] != expected) {
            set_error(error, error_length,
                      "detokenizer: wrong shape for '" + name + "'");
            return -EINVAL;
        }
        ++axis;
    }
    if (!tensor.data || tensor.bytes != tensor.elements * sizeof(float) ||
        tensor.bytes > std::numeric_limits<uint64_t>::max() - *bound) {
        set_error(error, error_length,
                  "detokenizer: invalid byte span for '" + name + "'");
        return -EINVAL;
    }
    *out = {
        .bytes = static_cast<const std::byte *>(tensor.data),
        .elements = tensor.elements,
        .rows = tensor.rank > 1 ? tensor.shape[0] : 1,
        .columns = tensor.rank > 1 ? tensor.elements / tensor.shape[0]
                                   : tensor.elements,
    };
    *bound += tensor.bytes;
    return 0;
}

void linear(const float *input, size_t rows, const F32View &weight,
            float *output) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(rows), static_cast<int>(weight.rows),
                static_cast<int>(weight.columns), 1.0f, input,
                static_cast<int>(weight.columns), f32_pointer(weight),
                static_cast<int>(weight.columns), 0.0f, output,
                static_cast<int>(weight.rows));
#else
    lfm_sgemm_nt_f32(input, f32_pointer(weight), output, rows, weight.rows,
                     weight.columns);
#endif
}

void dense(const float *left, size_t rows, size_t inner,
           const float *right, size_t columns, float *output) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(rows), static_cast<int>(columns),
                static_cast<int>(inner), 1.0f, left, static_cast<int>(inner),
                right, static_cast<int>(columns), 0.0f, output,
                static_cast<int>(columns));
#else
    lfm_sgemm_f32(left, right, output, rows, columns, inner);
#endif
}

void add_bias(float *values, size_t rows, size_t columns,
              const F32View &bias) {
    for (size_t row = 0; row < rows; ++row) {
        float *dst = values + row * columns;
        size_t column = 0;
#if defined(LFM_DETOK_NEON)
        for (; column + 4 <= columns; column += 4) {
            vst1q_f32(dst + column,
                      vaddq_f32(vld1q_f32(dst + column),
                                load4(bias, column)));
        }
#elif defined(LFM_DETOK_SSE)
        for (; column + 4 <= columns; column += 4) {
            _mm_storeu_ps(dst + column,
                          _mm_add_ps(_mm_loadu_ps(dst + column),
                                     load4(bias, column)));
        }
#endif
        for (; column < columns; ++column) dst[column] += load_f32(bias, column);
    }
}

void copy_values(const float *source, float *destination, size_t count) {
    size_t index = 0;
#if defined(LFM_DETOK_NEON)
    for (; index + 16 <= count; index += 16) {
        const float32x4x4_t value = {
            {vld1q_f32(source + index), vld1q_f32(source + index + 4),
             vld1q_f32(source + index + 8),
             vld1q_f32(source + index + 12)}};
        vst1q_f32(destination + index, value.val[0]);
        vst1q_f32(destination + index + 4, value.val[1]);
        vst1q_f32(destination + index + 8, value.val[2]);
        vst1q_f32(destination + index + 12, value.val[3]);
    }
#elif defined(LFM_DETOK_SSE)
    for (; index + 16 <= count; index += 16) {
        _mm_storeu_ps(destination + index, _mm_loadu_ps(source + index));
        _mm_storeu_ps(destination + index + 4,
                      _mm_loadu_ps(source + index + 4));
        _mm_storeu_ps(destination + index + 8,
                      _mm_loadu_ps(source + index + 8));
        _mm_storeu_ps(destination + index + 12,
                      _mm_loadu_ps(source + index + 12));
    }
#endif
    for (; index < count; ++index) destination[index] = source[index];
}

void add_values(float *destination, const float *source, size_t count) {
    size_t index = 0;
#if defined(LFM_DETOK_NEON)
    for (; index + 4 <= count; index += 4)
        vst1q_f32(destination + index,
                  vaddq_f32(vld1q_f32(destination + index),
                            vld1q_f32(source + index)));
#elif defined(LFM_DETOK_SSE)
    for (; index + 4 <= count; index += 4)
        _mm_storeu_ps(destination + index,
                      _mm_add_ps(_mm_loadu_ps(destination + index),
                                 _mm_loadu_ps(source + index)));
#endif
    for (; index < count; ++index) destination[index] += source[index];
}

void rmsnorm(const float *input, size_t rows, size_t columns,
             const F32View &weight, float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const float *src = input + row * columns;
        float *dst = output + row * columns;
        float sum = 0.0f;
        size_t column = 0;
#if defined(LFM_DETOK_NEON)
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        for (; column + 8 <= columns; column += 8) {
            const float32x4_t a = vld1q_f32(src + column);
            const float32x4_t b = vld1q_f32(src + column + 4);
            acc0 = vmlaq_f32(acc0, a, a);
            acc1 = vmlaq_f32(acc1, b, b);
        }
        sum = sum4(vaddq_f32(acc0, acc1));
#elif defined(LFM_DETOK_SSE)
        __m128 acc0 = _mm_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        for (; column + 8 <= columns; column += 8) {
            const __m128 a = _mm_loadu_ps(src + column);
            const __m128 b = _mm_loadu_ps(src + column + 4);
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(a, a));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(b, b));
        }
        sum = sum4(_mm_add_ps(acc0, acc1));
#endif
        for (; column < columns; ++column) sum += src[column] * src[column];
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(columns) +
                                             kEpsilon);
        column = 0;
#if defined(LFM_DETOK_NEON)
        const float32x4_t gain = vdupq_n_f32(scale);
        for (; column + 4 <= columns; column += 4)
            vst1q_f32(dst + column,
                      vmulq_f32(vmulq_f32(vld1q_f32(src + column),
                                         load4(weight, column)),
                                gain));
#elif defined(LFM_DETOK_SSE)
        const __m128 gain = _mm_set1_ps(scale);
        for (; column + 4 <= columns; column += 4)
            _mm_storeu_ps(dst + column,
                          _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(src + column),
                                               load4(weight, column)),
                                     gain));
#endif
        for (; column < columns; ++column)
            dst[column] = src[column] * load_f32(weight, column) * scale;
    }
}

int swiglu(float *gate, const float *up, float *scratch, size_t count) {
#ifndef __APPLE__
    (void)gate;
    (void)up;
    (void)scratch;
    (void)count;
    return -ENOTSUP;
#else
    vDSP_vneg(gate, 1, scratch, 1, static_cast<vDSP_Length>(count));
    const int length = static_cast<int>(count);
    vvexpf(scratch, scratch, &length);
    size_t index = 0;
#if defined(LFM_DETOK_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; index + 4 <= count; index += 4) {
        const float32x4_t sigmoid =
            vdivq_f32(one, vaddq_f32(one, vld1q_f32(scratch + index)));
        vst1q_f32(gate + index,
                  vmulq_f32(vmulq_f32(vld1q_f32(gate + index), sigmoid),
                            vld1q_f32(up + index)));
    }
#elif defined(LFM_DETOK_SSE)
    const __m128 one = _mm_set1_ps(1.0f);
    for (; index + 4 <= count; index += 4) {
        const __m128 sigmoid =
            _mm_div_ps(one, _mm_add_ps(one, _mm_loadu_ps(scratch + index)));
        _mm_storeu_ps(gate + index,
                      _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(gate + index),
                                           sigmoid),
                                 _mm_loadu_ps(up + index)));
    }
#endif
    for (; index < count; ++index)
        gate[index] = gate[index] / (1.0f + scratch[index]) * up[index];
    return 0;
#endif
}

float dot32(const float *left, const float *right) {
#if defined(LFM_DETOK_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    for (size_t index = 0; index < kHead; index += 8) {
        a0 = vmlaq_f32(a0, vld1q_f32(left + index),
                       vld1q_f32(right + index));
        a1 = vmlaq_f32(a1, vld1q_f32(left + index + 4),
                       vld1q_f32(right + index + 4));
    }
    return sum4(vaddq_f32(a0, a1));
#elif defined(LFM_DETOK_SSE)
    __m128 a0 = _mm_setzero_ps();
    __m128 a1 = _mm_setzero_ps();
    for (size_t index = 0; index < kHead; index += 8) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(left + index),
                                       _mm_loadu_ps(right + index)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(left + index + 4),
                                       _mm_loadu_ps(right + index + 4)));
    }
    return sum4(_mm_add_ps(a0, a1));
#else
    return 0.0f;
#endif
}

int softmax(float *scores, size_t count) {
#ifndef __APPLE__
    (void)scores;
    (void)count;
    return -ENOTSUP;
#else
    float maximum = scores[0];
    for (size_t index = 1; index < count; ++index)
        maximum = std::max(maximum, scores[index]);
    size_t index = 0;
#if defined(LFM_DETOK_NEON)
    const float32x4_t maxv = vdupq_n_f32(maximum);
    for (; index + 4 <= count; index += 4)
        vst1q_f32(scores + index,
                  vsubq_f32(vld1q_f32(scores + index), maxv));
#elif defined(LFM_DETOK_SSE)
    const __m128 maxv = _mm_set1_ps(maximum);
    for (; index + 4 <= count; index += 4)
        _mm_storeu_ps(scores + index,
                      _mm_sub_ps(_mm_loadu_ps(scores + index), maxv));
#endif
    for (; index < count; ++index) scores[index] -= maximum;
    const int length = static_cast<int>(count);
    vvexpf(scores, scores, &length);
    float sum = 0.0f;
    index = 0;
#if defined(LFM_DETOK_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; index + 4 <= count; index += 4)
        acc = vaddq_f32(acc, vld1q_f32(scores + index));
    sum = sum4(acc);
#elif defined(LFM_DETOK_SSE)
    __m128 acc = _mm_setzero_ps();
    for (; index + 4 <= count; index += 4)
        acc = _mm_add_ps(acc, _mm_loadu_ps(scores + index));
    sum = sum4(acc);
#endif
    for (; index < count; ++index) sum += scores[index];
    const float inverse = 1.0f / sum;
    index = 0;
#if defined(LFM_DETOK_NEON)
    const float32x4_t inv = vdupq_n_f32(inverse);
    for (; index + 4 <= count; index += 4)
        vst1q_f32(scores + index,
                  vmulq_f32(vld1q_f32(scores + index), inv));
#elif defined(LFM_DETOK_SSE)
    const __m128 inv = _mm_set1_ps(inverse);
    for (; index + 4 <= count; index += 4)
        _mm_storeu_ps(scores + index,
                      _mm_mul_ps(_mm_loadu_ps(scores + index), inv));
#endif
    for (; index < count; ++index) scores[index] *= inverse;
    return 0;
#endif
}

int rope_angles(const LfmAudioDetokenizerPlan *plan, uint64_t position,
                float cosine[kHead / 2], float sine[kHead / 2]);

void apply_rope(float *values, const float cosine[kHead / 2],
                const float sine[kHead / 2]) {
#if defined(LFM_DETOK_NEON)
    for (size_t index = 0; index < kHead / 2; index += 4) {
        const float32x4_t a = vld1q_f32(values + index);
        const float32x4_t b = vld1q_f32(values + kHead / 2 + index);
        const float32x4_t c = vld1q_f32(cosine + index);
        const float32x4_t s = vld1q_f32(sine + index);
        vst1q_f32(values + index,
                  vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, s)));
        vst1q_f32(values + kHead / 2 + index,
                  vaddq_f32(vmulq_f32(b, c), vmulq_f32(a, s)));
    }
#elif defined(LFM_DETOK_SSE)
    for (size_t index = 0; index < kHead / 2; index += 4) {
        const __m128 a = _mm_loadu_ps(values + index);
        const __m128 b = _mm_loadu_ps(values + kHead / 2 + index);
        const __m128 c = _mm_load_ps(cosine + index);
        const __m128 s = _mm_load_ps(sine + index);
        _mm_storeu_ps(values + index,
                      _mm_sub_ps(_mm_mul_ps(a, c), _mm_mul_ps(b, s)));
        _mm_storeu_ps(values + kHead / 2 + index,
                      _mm_add_ps(_mm_mul_ps(b, c), _mm_mul_ps(a, s)));
    }
#endif
}

} // namespace

struct LfmAudioDetokenizerPlan {
    F32View embedding;
    F32View final_norm;
    F32View projection;
    F32View bias;
    F32View window;
    std::array<LayerPlan, kLayers> layers{};
    std::array<float, kHead / 2> rope_inverse{};
    float *ifft_basis = nullptr;
    size_t ifft_basis_bytes = 0;
    uint64_t bound_bytes = 0;
    uint64_t derived_bytes = 0;
    uint64_t compatibility_copied_bytes = 0;
};

struct LfmAudioDetokenizerState {
    const LfmAudioDetokenizerPlan *plan = nullptr;
    float *memory = nullptr;
    size_t memory_bytes = 0;
    size_t persistent_floats = 0;
    float *conv = nullptr;
    float *keys = nullptr;
    float *values = nullptr;
    float *x = nullptr;
    float *norm = nullptr;
    float *wide0 = nullptr;
    float *wide1 = nullptr;
    float *wide2 = nullptr;
    float *terminal = nullptr;
    float *spectral = nullptr;
    float *ifft = nullptr;
    float *ola = nullptr;
    float *envelope = nullptr;
    uint64_t position = 0;
    uint64_t frames = 0;
    uint64_t emitted_raw = kPad;
    bool prefix_cleared = false;
    bool flushed = false;
};

namespace {

int rope_angles(const LfmAudioDetokenizerPlan *plan, uint64_t position,
                float cosine[kHead / 2], float sine[kHead / 2]) {
#ifndef __APPLE__
    (void)plan;
    (void)position;
    (void)cosine;
    (void)sine;
    return -ENOTSUP;
#else
    alignas(64) float angles[kHead / 2];
    for (size_t index = 0; index < kHead / 2; ++index)
        angles[index] =
            static_cast<float>(position) * plan->rope_inverse[index];
    const int length = static_cast<int>(kHead / 2);
    vvsincosf(sine, cosine, angles, &length);
    return 0;
#endif
}

float *carve(float **cursor, size_t count) {
    float *result = *cursor;
    *cursor += count;
    return result;
}

int run_mlp(LfmAudioDetokenizerState *state, const LayerPlan &layer) {
    rmsnorm(state->x, kRows, kHidden, layer.ffn_norm, state->norm);
    linear(state->norm, kRows, layer.w1, state->wide0);
    linear(state->norm, kRows, layer.w3, state->wide1);
    int status = swiglu(state->wide0, state->wide1, state->wide2,
                        kRows * kFfn);
    if (status != 0) return status;
    linear(state->wide0, kRows, layer.w2, state->terminal);
    add_values(state->x, state->terminal, kRows * kHidden);
    return 0;
}

void gathered_taps(const F32View &weight, size_t channel, float values[3]) {
    values[0] = load_f32(weight, channel * 3);
    values[1] = load_f32(weight, channel * 3 + 1);
    values[2] = load_f32(weight, channel * 3 + 2);
}

int run_conv(LfmAudioDetokenizerState *state, const LayerPlan &layer) {
    rmsnorm(state->x, kRows, kHidden, layer.operator_norm, state->norm);
    linear(state->norm, kRows, layer.in_proj, state->wide2);
    float *carry = state->conv + layer.state_index * kConvWidth * kHidden;
    for (size_t row = 0; row < kRows; ++row) {
        const float *projected = state->wide2 + row * 3 * kHidden;
        const float *gate = projected;
        const float *coefficient = projected + kHidden;
        const float *value = projected + 2 * kHidden;
        float *output = state->norm + row * kHidden;
        size_t channel = 0;
#if defined(LFM_DETOK_NEON)
        for (; channel + 4 <= kHidden; channel += 4) {
            const float32x4_t bx =
                vmulq_f32(vld1q_f32(gate + channel),
                           vld1q_f32(value + channel));
            const float32x4_t s0 =
                vld1q_f32(carry + 1 * kHidden + channel);
            const float32x4_t s1 =
                vld1q_f32(carry + 2 * kHidden + channel);
            float taps[3][4];
            for (size_t lane = 0; lane < 4; ++lane)
                gathered_taps(layer.conv, channel + lane, taps[lane]);
            const float32x4_t w0 = {taps[0][0], taps[1][0], taps[2][0],
                                    taps[3][0]};
            const float32x4_t w1 = {taps[0][1], taps[1][1], taps[2][1],
                                    taps[3][1]};
            const float32x4_t w2 = {taps[0][2], taps[1][2], taps[2][2],
                                    taps[3][2]};
            const float32x4_t convolved =
                vmlaq_f32(vmlaq_f32(vmulq_f32(s0, w0), s1, w1), bx, w2);
            vst1q_f32(output + channel,
                      vmulq_f32(vld1q_f32(coefficient + channel), convolved));
            vst1q_f32(carry + 0 * kHidden + channel, s0);
            vst1q_f32(carry + 1 * kHidden + channel, s1);
            vst1q_f32(carry + 2 * kHidden + channel, bx);
        }
#elif defined(LFM_DETOK_SSE)
        for (; channel + 4 <= kHidden; channel += 4) {
            const __m128 bx = _mm_mul_ps(_mm_loadu_ps(gate + channel),
                                         _mm_loadu_ps(value + channel));
            const __m128 s0 = _mm_loadu_ps(carry + kHidden + channel);
            const __m128 s1 = _mm_loadu_ps(carry + 2 * kHidden + channel);
            float taps[4][3];
            for (size_t lane = 0; lane < 4; ++lane)
                gathered_taps(layer.conv, channel + lane, taps[lane]);
            const __m128 w0 = _mm_setr_ps(taps[0][0], taps[1][0], taps[2][0],
                                          taps[3][0]);
            const __m128 w1 = _mm_setr_ps(taps[0][1], taps[1][1], taps[2][1],
                                          taps[3][1]);
            const __m128 w2 = _mm_setr_ps(taps[0][2], taps[1][2], taps[2][2],
                                          taps[3][2]);
            const __m128 convolved =
                _mm_add_ps(_mm_add_ps(_mm_mul_ps(s0, w0), _mm_mul_ps(s1, w1)),
                           _mm_mul_ps(bx, w2));
            _mm_storeu_ps(output + channel,
                          _mm_mul_ps(_mm_loadu_ps(coefficient + channel),
                                     convolved));
            _mm_storeu_ps(carry + channel, s0);
            _mm_storeu_ps(carry + kHidden + channel, s1);
            _mm_storeu_ps(carry + 2 * kHidden + channel, bx);
        }
#endif
        for (; channel < kHidden; ++channel) {
            const float bx = gate[channel] * value[channel];
            const float s0 = carry[kHidden + channel];
            const float s1 = carry[2 * kHidden + channel];
            float taps[3];
            gathered_taps(layer.conv, channel, taps);
            output[channel] = coefficient[channel] *
                              (s0 * taps[0] + s1 * taps[1] + bx * taps[2]);
            carry[channel] = s0;
            carry[kHidden + channel] = s1;
            carry[2 * kHidden + channel] = bx;
        }
    }
    linear(state->norm, kRows, layer.out_proj, state->terminal);
    add_values(state->x, state->terminal, kRows * kHidden);
    return run_mlp(state, layer);
}

int run_attention(LfmAudioDetokenizerState *state, const LayerPlan &layer) {
    rmsnorm(state->x, kRows, kHidden, layer.operator_norm, state->norm);
    linear(state->norm, kRows, layer.q_proj, state->wide0);
    linear(state->norm, kRows, layer.k_proj, state->wide1);
    linear(state->norm, kRows, layer.v_proj, state->wide2);
    float *keys = state->keys + layer.state_index * kWindow * kKvHeads * kHead;
    float *values =
        state->values + layer.state_index * kWindow * kKvHeads * kHead;
    for (size_t row = 0; row < kRows; ++row) {
        const uint64_t absolute = state->position + row;
        float *query = state->wide0 + row * kHidden;
        float *key = state->wide1 + row * kKvHeads * kHead;
        float *value = state->wide2 + row * kKvHeads * kHead;
        rmsnorm(query, kHeads, kHead, layer.q_norm, query);
        rmsnorm(key, kKvHeads, kHead, layer.k_norm, key);
        alignas(64) float cosine[kHead / 2];
        alignas(64) float sine[kHead / 2];
        const int angle_status =
            rope_angles(state->plan, absolute, cosine, sine);
        if (angle_status != 0) return angle_status;
        for (size_t head = 0; head < kHeads; ++head)
            apply_rope(query + head * kHead, cosine, sine);
        for (size_t head = 0; head < kKvHeads; ++head)
            apply_rope(key + head * kHead, cosine, sine);
        const size_t ring = static_cast<size_t>(absolute % kWindow);
        copy_values(key, keys + ring * kKvHeads * kHead, kKvHeads * kHead);
        copy_values(value, values + ring * kKvHeads * kHead, kKvHeads * kHead);
        const size_t count = static_cast<size_t>(std::min<uint64_t>(
            kWindow, absolute + 1));
        const uint64_t first = absolute + 1 - count;
        for (size_t head = 0; head < kHeads; ++head) {
            alignas(64) float score[kWindow];
            const float *q = query + head * kHead;
            const size_t kv_head = head / (kHeads / kKvHeads);
            for (size_t item = 0; item < count; ++item) {
                const size_t source =
                    static_cast<size_t>((first + item) % kWindow);
                score[item] =
                    dot32(q, keys + (source * kKvHeads + kv_head) * kHead) *
                    kAttentionScale;
            }
            const int status = softmax(score, count);
            if (status != 0) return status;
            float *destination = state->norm + row * kHidden + head * kHead;
#if defined(LFM_DETOK_NEON)
            std::array<float32x4_t, kHead / 4> acc{};
            for (auto &value_acc : acc) value_acc = vdupq_n_f32(0.0f);
            for (size_t item = 0; item < count; ++item) {
                const size_t source =
                    static_cast<size_t>((first + item) % kWindow);
                const float *v =
                    values + (source * kKvHeads + kv_head) * kHead;
                for (size_t lane = 0; lane < acc.size(); ++lane)
                    acc[lane] = vmlaq_n_f32(acc[lane],
                                            vld1q_f32(v + lane * 4), score[item]);
            }
            for (size_t lane = 0; lane < acc.size(); ++lane)
                vst1q_f32(destination + lane * 4, acc[lane]);
#elif defined(LFM_DETOK_SSE)
            std::array<__m128, kHead / 4> acc{};
            for (auto &value_acc : acc) value_acc = _mm_setzero_ps();
            for (size_t item = 0; item < count; ++item) {
                const size_t source =
                    static_cast<size_t>((first + item) % kWindow);
                const float *v =
                    values + (source * kKvHeads + kv_head) * kHead;
                const __m128 scale = _mm_set1_ps(score[item]);
                for (size_t lane = 0; lane < acc.size(); ++lane)
                    acc[lane] = _mm_add_ps(
                        acc[lane], _mm_mul_ps(_mm_loadu_ps(v + lane * 4), scale));
            }
            for (size_t lane = 0; lane < acc.size(); ++lane)
                _mm_storeu_ps(destination + lane * 4, acc[lane]);
#endif
        }
    }
    linear(state->norm, kRows, layer.out_proj, state->terminal);
    add_values(state->x, state->terminal, kRows * kHidden);
    return run_mlp(state, layer);
}

void embed_codes(LfmAudioDetokenizerState *state,
                 const uint32_t codes[kCodebooks]) {
    alignas(64) float fused[kHidden];
    for (size_t column = 0; column < kHidden; column += 4) {
#if defined(LFM_DETOK_NEON)
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (size_t codebook = 0; codebook < kCodebooks; ++codebook) {
            const uint64_t row = codebook * kCodeValues + codes[codebook];
            sum = vaddq_f32(sum,
                            load4(state->plan->embedding, row * kHidden + column));
        }
        vst1q_f32(fused + column, vmulq_n_f32(sum, 1.0f / kCodebooks));
#elif defined(LFM_DETOK_SSE)
        __m128 sum = _mm_setzero_ps();
        for (size_t codebook = 0; codebook < kCodebooks; ++codebook) {
            const uint64_t row = codebook * kCodeValues + codes[codebook];
            sum = _mm_add_ps(
                sum, load4(state->plan->embedding, row * kHidden + column));
        }
        _mm_store_ps(fused + column,
                     _mm_mul_ps(sum, _mm_set1_ps(1.0f / kCodebooks)));
#endif
    }
    for (size_t row = 0; row < kRows; ++row)
        copy_values(fused, state->x + row * kHidden, kHidden);
}

int polar_spectrum(LfmAudioDetokenizerState *state) {
#ifndef __APPLE__
    (void)state;
    return -ENOTSUP;
#else
    const size_t count = kRows * kBins;
    float *magnitude = state->wide2;
    float *sine = state->wide2 + count;
    float *cosine = state->wide0;
    float *angles = state->wide1;
    for (size_t row = 0; row < kRows; ++row) {
        copy_values(state->spectral + row * kProjection,
                    magnitude + row * kBins, kBins);
        copy_values(state->spectral + row * kProjection + kBins,
                    angles + row * kBins, kBins);
    }
    const int length = static_cast<int>(count);
    vvexpf(magnitude, magnitude, &length);
    vvsincosf(sine, cosine, angles, &length);
    for (size_t index = 0; index < count; ++index) {
        const float scale = magnitude[index];
        const size_t row = index / kBins;
        const size_t bin = index % kBins;
        state->spectral[row * kProjection + bin] = scale * cosine[index];
        state->spectral[row * kProjection + kBins + bin] =
            scale * sine[index];
    }
    return 0;
#endif
}

void clear_ring(LfmAudioDetokenizerState *state, uint64_t first,
                uint64_t end) {
    while (first < end) {
        const size_t offset = static_cast<size_t>(first % kRing);
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(end - first, kRing - offset));
        std::memset(state->ola + offset, 0, count * sizeof(float));
        std::memset(state->envelope + offset, 0, count * sizeof(float));
        first += count;
    }
}

void overlap_add(LfmAudioDetokenizerState *state) {
    for (size_t row = 0; row < kRows; ++row) {
        const uint64_t start = state->frames * kHop;
        size_t done = 0;
        while (done < kFft) {
            const size_t offset = static_cast<size_t>((start + done) % kRing);
            const size_t count = std::min(kFft - done, kRing - offset);
            size_t index = 0;
#if defined(LFM_DETOK_NEON)
            for (; index + 4 <= count; index += 4) {
                const float32x4_t window =
                    load4(state->plan->window, done + index);
                const float32x4_t signal =
                    vld1q_f32(state->ifft + row * kFft + done + index);
                vst1q_f32(state->ola + offset + index,
                          vmlaq_f32(vld1q_f32(state->ola + offset + index),
                                    signal, window));
                vst1q_f32(
                    state->envelope + offset + index,
                    vmlaq_f32(vld1q_f32(state->envelope + offset + index),
                              window, window));
            }
#elif defined(LFM_DETOK_SSE)
            for (; index + 4 <= count; index += 4) {
                const __m128 window = load4(state->plan->window, done + index);
                const __m128 signal =
                    _mm_loadu_ps(state->ifft + row * kFft + done + index);
                _mm_storeu_ps(
                    state->ola + offset + index,
                    _mm_add_ps(_mm_loadu_ps(state->ola + offset + index),
                               _mm_mul_ps(signal, window)));
                _mm_storeu_ps(
                    state->envelope + offset + index,
                    _mm_add_ps(_mm_loadu_ps(state->envelope + offset + index),
                               _mm_mul_ps(window, window)));
            }
#endif
            for (; index < count; ++index) {
                const float window = load_f32(state->plan->window, done + index);
                state->ola[offset + index] +=
                    state->ifft[row * kFft + done + index] * window;
                state->envelope[offset + index] += window * window;
            }
            done += count;
        }
        ++state->frames;
    }
}

int emit_range(LfmAudioDetokenizerState *state, uint64_t end, float *pcm,
               size_t capacity, size_t *out_samples) {
    if (end < state->emitted_raw || end - state->emitted_raw > capacity) {
        return -ENOSPC;
    }
    const size_t total = static_cast<size_t>(end - state->emitted_raw);
    uint64_t cursor = state->emitted_raw;
    size_t written = 0;
    while (cursor < end) {
        const size_t offset = static_cast<size_t>(cursor % kRing);
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(end - cursor, kRing - offset));
        size_t index = 0;
#if defined(LFM_DETOK_NEON)
        for (; index + 4 <= count; index += 4) {
            const float32x4_t envelope =
                vld1q_f32(state->envelope + offset + index);
            if (vminvq_f32(envelope) <= 1.0e-11f) return -ERANGE;
            vst1q_f32(pcm + written + index,
                      vdivq_f32(vld1q_f32(state->ola + offset + index),
                                envelope));
        }
#elif defined(LFM_DETOK_SSE)
        for (; index + 4 <= count; index += 4) {
            const __m128 envelope =
                _mm_loadu_ps(state->envelope + offset + index);
            alignas(16) float check[4];
            _mm_store_ps(check, envelope);
            if (*std::min_element(check, check + 4) <= 1.0e-11f)
                return -ERANGE;
            _mm_storeu_ps(pcm + written + index,
                          _mm_div_ps(_mm_loadu_ps(state->ola + offset + index),
                                     envelope));
        }
#endif
        for (; index < count; ++index) {
            const float envelope = state->envelope[offset + index];
            if (envelope <= 1.0e-11f) return -ERANGE;
            pcm[written + index] = state->ola[offset + index] / envelope;
        }
        std::memset(state->ola + offset, 0, count * sizeof(float));
        std::memset(state->envelope + offset, 0, count * sizeof(float));
        cursor += count;
        written += count;
    }
    state->emitted_raw = end;
    *out_samples = total;
    return 0;
}

} // namespace

extern "C" int lfm_detokenizer_plan_new_from_image(
    LfmAudioDetokenizerPlan **out, const LfmWeightImage *image, char *error,
    size_t error_length) {
    if (!out || !image) return -EINVAL;
    *out = nullptr;
#if !defined(LFM_DETOK_NEON) && !defined(LFM_DETOK_SSE)
    set_error(error, error_length,
              "detokenizer: no supported architecture vector ISA");
    return -ENOTSUP;
#endif
#ifndef __APPLE__
    set_error(error, error_length,
              "detokenizer: production vForce/Accelerate backend unavailable");
    return -ENOTSUP;
#endif
    if (lfm_weights_component_count(image, LFM_WEIGHT_COMPONENT_DETOKENIZER) !=
        79) {
        set_error(error, error_length,
                  "detokenizer: checkpoint must contain exactly 79 views");
        return -EINVAL;
    }
    LfmAudioDetokenizerPlan *plan =
        new (std::nothrow) LfmAudioDetokenizerPlan();
    if (!plan) return -ENOMEM;
    uint64_t bound = 0;
    uint64_t validated_unused = 0;
    F32View unused_input_embedding{};
    int status = bind_f32(image, "emb.emb.weight", {16384, 512},
                          &plan->embedding, &bound, error, error_length);
    if (status == 0)
        status = bind_f32(image, "lfm.embed_tokens.weight", {65536, 512},
                          &unused_input_embedding, &validated_unused, error,
                          error_length);
    if (status == 0)
        status = bind_f32(image, "lfm.embedding_norm.weight", {512},
                          &plan->final_norm, &bound, error, error_length);
    if (status == 0)
        status = bind_f32(image, "lin.weight", {1282, 512},
                          &plan->projection, &bound, error, error_length);
    if (status == 0)
        status = bind_f32(image, "lin.bias", {1282}, &plan->bias, &bound,
                          error, error_length);
    if (status == 0)
        status = bind_f32(image, "istft.window", {1280}, &plan->window,
                          &bound, error, error_length);
    uint32_t conv_index = 0;
    uint32_t attention_index = 0;
    for (size_t index = 0; status == 0 && index < kLayers; ++index) {
        LayerPlan &layer = plan->layers[index];
        layer.attention = kAttention[index];
        layer.state_index = layer.attention ? attention_index++ : conv_index++;
        const std::string prefix = "lfm.layers." + std::to_string(index) + ".";
        status = bind_f32(image, prefix + "operator_norm.weight", {512},
                          &layer.operator_norm, &bound, error, error_length);
        if (status == 0)
            status = bind_f32(image, prefix + "ffn_norm.weight", {512},
                              &layer.ffn_norm, &bound, error, error_length);
        if (status == 0)
            status = bind_f32(image, prefix + "feed_forward.w1.weight",
                              {2304, 512}, &layer.w1, &bound, error,
                              error_length);
        if (status == 0)
            status = bind_f32(image, prefix + "feed_forward.w2.weight",
                              {512, 2304}, &layer.w2, &bound, error,
                              error_length);
        if (status == 0)
            status = bind_f32(image, prefix + "feed_forward.w3.weight",
                              {2304, 512}, &layer.w3, &bound, error,
                              error_length);
        if (status != 0) break;
        if (layer.attention) {
            status = bind_f32(image, prefix + "self_attn.q_proj.weight",
                              {512, 512}, &layer.q_proj, &bound, error,
                              error_length);
            if (status == 0)
                status = bind_f32(image, prefix + "self_attn.k_proj.weight",
                                  {256, 512}, &layer.k_proj, &bound, error,
                                  error_length);
            if (status == 0)
                status = bind_f32(image, prefix + "self_attn.v_proj.weight",
                                  {256, 512}, &layer.v_proj, &bound, error,
                                  error_length);
            if (status == 0)
                status = bind_f32(image, prefix + "self_attn.out_proj.weight",
                                  {512, 512}, &layer.out_proj, &bound, error,
                                  error_length);
            if (status == 0)
                status = bind_f32(
                    image, prefix + "self_attn.q_layernorm.weight", {32},
                    &layer.q_norm, &bound, error, error_length);
            if (status == 0)
                status = bind_f32(
                    image, prefix + "self_attn.k_layernorm.weight", {32},
                    &layer.k_norm, &bound, error, error_length);
        } else {
            status = bind_f32(image, prefix + "conv.in_proj.weight",
                              {1536, 512}, &layer.in_proj, &bound, error,
                              error_length);
            if (status == 0)
                status = bind_f32(image, prefix + "conv.conv.weight",
                                  {512, 1, 3}, &layer.conv, &bound, error,
                                  error_length);
            if (status == 0)
                status = bind_f32(image, prefix + "conv.out_proj.weight",
                                  {512, 512}, &layer.out_proj, &bound, error,
                                  error_length);
        }
    }
    if (status != 0) {
        delete plan;
        return status;
    }
    const size_t basis_bytes = aligned_bytes(kProjection * kFft * sizeof(float));
    if (basis_bytes == 0) {
        delete plan;
        return -EOVERFLOW;
    }
    plan->ifft_basis = static_cast<float *>(std::aligned_alloc(kAlign, basis_bytes));
    if (!plan->ifft_basis) {
        delete plan;
        return -ENOMEM;
    }
    std::memset(plan->ifft_basis, 0, basis_bytes);
    for (size_t index = 0; index < plan->rope_inverse.size(); ++index) {
        plan->rope_inverse[index] =
            std::pow(kTheta, -static_cast<float>(2 * index) /
                                 static_cast<float>(kHead));
    }
    const double scale = 1.0 / static_cast<double>(kFft);
    for (size_t bin = 0; bin < kBins; ++bin) {
        for (size_t sample = 0; sample < kFft; ++sample) {
            const double phase = 2.0 * std::acos(-1.0) *
                                 static_cast<double>(bin * sample) /
                                 static_cast<double>(kFft);
            const double edge = bin == 0 || bin == kFft / 2 ? 1.0 : 2.0;
            plan->ifft_basis[bin * kFft + sample] =
                static_cast<float>(edge * std::cos(phase) * scale);
            plan->ifft_basis[(kBins + bin) * kFft + sample] =
                static_cast<float>(-edge * std::sin(phase) * scale);
        }
    }
    plan->ifft_basis_bytes = basis_bytes;
    plan->bound_bytes = bound;
    plan->derived_bytes = basis_bytes + sizeof(plan->rope_inverse);
    *out = plan;
    return 0;
}

extern "C" void
lfm_detokenizer_plan_free(LfmAudioDetokenizerPlan *plan) {
    if (!plan) return;
    std::free(plan->ifft_basis);
    delete plan;
}

extern "C" uint64_t lfm_detokenizer_plan_bound_weight_bytes(
    const LfmAudioDetokenizerPlan *plan) {
    return plan ? plan->bound_bytes : 0;
}

extern "C" uint64_t lfm_detokenizer_plan_derived_bytes(
    const LfmAudioDetokenizerPlan *plan) {
    return plan ? plan->derived_bytes : 0;
}

extern "C" uint64_t lfm_detokenizer_plan_compatibility_copied_bytes(
    const LfmAudioDetokenizerPlan *plan) {
    return plan ? plan->compatibility_copied_bytes : 0;
}

extern "C" int lfm_detokenizer_state_new(
    LfmAudioDetokenizerState **out, const LfmAudioDetokenizerPlan *plan,
    char *error, size_t error_length) {
    if (!out || !plan) return -EINVAL;
    *out = nullptr;
    LfmAudioDetokenizerState *state =
        new (std::nothrow) LfmAudioDetokenizerState();
    if (!state) return -ENOMEM;
    constexpr size_t conv = kConvLayers * kConvWidth * kHidden;
    constexpr size_t kv = kAttentionLayers * kWindow * kKvHeads * kHead;
    constexpr size_t p0 = kRows * kHidden;
    constexpr size_t p1 = kRows * kHidden;
    constexpr size_t p2 = kRows * kFfn;
    constexpr size_t p3 = kRows * kFfn;
    constexpr size_t p4 = kRows * kFfn;
    constexpr size_t p5 = kRows * kHidden;
    constexpr size_t p6 = kRows * kProjection;
    constexpr size_t p7 = kRows * kProjection;
    constexpr size_t total = conv + kv + kv + p0 + p1 + p2 + p3 + p4 + p5 +
                             p6 + p7 + kRing + kRing;
    const size_t bytes = aligned_bytes(total * sizeof(float));
    state->memory = static_cast<float *>(std::aligned_alloc(kAlign, bytes));
    if (!state->memory) {
        delete state;
        return -ENOMEM;
    }
    std::memset(state->memory, 0, bytes);
    state->plan = plan;
    state->memory_bytes = bytes;
    float *cursor = state->memory;
    state->conv = carve(&cursor, conv);
    state->keys = carve(&cursor, kv);
    state->values = carve(&cursor, kv);
    state->persistent_floats = static_cast<size_t>(cursor - state->memory);
    state->x = carve(&cursor, p0);
    state->norm = carve(&cursor, p1);
    state->wide0 = carve(&cursor, p2);
    state->wide1 = carve(&cursor, p3);
    state->wide2 = carve(&cursor, p4);
    state->terminal = carve(&cursor, p5);
    state->spectral = carve(&cursor, p6);
    state->ifft = carve(&cursor, p7);
    state->ola = carve(&cursor, kRing);
    state->envelope = carve(&cursor, kRing);
    if (cursor > state->memory + total) {
        set_error(error, error_length, "detokenizer: state arena overflow");
        std::free(state->memory);
        delete state;
        return -EOVERFLOW;
    }
    *out = state;
    return 0;
}

extern "C" void
lfm_detokenizer_state_free(LfmAudioDetokenizerState *state) {
    if (!state) return;
    std::free(state->memory);
    delete state;
}

extern "C" void
lfm_detokenizer_state_reset(LfmAudioDetokenizerState *state) {
    if (!state) return;
    std::memset(state->memory, 0,
                state->persistent_floats * sizeof(float));
    std::memset(state->ola, 0, kRing * sizeof(float));
    std::memset(state->envelope, 0, kRing * sizeof(float));
    state->position = 0;
    state->frames = 0;
    state->emitted_raw = kPad;
    state->prefix_cleared = false;
    state->flushed = false;
}

extern "C" uint64_t lfm_detokenizer_state_bytes(
    const LfmAudioDetokenizerState *state) {
    return state ? sizeof(*state) + state->memory_bytes : 0;
}

extern "C" int lfm_detokenizer_state_step(
    LfmAudioDetokenizerState *state,
    const uint32_t codes[LFM_DETOKENIZER_CODEBOOKS], float *pcm,
    size_t pcm_capacity, size_t *out_samples) {
    if (!state || !codes || !pcm || !out_samples || state->flushed)
        return -EINVAL;
    *out_samples = 0;
    for (size_t index = 0; index < kCodebooks; ++index)
        if (codes[index] >= kCodeValues) return -ERANGE;
    embed_codes(state, codes);
    for (const LayerPlan &layer : state->plan->layers) {
        const int status = layer.attention ? run_attention(state, layer)
                                           : run_conv(state, layer);
        if (status != 0) return status;
    }
    rmsnorm(state->x, kRows, kHidden, state->plan->final_norm, state->norm);
    linear(state->norm, kRows, state->plan->projection, state->spectral);
    add_bias(state->spectral, kRows, kProjection, state->plan->bias);
    int status = polar_spectrum(state);
    if (status != 0) return status;
    dense(state->spectral, kRows, kProjection, state->plan->ifft_basis, kFft,
          state->ifft);
    overlap_add(state);
    state->position += kRows;
    if (!state->prefix_cleared) {
        clear_ring(state, 0, kPad);
        state->prefix_cleared = true;
    }
    return emit_range(state, state->frames * kHop, pcm, pcm_capacity,
                      out_samples);
}

extern "C" int lfm_detokenizer_state_flush(
    LfmAudioDetokenizerState *state, float *pcm, size_t pcm_capacity,
    size_t *out_samples) {
    if (!state || !pcm || !out_samples || state->flushed)
        return -EINVAL;
    *out_samples = 0;
    if (state->frames == 0) {
        state->flushed = true;
        return 0;
    }
    const uint64_t end = state->frames * kHop + kPad;
    const int status = emit_range(state, end, pcm, pcm_capacity, out_samples);
    if (status == 0) state->flushed = true;
    return status;
}
