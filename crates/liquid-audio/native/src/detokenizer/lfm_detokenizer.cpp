// Native LFM2.5 audio detokenizer.
//
// Contract pinned under native/vendor/liquid_audio/reference/detokenizer.py:
// eight codebook rows -> mean embedding -> repeat x6 -> 8-layer F32 LFM2 ->
// complex STFT projection -> same-padded ISTFT. Checkpoint weights are borrowed
// byte views into the model-owned read-only image. No tensor object, weight
// widening, alignment repair, transpose, or packed copy exists in this file.

#include "lfm_detokenizer.h"
#include "lfm_detokenizer_kernels.h"
#include "lfm_detokenizer_program.h"

#include "lfm_safetensors.h"

#include <algorithm>
#include <array>
#include <cerrno>
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

#if defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64) ||                 \
    defined(__x86_64__) || defined(_M_X64)
#define LFM_DETOK_ASM 1
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

const float *f32_pointer(const F32View &view) {
    return reinterpret_cast<const float *>(view.bytes);
}

int bind_f32(const LfmWeightImage *image, const std::string &name,
             std::initializer_list<uint64_t> shape, F32View *out,
             uint64_t *bound, char *error, size_t error_length) {
    if (!image || !out || !bound) return -EINVAL;
    LfmTensorView tensor = {
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

void rmsnorm(const float *input, size_t rows, size_t columns,
             const F32View &weight, float *output) {
    for (size_t row = 0; row < rows; ++row) {
        const LfmDetokRmsArgs args = {
            .input = input + row * columns,
            .weight = weight.bytes,
            .output = output + row * columns,
            .columns = columns,
            .epsilon = kEpsilon,
        };
        lfm_detok_rms_f32(&args);
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
    lfm_detok_swiglu_f32(gate, up, scratch, count);
    return 0;
#endif
}

int softmax(float *scores, size_t count) {
#ifndef __APPLE__
    (void)scores;
    (void)count;
    return -ENOTSUP;
#else
    const float maximum = lfm_detok_max_f32(scores, count);
    lfm_detok_subtract_f32(scores, count, maximum);
    const int length = static_cast<int>(count);
    vvexpf(scores, scores, &length);
    const float sum = lfm_detok_sum_f32(scores, count);
    lfm_detok_normalize_f32(scores, count, sum);
    return 0;
#endif
}

int rope_angles(const LfmAudioDetokenizerPlan *plan, uint64_t position,
                float cosine[kHead / 2], float sine[kHead / 2]);

void apply_rope(float *values, const float cosine[kHead / 2],
                const float sine[kHead / 2]) {
    lfm_detok_rope_f32(values, cosine, sine);
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
    float *ola = nullptr;
    float *envelope = nullptr;
    alignas(64) std::array<float, kRows * (kHead / 2)> rope_cosine{};
    alignas(64) std::array<float, kRows * (kHead / 2)> rope_sine{};
    uint64_t position = 0;
    uint64_t frames = 0;
    uint64_t emitted_raw = kPad;
    bool prefix_cleared = false;
    bool flushed = false;
    bool active = false;
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
    lfm_detok_rope_angles_f32(plan->rope_inverse.data(), angles, kHead / 2,
                              position);
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

void lane_bounds(size_t count, uint32_t lane, uint32_t lanes, size_t *first,
                 size_t *last) {
    *first = count * lane / lanes;
    *last = count * (lane + 1) / lanes;
}

void vector_lane_bounds(size_t count, uint32_t lane, uint32_t lanes,
                        size_t width, size_t *first, size_t *last) {
    size_t first_vector = 0;
    size_t last_vector = 0;
    lane_bounds(count / width, lane, lanes, &first_vector, &last_vector);
    *first = first_vector * width;
    *last = last_vector * width;
    if (lane + 1 == lanes) *last = count;
}

void embed_codes_lane(LfmAudioDetokenizerState *state,
                      const uint32_t codes[kCodebooks], uint32_t lane,
                      uint32_t lanes) {
    size_t first = 0;
    size_t last = 0;
    vector_lane_bounds(kHidden, lane, lanes, 4, &first, &last);
    LfmDetokEmbedArgs args{};
    for (size_t codebook = 0; codebook < kCodebooks; ++codebook) {
        const uint64_t row = codebook * kCodeValues + codes[codebook];
        args.rows[codebook] =
            state->plan->embedding.bytes +
            (row * kHidden + first) * sizeof(float);
    }
    args.output = state->x + first;
    args.count = last - first;
    lfm_detok_embed_f32(&args);
}

void rmsnorm_rows_lane(const float *input, size_t rows, size_t columns,
                       const F32View &weight, float *output, uint32_t lane,
                       uint32_t lanes) {
    size_t first = 0;
    size_t last = 0;
    lane_bounds(rows, lane, lanes, &first, &last);
    if (first == last) return;
    rmsnorm(input + first * columns, last - first, columns, weight,
            output + first * columns);
}

void residual_norm_rows_lane(float *destination, const float *residual,
                             size_t rows, size_t columns,
                             const F32View &weight, float *normalized,
                             uint32_t lane, uint32_t lanes) {
    size_t first = 0;
    size_t last = 0;
    lane_bounds(rows, lane, lanes, &first, &last);
    for (size_t row = first; row < last; ++row) {
        float *value = destination + row * columns;
        lfm_detok_add_f32(value, residual + row * columns, columns);
        rmsnorm(value, 1, columns, weight, normalized + row * columns);
    }
}

int swiglu_lane(float *gate, const float *up, float *scratch, size_t count,
                uint32_t lane, uint32_t lanes) {
    size_t first = 0;
    size_t last = 0;
    vector_lane_bounds(count, lane, lanes, 4, &first, &last);
    return swiglu(gate + first, up + first, scratch + first, last - first);
}

void conv_mix_lane(LfmAudioDetokenizerState *state, const LayerPlan &layer,
                   uint32_t lane, uint32_t lanes) {
    float *carry = state->conv + layer.state_index * kConvWidth * kHidden;
    size_t first = 0;
    size_t last = 0;
    vector_lane_bounds(kHidden, lane, lanes, 4, &first, &last);
    for (size_t row = 0; row < kRows; ++row) {
        const LfmDetokConvArgs args = {
            .projected = state->wide2 + row * 3 * kHidden + first,
            .carry = carry + first,
            .weight = layer.conv.bytes + first * 3 * sizeof(float),
            .output = state->norm + row * kHidden + first,
            .count = last - first,
        };
        lfm_detok_conv_f32(&args);
    }
}

int prepare_attention(LfmAudioDetokenizerState *state,
                      const LayerPlan &layer) {
    linear(state->norm, kRows, layer.q_proj, state->wide0);
    linear(state->norm, kRows, layer.k_proj, state->wide1);
    linear(state->norm, kRows, layer.v_proj, state->wide2);
    for (size_t row = 0; row < kRows; ++row) {
        const int status = rope_angles(
            state->plan, state->position + row,
            state->rope_cosine.data() + row * (kHead / 2),
            state->rope_sine.data() + row * (kHead / 2));
        if (status != 0) return status;
    }
    return 0;
}

int attention_mix_lane(LfmAudioDetokenizerState *state,
                       const LayerPlan &layer, uint32_t lane,
                       uint32_t lanes) {
    float *keys = state->keys + layer.state_index * kWindow * kKvHeads * kHead;
    float *values =
        state->values + layer.state_index * kWindow * kKvHeads * kHead;
    size_t first_head = 0;
    size_t last_head = 0;
    lane_bounds(kKvHeads, lane, lanes, &first_head, &last_head);
    for (size_t row = 0; row < kRows; ++row) {
        const uint64_t absolute = state->position + row;
        float *query = state->wide0 + row * kHidden;
        float *key = state->wide1 + row * kKvHeads * kHead;
        float *value = state->wide2 + row * kKvHeads * kHead;
        const float *cosine =
            state->rope_cosine.data() + row * (kHead / 2);
        const float *sine = state->rope_sine.data() + row * (kHead / 2);
        const size_t ring = static_cast<size_t>(absolute % kWindow);
        const size_t count = static_cast<size_t>(std::min<uint64_t>(
            kWindow, absolute + 1));
        const uint64_t first = absolute + 1 - count;
        for (size_t kv_head = first_head; kv_head < last_head; ++kv_head) {
            float *key_head = key + kv_head * kHead;
            float *value_head = value + kv_head * kHead;
            rmsnorm(key_head, 1, kHead, layer.k_norm, key_head);
            apply_rope(key_head, cosine, sine);
            lfm_detok_copy_f32(
                key_head, keys + (ring * kKvHeads + kv_head) * kHead, kHead);
            lfm_detok_copy_f32(
                value_head, values + (ring * kKvHeads + kv_head) * kHead,
                kHead);
            const size_t q_first = kv_head * (kHeads / kKvHeads);
            const size_t q_last = q_first + (kHeads / kKvHeads);
            for (size_t head = q_first; head < q_last; ++head) {
                alignas(64) float score[kWindow];
                float *q = query + head * kHead;
                rmsnorm(q, 1, kHead, layer.q_norm, q);
                apply_rope(q, cosine, sine);
                for (size_t item = 0; item < count; ++item) {
                    const size_t source =
                        static_cast<size_t>((first + item) % kWindow);
                    score[item] = lfm_detok_dot32_scaled_f32(
                        q, keys + (source * kKvHeads + kv_head) * kHead,
                        kAttentionScale);
                }
                const int status = softmax(score, count);
                if (status != 0) return status;
                float *destination =
                    state->norm + row * kHidden + head * kHead;
                const LfmDetokWeightedArgs args = {
                    .values = values,
                    .scores = score,
                    .output = destination,
                    .ring_first = first % kWindow,
                    .count = count,
                    .kv_head = kv_head,
                };
                lfm_detok_weighted_f32(&args);
            }
        }
    }
    return 0;
}

int project_polar(LfmAudioDetokenizerState *state) {
#ifndef __APPLE__
    (void)state;
    return -ENOTSUP;
#else
    const size_t count = kRows * kBins;
    float *sine = state->wide2;
    float *cosine = state->wide2 + count;
    linear(state->norm, kRows, state->plan->projection, state->spectral);
    const float *bias = f32_pointer(state->plan->bias);
    for (size_t row = 0; row < kRows; ++row) {
        float *magnitude = state->spectral + row * kProjection;
        float *phase = magnitude + kBins;
        lfm_detok_add_f32(magnitude, bias, kBins);
        lfm_detok_add_f32(phase, bias + kBins, kBins);
        const int length = static_cast<int>(kBins);
        vvexpf(magnitude, magnitude, &length);
        vvsincosf(sine + row * kBins, cosine + row * kBins, phase, &length);
        const LfmDetokPolarArgs args = {
            .magnitude = magnitude,
            .sine = sine + row * kBins,
            .cosine = cosine + row * kBins,
            .real = magnitude,
            .imaginary = phase,
            .count = kBins,
        };
        lfm_detok_polar_f32(&args);
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

void overlap_segment_lane(LfmAudioDetokenizerState *state, size_t row,
                          size_t done, size_t offset, size_t count,
                          uint32_t lane, uint32_t lanes) {
    size_t lane_first = 0;
    size_t lane_last = 0;
    vector_lane_bounds(kRing, lane, lanes, 4, &lane_first, &lane_last);
    const size_t first = std::max(offset, lane_first);
    const size_t last = std::min(offset + count, lane_last);
    if (first >= last) return;
    const size_t source = done + first - offset;
    const LfmDetokOverlapArgs args = {
        .signal = state->wide2 + row * kFft + source,
        .window = state->plan->window.bytes + source * sizeof(float),
        .output = state->ola + first,
        .envelope = state->envelope + first,
        .count = last - first,
    };
    lfm_detok_overlap_f32(&args);
}

void overlap_add_lane(LfmAudioDetokenizerState *state, uint32_t lane,
                      uint32_t lanes) {
    for (size_t row = 0; row < kRows; ++row) {
        const uint64_t start = (state->frames + row) * kHop;
        size_t done = 0;
        while (done < kFft) {
            const size_t offset = static_cast<size_t>((start + done) % kRing);
            const size_t count = std::min(kFft - done, kRing - offset);
            overlap_segment_lane(state, row, done, offset, count, lane, lanes);
            done += count;
        }
    }
}

int emit_segment_owned_lane(LfmAudioDetokenizerState *state, float *pcm,
                            size_t written, size_t offset, size_t count,
                            uint32_t lane, uint32_t lanes) {
    size_t lane_first = 0;
    size_t lane_last = 0;
    vector_lane_bounds(kRing, lane, lanes, 4, &lane_first, &lane_last);
    const size_t first = std::max(offset, lane_first);
    const size_t last = std::min(offset + count, lane_last);
    if (first >= last) return 0;
    const size_t destination = written + first - offset;
    const size_t owned = last - first;
    const LfmDetokEmitArgs args = {
        .output = state->ola + first,
        .envelope = state->envelope + first,
        .pcm = pcm + destination,
        .count = owned,
        .epsilon = 1.0e-11f,
    };
    const int status = lfm_detok_emit_f32(&args);
    if (status != 0) return status;
    std::memset(state->ola + first, 0, owned * sizeof(float));
    std::memset(state->envelope + first, 0, owned * sizeof(float));
    return 0;
}

int emit_range_owned_lane(LfmAudioDetokenizerState *state, uint64_t end,
                          float *pcm, uint32_t lane, uint32_t lanes) {
    const size_t total = static_cast<size_t>(end - state->emitted_raw);
    uint64_t cursor = state->emitted_raw;
    size_t written = 0;
    while (written < total) {
        const size_t offset = static_cast<size_t>(cursor % kRing);
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(total - written, kRing - offset));
        const int status = emit_segment_owned_lane(
            state, pcm, written, offset, count, lane, lanes);
        if (status != 0) return status;
        cursor += count;
        written += count;
    }
    return 0;
}

} // namespace

extern "C" int lfm_detokenizer_plan_new_from_image(
    LfmAudioDetokenizerPlan **out, const LfmWeightImage *image, char *error,
    size_t error_length) {
    if (!out || !image) return -EINVAL;
    *out = nullptr;
#if !defined(LFM_DETOK_ASM)
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
    status = lfm_detok_rope_inverse_f32(plan->rope_inverse.data(),
                                        plan->rope_inverse.size(), kTheta,
                                        kHead);
    if (status == 0)
        status = lfm_detok_ifft_basis_f32(plan->ifft_basis, kBins, kFft);
    if (status != 0) {
        set_error(error, error_length,
                  "detokenizer: assembly table construction failed");
        std::free(plan->ifft_basis);
        delete plan;
        return status;
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
    constexpr size_t total = conv + kv + kv + p0 + p1 + p2 + p3 + p4 + p5 +
                             p6 + kRing + kRing;
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
    state->active = false;
}

extern "C" uint64_t lfm_detokenizer_state_bytes(
    const LfmAudioDetokenizerState *state) {
    return state ? sizeof(*state) + state->memory_bytes : 0;
}

extern "C" int lfm_detokenizer_program_begin(
    LfmAudioDetokenizerProgram *program, LfmAudioDetokenizerState *state,
    const uint32_t codes[LFM_DETOKENIZER_CODEBOOKS], float *pcm,
    size_t pcm_capacity, uint32_t flush) {
    if (!program || !state || !pcm || state->flushed || state->active ||
        flush > 1) {
        return -EINVAL;
    }
    if (!flush && !codes) return -EINVAL;
    LfmAudioDetokenizerProgram next{};
    next.state = state;
    next.pcm = pcm;
    next.pcm_capacity = pcm_capacity;
    next.phase = flush ? LFM_DETOKENIZER_PHASE_EMIT
                       : LFM_DETOKENIZER_PHASE_EMBED;
    next.flush = flush;
    next.active = 1;
    if (flush) {
        if (state->frames >
            (std::numeric_limits<uint64_t>::max() - kPad) / kHop) {
            return -EOVERFLOW;
        }
        next.emit_end = state->frames * kHop + kPad;
    } else {
        if (state->frames >
            (std::numeric_limits<uint64_t>::max() / kHop) - kRows) {
            return -EOVERFLOW;
        }
        next.emit_end = (state->frames + kRows) * kHop;
        for (size_t index = 0; index < kCodebooks; ++index) {
            if (codes[index] >= kCodeValues) return -ERANGE;
            next.codes[index] = codes[index];
        }
    }
    if (next.emit_end < state->emitted_raw ||
        next.emit_end - state->emitted_raw > pcm_capacity) {
        return -ENOSPC;
    }
    state->active = true;
    *program = next;
    return 0;
}

extern "C" int lfm_detokenizer_program_run(
    LfmAudioDetokenizerProgram *program, uint32_t lane, uint32_t lanes) {
    if (!program || !program->active || !program->state || lanes == 0 ||
        lane >= lanes) {
        return -EINVAL;
    }
    LfmAudioDetokenizerState *state = program->state;
    if (!state->active || !state->plan) return -ESTALE;
    if (program->phase >= LFM_DETOKENIZER_PHASE_DONE) return -EPROTO;
    const LayerPlan *layer = program->layer < kLayers
        ? &state->plan->layers[program->layer]
        : nullptr;
    switch (program->phase) {
    case LFM_DETOKENIZER_PHASE_EMBED:
        embed_codes_lane(state, program->codes, lane, lanes);
        return 0;
    case LFM_DETOKENIZER_PHASE_OPERATOR_NORM:
        if (!layer) return -EPROTO;
        rmsnorm_rows_lane(state->x, kRows, kHidden, layer->operator_norm,
                          state->norm, lane, lanes);
        return 0;
    case LFM_DETOKENIZER_PHASE_OPERATOR_PROJECT:
        if (!layer) return -EPROTO;
        if (lane != 0) return 0;
        if (layer->attention) return prepare_attention(state, *layer);
        linear(state->norm, kRows, layer->in_proj, state->wide2);
        return 0;
    case LFM_DETOKENIZER_PHASE_OPERATOR_MIX:
        if (!layer) return -EPROTO;
        if (layer->attention)
            return attention_mix_lane(state, *layer, lane, lanes);
        conv_mix_lane(state, *layer, lane, lanes);
        return 0;
    case LFM_DETOKENIZER_PHASE_OPERATOR_OUT:
        if (!layer) return -EPROTO;
        if (lane == 0)
            linear(state->norm, kRows, layer->out_proj, state->terminal);
        return 0;
    case LFM_DETOKENIZER_PHASE_OPERATOR_RESIDUAL_NORM:
        if (!layer) return -EPROTO;
        residual_norm_rows_lane(state->x, state->terminal, kRows, kHidden,
                                layer->ffn_norm, state->norm, lane, lanes);
        return 0;
    case LFM_DETOKENIZER_PHASE_FFN_PROJECT:
        if (!layer) return -EPROTO;
        if (lane == 0) {
            linear(state->norm, kRows, layer->w1, state->wide0);
            linear(state->norm, kRows, layer->w3, state->wide1);
        }
        return 0;
    case LFM_DETOKENIZER_PHASE_FFN_ACTIVATE:
        return swiglu_lane(state->wide0, state->wide1, state->wide2,
                           kRows * kFfn, lane, lanes);
    case LFM_DETOKENIZER_PHASE_FFN_DOWN:
        if (!layer) return -EPROTO;
        if (lane == 0)
            linear(state->wide0, kRows, layer->w2, state->terminal);
        return 0;
    case LFM_DETOKENIZER_PHASE_FFN_RESIDUAL_NORM: {
        if (!layer) return -EPROTO;
        const F32View &next_norm = program->layer + 1 < kLayers
            ? state->plan->layers[program->layer + 1].operator_norm
            : state->plan->final_norm;
        residual_norm_rows_lane(state->x, state->terminal, kRows, kHidden,
                                next_norm, state->norm, lane, lanes);
        return 0;
    }
    case LFM_DETOKENIZER_PHASE_FINAL_PROJECT:
        return lane == 0 ? project_polar(state) : 0;
    case LFM_DETOKENIZER_PHASE_IFFT:
        if (lane == 0)
            dense(state->spectral, kRows, kProjection,
                  state->plan->ifft_basis, kFft, state->wide2);
        return 0;
    case LFM_DETOKENIZER_PHASE_OVERLAP_EMIT: {
        overlap_add_lane(state, lane, lanes);
        return emit_range_owned_lane(state, program->emit_end, program->pcm,
                                     lane, lanes);
    }
    case LFM_DETOKENIZER_PHASE_EMIT:
        return emit_range_owned_lane(state, program->emit_end, program->pcm,
                                     lane, lanes);
    default:
        return -EPROTO;
    }
}

extern "C" int lfm_detokenizer_program_advance(
    LfmAudioDetokenizerProgram *program) {
    if (!program || !program->active || !program->state ||
        !program->state->active) {
        return -EINVAL;
    }
    LfmAudioDetokenizerState *state = program->state;
    switch (program->phase) {
    case LFM_DETOKENIZER_PHASE_EMBED:
        program->phase = LFM_DETOKENIZER_PHASE_OPERATOR_NORM;
        return 1;
    case LFM_DETOKENIZER_PHASE_OPERATOR_NORM:
        program->phase = LFM_DETOKENIZER_PHASE_OPERATOR_PROJECT;
        return 1;
    case LFM_DETOKENIZER_PHASE_OPERATOR_PROJECT:
        program->phase = LFM_DETOKENIZER_PHASE_OPERATOR_MIX;
        return 1;
    case LFM_DETOKENIZER_PHASE_OPERATOR_MIX:
        program->phase = LFM_DETOKENIZER_PHASE_OPERATOR_OUT;
        return 1;
    case LFM_DETOKENIZER_PHASE_OPERATOR_OUT:
        program->phase = LFM_DETOKENIZER_PHASE_OPERATOR_RESIDUAL_NORM;
        return 1;
    case LFM_DETOKENIZER_PHASE_OPERATOR_RESIDUAL_NORM:
        program->phase = LFM_DETOKENIZER_PHASE_FFN_PROJECT;
        return 1;
    case LFM_DETOKENIZER_PHASE_FFN_PROJECT:
        program->phase = LFM_DETOKENIZER_PHASE_FFN_ACTIVATE;
        return 1;
    case LFM_DETOKENIZER_PHASE_FFN_ACTIVATE:
        program->phase = LFM_DETOKENIZER_PHASE_FFN_DOWN;
        return 1;
    case LFM_DETOKENIZER_PHASE_FFN_DOWN:
        program->phase = LFM_DETOKENIZER_PHASE_FFN_RESIDUAL_NORM;
        return 1;
    case LFM_DETOKENIZER_PHASE_FFN_RESIDUAL_NORM:
        ++program->layer;
        program->phase = program->layer < kLayers
            ? LFM_DETOKENIZER_PHASE_OPERATOR_PROJECT
            : LFM_DETOKENIZER_PHASE_FINAL_PROJECT;
        return 1;
    case LFM_DETOKENIZER_PHASE_FINAL_PROJECT:
        program->phase = LFM_DETOKENIZER_PHASE_IFFT;
        return 1;
    case LFM_DETOKENIZER_PHASE_IFFT:
        program->phase = LFM_DETOKENIZER_PHASE_OVERLAP_EMIT;
        return 1;
    case LFM_DETOKENIZER_PHASE_OVERLAP_EMIT:
        state->frames += kRows;
        state->position += kRows;
        if (!state->prefix_cleared) {
            clear_ring(state, 0, kPad);
            state->prefix_cleared = true;
        }
        program->produced =
            static_cast<size_t>(program->emit_end - state->emitted_raw);
        state->emitted_raw = program->emit_end;
        state->active = false;
        program->active = 0;
        program->phase = LFM_DETOKENIZER_PHASE_DONE;
        return 0;
    case LFM_DETOKENIZER_PHASE_EMIT:
        program->produced =
            static_cast<size_t>(program->emit_end - state->emitted_raw);
        state->emitted_raw = program->emit_end;
        if (program->flush) state->flushed = true;
        state->active = false;
        program->active = 0;
        program->phase = LFM_DETOKENIZER_PHASE_DONE;
        return 0;
    default:
        return -EPROTO;
    }
}

extern "C" void lfm_detokenizer_program_cancel(
    LfmAudioDetokenizerProgram *program) {
    if (!program) return;
    if (program->state && program->active) program->state->active = false;
    program->active = 0;
    program->phase = LFM_DETOKENIZER_PHASE_DONE;
}
