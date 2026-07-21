#include "lfm_model.h"
#include "lfm_model_internal.h"

#include "flashkern_depth.h"
#include "flashkern_rope.h"
#include "lfm_audio_pass.h"
#include "lfm_conformer.h"
#include "lfm_frontend.h"
#include "lfm_mimi.h"
#include "lfm_model_plan.h"
#include "lfm_payload_reader.h"
#include "lfm_safetensors.h"
#include "lfm_tokenizer.h"
#include "lfm_types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

extern "C" void lfm_f32_to_bf16(const float *input, uint16_t *output, int count);

namespace {

static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);

int add_counter(std::atomic<uint64_t> *counter, uint64_t value) {
    uint64_t current = counter->load(std::memory_order_relaxed);
    for (;;) {
        if (value > UINT64_MAX - current) return -EOVERFLOW;
        if (counter->compare_exchange_weak(current, current + value,
                                           std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
            return 0;
        }
    }
}

enum class WeightKind : uint32_t {
    Bound,
    Derived,
    Materialized,
    Compatibility,
};

/* One ledger belongs to one model construction. Publication is a one-way
 * generation transition. Setup callers are construction-exclusive; after the
 * transition, the fixed atomics only record and reject forbidden attempts. */
struct ModelAccounting {
    static constexpr uint64_t kPublished = uint64_t{1} << 63;

    std::atomic<uint64_t> publication_generation{0};
    std::atomic<uint64_t> read_state{0};
    std::atomic<uint64_t> payload_read_calls{0};
    std::atomic<uint64_t> payload_read_bytes{0};
    std::atomic<uint64_t> post_publication_read_calls{0};
    std::atomic<uint64_t> post_publication_read_bytes{0};
    std::atomic<uint64_t> directly_bound_bytes{0};
    std::atomic<uint64_t> derived_immutable_bytes{0};
    std::atomic<uint64_t> materialized_weight_bytes{0};
    std::atomic<uint64_t> compatibility_copied_bytes{0};
    std::atomic<uint64_t> post_publication_materialization_attempts{0};
    std::atomic<uint64_t> post_publication_materialization_bytes{0};
    std::atomic<uint64_t> post_readiness_allocation_attempts{0};
    std::atomic<uint64_t> post_readiness_allocation_bytes{0};
    std::atomic<uint32_t> payload_read_coverage{0};
    std::atomic<uint32_t> installed_sources{0};
    std::atomic<uint32_t> flags{0};

    int reject_payload(uint64_t bytes) {
        int status = add_counter(&post_publication_read_calls, 1);
        if (status != 0) return status;
        status = add_counter(&post_publication_read_bytes, bytes);
        return status == 0 ? -EPERM : status;
    }

    int begin_payload(uint32_t declared, uint64_t attempted_bytes) {
        uint64_t state = read_state.load(std::memory_order_acquire);
        for (;;) {
            if ((state & kPublished) != 0) {
                return reject_payload(attempted_bytes);
            }
            if (state == kPublished - 1) return -EOVERFLOW;
            if (read_state.compare_exchange_weak(
                    state, state + 1, std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                installed_sources.fetch_or(declared,
                                           std::memory_order_relaxed);
                return 0;
            }
        }
    }

    int record_payload(uint32_t source, uint64_t bytes) {
        if (source == 0 || (source & (source - 1)) != 0 ||
            (source & (LFM_MODEL_PAYLOAD_READ_CONFIG |
                       LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
                       LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX |
                       LFM_MODEL_PAYLOAD_READ_TOKENIZER)) == 0) {
            return -EINVAL;
        }
        const uint64_t state = read_state.load(std::memory_order_acquire);
        if (state == 0 || (state & kPublished) != 0) return -EPERM;
        int status = add_counter(&payload_read_calls, 1);
        if (status != 0) return status;
        status = add_counter(&payload_read_bytes, bytes);
        if (status != 0) return status;
        payload_read_coverage.fetch_or(source, std::memory_order_relaxed);
        return 0;
    }

    void end_payload() {
        const uint64_t previous =
            read_state.fetch_sub(1, std::memory_order_release);
        (void)previous;
    }

    static int begin_payload(void *context, uint32_t declared,
                             uint64_t attempted_bytes) {
        return static_cast<ModelAccounting *>(context)->begin_payload(
            declared, attempted_bytes);
    }

    static int record_payload(void *context, uint32_t source,
                              uint64_t bytes) {
        return static_cast<ModelAccounting *>(context)->record_payload(source,
                                                                        bytes);
    }

    static void end_payload(void *context) {
        static_cast<ModelAccounting *>(context)->end_payload();
    }

    LfmPayloadReadOwner reader() {
        return {
            .context = this,
            .begin = &ModelAccounting::begin_payload,
            .record = &ModelAccounting::record_payload,
            .end = &ModelAccounting::end_payload,
        };
    }

    int weight(WeightKind kind, uint64_t bytes) {
        if (publication_generation.load(std::memory_order_acquire) != 0) {
            if (kind == WeightKind::Materialized ||
                kind == WeightKind::Compatibility) {
                int status = add_counter(
                    &post_publication_materialization_attempts, 1);
                if (status != 0) return status;
                status = add_counter(&post_publication_materialization_bytes,
                                     bytes);
                if (status != 0) return status;
            }
            return -EPERM;
        }
        if (kind == WeightKind::Bound) {
            return add_counter(&directly_bound_bytes, bytes);
        }
        if (kind == WeightKind::Derived) {
            return add_counter(&derived_immutable_bytes, bytes);
        }
        int status = add_counter(&materialized_weight_bytes, bytes);
        if (status != 0 || kind == WeightKind::Materialized) return status;
        return add_counter(&compatibility_copied_bytes, bytes);
    }

    int weight_policy() const {
        if (materialized_weight_bytes.load(std::memory_order_acquire) != 0 ||
            compatibility_copied_bytes.load(std::memory_order_acquire) != 0) {
            return -EINVAL;
        }
        return 0;
    }

    int reject_allocation(uint64_t bytes) {
        int status = add_counter(&post_readiness_allocation_attempts, 1);
        if (status != 0) return status;
        status = add_counter(&post_readiness_allocation_bytes, bytes);
        return status == 0 ? -EPERM : status;
    }

    int publish(uint32_t required_sources) {
        const uint32_t installed =
            installed_sources.load(std::memory_order_acquire);
        if ((installed & required_sources) != required_sources) {
            return -ENODATA;
        }
        uint64_t expected = 0;
        if (!read_state.compare_exchange_strong(
                expected, kPublished, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return (expected & kPublished) != 0 ? -EALREADY : -EBUSY;
        }
        flags.fetch_or(LFM_MODEL_ACCOUNTING_PAYLOAD_READS_COMPLETE,
                       std::memory_order_relaxed);
        publication_generation.store(1, std::memory_order_release);
        return 0;
    }
};

class ModelError final : public std::runtime_error {
  public:
    ModelError(int status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}

    int status() const { return status_; }

  private:
    int status_;
};

[[noreturn]] void fail(int status, const std::string &message) {
    throw ModelError(status, message);
}

void set_error(char *error, size_t length, const char *message) {
    if (!error || length == 0) return;
    std::snprintf(error, length, "%s", message ? message : "unknown model error");
}

Json read_json(const LfmPayloadReadOwner *owner, const fs::path &path) {
    LfmPayloadReadScope scope(owner, LFM_MODEL_PAYLOAD_READ_CONFIG);
    if (scope.status() != 0) {
        fail(scope.status(), "model config read rejected by its owner");
    }
    const std::string native = path.string();
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
        std::fopen(native.c_str(), "rb"), &std::fclose);
    if (!file) fail(-ENOENT, "cannot open model config '" + native + "'");
    if (std::fseek(file.get(), 0, SEEK_END) != 0) {
        fail(-EIO, "cannot size model config '" + native + "'");
    }
    const long end = std::ftell(file.get());
    if (end < 0 || end > 16 * 1024 * 1024) {
        fail(-EFBIG, "model config has an invalid size: '" + native + "'");
    }
    std::rewind(file.get());
    std::vector<char> bytes((size_t)end);
    const size_t count = bytes.empty()
                             ? 0
                             : std::fread(bytes.data(), 1, bytes.size(),
                                          file.get());
    const int status = scope.record(LFM_MODEL_PAYLOAD_READ_CONFIG,
                                    (uint64_t)count);
    if (status != 0) {
        fail(status, "cannot account model config read '" + native + "'");
    }
    if (count != bytes.size()) {
        fail(-EIO, "cannot read model config '" + native + "'");
    }
    try {
        return Json::parse(bytes.begin(), bytes.end());
    } catch (const std::exception &exception) {
        fail(-EINVAL, "invalid model config '" + native + "': " + exception.what());
    }
}

size_t integer(const Json &object, const char *name, size_t fallback = 0,
               bool required = true) {
    const auto found = object.find(name);
    if (found == object.end()) {
        if (!required) return fallback;
        fail(-EINVAL, std::string("model config is missing '") + name + "'");
    }
    if (!found->is_number_unsigned() && !found->is_number_integer()) {
        fail(-EINVAL, std::string("model config '") + name + "' is not an integer");
    }
    const int64_t value = found->get<int64_t>();
    if (value < 0 || (uint64_t)value > std::numeric_limits<size_t>::max()) {
        fail(-EOVERFLOW, std::string("model config '") + name + "' is out of range");
    }
    return (size_t)value;
}

int64_t signed_integer(const Json &object, const char *name, int64_t fallback) {
    const auto found = object.find(name);
    if (found == object.end()) return fallback;
    if (!found->is_number_integer() && !found->is_number_unsigned()) {
        fail(-EINVAL, std::string("model config '") + name + "' is not an integer");
    }
    if (found->is_number_unsigned()) {
        const uint64_t value = found->get<uint64_t>();
        if (value > (uint64_t)INT64_MAX) {
            fail(-EOVERFLOW, std::string("model config '") + name + "' is out of range");
        }
        return (int64_t)value;
    }
    return found->get<int64_t>();
}

double number(const Json &object, const char *name, double fallback) {
    const auto found = object.find(name);
    if (found == object.end()) return fallback;
    if (!found->is_number()) {
        fail(-EINVAL, std::string("model config '") + name + "' is not numeric");
    }
    const double value = found->get<double>();
    if (!std::isfinite(value)) {
        fail(-EINVAL, std::string("model config '") + name + "' is not finite");
    }
    return value;
}

bool boolean(const Json &object, const char *name, bool fallback) {
    const auto found = object.find(name);
    if (found == object.end()) return fallback;
    if (!found->is_boolean()) {
        fail(-EINVAL, std::string("model config '") + name + "' is not boolean");
    }
    return found->get<bool>();
}

size_t multiply(size_t left, size_t right, const char *what) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        fail(-EOVERFLOW, std::string(what) + " overflows size_t");
    }
    return left * right;
}

/* A view is metadata only. It never owns checkpoint bytes; `value.data` and
 * `value.shape` borrow the sealed LfmWeightImage for the model lifetime. */
struct View {
    LfmTensorView value{};

    const uint8_t *bytes() const {
        return static_cast<const uint8_t *>(value.data);
    }
};

struct BindingLedger {
    struct Span {
        const void *data;
        uint64_t bytes;
    };

    explicit BindingLedger(ModelAccounting *owner) : owner(owner) {}

    ModelAccounting *owner;
    std::vector<Span> spans;

    void add(const LfmTensorView &view) {
        const auto found = std::find_if(
            spans.begin(), spans.end(), [&](const Span &span) {
                return span.data == view.data && span.bytes == view.bytes;
            });
        if (found != spans.end()) return;
        const int status = owner->weight(WeightKind::Bound, view.bytes);
        if (status != 0) fail(status, "cannot account directly bound tensor bytes");
        spans.push_back({view.data, view.bytes});
    }
};

View bind_tensor(const LfmWeightImage *weights, const std::string &name,
                 std::initializer_list<uint64_t> shape,
                 BindingLedger *bindings) {
    View view;
    view.value.size = sizeof(view.value);
    view.value.abi_version = LFM_WEIGHT_ABI_VERSION;
    const int status = lfm_weights_find(weights, name.c_str(), &view.value);
    if (status != LFM_WEIGHT_OK) {
        fail(status, "missing model tensor '" + name + "'");
    }
    if (view.value.dtype != LFM_DTYPE_BF16 || view.value.rank != shape.size()) {
        fail(-EINVAL, "model tensor '" + name + "' has the wrong dtype or rank");
    }
    size_t axis = 0;
    for (uint64_t expected : shape) {
        if (view.value.shape[axis] != expected) {
            fail(-EINVAL, "model tensor '" + name + "' has the wrong shape");
        }
        ++axis;
    }
    bindings->add(view.value);
    return view;
}

bool bind_optional_tensor(const LfmWeightImage *weights,
                          const std::string &name, View *out,
                          BindingLedger *bindings) {
    out->value.size = sizeof(out->value);
    out->value.abi_version = LFM_WEIGHT_ABI_VERSION;
    if (lfm_weights_find(weights, name.c_str(), &out->value) != LFM_WEIGHT_OK) {
        return false;
    }
    bindings->add(out->value);
    return true;
}

View bind_matrix(const LfmWeightImage *weights, const std::string &name,
                 uint64_t columns, BindingLedger *bindings) {
    View view;
    view.value.size = sizeof(view.value);
    view.value.abi_version = LFM_WEIGHT_ABI_VERSION;
    const int status = lfm_weights_find(weights, name.c_str(), &view.value);
    if (status != LFM_WEIGHT_OK) fail(status, "missing model tensor '" + name + "'");
    if (view.value.dtype != LFM_DTYPE_BF16 || view.value.rank != 2 ||
        view.value.shape[0] == 0 || view.value.shape[1] != columns) {
        fail(-EINVAL, "model tensor '" + name + "' has the wrong matrix shape");
    }
    bindings->add(view.value);
    return view;
}

LfmDepthBufferV1 depth_buffer(const View &view) {
    if (view.value.elements > std::numeric_limits<size_t>::max()) {
        fail(-EOVERFLOW, "depthformer tensor exceeds the native address space");
    }
    return {
        .address = reinterpret_cast<uintptr_t>(view.bytes()),
        .count = (size_t)view.value.elements,
    };
}

LfmDepthBufferV1 depth_buffer(const std::vector<float> &values) {
    return {
        .address = reinterpret_cast<uintptr_t>(values.data()),
        .count = values.size(),
    };
}

std::string layer_root(size_t layer) {
    return "lfm.layers." + std::to_string(layer) + ".";
}

std::string depth_layer_root(size_t layer) {
    return "depthformer.layers." + std::to_string(layer) + ".";
}

size_t ffn_size(const Json &config, size_t hidden) {
    const size_t base = integer(config, "block_ff_dim", 0, false);
    const size_t initial = base == 0 ? multiply(4, hidden, "default FFN") : base;
    if (!boolean(config, "block_auto_adjust_ff_dim", true)) return initial;
    const size_t multiple = integer(config, "block_multiple_of", 256, false);
    if (multiple == 0) fail(-EINVAL, "block_multiple_of must be nonzero");
    const double multiplier = number(config, "block_ffn_dim_multiplier", 1.0);
    const size_t swiglu = multiply(2, initial, "adjusted FFN SwiGLU") / 3;
    const double scaled = multiplier * (double)swiglu;
    const double limit = std::ldexp(1.0, std::numeric_limits<size_t>::digits);
    if (scaled < 0.0 || !std::isfinite(scaled) || scaled >= limit) {
        fail(-EOVERFLOW, "adjusted FFN size is out of range");
    }
    const size_t value = (size_t)scaled;
    const size_t bias = multiple - 1;
    if (value > std::numeric_limits<size_t>::max() - bias) {
        fail(-EOVERFLOW, "adjusted FFN rounding overflows size_t");
    }
    return multiply((value + bias) / multiple, multiple,
                    "adjusted FFN rounding");
}

size_t depth_ffn_size(size_t hidden) {
    const size_t initial = multiply(4, hidden, "depthformer FFN");
    const size_t swiglu = multiply(2, initial, "depthformer SwiGLU") / 3;
    constexpr size_t multiple = 256;
    if (swiglu > std::numeric_limits<size_t>::max() - (multiple - 1)) {
        fail(-EOVERFLOW, "depthformer FFN rounding overflows size_t");
    }
    return ((swiglu + multiple - 1) / multiple) * multiple;
}

void build_rope_f32(size_t positions, size_t head_dim, float theta,
                    std::vector<float> *cosine, std::vector<float> *sine) {
    if (head_dim == 0 || head_dim % 2 != 0) {
        fail(-EINVAL, "rotary head dimension must be positive and even");
    }
    const size_t half = head_dim / 2;
    const size_t count = multiply(positions, half, "rotary table");
    cosine->resize(count);
    sine->resize(count);
    const int status = lfm_rope_table_f32(positions, head_dim, theta,
                                          cosine->data(), sine->data());
    if (status != 0) fail(status, "architecture RoPE table kernel rejected geometry");
}

} // namespace

namespace {

} // namespace

struct LfmModel {
    ModelAccounting accounting;
    void *engine = nullptr;
    LfmWeightImage *weights = nullptr;
    LfmFrontend *frontend = nullptr;
    LfmConformer *conformer = nullptr;
    LfmTokenizer *tokenizer = nullptr;
    MimiDecodePlan *mimi = nullptr;
    uint64_t plan_id = 0;
    uint64_t depth_plan_id = 0;
    uint64_t resident_bytes = 0;
    uint64_t source_bytes = 0;
    uint64_t load_ns = 0;
    uint32_t load_workers = 0;
    uint32_t load_tasks = 0;
    uint32_t hidden = 0;
    uint32_t ffn = 0;
    uint32_t layers = 0;
    uint32_t vocab = 0;
    uint32_t max_context = 0;
    uint32_t codebooks = 0;
    uint32_t lanes = 0;
    uint32_t preprocessor_rate = 0; // frontend/Conformer input
    uint32_t codec_rate = 0;        // Mimi PCM output
    uint32_t mel_features = 0;
    uint32_t interleaved_text = 0;
    uint32_t interleaved_audio = 0;
    size_t audio_rows = 0;
    LfmTokenizerSpecialV1 special{};
    std::vector<uint32_t> codebook_offsets;
    std::vector<uint32_t> initial_turn_tokens;
    std::vector<uint32_t> next_turn_tokens;
    std::vector<uint32_t> assistant_tokens;
    float rope_theta = 1000000.0f;
    std::vector<LfmLayerDesc> descriptors;
    std::vector<float> depth_rope_cos;
    std::vector<float> depth_rope_sin;
    std::mutex lifecycle;
    bool closing = false;
    std::atomic<uint32_t> conversations{0};
};

struct ConversationLayer {
    std::vector<uint16_t> keys;
    std::vector<uint16_t> values;
    std::vector<uint16_t> convolution;
};

enum : uint32_t {
    ADMISSION_NONE = 0,
    ADMISSION_PCM = 1,
    ADMISSION_TEXT = 2,
    ADMISSION_MIXED = 3,
};

enum : uint32_t {
    ADMISSION_AUDIO_ENCODE = 0,
    ADMISSION_PREFIX = 1,
    ADMISSION_TEXT_ROWS = 2,
    ADMISSION_AUDIO_ROWS = 3,
    ADMISSION_ASSISTANT = 4,
    ADMISSION_TERMINAL = 5,
};

struct ConversationAdmission {
    LfmConversation *conversation = nullptr;
    LfmAudioRouteHandle route{};
    KcTicketIdV1 ticket{};
    LfmNativeEmission *out = nullptr;
    LfmAudioRouteNotify notify = nullptr;
    void *notify_context = nullptr;
    LfmF32SpanChain pcm{};
    uint64_t adapted_values = 0;
    size_t offset = 0;
    size_t chunk = 0;
    uint32_t sampled = 0;
    uint32_t kind = ADMISSION_NONE;
    uint32_t phase = ADMISSION_TERMINAL;
    uint64_t generation = 0;
    int status = -EINPROGRESS;
    bool complete = false;
    bool initial_prefix = false;
};

struct LfmConversation {
    LfmModel *model = nullptr;
    void *prefill_workspace = nullptr;
    LfmFrontendWorkspace *frontend_workspace = nullptr;
    LfmConformerWorkspace *conformer_workspace = nullptr;
    LfmResampler *resampler = nullptr;
    LfmResamplerWorkspace *resampler_workspace = nullptr;
    LfmResamplerStream *playback_resampler_stream = nullptr;
    MimiDecodeState *mimi = nullptr;
    LfmAudioRouteResult audio_route{};
    uint32_t route_sampled = 0;
    LfmTokenizerWorkspace *tokenizer_workspace = nullptr;
    std::vector<ConversationLayer> memory;
    std::vector<LfmLayerState> states;
    std::vector<uint16_t> rope_cos;
    std::vector<uint16_t> rope_sin;
    std::vector<float> rope_cos_f32;
    std::vector<float> rope_sin_f32;
    std::vector<uint16_t> hidden;
    std::vector<float> resampled;
    std::array<float, LFM_MIMI_PCM_CAPACITY> codec_pcm{};
    std::vector<uint16_t> mel_bf16;
    std::vector<uint16_t> adapted;
    std::array<uint32_t, LFM_TEXT_COMMAND_MAX_BYTES> token_scratch{};
    size_t token_count = 0;
    LfmSamplerConfigV1 text_sampler{};
    LfmSamplerConfigV1 audio_sampler{};
    alignas(64) LfmPrngStateV1 initial_prng{};
    alignas(64) LfmPrngStateV1 prng{};
    LfmContextWindowState window{};
    size_t rope_half = 0;
    size_t prepared_samples = 0;
    uint32_t prepared_rate = 0;
    size_t playback_frames = 0;
    uint32_t playback_rate = 0;
    bool allocation_sealed = false;
    bool hidden_ready = false;
    uint32_t modality = 1;
    uint32_t modality_left = 0;
    uint32_t pending_ids[LFM_INPUT_MAX_IDS]{};
    uint32_t pending_count = 0;
    uint32_t pending_kind = 0;
    uint32_t generated = 0;
    bool text_done = false;
    bool generation_active = false;
    bool generation_ended = false;
    ConversationAdmission admission{};
    std::atomic_flag active = ATOMIC_FLAG_INIT;

    ~LfmConversation() {
        lfm_engine_prefill_workspace_destroy(prefill_workspace);
        lfm_tokenizer_workspace_destroy(tokenizer_workspace);
        if (mimi) mimi_decode_state_free(mimi);
        if (resampler_workspace) {
            (void)lfm_resampler_workspace_destroy(resampler_workspace);
        }
        if (resampler) (void)lfm_resampler_destroy(resampler);
        if (playback_resampler_stream) {
            (void)lfm_resampler_stream_destroy(playback_resampler_stream);
        }
        if (conformer_workspace) {
            (void)lfm_conformer_workspace_destroy(conformer_workspace);
        }
        if (frontend_workspace) {
            (void)lfm_frontend_workspace_destroy(frontend_workspace);
        }
    }
};

extern "C" int lfm_context_window_admit(const LfmContextWindowState *window,
                                         size_t needed) {
    if (!window || window->capacity == 0 || needed == 0 ||
        window->position > window->capacity || window->start > window->runway ||
        window->position > UINT64_MAX - window->rope_base ||
        window->rope_base + window->position != window->cursor) {
        return -EINVAL;
    }
    if (needed > window->capacity) return -ENOSPC;
    if ((uint64_t)needed > UINT64_MAX - window->cursor) return -EOVERFLOW;
    return 0;
}

extern "C" int lfm_context_window_prefill_chunk(
    const LfmContextWindowState *window, size_t remaining, size_t max_rows,
    size_t *out_rows) {
    if (!out_rows || max_rows == 0) return -EINVAL;
    *out_rows = 0;
    const int status = lfm_context_window_admit(window, remaining);
    if (status != 0) return status;
    const size_t available = (size_t)(window->capacity - window->position);
    const size_t causal = available == 0 ? 1 : available;
    *out_rows = std::min({remaining, max_rows, causal});
    return 0;
}

extern "C" int lfm_context_window_reserve(LfmContextWindowState *window,
                                           size_t needed,
                                           LfmContextWindowMove *move) {
    if (!move) return -EINVAL;
    const int status = lfm_context_window_admit(window, needed);
    if (status != 0) return status;
    *move = {};
    if (needed <= window->capacity - window->position) return 0;
    const uint64_t dropped = window->position + needed - window->capacity;
    if (dropped > window->position || dropped > UINT64_MAX - window->rope_base) {
        return -EOVERFLOW;
    }
    move->dropped = dropped;
    move->source = window->start + dropped;
    move->retained = window->position - dropped;
    if (dropped <= window->runway - window->start) {
        window->start += dropped;
    } else {
        move->compact = 1;
        window->start = 0;
    }
    window->position = move->retained;
    window->rope_base += dropped;
    return 0;
}

extern "C" int
lfm_context_window_can_commit(const LfmContextWindowState *window) {
    if (!window || window->capacity == 0 ||
        window->position >= window->capacity ||
        window->position > UINT64_MAX - window->rope_base ||
        window->rope_base + window->position != window->cursor ||
        window->cursor == UINT64_MAX) {
        return -EINVAL;
    }
    return 0;
}

extern "C" int lfm_context_window_commit(LfmContextWindowState *window) {
    const int status = lfm_context_window_can_commit(window);
    if (status != 0) return status;
    ++window->position;
    ++window->cursor;
    return 0;
}

extern "C" int lfm_context_compact_bf16(uint16_t *plane, size_t heads,
                                         size_t head_stride, size_t head_dim,
                                         size_t source_row,
                                         size_t retained_rows) {
    if (!plane || heads == 0 || head_stride == 0 || head_dim == 0 ||
        source_row > SIZE_MAX / head_dim ||
        retained_rows > SIZE_MAX / head_dim) {
        return -EINVAL;
    }
    const size_t source = source_row * head_dim;
    const size_t count = retained_rows * head_dim;
    if (source > head_stride || count > head_stride - source ||
        count > SIZE_MAX / sizeof(uint16_t) ||
        heads > SIZE_MAX / head_stride) {
        return -EINVAL;
    }
    for (size_t head = 0; head < heads; ++head) {
        uint16_t *row = plane + head * head_stride;
        std::memmove(row, row + source, count * sizeof(uint16_t));
    }
    return 0;
}

extern "C" int lfm_mixed_turn_plan(
    size_t capacity, size_t prefix_tokens, size_t text_tokens,
    size_t audio_rows, size_t assistant_tokens, LfmMixedTurnPlan *out) {
    if (!out || capacity == 0 || prefix_tokens == 0 || text_tokens == 0 ||
        audio_rows == 0 || assistant_tokens == 0) {
        return -EINVAL;
    }
    *out = {};
    LfmMixedTurnPlan plan{};
    size_t remaining = capacity;
    if (prefix_tokens > remaining) return -ENOSPC;
    remaining -= prefix_tokens;
    plan.text_offset = prefix_tokens;
    if (text_tokens > remaining) return -ENOSPC;
    remaining -= text_tokens;
    plan.audio_offset = plan.text_offset + text_tokens;
    if (audio_rows > remaining) return -ENOSPC;
    remaining -= audio_rows;
    plan.assistant_offset = plan.audio_offset + audio_rows;
    if (assistant_tokens > remaining) return -ENOSPC;
    plan.total = plan.assistant_offset + assistant_tokens;
    *out = plan;
    return 0;
}

namespace {

bool same_ticket(const KcTicketIdV1 &left, const KcTicketIdV1 &right) {
    return left.runtime_epoch == right.runtime_epoch &&
           left.sequence == right.sequence &&
           left.generation == right.generation && left.kind == right.kind;
}

class ConversationClaim {
  public:
    explicit ConversationClaim(LfmConversation *conversation) : conversation_(conversation) {
        held_ = conversation_ &&
                !conversation_->active.test_and_set(std::memory_order_acquire);
    }

    ~ConversationClaim() {
        if (held_) conversation_->active.clear(std::memory_order_release);
    }

    explicit operator bool() const { return held_; }
    void detach() { held_ = false; }

  private:
    LfmConversation *conversation_ = nullptr;
    bool held_ = false;
};

int fill_rope(LfmConversation &conversation, uint64_t first, size_t row,
              size_t rows) {
    if (conversation.rope_half == 0) return 0;
    if (rows == 0 || row > conversation.window.capacity + conversation.window.runway ||
        rows > conversation.window.capacity + conversation.window.runway - row ||
        rows > SIZE_MAX / conversation.rope_half) {
        return -EINVAL;
    }
    const size_t count = rows * conversation.rope_half;
    if (count > conversation.rope_cos_f32.size() ||
        count > conversation.rope_sin_f32.size() || count > INT_MAX) {
        return -EOVERFLOW;
    }
    const int status = lfm_rope_range_f32(
        first, rows, conversation.rope_half * 2,
        conversation.model->rope_theta, conversation.rope_cos_f32.data(),
        conversation.rope_sin_f32.data());
    if (status != 0) return status;
    const size_t offset = row * conversation.rope_half;
    lfm_f32_to_bf16(conversation.rope_cos_f32.data(),
                    conversation.rope_cos.data() + offset, (int)count);
    lfm_f32_to_bf16(conversation.rope_sin_f32.data(),
                    conversation.rope_sin.data() + offset, (int)count);
    return 0;
}

int build_rope(LfmConversation &conversation) {
    size_t head_dim = 0;
    for (const LfmLayerDesc &layer : conversation.model->descriptors) {
        if (layer.kind != 1) continue;
        if (head_dim != 0 && head_dim != layer.hd) {
            fail(-EINVAL, "native conversation requires one attention head dimension");
        }
        head_dim = layer.hd;
    }
    if (head_dim == 0) return 0;
    const size_t half = head_dim / 2;
    if (conversation.window.capacity > SIZE_MAX - conversation.window.runway) {
        fail(-EOVERFLOW, "sliding RoPE row capacity overflows size_t");
    }
    const size_t rows = (size_t)(conversation.window.capacity +
                                 conversation.window.runway);
    const size_t count = multiply(rows, half, "sliding RoPE table");
    if (count > INT_MAX) fail(-EOVERFLOW, "RoPE table exceeds the architecture ABI");
    conversation.rope_half = half;
    conversation.rope_cos.resize(count);
    conversation.rope_sin.resize(count);
    conversation.rope_cos_f32.resize(count);
    conversation.rope_sin_f32.resize(count);
    const int status = fill_rope(conversation, 0, 0, rows);
    if (status != 0) fail(status, "architecture RoPE range kernel rejected geometry");
    return 0;
}

void refresh_state_views(LfmConversation &conversation) {
    for (size_t index = 0; index < conversation.model->descriptors.size(); ++index) {
        const LfmLayerDesc &desc = conversation.model->descriptors[index];
        if (desc.kind != 1) continue;
        ConversationLayer &memory = conversation.memory[index];
        LfmLayerState &state = conversation.states[index];
        const size_t offset = (size_t)conversation.window.start * desc.hd;
        state.k_plane = memory.keys.data() + offset;
        state.v_plane = memory.values.data() + offset;
        state.k_len = memory.keys.size() - offset;
        state.v_len = memory.values.size() - offset;
    }
}

int compact_context(LfmConversation &conversation,
                    const LfmContextWindowMove &move) {
    if (!move.compact) {
        refresh_state_views(conversation);
        return 0;
    }
    for (size_t index = 0; index < conversation.model->descriptors.size(); ++index) {
        const LfmLayerDesc &desc = conversation.model->descriptors[index];
        if (desc.kind != 1) continue;
        ConversationLayer &memory = conversation.memory[index];
        LfmLayerState &state = conversation.states[index];
        int status = lfm_context_compact_bf16(
            memory.keys.data(), desc.n_kv, state.head_stride, desc.hd,
            (size_t)move.source, (size_t)move.retained);
        if (status != 0) return status;
        status = lfm_context_compact_bf16(
            memory.values.data(), desc.n_kv, state.head_stride, desc.hd,
            (size_t)move.source, (size_t)move.retained);
        if (status != 0) return status;
    }
    /* This is activation-state sliding continuation, not a replay of the tail:
     * retained K/V rows deliberately keep the hidden history they accumulated
     * before the evicted prefix. Re-prefill-equivalence would require retaining
     * raw multimodal inputs and recomputing every retained layer, which violates
     * the no-materialization/no-hot-allocation session contract. Rotary phase,
     * however, is exact: keys are never re-rotated and table rows remain tied to
     * their absolute monotonic positions. Short-convolution carry is untouched. */
    if (conversation.rope_half != 0) {
        const size_t retained = (size_t)move.retained;
        const size_t source = (size_t)move.source;
        const size_t count = retained * conversation.rope_half;
        std::memmove(conversation.rope_cos.data(),
                     conversation.rope_cos.data() + source * conversation.rope_half,
                     count * sizeof(uint16_t));
        std::memmove(conversation.rope_sin.data(),
                     conversation.rope_sin.data() + source * conversation.rope_half,
                     count * sizeof(uint16_t));
        const size_t physical = (size_t)(conversation.window.capacity +
                                         conversation.window.runway);
        const size_t tail = physical - retained;
        if (conversation.window.cursor > UINT64_MAX - tail) return -EOVERFLOW;
        const int status = fill_rope(conversation, conversation.window.cursor,
                                     retained, tail);
        if (status != 0) return status;
    }
    refresh_state_views(conversation);
    return 0;
}

int reserve_context(LfmConversation &conversation, size_t needed) {
    LfmContextWindowMove move{};
    const int status = lfm_context_window_reserve(&conversation.window, needed,
                                                  &move);
    if (status != 0) return status;
    return compact_context(conversation, move);
}

int commit_context(LfmConversation &conversation) {
    return lfm_context_window_commit(&conversation.window);
}

int admit_context(const LfmConversation &conversation, size_t needed) {
    return lfm_context_window_admit(&conversation.window, needed);
}

int reset_memory(LfmConversation &conversation) {
    for (ConversationLayer &layer : conversation.memory) {
        std::fill(layer.keys.begin(), layer.keys.end(), 0);
        std::fill(layer.values.begin(), layer.values.end(), 0);
        std::fill(layer.convolution.begin(), layer.convolution.end(), 0);
    }
    std::fill(conversation.hidden.begin(), conversation.hidden.end(), 0);
    std::memcpy(&conversation.prng, &conversation.initial_prng,
                sizeof(conversation.prng));
    conversation.window.position = 0;
    conversation.window.start = 0;
    conversation.window.cursor = 0;
    conversation.window.rope_base = 0;
    refresh_state_views(conversation);
    if (conversation.rope_half != 0) {
        const int status = fill_rope(
            conversation, 0, 0,
            (size_t)(conversation.window.capacity + conversation.window.runway));
        if (status != 0) return status;
    }
    conversation.hidden_ready = false;
    conversation.modality = 1;
    conversation.modality_left = 0;
    conversation.pending_count = 0;
    conversation.pending_kind = 0;
    conversation.token_count = 0;
    conversation.generated = 0;
    conversation.text_done = false;
    conversation.generation_active = false;
    conversation.generation_ended = false;
    if (conversation.mimi) mimi_decode_state_reset(conversation.mimi);
    if (conversation.playback_resampler_stream) {
        lfm_resampler_stream_reset(conversation.playback_resampler_stream);
    }
    return 0;
}

uint64_t logical_capture_bytes(size_t samples) {
    return samples > UINT64_MAX / sizeof(float)
        ? UINT64_MAX
        : (uint64_t)samples * sizeof(float);
}

uint64_t logical_playback_bytes(const LfmModel &model, uint32_t sample_rate) {
    if (model.codec_rate == 0) return UINT64_MAX;
    const uint64_t numerator =
        (uint64_t)LFM_MIMI_PCM_CAPACITY * sample_rate;
    const uint64_t frames =
        (numerator + model.codec_rate - 1) / model.codec_rate;
    return frames > UINT64_MAX / sizeof(float)
        ? UINT64_MAX
        : frames * sizeof(float);
}

int prepare_playback_claimed(LfmConversation &conversation,
                             uint32_t sample_rate,
                             size_t *out_playback_frames) {
    LfmModel *model = conversation.model;
    if (!model || model->codec_rate == 0 || sample_rate == 0 ||
        !out_playback_frames) {
        return -EINVAL;
    }
    if (conversation.playback_rate == sample_rate &&
        conversation.playback_frames != 0 &&
        ((sample_rate == model->codec_rate &&
          !conversation.playback_resampler_stream) ||
         (sample_rate != model->codec_rate &&
          conversation.playback_resampler_stream))) {
        *out_playback_frames = conversation.playback_frames;
        return 0;
    }
    if (conversation.allocation_sealed) {
        return model->accounting.reject_allocation(
            logical_playback_bytes(*model, sample_rate));
    }

    LfmResamplerStream *stream = nullptr;
    /* Every route admits Mimi's complete documented 3,840-sample result. The
     * rate-changing geometry therefore reserves the maximum corresponding
     * device-rate span, even though steady-state steps usually return 1,920. */
    uint64_t frames = LFM_MIMI_PCM_CAPACITY;
    int status = 0;
    if (sample_rate != model->codec_rate) {
        status = lfm_resampler_stream_create(
            model->codec_rate, sample_rate, LFM_MIMI_PCM_CAPACITY, &stream);
        if (status == 0) {
            status = lfm_resampler_stream_out_length(
                stream, LFM_MIMI_PCM_CAPACITY, &frames);
        }
    }
    if (status == 0 &&
        (frames == 0 || frames > std::numeric_limits<size_t>::max() ||
         frames > UINT32_MAX)) {
        status = -EOVERFLOW;
    }
    if (status != 0) {
        if (stream) (void)lfm_resampler_stream_destroy(stream);
        return status;
    }

    if (conversation.playback_resampler_stream) {
        (void)lfm_resampler_stream_destroy(
            conversation.playback_resampler_stream);
    }
    conversation.playback_resampler_stream = stream;
    conversation.playback_frames = (size_t)frames;
    conversation.playback_rate = sample_rate;
    *out_playback_frames = conversation.playback_frames;
    return 0;
}

int prepare_pcm_claimed(LfmConversation &conversation, size_t max_sample_count,
                        uint32_t capture_rate, uint32_t playback_rate,
                        size_t *out_playback_frames) {
    LfmModel *model = conversation.model;
    if (!model || !model->frontend || !model->conformer ||
        !conversation.frontend_workspace || !conversation.conformer_workspace ||
        max_sample_count == 0 || capture_rate == 0 || playback_rate == 0 ||
        model->preprocessor_rate == 0 || model->codec_rate == 0 ||
        model->mel_features == 0 || model->hidden == 0) {
        return -EINVAL;
    }
    const bool capture_ready =
        conversation.resampler && conversation.resampler_workspace &&
        conversation.prepared_rate == capture_rate &&
        conversation.prepared_samples >= max_sample_count;
    const bool playback_ready =
        conversation.playback_rate == playback_rate &&
        conversation.playback_frames != 0 &&
        ((playback_rate == model->codec_rate &&
          !conversation.playback_resampler_stream) ||
         (playback_rate != model->codec_rate &&
          conversation.playback_resampler_stream));
    if (conversation.allocation_sealed) {
        if (capture_ready && playback_ready) {
            *out_playback_frames = conversation.playback_frames;
            return 0;
        }
        uint64_t requested = 0;
        if (!capture_ready) {
            requested = logical_capture_bytes(max_sample_count);
        }
        if (!playback_ready) {
            const uint64_t bytes =
                logical_playback_bytes(*model, playback_rate);
            requested = bytes > UINT64_MAX - requested
                ? UINT64_MAX
                : requested + bytes;
        }
        return model->accounting.reject_allocation(requested);
    }
    if (capture_ready) {
        const int status = prepare_playback_claimed(
            conversation, playback_rate, out_playback_frames);
        if (status == 0) conversation.allocation_sealed = true;
        return status;
    }

    LfmResampler *plan = nullptr;
    LfmResamplerWorkspace *workspace = nullptr;
    int status = lfm_resampler_create(capture_rate,
                                      model->preprocessor_rate, &plan);
    if (status != 0) return status;
    status = lfm_resampler_workspace_create(&workspace);
    if (status == 0) {
        status = lfm_resampler_workspace_reserve(plan, workspace,
                                                 max_sample_count);
    }
    uint64_t target_samples = 0;
    if (status == 0) {
        status = lfm_resampler_out_length(plan, max_sample_count,
                                          &target_samples);
    }
    if (status == 0) {
        status = lfm_frontend_workspace_reserve(
            model->frontend, conversation.frontend_workspace, target_samples,
            LFM_FRONTEND_FORWARD_VALID_ONLY |
                LFM_FRONTEND_WORKSPACE_BF16_OUTPUT);
    }
    const uint64_t frames = status == 0
        ? lfm_frontend_seq_len(model->frontend, target_samples)
        : 0;
    if (status == 0 && frames == 0) status = -EINVAL;
    if (status == 0) {
        status = lfm_conformer_workspace_reserve(
            model->conformer, conversation.conformer_workspace, frames);
    }
    const uint64_t rows = status == 0
        ? lfm_conformer_out_rows(model->conformer, frames)
        : 0;
    if (status == 0 &&
        (target_samples > std::numeric_limits<size_t>::max() ||
         frames > std::numeric_limits<size_t>::max() / model->mel_features ||
         rows == 0 || rows > std::numeric_limits<size_t>::max() / model->hidden)) {
        status = -EOVERFLOW;
    }
    if (status == 0) {
        try {
            conversation.resampled.resize(
                capture_rate == model->preprocessor_rate
                    ? 0
                    : (size_t)target_samples);
            conversation.mel_bf16.resize((size_t)frames * model->mel_features);
            conversation.adapted.resize((size_t)rows * model->hidden);
        } catch (const std::bad_alloc &) {
            status = -ENOMEM;
        }
    }
    if (status != 0) {
        if (workspace) (void)lfm_resampler_workspace_destroy(workspace);
        if (plan) (void)lfm_resampler_destroy(plan);
        return status;
    }
    if (conversation.resampler_workspace) {
        (void)lfm_resampler_workspace_destroy(conversation.resampler_workspace);
    }
    if (conversation.resampler) (void)lfm_resampler_destroy(conversation.resampler);
    conversation.resampler = plan;
    conversation.resampler_workspace = workspace;
    conversation.prepared_samples = max_sample_count;
    conversation.prepared_rate = capture_rate;
    status = prepare_playback_claimed(conversation, playback_rate,
                                      out_playback_frames);
    if (status == 0) conversation.allocation_sealed = true;
    return status;
}

int encode_text(LfmConversation &conversation, const char *text,
                size_t text_bytes) {
    if (!conversation.model->tokenizer || !conversation.tokenizer_workspace ||
        (!text && text_bytes != 0)) {
        return -ENOTSUP;
    }
    conversation.token_count = 0;
    return lfm_tokenizer_encode_bounded(
        conversation.model->tokenizer, conversation.tokenizer_workspace, text,
        text_bytes, conversation.token_scratch.data(),
        conversation.token_scratch.size(), &conversation.token_count);
}

int encode_tokens(const LfmTokenizer *tokenizer, const char *text,
                  std::vector<uint32_t> *tokens) {
    if (!tokenizer || !text || !tokens) return -EINVAL;
    size_t count = 0;
    const size_t bytes = std::strlen(text);
    int status = lfm_tokenizer_encode(tokenizer, text, bytes, nullptr, 0, &count);
    if (status != 0 && status != -ENOSPC) return status;
    try {
        tokens->resize(count);
    } catch (const std::bad_alloc &) {
        return -ENOMEM;
    }
    return lfm_tokenizer_encode(tokenizer, text, bytes, tokens->data(),
                                tokens->size(), &count);
}

int validate_pending_claimed(const LfmConversation &conversation) {
    if (conversation.pending_count == 0 ||
        conversation.pending_count > LFM_INPUT_MAX_IDS) {
        return -EINVAL;
    }
    if ((conversation.pending_kind == 0 &&
         (conversation.pending_count != 1 ||
          conversation.pending_ids[0] >= conversation.model->vocab)) ||
        (conversation.pending_kind == 1 &&
         std::any_of(conversation.pending_ids,
                     conversation.pending_ids + conversation.pending_count,
                     [&](uint32_t id) {
                         return id >= conversation.model->audio_rows;
                     })) ||
        conversation.pending_kind > 1) {
        return -ERANGE;
    }
    return 0;
}

void clear_emission(LfmNativeEmission *out, size_t position) {
    std::memset(out, 0, sizeof(*out));
    out->position = position;
}

int emit_text_claimed(LfmConversation &conversation, uint32_t token,
                      LfmNativeEmission *out) {
    if (conversation.modality_left > 0) --conversation.modality_left;
    if (token == conversation.model->special.im_end) {
        conversation.pending_count = 0;
        conversation.generation_active = false;
        conversation.generation_ended = true;
        out->kind = LFM_NATIVE_EMISSION_FINISHED;
        return 0;
    }
    size_t bytes = 0;
    int status = lfm_tokenizer_decode_piece(
        conversation.model->tokenizer, token, 1, out->text, sizeof(out->text),
        &bytes);
    if (status != 0) return status;
    out->kind = LFM_NATIVE_EMISSION_TEXT;
    out->text_bytes = (uint32_t)bytes;
    conversation.pending_ids[0] = token;
    conversation.pending_count = 1;
    conversation.pending_kind = 0;
    ++conversation.generated;
    if (token == conversation.model->special.text_end) conversation.text_done = true;
    if (conversation.modality_left == 0 || conversation.text_done) {
        conversation.modality = 3;
        conversation.modality_left = conversation.model->interleaved_audio;
    }
    return 0;
}

int emit_audio_claimed(LfmConversation &conversation, const uint32_t *computed,
                       LfmNativeEmission *out) {
    if (conversation.modality_left > 0) --conversation.modality_left;
    if (!computed || conversation.model->depth_plan_id == 0 ||
        conversation.model->codebooks == 0 ||
        conversation.model->codebooks > LFM_AUDIO_TOKEN_CAPACITY ||
        conversation.model->codebook_offsets.size() != conversation.model->codebooks) {
        return -ENOTSUP;
    }
    uint32_t codes[LFM_AUDIO_TOKEN_CAPACITY] = {};
    std::copy(computed, computed + conversation.model->codebooks, codes);
    const bool end = codes[0] == 2048;
    if (end) {
        std::fill(codes, codes + conversation.model->codebooks, 2048);
    }
    uint32_t pending[LFM_AUDIO_TOKEN_CAPACITY] = {};
    for (size_t codebook = 0; codebook < conversation.model->codebooks;
         ++codebook) {
        const uint64_t id = (uint64_t)codes[codebook] +
                            conversation.model->codebook_offsets[codebook];
        if (id > UINT32_MAX) return -EOVERFLOW;
        if (id >= conversation.model->audio_rows) return -ERANGE;
        pending[codebook] = (uint32_t)id;
    }
    /* Validate the complete tuple before mutating either the reliable emission
     * or the recurrence state. A terminal schema error must not leave a
     * half-written pending code frame for teardown to commit. */
    out->kind = LFM_NATIVE_EMISSION_AUDIO_CODES;
    out->code_count = conversation.model->codebooks;
    out->flags = end ? 1u : 0u;
    std::copy(codes, codes + conversation.model->codebooks, out->codes);
    conversation.pending_count = conversation.model->codebooks;
    conversation.pending_kind = 1;
    std::copy(pending, pending + conversation.model->codebooks,
              conversation.pending_ids);
    ++conversation.generated;
    if (conversation.modality_left == 0 && !conversation.text_done) {
        conversation.modality = 1;
        conversation.modality_left = conversation.model->interleaved_text;
    }
    if (end) conversation.modality = 1;
    return 0;
}

int submit_next_text_emission_claimed(
    LfmConversation &conversation, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *handle) {
    if (!conversation.generation_active || conversation.generation_ended ||
        conversation.modality != 1 || !notify || !handle) {
        return -EINVAL;
    }
    int status = validate_pending_claimed(conversation);
    if (status != 0) return status;
    status = reserve_context(conversation, 1);
    if (status != 0) return status;
    LfmAudioRouteResult &result = conversation.audio_route;
    std::memset(&result, 0, sizeof(result));
    conversation.route_sampled = 0;
    const LfmTokenCommitRecord commit = {
        .window = &conversation.window,
        .expected_position = conversation.window.position,
        .expected_start = conversation.window.start,
        .expected_cursor = conversation.window.cursor,
        .expected_rope_base = conversation.window.rope_base,
        .token_committed = &result.token_committed,
    };
    return lfm_engine_token_route_submit(
        conversation.model->engine, conversation.model->plan_id,
        conversation.pending_ids, conversation.pending_count,
        conversation.pending_kind, conversation.states.data(),
        conversation.states.size(), (size_t)conversation.window.position,
        conversation.rope_cos.empty() ? nullptr
            : conversation.rope_cos.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_sin.empty() ? nullptr
            : conversation.rope_sin.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_cos.size() -
            conversation.window.start * conversation.rope_half,
        conversation.hidden.data(), conversation.hidden.size(),
        &conversation.text_sampler, &conversation.prng,
        &conversation.route_sampled, conversation.model->lanes, &commit,
        &result.token_completed, notify, notify_context, handle);
}

int finish_next_text_emission_claimed(LfmConversation &conversation,
                                      int status, LfmNativeEmission *out) {
    clear_emission(out, conversation.window.cursor);
    LfmAudioRouteResult &result = conversation.audio_route;
    if (result.token_committed != 0) {
        conversation.pending_count = 0;
        conversation.pending_kind = 0;
        conversation.hidden_ready = true;
        out->position = conversation.window.cursor;
    }
    if (status != 0) return status;
    if (result.token_completed == 0 || result.token_committed == 0) {
        return -EFAULT;
    }
    return emit_text_claimed(conversation, conversation.route_sampled, out);
}

int submit_next_emission_into_claimed(
    LfmConversation &conversation, const LfmAudioRouteTarget &target,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *async_handle) {
    if (!conversation.generation_active || conversation.generation_ended ||
        conversation.modality != 3 || !conversation.mimi ||
        conversation.model->codebooks != LFM_MIMI_CODEBOOKS || !notify ||
        !async_handle) {
        return -EINVAL;
    }
    int status = validate_pending_claimed(conversation);
    if (status != 0) return status;
    status = reserve_context(conversation, 1);
    if (status != 0) return status;

    LfmAudioRouteResult &result = conversation.audio_route;
    const LfmTokenCommitRecord commit = {
        .window = &conversation.window,
        .expected_position = conversation.window.position,
        .expected_start = conversation.window.start,
        .expected_cursor = conversation.window.cursor,
        .expected_rope_base = conversation.window.rope_base,
        .token_committed = &result.token_committed,
    };
    if (conversation.playback_rate == 0 || conversation.playback_frames == 0 ||
        target.pcm_capacity < conversation.playback_frames) {
        return -ENOBUFS;
    }
    LfmAudioRouteTarget bound_target = target;
    bound_target.codec_pcm = nullptr;
    bound_target.codec_pcm_capacity = 0;
    bound_target.resampler_stream = conversation.playback_resampler_stream;
    if (bound_target.resampler_stream) {
        bound_target.codec_pcm = conversation.codec_pcm.data();
        bound_target.codec_pcm_capacity = conversation.codec_pcm.size();
    }
    return lfm_engine_audio_route_submit(
        conversation.model->engine, conversation.model->plan_id,
        conversation.model->depth_plan_id, conversation.pending_ids,
        conversation.pending_count, conversation.pending_kind,
        conversation.states.data(), conversation.states.size(),
        (size_t)conversation.window.position,
        conversation.rope_cos.empty() ? nullptr
            : conversation.rope_cos.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_sin.empty() ? nullptr
            : conversation.rope_sin.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_cos.size() -
            conversation.window.start * conversation.rope_half,
        conversation.hidden.data(), conversation.hidden.size(),
        &conversation.audio_sampler, &conversation.prng, conversation.mimi,
        &bound_target, &result, conversation.model->lanes, &commit, notify,
        notify_context, async_handle);
}

int finish_next_emission_into_claimed(LfmConversation &conversation,
                                      int status, LfmNativeEmission *out,
                                      size_t *out_samples) {
    clear_emission(out, conversation.window.cursor);
    *out_samples = 0;
    LfmAudioRouteResult &result = conversation.audio_route;
    if (result.token_committed != 0) {
        conversation.pending_count = 0;
        conversation.pending_kind = 0;
        conversation.hidden_ready = true;
        out->position = conversation.window.cursor;
    }
    int emission_status = 0;
    if (result.depth_completed != 0) {
        emission_status = emit_audio_claimed(conversation, result.codes, out);
    }
    *out_samples = result.pcm_samples;
    if (emission_status != 0) return emission_status;
    if (status != 0) return status;
    if (result.token_completed == 0 || result.token_committed == 0 ||
        result.depth_completed == 0 ||
        (result.eoaudio == 0 &&
         (result.mimi_completed == 0 || result.pcm_samples == 0))) {
        return -EFAULT;
    }
    return 0;
}

int begin_generation_claimed(LfmConversation &conversation, uint32_t sampled,
                             LfmNativeEmission *out) {
    conversation.modality = 1;
    conversation.modality_left = conversation.model->interleaved_text;
    conversation.pending_count = 0;
    conversation.pending_kind = 0;
    conversation.generated = 0;
    conversation.text_done = false;
    conversation.generation_active = true;
    conversation.generation_ended = false;
    clear_emission(out, conversation.window.cursor);
    const int status = emit_text_claimed(conversation, sampled, out);
    if (status != 0) return status;
    /* Candle created a fresh Mimi streaming decoder for every response. Keep
     * that turn boundary here, after the turn has begun successfully and only
     * once: interleaved audio runs within the turn share both codec and output
     * rate-conversion state. */
    if (conversation.mimi) mimi_decode_state_reset(conversation.mimi);
    if (conversation.playback_resampler_stream) {
        lfm_resampler_stream_reset(conversation.playback_resampler_stream);
    }
    return 0;
}

void finish_admission(ConversationAdmission &admission, int status) {
    if (admission.complete) std::abort();
    admission.status = status;
    admission.phase = ADMISSION_TERMINAL;
    admission.complete = true;
    admission.notify(admission.notify_context);
}

int admission_preflight(ConversationAdmission &admission) {
    LfmConversation &conversation = *admission.conversation;
    LfmModel *model = conversation.model;
    if (!model || model->hidden == 0) return -EINVAL;
    if (admission.adapted_values == 0 ||
        admission.adapted_values % model->hidden != 0) {
        return -EINVAL;
    }
    const size_t rows = (size_t)(admission.adapted_values / model->hidden);
    const std::vector<uint32_t> &prefix = conversation.window.cursor == 0
        ? model->initial_turn_tokens
        : model->next_turn_tokens;
    if (admission.kind == ADMISSION_PCM) {
        if (prefix.size() > model->max_context ||
            rows > model->max_context - prefix.size() ||
            model->assistant_tokens.size() >
                model->max_context - prefix.size() - rows) {
            return -ENOSPC;
        }
        return admit_context(conversation,
                             prefix.size() + rows +
                                 model->assistant_tokens.size());
    }
    if (admission.kind != ADMISSION_MIXED) return -EPROTO;
    LfmMixedTurnPlan plan{};
    const int status = lfm_mixed_turn_plan(
        model->max_context, prefix.size(), conversation.token_count, rows,
        model->assistant_tokens.size(), &plan);
    return status == 0 ? admit_context(conversation, plan.total) : status;
}

int submit_admission_node(ConversationAdmission &admission);

void continue_admission(void *context) {
    ConversationAdmission *admission =
        static_cast<ConversationAdmission *>(context);
    if (!admission || !admission->conversation || admission->complete) {
        std::abort();
    }
    LfmConversation &conversation = *admission->conversation;
    int status = lfm_engine_audio_route_collect(
        conversation.model->engine, &admission->route);
    if (status == -EINPROGRESS) status = -EPROTO;
    if (status != 0) {
        finish_admission(*admission, status);
        return;
    }
    if (admission->phase == ADMISSION_AUDIO_ENCODE) {
        status = admission_preflight(*admission);
        if (status != 0) {
            finish_admission(*admission, status);
            return;
        }
        admission->phase = ADMISSION_PREFIX;
        admission->offset = 0;
    } else {
        for (size_t row = 0; row < admission->chunk; ++row) {
            status = commit_context(conversation);
            if (status != 0) {
                finish_admission(*admission, status);
                return;
            }
        }
        conversation.hidden_ready = true;
        admission->offset += admission->chunk;
        admission->chunk = 0;
    }
    status = submit_admission_node(*admission);
    if (status != 0) finish_admission(*admission, status);
}

int submit_admission_audio(ConversationAdmission &admission) {
    LfmConversation &conversation = *admission.conversation;
    LfmModel *model = conversation.model;
    if (!model || admission.pcm.count == 0 || admission.pcm.length == 0 ||
        !model->frontend || !model->conformer ||
        !conversation.frontend_workspace ||
        !conversation.conformer_workspace || !conversation.resampler ||
        !conversation.resampler_workspace ||
        admission.pcm.length > conversation.prepared_samples) {
        return -EINVAL;
    }
    const LfmAudioEncodePassV2 pass = {
        .size = sizeof(LfmAudioEncodePassV2),
        .abi_version = LFM_AUDIO_PASS_ABI,
        .resampler = conversation.resampler,
        .resampler_workspace = conversation.resampler_workspace,
        .frontend = model->frontend,
        .frontend_workspace = conversation.frontend_workspace,
        .conformer = model->conformer,
        .conformer_workspace = conversation.conformer_workspace,
        .pcm = admission.pcm,
        .resampled = conversation.resampled.empty()
                         ? nullptr
                         : conversation.resampled.data(),
        .resampled_capacity = conversation.resampled.size(),
        .mel = conversation.mel_bf16.data(),
        .mel_capacity = conversation.mel_bf16.size(),
        .adapted = conversation.adapted.data(),
        .adapted_capacity = conversation.adapted.size(),
    };
    const int status = lfm_engine_audio_encode_submit(
        model->engine, model->plan_id, &pass, &admission.adapted_values,
        continue_admission, &admission, &admission.route);
    if (status == 0 && admission.ticket.sequence == 0) {
        admission.ticket = admission.route.ticket;
    }
    return status;
}

int submit_admission_prefill(ConversationAdmission &admission,
                             const uint32_t *ids,
                             const uint16_t *provided, size_t remaining,
                             uint32_t kind, bool sample) {
    LfmConversation &conversation = *admission.conversation;
    size_t count = 0;
    int status = lfm_context_window_prefill_chunk(
        &conversation.window, remaining, LFM_PREFILL_MAX_ROWS, &count);
    if (status != 0) return status;
    status = reserve_context(conversation, count);
    if (status != 0) return status;
    admission.chunk = count;
    status = lfm_engine_prefill_submit(
        conversation.model->engine, conversation.model->plan_id,
        conversation.prefill_workspace, ids, provided, count, kind,
        conversation.states.data(), conversation.states.size(),
        (size_t)conversation.window.position,
        conversation.rope_cos.empty()
            ? nullptr
            : conversation.rope_cos.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_sin.empty()
            ? nullptr
            : conversation.rope_sin.data() +
                  conversation.window.start * conversation.rope_half,
        conversation.rope_cos.size() -
            conversation.window.start * conversation.rope_half,
        conversation.hidden.data(), conversation.hidden.size(),
        sample && count == remaining ? &conversation.text_sampler : nullptr,
        sample && count == remaining ? &conversation.prng : nullptr,
        sample && count == remaining ? &admission.sampled : nullptr,
        conversation.model->lanes,
        continue_admission, &admission, &admission.route);
    if (status == 0 && admission.ticket.sequence == 0) {
        admission.ticket = admission.route.ticket;
    }
    return status;
}

int submit_admission_node(ConversationAdmission &admission) {
    LfmConversation &conversation = *admission.conversation;
    LfmModel *model = conversation.model;
    for (;;) {
        const std::vector<uint32_t> &prefix = admission.initial_prefix
            ? model->initial_turn_tokens
            : model->next_turn_tokens;
        if (admission.phase == ADMISSION_AUDIO_ENCODE) {
            return submit_admission_audio(admission);
        }
        if (admission.phase == ADMISSION_PREFIX) {
            if (admission.offset < prefix.size()) {
                return submit_admission_prefill(
                    admission, prefix.data() + admission.offset, nullptr,
                    prefix.size() - admission.offset, 0, false);
            }
            admission.phase = admission.kind == ADMISSION_PCM
                ? ADMISSION_AUDIO_ROWS
                : ADMISSION_TEXT_ROWS;
            admission.offset = 0;
            continue;
        }
        if (admission.phase == ADMISSION_TEXT_ROWS) {
            if (admission.offset < conversation.token_count) {
                return submit_admission_prefill(
                    admission,
                    conversation.token_scratch.data() + admission.offset,
                    nullptr, conversation.token_count - admission.offset, 0,
                    false);
            }
            admission.phase = admission.kind == ADMISSION_TEXT
                ? ADMISSION_ASSISTANT
                : ADMISSION_AUDIO_ROWS;
            admission.offset = 0;
            continue;
        }
        if (admission.phase == ADMISSION_AUDIO_ROWS) {
            const size_t rows =
                (size_t)(admission.adapted_values / model->hidden);
            if (admission.offset < rows) {
                return submit_admission_prefill(
                    admission, nullptr,
                    conversation.adapted.data() +
                        admission.offset * model->hidden,
                    rows - admission.offset, 2, false);
            }
            admission.phase = ADMISSION_ASSISTANT;
            admission.offset = 0;
            continue;
        }
        if (admission.phase == ADMISSION_ASSISTANT) {
            const size_t count = model->assistant_tokens.size();
            if (admission.offset < count) {
                const size_t remaining = count - admission.offset;
                return submit_admission_prefill(
                    admission,
                    model->assistant_tokens.data() + admission.offset,
                    nullptr, remaining, 0,
                    remaining <= LFM_PREFILL_MAX_ROWS);
            }
            const int status = begin_generation_claimed(
                conversation, admission.sampled, admission.out);
            if (status != 0) return status;
            finish_admission(admission, 0);
            return 0;
        }
        return -EPROTO;
    }
}

} // namespace

int lfm_conversation_prepare_pcm_native(LfmConversation *conversation,
                                        size_t max_sample_count,
                                        uint32_t capture_rate,
                                        uint32_t playback_rate,
                                        size_t *out_playback_frames) {
    if (!conversation || max_sample_count == 0 || capture_rate == 0 ||
        playback_rate == 0 || !out_playback_frames) {
        return -EINVAL;
    }
    *out_playback_frames = 0;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return prepare_pcm_claimed(*conversation, max_sample_count, capture_rate,
                               playback_rate, out_playback_frames);
}

static int start_admission(
    LfmConversation *conversation, uint32_t kind,
    const LfmF32SpanChain *pcm, LfmNativeEmission *out,
    LfmAudioRouteNotify notify,
    void *notify_context, LfmConversationAdmissionHandle *out_handle) {
    if (!conversation || !out || !notify || !notify_context || !out_handle ||
        kind == ADMISSION_NONE || kind > ADMISSION_MIXED) {
        return -EINVAL;
    }
    *out_handle = {};
    ConversationAdmission &admission = conversation->admission;
    uint64_t generation = admission.generation + 1;
    if (generation == 0) generation = 1;
    admission = {};
    admission.conversation = conversation;
    admission.out = out;
    admission.notify = notify;
    admission.notify_context = notify_context;
    admission.pcm = pcm ? *pcm : LfmF32SpanChain{};
    admission.kind = kind;
    admission.phase = kind == ADMISSION_TEXT
        ? ADMISSION_PREFIX
        : ADMISSION_AUDIO_ENCODE;
    admission.generation = generation;
    admission.status = -EINPROGRESS;
    admission.initial_prefix = conversation->window.cursor == 0;
    clear_emission(out, conversation->window.cursor);
    const int status = submit_admission_node(admission);
    if (status != 0) {
        admission = {};
        admission.generation = generation;
        return status;
    }
    out_handle->record = &admission;
    out_handle->generation = generation;
    out_handle->ticket = admission.ticket;
    return 0;
}

int lfm_conversation_begin_pcm_submit_native(
    LfmConversation *conversation, const float *pcm, size_t sample_count,
    uint32_t sample_rate, LfmNativeEmission *out,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmConversationAdmissionHandle *out_handle) {
    const LfmF32Span span = {
        .data = pcm,
        .length = sample_count,
    };
    return lfm_conversation_begin_pcm_spans_submit_native(
        conversation, &span, 1, sample_rate, out, notify, notify_context,
        out_handle);
}

int lfm_conversation_begin_pcm_spans_submit_native(
    LfmConversation *conversation, const LfmF32Span *spans,
    uint32_t span_count, uint32_t sample_rate, LfmNativeEmission *out,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmConversationAdmissionHandle *out_handle) {
    if (!conversation || !spans || span_count == 0 || sample_rate == 0 ||
        !out || !notify || !notify_context || !out_handle) {
        return -EINVAL;
    }
    LfmF32SpanChain pcm{};
    int status = lfm_f32_span_chain_init(spans, span_count, &pcm);
    if (status != 0) return status;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    LfmModel *model = conversation->model;
    if (!model || !model->tokenizer || !model->frontend || !model->conformer ||
        model->depth_plan_id == 0 || model->codebooks == 0 ||
        model->codebooks > LFM_AUDIO_TOKEN_CAPACITY ||
        model->interleaved_text == 0 || model->interleaved_audio == 0) {
        return -ENOTSUP;
    }
    if (conversation->generation_active && !conversation->generation_ended) {
        return -EALREADY;
    }
    if (sample_rate != conversation->prepared_rate ||
        pcm.length > conversation->prepared_samples) {
        return -EINVAL;
    }
    status = start_admission(conversation, ADMISSION_PCM, &pcm, out, notify,
                             notify_context, out_handle);
    if (status == 0) claim.detach();
    return status;
}

int lfm_conversation_begin_text_submit_native(
    LfmConversation *conversation, const char *text, size_t text_bytes,
    LfmNativeEmission *out, LfmAudioRouteNotify notify,
    void *notify_context, LfmConversationAdmissionHandle *out_handle) {
    if (!conversation || !text || text_bytes == 0 || !out || !notify ||
        !notify_context || !out_handle) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    LfmModel *model = conversation->model;
    if (!model || !model->tokenizer || model->depth_plan_id == 0 ||
        model->codebooks == 0 || model->interleaved_text == 0 ||
        model->interleaved_audio == 0) {
        return -ENOTSUP;
    }
    if (conversation->generation_active && !conversation->generation_ended) {
        return -EALREADY;
    }
    int status = encode_text(*conversation, text, text_bytes);
    if (status != 0) return status;
    const std::vector<uint32_t> &prefix = conversation->window.cursor == 0
        ? model->initial_turn_tokens
        : model->next_turn_tokens;
    const size_t text_tokens = conversation->token_count;
    if (text_tokens == 0) return -EINVAL;
    if (prefix.size() > model->max_context ||
        text_tokens > model->max_context - prefix.size() ||
        model->assistant_tokens.size() >
            model->max_context - prefix.size() - text_tokens) {
        return -ENOSPC;
    }
    status = admit_context(*conversation, prefix.size() + text_tokens +
                                              model->assistant_tokens.size());
    if (status != 0) return status;
    status = start_admission(conversation, ADMISSION_TEXT, nullptr, out,
                             notify, notify_context, out_handle);
    if (status == 0) claim.detach();
    return status;
}

int lfm_conversation_begin_mixed_submit_native(
    LfmConversation *conversation, const char *text, size_t text_bytes,
    const float *pcm, size_t sample_count, uint32_t sample_rate,
    LfmNativeEmission *out, LfmAudioRouteNotify notify,
    void *notify_context, LfmConversationAdmissionHandle *out_handle) {
    if (!conversation || !text || text_bytes == 0 || !pcm ||
        sample_count == 0 || sample_rate == 0 || !out || !notify ||
        !notify_context || !out_handle) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    LfmModel *model = conversation->model;
    if (!model || !model->tokenizer || !model->frontend || !model->conformer ||
        model->hidden == 0 || model->depth_plan_id == 0 ||
        model->codebooks == 0 || model->codebooks > LFM_AUDIO_TOKEN_CAPACITY ||
        model->interleaved_text == 0 || model->interleaved_audio == 0) {
        return -ENOTSUP;
    }
    if (conversation->generation_active && !conversation->generation_ended) {
        return -EALREADY;
    }

    if (sample_rate != conversation->prepared_rate ||
        sample_count > conversation->prepared_samples) {
        return -EINVAL;
    }
    const LfmF32Span span = {
        .data = pcm,
        .length = sample_count,
    };
    LfmF32SpanChain chain{};
    int status = lfm_f32_span_chain_init(&span, 1, &chain);
    if (status != 0) return status;
    status = encode_text(*conversation, text, text_bytes);
    if (status != 0) return status;
    const size_t text_tokens = conversation->token_count;
    if (text_tokens == 0) return -EINVAL;

    status = start_admission(conversation, ADMISSION_MIXED, &chain,
                             out, notify, notify_context, out_handle);
    if (status == 0) claim.detach();
    return status;
}

int lfm_conversation_begin_collect_native(
    LfmConversation *conversation, LfmConversationAdmissionHandle *handle) {
    if (!conversation || !handle || handle->record != &conversation->admission ||
        handle->generation == 0 ||
        handle->generation != conversation->admission.generation ||
        !same_ticket(handle->ticket, conversation->admission.ticket)) {
        return -ESTALE;
    }
    ConversationAdmission &admission = conversation->admission;
    if (!admission.complete) return -EINPROGRESS;
    const int status = admission.status;
    const uint64_t generation = admission.generation;
    admission = {};
    admission.generation = generation;
    *handle = {};
    conversation->active.clear(std::memory_order_release);
    return status;
}

int lfm_conversation_next_requires_playback_native(
    LfmConversation *conversation) {
    if (!conversation) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if (!conversation->generation_active || conversation->generation_ended) {
        return 0;
    }
    return conversation->modality == 3 ? 1 : 0;
}

int lfm_conversation_next_submit_native(
    LfmConversation *conversation, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    if (!conversation || !notify || !out_handle) return -EINVAL;
    *out_handle = {};
    if (conversation->active.test_and_set(std::memory_order_acquire)) {
        return -EBUSY;
    }
    const int status = submit_next_text_emission_claimed(
        *conversation, notify, notify_context, out_handle);
    if (status != 0) conversation->active.clear(std::memory_order_release);
    return status;
}

int lfm_conversation_next_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle,
    LfmNativeEmission *out) {
    if (!conversation || !handle || !out) return -EINVAL;
    const int status = lfm_engine_audio_route_collect(
        conversation->model->engine, handle);
    if (status == -EINPROGRESS) return status;
    if (handle->record != nullptr) return status;
    const int finish = finish_next_text_emission_claimed(
        *conversation, status, out);
    conversation->active.clear(std::memory_order_release);
    return finish;
}

int lfm_conversation_next_into_submit_native(
    LfmConversation *conversation, const LfmAudioRouteTarget *target,
    LfmAudioRouteNotify notify, void *notify_context,
    LfmAudioRouteHandle *out_handle) {
    if (!conversation || !target || !notify || !out_handle) return -EINVAL;
    *out_handle = {};
    if (conversation->active.test_and_set(std::memory_order_acquire)) {
        return -EBUSY;
    }
    const int status = submit_next_emission_into_claimed(
        *conversation, *target, notify, notify_context, out_handle);
    if (status != 0) conversation->active.clear(std::memory_order_release);
    return status;
}

int lfm_conversation_next_into_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle,
    LfmNativeEmission *out, size_t *out_samples) {
    if (!conversation || !handle || !out || !out_samples) return -EINVAL;
    const int status = lfm_engine_audio_route_collect(
        conversation->model->engine, handle);
    if (status == -EINPROGRESS) return status;
    if (handle->record != nullptr) return status;
    const int finish = finish_next_emission_into_claimed(
        *conversation, status, out, out_samples);
    conversation->active.clear(std::memory_order_release);
    return finish;
}

int lfm_conversation_interrupt_submit_native(
    LfmConversation *conversation, LfmAudioRouteNotify notify,
    void *notify_context, LfmAudioRouteHandle *out_handle) {
    if (!conversation || !notify || !notify_context || !out_handle) {
        return -EINVAL;
    }
    *out_handle = {};
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    /* An emission is published before it becomes the input to the following
     * recurrence pass. Interrupt/truncation must close that one-pass seam: the
     * already-produced token belongs in KV and ShortConv even though interrupt
     * performs no sampling and publishes nothing. This also commits the
     * recurrence-only EOAudio code tuple; im_end is terminal and is never put
     * in pending_ids by emit_text_claimed. */
    int status = 0;
    std::memset(&conversation->audio_route, 0,
                sizeof(conversation->audio_route));
    if (conversation->pending_count == 0) {
        status = lfm_engine_control_route_submit(
            conversation->model->engine, notify, notify_context, out_handle);
    } else {
        status = validate_pending_claimed(*conversation);
        if (status == 0) status = reserve_context(*conversation, 1);
        if (status == 0) {
            LfmTokenCommitRecord commit = {
                .window = &conversation->window,
                .expected_position = conversation->window.position,
                .expected_start = conversation->window.start,
                .expected_cursor = conversation->window.cursor,
                .expected_rope_base = conversation->window.rope_base,
                .token_committed =
                    &conversation->audio_route.token_committed,
            };
            status = lfm_engine_token_commit_route_submit(
                conversation->model->engine, conversation->model->plan_id,
                conversation->pending_ids, conversation->pending_count,
                conversation->pending_kind, conversation->states.data(),
                conversation->states.size(),
                (size_t)conversation->window.position,
                conversation->rope_cos.empty()
                    ? nullptr
                    : conversation->rope_cos.data() +
                          conversation->window.start *
                              conversation->rope_half,
                conversation->rope_sin.empty()
                    ? nullptr
                    : conversation->rope_sin.data() +
                          conversation->window.start *
                              conversation->rope_half,
                conversation->rope_cos.size() -
                    conversation->window.start * conversation->rope_half,
                conversation->hidden.data(), conversation->hidden.size(),
                conversation->model->lanes, &commit,
                &conversation->audio_route.token_completed, notify,
                notify_context, out_handle);
        }
    }
    if (status == 0) claim.detach();
    return status;
}

int lfm_conversation_interrupt_collect_native(
    LfmConversation *conversation, LfmAudioRouteHandle *handle) {
    if (!conversation || !handle) return -EINVAL;
    const int status = lfm_engine_audio_route_collect(
        conversation->model->engine, handle);
    if (status == -EINPROGRESS) return status;
    if (handle->record != nullptr) return status;
    int result = status;
    if (result == 0 && conversation->pending_count != 0) {
        if (conversation->audio_route.token_completed == 0 ||
            conversation->audio_route.token_committed == 0) {
            result = -EFAULT;
        } else {
            conversation->pending_count = 0;
            conversation->pending_kind = 0;
            conversation->hidden_ready = true;
        }
    }
    if (result == 0) {
        conversation->generation_active = false;
        conversation->generation_ended = true;
    }
    conversation->active.clear(std::memory_order_release);
    return result;
}

int lfm_conversation_belongs_to(const LfmConversation *conversation,
                                const LfmModel *model) {
    return conversation && model && conversation->model == model ? 1 : 0;
}

extern "C" int lfm_model_open(void *engine, const char *path, LfmModel **out,
                              char *error, size_t error_length) {
    if (!engine || !path || !out) return -EINVAL;
    *out = nullptr;
    const auto load_started = std::chrono::steady_clock::now();
    LfmWeightImage *weights = nullptr;
    LfmFrontend *frontend = nullptr;
    LfmConformer *conformer = nullptr;
    LfmTokenizer *tokenizer = nullptr;
    MimiDecodePlan *mimi = nullptr;
    uint64_t plan_id = 0;
    uint64_t depth_plan_id = 0;
    std::vector<float> depth_rope_cos;
    std::vector<float> depth_rope_sin;
    std::unique_ptr<LfmModel> model(new (std::nothrow) LfmModel());
    if (!model) {
        set_error(error, error_length, "cannot allocate native model handle");
        return -ENOMEM;
    }
    const LfmPayloadReadOwner payload_owner = model->accounting.reader();
    BindingLedger bindings(&model->accounting);
    try {
        const auto tensor = [&bindings](const LfmWeightImage *image,
                                        const std::string &name,
                                        std::initializer_list<uint64_t> shape) {
            return bind_tensor(image, name, shape, &bindings);
        };
        const auto optional_tensor = [&bindings](const LfmWeightImage *image,
                                                 const std::string &name,
                                                 View *out_view) {
            return bind_optional_tensor(image, name, out_view, &bindings);
        };
        const auto matrix = [&bindings](const LfmWeightImage *image,
                                        const std::string &name,
                                        uint64_t columns) {
            return bind_matrix(image, name, columns, &bindings);
        };
        const fs::path root(path);
        const Json document = read_json(&payload_owner,
                                        root / "config.json");
        if (!document.is_object() || !document.contains("lfm") ||
            !document.at("lfm").is_object()) {
            fail(-EINVAL, "model config has no object 'lfm'");
        }
        const Json &config = document.at("lfm");
        const size_t hidden = integer(config, "hidden_size");
        const size_t layers = integer(config, "num_hidden_layers");
        const size_t heads = integer(config, "num_attention_heads");
        const size_t kv_heads = integer(config, "num_key_value_heads", 8, false);
        const size_t vocab = integer(config, "vocab_size");
        const size_t max_context =
            integer(config, "max_position_embeddings", 128000, false);
        const size_t conv_kernel = integer(config, "conv_L_cache",
                                           integer(config, "conv_l_cache", 3, false), false);
        const size_t ffn = ffn_size(config, hidden);
        if (hidden == 0 || layers == 0 || heads == 0 || kv_heads == 0 || vocab == 0 ||
            max_context == 0 || hidden % heads != 0 || heads % kv_heads != 0 ||
            hidden > UINT32_MAX || ffn > UINT32_MAX || layers > UINT32_MAX ||
            vocab > UINT32_MAX || max_context > UINT32_MAX) {
            fail(-EINVAL, "model dimensions are invalid or exceed the native ABI");
        }
        const size_t head_dim = hidden / heads;
        const float eps = (float)number(config, "norm_eps", 1e-5);
        const float rope_theta = (float)number(config, "rope_theta", 1000000.0);
        const size_t codebooks = integer(document, "codebooks", 0, false);

        const fs::path codec_path = root / "tokenizer-e351c8d8-checkpoint125.safetensors";
        const bool voice_model = document.contains("preprocessor") ||
                                 document.contains("encoder");
        if (voice_model && codebooks != LFM_MIMI_CODEBOOKS) {
            fail(-EINVAL, "native Mimi requires exactly eight audio codebooks");
        }
        std::error_code codec_error;
        bool codec_exists = false;
        {
            /* This lease closes the publication race around metadata I/O, but
             * it deliberately installs no payload source. Only the owned
             * loader below may prove the shard/index hooks are present. */
            LfmPayloadReadScope probe(&payload_owner, 0);
            if (probe.status() != 0) {
                fail(probe.status(),
                     "codec source inspection rejected by its model owner");
            }
            codec_exists = fs::is_regular_file(codec_path, codec_error);
        }
        if (voice_model && (!codec_exists || codec_error)) {
            fail(-ENOENT,
                 "native LFM2-Audio requires its Mimi codec checkpoint");
        }
        char weight_error[512] = {};
        const std::string codec_native = codec_path.string();
        int status = codec_exists
                         ? lfm_weights_open_bundle_owned(
                               path, codec_native.c_str(), &payload_owner,
                               &weights, weight_error, sizeof(weight_error))
                         : lfm_weights_open_owned(path, &payload_owner, &weights,
                                                  weight_error,
                                                  sizeof(weight_error));
        if (status != LFM_WEIGHT_OK) {
            fail(status, weight_error[0] ? weight_error : "cannot open model weights");
        }

        std::vector<LfmLayerDesc> descriptors(layers);
        const Json *types = nullptr;
        const auto type_entry = config.find("layer_types");
        if (type_entry != config.end()) {
            if (!type_entry->is_array()) fail(-EINVAL, "lfm.layer_types is not an array");
            if (type_entry->size() != layers) {
                fail(-EINVAL,
                     "lfm.layer_types length does not match num_hidden_layers");
            }
            types = &*type_entry;
        }

        for (size_t layer = 0; layer < layers; ++layer) {
            const std::string root_name = layer_root(layer);
            LfmLayerDesc &desc = descriptors[layer];
            desc.op_eps = eps;
            desc.ffn_eps = eps;
            desc.op_norm_w = tensor(weights, root_name + "operator_norm.weight", {hidden}).bytes();
            desc.ffn_norm_w = tensor(weights, root_name + "ffn_norm.weight", {hidden}).bytes();
            desc.w1 = tensor(weights, root_name + "feed_forward.w1.weight", {ffn, hidden}).bytes();
            desc.w3 = tensor(weights, root_name + "feed_forward.w3.weight", {ffn, hidden}).bytes();
            desc.w2 = tensor(weights, root_name + "feed_forward.w2.weight", {hidden, ffn}).bytes();

            std::string kind = "full_attention";
            if (types) {
                if (!types->at(layer).is_string()) {
                    fail(-EINVAL, "lfm.layer_types contains a non-string entry");
                }
                kind = types->at(layer).get<std::string>();
            }
            if (kind == "conv") {
                desc.kind = 0;
                desc.k = (uint32_t)conv_kernel;
                desc.in_w = tensor(weights, root_name + "conv.in_proj.weight",
                                   {3 * hidden, hidden}).bytes();
                desc.conv_w = tensor(weights, root_name + "conv.conv.weight",
                                     {hidden, 1, conv_kernel}).bytes();
                desc.out_w = tensor(weights, root_name + "conv.out_proj.weight",
                                    {hidden, hidden}).bytes();
                continue;
            }
            if (kind != "full_attention") {
                fail(-EINVAL, "unsupported lfm layer type '" + kind + "'");
            }
            desc.kind = 1;
            desc.n_head = (uint32_t)heads;
            desc.n_kv = (uint32_t)kv_heads;
            desc.hd = (uint32_t)head_dim;
            desc.qk_eps = eps;
            const std::string attention = root_name + "self_attn.";
            desc.q_w = tensor(weights, attention + "q_proj.weight",
                              {heads * head_dim, hidden}).bytes();
            desc.k_w = tensor(weights, attention + "k_proj.weight",
                              {kv_heads * head_dim, hidden}).bytes();
            desc.v_w = tensor(weights, attention + "v_proj.weight",
                              {kv_heads * head_dim, hidden}).bytes();
            desc.o_w = tensor(weights, attention + "out_proj.weight",
                              {hidden, heads * head_dim}).bytes();
            desc.qn_w = tensor(weights, attention + "q_layernorm.weight", {head_dim}).bytes();
            desc.kn_w = tensor(weights, attention + "k_layernorm.weight", {head_dim}).bytes();
        }

        status = lfm_ctx_build(engine, descriptors.data(), descriptors.size(), hidden,
                               ffn, max_context, &plan_id);
        if (status != 0) fail(status, "native executor rejected the backbone plan");

        const View text = tensor(weights, "lfm.embed_tokens.weight", {vocab, hidden});
        const View norm = tensor(weights, "lfm.embedding_norm.weight", {hidden});
        View audio;
        const uint8_t *audio_data = nullptr;
        size_t audio_elements = 0;
        size_t audio_rows = 0;
        if (optional_tensor(weights, "audio_embedding.embedding.weight", &audio)) {
            if (audio.value.dtype != LFM_DTYPE_BF16 || audio.value.rank != 2 ||
                audio.value.shape[1] != hidden) {
                fail(-EINVAL, "audio_embedding.embedding.weight has the wrong shape");
            }
            audio_data = audio.bytes();
            audio_elements = (size_t)audio.value.elements;
            audio_rows = (size_t)audio.value.shape[0];
        }
        constexpr size_t audio_vocabulary = 2049;
        if (codebooks != 0) {
            const size_t expected_audio_rows =
                multiply(codebooks, audio_vocabulary, "audio embedding rows");
            if (!audio_data || audio_rows != expected_audio_rows) {
                fail(-EINVAL,
                     "audio embedding vocabulary does not match configured codebooks");
            }
        }
        status = lfm_ctx_set_heads(engine, plan_id, text.bytes(), (size_t)text.value.elements,
                                   vocab, audio_data, audio_elements, audio_rows,
                                   norm.bytes(), (size_t)norm.value.elements, eps);
        if (status != 0) fail(status, "native executor rejected the model heads");

        const auto depth_entry = document.find("depthformer");
        if (depth_entry != document.end()) {
            if (!depth_entry->is_object()) {
                fail(-EINVAL, "model config 'depthformer' is not an object");
            }
            if (codebooks == 0 || codebooks > LFM_AUDIO_TOKEN_CAPACITY) {
                fail(-EINVAL, "depthformer requires a supported nonzero codebook count");
            }
            const Json &depth = *depth_entry;
            const size_t depth_layers = integer(depth, "layers");
            const size_t depth_dim = integer(depth, "dim");
            const size_t depth_heads = integer(depth, "heads", 32, false);
            const size_t depth_kv_heads = integer(depth, "kv_heads", 8, false);
            const float depth_eps = (float)number(depth, "norm_eps", 1e-5);
            const float depth_theta = (float)number(depth, "rope_theta", 1000000.0);
            if (depth_layers == 0 || depth_dim == 0 || depth_heads == 0 ||
                depth_kv_heads == 0 || depth_dim % depth_heads != 0 ||
                depth_heads % depth_kv_heads != 0 || depth_dim > UINT32_MAX ||
                depth_layers > UINT32_MAX) {
                fail(-EINVAL, "depthformer dimensions are invalid or exceed the native ABI");
            }
            const size_t depth_head_dim = depth_dim / depth_heads;
            const size_t depth_ffn = depth_ffn_size(depth_dim);
            if (depth_ffn > UINT32_MAX) {
                fail(-EOVERFLOW, "depthformer FFN exceeds the native ABI");
            }
            const size_t depth_kv_width = multiply(
                multiply(2, depth_kv_heads, "depthformer KV heads"),
                depth_head_dim, "depthformer KV width");
            if (depth_kv_width > std::numeric_limits<size_t>::max() - depth_dim) {
                fail(-EOVERFLOW, "depthformer QKV rows overflow size_t");
            }
            const size_t qkv_rows = depth_dim + depth_kv_width;
            const size_t depth_projection = multiply(codebooks, depth_dim,
                                                     "depthformer projection rows");
            std::vector<LfmDepthLayerV1> depth_descriptors(depth_layers);
            for (size_t layer = 0; layer < depth_layers; ++layer) {
                const std::string root_name = depth_layer_root(layer);
                const std::string attention = root_name + "operator.";
                LfmDepthLayerV1 &desc = depth_descriptors[layer];
                desc.qkv_w = depth_buffer(tensor(weights, attention + "qkv_proj.weight",
                                                 {qkv_rows, depth_dim}));
                desc.out_w = depth_buffer(tensor(weights, attention + "out_proj.weight",
                                                 {depth_dim, depth_dim}));
                desc.q_ln = depth_buffer(tensor(
                    weights, attention + "bounded_attention.q_layernorm.weight",
                    {depth_head_dim}));
                desc.k_ln = depth_buffer(tensor(
                    weights, attention + "bounded_attention.k_layernorm.weight",
                    {depth_head_dim}));
                desc.op_norm = depth_buffer(tensor(weights, root_name + "operator_norm.weight",
                                                   {depth_dim}));
                desc.ffn_norm = depth_buffer(tensor(weights, root_name + "ffn_norm.weight",
                                                    {depth_dim}));
                desc.w1 = depth_buffer(tensor(weights, root_name + "feed_forward.w1.weight",
                                              {depth_ffn, depth_dim}));
                desc.w3 = depth_buffer(tensor(weights, root_name + "feed_forward.w3.weight",
                                              {depth_ffn, depth_dim}));
                desc.w2 = depth_buffer(tensor(weights, root_name + "feed_forward.w2.weight",
                                              {depth_dim, depth_ffn}));
            }

            const View depth_linear_w = tensor(weights, "depth_linear.weight",
                                                {depth_projection, hidden});
            const View depth_linear_b = tensor(weights, "depth_linear.bias",
                                                {depth_projection});
            std::vector<LfmDepthHeadV1> depth_heads_table(codebooks);
            uint64_t depth_vocabulary = 0;
            for (size_t codebook = 0; codebook < codebooks; ++codebook) {
                const std::string root_name = "depth_embeddings." +
                                              std::to_string(codebook) + ".";
                const View embedding = matrix(weights, root_name + "embedding.weight",
                                              depth_dim);
                const View logits = matrix(weights, root_name + "to_logits.weight",
                                           depth_dim);
                if (embedding.value.shape[0] != logits.value.shape[0]) {
                    fail(-EINVAL, "depthformer embedding and logits vocabularies differ");
                }
                if (depth_vocabulary == 0) {
                    depth_vocabulary = embedding.value.shape[0];
                } else if (embedding.value.shape[0] != depth_vocabulary) {
                    fail(-EINVAL, "depthformer codebooks use mixed vocabularies");
                }
                depth_heads_table[codebook] = {
                    .embedding = depth_buffer(embedding),
                    .norm = depth_buffer(tensor(weights,
                                                root_name + "embedding_norm.weight",
                                                {depth_dim})),
                    .logits = depth_buffer(logits),
                    .vocab = (size_t)embedding.value.shape[0],
                };
            }
            if (depth_vocabulary != audio_vocabulary) {
                fail(-EINVAL, "depthformer vocabulary does not include EOAudio exactly");
            }

            build_rope_f32(codebooks, depth_head_dim, depth_theta,
                           &depth_rope_cos, &depth_rope_sin);
            const LfmDepthPlanV1 depth_plan = {
                .size = sizeof(depth_plan),
                .abi_version = LFM_DEPTH_ABI_VERSION,
                .dim = (uint32_t)depth_dim,
                .heads = (uint32_t)depth_heads,
                .kv_heads = (uint32_t)depth_kv_heads,
                .head_dim = (uint32_t)depth_head_dim,
                .ffn_dim = (uint32_t)depth_ffn,
                .codebooks = (uint32_t)codebooks,
                .backbone_dim = (uint32_t)hidden,
                .eps = depth_eps,
                .depth_linear_w = depth_buffer(depth_linear_w),
                .depth_linear_b = depth_buffer(depth_linear_b),
                .rope_cos = depth_buffer(depth_rope_cos),
                .rope_sin = depth_buffer(depth_rope_sin),
                .layers = depth_descriptors.data(),
                .layer_count = depth_descriptors.size(),
                .codebook_heads = depth_heads_table.data(),
                .codebook_head_count = depth_heads_table.size(),
            };
            status = lfm_engine_depth_build(engine, &depth_plan, &depth_plan_id);
            if (status != 0) fail(status, "native executor rejected the depthformer plan");
        }

        uint32_t sample_rate = 0;
        uint32_t mel_features = 0;
        uint64_t frontend_derived = 0;
        const auto preprocessor_entry = document.find("preprocessor");
        const auto encoder_entry = document.find("encoder");
        if ((preprocessor_entry == document.end()) != (encoder_entry == document.end())) {
            fail(-EINVAL, "model config must provide preprocessor and encoder together");
        }
        if (preprocessor_entry != document.end()) {
            if (!preprocessor_entry->is_object() || !encoder_entry->is_object()) {
                fail(-EINVAL, "model preprocessor/encoder config is not an object");
            }
            const Json &preprocessor = *preprocessor_entry;
            const Json &encoder = *encoder_entry;
            const size_t rate = integer(preprocessor, "sample_rate");
            const size_t features = integer(preprocessor, "features");
            const size_t n_fft = integer(preprocessor, "n_fft");
            const size_t pad_to = integer(preprocessor, "pad_to", 0, false);
            const double window_seconds = number(preprocessor, "window_size", 0.025);
            const double stride_seconds = number(preprocessor, "window_stride", 0.010);
            const double window_samples = std::round(window_seconds * (double)rate);
            const double stride_samples = std::round(stride_seconds * (double)rate);
            if (rate == 0 || features == 0 || n_fft == 0 || window_samples < 1.0 ||
                stride_samples < 1.0 || rate > UINT32_MAX || features > UINT32_MAX ||
                n_fft > UINT32_MAX || pad_to > UINT32_MAX ||
                window_samples > UINT32_MAX || stride_samples > UINT32_MAX) {
                fail(-EINVAL, "model preprocessor geometry is invalid");
            }
            const LfmFrontendConfig frontend_config = {
                .size = sizeof(LfmFrontendConfig),
                .abi_version = LFM_FRONTEND_ABI,
                .sample_rate = (uint32_t)rate,
                .n_window_size = (uint32_t)window_samples,
                .n_window_stride = (uint32_t)stride_samples,
                .n_fft = (uint32_t)n_fft,
                .nfilt = (uint32_t)features,
                .exact_pad = boolean(preprocessor, "exact_pad", false) ? 1u : 0u,
                .pad_to = (uint32_t)pad_to,
                .reserved0 = 0,
                .preemph = 0.97,
                .log_zero_guard_value = std::ldexp(1.0, -24),
                .mag_power = 2.0,
                .reserved = {0, 0, 0, 0},
            };
            status = lfm_frontend_create(&frontend_config, &frontend);
            if (status != 0) fail(status, "native frontend rejected model geometry");
            frontend_derived = lfm_frontend_derived_bytes(frontend);

            const size_t feat_in = integer(encoder, "feat_in");
            const size_t d_model = integer(encoder, "d_model");
            const size_t encoder_layers = integer(encoder, "n_layers");
            const size_t encoder_heads = integer(encoder, "n_heads");
            const size_t expansion = integer(encoder, "ff_expansion_factor");
            const size_t encoder_ff = multiply(d_model, expansion, "encoder FFN");
            const size_t conv_kernel_size = integer(encoder, "conv_kernel_size");
            const size_t subsampling = integer(encoder, "subsampling_factor");
            const int64_t configured_channels =
                signed_integer(encoder, "subsampling_conv_channels", -1);
            const size_t conv_channels = configured_channels > 0
                                             ? (size_t)configured_channels
                                             : d_model;
            const int64_t configured_out = signed_integer(encoder, "feat_out", -1);
            const size_t feat_out = configured_out > 0 ? (size_t)configured_out : d_model;
            /* These refuse cases this encoder does not implement, so an
             * unsupported config fails loudly instead of silently computing a
             * different transform. A guard may only state what the code does
             * NOT do; it must never assert a requirement the code never reads
             * (the old xscaling guard demanded a sqrt(d_model) scale that is
             * applied nowhere, and refused the exact case it implements). */
            const auto attention_entry = encoder.find("self_attention_model");
            if (attention_entry != encoder.end() &&
                (!attention_entry->is_string() ||
                 attention_entry->get<std::string>() != "rel_pos")) {
                fail(-EOPNOTSUPP, "native Conformer implements rel_pos attention only");
            }
            /* No sqrt(d_model) input scale exists in this encoder: it implements
             * xscaling=false. NeMo defaults the key to true, so an absent key is
             * refused rather than silently mis-scaled. */
            if (boolean(encoder, "xscaling", true)) {
                fail(-EOPNOTSUPP, "native Conformer does not implement encoder xscaling");
            }
            if (feat_in != features || feat_out != d_model || d_model == 0 ||
                encoder_layers == 0 || encoder_heads == 0 || encoder_ff == 0 ||
                conv_kernel_size == 0 || subsampling == 0 || conv_channels == 0 ||
                feat_in > UINT32_MAX || d_model > UINT32_MAX ||
                encoder_layers > UINT32_MAX || encoder_heads > UINT32_MAX ||
                encoder_ff > UINT32_MAX || conv_kernel_size > UINT32_MAX ||
                subsampling > UINT32_MAX || conv_channels > UINT32_MAX) {
                fail(-EINVAL, "native Conformer geometry is invalid or unsupported");
            }
            const LfmConformerGeometry conformer_geometry = {
                .size = sizeof(LfmConformerGeometry),
                .abi_version = LFM_CONFORMER_ABI,
                .feat_in = (uint32_t)feat_in,
                .d_model = (uint32_t)d_model,
                .n_layers = (uint32_t)encoder_layers,
                .n_heads = (uint32_t)encoder_heads,
                .d_ff = (uint32_t)encoder_ff,
                .conv_kernel = (uint32_t)conv_kernel_size,
                .subsampling = (uint32_t)subsampling,
                .conv_channels = (uint32_t)conv_channels,
                .adapter_hidden = (uint32_t)hidden,
                .adapter_out = (uint32_t)hidden,
                .reserved = {0, 0, 0, 0},
            };
            char conformer_error[512] = {};
            status = lfm_conformer_create(engine, weights, &conformer_geometry,
                                          &conformer, conformer_error,
                                          sizeof(conformer_error));
            if (status != 0) {
                fail(status, conformer_error[0] ? conformer_error
                                                : "cannot bind native Conformer");
            }
            sample_rate = (uint32_t)rate;
            mel_features = (uint32_t)features;

            const fs::path tokenizer_path = root / "tokenizer.json";
            const std::string tokenizer_native = tokenizer_path.string();
            char tokenizer_error[512] = {};
            status = lfm_tokenizer_open_owned(
                tokenizer_native.c_str(), &payload_owner, &tokenizer,
                tokenizer_error, sizeof(tokenizer_error));
            if (status != 0) {
                fail(status, tokenizer_error[0] ? tokenizer_error
                                                : "cannot load native tokenizer");
            }
        }

        if (voice_model) {
            char mimi_error[512] = {};
            status = mimi_decode_plan_new_from_image(&mimi, weights, mimi_error,
                                                     sizeof(mimi_error));
            if (status != 0) {
                fail(status, mimi_error[0] ? mimi_error
                                           : "cannot bind native Mimi decoder");
            }
        }

        model->engine = engine;
        model->weights = weights;
        model->frontend = frontend;
        model->conformer = conformer;
        model->tokenizer = tokenizer;
        model->mimi = mimi;
        model->plan_id = plan_id;
        model->depth_plan_id = depth_plan_id;
        LfmWeightLoadStatsV1 load_stats = {
            .size = sizeof(LfmWeightLoadStatsV1),
            .abi_version = LFM_WEIGHT_ABI_VERSION,
        };
        status = lfm_weights_load_stats(weights, &load_stats);
        if (status != LFM_WEIGHT_OK) {
            fail(status, "cannot read resident-image load accounting");
        }
        model->resident_bytes = load_stats.resident_bytes;
        model->source_bytes = load_stats.source_bytes;
        const uint64_t bound_parts[] = {
            lfm_conformer_bound_weight_bytes(conformer),
            mimi_decode_plan_bound_weight_bytes(mimi),
        };
        for (uint64_t bytes : bound_parts) {
            status = model->accounting.weight(WeightKind::Bound, bytes);
            if (status != 0) {
                fail(status, "cannot account directly bound component bytes");
            }
        }
        const uint64_t derived_parts[] = {
            frontend_derived,
            lfm_conformer_derived_bytes(conformer),
            (uint64_t)(depth_rope_cos.size() + depth_rope_sin.size()) *
                sizeof(float),
            mimi_decode_plan_derived_bytes(mimi),
        };
        for (uint64_t bytes : derived_parts) {
            status = model->accounting.weight(WeightKind::Derived, bytes);
            if (status != 0) {
                fail(status, "cannot account formula-derived immutable bytes");
            }
        }
        const uint64_t compatibility_parts[] = {
            lfm_conformer_materialized_weight_bytes(conformer),
            mimi_decode_plan_compatibility_copied_bytes(mimi),
        };
        for (uint64_t bytes : compatibility_parts) {
            status = model->accounting.weight(WeightKind::Compatibility,
                                              bytes);
            if (status != 0) {
                fail(status, "cannot account compatibility-copied weights");
            }
        }
        status = model->accounting.weight_policy();
        if (status != 0) {
            fail(status, "native model materialized checkpoint weights");
        }
        model->load_workers = load_stats.worker_count;
        model->load_tasks = load_stats.task_count;
        model->hidden = (uint32_t)hidden;
        model->ffn = (uint32_t)ffn;
        model->layers = (uint32_t)layers;
        model->vocab = (uint32_t)vocab;
        model->max_context = (uint32_t)max_context;
        model->codebooks = codebooks > UINT32_MAX ? 0 : (uint32_t)codebooks;
        model->lanes = lfm_engine_lanes(engine);
        model->preprocessor_rate = sample_rate;
        model->codec_rate = mimi ? LFM_MIMI_SAMPLE_RATE : 0;
        model->mel_features = mel_features;
        model->audio_rows = audio_rows;
        if (tokenizer) {
            model->special = {
                .size = sizeof(LfmTokenizerSpecialV1),
                .abi_version = LFM_TOKENIZER_ABI_VERSION,
            };
            status = lfm_tokenizer_special(tokenizer, &model->special);
            if (status != 0) fail(status, "cannot resolve native tokenizer control IDs");
            const size_t eos = integer(config, "eos_token_id");
            if (eos != model->special.im_end) {
                fail(-EINVAL, "lfm.eos_token_id disagrees with tokenizer <|im_end|>");
            }
            const size_t interleaved_text = integer(document, "interleaved_n_text");
            const size_t interleaved_audio = integer(document, "interleaved_n_audio");
            if (interleaved_text == 0 || interleaved_audio == 0 ||
                interleaved_text > UINT32_MAX || interleaved_audio > UINT32_MAX) {
                fail(-EINVAL, "interleaved generation cadence is invalid");
            }
            model->interleaved_text = (uint32_t)interleaved_text;
            model->interleaved_audio = (uint32_t)interleaved_audio;
            model->codebook_offsets.resize(codebooks);
            for (size_t codebook = 0; codebook < codebooks; ++codebook) {
                const size_t offset = multiply(codebook, audio_vocabulary,
                                               "audio codebook offset");
                if (offset > UINT32_MAX) fail(-EOVERFLOW, "audio codebook offset overflow");
                model->codebook_offsets[codebook] = (uint32_t)offset;
            }
            status = encode_tokens(
                tokenizer,
                "<|startoftext|><|im_start|>system\n"
                "Respond with interleaved text and audio.<|im_end|>\n"
                "<|im_start|>user\n",
                &model->initial_turn_tokens);
            if (status != 0) fail(status, "cannot tokenize native initial-turn grammar");
            status = encode_tokens(tokenizer,
                                   "<|im_end|>\n<|im_start|>user\n",
                                   &model->next_turn_tokens);
            if (status != 0) fail(status, "cannot tokenize native next-turn grammar");
            status = encode_tokens(tokenizer,
                                   "<|im_end|>\n<|im_start|>assistant\n",
                                   &model->assistant_tokens);
            if (status != 0) fail(status, "cannot tokenize native assistant grammar");
        }
        model->rope_theta = rope_theta;
        model->descriptors = std::move(descriptors);
        model->depth_rope_cos = std::move(depth_rope_cos);
        model->depth_rope_sin = std::move(depth_rope_sin);
        model->load_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - load_started)
                             .count();
        uint32_t required_sources =
            LFM_MODEL_PAYLOAD_READ_CONFIG |
            LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
            LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX;
        if (voice_model) required_sources |= LFM_MODEL_PAYLOAD_READ_TOKENIZER;
        status = model->accounting.publish(required_sources);
        if (status != 0) fail(status, "native model publication repeated");
        weights = nullptr;
        frontend = nullptr;
        conformer = nullptr;
        tokenizer = nullptr;
        mimi = nullptr;
        plan_id = 0;
        depth_plan_id = 0;
        *out = model.release();
        return 0;
    } catch (const ModelError &exception) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
        if (conformer) (void)lfm_conformer_destroy(conformer);
        if (frontend) (void)lfm_frontend_destroy(frontend);
        if (tokenizer) lfm_tokenizer_close(tokenizer);
        if (mimi) mimi_decode_plan_free(mimi);
        if (weights) lfm_weights_close(weights);
        set_error(error, error_length, exception.what());
        return exception.status();
    } catch (const std::bad_alloc &) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
        if (conformer) (void)lfm_conformer_destroy(conformer);
        if (frontend) (void)lfm_frontend_destroy(frontend);
        if (tokenizer) lfm_tokenizer_close(tokenizer);
        if (mimi) mimi_decode_plan_free(mimi);
        if (weights) lfm_weights_close(weights);
        set_error(error, error_length, "native model allocation failed");
        return -ENOMEM;
    } catch (const std::exception &exception) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
        if (conformer) (void)lfm_conformer_destroy(conformer);
        if (frontend) (void)lfm_frontend_destroy(frontend);
        if (tokenizer) lfm_tokenizer_close(tokenizer);
        if (mimi) mimi_decode_plan_free(mimi);
        if (weights) lfm_weights_close(weights);
        set_error(error, error_length, exception.what());
        return -EINVAL;
    }
}

extern "C" int lfm_model_close(LfmModel *model) {
    if (!model) return 0;
    {
        std::lock_guard<std::mutex> lock(model->lifecycle);
        if (model->conversations.load(std::memory_order_acquire) != 0) {
            return -EBUSY;
        }
        model->closing = true;
    }
    if (model->depth_plan_id != 0) {
        const int depth_status = lfm_engine_depth_clear(model->engine,
                                                        model->depth_plan_id);
        if (depth_status != 0) return depth_status;
        model->depth_plan_id = 0;
    }
    const int status = lfm_ctx_clear(model->engine, model->plan_id);
    if (status != 0) return status;
    if (model->conformer) (void)lfm_conformer_destroy(model->conformer);
    if (model->frontend) (void)lfm_frontend_destroy(model->frontend);
    if (model->tokenizer) lfm_tokenizer_close(model->tokenizer);
    if (model->mimi) mimi_decode_plan_free(model->mimi);
    lfm_weights_close(model->weights);
    delete model;
    return 0;
}

extern "C" int lfm_conversation_create(LfmModel *model,
                                        const LfmConversationConfigV1 *config,
                                        LfmConversation **out,
                                        char *error, size_t error_length) {
    if (!model || !config || !out || config->size < sizeof(*config) ||
        config->abi_version != LFM_MODEL_ABI_VERSION ||
        config->text_sampler.size < sizeof(config->text_sampler) ||
        config->text_sampler.abi_version != LFM_SAMPLE_ABI_VERSION ||
        config->audio_sampler.size < sizeof(config->audio_sampler) ||
        config->audio_sampler.abi_version != LFM_SAMPLE_ABI_VERSION) {
        return -EINVAL;
    }
    *out = nullptr;
    std::unique_lock<std::mutex> lifecycle(model->lifecycle);
    if (model->closing) return -ESHUTDOWN;
    try {
        std::unique_ptr<LfmConversation> conversation(
            new (std::nothrow) LfmConversation());
        if (!conversation) fail(-ENOMEM, "cannot allocate native conversation");
        conversation->model = model;
        conversation->text_sampler = config->text_sampler;
        conversation->audio_sampler = config->audio_sampler;
        conversation->memory.resize(model->descriptors.size());
        conversation->states.resize(model->descriptors.size());
        conversation->hidden.resize(model->hidden);
        conversation->window.capacity = model->max_context;
        conversation->window.runway = std::min<uint64_t>(model->max_context, 256);
        {
            const int status = lfm_engine_prefill_workspace_create(
                model->engine, model->plan_id, &conversation->prefill_workspace);
            if (status != 0) {
                fail(status, "cannot allocate native prefill workspace");
            }
        }
        if (model->tokenizer) {
            const int status = lfm_tokenizer_workspace_create(
                LFM_TEXT_COMMAND_MAX_BYTES, &conversation->tokenizer_workspace);
            if (status != 0) {
                fail(status, "cannot allocate bounded native tokenizer workspace");
            }
        }
        if (model->frontend) {
            const int status =
                lfm_frontend_workspace_create(&conversation->frontend_workspace);
            if (status != 0) fail(status, "cannot allocate native frontend workspace");
        }
        if (model->conformer) {
            const int status =
                lfm_conformer_workspace_create(&conversation->conformer_workspace);
            if (status != 0) fail(status, "cannot allocate native Conformer workspace");
        }
        if (model->mimi) {
            char mimi_error[512] = {};
            const int status = mimi_decode_state_new(&conversation->mimi,
                                                     model->mimi, mimi_error,
                                                     sizeof(mimi_error));
            if (status != 0) {
                fail(status, mimi_error[0] ? mimi_error
                                           : "cannot allocate native Mimi state");
            }
        }

        for (size_t index = 0; index < model->descriptors.size(); ++index) {
            const LfmLayerDesc &desc = model->descriptors[index];
            ConversationLayer &memory = conversation->memory[index];
            LfmLayerState &state = conversation->states[index];
            if (desc.kind == 1) {
                const size_t physical = (size_t)(conversation->window.capacity +
                                                 conversation->window.runway);
                const size_t stride = multiply(physical, desc.hd,
                                               "attention head stride");
                const size_t elements = multiply(desc.n_kv, stride,
                                                 "conversation KV planes");
                memory.keys.resize(elements);
                memory.values.resize(elements);
                state.k_plane = memory.keys.data();
                state.v_plane = memory.values.data();
                state.head_stride = stride;
                state.k_len = memory.keys.size();
                state.v_len = memory.values.size();
                continue;
            }
            const size_t tail = desc.k > 0 ? desc.k - 1 : 0;
            memory.convolution.resize(multiply(model->hidden, tail,
                                               "conversation convolution state"));
            state.conv_state = memory.convolution.data();
            state.conv_len = memory.convolution.size();
        }
        const int rope_status = build_rope(*conversation);
        if (rope_status != 0) fail(rope_status, "cannot build sliding RoPE state");
        const int seed_status = (config->flags & LFM_CONVERSATION_SEED_SYSTEM) != 0
                                    ? lfm_prng_seed_system(&conversation->initial_prng)
                                    : lfm_prng_seed_u64(&conversation->initial_prng,
                                                       config->seed);
        if (seed_status != 0) fail(seed_status, "cannot seed native conversation PRNG");
        const int reset_status = reset_memory(*conversation);
        if (reset_status != 0) fail(reset_status, "cannot reset native conversation");
        model->conversations.fetch_add(1, std::memory_order_acq_rel);
        *out = conversation.release();
        return 0;
    } catch (const ModelError &exception) {
        set_error(error, error_length, exception.what());
        return exception.status();
    } catch (const std::bad_alloc &) {
        set_error(error, error_length, "native conversation allocation failed");
        return -ENOMEM;
    } catch (const std::exception &exception) {
        set_error(error, error_length, exception.what());
        return -EINVAL;
    }
}

extern "C" int lfm_conversation_reset(LfmConversation *conversation) {
    if (!conversation) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return reset_memory(*conversation);
}

extern "C" int lfm_conversation_close(LfmConversation *conversation) {
    if (!conversation) return 0;
    if (conversation->active.test_and_set(std::memory_order_acquire)) return -EBUSY;
    LfmModel *model = conversation->model;
    delete conversation;
    std::lock_guard<std::mutex> lock(model->lifecycle);
    model->conversations.fetch_sub(1, std::memory_order_acq_rel);
    return 0;
}

extern "C" int lfm_model_info(const LfmModel *model, LfmModelInfoV1 *out) {
    if (!model || !out || out->size < sizeof(*out) ||
        out->abi_version != LFM_MODEL_ABI_VERSION) {
        return -EINVAL;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .resident_bytes = model->resident_bytes,
        .plan_id = model->plan_id,
        .depth_plan_id = model->depth_plan_id,
        .hidden = model->hidden,
        .ffn = model->ffn,
        .layers = model->layers,
        .vocab = model->vocab,
        .max_context = model->max_context,
        .codebooks = model->codebooks,
        .capabilities =
            (model->depth_plan_id != 0 ? LFM_MODEL_CAP_DEPTHFORMER : 0u) |
            (model->frontend != nullptr ? LFM_MODEL_CAP_FRONTEND : 0u) |
            (model->conformer != nullptr ? LFM_MODEL_CAP_CONFORMER : 0u) |
            (model->mimi != nullptr ? LFM_MODEL_CAP_MIMI : 0u),
        .reserved = {},
    };
    return 0;
}

extern "C" int lfm_model_memory(const LfmModel *model, LfmModelMemoryV1 *out) {
    if (!model || !out || out->size < sizeof(*out) ||
        out->abi_version != LFM_MODEL_ABI_VERSION) {
        return -EINVAL;
    }
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .source_bytes = model->source_bytes,
        .resident_image_bytes = model->resident_bytes,
        .directly_bound_bytes = model->accounting.directly_bound_bytes.load(
            std::memory_order_acquire),
        .derived_immutable_bytes =
            model->accounting.derived_immutable_bytes.load(
                std::memory_order_acquire),
        .materialized_weight_bytes =
            model->accounting.materialized_weight_bytes.load(
                std::memory_order_acquire),
        .compatibility_copied_bytes =
            model->accounting.compatibility_copied_bytes.load(
                std::memory_order_acquire),
        .payload_read_calls = model->accounting.payload_read_calls.load(
            std::memory_order_acquire),
        .payload_read_bytes = model->accounting.payload_read_bytes.load(
            std::memory_order_acquire),
        .post_publication_read_calls =
            model->accounting.post_publication_read_calls.load(
                std::memory_order_acquire),
        .post_publication_read_bytes =
            model->accounting.post_publication_read_bytes.load(
                std::memory_order_acquire),
        .post_publication_materialization_attempts =
            model->accounting.post_publication_materialization_attempts.load(
                std::memory_order_acquire),
        .post_publication_materialization_bytes =
            model->accounting.post_publication_materialization_bytes.load(
                std::memory_order_acquire),
        .publication_generation =
            model->accounting.publication_generation.load(
                std::memory_order_acquire),
        .load_ns = model->load_ns,
        .load_workers = model->load_workers,
        .load_tasks = model->load_tasks,
        .payload_read_coverage =
            model->accounting.payload_read_coverage.load(
                std::memory_order_acquire),
        .accounting_flags = model->accounting.flags.load(
            std::memory_order_acquire),
        .post_readiness_allocation_attempts =
            model->accounting.post_readiness_allocation_attempts.load(
                std::memory_order_acquire),
        .post_readiness_allocation_bytes =
            model->accounting.post_readiness_allocation_bytes.load(
                std::memory_order_acquire),
        .reserved = {},
    };
    return 0;
}

/* Focused native contract probe. It drives the real conversation playback
 * preparation with independently sealed preprocessor and codec rates; no
 * inference state, checkpoint tensor, or PCM plane is created. */
extern "C" LFM_INTERNAL_API int lfm_internal_playback_rate_contract_test(
    uint32_t preprocessor_rate, uint32_t playback_rate,
    uint32_t *out_preprocessor_rate, uint32_t *out_codec_rate,
    uint64_t *out_playback_frames, uint32_t *out_direct) {
    if (preprocessor_rate == 0 || playback_rate == 0 ||
        !out_preprocessor_rate || !out_codec_rate || !out_playback_frames ||
        !out_direct) {
        return -EINVAL;
    }
    *out_preprocessor_rate = 0;
    *out_codec_rate = 0;
    *out_playback_frames = 0;
    *out_direct = 0;

    LfmModel model;
    model.preprocessor_rate = preprocessor_rate;
    model.codec_rate = LFM_MIMI_SAMPLE_RATE;
    LfmConversation conversation;
    conversation.model = &model;
    size_t frames = 0;
    const int status = prepare_playback_claimed(
        conversation, playback_rate, &frames);
    if (status != 0) return status;
    *out_preprocessor_rate = model.preprocessor_rate;
    *out_codec_rate = model.codec_rate;
    *out_playback_frames = (uint64_t)frames;
    *out_direct = conversation.playback_resampler_stream ? 0u : 1u;
    return 0;
}

/* Focused preparation-policy gate. The real capture resampler/workspace and
 * playback stream are prepared through the production decision path. Opaque
 * frontend/Conformer handles are never dereferenced because the capture
 * high-water mark is already prepared; they only satisfy the same complete
 * resource predicate a production conversation established immediately
 * before this path. Every post-seal call below reaches prepare_pcm_claimed. */
extern "C" LFM_INTERNAL_API int
lfm_internal_conversation_allocation_seal_test(
    LfmModelMemoryV1 *after_compatible, LfmModelMemoryV1 *after_rejected,
    int *out_growth_status, int *out_capture_rate_status,
    int *out_playback_rate_status) {
    if (!after_compatible || after_compatible->size < sizeof(*after_compatible) ||
        after_compatible->abi_version != LFM_MODEL_ABI_VERSION ||
        !after_rejected || after_rejected->size < sizeof(*after_rejected) ||
        after_rejected->abi_version != LFM_MODEL_ABI_VERSION ||
        !out_growth_status || !out_capture_rate_status ||
        !out_playback_rate_status) {
        return -EINVAL;
    }

    LfmModel model;
    model.preprocessor_rate = 16000;
    model.codec_rate = LFM_MIMI_SAMPLE_RATE;
    model.mel_features = 128;
    model.hidden = 512;
    model.frontend = reinterpret_cast<LfmFrontend *>(uintptr_t{1});
    model.conformer = reinterpret_cast<LfmConformer *>(uintptr_t{1});

    LfmConversation conversation;
    conversation.model = &model;
    int status = lfm_resampler_create(16000, model.preprocessor_rate,
                                      &conversation.resampler);
    if (status != 0) return status;
    status = lfm_resampler_workspace_create(&conversation.resampler_workspace);
    if (status != 0) return status;
    status = lfm_resampler_workspace_reserve(
        conversation.resampler, conversation.resampler_workspace, 3200);
    if (status != 0) return status;
    conversation.prepared_samples = 3200;
    conversation.prepared_rate = 16000;
    conversation.frontend_workspace =
        reinterpret_cast<LfmFrontendWorkspace *>(uintptr_t{1});
    conversation.conformer_workspace =
        reinterpret_cast<LfmConformerWorkspace *>(uintptr_t{1});

    size_t frames = 0;
    status = prepare_pcm_claimed(conversation, 3200, 16000, 48000, &frames);
    if (status == 0 && !conversation.allocation_sealed) status = -EFAULT;
    if (status == 0) {
        status = prepare_pcm_claimed(conversation, 3200, 16000, 48000,
                                     &frames);
    }
    if (status == 0) {
        status = prepare_pcm_claimed(conversation, 1600, 16000, 48000,
                                     &frames);
    }
    if (status == 0) status = lfm_model_memory(&model, after_compatible);

    LfmResampler *const capture_plan = conversation.resampler;
    LfmResamplerWorkspace *const capture_workspace =
        conversation.resampler_workspace;
    LfmResamplerStream *const playback_plan =
        conversation.playback_resampler_stream;
    const size_t prepared_samples = conversation.prepared_samples;
    const uint32_t prepared_rate = conversation.prepared_rate;
    const size_t playback_frames = conversation.playback_frames;
    const uint32_t playback_rate = conversation.playback_rate;
    if (status == 0) {
        *out_growth_status = prepare_pcm_claimed(
            conversation, 6400, 16000, 48000, &frames);
        *out_capture_rate_status = prepare_pcm_claimed(
            conversation, 3200, 48000, 48000, &frames);
        *out_playback_rate_status = prepare_pcm_claimed(
            conversation, 3200, 16000, 24000, &frames);
        if (conversation.resampler != capture_plan ||
            conversation.resampler_workspace != capture_workspace ||
            conversation.playback_resampler_stream != playback_plan ||
            conversation.prepared_samples != prepared_samples ||
            conversation.prepared_rate != prepared_rate ||
            conversation.playback_frames != playback_frames ||
            conversation.playback_rate != playback_rate) {
            status = -EFAULT;
        }
    }
    if (status == 0) status = lfm_model_memory(&model, after_rejected);

    conversation.frontend_workspace = nullptr;
    conversation.conformer_workspace = nullptr;
    model.frontend = nullptr;
    model.conformer = nullptr;
    return status;
}

/* Focused non-production fault entry. The caller supplies tiny stack buffers;
 * both operations go through the same record-before-operation helpers used by
 * model construction. No numerical backend or checkpoint tensor is involved. */
extern "C" LFM_INTERNAL_API int lfm_internal_model_accounting_fault_test(
    const uint8_t *source, uint8_t *loaded, uint8_t *rejected, size_t bytes,
    LfmModelMemoryV1 *out, int *out_read_status,
    int *out_weight_status, int *out_policy_status) {
    if (!source || !loaded || !rejected || bytes == 0 || !out ||
        out->size < sizeof(*out) ||
        out->abi_version != LFM_MODEL_ABI_VERSION || !out_read_status ||
        !out_weight_status || !out_policy_status) {
        return -EINVAL;
    }

    ModelAccounting accounting;
    const LfmPayloadReadOwner owner = accounting.reader();
    int status = 0;
    {
        LfmPayloadReadScope source_scope(
            &owner, LFM_MODEL_PAYLOAD_READ_CONFIG, (uint64_t)bytes);
        status = source_scope.status();
        if (status != 0) return status;
        std::memcpy(loaded, source, bytes);
        status = source_scope.record(LFM_MODEL_PAYLOAD_READ_CONFIG,
                                     (uint64_t)bytes);
        if (status != 0) return status;
    }
    status = accounting.weight(WeightKind::Compatibility, (uint64_t)bytes);
    if (status != 0) return status;
    std::memcpy(loaded, source, bytes);
    status = accounting.publish(LFM_MODEL_PAYLOAD_READ_CONFIG);
    if (status != 0) return status;

    {
        LfmPayloadReadScope rejected_scope(
            &owner, LFM_MODEL_PAYLOAD_READ_CONFIG, (uint64_t)bytes);
        *out_read_status = rejected_scope.status();
        if (*out_read_status == 0) {
            std::memcpy(rejected, source, bytes);
            *out_read_status = rejected_scope.record(
                LFM_MODEL_PAYLOAD_READ_CONFIG, (uint64_t)bytes);
        }
    }
    *out_weight_status = accounting.weight(WeightKind::Compatibility,
                                           (uint64_t)bytes);
    if (*out_weight_status == 0) std::memcpy(rejected, source, bytes);
    *out_policy_status = accounting.weight_policy();
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .directly_bound_bytes = accounting.directly_bound_bytes.load(
            std::memory_order_acquire),
        .derived_immutable_bytes = accounting.derived_immutable_bytes.load(
            std::memory_order_acquire),
        .materialized_weight_bytes =
            accounting.materialized_weight_bytes.load(
                std::memory_order_acquire),
        .compatibility_copied_bytes =
            accounting.compatibility_copied_bytes.load(
                std::memory_order_acquire),
        .payload_read_calls = accounting.payload_read_calls.load(
            std::memory_order_acquire),
        .payload_read_bytes = accounting.payload_read_bytes.load(
            std::memory_order_acquire),
        .post_publication_read_calls =
            accounting.post_publication_read_calls.load(
                std::memory_order_acquire),
        .post_publication_read_bytes =
            accounting.post_publication_read_bytes.load(
                std::memory_order_acquire),
        .post_publication_materialization_attempts =
            accounting.post_publication_materialization_attempts.load(
                std::memory_order_acquire),
        .post_publication_materialization_bytes =
            accounting.post_publication_materialization_bytes.load(
                std::memory_order_acquire),
        .publication_generation =
            accounting.publication_generation.load(std::memory_order_acquire),
        .payload_read_coverage =
            accounting.payload_read_coverage.load(std::memory_order_acquire),
        .accounting_flags = accounting.flags.load(std::memory_order_acquire),
        .post_readiness_allocation_attempts =
            accounting.post_readiness_allocation_attempts.load(
                std::memory_order_acquire),
        .post_readiness_allocation_bytes =
            accounting.post_readiness_allocation_bytes.load(
                std::memory_order_acquire),
        .reserved = {},
    };
    return 0;
}

/* Actual source-entrypoint gate: after publication, each implementation must
 * return before inspecting or opening `path`. A nonexistent sentinel makes a
 * misplaced gate self-identify as an I/O error instead of -EPERM. */
extern "C" LFM_INTERNAL_API int lfm_internal_model_source_gate_test(
    const char *path, int *config_status, int *weights_status,
    int *tokenizer_status) {
    if (!path || !config_status || !weights_status || !tokenizer_status) {
        return -EINVAL;
    }
    ModelAccounting accounting;
    const LfmPayloadReadOwner owner = accounting.reader();
    {
        LfmPayloadReadScope install(
            &owner, LFM_MODEL_PAYLOAD_READ_CONFIG |
                        LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
                        LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX |
                        LFM_MODEL_PAYLOAD_READ_TOKENIZER);
        if (install.status() != 0) return install.status();
    }
    int status = accounting.publish(
        LFM_MODEL_PAYLOAD_READ_CONFIG |
        LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
        LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX |
        LFM_MODEL_PAYLOAD_READ_TOKENIZER);
    if (status != 0) return status;

    try {
        (void)read_json(&owner, fs::path(path));
        *config_status = 0;
    } catch (const ModelError &error) {
        *config_status = error.status();
    }

    LfmWeightImage *weights = nullptr;
    char weight_error[128] = {};
    *weights_status = lfm_weights_open_owned(
        path, &owner, &weights, weight_error, sizeof(weight_error));
    if (weights) lfm_weights_close(weights);

    LfmTokenizer *tokenizer = nullptr;
    char tokenizer_error[128] = {};
    *tokenizer_status = lfm_tokenizer_open_owned(
        path, &owner, &tokenizer, tokenizer_error,
        sizeof(tokenizer_error));
    if (tokenizer) lfm_tokenizer_close(tokenizer);
    return 0;
}
