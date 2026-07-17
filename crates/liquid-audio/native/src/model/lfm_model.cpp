#include "lfm_model.h"
#include "lfm_model_internal.h"

#include "flashkern_depth.h"
#include "flashkern_rope.h"
#include "lfm_audio_pass.h"
#include "lfm_conformer.h"
#include "lfm_frontend.h"
#include "lfm_mimi.h"
#include "lfm_model_plan.h"
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

extern "C" {
#include "kc_atomic.h"
#include "kc_port.h"
}

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

extern "C" void lfm_f32_to_bf16(const float *input, uint16_t *output, int count);

namespace {

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

Json read_json(const fs::path &path) {
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
    if (!bytes.empty() && std::fread(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
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

    std::vector<Span> spans;
    uint64_t total = 0;

    void add(const LfmTensorView &view) {
        const auto found = std::find_if(
            spans.begin(), spans.end(), [&](const Span &span) {
                return span.data == view.data && span.bytes == view.bytes;
            });
        if (found != spans.end()) return;
        if (view.bytes > UINT64_MAX - total) {
            fail(-EOVERFLOW, "directly bound tensor byte accounting overflow");
        }
        spans.push_back({view.data, view.bytes});
        total += view.bytes;
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

/* A fair, expected-value pass gate. The lane board intentionally executes one
 * complete pass at a time even though the SQ and per-ticket scratch have depth
 * two. The synchronous session path queues conversations at pass boundaries
 * rather than surfacing compatibility admission details. */
struct ExecutionGate {
    alignas(64) uint32_t next = 0;
    alignas(64) uint32_t serving = 0;
    kc_port_wait_word *wait = nullptr;

    int prepare() {
        if (!kc_atomic_u32_is_lock_free(&next) ||
            !kc_atomic_u32_is_lock_free(&serving)) {
            return -ENOTSUP;
        }
        return kc_port_wait_u32_prepare(&serving, &wait);
    }

    uint32_t acquire() {
        const uint32_t ticket = kc_atomic_u32_fetch_add_acq_rel(&next, 1);
        for (;;) {
            const uint32_t observed = kc_atomic_u32_load_acquire(&serving);
            if (observed == ticket) return ticket;
            (void)kc_port_wait_u32(wait, observed, 0);
        }
    }

    void release(uint32_t ticket) {
        (void)ticket;
        /* The release increment publishes all engine outputs before the next
         * ticket observes its turn with an acquire load. */
        (void)kc_atomic_u32_fetch_add_release(&serving, 1);
        kc_port_wake_u32_all(wait);
    }

    ~ExecutionGate() {
        if (wait) kc_port_wait_u32_release(wait);
    }
};

class ExecutionClaim {
  public:
    explicit ExecutionClaim(ExecutionGate &gate)
        : gate_(&gate), ticket_(gate.acquire()) {}
    ~ExecutionClaim() { gate_->release(ticket_); }

    ExecutionClaim(const ExecutionClaim &) = delete;
    ExecutionClaim &operator=(const ExecutionClaim &) = delete;

  private:
    ExecutionGate *gate_;
    uint32_t ticket_;
};

} // namespace

struct LfmModel {
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
    uint64_t directly_bound_bytes = 0;
    uint64_t derived_immutable_bytes = 0;
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
    uint32_t sample_rate = 0;
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
    ExecutionGate execution;
    std::mutex lifecycle;
    bool closing = false;
    std::atomic<uint32_t> conversations{0};
};

struct ConversationLayer {
    std::vector<uint16_t> keys;
    std::vector<uint16_t> values;
    std::vector<uint16_t> convolution;
};

struct LfmConversation {
    LfmModel *model = nullptr;
    void *prefill_workspace = nullptr;
    LfmFrontendWorkspace *frontend_workspace = nullptr;
    LfmConformerWorkspace *conformer_workspace = nullptr;
    LfmResampler *resampler = nullptr;
    LfmResamplerWorkspace *resampler_workspace = nullptr;
    MimiDecodeState *mimi = nullptr;
    LfmTokenizerWorkspace *tokenizer_workspace = nullptr;
    std::vector<ConversationLayer> memory;
    std::vector<LfmLayerState> states;
    std::vector<uint16_t> rope_cos;
    std::vector<uint16_t> rope_sin;
    std::vector<float> rope_cos_f32;
    std::vector<float> rope_sin_f32;
    std::vector<uint16_t> hidden;
    std::vector<float> resampled;
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
    std::atomic_flag active = ATOMIC_FLAG_INIT;

    ~LfmConversation() {
        lfm_engine_prefill_workspace_destroy(prefill_workspace);
        lfm_tokenizer_workspace_destroy(tokenizer_workspace);
        if (mimi) mimi_decode_state_free(mimi);
        if (resampler_workspace) {
            (void)lfm_resampler_workspace_destroy(resampler_workspace);
        }
        if (resampler) (void)lfm_resampler_destroy(resampler);
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

extern "C" int lfm_context_window_commit(LfmContextWindowState *window) {
    if (!window || window->capacity == 0 ||
        window->position >= window->capacity ||
        window->position > UINT64_MAX - window->rope_base ||
        window->rope_base + window->position != window->cursor ||
        window->cursor == UINT64_MAX) {
        return -EINVAL;
    }
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
    return 0;
}

int prefill_rows_claimed(LfmConversation &conversation, const uint16_t *rows,
                         size_t element_count, uint64_t *out_position,
                         bool sample_last = false, uint32_t *sampled = nullptr) {
    const size_t hidden = conversation.model->hidden;
    if (!rows || element_count == 0 || !out_position || hidden == 0 ||
        element_count % hidden != 0) {
        return -EINVAL;
    }
    const size_t row_count = element_count / hidden;
    int status = admit_context(conversation, row_count);
    if (status != 0) return status;
    ExecutionClaim execution(conversation.model->execution);
    for (size_t index = 0; index < row_count;) {
        size_t count = 0;
        status = lfm_context_window_prefill_chunk(
            &conversation.window, row_count - index, LFM_PREFILL_MAX_ROWS,
            &count);
        if (status != 0) return status;
        status = reserve_context(conversation, count);
        if (status != 0) return status;
        const bool sample = sample_last && index + count == row_count;
        status = lfm_engine_prefill(
            conversation.model->engine, conversation.model->plan_id,
            conversation.prefill_workspace, nullptr, rows + index * hidden,
            count, /* embedding_kind = provided BF16 rows */ 2,
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
            sample ? &conversation.text_sampler : nullptr,
            sample ? &conversation.prng : nullptr,
            sample ? sampled : nullptr, conversation.model->lanes);
        if (status != 0) return status;
        for (size_t row = 0; row < count; ++row) {
            status = commit_context(conversation);
            if (status != 0) return status;
        }
        conversation.hidden_ready = true;
        index += count;
    }
    *out_position = conversation.window.cursor;
    return 0;
}

int prepare_pcm_claimed(LfmConversation &conversation, size_t max_sample_count,
                        uint32_t sample_rate) {
    LfmModel *model = conversation.model;
    if (!model || !model->frontend || !model->conformer ||
        !conversation.frontend_workspace || !conversation.conformer_workspace ||
        max_sample_count == 0 || sample_rate == 0 || model->sample_rate == 0 ||
        model->mel_features == 0 || model->hidden == 0) {
        return -EINVAL;
    }
    if (conversation.resampler && conversation.resampler_workspace &&
        conversation.prepared_rate == sample_rate &&
        conversation.prepared_samples >= max_sample_count) {
        return 0;
    }

    LfmResampler *plan = nullptr;
    LfmResamplerWorkspace *workspace = nullptr;
    int status = lfm_resampler_create(sample_rate, model->sample_rate, &plan);
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
                sample_rate == model->sample_rate ? 0 : (size_t)target_samples);
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
    conversation.prepared_rate = sample_rate;
    return 0;
}

int prepare_pcm_rows_claimed(LfmConversation &conversation, const float *pcm,
                             size_t sample_count, uint32_t sample_rate,
                             size_t *out_values) {
    LfmModel *model = conversation.model;
    if (!pcm || sample_count == 0 || sample_rate == 0 || !out_values) return -EINVAL;
    if (!model->frontend || !model->conformer || !conversation.frontend_workspace ||
        !conversation.conformer_workspace || model->sample_rate == 0 ||
        model->mel_features == 0 || !conversation.resampler ||
        !conversation.resampler_workspace) {
        return -ENOTSUP;
    }
    if (sample_rate != conversation.prepared_rate) return -EINVAL;
    if (sample_count > conversation.prepared_samples) return -ENOBUFS;

    uint64_t adapted_values = 0;
    const LfmAudioEncodePassV1 pass = {
        .size = sizeof(LfmAudioEncodePassV1),
        .abi_version = LFM_AUDIO_PASS_ABI,
        .resampler = conversation.resampler,
        .resampler_workspace = conversation.resampler_workspace,
        .frontend = model->frontend,
        .frontend_workspace = conversation.frontend_workspace,
        .conformer = model->conformer,
        .conformer_workspace = conversation.conformer_workspace,
        .pcm = pcm,
        .sample_count = sample_count,
        .resampled = conversation.resampled.empty()
                         ? nullptr
                         : conversation.resampled.data(),
        .resampled_capacity = conversation.resampled.size(),
        .mel = conversation.mel_bf16.data(),
        .mel_capacity = conversation.mel_bf16.size(),
        .adapted = conversation.adapted.data(),
        .adapted_capacity = conversation.adapted.size(),
        .out_adapted_values = &adapted_values,
    };
    ExecutionClaim execution(model->execution);
    const int status = lfm_engine_audio_encode(model->engine, model->plan_id,
                                               &pass);
    if (status != 0) return status;
    if (adapted_values > std::numeric_limits<size_t>::max()) return -EOVERFLOW;
    *out_values = (size_t)adapted_values;
    return 0;
}

int prefill_pcm_claimed(LfmConversation &conversation, const float *pcm,
                        size_t sample_count, uint32_t sample_rate,
                        uint64_t *out_position, bool sample_last = false,
                        uint32_t *sampled = nullptr) {
    size_t values = 0;
    const int status = prepare_pcm_rows_claimed(
        conversation, pcm, sample_count, sample_rate, &values);
    if (status != 0) return status;
    return prefill_rows_claimed(conversation, conversation.adapted.data(),
                                values, out_position, sample_last, sampled);
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

int prefill_text_ids_claimed(LfmConversation &conversation,
                             const uint32_t *tokens, size_t token_count,
                             bool sample_last, uint32_t *sampled) {
    if (!tokens || token_count == 0) return -EINVAL;
    if (std::any_of(tokens, tokens + token_count, [&](uint32_t id) {
            return id >= conversation.model->vocab;
        })) {
        return -ERANGE;
    }
    int status = admit_context(conversation, token_count);
    if (status != 0) return status;
    ExecutionClaim execution(conversation.model->execution);
    for (size_t index = 0; index < token_count;) {
        size_t count = 0;
        status = lfm_context_window_prefill_chunk(
            &conversation.window, token_count - index, LFM_PREFILL_MAX_ROWS,
            &count);
        if (status != 0) return status;
        status = reserve_context(conversation, count);
        if (status != 0) return status;
        const bool sample = sample_last && index + count == token_count;
        status = lfm_engine_prefill(
            conversation.model->engine, conversation.model->plan_id,
            conversation.prefill_workspace, tokens + index, nullptr, count, 0,
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
            sample ? &conversation.text_sampler : nullptr,
            sample ? &conversation.prng : nullptr,
            sample ? sampled : nullptr, conversation.model->lanes);
        if (status != 0) return status;
        for (size_t row = 0; row < count; ++row) {
            status = commit_context(conversation);
            if (status != 0) return status;
        }
        conversation.hidden_ready = true;
        index += count;
    }
    return 0;
}

int prefill_text_claimed(LfmConversation &conversation, bool sample_last,
                         uint32_t *sampled) {
    return prefill_text_ids_claimed(conversation,
                                    conversation.token_scratch.data(),
                                    conversation.token_count,
                                    sample_last, sampled);
}

int forward_pending_claimed(LfmConversation &conversation, bool sample,
                            uint32_t *sampled) {
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
    int status = reserve_context(conversation, 1);
    if (status != 0) return status;
    ExecutionClaim execution(conversation.model->execution);
    status = lfm_engine_token_pass(
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
        conversation.hidden.data(),
        conversation.hidden.size(), nullptr, 0,
        sample ? &conversation.text_sampler : nullptr,
        sample ? &conversation.prng : nullptr, sample ? sampled : nullptr,
        conversation.model->lanes, nullptr);
    if (status != 0) return status;
    status = commit_context(conversation);
    if (status != 0) return status;
    conversation.hidden_ready = true;
    return 0;
}

int commit_pending_claimed(LfmConversation &conversation) {
    if (conversation.pending_count == 0) return 0;
    const int status = forward_pending_claimed(conversation, false, nullptr);
    if (status != 0) return status;
    conversation.pending_count = 0;
    conversation.pending_kind = 0;
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

int emit_audio_claimed(LfmConversation &conversation, LfmNativeEmission *out) {
    if (conversation.modality_left > 0) --conversation.modality_left;
    if (conversation.model->depth_plan_id == 0 || conversation.model->codebooks == 0 ||
        conversation.model->codebooks > LFM_AUDIO_TOKEN_CAPACITY ||
        conversation.model->codebook_offsets.size() != conversation.model->codebooks) {
        return -ENOTSUP;
    }
    uint32_t codes[LFM_AUDIO_TOKEN_CAPACITY] = {};
    int status = 0;
    {
        ExecutionClaim execution(conversation.model->execution);
        status = lfm_engine_depth_frame(
            conversation.model->engine, conversation.model->depth_plan_id,
            conversation.hidden.data(), conversation.hidden.size(),
            &conversation.audio_sampler, &conversation.prng, codes,
            conversation.model->codebooks);
    }
    if (status != 0) return status;
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

int next_emission_claimed(LfmConversation &conversation,
                          LfmNativeEmission *out) {
    clear_emission(out, conversation.window.cursor);
    if (!conversation.generation_active || conversation.generation_ended) {
        out->kind = LFM_NATIVE_EMISSION_FINISHED;
        return 0;
    }
    if (conversation.modality == 1) {
        uint32_t sampled = 0;
        const int status = forward_pending_claimed(conversation, true, &sampled);
        if (status != 0) return status;
        conversation.pending_count = 0;
        conversation.pending_kind = 0;
        out->position = conversation.window.cursor;
        return emit_text_claimed(conversation, sampled, out);
    }
    const int status = forward_pending_claimed(conversation, false, nullptr);
    if (status != 0) return status;
    conversation.pending_count = 0;
    conversation.pending_kind = 0;
    out->position = conversation.window.cursor;
    return emit_audio_claimed(conversation, out);
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
     * once: interleaved audio runs within the turn must share codec state. */
    if (conversation.mimi) mimi_decode_state_reset(conversation.mimi);
    return 0;
}

int prefill_turn_prefix_claimed(LfmConversation &conversation) {
    const std::vector<uint32_t> &prefix = conversation.window.cursor == 0
        ? conversation.model->initial_turn_tokens
        : conversation.model->next_turn_tokens;
    return prefill_text_ids_claimed(conversation, prefix.data(), prefix.size(),
                                    false, nullptr);
}

int prefill_assistant_claimed(LfmConversation &conversation,
                              uint32_t *sampled) {
    return prefill_text_ids_claimed(
        conversation, conversation.model->assistant_tokens.data(),
        conversation.model->assistant_tokens.size(), true, sampled);
}

} // namespace

int lfm_conversation_prepare_pcm_native(LfmConversation *conversation,
                                        size_t max_sample_count,
                                        uint32_t sample_rate) {
    if (!conversation || max_sample_count == 0 || sample_rate == 0) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return prepare_pcm_claimed(*conversation, max_sample_count, sample_rate);
}

int lfm_conversation_begin_pcm_native(LfmConversation *conversation,
                                      const float *pcm, size_t sample_count,
                                      uint32_t sample_rate,
                                      LfmNativeEmission *out) {
    if (!conversation || !pcm || sample_count == 0 || sample_rate == 0 || !out) {
        return -EINVAL;
    }
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
    size_t adapted_values = 0;
    int status = prepare_pcm_rows_claimed(*conversation, pcm, sample_count,
                                          sample_rate, &adapted_values);
    if (status != 0) return status;
    const size_t rows = adapted_values / model->hidden;
    const std::vector<uint32_t> &prefix = conversation->window.cursor == 0
        ? model->initial_turn_tokens
        : model->next_turn_tokens;
    if (prefix.size() > model->max_context ||
        rows > model->max_context - prefix.size() ||
        model->assistant_tokens.size() >
            model->max_context - prefix.size() - rows) {
        return -ENOSPC;
    }
    status = admit_context(*conversation,
                           prefix.size() + rows +
                               model->assistant_tokens.size());
    if (status != 0) return status;

    status = prefill_turn_prefix_claimed(*conversation);
    if (status != 0) return status;
    uint64_t position = conversation->window.cursor;
    status = prefill_rows_claimed(*conversation, conversation->adapted.data(),
                                  adapted_values, &position);
    if (status != 0) return status;

    uint32_t sampled = 0;
    status = prefill_assistant_claimed(*conversation, &sampled);
    if (status != 0) return status;
    return begin_generation_claimed(*conversation, sampled, out);
}

int lfm_conversation_begin_text_native(LfmConversation *conversation,
                                       const char *text, size_t text_bytes,
                                       LfmNativeEmission *out) {
    if (!conversation || !text || text_bytes == 0 || !out) return -EINVAL;
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
    status = admit_context(*conversation,
                           prefix.size() + text_tokens +
                               model->assistant_tokens.size());
    if (status != 0) return status;
    status = prefill_turn_prefix_claimed(*conversation);
    if (status != 0) return status;
    status = prefill_text_claimed(*conversation, false, nullptr);
    if (status != 0) return status;
    uint32_t sampled = 0;
    status = prefill_assistant_claimed(*conversation, &sampled);
    if (status != 0) return status;
    return begin_generation_claimed(*conversation, sampled, out);
}

int lfm_conversation_begin_mixed_native(
    LfmConversation *conversation, const char *text, size_t text_bytes,
    const float *pcm, size_t sample_count, uint32_t sample_rate,
    LfmNativeEmission *out) {
    if (!conversation || !text || text_bytes == 0 || !pcm ||
        sample_count == 0 || sample_rate == 0 || !out) {
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

    int status = encode_text(*conversation, text, text_bytes);
    if (status != 0) return status;
    const size_t text_tokens = conversation->token_count;
    if (text_tokens == 0) return -EINVAL;

    size_t adapted_values = 0;
    status = prepare_pcm_rows_claimed(*conversation, pcm, sample_count,
                                      sample_rate, &adapted_values);
    if (status != 0) return status;
    if (adapted_values == 0 || adapted_values % model->hidden != 0) {
        return -EINVAL;
    }
    const size_t rows = adapted_values / model->hidden;
    const std::vector<uint32_t> &prefix = conversation->window.cursor == 0
        ? model->initial_turn_tokens
        : model->next_turn_tokens;
    LfmMixedTurnPlan plan{};
    status = lfm_mixed_turn_plan(
        model->max_context, prefix.size(), text_tokens, rows,
        model->assistant_tokens.size(), &plan);
    if (status != 0) return status;
    status = admit_context(*conversation, plan.total);
    if (status != 0) return status;

    status = prefill_turn_prefix_claimed(*conversation);
    if (status != 0) return status;
    status = prefill_text_claimed(*conversation, false, nullptr);
    if (status != 0) return status;
    uint64_t position = conversation->window.cursor;
    status = prefill_rows_claimed(*conversation, conversation->adapted.data(),
                                  adapted_values, &position);
    if (status != 0) return status;
    uint32_t sampled = 0;
    status = prefill_assistant_claimed(*conversation, &sampled);
    if (status != 0) return status;
    return begin_generation_claimed(*conversation, sampled, out);
}

int lfm_conversation_next_native(LfmConversation *conversation,
                                 LfmNativeEmission *out) {
    if (!conversation || !out) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return next_emission_claimed(*conversation, out);
}

int lfm_conversation_interrupt_native(LfmConversation *conversation) {
    if (!conversation) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    /* An emission is published before it becomes the input to the following
     * recurrence pass. Interrupt/truncation must close that one-pass seam: the
     * already-produced token belongs in KV and ShortConv even though interrupt
     * performs no sampling and publishes nothing. This also commits the
     * recurrence-only EOAudio code tuple; im_end is terminal and is never put
     * in pending_ids by emit_text_claimed. */
    const int status = commit_pending_claimed(*conversation);
    if (status != 0) return status;
    conversation->generation_active = false;
    conversation->generation_ended = true;
    return 0;
}

/* Private implementation-backed test seams. They are intentionally absent
 * from every product and transitional header. */
extern "C" int lfm_internal_conversation_interrupt_for_test(
    LfmConversation *conversation) {
    return lfm_conversation_interrupt_native(conversation);
}

extern "C" int lfm_internal_conversation_stage_pending_for_test(
    LfmConversation *conversation, const uint32_t *ids, size_t id_count,
    uint32_t embedding_kind) {
    if (!conversation || !ids || id_count == 0 ||
        id_count > LFM_INPUT_MAX_IDS || embedding_kind > 1) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if (conversation->pending_count != 0 || conversation->generation_active) {
        return -EALREADY;
    }
    if ((embedding_kind == 0 &&
         (id_count != 1 || ids[0] >= conversation->model->vocab)) ||
        (embedding_kind == 1 &&
         std::any_of(ids, ids + id_count, [&](uint32_t id) {
             return id >= conversation->model->audio_rows;
         }))) {
        return -ERANGE;
    }
    std::copy(ids, ids + id_count, conversation->pending_ids);
    conversation->pending_count = (uint32_t)id_count;
    conversation->pending_kind = embedding_kind;
    conversation->generation_active = true;
    conversation->generation_ended = false;
    return 0;
}

extern "C" int lfm_internal_conversation_cache_digest_for_test(
    LfmConversation *conversation, uint64_t *out) {
    if (!conversation || !out) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    uint64_t digest = UINT64_C(14695981039346656037);
    const auto append = [&](const void *data, size_t bytes) {
        const uint8_t *source = static_cast<const uint8_t *>(data);
        for (size_t index = 0; index < bytes; ++index) {
            digest ^= source[index];
            digest *= UINT64_C(1099511628211);
        }
    };
    append(&conversation->window, sizeof(conversation->window));
    for (const ConversationLayer &layer : conversation->memory) {
        const size_t key_bytes = layer.keys.size() * sizeof(uint16_t);
        const size_t value_bytes = layer.values.size() * sizeof(uint16_t);
        const size_t convolution_bytes =
            layer.convolution.size() * sizeof(uint16_t);
        append(&key_bytes, sizeof(key_bytes));
        append(layer.keys.data(), key_bytes);
        append(&value_bytes, sizeof(value_bytes));
        append(layer.values.data(), value_bytes);
        append(&convolution_bytes, sizeof(convolution_bytes));
        append(layer.convolution.data(), convolution_bytes);
    }
    const size_t hidden_bytes = conversation->hidden.size() * sizeof(uint16_t);
    append(&hidden_bytes, sizeof(hidden_bytes));
    append(conversation->hidden.data(), hidden_bytes);
    *out = digest;
    return 0;
}

extern "C" int lfm_internal_conversation_prng_digest_for_test(
    LfmConversation *conversation, uint64_t *out) {
    if (!conversation || !out) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    uint64_t digest = UINT64_C(14695981039346656037);
    const uint8_t *bytes =
        reinterpret_cast<const uint8_t *>(&conversation->prng);
    for (size_t index = 0; index < sizeof(conversation->prng); ++index) {
        digest ^= bytes[index];
        digest *= UINT64_C(1099511628211);
    }
    *out = digest;
    return 0;
}

int lfm_conversation_decode_native(LfmConversation *conversation,
                                   const uint32_t *codes, size_t code_count,
                                   float *pcm, size_t pcm_capacity,
                                   size_t *out_samples) {
    if (!conversation || !codes || !pcm || !out_samples ||
        code_count != LFM_MIMI_CODEBOOKS ||
        pcm_capacity < LFM_MIMI_PCM_CAPACITY) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if (!conversation->mimi) return -ENOTSUP;
    ExecutionClaim execution(conversation->model->execution);
    return lfm_engine_mimi_decode(
        conversation->model->engine, conversation->model->plan_id,
        conversation->mimi, codes, code_count, pcm, pcm_capacity, out_samples);
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
    BindingLedger bindings;
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
        const Json document = read_json(root / "config.json");
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
        const bool codec_exists = fs::is_regular_file(codec_path, codec_error);
        if (voice_model && (!codec_exists || codec_error)) {
            fail(-ENOENT, "native LFM2-Audio requires its Mimi codec checkpoint");
        }
        char weight_error[512] = {};
        const std::string codec_native = codec_path.string();
        int status = codec_exists
                         ? lfm_weights_open_bundle(path, codec_native.c_str(), &weights,
                                                   weight_error, sizeof(weight_error))
                         : lfm_weights_open(path, &weights, weight_error,
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
            const auto attention_entry = encoder.find("self_attention_model");
            if (attention_entry != encoder.end() &&
                (!attention_entry->is_string() ||
                 attention_entry->get<std::string>() != "rel_pos")) {
                fail(-EOPNOTSUPP, "native Conformer requires rel_pos attention");
            }
            if (!boolean(encoder, "xscaling", true)) {
                fail(-EOPNOTSUPP, "native Conformer requires encoder xscaling");
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
            status = lfm_tokenizer_open(tokenizer_native.c_str(), &tokenizer,
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

        std::unique_ptr<LfmModel> model(new (std::nothrow) LfmModel());
        if (!model) fail(-ENOMEM, "cannot allocate native model handle");
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
            bindings.total,
            lfm_conformer_bound_weight_bytes(conformer),
            mimi_decode_plan_bound_weight_bytes(mimi),
        };
        for (uint64_t bytes : bound_parts) {
            if (bytes > UINT64_MAX - model->directly_bound_bytes) {
                fail(-EOVERFLOW, "directly bound tensor byte accounting overflow");
            }
            model->directly_bound_bytes += bytes;
        }
        const uint64_t derived_parts[] = {
            frontend_derived,
            lfm_conformer_derived_bytes(conformer),
            (uint64_t)(depth_rope_cos.size() + depth_rope_sin.size()) *
                sizeof(float),
            mimi_decode_plan_derived_bytes(mimi),
        };
        for (uint64_t bytes : derived_parts) {
            if (bytes > UINT64_MAX - model->derived_immutable_bytes) {
                fail(-EOVERFLOW, "derived model byte accounting overflow");
            }
            model->derived_immutable_bytes += bytes;
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
        model->sample_rate = sample_rate;
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
        status = model->execution.prepare();
        if (status != 0) fail(status, "cannot prepare the native model pass gate");
        model->load_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - load_started)
                             .count();
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

extern "C" int lfm_conversation_step(LfmConversation *conversation,
                                      const uint32_t *ids, size_t id_count,
                                      uint32_t embedding_kind,
                                      LfmTokenResultV1 *out) {
    if (!conversation || !ids || id_count == 0 || id_count > 8 || !out ||
        out->size < sizeof(*out) || out->abi_version != LFM_MODEL_ABI_VERSION ||
        embedding_kind > 1) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if ((embedding_kind == 0 && (id_count != 1 || ids[0] >= conversation->model->vocab)) ||
        (embedding_kind == 1 &&
         std::any_of(ids, ids + id_count, [&](uint32_t id) {
             return id >= conversation->model->audio_rows;
         }))) {
        return -ERANGE;
    }
    int status = reserve_context(*conversation, 1);
    if (status != 0) return status;
    uint32_t token = 0;
    ExecutionClaim execution(conversation->model->execution);
    status = lfm_engine_token_pass(
        conversation->model->engine, conversation->model->plan_id,
        ids, id_count, embedding_kind, conversation->states.data(),
        conversation->states.size(), (size_t)conversation->window.position,
        conversation->rope_cos.empty() ? nullptr
            : conversation->rope_cos.data() +
                  conversation->window.start * conversation->rope_half,
        conversation->rope_sin.empty() ? nullptr
            : conversation->rope_sin.data() +
                  conversation->window.start * conversation->rope_half,
        conversation->rope_cos.size() -
            conversation->window.start * conversation->rope_half,
        conversation->hidden.data(),
        conversation->hidden.size(), nullptr, 0, &conversation->text_sampler,
        &conversation->prng, &token, conversation->model->lanes, nullptr);
    if (status != 0) return status;
    const uint64_t completed_position = conversation->window.cursor;
    status = commit_context(*conversation);
    if (status != 0) return status;
    conversation->hidden_ready = true;
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .position = completed_position,
        .sampled_token = token,
        .input_count = (uint32_t)id_count,
        .embedding_kind = embedding_kind,
        .flags = 0,
        .reserved = {},
    };
    return 0;
}

extern "C" int lfm_conversation_prefill(LfmConversation *conversation,
                                          const LfmInputV1 *inputs,
                                          size_t input_count,
                                          uint64_t *out_position) {
    if (!conversation || !inputs || input_count == 0 || !out_position) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    for (size_t index = 0; index < input_count; ++index) {
        const LfmInputV1 &input = inputs[index];
        if (input.size < sizeof(input) ||
            input.abi_version != LFM_MODEL_ABI_VERSION ||
            input.embedding_kind > 1 || input.id_count == 0 ||
            input.id_count > LFM_INPUT_MAX_IDS ||
            (input.embedding_kind == 0 && input.id_count != 1)) {
            return -EINVAL;
        }
        if ((input.embedding_kind == 0 &&
             input.ids[0] >= conversation->model->vocab) ||
            (input.embedding_kind == 1 &&
             std::any_of(input.ids, input.ids + input.id_count,
                         [&](uint32_t id) {
                             return id >= conversation->model->audio_rows;
                         }))) {
            return -ERANGE;
        }
    }
    int status = admit_context(*conversation, input_count);
    if (status != 0) return status;
    for (size_t index = 0; index < input_count; ++index) {
        const LfmInputV1 &input = inputs[index];
        status = reserve_context(*conversation, 1);
        if (status != 0) return status;
        ExecutionClaim execution(conversation->model->execution);
        status = lfm_engine_token_pass(
            conversation->model->engine, conversation->model->plan_id,
            input.ids, input.id_count, input.embedding_kind,
            conversation->states.data(), conversation->states.size(),
            (size_t)conversation->window.position,
            conversation->rope_cos.empty() ? nullptr
                : conversation->rope_cos.data() +
                      conversation->window.start * conversation->rope_half,
            conversation->rope_sin.empty() ? nullptr
                : conversation->rope_sin.data() +
                      conversation->window.start * conversation->rope_half,
            conversation->rope_cos.size() -
                conversation->window.start * conversation->rope_half,
            conversation->hidden.data(),
            conversation->hidden.size(), nullptr, 0, nullptr, nullptr, nullptr,
            conversation->model->lanes, nullptr);
        if (status != 0) return status;
        status = commit_context(*conversation);
        if (status != 0) return status;
        conversation->hidden_ready = true;
    }
    *out_position = conversation->window.cursor;
    return 0;
}

// Native audio-in prefill: prefill a flat continuous-embedding plane (the
// Conformer/adapter output, `[row_count, model.hidden]` bf16) into KV, one native
// token pass per row via the provided-embedding path (`embed_kind == 2`). C++
// owns the geometry: callers provide only the actual element count. `rows` is a
// borrowed VIEW into the caller's buffer — no payload crosses the ABI, and the
// backbone reads it in place, exactly as the discrete-id prefill loops. Same
// sequential-per-position shape as `lfm_conversation_prefill`.
extern "C" int lfm_conversation_prefill_audio(LfmConversation *conversation,
                                              const uint16_t *rows, size_t element_count,
                                              uint64_t *out_position) {
    if (!conversation || !rows || element_count == 0 || !out_position) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return prefill_rows_claimed(*conversation, rows, element_count, out_position);
}

extern "C" int lfm_conversation_prefill_pcm_f32(
    LfmConversation *conversation, const float *pcm, size_t sample_count,
    uint32_t sample_rate, uint64_t *out_position) {
    if (!conversation || !pcm || sample_count == 0 || sample_rate == 0 ||
        !out_position) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    return prefill_pcm_claimed(*conversation, pcm, sample_count, sample_rate,
                               out_position);
}

extern "C" int lfm_conversation_audio_frame(LfmConversation *conversation,
                                              LfmAudioResultV1 *out) {
    if (!conversation || !out || out->size < sizeof(*out) ||
        out->abi_version != LFM_MODEL_ABI_VERSION) {
        return -EINVAL;
    }
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if (!conversation->hidden_ready) return -ENODATA;
    if (conversation->model->depth_plan_id == 0 ||
        conversation->model->codebooks == 0 ||
        conversation->model->codebooks > LFM_AUDIO_TOKEN_CAPACITY) {
        return -ENOTSUP;
    }
    uint32_t tokens[LFM_AUDIO_TOKEN_CAPACITY] = {};
    ExecutionClaim execution(conversation->model->execution);
    const int status = lfm_engine_depth_frame(
        conversation->model->engine, conversation->model->depth_plan_id,
        conversation->hidden.data(), conversation->hidden.size(),
        &conversation->audio_sampler, &conversation->prng, tokens,
        conversation->model->codebooks);
    if (status != 0) return status;
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .source_position = conversation->window.cursor - 1,
        .token_count = conversation->model->codebooks,
        .flags = 0,
        .tokens = {},
        .reserved = {},
    };
    std::copy(tokens, tokens + conversation->model->codebooks, out->tokens);
    return 0;
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
        .directly_bound_bytes = model->directly_bound_bytes,
        .derived_immutable_bytes = model->derived_immutable_bytes,
        .compatibility_copied_bytes =
            lfm_conformer_materialized_weight_bytes(model->conformer) +
            mimi_decode_plan_compatibility_copied_bytes(model->mimi),
        .load_ns = model->load_ns,
        .load_workers = model->load_workers,
        .load_tasks = model->load_tasks,
        .reserved = {},
    };
    return 0;
}
