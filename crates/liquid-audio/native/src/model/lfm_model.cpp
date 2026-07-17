#include "lfm_model.h"

#include "flashkern_depth.h"
#include "flashkern_rope.h"
#include "lfm_model_plan.h"
#include "lfm_safetensors.h"

#include <atomic>
#include <cerrno>
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

struct View {
    LfmTensorView value{};

    const uint16_t *bf16() const {
        return static_cast<const uint16_t *>(value.data);
    }
};

View tensor(const LfmWeightImage *weights, const std::string &name,
            std::initializer_list<uint64_t> shape) {
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
    return view;
}

bool optional_tensor(const LfmWeightImage *weights, const std::string &name,
                     View *out) {
    out->value.size = sizeof(out->value);
    out->value.abi_version = LFM_WEIGHT_ABI_VERSION;
    return lfm_weights_find(weights, name.c_str(), &out->value) == LFM_WEIGHT_OK;
}

View matrix(const LfmWeightImage *weights, const std::string &name,
            uint64_t columns) {
    View view;
    view.value.size = sizeof(view.value);
    view.value.abi_version = LFM_WEIGHT_ABI_VERSION;
    const int status = lfm_weights_find(weights, name.c_str(), &view.value);
    if (status != LFM_WEIGHT_OK) fail(status, "missing model tensor '" + name + "'");
    if (view.value.dtype != LFM_DTYPE_BF16 || view.value.rank != 2 ||
        view.value.shape[0] == 0 || view.value.shape[1] != columns) {
        fail(-EINVAL, "model tensor '" + name + "' has the wrong matrix shape");
    }
    return view;
}

LfmDepthBufferV1 depth_buffer(const View &view) {
    if (view.value.elements > std::numeric_limits<size_t>::max()) {
        fail(-EOVERFLOW, "depthformer tensor exceeds the native address space");
    }
    return {
        .address = reinterpret_cast<uintptr_t>(view.bf16()),
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
    const double scaled = multiplier * (double)((2 * initial) / 3);
    if (scaled < 0.0 || scaled > (double)std::numeric_limits<size_t>::max()) {
        fail(-EOVERFLOW, "adjusted FFN size is out of range");
    }
    const size_t value = (size_t)scaled;
    return ((value + multiple - 1) / multiple) * multiple;
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

struct LfmModel {
    void *engine = nullptr;
    LfmWeightImage *weights = nullptr;
    uint64_t plan_id = 0;
    uint64_t depth_plan_id = 0;
    uint64_t resident_bytes = 0;
    uint32_t hidden = 0;
    uint32_t ffn = 0;
    uint32_t layers = 0;
    uint32_t vocab = 0;
    uint32_t max_context = 0;
    uint32_t codebooks = 0;
    uint32_t lanes = 0;
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

struct LfmConversation {
    LfmModel *model = nullptr;
    std::vector<ConversationLayer> memory;
    std::vector<LfmLayerState> states;
    std::vector<uint16_t> rope_cos;
    std::vector<uint16_t> rope_sin;
    std::vector<uint16_t> hidden;
    LfmSamplerConfigV1 text_sampler{};
    LfmSamplerConfigV1 audio_sampler{};
    alignas(64) LfmPrngStateV1 initial_prng{};
    alignas(64) LfmPrngStateV1 prng{};
    size_t position = 0;
    bool hidden_ready = false;
    std::atomic_flag active = ATOMIC_FLAG_INIT;
};

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

void build_rope(LfmConversation &conversation) {
    size_t head_dim = 0;
    for (const LfmLayerDesc &layer : conversation.model->descriptors) {
        if (layer.kind != 1) continue;
        if (head_dim != 0 && head_dim != layer.hd) {
            fail(-EINVAL, "native conversation requires one attention head dimension");
        }
        head_dim = layer.hd;
    }
    if (head_dim == 0) return;
    const size_t half = head_dim / 2;
    const size_t count = multiply(conversation.model->max_context, half, "RoPE table");
    if (count > INT_MAX) fail(-EOVERFLOW, "RoPE table exceeds the architecture ABI");
    std::vector<float> cosine(count);
    std::vector<float> sine(count);
    const int status = lfm_rope_table_f32(conversation.model->max_context,
                                          head_dim, conversation.model->rope_theta,
                                          cosine.data(), sine.data());
    if (status != 0) fail(status, "architecture RoPE table kernel rejected geometry");
    conversation.rope_cos.resize(count);
    conversation.rope_sin.resize(count);
    lfm_f32_to_bf16(cosine.data(), conversation.rope_cos.data(), (int)count);
    lfm_f32_to_bf16(sine.data(), conversation.rope_sin.data(), (int)count);
}

void reset_memory(LfmConversation &conversation) {
    for (ConversationLayer &layer : conversation.memory) {
        std::fill(layer.keys.begin(), layer.keys.end(), 0);
        std::fill(layer.values.begin(), layer.values.end(), 0);
        std::fill(layer.convolution.begin(), layer.convolution.end(), 0);
    }
    std::fill(conversation.hidden.begin(), conversation.hidden.end(), 0);
    std::memcpy(&conversation.prng, &conversation.initial_prng,
                sizeof(conversation.prng));
    conversation.position = 0;
    conversation.hidden_ready = false;
}

} // namespace

extern "C" int lfm_model_open(void *engine, const char *path, LfmModel **out,
                              char *error, size_t error_length) {
    if (!engine || !path || !out) return -EINVAL;
    *out = nullptr;
    LfmWeightImage *weights = nullptr;
    uint64_t plan_id = 0;
    uint64_t depth_plan_id = 0;
    std::vector<float> depth_rope_cos;
    std::vector<float> depth_rope_sin;
    try {
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

        char weight_error[512] = {};
        int status = lfm_weights_open(path, &weights, weight_error, sizeof(weight_error));
        if (status != LFM_WEIGHT_OK) {
            fail(status, weight_error[0] ? weight_error : "cannot open model weights");
        }

        std::vector<LfmLayerDesc> descriptors(layers);
        const Json *types = nullptr;
        const auto type_entry = config.find("layer_types");
        if (type_entry != config.end()) {
            if (!type_entry->is_array()) fail(-EINVAL, "lfm.layer_types is not an array");
            types = &*type_entry;
        }

        for (size_t layer = 0; layer < layers; ++layer) {
            const std::string root_name = layer_root(layer);
            LfmLayerDesc &desc = descriptors[layer];
            desc.op_eps = eps;
            desc.ffn_eps = eps;
            desc.op_norm_w = tensor(weights, root_name + "operator_norm.weight", {hidden}).bf16();
            desc.ffn_norm_w = tensor(weights, root_name + "ffn_norm.weight", {hidden}).bf16();
            desc.w1 = tensor(weights, root_name + "feed_forward.w1.weight", {ffn, hidden}).bf16();
            desc.w3 = tensor(weights, root_name + "feed_forward.w3.weight", {ffn, hidden}).bf16();
            desc.w2 = tensor(weights, root_name + "feed_forward.w2.weight", {hidden, ffn}).bf16();

            std::string kind = "full_attention";
            if (types && layer < types->size()) {
                if (!types->at(layer).is_string()) {
                    fail(-EINVAL, "lfm.layer_types contains a non-string entry");
                }
                kind = types->at(layer).get<std::string>();
            }
            if (kind == "conv") {
                desc.kind = 0;
                desc.k = (uint32_t)conv_kernel;
                desc.in_w = tensor(weights, root_name + "conv.in_proj.weight",
                                   {3 * hidden, hidden}).bf16();
                desc.conv_w = tensor(weights, root_name + "conv.conv.weight",
                                     {hidden, 1, conv_kernel}).bf16();
                desc.out_w = tensor(weights, root_name + "conv.out_proj.weight",
                                    {hidden, hidden}).bf16();
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
                              {heads * head_dim, hidden}).bf16();
            desc.k_w = tensor(weights, attention + "k_proj.weight",
                              {kv_heads * head_dim, hidden}).bf16();
            desc.v_w = tensor(weights, attention + "v_proj.weight",
                              {kv_heads * head_dim, hidden}).bf16();
            desc.o_w = tensor(weights, attention + "out_proj.weight",
                              {hidden, heads * head_dim}).bf16();
            desc.qn_w = tensor(weights, attention + "q_layernorm.weight", {head_dim}).bf16();
            desc.kn_w = tensor(weights, attention + "k_layernorm.weight", {head_dim}).bf16();
        }

        status = lfm_ctx_build(engine, descriptors.data(), descriptors.size(), hidden,
                               ffn, max_context, &plan_id);
        if (status != 0) fail(status, "native executor rejected the backbone plan");

        const View text = tensor(weights, "lfm.embed_tokens.weight", {vocab, hidden});
        const View norm = tensor(weights, "lfm.embedding_norm.weight", {hidden});
        View audio;
        const uint16_t *audio_data = nullptr;
        size_t audio_elements = 0;
        size_t audio_rows = 0;
        if (optional_tensor(weights, "audio_embedding.embedding.weight", &audio)) {
            if (audio.value.dtype != LFM_DTYPE_BF16 || audio.value.rank != 2 ||
                audio.value.shape[1] != hidden) {
                fail(-EINVAL, "audio_embedding.embedding.weight has the wrong shape");
            }
            audio_data = audio.bf16();
            audio_elements = (size_t)audio.value.elements;
            audio_rows = (size_t)audio.value.shape[0];
        }
        status = lfm_ctx_set_heads(engine, plan_id, text.bf16(), (size_t)text.value.elements,
                                   vocab, audio_data, audio_elements, audio_rows,
                                   norm.bf16(), (size_t)norm.value.elements, eps);
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
                depth_heads_table[codebook] = {
                    .embedding = depth_buffer(embedding),
                    .norm = depth_buffer(tensor(weights,
                                                root_name + "embedding_norm.weight",
                                                {depth_dim})),
                    .logits = depth_buffer(logits),
                    .vocab = (size_t)embedding.value.shape[0],
                };
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

        std::unique_ptr<LfmModel> model(new (std::nothrow) LfmModel());
        if (!model) fail(-ENOMEM, "cannot allocate native model handle");
        model->engine = engine;
        model->weights = weights;
        model->plan_id = plan_id;
        model->depth_plan_id = depth_plan_id;
        model->resident_bytes = lfm_weights_resident_bytes(weights);
        model->hidden = (uint32_t)hidden;
        model->ffn = (uint32_t)ffn;
        model->layers = (uint32_t)layers;
        model->vocab = (uint32_t)vocab;
        model->max_context = (uint32_t)max_context;
        model->codebooks = codebooks > UINT32_MAX ? 0 : (uint32_t)codebooks;
        model->lanes = lfm_engine_lanes(engine);
        model->rope_theta = rope_theta;
        model->descriptors = std::move(descriptors);
        model->depth_rope_cos = std::move(depth_rope_cos);
        model->depth_rope_sin = std::move(depth_rope_sin);
        weights = nullptr;
        plan_id = 0;
        depth_plan_id = 0;
        *out = model.release();
        return 0;
    } catch (const ModelError &exception) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
        if (weights) lfm_weights_close(weights);
        set_error(error, error_length, exception.what());
        return exception.status();
    } catch (const std::bad_alloc &) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
        if (weights) lfm_weights_close(weights);
        set_error(error, error_length, "native model allocation failed");
        return -ENOMEM;
    } catch (const std::exception &exception) {
        if (depth_plan_id != 0) (void)lfm_engine_depth_clear(engine, depth_plan_id);
        if (plan_id != 0) (void)lfm_ctx_clear(engine, plan_id);
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

        for (size_t index = 0; index < model->descriptors.size(); ++index) {
            const LfmLayerDesc &desc = model->descriptors[index];
            ConversationLayer &memory = conversation->memory[index];
            LfmLayerState &state = conversation->states[index];
            if (desc.kind == 1) {
                const size_t stride = multiply(model->max_context, desc.hd,
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
        build_rope(*conversation);
        const int seed_status = (config->flags & LFM_CONVERSATION_SEED_SYSTEM) != 0
                                    ? lfm_prng_seed_system(&conversation->initial_prng)
                                    : lfm_prng_seed_u64(&conversation->initial_prng,
                                                       config->seed);
        if (seed_status != 0) fail(seed_status, "cannot seed native conversation PRNG");
        reset_memory(*conversation);
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
    if (conversation->position >= conversation->model->max_context) return -ENOSPC;
    uint32_t token = 0;
    const int status = lfm_engine_token_pass(
        conversation->model->engine, conversation->model->plan_id,
        ids, id_count, embedding_kind, conversation->states.data(),
        conversation->states.size(), conversation->position,
        conversation->rope_cos.empty() ? nullptr : conversation->rope_cos.data(),
        conversation->rope_sin.empty() ? nullptr : conversation->rope_sin.data(),
        conversation->rope_cos.size(), conversation->hidden.data(),
        conversation->hidden.size(), nullptr, 0, &conversation->text_sampler,
        &conversation->prng, &token, conversation->model->lanes, nullptr);
    if (status != 0) return status;
    const uint64_t completed_position = conversation->position++;
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
    if (input_count > conversation->model->max_context - conversation->position) {
        return -ENOSPC;
    }
    for (size_t index = 0; index < input_count; ++index) {
        const LfmInputV1 &input = inputs[index];
        if (input.size < sizeof(input) ||
            input.abi_version != LFM_MODEL_ABI_VERSION ||
            input.embedding_kind > 1 || input.id_count == 0 ||
            input.id_count > LFM_INPUT_MAX_IDS ||
            (input.embedding_kind == 0 && input.id_count != 1)) {
            return -EINVAL;
        }
    }
    for (size_t index = 0; index < input_count; ++index) {
        const LfmInputV1 &input = inputs[index];
        const int status = lfm_engine_token_pass(
            conversation->model->engine, conversation->model->plan_id,
            input.ids, input.id_count, input.embedding_kind,
            conversation->states.data(), conversation->states.size(),
            conversation->position,
            conversation->rope_cos.empty() ? nullptr : conversation->rope_cos.data(),
            conversation->rope_sin.empty() ? nullptr : conversation->rope_sin.data(),
            conversation->rope_cos.size(), conversation->hidden.data(),
            conversation->hidden.size(), nullptr, 0, nullptr, nullptr, nullptr,
            conversation->model->lanes, nullptr);
        if (status != 0) return status;
        ++conversation->position;
        conversation->hidden_ready = true;
    }
    *out_position = conversation->position;
    return 0;
}

// Native audio-in prefill: prefill `row_count` continuous embedding rows (the
// Conformer/adapter output, `[row_count, hidden]` bf16) into KV, one native token
// pass per row via the provided-embedding path (`embed_kind == 2`). `rows` is a
// borrowed VIEW into the caller's buffer — no payload crosses the ABI, and the
// backbone reads it in place, exactly as the discrete-id prefill loops. Same
// sequential-per-position shape as `lfm_conversation_prefill`.
extern "C" int lfm_conversation_prefill_audio(LfmConversation *conversation,
                                              const uint16_t *rows, size_t row_count,
                                              uint64_t *out_position) {
    if (!conversation || !rows || row_count == 0 || !out_position) return -EINVAL;
    ConversationClaim claim(conversation);
    if (!claim) return -EBUSY;
    if (row_count > conversation->model->max_context - conversation->position) {
        return -ENOSPC;
    }
    const size_t h = conversation->model->hidden;
    // embed_kind==2 ignores `ids`, but the pass's entry guard requires a non-null
    // id with count >= 1; a single dummy satisfies it without being read.
    static const uint32_t kDummyId = 0;
    for (size_t index = 0; index < row_count; ++index) {
        const int status = lfm_engine_token_pass(
            conversation->model->engine, conversation->model->plan_id,
            &kDummyId, 1, /*embedding_kind=*/2,
            conversation->states.data(), conversation->states.size(),
            conversation->position,
            conversation->rope_cos.empty() ? nullptr : conversation->rope_cos.data(),
            conversation->rope_sin.empty() ? nullptr : conversation->rope_sin.data(),
            conversation->rope_cos.size(), conversation->hidden.data(),
            conversation->hidden.size(), nullptr, 0, nullptr, nullptr, nullptr,
            conversation->model->lanes, rows + index * h);
        if (status != 0) return status;
        ++conversation->position;
        conversation->hidden_ready = true;
    }
    *out_position = conversation->position;
    return 0;
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
    const int status = lfm_engine_depth_frame(
        conversation->model->engine, conversation->model->depth_plan_id,
        conversation->hidden.data(), conversation->hidden.size(),
        &conversation->audio_sampler, &conversation->prng, tokens,
        conversation->model->codebooks);
    if (status != 0) return status;
    *out = {
        .size = sizeof(*out),
        .abi_version = LFM_MODEL_ABI_VERSION,
        .source_position = conversation->position - 1,
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
    reset_memory(*conversation);
    return 0;
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
        .capabilities = model->depth_plan_id != 0 ? LFM_MODEL_CAP_DEPTHFORMER : 0,
        .reserved = {},
    };
    return 0;
}
