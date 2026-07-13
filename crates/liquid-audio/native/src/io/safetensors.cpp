// Native resident safetensors loader.
//
// The span planning and whole-file residency discipline comes from the
// safetensors path in ember-ml. This version deliberately stops before UKM's
// numerical ingress: model payloads remain byte-exact checkpoint storage and
// kernels receive immutable pointers into one process-long aligned image.

#include "lfm_safetensors.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

constexpr size_t kWeightAlign = 64;
constexpr uint64_t kMaxHeaderBytes = 100'000'000;

class WeightError final : public std::runtime_error {
  public:
    WeightError(int status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}

    int status() const { return status_; }

  private:
    int status_;
};

[[noreturn]] void fail(int status, const std::string &message) {
    throw WeightError(status, message);
}

void set_error(char *err, size_t errlen, const char *message) {
    if (!err || errlen == 0) return;
    std::snprintf(err, errlen, "%s", message ? message : "unknown error");
}

size_t checked_align(size_t value) {
    if (value > std::numeric_limits<size_t>::max() - (kWeightAlign - 1)) {
        fail(LFM_WEIGHT_OUT_OF_MEMORY, "weight image size overflows size_t");
    }
    return (value + (kWeightAlign - 1)) & ~(kWeightAlign - 1);
}

size_t checked_add(size_t left, size_t right, const char *what) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        fail(LFM_WEIGHT_OUT_OF_MEMORY, std::string(what) + " overflows size_t");
    }
    return left + right;
}

uint64_t checked_mul(uint64_t left, uint64_t right, const std::string &what) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        fail(LFM_WEIGHT_FORMAT_ERROR, what + " overflows uint64_t");
    }
    return left * right;
}

class AlignedBytes {
  public:
    AlignedBytes() = default;

    explicit AlignedBytes(size_t bytes) : bytes_(bytes) {
        const size_t allocation = bytes == 0 ? kWeightAlign : checked_align(bytes);
        void *memory = nullptr;
        if (posix_memalign(&memory, kWeightAlign, allocation) != 0 || !memory) {
            throw std::bad_alloc();
        }
        data_.reset(static_cast<uint8_t *>(memory));
    }

    uint8_t *data() { return data_.get(); }
    const uint8_t *data() const { return data_.get(); }
    size_t size() const { return bytes_; }

  private:
    struct Free {
        void operator()(uint8_t *data) const { std::free(data); }
    };

    std::unique_ptr<uint8_t, Free> data_;
    size_t bytes_{0};
};

struct Source {
    fs::path path;
    std::string label;
    size_t bytes{0};
    size_t offset{0};
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint64_t offset{0};
    uint64_t elements{0};
    uint64_t bytes{0};
    uint32_t dtype{LFM_DTYPE_INVALID};
    uint32_t shard{0};
};

struct DTypeInfo {
    uint32_t value;
    uint32_t bits;
    const char *name;
};

constexpr DTypeInfo kDTypes[] = {
    {LFM_DTYPE_BOOL, 8, "BOOL"},
    {LFM_DTYPE_F4, 4, "F4"},
    {LFM_DTYPE_F6_E2M3, 6, "F6_E2M3"},
    {LFM_DTYPE_F6_E3M2, 6, "F6_E3M2"},
    {LFM_DTYPE_U8, 8, "U8"},
    {LFM_DTYPE_I8, 8, "I8"},
    {LFM_DTYPE_F8_E5M2, 8, "F8_E5M2"},
    {LFM_DTYPE_F8_E4M3, 8, "F8_E4M3"},
    {LFM_DTYPE_F8_E8M0, 8, "F8_E8M0"},
    {LFM_DTYPE_I16, 16, "I16"},
    {LFM_DTYPE_U16, 16, "U16"},
    {LFM_DTYPE_F16, 16, "F16"},
    {LFM_DTYPE_BF16, 16, "BF16"},
    {LFM_DTYPE_I32, 32, "I32"},
    {LFM_DTYPE_U32, 32, "U32"},
    {LFM_DTYPE_F32, 32, "F32"},
    {LFM_DTYPE_C64, 64, "C64"},
    {LFM_DTYPE_F64, 64, "F64"},
    {LFM_DTYPE_I64, 64, "I64"},
    {LFM_DTYPE_U64, 64, "U64"},
};

const DTypeInfo &dtype_from_name(const std::string &name,
                                 const std::string &tensor) {
    for (const auto &dtype : kDTypes) {
        if (name == dtype.name) return dtype;
    }
    fail(LFM_WEIGHT_FORMAT_ERROR,
         "unsupported safetensors dtype '" + name + "' for tensor '" + tensor + "'");
}

uint64_t json_u64(const Json &value, const std::string &what) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value >= 0) return static_cast<uint64_t>(signed_value);
    }
    fail(LFM_WEIGHT_FORMAT_ERROR, what + " must be a non-negative integer");
}

uint64_t read_le_u64(const uint8_t *data) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(data[i]) << (i * 8);
    }
    return value;
}

size_t weight_file_size(const fs::path &path) {
    std::error_code error;
    const uintmax_t bytes = fs::file_size(path, error);
    if (error) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot stat '" + path.string() + "': " + error.message());
    }
    if (bytes > std::numeric_limits<size_t>::max()) {
        fail(LFM_WEIGHT_OUT_OF_MEMORY,
             "file is too large for this process: '" + path.string() + "'");
    }
    return static_cast<size_t>(bytes);
}

void read_file(const fs::path &path, uint8_t *data, size_t bytes) {
    const std::string native = path.string();
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
        std::fopen(native.c_str(), "rb"), &std::fclose);
    if (!file) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot open '" + native + "': " + std::strerror(errno));
    }

    size_t read = 0;
    while (read < bytes) {
        const size_t count = std::fread(data + read, 1, bytes - read, file.get());
        if (count == 0) {
            if (std::ferror(file.get())) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "read failed for '" + native + "': " + std::strerror(errno));
            }
            fail(LFM_WEIGHT_IO_ERROR,
                 "file shrank while loading: '" + native + "'");
        }
        read += count;
    }
    if (std::fgetc(file.get()) != EOF) {
        fail(LFM_WEIGHT_IO_ERROR,
             "file grew while loading: '" + native + "'");
    }
    if (std::ferror(file.get())) {
        fail(LFM_WEIGHT_IO_ERROR,
             "EOF check failed for '" + native + "': " + std::strerror(errno));
    }
}

std::vector<uint8_t> read_small_file(const fs::path &path) {
    const size_t bytes = weight_file_size(path);
    if (bytes > kMaxHeaderBytes) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "JSON index exceeds the 100 MB safety limit: '" + path.string() + "'");
    }
    std::vector<uint8_t> data(bytes);
    read_file(path, data.data(), data.size());
    return data;
}

Json parse_json(const uint8_t *data, size_t bytes, const std::string &label) {
    try {
        return Json::parse(reinterpret_cast<const char *>(data),
                           reinterpret_cast<const char *>(data + bytes));
    } catch (const std::exception &error) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "invalid JSON in '" + label + "': " + error.what());
    }
}

struct Resolved {
    std::vector<Source> sources;
    std::unordered_map<std::string, std::string> index;
    bool indexed{false};
};

std::string safe_shard_name(const std::string &name) {
    const fs::path path(name);
    if (path.empty() || path.is_absolute()) {
        fail(LFM_WEIGHT_FORMAT_ERROR, "invalid absolute/empty shard path '" + name + "'");
    }
    for (const auto &part : path) {
        if (part == "..") {
            fail(LFM_WEIGHT_FORMAT_ERROR, "shard path escapes checkpoint: '" + name + "'");
        }
    }
    return path.lexically_normal().generic_string();
}

Resolved resolve_index(const fs::path &index_path) {
    const auto bytes = read_small_file(index_path);
    const Json root = parse_json(bytes.data(), bytes.size(), index_path.string());
    if (!root.is_object() || !root.contains("weight_map") ||
        !root.at("weight_map").is_object()) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "checkpoint index has no object weight_map: '" + index_path.string() + "'");
    }

    Resolved resolved;
    resolved.indexed = true;
    std::unordered_set<std::string> shards;
    for (const auto &item : root.at("weight_map").items()) {
        if (!item.value().is_string()) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "checkpoint index maps tensor '" + item.key() + "' to a non-string shard");
        }
        const std::string shard = safe_shard_name(item.value().get<std::string>());
        resolved.index.emplace(item.key(), shard);
        if (shards.insert(shard).second) {
            resolved.sources.push_back({index_path.parent_path() / fs::path(shard), shard});
        }
    }
    if (resolved.index.empty()) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "checkpoint index weight_map is empty: '" + index_path.string() + "'");
    }
    return resolved;
}

Resolved resolve_path(const fs::path &path) {
    std::error_code error;
    if (fs::is_regular_file(path, error)) {
        if (path.filename().string().ends_with(".safetensors.index.json")) {
            return resolve_index(path);
        }
        return Resolved{{Source{path, path.filename().generic_string()}}, {}, false};
    }
    if (error) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot inspect '" + path.string() + "': " + error.message());
    }
    if (!fs::is_directory(path, error)) {
        fail(LFM_WEIGHT_IO_ERROR, "weight path does not exist: '" + path.string() + "'");
    }

    const fs::path index = path / "model.safetensors.index.json";
    if (fs::is_regular_file(index, error)) return resolve_index(index);
    error.clear();

    const fs::path single = path / "model.safetensors";
    if (fs::is_regular_file(single, error)) {
        return Resolved{{Source{single, single.filename().generic_string()}}, {}, false};
    }
    error.clear();

    std::vector<fs::path> shards;
    for (fs::directory_iterator it(path, error), end; !error && it != end; it.increment(error)) {
        const fs::directory_entry &entry = *it;
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors" &&
            name.starts_with("model-")) {
            shards.push_back(entry.path());
        }
    }
    if (error) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot enumerate checkpoint directory '" + path.string() + "': " +
                 error.message());
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) {
        fail(LFM_WEIGHT_IO_ERROR,
             "no model safetensors found in checkpoint directory '" + path.string() + "'");
    }

    Resolved resolved;
    for (const auto &shard : shards) {
        resolved.sources.push_back({shard, shard.filename().generic_string()});
    }
    return resolved;
}

struct PendingTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint64_t start{0};
    uint64_t end{0};
    uint64_t elements{0};
    uint64_t bytes{0};
    uint32_t dtype{LFM_DTYPE_INVALID};
};

} // namespace

struct LfmWeightImage {
    AlignedBytes storage;
    std::vector<Source> sources;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> names;
};

namespace {

void parse_shard(LfmWeightImage &image, uint32_t shard) {
    const Source &source = image.sources.at(shard);
    const uint8_t *file = image.storage.data() + source.offset;
    if (source.bytes < 8) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "safetensors file is shorter than its header prefix: '" + source.path.string() + "'");
    }

    const uint64_t header_bytes = read_le_u64(file);
    if (header_bytes == 0 || header_bytes > kMaxHeaderBytes) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "invalid safetensors header length in '" + source.path.string() + "'");
    }
    if (header_bytes > static_cast<uint64_t>(source.bytes - 8)) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "safetensors header exceeds file bounds in '" + source.path.string() + "'");
    }

    const uint64_t payload_start = 8 + header_bytes;
    const uint64_t payload_bytes = source.bytes - payload_start;
    const Json root = parse_json(file + 8, static_cast<size_t>(header_bytes),
                                 source.path.string());
    if (!root.is_object()) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "safetensors header root is not an object in '" + source.path.string() + "'");
    }

    std::vector<PendingTensor> pending;
    pending.reserve(root.size());
    for (const auto &item : root.items()) {
        if (item.key() == "__metadata__") continue;
        if (item.key().find('\0') != std::string::npos || !item.value().is_object()) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "invalid tensor entry '" + item.key() + "' in '" + source.path.string() + "'");
        }
        const Json &entry = item.value();
        if (!entry.contains("dtype") || !entry.at("dtype").is_string() ||
            !entry.contains("shape") || !entry.at("shape").is_array() ||
            !entry.contains("data_offsets") || !entry.at("data_offsets").is_array() ||
            entry.at("data_offsets").size() != 2) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "tensor '" + item.key() + "' has an incomplete safetensors descriptor");
        }

        PendingTensor tensor;
        tensor.name = item.key();
        const DTypeInfo &dtype =
            dtype_from_name(entry.at("dtype").get<std::string>(), tensor.name);
        tensor.dtype = dtype.value;
        tensor.shape.reserve(entry.at("shape").size());
        tensor.elements = 1;
        for (size_t i = 0; i < entry.at("shape").size(); ++i) {
            const uint64_t dim =
                json_u64(entry.at("shape").at(i), "shape dimension for tensor '" + tensor.name + "'");
            tensor.shape.push_back(dim);
            tensor.elements = checked_mul(tensor.elements, dim,
                                          "element count for tensor '" + tensor.name + "'");
        }
        tensor.start = json_u64(entry.at("data_offsets").at(0),
                                "start offset for tensor '" + tensor.name + "'");
        tensor.end = json_u64(entry.at("data_offsets").at(1),
                              "end offset for tensor '" + tensor.name + "'");
        if (tensor.end < tensor.start || tensor.end > payload_bytes) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "tensor '" + tensor.name + "' has an out-of-bounds data span");
        }

        const uint64_t bits = checked_mul(tensor.elements, dtype.bits,
                                          "bit count for tensor '" + tensor.name + "'");
        if ((bits & 7u) != 0) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "tensor '" + tensor.name + "' has a sub-byte shape that is not byte aligned");
        }
        tensor.bytes = bits / 8;
        if (tensor.end - tensor.start != tensor.bytes) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "tensor '" + tensor.name + "' byte span does not match dtype and shape");
        }
        pending.push_back(std::move(tensor));
    }

    std::sort(pending.begin(), pending.end(), [](const auto &left, const auto &right) {
        return std::pair(left.start, left.end) < std::pair(right.start, right.end);
    });
    uint64_t cursor = 0;
    for (auto &tensor : pending) {
        if (tensor.start != cursor) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "tensor '" + tensor.name + "' leaves a gap or overlaps its predecessor");
        }
        cursor = tensor.end;
        if (image.names.contains(tensor.name)) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "duplicate tensor name across safetensors shards: '" + tensor.name + "'");
        }

        Tensor resident;
        resident.name = std::move(tensor.name);
        resident.shape = std::move(tensor.shape);
        resident.offset = source.offset + payload_start + tensor.start;
        resident.elements = tensor.elements;
        resident.bytes = tensor.bytes;
        resident.dtype = tensor.dtype;
        resident.shard = shard;
        const size_t index = image.tensors.size();
        image.names.emplace(resident.name, index);
        image.tensors.push_back(std::move(resident));
    }
    if (cursor != payload_bytes) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "safetensors payload is not fully described in '" + source.path.string() + "'");
    }
}

std::unique_ptr<LfmWeightImage> load(Resolved resolved) {
    if (resolved.sources.empty()) {
        fail(LFM_WEIGHT_INVALID_ARGUMENT, "no safetensors sources were provided");
    }
    if (resolved.sources.size() > std::numeric_limits<uint32_t>::max()) {
        fail(LFM_WEIGHT_FORMAT_ERROR, "too many safetensors shards");
    }

    size_t total = 0;
    for (auto &source : resolved.sources) {
        source.bytes = weight_file_size(source.path);
        source.offset = checked_align(total);
        total = checked_add(source.offset, source.bytes, "weight image size");
    }

    auto image = std::make_unique<LfmWeightImage>();
    image->storage = AlignedBytes(checked_align(total));
    image->sources = std::move(resolved.sources);
    for (const auto &source : image->sources) {
        read_file(source.path, image->storage.data() + source.offset, source.bytes);
    }
    for (uint32_t shard = 0; shard < image->sources.size(); ++shard) {
        parse_shard(*image, shard);
    }

    if (resolved.indexed) {
        if (resolved.index.size() != image->tensors.size()) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "checkpoint index and loaded shard tensor counts differ");
        }
        for (const auto &tensor : image->tensors) {
            const auto found = resolved.index.find(tensor.name);
            const std::string &source = image->sources.at(tensor.shard).label;
            if (found == resolved.index.end() || found->second != source) {
                fail(LFM_WEIGHT_FORMAT_ERROR,
                     "checkpoint index maps tensor '" + tensor.name + "' to the wrong shard");
            }
        }
    }
    return image;
}

void fill_view(const LfmWeightImage &image, const Tensor &tensor,
               LfmTensorView &view) {
    view = {};
    view.size = sizeof(LfmTensorView);
    view.abi_version = LFM_WEIGHT_ABI_VERSION;
    view.name = tensor.name.c_str();
    view.data = image.storage.data() + tensor.offset;
    view.shape = tensor.shape.data();
    view.offset = tensor.offset;
    view.elements = tensor.elements;
    view.bytes = tensor.bytes;
    view.rank = static_cast<uint32_t>(tensor.shape.size());
    view.dtype = tensor.dtype;
    view.shard = tensor.shard;
}

template <typename Open>
int open_c(Open &&open, LfmWeightImage **out, char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!out) {
        set_error(err, errlen, "null output pointer");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    *out = nullptr;
    try {
        auto image = open();
        *out = image.release();
        return LFM_WEIGHT_OK;
    } catch (const WeightError &error) {
        set_error(err, errlen, error.what());
        return error.status();
    } catch (const std::bad_alloc &) {
        set_error(err, errlen, "out of memory while loading safetensors");
        return LFM_WEIGHT_OUT_OF_MEMORY;
    } catch (const std::exception &error) {
        set_error(err, errlen, error.what());
        return LFM_WEIGHT_FORMAT_ERROR;
    } catch (...) {
        set_error(err, errlen, "unknown native safetensors error");
        return LFM_WEIGHT_FORMAT_ERROR;
    }
}

} // namespace

extern "C" int lfm_weights_open(const char *path, LfmWeightImage **out,
                                char *err, size_t errlen) {
    if (!path || path[0] == '\0') {
        if (out) *out = nullptr;
        set_error(err, errlen, "empty weight path");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    return open_c([&] { return load(resolve_path(fs::path(path))); }, out, err, errlen);
}

extern "C" int lfm_weights_open_files(const char *const *paths, size_t count,
                                      LfmWeightImage **out, char *err,
                                      size_t errlen) {
    if (!paths || count == 0) {
        if (out) *out = nullptr;
        set_error(err, errlen, "empty safetensors file list");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    return open_c(
        [&] {
            Resolved resolved;
            resolved.sources.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                if (!paths[i] || paths[i][0] == '\0') {
                    fail(LFM_WEIGHT_INVALID_ARGUMENT, "empty path in safetensors file list");
                }
                const fs::path path(paths[i]);
                resolved.sources.push_back({path, path.filename().generic_string()});
            }
            return load(std::move(resolved));
        },
        out, err, errlen);
}

extern "C" void lfm_weights_close(LfmWeightImage *image) { delete image; }

extern "C" const void *lfm_weights_data(const LfmWeightImage *image) {
    return image ? image->storage.data() : nullptr;
}

extern "C" uint64_t lfm_weights_resident_bytes(const LfmWeightImage *image) {
    return image ? image->storage.size() : 0;
}

extern "C" size_t lfm_weights_count(const LfmWeightImage *image) {
    return image ? image->tensors.size() : 0;
}

extern "C" int lfm_weights_at(const LfmWeightImage *image, size_t index,
                              LfmTensorView *out) {
    if (!image || !out) return LFM_WEIGHT_INVALID_ARGUMENT;
    if (index >= image->tensors.size()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, image->tensors[index], *out);
    return LFM_WEIGHT_OK;
}

extern "C" int lfm_weights_find(const LfmWeightImage *image, const char *name,
                                LfmTensorView *out) {
    if (!image || !name || !out) return LFM_WEIGHT_INVALID_ARGUMENT;
    const auto found = image->names.find(name);
    if (found == image->names.end()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, image->tensors[found->second], *out);
    return LFM_WEIGHT_OK;
}

extern "C" const char *lfm_weights_dtype_name(uint32_t dtype) {
    for (const auto &info : kDTypes) {
        if (info.value == dtype) return info.name;
    }
    return "INVALID";
}
