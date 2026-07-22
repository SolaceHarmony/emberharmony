// Native resident safetensors loader.
//
// The span planning and whole-file residency discipline comes from the
// safetensors path in ember-ml. This version deliberately stops before UKM's
// numerical ingress: model payloads remain byte-exact checkpoint storage and
// kernels receive immutable pointers into one process-long aligned image.

#include "lfm_safetensors.h"
#include "lfm_payload_reader.h"
#include "kcoro_stackless.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/proc.h>
#include <sys/sysctl.h>
#endif
#endif

#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

/* One layout granule covers Windows section alignment and every supported
 * POSIX page size. It is intentionally much larger than tensor alignment:
 * tensor offsets inside each verbatim safetensors source are never changed. */
constexpr size_t kWeightAlign = 64 * 1024;
constexpr size_t kSegmentHeaderBytes = kWeightAlign;
constexpr uint32_t kSegmentLayoutVersion = 2;
constexpr uint32_t kSegmentInvalid = 0;
constexpr uint32_t kSegmentInitializing = 1;
constexpr uint32_t kSegmentBuilding = 2;
constexpr uint32_t kSegmentReady = 3;
constexpr uint32_t kSegmentPoisoned = 4;
constexpr size_t kMaxSegmentSources = 512;
constexpr size_t kReadChunkBytes = 8 * 1024 * 1024;
constexpr size_t kReadWorkers = 4;
constexpr uint64_t kMaxHeaderBytes = 100'000'000;
static_assert(sizeof(LfmWeightLoadStatsV2) == 176);

using Digest = std::array<uint8_t, 32>;

class Sha256 {
  public:
    Sha256() = default;

    void update(const void *data, size_t bytes) {
        const auto *source = static_cast<const uint8_t *>(data);
        total_ += bytes;
        while (bytes != 0) {
            const size_t count = std::min(bytes, block_.size() - used_);
            std::memcpy(block_.data() + used_, source, count);
            source += count;
            bytes -= count;
            used_ += count;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    Digest finish() const {
        Sha256 copy = *this;
        const uint64_t bits = static_cast<uint64_t>(copy.total_) * 8u;
        const uint8_t one = 0x80;
        copy.update(&one, 1);
        const uint8_t zero = 0;
        while (copy.used_ != 56) copy.update(&zero, 1);
        uint8_t length[8]{};
        for (size_t i = 0; i < sizeof(length); ++i) {
            length[7 - i] = static_cast<uint8_t>(bits >> (i * 8));
        }
        copy.update(length, sizeof(length));

        Digest digest{};
        for (size_t word = 0; word < copy.state_.size(); ++word) {
            for (size_t byte = 0; byte < 4; ++byte) {
                digest[word * 4 + byte] = static_cast<uint8_t>(
                    copy.state_[word] >> ((3 - byte) * 8));
            }
        }
        return digest;
    }

  private:
    static uint32_t rotate(uint32_t value, unsigned bits) {
        return (value >> bits) | (value << (32u - bits));
    }

    void transform(const uint8_t *block) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t words[64]{};
        for (size_t i = 0; i < 16; ++i) {
            words[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                       (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                       static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotate(words[i - 15], 7) ^
                                rotate(words[i - 15], 18) ^
                                (words[i - 15] >> 3);
            const uint32_t s1 = rotate(words[i - 2], 17) ^
                                rotate(words[i - 2], 19) ^
                                (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t sum1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + sum1 + choice + constants[i] + words[i];
            const uint32_t sum0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    std::array<uint8_t, 64> block_{};
    size_t used_{0};
    size_t total_{0};
};

template <typename Integer>
void hash_integer(Sha256 &hash, Integer value) {
    static_assert(std::is_integral_v<Integer>);
    uint8_t bytes[sizeof(Integer)]{};
    using Unsigned = std::make_unsigned_t<Integer>;
    const Unsigned raw = static_cast<Unsigned>(value);
    for (size_t i = 0; i < sizeof(Integer); ++i) {
        bytes[i] = static_cast<uint8_t>(raw >> (i * 8));
    }
    hash.update(bytes, sizeof(bytes));
}

Digest hash_bytes(const void *data, size_t bytes) {
    Sha256 hash;
    hash.update(data, bytes);
    return hash.finish();
}

struct ReadTestHook;

struct LoadOptions {
    size_t workers{kReadWorkers};
    bool uncached{false};
    ReadTestHook *test{nullptr};
};

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

struct Source {
    fs::path path;
    std::string label;
    size_t bytes{0};
    size_t offset{0};
    uint32_t component{LFM_WEIGHT_COMPONENT_MAIN};
};

// Safetensors catalog metadata only. Payload bytes remain owned exclusively by
// the sealed resident image; this record never owns or materializes a tensor.
struct TensorMeta {
    std::string name;
    std::vector<uint64_t> shape;
    uint64_t offset{0};
    uint64_t elements{0};
    uint64_t bytes{0};
    uint32_t dtype{LFM_DTYPE_INVALID};
    uint32_t shard{0};
    uint32_t component{LFM_WEIGHT_COMPONENT_MAIN};
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

struct FileState {
    size_t bytes{0};
    uint64_t identity0{0};
    uint64_t identity1{0};
    int64_t modified_seconds{0};
    int64_t modified_nanos{0};
    int64_t changed_seconds{0};
    int64_t changed_nanos{0};

    bool operator==(const FileState &) const = default;
};

#ifdef _WIN32

using ReadEvent = HANDLE;

std::string system_message(DWORD error) {
    return std::system_category().message(static_cast<int>(error));
}

class OpenFile {
  public:
    explicit OpenFile(fs::path path, bool uncached = false)
        : path_(std::move(path)) {
        (void)uncached;
        handle_ = CreateFileW(path_.c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            const DWORD error = GetLastError();
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot open '" + path_.string() + "': " + system_message(error));
        }
        try {
            initial_ = state();
        } catch (...) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
            throw;
        }
    }

    ~OpenFile() {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    }

    OpenFile(const OpenFile &) = delete;
    OpenFile &operator=(const OpenFile &) = delete;

    OpenFile(OpenFile &&other) noexcept
        : path_(std::move(other.path_)), handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
          initial_(other.initial_) {}

    OpenFile &operator=(OpenFile &&) = delete;

    size_t bytes() const { return initial_.bytes; }
    const FileState &identity() const { return initial_; }

    void read_at(uint8_t *data, size_t bytes, size_t offset, ReadEvent event) const {
        size_t done = 0;
        while (done < bytes) {
            const uint64_t absolute = static_cast<uint64_t>(offset) + done;
            OVERLAPPED overlap{};
            overlap.Offset = static_cast<DWORD>(absolute);
            overlap.OffsetHigh = static_cast<DWORD>(absolute >> 32);
            overlap.hEvent = event;
            if (!ResetEvent(event)) {
                const DWORD error = GetLastError();
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot reset read event for '" + path_.string() + "': " +
                         system_message(error));
            }

            const size_t remaining = bytes - done;
            const DWORD request = static_cast<DWORD>(
                std::min<size_t>(remaining, std::numeric_limits<DWORD>::max()));
            DWORD count = 0;
            if (!ReadFile(handle_, data + done, request, &count, &overlap)) {
                const DWORD error = GetLastError();
                if (error != ERROR_IO_PENDING ||
                    !GetOverlappedResult(handle_, &overlap, &count, TRUE)) {
                    const DWORD result = error == ERROR_IO_PENDING ? GetLastError() : error;
                    fail(LFM_WEIGHT_IO_ERROR,
                         "positioned read failed for '" + path_.string() + "': " +
                             system_message(result));
                }
            }
            if (count == 0) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "file shrank while loading: '" + path_.string() + "'");
            }
            done += count;
        }
    }

    void verify() const {
        if (state() == initial_) return;
        fail(LFM_WEIGHT_IO_ERROR,
             "file changed while loading: '" + path_.string() + "'");
    }

  private:
    FileState state() const {
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(handle_, &info)) {
            const DWORD error = GetLastError();
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot stat open file '" + path_.string() + "': " +
                     system_message(error));
        }
        ULARGE_INTEGER bytes{};
        bytes.LowPart = info.nFileSizeLow;
        bytes.HighPart = info.nFileSizeHigh;
        if (bytes.QuadPart > std::numeric_limits<size_t>::max()) {
            fail(LFM_WEIGHT_OUT_OF_MEMORY,
                 "file is too large for this process: '" + path_.string() + "'");
        }
        ULARGE_INTEGER index{};
        index.LowPart = info.nFileIndexLow;
        index.HighPart = info.nFileIndexHigh;
        ULARGE_INTEGER modified{};
        modified.LowPart = info.ftLastWriteTime.dwLowDateTime;
        modified.HighPart = info.ftLastWriteTime.dwHighDateTime;
        return {static_cast<size_t>(bytes.QuadPart), info.dwVolumeSerialNumber,
                index.QuadPart, static_cast<int64_t>(modified.QuadPart), 0, 0, 0};
    }

    fs::path path_;
    HANDLE handle_{INVALID_HANDLE_VALUE};
    FileState initial_;
};

#else

using ReadEvent = int;

class OpenFile {
  public:
    explicit OpenFile(fs::path path, bool uncached = false)
        : path_(std::move(path)) {
        int flags = O_RDONLY;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        handle_ = ::open(path_.c_str(), flags);
        if (handle_ < 0) {
            const int error = errno;
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot open '" + path_.string() + "': " + std::strerror(error));
        }
        try {
            initial_ = state();
#ifdef __APPLE__
            if (uncached && ::fcntl(handle_, F_NOCACHE, 1) != 0) {
                const int error = errno;
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot disable file caching for benchmark source '" +
                         path_.string() + "': " + std::strerror(error));
            }
#elif defined(POSIX_FADV_DONTNEED)
            if (uncached) {
                const int error = ::posix_fadvise(handle_, 0, 0, POSIX_FADV_DONTNEED);
                if (error != 0) {
                    fail(LFM_WEIGHT_IO_ERROR,
                         "cannot evict benchmark source '" + path_.string() +
                             "' from the file cache: " + std::strerror(error));
                }
            }
#else
            if (uncached) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "cold-cache loader benchmarking is unsupported on this platform");
            }
#endif
        } catch (...) {
            ::close(handle_);
            handle_ = -1;
            throw;
        }
    }

    ~OpenFile() {
        if (handle_ >= 0) ::close(handle_);
    }

    OpenFile(const OpenFile &) = delete;
    OpenFile &operator=(const OpenFile &) = delete;

    OpenFile(OpenFile &&other) noexcept
        : path_(std::move(other.path_)), handle_(std::exchange(other.handle_, -1)),
          initial_(other.initial_) {}

    OpenFile &operator=(OpenFile &&) = delete;

    size_t bytes() const { return initial_.bytes; }
    const FileState &identity() const { return initial_; }

    void read_at(uint8_t *data, size_t bytes, size_t offset, ReadEvent) const {
        size_t done = 0;
        while (done < bytes) {
            const size_t absolute = checked_add(offset, done, "file read offset");
            if (absolute > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "file offset exceeds off_t for '" + path_.string() + "'");
            }
            const ssize_t count = ::pread(handle_, data + done, bytes - done,
                                           static_cast<off_t>(absolute));
            if (count < 0) {
                const int error = errno;
                if (error == EINTR) continue;
                fail(LFM_WEIGHT_IO_ERROR,
                     "positioned read failed for '" + path_.string() + "': " +
                         std::strerror(error));
            }
            if (count == 0) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "file shrank while loading: '" + path_.string() + "'");
            }
            done += static_cast<size_t>(count);
        }
    }

    void verify() const {
        if (state() == initial_) return;
        fail(LFM_WEIGHT_IO_ERROR,
             "file changed while loading: '" + path_.string() + "'");
    }

  private:
    FileState state() const {
        struct stat info {};
        if (::fstat(handle_, &info) != 0) {
            const int error = errno;
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot stat open file '" + path_.string() + "': " +
                     std::strerror(error));
        }
        if (!S_ISREG(info.st_mode)) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "weight source is not a regular file: '" + path_.string() + "'");
        }
        if (info.st_size < 0 ||
            static_cast<uintmax_t>(info.st_size) > std::numeric_limits<size_t>::max()) {
            fail(LFM_WEIGHT_OUT_OF_MEMORY,
                 "file is too large for this process: '" + path_.string() + "'");
        }
#ifdef __APPLE__
        const timespec modified = info.st_mtimespec;
        const timespec changed = info.st_ctimespec;
#else
        const timespec modified = info.st_mtim;
        const timespec changed = info.st_ctim;
#endif
        return {static_cast<size_t>(info.st_size), static_cast<uint64_t>(info.st_dev),
                static_cast<uint64_t>(info.st_ino), modified.tv_sec, modified.tv_nsec,
                changed.tv_sec, changed.tv_nsec};
    }

    fs::path path_;
    int handle_{-1};
    FileState initial_;
};

#endif

struct SegmentSourceRecord {
    uint64_t offset{0};
    uint64_t bytes{0};
    uint32_t component{0};
    uint32_t reserved0{0};
    uint8_t label_digest[32]{};
    uint64_t reserved1{0};
};
static_assert(sizeof(SegmentSourceRecord) == 64);

struct alignas(64) SegmentHeader {
    uint8_t magic[8]{};
    uint32_t layout_version{0};
    uint32_t header_bytes{0};
    uint32_t state{0};
    uint32_t source_count{0};
    uint64_t total_bytes{0};
    uint64_t source_bytes{0};
    uint64_t generation{0};
    uint64_t owner_pid{0};
    uint64_t owner_start_time{0};
    uint64_t owner_uid{0};
    uint64_t build_ns{0};
    uint32_t build_tasks{0};
    uint32_t build_workers{0};
    uint8_t identity_digest[32]{};
    uint8_t content_digest[32]{};
    SegmentSourceRecord sources[kMaxSegmentSources]{};
};
static_assert(sizeof(SegmentHeader) <= kSegmentHeaderBytes);
static_assert(offsetof(SegmentHeader, state) % alignof(uint32_t) == 0);
static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr));

constexpr uint8_t kSegmentMagic[8] = {'L', 'F', 'M', 'W', 'S', 'E', 'G', '2'};

uint32_t segment_state(const SegmentHeader *header) {
    return __atomic_load_n(&header->state, __ATOMIC_ACQUIRE);
}

void publish_segment_state(SegmentHeader *header, uint32_t state) {
    __atomic_store_n(&header->state, state, __ATOMIC_RELEASE);
}

bool digest_empty(const uint8_t digest[32]) {
    uint8_t value = 0;
    for (size_t i = 0; i < 32; ++i) value |= digest[i];
    return value == 0;
}

std::string digest_hex(const uint8_t digest[32], size_t bytes = 32) {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::string text(bytes * 2, '0');
    for (size_t i = 0; i < bytes; ++i) {
        text[i * 2] = alphabet[digest[i] >> 4];
        text[i * 2 + 1] = alphabet[digest[i] & 15];
    }
    return text;
}

std::string segment_name(const Digest &identity) {
    return "/lfm-" + digest_hex(identity.data(), 12);
}

Digest label_digest(std::string_view label) {
    return hash_bytes(label.data(), label.size());
}

Digest identity_digest(const std::vector<Source> &sources,
                       const std::vector<OpenFile> &files) {
    static constexpr char domain[] = "LFM-WEIGHT-IDENTITY-V1";
    Sha256 hash;
    hash.update(domain, sizeof(domain) - 1);
    hash_integer(hash, kSegmentLayoutVersion);
    hash_integer(hash, static_cast<uint64_t>(sources.size()));
    for (size_t index = 0; index < sources.size(); ++index) {
        const Source &source = sources[index];
        const FileState &state = files[index].identity();
        const Digest label = label_digest(source.label);
        hash_integer(hash, source.component);
        hash.update(label.data(), label.size());
        hash_integer(hash, static_cast<uint64_t>(state.bytes));
        hash_integer(hash, state.identity0);
        hash_integer(hash, state.identity1);
        hash_integer(hash, state.modified_seconds);
        hash_integer(hash, state.modified_nanos);
        hash_integer(hash, state.changed_seconds);
        hash_integer(hash, state.changed_nanos);
    }
    return hash.finish();
}

uint64_t current_pid();
uint64_t current_uid();

uint64_t segment_generation() {
    uint64_t generation = static_cast<uint64_t>(
                              std::chrono::steady_clock::now()
                                  .time_since_epoch()
                                  .count()) ^
                          (current_pid() << 17);
    return generation == 0 ? 1 : generation;
}

void initialize_segment_owner(SegmentHeader *header, uint64_t generation,
                              uint64_t owner_pid, uint64_t owner_start,
                              uint64_t owner_uid) {
    std::memset(header, 0, kSegmentHeaderBytes);
    header->generation = generation;
    header->owner_pid = owner_pid;
    header->owner_start_time = owner_start;
    header->owner_uid = owner_uid;
    publish_segment_state(header, kSegmentInitializing);
}

void publish_segment_build_header(SegmentHeader *header,
                                  const std::vector<Source> &sources,
                                  size_t bytes, size_t source_bytes,
                                  const Digest &identity) {
    std::memcpy(header->magic, kSegmentMagic, sizeof(kSegmentMagic));
    header->layout_version = kSegmentLayoutVersion;
    header->header_bytes = static_cast<uint32_t>(kSegmentHeaderBytes);
    header->source_count = static_cast<uint32_t>(sources.size());
    header->total_bytes = bytes;
    header->source_bytes = source_bytes;
    std::memcpy(header->identity_digest, identity.data(), identity.size());
    for (size_t index = 0; index < sources.size(); ++index) {
        SegmentSourceRecord &record = header->sources[index];
        const Source &source = sources[index];
        const Digest label = label_digest(source.label);
        record.offset = source.offset;
        record.bytes = source.bytes;
        record.component = source.component;
        std::memcpy(record.label_digest, label.data(), label.size());
    }
    publish_segment_state(header, kSegmentBuilding);
}

uint64_t process_start_time(uint64_t pid) {
#ifdef _WIN32
    const DWORD access = PROCESS_QUERY_LIMITED_INFORMATION;
    HANDLE process = OpenProcess(access, FALSE, static_cast<DWORD>(pid));
    if (!process) return 0;
    FILETIME created{}, exited{}, kernel{}, user{};
    const BOOL ok = GetProcessTimes(process, &created, &exited, &kernel, &user);
    CloseHandle(process);
    if (!ok) return 0;
    ULARGE_INTEGER value{};
    value.LowPart = created.dwLowDateTime;
    value.HighPart = created.dwHighDateTime;
    return value.QuadPart;
#elif defined(__APPLE__)
    if (pid > static_cast<uint64_t>(std::numeric_limits<int>::max())) return 0;
    int query[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, static_cast<int>(pid)};
    kinfo_proc info{};
    size_t bytes = sizeof(info);
    if (sysctl(query, 4, &info, &bytes, nullptr, 0) != 0 || bytes == 0) return 0;
    const timeval started = info.kp_proc.p_starttime;
    return static_cast<uint64_t>(started.tv_sec) * 1'000'000u +
           static_cast<uint64_t>(started.tv_usec);
#else
    const std::string path = "/proc/" + std::to_string(pid) + "/stat";
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
        std::fopen(path.c_str(), "r"), &std::fclose);
    if (!file) return 0;
    char line[4096]{};
    if (!std::fgets(line, sizeof(line), file.get())) return 0;
    char *field = std::strrchr(line, ')');
    if (!field || field[1] != ' ') return 0;
    field += 2;
    for (unsigned number = 3; number <= 22; ++number) {
        char *end = field;
        while (*end != '\0' && *end != ' ') ++end;
        if (number == 22) {
            const char saved = *end;
            *end = '\0';
            char *parsed = nullptr;
            const unsigned long long value = std::strtoull(field, &parsed, 10);
            *end = saved;
            return parsed == field ? 0 : static_cast<uint64_t>(value);
        }
        if (*end == '\0') return 0;
        field = end + 1;
    }
    return 0;
#endif
}

uint64_t current_pid() {
#ifdef _WIN32
    return static_cast<uint64_t>(GetCurrentProcessId());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

uint64_t current_uid() {
#ifdef _WIN32
    return 0;
#else
    return static_cast<uint64_t>(geteuid());
#endif
}

bool owner_alive(uint64_t pid, uint64_t started) {
    if (pid == 0 || started == 0) return false;
#ifndef _WIN32
    if (pid > static_cast<uint64_t>(std::numeric_limits<pid_t>::max())) return false;
    if (kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH) return false;
#endif
    return process_start_time(pid) == started;
}

#ifndef _WIN32
bool poison_abandoned_segment(const std::string &name, uint64_t generation,
                              uint64_t owner, uint64_t started,
                              uint64_t owner_uid) {
    const int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) return false;
    struct stat info {};
    if (fstat(fd, &info) != 0 || info.st_uid != geteuid() ||
        info.st_size < static_cast<off_t>(kSegmentHeaderBytes)) {
        (void)::close(fd);
        return false;
    }
    void *memory = mmap(nullptr, kSegmentHeaderBytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (memory == MAP_FAILED) {
        (void)::close(fd);
        return false;
    }
    auto *candidate = static_cast<SegmentHeader *>(memory);
    uint32_t expected = segment_state(candidate);
    const bool mutable_state = expected == kSegmentInitializing ||
                               expected == kSegmentBuilding;
    const bool same_owner = candidate->generation == generation &&
                            candidate->owner_pid == owner &&
                            candidate->owner_start_time == started &&
                            candidate->owner_uid == owner_uid;
    const bool poisoned = mutable_state && same_owner &&
                          !owner_alive(owner, started) &&
                          __atomic_compare_exchange_n(
                              &candidate->state, &expected, kSegmentPoisoned,
                              false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    if (poisoned) (void)shm_unlink(name.c_str());
    (void)munmap(memory, kSegmentHeaderBytes);
    (void)::close(fd);
    return poisoned;
}
#endif

std::string wire_failure(size_t bytes, int error) {
#ifdef _WIN32
    return "cannot wire the shared weight segment (" + std::to_string(bytes) +
           " bytes): " + system_message(static_cast<DWORD>(error)) +
           ". Increase the process working-set quota; unwired model operation is forbidden";
#else
    rlimit limit{};
    const bool have_limit = getrlimit(RLIMIT_MEMLOCK, &limit) == 0;
    std::string message = "cannot mlock the shared weight segment (" +
                          std::to_string(bytes) + " bytes): " +
                          std::strerror(error) + ". ";
    if (have_limit && limit.rlim_cur != RLIM_INFINITY) {
        message += "RLIMIT_MEMLOCK is " + std::to_string(limit.rlim_cur) +
                   " bytes; raise it (for example, `ulimit -l unlimited`). ";
    }
#ifdef __APPLE__
    message += "Also raise macOS vm.user_wire_limit and vm.global_user_wire_limit "
               "above the requested byte count. ";
#else
    message += "Raise RLIMIT_MEMLOCK or grant CAP_IPC_LOCK. ";
#endif
    return message + "Unwired model operation is forbidden";
#endif
}

struct ReadyTarget {
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
};

struct ReadySubscriber {
    koro_cont_t *continuation{nullptr};
    kc_ticket_id identity{};
};

bool ticket_equal(const kc_ticket_id &left, const kc_ticket_id &right) {
    return left.runtime_epoch == right.runtime_epoch &&
           left.sequence == right.sequence &&
           left.generation == right.generation && left.kind == right.kind;
}

class WeightSegment {
  public:
    WeightSegment() = default;
    WeightSegment(const WeightSegment &) = delete;
    WeightSegment &operator=(const WeightSegment &) = delete;
    WeightSegment(WeightSegment &&other) noexcept { swap(other); }
    WeightSegment &operator=(WeightSegment &&other) noexcept {
        if (this != &other) {
            WeightSegment empty;
            swap(empty);
            swap(other);
        }
        return *this;
    }

    ~WeightSegment() {
        if (creator_ && data_ && !published_) {
            publish_segment_state(header(), kSegmentPoisoned);
#ifndef _WIN32
            if (!name_.empty()) (void)shm_unlink(name_.c_str());
#endif
        }
        release();
    }

    static WeightSegment acquire(const std::vector<Source> &sources,
                                 const std::vector<OpenFile> &files,
                                 size_t bytes, size_t source_bytes,
                                 const Digest &identity,
                                 bool inject_wire_failure = false,
                                 bool takeover = true) {
        if (sources.size() > kMaxSegmentSources) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "shared weight segment source count exceeds header capacity");
        }
        WeightSegment segment;
        segment.name_ = segment_name(identity);
        segment.identity_ = identity;
        segment.bytes_ = bytes;
        const auto begin = std::chrono::steady_clock::now();
#ifdef _WIN32
        std::wstring name(segment.name_.begin() + 1, segment.name_.end());
        name.insert(0, L"Local\\");
        segment.mapping_ = CreateFileMappingW(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
            static_cast<DWORD>(static_cast<uint64_t>(bytes) >> 32),
            static_cast<DWORD>(bytes), name.c_str());
        if (!segment.mapping_) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot create/open shared weight section: " +
                     system_message(GetLastError()));
        }
        const bool created = GetLastError() != ERROR_ALREADY_EXISTS;
        segment.creator_ = created;
        const DWORD access = created ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
        segment.data_ = static_cast<uint8_t *>(
            MapViewOfFile(segment.mapping_, access, 0, 0, bytes));
        if (!segment.data_) {
            const DWORD error = GetLastError();
            segment.release();
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot map shared weight section: " + system_message(error));
        }
#else
        const int flags = O_RDWR | O_CREAT | O_EXCL;
        segment.fd_ = shm_open(segment.name_.c_str(), flags, 0600);
        if (segment.fd_ >= 0) {
#ifdef FD_CLOEXEC
            (void)fcntl(segment.fd_, F_SETFD, FD_CLOEXEC);
#endif
            segment.creator_ = true;
            if (ftruncate(segment.fd_, static_cast<off_t>(bytes)) != 0) {
                const int error = errno;
                (void)shm_unlink(segment.name_.c_str());
                segment.release();
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot size shared weight segment '" + segment.name_ +
                         "': " + std::strerror(error));
            }
            void *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, segment.fd_, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                (void)shm_unlink(segment.name_.c_str());
                segment.data_ = nullptr;
                segment.release();
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot map new shared weight segment '" + segment.name_ +
                         "': " + std::strerror(error));
            }
            segment.data_ = static_cast<uint8_t *>(mapping);
        } else {
            const int create_error = errno;
            if (create_error != EEXIST) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot elect shared weight builder for '" + segment.name_ +
                         "': " + std::strerror(create_error));
            }
            segment.fd_ = shm_open(segment.name_.c_str(), O_RDONLY, 0);
            if (segment.fd_ < 0) {
                const int error = errno;
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot attach shared weight segment '" + segment.name_ +
                         "': " + std::strerror(error));
            }
#ifdef FD_CLOEXEC
            (void)fcntl(segment.fd_, F_SETFD, FD_CLOEXEC);
#endif
            struct stat info {};
            if (fstat(segment.fd_, &info) != 0) {
                const int error = errno;
                segment.release();
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot stat shared weight segment '" + segment.name_ +
                         "': " + std::strerror(error));
            }
            if (info.st_uid != geteuid()) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "shared weight segment owner uid does not match this process");
            }
            if (info.st_size == 0) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "same-name shared weight object has no published storage");
            }
            if (info.st_size < 0 || static_cast<uint64_t>(info.st_size) != bytes) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "shared weight segment size does not match checkpoint layout");
            }
            void *mapping = mmap(nullptr, bytes, PROT_READ, MAP_SHARED,
                                 segment.fd_, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                segment.data_ = nullptr;
                segment.release();
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot map shared weight segment read-only: " +
                         std::string(std::strerror(error)));
            }
            segment.data_ = static_cast<uint8_t *>(mapping);
        }
#endif

        if (segment.creator_) {
            SegmentHeader *header = segment.header();
            const uint64_t pid = current_pid();
            const uint64_t started = process_start_time(pid);
            if (started == 0) {
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot establish shared weight builder process identity");
            }
            initialize_segment_owner(header, segment_generation(), pid,
                                     started, current_uid());
            publish_segment_build_header(header, sources, bytes, source_bytes,
                                         identity);
        } else {
            const SegmentHeader *header = segment.header();
            const uint32_t state = segment_state(header);
            if (state == kSegmentInvalid) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "same-name shared weight object has no published "
                     "initializer identity");
            }
            if (state != kSegmentInitializing && state != kSegmentBuilding &&
                state != kSegmentReady && state != kSegmentPoisoned) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "same-name shared weight object has an unknown "
                     "lifecycle state");
            }
            const uint64_t generation = header->generation;
            const uint64_t owner = header->owner_pid;
            const uint64_t started = header->owner_start_time;
            const uint64_t owner_uid = header->owner_uid;
            const bool owner_valid = generation != 0 && owner != 0 &&
                                     started != 0 && owner_uid == current_uid();
            if (!owner_valid) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "same-name shared weight object failed its initializer "
                     "identity");
            }
            if (state == kSegmentInitializing) {
                segment.release();
#ifndef _WIN32
                if (takeover && !owner_alive(owner, started)) {
                    (void)poison_abandoned_segment(
                        segment.name_, generation, owner, started, owner_uid);
                    return acquire(sources, files, bytes, source_bytes,
                                   identity, inject_wire_failure, false);
                }
#else
                if (!owner_alive(owner, started)) {
                    fail(LFM_WEIGHT_REJECTED,
                         "shared weight initializer died and takeover is unsupported");
                }
#endif
                fail(LFM_WEIGHT_IN_PROGRESS,
                     "shared weight segment is INITIALIZING under a live owner; "
                     "resume this open from its readiness callback");
            }
            const bool header_valid =
                std::memcmp(header->magic, kSegmentMagic,
                            sizeof(kSegmentMagic)) == 0 &&
                header->layout_version == kSegmentLayoutVersion &&
                header->header_bytes == kSegmentHeaderBytes &&
                header->total_bytes == bytes &&
                header->source_bytes == source_bytes &&
                header->source_count == sources.size() &&
                header->generation == generation &&
                header->owner_pid == owner &&
                header->owner_start_time == started &&
                header->owner_uid == owner_uid &&
                std::memcmp(header->identity_digest, identity.data(),
                            identity.size()) == 0;
            if (!header_valid) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "same-name shared weight object failed its identity/layout header");
            }
            for (size_t index = 0; index < sources.size(); ++index) {
                const SegmentSourceRecord &record = header->sources[index];
                const Source &source = sources[index];
                const Digest label = label_digest(source.label);
                if (record.offset != source.offset ||
                    record.bytes != source.bytes ||
                    record.component != source.component ||
                    std::memcmp(record.label_digest, label.data(), label.size()) != 0) {
                    segment.release();
                    fail(LFM_WEIGHT_REJECTED,
                         "shared weight source table does not match checkpoint layout");
                }
            }
            if (state == kSegmentBuilding) {
                segment.release();
#ifndef _WIN32
                if (takeover && !owner_alive(owner, started)) {
                    (void)poison_abandoned_segment(
                        segment.name_, generation, owner, started, owner_uid);
                    return acquire(sources, files, bytes, source_bytes,
                                   identity, inject_wire_failure, false);
                }
#else
                if (!owner_alive(owner, started)) {
                    fail(LFM_WEIGHT_REJECTED,
                         "shared weight builder died and takeover is unsupported");
                }
#endif
                fail(LFM_WEIGHT_IN_PROGRESS,
                     "shared weight segment is BUILDING under a live owner; "
                     "resume this open from its readiness callback");
            }
            if (state == kSegmentPoisoned) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "shared weight segment generation is POISONED; evict it explicitly");
            }
            if (state != kSegmentReady || digest_empty(header->content_digest)) {
                segment.release();
                fail(LFM_WEIGHT_REJECTED,
                     "shared weight segment has no valid READY publication");
            }
            segment.attached_ = true;
            segment.published_ = true;
        }

        if (inject_wire_failure) {
            fail(LFM_WEIGHT_IO_ERROR, wire_failure(bytes, ENOMEM));
        }
        segment.wire();
        const auto end = std::chrono::steady_clock::now();
        const uint64_t elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        if (segment.attached_) segment.attach_ns_ = elapsed;
        return segment;
    }

    uint8_t *mutable_data() {
        if (!creator_ || published_) {
            fail(LFM_WEIGHT_REJECTED,
                 "only an unpublished builder may mutate a weight segment");
        }
        return data_;
    }
    const uint8_t *data() const { return data_; }
    size_t size() const { return bytes_; }
    bool creator() const { return creator_; }
    bool attached() const { return attached_; }
    bool wired() const { return wired_; }
    uint64_t attach_ns() const { return attach_ns_; }
    const Digest &identity() const { return identity_; }
    const std::string &name() const { return name_; }
    const SegmentHeader *published_header() const { return header(); }

    void publish(const Digest &content, uint64_t build_ns, uint32_t tasks,
                 uint32_t workers) {
        if (!creator_ || published_) {
            fail(LFM_WEIGHT_REJECTED,
                 "invalid shared weight publication transition");
        }
        SegmentHeader *value = header();
        value->build_ns = build_ns;
        value->build_tasks = tasks;
        value->build_workers = workers;
        std::memcpy(value->content_digest, content.data(), content.size());
        publish_segment_state(value, kSegmentReady);
#ifdef _WIN32
        DWORD previous = 0;
        if (!VirtualProtect(data_, bytes_, PAGE_READONLY, &previous)) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot publish shared weight mapping read-only: " +
                     system_message(GetLastError()));
        }
#else
        if (mprotect(data_, bytes_, PROT_READ) != 0) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot publish shared weight mapping read-only: " +
                     std::string(std::strerror(errno)));
        }
#endif
        published_ = true;
    }

  private:
    SegmentHeader *header() {
        return reinterpret_cast<SegmentHeader *>(data_);
    }
    const SegmentHeader *header() const {
        return reinterpret_cast<const SegmentHeader *>(data_);
    }

    void wire() {
        if (wired_) return;
#ifdef _WIN32
        if (!VirtualLock(data_, bytes_)) {
            fail(LFM_WEIGHT_IO_ERROR,
                 wire_failure(bytes_, static_cast<int>(GetLastError())));
        }
#else
        if (mlock(data_, bytes_) != 0) {
            const int error = errno;
            fail(LFM_WEIGHT_IO_ERROR, wire_failure(bytes_, error));
        }
#endif
        wired_ = true;
    }

    void release() noexcept {
        if (data_) {
            if (wired_) {
#ifdef _WIN32
                (void)VirtualUnlock(data_, bytes_);
#else
                (void)munlock(data_, bytes_);
#endif
            }
#ifdef _WIN32
            (void)UnmapViewOfFile(data_);
#else
            (void)munmap(data_, bytes_);
#endif
        }
        data_ = nullptr;
        wired_ = false;
#ifdef _WIN32
        if (mapping_) CloseHandle(mapping_);
        mapping_ = nullptr;
#else
        if (fd_ >= 0) (void)::close(fd_);
        fd_ = -1;
#endif
    }

    void swap(WeightSegment &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(bytes_, other.bytes_);
        std::swap(name_, other.name_);
        std::swap(identity_, other.identity_);
        std::swap(creator_, other.creator_);
        std::swap(attached_, other.attached_);
        std::swap(published_, other.published_);
        std::swap(wired_, other.wired_);
        std::swap(attach_ns_, other.attach_ns_);
#ifdef _WIN32
        std::swap(mapping_, other.mapping_);
#else
        std::swap(fd_, other.fd_);
#endif
    }

    uint8_t *data_{nullptr};
    size_t bytes_{0};
    std::string name_;
    Digest identity_{};
    bool creator_{false};
    bool attached_{false};
    bool published_{false};
    bool wired_{false};
    uint64_t attach_ns_{0};
#ifdef _WIN32
    HANDLE mapping_{nullptr};
#else
    int fd_{-1};
#endif
};

struct ReadTask {
    size_t source{0};
    size_t offset{0};
    size_t bytes{0};
    uint8_t *destination{nullptr};
    Digest digest{};
    std::exception_ptr error;
};

struct ReadSummary {
    uint32_t tasks{0};
    uint32_t workers{0};
    Digest content_digest{};
};

/* Private deterministic fault injection for the loader integration test. It
 * is reachable only through the unadvertised test entry point below; product
 * opens always pass null and retain exactly the production read path. */
struct ReadTestHook {
    size_t fail_task{std::numeric_limits<size_t>::max()};
    fs::path change_after_reads;
    std::atomic<size_t> completed{0};
    size_t scheduled{0};
    bool fail_wire{false};
};

ReadSummary read_sources(const std::vector<OpenFile> &files, uint8_t *image,
                         const std::vector<Source> &sources,
                         size_t image_bytes, size_t worker_limit,
                         ReadTestHook *test,
                         const LfmPayloadReadScope *accounting = nullptr) {
    std::vector<ReadTask> tasks;
    for (size_t source = 0; source < sources.size(); ++source) {
        for (size_t offset = 0; offset < sources[source].bytes;) {
            const size_t bytes = std::min(kReadChunkBytes, sources[source].bytes - offset);
            tasks.push_back({source, offset, bytes,
                             image + sources[source].offset + offset, {}, {}});
            offset += bytes;
        }
    }
    if (tasks.size() > std::numeric_limits<uint32_t>::max()) {
        fail(LFM_WEIGHT_OUT_OF_MEMORY, "safetensors read task count exceeds uint32_t");
    }
    if (test) test->scheduled = tasks.size();

    std::atomic<size_t> next{0};
    const size_t count = std::min(worker_limit, tasks.size());

#ifdef _WIN32
    std::vector<ReadEvent> events;
    events.reserve(count);
    for (size_t worker = 0; worker < count; ++worker) {
        const HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!event) {
            const DWORD error = GetLastError();
            for (const HANDLE opened : events) CloseHandle(opened);
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot create positioned-read event: " + system_message(error));
        }
        try {
            events.push_back(event);
        } catch (...) {
            CloseHandle(event);
            for (const HANDLE opened : events) CloseHandle(opened);
            throw;
        }
    }
#endif

    try {
        std::vector<std::jthread> workers;
        workers.reserve(count);
        for (size_t worker = 0; worker < count; ++worker) {
            workers.emplace_back([&, worker] {
#ifdef _WIN32
                const ReadEvent event = events[worker];
#else
                (void)worker;
                constexpr ReadEvent event = 0;
#endif
                /* Do not stop the team after the first observed fault. All
                 * disjoint tasks must reach a terminal state before the image
                 * can be destroyed, and retaining every task error lets the
                 * ordered scan below choose source/offset deterministically
                 * instead of reporting whichever worker happened to lose the
                 * race first. */
                for (;;) {
                    const size_t index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= tasks.size()) return;
                    ReadTask &task = tasks[index];
                    try {
                        if (test && index == test->fail_task) {
                            fail(LFM_WEIGHT_IO_ERROR,
                                 "injected positioned-read failure");
                        }
                        files[task.source].read_at(task.destination, task.bytes,
                                                   task.offset, event);
                        task.digest = hash_bytes(task.destination, task.bytes);
                        if (accounting) {
                            const int status = accounting->record(
                                LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE,
                                (uint64_t)task.bytes);
                            if (status != 0) {
                                fail(status,
                                     "cannot account safetensors read task");
                            }
                        }
                    } catch (...) {
                        task.error = std::current_exception();
                    }
                    if (test) {
                        test->completed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
    } catch (const std::system_error &error) {
#ifdef _WIN32
        for (const HANDLE event : events) CloseHandle(event);
#endif
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot start safetensors read worker: " + std::string(error.what()));
    } catch (...) {
#ifdef _WIN32
        for (const HANDLE event : events) CloseHandle(event);
#endif
        throw;
    }

#ifdef _WIN32
    for (const HANDLE event : events) CloseHandle(event);
#endif

    if (test && !test->change_after_reads.empty()) {
        const std::string path = test->change_after_reads.string();
        std::unique_ptr<std::FILE, decltype(&std::fclose)> file(
            std::fopen(path.c_str(), "ab"), &std::fclose);
        if (!file || std::fputc(0, file.get()) == EOF ||
            std::fflush(file.get()) != 0) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot mutate loader test source '" + path + "'");
        }
    }

    std::vector<std::exception_ptr> changed(files.size());
    for (size_t source = 0; source < files.size(); ++source) {
        try {
            files[source].verify();
        } catch (...) {
            changed[source] = std::current_exception();
        }
    }

    size_t task = 0;
    for (size_t source = 0; source < files.size(); ++source) {
        if (changed[source]) std::rethrow_exception(changed[source]);
        while (task < tasks.size() && tasks[task].source == source) {
            if (tasks[task].error) std::rethrow_exception(tasks[task].error);
            ++task;
        }
    }
    /* Parallel tasks hash their own exact byte spans. The publication digest is
     * a deterministic tree over the layout and those ordered leaf hashes; it
     * covers every source byte without a second multi-gigabyte scan and binds
     * the zero-filled gaps/tail through their offsets and final segment size. */
    static constexpr char domain[] = "LFM-WEIGHT-CONTENT-V1";
    Sha256 content;
    content.update(domain, sizeof(domain) - 1);
    hash_integer(content, static_cast<uint64_t>(image_bytes));
    hash_integer(content, static_cast<uint64_t>(sources.size()));
    for (size_t source = 0; source < sources.size(); ++source) {
        const Source &value = sources[source];
        const Digest label = label_digest(value.label);
        hash_integer(content, static_cast<uint64_t>(value.offset));
        hash_integer(content, static_cast<uint64_t>(value.bytes));
        hash_integer(content, value.component);
        content.update(label.data(), label.size());
        for (const ReadTask &leaf : tasks) {
            if (leaf.source != source) continue;
            hash_integer(content, static_cast<uint64_t>(leaf.offset));
            hash_integer(content, static_cast<uint64_t>(leaf.bytes));
            content.update(leaf.digest.data(), leaf.digest.size());
        }
    }
    return {static_cast<uint32_t>(tasks.size()),
            static_cast<uint32_t>(count), content.finish()};
}

void read_small_file_exact(const fs::path &path, uint8_t *data, size_t bytes,
                           const LfmPayloadReadScope *accounting = nullptr) {
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
    if (accounting) {
        const int status = accounting->record(
            LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX, (uint64_t)bytes);
        if (status != 0) {
            fail(status, "cannot account checkpoint index read");
        }
    }
}

std::vector<uint8_t> read_small_file(
    const fs::path &path,
    const LfmPayloadReadScope *accounting = nullptr) {
    const size_t bytes = weight_file_size(path);
    if (bytes > kMaxHeaderBytes) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "JSON index exceeds the 100 MB safety limit: '" + path.string() + "'");
    }
    std::vector<uint8_t> data(bytes);
    read_small_file_exact(path, data.data(), data.size(), accounting);
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
    struct Index {
        uint32_t component{LFM_WEIGHT_COMPONENT_MAIN};
        std::unordered_map<std::string, std::string> weights;
    };
    std::vector<Index> indexes;
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

Resolved resolve_index(const fs::path &index_path, uint32_t component,
                       const LfmPayloadReadScope *accounting = nullptr) {
    const auto bytes = read_small_file(index_path, accounting);
    const Json root = parse_json(bytes.data(), bytes.size(), index_path.string());
    if (!root.is_object() || !root.contains("weight_map") ||
        !root.at("weight_map").is_object()) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "checkpoint index has no object weight_map: '" + index_path.string() + "'");
    }

    Resolved resolved;
    Resolved::Index index;
    index.component = component;
    std::unordered_set<std::string> shards;
    for (const auto &item : root.at("weight_map").items()) {
        if (!item.value().is_string()) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "checkpoint index maps tensor '" + item.key() + "' to a non-string shard");
        }
        const std::string shard = safe_shard_name(item.value().get<std::string>());
        index.weights.emplace(item.key(), shard);
        shards.insert(shard);
    }
    if (index.weights.empty()) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "checkpoint index weight_map is empty: '" + index_path.string() + "'");
    }
    std::vector<std::string> ordered_shards(shards.begin(), shards.end());
    std::sort(ordered_shards.begin(), ordered_shards.end());
    for (const std::string &shard : ordered_shards) {
        resolved.sources.push_back(
            {index_path.parent_path() / fs::path(shard), shard, 0, 0, component});
    }
    resolved.indexes.push_back(std::move(index));
    return resolved;
}

Resolved resolve_path(const fs::path &path, uint32_t component,
                      const LfmPayloadReadScope *accounting = nullptr) {
    std::error_code error;
    if (fs::is_regular_file(path, error)) {
        if (path.filename().string().ends_with(".safetensors.index.json")) {
            return resolve_index(path, component, accounting);
        }
        return Resolved{{Source{path, path.filename().generic_string(), 0, 0, component}}, {}};
    }
    if (error) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cannot inspect '" + path.string() + "': " + error.message());
    }
    if (!fs::is_directory(path, error)) {
        fail(LFM_WEIGHT_IO_ERROR, "weight path does not exist: '" + path.string() + "'");
    }

    const fs::path index = path / "model.safetensors.index.json";
    if (fs::is_regular_file(index, error)) {
        return resolve_index(index, component, accounting);
    }
    error.clear();

    const fs::path single = path / "model.safetensors";
    if (fs::is_regular_file(single, error)) {
        return Resolved{
            {Source{single, single.filename().generic_string(), 0, 0, component}}, {}};
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
        resolved.sources.push_back(
            {shard, shard.filename().generic_string(), 0, 0, component});
    }
    return resolved;
}

void append_resolved(Resolved &destination, Resolved source) {
    destination.sources.insert(destination.sources.end(),
                               std::make_move_iterator(source.sources.begin()),
                               std::make_move_iterator(source.sources.end()));
    destination.indexes.insert(destination.indexes.end(),
                               std::make_move_iterator(source.indexes.begin()),
                               std::make_move_iterator(source.indexes.end()));
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

struct WeightImageCore {
    WeightSegment segment;
    std::vector<Source> sources;
    std::vector<TensorMeta> tensors;
    std::unordered_map<std::string, size_t> names[LFM_WEIGHT_COMPONENT_COUNT];
    std::vector<size_t> components[LFM_WEIGHT_COMPONENT_COUNT];
    uint64_t source_bytes{0};
    uint32_t task_count{0};
    uint32_t worker_count{0};
    bool evicted{false};
};

struct LfmWeightImage {
    std::shared_ptr<WeightImageCore> core;
    uint32_t disposition{0};
};

namespace {

std::mutex &weight_registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct RegistryRecord {
    std::weak_ptr<WeightImageCore> core;
    std::vector<ReadySubscriber> subscribers;
    uint64_t claim{0};
    bool loading{false};
};

struct RegistryClaim {
    std::shared_ptr<WeightImageCore> core;
    uint64_t claim{0};
    bool loading{false};
};

struct RegistrySubscription {
    std::shared_ptr<WeightImageCore> core;
    bool retained{false};
};

struct RegistryPublication {
    std::vector<ReadySubscriber> subscribers;
    bool accepted{false};
};

std::unordered_map<std::string, RegistryRecord> &weight_registry() {
    static std::unordered_map<std::string, RegistryRecord> registry;
    return registry;
}

uint64_t next_registry_claim() {
    static uint64_t sequence = 0;
    ++sequence;
    if (sequence == 0) ++sequence;
    return sequence;
}

RegistryClaim registry_claim(const std::string &name) {
    std::lock_guard guard(weight_registry_mutex());
    RegistryRecord &record = weight_registry()[name];
    if (std::shared_ptr<WeightImageCore> resident = record.core.lock()) {
        if (!resident->evicted) return {.core = std::move(resident)};
        record.core.reset();
    }
    if (record.loading) return {.loading = true};
    record.loading = true;
    record.claim = next_registry_claim();
    return {.claim = record.claim};
}

void resume_subscribers(std::vector<ReadySubscriber> subscribers) {
    for (const ReadySubscriber &subscriber : subscribers) {
        (void)koro_cont_resume(subscriber.continuation, &subscriber.identity);
        koro_cont_release(subscriber.continuation);
    }
}

RegistrySubscription registry_subscribe(const std::string &name,
                                        const ReadyTarget &target) {
    if (!target.continuation || target.identity.runtime_epoch == 0 ||
        target.identity.sequence == 0) {
        return {};
    }
    std::lock_guard guard(weight_registry_mutex());
    const auto found = weight_registry().find(name);
    if (found == weight_registry().end()) return {};
    RegistryRecord &record = found->second;
    if (std::shared_ptr<WeightImageCore> resident = record.core.lock()) {
        if (!resident->evicted) return {.core = std::move(resident)};
        record.core.reset();
    }
    if (!record.loading) return {};
    for (const ReadySubscriber &subscriber : record.subscribers) {
        if (subscriber.continuation == target.continuation &&
            ticket_equal(subscriber.identity, target.identity)) {
            return {.retained = true};
        }
    }
    koro_cont_retain(target.continuation);
    record.subscribers.push_back({target.continuation, target.identity});
    return {.retained = true};
}

RegistryPublication registry_publish(
    const std::string &name, uint64_t claim,
    const std::shared_ptr<WeightImageCore> &image) {
    std::lock_guard guard(weight_registry_mutex());
    const auto found = weight_registry().find(name);
    if (found == weight_registry().end() || !found->second.loading ||
        found->second.claim != claim) return {};
    found->second.core = image;
    found->second.loading = false;
    found->second.claim = 0;
    return {
        .subscribers = std::move(found->second.subscribers),
        .accepted = true,
    };
}

std::vector<ReadySubscriber> registry_abandon(const std::string &name,
                                               uint64_t claim) {
    std::lock_guard guard(weight_registry_mutex());
    const auto found = weight_registry().find(name);
    if (found == weight_registry().end() || !found->second.loading ||
        found->second.claim != claim) return {};
    std::vector<ReadySubscriber> subscribers =
        std::move(found->second.subscribers);
    if (found->second.core.expired()) {
        weight_registry().erase(found);
        return subscribers;
    }
    found->second.loading = false;
    found->second.claim = 0;
    return subscribers;
}

void registry_cancel_ready(const ReadyTarget &target) {
    std::vector<koro_cont_t *> releases;
    {
        std::lock_guard guard(weight_registry_mutex());
        for (auto &[name, record] : weight_registry()) {
            (void)name;
            record.subscribers.erase(
                std::remove_if(
                    record.subscribers.begin(), record.subscribers.end(),
                    [&](const ReadySubscriber &subscriber) {
                        if (subscriber.continuation != target.continuation ||
                            !ticket_equal(subscriber.identity, target.identity)) {
                            return false;
                        }
                        releases.push_back(subscriber.continuation);
                        return true;
                    }),
                record.subscribers.end());
        }
    }
    for (koro_cont_t *continuation : releases) koro_cont_release(continuation);
}

class RegistryLoad final {
  public:
    RegistryLoad(std::string name, uint64_t claim)
        : name_(std::move(name)), claim_(claim) {}

    ~RegistryLoad() {
        if (claim_ == 0) return;
        resume_subscribers(registry_abandon(name_, claim_));
    }

    RegistryLoad(const RegistryLoad &) = delete;
    RegistryLoad &operator=(const RegistryLoad &) = delete;

    void publish(const std::shared_ptr<WeightImageCore> &image) {
        RegistryPublication publication =
            registry_publish(name_, claim_, image);
        if (!publication.accepted) {
            fail(LFM_WEIGHT_REJECTED,
                 "shared weight registry claim was evicted before publication");
        }
        claim_ = 0;
        resume_subscribers(std::move(publication.subscribers));
    }

  private:
    std::string name_;
    uint64_t claim_{0};
};

void parse_shard(WeightImageCore &image, uint32_t shard) {
    const Source &source = image.sources.at(shard);
    const uint8_t *file = image.segment.data() + source.offset;
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
        const uint32_t component = source.component;
        if (component >= LFM_WEIGHT_COMPONENT_COUNT) {
            fail(LFM_WEIGHT_FORMAT_ERROR, "invalid weight source component");
        }
        if (image.names[component].contains(tensor.name)) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "duplicate tensor name inside weight component: '" + tensor.name + "'");
        }

        TensorMeta resident;
        resident.name = std::move(tensor.name);
        resident.shape = std::move(tensor.shape);
        resident.offset = source.offset + payload_start + tensor.start;
        resident.elements = tensor.elements;
        resident.bytes = tensor.bytes;
        resident.dtype = tensor.dtype;
        resident.shard = shard;
        resident.component = component;
        const size_t index = image.tensors.size();
        image.names[component].emplace(resident.name, index);
        image.components[component].push_back(index);
        image.tensors.push_back(std::move(resident));
    }
    if (cursor != payload_bytes) {
        fail(LFM_WEIGHT_FORMAT_ERROR,
             "safetensors payload is not fully described in '" + source.path.string() + "'");
    }
}

LfmWeightImage *load(
    Resolved resolved, LoadOptions options = {},
    const LfmPayloadReadScope *accounting = nullptr,
    const ReadyTarget *ready = nullptr) {
    if (resolved.sources.empty()) {
        fail(LFM_WEIGHT_INVALID_ARGUMENT, "no safetensors sources were provided");
    }
    if (options.workers == 0 || options.workers > kReadWorkers) {
        fail(LFM_WEIGHT_INVALID_ARGUMENT,
             "safetensors worker count must be between one and four");
    }
#ifdef _WIN32
    if (options.uncached) {
        fail(LFM_WEIGHT_IO_ERROR,
             "cold-cache loader benchmarking is unsupported on Windows");
    }
#endif
    if (resolved.sources.size() > std::numeric_limits<uint32_t>::max()) {
        fail(LFM_WEIGHT_FORMAT_ERROR, "too many safetensors shards");
    }

    std::vector<OpenFile> files;
    files.reserve(resolved.sources.size());
    for (auto &source : resolved.sources) {
        files.emplace_back(source.path, options.uncached);
        source.bytes = files.back().bytes();
    }

    size_t total = kSegmentHeaderBytes;
    size_t source_bytes = 0;
    for (auto &source : resolved.sources) {
        source_bytes = checked_add(source_bytes, source.bytes, "weight source bytes");
        source.offset = checked_align(total);
        total = checked_add(source.offset, source.bytes, "weight image size");
    }

    total = checked_align(total);
    const Digest identity = identity_digest(resolved.sources, files);
    const std::string name = segment_name(identity);
    RegistryClaim claim;
    for (;;) {
        claim = registry_claim(name);
        if (claim.core) {
            return new LfmWeightImage{
                .core = std::move(claim.core),
                .disposition = LFM_WEIGHT_LOAD_REGISTRY_REUSED,
            };
        }
        if (!claim.loading) break;
        if (!ready) {
            fail(LFM_WEIGHT_IN_PROGRESS,
                 "shared weight image is INITIALIZING or BUILDING; "
                 "synchronous callers must "
                 "retry from a correlated continuation edge");
        }
        RegistrySubscription subscription = registry_subscribe(name, *ready);
        if (subscription.core) {
            return new LfmWeightImage{
                .core = std::move(subscription.core),
                .disposition = LFM_WEIGHT_LOAD_REGISTRY_REUSED,
            };
        }
        if (subscription.retained) {
            fail(LFM_WEIGHT_IN_PROGRESS,
                 "shared weight image is INITIALIZING or BUILDING; "
                 "continuation retained");
        }
        /* Publication or abandonment won between claim and subscription.
         * Re-observe the registry transition in this same active callback;
         * this is not a wait or a polling edge. */
    }
    RegistryLoad registry(name, claim.claim);
    const auto build_begin = std::chrono::steady_clock::now();
    auto image = std::make_shared<WeightImageCore>();
    try {
        image->segment = WeightSegment::acquire(
            resolved.sources, files, total, source_bytes, identity,
            options.test && options.test->fail_wire);
    } catch (const WeightError &error) {
        if (error.status() != LFM_WEIGHT_IN_PROGRESS) throw;
        if (ready) {
            fail(LFM_WEIGHT_REJECTED,
                 "a foreign process owns the active weight generation; "
                 "direct continuation admission has no cross-process callback "
                 "edge (enter through the native model host readiness ticket)");
        }
        fail(LFM_WEIGHT_IN_PROGRESS,
             "a foreign process owns the active weight generation; "
             "host-less synchronous open cannot wait or poll for it");
    }
    image->sources = std::move(resolved.sources);
    image->source_bytes = source_bytes;
    ReadSummary summary{};
    if (image->segment.creator()) {
        uint8_t *destination = image->segment.mutable_data();
        size_t cursor = kSegmentHeaderBytes;
        for (const auto &source : image->sources) {
            if (source.offset > cursor) {
                std::memset(destination + cursor, 0, source.offset - cursor);
            }
            cursor = checked_add(source.offset, source.bytes,
                                 "weight image cursor");
        }
        if (image->segment.size() > cursor) {
            std::memset(destination + cursor, 0,
                        image->segment.size() - cursor);
        }
        summary = read_sources(files, destination, image->sources,
                               image->segment.size(), options.workers,
                               options.test, accounting);
        image->task_count = summary.tasks;
        image->worker_count = summary.workers;
    } else {
        const SegmentHeader *header = image->segment.published_header();
        image->task_count = header->build_tasks;
        image->worker_count = header->build_workers;
    }
    files.clear();
    for (uint32_t shard = 0; shard < image->sources.size(); ++shard) {
        parse_shard(*image, shard);
    }

    for (const auto &index : resolved.indexes) {
        if (index.component >= LFM_WEIGHT_COMPONENT_COUNT) {
            fail(LFM_WEIGHT_FORMAT_ERROR, "invalid checkpoint index component");
        }
        if (index.weights.size() != image->components[index.component].size()) {
            fail(LFM_WEIGHT_FORMAT_ERROR,
                 "checkpoint index and loaded shard tensor counts differ");
        }
        for (const size_t tensor_index : image->components[index.component]) {
            const auto &tensor = image->tensors[tensor_index];
            const auto found = index.weights.find(tensor.name);
            const std::string &source = image->sources.at(tensor.shard).label;
            if (found == index.weights.end() || found->second != source) {
                fail(LFM_WEIGHT_FORMAT_ERROR,
                     "checkpoint index maps tensor '" + tensor.name + "' to the wrong shard");
            }
        }
    }
    if (image->segment.creator()) {
        const auto build_end = std::chrono::steady_clock::now();
        const uint64_t build_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                build_end - build_begin)
                .count());
        /* Publication is the ownership boundary: source handles have closed,
         * every typed span and shard index has validated, and READY is the
         * release edge after the content digest. The mapping becomes read-only
         * before any model receives a view. */
        image->segment.publish(summary.content_digest, build_ns,
                               summary.tasks, summary.workers);
    }
    const uint32_t disposition = image->segment.creator()
                                     ? LFM_WEIGHT_LOAD_BUILT
                                     : LFM_WEIGHT_LOAD_ATTACHED;
    registry.publish(image);
    return new LfmWeightImage{
        .core = std::move(image),
        .disposition = disposition,
    };
}

void fill_view(const LfmWeightImage &image, const TensorMeta &tensor,
               LfmTensorView &view) {
    const WeightImageCore &core = *image.core;
    view = {};
    view.size = sizeof(LfmTensorView);
    view.abi_version = LFM_WEIGHT_ABI_VERSION;
    view.name = tensor.name.c_str();
    view.data = core.segment.data() + tensor.offset;
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
        *out = open();
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

static int weights_open(const char *path, const LfmPayloadReadOwner *owner,
                        const ReadyTarget *ready,
                        LfmWeightImage **out, char *err, size_t errlen) {
    if (!path || path[0] == '\0') {
        if (out) *out = nullptr;
        set_error(err, errlen, "empty weight path");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    return open_c(
        [&] {
            LfmPayloadReadScope scope(
                owner, LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
                           LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX);
            if (scope.status() != 0) {
                fail(scope.status(),
                     "weight-image read rejected by its model owner");
            }
            return load(resolve_path(fs::path(path),
                                     LFM_WEIGHT_COMPONENT_MAIN, &scope),
                        {}, &scope, ready);
        },
        out, err, errlen);
}

extern "C" int lfm_weights_open(const char *path, LfmWeightImage **out,
                                char *err, size_t errlen) {
    return weights_open(path, nullptr, nullptr, out, err, errlen);
}

int lfm_weights_open_owned(const char *path,
                           const LfmPayloadReadOwner *owner,
                           LfmWeightImage **out, char *err, size_t errlen) {
    if (!owner) return LFM_WEIGHT_INVALID_ARGUMENT;
    return weights_open(path, owner, nullptr, out, err, errlen);
}

int lfm_weights_open_owned_continuation(
    const char *path, const LfmPayloadReadOwner *owner,
    koro_cont_t *continuation, LfmWeightImage **out, char *err,
    size_t errlen) {
    if (!owner || !continuation) return LFM_WEIGHT_INVALID_ARGUMENT;
    const ReadyTarget ready = {
        .continuation = continuation,
        .identity = koro_cont_identity(continuation),
    };
    return weights_open(path, owner, &ready, out, err, errlen);
}

void lfm_weights_cancel_readiness(koro_cont_t *continuation) {
    if (!continuation) return;
    registry_cancel_ready({continuation, koro_cont_identity(continuation)});
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
                resolved.sources.push_back(
                    {path, path.filename().generic_string(), 0, 0,
                     LFM_WEIGHT_COMPONENT_MAIN});
            }
            return load(std::move(resolved));
        },
        out, err, errlen);
}

static int weights_open_bundle(const char *main_path,
                               const char *detokenizer_path,
                               const LfmPayloadReadOwner *owner,
                               const ReadyTarget *ready,
                               LfmWeightImage **out, char *err,
                               size_t errlen) {
    if (!main_path || main_path[0] == '\0' || !detokenizer_path ||
        detokenizer_path[0] == '\0') {
        if (out) *out = nullptr;
        set_error(err, errlen, "empty main or detokenizer weight path");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    return open_c(
        [&] {
            LfmPayloadReadScope scope(
                owner, LFM_MODEL_PAYLOAD_READ_WEIGHT_IMAGE |
                           LFM_MODEL_PAYLOAD_READ_WEIGHT_INDEX);
            if (scope.status() != 0) {
                fail(scope.status(),
                     "weight-bundle read rejected by its model owner");
            }
            Resolved resolved;
            append_resolved(
                resolved,
                resolve_path(fs::path(main_path), LFM_WEIGHT_COMPONENT_MAIN,
                             &scope));
            append_resolved(
                resolved,
                resolve_path(fs::path(detokenizer_path),
                             LFM_WEIGHT_COMPONENT_DETOKENIZER, &scope));
            return load(std::move(resolved), {}, &scope, ready);
        },
        out, err, errlen);
}

extern "C" int lfm_weights_open_bundle(const char *main_path,
                                        const char *detokenizer_path,
                                        LfmWeightImage **out, char *err,
                                        size_t errlen) {
    return weights_open_bundle(main_path, detokenizer_path, nullptr, nullptr,
                               out, err, errlen);
}

int lfm_weights_open_bundle_owned(const char *main_path,
                                  const char *detokenizer_path,
                                  const LfmPayloadReadOwner *owner,
                                  LfmWeightImage **out, char *err,
                                  size_t errlen) {
    if (!owner) return LFM_WEIGHT_INVALID_ARGUMENT;
    return weights_open_bundle(main_path, detokenizer_path, owner, nullptr, out,
                               err, errlen);
}

int lfm_weights_open_bundle_owned_continuation(
    const char *main_path, const char *detokenizer_path,
    const LfmPayloadReadOwner *owner, koro_cont_t *continuation,
    LfmWeightImage **out, char *err, size_t errlen) {
    if (!owner || !continuation) return LFM_WEIGHT_INVALID_ARGUMENT;
    const ReadyTarget ready = {
        .continuation = continuation,
        .identity = koro_cont_identity(continuation),
    };
    return weights_open_bundle(main_path, detokenizer_path, owner, &ready, out,
                               err, errlen);
}

/* Deliberately absent from the installed header: the load benchmark needs to
 * run the exact production planner with one and four I/O workers, and (where
 * supported) with the file cache bypassed. This is not a model/loader ABI for
 * product code and may change with the benchmark without compatibility notice. */
extern "C" int lfm_internal_weights_open_bundle_benchmark(
    const char *main_path, const char *detokenizer_path, uint32_t workers,
    uint32_t uncached, LfmWeightImage **out, char *err, size_t errlen) {
    if (!main_path || main_path[0] == '\0' || !detokenizer_path ||
        detokenizer_path[0] == '\0' || uncached > 1) {
        if (out) *out = nullptr;
        set_error(err, errlen, "invalid native load benchmark arguments");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    return open_c(
        [&] {
            Resolved resolved;
            append_resolved(
                resolved,
                resolve_path(fs::path(main_path), LFM_WEIGHT_COMPONENT_MAIN));
            append_resolved(
                resolved,
                resolve_path(fs::path(detokenizer_path),
                             LFM_WEIGHT_COMPONENT_DETOKENIZER));
            return load(std::move(resolved),
                        LoadOptions{workers, uncached != 0});
        },
        out, err, errlen);
}

extern "C" int lfm_internal_weights_benchmark_cold_supported(void) {
#if defined(__APPLE__) || defined(POSIX_FADV_DONTNEED)
    return 1;
#else
    return 0;
#endif
}

extern "C" int lfm_internal_weights_open_fault_test(
    const char *path, uint32_t mode, uint32_t *scheduled,
    uint32_t *completed, char *err, size_t errlen) {
    if (!path || path[0] == '\0' || !scheduled || !completed ||
        (mode != 1 && mode != 2 && mode != 3)) {
        set_error(err, errlen, "invalid native loader fault-test arguments");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    *scheduled = 0;
    *completed = 0;
    ReadTestHook hook;
    if (mode == 1) {
        hook.fail_task = 0;
    } else if (mode == 2) {
        hook.change_after_reads = fs::path(path);
    } else {
        hook.fail_wire = true;
    }
    LfmWeightImage *image = nullptr;
    const int status = open_c(
        [&] {
            return load(resolve_path(fs::path(path),
                                     LFM_WEIGHT_COMPONENT_MAIN),
                        LoadOptions{kReadWorkers, false, &hook});
        },
        &image, err, errlen);
    lfm_weights_close(image);
    if (hook.scheduled <= std::numeric_limits<uint32_t>::max()) {
        *scheduled = static_cast<uint32_t>(hook.scheduled);
    }
    const size_t done = hook.completed.load(std::memory_order_relaxed);
    if (done <= std::numeric_limits<uint32_t>::max()) {
        *completed = static_cast<uint32_t>(done);
    }
    return status;
}

/* Hostile/stale namespace gate. It fabricates every persisted crash window
 * through the elected builder's real publication helpers: zero state before
 * INITIALIZING, live/dead INITIALIZING, live/dead BUILDING, malformed READY,
 * and POISONED. Other modes mutate exactly one contract field before entering
 * through the public attach path. Test code does not duplicate the validation
 * ladder it is meant to verify. */
extern "C" int lfm_internal_weights_hostile_segment_test(
    const char *path, uint32_t mode, int32_t *observed_status,
    uint64_t *abandoned_generation, uint64_t *published_generation,
    char *err, size_t errlen) {
#ifdef _WIN32
    (void)path;
    (void)mode;
    (void)observed_status;
    (void)abandoned_generation;
    (void)published_generation;
    set_error(err, errlen,
              "hostile POSIX shared-memory fixture is unsupported on Windows");
    return LFM_WEIGHT_REJECTED;
#else
    if (!path || !path[0] || mode < 1 || mode > 11 || !observed_status ||
        !abandoned_generation || !published_generation) {
        set_error(err, errlen, "invalid hostile-segment gate arguments");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    *observed_status = 0;
    *abandoned_generation = 0;
    *published_generation = 0;
    try {
        Resolved resolved =
            resolve_path(fs::path(path), LFM_WEIGHT_COMPONENT_MAIN);
        std::vector<OpenFile> files;
        files.reserve(resolved.sources.size());
        size_t total = kSegmentHeaderBytes;
        size_t source_bytes = 0;
        for (Source &source : resolved.sources) {
            files.emplace_back(source.path, false);
            source.bytes = files.back().bytes();
            source_bytes = checked_add(source_bytes, source.bytes,
                                       "hostile fixture source bytes");
            source.offset = checked_align(total);
            total = checked_add(source.offset, source.bytes,
                                "hostile fixture segment bytes");
        }
        total = checked_align(total);
        const Digest identity = identity_digest(resolved.sources, files);
        const std::string name = segment_name(identity);
        (void)lfm_weights_evict(identity.data(), nullptr, 0);

        const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0) {
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot create hostile shared weight fixture: " +
                     std::string(std::strerror(errno)));
        }
        const size_t object_bytes = mode == 2 ? total - kWeightAlign : total;
        if (ftruncate(fd, static_cast<off_t>(object_bytes)) != 0) {
            const int error = errno;
            (void)::close(fd);
            (void)shm_unlink(name.c_str());
            fail(LFM_WEIGHT_IO_ERROR,
                 "cannot size hostile shared weight fixture: " +
                     std::string(std::strerror(error)));
        }
        if (mode != 2) {
            void *mapping = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                (void)::close(fd);
                (void)shm_unlink(name.c_str());
                fail(LFM_WEIGHT_IO_ERROR,
                     "cannot map hostile shared weight fixture: " +
                         std::string(std::strerror(error)));
            }
            auto *header = static_cast<SegmentHeader *>(mapping);
            if (mode == 1) {
                std::memset(header, 0, kSegmentHeaderBytes);
            } else {
                const bool dead = mode == 6 || mode == 10;
                const bool initializing = mode == 9 || mode == 10;
                const uint64_t pid = dead ? UINT64_MAX : current_pid();
                const uint64_t started =
                    dead ? UINT64_C(1) : process_start_time(pid);
                const uint64_t generation = segment_generation();
                initialize_segment_owner(header, generation, pid, started,
                                         current_uid());
                *abandoned_generation = generation;
                if (!initializing) {
                    publish_segment_build_header(header, resolved.sources,
                                                 total, source_bytes, identity);
                    if (mode == 3) header->header_bytes = 1;
                    if (mode == 4) {
                        /* Deliberately publish READY without a content tree. */
                        publish_segment_state(header, kSegmentReady);
                    }
                    if (mode == 7) header->owner_uid = current_uid() + 1;
                    if (mode == 8) ++header->sources[0].offset;
                    if (mode == 11) {
                        publish_segment_state(header, kSegmentPoisoned);
                    }
                }
            }
            (void)munmap(mapping, total);
        }
        (void)::close(fd);

        LfmWeightImage *image = nullptr;
        char open_error[512]{};
        *observed_status = lfm_weights_open(path, &image, open_error,
                                            sizeof(open_error));
        const int32_t expected = mode == 5 || mode == 9
                                     ? LFM_WEIGHT_IN_PROGRESS
                                 : mode == 6 || mode == 10 ? LFM_WEIGHT_OK
                                                          : LFM_WEIGHT_REJECTED;
        int result = LFM_WEIGHT_OK;
        if (*observed_status != expected) {
            set_error(err, errlen,
                      open_error[0] ? open_error
                                    : "hostile object returned wrong status");
            result = LFM_WEIGHT_REJECTED;
        }
        if ((mode == 6 || mode == 10) && image) {
            LfmWeightLoadStatsV2 stats{
                .size = sizeof(LfmWeightLoadStatsV2),
                .abi_version = LFM_WEIGHT_ABI_VERSION,
            };
            if (lfm_weights_load_stats(image, &stats) != LFM_WEIGHT_OK ||
                !(stats.flags & LFM_WEIGHT_LOAD_BUILT) ||
                stats.generation == *abandoned_generation) {
                set_error(err, errlen,
                          "dead initializer generation was not replaced "
                          "exactly once");
                result = LFM_WEIGHT_REJECTED;
            }
            *published_generation = stats.generation;
        }
        lfm_weights_close(image);
        (void)lfm_weights_evict(identity.data(), nullptr, 0);
        return result;
    } catch (const WeightError &error) {
        set_error(err, errlen, error.what());
        return error.status();
    } catch (const std::exception &error) {
        set_error(err, errlen, error.what());
        return LFM_WEIGHT_FORMAT_ERROR;
    }
#endif
}

extern "C" void lfm_weights_close(LfmWeightImage *image) {
    delete image;
}

extern "C" int lfm_weights_evict(const uint8_t identity_digest[32],
                                  char *err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!identity_digest || digest_empty(identity_digest)) {
        set_error(err, errlen, "empty shared weight identity");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    Digest identity{};
    std::memcpy(identity.data(), identity_digest, identity.size());
    const std::string name = segment_name(identity);
    std::vector<ReadySubscriber> subscribers;
    {
        std::lock_guard guard(weight_registry_mutex());
        const auto found = weight_registry().find(name);
        if (found != weight_registry().end()) {
            if (std::shared_ptr<WeightImageCore> resident =
                    found->second.core.lock()) {
                resident->evicted = true;
            }
            subscribers = std::move(found->second.subscribers);
            weight_registry().erase(found);
        }
    }
    resume_subscribers(std::move(subscribers));
#ifdef _WIN32
    set_error(err, errlen,
              "Windows named sections are evicted by retiring the keeper lease; "
              "explicit namespace unlink is unavailable");
    return LFM_WEIGHT_REJECTED;
#else
    if (shm_unlink(name.c_str()) == 0 || errno == ENOENT) return LFM_WEIGHT_OK;
    const int error = errno;
    set_error(err, errlen,
              ("cannot evict shared weight segment '" + name + "': " +
               std::strerror(error))
                  .c_str());
    return LFM_WEIGHT_IO_ERROR;
#endif
}

/* Test cleanup resolves identity from file metadata only; it never opens or
 * reads a published segment and is deliberately absent from the installed
 * header. Product detach must never smuggle an unlink through close(). */
extern "C" int lfm_internal_weights_evict_path_for_test(const char *path) {
    if (!path || !path[0]) return LFM_WEIGHT_INVALID_ARGUMENT;
    try {
        Resolved resolved =
            resolve_path(fs::path(path), LFM_WEIGHT_COMPONENT_MAIN, nullptr);
        std::vector<OpenFile> files;
        files.reserve(resolved.sources.size());
        for (Source &source : resolved.sources) {
            files.emplace_back(source.path, false);
            source.bytes = files.back().bytes();
        }
        const Digest identity = identity_digest(resolved.sources, files);
        return lfm_weights_evict(identity.data(), nullptr, 0);
    } catch (...) {
        return LFM_WEIGHT_NOT_FOUND;
    }
}

extern "C" const void *lfm_weights_data(const LfmWeightImage *image) {
    return image && image->core ? image->core->segment.data() : nullptr;
}

extern "C" uint64_t lfm_weights_resident_bytes(const LfmWeightImage *image) {
    return image && image->core ? image->core->segment.size() : 0;
}

extern "C" size_t lfm_weights_count(const LfmWeightImage *image) {
    return image && image->core
               ? image->core->components[LFM_WEIGHT_COMPONENT_MAIN].size()
               : 0;
}

extern "C" size_t lfm_weights_component_count(const LfmWeightImage *image,
                                                uint32_t component) {
    if (!image || !image->core || component >= LFM_WEIGHT_COMPONENT_COUNT) return 0;
    return image->core->components[component].size();
}

extern "C" int lfm_weights_load_stats(const LfmWeightImage *image,
                                       LfmWeightLoadStatsV2 *out) {
    if (!image || !image->core || !out) return LFM_WEIGHT_INVALID_ARGUMENT;
    const WeightImageCore &core = *image->core;
    *out = {};
    out->size = static_cast<uint32_t>(sizeof(LfmWeightLoadStatsV2));
    out->abi_version = LFM_WEIGHT_ABI_VERSION;
    out->source_bytes = core.source_bytes;
    out->segment_bytes = core.segment.size();
    out->segment_constructed_bytes =
        image->disposition == LFM_WEIGHT_LOAD_BUILT ? core.segment.size() : 0;
    out->attached_shared_bytes =
        image->disposition == LFM_WEIGHT_LOAD_ATTACHED ? core.segment.size()
                                                      : 0;
    out->wired_bytes = core.segment.wired() ? core.segment.size() : 0;
    out->process_resident_bytes = out->wired_bytes;
    const SegmentHeader *header = core.segment.published_header();
    out->build_ns = header->build_ns;
    out->attach_ns = core.segment.attach_ns();
    out->generation = header->generation;
    out->task_count = core.task_count;
    out->worker_count = core.worker_count;
    out->flags = (image->disposition == LFM_WEIGHT_LOAD_BUILT
                      ? LFM_WEIGHT_LOAD_BUILT
                  : image->disposition == LFM_WEIGHT_LOAD_ATTACHED
                      ? LFM_WEIGHT_LOAD_ATTACHED
                      : 0u) |
                 (core.segment.wired() ? LFM_WEIGHT_LOAD_WIRED : 0u) |
                 (image->disposition == LFM_WEIGHT_LOAD_REGISTRY_REUSED
                      ? LFM_WEIGHT_LOAD_REGISTRY_REUSED
                      : 0u);
    out->source_count = static_cast<uint32_t>(core.sources.size());
    out->payload_read_calls =
        image->disposition == LFM_WEIGHT_LOAD_BUILT ? core.task_count : 0;
    out->payload_read_bytes =
        image->disposition == LFM_WEIGHT_LOAD_BUILT ? core.source_bytes : 0;
    std::memcpy(out->identity_digest, header->identity_digest,
                sizeof(out->identity_digest));
    std::memcpy(out->content_digest, header->content_digest,
                sizeof(out->content_digest));
    return LFM_WEIGHT_OK;
}

extern "C" int lfm_weights_at(const LfmWeightImage *image, size_t index,
                              LfmTensorView *out) {
    return lfm_weights_at_component(image, LFM_WEIGHT_COMPONENT_MAIN, index, out);
}

extern "C" int lfm_weights_find(const LfmWeightImage *image, const char *name,
                                LfmTensorView *out) {
    return lfm_weights_find_component(image, LFM_WEIGHT_COMPONENT_MAIN, name, out);
}

extern "C" int lfm_weights_at_component(const LfmWeightImage *image,
                                          uint32_t component, size_t index,
                                          LfmTensorView *out) {
    if (!image || !image->core || !out || component >= LFM_WEIGHT_COMPONENT_COUNT) {
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    const WeightImageCore &core = *image->core;
    if (index >= core.components[component].size()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, core.tensors[core.components[component][index]], *out);
    return LFM_WEIGHT_OK;
}

extern "C" int lfm_weights_find_component(const LfmWeightImage *image,
                                            uint32_t component,
                                            const char *name,
                                            LfmTensorView *out) {
    if (!image || !image->core || !name || !out ||
        component >= LFM_WEIGHT_COMPONENT_COUNT) {
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    const WeightImageCore &core = *image->core;
    const auto found = core.names[component].find(name);
    if (found == core.names[component].end()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, core.tensors[found->second], *out);
    return LFM_WEIGHT_OK;
}

extern "C" const char *lfm_weights_dtype_name(uint32_t dtype) {
    for (const auto &info : kDTypes) {
        if (info.value == dtype) return info.name;
    }
    return "INVALID";
}
