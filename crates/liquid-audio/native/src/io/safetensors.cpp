// Native resident safetensors loader.
//
// The span planning and whole-file residency discipline comes from the
// safetensors path in ember-ml. This version deliberately stops before UKM's
// numerical ingress: model payloads remain byte-exact checkpoint storage and
// kernels receive immutable pointers into one process-long aligned image.

#include "lfm_safetensors.h"
#include "lfm_payload_reader.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

constexpr size_t kWeightAlign = 64;
constexpr size_t kReadChunkBytes = 8 * 1024 * 1024;
constexpr size_t kReadWorkers = 4;
constexpr uint64_t kMaxHeaderBytes = 100'000'000;
static_assert(sizeof(LfmWeightLoadStatsV1) == 32);

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

class AlignedBytes {
  public:
    AlignedBytes() = default;

    explicit AlignedBytes(size_t bytes) : bytes_(bytes) {
#ifdef _WIN32
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        const size_t page = static_cast<size_t>(info.dwPageSize);
#else
        const long configured = sysconf(_SC_PAGESIZE);
        if (configured <= 0) {
            fail(LFM_WEIGHT_IO_ERROR, "cannot query the virtual-memory page size");
        }
        const size_t page = static_cast<size_t>(configured);
#endif
        const size_t logical = bytes == 0 ? kWeightAlign : bytes;
        if (page == 0 || logical > std::numeric_limits<size_t>::max() - (page - 1)) {
            fail(LFM_WEIGHT_OUT_OF_MEMORY, "weight image page alignment overflows size_t");
        }
        allocation_ = ((logical + page - 1) / page) * page;
#ifdef _WIN32
        data_ = static_cast<uint8_t *>(VirtualAlloc(
            nullptr, allocation_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!data_) throw std::bad_alloc();
#else
        void *memory = mmap(nullptr, allocation_, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED) throw std::bad_alloc();
        data_ = static_cast<uint8_t *>(memory);
#endif
    }

    AlignedBytes(const AlignedBytes &) = delete;
    AlignedBytes &operator=(const AlignedBytes &) = delete;

    AlignedBytes(AlignedBytes &&other) noexcept { swap(other); }

    AlignedBytes &operator=(AlignedBytes &&other) noexcept {
        if (this != &other) {
            AlignedBytes empty;
            swap(empty);
            swap(other);
        }
        return *this;
    }

    ~AlignedBytes() {
        if (!data_) return;
#ifdef _WIN32
        (void)VirtualFree(data_, 0, MEM_RELEASE);
#else
        (void)munmap(data_, allocation_);
#endif
    }

    uint8_t *data() { return data_; }
    const uint8_t *data() const { return data_; }
    size_t size() const { return bytes_; }

    void seal() {
        if (!data_ || sealed_) return;
#ifdef _WIN32
        DWORD previous = 0;
        if (!VirtualProtect(data_, allocation_, PAGE_READONLY, &previous)) {
            fail(LFM_WEIGHT_IO_ERROR, "cannot publish the weight image read-only");
        }
#else
        if (mprotect(data_, allocation_, PROT_READ) != 0) {
            fail(LFM_WEIGHT_IO_ERROR, "cannot publish the weight image read-only");
        }
#endif
        sealed_ = true;
    }

    bool sealed() const { return sealed_; }

  private:
    void swap(AlignedBytes &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(bytes_, other.bytes_);
        std::swap(allocation_, other.allocation_);
        std::swap(sealed_, other.sealed_);
    }

    uint8_t *data_{nullptr};
    size_t bytes_{0};
    size_t allocation_{0};
    bool sealed_{false};
};

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

struct ReadTask {
    size_t source{0};
    size_t offset{0};
    size_t bytes{0};
    uint8_t *destination{nullptr};
    std::exception_ptr error;
};

struct ReadSummary {
    uint32_t tasks{0};
    uint32_t workers{0};
};

/* Private deterministic fault injection for the loader integration test. It
 * is reachable only through the unadvertised test entry point below; product
 * opens always pass null and retain exactly the production read path. */
struct ReadTestHook {
    size_t fail_task{std::numeric_limits<size_t>::max()};
    fs::path change_after_reads;
    std::atomic<size_t> completed{0};
    size_t scheduled{0};
};

ReadSummary read_sources(const std::vector<OpenFile> &files, uint8_t *image,
                         const std::vector<Source> &sources,
                         size_t worker_limit, ReadTestHook *test,
                         const LfmPayloadReadScope *accounting = nullptr) {
    std::vector<ReadTask> tasks;
    for (size_t source = 0; source < sources.size(); ++source) {
        for (size_t offset = 0; offset < sources[source].bytes;) {
            const size_t bytes = std::min(kReadChunkBytes, sources[source].bytes - offset);
            tasks.push_back({source, offset, bytes,
                             image + sources[source].offset + offset, {}});
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
    return {static_cast<uint32_t>(tasks.size()), static_cast<uint32_t>(count)};
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

struct LfmWeightImage {
    AlignedBytes storage;
    std::vector<Source> sources;
    std::vector<TensorMeta> tensors;
    std::unordered_map<std::string, size_t> names[LFM_WEIGHT_COMPONENT_COUNT];
    std::vector<size_t> components[LFM_WEIGHT_COMPONENT_COUNT];
    uint64_t source_bytes{0};
    uint32_t task_count{0};
    uint32_t worker_count{0};
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

std::unique_ptr<LfmWeightImage> load(
    Resolved resolved, LoadOptions options = {},
    const LfmPayloadReadScope *accounting = nullptr) {
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

    size_t total = 0;
    size_t source_bytes = 0;
    for (auto &source : resolved.sources) {
        source_bytes = checked_add(source_bytes, source.bytes, "weight source bytes");
        source.offset = checked_align(total);
        total = checked_add(source.offset, source.bytes, "weight image size");
    }

    auto image = std::make_unique<LfmWeightImage>();
    image->storage = AlignedBytes(checked_align(total));
    image->sources = std::move(resolved.sources);

    size_t cursor = 0;
    for (const auto &source : image->sources) {
        if (source.offset > cursor) {
            std::memset(image->storage.data() + cursor, 0, source.offset - cursor);
        }
        cursor = checked_add(source.offset, source.bytes, "weight image cursor");
    }
    if (image->storage.size() > cursor) {
        std::memset(image->storage.data() + cursor, 0, image->storage.size() - cursor);
    }

    const ReadSummary summary = read_sources(files, image->storage.data(),
                                             image->sources, options.workers,
                                             options.test, accounting);
    files.clear();
    image->source_bytes = source_bytes;
    image->task_count = summary.tasks;
    image->worker_count = summary.workers;
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
    // Publication is the ownership boundary: every source handle has closed,
    // metadata and exact typed spans have validated, and all later consumers
    // receive const byte views. Enforce that invariant in the page tables so an
    // accidental C++ write faults instead of silently corrupting shared weights.
    image->storage.seal();
    return image;
}

void fill_view(const LfmWeightImage &image, const TensorMeta &tensor,
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

static int weights_open(const char *path, const LfmPayloadReadOwner *owner,
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
                        {}, &scope);
        },
        out, err, errlen);
}

extern "C" int lfm_weights_open(const char *path, LfmWeightImage **out,
                                char *err, size_t errlen) {
    return weights_open(path, nullptr, out, err, errlen);
}

int lfm_weights_open_owned(const char *path,
                           const LfmPayloadReadOwner *owner,
                           LfmWeightImage **out, char *err, size_t errlen) {
    if (!owner) return LFM_WEIGHT_INVALID_ARGUMENT;
    return weights_open(path, owner, out, err, errlen);
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
            return load(std::move(resolved), {}, &scope);
        },
        out, err, errlen);
}

extern "C" int lfm_weights_open_bundle(const char *main_path,
                                        const char *detokenizer_path,
                                        LfmWeightImage **out, char *err,
                                        size_t errlen) {
    return weights_open_bundle(main_path, detokenizer_path, nullptr, out, err,
                               errlen);
}

int lfm_weights_open_bundle_owned(const char *main_path,
                                  const char *detokenizer_path,
                                  const LfmPayloadReadOwner *owner,
                                  LfmWeightImage **out, char *err,
                                  size_t errlen) {
    if (!owner) return LFM_WEIGHT_INVALID_ARGUMENT;
    return weights_open_bundle(main_path, detokenizer_path, owner, out, err,
                               errlen);
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
        (mode != 1 && mode != 2)) {
        set_error(err, errlen, "invalid native loader fault-test arguments");
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    *scheduled = 0;
    *completed = 0;
    ReadTestHook hook;
    if (mode == 1) {
        hook.fail_task = 0;
    } else {
        hook.change_after_reads = fs::path(path);
    }
    LfmWeightImage *image = nullptr;
    const int status = open_c(
        [&] {
            return load(resolve_path(fs::path(path),
                                     LFM_WEIGHT_COMPONENT_MAIN),
                        LoadOptions{kReadWorkers, false, &hook});
        },
        &image, err, errlen);
    delete image;
    if (hook.scheduled <= std::numeric_limits<uint32_t>::max()) {
        *scheduled = static_cast<uint32_t>(hook.scheduled);
    }
    const size_t done = hook.completed.load(std::memory_order_relaxed);
    if (done <= std::numeric_limits<uint32_t>::max()) {
        *completed = static_cast<uint32_t>(done);
    }
    return status;
}

extern "C" void lfm_weights_close(LfmWeightImage *image) { delete image; }

extern "C" const void *lfm_weights_data(const LfmWeightImage *image) {
    return image ? image->storage.data() : nullptr;
}

extern "C" uint64_t lfm_weights_resident_bytes(const LfmWeightImage *image) {
    return image ? image->storage.size() : 0;
}

extern "C" size_t lfm_weights_count(const LfmWeightImage *image) {
    return image ? image->components[LFM_WEIGHT_COMPONENT_MAIN].size() : 0;
}

extern "C" size_t lfm_weights_component_count(const LfmWeightImage *image,
                                                uint32_t component) {
    if (!image || component >= LFM_WEIGHT_COMPONENT_COUNT) return 0;
    return image->components[component].size();
}

extern "C" int lfm_weights_load_stats(const LfmWeightImage *image,
                                       LfmWeightLoadStatsV1 *out) {
    if (!image || !out) return LFM_WEIGHT_INVALID_ARGUMENT;
    *out = {};
    out->size = static_cast<uint32_t>(sizeof(LfmWeightLoadStatsV1));
    out->abi_version = LFM_WEIGHT_ABI_VERSION;
    out->source_bytes = image->source_bytes;
    out->resident_bytes = image->storage.size();
    out->task_count = image->task_count;
    out->worker_count = image->worker_count;
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
    if (!image || !out || component >= LFM_WEIGHT_COMPONENT_COUNT) {
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    if (index >= image->components[component].size()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, image->tensors[image->components[component][index]], *out);
    return LFM_WEIGHT_OK;
}

extern "C" int lfm_weights_find_component(const LfmWeightImage *image,
                                            uint32_t component,
                                            const char *name,
                                            LfmTensorView *out) {
    if (!image || !name || !out || component >= LFM_WEIGHT_COMPONENT_COUNT) {
        return LFM_WEIGHT_INVALID_ARGUMENT;
    }
    const auto found = image->names[component].find(name);
    if (found == image->names[component].end()) return LFM_WEIGHT_NOT_FOUND;
    fill_view(*image, image->tensors[found->second], *out);
    return LFM_WEIGHT_OK;
}

extern "C" const char *lfm_weights_dtype_name(uint32_t dtype) {
    for (const auto &info : kDTypes) {
        if (info.value == dtype) return info.name;
    }
    return "INVALID";
}
