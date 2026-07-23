// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

#if !defined(_WIN32)
#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kc {

/*
 * One setup-time durable fatal record.
 *
 * The file, mapping, page residency, and header are established before work is
 * admitted. publish() therefore performs bounded memory stores only. The
 * shared mapping survives a process abort without adding formatting, storage
 * synchronization, allocation, or a syscall to the poisoned execution path.
 */
template <typename Record, std::size_t Isolation = 128>
class FatalStore final {
    static_assert(std::is_trivially_copyable_v<Record>);
    static_assert(Isolation >= 64);
    static_assert((Isolation & (Isolation - 1)) == 0);
    static_assert(alignof(Record) <= Isolation);
    static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free);

  public:
    static constexpr std::uint32_t committed = 1;

    struct alignas(Isolation) Header {
        std::uint64_t magic = 0;
        std::uint32_t publication = 0;
        std::uint64_t runtime_epoch = 0;
        std::uint64_t checksum = 0;
        std::array<unsigned char, Isolation - 32> padding{};
    };

    static_assert(sizeof(Header) == Isolation);
    static_assert(std::is_trivially_copyable_v<Header>);

    struct Config {
        std::uint64_t magic = 0;
        std::uint64_t runtime_epoch = 0;
        const char *path = nullptr;
    };

    FatalStore() noexcept = default;
    FatalStore(const FatalStore &) = delete;
    FatalStore &operator=(const FatalStore &) = delete;

    [[nodiscard]] int initialize(const Config &config) noexcept {
#if defined(_WIN32)
        (void)config;
        return -ENOTSUP;
#else
        if (config.magic == 0 || config.runtime_epoch == 0 ||
            mapping_ != MAP_FAILED || descriptor_ != -1) {
            return -EINVAL;
        }

        int descriptor = -1;
        if (config.path) {
            const int length =
                std::snprintf(path_.data(), path_.size(), "%s",
                              config.path);
            if (length <= 0 ||
                static_cast<std::size_t>(length) >= path_.size()) {
                return -ENAMETOOLONG;
            }
            descriptor = ::open(
                path_.data(),
                O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        } else {
            const int length = std::snprintf(
                path_.data(), path_.size(),
                "/tmp/kcoro-fatal-XXXXXX");
            if (length <= 0 ||
                static_cast<std::size_t>(length) >= path_.size()) {
                return -ENAMETOOLONG;
            }
            descriptor = ::mkstemp(path_.data());
            if (descriptor >= 0) {
                const int flags = ::fcntl(descriptor, F_GETFD);
                if (flags < 0 ||
                    ::fcntl(descriptor, F_SETFD,
                            flags | FD_CLOEXEC) != 0) {
                    const int error = errno;
                    ::close(descriptor);
                    ::unlink(path_.data());
                    path_[0] = '\0';
                    return -error;
                }
            }
        }
        if (descriptor < 0) {
            const int error = errno;
            path_[0] = '\0';
            return -error;
        }
        if (::ftruncate(descriptor,
                        static_cast<off_t>(bytes)) != 0) {
            const int error = errno;
            ::close(descriptor);
            ::unlink(path_.data());
            path_[0] = '\0';
            return -error;
        }
        void *mapping = ::mmap(
            nullptr, bytes, PROT_READ | PROT_WRITE,
            MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) {
            const int error = errno;
            ::close(descriptor);
            ::unlink(path_.data());
            path_[0] = '\0';
            return -error;
        }
        std::memset(mapping, 0, bytes);
        Header *header = ::new (mapping) Header();
        header->magic = config.magic;
        header->runtime_epoch = config.runtime_epoch;
        if (::mlock(mapping, bytes) != 0) {
            const int error = errno;
            ::munmap(mapping, bytes);
            ::close(descriptor);
            ::unlink(path_.data());
            path_[0] = '\0';
            return -error;
        }
        descriptor_ = descriptor;
        mapping_ = mapping;
        return 0;
#endif
    }

    void publish(const Record &record) noexcept {
#if defined(_WIN32)
        (void)record;
        std::abort();
#else
        if (mapping_ == MAP_FAILED || descriptor_ < 0)
            std::abort();
        Header *header = static_cast<Header *>(mapping_);
        Record *destination = reinterpret_cast<Record *>(
            static_cast<unsigned char *>(mapping_) + sizeof(Header));
        std::memcpy(destination, &record, sizeof(record));
        header->checksum = checksum(*destination);
        std::atomic_ref<std::uint32_t>(header->publication).store(
            committed, std::memory_order_release);
#endif
    }

    void destroy() noexcept {
#if !defined(_WIN32)
        if (mapping_ != MAP_FAILED) {
            (void)::munlock(mapping_, bytes);
            (void)::munmap(mapping_, bytes);
            mapping_ = MAP_FAILED;
        }
        if (descriptor_ >= 0) {
            (void)::close(descriptor_);
            descriptor_ = -1;
        }
        if (path_[0] != '\0') {
            (void)::unlink(path_.data());
            path_[0] = '\0';
        }
#endif
    }

    [[nodiscard]] bool ready() const noexcept {
#if defined(_WIN32)
        return false;
#else
        return mapping_ != MAP_FAILED && descriptor_ >= 0;
#endif
    }

    [[nodiscard]] const char *path() const noexcept {
#if defined(_WIN32)
        return "";
#else
        return path_.data();
#endif
    }

    [[nodiscard]] static std::uint64_t checksum(
        const Record &record) noexcept {
        const auto *data =
            reinterpret_cast<const unsigned char *>(&record);
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (std::size_t index = 0; index < sizeof(record); ++index) {
            hash ^= data[index];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    static constexpr std::size_t bytes =
        sizeof(Header) + sizeof(Record);

  private:
#if !defined(_WIN32)
    int descriptor_ = -1;
    void *mapping_ = MAP_FAILED;
    std::array<char, PATH_MAX> path_{};
#endif
};

} // namespace kc
