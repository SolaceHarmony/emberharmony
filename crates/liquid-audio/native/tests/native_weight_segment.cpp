// Native shared-weight single-flight test.
//
// Every ownership transition under test lives here: eight
// literal kcoro continuations rendezvous by correlated callbacks, one builds
// the named wired segment, seven dehydrate on BUILDING, and READY resumes the
// exact logical frames. No operation owns a pthread, condition variable,
// polling loop, sleep, or synchronous wait beside the segment.

#include "kc_runtime.h"
#include "kcoro_stackless.h"
#include "lfm_payload_reader.h"
#include "lfm_safetensors.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

constexpr uint32_t kOpenCount = 8;

struct OpenFrame {
    uint32_t index{0};
    uint32_t status{0};
    int32_t resume_status{0};
    uint32_t suspensions{0};
    uint32_t first_worker{UINT32_MAX};
    uint32_t resumed_worker{UINT32_MAX};
    LfmWeightImage *image{nullptr};
    LfmWeightLoadStats stats{};
    LfmTensorView view{};
    char error[256]{};
};

struct OpenHarness {
    const char *path{nullptr};
    LfmPayloadReadOwner owner{};
    std::atomic<uint32_t> arrived{0};
    std::array<koro_cont_t *, kOpenCount> continuations{};
    std::array<kc_ticket_id, kOpenCount> identities{};
};

int payload_begin(void *, uint32_t, uint64_t) { return 0; }
int payload_record(void *, uint32_t, uint64_t) { return 0; }
void payload_end(void *) {}

bool digest_empty(const uint8_t digest[32]) {
    uint8_t value = 0;
    for (size_t index = 0; index < 32; ++index) value |= digest[index];
    return value == 0;
}

void *finish(koro_cont_t *continuation) {
    return koro_cont_finish(continuation) ? reinterpret_cast<void *>(1)
                                          : nullptr;
}

void *open_step(koro_cont_t *continuation) {
    auto *harness = static_cast<OpenHarness *>(
        koro_cont_argument(continuation));
    auto *frame = static_cast<OpenFrame *>(koro_cont_frame(continuation));
    if (!harness || !frame) std::abort();

    switch (koro_cont_state_get(continuation)) {
    case 0: {
        frame->first_worker = koro_cont_current_worker(continuation);
        const uint32_t arrived =
            harness->arrived.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (arrived < kOpenCount) {
            koro_cont_state_set(continuation, 1, KORO_SUSPEND_CALLBACK);
            return nullptr;
        }
        /* The last arrival is the callback for this setup-only rendezvous.
         * It names every earlier frame exactly; wake-before-suspend is closed
         * by kcoro's retained wake bit. The last frame continues directly. */
        for (uint32_t index = 0; index < kOpenCount; ++index) {
            if (harness->continuations[index] == continuation) continue;
            frame->resume_status =
                koro_cont_resume(harness->continuations[index],
                                 &harness->identities[index]);
            if (frame->resume_status != 0) {
                frame->status = static_cast<uint32_t>(EFAULT);
                return finish(continuation);
            }
        }
        [[fallthrough]];
    }
    case 1:
        frame->status = static_cast<uint32_t>(
            lfm_weights_open_owned_continuation(
                harness->path, &harness->owner, continuation,
                &frame->image, frame->error, sizeof(frame->error)));
        if (static_cast<int32_t>(frame->status) == LFM_WEIGHT_IN_PROGRESS) {
            ++frame->suspensions;
            koro_cont_state_set(continuation, 2, KORO_SUSPEND_CALLBACK);
            return nullptr;
        }
        break;
    case 2:
        frame->resumed_worker = koro_cont_current_worker(continuation);
        frame->status = static_cast<uint32_t>(
            lfm_weights_open_owned_continuation(
                harness->path, &harness->owner, continuation,
                &frame->image, frame->error, sizeof(frame->error)));
        break;
    default:
        std::abort();
    }

    if (static_cast<int32_t>(frame->status) != LFM_WEIGHT_OK ||
        !frame->image) return finish(continuation);
    frame->status = static_cast<uint32_t>(
        lfm_weights_load_stats(frame->image, &frame->stats));
    if (static_cast<int32_t>(frame->status) != LFM_WEIGHT_OK) {
        return finish(continuation);
    }
    frame->status = static_cast<uint32_t>(
        lfm_weights_find(frame->image, "weight", &frame->view));
    return finish(continuation);
}

void set_error(char *error, size_t bytes, const char *message) {
    if (!error || bytes == 0) return;
    std::snprintf(error, bytes, "%s", message);
}

} // namespace

int run_singleflight(const char *path, char *error, size_t error_length) {
    if (error && error_length) error[0] = '\0';
    if (!path || !path[0]) {
        set_error(error, error_length,
                  "invalid native single-flight test path");
        return EINVAL;
    }
    uint32_t built = 0;
    uint32_t attached = 0;
    uint32_t reused = 0;
    uint32_t suspended = 0;

    OpenHarness harness{
        .path = path,
        .owner = {
            .context = nullptr,
            .begin = payload_begin,
            .record = payload_record,
            .end = payload_end,
        },
    };
    kc_runtime_t *runtime = nullptr;
    const kc_runtime_config runtime_config{
        .worker_count = kOpenCount,
    };
    int status = kc_runtime_create(&runtime_config, &runtime);
    if (status != 0) {
        set_error(error, error_length,
                  "cannot create kcoro weight-test runtime");
        return status;
    }

    for (uint32_t index = 0; index < kOpenCount; ++index) {
        const koro_cont_config config{
            .step = open_step,
            .argument = &harness,
            .frame_size = sizeof(OpenFrame),
            .worker_mask = 0,
            .completion = nullptr,
            .completion_context = nullptr,
        };
        status = koro_cont_create_on(runtime, &config,
                                     &harness.continuations[index]);
        if (status != 0) break;
        auto *frame = static_cast<OpenFrame *>(
            koro_cont_frame(harness.continuations[index]));
        frame->index = index;
        harness.identities[index] =
            koro_cont_identity(harness.continuations[index]);
    }
    if (status == 0) status = kc_runtime_start(runtime);
    if (status == 0) {
        for (koro_cont_t *continuation : harness.continuations) {
            status = koro_cont_start(continuation);
            if (status != 0) break;
        }
    }
    if (status == 0) status = kc_runtime_join_all(runtime);

    const void *base = nullptr;
    uint64_t view_bytes = 0;
    std::array<uint8_t, 32> identity{};
    std::array<uint8_t, 32> content{};
    for (koro_cont_t *continuation : harness.continuations) {
        if (!continuation) continue;
        auto *frame = static_cast<OpenFrame *>(koro_cont_frame(continuation));
        if (status == 0 && static_cast<int32_t>(frame->status) != LFM_WEIGHT_OK) {
            status = static_cast<int32_t>(frame->status);
            if (frame->resume_status != 0) {
                std::snprintf(error, error_length,
                              "continuation %u resume failed: %d",
                              frame->index, frame->resume_status);
            } else {
                std::snprintf(
                    error, error_length,
                    "continuation %u weight open failed: %s",
                    frame->index,
                    frame->error[0] ? frame->error : "no diagnostic");
            }
        }
        if (status == 0 &&
            (!(frame->stats.flags & LFM_WEIGHT_LOAD_WIRED) ||
             digest_empty(frame->stats.identity_digest) ||
             digest_empty(frame->stats.content_digest) ||
             !frame->view.data || frame->view.bytes == 0)) {
            status = EPROTO;
            set_error(error, error_length,
                      "one weight-open continuation published incomplete evidence");
        }
        if (frame->stats.flags & LFM_WEIGHT_LOAD_BUILT) ++built;
        if (frame->stats.flags & LFM_WEIGHT_LOAD_ATTACHED) ++attached;
        if (frame->stats.flags & LFM_WEIGHT_LOAD_REGISTRY_REUSED) ++reused;
        if (status == 0 &&
            ((frame->stats.flags & LFM_WEIGHT_LOAD_BUILT) != 0) !=
                (frame->stats.segment_constructed_bytes != 0)) {
            status = EPROTO;
            set_error(error, error_length,
                      "builder byte ownership did not match its disposition");
        }
        if (status == 0 &&
            ((frame->stats.flags & LFM_WEIGHT_LOAD_ATTACHED) != 0) !=
                (frame->stats.attached_shared_bytes != 0)) {
            status = EPROTO;
            set_error(error, error_length,
                      "attacher byte ownership did not match its disposition");
        }
        if (status == 0 &&
            (frame->stats.flags & LFM_WEIGHT_LOAD_REGISTRY_REUSED) != 0 &&
            (frame->stats.segment_constructed_bytes != 0 ||
             frame->stats.attached_shared_bytes != 0)) {
            status = EPROTO;
            set_error(error, error_length,
                      "registry lease falsely reported a new mapping");
        }
        suspended += frame->suspensions;
        if (!base && frame->image) {
            base = lfm_weights_data(frame->image);
            view_bytes = frame->view.bytes;
            std::memcpy(identity.data(), frame->stats.identity_digest,
                        identity.size());
            std::memcpy(content.data(), frame->stats.content_digest,
                        content.size());
        } else if (status == 0 && frame->image &&
                   (lfm_weights_data(frame->image) != base ||
                    frame->view.bytes != view_bytes ||
                    std::memcmp(frame->stats.identity_digest, identity.data(),
                                identity.size()) != 0 ||
                    std::memcmp(frame->stats.content_digest, content.data(),
                                content.size()) != 0)) {
            status = EPROTO;
            set_error(error, error_length,
                      "single-flight handles did not retain one identical image");
        }
    }
    if (status == 0 &&
        (built != 1 || attached != 0 || reused != kOpenCount - 1 ||
         suspended != kOpenCount - 1)) {
        status = EDOM;
        set_error(error, error_length,
                  "single-flight test did not observe one build and seven correlated registry leases");
    }

    for (koro_cont_t *continuation : harness.continuations) {
        if (!continuation) continue;
        auto *frame = static_cast<OpenFrame *>(koro_cont_frame(continuation));
        lfm_weights_close(frame->image);
        frame->image = nullptr;
        const int destroyed = koro_cont_destroy(continuation);
        if (status == 0 && destroyed != 0) status = destroyed;
    }
    if (!digest_empty(identity.data())) {
        char ignored[1]{};
        const int evicted = lfm_weights_evict(identity.data(), ignored,
                                               sizeof(ignored));
        if (status == 0 && evicted != LFM_WEIGHT_OK) status = evicted;
    }
    kc_runtime_request_stop(runtime);
    const int joined = kc_runtime_join(runtime);
    if (status == 0 && joined != 0) status = joined;
    const int destroyed = kc_runtime_destroy(runtime);
    if (status == 0 && destroyed != 0) status = destroyed;
    if (status == 0) {
        std::fprintf(stderr,
                     "native weight single-flight test: built=%u "
                     "attached=%u reused=%u suspended=%u\n",
                     built, attached, reused, suspended);
    }
    return status;
}

int write_fixture(const char *path) {
    constexpr uint64_t payload_bytes = UINT64_C(32) * 1024 * 1024;
    const std::string json =
        "{\"weight\":{\"dtype\":\"U8\",\"shape\":[" +
        std::to_string(payload_bytes) +
        "],\"data_offsets\":[0," + std::to_string(payload_bytes) + "]}}";
    const size_t header_bytes = (json.size() + 7u) & ~size_t{7u};
    std::FILE *file = std::fopen(path, "wb");
    if (!file) return errno;
    const uint64_t little_header = header_bytes;
    int status =
        std::fwrite(&little_header, sizeof(little_header), 1, file) == 1
            ? 0
            : EIO;
    if (status == 0 &&
        std::fwrite(json.data(), 1, json.size(), file) != json.size()) {
        status = EIO;
    }
    for (size_t index = json.size(); status == 0 && index < header_bytes;
         ++index) {
        if (std::fputc(' ', file) == EOF) status = EIO;
    }
    std::array<unsigned char, 64 * 1024> payload{};
    payload.fill(0x5a);
    for (uint64_t offset = 0; status == 0 && offset < payload_bytes;
         offset += payload.size()) {
        const size_t count = static_cast<size_t>(
            std::min<uint64_t>(payload.size(), payload_bytes - offset));
        if (std::fwrite(payload.data(), 1, count, file) != count) {
            status = EIO;
        }
    }
    if (std::fclose(file) != 0 && status == 0) status = errno;
    return status;
}

int main() {
    char path[] = "/tmp/lfm-weight-singleflight-XXXXXX";
    const int descriptor = mkstemp(path);
    if (descriptor < 0) {
        std::perror("mkstemp");
        return EXIT_FAILURE;
    }
    close(descriptor);
    int status = write_fixture(path);
    char error[512]{};
    if (status == 0) status = run_singleflight(path, error, sizeof(error));
    unlink(path);
    if (status != 0) {
        std::fprintf(stderr, "native weight single-flight test failed (%d): %s\n",
                     status, error[0] ? error : "no diagnostic");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
