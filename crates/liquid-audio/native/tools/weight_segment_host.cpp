// Standalone native control surface for the immutable shared weight segment.
//
// This is intentionally a C++23 process, not a Rust loader or a shell-script
// wrapper. `open` proves attach-or-build and exits. `serve` starts the native
// kcoro/Mach mailbox host; no legacy keeper or second inference path remains.

#include "lfm_host_mailbox.h"
#include "lfm_safetensors.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

struct Keeper {
    LfmWeightImage *image{nullptr};
};

const char *disposition(uint32_t flags) {
    if (flags & LFM_WEIGHT_LOAD_BUILT) return "built";
    if (flags & LFM_WEIGHT_LOAD_ATTACHED) return "attached";
    return "invalid";
}

void hex(const uint8_t digest[32], char output[65]) {
    static constexpr char alphabet[] = "0123456789abcdef";
    for (size_t index = 0; index < 32; ++index) {
        output[index * 2] = alphabet[digest[index] >> 4];
        output[index * 2 + 1] = alphabet[digest[index] & 15];
    }
    output[64] = '\0';
}

int open_image(const fs::path &root, LfmWeightImage **out,
               char *error, size_t error_length) {
    const fs::path detokenizer = root / "audio_detokenizer";
    if (fs::is_regular_file(detokenizer / "model.safetensors")) {
        const std::string main = root.string();
        const std::string audio = detokenizer.string();
        return lfm_weights_open_bundle(main.c_str(), audio.c_str(), out,
                                       error, error_length);
    }
    const std::string path = root.string();
    return lfm_weights_open(path.c_str(), out, error, error_length);
}

int report(const LfmWeightImage *image,
           LfmWeightLoadStatsV2 *evidence = nullptr) {
    LfmWeightLoadStatsV2 stats{
        .size = sizeof(stats),
        .abi_version = LFM_WEIGHT_ABI_VERSION,
    };
    const int status = lfm_weights_load_stats(image, &stats);
    if (status != LFM_WEIGHT_OK) return status;
    char identity[65]{};
    char content[65]{};
    hex(stats.identity_digest, identity);
    hex(stats.content_digest, content);
    std::printf(
        "{\"state\":\"ready\",\"disposition\":\"%s\","
        "\"source_bytes\":%llu,\"segment_bytes\":%llu,"
        "\"segment_constructed_bytes\":%llu,"
        "\"attached_shared_bytes\":%llu,\"wired_bytes\":%llu,"
        "\"payload_read_calls\":%llu,\"payload_read_bytes\":%llu,"
        "\"generation\":%llu,"
        "\"identity\":\"%s\","
        "\"content\":\"%s\"}\n",
        disposition(stats.flags),
        static_cast<unsigned long long>(stats.source_bytes),
        static_cast<unsigned long long>(stats.segment_bytes),
        static_cast<unsigned long long>(stats.segment_constructed_bytes),
        static_cast<unsigned long long>(stats.attached_shared_bytes),
        static_cast<unsigned long long>(stats.wired_bytes),
        static_cast<unsigned long long>(stats.payload_read_calls),
        static_cast<unsigned long long>(stats.payload_read_bytes),
        static_cast<unsigned long long>(stats.generation), identity, content);
    std::fflush(stdout);
    if (evidence) *evidence = stats;
    return LFM_WEIGHT_OK;
}

int parse_digest(const char *text,
                 lfm::host::CheckpointIdentity *identity) {
    if (!text || std::strlen(text) != 64) return EINVAL;
    auto *digest = reinterpret_cast<uint8_t *>(identity);
    for (size_t index = 0; index < sizeof(*identity); ++index) {
        char pair[3] = {text[index * 2], text[index * 2 + 1], '\0'};
        char *end = nullptr;
        const unsigned long value = std::strtoul(pair, &end, 16);
        if (!end || *end != '\0' || value > 255) return EINVAL;
        digest[index] = static_cast<uint8_t>(value);
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 3 && std::strcmp(argv[1], "evict") == 0) {
        lfm::host::CheckpointIdentity identity{};
        if (parse_digest(argv[2], &identity) != 0) {
            std::fprintf(stderr, "evict requires one 64-digit identity digest\n");
            return EXIT_FAILURE;
        }
        char error[1024]{};
        const int status = lfm_weights_evict(
            reinterpret_cast<const uint8_t *>(&identity), error,
            sizeof(error));
        if (status != LFM_WEIGHT_OK) {
            std::fprintf(stderr, "%s\n", error);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    const bool serve_command =
        argc >= 2 && std::strcmp(argv[1], "serve") == 0;
    if ((argc != 3 && !(serve_command && argc == 4)) ||
        (std::strcmp(argv[1], "open") != 0 &&
         std::strcmp(argv[1], "build") != 0 &&
         std::strcmp(argv[1], "attach") != 0 &&
         std::strcmp(argv[1], "serve") != 0)) {
        std::fprintf(stderr,
                     "usage: %s open|build|attach CHECKPOINT\n"
                     "       %s serve CHECKPOINT MACH_SERVICE\n"
                     "       %s evict IDENTITY_SHA256\n",
                     argv[0], argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    if (serve_command) {
        lfm::host::ServerConfig config{
            .checkpoint = argv[2],
            .service = argv[3],
        };
        std::string error;
        const lfm::host::Status status = lfm::host::serve(config, &error);
        if (status != lfm::host::Ok) {
            std::fprintf(stderr, "native host failed (%d): %s\n",
                         static_cast<int>(status), error.c_str());
        }
        return status == lfm::host::Ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    Keeper keeper;
    char error[1024]{};
    const int opened = open_image(fs::path(argv[2]), &keeper.image,
                                  error, sizeof(error));
    if (opened != LFM_WEIGHT_OK) {
        std::fprintf(stderr, "native weight open failed (%d): %s\n",
                     opened, error);
        return EXIT_FAILURE;
    }
    LfmWeightLoadStatsV2 stats{};
    const int reported = report(keeper.image, &stats);
    if (reported != LFM_WEIGHT_OK) {
        lfm_weights_close(keeper.image);
        return EXIT_FAILURE;
    }
    if (std::strcmp(argv[1], "build") == 0 &&
        !(stats.flags & LFM_WEIGHT_LOAD_BUILT)) {
        std::fprintf(stderr, "expected a build but attached generation %llu\n",
                     static_cast<unsigned long long>(stats.generation));
        lfm_weights_close(keeper.image);
        return EXIT_FAILURE;
    }
    if (std::strcmp(argv[1], "attach") == 0 &&
        (!(stats.flags & LFM_WEIGHT_LOAD_ATTACHED) ||
         stats.payload_read_calls != 0 || stats.payload_read_bytes != 0)) {
        std::fprintf(stderr, "expected a zero-payload attach\n");
        lfm_weights_close(keeper.image);
        return EXIT_FAILURE;
    }
    if (std::strcmp(argv[1], "open") == 0 ||
        std::strcmp(argv[1], "build") == 0 ||
        std::strcmp(argv[1], "attach") == 0) {
        lfm_weights_close(keeper.image);
        return EXIT_SUCCESS;
    }
    lfm_weights_close(keeper.image);
    return EXIT_FAILURE;
}
