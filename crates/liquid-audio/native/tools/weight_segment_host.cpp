// Standalone native keeper for the immutable shared weight segment.
//
// This is intentionally a C++23 process, not a Rust loader or a shell-script
// wrapper. `open` proves attach-or-build and exits. `host` retains the exact
// wired lease and gives process lifetime to the mapping; GCD owns signal
// dormancy and invokes the teardown callback. No thread waits on a condition,
// polls a flag, or sleeps beside the model.

#include "lfm_safetensors.h"

#include <array>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <spawn.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace fs = std::filesystem;

namespace {

struct Keeper {
    LfmWeightImage *image{nullptr};
#ifdef __APPLE__
    dispatch_source_t terminate{nullptr};
    dispatch_source_t interrupt{nullptr};
#endif
};

struct Ticket {
    uint64_t epoch{0};
    uint64_t sequence{0};
    uint32_t generation{0};
    uint32_t kind{0};
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

int report(const LfmWeightImage *image, LfmWeightLoadStatsV2 *evidence = nullptr,
           const Ticket *ticket = nullptr) {
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
        "\"generation\":%llu,\"ticket\":{\"epoch\":%llu,"
        "\"sequence\":%llu,\"generation\":%u,\"kind\":%u},"
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
        static_cast<unsigned long long>(stats.generation),
        static_cast<unsigned long long>(ticket ? ticket->epoch : 0),
        static_cast<unsigned long long>(ticket ? ticket->sequence : 0),
        ticket ? ticket->generation : 0, ticket ? ticket->kind : 0,
        identity, content);
    std::fflush(stdout);
    if (evidence) *evidence = stats;
    return LFM_WEIGHT_OK;
}

#ifdef __APPLE__
void stop(void *context) {
    auto *keeper = static_cast<Keeper *>(context);
    lfm_weights_close(keeper->image);
    keeper->image = nullptr;
    std::_Exit(EXIT_SUCCESS);
}

int host(Keeper *keeper) {
    std::signal(SIGTERM, SIG_IGN);
    std::signal(SIGINT, SIG_IGN);
    const dispatch_queue_t queue =
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    keeper->terminate = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0, queue);
    keeper->interrupt = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_SIGNAL, SIGINT, 0, queue);
    if (!keeper->terminate || !keeper->interrupt) return ENOMEM;
    dispatch_set_context(keeper->terminate, keeper);
    dispatch_set_context(keeper->interrupt, keeper);
    dispatch_source_set_event_handler_f(keeper->terminate, stop);
    dispatch_source_set_event_handler_f(keeper->interrupt, stop);
    dispatch_activate(keeper->terminate);
    dispatch_activate(keeper->interrupt);
    dispatch_main();
}

int spawn_host(const char *program, const char *checkpoint,
               pid_t *pid, char *ready, size_t ready_bytes) {
    int pipefd[2]{};
    if (pipe(pipefd) != 0) return errno;
    posix_spawn_file_actions_t actions{};
    int status = posix_spawn_file_actions_init(&actions);
    if (status == 0)
        status = posix_spawn_file_actions_adddup2(
            &actions, pipefd[1], STDOUT_FILENO);
    if (status == 0)
        status = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    if (status == 0)
        status = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    char *const arguments[] = {
        const_cast<char *>(program), const_cast<char *>("host"),
        const_cast<char *>(checkpoint),
        const_cast<char *>("1:1:1:8"), nullptr,
    };
    if (status == 0)
        status = posix_spawn(pid, program, &actions, nullptr,
                             arguments, environ);
    (void)posix_spawn_file_actions_destroy(&actions);
    (void)::close(pipefd[1]);
    if (status != 0) {
        (void)::close(pipefd[0]);
        return status;
    }
    size_t used = 0;
    while (used + 1 < ready_bytes) {
        const ssize_t count = ::read(pipefd[0], ready + used, 1);
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            status = errno;
            break;
        }
        if (ready[used++] == '\n') break;
    }
    ready[used] = '\0';
    (void)::close(pipefd[0]);
    return status;
}

int spawn_and_wait(const char *program, const char *verb,
                   const char *checkpoint) {
    pid_t pid = 0;
    char *const arguments[] = {
        const_cast<char *>(program), const_cast<char *>(verb),
        const_cast<char *>(checkpoint), nullptr,
    };
    const int spawned = posix_spawn(&pid, program, nullptr, nullptr,
                                    arguments, environ);
    if (spawned != 0) return spawned;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        return errno;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : ECHILD;
}

uint64_t wired_pages() {
    std::array<uint8_t, sizeof(uint64_t)> storage{};
    size_t bytes = storage.size();
    if (sysctlbyname("vm.page_wired_count", storage.data(), &bytes,
                     nullptr, 0) != 0) return 0;
    if (bytes == sizeof(uint32_t)) {
        uint32_t value = 0;
        std::memcpy(&value, storage.data(), sizeof(value));
        return value;
    }
    if (bytes == sizeof(uint64_t)) {
        uint64_t value = 0;
        std::memcpy(&value, storage.data(), sizeof(value));
        return value;
    }
    return 0;
}

int verify(const char *program, const fs::path &root) {
    char error[1024]{};
    LfmWeightImage *prior = nullptr;
    int status = open_image(root, &prior, error, sizeof(error));
    if (status != LFM_WEIGHT_OK) return status;
    LfmWeightLoadStatsV2 prior_stats{};
    status = report(prior, &prior_stats);
    lfm_weights_close(prior);
    if (status != LFM_WEIGHT_OK) return status;
    status = lfm_weights_evict(prior_stats.identity_digest,
                               error, sizeof(error));
    if (status != LFM_WEIGHT_OK) return status;
    const uint64_t before_pages = wired_pages();
    const long page_size = sysconf(_SC_PAGESIZE);
    if (before_pages == 0 || page_size <= 0) return ENOTSUP;

    pid_t keeper = 0;
    char ready[2048]{};
    const std::string path = root.string();
    status = spawn_host(program, path.c_str(), &keeper, ready, sizeof(ready));
    if (status != 0 || !std::strstr(ready, "\"disposition\":\"built\"") ||
        !std::strstr(ready,
                     "\"ticket\":{\"epoch\":1,\"sequence\":1,"
                     "\"generation\":1,\"kind\":8}")) {
        if (keeper > 0) (void)::kill(keeper, SIGTERM);
        return status != 0 ? status : EPROTO;
    }
    const uint64_t keeper_pages = wired_pages();
    const uint64_t expected_pages =
        (prior_stats.segment_bytes + static_cast<uint64_t>(page_size) - 1) /
        static_cast<uint64_t>(page_size);
    if (keeper_pages < before_pages ||
        keeper_pages - before_pages < expected_pages * 9 / 10) {
        (void)::kill(keeper, SIGTERM);
        return ENOMEM;
    }

    LfmWeightImage *client = nullptr;
    status = open_image(root, &client, error, sizeof(error));
    LfmWeightLoadStatsV2 attached{};
    if (status == LFM_WEIGHT_OK) status = report(client, &attached);
    if (status == LFM_WEIGHT_OK &&
        (!(attached.flags & LFM_WEIGHT_LOAD_ATTACHED) ||
         attached.payload_read_calls != 0 ||
         attached.payload_read_bytes != 0 ||
         std::memcmp(attached.identity_digest,
                     prior_stats.identity_digest, 32) != 0)) {
        status = EPROTO;
    }
    const uint64_t client_pages = wired_pages();
    if (status == LFM_WEIGHT_OK &&
        client_pages > keeper_pages + expected_pages / 10 + 4096) {
        status = EOVERFLOW;
    }
    lfm_weights_close(client);

    (void)::kill(keeper, SIGTERM);
    int keeper_status = 0;
    while (waitpid(keeper, &keeper_status, 0) < 0) {
        if (errno == EINTR) continue;
        if (status == LFM_WEIGHT_OK) status = errno;
        break;
    }
    if (status == LFM_WEIGHT_OK &&
        (!WIFEXITED(keeper_status) || WEXITSTATUS(keeper_status) != 0)) {
        status = ECHILD;
    }
    const uint64_t retired_pages = wired_pages();
    if (status == LFM_WEIGHT_OK &&
        (retired_pages >= keeper_pages ||
         keeper_pages - retired_pages < expected_pages * 8 / 10)) {
        status = EBUSY;
    }
    if (status == LFM_WEIGHT_OK)
        status = spawn_and_wait(program, "attach", path.c_str());
    if (status == LFM_WEIGHT_OK)
        status = lfm_weights_evict(attached.identity_digest,
                                   error, sizeof(error));
    if (status == LFM_WEIGHT_OK) {
        std::printf(
            "{\"state\":\"verified\",\"restart_attach\":true,"
            "\"simultaneous_keeper_attach\":true,"
            "\"payload_reads_on_attach\":0,"
            "\"keeper_wired_page_delta\":%llu,"
            "\"client_extra_wired_pages\":%llu,"
            "\"retired_wired_page_delta\":%llu}\n",
            static_cast<unsigned long long>(keeper_pages - before_pages),
            static_cast<unsigned long long>(
                client_pages > keeper_pages ? client_pages - keeper_pages : 0),
            static_cast<unsigned long long>(
                keeper_pages > retired_pages ? keeper_pages - retired_pages : 0));
    }
    return status;
}
#endif

int parse_digest(const char *text, std::array<uint8_t, 32> *digest) {
    if (!text || std::strlen(text) != 64) return EINVAL;
    for (size_t index = 0; index < digest->size(); ++index) {
        char pair[3] = {text[index * 2], text[index * 2 + 1], '\0'};
        char *end = nullptr;
        const unsigned long value = std::strtoul(pair, &end, 16);
        if (!end || *end != '\0' || value > 255) return EINVAL;
        (*digest)[index] = static_cast<uint8_t>(value);
    }
    return 0;
}

int parse_ticket(const char *text, Ticket *ticket) {
    if (!text || !ticket) return EINVAL;
    char *end = nullptr;
    ticket->epoch = std::strtoull(text, &end, 10);
    if (!end || *end != ':') return EINVAL;
    ticket->sequence = std::strtoull(end + 1, &end, 10);
    if (!end || *end != ':') return EINVAL;
    const unsigned long generation = std::strtoul(end + 1, &end, 10);
    if (!end || *end != ':' || generation > UINT32_MAX) return EINVAL;
    const unsigned long kind = std::strtoul(end + 1, &end, 10);
    if (!end || *end != '\0' || kind > UINT32_MAX || ticket->epoch == 0 ||
        ticket->sequence == 0 || generation == 0 || kind == 0) return EINVAL;
    ticket->generation = static_cast<uint32_t>(generation);
    ticket->kind = static_cast<uint32_t>(kind);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 3 && std::strcmp(argv[1], "evict") == 0) {
        std::array<uint8_t, 32> identity{};
        if (parse_digest(argv[2], &identity) != 0) {
            std::fprintf(stderr, "evict requires one 64-digit identity digest\n");
            return EXIT_FAILURE;
        }
        char error[1024]{};
        const int status = lfm_weights_evict(identity.data(), error,
                                              sizeof(error));
        if (status != LFM_WEIGHT_OK) {
            std::fprintf(stderr, "%s\n", error);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    const bool host_command = argc >= 2 && std::strcmp(argv[1], "host") == 0;
    if ((argc != 3 && !(host_command && argc == 4)) ||
        (std::strcmp(argv[1], "open") != 0 &&
         std::strcmp(argv[1], "build") != 0 &&
         std::strcmp(argv[1], "attach") != 0 &&
         std::strcmp(argv[1], "host") != 0 &&
         std::strcmp(argv[1], "verify") != 0)) {
        std::fprintf(stderr,
                     "usage: %s open|build|attach|verify CHECKPOINT\n"
                     "       %s host CHECKPOINT [EPOCH:SEQUENCE:GENERATION:KIND]\n"
                     "       %s evict IDENTITY_SHA256\n",
                     argv[0], argv[0], argv[0]);
        return EXIT_FAILURE;
    }
#ifdef __APPLE__
    if (std::strcmp(argv[1], "verify") == 0) {
        const int status = verify(argv[0], fs::path(argv[2]));
        if (status != 0)
            std::fprintf(stderr, "native cross-process gate failed: %s\n",
                         std::strerror(status));
        return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#endif
    Keeper keeper;
    Ticket ticket{};
    if (host_command && argc == 4 && parse_ticket(argv[3], &ticket) != 0) {
        std::fprintf(stderr, "invalid canonical readiness ticket\n");
        return EXIT_FAILURE;
    }
    char error[1024]{};
    const int opened = open_image(fs::path(argv[2]), &keeper.image,
                                  error, sizeof(error));
    if (opened != LFM_WEIGHT_OK) {
        std::fprintf(stderr, "native weight open failed (%d): %s\n",
                     opened, error);
        return EXIT_FAILURE;
    }
    LfmWeightLoadStatsV2 stats{};
    const int reported = report(keeper.image, &stats,
                                host_command && argc == 4 ? &ticket : nullptr);
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
#ifdef __APPLE__
    return host(&keeper);
#else
    std::fprintf(stderr,
                 "persistent native weight host is currently supported only "
                 "on macOS; this build will not substitute a waiter thread\n");
    lfm_weights_close(keeper.image);
    return EXIT_FAILURE;
#endif
}
