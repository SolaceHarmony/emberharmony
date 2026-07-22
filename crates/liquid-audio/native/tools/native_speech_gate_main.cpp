// Standalone native launcher for the complete in-memory speech truth gate.
//
// The gate itself lives in native/tests/native_speech_to_speech.cpp and owns
// both model conversations, PCM, kcoro continuations, callbacks, watchdog, and
// evidence. This file supplies only command-line policy; no Rust runtime or
// model implementation participates in the executable.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int lfm_native_speech_to_speech_gate(
    const char *model_path, uint32_t audible, uint32_t kernel_lanes,
    char *error, size_t error_length);

namespace {

int parse_lanes(const char *text, uint32_t *lanes) {
    if (!text || !lanes || !text[0]) return EINVAL;
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value == 0 ||
        value > UINT32_MAX) {
        return EINVAL;
    }
    *lanes = static_cast<uint32_t>(value);
    return 0;
}

int parse_audible(const char *text, uint32_t *audible) {
    if (!text || !audible) return EINVAL;
    if (std::strcmp(text, "silent") == 0) {
        *audible = 0;
        return 0;
    }
    if (std::strcmp(text, "buffered") == 0) {
        *audible = 1;
        return 0;
    }
    if (std::strcmp(text, "stream") == 0) {
        *audible = 2;
        return 0;
    }
    return EINVAL;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s CHECKPOINT [KERNEL_LANES] "
                     "[silent|buffered|stream]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }
    uint32_t lanes = 8;
    uint32_t audible = 0;
    if ((argc >= 3 && parse_lanes(argv[2], &lanes) != 0) ||
        (argc == 4 && parse_audible(argv[3], &audible) != 0)) {
        std::fprintf(stderr, "invalid native speech-gate policy\n");
        return EXIT_FAILURE;
    }
    char error[1024]{};
    const int status = lfm_native_speech_to_speech_gate(
        argv[1], audible, lanes, error, sizeof(error));
    if (status != 0) {
        std::fprintf(stderr, "native speech gate failed (%d): %s\n", status,
                     error[0] ? error : "no diagnostic");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
