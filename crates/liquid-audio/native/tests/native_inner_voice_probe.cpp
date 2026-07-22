/* Inner-voice listening-probe spike gate.
 *
 * For each paired item the gate establishes a short text context turn, then
 * feeds a synthesized user utterance through the production audio admission
 * seam one adapted row per pass while sampling the greedy text head at every
 * row (lfm_internal_conversation_listen_probe_*_for_test). It records the
 * per-row token trajectory and wall-clock cost, appending raw rows to
 * <probe_dir>/raw_rows.csv and per-item records to <probe_dir>/raw_items.csv.
 * Every row also carries the dual-head sample-seam readout (text head and
 * Depthformer codebook-0 head: top-8 ids with natural-log probabilities plus
 * full-distribution entropy) into <probe_dir>/per_row_v2.csv, and the gate
 * hard-fails any row whose text top-1 disagrees with the sampled token.
 *
 * The gate is deliberately sequential: one runtime, one model, one fresh
 * conversation per item, every asynchronous seam driven by a futex doorbell.
 * There are no fallback paths; any missing input is a hard error. An empty
 * or null `item_filter` runs every item; otherwise only the named item. */

#include "lfm_model_internal.h"
#include "lfm_runtime.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t ABI = LFM_RUNTIME_ABI_VERSION;
constexpr uint32_t CAPTURE_RATE = 16000;
constexpr uint32_t PLAYBACK_RATE = 24000;
constexpr size_t ROW_CAPACITY = 4096;
constexpr uint32_t CONTEXT_EMISSION_CAP = 24;

struct ProbeItem {
    const char *name;
    const char *cls;
    const char *context;
    const char *wav_stem;
};

constexpr ProbeItem ITEMS[] = {
    {"p1r", "res", "The doves came back to the balcony this morning.", "p1r"},
    {"p1n", "nov", "The doves came back to the balcony this morning.", "p1n"},
    {"p2r", "res", "Heavy rain is supposed to start around noon.", "p2r"},
    {"p2n", "nov", "Heavy rain is supposed to start around noon.", "p2n"},
    {"p3r", "res", "Your sister's flight lands at six tonight.", "p3r"},
    {"p3n", "nov", "Your sister's flight lands at six tonight.", "p3n"},
    {"p4r", "res", "The espresso machine you ordered just arrived.", "p4r"},
    {"p4n", "nov", "The espresso machine you ordered just arrived.", "p4n"},
    {"p5r", "res", "Your library books are due back tomorrow.", "p5r"},
    {"p5n", "nov", "Your library books are due back tomorrow.", "p5n"},
    {"p6r", "res", "Dinner is confirmed for seven at the Italian place.",
     "p6r"},
    {"p6n", "nov", "Dinner is confirmed for seven at the Italian place.",
     "p6n"},
};

struct Doorbell {
    std::atomic<uint32_t> rung{0};

    void arm() { rung.store(0, std::memory_order_relaxed); }

    void wait() {
        while (rung.load(std::memory_order_acquire) == 0) {
            rung.wait(0, std::memory_order_acquire);
        }
    }
};

/* LfmAudioRouteNotify: nonblocking, allocation-free, no host callback. */
void ring_doorbell(void *context) {
    auto *bell = static_cast<Doorbell *>(context);
    bell->rung.store(1, std::memory_order_release);
    bell->rung.notify_all();
}

void copy_error(char *destination, size_t capacity, const char *source) {
    if (!destination || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", source ? source : "unknown");
}

int fail_step(char *error, size_t error_length, const char *step,
              int status) {
    char message[256]{};
    std::snprintf(message, sizeof(message), "%s failed: %d", step, status);
    copy_error(error, error_length, message);
    return status == 0 ? LFM_STATUS_INTERNAL : status;
}

int read_f32_file(const std::string &path, std::vector<float> *out) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) return -errno;
    int status = std::fseek(file, 0, SEEK_END);
    const long bytes = status == 0 ? std::ftell(file) : -1;
    if (status == 0) status = std::fseek(file, 0, SEEK_SET);
    if (status != 0 || bytes <= 0 ||
        bytes % static_cast<long>(sizeof(float)) != 0) {
        std::fclose(file);
        return -EINVAL;
    }
    out->resize(static_cast<size_t>(bytes) / sizeof(float));
    const size_t read = std::fread(out->data(), sizeof(float), out->size(),
                                   file);
    std::fclose(file);
    return read == out->size() ? 0 : -EIO;
}

LfmConversationOptionsV1 conversation_options(uint64_t seed) {
    return {
        .size = sizeof(LfmConversationOptionsV1),
        .abi_version = ABI,
        .flags = 0,
        .reserved0 = 0,
        .seed = seed,
        .text = {
            .size = sizeof(LfmSamplingPolicyV1),
            .abi_version = ABI,
            .flags = LFM_SAMPLING_GREEDY,
            .top_k = 1,
            .temperature = 0.0,
            .reserved = 0,
        },
        .audio = {
            .size = sizeof(LfmSamplingPolicyV1),
            .abi_version = ABI,
            .flags = LFM_SAMPLING_GREEDY,
            .top_k = 1,
            .temperature = 0.0,
            .reserved = 0,
        },
        .reserved = {},
    };
}

/* One short deterministic assistant reply establishes the context turn. The
 * model's interleave cadence flips to audio after interleaved_n_text tokens;
 * the loop stays on the text seam and closes the turn with the production
 * interrupt commit at the first audio boundary, the emission cap, or im_end. */
int run_context_turn(LfmConversation *conversation, const char *text,
                     Doorbell *bell, std::string *reply, char *error,
                     size_t error_length) {
    LfmNativeEmission emission{};
    LfmConversationAdmissionHandle admission{};
    bell->arm();
    int status = lfm_conversation_begin_text_submit_native(
        conversation, text, std::strlen(text), &emission, ring_doorbell,
        bell, &admission);
    if (status != 0) {
        return fail_step(error, error_length, "context begin submit", status);
    }
    bell->wait();
    status = lfm_conversation_begin_collect_native(conversation, &admission);
    if (status != 0) {
        return fail_step(error, error_length, "context begin collect", status);
    }
    bool finished = emission.kind == LFM_NATIVE_EMISSION_FINISHED;
    if (emission.kind == LFM_NATIVE_EMISSION_TEXT) {
        reply->append(reinterpret_cast<const char *>(emission.text),
                      emission.text_bytes);
    }
    uint32_t emitted = 1;
    while (!finished && emitted < CONTEXT_EMISSION_CAP) {
        const int playback =
            lfm_conversation_next_requires_playback_native(conversation);
        if (playback < 0) {
            return fail_step(error, error_length,
                             "context playback query", playback);
        }
        if (playback != 0) break;
        LfmAudioRouteHandle route{};
        bell->arm();
        status = lfm_conversation_next_submit_native(
            conversation, ring_doorbell, bell, &route);
        if (status != 0) {
            return fail_step(error, error_length, "context next submit",
                             status);
        }
        bell->wait();
        status = lfm_conversation_next_collect_native(conversation, &route,
                                                      &emission);
        if (status != 0) {
            return fail_step(error, error_length, "context next collect",
                             status);
        }
        ++emitted;
        if (emission.kind == LFM_NATIVE_EMISSION_TEXT) {
            reply->append(reinterpret_cast<const char *>(emission.text),
                          emission.text_bytes);
        }
        finished = emission.kind == LFM_NATIVE_EMISSION_FINISHED;
    }
    if (!finished) {
        LfmAudioRouteHandle route{};
        bell->arm();
        status = lfm_conversation_interrupt_submit_native(
            conversation, ring_doorbell, bell, &route);
        if (status != 0) {
            return fail_step(error, error_length, "context interrupt submit",
                             status);
        }
        bell->wait();
        status = lfm_conversation_interrupt_collect_native(conversation,
                                                           &route);
        if (status != 0) {
            return fail_step(error, error_length, "context interrupt collect",
                             status);
        }
    }
    return 0;
}

void csv_sanitize(std::string *text) {
    for (char &value : *text) {
        if (value == ',' || value == '\n' || value == '\r' ||
            value == '"') {
            value = ' ';
        }
    }
}

/* Readout floats print as shortest round-trip decimals so that identical
 * bit patterns always render as identical CSV bytes. */
void print_readout_head(std::FILE *csv, const uint32_t *ids,
                        const float *logprobs, float entropy) {
    for (size_t rank = 0; rank < LFM_LISTEN_READOUT_TOP_K; ++rank) {
        std::fprintf(csv, ",%u,%.9g", ids[rank],
                     static_cast<double>(logprobs[rank]));
    }
    std::fprintf(csv, ",%.9g", static_cast<double>(entropy));
}

} // namespace

extern "C" int lfm_native_inner_voice_probe_gate(
    const char *model_path, const char *probe_dir, const char *item_filter,
    uint32_t kernel_lanes, char *error, size_t error_length) {
    if (!model_path || !*model_path || !probe_dir || !*probe_dir ||
        kernel_lanes == 0 || !error || error_length == 0) {
        return LFM_STATUS_INVALID_ARGUMENT;
    }
    error[0] = '\0';

    LfmRuntime *runtime = nullptr;
    LfmModel *model = nullptr;
    const LfmRuntimeConfigV1 config = {
        .size = sizeof(LfmRuntimeConfigV1),
        .abi_version = ABI,
        .coordination_workers = 2,
        .kernel_lanes = kernel_lanes,
        .event_capacity = 64,
        .session_capacity = 2,
        .reserved0 = 0,
        .reserved1 = 0,
        .flags = 0,
        .reserved = {},
    };
    int status = lfm_runtime_create(&config, &runtime);
    if (status == 0) status = lfm_runtime_start(runtime);
    if (status == 0) {
        status = lfm_runtime_model_open(runtime, model_path, &model, error,
                                        error_length);
    }
    if (status != 0) {
        if (error[0] == '\0') {
            fail_step(error, error_length, "native runtime bring-up", status);
        }
        if (runtime) {
            lfm_runtime_request_stop(runtime);
            (void)lfm_runtime_join(runtime);
            (void)lfm_runtime_destroy(runtime);
        }
        return status;
    }

    const std::string directory(probe_dir);
    const std::string rows_path = directory + "/raw_rows.csv";
    const std::string items_path = directory + "/raw_items.csv";
    const std::string v2_path = directory + "/per_row_v2.csv";
    std::FILE *rows_csv = std::fopen(rows_path.c_str(), "w");
    std::FILE *items_csv = rows_csv ? std::fopen(items_path.c_str(), "w")
                                    : nullptr;
    std::FILE *v2_csv = items_csv ? std::fopen(v2_path.c_str(), "w")
                                  : nullptr;
    if (!rows_csv || !items_csv || !v2_csv) {
        if (rows_csv) std::fclose(rows_csv);
        if (items_csv) std::fclose(items_csv);
        copy_error(error, error_length,
                   "cannot create raw probe CSV outputs in probe_dir");
        status = -EIO;
    } else {
        std::fprintf(rows_csv, "item,class,row,token_id,row_ns\n");
        std::fprintf(items_csv,
                     "item,class,rows,samples,encode_ns,context_reply\n");
        std::fprintf(v2_csv, "item,class,row,token_id,row_ns");
        for (const char *head : {"text", "audio"}) {
            for (size_t rank = 1; rank <= LFM_LISTEN_READOUT_TOP_K;
                 ++rank) {
                std::fprintf(v2_csv, ",%s_top%zu_id,%s_top%zu_logprob",
                             head, rank, head, rank);
            }
            std::fprintf(v2_csv, ",%s_entropy", head);
        }
        std::fprintf(v2_csv, ",row_ms\n");
    }

    Doorbell bell;
    std::vector<float> pcm;
    std::vector<uint32_t> tokens(ROW_CAPACITY);
    std::vector<uint64_t> row_ns(ROW_CAPACITY);
    std::vector<LfmListenReadoutForTest> readouts(ROW_CAPACITY);
    size_t items_run = 0;
    uint64_t seed = 0x51d70001;
    for (const ProbeItem &item : ITEMS) {
        if (status != 0) break;
        if (item_filter && *item_filter &&
            std::strcmp(item_filter, item.name) != 0) {
            ++seed;
            continue;
        }
        ++items_run;
        const std::string wav = directory + "/" + item.wav_stem + ".f32";
        status = read_f32_file(wav, &pcm);
        if (status != 0) {
            char message[512]{};
            std::snprintf(message, sizeof(message),
                          "cannot read probe utterance %s: %d", wav.c_str(),
                          status);
            copy_error(error, error_length, message);
            break;
        }

        LfmConversation *conversation = nullptr;
        const LfmConversationOptionsV1 options = conversation_options(seed++);
        status = lfm_runtime_conversation_create(
            runtime, model, &options, &conversation, error, error_length);
        if (status != 0) break;

        size_t playback_frames = 0;
        status = lfm_conversation_prepare_pcm_native(
            conversation, pcm.size(), CAPTURE_RATE, PLAYBACK_RATE,
            &playback_frames);
        if (status != 0) {
            fail_step(error, error_length, "conversation PCM preparation",
                      status);
            (void)lfm_runtime_conversation_close(runtime, conversation);
            break;
        }

        std::string reply;
        status = run_context_turn(conversation, item.context, &bell, &reply,
                                  error, error_length);
        if (status != 0) {
            (void)lfm_runtime_conversation_close(runtime, conversation);
            break;
        }

        void *probe = nullptr;
        bell.arm();
        status = lfm_internal_conversation_listen_probe_submit_for_test(
            conversation, pcm.data(), pcm.size(), CAPTURE_RATE,
            tokens.data(), row_ns.data(), readouts.data(), tokens.size(),
            ring_doorbell, &bell, &probe);
        if (status != 0) {
            fail_step(error, error_length, "listen probe submit", status);
            (void)lfm_runtime_conversation_close(runtime, conversation);
            break;
        }
        bell.wait();
        uint64_t rows = 0;
        uint64_t encode_ns = 0;
        status = lfm_internal_conversation_listen_probe_collect_for_test(
            conversation, probe, &rows, &encode_ns);
        if (status != 0 || rows == 0) {
            fail_step(error, error_length, "listen probe collect",
                      status != 0 ? status : -ENODATA);
            if (status == 0) status = -ENODATA;
            (void)lfm_runtime_conversation_close(runtime, conversation);
            break;
        }

        uint64_t total_row_ns = 0;
        for (uint64_t row = 0; row < rows; ++row) {
            total_row_ns += row_ns[row];
            std::fprintf(rows_csv, "%s,%s,%llu,%u,%llu\n", item.name,
                         item.cls, static_cast<unsigned long long>(row),
                         tokens[row],
                         static_cast<unsigned long long>(row_ns[row]));
            const LfmListenReadoutForTest &readout = readouts[row];
            if (readout.text_ids[0] != tokens[row]) {
                char message[256]{};
                std::snprintf(message, sizeof(message),
                              "%s row %llu: readout text top-1 %u disagrees "
                              "with sampled token %u",
                              item.name,
                              static_cast<unsigned long long>(row),
                              readout.text_ids[0], tokens[row]);
                copy_error(error, error_length, message);
                status = LFM_STATUS_INTERNAL;
                break;
            }
            std::fprintf(v2_csv, "%s,%s,%llu,%u,%llu", item.name, item.cls,
                         static_cast<unsigned long long>(row), tokens[row],
                         static_cast<unsigned long long>(row_ns[row]));
            print_readout_head(v2_csv, readout.text_ids,
                               readout.text_logprobs, readout.text_entropy);
            print_readout_head(v2_csv, readout.audio_ids,
                               readout.audio_logprobs,
                               readout.audio_entropy);
            std::fprintf(v2_csv, ",%.3f\n",
                         static_cast<double>(row_ns[row]) / 1e6);
        }
        if (status != 0) {
            (void)lfm_runtime_conversation_close(runtime, conversation);
            break;
        }
        csv_sanitize(&reply);
        std::fprintf(items_csv, "%s,%s,%llu,%zu,%llu,%s\n", item.name,
                     item.cls, static_cast<unsigned long long>(rows),
                     pcm.size(),
                     static_cast<unsigned long long>(encode_ns),
                     reply.c_str());
        std::fprintf(
            stderr,
            "inner-voice %s(%s): samples=%zu rows=%llu encode=%.1fms "
            "mean-row=%.1fms reply=\"%s\"\n",
            item.name, item.cls, pcm.size(),
            static_cast<unsigned long long>(rows),
            static_cast<double>(encode_ns) / 1e6,
            static_cast<double>(total_row_ns) / 1e6 /
                static_cast<double>(rows),
            reply.c_str());

        const int closed =
            lfm_runtime_conversation_close(runtime, conversation);
        if (closed != 0) {
            status = fail_step(error, error_length, "conversation close",
                               closed);
            break;
        }
    }

    if (rows_csv) std::fclose(rows_csv);
    if (items_csv) std::fclose(items_csv);

    if (model) {
        const int closed = lfm_runtime_model_close(runtime, model);
        if (status == 0 && closed != 0) {
            status = fail_step(error, error_length, "model close", closed);
        }
    }
    if (runtime) {
        lfm_runtime_request_stop(runtime);
        const int joined = lfm_runtime_join(runtime);
        if (status == 0 && joined != 0) {
            status = fail_step(error, error_length, "runtime join", joined);
        }
        const int destroyed = lfm_runtime_destroy(runtime);
        if (status == 0 && destroyed != 0) {
            status = fail_step(error, error_length, "runtime destroy",
                               destroyed);
        }
    }
    if (status != 0 && error[0] == '\0') {
        fail_step(error, error_length, "inner-voice probe gate", status);
    }
    return status;
}
