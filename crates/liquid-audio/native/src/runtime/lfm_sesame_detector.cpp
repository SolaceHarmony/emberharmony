// Exact native implementation of the recovered Sesame Web-Audio evidence
// detector. This unit owns setup-time formula tables and compact classifier
// state only. Value-producing DFT/magnitude/smoothing work is delegated to the
// architecture assembly leaf in flashkern_sesame.S.

#include "lfm_sesame_detector.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numbers>

extern "C" void lfm_sesame_selected_magnitudes(
    const float *pcm, const float *real_table, const float *imag_table,
    float *smoothed_magnitudes, size_t selected_bins);
extern "C" void lfm_sesame_selected_magnitudes_window(
    const LfmSesameWindow *window, const float *real_table,
    const float *imag_table, float *smoothed_magnitudes,
    size_t selected_bins);
extern "C" void lfm_sesame_selected_magnitudes_scatter(
    const LfmSesameScatterWindow *window, const float *real_table,
    const float *imag_table, float *smoothed_magnitudes,
    size_t selected_bins);
extern "C" void lfm_sesame_magnitudes_to_bytes(
    const float *magnitudes, const double *thresholds, uint8_t *bytes,
    size_t count);
extern "C" uint32_t lfm_sesame_classify_selected_bytes(
    const uint8_t *bytes, size_t count, uint8_t *minimum, uint8_t *maximum,
    uint32_t threshold, double *score);

namespace {

constexpr uint32_t kFft = LFM_SESAME_FFT_SIZE;
constexpr uint32_t kFrequencyBins = kFft / 2;
constexpr double kBandLow = 600.0;
constexpr double kBandHigh = 2400.0;
constexpr double kMinDb = -100.0;
constexpr double kMaxDb = -30.0;

struct StreamState {
    std::unique_ptr<float[]> magnitude;
    uint8_t minimum = 255;
    uint8_t maximum = 0;
};

bool stream_valid(uint32_t stream) {
    return stream == LFM_SESAME_STREAM_MIC ||
           stream == LFM_SESAME_STREAM_PLAYBACK;
}

uint32_t threshold_for(uint32_t stream) {
    return stream == LFM_SESAME_STREAM_MIC ? LFM_SESAME_MIC_THRESHOLD
                                           : LFM_SESAME_PLAYBACK_THRESHOLD;
}

} // namespace

struct LfmSesameDetector {
    uint32_t sample_rate = 0;
    uint32_t first_bin = 0;
    uint32_t end_bin = 0;
    size_t bins = 0;
    std::unique_ptr<float[]> real_table;
    std::unique_ptr<float[]> imag_table;
    std::unique_ptr<double[]> byte_thresholds;
    StreamState mic;
    StreamState playback;
};

namespace {

StreamState *state_for(LfmSesameDetector *detector, uint32_t stream) {
    if (!detector || !stream_valid(stream)) {
        return nullptr;
    }
    return stream == LFM_SESAME_STREAM_MIC ? &detector->mic
                                            : &detector->playback;
}

void fill_decision(const LfmSesameDetector &detector, uint32_t stream,
                   const StreamState &state, double score, uint32_t voice,
                   LfmSesameDecision *decision) {
    *decision = {};
    decision->sample_rate = detector.sample_rate;
    decision->stream = stream;
    decision->first_bin = detector.first_bin;
    decision->end_bin = detector.end_bin;
    decision->selected_bins = static_cast<uint32_t>(detector.bins);
    decision->threshold = threshold_for(stream);
    decision->voice = voice;
    decision->score = score;
    decision->adaptive_min = state.minimum;
    decision->adaptive_max = state.maximum;
}

int classify(LfmSesameDetector *detector, uint32_t stream,
             const uint8_t *bytes, size_t count,
             LfmSesameDecision *decision) {
    StreamState *state = state_for(detector, stream);
    if (!state || !bytes || count == 0 || count > kFrequencyBins || !decision) {
        return -EINVAL;
    }

    double score = 0.0;
    const uint32_t voice = lfm_sesame_classify_selected_bytes(
        bytes, count, &state->minimum, &state->maximum, threshold_for(stream),
        &score);
    fill_decision(*detector, stream, *state, score, voice, decision);
    return 0;
}

bool destination_valid(const LfmSesameDetector *detector,
                       const uint8_t *selected_bytes,
                       size_t selected_capacity,
                       const LfmSesameDecision *decision) {
    return detector && decision &&
           ((selected_bytes && selected_capacity >= detector->bins) ||
            (!selected_bytes && selected_capacity == 0));
}

int finish_process(LfmSesameDetector *detector, uint32_t stream,
                   StreamState *state, uint8_t *selected_bytes,
                   LfmSesameDecision *decision) {
    std::array<uint8_t, kFrequencyBins> evidence{};
    lfm_sesame_magnitudes_to_bytes(
        state->magnitude.get(), detector->byte_thresholds.get(),
        evidence.data(), detector->bins);
    const int status = classify(detector, stream, evidence.data(),
                                detector->bins, decision);
    if (status != 0) {
        return status;
    }
    if (selected_bytes) {
        std::copy_n(evidence.data(), detector->bins, selected_bytes);
    }
    return 0;
}

} // namespace

extern "C" int lfm_sesame_detector_create(uint32_t sample_rate,
                                            LfmSesameDetector **out) {
    if (!out || sample_rate == 0) {
        return -EINVAL;
    }
    *out = nullptr;

    const double bin_hz = static_cast<double>(sample_rate) / kFft;
    const uint32_t first_bin = static_cast<uint32_t>(std::floor(kBandLow / bin_hz));
    const uint32_t end_bin = static_cast<uint32_t>(std::floor(kBandHigh / bin_hz));
    if (first_bin >= end_bin || end_bin > kFrequencyBins) {
        return -EINVAL;
    }
    const size_t bins = static_cast<size_t>(end_bin - first_bin);
    if (bins > std::numeric_limits<size_t>::max() / kFft) {
        return -EOVERFLOW;
    }
    const size_t table_values = bins * kFft;

    std::unique_ptr<LfmSesameDetector> detector(
        new (std::nothrow) LfmSesameDetector());
    if (!detector) {
        return -ENOMEM;
    }
    detector->real_table.reset(new (std::nothrow) float[table_values]);
    detector->imag_table.reset(new (std::nothrow) float[table_values]);
    detector->mic.magnitude.reset(new (std::nothrow) float[bins]());
    detector->playback.magnitude.reset(new (std::nothrow) float[bins]());
    detector->byte_thresholds.reset(new (std::nothrow) double[256]);
    if (!detector->real_table || !detector->imag_table ||
        !detector->mic.magnitude || !detector->playback.magnitude ||
        !detector->byte_thresholds) {
        return -ENOMEM;
    }

    detector->sample_rate = sample_rate;
    detector->first_bin = first_bin;
    detector->end_bin = end_bin;
    detector->bins = bins;

    constexpr double alpha = 0.16;
    constexpr double a0 = (1.0 - alpha) / 2.0;
    constexpr double a1 = 0.5;
    constexpr double a2 = alpha / 2.0;
    constexpr double inverse_n = 1.0 / static_cast<double>(kFft);
    constexpr double turn = 2.0 * std::numbers::pi_v<double>;
    detector->byte_thresholds[0] = 0.0;
    for (size_t byte = 1; byte < 256; ++byte) {
        // floor((db-min)*255/(max-min)) reaches `byte` at this exact
        // magnitude. The assembly leaf finds the greatest reached threshold,
        // avoiding a per-window scalar log10 while preserving clamp/floor
        // boundary semantics in f64.
        const double db = kMinDb + static_cast<double>(byte) *
                                       (kMaxDb - kMinDb) / 255.0;
        detector->byte_thresholds[byte] = std::pow(10.0, db / 20.0);
    }
    for (size_t row = 0; row < bins; ++row) {
        const uint32_t bin = first_bin + static_cast<uint32_t>(row);
        for (uint32_t sample = 0; sample < kFft; ++sample) {
            const double phase = turn * static_cast<double>(sample) / kFft;
            const double window = a0 - a1 * std::cos(phase) +
                                  a2 * std::cos(2.0 * phase);
            const double angle = phase * static_cast<double>(bin);
            const size_t index = row * kFft + sample;
            detector->real_table[index] = static_cast<float>(
                window * std::cos(angle) * inverse_n);
            detector->imag_table[index] = static_cast<float>(
                -window * std::sin(angle) * inverse_n);
        }
    }

    *out = detector.release();
    return 0;
}

extern "C" int lfm_sesame_detector_destroy(LfmSesameDetector *detector) {
    if (!detector) {
        return -EINVAL;
    }
    delete detector;
    return 0;
}

extern "C" int lfm_sesame_detector_reset(LfmSesameDetector *detector,
                                           uint32_t stream) {
    StreamState *state = state_for(detector, stream);
    if (!state) {
        return -EINVAL;
    }
    std::fill_n(state->magnitude.get(), detector->bins, 0.0f);
    state->minimum = 255;
    state->maximum = 0;
    return 0;
}

extern "C" int lfm_sesame_detector_discontinuity(
    LfmSesameDetector *detector, uint32_t stream) {
    StreamState *state = state_for(detector, stream);
    if (!state) {
        return -EINVAL;
    }
    std::fill_n(state->magnitude.get(), detector->bins, 0.0f);
    return 0;
}

extern "C" uint32_t
lfm_sesame_detector_first_bin(const LfmSesameDetector *detector) {
    return detector ? detector->first_bin : 0;
}

extern "C" uint32_t
lfm_sesame_detector_end_bin(const LfmSesameDetector *detector) {
    return detector ? detector->end_bin : 0;
}

extern "C" uint64_t
lfm_sesame_detector_derived_bytes(const LfmSesameDetector *detector) {
    if (!detector) {
        return 0;
    }
    return static_cast<uint64_t>(detector->bins) * kFft * 2 * sizeof(float) +
           256 * sizeof(double);
}

extern "C" int lfm_sesame_detector_process(
    LfmSesameDetector *detector, uint32_t stream, const float *latest_256,
    uint8_t *selected_bytes, size_t selected_capacity,
    LfmSesameDecision *decision) {
    const LfmSesameWindow window = {
        .first = latest_256,
        .first_count = latest_256 ? kFft : 0,
        .second = nullptr,
        .second_count = 0,
    };
    return lfm_sesame_detector_process_window(
        detector, stream, &window, selected_bytes, selected_capacity,
        decision);
}

extern "C" int lfm_sesame_detector_process_window(
    LfmSesameDetector *detector, uint32_t stream,
    const LfmSesameWindow *window, uint8_t *selected_bytes,
    size_t selected_capacity, LfmSesameDecision *decision) {
    StreamState *state = state_for(detector, stream);
    if (!state || !window || !window->first || !decision ||
        window->first_count == 0 || window->first_count > kFft ||
        window->second_count > kFft - window->first_count ||
        window->first_count + window->second_count != kFft ||
        (window->second_count != 0 && !window->second) ||
        (window->second_count == 0 && window->second != nullptr) ||
        !destination_valid(detector, selected_bytes, selected_capacity,
                           decision)) {
        return -EINVAL;
    }

    if (window->second_count == 0) {
        lfm_sesame_selected_magnitudes(
            window->first, detector->real_table.get(),
            detector->imag_table.get(), state->magnitude.get(),
            detector->bins);
    } else {
        lfm_sesame_selected_magnitudes_window(
            window, detector->real_table.get(), detector->imag_table.get(),
            state->magnitude.get(), detector->bins);
    }

    return finish_process(detector, stream, state, selected_bytes, decision);
}

extern "C" int lfm_sesame_detector_process_scatter_window(
    LfmSesameDetector *detector, uint32_t stream,
    const LfmSesameScatterWindow *window, uint8_t *selected_bytes,
    size_t selected_capacity, LfmSesameDecision *decision) {
    StreamState *state = state_for(detector, stream);
    if (!state || !window || !window->spans || window->span_count == 0 ||
        window->span_count > kFft ||
        !destination_valid(detector, selected_bytes, selected_capacity,
                           decision)) {
        return -EINVAL;
    }

    size_t total = 0;
    for (size_t index = 0; index < window->span_count; ++index) {
        const LfmSesameSpan &span = window->spans[index];
        if (!span.samples || span.count == 0 || span.count > kFft ||
            total > kFft - span.count) {
            return -EINVAL;
        }
        total += span.count;
    }
    if (total != kFft) {
        return -EINVAL;
    }

    if (window->span_count == 1) {
        lfm_sesame_selected_magnitudes(
            window->spans[0].samples, detector->real_table.get(),
            detector->imag_table.get(), state->magnitude.get(),
            detector->bins);
    } else if (window->span_count == 2) {
        const LfmSesameWindow split = {
            .first = window->spans[0].samples,
            .first_count = window->spans[0].count,
            .second = window->spans[1].samples,
            .second_count = window->spans[1].count,
        };
        lfm_sesame_selected_magnitudes_window(
            &split, detector->real_table.get(), detector->imag_table.get(),
            state->magnitude.get(), detector->bins);
    } else {
        lfm_sesame_selected_magnitudes_scatter(
            window, detector->real_table.get(), detector->imag_table.get(),
            state->magnitude.get(), detector->bins);
    }
    return finish_process(detector, stream, state, selected_bytes, decision);
}

extern "C" int lfm_sesame_detector_classify_bytes(
    LfmSesameDetector *detector, uint32_t stream, const uint8_t *bytes,
    size_t count, LfmSesameDecision *decision) {
    return classify(detector, stream, bytes, count, decision);
}
