// Private native Sesame/Web-Audio turn-evidence detector.
//
// The detector consumes exactly the latest 256 mono f32 samples by pointer.
// Its immutable plan owns only formula-derived Blackman/DFT coefficients; its
// mutable state is the compact selected-bin smoothing state plus separate,
// sticky microphone/playback adaptive extrema. No spectrum or tensor plane is
// constructed. The selected DFT and magnitude smoothing run in the selected
// architecture assembly leaf.
#ifndef LFM_SESAME_DETECTOR_H
#define LFM_SESAME_DETECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_SESAME_FFT_SIZE 256u
#define LFM_SESAME_MIC_THRESHOLD 50u
#define LFM_SESAME_PLAYBACK_THRESHOLD 10u

#define LFM_SESAME_STREAM_MIC 1u
#define LFM_SESAME_STREAM_PLAYBACK 2u

typedef struct LfmSesameDetector LfmSesameDetector;

/* A circular capture window is at most two physical spans. The descriptor is
 * stack metadata only: both pointers remain borrowed and the assembly leaf
 * consumes exactly 256 logical samples in first-then-second order. */
typedef struct LfmSesameWindow {
    const float *first;
    size_t first_count;
    const float *second;
    size_t second_count;
} LfmSesameWindow;

/* One borrowed fragment of a logical analyser window. Descriptors are stack
 * metadata only; samples remain in their owning capture/playback buffers. */
typedef struct LfmSesameSpan {
    const float *samples;
    size_t count;
} LfmSesameSpan;

/* An ordered scatter view of exactly 256 logical mono samples. The detector
 * accepts at most 256 nonempty spans and never flattens them. */
typedef struct LfmSesameScatterWindow {
    const LfmSesameSpan *spans;
    size_t span_count;
} LfmSesameScatterWindow;

typedef struct LfmSesameDecision {
    uint32_t sample_rate;
    uint32_t stream;
    uint32_t first_bin;
    uint32_t end_bin;
    uint32_t selected_bins;
    uint32_t threshold;
    uint32_t voice;
    double score;
    uint32_t adaptive_min;
    uint32_t adaptive_max;
} LfmSesameDecision;

// Build one immutable 256-point selected-DFT plan and two independent stream
// states. sample_rate must leave the complete [600, 2400) Hz band inside the
// 128 non-negative analyser bins.
LFM_INTERNAL_API int lfm_sesame_detector_create(uint32_t sample_rate,
                                                LfmSesameDetector **out);
LFM_INTERNAL_API int lfm_sesame_detector_destroy(LfmSesameDetector *detector);

// Reset one stream's smoothing and adaptive extrema. Resetting microphone
// state never changes playback state, and vice versa.
LFM_INTERNAL_API int lfm_sesame_detector_reset(LfmSesameDetector *detector,
                                               uint32_t stream);
// Break analyser-window continuity after a sequenced device GAP while keeping
// the stream's session-persistent adaptive minimum and maximum.
LFM_INTERNAL_API int lfm_sesame_detector_discontinuity(
    LfmSesameDetector *detector, uint32_t stream);

LFM_INTERNAL_API uint32_t
lfm_sesame_detector_first_bin(const LfmSesameDetector *detector);
LFM_INTERNAL_API uint32_t
lfm_sesame_detector_end_bin(const LfmSesameDetector *detector);
LFM_INTERNAL_API uint64_t
lfm_sesame_detector_derived_bytes(const LfmSesameDetector *detector);

// Consume a borrowed view of exactly 256 latest mono samples. selected_bytes
// is an optional compact diagnostic/replay destination; production may pass
// null with capacity zero. It never contains bins outside [first_bin,end_bin).
LFM_INTERNAL_API int lfm_sesame_detector_process(
    LfmSesameDetector *detector, uint32_t stream, const float *latest_256,
    uint8_t *selected_bytes, size_t selected_capacity,
    LfmSesameDecision *decision);

/* Consume the same logical 256-sample window from one or two borrowed spans.
 * This is the circular-arena entry point; it never materializes a contiguous
 * window. A one-span view reaches the original SIMD leaf unchanged. */
LFM_INTERNAL_API int lfm_sesame_detector_process_window(
    LfmSesameDetector *detector, uint32_t stream,
    const LfmSesameWindow *window, uint8_t *selected_bytes,
    size_t selected_capacity, LfmSesameDecision *decision);

/* Consume an arbitrary-span logical window. One- and two-span inputs retain
 * the existing contiguous/circular fast leaves; fragmented inputs are read
 * directly from the descriptor sequence by the architecture assembly leaf. */
LFM_INTERNAL_API int lfm_sesame_detector_process_scatter_window(
    LfmSesameDetector *detector, uint32_t stream,
    const LfmSesameScatterWindow *window, uint8_t *selected_bytes,
    size_t selected_capacity, LfmSesameDecision *decision);

// Feed already-quantized selected-bin evidence through the same sticky
// adaptive classifier. This is used by browser-trace conformance and permits
// direct state-machine tests without duplicating classifier logic.
LFM_INTERNAL_API int lfm_sesame_detector_classify_bytes(
    LfmSesameDetector *detector, uint32_t stream, const uint8_t *bytes,
    size_t count, LfmSesameDecision *decision);

#ifdef __cplusplus
}
#endif

#endif // LFM_SESAME_DETECTOR_H
