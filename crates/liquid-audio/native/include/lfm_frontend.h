// Native audio frontend ABI: torchaudio-exact windowed-sinc resampling and the
// NeMo mel featurizer (preemphasis -> centered/exact-pad DFT-basis STFT ->
// |X|^2 -> slaney mel -> log(x+guard) -> per-feature ddof-1 normalization ->
// tail mask + pad_to). Replaces the Rust owners in processor.rs (mod mel) and
// runtime/resample.rs; parity is gated by the committed fixtures under
// native/tests/fixtures/{mel,resample}/ (captured from the deleted Rust at
// working tree e018540c).
//
// Threading: a created frontend is an immutable table plan and may be shared
// by concurrent calls. Mutable run storage lives in a separate workspace; use
// one workspace per session/lane. Calls sharing one workspace serialize while
// borrowing its prepared high-water planes. Forward calls never allocate and
// zero only mathematically required padding/tail cells.
//
// Numerics: tables are built once at create in f64 and stored f32, exactly as
// the reference built them (hann periodic=false, librosa slaney norm, DFT basis
// with the window folded in). Hot-path arithmetic lives in the architecture
// assembly leaves (flashkern_frontend.S); the two matmul-shaped stages (DFT
// basis, mel filterbank) dispatch through Accelerate cblas on Apple and the
// scalar reference path elsewhere, per the doc 09 split.
#ifndef LFM_FRONTEND_H
#define LFM_FRONTEND_H

#include <stddef.h>
#include <stdint.h>

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_FRONTEND_ABI 1u

typedef struct LfmFrontend LfmFrontend;
typedef struct LfmFrontendWorkspace LfmFrontendWorkspace;
typedef struct LfmResampler LfmResampler;
typedef struct LfmResamplerWorkspace LfmResamplerWorkspace;
typedef struct LfmResamplerStream LfmResamplerStream;

#define LFM_FRONTEND_FORWARD_VALID_ONLY 1u
#define LFM_FRONTEND_WORKSPACE_BF16_OUTPUT 2u

// A borrowed/read-only PCM span. Resampling publishes either the caller's
// input span (equal rates) or the caller's destination span (different rates).
// The span remains valid only as long as that corresponding caller allocation.
typedef struct LfmF32Span {
    const float *data;
    uint64_t length;
} LfmF32Span;

// Fixed metadata-only view over one logical PCM range. A circular native arena
// may split that range once at physical wrap, so two spans are sufficient. The
// descriptor itself is copied into every asynchronous owner; the pointed
// samples remain borrowed and read-only. `length` is the checked sum of the
// admitted non-empty spans. No numerical entry point retains a pointer to a
// caller's descriptor array.
#define LFM_F32_SPAN_CHAIN_CAPACITY 2u
typedef struct LfmF32SpanChain {
    uint32_t count;
    uint32_t reserved0;
    uint64_t length;
    LfmF32Span spans[LFM_F32_SPAN_CHAIN_CAPACITY];
} LfmF32SpanChain;

// Validate and copy one or two non-empty borrowed spans into a fixed chain.
// The descriptor array may be transient; the sample storage must outlive the
// asynchronous owner that copies the resulting chain.
LFM_INTERNAL_API int lfm_f32_span_chain_init(
    const LfmF32Span *spans, uint32_t span_count, LfmF32SpanChain *out);

typedef struct LfmFrontendConfig {
    uint32_t size;
    uint32_t abi_version;
    uint32_t sample_rate;     // e.g. 16000
    uint32_t n_window_size;   // win_length in samples (400)
    uint32_t n_window_stride; // hop_length in samples (160)
    uint32_t n_fft;           // 512
    uint32_t nfilt;           // mel bins (128)
    uint32_t exact_pad;       // 0: torch.stft center=True; 1: NeMo exact_pad
    uint32_t pad_to;          // pad frame count to a multiple (0 = off)
    uint32_t reserved0;
    double preemph;              // 0.97 (0.0 disables)
    double log_zero_guard_value; // 2^-24
    double mag_power;            // must be 2.0 (only production regime admitted)
    uint64_t reserved[4];
} LfmFrontendConfig;

// 0 on success; -EINVAL on a malformed config; -EOPNOTSUPP when mag_power is
// not 2.0 (the reference's general branch is not admitted to production).
LFM_INTERNAL_API int lfm_frontend_create(const LfmFrontendConfig *config,
                                       LfmFrontend **out);
LFM_INTERNAL_API int lfm_frontend_destroy(LfmFrontend *frontend);

// Immutable formula-derived table bytes owned by the plan (mel filterbank and
// windowed DFT basis). Source/model bytes are not included.
LFM_INTERNAL_API uint64_t
lfm_frontend_derived_bytes(const LfmFrontend *frontend);

// Reusable, session-owned run storage. Reserve the maximum admitted clip at
// session readiness; forward never allocates or grows it. A workspace may be
// used with any frontend plan, but concurrent lanes should own distinct
// workspaces. Reserve flags select padded f32 (0), tightly packed valid f32
// (VALID_ONLY), or tightly packed valid BF16 (VALID_ONLY|BF16_OUTPUT).
LFM_INTERNAL_API int
lfm_frontend_workspace_create(LfmFrontendWorkspace **out);
LFM_INTERNAL_API int
lfm_frontend_workspace_destroy(LfmFrontendWorkspace *workspace);
LFM_INTERNAL_API int lfm_frontend_workspace_reserve(
    const LfmFrontend *frontend, LfmFrontendWorkspace *workspace,
    uint64_t max_sample_count, uint32_t flags);

// Valid mel frames for a clip of sample_count samples — the reference
// get_seq_len floor-divide contract (integer arithmetic, single source of
// truth for callers that used the Rust featurizer's method).
LFM_INTERNAL_API uint64_t
lfm_frontend_seq_len(const LfmFrontend *frontend, uint64_t sample_count);

// Total output frames forward will produce for this clip (framing count plus
// pad_to rounding), so the caller sizes out_mel exactly: nfilt * out_frames.
LFM_INTERNAL_API int lfm_frontend_out_frames(const LfmFrontend *frontend,
                                           uint64_t sample_count,
                                           uint64_t *out_frames);

// pcm: mono f32 in [-1,1], sample_count > 0 samples at config sample_rate.
// out_mel: row-major (nfilt x out_frames) normalized log-mel.
// Returns 0; -EINVAL on null/empty input or undersized capacity.
LFM_INTERNAL_API int lfm_frontend_forward(
    const LfmFrontend *frontend, const float *pcm, uint64_t sample_count,
    float *out_mel, uint64_t out_capacity_values);

// Convenience span contract: read `pcm` in place and write only the valid
// row-major (nfilt x seq_len) mel plane. Unlike lfm_frontend_forward, this does
// not compute the centered extra frame or append pad_to columns. The caller
// sizes out_mel as nfilt * lfm_frontend_seq_len(frontend, sample_count).
// Returns 0; -EINVAL on null/empty input, a zero valid-frame count, or an
// undersized output. This wrapper owns a temporary workspace; production
// sessions use lfm_frontend_forward_workspace below.
LFM_INTERNAL_API int lfm_frontend_forward_valid(
    const LfmFrontend *frontend, const float *pcm, uint64_t sample_count,
    float *out_mel, uint64_t out_capacity_values);

// Allocation-free-after-warm form used by production. flags is either 0 for
// the padded compatibility contract or LFM_FRONTEND_FORWARD_VALID_ONLY for a
// tightly packed (nfilt x seq_len) destination. All other bits are rejected.
LFM_INTERNAL_API int lfm_frontend_forward_workspace(
    const LfmFrontend *frontend, LfmFrontendWorkspace *workspace,
    const float *pcm, uint64_t sample_count, float *out_mel,
    uint64_t out_capacity_values, uint32_t flags);

// Split-range production form. The frontend reads the logical span chain
// directly. Preemphasis carries the preceding original sample across the one
// possible arena-wrap boundary, and frame gather resolves logical indices
// without concatenating PCM.
LFM_INTERNAL_API int lfm_frontend_forward_spans_workspace(
    const LfmFrontend *frontend, LfmFrontendWorkspace *workspace,
    const LfmF32SpanChain *pcm, float *out_mel,
    uint64_t out_capacity_values, uint32_t flags);

// Production Conformer seam: computes tightly packed valid mel rows in the
// prepared workspace and rounds each normalized row directly into the caller's
// BF16 destination. No f32 mel plane is published or copied. The workspace
// must have been reserved with VALID_ONLY|BF16_OUTPUT.
LFM_INTERNAL_API int lfm_frontend_forward_bf16_workspace(
    const LfmFrontend *frontend, LfmFrontendWorkspace *workspace,
    const float *pcm, uint64_t sample_count, uint16_t *out_mel,
    uint64_t out_capacity_values);
LFM_INTERNAL_API int lfm_frontend_forward_bf16_spans_workspace(
    const LfmFrontend *frontend, LfmFrontendWorkspace *workspace,
    const LfmF32SpanChain *pcm, uint16_t *out_mel,
    uint64_t out_capacity_values);

// Immutable, pair-specific torchaudio resampling plan. Formula-derived f64
// phase kernels are built once here, never during execution.
LFM_INTERNAL_API int lfm_resampler_create(uint32_t orig_freq, uint32_t new_freq,
                                        LfmResampler **out);
LFM_INTERNAL_API int lfm_resampler_destroy(LfmResampler *resampler);
LFM_INTERNAL_API uint64_t
lfm_resampler_derived_bytes(const LfmResampler *resampler);
LFM_INTERNAL_API int lfm_resampler_out_length(const LfmResampler *resampler,
                                            uint64_t sample_count,
                                            uint64_t *out_length);

// Session-owned admission watermark. Reserve once before readiness; it owns no
// numerical plane. Processing reads the caller's f32 span directly, treats the
// prefix/suffix padding as logical zeros, and returns -ENOBUFS if a command
// exceeds the prepared sample count.
LFM_INTERNAL_API int
lfm_resampler_workspace_create(LfmResamplerWorkspace **out);
LFM_INTERNAL_API int
lfm_resampler_workspace_destroy(LfmResamplerWorkspace *workspace);
LFM_INTERNAL_API int lfm_resampler_workspace_reserve(
    const LfmResampler *resampler, LfmResamplerWorkspace *workspace,
    uint64_t max_sample_count);

// Allocation-free execution. With equal rates, destination may be null and
// result aliases input exactly. With different rates, destination receives the
// final convolution values directly (there is no intermediate/copy) and result
// aliases destination. For different rates input and destination must not
// overlap: preserving an overlapping source would require forbidden staging.
LFM_INTERNAL_API int lfm_resampler_process(
    const LfmResampler *resampler, LfmResamplerWorkspace *workspace,
    const float *input, uint64_t sample_count, float *destination,
    uint64_t destination_capacity, LfmF32Span *result);

// Split-range production form. Equal-rate processing returns the original
// fixed chain unchanged. Different-rate processing performs the sinc
// convolution directly over logical indices and returns one span over
// `destination`.
LFM_INTERNAL_API int lfm_resampler_process_spans(
    const LfmResampler *resampler, LfmResamplerWorkspace *workspace,
    const LfmF32SpanChain *input, float *destination,
    uint64_t destination_capacity, LfmF32SpanChain *result);

// Conversation-owned streaming band-limited rate converter. It uses the same
// immutable sinc/Hann polyphase table as the offline path, causalized by a
// fixed `width + reduced_input_rate - 1` source-sample group delay. For the
// 24 kHz -> 16 kHz path that is 12 source samples (8 device samples); its
// 23-tap support reaches back 22 source samples, and the circular carry keeps
// one additional sample for a phase-stride boundary. Chunk boundaries are
// numerically invisible. Process emits only the duration-aligned
// ceil(total_input * new/orig) prefix: it does not implicitly feed terminal
// zeros. A caller that needs the delayed FIR tail must submit zeros explicitly;
// reset deliberately discards that tail. The maximum admitted input span is
// fixed at creation; execution writes directly into the caller's final
// destination and never owns an output plane. Convolution and circular-history
// carry execute in the architecture leaf; C++ owns table setup, borrowed-view
// validation, and continuity-state commits.
LFM_INTERNAL_API int lfm_resampler_stream_create(
    uint32_t orig_freq, uint32_t new_freq, uint64_t max_sample_count,
    LfmResamplerStream **out);
LFM_INTERNAL_API int
lfm_resampler_stream_destroy(LfmResamplerStream *stream);
LFM_INTERNAL_API void lfm_resampler_stream_reset(LfmResamplerStream *stream);
LFM_INTERNAL_API int lfm_resampler_stream_out_length(
    LfmResamplerStream *stream, uint64_t sample_count,
    uint64_t *out_length);
LFM_INTERNAL_API int lfm_resampler_stream_process(
    LfmResamplerStream *stream, const float *input, uint64_t sample_count,
    float *destination, uint64_t destination_capacity,
    LfmF32Span *result);

// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width=6,
// rolloff=0.99), f64 kernels and accumulation, truncated to
// ceil(length * new_freq / orig_freq) samples. Transitional compatibility
// wrapper: it constructs a temporary plan/workspace and must copy when equal
// rates and output != input. Production capture uses the prepared plan/span API;
// streaming playback uses the retained phase/history state above.
LFM_INTERNAL_API int lfm_resample_f32(
    const float *x, uint64_t length, uint32_t orig_freq, uint32_t new_freq,
    float *out, uint64_t out_capacity, uint64_t *out_length);

#ifdef __cplusplus
}
#endif

#endif // LFM_FRONTEND_H
