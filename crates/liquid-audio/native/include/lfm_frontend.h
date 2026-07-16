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
// borrowing its high-water planes. Warm calls allocate and zero-fill no
// scratch memory.
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

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_FRONTEND_ABI 1u

typedef struct LfmFrontend LfmFrontend;
typedef struct LfmFrontendWorkspace LfmFrontendWorkspace;

#define LFM_FRONTEND_FORWARD_VALID_ONLY 1u

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
int lfm_frontend_create(const LfmFrontendConfig *config, LfmFrontend **out);
int lfm_frontend_destroy(LfmFrontend *frontend);

// Reusable, session-owned run storage. It grows to the largest clip submitted
// through it and then remains allocation-free. A workspace may be used with
// any frontend plan, but concurrent lanes should own distinct workspaces.
int lfm_frontend_workspace_create(LfmFrontendWorkspace **out);
int lfm_frontend_workspace_destroy(LfmFrontendWorkspace *workspace);

// Valid mel frames for a clip of sample_count samples — the reference
// get_seq_len floor-divide contract (integer arithmetic, single source of
// truth for callers that used the Rust featurizer's method).
uint64_t lfm_frontend_seq_len(const LfmFrontend *frontend, uint64_t sample_count);

// Total output frames forward will produce for this clip (framing count plus
// pad_to rounding), so the caller sizes out_mel exactly: nfilt * out_frames.
int lfm_frontend_out_frames(const LfmFrontend *frontend, uint64_t sample_count,
                            uint64_t *out_frames);

// pcm: mono f32 in [-1,1], sample_count > 0 samples at config sample_rate.
// out_mel: row-major (nfilt x out_frames) normalized log-mel.
// Returns 0; -EINVAL on null/empty input or undersized capacity.
int lfm_frontend_forward(const LfmFrontend *frontend, const float *pcm,
                         uint64_t sample_count, float *out_mel,
                         uint64_t out_capacity_values);

// Convenience span contract: read `pcm` in place and write only the valid
// row-major (nfilt x seq_len) mel plane. Unlike lfm_frontend_forward, this does
// not compute the centered extra frame or append pad_to columns. The caller
// sizes out_mel as nfilt * lfm_frontend_seq_len(frontend, sample_count).
// Returns 0; -EINVAL on null/empty input, a zero valid-frame count, or an
// undersized output. This wrapper owns a temporary workspace; production
// sessions use lfm_frontend_forward_workspace below.
int lfm_frontend_forward_valid(const LfmFrontend *frontend, const float *pcm,
                               uint64_t sample_count, float *out_mel,
                               uint64_t out_capacity_values);

// Allocation-free-after-warm form used by production. flags is either 0 for
// the padded compatibility contract or LFM_FRONTEND_FORWARD_VALID_ONLY for a
// tightly packed (nfilt x seq_len) destination. All other bits are rejected.
int lfm_frontend_forward_workspace(const LfmFrontend *frontend,
                                   LfmFrontendWorkspace *workspace,
                                   const float *pcm, uint64_t sample_count,
                                   float *out_mel, uint64_t out_capacity_values,
                                   uint32_t flags);

// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width=6,
// rolloff=0.99), f64 kernels and accumulation, truncated to
// ceil(length * new_freq / orig_freq) samples. orig == new copies through.
// Returns 0; -EINVAL on null/zero-rate args or undersized out capacity.
int lfm_resample_f32(const float *x, uint64_t length, uint32_t orig_freq,
                     uint32_t new_freq, float *out, uint64_t out_capacity,
                     uint64_t *out_length);

#ifdef __cplusplus
}
#endif

#endif // LFM_FRONTEND_H
