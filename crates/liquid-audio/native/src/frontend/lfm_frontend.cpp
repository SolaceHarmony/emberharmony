// Native audio frontend: torchaudio-exact resampling + the NeMo mel featurizer.
// See lfm_frontend.h for the contract and native/tests/fixtures/{mel,resample}
// for the parity oracle (captured from the deleted Rust owners at e018540c).
//
// Layering: this TU builds immutable tables at create (f64 math, run once, the
// same init-time class as Mimi's weight folding), owns pass layout (framing,
// masks, pads), and routes every hot value-producing loop to an architecture
// assembly leaf in flashkern_frontend.S. The two matmul-shaped stages ride
// Accelerate cblas on Apple and the scalar reference path elsewhere — the same
// guard mimi_decode.cpp ships.
//
// Faithful-port notes (final outputs are fixture-gated; the committed
// intermediate/table captures still need their own direct-observation gate):
// - The preemphasis time mask zeroes positions >= L of the PADDED signal in the
//   exact_pad path (the reference's seq_len_time is NOT shifted by the pad).
// - Statistics use [0, seq_len). The padded compatibility output normalizes
//   that retained prefix and zeroes [seq_len, T_out); the valid-only production
//   output neither computes nor transports the discarded tail.
// - mag_power != 2.0 is rejected at create (typed error, no silent fallback).
// - seq_len == 0 clips produce the reference's defined result — an all-zero
//   plane (its normalization NaNs are fully covered by the [0, t) tail mask).

#include "lfm_frontend.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

// Architecture assembly leaves (flashkern_frontend.S, both arches).
extern "C" {
int lfm_preemph_f32(const float *x, float *y, uint64_t n, float coef);
void lfm_power_spec_f32(const float *re, const float *im, float *out, uint64_t n);
int lfm_log_add_f32(float *x, uint64_t n, float guard);
int lfm_rowstat_f32(const float *x, uint64_t rows, uint64_t cols, uint64_t valid,
                    float constant, float *mean_out, float *std_out);
void lfm_norm_apply_f32(float *x, const float *mean, const float *std,
                        uint64_t rows, uint64_t cols);
void lfm_f32_to_bf16(const float *input, uint16_t *output, int count);
void lfm_resample_conv_f64(const double *padded, uint64_t padded_len,
                           const double *kernels, uint64_t phases, uint64_t klen,
                           uint64_t stride, float *out, uint64_t max_values);
}

namespace {

// f32 GEMM: C(MxN) = A(MxK) . B(KxN), all row-major, beta 0 — the shape both
// the DFT-basis and mel stages use. Apple: Accelerate (AMX). Elsewhere: the
// scalar reference path (parity oracle order: k ascending, f32 accumulate).
void sgemm_rm(int m, int n, int k, const float *a, int lda, const float *b,
              int ldb, float *c, int ldc) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a,
                lda, b, ldb, 0.0f, c, ldc);
#else
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += a[(size_t)i * lda + p] * b[(size_t)p * ldb + j];
            c[(size_t)i * ldc + j] = acc;
        }
    }
#endif
}

bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    *out = a + b;
    return true;
}

bool mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

} // namespace

struct LfmFrontend {
    LfmFrontendConfig cfg;
    uint32_t freq = 0;       // n_fft/2 + 1
    float *fb = nullptr;     // (nfilt x freq) slaney mel filterbank
    float *dft = nullptr;    // (2*freq x n_fft) DFT basis, hann window folded in
};

struct LfmFrontendWorkspace {
    std::mutex lock;
    float *values = nullptr;
    uint64_t capacity = 0;

    ~LfmFrontendWorkspace() { std::free(values); }
};

struct LfmResampler {
    uint32_t orig_freq = 0;
    uint32_t new_freq = 0;
    uint64_t orig = 0;       // gcd-reduced input rate
    uint64_t phases = 0;     // gcd-reduced output rate
    uint64_t width = 0;
    uint64_t kernel_len = 0;
    double *kernels = nullptr;

    ~LfmResampler() { std::free(kernels); }
};

struct LfmResamplerWorkspace {
    std::mutex lock;
    double *padded = nullptr;
    uint64_t capacity = 0;

    ~LfmResamplerWorkspace() { std::free(padded); }
};

namespace {

// --- init-time tables (f64 math, stored f32 — exactly the reference ctor) ---

// torch.hann_window(n, periodic=False).
void build_hann(uint32_t n, double *out) {
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    for (uint32_t i = 0; i < n; ++i)
        out[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / ((double)n - 1.0));
}

// librosa slaney mel scale.
double hz_to_mel(double f) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
}
double mel_to_hz(double m) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney"),
// flattened (nfilt x freq) row-major.
void build_melfb(uint32_t sr, uint32_t n_fft, uint32_t n_mels, float *fb) {
    const uint32_t freq = n_fft / 2 + 1;
    const double fmax = (double)sr / 2.0;
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(fmax);
    for (uint32_t m = 0; m < n_mels; ++m) {
        const double lower = mel_to_hz(mel_min + (mel_max - mel_min) * (double)m / (double)(n_mels + 1));
        const double center = mel_to_hz(mel_min + (mel_max - mel_min) * (double)(m + 1) / (double)(n_mels + 1));
        const double upper = mel_to_hz(mel_min + (mel_max - mel_min) * (double)(m + 2) / (double)(n_mels + 1));
        const double enorm = 2.0 / (upper - lower);
        for (uint32_t k = 0; k < freq; ++k) {
            const double f = (double)k * (double)sr / (double)n_fft;
            const double down = (f - lower) / (center - lower);
            const double up = (upper - f) / (upper - center);
            double w = down < up ? down : up;
            if (w < 0.0) w = 0.0;
            fb[(size_t)m * freq + k] = (float)(w * enorm);
        }
    }
}

// DFT-basis kernel (2*freq x n_fft): channel k < freq carries
// window[n]*cos(2*pi*k*n/N); channel freq+k carries -window[n]*sin(...).
// Window is centered/zero-padded to n_fft first (torch.stft win_length < n_fft).
bool build_dft(uint32_t n_fft, uint32_t win, float *dft) {
    const uint32_t freq = n_fft / 2 + 1;
    double *w = (double *)std::calloc(n_fft, sizeof(double));
    double *h = (double *)std::malloc((size_t)win * sizeof(double));
    if (!w || !h) {
        std::free(h);
        std::free(w);
        return false;
    }
    build_hann(win, h);
    const uint32_t left = (n_fft - win) / 2;
    for (uint32_t i = 0; i < win; ++i) w[left + i] = h[i];
    // The reference stored the padded window as f32 before folding; round-trip
    // through f32 so the folded products match bitwise.
    for (uint32_t i = 0; i < n_fft; ++i) w[i] = (double)(float)w[i];
    const double two_pi = 2.0 * M_PI;
    for (uint32_t k = 0; k < freq; ++k) {
        for (uint32_t n = 0; n < n_fft; ++n) {
            const double ang = two_pi * (double)k * (double)n / (double)n_fft;
            dft[(size_t)k * n_fft + n] = (float)(w[n] * std::cos(ang));
            dft[(size_t)(freq + k) * n_fft + n] = (float)(-(w[n] * std::sin(ang)));
        }
    }
    std::free(h);
    std::free(w);
    return true;
}

// Framing geometry shared by out_frames/forward.
bool frame_count(const LfmFrontendConfig &c, uint64_t li, uint64_t center_pad,
                 uint64_t *out) {
    uint64_t twice = 0;
    uint64_t padded = 0;
    if (!mul_u64(2, center_pad, &twice) || !add_u64(li, twice, &padded)) return false;
    if (padded < c.n_fft || c.n_window_stride == 0) {
        *out = 0;
        return true;
    }
    *out = (padded - c.n_fft) / c.n_window_stride + 1;
    return true;
}

uint64_t seq_len_of(const LfmFrontendConfig &c, uint64_t l) {
    const uint64_t pad_amount =
        c.exact_pad ? 2 * (((uint64_t)c.n_fft - c.n_window_stride) / 2)
                    : ((uint64_t)c.n_fft / 2) * 2;
    uint64_t sum = 0;
    if (!add_u64(l, pad_amount, &sum)) return 0;
    const uint64_t numer = sum > c.n_fft ? sum - c.n_fft : 0;
    return c.n_window_stride > 0 ? numer / c.n_window_stride : 0;
}

struct FrontendRun {
    uint64_t seq_len = 0;
    uint64_t pad = 0;
    uint64_t signal_len = 0;
    uint64_t center = 0;
    uint64_t frames = 0;
    uint64_t out_frames = 0;
    uint64_t cols = 0;
    uint64_t stride = 0;
    uint64_t power_values = 0;
    uint64_t plane_a_values = 0;
    uint64_t plane_b_values = 0;
    uint64_t workspace_values = 0;
    uint64_t output_values = 0;
};

int frontend_run(const LfmFrontend *f, uint64_t l, bool valid_only,
                 bool bf16_output, FrontendRun *run) {
    if (!f || !run || l == 0 || (bf16_output && !valid_only)) return -EINVAL;
    const LfmFrontendConfig &c = f->cfg;
    FrontendRun next{};
    next.seq_len = seq_len_of(c, l);
    next.pad = c.exact_pad
                   ? ((uint64_t)c.n_fft - c.n_window_stride) / 2
                   : 0;
    uint64_t twice = 0;
    if (!mul_u64(2, next.pad, &twice) ||
        !add_u64(l, twice, &next.signal_len)) {
        return -EOVERFLOW;
    }
    next.center = c.exact_pad ? 0 : (uint64_t)c.n_fft / 2;
    if (!frame_count(c, next.signal_len, next.center, &next.frames)) {
        return -EOVERFLOW;
    }
    if (next.frames == 0 || next.seq_len > next.frames) return -EINVAL;
    next.out_frames = next.frames;
    if (c.pad_to > 0 && next.frames % c.pad_to != 0 &&
        !add_u64(next.frames, c.pad_to - next.frames % c.pad_to,
                 &next.out_frames)) {
        return -EOVERFLOW;
    }
    next.cols = valid_only ? next.seq_len : next.frames;
    next.stride = valid_only ? next.seq_len : next.out_frames;
    if (valid_only && next.seq_len == 0) return -EINVAL;

    uint64_t frame_values = 0;
    uint64_t stft_values = 0;
    uint64_t stat_values = 0;
    uint64_t mel_values = 0;
    if (!mul_u64(c.n_fft, next.cols, &frame_values) ||
        !mul_u64(2 * (uint64_t)f->freq, next.cols, &stft_values) ||
        !mul_u64(f->freq, next.cols, &next.power_values) ||
        !mul_u64(2, c.nfilt, &stat_values) ||
        !mul_u64(c.nfilt, next.cols, &mel_values) ||
        !mul_u64(c.nfilt, next.stride, &next.output_values)) {
        return -EOVERFLOW;
    }

    // The signal plane becomes STFT/power. The frame plane becomes the f32 mel
    // plane only for the BF16 production seam; after mel GEMM the now-dead
    // power plane holds row statistics. No simultaneously-live values alias.
    next.plane_a_values = std::max(next.signal_len, stft_values);
    next.plane_b_values = std::max(frame_values, stat_values);
    if (bf16_output) {
        next.plane_a_values = std::max(next.plane_a_values, stat_values);
        next.plane_b_values = std::max(next.plane_b_values, mel_values);
    }
    if (!add_u64(next.plane_a_values, next.plane_b_values,
                 &next.workspace_values) ||
        next.workspace_values >
            std::numeric_limits<size_t>::max() / sizeof(float) ||
        next.output_values >
            std::numeric_limits<size_t>::max() /
                (bf16_output ? sizeof(uint16_t) : sizeof(float)) ||
        next.cols > INT_MAX || c.n_fft > INT_MAX || c.nfilt > INT_MAX ||
        f->freq > INT_MAX || 2 * (uint64_t)f->freq > INT_MAX ||
        next.stride > INT_MAX || (bf16_output && next.cols > INT_MAX)) {
        return -EOVERFLOW;
    }
    *run = next;
    return 0;
}

int reserve_frontend(const LfmFrontend *f, LfmFrontendWorkspace *workspace,
                     uint64_t max_samples, uint32_t flags) {
    if (!f || !workspace ||
        (flags & ~(LFM_FRONTEND_FORWARD_VALID_ONLY |
                   LFM_FRONTEND_WORKSPACE_BF16_OUTPUT)) ||
        ((flags & LFM_FRONTEND_WORKSPACE_BF16_OUTPUT) &&
         !(flags & LFM_FRONTEND_FORWARD_VALID_ONLY))) {
        return -EINVAL;
    }
    FrontendRun run{};
    const int status = frontend_run(
        f, max_samples, (flags & LFM_FRONTEND_FORWARD_VALID_ONLY) != 0,
        (flags & LFM_FRONTEND_WORKSPACE_BF16_OUTPUT) != 0, &run);
    if (status != 0) return status;
    std::lock_guard<std::mutex> guard(workspace->lock);
    if (workspace->capacity >= run.workspace_values) return 0;
    float *next = (float *)std::malloc((size_t)run.workspace_values * sizeof(float));
    if (!next) return -ENOMEM;
    std::free(workspace->values);
    workspace->values = next;
    workspace->capacity = run.workspace_values;
    return 0;
}

bool resampler_out_length(const LfmResampler &r, uint64_t length,
                          uint64_t *out) {
    if (r.orig == 0 || r.phases == 0 || !out) return false;
    const uint64_t whole = length / r.orig;
    const uint64_t tail = length % r.orig;
    uint64_t base = 0;
    uint64_t tail_scaled = 0;
    if (!mul_u64(whole, r.phases, &base) ||
        !mul_u64(tail, r.phases, &tail_scaled)) {
        return false;
    }
    const uint64_t extra = tail_scaled / r.orig +
                           (tail_scaled % r.orig != 0 ? 1 : 0);
    return add_u64(base, extra, out);
}

bool resampler_padded_length(const LfmResampler &r, uint64_t length,
                             uint64_t *out) {
    uint64_t edges = 0;
    return add_u64(2 * r.width, r.orig, &edges) &&
           add_u64(length, edges, out);
}

} // namespace

extern "C" int lfm_frontend_create(const LfmFrontendConfig *config, LfmFrontend **out) {
    if (!config || !out) return -EINVAL;
    *out = nullptr;
    if (config->size < sizeof(LfmFrontendConfig) || config->abi_version != LFM_FRONTEND_ABI)
        return -EINVAL;
    if (config->sample_rate == 0 || config->n_window_size == 0 ||
        config->n_window_stride == 0 || config->n_fft == 0 || config->nfilt == 0 ||
        config->n_window_size > config->n_fft ||
        (config->exact_pad && config->n_window_stride > config->n_fft))
        return -EINVAL;
    if (config->mag_power != 2.0) return -EOPNOTSUPP;

    LfmFrontend *f = new (std::nothrow) LfmFrontend();
    if (!f) return -ENOMEM;
    f->cfg = *config;
    f->freq = config->n_fft / 2 + 1;
    uint64_t fb_values = 0;
    uint64_t dft_rows = 0;
    uint64_t dft_values = 0;
    if (!mul_u64(config->nfilt, f->freq, &fb_values) ||
        !mul_u64(2, f->freq, &dft_rows) ||
        !mul_u64(dft_rows, config->n_fft, &dft_values) ||
        fb_values > std::numeric_limits<size_t>::max() / sizeof(float) ||
        dft_values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        delete f;
        return -EOVERFLOW;
    }
    f->fb = (float *)std::malloc((size_t)fb_values * sizeof(float));
    f->dft = (float *)std::malloc((size_t)dft_values * sizeof(float));
    if (!f->fb || !f->dft) {
        lfm_frontend_destroy(f);
        return -ENOMEM;
    }
    build_melfb(config->sample_rate, config->n_fft, config->nfilt, f->fb);
    if (!build_dft(config->n_fft, config->n_window_size, f->dft)) {
        lfm_frontend_destroy(f);
        return -ENOMEM;
    }
    *out = f;
    return 0;
}

extern "C" int lfm_frontend_destroy(LfmFrontend *f) {
    if (!f) return -EINVAL;
    std::free(f->fb);
    std::free(f->dft);
    delete f;
    return 0;
}

extern "C" uint64_t lfm_frontend_derived_bytes(const LfmFrontend *f) {
    if (!f) return 0;
    const uint64_t fb = (uint64_t)f->cfg.nfilt * f->freq;
    const uint64_t dft = (uint64_t)2 * f->freq * f->cfg.n_fft;
    return (fb + dft) * sizeof(float);
}

extern "C" int lfm_frontend_workspace_create(LfmFrontendWorkspace **out) {
    if (!out) return -EINVAL;
    *out = new (std::nothrow) LfmFrontendWorkspace();
    return *out ? 0 : -ENOMEM;
}

extern "C" int lfm_frontend_workspace_destroy(LfmFrontendWorkspace *workspace) {
    if (!workspace) return -EINVAL;
    delete workspace;
    return 0;
}

extern "C" int lfm_frontend_workspace_reserve(
    const LfmFrontend *f, LfmFrontendWorkspace *workspace,
    uint64_t max_sample_count, uint32_t flags) {
    return reserve_frontend(f, workspace, max_sample_count, flags);
}

extern "C" uint64_t lfm_frontend_seq_len(const LfmFrontend *f, uint64_t l) {
    return f ? seq_len_of(f->cfg, l) : 0;
}

extern "C" int lfm_frontend_out_frames(const LfmFrontend *f, uint64_t l,
                                       uint64_t *out_frames) {
    if (!f || !out_frames || l == 0) return -EINVAL;
    const LfmFrontendConfig &c = f->cfg;
    const uint64_t p = c.exact_pad ? ((uint64_t)c.n_fft - c.n_window_stride) / 2 : 0;
    uint64_t twice = 0;
    uint64_t li = 0;
    if (!mul_u64(2, p, &twice) || !add_u64(l, twice, &li)) return -EOVERFLOW;
    const uint64_t center = c.exact_pad ? 0 : (uint64_t)c.n_fft / 2;
    uint64_t t = 0;
    if (!frame_count(c, li, center, &t)) return -EOVERFLOW;
    if (c.pad_to > 0 && t % c.pad_to != 0 &&
        !add_u64(t, c.pad_to - t % c.pad_to, &t))
        return -EOVERFLOW;
    *out_frames = t;
    return 0;
}

namespace {

int frontend_forward(const LfmFrontend *f, LfmFrontendWorkspace *workspace,
                     const float *pcm, uint64_t l, void *out_mel,
                     uint64_t out_capacity_values, bool valid_only,
                     bool bf16_output) {
    if (!f || !workspace || !pcm || !out_mel || l == 0) return -EINVAL;
    const LfmFrontendConfig &c = f->cfg;
    const uint64_t freq = f->freq;
    FrontendRun run{};
    const int geometry = frontend_run(f, l, valid_only, bf16_output, &run);
    if (geometry != 0) return geometry;
    if (out_capacity_values < run.output_values) return -EINVAL;

    // seq_len == 0: the reference's normalization NaNs never survive because
    // the caller-side tail mask covers [0, t) — the entire clip. The defined
    // output is an all-zero plane; produce it without running the pipeline.
    if (run.seq_len == 0) {
        std::memset(out_mel, 0,
                    (size_t)run.output_values *
                        (bf16_output ? sizeof(uint16_t) : sizeof(float)));
        return 0;
    }

    std::lock_guard<std::mutex> guard(workspace->lock);
    if (workspace->capacity < run.workspace_values) return -ENOBUFS;
    float *a = workspace->values; // signal, then DFT/power
    float *b = a + run.plane_a_values; // frames, then mel or row statistics

    // PCM is always borrowed. Exact padding is represented as a logical source
    // offset during frame gather, not materialized with memcpy. Preemphasis can
    // likewise write straight into its final logical offset because the first
    // retained sample's predecessor is the virtual zero pad. The reference's
    // unusual post-preemphasis mask remains the logical [L, Li) interval.
    const float *signal = pcm;
    const uint64_t signal_offset = c.exact_pad ? run.pad : 0;
    if (c.preemph != 0.0 && l > 1) {
        float *preemphasized = a + signal_offset;
        lfm_preemph_f32(pcm, preemphasized, l, (float)c.preemph);
        signal = preemphasized;
    }

    // Every gathered cell is assigned; stale high-water workspace values are
    // never observed and the whole block does not need calloc/zero-fill.
    for (uint64_t tt = 0; tt < run.cols; ++tt) {
        const uint64_t start = tt * c.n_window_stride; // index into ypad
        for (uint64_t n = 0; n < c.n_fft; ++n) {
            const uint64_t idx = start + n;
            const uint64_t source = idx >= run.center ? idx - run.center : 0;
            b[n * run.cols + tt] = idx >= run.center &&
                                            source >= signal_offset && source < l
                                        ? signal[source - signal_offset]
                                        : 0.0f;
        }
    }

    // Frames are dead after DFT, and signal is dead before it: the two planes
    // alternate ownership. Power aliases the real DFT half exactly, and mel is
    // written directly into the caller's final row stride.
    sgemm_rm((int)(2 * freq), (int)run.cols, (int)c.n_fft, f->dft,
             (int)c.n_fft, b, (int)run.cols, a, (int)run.cols);
    lfm_power_spec_f32(a, a + run.power_values, a, run.power_values);
    float *mel = bf16_output ? b : static_cast<float *>(out_mel);
    const uint64_t mel_stride = bf16_output ? run.cols : run.stride;
    sgemm_rm((int)c.nfilt, (int)run.cols, (int)freq, f->fb, (int)freq,
             a, (int)run.cols, mel, (int)mel_stride);

    float *mean = bf16_output ? a : b;
    float *stdv = b + c.nfilt;
    if (bf16_output) stdv = a + c.nfilt;
    for (uint64_t r = 0; r < c.nfilt; ++r) {
        float *row = mel + r * mel_stride;
        if (lfm_log_add_f32(row, run.seq_len,
                            (float)c.log_zero_guard_value) != 0 ||
            lfm_rowstat_f32(row, 1, run.seq_len, run.seq_len, 1e-5f, mean + r,
                            stdv + r) != 0)
            return -EINVAL;
        lfm_norm_apply_f32(row, mean + r, stdv + r, 1, run.seq_len);
        if (bf16_output) {
            lfm_f32_to_bf16(row,
                            static_cast<uint16_t *>(out_mel) + r * run.cols,
                            (int)run.cols);
        } else if (!valid_only && run.stride > run.seq_len) {
            std::memset(row + run.seq_len, 0,
                        (size_t)(run.stride - run.seq_len) * sizeof(float));
        }
    }
    return 0;
}

} // namespace

extern "C" int lfm_frontend_forward_workspace(
    const LfmFrontend *f, LfmFrontendWorkspace *workspace, const float *pcm,
    uint64_t l, float *out_mel, uint64_t out_capacity_values, uint32_t flags) {
    if (flags & ~LFM_FRONTEND_FORWARD_VALID_ONLY) return -EINVAL;
    return frontend_forward(f, workspace, pcm, l, out_mel,
                            out_capacity_values,
                            (flags & LFM_FRONTEND_FORWARD_VALID_ONLY) != 0,
                            false);
}

extern "C" int lfm_frontend_forward_bf16_workspace(
    const LfmFrontend *f, LfmFrontendWorkspace *workspace, const float *pcm,
    uint64_t l, uint16_t *out_mel, uint64_t out_capacity_values) {
    return frontend_forward(f, workspace, pcm, l, out_mel,
                            out_capacity_values, true, true);
}

extern "C" int lfm_frontend_forward(const LfmFrontend *f, const float *pcm,
                                    uint64_t l, float *out_mel,
                                    uint64_t out_capacity_values) {
    LfmFrontendWorkspace workspace;
    const int reserved = reserve_frontend(f, &workspace, l, 0);
    if (reserved != 0) return reserved;
    return frontend_forward(f, &workspace, pcm, l, out_mel,
                            out_capacity_values, false, false);
}

extern "C" int lfm_frontend_forward_valid(const LfmFrontend *f,
                                          const float *pcm, uint64_t l,
                                          float *out_mel,
                                          uint64_t out_capacity_values) {
    LfmFrontendWorkspace workspace;
    const int reserved = reserve_frontend(
        f, &workspace, l, LFM_FRONTEND_FORWARD_VALID_ONLY);
    if (reserved != 0) return reserved;
    return frontend_forward(f, &workspace, pcm, l, out_mel,
                            out_capacity_values, true, false);
}

extern "C" int lfm_resampler_create(uint32_t orig_freq, uint32_t new_freq,
                                    LfmResampler **out) {
    if (!out || orig_freq == 0 || new_freq == 0) return -EINVAL;
    *out = nullptr;
    LfmResampler *r = new (std::nothrow) LfmResampler();
    if (!r) return -ENOMEM;
    r->orig_freq = orig_freq;
    r->new_freq = new_freq;

    uint64_t gcd = orig_freq;
    uint64_t b = new_freq;
    while (b != 0) {
        const uint64_t next = gcd % b;
        gcd = b;
        b = next;
    }
    r->orig = orig_freq / gcd;
    r->phases = new_freq / gcd;
    if (orig_freq == new_freq) {
        *out = r;
        return 0;
    }

    const double base = (double)std::min(r->orig, r->phases) * 0.99;
    r->width = (uint64_t)std::ceil(6.0 * (double)r->orig / base);
    if (!add_u64(2 * r->width, r->orig, &r->kernel_len)) {
        delete r;
        return -EOVERFLOW;
    }
    uint64_t kernel_values = 0;
    if (!mul_u64(r->phases, r->kernel_len, &kernel_values) ||
        kernel_values > std::numeric_limits<size_t>::max() / sizeof(double)) {
        delete r;
        return -EOVERFLOW;
    }
    r->kernels = (double *)std::malloc((size_t)kernel_values * sizeof(double));
    if (!r->kernels) {
        delete r;
        return -ENOMEM;
    }
    const double scale = base / (double)r->orig;
    for (uint64_t phase = 0; phase < r->phases; ++phase) {
        for (uint64_t j = 0; j < r->kernel_len; ++j) {
            const int64_t idx = -(int64_t)r->width + (int64_t)j;
            double tt = (-(double)phase / (double)r->phases +
                         (double)idx / (double)r->orig) *
                        base;
            if (tt < -6.0) tt = -6.0;
            if (tt > 6.0) tt = 6.0;
            const double window =
                std::pow(std::cos(tt * M_PI / 6.0 / 2.0), 2.0);
            const double tp = tt * M_PI;
            const double sinc = tp == 0.0 ? 1.0 : std::sin(tp) / tp;
            r->kernels[(size_t)phase * r->kernel_len + (size_t)j] =
                sinc * window * scale;
        }
    }
    *out = r;
    return 0;
}

extern "C" int lfm_resampler_destroy(LfmResampler *resampler) {
    if (!resampler) return -EINVAL;
    delete resampler;
    return 0;
}

extern "C" uint64_t lfm_resampler_derived_bytes(
    const LfmResampler *resampler) {
    if (!resampler || !resampler->kernels) return 0;
    uint64_t values = 0;
    return mul_u64(resampler->phases, resampler->kernel_len, &values)
               ? values * sizeof(double)
               : 0;
}

extern "C" int lfm_resampler_out_length(const LfmResampler *resampler,
                                         uint64_t sample_count,
                                         uint64_t *out_length) {
    if (!resampler || !out_length) return -EINVAL;
    return resampler_out_length(*resampler, sample_count, out_length)
               ? 0
               : -EOVERFLOW;
}

extern "C" int lfm_resampler_workspace_create(
    LfmResamplerWorkspace **out) {
    if (!out) return -EINVAL;
    *out = new (std::nothrow) LfmResamplerWorkspace();
    return *out ? 0 : -ENOMEM;
}

extern "C" int lfm_resampler_workspace_destroy(
    LfmResamplerWorkspace *workspace) {
    if (!workspace) return -EINVAL;
    delete workspace;
    return 0;
}

extern "C" int lfm_resampler_workspace_reserve(
    const LfmResampler *resampler, LfmResamplerWorkspace *workspace,
    uint64_t max_sample_count) {
    if (!resampler || !workspace) return -EINVAL;
    if (resampler->orig_freq == resampler->new_freq) return 0;
    uint64_t padded_len = 0;
    if (!resampler_padded_length(*resampler, max_sample_count, &padded_len) ||
        padded_len > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return -EOVERFLOW;
    }
    std::lock_guard<std::mutex> guard(workspace->lock);
    if (workspace->capacity >= padded_len) return 0;
    double *next = (double *)std::malloc((size_t)padded_len * sizeof(double));
    if (!next) return -ENOMEM;
    std::free(workspace->padded);
    workspace->padded = next;
    workspace->capacity = padded_len;
    return 0;
}

extern "C" int lfm_resampler_process(
    const LfmResampler *resampler, LfmResamplerWorkspace *workspace,
    const float *input, uint64_t sample_count, float *destination,
    uint64_t destination_capacity, LfmF32Span *result) {
    if (!resampler || !workspace || !result ||
        (!input && sample_count != 0)) {
        return -EINVAL;
    }
    result->data = nullptr;
    result->length = 0;
    uint64_t target = 0;
    if (!resampler_out_length(*resampler, sample_count, &target)) {
        return -EOVERFLOW;
    }
    if (resampler->orig_freq == resampler->new_freq || sample_count == 0) {
        result->data = input;
        result->length = sample_count;
        return 0;
    }
    if (!destination || destination_capacity < target) return -EINVAL;

    uint64_t padded_len = 0;
    if (!resampler_padded_length(*resampler, sample_count, &padded_len)) {
        return -EOVERFLOW;
    }
    std::lock_guard<std::mutex> guard(workspace->lock);
    if (workspace->capacity < padded_len) return -ENOBUFS;
    double *padded = workspace->padded;
    if (resampler->width > 0) {
        std::memset(padded, 0, (size_t)resampler->width * sizeof(double));
    }
    for (uint64_t i = 0; i < sample_count; ++i) {
        padded[resampler->width + i] = (double)input[i];
    }
    const uint64_t suffix = resampler->width + resampler->orig;
    if (suffix > 0) {
        std::memset(padded + resampler->width + sample_count, 0,
                    (size_t)suffix * sizeof(double));
    }
    lfm_resample_conv_f64(
        padded, padded_len, resampler->kernels, resampler->phases,
        resampler->kernel_len, resampler->orig, destination, target);
    result->data = destination;
    result->length = target;
    return 0;
}

extern "C" int lfm_resample_f32(const float *x, uint64_t length,
                                uint32_t orig_freq, uint32_t new_freq,
                                float *out, uint64_t out_capacity,
                                uint64_t *out_length) {
    if ((!x && length != 0) || !out || !out_length || orig_freq == 0 ||
        new_freq == 0) {
        return -EINVAL;
    }
    LfmResampler *plan = nullptr;
    int status = lfm_resampler_create(orig_freq, new_freq, &plan);
    if (status != 0) return status;
    LfmResamplerWorkspace *workspace = nullptr;
    status = lfm_resampler_workspace_create(&workspace);
    if (status == 0) {
        status = lfm_resampler_workspace_reserve(plan, workspace, length);
    }
    LfmF32Span span{};
    if (status == 0) {
        status = lfm_resampler_process(plan, workspace, x, length, out,
                                       out_capacity, &span);
    }
    if (status == 0 && span.data != out) {
        if (out_capacity < span.length) {
            status = -EINVAL;
        } else if (span.length != 0) {
            std::memcpy(out, span.data, (size_t)span.length * sizeof(float));
        }
    }
    if (status == 0) *out_length = span.length;
    if (workspace) (void)lfm_resampler_workspace_destroy(workspace);
    (void)lfm_resampler_destroy(plan);
    return status;
}
