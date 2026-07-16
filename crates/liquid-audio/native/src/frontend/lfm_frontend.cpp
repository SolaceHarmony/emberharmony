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
void lfm_resample_conv_f64(const double *padded, uint64_t padded_len,
                           const double *kernels, uint64_t phases, uint64_t klen,
                           uint64_t stride, float *out, uint64_t max_blocks);
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
                     const float *pcm, uint64_t l, float *out_mel,
                     uint64_t out_capacity_values, bool valid_only) {
    if (!f || !workspace || !pcm || !out_mel || l == 0) return -EINVAL;
    const LfmFrontendConfig &c = f->cfg;
    const uint64_t freq = f->freq;
    const uint64_t seq_len = seq_len_of(c, l);

    const uint64_t p = c.exact_pad ? ((uint64_t)c.n_fft - c.n_window_stride) / 2 : 0;
    uint64_t twice = 0;
    uint64_t li = 0;
    if (!mul_u64(2, p, &twice) || !add_u64(l, twice, &li)) return -EOVERFLOW;
    const uint64_t center = c.exact_pad ? 0 : (uint64_t)c.n_fft / 2;
    uint64_t t = 0;
    if (!frame_count(c, li, center, &t)) return -EOVERFLOW;
    if (t == 0 || seq_len > t) return -EINVAL;
    uint64_t t_out = t;
    if (c.pad_to > 0 && t % c.pad_to != 0 &&
        !add_u64(t, c.pad_to - t % c.pad_to, &t_out))
        return -EOVERFLOW;
    const uint64_t cols = valid_only ? seq_len : t;
    const uint64_t stride = valid_only ? seq_len : t_out;
    uint64_t output_values = 0;
    if (!mul_u64(c.nfilt, stride, &output_values) ||
        output_values > std::numeric_limits<size_t>::max() / sizeof(float))
        return -EOVERFLOW;
    if (out_capacity_values < output_values) return -EINVAL;
    if (valid_only && seq_len == 0) return -EINVAL;

    // seq_len == 0: the reference's normalization NaNs never survive because
    // the caller-side tail mask covers [0, t) — the entire clip. The defined
    // output is an all-zero plane; produce it without running the pipeline.
    if (seq_len == 0) {
        std::memset(out_mel, 0, (size_t)output_values * sizeof(float));
        return 0;
    }

    uint64_t n_frames = 0;
    uint64_t n_stft = 0;
    uint64_t n_power = 0;
    uint64_t stats = 0;
    if (!mul_u64(c.n_fft, cols, &n_frames) ||
        !mul_u64(2 * freq, cols, &n_stft) ||
        !mul_u64(freq, cols, &n_power) || !mul_u64(2, c.nfilt, &stats))
        return -EOVERFLOW;
    const uint64_t n_a = std::max(li, n_stft);
    const uint64_t n_b = std::max(n_frames, stats);
    uint64_t total = 0;
    if (!add_u64(n_a, n_b, &total) ||
        total > std::numeric_limits<size_t>::max() / sizeof(float) ||
        cols > INT_MAX || c.n_fft > INT_MAX || c.nfilt > INT_MAX ||
        freq > INT_MAX || 2 * freq > INT_MAX || stride > INT_MAX)
        return -EOVERFLOW;

    std::lock_guard<std::mutex> guard(workspace->lock);
    if (workspace->capacity < total) {
        float *next = (float *)std::malloc((size_t)total * sizeof(float));
        if (!next) return -ENOMEM;
        std::free(workspace->values);
        workspace->values = next;
        workspace->capacity = total;
    }
    float *a = workspace->values; // signal, then DFT/power
    float *b = a + n_a;           // frames, then row statistics

    // Centered production reads PCM directly into the preemphasis leaf. Exact
    // pad initializes only its actual prefix/suffix before applying that leaf
    // in place. The reference's unusual post-preemphasis mask remains [L, Li).
    const float *signal = pcm;
    if (c.exact_pad) {
        if (p > 0) std::memset(a, 0, (size_t)p * sizeof(float));
        std::memcpy(a + p, pcm, (size_t)l * sizeof(float));
        if (p > 0) std::memset(a + p + l, 0, (size_t)p * sizeof(float));
        signal = a;
        if (c.preemph != 0.0 && li > 1)
            lfm_preemph_f32(a, a, li, (float)c.preemph);
    } else if (c.preemph != 0.0 && li > 1) {
        lfm_preemph_f32(pcm, a, li, (float)c.preemph);
        signal = a;
    }
    if (li > l) std::memset(a + l, 0, (size_t)(li - l) * sizeof(float));

    // Every gathered cell is assigned; stale high-water workspace values are
    // never observed and the whole block does not need calloc/zero-fill.
    for (uint64_t tt = 0; tt < cols; ++tt) {
        const uint64_t start = tt * c.n_window_stride; // index into ypad
        for (uint64_t n = 0; n < c.n_fft; ++n) {
            const uint64_t idx = start + n;
            b[n * cols + tt] =
                idx >= center && idx < center + li ? signal[idx - center] : 0.0f;
        }
    }

    // Frames are dead after DFT, and signal is dead before it: the two planes
    // alternate ownership. Power aliases the real DFT half exactly, and mel is
    // written directly into the caller's final row stride.
    sgemm_rm((int)(2 * freq), (int)cols, (int)c.n_fft, f->dft,
             (int)c.n_fft, b, (int)cols, a, (int)cols);
    lfm_power_spec_f32(a, a + n_power, a, n_power);
    sgemm_rm((int)c.nfilt, (int)cols, (int)freq, f->fb, (int)freq, a,
             (int)cols, out_mel, (int)stride);

    float *mean = b;
    float *stdv = b + c.nfilt;
    for (uint64_t r = 0; r < c.nfilt; ++r) {
        float *row = out_mel + r * stride;
        if (lfm_log_add_f32(row, seq_len, (float)c.log_zero_guard_value) != 0 ||
            lfm_rowstat_f32(row, 1, seq_len, seq_len, 1e-5f, mean + r,
                            stdv + r) != 0)
            return -EINVAL;
        lfm_norm_apply_f32(row, mean + r, stdv + r, 1, seq_len);
        if (!valid_only && stride > seq_len)
            std::memset(row + seq_len, 0,
                        (size_t)(stride - seq_len) * sizeof(float));
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
                            (flags & LFM_FRONTEND_FORWARD_VALID_ONLY) != 0);
}

extern "C" int lfm_frontend_forward(const LfmFrontend *f, const float *pcm,
                                    uint64_t l, float *out_mel,
                                    uint64_t out_capacity_values) {
    LfmFrontendWorkspace workspace;
    return frontend_forward(f, &workspace, pcm, l, out_mel,
                            out_capacity_values, false);
}

extern "C" int lfm_frontend_forward_valid(const LfmFrontend *f,
                                          const float *pcm, uint64_t l,
                                          float *out_mel,
                                          uint64_t out_capacity_values) {
    LfmFrontendWorkspace workspace;
    return frontend_forward(f, &workspace, pcm, l, out_mel,
                            out_capacity_values, true);
}

extern "C" int lfm_resample_f32(const float *x, uint64_t length, uint32_t orig_freq,
                                uint32_t new_freq, float *out, uint64_t out_capacity,
                                uint64_t *out_length) {
    if (!x || !out || !out_length || orig_freq == 0 || new_freq == 0) return -EINVAL;
    if (orig_freq == new_freq || length == 0) {
        if (out_capacity < length) return -EINVAL;
        std::memcpy(out, x, (size_t)length * sizeof(float));
        *out_length = length;
        return 0;
    }
    // gcd-reduced rates; torchaudio defaults lowpass_filter_width=6, rolloff=0.99.
    uint64_t a = orig_freq, b = new_freq;
    while (b != 0) {
        const uint64_t r = a % b;
        a = b;
        b = r;
    }
    const int64_t orig = (int64_t)(orig_freq / a);
    const int64_t nw = (int64_t)(new_freq / a);
    const double base = (double)(orig < nw ? orig : nw) * 0.99;
    const int64_t width = (int64_t)std::ceil(6.0 * (double)orig / base);
    const uint64_t klen = (uint64_t)(2 * width + orig);
    const double scale = base / (double)orig;

    const uint64_t target =
        (uint64_t)std::ceil(((double)nw * (double)length) / (double)orig);
    if (out_capacity < target) return -EINVAL;

    double *kernels = (double *)std::malloc((size_t)nw * klen * sizeof(double));
    const uint64_t padded_len = (uint64_t)width + length + (uint64_t)(width + orig);
    double *padded = (double *)std::calloc(padded_len, sizeof(double));
    const uint64_t blocks = padded_len >= klen ? (padded_len - klen) / (uint64_t)orig + 1 : 0;
    float *conv = (float *)std::malloc((size_t)(blocks * (uint64_t)nw + 1) * sizeof(float));
    if (!kernels || !padded || !conv) {
        std::free(kernels);
        std::free(padded);
        std::free(conv);
        return -ENOMEM;
    }
    // One kernel per output phase i: t = (-i/new + idx/orig)*base clamped to
    // +/-6; hann^2 window; sinc; times scale. f64 throughout (the reference).
    for (int64_t i = 0; i < nw; ++i) {
        for (int64_t j = 0; j < 2 * width + orig; ++j) {
            const int64_t idx = -width + j;
            double tt = (-(double)i / (double)nw + (double)idx / (double)orig) * base;
            if (tt < -6.0) tt = -6.0;
            if (tt > 6.0) tt = 6.0;
            const double window = std::pow(std::cos(tt * M_PI / 6.0 / 2.0), 2.0);
            const double tp = tt * M_PI;
            const double sinc = tp == 0.0 ? 1.0 : std::sin(tp) / tp;
            kernels[(size_t)i * klen + (size_t)j] = sinc * window * scale;
        }
    }
    for (uint64_t i = 0; i < length; ++i) padded[(uint64_t)width + i] = (double)x[i];

    lfm_resample_conv_f64(padded, padded_len, kernels, (uint64_t)nw, klen,
                          (uint64_t)orig, conv, blocks);

    std::memcpy(out, conv, (size_t)target * sizeof(float));
    *out_length = target;
    std::free(conv);
    std::free(padded);
    std::free(kernels);
    return 0;
}
