// TiRex-2 sLSTM step — scalar reference implementation.
// Semantics transcribed from flashrnn 1.0.5 vanilla/slstm.py +
// tirex2 flashrnn_slstm.py; every deviation-relevant detail is documented in
// TIREX2_PORT.md. Parity target: torch CPU fp32.
// Build (standalone, no deps):
//   clang++ -std=c++17 -O2 -ffp-contract=off -c tirex2_slstm.cpp

#include "tirex2_kernel.h"

#include <cassert>
#include <cmath>

namespace tirex2 {

// torch.nn.functional.logsigmoid(x) computes -softplus(-x); torch's softplus
// at threshold=20 passes through linearly. Elementwise parity with torch CPU
// must be verified against the oracle before the NEON twin is written.
static inline float log_sigmoid(float x) {
    float nx = -x;
    float sp = (nx > 20.0f) ? nx : std::log1p(std::exp(nx));
    return -sp;
}

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void slstm_step_ref(const ModelConfig& cfg, const SlstmWeights& w, SlstmState& s,
                    const float* x, const float* x_conv, float* out,
                    int head_begin, int head_end) {
    const int H = cfg.num_slstm_heads;
    const int D = cfg.head_dim_slstm;
    assert(head_begin >= 0 && head_end <= H && D * H == cfg.embedding_dim);
    const bool first_step = (s.step_count == 0);

    for (int h = head_begin; h < head_end; ++h) {
        const float* xh  = x + h * D;       // z/o projections read raw x
        const float* xch = x_conv + h * D;  // f/i projections read the conv path
        float* yh = s.y + h * D;
        float* ch = s.c + h * D;
        float* nh = s.n + h * D;
        float* mh = s.m + h * D;

        // Headwise block-diagonal gate projections (bias=False), stacked into
        // pointwise slots 0..3 as (f_proj, i_proj, z_proj, o_proj) — the
        // layer's wiring, NOT the pointwise's (i,f,z,o) labels. See PORT.md.
        // Wx[g][j] for this head; g indexes the POINTWISE slot.
        float Wx0, Wx1, Wx2, Wx3;
        // Ry[g][j] = sum_p y_prev[p] * R[h, p, g, j]  (R stored [H][P][4][D])
        const float* Rh = w.R + (size_t)h * D * 4 * D;
        const float* bh = w.b + (size_t)h * 4 * D;

        // Precompute Ry for all four slots into stack buffers.
        // D is checkpoint-fixed and small (embedding_dim / heads); 256 covers
        // any plausible config — hard-error otherwise, no fallback.
        assert(D <= 256);
        float Ry[4][256];
        for (int g = 0; g < 4; ++g)
            for (int j = 0; j < D; ++j) Ry[g][j] = 0.0f;
        for (int p = 0; p < D; ++p) {
            const float yp = yh[p];
            const float* Rp = Rh + (size_t)p * 4 * D;
            for (int g = 0; g < 4; ++g) {
                const float* Rpg = Rp + (size_t)g * D;
                for (int j = 0; j < D; ++j) Ry[g][j] += yp * Rpg[j];
            }
        }

        for (int j = 0; j < D; ++j) {
            // Gate projections: row j of each head-block d×d matrix.
            const float* pf = w.gate_proj_f + ((size_t)h * D + j) * D;
            const float* pi = w.gate_proj_i + ((size_t)h * D + j) * D;
            const float* pz = w.gate_proj_z + ((size_t)h * D + j) * D;
            const float* po = w.gate_proj_o + ((size_t)h * D + j) * D;
            float af = 0.0f, ai = 0.0f, az = 0.0f, ao = 0.0f;
            for (int p = 0; p < D; ++p) {
                af += pf[p] * xch[p];
                ai += pi[p] * xch[p];
                az += pz[p] * xh[p];
                ao += po[p] * xh[p];
            }
            Wx0 = af; Wx1 = ai; Wx2 = az; Wx3 = ao;

            const float i_raw = Wx0 + Ry[0][j] + bh[0 * D + j];
            const float f_raw = Wx1 + Ry[1][j] + bh[1 * D + j];
            const float z_raw = Wx2 + Ry[2][j] + bh[2 * D + j];
            const float o_raw = Wx3 + Ry[3][j] + bh[3 * D + j];

            const float logfplusm = log_sigmoid(f_raw) + mh[j];
            const float m_new = first_step ? i_raw
                                           : (i_raw > logfplusm ? i_raw : logfplusm);
            const float og = sigmoid(o_raw);
            const float ig = std::exp(i_raw - m_new);
            const float fg = std::exp(logfplusm - m_new);
            const float c_new = fg * ch[j] + ig * std::tanh(z_raw);
            const float n_pre = fg * nh[j] + ig;
            const float n_new = n_pre > 1.0f ? n_pre : 1.0f;
            const float y_new = og * c_new / n_new;

            ch[j] = c_new; nh[j] = n_new; mh[j] = m_new; yh[j] = y_new;
        }

        // MultiHeadLayerNorm over this head's y (weight, no bias, eps 1e-6).
        // force_float32_reductions=True and inputs are fp32: accumulate in
        // FP32, sequentially — NOT double. Accumulation precision/order is
        // parity-critical (the LFM layernorm lesson); if the oracle diverges
        // here, match torch CPU's reduction tree before touching anything else.
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) mean += yh[j];
        mean /= (float)D;
        float var = 0.0f;
        for (int j = 0; j < D; ++j) {
            const float d = yh[j] - mean;
            var += d * d;
        }
        var /= (float)D;
        const float inv = 1.0f / std::sqrt(var + cfg.norm_eps);
        const float* gw = w.group_norm_w + (size_t)h * D;
        float* oh = out + (size_t)h * D;
        for (int j = 0; j < D; ++j)
            oh[j] = (yh[j] - mean) * inv * gw[j];
    }
    // step_count increment belongs to the caller once ALL heads of ALL lanes
    // have run this step (a lane must not flip first_step for its siblings).
}

void slstm_step(const ModelConfig& cfg, const SlstmWeights& w, SlstmState& s,
                const float* x, const float* x_conv, float* out,
                int head_begin, int head_end) {
    // NEON twin lands only after slstm_step_ref matches the torch oracle.
    slstm_step_ref(cfg, w, s, x, x_conv, out, head_begin, head_end);
}

}  // namespace tirex2
