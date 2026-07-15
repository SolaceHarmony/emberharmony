// TiRex-2 CPU kernel — standalone shaping stage (not linked by build.rs yet).
// Source of truth for all semantics: TIREX2_PORT.md in this directory.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tirex2 {

// Filled from the checkpoint's model config once the HF gate is open.
// Hard-error (assert) on anything unsupported — no fallbacks.
struct ModelConfig {
    int embedding_dim;      // D
    int num_blocks;         // stack depth (12 per paper; config decides)
    int num_heads;          // mLSTM / attention heads
    int num_slstm_heads;    // sLSTM heads (may differ)
    int head_dim_slstm;     // embedding_dim / num_slstm_heads
    int input_patch_size;   // P_in (paper: 32)
    int output_patch_size;  // P_out (== P_in for inference; asserted upstream)
    int input_ff_dim;       // patch-embed hidden
    int num_quantiles;      // K (9 per release notes; config decides)
    int conv1d_kernel_size; // 0 = no conv path in cells
    float norm_eps;         // MultiHeadLayerNorm/RMSNorm eps
    float scaler_eps;       // 1e-8
    bool scaler_arcsinh;
    bool scaler_binaryaware;
};

// ---- sLSTM ----------------------------------------------------------------
// Per-head recurrent weights as stored by the checkpoint:
//   R: [num_slstm_heads][head_dim (prev-y axis P)][4 gates][head_dim (out D)]
//   b: [num_slstm_heads][4 gates][head_dim]
// POINTWISE slot semantics are (i, f, z, o) — but the layer stacks its gate
// PROJECTIONS into Wx as (f_proj, i_proj, z_proj, o_proj). The trained
// weights are self-consistent with that wiring. Replicate the data path;
// do not relabel. See TIREX2_PORT.md "GATE-ORDER PLUMBING".
struct SlstmWeights {
    const float* gate_proj_f;  // headwise block-diag d×d per head → Wx slot 0
    const float* gate_proj_i;  // → Wx slot 1
    const float* gate_proj_z;  // → Wx slot 2
    const float* gate_proj_o;  // → Wx slot 3
    const float* R;            // [H][P][4][D]
    const float* b;            // [H][4][D]
    const float* group_norm_w; // MultiHeadLayerNorm weight [H*D] (bias absent)
    const float* conv_w;       // CausalConv1d depthwise [D][k] or nullptr
    const float* conv_b;       // [D] or nullptr
};

// Streaming state for ONE stream. (y,c,n,m) per head — the hibernation blob.
// step_count carries the vanilla backend's whole-tensor `all(n==0)` first-step
// branch as a per-stream counter (n >= 1 after the first step, so the branch
// is exactly step_count == 0 when states start zeroed).
struct SlstmState {
    float* y;  // [H][D]
    float* c;  // [H][D]
    float* n;  // [H][D]
    float* m;  // [H][D]
    float* conv_ring;  // [D][k-1] or nullptr
    uint64_t step_count;
};

// One recurrent step for a contiguous run of heads [head_begin, head_end).
// x is the token AFTER the layer's RMSNorm; x_conv is the SiLU'd conv path
// (== x when conv1d_kernel_size == 0). Writes group-normed output to out.
// Head-sliced so a kcoro lane can own its heads with R pinned (lanes-as-heads).
void slstm_step(const ModelConfig& cfg, const SlstmWeights& w, SlstmState& s,
                const float* x, const float* x_conv, float* out,
                int head_begin, int head_end);

// Scalar reference twin — bit-identical target for the NEON version.
void slstm_step_ref(const ModelConfig& cfg, const SlstmWeights& w, SlstmState& s,
                    const float* x, const float* x_conv, float* out,
                    int head_begin, int head_end);

// ---- mLSTM ----------------------------------------------------------------
// Equations NOT yet transcribed from xlstm/xlstm_large/model.py — declared so
// the block topology compiles; implementation is a hard TODO gated on that
// read (TIREX2_PORT.md "mLSTM cell"). Do not guess the math.
struct MlstmWeights;
struct MlstmState;  // per head: C[dqk][dv], n[dqk], m — fixed size

// ---- Input path -----------------------------------------------------------
// Scaler: loc = nanmean, scale = sqrt(nanmean((x-loc)^2)) clamped >= eps.
struct ScalerState { float loc, scale; bool is_binary; };
void scaler_fit(const ModelConfig& cfg, const float* x, size_t len, ScalerState& st);

// Patch + NaN-mask + embed: emits [2*P_in] (values nan_to_num'd, mask) then
// ResidualBlock: out = W2·act(W1·in + b1) + b2 + Wr·in + br.
struct PatchEmbedWeights {
    const float* w1; const float* b1;   // [ff][2P], [ff]
    const float* w2; const float* b2;   // [D][ff], [D]
    const float* wr; const float* br;   // [D][2P], [D]
};
void patch_embed(const ModelConfig& cfg, const PatchEmbedWeights& w,
                 const float* patch_scaled_masked, float* token_out);

}  // namespace tirex2
