// Native Conformer encoder + audio adapter ABI. Replaces the Rust/Candle
// owners in src/model/conformer/* and the adapter's Candle path; parity is
// gated by native/tests/fixtures/conformer/ (captured from the deleted Rust
// with real checkpoint weights on the production BF16 ladder).
//
// Execution: one segment forward is a lane-uniform Flashkern program submitted
// through lfm_engine_call — barriers via lfm_lane_fence, tiles via atomic
// claims, planes swapped at serial transitions. C++ sequences stages and moves
// bytes; every value is produced by an architecture assembly leaf
// (flashkern_conformer.S) or the approved matmul dispatch (bf16 GEMM leaves;
// f32 GEMM via Accelerate on Apple per the doc 09 split, lane-tiled scalar
// leaf elsewhere).
//
// Numerics: the production ladder, not an idealization. Linears stream bf16
// activations and checkpoint-layout weights directly, accumulate f32, add f32
// bias, then round bf16. Convolutions likewise unlift bf16 activation/tap words
// only in registers before the f32 bias and bf16 round boundaries. LayerNorm
// computes f32 statistics and applies weight/bias in
// bf16 arithmetic (layer_norm_slow). BatchNorm eval runs the all-bf16
// broadcast chain. Attention scores/probs/aggregation are f32. SiLU and
// gelu_erf round once; GLU rounds sigmoid then product (candle op-for-op).
#ifndef LFM_CONFORMER_H
#define LFM_CONFORMER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM_CONFORMER_ABI 1u

typedef struct LfmConformer LfmConformer;
typedef struct LfmConformerWorkspace LfmConformerWorkspace;

typedef struct LfmConformerGeometry {
    uint32_t size;
    uint32_t abi_version;
    uint32_t feat_in;        // mel bins (128)
    uint32_t d_model;        // 512
    uint32_t n_layers;       // 17
    uint32_t n_heads;        // 8
    uint32_t d_ff;           // 2048
    uint32_t conv_kernel;    // 9
    uint32_t subsampling;    // factor (8) — dw_striding only
    uint32_t conv_channels;  // 256
    uint32_t adapter_hidden; // 2048
    uint32_t adapter_out;    // 2048 (backbone hidden)
    uint64_t reserved[4];
} LfmConformerGeometry;

// Binds every encoder/adapter weight as views into the resident safetensors
// image (the same image the model owns; no duplicate bytes, no name lookups
// after this call). `weights` is the LfmWeights handle from lfm_weights_open.
// `engine` is the resident Flashkern engine (lfm_engine_new) whose lane team
// executes segment passes. Returns 0; -EINVAL on nulls/bad geometry; -ENOENT
// with `error` filled when a required tensor is missing or mis-shaped.
int lfm_conformer_create(void *engine, const void *weights,
                         const LfmConformerGeometry *geometry,
                         LfmConformer **out, char *error, size_t error_length);
int lfm_conformer_destroy(LfmConformer *conformer);

// Immutable residency accounting. `derived_bytes` is limited to formula-
// derived tables (BN denominators and relative-position frequencies). Bound
// checkpoint bytes remain views into the owner image. Materialized bytes must
// remain zero for every forward; direct GEMM calls is an execution witness for
// steady-state tests.
uint64_t lfm_conformer_bound_weight_bytes(const LfmConformer *conformer);
uint64_t lfm_conformer_derived_bytes(const LfmConformer *conformer);
uint64_t lfm_conformer_materialized_weight_bytes(const LfmConformer *conformer);
uint64_t lfm_conformer_direct_gemm_calls(const LfmConformer *conformer);

// Session-owned mutable planes. Production reserves the maximum admitted mel
// segment before readiness; subsequent forwards never grow and return
// -ENOBUFS if admission is violated. An unreserved workspace retains the
// transitional grow-on-first-forward behavior used by parity tests.
int lfm_conformer_workspace_create(LfmConformerWorkspace **out);
int lfm_conformer_workspace_destroy(LfmConformerWorkspace *workspace);
int lfm_conformer_workspace_reserve(const LfmConformer *conformer,
                                    LfmConformerWorkspace *workspace,
                                    uint64_t max_mel_frames);

// Output rows for a mel segment of `mel_frames`: the dw_striding length chain
// (three k3/s2/p1 stages), matching the Rust calc_length/mel2emb_len contract.
uint64_t lfm_conformer_out_rows(const LfmConformer *conformer, uint64_t mel_frames);

// mel: row-major (feat_in x mel_frames) BF16 bits — exactly the ChatState
// audio_in segment layout after the prefill BF16 cast.
// out_rows: row-major (out_rows x adapter_out) BF16 bits — adapted embedding
// rows, written to the caller's destination (the borrowed prefill plane).
// Blocks until the lane team completes the segment. Returns 0; -EINVAL on
// nulls, zero frames, or undersized capacity.
int lfm_conformer_forward(const LfmConformer *conformer,
                          LfmConformerWorkspace *workspace,
                          const uint16_t *mel, uint64_t mel_frames,
                          uint16_t *out_rows, uint64_t out_capacity_values);

#ifdef __cplusplus
}
#endif

#endif // LFM_CONFORMER_H
