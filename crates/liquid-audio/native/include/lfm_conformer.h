// Native production Conformer encoder + audio adapter.
//
// Execution: one segment is one retained Flashkern ticket. Every fixed-team
// final return advances a ticket-owned program cursor and eagerly dispatches
// its next stage; no lane or caller stack waits for an internal Conformer
// result. C++ sequences stages and moves bytes; every value is produced by an
// architecture assembly leaf
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

#include "lfm_visibility.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct LfmConformer LfmConformer;
typedef struct LfmConformerWorkspace LfmConformerWorkspace;

typedef struct LfmConformerGeometry {
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
} LfmConformerGeometry;

// Binds every encoder/adapter weight as byte views into the resident
// safetensors image (the same image the model owns; no duplicate bytes, no
// name lookups after this call). A view may be unaligned; kernels load it by
// bytes rather than exposing a dereferenceable aligned weight pointer.
// `weights` is the native model's private LfmWeightImage handle. `engine` is
// the runtime-owned Flashkern engine created internally by
// lfm_engine_new_status through lfm_runtime_create; the engine constructor and
// weight image are private native owners. Returns 0; -EINVAL on nulls/bad
// geometry; -ENOENT with `error` filled when a required weight field is
// missing or mis-shaped.
LFM_INTERNAL_API int lfm_conformer_create(
    void *engine, const void *weights, const LfmConformerGeometry *geometry,
    LfmConformer **out, char *error, size_t error_length);
LFM_INTERNAL_API int lfm_conformer_destroy(LfmConformer *conformer);

// Immutable residency accounting. `derived_bytes` is limited to formula-
// derived tables (BN denominators and relative-position frequencies). Bound
// checkpoint bytes remain views into the owner image. Materialized bytes must
// remain zero for every forward. `direct_gemm_calls` counts logical linear
// operations (fixed-team execution may issue multiple accumulator tiles for
// one operation); it is an execution witness for steady-state tests, not a
// physical-dispatch counter.
LFM_INTERNAL_API uint64_t
lfm_conformer_bound_weight_bytes(const LfmConformer *conformer);
LFM_INTERNAL_API uint64_t
lfm_conformer_derived_bytes(const LfmConformer *conformer);
LFM_INTERNAL_API uint64_t
lfm_conformer_materialized_weight_bytes(const LfmConformer *conformer);
LFM_INTERNAL_API uint64_t
lfm_conformer_direct_gemm_calls(const LfmConformer *conformer);

// Session-owned mutable planes. Production reserves the maximum admitted mel
// segment before readiness; ticket execution rejects an unsealed workspace,
// never grows, and returns -ENOBUFS if admission is violated.
LFM_INTERNAL_API int
lfm_conformer_workspace_create(LfmConformerWorkspace **out);
LFM_INTERNAL_API int
lfm_conformer_workspace_destroy(LfmConformerWorkspace *workspace);
LFM_INTERNAL_API int lfm_conformer_workspace_reserve(
    const LfmConformer *conformer, LfmConformerWorkspace *workspace,
    uint64_t max_mel_frames);

// Output rows for a mel segment of `mel_frames`: the pinned dw_striding length
// chain (three k3/s2/p1 stages), checked against the offline reference fixtures.
LFM_INTERNAL_API uint64_t
lfm_conformer_out_rows(const LfmConformer *conformer, uint64_t mel_frames);
LFM_INTERNAL_API uint64_t
lfm_conformer_out_width(const LfmConformer *conformer);

#ifdef __cplusplus
}
#endif

#endif // LFM_CONFORMER_H
