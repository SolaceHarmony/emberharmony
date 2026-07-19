#ifndef FLASHKERN_GEMM_H
#define FLASHKERN_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    LFM_GEMM_RHS_KN = 0,
    LFM_GEMM_RHS_NK = 1,
};

// Runtime ISA gate for the architecture leaf selected into this build.
int lfm_bf16_gemm_available(void);

// Architecture leaves. Callers must pass the runtime gate above first.
void lfm_bf16_gemm_f32_v2(const uint16_t *a, const uint16_t *b, float *c,
                          int m, int n, int k);
void lfm_bf16_gemv_f32(const uint16_t *a, const uint16_t *b, float *c,
                       int n, int k);
void lfm_bf16_gemm_nt_f32(const uint16_t *a, const void *weight_bytes, float *c,
                          int m, int n, int k);
// Small-M checkpoint-layout leaf with an explicit destination row stride. This
// lets fixed lanes own disjoint output-column bands while every resident weight
// row is still streamed once across all M activation rows.
void lfm_bf16_gemm_nt_strided_f32(const uint16_t *a,
                                  const void *weight_bytes, float *c,
                                  int m, int n, int k, int output_stride);
// Baseline architecture leaf for hosts where the tuned SIMD leaf is unavailable
// (notably Rosetta's deliberately failed AVX state gate). Both operands stay in
// checkpoint-native bf16 storage; conversion happens only in scalar registers.
void lfm_bf16_gemm_nt_f32_scalar(const uint16_t *a, const void *weight_bytes, float *c,
                                 int m, int n, int k);
void lfm_bf16_gemm_nt_strided_f32_scalar(const uint16_t *a,
                                         const void *weight_bytes, float *c,
                                         int m, int n, int k,
                                         int output_stride);

// Conformer linear epilogue. Checkpoint-native W[N,K] and optional BF16 bias
// are consumed as byte views. Each F32 dot receives its bias before the single
// logical BF16 storage round and is written directly with `output_stride`.
void lfm_bf16_gemm_nt_bias_bf16(const uint16_t *activation,
                                const void *weight_bytes,
                                const void *bias_bytes,
                                uint16_t *output, int rows,
                                int columns, int inner,
                                int output_stride);

// Decode projection superkernel:
//
//   dot = bf16 input[K] * checkpoint-native bf16 weights[N,K]^T
//   projected = RNE-bf16(dot)
//   output = RNE-bf16(f32(projected) + f32(residual[N]))
//
// The two logical bf16 boundaries are part of the numerical contract even
// though both are carried in registers. Checkpoint and residual views may be
// byte-unaligned. Only the terminal output plane is written; no f32 or rounded
// projection plane is materialized.
void lfm_bf16_gemv_rne_add_bf16(const void *input,
                                const void *weight_bytes,
                                const void *residual,
                                uint16_t *output,
                                size_t rows, size_t depth);

// The same register-FIFO dot with one logical BF16 storage boundary and no
// residual epilogue. It writes only the terminal BF16 destination.
void lfm_bf16_gemv_rne_bf16(const void *input,
                            const void *weight_bytes,
                            uint16_t *output,
                            size_t rows, size_t depth);

// Paired gate/up projection. Each activation block is loaded once and consumed
// by checkpoint-native W1 and W3 rows; both dots and the complete logical BF16
// SwiGLU ladder remain local until the terminal gate value is stored.
void lfm_bf16_gemv_pair_swiglu_bf16(const void *input,
                                    const void *gate_weight_bytes,
                                    const void *up_weight_bytes,
                                    uint16_t *output,
                                    size_t rows, size_t depth);

// One fixed-team SQ/CQ pass. Payloads remain borrowed until exact completion.
int lfm_engine_bf16_gemm_f32(void *engine,
                             const uint16_t *a, size_t a_count,
                             const uint16_t *rhs, size_t rhs_count,
                             float *out, size_t out_count,
                             size_t m, size_t n, size_t k,
                             uint32_t rhs_layout);

// Checkpoint-native [N,K] ticket. The fixed lane team streams bf16 A and W
// directly and widens values only in registers; no packed or f32 weight image
// exists on any architecture.
int lfm_engine_bf16_gemm_nt_direct_f32(void *engine,
                                       const uint16_t *a, size_t a_count,
                                       const void *weight_bytes,
                                       size_t weight_count,
                                       float *out, size_t out_count,
                                       size_t m, size_t n, size_t k);

// Private native direct-destination linear ticket. Bias may be null only when
// bias_count is zero. No F32 output plane crosses the ticket boundary.
int lfm_engine_bf16_gemm_nt_direct_bf16(
    void *engine, const uint16_t *activation, size_t activation_count,
    const void *weight_bytes, size_t weight_count,
    const void *bias_bytes, size_t bias_count, uint16_t *output,
    size_t output_count, size_t rows, size_t columns, size_t inner);

#ifdef __cplusplus
}
#endif

#endif
