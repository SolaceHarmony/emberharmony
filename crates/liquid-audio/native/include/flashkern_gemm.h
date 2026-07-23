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

// Readiness predicate for the one architecture leaf selected into this build.
// The engine checks this once before publishing readiness. Numerical callers
// never branch to another implementation.
int lfm_bf16_gemm_available(void);

// Architecture leaves. A ready engine has already established their ISA
// contract.
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

#ifdef __cplusplus
}
#endif

#endif
