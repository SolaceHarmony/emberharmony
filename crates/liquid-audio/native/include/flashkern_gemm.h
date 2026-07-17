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

#ifdef __cplusplus
}
#endif

#endif
