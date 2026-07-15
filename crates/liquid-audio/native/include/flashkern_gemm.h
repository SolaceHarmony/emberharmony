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
void lfm_bf16_gemm_nt_f32(const uint16_t *a, const uint16_t *w, float *c,
                          int m, int n, int k);

// One fixed-team SQ/CQ pass. Payloads remain borrowed until exact completion.
int lfm_engine_bf16_gemm_f32(void *engine,
                             const uint16_t *a, size_t a_count,
                             const uint16_t *rhs, size_t rhs_count,
                             float *out, size_t out_count,
                             size_t m, size_t n, size_t k,
                             uint32_t rhs_layout);

#ifdef __cplusplus
}
#endif

#endif
