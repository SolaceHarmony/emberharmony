// NEON BFMMLA bf16 GEMM micro-kernel — closes candle 0.9.2's CPU bf16-matmul gap
// (its gemm allowlist is F16/F32/F64 only) with the Arm BFloat16 matrix-multiply
// extension (FEAT_BF16). bf16 inputs, **f32 accumulate** — the same numerics torch's
// CPU bf16 matmul uses. Caller MUST verify FEAT_BF16 at runtime (sysctl
// hw.optional.arm.FEAT_BF16) before calling; BFMMLA SIGILLs without it.
//
// Compiled by build.rs (`cc`) with -march=armv8.2-a+bf16. bf16 values cross the FFI
// boundary as raw uint16_t (bit-identical to Rust `half::bf16` and C `bfloat16_t`).
//
// `vbfmmlaq_f32(acc, a, b)` treats a,b as 2x4 bf16 matrices and computes a · bᵀ
// (2x4 · 4x2 → 2x2), accumulating into the 2x2 f32 `acc` laid out [c00,c01,c10,c11].
// So packing b's lane-row r = column (jt+r) of B over a 4-deep K block makes
// (a · bᵀ)[i][j] = Σ_k A[it+i][k]·B[k][jt+j] — an ordinary C = A·B.

#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

// C (M×N, f32, row-major) = A (M×K, bf16) · B (K×N, bf16), all row-major.
void lfm_bf16_gemm_f32(const uint16_t *A_, const uint16_t *B_, float *C, int M, int N, int K) {
    const bfloat16_t *A = (const bfloat16_t *)A_;
    const bfloat16_t *B = (const bfloat16_t *)B_;

    const int Mp = (M + 1) & ~1; // round M up to a multiple of 2 (BFMMLA row pair)
    const int Np = (N + 1) & ~1; // round N up to a multiple of 2 (BFMMLA col pair)
    const int Kp = (K + 3) & ~3; // round K up to a multiple of 4 (BFMMLA K depth)
    const int kb = Kp / 4;       // number of 4-deep K blocks

    // Packed, zero-padded buffers in BFMMLA tile order (calloc → bf16 +0.0 padding,
    // which contributes nothing to the dot products).
    bfloat16_t *Ap = (bfloat16_t *)calloc((size_t)Mp * Kp, sizeof(bfloat16_t));
    bfloat16_t *Bp = (bfloat16_t *)calloc((size_t)Np * Kp, sizeof(bfloat16_t));
    if (!Ap || !Bp) { free(Ap); free(Bp); return; }

    // Ap: per row-pair (it/2), per K block, 8 bf16 = [A[it][k..k+3], A[it+1][k..k+3]].
    for (int i = 0; i < M; i++) {
        const int it = i & ~1, ir = i & 1;
        for (int k = 0; k < K; k++) {
            size_t blk = ((size_t)(it / 2) * kb + (k / 4)) * 8;
            Ap[blk + ir * 4 + (k & 3)] = A[(size_t)i * K + k];
        }
    }
    // Bp: per col-pair (jt/2), per K block, 8 bf16 = [B[k..k+3][jt], B[k..k+3][jt+1]].
    for (int j = 0; j < N; j++) {
        const int jt = j & ~1, jc = j & 1;
        for (int k = 0; k < K; k++) {
            size_t blk = ((size_t)(jt / 2) * kb + (k / 4)) * 8;
            Bp[blk + jc * 4 + (k & 3)] = B[(size_t)k * N + j];
        }
    }

    for (int it = 0; it < Mp; it += 2) {
        const bfloat16_t *ap0 = Ap + (size_t)(it / 2) * kb * 8;
        for (int jt = 0; jt < Np; jt += 2) {
            const bfloat16_t *ap = ap0;
            const bfloat16_t *bp = Bp + (size_t)(jt / 2) * kb * 8;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int b = 0; b < kb; b++) {
                bfloat16x8_t av = vld1q_bf16(ap);
                bfloat16x8_t bv = vld1q_bf16(bp);
                acc = vbfmmlaq_f32(acc, av, bv); // acc += [c00,c01,c10,c11]
                ap += 8;
                bp += 8;
            }
            float out[4];
            vst1q_f32(out, acc);
            if (it + 0 < M && jt + 0 < N) C[(size_t)(it + 0) * N + (jt + 0)] = out[0];
            if (it + 0 < M && jt + 1 < N) C[(size_t)(it + 0) * N + (jt + 1)] = out[1];
            if (it + 1 < M && jt + 0 < N) C[(size_t)(it + 1) * N + (jt + 0)] = out[2];
            if (it + 1 < M && jt + 1 < N) C[(size_t)(it + 1) * N + (jt + 1)] = out[3];
        }
    }
    free(Ap);
    free(Bp);
}
