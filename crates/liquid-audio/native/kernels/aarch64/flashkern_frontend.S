#include "lfm_asm_visibility.h"

// Audio-frontend leaves (mel featurizer + resampler hot loops). Scalar on
// purpose: these run once per utterance, not per token, and the parity oracle
// (native/tests/fixtures/mel) was captured from strictly-ordered scalar-
// equivalent tensor ops. Separate fmul/fadd everywhere — no FMA contraction:
// the deleted Rust reference never fused, and -ffp-contract=off guards the C++
// side of the same contract.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text
.p2align 2

// int lfm_preemph_f32(const float *x, float *y, uint64_t n, float coef /*s0*/)
// y[0] = x[0]; y[i] = x[i] - coef*x[i-1].
.globl LFM_SYM(lfm_preemph_f32)
LFM_PRIVATE(LFM_SYM(lfm_preemph_f32))
LFM_SYM(lfm_preemph_f32):
    cbz     x0, 9f
    cbz     x1, 9f
    cbz     x2, 9f
    ldr     s3, [x0]             // previous original sample
    str     s3, [x1]
    mov     x3, #1
1:
    cmp     x3, x2
    b.hs    2f
    ldr     s1, [x0, x3, lsl #2] // current original; load before x==y store
    fmul    s2, s3, s0
    fsub    s2, s1, s2
    str     s2, [x1, x3, lsl #2]
    fmov    s3, s1
    add     x3, x3, #1
    b       1b
2:
    mov     w0, #0
    ret
9:
    mov     w0, #-22
    ret

// void lfm_power_spec_f32(const float *re, const float *im, float *out, uint64_t n)
// out[i] = re[i]*re[i] + im[i]*im[i].
.p2align 2
.globl LFM_SYM(lfm_power_spec_f32)
LFM_PRIVATE(LFM_SYM(lfm_power_spec_f32))
LFM_SYM(lfm_power_spec_f32):
    cbz     x3, 8f
1:
    ldr     s0, [x0], #4
    ldr     s1, [x1], #4
    fmul    s0, s0, s0
    fmul    s1, s1, s1
    fadd    s0, s0, s1
    str     s0, [x2], #4
    subs    x3, x3, #1
    b.ne    1b
8:
    ret

// int lfm_log_add_f32(float *x, uint64_t n, float guard /*s0*/)
// x[i] = logf(x[i] + guard), in place. Calls libm logf (rope-leaf pattern).
.p2align 2
.globl LFM_SYM(lfm_log_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_log_add_f32))
LFM_SYM(lfm_log_add_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_log_add_f32)
#endif
    stp     x29, x30, [sp, #-48]!
#ifdef _WIN32
    .seh_save_fplr_x 48
#endif
    mov     x29, sp
#ifdef _WIN32
    .seh_set_fp
#endif
    stp     x19, x20, [sp, #16]
#ifdef _WIN32
    .seh_save_regp x19, 16
#endif
    str     d8, [sp, #32]
#ifdef _WIN32
    .seh_save_freg d8, 32
    .seh_endprologue
#endif
    cbz     x0, 9f
    mov     x19, x0
    mov     x20, x1
    fmov    s8, s0
1:
    cbz     x20, 2f
    ldr     s0, [x19]
    fadd    s0, s0, s8
    bl      LFM_CALL(logf)
    str     s0, [x19], #4
    sub     x20, x20, #1
    b       1b
2:
    mov     w0, #0
    b       7f
9:
    mov     w0, #-22
7:
#ifdef _WIN32
    .seh_startepilogue
#endif
    ldr     d8, [sp, #32]
#ifdef _WIN32
    .seh_save_freg d8, 32
#endif
    ldp     x19, x20, [sp, #16]
#ifdef _WIN32
    .seh_save_regp x19, 16
#endif
    ldp     x29, x30, [sp], #48
#ifdef _WIN32
    .seh_save_fplr_x 48
    .seh_endepilogue
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// int lfm_rowstat_f32(const float *x, uint64_t rows, uint64_t cols,
//                     uint64_t valid, float constant /*s0*/,
//                     float *mean_out, float *std_out)
// Per row over columns [0, valid): mean; ddof-1 std when valid > 1 else the
// reference's NaN-masked 0; std += constant. Sequential adds — the oracle's
// reduction order.
.p2align 2
.globl LFM_SYM(lfm_rowstat_f32)
LFM_PRIVATE(LFM_SYM(lfm_rowstat_f32))
LFM_SYM(lfm_rowstat_f32):
    cbz     x0, 9f
    cbz     x1, 9f
    cbz     x3, 9f
    cmp     x3, x2
    b.hi    9f
    mov     x6, #0              // row
    mov     x7, x0              // row base
1:
    cmp     x6, x1
    b.hs    8f
    fmov    s1, wzr             // sum
    mov     x8, #0
2:
    ldr     s2, [x7, x8, lsl #2]
    fadd    s1, s1, s2
    add     x8, x8, #1
    cmp     x8, x3
    b.lo    2b
    ucvtf   s3, x3
    fdiv    s1, s1, s3          // mean
    str     s1, [x4, x6, lsl #2]
    fmov    s2, wzr             // centered sumsq
    cmp     x3, #1
    b.ls    4f
    mov     x8, #0
3:
    ldr     s5, [x7, x8, lsl #2]
    fsub    s5, s5, s1
    fmul    s5, s5, s5
    fadd    s2, s2, s5
    add     x8, x8, #1
    cmp     x8, x3
    b.lo    3b
    sub     x9, x3, #1
    ucvtf   s6, x9
    fdiv    s2, s2, s6
    fsqrt   s2, s2
4:
    fadd    s2, s2, s0          // += CONSTANT (1e-5)
    str     s2, [x5, x6, lsl #2]
    add     x7, x7, x2, lsl #2
    add     x6, x6, #1
    b       1b
8:
    mov     w0, #0
    ret
9:
    mov     w0, #-22
    ret

// void lfm_norm_apply_f32(float *x, const float *mean, const float *std,
//                         uint64_t rows, uint64_t cols)
// x[r][c] = (x[r][c] - mean[r]) / std[r], over ALL columns (broadcast).
.p2align 2
.globl LFM_SYM(lfm_norm_apply_f32)
LFM_PRIVATE(LFM_SYM(lfm_norm_apply_f32))
LFM_SYM(lfm_norm_apply_f32):
    mov     x5, #0
1:
    cmp     x5, x3
    b.hs    8f
    ldr     s0, [x1, x5, lsl #2]
    ldr     s1, [x2, x5, lsl #2]
    mov     x6, #0
2:
    cmp     x6, x4
    b.hs    3f
    ldr     s2, [x0]
    fsub    s2, s2, s0
    fdiv    s2, s2, s1
    str     s2, [x0], #4
    add     x6, x6, #1
    b       2b
3:
    add     x5, x5, #1
    b       1b
8:
    ret

// void lfm_resample_conv_f64(const double *padded, uint64_t padded_len,
//                            const double *kernels, uint64_t phases,
//                            uint64_t klen, uint64_t stride,
//                            float *out, uint64_t max_values)
// The torchaudio strided conv: for each block start (step `stride`) while
// start+klen <= padded_len, one f64 dot per phase kernel, narrowed to f32.
// max_values permits the final partial phase block to land directly in the
// caller's exact-size output instead of a copied convolution staging plane.
.p2align 2
.globl LFM_SYM(lfm_resample_conv_f64)
LFM_PRIVATE(LFM_SYM(lfm_resample_conv_f64))
LFM_SYM(lfm_resample_conv_f64):
    mov     x9, #0              // start
    mov     x10, #0             // block
    mov     x16, #0             // values written
1:
    add     x11, x9, x4
    cmp     x11, x1
    b.hi    8f
    mov     x12, #0             // phase
2:
    cmp     x12, x3
    b.hs    5f
    cmp     x16, x7
    b.hs    8f
    mul     x13, x12, x4
    add     x13, x2, x13, lsl #3
    add     x14, x0, x9, lsl #3
    fmov    d0, xzr
    mov     x15, #0
3:
    cmp     x15, x4
    b.hs    4f
    ldr     d1, [x14, x15, lsl #3]
    ldr     d2, [x13, x15, lsl #3]
    fmul    d1, d1, d2
    fadd    d0, d0, d1
    add     x15, x15, #1
    b       3b
4:
    fcvt    s0, d0
    str     s0, [x6], #4
    add     x16, x16, #1
    add     x12, x12, #1
    b       2b
5:
    add     x9, x9, x5
    add     x10, x10, #1
    b       1b
8:
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
