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

// void lfm_resample_conv_f32(const float *input, uint64_t input_len,
//                            const double *kernels, uint64_t phases,
//                            uint64_t stride, uint64_t width,
//                            uint64_t first_value, float *out,
//                            uint64_t value_count)
// Torchaudio's strided f64 convolution over a direct f32 input view. Prefix and
// suffix padding are logical zeros: only the in-span taps are loaded, widened,
// and accumulated in the same ascending-j order as the materialized reference.
// first_value/value_count make disjoint output ranges independently runnable.
.p2align 2
.globl LFM_SYM(lfm_resample_conv_f32)
LFM_PRIVATE(LFM_SYM(lfm_resample_conv_f32))
LFM_SYM(lfm_resample_conv_f32):
    ldr     x8, [sp]            // value_count (ninth argument)
    add     x9, x4, x5, lsl #1 // klen = stride + 2*width
    udiv    x10, x6, x3         // initial block
    msub    x11, x10, x3, x6   // initial phase
1:
    cbz     x8, 8f
    mul     x12, x10, x4        // logical padded-window start
    mul     x13, x11, x9
    add     x13, x2, x13, lsl #3 // kernel row

    cmp     x12, x5
    b.hs    2f
    sub     x14, x5, x12        // j_begin = width - start
    mov     x15, #0             // input index 0
    b       3f
2:
    mov     x14, #0             // j_begin = 0
    sub     x15, x12, x5        // input index = start - width
3:
    add     x16, x5, x1         // one-past logical input coordinate
    cmp     x12, x16
    b.hs    6f                   // the entire window is suffix padding
    sub     x16, x16, x12       // first suffix j
    cmp     x16, x9
    csel    x16, x16, x9, lo    // j_end = min(klen, width+len-start)
    fmov    d0, xzr
4:
    cmp     x14, x16
    b.hs    7f
    ldr     s1, [x0, x15, lsl #2]
    fcvt    d1, s1
    ldr     d2, [x13, x14, lsl #3]
    fmul    d1, d1, d2
    fadd    d0, d0, d1
    add     x14, x14, #1
    add     x15, x15, #1
    b       4b
6:
    fmov    d0, xzr
7:
    fcvt    s0, d0
    str     s0, [x7], #4
    sub     x8, x8, #1
    add     x11, x11, #1
    cmp     x11, x3
    b.lo    1b
    mov     x11, #0
    add     x10, x10, #1
    b       1b
8:
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
