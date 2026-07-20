#include "lfm_asm_visibility.h"

// Selected-bin Web-Audio DFT and magnitude smoothing. The immutable real/imag
// tables already contain Blackman(window, denominator N) and 1/N, so execution
// reads the caller's 256-sample view once per selected bin and writes only the
// compact retained magnitude state. There is no full spectrum plane.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 2

// void lfm_sesame_selected_magnitudes(const float *pcm,
//     const float *real_table, const float *imag_table,
//     float *smoothed_magnitudes, size_t selected_bins)
.globl LFM_SYM(lfm_sesame_selected_magnitudes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_selected_magnitudes))
LFM_SYM(lfm_sesame_selected_magnitudes):
    cbz     x4, 9f
    mov     w8, #0xcccd
    movk    w8, #0x3e4c, lsl #16 // 0.2f
    fmov    s6, w8
    mov     w8, #0xcccd
    movk    w8, #0x3f4c, lsl #16 // 0.8f
    fmov    s7, w8
    mov     w8, #0x7f80
    lsl     w8, w8, #16
1:
    mov     x5, x0
    mov     x6, #64
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
2:
    ldr     q2, [x5], #16
    ldr     q3, [x1], #16
    ldr     q4, [x2], #16
    fmul    v3.4s, v2.4s, v3.4s
    fmul    v4.4s, v2.4s, v4.4s
    fadd    v0.4s, v0.4s, v3.4s
    fadd    v1.4s, v1.4s, v4.4s
    subs    x6, x6, #1
    b.ne    2b

    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v1.4s, v1.4s, v1.4s
    faddp   v1.4s, v1.4s, v1.4s
    fmul    s0, s0, s0
    fmul    s1, s1, s1
    fadd    s0, s0, s1
    fsqrt   s0, s0

    // NaN/Inf magnitude is no evidence. Clear before it can poison retained
    // smoothing state; finite subnormals remain valid Web-Audio magnitudes.
    fmov    w9, s0
    and     w9, w9, #0x7f800000
    cmp     w9, w8
    b.ne    3f
    fmov    s0, wzr
3:
    ldr     s1, [x3]
    fmul    s1, s1, s7
    fmul    s0, s0, s6
    fadd    s0, s1, s0
    str     s0, [x3], #4
    subs    x4, x4, #1
    b.ne    1b
9:
    ret

// void lfm_sesame_selected_magnitudes_window(
//     const LfmSesameWindowV1 *window, const float *real_table,
//     const float *imag_table, float *smoothed_magnitudes,
//     size_t selected_bins)
//
// The circular wrap can split one four-sample group. Reconstruct only that
// group in a vector register, then use the identical mul/add/reduction order as
// the contiguous leaf above. No 256-sample scratch plane is materialized.
.p2align 2
.globl LFM_SYM(lfm_sesame_selected_magnitudes_window)
LFM_PRIVATE(LFM_SYM(lfm_sesame_selected_magnitudes_window))
LFM_SYM(lfm_sesame_selected_magnitudes_window):
    cbz     x4, 19f
    mov     x13, x0
    mov     w8, #0xcccd
    movk    w8, #0x3e4c, lsl #16 // 0.2f
    fmov    s6, w8
    mov     w8, #0xcccd
    movk    w8, #0x3f4c, lsl #16 // 0.8f
    fmov    s7, w8
    mov     w8, #0x7f80
    lsl     w8, w8, #16
11:
    ldr     x5, [x13]
    ldr     x7, [x13, #8]
    ldr     x6, [x13, #16]
    mov     x9, #64
    eor     v0.16b, v0.16b, v0.16b
    eor     v1.16b, v1.16b, v1.16b
12:
    cmp     x7, #4
    b.hs    16f
    cbz     x7, 17f
    eor     v2.16b, v2.16b, v2.16b
    cmp     x7, #2
    b.eq    14f
    b.hi    15f
    // 1 + 3 samples.
    ld1     {v2.s}[0], [x5], #4
    ld1     {v2.s}[1], [x6], #4
    ld1     {v2.s}[2], [x6], #4
    ld1     {v2.s}[3], [x6], #4
    mov     x7, #0
    b       18f
14:
    // 2 + 2 samples.
    ld1     {v2.s}[0], [x5], #4
    ld1     {v2.s}[1], [x5], #4
    ld1     {v2.s}[2], [x6], #4
    ld1     {v2.s}[3], [x6], #4
    mov     x7, #0
    b       18f
15:
    // 3 + 1 samples.
    ld1     {v2.s}[0], [x5], #4
    ld1     {v2.s}[1], [x5], #4
    ld1     {v2.s}[2], [x5], #4
    ld1     {v2.s}[3], [x6], #4
    mov     x7, #0
    b       18f
16:
    ldr     q2, [x5], #16
    sub     x7, x7, #4
    b       18f
17:
    ldr     q2, [x6], #16
18:
    ldr     q3, [x1], #16
    ldr     q4, [x2], #16
    fmul    v3.4s, v2.4s, v3.4s
    fmul    v4.4s, v2.4s, v4.4s
    fadd    v0.4s, v0.4s, v3.4s
    fadd    v1.4s, v1.4s, v4.4s
    subs    x9, x9, #1
    b.ne    12b

    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v1.4s, v1.4s, v1.4s
    faddp   v1.4s, v1.4s, v1.4s
    fmul    s0, s0, s0
    fmul    s1, s1, s1
    fadd    s0, s0, s1
    fsqrt   s0, s0
    fmov    w9, s0
    and     w9, w9, #0x7f800000
    cmp     w9, w8
    b.ne    13f
    fmov    s0, wzr
13:
    ldr     s1, [x3]
    fmul    s1, s1, s7
    fmul    s0, s0, s6
    fadd    s0, s1, s0
    str     s0, [x3], #4
    subs    x4, x4, #1
    b.ne    11b
19:
    ret

// void lfm_sesame_magnitudes_to_bytes(const float *magnitudes,
//     const double thresholds[256], uint8_t *bytes, size_t count)
// thresholds[q] is the exact f64 magnitude where floor-affine Web-Audio byte
// q begins. Eight comparisons find the greatest reached threshold.
.p2align 2
.globl LFM_SYM(lfm_sesame_magnitudes_to_bytes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_magnitudes_to_bytes))
LFM_SYM(lfm_sesame_magnitudes_to_bytes):
    cbz     x3, 9f
1:
    ldr     s0, [x0], #4
    fcvt    d0, s0
    mov     w4, #0
    mov     w5, #256
.macro LFM_SESAME_BYTE_STEP
    add     w6, w4, w5
    lsr     w6, w6, #1
    ldr     d1, [x1, x6, lsl #3]
    fcmp    d0, d1
    csel    w4, w6, w4, ge
    csel    w5, w5, w6, ge
.endm
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    strb    w4, [x2], #1
    subs    x3, x3, #1
    b.ne    1b
9:
    ret

// uint32_t lfm_sesame_classify_selected_bytes(const uint8_t *bytes,
//     size_t count, uint8_t *minimum, uint8_t *maximum, uint32_t threshold,
//     double *score)
// Sticky extrema, byte sum/mean, normalized score, and equality-inclusive
// threshold comparison stay in the numerical leaf. C++ only publishes control
// metadata around the returned result.
.p2align 2
.globl LFM_SYM(lfm_sesame_classify_selected_bytes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_classify_selected_bytes))
LFM_SYM(lfm_sesame_classify_selected_bytes):
    ldrb    w6, [x2]
    ldrb    w7, [x3]
    mov     x8, #0
    mov     x9, x1
1:
    ldrb    w10, [x0], #1
    add     x8, x8, x10
    cmp     w10, w6
    csel    w6, w10, w6, lo
    cmp     w10, w7
    csel    w7, w10, w7, hi
    subs    x9, x9, #1
    b.ne    1b
    strb    w6, [x2]
    strb    w7, [x3]

    subs    w10, w7, w6
    b.eq    2f
    ucvtf   d0, x8
    ucvtf   d1, x1
    fdiv    d0, d0, d1
    ucvtf   d1, w6
    fsub    d0, d0, d1
    ucvtf   d1, w10
    fdiv    d0, d0, d1
    mov     w10, #100
    ucvtf   d1, w10
    fmul    d0, d0, d1
    b       3f
2:
    fmov    d0, xzr
3:
    str     d0, [x5]
    ucvtf   d1, w4
    fcmp    d0, d1
    cset    w0, ge
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
