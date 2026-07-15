#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 2
.globl LFM_SYM(lfm_rsqrt_size)
LFM_SYM(lfm_rsqrt_size):
    ucvtf   s1, x0
    fsqrt   s1, s1
    fmov    s0, #1.0
    fdiv    s0, s0, s1
    ret

.p2align 2
.globl LFM_SYM(lfm_inv_rms_f32)
LFM_SYM(lfm_inv_rms_f32):
    ucvtf   s2, x0
    fdiv    s0, s0, s2
    fadd    s0, s0, s1
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    ret

.p2align 2
.globl LFM_SYM(lfm_sum_f32)
LFM_SYM(lfm_sum_f32):
    fmov    s0, wzr
    cbz     x1, 2f
1:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    subs    x1, x1, #1
    b.ne    1b
2:
    ret

.p2align 2
.globl LFM_SYM(lfm_bf16_sumsq_stride_f32)
LFM_SYM(lfm_bf16_sumsq_stride_f32):
    fmov    s0, wzr
    cmp     x2, x1
    b.hs    2f
1:
    ldrh    w4, [x0, x2, lsl #1]
    lsl     w4, w4, #16
    fmov    s1, w4
    fmul    s1, s1, s1
    fadd    s0, s0, s1
    add     x2, x2, x3
    cmp     x2, x1
    b.lo    1b
2:
    ret

.p2align 2
.globl LFM_SYM(lfm_bf16_bias_add_f32)
LFM_SYM(lfm_bf16_bias_add_f32):
    cbz     x2, 2f
1:
    ldr     s0, [x0]
    ldrh    w3, [x1], #2
    lsl     w3, w3, #16
    fmov    s1, w3
    fadd    s0, s0, s1
    str     s0, [x0], #4
    subs    x2, x2, #1
    b.ne    1b
2:
    ret

.macro LFM_RNE_WIDEN source, target, bits, tie
    fmov    \bits, \source
    lsr     \tie, \bits, #16
    and     \tie, \tie, #1
    add     \bits, \bits, w14
    add     \bits, \bits, \tie
    lsr     \bits, \bits, #16
    lsl     \bits, \bits, #16
    fmov    \target, \bits
.endm

.p2align 2
.globl LFM_SYM(lfm_bf16_rope_neox)
LFM_SYM(lfm_bf16_rope_neox):
    cbz     x0, 3f
    cbz     x1, 3f
    cbz     x2, 3f
    cbz     x3, 3f
    tbnz    x3, #0, 3f
    lsr     x4, x3, #1
    add     x5, x0, x4, lsl #1
    mov     w14, #0x7fff
1:
    ldrh    w8, [x0]
    ldrh    w9, [x5]
    ldrh    w10, [x1], #2
    ldrh    w11, [x2], #2
    lsl     w8, w8, #16
    lsl     w9, w9, #16
    lsl     w10, w10, #16
    lsl     w11, w11, #16
    fmov    s0, w8
    fmov    s1, w9
    fmov    s2, w10
    fmov    s3, w11

    fmul    s4, s0, s2
    LFM_RNE_WIDEN s4, s4, w12, w13
    fneg    s5, s1
    fmul    s5, s5, s3
    LFM_RNE_WIDEN s5, s5, w12, w13
    fadd    s4, s4, s5
    LFM_RNE_WIDEN s4, s4, w12, w13
    lsr     w12, w12, #16
    strh    w12, [x0], #2

    fmul    s4, s1, s2
    LFM_RNE_WIDEN s4, s4, w12, w13
    fmul    s5, s0, s3
    LFM_RNE_WIDEN s5, s5, w12, w13
    fadd    s4, s4, s5
    LFM_RNE_WIDEN s4, s4, w12, w13
    lsr     w12, w12, #16
    strh    w12, [x5], #2

    subs    x4, x4, #1
    b.ne    1b
3:
    ret

#if !defined(__APPLE__)
.section .note.GNU-stack,"",%progbits
#endif
