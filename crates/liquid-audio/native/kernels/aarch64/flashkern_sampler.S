#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text

/* uint32_t lfm_sampler_argmax_f32(const float *x, size_t count) */
.p2align 2
.globl LFM_SYM(lfm_sampler_argmax_f32)
LFM_SYM(lfm_sampler_argmax_f32):
    cbnz    x1, 1f
    mov     w0, wzr
    ret
1:
    mov     w8, #0xff800000
    dup     v0.4s, w8
    mov     x2, xzr
2:
    sub     x3, x1, x2
    cmp     x3, #4
    b.lo    3f
    add     x4, x0, x2, lsl #2
    ldr     q1, [x4]
    fmaxnm  v0.4s, v0.4s, v1.4s
    add     x2, x2, #4
    b       2b
3:
    fmaxnmv s0, v0.4s
    cmp     x2, x1
    b.hs    5f
4:
    ldr     s1, [x0, x2, lsl #2]
    fcmp    s1, s0
    b.le    6f
    fmov    s0, s1
6:
    add     x2, x2, #1
    cmp     x2, x1
    b.lo    4b
5:
    mov     x2, xzr
7:
    ldr     s1, [x0, x2, lsl #2]
    fcmp    s1, s0
    b.eq    8f
    add     x2, x2, #1
    cmp     x2, x1
    b.lo    7b
    mov     w0, wzr
    ret
8:
    mov     w0, w2
    ret

/* uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count) */
.p2align 2
.globl LFM_SYM(lfm_sampler_argmax_bf16)
LFM_SYM(lfm_sampler_argmax_bf16):
    cbnz    x1, 1f
    mov     w0, wzr
    ret
1:
    mov     w8, #0xff800000
    dup     v0.4s, w8
    mov     x2, xzr
2:
    sub     x3, x1, x2
    cmp     x3, #4
    b.lo    3f
    add     x4, x0, x2, lsl #1
    ldr     d1, [x4]
    shll    v1.4s, v1.4h, #16
    fmaxnm  v0.4s, v0.4s, v1.4s
    add     x2, x2, #4
    b       2b
3:
    fmaxnmv s0, v0.4s
    cmp     x2, x1
    b.hs    5f
4:
    ldrh    w8, [x0, x2, lsl #1]
    lsl     w8, w8, #16
    fmov    s1, w8
    fcmp    s1, s0
    b.le    6f
    fmov    s0, s1
6:
    add     x2, x2, #1
    cmp     x2, x1
    b.lo    4b
5:
    mov     x2, xzr
7:
    ldrh    w8, [x0, x2, lsl #1]
    lsl     w8, w8, #16
    fmov    s1, w8
    fcmp    s1, s0
    b.eq    8f
    add     x2, x2, #1
    cmp     x2, x1
    b.lo    7b
    mov     w0, wzr
    ret
8:
    mov     w0, w2
    ret

/*
 * float lfm_sampler_exp_sum_f32(const float *x, float *weights,
 *                               size_t count, float scale,
 *                               float maximum, float threshold)
 *
 * The local exp leaf uses f32 range reduction and a degree-six polynomial.
 * Traversal, scaling, thresholding, exponentiation, writes, and reduction all
 * remain in this assembly object.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_exp_sum_f32)
LFM_SYM(lfm_sampler_exp_sum_f32):
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]
    stp     d10, d11, [sp, #64]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    fmov    s8, wzr
    fmov    s9, s0
    fmov    s10, s1
    fmov    s11, s2
    cbz     x21, 3f
1:
    ldr     s12, [x19], #4
    fmul    s12, s12, s9
    fcmp    s12, s11
    b.ge    2f
    fmov    s0, wzr
    b       4f
2:
    fsub    s0, s12, s10
    bl      .Lsampler_exp_f32
4:
    str     s0, [x20], #4
    fadd    s8, s8, s0
    subs    x21, x21, #1
    b.ne    1b
3:
    fmov    s0, s8
    ldp     d10, d11, [sp, #64]
    ldp     d8, d9, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

/*
 * float lfm_sampler_exp_sum_bf16(const uint16_t *x, float *weights,
 *                                size_t count, uint16_t bf16_scale,
 *                                float maximum, float threshold)
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_exp_sum_bf16)
LFM_SYM(lfm_sampler_exp_sum_bf16):
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]
    stp     d10, d11, [sp, #64]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    lsl     w8, w3, #16
    fmov    s9, w8
    fmov    s10, s0
    fmov    s11, s1
    fmov    s8, wzr
    mov     w22, #0x7fff
    cbz     x21, 3f
1:
    ldrh    w8, [x19], #2
    lsl     w8, w8, #16
    fmov    s12, w8
    fmul    s12, s12, s9
    fmov    w8, s12
    lsr     w9, w8, #16
    and     w9, w9, #1
    add     w8, w8, w22
    add     w8, w8, w9
    lsr     w8, w8, #16
    lsl     w8, w8, #16
    fmov    s12, w8
    fcmp    s12, s11
    b.ge    2f
    fmov    s0, wzr
    b       4f
2:
    fsub    s0, s12, s10
    bl      .Lsampler_exp_f32
4:
    str     s0, [x20], #4
    fadd    s8, s8, s0
    subs    x21, x21, #1
    b.ne    1b
3:
    fmov    s0, s8
    ldp     d10, d11, [sp, #64]
    ldp     d8, d9, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

/* uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count, float target) */
.p2align 2
.globl LFM_SYM(lfm_sampler_prefix_pick)
LFM_SYM(lfm_sampler_prefix_pick):
    fmov    s1, wzr
    mov     w8, wzr
    mov     x2, xzr
    cbz     x1, 4f
1:
    ldr     s2, [x0, x2, lsl #2]
    fcmp    s2, #0.0
    b.le    2f
    mov     w8, w2
2:
    fadd    s1, s1, s2
    fcmp    s0, s1
    b.lt    3f
    add     x2, x2, #1
    cmp     x2, x1
    b.lo    1b
4:
    mov     w0, w8
    ret
3:
    mov     w0, w2
    ret

/*
 * Private f32 exp for sampler inputs. The caller always supplies x <= 0 after
 * subtracting the vocabulary maximum. Values below the normal f32 exponent
 * range contribute zero to a probability sum.
 *
 * n = trunc(x * log2(e) - 0.5)
 * r = x - n * ln(2)
 * exp(x) = 2^n * P6(r)
 */
.p2align 2
.Lsampler_exp_f32:
    adr     x9, .Lsampler_exp_constants
    ldr     s1, [x9, #0]
    fmul    s1, s0, s1
    ldr     s2, [x9, #4]
    fadd    s1, s1, s2
    fcvtzs  w8, s1
    cmp     w8, #-126
    b.lt    .Lsampler_exp_f32_zero

    scvtf   s2, w8
    ldr     s3, [x9, #8]
    fmul    s2, s2, s3
    fsub    s4, s0, s2

    ldr     s3, [x9, #12]
    fmul    s3, s3, s4
    ldr     s2, [x9, #16]
    fadd    s3, s3, s2
    fmul    s3, s3, s4
    ldr     s2, [x9, #20]
    fadd    s3, s3, s2
    fmul    s3, s3, s4
    ldr     s2, [x9, #24]
    fadd    s3, s3, s2
    fmul    s3, s3, s4
    ldr     s2, [x9, #28]
    fadd    s3, s3, s2
    fmul    s3, s3, s4
    ldr     s2, [x9, #32]
    fadd    s3, s3, s2
    fmul    s3, s3, s4
    fadd    s3, s3, s2

    add     w10, w8, #127
    lsl     w10, w10, #23
    fmov    s2, w10
    fmul    s0, s3, s2
    ret

.Lsampler_exp_f32_zero:
    fmov    s0, wzr
    ret

.p2align 4
.Lsampler_exp_constants:
    .long   0x3fb8aa3b  /* log2(e) */
    .long   0xbf000000  /* -0.5 */
    .long   0x3f317218  /* ln(2) */
    .long   0x3ab60b61  /* 1 / 720 */
    .long   0x3c088889  /* 1 / 120 */
    .long   0x3d2aaaab  /* 1 / 24 */
    .long   0x3e2aaaab  /* 1 / 6 */
    .long   0x3f000000  /* 1 / 2 */
    .long   0x3f800000  /* 1 */

#if !defined(__APPLE__)
.section .note.GNU-stack,"",%progbits
#endif
