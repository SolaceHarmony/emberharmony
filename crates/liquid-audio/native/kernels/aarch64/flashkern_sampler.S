#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text

/*
 * flashkern_sampler.S — AArch64. Token selection: argmax, softmax-with-cutoff,
 * and the CDF walk. Feeds on the uniform draw produced by flashkern_prng.S.
 *
 * DETERMINISM CONTRACT — three guarantees, none of them obvious from the code:
 *
 * 1. TIE-BREAK IS LOWEST INDEX. Both argmax leaves run TWO passes: a vector
 *    reduction to find the maximum VALUE, then a scan from index 0 for the first
 *    element equal to it. A single fused pass that tracked the index while
 *    reducing would return whichever lane happened to see the max first, making
 *    the answer depend on vector width and loop peeling. Ties are common in
 *    quantised logits, so this is a real reproducibility hazard, not a
 *    theoretical one.
 *
 * 2. NaN CANNOT WIN. `fmaxnm` implements IEEE maxNum: a NaN operand returns the
 *    OTHER value rather than propagating. One poisoned logit therefore cannot
 *    capture the argmax and silently pin generation to a single token.
 *
 * 3. NO LIBM. The exponential is the local `.Lsampler_exp_f32` below — f32 range
 *    reduction plus a degree-six polynomial, entirely in this object. Unlike
 *    flashkern_rope.S (whose tables inherit the platform libm's last bits), this
 *    file is bit-identical on every machine and OS. Sampling reproducibility does
 *    not depend on which libm is linked.
 *
 * PRECISION SPLIT. The logits are handled at model precision (f32 and bf16
 * leaves); the uniform draw that indexes the distribution is a double built from
 * 53 PRNG bits. Those are deliberately different: narrowing the draw to bf16
 * resolution would make any token with mass below ~1/256 unreachable, which
 * truncates the tail off the distribution rather than merely rounding it.
 */

/*
 * uint32_t lfm_sampler_argmax_f32(const float *x, size_t count)
 *
 * Pass 1: 4-lane `fmaxnm` reduction + `fmaxnmv` horizontal max + scalar tail.
 * Pass 2: linear scan for the FIRST index equal to that maximum (guarantee 1).
 * Accumulator starts at 0xff800000 (-inf). count == 0 returns 0.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_argmax_f32)
LFM_PRIVATE(LFM_SYM(lfm_sampler_argmax_f32))
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

/*
 * uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count)
 *
 * As the f32 leaf, including the two-pass lowest-index tie-break. `shll
 * v1.4s, v1.4h, #16` lifts four BF16 lanes to F32 by bit-shift — exact and
 * lossless, never a conversion instruction.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_argmax_bf16)
LFM_PRIVATE(LFM_SYM(lfm_sampler_argmax_bf16))
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
 * One fused pass producing the unnormalised softmax weights and their sum:
 *
 *     v = x[i] * scale                 // scale = 1/temperature
 *     if (v < threshold)  w = 0        // top-k / top-p cutoff, no exp taken
 *     else                w = exp(v - maximum)
 *     weights[i] = w ;  sum += w
 *
 * Subtracting `maximum` is the standard overflow guard, so the exp argument is
 * always <= 0 — which is why the local exp has no overflow path. Note the
 * threshold is compared against the SCALED logit, before max-subtraction, so a
 * caller's cutoff is expressed in the same units as its temperature.
 *
 * `sum` accumulates sequentially in F32, in index order: that ordering is the
 * contract, since the CDF walk below must see exactly this total.
 *
 * The local exp leaf uses f32 range reduction and a degree-six polynomial.
 * Traversal, scaling, thresholding, exponentiation, writes, and reduction all
 * remain in this assembly object.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_exp_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_sampler_exp_sum_f32))
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
 *
 * As the f32 leaf, with one deliberate extra rounding step: after `v = x[i] *
 * scale`, the product is rounded BACK to BF16 precision (the open-coded
 * round-to-nearest-even at w22 = 0x7fff — same sequence as LFM_RNE_WIDEN in
 * flashkern_math.S) BEFORE the threshold compare and the exp.
 *
 * Keeping that product in full F32 would be more accurate and WRONG: the
 * reference materialises the scaled logit as BF16, so the cutoff decision and
 * the exp argument must both see the rounded value. Note `scale` arrives as
 * BF16 bits and is lifted by shift, so the multiply itself has BF16 inputs.
 *
 * The weights and their sum remain F32 — only the scaled logit is narrowed.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_exp_sum_bf16)
LFM_PRIVATE(LFM_SYM(lfm_sampler_exp_sum_bf16))
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

/*
 * uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count,
 *                                  float target)
 *
 * The CDF walk. Accumulates `weights` in index order and returns the first index
 * where the running sum exceeds `target` (the uniform draw scaled by the total
 * from exp_sum). Accumulation order matches exp_sum's exactly — both walk low to
 * high in F32 — so the partial sums here reproduce the total computed there.
 *
 * SHORTFALL GUARANTEE — the reason for the `w8` register:
 *
 * `target` is derived from a double uniform scaled by an F32 sum, so rounding can
 * leave it a few ULP ABOVE the final running sum. The loop then falls out
 * without ever triggering. Rather than returning 0 or count-1, `w8` carries the
 * last index whose weight was STRICTLY POSITIVE, and that is what is returned.
 *
 * This makes it impossible to select a token that was cut off by the threshold
 * (weight 0) purely because of a rounding shortfall — a bug that would be
 * invisible in aggregate statistics and would occasionally emit a token the
 * sampling policy had explicitly excluded.
 */
.p2align 2
.globl LFM_SYM(lfm_sampler_prefix_pick)
LFM_PRIVATE(LFM_SYM(lfm_sampler_prefix_pick))
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
 *
 * P6 is the degree-six Taylor series of e^r evaluated by Horner; the constants
 * below are literally 1/720, 1/120, 1/24, 1/6, 1/2, 1. `2^n` is built directly
 * as the exponent field ((n + 127) << 23) rather than by calling ldexp.
 *
 * The `- 0.5` in the range reduction makes `trunc` behave as round-to-nearest
 * over the negative domain this leaf is restricted to (trunc rounds toward zero,
 * so the bias is what centres the residual r about 0 and keeps |r| <= ln(2)/2,
 * where the degree-six truncation is accurate).
 *
 * Being local and libm-free is the point: sampling stays bit-identical across
 * platforms. Replacing this with expf() would make token choice depend on which
 * libm is linked.
 *
 * Underflow: n < -126 is below the normal f32 exponent range, so the result is
 * flushed to exactly 0 — such a token contributes nothing to the CDF and, by the
 * shortfall guarantee in prefix_pick, can never be selected. No overflow path
 * exists because x <= 0 always.
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
