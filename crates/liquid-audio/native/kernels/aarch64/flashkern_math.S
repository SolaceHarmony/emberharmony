// flashkern_math.S — AArch64 scalar numeric leaves for the LFM2 decode path.
//
// NUMERIC CONTRACT (why these are hand-written rather than compiler output):
//
// 1. Rounding points are placed, not minimised. Where the reference stores a
//    BF16 value, this code rounds to BF16 at that same point via LFM_RNE_WIDEN;
//    where the reference holds F32, the value stays F32. Keeping more precision
//    than the reference is a DEVIATION here, not an improvement — the weights
//    were trained through the reference's rounding, so faithfulness to its
//    intermediate precision outranks mathematical accuracy.
//
// 2. BF16 -> F32 lift is `lsl #16` — a pure bit move. Exact and lossless, never
//    a conversion instruction, so no implementation-defined rounding enters.
//
// 3. F32 -> BF16 is round-to-nearest-EVEN, open-coded (see LFM_RNE_WIDEN):
//        bits += 0x7fff + ((bits >> 16) & 1)
//    The `& 1` term is the destination LSB, which is what makes exact ties round
//    to even rather than always up. The result is left in an F32 register with
//    the low 16 bits cleared, so a "BF16 value" stays register-resident and is
//    only narrowed to 16 bits at a store.
//
// 4. Reductions are strictly sequential — one `fadd` per element, no vector
//    lanes and no pairwise tree. A tree reduction is usually MORE accurate and
//    would still be wrong here: it changes the summation order, and the order is
//    the contract. This is what makes results bit-reproducible run to run and
//    lane-count independent.
//
// 5. Division and square root use exact IEEE `fdiv`/`fsqrt`, never `frsqrte`/
//    `frecpe`. The estimate instructions are faster but their Newton refinement
//    diverges from the reference in the last bits, and they are permitted to
//    differ across implementations.
//
// Recorded evidence of these numerics lives in
// native/tests/fixtures/conformer/ (72 per-stage tensors, checkpoint
// LFM2.5-Audio-1.5B c362a062) — kept as provenance for the values, independent
// of any test harness that once consumed them.

#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text

// float lfm_rsqrt_size(uint64_t n) -> 1.0f / sqrtf((float)n)
// The RMSNorm/attention scale factor. Exact fsqrt + fdiv (contract note 5).
.p2align 2
.globl LFM_SYM(lfm_rsqrt_size)
LFM_PRIVATE(LFM_SYM(lfm_rsqrt_size))
LFM_SYM(lfm_rsqrt_size):
    ucvtf   s1, x0
    fsqrt   s1, s1
    fmov    s0, #1.0
    fdiv    s0, s0, s1
    ret

// float lfm_inv_rms_f32(float sumsq /*s0*/, float eps /*s1*/, uint64_t n /*x0*/)
//   -> 1.0f / sqrtf(sumsq / n + eps)
// Operand ORDER is the contract: divide by n first, THEN add eps, then sqrt,
// then reciprocate. Folding eps in before the division, or scaling after the
// root, changes the low bits for small sumsq and breaks reference agreement.
.p2align 2
.globl LFM_SYM(lfm_inv_rms_f32)
LFM_PRIVATE(LFM_SYM(lfm_inv_rms_f32))
LFM_SYM(lfm_inv_rms_f32):
    ucvtf   s2, x0
    fdiv    s0, s0, s2
    fadd    s0, s0, s1
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
    ret

// float lfm_sum_f32(const float *x /*x0*/, uint64_t n /*x1*/)
// Strictly sequential F32 accumulation, low index to high (contract note 4).
// Deliberately NOT vectorised: lane-parallel accumulation reorders the sum.
.p2align 2
.globl LFM_SYM(lfm_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_sum_f32))
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

// float lfm_bf16_sumsq_stride_f32(const uint16_t *x /*x0*/, uint64_t end /*x1*/,
//                                 uint64_t start /*x2*/, uint64_t stride /*x3*/)
//   -> sum of (bf16→f32 x[i])^2 for i = start, start+stride, ... < end
// The RMSNorm sum of squares over a strided view. Each element is lifted by
// `lsl #16` (exact, note 2), squared and accumulated in F32, sequentially
// (note 4). The stride lets a row of an interleaved plane be reduced without
// materialising a contiguous copy.
.p2align 2
.globl LFM_SYM(lfm_bf16_sumsq_stride_f32)
LFM_PRIVATE(LFM_SYM(lfm_bf16_sumsq_stride_f32))
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
LFM_PRIVATE(LFM_SYM(lfm_bf16_bias_add_f32))
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

.p2align 2
.globl LFM_SYM(lfm_bf16_copy_bytes)
LFM_PRIVATE(LFM_SYM(lfm_bf16_copy_bytes))
LFM_SYM(lfm_bf16_copy_bytes):
    cbz     x2, 2f
1:
    ldrh    w3, [x0], #2
    strh    w3, [x1], #2
    subs    x2, x2, #1
    b.ne    1b
2:
    ret

.p2align 2
.globl LFM_SYM(lfm_bf16_unlift_bits)
LFM_PRIVATE(LFM_SYM(lfm_bf16_unlift_bits))
LFM_SYM(lfm_bf16_unlift_bits):
    ldrh    w0, [x0]
    lsl     w0, w0, #16
    ret

// Round an F32 register to BF16 precision, in place, round-to-nearest-EVEN.
// The result stays an F32 whose low 16 mantissa bits are zero, so a logical
// BF16 value remains register-resident with no store/reload round trip.
//
//     tie   = (bits >> 16) & 1     // destination LSB — the "even" term
//     bits += 0x7fff + tie         // half-way cases round toward even
//     bits  = (bits >> 16) << 16   // clear what BF16 cannot represent
//
// Without the `tie` term this would be round-half-UP and would drift from the
// reference on exact ties. `bias` is an explicit operand: callers may choose any
// scratch register, and no macro expansion silently depends on a leaf-global
// w14 convention.
.macro LFM_RNE_WIDEN source, target, bits, tie, bias
    fmov    \bits, \source
    lsr     \tie, \bits, #16
    and     \tie, \tie, #1
    add     \bits, \bits, \bias
    add     \bits, \bits, \tie
    lsr     \bits, \bits, #16
    lsl     \bits, \bits, #16
    fmov    \target, \bits
.endm

// void lfm_bf16_rope_neox(uint16_t *x /*x0*/, const uint16_t *cos /*x1*/,
//                         const uint16_t *sin /*x2*/, uint64_t dim /*x3*/)
// Rotary position embedding, NeoX pairing, applied in place over one head.
//
// PAIRING: element i is rotated against element i + dim/2 (see `x5 = x0 +
// (dim/2)*2`). That is the GPT-NeoX split-half convention, NOT the GPT-J
// interleaved (i, i+1) one. Choosing the wrong pairing produces plausible
// audio that degrades with sequence length, so it is stated here explicitly.
// Odd `dim` is rejected (`tbnz x3, #0`) rather than silently truncated.
//
//     out[i]        =  x[i]*cos - x[i+d/2]*sin
//     out[i + d/2]  =  x[i+d/2]*cos + x[i]*sin
//
// ROUNDING: every product and every sum passes through LFM_RNE_WIDEN — six
// rounding points per pair, not one. Accumulating the pair in full F32 and
// rounding once at the end would be more accurate and WRONG: the reference
// materialises BF16 at each of these steps, and the model was trained through
// exactly that loss. This is contract note 1 in its sharpest form.
.p2align 2
.globl LFM_SYM(lfm_bf16_rope_neox)
LFM_PRIVATE(LFM_SYM(lfm_bf16_rope_neox))
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
    LFM_RNE_WIDEN s4, s4, w12, w13, w14
    fneg    s5, s1
    fmul    s5, s5, s3
    LFM_RNE_WIDEN s5, s5, w12, w13, w14
    fadd    s4, s4, s5
    LFM_RNE_WIDEN s4, s4, w12, w13, w14
    lsr     w12, w12, #16
    strh    w12, [x0], #2

    fmul    s4, s1, s2
    LFM_RNE_WIDEN s4, s4, w12, w13, w14
    fmul    s5, s0, s3
    LFM_RNE_WIDEN s5, s5, w12, w13, w14
    fadd    s4, s4, s5
    LFM_RNE_WIDEN s4, s4, w12, w13, w14
    lsr     w12, w12, #16
    strh    w12, [x5], #2

    subs    x4, x4, #1
    b.ne    1b
3:
    ret

// Decode projection superkernel. Four checkpoint rows share each activation
// load while eight accumulators preserve gemm_nt_impl's per-row arithmetic
// order. Both logical BF16 RNE boundaries remain register operations and only
// terminal output is published.
//
// ABI: x0 input bytes, x1 checkpoint weight bytes, x2 residual bytes,
//      x3 output bf16, x4 output rows, x5 reduction depth.
// All source views tolerate byte-unaligned starts. v8-v15 are untouched, so the
// leaf needs no frame and has no intermediate stores.
.p2align 2
.globl LFM_SYM(lfm_bf16_gemv_rne_bf16)
LFM_PRIVATE(LFM_SYM(lfm_bf16_gemv_rne_bf16))
LFM_SYM(lfm_bf16_gemv_rne_bf16):
    // Adapt the five-argument no-residual ABI to the shared six-register body.
    mov     x5, x4
    mov     x4, x3
    mov     x3, x2
    mov     x2, xzr
    b       .Lgemv_rne_body

.p2align 2
.globl LFM_SYM(lfm_bf16_gemv_rne_add_bf16)
LFM_PRIVATE(LFM_SYM(lfm_bf16_gemv_rne_add_bf16))
LFM_SYM(lfm_bf16_gemv_rne_add_bf16):
.Lgemv_rne_body:
    cbz     x4, 99f
    cbz     x5, 99f
    lsl     x6, x5, #1
    lsr     x14, x4, #2
    and     x4, x4, #3
    mov     w15, #0x7fff
    cbz     x14, 20f

10: // Four output rows, eight accumulators, one activation stream.
    mov     x7, x1
    add     x8, x7, x6
    add     x9, x8, x6
    add     x10, x9, x6
    mov     x11, x0
    lsr     x12, x5, #3
    and     x13, x5, #7
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    cbz     x12, 12f

11: // One activation load feeds four immutable checkpoint rows.
    ldr     q16, [x11], #16
    ldr     q19, [x7], #16
    ldr     q22, [x8], #16
    ldr     q25, [x9], #16
    ldr     q28, [x10], #16
    shll    v17.4s, v16.4h, #16
    shll2   v18.4s, v16.8h, #16
    shll    v20.4s, v19.4h, #16
    shll2   v21.4s, v19.8h, #16
    fmla    v0.4s, v17.4s, v20.4s
    fmla    v1.4s, v18.4s, v21.4s
    shll    v23.4s, v22.4h, #16
    shll2   v24.4s, v22.8h, #16
    fmla    v2.4s, v17.4s, v23.4s
    fmla    v3.4s, v18.4s, v24.4s
    shll    v26.4s, v25.4h, #16
    shll2   v27.4s, v25.8h, #16
    fmla    v4.4s, v17.4s, v26.4s
    fmla    v5.4s, v18.4s, v27.4s
    shll    v29.4s, v28.4h, #16
    shll2   v30.4s, v28.8h, #16
    fmla    v6.4s, v17.4s, v29.4s
    fmla    v7.4s, v18.4s, v30.4s
    subs    x12, x12, #1
    b.ne    11b

12: // Match vaddvq_f32(vaddq_f32(low, high)) for every row.
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    fadd    v2.4s, v2.4s, v3.4s
    faddp   v2.4s, v2.4s, v2.4s
    faddp   s2, v2.2s
    fadd    v4.4s, v4.4s, v5.4s
    faddp   v4.4s, v4.4s, v4.4s
    faddp   s4, v4.2s
    fadd    v6.4s, v6.4s, v7.4s
    faddp   v6.4s, v6.4s, v6.4s
    faddp   s6, v6.2s
    cbz     x13, 14f

13: // Scalar K tail remains after the vector reduction, as in production.
    ldrh    w16, [x11], #2
    lsl     w16, w16, #16
    fmov    s16, w16
    ldrh    w17, [x7], #2
    lsl     w17, w17, #16
    fmov    s17, w17
    fmadd   s0, s16, s17, s0
    ldrh    w17, [x8], #2
    lsl     w17, w17, #16
    fmov    s17, w17
    fmadd   s2, s16, s17, s2
    ldrh    w17, [x9], #2
    lsl     w17, w17, #16
    fmov    s17, w17
    fmadd   s4, s16, s17, s4
    ldrh    w17, [x10], #2
    lsl     w17, w17, #16
    fmov    s17, w17
    fmadd   s6, s16, s17, s6
    subs    x13, x13, #1
    b.ne    13b

14: // Pack four dots and perform both RNE stages entirely in registers.
    mov     v16.s[0], v0.s[0]
    mov     v16.s[1], v2.s[0]
    mov     v16.s[2], v4.s[0]
    mov     v16.s[3], v6.s[0]
    ushr    v17.4s, v16.4s, #16
    movi    v18.4s, #1
    and     v17.16b, v17.16b, v18.16b
    dup     v18.4s, w15
    add     v16.4s, v16.4s, v17.4s
    add     v16.4s, v16.4s, v18.4s
    ushr    v16.4s, v16.4s, #16
    shl     v16.4s, v16.4s, #16
    cbz     x2, 15f
    ldr     d17, [x2], #8
    shll    v17.4s, v17.4h, #16
    fadd    v16.4s, v16.4s, v17.4s
    ushr    v17.4s, v16.4s, #16
    movi    v18.4s, #1
    and     v17.16b, v17.16b, v18.16b
    dup     v18.4s, w15
    add     v16.4s, v16.4s, v17.4s
    add     v16.4s, v16.4s, v18.4s
    ushr    v16.4s, v16.4s, #16
    xtn     v16.4h, v16.4s
    str     d16, [x3], #8
    b       16f
15:
    ushr    v16.4s, v16.4s, #16
    xtn     v16.4h, v16.4s
    str     d16, [x3], #8

16:
    add     x1, x1, x6, lsl #2
    subs    x14, x14, #1
    b.ne    10b

20: // N%4 tail: the same arithmetic and terminal-store-only contract.
    cbz     x4, 99f
21:
    mov     x7, x1
    mov     x11, x0
    lsr     x12, x5, #3
    and     x13, x5, #7
    movi    v0.4s, #0
    movi    v1.4s, #0
    cbz     x12, 23f
22:
    ldr     q16, [x11], #16
    ldr     q19, [x7], #16
    shll    v17.4s, v16.4h, #16
    shll2   v18.4s, v16.8h, #16
    shll    v20.4s, v19.4h, #16
    shll2   v21.4s, v19.8h, #16
    fmla    v0.4s, v17.4s, v20.4s
    fmla    v1.4s, v18.4s, v21.4s
    subs    x12, x12, #1
    b.ne    22b
23:
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    cbz     x13, 25f
24:
    ldrh    w16, [x11], #2
    ldrh    w17, [x7], #2
    lsl     w16, w16, #16
    lsl     w17, w17, #16
    fmov    s16, w16
    fmov    s17, w17
    fmadd   s0, s16, s17, s0
    subs    x13, x13, #1
    b.ne    24b
25:
    fmov    w16, s0
    lsr     w17, w16, #16
    and     w17, w17, #1
    add     w16, w16, w15
    add     w16, w16, w17
    lsr     w17, w16, #16
    cbz     x2, 26f
    lsl     w18, w17, #16
    fmov    s16, w18
    ldrh    w18, [x2], #2
    lsl     w18, w18, #16
    fmov    s17, w18
    fadd    s16, s16, s17
    fmov    w16, s16
    lsr     w17, w16, #16
    and     w17, w17, #1
    add     w16, w16, w15
    add     w16, w16, w17
    lsr     w17, w16, #16
    strh    w17, [x3], #2
    b       27f
26:
    strh    w17, [x3], #2
27:
    mov     x1, x7
    subs    x4, x4, #1
    b.ne    21b

99:
    ret

#if !defined(__APPLE__)
.section .note.GNU-stack,"",%progbits
#endif
