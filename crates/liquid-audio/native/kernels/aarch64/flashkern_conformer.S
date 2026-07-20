#include "lfm_asm_visibility.h"

// Conformer/adapter leaves. Scalar, strictly ordered — the parity oracle
// (native/tests/fixtures/conformer) was captured from candle's op-for-op
// ladders, and bf16 rounding (RNE, the LFM_RNE_WIDEN convention from
// flashkern_math.S) is applied exactly where candle's bf16 arithmetic rounds:
// once per tensor op. No FMA contraction anywhere.
//
// "ONCE PER TENSOR OP" IS THE WHOLE CONTRACT. Every per-leaf note below that
// says "one round" / "two rounds" / "each step rounds" is counting candle's
// rounding points, not describing an implementation detail. Fusing two leaves to
// skip an intermediate round is the single easiest way to walk this file off its
// fixtures: it is arithmetically MORE accurate and produces a different tensor.
// The rounding count is load-bearing; the speed is not.
//
// LIBM DEPENDENCY — this is the LARGEST libm user in the kernel set:
// expf (silu, glu, softmax), erff (gelu), and sinf/cosf (positional encoding).
// Those are not bit-identical across platforms (Apple libm / glibc / musl), so
// the conformer output — and any end-to-end byte oracle downstream of it — is
// reproducible on a given platform but not guaranteed across them. Three files
// carry libm: this one, flashkern_rope.S (powf/cosf/sinf) and
// flashkern_frontend.S (logf). math/prng/sampler/sesame are libm-free and
// bit-identical everywhere.
//
// VARIANCE CONVENTIONS DIFFER ACROSS THE TREE, ON PURPOSE. Three normalisers,
// three formulas, each faithful to its own reference — do not harmonise them:
//   lfm_ln_bf16        (here)          var = sumsq / cols        BIASED, eps per candle
//   lfm_rowstat_f32    (frontend.S)    std = sqrt(sumsq/(n-1)) + eps   ddof-1, eps AFTER
//   lfm_inv_rms_f32    (math.S)        1 / sqrt(sumsq/n + eps)         eps INSIDE
// Seeing two of these side by side reads like a bug in one of them. It is not.
//
// WHY THE BF16 GEMM AVOIDS FEAT_BF16 (see lfm_bf16_gemm_nt_f32_scalar): the
// stated reason is weight movement — halfwords are unlifted straight into f32
// registers, never copied, packed or widened into memory. There is a second,
// numerical reason to leave it alone: BFDOT accumulates pairwise with its own
// rounding contract, which is NOT candle's sequential f32 k-ascending sum. A
// BFDOT rewrite would therefore change the arithmetic order as well as the data
// movement, and both are pinned by the fixtures.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

// f32 in \src (s-register) -> RNE bf16 bits in \wd (w-register), using \wt as
// scratch and \wk holding 0x7fff.
.macro BF16_RNE src, wd, wt, wk
    fmov    \wd, \src
    lsr     \wt, \wd, #16
    and     \wt, \wt, #1
    add     \wd, \wd, \wk
    add     \wd, \wd, \wt
    lsr     \wd, \wd, #16
.endm

// bf16 bits in \ws -> f32 in \dst.
.macro BF16_WIDEN ws, dst
    lsl     \ws, \ws, #16
    fmov    \dst, \ws
.endm

.text
.p2align 2

// void lfm_bf16_widen_f32(const uint16_t *x, float *y, uint64_t n)
.globl LFM_SYM(lfm_bf16_widen_f32)
LFM_PRIVATE(LFM_SYM(lfm_bf16_widen_f32))
LFM_SYM(lfm_bf16_widen_f32):
    cbz     x2, 8f
1:
    ldrh    w3, [x0], #2
    lsl     w3, w3, #16
    str     w3, [x1], #4
    subs    x2, x2, #1
    b.ne    1b
8:
    ret

// int lfm_ln_bf16(const uint16_t *x, const uint16_t *w, const uint16_t *b,
//                 uint16_t *y, uint64_t rows, uint64_t cols, float eps /*s0*/)
// layer_norm_slow: f32 mean (sum/cols), f32 biased var (sumsq/cols),
// normalize f32, round bf16, then weight/bias in bf16 arithmetic.
.p2align 2
.globl LFM_SYM(lfm_ln_bf16)
LFM_PRIVATE(LFM_SYM(lfm_ln_bf16))
LFM_SYM(lfm_ln_bf16):
    cbz     x0, 9f
    cbz     x3, 9f
    cbz     x5, 9f
    mov     w14, #0x7fff
    mov     x6, #0              // row
1:
    cmp     x6, x4
    b.hs    8f
    mul     x7, x6, x5
    add     x7, x0, x7, lsl #1  // x row
    // pass 1: sum
    fmov    s1, wzr
    mov     x8, #0
2:
    ldrh    w9, [x7, x8, lsl #1]
    BF16_WIDEN w9, s2
    fadd    s1, s1, s2
    add     x8, x8, #1
    cmp     x8, x5
    b.lo    2b
    ucvtf   s3, x5
    fdiv    s4, s1, s3          // mean
    // pass 2: biased variance
    fmov    s1, wzr
    mov     x8, #0
3:
    ldrh    w9, [x7, x8, lsl #1]
    BF16_WIDEN w9, s2
    fsub    s2, s2, s4
    fmul    s2, s2, s2
    fadd    s1, s1, s2
    add     x8, x8, #1
    cmp     x8, x5
    b.lo    3b
    fdiv    s1, s1, s3
    fadd    s1, s1, s0
    fsqrt   s5, s1              // denom
    // pass 3: normalize + bf16 weight/bias tail
    mul     x10, x6, x5
    add     x10, x3, x10, lsl #1 // y row
    mov     x8, #0
4:
    ldrh    w9, [x7, x8, lsl #1]
    BF16_WIDEN w9, s2
    fsub    s2, s2, s4
    fdiv    s2, s2, s5
    BF16_RNE s2, w9, w11, w14   // round normed to bf16
    BF16_WIDEN w9, s2
    ldrh    w12, [x1, x8, lsl #1]
    BF16_WIDEN w12, s6
    fmul    s2, s2, s6
    BF16_RNE s2, w9, w11, w14
    BF16_WIDEN w9, s2
    ldrh    w12, [x2, x8, lsl #1]
    BF16_WIDEN w12, s6
    fadd    s2, s2, s6
    BF16_RNE s2, w9, w11, w14
    strh    w9, [x10, x8, lsl #1]
    add     x8, x8, #1
    cmp     x8, x5
    b.lo    4b
    add     x6, x6, #1
    b       1b
8:
    mov     w0, #0
    ret
9:
    mov     w0, #-22
    ret

// void lfm_bn_bf16(const uint16_t *x, const uint16_t *mean,
//                  const uint16_t *denom, const uint16_t *w, const uint16_t *b,
//                  uint16_t *y, uint64_t channels /*x6*/, uint64_t t /*x7*/)
// BatchNorm eval, the all-bf16 broadcast chain: sub, div, mul, add — each
// step rounds. x is (channels x t) channel-major.
.p2align 2
.globl LFM_SYM(lfm_bn_bf16)
LFM_PRIVATE(LFM_SYM(lfm_bn_bf16))
LFM_SYM(lfm_bn_bf16):
    mov     w14, #0x7fff
    mov     x8, #0              // channel
1:
    cmp     x8, x6
    b.hs    8f
    ldrh    w9, [x1, x8, lsl #1]
    BF16_WIDEN w9, s1           // mean_c
    ldrh    w9, [x2, x8, lsl #1]
    BF16_WIDEN w9, s2           // denom_c
    ldrh    w9, [x3, x8, lsl #1]
    BF16_WIDEN w9, s3           // w_c
    ldrh    w9, [x4, x8, lsl #1]
    BF16_WIDEN w9, s4           // b_c
    mul     x10, x8, x7
    add     x11, x0, x10, lsl #1
    add     x12, x5, x10, lsl #1
    mov     x10, #0
2:
    cmp     x10, x7
    b.hs    3f
    ldrh    w9, [x11, x10, lsl #1]
    BF16_WIDEN w9, s5
    fsub    s5, s5, s1
    BF16_RNE s5, w9, w13, w14
    BF16_WIDEN w9, s5
    fdiv    s5, s5, s2
    BF16_RNE s5, w9, w13, w14
    BF16_WIDEN w9, s5
    fmul    s5, s5, s3
    BF16_RNE s5, w9, w13, w14
    BF16_WIDEN w9, s5
    fadd    s5, s5, s4
    BF16_RNE s5, w9, w13, w14
    strh    w9, [x12, x10, lsl #1]
    add     x10, x10, #1
    b       2b
3:
    add     x8, x8, #1
    b       1b
8:
    ret

// void lfm_relu_bf16(uint16_t *x, uint64_t n) — negative bf16 -> +0.
.p2align 2
.globl LFM_SYM(lfm_relu_bf16)
LFM_PRIVATE(LFM_SYM(lfm_relu_bf16))
LFM_SYM(lfm_relu_bf16):
    cbz     x1, 8f
1:
    ldrh    w2, [x0]
    tbz     w2, #15, 2f
    strh    wzr, [x0]
2:
    add     x0, x0, #2
    subs    x1, x1, #1
    b.ne    1b
8:
    ret

// int lfm_silu_bf16(uint16_t *x, uint64_t n) — x*sigmoid(x), f32, one round.
.p2align 2
.globl LFM_SYM(lfm_silu_bf16)
LFM_PRIVATE(LFM_SYM(lfm_silu_bf16))
LFM_SYM(lfm_silu_bf16):
    cbz     x0, 9f
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]
    mov     x19, x0
    mov     x20, x1
1:
    cbz     x20, 2f
    ldrh    w0, [x19]
    BF16_WIDEN w0, s8           // x
    fneg    s0, s8
    bl      LFM_CALL(expf)
    fmov    s1, #1.0
    fadd    s0, s0, s1          // 1 + e^-x
    fdiv    s0, s8, s0          // x / (1+e^-x)
    mov     w14, #0x7fff
    BF16_RNE s0, w0, w1, w14
    strh    w0, [x19], #2
    sub     x20, x20, #1
    b       1b
2:
    mov     w0, #0
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
9:
    mov     w0, #-22
    ret

// int lfm_glu_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, uint64_t n)
// candle order: sigmoid rounds to bf16, then the product rounds.
.p2align 2
.globl LFM_SYM(lfm_glu_bf16)
LFM_PRIVATE(LFM_SYM(lfm_glu_bf16))
LFM_SYM(lfm_glu_bf16):
    cbz     x0, 9f
    cbz     x1, 9f
    cbz     x2, 9f
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    str     d8, [sp, #48]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
1:
    cbz     x22, 2f
    ldrh    w0, [x20], #2       // b
    BF16_WIDEN w0, s0
    fneg    s0, s0
    bl      LFM_CALL(expf)
    fmov    s1, #1.0
    fadd    s0, s0, s1
    fdiv    s0, s1, s0          // sigmoid(b) f32
    mov     w14, #0x7fff
    BF16_RNE s0, w0, w1, w14    // round sigmoid
    BF16_WIDEN w0, s0
    ldrh    w1, [x19], #2       // a
    BF16_WIDEN w1, s8
    fmul    s0, s8, s0
    BF16_RNE s0, w0, w1, w14    // round product
    strh    w0, [x21], #2
    sub     x22, x22, #1
    b       1b
2:
    mov     w0, #0
    ldr     d8, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
9:
    mov     w0, #-22
    ret

// int lfm_gelu_erf_bf16(uint16_t *x, uint64_t n)
// 0.5 * x * (1 + erf(x / sqrt(2))), f32, one round.
.p2align 2
.globl LFM_SYM(lfm_gelu_erf_bf16)
LFM_PRIVATE(LFM_SYM(lfm_gelu_erf_bf16))
LFM_SYM(lfm_gelu_erf_bf16):
    cbz     x0, 9f
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    // 1/sqrt(2) as f32 bits 0x3f3504f3
    mov     w0, #0x04f3
    movk    w0, #0x3f35, lsl #16
    fmov    s9, w0
1:
    cbz     x20, 2f
    ldrh    w0, [x19]
    BF16_WIDEN w0, s8           // x
    fmul    s0, s8, s9
    bl      LFM_CALL(erff)
    fmov    s1, #1.0
    fadd    s0, s0, s1          // 1 + erf
    fmov    s1, #0.5
    fmul    s1, s1, s8          // 0.5 * x
    fmul    s0, s1, s0
    mov     w14, #0x7fff
    BF16_RNE s0, w0, w1, w14
    strh    w0, [x19], #2
    sub     x20, x20, #1
    b       1b
2:
    mov     w0, #0
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
9:
    mov     w0, #-22
    ret

// void lfm_residual_half_bf16(uint16_t *r, const uint16_t *h, uint64_t n)
// r = bf16(r + bf16(0.5*h)) — two rounds, candle op order.
.p2align 2
.globl LFM_SYM(lfm_residual_half_bf16)
LFM_PRIVATE(LFM_SYM(lfm_residual_half_bf16))
LFM_SYM(lfm_residual_half_bf16):
    cbz     x2, 8f
    mov     w14, #0x7fff
    fmov    s3, #0.5
1:
    ldrh    w3, [x1], #2
    BF16_WIDEN w3, s0
    fmul    s0, s0, s3
    BF16_RNE s0, w3, w4, w14
    BF16_WIDEN w3, s0
    ldrh    w4, [x0]
    BF16_WIDEN w4, s1
    fadd    s0, s1, s0
    BF16_RNE s0, w3, w4, w14
    strh    w3, [x0], #2
    subs    x2, x2, #1
    b.ne    1b
8:
    ret

// void lfm_add_bf16(uint16_t *r, const uint16_t *h, uint64_t n)
.p2align 2
.globl LFM_SYM(lfm_add_bf16)
LFM_PRIVATE(LFM_SYM(lfm_add_bf16))
LFM_SYM(lfm_add_bf16):
    cbz     x2, 8f
    mov     w14, #0x7fff
1:
    ldrh    w3, [x1], #2
    BF16_WIDEN w3, s0
    ldrh    w4, [x0]
    BF16_WIDEN w4, s1
    fadd    s0, s1, s0
    BF16_RNE s0, w3, w4, w14
    strh    w3, [x0], #2
    subs    x2, x2, #1
    b.ne    1b
8:
    ret

// int lfm_softmax_rows_f32(float *x, uint64_t rows, uint64_t cols)
// candle ladder: max, subtract, exp (libm expf), sum, divide — all f32.
.p2align 2
.globl LFM_SYM(lfm_softmax_rows_f32)
LFM_PRIVATE(LFM_SYM(lfm_softmax_rows_f32))
LFM_SYM(lfm_softmax_rows_f32):
    cbz     x0, 9f
    cbz     x2, 9f
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]
    mov     x19, x0             // row ptr
    mov     x21, x1             // rows left
    mov     x22, x2             // cols
1:
    cbz     x21, 7f
    // max
    ldr     s8, [x19]
    mov     x20, #1
2:
    cmp     x20, x22
    b.hs    3f
    ldr     s0, [x19, x20, lsl #2]
    fmax    s8, s8, s0
    add     x20, x20, #1
    b       2b
3:
    // exp(x - max) in place + sum
    fmov    s9, wzr
    mov     x20, #0
4:
    cmp     x20, x22
    b.hs    5f
    ldr     s0, [x19, x20, lsl #2]
    fsub    s0, s0, s8
    bl      LFM_CALL(expf)
    str     s0, [x19, x20, lsl #2]
    fadd    s9, s9, s0
    add     x20, x20, #1
    b       4b
5:
    // divide
    mov     x20, #0
6:
    cmp     x20, x22
    b.hs    65f
    ldr     s0, [x19, x20, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x19, x20, lsl #2]
    add     x20, x20, #1
    b       6b
65:
    add     x19, x19, x22, lsl #2
    sub     x21, x21, #1
    b       1b
7:
    mov     w0, #0
    ldp     d8, d9, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
9:
    mov     w0, #-22
    ret

// void lfm_sgemm_f32(const float *a, const float *b, float *c,
//                    uint64_t m, uint64_t n, uint64_t k)
// C(MxN) = A(MxK) . B(KxN), scalar, k-ascending accumulate (off-Apple path).
.p2align 2
.globl LFM_SYM(lfm_sgemm_f32)
LFM_PRIVATE(LFM_SYM(lfm_sgemm_f32))
LFM_SYM(lfm_sgemm_f32):
    mov     x6, #0              // i
1:
    cmp     x6, x3
    b.hs    8f
    mov     x7, #0              // j
2:
    cmp     x7, x4
    b.hs    5f
    fmov    s0, wzr
    mov     x8, #0              // p
    mul     x9, x6, x5          // i*K
    add     x9, x0, x9, lsl #2  // A row
3:
    cmp     x8, x5
    b.hs    4f
    ldr     s1, [x9, x8, lsl #2]
    mul     x10, x8, x4
    add     x10, x10, x7
    ldr     s2, [x1, x10, lsl #2]
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x8, x8, #1
    b       3b
4:
    mul     x10, x6, x4
    add     x10, x10, x7
    str     s0, [x2, x10, lsl #2]
    add     x7, x7, #1
    b       2b
5:
    add     x6, x6, #1
    b       1b
8:
    ret

// void lfm_sgemm_nt_f32(const float *a, const float *bt, float *c,
//                       uint64_t m, uint64_t n, uint64_t k)
// C(MxN) = A(MxK) . B(NxK)^T — both operands row-contiguous dots.
.p2align 2
.globl LFM_SYM(lfm_sgemm_nt_f32)
LFM_PRIVATE(LFM_SYM(lfm_sgemm_nt_f32))
LFM_SYM(lfm_sgemm_nt_f32):
    mov     x6, #0
1:
    cmp     x6, x3
    b.hs    8f
    mov     x7, #0
2:
    cmp     x7, x4
    b.hs    5f
    fmov    s0, wzr
    mul     x9, x6, x5
    add     x9, x0, x9, lsl #2
    mul     x10, x7, x5
    add     x10, x1, x10, lsl #2
    mov     x8, #0
3:
    cmp     x8, x5
    b.hs    4f
    ldr     s1, [x9, x8, lsl #2]
    ldr     s2, [x10, x8, lsl #2]
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x8, x8, #1
    b       3b
4:
    mul     x11, x6, x4
    add     x11, x11, x7
    str     s0, [x2, x11, lsl #2]
    add     x7, x7, #1
    b       2b
5:
    add     x6, x6, #1
    b       1b
8:
    ret

// Baseline direct bf16 checkpoint-layout GEMM. This leaf deliberately uses no
// FEAT_BF16 opcode: each halfword is unlifted into an f32 register and weights
// are never copied, packed, or widened into memory.
// void lfm_bf16_gemm_nt_f32_scalar(const uint16_t *a, const uint16_t *w,
//                                  float *c, int m, int n, int k)
.p2align 2
.globl LFM_SYM(lfm_bf16_gemm_nt_f32_scalar)
LFM_PRIVATE(LFM_SYM(lfm_bf16_gemm_nt_f32_scalar))
LFM_SYM(lfm_bf16_gemm_nt_f32_scalar):
    mov     x6, #0
1:
    cmp     w6, w3
    b.hs    8f
    mov     x7, #0
2:
    cmp     w7, w4
    b.hs    5f
    fmov    s0, wzr
    mul     x9, x6, x5
    add     x9, x0, x9, lsl #1
    mul     x10, x7, x5
    add     x10, x1, x10, lsl #1
    mov     x8, #0
3:
    cmp     w8, w5
    b.hs    4f
    ldrh    w11, [x9, x8, lsl #1]
    BF16_WIDEN w11, s1
    ldrh    w12, [x10, x8, lsl #1]
    BF16_WIDEN w12, s2
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x8, x8, #1
    b       3b
4:
    mul     x11, x6, x4
    add     x11, x11, x7
    str     s0, [x2, x11, lsl #2]
    add     x7, x7, #1
    b       2b
5:
    add     x6, x6, #1
    b       1b
8:
    ret

// void lfm_dwconv_tap_f32(const float *xpad, const float *w, float *y,
//                         uint64_t channels, uint64_t t_out, uint64_t t_pad,
//                         uint64_t k, uint64_t stride)
.p2align 2
.globl LFM_SYM(lfm_dwconv_tap_f32)
LFM_PRIVATE(LFM_SYM(lfm_dwconv_tap_f32))
LFM_SYM(lfm_dwconv_tap_f32):
    mov     x8, #0              // channel
1:
    cmp     x8, x3
    b.hs    8f
    mul     x9, x8, x5
    add     x9, x0, x9, lsl #2  // xpad row
    mul     x10, x8, x6
    add     x10, x1, x10, lsl #2 // w row
    mul     x11, x8, x4
    add     x11, x2, x11, lsl #2 // y row
    mov     x12, #0             // t
2:
    cmp     x12, x4
    b.hs    5f
    fmov    s0, wzr
    mul     x13, x12, x7        // t*stride
    add     x13, x9, x13, lsl #2
    mov     x14, #0             // j
3:
    cmp     x14, x6
    b.hs    4f
    ldr     s1, [x13, x14, lsl #2]
    ldr     s2, [x10, x14, lsl #2]
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x14, x14, #1
    b       3b
4:
    str     s0, [x11, x12, lsl #2]
    add     x12, x12, #1
    b       2b
5:
    add     x8, x8, #1
    b       1b
8:
    ret

// Direct-bf16 twin of lfm_dwconv_tap_f32. Activations and checkpoint taps are
// unlifted only in registers; the f32 destination preserves the reference
// bias/round boundary in the caller.
.p2align 2
.globl LFM_SYM(lfm_dwconv_tap_bf16_f32)
LFM_PRIVATE(LFM_SYM(lfm_dwconv_tap_bf16_f32))
LFM_SYM(lfm_dwconv_tap_bf16_f32):
    mov     x8, #0              // channel
1:
    cmp     x8, x3
    b.hs    8f
    mul     x9, x8, x5
    add     x9, x0, x9, lsl #1  // bf16 xpad row
    mul     x10, x8, x6
    add     x10, x1, x10, lsl #1 // bf16 w row
    mul     x11, x8, x4
    add     x11, x2, x11, lsl #2 // f32 y row
    mov     x12, #0
2:
    cmp     x12, x4
    b.hs    5f
    fmov    s0, wzr
    mul     x13, x12, x7
    add     x13, x9, x13, lsl #1
    mov     x14, #0
3:
    cmp     x14, x6
    b.hs    4f
    ldrh    w15, [x13, x14, lsl #1]
    BF16_WIDEN w15, s1
    ldrh    w16, [x10, x14, lsl #1]
    BF16_WIDEN w16, s2
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x14, x14, #1
    b       3b
4:
    str     s0, [x11, x12, lsl #2]
    add     x12, x12, #1
    b       2b
5:
    add     x8, x8, #1
    b       1b
8:
    ret

// void lfm_dwconv2d_k3s2_f32(const float *xpad, const float *w, float *y,
//                            uint64_t channels, uint64_t h_pad, uint64_t w_pad,
//                            uint64_t h_out /*x6*/, uint64_t w_out /*x7*/)
// Per-channel 3x3, stride 2, on a pre-padded plane.
.p2align 2
.globl LFM_SYM(lfm_dwconv2d_k3s2_f32)
LFM_PRIVATE(LFM_SYM(lfm_dwconv2d_k3s2_f32))
LFM_SYM(lfm_dwconv2d_k3s2_f32):
    stp     x25, x26, [sp, #-32]!
    stp     x27, x28, [sp, #16]
    mov     x8, #0              // channel
1:
    cmp     x8, x3
    b.hs    8f
    mul     x9, x4, x5
    mul     x9, x9, x8
    add     x9, x0, x9, lsl #2  // xpad plane
    mov     x10, #9
    mul     x10, x10, x8
    add     x10, x1, x10, lsl #2 // w (9)
    mul     x11, x6, x7
    mul     x11, x11, x8
    add     x11, x2, x11, lsl #2 // y plane
    mov     x12, #0             // oy
2:
    cmp     x12, x6
    b.hs    7f
    mov     x13, #0             // ox
3:
    cmp     x13, x7
    b.hs    6f
    fmov    s0, wzr
    mov     x14, #0             // dy
4:
    cmp     x14, #3
    b.hs    5f
    lsl     x25, x12, #1        // 2*oy
    add     x25, x25, x14       // 2*oy + dy
    mul     x25, x25, x5        // row offset
    lsl     x26, x13, #1        // 2*ox
    add     x25, x25, x26       // + 2*ox (dx added in inner)
    add     x25, x9, x25, lsl #2
    mov     x27, #3
    mul     x28, x14, x27       // dy*3
    mov     x26, #0             // dx
45:
    cmp     x26, #3
    b.hs    46f
    ldr     s1, [x25, x26, lsl #2]
    add     x27, x28, x26
    ldr     s2, [x10, x27, lsl #2]
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x26, x26, #1
    b       45b
46:
    add     x14, x14, #1
    b       4b
5:
    mul     x14, x12, x7
    add     x14, x14, x13
    str     s0, [x11, x14, lsl #2]
    add     x13, x13, #1
    b       3b
6:
    add     x12, x12, #1
    b       2b
7:
    add     x8, x8, #1
    b       1b
8:
    ldp     x27, x28, [sp, #16]
    ldp     x25, x26, [sp], #32
    ret

// Stem k3/s2: one padded bf16 input plane, resident (channels,1,3,3) bf16
// taps, and channel-major f32 output. Both bf16 operands are unlifted at the
// MAC; the checkpoint address may be byte-unaligned.
.p2align 2
.globl LFM_SYM(lfm_conv2d_stem_k3s2_bf16_f32)
LFM_PRIVATE(LFM_SYM(lfm_conv2d_stem_k3s2_bf16_f32))
LFM_SYM(lfm_conv2d_stem_k3s2_bf16_f32):
    stp     x25, x26, [sp, #-32]!
    stp     x27, x28, [sp, #16]
    mov     x8, #0
1:
    cmp     x8, x3
    b.hs    8f
    mov     x9, x0               // the stem has one shared input plane
    mov     x10, #9
    mul     x10, x10, x8
    add     x10, x1, x10, lsl #1 // resident bf16 taps
    mul     x11, x6, x7
    mul     x11, x11, x8
    add     x11, x2, x11, lsl #2 // f32 output plane
    mov     x12, #0
2:
    cmp     x12, x6
    b.hs    7f
    mov     x13, #0
3:
    cmp     x13, x7
    b.hs    6f
    fmov    s0, wzr
    mov     x14, #0
4:
    cmp     x14, #3
    b.hs    5f
    lsl     x25, x12, #1
    add     x25, x25, x14
    mul     x25, x25, x5
    lsl     x26, x13, #1
    add     x25, x25, x26
    add     x25, x9, x25, lsl #1
    mov     x27, #3
    mul     x28, x14, x27
    mov     x26, #0
45:
    cmp     x26, #3
    b.hs    46f
    ldrh    w15, [x25, x26, lsl #1]
    BF16_WIDEN w15, s1
    add     x27, x28, x26
    ldrh    w16, [x10, x27, lsl #1]
    BF16_WIDEN w16, s2
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x26, x26, #1
    b       45b
46:
    add     x14, x14, #1
    b       4b
5:
    mul     x14, x12, x7
    add     x14, x14, x13
    str     s0, [x11, x14, lsl #2]
    add     x13, x13, #1
    b       3b
6:
    add     x12, x12, #1
    b       2b
7:
    add     x8, x8, #1
    b       1b
8:
    ldp     x27, x28, [sp, #16]
    ldp     x25, x26, [sp], #32
    ret

// Direct-bf16 per-channel k3/s2 twin. The padded activation plane and the
// resident checkpoint taps stay bf16; each value is unlifted at the MAC.
.p2align 2
.globl LFM_SYM(lfm_dwconv2d_k3s2_bf16_f32)
LFM_PRIVATE(LFM_SYM(lfm_dwconv2d_k3s2_bf16_f32))
LFM_SYM(lfm_dwconv2d_k3s2_bf16_f32):
    stp     x25, x26, [sp, #-32]!
    stp     x27, x28, [sp, #16]
    mov     x8, #0
1:
    cmp     x8, x3
    b.hs    8f
    mul     x9, x4, x5
    mul     x9, x9, x8
    add     x9, x0, x9, lsl #1  // bf16 xpad plane
    mov     x10, #9
    mul     x10, x10, x8
    add     x10, x1, x10, lsl #1 // bf16 taps
    mul     x11, x6, x7
    mul     x11, x11, x8
    add     x11, x2, x11, lsl #2 // f32 output plane
    mov     x12, #0
2:
    cmp     x12, x6
    b.hs    7f
    mov     x13, #0
3:
    cmp     x13, x7
    b.hs    6f
    fmov    s0, wzr
    mov     x14, #0
4:
    cmp     x14, #3
    b.hs    5f
    lsl     x25, x12, #1
    add     x25, x25, x14
    mul     x25, x25, x5
    lsl     x26, x13, #1
    add     x25, x25, x26
    add     x25, x9, x25, lsl #1
    mov     x27, #3
    mul     x28, x14, x27
    mov     x26, #0
45:
    cmp     x26, #3
    b.hs    46f
    ldrh    w15, [x25, x26, lsl #1]
    BF16_WIDEN w15, s1
    add     x27, x28, x26
    ldrh    w16, [x10, x27, lsl #1]
    BF16_WIDEN w16, s2
    fmul    s1, s1, s2
    fadd    s0, s0, s1
    add     x26, x26, #1
    b       45b
46:
    add     x14, x14, #1
    b       4b
5:
    mul     x14, x12, x7
    add     x14, x14, x13
    str     s0, [x11, x14, lsl #2]
    add     x13, x13, #1
    b       3b
6:
    add     x12, x12, #1
    b       2b
7:
    add     x8, x8, #1
    b       1b
8:
    ldp     x27, x28, [sp, #16]
    ldp     x25, x26, [sp], #32
    ret

// void lfm_bias_rows_f32(float *y, const uint16_t *bias, uint64_t rows, uint64_t n)
// y[r][i] += widen(bias[i]) — the f32 bias add of the linear contract.
.p2align 2
.globl LFM_SYM(lfm_bias_rows_f32)
LFM_PRIVATE(LFM_SYM(lfm_bias_rows_f32))
LFM_SYM(lfm_bias_rows_f32):
    cbz     x1, 8f
    mov     x4, #0              // row
1:
    cmp     x4, x2
    b.hs    8f
    mov     x5, #0
2:
    cmp     x5, x3
    b.hs    3f
    ldrh    w6, [x1, x5, lsl #1]
    BF16_WIDEN w6, s1
    ldr     s0, [x0]
    fadd    s0, s0, s1
    str     s0, [x0], #4
    add     x5, x5, #1
    b       2b
3:
    add     x4, x4, #1
    b       1b
8:
    ret

// void lfm_bias_bcast_f32(float *y, const uint16_t *bias, uint64_t channels,
//                         uint64_t t) — y[c][*] += widen(bias[c]).
.p2align 2
.globl LFM_SYM(lfm_bias_bcast_f32)
LFM_PRIVATE(LFM_SYM(lfm_bias_bcast_f32))
LFM_SYM(lfm_bias_bcast_f32):
    cbz     x1, 8f
    mov     x4, #0
1:
    cmp     x4, x2
    b.hs    8f
    ldrh    w6, [x1, x4, lsl #1]
    BF16_WIDEN w6, s1
    mov     x5, #0
2:
    cmp     x5, x3
    b.hs    3f
    ldr     s0, [x0]
    fadd    s0, s0, s1
    str     s0, [x0], #4
    add     x5, x5, #1
    b       2b
3:
    add     x4, x4, #1
    b       1b
8:
    ret

// void lfm_add_scale_f32(float *acc, const float *addend, uint64_t rows,
//                        uint64_t cols, uint64_t addend_stride, float scale /*s0*/)
// acc[r][c] = (acc[r][c] + addend[r*stride + c]) * scale — f32 (the score
// combine (ac + bd[:, :T]) / sqrt(dk)).
.p2align 2
.globl LFM_SYM(lfm_add_scale_f32)
LFM_PRIVATE(LFM_SYM(lfm_add_scale_f32))
LFM_SYM(lfm_add_scale_f32):
    mov     x5, #0              // row
1:
    cmp     x5, x2
    b.hs    8f
    mul     x6, x5, x4
    add     x6, x1, x6, lsl #2  // addend row
    mov     x7, #0
2:
    cmp     x7, x3
    b.hs    3f
    ldr     s1, [x0]
    ldr     s2, [x6, x7, lsl #2]
    fadd    s1, s1, s2
    fmul    s1, s1, s0
    str     s1, [x0], #4
    add     x7, x7, #1
    b       2b
3:
    add     x5, x5, #1
    b       1b
8:
    ret

// void lfm_add_bias_hd_bf16(const uint16_t *x, const uint16_t *bias,
//                           uint16_t *y, uint64_t t, uint64_t heads, uint64_t dk)
// y[t][h*dk+d] = bf16(x + bias[h][d]) — one round (bf16 broadcast_add).
.p2align 2
.globl LFM_SYM(lfm_add_bias_hd_bf16)
LFM_PRIVATE(LFM_SYM(lfm_add_bias_hd_bf16))
LFM_SYM(lfm_add_bias_hd_bf16):
    mov     w14, #0x7fff
    mul     x6, x4, x5          // D = heads*dk
    mov     x7, #0              // t
1:
    cmp     x7, x3
    b.hs    8f
    mov     x8, #0              // d over D
2:
    cmp     x8, x6
    b.hs    3f
    ldrh    w9, [x0], #2
    BF16_WIDEN w9, s0
    ldrh    w10, [x1, x8, lsl #1]  // bias is (heads*dk) contiguous
    BF16_WIDEN w10, s1
    fadd    s0, s0, s1
    BF16_RNE s0, w9, w10, w14
    strh    w9, [x2], #2
    add     x8, x8, #1
    b       2b
3:
    add     x7, x7, #1
    b       1b
8:
    ret

// int lfm_pe_build_bf16(const float *div, uint64_t half, uint64_t t, uint16_t *out)
// Rel-pos table (2t-1, 2*half): positions (t-1-i); pe[i][2j]=sin(pos*div[j]),
// pe[i][2j+1]=cos(...), f32 angles (libm sinf/cosf), stored bf16.
.p2align 2
.globl LFM_SYM(lfm_pe_build_bf16)
LFM_PRIVATE(LFM_SYM(lfm_pe_build_bf16))
LFM_SYM(lfm_pe_build_bf16):
    cbz     x0, 9f
    cbz     x1, 9f
    cbz     x2, 9f
    cbz     x3, 9f
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8, d9, [sp, #64]
    str     x25, [sp, #80]
    mov     x19, x0             // div
    mov     x20, x1             // half
    mov     x21, x2             // t
    mov     x22, x3             // out
    lsl     x23, x21, #1
    sub     x23, x23, #1        // P = 2t-1
    mov     x24, #0             // i
1:
    cmp     x24, x23
    b.hs    7f
    // pos = (float)(t - 1 - i), signed
    sub     x2, x21, #1
    sub     x2, x2, x24
    scvtf   s8, x2
    mov     x25, #0             // j
2:
    cmp     x25, x20
    b.hs    6f
    ldr     s0, [x19, x25, lsl #2]
    fmul    s9, s8, s0          // angle
    fmov    s0, s9
    bl      LFM_CALL(sinf)
    mov     w14, #0x7fff
    BF16_RNE s0, w0, w1, w14
    lsl     x3, x20, #1         // D = 2*half
    mul     x4, x24, x3
    lsl     x5, x25, #1
    add     x4, x4, x5
    strh    w0, [x22, x4, lsl #1]
    fmov    s0, s9
    bl      LFM_CALL(cosf)
    mov     w14, #0x7fff
    BF16_RNE s0, w0, w1, w14
    lsl     x3, x20, #1
    mul     x4, x24, x3
    lsl     x5, x25, #1
    add     x4, x4, x5
    add     x4, x4, #1
    strh    w0, [x22, x4, lsl #1]
    add     x25, x25, #1
    b       2b
6:
    add     x24, x24, #1
    b       1b
7:
    mov     w0, #0
    ldr     x25, [sp, #80]
    ldp     d8, d9, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret
9:
    mov     w0, #-22
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
