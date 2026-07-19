#ifdef __APPLE__
#define LFM_BENCH_SYM(name) _##name
#else
#define LFM_BENCH_SYM(name) name
#endif

// Faithful fused form of Flashkern ST_DOWN for one decode row:
//
//   f32 dot = bf16 input[K] * bf16 weight[N,K]^T
//   bf16 projected = RNE(dot)
//   bf16 output = RNE(f32(projected) + f32(residual))
//
// ABI:
//   x0 input bytes, x1 checkpoint weight bytes, x2 residual bytes,
//   x3 output bytes, x4 N, x5 K.
//
// Every source and destination is byte addressed. A row may begin at any byte
// address; no C++ uint16_t lvalue is manufactured for checkpoint storage. The
// vector loop reproduces gemm_nt_impl's two F32x4 accumulators and its
// vaddvq_f32(vaddq_f32(acc0, acc1)) reduction. The scalar tail reproduces its
// post-reduction fmaf tail. The leaf has no frame, calls, or intermediate
// stores: only the terminal STRH publishes a value.

.text
.p2align 2
.globl LFM_BENCH_SYM(lfm_down_projection_chain_fused_aarch64)
LFM_BENCH_SYM(lfm_down_projection_chain_fused_aarch64):
    cbz     x4, 9f
    lsl     x13, x5, #1
    mov     w15, #0x7fff

1:  // One checkpoint row / one terminal output.
    mov     x6, x0
    mov     x7, x1
    lsr     x8, x5, #3
    and     x14, x5, #7
    prfm    pldl1strm, [x1, x13]
    movi    v0.4s, #0
    movi    v1.4s, #0

    cmp     x8, #2
    b.lo    2f
6:  // Two blocks amortize the branch and expose the next loads to the OoO core.
    ldr     q2, [x6], #16
    ldr     q3, [x7], #16
    ldr     q16, [x6], #16
    ldr     q17, [x7], #16
    shll    v4.4s, v2.4h, #16
    shll2   v5.4s, v2.8h, #16
    shll    v6.4s, v3.4h, #16
    shll2   v7.4s, v3.8h, #16
    fmla    v0.4s, v4.4s, v6.4s
    fmla    v1.4s, v5.4s, v7.4s
    shll    v18.4s, v16.4h, #16
    shll2   v19.4s, v16.8h, #16
    shll    v20.4s, v17.4h, #16
    shll2   v21.4s, v17.8h, #16
    fmla    v0.4s, v18.4s, v20.4s
    fmla    v1.4s, v19.4s, v21.4s
    sub     x8, x8, #2
    cmp     x8, #2
    b.hs    6b

2:  // A possible final eight-product block preserves the same order.
    cbz     x8, 3f
    ldr     q2, [x6], #16
    ldr     q3, [x7], #16
    shll    v4.4s, v2.4h, #16
    shll2   v5.4s, v2.8h, #16
    shll    v6.4s, v3.4h, #16
    shll2   v7.4s, v3.8h, #16
    fmla    v0.4s, v4.4s, v6.4s
    fmla    v1.4s, v5.4s, v7.4s

3:  // vaddvq_f32(vaddq_f32(acc0, acc1)), in its pinned pairwise order.
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    cbz     x14, 5f

4:  // Production handles K%8 only after reducing the vector accumulators.
    ldrh    w9, [x6], #2
    ldrh    w10, [x7], #2
    lsl     w9, w9, #16
    lsl     w10, w10, #16
    fmov    s2, w9
    fmov    s3, w10
    fmadd   s0, s2, s3, s0
    subs    x14, x14, #1
    b.ne    4b

5:  // First RNE: materialize only in a GPR, then unlift in-register.
    fmov    w9, s0
    lsr     w10, w9, #16
    and     w10, w10, #1
    add     w9, w9, w15
    add     w9, w9, w10
    lsr     w11, w9, #16
    lsl     w10, w11, #16
    fmov    s0, w10

    // Residual add, followed by the chain's second RNE.
    ldrh    w12, [x2], #2
    lsl     w12, w12, #16
    fmov    s1, w12
    fadd    s0, s0, s1
    fmov    w9, s0
    lsr     w10, w9, #16
    and     w10, w10, #1
    add     w9, w9, w15
    add     w9, w9, w10
    lsr     w11, w9, #16
    strh    w11, [x3], #2

    mov     x1, x7
    subs    x4, x4, #1
    b.ne    1b

9:
    ret

// Four-output register FIFO. Each activation block is loaded and widened once,
// then consumed by four raw checkpoint weight rows. v0-v7 are the eight pinned
// production accumulators; v16-v30 are caller-clobbered load/widen temporaries.
// No v8-v15 callee-saved state is touched.
.p2align 2
.globl LFM_BENCH_SYM(lfm_down_projection_chain_fused4_aarch64)
LFM_BENCH_SYM(lfm_down_projection_chain_fused4_aarch64):
    cbz     x4, 99f
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

11: // One activation load and four independent checkpoint-row loads.
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

12: // Pin every row to acc-low + acc-high, then its pairwise horizontal sum.
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

13: // K%8 tail: load one activation and apply it to all four rows in order.
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

14: // Pack four dots, then run both RNE stages as one register-only vector epilogue.
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

    add     x1, x1, x6, lsl #2
    subs    x14, x14, #1
    b.ne    10b

20: // N%4 tail: faithful one-row body, still inline and final-store-only.
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
    mov     x1, x7
    subs    x4, x4, #1
    b.ne    21b

99:
    ret
