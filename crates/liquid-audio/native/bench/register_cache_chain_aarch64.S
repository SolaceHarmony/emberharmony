// register_cache_chain_aarch64.S -- storage-boundary controls for
// register_cache_chain.cpp.
//
// All six entry points execute the same four-stage numerical chain:
//
//   RMS normalization -> multiplicative gate -> causal ShortConv-3 ->
//   four-row projection
//
// `register_cache_chain_fused` keeps all stage values in caller-clobbered
// v0-v7/v16-v31 and writes
// only the four terminal projection values. It has no stack frame, calls, or
// intermediate stores. `register_cache_chain_stack` intentionally writes and
// reloads three 64-byte planes in a 192-byte stack frame.
// `register_cache_chain_arena` does the same through caller-owned scratch. The
// two batch leaves contrast per-tile register fusion with delayed, full-plane
// reuse over an identical terminal-output span.
//
// This is a standalone probe, not a product kernel.

#ifdef __APPLE__
#define CHAIN_SYM(name) _##name
#else
#define CHAIN_SYM(name) name
#endif

.text

// x0: input[16], x3: Params
// result: normalized input in v0-v3
.macro CHAIN_NORM
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]

    fmul    v30.4s, v0.4s, v0.4s
    fmla    v30.4s, v1.4s, v1.4s
    fmla    v30.4s, v2.4s, v2.4s
    fmla    v30.4s, v3.4s, v3.4s
    faddp   v30.4s, v30.4s, v30.4s
    faddp   s30, v30.2s
    ldr     s31, [x3, #4]
    fmul    s30, s30, s31
    ldr     s31, [x3]
    fadd    s30, s30, s31
    fsqrt   s30, s30
    ldr     s31, [x3, #8]
    fdiv    s30, s31, s30
    dup     v30.4s, v30.s[0]

    fmul    v0.4s, v0.4s, v30.4s
    fmul    v1.4s, v1.4s, v30.4s
    fmul    v2.4s, v2.4s, v30.4s
    fmul    v3.4s, v3.4s, v30.4s
.endm

// x1: gate[16]; normalized input in v0-v3
// result: gated input in v0-v3
.macro CHAIN_GATE
    ldp     q4, q5, [x1]
    ldp     q6, q7, [x1, #32]
    fmul    v0.4s, v0.4s, v4.4s
    fmul    v1.4s, v1.4s, v5.4s
    fmul    v2.4s, v2.4s, v6.4s
    fmul    v3.4s, v3.4s, v7.4s
.endm

// Load the three ShortConv coefficients and construct a prefix register
// [0, 0, carry[-2], carry[-1]].
.macro CHAIN_CONV_SETUP
    ldr     s17, [x3, #12]
    ldr     s18, [x3, #16]
    ldr     s19, [x3, #20]
    dup     v17.4s, v17.s[0]
    dup     v18.4s, v18.s[0]
    dup     v19.4s, v19.s[0]
    movi    v16.4s, #0
    add     x9, x3, #24
    ld1     {v16.2s}, [x9]
    ext     v16.16b, v16.16b, v16.16b, #8
.endm

// Compute one four-value ShortConv block in v12. `cur` is its source,
// `left` supplies the preceding bytes, and offsets 8/12 select x[-2]/x[-1].
.macro CHAIN_CONV_BLOCK cur, left
    ext     v28.16b, \left\().16b, \cur\().16b, #8
    ext     v29.16b, \left\().16b, \cur\().16b, #12
    fmul    v30.4s, \cur\().4s, v17.4s
    fmla    v30.4s, v29.4s, v18.4s
    fmla    v30.4s, v28.4s, v19.4s
.endm

// Four vector accumulators hold one projection row apiece. Each accumulator
// lane carries positions {lane, lane+4, lane+8, lane+12} until final faddv.
.macro CHAIN_PROJ_ZERO
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
.endm

.macro CHAIN_PROJ_BLOCK offset0, offset1, offset2, offset3
    ldr     q24, [x2, #\offset0]
    ldr     q25, [x2, #\offset1]
    ldr     q26, [x2, #\offset2]
    ldr     q27, [x2, #\offset3]
    fmla    v20.4s, v24.4s, v30.4s
    fmla    v21.4s, v25.4s, v30.4s
    fmla    v22.4s, v26.4s, v30.4s
    fmla    v23.4s, v27.4s, v30.4s
.endm

.macro CHAIN_PROJ_STORE out
    faddp   v20.4s, v20.4s, v20.4s
    faddp   v21.4s, v21.4s, v21.4s
    faddp   v22.4s, v22.4s, v22.4s
    faddp   v23.4s, v23.4s, v23.4s
    faddp   s20, v20.2s
    faddp   s21, v21.2s
    faddp   s22, v22.2s
    faddp   s23, v23.2s
    ins     v20.s[1], v21.s[0]
    ins     v20.s[2], v22.s[0]
    ins     v20.s[3], v23.s[0]
    str     q20, [\out]
.endm

.p2align 4
.globl CHAIN_SYM(register_cache_chain_fused)
CHAIN_SYM(register_cache_chain_fused):
    CHAIN_NORM
    CHAIN_GATE
    CHAIN_CONV_SETUP
    CHAIN_PROJ_ZERO

    CHAIN_CONV_BLOCK v0, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v1, v0
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v2, v1
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v3, v2
    CHAIN_PROJ_BLOCK 48, 112, 176, 240

    CHAIN_PROJ_STORE x4
    ret

// Shared macro for the two deliberately materialized controls.
// x4: scratch[48], x5: out[4]
.macro CHAIN_MATERIALIZED_BODY
    CHAIN_NORM
    stp     q0, q1, [x4]
    stp     q2, q3, [x4, #32]
    ldp     q0, q1, [x4]
    ldp     q2, q3, [x4, #32]

    CHAIN_GATE
    stp     q0, q1, [x4, #64]
    stp     q2, q3, [x4, #96]
    ldp     q0, q1, [x4, #64]
    ldp     q2, q3, [x4, #96]

    CHAIN_CONV_SETUP
    CHAIN_CONV_BLOCK v0, v16
    str     q30, [x4, #128]
    CHAIN_CONV_BLOCK v1, v0
    str     q30, [x4, #144]
    CHAIN_CONV_BLOCK v2, v1
    str     q30, [x4, #160]
    CHAIN_CONV_BLOCK v3, v2
    str     q30, [x4, #176]

    CHAIN_PROJ_ZERO
    ldr     q30, [x4, #128]
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    ldr     q30, [x4, #144]
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    ldr     q30, [x4, #160]
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    ldr     q30, [x4, #176]
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x5
.endm

.p2align 4
.globl CHAIN_SYM(register_cache_chain_stack)
CHAIN_SYM(register_cache_chain_stack):
    mov     x5, x4
    sub     sp, sp, #192
    mov     x4, sp
    CHAIN_MATERIALIZED_BODY
    add     sp, sp, #192
    ret

.p2align 4
.globl CHAIN_SYM(register_cache_chain_arena)
CHAIN_SYM(register_cache_chain_arena):
    CHAIN_MATERIALIZED_BODY
    ret

// BatchDesc at x4:
//   +0  norm plane (count * 64 B)
//   +8  gate plane (count * 64 B)
//   +16 conv plane (count * 64 B)
//   +24 terminal output (count * 16 B)
//   +32 tile count
//
// Inputs, gates, weights, and parameters are deliberately fixed across tiles.
// This isolates intermediate-plane residency and reuse distance. Both batch
// functions execute the same arithmetic and write the same terminal span.

.p2align 4
.globl CHAIN_SYM(register_cache_chain_batch_fused)
CHAIN_SYM(register_cache_chain_batch_fused):
    ldr     x8, [x4, #24]
    ldr     x10, [x4, #32]
    cbz     x10, 2f
1:
    CHAIN_NORM
    CHAIN_GATE
    CHAIN_CONV_SETUP
    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v0, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v1, v0
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v2, v1
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v3, v2
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16
    subs    x10, x10, #1
    b.ne    1b
2:
    ret

// Four independent tiles share one register FIFO. Activation banks are
// v0-v3, v4-v7, v8-v11, and v12-v15. Their norm reductions and scalar
// fsqrt/fdiv dependency chains are interleaved before the tile banks advance
// through gate, ShortConv, and projection. The hot group loop writes terminal
// outputs only.
//
// AAPCS64 requires the low 64 bits of v8-v15 to survive a public call. Save and
// restore those 64 bytes once per whole batch (128 B total traffic), never per
// tile. Upper halves are caller-clobbered by the ABI.
.p2align 4
.globl CHAIN_SYM(register_cache_chain_batch_fifo4)
CHAIN_SYM(register_cache_chain_batch_fifo4):
    sub     sp, sp, #64
    stp     d8, d9, [sp]
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    ldr     x8, [x4, #24]
    ldr     x10, [x4, #32]
    and     x12, x10, #3
    lsr     x11, x10, #2
    cbz     x11, 2f
1:
    // Four independent activation banks. Fixed source data is intentional;
    // every bank still executes the complete arithmetic chain.
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ldp     q4, q5, [x0]
    ldp     q6, q7, [x0, #32]
    ldp     q8, q9, [x0]
    ldp     q10, q11, [x0, #32]
    ldp     q12, q13, [x0]
    ldp     q14, q15, [x0, #32]

    // Independent norm reductions expose four scalar dependency chains.
    fmul    v16.4s, v0.4s, v0.4s
    fmul    v17.4s, v4.4s, v4.4s
    fmul    v18.4s, v8.4s, v8.4s
    fmul    v19.4s, v12.4s, v12.4s
    fmla    v16.4s, v1.4s, v1.4s
    fmla    v17.4s, v5.4s, v5.4s
    fmla    v18.4s, v9.4s, v9.4s
    fmla    v19.4s, v13.4s, v13.4s
    fmla    v16.4s, v2.4s, v2.4s
    fmla    v17.4s, v6.4s, v6.4s
    fmla    v18.4s, v10.4s, v10.4s
    fmla    v19.4s, v14.4s, v14.4s
    fmla    v16.4s, v3.4s, v3.4s
    fmla    v17.4s, v7.4s, v7.4s
    fmla    v18.4s, v11.4s, v11.4s
    fmla    v19.4s, v15.4s, v15.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v17.4s, v17.4s, v17.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v19.4s, v19.4s, v19.4s
    faddp   s16, v16.2s
    faddp   s17, v17.2s
    faddp   s18, v18.2s
    faddp   s19, v19.2s
    ldr     s20, [x3, #4]
    fmul    s16, s16, s20
    fmul    s17, s17, s20
    fmul    s18, s18, s20
    fmul    s19, s19, s20
    ldr     s20, [x3]
    fadd    s16, s16, s20
    fadd    s17, s17, s20
    fadd    s18, s18, s20
    fadd    s19, s19, s20
    fsqrt   s16, s16
    fsqrt   s17, s17
    fsqrt   s18, s18
    fsqrt   s19, s19
    ldr     s20, [x3, #8]
    fdiv    s16, s20, s16
    fdiv    s17, s20, s17
    fdiv    s18, s20, s18
    fdiv    s19, s20, s19
    dup     v16.4s, v16.s[0]
    dup     v17.4s, v17.s[0]
    dup     v18.4s, v18.s[0]
    dup     v19.4s, v19.s[0]

    fmul    v0.4s, v0.4s, v16.4s
    fmul    v1.4s, v1.4s, v16.4s
    fmul    v2.4s, v2.4s, v16.4s
    fmul    v3.4s, v3.4s, v16.4s
    fmul    v4.4s, v4.4s, v17.4s
    fmul    v5.4s, v5.4s, v17.4s
    fmul    v6.4s, v6.4s, v17.4s
    fmul    v7.4s, v7.4s, v17.4s
    fmul    v8.4s, v8.4s, v18.4s
    fmul    v9.4s, v9.4s, v18.4s
    fmul    v10.4s, v10.4s, v18.4s
    fmul    v11.4s, v11.4s, v18.4s
    fmul    v12.4s, v12.4s, v19.4s
    fmul    v13.4s, v13.4s, v19.4s
    fmul    v14.4s, v14.4s, v19.4s
    fmul    v15.4s, v15.4s, v19.4s

    // Stream each gate vector through all four banks before reusing v20.
    ldr     q20, [x1]
    fmul    v0.4s, v0.4s, v20.4s
    fmul    v4.4s, v4.4s, v20.4s
    fmul    v8.4s, v8.4s, v20.4s
    fmul    v12.4s, v12.4s, v20.4s
    ldr     q20, [x1, #16]
    fmul    v1.4s, v1.4s, v20.4s
    fmul    v5.4s, v5.4s, v20.4s
    fmul    v9.4s, v9.4s, v20.4s
    fmul    v13.4s, v13.4s, v20.4s
    ldr     q20, [x1, #32]
    fmul    v2.4s, v2.4s, v20.4s
    fmul    v6.4s, v6.4s, v20.4s
    fmul    v10.4s, v10.4s, v20.4s
    fmul    v14.4s, v14.4s, v20.4s
    ldr     q20, [x1, #48]
    fmul    v3.4s, v3.4s, v20.4s
    fmul    v7.4s, v7.4s, v20.4s
    fmul    v11.4s, v11.4s, v20.4s
    fmul    v15.4s, v15.4s, v20.4s

    // Norm scales are dead. Rotate v16-v19 into prefix/coefficients and carry
    // each activation bank through ShortConv/projection without a spill.
    CHAIN_CONV_SETUP

    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v0, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v1, v0
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v2, v1
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v3, v2
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16

    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v4, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v5, v4
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v6, v5
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v7, v6
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16

    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v8, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v9, v8
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v10, v9
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v11, v10
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16

    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v12, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v13, v12
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v14, v13
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v15, v14
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16

    subs    x11, x11, #1
    b.ne    1b

    // Count tails execute the exact single-tile order, still without spills.
2:
    cbz     x12, 4f
3:
    CHAIN_NORM
    CHAIN_GATE
    CHAIN_CONV_SETUP
    CHAIN_PROJ_ZERO
    CHAIN_CONV_BLOCK v0, v16
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    CHAIN_CONV_BLOCK v1, v0
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    CHAIN_CONV_BLOCK v2, v1
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    CHAIN_CONV_BLOCK v3, v2
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x8, x8, #16
    subs    x12, x12, #1
    b.ne    3b
4:
    ldp     d8, d9, [sp]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    add     sp, sp, #64
    ret

.p2align 4
.globl CHAIN_SYM(register_cache_chain_batch_planes)
CHAIN_SYM(register_cache_chain_batch_planes):
    ldr     x5, [x4]
    ldr     x10, [x4, #32]
    cbz     x10, 8f

    // Phase 1: materialize normalized activations for every tile.
1:
    CHAIN_NORM
    stp     q0, q1, [x5]
    stp     q2, q3, [x5, #32]
    add     x5, x5, #64
    subs    x10, x10, #1
    b.ne    1b

    // Phase 2: revisit the complete norm plane and materialize the gate plane.
    ldr     x5, [x4]
    ldr     x6, [x4, #8]
    ldr     x10, [x4, #32]
2:
    ldp     q0, q1, [x5]
    ldp     q2, q3, [x5, #32]
    CHAIN_GATE
    stp     q0, q1, [x6]
    stp     q2, q3, [x6, #32]
    add     x5, x5, #64
    add     x6, x6, #64
    subs    x10, x10, #1
    b.ne    2b

    // Phase 3: revisit the gate plane and materialize ShortConv results.
    ldr     x6, [x4, #8]
    ldr     x7, [x4, #16]
    ldr     x10, [x4, #32]
3:
    ldp     q0, q1, [x6]
    ldp     q2, q3, [x6, #32]
    CHAIN_CONV_SETUP
    CHAIN_CONV_BLOCK v0, v16
    str     q30, [x7]
    CHAIN_CONV_BLOCK v1, v0
    str     q30, [x7, #16]
    CHAIN_CONV_BLOCK v2, v1
    str     q30, [x7, #32]
    CHAIN_CONV_BLOCK v3, v2
    str     q30, [x7, #48]
    add     x6, x6, #64
    add     x7, x7, #64
    subs    x10, x10, #1
    b.ne    3b

    // Phase 4: revisit the complete convolution plane and write only terminal
    // projection outputs. The projection reduction order matches the fused loop.
    ldr     x7, [x4, #16]
    ldr     x8, [x4, #24]
    ldr     x10, [x4, #32]
4:
    CHAIN_PROJ_ZERO
    ldr     q30, [x7]
    CHAIN_PROJ_BLOCK 0, 64, 128, 192
    ldr     q30, [x7, #16]
    CHAIN_PROJ_BLOCK 16, 80, 144, 208
    ldr     q30, [x7, #32]
    CHAIN_PROJ_BLOCK 32, 96, 160, 224
    ldr     q30, [x7, #48]
    CHAIN_PROJ_BLOCK 48, 112, 176, 240
    CHAIN_PROJ_STORE x8
    add     x7, x7, #64
    add     x8, x8, #16
    subs    x10, x10, #1
    b.ne    4b
8:
    ret

#if !defined(__APPLE__)
.section .note.GNU-stack,"",%progbits
#endif
