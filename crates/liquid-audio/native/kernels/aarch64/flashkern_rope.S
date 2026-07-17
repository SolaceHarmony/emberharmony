#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text
.p2align 2
.globl LFM_SYM(lfm_rope_table_f32)
LFM_PRIVATE(LFM_SYM(lfm_rope_table_f32))
LFM_SYM(lfm_rope_table_f32):
    cbz     x0, 3f
    cbz     x1, 3f
    tbnz    x1, #0, 3f
    cbz     x2, 3f
    cbz     x3, 3f
    fcmp    s0, #0.0
    b.le    3f
    b.vs    3f

    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     d8, d9, [sp, #64]
    str     x25, [sp, #80]

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    lsr     x24, x20, #1
    fmov    s8, s0
    mov     x23, #0

1:
    mov     x25, #0
2:
    ucvtf   s1, x25
    fadd    s1, s1, s1
    ucvtf   s2, x20
    fdiv    s1, s1, s2
    fmov    s0, s8
    bl      LFM_CALL(powf)

    ucvtf   s9, x23
    fdiv    s9, s9, s0
    fmov    s0, s9
    bl      LFM_CALL(cosf)
    str     s0, [x21], #4
    fmov    s0, s9
    bl      LFM_CALL(sinf)
    str     s0, [x22], #4

    add     x25, x25, #1
    cmp     x25, x24
    b.lo    2b
    add     x23, x23, #1
    cmp     x23, x19
    b.lo    1b

    mov     w0, #0
    ldr     x25, [sp, #80]
    ldp     d8, d9, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

3:
    mov     w0, #22
    neg     w0, w0
    ret

.p2align 2
.globl LFM_SYM(lfm_rope_range_f32)
LFM_PRIVATE(LFM_SYM(lfm_rope_range_f32))
LFM_SYM(lfm_rope_range_f32):
    cbz     x1, 6f
    cbz     x2, 6f
    tbnz    x2, #0, 6f
    cbz     x3, 6f
    cbz     x4, 6f
    fcmp    s0, #0.0
    b.le    6f
    b.vs    6f
    sub     x5, x1, #1
    adds    x5, x0, x5
    b.cs    7f
    tbnz    x5, #63, 7f

    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     d8, d9, [sp, #80]

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    mov     x23, x4
    lsr     x24, x21, #1
    fmov    s8, s0
    mov     x25, #0

4:
    mov     x26, #0
5:
    ucvtf   s1, x26
    fadd    s1, s1, s1
    ucvtf   s2, x21
    fdiv    s1, s1, s2
    fmov    s0, s8
    bl      LFM_CALL(powf)

    add     x5, x19, x25
    ucvtf   s9, x5
    fdiv    s9, s9, s0
    fmov    s0, s9
    bl      LFM_CALL(cosf)
    str     s0, [x22], #4
    fmov    s0, s9
    bl      LFM_CALL(sinf)
    str     s0, [x23], #4

    add     x26, x26, #1
    cmp     x26, x24
    b.lo    5b
    add     x25, x25, #1
    cmp     x25, x20
    b.lo    4b

    mov     w0, #0
    ldp     d8, d9, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #112
    ret

6:
    mov     w0, #22
    neg     w0, w0
    ret
7:
    mov     w0, #75
    neg     w0, w0
    ret

#if !defined(__APPLE__)
.section .note.GNU-stack,"",%progbits
#endif
