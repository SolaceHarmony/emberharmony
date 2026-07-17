#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text
.p2align 4
.globl LFM_SYM(lfm_rope_table_f32)
LFM_SYM(lfm_rope_table_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r9, %rax
    movq    40(%rsp), %rdx
    movaps  %xmm2, %xmm0
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
    movq    %rcx, %rdx
#endif
    testq   %r10, %r10
    jz      4f
    testq   %r11, %r11
    jz      4f
    testq   $1, %r11
    jnz     4f
    testq   %rax, %rax
    jz      4f
    testq   %rdx, %rdx
    jz      4f
    xorps   %xmm1, %xmm1
    ucomiss %xmm1, %xmm0
    jbe     4f

    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
#ifdef _WIN32
    subq    $72, %rsp
    movss   %xmm0, 32(%rsp)
#define LFM_THETA 32
#define LFM_ANGLE 36
#else
    subq    $40, %rsp
    movss   %xmm0, 0(%rsp)
#define LFM_THETA 0
#define LFM_ANGLE 4
#endif
    movq    %r10, %r12
    movq    %r11, %r13
    movq    %rax, %r14
    movq    %rdx, %r15
    xorl    %ebx, %ebx

1:
    xorl    %ebp, %ebp
2:
    pxor    %xmm1, %xmm1
    cvtsi2ssq %rbp, %xmm1
    addss   %xmm1, %xmm1
    pxor    %xmm2, %xmm2
    cvtsi2ssq %r13, %xmm2
    divss   %xmm2, %xmm1
    movss   LFM_THETA(%rsp), %xmm0
    call    LFM_CALL(powf)

    pxor    %xmm1, %xmm1
    cvtsi2ssq %rbx, %xmm1
    divss   %xmm0, %xmm1
    movss   %xmm1, LFM_ANGLE(%rsp)
    movaps  %xmm1, %xmm0
    call    LFM_CALL(cosf)
    movss   %xmm0, (%r14)
    movss   LFM_ANGLE(%rsp), %xmm0
    call    LFM_CALL(sinf)
    movss   %xmm0, (%r15)

    addq    $4, %r14
    addq    $4, %r15
    incq    %rbp
    movq    %r13, %rax
    shrq    $1, %rax
    cmpq    %rax, %rbp
    jb      2b
    incq    %rbx
    cmpq    %r12, %rbx
    jb      1b

    xorl    %eax, %eax
#ifdef _WIN32
    addq    $72, %rsp
#else
    addq    $40, %rsp
#endif
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret

4:
    movl    $-22, %eax
    ret

.p2align 4
.globl LFM_SYM(lfm_rope_range_f32)
LFM_SYM(lfm_rope_range_f32):
#ifdef _WIN32
    movq    40(%rsp), %r10
    movq    48(%rsp), %r11
    movd    %xmm3, %eax
#endif
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
#ifdef _WIN32
    subq    $56, %rsp
    movq    %rcx, %r12
    movq    %rdx, %r13
    movq    %r8, %r14
    movq    %r10, %r15
    movq    %r11, %rbx
    movl    %eax, 32(%rsp)
#define LFM_RANGE_THETA 32
#define LFM_RANGE_ANGLE 36
#define LFM_RANGE_DIM 40
#else
    subq    $24, %rsp
    movq    %rdi, %r12
    movq    %rsi, %r13
    movq    %rdx, %r14
    movq    %rcx, %r15
    movq    %r8, %rbx
    movss   %xmm0, 0(%rsp)
#define LFM_RANGE_THETA 0
#define LFM_RANGE_ANGLE 4
#define LFM_RANGE_DIM 8
#endif
    testq   %r13, %r13
    jz      8f
    testq   %r14, %r14
    jz      8f
    testq   $1, %r14
    jnz     8f
    testq   %r15, %r15
    jz      8f
    testq   %rbx, %rbx
    jz      8f
    xorps   %xmm1, %xmm1
    movss   LFM_RANGE_THETA(%rsp), %xmm0
    ucomiss %xmm1, %xmm0
    jbe     8f
    movq    %r13, %rax
    decq    %rax
    addq    %r12, %rax
    jc      9f
    testq   %rax, %rax
    js      9f

    xorl    %ebp, %ebp
5:
    movq    $0, LFM_RANGE_DIM(%rsp)
6:
    pxor    %xmm1, %xmm1
    cvtsi2ssq LFM_RANGE_DIM(%rsp), %xmm1
    addss   %xmm1, %xmm1
    pxor    %xmm2, %xmm2
    cvtsi2ssq %r14, %xmm2
    divss   %xmm2, %xmm1
    movss   LFM_RANGE_THETA(%rsp), %xmm0
    call    LFM_CALL(powf)

    leaq    (%r12,%rbp), %rax
    pxor    %xmm1, %xmm1
    cvtsi2ssq %rax, %xmm1
    divss   %xmm0, %xmm1
    movss   %xmm1, LFM_RANGE_ANGLE(%rsp)
    movaps  %xmm1, %xmm0
    call    LFM_CALL(cosf)
    movss   %xmm0, (%r15)
    movss   LFM_RANGE_ANGLE(%rsp), %xmm0
    call    LFM_CALL(sinf)
    movss   %xmm0, (%rbx)

    addq    $4, %r15
    addq    $4, %rbx
    incq    LFM_RANGE_DIM(%rsp)
    movq    %r14, %rax
    shrq    $1, %rax
    cmpq    %rax, LFM_RANGE_DIM(%rsp)
    jb      6b
    incq    %rbp
    cmpq    %r13, %rbp
    jb      5b
    xorl    %eax, %eax
    jmp     10f

8:
    movl    $-22, %eax
    jmp     10f
9:
    movl    $-75, %eax
10:
#ifdef _WIN32
    addq    $56, %rsp
#else
    addq    $24, %rsp
#endif
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret

#undef LFM_RANGE_THETA
#undef LFM_RANGE_ANGLE
#undef LFM_RANGE_DIM

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif
