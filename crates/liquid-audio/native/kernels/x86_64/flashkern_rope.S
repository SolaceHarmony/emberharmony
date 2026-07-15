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

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif
