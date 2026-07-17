#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 4
.globl LFM_SYM(lfm_rsqrt_size)
LFM_PRIVATE(LFM_SYM(lfm_rsqrt_size))
LFM_SYM(lfm_rsqrt_size):
#ifdef _WIN32
    pxor      %xmm1, %xmm1
    cvtsi2ssq %rcx, %xmm1
#else
    pxor      %xmm1, %xmm1
    cvtsi2ssq %rdi, %xmm1
#endif
    sqrtss    %xmm1, %xmm1
    movl      $0x3f800000, %eax
    movd      %eax, %xmm0
    divss     %xmm1, %xmm0
    ret

.p2align 4
.globl LFM_SYM(lfm_inv_rms_f32)
LFM_PRIVATE(LFM_SYM(lfm_inv_rms_f32))
LFM_SYM(lfm_inv_rms_f32):
#ifdef _WIN32
    pxor      %xmm3, %xmm3
    cvtsi2ssq %rdx, %xmm3
    divss     %xmm3, %xmm0
    addss     %xmm2, %xmm0
#else
    pxor      %xmm2, %xmm2
    cvtsi2ssq %rdi, %xmm2
    divss     %xmm2, %xmm0
    addss     %xmm1, %xmm0
#endif
    sqrtss    %xmm0, %xmm0
    movl      $0x3f800000, %eax
    movd      %eax, %xmm1
    divss     %xmm0, %xmm1
    movaps    %xmm1, %xmm0
    ret

.p2align 4
.globl LFM_SYM(lfm_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_sum_f32))
LFM_SYM(lfm_sum_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    xorps     %xmm0, %xmm0
    testq     %r11, %r11
    jz        2f
1:
    addss     (%r10), %xmm0
    addq      $4, %r10
    decq      %r11
    jnz       1b
2:
    ret

.p2align 4
.globl LFM_SYM(lfm_bf16_sumsq_stride_f32)
LFM_PRIVATE(LFM_SYM(lfm_bf16_sumsq_stride_f32))
LFM_SYM(lfm_bf16_sumsq_stride_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rax
    movq      %r9, %rcx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
    movq      %rdx, %rax
#endif
    xorps     %xmm0, %xmm0
    cmpq      %r11, %rax
    jae       2f
1:
    movzwl    (%r10,%rax,2), %edx
    shll      $16, %edx
    movd      %edx, %xmm1
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    addq      %rcx, %rax
    cmpq      %r11, %rax
    jb        1b
2:
    ret

.p2align 4
.globl LFM_SYM(lfm_bf16_bias_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_bf16_bias_add_f32))
LFM_SYM(lfm_bf16_bias_add_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rax
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
    movq      %rdx, %rax
#endif
    testq     %rax, %rax
    jz        2f
1:
    movzwl    (%r11), %edx
    shll      $16, %edx
    movd      %edx, %xmm0
    addss     (%r10), %xmm0
    movss     %xmm0, (%r10)
    addq      $4, %r10
    addq      $2, %r11
    decq      %rax
    jnz       1b
2:
    ret

.p2align 4
.globl LFM_SYM(lfm_bf16_copy_bytes)
LFM_PRIVATE(LFM_SYM(lfm_bf16_copy_bytes))
LFM_SYM(lfm_bf16_copy_bytes):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rax
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
    movq      %rdx, %rax
#endif
    testq     %rax, %rax
    jz        2f
1:
    movzwl    (%r10), %r9d
    movw      %r9w, (%r11)
    addq      $2, %r10
    addq      $2, %r11
    decq      %rax
    jnz       1b
2:
    ret

.p2align 4
.globl LFM_SYM(lfm_bf16_unlift_bits)
LFM_PRIVATE(LFM_SYM(lfm_bf16_unlift_bits))
LFM_SYM(lfm_bf16_unlift_bits):
#ifdef _WIN32
    movzwl    (%rcx), %eax
#else
    movzwl    (%rdi), %eax
#endif
    shll      $16, %eax
    ret

.macro LFM_WIDEN_BF16 address, target, bits
    movzwl    \address, \bits
    shll      $16, \bits
    movd      \bits, \target
.endm

.macro LFM_RNE_WIDEN source, target, bits, tie
    movd      \source, \bits
    movl      \bits, \tie
    shrl      $16, \tie
    andl      $1, \tie
    addl      $0x7fff, \bits
    addl      \tie, \bits
    shrl      $16, \bits
    shll      $16, \bits
    movd      \bits, \target
.endm

.p2align 4
.globl LFM_SYM(lfm_bf16_rope_neox)
LFM_PRIVATE(LFM_SYM(lfm_bf16_rope_neox))
LFM_SYM(lfm_bf16_rope_neox):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rax
    movq      %r9, %rcx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
    movq      %rdx, %rax
#endif
    testq     %r10, %r10
    jz        3f
    testq     %r11, %r11
    jz        3f
    testq     %rax, %rax
    jz        3f
    testq     %rcx, %rcx
    jz        3f
    testq     $1, %rcx
    jnz       3f

    pushq     %r12
    pushq     %r13
    pushq     %r14
    movq      %rax, %r12
    movq      %rcx, %r14
    shrq      $1, %r14
    leaq      (%r10,%r14,2), %r13
1:
    LFM_WIDEN_BF16 (%r10), %xmm0, %r8d
    LFM_WIDEN_BF16 (%r13), %xmm1, %r8d
    LFM_WIDEN_BF16 (%r11), %xmm2, %r8d
    LFM_WIDEN_BF16 (%r12), %xmm3, %r8d

    movaps    %xmm0, %xmm4
    mulss     %xmm2, %xmm4
    LFM_RNE_WIDEN %xmm4, %xmm4, %r8d, %r9d
    movaps    %xmm1, %xmm5
    mulss     %xmm3, %xmm5
    movd      %xmm5, %r8d
    xorl      $0x80000000, %r8d
    movd      %r8d, %xmm5
    LFM_RNE_WIDEN %xmm5, %xmm5, %r8d, %r9d
    addss     %xmm5, %xmm4
    LFM_RNE_WIDEN %xmm4, %xmm4, %r8d, %r9d
    shrl      $16, %r8d
    movw      %r8w, (%r10)

    movaps    %xmm1, %xmm4
    mulss     %xmm2, %xmm4
    LFM_RNE_WIDEN %xmm4, %xmm4, %r8d, %r9d
    movaps    %xmm0, %xmm5
    mulss     %xmm3, %xmm5
    LFM_RNE_WIDEN %xmm5, %xmm5, %r8d, %r9d
    addss     %xmm5, %xmm4
    LFM_RNE_WIDEN %xmm4, %xmm4, %r8d, %r9d
    shrl      $16, %r8d
    movw      %r8w, (%r13)

    addq      $2, %r10
    addq      $2, %r11
    addq      $2, %r12
    addq      $2, %r13
    decq      %r14
    jnz       1b
    popq      %r14
    popq      %r13
    popq      %r12
3:
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif
