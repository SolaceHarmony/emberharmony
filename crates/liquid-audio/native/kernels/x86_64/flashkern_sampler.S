#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.intel_syntax noprefix
.text

/* uint32_t lfm_sampler_argmax_f32(const float *x, size_t count) */
.p2align 4
.globl LFM_SYM(lfm_sampler_argmax_f32)
LFM_PRIVATE(LFM_SYM(lfm_sampler_argmax_f32))
LFM_SYM(lfm_sampler_argmax_f32):
#ifdef _WIN32
    mov     r10, rcx
    mov     r11, rdx
#else
    mov     r10, rdi
    mov     r11, rsi
#endif
    xor     eax, eax
    test    r11, r11
    jz      .Largmax_f32_done
    mov     edx, 0xff800000
    movd    xmm0, edx
    pshufd  xmm0, xmm0, 0
    movaps  xmm3, xmm0
    xor     r8d, r8d
.Largmax_f32_max:
    mov     r9, r11
    sub     r9, r8
    cmp     r9, 4
    jb      .Largmax_f32_tail
    movups  xmm1, XMMWORD PTR [r10 + r8*4]
    movaps  xmm2, xmm1
    cmpordps xmm2, xmm2
    andps   xmm1, xmm2
    andnps  xmm2, xmm3
    orps    xmm1, xmm2
    maxps   xmm0, xmm1
    add     r8, 4
    jmp     .Largmax_f32_max
.Largmax_f32_tail:
    movaps  xmm1, xmm0
    movhlps xmm1, xmm0
    maxps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 1
    maxss   xmm0, xmm1
    cmp     r8, r11
    jae     .Largmax_f32_scan
.Largmax_f32_tail_loop:
    movss   xmm1, DWORD PTR [r10 + r8*4]
    ucomiss xmm1, xmm0
    jbe     .Largmax_f32_next
    movaps  xmm0, xmm1
.Largmax_f32_next:
    inc     r8
    cmp     r8, r11
    jb      .Largmax_f32_tail_loop
.Largmax_f32_scan:
    xor     r8d, r8d
.Largmax_f32_find:
    movss   xmm1, DWORD PTR [r10 + r8*4]
    ucomiss xmm1, xmm0
    jp      .Largmax_f32_find_next
    je      .Largmax_f32_found
.Largmax_f32_find_next:
    inc     r8
    cmp     r8, r11
    jb      .Largmax_f32_find
    ret
.Largmax_f32_found:
    mov     eax, r8d
.Largmax_f32_done:
    ret

/* uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count) */
.p2align 4
.globl LFM_SYM(lfm_sampler_argmax_bf16)
LFM_PRIVATE(LFM_SYM(lfm_sampler_argmax_bf16))
LFM_SYM(lfm_sampler_argmax_bf16):
#ifdef _WIN32
    mov     r10, rcx
    mov     r11, rdx
#else
    mov     r10, rdi
    mov     r11, rsi
#endif
    xor     eax, eax
    test    r11, r11
    jz      .Largmax_bf16_done
    mov     edx, 0xff800000
    movd    xmm0, edx
    pshufd  xmm0, xmm0, 0
    movaps  xmm3, xmm0
    xor     r8d, r8d
.Largmax_bf16_max:
    mov     r9, r11
    sub     r9, r8
    cmp     r9, 4
    jb      .Largmax_bf16_tail
    movq    xmm1, QWORD PTR [r10 + r8*2]
    pxor    xmm2, xmm2
    punpcklwd xmm1, xmm2
    pslld   xmm1, 16
    movaps  xmm2, xmm1
    cmpordps xmm2, xmm2
    andps   xmm1, xmm2
    andnps  xmm2, xmm3
    orps    xmm1, xmm2
    maxps   xmm0, xmm1
    add     r8, 4
    jmp     .Largmax_bf16_max
.Largmax_bf16_tail:
    movaps  xmm1, xmm0
    movhlps xmm1, xmm0
    maxps   xmm0, xmm1
    movaps  xmm1, xmm0
    shufps  xmm1, xmm1, 1
    maxss   xmm0, xmm1
    cmp     r8, r11
    jae     .Largmax_bf16_scan
.Largmax_bf16_tail_loop:
    movzx   edx, WORD PTR [r10 + r8*2]
    shl     edx, 16
    movd    xmm1, edx
    ucomiss xmm1, xmm0
    jbe     .Largmax_bf16_next
    movaps  xmm0, xmm1
.Largmax_bf16_next:
    inc     r8
    cmp     r8, r11
    jb      .Largmax_bf16_tail_loop
.Largmax_bf16_scan:
    xor     r8d, r8d
.Largmax_bf16_find:
    movzx   edx, WORD PTR [r10 + r8*2]
    shl     edx, 16
    movd    xmm1, edx
    ucomiss xmm1, xmm0
    jp      .Largmax_bf16_find_next
    je      .Largmax_bf16_found
.Largmax_bf16_find_next:
    inc     r8
    cmp     r8, r11
    jb      .Largmax_bf16_find
    ret
.Largmax_bf16_found:
    mov     eax, r8d
.Largmax_bf16_done:
    ret

/*
 * The exp-sum leaves use the private f32 range-reduced polynomial below. All
 * traversal, scaling, thresholding, exponentiation, destination writes, and
 * ordered reduction are hand-written assembly.
 */
.p2align 4
.globl LFM_SYM(lfm_sampler_exp_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_sampler_exp_sum_f32))
LFM_SYM(lfm_sampler_exp_sum_f32):
    push    rbp
    mov     rbp, rsp
    push    r12
    push    r13
    push    r14
#ifdef _WIN32
    sub     rsp, 72
    mov     r12, rcx
    mov     r13, rdx
    mov     r14, r8
    movss   DWORD PTR [rbp - 32], xmm3
    movss   xmm0, DWORD PTR [rbp + 48]
    movss   DWORD PTR [rbp - 36], xmm0
    movss   xmm0, DWORD PTR [rbp + 56]
    movss   DWORD PTR [rbp - 40], xmm0
#else
    sub     rsp, 40
    mov     r12, rdi
    mov     r13, rsi
    mov     r14, rdx
    movss   DWORD PTR [rbp - 32], xmm0
    movss   DWORD PTR [rbp - 36], xmm1
    movss   DWORD PTR [rbp - 40], xmm2
#endif
    xorps   xmm0, xmm0
    movss   DWORD PTR [rbp - 44], xmm0
    test    r14, r14
    jz      .Lexp_f32_done
.Lexp_f32_loop:
    movss   xmm0, DWORD PTR [r12]
    mulss   xmm0, DWORD PTR [rbp - 32]
    ucomiss xmm0, DWORD PTR [rbp - 40]
    jp      .Lexp_f32_zero
    jb      .Lexp_f32_zero
    subss   xmm0, DWORD PTR [rbp - 36]
    call    .Lsampler_exp_f32
    jmp     .Lexp_f32_store
.Lexp_f32_zero:
    xorps   xmm0, xmm0
.Lexp_f32_store:
    movss   DWORD PTR [r13], xmm0
    addss   xmm0, DWORD PTR [rbp - 44]
    movss   DWORD PTR [rbp - 44], xmm0
    add     r12, 4
    add     r13, 4
    dec     r14
    jnz     .Lexp_f32_loop
.Lexp_f32_done:
    movss   xmm0, DWORD PTR [rbp - 44]
#ifdef _WIN32
    add     rsp, 72
#else
    add     rsp, 40
#endif
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    ret

.p2align 4
.globl LFM_SYM(lfm_sampler_exp_sum_bf16)
LFM_PRIVATE(LFM_SYM(lfm_sampler_exp_sum_bf16))
LFM_SYM(lfm_sampler_exp_sum_bf16):
    push    rbp
    mov     rbp, rsp
    push    r12
    push    r13
    push    r14
#ifdef _WIN32
    sub     rsp, 72
    mov     r12, rcx
    mov     r13, rdx
    mov     r14, r8
    mov     eax, r9d
    shl     eax, 16
    movd    xmm2, eax
    movss   DWORD PTR [rbp - 32], xmm2
    movss   xmm0, DWORD PTR [rbp + 48]
    movss   DWORD PTR [rbp - 36], xmm0
    movss   xmm0, DWORD PTR [rbp + 56]
    movss   DWORD PTR [rbp - 40], xmm0
#else
    sub     rsp, 40
    mov     r12, rdi
    mov     r13, rsi
    mov     r14, rdx
    mov     eax, ecx
    shl     eax, 16
    movd    xmm2, eax
    movss   DWORD PTR [rbp - 32], xmm2
    movss   DWORD PTR [rbp - 36], xmm0
    movss   DWORD PTR [rbp - 40], xmm1
#endif
    xorps   xmm0, xmm0
    movss   DWORD PTR [rbp - 44], xmm0
    test    r14, r14
    jz      .Lexp_bf16_done
.Lexp_bf16_loop:
    movzx   eax, WORD PTR [r12]
    shl     eax, 16
    movd    xmm0, eax
    mulss   xmm0, DWORD PTR [rbp - 32]
    movd    eax, xmm0
    mov     ecx, eax
    shr     ecx, 16
    and     ecx, 1
    add     eax, 0x7fff
    add     eax, ecx
    shr     eax, 16
    shl     eax, 16
    movd    xmm0, eax
    ucomiss xmm0, DWORD PTR [rbp - 40]
    jp      .Lexp_bf16_zero
    jb      .Lexp_bf16_zero
    subss   xmm0, DWORD PTR [rbp - 36]
    call    .Lsampler_exp_f32
    jmp     .Lexp_bf16_store
.Lexp_bf16_zero:
    xorps   xmm0, xmm0
.Lexp_bf16_store:
    movss   DWORD PTR [r13], xmm0
    addss   xmm0, DWORD PTR [rbp - 44]
    movss   DWORD PTR [rbp - 44], xmm0
    add     r12, 2
    add     r13, 4
    dec     r14
    jnz     .Lexp_bf16_loop
.Lexp_bf16_done:
    movss   xmm0, DWORD PTR [rbp - 44]
#ifdef _WIN32
    add     rsp, 72
#else
    add     rsp, 40
#endif
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    ret

/* uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count, float target) */
.p2align 4
.globl LFM_SYM(lfm_sampler_prefix_pick)
LFM_PRIVATE(LFM_SYM(lfm_sampler_prefix_pick))
LFM_SYM(lfm_sampler_prefix_pick):
#ifdef _WIN32
    mov     r10, rcx
    mov     r11, rdx
    movaps  xmm0, xmm2
#else
    mov     r10, rdi
    mov     r11, rsi
#endif
    xor     eax, eax
    xor     r8d, r8d
    xorps   xmm1, xmm1
    xorps   xmm3, xmm3
    test    r11, r11
    jz      .Lprefix_done
.Lprefix_loop:
    movss   xmm2, DWORD PTR [r10 + r8*4]
    ucomiss xmm2, xmm3
    jbe     .Lprefix_sum
    mov     eax, r8d
.Lprefix_sum:
    addss   xmm1, xmm2
    ucomiss xmm0, xmm1
    jp      .Lprefix_next
    jb      .Lprefix_found
.Lprefix_next:
    inc     r8
    cmp     r8, r11
    jb      .Lprefix_loop
.Lprefix_done:
    ret
.Lprefix_found:
    mov     eax, r8d
    ret

/*
 * Private f32 exp for x <= 0 sampler inputs.
 * n = trunc(x * log2(e) - 0.5), r = x - n * ln(2),
 * exp(x) = 2^n * P6(r). Values below the normal f32 exponent range are zero.
 */
.p2align 4
.Lsampler_exp_f32:
    movaps  xmm4, xmm0
    movss   xmm1, DWORD PTR [rip + .Lsampler_exp_constants]
    mulss   xmm1, xmm0
    addss   xmm1, DWORD PTR [rip + .Lsampler_exp_constants + 4]
    cvttss2si eax, xmm1
    cmp     eax, -126
    jl      .Lsampler_exp_f32_zero

    cvtsi2ss xmm2, eax
    mulss   xmm2, DWORD PTR [rip + .Lsampler_exp_constants + 8]
    subss   xmm4, xmm2

    movss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 12]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 16]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 20]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 24]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 28]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 32]
    mulss   xmm3, xmm4
    addss   xmm3, DWORD PTR [rip + .Lsampler_exp_constants + 32]

    mov     ecx, eax
    add     ecx, 127
    shl     ecx, 23
    movd    xmm2, ecx
    mulss   xmm3, xmm2
    movaps  xmm0, xmm3
    ret

.Lsampler_exp_f32_zero:
    xorps   xmm0, xmm0
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

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif
