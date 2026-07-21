#include "lfm_asm_visibility.h"

// x86-64 SSE2 twin of the AArch64 selected-bin Sesame/Web-Audio leaf. Tables
// contain the Blackman window and 1/N; only compact smoothed magnitudes leave
// this function.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 4

// void lfm_sesame_selected_magnitudes(const float *pcm,
//     const float *real_table, const float *imag_table,
//     float *smoothed_magnitudes, size_t selected_bins)
.globl LFM_SYM(lfm_sesame_selected_magnitudes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_selected_magnitudes))
LFM_SYM(lfm_sesame_selected_magnitudes):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      40(%rsp), %r8
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    testq     %r8, %r8
    jz        9f
    movl      $0x3e4ccccd, %eax
    movd      %eax, %xmm5       // 0.2f
1:
    movq      %r10, %r9
    movl      $64, %eax
    xorps     %xmm0, %xmm0
    xorps     %xmm1, %xmm1
2:
    movups    (%r9), %xmm2
    movups    (%r11), %xmm3
    movups    (%rdx), %xmm4
    mulps     %xmm2, %xmm3
    mulps     %xmm2, %xmm4
    addps     %xmm3, %xmm0
    addps     %xmm4, %xmm1
    addq      $16, %r9
    addq      $16, %r11
    addq      $16, %rdx
    decl      %eax
    jnz       2b

    movaps    %xmm0, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm0
    movaps    %xmm0, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm0
    movaps    %xmm1, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm1
    movaps    %xmm1, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm1

    mulss     %xmm0, %xmm0
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    sqrtss    %xmm0, %xmm0
    movd      %xmm0, %eax
    andl      $0x7f800000, %eax
    cmpl      $0x7f800000, %eax
    jne       3f
    xorps     %xmm0, %xmm0
3:
    mulss     %xmm5, %xmm0
    movss     (%rcx), %xmm1
    movl      $0x3f4ccccd, %eax
    movd      %eax, %xmm4       // 0.8f; xmm6 is nonvolatile on Win64
    mulss     %xmm4, %xmm1
    addss     %xmm1, %xmm0
    movss     %xmm0, (%rcx)
    addq      $4, %rcx
    decq      %r8
    jnz       1b
9:
    ret

// void lfm_sesame_selected_magnitudes_window(
//     const LfmSesameWindowV1 *window, const float *real_table,
//     const float *imag_table, float *smoothed_magnitudes,
//     size_t selected_bins)
// Rebuild only the one SIMD group that can straddle the circular wrap. Every
// mul/add and horizontal reduction remains ordered exactly like the contiguous
// leaf, so split-point placement cannot change classifier bits.
.p2align 4
.globl LFM_SYM(lfm_sesame_selected_magnitudes_window)
LFM_PRIVATE(LFM_SYM(lfm_sesame_selected_magnitudes_window))
LFM_SYM(lfm_sesame_selected_magnitudes_window):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_sesame_selected_magnitudes_window)
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
    .seh_endprologue
    movq      %rcx, %r12
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      56(%rsp), %r8
#else
    movq      %rdi, %r12
    movq      %rsi, %r11
#endif
    testq     %r8, %r8
    jz        .Lwindow_done
    movl      $0x3e4ccccd, %eax
    movd      %eax, %xmm5       // 0.2f
.Lwindow_bin:
    movq      0(%r12), %r9
    movq      8(%r12), %rax
    movq      16(%r12), %r10
    movl      $64, %r13d
    xorps     %xmm0, %xmm0
    xorps     %xmm1, %xmm1
.Lwindow_group:
    cmpq      $4, %rax
    jae       .Lwindow_first4
    testq     %rax, %rax
    jz        .Lwindow_second4
    cmpq      $2, %rax
    je        .Lwindow_split2
    ja        .Lwindow_split3
    // 1 + 3 samples.
    movss     0(%r9), %xmm2
    movss     0(%r10), %xmm3
    unpcklps  %xmm3, %xmm2
    movss     4(%r10), %xmm3
    movss     8(%r10), %xmm4
    unpcklps  %xmm4, %xmm3
    movlhps   %xmm3, %xmm2
    addq      $4, %r9
    addq      $12, %r10
    xorl      %eax, %eax
    jmp       .Lwindow_accumulate
.Lwindow_split2:
    movss     0(%r9), %xmm2
    movss     4(%r9), %xmm3
    unpcklps  %xmm3, %xmm2
    movss     0(%r10), %xmm3
    movss     4(%r10), %xmm4
    unpcklps  %xmm4, %xmm3
    movlhps   %xmm3, %xmm2
    addq      $8, %r9
    addq      $8, %r10
    xorl      %eax, %eax
    jmp       .Lwindow_accumulate
.Lwindow_split3:
    movss     0(%r9), %xmm2
    movss     4(%r9), %xmm3
    unpcklps  %xmm3, %xmm2
    movss     8(%r9), %xmm3
    movss     0(%r10), %xmm4
    unpcklps  %xmm4, %xmm3
    movlhps   %xmm3, %xmm2
    addq      $12, %r9
    addq      $4, %r10
    xorl      %eax, %eax
    jmp       .Lwindow_accumulate
.Lwindow_first4:
    movups    0(%r9), %xmm2
    addq      $16, %r9
    subq      $4, %rax
    jmp       .Lwindow_accumulate
.Lwindow_second4:
    movups    0(%r10), %xmm2
    addq      $16, %r10
.Lwindow_accumulate:
    movups    0(%r11), %xmm3
    movups    0(%rdx), %xmm4
    mulps     %xmm2, %xmm3
    mulps     %xmm2, %xmm4
    addps     %xmm3, %xmm0
    addps     %xmm4, %xmm1
    addq      $16, %r11
    addq      $16, %rdx
    decl      %r13d
    jnz       .Lwindow_group

    movaps    %xmm0, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm0
    movaps    %xmm0, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm0
    movaps    %xmm1, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm1
    movaps    %xmm1, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm1
    mulss     %xmm0, %xmm0
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    sqrtss    %xmm0, %xmm0
    movd      %xmm0, %eax
    andl      $0x7f800000, %eax
    cmpl      $0x7f800000, %eax
    jne       .Lwindow_finite
    xorps     %xmm0, %xmm0
.Lwindow_finite:
    mulss     %xmm5, %xmm0
    movss     0(%rcx), %xmm1
    movl      $0x3f4ccccd, %eax
    movd      %eax, %xmm4       // 0.8f
    mulss     %xmm4, %xmm1
    addss     %xmm1, %xmm0
    movss     %xmm0, 0(%rcx)
    addq      $4, %rcx
    decq      %r8
    jnz       .Lwindow_bin
.Lwindow_done:
    popq      %r13
    popq      %r12
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// Load one logical sample into the requested SIMD lane source register,
// advancing to the next validated descriptor only after this span is empty.
.macro LFM_SESAME_SCATTER_SCALAR target
    testq     %rax, %rax
    jnz       .Lscatter_have\@
    movq      0(%r10), %r9
    movq      8(%r10), %rax
    addq      $16, %r10
.Lscatter_have\@:
    movss     0(%r9), \target
    addq      $4, %r9
    decq      %rax
.endm

// void lfm_sesame_selected_magnitudes_scatter(
//     const LfmSesameScatterWindowV1 *window, const float *real_table,
//     const float *imag_table, float *smoothed_magnitudes,
//     size_t selected_bins)
//
// Full four-sample groups load directly from their owning span. Groups that
// cross one or more descriptor boundaries are assembled in xmm2. The math and
// reduction order are identical to the contiguous leaf and no PCM plane or
// stack spill is created.
.p2align 4
.globl LFM_SYM(lfm_sesame_selected_magnitudes_scatter)
LFM_PRIVATE(LFM_SYM(lfm_sesame_selected_magnitudes_scatter))
LFM_SYM(lfm_sesame_selected_magnitudes_scatter):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_sesame_selected_magnitudes_scatter)
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
    .seh_endprologue
    movq      %rcx, %r12
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      56(%rsp), %r8
#else
    movq      %rdi, %r12
    movq      %rsi, %r11
#endif
    testq     %r8, %r8
    jz        .Lscatter_done
    movl      $0x3e4ccccd, %eax
    movd      %eax, %xmm5       // 0.2f
.Lscatter_bin:
    movq      0(%r12), %r10
    movq      0(%r10), %r9
    movq      8(%r10), %rax
    addq      $16, %r10
    movl      $64, %r13d
    xorps     %xmm0, %xmm0
    xorps     %xmm1, %xmm1
.Lscatter_group:
    cmpq      $4, %rax
    jb        .Lscatter_fragment
    movups    0(%r9), %xmm2
    addq      $16, %r9
    subq      $4, %rax
    jmp       .Lscatter_accumulate
.Lscatter_fragment:
    LFM_SESAME_SCATTER_SCALAR %xmm2
    LFM_SESAME_SCATTER_SCALAR %xmm3
    unpcklps  %xmm3, %xmm2
    LFM_SESAME_SCATTER_SCALAR %xmm3
    LFM_SESAME_SCATTER_SCALAR %xmm4
    unpcklps  %xmm4, %xmm3
    movlhps   %xmm3, %xmm2
.Lscatter_accumulate:
    movups    0(%r11), %xmm3
    movups    0(%rdx), %xmm4
    mulps     %xmm2, %xmm3
    mulps     %xmm2, %xmm4
    addps     %xmm3, %xmm0
    addps     %xmm4, %xmm1
    addq      $16, %r11
    addq      $16, %rdx
    decl      %r13d
    jnz       .Lscatter_group

    movaps    %xmm0, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm0
    movaps    %xmm0, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm0
    movaps    %xmm1, %xmm2
    shufps    $0x4e, %xmm2, %xmm2
    addps     %xmm2, %xmm1
    movaps    %xmm1, %xmm2
    shufps    $0xb1, %xmm2, %xmm2
    addss     %xmm2, %xmm1
    mulss     %xmm0, %xmm0
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    sqrtss    %xmm0, %xmm0
    movd      %xmm0, %eax
    andl      $0x7f800000, %eax
    cmpl      $0x7f800000, %eax
    jne       .Lscatter_finite
    xorps     %xmm0, %xmm0
.Lscatter_finite:
    mulss     %xmm5, %xmm0
    movss     0(%rcx), %xmm1
    movl      $0x3f4ccccd, %eax
    movd      %eax, %xmm4       // 0.8f
    mulss     %xmm4, %xmm1
    addss     %xmm1, %xmm0
    movss     %xmm0, 0(%rcx)
    addq      $4, %rcx
    decq      %r8
    jnz       .Lscatter_bin
.Lscatter_done:
    popq      %r13
    popq      %r12
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_sesame_magnitudes_to_bytes(const float *magnitudes,
//     const double thresholds[256], uint8_t *bytes, size_t count)
.p2align 4
.globl LFM_SYM(lfm_sesame_magnitudes_to_bytes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_magnitudes_to_bytes))
LFM_SYM(lfm_sesame_magnitudes_to_bytes):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    testq     %rcx, %rcx
    jz        9f
.macro LFM_SESAME_BYTE_STEP
    leal      (%eax,%r8d), %r9d
    shrl      %r9d
    comisd    (%r11,%r9,8), %xmm0
    cmovael   %r9d, %eax
    cmovbl    %r9d, %r8d
.endm
1:
    cvtss2sd  (%r10), %xmm0
    xorl      %eax, %eax
    movl      $256, %r8d
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    LFM_SESAME_BYTE_STEP
    movb      %al, (%rdx)
    addq      $4, %r10
    incq      %rdx
    decq      %rcx
    jnz       1b
9:
    ret

// uint32_t lfm_sesame_classify_selected_bytes(const uint8_t *bytes,
//     size_t count, uint8_t *minimum, uint8_t *maximum, uint32_t threshold,
//     double *score)
.p2align 4
.globl LFM_SYM(lfm_sesame_classify_selected_bytes)
LFM_PRIVATE(LFM_SYM(lfm_sesame_classify_selected_bytes))
LFM_SYM(lfm_sesame_classify_selected_bytes):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movl      40(%rsp), %r8d
    movq      48(%rsp), %r9
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movq      %rdx, %xmm4       // minimum pointer; xmm0-5 are Win64 volatile
    movq      %rcx, %xmm5       // maximum pointer
    movzbl    (%rdx), %edx
    movzbl    (%rcx), %ecx
    cvtsi2sdl %r8d, %xmm3       // threshold survives the byte loop
    cvtsi2sdq %r11, %xmm2       // count survives the byte loop
    xorl      %eax, %eax         // sum
1:
    movzbl    (%r10), %r8d
    addq      %r8, %rax
    cmpl      %edx, %r8d
    cmovbl    %r8d, %edx
    cmpl      %ecx, %r8d
    cmoval    %r8d, %ecx
    incq      %r10
    decq      %r11
    jnz       1b
    movq      %xmm4, %r10
    movq      %xmm5, %r11
    movb      %dl, (%r10)
    movb      %cl, (%r11)

    subl      %edx, %ecx
    jz        2f
    cvtsi2sdq %rax, %xmm0
    divsd     %xmm2, %xmm0
    cvtsi2sdl %edx, %xmm1
    subsd     %xmm1, %xmm0
    cvtsi2sdl %ecx, %xmm1
    divsd     %xmm1, %xmm0
    movabsq   $0x4059000000000000, %rax // 100.0
    movq      %rax, %xmm1
    mulsd     %xmm1, %xmm0
    jmp       3f
2:
    xorpd     %xmm0, %xmm0
3:
    movsd     %xmm0, (%r9)
    comisd    %xmm3, %xmm0
    setae     %al
    movzbl    %al, %eax
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
