#include "lfm_asm_visibility.h"

// Audio-frontend leaves, x86_64 twin of aarch64/flashkern_frontend.S.
// Scalar SSE2 on purpose (utterance-rate, parity-oracle op order, no FMA).
// Dual ABI like the sibling leaves: SysV (rdi rsi rdx rcx r8 r9 / xmm0-) and
// Win64 GNU (rcx rdx r8 r9 + stack, floats positional).

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text

// int lfm_preemph_f32(const float *x, float *y, uint64_t n, float coef)
.p2align 4
.globl LFM_SYM(lfm_preemph_f32)
LFM_PRIVATE(LFM_SYM(lfm_preemph_f32))
LFM_SYM(lfm_preemph_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movaps    %xmm3, %xmm0
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    testq     %r10, %r10
    jz        9f
    testq     %r11, %r11
    jz        9f
    testq     %rdx, %rdx
    jz        9f
    movss     (%r10), %xmm3       // previous original sample
    movss     %xmm3, (%r11)
    movq      $1, %rax
1:
    cmpq      %rdx, %rax
    jae       2f
    movss     (%r10,%rax,4), %xmm1 // load current before x==y store
    movaps    %xmm1, %xmm4
    movaps    %xmm3, %xmm2
    mulss     %xmm0, %xmm2
    subss     %xmm2, %xmm1
    movss     %xmm1, (%r11,%rax,4)
    movaps    %xmm4, %xmm3
    incq      %rax
    jmp       1b
2:
    xorl      %eax, %eax
    ret
9:
    movl      $-22, %eax
    ret

// void lfm_power_spec_f32(const float *re, const float *im, float *out, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_power_spec_f32)
LFM_PRIVATE(LFM_SYM(lfm_power_spec_f32))
LFM_SYM(lfm_power_spec_f32):
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
    jz        8f
1:
    movss     (%r10), %xmm0
    movss     (%r11), %xmm1
    mulss     %xmm0, %xmm0
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    movss     %xmm0, (%rdx)
    addq      $4, %r10
    addq      $4, %r11
    addq      $4, %rdx
    decq      %rcx
    jnz       1b
8:
    ret

// int lfm_log_add_f32(float *x, uint64_t n, float guard)
.p2align 4
.globl LFM_SYM(lfm_log_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_log_add_f32))
LFM_SYM(lfm_log_add_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_log_add_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    pushq     %rbx
    .seh_pushreg %rbx
    pushq     %r12
    .seh_pushreg %r12
    subq      $56, %rsp          // 32 shadow + guard spill + alignment
    .seh_stackalloc 56
    .seh_endprologue
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movaps    %xmm2, %xmm0
    testq     %rdi, %rdi
    jz        9f
#else
    testq     %rdi, %rdi
    jz        9f
    pushq     %rbx
    pushq     %r12
    subq      $56, %rsp          // call alignment + guard spill
#endif
    movq      %rdi, %rbx
    movq      %rsi, %r12
    movss     %xmm0, 40(%rsp)
1:
    testq     %r12, %r12
    jz        2f
    movss     (%rbx), %xmm0
    addss     40(%rsp), %xmm0
    call      LFM_CALL(logf)
    movss     %xmm0, (%rbx)
    addq      $4, %rbx
    decq      %r12
    jmp       1b
2:
    xorl      %eax, %eax
    jmp       7f
9:
    movl      $-22, %eax
#ifndef _WIN32
    ret
#endif
7:
    addq      $56, %rsp
    popq      %r12
    popq      %rbx
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// int lfm_rowstat_f32(const float *x, uint64_t rows, uint64_t cols,
//                     uint64_t valid, float constant, float *mean, float *std)
.p2align 4
.globl LFM_SYM(lfm_rowstat_f32)
LFM_PRIVATE(LFM_SYM(lfm_rowstat_f32))
LFM_SYM(lfm_rowstat_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_rowstat_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // x
    movq      %rdx, %rsi          // rows
    movq      %r8,  %rdx          // cols
    movq      %r9,  %rcx          // valid
    movss     56(%rsp), %xmm0     // constant
    movq      64(%rsp), %r8       // mean
    movq      72(%rsp), %r9       // std
#endif
    testq     %rdi, %rdi
    jz        9f
    testq     %rsi, %rsi
    jz        9f
    testq     %rcx, %rcx
    jz        9f
    cmpq      %rdx, %rcx
    ja        9f
    xorq      %r10, %r10          // row
    movq      %rdi, %r11          // row base
1:
    cmpq      %rsi, %r10
    jae       8f
    xorps     %xmm1, %xmm1        // sum
    xorq      %rax, %rax
2:
    addss     (%r11,%rax,4), %xmm1
    incq      %rax
    cmpq      %rcx, %rax
    jb        2b
    pxor      %xmm3, %xmm3
    cvtsi2ssq %rcx, %xmm3
    divss     %xmm3, %xmm1        // mean
    movss     %xmm1, (%r8,%r10,4)
    xorps     %xmm2, %xmm2        // centered sumsq
    cmpq      $1, %rcx
    jbe       4f
    xorq      %rax, %rax
3:
    movss     (%r11,%rax,4), %xmm5
    subss     %xmm1, %xmm5
    mulss     %xmm5, %xmm5
    addss     %xmm5, %xmm2
    incq      %rax
    cmpq      %rcx, %rax
    jb        3b
    leaq      -1(%rcx), %rax
    pxor      %xmm4, %xmm4
    cvtsi2ssq %rax, %xmm4
    divss     %xmm4, %xmm2
    sqrtss    %xmm2, %xmm2
4:
    addss     %xmm0, %xmm2        // += CONSTANT
    movss     %xmm2, (%r9,%r10,4)
    leaq      (%r11,%rdx,4), %r11
    incq      %r10
    jmp       1b
8:
    xorl      %eax, %eax
    jmp       7f
9:
    movl      $-22, %eax
7:
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_norm_apply_f32(float *x, const float *mean, const float *std,
//                         uint64_t rows, uint64_t cols)
.p2align 4
.globl LFM_SYM(lfm_norm_apply_f32)
LFM_PRIVATE(LFM_SYM(lfm_norm_apply_f32))
LFM_SYM(lfm_norm_apply_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_norm_apply_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      56(%rsp), %r8       // cols
#else
    // SysV: x=rdi mean=rsi std=rdx rows=rcx cols=r8
#endif
    xorq      %r10, %r10          // row
1:
    cmpq      %rcx, %r10
    jae       8f
    movss     (%rsi,%r10,4), %xmm0
    movss     (%rdx,%r10,4), %xmm1
    xorq      %r11, %r11
2:
    cmpq      %r8, %r11
    jae       3f
    movss     (%rdi), %xmm2
    subss     %xmm0, %xmm2
    divss     %xmm1, %xmm2
    movss     %xmm2, (%rdi)
    addq      $4, %rdi
    incq      %r11
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_resample_conv_f32(const float *input, uint64_t input_len,
//                            const double *kernels, uint64_t phases,
//                            uint64_t stride, uint64_t width,
//                            uint64_t first_value, float *out,
//                            uint64_t value_count)
// Direct-span torchaudio convolution. Logical zero prefix/suffix taps are
// skipped; in-span f32 samples are widened and accumulated in ascending-j f64
// order. first_value/value_count admit independently scheduled output ranges.
.p2align 4
.globl LFM_SYM(lfm_resample_conv_f32)
LFM_PRIVATE(LFM_SYM(lfm_resample_conv_f32))
LFM_SYM(lfm_resample_conv_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_resample_conv_f32)
#endif
    pushq     %rbx
#ifdef _WIN32
    .seh_pushreg %rbx
#endif
    pushq     %rbp
#ifdef _WIN32
    .seh_pushreg %rbp
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
#endif
    pushq     %r14
#ifdef _WIN32
    .seh_pushreg %r14
#endif
    pushq     %r15
#ifdef _WIN32
    .seh_pushreg %r15
#endif
    pushq     %rdi
#ifdef _WIN32
    .seh_pushreg %rdi
#endif
    pushq     %rsi
#ifdef _WIN32
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // input
    movq      %rdx, %rsi          // input_len
    movq      %r8,  %rbx          // kernels
    movq      %r9,  %rcx          // phases
    movq      104(%rsp), %r8      // stride (ret 8 + shadow 32 + pushes 64)
    movq      112(%rsp), %r9      // width
    movq      120(%rsp), %r10     // first_value
    movq      128(%rsp), %r11     // out
    movq      136(%rsp), %r12     // value_count
#else
    movq      %rdx, %rbx          // kernels
    movq      72(%rsp), %r10      // first_value (ret 8 + pushes 64)
    movq      80(%rsp), %r11      // out
    movq      88(%rsp), %r12      // value_count
#endif
    leaq      (%r8,%r9,2), %r13   // klen = stride + 2*width
    addq      %r9, %rsi           // one-past logical input coordinate
    movq      %r10, %rax
    xorq      %rdx, %rdx
    divq      %rcx
    movq      %rax, %r10          // initial block
    movq      %rdx, %r14          // initial phase
1:
    testq     %r12, %r12
    jz        8f
    movq      %r14, %rax
    imulq     %r13, %rax
    leaq      (%rbx,%rax,8), %r15 // kernel row
    movq      %r10, %rax
    imulq     %r8, %rax            // logical padded-window start

    cmpq      %rsi, %rax
    jae       6f                   // the entire window is suffix padding
    movq      %rsi, %rdx
    subq      %rax, %rdx           // first suffix j
    cmpq      %r13, %rdx
    jbe       2f
    movq      %r13, %rdx           // j_end = min(klen, width+len-start)
2:
    cmpq      %r9, %rax
    jae       3f
    movq      %r9, %rbp
    subq      %rax, %rbp           // j_begin = width - start
    xorq      %rax, %rax          // input index 0
    jmp       4f
3:
    xorq      %rbp, %rbp          // j_begin = 0
    subq      %r9, %rax           // input index = start - width
4:
    xorpd     %xmm0, %xmm0        // acc
5:
    cmpq      %rdx, %rbp
    jae       7f
    cvtss2sd  (%rdi,%rax,4), %xmm1
    movsd     (%r15,%rbp,8), %xmm2
    mulsd     %xmm2, %xmm1
    addsd     %xmm1, %xmm0
    incq      %rax
    incq      %rbp
    jmp       5b
6:
    xorpd     %xmm0, %xmm0
7:
    cvtsd2ss  %xmm0, %xmm0
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decq      %r12
    incq      %r14
    cmpq      %rcx, %r14
    jb        1b
    xorq      %r14, %r14
    incq      %r10
    jmp       1b
8:
#ifdef _WIN32
    // rsi/rdi are restored by the common epilogue below.
#endif
    popq      %rsi
    popq      %rdi
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbp
    popq      %rbx
    ret
#ifdef _WIN32
    .seh_endproc
#endif

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
