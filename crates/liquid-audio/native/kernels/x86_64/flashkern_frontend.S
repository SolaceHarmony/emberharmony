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

// void lfm_resample_conv_f64(const double *padded, uint64_t padded_len,
//                            const double *kernels, uint64_t phases,
//                            uint64_t klen, uint64_t stride,
//                            float *out, uint64_t max_blocks)
.p2align 4
.globl LFM_SYM(lfm_resample_conv_f64)
LFM_SYM(lfm_resample_conv_f64):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_resample_conv_f64)
#endif
    pushq     %rbx
#ifdef _WIN32
    .seh_pushreg %rbx
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
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // padded
    movq      %rdx, %rsi          // padded_len
    movq      %r8,  %rdx          // kernels
    movq      %r9,  %rcx          // phases
    movq      96(%rsp), %r8       // klen   (ret 8 + shadow 32 + 7 pushes 56)
    movq      104(%rsp), %r9      // stride
    movq      112(%rsp), %rbx     // out
    movq      120(%rsp), %r12     // max_blocks
#else
    movq      48(%rsp), %rbx      // out   (ret 8 + 5 pushes 40 = 7th arg)
    movq      56(%rsp), %r12      // max_blocks (8th arg)
#endif
    xorq      %r10, %r10          // start
    xorq      %r11, %r11          // block
1:
    leaq      (%r10,%r8), %rax
    cmpq      %rsi, %rax
    ja        8f
    cmpq      %r12, %r11
    jae       8f
    xorq      %r13, %r13          // phase
2:
    cmpq      %rcx, %r13
    jae       5f
    movq      %r13, %rax
    imulq     %r8, %rax
    leaq      (%rdx,%rax,8), %r14 // kernel row
    leaq      (%rdi,%r10,8), %r15 // padded + start
    xorpd     %xmm0, %xmm0        // acc
    xorq      %rax, %rax
3:
    cmpq      %r8, %rax
    jae       4f
    movsd     (%r15,%rax,8), %xmm1
    movsd     (%r14,%rax,8), %xmm2
    mulsd     %xmm2, %xmm1
    addsd     %xmm1, %xmm0
    incq      %rax
    jmp       3b
4:
    cvtsd2ss  %xmm0, %xmm0
    movss     %xmm0, (%rbx)
    addq      $4, %rbx
    incq      %r13
    jmp       2b
5:
    addq      %r9, %r10
    incq      %r11
    jmp       1b
8:
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
#ifdef _WIN32
    .seh_endproc
#endif

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
