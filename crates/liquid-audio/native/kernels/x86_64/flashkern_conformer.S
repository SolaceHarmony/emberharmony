// Conformer/adapter leaves, x86_64 twin of aarch64/flashkern_conformer.S.
// Scalar SSE2, strictly ordered, RNE bf16 rounding at exactly the points the
// candle ladder rounds. SysV plus Win64-GNU arg normalization (the sibling
// leaves' convention). No FMA.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

// f32 in %xmm_src -> RNE bf16 bits in %eax. Uses \scr as scratch (default
// %ecx); callers that keep a live value in rcx across the round pass a free reg.
.macro BF16_RNE_XMM src, scr=%ecx
    movd      \src, %eax
    movl      %eax, \scr
    shrl      $16, \scr
    andl      $1, \scr
    addl      $0x7fff, %eax
    addl      \scr, %eax
    shrl      $16, %eax
.endm

.text

// void lfm_bf16_widen_f32(const uint16_t *x, float *y, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_bf16_widen_f32)
LFM_SYM(lfm_bf16_widen_f32):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
#endif
    testq     %rdx, %rdx
    jz        8f
1:
    movzwl    (%rdi), %eax
    shll      $16, %eax
    movl      %eax, (%rsi)
    addq      $2, %rdi
    addq      $4, %rsi
    decq      %rdx
    jnz       1b
8:
    ret

// int lfm_ln_bf16(const uint16_t *x, const uint16_t *w, const uint16_t *b,
//                 uint16_t *y, uint64_t rows, uint64_t cols, float eps)
// SysV: rdi rsi rdx rcx r8 r9 xmm0. Win64: rcx rdx r8 r9 + stack{rows? no —
// 5th arg 40(%rsp), 6th 48(%rsp)}, eps is 7th -> 56(%rsp).
.p2align 4
.globl LFM_SYM(lfm_ln_bf16)
LFM_SYM(lfm_ln_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      40(%rsp), %r8
    movq      48(%rsp), %r9
    movss     56(%rsp), %xmm0
#endif
    testq     %rdi, %rdi
    jz        9f
    testq     %rcx, %rcx
    jz        9f
    testq     %r9, %r9
    jz        9f
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    // xmm0=eps stays; r10=row, r11=x row ptr, r12=y row ptr, r13=col
    xorq      %r10, %r10
1:
    cmpq      %r8, %r10
    jae       8f
    // r11 = x row, r12 = y row; imulq keeps rdx (bias ptr) intact.
    movq      %r10, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,2), %r11
    leaq      (%rcx,%rax,2), %r12
    // pass 1: sum -> xmm1
    xorps     %xmm1, %xmm1
    xorq      %r13, %r13
2:
    movzwl    (%r11,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm2
    addss     %xmm2, %xmm1
    incq      %r13
    cmpq      %r9, %r13
    jb        2b
    pxor      %xmm3, %xmm3
    cvtsi2ssq %r9, %xmm3
    divss     %xmm3, %xmm1         // mean -> xmm1 (keep xmm4 = mean)
    movaps    %xmm1, %xmm4
    // pass 2: biased variance -> xmm5
    xorps     %xmm5, %xmm5
    xorq      %r13, %r13
3:
    movzwl    (%r11,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm2
    subss     %xmm4, %xmm2
    mulss     %xmm2, %xmm2
    addss     %xmm2, %xmm5
    incq      %r13
    cmpq      %r9, %r13
    jb        3b
    divss     %xmm3, %xmm5
    addss     %xmm0, %xmm5
    sqrtss    %xmm5, %xmm5         // denom
    // pass 3
    xorq      %r13, %r13
4:
    movzwl    (%r11,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm2
    subss     %xmm4, %xmm2
    divss     %xmm5, %xmm2
    BF16_RNE_XMM %xmm2, %ebx
    shll      $16, %eax
    movd      %eax, %xmm2          // widened normed
    movzwl    (%rsi,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm6
    mulss     %xmm6, %xmm2
    BF16_RNE_XMM %xmm2, %ebx
    shll      $16, %eax
    movd      %eax, %xmm2
    movzwl    (%rdx,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm6
    addss     %xmm6, %xmm2
    BF16_RNE_XMM %xmm2, %ebx
    movw      %ax, (%r12,%r13,2)
    incq      %r13
    cmpq      %r9, %r13
    jb        4b
    incq      %r10
    jmp       1b
8:
    xorl      %eax, %eax
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

// void lfm_bn_bf16(x, mean, denom, w, b, y, channels, t)
// SysV: rdi rsi rdx rcx r8 r9 + 8(%rsp)=channels 16(%rsp)=t.
// Win64: rcx rdx r8 r9 + 40..64(%rsp).
.p2align 4
.globl LFM_SYM(lfm_bn_bf16)
LFM_SYM(lfm_bn_bf16):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      80(%rsp), %r8       // b      (40 + 5*8)
    movq      88(%rsp), %r9       // y
    movq      96(%rsp), %r10      // channels
    movq      104(%rsp), %r11     // t
#else
    movq      48(%rsp), %r10      // channels (8 + 5*8)
    movq      56(%rsp), %r11      // t
#endif
    xorq      %r12, %r12          // channel
1:
    cmpq      %r10, %r12
    jae       8f
    movzwl    (%rsi,%r12,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm1         // mean_c
    movzwl    (%rdx,%r12,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm2         // denom_c
    movzwl    (%rcx,%r12,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm3         // w_c
    movzwl    (%r8,%r12,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm4         // b_c
    movq      %r12, %r13
    imulq     %r11, %r13
    leaq      (%rdi,%r13,2), %r14 // x row
    leaq      (%r9,%r13,2), %r15  // y row
    xorq      %r13, %r13
2:
    cmpq      %r11, %r13
    jae       3f
    movzwl    (%r14,%r13,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm5
    subss     %xmm1, %xmm5
    BF16_RNE_XMM %xmm5, %ebx
    shll      $16, %eax
    movd      %eax, %xmm5
    divss     %xmm2, %xmm5
    BF16_RNE_XMM %xmm5, %ebx
    shll      $16, %eax
    movd      %eax, %xmm5
    mulss     %xmm3, %xmm5
    BF16_RNE_XMM %xmm5, %ebx
    shll      $16, %eax
    movd      %eax, %xmm5
    addss     %xmm4, %xmm5
    BF16_RNE_XMM %xmm5, %ebx
    movw      %ax, (%r15,%r13,2)
    incq      %r13
    jmp       2b
3:
    incq      %r12
    jmp       1b
8:
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// void lfm_relu_bf16(uint16_t *x, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_relu_bf16)
LFM_SYM(lfm_relu_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
#endif
    testq     %rsi, %rsi
    jz        8f
1:
    movzwl    (%rdi), %eax
    testl     $0x8000, %eax
    jz        2f
    movw      $0, (%rdi)
2:
    addq      $2, %rdi
    decq      %rsi
    jnz       1b
8:
    ret

// int lfm_silu_bf16(uint16_t *x, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_silu_bf16)
LFM_SYM(lfm_silu_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
#endif
    testq     %rdi, %rdi
    jz        9f
    pushq     %rbx
    pushq     %r12
    subq      $56, %rsp
    movq      %rdi, %rbx
    movq      %rsi, %r12
1:
    testq     %r12, %r12
    jz        2f
    movzwl    (%rbx), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    movss     %xmm0, 40(%rsp)     // save x
    xorps     %xmm1, %xmm1
    subss     %xmm0, %xmm1
    movaps    %xmm1, %xmm0        // -x
    call      LFM_CALL(expf)
    movl      $0x3f800000, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0        // 1 + e^-x
    movss     40(%rsp), %xmm2
    divss     %xmm0, %xmm2        // x / (1+e^-x)
    BF16_RNE_XMM %xmm2
    movw      %ax, (%rbx)
    addq      $2, %rbx
    decq      %r12
    jmp       1b
2:
    xorl      %eax, %eax
    addq      $56, %rsp
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

// int lfm_glu_bf16(const uint16_t *a, const uint16_t *b, uint16_t *y, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_glu_bf16)
LFM_SYM(lfm_glu_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#endif
    testq     %rdi, %rdi
    jz        9f
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    subq      $40, %rsp
    movq      %rdi, %rbx          // a
    movq      %rsi, %r12          // b
    movq      %rdx, %r13          // y
    movq      %rcx, %r14          // n
1:
    testq     %r14, %r14
    jz        2f
    movzwl    (%r12), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    xorps     %xmm1, %xmm1
    subss     %xmm0, %xmm1
    movaps    %xmm1, %xmm0
    call      LFM_CALL(expf)
    movl      $0x3f800000, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0
    movaps    %xmm1, %xmm2
    divss     %xmm0, %xmm2        // sigmoid(b)
    BF16_RNE_XMM %xmm2
    shll      $16, %eax
    movd      %eax, %xmm2         // rounded sigmoid widened
    movzwl    (%rbx), %eax
    shll      $16, %eax
    movd      %eax, %xmm3
    mulss     %xmm3, %xmm2
    BF16_RNE_XMM %xmm2
    movw      %ax, (%r13)
    addq      $2, %rbx
    addq      $2, %r12
    addq      $2, %r13
    decq      %r14
    jmp       1b
2:
    xorl      %eax, %eax
    addq      $40, %rsp
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

// int lfm_gelu_erf_bf16(uint16_t *x, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_gelu_erf_bf16)
LFM_SYM(lfm_gelu_erf_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
#endif
    testq     %rdi, %rdi
    jz        9f
    pushq     %rbx
    pushq     %r12
    subq      $56, %rsp
    movq      %rdi, %rbx
    movq      %rsi, %r12
1:
    testq     %r12, %r12
    jz        2f
    movzwl    (%rbx), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    movss     %xmm0, 40(%rsp)     // x
    movl      $0x3f3504f3, %eax   // 1/sqrt(2)
    movd      %eax, %xmm1
    mulss     %xmm1, %xmm0
    call      LFM_CALL(erff)
    movl      $0x3f800000, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0        // 1 + erf
    movl      $0x3f000000, %eax
    movd      %eax, %xmm1         // 0.5
    mulss     40(%rsp), %xmm1     // 0.5*x
    mulss     %xmm1, %xmm0
    BF16_RNE_XMM %xmm0
    movw      %ax, (%rbx)
    addq      $2, %rbx
    decq      %r12
    jmp       1b
2:
    xorl      %eax, %eax
    addq      $56, %rsp
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

// void lfm_residual_half_bf16(uint16_t *r, const uint16_t *h, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_residual_half_bf16)
LFM_SYM(lfm_residual_half_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
#endif
    testq     %rdx, %rdx
    jz        8f
    movl      $0x3f000000, %r10d
    movd      %r10d, %xmm3        // 0.5
1:
    movzwl    (%rsi), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    mulss     %xmm3, %xmm0
    BF16_RNE_XMM %xmm0
    shll      $16, %eax
    movd      %eax, %xmm0
    movzwl    (%rdi), %eax
    shll      $16, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0
    BF16_RNE_XMM %xmm0
    movw      %ax, (%rdi)
    addq      $2, %rdi
    addq      $2, %rsi
    decq      %rdx
    jnz       1b
8:
    ret

// void lfm_add_bf16(uint16_t *r, const uint16_t *h, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_add_bf16)
LFM_SYM(lfm_add_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
#endif
    testq     %rdx, %rdx
    jz        8f
1:
    movzwl    (%rsi), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    movzwl    (%rdi), %eax
    shll      $16, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0
    BF16_RNE_XMM %xmm0
    movw      %ax, (%rdi)
    addq      $2, %rdi
    addq      $2, %rsi
    decq      %rdx
    jnz       1b
8:
    ret

// int lfm_softmax_rows_f32(float *x, uint64_t rows, uint64_t cols)
.p2align 4
.globl LFM_SYM(lfm_softmax_rows_f32)
LFM_SYM(lfm_softmax_rows_f32):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
#endif
    testq     %rdi, %rdi
    jz        9f
    testq     %rdx, %rdx
    jz        9f
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    subq      $64, %rsp
    movq      %rdi, %rbx          // row ptr
    movq      %rsi, %r12          // rows
    movq      %rdx, %r13          // cols
1:
    testq     %r12, %r12
    jz        7f
    // max
    movss     (%rbx), %xmm0
    movq      $1, %rax
2:
    cmpq      %r13, %rax
    jae       3f
    maxss     (%rbx,%rax,4), %xmm0
    incq      %rax
    jmp       2b
3:
    movss     %xmm0, 40(%rsp)     // max
    xorps     %xmm1, %xmm1
    movss     %xmm1, 44(%rsp)     // sum
    xorq      %rax, %rax
4:
    cmpq      %r13, %rax
    jae       5f
    movq      %rax, 48(%rsp)
    movss     (%rbx,%rax,4), %xmm0
    subss     40(%rsp), %xmm0
    call      LFM_CALL(expf)
    movq      48(%rsp), %rax
    movss     %xmm0, (%rbx,%rax,4)
    addss     44(%rsp), %xmm0
    movss     %xmm0, 44(%rsp)
    incq      %rax
    jmp       4b
5:
    movss     44(%rsp), %xmm1
    xorq      %rax, %rax
6:
    cmpq      %r13, %rax
    jae       65f
    movss     (%rbx,%rax,4), %xmm0
    divss     %xmm1, %xmm0
    movss     %xmm0, (%rbx,%rax,4)
    incq      %rax
    jmp       6b
65:
    leaq      (%rbx,%r13,4), %rbx
    decq      %r12
    jmp       1b
7:
    xorl      %eax, %eax
    addq      $64, %rsp
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

// void lfm_sgemm_f32(a, b, c, m, n, k) — C(MxN) = A(MxK).B(KxN), scalar.
.p2align 4
.globl LFM_SYM(lfm_sgemm_f32)
LFM_SYM(lfm_sgemm_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      80(%rsp), %r8
    movq      88(%rsp), %r9
#endif
    // SysV: a=rdi b=rsi c=rdx m=rcx n=r8 k=r9 — all in registers.
    xorq      %r10, %r10          // i
1:
    cmpq      %rcx, %r10
    jae       8f
    xorq      %r11, %r11          // j
2:
    cmpq      %r8, %r11
    jae       5f
    xorps     %xmm0, %xmm0
    movq      %r10, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,4), %r12 // A row
    xorq      %r13, %r13          // p
3:
    cmpq      %r9, %r13
    jae       4f
    movss     (%r12,%r13,4), %xmm1
    movq      %r13, %rax
    imulq     %r8, %rax
    addq      %r11, %rax
    mulss     (%rsi,%rax,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %r13
    jmp       3b
4:
    movq      %r10, %rax
    imulq     %r8, %rax
    addq      %r11, %rax
    movss     %xmm0, (%rdx,%rax,4)
    incq      %r11
    jmp       2b
5:
    incq      %r10
    jmp       1b
8:
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// void lfm_sgemm_nt_f32(a, bt, c, m, n, k) — C = A(MxK) . B(NxK)^T.
.p2align 4
.globl LFM_SYM(lfm_sgemm_nt_f32)
LFM_SYM(lfm_sgemm_nt_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      80(%rsp), %r8
    movq      88(%rsp), %r9
#endif
    // SysV: a=rdi bt=rsi c=rdx m=rcx n=r8 k=r9 — all in registers.
    xorq      %r10, %r10
1:
    cmpq      %rcx, %r10
    jae       8f
    xorq      %r11, %r11
2:
    cmpq      %r8, %r11
    jae       5f
    xorps     %xmm0, %xmm0
    movq      %r10, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,4), %r12
    movq      %r11, %rax
    imulq     %r9, %rax
    leaq      (%rsi,%rax,4), %r13
    xorq      %r14, %r14
3:
    cmpq      %r9, %r14
    jae       4f
    movss     (%r12,%r14,4), %xmm1
    mulss     (%r13,%r14,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %r14
    jmp       3b
4:
    movq      %r10, %rax
    imulq     %r8, %rax
    addq      %r11, %rax
    movss     %xmm0, (%rdx,%rax,4)
    incq      %r11
    jmp       2b
5:
    incq      %r10
    jmp       1b
8:
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// Baseline direct bf16 checkpoint-layout GEMM. Uses only scalar SSE2 integer
// shifts plus f32 arithmetic, so Rosetta can execute it even when its AVX state
// gate rejects the tuned leaf. No operand is packed or widened into memory.
.p2align 4
.globl LFM_SYM(lfm_bf16_gemm_nt_f32_scalar)
LFM_SYM(lfm_bf16_gemm_nt_f32_scalar):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      80(%rsp), %r8
    movq      88(%rsp), %r9
#endif
    xorq      %r10, %r10
1:
    cmpq      %rcx, %r10
    jae       8f
    xorq      %r11, %r11
2:
    cmpq      %r8, %r11
    jae       5f
    xorps     %xmm0, %xmm0
    movq      %r10, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,2), %r12
    movq      %r11, %rax
    imulq     %r9, %rax
    leaq      (%rsi,%rax,2), %r13
    xorq      %r14, %r14
3:
    cmpq      %r9, %r14
    jae       4f
    pxor      %xmm1, %xmm1
    pinsrw    $0, (%r12,%r14,2), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%r14,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %r14
    jmp       3b
4:
    movq      %r10, %rax
    imulq     %r8, %rax
    addq      %r11, %rax
    movss     %xmm0, (%rdx,%rax,4)
    incq      %r11
    jmp       2b
5:
    incq      %r10
    jmp       1b
8:
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// void lfm_dwconv_tap_f32(xpad, w, y, channels, t_out, t_pad, k, stride)
.p2align 4
.globl LFM_SYM(lfm_dwconv_tap_f32)
LFM_SYM(lfm_dwconv_tap_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    pushq     %rbp
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      88(%rsp), %r8       // t_out (40 shadow+ret adj + 6 pushes 48)
    movq      96(%rsp), %r9       // t_pad
    movq      104(%rsp), %r10     // k
    movq      112(%rsp), %r11     // stride
#else
    // SysV: xpad rsi rdx channels t_out(r8) t_pad(r9); k/stride on stack.
    movq      56(%rsp), %r10      // k   (ret 8 + 6 pushes 48)
    movq      64(%rsp), %r11      // stride
#endif
    xorq      %rbx, %rbx          // channel
1:
    cmpq      %rcx, %rbx
    jae       8f
    movq      %rbx, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,4), %r12 // xpad row
    movq      %rbx, %rax
    imulq     %r10, %rax
    leaq      (%rsi,%rax,4), %r13 // w row
    movq      %rbx, %rax
    imulq     %r8, %rax
    leaq      (%rdx,%rax,4), %r14 // y row
    xorq      %r15, %r15          // t
2:
    cmpq      %r8, %r15
    jae       5f
    xorps     %xmm0, %xmm0
    movq      %r15, %rax
    imulq     %r11, %rax          // t*stride
    leaq      (%r12,%rax,4), %rbp // window base = xpad_row + t*stride (rbp free)
    xorq      %rax, %rax          // j
3:
    cmpq      %r10, %rax
    jae       4f
    movss     (%rbp,%rax,4), %xmm1
    mulss     (%r13,%rax,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    jmp       3b
4:
    movss     %xmm0, (%r14,%r15,4)
    incq      %r15
    jmp       2b
5:
    incq      %rbx
    jmp       1b
8:
    popq      %rbp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// Direct-bf16 twin of lfm_dwconv_tap_f32. The source and resident taps remain
// halfwords; each MAC unlift happens in an SSE register.
.p2align 4
.globl LFM_SYM(lfm_dwconv_tap_bf16_f32)
LFM_SYM(lfm_dwconv_tap_bf16_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    pushq     %rbp
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      88(%rsp), %r8
    movq      96(%rsp), %r9
    movq      104(%rsp), %r10
    movq      112(%rsp), %r11
#else
    movq      56(%rsp), %r10
    movq      64(%rsp), %r11
#endif
    xorq      %rbx, %rbx
1:
    cmpq      %rcx, %rbx
    jae       8f
    movq      %rbx, %rax
    imulq     %r9, %rax
    leaq      (%rdi,%rax,2), %r12
    movq      %rbx, %rax
    imulq     %r10, %rax
    leaq      (%rsi,%rax,2), %r13
    movq      %rbx, %rax
    imulq     %r8, %rax
    leaq      (%rdx,%rax,4), %r14
    xorq      %r15, %r15
2:
    cmpq      %r8, %r15
    jae       5f
    xorps     %xmm0, %xmm0
    movq      %r15, %rax
    imulq     %r11, %rax
    leaq      (%r12,%rax,2), %rbp
    xorq      %rax, %rax
3:
    cmpq      %r10, %rax
    jae       4f
    pxor      %xmm1, %xmm1
    pinsrw    $0, (%rbp,%rax,2), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    jmp       3b
4:
    movss     %xmm0, (%r14,%r15,4)
    incq      %r15
    jmp       2b
5:
    incq      %rbx
    jmp       1b
8:
    popq      %rbp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// void lfm_dwconv2d_k3s2_f32(xpad, w, y, channels, h_pad, w_pad, h_out, w_out)
// Scalar params spilled to a stack frame so all callee-saved regs are free for
// pointers/indices (utterance-rate; clarity over register pressure).
.p2align 4
.globl LFM_SYM(lfm_dwconv2d_k3s2_f32)
LFM_SYM(lfm_dwconv2d_k3s2_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    subq      $48, %rsp           // frame: [0]=channels [8]=h_pad [16]=w_pad
                                  //        [24]=h_out    [32]=w_out
    // Offset base = entry_arg_offset + 40 (5 pushes) + 48 (frame).
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx          // channels
    movq      128(%rsp), %r8      // h_pad (entry 40 + 88)
    movq      136(%rsp), %r9      // w_pad (entry 48 + 88)
    movq      144(%rsp), %rax     // h_out (entry 56 + 88)
    movq      %rax, 24(%rsp)
    movq      152(%rsp), %rax     // w_out (entry 64 + 88)
    movq      %rax, 32(%rsp)
#else
    movq      96(%rsp), %rax      // h_out (entry 8 + 88)
    movq      %rax, 24(%rsp)
    movq      104(%rsp), %rax     // w_out (entry 16 + 88)
    movq      %rax, 32(%rsp)
    // channels=rcx h_pad=r8 w_pad=r9 in registers
#endif
    movq      %rcx, 0(%rsp)
    movq      %r8, 8(%rsp)
    movq      %r9, 16(%rsp)
    xorq      %rbx, %rbx          // channel
1:
    cmpq      0(%rsp), %rbx
    jae       8f
    // xpad plane = xpad + channel * h_pad * w_pad
    movq      8(%rsp), %rax
    imulq     16(%rsp), %rax
    imulq     %rbx, %rax
    leaq      (%rdi,%rax,4), %r12
    // w(9) = w + channel*9
    leaq      (%rbx,%rbx,8), %rax
    leaq      (%rsi,%rax,4), %r13
    // y plane = y + channel * h_out * w_out
    movq      24(%rsp), %rax
    imulq     32(%rsp), %rax
    imulq     %rbx, %rax
    leaq      (%rdx,%rax,4), %r14
    xorq      %r15, %r15          // oy
2:
    cmpq      24(%rsp), %r15
    jae       7f
    xorq      %rcx, %rcx          // ox
3:
    cmpq      32(%rsp), %rcx
    jae       6f
    xorps     %xmm0, %xmm0
    xorq      %r8, %r8            // dy
4:
    cmpq      $3, %r8
    jae       5f
    // window row base = xpad_plane + ((2*oy+dy)*w_pad + 2*ox) * 4
    leaq      (%r15,%r15), %rax   // 2*oy
    addq      %r8, %rax           // + dy
    imulq     16(%rsp), %rax      // * w_pad
    leaq      (%rcx,%rcx), %r9    // 2*ox
    addq      %r9, %rax
    leaq      (%r12,%rax,4), %r10 // window base
    leaq      (%r8,%r8,2), %rax   // dy*3
    movss     (%r10), %xmm1
    mulss     (%r13,%rax,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    movss     4(%r10), %xmm1
    mulss     (%r13,%rax,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    movss     8(%r10), %xmm1
    mulss     (%r13,%rax,4), %xmm1
    addss     %xmm1, %xmm0
    incq      %r8
    jmp       4b
5:
    movq      %r15, %rax
    imulq     32(%rsp), %rax
    addq      %rcx, %rax
    movss     %xmm0, (%r14,%rax,4)
    incq      %rcx
    jmp       3b
6:
    incq      %r15
    jmp       2b
7:
    incq      %rbx
    jmp       1b
8:
    addq      $48, %rsp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// Stem k3/s2: one padded bf16 input plane, resident (channels,1,3,3) bf16
// taps, and channel-major f32 output. Both bf16 operands are unlifted at the
// MAC; the checkpoint address may be byte-unaligned.
.p2align 4
.globl LFM_SYM(lfm_conv2d_stem_k3s2_bf16_f32)
LFM_SYM(lfm_conv2d_stem_k3s2_bf16_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    subq      $48, %rsp
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      128(%rsp), %r8
    movq      136(%rsp), %r9
    movq      144(%rsp), %rax
    movq      %rax, 24(%rsp)
    movq      152(%rsp), %rax
    movq      %rax, 32(%rsp)
#else
    movq      96(%rsp), %rax
    movq      %rax, 24(%rsp)
    movq      104(%rsp), %rax
    movq      %rax, 32(%rsp)
#endif
    movq      %rcx, 0(%rsp)
    movq      %r8, 8(%rsp)
    movq      %r9, 16(%rsp)
    xorq      %rbx, %rbx
1:
    cmpq      0(%rsp), %rbx
    jae       8f
    movq      %rdi, %r12          // the stem has one shared input plane
    leaq      (%rbx,%rbx,8), %rax
    leaq      (%rsi,%rax,2), %r13
    movq      24(%rsp), %rax
    imulq     32(%rsp), %rax
    imulq     %rbx, %rax
    leaq      (%rdx,%rax,4), %r14
    xorq      %r15, %r15
2:
    cmpq      24(%rsp), %r15
    jae       7f
    xorq      %rcx, %rcx
3:
    cmpq      32(%rsp), %rcx
    jae       6f
    xorps     %xmm0, %xmm0
    xorq      %r8, %r8
4:
    cmpq      $3, %r8
    jae       5f
    leaq      (%r15,%r15), %rax
    addq      %r8, %rax
    imulq     16(%rsp), %rax
    leaq      (%rcx,%rcx), %r9
    addq      %r9, %rax
    leaq      (%r12,%rax,2), %r10
    leaq      (%r8,%r8,2), %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, (%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, 2(%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, 4(%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %r8
    jmp       4b
5:
    movq      %r15, %rax
    imulq     32(%rsp), %rax
    addq      %rcx, %rax
    movss     %xmm0, (%r14,%rax,4)
    incq      %rcx
    jmp       3b
6:
    incq      %r15
    jmp       2b
7:
    incq      %rbx
    jmp       1b
8:
    addq      $48, %rsp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// Direct-bf16 per-channel k3/s2 twin. Input padding and checkpoint taps stay
// bf16, while the destination remains f32 for the caller's bias/round ladder.
.p2align 4
.globl LFM_SYM(lfm_dwconv2d_k3s2_bf16_f32)
LFM_SYM(lfm_dwconv2d_k3s2_bf16_f32):
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    subq      $48, %rsp
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      128(%rsp), %r8
    movq      136(%rsp), %r9
    movq      144(%rsp), %rax
    movq      %rax, 24(%rsp)
    movq      152(%rsp), %rax
    movq      %rax, 32(%rsp)
#else
    movq      96(%rsp), %rax
    movq      %rax, 24(%rsp)
    movq      104(%rsp), %rax
    movq      %rax, 32(%rsp)
#endif
    movq      %rcx, 0(%rsp)
    movq      %r8, 8(%rsp)
    movq      %r9, 16(%rsp)
    xorq      %rbx, %rbx
1:
    cmpq      0(%rsp), %rbx
    jae       8f
    movq      8(%rsp), %rax
    imulq     16(%rsp), %rax
    imulq     %rbx, %rax
    leaq      (%rdi,%rax,2), %r12
    leaq      (%rbx,%rbx,8), %rax
    leaq      (%rsi,%rax,2), %r13
    movq      24(%rsp), %rax
    imulq     32(%rsp), %rax
    imulq     %rbx, %rax
    leaq      (%rdx,%rax,4), %r14
    xorq      %r15, %r15
2:
    cmpq      24(%rsp), %r15
    jae       7f
    xorq      %rcx, %rcx
3:
    cmpq      32(%rsp), %rcx
    jae       6f
    xorps     %xmm0, %xmm0
    xorq      %r8, %r8
4:
    cmpq      $3, %r8
    jae       5f
    leaq      (%r15,%r15), %rax
    addq      %r8, %rax
    imulq     16(%rsp), %rax
    leaq      (%rcx,%rcx), %r9
    addq      %r9, %rax
    leaq      (%r12,%rax,2), %r10
    leaq      (%r8,%r8,2), %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, (%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, 2(%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %rax
    pxor      %xmm1, %xmm1
    pinsrw    $0, 4(%r10), %xmm1
    pslld     $16, %xmm1
    pxor      %xmm2, %xmm2
    pinsrw    $0, (%r13,%rax,2), %xmm2
    pslld     $16, %xmm2
    mulss     %xmm2, %xmm1
    addss     %xmm1, %xmm0
    incq      %r8
    jmp       4b
5:
    movq      %r15, %rax
    imulq     32(%rsp), %rax
    addq      %rcx, %rax
    movss     %xmm0, (%r14,%rax,4)
    incq      %rcx
    jmp       3b
6:
    incq      %r15
    jmp       2b
7:
    incq      %rbx
    jmp       1b
8:
    addq      $48, %rsp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret

// void lfm_bias_rows_f32(float *y, const uint16_t *bias, uint64_t rows, uint64_t n)
.p2align 4
.globl LFM_SYM(lfm_bias_rows_f32)
LFM_SYM(lfm_bias_rows_f32):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#endif
    testq     %rsi, %rsi
    jz        8f
    xorq      %r10, %r10          // row
1:
    cmpq      %rdx, %r10
    jae       8f
    xorq      %r11, %r11
2:
    cmpq      %rcx, %r11
    jae       3f
    movzwl    (%rsi,%r11,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm1
    movss     (%rdi), %xmm0
    addss     %xmm1, %xmm0
    movss     %xmm0, (%rdi)
    addq      $4, %rdi
    incq      %r11
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
    ret

// void lfm_bias_bcast_f32(float *y, const uint16_t *bias, uint64_t channels, uint64_t t)
.p2align 4
.globl LFM_SYM(lfm_bias_bcast_f32)
LFM_SYM(lfm_bias_bcast_f32):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#endif
    testq     %rsi, %rsi
    jz        8f
    xorq      %r10, %r10          // channel
1:
    cmpq      %rdx, %r10
    jae       8f
    movzwl    (%rsi,%r10,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm1
    xorq      %r11, %r11
2:
    cmpq      %rcx, %r11
    jae       3f
    movss     (%rdi), %xmm0
    addss     %xmm1, %xmm0
    movss     %xmm0, (%rdi)
    addq      $4, %rdi
    incq      %r11
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
    ret

// void lfm_add_scale_f32(float *acc, const float *addend, uint64_t rows,
//                        uint64_t cols, uint64_t addend_stride, float scale)
.p2align 4
.globl LFM_SYM(lfm_add_scale_f32)
LFM_SYM(lfm_add_scale_f32):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      40(%rsp), %r8
    movss     48(%rsp), %xmm0
#endif
    xorq      %r10, %r10          // row
1:
    cmpq      %rdx, %r10
    jae       8f
    movq      %r10, %rax
    imulq     %r8, %rax
    leaq      (%rsi,%rax,4), %r11 // addend row
    xorq      %rax, %rax
2:
    cmpq      %rcx, %rax
    jae       3f
    movss     (%rdi), %xmm1
    addss     (%r11,%rax,4), %xmm1
    mulss     %xmm0, %xmm1
    movss     %xmm1, (%rdi)
    addq      $4, %rdi
    incq      %rax
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
    ret

// void lfm_add_bias_hd_bf16(x, bias, y, t, heads, dk)
.p2align 4
.globl LFM_SYM(lfm_add_bias_hd_bf16)
LFM_SYM(lfm_add_bias_hd_bf16):
    pushq     %rbx
    pushq     %r12
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      56(%rsp), %r8       // heads (40 + 2*8)
    movq      64(%rsp), %r9       // dk
#else
    // SysV: x=rdi bias=rsi y=rdx t=rcx heads=r8 dk=r9
#endif
    movq      %r8, %rbx
    imulq     %r9, %rbx           // D = heads*dk
    xorq      %r10, %r10          // t
1:
    cmpq      %rcx, %r10
    jae       8f
    xorq      %r11, %r11          // d over D
2:
    cmpq      %rbx, %r11
    jae       3f
    movzwl    (%rdi), %eax
    shll      $16, %eax
    movd      %eax, %xmm0
    movzwl    (%rsi,%r11,2), %eax
    shll      $16, %eax
    movd      %eax, %xmm1
    addss     %xmm1, %xmm0
    BF16_RNE_XMM %xmm0, %r12d
    movw      %ax, (%rdx)
    addq      $2, %rdi
    addq      $2, %rdx
    incq      %r11
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
    popq      %r12
    popq      %rbx
    ret

// int lfm_pe_build_bf16(const float *div, uint64_t half, uint64_t t, uint16_t *out)
.p2align 4
.globl LFM_SYM(lfm_pe_build_bf16)
LFM_SYM(lfm_pe_build_bf16):
#ifdef _WIN32
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#endif
    testq     %rdi, %rdi
    jz        9f
    testq     %rsi, %rsi
    jz        9f
    testq     %rdx, %rdx
    jz        9f
    pushq     %rbx
    pushq     %r12
    pushq     %r13
    pushq     %r14
    pushq     %r15
    subq      $40, %rsp
    movq      %rdi, %rbx          // div
    movq      %rsi, %r12          // half
    movq      %rdx, %r13          // t
    movq      %rcx, %r14          // out
    leaq      -1(%r13,%r13), %r15 // P = 2t-1
    xorq      %rdi, %rdi          // i (reuse rdi as loop counter)
1:
    cmpq      %r15, %rdi
    jae       7f
    xorq      %rsi, %rsi          // j
2:
    cmpq      %r12, %rsi
    jae       6f
    // pos = (float)(t-1-i)
    movq      %r13, %rax
    decq      %rax
    subq      %rdi, %rax
    pxor      %xmm0, %xmm0
    cvtsi2ssq %rax, %xmm0
    mulss     (%rbx,%rsi,4), %xmm0 // angle
    movss     %xmm0, 16(%rsp)
    movq      %rdi, 24(%rsp)
    movq      %rsi, 32(%rsp)
    call      LFM_CALL(sinf)
    BF16_RNE_XMM %xmm0
    movq      24(%rsp), %rdi
    movq      32(%rsp), %rsi
    // out[i*2*half + 2j] = sin
    movq      %rdi, %rcx
    imulq     %r12, %rcx
    shlq      $1, %rcx
    leaq      (%rcx,%rsi,2), %rcx
    movw      %ax, (%r14,%rcx,2)
    movss     16(%rsp), %xmm0
    call      LFM_CALL(cosf)
    BF16_RNE_XMM %xmm0
    movq      24(%rsp), %rdi
    movq      32(%rsp), %rsi
    movq      %rdi, %rcx
    imulq     %r12, %rcx
    shlq      $1, %rcx
    leaq      1(%rcx,%rsi,2), %rcx
    movw      %ax, (%r14,%rcx,2)
    incq      %rsi
    jmp       2b
6:
    incq      %rdi
    jmp       1b
7:
    xorl      %eax, %eax
    addq      $40, %rsp
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
9:
    movl      $-22, %eax
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
