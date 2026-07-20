// flashkern_rope.S — RoPE cos/sin TABLE GENERATION (x86-64).
//
// x86-64 twin of kernels/aarch64/flashkern_rope.S. The computation is identical
// and was verified instruction-for-instruction; only register preservation
// differs (see ABI note). This file builds the tables only — the rotation that
// consumes them is `lfm_bf16_rope_neox` in flashkern_math.S. Precision seam:
// tables are generated here in F32 and narrowed to BF16 before that read.
//
// WHAT IS COMPUTED, per position `p` and pair index `i` in [0, dim/2):
//
//     exponent = (2*i) / dim
//     freq     = powf(theta, exponent)
//     angle    = p / freq                 // i.e. p * theta^-(2i/dim)
//     cos[p][i] = cosf(angle)
//     sin[p][i] = sinf(angle)
//
// Both planes are [n_pos][dim/2] row-major F32. `angle` DIVIDES by the power
// rather than multiplying a precomputed reciprocal — those differ in the last
// bits and the division is the contract.
//
// LIBM DEPENDENCY — the one thing to know about this file:
//
// It calls `powf`, `cosf`, `sinf` from the platform libm — hence the frame,
// since these are not leaf routines. Those are NOT bit-identical across platforms:
// Apple libm, glibc and musl may each return different last bits. The tables are
// reproducible on a given platform but not guaranteed across them, so any
// end-to-end byte oracle is implicitly platform-tied through this file.
//
// THREE files in the kernel set call libm and are therefore platform-tied:
// this one (powf/cosf/sinf), flashkern_frontend.S (logf), and
// flashkern_conformer.S (expf/erff/sinf/cosf — the largest user by far).
// math/prng/sampler/sesame are libm-free and bit-identical everywhere. If a
// hash differs between machines with everything else equal, those three are
// where to look.
//
// ABI NOTE (the only structural difference from AArch64): neither SysV nor
// Win64 preserves xmm registers across a call, so `theta` and `angle` are
// spilled to the stack slots LFM_THETA/LFM_ANGLE around the libm calls. The
// AArch64 twin keeps them in callee-saved d8/d9 instead. Same values, same
// order; only the holding location differs.
//
// VALIDATION (no silent coercion):
//   -EINVAL (-22)     null plane, odd `dim`, or theta <= 0 / NaN
//   -EOVERFLOW (-75)  `lfm_rope_range_f32` only: pos0 + (n_pos-1) wraps
//
// INVARIANT, and it is gated: `lfm_rope_range_f32` with pos0 == 0 must produce
// bit-identical output to `lfm_rope_table_f32`. Proven by
// `absolute_rope_range_is_bit_exact_with_the_zero_based_table` in
// tests/native_context_rollover.rs — what keeps KV rollover from silently
// re-phasing the cache against a differently-built table.

#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text

// int lfm_rope_table_f32(uint64_t n_pos, uint64_t dim,
//                        float *cos, float *sin, float theta)
// Zero-based table: positions 0 .. n_pos-1. Returns 0, or -EINVAL.
.p2align 4
.globl LFM_SYM(lfm_rope_table_f32)
LFM_PRIVATE(LFM_SYM(lfm_rope_table_f32))
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

// int lfm_rope_range_f32(uint64_t pos0, uint64_t n_pos, uint64_t dim,
//                        float *cos, float *sin, float theta)
// Absolute-position window: positions pos0 .. pos0+n_pos-1. This is the KV
// rollover path — once the context window slides, entries must be re-phased at
// their ABSOLUTE positions rather than re-based to zero, or the cache and the
// incoming tokens disagree about phase.
// Returns 0, -EINVAL, or -EOVERFLOW when pos0 + (n_pos-1) wraps.
.p2align 4
.globl LFM_SYM(lfm_rope_range_f32)
LFM_PRIVATE(LFM_SYM(lfm_rope_range_f32))
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
