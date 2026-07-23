#include "lfm_asm_visibility.h"
#include "../../src/detokenizer/lfm_detokenizer_kernels.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

// flashkern_detokenizer.S — x86-64 leaves of the LFM2.5 audio detokenizer.
//
// x86-64 twin of aarch64/flashkern_detokenizer.S — read that file's header and
// per-symbol blocks for the full port rationale (oracle: vendored
// reference/detokenizer.py + audio_detokenizer/config.json, checkpoint
// c362a062, weights sha256 d294aef1...). Same 19 symbols, same argument-block
// ABI (lfm_detokenizer_kernels.h), same setup-only libm surface (powf in the
// RoPE-inverse builder, cos/sin in the irfft-basis builder; every hot-pass
// leaf is libm-free).
//
// CROSS-ARCH BIT-IDENTITY DOES NOT HOLD FOR THIS FILE — stated here so nobody
// hunts a phantom bug: this is baseline SSE2 with ZERO fused operations
// (mulps + addps = two roundings per multiply-add), while the AArch64 twin's
// committed order is fused fmla (one rounding). The reduction TREES match
// (two accumulators, 8-float stride, pairwise collapse — haddps here, faddp
// there); the fusion does not, because SSE2 has no FMA and NEON has nothing
// else. Consequence: the real-checkpoint trace gate
// (docs/AUDIO_DETOKENIZER_PORT.md) is an ARCH-SCOPED contract — each
// architecture locks its own stream. A cross-arch PCM hash comparison is
// expected to differ in the last bits and is not a regression.
//
// NaN semantics in max match the AArch64 twin exactly (a NaN at [0] sticks,
// later NaNs never replace the running max — corruption is the trace gate's
// job). Win64/SysV dual ABI like every sibling leaf.

.text

// x86_64 twin of the LFM2.5 detokenizer leaf family. Hot passes own complete
// arithmetic tiles and return only after terminal destination writes. The two
// setup-only table constructors call platform powf/cos/sin; per-frame leaves
// are libm-free. SysV and Win64 guards preserve one semantic ABI.

.p2align 4
// void lfm_detok_copy_f32(const float *src, float *dst, uint64_t count)
// Plain view copy (movups). See AArch64 twin for role.
.globl LFM_SYM(lfm_detok_copy_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_copy_f32))
LFM_SYM(lfm_detok_copy_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %rax
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
#endif
1:  cmpq    $16, %rax
    jb      2f
    movups  0(%r10), %xmm0
    movups  16(%r10), %xmm1
    movups  32(%r10), %xmm2
    movups  48(%r10), %xmm3
    movups  %xmm0, 0(%r11)
    movups  %xmm1, 16(%r11)
    movups  %xmm2, 32(%r11)
    movups  %xmm3, 48(%r11)
    addq    $64, %r10
    addq    $64, %r11
    subq    $16, %rax
    jmp     1b
2:  testq   %rax, %rax
    jz      9f
3:  movss   (%r10), %xmm0
    movss   %xmm0, (%r11)
    addq    $4, %r10
    addq    $4, %r11
    decq    %rax
    jnz     3b
9:  ret

.p2align 4
// void lfm_detok_add_f32(float *accum, const float *addend, uint64_t count)
// accum[i] += addend[i] — residual adds. Unfused by ISA (header note).
.globl LFM_SYM(lfm_detok_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_add_f32))
LFM_SYM(lfm_detok_add_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %rax
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
#endif
1:  cmpq    $4, %rax
    jb      2f
    movups  (%r10), %xmm0
    movups  (%r11), %xmm1
    addps   %xmm1, %xmm0
    movups  %xmm0, (%r10)
    addq    $16, %r10
    addq    $16, %r11
    subq    $4, %rax
    jmp     1b
2:  testq   %rax, %rax
    jz      9f
3:  movss   (%r10), %xmm0
    addss   (%r11), %xmm0
    movss   %xmm0, (%r10)
    addq    $4, %r10
    addq    $4, %r11
    decq    %rax
    jnz     3b
9:  ret

.p2align 4
// void lfm_detok_embed_f32(args: LFM_DETOK_EMBED_* block)
// FusedEmbedding: mean of 8 offset codebook embeddings per frame
// (detokenizer.py line 24). Same gather-accumulate-scale as the twin.
.globl LFM_SYM(lfm_detok_embed_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_embed_f32))
LFM_SYM(lfm_detok_embed_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_EMBED_OUTPUT(%r10), %r11
    movq    LFM_DETOK_EMBED_COUNT(%r10), %r9
    xorq    %r8, %r8
    movl    $0x3e000000, %eax
    movd    %eax, %xmm2
    shufps  $0, %xmm2, %xmm2
1:  testq   %r9, %r9
    jz      9f
    xorps   %xmm0, %xmm0
    movq    0(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    8(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    16(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    24(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    32(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    40(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    48(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    movq    56(%r10), %rax
    movups  (%rax,%r8), %xmm1
    addps   %xmm1, %xmm0
    mulps   %xmm2, %xmm0
    movups  %xmm0, 0(%r11)
    movups  %xmm0, 2048(%r11)
    movups  %xmm0, 4096(%r11)
    movups  %xmm0, 6144(%r11)
    movups  %xmm0, 8192(%r11)
    movups  %xmm0, 10240(%r11)
    addq    $16, %r8
    addq    $16, %r11
    subq    $4, %r9
    jmp     1b
9:  ret

.p2align 4
// void lfm_detok_rms_f32(args: LFM_DETOK_RMS_* block)
// Backbone RMSNorm (block_norm_eps 1e-5). Same op order as the twin:
// sumsq -> /dim -> +eps -> rsqrt via exact sqrtss+divss, never rsqrtss.
.globl LFM_SYM(lfm_detok_rms_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rms_f32))
LFM_SYM(lfm_detok_rms_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_RMS_INPUT(%r10), %r11
    movq    LFM_DETOK_RMS_WEIGHT(%r10), %r8
    movq    LFM_DETOK_RMS_OUTPUT(%r10), %r9
    movq    LFM_DETOK_RMS_COLUMNS(%r10), %rcx
    movq    %r11, %rdx
    movq    %rcx, %rax
    xorps   %xmm0, %xmm0
    xorps   %xmm1, %xmm1
1:  testq   %rax, %rax
    jz      2f
    movups  0(%rdx), %xmm2
    movups  16(%rdx), %xmm3
    mulps   %xmm2, %xmm2
    mulps   %xmm3, %xmm3
    addps   %xmm2, %xmm0
    addps   %xmm3, %xmm1
    addq    $32, %rdx
    subq    $8, %rax
    jmp     1b
2:  addps   %xmm1, %xmm0
    haddps  %xmm0, %xmm0
    haddps  %xmm0, %xmm0
    pxor    %xmm2, %xmm2
    cvtsi2ssq %rcx, %xmm2
    divss   %xmm2, %xmm0
    addss   LFM_DETOK_RMS_EPSILON(%r10), %xmm0
    sqrtss  %xmm0, %xmm0
    movl    $0x3f800000, %eax
    movd    %eax, %xmm2
    divss   %xmm0, %xmm2
    shufps  $0, %xmm2, %xmm2
3:  testq   %rcx, %rcx
    jz      9f
    movups  (%r11), %xmm0
    movups  (%r8), %xmm1
    mulps   %xmm1, %xmm0
    mulps   %xmm2, %xmm0
    movups  %xmm0, (%r9)
    addq    $16, %r11
    addq    $16, %r8
    addq    $16, %r9
    subq    $4, %rcx
    jmp     3b
9:  ret

.p2align 4
// void lfm_detok_swiglu_f32(gate*, up*, exp_neg_gate*, count)
// out = gate * up / (1 + exp(-gate)); exp precomputed by the C++ owner.
// Pure mul/div/add here — hot-pass libm-free rule.
.globl LFM_SYM(lfm_detok_swiglu_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_swiglu_f32))
LFM_SYM(lfm_detok_swiglu_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %r8
    movq    %r9, %rax
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %r8
    movq    %rcx, %rax
#endif
    movl    $0x3f800000, %ecx
    movd    %ecx, %xmm4
    shufps  $0, %xmm4, %xmm4
1:  testq   %rax, %rax
    jz      9f
    movups  (%r10), %xmm0
    movups  (%r11), %xmm1
    movups  (%r8), %xmm2
    addps   %xmm4, %xmm2
    movaps  %xmm4, %xmm3
    divps   %xmm2, %xmm3
    mulps   %xmm3, %xmm0
    mulps   %xmm1, %xmm0
    movups  %xmm0, (%r10)
    addq    $16, %r10
    addq    $16, %r11
    addq    $16, %r8
    subq    $4, %rax
    jmp     1b
9:  ret

.p2align 4
// float lfm_detok_dot32_scaled_f32(q*, k*, scale)
// One attention score over exactly 32 floats (head_dim). Two xmm
// accumulators, 8-float stride, haddps x2 collapse, then *scale. Same tree
// as the twin's fmla/faddp — minus the fusion (header note).
.globl LFM_SYM(lfm_detok_dot32_scaled_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_dot32_scaled_f32))
LFM_SYM(lfm_detok_dot32_scaled_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movaps  %xmm2, %xmm5
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movaps  %xmm0, %xmm5
#endif
    xorps   %xmm0, %xmm0
    xorps   %xmm1, %xmm1
    movl    $4, %eax
1:  movups  0(%r10), %xmm2
    movups  16(%r10), %xmm3
    mulps   0(%r11), %xmm2
    mulps   16(%r11), %xmm3
    addps   %xmm2, %xmm0
    addps   %xmm3, %xmm1
    addq    $32, %r10
    addq    $32, %r11
    decl    %eax
    jnz     1b
    addps   %xmm1, %xmm0
    haddps  %xmm0, %xmm0
    haddps  %xmm0, %xmm0
    mulss   %xmm5, %xmm0
    ret

.p2align 4
// float lfm_detok_max_f32(x*, count) — softmax guard max, scalar.
// ucomiss + jbe: NaN at [0] sticks, later NaNs never win (matches twin;
// deliberate — the trace gate owns corruption detection).
.globl LFM_SYM(lfm_detok_max_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_max_f32))
LFM_SYM(lfm_detok_max_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
#endif
    movss   (%r10), %xmm0
    addq    $4, %r10
    decq    %r11
1:  testq   %r11, %r11
    jz      9f
    movss   (%r10), %xmm1
    ucomiss %xmm0, %xmm1
    jbe     2f
    movaps  %xmm1, %xmm0
2:  addq    $4, %r10
    decq    %r11
    jmp     1b
9:  ret

.p2align 4
// void lfm_detok_subtract_f32(x*, count, value) — x[i] -= value.
// Softmax max-subtraction before the C++ owner exponentiates.
.globl LFM_SYM(lfm_detok_subtract_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_subtract_f32))
LFM_SYM(lfm_detok_subtract_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movaps  %xmm2, %xmm0
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
#endif
    shufps  $0, %xmm0, %xmm0
1:  cmpq    $4, %r11
    jb      2f
    movups  (%r10), %xmm1
    subps   %xmm0, %xmm1
    movups  %xmm1, (%r10)
    addq    $16, %r10
    subq    $4, %r11
    jmp     1b
2:  testq   %r11, %r11
    jz      9f
3:  movss   (%r10), %xmm1
    subss   %xmm0, %xmm1
    movss   %xmm1, (%r10)
    addq    $4, %r10
    decq    %r11
    jnz     3b
9:  ret

.p2align 4
// float lfm_detok_sum_f32(x*, count) — softmax denominator.
// Vectorized-pairwise (addps + haddps) IS the committed order here; the
// frontend's strictly-sequential rule belongs to a different oracle.
.globl LFM_SYM(lfm_detok_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_sum_f32))
LFM_SYM(lfm_detok_sum_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
#endif
    xorps   %xmm0, %xmm0
1:  cmpq    $4, %r11
    jb      2f
    movups  (%r10), %xmm1
    addps   %xmm1, %xmm0
    addq    $16, %r10
    subq    $4, %r11
    jmp     1b
2:  haddps  %xmm0, %xmm0
    haddps  %xmm0, %xmm0
    testq   %r11, %r11
    jz      9f
3:  addss   (%r10), %xmm0
    addq    $4, %r10
    decq    %r11
    jnz     3b
9:  ret

.p2align 4
// void lfm_detok_normalize_f32(x*, count, sum) — softmax normalize.
// Reciprocal computed ONCE then multiplied (contrast frontend norm_apply,
// which divides per element — each is its own reference's committed order).
.globl LFM_SYM(lfm_detok_normalize_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_normalize_f32))
LFM_SYM(lfm_detok_normalize_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movaps  %xmm2, %xmm0
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
#endif
    movl    $0x3f800000, %eax
    movd    %eax, %xmm2
    divss   %xmm0, %xmm2
    movaps  %xmm2, %xmm0
    shufps  $0, %xmm0, %xmm0
1:  cmpq    $4, %r11
    jb      2f
    movups  (%r10), %xmm1
    mulps   %xmm0, %xmm1
    movups  %xmm1, (%r10)
    addq    $16, %r10
    subq    $4, %r11
    jmp     1b
2:  testq   %r11, %r11
    jz      9f
3:  movss   (%r10), %xmm1
    mulss   %xmm0, %xmm1
    movss   %xmm1, (%r10)
    addq    $4, %r10
    decq    %r11
    jnz     3b
9:  ret

.p2align 4
// void lfm_detok_rope_angles_f32(inv_freq*, out*, count, position)
// out[i] = inv_freq[i] * (float)position; cos/sin taken by the C++ owner.
.globl LFM_SYM(lfm_detok_rope_angles_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_angles_f32))
LFM_SYM(lfm_detok_rope_angles_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %rax
    pxor    %xmm0, %xmm0
    cvtsi2ssq %r9, %xmm0
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
    pxor    %xmm0, %xmm0
    cvtsi2ssq %rcx, %xmm0
#endif
    shufps  $0, %xmm0, %xmm0
1:  testq   %rax, %rax
    jz      9f
    movups  (%r10), %xmm1
    mulps   %xmm0, %xmm1
    movups  %xmm1, (%r11)
    addq    $16, %r10
    addq    $16, %r11
    subq    $4, %rax
    jmp     1b
9:  ret

.p2align 4
// void lfm_detok_rope_f32(head*, cos*, sin*) — rotary, NeoX split-half
// (i vs i+16 over one 32-float head), full F32, no BF16 rounding: the
// detokenizer reference runs torch F32 (see twin for the faithfulness note).
.globl LFM_SYM(lfm_detok_rope_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_f32))
LFM_SYM(lfm_detok_rope_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %rax
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
#endif
    movl    $4, %ecx
1:  movups  0(%r10), %xmm0
    movups  64(%r10), %xmm1
    movups  (%r11), %xmm2
    movups  (%rax), %xmm3
    movaps  %xmm0, %xmm4
    mulps   %xmm2, %xmm4
    movaps  %xmm1, %xmm5
    mulps   %xmm3, %xmm5
    subps   %xmm5, %xmm4
    mulps   %xmm2, %xmm1
    mulps   %xmm3, %xmm0
    addps   %xmm0, %xmm1
    movups  %xmm4, 0(%r10)
    movups  %xmm1, 64(%r10)
    addq    $16, %r10
    addq    $16, %r11
    addq    $16, %rax
    decl    %ecx
    jnz     1b
    ret

.p2align 4
// int lfm_detok_rope_inverse_f32(out*, count, head_dim, theta)
// SETUP-ONLY: out[i] = theta^(-2i/head_dim) via libm powf (platform-tied
// builder #1, header note). Returns 0 / -EINVAL.
.globl LFM_SYM(lfm_detok_rope_inverse_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_inverse_f32))
LFM_SYM(lfm_detok_rope_inverse_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r9, %rax
    movaps  %xmm2, %xmm0
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
#endif
    testq   %r10, %r10
    jz      8f
    testq   %r11, %r11
    jz      8f
    testq   %rax, %rax
    jz      8f
    xorps   %xmm1, %xmm1
    ucomiss %xmm1, %xmm0
    jbe     8f
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $40, %rsp
    movq    %r10, %r12
    movq    %r11, %r13
    movq    %rax, %r14
    movss   %xmm0, 32(%rsp)
    xorq    %rbx, %rbx
1:  pxor    %xmm1, %xmm1
    cvtsi2ssq %rbx, %xmm1
    addss   %xmm1, %xmm1
    movl    $0x80000000, %eax
    movd    %eax, %xmm2
    xorps   %xmm2, %xmm1
    pxor    %xmm2, %xmm2
    cvtsi2ssq %r14, %xmm2
    divss   %xmm2, %xmm1
    movss   32(%rsp), %xmm0
    call    LFM_CALL(powf)
    movss   %xmm0, (%r12)
    addq    $4, %r12
    incq    %rbx
    cmpq    %r13, %rbx
    jb      1b
    xorl    %eax, %eax
    addq    $40, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
8:  movl    $-22, %eax
    ret

.p2align 4
// int lfm_detok_ifft_basis_f32(basis*, bins, n_fft)
// SETUP-ONLY: the one-sided irfft basis (torch.fft.irfft, norm="backward")
// as [real|imag] GEMM planes; F64 construction with libm cos/sin, doubling
// 2 except DC/Nyquist, -sin for the conjugate side, /n_fft, narrowed once.
// Platform-tied builder #2. Guards bins == n_fft/2 + 1. See twin for math.
.globl LFM_SYM(lfm_detok_ifft_basis_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_ifft_basis_f32))
LFM_SYM(lfm_detok_ifft_basis_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    movq    %rdx, %r11
    movq    %r8, %rax
#else
    movq    %rdi, %r10
    movq    %rsi, %r11
    movq    %rdx, %rax
#endif
    testq   %r10, %r10
    jz      8f
    testq   %r11, %r11
    jz      8f
    testq   %rax, %rax
    jz      8f
    movq    %rax, %rcx
    shrq    $1, %rcx
    incq    %rcx
    cmpq    %rcx, %r11
    jne     8f
    pushq   %rbp
    pushq   %rbx
    pushq   %r12
    pushq   %r13
    pushq   %r14
    pushq   %r15
    subq    $72, %rsp
    movq    %r10, %r12
    movq    %r11, %r14
    movq    %rax, %r15
    imulq   %r15, %r11
    leaq    (%r12,%r11,4), %r13
    movabsq $0x401921fb54442d18, %rax
    movq    %rax, 32(%rsp)
    pxor    %xmm0, %xmm0
    cvtsi2sdq %r15, %xmm0
    movabsq $0x3ff0000000000000, %rax
    movq    %rax, %xmm1
    divsd   %xmm0, %xmm1
    movsd   %xmm1, 40(%rsp)
    xorq    %rbp, %rbp
1:  movabsq $0x4000000000000000, %rax
    movq    %rax, %xmm0
    testq   %rbp, %rbp
    jz      2f
    leaq    (%rbp,%rbp), %rax
    cmpq    %r15, %rax
    jne     3f
2:  movabsq $0x3ff0000000000000, %rax
    movq    %rax, %xmm0
3:  movsd   %xmm0, 56(%rsp)
    xorq    %rbx, %rbx
4:  movq    %rbp, %rax
    imulq   %rbx, %rax
    pxor    %xmm0, %xmm0
    cvtsi2sdq %rax, %xmm0
    mulsd   32(%rsp), %xmm0
    pxor    %xmm1, %xmm1
    cvtsi2sdq %r15, %xmm1
    divsd   %xmm1, %xmm0
    movsd   %xmm0, 48(%rsp)
    call    LFM_CALL(cos)
    mulsd   56(%rsp), %xmm0
    mulsd   40(%rsp), %xmm0
    cvtsd2ss %xmm0, %xmm0
    movss   %xmm0, (%r12)
    movsd   48(%rsp), %xmm0
    call    LFM_CALL(sin)
    mulsd   56(%rsp), %xmm0
    mulsd   40(%rsp), %xmm0
    movabsq $0x8000000000000000, %rax
    movq    %rax, %xmm1
    xorpd   %xmm1, %xmm0
    cvtsd2ss %xmm0, %xmm0
    movss   %xmm0, (%r13)
    addq    $4, %r12
    addq    $4, %r13
    incq    %rbx
    cmpq    %r15, %rbx
    jb      4b
    incq    %rbp
    cmpq    %r14, %rbp
    jb      1b
    xorl    %eax, %eax
    addq    $72, %rsp
    popq    %r15
    popq    %r14
    popq    %r13
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
8:  movl    $-22, %eax
    ret

.macro LFM_GATHER3 offset, target
    xorps   \target, \target
    movss   \offset(%r9), \target
    insertps $0x10, \offset+12(%r9), \target
    insertps $0x20, \offset+24(%r9), \target
    insertps $0x30, \offset+36(%r9), \target
.endm

.p2align 4
// void lfm_detok_conv_f32(args: LFM_DETOK_CONV_* block)
// LFM2 gated shortconv, width 3 (conv_L_cache): out = C.(w0*s0 + w1*s1 +
// w2*(B.x)) with the two-entry carry rotated in place. See twin.
.globl LFM_SYM(lfm_detok_conv_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_conv_f32))
LFM_SYM(lfm_detok_conv_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_CONV_PROJECTED(%r10), %r11
    movq    LFM_DETOK_CONV_CARRY(%r10), %r8
    movq    LFM_DETOK_CONV_WEIGHT(%r10), %r9
    movq    LFM_DETOK_CONV_OUTPUT(%r10), %rdx
    movq    LFM_DETOK_CONV_COUNT(%r10), %rcx
1:  testq   %rcx, %rcx
    jz      9f
    movups  0(%r11), %xmm0
    mulps   4096(%r11), %xmm0
    movups  2048(%r8), %xmm1
    movups  4096(%r8), %xmm2
    LFM_GATHER3 0, %xmm3
    LFM_GATHER3 4, %xmm4
    LFM_GATHER3 8, %xmm5
    mulps   %xmm1, %xmm3
    mulps   %xmm2, %xmm4
    addps   %xmm4, %xmm3
    mulps   %xmm0, %xmm5
    addps   %xmm5, %xmm3
    mulps   2048(%r11), %xmm3
    movups  %xmm3, (%rdx)
    movups  %xmm1, 0(%r8)
    movups  %xmm2, 2048(%r8)
    movups  %xmm0, 4096(%r8)
    addq    $16, %r11
    addq    $16, %r8
    addq    $48, %r9
    addq    $16, %rdx
    subq    $4, %rcx
    jmp     1b
9:  ret

.p2align 4
// void lfm_detok_weighted_f32(args: LFM_DETOK_WEIGHTED_* block)
// Weighted-V for one head: out[32] = sum_j scores[j] * V[(first+j) mod 30].
// The mod-30 ring is the reference's sliding window (sliding_window_size=30).
// Eight xmm accumulators hold the full head; ascending-j committed order.
.globl LFM_SYM(lfm_detok_weighted_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_weighted_f32))
LFM_SYM(lfm_detok_weighted_f32):
#ifdef _WIN32
    movq    %rcx, %r10
    // Win64 keeps only xmm0-xmm5 volatile: one register tile at a time.
    movq    LFM_DETOK_WEIGHTED_OUTPUT(%r10), %r11
    xorl    %r9d, %r9d
7:  cmpl    $8, %r9d
    jae     9f
    xorps   %xmm0, %xmm0
    xorq    %r8, %r8
1:  cmpq    LFM_DETOK_WEIGHTED_COUNT(%r10), %r8
    jae     2f
    movq    LFM_DETOK_WEIGHTED_FIRST(%r10), %rax
    addq    %r8, %rax
    cmpq    $30, %rax
    jb      3f
    subq    $30, %rax
3:  shlq    $10, %rax
    addq    LFM_DETOK_WEIGHTED_VALUES(%r10), %rax
    movq    LFM_DETOK_WEIGHTED_HEAD(%r10), %rdx
    shlq    $7, %rdx
    addq    %rdx, %rax
    movq    LFM_DETOK_WEIGHTED_SCORES(%r10), %rcx
    movss   (%rcx,%r8,4), %xmm1
    shufps  $0, %xmm1, %xmm1
    movq    %r9, %rdx
    shlq    $4, %rdx
    movups  (%rax,%rdx), %xmm2
    mulps   %xmm1, %xmm2
    addps   %xmm2, %xmm0
    incq    %r8
    jmp     1b
2:  movq    %r9, %rdx
    shlq    $4, %rdx
    movups  %xmm0, (%r11,%rdx)
    incl    %r9d
    jmp     7b
9:  ret
#else
    movq    %rdi, %r10
    movq    LFM_DETOK_WEIGHTED_VALUES(%r10), %r11
    movq    LFM_DETOK_WEIGHTED_SCORES(%r10), %rdx
    movq    LFM_DETOK_WEIGHTED_OUTPUT(%r10), %rcx
    movq    LFM_DETOK_WEIGHTED_FIRST(%r10), %r8
    movq    LFM_DETOK_WEIGHTED_COUNT(%r10), %r9
    movq    LFM_DETOK_WEIGHTED_HEAD(%r10), %rax
    shlq    $7, %rax
    addq    %rax, %r11
    xorps   %xmm0, %xmm0
    xorps   %xmm1, %xmm1
    xorps   %xmm2, %xmm2
    xorps   %xmm3, %xmm3
    xorps   %xmm4, %xmm4
    xorps   %xmm5, %xmm5
    xorps   %xmm6, %xmm6
    xorps   %xmm7, %xmm7
    xorq    %rax, %rax
1:  cmpq    %r9, %rax
    jae     2f
    leaq    (%r8,%rax), %rdi
    cmpq    $30, %rdi
    jb      3f
    subq    $30, %rdi
3:  shlq    $10, %rdi
    addq    %r11, %rdi
    movss   (%rdx,%rax,4), %xmm8
    shufps  $0, %xmm8, %xmm8
    movups  0(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm0
    movups  16(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm1
    movups  32(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm2
    movups  48(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm3
    movups  64(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm4
    movups  80(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm5
    movups  96(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm6
    movups  112(%rdi), %xmm9
    mulps   %xmm8, %xmm9
    addps   %xmm9, %xmm7
    incq    %rax
    jmp     1b
2:  movups  %xmm0, 0(%rcx)
    movups  %xmm1, 16(%rcx)
    movups  %xmm2, 32(%rcx)
    movups  %xmm3, 48(%rcx)
    movups  %xmm4, 64(%rcx)
    movups  %xmm5, 80(%rcx)
    movups  %xmm6, 96(%rcx)
    movups  %xmm7, 112(%rcx)
    ret
#endif

.p2align 4
// void lfm_detok_polar_f32(args: LFM_DETOK_POLAR_* block)
// real = mag*cos, imag = mag*sin — the multiply half of torch.polar; exp/
// cos/sin precomputed by the C++ owner (hot-pass libm rule).
.globl LFM_SYM(lfm_detok_polar_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_polar_f32))
LFM_SYM(lfm_detok_polar_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_POLAR_MAGNITUDE(%r10), %r11
    movq    LFM_DETOK_POLAR_SINE(%r10), %r8
    movq    LFM_DETOK_POLAR_COSINE(%r10), %r9
    movq    LFM_DETOK_POLAR_REAL(%r10), %rdx
    movq    LFM_DETOK_POLAR_IMAGINARY(%r10), %rcx
    movq    LFM_DETOK_POLAR_COUNT(%r10), %rax
1:  cmpq    $4, %rax
    jb      2f
    movups  (%r11), %xmm0
    movups  (%r8), %xmm1
    movups  (%r9), %xmm2
    mulps   %xmm0, %xmm1
    mulps   %xmm0, %xmm2
    movups  %xmm2, (%rdx)
    movups  %xmm1, (%rcx)
    addq    $16, %r11
    addq    $16, %r8
    addq    $16, %r9
    addq    $16, %rdx
    addq    $16, %rcx
    subq    $4, %rax
    jmp     1b
2:  testq   %rax, %rax
    jz      9f
3:  movss   (%r11), %xmm0
    movss   (%r8), %xmm1
    movss   (%r9), %xmm2
    mulss   %xmm0, %xmm1
    mulss   %xmm0, %xmm2
    movss   %xmm2, (%rdx)
    movss   %xmm1, (%rcx)
    addq    $4, %r11
    addq    $4, %r8
    addq    $4, %r9
    addq    $4, %rdx
    addq    $4, %rcx
    decq    %rax
    jnz     3b
9:  ret

.p2align 4
// void lfm_detok_overlap_f32(args: LFM_DETOK_OVERLAP_* block)
// Both iSTFT folds per pass: output += frame*window; envelope += window^2
// (Hann 1280, hop 320). Same coupled traversal as the twin.
.globl LFM_SYM(lfm_detok_overlap_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_overlap_f32))
LFM_SYM(lfm_detok_overlap_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_OVERLAP_SIGNAL(%r10), %r11
    movq    LFM_DETOK_OVERLAP_WINDOW(%r10), %r8
    movq    LFM_DETOK_OVERLAP_OUTPUT(%r10), %r9
    movq    LFM_DETOK_OVERLAP_ENVELOPE(%r10), %rdx
    movq    LFM_DETOK_OVERLAP_COUNT(%r10), %rax
1:  cmpq    $4, %rax
    jb      2f
    movups  (%r11), %xmm0
    movups  (%r8), %xmm1
    movups  (%r9), %xmm2
    movups  (%rdx), %xmm3
    mulps   %xmm1, %xmm0
    addps   %xmm0, %xmm2
    mulps   %xmm1, %xmm1
    addps   %xmm1, %xmm3
    movups  %xmm2, (%r9)
    movups  %xmm3, (%rdx)
    addq    $16, %r11
    addq    $16, %r8
    addq    $16, %r9
    addq    $16, %rdx
    subq    $4, %rax
    jmp     1b
2:  testq   %rax, %rax
    jz      9f
3:  movss   (%r11), %xmm0
    movss   (%r8), %xmm1
    movss   (%r9), %xmm2
    movss   (%rdx), %xmm3
    mulss   %xmm1, %xmm0
    addss   %xmm0, %xmm2
    mulss   %xmm1, %xmm1
    addss   %xmm1, %xmm3
    movss   %xmm2, (%r9)
    movss   %xmm3, (%rdx)
    addq    $4, %r11
    addq    $4, %r8
    addq    $4, %r9
    addq    $4, %rdx
    decq    %rax
    jnz     3b
9:  ret

.p2align 4
// int lfm_detok_emit_f32(args: LFM_DETOK_EMIT_* block)
// pcm[i] = output[i]/envelope[i] with the NOLA guard: any envelope <=
// epsilon returns -ERANGE before emitting (the reference's assert
// window_envelope > 1e-11, promoted to a typed hard error).
.globl LFM_SYM(lfm_detok_emit_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_emit_f32))
LFM_SYM(lfm_detok_emit_f32):
#ifdef _WIN32
    movq    %rcx, %r10
#else
    movq    %rdi, %r10
#endif
    movq    LFM_DETOK_EMIT_OUTPUT(%r10), %r11
    movq    LFM_DETOK_EMIT_ENVELOPE(%r10), %r8
    movq    LFM_DETOK_EMIT_PCM(%r10), %r9
    movq    LFM_DETOK_EMIT_COUNT(%r10), %rax
    movss   LFM_DETOK_EMIT_EPSILON(%r10), %xmm4
    shufps  $0, %xmm4, %xmm4
1:  cmpq    $4, %rax
    jb      2f
    movups  (%r8), %xmm0
    movaps  %xmm0, %xmm1
    cmpleps %xmm4, %xmm1
    movmskps %xmm1, %ecx
    testl   %ecx, %ecx
    jnz     8f
    movups  (%r11), %xmm1
    divps   %xmm0, %xmm1
    movups  %xmm1, (%r9)
    addq    $16, %r11
    addq    $16, %r8
    addq    $16, %r9
    subq    $4, %rax
    jmp     1b
2:  testq   %rax, %rax
    jz      7f
3:  movss   (%r8), %xmm0
    ucomiss %xmm4, %xmm0
    jbe     8f
    movss   (%r11), %xmm1
    divss   %xmm0, %xmm1
    movss   %xmm1, (%r9)
    addq    $4, %r11
    addq    $4, %r8
    addq    $4, %r9
    decq    %rax
    jnz     3b
7:  xorl    %eax, %eax
    ret
8:  movl    $-34, %eax
    ret
