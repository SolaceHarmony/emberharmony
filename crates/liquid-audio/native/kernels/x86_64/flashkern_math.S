// flashkern_math.S — x86-64 scalar numeric leaves for the LFM2 decode path.
//
// This is the x86-64 twin of kernels/aarch64/flashkern_math.S. The numeric
// contract is IDENTICAL by construction and verified instruction-for-instruction;
// it is restated here so neither file has to be read to trust the other.
//
// NUMERIC CONTRACT:
//
// 1. Rounding points are placed, not minimised. Where the reference stores BF16,
//    this rounds to BF16 at that same point (LFM_RNE_WIDEN); where the reference
//    holds F32, the value stays F32. Carrying MORE precision than the reference
//    is a deviation, not an improvement — the weights were trained through the
//    reference's rounding.
//
// 2. BF16 -> F32 lift is `movzwl` + `shll $16` — a pure bit move (LFM_WIDEN_BF16).
//    Exact and lossless; never a conversion instruction, so no implementation-
//    defined rounding enters.
//
// 3. F32 -> BF16 is round-to-nearest-EVEN, open-coded:
//        bits += 0x7fff + ((bits >> 16) & 1)
//    The `& 1` term is the destination LSB, which makes exact ties round to even
//    rather than always up.
//
// 4. Reductions are strictly sequential — one `addss` per element. No SSE/AVX
//    lane parallelism and no pairwise tree. A tree reduction is usually MORE
//    accurate and would still be wrong: it changes summation order, and the
//    order is the contract.
//
// 5. Division and square root use exact `divss`/`sqrtss`, never `rsqrtss`/
//    `rcpss`. The estimate instructions are ~12-bit accurate and their refinement
//    diverges from the reference in the last bits.
//
// X86-SPECIFIC NOTES:
//
// - Every leaf carries a `_WIN32` branch: Microsoft x64 passes integers in
//   rcx/rdx/r8/r9, System V in rdi/rsi/rdx/rcx. Arguments are shuffled into
//   fixed scratch registers immediately so the body below is ABI-agnostic. This
//   is the only structural difference from the AArch64 file.
// - `lfm_bf16_gemv_rne_bf16` / `_add_bf16` (the decode projection superkernel)
//   exist ONLY in the AArch64 file. x86-64 has no twin; callers on this target
//   must route those projections elsewhere.
//
// Recorded evidence of these numerics lives in
// native/tests/fixtures/conformer/ (72 per-stage tensors, checkpoint
// LFM2.5-Audio-1.5B c362a062), kept as provenance for the values independent of
// any harness that once consumed them.

#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text

// float lfm_rsqrt_size(uint64_t n) -> 1.0f / sqrtf((float)n)
// Exact sqrtss + divss (contract note 5). 0x3f800000 is the bit pattern of 1.0f.
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

// float lfm_inv_rms_f32(float sumsq, float eps, uint64_t n)
//   -> 1.0f / sqrtf(sumsq / n + eps)
// Operand ORDER is the contract: divide by n first, THEN add eps, then sqrt,
// then reciprocate. Folding eps in before the division, or scaling after the
// root, changes the low bits for small sumsq and breaks reference agreement.
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

// float lfm_sum_f32(const float *x, uint64_t n)
// Strictly sequential F32 accumulation, low index to high (contract note 4).
// Deliberately scalar `addss`: packed accumulation would reorder the sum.
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

// BF16 -> F32 lift: load 16 bits, shift into the high half. Exact, lossless,
// no conversion instruction involved (contract note 2).
.macro LFM_WIDEN_BF16 address, target, bits
    movzwl    \address, \bits
    shll      $16, \bits
    movd      \bits, \target
.endm

// Round an F32 register to BF16 precision, in place, round-to-nearest-EVEN.
// The result stays an F32 whose low 16 mantissa bits are zero, so a logical
// BF16 value remains register-resident with no store/reload round trip.
//
//     tie   = (bits >> 16) & 1     // destination LSB — the "even" term
//     bits += 0x7fff + tie         // half-way cases round toward even
//     bits  = (bits >> 16) << 16   // clear what BF16 cannot represent
//
// Without the `tie` term this is round-half-UP and drifts from the reference on
// exact ties. Behaviourally identical to the AArch64 macro of the same name.
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

// void lfm_bf16_rope_neox(uint16_t *x, const uint16_t *cos,
//                         const uint16_t *sin, uint64_t dim)
// Rotary position embedding, NeoX pairing, applied in place over one head.
//
// PAIRING: element i is rotated against element i + dim/2 — the GPT-NeoX
// split-half convention, NOT the GPT-J interleaved (i, i+1) one. The wrong
// pairing yields plausible audio that degrades with sequence length, so it is
// stated rather than left to be inferred. Odd `dim` is rejected, not truncated.
//
//     out[i]        =  x[i]*cos - x[i+d/2]*sin
//     out[i + d/2]  =  x[i+d/2]*cos + x[i]*sin
//
// ROUNDING: every product and every sum passes through LFM_RNE_WIDEN — six
// rounding points per pair, not one, matching the AArch64 twin exactly.
// Accumulating each pair in full F32 and rounding once would be more accurate
// and WRONG: the reference materialises BF16 at each of these steps and the
// model was trained through exactly that loss.
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
