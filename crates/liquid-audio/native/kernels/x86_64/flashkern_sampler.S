#include "lfm_asm_visibility.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.intel_syntax noprefix
.text

/*
 * flashkern_sampler.S — x86-64. Token selection: argmax, softmax-with-cutoff,
 * and the CDF walk. Feeds on the uniform draw from flashkern_prng.S.
 *
 * x86-64 twin of kernels/aarch64/flashkern_sampler.S. Same five entry points,
 * byte-identical exp constants, and the same three guarantees below — but the
 * NaN guarantee costs real instructions here (see NaN SCRUB).
 *
 * DETERMINISM CONTRACT:
 *
 * 1. TIE-BREAK IS LOWEST INDEX. Both argmax leaves run TWO passes: a vector
 *    reduction for the maximum VALUE, then a scan from index 0 for the first
 *    element equal to it (.Largmax_*_scan). Fusing them would return whichever
 *    lane saw the max first, making the answer depend on vector width and loop
 *    peeling. Ties are common in quantised logits.
 *
 * 2. NaN CANNOT WIN — and this is where the architectures differ in MEANS, not
 *    in result. AArch64 gets it free from `fmaxnm` (IEEE maxNum: a NaN operand
 *    yields the other value). x86 `maxps`/`maxss` do the OPPOSITE — on an
 *    unordered compare they return the SECOND operand, so a NaN lane would
 *    capture the maximum and pin generation to one token. The loop therefore
 *    scrubs NaN before every max:
 *
 *        movaps   xmm2, xmm1
 *        cmpordps xmm2, xmm2     ; ordered mask: all-ones where NOT NaN
 *        andps    xmm1, xmm2     ; keep the real lanes
 *        andnps   xmm2, xmm3     ; xmm3 holds broadcast -inf
 *        orps     xmm1, xmm2     ; NaN lanes become -inf
 *        maxps    xmm0, xmm1
 *
 *    xmm3 is kept live across the whole loop solely as that -inf source. This is
 *    a hand-built fmaxnm; deleting it as "redundant masking" silently
 *    reintroduces the NaN-wins bug.
 *
 * 3. NO LIBM. The exponential is the local .Lsampler_exp_f32 below — f32 range
 *    reduction plus a degree-six polynomial, entirely in this object, with
 *    constants byte-identical to the AArch64 twin. Unlike flashkern_rope.S
 *    (whose tables inherit the platform libm's last bits), sampling is
 *    bit-identical on every machine and OS.
 *
 * PRECISION SPLIT. Logits are handled at model precision (f32 and bf16 leaves);
 * the uniform draw indexing the distribution is a double built from 53 PRNG
 * bits. Narrowing the draw to bf16 resolution would make any token with mass
 * below ~1/256 unreachable — truncating the tail off the distribution rather
 * than merely rounding it.
 */

/*
 * uint32_t lfm_sampler_argmax_f32(const float *x, size_t count)
 *
 * Pass 1: NaN-scrubbed 4-lane `maxps`, then movhlps/shufps horizontal reduce,
 * then scalar tail. Pass 2: linear scan for the FIRST index equal to that
 * maximum. Accumulator starts at 0xff800000 (-inf). count == 0 returns 0.
 */
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

/*
 * uint32_t lfm_sampler_argmax_bf16(const uint16_t *x, size_t count)
 *
 * As the f32 leaf, including the NaN scrub and the two-pass lowest-index
 * tie-break. BF16 lanes are lifted to F32 by shifting into the high half —
 * exact and lossless, never a conversion instruction.
 */
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
 *
 * One fused pass producing the unnormalised softmax weights and their sum:
 *
 *     v = x[i] * scale                 // scale = 1/temperature
 *     if (v < threshold)  w = 0        // top-k / top-p cutoff, no exp taken
 *     else                w = exp(v - maximum)
 *     weights[i] = w ;  sum += w
 *
 * Subtracting `maximum` is the overflow guard, so the exp argument is always
 * <= 0 — which is why the local exp has no overflow path. The threshold is
 * compared against the SCALED logit, before max-subtraction, so a caller's
 * cutoff is expressed in the same units as its temperature.
 *
 * `sum` accumulates sequentially in F32, in index order. That ordering is the
 * contract: prefix_pick must reproduce these partial sums exactly.
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
/*
 * float lfm_sampler_exp_sum_bf16(const uint16_t *x, float *weights,
 *                                size_t count, uint16_t bf16_scale,
 *                                float maximum, float threshold)
 *
 * As the f32 leaf, with one deliberate extra rounding step: after v = x[i] *
 * scale, the product is rounded BACK to BF16 precision (open-coded
 * round-to-nearest-even, the same sequence as LFM_RNE_WIDEN in
 * flashkern_math.S) BEFORE the threshold compare and the exp.
 *
 * Keeping that product in full F32 would be more accurate and WRONG: the
 * reference materialises the scaled logit as BF16, so both the cutoff decision
 * and the exp argument must see the rounded value. `scale` arrives as BF16 bits
 * and is lifted by shift, so the multiply itself has BF16 inputs.
 *
 * Weights and their sum remain F32 — only the scaled logit is narrowed.
 */
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

/*
 * uint32_t lfm_sampler_prefix_pick(const float *weights, size_t count,
 *                                  float target)
 *
 * The CDF walk. Accumulates `weights` in index order and returns the first index
 * where the running sum exceeds `target` (the uniform draw scaled by the total
 * from exp_sum). Accumulation order matches exp_sum's exactly — both walk low to
 * high in F32 — so the partial sums here reproduce the total computed there.
 *
 * SHORTFALL GUARANTEE — the reason a "last positive index" register is carried:
 *
 * `target` derives from a double uniform scaled by an F32 sum, so rounding can
 * leave it a few ULP ABOVE the final running sum, and the loop falls out without
 * ever triggering. Rather than returning 0 or count-1, the last index whose
 * weight was STRICTLY POSITIVE is returned.
 *
 * This makes it impossible to select a token the threshold cut off (weight 0)
 * purely because of a rounding shortfall — a bug invisible in aggregate
 * statistics that would occasionally emit a token the policy had excluded.
 */
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
