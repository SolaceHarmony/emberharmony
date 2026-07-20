#include "lfm_asm_visibility.h"

#if defined(__APPLE__)
#define CNAME(name) _##name
#else
#define CNAME(name) name
#endif

.text
.p2align 4

/*
 * flashkern_prng.S — AArch64. TWO functions with OPPOSITE reproducibility
 * guarantees. Confusing them silently destroys fixed-seed byte-identity, so the
 * distinction is stated first:
 *
 *   lfm_chacha20_block       DETERMINISTIC. Pure function of input[16].
 *   lfm_apple_secure_random  NON-DETERMINISTIC. Kernel entropy, by design.
 *
 * Any path that claims reproducible output from a seed must reach the first and
 * never the second.
 *
 * DETERMINISM CONTRACT (lfm_chacha20_block):
 *
 * - Integer only: add / eor / ror. No floating point, no libm, no platform math.
 *   Unlike flashkern_rope.S — whose tables inherit the platform libm's last bits
 *   — this kernel is bit-identical on every machine and every OS. Same
 *   input[16] gives the same output[16], always.
 * - The x86-64 twin packs one ROW per XMM register and realigns with pshufd.
 *   Different data layout, IDENTICAL stream. That equivalence is an invariant of
 *   the pair, not a coincidence.
 * - 20 rounds, as 10 iterations of the double-round (w19 = 10): four column
 *   quarter-rounds then four diagonal quarter-rounds, written out explicitly
 *   here because all sixteen words are already in registers.
 *
 * LAYOUT — the interop detail that matters:
 *
 * This is the ORIGINAL ChaCha20 word layout: 64-bit counter + 64-bit nonce.
 * It is NOT RFC 8439, which splits the same 128 bits as a 32-bit counter plus a
 * 96-bit nonce. The rounds are identical either way, so a mismatch does not fail
 * loudly — it just yields a completely different keystream for the same seed
 * material. Anything interoperating with this PRNG must use the same split.
 *
 * ChaCha20 block function, original 64-bit-counter/64-bit-nonce layout.
 *
 *   x0: const uint32_t input[16]
 *   x1: uint32_t output[16]
 *
 * Sixteen live words stay in w2-w17 for all twenty rounds. x18 is deliberately
 * untouched because Apple reserves it as the platform register.
 */
.macro QR a, b, c, d
    add     \a, \a, \b
    eor     \d, \d, \a
    ror     \d, \d, #16
    add     \c, \c, \d
    eor     \b, \b, \c
    ror     \b, \b, #20
    add     \a, \a, \b
    eor     \d, \d, \a
    ror     \d, \d, #24
    add     \c, \c, \d
    eor     \b, \b, \c
    ror     \b, \b, #25
.endm

.globl CNAME(lfm_chacha20_block)
LFM_PRIVATE(CNAME(lfm_chacha20_block))
#if !defined(__APPLE__)
.type CNAME(lfm_chacha20_block), %function
#endif
CNAME(lfm_chacha20_block):
    stp     x19, x20, [sp, #-16]!
    ldp     w2,  w3,  [x0, #0]
    ldp     w4,  w5,  [x0, #8]
    ldp     w6,  w7,  [x0, #16]
    ldp     w8,  w9,  [x0, #24]
    ldp     w10, w11, [x0, #32]
    ldp     w12, w13, [x0, #40]
    ldp     w14, w15, [x0, #48]
    ldp     w16, w17, [x0, #56]
    mov     w19, #10
    b       .Lchacha_rounds

.Lchacha_rounds:
    QR      w2, w6, w10, w14
    QR      w3, w7, w11, w15
    QR      w4, w8, w12, w16
    QR      w5, w9, w13, w17
    QR      w2, w7, w12, w17
    QR      w3, w8, w13, w14
    QR      w4, w9, w10, w15
    QR      w5, w6, w11, w16
    subs    w19, w19, #1
    b.ne    .Lchacha_rounds

    ldp     w19, w20, [x0, #0]
    add     w2, w2, w19
    add     w3, w3, w20
    stp     w2, w3, [x1, #0]
    ldp     w19, w20, [x0, #8]
    add     w4, w4, w19
    add     w5, w5, w20
    stp     w4, w5, [x1, #8]
    ldp     w19, w20, [x0, #16]
    add     w6, w6, w19
    add     w7, w7, w20
    stp     w6, w7, [x1, #16]
    ldp     w19, w20, [x0, #24]
    add     w8, w8, w19
    add     w9, w9, w20
    stp     w8, w9, [x1, #24]
    ldp     w19, w20, [x0, #32]
    add     w10, w10, w19
    add     w11, w11, w20
    stp     w10, w11, [x1, #32]
    ldp     w19, w20, [x0, #40]
    add     w12, w12, w19
    add     w13, w13, w20
    stp     w12, w13, [x1, #40]
    ldp     w19, w20, [x0, #48]
    add     w14, w14, w19
    add     w15, w15, w20
    stp     w14, w15, [x1, #48]
    ldp     w19, w20, [x0, #56]
    add     w16, w16, w19
    add     w17, w17, w20
    stp     w16, w17, [x1, #56]
    ldp     x19, x20, [sp], #16
    ret
#if !defined(__APPLE__)
.size CNAME(lfm_chacha20_block), .-CNAME(lfm_chacha20_block)
#endif

#if defined(__APPLE__)
/*
 * int lfm_apple_secure_random(void *bytes, size_t count)
 *
 * TRUE ENTROPY — the deliberate opposite of lfm_chacha20_block. Output is not
 * reproducible and must never feed a path that promises fixed-seed byte
 * identity. It exists to SEED that path, not to replace it.
 *
 * Correct SecRandomCopyBytes ABI:
 *   x0 = kSecRandomDefault (NULL)
 *   x1 = count
 *   x2 = bytes
 */
.p2align 2
.globl CNAME(lfm_apple_secure_random)
LFM_PRIVATE(CNAME(lfm_apple_secure_random))
CNAME(lfm_apple_secure_random):
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x2, x0
    mov     x0, xzr
    bl      CNAME(SecRandomCopyBytes)
    ldp     x29, x30, [sp], #16
    ret
#endif

#if defined(__APPLE__)
.subsections_via_symbols
#endif
