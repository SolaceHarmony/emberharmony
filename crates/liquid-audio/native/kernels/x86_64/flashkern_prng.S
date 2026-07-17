#include "lfm_asm_visibility.h"

#if defined(__APPLE__)
#define CNAME(name) _##name
#else
#define CNAME(name) name
#endif

.intel_syntax noprefix
.text
.p2align 4

/* Four-way SSE2 ChaCha20 block. Each XMM register is one row of four words. */
.macro ROTL reg, left, right
    movdqa  xmm4, \reg
    pslld   \reg, \left
    psrld   xmm4, \right
    por     \reg, xmm4
.endm

.macro QR a, b, c, d
    paddd   \a, \b
    pxor    \d, \a
    ROTL    \d, 16, 16
    paddd   \c, \d
    pxor    \b, \c
    ROTL    \b, 12, 20
    paddd   \a, \b
    pxor    \d, \a
    ROTL    \d, 8, 24
    paddd   \c, \d
    pxor    \b, \c
    ROTL    \b, 7, 25
.endm

.globl CNAME(lfm_chacha20_block)
LFM_PRIVATE(CNAME(lfm_chacha20_block))
#if !defined(__APPLE__)
.type CNAME(lfm_chacha20_block), @function
#endif
CNAME(lfm_chacha20_block):
#if defined(_WIN32)
    mov     r8, rcx
    mov     r9, rdx
#else
    mov     r8, rdi
    mov     r9, rsi
#endif
    movdqu  xmm0, XMMWORD PTR [r8 + 0]
    movdqu  xmm1, XMMWORD PTR [r8 + 16]
    movdqu  xmm2, XMMWORD PTR [r8 + 32]
    movdqu  xmm3, XMMWORD PTR [r8 + 48]
    mov     eax, 10
    jmp     .Lchacha_rounds

.Lchacha_rounds:
    QR      xmm0, xmm1, xmm2, xmm3
    pshufd  xmm1, xmm1, 0x39
    pshufd  xmm2, xmm2, 0x4e
    pshufd  xmm3, xmm3, 0x93
    QR      xmm0, xmm1, xmm2, xmm3
    pshufd  xmm1, xmm1, 0x93
    pshufd  xmm2, xmm2, 0x4e
    pshufd  xmm3, xmm3, 0x39
    dec     eax
    jne     .Lchacha_rounds

    paddd   xmm0, XMMWORD PTR [r8 + 0]
    paddd   xmm1, XMMWORD PTR [r8 + 16]
    paddd   xmm2, XMMWORD PTR [r8 + 32]
    paddd   xmm3, XMMWORD PTR [r8 + 48]
    movdqu  XMMWORD PTR [r9 + 0], xmm0
    movdqu  XMMWORD PTR [r9 + 16], xmm1
    movdqu  XMMWORD PTR [r9 + 32], xmm2
    movdqu  XMMWORD PTR [r9 + 48], xmm3
    ret
#if !defined(__APPLE__)
.size CNAME(lfm_chacha20_block), .-CNAME(lfm_chacha20_block)
#endif

#if defined(__APPLE__)
/* int lfm_apple_secure_random(void *bytes, size_t count) */
.p2align 4
.globl CNAME(lfm_apple_secure_random)
LFM_PRIVATE(CNAME(lfm_apple_secure_random))
CNAME(lfm_apple_secure_random):
    push    rbp
    mov     rbp, rsp
    mov     rdx, rdi
    xor     edi, edi
    call    CNAME(SecRandomCopyBytes)
    pop     rbp
    ret
.subsections_via_symbols
#elif defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
