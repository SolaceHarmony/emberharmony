#include "lfm_asm_visibility.h"

// SSE2 twin of the realtime capture-format leaves. Mono and stereo device
// layouts run across frames in vectors; uncommon channel counts preserve the
// source implementation's left-to-right channel order in scalar SSE. Both
// SysV and GNU Win64 ABIs are supported.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 4

// void lfm_capture_downmix_f32_dd(const float *source, float *destination,
//                                 size_t frames, uint32_t channels)
.globl LFM_SYM(lfm_capture_downmix_f32_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_f32_dd))
LFM_SYM(lfm_capture_downmix_f32_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movl      %ecx, %eax
    cvtsi2ssq %rax, %xmm5
    shufps    $0, %xmm5, %xmm5
    cmpl      $1, %ecx
    je        10f
    cmpl      $2, %ecx
    je        20f
    jmp       40f
10:
    cmpq      $4, %rdx
    jb        40f
11:
    movups    (%r10), %xmm0
    movups    %xmm0, (%r11)
    addq      $16, %r10
    addq      $16, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       11b
    jmp       40f
20:
    cmpq      $4, %rdx
    jb        40f
21:
    movups    (%r10), %xmm0
    movups    16(%r10), %xmm1
    movaps    %xmm0, %xmm2
    movaps    %xmm0, %xmm3
    shufps    $0x88, %xmm1, %xmm2
    shufps    $0xdd, %xmm1, %xmm3
    addps     %xmm3, %xmm2
    divps     %xmm5, %xmm2
    movups    %xmm2, (%r11)
    addq      $32, %r10
    addq      $16, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       21b
40:
    testq     %rdx, %rdx
    jz        49f
41:
    xorps     %xmm0, %xmm0
    movl      %ecx, %r8d
42:
    movss     (%r10), %xmm1
    addss     %xmm1, %xmm0
    addq      $4, %r10
    decl      %r8d
    jnz       42b
    divss     %xmm5, %xmm0
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decq      %rdx
    jnz       41b
49:
    ret

// void lfm_capture_downmix_i16_dd(const int16_t *source, float *destination,
//                                 size_t frames, uint32_t channels)
.p2align 4
.globl LFM_SYM(lfm_capture_downmix_i16_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_i16_dd))
LFM_SYM(lfm_capture_downmix_i16_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movl      %ecx, %eax
    cvtsi2ssq %rax, %xmm5
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm4
    mulss     %xmm4, %xmm5
    shufps    $0, %xmm5, %xmm5
    cmpl      $1, %ecx
    je        110f
    cmpl      $2, %ecx
    je        120f
    jmp       140f
110:
    cmpq      $8, %rdx
    jb        140f
    pxor      %xmm3, %xmm3
111:
    movdqu    (%r10), %xmm0
    movdqa    %xmm3, %xmm1
    movdqa    %xmm0, %xmm2
    pcmpgtw   %xmm0, %xmm1
    punpcklwd %xmm1, %xmm0
    punpckhwd %xmm1, %xmm2
    cvtdq2ps  %xmm0, %xmm0
    cvtdq2ps  %xmm2, %xmm2
    divps     %xmm5, %xmm0
    divps     %xmm5, %xmm2
    movups    %xmm0, (%r11)
    movups    %xmm2, 16(%r11)
    addq      $16, %r10
    addq      $32, %r11
    subq      $8, %rdx
    cmpq      $8, %rdx
    jae       111b
    jmp       140f
120:
    cmpq      $8, %rdx
    jb        140f
    pcmpeqw   %xmm4, %xmm4
    psrlw     $15, %xmm4
121:
    movdqu    (%r10), %xmm0
    movdqu    16(%r10), %xmm1
    pmaddwd   %xmm4, %xmm0
    pmaddwd   %xmm4, %xmm1
    cvtdq2ps  %xmm0, %xmm0
    cvtdq2ps  %xmm1, %xmm1
    divps     %xmm5, %xmm0
    divps     %xmm5, %xmm1
    movups    %xmm0, (%r11)
    movups    %xmm1, 16(%r11)
    addq      $32, %r10
    addq      $32, %r11
    subq      $8, %rdx
    cmpq      $8, %rdx
    jae       121b
140:
    testq     %rdx, %rdx
    jz        149f
141:
    xorps     %xmm0, %xmm0
    movl      %ecx, %r8d
142:
    movswl    (%r10), %eax
    cvtsi2ssl %eax, %xmm1
    addss     %xmm1, %xmm0
    addq      $2, %r10
    decl      %r8d
    jnz       142b
    divss     %xmm5, %xmm0
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decq      %rdx
    jnz       141b
149:
    ret

// void lfm_capture_downmix_u16_dd(const uint16_t *source, float *destination,
//                                 size_t frames, uint32_t channels)
.p2align 4
.globl LFM_SYM(lfm_capture_downmix_u16_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_u16_dd))
LFM_SYM(lfm_capture_downmix_u16_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movl      %ecx, %eax
    cvtsi2ssq %rax, %xmm5
    shufps    $0, %xmm5, %xmm5
    movl      $32768, %eax
    cvtsi2ssl %eax, %xmm4
    shufps    $0, %xmm4, %xmm4
    pcmpeqw   %xmm3, %xmm3
    psllw     $15, %xmm3
    cmpl      $1, %ecx
    je        210f
    cmpl      $2, %ecx
    je        220f
    jmp       240f
210:
    cmpq      $8, %rdx
    jb        240f
211:
    movdqu    (%r10), %xmm0
    pxor      %xmm3, %xmm0
    pxor      %xmm2, %xmm2
    movdqa    %xmm0, %xmm1
    pcmpgtw   %xmm0, %xmm2
    punpcklwd %xmm2, %xmm0
    punpckhwd %xmm2, %xmm1
    cvtdq2ps  %xmm0, %xmm0
    cvtdq2ps  %xmm1, %xmm1
    divps     %xmm4, %xmm0
    divps     %xmm4, %xmm1
    divps     %xmm5, %xmm0
    divps     %xmm5, %xmm1
    movups    %xmm0, (%r11)
    movups    %xmm1, 16(%r11)
    addq      $16, %r10
    addq      $32, %r11
    subq      $8, %rdx
    cmpq      $8, %rdx
    jae       211b
    jmp       240f
220:
    cmpq      $8, %rdx
    jb        240f
    pcmpeqw   %xmm2, %xmm2
    psrlw     $15, %xmm2
221:
    movdqu    (%r10), %xmm0
    movdqu    16(%r10), %xmm1
    pxor      %xmm3, %xmm0
    pxor      %xmm3, %xmm1
    pmaddwd   %xmm2, %xmm0
    pmaddwd   %xmm2, %xmm1
    cvtdq2ps  %xmm0, %xmm0
    cvtdq2ps  %xmm1, %xmm1
    divps     %xmm4, %xmm0
    divps     %xmm4, %xmm1
    divps     %xmm5, %xmm0
    divps     %xmm5, %xmm1
    movups    %xmm0, (%r11)
    movups    %xmm1, 16(%r11)
    addq      $32, %r10
    addq      $32, %r11
    subq      $8, %rdx
    cmpq      $8, %rdx
    jae       221b
240:
    testq     %rdx, %rdx
    jz        249f
241:
    xorps     %xmm0, %xmm0
    movl      %ecx, %r8d
242:
    movzwl    (%r10), %eax
    cvtsi2ssl %eax, %xmm1
    subss     %xmm4, %xmm1
    divss     %xmm4, %xmm1
    addss     %xmm1, %xmm0
    addq      $2, %r10
    decl      %r8d
    jnz       242b
    divss     %xmm5, %xmm0
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decq      %rdx
    jnz       241b
249:
    ret

// void lfm_playback_fanout_f32_dd(const float *source, float *destination,
//     size_t frames, uint32_t channels)
.p2align 4
.globl LFM_SYM(lfm_playback_fanout_f32_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_fanout_f32_dd))
LFM_SYM(lfm_playback_fanout_f32_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    cmpl      $1, %ecx
    je        310f
    cmpl      $2, %ecx
    je        320f
    jmp       340f
310:
    cmpq      $4, %rdx
    jb        340f
311:
    movups    (%r10), %xmm0
    movups    %xmm0, (%r11)
    addq      $16, %r10
    addq      $16, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       311b
    jmp       340f
320:
    cmpq      $4, %rdx
    jb        340f
321:
    movups    (%r10), %xmm0
    movaps    %xmm0, %xmm1
    movaps    %xmm0, %xmm2
    unpcklps  %xmm1, %xmm1
    unpckhps  %xmm2, %xmm2
    movups    %xmm1, (%r11)
    movups    %xmm2, 16(%r11)
    addq      $16, %r10
    addq      $32, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       321b
340:
    testq     %rdx, %rdx
    jz        348f
341:
    movss     (%r10), %xmm0
    addq      $4, %r10
    movl      %ecx, %eax
342:
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decl      %eax
    jnz       342b
    decq      %rdx
    jnz       341b
348:
    ret

// void lfm_playback_fanout_i16_dd(const float *source, int16_t *destination,
//     size_t frames, uint32_t channels)
.p2align 4
.globl LFM_SYM(lfm_playback_fanout_i16_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_fanout_i16_dd))
LFM_SYM(lfm_playback_fanout_i16_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movl      $0x3f800000, %eax
    movd      %eax, %xmm3
    shufps    $0, %xmm3, %xmm3
    movl      $0xbf800000, %eax
    movd      %eax, %xmm4
    shufps    $0, %xmm4, %xmm4
    cmpl      $1, %ecx
    je        410f
    cmpl      $2, %ecx
    je        420f
    jmp       440f
410:
    cmpq      $4, %rdx
    jb        440f
411:
    movups    (%r10), %xmm0
    movaps    %xmm0, %xmm2
    cmpordps  %xmm0, %xmm2
    andps     %xmm2, %xmm0
    maxps     %xmm4, %xmm0
    minps     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm2
    shufps    $0, %xmm2, %xmm2
    mulps     %xmm2, %xmm0
    cvttps2dq %xmm0, %xmm0
    packssdw  %xmm0, %xmm0
    movq      %xmm0, (%r11)
    addq      $16, %r10
    addq      $8, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       411b
    jmp       440f
420:
    cmpq      $4, %rdx
    jb        440f
421:
    movups    (%r10), %xmm0
    movaps    %xmm0, %xmm2
    cmpordps  %xmm0, %xmm2
    andps     %xmm2, %xmm0
    maxps     %xmm4, %xmm0
    minps     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm2
    shufps    $0, %xmm2, %xmm2
    mulps     %xmm2, %xmm0
    cvttps2dq %xmm0, %xmm0
    packssdw  %xmm0, %xmm0
    punpcklwd %xmm0, %xmm0
    movdqu    %xmm0, (%r11)
    addq      $16, %r10
    addq      $16, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       421b
440:
    testq     %rdx, %rdx
    jz        448f
441:
    movss     (%r10), %xmm0
    addq      $4, %r10
    movaps    %xmm0, %xmm1
    cmpordss  %xmm0, %xmm1
    andps     %xmm1, %xmm0
    maxss     %xmm4, %xmm0
    minss     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm1
    mulss     %xmm1, %xmm0
    cvttss2si %xmm0, %r8d
    movl      %ecx, %eax
442:
    movw      %r8w, (%r11)
    addq      $2, %r11
    decl      %eax
    jnz       442b
    decq      %rdx
    jnz       441b
448:
    ret

// void lfm_playback_fanout_u16_dd(const float *source, uint16_t *destination,
//     size_t frames, uint32_t channels)
.p2align 4
.globl LFM_SYM(lfm_playback_fanout_u16_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_fanout_u16_dd))
LFM_SYM(lfm_playback_fanout_u16_dd):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8, %rdx
    movl      %r9d, %ecx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    movl      $0x3f800000, %eax
    movd      %eax, %xmm3
    shufps    $0, %xmm3, %xmm3
    movl      $0xbf800000, %eax
    movd      %eax, %xmm4
    shufps    $0, %xmm4, %xmm4
    cmpl      $1, %ecx
    je        510f
    cmpl      $2, %ecx
    je        520f
    jmp       540f
510:
    cmpq      $4, %rdx
    jb        540f
511:
    movups    (%r10), %xmm0
    movaps    %xmm0, %xmm2
    cmpordps  %xmm0, %xmm2
    andps     %xmm2, %xmm0
    maxps     %xmm4, %xmm0
    minps     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm2
    shufps    $0, %xmm2, %xmm2
    mulps     %xmm2, %xmm0
    cvttps2dq %xmm0, %xmm0
    packssdw  %xmm0, %xmm0
    pcmpeqw   %xmm2, %xmm2
    psllw     $15, %xmm2
    pxor      %xmm2, %xmm0
    movq      %xmm0, (%r11)
    addq      $16, %r10
    addq      $8, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       511b
    jmp       540f
520:
    cmpq      $4, %rdx
    jb        540f
521:
    movups    (%r10), %xmm0
    movaps    %xmm0, %xmm2
    cmpordps  %xmm0, %xmm2
    andps     %xmm2, %xmm0
    maxps     %xmm4, %xmm0
    minps     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm2
    shufps    $0, %xmm2, %xmm2
    mulps     %xmm2, %xmm0
    cvttps2dq %xmm0, %xmm0
    packssdw  %xmm0, %xmm0
    pcmpeqw   %xmm2, %xmm2
    psllw     $15, %xmm2
    pxor      %xmm2, %xmm0
    punpcklwd %xmm0, %xmm0
    movdqu    %xmm0, (%r11)
    addq      $16, %r10
    addq      $16, %r11
    subq      $4, %rdx
    cmpq      $4, %rdx
    jae       521b
540:
    testq     %rdx, %rdx
    jz        548f
541:
    movss     (%r10), %xmm0
    addq      $4, %r10
    movaps    %xmm0, %xmm1
    cmpordss  %xmm0, %xmm1
    andps     %xmm1, %xmm0
    maxss     %xmm4, %xmm0
    minss     %xmm3, %xmm0
    movl      $32767, %eax
    cvtsi2ssl %eax, %xmm1
    mulss     %xmm1, %xmm0
    cvttss2si %xmm0, %r8d
    addl      $32768, %r8d
    movl      %ecx, %eax
542:
    movw      %r8w, (%r11)
    addq      $2, %r11
    decl      %eax
    jnz       542b
    decq      %rdx
    jnz       541b
548:
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
