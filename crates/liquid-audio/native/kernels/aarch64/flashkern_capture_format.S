#include "lfm_asm_visibility.h"

// Realtime device capture conversion. The common mono/stereo/quad device
// layouts run across frames in NEON; uncommon channel counts retain the exact
// left-to-right channel order in the scalar FP lane. All paths are assembly —
// there is no C++ numerical fallback.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#else
#define LFM_SYM(name) name
#endif

.text
.p2align 2

// void lfm_capture_downmix_f32_dd(const float *source, float *destination,
//                                 size_t frames, uint32_t channels)
.globl LFM_SYM(lfm_capture_downmix_f32_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_f32_dd))
LFM_SYM(lfm_capture_downmix_f32_dd):
    ucvtf   s5, w3
    dup     v5.4s, v5.s[0]
    cmp     w3, #1
    b.eq    10f
    cmp     w3, #2
    b.eq    20f
    cmp     w3, #4
    b.eq    30f
    b       40f
10:
    cmp     x2, #4
    b.lo    40f
11:
    ldr     q0, [x0], #16
    str     q0, [x1], #16
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    11b
    b       40f
20:
    cmp     x2, #4
    b.lo    40f
21:
    ld2     {v0.4s, v1.4s}, [x0], #32
    fadd    v0.4s, v0.4s, v1.4s
    fdiv    v0.4s, v0.4s, v5.4s
    str     q0, [x1], #16
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    21b
    b       40f
30:
    cmp     x2, #4
    b.lo    40f
31:
    ld4     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v0.4s, v0.4s, v2.4s
    fadd    v0.4s, v0.4s, v3.4s
    fdiv    v0.4s, v0.4s, v5.4s
    str     q0, [x1], #16
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    31b
40:
    cbz     x2, 49f
41:
    fmov    s0, wzr
    uxtw    x4, w3
42:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    subs    x4, x4, #1
    b.ne    42b
    fdiv    s0, s0, s5
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    41b
49:
    ret

// void lfm_capture_downmix_i16_dd(const int16_t *source, float *destination,
//                                 size_t frames, uint32_t channels)
.p2align 2
.globl LFM_SYM(lfm_capture_downmix_i16_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_i16_dd))
LFM_SYM(lfm_capture_downmix_i16_dd):
    ucvtf   s5, w3
    mov     w8, #32767
    ucvtf   s6, w8
    fmul    s5, s5, s6
    dup     v5.4s, v5.s[0]
    cmp     w3, #1
    b.eq    110f
    cmp     w3, #2
    b.eq    120f
    cmp     w3, #4
    b.eq    130f
    b       140f
110:
    cmp     x2, #8
    b.lo    140f
111:
    ldr     q0, [x0], #16
    sxtl    v1.4s, v0.4h
    sxtl2   v2.4s, v0.8h
    scvtf   v1.4s, v1.4s
    scvtf   v2.4s, v2.4s
    fdiv    v1.4s, v1.4s, v5.4s
    fdiv    v2.4s, v2.4s, v5.4s
    stp     q1, q2, [x1], #32
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    111b
    b       140f
120:
    cmp     x2, #8
    b.lo    140f
121:
    ld2     {v0.8h, v1.8h}, [x0], #32
    sxtl    v2.4s, v0.4h
    sxtl    v3.4s, v1.4h
    sxtl2   v4.4s, v0.8h
    sxtl2   v6.4s, v1.8h
    add     v2.4s, v2.4s, v3.4s
    add     v4.4s, v4.4s, v6.4s
    scvtf   v2.4s, v2.4s
    scvtf   v4.4s, v4.4s
    fdiv    v2.4s, v2.4s, v5.4s
    fdiv    v4.4s, v4.4s, v5.4s
    stp     q2, q4, [x1], #32
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    121b
    b       140f
130:
    cmp     x2, #8
    b.lo    140f
131:
    ld4     {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
    sxtl    v4.4s, v0.4h
    sxtl    v6.4s, v1.4h
    add     v4.4s, v4.4s, v6.4s
    sxtl    v6.4s, v2.4h
    add     v4.4s, v4.4s, v6.4s
    sxtl    v6.4s, v3.4h
    add     v4.4s, v4.4s, v6.4s
    scvtf   v4.4s, v4.4s
    fdiv    v4.4s, v4.4s, v5.4s
    str     q4, [x1], #16
    sxtl2   v4.4s, v0.8h
    sxtl2   v6.4s, v1.8h
    add     v4.4s, v4.4s, v6.4s
    sxtl2   v6.4s, v2.8h
    add     v4.4s, v4.4s, v6.4s
    sxtl2   v6.4s, v3.8h
    add     v4.4s, v4.4s, v6.4s
    scvtf   v4.4s, v4.4s
    fdiv    v4.4s, v4.4s, v5.4s
    str     q4, [x1], #16
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    131b
140:
    cbz     x2, 149f
141:
    fmov    s0, wzr
    uxtw    x4, w3
142:
    ldrsh   w5, [x0], #2
    scvtf   s1, w5
    fadd    s0, s0, s1
    subs    x4, x4, #1
    b.ne    142b
    fdiv    s0, s0, s5
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    141b
149:
    ret

// void lfm_capture_downmix_u16_dd(const uint16_t *source, float *destination,
//                                 size_t frames, uint32_t channels)
.p2align 2
.globl LFM_SYM(lfm_capture_downmix_u16_dd)
LFM_PRIVATE(LFM_SYM(lfm_capture_downmix_u16_dd))
LFM_SYM(lfm_capture_downmix_u16_dd):
    ucvtf   s5, w3
    mov     w8, #32768
    ucvtf   s6, w8
    dup     v5.4s, v5.s[0]
    dup     v6.4s, v6.s[0]
    movi    v7.8h, #0x80, lsl #8
    cmp     w3, #1
    b.eq    210f
    cmp     w3, #2
    b.eq    220f
    cmp     w3, #4
    b.eq    230f
    b       240f
210:
    cmp     x2, #8
    b.lo    240f
211:
    ldr     q0, [x0], #16
    eor     v0.16b, v0.16b, v7.16b
    sxtl    v1.4s, v0.4h
    sxtl2   v2.4s, v0.8h
    scvtf   v1.4s, v1.4s
    scvtf   v2.4s, v2.4s
    fdiv    v1.4s, v1.4s, v6.4s
    fdiv    v2.4s, v2.4s, v6.4s
    fdiv    v1.4s, v1.4s, v5.4s
    fdiv    v2.4s, v2.4s, v5.4s
    stp     q1, q2, [x1], #32
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    211b
    b       240f
220:
    cmp     x2, #8
    b.lo    240f
221:
    ld2     {v0.8h, v1.8h}, [x0], #32
    eor     v0.16b, v0.16b, v7.16b
    eor     v1.16b, v1.16b, v7.16b
    sxtl    v2.4s, v0.4h
    sxtl    v3.4s, v1.4h
    sxtl2   v4.4s, v0.8h
    sxtl2   v0.4s, v1.8h
    add     v2.4s, v2.4s, v3.4s
    add     v4.4s, v4.4s, v0.4s
    scvtf   v2.4s, v2.4s
    scvtf   v4.4s, v4.4s
    fdiv    v2.4s, v2.4s, v6.4s
    fdiv    v4.4s, v4.4s, v6.4s
    fdiv    v2.4s, v2.4s, v5.4s
    fdiv    v4.4s, v4.4s, v5.4s
    stp     q2, q4, [x1], #32
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    221b
    b       240f
230:
    cmp     x2, #8
    b.lo    240f
231:
    ld4     {v0.8h, v1.8h, v2.8h, v3.8h}, [x0], #64
    eor     v0.16b, v0.16b, v7.16b
    eor     v1.16b, v1.16b, v7.16b
    eor     v2.16b, v2.16b, v7.16b
    eor     v3.16b, v3.16b, v7.16b
    sxtl    v4.4s, v0.4h
    sxtl    v6.4s, v1.4h
    add     v4.4s, v4.4s, v6.4s
    sxtl    v6.4s, v2.4h
    add     v4.4s, v4.4s, v6.4s
    sxtl    v6.4s, v3.4h
    add     v4.4s, v4.4s, v6.4s
    scvtf   v4.4s, v4.4s
    mov     w8, #32768
    ucvtf   s6, w8
    dup     v6.4s, v6.s[0]
    fdiv    v4.4s, v4.4s, v6.4s
    fdiv    v4.4s, v4.4s, v5.4s
    str     q4, [x1], #16
    sxtl2   v4.4s, v0.8h
    sxtl2   v6.4s, v1.8h
    add     v4.4s, v4.4s, v6.4s
    sxtl2   v6.4s, v2.8h
    add     v4.4s, v4.4s, v6.4s
    sxtl2   v6.4s, v3.8h
    add     v4.4s, v4.4s, v6.4s
    scvtf   v4.4s, v4.4s
    mov     w8, #32768
    ucvtf   s6, w8
    dup     v6.4s, v6.s[0]
    fdiv    v4.4s, v4.4s, v6.4s
    fdiv    v4.4s, v4.4s, v5.4s
    str     q4, [x1], #16
    // Rebuild the 32768 vector clobbered as quad-work scratch.
    mov     w8, #32768
    ucvtf   s6, w8
    dup     v6.4s, v6.s[0]
    subs    x2, x2, #8
    cmp     x2, #8
    b.hs    231b
240:
    cbz     x2, 249f
241:
    fmov    s0, wzr
    uxtw    x4, w3
242:
    ldrh    w5, [x0], #2
    ucvtf   s1, w5
    fsub    s1, s1, s6
    fdiv    s1, s1, s6
    fadd    s0, s0, s1
    subs    x4, x4, #1
    b.ne    242b
    fdiv    s0, s0, s5
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    241b
249:
    ret

// Playback leaves carry callback-local sum-of-squares through meter offset 16,
// rendered frames at offset 8, and publish aggregate RMS at offset 20. The
// reduction order remains source-frame order even on the vector output paths.

// void lfm_playback_render_f32_dd(const float *source, float *destination,
//     size_t frames, uint32_t channels, LfmPlaybackMeterV1 *meter)
.p2align 2
.globl LFM_SYM(lfm_playback_render_f32_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_render_f32_dd))
LFM_SYM(lfm_playback_render_f32_dd):
    ldr     x8, [x4, #8]
    add     x8, x8, x2
    str     x8, [x4, #8]
    ldr     s6, [x4, #16]
    cmp     w3, #1
    b.eq    310f
    cmp     w3, #2
    b.eq    320f
    b       340f
310:
    cmp     x2, #4
    b.lo    340f
311:
    ldr     q0, [x0], #16
    str     q0, [x1], #16
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    311b
    b       340f
320:
    cmp     x2, #4
    b.lo    340f
321:
    ldr     q0, [x0], #16
    mov     v1.16b, v0.16b
    st2     {v0.4s, v1.4s}, [x1], #32
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    321b
340:
    cbz     x2, 348f
341:
    ldr     s0, [x0], #4
    fmul    s1, s0, s0
    fadd    s6, s6, s1
    uxtw    x9, w3
342:
    str     s0, [x1], #4
    subs    x9, x9, #1
    b.ne    342b
    subs    x2, x2, #1
    b.ne    341b
348:
    str     s6, [x4, #16]
    scvtf   s0, x8
    fdiv    s1, s6, s0
    fsqrt   s1, s1
    str     s1, [x4, #20]
    ret

// void lfm_playback_render_i16_dd(const float *source, int16_t *destination,
//     size_t frames, uint32_t channels, LfmPlaybackMeterV1 *meter)
.p2align 2
.globl LFM_SYM(lfm_playback_render_i16_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_render_i16_dd))
LFM_SYM(lfm_playback_render_i16_dd):
    ldr     x8, [x4, #8]
    add     x8, x8, x2
    str     x8, [x4, #8]
    ldr     s6, [x4, #16]
    fmov    v4.4s, #1.0
    fneg    v5.4s, v4.4s
    mov     w11, #32767
    scvtf   s3, w11
    dup     v3.4s, v3.s[0]
    cmp     w3, #1
    b.eq    410f
    cmp     w3, #2
    b.eq    420f
    b       440f
410:
    cmp     x2, #4
    b.lo    440f
411:
    ldr     q0, [x0], #16
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    fcmeq   v2.4s, v0.4s, v0.4s
    and     v0.16b, v0.16b, v2.16b
    fmax    v0.4s, v0.4s, v5.4s
    fmin    v0.4s, v0.4s, v4.4s
    fmul    v0.4s, v0.4s, v3.4s
    fcvtzs  v0.4s, v0.4s
    sqxtn   v0.4h, v0.4s
    str     d0, [x1], #8
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    411b
    b       440f
420:
    cmp     x2, #4
    b.lo    440f
421:
    ldr     q0, [x0], #16
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    fcmeq   v2.4s, v0.4s, v0.4s
    and     v0.16b, v0.16b, v2.16b
    fmax    v0.4s, v0.4s, v5.4s
    fmin    v0.4s, v0.4s, v4.4s
    fmul    v0.4s, v0.4s, v3.4s
    fcvtzs  v0.4s, v0.4s
    sqxtn   v0.4h, v0.4s
    zip1    v1.8h, v0.8h, v0.8h
    str     q1, [x1], #16
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    421b
440:
    cbz     x2, 448f
441:
    ldr     s0, [x0], #4
    fmul    s1, s0, s0
    fadd    s6, s6, s1
    fcmp    s0, s0
    fmov    s2, wzr
    fcsel   s0, s0, s2, vc
    fmax    s0, s0, s5
    fmin    s0, s0, s4
    fmul    s0, s0, s3
    fcvtzs  w11, s0
    uxtw    x9, w3
442:
    strh    w11, [x1], #2
    subs    x9, x9, #1
    b.ne    442b
    subs    x2, x2, #1
    b.ne    441b
448:
    str     s6, [x4, #16]
    scvtf   s0, x8
    fdiv    s1, s6, s0
    fsqrt   s1, s1
    str     s1, [x4, #20]
    ret

// void lfm_playback_render_u16_dd(const float *source, uint16_t *destination,
//     size_t frames, uint32_t channels, LfmPlaybackMeterV1 *meter)
.p2align 2
.globl LFM_SYM(lfm_playback_render_u16_dd)
LFM_PRIVATE(LFM_SYM(lfm_playback_render_u16_dd))
LFM_SYM(lfm_playback_render_u16_dd):
    ldr     x8, [x4, #8]
    add     x8, x8, x2
    str     x8, [x4, #8]
    ldr     s6, [x4, #16]
    fmov    v4.4s, #1.0
    fneg    v5.4s, v4.4s
    mov     w11, #32767
    scvtf   s3, w11
    dup     v3.4s, v3.s[0]
    cmp     w3, #1
    b.eq    510f
    cmp     w3, #2
    b.eq    520f
    b       540f
510:
    cmp     x2, #4
    b.lo    540f
511:
    ldr     q0, [x0], #16
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    fcmeq   v2.4s, v0.4s, v0.4s
    and     v0.16b, v0.16b, v2.16b
    fmax    v0.4s, v0.4s, v5.4s
    fmin    v0.4s, v0.4s, v4.4s
    fmul    v0.4s, v0.4s, v3.4s
    fcvtzs  v0.4s, v0.4s
    mov     w11, #32768
    dup     v2.4s, w11
    add     v0.4s, v0.4s, v2.4s
    uqxtn   v0.4h, v0.4s
    str     d0, [x1], #8
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    511b
    b       540f
520:
    cmp     x2, #4
    b.lo    540f
521:
    ldr     q0, [x0], #16
    fmul    v1.4s, v0.4s, v0.4s
    fadd    s6, s6, s1
    dup     v2.4s, v1.s[1]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[2]
    fadd    s6, s6, s2
    dup     v2.4s, v1.s[3]
    fadd    s6, s6, s2
    fcmeq   v2.4s, v0.4s, v0.4s
    and     v0.16b, v0.16b, v2.16b
    fmax    v0.4s, v0.4s, v5.4s
    fmin    v0.4s, v0.4s, v4.4s
    fmul    v0.4s, v0.4s, v3.4s
    fcvtzs  v0.4s, v0.4s
    mov     w11, #32768
    dup     v2.4s, w11
    add     v0.4s, v0.4s, v2.4s
    uqxtn   v0.4h, v0.4s
    zip1    v1.8h, v0.8h, v0.8h
    str     q1, [x1], #16
    subs    x2, x2, #4
    cmp     x2, #4
    b.hs    521b
540:
    cbz     x2, 548f
541:
    ldr     s0, [x0], #4
    fmul    s1, s0, s0
    fadd    s6, s6, s1
    fcmp    s0, s0
    fmov    s2, wzr
    fcsel   s0, s0, s2, vc
    fmax    s0, s0, s5
    fmin    s0, s0, s4
    fmul    s0, s0, s3
    fcvtzs  w11, s0
    add     w11, w11, #8, lsl #12
    uxtw    x9, w3
542:
    strh    w11, [x1], #2
    subs    x9, x9, #1
    b.ne    542b
    subs    x2, x2, #1
    b.ne    541b
548:
    str     s6, [x4, #16]
    scvtf   s0, x8
    fdiv    s1, s6, s0
    fsqrt   s1, s1
    str     s1, [x4, #20]
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
