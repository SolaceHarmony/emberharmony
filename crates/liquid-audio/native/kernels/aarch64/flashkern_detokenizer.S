#include "lfm_asm_visibility.h"
#include "../../src/detokenizer/lfm_detokenizer_kernels.h"

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text
.p2align 2

// flashkern_detokenizer.S — AArch64 leaves of the LFM2.5 audio detokenizer
// (the vendor vocoder: codes -> embed -> mixer blocks -> iSTFT -> PCM).
// Orchestration and view binding live in src/detokenizer/lfm_detokenizer.cpp;
// the byte-view ABI (fixed argument-block offsets, records own no storage) is
// lfm_detokenizer_kernels.h, shared with the C++ side.
//
// THIS FILE'S NUMERIC CONTRACT IS NOT THE CONFORMER'S — read this before
// "fixing" anything:
//
//   conformer/frontend/math leaves   scalar, strictly ordered, NO FMA —
//                                    their oracle is candle's scalar op ladder
//   THIS file                        4-lane NEON, fmla, pairwise reductions —
//                                    its oracle is a fixed real-checkpoint code
//                                    trace (docs/AUDIO_DETOKENIZER_PORT.md)
//
// Both are FIXED-ORDER contracts; they are fixed to different references. The
// vector width, the fmla contraction, and the faddp reduction tree here are the
// committed rounding order the trace gate locks — de-vectorising this file, or
// splitting its fmla into fmul+fadd "for consistency", changes the stream and
// fails the gate exactly the way fusing the conformer's fmul+fadd would fail
// its fixtures. Same law, opposite conclusion.
//
// HOT PASSES ARE LIBM-FREE. The setup-only RoPE-inverse and inverse-DFT basis
// constructors below call platform powf/cos/sin from assembly and therefore
// retain the same platform-tied last-bit contract as the deleted C++ formulas.
// Every per-frame leaf consumes those immutable derived views without libm.
//
// The complete stream is exactly 1920 x code_frames samples (the port
// contract); every leaf writes its terminal value to a borrowed destination
// view before returning, allocates nothing, and never crosses a coroutine or
// C ABI boundary mid-tile.

// Returning leaves in this file own complete arithmetic tiles. They accept
// borrowed buffer views only, allocate nothing, and never cross a coroutine or
// C ABI boundary until the terminal value has been written.

// void lfm_detok_copy_f32(const float *src /*x0*/, float *dst /*x1*/,
//                         uint64_t count /*x2*/)
// Straight view copy, 16/4/1-lane tiers. Movement only — no rounding points.
.globl LFM_SYM(lfm_detok_copy_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_copy_f32))
LFM_SYM(lfm_detok_copy_f32):
1:  cmp     x2, #16
    b.lo    2f
    ld1     {v0.4s-v3.4s}, [x0], #64
    st1     {v0.4s-v3.4s}, [x1], #64
    sub     x2, x2, #16
    b       1b
2:  cmp     x2, #4
    b.lo    3f
    ld1     {v0.4s}, [x0], #16
    st1     {v0.4s}, [x1], #16
    sub     x2, x2, #4
    b       2b
3:  cbz     x2, 9f
4:  ldr     s0, [x0], #4
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    4b
9:  ret

.p2align 2
// void lfm_detok_add_f32(float *accum /*x0*/, const float *addend /*x1*/,
//                        uint64_t count /*x2*/)
// In-place accumulate, 4-lane then scalar tail. Lane-parallel addition is part
// of this file's fixed order (see header) — do not serialise it.
.globl LFM_SYM(lfm_detok_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_add_f32))
LFM_SYM(lfm_detok_add_f32):
1:  cmp     x2, #4
    b.lo    2f
    ld1     {v0.4s}, [x0]
    ld1     {v1.4s}, [x1], #16
    fadd    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x0], #16
    sub     x2, x2, #4
    b       1b
2:  cbz     x2, 9f
3:  ldr     s0, [x0]
    ldr     s1, [x1], #4
    fadd    s0, s0, s1
    str     s0, [x0], #4
    subs    x2, x2, #1
    b.ne    3b
9:  ret

.p2align 2
// void lfm_detok_embed_f32(args /*x0: LFM_DETOK_EMBED_* block*/)
// Code embedding: sums the EIGHT codebook embedding rows (x1..x8) and scales
// by 0.125 — the MEAN over the eight RVQ codebooks, s2 = 1/8 exactly
// (a power of two; the scale is lossless). The averaged 512-float row is then
// broadcast to six destination rows 2048 bytes apart — the six mixer blocks
// each receive an identical private copy of the conditioning row.
.globl LFM_SYM(lfm_detok_embed_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_embed_f32))
LFM_SYM(lfm_detok_embed_f32):
    ldp     x1, x2, [x0, #0]
    ldp     x3, x4, [x0, #16]
    ldp     x5, x6, [x0, #32]
    ldp     x7, x8, [x0, #48]
    ldr     x9, [x0, #LFM_DETOK_EMBED_OUTPUT]
    ldr     x10, [x0, #LFM_DETOK_EMBED_COUNT]
    fmov    s2, #0.125
1:  cbz     x10, 9f
    movi    v0.4s, #0
    ld1     {v1.4s}, [x1], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x2], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x3], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x4], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x5], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x6], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x7], #16
    fadd    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x8], #16
    fadd    v0.4s, v0.4s, v1.4s
    fmul    v0.4s, v0.4s, v2.s[0]
    str     q0, [x9, #0]
    str     q0, [x9, #2048]
    str     q0, [x9, #4096]
    str     q0, [x9, #6144]
    str     q0, [x9, #8192]
    str     q0, [x9, #10240]
    add     x9, x9, #16
    subs    x10, x10, #4
    b.ne    1b
9:  ret

.p2align 2
// void lfm_detok_rms_f32(args /*x0: LFM_DETOK_RMS_* block*/)
// RMSNorm: out = in * weight / sqrt(sumsq/cols + eps).
// FOURTH normaliser convention in the tree (conformer.S header lists the other
// three): eps INSIDE the sqrt like math.S's inv_rms, but the sum of squares is
// accumulated in TWO fmla vector lanes folded by faddp — a pairwise tree, not
// a sequential walk. That reduction order is this file's committed contract.
.globl LFM_SYM(lfm_detok_rms_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rms_f32))
LFM_SYM(lfm_detok_rms_f32):
    ldr     x1, [x0, #LFM_DETOK_RMS_INPUT]
    ldr     x2, [x0, #LFM_DETOK_RMS_WEIGHT]
    ldr     x3, [x0, #LFM_DETOK_RMS_OUTPUT]
    ldr     x4, [x0, #LFM_DETOK_RMS_COLUMNS]
    ldr     s7, [x0, #LFM_DETOK_RMS_EPSILON]
    mov     x5, x1
    mov     x6, x4
    movi    v0.4s, #0
    movi    v1.4s, #0
1:  cbz     x6, 2f
    ld1     {v2.4s-v3.4s}, [x5], #32
    fmla    v0.4s, v2.4s, v2.4s
    fmla    v1.4s, v3.4s, v3.4s
    sub     x6, x6, #8
    b       1b
2:  fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    ucvtf   s1, x4
    fdiv    s0, s0, s1
    fadd    s0, s0, s7
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0
3:  cbz     x4, 9f
    ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    fmul    v1.4s, v1.4s, v2.4s
    fmul    v1.4s, v1.4s, v0.s[0]
    st1     {v1.4s}, [x3], #16
    sub     x4, x4, #4
    b       3b
9:  ret

.p2align 2
// void lfm_detok_swiglu_f32(float *gate /*x0, in-place*/, const float *up /*x1*/,
//                            const float *exp_neg_gate /*x2*/, uint64_t count /*x3*/)
// SwiGLU gate for the backbone FF (config: block_use_swiglu, ff 3328):
//     out = gate * sigmoid(gate) * up,   sigmoid(g) = 1 / (1 + exp(-g))
// The transcendental is NOT here: x2 is exp(-gate) precomputed by the C++
// owner, so this leaf is pure mul/div/add — the header's hot-pass libm-free
// rule. detokenizer.py delegates this to Lfm2Model's FF; the split point
// (exp upstream, algebra here) is this port's committed rounding order.
.globl LFM_SYM(lfm_detok_swiglu_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_swiglu_f32))
LFM_SYM(lfm_detok_swiglu_f32):
    fmov    v4.4s, #1.0
1:  cbz     x3, 9f
    ld1     {v0.4s}, [x0]
    ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    fadd    v2.4s, v2.4s, v4.4s
    fdiv    v2.4s, v4.4s, v2.4s
    fmul    v0.4s, v0.4s, v2.4s
    fmul    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x0], #16
    sub     x3, x3, #4
    b       1b
9:  ret

.p2align 2
// float lfm_detok_dot32_scaled_f32(const float *q /*x0*/, const float *k /*x1*/,
//                                   float scale /*s0*/)
// One attention score: dot(q, k) * scale over exactly 32 floats (head_dim 32 =
// 512 dim / 16 heads; scale = 1/sqrt(32)). Two 4-lane fmla accumulators over
// 8-float strides, then faddp pairwise collapse — a FIXED vectorized reduction
// tree, committed by the trace gate (see header: this is the anti-conformer
// contract; do not "simplify" to a sequential scalar loop).
.globl LFM_SYM(lfm_detok_dot32_scaled_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_dot32_scaled_f32))
LFM_SYM(lfm_detok_dot32_scaled_f32):
    fmov    s6, s0
    movi    v0.4s, #0
    movi    v1.4s, #0
    mov     x2, #4
1:  ld1     {v2.4s-v3.4s}, [x0], #32
    ld1     {v4.4s-v5.4s}, [x1], #32
    fmla    v0.4s, v2.4s, v4.4s
    fmla    v1.4s, v3.4s, v5.4s
    subs    x2, x2, #1
    b.ne    1b
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    fmul    s0, s0, s6
    ret

.p2align 2
// float lfm_detok_max_f32(const float *x /*x0*/, uint64_t count /*x1*/)
// Softmax guard maximum, scalar sequential. NaN semantics are asymmetric and
// deliberate-by-omission: plain fcmp + b.le means a NaN at x[0] sticks (every
// unordered compare keeps it) while a NaN later never replaces the running
// max. Corruption detection belongs to the real-checkpoint trace gate, not to
// this leaf — contrast lfm_sampler_argmax_*, where NaN must not win because a
// poisoned logit would pin token choice in PRODUCTION output.
.globl LFM_SYM(lfm_detok_max_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_max_f32))
LFM_SYM(lfm_detok_max_f32):
    ldr     s0, [x0], #4
    sub     x1, x1, #1
1:  cbz     x1, 9f
    ldr     s1, [x0], #4
    fcmp    s1, s0
    b.le    2f
    fmov    s0, s1
2:  sub     x1, x1, #1
    b       1b
9:  ret

.p2align 2
// void lfm_detok_subtract_f32(float *x /*x0, in-place*/, uint64_t count /*x1*/,
//                             float value /*s0*/)
// x[i] -= value: the softmax max-subtraction (overflow guard before the C++
// owner exponentiates). 4-lane body + scalar tail.
.globl LFM_SYM(lfm_detok_subtract_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_subtract_f32))
LFM_SYM(lfm_detok_subtract_f32):
    dup     v2.4s, v0.s[0]
1:  cmp     x1, #4
    b.lo    2f
    ld1     {v1.4s}, [x0]
    fsub    v1.4s, v1.4s, v2.4s
    st1     {v1.4s}, [x0], #16
    sub     x1, x1, #4
    b       1b
2:  cbz     x1, 9f
3:  ldr     s1, [x0]
    fsub    s1, s1, s0
    str     s1, [x0], #4
    subs    x1, x1, #1
    b.ne    3b
9:  ret

.p2align 2
// float lfm_detok_sum_f32(const float *x /*x0*/, uint64_t count /*x1*/)
// Softmax denominator: 4-lane fadd accumulator + faddp pairwise collapse +
// scalar tail. Vectorized-pairwise IS the committed order here (header rule) —
// the frontend/math leaves sum strictly sequentially against a different
// oracle. Same law, different reference, opposite instruction choice.
.globl LFM_SYM(lfm_detok_sum_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_sum_f32))
LFM_SYM(lfm_detok_sum_f32):
    movi    v0.4s, #0
1:  cmp     x1, #4
    b.lo    2f
    ld1     {v1.4s}, [x0], #16
    fadd    v0.4s, v0.4s, v1.4s
    sub     x1, x1, #4
    b       1b
2:  faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    cbz     x1, 9f
3:  ldr     s1, [x0], #4
    fadd    s0, s0, s1
    subs    x1, x1, #1
    b.ne    3b
9:  ret

.p2align 2
// void lfm_detok_normalize_f32(float *x /*x0, in-place*/, uint64_t count /*x1*/,
//                              float sum /*s0*/)
// Softmax normalize: computes 1/sum ONCE, then multiplies every weight —
// reciprocal-multiply, exactly what frontend norm_apply refuses to do (it
// divides per element). Not an inconsistency: each is the committed rounding
// of ITS reference. The trace gate locked this one with the reciprocal.
.globl LFM_SYM(lfm_detok_normalize_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_normalize_f32))
LFM_SYM(lfm_detok_normalize_f32):
    fmov    s2, #1.0
    fdiv    s0, s2, s0
1:  cmp     x1, #4
    b.lo    2f
    ld1     {v1.4s}, [x0]
    fmul    v1.4s, v1.4s, v0.s[0]
    st1     {v1.4s}, [x0], #16
    sub     x1, x1, #4
    b       1b
2:  cbz     x1, 9f
3:  ldr     s1, [x0]
    fmul    s1, s1, s0
    str     s1, [x0], #4
    subs    x1, x1, #1
    b.ne    3b
9:  ret

.p2align 2
// void lfm_detok_rope_angles_f32(const float *inv_freq /*x0*/, float *out /*x1*/,
//                                uint64_t count /*x2*/, uint64_t position /*x3*/)
// Per-position rotary angles: out[i] = inv_freq[i] * (float)position, where
// inv_freq is the theta^(-2i/head) table built once by rope_inverse below.
// cos/sin of these angles are produced by the C++ owner (hot-pass libm rule).
.globl LFM_SYM(lfm_detok_rope_angles_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_angles_f32))
LFM_SYM(lfm_detok_rope_angles_f32):
    ucvtf   s0, x3
1:  cbz     x2, 9f
    ld1     {v1.4s}, [x0], #16
    fmul    v1.4s, v1.4s, v0.s[0]
    st1     {v1.4s}, [x1], #16
    sub     x2, x2, #4
    b       1b
9:  ret

.p2align 2
// void lfm_detok_rope_f32(float *head /*x0, in-place*/, const float *cos /*x1*/,
//                         const float *sin /*x2*/)
// Rotary embedding, NeoX split-half pairing over one 32-float head: element i
// rotates against element i+16 (the q1 load at [x0,#64] = 16 floats in).
//     lo' = lo*cos - hi*sin ;  hi' = hi*cos + lo*sin
// Full F32 throughout — no per-step BF16 rounding. Contrast lfm_bf16_rope_neox
// in flashkern_math.S (six RNE points per pair): the backbone model was
// trained through BF16 materialization; this detokenizer's reference runs
// torch F32, so carrying full precision here is the FAITHFUL choice, not an
// upgrade. Fixed 4x4-lane iteration order.
.globl LFM_SYM(lfm_detok_rope_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_f32))
LFM_SYM(lfm_detok_rope_f32):
    mov     x3, #4
1:  ld1     {v0.4s}, [x0]
    ldr     q1, [x0, #64]
    ld1     {v2.4s}, [x1], #16
    ld1     {v3.4s}, [x2], #16
    fmul    v4.4s, v0.4s, v2.4s
    fmul    v6.4s, v1.4s, v3.4s
    fsub    v4.4s, v4.4s, v6.4s
    fmul    v5.4s, v1.4s, v2.4s
    fmul    v6.4s, v0.4s, v3.4s
    fadd    v5.4s, v5.4s, v6.4s
    str     q4, [x0]
    str     q5, [x0, #64]
    add     x0, x0, #16
    subs    x3, x3, #1
    b.ne    1b
    ret

.p2align 2
// int lfm_detok_rope_inverse_f32(float *out /*x0*/, uint64_t count /*x1*/,
//                                uint64_t head_dim /*x2*/, float theta /*s0*/)
// SETUP-ONLY (plan build, once): out[i] = theta^(-2i/head_dim) via libm powf —
// one of the file's two platform-tied builders (see header). The derived view
// is immutable; hot passes only read it. Validates pointers, count, and
// theta > 0 (b.vs catches NaN); returns 0 or -EINVAL.
.globl LFM_SYM(lfm_detok_rope_inverse_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_rope_inverse_f32))
LFM_SYM(lfm_detok_rope_inverse_f32):
    cbz     x0, 8f
    cbz     x1, 8f
    cbz     x2, 8f
    fcmp    s0, #0.0
    b.le    8f
    b.vs    8f
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    str     d8, [sp, #48]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, #0
    fmov    s8, s0
1:  ucvtf   s1, x22
    fadd    s1, s1, s1
    fneg    s1, s1
    ucvtf   s2, x21
    fdiv    s1, s1, s2
    fmov    s0, s8
    bl      LFM_CALL(powf)
    str     s0, [x19], #4
    add     x22, x22, #1
    cmp     x22, x20
    b.lo    1b
    mov     w0, #0
    ldr     d8, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret
8:  mov     w0, #-22
    ret

.p2align 2
// int lfm_detok_ifft_basis_f32(float *basis /*x0: [real|imag] planes*/,
//                              uint64_t bins /*x1*/, uint64_t n_fft /*x2*/)
// SETUP-ONLY (plan build, once): the one-sided inverse-rFFT basis realizing
// torch.fft.irfft(spec, n_fft, norm="backward") as two GEMMs (real plane,
// imag plane) — detokenizer.py ISTFT.forward line 82. Per bin k, sample n:
//     real[k][n] = doubling *  cos(2*pi*k*n/n_fft) / n_fft
//     imag[k][n] = doubling * -sin(2*pi*k*n/n_fft) / n_fft
// doubling = 2 except k==0 and k==n_fft/2 (DC/Nyquist appear once in the
// hermitian expansion; interior one-sided bins stand for a conjugate pair).
// The -sin encodes the conjugate: spec is one-sided, x = Re(sum spec_k e^+).
// Guards bins == n_fft/2 + 1 (641 for 1280). Built in FULL F64 — 2*pi as the
// exact 0x401921FB54442D18, libm cos/sin on doubles — narrowed to f32 once
// per entry. The file's second platform-tied builder (header note); hot
// passes read the immutable planes through the Accelerate GEMM seam only.
.globl LFM_SYM(lfm_detok_ifft_basis_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_ifft_basis_f32))
LFM_SYM(lfm_detok_ifft_basis_f32):
    cbz     x0, 8f
    cbz     x1, 8f
    cbz     x2, 8f
    lsr     x3, x2, #1
    add     x3, x3, #1
    cmp     x1, x3
    b.ne    8f
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    mov     x19, x0
    mul     x25, x1, x2
    add     x20, x0, x25, lsl #2
    mov     x21, x1
    mov     x22, x2
    movz    x25, #0x2d18
    movk    x25, #0x5444, lsl #16
    movk    x25, #0x21fb, lsl #32
    movk    x25, #0x4019, lsl #48
    fmov    d8, x25
    ucvtf   d0, x22
    fmov    d1, #1.0
    fdiv    d9, d1, d0
    mov     x23, #0
1:  fmov    d11, #2.0
    cbz     x23, 2f
    lsl     x25, x23, #1
    cmp     x25, x22
    b.ne    3f
2:  fmov    d11, #1.0
3:  mov     x24, #0
4:  mul     x25, x23, x24
    ucvtf   d10, x25
    fmul    d10, d10, d8
    ucvtf   d0, x22
    fdiv    d10, d10, d0
    fmov    d0, d10
    bl      LFM_CALL(cos)
    fmul    d0, d0, d11
    fmul    d0, d0, d9
    fcvt    s0, d0
    str     s0, [x19], #4
    fmov    d0, d10
    bl      LFM_CALL(sin)
    fmul    d0, d0, d11
    fmul    d0, d0, d9
    fneg    d0, d0
    fcvt    s0, d0
    str     s0, [x20], #4
    add     x24, x24, #1
    cmp     x24, x22
    b.lo    4b
    add     x23, x23, #1
    cmp     x23, x21
    b.lo    1b
    mov     w0, #0
    ldp     d10, d11, [sp, #112]
    ldp     d8, d9, [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #128
    ret
8:  mov     w0, #-22
    ret

.p2align 2
// void lfm_detok_conv_f32(args /*x0: LFM_DETOK_CONV_* block*/)
// The LFM2 gated short convolution (config: conv_L_cache=3), one 512-lane
// step:  out = C (.) ( w0*s0 + w1*s1 + w2*(B (.) x) )
// where the projected block holds three 512-float planes (+0, +2048, +4096
// bytes) = the B/C/x projections, weights are ld3-interleaved 3-tap kernels
// per lane, and the carry holds the two previous gated inputs s0=t-2, s1=t-1.
// After the tap sum the carry ROTATES IN PLACE: s0 <- s1, s1 <- s2's slot,
// current (B.x) becomes the newest entry — the streaming-state equivalent of
// the reference's causal width-3 conv, no re-read of history.
.globl LFM_SYM(lfm_detok_conv_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_conv_f32))
LFM_SYM(lfm_detok_conv_f32):
    ldr     x1, [x0, #LFM_DETOK_CONV_PROJECTED]
    ldr     x2, [x0, #LFM_DETOK_CONV_CARRY]
    ldr     x3, [x0, #LFM_DETOK_CONV_WEIGHT]
    ldr     x4, [x0, #LFM_DETOK_CONV_OUTPUT]
    ldr     x5, [x0, #LFM_DETOK_CONV_COUNT]
    add     x6, x1, #2048
    add     x7, x1, #4096
    add     x8, x2, #2048
    add     x9, x2, #4096
1:  cbz     x5, 9f
    ld1     {v0.4s}, [x1], #16
    ld1     {v1.4s}, [x7], #16
    fmul    v0.4s, v0.4s, v1.4s
    ld1     {v1.4s}, [x8], #16
    ld1     {v2.4s}, [x9], #16
    ld3     {v3.4s-v5.4s}, [x3], #48
    fmul    v6.4s, v1.4s, v3.4s
    fmla    v6.4s, v2.4s, v4.4s
    fmla    v6.4s, v0.4s, v5.4s
    ld1     {v7.4s}, [x6], #16
    fmul    v6.4s, v6.4s, v7.4s
    st1     {v6.4s}, [x4], #16
    st1     {v1.4s}, [x2], #16
    stur    q2, [x8, #-16]
    stur    q0, [x9, #-16]
    sub     x5, x5, #4
    b       1b
9:  ret

.p2align 2
// void lfm_detok_weighted_f32(args /*x0: LFM_DETOK_WEIGHTED_* block*/)
// Attention weighted-V accumulation for one head: out[32] = sum_j scores[j] *
// V[ (first+j) mod 30 ][head]. The mod-30 ring IS the reference's sliding
// window (detokenizer.py: sliding_window_size=30; mask d_idx<=0 && >-30) —
// the KV cache is a 30-entry ring, and (first+j) wraps rather than shifts.
// V layout: [position][head][32] (position stride 1024B, head stride 128B).
// Eight 4-lane fmla accumulators = the full 32-float head held in registers;
// ascending-j accumulation is the committed order.
.globl LFM_SYM(lfm_detok_weighted_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_weighted_f32))
LFM_SYM(lfm_detok_weighted_f32):
    ldr     x1, [x0, #LFM_DETOK_WEIGHTED_VALUES]
    ldr     x2, [x0, #LFM_DETOK_WEIGHTED_SCORES]
    ldr     x3, [x0, #LFM_DETOK_WEIGHTED_OUTPUT]
    ldr     x4, [x0, #LFM_DETOK_WEIGHTED_FIRST]
    ldr     x5, [x0, #LFM_DETOK_WEIGHTED_COUNT]
    ldr     x6, [x0, #LFM_DETOK_WEIGHTED_HEAD]
    add     x1, x1, x6, lsl #7
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    mov     x7, #0
1:  cmp     x7, x5
    b.hs    2f
    add     x8, x4, x7
    cmp     x8, #30
    b.lo    3f
    sub     x8, x8, #30
3:
    add     x9, x1, x8, lsl #10
    ldr     s16, [x2, x7, lsl #2]
    ld1     {v17.4s}, [x9], #16
    fmla    v0.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v1.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v2.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v3.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v4.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v5.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9], #16
    fmla    v6.4s, v17.4s, v16.s[0]
    ld1     {v17.4s}, [x9]
    fmla    v7.4s, v17.4s, v16.s[0]
    add     x7, x7, #1
    b       1b
2:  st1     {v0.4s-v3.4s}, [x3], #64
    st1     {v4.4s-v7.4s}, [x3]
    ret

.p2align 2
// void lfm_detok_polar_f32(args /*x0: LFM_DETOK_POLAR_* block*/)
// torch.polar(exp(log_abs), angle) — the multiply half only:
//     real = magnitude * cos ;  imag = magnitude * sin
// exp(log_abs), cos(angle), sin(angle) are computed by the C++ owner into
// planes first (hot-pass libm rule); this leaf never sees a transcendental.
// Feeds the ifft_basis GEMMs. detokenizer.py line 134.
.globl LFM_SYM(lfm_detok_polar_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_polar_f32))
LFM_SYM(lfm_detok_polar_f32):
    ldr     x1, [x0, #LFM_DETOK_POLAR_MAGNITUDE]
    ldr     x2, [x0, #LFM_DETOK_POLAR_SINE]
    ldr     x3, [x0, #LFM_DETOK_POLAR_COSINE]
    ldr     x4, [x0, #LFM_DETOK_POLAR_REAL]
    ldr     x5, [x0, #LFM_DETOK_POLAR_IMAGINARY]
    ldr     x6, [x0, #LFM_DETOK_POLAR_COUNT]
1:  cmp     x6, #4
    b.lo    2f
    ld1     {v0.4s}, [x1], #16
    ld1     {v1.4s}, [x2], #16
    ld1     {v2.4s}, [x3], #16
    fmul    v2.4s, v0.4s, v2.4s
    fmul    v1.4s, v0.4s, v1.4s
    st1     {v2.4s}, [x4], #16
    st1     {v1.4s}, [x5], #16
    sub     x6, x6, #4
    b       1b
2:  cbz     x6, 9f
3:  ldr     s0, [x1], #4
    ldr     s1, [x2], #4
    ldr     s2, [x3], #4
    fmul    s2, s0, s2
    fmul    s1, s0, s1
    str     s2, [x4], #4
    str     s1, [x5], #4
    subs    x6, x6, #1
    b.ne    3b
9:  ret

.p2align 2
// void lfm_detok_overlap_f32(args /*x0: LFM_DETOK_OVERLAP_* block*/)
// Both folds of the Vocos iSTFT in one pass (detokenizer.py lines 87-101):
//     output[i]   += frame[i]  * window[i]      // overlap-add of the frame
//     envelope[i] += window[i] * window[i]      // the NOLA window envelope
// window is Hann(1280), hop 320; the caller advances output/envelope by hop
// per frame. Accumulating the envelope alongside the signal (rather than
// precomputing it) keeps the two folds bit-coupled to the same traversal.
.globl LFM_SYM(lfm_detok_overlap_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_overlap_f32))
LFM_SYM(lfm_detok_overlap_f32):
    ldr     x1, [x0, #LFM_DETOK_OVERLAP_SIGNAL]
    ldr     x2, [x0, #LFM_DETOK_OVERLAP_WINDOW]
    ldr     x3, [x0, #LFM_DETOK_OVERLAP_OUTPUT]
    ldr     x4, [x0, #LFM_DETOK_OVERLAP_ENVELOPE]
    ldr     x5, [x0, #LFM_DETOK_OVERLAP_COUNT]
1:  cmp     x5, #4
    b.lo    2f
    ld1     {v0.4s}, [x1], #16
    ld1     {v1.4s}, [x2], #16
    ld1     {v2.4s}, [x3]
    ld1     {v3.4s}, [x4]
    fmla    v2.4s, v0.4s, v1.4s
    fmla    v3.4s, v1.4s, v1.4s
    st1     {v2.4s}, [x3], #16
    st1     {v3.4s}, [x4], #16
    sub     x5, x5, #4
    b       1b
2:  cbz     x5, 9f
3:  ldr     s0, [x1], #4
    ldr     s1, [x2], #4
    ldr     s2, [x3]
    ldr     s3, [x4]
    fmul    s0, s0, s1
    fadd    s2, s2, s0
    fmul    s1, s1, s1
    fadd    s3, s3, s1
    str     s2, [x3], #4
    str     s3, [x4], #4
    subs    x5, x5, #1
    b.ne    3b
9:  ret

.p2align 2
// int lfm_detok_emit_f32(args /*x0: LFM_DETOK_EMIT_* block*/)
// The final divide + NOLA guard: pcm[i] = output[i] / envelope[i], but ANY
// envelope value <= epsilon aborts with -ERANGE first (fminv per 4-lane
// block, scalar compare on the tail). This is the reference's
//     assert (window_envelope > 1e-11).all()
// (detokenizer.py line 104) promoted from a Python assert to a hard typed
// error — a NOLA violation means the window/hop geometry is broken and the
// PCM would be garbage; fail loudly, never emit it. Returns 0 or -ERANGE.
.globl LFM_SYM(lfm_detok_emit_f32)
LFM_PRIVATE(LFM_SYM(lfm_detok_emit_f32))
LFM_SYM(lfm_detok_emit_f32):
    ldr     x1, [x0, #LFM_DETOK_EMIT_OUTPUT]
    ldr     x2, [x0, #LFM_DETOK_EMIT_ENVELOPE]
    ldr     x3, [x0, #LFM_DETOK_EMIT_PCM]
    ldr     x4, [x0, #LFM_DETOK_EMIT_COUNT]
    ldr     s4, [x0, #LFM_DETOK_EMIT_EPSILON]
1:  cmp     x4, #4
    b.lo    2f
    ld1     {v0.4s}, [x2]
    fminv   s1, v0.4s
    fcmp    s1, s4
    b.le    8f
    ld1     {v1.4s}, [x1], #16
    fdiv    v1.4s, v1.4s, v0.4s
    st1     {v1.4s}, [x3], #16
    add     x2, x2, #16
    sub     x4, x4, #4
    b       1b
2:  cbz     x4, 7f
3:  ldr     s0, [x2], #4
    fcmp    s0, s4
    b.le    8f
    ldr     s1, [x1], #4
    fdiv    s1, s1, s0
    str     s1, [x3], #4
    subs    x4, x4, #1
    b.ne    3b
7:  mov     w0, #0
    ret
8:  mov     w0, #-34
    ret
