#include "lfm_asm_visibility.h"

// Audio-frontend leaves (mel featurizer + resampler hot loops). Scalar on
// purpose: these run once per utterance, not per token, and the parity oracle
// (native/tests/fixtures/mel) was captured from strictly-ordered scalar-
// equivalent tensor ops. Separate fmul/fadd everywhere — no FMA contraction:
// the deleted Rust reference never fused, and -ffp-contract=off guards the C++
// side of the same contract.
//
// WHY NO-FMA IS LOAD-BEARING, not stylistic: `fmadd` keeps the product at full
// internal width and rounds once; separate `fmul` + `fadd` rounds twice. The
// results differ in the last bits, so a compiler permitted to contract would
// silently walk the mel plane off the captured fixtures. This is why the leaves
// are hand-written rather than left to the optimiser — and why an "obvious"
// fmul/fadd fusion here is a regression, not a speedup.
//
// PLATFORM TIE — the one thing that is NOT bit-portable in this file:
//
// `lfm_log_add_f32` calls the platform libm `logf`. Apple libm, glibc and musl
// are not required to agree in the last ulp, so the mel plane — and therefore
// every downstream feature and any end-to-end byte oracle — is reproducible on
// a given platform but not guaranteed across them. flashkern_rope.S has the
// same property via powf/cosf/sinf, and flashkern_conformer.S has it most of
// all (expf/erff/sinf/cosf). Those THREE files are the whole libm surface of
// the kernel set; math/prng/sampler/sesame are libm-free and bit-identical
// everywhere. If a hash differs between machines with everything else equal,
// those three are where to look.

#ifdef __APPLE__
#define LFM_SYM(name) _##name
#define LFM_CALL(name) _##name
#else
#define LFM_SYM(name) name
#define LFM_CALL(name) name
#endif

.text
.p2align 2

// int lfm_preemph_spans_f32(const LfmF32SpanChain *x, float *y,
//                           float coef /*s0*/)
// The fixed chain is laid out as {u32 count,u32 reserved,u64 total,
// {ptr,u64 len}[2]}. Carrying the previous ORIGINAL sample in s3 makes the slab
// boundary numerically invisible. The result is one mathematically-required
// preemphasis plane, never a concatenated PCM staging plane.
.globl LFM_SYM(lfm_preemph_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_preemph_spans_f32))
LFM_SYM(lfm_preemph_spans_f32):
    cbz     x0, 9f
    cbz     x1, 9f
    ldr     w2, [x0]             // span count, validated as 1 or 2 by C++
    cbz     w2, 9f
    cmp     w2, #2
    b.hi    9f
    add     x3, x0, #16          // first {data,length}
    ldr     x4, [x3]
    ldr     x5, [x3, #8]
    cbz     x4, 9f
    cbz     x5, 9f
    ldr     s3, [x4]             // previous original sample
    str     s3, [x1], #4
    mov     x6, #1               // index within current span
    mov     w7, #0               // current span
1:
    cmp     x6, x5
    b.lo    3f
    add     w7, w7, #1
    cmp     w7, w2
    b.hs    2f
    add     x3, x3, #16
    ldr     x4, [x3]
    ldr     x5, [x3, #8]
    cbz     x4, 9f
    cbz     x5, 9f
    mov     x6, #0
3:
    ldr     s1, [x4, x6, lsl #2]
    fmul    s2, s3, s0
    fsub    s2, s1, s2
    str     s2, [x1], #4
    fmov    s3, s1
    add     x6, x6, #1
    b       1b
2:
    mov     w0, #0
    ret
9:
    mov     w0, #-22
    ret

// void lfm_frame_gather_spans_f32(chain, logical_offset, logical_limit,
//                                  center, hop, n_fft, cols, out)
// Writes the transposed frame plane out[n*cols+t]. The selected sample is read
// directly from either borrowed slab; logical padding is emitted as zero.
.p2align 2
.globl LFM_SYM(lfm_frame_gather_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_frame_gather_spans_f32))
LFM_SYM(lfm_frame_gather_spans_f32):
    ldr     x8, [x0, #16]        // span 0 data
    ldr     x9, [x0, #24]        // span 0 length
    ldr     x10, [x0, #32]       // span 1 data (zero for one span)
    mov     x11, #0              // n
1:
    cmp     x11, x5
    b.hs    8f
    mov     x12, #0              // t
2:
    cmp     x12, x6
    b.hs    7f
    madd    x13, x12, x4, x11   // idx = t*hop+n
    cmp     x13, x3
    b.lo    6f
    sub     x14, x13, x3        // source after centered padding
    cmp     x14, x1
    b.lo    6f
    cmp     x14, x2
    b.hs    6f
    sub     x15, x14, x1        // physical logical-chain index
    cmp     x15, x9
    b.hs    5f
    ldr     s0, [x8, x15, lsl #2]
    b       4f
5:
    sub     x15, x15, x9
    ldr     s0, [x10, x15, lsl #2]
    b       4f
6:
    fmov    s0, wzr
4:
    str     s0, [x7], #4
    add     x12, x12, #1
    b       2b
7:
    add     x11, x11, #1
    b       1b
8:
    ret

// void lfm_power_spec_f32(const float *re, const float *im, float *out, uint64_t n)
// out[i] = re[i]*re[i] + im[i]*im[i].
//
// The POWER spectrum, |X|^2 — this is what `mag_power == 2` names in the
// frontend config. Magnitude (|X|, mag_power 1.0) is not implemented and is
// refused at create in lfm_frontend.cpp rather than silently returning power.
// Two separate fmuls and an fadd, never fused (see header).
.p2align 2
.globl LFM_SYM(lfm_power_spec_f32)
LFM_PRIVATE(LFM_SYM(lfm_power_spec_f32))
LFM_SYM(lfm_power_spec_f32):
    cbz     x3, 8f
1:
    ldr     s0, [x0], #4
    ldr     s1, [x1], #4
    fmul    s0, s0, s0
    fmul    s1, s1, s1
    fadd    s0, s0, s1
    str     s0, [x2], #4
    subs    x3, x3, #1
    b.ne    1b
8:
    ret

// int lfm_log_add_f32(float *x, uint64_t n, float guard /*s0*/)
// x[i] = logf(x[i] + guard), in place. Calls libm logf (rope-leaf pattern).
//
// The log-mel step. `guard` is added BEFORE the log — that ordering is the
// contract, and it is what keeps a silent bin (power 0) from producing -inf and
// poisoning the row statistics downstream. Folding the guard in afterwards, or
// clamping instead of adding, changes every value.
//
// This is the file's only libm call and therefore its only non-portable step:
// see the PLATFORM TIE note in the header. It is also why this leaf carries a
// frame while its neighbours do not.
.p2align 2
.globl LFM_SYM(lfm_log_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_log_add_f32))
LFM_SYM(lfm_log_add_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_log_add_f32)
#endif
    stp     x29, x30, [sp, #-48]!
#ifdef _WIN32
    .seh_save_fplr_x 48
#endif
    mov     x29, sp
#ifdef _WIN32
    .seh_set_fp
#endif
    stp     x19, x20, [sp, #16]
#ifdef _WIN32
    .seh_save_regp x19, 16
#endif
    str     d8, [sp, #32]
#ifdef _WIN32
    .seh_save_freg d8, 32
    .seh_endprologue
#endif
    cbz     x0, 9f
    mov     x19, x0
    mov     x20, x1
    fmov    s8, s0
1:
    cbz     x20, 2f
    ldr     s0, [x19]
    fadd    s0, s0, s8
    bl      LFM_CALL(logf)
    str     s0, [x19], #4
    sub     x20, x20, #1
    b       1b
2:
    mov     w0, #0
    b       7f
9:
    mov     w0, #-22
7:
#ifdef _WIN32
    .seh_startepilogue
#endif
    ldr     d8, [sp, #32]
#ifdef _WIN32
    .seh_save_freg d8, 32
#endif
    ldp     x19, x20, [sp, #16]
#ifdef _WIN32
    .seh_save_regp x19, 16
#endif
    ldp     x29, x30, [sp], #48
#ifdef _WIN32
    .seh_save_fplr_x 48
    .seh_endepilogue
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// int lfm_rowstat_f32(const float *x, uint64_t rows, uint64_t cols,
//                     uint64_t valid, float constant /*s0*/,
//                     float *mean_out, float *std_out)
// Per row over columns [0, valid): mean; ddof-1 std when valid > 1 else the
// reference's NaN-masked 0; std += constant. Sequential adds — the oracle's
// reduction order.
//
// TWO PASSES ON PURPOSE. The mean is completed first, then a SECOND traversal
// accumulates (x - mean)^2. The one-pass shortcut — sum and sum-of-squares
// together, then variance = E[x^2] - E[x]^2 — is algebraically identical and
// numerically catastrophic: for mel bins whose mean is large relative to their
// spread it subtracts two nearly equal large numbers and loses most of the
// significant bits. Cost here is one extra read of a row that is already in
// cache; the payoff is a variance that does not go negative.
//
// Both accumulations are sequential in index order (the ordering contract).
// cols <= 1 skips the centered pass — a single sample has no spread, and the
// guard keeps a divide-by-zero out of the scale.
//
// EPSILON PLACEMENT — looks like an inconsistency with the rest of the tree and
// is not:
//
//   std = sqrt(sumsq / (n - 1)) + constant     <- constant added AFTER the root
//
// Compare lfm_inv_rms_f32 in flashkern_math.S, which computes
// 1/sqrt(sumsq/n + eps) with eps INSIDE, before the root. Two different formulas
// from two different references, each correct for its own. Do not "harmonise"
// them — the difference is the point. (The ddof-1 divisor noted above is the
// same story: sample variance, matching the reference featuriser, not a
// population-variance bug.)
.p2align 2
.globl LFM_SYM(lfm_rowstat_f32)
LFM_PRIVATE(LFM_SYM(lfm_rowstat_f32))
LFM_SYM(lfm_rowstat_f32):
    cbz     x0, 9f
    cbz     x1, 9f
    cbz     x3, 9f
    cmp     x3, x2
    b.hi    9f
    mov     x6, #0              // row
    mov     x7, x0              // row base
1:
    cmp     x6, x1
    b.hs    8f
    fmov    s1, wzr             // sum
    mov     x8, #0
2:
    ldr     s2, [x7, x8, lsl #2]
    fadd    s1, s1, s2
    add     x8, x8, #1
    cmp     x8, x3
    b.lo    2b
    ucvtf   s3, x3
    fdiv    s1, s1, s3          // mean
    str     s1, [x4, x6, lsl #2]
    fmov    s2, wzr             // centered sumsq
    cmp     x3, #1
    b.ls    4f
    mov     x8, #0
3:
    ldr     s5, [x7, x8, lsl #2]
    fsub    s5, s5, s1
    fmul    s5, s5, s5
    fadd    s2, s2, s5
    add     x8, x8, #1
    cmp     x8, x3
    b.lo    3b
    sub     x9, x3, #1
    ucvtf   s6, x9
    fdiv    s2, s2, s6
    fsqrt   s2, s2
4:
    fadd    s2, s2, s0          // += CONSTANT (1e-5)
    str     s2, [x5, x6, lsl #2]
    add     x7, x7, x2, lsl #2
    add     x6, x6, #1
    b       1b
8:
    mov     w0, #0
    ret
9:
    mov     w0, #-22
    ret

// void lfm_norm_apply_f32(float *x, const float *mean, const float *std,
//                         uint64_t rows, uint64_t cols)
// x[r][c] = (x[r][c] - mean[r]) / std[r], over ALL columns (broadcast).
// void lfm_norm_apply_f32(float *x, const float *means, const float *scales,
//                         uint64_t rows, uint64_t cols)
// Per-row normalisation in place: x[r][c] = (x[r][c] - means[r]) / scales[r].
//
// DIVIDES per element rather than multiplying by a precomputed reciprocal.
// Hoisting `1/scale` out of the inner loop is the obvious optimisation and it
// changes the result: reciprocal-then-multiply rounds twice where divide rounds
// once, so the plane drifts from the captured mel fixtures. Same reasoning as
// the exact fdiv/fsqrt rule in flashkern_math.S.
.p2align 2
.globl LFM_SYM(lfm_norm_apply_f32)
LFM_PRIVATE(LFM_SYM(lfm_norm_apply_f32))
LFM_SYM(lfm_norm_apply_f32):
    mov     x5, #0
1:
    cmp     x5, x3
    b.hs    8f
    ldr     s0, [x1, x5, lsl #2]
    ldr     s1, [x2, x5, lsl #2]
    mov     x6, #0
2:
    cmp     x6, x4
    b.hs    3f
    ldr     s2, [x0]
    fsub    s2, s2, s0
    fdiv    s2, s2, s1
    str     s2, [x0], #4
    add     x6, x6, #1
    b       2b
3:
    add     x5, x5, #1
    b       1b
8:
    ret

// uint64_t lfm_resampler_stream_linear_f32(const StreamKernel *kernel)
// Retained linear rate conversion over one borrowed codec frame. The kernel
// descriptor is {input@0, destination@8, sample_count@16, orig@24,
// phases@32, phase@40, previous@48, ready@52}. C++ owns validation and the
// bounded phase/history update; this leaf owns every per-sample interpolation.
// Separate fsub/fmul/fadd retains the former unfused f32 operation order.
.p2align 2
.globl LFM_SYM(lfm_resampler_stream_linear_f32)
LFM_PRIVATE(LFM_SYM(lfm_resampler_stream_linear_f32))
LFM_SYM(lfm_resampler_stream_linear_f32):
    ldr     x1, [x0]             // input
    ldr     x2, [x0, #8]         // destination
    ldr     x3, [x0, #16]        // remaining input samples
    ldr     x4, [x0, #24]        // reduced input rate
    ldr     x5, [x0, #32]        // reduced output phases
    ldr     x6, [x0, #40]        // retained phase
    ldr     s4, [x0, #48]        // retained previous sample
    ldr     w7, [x0, #52]        // previous sample is ready
    ucvtf   s5, x5
    mov     x8, #0               // written
1:
    cbz     x3, 8f
    ldr     s0, [x1], #4         // sample
    cbnz    w7, 2f
    fmov    s1, s0               // first call starts from this sample
    b       3f
2:
    fmov    s1, s4               // previous
3:
    mov     x12, x6              // phase before this input sample
    add     x9, x6, x5
    udiv    x10, x9, x4          // output count for this input
    msub    x6, x10, x4, x9      // next retained phase
    fsub    s2, s0, s1           // sample - previous
    mov     x11, #1
4:
    cmp     x11, x10
    b.hi    5f
    mul     x13, x11, x4
    sub     x13, x13, x12        // output*orig - old phase
    ucvtf   s3, x13
    fdiv    s3, s3, s5           // interpolation fraction
    fmul    s3, s2, s3
    fadd    s3, s1, s3
    str     s3, [x2], #4
    add     x8, x8, #1
    add     x11, x11, #1
    b       4b
5:
    fmov    s4, s0
    mov     w7, #1
    sub     x3, x3, #1
    b       1b
8:
    mov     x0, x8
    ret

// void lfm_resample_conv_spans_f32(const LfmF32SpanChain *input,
//                                  const double *kernels, uint64_t phases,
//                                  uint64_t stride, uint64_t width,
//                                  uint64_t first_value, float *out,
//                                  uint64_t value_count)
// Generic polyphase sample-rate conversion from the source rate to the
// destination rate selected by the immutable resampler plan.
//
// Each output sample picks its kernel by PHASE and its input window by BLOCK:
//     block = start / up ;  phase = start % up      (udiv + msub)
//     kernel row = kernels + phase * klen           (klen = stride + 2*width)
// so the filter bank is indexed rather than recomputed, and the `up` phases are
// a compile-time-fixed set rather than something interpolated at runtime.
//
// Window edges are handled by explicit index clamping, not by materialising a
// zero-padded copy of the signal: j_begin skips the prefix when the window
// starts before sample 0, and j_end is clamped to min(klen, width + len -
// start) so the suffix past the end is never read. A window entirely inside the
// suffix produces no taps at all. That is why there is no padded scratch buffer
// anywhere in the resample path — the padding is arithmetic, not memory.
//
// F64 ACCUMULATION — the one place in the kernel set that carries more than f32.
// The kernel table is `const double *`, each f32 input is widened (fcvt d, s),
// and the tap sum accumulates in f64 in ascending-j order before a single fcvt
// back to f32 at the end. A long filter window summed in f32 would accumulate
// visible error across its taps; everywhere else in these kernels f32/bf16 is
// the contract, so this deliberate exception is worth knowing before someone
// "simplifies" it to match its neighbours. Still no FMA — fmul and fadd stay
// separate in f64 too.
//
// The input-chain descriptor is copied into the retained pass. Each tap maps
// its logical source index to span 0 or span 1, so a slab wrap is invisible and
// no PCM concatenation plane exists.
.p2align 2
.globl LFM_SYM(lfm_resample_conv_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_resample_conv_spans_f32))
LFM_SYM(lfm_resample_conv_spans_f32):
    add     x8, x3, x4, lsl #1 // klen = stride + 2*width
    udiv    x9, x5, x2          // initial block
    msub    x10, x9, x2, x5    // initial phase
1:
    cbz     x7, 8f
    mul     x11, x9, x3         // logical padded-window start
    mul     x12, x10, x8
    add     x12, x1, x12, lsl #3 // kernel row

    cmp     x11, x4
    b.hs    2f
    sub     x13, x4, x11        // j_begin = width - start
    mov     x14, #0             // input index 0
    b       3f
2:
    mov     x13, #0             // j_begin = 0
    sub     x14, x11, x4        // input index = start - width
3:
    ldr     x15, [x0, #8]       // total input length
    add     x15, x4, x15        // one-past logical input coordinate
    cmp     x11, x15
    b.hs    6f                   // the entire window is suffix padding
    sub     x15, x15, x11       // first suffix j
    cmp     x15, x8
    csel    x15, x15, x8, lo    // j_end = min(klen, width+len-start)
    fmov    d0, xzr
4:
    cmp     x13, x15
    b.hs    7f
    ldr     x16, [x0, #24]      // span 0 length
    cmp     x14, x16
    b.hs    5f
    ldr     x17, [x0, #16]
    ldr     s1, [x17, x14, lsl #2]
    b       9f
5:
    sub     x16, x14, x16
    ldr     x17, [x0, #32]
    ldr     s1, [x17, x16, lsl #2]
9:
    fcvt    d1, s1
    ldr     d2, [x12, x13, lsl #3]
    fmul    d1, d1, d2
    fadd    d0, d0, d1
    add     x13, x13, #1
    add     x14, x14, #1
    b       4b
6:
    fmov    d0, xzr
7:
    fcvt    s0, d0
    str     s0, [x6], #4
    sub     x7, x7, #1
    add     x10, x10, #1
    cmp     x10, x2
    b.lo    1b
    mov     x10, #0
    add     x9, x9, #1
    b       1b
8:
    ret

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
