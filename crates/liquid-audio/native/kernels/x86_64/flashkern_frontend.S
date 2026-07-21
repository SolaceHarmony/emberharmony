#include "lfm_asm_visibility.h"

// Audio-frontend leaves, x86_64 twin of aarch64/flashkern_frontend.S.
// Scalar SSE2 on purpose (utterance-rate, parity-oracle op order, no FMA).
// Dual ABI like the sibling leaves: SysV (rdi rsi rdx rcx r8 r9 / xmm0-) and
// Win64 GNU (rcx rdx r8 r9 + stack, floats positional).
//
// Read the AArch64 twin for the full per-leaf rationale; the contracts are
// identical and were verified leaf-by-leaf. Two consequences restated here
// because they bite on this target specifically:
//
// NO FMA — and this matters MORE on x86 than on AArch64, because FMA3 makes the
// "optimisation" so easy to reach for. `vfmadd231ss` keeps the product at full
// internal width and rounds ONCE; the mulss + addss pair here rounds TWICE. The
// results differ in the last bits, so contracting any of these leaves walks the
// mel plane off native/tests/fixtures/mel. Every multiply-then-add in this file
// is deliberately unfused, and -ffp-contract=off guards the C++ side of the same
// contract. An FMA rewrite here is a regression, not a speedup.
//
// PLATFORM TIE — `lfm_log_add_f32` calls the platform libm `logf`, so the mel
// plane and everything downstream of it (including any end-to-end byte oracle)
// is reproducible on a given platform but not guaranteed across them. This
// file, flashkern_rope.S (powf/cosf/sinf) and flashkern_conformer.S
// (expf/erff/sinf/cosf — the largest user) are the entire libm surface of the
// kernel set; math/prng/sampler/sesame are libm-free and bit-identical
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

// int lfm_preemph_spans_f32(const LfmF32SpanChain *x, float *y, float coef)
// y[0] = x[0]; y[i] = x[i] - coef*x[i-1], where x is the logical concatenation
// of its one or two physical spans. The previous ORIGINAL sample stays in xmm3
// across the split, so the boundary is numerically invisible. Each original is
// loaded before its result is stored, preserving the old in-place ordering
// within a physically contiguous source. No old direct-span numerical leaf
// remains beside this one.
//
// The public C++ binder constructs a sealed chain, but this returning leaf also
// defensively checks its local structural assumptions before touching samples:
// count is one or two, the total is non-zero, non-empty spans have data, and
// their overflow-checked length sum equals chain.length. The binder owns full
// address-range validation. Returns -EINVAL on a failed local check.
.p2align 4
.globl LFM_SYM(lfm_preemph_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_preemph_spans_f32))
LFM_SYM(lfm_preemph_spans_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movaps    %xmm2, %xmm0
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    testq     %r10, %r10
    jz        9f
    testq     %r11, %r11
    jz        9f
    cmpl      $0, 4(%r10)
    jne       9f
    movl      0(%r10), %edx      // admitted span count
    testl     %edx, %edx
    jz        9f
    cmpl      $2, %edx
    ja        9f
    movq      8(%r10), %r8       // checked logical length
    testq     %r8, %r8
    jz        9f

    movq      24(%r10), %rcx     // span 0 length
    testq     %rcx, %rcx
    jz        9f
    cmpq      $0, 16(%r10)
    je        9f
    movq      %rcx, %rax
    cmpl      $2, %edx
    jb        4f
    movq      40(%r10), %r9      // span 1 length
    testq     %r9, %r9
    jz        9f
    cmpq      $0, 32(%r10)
    je        9f
    addq      %r9, %rax
    jc        9f
4:
    cmpq      %r8, %rax
    jne       9f

    xorq      %rax, %rax         // physical span index
    xorl      %ecx, %ecx         // no previous logical sample yet
1:
    cmpq      %rdx, %rax
    jae       8f
    movq      %rax, %r8
    shlq      $4, %r8
    addq      %r10, %r8
    movq      24(%r8), %r9
    movq      16(%r8), %r8
    testq     %r9, %r9
    jz        7f
    testl     %ecx, %ecx
    jnz       2f
    movss     (%r8), %xmm3       // first original sample
    movss     %xmm3, (%r11)
    addq      $4, %r8
    addq      $4, %r11
    decq      %r9
    movl      $1, %ecx
2:
    testq     %r9, %r9
    jz        7f
    movss     (%r8), %xmm1       // load current before an aliasing store
    movaps    %xmm1, %xmm4
    movaps    %xmm3, %xmm2
    mulss     %xmm0, %xmm2       // intentionally unfused
    subss     %xmm2, %xmm1
    movss     %xmm1, (%r11)
    movaps    %xmm4, %xmm3       // carry previous ORIGINAL across spans
    addq      $4, %r8
    addq      $4, %r11
    decq      %r9
    jmp       2b
7:
    incq      %rax
    jmp       1b
8:
    xorl      %eax, %eax
    ret
9:
    movl      $-22, %eax
    ret

// void lfm_frame_gather_spans_f32(const LfmF32SpanChain *x,
//     uint64_t logical_offset, uint64_t logical_limit, uint64_t center,
//     uint64_t hop, uint64_t nfft, uint64_t cols, float *out)
//
// Writes the transposed frame plane out[n][t] directly. For frame t/sample n,
// logical = t*hop+n-center. Coordinates outside [logical_offset,logical_limit)
// are logical zero padding; admitted coordinates resolve against the fixed
// chain without materialising or joining its split spans. Geometry, output
// capacity, and logical_limit-offset <= chain.length are validated by C++.
.p2align 4
.globl LFM_SYM(lfm_frame_gather_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_frame_gather_spans_f32))
LFM_SYM(lfm_frame_gather_spans_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_frame_gather_spans_f32)
#endif
    pushq     %rbx
#ifdef _WIN32
    .seh_pushreg %rbx
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
#endif
    pushq     %r14
#ifdef _WIN32
    .seh_pushreg %r14
#endif
    pushq     %r15
#ifdef _WIN32
    .seh_pushreg %r15
#endif
    pushq     %rdi
#ifdef _WIN32
    .seh_pushreg %rdi
#endif
    pushq     %rsi
#ifdef _WIN32
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // chain
    movq      %rdx, %rsi          // logical_offset
    movq      %r8,  %rdx          // logical_limit
    movq      %r9,  %rcx          // center
    movq      96(%rsp), %r8       // hop (entry 40 + pushes 56)
    movq      104(%rsp), %r9      // nfft
    movq      112(%rsp), %r10     // cols
    movq      120(%rsp), %r11     // out
#else
    movq      64(%rsp), %r10      // cols (entry 8 + pushes 56)
    movq      72(%rsp), %r11      // out
#endif
    movq      24(%rdi), %r14      // span 0 length
    movq      16(%rdi), %r15      // span 0 data
    movq      32(%rdi), %rbx      // span 1 data (used only past span 0)
    xorq      %r12, %r12          // sample n
6:
    cmpq      %r9, %r12
    jae       8f
    xorq      %r13, %r13          // frame t
7:
    cmpq      %r10, %r13
    jae       5f
    movq      %r13, %rax
    imulq     %r8, %rax
    addq      %r12, %rax          // padded coordinate t*hop+n
    cmpq      %rcx, %rax
    jb        4f
    subq      %rcx, %rax          // logical source coordinate
    cmpq      %rsi, %rax
    jb        4f
    cmpq      %rdx, %rax
    jae       4f
    subq      %rsi, %rax          // chain-relative sample index
    cmpq      %r14, %rax
    jae       3f
    movss     (%r15,%rax,4), %xmm0
    jmp       2f
3:
    subq      %r14, %rax
    movss     (%rbx,%rax,4), %xmm0
    jmp       2f
4:
    xorps     %xmm0, %xmm0
2:
    movss     %xmm0, (%r11)
    addq      $4, %r11
    incq      %r13
    jmp       7b
5:
    incq      %r12
    jmp       6b
8:
    popq      %rsi
    popq      %rdi
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbx
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_power_spec_f32(const float *re, const float *im, float *out, uint64_t n)
// out[i] = re[i]*re[i] + im[i]*im[i] — the POWER spectrum, |X|^2. This is what
// `mag_power == 2` names in the frontend config; magnitude (1.0) is not
// implemented and is refused at create rather than silently returning power.
// Two separate mulss and an addss, never fused (see header).
.p2align 4
.globl LFM_SYM(lfm_power_spec_f32)
LFM_PRIVATE(LFM_SYM(lfm_power_spec_f32))
LFM_SYM(lfm_power_spec_f32):
#ifdef _WIN32
    movq      %rcx, %r10
    movq      %rdx, %r11
    movq      %r8,  %rdx
    movq      %r9,  %rcx
#else
    movq      %rdi, %r10
    movq      %rsi, %r11
#endif
    testq     %rcx, %rcx
    jz        8f
1:
    movss     (%r10), %xmm0
    movss     (%r11), %xmm1
    mulss     %xmm0, %xmm0
    mulss     %xmm1, %xmm1
    addss     %xmm1, %xmm0
    movss     %xmm0, (%rdx)
    addq      $4, %r10
    addq      $4, %r11
    addq      $4, %rdx
    decq      %rcx
    jnz       1b
8:
    ret

// int lfm_log_add_f32(float *x, uint64_t n, float guard)
// x[i] = logf(x[i] + guard), in place — the log-mel step. `guard` is added
// BEFORE the log; that ordering is the contract and is what stops a silent bin
// (power 0) becoming -inf and poisoning the row statistics. This is the file's
// only libm call and therefore its only non-portable step (see header).
.p2align 4
.globl LFM_SYM(lfm_log_add_f32)
LFM_PRIVATE(LFM_SYM(lfm_log_add_f32))
LFM_SYM(lfm_log_add_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_log_add_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    pushq     %rbx
    .seh_pushreg %rbx
    pushq     %r12
    .seh_pushreg %r12
    subq      $56, %rsp          // 32 shadow + guard spill + alignment
    .seh_stackalloc 56
    .seh_endprologue
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movaps    %xmm2, %xmm0
    testq     %rdi, %rdi
    jz        9f
#else
    testq     %rdi, %rdi
    jz        9f
    pushq     %rbx
    pushq     %r12
    subq      $56, %rsp          // call alignment + guard spill
#endif
    movq      %rdi, %rbx
    movq      %rsi, %r12
    movss     %xmm0, 40(%rsp)
1:
    testq     %r12, %r12
    jz        2f
    movss     (%rbx), %xmm0
    addss     40(%rsp), %xmm0
    call      LFM_CALL(logf)
    movss     %xmm0, (%rbx)
    addq      $4, %rbx
    decq      %r12
    jmp       1b
2:
    xorl      %eax, %eax
    jmp       7f
9:
    movl      $-22, %eax
#ifndef _WIN32
    ret
#endif
7:
    addq      $56, %rsp
    popq      %r12
    popq      %rbx
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// int lfm_rowstat_f32(const float *x, uint64_t rows, uint64_t cols,
//                     uint64_t valid, float constant, float *mean, float *std)
// Per-row mean and standard deviation for the per-feature normalisation.
//
// TWO PASSES ON PURPOSE: the mean completes first, then a SECOND traversal
// accumulates (x - mean)^2. The one-pass shortcut E[x^2] - E[x]^2 is
// algebraically identical and numerically catastrophic — for bins whose mean is
// large relative to their spread it subtracts two nearly equal large numbers.
// The extra read hits cache; the payoff is a variance that cannot go negative.
//
// TWO CONVENTIONS THAT LOOK LIKE INCONSISTENCIES AND ARE NOT:
//     std = sqrt(sumsq / (n - 1)) + constant
// 1. n-1, not n — sample variance (Bessel), matching the reference featuriser.
// 2. The epsilon is added to the STANDARD DEVIATION after the sqrt, NOT inside
//    it. Compare lfm_inv_rms_f32 in flashkern_math.S: 1/sqrt(sumsq/n + eps),
//    eps INSIDE. Both are correct for their own reference. Do not harmonise.
// valid <= 1 skips the centered pass; both accumulations are sequential.
.p2align 4
.globl LFM_SYM(lfm_rowstat_f32)
LFM_PRIVATE(LFM_SYM(lfm_rowstat_f32))
LFM_SYM(lfm_rowstat_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_rowstat_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // x
    movq      %rdx, %rsi          // rows
    movq      %r8,  %rdx          // cols
    movq      %r9,  %rcx          // valid
    movss     56(%rsp), %xmm0     // constant
    movq      64(%rsp), %r8       // mean
    movq      72(%rsp), %r9       // std
#endif
    testq     %rdi, %rdi
    jz        9f
    testq     %rsi, %rsi
    jz        9f
    testq     %rcx, %rcx
    jz        9f
    cmpq      %rdx, %rcx
    ja        9f
    xorq      %r10, %r10          // row
    movq      %rdi, %r11          // row base
1:
    cmpq      %rsi, %r10
    jae       8f
    xorps     %xmm1, %xmm1        // sum
    xorq      %rax, %rax
2:
    addss     (%r11,%rax,4), %xmm1
    incq      %rax
    cmpq      %rcx, %rax
    jb        2b
    pxor      %xmm3, %xmm3
    cvtsi2ssq %rcx, %xmm3
    divss     %xmm3, %xmm1        // mean
    movss     %xmm1, (%r8,%r10,4)
    xorps     %xmm2, %xmm2        // centered sumsq
    cmpq      $1, %rcx
    jbe       4f
    xorq      %rax, %rax
3:
    movss     (%r11,%rax,4), %xmm5
    subss     %xmm1, %xmm5
    mulss     %xmm5, %xmm5
    addss     %xmm5, %xmm2
    incq      %rax
    cmpq      %rcx, %rax
    jb        3b
    leaq      -1(%rcx), %rax
    pxor      %xmm4, %xmm4
    cvtsi2ssq %rax, %xmm4
    divss     %xmm4, %xmm2
    sqrtss    %xmm2, %xmm2
4:
    addss     %xmm0, %xmm2        // += CONSTANT
    movss     %xmm2, (%r9,%r10,4)
    leaq      (%r11,%rdx,4), %r11
    incq      %r10
    jmp       1b
8:
    xorl      %eax, %eax
    jmp       7f
9:
    movl      $-22, %eax
7:
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_norm_apply_f32(float *x, const float *mean, const float *std,
//                         uint64_t rows, uint64_t cols)
// In place: x[r][c] = (x[r][c] - mean[r]) / std[r].
//
// DIVIDES per element rather than hoisting a reciprocal out of the inner loop.
// Reciprocal-then-multiply rounds twice where divide rounds once, so the
// obvious optimisation drifts the plane off the captured mel fixtures. Same
// reasoning as the exact divss/sqrtss rule in flashkern_math.S.
.p2align 4
.globl LFM_SYM(lfm_norm_apply_f32)
LFM_PRIVATE(LFM_SYM(lfm_norm_apply_f32))
LFM_SYM(lfm_norm_apply_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_norm_apply_f32)
    pushq     %rdi
    .seh_pushreg %rdi
    pushq     %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi
    movq      %rdx, %rsi
    movq      %r8,  %rdx
    movq      %r9,  %rcx
    movq      56(%rsp), %r8       // cols
#else
    // SysV: x=rdi mean=rsi std=rdx rows=rcx cols=r8
#endif
    xorq      %r10, %r10          // row
1:
    cmpq      %rcx, %r10
    jae       8f
    movss     (%rsi,%r10,4), %xmm0
    movss     (%rdx,%r10,4), %xmm1
    xorq      %r11, %r11
2:
    cmpq      %r8, %r11
    jae       3f
    movss     (%rdi), %xmm2
    subss     %xmm0, %xmm2
    divss     %xmm1, %xmm2
    movss     %xmm2, (%rdi)
    addq      $4, %rdi
    incq      %r11
    jmp       2b
3:
    incq      %r10
    jmp       1b
8:
#ifdef _WIN32
    popq      %rsi
    popq      %rdi
#endif
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// uint64_t lfm_resampler_stream_linear_f32(const StreamKernel *kernel)
// Retained linear rate conversion over one borrowed codec frame. The kernel
// descriptor is {input@0, destination@8, sample_count@16, orig@24,
// phases@32, phase@40, previous@48, ready@52}. C++ validates those views and
// updates the conversation-owned phase/history once this leaf returns; all
// per-sample interpolation stays here. Separate subss/mulss/addss preserve the
// unfused f32 operation order of the former implementation.
.p2align 4
.globl LFM_SYM(lfm_resampler_stream_linear_f32)
LFM_PRIVATE(LFM_SYM(lfm_resampler_stream_linear_f32))
LFM_SYM(lfm_resampler_stream_linear_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_resampler_stream_linear_f32)
#endif
    pushq     %rbx
#ifdef _WIN32
    .seh_pushreg %rbx
#endif
    pushq     %rbp
#ifdef _WIN32
    .seh_pushreg %rbp
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
#endif
    pushq     %r14
#ifdef _WIN32
    .seh_pushreg %r14
#endif
    pushq     %r15
#ifdef _WIN32
    .seh_pushreg %r15
#endif
    pushq     %rdi
#ifdef _WIN32
    .seh_pushreg %rdi
    .seh_endprologue
    movq      %rcx, %r15
#else
    movq      %rdi, %r15
#endif
    movq      0(%r15), %r12      // input
    movq      8(%r15), %r13      // destination
    movq      16(%r15), %r14     // remaining input samples
    movq      24(%r15), %rbx     // reduced input rate
    movq      32(%r15), %rbp     // reduced output phases
    movq      40(%r15), %r10     // retained phase
    movss     48(%r15), %xmm4    // retained previous sample
    movl      52(%r15), %r11d    // previous sample is ready
    cvtsi2ssq %rbp, %xmm5
    xorq      %r9, %r9           // written
1:
    testq     %r14, %r14
    jz        8f
    movss     (%r12), %xmm0      // sample
    testl     %r11d, %r11d
    jnz       2f
    movaps    %xmm0, %xmm1       // first call starts from this sample
    jmp       3f
2:
    movaps    %xmm4, %xmm1       // previous
3:
    movq      %r10, %rdi         // phase before this input sample
    leaq      (%r10,%rbp), %rax  // phase + phases
    xorq      %rdx, %rdx
    divq      %rbx
    movq      %rax, %r8          // output count for this input
    movq      %rdx, %r10         // next retained phase
    movaps    %xmm0, %xmm2
    subss     %xmm1, %xmm2       // sample - previous
    movq      $1, %rcx
4:
    cmpq      %r8, %rcx
    ja        5f
    movq      %rcx, %rax
    imulq     %rbx, %rax
    subq      %rdi, %rax         // output*orig - old phase
    cvtsi2ssq %rax, %xmm3
    divss     %xmm5, %xmm3       // interpolation fraction
    mulss     %xmm2, %xmm3
    addss     %xmm1, %xmm3
    movss     %xmm3, (%r13)
    addq      $4, %r13
    incq      %r9
    incq      %rcx
    jmp       4b
5:
    movaps    %xmm0, %xmm4
    movl      $1, %r11d
    addq      $4, %r12
    decq      %r14
    jmp       1b
8:
    movq      %r9, %rax
    popq      %rdi
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbp
    popq      %rbx
    ret
#ifdef _WIN32
    .seh_endproc
#endif

// void lfm_resample_conv_spans_f32(const LfmF32SpanChain *input,
//     const double *kernels, uint64_t phases, uint64_t stride, uint64_t width,
//     uint64_t first_value, float *out, uint64_t value_count)
// Torchaudio's polyphase convolution over the logical concatenation of the
// fixed PCM spans. Prefix/suffix padding remains arithmetic, and a filter
// window crossing the physical split reads straight through it. Each f32 tap
// is widened and accumulated in ascending-j f64 order using separate mulsd and
// addsd, preserving the old numerical contract exactly. There is no direct-
// span duplicate leaf.
.p2align 4
.globl LFM_SYM(lfm_resample_conv_spans_f32)
LFM_PRIVATE(LFM_SYM(lfm_resample_conv_spans_f32))
LFM_SYM(lfm_resample_conv_spans_f32):
#ifdef _WIN32
    .seh_proc LFM_SYM(lfm_resample_conv_spans_f32)
#endif
    pushq     %rbx
#ifdef _WIN32
    .seh_pushreg %rbx
#endif
    pushq     %rbp
#ifdef _WIN32
    .seh_pushreg %rbp
#endif
    pushq     %r12
#ifdef _WIN32
    .seh_pushreg %r12
#endif
    pushq     %r13
#ifdef _WIN32
    .seh_pushreg %r13
#endif
    pushq     %r14
#ifdef _WIN32
    .seh_pushreg %r14
#endif
    pushq     %r15
#ifdef _WIN32
    .seh_pushreg %r15
#endif
    pushq     %rdi
#ifdef _WIN32
    .seh_pushreg %rdi
#endif
    pushq     %rsi
#ifdef _WIN32
    .seh_pushreg %rsi
    .seh_endprologue
    movq      %rcx, %rdi          // chain
    movq      %rdx, %rbx          // kernels
    movq      %r8,  %rcx          // phases
    movq      %r9,  %r8           // stride
    movq      104(%rsp), %r9      // width (entry 40 + pushes 64)
    movq      112(%rsp), %r10     // first_value
    movq      120(%rsp), %r11     // out
    movq      128(%rsp), %r12     // value_count
#else
    movq      %rsi, %rbx          // kernels
    movq      %rcx, %rbp          // preserve stride
    movq      %r8,  %r13          // preserve width
    movq      %r9,  %r10          // first_value
    movq      %rdx, %rcx          // phases
    movq      %rbp, %r8           // stride
    movq      %r13, %r9           // width
    movq      72(%rsp), %r11      // out (entry 8 + pushes 64)
    movq      80(%rsp), %r12      // value_count
#endif
    leaq      (%r8,%r9,2), %r13   // klen = stride + 2*width
    movq      %r10, %rax
    xorq      %rdx, %rdx
    divq      %rcx
    movq      %rax, %r10          // initial block
    movq      %rdx, %r14          // initial phase
1:
    testq     %r12, %r12
    jz        8f
    movq      %r14, %rax
    imulq     %r13, %rax
    leaq      (%rbx,%rax,8), %r15 // kernel row
    movq      %r10, %rax
    imulq     %r8, %rax            // logical padded-window start

    movq      8(%rdi), %rdx       // logical input length
    addq      %r9, %rdx           // one-past padded logical coordinate
    cmpq      %rdx, %rax
    jae       6f                   // the entire window is suffix padding
    subq      %rax, %rdx           // first suffix j
    cmpq      %r13, %rdx
    jbe       2f
    movq      %r13, %rdx           // j_end = min(klen, width+len-start)
2:
    cmpq      %r9, %rax
    jae       3f
    movq      %r9, %rbp
    subq      %rax, %rbp           // j_begin = width - start
    xorq      %rax, %rax          // input index 0
    jmp       4f
3:
    xorq      %rbp, %rbp          // j_begin = 0
    subq      %r9, %rax           // input index = start - width
4:
    xorpd     %xmm0, %xmm0        // acc
5:
    cmpq      %rdx, %rbp
    jae       7f
    cmpq      24(%rdi), %rax
    jae       9f
    movq      16(%rdi), %rsi
    cvtss2sd  (%rsi,%rax,4), %xmm1
    jmp       10f
9:
    subq      24(%rdi), %rax
    movq      32(%rdi), %rsi
    cvtss2sd  (%rsi,%rax,4), %xmm1
    addq      24(%rdi), %rax
10:
    movsd     (%r15,%rbp,8), %xmm2
    mulsd     %xmm2, %xmm1
    addsd     %xmm1, %xmm0
    incq      %rax
    incq      %rbp
    jmp       5b
6:
    xorpd     %xmm0, %xmm0
7:
    cvtsd2ss  %xmm0, %xmm0
    movss     %xmm0, (%r11)
    addq      $4, %r11
    decq      %r12
    incq      %r14
    cmpq      %rcx, %r14
    jb        1b
    xorq      %r14, %r14
    incq      %r10
    jmp       1b
8:
#ifdef _WIN32
    // rsi/rdi are restored by the common epilogue below.
#endif
    popq      %rsi
    popq      %rdi
    popq      %r15
    popq      %r14
    popq      %r13
    popq      %r12
    popq      %rbp
    popq      %rbx
    ret
#ifdef _WIN32
    .seh_endproc
#endif

#if !defined(__APPLE__) && !defined(_WIN32)
.section .note.GNU-stack,"",%progbits
#endif
