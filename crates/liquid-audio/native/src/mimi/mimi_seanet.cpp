// mimi_seanet.cpp — Unit #3: the SEANet DECODER stack.
//
// Faithful C++ port of moshi 0.6.4 `SeaNetDecoder::step` + `SeaNetResnetBlock::step`
// (src/seanet.rs), config `mimi.rs Config::v0_1(8)`. This file owns the STRUCTURE
// of the decoder: the layer list, ELU placement, the residual skip add, the
// streaming state layout, and the frame-count plumbing. It does NOT implement any
// convolution itself — every conv/convtranspose is delegated to the unit-2
// primitives declared in mimi_kernel.h (mimi_conv_* / mimi_convtr_*), which own
// the streaming left-context / partial-frame state.
//
// Ported:   SeaNetResnetBlock::step (true_skip=true => identity skip),
//           SeaNetDecoder::step (init conv -> 4x[ELU, convtr upsample, resnet],
//           final ELU + last conv), ELU(alpha=1.0), reset_state.
// Skipped:  SeaNetEncoder, lstm (lstm=0), final_activation (None).
//
// Engine discipline: zero allocation in steady state (all scratch carved from the
// MimiArena once at init, sized worst-case), POD state, f32 accumulate, scalar
// reference siblings under MIMI_SCALAR_REF, no exceptions across the ABI.

#include "mimi_kernel.h"

#include <cstdio>   // snprintf
#include <cstring>  // strlen (unused guard-free)

#if defined(__aarch64__) && !defined(MIMI_SCALAR_REF)
#include <arm_neon.h>
#endif

/* ------------------------------------------------------------------------- *
 * Derived geometry (from SeaNetDecoder::new, cfg v0_1).
 *
 *   n_filters = 64, dimension = 512, channels = 1, ratios = [8,6,5,4]
 *   (decoder iterates ratios in FORWARD order — the encoder is the one that
 *    reverses them), n_residual_layers = 1, compress = 2, dilation_base = 2,
 *   kernel_size = 7, residual_kernel_size = 3, last_kernel_size = 3.
 *
 *   mult starts at (1 << ratios.len()) = 16 and halves after each layer.
 *
 * decoder.model.N index map (nn.Sequential slots; odd/activation slots hold no
 * weights, hence the `layer_idx + 1` offsets in the Rust constructor):
 *   0  init conv        StreamableConv1d(512 -> 1024, k=7,  s=1)
 *   1  ELU              (no weights)
 *   2  convtr  ratio 8  StreamableConvTranspose1d(1024 -> 512, k=16, s=8)
 *   3  resnet  ratio 8  block.1: conv(512 ->256,k=3)  block.3: conv(256 ->512,k=1)
 *   4  ELU
 *   5  convtr  ratio 6  StreamableConvTranspose1d(512 -> 256, k=12, s=6)
 *   6  resnet  ratio 6  block.1: conv(256 ->128,k=3)  block.3: conv(128 ->256,k=1)
 *   7  ELU
 *   8  convtr  ratio 5  StreamableConvTranspose1d(256 -> 128, k=10, s=5)
 *   9  resnet  ratio 5  block.1: conv(128 -> 64,k=3)  block.3: conv( 64 ->128,k=1)
 *   10 ELU
 *   11 convtr  ratio 4  StreamableConvTranspose1d(128 ->  64, k=8,  s=4)
 *   12 resnet  ratio 4  block.1: conv( 64 -> 32,k=3)  block.3: conv( 32 -> 64,k=1)
 *   13 ELU
 *   14 last conv        StreamableConv1d(64 -> 1, k=3, s=1)
 *
 * Resnet hidden = resnet_dim / compress; resnet_dim = upsample out channels.
 * true_skip=true => shortcut is None => skip is the raw input (identity), and the
 * StreamingBinOp(Add) buffering degenerates to a plain elementwise add: both
 * branches are frame-count preserving (stride-1 causal convs and the identity
 * skip each emit n_in frames per step), so common_len == n every step and the
 * op's prev_lhs/prev_rhs buffers stay empty. See NOTES (a),(d).
 * ------------------------------------------------------------------------- */

static constexpr int NUM_LAYERS = 4;

// Per-layer upsample (convtranspose) geometry.
static constexpr int LAYER_UP_IN[NUM_LAYERS]  = {1024, 512, 256, 128}; // in channels
static constexpr int LAYER_UP_OUT[NUM_LAYERS] = { 512, 256, 128,  64}; // out channels = resnet dim
static constexpr int LAYER_UP_K[NUM_LAYERS]   = {  16,  12,  10,   8}; // ratio*2
static constexpr int LAYER_RATIO[NUM_LAYERS]  = {   8,   6,   5,   4}; // stride
static constexpr int LAYER_UP_IDX[NUM_LAYERS] = {   2,   5,   8,  11}; // decoder.model.N
static constexpr int LAYER_RES_IDX[NUM_LAYERS]= {   3,   6,   9,  12}; // decoder.model.N (resnet node)

// Init/final conv geometry.
static const int INIT_IN = MIMI_DIM;          // 512
static const int INIT_OUT = 1024;             // n_filters * (1<<len(ratios)) = 64*16
static const int FINAL_IN = MIMI_N_FILTERS;   // 64
static const int FINAL_OUT = 1;               // channels

// Worst-case buffer sizing.  n_in latent frames -> n_in*960 samples.  pcm_out
// capacity is MIMI_FRAME_OUT*2 = 3840 = 960*4, so n_in is bounded by 4.
enum { MIMI_SEANET_MAX_N_IN = 4 };
// Largest (channels * frames) of any inter-stage buffer, per input frame:
//   layer-3 upsample out / resnet:  64 ch * (8*6*5*4=960) frames = 61440.
static const int MIMI_SEANET_MAX_ELEMS_PER_NIN = 61440;
static const int MIMI_SEANET_BUF_ELEMS =
    MIMI_SEANET_MAX_ELEMS_PER_NIN * MIMI_SEANET_MAX_N_IN;
// During L0 residual conv0, b2's output prefix is live while its matrix gather
// is consumed. Reserve the worst supported prefix (256 hidden channels *
// 4 latent frames * ratio 8); the suffix is still ample for the gather.
static const int MIMI_SEANET_L0_RES_LIVE_ELEMS =
    (LAYER_UP_OUT[0] / 2) * MIMI_SEANET_MAX_N_IN * LAYER_RATIO[0];
static_assert(MIMI_SEANET_L0_RES_LIVE_ELEMS == 8192,
              "SeaNet L0 b2 live-prefix geometry drifted");
static_assert((size_t)MIMI_SEANET_L0_RES_LIVE_ELEMS +
                      (size_t)LAYER_UP_OUT[0] * MIMI_RES_KERNEL *
                          MIMI_SEANET_MAX_N_IN * LAYER_RATIO[0] <=
                  MIMI_SEANET_BUF_ELEMS,
              "SeaNet L0 residual gather no longer fits b2 suffix");
static_assert((size_t)LAYER_UP_K[3] * LAYER_UP_OUT[3] *
                      2 * LAYER_RATIO[0] * LAYER_RATIO[1] * LAYER_RATIO[2] ==
                  MIMI_SEANET_BUF_ELEMS,
              "two-latent deepest convtr must exactly fill shared b2");
static_assert((size_t)LAYER_UP_K[2] * LAYER_UP_OUT[2] *
                      MIMI_SEANET_MAX_N_IN * LAYER_RATIO[0] * LAYER_RATIO[1] ==
                  MIMI_SEANET_BUF_ELEMS,
              "four-latent L2 convtr must exactly fill shared b2");

/* ---- POD streaming state (carved from the arena at init) ----------------- */
struct MimiSeanetState {
    MimiConvState   *init_conv;
    MimiConvTrState *upsample[NUM_LAYERS];  // decoder.model.{2,5,8,11}
    MimiConvState   *res_conv0[NUM_LAYERS]; // resnet block.1 (k=3, dim -> hidden)
    MimiConvState   *res_conv1[NUM_LAYERS]; // resnet block.3 (k=1, hidden -> dim)
    MimiConvState   *final_conv;

    // Three ping-pong work buffers, each sized to the global worst case. b2 is
    // also the single sequential matrix workspace: full for init/convtr and,
    // while L0 hidden output is live, its suffix after the reserved prefix.
    float *b0;  // running chain tensor / resnet skip input
    float *b1;  // activation output / ys / convtr input
    float *b2;  // resnet hidden intermediate
};

/* ---- elementwise helpers ------------------------------------------------- *
 * Both delegate to the header's NEON sweep primitives (math is assembly at
 * every step; one implementation, owned by mimi_decode.cpp). Safe in place
 * (src==dst) — the sweeps are elementwise with no cross-lane dependence. */
static inline void seanet_elu(const float *src, float *dst, int n) {
    mimi_elu_vec_f32(src, dst, n, MIMI_ELU_ALPHA);
}

static inline void seanet_vadd(const float *a, const float *b, float *y, int n) {
    mimi_add_vec_f32(a, b, y, n);
}

/* ---- init ---------------------------------------------------------------- */
int mimi_seanet_init(MimiSeanetState **st, const MimiWeightTable *w,
                     MimiArena *a, char *err, size_t errlen) {
    MimiSeanetState *s =
        static_cast<MimiSeanetState *>(mimi_arena_alloc(a, sizeof(MimiSeanetState)));
    *st = s;

    // Carve the liveness buffers before child state so their spans can be
    // borrowed by every matrix route. No convolution owns matrix scratch.
    s->b0 = static_cast<float *>(
        mimi_arena_alloc(a, MIMI_SEANET_BUF_ELEMS * sizeof(float)));
    s->b1 = static_cast<float *>(
        mimi_arena_alloc(a, MIMI_SEANET_BUF_ELEMS * sizeof(float)));
    s->b2 = static_cast<float *>(
        mimi_arena_alloc(a, MIMI_SEANET_BUF_ELEMS * sizeof(float)));

    char prefix[128];

    // decoder.model.0 : init conv  (512 -> 1024, k=7, s=1, d=1, g=1, causal)
    int rc = mimi_conv_init(&s->init_conv, w, "decoder.model.0",
                            INIT_IN, INIT_OUT, MIMI_KERNEL,
                            /*stride*/ 1, /*dilation*/ 1, /*groups*/ 1,
                            /*causal*/ 1, s->b2, MIMI_SEANET_BUF_ELEMS,
                            a, err, errlen);
    if (rc != 0) return rc;

    for (int L = 0; L < NUM_LAYERS; ++L) {
        const int dim    = LAYER_UP_OUT[L];      // resnet dim
        const int hidden = dim / 2;              // compress = 2

        // decoder.model.{UP_IDX} : convtranspose upsample, stride = ratio, groups = 1
        snprintf(prefix, sizeof(prefix), "decoder.model.%d", LAYER_UP_IDX[L]);
        rc = mimi_convtr_init(&s->upsample[L], w, prefix,
                              LAYER_UP_IN[L], LAYER_UP_OUT[L], LAYER_UP_K[L],
                              /*stride*/ LAYER_RATIO[L], /*causal*/ 1,
                              s->b2, MIMI_SEANET_BUF_ELEMS, a, err, errlen);
        if (rc != 0) return rc;

        // decoder.model.{RES_IDX}.block.1 : conv (dim -> hidden, k=3, d=1)
        snprintf(prefix, sizeof(prefix), "decoder.model.%d.block.1", LAYER_RES_IDX[L]);
        float *matrix = L == 0 ? s->b2 + MIMI_SEANET_L0_RES_LIVE_ELEMS : nullptr;
        const size_t matrix_capacity =
            L == 0 ? MIMI_SEANET_BUF_ELEMS - MIMI_SEANET_L0_RES_LIVE_ELEMS : 0;
        rc = mimi_conv_init(&s->res_conv0[L], w, prefix,
                            dim, hidden, MIMI_RES_KERNEL,
                            /*stride*/ 1, /*dilation*/ 1, /*groups*/ 1,
                            /*causal*/ 1, matrix, matrix_capacity,
                            a, err, errlen);
        if (rc != 0) return rc;

        // decoder.model.{RES_IDX}.block.3 : conv (hidden -> dim, k=1, d=1)
        snprintf(prefix, sizeof(prefix), "decoder.model.%d.block.3", LAYER_RES_IDX[L]);
        rc = mimi_conv_init(&s->res_conv1[L], w, prefix,
                            hidden, dim, /*ksize*/ 1,
                            /*stride*/ 1, /*dilation*/ 1, /*groups*/ 1,
                            /*causal*/ 1, nullptr, 0, a, err, errlen);
        if (rc != 0) return rc;
    }

    // decoder.model.14 : final conv (64 -> 1, k=3, s=1)
    rc = mimi_conv_init(&s->final_conv, w, "decoder.model.14",
                        FINAL_IN, FINAL_OUT, MIMI_LAST_KERNEL,
                        /*stride*/ 1, /*dilation*/ 1, /*groups*/ 1,
                        /*causal*/ 1, nullptr, 0, a, err, errlen);
    if (rc != 0) return rc;
    return 0;
}

/* ---- step ---------------------------------------------------------------- *
 * x   : latent [MIMI_DIM, n_in] conv-layout (channel-major).
 * pcm : [1, n_in*960] output samples.
 * returns n_out samples (== n_in*960), or a negative code on misuse.
 *
 * Mirrors SeaNetDecoder::step exactly:
 *   xs = init_conv.step(xs)
 *   for layer: xs = upsample.step( ELU(xs) ); xs = resnet.step(xs)
 *   xs = final_conv.step( ELU(xs) )
 * and SeaNetResnetBlock::step:
 *   ys = xs
 *   ys = conv0.step( ELU(ys) ); ys = conv1.step( ELU(ys) )
 *   out = ys + xs                                    (true_skip identity add)
 * ELU is applied BEFORE every conv/convtr; there is NO activation before the
 * init conv. */
int mimi_seanet_step(MimiSeanetState *st, const float *x, int n_in, float *pcm) {
    if (n_in <= 0) return 0;                       // empty StreamTensor -> 0 frames
    if (n_in > MIMI_SEANET_MAX_N_IN) return -1;    // buffer overflow: sizing/contract bug

    // init conv (stride 1 -> frame-preserving): b0 = [1024, n]
    int n = mimi_conv_step(st->init_conv, x, n_in, st->b0);

    for (int L = 0; L < NUM_LAYERS; ++L) {
        const int c_in  = LAYER_UP_IN[L];
        const int dim   = LAYER_UP_OUT[L];    // channels after upsample == resnet dim
        const int hidden = dim / 2;

        // ELU before the upsample, then convtr (stride = ratio -> n *= ratio).
        seanet_elu(st->b0, st->b1, c_in * n);                 // b1 = ELU(b0)
        n = mimi_convtr_step(st->upsample[L], st->b1, n, st->b0); // b0 = [dim, n]

        // --- SeaNetResnetBlock::step, skip input is b0 (dim x n) ---
        // block.1: conv0( ELU(skip) ) -> hidden
        seanet_elu(st->b0, st->b1, dim * n);                  // b1 = ELU(skip_in)
        int nh = mimi_conv_step(st->res_conv0[L], st->b1, n, st->b2); // b2 = [hidden, nh]
        // block.3: conv1( ELU(hidden) ) -> dim
        seanet_elu(st->b2, st->b2, hidden * nh);              // b2 = ELU(b2) (in place)
        int ny = mimi_conv_step(st->res_conv1[L], st->b2, nh, st->b1); // b1 = ys [dim, ny]
        // identity skip: b0 = ys + skip_in.  ny == n (stride-1 preserving); add over
        // the common length like StreamingBinOp(Add) — degenerate to full n here.
        int common = ny < n ? ny : n;
        seanet_vadd(st->b1, st->b0, st->b0, dim * common);    // b0 = ys + skip_in
    }

    // final ELU + last conv (stride 1): pcm = [1, n]
    seanet_elu(st->b0, st->b1, FINAL_IN * n);                 // b1 = ELU(b0)  [64, n]
    int n_out = mimi_conv_step(st->final_conv, st->b1, n, pcm);
    // final_activation is None => no post-conv activation.
    return n_out;
}

/* ---- reset --------------------------------------------------------------- *
 * SeaNetDecoder::reset_state: reset init/final convs, and every layer's
 * upsample + residual convs. The resnet skip_op holds no state in this config
 * (true_skip identity add, prev buffers always empty) so there is nothing else
 * to clear. */
void mimi_seanet_reset(MimiSeanetState *st) {
    mimi_conv_reset(st->init_conv);
    for (int L = 0; L < NUM_LAYERS; ++L) {
        mimi_convtr_reset(st->upsample[L]);
        mimi_conv_reset(st->res_conv0[L]);
        mimi_conv_reset(st->res_conv1[L]);
    }
    mimi_conv_reset(st->final_conv);
}

/* ========================================================================= *
 * NOTES
 * =========================================================================
 *
 * (a) RUST -> C++ MAPPING + decoder.model.N INDEX MAP
 *
 *   SeaNetDecoder::step        -> mimi_seanet_step
 *   SeaNetDecoder::reset_state -> mimi_seanet_reset
 *   SeaNetDecoder::new         -> mimi_seanet_init (layer list + weight capture,
 *                                 via the unit-2 conv primitives)
 *   SeaNetResnetBlock::step    -> inlined inside the per-layer loop of
 *                                 mimi_seanet_step (block.1 conv, block.3 conv,
 *                                 identity skip add)
 *   candle_nn::Activation::Elu(1.0) -> seanet_elu (calls mimi_elu_f32, alpha 1.0)
 *   StreamingBinOp(Add, D::Minus1)  -> seanet_vadd over common frame length
 *                                      (see (d): degenerate to full n here)
 *
 *   decoder.model.N -> structural element (weight-bearing slots only; the odd
 *   ELU slots 1/4/7/10/13 carry no weights):
 *     model.0        init_conv           StreamableConv1d 512->1024 k7 s1
 *                    weights: decoder.model.0.conv.conv.{weight[1024,512,7],bias[1024]}
 *     model.2        upsample[0]         ConvTranspose 1024->512 k16 s8
 *                    weights: decoder.model.2.convtr.convtr.{weight[1024,512,16],bias[512]}
 *     model.3        res_conv0[0]/[1]    block.1 conv 512->256 k3 ; block.3 conv 256->512 k1
 *                    weights: decoder.model.3.block.1.conv.conv.{weight[256,512,3],bias[256]}
 *                             decoder.model.3.block.3.conv.conv.{weight[512,256,1],bias[512]}
 *     model.5        upsample[1]         ConvTranspose 512->256 k12 s6   [512,256,12]
 *     model.6        res_conv0[1]/[1]    256->128 k3 [128,256,3] ; 128->256 k1 [256,128,1]
 *     model.8        upsample[2]         ConvTranspose 256->128 k10 s5   [256,128,10]
 *     model.9        res_conv0[2]/[1]    128->64 k3 [64,128,3] ; 64->128 k1 [128,64,1]
 *     model.11       upsample[3]         ConvTranspose 128->64 k8 s4     [128,64,8]
 *     model.12       res_conv0[3]/[1]    64->32 k3 [32,64,3] ; 32->64 k1 [64,32,1]
 *     model.14       final_conv          StreamableConv1d 64->1 k3 s1    [1,64,3]
 *   Every weight name above was cross-checked against the checkpoint dump; all
 *   present and shape-matched.
 *
 * (b) PER-LAYER GEOMETRY (in_c, out_c, k, stride, dilation), all causal, groups=1:
 *     init conv       512  -> 1024   k7  s1 d1
 *     L0 upsample    1024  ->  512   k16 s8            (ratio 8)
 *        res block.1  512  ->  256   k3  s1 d1
 *        res block.3  256  ->  512   k1  s1 d1
 *     L1 upsample     512  ->  256   k12 s6            (ratio 6)
 *        res block.1  256  ->  128   k3  s1 d1
 *        res block.3  128  ->  256   k1  s1 d1
 *     L2 upsample     256  ->  128   k10 s5            (ratio 5)
 *        res block.1  128  ->   64   k3  s1 d1
 *        res block.3   64  ->  128   k1  s1 d1
 *     L3 upsample     128  ->   64   k8  s4            (ratio 4)
 *        res block.1   64  ->   32   k3  s1 d1
 *        res block.3   32  ->   64   k1  s1 d1
 *     final conv       64  ->    1   k3  s1 d1
 *   Dilations: residual conv dilation = dilation_base^j; n_residual_layers=1 so
 *   only j=0 => dilation 2^0 = 1 for both residual convs (the block list is
 *   [(residual_kernel_size, 1), (1, 1)]). No dilated convs anywhere in the
 *   decoder. Hidden = dim/compress = dim/2 per residual block.
 *
 * (c) ACTIVATION PLACEMENT (read from step(), not forward()): ELU(alpha=1.0) is
 *   applied BEFORE each conv/convtr — before every upsample, before both residual
 *   convs, and before the final conv. There is NO activation before the init
 *   conv, and NO activation after the final conv (final_activation = None). This
 *   is the "pre-activation" placement of SeaNetResnetBlock::step
 *   (`block.step(&ys.apply(activation)?)`) and SeaNetDecoder::step
 *   (`upsample.step(&xs.apply(activation)?)`, `final_conv.step(&xs.apply(activation)?)`).
 *
 * (d) FRAME-COUNT ARITHMETIC through the chain (n = latent frames in):
 *     Every StreamableConv1d (stride 1, causal) is frame-preserving: it left-pads
 *     (k-1)*d zeros ONCE on the first step, so it emits exactly n_in frames every
 *     step (including the first) and retains (k-1)*d frames of left context.
 *     Every StreamableConvTranspose1d emits exactly n_in*stride frames every step
 *     (raw convtr length (n-1)*s+k, minus invalid_steps=(k-s) split off as state,
 *     leaving n*s), including the first. Therefore:
 *         init:  n
 *         L0:    n -> *8  = 8n     (resnet preserves)
 *         L1:    8n -> *6 = 48n
 *         L2:    48n -> *5 = 240n
 *         L3:    240n -> *4 = 960n
 *         final: 960n            => n_out = 960 * n_in, no warmup drop.
 *     For the per-decode_step case n_in = 2 (12.5Hz latent -> x2 upsample ->
 *     transformer -> seanet) => 1920 samples = MIMI_FRAME_OUT. Buffers are sized
 *     for n_in up to MIMI_SEANET_MAX_N_IN=4 (960*4=3840 = pcm_out capacity).
 *     Because both branches of the residual skip are frame-preserving, ys and the
 *     identity skip always have equal length n, so StreamingBinOp(Add)'s
 *     prev_lhs/prev_rhs buffers never fill; the add is a plain elementwise add
 *     over dim*n. (The `common = min(ny,n)` guard is defensive only.)
 *
 * (e) UNCERTAINTIES / ABI FRICTION
 *   1. WEIGHT-NAME PREFIX CONVENTION (primary): this file passes each conv/convtr
 *      the StreamableModule VarBuilder node as `prefix` (e.g. "decoder.model.0",
 *      "decoder.model.3.block.1", "decoder.model.2") and assumes unit-2
 *      (mimi_conv.cpp) appends the primitive's internal nesting —
 *      ".conv.conv.{weight,bias}" for mimi_conv_init and
 *      ".convtr.convtr.{weight,bias}" for mimi_convtr_init — matching the Rust
 *      wrapping (StreamableConv1d -> NormConv1d(pp"conv") -> Conv1d(pp"conv");
 *      StreamableConvTranspose1d -> NormConvTranspose1d(pp"convtr") ->
 *      get("convtr"/"weight")). The header's doc example ("decoder.model.0.conv.weight")
 *      drops a ".conv", so if unit-2 instead expects the full leaf prefix, only
 *      the prefix string literals in mimi_seanet_init change — the geometry is
 *      unaffected. Full expected leaf names are enumerated in NOTE (a) for the
 *      arbiter to reconcile against unit-2.
 *   2. WEIGHT-NORM FOLD: NormConv1d/NormConvTranspose1d use Norm::WeightNorm
 *      (g*v/||v||, ||.|| over dims (1,2)). That fold is unit-2's / the weight-table
 *      capture's responsibility (folds once at init per the manifest); this file
 *      does not touch weights.
 *   3. BIAS ON CONVTR: NormConvTranspose1d applies bias once after conv_transpose;
 *      StreamableConvTranspose1d subtracts the bias from the overlap-added tail so
 *      it is not double-counted across steps. This is entirely inside
 *      mimi_convtr_step (unit-2); this file assumes that carry is handled there.
 *   4. n_in CONTRACT: MIMI_SEANET_MAX_N_IN=4 is inferred from the pcm_out 2x drain
 *      headroom (3840/960). The real pipeline always feeds n_in=2. If a caller
 *      ever exceeds 4, mimi_seanet_step returns -1 rather than overflow a buffer.
 *   5. ELU EXACTNESS: candle Elu(1.0) == x>0 ? x : (exp(x)-1); reproduced via
 *      mimi_elu_f32(x, 1.0). Numeric tier is "faithful" (libm expf), not bit-exact.
 */
