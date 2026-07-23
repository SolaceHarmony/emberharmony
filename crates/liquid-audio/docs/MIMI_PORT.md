# Native Mimi port manifest

Status: **retained native codec for the future Moshi tranche; deliberately
unwired from released LFM2.5-Audio.** `liblfm_mimi.a` remains build-checked and
its private native ABI remains available to the future Moshi model owner.
`mimi_decode_plan_new_from_image` binds only
`LFM_WEIGHT_COMPONENT_MIMI`, a component that the LFM2.5 loader never
populates. There is no LFM2.5 request kind, route edge, conversation state, or
fallback that can select Mimi.

Released LFM2.5 output is instead
`REQ_TOKEN_PASS → REQ_DEPTH_FRAME → REQ_AUDIO_DETOKENIZE`, using the checkpoint's
required `audio_detokenizer/` component. This manifest preserves the native
Mimi implementation and its remaining optimization work so Moshi can adopt it
without confusing the two model contracts.

## Source of truth

`moshi 0.6.4` (crates.io registry copy:
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/moshi-0.6.4/src/`),
originally entered from the former Rust `MimiDetokenizer` implementation:

- **Hot path (per generated frame): `Mimi::decode_step`** — mimi.rs:214
  1. `SplitResidualVectorQuantizer::decode` (quantization.rs:383)
  2. `ConvTrUpsample1d::step` (conv.rs:603) — stride 2, dim 512 (12.5 Hz → 25 Hz)
  3. `decoder_transformer` `ProjectedTransformer::step` (transformer.rs) — 8
     layers, d_model 512, 8 heads, causal, context 250, RoPE (max_period 10⁴),
     LayerNorm, LayerScale 0.01, MLP = linear1 → gelu_erf → linear2 (ff 2048,
     no biases), kv_repeat 1, conv_layout=true
  4. `SeaNetDecoder::step` (seanet.rs:452) — ratios [8,6,5,4] (×960), n_filters
     64, kernel 7 / residual 3 / last 3, 1 residual unit per block, ELU(1.0),
     causal, WeightNorm, Constant pad, true_skip, compress 2
  → 1920 f32 samples @ 24 kHz per latent frame (80 ms)
- **Turn boundary: `reset_state`** (decoder half only)
- Weights: `tokenizer-e351c8d8-checkpoint125.safetensors`, f32, n_q=8
  (rvq_first 1 + rvq_rest 7), bins 2048, quantizer dim 256 ↔ model dim 512.

**Out of the retained decoder scope (oracle/training only):** the encoder half
(`encode*` — used only by training preprocessing), batching >1,
`StreamMask` batched masking, quantized-weight paths (`MaybeQuantized*` —
this checkpoint is unquantized f32), cross-attention / gating / RmsNorm /
conv-block transformer variants (config says None/LayerNorm/false), LSTM
(seanet lstm=0).

## Native port units

| # | C++ file (native/src/mimi/) | Rust source | Port | Skip |
|---|---|---|---|---|
| 1 | `mimi_quant.cpp` | quantization.rs | `EuclideanCodebook::decode`, `VectorQuantization::decode`, `ResidualVectorQuantization::decode`, `ResidualVectorQuantizer::decode`, `SplitResidualVectorQuantizer::decode` (+ in/out projections) | all `encode*`, CustomOp2, training |
| 2 | `mimi_conv.cpp` | conv.rs | `NormConv1d`/`NormConvTranspose1d` forward math incl. weight-norm fold semantics, `StreamableConv1d::step`, `StreamableConvTranspose1d::step` (pending/partial-frame state carry), `ConvTrUpsample1d::step` | `ConvDownsample1d` (encode-only), batched mask paths |
| 3 | `mimi_seanet.cpp` | seanet.rs | `SeaNetResnetBlock::step`, `SeaNetDecoder::step` (+ ELU), reset | `SeaNetEncoder` |
| 4 | `mimi_transformer.cpp` | transformer.rs | `LayerScale`, `RotaryEmbedding`/`Rope` (rope_i interleaved), `StreamingMultiheadAttention::forward` (causal, kv_repeat 1), `Mlp::NoGating` (gelu_erf), `LayerNorm`, `StreamingTransformerLayer`, `StreamingTransformer::step`, `ProjectedTransformer::step` (projs are None at 512↔512) | cross-attn, gating, RmsNorm, conv-block, batched |
| 5 | `mimi_decode.cpp` | mimi.rs (+ streaming.rs as reference) | `Mimi::decode_step` orchestration over units 1–4, `reset_state` (decoder half), config `v0_1(8)` constants | `encode*`, `load*` (weights arrive via the table), batching |

Shared contract: `native/src/mimi/mimi_kernel.h` (arbiter-authored) — weight
table, state arena, C ABI. nn.rs collapses into the header's plain-f32 linear.
streaming.rs (`StreamTensor` = Option<Tensor>) becomes explicit
`n_in/n_out` frame counts — no optional-tensor plumbing in C++.

Future Moshi ownership is split: `MimiDecodePlan` belongs to the native Moshi
model owner and owns validated byte views plus the formula-derived immutable
arena; `MimiDecodeState` belongs to one Moshi conversation and owns
KV/carry/scratch. Plan construction probes the exact state arena size, seals
derived storage, and subsequent state creation replays only the derived
offsets—never the fold math. The isolated implementation measured 16,777,344
derived bytes once per model and 48,808,616 mutable bytes per conversation,
with zero compatibility-copied weight bytes. Those bytes are not allocated by
an LFM2.5 model or conversation.

## Discipline (same as the engine, non-negotiable)

- **Weights are bytes and typed views**: Mimi binds the Codec component of the
  model-owned resident image as little-endian F32 byte spans. Unaligned views
  use safe byte loads; aligned production views dispatch directly to AMX/NEON.
  Weight-norm and codebook folds are formula-changing derived storage and are
  accounted separately. Layout, alignment, dtype, and transpose copies are
  forbidden. ConvTranspose uses the checkpoint matrix through a transposed
  GEMM view; depthwise upsample deinterleaves checkpoint taps in registers.
- **Zero allocation in steady state**: all streaming state (conv left-context,
  partial-frame pendings, KV rings, scratch) lives in ONE arena sized at init.
  State structs are POD and explicitly serializable (hibernation-friendly).
- **Destination-direct activations**: the transformer in-projection keeps only
  its bounded Q plane (8×512 f32 = 16 KiB). Completed K and V head spans write
  directly into their generation-selected ring slots from the packed resident
  checkpoint view; K is then rotated in place. The former packed QKV plane and
  both ring `memcpy` operations are gone. Remaining activation planes are
  listed honestly in the status section rather than treated as tensors.
- **Math**: f32 in, f32 accumulate. **Assembly at every step** (her rule,
  2026-07-09): no tensor-op thinking — NEON/aarch64 has an equivalent for
  every one of them. GEMM/GEMV tier is **AMX via Accelerate**
  (`cblas_sgemm`/`cblas_sgemv`) — we have a matrix coprocessor, nobody
  hand-rolls a vanilla GEMM. EVERYTHING else — conv inner loops, softmax
  reductions, layernorm, RoPE rotation, ELU/GELU sweeps, elementwise
  add/scale — is architecture SIMD from the first pass; scalar C++ exists only
  in exact-order reductions, setup control, and sub-vector tails. Transcendentals
  (erff/expf) stay lane-wise libm inside the NEON sweeps on the faithful tier;
  polynomial vector approximations are not linked.
- **Accumulation order is documented per kernel.** Target numerics tier:
  *faithful* — ulp-band parity vs moshi-Rust per module (candle's blocked gemm
  is not economically bit-reproducible); thresholds measured by the harness,
  not asserted blind. The end-to-end wav byte hash WILL move → oracle re-arm
  is deliberate and hers, with the directly invoked native real-checkpoint
  trace and DECODE_ENGINE.md updated together, never alone.
- **No cross-model fallback**: Moshi adoption will require the native Mimi
  component and fail when it is absent. LFM2.5 requires the released audio
  detokenizer and fails when it is absent. Neither substitutes for the other.
- C linkage entry points; no exceptions across the ABI; no candle, no Rust
  types below the seam.

## Verification required before Moshi adoption

1. C++23 full-value component traces cover quantization, convolution carry,
   KV-ring wrap, reset, and the complete native decode chain. No Rust oracle or
   FFI execution path is restored.
2. A fixed-token native chain carries state across at least 130 frames, crosses
   the 250-slot KV wrap, resets, and repeats deterministically.
3. The Mimi kernel runs as a typed pass on the same
   Flashkern lane team as the Moshi backbone, with its own request kind at the
   pass boundary.
   Because it is a native C++ program (no Rust frames cross the fences), its lane
   fences use the shared expected-value word and block without a spin tier — the
   depthformer's transitional pure-spin barrier does not apply here. Unit inner
   loops are written
   band-splittable so lane-cutting is a schedule change, not a math change.

## Status

- [x] first passes (6 agents, one file each) — ALL LANDED
- [x] arbiter review vs Rust source, per file: quant ✅ seanet ✅ decode ✅
      transformer ✅ conv ✅ (kv parked — RotatingKvCache ruling, see unit 5
      row). Whole-kernel link green (5 objects + Accelerate).
      Arbiter catches recorded: layer_norm two-candles fork (unit 6 matched
      the slow tensor-op path; the REAL path is ops.rs cpu_fwd — one-pass
      sum/sum², naive var, recip(sqrt)) — rewritten; softmax final op is
      per-element DIVISION (`*d /= sum_exp`), not reciprocal-multiply —
      rewritten; **builds MUST use `-ffp-contract=off`**. Checkpoint
      stores PRE-FOLDED conv weights (0 weight_g/v) — fold path dormant,
      arena tightens toward ~16 MiB zero-copy.
      Prefix seam reconciled: seanet passes streamable-node prefixes, conv
      appends `.conv.conv`/`.convtr.convtr` — both sides chose the same
      convention independently.
- [x] Shadow-review disposition (independent review she commissioned;
      findings verified against candle source, never taken on faith):
      score-alias + layernorm
      findings already fixed before the review landed; its two NEW catches
      confirmed against candle source and fixed: (1) gelu_erf — candle calls
      the RUST-libm erff (erf_f32 == libm::erff, whose erfc2 uses libm's own
      expf), and associates ((erf+1)·0.5)·v — Rust-libm erff/expf/scalbnf now
      ported VERBATIM (SunPro float ports, selftest vs reference values green
      across all branches) and the association fixed in scalar + NEON sweeps;
      (2) softmax — candle exps the WHOLE row then reduces separately with
      vec_sum's exact NEON blocking (STEP=16, four q-accumulators, tree
      reduce, scalar leftovers appended) — restructured to match bit-exactly.
      Also: elu selects on is_sign_positive (sign-bit, not x>0) — matched.
      Declared remaining ulp sources (thresholds ledger): AMX cblas GEMMs vs
      candle's gemm crate blocking; layernorm's NEON lane-blocked sums vs
      candle's strictly-sequential scalar sums (kept NEON per the
      assembly-at-every-step rule); NEON max reduction (exact-equal for
      non-NaN rows). Everything else in the transformer chain is bit-matched
      by construction.
      Review perf finding — FIXED (a4a11d43, was deferred until the final
      verdict measured it at 3.3x slower than Rust/moshi): the widest seanet
      layers receive n=2 time samples, so time-axis NEON ran in its scalar
      tail (~45 of ~70 ms). AMX routes via mimi_gemm_f32: conv1d im2col +
      single GEMM (weight rows already (ic,kk)-contiguous — zero-copy A);
      convtr now presents checkpoint `[ic,oc,kk]` directly as the transpose of
      `[ic,oc*kk]` to ONE GEMM, zero-copy X and zero weight re-arm.
      Measured, real checkpoint, 130 frames across the KV wrap:
      **76.5 -> 13.8 ms median/frame (max 14.2; old spiked to 103), vs
      Rust/moshi ~20.8 ms** — 5.6x on ourselves, 1.5x on candle, 5.8x
      real-time headroom, single-threaded, before any lane banding.
      Route-parity vs the proven build: 2.5e-6 (chain bar vs moshi 4.1e-6).
      Verdict's remaining P1s also closed this pass: convtr/upsample n_in
      ABI bounds (arena overrun), exact-shape + null-data weight validation.
      Still open from the verdict: AMX dispatch is inferred from the 5.6x on
      GEMM-bound shapes, not proven by counters; cold init ~665 ms (page
      faults + re-arm) needs one measurement pass at integration.
- [x] isolated build wiring: CMake and the native tools Makefile expose
      `liblfm_mimi.a` as an explicit C++23 component with
      `-ffp-contract=off` (load-bearing). It is excluded from the default
      LFM2.5 product and speech-test link. The private plan binds only the
      reserved Mimi component.
- [x] LFM2.5 isolation: native LFM2.5 recurrence has no `REQ_MIMI_DECODE`, no
      Mimi state, no Mimi weights, and no version fallback. Its released
      detokenizer is mandatory.
- [ ] Moshi production adoption: the future native Moshi model owner must load
      its main and Mimi sources into one image, construct the retained native
      plan/state, and mount the request on its coroutine route.
- [ ] native C++ trace gate: the former Rust FFI parity harness was retired
      with Rust inference ownership. Before Moshi can mount this retained
      component, replace its historical 130-frame/KV-wrap evidence with a
      direct C++23 full-value trace over the component API.
- [ ] Moshi engine integration: add a Moshi-owned typed Mimi request to the
      native SQ/CQ. Equal-rate output must write its retained playback
      reservation directly; other device rates use conversation-owned scratch
      and the same route's native stream resampler.
- [x] transformer QKV destination cut: Q uses a 16 KiB fixed scratch plane;
      each K/V output head projects directly from the packed resident byte view
      into its final rotating-cache slot. The original 48 KiB QKV plane is gone,
      saving 32 KiB per state and deleting both K/V transport copies without
      changing any dot-product, bias, RoPE, or ring-position boundary.
- [x] convolution carry publication cut: Conv1d, ConvTranspose1d, and the
      depthwise upsampler gather/accumulate the next carry in the existing
      equal-shaped write bank, then rotate the read/write pointers after the
      old bank's final read. Actual-kernel aligned/unaligned, priming,
      multi-step, matrix-route, and reset goldens are bit exact. This removes
      50,688 copied bytes and 2,497 nonzero `memcpy` calls per steady decode
      without changing arithmetic or the two-bank state footprint.
- [ ] cooperative interior: split Mimi units across the fixed team when Moshi
      mounts it, using the NOTES maps (conv: out-channel; attention: head;
      sweeps: sub-range). The isolated graph is correct and fast
      (13.8 ms/frame), but its stateful interior remains serial-with-AMX. The
      remaining large mutable
      planes are `attn_cat`, `branch`, MLP hidden storage, and SeaNet activation/
      residual ping-pong. Carry banks remain necessary while old overlap and the
      next tail coexist, but their publication is now pointer-only. Eliminate or
      alias the other planes only where their last-consumer lifetimes prove it safe.
