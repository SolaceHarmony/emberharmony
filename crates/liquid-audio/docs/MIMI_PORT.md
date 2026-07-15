# Mimi → C++/NEON port manifest

The mission clause this executes: the voice pipeline decode path ports to
kcoro/NEON/C++ as a tight kernel. After the backbone (REQ_TOKEN_PASS) and the
depthformer (REQ_CALL), **Mimi is the largest candle compute left per frame**:
every 80 ms audio frame runs a full candle graph (moshi crate) → PCM. This
manifest scopes the port; the first pass is swarmed one-file-per-agent,
arbitered locally, then parity-gated.

## Source of truth

`moshi 0.6.4` (crates.io registry copy:
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/moshi-0.6.4/src/`),
entered from our `MimiDetokenizer` (src/runtime/audio_out.rs):

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

**Out of scope (stays on the Rust moshi crate):** the encoder half
(`encode*` — used only by training preprocessing, turn-level), batching >1,
`StreamMask` batched masking, quantized-weight paths (`MaybeQuantized*` —
this checkpoint is unquantized f32), cross-attention / gating / RmsNorm /
conv-block transformer variants (config says None/LayerNorm/false), LSTM
(seanet lstm=0).

## Port units — one file per agent

| # | C++ file (native/src/mimi/) | Rust source | Port | Skip |
|---|---|---|---|---|
| 1 | `mimi_quant.cpp` | quantization.rs | `EuclideanCodebook::decode`, `VectorQuantization::decode`, `ResidualVectorQuantization::decode`, `ResidualVectorQuantizer::decode`, `SplitResidualVectorQuantizer::decode` (+ in/out projections) | all `encode*`, CustomOp2, training |
| 2 | `mimi_conv.cpp` | conv.rs | `NormConv1d`/`NormConvTranspose1d` forward math incl. weight-norm fold semantics, `StreamableConv1d::step`, `StreamableConvTranspose1d::step` (pending/partial-frame state carry), `ConvTrUpsample1d::step` | `ConvDownsample1d` (encode-only), batched mask paths |
| 3 | `mimi_seanet.cpp` | seanet.rs | `SeaNetResnetBlock::step`, `SeaNetDecoder::step` (+ ELU), reset | `SeaNetEncoder` |
| 4 | `mimi_transformer.cpp` | transformer.rs | `LayerScale`, `RotaryEmbedding`/`Rope` (rope_i interleaved), `StreamingMultiheadAttention::forward` (causal, kv_repeat 1), `Mlp::NoGating` (gelu_erf), `LayerNorm`, `StreamingTransformerLayer`, `StreamingTransformer::step`, `ProjectedTransformer::step` (projs are None at 512↔512) | cross-attn, gating, RmsNorm, conv-block, batched |
| 5 | `mimi_kv.cpp` | kv_cache.rs | ~~ScatteredKvCache~~ **PARKED** — unit 5's port proved the streaming path never uses it: transformer.rs imports `kv_cache::KvCache` = an enum over **candle_nn `RotatingKvCache`**, mask built inline in StreamingTransformer::step (t==1 no-mask fast path; allow-rule `last_reset_pos <= k_pos <= t_pos <= k_pos+context`). The real cache is ported inside unit 4 from candle-nn-0.9.2 kv_cache.rs:336 + the inline mask. mimi_kv.cpp stays in-tree unwired (correct work, batched-serving consumer only). | |
| 6 | `mimi_decode.cpp` | mimi.rs (+ streaming.rs as reference) | `Mimi::decode_step` orchestration over units 1–5, `reset_state` (decoder half), config `v0_1(8)` constants | `encode*`, `load*` (weights arrive via the table), batching |

Shared contract: `native/src/mimi/mimi_kernel.h` (arbiter-authored) — weight
table, state arena, C ABI. nn.rs collapses into the header's plain-f32 linear.
streaming.rs (`StreamTensor` = Option<Tensor>) becomes explicit
`n_in/n_out` frame counts — no optional-tensor plumbing in C++.

## Discipline (same as the engine, non-negotiable)

- **Weights are a buffer**: one flat `name → (f32*, len)` table captured
  zero-copy from the native resident safetensors image. Weight-norm folds ONCE at capture
  (g·v/‖v‖ per output channel), never per step. No transpose/repack per call —
  if a layout re-arm is needed, it happens once at init into the arena, and the
  manifest documents it.
- **Zero allocation in steady state**: all streaming state (conv left-context,
  partial-frame pendings, KV rings, scratch) lives in ONE arena sized at init.
  State structs are POD and explicitly serializable (hibernation-friendly).
- **Math**: f32 in, f32 accumulate. **Assembly at every step** (her rule,
  2026-07-09): no tensor-op thinking — NEON/aarch64 has an equivalent for
  every one of them. GEMM/GEMV tier is **AMX via Accelerate**
  (`cblas_sgemm`/`cblas_sgemv`) — we have a matrix coprocessor, nobody
  hand-rolls a vanilla GEMM. EVERYTHING else — conv inner loops, softmax
  reductions, layernorm, RoPE rotation, ELU/GELU sweeps, elementwise
  add/scale — is hand NEON intrinsics from the FIRST pass; scalar C++ exists
  only in the `_ref` parity siblings and sub-vector tails. Transcendentals
  (erff/expf) stay lane-wise libm inside the NEON sweeps on the faithful tier;
  polynomial vector approximations are fast-tier, admitted only behind the
  parity gate.
- **Accumulation order is documented per kernel.** Target numerics tier:
  *faithful* — ulp-band parity vs moshi-Rust per module (candle's blocked gemm
  is not economically bit-reproducible); thresholds measured by the harness,
  not asserted blind. The end-to-end wav byte hash WILL move → oracle re-arm
  is deliberate and hers, gate.sh + DECODE_ENGINE.md together, never alone.
- **No fallbacks**: the native Mimi, once gated in, is required — absent it,
  hard error (Mimi always ships; never the LFM2 detokenizer silently).
- C linkage entry points; no exceptions across the ABI; no candle, no Rust
  types below the seam.

## Verification ladder (arbiter-owned)

1. Per-module parity harness: Rust `#[test]`s feed identical weights + seeded
   inputs to the moshi module and the C++ unit via FFI; report max |Δ| and ulp
   band per stage. Bisect with the scalar reference path.
2. Chain parity: N-frame `decode_step` stream vs moshi, state carried across
   frames (the streaming semantics are where ports rot — partial frames,
   left-context, KV ring wrap at 250).
3. e2e: perf-chain wav vs current PERF hash — expected to move; the audible
   dual-path e2e + her ear bless the re-arm.
4. Integration (after parity — her directive, structural not optional): the
   Mimi kernel runs INSIDE the same kcoro engine as the backbone/depthformer
   (flashkern_engine.cpp) — same persistent lane team, same doorbell, its own
   REQ kind at the pass boundary. Because it is a native C++ program (no Rust
   frames cross the fences), its lane fences PARK precisely after the bounded
   spin per the two-barrier doctrine — the depthformer's pure-spin barrier
   compromise does not apply here. Unit inner loops are written
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
      rewritten; **builds MUST use `-ffp-contract=off`** (clang default
      contracts a*b+c into fma even in scalar _ref paths; rustc never does —
      without this flag the parity siblings are not oracles). Checkpoint
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
      convtr re-armed once at init to [kk][oc][ic] + ONE GEMM, zero-copy X.
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
- [x] build wiring: build.rs compiles the five active units (c++23,
      -ffp-contract=off — load-bearing); Rust rim = src/mimi_native.rs
      (zero-copy weight table over the native-owned checkpoint image, infallible-or-Err
      init, Mutex'd single-slot decoder).
- [x] PRODUCTION SWAP: MimiDetokenizer::decode_step runs the NATIVE kernel —
      the moshi decode_step call is out of the streaming pipeline. moshi
      remains ONLY turn-level tooling (encode for the trainer; one-shot
      whole-clip decode, which the byte-oracle example pins — so REF/PERF
      hashes did NOT move this rung; they re-arm if/when one-shot decode
      goes native).
- [x] chain parity, in-repo (tests/mimi_native_parity.rs, gate rung 2/6):
      130 frames across the 250-slot KV wrap through the production FFI —
      worst |Δ| = 3.085e-6 (assert 5e-5), post-reset 2.9e-7. Tighter than
      the shadow review's 4.11e-6: the sequential-layernorm fix landed in
      between (lane-blocked reduction NaN'd on near-constant rows — probe-
      proven; accumulation now bit-matches candle's sequential order, apply
      stays NEON). Final verdict's P2s also closed: stage errors propagate
      through mimi_decoder_step (negative rc never reads as priming);
      upsample weight validated exact-shape + non-null.
- [ ] engine integration (the remaining rung): mimi as a native C++ lane
      program on the SAME kcoro engine team — REQ_MIMI at the F4 doorbell,
      units band-split per the NOTES maps (conv: out-channel; attention:
      head; sweeps: sub-range), parked fences (native program, two-barrier
      doctrine). Today the kernel is serial-with-AMX inside one rim call —
      correct and fast (13.8 ms/frame), but not yet ON the lane team.
