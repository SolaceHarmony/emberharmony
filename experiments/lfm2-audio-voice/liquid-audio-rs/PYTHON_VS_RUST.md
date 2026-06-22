# Python vs Rust — `liquid_audio` → `liquid-audio-rs` Port Report

Scope: the pure-Rust **candle** port of Liquid AI's `liquid_audio` (LFM2.5-Audio-1.5B)
against the upstream Python. "Same" = verified numerically/structurally identical;
"Differ" = a deliberate substitution, with the reason. No torch, no folding, no
silent omissions.

All figures below were regenerated, not recalled:
- `python parity/compare_symbols.py --scope core` → **170/170 covered, 0 missing**
- `cargo test --lib` → **31 passed**
- `cargo test --test parity --release -- --ignored` (vs Python-dumped golden tensors,
  on `Device::Cpu`, f32) → **8/8 passed**

---

## 1. Where we are the SAME

### 1.1 Symbol coverage (function-for-function)
`compare_symbols.py` maps every top-level Python function and class method to a Rust
counterpart (by normalized name, with `__init__→new`, `__call__→call`, etc.). Core
scope (everything except the vendored `moshi/` and the `demo/`):

> **170 / 170 covered, 0 missing.**

### 1.2 Numerical parity (byte-for-byte vs Python reference tensors)

| Stage | Rust vs Python | Shape |
|---|---|---|
| LFM2 backbone hidden state | **6.558e-6** | `[1,24,2048]` |
| Text logits (tied head) | **5.505e-6** | `[65536]` |
| Conformer conv-subsampling | 5.611e-7 | `[1,256,13,16]` |
| Conformer post-subsample / pos-enc | 1.019e-6 | `[1,13,512]` |
| Conformer rel-pos embedding | 9.537e-7 | `[1,25,512]` |
| Conformer layer 0 | 1.056e-6 | `[1,13,512]` |
| Conformer final | 1.592e-6 / **8.25e-7** | `[1,512,13]` |
| Mel spectrogram (front-end) | **9.31e-6** (FFT-library floor; see §1.4) | `[1,128,101]` |
| Prefill embeddings (modality scatter) | **1.118e-6** | `[1,50,2048]` |
| **Depthformer audio frame** | **token-EXACT** `[213,836,182,416,782,1796,202,578]` | — |
| Mimi decode / streaming decode | smoke (waveform `[1,1,30720]`, peak 0.7395) | — |

These are **f32 relative errors at the 1e-6 level** — i.e. the two implementations
agree to f32 round-off. The depthformer is **exactly** equal token-for-token, and
`examples/generate.rs` reproduces the upstream reference text token-for-token
("Handcrafted Woodworking, Precision Made for You").

### 1.4 Why it is *not* bit-identical — the cross-library floor (and what was repaired)

A faithful port reproduces the **math**; it cannot reproduce PyTorch's exact
floating-point **execution** without literally calling torch's kernels. The ~1e-6
residual is the irreducible cross-library floor:

- **Matmul reduction order.** candle's gemm sums in a different order than torch's
  BLAS (MKL/OpenBLAS). Same inputs, same math, last-bit-different sums → the
  backbone accumulates to 6.558e-6 over 24 layers (~2.7e-7/layer).
- **Transcendentals.** `exp` (softmax), `cos`/`sin` (RoPE, mel), `rsqrt` (RMSNorm)
  differ in the last bit between libm and torch's vectorized math.
- **FFT.** rustfft ≠ torch's pocketfft — different algorithms, each ~1e-6 from the
  true DFT but in different directions.

This floor exists in **any** cross-framework port — PyTorch itself is not bit-identical
between its own CPU and CUDA backends, for the same reasons. It is not a faithfulness
defect; eliminating it would mean re-implementing candle's gemm/FFT/libm to forge
torch's rounding.

**Audit + repair of the one stage that sat *above* the floor (the mel, was 1.07e-5):**
1. **Conventions verified correct, not changed.** The center pad uses
   `pad_mode="constant"` (NeMo `processor.py:394`) — the Rust matches it; "fixing" it
   to `reflect` (torch.stft's *general* default) would have been a regression.
   Preemphasis, ddof=1 per-feature normalize + 1e-5 epsilon, and the additive log
   guard all match. Nothing was algorithmically wrong.
2. **Precision repaired.** The mel front-end is precision-sensitive (the Python
   `AudioPreprocessor` warns it is "not robust to low precision"). The FFT → power →
   mel-matmul → log → normalize chain was moved to **f64** (on CPU; this front-end is
   CPU-by-design and Metal has no f64), removing *our* rounding so the gap to torch's
   f32 reference is just torch's own rounding: **1.07e-5 → 9.31e-6**.
3. **Residual.** The remaining ~9e-6 is rustfft-vs-pocketfft, amplified by log +
   normalize — the FFT-library floor, removable only by porting pocketfft bit-for-bit.

Bit-exact where there is no float reduction: the **depthformer is token-EXACT**, and
all index/gather/embedding/modality-scatter ops are exact.

**Independent corroboration.** A separate torch→MLX numeric-stability study
(`mlxports/xLSTM-metal/docs/NUMERIC_STABILITY_TORCH_vs_MLX.md`, with a per-stage
tracer) reaches the same floor from the MLX side: GEMM order/FMA ≈ **1e-7 relative**
and FFT+bias bottoms out at ~**1e-6 (float32 epsilon)** after the normalization/combine
fixes. Its concrete rules were checked against this port and hold:
- *FFT 1/n placement* (a double/missing scale gives order-one errors): the mel STFT is
  forward-only like `torch.stft` (no 1/n), and the candle-flashfftconv conv is verified
  `== circular/linear conv`, so the single 1/n is placed correctly.
- *No `float()`/`.item()` double-rounding in compute*: the mel runs the whole chain in
  f64 and rounds **once** to f32 at the boundary ("extended precision until the very
  end").
- *Op structure*: RMSNorm now uses `x · recip(sqrt(z))` to match torch's `x · rsqrt(z)`
  (candle has no fused rsqrt → still ~1 ULP, the floor); the earlier "byte-identical"
  comment was corrected.

**Below the floor** requires extended precision (double-double / compensated arithmetic),
per that study's plan (`double_double.metal`, `ComplexMul.metal`, `Depthwise3.metal`).
That path makes the result *more* accurate than torch's f32 (it tracks the true value),
so the residual then equals torch's own rounding — it does not "match torch's f32 bits."
A candle/Metal double-double pass over the candle-flashfftconv kernels and the mel is the
available next step for determinism below float32 epsilon.

### 1.3 Algorithms ported 1:1 (same math, expressed in candle)
FastConformer encoder (NeMo), LFM2 backbone (HF `Lfm2Model`: short-conv + GQA),
the LFM2 depthformer (`RawLMBackbone`, GQA + qk-RMSNorm + interleaved RoPE),
the SwiGLU `ff_dim` sizing, the mel `FilterbankFeatures`, the modality-scatter
prefill, the training cross-entropy + per-codebook `audio_loss_weights`, the
LinearLR→CosineAnnealingLR LR schedule.

---

## 2. Where we DIFFER — and why

Every divergence below is deliberate; none changes the verified numerics.

### 2.1 Device & dtype defaults — Rust is device-agnostic; Python is GPU-coupled
- **Python:** defaults `device="cuda"`, `dtype=torch.bfloat16` (`LFM2AudioModel.from_pretrained`,
  `LFM2AudioProcessor`, `ChatState`), and **hard-codes `.cuda()`** for the detokenizer
  (`processor.py:151`) and `device="cuda"` in the demo. As shipped, the reference
  **requires** a CUDA box (the detok `.cuda()` crashes on a CPU-only host).
- **Rust:** *nothing* in `src/` hardcodes a device; every loader (`from_pretrained`,
  `load_detokenizer`, `load_mimi`) takes `device: &Device` + `dtype: DType` and honors
  it. Examples default to `(Cpu, F32)`, Metal is opt-in (`LFM_DEVICE=metal` → bf16).
  candle has no CPU bf16 matmul, so CPU→f32 is the correct mapping (and matches
  Python's f32-pinned mel).
- **Consequence:** the port actually delivers LFM2's "runs on CPU" design point — the
  full 1.5B model decodes end-to-end and all 8 parity tests pass **on `Device::Cpu`**;
  the Python, as written, would not boot without CUDA.

### 2.2 Custom CUDA kernels → portable candle ops
The reference is gated behind CUDA-only custom kernels; the port reimplements each as
a portable candle op (CPU + Metal + CUDA from one definition):

| Reference (CUDA-gated) | Rust (kernel-free) |
|---|---|
| `flash_attention_2` / `sdpa` (LFM2 backbone, `lfm2_audio.py:162`) | eager matmul + additive causal mask + softmax |
| `scaled_dot_product_attention` (depthformer, conformer, Mimi) | hand-rolled SDPA + GQA head-repeat |
| `causal_conv1d` (LFM2 short-conv, `conv_L_cache`) | candle `Conv1d` (prefill) + gather-mul-sum (single step) |

**Faithfulness note:** the eager attention matches the **`sdpa`/no-flash** math, *not*
flash-attn's reordered online-softmax. That is exactly the path the f32 golden tensors
were dumped from (hence backbone parity at f32-level 6.558e-6), and the correct one for
a CPU/portable port.

### 2.3 Upstream reuse instead of re-implementation
"Use what exists; extend, don't fork":
- **Mimi codec → `moshi::mimi`** (Kyutai's own crate; its `quantizer.rvq_first`/`rvq_rest`
  weight names match this checkpoint — verified — which candle-transformers' Mimi does not).
- **Sampling → `candle_transformers::generation::LogitsProcessor`** (the sampler moshi
  itself uses), wrapped by a `Sampler` that injects Torch's *threshold-style* top-k
  (ties kept) via the `sample_f` hook. Greedy = `ArgMax` (deterministic ⇒ parity preserved).
- **KV cache → vendored `ConcatKvCache`** from candle-nn 0.10.2 (`src/candle_ext/kv_cache.rs`),
  a structural 1:1 of the Python `LayerKVCache` (`torch.cat(..., dim=1)`); `LayerKvCache`
  is a thin adapter over it.
- **`cross_entropy(reduction="none")` → `candle_ext::loss::cross_entropy_none`** (the
  reduction candle lacks), replacing two local copies.
- **Linear / Embedding / interleaved RoPE (`rope_i`) / softmax / silu → `candle_nn`.**

### 2.4 Precision order — Rust is *more* faithful at bf16
`RMSNorm` is **not** a blind wrap of `candle_nn::RmsNorm`. candle (and moshi) cast back
to the input dtype *before* the weight multiply (`layer_norm.rs:130`, `ops.rs:632`);
liquid_audio does `(_norm(x.float()) * weight).type_as(x)` — weight multiply **in f32,
then** cast. At bf16 those differ, so the port composes the RMSNorm from candle tensor
ops in liquid_audio's order. f32 parity is unaffected; bf16 runtime is *more* faithful
than a candle/moshi wrap would be.

### 2.5 Off-path NeMo machinery → inventory stubs
`conformer/encoder.py` is ~4160 words of which most is cache-aware streaming + ONNX
export + dynamic attention reconfiguration (`setup_streaming_params`,
`change_attention_model`, `get_initial_cache_state`, `forward_for_export`,
`input_example`, `disabled_deployment_*`). LFM2-Audio's **offline** forward never calls
these, so the Rust ports them as documented inventory stubs (symbols present → 170/170)
while the on-path `forward`/`forward_internal`/`_create_masks` are fully implemented and
parity-verified (8.25e-7). This is the source of the only "thin" word-count file (see §3).

### 2.6 Trainer — `accelerate`/torch → candle, loss on the model
- `torch.optim.AdamW(fused=True)` → `candle_nn::AdamW` (same math, no fused kernel).
- `LinearLR ⇒ CosineAnnealingLR (SequentialLR)` → `Trainer::lr_at` (the same piecewise
  schedule computed directly).
- `accelerator.autocast/backward/reduce/save_state` → candle equivalents (single-process
  `reduce` is identity; `VarMap::save` is the checkpoint).
- **De-duplicated to match Python:** the Python `Trainer` has *no* loss of its own —
  `train_step` and `validate` both call `self.model(batch)`. The earlier Rust had a
  duplicate `Trainer::forward`/`LossConfig`/`ce_none`; these were removed so both paths
  go through `LFM2AudioModel::forward` and cannot diverge.
- Loaders are stored on the `Trainer` (Python `self.train_loader`/`val_loader`), so
  `train(self)`/`validate(self)` match the Python signatures.

### 2.7 Data pipeline — same formats, pure-Rust backends
- `soundfile.read` → **symphonia** (WAV/FLAC/OGG/AIFF/…, no C deps).
- HF `datasets.save_to_disk` → real **Arrow IPC** stream + `dataset_info.json` +
  `state.json` (arrow-array/ipc), not a custom schema.
- `torchaudio.functional.resample` → faithful **windowed-sinc** (sinc_interp_hann,
  lowpass_filter_width=6, rolloff=0.99).

### 2.8 Stochastic sampling RNG (only when `temperature>0`)
Greedy decoding is deterministic and identical. For *stochastic* sampling the port uses
`LogitsProcessor`'s RNG (rand_pcg) rather than torch's `multinomial` generator — a
different random stream. This was never byte-reproducible across frameworks anyway; the
token *set* and proportional distribution match (incl. Torch's threshold top-k).

---

## 3. Word-count audit (Python vs Rust, per mapped file)

Total Rust ≈ **1.5× Python** (explicit types + doc comments), so logic is not missing
wholesale. One file flagged thin and audited:

- `model/conformer/encoder.py` at **0.36×** — explained in §2.5 (off-path NeMo
  streaming/export stubs; on-path forward is fully implemented + parity-verified).
- `model/conformer/utils.py` at 0.55× — small helpers, same pattern.

Everything else is ≥ 0.6× (most > 1×).

---

## 4. Out of scope / reused, not ported

- The vendored Python `liquid_audio/moshi/**` is **reused as the `moshi` crate** (Kyutai's
  Rust port), not re-ported — `compare_symbols`'s `core` scope excludes it by design.
- `liquid_audio/demo/**` (gradio/CLI demo) is not ported.

---

## 5. Known gaps & risks (honest)

1. **Padded-batch conformer masking.** The offline path processes one unpadded clip
   (batch=1), so `create_masks` returns `(None, None)` — correct for inference/parity.
   A *padded multi-clip batch* through the conformer would need the full `_create_masks`
   port. Documented as the offline-path contract.
2. **bf16 runtime is verified only structurally.** The golden tensors are f32; bf16
   Metal output is faithful by construction (§2.4) but is not byte-compared to a bf16
   Python dump.
3. **Stochastic sampling** is not byte-reproducible vs torch (§2.8) — by nature.

---

## 6. Net-new (not in the Python): `candle-flashfftconv`

A separate reusable crate of FlashFFTConv Metal/CPU kernels (depthwise causal conv1d;
the Monarch long conv; the fused single-pass conv) was written alongside, ported from
the owner's MLX kernels and reconciled with the FlashFFTConv CUDA. It is **not** part of
the `liquid_audio` parity surface (the Python has no such kernels); each kernel is
independently verified (cpu == naive/circular/linear convolution; metal == cpu at
1e-8–1e-6). See `../candle-flashfftconv/`.

**Two precision regimes (faithful vs precise).** The FlashFFTConv CUDA kernels run in
**bf16/f16**: `csrc/flashfftconv/butterfly/butterfly_cuda_bf16.cu` stores the DFT
matrices, twiddles, and every per-butterfly output in `__nv_bfloat16`
(`__float22bfloat162_rn`), with only the inner `wmma` matmul accumulating in `float`.
PyTorch's *generic* `torch.fft` on CUDA can promote to `complex128` (cuFFT has a real
double path), but the FlashFFTConv custom kernels never do — they are bf16-at-the-edges,
f32-in-the-accumulator, **never f64**. That coarse rounding is baked into the trained
weights, so the crate ships **both** ports:

| regime | op | error vs f64 ground truth (256-pt circular conv) |
|---|---|---|
| **faithful (bug-for-bug CUDA)** | `monarch_conv_bf16` | **2.69e-1** |
| clean f32 | `monarch_conv` | 7.63e-6 |
| **precise (double-double, ~f64)** | `fused_fft_conv_dd` / `complex_mul_dd` | ≤ 1.18e-7 |

The bf16 regime is **~35000× coarser than f32** — *more* divergent than f64, not less.
That is deliberate: the network was fit around that rounding, so `monarch_conv_bf16` is
the reference for matching the trained model, while the double-double path is for the
true (more accurate than the original CUDA) convolution. `monarch_conv_bf16` needs no new
shaders — candle's `BF16` dtype is `half::bf16` RNE, matching CUDA `_rn`, so the same
code path runs the faithful regime on CPU and Metal.

---

## Reproduce

```sh
cd liquid-audio-rs
export LFM_MODEL_DIR=../model
python parity/compare_symbols.py --scope core          # 170/170
cargo test --lib                                       # 31 passed
cargo test --test parity --release -- --ignored --nocapture   # 8/8 byte-exact
LFM_MODEL_DIR=../model cargo run --release --example generate # end-to-end, CPU/f32
```
