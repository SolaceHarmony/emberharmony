<!-- topic: Moshi Utilities -->
# MU04 · int8 quantize helpers
**Code:** `MU04` · **Source:** `moshi/utils/quantize.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
A 58-line bitsandbytes (`bnb`) int8 weight-quantization helper vendored from Kyutai's Moshi. It defines `QLinear` — a drop-in replacement for `nn.Linear` whose weight is stored as an int8 vector-wise-quantized matrix plus an fp32 per-row scale — and `replace_linear_with_qlinear`, which recursively swaps every `nn.Linear` in a module tree for a `QLinear`. It exists to shrink the **Moshi 7B LM** (`lm.py`) and the Moshi codec/projection `StreamingTransformer` (`transformer.py`) for low-VRAM inference. It is **off the LFM2-Audio path**: LFM2-Audio uses its own HF `Lfm2Model` backbone + depthformer (full-precision bf16), never the Moshi LM, and both call sites default `quantize=False`.

## How it works
Two symbols, both thin wrappers around bitsandbytes' LLM.int8() (`int8_vectorwise_quant` + `MatmulLtState`) machinery.

**`QLinear.__init__(linear)`** (`quantize.py:13-22`) — bias-free, float-only constraint:
- Asserts the source weight is floating point (`:18`) and that there is **no bias** (`:19`); `QLinear.forward` never adds a bias, so a biased `Linear` would silently lose it — the assert makes that a hard error.
- `CB, SCB, _ = bnbF.int8_vectorwise_quant(weight.data.to(torch.float16))` (`:20`). The weight is cast bf16/f32 → **fp16** first, then quantized **per output row (vector-wise)**: each row of the `[out, in]` weight is scaled by its own absmax so the int8 range `[-127,127]` spans that row. `CB` = int8 quantized weight `[out,in]`; `SCB` = fp32 scale, one entry per output row `[out]` (`SCB = row_absmax / 127`); the discarded third return is the per-row outlier mask / `outlier_cols` unused here.
- Stores `CB` as `self.weight` and `SCB` as `self.weight_scb`, both `requires_grad=False` (`:21-22`). The `_scb` suffix is the on-disk convention the Moshi `transformer.py` state-dict loader keys off (`transformer.py:422` "_scb suffix is for quantized data", `:443` `isinstance(in_proj, QLinear)`).

**`QLinear.forward(x)`** (`quantize.py:24-40`) — LLM.int8() mixed-precision matmul:
- Builds a fresh `bnb.MatmulLtState()` each call (`:26`), attaches `CB`/`SCB` (`:27-29`), and **guards the scale dtype**: if `SCB.dtype != torch.float` it raises (`:31-36`) — the named failure mode is a `.to(bfloat16)` on the whole model having dragged the scale to bf16 and destroyed precision. `has_fp16_weights=False` (`:37`) tells bnb the weights live in int8, not fp16.
- `y = bnb.matmul(x.half(), state.CB, state=state)` (`:38`): activations are cast to **fp16**, multiplied against the int8 weight via bnb's LLM.int8() kernel (int8×int8 GEMM in the inlier path, fp16 outlier-column fallback), then dequantized with `SCB`. Output `y` is fp16. There is **no causal mask, no norm, no activation** here — it is purely a quantized linear.

**`replace_linear_with_qlinear(module)`** (`quantize.py:43-57`) — recursive in-place swap:
- For each named child: if it is `nn.Linear`, replace it with `QLinear(child)` (`:46-47`). If it is **already** a `QLinear`, call `child.float()` (`:48-55`) to re-pin the fp32 scale — the long comment explains the LM calls this twice (once layer-by-layer to cap peak memory, once after full init) and a global `.bfloat16()` between the passes would have cast `weight_scb` to bf16; `.float()` undoes that. Otherwise recurse (`:56-57`). Must run **before** `load_state_dict` so the quantized param shapes/keys match the checkpoint.

Note this is a **weight-only int8** scheme (W8A16-ish: int8 weights, fp16 activations), distinct from the codec's **vector-quantization** (`core_vq`/`moshi_vq`, RVQ codebooks) — same word "quantize", unrelated mechanism.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `QLinear.__init__` weight | `Linear.weight` bf16/f32 `[out,in]` | `weight` int8 `[out,in]` + `weight_scb` fp32 `[out]` (after fp16 cast in `int8_vectorwise_quant`) |
| `QLinear.forward` | `x` any float `[*, in]` → cast `.half()` fp16 | `y` fp16 `[*, out]` (int8 weight × fp16 act, dequant by fp32 SCB) |
| `replace_linear_with_qlinear` | `nn.Module` tree | same tree, `nn.Linear` → `QLinear` in place (no tensor returned) |

Internal dtype facts: weight quant path is **f→fp16→int8** (`:20`); scale `SCB` is **fp32 and must stay fp32** (`:31-36`, re-pinned by `.float()` at `:55`); activations are forced **fp16** at matmul (`:38`); output is **fp16**. No bf16 path inside (bnb int8 kernels are fp16-coupled), no f64, no int64.

## Wiring
Off the LFM2-Audio tensor path; both edges live entirely inside the vendored Moshi LM stack and are dormant by default.

**Upstream (importers / callers):**
- `replace_linear_with_qlinear` imported by [moshi LM](../models/lm.py) (`lm.py:24`) and called at `lm.py:237` only when `LMModel(quantize=True)` — i.e. it walks the Moshi 7B `lm.py` module tree (full LM `nn.Linear`s) and swaps them. Edge: `nn.Module` tree (bf16/f32 Linear weights) → in-place QLinear.
- `replace_linear_with_qlinear` + `quantize` module imported by [moshi transformer](../modules/transformer.py) (`transformer.py:20-21`); called at `transformer.py:862` per-layer when `StreamingTransformerLayer(quantize=True)`; the in/out-proj state-dict loader special-cases `QLinear` at `transformer.py:443`. Edge: per-layer `nn.Linear` (the attention/FF projections) → QLinear.

**Downstream (consumers of QLinear output):** the dequantized fp16 activations feed straight back into the surrounding [moshi LM](../models/lm.py) / [moshi transformer](../modules/transformer.py) forward (attention, gated FFN, output heads) exactly where the original `Linear` output went — `QLinear` is API-transparent. No new tensor leaves the module.

No LFM2-Audio component ([core_processor](CO01-Processor-ChatState), [model_lfm2_audio](MD01-LFM2AudioModel), [model_transformer](MD04-Depthformer), [moshi_compression](MM01-Mimi-Codec)) imports or instantiates this; the codec and backbone run full-precision.

## Python ↔ Rust
**No Rust port (`Rust: -`).** Confirmed by grep: no `QLinear`/`int8_vectorwise`/`MatmulLtState`/`weight_scb` symbol anywhere under `liquid-audio-rs/src/`. This is a **deliberate omission**, consistent with two reference-doc choices:
- **PYTHON_VS_RUST.md §4 "Out of scope / reused, not ported":** the vendored Python `moshi/**` is reused as Kyutai's `moshi` crate, not re-ported, and `compare_symbols.py --scope core` **excludes** `moshi/` by design (so this file is not part of the 170/170 surface).
- **PYTHON_VS_RUST.md §2.3 "Upstream reuse":** the only Moshi piece the port actually exercises is `moshi::mimi` (the codec). The Moshi **LM** and its `quantize=True` finetune/inference knob are off-path (cf. the off-path `moshi_lora`, `moshi_lm`, `moshi_tts`), so its int8 helper has no call site to satisfy.

Had it been ported, the candle analog would be candle-transformers/candle-nn int8 GEMM (or a custom QMatMul over a stored int8 `Tensor` + f32 scale), since bitsandbytes is a CUDA-only C/PTX library with no candle equivalent — the same "CUDA-only kernel → portable candle op" pattern as PYTHON_VS_RUST.md §2.2, but here simply skipped because nothing on-path needs it.

## Precision / gotchas
- **CUDA-only dependency.** `bitsandbytes` int8 matmul requires an NVIDIA GPU; this module cannot run on CPU or Metal at all. It is one more reason it is absent from the device-agnostic Rust port (PYTHON_VS_RUST.md §2.1: Rust delivers the CPU/Metal design point, Python is CUDA-coupled).
- **The fp32-scale trap is real and load-bearing.** `weight_scb` must stay fp32. A global `model.bfloat16()`/`model.half()` after construction silently casts the scale, and the only thing catching it is the `forward`-time `RuntimeError` (`:31-36`) and the `.float()` re-pin in the double-pass swap (`:55`). The comment at `:49-54` is the canonical warning: quantize → set dtype in the wrong order = precision loss before the state dict even loads.
- **Bias is forbidden** (`:19`): `QLinear` drops bias entirely; swapping a biased `Linear` asserts rather than miscomputing.
- **Lossy by construction.** Per-row int8 (127 levels) + fp16 activations is far coarser than the bf16/f32 the rest of the stack uses; it trades accuracy for VRAM and is only ever opt-in (`quantize=False` default at both `lm.py:106` and `transformer.py:823`).
- **Naming collision.** This "quantize" is **weight int8**, unrelated to the RVQ/codebook "quantize" in [moshi_vq](QZ01-Split-RVQ)/[moshi_core_vq](QZ02-VQ-Core) and the `quantize=` flag on the codec's `set_num_codebooks` — do not conflate them. No EOAudio/special-token or off-by-one concerns apply here (no sequence logic in this file).
