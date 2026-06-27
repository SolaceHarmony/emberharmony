# model_mlp
**Code:** `MD03` · **Source:** `model/mlp.py` · **Rust:** `model/mlp.rs` · **On the LFM2-Audio inference path:** yes

## Role
A small, generic feed-forward stack (`MLP(nn.Module)`) that LFM2-Audio instantiates exactly once on the inference path as the **`audio_adapter`** (`lfm2_audio.py:87`). Its job is a dimension/representation bridge: it projects the FastConformer encoder's per-frame acoustic features (`feat_out` = 512) up to the LFM2 backbone hidden size (`hidden_size` = 2048) so that encoded microphone audio can be scattered into the same token-embedding stream as text and audio-out embeddings. It is the only learnable glue between the audio front-end and the language backbone; without it the 512-dim conformer output and the 2048-dim backbone embedding space are incompatible.

## How it works
`MLP.__init__` (`mlp.py:6-37`) builds a plain `nn.Sequential` from a channel list `channels = [in_channels, *hidden_dim, out_channels]` (`mlp.py:17`). For the `audio_adapter` the call is `MLP(512, 2048, [2048])`, so `channels = [512, 2048, 2048]`. The default flags (`bias=True`, `use_layer_norm=True`, `dropout=0.0`) apply — the call site passes none of them.

Layer assembly, in order:
1. **Input LayerNorm** (`mlp.py:20-21`): because `use_layer_norm=True`, a `nn.LayerNorm(channels[0])` = `LayerNorm(512)` is prepended. This is a true **LayerNorm** (mean-subtract + variance-normalize over the last dim, learnable `weight` *and* `bias`, default `eps=1e-5`), *not* RMSNorm — it lands on the **input** (512-dim conformer features), not between hidden layers. PyTorch computes the normalization statistics by upcasting to f32 internally, then applies `weight*x_hat + bias`.
2. **Linear → GELU** blocks (`mlp.py:23-35`): for each adjacent channel pair it appends `nn.Linear(channels[i], channels[i+1], bias=True)`. After every Linear *except the last* (`i != len(channels)-2`, `mlp.py:32`) it appends `nn.GELU()` and, only if `dropout>0`, a `nn.Dropout`. With `channels=[512,2048,2048]` this yields: `Linear(512→2048)`, `GELU`, `Linear(2048→2048)`. There is **no** activation after the final Linear (clean linear projection into backbone space), and **no** dropout (inference and `dropout=0.0`).

So the concrete `audio_adapter` graph is:
`LayerNorm(512) → Linear(512→2048) → GELU → Linear(2048→2048)`.

**Activation — exact GELU.** `nn.GELU()` defaults to `approximate='none'`, i.e. the **exact erf form** `0.5·x·(1+erf(x/√2))`, *not* the tanh approximation. This is load-bearing for parity (see Rust mapping).

**Forward** (`mlp.py:39-40`) is a single `self.model(x)` — no residual, no masking, no streaming state. It is applied pointwise over the frame axis.

**Call-site mechanism** (`lfm2_audio.py:339-355`, in `_prefill`): audio-in clips are split by `audio_in_lens`, right-padded into a batch, and run through the conformer (`audio_enc, audio_in_len = self.conformer(...)`, `lfm2_audio.py:346`). The padded encoder output is then **un-padded back to a flat concatenation** of valid frames via a length mask (`audio_enc_concatenated = audio_enc.mT[len_mask]`, `lfm2_audio.py:350`) — shape `(ΣT', 512)`. The adapter runs on that flat tensor: `audio_in_emb = self.audio_adapter(audio_enc_concatenated)` (`lfm2_audio.py:353`), producing `(ΣT', 2048)`. An `assert` (`lfm2_audio.py:355`) checks the row count equals the number of `AUDIO_IN` slots in `modality_flag`. The result is then scattered into the prefill buffer by boolean mask: `in_emb[audio_in_mask] = audio_in_emb` (`lfm2_audio.py:369`), interleaved with `text_emb` and `audio_out_emb` to form the `(B,L,2048)` backbone input.

No sampling, attention, RoPE, convolution, or quantization lives here — this component is purely `LayerNorm + 2×Linear + 1×GELU`.

## Dtypes & shapes

| Stage | dtype | shape |
|---|---|---|
| Input `audio_enc_concatenated` (flat valid frames) | model dtype (bf16 GPU / f32 CPU) | `(ΣT', 512)` |
| weights: LayerNorm `weight`/`bias` | model dtype (bf16 on disk) | `(512,)`, `(512,)` |
| LayerNorm internal stats (mean/var) | **f32 upcast** (PyTorch LayerNorm autocasts stats), eps=1e-5 | `(ΣT', 512)` |
| after LayerNorm | model dtype | `(ΣT', 512)` |
| `Linear(512→2048)` weight / bias | model dtype | `(2048,512)` / `(2048,)` |
| after Linear+GELU | model dtype | `(ΣT', 2048)` |
| `Linear(2048→2048)` weight / bias | model dtype | `(2048,2048)` / `(2048,)` |
| **Output** `audio_in_emb` | model dtype | `(ΣT', 2048)` |

`ΣT'` = total post-subsampled conformer frames across all audio-in segments in the turn (conformer subsamples mel by 8×). Token ids do not appear here; this is a dense feature→embedding map.

## Wiring
**Upstream**
- [conformer_encoder](conformer/encoder.md) → emits `audio_enc` `(B, 512, T')` in model dtype; `_prefill` length-masks and flattens it to `(ΣT', 512)` (model dtype) before it enters this MLP. The `512` is exactly `conformer._feat_out`, read at construction (`lfm2_audio.py:87`).

**Downstream**
- [model_lfm2_audio](lfm2_audio.md) — consumes the `(ΣT', 2048)` output directly inside `_prefill`: it is mask-scattered into `in_emb[audio_in_mask]` (`lfm2_audio.py:369`), alongside `text_emb` and `audio_out_emb`, to assemble the `(1, L, 2048)` backbone input. That assembled tensor then feeds [model_lfm2_backbone](lfm2_audio.md) (HF `Lfm2Model`). The MLP's output is a *contribution* to the backbone input embedding, not a standalone tensor passed further on its own.

## Python ↔ Rust
Symbol map: `MLP(nn.Module)` → `struct MLP` (`mlp.rs:21`); `__init__` → `MLP::new` (`mlp.rs:27`); `forward` → `impl Module::forward` (`mlp.rs:72-76`). The Rust builds the identical `candle_nn::seq()` chain over `channels = [in, *hidden_dim, out]` and **mirrors the `nn.Sequential` child indices in the weight paths** (`"model.{idx}"`, `mlp.rs:42-64`) so a trained checkpoint loads 1:1 — including advancing `idx` over the no-weight GELU slot, and (when `dropout>0`) reserving the Dropout index without instantiating it (`mlp.rs:61-64`), since Dropout is identity at inference.

Deliberate, parity-preserving choices (per PYTHON_VS_RUST.md):
- **Exact-erf GELU.** PyTorch `nn.GELU()` = erf form; candle `Activation::Gelu` maps to `gelu_erf` (exact), explicitly *not* the tanh approx (`NewGelu`/`GeluPytorchTanh`) — documented inline at `mlp.rs:18-20`. This is the right match.
- **LayerNorm via `ops::layer_norm_slow`.** `mlp.rs` calls `crate::model::norm::layer_norm(dim, 1e-5, …)` (`mlp.rs:45`), which wraps `candle_nn::ops::layer_norm_slow` (`norm.rs:30`) with learnable weight+bias and `eps=1e-5` — matching `nn.LayerNorm`'s default eps. (This is the standard mean-var LayerNorm, distinct from the bf16-order-sensitive **RMSNorm** repaired in PYTHON_VS_RUST.md §2.4, which lives in the backbone/depthformer, not here.)
- **Device/dtype-agnostic** (PYTHON_VS_RUST.md §2.1): no `.cuda()`; runs on `Device::Cpu`/`F32` for parity, Metal/bf16 opt-in. The MLP itself is plain matmuls + a norm, so it inherits the cross-library f32 floor with no component-specific divergence.

This is one of the cleanest 1:1 ports — pure `candle_nn` primitives, no custom kernel, no hand-rolled SDPA.

## Precision / gotchas
- **LayerNorm, not RMSNorm.** Easy to misread given the rest of the model is RMSNorm-heavy. This adapter uses true `nn.LayerNorm` (subtract mean, divide by std, **weight *and* bias**), eps=1e-5, on the **input** only. The RMSNorm "weight-multiply-in-f32-then-cast" bf16 ordering gotcha (PYTHON_VS_RUST.md §2.4) does **not** apply here.
- **f32-upcast for normalization stats.** PyTorch's `LayerNorm` computes mean/variance in f32 even under a bf16 module; the candle `layer_norm_slow` path follows the same normalize-in-higher-precision pattern. At bf16 this matters for the front of the adapter; output Linear/GELU are plain model-dtype matmuls (the ~1e-6 cross-library gemm floor, PYTHON_VS_RUST.md §1.4).
- **No activation / no dropout after the last Linear** (`mlp.py:32`). The output is a raw linear projection into backbone embedding space — adding a trailing GELU would be a correctness bug. Dropout is configured-out (`dropout=0.0`) and is identity at inference regardless.
- **Weight-path index alignment.** Because the Rust reproduces the exact `nn.Sequential` child numbering (LayerNorm=`model.0`, Linear=`model.1`, GELU=`model.2` (no weights), Linear=`model.3`), an off-by-one in the index bookkeeping would silently load the wrong tensors. The Rust deliberately advances `idx` past the activation/dropout slots to keep `model.{idx}` aligned (`mlp.rs:42-64`).
- **Batch contract.** The adapter sees a **flat `(ΣT', 512)`** tensor (already un-padded by `len_mask` at `lfm2_audio.py:350`), so padding frames never reach it — the row-count `assert` (`lfm2_audio.py:355`) guards that the conformer's valid-frame count matches the `AUDIO_IN` modality slots. (Padded multi-clip masking is an upstream conformer concern, PYTHON_VS_RUST.md §5.1, not this component's.)
