# model_mlp (Rust port)
**Source:** `liquid-audio/src/model/mlp.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/mlp.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/mlp.md`](../../../wiki/model/mlp.md). The original
> documents the Python `MLP(nn.Module)`; this documents the Rust `struct MLP`
> and where it diverges from the source.
>
> **As-built update (Claude's `Send` fix):** `MLP` now holds
> `Vec<Box<dyn Module + Send>>` instead of `candle_nn::Sequential` (whose
> `Vec<Box<dyn Module>>` is not `Send`), so `LFM2AudioModel: Send` and the model
> can move to a worker thread. The forward is a manual left fold; the
> `model.{idx}` weight paths are unchanged. See
> [`AS_BUILT_claude_changes.md`](../AS_BUILT_claude_changes.md) §3.

## Role
Identical purpose: a small, generic feed-forward stack that LFM2-Audio
instantiates once on the inference path as the **`audio_adapter`**
(`lfm2_audio.rs:87` analog). It projects the FastConformer encoder's
per-frame 512-dim acoustic features up to the 2048-dim backbone hidden size so
encoded microphone audio can be scattered into the same token-embedding stream
as text and audio-out embeddings. The only learnable glue between the audio
front-end and the language backbone.

## How it works (Rust)
`MLP::new(in_channels, out_channels, hidden_dim, bias, use_layer_norm, dropout, vb)`
(`mlp.rs:27`) builds a `candle_nn::Sequential` from the same channel list
`channels = [in_channels, *hidden_dim, out_channels]` (`mlp.rs:36-39`). For
the `audio_adapter` the call is `MLP(512, 2048, [2048], true, true, 0.0, vb)`
so `channels = [512, 2048, 2048]`.

Layer assembly, in order (mirroring `mlp.rs:41-66`):
1. **Input LayerNorm** (`mlp.rs:44-47`): when `use_layer_norm=true`, a
   `crate::model::norm::layer_norm(channels[0], 1e-5, vb.pp("model.0"))` is
   prepended. This is the Rust port's own `LayerNorm` (`norm.rs:16`), a thin
   wrapper around `candle_nn::ops::layer_norm_slow` — true LayerNorm (mean +
   variance + learnable `weight` *and* `bias`, `eps=1e-5`), *not* RMSNorm.
   `idx` advances to 1.
2. **Linear → GELU blocks** (`mlp.rs:49-66`): for each adjacent channel pair,
   `linear` or `linear_no_bias` (per the `bias` flag) is added at
   `vb.pp("model.{idx}")`, then `idx` advances. After every Linear *except the
   last* (`i != channels.len() - 2`, `mlp.rs:58`), `Activation::Gelu` is added
   and `idx` advances; if `dropout > 0.0`, `idx` advances **again without
   instantiating anything** — Dropout is identity at inference, so only the
   slot is reserved to keep the child-index alignment.

So the concrete `audio_adapter` graph is:
`LayerNorm(512) → Linear(512→2048) → GELU → Linear(2048→2048)`. Identical to
Python.

**Activation — exact-erf GELU.** `Activation::Gelu` in candle maps to
`gelu_erf` (the exact `0.5·x·(1+erf(x/√2))` form), *not* the tanh approximation.
This matches PyTorch's `nn.GELU()` default (`approximate='none'`). Documented
inline at `mlp.rs:18-20`.

**Forward** (`mlp.rs:73`) is `self.model.forward(x)` — no residual, no masking,
no streaming state. `impl Module for MLP` is the candle trait, so the adapter is
called as `self.audio_adapter.forward(&x)` wherever the prefill path needs it.

**Call-site mechanism** (`lfm2_audio.rs` prefill analog): audio-in clips are
split by `audio_in_lens`, run through the conformer, the padded encoder output
is un-padded to a flat concatenation of valid frames, the adapter runs on that
flat `(ΣT', 512)` tensor producing `(ΣT', 2048)`, and the result is
mask-scattered into the prefill buffer interleaved with `text_emb` and
`audio_out_emb` to form the `(B, L, 2048)` backbone input. Same as Python.

## Dtypes & shapes (Rust)

| Stage | dtype | shape |
|---|---|---|
| Input `audio_enc_concatenated` | `DType::F32` (CPU) or `BF16` (Metal/CUDA) | `(ΣT', 512)` |
| weights: LayerNorm `weight`/`bias` | model dtype (bf16 on disk) | `(512,)`, `(512,)` |
| LayerNorm forward (`ops::layer_norm_slow`) | stays in the input dtype | `(ΣT', 512)` |
| after LayerNorm | model dtype | `(ΣT', 512)` |
| `Linear(512→2048)` weight / bias | model dtype | `(2048,512)` / `(2048,)` |
| after Linear + `Activation::Gelu` | model dtype | `(ΣT', 2048)` |
| `Linear(2048→2048)` weight / bias | model dtype | `(2048,2048)` / `(2048,)` |
| **Output** `audio_in_emb` | model dtype | `(ΣT', 2048)` |

`ΣT'` = total post-subsampled conformer frames across all audio-in segments in
the turn.

## Wiring (Rust)
**Upstream**
- `model/conformer/encoder.rs` → emits `audio_enc` `(B, 512, T')`; the prefill
  path length-masks and flattens it to `(ΣT', 512)` before it enters this MLP.
  The `512` is `conformer._feat_out`, read at construction
  (`model/lfm2_audio.rs:87` analog). See
  [`glm-version/model/conformer/encoder.md`](conformer/encoder.md).

**Downstream**
- `model/lfm2_audio.rs` consumes the `(ΣT', 2048)` output directly inside
  `_prefill`: mask-scattered into `in_emb[audio_in_mask]` alongside `text_emb`
  and `audio_out_emb` to assemble the `(1, L, 2048)` backbone input. See
  [`glm-version/model/lfm2_audio.md`](lfm2_audio.md).

## Python ↔ Rust — where the port differs

| Python | Rust | Difference | Why |
|---|---|---|---|
| `MLP(nn.Module)` with `nn.Sequential` | `struct MLP { model: Sequential }` | **type** | `candle_nn::Sequential` is the direct analog; the `Module` trait impl replaces `__call__`. |
| `__init__(in, out, hidden_dim, *, bias=True, use_layer_norm=True, dropout=0.0)` | `MLP::new(in, out, hidden_dim, bias, dropout, vb)` — **all args explicit, no defaults** | **no default args** | Rust has no keyword/default args; the caller (`lfm2_audio.rs`) passes `true, true, 0.0` explicitly, matching the Python defaults. The `vb: VarBuilder` arg is the Rust weight-source (replaces Python's `self.*` parameter access via `state_dict`). |
| `nn.LayerNorm(channels[0])` (default `eps=1e-5`) | `crate::model::norm::layer_norm(channels[0], 1e-5, vb.pp("model.0"))` | **deliberate: custom LayerNorm** | `candle_nn::LayerNorm`'s `forward` takes a fused `apply_op*_no_bwd` path that **severs autograd** — fine for inference, zero gradient during training. The port's `norm.rs:16` wrapper calls `candle_nn::ops::layer_norm_slow` (the basic-op differentiable path), so the same code works for both inference and training. Same forward output; only the backward path differs. See `norm.rs` header comment. |
| `nn.Linear(in, out, bias=True/False)` | `candle_nn::linear` / `candle_nn::linear_no_bias` | **API split** | candle has one function per bias setting rather than a `bias=` kwarg. Functionally identical. |
| `nn.GELU()` (exact erf) | `Activation::Gelu` (→ `gelu_erf`) | **identical** | The comment at `mlp.rs:18-20` records the deliberate choice of erf over tanh. |
| `nn.Dropout(p)` (identity at inference) | **not instantiated**; only `idx += 1` to reserve the child slot | **deliberate omission** | Dropout is identity at inference; instantiating a no-op module would only add code. Reserving the index keeps the `model.{idx}` weight paths aligned with the checkpoint. |
| Weight paths via `nn.Sequential` child numbering | `vb.pp(format!("model.{idx}"))` with explicit `idx` counter | **manual index bookkeeping** | candle's `Sequential` doesn't expose child indexing the way `nn.Sequential` does; the port walks `idx` by hand (advancing past no-weight GELU/Dropout slots) to keep `model.0/1/3` aligned with the checkpoint. This is the easiest-to-miss correctness point — see §Gotchas. |
| `forward(self, x)` | `impl Module for MLP { fn forward(&self, x: &Tensor) -> Result<Tensor> }` | **&Tensor, Result** | candle passes tensors by reference and returns `Result<Tensor>`; PyTorch passes by value/ownership. The `?` propagation replaces Python's exception model. |
| runs on `device="cuda"`, `dtype=torch.bfloat16` (Python defaults) | device/dtype-agnostic — `vb` carries them | **deliberate: device-agnostic** | PYTHON_VS_RUST.md §2.1. No `.cuda()`; runs on `(Cpu, F32)` for parity, Metal/bf16 opt-in. |

## Precision / gotchas (Rust-specific)
- **The `norm.rs` LayerNorm is not `candle_nn::LayerNorm`.** The port uses its
  own `crate::model::norm::LayerNorm` (`norm.rs:16`) which calls
  `ops::layer_norm_slow`, *not* the fused `candle_nn::LayerNorm`. The fused path
  severs autograd — irrelevant for inference, fatal for training. The wrapper's
  forward output is identical; only the backward path differs. This is the
  single Rust-specific subtlety not present in the Python module (Python's
  `nn.LayerNorm` is always differentiable). See the `norm.rs` header comment
  for the verification (backbone/conformer params got no grad with the fused
  path; fixed with `_slow`).
- **`idx` bookkeeping is manual.** `mlp.rs:42` declares `let mut idx = 0usize;`
  and advances it on every added module *and* on the no-weight GELU/Dropout
  slots (`mlp.rs:60, 63`). An off-by-one here would silently load the wrong
  tensors from `model.{idx}`. The Python `nn.Sequential` handles this
  implicitly via its child registry; the Rust port reproduces it by hand. The
  unit test that exercises the adapter through the full prefill parity harness
  is the guard against this (PARITY.md prefill 1.1e-6).
- **`Activation::Gelu` vs `Activation::NewGelu`.** The two candle variants are
  easy to confuse: `Gelu` = erf (matches `nn.GELU()`), `NewGelu` = tanh
  approximation. The port uses `Gelu` and documents it inline. Picking `NewGelu`
  would be a silent parity regression.
- **No `dropout` instantiation.** The Python `nn.Dropout` is a module in the
  `Sequential`; the Rust port reserves its index slot but adds no module. If
  the upstream ever runs this with `dropout > 0.0` in *training* mode, the Rust
  port would need a real `Dropout` module added at that index. For inference
  (the only path this port exercises) the current behavior is correct.
- **`&Tensor` + `Result<Tensor>`.** Every forward takes `&Tensor` and returns
  `Result<Tensor>`; errors propagate with `?`. There is no Python-style
  exception — a matmul failure is an `Err(candle_core::Error)` the caller must
  handle. The `MLP::forward` body is one line (`self.model.forward(x)`) so the
  `?` is implicit in the `Module` trait's return type.

## Cross-references
- [`wiki/model/mlp.md`](../../../wiki/model/mlp.md) — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.4 (the RMSNorm
  bf16-ordering fix — lives in the backbone, *not* here; this module uses true
  LayerNorm).
- `liquid-audio/src/model/norm.rs` — the differentiable LayerNorm wrapper.
- `liquid-audio/parity/PARITY.md` — prefill modality-scatter 1.1e-6 (which
  exercises the adapter end-to-end).