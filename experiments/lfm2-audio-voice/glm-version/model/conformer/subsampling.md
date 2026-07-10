# conformer_subsampling (Rust port)
**Source:** `liquid-audio/src/model/conformer/subsampling.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/subsampling.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/conformer/subsampling.md`](../../../wiki/model/conformer/subsampling.md).

## Role
`ConvSubsampling` (`subsampling.rs:222`) is the FastConformer **pre-encoder**
(`self.pre_encode`): it consumes the 128-bin log-mel front-end and downsamples
it **8×** along the time axis with a stack of strided Conv2d layers, then
flattens channels×frequency and projects to the encoder hidden size
`d_model=512`. It exists so the heavy 17-layer ConformerLayer stack runs at
~12.5 Hz frame rate (one frame per 8 mel hops) instead of the raw 100 Hz mel
rate, cutting attention cost ~64×. The Rust port implements the `dw_striding`
scheme (NVIDIA NeMo, vendored verbatim); the other four schemes
(`vggnet`/`striding`/`striding_conv1d`/`dw_striding_conv1d`) are present in
`new_scheme` (`:265`) for inventory completeness but unused by LFM2-Audio.

## How it works (Rust)
**Config that the model instantiates** (`encoder.rs`, from `config.json`):
`subsampling="dw_striding"`, `subsampling_factor=8` ⇒
`sampling_num = log2(8) = 3` (`:274`), `feat_in=128`, `feat_out=d_model=512`,
`conv_channels=256`, `is_causal=false`. Stride/kernel for `dw_striding`:
`_stride=2`, `_kernel_size=3`, `_ceil_mode=false`; non-causal ⇒
`_left_padding=_right_padding=(3-1)//2=1`.

**Layer construction** (`subsampling.rs:265` `new_scheme`). The conv stack is
held in a `MaskedConvSequential` (`:111`) — a `Vec<Op>` where `Op` is an enum
(`:98-107`) of `Conv(Conv2d)`/`Conv1d`/`CausalConv1d`/`MaxPool2dCeil`/`Relu`.
The closure `next(&mut idx)` (`:279`) increments the `vb` prefix for **every**
pushed module incl. ReLU, which is what makes the state-dict keys
`conv.{i}.weight` line up with Python exactly.
- **Stem (layer 1):** full `Conv2d(in=1, out=256, k=3, stride=2, pad=1)` + `ReLU`
  — the only conv that mixes the single input channel into 256, and the first
  2× downsample on **both** time and freq axes.
- **2 depthwise/pointwise blocks** (loop `range(sampling_num-1)=2`), each:
  `Conv2d(256→256, k=3, stride=2, pad=1, groups=256)` (depthwise, the 2×
  downsample) → `Conv2d(256→256, k=1, stride=1, pad=0, groups=1)` (pointwise
  channel mix) → `ReLU`.

Three stride-2 convs total ⇒ time **8×** and freq **8×** downsample. The input
is unsqueezed to `(B,1,T,F)` (`:185`), treated as a 1-channel image with T=time,
F=mel-freq.

**Frequency-axis size & the `out` Linear** (`subsampling.rs:406-409`).
`calc_length` (`:17`) is applied to `feat_in=128` with `all_paddings=2`, `k=3`,
`s=2`, `ceil_mode=false`, `repeat_num=3`:
```
L_{i+1} = floor((L_i + (all_paddings - kernel)) / stride + 1)
128 -> 64 -> 32 -> 16
```
so post-conv `F'=16`. The flatten is `C·F' = 256·16 = 4096`, and
`self.out = linear(4096, 512, vb.pp("out"))`. `conv2d_subsampling=true` (the
conv2d schemes set this; conv1d schemes set it false and have `out=None`).

**`forward`** (`subsampling.rs:443`). For `dw_striding` (conv2d): unsqueeze
channel, run `forward_conv` (the no-mask single-clip path, `:136`), then
`transpose(1,2).reshape(b, t, c*f)` and the `out` Linear. Output is
`(x:(B,T',512), lengths:(B,) int)`. The per-clip `out_lengths` helper (`:433`)
runs the same `calc_length` recurrence on the *time* lengths.

**`MaskedConvSequential`** (`subsampling.rs:111`). Two forward paths:
- `forward_conv` (`:136`) — the no-mask path used by the single-clip offline
  path. For one clip the length-mask is all-ones, so this is numerically
  identical to the masked path (parity 5.6e-7). This is the path the offline
  LFM2-Audio inference uses.
- `forward` (`:184`) — the general masked path: `apply_channel_mask` before
  each layer, and **only for layers whose `stride != 1`** (read from the conv's
  own `config()`, `:196`) update `cur` lengths via
  `calculate_conv_output_size` (`:36`) and rebuild the mask. The interleaved
  pointwise convs (`k=1, s=1`) leave length unchanged. Reading the per-conv
  `config()` is what keeps the pointwise convs from shrinking the length.

**No norm / no attention / no RoPE here.** This module is pure conv + ReLU +
one Linear. Training-only `reset_parameters` is omitted; weights come from `VarBuilder`.

## Dtypes & shapes (Rust)
| stage | dtype | shape |
|---|---|---|
| input `x` (mel, post-`mT`) | model dtype (f32 CPU / bf16 Metal) | `(B, T, 128)` — caller passes mel as `(B,128,T)` and the encoder transposes to `(B,T,128)` before `pre_encode` |
| input `lengths` | — | `&[usize]` (masked path) / computed via `calc_length` (no-mask path) |
| internal conv image | model dtype | `(B,1,T,128)` → stem `(B,256, ⌈T/2⌉, 64)` → `(B,256, ⌈T/4⌉, 32)` → `(B,256, T', 16)` |
| flatten before `out` | model dtype | `(B, T', 256·16=4096)` |
| output `x` | model dtype | `(B, T', 512)` |
| output `lengths` | — | `Vec<usize>`, each `= calc_length(len, repeat=3)` |

Internal promotions: `calc_length`/`calculate_conv_output_size` run in **f64**
(`:17-25`) then cast to `usize`/`i64`. No softmax/norm here, so no f32 upcast
inside the convs — they run entirely in model dtype. The mel input was
computed in f32/f64 upstream and cast to `text_emb.dtype` before the conformer
(`lfm2_audio.rs:682`). Parity vs Python: conv-stack out **5.611e-7** at
`[1,256,13,16]`, post-subsample/pos-enc **1.019e-6** at `[1,13,512]`.

## Wiring (Rust)
**Upstream** — `model/conformer/processor.rs` produces the mel features
`(B,128,T)`. They enter via `model/conformer/encoder.rs`, which transposes to
`(B,T,128)` and calls `pre_encode(x, lengths)`. The caller path is
`model/lfm2_audio.rs::prefill_inputs`: it splits per-clip, casts each segment
to model dtype, and hands it to `self.conformer.forward(&seg)`
(`lfm2_audio.rs:683`). See [`glm-version/processor.md`](processor.md) and
[`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).

**Downstream** — output `(B,T',512)` flows back into `encoder.rs`, which adds
`RelPositionalEncoding` and runs the 17 `ConformerLayer`s
([`glm-version/model/conformer/modules.md`](modules.md),
[`glm-version/model/conformer/mha.md`](mha.md)), returning `(B,512,T')`. The
encoder result is concatenated and fed to the `audio_adapter`
[`glm-version/model/mlp.md`](model/mlp.md) (`MLP(512 → 2048)`, GELU-erf), whose
output `audio_in_emb=(ΣT', 2048)` is scattered into the LFM2 backbone token
stream. Net: `subsampling → encoder layers → audio_adapter → backbone`.

## Python ↔ Rust — where the port differs

| Python (`subsampling.py`) | Rust (`subsampling.rs`) | Difference | Why |
|---|---|---|---|
| `ConvSubsampling.__init__` (all 5 schemes) | `ConvSubsampling::new_scheme` (all 5 schemes, `:265`) + `new` thin `dw_striding` wrapper (`:250`) | identical (1:1) | — |
| `nn.Sequential` of conv/relu | `MaskedConvSequential { layers: Vec<Op> }` where `Op` is an enum | **deliberate: enum dispatch** | Rust has no `nn.Sequential`; the `Vec<Op>` + enum-match is the idiomatic analog. The `Op` enum covers `Conv2d`/`Conv1d`/`CausalConv1d`/`MaxPool2dCeil`/`Relu`. |
| `forward(x, lengths)` masked path | `forward(x, lengths)` masked path (`:184`) + `forward_conv` no-mask path (`:136`) | **deliberate: single-clip fast path** | for one clip the length-mask is all-ones, so `forward_conv` (no mask) is numerically identical to the masked path (parity 5.6e-7). The offline path uses `forward_conv`; the masked path is kept for the padded-batch case. |
| `calc_length` in `torch.float` then cast to int | `calc_length` in `f64` (`:17-25`) | **deliberate: f64** | Rust does the recurrence in `f64` (no `torch.float`); `f64` is strictly more precise, and the `floor` is exact either way. |
| `apply_channel_mask` broadcasts `(B,T,F)` over channels | `apply_channel_mask` (`:29`): `mask.unsqueeze(1)?.broadcast_as((b,c,t,f))?` then `broadcast_mul` | identical | — |
| `calculate_conv_output_size` | `calculate_conv_output_size` (`:36`) | identical | — |
| `reset_parameters` (NeMo uniform init, training only) | omitted | **deliberate omission** | weights come from `VarBuilder`; init is dead at inference. |
| pytorch#80020 batch/channel tiling (`conv_split_by_batch`/`conv_split_by_channel`/`channel_chunked_conv`) | return the plain un-tiled conv (`:495-519`) | **deliberate** | candle has no 2³¹ limit; the tiling workarounds are unnecessary. Output-equal. |
| `CausalConv2D` (imported-but-undefined upstream) | `causal_conv2d_unsupported` error (`:92`) | **deliberate: error loudly** | NeMo's `CausalConv2D` has no source to port; the Rust errors rather than guessing. LFM2 uses `is_causal=false`, so this never triggers. |
| `MaxPool2d(ceil_mode=True)` (vggnet) | `ceil_pool2d` (`:74`): edge-replicate odd dim then floor-pool | **deliberate** | candle's `max_pool2d` is floor-mode only; the edge-replication reproduces torch's ceil mode bit-identically for `k=s=2`. Off the LFM2 path. |
| `nn.Conv2d` stride>1 backward | `pad_even_hw`/`pad_even_1d` (`:45-67`) before strided convs | **deliberate: candle workaround** | candle's conv2d stride>1 backward mis-sizes the grad-input for odd spatial dims (T=101→51 ambiguity). Padding odd H/W to even with a trailing zero is **forward-identical** (the appended zero is the same one `padding` adds) and only fixes the backward. Stride-1 convs are skipped. |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. f32 on CPU (no candle CPU bf16 matmul); bf16 on Metal. |

## Precision / gotchas (Rust-specific)
- **Length math is `f64`-then-floor.** `calc_length` (`:17`) does
  `floor((L + (paddings-kernel))/stride + 1)` in `f64`, repeated
  `sampling_num` times. For the 128-bin freq axis: `128→64→32→16` exactly; for
  time it must be computed **per layer** in the masked path (`:200-203`) —
  collapsing it to a single uniform formula would over-shrink length by the
  pointwise steps.
- **The `out` Linear width is config-derived, not hardcoded.**
  `linear(calc_length(feat_in, …) * conv_channels, feat_out, …)` (`:406-409`)
  = `linear(4096, 512)`. Any change to mel bins or `conv_channels` resizes
  this matrix; the Rust recomputes it identically from `feat_in`/`ceil_mode`.
- **`pad_even_hw`/`pad_even_1d` is forward-identical.** The appended zero is the
  same zero `padding=1` adds; the output count is unchanged (measured 0.0
  forward diff). It only fixes the (otherwise-failing) conv2d stride>1 backward
  — a candle bug, not a faithfulness change. Stride-1 convs are skipped
  (`:151-155`) because padding would change their output length.
- **`forward_conv` vs `forward`.** The offline path uses `forward_conv` (no
  mask); the masked path is kept for padded-batch. For one clip they're
  numerically identical (parity 5.6e-7). The prefill parity test exercises
  `forward_conv`.
- **`Op` enum dispatch.** The `Vec<Op>` + `match` is the idiomatic Rust analog
  of `nn.Sequential`. An off-by-one in the `next(idx)` closure (`:279`) would
  mis-align the `conv.{i}` weight paths — the closure increments for every
  pushed module incl. ReLU, matching Python.
- **Causal conv2d errors loudly.** `causal_conv2d_unsupported` (`:92`) returns
  an `Err` rather than silently degrading. LFM2 uses `is_causal=false`, so this
  never triggers on the live path.
- **`ceil_pool2d` only implements `k=s=2`.** The `debug_assert_eq!((kernel,
  stride), (2, 2))` (`:75`) pins this — it's the only case the vggnet scheme
  uses. Off the LFM2 path.
- **No special tokens / EOAudio here.** This is a feature extractor, not a
  token producer — codes/EOAudio (2048) live in the Mimi/depthformer audio-out
  path, not in audio-in subsampling.
- **Cross-library f32 floor.** On Rust CPU there is no bf16 matmul, so this
  module computes in f32 even though weights are bf16 on disk (Metal stays bf16).
  The conv-subsample parity (5.6e-7) is well under the mel front-end's
  FFT-library floor (9.31e-6), so subsampling is not the precision bottleneck.

## Cross-references
- [`wiki/model/conformer/subsampling.md`](../../../wiki/model/conformer/subsampling.md)
  — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.2 (kernel-free
  convs), §2.5 (off-path schemes).
- `liquid-audio/parity/PARITY.md` — conv-stack out 5.611e-7, post-subsample
  1.019e-6.
