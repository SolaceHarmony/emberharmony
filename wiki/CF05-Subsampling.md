<!-- topic: Conformer Encoder -->
# CF05 · ConvSubsampling (dw_striding 8x)
**Code:** `CF05` · **Source:** `model/conformer/subsampling.py` · **Rust:** `model/conformer/subsampling.rs` · **On the LFM2-Audio inference path:** yes

## Role
`ConvSubsampling` is the FastConformer **pre-encoder** (`self.pre_encode`): it consumes the 128-bin log-mel front-end and downsamples it **8×** along the time axis with a stack of strided Conv2d layers, then flattens channels×frequency and projects to the encoder hidden size `d_model=512`. It exists so the heavy `N=17` ConformerLayer stack runs at ~12.5 Hz frame rate (one frame per 8 mel hops) instead of the raw 100 Hz mel rate, cutting attention cost ~64×. The model uses the `dw_striding` scheme (NVIDIA NeMo, vendored verbatim); the other four schemes (`vggnet`/`striding`/`striding_conv1d`/`dw_striding_conv1d`) are present in the class but unused by LFM2-Audio.

## How it works
**Config that the model instantiates** (`encoder.py:324`, from `config.json`): `subsampling="dw_striding"`, `subsampling_factor=8` ⇒ `_sampling_num = log2(8) = 3` (`subsampling.py:62`), `feat_in=128` (mel bins), `feat_out=d_model=512`, `conv_channels=256`, `activation=nn.ReLU(True)`, `is_causal=False`. Stride/kernel for `dw_striding` (`subsampling.py:108-120`): `_stride=2`, `_kernel_size=3`, `_ceil_mode=False`; non-causal ⇒ `_left_padding=_right_padding=(3-1)//2=1`.

**Layer construction** (`subsampling.py:122-181`). The conv stack is a flat `nn.Sequential` (`MaskedConvSequential`), and the Sequential index advances for **every** appended module incl. ReLU, which is what makes the state-dict keys `conv.{i}.weight` line up:
- **Stem (layer 1):** full `Conv2d(in=1, out=256, k=3, stride=2, pad=1)` + `ReLU` — the only conv that mixes the single input channel into 256, and the first 2× downsample on **both** time and freq axes.
- **2 depthwise/pointwise blocks** (loop `range(_sampling_num-1)=2`), each: `Conv2d(256→256, k=3, stride=2, pad=1, groups=256)` (depthwise, the 2× downsample) → `Conv2d(256→256, k=1, stride=1, pad=0, groups=1)` (pointwise channel mix) → `ReLU`.

So three stride-2 convs total ⇒ time **8×** and freq **8×** downsample. The input is unsqueezed to `(B,1,T,F)` (`subsampling.py:561`), treated as a 1-channel image with T=time, F=mel-freq.

**Frequency-axis size & the `out` Linear** (`subsampling.py:324-336`). `calc_length` (`subsampling.py:545`) is applied to `feat_in=128` with `all_paddings=2`, `k=3`, `s=2`, `ceil_mode=False`, `repeat_num=3`:
```
L_{i+1} = floor((L_i + (all_paddings - kernel)) / stride + 1)  ; here floor((L+2-3)/2 + 1)
128 -> 64 -> 32 -> 16
```
so post-conv `F'=16`. The flatten is `C·F' = 256·16 = 4096`, and `self.out = nn.Linear(4096, 512)`. `conv2d_subsampling=True` (the conv2d schemes set this; conv1d schemes set it False and have `out=None`).

**forward** (`subsampling.py:351`). `out_lengths = calc_length(lengths, …, repeat_num=3)` precomputes the per-clip output frame count (same recurrence, applied to the *time* lengths). For `dw_striding` (`conv2d_subsampling=True`) the chunking guard (`subsampling.py:366-392`) only splits the batch when the predicted output element count `≥ 2³¹` (PyTorch indexing limit, pytorch#80020) — for normal clips `need_to_split=False` so it calls `self.conv(x)` once via `MaskedConvSequential.forward`. After the convs, the conv2d branch does `b,c,t,f = x.size(); x = self.out(x.transpose(1,2).reshape(b, t, c*f))` (`subsampling.py:397-399`): permute to `(B,T',C,F')`, flatten the last two into `(B,T',4096)`, project to `(B,T',512)`. Output is `(x:(B,T',512), lengths:(B,) int)`.

**MaskedConvSequential masking** (`subsampling.py:558-586`) — this is NeMo's length-aware variant. Before each layer it zeroes time positions beyond each clip's valid length (`apply_channel_mask`, broadcasting a `(B,T,F)` 0/1 mask over channels, `subsampling.py:594-600`), runs the layer, and **only for layers whose `stride != (1,1)`** updates `current_lengths` via `calculate_conv_output_size = (L + l_pad + r_pad - k)//s + 1` (`subsampling.py:603-605`) and rebuilds the mask. The interleaved pointwise convs (`k=1,s=1`) leave length unchanged. This makes padded-batch results bit-equal to per-clip results — load-bearing because audio-in clips are right-padded (`lfm2_audio.py:341` `pad_sequence`) before the single batched `self.conv` call.

**No norm / no attention / no RoPE here.** This module is pure conv + ReLU + one Linear; RMSNorm/LayerNorm, GQA, rel-pos and SiLU live in the downstream ConformerLayers. `reset_parameters` (`subsampling.py:406`) does NeMo uniform init for training only; at inference, pretrained weights are loaded so it is dead.

## Dtypes & shapes
| stage | dtype | shape |
|---|---|---|
| input `x` (mel features, post-`mT`) | model dtype (bf16 cuda / f32 Rust-CPU / bf16 Metal) | `(B, T, 128)` — note caller passes mel as `(B,128,T)` and the encoder transposes to `(B,T,128)` before `pre_encode` |
| input `lengths` | int64 | `(B,)` |
| internal conv image | model dtype | `(B,1,T,128)` → stem `(B,256, ⌈T/2⌉, 64)` → `(B,256, ⌈T/4⌉, 32)` → `(B,256, T'=⌊…⌋, 16)` |
| flatten before `out` | model dtype | `(B, T', 256·16=4096)` |
| output `x` | model dtype | `(B, T', 512)` |
| output `lengths` | int64 | `(B,)`, each `= calc_length(len, repeat=3)` |

Internal promotions: `calc_length`/`calculate_conv_output_size` run in **f32/f64 then cast to int** (`subsampling.py:550` casts to `torch.float`, `subsampling.py:555` casts back to int; Rust does the recurrence in `f64`, `subsampling.rs:17-25`). No softmax/norm here, so no f32 upcast inside the convs — they run entirely in model dtype. The mel input itself was computed in f32/f64 upstream and stored bf16 in `ChatState`; here it is cast to `text_emb.dtype` right before the conformer (`lfm2_audio.py:346`). Parity vs Python: conv-stack out **5.611e-7** at `[1,256,13,16]`, post-subsample/pos-enc **1.019e-6** at `[1,13,512]` (PYTHON_VS_RUST.md §lines 31-32).

## Wiring
**Upstream** — [conformer_processor](CF04-Mel-Frontend) produces the mel features `(B,128,T)` bf16 (precision-sensitive f32/f64 front-end, stored bf16). They enter via [conformer_encoder](CF01-Conformer-Encoder), which transposes to `(B,T,128)` and calls `pre_encode(x, lengths)`. The caller path is [model_lfm2_audio](MD01-LFM2AudioModel) `forward`: it splits per-clip, right-pads to `(N, T_max, 128)` (`lfm2_audio.py:341`), casts to model dtype, transposes back to `(N,128,T_max)` and hands it to `self.conformer(...)` (`lfm2_audio.py:346`).

**Downstream** — output `(B,T',512)` + lengths flows back into [conformer_encoder](CF01-Conformer-Encoder), which adds `RelPositionalEncoding` and runs the 17 `ConformerLayer`s ([conformer_modules](CF03-Conformer-Layer), [conformer_mha](CF02-RelPos-MHA)), returning `(B,512,T')`. The encoder result is masked/concatenated and fed to the `audio_adapter` [model_mlp](MD03-Audio-Adapter-MLP) (`MLP(512 → 2048)`, GELU-erf), whose output `audio_in_emb=(ΣT', 2048)` is scattered into the LFM2 backbone token stream ([model_lfm2_backbone](MD01-LFM2AudioModel)). Net: `subsampling → encoder layers → audio_adapter → backbone`.

## Python ↔ Rust
Symbol map (all in `subsampling.rs`):
- `ConvSubsampling.__init__` → `ConvSubsampling::new` (thin `dw_striding`, `is_causal=false` wrapper, `subsampling.rs:250`) over `ConvSubsampling::new_scheme` (all 5 schemes, `subsampling.rs:265`). The Rust closure `next(idx)` increments the vb prefix for every pushed module incl. ReLU/MaxPool so `conv.{i}` keys match Python exactly.
- `forward` → `ConvSubsampling::forward` (`subsampling.rs:443`): unsqueeze channel, `forward_conv`, `transpose(1,2).reshape(b,t,c*f)`, `out` Linear.
- `calc_length` → `calc_length` (`subsampling.rs:17`, `f64` recurrence); `out_lengths` (`subsampling.rs:433`) is the per-clip length helper for streaming.
- `MaskedConvSequential.forward` → `MaskedConvSequential::forward` (`subsampling.rs:184`, the masked length-tracking path, reads each conv's own `config()` to skip length updates on pointwise convs). `_create_mask`/`apply_channel_mask`/`calculate_conv_output_size` → same names.
- Training-only `reset_parameters` is omitted; weights come from `VarBuilder`.

**Deliberate divergences (not bugs):**
- **Single-clip fast path.** The offline path uses `forward_conv` (no mask, `subsampling.rs:136`): for one clip the length-mask is all-ones, so it is numerically identical to the masked path (parity 5.6e-7) — verified, intentional.
- **candle stride>1 backward workaround `pad_even_hw`/`pad_even_1d`** (`subsampling.rs:45-67`). candle's strided conv2d backward mis-sizes the grad-input for **odd** spatial dims (T=101→51 ambiguity). Padding an odd H/W to even with a trailing zero is **forward-identical** (the appended zero is the same one `padding=1` adds; output count unchanged, measured 0.0 forward diff) and only fixes training backward. Stride-1 (pointwise) convs are skipped so their length is untouched.
- **`ceil_pool2d`** (`subsampling.rs:74`) reproduces torch `MaxPool2d(ceil_mode=True)` for the (unused) `vggnet` scheme by edge-replicating an odd dim then floor-pooling — `max(x_last,x_last)=x_last`, bit-identical. Off the LFM2 path.
- **`conv_split_by_batch`/`conv_split_by_channel`/`channel_chunked_conv`** are pure memory-tiling workarounds for pytorch#80020; candle has no 2³¹ limit, so the Rust ports return the plain un-tiled conv (`subsampling.rs:495-519`), which is output-equal.
- **Causal conv2d unsupported** (`subsampling.rs:92`): the `dw_striding`/`striding` `is_causal` paths need NeMo's `CausalConv2D`, which is *imported-but-undefined* in the upstream repo (no source) — the Rust errors loudly rather than guessing. LFM2 uses `causal_downsampling=False`, so this never triggers.

## Precision / gotchas
- **Length math is float-then-floor.** `calc_length` does `floor((L + (paddings-kernel))/stride + 1)` in float, repeated `_sampling_num` times. For the model's 128-bin freq axis this gives `128→64→32→16` exactly; for time it must be computed **per layer** in the masked path (each strided conv shrinks length, pointwise convs do not) — collapsing it to a single uniform formula would over-shrink length by the pointwise steps.
- **The `out` Linear width is config-derived, not hardcoded.** It is `Linear(conv_channels·calc_length(feat_in), feat_out)` = `Linear(256·16=4096, 512)`. Any change to mel bins or `conv_channels` resizes this matrix; the Rust recomputes it identically from `feat_in`/`ceil_mode` (`subsampling.rs:406-409`).
- **No special tokens / EOAudio here.** This is a feature extractor, not a token producer — codes/EOAudio (2048) live in the Mimi/depthformer audio-out path, not in audio-in subsampling.
- **The `padded_audio_in.new_empty((0, 8+1, 128))` empty-batch shape** (`lfm2_audio.py:343`) uses `8 = subsampling_factor` as a minimal time length so an empty audio-in batch still has a valid pre-encode shape; it never reaches the conv math at runtime when audio-in is present.
- **Cross-library f32 floor.** On Rust CPU there is no bf16 matmul, so this module computes in f32 even though weights are bf16 on disk (Metal stays bf16, Python default is cuda/bf16). The conv-subsample parity (5.6e-7) is well under the mel front-end's FFT-library floor (9.31e-6), so subsampling is not the precision bottleneck.
