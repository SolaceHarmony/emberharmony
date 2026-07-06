<!-- topic: Mimi Codec — Modules -->
# MO02 · StreamingConv1d / ConvTranspose1d
**Code:** `MO02` · **Source:** `moshi/modules/conv.py` · **Rust:** `moshi crate conv` · **On the LFM2-Audio inference path:** yes

## Role
The low-level 1-D convolution primitives of the Mimi codec. `StreamingConv1d` / `StreamingConvTranspose1d` wrap plain `nn.Conv1d` / `nn.ConvTranspose1d` with three things stock PyTorch convs do not give you: (1) the **causal asymmetric padding** math (left-pad by `kernel-stride`, zero right-pad), (2) **streaming state** so a chunked real-time stream produces bit-identical output to a one-shot pass over the whole clip, and (3) a uniform **norm wrapper** (`NormConv*`) that supports weight-norm folding. Every conv in SEANet's encoder/decoder (`seanet.py`) and the learnt framerate resamplers (`resample.py`) is one of these two classes — so this file is the conv substrate of the entire Mimi waveform⇄latent path. On the LFM2-Audio inference path it carries the audio-OUT decode (`mimi.decode` of generated 8-code frames → 24 kHz wav) and, in training, the audio-OUT encode of reference speech into target codes.

## How it works

**The padding math (the load-bearing part).** A causal conv must emit, at step *t*, a value depending only on inputs ≤ *t*. For an effective kernel `K_eff = (kernel-1)·dilation + 1` and stride `S`, the total padding needed to keep length consistent is `padding_total = K_eff - S` (`conv.py:223-231`, `_effective_kernel_size`/`_padding_total`). In the **causal** branch this is all applied on the *left*; the right gets only `get_extra_padding_for_conv1d` — the slack needed so the **last** convolution window is full and an exact-inverse transpose-conv can rebuild the same length (`conv.py:52-76`, `pad_for_conv1d` docstring walks the `total=4,k=4,s=2` example showing why the off-by-one matters). `pad1d` (`conv.py:79-101`) is a thin `F.pad` wrapper whose only subtlety is `mode="reflect"` on inputs shorter than the pad width: it inserts temporary right zeros, reflects, then trims (`conv.py:91-99`) so reflection never reads past the tensor.

**One-shot vs streaming — the equivalence contract.** `StreamingConv1d.forward` (`conv.py:245-274`):
- *No streaming state* — padding is applied by the surrounding SEANet block (via `pad1d`), the conv runs once.
- *Streaming state present* — it keeps a ring buffer `state.previous` of the last `K_eff - S` input samples (`conv.py:240-243`). Each call **prepends** `previous` to the new chunk (`conv.py:260-261`), runs the conv, then **saves the trailing `TP=K_eff-S` samples** of the (concatenated) input back into `previous` for the next call (`conv.py:263-267`). That overlap of exactly `K_eff-S` is precisely the receptive-field carry-over, which is why streamed output equals one-shot output to ~1e-6 (the `test()` harness at `conv.py:365-419` asserts `delta ≤ 1e-6`). The hard precondition is `T % S == 0` — **steps must be a multiple of stride** (`conv.py:248`); the caller is responsible for buffering to frame boundaries.
- `pad_mode="replicate"` adds a first-frame wrinkle: on the very first streaming step there is no history, so `state.previous` is seeded by replicating the first input column `x[...,:1]` (gated by `state.first & state.exec_mask`, `conv.py:253-259`), and `first` is cleared after (`conv.py:268-273`). `constant` mode skips this (history starts as zeros).

**Transpose conv (decoder side).** `StreamingConvTranspose1d.forward` (`conv.py:340-362`). One-shot: run `convtr`, then `unpad1d(y,(0,K-S))` — trim `K-S` from the **right** only (causal, `trim_right_ratio==1`, asserted at `conv.py:308`). Streaming uses **partial-frame overlap-add**: transpose-conv outputs overlap by `PT=K-S` between adjacent input frames, so it (a) **adds** the carried `state.partial` into the head `y[...,:PT]` (`conv.py:352`), (b) saves the new tail `y[...,-PT:]` as the next `partial`, **subtracting the bias first** so bias isn't double-counted when the tail is re-added next step (`conv.py:353-360`), and (c) emits `y[...,:-PT]`. This is overlap-add reconstruction done incrementally.

**Norm wrapper / weight-norm.** `NormConv1d` / `NormConvTranspose1d` (`conv.py:113-158`) just hold the conv plus `norm_type`. `apply_parametrization_norm` (`conv.py:42-49`) applies `torch.nn.utils.weight_norm` when `norm=="weight_norm"`, else returns the bare conv. **For Mimi `norm="none"`** (`loaders.py` `_seanet_kwargs`), so at inference these are ordinary conv weights with no reparametrization to fold — the weight-norm path is dead for this checkpoint. `CONV_NORMALIZATIONS = {"none","weight_norm"}` only (`conv.py:25`); there is no group/layer norm inside a conv here (`TransposedLayerNorm` at `conv.py:29-39` exists but is used by the codec transformer, not these convs).

**Streaming state machinery.** State is a `@dataclass(State)` (`conv.py:161-169`, `277-287`) carrying `previous`/`first` (conv) or `partial` (convtr), allocated lazily in `_init_streaming_state` from a model parameter's dtype/device (`conv.py:233-243`, `330-338`). `reset(reset_mask)` zeroes the buffer per-batch-row via `torch.where(reset_mask, 0, buf)` (the turn boundary), and `exec_mask` (from `streaming.py:35-42`) gates which batch rows actually advance their state — both buffer-save sites are guarded by `state.exec_mask` so a paused/finished stream row doesn't corrupt its carry. There is **no nonlinearity, no normalization of activations, and no attention** in this file — it is pure conv + bookkeeping (SEANet supplies the ELU/`activation` around these convs).

## Dtypes & shapes
Mimi runs in module dtype (Python default cuda/bf16; Rust CPU=f32, Metal=bf16). Convs do not promote — weights and activations share the module dtype; no f32 upcast happens here (unlike RMSNorm/softmax elsewhere).

| Symbol | Input dtype+shape | Output dtype+shape | Notes |
|---|---|---|---|
| `StreamingConv1d.forward` (encoder side, `stride=ratio`) | model dtype `(B, C_in, T)`, `T % S == 0` | model dtype `(B, C_out, ⌈(T+pad−K_eff)/S⌉+1)` | hop reduces length by `S`; one SEANet downsample ratio ∈ {8,6,5,4} |
| `StreamingConv1d.forward` (k=1 pointwise / k=3 resnet) | model dtype `(B,C,T)` | model dtype `(B,C,T)` | length-preserving |
| `StreamingConvTranspose1d.forward` (decoder upsample) | model dtype `(B, C_in, T)` | model dtype `(B, C_out, T·S)` after right-trim `K−S` | one decode ratio |
| `state.previous` (conv buffer) | model dtype `(B, C_in, K_eff−S)` | — | persists across stream calls |
| `state.partial` (convtr buffer) | model dtype `(B, C_out, K−S)` | — | overlap-add carry |
| full-codec view (this file, composed) | waveform f32 `(B,1,T)`@24kHz → / ← codes int (u32) `(B,8,T/1920)` | — | conv layer is one stage of the 960× SEANet hop |

Internal promotions: **none in this file.** (Mimi weights = bf16 on disk; the f64 mel and the f32-upcast norms live in other components, not here.)

## Wiring
**Upstream (feeds these convs):**
- [SEANetEncoder/Decoder](MO01-SEANet) instantiates every `StreamingConv1d`/`StreamingConvTranspose1d` and feeds them model-dtype `(B,C,T)` activations between ELU/resnet blocks. Encoder convs take waveform-derived f32 `(B,1,T)`@24kHz at the input conv; decoder transpose-convs take latent `(B,512,T')` model dtype.
- [moshi_resample](MO04-Framerate-Resample) (`ConvDownsample1d`/`ConvTrUpsample1d`) wraps a single `StreamingConv1d` (k=`2·stride`, `pad_mode="replicate"`, `groups=1` or `dim` for channel-wise) / `StreamingConvTranspose1d` to bridge 25Hz↔12.5Hz; edge dtype = model dtype `(B,512,T)`.
- [moshi_streaming](MO06-Streaming-Module) supplies the `State`/`exec_mask`/`reset` protocol these convs store their `previous`/`partial`/`first` buffers in.

**Downstream (consumes these convs' output):**
- [SEANetEncoder/Decoder](MO01-SEANet) — the immediate consumer; conv outputs flow into the next ELU/resnet stage and ultimately to the encoder transformer (latent `(B,512,·)`@25Hz) or to the output waveform f32 `(B,1,T')`@24kHz.
- [moshi_compression](MM01-Mimi-Codec) (`MimiModel`) — owns the whole SEANet stack; its `encode`/`decode`/`decode_step` are what the rest of the system calls. Output: codes int `(B,8,·)` (encode) or waveform f32 `(B,1,1920)`@24kHz per frame (decode).
- [core_processor](CO01-Processor-ChatState) / [demo_chat](DM01-Realtime-Chat) — call `mimi.decode`/`mimi.streaming(1)` on generated **audio frames `(8,)` int (codes 0..2047)** to produce playback wav f32@24kHz; the conv buffering here is what makes that gapless.

## Python ↔ Rust
Reused as Kyutai's published **`moshi` crate v0.6.4** (`Cargo.toml: moshi = "0.6"`), `src/conv.rs` — not re-ported in `liquid-audio-rs/src/`. Symbol map:

| Python (`conv.py`) | Rust (`moshi::conv`) |
|---|---|
| `StreamingConv1d` | `StreamableConv1d` (`forward` = one-shot, `step` = streaming) |
| `StreamingConvTranspose1d` | `StreamableConvTranspose1d` |
| `NormConv1d` / `NormConvTranspose1d` | `NormConv1d` / `NormConvTranspose1d` |
| `apply_parametrization_norm` (weight_norm) | `conv1d_weight_norm` — folds `weight_g`/`weight_v` → `weight` at load (`w_v·w_g/‖w_v‖`, `conv.rs:27-45`); also handles already-folded `weight` |
| `get_extra_padding_for_conv1d` / `pad_for_conv1d` | `get_extra_padding_for_conv1d` (`conv.rs:197-208`) |
| `pad1d` / `unpad1d` | `pad1d` / `unpad1d` (`conv.rs:210-224`) |
| `_StreamingConv1dState.previous/first` | `state_prev_xs: StreamTensor` + `left_pad_applied: bool` (`conv.rs:231-233`) |
| `_StreamingConvTr1dState.partial` | `state_prev_ys: StreamTensor` (`conv.rs:377`) |
| `state.exec_mask` gating | `StreamMask` → `where_cond` in `step` (`conv.rs:347-367`, `478-498`) |
| `resample.ConvDownsample1d/ConvTrUpsample1d` | `ConvDownsample1d`/`ConvTrUpsample1d` (`conv.rs:504-606`) |

**Deliberate divergences** (per PYTHON_VS_RUST.md §2.1/§2.2, ARCH_1 §4/§7):
- **Device/dtype-agnostic.** Python is CUDA-coupled; Rust honors `device:&Device`,`dtype:DType` (CPU=f32 — candle has no CPU bf16 matmul — Metal=bf16). Numerically irrelevant to the conv math.
- **No CUDA graphs.** Python wraps the codec in `CUDAGraphed` (disabled off-CUDA); Rust runs candle ops eagerly. Latency-only.
- **`cudnn_fwd_algo: ImplicitGemm`** is set on the Rust `Conv1dConfig` (`conv.rs:257`) — a candle backend hint, not a math change.
- **Tensor layout in `step`.** Rust `StreamableConv1d::forward` reads dims as `(_b,_t,_c)` (`conv.rs:287`) reflecting the moshi crate's conv-layout convention; the *math* (left-pad `padding_total`, stride windows) is identical.
- **`pad_mode="reflect"` not supported in Rust.** `pad1d` bails on `Reflect` (`conv.rs:213`). This is safe here because **Mimi's `seanet.pad_mode` is `Constant`** in the Rust v0_1 config (`mimi.rs:49`) and the framerate resamplers use `Replicate` (`conv.rs:530`) — so no on-path conv ever requests reflect. (SEANet's Python *default* arg is `"reflect"`, but the Mimi build overrides it; see gotchas.)
- **`reset_batch_idx`** (`conv.rs:274-281`,`415-422`) is a Rust-side per-row state-zero helper; the Python analog is `State.reset(reset_mask)`.

## Precision / gotchas
- **Streaming↔one-shot equivalence is a contract, not a coincidence.** It holds *only* if every chunk length is a multiple of stride (`assert T % S == 0`, `conv.py:248`) and the `previous`/`partial` carry is exactly `K_eff−S` / `K−S`. Feed a non-multiple chunk and you silently break frame alignment. The demo/`mic_chat` always feed whole 1920-sample Mimi frames for this reason.
- **The off-by-one in `pad_for_conv1d`** (`conv.py:62-76`) is the reason transpose-conv can reconstruct the exact input length; it is *not* redundant padding. Removing it drops the last time step.
- **Bias double-count in streaming transpose-conv.** The carried `partial` has its bias subtracted before storage (`conv.py:353-356` / `conv.rs:464-470`); skipping that would add the bias twice in the overlap region. Subtle correctness bug if reimplemented naively.
- **`pad_mode` mismatch is benign here but a real footgun elsewhere.** PYTHON_VS_RUST.md §1.4 audited the *mel* center-pad (`constant`, correct) — a different conv. For *these* convs, Mimi pins `Constant` (SEANet) / `Replicate` (resample), so the Rust "no reflect" restriction never fires on-path. If a future config flipped SEANet back to its `reflect` default, the Rust crate would error rather than silently diverge.
- **weight-norm is folded, not live.** With `norm="none"` for Mimi the fold path is unused; if you ever load a `weight_g`/`weight_v` checkpoint the Rust `conv1d_weight_norm` reproduces `w = w_v · (w_g/‖w_v‖)` at load (recompute, inference-only — not the training parametrization).
- **No activation/normalization of activations in this file** — don't look here for the ELU or any RMSNorm/bf16-order subtlety; those belong to [SEANet](MO01-SEANet) and [moshi_transformer](MO03-Codec-Transformer). The conv layer is numerically a plain linear op + buffered concat; its cross-library f32 floor (~1e-6, PYTHON_VS_RUST.md §1.4) is just candle-gemm-vs-BLAS reduction order, and the moshi crate's own `conv1d`/`conv_tr1d` tests assert streamed==one-shot to `1e-5` (`conv.rs:651,689`).
- **EOAudio / code-range note.** This file never sees code indices — it operates on continuous activations. The `2048=EOAudio` special token and the `[0,2047]` code-range rejection live upstream in the quantizer/processor, not in the conv buffers.
