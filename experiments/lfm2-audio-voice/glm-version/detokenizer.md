# core_detokenizer (Rust port)
**Source:** `liquid-audio-rs/src/detokenizer.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/detokenizer.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`ARCH/detokenizer.md`](../ARCH/detokenizer.md). The original
> documents the Python `LFM2AudioDetokenizer` + `Istft`; this documents the Rust
> port and where it diverges.

## Role
`LFM2AudioDetokenizer` (`detokenizer.rs:158`) is the high-quality audio-out
vocoder for LFM2.5 models in the Rust port: it turns the 8-codebook discrete
audio codes emitted by the depthformer head into a 24 kHz mono waveform. Unlike
Mimi (the codec the codes were *encoded* from, also the v1/streaming decoder),
this is an LFM2-backbone-based ISTFT vocoder shipped as a separate
`audio_detokenizer/` checkpoint in the HF snapshot. It exists because the same
8×2048 code space can be re-synthesized at higher fidelity by a small dedicated
LFM2 model that predicts a complex STFT spectrogram and inverts it, rather than
running Mimi's SEANet decoder. Entry point is `processor.decode((1,8,T))`
(`processor.rs:138`), dispatched through the `AudioDetokenizer` trait
(`audio_out.rs`).

## How it works (Rust)
Four stages: code→embedding fusion, ×6 temporal upsample + LFM2 backbone under
an explicit sliding-window causal mask, a `Linear` projecting to a complex
STFT spectrogram, and a Vocos-style "same"-padded ISTFT.

**1. `FusedEmbedding` (`detokenizer.rs:26`).** A single
`Embedding(8*2048, 512)` holds one shared table for all codebooks; codebook *k*
lives in slots `[k*2048, (k+1)*2048)`. Input `x` is `(B, L, 8)` u32 codes in
`[0,2047]`. `forward` (`:40`): `offsets.reshape((1, 1, 8))`, `offset_x =
x.broadcast_add(&offsets)` → `(B, L, 8)`, `flat = offset_x.reshape((B*L*8,))`,
`emb.forward(&flat)?.reshape((B, L, 8, 512))`, `emb.mean(2)` → `(B, L, 512)`.
So the per-frame embedding is the *average* of the 8 per-codebook embeddings,
not a concat — the RVQ residual structure is collapsed by averaging.

**2. ×6 upsample + LFM2 backbone (`detokenizer.rs:192`).** `x.unsqueeze(2)?
.broadcast_as((B, L, 6, D))?.reshape((B, 6L, D))` repeat-interleaves each frame
6× (`:197`) — nearest-exact at an integer factor is exactly repeat-interleave.
This bridges the code frame-rate (Mimi's 12.5 Hz acoustic rate) up to the
detokenizer backbone's internal rate so the subsequent STFT inverts to 24 kHz.
The backbone is the in-tree `lfm2_hf::Model` (hybrid short-conv + GQA, RMSNorm,
SwiGLU, RoPE — see [`glm-version/model/lfm2_backbone.md`](model/lfm2_backbone.md)),
run with `Cache::new(false, …)` (no KV cache) and an **explicit additive
attention mask** built in `sliding_mask` (`:177`): `d = j - i`; attend where
`d <= 0 && d > -w` (window default 30); `-inf` elsewhere. That is a causal band
— token *i* attends to *j* iff `i - sliding_window < j <= i`. Critically, the
loader rewrites `layer_types` `"sliding_attention"→"full_attention"`
(`loader.rs:309-316` analog): the backbone's *per-layer* HF sliding logic is
turned off so this module's hand-built band mask is the *sole* windowing.
`forward_embeds(&x, 0, &mut cache, Some(&mask))` → `last_hidden_state`
`(B, 6L, 512)`.

**3. `Linear` → complex spectrogram (`detokenizer.rs:203`).** `lin = linear(512,
1282, …)` → `(B, 6L, 1282)`. Transpose to `(B, 1282, 6L)` and `narrow` into
`log_abs` and `angle`, each `(B, 641, 6L)` (641 = n_fft/2+1 = 1280/2+1). The
complex spectrum is `abs = log_abs.exp()`, `re = abs * angle.cos()`, `im = abs *
angle.sin()` (`:211-213`) — i.e. magnitude `exp(log_abs)` (the head predicts
log-magnitude so the exp guarantees a positive modulus) at phase `angle`.

**4. `Istft`, "same" padding (`detokenizer.rs:54`).** n_fft=1280, hop=320,
win=1280, Hann window (loaded from the checkpoint, `:74`). The custom
Vocos-adapted ISTFT:
- `pad = (win - hop) / 2 = (1280 - 320) / 2 = 480` (`:116`).
- **irfft as an inverse-DFT basis matmul** (`:131-135`): cast `re`/`im` to f32,
  `re_t.matmul(&cw) + im_t.matmul(&sw)` → `(B·T, n_fft)`. The basis `cw`/`sw`
  carry the Hermitian weights `a_k` (DC/Nyquist ×1, rest ×2) and the
  `norm="backward"` `1/n` scale (`:90-100`), computed in f64 and stored f32.
- Window: `frames.broadcast_mul(&window)` (`:138`).
- **Overlap-add via `conv_transpose1d`** (`:139`) with an identity kernel
  `(n_fft, 1, n_fft)` at stride=hop — `F.fold` overlap-add ≡ `conv_transpose1d`
  with an identity kernel.
- Window envelope: `win_sq.broadcast_as((B, n_fft, T)).conv_transpose1d(&ola,
  …)` (`:141-146`), trimmed identically.
- Trim `[pad, pad+valid)` (`:151-152`), normalize `y / env` (`:153`). Output
  waveform `(B, L)` f32 @ 24 kHz.

No sampling, no streaming state, no RVQ decode loop here — the codes are
*already* sampled upstream; this module is a deterministic forward synthesizer.

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| `decode` guard (`processor.rs:138`) | `audio_codes` u32/I64 `(1, 8, T)`, values `[0,2047]` | passthrough (errors if `≥2048`) |
| `FusedEmbedding` | u32 codes `(B, L, 8)` (Rust layout: codebooks last) | model dtype `(B, L, 512)` |
| ×6 interpolate | `(B, L, 512)` | `(B, 6L, 512)` |
| `lfm2_hf::Model` backbone | embeds `(B, 6L, 512)` model dtype + additive mask `(1, 1, 6L, 6L)` f32 | hidden `(B, 6L, 512)` model dtype |
| `Linear` `lin` | `(B, 6L, 512)` | `(B, 6L, 1282)` |
| split + polar | `(B, 1282, 6L)` | `re`, `im` f32 `(B, 641, 6L)` |
| `Istft` | `re`, `im` `(B, 641, 6L)` | **f32 waveform `(B, L)` @ 24 kHz** |

Promotions: backbone WEIGHTS bf16 on disk; compute = model dtype (bf16 Metal /
f32 CPU — no CPU bf16 matmul). The `Linear` output is cast to f32 before the
ISTFT (`:131-133`), matching torch's `torch.polar` upcast of the bf16 head
output — the entire FFT path is f32 in both implementations. The irfft basis
(`cw`/`sw`) is computed in f64, stored f32 (`:90-100`). Codes are integer
indices throughout (no float math on them before the embedding gather).

## Wiring (Rust)
**Upstream:** `processor.rs::decode` (`:138`) feeds `audio_codes` `(1, 8, T)`
— the EOAudio frame (code 2048) **must be stripped first** (the `[0,2047]`
guard rejects it, `:147-151`). Those codes are produced by the depthformer
audio head in `lfm2_audio.rs::sample_audio_frame` → frame `(8,)` int,
`2048`=EOAudio) and accumulated across the turn. See
[`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md). The backbone inside
this module is `lfm2_hf::Model` — see
[`glm-version/model/lfm2_backbone.md`](model/lfm2_backbone.md). `decode`
dispatches through the `AudioDetokenizer` trait (`audio_out.rs`).

**Downstream:** the f32 `(1, L)` 24 kHz waveform returns through
`processor.rs::decode` to the caller (demo playback / file write). The
alternative audio-out backend for the same codes is `MimiDetokenizer`
(`audio_out.rs`, v1/streaming). See
[`glm-version/processor.md`](processor.md).

## Python ↔ Rust — where the port differs

| Python (`detokenizer.py`) | Rust (`detokenizer.rs`) | Difference | Why |
|---|---|---|---|
| `FusedEmbedding.forward` (`offsets[:,None]+x`, codebooks axis=1, `.mean(1)`) | `FusedEmbedding::forward` (codebooks **last**, `(B, L, 8)`, `.mean(2)`, `:40-48`) | **deliberate: transposed layout** | Rust takes `(B, L, 8)`, Python takes `(B, 8, T)`; both reduce the codebook axis. The Rust layout matches how `processor.decode` reshapes the codes. |
| `Lfm2Model` (external HF) | `lfm2_hf::Model` (`:170`) | **deliberate: in-tree port** | external `transformers.Lfm2Model` → in-tree `lfm2_hf.rs` (the readable spec). `Cache::new(false, …)` (no KV cache); `forward_embeds(&x, 0, &mut cache, Some(&mask))`. |
| `interpolate(..., mode="nearest-exact")` ×6 | `unsqueeze→broadcast_as→reshape` repeat-interleave ×6 (`:197`) | **deliberate: repeat-interleave** | nearest-exact at an integer factor is exactly repeat-interleave. |
| sliding mask `(d<=0)&(d>-w)` bool | `sliding_mask` additive `0 / -inf` `(1,1,n,n)` f32 (`:177-189`) | **deliberate: bool → additive** | the port's eager-SDPA convention (matches `mha.rs`'s `masked_softmax` using `where_cond`). |
| `torch.polar(exp(log_abs), angle)` | `abs=exp; re=abs·cos; im=abs·sin` (`:211-213`) | **deliberate: explicit polar** | candle has no `torch.polar`; built explicitly. The bf16→f32 upcast is made explicit (`:131-133`). |
| `torch.fft.irfft(spec, n_fft, dim=1, norm="backward")` | inverse-DFT **basis matmul** `y = Re·cw + Im·sw` (`:131-135`) | **deliberate: DFT-basis matmul, not FFT** | §2.9. Cooley-Tukey FFT → DFT-basis matmul; Hermitian weights `a_k` (DC/Nyquist ×1, rest ×2), `1/n` folded into the basis. Device-resident (runs on Metal/GPU). |
| `F.fold(..., stride=hop)` overlap-add | `conv_transpose1d` with identity kernel `(n_fft, 1, n_fft)`, stride=hop (`:139`) | **deliberate: `conv_transpose1d`** | `F.fold` overlap-add ≡ `conv_transpose1d` with an identity kernel. candle has no `F.fold`; `conv_transpose1d` is the faithful equivalent. |
| window-envelope normalize via `F.fold` | `win_sq.broadcast_as(…).conv_transpose1d(&ola, …)` (`:141-146`) | identical (same overlap-add) | — |
| device/dtype hardcoded `.cuda()` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. Python hard-codes `.cuda()` for the detokenizer (`processor.py:151`) — won't boot CPU-only. Rust takes `device`+`dtype`, defaults `(Cpu, F32)`, Metal opt-in. |

**Deliberate divergences** (PYTHON_VS_RUST.md §2.9, §2.10): both FFTs are
**candle-native ports run in f32 on the model device** (CPU or Metal), not an
external FFT lib — `rustfft` was dropped. Validated `== f64 reference at
1.4e-7` (the `candle_istft_matches_f64_reference` test at `:254`) and
end-to-end on the full HF snapshot at Metal/bf16. §2.10 documents that an
f64/double-double detour was **reverted** — torch's irfft is f32 (MPS
`HermiteanToRealFFTWithTensor` is f32-only) and the net was trained against f32,
so f64 would be out-of-distribution precision; f32 is the faithful match.

## Precision / gotchas (Rust-specific)
- **EOAudio / range guard.** `processor.decode` errors on any code `≥2048`
  (`processor.rs:147-151`). The depthformer emits `2048` as the EOAudio
  sentinel; that frame **must be stripped** before decode. Valid codes are
  exactly `[0,2047]`.
- **Mean-not-concat fusion.** `FusedEmbedding` *averages* the 8 codebook
  embeddings (`emb.mean(2)`, `:47`); the per-codebook contributions are not
  separable downstream. This is the deliberate RVQ-collapse, not a bug.
- **Sliding mask is this module's, not the backbone's.** The `layer_types`
  rename to `full_attention` (in `loader.rs`) disables HF per-layer sliding so
  the explicit band mask (window 30, `sliding_mask` at `:177`) is
  authoritative. Miss this and the backbone double-masks. The mask is additive
  (`0 / -inf`), matching the port's eager-SDPA convention.
- **f32 FFT floor.** The cross-library residual on the ISTFT is f32-level (`==
  f64 ref 1.4e-7`, pinned by `candle_istft_matches_f64_reference`); it is
  faithful to torch's f32 irfft, *not* bit-identical to pocketfft. Going below
  f32 epsilon would make it *more* accurate than torch — and
  out-of-distribution for the trained vocoder (§2.10).
- **`norm="backward"` 1/n placement.** The single `1/n` lives on the inverse
  only (folded into the Rust basis `cw`/`sw`, `:90`); the forward mel STFT
  carries none. A doubled or missing scale gives order-one errors.
- **NOLA / envelope trim.** The "same" padding (`pad=480`, `:116`) violates
  `torch.istft`'s NOLA check at the edges; the custom ISTFT trims
  `[pad, pad+valid)` (`:151-152`) so the interior envelope is strictly positive.
  Don't swap in `torch.istft`.
- **`conv_transpose1d` identity kernel.** `ola` is `(n_fft, 1, n_fft)` with
  `eye[i*n_fft + i] = 1.0` (`:110-113`). `conv_transpose1d` with stride=hop
  places frame sample `j` at output `t·hop + j` — exactly overlap-add.
- **`FusedEmbedding` layout: codebooks last.** Rust takes `(B, L, 8)`, Python
  takes `(B, 8, T)`. The `processor.decode` reshapes `(1, 8, T)` → `(1, T, 8)`
  before calling `FusedEmbedding::forward`. Don't pass `(B, 8, T)` to the Rust
  `forward` — the offset broadcast assumes codebooks last.
- **Operational.** The local `../model` tree omits `audio_detokenizer/`, so the
  loader falls back to Mimi there; run this path against the **full HF
  snapshot** (§2.11).
- **`Istft::new` loads the window from the checkpoint.** `vb.get(win_length,
  "window")` (`:74`) — the Hann window is a checkpoint tensor, not computed
  (unlike the mel front-end's `hann`). If `win_length < n_fft` it's centered
  (`:76-83`).

## Cross-references
- [`ARCH/detokenizer.md`](../ARCH/detokenizer.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.9 (audio FFTs
  — candle-native ports), §2.10 (the reverted f64 detour), §2.11 (the local
  `../model` gap).
- `liquid-audio-rs/src/audio_out.rs` — the `AudioDetokenizer` trait + the
  `MimiDetokenizer` backend.
- `liquid-audio-rs/src/lfm2_hf.rs` — the backbone used here.