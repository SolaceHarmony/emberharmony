<!-- topic: Core -->
# CO02 · LFM2AudioDetokenizer (ISTFT vocoder)
**Code:** `CO02` · **Source:** `detokenizer.py` · **Rust:** `detokenizer.rs / LFM2AudioDetokenizer + Istft` · **On the LFM2-Audio inference path:** yes

## Role
The high-quality audio-out vocoder for LFM2.5 models: it turns the 8-codebook discrete audio codes emitted by the depthformer head into a 24 kHz mono waveform. Unlike Mimi (the codec the codes were *encoded* from, also the v1/streaming decoder), this is an LFM2-backbone-based ISTFT vocoder shipped as a separate `audio_detokenizer/` checkpoint in the HF snapshot. It exists because the same 8×2048 code space can be re-synthesized at higher fidelity by a small dedicated LFM2 model that predicts a complex STFT spectrogram and inverts it, rather than running Mimi's SEANet decoder. Entry point is `processor.decode((1,8,T))` (`detokenizer.py:120`, dispatched `processor.py:165`).

## How it works
Four stages: code→embedding fusion, ×6 temporal upsample + LFM2 backbone under an explicit sliding-window causal mask, a Linear projecting to a complex STFT spectrogram, and a Vocos-style "same"-padded ISTFT.

**1. FusedEmbedding (`detokenizer.py:6-24`).** A single `nn.Embedding(8*2048, 512)` holds one shared table for all codebooks; codebook *k* lives in slots `[k*2048, (k+1)*2048)`. Input `x` is `(B, 8, T)` int codes in `[0,2047]`. The forward computes `offsets = arange(8)*2048` (shape `(8,)`), broadcasts `offsets[:,None] + x` so each codebook's codes are shifted into its slice (`(8,1)+(B,8,T)→(B,8,T)`), gathers the embeddings (`(B,8,T,512)`), and **means over the codebook axis** (`.mean(1)`) → `(B, T, 512)`. So the per-frame embedding is the *average* of the 8 per-codebook embeddings, not a concat — the RVQ residual structure is collapsed by averaging.

**2. ×6 upsample + LFM2 backbone (`detokenizer.py:121-130`).** `upsample_size = 6 * L` along time; `nn.functional.interpolate(x.mT, upsample_size, mode="nearest-exact").mT` repeat-interleaves each frame 6× (`(B,T,512)→(B,6T,512)`). This bridges the code frame-rate (Mimi's 12.5 Hz acoustic rate) up to the detokenizer backbone's internal rate so the subsequent STFT inverts to 24 kHz. The backbone is a full HF `Lfm2Model` (hybrid short-conv + GQA, RMSNorm, SwiGLU, RoPE — see [model_lfm2_backbone](MD01-LFM2AudioModel)), run with `use_cache=False` and an **explicit additive attention mask** built in this module: `d_idx = idx - idx[:,None]`, `mask = (d_idx <= 0) & (d_idx > -sliding_window)` (`detokenizer.py:126-128`). That is a causal band — token *i* attends to *j* iff `i - sliding_window < j <= i`, window default 30 (`config.sliding_window`, `detokenizer.py:118`). Critically, the loader rewrites `layer_types` `"sliding_attention"→"full_attention"` (`processor.py:140-149`, Rust `loader.rs:309-316`): the backbone's *per-layer* HF sliding logic is turned off so this module's hand-built band mask is the *sole* windowing. Output is `last_hidden_state` `(B, 6T, 512)`.

**3. Linear → complex spectrogram (`detokenizer.py:131-134`).** `nn.Linear(512, 1282)` → `(B, 6T, 1282)`. Transpose to `(B, 1282, 6T)` and `chunk(2, dim=1)` into `log_abs` and `angle`, each `(B, 641, 6T)` (641 = n_fft/2+1 = 1280/2+1). The complex spectrum is `y = polar(log_abs.exp(), angle)` — i.e. magnitude `exp(log_abs)` (the head predicts log-magnitude so the exp guarantees a positive modulus) at phase `angle`, giving `re = exp(log_abs)·cos(angle)`, `im = exp(log_abs)·sin(angle)`. Shape `(B, 641, T_frames=6T)` complex.

**4. ISTFT, "same" padding (`detokenizer.py:27-107`).** n_fft=1280, hop=320, win=1280, Hann window. Because `torch.istft` rejects non-center padding under NOLA, this is a custom Vocos-adapted ISTFT (`detokenizer.py:35`):
- `pad = (win - hop)//2 = (1280-320)//2 = 480`.
- `ifft = torch.fft.irfft(spec, n_fft, dim=1, norm="backward")` → real frames `(B, 1280, T_frames)`; windowed `ifft * window[None,:,None]` (`detokenizer.py:82-83`).
- Overlap-add via `F.fold` with `kernel_size=(1,win)`, `stride=(1,hop)` → `output_size=(T_frames-1)*hop + win`, then trim `[pad:-pad]` (`detokenizer.py:86-92`).
- Window envelope: `window.square()` folded the same way (`detokenizer.py:95-101`), trimmed identically; normalize `y = y / window_envelope` (`detokenizer.py:104-105`) with an assert that the envelope `> 1e-11` everywhere (NOLA holds in the interior since the edges are trimmed). Output waveform `(B, L)` f32 @ 24 kHz.

No sampling, no streaming state, no RVQ decode loop here — the codes are *already* sampled upstream; this module is a deterministic forward synthesizer.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `decode` guard (`processor.py:165`) | `audio_codes` int64/int `(1, 8, T)`, values `[0,2047]` | passthrough (raises if `≥2048` or `<0`) |
| FusedEmbedding | int codes `(B, 8, T)` (Rust: u32 `(B,L,8)`) | model dtype `(B, T, 512)` |
| ×6 interpolate | `(B, T, 512)` | `(B, 6T, 512)` |
| Lfm2Model backbone | embeds `(B, 6T, 512)` model dtype + bool/additive mask `(1,1,6T,6T)` | hidden `(B, 6T, 512)` model dtype |
| Linear `lin` | `(B, 6T, 512)` | `(B, 6T, 1282)` |
| chunk + polar | `(B, 1282, 6T)` | complex spec `(B, 641, 6T)` (Rust: `re`,`im` f32) |
| ISTFT | complex `(B, 641, 6T)` | **f32 waveform `(B, L)` @ 24 kHz** |

Promotions: backbone WEIGHTS bf16 on disk; compute = model dtype (bf16 Metal / Python cuda; f32 on Rust CPU — no CPU bf16 matmul). `torch.polar` **upcasts the bf16 head output to f32**, so the entire FFT path is f32 in both implementations; the Rust `Istft` explicitly `to_dtype(F32)` on `re`/`im` (`detokenizer.rs:131-133`). irfft basis (cos/sin, `1/n` scale) computed in f64, stored f32 (`detokenizer.rs:90-100`). Codes are integer indices throughout (no float math on them before the embedding gather).

## Wiring
**Upstream:** [core_processor](CO01-Processor-ChatState) `decode()` feeds `audio_codes` `(1,8,T)` int64 — the EOAudio frame (code 2048) **must be stripped first** (the `[0,2047]` guard rejects it). Those codes are produced by the depthformer audio head in [model_lfm2_audio](MD01-LFM2AudioModel) (`_sample_audio_frame` → frame `(8,)` int, `2048`=EOAudio) and accumulated across the turn. The backbone inside this module is the HF [model_lfm2_backbone](MD01-LFM2AudioModel) (`Lfm2Model`, the Rust spec is `lfm2_hf.rs`).

**Downstream:** the f32 `(1,L)` 24 kHz waveform returns through [core_processor](CO01-Processor-ChatState) `decode()` to the caller (demo playback / file write). On the Rust side it is surfaced via the `AudioDetokenizer` trait so [core_processor](CO01-Processor-ChatState) dispatches LFM2-detok vs Mimi uniformly. The alternative audio-out backend for the same codes is [moshi_compression](MM01-Mimi-Codec) (`MimiModel.decode`, v1/streaming).

## Python ↔ Rust
| Python (`detokenizer.py`) | Rust (`detokenizer.rs`) | note |
|---|---|---|
| `FusedEmbedding.forward` (`offsets[:,None]+x`, codebooks axis=1, `.mean(1)`) | `FusedEmbedding::forward` (codebooks **last**, `(B,L,8)`, `.mean(2)`) | same math, transposed layout — Rust takes `(B,L,8)`, Python takes `(B,8,T)`; both reduce the codebook axis |
| `Lfm2Model(backbone_config)` | `lfm2_hf::Model` (`Cache::new(false,…)`, `forward_embeds`) | external `transformers.Lfm2Model` → in-tree `lfm2_hf.rs` (the readable spec) |
| `interpolate(..., mode="nearest-exact")` ×6 | `unsqueeze→broadcast_as→reshape` repeat-interleave ×6 (`detokenizer.rs:197`) | nearest-exact at an integer factor is exactly repeat-interleave |
| sliding mask `(d<=0)&(d>-w)` bool | `sliding_mask` additive `0 / -inf` `(1,1,n,n)` f32 (`detokenizer.rs:177-189`) | bool-mask → additive mask (the port's eager-SDPA convention) |
| `torch.polar(exp(log_abs), angle)` | `abs=exp; re=abs·cos; im=abs·sin` (`detokenizer.rs:211-213`) | polar built explicitly; bf16→f32 upcast made explicit |
| `torch.fft.irfft(spec, n_fft, dim=1, norm="backward")` | inverse-DFT **basis matmul** `y=Re·cw+Im·sw` (`detokenizer.rs:88-134`) | Cooley-Tukey FFT → DFT-basis matmul; Hermitian weights `a_k` (DC/Nyquist ×1, rest ×2), `1/n` folded into the basis |
| `F.fold(... stride=hop)` overlap-add + window-envelope normalize | `conv_transpose1d` with identity kernel `(n_fft,1,n_fft)`, stride=hop (`detokenizer.rs:139-153`) | `F.fold` overlap-add ≡ `conv_transpose1d` with an identity kernel |

Deliberate divergences (PYTHON_VS_RUST.md §2.9, §2.10): both FFTs are **candle-native ports run in f32 on the model device** (CPU or Metal), not an external FFT lib — `rustfft` was dropped. Validated `== f64 reference at 1.4e-7` and end-to-end on the full HF snapshot at Metal/bf16. §2.10 documents that an f64/double-double detour was **reverted** — torch's irfft is f32 (MPS `HermiteanToRealFFTWithTensor` is f32-only) and the net was trained against f32, so f64 would be out-of-distribution precision; f32 is the faithful match. §2.1: Python hard-codes `.cuda()` for the detokenizer (`processor.py:151`) so it cannot boot CPU-only; Rust is device-agnostic.

## Precision / gotchas
- **EOAudio / range guard.** `decode` raises on any code `≥2048` or `<0` (`processor.py:165`, Rust `processor.rs:138-150`). The depthformer emits `2048` as the EOAudio sentinel; that frame **must be stripped** before decode — passing it crashes. Valid codes are exactly `[0,2047]`.
- **Mean-not-concat fusion.** FusedEmbedding *averages* the 8 codebook embeddings; the per-codebook contributions are not separable downstream. This is the deliberate RVQ-collapse, not a bug.
- **Sliding mask is this module's, not the backbone's.** The `layer_types` rename to `full_attention` (`processor.py:140-149`) disables HF per-layer sliding so the explicit band mask (window 30) is authoritative. Miss this and the backbone double-masks.
- **f32 FFT floor.** The cross-library residual on the ISTFT is f32-level (`== f64 ref 1.4e-7`); it is faithful to torch's f32 irfft, *not* bit-identical to pocketfft. Going below f32 epsilon would make it *more* accurate than torch — and out-of-distribution for the trained vocoder (PYTHON_VS_RUST.md §2.10).
- **NOLA / envelope trim.** The "same" padding (`pad=480`) violates `torch.istft`'s NOLA check at the edges; the custom ISTFT trims `[pad:-pad]` so the interior envelope is strictly positive (`assert > 1e-11`). Don't swap in `torch.istft`.
- **`norm="backward"` 1/n placement.** The single `1/n` lives on the inverse only (folded into the Rust basis `cw`/`sw`, `detokenizer.rs:90`); the forward mel STFT carries none. A doubled or missing scale gives order-one errors (corroborated by the MLX numeric-stability study, PYTHON_VS_RUST.md §1.4).
- **Operational.** The local `../model` tree omits `audio_detokenizer/`, so the loader falls back to Mimi there; run this path against the **full HF snapshot** (PYTHON_VS_RUST.md §2.11).
