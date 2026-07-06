# conformer_processor (Rust port)
**Source:** `liquid-audio/src/model/conformer/processor.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/processor.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/conformer/processor.md`](../../../wiki/model/conformer/processor.md).

## Role
This is the **mel front-end** for audio-IN in the Rust port: it turns mic PCM
into the 128-bin log-mel spectrogram the FastConformer encoder eats. It is a
faithful copy of NeMo's `AudioToMelSpectrogramPreprocessor` →
`FilterbankFeatures` chain (preemphasis → STFT → power → slaney mel → log →
per-feature normalize). LFM2-Audio's *input* audio path is **conformer-mel,
not Mimi** — mic audio never touches the codec; only audio-OUT codes round-trip
through Mimi/the LFM2 detokenizer. The module is precision-pinned: NeMo warns
the featurizer "is not robust to low precision mathematics," so the Rust port
runs the whole chain in **f32** on the model device (CPU or Metal) — no
external FFT library, no host round-trip.

## How it works (Rust)
Forward pass for a single clip `samples` of shape `(L,)` or `(1, L)` at 16 kHz
(`FilterbankFeatures::forward`, `processor.rs:278`). The compute order is:

1. **Force-cast to f32 on the model device** (`:282`):
   `samples.flatten_all()?.to_dtype(DType::F32)?.to_device(dev)?`. This is the
   Rust analog of NeMo's `AudioPreprocessor.forward` f32 input-guard.
2. **Frame-count math** (`get_seq_len`, `:206`): `pad_amount = stft_pad_amount*2`
   (exact_pad) else `(n_fft // 2) * 2` (centered); `seq_len =
   (seq_len + pad_amount).saturating_sub(n_fft) / hop`. For the centered path
   with even `n_fft`, this collapses to `floor(L/hop)`.
3. **Preemphasis** (`:299-307`): `y[0] = x[0]`; `y[i] = x[i] − preemph·x[i−1]`
   (`preemph=0.97`), via `head = x.narrow(1, 0, 1)` and
   `tail = x.narrow(1, 1, li-1) − x.narrow(1, 0, li-1).affine(preemph, 0.0)`,
   then `cat([head, tail])`. A `masked_fill` zeros positions ≥ the valid sample
   length (`:310-316`). Dither/narrowband are training-only and skipped.
4. **STFT** (`stft`, `:240`): `torch.stft(n_fft=512, hop=160, win=400,
   center=True, window=hann(400, periodic=False), return_complex=True,
   pad_mode="constant")` — realized as a **`Conv1d` against a precomputed
   DFT-basis kernel** `(2·freq, 1, n_fft)` at stride=hop (`:142-155`). Channels
   `[0, freq)` carry `window[n]·cos(2πkn/N)`; channels `[freq, 2·freq)` carry
   `−window[n]·sin(2πkn/N)`. Cross-correlation (no kernel flip) → exactly
   `Re`/`Im` of each bin. `pad_mode="constant"`: symmetric `n_fft/2` zeros each
   side (`:247`). `T = 1 + (L + 2·center_pad − n_fft)/hop`.
5. **Magnitude → power** (`:325-330`): `|X|² = re²+im²`; `mag_power=2.0` ⇒
   `p2` directly; else `sqrt(p2).powf(mag_power)`. The `guard` is 0 on the
   inference path (`use_grads=false`).
6. **Mel projection** (`:334`): `fb @ power` where `fb` is
   `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000,
   norm="slaney")` — area-normalized triangular filters `(128, 257)`, computed
   at construction by `mel_filterbank` (`:96-121`).
7. **Log** (`:337-338`): `log_zero_guard_type="add"` → `log(mel + 2^-24)`
   (additive epsilon, *not* clamp). The guard value is pre-resolved at config
   load (`:265`).
8. **Per-feature normalize** (`normalize_batch`, `:393`): for
   `NormalizeType::PerFeature`, compute per-mel-bin mean/std **over the valid
   frames only** (masked by `seq_len`), with **ddof=1** bias correction
   (`/(count−1)`). The single-frame `0/0 → NaN` guard: Rust short-circuits
   `valid ≤ 1 → std=0` (`:400-404`) to avoid producing the NaN at all; a
   regression test (`normalize_one_frame_matches_python`) pins the actual Python
   golden. Then `x_std += 1e-5` (`CONSTANT`), broadcast `(x−mean)/std` over **all**
   time steps.
9. **Mask + pad** (`:343-355`): zero the trailing pad frames `[seq_len, T)`, then
   zero-pad time up to a multiple of `pad_to=16`.

There is **no learnable parameter** here — `window`, `fb`, and the STFT DFT-basis
kernel are all *computed at construction* from `MelConfig` (`:171-178`), not
checkpoint tensors. The downstream 8× temporal reduction and the d_model lift
live in `conformer_subsampling`, not here; this module only emits the raw
128-bin features.

## Dtypes & shapes (Rust)
| Stage | dtype | shape |
|---|---|---|
| Input `samples` | f32 (force-cast on entry, `:282`) | `(L,)` or `(1, L)` |
| Preemphasis / STFT / power / mel / log / normalize | **f32** (precision-pinned, on device) | mel `(1, 128, T)`, `T = 1 + L/160` then padded ↑16 |
| `mel[0]` returned by `forward` | f32 | `(128, T)` (then `(1, 128, T)` unsqueezed, `:356`) |
| Stored in `ChatState.audio_in` | **bf16** (cast by the caller, `processor.rs:238` analog) | `(128, ΣT)` |

Internal promotions: the whole chain is f32 (NeMo precision pin); `fb`,
`window`, mel-scale `hz↔mel` (`:71-92`), and the DFT twiddles (`:142-155`) are
evaluated in **f64** then materialized to f32 (filter/basis accuracy,
single-precision storage). No int / u32 codes here.

## Wiring (Rust)
**Upstream:** `processor.rs` (`ChatState::add_audio`, the analog of
`processor.py:226-250`) feeds it. Mic PCM is resampled to 16 kHz by
`crate::resample` (the `torchaudio.functional.resample` port, windowed-sinc),
then handed in as f32 `(1, L)`. See
[`glm-version/processor.md`](processor.md).

**Downstream (tensor flow):** the mel `(128, ΣT)` bf16 in `ChatState.audio_in`
is consumed by `ConformerEncoder` (`lfm2_audio.rs:683`,
`conformer.forward(&seg)`), which subsamples 8× and runs N=17 conformer layers.
The number of model-sequence `AUDIO_IN` slots is `mel2emb_len(T) = ceil(T/8)`.
The encoder output then flows through `model_mlp` (audio_adapter, 512→2048)
into the backbone — but this module's direct consumer is the conformer encoder.
See [`glm-version/model/conformer/subsampling.md`](subsampling.md) for the 8×
math and [`glm-version/model/conformer/encoder.md`](encoder.md).

## Python ↔ Rust — where the port differs

| Python (`processor.py`) | Rust (`processor.rs`) | Difference | Why |
|---|---|---|---|
| `torch.stft` (pocketfft) | `Conv1d` against a precomputed DFT-basis kernel (`:142-155`, `:240-253`) | **deliberate: STFT as Conv1d, not FFT** | candle has no STFT builtin; the DFT-basis conv is a matmul/conv form of the DFT (no Cooley-Tukey) that is **device-resident** — runs on Metal/GPU like the rest of the model, where an external FFT library could not. Same result to the f32 floor. |
| `_fft_r2c` (cuFFT/MKL) | real/imag `narrow` from the conv output (`:250-251`) | **deliberate** | the conv produces `(1, 2·freq, T)`; the first `freq` channels are `Re`, the next `freq` are `Im`. |
| `torch.hann_window(N, periodic=False)` | `hann(N)` (`:60-68`) computed in f64 then cast to f32 | **deliberate: hand-rolled** | candle has no Hann window builtin. `0.5 − 0.5·cos(2πi/(N−1))` in f64, cast to f32. |
| `librosa.filters.mel(sr, n_fft, n_mels, norm="slaney")` | `mel_filterbank(sr, n_fft, n_mels)` (`:96-121`) computed in f64 then cast to f32 | **deliberate: hand-rolled** | candle has no slaney mel-filterbank builtin. `hz_to_mel`/`mel_to_hz` (`:71-92`) reproduce librosa's slaney scale; `enorm = 2/(upper-lower)` is the slaney normalization. |
| `autocast(enabled=False)` around STFT + mel | (no equivalent needed) | **deliberate: no autocast** | candle has no implicit autocast; the chain runs in explicit f32 throughout. |
| `AudioPreprocessor.forward` f32 input-guard + output cast | `forward` force-casts to f32 on entry (`:282`) | identical | — |
| `FilterbankFeatures.forward` | `FilterbankFeatures::forward` (`:278`) | identical (1:1) | — |
| `normalize_batch` with `normalize_type` string | `normalize_batch` with `NormalizeType` enum (`:393`) — `PerFeature`/`AllFeatures`/`Fixed`/`None` | **deliberate: string → enum** | Rust's enum is the idiomatic analog of Python's string dispatch. |
| `log_zero_guard_value_fn(self, x)` (string `"tiny"`/`"eps"` resolved via `torch.finfo`) | `log_zero_guard_value_fn(&self, _x)` returns the pre-resolved `f64` (`:265`) | **deliberate: pre-resolved** | the checkpoint configs use the numeric default `2**-24`; the string branches are pre-resolved at config load into `MelConfig.log_zero_guard_value`. The `_x` arg is kept for 1:1 signature parity but unused. |
| `save_to`/`restore_from` (NeMo pickle) | no-op stubs (`:564-575`) | **deliberate: no-op** | persistence is safetensors + `from_pretrained`; NeMo pickle has no candle analog. Kept for 1:1 inventory. |
| `input_example` | no-op stub | **deliberate: no-op** | ONNX export hook; no export path. |
| dither, narrowband augmentation, frame splicing | skipped (inference-only) | **deliberate: skipped** | training-only; `self.training` guards in Python, simply not ported. |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic; f32 on device | **deliberate** | §2.1. The chain runs on `Device::Cpu` or Metal; no host round-trip. |
| `exact_pad=True` pre-pads the raw signal | `stft_pad_amount` from `MelConfig` (`:50-56`) | identical | the centered (`exact_pad=False`) path pads inside `stft`; the exact_pad path pre-pads the signal and uses `center_pad=0`. |

**Deliberate divergences** (PYTHON_VS_RUST §1.4 / §2.9, `ARCHAEOLOGY.md`):
- **STFT as Conv1d, not FFT.** `torch.stft`/`_fft_r2c` is realized as a strided
  `Conv1d` against a precomputed DFT-basis kernel. This is a matmul/conv form
  of the DFT (no Cooley-Tukey), and crucially it is **device-resident** — it
  runs on Metal/GPU like the rest of the model, where an external FFT library
  could not. Parity vs torch golden: **mel 1.18e-5**, conformer-through-mel
  **5.6e-7** (PYTHON_VS_RUST §2.9).
- **No external FFT / no host round-trip.** candle has no Hann/slaney-mel/STFT
  builtin and candle-transformers' Whisper mel uses a different convention
  (power spec, precomputed filters, log10, Whisper-norm, no preemphasis) with
  module-private helpers — so the NeMo chain is re-implemented locally
  (`processor.rs:19-25`).
- **f32 vs f64 precision note.** The `forward` as shipped runs in candle
  `DType::F32` end-to-end (matching torch's f32-pinned reference), with
  window/filterbank/twiddles computed in f64 and stored f32. PYTHON_VS_RUST
  §1.4 documents a *further* precision-repair that moved the
  FFT→power→mel→log→normalize chain to **f64 on CPU** (Metal has no f64) to
  shave the mel residual from 1.07e-5 → 9.31e-6; treat that as the intended
  CPU-parity target, not a bug in the f32 form.
- **Device-agnostic / training bits dropped.** dither, narrowband
  augmentation, frame-splicing are skipped (inference-only); the
  `dtype_sentinel_tensor` buffer and `torch_windows` dict have no candle
  analog (compute dtype is explicit).

## Precision / gotchas (Rust-specific)
- **The f32 floor is load-bearing here.** This is the *one* stage that
  historically sat above the ~1e-6 cross-library floor (mel ≈ 1.07e-5, repaired
  to 9.31e-6). NeMo's own warning (`processor.py:64`) is real: casting the
  input through bf16 and back drops WER up to ~0.1% because bf16 lacks
  mantissa bits in `[-1,1]`. The Rust port runs the whole chain in explicit f32
  (`:282` force-cast; no autocast to disable).
- **STFT-as-Conv1d is the precision-critical substitution.** The DFT-basis
  kernel is computed in f64 (`:144-153`) and stored f32 — accurate basis,
  single-precision storage, matching torch's f32 FFT. The conv form is
  mathematically exact (no Cooley-Tukey approximation); the only residual is
  the f32 matmul reduction order vs pocketfft, which is the irreducible
  cross-library floor.
- **`pad_mode="constant"` is the checkpoint default**, not `reflect`. The
  `pad_with_zeros` (`:247`) matches torch's `center=True` symmetric zero-pad.
  Switching to `reflect` would be a regression (§1.4 item 1).
- **ddof=1, not ddof=0.** Per-feature std uses `/(count−1)`. For a clip with a
  single valid frame this is `0/0 → NaN`; Rust short-circuits `valid ≤ 1 →
  std=0` (`:400-404`) to avoid producing the NaN at all. The
  `normalize_one_frame_matches_python` test pins the actual Python golden.
- **Additive log guard, not clamp.** `log(mel + 2^-24)` (`:337-338`) — using
  `clamp` would change small-energy bins. `log_zero_guard_value=2^-24`
  (`MelConfig`).
- **Output is f32, stored bf16 by the caller.** `forward` returns f32
  `(1, 128, T)` (`:356`); the *caller* (`ChatState::add_audio`) is what casts to
  bf16 before storage. The bf16 cast is therefore outside this module's
  precision-pinned region — by design.
- **Even-hop requirement** for `exact_pad=True` (`MelConfig.stft_pad_amount`,
  `:50-56`), since an odd hop would break `frames == L // hop`. The checkpoint
  config uses the centered path so this doesn't bite, but it's a real guard.
- **`stft_kernel` is computed once, not per-call.** The DFT-basis
  `(2·freq, 1, n_fft)` kernel is built in `new` (`:177`) and reused — it's a
  constant (the window is folded in). Don't rebuild it per forward.
- **`fb` is computed once, not per-call.** The slaney mel filterbank `(nfilt,
  freq)` is built in `new` (`:174-175`) and reused.
- **No int/u32 codes here** — codes belong to the audio-OUT (Mimi/detok) path,
  not this featurizer.

## Cross-references
- [`wiki/model/conformer/processor.md`](../../../wiki/model/conformer/processor.md)
  — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §1.4 (the mel precision repair), §2.9
  (audio FFTs — candle-native ports).
- `liquid-audio/parity/PARITY.md` — mel 9.31e-6, conformer-through-mel 5.6e-7.