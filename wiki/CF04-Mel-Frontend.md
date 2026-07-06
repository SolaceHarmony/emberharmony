<!-- topic: Conformer Encoder -->
# CF04 · FilterbankFeatures mel front-end
**Code:** `CF04` · **Source:** `model/conformer/processor.py` · **Rust:** `model/conformer/processor.rs` · **On the LFM2-Audio inference path:** yes

## Role
This is the **mel front-end** for audio-IN: it turns mic PCM into the 128-bin log-mel spectrogram the FastConformer encoder eats. It is a faithful copy of NeMo's `AudioToMelSpectrogramPreprocessor` → `FilterbankFeatures` chain (preemphasis → STFT → power → slaney mel → log → per-feature normalize). It exists because LFM2-Audio's *input* audio path is **conformer-mel, not Mimi** — mic audio never touches the codec; only audio-OUT codes round-trip through Mimi/the LFM2 detokenizer. The module is deliberately precision-pinned: NeMo warns the featurizer "is not robust to low precision mathematics," so it always runs in f32 even when the rest of the model is bf16 (`processor.py:62-67`).

## How it works
Forward pass for a single clip `x` of shape `(1, L)` at 16 kHz (`FilterbankFeatures.forward`, `processor.py:422`). The base `AudioPreprocessor.forward` (`processor.py:60-68`) first **force-casts the input to f32** (warning if it wasn't) and, at the end, casts the *output* back to the dtype sentinel (also f32, `processor.py:58/67`). The compute order is:

1. **Frame-count math** (`get_seq_len`, `processor.py:412-416`): `pad_amount = (n_fft//2)*2` for the centered path (checkpoint default `exact_pad=False`), so valid frames `= floor_divide(L + n_fft - n_fft, hop) = floor(L/hop)` for even `n_fft`. The `exact_pad=True` branch instead pre-pads the *raw* signal by `(n_fft-hop)//2` each side and uses `floor_divide(L + 2·stft_pad_amount − n_fft, hop)`.
2. **Preemphasis** (`processor.py:438-441`): first sample kept, then `y[i] = x[i] − 0.97·x[i−1]` (a 1-tap high-pass, `preemph=0.97`), then `masked_fill` zeros positions ≥ the valid sample length. Dither and narrowband augmentation are **training-only** (`self.training` guards, `processor.py:434/453`) and are skipped at inference.
3. **STFT** (`stft`, `processor.py:385-395`): `torch.stft(n_fft=512, hop_length=160, win_length=400, center=True, window=hann(400, periodic=False), return_complex=True, pad_mode="constant")`. Wrapped in `autocast(enabled=False)` (`processor.py:444`) so it stays f32 regardless of the surrounding autocast region — this is the key precision pin. `center=True` symmetric-pads `n_fft/2=256` zeros each side, so `T = 1 + L/hop` for the centered path. **`pad_mode="constant"` is deliberate** and differs from `torch.stft`'s general `reflect` default (PYTHON_VS_RUST §1.4).
4. **Magnitude → power** (`processor.py:450-460`): `|X|² = re²+im²` via `view_as_real` + `sqrt(·²·sum)`, then raised to `mag_power=2.0`. On the inference path (`use_grads=False`) the sqrt `guard` is 0.
5. **Mel projection** (`processor.py:468-470`): `fb @ |X|^p` where `fb` is `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney")` — area-normalized triangular filters, shape `(128, 257)`. Again wrapped in `autocast(enabled=False)` to avoid fp16 overflow-to-NaN.
6. **Log** (`processor.py:472-474`): `log_zero_guard_type="add"` → `log(x + 2^-24)` (additive epsilon, *not* clamp).
7. **Per-feature normalize** (`normalize_batch`, `processor.py:503-537`): for `normalize_type="per_feature"`, compute per-mel-bin mean/std **over the valid frames only** (masked by `seq_len`), with **ddof=1** bias correction (`/(count−1)`, `processor.py:532`). Then `x_std.masked_fill(isnan, 0)` (the single-frame `0/0 → NaN` guard) and `x_std += 1e-5` (`CONSTANT`, prevents divide-by-zero), broadcast `(x−mean)/std` over **all** time steps.
8. **Mask + pad** (`processor.py:488-500`): zero the trailing pad frames `[seq_len, T)`, then zero-pad time up to a multiple of `pad_to=16`.

There is **no learnable parameter** here — `window`, `fb`, and the STFT basis are all *computed at construction* from config (`processor.py:325/337-343`), not checkpoint tensors. The downstream 8× temporal reduction and the d_model lift live in `conformer_subsampling`, not here; this module only emits the raw 128-bin features.

## Dtypes & shapes
| Stage | dtype | shape |
|---|---|---|
| Input `wave` (post-resample to 16 kHz) | f32 | `(1, L)` |
| Force-cast guard (`AudioPreprocessor.forward`) | f32 | `(1, L)` |
| Preemphasis / STFT / power / mel / log / normalize | **f32** (autocast disabled; precision-pinned) | mel `(1, 128, T)`, `T = 1 + L/160` then padded ↑16 |
| `mel[0]` returned by preprocessor | f32 | `(128, T)` |
| Stored in `ChatState.audio_in` (`processor.py:238`) | **bf16** (`.to(self.dtype)`) | `(128, ΣT)` |

Internal promotions: the whole chain is f32 (NeMo precision pin); `fb`, `window`, mel-scale `hz↔mel`, and the DFT twiddles are evaluated in f64 then materialized to f32 (filter/basis accuracy, single-precision storage). No int / u32 codes here — codes belong to the audio-OUT (Mimi/detok) path, not this featurizer.

## Wiring
**Upstream:** `core_processor` (`ChatState.add_audio`, `processor.py:226-250`) feeds it. Mic PCM is resampled to 16 kHz by `torchaudio.functional.resample` (`processor.py:233`), then handed in as f32 `(1, L)`. See [core_processor](CO01-Processor-ChatState).

**Downstream (tensor flow):** the mel `(128, ΣT)` bf16 in `ChatState.audio_in` is consumed by [ConformerEncoder](CF01-Conformer-Encoder) (`lfm2_audio.py:346`, `padded_audio_in.mT` cast to model dtype), which subsamples 8× and runs N=17 conformer layers. The number of model-sequence AUDIO_IN slots is `mel2emb_len(T) = ceil(T/8)` (`utils.py:15`). The encoder output then flows through `model_mlp` (audio_adapter, 512→2048) into the backbone — but this module's direct consumer is the conformer encoder. See [conformer_subsampling](CF05-Subsampling) for the 8× math.

## Python ↔ Rust
Symbol map:
- `AudioToMelSpectrogramPreprocessor` → `AudioToMelSpectrogramPreprocessor` (wraps a `FilterbankFeatures`; `save_to`/`restore_from`/`input_example` are no-op stubs kept for 1:1 inventory, `processor.rs:564-575`).
- `AudioPreprocessor` (abstract base) → `AudioPreprocessor` struct (Rust composition for Python's `super().__init__`; the f32 input-guard `forward` is preserved, `processor.rs:511-520`).
- `FilterbankFeatures.forward` → `FilterbankFeatures::forward` (`processor.rs:278`).
- `FilterbankFeatures.stft` → `FilterbankFeatures::stft` (`processor.rs:240`).
- `normalize_batch` → `normalize_batch` with a `NormalizeType` enum covering all 4 branches (`PerFeature`/`AllFeatures`/`Fixed`/`None`, `processor.rs:393`).
- `get_seq_len` → `get_seq_len`; `log_zero_guard_value_fn` → `log_zero_guard_value_fn` (string `"tiny"`/`"eps"` cases pre-resolved to the numeric `2^-24` at config load, `processor.rs:265`).

Deliberate divergences (PYTHON_VS_RUST §1.4 / §"input torch.stft", `ARCHAEOLOGY.md:117-120`):
- **STFT as Conv1d, not FFT.** `torch.stft`/`_fft_r2c` is realized as a strided `Conv1d` against a precomputed DFT-basis kernel `(2·freq, 1, n_fft)` at stride=hop (`processor.rs:142-155/240-253`): channels `[0,freq)` carry `window[n]·cos(2πkn/N)`, channels `[freq,2·freq)` carry `−window[n]·sin(2πkn/N)`. Cross-correlation (no kernel flip) → exactly `Re`/`Im` of each bin. This is a matmul/conv form of the DFT (no Cooley–Tukey), and crucially it is **device-resident** — it runs on Metal/GPU like the rest of the model, where an external FFT library could not.
- **No external FFT / no host round-trip.** candle has no Hann/slaney-mel/STFT builtin and candle-transformers' Whisper mel uses a different convention (power spec, precomputed filters, log10, Whisper-norm, no preemphasis) with module-private helpers — so the NeMo chain is re-implemented locally (`processor.rs:19-25`).
- **f32 vs f64 precision note.** The `forward` as shipped here runs in candle `DType::F32` end-to-end (matching torch's f32-pinned reference), with window/filterbank/twiddles computed in f64 and stored f32. PYTHON_VS_RUST §1.4 documents a *further* precision-repair that moved the FFT→power→mel→log→normalize chain to **f64 on CPU** (Metal has no f64) to shave the mel residual from 1.07e-5 → 9.31e-6; treat that as the intended CPU-parity target, not a bug in the f32 form.
- **Device-agnostic / training bits dropped.** dither, narrowband augmentation, frame-splicing are skipped (inference-only); the `dtype_sentinel_tensor` buffer and `torch_windows` dict have no candle analog (compute dtype is explicit).

## Precision / gotchas
- **The f32 floor is load-bearing here.** This is the *one* stage that historically sat above the ~1e-6 cross-library floor (mel ≈ 1.07e-5, repaired to 9.31e-6). NeMo's own warning (`processor.py:64`) is real: casting the input through bf16 and back drops WER up to ~0.1% because bf16 lacks mantissa bits in `[-1,1]`. The two `autocast(enabled=False)` wrappers (STFT and mel-matmul, `processor.py:444/468`) exist precisely to prevent an outer bf16 autocast from poisoning the front-end — and the mel-matmul one *also* guards against fp16 NaN at magnitude 65520.
- **ddof=1, not ddof=0.** Per-feature std uses `/(count−1)` (`processor.py:532`). For a clip with a single valid frame this is `0/0 → NaN`, which Python masks to 0 so `std == CONSTANT (1e-5)`. The Rust short-circuits `valid ≤ 1 → std=0` to avoid producing the NaN at all (`processor.rs:400-404`); a regression test pins the actual Python golden (`normalize_one_frame_matches_python`, `processor.rs:601`).
- **`pad_mode="constant"` is the checkpoint default**, not `reflect`. Switching it to torch.stft's general default would be a regression (PYTHON_VS_RUST §1.4 item 1).
- **Additive log guard, not clamp.** `log(x + 2^-24)` — using `clamp` would change small-energy bins. `log_zero_guard_value=2^-24` (`processor.py:168`).
- **Output is f32, stored bf16.** The preprocessor returns f32 `(1,128,T)`; the *caller* (`ChatState.add_audio`) is what casts to bf16 (`processor.py:238`) before storage. The bf16 cast is therefore outside this module's precision-pinned region — by design.
- **Even-hop requirement** for `exact_pad=True` (`processor.py:287`), since an odd hop would break `frames == L // hop`. Checkpoint config uses the centered path so this doesn't bite, but it's a real guard.
