<!-- topic: Core -->
# CO01 · LFM2AudioProcessor + ChatState
**Code:** `CO01` · **Source:** `processor.py` · **Rust:** `processor.rs / LFM2AudioProcessor, ChatState` · **On the LFM2-Audio inference path:** yes

## Role
`LFM2AudioProcessor` is the I/O container that bundles every non-model transform LFM2-Audio needs: the HF text tokenizer (`AutoTokenizer`), the precision-sensitive mel front-end (`AudioToMelSpectrogramPreprocessor`), and the two audio-out backends (the LFM2 ISTFT `LFM2AudioDetokenizer` and the Kyutai `MimiModel` codec). `ChatState` is the turn-assembly buffer: it accumulates the five model-input tensors (`text`, `audio_in`, `audio_in_lens`, `audio_out`, `modality_flag`) across `new_turn`/`add_text`/`add_audio`/`append`/`end_turn` calls so they can be unpacked straight into `LFM2AudioModel._prefill`. It exists to keep all tokenization/featurization/codec dispatch out of the model proper, and to hold the running conversation state for streaming generation.

## How it works
The processor does **no neural compute of its own** beyond dispatch — its job is feature extraction, tokenizer encode, and code→waveform routing. The mechanism is in three places: the mel/tokenizer encode path, the `ChatState` accumulation invariants, and `decode()`.

**Construction & lazy backends.** `from_pretrained` (`processor.py:55-79`) resolves the snapshot dir (`get_model_dir` → `snapshot_download`, `utils.py:40`), reads `config.json`, and builds the mel preprocessor from `config["preprocessor"]` (a `PreprocessorConfig`). The Mimi codec (`_mimi`) and LFM2 detokenizer (`_audio_detokenizer`) are both **lazy `@property`** singletons (`processor.py:101-163`): Mimi is built empty by `moshi.models.loaders.get_mimi(None, …)` then `load_state_dict(strict=True)` from `tokenizer-e351c8d8-checkpoint125.safetensors`; the detok reads `audio_detokenizer/config.json` as an `Lfm2Config`, **rewrites `layer_types`** (`sliding_attention`→`full_attention`, `processor.py:137-149`) to make the llama.cpp-flavored config compatible with `transformers.Lfm2Model`, then `.eval().cuda()` (hard-coded device, `processor.py:151`) and loads `model.safetensors`.

**Text encode** (`add_text`, `processor.py:220-224`): `tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")` → `(1, n)` **int64** ids. No special tokens are auto-inserted; chat-template tokens are emitted as *literal text* by `new_turn`/`end_turn` (`<|im_start|>{role}\n`, `<|im_end|>\n`, `processor.py:252-256`). `ChatState.__init__` seeds the buffer with `<|startoftext|>` (`processor.py:194`).

**Audio-in / mel** (`add_audio`, `processor.py:226-250`): the wave is asserted `(1, L)` mono, moved to the audio buffer's device, then `torchaudio.functional.resample(wave, sampling_rate, 16_000)` — the model's mel front-end runs at **16 kHz** (distinct from the 24 kHz Mimi/output rate). It then calls `self.proc.audio(wave, length)` → mel `(1, 128, F)` (the NeMo `FilterbankFeatures` chain; STFT n_fft 512 / hop 160, 128 slaney mel bins, per-feature normalize), takes `mel[0]` → `(128, F)` and casts to `self.dtype` (**bf16**) for storage. The modality run appended is `LFMModality.AUDIO_IN` repeated `mel2emb_len(F)` times, where `mel2emb_len(l) = -(l // -8)` = ceil(F/8) (`utils.py:15`) — the conformer's 8× subsample factor. The raw mel **frame count** F (not the embedding count) is what is appended to `audio_in_lens`; the conformer re-derives the subsampled length from it. Three `torch.cat`s grow `audio_in` (dim 1), `modality_flag` (dim 1), `audio_in_lens` (dim 0).

**`append`** (`processor.py:258-269`) is how generated tokens re-enter the state: it asserts `text` is one row, `audio_out` has exactly `codebooks` (=8) rows, `modality_flag` is one row, and the **key invariant** `modality_flag.shape[1] == text.shape[1] + audio_out.shape[1]` — i.e. every generated step is one modality slot, scattered later by the model. These invariants exactly mirror the prefill asserts (`lfm2_audio.py:328-330`): `(flag==TEXT).sum()==text_len`, `(flag==AUDIO_OUT).sum()==audio_out_len`, `(flag==AUDIO_IN).sum()==mel2emb_len(audio_in_lens).sum()`.

**`decode()`** (`processor.py:165-177`) is the only output-side compute dispatch. It range-checks `0 ≤ code ≤ 2047` (rejecting the EOAudio sentinel **2048**, which the caller must strip) then calls `self.audio_detokenizer(audio_codes)` with `audio_codes` shaped `(1, 8, T)`. Inside the LFM2 detok, the codes path is `FusedEmbedding` → `Lfm2Model` → Linear → polar → ISTFT; note the detok's `FusedEmbedding.forward` (`detokenizer.py:23`) fuses the 8 codebooks with `offsets = arange(8)*2048` then `self.emb(offset_x).mean(1)` — a **mean** over codebooks, whereas the model's *prefill* audio embedding uses `.sum(0)` with offset stride **2049** (`lfm2_audio.py:358-359`); these are two distinct embedding tables (detok vocab 2048/codebook, model fused vocab 2049/codebook including EOAudio). The processor itself does not implement either — it just routes codes to the detok.

`decode` is wrapped in `@torch.no_grad()`; there is no sampling, RoPE, norm, or attention in this file — those live in the model/conformer/codec components it dispatches to.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `add_text` / encode | `str` | int64 `(1, n)` token ids |
| `add_audio` resample | f32 `(1, L)` @ `sampling_rate` | f32 `(1, L')` @ 16 kHz, `L'=ceil(L·16000/sr)` |
| mel front-end (`self.audio`) | f32 `(1, L')` | mel computed in f32/f64, returned f32 `(1, 128, F)`; **cast bf16** → stored `(128, F)` |
| `audio_in_lens` append | — | int64 `(k,)`, each entry = raw mel frame count F |
| `modality_flag` (audio) | — | int64 `(1, mel2emb_len(F))` filled `AUDIO_IN=2` |
| `append` (generated) | text int64 `(1,t)`, audio_out int64 `(8,a)`, flag int64 `(1,t+a)` | grows ChatState buffers |
| `decode` | int codes `(1, 8, T)`, values 0..2047 (u32 in Rust) | f32 waveform `(1, T')` @ 24 kHz |

Internal promotions: tokenizer ids are **int64** and every id-derived field inherits it (`audio_out = text.new_empty`, `modality_flag = full_like(text)`); the mel chain upcasts to **f64** internally (precision-sensitive front-end) and rounds **once** to f32, then the processor casts that to **bf16** for `audio_in` storage; codes are `int` (u32 in Rust). Weights on disk are bf16; Rust CPU compute promotes to f32 (no CPU bf16 matmul), Metal stays bf16, Python default is cuda/bf16.

## Wiring
**Upstream (feeds ChatState):**
- mic/file wav f32 `(1, L)` → `add_audio` → routed to the mel front-end [conformer_processor](CF04-Mel-Frontend) as f32 `(1, L')` @16 kHz.
- generated text token (int64) + audio frame `(8,)` int + modality flag from [model_lfm2_audio](MD01-LFM2AudioModel)'s `generate_interleaved` → `append`.
- `LFMModality` enum, `mel2emb_len`, `get_model_dir` from [core_utils](CO03-Utils).

**Downstream (consumes processor / ChatState output):**
- The five-tensor ChatState bundle (`text` int64 `(1,L)`, `audio_in` bf16 `(128,ΣF)`, `audio_in_lens` int64 `(k,)`, `audio_out` int64 `(8,m)`, `modality_flag` int64 `(1,L)`) → [model_lfm2_audio](MD01-LFM2AudioModel) `_prefill`/`generate_interleaved`.
- `decode((1,8,T))` int codes → [core_detokenizer](CO02-Detokenizer) (LFM2.5) for ISTFT vocoding, or → [moshi_compression](MM01-Mimi-Codec) `MimiModel.decode` (v1/demo streaming) → f32 `(1,T')` @24 kHz.
- `mimi.encode` (data prep) routes to [moshi_compression](MM01-Mimi-Codec) for building `audio_out` targets.

## Python ↔ Rust
Symbol map: `LFM2AudioProcessor`→`LFM2AudioProcessor` (`processor.rs:63`); `from_pretrained`→`from_pretrained` (delegates to `loader::from_pretrained`, `processor.rs:109`); `add_text/add_audio/new_turn/end_turn/append`→same on `ChatState<'a>` (`processor.rs:166-323`); `decode`→`decode` (`processor.rs:138`); the lazy `mimi`/`audio_detokenizer` properties → the two `Option<Box<dyn AudioDetokenizer>>` fields `mimi` and `audio_out` (`processor.rs:70-76`). Python's `to`/`eval`/`train` bookkeeping is omitted because candle places tensors at load and inference mode is explicit.

Deliberate divergences (PYTHON_VS_RUST.md):
- **§2.1 device/dtype.** Python hard-codes `device="cuda"` and `.cuda()` on the detok (`processor.py:151`) — won't boot CPU-only. Rust is device-agnostic; every loader takes `device:&Device`+`dtype:DType`, default `(Cpu,F32)`, Metal opt-in. CPU→f32 is correct (no candle CPU bf16 matmul) and matches Python's f32-pinned mel.
- **Two-field codec split preserved.** Python keeps `_mimi` and `_audio_detokenizer` independent (Mimi still needed for the data mapper's `encode` even on full LFM2.5 snapshots); Rust mirrors this with separate `mimi` and `audio_out` fields rather than one shared backend (`processor.rs:70-76`). `decode` dispatches `audio_out.or(mimi)`.
- **§2.7 resample.** `torchaudio.functional.resample` → faithful windowed-sinc port (`resample.rs`, sinc_interp_hann, width 6, rolloff 0.99) — Python resample lives in `add_audio`; Rust splits a `resample_16k` + `add_audio_16k` so the parity-tested mel path is shared.
- **AutoTokenizer → `tokenizers` crate** (`tokenizer.json` directly); `snapshot_download` → `hf-hub` crate (`get_model_dir`).
- **Empty-buffer init.** Python uses `torch.empty((128,0))`; candle can't allocate a zero-size buffer on Metal, so Rust holds a 1-col buffer narrowed to length 0 and **replaces** (not cat) on the first add (`processor.rs:187-196`).

## Precision / gotchas
- **Two output rates.** Audio-IN mel runs at **16 kHz**; audio-OUT (Mimi/detok) is **24 kHz**. `add_audio` resamples to 16 kHz; do not confuse with the codec rate.
- **EOAudio = 2048.** `decode` rejects codes ≥ 2048; the EOAudio sentinel must be stripped from the last frame before decode. The model's fused audio vocab is **2049** per codebook (offset stride 2049, `lfm2_audio.py`), but the *detok's* `FusedEmbedding` vocab is **2048** (offset stride 2048) — different tables.
- **sum vs mean.** Model prefill embeds codes with `.sum(0)` (stride 2049); the detok embeds with `.mean(1)` (stride 2048). The processor routes to whichever — they are not interchangeable.
- **`audio_in_lens` stores raw mel frames F, not embedding length** — the conformer re-derives `mel2emb_len(F)=ceil(F/8)` itself; storing the embedding length would double-subsample. Smallest valid mel length for the encoder is 9 (`utils.py:19`).
- **mel cast order.** The mel is computed in f32/f64 and cast to bf16 *after* the precision-sensitive chain (`new_audio_in = mel[0].to(self.dtype)`, `processor.py:238`) — bf16 only at the storage boundary, never mid-FFT. The Rust port runs the whole mel chain in f64 and rounds once to f32 (PYTHON_VS_RUST.md §1.4), keeping it above the cross-library float floor.
- **int64 throughout.** All id-bearing fields are torch.long; the generation loop hands back u32/u8 sampled tokens, so Rust `append` re-casts incoming ids to I64 to match the buffer (`processor.rs:309-312`).
- **Prefill invariant is load-bearing.** `modality_flag` length must equal `text_len + audio_out_len` per `append`, and the per-modality `.sum()`s must match each source tensor's length or `_prefill` asserts fire — the modality scatter (`in_emb[mask]=…`) silently mis-aligns otherwise.
