# core_processor (Rust port)
**Source:** `liquid-audio-rs/src/processor.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/processor.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`ARCH/processor.md`](../ARCH/processor.md). The original
> documents the Python `LFM2AudioProcessor` + `ChatState`; this documents the
> Rust port and where it diverges.

## Role
`LFM2AudioProcessor` (`processor.rs:63`) is the I/O container in the Rust port
that bundles every non-model transform LFM2-Audio needs: the text tokenizer
(HF `AutoTokenizer` → the `tokenizers` crate, reading `tokenizer.json`
directly), the precision-sensitive mel front-end (`FilterbankFeatures`), and
the two audio-out backends behind the `AudioDetokenizer` trait
(`audio_out.rs`). `ChatState` (`processor.rs:166`) is the turn-assembly buffer:
it accumulates the five model-input tensors (`text`, `audio_in`,
`audio_in_lens`, `audio_out`, `modality_flag`) across `new_turn`/`add_text`/
`add_audio`/`append`/`end_turn` calls so they can be unpacked straight into
`LFM2AudioModel::prefill_inputs`. It keeps all tokenization/featurization/codec
dispatch out of the model proper and holds the running conversation state for
streaming generation.

## How it works (Rust)
The processor does **no neural compute of its own** beyond dispatch — its job
is feature extraction, tokenizer encode, and code→waveform routing. The
mechanism is in three places: the mel/tokenizer encode path, the `ChatState`
accumulation invariants, and `decode()`.

**Construction & lazy backends.** `LFM2AudioProcessor::new` (`:88`) takes an
already-built `Tokenizer`, `FilterbankFeatures`, and two
`Option<Box<dyn AudioDetokenizer>>` fields (`audio_out`, `mimi`). The
`from_pretrained` classmethod (`:109`) delegates to
`crate::loader::from_pretrained` (which builds both the model and the processor
in one pass over the checkpoint) and returns just the processor — the model is
dropped here (the Python classmethod likewise only returns the processor). The
Mimi codec (`mimi`) and LFM2 detokenizer (`audio_out`) are **two independent
fields**, not one shared backend (`:70-76`): Mimi is still needed for the data
mapper's `encode` even on full LFM2.5 snapshots where the decode backend is the
LFM2 detokenizer. `decode` dispatches `audio_out.or(mimi)` (`:156-159`).

**Text encode** (`encode`, `:124`): `tokenizer.encode(text, false)` (no special
tokens) → `Vec<i64>` ids → `(1, n)` **I64** tensor on the device. No special
tokens are auto-inserted; chat-template tokens are emitted as *literal text* by
`new_turn`/`end_turn` (`<|im_start|>{role}\n`, `<|im_end|>\n`, `:276-280`).
`ChatState::new` (`:177`) seeds the buffer with `<|startoftext|>` and
`LFMModality::Text` flags (`:179-182`).

**Audio-in / mel** (`add_audio`, `:247`): asserts `wave` is `(1, L)` mono
(`:249-260`), resamples to 16 kHz via `resample_16k` (`:271`, the faithful
windowed-sinc `crate::resample` port of `torchaudio.functional.resample`,
shared with `data::mapper`), then delegates to `add_audio_16k` (`:210`):
`mel = self.proc.audio.forward(wave)` → `(1, 128, F)`, `new_audio_in = mel.i(0)`
→ `(128, F)`, `emb_len = mel2emb_len(frames)` (ceil(F/8)), three appends grow
`audio_in` (dim 1), `modality_flag` (dim 1), `audio_in_lens` (dim 0). The
modality run appended is `LFMModality::AudioIn` repeated `emb_len` times
(`:215`). The raw mel **frame count** F (not the embedding count) is what is
appended to `audio_in_lens` (`:216`); the conformer re-derives the subsampled
length from it.

**`append`** (`:288`) is how generated tokens re-enter the state: it asserts
`text` is one row, `audio_out` has exactly `codebooks` (=8) rows, `modality_flag`
is one row, and the **key invariant** `n_flag == n_text + n_audio` (`:304-307`).
The state carries I64; incoming ids (U32 from the generation loop) are cast to
I64 to match (`:311-312`). On the first append, the empty `audio_out`
placeholder is **replaced** (not cat'd) to avoid a zero-length cat on Metal
(`:315-316`).

**`decode()`** (`:138`) is the only output-side compute dispatch. It
range-checks `0 ≤ code ≤ 2047` (rejecting the EOAudio sentinel **2048**, which
the caller must strip) via a `max` over the u32-cast codes (`:142-151`), then
calls `self.audio_out.as_ref().or(self.mimi.as_ref())?.decode(audio_codes)`.
The processor dispatches through the `AudioDetokenizer` trait — it doesn't
know which concrete backend it holds.

No sampling, RoPE, norm, or attention lives in this file — those live in the
model/conformer/codec components it dispatches to.

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| `encode` / `add_text` | `&str` | I64 `(1, n)` token ids |
| `add_audio` resample | f32 `(1, L)` @ `sampling_rate` | f32 `(1, L')` @ 16 kHz, `L'=ceil(L·16000/sr)` |
| mel front-end (`audio.forward`) | f32 `(1, L')` | mel computed in f32 (chain), returned f32 `(1, 128, F)`; `mel.i(0)` → `(128, F)` |
| `audio_in_lens` append | — | I64 `(k,)`, each entry = raw mel frame count F |
| `modality_flag` (audio) | — | I64 `(1, mel2emb_len(F))` filled `AUDIO_IN=2` |
| `append` (generated) | text I64/U32 `(1,t)`, audio_out I64/U32 `(8,a)`, flag I64/U32 `(1,t+a)` | grows `ChatState` buffers (cast to I64) |
| `decode` | codes `(1, 8, T)`, values 0..2047 (u32-checked) | f32 waveform `(1, T')` @ 24 kHz |

Internal promotions: tokenizer ids are **I64** (`:129`) and every id-derived
field inherits it (`audio_out`, `modality_flag`); the mel chain runs in **f32**
on device (the `FilterbankFeatures` precision pin, with window/filterbank/twiddles
computed in f64 then stored f32); codes are checked as **u32** then passed to
the detokenizer. Weights on disk are bf16; Rust CPU compute promotes to f32 (no
CPU bf16 matmul), Metal stays bf16.

## Wiring (Rust)
**Upstream (feeds `ChatState`):**
- mic/file wav f32 `(1, L)` → `add_audio` → resampled to 16 kHz → routed to the
  mel front-end `FilterbankFeatures` as f32 `(1, L')`. See
  [`glm-version/model/conformer/processor.md`](model/conformer/processor.md).
- generated text token (I64/U32) + audio frame `(8,)` int + modality flag from
  `lfm2_audio.rs::generate_interleaved` → `append`. See
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).
- `LFMModality` enum, `mel2emb_len`, `get_model_dir` from `utils.rs`. See
  [`glm-version/utils.md`](utils.md).

**Downstream (consumes processor / `ChatState` output):**
- The five-tensor `ChatState` bundle (`text` I64 `(1,L)`, `audio_in` f32
  `(128,ΣF)`, `audio_in_lens` I64 `(k,)`, `audio_out` I64 `(8,m)`,
  `modality_flag` I64 `(1,L)`) → `LFM2AudioModel::prefill_inputs`. See
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).
- `decode((1,8,T))` codes → the LFM2 detokenizer (LFM2.5) for ISTFT vocoding
  (see [`glm-version/detokenizer.md`](detokenizer.md)), or → `MimiDetokenizer`
  (`audio_out.rs`, v1/demo streaming fallback) → f32 `(1,T')` @24 kHz.
- `mimi.encode` (data prep) routes to `MimiDetokenizer` for building `audio_out`
  targets. See `glm-version/data/mapper.md`.

## Python ↔ Rust — where the port differs

| Python (`processor.py`) | Rust (`processor.rs`) | Difference | Why |
|---|---|---|---|
| `AutoTokenizer.from_pretrained` (HF) | `tokenizers::Tokenizer::from_file` reading `tokenizer.json` directly (`:115`) | **deliberate: `tokenizers` crate** | no HF `transformers` dep in the Rust port; the `tokenizers` crate reads the same `tokenizer.json`. |
| `LFM2AudioProcessor.from_pretrained` | `from_pretrained` delegates to `crate::loader::from_pretrained` (`:109`) | **deliberate: shared loader** | the loader builds both model + processor in one pass; `from_pretrained` drops the model and returns the processor (matching the Python classmethod's return). No loader logic is duplicated. |
| `_mimi` / `_audio_detokenizer` lazy `@property` singletons | `mimi` / `audio_out`: `Option<Box<dyn AudioDetokenizer>>` fields (`:70-76`) | **deliberate: trait objects, not lazy properties** | Rust has no `@property`; the backends are built at load time and held as `Option<Box<dyn …>>`. The `AudioDetokenizer` trait unifies both; `decode` dispatches `audio_out.or(mimi)`. |
| `to`/`eval`/`train` | no-op stubs | **deliberate: no-op** | candle places dtype/device at load; inference is always eval. |
| `tokenizer.encode(..., return_tensors="pt")` → int64 | `tokenizer.encode(text, false)` → `Vec<i64>` → `Tensor` I64 (`:124-131`) | identical (I64) | — |
| `torchaudio.functional.resample` | `crate::resample::resample` (faithful windowed-sinc, `:271`) | **deliberate: in-tree port** | §2.7. `sinc_interp_hann`, width 6, rolloff 0.99. Shared with `data::mapper`. |
| `add_audio` resamples inline | `add_audio` → `resample_16k` → `add_audio_16k` (`:247-264`) | **deliberate: split** | the resample is split out so the parity-tested mel path (`add_audio_16k`) is shared and unchanged. |
| `torch.empty((128,0))` empty init | 1-element buffer `narrow` to length 0, **replaced** (not cat) on first add (`:187-196`, `:219-220`, `:315-316`) | **deliberate: no zero-size buffer on Metal** | candle can't allocate a zero-size buffer on Metal; a valid 1-col buffer narrowed to 0 reports 0 elements and is replaced on the first add, so no zero-size buffer is ever created. |
| device/dtype hardcoded `cuda`/`bf16`, `.cuda()` on the detok | device/dtype-agnostic via `Device`/`DType` args (`:109`) | **deliberate: device-agnostic** | §2.1. The Python hard-codes `device="cuda"` and `.cuda()` on the detok (`processor.py:151`) — won't boot CPU-only. Rust takes `device`+`dtype`, defaults `(Cpu, F32)`, Metal opt-in. |
| `FusedEmbedding` (detok) `.mean(1)` vs model `.sum(0)` | (routed, not implemented here) | — | the processor routes codes to whichever backend; the detok's `FusedEmbedding` is in `detokenizer.rs`. See [`glm-version/detokenizer.md`](detokenizer.md). |

## Precision / gotchas (Rust-specific)
- **Two output rates.** Audio-IN mel runs at **16 kHz** (`add_audio` resamples
  to 16 kHz, `:271`); audio-OUT (Mimi/detok) is **24 kHz**. Do not confuse the
  codec rate with the mel rate.
- **EOAudio = 2048.** `decode` rejects codes ≥ 2048 (`:147-151`); the EOAudio
  sentinel must be stripped from the last frame before decode. The model's
  fused audio vocab is **2049** per codebook (offset stride 2049); the *detok's*
  `FusedEmbedding` vocab is **2048** (offset stride 2048) — different tables.
- **sum vs mean.** Model prefill embeds codes with `.sum(0)` (stride 2049); the
  detok embeds with `.mean(1)` (stride 2048). The processor routes to whichever
  — they are not interchangeable.
- **`audio_in_lens` stores raw mel frames F, not embedding length.** The
  conformer re-derives `mel2emb_len(F)=ceil(F/8)` itself; storing the embedding
  length would double-subsample. Smallest valid mel length for the encoder is 9.
- **mel cast order.** The mel is computed in f32 (the `FilterbankFeatures`
  precision pin) and stored as f32 in `ChatState.audio_in` (`:192` placeholder
  is f32). The Python casts to bf16 after the chain (`processor.py:238`); the
  Rust port keeps f32 (matching the f32 CPU parity path). On Metal the caller
  may cast to bf16 before the conformer (`lfm2_audio.rs:682`).
- **I64 throughout.** All id-bearing fields are I64 (`:129`, `:311-312`); the
  generation loop hands back U32 sampled tokens, so `append` re-casts incoming
  ids to I64 to match the buffer. Don't narrow to U32 — candle's
  `index_select`/embedding accept I64 and there's no reason to narrow.
- **Empty-buffer init (Metal).** `ChatState::new` (`:187-196`) uses 1-element
  buffers `narrow` to length 0 rather than `zeros((nfilt, 0))` — candle can't
  allocate a zero-size buffer on Metal. The first `add_audio_16k`/`append`
  **replaces** (not cat'd) the placeholder (`:219-220`, `:315-316`).
- **`add_audio` split.** `add_audio` → `resample_16k` → `add_audio_16k` so the
  parity-tested mel path is shared. Don't inline the resample back into
  `add_audio_16k` — the split keeps the mel path testable in isolation.
- **Prefill invariant is load-bearing.** `modality_flag` length must equal
  `text_len + audio_out_len` per `append` (`:304-307`), and the per-modality
  `.sum()`s must match each source tensor's length or `prefill_inputs` errors
  (`lfm2_audio.rs:747`) — the modality scatter (`index_select`) silently
  mis-aligns otherwise.
- **`decode` dispatch order.** `audio_out.as_ref().or(mimi.as_ref())` (`:156`):
  the LFM2 detokenizer takes precedence when present (LFM2.5); Mimi is the v1
  fallback. Both are loaded independently so a full snapshot keeps both.

## Cross-references
- [`ARCH/processor.md`](../ARCH/processor.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.7 (resample
  port).
- `liquid-audio-rs/src/loader.rs` — `from_pretrained` (shared model+processor
  loader).
- `liquid-audio-rs/src/audio_out.rs` — the `AudioDetokenizer` trait + both
  backends.
- `liquid-audio-rs/src/resample.rs` — the windowed-sinc resampler.