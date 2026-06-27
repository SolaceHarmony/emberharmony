# data_mapper (Rust port)
**Source:** `liquid-audio-rs/src/data/mapper.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/data/mapper.py` · **On the LFM2-Audio inference path:** no

> Companion to [`ARCH/data/mapper.md`](../../ARCH/data/mapper.md). The original
> is already Rust-aware; this is the Rust-first version.

## Role
`LFM2AudioChatMapper` (`mapper.rs`) turns one chat (a `Vec<ChatMessage>`) into
one packed supervised **training** sample (`LFM2AudioTrainingSample`) in the
Rust port. It is the single-sample front of the data pipeline: it linearizes a
multi-turn, multi-modal conversation into the six parallel tensors the trainer
consumes. It is the **offline, whole-conversation twin** of `processor.rs`'s
`ChatState` (which assembles inference inputs turn-by-turn); it never runs on
the inference path. It is invoked only from `data/preprocess.rs` to materialize
an Arrow dataset.

## How it works (Rust)
`call` (`mapper.rs:124`) walks the chat and appends to six accumulators bundled
into a single `Acc` struct (`:71`) so the `append_*` helpers can borrow them
mutably — the Python pass-six-lists-by-ref signature is preserved structurally.
The token stream is a ChatML-style layout; every helper keeps
`modality_seq`/`supervision_seq` in lock-step with the **embedding** count it
contributes.

**Prologue / framing tokens.** Emit `<|startoftext|>` once, unsupervised. Per
message: emit `<|im_start|>{role}\n` unsupervised, then the content segments,
then `<|im_end|>\n` supervised iff `role == "assistant"`. All framing text goes
through `append_text`: tokenize via `processor.text().encode(...)`, push the
`(n,)` row, extend both sequences by `n` with `LFMModality::Text` and the given
`supervised` flag.

**Segment dispatch:**
- `TextSegment` → `append_text`, supervised iff assistant.
- `AudioSegment`: decode bytes to mono f32 wav. If assistant, emit a supervised
  `<|audio_start|>` text token then `append_audio_out`; if user/system,
  `append_audio_in`. The **same audio bytes** are a *target* (codec codes) when
  spoken by the assistant and *input features* (mel) when spoken by the user.
- `InterleavedSegment`: assistant-only (errors otherwise); produces a
  text/audio interleaved target via `append_interleaved_out`.

**`append_audio_in`.** Resample to **16 kHz** via `crate::resample` (the
faithful windowed-sinc port of `torchaudio.functional.resample`). Run the mel
front-end `processor.audio().forward(wav)` → `(1, 128, T)`, slice to
`(128, cur_len)` F32. The **modality/supervision extension uses the embedding
length, not the mel length**: `n_emb = mel2emb_len(cur_len)` (`ceil(cur_len/8)`),
extended with `LFMModality::AudioIn` and `supervised=false`. `mel_parts` carries
`cur_len` mel frames while the sequences carry `ceil(cur_len/8)` positions.

**`append_audio_out` / `encode_audio_out`.** Resample to the **codec rate**
(`processor.mimi`'s 24 kHz). Encode: `processor.mimi.encode(wav)` → codes
`(n_q, T)`; keep the first `codebooks=8` rows, widen to I64. Append one EOAudio
frame: `Tensor::full((codebooks, 1), 2048)` concatenated on dim 1. The Mimi
codebook cardinality is 2048 (codes `0..2047`); `2048` is the out-of-range
end-of-audio sentinel. The non-interleaved path extends the sequences by
`T + 1` with `AUDIO_OUT`, all supervised.

**`append_interleaved_out`.** Tokenize `"{text}<|text_end|>"`; encode audio →
`(8, n_audio)` codes (with EOAudio). Round-robin the two streams in fixed
chunks: 6 text positions, then 12 audio positions, repeat. Each chunk extends
only `modality_seq`/`supervision_seq` (both supervised) — the actual
tokens/codes were already pushed whole to `text_parts`/`audio_out_parts`. The
interleave pattern lives **only in the modality flag**.

**Epilogue / concat (`finish`, `:173`):**
- `text = cat(text_parts, 0).unsqueeze(0)` → `(1, L_text)` I64. Empty fallback
  `(1, 0)` guard (`:179`) that Python lacks; in practice unreachable since
  `<|startoftext|>` always seeds `text_parts`.
- `audio_in = cat(mel_parts, 1)` → `(128, ΣT_mel)` F32, else
  `zeros((nfilt, 0))` (parameterized `nfilt`, not the literal 128, `:188`).
- `audio_in_lens = tensor(audio_in_lens)` → `(n_seg,)` I64.
- `audio_out = cat(audio_out_parts, 1)` → `(8, ΣT_code)` I64, else
  `zeros((codebooks, 0))`.
- `modality_flag = tensor(modality_seq).unsqueeze(0)` → `(1, L)` I64.
- `supervision_mask = tensor(supervision_seq).unsqueeze(0)` → `(1, L)` **U8**
  (candle has no bool dtype; `:215`).

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| `load_audio_bytes` | `Vec<u8>` (WAV/FLAC/…) | wav `(1, L)` F32, `sampling_rate: u32` |
| `append_audio_in` resample | wav `(1, L)` F32 @ sr | wav `(1, L')` F32 @ 16 kHz |
| mel front-end | wav `(1, L')` F32 | mel `(1, 128, T)` F32; slice → `(128, cur_len)` F32 |
| audio-in seq contribution | `cur_len` (mel frames) | `ceil(cur_len/8)` AUDIO_IN positions |
| `encode_audio_out` resample | wav `(1, L)` F32 @ sr | wav `(1, L')` F32 @ 24 kHz |
| Mimi encode | wav `(1, 1, L')` F32 | codes `(8, T)` U32 → cat EOAudio → `(8, T+1)` I64 |
| `call` text | token rows `(n,)` I64 | `text (1, L_text)` I64 |
| `call` final | — | `audio_in (128, ΣT_mel)` F32; `audio_in_lens (n_seg,)` I64; `audio_out (8, ΣT_code)` I64; `modality_flag (1, L)` I64; `supervision_mask (1, L)` U8 |

## Wiring (Rust)
**Upstream:** `data/types.rs` (`ChatMessage`/`ChatContentSegment`), `processor.rs`
(tokenizer + mel + Mimi), `model/conformer/processor.rs` (mel front-end),
`audio_out.rs::MimiDetokenizer` (codec encode), `utils.rs` (`LFMModality` +
`mel2emb_len`).

**Downstream:** `data/preprocess.rs` — the sole caller; serializes the six
tensors to an Arrow `Features` schema. `data/dataloader.rs` — reads the Arrow
rows back. See [`glm-version/data/preprocess.md`](preprocess.md) and
[`glm-version/data/dataloader.md`](dataloader.md).

## Python ↔ Rust — where the port differs

| Python (`mapper.py`) | Rust (`mapper.rs`) | Difference | Why |
|---|---|---|---|
| `soundfile.read` (libsndfile) | `symphonia` (`:408`) | **deliberate: pure-Rust decode** | §2.7. WAV/FLAC/OGG/AIFF/MP3 (a superset of libsndfile), no C deps. Same mono-downmix, same `(1, L)` F32 contract. |
| `torchaudio.functional.resample` | `crate::resample` (windowed-sinc, `sinc_interp_hann`, width 6, rolloff 0.99) | **deliberate: in-tree port** | §2.7. 1:1 port, shared with `processor.rs`. |
| `torch.bool` supervision_mask | candle **U8** (`:215`) | **deliberate: U8** | candle has no bool dtype; stored as U8 (model reads via `to_dtype(U8)`). |
| `empty((128,0))` hardcode | parameterized `processor.audio().nfilt()` (`:188`) | **deliberate: parameterized** | Rust uses the featurizer's `nfilt()` instead of the literal 128. |
| codes dtype: Python `long` | Mimi's **U32** through `encode_audio_out`, then widen to I64 in `finish` (`:206`) | **deliberate: U32 → I64** | the codec returns U32; the sample carries I64 to match the pipeline. Rust also **zero-pads** kept codes up to `codebooks` rows if the codec returns fewer (`:346`) — a defensive widen Python omits (Python assumes ≥8 rows). |
| empty-text fallback: none | Rust adds an `empty((1,0))` guard (`:179`) | **deliberate: defensive** | in practice unreachable since `<|startoftext|>` always seeds `text_parts`. |
| six list accumulators (pass-by-ref) | `Acc` struct (`:71`) | **deliberate: bundled** | the Python pass-six-lists-by-ref signature is preserved structurally; the `append_*` helpers borrow `Acc` mutably. |
| device: `processor.device` (CUDA) | device-agnostic | **deliberate** | §2.1. No CUDA assumption; torch tensor ops → candle. |

## Precision / gotchas (Rust-specific)
- **`mel2emb_len` is ceiling, not floor.** `-(l // -8)` rounds up; the docstring
  saying "floor division" is wrong. Off-by-one here would desync
  `modality_flag` from the encoder's actual embedding count. Smallest valid
  mel length is 9.
- **The audio-in count mismatch is intentional.** `mel_parts` stores `cur_len`
  raw mel frames but `modality_seq`/`supervision_seq` advance by
  `ceil(cur_len/8)`. Consumers must use `audio_in_lens` + `mel2emb_len` to
  reconcile the two widths.
- **EOAudio = 2048** is appended to *every* audio-out clip and counts as a
  supervised AUDIO_OUT position. Mimi codebooks are size 2048 (valid codes
  `0..2047`); `2048` is the reserved end sentinel. Only `codebooks=8` of Mimi's
  RVQ levels are kept — extra acoustic levels are dropped before EOAudio.
- **Resample targets differ by direction:** audio-**in** → 16 kHz (conformer
  mel rate), audio-**out** → 24 kHz (codec rate). Mixing them silently corrupts
  features/codes.
- **`supervision_mask` is U8, not bool.** candle has no bool dtype; `0`/`non-0`
  is the semantic.
- **Supervision asymmetry:** framing is always unsupervised; assistant
  `<|im_end|>\n`, all assistant text, `<|audio_start|>`, audio-out, and
  interleaved positions are supervised; user/system text and **all** audio-in
  are unsupervised. The loss is computed only where `supervision_mask != 0`.
- **`symphonia` mono-downmix.** Stereo files are averaged (`mean` over
  channels), never kept. The `data.T` transpose is needed because symphonia
  returns frame-major.

## Cross-references
- [`ARCH/data/mapper.md`](../../ARCH/data/mapper.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.7 (data pipeline — `soundfile` →
  symphonia, `torchaudio.resample` → windowed-sinc).
- `liquid-audio-rs/src/resample.rs` — the windowed-sinc resampler.