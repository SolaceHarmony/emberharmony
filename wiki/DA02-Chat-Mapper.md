<!-- topic: Data & Training -->
# DA02 · LFM2AudioChatMapper
**Code:** `DA02` · **Source:** `data/mapper.py` · **Rust:** `data/mapper.rs` · **On the LFM2-Audio inference path:** no

## Role
`LFM2AudioChatMapper` turns one chat (a `list[ChatMessage]`) into one packed supervised **training** sample (`LFM2AudioTrainingSample`). It is the single-sample front of the data pipeline: it linearizes a multi-turn, multi-modal conversation into the six parallel tensors the trainer consumes — interleaved text token ids, audio-in mel features, audio-out codec codes, per-segment mel lengths, a per-position modality flag, and a per-position supervision (loss) mask. It is the **offline, whole-conversation twin** of `processor.py`'s `ChatState` (which assembles inference inputs turn-by-turn); it never runs on the inference path. It is invoked only from `data/preprocess.py` (`mapper(messages)`) to materialize an Arrow dataset.

## How it works
`__call__` (`mapper.py:30`) walks the chat and appends to six Python list accumulators (`text_parts`, `mel_parts`, `audio_out_parts`, `audio_in_lens`, `modality_seq`, `supervision_seq`), then concatenates them. The token stream is a ChatML-style layout, and every helper keeps `modality_seq`/`supervision_seq` in lock-step with the **embedding** count it contributes (not always the token count — audio-in differs; see below).

**Prologue / framing tokens.** Emit `<|startoftext|>` once, unsupervised (`mapper.py:38`). Per message: emit `<|im_start|>{role}\n` unsupervised (`mapper.py:47`), then the content segments, then `<|im_end|>\n` supervised iff `role == "assistant"` (`mapper.py:102`). All framing text goes through `_append_text` (`mapper.py:166`): tokenize with `processor.text.encode(..., add_special_tokens=False, return_tensors="pt").squeeze(0)`, push the `(n,)` row, and extend both sequences by `n` with `LFMModality.TEXT` and the given `supervised` flag.

**Segment dispatch** (`mapper.py:55`):
- `TextSegment` → `_append_text`, supervised iff assistant (`mapper.py:67`).
- `AudioSegment` (`mapper.py:75`): decode bytes to a mono f32 wav. If assistant, first emit a supervised `<|audio_start|>` text token then `_append_audio_out`; if user/system, `_append_audio_in`. So the **same audio bytes** are treated as a *target* (codec codes) when spoken by the assistant and as *input features* (mel) when spoken by the user — only the assistant path appends EOAudio and is supervised.
- `InterleavedSegment` (`mapper.py:56`): assistant-only (raises `ValueError` otherwise); produces a textaudio interleaved target.

**`_append_audio_in`** (`mapper.py:181`). Move wav to processor device, cast f32; resample to **16 kHz** with `torchaudio.functional.resample` if the file rate differs (`mapper.py:192`). Run the mel front-end `mel, mel_len = self.processor.audio(wav, wav_len)` (`mapper.py:196`) — this is `conformer_processor`'s `AudioToMelSpectrogramPreprocessor` (128 mel bins). Take the valid length `cur_len = mel_len[0]`, slice `mel[0, :, :cur_len]` to `(128, cur_len)` f32 on CPU, push it and `cur_len`. Crucially, the **modality/supervision extension uses the embedding length, not the mel length**: `n_emb = mel2emb_len(cur_len)` (`mapper.py:203`), extended with `LFMModality.AUDIO_IN` and `supervised=False`. This `8` ratio mirrors the conformer subsampling (`conformer_subsampling`, 8 striding) — `cur_len` mel frames collapse to `ceil(cur_len/8)` encoder embeddings. `mel2emb_len(l) = -(l // -8)` (`utils.py:21`) is **ceiling** division (the docstring's "floor division" is a misnomer; Python `-(l // -8)` rounds up). Audio-in is the only place where the count appended to the modality/supervision sequence does not equal the count appended to its data part — `mel_parts` carries `cur_len` mel frames while the sequences carry `ceil(cur_len/8)` positions; the trainer re-expands mel→emb downstream.

**`_append_audio_out`** (`mapper.py:207`) / **`_encode_audio_out`** (`mapper.py:223`). Cast f32, resample to the **codec rate** `processor.mimi.sample_rate` (24 kHz) if needed (`mapper.py:226`). Encode: `codes = processor.mimi.encode(wav.unsqueeze(0))[0].cpu()` (`mapper.py:229`) — Mimi expects `(B,1,L)`, returns `(B, n_q, T)`; index `[0]` → `(n_q, T)`. Keep the first `self.codebooks` (=8) rows, cast to `long` (`mapper.py:230`). Then **append one EOAudio frame**: `torch.full((codebooks,1), 2048)` concatenated on `dim=1` (`mapper.py:231`), so every audio-out clip ends with an all-`2048` column across all 8 codebooks. The Mimi codebook cardinality is 2048 (codes `0..2047`), so `2048` is an out-of-range sentinel = end-of-audio. The non-interleaved path extends the sequences by `codes.shape[1]` (= `T+1`, including EOAudio) with `AUDIO_OUT`, all supervised (`mapper.py:219`).

**`_append_interleaved_out`** (`mapper.py:130`). Tokenize `"{text}<|text_end|>"` → `(n_text,)`; encode audio → `(8, n_audio)` codes (with EOAudio). Then **round-robin** the two streams in fixed chunks: `interleaved_text_tokens=6` text positions, then `interleaved_audio_tokens=12` audio positions, repeat until both are drained (`mapper.py:153`). Each chunk extends only `modality_seq`/`supervision_seq` (both supervised) — the actual tokens/codes were already pushed whole to `text_parts`/`audio_out_parts`. The interleave pattern lives **only in the modality flag**; the data tensors stay stream-contiguous and the model uses `modality_flag` to scatter them back into the right positions. The order is text-first within each chunk; a final partial chunk takes `min(n, left)` of each.

**Epilogue / concat** (`mapper.py:110`):
- `text = cat(text_parts, 0).unsqueeze(0).long()` → `(1, L_text)` int64. Empty fallback is **not** present in Python (it would crash on an empty chat — there is always at least `<|startoftext|>`).
- `audio_in = cat(mel_parts, 1)` → `(128, ΣT_mel)` f32, else `empty((128,0))`.
- `audio_in_lens = tensor(audio_in_lens, long)` → `(n_audio_in_segments,)` int64.
- `audio_out = cat(audio_out_parts, 1).long()` → `(8, ΣT_code)` int64, else `empty((8,0))`.
- `modality_flag = tensor(modality_seq).unsqueeze(0)` → `(1, L)` int64.
- `supervision_mask = tensor(supervision_seq, bool).unsqueeze(0)` → `(1, L)` bool.

`L = Σ(text tokens) + Σ(audio-in n_emb) + Σ(audio-out frames incl. EOAudio)` and is the canonical sequence length; `modality_flag`/`supervision_mask` share it. `audio_in` and `audio_out` carry the *raw* features/codes whose total widths are `ΣT_mel` and `ΣT_code` (≠ `L`).

There is **no normalization, no attention, no RoPE, no sampling** here — this is pure data marshaling. The only numerics are the resamplers (windowed-sinc, inside torchaudio) and the mel front-end + Mimi encoder, both delegated to other components.

**`_load_audio_bytes`** (`mapper.py:234`) decodes the encoded-audio `bytes` via `soundfile.read(stream, dtype="float32", always_2d=True)`, transposes to channel-major (`data.T`), and mono-downmixes by `mean(dim=0, keepdim=True)` if multichannel → `(1, L)` f32 plus the file's native sample rate.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `_load_audio_bytes` | `bytes` (WAV/FLAC/…) | wav `(1, L)` f32, `sampling_rate: int` |
| `_append_audio_in` resample | wav `(1,L)` f32 @ sr | wav `(1,L')` f32 @ 16 kHz |
| mel front-end (`processor.audio`) | wav `(1,L')` f32, wav_len i64 | mel `(1,128,T_mel)` f32 (computed in f32/f64, slaney mel); slice → `(128,cur_len)` f32 |
| audio-in seq contribution | `cur_len` (mel frames) | `ceil(cur_len/8)` AUDIO_IN positions |
| `_encode_audio_out` resample | wav `(1,L)` f32 @ sr | wav `(1,L')` f32 @ 24 kHz |
| Mimi encode (`processor.mimi`) | wav `(1,1,L')` f32 | codes `(8, T)` → cat EOAudio → `(8, T+1)` int64 (codes `0..2047`, `2048`=EOAudio) |
| `__call__` text | token rows `(n,)` int64 | `text (1, L_text)` int64 |
| `__call__` final | — | `audio_in (128, ΣT_mel)` f32; `audio_in_lens (n_seg,)` i64; `audio_out (8, ΣT_code)` i64; `modality_flag (1, L)` i64; `supervision_mask (1, L)` bool |

Promotions/casts to note: wav forced to **f32** before both resample and mel/codec (`mapper.py:191,224`); mel internally precision-sensitive (**f32/f64**, see `conformer_processor`) but **stored f32** here; token ids and codes are **int64** in the sample (codes start as Mimi's int codes, `u32` in Rust, then widened); supervision is true **bool**.

## Wiring
**Upstream**
- `data/types.py` — `ChatMessage` / `TextSegment` / `AudioSegment` / `InterleavedSegment` carry the raw chat (`bytes` audio, `str` text). → [data_types](DA04-Data-Types)
- `processor.py` — provides `processor.text` (tokenizer, int64 ids), `processor.audio` (mel front-end → `(1,128,T)` f32) and `processor.mimi` (codec encode → `(8,T)` int codes). → [core_processor](CO01-Processor-ChatState)
- `model/conformer/processor.py` — the actual `AudioToMelSpectrogramPreprocessor` behind `processor.audio`; 16 kHz wav f32 → mel `(1,128,T)` f32. → [conformer_processor](CF04-Mel-Frontend)
- `moshi/models/compression.py` — the `MimiModel` behind `processor.mimi`; 24 kHz wav f32 → split-RVQ codes `(8,T)` int. → [moshi_compression](MM01-Mimi-Codec)
- `utils.py` — `LFMModality` enum + `mel2emb_len` (mel-frame → emb-length, ceil/8). → [core_utils](CO03-Utils)

**Downstream** (consume the `LFM2AudioTrainingSample`)
- `data/preprocess.py` — the sole caller; serializes the six tensors to an Arrow `Features` schema (`text/audio_out/modality_flag/audio_in_lens` int64, `audio_in` float32, `supervision_mask` bool) via `save_to_disk`. → [data_preprocess](DA03-Preprocess-Arrow)
- `data/dataloader.py` — reads the Arrow rows back (`text` i64 `(1,L_text)`, `audio_in` f32 `(128,ΣT_mel)`, `audio_out` i64 `(8,ΣT_code)`, `modality_flag` i64 `(1,L)` right-padded to `context_length=4096` with TEXT, `supervision_mask` bool padded False) and collates into the batched `LFM2AudioModelInput`. → [data_dataloader](DA01-DataLoader)

## Python ↔ Rust
Symbol-level (all in `data/mapper.rs`):
- `LFM2AudioChatMapper.__call__` → `call` (`mapper.rs:124`); the six list locals are bundled into a single `Acc` struct (`mapper.rs:71`) so the `_append_*` helpers can borrow them mutably — the Python pass-six-lists-by-ref signature is preserved structurally.
- `_append_text/_append_audio_in/_append_audio_out/_append_interleaved_out` → `append_text/append_audio_in/append_audio_out/append_interleaved_out` (one-to-one).
- `_encode_audio_out` → `encode_audio_out` (`mapper.rs:326`); `_load_audio_bytes` → `load_audio_bytes` (`mapper.rs:371`).
- the concat epilogue → `finish` (`mapper.rs:173`).

DELIBERATE divergences (PYTHON_VS_RUST.md "soundfile/torchaudio/datasets" section):
- **`soundfile.read` → `symphonia`** (`mapper.rs:408`): pure-Rust container decode (WAV/FLAC/OGG/AIFF/MP3 — a superset of libsndfile's formats), no C deps. Same mono-downmix (mean over channels), same `(1,L)` f32 contract.
- **`torchaudio.functional.resample` → in-tree windowed-sinc** (`crate::resample`, default `sinc_interp_hann`, width 6, rolloff 0.99) — a 1:1 port, shared with the processor.
- **`torch.bool` supervision_mask → candle `u8`** (`mapper.rs:215`): candle has no Bool dtype; stored as `u8` (model reads via `to_dtype(U8)`).
- **`empty((128,0))` hardcode → parameterized `nfilt`** (`mapper.rs:188`): Rust uses `processor.audio().nfilt()` instead of the literal `128`.
- **codes dtype**: Python keeps `long`; Rust keeps Mimi's `u32` through `encode_audio_out` then widens to `I64` in `finish` (`mapper.rs:206`). Rust also **zero-pads** kept codes up to `codebooks` rows if the codec returns fewer (`mapper.rs:346`) — a defensive widen Python omits (Python assumes ≥8 rows).
- **empty-text fallback**: Rust adds an `empty((1,0))` guard (`mapper.rs:179`) that Python lacks; in practice unreachable since `<|startoftext|>` always seeds `text_parts`.
- Device-agnostic throughout (no `processor.device`/CUDA assumption); torch tensor ops → candle.

## Precision / gotchas
- **`mel2emb_len` is ceiling, not floor.** `-(l // -8)` (`utils.py:21`) rounds up; the docstring saying "floor division" is wrong. Off-by-one here would desync `modality_flag` from the encoder's actual embedding count. Smallest valid mel length is 9 (per the docstring note).
- **The audio-in count mismatch is intentional.** `mel_parts` stores `cur_len` raw mel frames but `modality_seq`/`supervision_seq` advance by `ceil(cur_len/8)`. Consumers must use `audio_in_lens` + `mel2emb_len` to reconcile the two widths; treating `audio_in.shape[1]` as a sequence-position count is a bug.
- **EOAudio = 2048** is appended to *every* audio-out clip (interleaved and plain) and counts as a supervised AUDIO_OUT position. Mimi codebooks are size 2048 (valid codes `0..2047`); `2048` is the reserved end sentinel the model must learn to emit. Only `self.codebooks=8` of Mimi's RVQ levels are kept (`mapper.py:230`) — extra acoustic levels are dropped before EOAudio.
- **Resample targets differ by direction:** audio-**in** → 16 kHz (conformer mel rate), audio-**out** → 24 kHz (`processor.mimi.sample_rate`, the codec rate). Mixing them silently corrupts features/codes.
- **Mono downmix + f32 cast happen at decode and again before each numeric stage;** stereo files are averaged, never kept. The `data.T.copy()` is needed because soundfile returns frame-major.
- **Supervision asymmetry:** framing (`<|startoftext|>`, `<|im_start|>{role}\n`) is always unsupervised; assistant `<|im_end|>\n`, all assistant text, `<|audio_start|>`, audio-out, and interleaved positions are supervised; user/system text and **all** audio-in are unsupervised. The loss is computed only where `supervision_mask` is True.
- This component is **f32-floored where it touches numerics** only via its delegates (mel front-end is the precision-sensitive f32/f64 path; codec encode runs in model dtype). The mapper itself does no reduced-precision matmul, so no bf16-order concerns apply here.
