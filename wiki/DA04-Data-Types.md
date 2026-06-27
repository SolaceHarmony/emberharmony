<!-- topic: Data & Training -->
# DA04 · Data types
**Code:** `DA04` · **Source:** `data/types.py` · **Rust:** `data/types.rs` · **On the LFM2-Audio inference path:** no

## Role
`data/types.py` is the value-type vocabulary of the training/data subsystem — pure
`@dataclass` containers, no compute. It defines the *input* description language
(`ChatMessage` + the `TextSegment | AudioSegment | InterleavedSegment` union that
describes one conversation turn) and the three structurally-identical six-tensor
bundles that move a packed example through the data pipeline (`LFM2AudioTrainingSample`
→ `LFM2AudioRow` → `LFM2AudioModelInput`). It exists only to give the loss/training
path a typed, named contract; nothing here runs at inference time (the inference path
uses `ChatState` in `processor.py`, not these dataclasses).

## How it works
There is no forward pass, normalization, attention, or convolution in this file — it is
all dataclass declarations. The "mechanism" is the **field schema and the pipeline
staging**, which the neighbors enforce.

**Chat-content segments (py 9–34).** Three frozen, slotted dataclasses each pin a
`kind: Literal[...]` discriminator to a single string and carry their payload:
`TextSegment.text: str` (py 10–11), `AudioSegment.audio: bytes` (py 16–18, raw
*encoded* audio bytes — decoded later by `soundfile` in `mapper.py`),
`InterleavedSegment` carries both `text` and `audio` (py 22–25) for assistant
real-time S2S turns. `ChatContentSegment = TextSegment | AudioSegment | InterleavedSegment`
(py 34) is a PEP-604 sum type; consumers branch on it with `isinstance` (e.g.
`mapper.py:56/67/75`). `ChatMessage` (py 28–31, `kw_only=True`) is `{role: Literal["user"|"system"|"assistant"], content: list[ChatContentSegment]}` — the one type that crosses
into `preprocess.py` and `mapper.py` as the dataset's raw row.

**The six-tensor bundle (py 37–67).** All three of `LFM2AudioTrainingSample`,
`LFM2AudioRow`, `LFM2AudioModelInput` declare the *identical* six fields:
`text, audio_in, audio_in_lens, audio_out, modality_flag, supervision_mask`. They are
distinct named types to mark three pipeline stages, not three different schemas:
- **`LFM2AudioTrainingSample`** = one packed example, produced by
  `LFM2AudioChatMapper.__call__` (`mapper.py:121`). The mapper fixes the per-field
  dtypes: `text` → `torch.long`, shape `(1, L)` (`mapper.py:110`); `audio_in` → mel
  features `torch.float32`, shape `(128, ΣT)` i.e. 128 mel bins concatenated over input
  segments along time (`mapper.py:111`); `audio_in_lens` → `torch.long`, shape `(n_seg,)`
  (`mapper.py:112`); `audio_out` → `torch.long`, shape `(codebooks, L_ao)` = the
  Mimi-encoded output codes + EOAudio (`mapper.py:113-117`); `modality_flag` →
  `torch.long`, shape `(1, L)`, per-position `LFMModality` enum (`mapper.py:118`);
  `supervision_mask` → `torch.bool`, shape `(1, L)`, marking loss-bearing positions
  (`mapper.py:119`).
- **`LFM2AudioRow`** = one *padded* row read back from the Arrow dataset by
  `LFM2DataLoader.__getitem__` (`dataloader.py:48`). Same dtypes are re-asserted via
  `torch.as_tensor(..., dtype=...)` (`dataloader.py:30-35`), then `text`,
  `modality_flag`, `supervision_mask` are right-padded to `context_length=4096` with
  `F.pad` (`dataloader.py:44-46`); `modality_flag` pads with `LFMModality.TEXT`,
  `supervision_mask` pads with `False` so padding never enters the loss. `audio_in`
  / `audio_in_lens` / `audio_out` are *not* padded here (variable, concatenated later).
- **`LFM2AudioModelInput`** = the collated batch, built by `lfm2_collator`
  (`dataloader.py:68`). Batching is **concatenation, not stacking**: `text`,
  `audio_out` cat on `dim=1` (time), `modality_flag`, `supervision_mask` cat on `dim=0`
  (batch rows), `audio_in` cat on `dim=1`, `audio_in_lens` cat on `dim=0`
  (`dataloader.py:59-66`). This is the only one of the three that carries behavior: a
  `to(device)` method (py 69-77) that moves every field with `.to(device)`.

**The schema is load-bearing — it is asserted downstream.** `LFM2AudioModel._prefill`
(consuming the same fields) hard-asserts the contract: `len(audio_in_lens.shape)==1`,
`len(modality_flag.shape)==2`, `audio_in.shape[0]==128`, `audio_out.shape[0]>=codebooks`,
and the cross-field counting identities `(modality_flag==TEXT).sum()==text.shape[1]`,
`(modality_flag==AUDIO_OUT).sum()==audio_out.shape[1]`,
`(modality_flag==AUDIO_IN).sum()==mel2emb_len(audio_in_lens).sum()`,
`audio_in.shape[1]==audio_in_lens.sum()` (`lfm2_audio.py:317-331`). So `modality_flag`
is the index that scatters `text`/`audio_in`/`audio_out` embeddings into the backbone
input — these dataclasses are the carrier of that alignment.

## Dtypes & shapes
Per-field, as produced/consumed (B = batch rows, L = padded seq len 4096, ΣT = total
input-mel frames, L_ao = output-code length, n_seg = number of audio-in segments,
codebooks = 8):

| Field | dtype | shape | meaning |
|---|---|---|---|
| `text` | int64 (`torch.long`) | `(1, L)` sample → `(B, L)` after collate | text token ids |
| `audio_in` | f32 | `(128, ΣT)` | mel features, 128 bins × concat time (mel computed f32/f64, materialized f32 here) |
| `audio_in_lens` | int64 | `(n_seg,)` | per-segment input-mel frame counts |
| `audio_out` | int64 | `(codebooks, L_ao)` (≥8 rows) | Mimi output codes + EOAudio (2048) |
| `modality_flag` | int64 | `(1, L)` → `(B, L)` | per-position `LFMModality` enum (TEXT/AUDIO_IN/AUDIO_OUT) |
| `supervision_mask` | bool | `(1, L)` → `(B, L)` | loss-bearing positions |

No internal promotions happen *in this file* (no math). The dtypes are fixed by the
producers: `mapper.py` casts to `long`/`float32`/`bool`; `dataloader.py` re-casts on
read; `preprocess.py` Arrow schema stores `int64` for text/audio_out/audio_in_lens/
modality_flag, `float32` for audio_in, `bool` for supervision_mask
(`preprocess.py:26-29`). Note `modality_flag` is `bool` *semantically* but stored as
`int64` because it is an enum, not a flag.

## Wiring
**Upstream (producers):**
- [`LFM2AudioChatMapper`](DA02-Chat-Mapper) consumes `list[ChatMessage]` and emits
  `LFM2AudioTrainingSample` — edge: int64 `text (1,L)` + f32 `audio_in (128,ΣT)` +
  int64 `audio_in_lens/audio_out/modality_flag` + bool `supervision_mask`.
- [`preprocess_dataset`](DA03-Preprocess-Arrow) consumes `Iterable[list[ChatMessage]]`,
  re-serializes each `LFM2AudioTrainingSample`'s six fields to an Arrow `Features`
  schema (`preprocess.py:24-30`) — edge: the same six fields as nested int64/float32/
  bool sequences.
- [`LFM2DataLoader`](DA01-DataLoader) reads the Arrow row back and emits `LFM2AudioRow`
  (padded to 4096) — edge: same six fields, `text`/`modality_flag`/`supervision_mask`
  now `(1, 4096)`.
- [`lfm2_collator`](DA01-DataLoader) consumes `list[LFM2AudioRow]` and emits
  `LFM2AudioModelInput` — edge: six fields concatenated to `(B, …)`.

**Downstream (consumers):**
- [`LFM2AudioModel`](MD01-LFM2AudioModel) — `.logits(batch)` / `.forward(batch)`
  consume `LFM2AudioModelInput`; `_prefill` scatters `text`/`audio_in`/`audio_out`
  embeddings by `modality_flag` and `forward` builds the CE loss masks from
  `supervision_mask` (`lfm2_audio.py:393-413`). Edge: int64 `text (B,L)`, f32
  `audio_in (128,ΣT)`, int64 `modality_flag (B,L)`, bool `supervision_mask (B,L)`.
- [`Trainer`](CO04-Trainer) — `train_step(batch: LFM2AudioModelInput)` /
  `validate` move the batch with `.to(device)` and call `self.model(batch)`
  (`trainer.py:171`). Edge: the whole `LFM2AudioModelInput` bundle.

## Python ↔ Rust
Symbol-level (`data/types.py` → `data/types.rs`):
- `TextSegment` / `AudioSegment` / `InterleavedSegment` / `ChatMessage` →
  same-named Rust structs with private fields + accessors (the `frozen=True`
  immutability becomes "owning constructor + read-only getters").
- `kind: Literal[...]` discriminators → the `SegmentKind` enum (`Text`/`Audio`/
  `Interleaved`) with `as_str()` returning the wire string; `role: Literal[...]` →
  the `Role` enum.
- `ChatContentSegment = A | B | C` (PEP-604 union) → the `ChatContentSegment` Rust
  `enum` with `From<…>` impls and a `kind()` reader.
- `audio: bytes` → `Vec<u8>`.
- `LFM2AudioTrainingSample` / `LFM2AudioRow` / `LFM2AudioModelInput` → three Rust
  structs holding six candle `Tensor` fields each.

**Deliberate divergences** (PYTHON_VS_RUST.md §2.1, §2.7; PORT_STATUS.md):
- **`to(device)` is added to all three bundles in Rust, not just `LFM2AudioModelInput`.**
  Python only `LFM2AudioModelInput` has `to` (py 69); the Rust gives
  `LFM2AudioTrainingSample` and `LFM2AudioRow` a `to(&Device)` too (`types.rs:350,397`),
  because candle is device-agnostic / explicit-placement (PYTHON_VS_RUST.md §2.1) so a
  per-field move is the real device-transfer for every bundle — the Python relied on the
  whole pipeline being implicitly on `cuda`.
- **`LFM2AudioModelInput` is defined in `model/lfm2_audio.rs`, re-exported from
  `data/types.rs`** (`types.rs:19-23`), mirroring that the Python `model/lfm2_audio.py`
  imports it from `data/types.py` — one canonical type, defined where it is consumed.
- The data pipeline that *fills* these (off-path): `soundfile.read` → symphonia, HF
  `datasets.save_to_disk` → real Arrow IPC, `torchaudio.resample` → windowed-sinc
  (PYTHON_VS_RUST.md §2.7) — but those are the producers, not this file.
- This whole file is **off the inference path** (PORT_STATUS.md "training loss" `// PORT:`
  note): the bundles feed `LFM2AudioModel.forward`/`logits`, i.e. the training subsystem
  outside the inference port; the types are provided for inventory completeness (38/38,
  170/170 symbols).

## Precision / gotchas
- **`audio_in` is f32 here, but the mel that fills it is precision-sensitive.** The mel
  front-end (`conformer/processor.py`) computes in f32/f64 ("not robust to low
  precision", repaired to f64 on CPU — PYTHON_VS_RUST.md §1.4) and only *materializes*
  f32 in this bundle; in `ChatState` (inference) the same mel is stored bf16. Do not
  conflate the storage dtype here (f32) with a license to compute mel in low precision.
- **`modality_flag` is an int64 enum, not a boolean.** It carries the three
  `LFMModality` values; storing it as `bool` in Arrow would collapse AUDIO_IN/AUDIO_OUT.
  `supervision_mask` *is* the real bool. The two are often `logical_and`-ed downstream
  (`lfm2_audio.py:398/409`).
- **Off-by-one / shift in the loss masks (consumer-side, but the contract lives here).**
  `supervision_mask` is combined with the modality masks and sliced `[:, 1:]` for the
  next-token target while the shifted-input mask keeps the full length
  (`lfm2_audio.py:398-413`) — so a padded position must be `supervision=False` (enforced
  at `dataloader.py:46`) or it would leak into the CE loss.
- **EOAudio = 2048** lives in `audio_out` (code values `0..2048`, `2048` = end-of-audio),
  appended by the mapper; the bundle just transports it. `audio_out` having `>= codebooks`
  rows (asserted `lfm2_audio.py:326`) allows extra delay/EOS rows beyond the 8 codebooks.
- **Batching is concat, not stack** (`dataloader.py:59-66`): `modality_flag`/
  `supervision_mask` grow on `dim=0` but `text`/`audio_out` grow on `dim=1` — an easy
  axis-swap bug if reimplemented. The Rust must preserve the per-field axis exactly.
