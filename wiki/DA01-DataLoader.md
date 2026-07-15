<!-- topic: Data & Training -->
# DA01 · LFM2DataLoader + collator
**Code:** `DA01` · **Source:** `data/dataloader.py` · **Rust:** `data/dataloader.rs` · **On the LFM2-Audio inference path:** no

## Role
`LFM2DataLoader` is a map-style PyTorch `Dataset` over a HuggingFace Arrow dataset of *pre-packed* training rows, plus `lfm2_collator`, the batch-collation function that assembles a `LFM2AudioModelInput` from a list of rows. It exists purely for the **training** subsystem: each on-disk row already holds the fully-tokenized/encoded sequences produced upstream by `data/mapper.py` (text token ids, audio-in mel features, Mimi-encoded audio-out codes, plus per-position modality flags and a supervision mask). This component's only real work is (a) dtype-casting the Arrow columns to torch tensors and (b) right-padding the per-position sequences to a fixed `context_length`. It is **not on the inference path** — `processor.py`/`ChatState` build model inputs at chat time; this loader feeds `trainer.py`.

## How it works
The loader is a thin wrapper; the mechanism is entirely in `__getitem__` (py:27) and `lfm2_collator` (py:58).

**Construction** (`__init__`, py:15) stores `dataset_path` and `context_length` (default `4096`, py:18) and calls `load_from_disk(self.dataset_path)` (py:22) to memory-map the Arrow-backed `datasets.Dataset`. `__len__` (py:24) forwards to `len(self.dataset)`.

**`__getitem__(idx)` — cast + pad (py:27-55):**
1. `row = self.dataset[idx]` (py:28) yields a dict of Arrow columns produced by `data/mapper.py`.
2. **dtype casts** via `torch.as_tensor(..., dtype)` (py:30-35), one per column. These are the *normative* dtypes for the whole training pipeline:
   - `text` → `torch.long` (int64) — token ids, shape `(1, n)`.
   - `audio_in` → `torch.float32` — mel features, shape `(n_mels=128, ΣT)`. f32, not bf16: the mel front-end (`conformer_processor`) is precision-sensitive; bf16 storage happens later in `ChatState`, not here.
   - `audio_in_lens` → `torch.long` — per-segment mel frame counts, shape `(k,)`. Consumed downstream to split the concatenated mel back into per-utterance chunks and to compute encoder output lengths via `mel2emb_len` (utils.py:15, ceil-div-by-8).
   - `audio_out` → `torch.long` — Mimi RVQ codes (+EOAudio sentinel), shape `(codebooks, m)`.
   - `modality` → `torch.long` — per-position `LFMModality` flag, shape `(1, n)`.
   - `supervision` → `torch.bool` — per-position loss mask, shape `(1, n)`.
3. **pad length** `pad_len = context_length - modality.shape[1]` (py:37). If negative, raise `ValueError` (py:38-42) — a sample longer than `context_length` is a hard error, never truncated.
4. **right-pad the three per-position sequences only** (py:44-46), using `F.pad(x, (0, pad_len), value=...)` — the `(0, pad_len)` tuple pads the *last* dim on the right:
   - `text`: pad value `0` (default) — id 0. The padded text positions are inert because…
   - `modality`: pad value `int(LFMModality.TEXT)` (py:45). `LFMModality` is an `IntEnum` with `auto()` starting at 1 → **TEXT=1, AUDIO_IN=2, AUDIO_OUT=3** (utils.py:9). Padding the tail as TEXT routes those positions through the text embedding table (id 0), the cheapest no-op modality.
   - `supervision`: pad value `False` (py:46) — padded positions are masked out of the loss, so the pad-token-0/pad-modality-TEXT choices are numerically irrelevant to the gradient; they only need to be *valid* embedding indices.
5. `audio_in`, `audio_in_lens`, `audio_out` are returned **unpadded** (variable length per row); the collator concatenates them, not pads them.
6. Returns a `LFM2AudioRow` dataclass (types.py:48).

There is **no normalization, no attention, no RoPE, no activation, no convolution, no sampling** here — this component is pure tensor plumbing. Every "model" op lives downstream.

**`lfm2_collator(batch)` — concat, not stack (py:58):** torch `cat` along carefully chosen dims (the batch is built by concatenation, not a new batch axis, because rows carry a leading singleton axis):
- `audio_in`: `cat(dim=1)` (py:59) — frames axis; `(128, ΣT_batch)`.
- `audio_in_lens`: `cat(dim=0)` (py:60) — 1-D concat of all segment counts.
- `text`: `cat(dim=1)` (py:62) — rows are `(1, context_length)` → `(1, B·context_length)`.
- `audio_out`: `cat(dim=1)` (py:63) — `(codebooks, Σm_batch)`.
- `modality_flag`: `cat(dim=0)` (py:65) — `(1,ctx)` rows stack into `(B, context_length)`.
- `supervision_mask`: `cat(dim=0)` (py:66) — `(B, context_length)`.

Note the asymmetry: `text`/`audio_in`/`audio_out` flatten on dim=1 (the model un-flattens using `modality_flag`/`audio_in_lens`), while `modality_flag`/`supervision_mask` become a true `(B, ctx)` matrix. Output is `LFM2AudioModelInput` (types.py:59), which has a `.to(device)` mover (types.py:69).

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `__getitem__` text | Arrow int → | `text` int64 `(1, n)` → padded `(1, context_length)` |
| `__getitem__` audio_in | Arrow float → | `audio_in` **f32** `(128, ΣT)` (unpadded) |
| `__getitem__` audio_in_lens | Arrow int → | int64 `(k,)` (unpadded) |
| `__getitem__` audio_out | Arrow int → | int64 `(codebooks, m)` (unpadded) |
| `__getitem__` modality_flag | Arrow int → | int64 `(1, n)` → padded `(1, ctx)`, pad=TEXT(1) |
| `__getitem__` supervision_mask | Arrow bool → | bool `(1, n)` → padded `(1, ctx)`, pad=False |
| `lfm2_collator` (B rows) | `B × LFM2AudioRow` → | `text` int64 `(1, B·ctx)`, `audio_in` f32 `(128, ΣT_B)`, `audio_in_lens` int64 `(Σk,)`, `audio_out` int64 `(codebooks, Σm)`, `modality_flag` int64 `(B, ctx)`, `supervision_mask` bool `(B, ctx)` |

No dtype promotions occur in this file (no norm/softmax/mel). The only dtype event is the explicit `as_tensor` cast of each Arrow column. Notably `audio_in` stays **f32** here (storage bf16 happens only when chat-time `ChatState` caches mel, not in training rows); `audio_out` codes are **int64** here (the Rust/Mimi path treats codes as **u32**, but this file's torch contract is int64).

## Wiring
**Upstream (feeds this):**
- [LFM2AudioChatMapper](DA02-Chat-Mapper) — produces every Arrow column read in `__getitem__`: text ids (int64 `(1,n)`), mel `audio_in` (f32 `(128,ΣT)`), `audio_in_lens` (int64), Mimi codes `audio_out`+EOAudio (int64 `(codebooks,m)`), `modality_flag` (int64), `supervision_mask` (bool). The Arrow schema/on-disk write is done by [preprocess_dataset](DA03-Preprocess-Arrow) (int64/float32/bool features).
- [data_types](DA04-Data-Types) — `LFM2AudioRow` / `LFM2AudioModelInput` dataclasses returned/produced here.
- [core_utils](CO03-Utils) — `LFMModality` enum (TEXT pad value); `mel2emb_len` is the downstream consumer of `audio_in_lens`.

**Downstream (consumes this output):**
- [trainer](CO04-Trainer) — the sole consumer: iterates the loader through a torch `DataLoader` with `lfm2_collator`, moves the `LFM2AudioModelInput` to device, and feeds it to the model's forward/loss. Edge: `LFM2AudioModelInput` (text int64 `(1,B·ctx)`, audio_in f32 `(128,ΣT)`, audio_in_lens int64, audio_out int64 `(cb,Σm)`, modality_flag int64 `(B,ctx)`, supervision_mask bool `(B,ctx)`).
- [model_lfm2_audio](MD01-LFM2AudioModel) — indirect consumer (via trainer): scatters the flattened `text`/`audio_in`/`audio_out` back into modality slots using `modality_flag`, computes loss masked by `supervision_mask`.

## Python ↔ Rust
Symbol map:
- `LFM2DataLoader.__init__/__len__/__getitem__` → `LFM2DataLoader::new` / `len` (+ `is_empty`) / `get` (dataloader.rs:72-177).
- `load_from_disk(path)` → `LFM2DataLoader::load_from_disk` → `crate::data::arrow_io::load_from_disk` (dataloader.rs:89), reading real Arrow IPC + `state.json`/`dataset_info.json` — PYTHON_VS_RUST.md §2.7 (HF `datasets` → pure-Rust `arrow` backend, same on-disk format, no Arrow→custom-schema drift).
- `F.pad(x,(0,pad_len),value=v)` → `Tensor::pad_with_zeros` for the value-0 cases (`text`, `supervision`) and a hand-built `pad_right_with` (cat with a constant tensor) for the non-zero `modality`=TEXT pad (dataloader.rs:151-159, 183-193). candle has no general value-fill pad.
- `lfm2_collator` → free fn `lfm2_collator(&[LFM2AudioRow])` with the identical `cat` dims (dataloader.rs:210-231).
- `RawRow` (dataloader.rs:38) is net-new scaffolding: the per-row record an Arrow reader yields *before* padding (Python re-reads `self.dataset[idx]` each call; Rust owns decoded rows in a `Vec`). Same observable order/semantics.

**Deliberate divergences (not bugs):**
- **bool → U8.** candle has no bool dtype, so `supervision_mask` is carried as `U8` 0/1 (dataloader.rs:130-137, 159). The one forced dtype deviation; `False` pad is still a 0-fill so `pad_with_zeros` stays faithful.
- **int64 kept, not narrowed to u32.** `text`/`modality`/`audio_out` stay `I64` (torch.long) — the port deliberately does *not* narrow to U32 at the loader (dataloader.rs:127-135), since candle `index_select`/`embedding` accept I64 and the model casts to U32 only at the ops that need it, avoiding a lossy i64→u32→i64 round-trip. Aligns with PYTHON_VS_RUST.md §2.1 (device/dtype-agnostic ports).
- **Device-agnostic.** Padded tensors are built on a passed-in `Device` (dataloader.rs:69,79); Python leaves them on CPU. PYTHON_VS_RUST.md §2.1.

## Precision / gotchas
- **Pad values are correctness-inert by design, but must be valid indices.** Tail positions get `text`=0, `modality`=TEXT(1), `supervision`=False. Because supervision is False there, the loss ignores them — but `modality`=TEXT and `text`=0 must still be a legal text-embedding lookup or the forward pass would gather an out-of-range index. Do not "optimize" the pad modality to AUDIO_OUT/AUDIO_IN, which would route pad positions through audio embedding tables expecting real codes.
- **`LFMModality` is 1-indexed** (`auto()` from 1): TEXT=1, AUDIO_IN=2, AUDIO_OUT=3 (utils.py:9). A naive C-style 0-based enum would mis-pad. The Rust port must mirror this (`LFMModality::Text as i64 == 1`).
- **Overlong samples raise, never truncate** (py:38-42 / dataloader.rs:141-148). A row whose `modality.shape[1] > context_length` is a hard `ValueError` — packing must respect 4096 upstream in `mapper.py`/`preprocess.py`.
- **Concat-not-stack semantics.** `text`/`audio_in`/`audio_out` collate along dim=1 (flattened), while `modality_flag`/`supervision_mask` collate along dim=0 (true batch). The model relies on `modality_flag` (per-position) + `audio_in_lens` to re-segment the flattened streams — a wrong cat dim here silently corrupts the batch with no shape error if `B` and lengths happen to align.
- **EOAudio lives in `audio_out`, not handled here.** The audio-out codes already include the EOAudio sentinel (code 2048) appended by `mapper.py`; this loader treats it as an opaque int64 — no special-casing.
