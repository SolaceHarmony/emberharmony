# data_dataloader (Rust port)
**Source:** `liquid-audio-rs/src/data/dataloader.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/data/dataloader.py` · **On the LFM2-Audio inference path:** no

> Companion to [`ARCH/data/dataloader.md`](../../ARCH/data/dataloader.md). The
> original is already Rust-aware; this is the Rust-first version.

## Role
`LFM2DataLoader` (`dataloader.rs`) is a map-style loader over a HuggingFace
Arrow dataset of *pre-packed* training rows in the Rust port, plus
`lfm2_collator`, the batch-collation function that assembles an
`LFM2AudioModelInput` from a list of rows. It exists purely for the **training**
subsystem: each on-disk row already holds the fully-tokenized/encoded sequences
produced upstream by `data/mapper.rs`. This component's only real work is (a)
dtype-casting the Arrow columns to candle tensors and (b) right-padding the
per-position sequences to a fixed `context_length`. It is **not on the
inference path** — `processor.rs`/`ChatState` build model inputs at chat time;
this loader feeds `trainer.rs`.

## How it works (Rust)
The loader is a thin wrapper; the mechanism is in `get` and `lfm2_collator`.

**Construction** (`LFM2DataLoader::new`, `:72`-ish) stores `dataset_path` and
`context_length` (default `4096`) and calls `load_from_disk` (`:89`) →
`crate::data::arrow_io::load_from_disk`, reading real Arrow IPC +
`state.json`/`dataset_info.json` (§2.7: HF `datasets` → pure-Rust `arrow`
backend, same on-disk format). `len`/`is_empty` forward to the dataset.

**`get(idx)` — cast + pad:**
1. `row = self.dataset[idx]` yields a `RawRow` (`:38`) — net-new scaffolding: the
   per-row record an Arrow reader yields *before* padding. Python re-reads
   `self.dataset[idx]` each call; Rust owns decoded rows in a `Vec`. Same
   observable order/semantics.
2. **dtype casts** (`:127-135`), one per column: `text` → I64, `audio_in` → F32,
   `audio_in_lens` → I64, `audio_out` → I64, `modality` → I64, `supervision` →
   U8. These are the *normative* dtypes for the whole training pipeline.
3. **pad length** `pad_len = context_length - modality.shape[1]`. If negative,
   `Err` (`:141-148`) — a sample longer than `context_length` is a hard error,
   never truncated.
4. **right-pad the three per-position sequences only** (`:151-159`, `:183-193`):
   - `text`: `pad_with_zeros` (value 0).
   - `modality`: **hand-built `pad_right_with`** (cat with a constant
     `LFMModality::Text as i64` tensor) — candle has no general value-fill pad,
     so the non-zero TEXT pad needs a cat with a constant tensor.
   - `supervision`: `pad_with_zeros` (value 0 = False).
5. `audio_in`, `audio_in_lens`, `audio_out` returned **unpadded**.
6. Returns a `LFM2AudioRow`.

**`lfm2_collator(batch)` — concat, not stack** (`:210-231`): `Tensor::cat` along
carefully chosen dims:
- `audio_in`: `cat(dim=1)` — frames axis; `(128, ΣT_batch)`.
- `audio_in_lens`: `cat(dim=0)` — 1-D concat of all segment counts.
- `text`: `cat(dim=1)` — rows are `(1, ctx)` → `(1, B·ctx)`.
- `audio_out`: `cat(dim=1)` — `(codebooks, Σm_batch)`.
- `modality_flag`: `cat(dim=0)` — `(1, ctx)` rows stack into `(B, ctx)`.
- `supervision_mask`: `cat(dim=0)` — `(B, ctx)`.

Output is `LFM2AudioModelInput` (`types.rs:33`), which has a `.to(&Device)`
mover.

## Dtypes & shapes (Rust)
| Stage | Input | Output |
|---|---|---|
| `get` text | Arrow int | I64 `(1, n)` → padded `(1, ctx)` |
| `get` audio_in | Arrow float | F32 `(128, ΣT)` (unpadded) |
| `get` audio_in_lens | Arrow int | I64 `(k,)` (unpadded) |
| `get` audio_out | Arrow int | I64 `(codebooks, m)` (unpadded) |
| `get` modality_flag | Arrow int | I64 `(1, n)` → padded `(1, ctx)`, pad=TEXT(1) |
| `get` supervision_mask | Arrow bool | U8 `(1, n)` → padded `(1, ctx)`, pad=0(False) |
| `lfm2_collator` (B rows) | `B × LFM2AudioRow` | `text` I64 `(1, B·ctx)`, `audio_in` F32 `(128, ΣT_B)`, `audio_in_lens` I64 `(Σk,)`, `audio_out` I64 `(codebooks, Σm)`, `modality_flag` I64 `(B, ctx)`, `supervision_mask` U8 `(B, ctx)` |

## Wiring (Rust)
**Upstream:** `data/mapper.rs` produces every Arrow column read in `get`; the
Arrow schema/on-disk write is done by `data/preprocess.rs`. See
[`glm-version/data/mapper.md`](mapper.md) and
[`glm-version/data/preprocess.md`](preprocess.md). `LFMModality` enum +
`mel2emb_len` from `utils.rs` (TEXT pad value; `audio_in_lens` consumer).

**Downstream:** `trainer.rs` — the sole consumer: iterates the loader through
`LoaderDataIter` with `lfm2_collator`, moves the `LFM2AudioModelInput` to device,
and feeds it to the model's `forward`/`logits`. See
[`glm-version/trainer.md`](../trainer.md). `model/lfm2_audio.rs` — indirect
consumer (via trainer): scatters the flattened `text`/`audio_in`/`audio_out`
back into modality slots using `modality_flag`, computes loss masked by
`supervision_mask`.

## Python ↔ Rust — where the port differs

| Python (`dataloader.py`) | Rust (`dataloader.rs`) | Difference | Why |
|---|---|---|---|
| `load_from_disk(path)` (HF `datasets`) | `LFM2DataLoader::load_from_disk` → `arrow_io::load_from_disk` (`:89`) | **deliberate: pure-Rust `arrow`** | §2.7. Same on-disk format (Arrow IPC + `state.json`/`dataset_info.json`), no Arrow→custom-schema drift. |
| `F.pad(x, (0, pad_len), value=v)` | `Tensor::pad_with_zeros` (value 0) + hand-built `pad_right_with` (non-zero TEXT pad, `:151-159`, `:183-193`) | **deliberate: hand-built value pad** | candle has no general value-fill pad; the non-zero `modality`=TEXT pad needs a cat with a constant tensor. |
| `lfm2_collator(batch)` | free fn `lfm2_collator(&[LFM2AudioRow])` (`:210-231`) | identical (same `cat` dims) | — |
| `self.dataset[idx]` (re-read each call) | `RawRow` (`:38`) — decoded rows owned in a `Vec` | **deliberate: owned rows** | net-new scaffolding; same observable order/semantics. |
| `supervision_mask: torch.bool` | `supervision_mask: Tensor` U8 0/1 (`:130-137`, `:159`) | **deliberate: U8** | candle has no bool dtype; `False` pad is still a 0-fill so `pad_with_zeros` stays faithful. |
| `text`/`modality`/`audio_out` int64 | kept I64, **not narrowed to U32** (`:127-135`) | **deliberate: I64** | candle `index_select`/`embedding` accept I64; narrowing to U32 and back would be lossy. The model casts to U32 only at the ops that need it. §2.1. |
| device: CPU (Python leaves them on CPU) | device: passed-in `Device` (`:69`, `:79`) | **deliberate: device-agnostic** | §2.1. Padded tensors are built on the passed-in device. |
| `__getitem__` raises on overlong | `Err` on overlong (`:141-148`) | identical | a row whose `modality.shape[1] > context_length` is a hard error, never truncated. |

## Precision / gotchas (Rust-specific)
- **Pad values are correctness-inert by design, but must be valid indices.**
  Tail positions get `text`=0, `modality`=TEXT(1), `supervision`=0(False).
  Because supervision is False there, the loss ignores them — but
  `modality`=TEXT and `text`=0 must still be a legal text-embedding lookup.
- **`LFMModality` is 1-indexed** (`auto()` from 1): TEXT=1, AUDIO_IN=2,
  AUDIO_OUT=3. The Rust `LFMModality::Text as i64 == 1` must mirror this.
- **Overlong samples `Err`, never truncate** (`:141-148`). Packing must respect
  4096 upstream in `mapper.rs`/`preprocess.rs`.
- **Concat-not-stack semantics.** `text`/`audio_in`/`audio_out` collate along
  dim=1 (flattened), while `modality_flag`/`supervision_mask` collate along dim=0
  (true batch). The model relies on `modality_flag` + `audio_in_lens` to
  re-segment the flattened streams — a wrong cat dim silently corrupts the
  batch with no shape error if `B` and lengths happen to align.
- **`supervision_mask` is U8, not bool.** candle has no bool dtype; `0`/`non-0`
  is the semantic. Consumers cast to U8 and compare.
- **EOAudio lives in `audio_out`, not handled here.** The audio-out codes
  already include the EOAudio sentinel (2048) appended by `mapper.rs`; this
  loader treats it as an opaque I64 — no special-casing.
- **`RawRow` is owned, not re-read.** Python re-reads `self.dataset[idx]` each
  call; Rust owns decoded rows in a `Vec` (`:38`). Same observable semantics.

## Cross-references
- [`ARCH/data/dataloader.md`](../../ARCH/data/dataloader.md) — Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.7 (data
  pipeline — HF `datasets` → pure-Rust `arrow`).
- `liquid-audio-rs/src/data/arrow_io.rs` — the Arrow IPC reader.