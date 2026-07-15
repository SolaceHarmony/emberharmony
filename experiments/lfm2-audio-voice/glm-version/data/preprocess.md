# data_preprocess (Rust port)
**Source:** `liquid-audio/src/data/preprocess.rs` + `liquid-audio/src/data/arrow_io.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/data/preprocess.py` · **On the LFM2-Audio inference path:** no

> Companion to [`wiki/data/preprocess.md`](../../../wiki/data/preprocess.md). The
> original is already Rust-aware; this is the Rust-first version.

## Role
`preprocess_dataset` (`preprocess.rs:85`) is the **offline dataset-build step**
for *training* in the Rust port: it consumes an iterable of chats, runs each
through an `LFM2AudioChatMapper` to pack it into an `LFM2AudioTrainingSample`,
applies an optional context-length filter, and materializes the kept samples
to disk as a real HuggingFace `datasets.Dataset` (`save_to_disk`). It is pure
plumbing — no neural net, no inference — so the expensive per-sample work
happens once at corpus-build time and the `LFM2DataLoader` can later
mmap-cheaply iterate Arrow rows. It is **not on the inference path**.

## How it works (Rust)
`preprocess_dataset` (`preprocess.rs:85-113`) is a faithful port with one
structural change: the Python `generator` lazily yields and `from_generator`
accumulates one batch internally; Rust instead collects
`kept: Vec<LFM2AudioTrainingSample>` (`:97-108`) and hands the whole vec to
`arrow_io::save_to_disk`. The skip predicate is written as the explicit
half-open range `(0..sample_len).contains(&max_context_length)` (`:103`) —
provably identical to `0 <= max_context_length < sample_len`. The mapper is
injected behind a `ChatMapper` trait (`:33-54`) with a blanket impl for
closures, so the preprocessor depends on the `messages → sample` behavior, not
the concrete type. Return is `usize` (rows written) rather than Python's
`None`, purely so callers/tests can assert the skip count.

`arrow_io::save_to_disk` (`arrow_io.rs:104-150`) reproduces the **byte-level HF
on-disk layout** by hand with the pure-Rust `arrow` crates (no pyarrow / no C
deps):
- Six column builders, `List<List<T>>` (or `List<Int64>` for `audio_in_lens`).
  Each sample's 2-D tensor is pushed row-by-row: inner builder appends a row
  slice then `append(true)`, outer `append(true)` closes the sample
  (`push_ll_i64`/`_f32`/`_bool`, `:75-100`). `bool` is materialized from the
  `u8` tensor via `x != 0` (`rows_bool`, `:52-54`).
- Column order matches the Python `Features` dict (`:124-132`).
- The HF `Features` JSON is embedded in the Arrow schema metadata under key
  `"huggingface"` (`:137-139`) exactly as pyarrow does, so the shard is
  self-describing.
- Writes the IPC **stream** shard `data-00000-of-00001.arrow` (`StreamWriter`,
  `:143-146`) plus the two JSON sidecars `dataset_info.json` (the `Features`
  schema) and `state.json` (`{_data_files, _fingerprint, _format_*, _split,…}`)
  (`write_sidecars`, `:154-171`). `load_from_disk` (`:195-235`) reads
  `state.json` for the shard list (falling back to scanning `*.arrow`), then
  reconstructs each row's six tensors on the target device.

There is **no sampling, no normalization, no attention** here — those belong to
the components the *mapper* calls.

## Dtypes & shapes (Rust)
| Field | In (mapper tensor) | On disk (Arrow) | Notes |
|---|---|---|---|
| `text` | I64 `(1, L)` | `List<List<Int64>>` | token ids |
| `audio_in` | F32 `(128, T_mel)` | `List<List<Float32>>` | log-mel; stored f32 |
| `audio_in_lens` | I64 `(n_audio_in,)` | `List<Int64>` | one mel-len per audio-in segment |
| `audio_out` | I64 `(8, T_codes)` | `List<List<Int64>>` | Mimi codes 0..2047 + EOAudio col 2048 |
| `modality_flag` | I64 `(1, L)` | `List<List<Int64>>` | `LFMModality`: TEXT=1, AUDIO_IN=2, AUDIO_OUT=3 |
| `supervision_mask` | U8 `(1, L)` | `List<List<Boolean>>` | loss mask (`u8 != 0` → bool) |

No float promotion / softmax / norm happens here. The only dtype events are
the `u8`↔`bool` and `to_dtype` casts in the Rust Arrow helpers (`arrow_io.rs:
46-70`), which preserve the schema dtypes.

## Wiring (Rust)
**Upstream:** `data/mapper.rs` — `mapper(messages) → LFM2AudioTrainingSample`
(the six tensors above). This is the only producer; `preprocess_dataset` calls
it once per chat. See [`glm-version/data/mapper.md`](mapper.md). The caller
supplies `data: impl Iterator<Item = Vec<ChatMessage>>` (`ChatMessage`/
segments from `data/types.rs`).

**Downstream:** `data/dataloader.rs` — `LFM2DataLoader::load_from_disk(out_dir)`
reads the Arrow shard back into rows, right-pads each to `context_length=4096`,
and `lfm2_collator` batches them into an `LFM2AudioModelInput`. See
[`glm-version/data/dataloader.md`](dataloader.md).

## Python ↔ Rust — where the port differs

| Python (`preprocess.py`) | Rust (`preprocess.rs` + `arrow_io.rs`) | Difference | Why |
|---|---|---|---|
| `preprocess_dataset(...)` returns `None` | `preprocess::preprocess_dataset` returns `usize` (`:85`) | **deliberate: rows written** | so callers/tests can assert the skip count. |
| `mapper(messages)` callable | `ChatMapper` trait + blanket closure impl (`:33-54`) | **deliberate: trait-object** | depends on behavior, not concrete type. |
| `out_dir.mkdir(parents=True, exist_ok=False)` | `create_output_dir` (`:58-66`) | identical | pre-existing dir → error. |
| `if 0 <= max_ctx < shape[-1]` | `(0..sample_len).contains(&max_ctx)` (`:103`) | identical | provably identical predicate. |
| `Dataset.from_generator(gen, features).save_to_disk` | `arrow_io::save_to_disk` (`arrow_io.rs:104`) | **deliberate: eager `Vec` collect + pure-Rust `arrow`** | the lazy generator/one-batch → eager `Vec` collect; real Arrow IPC via `arrow-array`/`arrow-ipc` instead of pyarrow/C. §2.7. |
| `datasets.load_from_disk` | `arrow_io::load_from_disk` (`arrow_io.rs:195`) | **deliberate: hand-written reader** | reads `state.json` + IPC, reconstructs tensors on the target device. |
| `Features({...})` schema | `features_json()` + schema metadata key `"huggingface"` (`:137`, `:175`) | identical | byte-compatible HF descriptor. |
| `bool` supervision_mask | `u8` in Rust, `x != 0` → `bool` for Arrow (`:52-54`) | **deliberate: U8** | candle has no bool dtype; the Arrow `Boolean` column is built from `u8 != 0`. |

**Deliberate divergence** (PYTHON_VS_RUST §2.7): HF `datasets.save_to_disk` →
real Arrow IPC stream + `dataset_info.json` + `state.json` via
`arrow-array`/`arrow-ipc`, *not* a custom schema and *not* pyarrow. This keeps
the on-disk corpus interchangeable with the Python `datasets` reader.

## Precision / gotchas (Rust-specific)
- **`exist_ok=false`**: re-running a build into an existing dir is a hard error
  in both (`create_output_dir`, `:58-66`) — prevents silent corpus corruption.
- **Skip predicate boundary**: `max_context_length=-1` (default) keeps
  everything (`0 <= -1` is false). A sample whose `L == max_context_length` is
  **kept** (half-open `[0, L)`); only strictly-longer samples are dropped. The
  length inspected is `modality_flag.shape[-1]` (the packed `L`), not the audio
  length.
- **EOAudio token = 2048**: the `audio_out` codes are `0..2047` plus an appended
  all-`2048` column per audio-out segment. 2048 is the EOAudio sentinel and is
  stored faithfully as I64 — this component must not clamp/strip it.
- **Eager memory**: Rust holds *all* kept samples in a `Vec` before the single
  Arrow flush (`:97`), so peak memory ≈ whole corpus; Python's `from_generator`
  likewise accumulates one batch. Fine for the small fine-tune corpora this
  targets; not a streaming sharded writer.
- **Single shard**: always one `data-00000-of-00001.arrow` (no multi-shard
  splitting); `state.json._data_files` lists exactly that one filename.
- **`u8`↔`bool` for `supervision_mask`**: candle carries U8; Arrow stores
  `Boolean`; the `rows_bool` helper (`:52-54`) materializes `x != 0`. The round
  trip is exact.
- **No mel/Mimi recompute here**: all numerical precision concerns live in the
  *mapper* and its callees; this component only relays already-computed
  tensors, so it introduces no new numerical error beyond the Arrow round-trip
  (exact for int64/bool; f32 is bit-preserved by the pinned `float32` schema).

## Cross-references
- [`wiki/data/preprocess.md`](../../../wiki/data/preprocess.md) — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.7 (data pipeline — HF `datasets` →
  pure-Rust `arrow`).
- `liquid-audio/src/data/arrow_io.rs` — the Arrow IPC reader/writer.