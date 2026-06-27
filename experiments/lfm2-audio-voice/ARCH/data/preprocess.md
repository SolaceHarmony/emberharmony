# data_preprocess
**Code:** `DA03` · **Source:** `data/preprocess.py` · **Rust:** `data/preprocess.rs + arrow_io.rs` · **On the LFM2-Audio inference path:** no

## Role
`preprocess_dataset` is the **offline dataset-build step** for *training*: it consumes an iterable of chats (`list[ChatMessage]`), runs each through an [`LFM2AudioChatMapper`](mapper.md) to pack it into an `LFM2AudioTrainingSample`, applies an optional context-length filter, and materializes the kept samples to disk as a real HuggingFace `datasets.Dataset` (`save_to_disk`). It is pure plumbing — no neural net, no inference — and exists so that the expensive per-sample work (tokenize, mel front-end, Mimi-encode audio-out) happens once at corpus-build time and the [`LFM2DataLoader`](dataloader.md) can later mmap-cheaply iterate Arrow rows. It is **not on the inference path**; nothing here runs at chat/generate time.

## How it works
The whole module is one function plus a nested generator (`preprocess.py:13-50`). Mechanism, step by step:

1. **Dir guard** (`:19-20`): `Path(output_path).mkdir(parents=True, exist_ok=False)`. The `exist_ok=False` is load-bearing — a pre-existing output dir is a hard error, so a build never silently appends to / overwrites a prior corpus.

2. **Fixed Arrow `Features` schema** (`:22-31`) — declared up front and passed to `from_generator`, which *pins* the column dtypes (rather than letting Arrow infer them from the first row). Six columns, exactly mirroring `LFM2AudioTrainingSample` (`data/types.py:37-45`):
   - `text` → `Sequence(Sequence(int64))` (2-D: `(1, L)` token ids)
   - `audio_in` → `Sequence(Sequence(float32))` (2-D: `(128, ΣT_mel)` log-mel, 128 mel bins)
   - `audio_in_lens` → `Sequence(int64)` (1-D: per-audio-in mel frame counts)
   - `audio_out` → `Sequence(Sequence(int64))` (2-D: `(8, ΣT_codes)` Mimi codes, 8 codebooks)
   - `modality_flag` → `Sequence(Sequence(int64))` (2-D: `(1, L)` of `LFMModality` ∈ {1,2,3})
   - `supervision_mask` → `Sequence(Sequence(bool))` (2-D: `(1, L)`)

   Note the **nesting**: each row's value is the *whole* 2-D tensor of one sample, stored as an outer list of inner per-row lists. `audio_in_lens` is the only singly-nested (1-D) column.

3. **`generator()`** (`:33-47`) — lazily drives the corpus:
   - `sample = mapper(messages)` (`:35`): the mapper does all the real packing — token ids via the tokenizer, mel via the conformer front-end (`processor.audio`), audio-out via `processor.mimi.encode` then a column of `2048` (EOAudio) appended (`mapper.py:229-232`), and the parallel `modality_flag` / `supervision_mask` sequences. Interleaving cadence (6 text / 12 audio tokens) is decided *inside* the mapper (`mapper.py:149-164`), not here.
   - **Context-length skip** (`:36-39`): `sample_len = int(sample.modality_flag.shape[-1])` (the packed sequence length `L`), and `if 0 <= max_context_length < sample_len: print(WARNING…); continue`. The default `max_context_length=-1` makes `0 <= -1` false → **nothing is ever skipped**. The predicate is the half-open membership test `max_context_length ∈ [0, sample_len)`; a sample exactly equal to `max_context_length` is *kept*.
   - **`yield`** (`:40-47`): every tensor field is `.tolist()`-ified into native Python nested lists so `datasets`/pyarrow can serialize them. (This is the dtype-narrowing boundary: torch `long`→int64, `float32`→float32, `bool`→bool, all preserved by the pinned schema.)

4. **Materialize** (`:49-50`): `datasets.Dataset.from_generator(generator, features=features)` builds the in-memory (single-batch) Arrow table, then `.save_to_disk(out_dir)` flushes the shard + sidecars.

There is **no sampling, no normalization, no attention** here — those belong to the components the *mapper* calls. The only "math" is the floor-division length bookkeeping that already happened upstream: `mel2emb_len(l) = -(l // -8)` = `ceil(l/8)` (`utils.py:15-21`), the conformer 8× subsampling that sets how many `AUDIO_IN` modality slots each mel chunk contributes.

**Rust mechanism.** `preprocess_dataset` (`preprocess.rs:85-113`) is a faithful port with one structural change: the Python `generator` lazily yields and `from_generator` accumulates one batch internally; Rust instead collects `kept: Vec<LFM2AudioTrainingSample>` (`:97-108`) and hands the whole vec to `arrow_io::save_to_disk`. The skip predicate is written as the explicit half-open range `(0..sample_len).contains(&max_context_length)` (`:103`) — provably identical to `0 <= max_context_length < sample_len`. The mapper is injected behind a `ChatMapper` trait (`:33-54`) with a blanket impl for closures, so the preprocessor depends on the `messages → sample` behavior, not the concrete type. Return is `usize` (rows written) rather than Python's `None`, purely so callers/tests can assert the skip count.

`arrow_io::save_to_disk` (`arrow_io.rs:104-150`) reproduces the **byte-level HF on-disk layout** by hand with the pure-Rust `arrow` crates (no pyarrow / no C deps):
- Six column builders, `List<List<T>>` (or `List<Int64>` for `audio_in_lens`). Each sample's 2-D tensor is pushed row-by-row: inner builder appends a row slice then `append(true)`, outer `append(true)` closes the sample (`push_ll_i64`/`_f32`/`_bool`, `:75-100`). `bool` is materialized from the `u8` tensor via `x != 0` (`rows_bool`, `:52-54`).
- Column order **matches the Python `Features` dict** (`:124-132`).
- The HF `Features` JSON is embedded in the Arrow schema metadata under key `"huggingface"` (`:137-139`) exactly as pyarrow does, so the shard is self-describing.
- Writes the IPC **stream** shard `data-00000-of-00001.arrow` (`StreamWriter`, `:143-146`) plus the two JSON sidecars `dataset_info.json` (the `Features` schema) and `state.json` (`{_data_files, _fingerprint, _format_*, _split,…}`) (`write_sidecars`, `:154-171`). `load_from_disk` (`:195-235`) reads `state.json` for the shard list (falling back to scanning `*.arrow`), then reconstructs each row's six tensors on the target device.

## Dtypes & shapes
Per-sample (one chat → one dataset row). `L` = packed seq length; `T_mel` = total mel frames; `T_codes` = total Mimi code frames (incl. one EOAudio column per audio-out segment).

| Field | In (mapper tensor) | On disk (Arrow) | Notes |
|---|---|---|---|
| `text` | int64 `(1, L)` | `List<List<Int64>>` | token ids; `long` ← tokenizer |
| `audio_in` | f32 `(128, T_mel)` | `List<List<Float32>>` | log-mel; computed f32/f64 front-end, stored f32 (empty `(128,0)` if no audio-in) |
| `audio_in_lens` | int64 `(n_audio_in,)` | `List<Int64>` | one mel-len per audio-in segment |
| `audio_out` | int64 `(8, T_codes)` | `List<List<Int64>>` | Mimi codes 0..2047 + EOAudio col `2048`; empty `(8,0)` if none |
| `modality_flag` | int64 `(1, L)` | `List<List<Int64>>` | `LFMModality`: TEXT=1, AUDIO_IN=2, AUDIO_OUT=3 |
| `supervision_mask` | bool `(1, L)` | `List<List<Boolean>>` | loss mask (assistant tokens True) |

No float promotion / softmax / norm happens in this component — those live upstream in the mapper's mel front-end (f64-sensitive) and Mimi encode. The only dtype events here are `.tolist()` serialization (Python) and the `u8`↔`bool` and `to_dtype` casts in the Rust Arrow helpers (`arrow_io.rs:46-70`), which preserve the schema dtypes. Mimi codebook codes are `int64` here (Python `long`); they become `u32` only later inside the Mimi/detok codec at inference time, which this offline path never touches.

## Wiring
**Upstream (feeds this):**
- [`LFM2AudioChatMapper`](mapper.md) — `mapper(messages) → LFM2AudioTrainingSample` (the six tensors above). This is the only producer; `preprocess_dataset` calls it once per chat.
- Caller supplies `data: Iterable[list[ChatMessage]]` (`ChatMessage`/segment dataclasses from [`data/types.py`](types.md)).

**Downstream (consumes this output):**
- [`LFM2DataLoader`](dataloader.md) — `load_from_disk(out_dir)` reads the Arrow shard back into rows, then right-pads each to `context_length=4096` and `lfm2_collator` batches them into an `LFM2AudioModelInput`. Edge: the six on-disk Arrow columns (int64 `(1,L)` text/modality, f32 `(128,T_mel)` audio_in, int64 `(8,T_codes)` audio_out, int64 `(n,)` lens, bool `(1,L)` mask) → padded/batched tensors for [`trainer.py`](trainer.md)'s loss-on-the-model forward.

## Python ↔ Rust
| Python symbol | Rust symbol | Divergence |
|---|---|---|
| `preprocess_dataset(...)` | `preprocess::preprocess_dataset` (`preprocess.rs:85`) | returns `usize` (rows written) vs `None` |
| `mapper(messages)` callable | `ChatMapper` trait + blanket closure impl (`preprocess.rs:33-54`) | trait-object on behavior, not concrete type |
| `out_dir.mkdir(parents=True, exist_ok=False)` | `create_output_dir` (`:58-66`) | faithful; pre-existing dir → error |
| `if 0 <= max_ctx < shape[-1]` | `(0..sample_len).contains(&max_ctx)` (`:103`) | provably identical predicate |
| `Dataset.from_generator(gen, features).save_to_disk` | `arrow_io::save_to_disk` (`arrow_io.rs:104`) | **lazy generator/one-batch → eager `Vec` collect**; real Arrow IPC via `arrow-array`/`arrow-ipc` crates instead of pyarrow/C |
| `datasets.load_from_disk` | `arrow_io::load_from_disk` (`arrow_io.rs:195`) | hand-written `state.json`+IPC reader |
| `Features({...})` schema | `features_json()` + schema metadata key `"huggingface"` (`arrow_io.rs:137,175`) | byte-compatible HF descriptor |

DELIBERATE divergence per **PYTHON_VS_RUST.md §2.7** ("Data pipeline — same formats, pure-Rust backends"): HF `datasets.save_to_disk` → real Arrow IPC stream + `dataset_info.json` + `state.json` via `arrow-array`/`arrow-ipc`, *not* a custom schema and *not* pyarrow. This keeps the on-disk corpus interchangeable with the Python `datasets` reader.

## Precision / gotchas
- **`exist_ok=False`**: re-running a build into an existing dir is a hard error in both Python and Rust (`create_output_dir`, `preprocess.rs:58-66`) — intentional, prevents silent corpus corruption.
- **Skip predicate boundary**: `max_context_length=-1` (default) keeps everything (`0 <= -1` is false). A sample whose `L == max_context_length` is **kept** (half-open `[0, L)`); only strictly-longer samples are dropped. The length inspected is `modality_flag.shape[-1]` (the packed `L`), not the audio length.
- **EOAudio token = 2048**: the `audio_out` codes are `0..2047` plus an appended all-`2048` column per audio-out segment (`mapper.py:231`). 2048 is the EOAudio sentinel and is stored faithfully as int64 — this component must not clamp/strip it.
- **Lazy vs eager memory**: Rust holds *all* kept samples in a `Vec` before the single Arrow flush (`preprocess.rs:97`), so peak memory ≈ whole corpus; Python's `from_generator` likewise accumulates one batch. Fine for the small fine-tune corpora this targets; not a streaming sharded writer.
- **Single shard**: always one `data-00000-of-00001.arrow` (no multi-shard splitting); `state.json._data_files` lists exactly that one filename.
- **No mel/Mimi recompute here**: all numerical precision concerns (f64-sensitive mel front-end, Mimi RVQ) live in the *mapper* and its callees; this component only relays already-computed tensors, so it introduces no new numerical error beyond `.tolist()`/Arrow round-trip (exact for int64/bool; f32 is bit-preserved by the pinned `float32` schema).
