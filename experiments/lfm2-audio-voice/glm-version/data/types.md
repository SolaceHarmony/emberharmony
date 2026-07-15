# data_types (Rust port)
**Source:** `liquid-audio/src/data/types.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/data/types.py` · **On the LFM2-Audio inference path:** no

> Companion to [`wiki/data/types.md`](../../../wiki/data/types.md). The original is
> already Rust-aware; this is the Rust-first version.

## Role
`data/types.rs` is the value-type vocabulary of the training/data subsystem in
the Rust port — plain structs, no compute. It defines the *input* description
language (`ChatMessage` + the `ChatContentSegment` enum that describes one
conversation turn) and the three structurally-identical six-tensor bundles that
move a packed example through the data pipeline (`LFM2AudioTrainingSample` →
`LFM2AudioRow` → `LFM2AudioModelInput`). Nothing here runs at inference time
(the inference path uses `ChatState` in `processor.rs`, not these structs).

## How it works (Rust)
There is no forward pass, normalization, attention, or convolution in this
file — it is all struct declarations. The "mechanism" is the **field schema and
the pipeline staging**, which the neighbors enforce.

**Chat-content segments (`types.rs:60+`).** `SegmentKind` enum (`:61`) with
`Text`/`Audio`/`Interleaved` variants + `as_str()` returning the wire string
(`:72`) — the Rust analog of Python's `Literal["text"]`/`["audio"]`/
`["interleaved"]`. `Role` enum (`User`/`System`/`Assistant`) similarly. The
`ChatContentSegment` enum (`Text`/`Audio`/`Interleaved` variants carrying their
payload) replaces the PEP-604 union `TextSegment | AudioSegment |
InterleavedSegment`. `audio: Vec<u8>` (`bytes` → `Vec<u8>`). `ChatMessage` is
`{role: Role, content: Vec<ChatContentSegment>}`.

**The six-tensor bundle (`types.rs:33+`).** All three of
`LFM2AudioTrainingSample`, `LFM2AudioRow`, `LFM2AudioModelInput` declare the
*identical* six `Tensor` fields: `text`, `audio_in`, `audio_in_lens`,
`audio_out`, `modality_flag`, `supervision_mask`. They are distinct named types
to mark three pipeline stages. `LFM2AudioModelInput` (`:33`) is defined here
(where Python defines it) and re-exported from `model::lfm2_audio` (which
consumes it in `logits`/`forward`) — one canonical type. Its `to(&Device)`
method (`:44`) moves every field with `.to_device(device)`.

The schema is load-bearing — it is asserted downstream by
`LFM2AudioModel::prefill_inputs` (`lfm2_audio.rs:747`): the per-modality counts
must match each source tensor's length or the modality scatter errors.

## Dtypes & shapes (Rust)
| Field | dtype | shape | meaning |
|---|---|---|---|
| `text` | I64 | `(1, L)` sample → `(B, L)` after collate | text token ids |
| `audio_in` | F32 | `(128, ΣT)` | mel features, 128 bins × concat time |
| `audio_in_lens` | I64 | `(n_seg,)` | per-segment input-mel frame counts |
| `audio_out` | I64 | `(codebooks, L_ao)` (≥8 rows) | Mimi output codes + EOAudio (2048) |
| `modality_flag` | I64 | `(1, L)` → `(B, L)` | per-position `LFMModality` enum |
| `supervision_mask` | U8/Bool | `(1, L)` → `(B, L)` | loss-bearing positions |

No internal promotions happen *in this file* (no math). The dtypes are fixed by
the producers (`mapper.rs` casts to I64/F32/U8; `dataloader.rs` re-casts on
read; `preprocess.rs` Arrow schema stores I64/F32/U8). `modality_flag` is
stored as I64 (it's an enum, not a flag); `supervision_mask` is the real bool
(stored as U8 in Arrow/candle).

## Wiring (Rust)
**Upstream (producers):**
- `data/mapper.rs` — `LFM2AudioChatMapper` consumes `Vec<ChatMessage>` and emits
  `LFM2AudioTrainingSample`. See [`glm-version/data/mapper.md`](mapper.md).
- `data/preprocess.rs` — `preprocess_dataset` re-serializes each
  `LFM2AudioTrainingSample`'s six fields to an Arrow `Features` schema. See
  [`glm-version/data/preprocess.md`](preprocess.md).
- `data/dataloader.rs` — `LFM2DataLoader` reads the Arrow row back and emits
  `LFM2AudioRow` (padded to 4096); `lfm2_collator` consumes `Vec<LFM2AudioRow>`
  and emits `LFM2AudioModelInput`. See
  [`glm-version/data/dataloader.md`](dataloader.md).

**Downstream (consumers):**
- `model/lfm2_audio.rs` — `.logits(batch)` / `.forward(batch)` consume
  `LFM2AudioModelInput`; `prefill_inputs` scatters `text`/`audio_in`/`audio_out`
  embeddings by `modality_flag`, and `forward` builds the CE loss masks from
  `supervision_mask`. See
  [`glm-version/model/lfm2_audio.md`](../model/lfm2_audio.md).
- `trainer.rs` — `train_step(batch: LFM2AudioModelInput)` / `validate` move the
  batch with `.to(device)` and call `self.model.forward(batch)`. See
  [`glm-version/trainer.md`](../trainer.md).

## Python ↔ Rust — where the port differs

| Python (`data/types.py`) | Rust (`data/types.rs`) | Difference | Why |
|---|---|---|---|
| `@dataclass(frozen=True)` | plain `struct` with `pub` fields + `#[derive(Debug, Clone)]` | **deliberate: struct** | Rust has no `@dataclass`; the `frozen=True` immutability becomes "owning constructor + read-only `pub` fields" (Rust's default field privacy + `pub`). |
| `kind: Literal["text"]` etc. | `SegmentKind` enum (`Text`/`Audio`/`Interleaved`) + `as_str()` (`:61-78`) | **deliberate: string literal → enum** | Rust's enum is the closest faithful equivalent to a closed string-literal type. |
| `role: Literal["user"\|"system"\|"assistant"]` | `Role` enum (`User`/`System`/`Assistant`) | **deliberate: string literal → enum** | same pattern. |
| `ChatContentSegment = TextSegment \| AudioSegment \| InterleavedSegment` (PEP-604 union) | `ChatContentSegment` enum with `From<…>` impls + `kind()` reader | **deliberate: union → enum** | Rust's sum type is the enum; `isinstance` dispatch becomes `match`. |
| `audio: bytes` | `audio: Vec<u8>` | identical | `bytes` → `Vec<u8>` is the direct Rust analog. |
| `LFM2AudioTrainingSample` / `LFM2AudioRow` / `LFM2AudioModelInput` (`@dataclass`) | three Rust structs holding six `Tensor` fields each | identical (field-for-field) | — |
| `LFM2AudioModelInput.to(device)` (only this one has `to` in Python) | **all three bundles have `to(&Device)`** (`types.rs:350`, `:397`, `:44`) | **deliberate: `to` on all three** | candle is device-agnostic / explicit-placement (§2.1); a per-field move is the real device-transfer for every bundle — the Python relied on the whole pipeline being implicitly on `cuda`. |
| `LFM2AudioModelInput` defined in `data/types.py` | defined in `data/types.rs:33`, re-exported from `model::lfm2_audio` (`lib.rs`) | **deliberate: re-export** | mirrors that the Python `model/lfm2_audio.py` imports it from `data/types.py` — one canonical type, defined where it is consumed. |
| `torch.Tensor` fields | `candle_core::Tensor` fields | identical (the direct analog) | — |
| `supervision_mask: torch.bool` | `supervision_mask: Tensor` (U8 in practice) | **deliberate: U8** | candle has no `Bool` dtype; `U8` is the storage type, with `0`/`non-0` as the semantic. Consumers cast to U8 and compare. |

## Precision / gotchas (Rust-specific)
- **`audio_in` is F32 here, but the mel that fills it is precision-sensitive.**
  The mel front-end (`crates/liquid-audio/src/processor.rs`) computes in f32 (with f64
  window/filterbank/twiddles) and only *materializes* f32 in this bundle; in
  `ChatState` (inference) the same mel is stored bf16. Do not conflate the
  storage dtype here (f32) with a license to compute mel in low precision.
- **`modality_flag` is an I64 enum, not a boolean.** It carries the three
  `LFMModality` values; `supervision_mask` is the real bool (U8). The two are
  often `logical_and`-ed downstream (`lfm2_audio.rs:462-470`).
- **Off-by-one / shift in the loss masks (consumer-side, but the contract lives
  here).** `supervision_mask` is combined with the modality masks and sliced
  `[:, 1:]` (via `narrow`) for the next-token target while the shifted-input
  mask keeps the full length — so a padded position must be
  `supervision_mask == 0` (enforced at `dataloader.rs`) or it would leak into
  the CE loss.
- **EOAudio = 2048** lives in `audio_out` (code values `0..2048`, `2048` =
  end-of-audio), appended by the mapper; the bundle just transports it.
  `audio_out` having `>= codebooks` rows allows extra delay/EOS rows beyond the
  8 codebooks.
- **Batching is concat, not stack** (`dataloader.rs`): `modality_flag`/
  `supervision_mask` grow on `dim=0` but `text`/`audio_out` grow on `dim=1` — an
  easy axis-swap bug if reimplemented. The Rust must preserve the per-field axis
  exactly.
- **`to(&Device)` on all three bundles.** Unlike Python (where only
  `LFM2AudioModelInput` has `to`), the Rust gives all three a `to` because
  candle is device-agnostic — a per-field move is the real device-transfer for
  every bundle.

## Cross-references
- [`wiki/data/types.md`](../../../wiki/data/types.md) — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.7 (data
  pipeline backends).
- `liquid-audio/PORT_STATUS.md` — the 38/38 inventory + 170/170 symbol
  coverage.
