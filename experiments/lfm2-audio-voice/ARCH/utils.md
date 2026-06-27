# core_utils
**Code:** `CO03` · **Source:** `utils.py` · **Rust:** `utils.rs` · **On the LFM2-Audio inference path:** yes

## Role
The shared leaf module of `liquid_audio`. It holds the four things every other
component imports but none of them owns: the `LFMModality` enum that labels each
token slot in the interleaved sequence (TEXT / AUDIO_IN / AUDIO_OUT), the
`mel2emb_len` / `emb2mel_len` length-conversion arithmetic that ties the
conformer's 8× temporal subsampling to the number of AUDIO_IN slots the backbone
sees, `get_model_dir` (the `snapshot_download` resolver behind `from_pretrained`),
and `module_exists` (the feature-probe that selects the flash-attn backbone path).
It has no tensors of its own — it is the type/arithmetic glue that keeps the
modality bookkeeping consistent across processor, model, mapper, and dataloader.

## How it works
Four independent pieces; none allocate model state.

**`LFMModality(IntEnum)` (`utils.py:9`).** Three members built with `auto()`.
Because Python's `IntEnum.auto()` starts at **1**, the integer values are
`TEXT=1, AUDIO_IN=2, AUDIO_OUT=3` — not 0-based. This matters: the values are
materialized into a real `int64` tensor (`modality_flag`) and compared with `==`
all over the model. `processor.py:199` seeds the flag with
`torch.full_like(self.text, LFMModality.TEXT)`; the prefill scatter in
`lfm2_audio.py:335,354,360` builds three boolean masks `modality_flag == TEXT /
AUDIO_IN / AUDIO_OUT` and uses them to route per-token embeddings. Being an
`IntEnum` (not `Enum`) is load-bearing — `torch.full`/`new_tensor`/`F.pad` accept
it directly as an integer, and the data pipeline writes `int(LFMModality.TEXT)`
into Arrow int64 columns (`mapper.py:156`, `dataloader.py:45`).

**`mel2emb_len[T](l) -> T` (`utils.py:15`).** One line: `return -(l // -8)`.
This is the **ceil-division idiom**: `-(l // -8) == ceil(l / 8)` for the int and
`torch.Tensor` cases alike (the generic `T: (int, torch.Tensor)` makes it work on
a Python scalar or a length tensor without branching). The `8` is the conformer
encoder's total temporal subsampling factor — `dw_striding` ConvSubsampling does
2×2×2 = 8× downsampling (`subsampling.py`), so a log-mel feature of width `T`
frames produces `ceil(T/8)` backbone embedding steps. `emb2mel_len(l) = l * 8`
(`utils.py:24`) is the inverse **upper bound** (it ignores the ceil remainder).
The docstring pins the contract: smallest valid mel-length for the encoder is 9
(→ 2 emb steps). This function is the single source of truth for "how many
AUDIO_IN slots does this clip occupy":
- `processor.py:242` — when a user audio turn is appended, it emits exactly
  `mel2emb_len(mel_width)` copies of `LFMModality.AUDIO_IN` into the flag tensor,
  so the flag length stays in lockstep with the conformer output length.
- `lfm2_audio.py:330` — a prefill **invariant assert**:
  `(modality_flag == AUDIO_IN).sum() == mel2emb_len(audio_in_lens).sum()`. If the
  arithmetic and the scattered conformer embeddings disagree, prefill aborts here
  rather than silently misaligning the sequence.
- `mapper.py:203` — training-time, the same `mel2emb_len(cur_len)` decides how
  many AUDIO_IN entries go into the modality/supervision sequences.

**`module_exists(name) -> bool` (`utils.py:32`).** Wraps
`importlib.util.find_spec(name) is not None` — a pure import-availability probe,
no side effects. Its only on-path caller is `lfm2_audio.py:162`:
`if module_exists("flash_attn"):` selects `attn_implementation="flash_attention_2"`
for the HF `Lfm2Model` backbone, else falls back to eager/SDPA. So this one
boolean picks the attention kernel.

**`get_model_dir(repo_id, *, revision=None) -> Path` (`utils.py:40`).**
`@cache`-memoized (so repeated `from_pretrained` calls resolve once per process).
Branch on the argument type: a `str` is treated as a HF repo id and run through
`huggingface_hub.snapshot_download(repo_id, revision=revision)`, returning the
local snapshot directory; a `Path` is treated as an already-local checkpoint and
returned as-is, with the rule that passing `revision` **and** a path is a
`RuntimeError` (you cannot pin a revision on a local dir). This is the function
behind both entry points — `processor.py:63` and `lfm2_audio.py:144` both call it
to turn a repo id or path into the directory they then load `config.json` and
safetensors from.

## Dtypes & shapes
| Symbol | Input(s) | Output(s) | Notes |
|---|---|---|---|
| `LFMModality` | — | enum int (1/2/3) | materialized into `modality_flag` **int64** `(1,L)` |
| `mel2emb_len` | `int` **or** `torch.Tensor` (int64) mel width/lengths | same type, `ceil(l/8)` | scalar or length-vector; no float anywhere |
| `emb2mel_len` | `int`/`int64` emb length | `int`/`int64`, `l*8` | upper bound (drops ceil remainder) |
| `module_exists` | `str` module name | `bool` | no tensor |
| `get_model_dir` | `str` repo id **or** `Path`, opt `revision` | `Path` | cached; pure filesystem/network, no tensor |

No dtype promotions occur here — the only numeric op is integer floor-division.
The enum's int64 materialization happens in the **caller** (`torch.full_like` /
`new_tensor` inherit the existing `modality_flag` int64 dtype).

## Wiring
**Upstream (who feeds these):**
- The conformer mel front-end produces a log-mel of width `T` (f32/f64 computed,
  stored bf16 in `ChatState`); its width is what `mel2emb_len` consumes —
  [conformer_processor](model/conformer/processor.md) (mel `(128,T)`) and the
  8×-subsample length contract from [conformer_subsampling](model/conformer/subsampling.md).
- The on-disk checkpoint (bf16 weights) is what `get_model_dir` resolves a
  directory for; consumed by the loaders in
  [core_processor](processor.md) and [model_lfm2_audio](model/lfm2_audio.md).

**Downstream (who consumes these outputs):**
- [core_processor](processor.md) — imports `LFMModality`, `get_model_dir`,
  `mel2emb_len`; builds/extends the int64 `modality_flag (1,L)` and resolves the
  checkpoint dir.
- [model_lfm2_audio](model/lfm2_audio.md) — imports all four; `module_exists`
  selects the attention backend, `mel2emb_len` guards the prefill modality
  invariant, `LFMModality` masks drive the modality-scatter into hidden
  `(1,L,2048)`.
- [data_mapper](data/mapper.md) and [data_dataloader](data/dataloader.md) —
  use `LFMModality` ints and `mel2emb_len` to build training modality/supervision
  sequences (int64) and pad with `LFMModality.TEXT`.
- [demo_chat](demo/chat.md) — uses `LFMModality.TEXT`/`AUDIO_OUT` to tag the
  per-step output modality of the streaming generator.

## Python ↔ Rust
Symbol-level mapping (`utils.py` → `utils.rs`):

| Python | Rust | Note |
|---|---|---|
| `class LFMModality(IntEnum)` TEXT/AUDIO_IN/AUDIO_OUT | `enum LFMModality { Text=1, AudioIn=2, AudioOut=3 }` `#[repr(i64)]` | values hardcoded to **1/2/3** because Rust has no `auto()`; comment records the IntEnum-starts-at-1 fact (the easy-to-miss bug surface) |
| `mel2emb_len = -(l // -8)` | `mel2emb_len(l) = -floordiv(l, -8)` + a hand-written `floordiv` | **deliberate**: Rust `/` truncates toward zero, Python `//` floors toward −∞. A naive `-( l / -8)` would be wrong for the negative dividend; `floordiv` reproduces Python's flooring so `mel2emb_len` stays exact ceil-division. Unit test asserts `9→2, 16→2, 17→3` |
| `emb2mel_len = l*8` | `emb2mel_len(l) = l*8` | identical |
| `module_exists` via `importlib.find_spec` | `module_exists(name)` matches `"flash_attn" → cfg!(feature="flash-attn")` | **deliberate**: Rust has no runtime module table; the runtime import-probe maps to a **compile-time Cargo feature**. Semantically equivalent for its only caller (selecting the flash-attn attention path) |
| `get_model_dir` via `huggingface_hub.snapshot_download`, `@cache` | `get_model_dir(repo_or_path, revision) -> io::Result<PathBuf>` via `hf-hub` crate (`download_snapshot`), `#[cfg(feature="download")]` | **deliberate**: `snapshot_download` → `hf-hub` sync API; local-dir passthrough and the revision-with-path `RuntimeError` are preserved. No `@cache` (Rust callers resolve once). Disabling the `download` feature turns the network branch into a clear "clone the repo and pass its path" error |

Cross-references in PYTHON_VS_RUST.md: the device-agnostic loader story (§2.1 —
nothing in Rust hardcodes `cuda`, loaders take `device`+`dtype`) and the
flash-attn vs eager/SDPA substitution (§2.2 — `module_exists("flash_attn")`'s
caller goes to eager matmul + additive causal mask). PORT_STATUS.md notes
`get_model_dir`'s snapshot-download is done and `from_pretrained_hub` is the
faithful repo-id entry point.

## Precision / gotchas
- **`auto()` starts at 1, not 0.** The enum is compared against materialized
  int64 tensors throughout the model; an off-by-one here (e.g. assuming 0-based)
  would silently mis-mask every modality. The Rust port pins 1/2/3 explicitly and
  documents why — this is the single subtle correctness point in the file.
- **`-(l // -8)` is ceil, and the sign matters.** It is *not* `l // 8` (floor).
  For `l=17` floor gives 2 but ceil gives 3 — the conformer emits
  `ceil(T/8)` steps, so floor would drop the last partial frame and desync the
  modality flag from the actual embeddings, tripping the `lfm2_audio.py:330`
  assert. The Rust `floordiv` exists solely to keep this exact under Rust's
  truncating division.
- **`mel2emb_len` is integer-only** — no float, so it is bit-exact across Python
  and Rust (no place in the ~1e-6 cross-library floor; index/length math is
  always exact in this port).
- **`emb2mel_len` is an upper bound**, not a true inverse: `mel2emb_len(17)=3`
  but `emb2mel_len(3)=24 ≠ 17`. Callers that need the real mel width must track it
  separately (the model carries `audio_in_lens`); never round-trip lengths
  through `emb2mel_len` expecting recovery.
- **`get_model_dir` `@cache` + `Path` identity:** the memoization keys on the
  argument; a `Path` returns instantly and a repo-id string downloads once. A
  `revision` alongside a `Path` is a hard error in both implementations — do not
  pass it when pointing at a local snapshot dir.
