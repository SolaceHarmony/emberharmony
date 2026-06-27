# core_utils (Rust port)
**Source:** `liquid-audio-rs/src/utils.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/utils.py` · **On the LFM2-Audio inference path:** yes

> This is the Rust-side companion to [`ARCH/utils.md`](../ARCH/utils.md). Claude's
> original documents the **Python** module; this one documents the **Rust port**
> in `liquid-audio-rs/src/utils.rs` and calls out where the two diverge. Same
> topic, different language.

## Role
Identical purpose to the Python module: the shared leaf glue of the Rust port.
Four symbols, no tensor state of their own — the `LFMModality` enum, the
`mel2emb_len` / `emb2mel_len` length arithmetic, `module_exists` (a Cargo-feature
probe, not a runtime import probe), and `get_model_dir` (an `hf-hub`
snapshot resolver). Every other module in `src/` that touches modality
bookkeeping imports from here.

## How it works (Rust)
Four independent pieces; none allocate model state.

**`enum LFMModality` (`utils.rs:7`).** `#[repr(i64)]` with `Text=1, AudioIn=2,
AudioOut=3`. The values are **hardcoded**, not derived — Rust has no `auto()`,
so the Python `IntEnum`+`auto()` starts-at-1 convention is pinned explicitly
with a comment recording why. The `#[repr(i64)]` is load-bearing: the enum is
cast to `i64` and materialized into a real candle `Tensor` of dtype `I64`
(`processor.rs:182`, `mapper.rs:243`) and compared with `==` against other
`i64` tensors throughout the model (`lfm2_audio.rs:726-734`). `derive(Debug,
Clone, Copy, PartialEq, Eq)` is the minimum trait set the consumers need.

**`mel2emb_len(l: i64) -> i64` (`utils.rs:29`).** One line of real work:
`-floordiv(l, -8)`. The `floordiv` helper (`utils.rs:15`) is the **deliberate
substitution** for Python's `//` operator. Rust's `/` truncates toward zero;
Python's `//` floors toward −∞. A naive `-((l as i64) / -8)` would be wrong for
negative dividends (none occur on the inference path, but the helper keeps the
identity exact). `floordiv` reproduces Python's flooring:
```
fn floordiv(a, b) -> i64 {
    let q = a / b; let r = a % b;
    if r != 0 && (r < 0) != (b < 0) { q - 1 } else { q }
}
```
The `8` is the conformer encoder's total temporal subsampling factor —
`dw_striding` ConvSubsampling does 2×2×2 = 8× downsampling, so a log-mel of
width `T` produces `ceil(T/8)` backbone embedding steps. `emb2mel_len(l) =
l * 8` (`utils.rs:36`) is the inverse **upper bound** (drops the ceil
remainder). The docstring pins the same contract as Python: smallest valid
mel-length for the encoder is 9 (→ 2 emb steps). Unit test `mel_emb_len_roundtrip`
(`utils.rs:117`) asserts `9→2, 16→2, 17→3` — the three cases that distinguish
ceil from floor.

**`module_exists(name: &str) -> bool` (`utils.rs:43`).** `match name { "flash_attn"
=> cfg!(feature="flash-attn"), _ => false }`. This is the **compile-time
substitution** for Python's `importlib.util.find_spec`. Rust has no runtime
module table, so the runtime import-probe maps to a Cargo feature gate. Its only
caller in the Rust port is the attention-backend selection — see §Differences
below for why this is a stricter contract than the Python version.

**`get_model_dir(repo_or_path: &str, revision: Option<&str>) -> io::Result<PathBuf>`
(`utils.rs:56`).** Branches on whether the string is an existing local dir:
- **Local dir** (`p.is_dir()`): returned as-is; a `revision` alongside is an
  `Err(InvalidInput)` mirroring Python's `RuntimeError`.
- **Otherwise** (treated as a HF repo id): `download_snapshot(repo_id, revision)`
  via the `hf-hub` crate's sync `Api`. Lists siblings, fetches each
  (`snapshot_download` analog), and returns the directory holding
  `config.json`. The download branch is `#[cfg(feature = "download")]` (on by
  default); without it, the branch is a clear error telling the user to clone
  the repo and pass its path.

## Dtypes & shapes (Rust)
| Symbol | Input(s) | Output(s) | Notes |
|---|---|---|---|
| `LFMModality` | — | enum, `#[repr(i64)]` 1/2/3 | cast `as i64`; materialized into `Tensor` I64 `(1,L)` by callers |
| `mel2emb_len` | `i64` mel width | `i64`, `ceil(l/8)` | scalar only — **no** tensor overload (see §Differences) |
| `emb2mel_len` | `i64` emb length | `i64`, `l*8` | upper bound |
| `module_exists` | `&str` | `bool` | compile-time `cfg!`, no tensor |
| `get_model_dir` | `&str` repo id or path, `Option<&str>` revision | `io::Result<PathBuf>` | no `@cache`; download gated by `download` feature |

## Wiring (Rust)
**Upstream (who feeds these):**
- The conformer mel front-end produces a log-mel of width `T`; its width is what
  `mel2emb_len` consumes — [`glm-version/model/conformer/processor.md`](model/conformer/processor.md)
  and the 8×-subsample length contract from
  [`glm-version/model/conformer/subsampling.md`](model/conformer/subsampling.md).
- The on-disk checkpoint is what `get_model_dir` resolves a directory for;
  consumed by the loaders in [`glm-version/processor.md`](processor.md) and
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md) and by the
  `from_pretrained_hub` entry point in `loader.rs:95`.

**Downstream (who consumes these outputs):**
- `processor.rs:21` imports `mel2emb_len, LFMModality`; builds/extends the I64
  `modality_flag (1,L)` and seeds it with `LFMModality::Text` (`processor.rs:182`,
  `202`) and `AudioIn` (`processor.rs:215`).
- `model/lfm2_audio.rs:24` imports `mel2emb_len, LFMModality`; `mel2emb_len`
  guards the prefill modality invariant (`lfm2_audio.rs:618`) and `LFMModality`
  masks drive the modality-scatter into hidden `(1,L,2048)` (`lfm2_audio.rs:726-734`).
- `data/mapper.rs:33` imports `mel2emb_len, LFMModality`; builds training
  modality/supervision sequences (I64) and pads with `LFMModality::Text`
  (`mapper.rs:243,250,264`).
- `data/types.rs:339,388` documents `modality_flag` as per-position
  `LFMModality` flags.
- `loader.rs:95` calls `get_model_dir` inside `from_pretrained_hub`.

## Python ↔ Rust — where the port differs

| Python | Rust | Difference | Why |
|---|---|---|---|
| `class LFMModality(IntEnum)` with `auto()` | `enum LFMModality { Text=1, AudioIn=2, AudioOut=3 }` `#[repr(i64)]` | **values hardcoded** | Rust has no `auto()`; the 1-based convention is pinned explicitly + documented. Same load-bearing int64 materialization. |
| `mel2emb_len[T: (int, torch.Tensor)]` generic over scalar **and** tensor | `mel2emb_len(l: i64) -> i64` scalar only | **no tensor overload** | Rust callers compute lengths on the host and broadcast into tensors themselves (`processor.rs:214` casts `frames as i64`, then builds a `Tensor` of the resulting count). The generic tensor path adds no value when the caller already has the scalar. |
| `-(l // -8)` with Python's flooring `//` | `-floordiv(l, -8)` with a hand-written `floordiv` | **deliberate** | Rust `/` truncates toward zero; `floordiv` reproduces Python's flooring so ceil-division stays exact. Unit test pins `9→2, 16→2, 17→3`. |
| `emb2mel_len = l*8` | `emb2mel_len(l) = l*8` | identical | — |
| `module_exists` via `importlib.find_spec` (runtime) | `module_exists(name) => match name { "flash_attn" => cfg!(feature="flash-attn"), _ => false }` | **deliberate: runtime → compile-time** | Rust has no runtime module table. The only caller selects the attention backend; a Cargo feature is the faithful static analog. Note this is **stricter than Python**: an unknown name returns `false` unconditionally rather than probing, so the `_ => false` arm is a deliberate dead-end for anything but `flash_attn`. |
| `get_model_dir` via `huggingface_hub.snapshot_download`, `@cache` | `get_model_dir(repo_or_path, revision) -> io::Result<PathBuf>` via `hf-hub` sync `Api` | **deliberate** | `snapshot_download` → `hf-hub` sync API; local-dir passthrough and the revision-with-path error are preserved. **No `@cache`** (Rust callers resolve once per process and hold the `PathBuf`; the Python `@cache` exists because `from_pretrained` can be called many times cheaply). Disabling the `download` feature turns the network branch into a clear "clone the repo and pass its path" error. The Python `isinstance(repo_id, str)` branch becomes `p.is_dir()` (a string that isn't an existing dir is treated as a repo id) — semantically equivalent for the valid inputs. |
| returns `Path` | returns `io::Result<PathBuf>` | **error model** | Python raises `RuntimeError`; Rust returns `Err`. The revision-with-path case maps `RuntimeError` → `Err(InvalidInput)` with the same message text. |

## Precision / gotchas (Rust-specific)
- **`#[repr(i64)]` is load-bearing.** Without it the discriminant width is
  implementation-defined and the `as i64` casts in callers
  (`LFMModality::Text as i64`) would not be stable. The Python `IntEnum` is
  always a Python `int` (arbitrary precision); the Rust port commits to `i64`
  explicitly, which matches the `torch.long` (int64) `modality_flag` the Python
  reference materializes.
- **`floordiv`'s sign logic.** `(r < 0) != (b < 0)` is the standard "did the
  remainder and divisor have opposite signs" check. For `mel2emb_len` the
  divisor is always `-8` and the dividend is a non-negative length on the
  inference path, so the branch is unreachable in practice — but the helper is
  written correctly for the general case so it can be reused if a negative
  length ever appears (e.g. a signed offset arithmetic change upstream).
- **`mel2emb_len` is `i64`-only, integer-exact.** No float, so bit-exact vs
  Python. The Rust port inherits the "single source of truth for how many
  AUDIO_IN slots this clip occupies" property.
- **`module_exists` is not extensible at runtime.** A Python user could add a
  new `module_exists("xformers")` check; the Rust port requires a new `match`
  arm + Cargo feature. This is acceptable for a faithful port — the upstream
  only ever calls it with `"flash_attn"`.
- **`get_model_dir` `@cache` loss.** The Python `@cache` means
  `get_model_dir("LiquidAI/LFM2.5-Audio-1.5B")` downloads once per process even
  if called from many `from_pretrained` invocations. The Rust port re-fetches
  sibling metadata every call (the `hf-hub` cache layer still avoids
  re-downloading bytes, so this is a metadata-list round-trip per call, not a
  weight re-download). Callers that want Python-equivalent behavior hold the
  returned `PathBuf` and pass it to subsequent loads. If many-call patterns
  become hot, a `once_cell::sync::Lazy<Mutex<HashMap>>` mirror is the fix.
- **The `download` feature gate.** With `default-features = false`, the
  `download_snapshot` branch becomes a `NotFound` error with actionable text.
  This is the offline-build / air-gapped path. The Python has no such gate —
  `snapshot_download` is always available (and always tries the network).

## Cross-references
- [`ARCH/utils.md`](../ARCH/utils.md) — the Python original this is the Rust
  companion to. That doc has the Python `IntEnum`/`auto()` provenance and the
  `@cache` semantics; this one has the `floordiv`/`cfg!`/`hf-hub` differences.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic loaders —
  `get_model_dir` takes no `device`/`dtype`, the loaders do) and §2.2
  (`module_exists("flash_attn")` → eager matmul + additive causal mask).
- `liquid-audio-rs/PORT_STATUS.md` — `utils.rs` marked ✅ done, including
  `get_model_dir` snapshot-download via `hf-hub`.
- `liquid-audio-rs/parity/SIGNATURE_AUDIT.md` — symbol-level coverage for this
  module.