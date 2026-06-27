# conformer_utils (Rust port)
**Source:** `liquid-audio-rs/src/model/conformer/utils.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/utils.py` · **On the LFM2-Audio inference path:** yes (but inert on the offline path)

> Companion to [`ARCH/model/conformer/utils.md`](../../ARCH/model/conformer/utils.md).
> The original documents the Python NeMo helpers; this documents the Rust port
> and where it diverges.

## Role
A 142-line grab-bag of NeMo-derived helpers used by the FastConformer encoder:
an autocast-dtype guard (`avoid_float16_autocast_context`), a streaming-config
struct (`CacheAwareStreamingConfig` + `IntOrPair`), and a stochastic-depth
drop-probability schedule (`compute_stochastic_depth_drop_probs`). None of the
three carry tensors; they configure or guard the conformer rather than
transform activations. On the LFM2-Audio *offline* inference path all three are
effectively inert (see below), but ported 1:1 for inventory completeness — the
header comment (`utils.rs:1-5`) records this.

## How it works (Rust)

**`avoid_float16_autocast_context(autocast_dtype, bf16_supported) -> Option<DType>`
(`utils.rs:74-81`).** The Rust port keeps the *decision* but drops the
*context-manager* form — candle has no implicit autocast, so the function is a
pure decision over the would-be autocast dtype:
- `Some(DType::F16)` active → `Some(BF16)` if `bf16_supported`, else `Some(F32)`.
- Anything else (`None`, `Some(BF16)`, `Some(F32)`) → `None` (= `nullcontext()`,
  no dtype override).

The `torch.jit.is_scripting()/is_tracing()` branch (which forces f32 in Python)
has no candle analog and is taken as false. On the offline path the conformer
attention already upcasts to f32 explicitly inside `mha.rs`, so this function
is effectively dead — it exists for inventory parity. Unit test
`avoid_float16_decision` (`:127`) pins the four cases.

**`IntOrPair` (`utils.rs:14-36`) + `CacheAwareStreamingConfig` (`:40-60`).**
Python stores `chunk_size`/`shift_size`/`pre_encode_cache_size` as a bare `int`
OR a 2-element `[first_step, others]` list. The Rust port makes the
polymorphism explicit in the type system: `enum IntOrPair { Int(i64),
Pair(i64, i64) }` with a `.second()` accessor mirroring Python's `cfg.x[1]`
access (for a scalar it returns the value itself; for a pair it returns the
second element). The `subsampling::ConvSubsampling` reports
`[1, subsampling_factor]`, so for this model these fields are always `Pair`.
`CacheAwareStreamingConfig` is a `#[derive(Debug, Clone, Default)]` struct
with the same nine fields as the Python dataclass. `IntOrPair::default()` is
`Int(0)`. Unit test `int_or_pair_second` (`:138`) pins the accessor.

**`compute_stochastic_depth_drop_probs(num_layers, p, mode, start_layer) -> Vec<f64>`
(`utils.rs:86-120`).** Faithful 1:1 port:
- `assert!` validates `0.0 ≤ p < 1.0` and `1 ≤ start_layer ≤ num_layers`
  (Python's `ValueError` becomes `panic!`).
- Layers `[0, start_layer)` get `0.0` (`:102`).
- For `big_l = num_layers - start_layer` (`:105`):
  - `"linear"`: `(1..=big_l).map(|l| l as f64 / big_l as f64 * p)` (`:109-111`)
    — the linear ramp from `1/L·p` to `p`.
  - `"uniform"`: `repeat_n(p, big_l)` (`:113`).
  - any other mode: `panic!` (`:114-116`).

Called once at encoder construction with `len(self.layers)=17`. For LFM2-Audio
inference `p = 0.0`, so the linear branch yields `l/L · 0 = 0` for every layer
— `layer_drop_probs` is all zeros and no layer is ever dropped.

## Dtypes & shapes (Rust)
This component is a config/guard layer; it carries **no activation tensors**.

| Symbol | Input(s) | Output |
|---|---|---|
| `avoid_float16_autocast_context` | `autocast_dtype: Option<DType>`, `bf16_supported: bool` | `Option<DType>`: `Some(BF16/F32)` if f16 active, else `None` |
| `IntOrPair::second` | `&self` | `i64` |
| `CacheAwareStreamingConfig` | int/`IntOrPair` field values (default 0/`Int(0)`) | struct instance (no tensors) |
| `compute_stochastic_depth_drop_probs` | `num_layers: usize`, `p: f64`, `mode: &str`, `start_layer: usize` | `Vec<f64>` of length `num_layers`, values in `[0, p]` |

## Wiring (Rust)
**Upstream (who configures/invokes this):**
- `model/conformer/mha.rs` would call `avoid_float16_autocast_context` to wrap
  the QKV/attention block — but on the offline path it's a no-op (the attention
  upcasts to f32 explicitly). See
  [`glm-version/model/conformer/mha.md`](mha.md).
- `model/conformer/encoder.rs` calls `compute_stochastic_depth_drop_probs` at
  construction and constructs `CacheAwareStreamingConfig`. See
  [`glm-version/model/conformer/encoder.md`](encoder.md).

**Downstream (who consumes this output):**
- `encoder.rs` consumes the `Vec<f64>` (length 17, all 0.0 at inference) as
  `layer_drop_probs`, and the `CacheAwareStreamingConfig` as `streaming_cfg`.
  Both are config, not tensors.
- `mha.rs` consumes the `Option<DType>` (always `None` on the offline path) —
  the attention runs in the ambient model dtype with its own explicit f32
  upcast.

## Python ↔ Rust — where the port differs

| Python (`utils.py`) | Rust (`utils.rs`) | Difference | Why |
|---|---|---|---|
| `avoid_float16_autocast_context()` returns a *context manager* | `avoid_float16_autocast_context(autocast_dtype, bf16_supported) -> Option<DType>` returns a *dtype decision* | **deliberate: context manager → pure function** | candle has no implicit autocast; the port keeps the decision but drops the context-manager form. The `torch.jit.is_scripting()/is_tracing()` branch has no candle analog and is folded to "not taken." On the offline path it's effectively dead. |
| `torch.is_autocast_enabled()` global probe | `autocast_dtype: Option<DType>` explicit argument | **deliberate: global → explicit** | Rust has no global autocast state; the caller passes the would-be autocast dtype. |
| `CacheAwareStreamingConfig` (`@dataclass`) | `struct CacheAwareStreamingConfig` (`#[derive(Default)]`) | identical (field-for-field) | — |
| `chunk_size`/`shift_size`/`pre_encode_cache_size` are `int \| [int, int]` | `IntOrPair` enum (`Int(i64)` / `Pair(i64, i64)`) + `.second()` accessor | **deliberate: polymorphism explicit in types** | Rust makes the int-vs-pair polymorphism explicit; Python's duck typing becomes a sum type. For this model the subsampling pre-encoder always reports `[1, subsampling_factor]`, so the `Pair` branch is the live one. |
| `compute_stochastic_depth_drop_probs` returns `list[float]` | `compute_stochastic_depth_drop_probs` returns `Vec<f64>` | identical | — |
| `ValueError` on bad `p`/`start_layer`/`mode` | `panic!` on bad `p`/`start_layer`/`mode` | **deliberate: exception → panic** | Rust's `assert!`/`panic!` is the analog of Python's `ValueError` for programmer errors. These are construction-time invariants, not runtime recoverable errors. |
| `0 ≤ p < 1` check | `(0.0..1.0).contains(&p)` | identical | Rust's `Range::contains` is the analog of Python's chained comparison. |
| `1 ≤ start_layer ≤ num_layers` check | `(1..=num_layers).contains(&start_layer)` | identical | — |
| `"linear"` ramp via `numpy.linspace`-style or manual | `(1..=big_l).map(|l| l as f64 / big_l as f64 * p)` | **manual** | Rust has no `linspace`; the explicit `(1..=big_l)` iterator reproduces the ramp. |
| `"uniform"` via `[p]*big_l` | `std::iter::repeat_n(p, big_l)` | identical | `repeat_n` is the Rust analog of Python's list repetition. |

## Precision / gotchas (Rust-specific)
- **The autocast guard is a no-op on the LFM2-Audio offline path.** The Rust
  function takes an explicit `autocast_dtype: Option<DType>` — on the offline
  path the caller passes `None` (no autocast), so the function returns `None`
  and the attention runs in the ambient model dtype with its own explicit f32
  upcast inside `mha.rs`. Do not mistake this helper for the source of conformer
  attention precision.
- **Stochastic depth is inference-inert.** With `p = 0.0`, the linear branch
  yields `l/L · 0 = 0` for every layer, so `layer_drop_probs` is all zeros and
  the 17-layer drop logic never fires. It only matters during training.
- **`start_layer` is 1-based and inclusive on both ends** (`1 ≤ start_layer ≤
  num_layers`); off-by-one care: layers `[0, start_layer)` are the never-dropped
  prefix, and the linear ramp indexes `l = 1..=big_l` so the *final* layer (not
  the first droppable one) receives the full `p`.
- **`IntOrPair::second()` on a scalar returns the scalar.** Python would index
  an int (erroring), but the Rust `Int(v).second() == v` is a deliberate
  convenience — the list branch is the one this model takes, but the scalar
  branch is kept benign rather than panicking.
- **`panic!` vs `Result`.** The validation `assert!`s and the bad-mode `panic!`
  are construction-time invariants. If a future caller passes a bad `p` or
  `mode`, the panic is a programmer-error abort, not a recoverable `Err`. This
  matches the Python `ValueError` semantics (a construction-time crash, not a
  runtime retry).
- **Word-count audit:** this file is **0.55×** the Python (small helpers,
  expected) — logged in `PYTHON_VS_RUST.md:249`. The `// PORT:` no-op note for
  the autocast guard is recorded in `PORT_STATUS.md`.

## Cross-references
- [`ARCH/model/conformer/utils.md`](../../ARCH/model/conformer/utils.md) —
  Python original.
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.5 (off-path NeMo machinery → inventory
  stubs), §2.6 (accelerator.autocast → candle equivalents).
- `liquid-audio-rs/PORT_STATUS.md` — the `// PORT:` no-op note for
  `avoid_float16_autocast_context`.