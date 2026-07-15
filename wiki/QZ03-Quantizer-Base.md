<!-- topic: Mimi Codec — Quantization -->
# QZ03 · BaseQuantizer / QuantizedResult
**Code:** `QZ03` · **Source:** `moshi/quantization/base.py` · **Rust:** `moshi crate` · **On the LFM2-Audio inference path:** yes

## Role
`base.py` defines the abstract **quantizer contract** that every Mimi vector-quantizer obeys: the `BaseQuantizer` `nn.Module` interface and the `QuantizedResult` dataclass that bundles a forward pass's outputs. It is pure interface plumbing — it holds no codebooks and does no math itself; the real RVQ lives in `vq.py`/`core_vq.py`. It exists so that `MimiModel` (`compression.py`) can hold a `quantizer: BaseQuantizer` and call a stable API (`forward`/`encode`/`decode`/`set_num_codebooks`/`cardinality`/`total_codebooks`/`num_codebooks`/semantic-vs-acoustic accessors) without knowing whether the concrete quantizer is the split-RVQ used in production, a flat RVQ, or the `DummyQuantizer` identity passthrough. On the LFM2-Audio path the concrete instance is always `SplitResidualVectorQuantizer` (`vq.py`), reached through this interface.

## How it works
This file is an **abstract base + dataclass**, so the "forward pass" it specifies is a *signature*, not an implementation. The mechanism worth documenting is exactly *what the contract pins down* and *what `compression.py` does with it*.

**`QuantizedResult` (base.py:22-28)** — the carrier struct returned by every `forward`. Fields: `x` (the quantized/dequantized continuous latent, `[B,C,T]`), `codes` (the discrete integer indices, `[B,K,T]`), `bandwidth` (a 0-dim/per-batch tensor in kb/s), `penalty` (optional commitment loss), `metrics` (dict; RVQ dead-code-replacement rate, entropy). On the inference path only `x` and `codes` matter; `penalty`/`metrics` are training-only.

**`BaseQuantizer` (base.py:31-97)** — `nn.Module` subclass. `__init__` sets one piece of real state: `self._ema_frozen = False` (base.py:36), the EMA-codebook-update gate read by `CompressionModel` via the `ema_frozen` property (base.py:90-93) and flipped by `ema_frozen_()` (base.py:95-97). `CompressionModel.__init__` calls `quantizer.ema_frozen_(True)` when `freeze_quantizer` is set (`compression.py:166-167`) — inference checkpoints ship frozen, so EMA never runs.

The interface methods are all `raise NotImplementedError()` stubs that the subclass must fill:
- `forward(x, frame_rate) -> QuantizedResult` (base.py:38-45) — note the **`frame_rate` argument**: it is threaded through purely to compute `bandwidth = num_codebooks * log2(bins) * frame_rate / 1000` (the actual formula lives in `vq.py:114,123`). The base only fixes the signature.
- `encode(x) -> codes` (base.py:47-49) — continuous latent `[B,C,T]` → int codes `[B,K,T]`. This is the inference-relevant entry point: `CompressionModel.encode` calls `self.quantizer.encode(emb)` after the SEANet encoder + framerate downsample (`compression.py:382-384`).
- `decode(codes) -> x` (base.py:51-53) — int codes `[B,K,T]` → continuous latent `[B,C,T]`. `CompressionModel.decode_latent` is literally `return self.quantizer.decode(codes)` (`compression.py:433`).

**Codebook-count API.** Four members govern RVQ depth: `cardinality` (codebook size = bins, base.py:55-58), `total_codebooks` (max RVQ levels available, base.py:60-63), `num_codebooks` (active levels, base.py:65-68), and `set_num_codebooks(n)` (base.py:86-88). `CompressionModel` forwards all four straight through (`compression.py:249-265`), and `loaders.get_mimi` calls `model.set_num_codebooks(8)` to activate **8** of the 32 available levels — the mechanism that produces the 8-codebook `(B,8,T)` frame layout the whole LFM2-Audio token flow assumes.

**Semantic/acoustic accessors.** `semantic_quantizer`/`acoustic_quantizer` (base.py:70-84) default to returning `self` — i.e. a flat quantizer *is* its own semantic and acoustic head. `SplitResidualVectorQuantizer` overrides these to return `rvq_first` (semantic, the 1 codebook whose loss is upweighted) and `rvq_rest` (acoustic, the other 7). This is the hook that lets callers reach into codebook-0 independently.

**`DummyQuantizer` (base.py:100-170)** — the concrete fallback defined *in this file*. It does **no quantization**: `forward` (base.py:128-133) just runs `input_proj` then `output_proj` (each either `nn.Identity()` when `input_dimension == dimension`, else a bias-free `Conv1d(·,·,1)` 1×1 pointwise projection, base.py:115-126) and stashes `x.unsqueeze(1)` as the "codes" — so the "codes" are continuous, not integers. `encode` returns `input_proj(x).unsqueeze(1)`; `decode` squeezes dim 1 and runs `output_proj`. Its `cardinality`/`total_codebooks`/`num_codebooks` are all **1**, and `set_num_codebooks` deliberately raises `AttributeError` (base.py:161-165). The Mimi checkpoint never uses it; it is the degenerate identity baseline the contract permits.

## Dtypes & shapes
The base class is dtype-agnostic — it only fixes shapes/roles; concrete dtypes come from the SEANet latent (model dtype) and the RVQ codebooks. Values below are what flows through the contract on the Mimi path.

| Member | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `forward(x, frame_rate)` | x: latent `[B,512→256,T]` model dtype (bf16 cuda / f32 cpu / bf16 metal); `frame_rate`: python `int` (12.5→passed as int) | `QuantizedResult{ x: latent `[B,512,T]` model dtype; codes: `[B,8,T]` int64 (u32 in Rust); bandwidth: `[]`/`[B]` f32; penalty: f32 scalar; metrics: dict }` |
| `encode(x)` | latent `[B,512,T]` model dtype | codes `[B,8,T]` int64 (Rust u32) |
| `decode(codes)` | codes `[B,8,T]` int64 (Rust u32), each ∈ `[0,2047]` | latent `[B,512,T]` model dtype |
| `cardinality` / `total_codebooks` / `num_codebooks` | — | python `int` (2048 / 32 / 8) |
| `DummyQuantizer.encode` | `[B,C,T]` model dtype | `[B,1,C,T]` model dtype (continuous "codes", not int) |

Promotions internal to the base file: none — `QuantizedResult` and the projections preserve incoming dtype. The f32 upcast (cdist argmin) and the int-cast of codes happen one layer down in `core_vq.py`; the f64-mel front-end and bf16-weight facts belong to upstream/peer components. `bandwidth` in `DummyQuantizer.forward` is built with `.to(x)` so it inherits `x`'s dtype/device.

## Wiring
**Upstream (who fills/feeds this contract):**
- [moshi_vq](QZ01-Split-RVQ) — `SplitResidualVectorQuantizer` and `ResidualVectorQuantizer` **subclass** `BaseQuantizer` and return `QuantizedResult`. This is the concrete implementation on the path; the base is its interface.
- [moshi_core_vq](QZ02-VQ-Core) — `ResidualVectorQuantization` (the residual loop + `EuclideanCodebook`) is what `vq.py` calls under the hood; not a subclass of `BaseQuantizer` but the engine behind `encode`/`decode`.
- [moshi_compression](MM01-Mimi-Codec) — `CompressionModel`/`MimiModel` *constructs and owns* the quantizer (`quantizer: BaseQuantizer`, `compression.py:132,151`), feeding it the SEANet+downsample latent `[B,512,T]` (model dtype) and consuming `QuantizedResult.x` / `codes`.

**Downstream (who consumes this output):**
- [moshi_compression](MM01-Mimi-Codec) — `CompressionModel.encode` consumes `codes` `[B,8,T]` int64; `CompressionModel.decode_latent`→`decode` consumes the latent `[B,512,T]` model dtype. This is the only direct consumer; everything else (processor, mapper, detokenizer, demo) consumes Mimi's `encode`/`decode`, not the quantizer directly.

## Python ↔ Rust
There is **no standalone Rust file** for `base.py`. Per `PYTHON_VS_RUST.md §2.3 / §4`, the entire vendored `liquid_audio/moshi/**` codec — including this quantizer interface — is **reused as the published `moshi` crate** (Kyutai's own Rust port), not re-ported. The decisive reason (Cargo.toml:39-48, `ARCH_1_MIMI_CODEC.md §6-7`): this checkpoint's RVQ weight keys are `quantizer.rvq_first.*` / `quantizer.rvq_rest.*`, which `moshi::mimi` matches natively while `candle-transformers`' Mimi cannot.

Symbol-level mapping:
| Python (`base.py`) | Rust (`moshi` crate) |
|---|---|
| `BaseQuantizer` abstract interface | folded into `moshi::quantization` / `moshi::mimi` quantizer types — no separate trait surfaced to `liquid-audio-rs` |
| `QuantizedResult` dataclass | internal to `moshi::mimi::encode`/`decode`; the Rust app only sees the final `codes` (u32) / waveform |
| `set_num_codebooks(8)` | `Some(codebooks)` argument to `moshi::mimi::load` / `Config::v0_1(codebooks)` (`loader.rs:296-303`) |
| `cardinality`/`total_codebooks`/`num_codebooks` | implicit in the moshi `Config` (bins=2048, n_q=32, active=8) |
| `DummyQuantizer` | not ported (off-path; identity baseline never used by the checkpoint) |
| `ema_frozen_` / EMA gate | not exposed — inference-only Rust never trains, so the EMA-freeze toggle is moot |

**Deliberate divergence (not a bug):** the `liquid-audio-rs` side has *no* `BaseQuantizer`/`QuantizedResult` mirror because the moshi-crate reuse hides the interface; `compare_symbols`'s `core` scope explicitly excludes `liquid_audio/moshi/**` (`PYTHON_VS_RUST.md §4`). Device-agnostic (candle, CPU-F32 / Metal-bf16) vs Python's CUDA-coupled module is the standard codec divergence (`ARCH_1 §4,7`), not specific to this file.

## Precision / gotchas
- **Interface only — read the subclass for real numerics.** Nothing in `base.py` does cdist/argmin/softmax; the f32-floor and quantization rounding live in `core_vq.py`. Don't attribute precision behavior to this file.
- **`DummyQuantizer` "codes" are continuous, not integers** (base.py:129,141) — its `codes` is `x.unsqueeze(1)`, a float latent. Any code path that assumes `QuantizedResult.codes` is always integer-castable would break on `DummyQuantizer`. Harmless on the LFM2-Audio path (split-RVQ is always used), but a real footgun for generic consumers.
- **The `[0, 2047]` / EOAudio contract is enforced elsewhere.** `vq.decode`'s docstring (`vq.py:144-146`) warns that out-of-range codes cause a "dramatic CUDA crash" with no guard (to avoid a sync point). `base.py` does not validate either. The `2048 = EOAudio` sentinel is stripped *before* codes reach the quantizer (by the processor/detokenizer), so the quantizer only ever sees `0..=2047 = cardinality-1`.
- **`set_num_codebooks` is the single most load-bearing call.** Activating 8 of 32 (`compression.py:258-260` → `vq.py:315-317`) is what defines the `(B,8,T)` frame the entire token interleave depends on; `SplitResidualVectorQuantizer.set_num_codebooks` asserts `n >= n_q_semantic` so you can never drop below the 1 semantic codebook.
- **`frame_rate` is a forward *argument*, not state.** It exists only for the bandwidth metric; it does not alter quantization. The actual 25→12.5 Hz resample is the codec's `_to_framerate` (`compression.py`), upstream of the quantizer — don't conflate the two.
- **EMA freeze is inference-irrelevant in Rust.** `_ema_frozen` matters only for training-time codebook updates; the Rust port is inference-only so the toggle has no analog.
