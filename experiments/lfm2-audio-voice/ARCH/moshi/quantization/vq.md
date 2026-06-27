# moshi_vq
**Code:** `QZ01` · **Source:** `moshi/quantization/vq.py` · **Rust:** `moshi crate quantization` · **On the LFM2-Audio inference path:** yes

## Role
The quantizer head of the Mimi codec. `SplitResidualVectorQuantizer` is the top-level VQ wired into `MimiModel`: it turns a continuous 12.5 Hz latent `(B, 256, T)` into `(B, 8, T)` discrete integer codes (encode), and reconstructs that latent from codes (decode). It is *split* — one **semantic** RVQ stack (`rvq_first`, `n_q_semantic=1`) plus one **acoustic** RVQ stack (`rvq_rest`, the remaining 7) — each a `ResidualVectorQuantizer` that owns the per-level codebooks (in `core_vq.py`). These 8-codebook frames are exactly the audio-out vocabulary the LFM2-Audio model emits and consumes, so this component is the bridge between waveforms and tokens at both ends of training and the output end of inference.

## How it works
This file is two `BaseQuantizer` subclasses; the actual nearest-centroid math lives in `core_vq.py` (`EuclideanCodebook`), which `ResidualVectorQuantizer` wraps.

**`ResidualVectorQuantizer` (vq.py:21–167).** Holds an `input_proj` and `output_proj` plus a `ResidualVectorQuantization` of `n_q` `VectorQuantization` layers.
- *Projections (vq.py:73–84):* `input_proj`/`output_proj` are `Conv1d(·, ·, kernel=1, bias=False)` — pointwise 1×1 channel maps, here `512↔256` (latent dim 512, codebook dim 256). When `input_dimension == dimension` and `force_projection=False` they collapse to `Identity`; in the split quantizer `force_projection=True` forces real 1×1 convs on both stacks.
- *encode (vq.py:126–139):* short-circuits to an empty `(B, n_q, 0)` int64 tensor if `T==0` (vq.py:132); else `x = input_proj(x)` then `codes = self.vq.encode(x, n_q)`; `codes` come back `[K, B, T]` and are `transpose(0,1)` → `[B, K, T]`. No projection is applied on the encode codes (codes are indices).
- *decode (vq.py:141–151):* `[B, K, T]` codes are `transpose(0,1)` → `[K, B, T]` (what `vq.decode` expects), `vq.decode` sums the per-level dequantized centroids, then `output_proj` maps `256→512`.
- *forward (vq.py:95–124):* training/loss path — `input_proj`, optional `q_dropout` (random `n_q` in `[1,n_q]`), `self.vq(x, n_q)` returns `(quantized, codes, commit_loss, metrics)`, optional `no_quantization_rate` straight-through mask, `output_proj`, bandwidth `bw = n_q * log2(bins) * frame_rate / 1000` (kbit/s, vq.py:114). Inference uses only `encode`/`decode`.

**`ResidualVectorQuantization` residual loop (core_vq.py:437–528).** The RVQ algorithm (Algorithm 1, SoundStream).
- *encode (core_vq.py:507–519):* `residual = x`; for each of the first `n_q` layers: `indices = layer.encode(residual)`, `quantized = layer.decode(indices)`, `residual = residual − quantized`, append `indices`. Stack → `[K, B, T]`. Each level quantizes what the previous levels failed to capture.
- *decode (core_vq.py:521–528):* `quantized = Σ_k layer_k.decode(codes_k)` — pure sum of per-level centroid lookups, no residual subtraction.

**`VectorQuantization` (core_vq.py:340–434).** One codebook level. `_rearrange_input` `[B,D,T]→[B,T,D]` (core_vq.py:399); optional `project_in`/`project_out` (Identity here since codebook_dim==dim); delegates to `EuclideanCodebook`; `_rearrange_output` back to `[B,D,T]`.

**`EuclideanCodebook` — the quantize/dequantize core (core_vq.py:105–337).**
- *quantize (core_vq.py:270–276):* flatten to `[N, D]`, then `dists = torch.cdist(x[None], embedding[None], p=2)[0]; codes = dists.argmin(dim=-1)`. This is the only "attention-free" similarity op in the codec: **L2 (Euclidean) cdist + argmin** = nearest-centroid assignment. No temperature, no sampling — deterministic argmin.
- *dequantize / decode (core_vq.py:289–297):* `F.embedding(codes, embedding)` — plain index-into-codebook lookup. Asserts codes are integer dtype.
- *codebook materialization (core_vq.py:178–186):* `embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]` — the stored EMA buffers are an un-normalized sum and a usage count; the actual centroid table is their ratio, cached in a non-persistent `_embedding` buffer. `epsilon = 1e-5`.
- *EMA / k-means init / dead-code replacement (core_vq.py:196–337):* training-only. `_run_kmeans` (core_vq.py:77–97) also uses `cdist`+`argmin`. Inference never touches these.

**`SplitResidualVectorQuantizer` (vq.py:170–322).** Two independent `ResidualVectorQuantizer`s.
- *construction (vq.py:195–204):* `rvq_first = RVQ(n_q=n_q_semantic=1, force_projection=True, q_dropout=False)`; `rvq_rest = RVQ(n_q=n_q−n_q_semantic, codebook_offset=1, force_projection=True, q_dropout=q_dropout)`. `codebook_offset=1` only renames metric keys; it does not change codebook content.
- *encode (vq.py:269–279):* `codes = rvq_first.encode(x)`; if `n_q > n_q_semantic`, `acoustic = rvq_rest.encode(x)` and `codes = cat([codes, acoustic], dim=1)` → `[B, 8, T]`. **Both stacks encode the same input `x`**, not a residual of each other — the split is semantic-vs-acoustic specialization, learned via separate projections, not a residual chain across the split.
- *decode (vq.py:281–287):* `quantized = rvq_first.decode(codes[:, :n_q_semantic])`; if more codebooks present, `quantized += rvq_rest.decode(codes[:, n_q_semantic:])` — the two latents are **summed** (vq.py:286). Mirrors `MimiModel.decode` reconstructing the 12.5 Hz latent.
- *cardinality / codebooks (vq.py:289–322):* `total_codebooks = 32`, active `num_codebooks = 8` after `MimiModel.set_num_codebooks(8)` (which routes to `rvq_rest.set_num_codebooks(7)`, vq.py:315–317). `cardinality = bins = 2048` per codebook (vq.py:319–322), with an assert that both stacks share the same `bins`.

For LFM2-Audio: `dim=256, n_q=32 (32 codebooks built), bins=2048`, projected `512↔256`, active 8 = **1 semantic + 7 acoustic** (`ARCH_1_MIMI_CODEC.md §2`). This is invoked through `MimiModel.encode`/`.decode` (`moshi/models/compression.py`), never directly by the LFM2-Audio model.

## Dtypes & shapes
| Stage | Input | Output |
|---|---|---|
| `Split…encode` (input_proj) | latent `(B, 512, T)` model dtype (bf16 cuda / f32 cpu / bf16 metal) | proj → `(B, 256, T)` same dtype |
| `EuclideanCodebook._quantize` cdist+argmin | `[N, 256]` float | `[N]` int64 indices (argmin) |
| `Split…encode` (full) | latent `(B, 512, T)` | **codes `(B, 8, T)` int64** (values `0..2047`) |
| `Split…decode` (F.embedding lookup) | codes `(B, 8, T)` int (u32 in Rust) | dequant latent `(B, 256, T)` float → output_proj → `(B, 512, T)` model dtype |

Promotions/notes: cdist/argmin run in the **module compute dtype** (bf16 on cuda/metal, f32 on cpu) — no special f32 upcast, the distance is just an L2 reduction. Codes are **int64** in Python (`.to(dtype=torch.long)` at the mapper, `mapper.py:230`), **u32** on the Rust/candle side. The codec weights (codebook EMA buffers, the two 1×1 conv projections) are **bf16 on disk**. `EOAudio=2048` is **not** a codebook entry — it is appended *after* `mimi.encode` by the mapper (`mapper.py:231`, `torch.full((codebooks,1), 2048)`), so valid codebook indices are `0..2047` (cardinality 2048) and `2048` is the out-of-band audio-EOS the LFM2-Audio head learns.

## Wiring
**Upstream (feeds this):** `MimiModel`'s encoder + framerate downsample produces the 12.5 Hz latent `(B, 512, T)` model dtype that enters `Split…encode`; on decode, `MimiModel` passes `(B, 8, T)` int codes into `Split…decode`. The quantizer is owned and driven by [moshi_compression](../models/compression.md). Its build/hyperparameters (`_quantizer_kwargs`: `dim=256, n_q=32, bins=2048`, `set_num_codebooks(8)`) come from [moshi_loaders](../models/loaders.md).

**Downstream (consumes this output):**
- Encode codes `(B, 8, T)` int → bubble up through [moshi_compression](../models/compression.md) `MimiModel.encode` to the training data path [data_mapper](../../data/mapper.md), which appends `EOAudio=2048` and stores the `audio_out` target `(8, L+1)` int64.
- Decode latent `(B, 512, T)` model dtype → consumed inside [moshi_compression](../models/compression.md) by the framerate upsample + decoder transformer + SEANet decoder to produce waveform; ultimately the [core_processor](../../processor.md) `decode()` dispatch and the demo streaming-decode path.
- Interface/dataclass contract: `BaseQuantizer` / `QuantizedResult` from [moshi_quant_base](base.md); the codebook engine is [moshi_core_vq](core_vq.md).

## Python ↔ Rust
There is **no in-tree Rust port of `vq.py`**. The Rust side reuses the published **`moshi` crate** (`moshi = "0.6"`, Cargo.toml:51), whose `mimi` module already contains the equivalent `SplitResidualVectorQuantizer`/`ResidualVectorQuantizer`/`EuclideanCodebook`. It is loaded via `moshi::mimi::load(path, Some(codebooks=8), device)` (`loader.rs:296–303`, `load_mimi`) and consumed only through `MimiModel`/`Mimi` — never called symbol-by-symbol from the LFM2-Audio Rust code.

| Python symbol | Rust |
|---|---|
| `SplitResidualVectorQuantizer` / `ResidualVectorQuantizer` / `EuclideanCodebook` | inside `moshi::mimi` (upstream crate, not vendored) |
| `quantizer.encode/decode` (via `MimiModel`) | `moshi::mimi::Mimi::encode` / `::decode` / `::decode_step` (`audio_out.rs:88–118`) |
| `set_num_codebooks(8)` | `Some(codebooks)` arg to `moshi::mimi::load` (`loader.rs:302`) |
| weight keys `quantizer.rvq_first.*` / `quantizer.rvq_rest.*` | matched **natively** by `moshi::mimi` |

**Deliberate divergence (PYTHON_VS_RUST.md §2.3 "Upstream reuse"):** `moshi::mimi` was chosen specifically because this checkpoint (`tokenizer-e351c8d8-checkpoint125.safetensors`) uses the `rvq_first`/`rvq_rest` naming (0 HF-format keys). `candle-transformers`' Mimi reads the alternate `…semantic_residual_vector_quantizer.*` / `…acoustic_…` names and **cannot load these weights** without a remap (Cargo.toml:39–51). Vendoring would re-port the identical algorithm — the opposite of "use upstream." Other deliberate divergences (ARCH_1 §7): device-agnostic vs CUDA-coupled; candle eager ops vs CUDA-graph capture (`CUDAGraphed` disabled off-cuda); F32 on CPU / bf16 on Metal vs CUDA bf16/fp32 — all numerically irrelevant to the argmin assignment, latency-relevant only.

## Precision / gotchas
- **Argmin is deterministic, dtype-tolerant.** Codebook assignment is `cdist`+`argmin` (core_vq.py:274–275) — no softmax, no sampling, no temperature. Cross-library f32-vs-bf16 differences only matter at the L2-distance tie boundary; a flipped argmin at a tie is the sole numerical risk, and it is rare and reconstruction-bounded. This is why the codec is robust to the f32-CPU / bf16-Metal floor.
- **`EOAudio=2048` ≠ codebook 2048.** Cardinality is exactly 2048 (indices `0..2047`, vq.py:319–322). The value `2048` is injected by the mapper after encode (`mapper.py:231`) and rejected on decode by the processor (`processor.py:174`: `audio_codes >= 2048` raises). The quantizer itself never sees or emits `2048`.
- **Both split stacks see the same `x`.** The semantic/acoustic split is *not* a residual chain across the split — `rvq_first` and `rvq_rest` each independently `encode(x)` and their decoded latents are summed (vq.py:274–277, 281–287). The residual subtraction happens only *within* each stack's `ResidualVectorQuantization` loop (core_vq.py:512–516).
- **Codes int64 (py) / u32 (rust).** The mapper forces `torch.long` (`mapper.py:230`); `F.embedding`/`EuclideanCodebook.decode` assert integer dtype (core_vq.py:293–295). Feeding floats, or any index `≥ cardinality`, is a hard error (and on CUDA an unrecoverable crash — vq.py:144 warns the bound can't be checked without a sync point).
- **`codebook_offset=1` is cosmetic.** On `rvq_rest` it only offsets *metric key names* (core_vq.py:493), not the codebook indices stored in `codes` — the acoustic codes are still `0..2047` per level.
- **Codebook table is a ratio of EMA buffers.** The on-disk state is `embedding_sum` + `cluster_usage`, not the centroids; the usable table is `embedding_sum / cluster_usage.clamp(min=1e-5)` (core_vq.py:181–183). A loader that reads `embedding_sum` as the codebook directly would be wrong — `moshi::mimi` reproduces this division.
