# moshi_core_vq
**Code:** `QZ02` · **Source:** `moshi/quantization/core_vq.py` · **Rust:** `moshi crate` · **On the LFM2-Audio inference path:** yes

## Role
This is the **lowest level of the Mimi codec's quantizer** — the actual nearest-centroid lookup tables. `EuclideanCodebook` is a single learned codebook (one $K\times D$ table of centroids) that turns a continuous latent vector into the integer index of its nearest centroid (encode) and back into the centroid vector (decode). `VectorQuantization` wraps one codebook with optional in/out linear projections and the `[B,C,T]↔[B,T,C]` reshaping; `ResidualVectorQuantization` stacks several `VectorQuantization` layers in a **residual loop** so each level quantizes what the previous level left behind. The codebook **decode path is on the LFM2-Audio inference path**: every generated 8-code audio frame is turned back into a 256-d latent here before the SEANet decoder reconstructs the 24 kHz waveform. The encode path is on the *training-data* path (building `audio_out` target codes from reference speech), not on live inference of the model itself.

## How it works

### EuclideanCodebook — the table
The codebook is stored not as raw centroids but as two EMA buffers: `embedding_sum` (`[K,D]`) and `cluster_usage` (`[K]`). The usable centroid table is derived lazily as `embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:,None]` (`core_vq.py:178-186`, `epsilon=1e-5`). This is cached in a non-persistent `_embedding` buffer and invalidated on every training step (`register_buffer('_embedding', None)`, `core_vq.py:335`); at inference the property just returns the divided table. `K = codebook_size = 2048`, `D = codebook_dim = 256` for Mimi.

**Encode (quantize) — `cdist` + argmin** (`core_vq.py:270-287`). Input `[*, D]` is flattened to `[N, D]` (`_reshape_input`, einops `"... d -> (...) d"`, `core_vq.py:262-265`). `_quantize` computes the full pairwise Euclidean distance matrix with `torch.cdist(x[None], embedding[None], p=2)[0]` → `[N, K]`, then `dists.argmin(dim=-1)` → `[N]` integer codes (`core_vq.py:274-276`). Codes are reshaped back to the leading dims via `codes.view(*shape[:-1])`. There is an explicit `assert x.dtype.is_floating_point` on encode and an `assert not codes.dtype.is_floating_point` on decode (`core_vq.py:282, 293-295`).

**Decode (dequantize)** (`core_vq.py:289-297`). Pure table lookup: `F.embedding(codes, self.embedding)` gathers the centroid row for each code index → `[*, D]`. No arithmetic, no projection at this level.

**EMA update (training only, off inference path)** (`core_vq.py:317-335`). On the first training batch with `initialize=True`, `_init_embedding` runs k-means (`_run_kmeans`, 50 iters, itself `cdist`+argmin per iter, `core_vq.py:77-97`) to seed centroids. Each step then scatter-adds the per-code hit counts into a fresh `cluster_usage` and scatter-adds the assigned input vectors into a fresh `embedding_sum`, and folds both into the buffers with `_ema_inplace(buf, new, decay)` = `buf.mul_(decay).add_(new, alpha=1-decay)` (`core_vq.py:34-35`, `decay=0.99`). Dead-centroid replacement (`_check_expired_codes`/`_replace_expired_codes`, `core_vq.py:229-260`) swaps any centroid whose usage falls below `threshold_usage_ratio(0.1)·mean_usage` with a random batch vector, checked only every `check_unused_every=5` steps to limit CUDA sync points. None of this runs at inference (`self.training` is False).

### VectorQuantization — projection + layout
`forward`/`encode`/`decode` first `rearrange("b d n -> b n d")` so the channel dim is last (`core_vq.py:399-405`), apply `project_in` (a `nn.Linear(dim→codebook_dim)` or `Identity` when `codebook_dim==dim`), call the inner `EuclideanCodebook`, then on decode apply `project_out` (`nn.Linear(codebook_dim→dim)`) and rearrange back to `"b d n"` (`core_vq.py:407-419`). For Mimi the **dim↔codebook_dim projection lives one level up in `vq.py`** (the `512↔256` `input_proj`/`output_proj` `Conv1d`); inside `core_vq` the projection is `Identity` because `VectorQuantization` is constructed with `dim==codebook_dim==256`. The straight-through estimator `quantized = x + (quantized - x).detach()` and the `F.mse_loss` commitment term are **training-only** (`core_vq.py:425-429`).

### ResidualVectorQuantization — the residual loop
This is the RVQ core (Algorithm 1 of the SoundStream paper). `encode` (`core_vq.py:507-519`) walks the first `n_q` layers; for each layer it (a) encodes the running `residual` to an index, (b) immediately decodes that index back to a quantized vector, (c) subtracts it: `residual = residual - quantized`, and appends the index. The result is `torch.stack(all_indices)` → `[n_q, B, T]` (codebook-major). `decode` (`core_vq.py:521-528`) is the inverse sum: iterate over the codebook-major code tensor, `quantized = quantized + layers[idx].decode(layer_codes)`, accumulating the per-level centroids back into one latent `[B, D, T]`. The `forward` path additionally tracks losses/metrics and applies the encodec STE fix `quantized_out = x + (quantized_out - x).detach()` (`core_vq.py:496-497`), all training-only.

**The split-RVQ shape contract.** This module never sees the `512↔256` projection or the semantic/acoustic split — those are in [`moshi_vq`](vq.md) (`SplitResidualVectorQuantizer` → two `ResidualVectorQuantizer`s `rvq_first` n_q=1 + `rvq_rest` n_q=7, each owning one `ResidualVectorQuantization` here). What `core_vq` guarantees is the **codebook-major `[n_q, B, T]`** code layout that `vq.py` transposes to the canonical `[B, K, T]` (`vq.py:121, 137, 148`). For Mimi: `K=8` (1 semantic + 7 acoustic), `bins=2048`, latent `D=256`.

## Dtypes & shapes

| Stage | Input | Output |
|---|---|---|
| `EuclideanCodebook.encode` | latent `[*, 256]` **float** (model dtype: bf16 CUDA / f32 CPU / bf16 Metal) | codes `[*]` int64 |
| `cdist` distance matrix | `x[None] [1,N,256]`, `embedding[None] [1,2048,256]` | `dists [N,2048]` float |
| `argmin` | `dists [N,2048]` | `codes [N]` int64 |
| `EuclideanCodebook.decode` | codes `[*]` int (u32 in Rust) | centroids `[*, 256]` model dtype |
| `VectorQuantization.encode` | `[B,256,T]` float | `[B,T]` int64 |
| `VectorQuantization.decode` | `[B,T]` int | `[B,256,T]` model dtype |
| `ResidualVectorQuantization.encode` | residual `[B,256,T]` float | codes `[n_q,B,T]` int64 |
| `ResidualVectorQuantization.decode` | codes `[n_q,B,T]` int | latent `[B,256,T]` model dtype |

Internal dtype notes: the centroid table `embedding` is materialized at the **buffer dtype** (`embedding_sum`/`cluster_usage` are loaded from the bf16-on-disk checkpoint, computed at model dtype). `cdist` and `argmin` run in that same model dtype (no f32 upcast in the Python — there is no norm/softmax here, just a Euclidean distance, so the only precision lever is the global model dtype). Codes are **int64** in Python (`argmin` default), narrowed to **u32** in Rust for `index_select`. There is no f64 anywhere in this file (the f64 precision-sensitive front-end is the mel preprocessor, not the VQ).

## Wiring
**Upstream (encode path, training-data only):** SEANet-encoded + downsampled latent `[B,512,T]@12.5Hz` from [`MimiModel`](../models/compression.md) → `vq.py` `input_proj` `Conv1d 512→256` → residual `[B,256,T]` f32 enters `ResidualVectorQuantization.encode` here. Driven by [`moshi_vq`](vq.md) (`SplitResidualVectorQuantizer.encode`).

**Upstream (decode path, live inference):** generated 8-code audio frame `(8,)` int from the depthformer audio head ([`model_lfm2_audio`](../../model/lfm2_audio.md)) → demo reshapes to `[1,8,1]` → [`MimiModel.decode`](../models/compression.md) → `vq.py` `SplitResidualVectorQuantizer.decode` splits codes `[B,8,T]` into semantic/acoustic → `ResidualVectorQuantization.decode` here, **codes int (u32 in Rust)**.

**Downstream (decode path):** summed latent `[B,256,T]` model dtype → `vq.py` `output_proj` `Conv1d 256→512` → upsample → decoder transformer → SEANet decoder → waveform `f32 @24kHz`, all inside [`MimiModel`](../models/compression.md). The immediate consumer of `core_vq`'s decode output is [`moshi_vq`](vq.md)'s `output_proj`.

## Python ↔ Rust
Symbol-level: there is **no per-symbol Rust port of `core_vq.py`** in `liquid-audio-rs`. The entire `EuclideanCodebook`/`VectorQuantization`/`ResidualVectorQuantization` stack is **reused from the published `moshi` crate** (`moshi::mimi`), built wholesale by `moshi::mimi::load(path, Some(codebooks), device)` (`loader.rs:296-303`, `audio_out.rs` `MimiDetokenizer`). The choice is deliberate and documented in PYTHON_VS_RUST.md §2.3 ("Upstream reuse instead of re-implementation"): Kyutai's own crate is used because its weight-key naming `quantizer.rvq_first.*` / `quantizer.rvq_rest.*` matches this checkpoint exactly (candle-transformers' Mimi uses different keys and cannot load it — ARCH_1 §6). So the mapping is module-level, not line-level:

| Python (`core_vq.py`) | Rust |
|---|---|
| `EuclideanCodebook` (cdist+argmin lookup) | `moshi::mimi` internal codebook (candle `index_select`-based gather, eager distance) |
| `VectorQuantization` (project + reshape) | `moshi::mimi` internal VQ |
| `ResidualVectorQuantization` (residual loop) | `moshi::mimi` internal RVQ |
| `RVQ.decode(codes)` | reached via `Mimi::decode` / `Mimi::decode_step` (`audio_out.rs:88-93, 113-118`) |
| `RVQ.encode(x)` | reached via `Mimi::encode` (`audio_out.rs:98-102`) |

Deliberate divergences (vs the codec's general Python↔Rust story): **device-agnostic** (Python codec is CUDA-coupled, won't boot CPU-only; Rust runs `(Cpu,F32)` by default, Metal opt-in); **candle eager ops, no CUDA graphs** (the `CUDAGraphed` wrapping in `compression.py` is GPU-only and absent in Rust — numerically irrelevant here, the VQ is just distance + gather); **bf16 vs f32 compute floor** (Python codec runs module bf16/f32 on CUDA; Rust uses F32 on CPU because there's no CPU bf16 matmul, bf16 on Metal). The training-only EMA/k-means/dead-code machinery (`_run_kmeans`, `_ema_inplace`, `_check_expired_codes`, `_average_tensors` distributed all-reduce) has **no inference counterpart** in either language.

## Precision / gotchas
- **Codes must be in `[0, bins-1] = [0, 2047]`.** `vq.py:141-146` warns that out-of-range codes cause "a dramatic CUDA crash" and the condition is deliberately *not* checked (to avoid a sync point). The model's audio vocab is `2049` wide because `2048 = EOAudio` is a sentinel the model emits but which **must never reach this codebook** — the processor's `decode()` is the gate that rejects codes outside `[0,2047]` before they hit Mimi (ARCH_1 §1; processor.py:165-177). EOAudio is consumed upstream as a turn-end signal, not decoded.
- **Codebook 0 is semantic, codebooks 1–7 are acoustic.** The residual ordering is load-bearing: index 0 (`rvq_first`) carries the most information, which is why LFM2-Audio's `audio_loss_weights` upweight codebook 0 (ARCH_1 §8). Mis-ordering the codebook-major `[n_q,B,T]` stack silently corrupts reconstruction.
- **int64 → u32 narrowing in Rust.** Python `argmin` yields int64; the Rust decode path explicitly `to_dtype(DType::U32)` because "RVQ `index_select` wants u32" (`audio_out.rs:89, 114`). Values are ≤2047 so no overflow, but a negative or out-of-range value would wrap rather than error.
- **No f32 upcast for the distance.** Unlike RMSNorm/softmax elsewhere in the model, `cdist`+`argmin` run at the model dtype with no promotion. At bf16 the distance matrix is bf16; because `argmin` only needs the *ranking* of distances (not their exact value) this is robust in practice, but two near-tied centroids can flip assignment between bf16 (CUDA/Metal) and f32 (Rust CPU). This is the cross-library precision floor for the codec — reconstruction is perceptually identical but not bit-exact across dtypes.
- **EMA-derived table, not stored centroids.** The on-disk weights are `embedding_sum`/`cluster_usage`, not centroids; the `embedding` property does the `clamp(min=1e-5)` division at load/first-use. A checkpoint with a stale cached `_embedding` (non-persistent) is fine because it's never serialized.
- **Old checkpoint key remap.** `_load_from_state_dict` (`core_vq.py:162-176`) maps legacy names (`inited→_initialized`, `cluster_size→cluster_usage`, `embed_avg`/`embed_sum→embedding_sum`); the Rust path sidesteps this entirely by loading the already-current `tokenizer-e351c8d8-checkpoint125.safetensors` keys.
