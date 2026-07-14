# conformer_encoder (Rust port)
**Source:** `liquid-audio/src/model/conformer/encoder.rs` · **Python:** `upstream-liquid-audio/src/liquid_audio/model/conformer/encoder.py` · **On the LFM2-Audio inference path:** yes

> Companion to [`wiki/model/conformer/encoder.md`](../../../wiki/model/conformer/encoder.md).

## Role
`ConformerEncoder` (`encoder.rs:49`) is the **audio-IN** acoustic encoder in the
Rust port: a NeMo FastConformer that turns the mel front-end's 128-bin log-mel
features into 512-dim frame embeddings the LFM2 backbone can consume. It is
what lets the model *listen* — microphone audio flows mel → conformer →
adapter → backbone (it is **not** the Mimi codec, which is the audio-OUT path).
The config used by LFM2.5-Audio-1.5B is `n_layers=17`, `d_model=512`,
`n_heads=8` (head_dim 64), `conv_kernel_size=9`, `subsampling_factor=8`
(`dw_striding`), `ff_expansion_factor=4` (d_ff=2048),
`self_attention_model="rel_pos"`, `feat_in=128`. On inference it runs the
**offline** forward only: one unpadded clip, unlimited attention context
`[-1,-1]`, no streaming caches.

## How it works (Rust)

**Forward orchestration** (`ConformerEncoder::forward`, `encoder.rs:165`).
Input `audio_signal` is `(B, feat_in=128, T)` mel features. The contract is
**one unpadded clip** (effectively `B==1`, all `T` frames valid) — the
padded-batch machinery (`MaskedConvSequential`, per-step length tracking,
`_create_masks`) is intentionally not ported; masks are `None` (`:170`).
Callers with multiple segments must encode each individually (as `prefill_inputs`
does), which is numerically equivalent to the Python padded-batch+length-mask
path (verified `prefill_parity`, 2 segments, 1.1e-6) precisely because that
masking only exists to neutralize padding.

The `forward` path (`:165-176`):
1. **Transpose + subsample (8×).** `x = audio_signal.transpose(1, 2)` →
   `(B, T, 128)`; `x = self.pre_encode.forward(&x)` → `(B, T', d_model)`
   (`:166-167`). `pre_encode` is a `ConvSubsampling` (`dw_striding`, factor=8) —
   see [`glm-version/model/conformer/subsampling.md`](subsampling.md).
2. **Relative positional encoding.** `(x, pos_emb) = self.pos_enc.forward(&x)`
   (`:168`) — scales `x *= sqrt(d_model)` and returns the centered
   Transformer-XL rel-pos table `(1, 2·T'−1, 512)`. See
   [`glm-version/model/conformer/mha.md`](mha.md).
3. **N=17 ConformerLayers.** `for layer in &self.layers { x =
   layer.forward(&x, None, &pos_emb, None)?; }` (`:169-171`) — `att_mask` and
   `pad_mask` are `None` (full bidirectional attention). See
   [`glm-version/model/conformer/modules.md`](modules.md).
4. **Optional `out_proj`.** `if let Some(p) = &self.out_proj { x =
   p.forward(&x)?; }` (`:172-174`). `out_proj` is `None` because `feat_out` is
   not set distinct from `d_model` (`:99-103`).
5. **Transpose back.** `x.transpose(1, 2)` → `(B, d_out, T')` (`:175`).

**Construction** (`ConformerEncoder::new`, `:84`). Builds `pre_encode`
(`ConvSubsampling::new`), `pos_enc` (`RelPositionalEncoding::new` with `xscale =
Some(sqrt(d_model))` if `cfg.xscaling`, `:90`), `layers` (a `Vec<ConformerLayer>`
of `cfg.n_layers=17`), and `out_proj` (only if `feat_out > 0 && feat_out !=
d_model`). The `d_ff = cfg.d_model * cfg.ff_expansion_factor = 512*4 = 2048`
(`:85`); `conv_channels = if cfg.subsampling_conv_channels == 0 { cfg.d_model }
else { cfg.subsampling_conv_channels }` (`:86`).

The struct also carries a large block of config/streaming state
(`:54-81`) — `att_context_style`, `self_attention_model`, `att_context_size_all`,
`att_context_size`, `att_context_probs`, `conv_context_size`, `pos_emb_max_len`,
`max_audio_length`, `use_pad_mask`, `export_cache_support`, `streaming_cfg` —
mirroring the Python `__init__` attributes 1:1 for the streaming/export
methods. Cold on the offline forward but maintained for inventory completeness
(§2.5). The streaming config is computed at construction via
`Self::compute_streaming_cfg` (`:118-130`) with the offline defaults
(`att_context_style="regular"`, `conv_context_size` from `calc_context_sizes`,
pre-encode sampling frames `Pair(1, subsampling_factor)`,
`get_streaming_cache_size` `Pair(0, subsampling_factor + 1)`).

**`forward_streaming`** (`:189`) is the cache-aware streaming path — ported 1:1,
off the inference path. It threads `cache_last_channel` (attention KV),
`cache_last_time` (depthwise-conv state), and `cache_last_channel_len`, drops
`drop_extra_pre_encoded` pre-encoded frames per chunk, builds the masks via
`create_masks`, runs the layers with `forward_cache`, and returns the next
caches. The header comment (`:178-188`) documents the contract.

**`create_masks`/`build_masks`** (`:356`/`:380`) — explicit u8 loops,
`1=IGNORE`; offline → all-zero (effectively `None`). The chunked/limited
triangular (`triu(-left)`/`tril(right)`) and chunk-bucket logic is the streaming
path only.

**`calc_context_sizes`** (`:286`) resolves `att_context_size_all`,
`att_context_size`, `att_context_probs`, and `conv_context_size` from the
offline defaults (`att_context_size=None` ⇒ `[[-1,-1]]`,
`att_context_style="regular"`, `conv_context_size=None`). The
`ConvContextSize` enum (`:42`) mirrors Python's `"causal"`-or-`[left, right]`
union.

**Off-path inventory methods** (`:760+`): `forward_for_export`,
`streaming_post_process`, `setup_streaming_params`, `get_initial_cache_state`,
`change_attention_model`, `input_example`. Ported 1:1, cold at inference.
`change_attention_model` only wires `rel_pos→rel_pos` in Rust;
`abs_pos`/`rel_pos_local_attn` runtime swaps (which `load_state_dict` new
weights) have no candle analog and error rather than no-op.

## Dtypes & shapes (Rust)
| Stage | dtype | shape |
|---|---|---|
| **Input** mel features `audio_signal` | model dtype (f32 CPU / bf16 Metal; cast from mel via `to_dtype(text_emb.dtype)` upstream) | `(B, 128, T)` |
| after transpose | model dtype | `(B, T, 128)` |
| subsampling conv image | model dtype | `(B, 1, T, 128)` → `(B, C, T', 16)` |
| post-subsample (Linear) | model dtype | `(B, T', 512)` |
| sinusoid table `create_pe` | **f32 internal** (`div` built in f32, `:75-77`), cast to model dtype | — |
| `pos_emb` (rel, centered) | model dtype | `(1, 2·T'−1, 512)` |
| per-layer hidden | model dtype | `(B, T', 512)` |
| attention `scores` | model dtype (`softmax` upcasts to f32 via `ops::softmax`) | `(B, 8, T', T')` |
| **Output** `audio_enc` | model dtype | `(B, 512, T')` |

Promotions: the rel-pos **sinusoid is computed in f32** then cast down; the
`softmax` inside `mha.rs` uses `ops::softmax` (differentiable, upcasts to f32).
On Rust CPU, model dtype is f32 (no CPU bf16 matmul); Metal runs bf16; the
encoder runs end-to-end on `Device::Cpu`, where the Python (as written) needs
CUDA.

## Wiring (Rust)
**Upstream — feeds this encoder:**
- `crates/liquid-audio/src/processor.rs` → mel front-end produces per-clip log-mel
  `(128, F)` (stored bf16 in `ChatState`); `model/lfm2_audio.rs::prefill_inputs`
  casts each segment to model dtype and calls `self.conformer.forward(&seg)`
  (`lfm2_audio.rs:683`). See
  [`glm-version/model/conformer/processor.md`](processor.md) and
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md). **Edge:** mel
  features `(B, 128, T)` model dtype.

**Downstream — consumes this output:**
- `model/lfm2_audio.rs::prefill_inputs` — the `(B, 512, T')` output is
  transposed to `(T', 512)` and fed to the `audio_adapter` MLP. See
  [`glm-version/model/lfm2_audio.md`](model/lfm2_audio.md).
- `model/mlp.rs` audio_adapter — `MLP(512→2048, GELU-erf)` maps the encoder
  output to backbone width, giving `audio_in_emb (ΣT', 2048)` scattered into
  the `AUDIO_IN` slots of the backbone sequence. See
  [`glm-version/model/mlp.md`](model/mlp.md). **Edge:** `(ΣT', 512)` model dtype
  → `(ΣT', 2048)`.

## Python ↔ Rust — where the port differs

| Python (`encoder.py`) | Rust (`encoder.rs`) | Difference | Why |
|---|---|---|---|
| `ConformerEncoder.forward`/`forward_internal` (offline) | `ConformerEncoder::forward` (`:165`) | **contract: one unpadded clip** | the padded-batch machinery is intentionally not ported; masks are `None`. Callers with multiple segments encode each individually. |
| `pre_encode` `ConvSubsampling` | `ConvSubsampling` (`subsampling.rs`) | identical (dw_striding 8×) | — |
| `pos_enc` `RelPositionalEncoding` (mutable `pe` buffer) | `RelPositionalEncoding` (table recomputed per-forward, no buffer) | **deliberate: no mutable buffer** | Python registers/extends a non-persistent `pe` buffer; Rust returns a fresh table per call, sidestepping the `center_pos` slicing. See [`glm-version/model/conformer/mha.md`](mha.md). |
| `RelPositionMultiHeadAttention` (eager branch, `use_pytorch_sdpa=False`) | hand-rolled SDPA + `rel_shift` (`mha.rs`) | **deliberate: kernel-free** | §2.2. The eager path matches the `sdpa`/no-flash math (the f32 goldens were dumped from exactly this). Both Python branches (eager + SDPA) collapse to one Rust path. |
| `_create_masks` (Python inverts with `~`) | `create_masks`/`build_masks` (`:356`/`:380`) — explicit u8 loops, `1=IGNORE` | **deliberate: u8 not bool** | Rust uses `u8` masks with `1=IGNORE` (vs Python's `True=IGNORE`). Offline → all-zero (effectively `None`). |
| `forward_internal` streaming | `forward_streaming` (`:189`) | ported 1:1, off inference path | cold on the offline forward; maintained for inventory. |
| `forward_for_export`, `streaming_post_process`, `setup_streaming_params`, `get_initial_cache_state`, `change_attention_model`, `input_example` | same-named `pub fn`s (`:760+`) | **ported for inventory, cold at inference** | §2.5. `change_attention_model` only wires `rel_pos→rel_pos`; `abs_pos`/`rel_pos_local_attn` runtime swaps error. |
| `_calc_context_sizes` | `calc_context_sizes` (`:286`) + `ConvContextSize` enum (`:42`) | **deliberate: union → enum** | Rust's enum is the analog of Python's `"causal"`-or-`[left, right]` union. |
| `register_buffer`/non-persistent buffers | plain fields / recomputed tables | **deliberate: no buffers** | Rust has no `register_buffer`; config is held as plain fields, tables are recomputed per call. |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `VarBuilder` | **deliberate** | §2.1. f32 on CPU; bf16 on Metal. The encoder runs end-to-end on `Device::Cpu`. |
| `feat_out = -1` (Python) | `feat_out = 0` means "= d_model" (`:25`) | **deliberate: -1 → 0** | Rust `usize` can't be -1; `0` is the "not set" sentinel. `out_proj` is `None` when `feat_out == 0 || feat_out == d_model` (`:99`). |
| `ConvSubsampling` `feat_out` resolution (`feat_out=-1 ⇒ d_model`) | `ConvSubsampling::new` passes `feat_out=d_model` (`:88`) | identical | the encoder resolves the sentinel before constructing the subsampling. |

**Deliberate divergences** (PYTHON_VS_RUST.md):
- **§2.2 kernel-free attention:** Python's `scaled_dot_product_attention`
  (conformer) → Rust hand-rolled SDPA + `rel_shift`; the eager path matches the
  `sdpa`/no-flash math, *not* flash-attn's reordered online-softmax.
- **§2.5 / §5(1) padded-batch masking is intentionally not ported.** Offline
  encodes one clip at a time; `MaskedConvSequential`, per-step length tracking,
  and `_create_masks` exist only to neutralize padding, so the per-clip path is
  numerically equivalent (verified `prefill_parity`, 2 segments, 1.1e-6). Do
  NOT feed a zero-padded batch into Rust `forward`.
- **§2.1 device-agnostic f32 floor:** Rust CPU = f32 (no CPU bf16 matmul),
  Metal = bf16; the encoder runs end-to-end on `Device::Cpu`, where the Python
  (as written) needs CUDA.
- **`change_attention_model`** only wires `rel_pos→rel_pos` in Rust;
  `abs_pos`/`rel_pos_local_attn` runtime swaps error rather than no-op.

## Precision / gotchas (Rust-specific)
- **One-clip contract.** `forward` (`:165`) takes `(B, feat_in, T)` and passes
  `None` masks to every layer. The contract is **one unpadded clip** — a
  zero-padded batch would need the full `_create_masks` port (documented gap,
  §5.1). `prefill_inputs` encodes each segment individually, which is
  numerically equivalent.
- **Rel-pos table fp32 then cast.** `create_pe` (`mha.rs:71`) builds the
  sinusoid in f32 and casts to model dtype; the table is **centered width
  `2L−1`** and `rel_shift` (`mha.rs:284`) realigns it — an off-by-one here
  silently misaligns every relative position. The Rust avoids Python's
  `center_pos = pe.size(1)//2 + 1` by sizing the table exactly.
- **Mask convention is `1=IGNORE`** (u8). The masked softmax (`mha.rs:39`) uses
  `−INF_VAL = 10000.0` (not `−inf`) and a **second** post-softmax
  `where_cond(0)` to zero fully-masked rows (`:46`). Harmless on the offline
  path (mask ≈ None) but load-bearing for streaming/padded batches.
- **Offline masks are `None`.** Correct only because `B==1` and all `T` frames
  are valid; a padded multi-clip batch would need the full `_create_masks` port.
- **GLU halves channels** (1024→512) before the depthwise conv — the
  `pointwise_conv1` deliberately doubles to 1024 to feed GLU; mis-sizing the
  depthwise `groups`/in-channels breaks silently.
- **BatchNorm1d in the conv module** runs in eval/running-stats mode
  (`forward_t(x, false)`, `modules.rs:96`); it folds into a per-channel affine
  — order matters relative to the SiLU that follows.
- **Length convention:** the conformer consumes the **full mel width** (padded
  to ×8); `audio_in_len = mel2emb_len(audio_in_lens)` drives the unpad mask, and
  the `prefill_inputs` count check (`lfm2_audio.rs:747`) ties the frame count to
  the scattered `AUDIO_IN` slots — a subsampling off-by-one would trip it.
- **`feat_out = 0` means "= d_model"** (`:25`). Rust `usize` can't be -1; `0` is
  the "not set" sentinel. `out_proj` is `None` when `feat_out == 0 ||
  feat_out == d_model` (`:99`). Easy to misread as "no output projection
  configured."
- **No EOAudio / special tokens here** — those live on the audio-OUT
  depthformer/Mimi side; the conformer is a pure feature encoder with no
  vocabulary.
- **`pos_emb_max_len = 5000`** (`:19`) is the Python default; the
  `RelPositionalEncoding` table max length. The Rust recomputes the table per
  call, so this is just a config field (the table grows to the input length).
- **Parity:** conformer-through-mel 5.6e-7; conv-subsampling 5.6e-7; pos-enc
  1.0e-6; layer-0 1.06e-6; **final 8.25e-7** (PARITY.md).

## Cross-references
- [`wiki/model/conformer/encoder.md`](../../../wiki/model/conformer/encoder.md)
  — Python original.
- `liquid-audio/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.2 (kernel-free
  SDPA), §2.5 (off-path streaming stubs), §5.1 (padded-batch masking gap).
- `liquid-audio/parity/PARITY.md` — conformer final 8.25e-7, prefill 1.1e-6.
