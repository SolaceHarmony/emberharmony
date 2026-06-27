<!-- topic: Conformer Encoder -->
# CF01 · ConformerEncoder
**Code:** `CF01` · **Source:** `model/conformer/encoder.py` · **Rust:** `model/conformer/encoder.rs` · **On the LFM2-Audio inference path:** yes

## Role
`ConformerEncoder` is the **audio-IN** acoustic encoder: a NeMo FastConformer (adapted from NVIDIA NeMo `conformer_encoder.py`) that turns the mel front-end's 128-bin log-mel features into 512-dim frame embeddings the LFM2 backbone can consume. It is what lets the model *listen* — microphone audio flows mel → conformer → adapter → backbone (it is **not** the Mimi codec, which is the audio-OUT path). The config used by LFM2.5-Audio-1.5B is `n_layers=17`, `d_model=512`, `n_heads=8` (head_dim 64), `conv_kernel_size=9`, `subsampling_factor=8` (`dw_striding`), `ff_expansion_factor=4` (d_ff=2048), `self_attention_model="rel_pos"`, `feat_in=128`. On inference it runs the **offline** forward only: one unpadded clip, unlimited attention context `[-1,-1]`, no streaming caches.

## How it works

**Forward orchestration** (`forward` → `forward_internal`, `encoder.py:491,537`). Input `audio_signal` is `(B, feat_in=128, T)` mel features. `forward` (L511) range-checks the channel dim, calls `update_max_seq_length` to grow the positional table (L523-527), then delegates to `forward_internal`. The offline path with no caches is the only one exercised at inference; streaming/export/reduction/stochastic-depth code is present but dormant.

**1. Transpose + subsample (8×).** `forward_internal` transposes to `(B, T, 128)` (L572) and runs `self.pre_encode` = `ConvSubsampling(dw_striding, factor=8)` (`subsampling.py`). dw_striding stacks `log2(8)=3` strided-conv stages on a `(B,1,T,F)` image: stage 0 is a full `Conv2d(1→C, k=3, stride=2, pad=1)` + ReLU; stages 1-2 are depthwise `Conv2d(C→C, k=3, s=2, p=1, groups=C)` + pointwise `Conv2d(C→C, k=1)` + ReLU. Each stage halves **both** time and the 128-freq axis (non-causal symmetric pad = (3-1)//2 = 1). After 3 stages: time `T' = floor((T+2·1−3)/2)+1` applied 3× (`calc_length`, `subsampling.py:545`, float div → floor, `ceil_mode=False`), freq `F' = 128 → 64 → 32 → 16`. The conv output `(B, C, T', 16)` is flattened over channel×freq and projected by `self.out = Linear(C·16, 512)` to `(B, T', 512)` (`subsampling.py:399`). The mel front-end pads `T` to a multiple of 8, so `T' = ceil(F/8)` frames (`ARCHAEOLOGY.md:27`).

**2. Relative positional encoding** (`pos_enc`, `mha.py:108`). `self.pos_enc(x, cache_len=0)` (L600) scales `x *= sqrt(d_model)` (`xscale`, L307/`mha.py:139`) and slices a **centered, Transformer-XL** rel-pos table of width `2·input_len−1`. `extend_pe` (`mha.py:119`) builds `pe` over positions from `+(L−1)` down to `−(L−1)` (positive = left/past, negative = right/future); `create_pe` (`mha.py:67`) computes the sinusoid in **fp32** (`div_term = exp(arange(0,d,2) · −ln(10000)/d)`, even=sin/odd=cos) then casts to model dtype. It returns `(x_scaled, pos_emb)` where `pos_emb` is `(1, 2·T'−1, 512)`. No additive injection — the rel-pos table is consumed inside attention, not added to `x` (this is the key difference from absolute PE).

**3. Masks.** `_create_masks` (`encoder.py:737`) builds `(pad_mask, att_mask)` with the convention **`True = IGNORE`**. For the offline case (`att_context_size=[-1,-1]`, full `padding_length`, no `offset`) `att_mask` is all-visible and `pad_mask` all-valid, so both invert to all-`False` (effectively `None`). The chunked/limited triangular (`triu(-left)`/`tril(right)`) and chunk-bucket logic is the streaming path only.

**4. N=17 ConformerLayers** (`modules.py:28`). Each layer is a **pre-LN macaron** block: `FF₁(½) → MHA → Conv → FF₂(½) → final-LN`, `fc_factor=0.5` (`modules.py:84,168-208`):
- `norm_feed_forward1` (LayerNorm) → `ConformerFeedForward` (Linear 512→2048, **SiLU**, Linear 2048→512); `residual += 0.5·FF(x)` (half-step macaron).
- `norm_self_att` (LayerNorm) → `RelPositionMultiHeadAttention`; `residual += attn`.
- `norm_conv` (LayerNorm) → `ConformerConvolution`; `residual += conv`.
- `norm_feed_forward2` (LayerNorm) → second FF; `residual += 0.5·FF₂(x)`.
- `norm_out` (LayerNorm) emits `(B, T', 512)`.

**Attention** (`RelPositionMultiHeadAttention`, `mha.py:315`). MHA (not GQA): 8 heads, `d_k=64`. q/k/v come from separate `Linear(512,512)` (`forward_qkv`, `mha.py:204`). Transformer-XL relative scoring: `matrix_ac = (q+pos_bias_u)·kᵀ` (content) and `matrix_bd = (q+pos_bias_v)·pᵀ` with `p = linear_pos(pos_emb)` then **`rel_shift`** (`mha.py:362`: pad-left-1, reshape, drop-row — the diagonal-shift trick), truncated to key length; `scores = (matrix_ac + matrix_bd) / sqrt(d_k)` (`mha.py:451`). Softmax over keys (`forward_attention`, `mha.py:227`); when masked, `masked_fill(-INF_VAL=10000)` then a second `masked_fill(0)` post-softmax. `linear_out` projects back. `pos_bias_u/v` are zero-init `(8,64)` params (untied per the model config; init zeros). The model config has `use_pytorch_sdpa=False` → the **eager matmul** branch runs (not the SDPA branch).

**Convolution module** (`ConformerConvolution`, `modules.py:229`). Order: transpose to `(B,512,T')` → `pointwise_conv1` `Conv1d(512→1024,k=1)` → **GLU** over channels (`F.glu(dim=1)`, halves 1024→512) → optional pad-mask zero-fill → `depthwise_conv` = `CausalConv1D(512→512, k=9, groups=512)` → `BatchNorm1d(512)` → **SiLU** → `pointwise_conv2` `Conv1d(512→512,k=1)` → transpose back. The depthwise conv is symmetric-padded `(k−1)//2 = 4` left/right offline (`conv_context_size` resolves to `[4,4]`, `encoder.py:850`); `CausalConv1D` (`modules.py:393`) only becomes truly causal when fed a streaming cache (then left-pad k−1, right-pad stride−1). `assert (kernel_size−1)%2==0` requires odd `k` (9 ✓).

**Stochastic depth, reduction, out_proj.** Layer-drop (`encoder.py:641`) is training-only (`self.training` gated). `reduction_subsampling` is `None` for this model. `out_proj` is `None` because `feat_out` is not set distinct from `d_model`, so the encoder output stays 512-dim. Final transpose returns `(B, 512, T')` plus `length` (int64).

## Dtypes & shapes

| Stage | dtype | shape |
|---|---|---|
| **Input** mel features `audio_signal` | model dtype (bf16/f32; cast from mel-bf16 via `.to(text_emb.dtype)`) | `(B, 128, T)` |
| `length` | int64 | `(B,)` |
| after transpose | model dtype | `(B, T, 128)` |
| subsampling conv image | model dtype | `(B, 1, T, 128)` → `(B, C, T', 16)` |
| post-subsample (Linear) | model dtype | `(B, T', 512)` |
| sinusoid table `create_pe` | **fp32 internal**, cast to model dtype | — |
| `pos_emb` (rel, centered) | model dtype | `(1, 2·T'−1, 512)` |
| per-layer hidden | model dtype | `(B, T', 512)` |
| attention `scores` | model dtype (softmax in model dtype) | `(B, 8, T', T')` |
| **Output** `audio_enc` | model dtype | `(B, 512, T')` |
| **Output** `audio_in_len` | int64 | `(B,)` |

Promotions: the rel-pos **sinusoid is computed in fp32** then cast down (`mha.py:71`); softmax/matmul run in **model dtype** here (no fp32 upcast inside attention on this path — the fp32-pin in `forward` only triggers under torch autocast, which inference does not use). On Rust CPU, model dtype is f32 (no CPU bf16 matmul); Metal runs bf16; Python default cuda/bf16.

## Wiring

**Upstream — feeds this encoder:**
- `core_processor` ([processor.py](CO01-Processor-ChatState)) → mel front-end `conformer_processor` ([conformer/processor.py](CF04-Mel-Frontend)) produces per-clip log-mel `(128, F)` (stored bf16 in `ChatState`); `model_lfm2_audio` ([model/lfm2_audio.py](MD01-LFM2AudioModel), `lfm2_audio.py:339-346`) pads the clip list and calls `self.conformer(padded_audio_in.mT.to(text_emb.dtype), audio_in_lens)`. **Edge:** mel features `(B, 128, T)` model dtype + `length` int64.

**Downstream — consumes this output:**
- `model_lfm2_audio` ([model/lfm2_audio.md](MD01-LFM2AudioModel)) unpads via the length mask and concatenates to `audio_enc_concatenated` (`lfm2_audio.py:349-350`). **Edge:** `(ΣT', 512)` model dtype.
- `model_mlp` audio_adapter ([model/mlp.py](MD03-Audio-Adapter-MLP), `lfm2_audio.py:87,353`) — MLP **512→2048** (GELU exact-erf, hidden=2048) maps the encoder output to backbone width, giving `audio_in_emb (ΣT', 2048)` scattered into the AUDIO_IN slots of the backbone sequence. **Edge:** `(ΣT', 512)` model dtype → `(ΣT', 2048)`.

## Python ↔ Rust

| Python (`encoder.py`) | Rust (`encoder.rs`) | note |
|---|---|---|
| `ConformerEncoder.forward`/`forward_internal` (offline) | `ConformerEncoder::forward` (L165) | **contract: one unpadded clip**, masks `None` |
| `pre_encode` `ConvSubsampling` | `ConvSubsampling` (subsampling.rs) | dw_striding 8×; `calc_length` → `out_lengths` |
| `pos_enc` `RelPositionalEncoding` | `RelPositionalEncoding` (mha.rs) | table recomputed per-forward, no `extend_pe` buffer |
| `RelPositionMultiHeadAttention` (eager branch) | hand-rolled SDPA + rel_shift (mha.rs) | matches `use_pytorch_sdpa=False` math |
| `_create_masks` | `create_masks`/`build_masks` (L356/L380) | explicit u8 loops, `1=IGNORE`; offline → all-zero |
| `forward_internal` streaming | `forward_streaming` (L189) | ported 1:1, off inference path |
| `forward_for_export`, `streaming_post_process`, `setup_streaming_params`, `get_initial_cache_state`, `_calc_context_sizes`, `change_attention_model`, `input_example` | same-named `pub fn`s | ported for inventory; cold at inference |

**Deliberate divergences** (`PYTHON_VS_RUST.md`):
- **§2.2 kernel-free attention:** Python's `scaled_dot_product_attention` (conformer) → Rust **hand-rolled SDPA + rel_shift**; the eager path matches the **sdpa/no-flash** math (the f32 golden tensors were dumped from exactly this), *not* flash-attn's reordered online-softmax.
- **§2.5 / §5(1) padded-batch masking is intentionally not ported.** Offline encodes one clip at a time (`_prefill`); `MaskedConvSequential`, per-step length tracking, and `_create_masks` (`att_mask`/`pad_mask`) exist only to neutralize padding, so the per-clip path is numerically equivalent (verified `prefill_parity`, 2 segments, **1.1e-6**). Do NOT feed a zero-padded batch into Rust `forward`.
- **§2.1 device-agnostic f32 floor:** Rust CPU = f32 (no CPU bf16 matmul), Metal = bf16; the encoder runs end-to-end on `Device::Cpu`, where the Python (as written) needs CUDA.
- **`change_attention_model`** only wires `rel_pos→rel_pos` in Rust; `abs_pos`/`rel_pos_local_attn` runtime swaps (which `load_state_dict` new weights) have no candle analog and error rather than no-op.

## Precision / gotchas
- **Rel-pos table fp32 then cast.** `create_pe` builds the sinusoid in fp32 (`mha.py:71`) and casts to model dtype; the table is **centered width `2L−1`** and **rel_shift** (pad-left, reshape, drop-row) realigns it — an off-by-one here silently misaligns every relative position. `center_pos = pe.size(1)//2 + 1` (`mha.py:146`).
- **Mask convention is `True = IGNORE`** (Python inverts at the end with `~`; Rust uses `1=IGNORE`). The masked softmax uses `−INF_VAL = 10000.0` (not `−inf`) and a **second** post-softmax `masked_fill(0)` to zero fully-masked rows (`mha.py:240-241`).
- **Offline masks are `(None, None)`** — correct only because `B==1` and all `T` frames are valid; a padded multi-clip batch would need the full `_create_masks` port (documented gap).
- **GLU halves channels** (1024→512) before the depthwise conv — the pointwise_conv1 deliberately doubles to 1024 to feed GLU; mis-sizing the depthwise `groups`/in-channels breaks silently.
- **BatchNorm1d in the conv module** runs in eval/running-stats mode at inference (model is `.eval()`); it folds into a per-channel affine — order matters relative to the SiLU that follows.
- **Length convention:** the conformer consumes the **full mel width** (padded to ×8) and `audio_in_len = mel2emb_len(audio_in_lens)` drives the unpad mask; the assert `(modality_flag==AUDIO_IN).sum() == mel2emb_len(...).sum()` (`lfm2_audio.py:330`) ties frame count to the scattered AUDIO_IN slots — a subsampling off-by-one would trip it.
- **No EOAudio / special tokens here** — those live on the audio-OUT depthformer/Mimi side; the conformer is a pure feature encoder with no vocabulary.
- **Parity:** conformer-through-mel 5.6e-7; conv-subsampling 5.6e-7; pos-enc 1.0e-6; layer-0 1.06e-6; **final 8.25e-7** (`PYTHON_VS_RUST.md:31-35`).
