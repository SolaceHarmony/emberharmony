<!-- topic: Conformer Encoder -->
# CF03 · ConformerLayer / Conv / FeedForward
**Code:** `CF03` · **Source:** `model/conformer/modules.py` · **Rust:** `model/conformer/modules.rs` · **On the LFM2-Audio inference path:** yes

## Role
The per-layer building blocks of the FastConformer audio-in encoder: `ConformerLayer` (one Macaron-style Conformer block), its two sub-modules `ConformerFeedForward` and `ConformerConvolution`, and the streaming-capable `CausalConv1D` (an `nn.Conv1d` subclass). `ConformerEncoder` stacks `n_layers=17` of these `ConformerLayer`s over the subsampled, rel-pos-encoded mel features (`d_model=512`) to produce the contextualized acoustic encoding that the `audio_adapter` MLP lifts to the 2048-dim backbone embedding space. This is the body of the speech-understanding (audio-in) front-end; it does not touch audio generation.

## How it works
**ConformerLayer macaron structure** (`modules.py:153-226`). One block is `FF/2 → MHSA → Conv → FF/2 → out-norm`, all pre-LayerNorm with residual adds, with the two feed-forwards scaled by `fc_factor = 0.5` (`modules.py:84`, the half-step "Macaron" weighting from the Conformer paper). Exact order:
1. `residual = x`; `x = norm_feed_forward1(x)` (LayerNorm); `x = feed_forward1(x)`; `residual = residual + dropout(x) * 0.5` (`modules.py:167-170`).
2. `x = norm_self_att(residual)`; self-attention `self_attn(query=x,key=x,value=x, mask=att_mask, pos_emb=pos_emb)` (`modules.py:172-174`); `residual = residual + dropout(x)` (`modules.py:185`).
3. `x = norm_conv(residual)`; `x = conv(x, pad_mask=pad_mask)`; `residual = residual + dropout(x)` (`modules.py:198-202`).
4. `x = norm_feed_forward2(residual)`; `x = feed_forward2(x)`; `residual = residual + dropout(x) * 0.5` (`modules.py:204-206`).
5. `x = norm_out(residual)` (final LayerNorm; `modules.py:208`).

So there are **5 LayerNorms per block** (3 pre-norms for FF1/attn/conv, a pre-norm for FF2, and the output norm). Dropout is identity at inference. All norms are `nn.LayerNorm(d_model)` (eps=1e-5 default) — standard LayerNorm (mean-subtract + var-normalize + affine), **not** RMSNorm (RMSNorm appears in the LFM2 backbone/depthformer, not in the NeMo conformer).

**Self-attention selection** (`modules.py:104-144`). For this checkpoint `self_attention_model='rel_pos'`, so `self_attn` is a `RelPositionMultiHeadAttention` (Transformer-XL relative-position MHA; see [conformer_mha](CF02-RelPos-MHA)). It receives the rel-pos table `pos_emb` produced by the encoder's `RelPositionalEncoding`. `n_heads=8`, `d_model=512` ⇒ `d_k=64`, scale `1/sqrt(64)`. With `att_context_size=[-1,-1]` (unlimited) and offline batch-1, the encoder's `_create_masks` returns `att_mask=None`/`pad_mask=None` (full bidirectional attention — this is a non-causal acoustic encoder). The `rel_pos_local_attn` (Longformer) and `abs_pos` branches exist but are never constructed for LFM2-Audio.

**ConformerConvolution module** (`modules.py:229-344`). The gated convolution path, in channel-first layout:
- `x.transpose(1,2)` → `(B, d_model, T)` (`modules.py:315`).
- `pointwise_conv1`: `Conv1d(d_model → 2*d_model, k=1)` (`modules.py:271-278`).
- **GLU** over the channel dim: `F.glu(x, dim=1)` splits the `2*d_model` channels into halves `a,b` and returns `a * sigmoid(b)` → back to `d_model` channels (`modules.py:319-320`; `pointwise_activation='glu_'`, the paper-original gate).
- Optional pad-mask zeroing `masked_fill(pad_mask, 0)` (`modules.py:324-325`; no-op offline).
- `depthwise_conv`: a `CausalConv1D(d_model → d_model, k=conv_kernel_size, groups=d_model)` — **depthwise** (one filter per channel). For this checkpoint `conv_kernel_size=9`; `conv_context_size` defaults to `(k-1)//2 = 4` ⇒ **symmetric** padding `[4,4]`, so offline it is mathematically a plain "same"-padded depthwise conv. The `CausalConv1D` machinery is only asymmetric/causal when a streaming cache is threaded (see below). The assert `(kernel_size-1)%2==0` (`modules.py:251`) guarantees an odd kernel.
- `batch_norm`: `nn.BatchNorm1d(d_model)` for `norm_type='batch_norm'` (this config). At inference BN uses frozen running mean/var (eps 1e-5). (Other norm types — instance/layer/group/fused — are config-selectable but unused here; `modules.py:290-302`.)
- `activation`: `nn.SiLU()` (`modules.py:304`) — swish, **not** GLU here; the GLU was the pointwise-1 gate, SiLU is the post-BN nonlinearity.
- `pointwise_conv2`: `Conv1d(d_model → d_model, k=1)` (`modules.py:305-312`).
- `x.transpose(1,2)` back to `(B, T, d_model)` (`modules.py:340`).

**ConformerFeedForward** (`modules.py:360-381`). Plain 2-layer MLP `Linear(d_model→d_ff) → SiLU → dropout → Linear(d_ff→d_model)` (`modules.py:376-380`). `d_ff = d_model * ff_expansion_factor = 512*4 = 2048`. Activation is `nn.SiLU()` (passed as default, `modules.py:366`) — **not** GLU/SwiGLU; the conformer FF is a simple SiLU-MLP (contrast the backbone/depthformer SwiGLU).

**CausalConv1D** (`modules.py:393-471`). Subclasses `nn.Conv1d` but forces the inner conv to `padding=0` and applies padding manually in `update_cache` so it can support streaming. Padding policy (`modules.py:420-433`): `padding=None` ⇒ causal (`left=k-1`, `right=stride-1`); `padding=int` ⇒ symmetric `left=right=p`; `padding=[l,r]` ⇒ asymmetric (requires `l+r==k-1`, stride==1). Striding is rejected for non-symmetric padding (`modules.py:424-425`). `forward(x, cache)` = `update_cache(x,cache)` then `super().forward` (`modules.py:465-471`):
- Offline (`cache=None`): `F.pad(x, (left,right))` then conv; cache stays `None` (`modules.py:452-454`).
- Streaming (`cache` given): pad **right only**, prepend the cached left context via `torch.cat([cache, new_x], dim=-1)`, then roll the next cache to the original cache length, dropping `cache_drop_size` trailing (lookahead) steps (`modules.py:455-463`). This is the `cache_last_time` conv state threaded through `ConformerLayer.forward(..., cache_last_time)`.

**Streaming caches in ConformerLayer** (`modules.py:153,182-184,200-201,223-226`). When `cache_last_channel`/`cache_last_time` are non-None the layer returns the tuple `(x, cache_last_channel, cache_last_time)` and the attention/conv return their next caches. Offline LFM2-Audio passes both `None`, so the layer returns a bare tensor — the entire streaming path is cold.

## Dtypes & shapes
All compute is in **model dtype** (Python cuda default bf16; Rust CPU f32, Metal bf16). LayerNorm/BatchNorm normalize statistics in f32 internally then cast back. `d_model=512`, `d_ff=2048`, `n_heads=8`, `d_k=64`, `conv_kernel_size=9`.

| Tensor | dtype | shape |
|---|---|---|
| `ConformerLayer` input `x` | model (bf16/f32) | `(B, T', 512)` (T' = subsampled frames) |
| `pos_emb` (rel-pos table) | model | `(1, 2T'−1, 512)` |
| `att_mask` / `pad_mask` (offline) | — | `None` (unlimited ctx, batch-1) |
| `ConformerLayer` output `x` | model | `(B, T', 512)` |
| FF internal hidden | model | `(B, T', 2048)` |
| Conv after pointwise1 | model | `(B, 1024, T')` |
| Conv after GLU | model | `(B, 512, T')` |
| Conv depthwise out (symmetric pad) | model | `(B, 512, T')` |
| `cache_last_time` (streaming only) | model | `(B, 512, T_cache)` |
| `cache_last_channel` (streaming only) | model | `(B, T_cache, 512)` |

LayerNorm/BatchNorm promote to f32 for the mean/variance reduction and `rsqrt`, then cast to model dtype. No int/u32 codes are involved here (this is the audio-in understanding path, upstream of any RVQ/codes).

## Wiring
**Upstream** — fed by:
- [conformer_subsampling](CF05-Subsampling): the `dw_striding` ConvSubsampling 8× downsamples the `(B,128,T_mel)` mel into `(B, T', 512)` model-dtype features.
- [conformer_mha](CF02-RelPos-MHA)'s `RelPositionalEncoding`, which (in [conformer_encoder](CF01-Conformer-Encoder)) adds positional info to the subsampled features and emits the `pos_emb` table `(1, 2T'−1, 512)` consumed by every layer's `RelPositionMultiHeadAttention`.
- [conformer_encoder](CF01-Conformer-Encoder): owns the `nn.ModuleList` of 17 `ConformerLayer`s, threads `att_mask`/`pad_mask`/`pos_emb`, and (offline) calls each layer with `None` caches.

**Downstream** — consumes this output:
- [conformer_encoder](CF01-Conformer-Encoder): collects the stacked-layer output, transposes to `(B,512,T')`, and (no `out_proj`, `feat_out=-1`) returns `(B, T', 512)` model dtype.
- [model_mlp](MD03-Audio-Adapter-MLP): the `audio_adapter` MLP `Linear(512→2048) → GELU(erf) → Linear(2048→2048)` lifts the `_feat_out=512` encoding to backbone width `(ΣT', 2048)` model dtype (`lfm2_audio.py:87,346,353`).
- [model_lfm2_audio](MD01-LFM2AudioModel)/[model_lfm2_backbone](MD02-LFM2-Backbone): the adapted `audio_in_emb` is modality-scattered into the prefill sequence and fed to the LFM2 backbone.

## Python ↔ Rust
Symbol map (`modules.py` → `modules.rs`):
- `ConformerLayer.__init__/forward` → `ConformerLayer::new` / `forward` + `forward_cache` (`modules.rs:232-316`). The Rust splits offline `forward` (passes `None` caches) from `forward_cache` (threads `cache_last_channel`/`cache_last_time`), matching the Python tuple-return contract. `fc_factor` is a `const FC: f64 = 0.5`.
- `self_attn` dispatch → `enum SelfAttention { RelPos, Abs }` (`modules.rs:215-218`); only `rel_pos` is wired for this checkpoint (`abs_pos` constructible but unused, `rel_pos_local_attn`/Longformer not ported — off-path).
- `ConformerConvolution.forward` → `ConformerConvolution::forward`/`forward_cache` (`modules.rs:73-100`). **GLU is hand-rolled** as `a * sigmoid(b)` via `narrow` on dim 1 (`modules.rs:84-87`) since candle has no `F.glu`. `nn.SiLU` → `candle_nn::ops::silu`. `nn.BatchNorm1d` → `candle_nn::BatchNorm` with `forward_t(x, false)` (inference/frozen-stats mode).
- `ConformerFeedForward` → `ConformerFeedForward` (`modules.rs:19-42`); `Linear→silu→Linear`.
- `CausalConv1D` (`nn.Conv1d` subclass) → `CausalConv1D` struct wrapping a `padding=0` `Conv1d` plus manual `pad_with_zeros` (`modules.rs:130-199`), with a `CausalPadding` enum mirroring Python's `None`/`int`/`[l,r]` cases. `update_cache`/`forward`/`set_cache_drop_size` are 1:1.
- `reset_parameters_ff`/`reset_parameters_conv` are omitted: they are training-time uniform weight re-inits, while the port loads pretrained weights via `VarBuilder`.

**Deliberate divergences** (PYTHON_VS_RUST.md §2.2, §2.5): the Python `scaled_dot_product_attention` (CUDA-gated) inside the rel-pos MHA becomes hand-rolled eager SDPA in candle (`mha.rs`); the entire cache-aware streaming/ONNX-export apparatus around these blocks (§2.5) is cold off-path. The offline conformer is parity-verified to f32 floor: conformer layer 0 = 1.056e-6, final = 8.25e-7 vs Python golden tensors (PYTHON_VS_RUST.md §1.2).

## Precision / gotchas
- **LayerNorm not RMSNorm here.** This NeMo conformer uses standard `nn.LayerNorm` (5 per block) and `nn.BatchNorm1d` in the conv — the f32-multiply-order RMSNorm subtlety (PYTHON_VS_RUST.md §2.4) applies to the LFM2 backbone/depthformer, **not** to these modules. Do not "fix" the conv BatchNorm into a LayerNorm.
- **Symmetric vs causal conv.** The depthwise conv is named `CausalConv1D` but is configured **symmetric** (`padding=(k−1)/2=4`) for the offline non-causal encoder. It is *not* causal in LFM2-Audio's forward; the causal/asymmetric and `cache_drop_size` logic only activates when a streaming `cache_last_time` is threaded, which the offline path never does. Treating it as causal would be wrong.
- **Two distinct gates.** `ConformerConvolution` has a GLU (pointwise-1 → `a*sigmoid(b)`) *and* a SiLU (post-BatchNorm). The FF modules use SiLU only (no GLU/SwiGLU). Easy to conflate; they are different ops at different positions.
- **fc_factor 0.5 on both FFs** is load-bearing (Macaron half-steps); dropping it doubles the FF contribution.
- **BatchNorm running stats.** Inference must use frozen running mean/var (Rust `forward_t(x, false)`), not batch statistics — a batch-1 BN in training mode would normalize each frame to ~0 and corrupt the encoding.
- **Offline masking contract.** With batch-1 unlimited-context, masks are `None` and attention is fully bidirectional. A padded multi-clip batch would require the full `_create_masks` port (PYTHON_VS_RUST.md §5.1) — a documented gap, not a bug.
- **Cross-library f32 floor** (~1e-6): candle gemm reduction order and libm transcendentals differ last-bit from torch; the conformer agrees to that floor (8.25e-7 final), which is the irreducible limit of any cross-framework port, not a faithfulness defect.
