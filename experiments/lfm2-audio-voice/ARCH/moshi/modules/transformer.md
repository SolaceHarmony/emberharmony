# moshi_transformer
**Code:** `MO03` · **Source:** `moshi/modules/transformer.py` · **Rust:** `moshi crate transformer` · **On the LFM2-Audio inference path:** yes

## Role
Kyutai's streaming, CUDA-graphable transformer stack: `StreamingMultiheadAttention` → `StreamingTransformerLayer` → `StreamingTransformer`, wrapped by `ProjectedTransformer` (with optional in/out linear projections and a `[B,C,T]↔[B,T,C]` layout flip). On the LFM2-Audio path it is **not** the language model — LFM2-Audio uses its own backbone (`model_lfm2_backbone`) and depthformer (`model_transformer`). Instead, two instances of `ProjectedTransformer` sit **inside the Mimi codec** as the `encoder_transformer` and `decoder_transformer` (`loaders.py:302/305`), refining SEANet latents on either side of the RVQ bottleneck. The same file also defines `StreamingTransformer` for the off-path Moshi 7B LM (`lm.py:145/198`), which LFM2-Audio never instantiates.

## How it works
The on-path config is fixed by `_transformer_kwargs` (`loaders.py:65`): `d_model=512`, `num_heads=8` (→ `head_dim=64`), `num_layers=8`, `causal=True`, `context=250`, `layer_scale=0.01`, `gating="none"`, `norm="layer_norm"`, `positional_embedding="rope"`, `max_period=10000`, `dim_feedforward=2048`, `conv_layout=True`, `input_dimension=output_dimensions=[512]`. Because `d_model==input_dimension==output_dimension==512`, `input_proj` is absent and each `output_proj` is `nn.Identity` (`transformer.py:932-943`) — the projections exist for generality but are no-ops in Mimi.

**ProjectedTransformer.forward (`:945`).** `conv_layout=True` so input arrives `[B,512,T]` and is transposed to `[B,T,512]` (`:947`); no input projection; run `StreamingTransformer`; each `output_proj` (Identity) then transpose back to `[B,512,T]`; returns a **list** `[y]` (single output), which `compression.py:313/326` unpacks `(emb,) = ...`.

**StreamingTransformer.forward (`:868`).** Reads per-batch `offsets` from streaming state (`:874-876`). Positional scheme is pure `"rope"` so the **sinusoidal additive branch is skipped** (`:878` gate is false for `"rope"`; a single shared `RotaryEmbedding(max_period=10000)` is built at `:840` and threaded into every layer's attention). Then a plain loop over the 8 layers (`:886-896`; `checkpointing=False` at inference). On exit, advances `offsets += T` masked by `exec_mask` (`:898-902`) and casts back to the input dtype (`:903`).

**StreamingTransformerLayer.forward (`:763`).** Pre-LN residual blocks in the order **self-attention → (optional cross-attn, absent in Mimi) → feed-forward** (`:767-773`). An `ExitStack` enters `no_compile()` whenever `x.device.type != 'cuda'` (`:764-766`) — i.e. `torch.compile` is disabled off-CUDA, the seam the Rust port mirrors by never compiling.
- `_sa_block` (`:746`): `x_orig` saved; `norm1(x)` (LayerNorm); `self_attn(x,x,x)`; residual `x_orig.to(update) + layer_scale_1(update)`. `layer_scale_1` is a `LayerScale(512, init=0.01)` — a learnt per-channel diagonal scale `scale * x` (`:99-103`) that starts residual contributions near-zero.
- `_ff_block` (`:727`): `norm2(x)` (LayerNorm); since `gating=="none"`, `linear2(activation(linear1(x)))` with `activation=F.gelu` (the **default** `gelu`, i.e. exact erf-GELU, `:624`), `linear1: 512→2048`, `linear2: 2048→512`, both bias-free; residual via `layer_scale_2`. (When `gating!="none"`, this branch swaps in a `make_gating` GLU and the FF linears are `None` — not used by Mimi.)

**Normalization — LayerNorm, not RMSNorm, on the Mimi path.** `create_norm_fn("layer_norm", 512)` → `nn.LayerNorm(512, eps=1e-5)` (`:117`). The file *also* defines `RMSNorm` (`:52`) whose `_rms_norm` (`:37`) upcasts to an optional `dtype`, computes `var = eps + mean(x², dim=2)`, multiplies by `alpha * rsqrt(var)`, then casts back to `x_dtype` — and `LayerNormF32` (`:29`) which forces the LayerNorm into f32 then back — but neither is selected by `_transformer_kwargs` (`norm="layer_norm"`). So Mimi's transformer normalizes in the running dtype with standard `nn.LayerNorm` (eps 1e-5).

**Attention — `StreamingMultiheadAttention` (`:328`).** MHA (not GQA): 8 query heads = 8 kv heads, `head_dim=64`. Fused QKV via `in_projs` (one `nn.Linear(512, 3*512, bias=False)`; `weights_per_step=0` so `mult=1` and `apply_weights_per_step` just calls `in_projs[0]`, `:295`). `rearrange("b t (p h d) -> p b h t d", p=3, h=8)` splits Q,K,V (`:544`). RoPE applied to Q,K with `time_before_heads=False` (`:548`). KV are pushed through `_complete_kv` → `RingKVCache.complete` (`:227`), the CUDA-graph-friendly **ring buffer** of fixed `capacity=context=250`: writes K/V at `(end_offset + arange(T)) % capacity` via `scatter_`, returns the **full** `[B,8,250,64]` cache plus a `positions` vector where slots not yet written are marked `-1` (`:264-277`). Attention is `F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)` (`:562`) — torch SDPA with the implicit `1/sqrt(64)` scale. The causal+context mask is built explicitly as `attn_bias` (`:552-559`): valid where `pos_k>=0 AND (pos_q-pos_k)>=0 AND (pos_q-pos_k)<context` — i.e. **causal with a 250-step sliding window**. Output `rearrange("b h t d -> b t (h d)")` then `out_projs[0]` (`nn.Linear(512,512,bias=False)`). Streaming offsets advance by `T` under `exec_mask` (`:568-573`). A `_load_hook` (`:409`) re-splits the legacy fused `in_proj_weight`/`out_proj.weight` checkpoint tensors into the `in_projs.{i}`/`out_projs.{i}` ModuleList (here `mult=1`).

**RoPE (`rope.py:apply_rope`).** Half-split / interleaved-pair convention: `q.view(..., D//2, 2)` then real part `[...,0]`, imag `[...,1]`; `freqs = exp(arange(D//2) * (-log(10000)*2/D))`; `ts = offset + arange(T)`; rotate `(qr,qi)` by `(cos, sin)` of `freqs*ts` and re-stack to `[...,D]`. **The rotation math runs in f32** (`qr/qi/kr/ki = .float()`, `:50-53`) then casts back to the input dtype (`:64-66`) — an f32 island for numerical safety. `max_period=10000`.

**Streaming state.** `_MHAState` holds the `RingKVCache` + per-batch `offset`; `_LayerState`/`_TransformerState` hold CPU/tensor offsets; `reset(reset_mask)` zeroes offsets and the ring `end_offset` per-batch. Cross-attention plumbing (`_compute/_get/update_streaming_cross_attention_src`, `:482-518`) and `weights_per_step` multi-linear scheduling exist for the Moshi LM/depformer but are inert in the Mimi codec config.

## Dtypes & shapes
On-path = the Mimi encoder/decoder transformer. Weights bf16 on disk; compute = model dtype (Python cuda/bf16; Rust CPU f32, Metal bf16).

| Stage | Input | Output | Internal promotions |
|---|---|---|---|
| ProjectedTransformer (enc) | SEANet latent `[B,512,T']` model-dtype, `conv_layout` | `[B,512,T']` model-dtype | transpose to `[B,T',512]`; proj = no-op (512==512) |
| StreamingTransformer ×8 layers | `[B,T',512]` | `[B,T',512]` | LayerNorm in running dtype (eps 1e-5) |
| SMHA QKV | `[B,T',512]` | `[B,8,T',64]` ×3 | RoPE cos/sin in **f32** then cast back |
| RingKVCache | k,v `[B,8,T',64]` | keys/values `[B,8,250,64]`, positions `[B,250]` int64 (−1 = empty) | cache dtype = model dtype |
| SDPA | q,k,v + bool `attn_bias` | `[B,8,T',64]` | softmax in f32 internally (torch SDPA) |
| FF (gelu) | `[B,T',512]` | `[B,T',512]` | 512→2048→512, exact erf-GELU |
| ProjectedTransformer (dec) | quantized latent `[B,512,T']` | `[B,512,T']` | same, transpose back at end |

Positions/offsets = int64. No token ids here (this is a latent-space transformer, not an embedder).

## Wiring
This component lives **inside the Mimi codec**, sandwiched between SEANet and the RVQ quantizer.

**Encoder side (audio-in / training only on this path):**
- Upstream: [moshi_seanet](../seanet.md) `SEANetEncoder` → encoder latent `[B,512,T']` model-dtype, `conv_layout` (channels-first), CUDA-graphed by [moshi_compression](../models/compression.md).
- Downstream: [moshi_vq](../quantization/vq.md) `SplitResidualVectorQuantizer` consumes the refined `[B,512,T']` (it owns the 512↔256 input/output projections).

**Decoder side (audio-out, the active LFM2-Audio detok-via-Mimi path):**
- Upstream: [moshi_vq](../quantization/vq.md) `.decode()` → reconstructed latent `[B,512,T']` model-dtype, fed via [moshi_compression](../models/compression.md) `decoder_transformer`.
- Downstream: [moshi_seanet](../seanet.md) `SEANetDecoder` → waveform `f32 @ 24kHz`.

The whole codec is driven by [core_processor](../../processor.md) (`decode()` / Mimi dispatch) and, in the streaming demo, by [demo_chat](../../../demo/chat.md) via `mimi.streaming(1)`. RoPE tables come from [moshi_rope](rope.md); streaming-state machinery from [moshi_streaming](streaming.md); CUDA-graph capture from [moshi_util_compile](../utils/compile.md). Note: [moshi_lm](../models/lm.md) also imports `StreamingTransformer` from here but is **off-path** (different model).

## Python ↔ Rust
Per PYTHON_VS_RUST.md §2.3 ("upstream reuse instead of re-implementation"), the Mimi codec — including these two transformers — is **not re-ported**; `liquid-audio-rs` depends on Kyutai's own **`moshi` crate** (`moshi::mimi`), whose `quantizer.rvq_first`/`rvq_rest` weight names match this checkpoint. So the symbol map is Python `moshi/modules/transformer.py` → the equivalent module in the `moshi` Rust crate, not a hand-written file in `src/`. (`compare_symbols.py --scope core` excludes the vendored `moshi/**` by design — §4.)

Symbol-level: `ProjectedTransformer`→`ProjectedTransformer`; `StreamingTransformer`→`StreamingTransformer`; `StreamingTransformerLayer`→layer struct; `StreamingMultiheadAttention`→streaming MHA; `RingKVCache`→ring KV cache; `LayerScale`/`RMSNorm`/`LayerNormF32`→norm/scale structs; `apply_rope`→candle RoPE.

Deliberate divergences carried by the crate (ARCHAEOLOGY.md:108-115):
- **SDPA, not flash.** Python `F.scaled_dot_product_attention` (`:562`) + `torch.compile` engage flash only on CUDA; the crate runs eager `matmul + additive mask + softmax` — the **sdpa/no-flash** math, which is exactly the path the f32 golden tensors were dumped from. The `no_compile()`-off-CUDA guard (`:765`) is honored by simply never compiling.
- **Device-agnostic.** Python `RingKVCache` defaults `device="cuda"`, `dtype=bfloat16` (`:205-206`); the crate takes `&Device`/`DType` (CPU/f32 default, Metal/bf16 opt-in), no hardcoded device.
- **Mask form.** Python builds a boolean `attn_bias`; the eager Rust SDPA uses an additive `-inf` causal+context mask — same semantics.

## Precision / gotchas
- **This is LayerNorm, not RMSNorm.** The `_transformer_kwargs` pick `norm="layer_norm"` (eps **1e-5**), so the §2.4 "RMSNorm f32-multiply-order" subtlety from `model_transformer`/`model_lfm2_backbone` does **not** apply to the Mimi codec transformer. The `RMSNorm`/`LayerNormF32` classes in this file are defined but unselected on-path.
- **RoPE is an f32 island.** `apply_rope` upcasts q/k components to f32 for the cos/sin rotation, then casts back (`rope.py:50-66`) — independent of the bf16/f32 compute dtype; this is where the cross-library `cos/sin` last-bit floor lives (PYTHON_VS_RUST.md §1.4).
- **Ring cache `positions = -1` sentinel.** Unwritten ring slots return position `-1` (`transformer.py:277`); the causal `attn_bias` masks them via `pos_k >= 0` (`:556`). An off-by-one in the `last_offset = end_offset + T - 1` / `% capacity` wrap math (`:254-268`) would silently corrupt attention — it is the subtle part of the streaming port.
- **`context=250` sliding window.** Attention is causal **and** windowed to 250 steps (`delta < self.context`, `:558`); the KV ring capacity equals the context, so the cache and the mask are consistent by construction.
- **`gelu` is exact-erf.** FF uses the default `F.gelu` (erf form), not the tanh approximation — matters for matching audio-quality numerics.
- **conv_layout flip.** Forgetting the `[B,C,T]↔[B,T,C]` transpose pair (`:947`/`:954`) would feed the transformer time-as-channels; it brackets the whole forward and must round-trip.
- **No EOAudio / special tokens here.** This transformer operates on continuous latents; special-token logic (EOAudio=2048, sampling) belongs to `model_lfm2_audio` / the depthformer, not this component.
