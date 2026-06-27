# model_lfm2_backbone (Rust port)
**Source:** `liquid-audio-rs/src/model/lfm2_hf.rs` · **Python:** `transformers.Lfm2Model` (imported by `upstream-liquid-audio/src/liquid_audio/model/lfm2_audio.py`) · **On the LFM2-Audio inference path:** yes

> Companion to [`ARCH/model/lfm2_backbone.md`](../../ARCH/model/lfm2_backbone.md).
> The original documents the *external* HF `transformers.Lfm2Model` and uses the
> Rust port as the readable spec; this documents the Rust port
> (`liquid-audio-rs/src/model/lfm2_hf.rs`) directly and where it diverges from
> the HF Python source.

## Role
`lfm2_hf::Model` (`lfm2_hf.rs:381`) is the LFM2 **hybrid backbone** in the Rust
port — the 16-layer sequence model that is the brain of LFM2-Audio. It is the
bare `Lfm2Model` (not `Lfm2ForCausalLM`): it consumes a pre-assembled
`inputs_embeds` stream (text + audio-in + audio-out rows already scattered by
modality) and returns the all-position `last_hidden_state` after a final
`embedding_norm`. It carries no LM head — both consumers (the text head and the
depthformer audio head in `lfm2_audio.rs`, and the detokenizer) do their own
projection against this hidden state. The hybrid design is the LFM2 signature:
most layers are cheap gated **short convolutions** (`conv_L_cache=3`),
interleaved with a minority of **GQA attention** layers, which is what makes
LFM2 cheap to run on CPU.

Adapted from candle-transformers' `models/lfm2.rs` onto plain `candle_nn`
(candle 0.9.x has `mimi`/`quantized_lfm2` but not full-precision `lfm2`). The
header comment (`lfm2_hf.rs:1-21`) records the differences from the candle
reference that liquid_audio needs.

## How it works (Rust)
The backbone runs over `inputs_embeds` of shape `(1, L, 2048)` and is invoked
two ways from `lfm2_audio.rs`: once at prefill (full sequence, returns hidden +
a fresh `Cache`), then once per generated token with a 1-token `in_emb` and the
threaded `cache` (`lfm2_audio.rs:816`/`:875`). Layout config from
`config.json/lfm` → `Lfm2Config` (`lfm2_hf.rs:39`): `hidden_size=2048`,
`num_hidden_layers=16`, `num_attention_heads=32`, `num_key_value_heads=8`,
`head_dim=64` (`2048/32`), `rope_theta=1e6`, `norm_eps=1e-5`,
`conv_l_cache=3`, `vocab_size=65536`, SwiGLU FFN with `block_ff_dim=12288` and
`block_auto_adjust_ff_dim=true`. Defaults are encoded as `serde` default fns
(`:72-79`): `d_kv_heads=8`, `d_eps=1e-5`, `d_theta=1e6`, `d_maxpos=128_000`,
`d_lcache=3`, `d_ffn_mult=1.0`, `d_mult_of=256`, `d_true=true`.

**Layer schedule (hybrid).** `layer_types: Vec<LayerType>` (`:57`) is a
length-16 list; `LayerType::FullAttention` at positions `2,5,8,10,12,14` (6
attention layers) and `LayerType::Conv` for the remaining 10. `DecoderLayer`
(`:347`) is pre-norm with two residual adds (`:366-377`):
```rust
let h = self.operator_norm.forward(x)?;
let h = match &self.kind {
    LayerKind::Attention(a) => a.forward(&h, index_pos, block_idx, cache, add_mask)?,
    LayerKind::ShortConv(c) => c.forward(&h, block_idx, cache)?,
};
let x = (h + residual)?;
let h = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
h + residual
```
After all 16 layers a final `embedding_norm` `RmsNorm` is applied (`:420`) — the
only top-level norm, since this is a bare `Lfm2Model` with no `lm_head`.

**`RmsNorm` (normalize-in-f32, then weight-multiply, then cast).** Reused from
`crate::model::transformer::RmsNorm` (`:30`) — the same hand-composed norm as
the depthformer. `operator_norm`, `ffn_norm`, the per-head `q_norm`/`k_norm`,
and `embedding_norm` are all `RmsNorm` with `eps=1e-5`. The exact op order
(`transformer.rs:135-141`): upcast `x` to f32 →
`x * recip(sqrt(mean(x², -1, keepdim) + eps))` → multiply by the weight (also
upcast to f32) → cast back to input dtype. The weight multiply happens **in f32
before the down-cast** — the faithful LFM2/`liquid_audio` order, distinct from
`candle_nn::RmsNorm` which casts back *before* the weight multiply. The comment
at `:26-29` records that the differentiable `RmsNorm` (basic ops) is used, *not*
`candle_nn::RmsNorm`, because the fused `ops::rms_norm` severs autograd — the
backbone is trained.

**Attention layer — GQA, qk-RMSNorm, half-split RoPE, eager SDPA-math**
(`lfm2_hf.rs:187-271`). `Attention::new` (`:200`) builds bias-free
`linear_no_bias`: `q_proj` → `32*64`, `k_proj`/`v_proj` → `8*64` (8 KV heads,
4× grouping), `o_proj` back to 2048. `q_norm`/`k_norm` are `RmsNorm::new(hd,
cfg.norm_eps, …)` — per-head RMSNorm over `head_dim=64`, applied *before* RoPE
(`:234-235`). `rope` (`:218`) uses `candle_nn::rotary_emb::rope_slow` — the
**NeoX half-split** rotation (not the interleaved `rope_i` the depthformer
uses). The cos/sin tables are built in `Cache::new` (`:113-134`) from
`inv_freq[i] = 1/theta^(i/head_dim)` for even `i`, `theta=1e6`,
`max_position_embeddings=128_000`, cast to the model dtype. KV cache
concatenates new k/v along the time axis (dim 2) per layer (`:239-250`) when
`cache.use_kv_cache && index_pos > 0`. GQA is realized by `repeat_kv` (`:142`)
expanding the 8 KV heads to 32 (`n_head/n_kv = 4`, `:252-253`). The score is the
**eager** SDPA math (`:255-270`): upcast q,k,v to f32 →
`att = q·kᵀ / sqrt(head_dim)` (scale `1/8`) → add the additive causal mask
(`-inf` above the diagonal; **skipped when `seq_len==1`** since a single new
token attends to all of history, `:261`) or the `add_mask` override →
`candle_nn::ops::softmax` over the last dim (differentiable, not the fused
`softmax_last_dim`, `:266`) → `att·v` → cast back to model dtype → reshape
`(b, seq, 2048)` → `o_proj`. This is the `sdpa`/no-flash path, exactly the path
the f32 golden tensors were dumped from.

**`ShortConv` layer — gated causal depthwise conv** (`lfm2_hf.rs:273-339`).
This is the LFM2 `conv_L_cache` operator. `in_proj` maps `2048 → 3*2048` and
`forward` (`:293`) splits into three streams along the channel axis: `bgate`
(`:296`), `c` (`:297`), `x_proj` (`:298`). Compute `bx = bgate * x_proj`
(elementwise gating, `:299`). Then:
- **Multi-token path** (`seq_len > 1`, prefill, `:317-334`): a candle `Conv1d`
  with `Conv1dConfig { padding: l_cache-1=2, groups: hidden_size=2048 }` over
  `bx`, narrowed back to `seq_len` (`:323`) to keep it causal. The conv state
  for the next step is the last `l_cache` frames of `bx` (left-padded with zeros
  if `seq_len < l_cache`, `:324-333`).
- **Single-step path** (`seq_len == 1`, decode, `:301-316`): instead of a
  `Conv1d`, it keeps a rolling conv state `(b, hidden_size, l_cache)` in
  `cache.conv_states[block_idx]`, shifts in the new `bx` (`:308-312`), and
  computes `(state * conv_weight.unsqueeze(0)?)?.sum_keepdim(2)?` (`:316`) — a
  gather-mul-sum equivalent of the conv. `conv_bias=false`.

The result is gated again by `c`, transposed back, and passed through
`out_proj` (`:337-338`). This is the candle reimplementation of the CUDA
`causal_conv1d` kernel.

**SwiGLU FFN (`Mlp`, `lfm2_hf.rs:164-185`).** Three bias-free linears named
`w1`/`w3`/`w2` (matching the `lfm.*` checkpoint keys): `gate = silu(w1(x))`,
`up = w3(x)`, `out = w2(gate * up)` (`:180-184`). SiLU is the activation (LFM2
SwiGLU). The intermediate size is `Lfm2Config::intermediate_size` (`:91-100`):
from `block_ff_dim=12288`, with `block_auto_adjust_ff_dim`, reduce by `2/3` →
`int(2*12288/3)=8192`, scale by ffn multiplier (1.0), round **up** to
`block_multiple_of=256` via `scaled.div_ceil(self.block_multiple_of) *
self.block_multiple_of` → **8192**. The comment at `:88-90` notes the previous
`hidden*4` form only coincided when `block_ff_dim == 6*hidden`; it underrounds
the audio detokenizer (`block_ff_dim=3328` ⇒ 2304, not 2048).

The hidden state is the all-position tensor post-`embedding_norm`;
`lfm2_audio.rs` slices the **last** position for sampling (`h.i((0, seq_len -
1))`, `lfm2_audio.rs:818/877`).

## Dtypes & shapes (Rust)
| Input(s) | Output(s) |
|---|---|
| `inputs_embeds` model dtype (bf16 Metal / f32 CPU) `(1, L, 2048)` | `last_hidden_state` model dtype `(1, L, 2048)` |
| (decode step) `in_emb` model dtype `(1, 1, 2048)` + `Cache` | `(1, 1, 2048)` + updated `Cache` |
| weights on disk: **bf16** | — |

Internal promotions: every `RmsNorm` upcasts `x`→**f32** for the mean-square +
weight multiply, then casts back. Attention upcasts q/k/v→**f32** for `q·kᵀ`,
`softmax`, `att·v`, then casts the context back to model dtype (`:255-267`).
RoPE cos/sin tables are built in f32 then cast to model dtype (`:125-126`). On
CPU the model dtype is f32 throughout (candle has no CPU bf16 matmul); on Metal
it is bf16 with the f32 islands above. The KV cache stores k/v in model dtype;
the short-conv state in model dtype.

## Wiring (Rust)
**Upstream:** `lfm2_audio.rs::prefill_inputs` assembles `in_emb` `(1, L, 2048)`
model dtype by `index_select`-scattering three sources into one buffer by
`modality_flag` and feeds it here via `forward_embeds` (`lfm2_audio.rs:816`).
Those sources are: text token embeddings via `Model::embed` (the backbone's own
`embed_tokens` table, int ids → model dtype `(N, 2048)`); audio-in embeddings
from `ConformerEncoder` → `audio_adapter MLP` `(ΣT', 2048)`; audio-out
embeddings from the `SharedEmbedding`/`audio_embedding` `(L_ao, 2048)`. See
[`glm-version/model/lfm2_audio.md`](lfm2_audio.md) for the scatter.

**Downstream:** the `(1, L, 2048)` `last_hidden_state` feeds back into
`lfm2_audio.rs`: its last row is projected by `text_logits`
(`lfm.embed_weight().to_dtype(F32)?.matmul(...)`) → text logits `(65536,)`
(the **tied** text head reuses this backbone's `embed_tokens.weight`), and by
`depth_linear` → the depthformer audio head producing an 8-codebook audio frame
(see [`glm-version/model/transformer.md`](transformer.md)). The same backbone
(a separate weight set) is also the core of the
[`glm-version/detokenizer.md`](detokenizer.md) ISTFT vocoder (via
`forward_embeds` with an `add_mask` sliding window).

## Python ↔ Rust — where the port differs

| Python (`transformers.Lfm2Model`) | Rust (`lfm2_hf.rs::Model`) | Difference | Why |
|---|---|---|---|
| `Lfm2Model` (external HF class) | `Model` adapted from candle-transformers `models/lfm2.rs` | **in-tree port** | candle 0.9.x has no full-precision `lfm2`; the port is the readable spec. Cross-checked against `config.json` and the safetensors keys. |
| returns `last_hidden_state` (all positions) | `forward_embeds` returns all-position hidden post-`embedding_norm` | identical | both are the bare `Lfm2Model` (no `lm_head`). |
| `forward(inputs_embeds=…, past_key_values=…, use_cache=…)` | `forward_embeds(embeds, index_pos, &mut cache, add_mask)` | **`add_mask` override** | the detokenizer's sliding window needs a custom additive mask; the main path passes `None` and gets a causal mask. HF has no such hook. |
| `embed_tokens` (`nn.Embedding`) | `embed_tokens: Embedding` (`candle_nn::Embedding`) | identical | — |
| `Lfm2HybridConvCache` | `Cache { use_kv_cache, kvs: Vec<Option<(Tensor, Tensor)>>, conv_states: Vec<Option<Tensor>>, cos, sin }` | **struct layout** | per-layer KV (k/v tuples) + per-layer conv state + RoPE tables, all in one struct. Python's hybrid cache is the same fields under different names. |
| HF selects `flash_attention_2`/`sdpa` (`lfm2_audio.py:162`) | eager matmul + additive causal mask + softmax | **deliberate: kernel-free** | §2.2. The `sdpa`/no-flash math (not flash-attn's reordered online-softmax) is exactly the path the f32 goldens were dumped from. No custom CUDA kernels → byte-exact on CPU. |
| `causal_conv1d` CUDA kernel (when importable) | candle `Conv1d` (prefill) + gather-mul-sum (single step) | **deliberate: kernel-free** | §2.2. The single-step `state * conv_weight` then `sum_keepdim(2)` is the gather-mul-sum equivalent of the conv. |
| `RMSNorm` (HF `modeling_lfm2`) | `crate::model::transformer::RmsNorm` (hand-composed) | **deliberate: f32-multiply order + differentiable** | §2.4. `candle_nn::RmsNorm` casts back *before* the weight multiply; liquid_audio multiplies in f32 then casts. Also: the fused `ops::rms_norm` severs autograd — the backbone is trained, so the differentiable basic-op path is used (`:26-29`). |
| RoPE (`rope_theta=1e6`, NeoX half-split) | `candle_nn::rotary_emb::rope_slow` | **deliberate: `rope_slow` not `rope`** | the fused `rope` (`apply_op3_no_bwd`) severs autograd; `rope_slow` is the basic-op differentiable path. Same NeoX half-split rotation, same forward values (`:222-225`). |
| `softmax` in attention | `candle_nn::ops::softmax` | **deliberate: not `softmax_last_dim`** | the fused `softmax_last_dim` severs autograd; `ops::softmax` is differentiable. Same forward values (`:264-266`). |
| device/dtype hardcoded `cuda`/`bf16` | device/dtype-agnostic via `Lfm2Config` + `VarBuilder` | **deliberate** | §2.1. `(Cpu, F32)` for parity (no candle CPU bf16 matmul); Metal/bf16 opt-in. This is what makes the backbone actually run on CPU. |
| `LayerType` from `config.json` | `enum LayerType { FullAttention, Conv }` with `serde(rename_all="snake_case")` | identical | deserializes the same `"full_attention"`/`"conv"` strings. |
| `intermediate_size` formula (`block_auto_adjust_ff_dim`) | `Lfm2Config::intermediate_size` (`:91-100`) | **deliberate: `div_ceil`** | `scaled.div_ceil(self.block_multiple_of) * self.block_multiple_of` is the Rust idiom for Python's `ceil(scaled / multiple_of) * multiple_of`. `usize::div_ceil` is stable since Rust 1.73. The comment at `:88-90` records the previous `hidden*4` form's underrounding bug. |
| `q_proj`/`k_proj`/`v_proj`/`out_proj` | `q_proj`/`k_proj`/`v_proj`/`o_proj` | **name: `o_proj`** | the checkpoint key is `out_proj` (Python) but the Rust field is named `o_proj` and loaded via `vb.pp("out_proj")` — the weight path matches, the struct field name is local. Actually: `:209` uses `vb.pp("out_proj")`, so the weight path is `out_proj`; the field is named `o_proj` but the pp path is correct. |
| `q_layernorm`/`k_layernorm` | `q_norm`/`k_norm` (fields) loaded via `vb.pp("q_layernorm")`/`vb.pp("k_layernorm")` | **field name vs weight path** | the fields are named `q_norm`/`k_norm` but the weight paths are `q_layernorm`/`k_layernorm` (`:210-211`), matching the checkpoint. |
| weights under `lfm.` prefix (bare `Lfm2Model`) | weights under `vb.pp("lfm")` | identical | no `.model.` wrapper; final norm key is `lfm.embedding_norm`. Verified against the safetensors keys (`:389-392`). |

**Parity:** backbone hidden 6.558e-6 over 24 stacked layers (~2.7e-7/layer),
tied text logits 5.505e-6 (PARITY.md).

## Precision / gotchas (Rust-specific)
- **`RmsNorm` is `crate::model::transformer::RmsNorm`, not `candle_nn::RmsNorm`.**
  Two reasons: (1) the f32-multiply-then-cast order (bf16 faithfulness, §2.4),
  and (2) the differentiable basic-op path (the fused `ops::rms_norm` severs
  autograd — the backbone is trained). Same forward values. The comment at
  `:26-29` records both reasons.
- **`rope_slow` vs `rope`.** The fused `rope` (`apply_op3_no_bwd`) severs
  autograd; `rope_slow` is the basic-op differentiable path. Same NeoX
  half-split rotation. Comment at `:222-225`.
- **`softmax` vs `softmax_last_dim`.** The fused `softmax_last_dim` severs
  autograd; `candle_nn::ops::softmax` is differentiable. Comment at `:264-266`.
- **Backbone parity 6.558e-6** over 24 stacked layers (~2.7e-7/layer) is the
  cross-library f32 floor (candle gemm reduction order + libm transcendentals
  vs torch/BLAS), not a bug. The tied **text logits** parity is 5.505e-6.
- **RMSNorm bf16 order** is load-bearing: multiply the weight in f32 *before*
  casting down. Getting this backwards (cast-then-multiply, as
  `candle_nn::RmsNorm` does) diverges at bf16.
- **qk-norm is per-head and pre-RoPE** (dim=64, eps=1e-5). `q_norm`/`k_norm` are
  `RmsNorm::new(head_dim, …)` — applying at the wrong dimensionality (e.g. the
  full 2048) or after RoPE breaks parity.
- **Attention scale is `1/sqrt(head_dim)=1/8`**, on `head_dim=64` — not on the
  full 2048 width. `(self.head_dim as f64).sqrt()` (`:258`).
- **Single-token causal shortcut:** when `seq_len==1` the additive causal mask
  is skipped (`:261`) — the new token legitimately attends to all cached
  history. Only multi-token prefill builds the `-inf`-above-diagonal mask. The
  short-conv has a parallel `seq_len==1` rolling-state branch (`:301-316`).
- **`add_mask` override.** `forward_embeds` takes `add_mask: Option<&Tensor>`
  (`:415`). The main path passes `None` → causal mask. The detokenizer passes
  a sliding-window mask. This is the Rust-specific hook HF doesn't have.
- **Hybrid schedule must follow `layer_types` exactly.** 6 attention layers at
  indices `2,5,8,10,12,14`, 10 short-conv layers elsewhere. `DecoderLayer::new`
  (`:355`) reads `cfg.layer_types.get(layer_idx)` and picks `LayerKind::Attention`
  or `LayerKind::ShortConv`. A mismatched schedule loads wrong weights into wrong
  operators. `LayerType` deserializes from `"full_attention"`/`"conv"` via serde.
- **SwiGLU intermediate size** must come from the `2/3` + `div_ceil`-to-256
  formula (`intermediate_size`, `:91-100`), not a hardcoded `4*hidden`; the
  formula is what keeps the differently-sized detokenizer FFN
  (`block_ff_dim=3328` ⇒ 2304) correct. The previous `hidden*4` form only
  coincided for the main backbone.
- **`conv_bias=false`** by default (`:56`); the `ShortConv` has no conv bias
  (`Conv1d` built with `None` bias, `:319`).
- **`Conv1dConfig { padding: l_cache-1, groups: hidden_size }`.** The
  depthwise groups=2048 is load-bearing — without it the conv mixes channels,
  which is wrong for LFM2's per-channel short conv. `padding: l_cache-1=2`
  keeps the output the same length as the input; `narrow(2, 0, seq_len)`
  (`:323`) drops the right padding to keep it causal.
- **`conv_states` is per-layer `Option<Tensor>`.** The single-step path clones
  the state, shifts in the new `bx`, and writes it back (`:303-314`). The
  multi-token path seeds the state from the last `l_cache` frames of `bx`
  (`:324-333`), left-padded with zeros if `seq_len < l_cache`.
- **This bare `Lfm2Model` has no `lm_head`.** The text head is *tied* to
  `embed_tokens.weight` (exposed via `embed_weight()`, `:409`). Special tokens
  are read off the sampled text token in `lfm2_audio.rs`, not here:
  `128 = <|audio_start|>`, `130 = <|text_end|>`, `7 = <|im_end|>` (EOS).
- **Field names vs weight paths.** The `Attention` struct fields `o_proj`,
  `q_norm`, `k_norm` are loaded via `vb.pp("out_proj")`, `vb.pp("q_layernorm")`,
  `vb.pp("k_layernorm")` (`:209-211`) — the weight paths match the checkpoint,
  the field names are local. Don't confuse the two when reading the code.
- **`Cache::clear`** (`:136`) resets `kvs` and `conv_states` to `None`. The
  RoPE tables (`cos`/`sin`) are not cleared (they're constant). Used when
  starting a new generation turn with the same `Cache`.

## Cross-references
- [`ARCH/model/lfm2_backbone.md`](../../ARCH/model/lfm2_backbone.md) — Python
  original (which uses this Rust port as the readable spec).
- [`glm-version/model/transformer.md`](transformer.md) — the `RmsNorm` shared
  with this module, and the depthformer (which uses the *interleaved* `rope_i`,
  not the NeoX half-split `rope` used here).
- `liquid-audio-rs/PYTHON_VS_RUST.md` §2.1 (device-agnostic), §2.2 (CUDA kernels
  → portable candle ops), §2.4 (RMSNorm bf16 order).
- `liquid-audio-rs/parity/PARITY.md` — backbone hidden 6.558e-6, text logits
  5.505e-6.