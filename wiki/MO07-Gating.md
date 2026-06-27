<!-- topic: Mimi Codec — Modules -->
# MO07 · ActivationGating
**Code:** `MO07` · **Source:** `moshi/modules/gating.py` · **Rust:** `moshi crate` · **On the LFM2-Audio inference path:** yes

## Role
`ActivationGating` is the *gated* feed-forward (GLU-family) block for a Kyutai `TransformerLayer`: a `2*hidden`-wide input projection split into a gate half and a value half, `activation(gate) * value`, then a `hidden -> dim` output projection — the SwiGLU/GeGLU pattern with param budget pinned at `8·d²`. It exists so a transformer FFN can be swapped from the plain `linear1 -> act -> linear2` MLP to a gated variant by config string (`make_gating(name, ...)`). On the LFM2-Audio path the only Kyutai transformer that runs is the **Mimi codec** transformer (`moshi_compression`), and its config sets `gating="none"` (`loaders.py:74`), so `ActivationGating` is *not* instantiated for codec inference — the codec uses the non-gated branch. The gated branch is reached only by the Moshi 7B LM (`moshi_lm`, `gating="silu"`, `loaders.py:96,107`), which is reference-only. This module is "on path" as the FFN factory the codec transformer dispatches through, with the gated arm dormant.

## How it works
Two callers, one math. The dispatch lives in `TransformerLayer.__init__` (`transformer.py:670-699`): if `gating == "none"` it builds plain `nn.Linear` `linear1 (d_model -> dim_feedforward)` / `linear2 (dim_feedforward -> d_model)` and the FFN update is `linear2(activation(linear1(x)))` (`transformer.py:737`); otherwise it builds `self.gating = make_gating(name, d_model, dim_feedforward)` and the update is `self.gating(x)` (`transformer.py:743`). The Mimi codec hits the first arm; everything below is the second arm (`ActivationGating`).

**Hidden-dim sizing** (`gating.py:50-58`). Target param budget is `8·d²` (the dense-FFN-equivalent). A gated FFN has `2·h·d` (in-proj) + `h·d` (out-proj) `= 3·h·d` params, so `h = 8d/3`. Per the comment, Kyutai approximates `8/3 ≈ 21/8`:
- if `dim_feedforward == 4*dim`: `hidden = (21 * dim) // 8`  (integer floor; e.g. `dim=4096 -> 10752`)
- else: `hidden = (2 * dim_feedforward) // 3`

Then `linear_in = Linear(dim, 2*hidden, bias=False)` and `linear_out = Linear(hidden, dim, bias=False)` (`gating.py:60-61`). Both biasless. `make_gating` asserts `params <= 2*dim*dim_feedforward` (`gating.py:110-114`) — a budget guard, not a correctness check.

**Forward / the gate split** (`gating.py:14-22` kernel, `:25-36` generic — identical math). Given `x: (B, T, dim)`:
1. `x = F.linear(x, weight_in)` → `(B, T, 2*hidden)`  (`gating.py:17`)
2. `x = x.view(B, T, 2, hidden)`  (`gating.py:19`) — reshape **last** axis into `(2, hidden)`, so the `2*hidden` columns are split into the first `hidden` (index `[...,0,:]`) and the second `hidden` (index `[...,1,:]`). This is a contiguous *half-split*, not interleaved: gate = first half, value = second half.
3. `x = activation(x[..., 0, :]) * x[..., 1, :]`  (`gating.py:20`) → `(B, T, hidden)`. Activation is applied to the gate half only; elementwise multiply with the un-activated value half. For `silu` this is exactly SwiGLU.
4. `x = F.linear(x, weight_out)`  (`gating.py:21`) → `(B, T, dim)`.

**Activation selection** (`gating.py:85-93`, `_get_activation`): `sigmoid/tanh/relu` from `torch`; `leaky_relu/elu/gelu/silu/mish/softsign` from `torch.nn.functional`; `identity` → `nn.Identity()`; else raises. No GELU-erf/tanh distinction is forced here — `gelu` resolves to `F.gelu` (erf form by default). The Moshi LM uses `"silu"` (SwiGLU); the codec, were it gated, would inherit whatever string `loaders.py` passed (it passes `"none"`).

**Kernel vs generic, compile gating** (`gating.py:67-82`). `ActivationGating.forward` checks whether `linear_in` is a real `nn.Linear`: if so it calls `gating_forward_kernel(weight_in, weight_out, activation, x)` directly on the weight tensors; else (e.g. a quantized linear module) it calls `gating_forward_generic(linear_in, linear_out, ...)` which invokes the modules. `gating_forward_kernel` is wrapped in `@torch_compile_lazy` (`gating.py:13`) so it can be fused by `torch.compile`; during training the forward enters `no_compile()` via an `ExitStack` (`gating.py:70-72`) to disable compilation. Both `torch_compile_lazy` and `no_compile` are CUDA-oriented gates from `moshi/utils/compile.py` (`moshi_util_compile`) — no-ops off CUDA. There is **no normalization, no RoPE, no attention, no convolution, and no streaming state** in this module; it is a pure pointwise+matmul FFN. Norm (LayerNorm for the codec, `transformer.py:75`) is applied by the enclosing `TransformerLayer` before the FFN (pre-norm), not here.

## Dtypes & shapes
| Stage | Dtype | Shape |
|---|---|---|
| Input `x` | model dtype (codec: bf16 Metal / f32 Rust-CPU / bf16 cuda) | `(B, T, dim)` — codec `dim=512`, T = codec frames @ 12.5 Hz |
| `linear_in` weight | bf16 on disk; compute dtype = input | `(2*hidden, dim)` |
| after `linear_in` | model dtype | `(B, T, 2*hidden)` |
| after `view` | model dtype | `(B, T, 2, hidden)` |
| gate `activation(x[...,0,:])` · value `x[...,1,:]` | model dtype (no f32 upcast in this op) | `(B, T, hidden)` |
| `linear_out` weight | bf16 on disk; compute = input | `(dim, hidden)` |
| Output | model dtype | `(B, T, dim)` |

No internal dtype promotion happens inside this module: the gate multiply runs entirely in the model compute dtype (bf16/f32). f32 upcasts for norm/softmax live in the surrounding `TransformerLayer` (LayerNorm) and attention, not here. No int/u32 codes, no f64 — those are front-end (mel) and quantizer concerns elsewhere. For the codec, `dim_feedforward=2048`, `d_model=512`, so `dim_feedforward != 4*dim` is false (`2048 == 4*512`) — *if* gating were on it would take the `21*dim//8 = 1344` branch; but `gating="none"` means the codec actually uses dense `linear1 (512->2048)/linear2 (2048->512)`.

## Wiring
**Upstream (feeds this):** the enclosing `moshi_transformer` `TransformerLayer` — after pre-norm (LayerNorm in the codec, `transformer.py:75`) and the attention sublayer's residual add, the normalized hidden `(B, T, dim)` in model dtype is handed to the FFN. The transformer itself is part of the [Mimi codec](MM01-Mimi-Codec) encoder/decoder transformers (`moshi_compression`). See [moshi_transformer](MO03-Codec-Transformer) for the layer that calls `make_gating` / the `gating="none"` dense branch.

**Downstream (consumes this output):** the FFN output `(B, T, dim)` model dtype is residual-added back inside [moshi_transformer](MO03-Codec-Transformer) `TransformerLayer.forward`, then the `StreamingTransformer` stack returns to the [Mimi codec](MM01-Mimi-Codec) (`moshi_compression`) enc/dec transformer, which feeds the SEANet decoder / split-RVQ. The actual on-path FFN is the **non-gated** dense branch; `ActivationGating` proper only feeds back into the Moshi 7B LM stack (`moshi_lm`, off-path).

Id-map neighbors on the edges:
- in: `moshi_transformer` — `(B,T,512)` model dtype → this FFN
- out: `moshi_transformer` → `moshi_compression` — `(B,T,512)` model dtype
- (gated arm only, off-path): `moshi_lm` — `(B,T,4096)` / depformer `(B,T,1024)`

## Python ↔ Rust
The Rust counterpart is **reused wholesale from the external `moshi` crate** (Kyutai's own Rust port, `moshi = "0.6"`, `moshi-0.6.4/src/transformer.rs`), not re-ported into `liquid-audio-rs/src/`. `liquid-audio-rs` pulls the entire Mimi codec — SEANet, RVQ, *and* this FFN — through `moshi::mimi::Mimi` (`audio_out.rs:78`, `MimiDetokenizer`). So `gating.py` maps to the moshi crate's `Mlp` enum:

| Python (`gating.py` / `transformer.py`) | Rust (`moshi-0.6.4/src/transformer.rs`) |
|---|---|
| `gating == "none"` dense branch (`transformer.py:677-737`) | `Mlp::NoGating { linear1, linear2 }` (`:527,542-545,565`) |
| `ActivationGating` gated branch (`gating.py:39-82`) | `Mlp::Gating { linear_in, linear_out, activation }` (`:531,547-556,566-571`) |
| `hidden` sizing (`gating.py:55-58`) | `Mlp::new` (`:549-553`) |
| `view(B,T,2,-1)` + `act(x[...,0,:]) * x[...,1,:]` (`gating.py:19-20`) | `reshape((b,t,2,()))` + `i(..0).apply(act) * i(..1)` (`:569-570`) |
| `_get_activation` string→fn (`gating.py:85-93`) | `cfg.gating: Option<candle_nn::Activation>` (`:35`) |
| `@torch_compile_lazy` / `no_compile()` (`gating.py:13,70-72`) | none — candle eager; `torch.compile` gating dropped (PYTHON_VS_RUST.md "Reference (CUDA-gated) → Rust (kernel-free)") |

**Deliberate divergences** (per PYTHON_VS_RUST.md): device-agnostic, eager candle ops instead of CUDA-gated `torch.compile`/fused kernels; the whole codec (this FFN included) is moshi-crate reuse rather than an in-tree port, chosen because the moshi checkpoint's `quantizer.rvq_first/rvq_rest` weight names load natively (PYTHON_VS_RUST.md:148-149, PORT_STATUS.md:93-97). None of these change FFN numerics on the codec path.

## Precision / gotchas
- **The codec never gates.** `loaders.py:74` sets `gating="none"` for `_transformer_kwargs`; only `_lm_kwargs` (`gating="silu"`, `depformer_gating="silu"`, `:96,107`) — the off-path Moshi 7B LM — reaches `ActivationGating`. So on the LFM2-Audio inference path the active FFN is the dense `linear1/act/linear2` branch, and in Rust the active arm is `Mlp::NoGating`, whose forward **hardcodes `gelu_erf()`** (`transformer.rs:565`) — the codec's non-gated activation is GELU-erf, fixed in Rust rather than read from config.
- **Latent hidden-dim divergence in the gated arm.** Python uses `(21*dim)//8` for the `dim_feedforward == 4*dim` case (`gating.py:56`); the moshi crate uses `11*d_model/4` (`transformer.rs:550`). These differ: `21/8 = 2.625` vs `11/4 = 2.75` (e.g. `dim=4096` → 10752 vs 11264). This would mismatch weights/shapes **if** the gated branch ran, but it is dead on the LFM2-Audio path (codec is `NoGating`), so it never affects inference here. Flag it before reusing the moshi crate for any *gated* Kyutai transformer.
- **Gate split is a contiguous half-split, not interleaved.** `view(...,2,-1)` then index `[...,0,:]` / `[...,1,:]` (`gating.py:19-20`) takes the first `hidden` columns as the gate and the next `hidden` as the value. Do not confuse with RoPE's interleaving — there is no permutation here.
- **No f32 upcast in the gate.** Unlike RMSNorm/softmax, the `activation(gate)*value` multiply stays in model compute dtype (bf16 on Metal/cuda, f32 on Rust CPU per the cross-library f32 floor). Any precision difference vs Python bf16-cuda for the codec comes from the surrounding LayerNorm/attention and the CPU-f32 floor, not this op.
- **Biasless linears** (`gating.py:60-61`), consistent with Kyutai's MHA-style convention noted at `gating.py:63`; the `make_gating` param assert (`:110-114`) is a budget guard and will not catch a wrong-shape checkpoint.
- **No EOAudio / special-token logic and no streaming state here** — this is a stateless pointwise FFN; turn/EOAudio handling and conv/transformer streaming state live in `moshi_compression` / `moshi_streaming` / `core_processor`.
