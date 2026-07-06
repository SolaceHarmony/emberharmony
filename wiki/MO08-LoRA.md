<!-- topic: Mimi Codec — Modules -->
# MO08 · LoRALinear (off-path)
**Code:** `MO08` · **Source:** `moshi/modules/lora.py` · **Rust:** `-` · **On the LFM2-Audio inference path:** no

## Role
`lora.py` is Kyutai's vendored Low-Rank Adaptation (LoRA, arXiv:2106.09685) layer for the **Moshi 7B multi-stream LM** finetune path — `LoRALinear` plus two whole-network surgery helpers (`replace_all_linear_with_lora`, `replace_lora_with_linear`). It exists so a frozen base `nn.Linear` can carry a trainable low-rank residual `scaling·B·A` without touching the original weight, and so those residuals can be fused back into a plain `Linear` at load time for zero-overhead inference. It is **entirely off the LFM2-Audio path**: LFM2.5-Audio uses its own HF `Lfm2Model` backbone (`model_lfm2_backbone`) and its own depthformer (`model_transformer`), neither of which contains a `LoRALinear`; this module is reached only via the Moshi `loaders.get_lora_moshi` → `transformer.StreamingMultiheadAttention` reference stack. It has no Rust port (the `liquid-audio-rs` `core` parity scope deliberately excludes vendored `moshi/`).

## How it works
**`LoRALinear` structure** (`lora.py:44-97`). Three sub-`nn.Linear`, all `bias=False` (an `assert not bias` at `lora.py:71` hard-forbids bias):
- `lora_A: Linear(in_features → rank)` — the down-projection ("down_weight", `lora.py:76-82`).
- `lora_B: Linear(rank → out_features)` — the up-projection ("up_weight", `lora.py:83-89`), conventionally zero-initialized in LoRA so the adapter is a no-op at step 0 (init is left to the network level — see the docstring note "Freezing is handled at the network level").
- `frozen_W: Linear(in_features → out_features)` — the base weight, held as a full child module (`lora.py:91-95`).

Default `dtype=torch.bfloat16` (`lora.py:65`); `scaling: float` is stored verbatim (`lora.py:74`), not divided by rank (this is the "scaling-as-given" LoRA variant, not `alpha/r`).

**Forward** (`lora.py:116-118`). Two independent matmuls summed:
```
lora = lora_B(lora_A(x))          # (·,in)→(·,rank)→(·,out)
return frozen_W(x) + lora * scaling
```
So `y = x·Wᵀ + scaling·(x·Aᵀ)·Bᵀ`. The base path and the low-rank path run as **separate GEMMs** at runtime (not merged) — the rank-`r` bottleneck makes the second path cheap (`2·in·r + 2·r·out` flops vs the dense `2·in·out`). No activation, no norm, no dropout — a pure affine residual.

**`merge_weight`** (`lora.py:99-107`), under `torch.no_grad()`: computes the effective dense weight `W' = (B·A)·scaling + frozen_W.weight`. Note the multiply order — `up_weight.mm(down_weight)` is `B(out,rank)·A(rank,in) → (out,in)`, matching `frozen_W.weight`'s `(out,in)` layout, then `*scaling`, then `+= frozen_W.weight` in place on the product. This returns a tensor; it does **not** mutate the module.

**`_load_hook`** (`lora.py:109-114`, registered at `lora.py:97` via `_register_load_state_dict_pre_hook(..., with_module=True)`). A checkpoint key surgery run *before* `load_state_dict`: any `<prefix>weight` in the incoming `state_dict` is popped and re-keyed to `<prefix>frozen_W.weight`. This lets a **base checkpoint that was saved as a plain `Linear`** (key `…weight`) load straight into a `LoRALinear` (whose base lives under `frozen_W.weight`), while the separately-shipped LoRA safetensors supplies `lora_A.weight`/`lora_B.weight` untouched.

**`replace_all_linear_with_lora`** (`lora.py:5-22`). Recursive module walk (`named_children`): every `nn.Linear` child is swapped for a fresh `LoRALinear(in,out,rank,scaling,…)`, then the original `Linear` is re-attached as `lora.frozen_W = child` (`lora.py:19`) — i.e. the freshly-constructed `frozen_W` is *thrown away* and replaced by the real pretrained `Linear`, preserving its loaded weights. Device/dtype default to the child's own (`lora.py:9-16`) unless overridden. Non-`Linear` modules recurse. Caller-side this is exactly what `loaders.get_lora_moshi` does (`loaders.py:468`), followed by a `strict=False, assign=True` load of the LoRA safetensors (`loaders.py:471-476`) so only `lora_A/lora_B` keys land and base weights are untouched.

**`replace_lora_with_linear`** (`lora.py:25-41`) — the fuse path for inference. Every `LoRALinear` child is collapsed to a plain `nn.Linear(bias=False)` whose weight is `frozen_W.weight.data + scaling·(lora_B.weight @ lora_A.weight)` (`lora.py:30-31`). The new `Linear` is built on `device='meta'` (`lora.py:35`, no allocation) and the merged tensor is assigned as its `Parameter` (`lora.py:37-38`), inheriting `requires_grad` from the merged result. After this pass the network is byte-for-byte a normal transformer with no LoRA runtime cost — gated in the loader by `fuse_lora` (`loaders.py:482-483`, default `True`; CLI `--no_fuse_lora` flips it, `server.py:195`).

**Interaction with packed `in_proj`** (`transformer.py:409-433`). `StreamingMultiheadAttention` packs Q/K/V (and multi-step weights) into one projection then splits via a `_load_hook` that knows the LoRA key names (`in_proj.lora_A.weight`→`in_projs.{i}.lora_A.weight`, etc.), so LoRA composes with the per-step `in_projs`/`out_projs` `ModuleList`. `_init_streaming_state` (`transformer.py:437-439`) special-cases `isinstance(in_proj, LoRALinear)` to pull device/dtype off `lora_A.weight`. None of this is on the LFM2-Audio path.

## Dtypes & shapes
LoRA preserves the wrapped `Linear`'s contract; nothing here promotes to f32 (no norm/softmax). All three sub-Linears default to **bf16** (`lora.py:65`).

| Op | Input dtype+shape | Output dtype+shape |
|---|---|---|
| `LoRALinear.forward(x)` | model dtype `(B, T, in_features)` (Moshi: bf16) | model dtype `(B, T, out_features)` |
| `lora_A(x)` | `(·, in_features)` | `(·, rank)`, rank≈128 (loader default `lora_rank=128`, `loaders.py:370`) |
| `lora_B(·)` | `(·, rank)` | `(·, out_features)` |
| `merge_weight()` | `B(out,rank)`, `A(rank,in)` | `W'(out_features, in_features)`, model dtype |
| `replace_lora_with_linear` weight | `frozen_W.weight(out,in)` + `scaling·(B@A)(out,in)` | `(out_features, in_features)`, `merged_weight.dtype` |
| `_load_hook` | `state_dict[prefix+"weight"]` | re-keyed `state_dict[prefix+"frozen_W.weight"]` (no cast) |

Internal promotions: **none** — pure GEMM + add in the loaded dtype (bf16 by default). The `*scaling` is a scalar broadcast in the same dtype, so at bf16 the scaled-residual add rounds at bf16 (see gotchas).

## Wiring
This module is isolated from the LFM2-Audio tensor graph; its neighbors are all in the Moshi reference stack.

**Upstream (who builds/feeds it):**
- [moshi_loaders](MM02-Mimi-Loaders) — `get_lora_moshi` (`loaders.py:456-483`) calls `replace_all_linear_with_lora` on a freshly built Moshi `LMModel`, loads the LoRA safetensors (bf16 `lora_A/lora_B` weights, `(rank,in)`/`(out,rank)`), then optionally `replace_lora_with_linear` to fuse. This is the only construction site.
- [moshi_transformer](MO03-Codec-Transformer) — `StreamingMultiheadAttention` (`transformer.py:25` import; `:414-418`, `:437-439`) holds `LoRALinear` instances as its `in_projs`/`out_projs` entries; activations flowing in are model-dtype `(B,T,embed_dim)`.

**Downstream (who consumes its output):**
- [moshi_transformer](MO03-Codec-Transformer) — the `LoRALinear.forward` output (model-dtype `(B,T,out_features)`) feeds the rest of the attention/FFN block exactly as a `Linear` output would. After `fuse_lora`, the consumer instead sees a plain `nn.Linear` (the `LoRALinear` no longer exists in the graph).
- [moshi_lm](MM03-Moshi-LM) — transitively, the Moshi `LMModel`/`LMGen` step runs the LoRA-adapted (or fused) attention. The LFM2-Audio top model (`model_lfm2_audio`) never reaches here.

## Python ↔ Rust
No Rust symbol. `liquid-audio-rs` reuses Kyutai's `moshi` crate for the Mimi codec only and excludes the Moshi LM/finetune surface from its `core` parity scope (PYTHON_VS_RUST.md §4: "vendored `liquid_audio/moshi/**` is reused as the `moshi` crate … `compare_symbols`'s `core` scope excludes it by design"). PORT_STATUS.md confirms `moshi/*` is "♻ reuse the moshi crate" rather than re-ported. Mapping:

| Python | Rust | Note |
|---|---|---|
| `LoRALinear` | — | off-path; not in `liquid-audio-rs/src/` (grep: 0 hits) |
| `replace_all_linear_with_lora` / `replace_lora_with_linear` | — | finetune-time surgery; no inference relevance to LFM2-Audio |
| `merge_weight` / `_load_hook` | — | checkpoint plumbing for the Moshi LM only |

The deliberate divergence is the **scope cut**, not an op substitution: LFM2-Audio ships its weights already-merged into its own backbone, so even conceptually there is no LoRA to fuse on the audio path.

## Precision / gotchas
- **Off-path — do not place on the LFM2-Audio graph.** Every callsite is in `moshi/` (Moshi 7B LM finetune). Importing it for LFM2-Audio is a category error.
- **`scaling` is raw, not `alpha/rank`.** `merge_weight` and `forward` use `scaling` directly (`lora.py:104`, `:118`); loader default `lora_scaling=2.0` (`loaders.py:371`). Re-deriving an `alpha/r` would change the math.
- **Two GEMMs at inference unless fused.** Unfused `LoRALinear.forward` runs base + low-rank as separate matmuls every step. `fuse_lora=True` (`loaders.py:482`) collapses them to one `Linear`; `--no_fuse_lora` keeps the runtime overhead.
- **bf16 residual rounding.** With the bf16 default (`lora.py:65`), `frozen_W(x) + lora*scaling` (forward) and `frozen_W.weight + scaling·(B@A)` (fuse) both round the residual add at bf16. The two are not bit-identical: forward adds in activation space per-token, fuse adds in weight space once — a (small) numerical divergence between the fused and unfused models, on top of the cross-library GEMM floor PYTHON_VS_RUST.md §1.4 describes for the on-path code.
- **Multiply order in `merge_weight`.** `up.mm(down)` gives `(out,in)` to match `frozen_W.weight`; reversing the factors would transpose-mismatch. The `+= frozen_W.weight` mutates the scaled product in place (harmless under `no_grad`).
- **`_load_hook` key rewrite is load-order-sensitive.** It re-keys a bare `…weight` to `…frozen_W.weight` *before* the LoRA safetensors loads with `strict=False, assign=True` (`loaders.py:476`); the base checkpoint must therefore carry `…weight` (plain-Linear layout), and the LoRA file must carry only `lora_A/lora_B` keys (any stray base key would be silently relocated). LoRA + quantization is explicitly rejected (`loaders.py:398-401`).
- **`bias=False` only.** The `assert not bias` (`lora.py:71`) means this layer cannot adapt biased projections; the Moshi projections it wraps are all bias-free.
